//! Pure logic backing `IDirect3DQuery9::OCCLUSION` queries.
//!
//! Metal exposes per-fragment visibility counting via
//! `MTLRenderPassDescriptor.visibilityResultBuffer` + per-encoder
//! `setVisibilityResultMode:offset:` state setters. D3D9 bracketing
//! (`Issue(BEGIN)` / `Issue(END)`) can straddle render-pass boundaries
//! and command-buffer submits, so each BEGIN→END span is summed across
//! a set of u64 slots at GPU completion.
//!
//! A slot array belongs to one submit, so a span that outlives the submit
//! is cut into segments: the frame boundary closes the open span at the
//! allocator's high-water mark, queues that segment, and reopens the span
//! in the continuation frame. Each segment is summed against its own
//! frame's buffer and folded into a running total the closing segment
//! publishes. A span that could not be counted at any point (the slot
//! budget ran out, or the buffer it counted into is gone) answers with the
//! permissive `u32::MAX` rather than a count that reads as full occlusion,
//! unless no draw was issued inside it at all: zero is then the exact
//! answer rather than a guess.
//!
//! Split so this module holds only logic that needs no Metal handles:
//! the slot allocator, the per-query state machine, the sum-across-span
//! function, and the retired-buffer pool. d3d9.dll owns the COM wrapper
//! and the encoder-side wiring; mtld3d-unix owns the Metal descriptor
//! binding and command dispatch.

use core::sync::atomic::Ordering;
use std::{collections::VecDeque, sync::Arc};

use mtld3d_shared::{
    MetalHandle,
    encoder_wire::{LeaseCompletion, LeaseCompletionPtr},
    mtl_handle::MTLBufferKind,
    query_mailbox::QueryMailbox,
};

use crate::page_box::PageBox;

/// 1024 u64 slots = 8 KiB.
///
/// Covers every realistic D3D9 occlusion workload (heavy-occlusion
/// titles issue ≤64 queries/frame; with a typical 4–6 passes per frame
/// the worst-case slot use is N*(2+P) ≈ 450 for 64 queries).
pub const MAX_SLOTS: u32 = 1024;

/// Byte size of a visibility result slot — Metal writes a u64 counter per slot.
pub const SLOT_BYTES: u32 = 8;

/// Where a query sits in its lifecycle (begin → end → issued).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryStatus {
    /// Created but never `Issue`'d.
    ///
    /// `GetData` returns the permissive "fully visible" count (`u32::MAX`)
    /// `+ S_OK`, so an un-issued query never makes a title cull geometry.
    NeverIssued,
    /// `Issue(BEGIN)` / `Issue(END)` fired, sum not yet available.
    ///
    /// `GetData` returns `S_FALSE` until `coherent_seq` catches up.
    Pending,
    /// GPU completed and the sum was folded into `accumulated`.
    Issued,
}

/// What an `Issue(D3DISSUE_END)` without `D3DISSUE_BEGIN` does to an occlusion query.
///
/// Only an open span has anything to close. An END on a query that was never
/// begun opens and closes an empty span, so the query answers 0, since no
/// draw is inside it. An END on a span already closed changes nothing, so
/// the query keeps the result of the span the first END closed.
#[derive(Debug, PartialEq, Eq)]
pub enum EndIssue {
    /// A span is open: the END closes it.
    Close,
    /// The query was never begun: the END opens a span and closes it at once.
    OpenAndClose,
    /// The last span is already closed: the END is ignored.
    Keep,
}

/// Shared counted object behind a `Direct3DQuery9`.
///
/// Held by `Arc` so the encoder-side pending list can keep the core alive
/// past the COM wrapper's refcount reaching zero.
pub struct VisibilityQueryCore {
    mailbox: QueryStorage,
}

/// A query owns its mailbox locally or borrows it under a guest allocation lease.
enum QueryStorage {
    Owned(QueryMailbox),
    Guest {
        pointer: u64,
        completion: LeaseCompletionPtr,
    },
}

impl VisibilityQueryCore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            mailbox: QueryStorage::Owned(QueryMailbox::new()),
        })
    }

    /// Borrow one query mailbox pinned by its guest lease.
    ///
    /// # Safety
    ///
    /// `pointer` names a live, aligned mailbox. The guest retains both it and
    /// `completion` until the final native core publishes completion from Drop.
    #[must_use]
    pub const unsafe fn from_guest(pointer: u64, completion: LeaseCompletionPtr) -> Self {
        Self {
            mailbox: QueryStorage::Guest {
                pointer,
                completion,
            },
        }
    }

    /// Address passed with a query lease, never ownership of its Rust wrapper.
    #[must_use]
    pub fn mailbox_address(&self) -> u64 {
        core::ptr::from_ref(self.mailbox()) as u64
    }

    const fn mailbox(&self) -> &QueryMailbox {
        match &self.mailbox {
            QueryStorage::Owned(mailbox) => mailbox,
            QueryStorage::Guest { pointer, .. } => {
                // SAFETY: from_guest pins the aligned mailbox through the final native core drop.
                unsafe { &*(*pointer as *const QueryMailbox) }
            }
        }
    }

    pub fn status(&self) -> QueryStatus {
        let requested = self.mailbox().requested_generation.load(Ordering::Acquire);
        let status = self.mailbox().status.load(Ordering::Acquire);
        if status == QueryStatus::Issued as u32
            && self.mailbox().result_generation.load(Ordering::Acquire) == requested
        {
            QueryStatus::Issued
        } else if requested == 0 && status == QueryStatus::NeverIssued as u32 {
            QueryStatus::NeverIssued
        } else {
            QueryStatus::Pending
        }
    }

    /// Called by the encoder thread on the BEGIN closure.
    ///
    /// Captures the current frame's submit seq and the first slot the
    /// Metal encoder will write to. Moves the query from
    /// `NeverIssued`/`Issued` back into `Pending` — a second Issue on the
    /// same wrapper reuses the core.
    ///
    /// # Panics
    ///
    /// Panics after `u64::MAX` brackets rather than recycling an issue
    /// generation that a pending segment could still carry.
    pub fn begin(
        &self,
        seq: u64,
        offset: u32,
        logical: (u32, u32),
        render: (u32, u32),
        draws_seen: u64,
    ) {
        let generation = self.mark_armed();
        self.begin_recorded(generation, seq, offset, logical, render, draws_seen);
    }

    /// Execute the BEGIN generation captured by the API thread.
    ///
    /// Queued brackets use their recorded identities even if the API has already requested a
    /// later bracket. Only completion of the currently requested generation is visible to
    /// `GetData`.
    pub fn begin_recorded(
        &self,
        generation: u64,
        seq: u64,
        offset: u32,
        logical: (u32, u32),
        render: (u32, u32),
        draws_seen: u64,
    ) {
        self.mailbox()
            .issue_generation
            .store(generation, Ordering::Release);
        self.mailbox().seq_begin.store(seq, Ordering::Release);
        self.mailbox().offset_begin.store(offset, Ordering::Release);
        self.mailbox()
            .logical_area
            .store(area(logical), Ordering::Release);
        self.mailbox()
            .render_area
            .store(area(render), Ordering::Release);
        self.mailbox()
            .draws_at_begin
            .store(draws_seen, Ordering::Release);
        self.mailbox().draws_in_span.store(0, Ordering::Release);
        // Reset the accumulators in case this core was previously issued
        // and the app is re-issuing.
        self.mailbox().carried.store(0, Ordering::Release);
        self.mailbox().uncounted.store(0, Ordering::Release);
        self.mailbox().accumulated.store(0, Ordering::Release);
        self.mailbox()
            .status
            .store(QueryStatus::Pending as u32, Ordering::Release);
    }

    /// Reopen the span in the continuation of a frame that ended mid-span.
    ///
    /// The segment the boundary closed keeps its own slots and its own
    /// frame's buffer; this only points the next segment at the fresh
    /// frame's allocator. The running total and the areas the count is
    /// reported in both carry over.
    pub fn resume(&self, seq: u64, offset: u32) {
        self.mailbox().seq_begin.store(seq, Ordering::Release);
        self.mailbox().offset_begin.store(offset, Ordering::Release);
    }

    /// Called by the encoder thread on the END closure.
    ///
    /// Records the submit seq the closing segment retires at, which is what
    /// `GetData(D3DGETDATA_FLUSH)` waits on, and how many draws the span held.
    pub fn end(&self, seq: u64, draws_seen: u64) {
        self.end_recorded(self.issue_generation(), seq, draws_seen);
    }

    /// Execute END using the API-captured bracket identity.
    pub fn end_recorded(&self, generation: u64, seq: u64, draws_seen: u64) {
        self.mailbox().seq_end.store(seq, Ordering::Release);
        self.mailbox()
            .end_generation
            .store(generation, Ordering::Release);
        self.mailbox().draws_in_span.store(
            draws_seen.wrapping_sub(self.mailbox().draws_at_begin.load(Ordering::Acquire)),
            Ordering::Release,
        );
    }

    /// Record that part of this span was never counted.
    ///
    /// Called when the frame's slot budget is spent or the buffer a segment
    /// counted into is gone. The published result is then `u32::MAX` rather
    /// than the partial sum the slots that did exist add up to.
    pub fn mark_uncounted(&self) {
        self.mailbox().uncounted.store(1, Ordering::Release);
    }

    /// `GetData` result: DWORD visible-pixel count clamped at `u32::MAX`.
    ///
    /// # Panics
    ///
    /// Panics if the clamped accumulator overflows `u32::MAX` — unreachable
    /// because the `.min(u32::MAX)` directly above bounds it.
    pub fn get_u32(&self) -> u32 {
        let v = self.get_u64();
        u32::try_from(v.min(u64::from(u32::MAX))).expect("clamped above to u32::MAX")
    }

    /// Raw 64-bit visible-sample sum (un-clamped).
    ///
    /// `GetData` writes up to 8 bytes from this when the caller's buffer
    /// exceeds the advertised DWORD size, matching the runtime's internal
    /// UINT64 counter.
    pub fn get_u64(&self) -> u64 {
        if self.mailbox().result_generation.load(Ordering::Acquire)
            == self.mailbox().requested_generation.load(Ordering::Acquire)
        {
            self.mailbox().accumulated.load(Ordering::Acquire)
        } else {
            0
        }
    }

    /// Mark the query armed (`Pending`) the instant `Issue(D3DISSUE_BEGIN)` fires.
    ///
    /// Recorded on the API thread, before the encoder-side `begin` closure
    /// runs. The Present-driven encoder may not have drained that closure
    /// yet (a D3D9 app can poll a query without an intervening Present), so
    /// without this a no-Present `GetData` would observe the initial
    /// `NeverIssued` state and short-circuit to the permissive stub instead
    /// of flushing. The `begin` closure still resets the accumulator and
    /// assigns the slot when it eventually runs.
    ///
    /// # Panics
    ///
    /// Panics after `u64::MAX` brackets rather than reusing an outstanding identity.
    pub fn mark_armed(&self) -> u64 {
        let generation = self
            .mailbox()
            .requested_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .expect("visibility query requested generation exhausted")
            + 1;
        self.mailbox().end_requested.store(0, Ordering::Release);
        self.mailbox()
            .status
            .store(QueryStatus::Pending as u32, Ordering::Release);
        generation
    }

    /// Record that `Issue(D3DISSUE_END)` was called on the API thread.
    ///
    /// Set before the encoder-side `end` closure runs. Read by the
    /// blocking `GetData(FLUSH)` to decide whether the query is safe to
    /// flush + read (ended) or must report `S_FALSE` (still open).
    pub fn mark_end_requested(&self) -> u64 {
        self.mailbox().end_requested.store(1, Ordering::Release);
        self.mailbox().requested_generation.load(Ordering::Acquire)
    }

    /// Whether an `Issue(D3DISSUE_END)` has been recorded since the last `Issue(D3DISSUE_BEGIN)`.
    ///
    /// See [`Self::mark_end_requested`].
    pub fn end_requested(&self) -> bool {
        self.mailbox().end_requested.load(Ordering::Acquire) != 0
    }

    /// Whether the API thread has a span open: begun and not ended since.
    #[must_use]
    pub fn span_open(&self) -> bool {
        self.mailbox().requested_generation.load(Ordering::Acquire) != 0 && !self.end_requested()
    }

    /// What an `Issue(D3DISSUE_END)` without `D3DISSUE_BEGIN` does now, see [`EndIssue`].
    #[must_use]
    pub fn end_issue(&self) -> EndIssue {
        if self.mailbox().requested_generation.load(Ordering::Acquire) == 0 {
            EndIssue::OpenAndClose
        } else if self.end_requested() {
            EndIssue::Keep
        } else {
            EndIssue::Close
        }
    }

    /// Slot where Metal started counting the segment currently open.
    ///
    /// Paired with the allocator's next index to make the half-open span
    /// a segment covers.
    pub fn offset_begin(&self) -> u32 {
        self.mailbox().offset_begin.load(Ordering::Acquire)
    }

    /// Submit-seq the encoder will retire this query at, set when the END closure runs.
    ///
    /// `0` means the encoder has not processed the END of the bracket the
    /// API thread last requested.
    pub fn seq_end_loaded(&self) -> u64 {
        if self.mailbox().end_generation.load(Ordering::Acquire)
            == self.mailbox().requested_generation.load(Ordering::Acquire)
        {
            self.mailbox().seq_end.load(Ordering::Acquire)
        } else {
            0
        }
    }

    /// Identity of the bracket whose BEGIN the encoder processed most recently.
    fn issue_generation(&self) -> u64 {
        self.mailbox().issue_generation.load(Ordering::Acquire)
    }

    /// Fold one retired segment's sample count into the span's running total.
    ///
    /// Used only from `VisibilityQueryState::intake_completed` in this
    /// module.
    fn accumulate_segment(&self, summed: u64) {
        let carried = self.mailbox().carried.load(Ordering::Acquire);
        self.mailbox()
            .carried
            .store(carried.saturating_add(summed), Ordering::Release);
    }

    /// Publish the span's total and flip status to `Issued`.
    ///
    /// Used only from `VisibilityQueryState::intake_completed` in this
    /// module.
    fn publish_span(&self) {
        // A span with no draw in it counted nothing, whatever became of its
        // slots, so its sum is exact and the permissive answer would be
        // invented.
        let unknown = self.mailbox().uncounted.load(Ordering::Acquire) != 0
            && self.mailbox().draws_in_span.load(Ordering::Acquire) != 0;
        let result = if unknown {
            u64::from(u32::MAX)
        } else {
            logical_samples(
                self.mailbox().carried.load(Ordering::Acquire),
                self.mailbox().render_area.load(Ordering::Acquire),
                self.mailbox().logical_area.load(Ordering::Acquire),
            )
        };
        self.mailbox().accumulated.store(result, Ordering::Release);
        self.mailbox()
            .result_generation
            .store(self.issue_generation(), Ordering::Release);
        self.mailbox()
            .status
            .store(QueryStatus::Issued as u32, Ordering::Release);
    }
}

impl Drop for VisibilityQueryCore {
    fn drop(&mut self) {
        if let QueryStorage::Guest { completion, .. } = &self.mailbox {
            // SAFETY: from_guest retains the acknowledgment cell until this final publication.
            let completion = unsafe { &*(completion.raw() as *const LeaseCompletion) };
            completion.publish();
        }
    }
}

/// Monotonic u64-slot allocator reset at each frame boundary.
///
/// Encapsulated inside `VisibilityQueryState` — the encoder reaches it
/// via `VisibilityQueryState::bump_slot`.
/// The pixel area of an extent, for the area ratio a count is rescaled by.
const fn area((width, height): (u32, u32)) -> u64 {
    (width as u64) * (height as u64)
}

/// Convert a sample count produced on a `render_area` grid into `logical_area` pixels.
///
/// A target rasterized below the resolution D3D9 reports holds fewer pixels
/// than the game was told, so the counter is multiplied by the ratio of the
/// reported area to the render area, rounded to nearest. Equal areas (the
/// identity, and every target the scale does not reach) and a zero area (a
/// query begun before any target was bound) pass the count through; the
/// result saturates at `u64::MAX`.
#[must_use]
pub fn logical_samples(render_samples: u64, render_area: u64, logical_area: u64) -> u64 {
    if render_area == 0 || logical_area == 0 || render_area == logical_area {
        return render_samples;
    }
    let scaled = (u128::from(render_samples) * u128::from(logical_area)
        + u128::from(render_area) / 2)
        / u128::from(render_area);
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

struct VisibilityOffsetAllocator {
    next: u32,
    exhausted: bool,
}

impl VisibilityOffsetAllocator {
    const fn new() -> Self {
        Self {
            next: 0,
            exhausted: false,
        }
    }

    /// Bump to a fresh slot.
    ///
    /// Returns the newly-allocated index, or `None` if the frame's slot
    /// budget is exhausted.
    const fn bump(&mut self) -> Option<u32> {
        if self.next >= MAX_SLOTS {
            self.exhausted = true;
            return None;
        }
        let slot = self.next;
        self.next += 1;
        Some(slot)
    }

    /// Called at `begin_frame`.
    ///
    /// Resets the counter so the next frame starts at slot 0.
    const fn reset(&mut self) {
        self.next = 0;
        self.exhausted = false;
    }
}

/// Sum u64 visibility counts across a BEGIN→END span.
///
/// `slots` is the shared-storage buffer readable once `coherent_seq` has
/// caught up to the frame containing END. The span is half-open:
/// `[begin, end)`.
fn sum_slots(slots: &[u64], begin: u32, end: u32) -> u64 {
    let begin = begin as usize;
    let end = end as usize;
    let end = end.min(slots.len());
    if begin >= end {
        return 0;
    }
    slots[begin..end].iter().sum()
}

/// A retired visibility buffer awaiting GPU completion.
pub struct RetiredVisibilityBuffer {
    backing: PageBox,
    /// Metal `MTLBuffer*`. Typed via `MetalHandle<MTLBufferKind>`.
    metal_handle: MetalHandle<MTLBufferKind>,
    /// `submit_seq` of the frame that used this buffer.
    ///
    /// Pool recycles it only once `coherent_seq >= release_seq`.
    release_seq: u64,
}

impl RetiredVisibilityBuffer {
    /// Construct a retired buffer.
    ///
    /// `release_seq = 0` is valid and means "ready immediately" — used
    /// for the freshly-allocated buffer installed on first BEGIN (its
    /// real `release_seq` is stamped later, at submit time, by
    /// `retire_current_buffer`).
    #[must_use]
    pub const fn new(
        backing: PageBox,
        metal_handle: MetalHandle<MTLBufferKind>,
        release_seq: u64,
    ) -> Self {
        Self {
            backing,
            metal_handle,
            release_seq,
        }
    }

    /// Mutable backing so the encoder can zero the region on reuse.
    pub const fn backing_mut(&mut self) -> &mut PageBox {
        &mut self.backing
    }

    const fn metal_handle(&self) -> MetalHandle<MTLBufferKind> {
        self.metal_handle
    }

    const fn backing(&self) -> &PageBox {
        &self.backing
    }

    #[must_use]
    pub const fn release_seq(&self) -> u64 {
        self.release_seq
    }

    /// Consume and hand out the three lifecycle pieces.
    ///
    /// The caller can then route an evicted entry through the encoder's
    /// seq-gated `PendingBufferWrapperRetention` drain: destroy the
    /// `MTLBuffer` wrapper first, then drop the `PageBox` only once GPU
    /// work on `release_seq` has retired. Dropping the
    /// `RetiredVisibilityBuffer` directly would free the PE backing while
    /// Metal still holds a `bytesNoCopy` pointer into it, leaving the GPU
    /// writing into freed heap.
    #[must_use]
    pub fn into_parts(self) -> (PageBox, MetalHandle<MTLBufferKind>, u64) {
        (self.backing, self.metal_handle, self.release_seq)
    }
}

/// Pool of retired-but-not-yet-reusable visibility buffers, plus a free list.
///
/// The free list holds the buffers that are safe to hand back to a new
/// frame. Bounded so a title that somehow retires buffers faster than the
/// GPU catches up doesn't leak unbounded memory.
struct VisibilityBufferPool {
    /// Waiting for `coherent_seq >= release_seq` before they can be handed back out.
    retired: Vec<RetiredVisibilityBuffer>,
    /// Ready to reuse.
    free: Vec<RetiredVisibilityBuffer>,
    /// Maximum total buffers the pool will hold onto.
    ///
    /// Extra retirees past the cap are evicted to the drop site.
    cap: usize,
}

impl VisibilityBufferPool {
    const fn new(cap: usize) -> Self {
        Self {
            retired: Vec::new(),
            free: Vec::new(),
            cap,
        }
    }

    fn acquire(&mut self) -> Option<RetiredVisibilityBuffer> {
        self.free.pop()
    }

    fn retire(&mut self, buf: RetiredVisibilityBuffer) -> Option<RetiredVisibilityBuffer> {
        self.retired.push(buf);
        let total = self.retired.len() + self.free.len();
        if total > self.cap {
            // Prefer evicting free first; fall back to the oldest
            // retired.
            self.free.pop().or_else(|| {
                if self.retired.is_empty() {
                    None
                } else {
                    Some(self.retired.remove(0))
                }
            })
        } else {
            None
        }
    }

    fn release_up_to(&mut self, coherent_seq: u64) {
        let mut i = 0;
        while i < self.retired.len() {
            if self.retired[i].release_seq <= coherent_seq {
                self.free.push(self.retired.swap_remove(i));
            } else {
                i += 1;
            }
        }
    }

    /// Read back the slot array of the retired buffer tagged `seq`.
    ///
    /// SAFETY: caller guarantees the buffer's GPU work has completed
    /// (`coherent_seq` >= `release_seq`), so CPU reads of the Shared
    /// backing observe the final counter writes. Backing is allocated
    /// via `PageBox::new_uninit_with_align(SLOT_BYTES * MAX_SLOTS)` which
    /// returns an 8-aligned page-mapped region — but clippy can't see
    /// that, so we use `read_unaligned` per slot to make alignment a
    /// non-question at the cast.
    fn retired_backing_for(&self, seq: u64) -> Option<Vec<u64>> {
        let buf = self
            .retired
            .iter()
            .find(|b| b.release_seq == seq)?
            .backing();
        let ptr = buf.as_ptr();
        let slots: Vec<u64> = (0..MAX_SLOTS as usize)
            .map(|i| {
                // SAFETY: `ptr + i * 8` stays within the visibility buffer
                // (size = MAX_SLOTS * 8 bytes).
                let byte_ptr = unsafe { ptr.add(i * 8) };
                // SAFETY: read_unaligned is the correct primitive for
                // byte-addressed access to a `u64` slot.
                unsafe { core::ptr::read_unaligned(byte_ptr.cast::<u64>()) }
            })
            .collect();
        Some(slots)
    }
}

/// One segment of a span, awaiting the GPU retiring the frame it counted in.
///
/// Held keyed by that frame's `submit_seq`; the encoder thread folds the
/// segment's slots into the core once `coherent_seq >= submit_seq`. The
/// half-open slot range travels with the segment rather than being read
/// off the core, because a span reopened in a continuation frame has
/// already moved the core's own `offset_begin` on to the next segment.
struct PendingSegment {
    submit_seq: u64,
    core: Arc<VisibilityQueryCore>,
    issue_generation: u64,
    span: (u32, u32),
    /// Whether this segment carries the query's `Issue(END)`.
    ///
    /// The closing segment publishes the total; every earlier one only
    /// adds to it.
    closes_span: bool,
}

/// Composite encoder-side state for visibility queries.
///
/// Lives on `FrameEncoder`: persistent across frames (pool, pending list)
/// with per-frame pieces (allocator, `active_count`, current buffer) reset in
/// `begin_frame`.
pub struct VisibilityQueryState {
    allocator: VisibilityOffsetAllocator,
    pool: VisibilityBufferPool,
    /// Segments whose slots were emitted on some frame.
    ///
    /// Folded in once `coherent_seq` catches up to their `submit_seq`.
    pending: VecDeque<PendingSegment>,
    /// Queries currently between BEGIN and END on the encoder thread.
    ///
    /// Pushed on BEGIN, removed on END, and carried across frame
    /// boundaries so a span that outlives its submit is closed and
    /// reopened rather than lost. Decides whether a freshly opened pass
    /// needs a Counting-mode arm and whether END emits Disabled or a
    /// fresh Counting slot.
    active: Vec<Arc<VisibilityQueryCore>>,
    /// Visibility buffer reserved for the frame currently being encoded.
    ///
    /// `None` until the first BEGIN in a frame allocates. At submit time
    /// the encoder moves this into the pool keyed by the frame's
    /// `submit_seq`.
    current_buffer: Option<RetiredVisibilityBuffer>,
    /// `true` once any BEGIN this frame hit allocator exhaustion.
    ///
    /// The encoder uses this to short-circuit subsequent commands and to
    /// finalize overflowing queries with the safe `u32::MAX` fallback.
    exhausted_this_frame: bool,
    /// Draws the encoder has issued, counted from the first frame.
    ///
    /// Monotonic across frames on purpose: a span is bracketed by two reads
    /// of it, and a span cut by a submit boundary spans two frames.
    draws_seen: u64,
}

impl VisibilityQueryState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allocator: VisibilityOffsetAllocator::new(),
            // Cap at 16: sync_channel(1) + GPU in-flight gives at most
            // ~3 frames coexisting in the steady state, but a
            // `drawable_wait` hitch or a block on the encoder channel
            // can stack several more retirees before `intake_completed`
            // drains them. At 8 KiB per buffer the 16-entry ceiling
            // is 128 KiB — cheap — and eviction becomes a genuinely
            // exceptional path rather than a predictable one on hitch
            // frames. Evictions are still routed through the encoder's
            // `PendingBufferWrapperRetention` drain so the MTLBuffer
            // wrapper is destroyed before the `PageBox` drops.
            pool: VisibilityBufferPool::new(16),
            pending: VecDeque::new(),
            active: Vec::new(),
            current_buffer: None,
            exhausted_this_frame: false,
            draws_seen: 0,
        }
    }

    /// Reset per-frame fields.
    ///
    /// Called on the encoder thread at `begin_frame` *after* the current
    /// frame's buffer (if any) has been moved into retention via
    /// `retire_current_buffer`. Does not touch the pool, the pending list
    /// or the open spans; those drain separately, and a span open across
    /// the boundary is reopened by [`Self::resume_open_spans`].
    pub const fn reset_frame(&mut self) {
        self.allocator.reset();
        self.exhausted_this_frame = false;
    }

    #[must_use]
    pub const fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Count one draw against every span open now and every one opened later.
    ///
    /// Called from the encoder's draw-site pass entry. What it buys is the
    /// distinction between a span that lost its count and one that never had
    /// anything to count.
    pub const fn note_draw(&mut self) {
        self.draws_seen = self.draws_seen.wrapping_add(1);
    }

    #[must_use]
    pub const fn draws_seen(&self) -> u64 {
        self.draws_seen
    }

    #[must_use]
    pub fn current_buffer_handle(&self) -> MetalHandle<MTLBufferKind> {
        self.current_buffer
            .as_ref()
            .map_or(MetalHandle::NULL, RetiredVisibilityBuffer::metal_handle)
    }

    /// Install a buffer as this frame's visibility buffer.
    ///
    /// Called on first Issue(BEGIN). Takes either a pool-acquired buffer
    /// (via `try_acquire_reusable`) or a freshly allocated one; the caller
    /// has zero-initialized the backing before handing it in.
    pub fn install_current_buffer(&mut self, buf: RetiredVisibilityBuffer) {
        debug_assert!(self.current_buffer.is_none());
        self.current_buffer = Some(buf);
    }

    /// Pool-backed acquire of a reusable visibility buffer.
    ///
    /// Its backing still holds the last frame's counter values (caller
    /// must zero).
    pub fn try_acquire_reusable(&mut self) -> Option<RetiredVisibilityBuffer> {
        self.pool.acquire()
    }

    /// Move this frame's buffer into the pool's retired list.
    ///
    /// Tagged with `submit_seq` so `release_up_to` can free it for reuse
    /// once the GPU retires the frame. Called from the encoder's submit
    /// path.
    ///
    /// Returns the evicted entry (if any) when the pool exceeds its
    /// cap. The caller **must** route it through the encoder's
    /// seq-gated `PendingBufferWrapperRetention` drain — destroying
    /// the `MTLBuffer` wrapper before the `PageBox` drops — otherwise
    /// Metal's `bytesNoCopy` pointer outlives the PE allocation and
    /// the GPU will write into freed snmalloc heap, corrupting
    /// allocator metadata.
    pub fn retire_current_buffer(&mut self, submit_seq: u64) -> Option<RetiredVisibilityBuffer> {
        let buf = self.current_buffer.take()?;
        let (backing, handle, _) = buf.into_parts();
        let retired = RetiredVisibilityBuffer::new(backing, handle, submit_seq);
        self.pool.retire(retired)
    }

    #[must_use]
    pub const fn exhausted_this_frame(&self) -> bool {
        self.exhausted_this_frame
    }

    /// Record that the frame's slot budget is spent.
    ///
    /// Every span open at that point loses the rest of its count, so each
    /// one is marked uncounted: the alternative is publishing a partial
    /// sum, which reads as occlusion the scene does not have.
    pub fn mark_exhausted(&mut self) {
        self.exhausted_this_frame = true;
        for core in &self.active {
            core.mark_uncounted();
        }
    }

    /// Encoder-side slot allocation.
    ///
    /// Wraps the private allocator so the encoder never touches it
    /// directly.
    pub const fn bump_slot(&mut self) -> Option<u32> {
        self.allocator.bump()
    }

    /// The slot the next allocation would hand out: the frame's high-water mark.
    ///
    /// Where a span begun without a slot starts. Every slot below it was
    /// reserved before the span began, and once the frame is exhausted no slot
    /// is reserved after it, so a segment cut from there covers nothing.
    #[must_use]
    pub const fn next_slot(&self) -> u32 {
        self.allocator.next
    }

    /// Record a query as open between BEGIN and END.
    ///
    /// A second BEGIN on a query that is already open leaves one entry:
    /// D3D9 restarts the span there, and the core's own `begin` has
    /// already done that.
    pub fn push_active(&mut self, core: &Arc<VisibilityQueryCore>) {
        if self.active.iter().any(|c| Arc::ptr_eq(c, core)) {
            return;
        }
        self.active.push(core.clone());
    }

    /// Drop a query from the open set at its END.
    pub fn remove_active(&mut self, core: &Arc<VisibilityQueryCore>) {
        self.active.retain(|c| !Arc::ptr_eq(c, core));
    }

    /// Queue one segment of a span for the intake that follows its frame.
    ///
    /// `span` is the half-open slot range the segment counted into on the
    /// frame `submit_seq` names. `closes_span` marks the segment carrying
    /// `Issue(END)`, the one that publishes the total.
    pub fn push_pending(
        &mut self,
        submit_seq: u64,
        core: Arc<VisibilityQueryCore>,
        span: (u32, u32),
        closes_span: bool,
    ) {
        let issue_generation = core.issue_generation();
        self.pending.push_back(PendingSegment {
            submit_seq,
            core,
            issue_generation,
            span,
            closes_span,
        });
    }

    /// Cut every open span at a submit boundary.
    ///
    /// The frame's slot array retires with the submit, so each open span
    /// contributes the segment it has counted so far, ending at the
    /// allocator's high-water mark. Called from the encoder's submit path,
    /// after the last pass of the frame is closed.
    pub fn split_open_spans(&mut self, submit_seq: u64) {
        let end = self.allocator.next;
        for core in &self.active {
            self.pending.push_back(PendingSegment {
                submit_seq,
                core: core.clone(),
                issue_generation: core.issue_generation(),
                span: (core.offset_begin(), end),
                closes_span: false,
            });
        }
    }

    /// Reopen every span the previous submit cut, in the frame that continues it.
    ///
    /// Called at `begin_frame`, after [`Self::reset_frame`], so the spans
    /// point at the fresh allocator. The pass the next draw opens arms
    /// itself from the open set, so no slot is reserved here.
    pub fn resume_open_spans(&mut self, submit_seq: u64) {
        let offset = self.allocator.next;
        for core in &self.active {
            core.resume(submit_seq, offset);
        }
    }

    /// Drain every owned `RetiredVisibilityBuffer`, leaving the queries alone.
    ///
    /// Covers the current-frame slot plus the pool's retired and free lists.
    /// Caller takes ownership of the returned vec; each entry's `into_parts`
    /// yields the (`PageBox`, `metal_handle`, `release_seq`) triple the encoder
    /// feeds through its destroy-then-drop ordering.
    ///
    /// The caller finalizes first, so every segment whose frame the GPU has
    /// retired is summed while the buffer it counted into is still here and
    /// the pending list is empty by the time this runs. A span still open is
    /// not finished at all: the submit that precedes the drain cut it at its
    /// own boundary, and the frame that continues it reopens it through
    /// [`Self::resume_open_spans`], so the open set survives. A segment that
    /// somehow outlives the finalize keeps its place too and answers
    /// permissively at the next intake, which cannot find the buffer it names:
    /// dropping it would leave its query `Pending` for the rest of the process.
    pub fn drain_all_buffers(&mut self) -> Vec<RetiredVisibilityBuffer> {
        let mut all = Vec::new();
        if let Some(cur) = self.current_buffer.take() {
            all.push(cur);
        }
        all.append(&mut self.pool.retired);
        all.append(&mut self.pool.free);
        all
    }

    /// Drain pending → finalize → release pool entries up to `coherent_seq`.
    ///
    /// Every queued segment whose frame has retired on the GPU is summed from
    /// the retired visibility buffer matching its `submit_seq` and folded into
    /// its query's running total; the segment carrying `Issue(END)` publishes
    /// that total. An empty span sums to zero without consulting a buffer,
    /// since a frame that reserved no slot for the query reserved no buffer
    /// either. A non-empty span whose buffer cannot be found (retired and
    /// evicted before intake ran, which the normal flow never produces)
    /// leaves the whole span uncounted. A segment from a bracket abandoned by
    /// a later BEGIN is skipped instead of being folded into that replacement
    /// bracket. Its retired buffer still moves through the normal release path
    /// once its seq has been reached, so abandoning a bracket changes no
    /// storage lifetime.
    ///
    /// Each retired buffer's `PageBox` points at PE-allocated Shared storage
    /// wrapped by Metal. Once `coherent_seq >= release_seq` the GPU is done
    /// writing, so CPU reads of the backing are coherent; the caller establishes
    /// that by passing the atomic's observed value.
    ///
    /// # Panics
    ///
    /// Panics if a segment is removed from `pending` at an index the loop
    /// above has already bounds-checked, which the `while i < len` guard
    /// makes unreachable.
    pub fn intake_completed(&mut self, coherent_seq: u64) {
        // Fold first (reads retired buffers by seq), then release
        // pool entries. Order matters: release_up_to moves retired →
        // free which clears the seq association.
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].submit_seq > coherent_seq {
                i += 1;
                continue;
            }
            let entry = self.pending.remove(i).expect("bound-checked");
            if entry.issue_generation != entry.core.issue_generation() {
                continue;
            }
            let (begin, end) = entry.span;
            let sum = if begin >= end {
                0
            } else if let Some(slots) = self.pool.retired_backing_for(entry.submit_seq) {
                sum_slots(&slots, begin, end)
            } else {
                entry.core.mark_uncounted();
                0
            };
            entry.core.accumulate_segment(sum);
            if entry.closes_span {
                entry.core.publish_span();
            }
        }
        self.pool.release_up_to(coherent_seq);
    }
}

impl Default for VisibilityQueryState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
