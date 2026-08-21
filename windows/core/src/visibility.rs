//! Pure logic backing `IDirect3DQuery9::OCCLUSION` queries.
//!
//! Metal exposes per-fragment visibility counting via
//! `MTLRenderPassDescriptor.visibilityResultBuffer` + per-encoder
//! `setVisibilityResultMode:offset:` state setters. D3D9 bracketing
//! (`Issue(BEGIN)` / `Issue(END)`) can straddle render-pass boundaries
//! and command-buffer submits, so each BEGIN→END span is summed across
//! a set of u64 slots at GPU completion.
//!
//! Split so this module holds only logic that needs no Metal handles:
//! the slot allocator, the per-query state machine, the sum-across-span
//! function, and the retired-buffer pool. d3d9.dll owns the COM wrapper
//! and the encoder-side wiring; mtld3d-unix owns the Metal descriptor
//! binding and command dispatch.

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::{collections::VecDeque, sync::Arc};

use mtld3d_shared::{MetalHandle, mtl_handle::MTLBufferKind};

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

/// Shared counted object behind a `Direct3DQuery9`.
///
/// Held by `Arc` so the encoder-side pending list can keep the core alive
/// past the COM wrapper's refcount reaching zero.
pub struct VisibilityQueryCore {
    /// Frame submit-seq at `Issue(BEGIN)`.
    ///
    /// Only valid once `status` leaves `NeverIssued`.
    seq_begin: AtomicU64,
    /// Frame submit-seq at `Issue(END)`.
    ///
    /// Valid once `status` reaches `Pending` following an END — v1
    /// requires begin and end in the same frame, so this equals
    /// `seq_begin` after a legal end.
    seq_end: AtomicU64,
    /// First slot index written for this query (Begin-side bump).
    offset_begin: AtomicU32,
    /// Slot index written after this query's End-side bump.
    ///
    /// Slots in `[offset_begin, offset_end)` are summed at readback.
    offset_end: AtomicU32,
    /// u64 running sum; clamped to `u32::MAX` on `get_u32`.
    accumulated: AtomicU64,
    /// `QueryStatus` encoded as u8.
    ///
    /// Atomic so `GetData` can observe transitions without locking.
    status: AtomicU64,
    /// Set the instant `Issue(D3DISSUE_END)` is recorded (API thread).
    ///
    /// Cleared on `Issue(D3DISSUE_BEGIN)`. Lets the blocking
    /// `GetData(FLUSH)` tell an *ended* query (safe to flush + read)
    /// from one still *open* (begun, not ended) — flushing the latter
    /// would split its span across two submits and zero the count.
    end_requested: AtomicBool,
}

impl VisibilityQueryCore {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            seq_begin: AtomicU64::new(0),
            seq_end: AtomicU64::new(0),
            offset_begin: AtomicU32::new(0),
            offset_end: AtomicU32::new(0),
            accumulated: AtomicU64::new(0),
            status: AtomicU64::new(QueryStatus::NeverIssued as u64),
            end_requested: AtomicBool::new(false),
        })
    }

    pub fn status(&self) -> QueryStatus {
        match self.status.load(Ordering::Acquire) {
            x if x == QueryStatus::Pending as u64 => QueryStatus::Pending,
            x if x == QueryStatus::Issued as u64 => QueryStatus::Issued,
            _ => QueryStatus::NeverIssued,
        }
    }

    /// Called by the encoder thread on the BEGIN closure.
    ///
    /// Captures the current frame's submit seq and the first slot the
    /// Metal encoder will write to. Moves the query from
    /// `NeverIssued`/`Issued` back into `Pending` — a second Issue on the
    /// same wrapper reuses the core.
    pub fn begin(&self, seq: u64, offset: u32) {
        self.seq_begin.store(seq, Ordering::Release);
        self.offset_begin.store(offset, Ordering::Release);
        // Reset the accumulator in case this core was previously issued
        // and the app is re-issuing.
        self.accumulated.store(0, Ordering::Release);
        self.status
            .store(QueryStatus::Pending as u64, Ordering::Release);
    }

    /// Called by the encoder thread on the END closure.
    ///
    /// Records the frame's submit seq and the slot index just past the
    /// query's last write, so `sum_slots` knows the half-open range to
    /// accumulate.
    pub fn end(&self, seq: u64, offset: u32) {
        self.seq_end.store(seq, Ordering::Release);
        self.offset_end.store(offset, Ordering::Release);
    }

    /// `GetData` result: DWORD visible-pixel count clamped at `u32::MAX`.
    ///
    /// # Panics
    ///
    /// Panics if the clamped accumulator overflows `u32::MAX` — unreachable
    /// because the `.min(u32::MAX)` directly above bounds it.
    pub fn get_u32(&self) -> u32 {
        let v = self.accumulated.load(Ordering::Acquire);
        u32::try_from(v.min(u64::from(u32::MAX))).expect("clamped above to u32::MAX")
    }

    /// Raw 64-bit visible-sample sum (un-clamped).
    ///
    /// `GetData` writes up to 8 bytes from this when the caller's buffer
    /// exceeds the advertised DWORD size, matching the runtime's internal
    /// UINT64 counter.
    pub fn get_u64(&self) -> u64 {
        self.accumulated.load(Ordering::Acquire)
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
    pub fn mark_armed(&self) {
        self.end_requested.store(false, Ordering::Release);
        self.status
            .store(QueryStatus::Pending as u64, Ordering::Release);
    }

    /// Record that `Issue(D3DISSUE_END)` was called on the API thread.
    ///
    /// Set before the encoder-side `end` closure runs. Read by the
    /// blocking `GetData(FLUSH)` to decide whether the query is safe to
    /// flush + read (ended) or must report `S_FALSE` (still open).
    pub fn mark_end_requested(&self) {
        self.end_requested.store(true, Ordering::Release);
    }

    /// Whether an `Issue(D3DISSUE_END)` has been recorded since the last `Issue(D3DISSUE_BEGIN)`.
    ///
    /// See [`Self::mark_end_requested`].
    pub fn end_requested(&self) -> bool {
        self.end_requested.load(Ordering::Acquire)
    }

    /// Slot where Metal started counting for this query.
    ///
    /// Encoder reads it in the exhaustion fallback to emit a
    /// `[begin, begin)` span (summing to 0, then overridden to `u32::MAX`
    /// at intake).
    pub fn offset_begin(&self) -> u32 {
        self.offset_begin.load(Ordering::Acquire)
    }

    /// Submit-seq the encoder will retire this query at, set when the END closure runs.
    ///
    /// `0` means END has not yet been processed. API-thread
    /// `GetData(FLUSH)` reads this against the device's `coherent_seq` to
    /// skip the encoder round-trip when intake provably can't finalize
    /// this query yet.
    pub fn seq_end_loaded(&self) -> u64 {
        self.seq_end.load(Ordering::Acquire)
    }

    /// Stores the completed sum and flips status to `Issued`.
    ///
    /// Used only from `VisibilityQueryState::intake_completed` in this
    /// module.
    fn finalize(&self, summed: u64) {
        self.accumulated.store(summed, Ordering::Release);
        self.status
            .store(QueryStatus::Issued as u64, Ordering::Release);
    }

    fn offset_end_internal(&self) -> u32 {
        self.offset_end.load(Ordering::Acquire)
    }
}

/// Monotonic u64-slot allocator reset at each frame boundary.
///
/// Encapsulated inside `VisibilityQueryState` — the encoder reaches it
/// via `VisibilityQueryState::bump_slot`.
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

/// A query whose END has been emitted but whose sum is not yet known.
///
/// Held keyed by the `submit_seq` of the frame that contained END; the
/// encoder thread finalizes the core once `coherent_seq >= submit_seq`.
struct PendingQuery {
    submit_seq: u64,
    core: Arc<VisibilityQueryCore>,
}

/// Composite encoder-side state for visibility queries.
///
/// Lives on `FrameEncoder`: persistent across frames (pool, pending list)
/// with per-frame pieces (allocator, `active_count`, current buffer) reset in
/// `begin_frame`.
pub struct VisibilityQueryState {
    allocator: VisibilityOffsetAllocator,
    pool: VisibilityBufferPool,
    /// Queries whose END was emitted on some frame.
    ///
    /// Finalized once `coherent_seq` catches up to their `submit_seq`.
    pending: VecDeque<PendingQuery>,
    /// Set of queries currently between BEGIN and END on the encoder thread.
    ///
    /// Incremented on BEGIN, decremented on END. Used to decide whether a
    /// pass boundary needs a Counting-mode re-arm and whether END should
    /// emit Disabled vs a fresh Counting slot.
    active_count: u32,
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
            active_count: 0,
            current_buffer: None,
            exhausted_this_frame: false,
        }
    }

    /// Reset per-frame fields.
    ///
    /// Called on the encoder thread at `begin_frame` *after* the current
    /// frame's buffer (if any) has been moved into retention via
    /// `retire_current_buffer`. Does not touch the pool or the pending
    /// list — those drain separately.
    pub const fn reset_frame(&mut self) {
        self.allocator.reset();
        self.active_count = 0;
        self.exhausted_this_frame = false;
    }

    #[must_use]
    pub const fn active_count(&self) -> u32 {
        self.active_count
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

    pub const fn mark_exhausted(&mut self) {
        self.exhausted_this_frame = true;
    }

    /// Encoder-side slot allocation.
    ///
    /// Wraps the private allocator so the encoder never touches it
    /// directly.
    pub const fn bump_slot(&mut self) -> Option<u32> {
        self.allocator.bump()
    }

    pub const fn inc_active(&mut self) {
        self.active_count += 1;
    }

    pub const fn dec_active(&mut self) {
        if self.active_count > 0 {
            self.active_count -= 1;
        }
    }

    pub fn push_pending(&mut self, submit_seq: u64, core: Arc<VisibilityQueryCore>) {
        self.pending.push_back(PendingQuery { submit_seq, core });
    }

    /// Drain every owned `RetiredVisibilityBuffer` and clear the pending list.
    ///
    /// Covers the current-frame slot plus the pool's retired and free lists.
    /// Caller takes ownership of the returned vec; each entry's `into_parts`
    /// yields the (`PageBox`, `metal_handle`, `release_seq`) triple the encoder
    /// feeds through its destroy-then-drop ordering at shutdown.
    pub fn drain_all_buffers(&mut self) -> Vec<RetiredVisibilityBuffer> {
        let mut all = Vec::new();
        if let Some(cur) = self.current_buffer.take() {
            all.push(cur);
        }
        all.append(&mut self.pool.retired);
        all.append(&mut self.pool.free);
        self.pending.clear();
        all
    }

    /// Drain pending → finalize → release pool entries up to `coherent_seq`.
    ///
    /// Every pending query whose END frame has retired on the GPU is finalized:
    /// its slot span is summed from the retired visibility buffer matching its
    /// `submit_seq`. A buffer that cannot be found — retired and evicted before
    /// intake ran, which the normal flow never produces — falls back to the
    /// permissive `u32::MAX`. Retired buffers whose seq has been reached then
    /// move into the free list so the next frame can reuse them.
    ///
    /// Each retired buffer's `PageBox` points at PE-allocated Shared storage
    /// wrapped by Metal. Once `coherent_seq >= release_seq` the GPU is done
    /// writing, so CPU reads of the backing are coherent; the caller establishes
    /// that by passing the atomic's observed value.
    ///
    /// # Panics
    ///
    /// Panics if `pending` and `slot_used` fall out of sync with the slot
    /// allocator — an invariant maintained by the
    /// [`VisibilityQueryCore::begin`] / [`VisibilityQueryCore::end`] helpers,
    /// so unreachable on a well-formed call sequence.
    pub fn intake_completed(&mut self, coherent_seq: u64) {
        // Finalize first (reads retired buffers by seq), then release
        // pool entries. Order matters: release_up_to moves retired →
        // free which clears the seq association.
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].submit_seq > coherent_seq {
                i += 1;
                continue;
            }
            let entry = self.pending.remove(i).expect("bound-checked");
            let sum = match self.pool.retired_backing_for(entry.submit_seq) {
                Some(slots) => sum_slots(
                    &slots,
                    entry.core.offset_begin(),
                    entry.core.offset_end_internal(),
                ),
                None => u64::from(u32::MAX),
            };
            entry.core.finalize(sum);
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
