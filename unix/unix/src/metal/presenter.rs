//! Per-queue presentation: packets, snapshot slots, and the thread that presents them.
//!
//! A frame's render command buffer commits without a drawable. What it
//! leaves for presentation is a [`PresentPacket`]: the layer, the texture the
//! present reads, and the frame's sequence. One native thread per queue takes
//! the packets in order, acquires the drawable, encodes the present route into
//! a command buffer of its own and commits it. A read-back that flushes the
//! frame in progress therefore waits for committed render work and the GPU,
//! never for the display.
//!
//! The one hazard the split creates is a later render overwriting the back
//! buffer before a pending present has read it. A present-bearing submit
//! waits for the previous present to commit, which is the cadence the display
//! set before the split and costs nothing. A submit that must not wait, a
//! no-present partial frame or one a barrier hurries, copies the pending
//! present's source into a slot and retargets the packet at it, so the frame
//! it shows is the one it was given. At most one pending packet reads the back
//! buffer; every later one reads a slot. A slot is reused once the present
//! that read it has committed: the next copy into it is a later buffer on the
//! same queue, and Metal executes a queue's buffers in commit order.
//!
//! Lock order, top to bottom: a state's `inner`, held across the encode and
//! commit of one present
//! or snapshot buffer and never across `nextDrawable`, `waitUntilCompleted`,
//! the gate or a synchronous main-thread hop; then the upscale caches, the
//! in-flight command buffer registry, the attachment registry and its
//! per-record streak, the cursor overlay's shared state, in that order, each
//! held for a lookup or one encode and taking nothing below it. A present's
//! completion handler takes the in-flight registry alone. The only blocking
//! under `inner` is a wait on one of the state's own condition variables.

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use block2::RcBlock;
use mtld3d_shared::{
    SubmitFrameParams,
    mtl::{PRESENT_PIPELINE_DEPTH, PresentWaitPolicy, SnapshotFlags},
    mtl_handle::{CAMetalLayerKind, MTLCommandQueueKind, MTLTextureKind, MetalHandle},
    perf::NanosSetTimer,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLOrigin, MTLPixelFormat, MTLResource, MTLSize, MTLTexture,
};
use objc2_quartz_core::CAMetalLayer;

use super::{
    command::{self, PresentEncode, diagnostics},
    handle::{IntoRetained, IntoRetainedLayer, ReleaseRetain},
    macdrv::attachment,
    record::DeviceRecord,
    texture,
};
use crate::LOG_TARGET;

/// Slots a snapshot can copy into, per queue.
///
/// A slot is busy from the copy until the present reading it commits, and
/// the presents that can be pending at once are as many as the PE pipeline
/// holds ahead of a partial submit, which is what [`PRESENT_PIPELINE_DEPTH`]
/// counts. A read-back behind a full pipeline hurries every one of those
/// frames past the presenter and copies each frame's image once, so this
/// many slots keep one read-back off the display whatever the pipeline
/// holds; one fewer and the last copy would wait for the oldest present to
/// commit, which is the display's pacing again. Read-backs on consecutive
/// frames that outrun the display still meet a busy ring, and the perf
/// grid's `Slot waits` counts that. Textures are allocated only when a copy
/// needs the slot.
pub const SNAPSHOT_SLOTS: usize = PRESENT_PIPELINE_DEPTH;

/// How often a parked presenter looks for the gate file to be gone.
const GATE_POLL: Duration = Duration::from_millis(1);

bitflags::bitflags! {
    /// The two switches a state carries besides its queue of packets.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct PresenterFlags: u8 {
        /// Snapshot a pending present instead of waiting for it.
        ///
        /// A level the PE-side barrier sets before it waits for the submits
        /// in flight and clears after, so a submit parked in its wait wakes
        /// and copies, and one that reaches the decision meanwhile copies
        /// too. Not a one-shot: two submits can be in flight behind one
        /// barrier.
        const HURRY = 1 << 0;
        /// The queue is going away: drop every packet, wake every waiter.
        const STOP = 1 << 1;
    }
}

/// What one queue keeps for presentation, shared by its threads.
pub struct PresentState {
    inner: Mutex<Inner>,
    /// Woken when a present commits or drops, a slot frees, or a flag moves.
    submit_cv: Condvar,
    /// Woken when a packet is pushed or `STOP` is set.
    presenter_cv: Condvar,
    /// Highest sequence whose present buffer retired.
    ///
    /// The counter the present buffers register under in the in-flight map,
    /// so the retirement wait serves presentation as it serves rendering. It
    /// lives here rather than on the PE side because nothing there reads it.
    /// Only a present buffer's completion handler advances it: a packet the
    /// presenter drops has no buffer, and the idle wait targets the last
    /// buffer that was committed (`Inner::presented_seq`), never a dropped
    /// sequence.
    present_retired: AtomicU64,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Whether the last drawable acquisition failed, for the stall dump.
    ///
    /// Written by this device's presenter thread alone; an atomic because
    /// the record it lives on is shared. Per device so two presenters do not
    /// clear each other's edge and swallow the dump.
    stalled: AtomicBool,
}

struct Inner {
    /// Packets in presentation order; the front is the one being presented.
    pending: VecDeque<PresentPacket>,
    /// Sequence of the last packet the presenter committed or dropped.
    ///
    /// What a submit waits for and what frees a slot: a dropped packet
    /// consumed its sequence like a committed one.
    committed_present_seq: u64,
    /// Sequence of the last present buffer the presenter committed.
    ///
    /// What an idle wait retires. A dropped packet leaves it alone, since no
    /// buffer carries the dropped sequence and none will retire it.
    presented_seq: u64,
    flags: PresenterFlags,
    slots: [Option<Slot>; SNAPSHOT_SLOTS],
    /// The last present's `nextDrawable` wait, handed back to the next submit.
    last_drawable_wait_ns: u64,
    /// The gate file `debug.presentGateFile` named, `None` = no gate.
    gate: Option<PathBuf>,
}

/// One frame's presentation, from the render buffer's commit to the present.
///
/// Owns a retain on the texture it presents and on the layer, as wire
/// handles rather than `Retained` objects so the state stays `Send` without
/// asserting it. The texture is the back buffer until a snapshot retargets
/// the packet at a slot.
pub struct PresentPacket {
    seq: u64,
    source: MetalHandle<MTLTextureKind>,
    layer: MetalHandle<CAMetalLayerKind>,
    slot: Option<usize>,
    /// The `NSView*` the layer belongs to: the attachment record's key.
    view: usize,
}

impl PresentPacket {
    /// Take ownership of one retain on each object for the packet's lifetime.
    pub fn new(
        seq: u64,
        source: Retained<ProtocolObject<dyn MTLTexture>>,
        layer: Retained<CAMetalLayer>,
        view: usize,
    ) -> Self {
        // SAFETY: `Retained::into_raw` transfers the retain into the raw
        // address; the handle carries it until `Drop` releases it.
        let source =
            unsafe { MetalHandle::<MTLTextureKind>::new(Retained::into_raw(source) as u64) };
        // SAFETY: as above, for the layer.
        let layer =
            unsafe { MetalHandle::<CAMetalLayerKind>::new(Retained::into_raw(layer) as u64) };
        Self {
            seq,
            source,
            layer,
            slot: None,
            view,
        }
    }
}

impl Drop for PresentPacket {
    fn drop(&mut self) {
        // SAFETY: the handle holds the retain `new` took and no copy of it
        // outlives the packet; a command buffer that encoded the object holds
        // its own.
        unsafe { self.source.release_retain() };
        // SAFETY: as above, for the layer.
        unsafe { self.layer.release_retain() };
    }
}

/// A texture a snapshot copies the back buffer into.
///
/// Busy while `reader` names a packet the presenter has not committed or
/// dropped; free once it has, since the next copy into it is a later command
/// buffer on the same queue.
struct Slot {
    texture: MetalHandle<MTLTextureKind>,
    width: u32,
    height: u32,
    format: MTLPixelFormat,
    /// Sequence of the packet retargeted at this slot.
    reader: u64,
}

impl Slot {
    fn matches(&self, width: u32, height: u32, format: MTLPixelFormat) -> bool {
        self.width == width && self.height == height && self.format == format
    }

    const fn busy(&self, committed_present_seq: u64) -> bool {
        self.reader > committed_present_seq
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        texture::destroy_texture(self.texture.raw());
    }
}

/// What a submit does about the pending present that reads the back buffer.
#[derive(Debug, PartialEq, Eq)]
enum Decision {
    /// No pending present reads the back buffer.
    Proceed,
    /// Wait for the present with this sequence to commit.
    Wait(u64),
    /// Copy the back buffer into a slot and retarget the present with this sequence.
    Snapshot(u64),
}

/// Which slot a snapshot takes.
#[derive(Debug, PartialEq, Eq)]
enum SlotChoice {
    Free(usize),
    /// Every slot is busy; this one's reader is the oldest.
    Busy(usize),
}

impl PresentState {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn present_retired_ptr(&self) -> u64 {
        core::ptr::from_ref(&self.present_retired) as u64
    }

    /// Build a device's presentation state, with no thread yet.
    ///
    /// The thread starts separately, once the record that owns this state
    /// exists, since the thread holds a reference to it.
    pub fn new(gate: Option<PathBuf>) -> Self {
        if let Some(path) = &gate {
            log::info!(
                target: LOG_TARGET,
                "presenter: gated at {} (parks before each drawable while it exists)",
                path.display(),
            );
        }
        Self {
            inner: Mutex::new(Inner {
                pending: VecDeque::new(),
                committed_present_seq: 0,
                presented_seq: 0,
                flags: PresenterFlags::empty(),
                slots: [const { None }; SNAPSHOT_SLOTS],
                last_drawable_wait_ns: 0,
                gate,
            }),
            submit_cv: Condvar::new(),
            presenter_cv: Condvar::new(),
            present_retired: AtomicU64::new(0),
            thread: Mutex::new(None),
            stalled: AtomicBool::new(false),
        }
    }

    /// Dump the recent crumb ring on each edge of this device's stall condition.
    ///
    /// An intermittent present stall then self-documents in the log with no
    /// manual timing. `stalled` is whether the current present failed to
    /// acquire its drawable; the ring is written when that flips in either
    /// direction, so the rising edge captures the lead-up and the falling
    /// edge the whole episode.
    fn note_stall(&self, stalled: bool) {
        if self.stalled.swap(stalled, Ordering::Relaxed) != stalled {
            mtld3d_shared::crumb::dump_recent(512);
        }
    }
}

/// The presented-cadence probe's state, one per device.
///
/// Read and written by the presented handlers Metal runs on its own thread,
/// so both halves are atomics. Per device because the probe reports the
/// interval between consecutive frames of one swap chain: two devices
/// sharing it would report the interleaving of two cadences as one.
pub struct Presented {
    last_ns: AtomicU64,
    typical_ns: AtomicU64,
}

impl Default for Presented {
    fn default() -> Self {
        Self::new()
    }
}

impl Presented {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_ns: AtomicU64::new(0),
            typical_ns: AtomicU64::new(0),
        }
    }

    /// Host time of the previous presented frame, replaced by this one's.
    ///
    /// Zero means unseeded: the first frame of a device has no interval.
    pub fn swap_last(&self, now_ns: u64) -> u64 {
        self.last_ns.swap(now_ns, Ordering::AcqRel)
    }

    /// Exponential running average of the presented interval, ns; 0 = unseeded.
    pub fn typical(&self) -> u64 {
        self.typical_ns.load(Ordering::Relaxed)
    }

    pub fn set_typical(&self, ns: u64) {
        self.typical_ns.store(ns, Ordering::Relaxed);
    }
}

/// Decide for one submit, given what is pending.
///
/// A present-bearing submit waits for every pending present to commit,
/// whether the newest reads the back buffer or a copy: that is the pacing
/// the display sets, and it keeps the queue one deep except behind a
/// barrier. A submit that must not wait, a no-present partial frame or one
/// a barrier hurried, copies the newest pending present's frame if it still
/// reads the back buffer, and proceeds if it does not. The pending packet
/// reading the back buffer is at most one and always the last pushed, since
/// every submit waits for it or retargets it before pushing its own. `STOP`
/// means every packet is about to be dropped, so nothing will read the back
/// buffer.
fn decide(inner: &Inner, present_bearing: bool) -> Decision {
    if inner.flags.contains(PresenterFlags::STOP) {
        return Decision::Proceed;
    }
    let Some(newest) = inner.pending.back() else {
        return Decision::Proceed;
    };
    if present_bearing && !inner.flags.contains(PresenterFlags::HURRY) {
        Decision::Wait(newest.seq)
    } else if newest.slot.is_none() {
        Decision::Snapshot(newest.seq)
    } else {
        Decision::Proceed
    }
}

/// The first free slot, else the one whose reader is oldest.
fn choose_slot(inner: &Inner) -> SlotChoice {
    let committed = inner.committed_present_seq;
    let mut oldest = (0, u64::MAX);
    for (index, slot) in inner.slots.iter().enumerate() {
        match slot {
            Some(slot) if slot.busy(committed) => {
                if slot.reader < oldest.1 {
                    oldest = (index, slot.reader);
                }
            }
            _ => return SlotChoice::Free(index),
        }
    }
    SlotChoice::Busy(oldest.0)
}

/// Start the presenter thread for `record`.
///
/// `false` when the thread cannot start, which leaves the device unable to
/// present; the caller fails the device rather than run one that renders
/// into nothing. The thread holds an `Arc` of the record, so the record
/// outlives it whatever the PE side does with its handle.
pub fn spawn(record: &Arc<DeviceRecord>) -> bool {
    let worker = Arc::clone(record);
    let queue = record.queue();
    let spawned = thread::Builder::new()
        .name("mtld3d-present".to_owned())
        .spawn(move || presenter_main(&worker, queue));
    let handle = match spawned {
        Ok(handle) => handle,
        Err(error) => {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "presenter: cannot start the presenter thread for queue {:#x} ({error}); \
                 the device is refused",
                queue.raw(),
            );
            return false;
        }
    };
    *record
        .present()
        .thread
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = Some(handle);
    true
}

/// Stop the presenter thread and join it: drop every packet, wake every waiter.
///
/// Runs before the queue is released, and after the PE side has drained its
/// submit thread and waited for presentation to go idle, so the stop finds
/// nothing pending in the ordinary case; a packet it does find is dropped
/// with a warning. A thread inside `nextDrawable` is joined once that call
/// returns, within the layer's timeout.
pub fn stop_and_join(state: &PresentState) {
    state.lock().flags.insert(PresenterFlags::STOP);
    state.presenter_cv.notify_all();
    state.submit_cv.notify_all();
    let handle = state
        .thread
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(handle) = handle
        && handle.join().is_err()
    {
        log::error!(target: LOG_TARGET, "presenter: the presenter thread panicked");
    }
}

/// Set or clear the hurry level for `queue`.
pub fn set_wait_policy(state: &PresentState, policy: PresentWaitPolicy) {
    {
        let mut inner = state.lock();
        match policy {
            PresentWaitPolicy::WaitForCommit => inner.flags.remove(PresenterFlags::HURRY),
            PresentWaitPolicy::SnapshotPending => inner.flags.insert(PresenterFlags::HURRY),
        }
    }
    state.submit_cv.notify_all();
}

/// Wait until every queued present has committed or dropped, and the last one retired.
///
/// The caller has drained its submit thread, so no packet is pushed
/// meanwhile and the wait ends. The retirement wait runs outside the lock,
/// against the present counter and with no failed-submit sink: a present the
/// GPU killed is logged by its completion handler and never marks the frame's
/// uploads as failed. It targets the last present buffer the presenter
/// committed, not the last packet it consumed: a dropped packet has no
/// buffer, so its sequence never retires and must not be waited for, and a
/// drop behind a committed present must not stand in for that present's
/// retirement either.
pub fn wait_for_present_idle(record: &DeviceRecord) {
    let state = record.present();
    let presented = {
        let inner = state.lock();
        let inner = state
            .submit_cv
            .wait_while(inner, |inner| {
                !inner.pending.is_empty() && !inner.flags.contains(PresenterFlags::STOP)
            })
            .unwrap_or_else(PoisonError::into_inner);
        inner.presented_seq
    };
    command::wait_for_gpu_retire(record.pending(), presented, state.present_retired_ptr(), 0);
}

/// Hand a frame's presentation to the presenter; returns the last drawable wait.
///
/// The wait is the previous present's, which is what the perf grid reports
/// for the submit that hands the next one over.
pub fn push(state: &PresentState, packet: PresentPacket) -> u64 {
    let mut inner = state.lock();
    if inner.flags.contains(PresenterFlags::STOP) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: a frame was pushed after the queue's presenter stopped; it is not presented",
        );
        drop_packet(&mut inner, packet);
        return inner.last_drawable_wait_ns;
    }
    inner.pending.push_back(packet);
    state.presenter_cv.notify_one();
    inner.last_drawable_wait_ns
}

/// Make the pending present and this submit's render work compatible.
///
/// Called after the frame's buffers are encoded and before they commit.
/// Waits for the pending present to commit, or copies its source into a slot
/// and retargets it, as the module doc says; `params.present_wait_ns` takes
/// the wait and `params.snapshot_flags` says whether a copy was made and
/// whether it had to wait for a slot.
pub fn resolve_present_conflict(
    state: &PresentState,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    params: &mut SubmitFrameParams,
    present_bearing: bool,
) {
    let mut inner = state.lock();
    loop {
        match decide(&inner, present_bearing) {
            Decision::Proceed => break,
            Decision::Wait(seq) => {
                mtld3d_shared::crumb!("submit:presentwait", seq);
                let _wait = NanosSetTimer::start(&raw mut params.present_wait_ns);
                inner = state
                    .submit_cv
                    .wait_while(inner, |inner| {
                        inner.committed_present_seq < seq
                            && !inner
                                .flags
                                .intersects(PresenterFlags::HURRY | PresenterFlags::STOP)
                    })
                    .unwrap_or_else(PoisonError::into_inner);
            }
            Decision::Snapshot(seq) => match choose_slot(&inner) {
                SlotChoice::Busy(index) => {
                    let reader = inner.slots[index].as_ref().map_or(0, |slot| slot.reader);
                    params.snapshot_flags.insert(SnapshotFlags::SLOT_WAITED);
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "presenter: every snapshot slot holds a present still waiting for its \
                         drawable; this submit waits for the oldest to commit",
                    );
                    inner = state
                        .submit_cv
                        .wait_while(inner, |inner| {
                            inner.committed_present_seq < reader
                                && !inner.flags.contains(PresenterFlags::STOP)
                        })
                        .unwrap_or_else(PoisonError::into_inner);
                }
                SlotChoice::Free(index) => {
                    if snapshot_into(&mut inner, index, seq, queue) {
                        params.snapshot_flags.insert(SnapshotFlags::TAKEN);
                        break;
                    }
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "presenter: no snapshot could be taken; this submit waits for the \
                         pending present to commit instead",
                    );
                    inner = state
                        .submit_cv
                        .wait_while(inner, |inner| {
                            inner.committed_present_seq < seq
                                && !inner.flags.contains(PresenterFlags::STOP)
                        })
                        .unwrap_or_else(PoisonError::into_inner);
                }
            },
        }
    }
}

/// Copy the pending present's source into slot `index` and retarget the packet.
///
/// The copy rides a command buffer of its own, committed here, ahead of the
/// caller's render work on the same queue. `false` when the slot cannot be
/// allocated or the buffer cannot be created; nothing is retargeted then.
fn snapshot_into(
    inner: &mut Inner,
    index: usize,
    seq: u64,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> bool {
    let Some(source) = inner
        .pending
        .iter()
        .find(|packet| packet.seq == seq)
        .and_then(|packet| packet.source.into_retained())
    else {
        return false;
    };
    let width = u32::try_from(source.width()).expect("a texture width fits u32");
    let height = u32::try_from(source.height()).expect("a texture height fits u32");
    let format = source.pixelFormat();
    let fits = inner.slots[index]
        .as_ref()
        .is_some_and(|slot| slot.matches(width, height, format));
    if !fits {
        let Some(wire_format) = texture::wire_pixel_format(format) else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "presenter: {format:?} is not a format a snapshot slot can take",
            );
            return false;
        };
        let device = queue.device();
        let Some(handle) = texture::create_upscale_target(&device, width, height, wire_format)
        else {
            return false;
        };
        if let Some(slot_texture) = handle.into_retained() {
            let label =
                objc2_foundation::NSString::from_str(&format!("mtld3d-snapshot-slot-{index}"));
            slot_texture.setLabel(Some(&label));
        }
        // A replaced slot's texture stays alive in the buffers that encoded it.
        inner.slots[index] = Some(Slot {
            texture: handle,
            width,
            height,
            format,
            reader: 0,
        });
    }
    let Some(slot_texture) = inner.slots[index]
        .as_ref()
        .and_then(|slot| slot.texture.into_retained())
    else {
        return false;
    };
    let Some(cb) = diagnostics::command_buffer(queue) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: commandBuffer() returned nil for a snapshot",
        );
        return false;
    };
    let label = objc2_foundation::NSString::from_str(&format!("mtld3d-snapshot-{seq:#x}"));
    cb.setLabel(Some(&label));
    let Some(blit) = cb.blitCommandEncoder() else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: blitCommandEncoder() returned nil for a snapshot",
        );
        return false;
    };
    let blit_label = objc2_foundation::NSString::from_str("mtld3d-snapshot-blit");
    blit.setLabel(Some(&blit_label));
    let origin = MTLOrigin { x: 0, y: 0, z: 0 };
    let size = MTLSize {
        width: source.width(),
        height: source.height(),
        depth: 1,
    };
    // SAFETY: objc2 typed binding; both textures are live, same format and
    // extent, and the region is their whole level 0.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &source, 0, 0, origin, size, &slot_texture, 0, 0, origin,
        );
    }
    blit.endEncoding();
    mtld3d_shared::crumb!("submit:snapshot", seq, index as u64);
    cb.commit();
    if let Some(packet) = inner.pending.iter_mut().find(|packet| packet.seq == seq) {
        packet.slot = Some(index);
    }
    if let Some(slot) = inner.slots[index].as_mut() {
        slot.reader = seq;
    }
    true
}

/// Consume a packet that presents nothing.
///
/// Marks it committed so a submit waiting for it goes on and a slot it read
/// frees. The retirement counter stays where it is: no present buffer
/// carries this sequence, and advancing the counter past a committed
/// present still on the GPU would end an idle wait early.
fn drop_packet(inner: &mut Inner, packet: PresentPacket) {
    inner.committed_present_seq = inner.committed_present_seq.max(packet.seq);
    drop(packet);
}

/// Drop the front packet, which the caller peeked as `seq`.
fn drop_front(state: &PresentState, seq: u64) {
    {
        let mut inner = state.lock();
        if let Some(packet) = inner.pending.pop_front() {
            debug_assert_eq!(packet.seq, seq, "only the presenter pops");
            drop_packet(&mut inner, packet);
        }
    }
    state.submit_cv.notify_all();
}

/// The presenter thread's body.
fn presenter_main(record: &Arc<DeviceRecord>, queue: MetalHandle<MTLCommandQueueKind>) {
    let Some(queue) = queue.into_retained() else {
        log::error!(target: LOG_TARGET, "presenter: queue retain failed (handle={queue:#x})");
        return;
    };
    while objc2::rc::autoreleasepool(|_| present_frame(record, &queue)) {}
}

/// One presenter iteration: take the front packet through to its commit.
///
/// `false` once the state is stopped and every packet dropped.
fn present_frame(record: &Arc<DeviceRecord>, queue: &ProtocolObject<dyn MTLCommandQueue>) -> bool {
    let state = record.present();
    let (seq, layer, view, gate) = {
        let inner = state.lock();
        let mut inner = state
            .presenter_cv
            .wait_while(inner, |inner| {
                inner.pending.is_empty() && !inner.flags.contains(PresenterFlags::STOP)
            })
            .unwrap_or_else(PoisonError::into_inner);
        if inner.flags.contains(PresenterFlags::STOP) {
            if !inner.pending.is_empty() {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "presenter: stopped with {} present(s) still queued; they are dropped",
                    inner.pending.len(),
                );
            }
            while let Some(packet) = inner.pending.pop_front() {
                drop_packet(&mut inner, packet);
            }
            state.submit_cv.notify_all();
            return false;
        }
        let front = inner.pending.front().expect("the wait ended on a packet");
        (front.seq, front.layer, front.view, inner.gate.clone())
    };
    if let Some(gate) = gate {
        while gate.exists() {
            if state.lock().flags.contains(PresenterFlags::STOP) {
                return true;
            }
            thread::sleep(GATE_POLL);
        }
    }
    let attachment = attachment::find(view);
    if attachment.is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: view {view:#x} has no attachment record; presenting as not \
             occluded, headroom 1.0, unthrottled, stretch route",
        );
    }
    if attachment.as_ref().is_some_and(|att| att.window_occluded()) {
        // Window fully occluded: the compositor is not recycling drawables,
        // so `nextDrawable` would block its full timeout for nothing that
        // reaches the screen. The render work committed already; the frame
        // is simply not shown.
        mtld3d_shared::crumb!("present:occluded-skip", layer.raw());
        drop_front(state, seq);
        return true;
    }
    let Some(layer) = IntoRetainedLayer::into_retained(layer) else {
        drop_front(state, seq);
        return true;
    };
    // Re-point the drawable at the layer's backing store before acquiring
    // one. A window resize changes the layer under us and `drawableSize`
    // does not follow on its own, so without this the frames between the
    // resize and the guest's own reaction would be composited at the old
    // size, which means rescaled.
    super::macdrv::sync_drawable_size(&layer);
    mtld3d_shared::crumb!("present:nextdraw", seq);
    let mut drawable_wait_ns: u64 = 0;
    let drawable = {
        let _wait = NanosSetTimer::start(&raw mut drawable_wait_ns);
        layer.nextDrawable()
    };
    // A nil drawable means `nextDrawable` exhausted its timeout; self-dump
    // the ring on the onset and on recovery so an intermittent stall is
    // captured in the log without manual timing.
    state.note_stall(drawable.is_none());
    let Some(drawable) = drawable else {
        // Visible, yet no drawable within the timeout: a rare compositor
        // stall, or an occlusion signal that has not propagated yet. The
        // frame is dropped; its render work is committed already.
        mtld3d_shared::crumb!("present:nodrawable", seq, drawable_wait_ns);
        drop_front(state, seq);
        return true;
    };

    let mut inner = state.lock();
    let Some(front) = inner.pending.front() else {
        return true;
    };
    debug_assert_eq!(front.seq, seq, "only the presenter pops");
    let source_handle = front.slot.map_or(front.source, |index| {
        inner.slots[index]
            .as_ref()
            .map_or(MetalHandle::NULL, |slot| slot.texture)
    });
    let Some(source) = source_handle.into_retained() else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: the frame's texture could not be retained; the frame is not presented",
        );
        pop_and_drop(&mut inner, state);
        return true;
    };
    let Some(cb) = diagnostics::command_buffer(queue) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "presenter: commandBuffer() returned nil; the frame is not presented",
        );
        pop_and_drop(&mut inner, state);
        return true;
    };
    let label = objc2_foundation::NSString::from_str(&format!("mtld3d-present-{seq:#x}"));
    cb.setLabel(Some(&label));
    command::encode_present(
        &cb,
        &drawable,
        &PresentEncode {
            record,
            attachment: attachment.as_ref(),
            source: &source,
            source_raw: source_handle.raw(),
            seq,
            drawable_wait_ns,
        },
    );
    super::upscale::retire_evicted(&cb, record.upscale());
    let present_ptr = state.present_retired_ptr();
    let owner = Arc::clone(record);
    let handler = RcBlock::new(
        move |cb_ptr: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal invokes the block with the completed command
            // buffer; the pointer is valid for the handler's duration.
            let cb = unsafe { cb_ptr.as_ref() };
            let status = cb.status();
            diagnostics::completion(cb, status, Some(seq), "present-callback");
            if status == MTLCommandBufferStatus::Error {
                let error = cb.error();
                diagnostics::failure(cb, Some(seq), "present-callback", error.as_deref());
                let (code, desc) = command::command_buffer_error(error.as_deref());
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: code,
                    "presenter: present buffer for frame seq={seq} failed on the GPU (code \
                     {code}: {desc}); the drawable showed undefined memory",
                );
            }
            owner
                .present()
                .present_retired
                .fetch_max(seq, Ordering::Release);
            command::unregister_pending(owner.pending(), present_ptr, seq);
        },
    );
    // SAFETY: objc2 typed binding; Metal retains the block on registration,
    // so the local `handler` may drop when this returns.
    unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&handler)) };
    mtld3d_shared::crumb!("present:commit", seq);
    command::commit_registered(record.pending(), &cb, present_ptr, seq);
    let packet = inner.pending.pop_front();
    inner.committed_present_seq = inner.committed_present_seq.max(seq);
    inner.presented_seq = seq;
    inner.last_drawable_wait_ns = drawable_wait_ns;
    drop(inner);
    state.submit_cv.notify_all();
    drop(packet);
    true
}

/// Pop the front packet under the caller's lock and consume it unpresented.
fn pop_and_drop(inner: &mut MutexGuard<'_, Inner>, state: &PresentState) {
    if let Some(packet) = inner.pending.pop_front() {
        drop_packet(inner, packet);
    }
    state.submit_cv.notify_all();
}

#[cfg(test)]
mod tests;
