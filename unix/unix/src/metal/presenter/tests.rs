//! The presenter's decisions and bookkeeping, without a device where possible.
//!
//! `Inner` is built by hand with null handles: a packet's `Drop` and a slot's
//! `Drop` both accept a null, so no Metal object is ever touched. The tests
//! that need a queue skip when the machine has no Metal device.

use std::{
    sync::{Arc, atomic::Ordering},
    thread,
    time::Duration,
};

use mtld3d_shared::mtl_handle::{CAMetalLayerKind, MTLCommandQueueKind, MTLTextureKind};
use objc2_metal::MTLPixelFormat;

use super::*;

fn packet(seq: u64, slot: Option<usize>) -> PresentPacket {
    PresentPacket {
        seq,
        source: MetalHandle::<MTLTextureKind>::NULL,
        layer: MetalHandle::<CAMetalLayerKind>::NULL,
        slot,
        view: 0,
    }
}

fn slot(reader: u64) -> Slot {
    Slot {
        texture: MetalHandle::<MTLTextureKind>::NULL,
        width: 640,
        height: 480,
        format: MTLPixelFormat::BGRA8Unorm,
        reader,
    }
}

fn empty_inner() -> Inner {
    Inner {
        queue: MetalHandle::<MTLCommandQueueKind>::NULL,
        pending: VecDeque::new(),
        committed_present_seq: 0,
        presented_seq: 0,
        flags: PresenterFlags::empty(),
        slots: [const { None }; SNAPSHOT_SLOTS],
        last_drawable_wait_ns: 0,
        gate: None,
    }
}

fn state_with(inner: Inner) -> Arc<PresentState> {
    Arc::new(PresentState {
        inner: Mutex::new(inner),
        submit_cv: Condvar::new(),
        presenter_cv: Condvar::new(),
        present_retired: AtomicU64::new(0),
        thread: Mutex::new(None),
        stalled: AtomicBool::new(false),
    })
}

/// The record owns the queue's only retain, so its thread runs on a live queue.
///
/// `create_command_queue` hands the retain to the record and keeps no copy,
/// and the presenter thread retains the queue again at start. Without the
/// record's retain this is a use after free that a loaded machine turns into
/// a crash at thread start.
#[test]
fn a_record_keeps_the_queue_alive_for_its_thread() {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let queue = device
        .newCommandQueue()
        .expect("a queue on the default device");
    // SAFETY: the raw address carries this retain, which the record adopts
    // and releases when it drops at the end of this test.
    let handle =
        unsafe { MetalHandle::<MTLCommandQueueKind>::new(Retained::into_raw(queue) as u64) };
    let record = DeviceRecord::new(handle, None);
    assert!(spawn(&record), "the presenter thread starts");
    stop_and_join(record.present());
}

/// A handle round-trips to its record, and only the destroying caller ends it.
///
/// `borrow` hands out a reference without consuming the handle's own, so a
/// thunk that resolves a device leaves the record live for the next one;
/// `consume` takes that last reference back.
#[test]
fn a_record_handle_round_trips_and_only_consume_ends_it() {
    let record = DeviceRecord::new(MetalHandle::<MTLCommandQueueKind>::NULL, None);
    let handle = Arc::clone(&record).into_handle();
    {
        // SAFETY: the handle came from `into_handle` above and has not been
        // consumed.
        let borrowed = unsafe { DeviceRecord::borrow(handle) }.expect("the record is live");
        assert!(Arc::ptr_eq(&borrowed, &record), "the same record");
    }
    // SAFETY: as above, and nothing names the handle after this.
    let consumed = unsafe { DeviceRecord::consume(handle) }.expect("the record is live");
    assert!(Arc::ptr_eq(&consumed, &record));
    drop(consumed);
    assert_eq!(
        Arc::strong_count(&record),
        1,
        "the handle's reference is gone"
    );
}

#[test]
fn a_present_bearing_submit_waits_for_the_packet_on_the_back_buffer() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, None));
    assert_eq!(decide(&inner, true), Decision::Wait(5));
}

#[test]
fn a_no_present_submit_snapshots_instead_of_waiting() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, None));
    assert_eq!(decide(&inner, false), Decision::Snapshot(5));
}

#[test]
fn a_retargeted_packet_still_paces_a_present_bearing_submit() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, Some(0)));
    assert_eq!(
        decide(&inner, true),
        Decision::Wait(5),
        "the queue stays one deep whatever the pending present reads"
    );
    assert_eq!(
        decide(&inner, false),
        Decision::Proceed,
        "a partial frame conflicts with nothing once the present reads a copy"
    );
    inner.flags.insert(PresenterFlags::HURRY);
    assert_eq!(
        decide(&inner, true),
        Decision::Proceed,
        "nor does a hurried one"
    );
    assert_eq!(
        decide(&empty_inner(), true),
        Decision::Proceed,
        "nothing pending"
    );
}

#[test]
fn a_submit_waits_for_the_newest_pending_present() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, Some(0)));
    inner.pending.push_back(packet(6, None));
    assert_eq!(decide(&inner, true), Decision::Wait(6));
    assert_eq!(decide(&inner, false), Decision::Snapshot(6));
}

#[test]
fn hurry_turns_a_wait_into_a_snapshot_and_stop_into_nothing() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, None));
    inner.flags.insert(PresenterFlags::HURRY);
    assert_eq!(decide(&inner, true), Decision::Snapshot(5));
    inner.flags.remove(PresenterFlags::HURRY);
    assert_eq!(decide(&inner, true), Decision::Wait(5), "the level cleared");
    inner.flags.insert(PresenterFlags::STOP);
    assert_eq!(
        decide(&inner, true),
        Decision::Proceed,
        "a stopping presenter drops every packet, so nothing reads the back buffer"
    );
}

#[test]
fn slots_prefer_free_then_the_oldest_reader() {
    let mut inner = empty_inner();
    assert_eq!(choose_slot(&inner), SlotChoice::Free(0), "never allocated");
    inner.slots[0] = Some(slot(5));
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(1),
        "slot 0 busy, slot 1 unallocated"
    );
    inner.slots[1] = Some(slot(6));
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(2),
        "slots 0 and 1 busy, slot 2 unallocated"
    );
    inner.slots[2] = Some(slot(7));
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(3),
        "slots 0 to 2 busy, slot 3 unallocated"
    );
    inner.slots[3] = Some(slot(8));
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(4),
        "slots 0 to 3 busy, slot 4 unallocated"
    );
    inner.slots[4] = Some(slot(9));
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Busy(0),
        "all busy: the oldest reader"
    );
    inner.committed_present_seq = 5;
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(0),
        "reader 5 committed frees slot 0"
    );
    inner.committed_present_seq = 8;
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(0),
        "the first free one wins"
    );
}

#[test]
fn a_slot_at_another_geometry_is_replaced() {
    let slot = slot(0);
    assert!(slot.matches(640, 480, MTLPixelFormat::BGRA8Unorm));
    assert!(!slot.matches(641, 480, MTLPixelFormat::BGRA8Unorm));
    assert!(!slot.matches(640, 481, MTLPixelFormat::BGRA8Unorm));
    assert!(!slot.matches(640, 480, MTLPixelFormat::RGBA16Float));
}

#[test]
fn a_dropped_packet_frees_its_slot_and_leaves_retirement_alone() {
    let state = state_with(empty_inner());
    let mut inner = state.lock();
    inner.slots[1] = Some(slot(7));
    inner.slots[2] = Some(slot(8));
    inner.slots[3] = Some(slot(9));
    inner.slots[4] = Some(slot(10));
    assert_eq!(choose_slot(&inner), SlotChoice::Free(0));
    inner.slots[0] = Some(slot(6));
    assert_eq!(choose_slot(&inner), SlotChoice::Busy(0));
    drop_packet(&mut inner, packet(7, Some(1)));
    assert_eq!(inner.committed_present_seq, 7, "consumed like a commit");
    assert_eq!(inner.presented_seq, 0, "nothing was committed to the GPU");
    assert_eq!(
        state.present_retired.load(Ordering::Acquire),
        0,
        "no present buffer carries a dropped sequence"
    );
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(0),
        "readers 6 and 7 are at or below the committed sequence"
    );
}

#[test]
fn a_drop_behind_a_committed_present_keeps_its_retirement_pending() {
    let mut inner = empty_inner();
    // Present 5 committed and is still on the GPU; the counter is behind it.
    inner.committed_present_seq = 5;
    inner.presented_seq = 5;
    inner.pending.push_back(packet(6, None));
    let state = state_with(inner);
    state.present_retired.store(4, Ordering::Release);
    drop_front(&state, 6);
    let inner = state.lock();
    assert_eq!(inner.committed_present_seq, 6, "the drop is consumed");
    assert_eq!(
        inner.presented_seq, 5,
        "an idle wait retires the last committed present"
    );
    assert_eq!(
        state.present_retired.load(Ordering::Acquire),
        4,
        "only present 5's completion handler moves the counter"
    );
}

#[test]
fn stop_wakes_a_submit_parked_in_its_wait() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, None));
    let state = state_with(inner);
    let waiter = Arc::clone(&state);
    let worker = thread::spawn(move || {
        let mut inner = waiter.lock();
        let seq = match decide(&inner, true) {
            Decision::Wait(seq) => seq,
            other => panic!("expected a wait, got {other:?}"),
        };
        inner = waiter
            .submit_cv
            .wait_while(inner, |inner| {
                inner.committed_present_seq < seq
                    && !inner
                        .flags
                        .intersects(PresenterFlags::HURRY | PresenterFlags::STOP)
            })
            .unwrap_or_else(PoisonError::into_inner);
        inner.flags
    });
    thread::sleep(Duration::from_millis(20));
    state.lock().flags.insert(PresenterFlags::STOP);
    state.submit_cv.notify_all();
    let flags = worker.join().expect("the waiter returned");
    assert!(flags.contains(PresenterFlags::STOP));
}

#[test]
fn hurry_wakes_a_submit_parked_in_its_wait() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(5, None));
    let state = state_with(inner);
    let waiter = Arc::clone(&state);
    let worker = thread::spawn(move || {
        let inner = waiter.lock();
        let inner = waiter
            .submit_cv
            .wait_while(inner, |inner| {
                inner.committed_present_seq < 5
                    && !inner
                        .flags
                        .intersects(PresenterFlags::HURRY | PresenterFlags::STOP)
            })
            .unwrap_or_else(PoisonError::into_inner);
        decide(&inner, true)
    });
    thread::sleep(Duration::from_millis(20));
    state.lock().flags.insert(PresenterFlags::HURRY);
    state.submit_cv.notify_all();
    assert_eq!(
        worker.join().expect("the waiter returned"),
        Decision::Snapshot(5),
        "a hurried waiter re-decides into a snapshot"
    );
}

#[test]
fn an_idle_wait_returns_once_the_presenter_pops_the_last_packet() {
    let mut inner = empty_inner();
    inner.pending.push_back(packet(3, None));
    let state = state_with(inner);
    let waiter = Arc::clone(&state);
    let worker = thread::spawn(move || {
        let inner = waiter.lock();
        let inner = waiter
            .submit_cv
            .wait_while(inner, |inner| {
                !inner.pending.is_empty() && !inner.flags.contains(PresenterFlags::STOP)
            })
            .unwrap_or_else(PoisonError::into_inner);
        (inner.committed_present_seq, inner.presented_seq)
    });
    thread::sleep(Duration::from_millis(20));
    drop_front(&state, 3);
    assert_eq!(
        worker.join().expect("the waiter returned"),
        (3, 0),
        "the dropped packet is consumed and leaves nothing to retire"
    );
}

#[test]
fn two_devices_keep_their_own_wait_policy() {
    let state_a = state_with(empty_inner());
    let state_b = state_with(empty_inner());
    set_wait_policy(&state_a, PresentWaitPolicy::SnapshotPending);
    assert!(state_a.lock().flags.contains(PresenterFlags::HURRY));
    assert!(
        state_b.lock().flags.is_empty(),
        "the other device is untouched"
    );
    set_wait_policy(&state_a, PresentWaitPolicy::WaitForCommit);
    assert!(state_a.lock().flags.is_empty());
}
