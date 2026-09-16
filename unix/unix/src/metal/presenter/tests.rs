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
    })
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
fn a_dropped_packet_retires_its_sequence_and_frees_its_slot() {
    let state = state_with(empty_inner());
    let mut inner = state.lock();
    inner.slots[1] = Some(slot(7));
    inner.slots[2] = Some(slot(8));
    inner.slots[3] = Some(slot(9));
    assert_eq!(choose_slot(&inner), SlotChoice::Free(0));
    inner.slots[0] = Some(slot(6));
    assert_eq!(choose_slot(&inner), SlotChoice::Busy(0));
    drop_packet(&mut inner, &state, packet(7, Some(1)));
    assert_eq!(inner.committed_present_seq, 7);
    assert_eq!(state.present_retired.load(Ordering::Acquire), 7);
    assert_eq!(
        choose_slot(&inner),
        SlotChoice::Free(0),
        "readers 6 and 7 are at or below the committed sequence"
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
        inner.committed_present_seq
    });
    thread::sleep(Duration::from_millis(20));
    drop_front(&state, 3);
    assert_eq!(worker.join().expect("the waiter returned"), 3);
}

#[test]
fn two_queues_keep_their_own_flags_and_records() {
    let a = MetalHandle::<MTLCommandQueueKind>::NULL;
    // Distinct opaque keys: nothing dereferences a registry key.
    // SAFETY: test-only opaque values that are never dereferenced.
    let key_a = unsafe { MetalHandle::<MTLCommandQueueKind>::new(0x7a00_0010) };
    // SAFETY: as above.
    let key_b = unsafe { MetalHandle::<MTLCommandQueueKind>::new(0x7a00_0020) };
    let _ = a;
    let state_a = state_with(empty_inner());
    let state_b = state_with(empty_inner());
    {
        let mut map = PRESENTERS.lock().unwrap_or_else(PoisonError::into_inner);
        map.insert(key_a.raw(), Arc::clone(&state_a));
        map.insert(key_b.raw(), Arc::clone(&state_b));
    }
    set_wait_policy(key_a, PresentWaitPolicy::SnapshotPending);
    assert!(state_a.lock().flags.contains(PresenterFlags::HURRY));
    assert!(
        state_b.lock().flags.is_empty(),
        "the other queue is untouched"
    );
    set_wait_policy(key_a, PresentWaitPolicy::WaitForCommit);
    assert!(state_a.lock().flags.is_empty());
    {
        let mut map = PRESENTERS.lock().unwrap_or_else(PoisonError::into_inner);
        map.remove(&key_a.raw());
        assert!(
            map.contains_key(&key_b.raw()),
            "removing one leaves the other"
        );
        map.remove(&key_b.raw());
    }
}
