use std::sync::Arc;

use mtld3d_shared::{
    MetalHandle,
    encoder_wire::{FrameSlab, WireReader},
};

use super::{GuestQueryDescriptor, GuestQueryLease, QueryLeaseCache};
use crate::{
    encoder_value::WireValue,
    page_box::PageBox,
    visibility::{QueryStatus, RetiredVisibilityBuffer, VisibilityQueryCore, VisibilityQueryState},
};

fn descriptor(lease: &GuestQueryLease) -> GuestQueryDescriptor {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| lease.descriptor().write_wire(writer))
        .expect("encode lease");
    let mut reader = WireReader::new(slab.as_bytes());
    let mut record = reader
        .next_record()
        .expect("record stream")
        .expect("one record");
    GuestQueryDescriptor::read_wire(&mut record.payload).expect("query descriptor")
}

#[test]
fn begin_and_end_share_native_arc_identity() {
    let original = VisibilityQueryCore::new();
    let begin_lease = GuestQueryLease::new(Arc::clone(&original));
    let end_lease = GuestQueryLease::new(original);
    let mut cache = QueryLeaseCache::default();
    // SAFETY: distinct leases remain live until completion and each is adopted exactly once.
    let begin = unsafe { cache.adopt(descriptor(&begin_lease)) }.expect("begin core");
    // SAFETY: end_lease is a distinct retained publication of the same mailbox.
    let end = unsafe { cache.adopt(descriptor(&end_lease)) }.expect("end core");
    assert!(Arc::ptr_eq(&begin, &end));
    assert!(end_lease.completed());
    assert!(!begin_lease.completed());
    let mut state = VisibilityQueryState::new();
    state.push_active(&begin);
    state.remove_active(&end);
    assert_eq!(state.active_count(), 0);
    drop(begin);
    assert!(!begin_lease.completed());
    drop(end);
    assert!(begin_lease.completed());
}

#[test]
fn original_query_survives_com_owner_release_until_native_drop() {
    let original = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&original);
    let lease = GuestQueryLease::new(original);
    let mut cache = QueryLeaseCache::default();
    // SAFETY: lease is the sole guest owner and remains retained through native completion.
    let native = unsafe { cache.adopt(lease.descriptor()) }.expect("native core");
    native.mark_armed();
    assert_eq!(
        weak.upgrade().expect("guest remains live").status(),
        QueryStatus::Pending
    );
    assert!(!lease.completed());
    drop(native);
    assert!(lease.completed());
    drop(lease);
    assert!(weak.upgrade().is_none());
    cache.maintain();
    assert!(cache.cores.is_empty());
}

#[test]
fn rejection_and_malformed_descriptor_leave_no_native_owner() {
    let lease = GuestQueryLease::new(VisibilityQueryCore::new());
    let mut bad = lease.descriptor();
    bad.mailbox = 1;
    let mut cache = QueryLeaseCache::default();
    // SAFETY: the malformed address is rejected numerically before any pointer is accessed.
    assert!(unsafe { cache.adopt(bad) }.is_err());
    assert!(!lease.completed());
    // SAFETY: adoption failed without constructing a core, and no other consumer exists.
    unsafe { lease.cancel_unadopted() };
    assert!(lease.completed());
    assert!(cache.cores.is_empty());
}

#[test]
fn reissue_preserves_pending_segment_generation_checks() {
    let original = VisibilityQueryCore::new();
    let first_lease = GuestQueryLease::new(Arc::clone(&original));
    let second_lease = GuestQueryLease::new(Arc::clone(&original));
    let mut cache = QueryLeaseCache::default();
    // SAFETY: first_lease retains the query mailbox until the final native reference drops.
    let first = unsafe { cache.adopt(first_lease.descriptor()) }.expect("first core");
    let mut state = VisibilityQueryState::new();
    let mut backing = PageBox::new_zeroed(16);
    backing.as_mut_slice()[..8].copy_from_slice(&100u64.to_le_bytes());
    backing.as_mut_slice()[8..16].copy_from_slice(&7u64.to_le_bytes());
    state.install_current_buffer(RetiredVisibilityBuffer::new(backing, MetalHandle::NULL, 0));
    assert!(state.retire_current_buffer(1).is_none());
    first.begin(1, 0, (1, 1), (1, 1), 0);
    first.end(1, 1);
    state.push_pending(1, Arc::clone(&first), (0, 1), true);
    // SAFETY: second_lease is a fresh retained publication of the original mailbox.
    let second = unsafe { cache.adopt(second_lease.descriptor()) }.expect("second core");
    second.begin(1, 1, (1, 1), (1, 1), 1);
    second.end(1, 2);
    state.push_pending(1, Arc::clone(&second), (1, 2), true);
    state.intake_completed(1);
    assert_eq!(original.status(), QueryStatus::Issued);
    assert_eq!(original.get_u32(), 7);
    drop(first);
    drop(second);
    assert!(first_lease.completed());
    assert!(second_lease.completed());
}

#[test]
fn api_reissue_rejects_delayed_native_result_before_new_begin() {
    let original = VisibilityQueryCore::new();
    let lease = GuestQueryLease::new(Arc::clone(&original));
    let mut cache = QueryLeaseCache::default();
    // SAFETY: lease retains its original mailbox and completion through final native drop.
    let native = unsafe { cache.adopt(lease.descriptor()) }.expect("native core");
    let first = original.mark_armed();
    native.begin_recorded(first, 1, 0, (1, 1), (1, 1), 0);
    assert_eq!(original.mark_end_requested(), first);
    native.end_recorded(first, 1, 0);
    let mut state = VisibilityQueryState::new();
    state.push_pending(1, Arc::clone(&native), (0, 0), true);
    let second = original.mark_armed();
    state.intake_completed(1);
    assert_eq!(original.status(), QueryStatus::Pending);
    assert_eq!(original.seq_end_loaded(), 0);
    assert_eq!(original.mark_end_requested(), second);
    native.begin_recorded(second, 2, 0, (1, 1), (1, 1), 0);
    native.end_recorded(second, 2, 0);
    state.push_pending(2, Arc::clone(&native), (0, 0), true);
    state.intake_completed(2);
    assert_eq!(original.status(), QueryStatus::Issued);
    assert_eq!(original.seq_end_loaded(), 2);
    drop(native);
    assert!(lease.completed());
}
