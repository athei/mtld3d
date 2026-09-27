use std::sync::Arc;

use mtld3d_shared::encoder_wire::{FrameSlab, WireReader};

use super::{GuestPageDescriptor, GuestPageLease};
use crate::{
    encoder_value::WireValue,
    page_box::{PAGE_SIZE, PageBox, PageBoxRead},
    page_box_pool::PageBoxPool,
};

fn round_trip(value: &GuestPageDescriptor) -> GuestPageDescriptor {
    let mut slab = FrameSlab::new();
    slab.push_record(1, |writer| value.write_wire(writer))
        .expect("encode descriptor");
    let mut reader = WireReader::new(slab.as_bytes());
    let mut record = reader
        .next_record()
        .expect("record stream")
        .expect("one record");
    let descriptor =
        GuestPageDescriptor::read_wire(&mut record.payload).expect("decode descriptor");
    assert!(record.payload.is_empty());
    descriptor
}

#[test]
fn read_handoff_never_exposes_zero_before_native_read_ends() {
    let original = Arc::new(PageBox::new_zeroed(7));
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(Arc::clone(&original)));
    assert!(original.has_readers());
    assert!(!lease.maintain());
    assert!(original.has_readers());
    let descriptor = round_trip(&lease.descriptor());
    // SAFETY: lease retains the original read and all cells, with exactly one native adoption.
    let native_read = unsafe { descriptor.adopt_read() }.expect("native read");
    assert!(original.has_readers());
    assert!(!lease.maintain());
    assert!(original.has_readers());
    let cached_wrapper = Arc::clone(native_read.backing());
    drop(native_read);
    assert!(!original.has_readers());
    assert!(!lease.maintain());
    assert_eq!(cached_wrapper.as_ptr(), original.as_ptr());
    drop(cached_wrapper);
    assert!(lease.maintain());
}

#[test]
fn cached_native_ownership_keeps_original_owner_alive() {
    let original = Arc::new(PageBox::new_zeroed(4));
    let weak = Arc::downgrade(&original);
    let mut lease = GuestPageLease::for_shared(original);
    let descriptor = lease.descriptor();
    // SAFETY: the retained lease grants one native shared owner and no concurrent mutation.
    let native = unsafe { descriptor.adopt_shared() }.expect("native owner");
    assert!(weak.upgrade().is_some());
    assert!(!lease.maintain());
    drop(native);
    assert!(lease.maintain());
    drop(lease);
    assert!(weak.upgrade().is_none());
}

#[test]
fn rejection_releases_unacquired_read_and_ownership() {
    let original = Arc::new(PageBox::new_zeroed(4));
    let weak = Arc::downgrade(&original);
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(Arc::clone(&original)));
    let descriptor = lease.descriptor();
    // SAFETY: the descriptor was not adopted, and this is its sole terminal consumer.
    unsafe { descriptor.cancel_unadopted() }.expect("cancel descriptor");
    assert!(lease.maintain());
    assert!(!original.has_readers());
    drop(original);
    drop(lease);
    assert!(weak.upgrade().is_none());
}

#[test]
fn failed_frame_admission_can_cancel_from_pe() {
    let original = Arc::new(PageBox::new_zeroed(4));
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(Arc::clone(&original)));
    // SAFETY: no descriptor reached native code.
    unsafe { lease.cancel_unadopted() };
    assert!(!original.has_readers());
    assert!(lease.maintain());
}

#[test]
fn native_pool_rejects_borrowed_guest_allocation() {
    let pool = PageBoxPool::new(PAGE_SIZE * 2);
    let mut lease = GuestPageLease::for_owned(PageBox::new_zeroed(12));
    let descriptor = lease.descriptor();
    // SAFETY: the lease is retained for the sole native ownership borrow.
    let native = unsafe { descriptor.adopt_owned() }.expect("native box");
    assert!(!native.is_native_owned());
    let returned = pool
        .recycle(native)
        .expect("guest allocations cannot enter native pool");
    assert!(!lease.maintain());
    drop(returned);
    assert!(lease.maintain());
}

#[test]
fn invalid_descriptor_does_not_publish_completion() {
    let mut lease = GuestPageLease::for_owned(PageBox::new_zeroed(4));
    let mut descriptor = lease.descriptor();
    descriptor.logical_len = descriptor.padded_len + 1;
    // SAFETY: retained cells are valid, and the invalid range is rejected without dereferencing.
    assert!(unsafe { descriptor.adopt_owned() }.is_err());
    assert!(!lease.maintain());
    // SAFETY: failed adoption never constructed a native owner.
    unsafe { lease.cancel_unadopted() };
    assert!(lease.maintain());
}
