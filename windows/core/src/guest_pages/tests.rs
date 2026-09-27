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

fn retire_pooled_lease(
    mut lease: GuestPageLease,
    cells: &crate::guest_completions::CompletionPool,
) {
    // SAFETY: this fixture retains the original allocation until the unique native owner drops.
    let native = unsafe { lease.descriptor().adopt_owned() }.expect("native owner");
    assert!(!lease.maintain());
    drop(native);
    cells.drain(
        &mut crate::guest_completions::CompletionDrain::default(),
        16,
        |_| {},
    );
    assert!(lease.maintain());
    cells.recycle(lease.into_slot().expect("pooled completion"));
}

#[test]
fn retired_vbib_returns_original_pages_and_contents_to_guest_pool() {
    // The runtime uses an existing process-lifetime pool; retain that lifetime in this fixture.
    let pool = Box::leak(Box::new(PageBoxPool::new(PAGE_SIZE * 2)));
    let mut original = PageBox::new_zeroed(12);
    // SAFETY: the original allocation is uniquely owned and the first byte is in bounds.
    unsafe { original.as_mut_ptr().write(73) };
    let address = original.as_ptr();
    assert!(pool.recycle(original).is_none());
    let original = pool.acquire(12).expect("initial pool acquire");
    let cells = crate::guest_completions::CompletionPool::new();
    let mut lease = GuestPageLease::for_recyclable_pooled(original, &cells, Some(pool));
    // SAFETY: the lease remains alive through native retirement and notification consumption.
    let native = unsafe { lease.descriptor().adopt_owned() }.expect("native owner");
    assert!(pool.acquire(12).is_none());
    assert!(!lease.maintain());
    drop(native);
    assert!(
        !lease.maintain(),
        "publication alone cannot reclaim queued completion"
    );
    cells.drain(
        &mut crate::guest_completions::CompletionDrain::default(),
        16,
        |_| {},
    );
    assert!(lease.maintain());
    cells.recycle(lease.into_slot().expect("pooled completion"));
    let reused = pool.acquire(12).expect("original guest pages returned");
    assert!(reused.is_native_owned());
    assert_eq!(reused.as_ptr(), address);
    // SAFETY: the first byte was initialized and no native consumer retains the allocation.
    assert_eq!(unsafe { reused.as_ptr().read() }, 73);
}

#[test]
fn guest_pool_return_preserves_disabled_and_capacity_filters() {
    for cap in [0, PAGE_SIZE] {
        let pool = Box::leak(Box::new(PageBoxPool::new(cap)));
        let cells = crate::guest_completions::CompletionPool::new();
        let lease = GuestPageLease::for_recyclable_pooled(
            PageBox::new_zeroed(PAGE_SIZE * 2),
            &cells,
            Some(pool),
        );
        retire_pooled_lease(lease, &cells);
        assert!(pool.acquire(PAGE_SIZE * 2).is_none());
    }
}

#[test]
fn guest_pool_return_excludes_aliases_readers_and_texture_owners() {
    let pool = Box::leak(Box::new(PageBoxPool::new(PAGE_SIZE * 8)));
    for hold_read in [false, true] {
        let cells = crate::guest_completions::CompletionPool::new();
        let lease =
            GuestPageLease::for_recyclable_pooled(PageBox::new_zeroed(12), &cells, Some(pool));
        let external = Arc::clone(&lease.owner);
        let read = hold_read.then(|| PageBoxRead::new(Arc::clone(&external)));
        retire_pooled_lease(lease, &cells);
        assert!(pool.acquire(12).is_none());
        drop(read);
        drop(external);
    }
    let cells = crate::guest_completions::CompletionPool::new();
    let texture = GuestPageLease::for_shared_pooled(Arc::new(PageBox::new_zeroed(12)), &cells);
    assert!(texture.recycle_pool.is_none());
    retire_pooled_lease(texture, &cells);
    assert!(pool.acquire(12).is_none());
}
