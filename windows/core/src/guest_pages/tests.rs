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
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(original));
    let descriptor = lease.descriptor();
    // SAFETY: the retained lease grants exactly one native read handoff.
    let read = unsafe { descriptor.adopt_read() }.expect("native read");
    let native = Arc::clone(read.backing());
    assert!(!lease.maintain());
    drop(read);
    assert!(!native.has_readers());
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
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(Arc::new(PageBox::new_zeroed(12))));
    let descriptor = lease.descriptor();
    // SAFETY: the lease retains the original read and cells for one native adoption.
    let read = unsafe { descriptor.adopt_read() }.expect("native read");
    assert!(!lease.maintain());
    let native = Arc::clone(read.backing());
    drop(read);
    let native = Arc::try_unwrap(native).unwrap_or_else(|_| panic!("sole native cached owner"));
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
    let mut lease = GuestPageLease::for_read(PageBoxRead::new(Arc::new(PageBox::new_zeroed(4))));
    let mut descriptor = lease.descriptor();
    descriptor.logical_len = descriptor.padded_len + 1;
    // SAFETY: retained cells are valid, and the invalid range is rejected without dereferencing.
    assert!(unsafe { descriptor.adopt_read() }.is_err());
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
    let native = unsafe { lease.descriptor().adopt_read() }.expect("native owner");
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
fn shared_page_lease_preserves_other_owners_and_readers() {
    for hold_read in [false, true] {
        let cells = crate::guest_completions::CompletionPool::new();
        let external = Arc::new(PageBox::new_zeroed(12));
        let lease =
            GuestPageLease::for_read_pooled(PageBoxRead::new(Arc::clone(&external)), &cells, None);
        let read = hold_read.then(|| PageBoxRead::new(Arc::clone(&external)));
        retire_pooled_lease(lease, &cells);
        assert_eq!(Arc::strong_count(&external), if hold_read { 2 } else { 1 });
        assert_eq!(external.has_readers(), hold_read);
        drop(read);
        assert!(!external.has_readers());
    }
}

/// A pooled lease of `owner`'s pages that offers them to `pages` at retirement.
fn staging_lease(
    owner: &Arc<PageBox>,
    cells: &crate::guest_completions::CompletionPool,
    pages: &'static PageBoxPool,
) -> GuestPageLease {
    GuestPageLease::for_read_pooled(PageBoxRead::new(Arc::clone(owner)), cells, Some(pages))
}

#[test]
fn a_retired_lease_parks_staging_the_texture_released_before_it() {
    let cells = crate::guest_completions::CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let texture = Arc::new(PageBox::new_uninit(3 * PAGE_SIZE));
    let (address, generation) = (texture.as_ptr(), texture.generation());
    let mut lease = staging_lease(&texture, &cells, pages);
    // SAFETY: the lease retains the allocation and cells until the native read drops.
    let native = unsafe { lease.descriptor().adopt_read() }.expect("native read");
    assert!(
        !pages.recycle_staging(texture),
        "the texture's release leaves the pages with the lease"
    );
    assert!(
        pages.acquire_staging(3 * PAGE_SIZE).is_none(),
        "no create can take pages native code still reads"
    );
    assert!(!lease.maintain());
    drop(native);
    cells.drain(
        &mut crate::guest_completions::CompletionDrain::default(),
        16,
        |_| {},
    );
    assert!(lease.maintain());
    cells.recycle(lease.into_slot().expect("pooled completion"));
    assert_eq!(pages.staging_bytes(), 3 * PAGE_SIZE);
    let reused = pages
        .acquire_staging(3 * PAGE_SIZE)
        .expect("parked at retirement");
    assert_eq!(reused.as_ptr(), address);
    assert_eq!(reused.generation(), generation);
    assert!(!reused.has_readers());
}

#[test]
fn a_retired_lease_leaves_staging_the_texture_still_owns() {
    let cells = crate::guest_completions::CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let texture = Arc::new(PageBox::new_uninit(PAGE_SIZE));
    retire_pooled_lease(staging_lease(&texture, &cells, pages), &cells);
    assert_eq!(pages.staging_bytes(), 0);
    assert_eq!(Arc::strong_count(&texture), 1);
    assert!(!texture.has_readers());
    assert!(
        pages.recycle_staging(texture),
        "the texture's release parks it"
    );
    assert_eq!(pages.staging_bytes(), PAGE_SIZE);
}

#[test]
fn an_unacknowledged_lease_never_offers_its_pages() {
    let cells = crate::guest_completions::CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let texture = Arc::new(PageBox::new_uninit(PAGE_SIZE));
    let lease = staging_lease(&texture, &cells, pages);
    drop(texture);
    // Neither acknowledgment arrived, so the owner drops instead of parking.
    drop(lease.into_slot());
    assert_eq!(pages.staging_bytes(), 0);
    assert!(pages.acquire_staging(PAGE_SIZE).is_none());
}
