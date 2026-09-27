use super::GuestRetirementLease;
use crate::{
    guest_completions::{CompletionDrain, CompletionPool},
    page_box::{PAGE_SIZE, PageBox},
    page_box_pool::PageBoxPool,
};

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
    let lease = GuestRetirementLease::new(original, &cells, Some(pool));
    // SAFETY: the lease remains alive through native retirement and notification consumption.
    let native = unsafe { lease.descriptor().adopt() }.expect("native owner");
    assert!(pool.acquire(12).is_none());
    assert!(!lease.completed());
    drop(native);
    assert!(
        !lease.completed(),
        "publication alone cannot reclaim queued completion"
    );
    cells.drain(
        &mut crate::guest_completions::CompletionDrain::default(),
        16,
        |_| {},
    );
    assert!(lease.completed());
    cells.recycle(lease.into_slot());
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
        let lease =
            GuestRetirementLease::new(PageBox::new_zeroed(PAGE_SIZE * 2), &cells, Some(pool));
        // SAFETY: this fixture has no native consumers or prior GPU work.
        unsafe { lease.cancel_unadopted() };
        cells.drain(&mut CompletionDrain::default(), 16, |_| {});
        cells.recycle(lease.into_slot());
        assert!(pool.acquire(PAGE_SIZE * 2).is_none());
    }
}

#[test]
fn moved_retirement_owner_keeps_only_independent_addresses_published() {
    let cells = CompletionPool::new();
    let mut page = PageBox::new_zeroed(12);
    // SAFETY: this fixture exclusively owns the initialized, nonempty allocation.
    unsafe { page.as_mut_ptr().write(91) };
    let address = page.as_ptr();
    let generation = page.generation();
    let lease = GuestRetirementLease::new(page, &cells, None);
    let descriptor = lease.descriptor();
    let mut owners = Vec::with_capacity(1);
    owners.push(lease);
    for _ in 0..256 {
        owners.push(GuestRetirementLease::new(
            PageBox::new_zeroed(12),
            &cells,
            None,
        ));
    }
    // SAFETY: the growing vector still retains the unique byte allocation and stable cell.
    let native = unsafe { descriptor.adopt() }.expect("native retirement");
    assert_eq!(native.as_ptr(), address);
    assert_eq!(native.generation(), generation);
    // SAFETY: the native guard retains the initialized first byte until it drops.
    assert_eq!(unsafe { native.as_ptr().read() }, 91);
    drop(native);
    for lease in &owners[1..] {
        // SAFETY: these descriptors were never adopted and have no prior native users.
        unsafe { lease.cancel_unadopted() };
    }
    assert!(!owners[0].completed());
    cells.drain(&mut CompletionDrain::default(), 4096, |_| {});
    for lease in owners {
        cells.recycle(lease.into_slot());
    }
}

#[test]
fn invalid_retirement_descriptor_never_acknowledges_its_owner() {
    let cells = CompletionPool::new();
    let lease = GuestRetirementLease::new(PageBox::new_zeroed(12), &cells, None);
    for field in 0..4 {
        let mut descriptor = lease.descriptor();
        match field {
            0 => descriptor.source += 1,
            1 => descriptor.padded_len = 0,
            2 => descriptor.padded_len = u64::MAX,
            _ => descriptor.completion += 1,
        }
        // SAFETY: the genuine lease remains retained; invalid extents must be rejected before use.
        assert!(unsafe { descriptor.adopt() }.is_err());
        let mut notifications = 0;
        cells.drain(&mut CompletionDrain::default(), 16, |_| notifications += 1);
        assert_eq!(notifications, 0);
        assert!(!lease.completed());
    }
    // SAFETY: no descriptor was accepted or can be adopted later, and there are no prior users.
    unsafe { lease.cancel_unadopted() };
    cells.drain(&mut CompletionDrain::default(), 16, |_| {});
    cells.recycle(lease.into_slot());
}
