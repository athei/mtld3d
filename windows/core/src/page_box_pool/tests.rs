use super::{MAX_POOL_CLASSES, PageBoxPool};
use crate::page_box::{PAGE_SIZE, PageBox};

#[test]
fn disabled_pool_never_stores() {
    let pool = PageBoxPool::new(0);
    assert!(!pool.enabled());
    let pb = PageBox::new_uninit(PAGE_SIZE);
    assert!(pool.recycle(pb).is_some(), "disabled pool must reject");
    assert!(pool.acquire(PAGE_SIZE).is_none());
    assert_eq!(pool.pooled_bytes(), 0);
}

#[test]
fn exact_class_round_trip() {
    let pool = PageBoxPool::new(1024 * 1024);
    let pb = PageBox::new_uninit(3 * PAGE_SIZE);
    let ptr = pb.as_ptr();
    assert!(pool.recycle(pb).is_none(), "under cap must park");
    assert_eq!(pool.pooled_bytes(), 3 * PAGE_SIZE);

    // A different class misses.
    assert!(pool.acquire(PAGE_SIZE).is_none());
    // Same padded class hits and is retargeted, same backing pages.
    let hit = pool.acquire(2 * PAGE_SIZE + 1).expect("class hit");
    assert_eq!(hit.as_ptr(), ptr);
    assert_eq!(hit.len(), 3 * PAGE_SIZE);
    assert_eq!(hit.logical_len(), 2 * PAGE_SIZE + 1);
    assert_eq!(pool.pooled_bytes(), 0);
}

#[test]
fn cap_rejection_returns_the_box() {
    let pool = PageBoxPool::new(2 * PAGE_SIZE);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    // Third box would exceed the cap; it comes back for a plain drop.
    let reject = pool.recycle(PageBox::new_uninit(PAGE_SIZE));
    assert!(reject.is_some());
    assert_eq!(pool.pooled_bytes(), 2 * PAGE_SIZE);
}

#[test]
fn oversize_class_is_rejected() {
    let pool = PageBoxPool::new(usize::MAX);
    let jumbo = PageBox::new_uninit((MAX_POOL_CLASSES + 1) * PAGE_SIZE);
    assert!(pool.recycle(jumbo).is_some());
    assert!(pool.acquire((MAX_POOL_CLASSES + 1) * PAGE_SIZE).is_none());
}

#[test]
fn largest_class_is_accepted() {
    let pool = PageBoxPool::new(usize::MAX);
    let pb = PageBox::new_uninit(MAX_POOL_CLASSES * PAGE_SIZE);
    assert!(pool.recycle(pb).is_none());
    assert!(pool.acquire(MAX_POOL_CLASSES * PAGE_SIZE).is_some());
}

#[test]
fn lifo_returns_most_recently_parked() {
    let pool = PageBoxPool::new(usize::MAX);
    let first = PageBox::new_uninit(PAGE_SIZE);
    let second = PageBox::new_uninit(PAGE_SIZE);
    let second_ptr = second.as_ptr();
    assert!(pool.recycle(first).is_none());
    assert!(pool.recycle(second).is_none());
    let hit = pool.acquire(PAGE_SIZE).expect("hit");
    assert_eq!(hit.as_ptr(), second_ptr, "LIFO: newest box pops first");
}

#[test]
fn byte_accounting_across_mixed_classes() {
    let pool = PageBoxPool::new(usize::MAX);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.recycle(PageBox::new_uninit(4 * PAGE_SIZE)).is_none());
    assert_eq!(pool.pooled_bytes(), 5 * PAGE_SIZE);
    let _ = pool.acquire(4 * PAGE_SIZE).expect("hit");
    assert_eq!(pool.pooled_bytes(), PAGE_SIZE);
}
