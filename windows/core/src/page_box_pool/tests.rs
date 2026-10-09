//! Unit tests for the bounded `PageBox` recycle pool.
//!
//! The tests cover one rule each: exact padded-size class matching (a hit keeps the same backing
//! pages and is retargeted to the new logical length), LIFO order within a class, the largest
//! accepted class, and the three refusal paths (disabled pool, oversize class, byte cap) handing
//! the box back for a plain drop. A miss through `acquire_or_retire` runs the retirement and pops
//! what it parked; a hit, a disabled pool, an oversize class and a pool with no room under its
//! cap for the class never run it. The exact round trip and the mixed-class case also walk
//! `pooled_bytes` across transitions, pinning the lock-free gauge to the parked set. The cap moves
//! after construction: a pool built disabled parks once it is given a budget, and stops again
//! when the budget is taken away.
//!
//! The texture staging lane is covered the same way: a same-size box comes back with its pages
//! and generation, the two lanes never serve each other, the staging share and the shared cap
//! both refuse, the class limit applies, a box with another owner or a counted reader is never
//! parked, and draining the lane frees only staging.

use std::sync::Arc;

use super::{MAX_POOL_CLASSES, PageBoxPool, STAGING_SHARE_DIVISOR};
use crate::page_box::{PAGE_SIZE, PageBox, PageBoxRead};

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
fn cap_moves_after_construction() {
    let pool = PageBoxPool::new(0);
    pool.set_cap(1024 * 1024);
    assert!(pool.enabled());
    assert_eq!(pool.cap_bytes(), 1024 * 1024);
    let pb = PageBox::new_uninit(PAGE_SIZE);
    assert!(pool.recycle(pb).is_none(), "an enabled pool parks");
    assert!(pool.acquire(PAGE_SIZE).is_some(), "and hands the box back");

    pool.set_cap(0);
    assert!(!pool.enabled());
    let pb = PageBox::new_uninit(PAGE_SIZE);
    assert!(pool.recycle(pb).is_some(), "a disabled pool rejects again");
    assert!(pool.acquire(PAGE_SIZE).is_none());
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
fn miss_retires_then_pops_what_the_retirement_parked() {
    let pool = PageBoxPool::new(1024 * 1024);
    let finished = PageBox::new_uninit(2 * PAGE_SIZE);
    let ptr = finished.as_ptr();
    let mut finished = Some(finished);
    let hit = pool
        .acquire_or_retire(2 * PAGE_SIZE, || {
            let pb = finished.take().expect("retire runs once");
            assert!(pool.recycle(pb).is_none(), "the retirement parks the box");
        })
        .expect("the retry pops the box the retirement parked");
    assert!(finished.is_none(), "an empty pool runs the retirement");
    assert_eq!(hit.as_ptr(), ptr);
    assert_eq!(pool.pooled_bytes(), 0);
}

#[test]
fn hit_does_not_retire() {
    let pool = PageBoxPool::new(1024 * 1024);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    let mut retired = false;
    assert!(
        pool.acquire_or_retire(PAGE_SIZE, || retired = true)
            .is_some()
    );
    assert!(!retired, "a hit returns before the retirement");
}

#[test]
fn retire_is_skipped_where_no_retry_can_hit() {
    let mut retired = false;
    let disabled = PageBoxPool::new(0);
    assert!(
        disabled
            .acquire_or_retire(PAGE_SIZE, || retired = true)
            .is_none()
    );
    let pool = PageBoxPool::new(usize::MAX);
    let oversize = (MAX_POOL_CLASSES + 1) * PAGE_SIZE;
    assert!(
        pool.acquire_or_retire(oversize, || retired = true)
            .is_none()
    );
    assert!(
        !retired,
        "a disabled pool or an oversize class never retires"
    );
    // A retirement that parks nothing of the class still misses.
    assert!(
        pool.acquire_or_retire(PAGE_SIZE, || retired = true)
            .is_none()
    );
    assert!(retired);
}

#[test]
fn full_pool_does_not_retire() {
    let pool = PageBoxPool::new(2 * PAGE_SIZE);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    // The cap holds two pages, both parked in another class than the one asked for.
    let mut retired = false;
    assert!(
        pool.acquire_or_retire(2 * PAGE_SIZE, || retired = true)
            .is_none()
    );
    assert!(
        !retired,
        "with no room for the class, the miss falls through to allocate"
    );
    assert_eq!(pool.pooled_bytes(), 2 * PAGE_SIZE);
}

#[test]
fn room_for_exactly_the_class_still_retires() {
    let pool = PageBoxPool::new(3 * PAGE_SIZE);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    // One page parked, two pages of room: a two-page box still fits under the cap.
    let mut retired = false;
    assert!(
        pool.acquire_or_retire(2 * PAGE_SIZE, || retired = true)
            .is_none()
    );
    assert!(
        retired,
        "a box that fits under the cap exactly is worth a retirement"
    );
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

#[cfg(perf_tracking)]
#[test]
fn diagnostics_partition_acquire_and_recycle_outcomes() {
    let pool = PageBoxPool::new(0);
    assert!(pool.acquire(PAGE_SIZE).is_none());
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_some());
    pool.set_cap(PAGE_SIZE);
    assert!(pool.acquire(PAGE_SIZE).is_none());
    let oversized = (MAX_POOL_CLASSES + 1) * PAGE_SIZE;
    assert!(pool.acquire(oversized).is_none());
    assert!(pool.recycle(PageBox::new_uninit(oversized)).is_some());
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_some());
    assert!(pool.acquire(PAGE_SIZE).is_some());
    assert!(pool.acquire(PAGE_SIZE).is_none());
    let summary = pool.diagnostics_summary();
    assert_eq!(
        summary,
        format!(
            "pagebox-pool cumulative: hit=1 empty=2 oversize=1 disabled=1 \
         oversize_requested_bytes={oversized} largest_oversize_request={oversized} \
         recycle_parked=1 recycle_cap=1 recycle_oversize=1 recycle_disabled=1 \
         staging: pagebox-pool cumulative: hit=0 empty=0 oversize=0 disabled=0 \
         oversize_requested_bytes=0 largest_oversize_request=0 \
         recycle_parked=0 recycle_cap=0 recycle_oversize=0 recycle_disabled=0"
        )
    );
}

/// The perf summary's traffic counts the VB/IB lane's parks and both lanes' parked bytes.
#[cfg(perf_tracking)]
#[test]
fn buffer_traffic_counts_buffer_parks_and_all_parked_bytes() {
    let pool = PageBoxPool::new(usize::MAX);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.recycle(PageBox::new_uninit(2 * PAGE_SIZE)).is_none());
    assert!(pool.recycle_staging(Arc::new(PageBox::new_uninit(PAGE_SIZE))));
    let _ = pool.acquire(PAGE_SIZE).expect("hit");
    let traffic = pool.buffer_traffic();
    assert_eq!(
        traffic.recycled, 2,
        "staging parks are not the buffer lane's"
    );
    assert_eq!(traffic.recycled_bytes, 3 * PAGE_SIZE as u64);
    assert_eq!(
        traffic.parked_bytes,
        3 * PAGE_SIZE as u64,
        "what is parked now, both lanes"
    );
}

#[cfg(perf_tracking)]
#[test]
fn diagnostics_keep_concurrent_acquire_totals() {
    let pool = PageBoxPool::new(PAGE_SIZE);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..32 {
                    assert!(pool.acquire(PAGE_SIZE).is_none());
                }
            });
        }
    });
    assert!(
        pool.diagnostics_summary()
            .starts_with("pagebox-pool cumulative: hit=0 empty=128 oversize=0")
    );
}

#[test]
fn staging_reuses_a_same_size_box_with_its_pages_and_generation() {
    let pool = PageBoxPool::new(usize::MAX);
    let backing = Arc::new(PageBox::new_uninit(3 * PAGE_SIZE));
    let (ptr, generation) = (backing.as_ptr(), backing.generation());
    assert!(pool.recycle_staging(backing), "the last owner parks");
    assert_eq!(pool.staging_bytes(), 3 * PAGE_SIZE);
    assert_eq!(pool.pooled_bytes(), 3 * PAGE_SIZE);
    assert!(
        pool.acquire_staging(PAGE_SIZE).is_none(),
        "another class misses"
    );
    let hit = pool
        .acquire_staging(2 * PAGE_SIZE + 5)
        .expect("same padded class");
    assert_eq!(hit.as_ptr(), ptr);
    assert_eq!(
        hit.generation(),
        generation,
        "parked pages were never freed"
    );
    assert_eq!(hit.len(), 3 * PAGE_SIZE);
    assert_eq!(hit.logical_len(), 2 * PAGE_SIZE + 5);
    assert!(!hit.has_readers());
    assert_eq!(pool.staging_bytes(), 0);
    assert_eq!(pool.pooled_bytes(), 0);
}

#[test]
fn the_lanes_never_serve_each_other() {
    let pool = PageBoxPool::new(usize::MAX);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    assert!(pool.acquire_staging(PAGE_SIZE).is_none());
    assert!(pool.recycle_staging(Arc::new(PageBox::new_uninit(2 * PAGE_SIZE))));
    assert!(pool.acquire(2 * PAGE_SIZE).is_none());
    assert_eq!(pool.pooled_bytes(), 3 * PAGE_SIZE);
    assert!(pool.acquire(PAGE_SIZE).is_some());
    assert!(pool.acquire_staging(2 * PAGE_SIZE).is_some());
}

#[test]
fn staging_stops_at_its_share_while_buffers_keep_the_rest_of_the_cap() {
    let cap = 8 * PAGE_SIZE;
    let share = cap / STAGING_SHARE_DIVISOR;
    let pool = PageBoxPool::new(cap);
    for _ in 0..share / PAGE_SIZE {
        assert!(pool.recycle_staging(Arc::new(PageBox::new_uninit(PAGE_SIZE))));
    }
    assert!(
        !pool.recycle_staging(Arc::new(PageBox::new_uninit(PAGE_SIZE))),
        "the staging share is full"
    );
    assert_eq!(pool.staging_bytes(), share);
    for _ in 0..(cap - share) / PAGE_SIZE {
        assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    }
    assert!(
        pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_some(),
        "the shared cap is full"
    );
    assert_eq!(pool.pooled_bytes(), cap);
}

#[test]
fn staging_is_refused_once_buffers_fill_the_shared_cap() {
    let pool = PageBoxPool::new(8 * PAGE_SIZE);
    for _ in 0..8 {
        assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    }
    assert!(!pool.recycle_staging(Arc::new(PageBox::new_uninit(PAGE_SIZE))));
    assert_eq!(pool.staging_bytes(), 0);
    assert_eq!(pool.pooled_bytes(), 8 * PAGE_SIZE);
}

#[test]
fn staging_keeps_the_class_limit() {
    let pool = PageBoxPool::new(usize::MAX);
    assert!(pool.recycle_staging(Arc::new(PageBox::new_uninit(MAX_POOL_CLASSES * PAGE_SIZE))));
    assert!(pool.acquire_staging(MAX_POOL_CLASSES * PAGE_SIZE).is_some());
    let jumbo = (MAX_POOL_CLASSES + 1) * PAGE_SIZE;
    assert!(!pool.recycle_staging(Arc::new(PageBox::new_uninit(jumbo))));
    assert!(pool.acquire_staging(jumbo).is_none());
    assert_eq!(pool.staging_bytes(), 0);
}

#[test]
fn staging_with_another_owner_or_reader_is_not_parked() {
    let pool = PageBoxPool::new(usize::MAX);
    let texture = Arc::new(PageBox::new_uninit(PAGE_SIZE));
    let upload = Arc::clone(&texture);
    assert!(!pool.recycle_staging(texture), "the upload still owns it");
    assert_eq!(pool.staging_bytes(), 0);
    assert!(
        pool.acquire_staging(PAGE_SIZE).is_none(),
        "nothing to hand out"
    );
    let read = PageBoxRead::new(upload);
    assert!(read.backing().has_readers());
    assert!(!pool.recycle_staging(Arc::clone(read.backing())));
    assert!(pool.acquire_staging(PAGE_SIZE).is_none());
    let backing = Arc::clone(read.backing());
    drop(read);
    assert!(
        pool.recycle_staging(backing),
        "the reader's end frees the last owner"
    );
    assert!(pool.acquire_staging(PAGE_SIZE).is_some());
}

#[test]
fn a_disabled_pool_parks_no_staging_and_the_take_counts_nothing() {
    let pool = PageBoxPool::new(0);
    assert!(!pool.recycle_staging(Arc::new(PageBox::new_uninit(PAGE_SIZE))));
    let mut take = pool.take_staging();
    let page = take.take(PAGE_SIZE);
    assert_eq!(page.logical_len(), PAGE_SIZE);
    assert_eq!(take.finish(), (0, 0));
}

#[test]
fn the_take_counts_hits_and_misses_and_a_miss_allocates() {
    let pool = PageBoxPool::new(usize::MAX);
    let mut take = pool.take_staging();
    let first = take.take(PAGE_SIZE);
    assert_eq!(take.finish(), (0, 1));
    let ptr = first.as_ptr();
    assert!(pool.recycle_staging(Arc::new(first)));
    let mut take = pool.take_staging();
    let reused = take.take(100);
    assert_eq!(reused.as_ptr(), ptr);
    assert_eq!(reused.logical_len(), 100);
    let fresh = take.take(PAGE_SIZE);
    assert_ne!(fresh.as_ptr(), ptr);
    assert_eq!(take.finish(), (1, 1));
    assert_eq!(pool.pooled_bytes(), 0);
}

#[test]
fn one_take_serves_every_level_of_a_create_under_one_lock() {
    let pool = PageBoxPool::new(usize::MAX);
    let levels = [4 * PAGE_SIZE, 2 * PAGE_SIZE, PAGE_SIZE, PAGE_SIZE];
    let mut parked = Vec::new();
    for &len in &levels {
        let pb = PageBox::new_uninit(len);
        parked.push(pb.as_ptr());
        assert!(pool.recycle_staging(Arc::new(pb)));
    }
    let mut take = pool.take_staging();
    let mut taken: Vec<PageBox> = levels.iter().map(|&len| take.take(len)).collect();
    // A fifth level no box was parked for falls through to the allocator.
    taken.push(take.take(8 * PAGE_SIZE));
    assert_eq!(take.finish(), (4, 1));
    // The lock is free again once the take has finished.
    assert_eq!(pool.staging_bytes(), 0);
    assert_eq!(pool.pooled_bytes(), 0);
    // Same-class boxes come back last parked first.
    assert_eq!(taken[0].as_ptr(), parked[0]);
    assert_eq!(taken[1].as_ptr(), parked[1]);
    assert_eq!(taken[2].as_ptr(), parked[3]);
    assert_eq!(taken[3].as_ptr(), parked[2]);
}

#[test]
fn draining_frees_only_the_staging_lane() {
    let pool = PageBoxPool::new(usize::MAX);
    assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
    for pages in [1, 2, 2] {
        assert!(pool.recycle_staging(Arc::new(PageBox::new_uninit(pages * PAGE_SIZE))));
    }
    assert_eq!(pool.drain_staging(), 5 * PAGE_SIZE);
    assert_eq!(pool.staging_bytes(), 0);
    assert_eq!(pool.pooled_bytes(), PAGE_SIZE);
    assert!(pool.acquire_staging(2 * PAGE_SIZE).is_none());
    assert!(
        pool.acquire(PAGE_SIZE).is_some(),
        "the buffer lane is untouched"
    );
    assert_eq!(pool.drain_staging(), 0);
}
