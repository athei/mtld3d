//! Shared host and PE checks of allocator rounding and the cache cutoff.
//!
//! Host tests register separately. The PE caller runs them inside its decommit
//! test so no other test can reuse an address while its state is being probed.

#[cfg(target_family = "windows")]
use std::alloc::Layout;
#[cfg(not(target_family = "windows"))]
use std::mem::MaybeUninit;

use mtld3d_core::page_box::{
    PAGE_SIZE, PageBox, SNMALLOC_LOCAL_CACHE_BYTES, bypasses_local_cache, snmalloc_chunk_size,
};
use snmalloc_rs::SnMalloc;

#[cfg(target_family = "windows")]
pub mod allocation;

const MIB: usize = 1024 * 1024;

/// Smallest logical length this file may assert on.
///
/// Above this ceiling, every supported target uses power-of-two chunks.
const MIN_ASSERTABLE: usize = 64 * 1024;

#[cfg_attr(not(target_family = "windows"), test)]
pub fn chunk_size_model_matches_the_allocator() {
    let cases = [
        MIN_ASSERTABLE + 1,
        128 * 1024 + 1,
        256 * 1024 + 1,
        512 * 1024,
        MIB,
        MIB + 1,
        176 * PAGE_SIZE,
        2 * MIB,
        3 * MIB,
        4 * MIB,
    ];
    for logical in cases {
        assert!(
            logical > MIN_ASSERTABLE,
            "case {logical} is below the floor"
        );
        let padded = PageBox::padded_len(logical);
        assert_eq!(
            observed_chunk(logical),
            snmalloc_chunk_size(padded),
            "snmalloc_chunk_size drifted from the allocator at logical={logical} padded={padded}"
        );
    }
}

/// The derived cutoff lines up with what the allocator really returns.
///
/// Just over 1 MiB must land on a chunk at or past the cache budget, and
/// exactly 1 MiB must not. If either the rounding rule or the budget moves,
/// one of these two stops holding.
#[cfg_attr(not(target_family = "windows"), test)]
pub fn cutoff_agrees_with_observed_chunks() {
    let just_under = PageBox::padded_len(MIB);
    assert!(!bypasses_local_cache(just_under));
    assert!(observed_chunk(MIB) < SNMALLOC_LOCAL_CACHE_BYTES);

    let just_over = PageBox::padded_len(MIB + 1);
    assert!(bypasses_local_cache(just_over));
    assert!(observed_chunk(MIB + 1) >= SNMALLOC_LOCAL_CACHE_BYTES);
}

/// Ordinary alignment exposes the small-to-large boundary without page rounding.
#[cfg_attr(not(target_family = "windows"), test)]
pub fn large_byte_allocations_round_to_chunks() {
    for size in [64 * 1024 + 1, 128 * 1024 + 1, 256 * 1024 + 1] {
        #[cfg(not(target_family = "windows"))]
        let bytes = vec![MaybeUninit::<u8>::uninit(); size].into_boxed_slice();
        #[cfg(target_family = "windows")]
        let bytes = allocation::SnmallocAllocation::new(
            Layout::from_size_align(size, 1).expect("valid byte layout"),
        );
        let observed = SnMalloc
            .usable_size(bytes.as_ptr().cast())
            .expect("live byte allocation");
        assert_eq!(observed, snmalloc_chunk_size(size), "size={size}");
    }
}

/// Ask snmalloc what it actually handed out for `logical` bytes.
fn observed_chunk(logical: usize) -> usize {
    #[cfg(not(target_family = "windows"))]
    let pb = PageBox::new_uninit(logical);
    #[cfg(target_family = "windows")]
    let pb = allocation::SnmallocAllocation::new(
        Layout::from_size_align(PageBox::padded_len(logical), PAGE_SIZE)
            .expect("valid page-aligned layout"),
    );
    SnMalloc
        .usable_size(pb.as_ptr())
        .expect("PageBox pointer is never null")
}
