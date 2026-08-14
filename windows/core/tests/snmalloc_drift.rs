//! Guards `page_box`'s model of snmalloc against the real allocator.
//!
//! `snmalloc_chunk_size` and `bypasses_local_cache` encode two facts read
//! out of the vendored C++: large requests round up to the next power of
//! two (`mem/sizeclasstable.h`), and the per-thread large-object cache
//! budget is 2 MiB (`backend/base_constants.h`, `LocalCacheSizeBits = 21`).
//! Neither crosses the C ABI, so an upgrade could move either one with
//! nothing to notice. This binary installs snmalloc as its own global
//! allocator and asserts the rounding half against `usable_size`.
//!
//! # What this does not cover
//!
//! The 2 MiB budget itself is not observable here. Its behavioural
//! signature is the decommit-on-free, and only the Windows PAL returns
//! pages; the POSIX one is a no-op. That half is covered on the PE side by
//! `mtld3d-tests`.
//!
//! # Why nothing below 512 KiB is asserted
//!
//! The small/large boundary is not the same on every target. snmalloc-sys
//! 0.7.4's `build_cc` path emits `-DSNMALLOC_QEMU_WORKAROUND=OFF`, which
//! *defines* the macro, and `ds/allocconfig.h` only tests `defined(...)`,
//! so every 64-bit build takes a branch meant for QEMU CI: the small-class
//! ceiling is 512 KiB there against 64 KiB on i686. Above 512 KiB every
//! configuration agrees the allocation is large and pow2-rounded, so the
//! assertions live there and hold on the host and on both PE targets.

use mtld3d_core::page_box::{
    PAGE_SIZE, PageBox, SNMALLOC_LOCAL_CACHE_BYTES, bypasses_local_cache, snmalloc_chunk_size,
};
use snmalloc_rs::SnMalloc;

#[global_allocator]
static ALLOCATOR: SnMalloc = SnMalloc;

const MIB: usize = 1024 * 1024;

/// Smallest logical length this file may assert on.
///
/// See the module docs: below it the small/large split differs per target.
const MIN_ASSERTABLE: usize = 512 * 1024;

/// Ask snmalloc what it actually handed out for `logical` bytes.
fn observed_chunk(logical: usize) -> usize {
    let pb = PageBox::new_uninit(logical);
    SnMalloc
        .usable_size(pb.as_ptr())
        .expect("PageBox pointer is never null")
}

#[test]
fn chunk_size_model_matches_the_allocator() {
    let cases = [
        MIN_ASSERTABLE + 1,
        MIB,
        MIB + 1,
        // The measured dominant renamed VB: 176 pages.
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
#[test]
fn cutoff_agrees_with_observed_chunks() {
    let just_under = PageBox::padded_len(MIB);
    assert!(!bypasses_local_cache(just_under));
    assert!(observed_chunk(MIB) < SNMALLOC_LOCAL_CACHE_BYTES);

    let just_over = PageBox::padded_len(MIB + 1);
    assert!(bypasses_local_cache(just_over));
    assert!(observed_chunk(MIB + 1) >= SNMALLOC_LOCAL_CACHE_BYTES);
}
