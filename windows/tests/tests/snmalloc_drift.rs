//! Proves snmalloc still decommits every free past its per-thread budget.
//!
//! `page_box::SNMALLOC_LOCAL_CACHE_BYTES` encodes `LocalCacheSizeBits = 21`
//! from the vendored `backend/base_constants.h`. That constant does not
//! cross the C ABI, so an snmalloc upgrade could move it with nothing in
//! the tree to notice. Its observable signature is the decommit: at or past
//! the budget, `backend_helpers/largebuddyrange.h` forwards both the alloc
//! and the free straight to `CommitRange`, which calls the Windows PAL's
//! `VirtualAlloc(MEM_COMMIT)` and `VirtualFree(MEM_DECOMMIT)`.
//!
//! This has to run on a PE target: the POSIX PAL never returns pages, so
//! the host-side twin in `mtld3d-core/tests/snmalloc_drift.rs` can only
//! check the pow2 rounding half.
//!
//! The test binary links its own snmalloc rather than reaching into
//! `d3d9.dll`, but it is built for the same target with the same features,
//! so it exercises the configuration that ships.
//!
//! # Why there is no "stays committed below the cutoff" twin
//!
//! It would be flaky, not merely redundant. Below the budget a free goes
//! into the thread-local buddy, where `add_block` may coalesce with a
//! sibling; if the merged block reaches the budget, `dealloc_overflow`
//! forwards it to the parent and it decommits after all. Whether that
//! happens depends on buddy occupancy at that moment, which the test does
//! not control. Above the budget there is no coalescing step at all, so
//! that direction is deterministic and is the one asserted here.

use core::ffi::c_void;
use std::alloc::{Layout, alloc, dealloc};

use mtld3d_core::page_box::{PAGE_SIZE, SNMALLOC_LOCAL_CACHE_BYTES, bypasses_local_cache};
use snmalloc_rs::SnMalloc;

#[global_allocator]
static ALLOCATOR: SnMalloc = SnMalloc;

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;

/// `MEMORY_BASIC_INFORMATION`, correct on both PE targets.
///
/// Pointer and `SIZE_T` fields are `usize`, so `repr(C)` lays out the
/// 28-byte i686 form and the 48-byte x64 form (with its two alignment
/// holes) without a per-target definition.
#[repr(C)]
#[derive(Default)]
struct MemoryBasicInformation {
    base_address: usize,
    allocation_base: usize,
    allocation_protect: u32,
    region_size: usize,
    state: u32,
    protect: u32,
    mem_type: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualQuery(
        address: *const c_void,
        buffer: *mut MemoryBasicInformation,
        length: usize,
    ) -> usize;
}

/// `State` field of the region containing `addr`.
///
/// Panics rather than returning a sentinel: a `VirtualQuery` that fails
/// here means the probe itself is broken, and a broken probe that returned
/// "not committed" would make the test pass for the wrong reason.
fn region_state(addr: *const u8) -> u32 {
    let mut info = MemoryBasicInformation::default();
    let size = core::mem::size_of::<MemoryBasicInformation>();
    // SAFETY: `addr` is an address in this process; `info` is a live,
    // correctly sized `MEMORY_BASIC_INFORMATION` we own.
    let written = unsafe { VirtualQuery(addr.cast(), &raw mut info, size) };
    assert_eq!(
        written, size,
        "VirtualQuery wrote {written} bytes, expected {size} — probe is broken, not the allocator"
    );
    info.state
}

#[test]
fn free_past_the_local_cache_budget_decommits() {
    // 4 MiB: over 1 MiB, so it rounds to a chunk at or past the budget and
    // takes the forward-to-parent path on both the alloc and the free.
    let size = 4 * 1024 * 1024;
    assert!(
        bypasses_local_cache(size),
        "test size must be past the {SNMALLOC_LOCAL_CACHE_BYTES}-byte budget"
    );

    let layout = Layout::from_size_align(size, PAGE_SIZE).expect("valid page-aligned layout");
    // SAFETY: non-zero size, power-of-two alignment.
    let ptr = unsafe { alloc(layout) };
    assert!(!ptr.is_null(), "allocation failed");

    // Touch the first page so the region is unambiguously live, then
    // confirm the probe reports COMMIT. Without this the post-free
    // assertion could pass against a probe that never reports COMMIT.
    // SAFETY: `ptr` owns `size` bytes and `size` is a whole page multiple.
    unsafe { ptr.write(0xab) };
    assert_eq!(
        region_state(ptr),
        MEM_COMMIT,
        "region should be committed while the allocation is live"
    );

    // SAFETY: same pointer and layout the allocation came from.
    unsafe { dealloc(ptr, layout) };

    // Nothing allocates between the free and the query, so the address is
    // still the one snmalloc just released.
    assert_eq!(
        region_state(ptr),
        MEM_RESERVE,
        "snmalloc no longer decommits at {SNMALLOC_LOCAL_CACHE_BYTES} bytes: \
         LocalCacheSizeBits has moved, so SNMALLOC_LOCAL_CACHE_BYTES and the \
         uncached-alloc counter built on it are now wrong"
    );
}
