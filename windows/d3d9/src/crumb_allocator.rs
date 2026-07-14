//! `cfg(mtld3d_crumb)`-gated wrapper that records PageBox-shape allocations into the crumb ring.
//!
//! Active only when `--cfg mtld3d_crumb` is set; the production
//! `#[global_allocator]` is plain `SnMalloc`. The `PageBox` filter
//! (align ≥ 16 KiB && size ≥ 16 KiB) is a zero-false-positive heuristic
//! — that combination is unique to `mtld3d_core::page_box::PageBox` in this
//! project. Probes record at the shared ring buffer so a unix-side or
//! PE-side crash handler can dump them in order.

use std::alloc::{GlobalAlloc, Layout};

use snmalloc_rs::SnMalloc;

/// 16 KiB — must match `mtld3d_core::page_box::PAGE_SIZE`.
///
/// Hardcoded to keep this module self-contained.
const PAGE_SIZE: usize = 16 * 1024;

#[inline]
const fn looks_like_pagebox(layout: Layout) -> bool {
    layout.align() >= PAGE_SIZE && layout.size() >= PAGE_SIZE
}

/// Drop-in replacement for `SnMalloc`, active only under `cfg(mtld3d_crumb)`.
///
/// Records PageBox-shape allocator events into the shared crumb.
pub struct CrumbAllocator;

// SAFETY: forwards every operation to `SnMalloc`, which is itself a
// valid `GlobalAlloc`; the only added work is breadcrumb recording,
// which has no allocator-recursion side effects (the breadcrumb mmap
// is set up once during init and stays mapped for process lifetime).
unsafe impl GlobalAlloc for CrumbAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding to SnMalloc with the same layout invariant.
        let p = unsafe { SnMalloc.alloc(layout) };
        if !p.is_null() && looks_like_pagebox(layout) {
            mtld3d_shared::crumb!("pb:alloc", layout.size() as u64, p as usize as u64);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding to SnMalloc with the same layout invariant.
        let p = unsafe { SnMalloc.alloc_zeroed(layout) };
        if !p.is_null() && looks_like_pagebox(layout) {
            mtld3d_shared::crumb!("pb:alloc0", layout.size() as u64, p as usize as u64);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if looks_like_pagebox(layout) {
            mtld3d_shared::crumb!("pb:drop", layout.size() as u64, ptr as usize as u64);
        }
        // SAFETY: forwarding to SnMalloc with the same layout invariant.
        unsafe { SnMalloc.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Realloc on a PageBox isn't exercised in this codebase, but
        // forward correctly anyway for non-PageBox callers.
        // SAFETY: forwarding to SnMalloc with the same layout invariant.
        unsafe { SnMalloc.realloc(ptr, layout, new_size) }
    }
}
