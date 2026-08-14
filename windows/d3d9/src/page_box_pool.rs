//! Process-wide instance of the `PageBox` recycle pool.
//!
//! One static rather than per-device state: the pool exists to keep pages
//! committed across the global allocator, which is process-wide by
//! nature, and a static lets the API thread (Lock-rename pop in
//! `DeviceInner::alloc_pagebox_capped`) and the encoder thread
//! (push in `drain_retired_resource_retention`) reach it without pointer
//! plumbing through `FrameData`.

use std::sync::LazyLock;

use mtld3d_core::page_box_pool::PageBoxPool;

/// The pool, sized from `memory.pageboxPoolCapMB` on first use.
///
/// A cap of 0 disables the pool: `acquire` never hits and `recycle`
/// hands every box back for a plain drop, restoring the
/// everything-through-the-allocator behaviour (the measured baseline
/// arm of the warm-page A/B).
pub static PAGEBOX_POOL: LazyLock<PageBoxPool> = LazyLock::new(|| {
    let cap = usize::try_from(crate::config::CONFIG.pagebox_pool_cap_bytes).unwrap_or(usize::MAX);
    PageBoxPool::new(cap)
});
