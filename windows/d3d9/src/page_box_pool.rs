//! Process-wide instance of the `PageBox` recycle pool.
//!
//! One static rather than per-device state, on two of the accepted
//! arguments. The VB/IB lane outlives every object: it keeps pages
//! committed across the global allocator, which is process-wide by nature,
//! and a box one device retired serves the next device's renames. The byte
//! cap bounds one process-wide resource, the 32-bit address space, and the
//! texture staging lane shares it, so both lanes live under the one cap.
//! A staging lease carries the pool as a `&'static` reference because it
//! can retire, and hand over the last owner of its pages, after the
//! texture that allocated them was released. The staging lane's contents
//! still end with a device: `EncoderThread::shutdown` drains the lane once
//! native destruction has retired every lease, and a texture detached from
//! its device drops its staging instead of parking it. The drain empties
//! the whole process-wide lane, so a device that is still alive loses its
//! parked boxes too (warm pages, not correctness).
//!
//! Pops run inside D3D9 calls (`DeviceInner::alloc_pagebox_capped` and
//! texture staging creation); pushes run at lease retirement, under the
//! device's retirement lock, and at texture release or Lock-rename.

use std::sync::LazyLock;

use mtld3d_core::{config::DEFAULT_PAGEBOX_POOL_CAP_BYTES, page_box_pool::PageBoxPool};

/// The pool, at the default cap until a `Direct3DCreate9` applies `memory.pageboxPoolCapMB`.
///
/// A cap of 0 disables the pool: `acquire` never hits and `recycle`
/// hands every box back for a plain drop, restoring the
/// everything-through-the-allocator behaviour (the measured baseline
/// arm of the warm-page A/B).
pub static PAGEBOX_POOL: LazyLock<PageBoxPool> = LazyLock::new(|| {
    PageBoxPool::new(usize::try_from(DEFAULT_PAGEBOX_POOL_CAP_BYTES).unwrap_or(usize::MAX))
});
