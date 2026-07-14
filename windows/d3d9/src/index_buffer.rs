//! `IDirect3DIndexBuffer9` COM wrapper.
//!
//! Mirrors `vertex_buffer.rs`; the vtable type, the `fvf` → `format` field
//! swap, the reported resource type, and priority handling differ — the index
//! buffer's `SetPriority` / `GetPriority` are no-ops rather than round-tripping
//! an app-set value.

use core::ffi::c_void;
use std::sync::atomic::Ordering;

use log::trace;
use mtld3d_core::{
    buffer_rename::{BufferMapMode, LockPlan, PreserveKind, classify_map_mode, plan_lock},
    dirty_range::DirtyRange,
    ids::BufferId,
    page_box::PageBox,
};
use mtld3d_shared::{InPtr, InPtrMut};
use mtld3d_types::{
    D3DINDEXBUFFER_DESC, D3DLOCK_DISCARD, D3DLOCK_KNOWN_BITS, D3DLOCK_NOOVERWRITE,
    D3DLOCK_READONLY, D3DRTYPE_INDEXBUFFER, Guid, IDirect3DIndexBuffer9Vtbl,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET, com_ref::ComUnknown,
    device::DeviceInner, null_out,
};

static DIRECT3D_INDEX_BUFFER9_VTBL: IDirect3DIndexBuffer9Vtbl = IDirect3DIndexBuffer9Vtbl {
    query_interface: ib_query_interface,
    add_ref: ib_add_ref,
    release: ib_release,
    get_device: ib_get_device,
    set_private_data: ib_set_private_data,
    get_private_data: ib_get_private_data,
    free_private_data: ib_free_private_data,
    set_priority: ib_set_priority,
    get_priority: ib_get_priority,
    pre_load: ib_pre_load,
    get_type: ib_get_type,
    lock: ib_lock,
    unlock: ib_unlock,
    get_desc: ib_get_desc,
};

#[repr(C)]
pub struct Direct3DIndexBuffer9 {
    vtbl: *const IDirect3DIndexBuffer9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut IndexBufferInner,
}

pub struct IndexBufferInner {
    device_inner: *mut DeviceInner,
    buffer_id: BufferId,
    length: u32,
    usage: u32,
    format: u32,
    pool: u32,
    /// `Direct` (zero-copy) vs `Staged` (separate device buffer + dirty-range upload).
    ///
    /// Decided once at creation from `usage`/`pool`.
    map_mode: BufferMapMode,
    /// `Staged` only: byte range dirtied since the last upload.
    dirty: DirtyRange,
    current_box: PageBox,
    last_submit_seq: u64,
    locked: bool,
}

impl IndexBufferInner {
    pub const fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    /// Upload a still-mapped `Staged` buffer's dirty span without ending the lock.
    ///
    /// So an indexed draw issued while the buffer is mapped reads the latest
    /// CPU writes, per D3D9 behaviour. `dirty` is left set so `Unlock` (and any
    /// later draw) re-flushes. No-op unless locked + `Staged` + dirty. Mirrors
    /// `ib_unlock`'s upload, minus the clear.
    pub fn flush_staged_if_mapped(&mut self, dev: &mut DeviceInner) {
        if !self.locked || !matches!(self.map_mode, BufferMapMode::Staged) {
            return;
        }
        let Some((min, max)) = self.dirty.span() else {
            return;
        };
        let size = (max - min) as usize;
        let mut transient = dev.alloc_pagebox_with_recovery(size);
        // SAFETY: `min <= length` and `current_box` is allocated for `length`
        // bytes, so the offset stays in-bounds.
        let src = unsafe { self.current_box.as_ptr().add(min as usize) };
        // SAFETY: `src` spans `[min, max)` of `current_box`; `transient` is a
        // fresh `PageBox` of ≥ `size` bytes; the two allocations are disjoint.
        unsafe {
            core::ptr::copy_nonoverlapping(src, transient.as_mut_ptr(), size);
        }
        dev.push_stage_upload(self.buffer_id, transient, min, max - min);
    }

    pub const fn map_mode(&self) -> BufferMapMode {
        self.map_mode
    }

    pub const fn format(&self) -> u32 {
        self.format
    }

    pub fn current_backing_ptr(&self) -> u64 {
        self.current_box.as_ptr() as u64
    }

    pub const fn current_backing_len(&self) -> u64 {
        self.current_box.len() as u64
    }

    pub const fn stamp_submit_seq(&mut self, seq: u64) {
        if seq > self.last_submit_seq {
            self.last_submit_seq = seq;
        }
    }
}

pub struct IndexBufferCreateInfo {
    pub device_inner: *mut DeviceInner,
    pub length: u32,
    pub usage: u32,
    pub format: u32,
    pub pool: u32,
}

impl Direct3DIndexBuffer9 {
    pub fn new(info: &IndexBufferCreateInfo) -> Self {
        let current_box = PageBox::new_uninit(info.length as usize);
        let inner = Box::into_raw(Box::new(IndexBufferInner {
            device_inner: info.device_inner,
            buffer_id: BufferId::new_unique(),
            length: info.length,
            usage: info.usage,
            format: info.format,
            pool: info.pool,
            map_mode: classify_map_mode(info.usage, info.pool),
            dirty: DirtyRange::empty(),
            current_box,
            last_submit_seq: 0,
            locked: false,
        }));
        Self {
            vtbl: &raw const DIRECT3D_INDEX_BUFFER9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    pub const fn vtbl(&self) -> &IDirect3DIndexBuffer9Vtbl {
        // SAFETY: `self.vtbl` is the `'static`
        // `DIRECT3D_INDEX_BUFFER9_VTBL` installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn inner(&self) -> &IndexBufferInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `ib_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }

    pub fn inner_mut(&mut self) -> &mut IndexBufferInner {
        // SAFETY: see [`Self::inner`] — same `Box::into_raw` lifetime
        // contract; `&mut self` guarantees exclusive access.
        unsafe { &mut *self.inner }
    }
}

// ── IUnknown ──

#[inline]
fn ib_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DIndexBuffer9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::IndexBuffer)
}

extern "system" fn ib_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable in-param; `riid` is *const Guid per IUnknown::QueryInterface ABI.
    let riid_lo = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).map_or(0, |g| g.data1);
    trace!(target: LOG_TARGET, "IDirect3DIndexBuffer9::QueryInterface(riid_lo={riid_lo:#010x})");
    null_out(ppv);
    E_NOINTERFACE
}

extern "system" fn ib_add_ref(this: *mut c_void) -> u32 {
    let _timer = ib_timer(this);
    // SAFETY: IDirect3DIndexBuffer9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DIndexBuffer9>(this) }
}

extern "system" fn ib_release(this: *mut c_void) -> u32 {
    let _timer = ib_timer(this);
    // SAFETY: IDirect3DIndexBuffer9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DIndexBuffer9>(this) }
}

/// Destroy a `Direct3DIndexBuffer9` wrapper once `refcount` and `private_refcount` reach zero.
///
/// Hands the current backing `PageBox` off to the device's retention
/// pipeline so any in-flight GPU reads see live memory until the matching
/// submit retires.
///
/// # Safety
/// `this` must point to a live `Direct3DIndexBuffer9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_index_buffer(this: *mut Direct3DIndexBuffer9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(IndexBufferInner)` from `Self::new` and no
    // other reference can survive.
    let inner_box = unsafe { Box::from_raw(inner_ptr) };
    let IndexBufferInner {
        device_inner,
        buffer_id,
        current_box,
        last_submit_seq,
        ..
    } = *inner_box;
    if !device_inner.is_null() {
        // SAFETY: `device_inner` was stamped at `Self::new` from a
        // live `DeviceInner`; the device outlives all its child
        // resources per D3D9 lifetime rules.
        let dev = unsafe { &mut *device_inner };
        dev.queue_vbib_retention(buffer_id, current_box, last_submit_seq);
    }
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DIndexBuffer9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DIndexBuffer9 {
    fn vtbl_add_ref(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().add_ref
    }
    fn vtbl_release(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().release
    }
    fn private_refcount_inc(&mut self) {
        self.private_refcount += 1;
    }
    unsafe fn private_refcount_dec_maybe_finalize(this: *mut Self) {
        // SAFETY: caller asserts `this` points to a live wrapper with
        // at least one private refcount outstanding.
        let obj = unsafe { &mut *this };
        obj.private_refcount -= 1;
        if obj.refcount == 0 && obj.private_refcount == 0 {
            // SAFETY: both counters reached zero — no other reference
            // can survive; finalize takes exclusive ownership.
            unsafe { finalize_index_buffer(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DIndexBuffer9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn private_refcount(&self) -> u32 {
        self.private_refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        crate::device::device_wrapper_from(self.inner().device_inner)
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — both counters are zero.
        unsafe { finalize_index_buffer(this) };
    }
}

// ── IDirect3DResource9 ──

extern "system" fn ib_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DIndexBuffer9::GetDevice → INVALIDCALL");
    null_out(device);
    D3DERR_INVALIDCALL
}

extern "system" fn ib_set_private_data(
    this: *mut c_void,
    _guid: *const Guid,
    _data: *const c_void,
    _size: u32,
    _flags: u32,
) -> i32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DIndexBuffer9::SetPrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn ib_get_private_data(
    this: *mut c_void,
    _guid: *const Guid,
    _data: *mut c_void,
    _size: *mut u32,
) -> i32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DIndexBuffer9::GetPrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn ib_free_private_data(this: *mut c_void, _guid: *const Guid) -> i32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DIndexBuffer9::FreePrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn ib_set_priority(this: *mut c_void, _priority: u32) -> u32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DIndexBuffer9::SetPriority: no Metal analog, no-op"
    );
    0
}

extern "system" fn ib_get_priority(this: *mut c_void) -> u32 {
    let _timer = ib_timer(this);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DIndexBuffer9::GetPriority: no Metal analog, no-op"
    );
    0
}

extern "system" fn ib_pre_load(this: *mut c_void) {
    let _timer = ib_timer(this);
    // See IDirect3DTexture9::PreLoad — Metal has no resident-set hint.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DIndexBuffer9::PreLoad: no Metal analog, no-op"
    );
}

extern "system" fn ib_get_type(this: *mut c_void) -> u32 {
    let _timer = ib_timer(this);
    D3DRTYPE_INDEXBUFFER
}

// ── IDirect3DIndexBuffer9 ──

extern "system" fn ib_lock(
    this: *mut c_void,
    offset_to_lock: u32,
    size_to_lock: u32,
    pp_data: *mut *mut c_void,
    flags: u32,
) -> i32 {
    let _timer = ib_timer(this);
    if pp_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner_mut();
    if offset_to_lock > inner.length {
        // SAFETY: `pp_data` is non-null (checked above) and per the D3D9
        // ABI points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *pp_data = core::ptr::null_mut() };
        return D3DERR_INVALIDCALL;
    }
    if size_to_lock != 0 && offset_to_lock.saturating_add(size_to_lock) > inner.length {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ib_lock: clamping out-of-range range (off={offset_to_lock}, size={size_to_lock}, len={})",
            inner.length
        );
    }

    if matches!(inner.map_mode, BufferMapMode::Staged) {
        // Separate CPU staging: record the dirtied range for the Unlock
        // upload. No rename / no `plan_lock` — the GPU reads a distinct
        // device buffer, so a partial write can't race an in-flight draw.
        // READONLY contributes nothing (the game promises not to write).
        if flags & D3DLOCK_READONLY == 0 {
            if flags & D3DLOCK_DISCARD != 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "ib_lock: D3DLOCK_DISCARD on a non-DYNAMIC (Staged) buffer — treating as a normal dirtied-range upload");
            }
            inner
                .dirty
                .conjoin(offset_to_lock, size_to_lock, inner.length);
        }
    }

    let bypass_rename = flags & (D3DLOCK_NOOVERWRITE | D3DLOCK_READONLY) != 0;
    if matches!(inner.map_mode, BufferMapMode::Direct)
        && !bypass_rename
        && inner.device_inner.is_null()
    {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_lock: device_inner null on rename path");
    }

    if matches!(inner.map_mode, BufferMapMode::Direct) && !inner.device_inner.is_null() {
        // SAFETY: `inner.device_inner` was stamped at `Self::new` from a
        // live `DeviceInner`; the device outlives all its child
        // resources per D3D9 lifetime rules.
        let dev = unsafe { &mut *inner.device_inner };
        let coh = dev.coherent_seq_arc().load(Ordering::Acquire);
        match plan_lock(
            flags,
            inner.usage,
            inner.length,
            offset_to_lock,
            size_to_lock,
            inner.last_submit_seq,
            coh,
        ) {
            LockPlan::Rename { preserve } => {
                let buffer_id = inner.buffer_id;
                let old_seq = inner.last_submit_seq;
                let logical_len = inner.length as usize;
                let fresh = dev.alloc_pagebox_with_recovery(logical_len);
                let old_box = core::mem::replace(&mut inner.current_box, fresh);
                match preserve {
                    PreserveKind::None => {
                        // Rename without preserve. Either explicit DISCARD
                        // or whole-buffer WRITEONLY contended.
                        dev.perf_mut().bump_ib_discard();
                    }
                    PreserveKind::Cpu => {
                        // Whole-buffer non-WRITEONLY contended: the game
                        // might read the whole buffer through the Lock
                        // pointer, so carry the old bytes across
                        // synchronously.
                        dev.perf_mut().bump_vbib_preserve_cpu();
                        // SAFETY: both `old_box` and `inner.current_box` are
                        // freshly allocated `PageBox`es of `logical_len`
                        // bytes; the two allocations don't alias.
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                old_box.as_ptr(),
                                inner.current_box.as_mut_ptr(),
                                logical_len,
                            );
                        }
                    }
                }
                dev.perf_mut().bump_ib_rename();
                dev.queue_vbib_retention(buffer_id, old_box, old_seq);
                inner.last_submit_seq = 0;
            }
            LockPlan::WriteInPlace => {}
        }
    }

    inner.locked = true;

    let unknown = flags & !D3DLOCK_KNOWN_BITS;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_lock: unrecognised D3DLOCK bits {unknown:#x} ignored");
    }
    // SAFETY: `offset_to_lock <= inner.length` (checked above) and
    // `inner.current_box` is allocated for `inner.length` bytes, so the
    // pointer arithmetic stays within the allocation.
    let ptr = unsafe { inner.current_box.as_mut_ptr().add(offset_to_lock as usize) };
    // SAFETY: `pp_data` is non-null (checked above) and per the D3D9
    // ABI points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *pp_data = ptr.cast::<c_void>() };
    D3D_OK
}

extern "system" fn ib_unlock(this: *mut c_void) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner_mut();
    if !inner.locked {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_unlock: Unlock without matching Lock → S_OK");
    }
    inner.locked = false;
    if matches!(inner.map_mode, BufferMapMode::Staged)
        && let Some((min, max)) = inner.dirty.span()
    {
        if !inner.device_inner.is_null() {
            // SAFETY: `inner.device_inner` was stamped at `Self::new` from
            // a live `DeviceInner` that outlives its children.
            let dev = unsafe { &mut *inner.device_inner };
            let size = (max - min) as usize;
            let mut transient = dev.alloc_pagebox_with_recovery(size);
            // SAFETY: `min <= length` and `current_box` is allocated for
            // `length` bytes, so the offset stays in-bounds.
            let src = unsafe { inner.current_box.as_ptr().add(min as usize) };
            // SAFETY: `src` spans `[min, max)` of `current_box`;
            // `transient` is a fresh `PageBox` of ≥ `size` bytes; the two
            // allocations are disjoint.
            unsafe {
                core::ptr::copy_nonoverlapping(src, transient.as_mut_ptr(), size);
            }
            // Push the upload as an inline op so the encoder sees it in
            // draw order (for rename-at-overlap). No Metal thunk here.
            dev.push_stage_upload(inner.buffer_id, transient, min, max - min);
        }
        inner.dirty.clear();
    }
    D3D_OK
}

extern "system" fn ib_get_desc(this: *mut c_void, desc: *mut D3DINDEXBUFFER_DESC) -> i32 {
    let _timer = ib_timer(this);
    if desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    // SAFETY: `desc` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DINDEXBUFFER_DESC` slot owned by the caller.
    unsafe {
        *desc = D3DINDEXBUFFER_DESC {
            format: inner.format,
            resource_type: D3DRTYPE_INDEXBUFFER,
            usage: inner.usage,
            pool: inner.pool,
            size: inner.length,
        };
    }
    D3D_OK
}
