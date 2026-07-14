//! `IDirect3DVertexBuffer9` COM wrapper.
//!
//! Per-buffer `PageBox` backing: every Lock that renames
//! (`!NOOVERWRITE && !READONLY && last_submit_seq > coherent_seq`) swaps
//! `current_box` for a fresh uninit `PageBox`; the old one goes onto the
//! device's retention pipeline, picked up by the encoder and paired with
//! its wrapped `MTLBuffer` for GPU-retirement-gated destruction. Unlock is a
//! no-op for `Direct` buffers; `Staged` buffers upload their dirty span there.
//!
//! Layout follows the "state on Inner" pattern: the `#[repr(C)]` outer
//! struct only carries the vtable, refcount, and an opaque pointer to
//! the real state; everything else lives on `VertexBufferInner`.

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
    D3DFMT_VERTEXDATA, D3DLOCK_DISCARD, D3DLOCK_KNOWN_BITS, D3DLOCK_NOOVERWRITE, D3DLOCK_READONLY,
    D3DRTYPE_VERTEXBUFFER, D3DVERTEXBUFFER_DESC, Guid, IDirect3DVertexBuffer9Vtbl,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET, com_ref::ComUnknown,
    device::DeviceInner, null_out,
};

static DIRECT3D_VERTEX_BUFFER9_VTBL: IDirect3DVertexBuffer9Vtbl = IDirect3DVertexBuffer9Vtbl {
    query_interface: vb_query_interface,
    add_ref: vb_add_ref,
    release: vb_release,
    get_device: vb_get_device,
    set_private_data: vb_set_private_data,
    get_private_data: vb_get_private_data,
    free_private_data: vb_free_private_data,
    set_priority: vb_set_priority,
    get_priority: vb_get_priority,
    pre_load: vb_pre_load,
    get_type: vb_get_type,
    lock: vb_lock,
    unlock: vb_unlock,
    get_desc: vb_get_desc,
};

#[repr(C)]
pub struct Direct3DVertexBuffer9 {
    vtbl: *const IDirect3DVertexBuffer9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut VertexBufferInner,
}

pub struct VertexBufferInner {
    device_inner: *mut DeviceInner,
    buffer_id: BufferId,
    length: u32,
    usage: u32,
    fvf: u32,
    pool: u32,
    /// `Direct` (zero-copy `bytesNoCopy`) vs `Staged` (separate CPU staging + GPU device buffer).
    ///
    /// `Staged` does a dirty-range upload on Unlock. Decided once at
    /// creation from `usage`/`pool`; selects the `Lock`/`Unlock` path
    /// below.
    map_mode: BufferMapMode,
    /// `Staged` only: the byte range dirtied since the last upload.
    ///
    /// Accumulated across `Lock`s and flushed on `Unlock`. Unused for
    /// `Direct`.
    dirty: DirtyRange,
    /// Canonical CPU backing.
    ///
    /// For `Direct`, the GPU reads this directly and rename swaps it out
    /// onto the retention pipeline. For `Staged`, this is pure CPU
    /// staging — the game writes it; `Unlock` snapshots the dirty range up
    /// to the device buffer.
    current_box: PageBox,
    /// Submit seq of the most recent frame that drew from this buffer.
    ///
    /// Stamped at Draw snapshot time; read by `lock` to decide whether
    /// a rename is needed.
    last_submit_seq: u64,
    /// Lock/Unlock pairing sanity.
    ///
    /// Non-fatal mismatches are logged once via `log_once_warn!`.
    locked: bool,
    /// App-set managed-resource priority, round-tripped by `GetPriority` / `SetPriority`.
    ///
    /// D3D9 only honours priority for `D3DPOOL_MANAGED` buffers (it drives
    /// the resource manager's eviction order); for every other pool both
    /// accessors are fixed at `0`. Metal has no eviction-order hint, so
    /// this is app-visible state only and never acted upon.
    priority: u32,
}

impl VertexBufferInner {
    pub const fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    /// Upload a still-mapped `Staged` buffer's dirty span without ending the lock.
    ///
    /// A draw issued while the buffer is mapped reads the latest CPU writes,
    /// per the D3D9 buffer-mapping model. `dirty` is left set, so `Unlock`
    /// (and any later draw) re-flushes whatever the app writes next. No-op
    /// unless locked + `Staged` + dirty. Mirrors `vb_unlock`'s upload, minus
    /// the clear.
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

    pub fn current_backing_ptr(&self) -> u64 {
        self.current_box.as_ptr() as u64
    }

    pub const fn current_backing_len(&self) -> u64 {
        self.current_box.len() as u64
    }

    /// Stamp the current frame's submit seq onto the buffer.
    ///
    /// Called from Draw snapshot on the API thread so the retention
    /// pipeline knows this backing is live until that seq retires.
    pub const fn stamp_submit_seq(&mut self, seq: u64) {
        if seq > self.last_submit_seq {
            self.last_submit_seq = seq;
        }
    }
}

pub struct VertexBufferCreateInfo {
    pub device_inner: *mut DeviceInner,
    pub length: u32,
    pub usage: u32,
    pub fvf: u32,
    pub pool: u32,
}

impl Direct3DVertexBuffer9 {
    pub fn new(info: &VertexBufferCreateInfo) -> Self {
        let current_box = PageBox::new_uninit(info.length as usize);
        let inner = Box::into_raw(Box::new(VertexBufferInner {
            device_inner: info.device_inner,
            buffer_id: BufferId::new_unique(),
            length: info.length,
            usage: info.usage,
            fvf: info.fvf,
            pool: info.pool,
            map_mode: classify_map_mode(info.usage, info.pool),
            dirty: DirtyRange::empty(),
            current_box,
            last_submit_seq: 0,
            locked: false,
            priority: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_VERTEX_BUFFER9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    pub const fn vtbl(&self) -> &IDirect3DVertexBuffer9Vtbl {
        // SAFETY: `self.vtbl` is the `'static`
        // `DIRECT3D_VERTEX_BUFFER9_VTBL` installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn inner(&self) -> &VertexBufferInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `vb_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }

    pub fn inner_mut(&mut self) -> &mut VertexBufferInner {
        // SAFETY: see [`Self::inner`] — same `Box::into_raw` lifetime
        // contract; `&mut self` guarantees exclusive access.
        unsafe { &mut *self.inner }
    }
}

// ── IUnknown ──

#[inline]
fn vb_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DVertexBuffer9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::VertexBuffer)
}

extern "system" fn vb_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = vb_timer(this);
    // SAFETY: vtable in-param; `riid` is *const Guid per IUnknown::QueryInterface ABI.
    let riid_lo = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).map_or(0, |g| g.data1);
    trace!(target: LOG_TARGET, "IDirect3DVertexBuffer9::QueryInterface(riid_lo={riid_lo:#010x})");
    null_out(ppv);
    E_NOINTERFACE
}

extern "system" fn vb_add_ref(this: *mut c_void) -> u32 {
    let _timer = vb_timer(this);
    // SAFETY: IDirect3DVertexBuffer9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DVertexBuffer9>(this) }
}

extern "system" fn vb_release(this: *mut c_void) -> u32 {
    let _timer = vb_timer(this);
    // SAFETY: IDirect3DVertexBuffer9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DVertexBuffer9>(this) }
}

/// Destroy a `Direct3DVertexBuffer9` wrapper.
///
/// Called once both `refcount` and `private_refcount` have reached zero.
/// Hands the current backing `PageBox` off to the device's retention
/// pipeline so any in-flight GPU reads see live memory until the matching
/// submit retires.
///
/// # Safety
///
/// `this` must point to a live `Direct3DVertexBuffer9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_vertex_buffer(this: *mut Direct3DVertexBuffer9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // Take ownership of the inner on the API thread; its state has
    // to survive transit into the encoder-thread retention closure.
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(VertexBufferInner)` from `Self::new` and no
    // other reference can survive.
    let inner_box = unsafe { Box::from_raw(inner_ptr) };
    let VertexBufferInner {
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
    // `Box::into_raw(Direct3DVertexBuffer9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DVertexBuffer9 {
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
            unsafe { finalize_vertex_buffer(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DVertexBuffer9 {
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
        unsafe { finalize_vertex_buffer(this) };
    }
}

// ── IDirect3DResource9 ──

extern "system" fn vb_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = vb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexBuffer9::GetDevice → INVALIDCALL");
    null_out(device);
    D3DERR_INVALIDCALL
}

extern "system" fn vb_set_private_data(
    this: *mut c_void,
    _guid: *const Guid,
    _data: *const c_void,
    _size: u32,
    _flags: u32,
) -> i32 {
    let _timer = vb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexBuffer9::SetPrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn vb_get_private_data(
    this: *mut c_void,
    _guid: *const Guid,
    _data: *mut c_void,
    _size: *mut u32,
) -> i32 {
    let _timer = vb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexBuffer9::GetPrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn vb_free_private_data(this: *mut c_void, _guid: *const Guid) -> i32 {
    let _timer = vb_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexBuffer9::FreePrivateData → INVALIDCALL");
    D3DERR_INVALIDCALL
}

// Priority is honoured only for `D3DPOOL_MANAGED` resources (D3D9 manager
// eviction order). For every other pool both accessors are fixed at `0`.
// Metal has no eviction-order hint, so the value is stored and round-tripped
// but never acted upon.
extern "system" fn vb_set_priority(this: *mut c_void, priority: u32) -> u32 {
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVertexBuffer9>::opt(this) }) else {
        return 0;
    };
    let inner = obj.inner_mut();
    if inner.pool != mtld3d_types::D3DPOOL_MANAGED {
        return 0;
    }
    core::mem::replace(&mut inner.priority, priority)
}

extern "system" fn vb_get_priority(this: *mut c_void) -> u32 {
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DVertexBuffer9>::opt(this) }) else {
        return 0;
    };
    obj.inner().priority
}

extern "system" fn vb_pre_load(this: *mut c_void) {
    let _timer = vb_timer(this);
    // See IDirect3DTexture9::PreLoad — Metal has no resident-set hint.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DVertexBuffer9::PreLoad: no Metal analog, no-op"
    );
}

extern "system" fn vb_get_type(this: *mut c_void) -> u32 {
    let _timer = vb_timer(this);
    D3DRTYPE_VERTEXBUFFER
}

// ── IDirect3DVertexBuffer9 ──

/// Honor the D3DLOCK flags per `buffer_rename::plan_lock`:
///
/// - `NOOVERWRITE | READONLY`, or uncontended: return the existing
///   backing pointer.
/// - Contended `DISCARD`: swap `current_box` for a fresh uninit
///   `PageBox`; old goes to seq-gated retention.
/// - Contended whole-buffer (any flag combo): same swap. If the
///   buffer is non-WRITEONLY, memcpy the old bytes across (game
///   might read the whole buffer through the Lock pointer).
/// - Contended partial non-DISCARD, `D3DUSAGE_DYNAMIC`: `WriteInPlace`.
///   The game opted into the DISCARD/NOOVERWRITE timing contract — the
///   same one non-persistent mapped-buffer APIs (e.g. OpenGL
///   `glBufferSubData`) make implicitly.
/// - Contended partial non-DISCARD, non-DYNAMIC: same fresh-`PageBox`
///   swap, with the old bytes carried across via synchronous CPU
///   memcpy (a "static" buffer repacked while a draw is in flight).
extern "system" fn vb_lock(
    this: *mut c_void,
    offset_to_lock: u32,
    size_to_lock: u32,
    pp_data: *mut *mut c_void,
    flags: u32,
) -> i32 {
    let _timer = vb_timer(this);
    if pp_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVertexBuffer9>::opt(this) }) else {
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
            "vb_lock: clamping out-of-range range (off={offset_to_lock}, size={size_to_lock}, len={})",
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
                    "vb_lock: D3DLOCK_DISCARD on a non-DYNAMIC (Staged) buffer — treating as a normal dirtied-range upload");
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
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vb_lock: device_inner null on rename path");
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
                        // or whole-buffer WRITEONLY contended (game writes
                        // every byte; old contents need not survive).
                        dev.perf_mut().bump_vb_discard();
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
                dev.perf_mut().bump_vb_rename();
                dev.queue_vbib_retention(buffer_id, old_box, old_seq);
                inner.last_submit_seq = 0;
            }
            LockPlan::WriteInPlace => {}
        }
    }

    inner.locked = true;

    let unknown = flags & !D3DLOCK_KNOWN_BITS;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vb_lock: unrecognised D3DLOCK bits {unknown:#x} ignored");
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

extern "system" fn vb_unlock(this: *mut c_void) -> i32 {
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVertexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner_mut();
    if !inner.locked {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vb_unlock: Unlock without matching Lock → S_OK");
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

extern "system" fn vb_get_desc(this: *mut c_void, desc: *mut D3DVERTEXBUFFER_DESC) -> i32 {
    let _timer = vb_timer(this);
    if desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DVertexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    // SAFETY: `desc` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DVERTEXBUFFER_DESC` slot owned by the
    // caller.
    unsafe {
        *desc = D3DVERTEXBUFFER_DESC {
            format: D3DFMT_VERTEXDATA,
            resource_type: D3DRTYPE_VERTEXBUFFER,
            usage: inner.usage,
            pool: inner.pool,
            size: inner.length,
            fvf: inner.fvf,
        };
    }
    D3D_OK
}
