//! `IDirect3DVertexBuffer9` COM wrapper.
//!
//! Per-buffer `PageBox` backing: a Lock that renames swaps the backing
//! for a fresh uninit `PageBox`; the old one goes onto the device's
//! retention pipeline, picked up by the encoder and paired with its
//! wrapped `MTLBuffer` for GPU-retirement-gated destruction. Renaming
//! takes contention (`last_submit_seq > coherent_seq`, without
//! `NOOVERWRITE` or `READONLY`) plus either `DISCARD` or a whole-buffer
//! range: a contended *partial* Lock keeps the live pointer, the
//! divergence the README lists under "Faster than conformant". Unlock is
//! a no-op for `Direct` buffers; `Staged` buffers upload their dirty
//! span there.
//!
//! Layout follows the "state on Inner" pattern: the `#[repr(C)]` outer
//! struct only carries the vtable, refcount, and an opaque pointer to
//! the real state; everything else lives on `VertexBufferInner`.

use core::ffi::c_void;
use std::sync::atomic::Ordering;

use mtld3d_core::{
    buffer_backing::{BufferBacking, classify_backing, may_release_backing},
    buffer_rename::{
        BufferMapMode, LockPlan, PreserveKind, classify_map_mode, may_trust_lock_bounds, plan_lock,
        records_dirty_range,
    },
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
    D3D_OK, D3DERR_INVALIDCALL, com_ref::ComUnknown, device::DeviceInner,
    private_data::PrivateDataStore,
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
    /// GUID-keyed application private data (`Set/Get/FreePrivateData`).
    ///
    /// Any stored `IUnknown` is released when this `VertexBufferInner` drops.
    private_data: PrivateDataStore,
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
    /// to the device buffer, and a buffer D3D9 promises no readback
    /// releases it there (see `mtld3d_core::buffer_backing`).
    backing: BufferBacking,
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

    /// The buffer's CPU backing bytes.
    ///
    /// Used as the `ProcessVertices` source read: the source stream is a
    /// system-memory or default buffer whose current backing holds what the
    /// app wrote. Empty for a buffer that released its backing, which
    /// `may_release_backing` keeps out of this path and the caller checks.
    #[must_use]
    pub fn backing(&self) -> &[u8] {
        self.backing.as_slice()
    }

    /// Whether the buffer holds no CPU copy of its contents.
    ///
    /// True only for a `D3DUSAGE_WRITEONLY` default-pool buffer whose
    /// upload has been queued: its bytes live on the GPU alone until the
    /// next `Lock` re-creates the backing.
    #[must_use]
    pub const fn backing_is_released(&self) -> bool {
        self.backing.is_released()
    }

    /// Give a released buffer a backing again, zeroed.
    ///
    /// Every write path calls this before it touches the backing. The
    /// fresh pages hold no contents: only what is written into them from
    /// here on matches the device buffer, which is what
    /// `BackingState::Partial` records.
    fn restore_backing(&mut self) {
        if !self.backing.is_released() {
            return;
        }
        self.backing
            .restore(PageBox::new_zeroed(self.length as usize));
        mtld3d_shared::log_once_trace_by!(
            target: crate::LOG_TARGET,
            key: self.buffer_id.raw(),
            "vertex buffer {:#x}: backing re-created for a write",
            self.buffer_id.raw()
        );
    }

    /// The buffer's creation FVF.
    #[must_use]
    pub const fn fvf(&self) -> u32 {
        self.fvf
    }

    /// Write processed vertices into the backing at `offset`, uploading a `Staged` destination.
    ///
    /// The `ProcessVertices` destination sink: the transformed bytes land in
    /// the CPU backing (which a later `Lock` reads back for a system-memory
    /// buffer), and a `Staged` buffer additionally uploads the written span so
    /// a later draw from a default-pool destination sees it. Out-of-range
    /// writes are clipped to the backing.
    pub fn write_processed(&mut self, offset: usize, bytes: &[u8], dev: &mut DeviceInner) {
        self.restore_backing();
        let dst = self.backing.as_mut_slice();
        let end = offset.saturating_add(bytes.len()).min(dst.len());
        if offset >= end {
            return;
        }
        dst[offset..end].copy_from_slice(&bytes[..end - offset]);
        if !matches!(self.map_mode, BufferMapMode::Staged) {
            return;
        }
        let size = end - offset;
        let Some(src) = self.backing.read_ptr_at(offset) else {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "write_processed: no CPU backing to upload the processed vertices from");
            return;
        };
        let mut transient = dev.alloc_pagebox_capped(size);
        // SAFETY: `src` spans `[offset, end)` of the backing; `transient` is a
        // fresh `PageBox` of at least `size` bytes; the two are disjoint.
        unsafe { core::ptr::copy_nonoverlapping(src, transient.as_mut_ptr(), size) };
        let (min, len) = (
            u32::try_from(offset).expect("VB offset fits u32"),
            u32::try_from(size).expect("VB span fits u32"),
        );
        self.backing.note_upload(min, min + len);
        dev.push_stage_upload(self.buffer_id, transient, min, len);
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
        let Some(src) = self.backing.read_ptr_at(min as usize) else {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "flush_staged_if_mapped: no CPU backing behind a mapped dirty range");
            return;
        };
        let size = (max - min) as usize;
        let mut transient = dev.alloc_pagebox_capped(size);
        // SAFETY: `src` spans `[min, max)` of the backing; `transient` is a
        // fresh `PageBox` of ≥ `size` bytes; the two allocations are disjoint.
        unsafe {
            core::ptr::copy_nonoverlapping(src, transient.as_mut_ptr(), size);
        }
        self.backing.note_upload(min, max);
        dev.push_stage_upload(self.buffer_id, transient, min, max - min);
    }

    pub const fn map_mode(&self) -> BufferMapMode {
        self.map_mode
    }

    pub fn current_backing_ptr(&self) -> u64 {
        self.backing.ptr()
    }

    /// The current backing allocation's identity (see `BufferBacking::generation`).
    pub fn current_backing_generation(&self) -> u64 {
        self.backing.generation()
    }

    pub const fn current_backing_len(&self) -> u64 {
        self.backing.padded_len()
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
        // Zeroed, not uninit: a `Staged` buffer's first upload carries the
        // whole staging region, so any byte the game left alone has to be
        // a defined value rather than heap residue. One `bzero` per create,
        // and renames keep using the recycle pool.
        let backing = BufferBacking::new(
            PageBox::new_zeroed(info.length as usize),
            info.length,
            classify_backing(info.usage, info.pool),
        );
        let map_mode = classify_map_mode(info.usage, info.pool);
        // A `Staged` buffer starts full-dirty, whatever
        // `buffer.ignoreLockBounds` says. Its device allocation is
        // `Private`, Metal does not zero it, and no blit command can fill
        // a buffer, so the opening upload is the only thing that can give
        // it defined contents. A fill made entirely through locks that
        // record no range (a `MANAGED` buffer's `READONLY` locks)
        // announces nothing, so that upload has to carry every byte or
        // the GPU reads an undefined buffer. It is also the one whole-buffer
        // range that is free: no draw has read the buffer yet, so it
        // cannot trip rename-at-overlap. `Direct` buffers share one
        // allocation with the GPU and never read `dirty`.
        // `texture_unlock_rect`'s `was_uploaded` gate is the same rule for
        // mips: a level filled only through READONLY locks still has to
        // reach the GPU once.
        let dirty = if matches!(map_mode, BufferMapMode::Staged) {
            DirtyRange::full(info.length)
        } else {
            DirtyRange::empty()
        };
        let inner = Box::into_raw(Box::new(VertexBufferInner {
            device_inner: info.device_inner,
            buffer_id: BufferId::new_unique(),
            length: info.length,
            usage: info.usage,
            fvf: info.fvf,
            pool: info.pool,
            private_data: PrivateDataStore::default(),
            map_mode,
            dirty,
            backing,
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
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this`, `riid` and `ppv` are the caller's per the
    // IUnknown::QueryInterface ABI.
    unsafe {
        crate::com_ref::com_query_interface(
            this,
            riid,
            ppv,
            &[
                mtld3d_types::IID_IUNKNOWN,
                mtld3d_types::IID_IDIRECT3DRESOURCE9,
                mtld3d_types::IID_IDIRECT3DVERTEXBUFFER9,
            ],
            vb_add_ref,
            "IDirect3DVertexBuffer9",
        )
    }
}

extern "system" fn vb_add_ref(this: *mut c_void) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: IDirect3DVertexBuffer9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DVertexBuffer9>(this) }
}

extern "system" fn vb_release(this: *mut c_void) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
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
    let mut inner_box = unsafe { Box::from_raw(inner_ptr) };
    // A buffer that already released its backing has nothing left to
    // retain: the GPU reads its device buffer, not this memory.
    let current_box = inner_box.backing.release();
    let VertexBufferInner {
        device_inner,
        buffer_id,
        last_submit_seq,
        ..
    } = *inner_box;
    if !device_inner.is_null()
        && let Some(current_box) = current_box
    {
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
    fn blocks_reset_while_referenced(&self) -> bool {
        self.inner().pool == mtld3d_types::D3DPOOL_DEFAULT
    }
    fn private_refcount(&self) -> u32 {
        self.private_refcount
    }
    fn owning_device(&self) -> *mut c_void {
        crate::device::device_wrapper_from(self.inner().device_inner)
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — both counters are zero.
        unsafe { finalize_vertex_buffer(this) };
    }
}

// ── IDirect3DResource9 ──

extern "system" fn vb_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per its ABI, and `device` is
    // the caller's out-param.
    unsafe { crate::com_ref::com_get_device::<Direct3DVertexBuffer9>(this, device) }
}

extern "system" fn vb_set_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *const c_void,
    size: u32,
    flags: u32,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVertexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let store = &mut obj.inner_mut().private_data;
    // SAFETY: `data`/`size`/`flags` are the caller-supplied payload per the
    // D3D9 ABI; `set` validates them.
    unsafe { store.set(&guid, data, size, flags) }
}

extern "system" fn vb_get_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *mut c_void,
    size: *mut u32,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DVertexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size` are the caller-owned out buffer + size slot per
    // the D3D9 ABI; the store validates the size before any copy.
    unsafe { obj.inner().private_data.get(&guid, data, size) }
}

extern "system" fn vb_free_private_data(this: *mut c_void, guid: *const Guid) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DVertexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner_mut().private_data.free(&guid)
}

// Priority is honoured only for `D3DPOOL_MANAGED` resources (D3D9 manager
// eviction order). For every other pool both accessors are fixed at `0`.
// Metal has no eviction-order hint, so the value is stored and round-tripped
// but never acted upon.
extern "system" fn vb_set_priority(this: *mut c_void, priority: u32) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
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
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DVertexBuffer9>::opt(this) }) else {
        return 0;
    };
    obj.inner().priority
}

extern "system" fn vb_pre_load(this: *mut c_void) {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    // See IDirect3DTexture9::PreLoad — Metal has no resident-set hint.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DVertexBuffer9::PreLoad: no Metal analog, no-op"
    );
}

extern "system" fn vb_get_type(this: *mut c_void) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
    let _timer = vb_timer(this);
    D3DRTYPE_VERTEXBUFFER
}

// ── IDirect3DVertexBuffer9 ──

/// Honor the D3DLOCK flags per `buffer_rename::plan_lock`:
///
/// - `NOOVERWRITE | READONLY`, or uncontended: return the existing
///   backing pointer.
/// - Contended `DISCARD`: swap the backing for a fresh uninit
///   `PageBox`; old goes to seq-gated retention.
/// - Contended whole-buffer (any flag combo): same swap. If the
///   buffer is non-WRITEONLY, memcpy the old bytes across (game
///   might read the whole buffer through the Lock pointer).
/// - Contended partial non-DISCARD, `D3DUSAGE_DYNAMIC`: `WriteInPlace`.
///   The game opted into the DISCARD/NOOVERWRITE timing contract, the
///   same one non-persistent mapped-buffer APIs (e.g. OpenGL
///   `glBufferSubData`) make implicitly. This is the divergence the
///   README lists under "Faster than conformant"; the only trace it
///   leaves is the `in-place` perf counter bumped below.
/// - Non-DYNAMIC buffers never reach `plan_lock`: they are `Staged`, and
///   a partial write there uploads the dirtied range to a separate
///   device buffer on Unlock, so it cannot land under a queued draw.
extern "system" fn vb_lock(
    this: *mut c_void,
    offset_to_lock: u32,
    size_to_lock: u32,
    pp_data: *mut *mut c_void,
    flags: u32,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
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

    if inner.backing.is_released() {
        // The bytes live on the GPU alone. A read through this pointer sees
        // zeros rather than the buffer's contents, which is why only the
        // usages D3D9 promises no readback release their backing at all.
        if flags & D3DLOCK_READONLY != 0 {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "vb_lock: D3DLOCK_READONLY on a D3DUSAGE_WRITEONLY buffer whose backing was released; the mapped bytes read as zero");
        }
        inner.restore_backing();
    }

    if matches!(inner.map_mode, BufferMapMode::Staged) {
        // Separate CPU staging: record the dirtied range for the Unlock
        // upload. No rename / no `plan_lock` — the GPU reads a distinct
        // device buffer, so a partial write can't race an in-flight draw.
        // `records_dirty_range` drops the locks that leave the device
        // buffer nothing to pick up: READONLY, in the one pool that keeps
        // a system-memory copy whose upload it can skip, and
        // NO_DIRTY_UPDATE is not honoured: it is not a promise that
        // nothing was written, and this path has no later upload to carry
        // the bytes, so dropping the range would drop the write.
        if records_dirty_range(flags, inner.pool) {
            if flags & D3DLOCK_DISCARD != 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "vb_lock: D3DLOCK_DISCARD on a non-DYNAMIC (Staged) buffer — treating as a normal dirtied-range upload");
            }
            // The announced window is normally taken as a bound on what
            // the game wrote. It is not under `buffer.ignoreLockBounds`,
            // nor for the two shapes that name no narrower window at all
            // (`D3DLOCK_DISCARD`, and a zero `SizeToLock`, which D3D9
            // documents as locking the whole buffer). Then the upload
            // widens to `(0, 0)`, which is `conjoin`'s "to end of buffer"
            // from offset zero, so the head is covered too.
            //
            // A backing re-created after a release is the one case where
            // widening cannot help: it holds zeros outside what the game
            // writes through this very Lock, so the wider upload would push
            // those zeros over device bytes nobody rewrote. The announced
            // window stands instead. `D3DLOCK_DISCARD` is exempt because it
            // abandons the buffer's contents by definition, so a whole-buffer
            // upload of whatever the game leaves behind is what D3D9 promises,
            // and it puts the backing back in step with the device buffer.
            let trusted = may_trust_lock_bounds(
                flags,
                inner.usage,
                inner.pool,
                size_to_lock,
                ignore_lock_bounds(inner),
            );
            let discard = flags & D3DLOCK_DISCARD != 0;
            let widen = !trusted && (inner.backing.may_widen_upload() || discard);
            if !trusted && !widen {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "vb_lock: keeping the announced lock window on a re-created backing; a widened upload would overwrite device bytes the buffer no longer holds");
            }
            let (dirty_offset, dirty_size) = if widen {
                (0, 0)
            } else {
                (offset_to_lock, size_to_lock)
            };
            inner.dirty.conjoin(dirty_offset, dirty_size, inner.length);
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
        // The same contention test `plan_lock` applies, named here so
        // both it and the plan read one sampled `coh`. A stale `coh` is
        // a lower bound on GPU progress (only the unix side raises it,
        // with a `fetch_max` once the GPU retires the frame), so it can
        // turn a legal in-place write into an unnecessary rename, never
        // the reverse.
        let contended = inner.last_submit_seq > coh;
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
                let fresh = dev.alloc_pagebox_capped(logical_len);
                // `Direct` buffers never release their backing (the GPU
                // reads it), so the swap always hands the old one back.
                let Some(old_box) = inner.backing.replace(fresh) else {
                    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                        "vb_lock: rename found no backing to retain on a zero-copy buffer");
                    return D3DERR_INVALIDCALL;
                };
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
                        let dst = inner
                            .backing
                            .write_ptr_at(0)
                            .expect("the fresh rename backing was just installed");
                        // SAFETY: both `old_box` and the fresh backing are
                        // `PageBox`es of `logical_len` bytes; the two
                        // allocations don't alias.
                        unsafe {
                            core::ptr::copy_nonoverlapping(old_box.as_ptr(), dst, logical_len);
                        }
                    }
                }
                dev.perf_mut().bump_vb_rename();
                let renamed_bytes = usize::try_from(inner.backing.padded_len())
                    .expect("a PageBox length fits the host address space");
                dev.perf_mut().bump_vbib_rename_bytes(renamed_bytes);
                dev.queue_vbib_retention(buffer_id, old_box, old_seq);
                inner.last_submit_seq = 0;
            }
            LockPlan::WriteInPlace => {
                // Count the kept divergence: a contended partial Lock
                // without DISCARD or NOOVERWRITE hands back a pointer
                // into the backing a queued draw may still be reading
                // (README, "Faster than conformant"). Counted and not
                // warned because it is a by-design no-op on a per-frame
                // batcher path, not a stub or a fallback. The other two
                // ways to reach `WriteInPlace` (NOOVERWRITE/READONLY,
                // uncontended) are conformant, so it takes both tests to
                // select this arm.
                if contended && !bypass_rename {
                    dev.perf_mut().bump_vbib_write_in_place_contended();
                }
            }
        }
    }

    inner.locked = true;

    let unknown = flags & !D3DLOCK_KNOWN_BITS;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vb_lock: unrecognised D3DLOCK bits {unknown:#x} ignored");
    }
    // `offset_to_lock <= inner.length` is checked above and the backing is
    // allocated for `inner.length` bytes, so the offset lands inside it.
    let Some(ptr) = inner.backing.write_ptr_at(offset_to_lock as usize) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vb_lock: no backing to map");
        // SAFETY: `pp_data` is non-null (checked above) and per the D3D9
        // ABI points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *pp_data = core::ptr::null_mut() };
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `pp_data` is non-null (checked above) and per the D3D9
    // ABI points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *pp_data = ptr.cast::<c_void>() };
    D3D_OK
}

extern "system" fn vb_unlock(this: *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
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
        && !inner.device_inner.is_null()
    {
        // SAFETY: `inner.device_inner` was stamped at `Self::new` from
        // a live `DeviceInner` that outlives its children.
        let dev = unsafe { &mut *inner.device_inner };
        let Some(src) = inner.backing.read_ptr_at(min as usize) else {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "vb_unlock: dirty range with no CPU backing behind it, dropping the upload");
            inner.dirty.clear();
            return D3D_OK;
        };
        let size = (max - min) as usize;
        let mut transient = dev.alloc_pagebox_capped(size);
        // SAFETY: `src` spans `[min, max)` of the backing; `transient` is
        // a fresh `PageBox` of ≥ `size` bytes; the two allocations are
        // disjoint.
        unsafe {
            core::ptr::copy_nonoverlapping(src, transient.as_mut_ptr(), size);
        }
        // Push the upload as an inline op so the encoder sees it in
        // draw order (for rename-at-overlap). No Metal thunk here.
        inner.backing.note_upload(min, max);
        dev.push_stage_upload(inner.buffer_id, transient, min, max - min);
        // Only once the upload is actually queued: the range is the
        // only record that these bytes still owe a copy to the device
        // buffer, so clearing it on a path that queued nothing would
        // drop them silently.
        inner.dirty.clear();
        release_backing_after_upload(inner);
    }
    D3D_OK
}

/// `buffer.ignoreLockBounds` of the device that owns this buffer.
///
/// A buffer with no device behind it (the pointer is stamped at creation and
/// is null only for a buffer that outlived a teardown) answers the default.
fn ignore_lock_bounds(inner: &VertexBufferInner) -> bool {
    if inner.device_inner.is_null() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "vb: no device behind the buffer; buffer.ignoreLockBounds reads as false");
        return false;
    }
    // SAFETY: `inner.device_inner` was stamped at `Self::new` from a live
    // `DeviceInner`; the device outlives all its child resources per D3D9
    // lifetime rules.
    let dev = unsafe { &*inner.device_inner };
    dev.config().buffer_ignore_lock_bounds
}

/// Release the CPU backing of a buffer whose upload just carried every byte.
///
/// The transient the upload owns holds the bytes until the GPU has them, so
/// the buffer's own copy is dead weight from here: `D3DUSAGE_WRITEONLY` in
/// `D3DPOOL_DEFAULT` is the one class D3D9 promises no readback, and inside a
/// 32-bit title that copy competes with the title's own address space. A
/// later `Lock` re-creates the backing and uploads only what it announces.
///
/// The trade is the one the texture-staging release already accepts: the GPU
/// holds the only copy, so nothing can re-upload these buffers if the Metal
/// device is recreated under them.
fn release_backing_after_upload(inner: &mut VertexBufferInner) {
    // `buffer.ignoreLockBounds` says this title writes outside the windows it
    // announces, and the only way to carry those writes is to upload the whole
    // buffer out of the CPU copy. Releasing the copy would leave that upload
    // nothing true to carry, so the knob keeps it.
    if inner.locked
        || ignore_lock_bounds(inner)
        || !may_release_backing(inner.usage, inner.pool)
        || !inner.backing.may_widen_upload()
    {
        return;
    }
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
        "releasing the CPU backing of uploaded D3DPOOL_DEFAULT D3DUSAGE_WRITEONLY buffers; \
         their bytes then live on the GPU alone and a device recreate cannot restore them");
    drop(inner.backing.release());
}

extern "system" fn vb_get_desc(this: *mut c_void, desc: *mut D3DVERTEXBUFFER_DESC) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DVertexBuffer9>(this);
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
