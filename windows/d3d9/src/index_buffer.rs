//! `IDirect3DIndexBuffer9` COM wrapper.
//!
//! Mirrors `vertex_buffer.rs`; the vtable type, the `fvf` → `format` field
//! swap, the reported resource type, and priority handling differ — the index
//! buffer's `SetPriority` / `GetPriority` are no-ops rather than round-tripping
//! an app-set value.

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
    D3DINDEXBUFFER_DESC, D3DLOCK_DISCARD, D3DLOCK_KNOWN_BITS, D3DLOCK_NOOVERWRITE,
    D3DLOCK_READONLY, D3DRTYPE_INDEXBUFFER, Guid, IDirect3DIndexBuffer9Vtbl,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, com_ref::ComUnknown, device::DeviceInner,
    private_data::PrivateDataStore,
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
    /// GUID-keyed application private data (`Set/Get/FreePrivateData`).
    ///
    /// Any stored `IUnknown` is released when this `IndexBufferInner` drops.
    private_data: PrivateDataStore,
    /// `Direct` (zero-copy) vs `Staged` (separate device buffer + dirty-range upload).
    ///
    /// Decided once at creation from `usage`/`pool`.
    map_mode: BufferMapMode,
    /// `Staged` only: byte range dirtied since the last upload.
    dirty: DirtyRange,
    /// Canonical CPU backing.
    ///
    /// Released like a vertex buffer's once an upload has carried every
    /// byte of a buffer D3D9 promises no readback (see
    /// `mtld3d_core::buffer_backing`). The indexed triangle-fan rewrite is
    /// the one path that still reads these bytes on the CPU; it reads them
    /// back off the GPU instead and pins what it installs.
    backing: BufferBacking,
    last_submit_seq: u64,
    locked: bool,
}

impl IndexBufferInner {
    pub const fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    /// The buffer's D3D9-visible length in bytes.
    pub const fn length(&self) -> u32 {
        self.length
    }

    /// Whether the buffer holds no CPU copy of its contents.
    ///
    /// True only for a `D3DUSAGE_WRITEONLY` default-pool buffer whose
    /// upload has been queued: its bytes live on the GPU alone until a
    /// later `Lock` re-creates the backing, or the indexed triangle-fan
    /// rewrite reads them back.
    #[must_use]
    pub const fn backing_is_released(&self) -> bool {
        self.backing.is_released()
    }

    /// Adopt index bytes read back out of the device buffer.
    ///
    /// Installs them as the backing and pins it, so the buffer never pays
    /// the read's GPU stall twice.
    pub fn adopt_device_copy(&mut self, page_box: PageBox) {
        self.backing.adopt_device_copy(page_box);
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
            "index buffer {:#x}: backing re-created for a write",
            self.buffer_id.raw()
        );
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

    pub const fn format(&self) -> u32 {
        self.format
    }

    pub fn current_backing_ptr(&self) -> u64 {
        self.backing.ptr()
    }

    pub const fn current_backing_len(&self) -> u64 {
        self.backing.padded_len()
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
        // Zeroed and full-dirty for the same reasons as
        // `Direct3DVertexBuffer9::new`: a `Staged` buffer's opening upload
        // carries every byte, so untouched bytes must be defined, and that
        // opening upload is unconditional because the device buffer starts
        // undefined, no blit command can fill it, and a fill made only
        // through locks `records_dirty_range` rejects announces nothing.
        let backing = BufferBacking::new(
            PageBox::new_zeroed(info.length as usize),
            info.length,
            classify_backing(info.usage, info.pool),
        );
        let map_mode = classify_map_mode(info.usage, info.pool);
        let dirty = if matches!(map_mode, BufferMapMode::Staged) {
            DirtyRange::full(info.length)
        } else {
            DirtyRange::empty()
        };
        let inner = Box::into_raw(Box::new(IndexBufferInner {
            device_inner: info.device_inner,
            buffer_id: BufferId::new_unique(),
            length: info.length,
            usage: info.usage,
            format: info.format,
            pool: info.pool,
            private_data: PrivateDataStore::default(),
            map_mode,
            dirty,
            backing,
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
                mtld3d_types::IID_IDIRECT3DINDEXBUFFER9,
            ],
            ib_add_ref,
            "IDirect3DIndexBuffer9",
        )
    }
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
    let mut inner_box = unsafe { Box::from_raw(inner_ptr) };
    let current_box = inner_box.backing.release();
    let IndexBufferInner {
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
        unsafe { finalize_index_buffer(this) };
    }
}

// ── IDirect3DResource9 ──

extern "system" fn ib_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per its ABI, and `device` is
    // the caller's out-param.
    unsafe { crate::com_ref::com_get_device::<Direct3DIndexBuffer9>(this, device) }
}

extern "system" fn ib_set_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *const c_void,
    size: u32,
    flags: u32,
) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let store = &mut obj.inner_mut().private_data;
    // SAFETY: `data`/`size`/`flags` are the caller-supplied payload per the
    // D3D9 ABI; `set` validates them.
    unsafe { store.set(&guid, data, size, flags) }
}

extern "system" fn ib_get_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *mut c_void,
    size: *mut u32,
) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size` are the caller-owned out buffer + size slot per
    // the D3D9 ABI; the store validates the size before any copy.
    unsafe { obj.inner().private_data.get(&guid, data, size) }
}

extern "system" fn ib_free_private_data(this: *mut c_void, guid: *const Guid) -> i32 {
    let _timer = ib_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DIndexBuffer9 per ABI.
    let Some(mut obj) = (unsafe { InPtrMut::<Direct3DIndexBuffer9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner_mut().private_data.free(&guid)
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

    if inner.backing.is_released() {
        // The bytes live on the GPU alone. A read through this pointer sees
        // zeros rather than the buffer's contents, which is why only the
        // usages D3D9 promises no readback release their backing at all.
        if flags & D3DLOCK_READONLY != 0 {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "ib_lock: D3DLOCK_READONLY on a D3DUSAGE_WRITEONLY buffer whose backing was released; the mapped bytes read as zero");
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
                    "ib_lock: D3DLOCK_DISCARD on a non-DYNAMIC (Staged) buffer — treating as a normal dirtied-range upload");
            }
            // `(0, 0)` is `conjoin`'s "to end of buffer"; see the twin
            // comment in `vb_lock`, which also carries the reason a
            // re-created backing may not widen.
            let trusted = may_trust_lock_bounds(
                flags,
                inner.usage,
                inner.pool,
                size_to_lock,
                crate::config::CONFIG.buffer_ignore_lock_bounds,
            );
            let discard = flags & D3DLOCK_DISCARD != 0;
            let widen = !trusted && (inner.backing.may_widen_upload() || discard);
            if !trusted && !widen {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "ib_lock: keeping the announced lock window on a re-created backing; a widened upload would overwrite device bytes the buffer no longer holds");
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
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_lock: device_inner null on rename path");
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
                        "ib_lock: rename found no backing to retain");
                    return D3DERR_INVALIDCALL;
                };
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
                dev.perf_mut().bump_ib_rename();
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
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_lock: unrecognised D3DLOCK bits {unknown:#x} ignored");
    }
    // `offset_to_lock <= inner.length` is checked above and the backing is
    // allocated for `inner.length` bytes, so the offset lands inside it.
    let Some(ptr) = inner.backing.write_ptr_at(offset_to_lock as usize) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ib_lock: no backing to map");
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
        && !inner.device_inner.is_null()
    {
        // SAFETY: `inner.device_inner` was stamped at `Self::new` from
        // a live `DeviceInner` that outlives its children.
        let dev = unsafe { &mut *inner.device_inner };
        let Some(src) = inner.backing.read_ptr_at(min as usize) else {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "ib_unlock: dirty range with no CPU backing behind it, dropping the upload");
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

/// Release the CPU backing of an index buffer whose upload just carried every byte.
///
/// Same class and same guards as the vertex-buffer release: the transient
/// the upload owns holds the bytes until the GPU has them, `D3DPOOL_DEFAULT`
/// `D3DUSAGE_WRITEONLY` is the one class D3D9 promises no readback, and
/// `buffer.ignoreLockBounds` names the titles whose uploads have to be able
/// to widen out of the CPU copy. A pinned backing is the extra guard an
/// index buffer needs: the indexed triangle-fan rewrite has already paid a
/// GPU stall to read these bytes back, so releasing them would only buy
/// another one.
fn release_backing_after_upload(inner: &mut IndexBufferInner) {
    if inner.locked
        || inner.backing.is_pinned()
        || crate::config::CONFIG.buffer_ignore_lock_bounds
        || !may_release_backing(inner.usage, inner.pool)
        || !inner.backing.may_widen_upload()
    {
        return;
    }
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
        "releasing the CPU backing of uploaded D3DPOOL_DEFAULT D3DUSAGE_WRITEONLY index buffers; \
         their bytes then live on the GPU alone and a device recreate cannot restore them");
    drop(inner.backing.release());
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
