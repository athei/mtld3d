use core::ffi::c_void;

use log::trace;
use mtld3d_core::page_box::PageBox;
use mtld3d_shared::{
    BlitTextureToBufferParams, InPtr, InPtrMut, MetalHandle, ValueIn, VtableThis,
    mtl_handle::MTLTextureKind,
};
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A8R8G8B8, D3DFMT_R5G6B5, D3DFMT_R8G8B8, D3DFMT_X1R5G5B5,
    D3DFMT_X8R8G8B8, D3DLOCK_DISCARD, D3DLOCK_READONLY, D3DLOCKED_RECT, D3DPOOL_DEFAULT,
    D3DPRESENTFLAG_LOCKABLE_BACKBUFFER, D3DRECT, D3DSURFACE_DESC, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET, Guid, IDirect3DSurface9Vtbl,
    IID_IDIRECT3DBASETEXTURE9, IID_IDIRECT3DRESOURCE9, IID_IDIRECT3DTEXTURE9, IID_IUNKNOWN,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET,
    com_ref::ComUnknown,
    device::{DeviceInner, Direct3DDevice9},
    null_out,
    private_data::PrivateDataStore,
    texture::Direct3DTexture9,
    unix_call::unix_call,
};

static DIRECT3D_SURFACE9_VTBL: IDirect3DSurface9Vtbl = IDirect3DSurface9Vtbl {
    query_interface: surface_query_interface,
    add_ref: surface_add_ref,
    release: surface_release,
    get_device: surface_get_device,
    set_private_data: surface_set_private_data,
    get_private_data: surface_get_private_data,
    free_private_data: surface_free_private_data,
    set_priority: surface_set_priority,
    get_priority: surface_get_priority,
    pre_load: surface_pre_load,
    get_type: surface_get_type,
    get_container: surface_get_container,
    get_desc: surface_get_desc,
    lock_rect: surface_lock_rect,
    unlock_rect: surface_unlock_rect,
    get_dc: surface_get_dc,
    release_dc: surface_release_dc,
};

/// Marks a surface as one of the device's **implicit** (device-owned) surfaces.
///
/// `None` marks an ordinary app-owned surface.
///
/// Implicit surfaces are cached on `DeviceInner`, created with refcount 0,
/// forward their refcount to the device on the 0↔1 boundary, are never freed
/// when their refcount reaches 0 (destroyed only at device teardown), and
/// resolve their Metal handle + dimensions **live** from the device every call
/// — so a backbuffer/depth texture recreated by a window resize
/// (`DeviceInner::apply_auto_resize`) or `Reset` is never observed through a
/// freed handle.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImplicitKind {
    /// Ordinary surface: normal refcount-1 lifecycle, snapshot fields.
    None,
    /// The implicit render target == backbuffer (`GetRenderTarget(0)` / `GetBackBuffer(0)`).
    ///
    /// Resolves color handle + dims from the device's current backbuffer.
    Backbuffer,
    /// The implicit depth-stencil (`GetDepthStencilSurface`).
    ///
    /// Resolves depth handle + dims + format from the device's current
    /// depth-stencil.
    DepthStencil,
}

/// Resource-wide `LockRect`/`GetDC` mutual-exclusion state.
///
/// Tracks the D3D9 per-resource map bookkeeping: an outstanding-map count plus
/// a DC-in-use flag. Stored on a `SurfaceInner` for a standalone surface (a
/// single-"face" resource) and on a `TextureInner` for a cube map's six face
/// shells, which share one per-resource state — D3D9 gates `GetDC` against the
/// *whole* cube while still permitting two distinct faces to be
/// `LockRect`-mapped at once.
///
/// The per-face lock flag (`SurfaceInner::mapped`, the D3D9 per-sub-resource
/// map count) lives on each surface; this struct holds only the
/// resource-wide pieces:
/// * `map_count` — number of outstanding face locks **plus** an outstanding DC.
///   `GetDC` is rejected whenever it is non-zero.
/// * `dc_in_use` — a `GetDC` is outstanding somewhere on the resource. Blocks
///   every `LockRect`, and turns an `UnlockRect` of an unmapped face into a
///   no-op success (the D3D9 behavior the conformance test asserts).
/// * `held_dc` — the GDI objects of that outstanding `GetDC`.
pub struct DcLockState {
    map_count: u32,
    dc_in_use: bool,
    held_dc: GdiDc,
}

impl Default for DcLockState {
    fn default() -> Self {
        Self {
            map_count: 0,
            dc_in_use: false,
            held_dc: GdiDc::NULL,
        }
    }
}

impl DcLockState {
    /// Tear down any GDI objects of an outstanding `GetDC` and reset to the unlocked state.
    ///
    /// Called at resource teardown (a cube texture finalizing with a face's DC
    /// never released) so the memory DC + DIB do not leak.
    pub fn teardown(&mut self) {
        teardown_gdi_dc(self.held_dc);
        *self = Self::default();
    }
}

#[repr(C)]
pub struct Direct3DSurface9 {
    vtbl: *const IDirect3DSurface9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut SurfaceInner,
}

impl Direct3DSurface9 {
    /// Standalone color render-target surface.
    ///
    /// Created by `CreateRenderTarget` and by `CreateOffscreenPlainSurface`
    /// with `D3DPOOL_DEFAULT`. Wraps a persistent render-target-capable
    /// `MTLTexture` via `metal_color_handle`, identical to the backbuffer
    /// surface, so `StretchRect` and `GetRenderTargetData` resolve it for free.
    /// `usage` is `D3DUSAGE_RENDERTARGET` for a render target, `0` for an
    /// offscreen-plain `D3DPOOL_DEFAULT` surface.
    pub fn new_color_target(
        device_inner: *mut DeviceInner,
        metal_color_handle: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
        format: u32,
        usage: u32,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture: core::ptr::null_mut(),
            mip_level: 0,
            standalone_width: width,
            standalone_height: height,
            standalone_format: format,
            standalone_usage: usage,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::None,
            container: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    /// Standalone depth-stencil surface backed by a real Metal depth texture.
    ///
    /// `SetDepthStencilSurface` reads `metal_depth_handle()` and binds it to
    /// the next pass; without a backing handle the bind would silently fall
    /// back to the device default depth, mismatching the color attachment's
    /// size when `WoW` renders to an offscreen RT.
    pub fn new_depth_stencil(
        device_inner: *mut DeviceInner,
        metal_depth_handle: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
        format: u32,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture: core::ptr::null_mut(),
            mip_level: 0,
            standalone_width: width,
            standalone_height: height,
            standalone_format: format,
            // CreateDepthStencilSurface → a D3DUSAGE_DEPTHSTENCIL surface.
            standalone_usage: D3DUSAGE_DEPTHSTENCIL,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::None,
            container: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    /// The device's implicit render target == backbuffer.
    ///
    /// Handed out by `GetRenderTarget(0)` / `GetBackBuffer(0)`. Device-owned:
    /// refcount starts at 0, forwards to the device on the 0↔1 boundary, is
    /// never freed at refcount 0 (destroyed at device teardown), and resolves
    /// its color handle + dimensions live from the device (see
    /// [`ImplicitKind`]). `container` is the implicit swapchain wrapper
    /// `GetContainer` returns.
    pub fn new_implicit_backbuffer(device_inner: *mut DeviceInner, container: u64) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture: core::ptr::null_mut(),
            mip_level: 0,
            // Extent + Metal handle resolve live; only the pinned D3D format is a
            // snapshot (`live_format` returns it for a `Backbuffer` surface).
            standalone_width: 0,
            standalone_height: 0,
            standalone_format: D3DFMT_X8R8G8B8,
            // The implicit backbuffer/render target.
            standalone_usage: D3DUSAGE_RENDERTARGET,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::Backbuffer,
            container,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 0,
            private_refcount: 0,
            inner,
        }
    }

    /// The device's implicit depth-stencil (`GetDepthStencilSurface`).
    ///
    /// Device-owned exactly like [`Self::new_implicit_backbuffer`]; resolves its
    /// depth handle + dimensions + format live from the device. `container` is
    /// the device wrapper `GetContainer` returns.
    pub fn new_implicit_depth_stencil(device_inner: *mut DeviceInner, container: u64) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture: core::ptr::null_mut(),
            mip_level: 0,
            standalone_width: 0,
            standalone_height: 0,
            standalone_format: 0,
            // The implicit depth-stencil surface.
            standalone_usage: D3DUSAGE_DEPTHSTENCIL,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::DepthStencil,
            container,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 0,
            private_refcount: 0,
            inner,
        }
    }

    pub fn new_texture_backed(
        device_inner: *mut DeviceInner,
        parent_texture: *mut Direct3DTexture9,
        mip_level: u32,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture,
            mip_level,
            standalone_width: 0,
            standalone_height: 0,
            standalone_format: 0,
            standalone_usage: 0,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::None,
            container: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    /// Texture-backed surface that *owns* its parent texture.
    ///
    /// A `D3DPOOL_DEFAULT` offscreen-plain surface backed by an internal
    /// texture the app never sees. Takes over the texture's single create-ref
    /// (no `AddRef`); `finalize_surface` `Release`s it. Lock/Unlock/upload
    /// reuse the texture machinery (CPU staging + encoder
    /// `CopyBufferToTexture`), and `StretchRect` resolves the texture's Metal
    /// handle — so a single storage-with-CPU-sync path covers the surface.
    pub fn new_owned_texture_backed(
        device_inner: *mut DeviceInner,
        parent_texture: *mut Direct3DTexture9,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture,
            mip_level: 0,
            standalone_width: 0,
            standalone_height: 0,
            standalone_format: 0,
            standalone_usage: 0,
            standalone_pool: D3DPOOL_DEFAULT,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: None,
            flags: SurfaceFlags::OWNS_PARENT_TEXTURE,
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::None,
            container: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    /// System-memory offscreen plain surface.
    ///
    /// Created by `CreateOffscreenPlainSurface` with `D3DPOOL_SYSTEMMEM`.
    /// Backed by a page-aligned PE-heap buffer that `GetRenderTargetData` blits
    /// into and `LockRect` reads back — the destination half of the conformance
    /// / portrait readback path.
    pub fn new_system_memory(
        device_inner: *mut DeviceInner,
        width: u32,
        height: u32,
        format: u32,
        pool: u32,
        backing: PageBox,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SurfaceInner {
            device_inner,
            parent_texture: core::ptr::null_mut(),
            mip_level: 0,
            standalone_width: width,
            standalone_height: height,
            standalone_format: format,
            standalone_usage: 0,
            standalone_pool: pool,
            metal_depth_handle: MetalHandle::NULL,
            metal_color_handle: MetalHandle::NULL,
            readback: None,
            system_memory: Some(backing),
            flags: SurfaceFlags::empty(),
            private_data: PrivateDataStore::default(),
            dc_lock: DcLockState::default(),
            lock_flags: 0,
            state_owner_texture: core::ptr::null_mut(),
            implicit_kind: ImplicitKind::None,
            container: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SURFACE9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    /// Attach a CPU staging buffer to a standalone colour render target.
    ///
    /// Turns it into a **lockable** render target (`CreateRenderTarget` with
    /// `Lockable == TRUE`). The surface stays a standalone RT in every other
    /// respect — `parent_texture` null, `container` 0, `owns_parent_texture`
    /// false — so `GetContainer`/`GetDesc`/`StretchRect` behave exactly as for
    /// a non-lockable RT. Only `LockRect`/`UnlockRect` change: they now read /
    /// write `backing` and upload it to `metal_color_handle` on unlock. The
    /// `backing` must be sized `width * height * bpp` (tight pitch).
    pub fn set_lockable_staging(&mut self, backing: PageBox) {
        // SAFETY: `self.inner` is the live `SurfaceInner` for this wrapper,
        // freshly created by `new_color_target` (refcount 1, single-threaded).
        unsafe { &mut *self.inner }.system_memory = Some(backing);
    }

    /// Point this surface's `LockRect`/`GetDC` mutual-exclusion state at the owning cube.
    ///
    /// The cube is a `Direct3DTexture9`, and all six of its faces share one
    /// per-resource state. The caller (`GetCubeMapSurface`) must have taken a
    /// reference on `cube` so it outlives the face; the face's finalize
    /// releases that reference. Only valid on a freshly created face surface
    /// (refcount 1).
    pub fn set_cube_state_owner(&mut self, cube: *mut Direct3DTexture9) {
        // SAFETY: `self.inner` is the live `SurfaceInner` for this wrapper.
        unsafe { &mut *self.inner }.state_owner_texture = cube;
    }

    fn inner(&self) -> &SurfaceInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `surface_release` at
        // refcount zero, so it stays live for every live wrapper
        // reference.
        unsafe { &*self.inner }
    }

    pub const fn vtbl(&self) -> &IDirect3DSurface9Vtbl {
        // SAFETY: `self.vtbl` is the `'static` `DIRECT3D_SURFACE9_VTBL`
        // installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    /// Non-null parent for texture-backed surfaces (from `GetSurfaceLevel`).
    ///
    /// Null for standalone surfaces (backbuffer from `GetBackBuffer`,
    /// depth-stencil from `CreateDepthStencilSurface`). Used by the
    /// `SetRenderTarget` pass-break path to decide between rebinding a
    /// texture's Metal handle and restoring the backbuffer.
    pub fn parent_texture(&self) -> *mut Direct3DTexture9 {
        self.inner().parent_texture
    }

    /// Whether this surface currently has an outstanding `LockRect`.
    ///
    /// D3D9 rejects `UpdateSurface` when either endpoint is mapped. A
    /// texture-level surface records its lock on the parent texture
    /// (`LockRect` delegates there); a standalone surface uses its own `mapped`
    /// flag.
    pub fn is_locked(&self) -> bool {
        let parent = self.parent_texture();
        if parent.is_null() {
            self.inner().flags.contains(SurfaceFlags::MAPPED)
        } else {
            // SAFETY: `parent` is non-null (checked) and a live `Direct3DTexture9`
            // whose refcount keeps it alive while this surface is alive.
            unsafe { (*parent).inner() }.is_level_locked(self.mip_level() as usize)
        }
    }

    /// The `DeviceInner` that created this surface.
    ///
    /// Used by `SetRenderTarget` to reject a surface owned by a different
    /// device; mirrors `GetDevice`'s back-reference.
    pub fn device_inner(&self) -> *mut DeviceInner {
        self.inner().device_inner
    }

    /// True only for a `CreateOffscreenPlainSurface(D3DPOOL_DEFAULT)` surface.
    ///
    /// Such a surface owns its backing texture. D3D9 treats this as a valid
    /// `StretchRect` destination (a DEFAULT offscreen-plain surface), unlike an
    /// ordinary texture-level surface from `GetSurfaceLevel`.
    pub fn owns_parent_texture(&self) -> bool {
        self.inner()
            .flags
            .contains(SurfaceFlags::OWNS_PARENT_TEXTURE)
    }

    /// Metal depth texture backing this *standalone* surface, or null.
    ///
    /// Texture-backed depth surfaces (sampleable shadow maps from
    /// `CreateTexture(D24X8, DEPTHSTENCIL)` → `GetSurfaceLevel`) report
    /// null here — call `depth_texture_info` instead and resolve the
    /// Metal handle on the encoder thread via `get_or_create_texture`.
    pub fn metal_depth_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.inner().live_depth_handle()
    }

    /// Snapshot the parent depth-format texture's `TextureInfo`.
    ///
    /// For `SetDepthStencilSurface` lazy-fetch. Returns `None` for standalone
    /// surfaces (use `metal_depth_handle()` instead) and for texture-backed
    /// color surfaces.
    pub fn depth_texture_info(&self) -> Option<crate::encoder::TextureInfo> {
        let parent = self.inner().parent_texture;
        if parent.is_null() {
            return None;
        }
        // SAFETY: `parent` is non-null (checked above); texture-backed
        // surfaces are created by `GetSurfaceLevel` with a live
        // `Direct3DTexture9*`, and the parent texture's refcount keeps
        // it alive for as long as this surface is live.
        let tex = unsafe { &*parent };
        if !tex.is_depth_format() {
            return None;
        }
        Some(tex.inner().texture_info())
    }

    /// D3D9 format the standalone surface was created with (D3DFMT_*).
    ///
    /// Only meaningful when `parent_texture()` is null — texture-backed
    /// surfaces report their parent texture's format instead. Used by
    /// the API-thread snapshot to derive `has_depth/has_stencil` for
    /// the pipeline matching the currently bound depth attachment.
    pub fn standalone_format(&self) -> u32 {
        self.inner().live_format()
    }

    /// Width of a standalone surface (backbuffer / depth-stencil).
    ///
    /// Zero for texture-backed surfaces — query the parent texture mip instead
    /// via `mip_level()` + `Direct3DTexture9::mip_width()`.
    pub fn standalone_width(&self) -> u32 {
        self.inner().live_width()
    }

    pub fn standalone_height(&self) -> u32 {
        self.inner().live_height()
    }

    /// Mip level inside the parent texture.
    ///
    /// Only meaningful when `parent_texture()` is non-null.
    pub fn mip_level(&self) -> u32 {
        self.inner().mip_level
    }

    /// Persistent `MTLTexture*` for standalone color surfaces.
    ///
    /// The backbuffer returned by `GetBackBuffer`. Null on texture-backed or
    /// depth-stencil surfaces — `StretchRect` uses this to source / destination
    /// the backbuffer in a Metal blit without touching the transient
    /// `CAMetalLayer` drawable.
    pub fn metal_color_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.inner().live_color_handle()
    }

    /// Raw `(ptr, len)` of a system-memory offscreen surface's backing buffer.
    ///
    /// For use as a `BlitTextureToBuffer` destination. `None` for GPU-backed
    /// surfaces. The pointer stays valid for the surface's lifetime.
    ///
    /// A lockable render target also owns a CPU staging buffer, but it is a
    /// `D3DPOOL_DEFAULT` render target (it keeps its renderable colour handle),
    /// which D3D9 rejects as a `GetRenderTargetData` destination — so this
    /// returns `None` for it, leaving only a true `D3DPOOL_SYSTEMMEM`
    /// offscreen-plain surface (CPU backing, no colour handle) as a valid dest.
    pub fn system_memory_blit_dst(&self) -> Option<(u64, u64)> {
        // SAFETY: `self.inner` is the live `SurfaceInner` for this wrapper;
        // surfaces are single-threaded D3D9 objects, so the transient
        // exclusive borrow taken to read the buffer pointer is sound.
        let inner = unsafe { &mut *self.inner };
        if !inner.metal_color_handle.is_null() {
            return None;
        }
        inner
            .system_memory
            .as_mut()
            .map(|p| (p.as_mut_ptr() as u64, p.len() as u64))
    }

    /// Standalone `D3DPOOL_SYSTEMMEM`/`SCRATCH` offscreen surface as an `UpdateSurface` *source*.
    ///
    /// Yields `(ptr, len, width, height, format)` of its CPU backing. `None`
    /// for any GPU-backed or texture-backed surface. The row pitch is
    /// `width * bpp` rounded up to 4 (the layout `CreateOffscreenPlainSurface`
    /// allocated and `systemmem_lock_rect` reports), so the caller recomputes
    /// it from `format`.
    pub fn system_memory_source(&self) -> Option<(*const u8, usize, u32, u32, u32)> {
        if !self.inner().parent_texture.is_null() {
            return None;
        }
        let inner = self.inner();
        inner.system_memory.as_ref().map(|p| {
            (
                p.as_ptr(),
                p.len(),
                inner.standalone_width,
                inner.standalone_height,
                inner.standalone_format,
            )
        })
    }
}

bitflags::bitflags! {
    /// Per-surface boolean state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SurfaceFlags: u8 {
        /// The surface *owns* `parent_texture` and must `Release` it on finalize.
        ///
        /// A `D3DPOOL_DEFAULT` offscreen-plain surface backed by an internal
        /// texture the app never sees. Clear for ordinary `GetSurfaceLevel`
        /// surfaces, where the app holds the texture and the surface only
        /// borrows it.
        const OWNS_PARENT_TEXTURE = 1 << 0;
        /// This surface's own `LockRect` flag (the D3D9 per-sub-resource map count).
        ///
        /// Set between a successful `LockRect` and its `UnlockRect`. Distinct
        /// faces of one cube map each carry their own, so two faces can be
        /// mapped at once even though they share `DcLockState`.
        const MAPPED = 1 << 1;
    }
}

struct SurfaceInner {
    /// `DeviceInner` back-reference for `ApiTimer` accumulation.
    ///
    /// May be null for surfaces whose creation predated device tracking; the
    /// timer treats null as a no-op.
    device_inner: *mut DeviceInner,
    /// Non-null for texture-backed surfaces (from `GetSurfaceLevel`).
    ///
    /// Null for standalone surfaces wrapping a Metal object directly (e.g. the
    /// backbuffer surface `GetBackBuffer` hands out).
    parent_texture: *mut Direct3DTexture9,
    /// Per-surface boolean state (`OWNS_PARENT_TEXTURE` / `MAPPED`).
    ///
    /// See [`SurfaceFlags`].
    flags: SurfaceFlags,
    mip_level: u32,
    /// Standalone-surface description; only used when `parent_texture` is null.
    standalone_width: u32,
    standalone_height: u32,
    standalone_format: u32,
    /// `D3DUSAGE_*` flags a standalone surface reports from `GetDesc`.
    ///
    /// Only meaningful when `parent_texture` is null. `D3DUSAGE_RENDERTARGET`
    /// for a `CreateRenderTarget` color surface; `0` for the backbuffer,
    /// depth-stencil, offscreen-plain (system-memory or `D3DPOOL_DEFAULT`)
    /// surfaces.
    standalone_usage: u32,
    /// `D3DPOOL_*` a standalone surface reports from `GetDesc`.
    ///
    /// Only meaningful when `parent_texture` is null. `D3DPOOL_DEFAULT` for
    /// RT / depth-stencil / backbuffer; the actual creation pool
    /// (`D3DPOOL_SYSTEMMEM` or `D3DPOOL_SCRATCH`) for a system-memory
    /// offscreen-plain surface.
    standalone_pool: u32,
    /// Non-null for depth-stencil standalone surfaces created via `CreateDepthStencilSurface`.
    ///
    /// Holds the retained `MTLTexture*`.
    metal_depth_handle: MetalHandle<MTLTextureKind>,
    /// Non-null for standalone color surfaces (the backbuffer returned by `GetBackBuffer`).
    ///
    /// Holds the persistent offscreen `MTLTexture*` (same handle
    /// `DeviceInner::backbuffer_handle` carries). Enables synchronous
    /// `LockRect` readback via texture→buffer blit.
    metal_color_handle: MetalHandle<MTLTextureKind>,
    /// Page-aligned PE-addressable readback buffer held between `LockRect` and `UnlockRect`.
    ///
    /// Allocated on the `LockRect` readback path (backbuffer), dropped on
    /// `UnlockRect`. Persists across the Lock so the game can read the returned
    /// pointer.
    readback: Option<PageBox>,
    /// Backing store for a `D3DPOOL_SYSTEMMEM` offscreen plain surface.
    ///
    /// From `CreateOffscreenPlainSurface`. Allocated full-size at creation;
    /// `GetRenderTargetData`/`GetFrontBufferData` blit a render target's pixels
    /// into it, and `LockRect` hands back a pointer into it. `None` for every
    /// GPU-backed surface kind.
    system_memory: Option<PageBox>,
    /// GUID-keyed application private data (`Set/Get/FreePrivateData`).
    ///
    /// Any stored `IUnknown` is released when this `SurfaceInner` drops.
    private_data: PrivateDataStore,
    /// `None` for an ordinary app-owned surface.
    ///
    /// `Backbuffer`/`DepthStencil` for the device's implicit (device-owned)
    /// surfaces — see [`ImplicitKind`].
    implicit_kind: ImplicitKind,
    /// The COM object `GetContainer` hands back (`AddRef`'d).
    ///
    /// The implicit swapchain for an implicit `Backbuffer`, the device wrapper
    /// for an implicit `DepthStencil`. `0` for ordinary surfaces
    /// (`GetContainer` → INVALIDCALL).
    container: u64,
    /// Resource-wide `LockRect`/`GetDC` state for this surface alone (see [`DcLockState`]).
    ///
    /// For a cube-map face the *effective* resource-wide state lives on the
    /// owning cube texture (`state_owner_texture`) and this field is unused;
    /// the `MAPPED` flag is always this surface's own per-face flag.
    dc_lock: DcLockState,
    /// `D3DLOCK_*` flags captured by the most recent successful `LockRect`.
    ///
    /// Consumed by `UnlockRect`. Only meaningful for a lockable standalone
    /// render target: a `D3DLOCK_READONLY` lock must skip the staging→GPU
    /// upload on unlock (the staging was filled by a read-back, never written,
    /// so re-uploading it would clobber the rendered pixels). `0` otherwise.
    lock_flags: u32,
    /// Non-null for a cube-map face shell.
    ///
    /// The owning cube `Direct3DTexture9`, whose `TextureInner` carries the
    /// per-resource lock/DC state shared by all six faces (D3D9 gates the whole
    /// cube, not the individual face). The face holds a reference on it (taken
    /// by `GetCubeMapSurface`) so it outlives the face; the face's finalize
    /// releases it. Null for every other surface.
    state_owner_texture: *mut Direct3DTexture9,
}

/// The GDI objects backing an outstanding `GetDC`.
///
/// A memory DC and its DIB section, both produced by
/// `D3DKMTCreateDCFromMemory` so the DIB aliases the surface's own pixel store
/// directly (no separate GDI-allocated bits, no seed or write-back copy). Torn
/// down as a pair by `D3DKMTDestroyDCFromMemory`. All-null when no DC is held.
#[derive(Clone, Copy)]
struct GdiDc {
    dc: *mut c_void,
    bitmap: *mut c_void,
}

impl GdiDc {
    const NULL: Self = Self {
        dc: core::ptr::null_mut(),
        bitmap: core::ptr::null_mut(),
    };
}

impl SurfaceInner {
    /// Raw pointer to the effective resource-wide `DcLockState` for this surface.
    ///
    /// The owning cube texture's shared state for a cube-map face, else this
    /// surface's own. Returned as a raw pointer so a caller can hold it
    /// alongside a borrow of this surface's own `mapped` flag (they live in
    /// separate allocations for a cube face, and the same one otherwise).
    fn dc_lock_ptr(&mut self) -> *mut DcLockState {
        if self.state_owner_texture.is_null() {
            return &raw mut self.dc_lock;
        }
        // SAFETY: `state_owner_texture` is the owning cube `Direct3DTexture9`,
        // kept alive by the reference this face holds (released only at the
        // face's finalize); D3D9 objects are single-threaded so the exclusive
        // borrow of its inner state is sound for this call.
        let tex = unsafe { &*self.state_owner_texture };
        tex.dc_lock_state_ptr()
    }

    /// `LockRect` state transition.
    ///
    /// `Err(hr)` rejects the lock with `hr` (the caller must not write the
    /// out-`D3DLOCKED_RECT`); `Ok(())` permits it and records the lock. A
    /// `GetDC` anywhere on the resource blocks it; so does this surface already
    /// being mapped (same-face double-lock).
    fn try_begin_lock(&mut self) -> Result<(), i32> {
        // A cube-face surface shares the level's lock with its parent cube
        // texture: if the level is already locked through the cube's own
        // `LockRect`, locking the face surface is INVALIDCALL. Cube faces carry
        // the parent only as `state_owner_texture` (parent_texture is null), so
        // the texture-level delegation in `surface_lock_rect` doesn't see it.
        if !self.state_owner_texture.is_null() {
            // SAFETY: `state_owner_texture` is the live owning cube texture set
            // by `set_cube_state_owner`; single-threaded access is sound.
            let cube = unsafe { &*self.state_owner_texture };
            if cube.inner().is_level_locked(self.mip_level as usize) {
                return Err(D3DERR_INVALIDCALL);
            }
        }
        let shared = self.dc_lock_ptr();
        // SAFETY: `shared` is the live resource-wide state (own field or the
        // owning cube texture's); single-threaded access makes the deref sound.
        let shared = unsafe { &mut *shared };
        if shared.dc_in_use || self.flags.contains(SurfaceFlags::MAPPED) {
            return Err(D3DERR_INVALIDCALL);
        }
        self.flags.insert(SurfaceFlags::MAPPED);
        shared.map_count += 1;
        Ok(())
    }

    /// `UnlockRect` state transition.
    ///
    /// Unmapping an unmapped surface is `INVALIDCALL` — except while a `GetDC`
    /// is outstanding on the resource, when it is a no-op `S_OK` (the D3D9
    /// behavior the conformance test asserts when unlocking the non-DC face of
    /// a cube).
    fn try_end_lock(&mut self) -> i32 {
        let shared = self.dc_lock_ptr();
        // SAFETY: see `try_begin_lock`.
        let shared = unsafe { &mut *shared };
        if !self.flags.contains(SurfaceFlags::MAPPED) {
            return if shared.dc_in_use {
                D3D_OK
            } else {
                D3DERR_INVALIDCALL
            };
        }
        self.flags.remove(SurfaceFlags::MAPPED);
        shared.map_count = shared.map_count.saturating_sub(1);
        D3D_OK
    }
}

impl SurfaceInner {
    /// Live Metal color handle.
    ///
    /// The device's current backbuffer for an implicit `Backbuffer` surface,
    /// else the stored snapshot. Implicit surfaces never observe a stale handle
    /// after a resize/`Reset` recreates the backbuffer.
    fn live_color_handle(&self) -> MetalHandle<MTLTextureKind> {
        if self.implicit_kind == ImplicitKind::Backbuffer && !self.device_inner.is_null() {
            // SAFETY: `device_inner` is the live owning device; it outlives its
            // child surfaces per D3D9 lifetime rules.
            unsafe { (*self.device_inner).backbuffer_handle() }
        } else {
            self.metal_color_handle
        }
    }

    /// Live Metal depth handle.
    ///
    /// The device's current depth-stencil for an implicit `DepthStencil`
    /// surface, else the stored snapshot.
    fn live_depth_handle(&self) -> MetalHandle<MTLTextureKind> {
        if self.implicit_kind == ImplicitKind::DepthStencil && !self.device_inner.is_null() {
            // SAFETY: `device_inner` is the live owning device (see above).
            unsafe { (*self.device_inner).depth_stencil_handle() }
        } else {
            self.metal_depth_handle
        }
    }

    /// Live surface width.
    ///
    /// The device's current backbuffer width for any implicit surface
    /// (backbuffer and auto depth-stencil share the device dimensions), else
    /// the stored snapshot.
    fn live_width(&self) -> u32 {
        if self.implicit_kind != ImplicitKind::None && !self.device_inner.is_null() {
            // SAFETY: `device_inner` is the live owning device (see above).
            unsafe { (*self.device_inner).backbuffer_width() }
        } else {
            self.standalone_width
        }
    }

    /// Live surface height — counterpart to [`Self::live_width`].
    fn live_height(&self) -> u32 {
        if self.implicit_kind != ImplicitKind::None && !self.device_inner.is_null() {
            // SAFETY: `device_inner` is the live owning device (see above).
            unsafe { (*self.device_inner).backbuffer_height() }
        } else {
            self.standalone_height
        }
    }

    /// Live surface format.
    ///
    /// The device's current depth-stencil format for an implicit
    /// `DepthStencil` surface, else the stored snapshot (the implicit
    /// `Backbuffer` keeps its pinned `D3DFMT_X8R8G8B8`).
    fn live_format(&self) -> u32 {
        if self.implicit_kind == ImplicitKind::DepthStencil && !self.device_inner.is_null() {
            // SAFETY: `device_inner` is the live owning device (see above).
            unsafe { (*self.device_inner).depth_stencil_format() }
        } else {
            self.standalone_format
        }
    }

    /// The owning device's `Direct3DDevice9`* wrapper, or null when unset.
    ///
    /// The forward target for a device-owned implicit surface's refcount.
    fn device_wrapper(&self) -> *mut c_void {
        if self.device_inner.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: `device_inner` is the live owning device (see above).
        unsafe { (*self.device_inner).device_wrapper() }
    }
}

// ── IUnknown ──

#[inline]
fn surf_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DSurface9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::Surface)
}

extern "system" fn surface_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable in-param; `riid` is *const Guid per IUnknown::QueryInterface ABI.
    let riid_lo = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).map_or(0, |g| g.data1);
    trace!(target: LOG_TARGET, "IDirect3DSurface9::QueryInterface(riid_lo={riid_lo:#010x})");
    null_out(ppv);
    E_NOINTERFACE
}

/// The parent texture a `GetSurfaceLevel` sub-surface forwards its public refcount to.
///
/// The D3D9 "texture and surface refcount are identical" container model —
/// Wine `d3d9_surface_AddRef`/`Release` forward to `surface->texture`. Null
/// when this surface is not such a sub-surface (a standalone RT/DS, an owned
/// offscreen-plain, a system-memory, or a device-owned implicit surface). Those
/// use the central engine instead.
///
/// # Safety
/// `this` is a live `Direct3DSurface9` wrapper.
unsafe fn container_forward_texture(this: *mut c_void) -> *mut Direct3DTexture9 {
    // SAFETY: live wrapper per the caller's contract.
    let obj = unsafe { &*this.cast::<Direct3DSurface9>() };
    let inner = obj.inner();
    if inner.implicit_kind == ImplicitKind::None
        && !inner.parent_texture.is_null()
        && !inner.flags.contains(SurfaceFlags::OWNS_PARENT_TEXTURE)
    {
        inner.parent_texture
    } else {
        core::ptr::null_mut()
    }
}

extern "system" fn surface_add_ref(this: *mut c_void) -> u32 {
    let _timer = surf_timer(this);
    // SAFETY: IDirect3DSurface9 AddRef thunk; `this` is the live wrapper.
    let tex = unsafe { container_forward_texture(this) };
    if tex.is_null() {
        // SAFETY: a standalone/implicit surface — the central engine forwards
        // the device reference on a device-owned implicit surface's 0→1 edge.
        return unsafe { crate::com_ref::com_add_ref::<Direct3DSurface9>(this) };
    }
    // A texture sub-surface: forward the public AddRef to the container texture
    // (so the shared count the test observes is the texture's), and bump our own
    // refcount so the surface Box is freed when its last app/bound ref drops —
    // the texture references are released in lockstep in `surface_release`.
    {
        // SAFETY: live wrapper; the bump is a plain field write after the borrow.
        let mut wrap = unsafe { VtableThis::<Direct3DSurface9>::new(this) };
        wrap.refcount += 1;
    }
    // SAFETY: `tex` is the live parent texture; forward to its AddRef thunk,
    // whose return is the shared (texture) refcount D3D9 reports for the surface.
    unsafe { crate::com_ref::com_add_ref::<Direct3DTexture9>(tex.cast::<c_void>()) }
}

extern "system" fn surface_release(this: *mut c_void) -> u32 {
    let _timer = surf_timer(this);
    // SAFETY: IDirect3DSurface9 Release thunk; `this` is the live wrapper.
    let tex = unsafe { container_forward_texture(this) };
    if tex.is_null() {
        // SAFETY: a standalone/implicit surface — the central engine finalizes it
        // (or forwards the device release for a device-owned implicit surface).
        return unsafe { crate::com_ref::com_release::<Direct3DSurface9>(this) };
    }
    // A texture sub-surface: forward the public Release to the container texture
    // first (its return is the shared count we report), then drop our own
    // refcount and free the surface Box once no app/bound reference remains.
    // SAFETY: `tex` is the live parent texture; forward to its Release thunk.
    let rc = unsafe { crate::com_ref::com_release::<Direct3DTexture9>(tex.cast::<c_void>()) };
    let finalize_now = {
        // SAFETY: the surface Box is a separate allocation from the texture
        // (which the forward above may have freed); `this` is still live.
        let mut wrap = unsafe { VtableThis::<Direct3DSurface9>::new(this) };
        wrap.refcount -= 1;
        wrap.refcount == 0 && wrap.private_refcount == 0
    };
    if finalize_now {
        // SAFETY: both surface counters are zero — no other reference survives;
        // a sub-surface does not own the texture, so this only frees the Box.
        unsafe { finalize_surface(this.cast::<Direct3DSurface9>()) };
    }
    rc
}

/// Destroy a `Direct3DSurface9` wrapper once `refcount` and `private_refcount` reach zero.
///
/// Frees the inner + outer allocations; no encoder-thread or registry
/// interaction is required.
///
/// # Safety
/// `this` must point to a live `Direct3DSurface9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_surface(this: *mut Direct3DSurface9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(SurfaceInner)` from `Self::new` and no other
    // reference can survive.
    let inner = unsafe { Box::from_raw(inner_ptr) };
    // A device-owned implicit surface must reach this point ONLY via
    // `finalize_implicit_surface` at device teardown (which clears `implicit_kind`
    // first). Any other path here with a live `implicit_kind` is a gating bug
    // that would free a surface the device still hands out — fail loudly.
    debug_assert!(
        inner.implicit_kind == ImplicitKind::None,
        "device-owned implicit surface finalized outside device teardown"
    );
    let owner_tex = inner.state_owner_texture;
    if owner_tex.is_null() {
        // A standalone surface released while a `GetDC` is still outstanding
        // (the app skipped `ReleaseDC`) would leak its memory DC + DIB; tear
        // them down. (A cube face's DC lives on the shared cube state, torn
        // down when the cube texture itself finalizes — the cube outlives every
        // face that references it, so no face frees the shared DC here.)
        teardown_gdi_dc(inner.dc_lock.held_dc);
    } else {
        // SAFETY: `owner_tex` is the cube `Direct3DTexture9` whose reference
        // this face took at `GetCubeMapSurface`; drop it to balance that AddRef.
        let release = unsafe { (*owner_tex).vtbl().release };
        // SAFETY: `release` is the texture's IUnknown::Release thunk.
        unsafe { release(owner_tex.cast::<c_void>()) };
    }
    // Release the internal texture an owned offscreen-plain surface holds. The
    // surface took over the texture's single create-ref, so this drops it to
    // zero and finalizes the texture (its Metal handle + staging teardown).
    if inner.flags.contains(SurfaceFlags::OWNS_PARENT_TEXTURE) && !inner.parent_texture.is_null() {
        // SAFETY: `parent_texture` is the live internal `Direct3DTexture9` this
        // surface owns; calling its Release thunk consumes the owned ref.
        let release = unsafe { (*inner.parent_texture).vtbl().release };
        // SAFETY: `release` is the texture's IUnknown::Release; `parent_texture`
        // is a valid `*mut Direct3DTexture9` per the ABI.
        unsafe { release(inner.parent_texture.cast::<c_void>()) };
    }
    drop(inner);
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DSurface9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

/// Finalize a device-owned implicit surface at device teardown.
///
/// Drop its `SurfaceInner` — which releases any registered `D3DSPD_IUNKNOWN`
/// private-data object via `PrivateDataStore` — and free the wrapper shell.
/// Device-owned implicit surfaces are never finalized by `surface_release`
/// (they outlive the app's `Release`), so this is the single finalize site,
/// called once from `device_release` per cached surface.
///
/// # Safety
/// `ptr` must be `0` or a live `*mut Direct3DSurface9` produced by
/// `new_implicit_backbuffer`/`new_implicit_depth_stencil`; after this returns the
/// pointer is dangling and must not be used again.
pub unsafe fn finalize_implicit_surface(ptr: u64) {
    if ptr == 0 {
        return;
    }
    let surf = ptr as *mut Direct3DSurface9;
    // Mark this surface as no longer device-owned before finalizing: teardown is
    // the one legitimate finalize of an implicit surface, so clearing the kind
    // lets `finalize_surface`'s invariant assert catch every OTHER path.
    // SAFETY: `surf` is the live cached implicit surface wrapper.
    let inner_ptr = unsafe { (*surf).inner };
    // SAFETY: `inner_ptr` is its valid `SurfaceInner` for the duration of teardown.
    unsafe { (*inner_ptr).implicit_kind = ImplicitKind::None };
    // SAFETY: the caller guarantees `ptr` is a live device-owned implicit surface
    // wrapper; D3D9 is single-threaded so nothing else references it at teardown.
    unsafe { finalize_surface(surf) };
}

impl ComUnknown for Direct3DSurface9 {
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
        // Device-owned implicit surfaces are NEVER finalized by the private-ref
        // path (e.g. `bound_rt` unbinding the implicit RT) — they are destroyed
        // only at device teardown. Gating on `implicit_kind == None` keeps the
        // device's cached surface pointer valid (see `ImplicitKind`).
        if obj.refcount == 0
            && obj.private_refcount == 0
            && obj.inner().implicit_kind == ImplicitKind::None
        {
            // SAFETY: both counters reached zero — no other reference
            // can survive; finalize takes exclusive ownership.
            unsafe { finalize_surface(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees a standalone surface exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DSurface9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn private_refcount(&self) -> u32 {
        self.private_refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        let inner = self.inner();
        if inner.implicit_kind != ImplicitKind::None {
            // Device-owned implicit surface: forwards on its 0↔1 boundary
            // (D3D9 implicit-object model).
            return inner.device_wrapper();
        }
        if inner.flags.contains(SurfaceFlags::OWNS_PARENT_TEXTURE) {
            // Owned offscreen-plain surface: its internal texture forwards the
            // device reference, so the surface itself must not (no double-count).
            return core::ptr::null_mut();
        }
        if !inner.parent_texture.is_null() {
            // A `GetSurfaceLevel` sub-surface forwards to its container texture,
            // not the device (handled separately); creation does not register it.
            return core::ptr::null_mut();
        }
        // Standalone surface (render target / depth-stencil / system-memory):
        // forwards one device reference for its public lifetime.
        inner.device_wrapper()
    }
    fn finalizes_on_zero(&self) -> bool {
        // Implicit surfaces are never freed by `Release` — they are destroyed
        // only at device teardown (see `ImplicitKind`).
        self.inner().implicit_kind == ImplicitKind::None
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — both counters are zero and the
        // surface is standalone (`finalizes_on_zero()` true).
        unsafe { finalize_surface(this) };
    }
}

// ── IDirect3DResource9 stubs ──

extern "system" fn surface_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = surf_timer(this);
    trace!(target: LOG_TARGET, "IDirect3DSurface9::GetDevice()");
    if device.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        null_out(device);
        return D3DERR_INVALIDCALL;
    };
    let dev_inner = obj.inner().device_inner;
    if dev_inner.is_null() {
        null_out(device);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `dev_inner` is the live `DeviceInner` that created this surface;
    // the device outlives its child resources per D3D9 lifetime rules.
    let wrapper = unsafe { (*dev_inner).device_wrapper() };
    if wrapper.is_null() {
        null_out(device);
        return D3DERR_INVALIDCALL;
    }
    // AddRef per COM — the caller owns one reference on return.
    // SAFETY: `wrapper` is the live `Direct3DDevice9` COM object that owns
    // `dev_inner`; D3D9 objects are single-threaded, so the transient
    // exclusive borrow to bump the refcount is sound.
    unsafe { (*wrapper.cast::<Direct3DDevice9>()).add_ref_self() };
    // SAFETY: `device` is non-null (checked) and points to a writable
    // `*mut c_void` slot per the D3D9 ABI.
    unsafe { *device = wrapper };
    D3D_OK
}

extern "system" fn surface_set_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *const c_void,
    size: u32,
    flags: u32,
) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner_mut = obj.inner;
    // SAFETY: `inner_mut` lives until `surface_release` at refcount zero;
    // `InPtrMut` above guarantees exclusive access.
    let store = unsafe { &mut (*inner_mut).private_data };
    // SAFETY: `data`/`size`/`flags` are the caller-supplied private-data
    // payload per the D3D9 ABI; `set` validates them.
    unsafe { store.set(&guid, data, size, flags) }
}

extern "system" fn surface_get_private_data(
    this: *mut c_void,
    guid: *const Guid,
    data: *mut c_void,
    size: *mut u32,
) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `data`/`size` are the caller-owned out buffer + size slot per
    // the D3D9 ABI; the store validates the size before any copy.
    unsafe { obj.inner().private_data.get(&guid, data, size) }
}

extern "system" fn surface_free_private_data(this: *mut c_void, guid: *const Guid) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable in-param; `guid` is *const Guid per IDirect3DResource9 ABI.
    let Some(guid) = (unsafe { InPtr::<Guid>::opt(guid.cast()) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner_mut = obj.inner;
    // SAFETY: `inner_mut` lives until `surface_release`; `InPtrMut` guarantees
    // exclusive access.
    unsafe { (*inner_mut).private_data.free(&guid) }
}

extern "system" fn surface_set_priority(this: *mut c_void, _priority: u32) -> u32 {
    let _timer = surf_timer(this);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DSurface9::SetPriority: no Metal analog, no-op"
    );
    0
}

extern "system" fn surface_get_priority(this: *mut c_void) -> u32 {
    let _timer = surf_timer(this);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DSurface9::GetPriority: no Metal analog, no-op"
    );
    0
}

extern "system" fn surface_pre_load(this: *mut c_void) {
    let _timer = surf_timer(this);
    // See IDirect3DTexture9::PreLoad — Metal has no resident-set hint.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DSurface9::PreLoad: no Metal analog, no-op"
    );
}

extern "system" fn surface_get_type(this: *mut c_void) -> u32 {
    let _timer = surf_timer(this);
    trace!(target: LOG_TARGET, "IDirect3DSurface9::GetType()");
    1 // D3DRTYPE_SURFACE
}

// ── IDirect3DSurface9 ──

/// Minimal `IUnknown` layout for `AddRef`'ing a surface's container.
///
/// The container (the implicit swapchain or the device) is reached without
/// naming its full vtable type — every d3d9 wrapper is `repr(C)` with
/// `{query_interface, add_ref, release}` as its first three vtable entries, so
/// the shared prologue suffices.
#[repr(C)]
struct ContainerIUnknown {
    vtbl: *const ContainerIUnknownVtbl,
}
#[repr(C)]
struct ContainerIUnknownVtbl {
    _query_interface: extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    add_ref: extern "system" fn(*mut c_void) -> u32,
    _release: extern "system" fn(*mut c_void) -> u32,
}

extern "system" fn surface_get_container(
    this: *mut c_void,
    riid: *const Guid,
    container: *mut *mut c_void,
) -> i32 {
    let _timer = surf_timer(this);
    if container.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        null_out(container);
        return D3DERR_INVALIDCALL;
    };
    // A texture-level surface (from `GetSurfaceLevel`) reports its parent
    // texture as the container. `GetContainer` is `QueryInterface` against
    // that container, so it answers the texture's own interface IIDs and
    // returns `E_NOINTERFACE` for anything else (notably IID_IDirect3DSurface9
    // — a surface is not its own container). The internal texture behind an
    // *owned* surface (offscreen-plain DEFAULT) is private and not reported.
    let inner = obj.inner();
    if !inner.parent_texture.is_null() && !inner.flags.contains(SurfaceFlags::OWNS_PARENT_TEXTURE) {
        // SAFETY: `riid` is a *const Guid per the QueryInterface ABI.
        let matches_texture = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).is_some_and(|g| {
            let g = *g;
            g == IID_IUNKNOWN
                || g == IID_IDIRECT3DRESOURCE9
                || g == IID_IDIRECT3DBASETEXTURE9
                || g == IID_IDIRECT3DTEXTURE9
        });
        if !matches_texture {
            null_out(container);
            return E_NOINTERFACE;
        }
        let parent_ptr = inner.parent_texture.cast::<c_void>();
        // SAFETY: `parent_texture` is the live texture wrapper that owns this
        // level surface; its vtable's 2nd entry is `AddRef`.
        let unk = unsafe { &*(parent_ptr.cast::<ContainerIUnknown>()) };
        // SAFETY: `unk.vtbl` is the texture wrapper's `'static` vtable.
        let vtbl = unsafe { &*unk.vtbl };
        (vtbl.add_ref)(parent_ptr);
        // SAFETY: `container` is non-null (checked) and a writable out-pointer.
        unsafe { *container = parent_ptr };
        return D3D_OK;
    }
    let container_ptr = inner.container as *mut c_void;
    if container_ptr.is_null() {
        // Ordinary surfaces (texture-backed / standalone / system-memory) carry
        // no container. Only the device's implicit surfaces report one (the
        // implicit swapchain for the backbuffer, the device for depth-stencil).
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "IDirect3DSurface9::GetContainer on a surface with no container → INVALIDCALL");
        null_out(container);
        return D3DERR_INVALIDCALL;
    }
    // `GetContainer` returns an owned reference. We return the stored container
    // regardless of `riid` (the implicit swapchain answers IID_IDirect3DSwapChain9,
    // the device answers IID_IDirect3DDevice9 — the only riids the callers use)
    // and AddRef it via the shared IUnknown prologue.
    // SAFETY: `container_ptr` is the live swapchain/device wrapper stamped at the
    // implicit surface's creation; its vtable's 2nd entry is `AddRef`.
    let unk = unsafe { &*(container_ptr.cast::<ContainerIUnknown>()) };
    // SAFETY: `unk.vtbl` is the container wrapper's `'static` vtable.
    let vtbl = unsafe { &*unk.vtbl };
    (vtbl.add_ref)(container_ptr);
    // SAFETY: `container` is non-null (checked above) and per the D3D9 ABI points
    // to a writable out-pointer slot owned by the caller.
    unsafe { *container = container_ptr };
    D3D_OK
}

extern "system" fn surface_get_desc(this: *mut c_void, desc: *mut D3DSURFACE_DESC) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    if desc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `desc` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DSURFACE_DESC` slot owned by the caller.
    let out = unsafe { &mut *desc };
    if inner.parent_texture.is_null() {
        // Implicit surfaces resolve format + dims live from the device (see
        // `ImplicitKind`); ordinary standalone surfaces read their snapshot.
        out.format = inner.live_format();
        out.resource_type = 1; // D3DRTYPE_SURFACE
        out.usage = inner.standalone_usage;
        // A standalone surface reports its creation pool: RT / depth-stencil /
        // backbuffer / lockable-RT are `D3DPOOL_DEFAULT`; a system-memory
        // offscreen-plain surface reports the pool it was created with
        // (`D3DPOOL_SYSTEMMEM` or `D3DPOOL_SCRATCH`).
        out.pool = inner.standalone_pool;
        out.multi_sample_type = 0;
        out.multi_sample_quality = 0;
        out.width = inner.live_width();
        out.height = inner.live_height();
        return 0; // S_OK
    }
    // SAFETY: `inner.parent_texture` is non-null (checked above) and
    // points to a live `Direct3DTexture9` whose refcount keeps it alive
    // for as long as this surface is live.
    let tex = unsafe { &*inner.parent_texture };
    let ti = tex.inner();
    let level = inner.mip_level as usize;

    out.format = tex.d3d_format();
    out.resource_type = 1; // D3DRTYPE_SURFACE
    // A texture-level surface reports its parent texture's usage and pool, not
    // 0/0 — SetRenderTarget / SetDepthStencilSurface gate on them.
    out.usage = tex.d3d_usage();
    out.pool = tex.d3d_pool();
    out.multi_sample_type = 0;
    out.multi_sample_quality = 0;
    out.width = ti.mip_width(level);
    out.height = ti.mip_height(level);
    0 // S_OK
}

extern "system" fn surface_lock_rect(
    this: *mut c_void,
    locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    let _timer = surf_timer(this);
    if locked_rect.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // A surface lock and an outstanding GDI DC are mutually exclusive: reject a
    // `LockRect` while a `GetDC` is held anywhere on the resource (for a cube
    // map, on any face). The systemmem path additionally tracks the per-face
    // map flag in `systemmem_lock_rect` via `try_begin_lock`.
    // SAFETY: `obj.inner` is the live `SurfaceInner`; surfaces are
    // single-threaded so the transient exclusive borrow is sound.
    let inner_mut = unsafe { &mut *obj.inner };
    // SAFETY: `dc_lock_ptr` returns the live resource-wide state.
    if (unsafe { &*inner_mut.dc_lock_ptr() }).dc_in_use {
        return D3DERR_INVALIDCALL;
    }
    let parent_tex = obj.inner().parent_texture;
    if !parent_tex.is_null() {
        // D3D9 forbids LockRect on a non-lockable resource: a D3DPOOL_DEFAULT
        // texture-level surface that is not D3DUSAGE_DYNAMIC and not an
        // offscreen-plain surface (owns_parent_texture) lives in VRAM with no
        // CPU-mappable backing. MANAGED / SYSTEMMEM
        // / SCRATCH and DYNAMIC-DEFAULT textures remain lockable, as does the
        // offscreen-plain surface used by ColorFill / read-back.
        // SAFETY: `parent_tex` is a live `Direct3DTexture9` whose refcount keeps
        // it alive while this surface is live.
        let tex = unsafe { &*parent_tex };
        let (pool, usage) = (tex.d3d_pool(), tex.d3d_usage());
        if pool == D3DPOOL_DEFAULT
            && usage & D3DUSAGE_DYNAMIC == 0
            && !obj
                .inner()
                .flags
                .contains(SurfaceFlags::OWNS_PARENT_TEXTURE)
        {
            return D3DERR_INVALIDCALL;
        }
        mtld3d_shared::crumb!(
            "api:surf_lock",
            (u64::from(obj.inner().mip_level) << 32) | u64::from(flags),
        );
        // SAFETY: `parent_tex` is non-null (checked above) and points to
        // a live `Direct3DTexture9` whose refcount keeps it alive while
        // this surface is live.
        let tex_vtbl = unsafe { (*parent_tex).vtbl() };
        // SAFETY: calling the just-loaded `lock_rect` thunk through
        // `tex_vtbl` with the parent texture as `this`; `locked_rect`,
        // `rect`, `flags` are forwarded straight from the caller.
        return unsafe {
            (tex_vtbl.lock_rect)(
                parent_tex.cast::<c_void>(),
                obj.inner().mip_level,
                locked_rect,
                rect,
                flags,
            )
        };
    }
    // A lockable standalone render target (`CreateRenderTarget` with
    // `Lockable == TRUE`) carries BOTH a renderable colour handle AND a CPU
    // staging buffer. `LockRect` maps the staging (so the app can write source
    // pixels); `UnlockRect` uploads it to the colour texture. Checked before
    // the backbuffer read-back path, which only services read-only locks of a
    // staging-less colour surface.
    if !obj.inner().live_color_handle().is_null() {
        if obj.inner().system_memory.is_some() {
            return lockable_rt_lock_rect(&obj, locked_rect, rect, flags);
        }
        return backbuffer_lock_readback(&obj, locked_rect, rect, flags);
    }
    if obj.inner().system_memory.is_some() {
        return systemmem_lock_rect(&obj, locked_rect, rect);
    }
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
        "stub IDirect3DSurface9::LockRect on standalone non-color surface → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

extern "system" fn surface_unlock_rect(this: *mut c_void) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let parent_tex = obj.inner().parent_texture;
    if !parent_tex.is_null() {
        // An offscreen-plain D3DPOOL_DEFAULT surface (it owns its backing texture)
        // rejects an unlock-without-lock / double-unlock with INVALIDCALL, unlike
        // a regular GetSurfaceLevel texture surface which returns S_OK. Check
        // the backing texture's per-level lock state before forwarding the Unlock.
        if obj
            .inner()
            .flags
            .contains(SurfaceFlags::OWNS_PARENT_TEXTURE)
        {
            let level = obj.inner().mip_level as usize;
            // SAFETY: `parent_tex` is non-null (checked) and points to a live
            // `Direct3DTexture9` whose refcount keeps it alive while this surface
            // is live.
            if !unsafe { (*parent_tex).inner() }.is_level_locked(level) {
                return D3DERR_INVALIDCALL;
            }
        }
        mtld3d_shared::crumb!("api:surf_ulock", u64::from(obj.inner().mip_level));
        // SAFETY: `parent_tex` is non-null (checked above) and points to
        // a live `Direct3DTexture9` whose refcount keeps it alive while
        // this surface is live.
        let tex_vtbl = unsafe { (*parent_tex).vtbl() };
        // SAFETY: calling the just-loaded `unlock_rect` thunk through
        // `tex_vtbl` with the parent texture as `this`.
        return unsafe {
            (tex_vtbl.unlock_rect)(parent_tex.cast::<c_void>(), obj.inner().mip_level)
        };
    }
    mtld3d_shared::crumb!("api:surf_ulk_sa");
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj_mut) = (unsafe { InPtrMut::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner_mut = obj_mut.inner;
    // SAFETY: `inner_mut` was installed by `Self::new` as a `Box::into_raw`
    // and lives until `surface_release` at refcount zero; `InPtrMut` above
    // guarantees exclusive access.
    let inner = unsafe { &mut *inner_mut };
    if inner.system_memory.is_some() {
        // A lockable standalone render target also carries a renderable colour
        // handle: push the just-written staging up to that GPU texture before
        // closing the lock so `StretchRect`/sampling observe the new pixels.
        // (A staging-less colour surface never reaches here — its `UnlockRect`
        // falls through to the read-back drop below.) A `D3DLOCK_READONLY` lock
        // never wrote the staging (it only read the read-back), so uploading it
        // would clobber the rendered pixels with a stale copy — skip the upload.
        let read_only = inner.lock_flags & D3DLOCK_READONLY != 0;
        if inner.flags.contains(SurfaceFlags::MAPPED)
            && !read_only
            && !inner.live_color_handle().is_null()
        {
            lockable_rt_upload(inner);
        }
        // A system-memory surface (offscreen-plain or a cube-map face shell)
        // tracks the lock/DC state machine: a double-`UnlockRect` returns
        // INVALIDCALL, but unmapping an unmapped face while a `GetDC` is held on
        // the (shared cube) resource is a no-op success — see `try_end_lock`.
        return inner.try_end_lock();
    }
    // Standalone color surface (the backbuffer readback path). A successful
    // `LockRect` stashes `readback = Some(page)`, so `Some` ⇒ a real lock was
    // held ⇒ drop it and return S_OK (the performance-critical portrait
    // Lock→read→Unlock cycle). `None` ⇒ no successful Lock was outstanding (a
    // non-lockable `CreateRenderTarget`/`CreateDepthStencilSurface` surface whose
    // LockRect returns INVALIDCALL, or a double-Unlock) ⇒ D3D9 returns
    // INVALIDCALL.
    if inner.readback.take().is_some() {
        D3D_OK
    } else {
        D3DERR_INVALIDCALL
    }
}

/// Synchronous backbuffer readback.
///
/// Flush the in-progress frame to Metal so the most recent draws land in the
/// backbuffer texture, then blit the requested sub-rect into a page-aligned
/// PE-heap buffer that stays alive until `UnlockRect`. `WoW` uses this for
/// character portraits (a small sub-rect blit back to a game-side texture).
fn backbuffer_lock_readback(
    obj: &Direct3DSurface9,
    locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    const BPP: u32 = 4;
    let inner_ptr = obj.inner;
    // SAFETY: `inner_ptr` is the live `SurfaceInner` allocation for this
    // wrapper; surfaces are single-threaded objects in D3D9 so the
    // exclusive borrow is sound for the duration of this fn.
    let inner = unsafe { &mut *inner_ptr };

    // A backbuffer created with D3DPRESENTFLAG_LOCKABLE_BACKBUFFER accepts a
    // LockRect with any flags (e.g. D3DLOCK_DISCARD);
    // a non-lockable backbuffer rejects a non-READONLY lock. The portrait
    // read-back path (WoW) always locks D3DLOCK_READONLY, so it is unaffected
    // either way. The relaxation applies ONLY to the implicit backbuffer surface
    // (this fn also serves standalone non-lockable render targets, whose lock
    // rules are independent of the backbuffer's lockable flag).
    let lockable = inner.implicit_kind == ImplicitKind::Backbuffer
        && !inner.device_inner.is_null()
        // SAFETY: `device_inner` is non-null (checked) and points to the live
        // owning device, which outlives its child surfaces.
        && (unsafe { (*inner.device_inner).present_params() }.flags
            & D3DPRESENTFLAG_LOCKABLE_BACKBUFFER)
            != 0;
    if flags & D3DLOCK_READONLY == 0 && !lockable {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "backbuffer LockRect without D3DLOCK_READONLY (flags={flags:#x}) on a non-lockable backbuffer → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }

    // Implicit backbuffer surface resolves its extent + Metal handle live from
    // the device (see `ImplicitKind`) — read before borrowing `device_inner`
    // below so the two derefs of the raw device pointer never overlap.
    let full_w = inner.live_width();
    let full_h = inner.live_height();
    let tex_handle = inner.live_color_handle();
    let (x, y, w, h) = parse_surface_rect(rect, full_w, full_h);
    if w == 0 || h == 0 {
        return D3DERR_INVALIDCALL;
    }

    // BGRA8 backbuffer → 4 bytes per pixel. Backbuffer format is
    // pinned to Bgra8Unorm by the CAMetalLayer allow-list, so 4 bytes
    // per pixel is fixed.
    let bytes_per_row = w.saturating_mul(BPP);
    let bytes = (bytes_per_row as usize).saturating_mul(h as usize);
    if bytes == 0 {
        return D3DERR_INVALIDCALL;
    }
    let mut page = PageBox::new_uninit(bytes);

    if inner.device_inner.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `inner.device_inner` was stamped at `Self::new` from a
    // live `DeviceInner`; non-null here, and the device outlives all
    // its child resources per D3D9 lifetime rules.
    let device_inner = unsafe { &mut *inner.device_inner };
    device_inner.flush_current_frame_blocking();

    let mut params = BlitTextureToBufferParams {
        queue_handle: device_inner.queue_handle(),
        device_handle: device_inner.device_handle(),
        tex_handle,
        dst_ptr: page.as_mut_ptr() as u64,
        dst_len: page.len() as u64,
        mip_level: 0,
        origin_x: x,
        origin_y: y,
        width: w,
        height: h,
        bytes_per_row,
        pad0: 0,
    };
    let status = unix_call(&mut params);
    if status != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "backbuffer LockRect: BlitTextureToBuffer failed status={status:#x} → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }

    // SAFETY: `locked_rect` is non-null (checked by the caller before
    // entry) and per the D3D9 ABI points to a writable `D3DLOCKED_RECT`
    // slot owned by the caller.
    let out = unsafe { &mut *locked_rect };
    out.pitch = bytes_per_row.cast_signed();
    out.bits = page.as_mut_ptr().cast::<c_void>();
    inner.readback = Some(page);
    D3D_OK
}

/// Read the FULL implicit backbuffer texture into a fresh PE-heap page.
///
/// The page is held on `inner.readback`, returning `(width, height,
/// bytes_per_row)` or `None` on failure. Used by `GetDC` on the backbuffer (the
/// `LockRect` portrait path has its own sub-rect variant above). The backbuffer
/// is pinned to `Bgra8Unorm` (4 bytes/pixel), whose byte order matches an
/// X8R8G8B8 DIB.
fn readback_full_backbuffer(inner: &mut SurfaceInner) -> Option<(u32, u32, u32)> {
    const BPP: u32 = 4;
    if inner.device_inner.is_null() {
        return None;
    }
    let w = inner.live_width();
    let h = inner.live_height();
    let tex_handle = inner.live_color_handle();
    if w == 0 || h == 0 || tex_handle.is_null() {
        return None;
    }
    let bytes_per_row = w.saturating_mul(BPP);
    let bytes = (bytes_per_row as usize).saturating_mul(h as usize);
    if bytes == 0 {
        return None;
    }
    let mut page = PageBox::new_uninit(bytes);
    // SAFETY: `device_inner` is non-null (checked) and points to the live owning
    // device, which outlives its child surfaces per D3D9 lifetime rules.
    let device_inner = unsafe { &mut *inner.device_inner };
    device_inner.flush_current_frame_blocking();
    let mut params = BlitTextureToBufferParams {
        queue_handle: device_inner.queue_handle(),
        device_handle: device_inner.device_handle(),
        tex_handle,
        dst_ptr: page.as_mut_ptr() as u64,
        dst_len: page.len() as u64,
        mip_level: 0,
        origin_x: 0,
        origin_y: 0,
        width: w,
        height: h,
        bytes_per_row,
        pad0: 0,
    };
    if unix_call(&mut params) != 0 {
        return None;
    }
    inner.readback = Some(page);
    Some((w, h, bytes_per_row))
}

/// `LockRect` for a system-memory offscreen surface.
///
/// Hand back a pointer into the persistent backing buffer (no GPU work). A
/// sub-rect offsets `bits`; `NULL` locks the whole surface. `pitch` is always
/// the full-surface stride.
fn systemmem_lock_rect(
    obj: &Direct3DSurface9,
    locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
) -> i32 {
    let inner_ptr = obj.inner;
    // SAFETY: `inner_ptr` is the live `SurfaceInner` for this wrapper; D3D9
    // surfaces are single-threaded, so the exclusive borrow is sound.
    let inner = unsafe { &mut *inner_ptr };
    if inner.system_memory.is_none() {
        return D3DERR_INVALIDCALL;
    }
    // Re-locking an already-mapped surface, or locking one while a `GetDC` is
    // outstanding on the resource, returns INVALIDCALL with the caller's
    // `D3DLOCKED_RECT` untouched — checked + committed before any out-param
    // write. For a cube-map face this consults the shared cube state, so a
    // `GetDC` on a sibling face also blocks (but a lock on a sibling does not).
    if let Err(hr) = inner.try_begin_lock() {
        return hr;
    }
    let full_w = inner.standalone_width;
    let Some(fmt) = mtld3d_core::format::map_d3d_format(inner.standalone_format) else {
        return D3DERR_INVALIDCALL;
    };
    // Standalone SYSTEMMEM/SCRATCH surfaces accept ANY lock rect and return
    // pBits at the RAW, unclamped origin `top*pitch + left*bpp` — the XP
    // accept-invalid behaviour. An out-of-bounds or negative rect yields a
    // pointer outside the allocation that the caller is not expected to
    // dereference; we compute it by integer arithmetic (no UB pointer `add`). A
    // NULL rect locks the whole surface (origin 0,0). (DEFAULT offscreen-plain
    // surfaces are texture-backed and validate strictly in `texture_lock_rect`.)
    // SAFETY: `rect` is the *const D3DRECT delivered by LockRect; null → None.
    let (left, top) = unsafe { ValueIn::<D3DRECT>::read_opt(rect) }
        .map_or((0, 0), |r| (r.x1 as isize, r.y1 as isize));
    // Block-compressed surfaces lock at block granularity: the pitch is the
    // bytes per row of blocks (`ceil(width/block_width) * block_bytes`) and a
    // sub-rect origin steps in whole blocks. Linear formats use the
    // bytes-per-pixel row stride, rounded up to a 4-byte boundary (D3D9 reports
    // a 4-byte-aligned pitch and the backing buffer is sized to match).
    let to_i = |v: u32| isize::try_from(v).unwrap_or(isize::MAX);
    let bpp = fmt.bytes_per_pixel();
    let (pitch, offset) = if bpp == 0 {
        let bw = fmt.block_width().max(1);
        let bh = fmt.block_height().max(1);
        let bb = fmt.block_bytes();
        let pitch = full_w.div_ceil(bw).saturating_mul(bb);
        // Origin steps in whole blocks: block row `top / bh`, block column `left / bw`.
        let offset = top.div_euclid(to_i(bh)) * to_i(pitch) + left.div_euclid(to_i(bw)) * to_i(bb);
        (pitch, offset)
    } else {
        let pitch = full_w.saturating_mul(bpp).next_multiple_of(4);
        let offset = top * to_i(pitch) + left * to_i(bpp);
        (pitch, offset)
    };
    let Some(page) = inner.system_memory.as_mut() else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `locked_rect` is non-null (checked by the caller before entry)
    // and per the D3D9 ABI points to a writable `D3DLOCKED_RECT`.
    let out = unsafe { &mut *locked_rect };
    out.pitch = pitch.cast_signed();
    // Integer pointer arithmetic: the raw offset may land outside the
    // allocation for an out-of-bounds rect, which is the documented
    // accept-invalid contract (the caller does not dereference it).
    out.bits = page.as_mut_ptr().wrapping_offset(offset).cast::<c_void>();
    D3D_OK
}

/// `LockRect` for a lockable standalone render target.
///
/// `CreateRenderTarget` with `Lockable == TRUE`: hand back a pointer into the
/// CPU staging buffer at the requested sub-rect, with a tight `width * bpp`
/// pitch. The staging was sized `width * height * bpp` at creation.
/// `UnlockRect` later uploads it to the renderable colour texture
/// (`lockable_rt_upload`). A `NULL` rect locks the whole surface (origin 0,0).
/// Mirrors the systemmem accept-any-rect contract, but the pitch is tight (no
/// 4-byte rounding) so the staging size and the GPU upload region agree
/// exactly.
///
/// Unless the lock is `D3DLOCK_DISCARD` (the app will overwrite every pixel),
/// the GPU colour texture is first read back into the staging so a read lock
/// (`D3DLOCK_READONLY`, or a default read/modify lock) observes the rendered /
/// blitted pixels rather than stale staging. The read-back reuses the same
/// `BlitTextureToBuffer` core as the backbuffer / `GetRenderTargetData` path.
fn lockable_rt_lock_rect(
    obj: &Direct3DSurface9,
    locked_rect: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    let inner_ptr = obj.inner;
    // SAFETY: `inner_ptr` is the live `SurfaceInner` for this wrapper; D3D9
    // surfaces are single-threaded, so the exclusive borrow is sound.
    let inner = unsafe { &mut *inner_ptr };
    if inner.system_memory.is_none() {
        return D3DERR_INVALIDCALL;
    }
    // Re-locking an already-mapped surface returns INVALIDCALL with the
    // caller's `D3DLOCKED_RECT` untouched (committed before any out-write).
    if let Err(hr) = inner.try_begin_lock() {
        return hr;
    }
    // Record the lock flags so `UnlockRect` can skip the staging→GPU upload for
    // a `D3DLOCK_READONLY` lock (which never wrote the staging).
    inner.lock_flags = flags;
    let Some(fmt) = mtld3d_core::format::map_d3d_format(inner.standalone_format) else {
        return D3DERR_INVALIDCALL;
    };
    let bpp = fmt.bytes_per_pixel();
    if bpp == 0 {
        // Lockable RTs are uncompressed colour formats; a compressed format
        // never reaches here (CreateRenderTarget rejects it).
        return D3DERR_INVALIDCALL;
    }
    // Tight pitch (no 4-byte rounding): the staging is `width * bpp` per row, so
    // a full-surface fill writes exactly `width * height * bpp` bytes — the
    // region the unlock upload then copies. A sub-rect origin steps in whole
    // pixels: `top * pitch + left * bpp`.
    let pitch = inner.standalone_width.saturating_mul(bpp);
    // A read (or read/modify) lock sees the current GPU content: sync the colour
    // texture into the staging before handing back the pointer. `D3DLOCK_DISCARD`
    // skips it — the app will overwrite the whole surface. The read-back fills
    // the full surface (origin 0,0, tight pitch) so any sub-rect pointer derived
    // below indexes valid bytes.
    if flags & D3DLOCK_DISCARD == 0 {
        lockable_rt_readback_fill(inner, bpp);
    }
    // SAFETY: `rect` is the *const D3DRECT delivered by LockRect; null → None.
    let (left, top) = unsafe { ValueIn::<D3DRECT>::read_opt(rect) }
        .map_or((0, 0), |r| (r.x1 as isize, r.y1 as isize));
    let to_i = |v: u32| isize::try_from(v).unwrap_or(isize::MAX);
    let offset = top * to_i(pitch) + left * to_i(bpp);
    let Some(page) = inner.system_memory.as_mut() else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `locked_rect` is non-null (checked by the caller before entry)
    // and per the D3D9 ABI points to a writable `D3DLOCKED_RECT`.
    let out = unsafe { &mut *locked_rect };
    out.pitch = pitch.cast_signed();
    // Integer pointer arithmetic: an out-of-bounds rect lands outside the
    // allocation (accept-invalid; the caller does not dereference it).
    out.bits = page.as_mut_ptr().wrapping_offset(offset).cast::<c_void>();
    D3D_OK
}

/// Read the lockable render target's colour `MTLTexture` back into its CPU staging buffer.
///
/// The read-half of a non-discard `LockRect`. Flushes the in-progress frame so
/// the latest draws / `StretchRect` land in the texture, then blits the full
/// surface into the tight-pitch staging via the same `BlitTextureToBuffer` core
/// the backbuffer / `GetRenderTargetData` path uses. Marks the texture
/// read-back BEFORE the flush so the store-action optimiser (Rule D) keeps the
/// rendered content. A blit failure leaves the staging as-is (the zero-init /
/// prior content) — the lock still succeeds.
fn lockable_rt_readback_fill(inner: &mut SurfaceInner, bpp: u32) {
    let (width, height) = (inner.standalone_width, inner.standalone_height);
    let tex_handle = inner.live_color_handle();
    if bpp == 0 || width == 0 || height == 0 || tex_handle.is_null() {
        return;
    }
    let bytes_per_row = width.saturating_mul(bpp);
    let needed = (bytes_per_row as usize).saturating_mul(height as usize);
    if needed == 0 {
        return;
    }
    let device_ptr = inner.device_inner;
    if device_ptr.is_null() {
        return;
    }
    // Borrow the staging (the blit destination) and the device separately;
    // `device_inner` is a distinct allocation from the `system_memory` PageBox,
    // so the two raw-pointer derefs never overlap.
    let Some(page) = inner.system_memory.as_mut() else {
        return;
    };
    if page.len() < needed {
        return;
    }
    let dst_ptr = page.as_mut_ptr() as u64;
    let dst_len = page.len() as u64;
    // SAFETY: `device_ptr` was stamped at `Self::new` from a live `DeviceInner`;
    // non-null here, and the device outlives all its child resources per D3D9
    // lifetime rules. It is a different allocation from `page` above.
    let device_inner = unsafe { &mut *device_ptr };
    // The store-action optimiser runs at flush time and would discard this RT's
    // colour store when nothing samples it in-frame (Rule D) — but this blit
    // reads it right after. Mark it read-back BEFORE the flush so
    // `finalize_store_actions` keeps the rendered content.
    device_inner.push_op(Box::new(move |enc| enc.note_color_read_back(tex_handle)));
    device_inner.flush_current_frame_blocking();
    let mut params = BlitTextureToBufferParams {
        queue_handle: device_inner.queue_handle(),
        device_handle: device_inner.device_handle(),
        tex_handle,
        dst_ptr,
        dst_len,
        mip_level: 0,
        origin_x: 0,
        origin_y: 0,
        width,
        height,
        bytes_per_row,
        pad0: 0,
    };
    let status = unix_call(&mut params);
    if status != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "lockable RT LockRect read-back: BlitTextureToBuffer failed status={status:#x} (staging left as-is)"
        );
    }
}

/// `UnlockRect` upload for a lockable standalone render target.
///
/// Push the CPU staging buffer up to the renderable colour `MTLTexture` so a
/// subsequent `StretchRect` / sample observes the just-written pixels. The
/// staging bytes are *copied* into the pushed encoder op (a `Vec<u8>` the
/// closure owns) so the surface's `PageBox` is never aliased across the
/// API/encoder boundary.
fn lockable_rt_upload(inner: &mut SurfaceInner) {
    let Some(fmt) = mtld3d_core::format::map_d3d_format(inner.standalone_format) else {
        return;
    };
    let bpp = fmt.bytes_per_pixel();
    let (width, height) = (inner.standalone_width, inner.standalone_height);
    let color_handle = inner.live_color_handle().raw();
    if bpp == 0 || width == 0 || height == 0 || color_handle == 0 {
        return;
    }
    let Some(page) = inner.system_memory.as_ref() else {
        return;
    };
    let needed = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(bpp as usize);
    if page.len() < needed {
        return;
    }
    // Copy the tight `width*height*bpp` bytes the lock just received into a
    // heap buffer the op owns — the encoder thread reads it long after this
    // returns, so it must not borrow the surface's staging (no-thunk rule).
    let bytes: Vec<u8> = page.as_slice()[..needed].to_vec();
    if inner.device_inner.is_null() {
        return;
    }
    // SAFETY: `inner.device_inner` was stamped at `Self::new` from a live
    // `DeviceInner`; non-null here, and the device outlives all its child
    // resources per D3D9 lifetime rules.
    let device_inner = unsafe { &mut *inner.device_inner };
    device_inner.push_op(Box::new(move |enc| {
        enc.upload_bytes_to_color_handle(color_handle, &bytes, width, height, bpp);
    }));
}

/// Parse a `RECT*` pointer passed to `LockRect` and clamp it against `(full_w, full_h)`.
///
/// `NULL` means "full surface". Returns `(x, y, w, h)` with
/// `w == 0 || h == 0` indicating a zero-area (empty) rect the caller should
/// treat as `INVALIDCALL`.
fn parse_surface_rect(rect: *const c_void, full_w: u32, full_h: u32) -> (u32, u32, u32, u32) {
    // SAFETY: vtable in-param; `rect` is *const D3DRECT per ABI.
    let Some(r) = (unsafe { ValueIn::<D3DRECT>::read_opt(rect) }) else {
        return (0, 0, full_w, full_h);
    };
    let x1 = r.x1.max(0).cast_unsigned();
    let y1 = r.y1.max(0).cast_unsigned();
    let x2 = r.x2.max(0).cast_unsigned();
    let y2 = r.y2.max(0).cast_unsigned();
    let x = x1.min(full_w);
    let y = y1.min(full_h);
    let right = x2.min(full_w);
    let bottom = y2.min(full_h);
    if right <= x || bottom <= y {
        return (0, 0, 0, 0);
    }
    (x, y, right - x, bottom - y)
}

// ── GDI FFI (IDirect3DSurface9::GetDC / ReleaseDC over a memory-backed DC) ──

#[link(name = "gdi32")]
unsafe extern "system" {
    fn CreateCompatibleDC(dc: *mut c_void) -> *mut c_void;
    fn DeleteDC(dc: *mut c_void) -> i32;
    fn D3DKMTCreateDCFromMemory(desc: *mut D3DKMT_CREATEDCFROMMEMORY) -> u32;
    fn D3DKMTDestroyDCFromMemory(desc: *const D3DKMT_DESTROYDCFROMMEMORY) -> u32;
}

/// Win32 `D3DKMT_CREATEDCFROMMEMORY`: wraps an existing pixel store, no copy.
///
/// The pixel store is wrapped in a memory DC + DIB section. Unlike
/// `CreateDIBSection`, the kernel side (`NtGdiDdDDICreateDCFromMemory`) leaves
/// `biSizeImage` 0 and stores `biHeight` as `-Height` (top-down), which
/// `GetObject` flips back to `+Height` — matching the `DIBSECTION` values GDI
/// reports for such a DIB. `#[repr(C)]` matches the documented struct; `h_dc` /
/// `h_bitmap` are filled by the call.
// The field names mirror the canonical Win32 SDK names verbatim.
#[repr(C)]
struct D3DKMT_CREATEDCFROMMEMORY {
    p_memory: *mut c_void,
    format: u32,
    width: u32,
    height: u32,
    pitch: u32,
    h_device_dc: *mut c_void,
    p_color_table: *const c_void,
    h_dc: *mut c_void,
    h_bitmap: *mut c_void,
}

/// Win32 `D3DKMT_DESTROYDCFROMMEMORY`: the teardown counterpart.
///
/// Destroys the DC + DIB pair produced by [`D3DKMT_CREATEDCFROMMEMORY`].
#[repr(C)]
struct D3DKMT_DESTROYDCFROMMEMORY {
    h_dc: *mut c_void,
    h_bitmap: *mut c_void,
}

/// Per-format GDI mapping for `GetDC`.
///
/// The DIB bit count and, for 16-bit `BI_BITFIELDS` formats, the R/G/B channel
/// masks. `None` for formats D3D9 does not expose a GDI DC for (those reject
/// `GetDC` with `INVALIDCALL`).
const fn dc_format_info(d3d_format: u32) -> Option<(u16, Option<[u32; 3]>)> {
    match d3d_format {
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 => Some((32, None)),
        D3DFMT_R8G8B8 => Some((24, None)),
        D3DFMT_R5G6B5 => Some((16, Some([0x0000_f800, 0x0000_07e0, 0x0000_001f]))),
        // X1R5G5B5 and A1R5G5B5 share the GDI 5-5-5 mask set; the alpha /
        // padding bit is not represented in the DIB masks.
        D3DFMT_X1R5G5B5 | D3DFMT_A1R5G5B5 => {
            Some((16, Some([0x0000_7c00, 0x0000_03e0, 0x0000_001f])))
        }
        _ => None,
    }
}

/// Pixel store + geometry a `GetDC` reads/writes.
///
/// The surface's CPU pixels, its row pitch, and its dimensions. Resolved from
/// the system-memory backing (offscreen-plain + cube-face shells) or the parent
/// texture's level-0 staging (a `D3DPOOL_MANAGED` texture's `GetSurfaceLevel`
/// surface).
struct SurfacePixels {
    bits: *mut u8,
    width: u32,
    height: u32,
    src_pitch: usize,
    d3d_format: u32,
}

impl SurfaceInner {
    /// True for the implicit backbuffer surface only when it is lockable.
    ///
    /// That is, when the backbuffer was created/Reset with
    /// `D3DPRESENTFLAG_LOCKABLE_BACKBUFFER`. Mirrors the gate in
    /// [`backbuffer_lock_readback`]; `LockRect`/`GetDC` on a non-lockable
    /// backbuffer are rejected.
    fn is_lockable_backbuffer(&self) -> bool {
        self.implicit_kind == ImplicitKind::Backbuffer
            && !self.device_inner.is_null()
            // SAFETY: `device_inner` is non-null (checked) and points to the live
            // owning device, which outlives its child surfaces.
            && (unsafe { (*self.device_inner).present_params() }.flags
                & D3DPRESENTFLAG_LOCKABLE_BACKBUFFER)
                != 0
    }

    /// Resolve the CPU pixel store this surface exposes through `GetDC`.
    ///
    /// `None` for a surface kind with no host-readable pixels (a GPU-only
    /// render target / depth-stencil / backbuffer).
    fn dc_pixels(&self) -> Option<SurfacePixels> {
        if self.system_memory.is_some() {
            let format = self.standalone_format;
            let bpp = mtld3d_core::format::map_d3d_format(format)?.bytes_per_pixel();
            if bpp == 0 {
                // Block-compressed surfaces have no GDI representation.
                return None;
            }
            let width = self.standalone_width;
            let height = self.standalone_height;
            // The backing buffer pitch matches the DIB's `bmWidthBytes`:
            // `width * bpp` rounded up to a 4-byte boundary.
            let src_pitch = (width.saturating_mul(bpp).next_multiple_of(4)) as usize;
            // SAFETY: `system_memory` is `Some` (checked above); the `PageBox`
            // pointer stays valid for the surface's lifetime and is only read
            // through this borrow on the single-threaded API thread.
            let bits = self
                .system_memory
                .as_ref()
                .map_or(core::ptr::null_mut(), |p| p.as_ptr().cast_mut());
            return Some(SurfacePixels {
                bits,
                width,
                height,
                src_pitch,
                d3d_format: format,
            });
        }
        // The implicit backbuffer is a GPU-only colour surface with no
        // persistent CPU store. `surface_get_dc` performs a full read-back into
        // `self.readback` immediately before this call, so a held DC reads the
        // backbuffer's current pixels (BGRA8, matching the X8R8G8B8 DIB order).
        if self.implicit_kind == ImplicitKind::Backbuffer {
            let page = self.readback.as_ref()?;
            let format = self.live_format();
            let bpp = mtld3d_core::format::map_d3d_format(format)?.bytes_per_pixel();
            if bpp == 0 {
                return None;
            }
            let width = self.live_width();
            return Some(SurfacePixels {
                bits: page.as_ptr().cast_mut(),
                width,
                height: self.live_height(),
                src_pitch: width.saturating_mul(bpp) as usize,
                d3d_format: format,
            });
        }
        if self.parent_texture.is_null() || self.flags.contains(SurfaceFlags::OWNS_PARENT_TEXTURE) {
            return None;
        }
        // A texture sub-surface (`GetSurfaceLevel`): the pixels live in the
        // parent texture's per-level CPU staging. `D3DPOOL_MANAGED` textures
        // keep a permanent staging copy, which is what `GetDC` exposes.
        // SAFETY: `parent_texture` is non-null (checked above) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive for as long as
        // this surface is live.
        let tex = unsafe { &*self.parent_texture };
        let level = self.mip_level as usize;
        let ti = tex.inner();
        let (bits, row_pitch, _slice_pitch) = ti.lock_box(level)?;
        Some(SurfacePixels {
            bits,
            width: ti.mip_width(level),
            height: ti.mip_height(level),
            src_pitch: usize::try_from(row_pitch).unwrap_or(0),
            d3d_format: tex.d3d_format(),
        })
    }
}

extern "system" fn surface_get_dc(this: *mut c_void, hdc: *mut *mut c_void) -> i32 {
    let _timer = surf_timer(this);
    if hdc.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner_ptr = obj.inner;
    // SAFETY: `inner_ptr` is the live `SurfaceInner` for this wrapper; D3D9
    // surfaces are single-threaded, so the exclusive borrow is sound.
    let inner = unsafe { &mut *inner_ptr };

    // A failed `GetDC` must leave the caller's out-`HDC` slot untouched, so
    // every reject below returns before writing through `hdc`.
    // `GetDC` is INVALIDCALL while any lock OR a DC is outstanding on the
    // resource (`map_count != 0`). For a cube-map face this is the shared cube
    // map count, so a lock on any sibling face blocks it too.
    // SAFETY: `dc_lock_ptr` returns the live resource-wide state; read-only here.
    if (unsafe { &*inner.dc_lock_ptr() }).map_count != 0 {
        return D3DERR_INVALIDCALL;
    }
    // A LOCKABLE implicit backbuffer is GPU-only: snapshot it via a full
    // read-back into `readback` so `dc_pixels` (and the DIB it seeds) have host
    // pixels. A NON-lockable backbuffer rejects GetDC: the
    // `is_lockable_backbuffer` guard is false, so we skip the read-back and fall
    // through to `dc_pixels` → None → INVALIDCALL with the out-HDC left untouched.
    if inner.is_lockable_backbuffer() && readback_full_backbuffer(inner).is_none() {
        return D3DERR_INVALIDCALL;
    }
    let Some(px) = inner.dc_pixels() else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "IDirect3DSurface9::GetDC on a surface with no host pixel store → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    // Reject a format with no GDI mapping; the DIB layout itself is derived
    // from `format` by the kernel.
    if dc_format_info(px.d3d_format).is_none() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "IDirect3DSurface9::GetDC: format has no GDI mapping → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // The DC aliases the surface pixels directly, so there must be a live store.
    if px.bits.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let Ok(pitch) = u32::try_from(px.src_pitch) else {
        return D3DERR_INVALIDCALL;
    };

    // Wrap the surface's own pixel store in a memory DC + DIB via
    // `D3DKMTCreateDCFromMemory` — no copy, and (unlike `CreateDIBSection`) the
    // kernel leaves `biSizeImage` 0 and stores a negative top-down `biHeight`,
    // matching the `DIBSECTION` values GDI reports for such a DIB. `h_device_dc`
    // is a throwaway template DC deleted right after the call.
    // SAFETY: `CreateCompatibleDC(NULL)` is the documented form for a memory DC
    // compatible with the screen; null on failure is tolerated (template only —
    // `D3DKMTCreateDCFromMemory` reports the real failure via its status).
    let device_dc = unsafe { CreateCompatibleDC(core::ptr::null_mut()) };
    let mut desc = D3DKMT_CREATEDCFROMMEMORY {
        p_memory: px.bits.cast::<c_void>(),
        format: px.d3d_format,
        width: px.width,
        height: px.height,
        pitch,
        h_device_dc: device_dc,
        p_color_table: core::ptr::null(),
        h_dc: core::ptr::null_mut(),
        h_bitmap: core::ptr::null_mut(),
    };
    // SAFETY: `&mut desc` is a live, aligned `D3DKMT_CREATEDCFROMMEMORY`; the call
    // fills `h_dc` / `h_bitmap` on success and returns a non-zero NTSTATUS on
    // failure. `p_memory` stays valid for the DC's lifetime (surface store).
    let status = unsafe { D3DKMTCreateDCFromMemory(&raw mut desc) };
    if !device_dc.is_null() {
        // SAFETY: `device_dc` is the live template memory DC created above; it has
        // served its purpose (the created DC is `desc.h_dc`) and is deleted here.
        unsafe { DeleteDC(device_dc) };
    }
    if status != 0 || desc.h_dc.is_null() {
        return D3DERR_INVALIDCALL;
    }

    // SAFETY: `dc_lock_ptr` returns the live resource-wide state; the prior
    // immutable borrow above ended, so this exclusive deref does not alias it.
    let dc_lock = unsafe { &mut *inner.dc_lock_ptr() };
    dc_lock.held_dc = GdiDc {
        dc: desc.h_dc,
        bitmap: desc.h_bitmap,
    };
    dc_lock.dc_in_use = true;
    // A `GetDC` counts toward the resource map count (per D3D9), so a
    // subsequent `GetDC`/`LockRect` on any face is rejected until `ReleaseDC`.
    dc_lock.map_count += 1;
    // SAFETY: `hdc` is non-null (checked above) and per the D3D9 ABI points to a
    // writable `HDC` out-slot owned by the caller; written only on success.
    unsafe { *hdc = desc.h_dc };
    D3D_OK
}

extern "system" fn surface_release_dc(this: *mut c_void, hdc: *mut c_void) -> i32 {
    let _timer = surf_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSurface9 per IDirect3DSurface9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DSurface9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner_ptr = obj.inner;
    // SAFETY: `inner_ptr` is the live `SurfaceInner` for this wrapper; D3D9
    // surfaces are single-threaded, so the exclusive borrow is sound.
    let inner = unsafe { &mut *inner_ptr };

    // A `ReleaseDC` is valid only while a `GetDC` is outstanding, and only for
    // the exact `HDC` that `GetDC` returned. A second (stale) `ReleaseDC` of
    // the same now-deleted DC is INVALIDCALL. The state is the shared cube
    // state for a cube-map face; snapshot it, then release the borrow so the
    // immutable `dc_pixels` read below does not alias it.
    let held = {
        // SAFETY: `dc_lock_ptr` returns the live resource-wide state.
        let dc_lock = unsafe { &mut *inner.dc_lock_ptr() };
        if !dc_lock.dc_in_use || hdc != dc_lock.held_dc.dc {
            return D3DERR_INVALIDCALL;
        }
        dc_lock.held_dc
    };

    // The DC's DIB aliased the surface store directly, so any drawing the app
    // did is already in place — no write-back copy, just tear the GDI pair down.
    // A GetDC/ReleaseDC cycle records a source dirty rect (the app may have
    // drawn into the DC): a subsequent UpdateTexture from this surface's parent
    // texture must re-copy it.
    let parent_tex = inner.parent_texture;
    let level = inner.mip_level as usize;
    if !parent_tex.is_null() {
        // SAFETY: `parent_tex` is a live `Direct3DTexture9` (its refcount keeps
        // it alive while this surface is live); distinct from the surface inner.
        unsafe { (*parent_tex).inner_mut() }.mark_update_dirty(level, None);
    }
    teardown_gdi_dc(held);
    // A backbuffer GetDC stashed a full read-back snapshot in `readback`; drop it
    // now the DC is gone (the LockRect path reuses the same slot). No-op for
    // every other surface kind (their `readback` is already None here).
    inner.readback = None;
    // SAFETY: `dc_lock_ptr` returns the live resource-wide state; the prior
    // borrows ended, so this exclusive deref does not alias them.
    let dc_lock = unsafe { &mut *inner.dc_lock_ptr() };
    dc_lock.held_dc = GdiDc::NULL;
    dc_lock.dc_in_use = false;
    // The DC's contribution to the resource map count is released; no face can
    // be mapped while a DC is held, so the count returns to zero.
    dc_lock.map_count = dc_lock.map_count.saturating_sub(1);
    D3D_OK
}

/// Tear down the GDI objects of an outstanding `GetDC`.
///
/// Destroy the memory DC and its DIB section as a pair via
/// `D3DKMTDestroyDCFromMemory` (the counterpart to the
/// `D3DKMTCreateDCFromMemory` that produced them). A no-op on a [`GdiDc::NULL`]
/// (no DC held).
fn teardown_gdi_dc(held: GdiDc) {
    if held.dc.is_null() {
        return;
    }
    let desc = D3DKMT_DESTROYDCFROMMEMORY {
        h_dc: held.dc,
        h_bitmap: held.bitmap,
    };
    // SAFETY: `held.dc` + `held.bitmap` are the live DC / DIB pair produced by
    // `D3DKMTCreateDCFromMemory` at `GetDC`; `&desc` is a live, aligned struct
    // read only for the duration of the call.
    unsafe { D3DKMTDestroyDCFromMemory(&raw const desc) };
}
