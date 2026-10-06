//! The `IDirect3D9Ex` and `IDirect3DDevice9Ex` calls a [`Harness`] makes.
//!
//! These reach past the base vtables, so they are only valid on a harness
//! created with [`Factory::Extended`](super::Factory::Extended) or
//! [`Factory::ExtendedDeviceEx`](super::Factory::ExtendedDeviceEx); the
//! tests that call them create one first.

use core::{ffi::c_void, marker::PhantomData};

use mtld3d_types::{
    D3DDEVTYPE_HAL, D3DDISPLAYMODEEX, D3DDISPLAYMODEFILTER, D3DPRESENT_PARAMETERS,
    IDirect3D9ExVtbl, IDirect3DDevice9ExVtbl, LUID,
};

use super::{Harness, HarnessConfig};
use crate::{resource::Surface, vtbl::deref_vtbl};

/// A create's `pSharedHandle` argument: none, or a slot the call reads.
///
/// An extended device reads a slot holding a pointer to the level's pixels
/// as user memory, and a slot holding null as a request to share. The slot
/// is borrowed for as long as the argument exists.
pub struct SharedHandle<'a> {
    ptr: *mut c_void,
    slot: PhantomData<&'a mut *mut c_void>,
}

impl<'a> SharedHandle<'a> {
    /// No `pSharedHandle`: the null argument every base create takes.
    pub const NONE: SharedHandle<'static> = SharedHandle {
        ptr: core::ptr::null_mut(),
        slot: PhantomData,
    };

    /// A `pSharedHandle` naming `slot`: user memory when it holds a pointer, sharing when null.
    pub const fn to<T>(slot: &'a mut *mut T) -> Self {
        Self {
            ptr: core::ptr::from_mut(slot).cast::<c_void>(),
            slot: PhantomData,
        }
    }
}

/// A `D3DDISPLAYMODEEX` with every field zero but `size`.
#[must_use]
pub const fn display_mode_ex(size: u32) -> D3DDISPLAYMODEEX {
    D3DDISPLAYMODEEX {
        size,
        width: 0,
        height: 0,
        refresh_rate: 0,
        format: 0,
        scan_line_ordering: 0,
    }
}

/// `CreateDeviceEx` on adapter 0 with a null focus window and no display mode.
pub fn create_device_ex(
    d3d9: *mut c_void,
    behavior_flags: u32,
    pp: &mut D3DPRESENT_PARAMETERS,
    device: *mut *mut c_void,
) -> i32 {
    // SAFETY: `d3d9` is a live interface `Direct3DCreate9Ex` made.
    let vtbl = unsafe { deref_vtbl::<IDirect3D9ExVtbl>(d3d9) };
    // SAFETY: extended vtable thunk; `pp` and `device` are writable for the call.
    unsafe {
        (vtbl.create_device_ex)(
            d3d9,
            0,
            D3DDEVTYPE_HAL,
            core::ptr::null_mut(),
            behavior_flags,
            core::ptr::from_mut(pp).cast::<c_void>(),
            core::ptr::null_mut(),
            device,
        )
    }
}

impl Harness {
    fn factory_ex_vtbl(&self) -> &'static IDirect3D9ExVtbl {
        // SAFETY: an extended harness's interface carries the extended vtable.
        unsafe { deref_vtbl::<IDirect3D9ExVtbl>(self.d3d9) }
    }

    fn dev_ex_vtbl(&self) -> &'static IDirect3DDevice9ExVtbl {
        // SAFETY: an extended harness's device carries the extended vtable.
        unsafe { deref_vtbl::<IDirect3DDevice9ExVtbl>(self.device) }
    }

    // ── IDirect3D9Ex ──

    /// `GetAdapterLUID` on adapter 0, as `(hr, (LowPart, HighPart))`.
    #[must_use]
    pub fn adapter_luid(&self) -> (i32, (u32, i32)) {
        let mut luid = LUID {
            low_part: 0,
            high_part: 0,
        };
        // SAFETY: extended vtable thunk; `luid` is writable.
        let hr = unsafe { (self.factory_ex_vtbl().get_adapter_luid)(self.d3d9, 0, &raw mut luid) };
        (hr, (luid.low_part, luid.high_part))
    }

    /// `GetAdapterLUID` with a null out slot.
    #[must_use]
    pub fn adapter_luid_null_hr(&self) -> i32 {
        // SAFETY: extended vtable thunk; a null out slot is a rejected call.
        unsafe { (self.factory_ex_vtbl().get_adapter_luid)(self.d3d9, 0, core::ptr::null_mut()) }
    }

    /// `GetAdapterModeCountEx` on adapter 0 under `filter`, `None` for a null filter.
    #[must_use]
    pub fn adapter_mode_count_ex(&self, filter: Option<&D3DDISPLAYMODEFILTER>) -> u32 {
        let filter = filter.map_or(core::ptr::null(), core::ptr::from_ref);
        // SAFETY: extended vtable thunk; `filter` is null or a live filter.
        unsafe { (self.factory_ex_vtbl().get_adapter_mode_count_ex)(self.d3d9, 0, filter) }
    }

    /// `EnumAdapterModesEx` on adapter 0, returning the hr and the mode it wrote.
    #[must_use]
    pub fn enum_adapter_modes_ex(
        &self,
        filter: &D3DDISPLAYMODEFILTER,
        index: u32,
    ) -> (i32, D3DDISPLAYMODEEX) {
        let mut mode = display_mode_ex(0);
        // SAFETY: extended vtable thunk; `filter` is live and `mode` writable.
        let hr = unsafe {
            (self.factory_ex_vtbl().enum_adapter_modes_ex)(
                self.d3d9,
                0,
                core::ptr::from_ref(filter),
                index,
                &raw mut mode,
            )
        };
        (hr, mode)
    }

    /// `GetAdapterDisplayModeEx` on adapter 0 with `mode.Size = size`.
    ///
    /// Returns the hr, the mode and the rotation written, the rotation reading
    /// 0 when the call left it alone.
    #[must_use]
    pub fn adapter_display_mode_ex(&self, size: u32) -> (i32, D3DDISPLAYMODEEX, u32) {
        let mut mode = display_mode_ex(size);
        let mut rotation = 0u32;
        // SAFETY: extended vtable thunk; `mode` and `rotation` are writable.
        let hr = unsafe {
            (self.factory_ex_vtbl().get_adapter_display_mode_ex)(
                self.d3d9,
                0,
                &raw mut mode,
                &raw mut rotation,
            )
        };
        (hr, mode, rotation)
    }

    // ── IDirect3DDevice9Ex ──

    /// `PresentEx` of the whole back buffer with `flags`.
    pub fn present_ex(&self, flags: u32) -> i32 {
        // SAFETY: extended vtable thunk; null rects, window and region.
        unsafe {
            (self.dev_ex_vtbl().present_ex)(
                self.device,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null(),
                flags,
            )
        }
    }

    /// The windowed present parameters [`Harness::reset`] passes for a back buffer of this size.
    #[must_use]
    pub fn windowed_present_params(&self, width: u32, height: u32) -> D3DPRESENT_PARAMETERS {
        let cfg = HarnessConfig {
            width,
            height,
            back_buffer_format: self.back_buffer_format,
            depth_format: self.depth_format,
            windowed: 1,
            present_flags: self.present_flags,
            multi_sample_type: self.multi_sample_type,
            ..HarnessConfig::default()
        };
        super::present_params(&cfg, self.hwnd)
    }

    /// `ResetEx` with `pp` and an optional display mode.
    ///
    /// A successful one moves the harness's tracked back-buffer size to
    /// `pp`'s, as [`Harness::reset`] does.
    pub fn reset_ex(&self, pp: &mut D3DPRESENT_PARAMETERS, mode: Option<&D3DDISPLAYMODEEX>) -> i32 {
        if pp.windowed == 0 {
            self.hold_display_mode();
        }
        let mode = mode.map_or(core::ptr::null_mut(), |m| {
            core::ptr::from_ref(m).cast_mut().cast::<c_void>()
        });
        // SAFETY: extended vtable thunk; `pp` is writable and `mode` null or live.
        let hr = unsafe {
            (self.dev_ex_vtbl().reset_ex)(
                self.device,
                core::ptr::from_mut(pp).cast::<c_void>(),
                mode,
            )
        };
        if hr == 0 {
            self.width.set(pp.back_buffer_width);
            self.height.set(pp.back_buffer_height);
        }
        hr
    }

    /// `SetMaximumFrameLatency`.
    pub fn set_maximum_frame_latency(&self, latency: u32) -> i32 {
        // SAFETY: extended vtable thunk.
        unsafe { (self.dev_ex_vtbl().set_maximum_frame_latency)(self.device, latency) }
    }

    /// `GetMaximumFrameLatency`, returning the hr and the latency it wrote.
    #[must_use]
    pub fn maximum_frame_latency(&self) -> (i32, u32) {
        let mut latency = 0u32;
        // SAFETY: extended vtable thunk; `latency` is writable.
        let hr = unsafe {
            (self.dev_ex_vtbl().get_maximum_frame_latency)(self.device, &raw mut latency)
        };
        (hr, latency)
    }

    /// `CheckDeviceState` against the device window.
    #[must_use]
    pub fn check_device_state(&self) -> i32 {
        // SAFETY: extended vtable thunk; the window is the harness's own.
        unsafe { (self.dev_ex_vtbl().check_device_state)(self.device, self.hwnd as *mut c_void) }
    }

    /// The device's `GetDisplayModeEx` with `mode.Size = size`: hr, mode and rotation.
    #[must_use]
    pub fn device_display_mode_ex(
        &self,
        swap_chain: u32,
        size: u32,
    ) -> (i32, D3DDISPLAYMODEEX, u32) {
        let mut mode = display_mode_ex(size);
        let mut rotation = 0u32;
        // SAFETY: extended vtable thunk; `mode` and `rotation` are writable.
        let hr = unsafe {
            (self.dev_ex_vtbl().get_display_mode_ex)(
                self.device,
                swap_chain,
                (&raw mut mode).cast::<c_void>(),
                &raw mut rotation,
            )
        };
        (hr, mode, rotation)
    }

    /// `GetGPUThreadPriority`, returning the hr and the priority it wrote.
    #[must_use]
    pub fn gpu_thread_priority(&self) -> (i32, i32) {
        let mut priority = -1i32;
        // SAFETY: extended vtable thunk; `priority` is writable.
        let hr =
            unsafe { (self.dev_ex_vtbl().get_gpu_thread_priority)(self.device, &raw mut priority) };
        (hr, priority)
    }

    /// `SetGPUThreadPriority`.
    pub fn set_gpu_thread_priority(&self, priority: i32) -> i32 {
        // SAFETY: extended vtable thunk.
        unsafe { (self.dev_ex_vtbl().set_gpu_thread_priority)(self.device, priority) }
    }

    /// `WaitForVBlank`.
    pub fn wait_for_vblank(&self, swap_chain: u32) -> i32 {
        // SAFETY: extended vtable thunk.
        unsafe { (self.dev_ex_vtbl().wait_for_vblank)(self.device, swap_chain) }
    }

    /// `CheckResourceResidency` with no resources.
    #[must_use]
    pub fn check_resource_residency(&self) -> i32 {
        // SAFETY: extended vtable thunk; an empty array.
        unsafe {
            (self.dev_ex_vtbl().check_resource_residency)(self.device, core::ptr::null_mut(), 0)
        }
    }

    /// `SetConvolutionMonoKernel` with no kernel.
    #[must_use]
    pub fn set_convolution_mono_kernel(&self) -> i32 {
        // SAFETY: extended vtable thunk; null rows and columns.
        unsafe {
            (self.dev_ex_vtbl().set_convolution_mono_kernel)(
                self.device,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        }
    }

    /// `ComposeRects` with null surfaces and descriptor buffers.
    #[must_use]
    pub fn compose_rects(&self) -> i32 {
        // SAFETY: extended vtable thunk; the stub reads none of its arguments.
        unsafe {
            (self.dev_ex_vtbl().compose_rects)(
                self.device,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                1,
                0,
                0,
            )
        }
    }

    /// `CreateRenderTargetEx`, single-sampled and not lockable, with a usage and a shared handle.
    #[must_use]
    pub fn create_render_target_ex(
        &self,
        size: (u32, u32),
        format: u32,
        shared_handle: &SharedHandle<'_>,
        usage: u32,
    ) -> (i32, Option<Surface<'_>>) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: extended vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_ex_vtbl().create_render_target_ex)(
                self.device,
                size.0,
                size.1,
                format,
                0,
                0,
                0,
                &raw mut out,
                shared_handle.ptr,
                usage,
            )
        };
        (hr, (!out.is_null()).then(|| Surface::from_raw(out)))
    }

    /// `CreateOffscreenPlainSurfaceEx` with a usage and a shared handle.
    #[must_use]
    pub fn create_offscreen_plain_surface_ex(
        &self,
        size: (u32, u32),
        format: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
        usage: u32,
    ) -> (i32, Option<Surface<'_>>) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: extended vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_ex_vtbl().create_offscreen_plain_surface_ex)(
                self.device,
                size.0,
                size.1,
                format,
                pool,
                &raw mut out,
                shared_handle.ptr,
                usage,
            )
        };
        (hr, (!out.is_null()).then(|| Surface::from_raw(out)))
    }

    /// `CreateDepthStencilSurfaceEx`, single-sampled, with a usage and a shared handle.
    #[must_use]
    pub fn create_depth_stencil_surface_ex(
        &self,
        size: (u32, u32),
        format: u32,
        shared_handle: &SharedHandle<'_>,
        usage: u32,
    ) -> (i32, Option<Surface<'_>>) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: extended vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_ex_vtbl().create_depth_stencil_surface_ex)(
                self.device,
                size.0,
                size.1,
                format,
                0,
                0,
                0,
                &raw mut out,
                shared_handle.ptr,
                usage,
            )
        };
        (hr, (!out.is_null()).then(|| Surface::from_raw(out)))
    }

    /// `CreateDepthStencilSurfaceEx` with an out slot that starts on a sentinel.
    ///
    /// Returns the hr and whether a failing call left the slot as it was; a
    /// surface a successful call hands out is released.
    #[must_use]
    pub fn create_depth_stencil_surface_ex_slot(&self, usage: u32) -> (i32, bool) {
        let sentinel = core::ptr::without_provenance_mut::<c_void>(0xdead_beef);
        let mut out = sentinel;
        // SAFETY: extended vtable thunk; `out` is writable.
        let hr = unsafe {
            (self.dev_ex_vtbl().create_depth_stencil_surface_ex)(
                self.device,
                64,
                64,
                mtld3d_types::D3DFMT_D24S8,
                0,
                0,
                1,
                &raw mut out,
                core::ptr::null_mut(),
                usage,
            )
        };
        if hr == 0 {
            drop(Surface::from_raw(out));
            return (hr, false);
        }
        (hr, out == sentinel)
    }

    // ── pSharedHandle on the base creates ──

    /// `CreateTexture` with a `pSharedHandle`, returning the hr and the texture.
    ///
    /// A non-null handle reaching an extended device's single-level
    /// system-memory texture is a pointer to the pointer of its pixels.
    #[must_use]
    pub fn try_create_texture_shared(
        &self,
        size: (u32, u32),
        levels: u32,
        format: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_texture)(
                self.device,
                size.0,
                size.1,
                levels,
                0,
                format,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateCubeTexture` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_cube_texture_shared(
        &self,
        edge: u32,
        format: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_cube_texture)(
                self.device,
                edge,
                1,
                0,
                format,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateVolumeTexture` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_volume_texture_shared(
        &self,
        extent: [u32; 3],
        format: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_volume_texture)(
                self.device,
                extent[0],
                extent[1],
                extent[2],
                1,
                0,
                format,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateVertexBuffer` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_vertex_buffer_shared(
        &self,
        length: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_vertex_buffer)(
                self.device,
                length,
                0,
                0,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateIndexBuffer` (16-bit indices) with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_index_buffer_shared(
        &self,
        length: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_index_buffer)(
                self.device,
                length,
                0,
                mtld3d_types::D3DFMT_INDEX16,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateRenderTarget` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_render_target_shared(
        &self,
        size: (u32, u32),
        format: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_render_target)(
                self.device,
                size.0,
                size.1,
                format,
                0,
                0,
                0,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateDepthStencilSurface` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_depth_stencil_surface_shared(
        &self,
        size: (u32, u32),
        format: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_depth_stencil_surface)(
                self.device,
                size.0,
                size.1,
                format,
                0,
                0,
                0,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }

    /// `CreateOffscreenPlainSurface` with a `pSharedHandle`.
    #[must_use]
    pub fn try_create_offscreen_plain_surface_shared(
        &self,
        size: (u32, u32),
        format: u32,
        pool: u32,
        shared_handle: &SharedHandle<'_>,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `out` is writable and the handle null or live.
        let hr = unsafe {
            (self.dev_vtbl().create_offscreen_plain_surface)(
                self.device,
                size.0,
                size.1,
                format,
                pool,
                &raw mut out,
                shared_handle.ptr,
            )
        };
        (hr, out)
    }
}
