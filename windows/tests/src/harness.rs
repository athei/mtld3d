//! The [`Harness`]: one D3D9 factory + window + device per test.
//!
//! Safe wrappers around every device/factory vtable method the suite needs.
//! All `unsafe` COM dispatch lives here and in [`crate::resource`]; test files
//! call only safe methods and assert on the returned `HRESULT`s / pixels.

use core::{cell::Cell, ffi::c_void};

use mtld3d_types::{
    D3DADAPTER_IDENTIFIER9, D3DCAPS9, D3DCLEAR_TARGET, D3DCREATE_HARDWARE_VERTEXPROCESSING,
    D3DDEVTYPE_HAL, D3DLIGHT9, D3DMATERIAL9, D3DPRESENT_PARAMETERS, D3DRECT, D3DSDK_VERSION,
    D3DSWAPEFFECT_DISCARD, D3DTA_DIFFUSE, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP,
    D3DTSS_COLORARG1, D3DTSS_COLOROP, D3DVIEWPORT9, Guid, IDirect3D9Vtbl, IDirect3DDevice9Vtbl,
};

use crate::{
    check::expect_ok,
    ffi::Direct3DCreate9,
    resource::{
        IndexBuffer, PixelShader, Query, StateBlock, Surface, Texture, VertexBuffer,
        VertexDeclaration, VertexShader,
    },
    vtbl::deref_vtbl,
    win32,
};

/// How a [`Harness`] device is created.
pub struct HarnessConfig {
    pub width: u32,
    pub height: u32,
    pub back_buffer_format: u32,
    /// `Some(fmt)` enables an auto depth-stencil of `fmt` (e.g. `D3DFMT_D24S8`).
    pub depth_format: Option<u32>,
    /// `WS_VISIBLE`. Hidden (default) keeps parallel runs off-screen.
    pub visible: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            back_buffer_format: mtld3d_types::D3DFMT_X8R8G8B8,
            depth_format: None,
            visible: false,
        }
    }
}

/// Scalar arguments for [`Harness::draw_indexed_primitive_up`].
///
/// The fixed D3D9 `DrawIndexedPrimitiveUP` parameters; the index and vertex
/// slices are passed separately.
pub struct DrawIndexedUpParams {
    pub prim: u32,
    pub min_vertex_index: u32,
    pub num_vertices: u32,
    pub prim_count: u32,
    pub index_format: u32,
}

/// A live device with its factory and window. Drops them in COM-correct order.
pub struct Harness {
    d3d9: *mut c_void,
    device: *mut c_void,
    hwnd: usize,
    width: Cell<u32>,
    height: Cell<u32>,
    back_buffer_format: u32,
    depth_format: Option<u32>,
}

impl Harness {
    /// A 640×480 X8R8G8B8 device with no depth buffer.
    ///
    /// # Panics
    /// Panics if the factory, window, or device cannot be created.
    #[must_use]
    pub fn new() -> Self {
        Self::create(&HarnessConfig::default())
    }

    /// A 640×480 device with an auto D24S8 depth-stencil.
    #[must_use]
    pub fn with_depth() -> Self {
        Self::create(&HarnessConfig {
            depth_format: Some(mtld3d_types::D3DFMT_D24S8),
            ..HarnessConfig::default()
        })
    }

    /// A factory only — no window or device.
    ///
    /// For pure `IDirect3D9` query tests (`Check*`, adapter enumeration, caps);
    /// device methods must not be called.
    ///
    /// # Panics
    /// Panics if the factory cannot be created.
    #[must_use]
    pub fn factory_only() -> Self {
        // SAFETY: Win32-style factory entrypoint with no preconditions.
        let d3d9 = unsafe { Direct3DCreate9(D3DSDK_VERSION) };
        assert!(!d3d9.is_null(), "Direct3DCreate9 returned null");
        Self {
            d3d9,
            device: core::ptr::null_mut(),
            hwnd: 0,
            width: Cell::new(0),
            height: Cell::new(0),
            back_buffer_format: mtld3d_types::D3DFMT_X8R8G8B8,
            depth_format: None,
        }
    }

    /// Create a device from an explicit [`HarnessConfig`].
    ///
    /// # Panics
    /// Panics if the factory, window, or device cannot be created.
    #[must_use]
    pub fn create(cfg: &HarnessConfig) -> Self {
        // SAFETY: Win32-style factory entrypoint with no preconditions.
        let d3d9 = unsafe { Direct3DCreate9(D3DSDK_VERSION) };
        assert!(!d3d9.is_null(), "Direct3DCreate9 returned null");

        let width = i32::try_from(cfg.width).expect("width fits i32");
        let height = i32::try_from(cfg.height).expect("height fits i32");
        let hwnd = win32::create_window(width, height, cfg.visible);

        let mut pp = present_params(cfg, hwnd);
        let mut device: *mut c_void = core::ptr::null_mut();
        // SAFETY: D3D9 factory vtable thunk; `d3d9` is live, `&mut pp` and
        // `&mut device` are writable, focus window null is permitted.
        let vtbl = unsafe { deref_vtbl::<IDirect3D9Vtbl>(d3d9) };
        // SAFETY: D3D9 vtable thunk; all pointers above are valid for the call.
        let hr = unsafe {
            (vtbl.create_device)(
                d3d9,
                0,
                D3DDEVTYPE_HAL,
                core::ptr::null_mut(),
                D3DCREATE_HARDWARE_VERTEXPROCESSING,
                (&raw mut pp).cast::<c_void>(),
                &raw mut device,
            )
        };
        assert_eq!(hr, 0, "CreateDevice failed: 0x{hr:08X}");
        assert!(!device.is_null(), "CreateDevice returned null device");

        Self {
            d3d9,
            device,
            hwnd,
            width: Cell::new(cfg.width),
            height: Cell::new(cfg.height),
            back_buffer_format: cfg.back_buffer_format,
            depth_format: cfg.depth_format,
        }
    }

    // ── Accessors ──

    /// The raw `IDirect3DDevice9*`.
    #[must_use]
    pub const fn device(&self) -> *mut c_void {
        self.device
    }

    /// The raw `IDirect3D9*` factory.
    #[must_use]
    pub const fn factory(&self) -> *mut c_void {
        self.d3d9
    }

    /// The device window handle.
    #[must_use]
    pub const fn hwnd(&self) -> usize {
        self.hwnd
    }

    /// Current backbuffer dimensions (tracks `reset`).
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        (self.width.get(), self.height.get())
    }

    // ── IUnknown / device-misc plumbing ──

    /// `IDirect3D9::AddRef` — returns the new reference count.
    pub fn add_ref_factory(&self) -> u32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe { (self.factory_vtbl().add_ref)(self.d3d9) }
    }

    /// `IDirect3D9::Release` — returns the reference count after the decrement.
    pub fn release_factory(&self) -> u32 {
        // SAFETY: vtable thunk; balances a prior `add_ref_factory`, so the object stays live.
        unsafe { (self.factory_vtbl().release)(self.d3d9) }
    }

    /// The device's current public refcount.
    ///
    /// `AddRef` then `Release`, returning the post-decrement count — the
    /// `get_refcount` idiom the D3D9 conformance suite uses. A child resource
    /// holds one device reference for its public lifetime, so creating one
    /// raises this by one and releasing it lowers it.
    pub fn device_refcount(&self) -> u32 {
        // SAFETY: vtable thunks; `self.device` is live for the harness lifetime.
        unsafe { (self.dev_vtbl().add_ref)(self.device) };
        // SAFETY: balances the AddRef above; the device stays live.
        unsafe { (self.dev_vtbl().release)(self.device) }
    }

    /// `IDirect3DDevice9::QueryInterface` for an unknown GUID. Returns the hr.
    pub fn device_query_interface_unknown(&self) -> i32 {
        let guid = Guid {
            data1: 0xDEAD_BEEF,
            data2: 0,
            data3: 0,
            data4: [0; 8],
        };
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&guid` and `&mut out` are valid for the call.
        unsafe { (self.dev_vtbl().query_interface)(self.device, &raw const guid, &raw mut out) }
    }

    /// `GetAvailableTextureMem`.
    #[must_use]
    pub fn available_texture_mem(&self) -> u32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().get_available_texture_mem)(self.device) }
    }

    /// `EvictManagedResources`. Returns the hr.
    pub fn evict_managed_resources(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().evict_managed_resources)(self.device) }
    }

    /// `ValidateDevice`. Returns the hr (the pass count is discarded).
    pub fn validate_device_hr(&self) -> i32 {
        let mut passes = 0u32;
        // SAFETY: vtable thunk; `&mut passes` is writable.
        unsafe { (self.dev_vtbl().validate_device)(self.device, &raw mut passes) }
    }

    /// `SetClipPlane(index, plane)`. Returns the hr.
    pub fn set_clip_plane(&self, index: u32, plane: [f32; 4]) -> i32 {
        // SAFETY: vtable thunk; `plane` is 4 floats, read-only for the call.
        unsafe { (self.dev_vtbl().set_clip_plane)(self.device, index, plane.as_ptr()) }
    }

    /// `GetClipPlane(index)`. Returns `(hr, plane)`.
    pub fn get_clip_plane(&self, index: u32) -> (i32, [f32; 4]) {
        let mut plane = [0.0f32; 4];
        // SAFETY: vtable thunk; `plane` is 4 writable floats.
        let hr =
            unsafe { (self.dev_vtbl().get_clip_plane)(self.device, index, plane.as_mut_ptr()) };
        (hr, plane)
    }

    /// `SetGammaRamp` — a no-op that must not crash (returns void).
    pub fn set_gamma_ramp_noop(&self) {
        // SAFETY: vtable thunk; null ramp is tolerated by the no-op handler.
        unsafe { (self.dev_vtbl().set_gamma_ramp)(self.device, 0, 0, core::ptr::null()) };
    }

    /// `SetPaletteEntries` (a documented stub today). Returns the hr.
    pub fn set_palette_entries_hr(&self) -> i32 {
        let palette = [0u32; 256];
        // SAFETY: vtable thunk; `palette` is 256 PALETTEENTRYs, read-only for the call.
        unsafe {
            (self.dev_vtbl().set_palette_entries)(self.device, 0, palette.as_ptr().cast::<c_void>())
        }
    }

    /// `GetRasterStatus` (a documented stub today). Returns the hr.
    pub fn get_raster_status_hr(&self) -> i32 {
        let mut status = [0u32; 2];
        // SAFETY: vtable thunk; `status` covers D3DRASTER_STATUS (BOOL + u32).
        unsafe {
            (self.dev_vtbl().get_raster_status)(
                self.device,
                0,
                status.as_mut_ptr().cast::<c_void>(),
            )
        }
    }

    /// `GetClipStatus` (a documented stub today). Returns the hr.
    pub fn get_clip_status_hr(&self) -> i32 {
        let mut status = [0u32; 2];
        // SAFETY: vtable thunk; `status` covers D3DCLIPSTATUS9 (two u32 fields).
        unsafe {
            (self.dev_vtbl().get_clip_status)(self.device, status.as_mut_ptr().cast::<c_void>())
        }
    }

    /// `SetDialogBoxMode` (a documented stub today). Returns the hr.
    pub fn set_dialog_box_mode_hr(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_dialog_box_mode)(self.device, 0) }
    }

    fn dev_vtbl(&self) -> &'static IDirect3DDevice9Vtbl {
        // SAFETY: `self.device` is a live IDirect3DDevice9 for the harness lifetime.
        unsafe { deref_vtbl::<IDirect3DDevice9Vtbl>(self.device) }
    }

    fn factory_vtbl(&self) -> &'static IDirect3D9Vtbl {
        // SAFETY: `self.d3d9` is a live IDirect3D9 for the harness lifetime.
        unsafe { deref_vtbl::<IDirect3D9Vtbl>(self.d3d9) }
    }

    // ── Frame loop ──

    /// Drain the message queue once. Returns `false` on `WM_QUIT`.
    pub fn pump(&self) -> bool {
        let mut msg = win32::zeroed_msg();
        win32::pump_messages(&mut msg)
    }

    /// `BeginScene`.
    pub fn begin_scene(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().begin_scene)(self.device) }
    }

    /// `EndScene`.
    pub fn end_scene(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().end_scene)(self.device) }
    }

    /// `Present` to the whole backbuffer.
    pub fn present(&self) -> i32 {
        // SAFETY: vtable thunk; all-null args present the entire backbuffer.
        unsafe {
            (self.dev_vtbl().present)(
                self.device,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        }
    }

    /// `Clear` with explicit flags / colour / depth / stencil.
    pub fn clear(&self, flags: u32, color: u32, z: f32, stencil: u32) -> i32 {
        // SAFETY: vtable thunk; null rect array clears the whole target.
        unsafe {
            (self.dev_vtbl().clear)(self.device, 0, core::ptr::null(), flags, color, z, stencil)
        }
    }

    /// `Clear(D3DCLEAR_TARGET)` to a solid colour.
    pub fn clear_target(&self, color: u32) -> i32 {
        self.clear(D3DCLEAR_TARGET, color, 1.0, 0)
    }

    /// Run one frame: pump → begin → clear → `body` → end → present.
    ///
    /// Asserts each step succeeds. Pair with [`Self::read_pixel`] (which flushes)
    /// to verify the result deterministically — one frame suffices.
    ///
    /// # Panics
    /// Panics if any step returns a failing `HRESULT` or `WM_QUIT` arrives.
    pub fn render_once(&self, clear_color: u32, body: impl FnOnce(&Self)) {
        assert!(self.pump(), "WM_QUIT before render");
        assert_eq!(self.begin_scene(), 0, "BeginScene failed");
        assert_eq!(self.clear_target(clear_color), 0, "Clear failed");
        body(self);
        assert_eq!(self.end_scene(), 0, "EndScene failed");
        assert_eq!(self.present(), 0, "Present failed");
    }

    /// Read a backbuffer pixel as `0xAARRGGBB` through the D3D9 read-back chain.
    ///
    /// The same chain the Wine conformance suite uses: `GetRenderTarget(0)` →
    /// `CreateOffscreenPlainSurface(D3DPOOL_SYSTEMMEM)` → `GetRenderTargetData`
    /// (which flushes pending GPU work) → `LockRect(READONLY)`. The returned
    /// value reflects every submitted frame. The system-memory surface is
    /// always `D3DFMT_A8R8G8B8`, so a locked row is `pitch / 4` `u32` pixels in
    /// `0xAARRGGBB` order.
    ///
    /// # Panics
    /// Panics if any step of the read-back chain fails.
    #[must_use]
    pub fn read_pixel(&self, x: u32, y: u32) -> u32 {
        let rt = self.render_target(0);
        let (hr, desc) = rt.desc();
        expect_ok(hr, "GetRenderTarget desc for read_pixel");
        let sysmem = self.create_offscreen_plain_surface(
            desc.width,
            desc.height,
            mtld3d_types::D3DFMT_A8R8G8B8,
            mtld3d_types::D3DPOOL_SYSTEMMEM,
        );
        expect_ok(
            self.get_render_target_data_hr(&rt, &sysmem),
            "GetRenderTargetData for read_pixel",
        );
        let locked = sysmem.lock_rect(mtld3d_types::D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        let idx = (y * pitch_px + x) as usize;
        locked.as_u32(idx + 1)[idx]
    }

    // ── Fixed-function / pipeline state ──

    /// `SetFVF`.
    pub fn set_fvf(&self, fvf: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_fvf)(self.device, fvf) }
    }

    /// `GetFVF`.
    #[must_use]
    pub fn fvf(&self) -> u32 {
        let mut fvf = 0u32;
        // SAFETY: vtable thunk; `&mut fvf` is writable.
        let hr = unsafe { (self.dev_vtbl().get_fvf)(self.device, &raw mut fvf) };
        expect_ok(hr, "GetFVF");
        fvf
    }

    /// `SetRenderState`.
    pub fn set_render_state(&self, state: u32, value: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_render_state)(self.device, state, value) }
    }

    /// `GetRenderState`, asserting success.
    ///
    /// Use [`Self::try_render_state`] for the raw `HRESULT`.
    #[must_use]
    pub fn render_state(&self, state: u32) -> u32 {
        let (hr, value) = self.try_render_state(state);
        expect_ok(hr, "GetRenderState");
        value
    }

    /// `GetRenderState` returning `(hr, value)`.
    #[must_use]
    pub fn try_render_state(&self, state: u32) -> (i32, u32) {
        let mut value = 0u32;
        // SAFETY: vtable thunk; `&mut value` is writable.
        let hr = unsafe { (self.dev_vtbl().get_render_state)(self.device, state, &raw mut value) };
        (hr, value)
    }

    /// `SetSamplerState`.
    pub fn set_sampler_state(&self, sampler: u32, state: u32, value: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_sampler_state)(self.device, sampler, state, value) }
    }

    /// `GetSamplerState`, asserting success.
    #[must_use]
    pub fn sampler_state(&self, sampler: u32, state: u32) -> u32 {
        let mut value = 0u32;
        // SAFETY: vtable thunk; `&mut value` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_sampler_state)(self.device, sampler, state, &raw mut value)
        };
        expect_ok(hr, "GetSamplerState");
        value
    }

    /// `SetTextureStageState`.
    pub fn set_texture_stage_state(&self, stage: u32, ts_state: u32, value: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_texture_stage_state)(self.device, stage, ts_state, value) }
    }

    /// Route `stage` to pass vertex DIFFUSE through unchanged (colour + alpha).
    ///
    /// The common fixed-function setup when no texture is bound.
    ///
    /// # Panics
    /// Panics if any `SetTextureStageState` fails.
    pub fn select_diffuse_stage(&self, stage: u32) {
        for (ts_state, value) in [
            (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
            (D3DTSS_COLORARG1, D3DTA_DIFFUSE),
            (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
            (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
        ] {
            expect_ok(
                self.set_texture_stage_state(stage, ts_state, value),
                "SetTextureStageState",
            );
        }
    }

    /// Route `stage` to emit the sampled texel directly (colour + alpha).
    ///
    /// Ignores vertex diffuse — for sampler/format tests.
    ///
    /// # Panics
    /// Panics if any `SetTextureStageState` fails.
    pub fn select_texture_stage(&self, stage: u32) {
        for (ts_state, value) in [
            (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
            (D3DTSS_COLORARG1, mtld3d_types::D3DTA_TEXTURE),
            (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
            (D3DTSS_ALPHAARG1, mtld3d_types::D3DTA_TEXTURE),
        ] {
            expect_ok(
                self.set_texture_stage_state(stage, ts_state, value),
                "SetTextureStageState",
            );
        }
    }

    /// `GetTextureStageState`, asserting success.
    #[must_use]
    pub fn texture_stage_state(&self, stage: u32, ts_state: u32) -> u32 {
        let mut value = 0u32;
        // SAFETY: vtable thunk; `&mut value` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_texture_stage_state)(self.device, stage, ts_state, &raw mut value)
        };
        expect_ok(hr, "GetTextureStageState");
        value
    }

    /// `SetTexture(stage, texture)`.
    pub fn set_texture(&self, stage: u32, texture: &Texture<'_>) -> i32 {
        // SAFETY: vtable thunk; `texture` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_texture)(self.device, stage, texture.as_ptr()) }
    }

    /// `SetTexture(stage, null)` — unbind whatever is on `stage`.
    pub fn clear_texture(&self, stage: u32) -> i32 {
        // SAFETY: vtable thunk; null unbinds the stage.
        unsafe { (self.dev_vtbl().set_texture)(self.device, stage, core::ptr::null_mut()) }
    }

    /// `GetTexture(stage)` — the raw bound `this` (null if unbound).
    #[must_use]
    pub fn texture_raw(&self, stage: u32) -> *mut c_void {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().get_texture)(self.device, stage, &raw mut out) };
        expect_ok(hr, "GetTexture");
        out
    }

    /// `SetTransform`.
    pub fn set_transform(&self, state: u32, matrix: &[f32; 16]) -> i32 {
        // SAFETY: vtable thunk; `matrix` is 16 floats, read-only for the call.
        unsafe {
            (self.dev_vtbl().set_transform)(self.device, state, matrix.as_ptr().cast::<c_void>())
        }
    }

    /// `GetTransform`, asserting success.
    #[must_use]
    pub fn transform(&self, state: u32) -> [f32; 16] {
        let mut m = [0f32; 16];
        // SAFETY: vtable thunk; `m` is 16 writable floats.
        let hr = unsafe {
            (self.dev_vtbl().get_transform)(self.device, state, m.as_mut_ptr().cast::<c_void>())
        };
        expect_ok(hr, "GetTransform");
        m
    }

    /// `MultiplyTransform`.
    pub fn multiply_transform(&self, state: u32, matrix: &[f32; 16]) -> i32 {
        // SAFETY: vtable thunk; `matrix` is 16 floats, read-only for the call.
        unsafe {
            (self.dev_vtbl().multiply_transform)(
                self.device,
                state,
                matrix.as_ptr().cast::<c_void>(),
            )
        }
    }

    /// `SetViewport`.
    pub fn set_viewport(&self, vp: &D3DVIEWPORT9) -> i32 {
        // SAFETY: vtable thunk; `vp` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_viewport)(self.device, core::ptr::from_ref(vp).cast::<c_void>())
        }
    }

    /// `GetViewport`, asserting success.
    #[must_use]
    pub fn viewport(&self) -> D3DVIEWPORT9 {
        let mut vp = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            min_z: 0.0,
            max_z: 0.0,
        };
        // SAFETY: vtable thunk; `&mut vp` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_viewport)(
                self.device,
                core::ptr::from_mut(&mut vp).cast::<c_void>(),
            )
        };
        expect_ok(hr, "GetViewport");
        vp
    }

    /// `SetScissorRect`.
    pub fn set_scissor_rect(&self, rect: &D3DRECT) -> i32 {
        // SAFETY: vtable thunk; `rect` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_scissor_rect)(
                self.device,
                core::ptr::from_ref(rect).cast::<c_void>(),
            )
        }
    }

    /// `GetScissorRect`, asserting success.
    #[must_use]
    pub fn scissor_rect(&self) -> D3DRECT {
        let mut rect = D3DRECT {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        };
        // SAFETY: vtable thunk; `&mut rect` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_scissor_rect)(
                self.device,
                core::ptr::from_mut(&mut rect).cast::<c_void>(),
            )
        };
        expect_ok(hr, "GetScissorRect");
        rect
    }

    /// `TestCooperativeLevel`.
    pub fn test_cooperative_level(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().test_cooperative_level)(self.device) }
    }

    // ── Draws ──

    /// `DrawPrimitiveUP` with inline vertices; stride is `size_of::<V>()`.
    ///
    /// # Panics
    /// Panics if `size_of::<V>()` does not fit in a `u32`.
    pub fn draw_primitive_up<V>(&self, prim: u32, prim_count: u32, verts: &[V]) -> i32 {
        let stride = u32::try_from(core::mem::size_of::<V>()).expect("vertex stride fits u32");
        // SAFETY: vtable thunk; `verts` is read-only for the call.
        unsafe {
            (self.dev_vtbl().draw_primitive_up)(
                self.device,
                prim,
                prim_count,
                verts.as_ptr().cast::<c_void>(),
                stride,
            )
        }
    }

    /// `DrawPrimitive` against the bound stream source.
    pub fn draw_primitive(&self, prim: u32, start_vertex: u32, prim_count: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().draw_primitive)(self.device, prim, start_vertex, prim_count) }
    }

    /// `DrawIndexedPrimitive` against the bound stream source + indices.
    pub fn draw_indexed_primitive(
        &self,
        prim: u32,
        base_vertex_index: i32,
        min_vertex_index: u32,
        num_vertices: u32,
        start_index: u32,
        prim_count: u32,
    ) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe {
            (self.dev_vtbl().draw_indexed_primitive)(
                self.device,
                prim,
                base_vertex_index,
                min_vertex_index,
                num_vertices,
                start_index,
                prim_count,
            )
        }
    }

    /// `DrawIndexedPrimitiveUP`. Returns the hr.
    ///
    /// # Panics
    /// Panics if `size_of::<V>()` does not fit in a `u32`.
    pub fn draw_indexed_primitive_up<I, V>(
        &self,
        params: &DrawIndexedUpParams,
        indices: &[I],
        verts: &[V],
    ) -> i32 {
        let &DrawIndexedUpParams {
            prim,
            min_vertex_index,
            num_vertices,
            prim_count,
            index_format,
        } = params;
        let stride = u32::try_from(core::mem::size_of::<V>()).expect("vertex stride fits u32");
        // SAFETY: vtable thunk; both slices are read-only for the call.
        unsafe {
            (self.dev_vtbl().draw_indexed_primitive_up)(
                self.device,
                prim,
                min_vertex_index,
                num_vertices,
                prim_count,
                indices.as_ptr().cast::<c_void>(),
                index_format,
                verts.as_ptr().cast::<c_void>(),
                stride,
            )
        }
    }

    /// `SetStreamSource(stream, vb, offset, stride)`.
    pub fn set_stream_source(
        &self,
        stream: u32,
        vb: &VertexBuffer<'_>,
        offset: u32,
        stride: u32,
    ) -> i32 {
        // SAFETY: vtable thunk; `vb` is a live binding for the call.
        unsafe {
            (self.dev_vtbl().set_stream_source)(self.device, stream, vb.as_ptr(), offset, stride)
        }
    }

    /// `SetStreamSource(stream, NULL, offset, stride)` — clears the vertex-buffer binding.
    ///
    /// D3D9 retains the previous offset/stride on a NULL bind, so callers pass
    /// `0, 0` like the runtime requires.
    pub fn set_stream_source_null(&self, stream: u32, offset: u32, stride: u32) -> i32 {
        // SAFETY: vtable thunk; a null stream source is the documented "unbind".
        unsafe {
            (self.dev_vtbl().set_stream_source)(
                self.device,
                stream,
                core::ptr::null_mut(),
                offset,
                stride,
            )
        }
    }

    /// `SetIndices(ib)`.
    pub fn set_indices(&self, ib: &IndexBuffer<'_>) -> i32 {
        // SAFETY: vtable thunk; `ib` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_indices)(self.device, ib.as_ptr()) }
    }

    /// `ProcessVertices` (a documented stub today). Returns the hr.
    pub fn process_vertices_hr(&self) -> i32 {
        // SAFETY: vtable thunk; the stub ignores its (null) buffer/decl args.
        unsafe {
            (self.dev_vtbl().process_vertices)(
                self.device,
                0,
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
            )
        }
    }

    /// `GetStreamSource(stream)`.
    ///
    /// Returns `(hr, vb, offset, stride)`. On success the bound stream-0 buffer
    /// (or `None` when nothing is bound) is wrapped so its `Drop` balances the
    /// `AddRef` D3D9 applies to the out-pointer.
    pub fn get_stream_source(&self, stream: u32) -> (i32, Option<VertexBuffer<'_>>, u32, u32) {
        let mut vb: *mut c_void = core::ptr::null_mut();
        let mut offset = 0u32;
        let mut stride = 0u32;
        // SAFETY: vtable thunk; all out-pointers are writable.
        let hr = unsafe {
            (self.dev_vtbl().get_stream_source)(
                self.device,
                stream,
                &raw mut vb,
                &raw mut offset,
                &raw mut stride,
            )
        };
        let wrapped = (!vb.is_null()).then(|| VertexBuffer::from_raw(vb));
        (hr, wrapped, offset, stride)
    }

    /// `GetIndices()`.
    ///
    /// Returns `(hr, ib)`, with the bound index buffer (or `None`) wrapped so
    /// its `Drop` balances the `AddRef` on the out-pointer.
    pub fn get_indices(&self) -> (i32, Option<IndexBuffer<'_>>) {
        let mut ib: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut ib` is writable.
        let hr = unsafe { (self.dev_vtbl().get_indices)(self.device, &raw mut ib) };
        let wrapped = (!ib.is_null()).then(|| IndexBuffer::from_raw(ib));
        (hr, wrapped)
    }

    /// `SetStreamSourceFreq` (a documented stub today). Returns the hr.
    pub fn set_stream_source_freq(&self, stream: u32, freq: u32) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().set_stream_source_freq)(self.device, stream, freq) }
    }

    // ── Resource creation ──

    /// `CreateTexture`, asserting success. Returns an owned [`Texture`].
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_texture(
        &self,
        width: u32,
        height: u32,
        levels: u32,
        usage: u32,
        format: u32,
        pool: u32,
    ) -> Texture<'_> {
        let (hr, ptr) = self.try_create_texture(width, height, levels, usage, format, pool);
        assert_eq!(hr, 0, "CreateTexture failed: 0x{hr:08X}");
        assert!(!ptr.is_null(), "CreateTexture returned null");
        Texture::from_raw(ptr)
    }

    /// `CreateTexture` returning `(hr, this)` for error-path tests.
    #[must_use]
    pub fn try_create_texture(
        &self,
        width: u32,
        height: u32,
        levels: u32,
        usage: u32,
        format: u32,
        pool: u32,
    ) -> (i32, *mut c_void) {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle is allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_texture)(
                self.device,
                width,
                height,
                levels,
                usage,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        (hr, out)
    }

    /// `CreateCubeTexture`.
    ///
    /// Returns the hr; on success the created cube texture is released here
    /// (callers only inspect the hr), so the helper never leaks a COM object
    /// via its own `IUnknown::Release` (vtbl slot 2).
    pub fn create_cube_texture(
        &self,
        edge: u32,
        levels: u32,
        usage: u32,
        format: u32,
        pool: u32,
    ) -> i32 {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle is allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_cube_texture)(
                self.device,
                edge,
                levels,
                usage,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        if hr == 0 && !out.is_null() {
            type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
            // SAFETY: a live COM object's first field is its vtable pointer.
            let vtbl = unsafe { *out.cast::<*const ReleaseFn>() };
            // SAFETY: `IUnknown::Release` is slot 2 of every D3D9 vtable.
            let slot = unsafe { vtbl.add(2) };
            // SAFETY: `slot` points at the live `Release` fn pointer.
            let release = unsafe { *slot };
            // SAFETY: releasing the cube texture created just above at refcount 1.
            unsafe { release(out) };
        }
        hr
    }

    /// `CreateVolumeTexture`, returning the raw hr.
    ///
    /// `extent` is `[width, height, depth]`.
    pub fn create_volume_texture(
        &self,
        extent: [u32; 3],
        levels: u32,
        usage: u32,
        format: u32,
        pool: u32,
    ) -> i32 {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle is allowed.
        unsafe {
            (self.dev_vtbl().create_volume_texture)(
                self.device,
                extent[0],
                extent[1],
                extent[2],
                levels,
                usage,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        }
    }

    /// `CreateVertexBuffer`, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_vertex_buffer(
        &self,
        length: u32,
        usage: u32,
        fvf: u32,
        pool: u32,
    ) -> VertexBuffer<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle is allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_vertex_buffer)(
                self.device,
                length,
                usage,
                fvf,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(hr, 0, "CreateVertexBuffer failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateVertexBuffer returned null");
        VertexBuffer::from_raw(out)
    }

    /// `CreateIndexBuffer`, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_index_buffer(
        &self,
        length: u32,
        usage: u32,
        format: u32,
        pool: u32,
    ) -> IndexBuffer<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle is allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_index_buffer)(
                self.device,
                length,
                usage,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(hr, 0, "CreateIndexBuffer failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateIndexBuffer returned null");
        IndexBuffer::from_raw(out)
    }

    /// `CreateVertexShader` from DXSO bytecode, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_vertex_shader(&self, bytecode: &[u32]) -> VertexShader<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `bytecode` is read-only, `&mut out` writable.
        let hr = unsafe {
            (self.dev_vtbl().create_vertex_shader)(self.device, bytecode.as_ptr(), &raw mut out)
        };
        assert_eq!(hr, 0, "CreateVertexShader failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateVertexShader returned null");
        VertexShader::from_raw(out)
    }

    /// `CreatePixelShader` from DXSO bytecode, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_pixel_shader(&self, bytecode: &[u32]) -> PixelShader<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `bytecode` is read-only, `&mut out` writable.
        let hr = unsafe {
            (self.dev_vtbl().create_pixel_shader)(self.device, bytecode.as_ptr(), &raw mut out)
        };
        assert_eq!(hr, 0, "CreatePixelShader failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreatePixelShader returned null");
        PixelShader::from_raw(out)
    }

    /// `SetVertexShader(shader)`.
    pub fn set_vertex_shader(&self, shader: &VertexShader<'_>) -> i32 {
        // SAFETY: vtable thunk; `shader` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_vertex_shader)(self.device, shader.as_ptr()) }
    }

    /// `SetVertexShader(null)`.
    pub fn clear_vertex_shader(&self) -> i32 {
        // SAFETY: vtable thunk; null unbinds the vertex shader.
        unsafe { (self.dev_vtbl().set_vertex_shader)(self.device, core::ptr::null_mut()) }
    }

    /// `SetPixelShader(shader)`.
    pub fn set_pixel_shader(&self, shader: &PixelShader<'_>) -> i32 {
        // SAFETY: vtable thunk; `shader` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_pixel_shader)(self.device, shader.as_ptr()) }
    }

    /// `SetPixelShader(null)`.
    pub fn clear_pixel_shader(&self) -> i32 {
        // SAFETY: vtable thunk; null unbinds the pixel shader.
        unsafe { (self.dev_vtbl().set_pixel_shader)(self.device, core::ptr::null_mut()) }
    }

    /// `SetVertexShaderConstantF`. `data` is whole vec4 registers (len % 4 == 0).
    ///
    /// # Panics
    /// Panics if the vec4 count does not fit in a `u32`.
    pub fn set_vertex_shader_constant_f(&self, start: u32, data: &[f32]) -> i32 {
        let count = u32::try_from(data.len() / 4).expect("vec4 count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_vertex_shader_constant_f)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `SetPixelShaderConstantF`. `data` is whole vec4 registers (len % 4 == 0).
    ///
    /// # Panics
    /// Panics if the vec4 count does not fit in a `u32`.
    pub fn set_pixel_shader_constant_f(&self, start: u32, data: &[f32]) -> i32 {
        let count = u32::try_from(data.len() / 4).expect("vec4 count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_pixel_shader_constant_f)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `SetVertexShaderConstantI` (each register is 4 ints).
    ///
    /// # Panics
    /// Panics if the register count does not fit in a `u32`.
    pub fn set_vertex_shader_constant_i(&self, start: u32, data: &[i32]) -> i32 {
        let count = u32::try_from(data.len() / 4).expect("ivec4 count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_vertex_shader_constant_i)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `SetVertexShaderConstantB` (each register is one BOOL).
    ///
    /// # Panics
    /// Panics if the register count does not fit in a `u32`.
    pub fn set_vertex_shader_constant_b(&self, start: u32, data: &[i32]) -> i32 {
        let count = u32::try_from(data.len()).expect("bool count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_vertex_shader_constant_b)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `SetPixelShaderConstantI` (each register is 4 ints).
    ///
    /// # Panics
    /// Panics if the register count does not fit in a `u32`.
    pub fn set_pixel_shader_constant_i(&self, start: u32, data: &[i32]) -> i32 {
        let count = u32::try_from(data.len() / 4).expect("ivec4 count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_pixel_shader_constant_i)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `SetPixelShaderConstantB` (each register is one BOOL).
    ///
    /// # Panics
    /// Panics if the register count does not fit in a `u32`.
    pub fn set_pixel_shader_constant_b(&self, start: u32, data: &[i32]) -> i32 {
        let count = u32::try_from(data.len()).expect("bool count fits u32");
        // SAFETY: vtable thunk; `data` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_pixel_shader_constant_b)(self.device, start, data.as_ptr(), count)
        }
    }

    /// `GetVertexShaderConstantF` reading `count` vec4 registers from `start`.
    ///
    /// Returns the hr and the read-back floats (`count * 4`).
    pub fn get_vertex_shader_constant_f(&self, start: u32, count: u32) -> (i32, Vec<f32>) {
        let mut out = vec![0f32; count as usize * 4];
        // SAFETY: vtable thunk; `out` holds `count * 4` writable floats.
        let hr = unsafe {
            (self.dev_vtbl().get_vertex_shader_constant_f)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetPixelShaderConstantF` — see [`Self::get_vertex_shader_constant_f`].
    pub fn get_pixel_shader_constant_f(&self, start: u32, count: u32) -> (i32, Vec<f32>) {
        let mut out = vec![0f32; count as usize * 4];
        // SAFETY: vtable thunk; `out` holds `count * 4` writable floats.
        let hr = unsafe {
            (self.dev_vtbl().get_pixel_shader_constant_f)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetVertexShaderConstantI` reading `count` ivec4 registers from `start`.
    pub fn get_vertex_shader_constant_i(&self, start: u32, count: u32) -> (i32, Vec<i32>) {
        let mut out = vec![0i32; count as usize * 4];
        // SAFETY: vtable thunk; `out` holds `count * 4` writable ints.
        let hr = unsafe {
            (self.dev_vtbl().get_vertex_shader_constant_i)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetPixelShaderConstantI` — see [`Self::get_vertex_shader_constant_i`].
    pub fn get_pixel_shader_constant_i(&self, start: u32, count: u32) -> (i32, Vec<i32>) {
        let mut out = vec![0i32; count as usize * 4];
        // SAFETY: vtable thunk; `out` holds `count * 4` writable ints.
        let hr = unsafe {
            (self.dev_vtbl().get_pixel_shader_constant_i)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetVertexShaderConstantB` reading `count` BOOL registers from `start`.
    pub fn get_vertex_shader_constant_b(&self, start: u32, count: u32) -> (i32, Vec<i32>) {
        let mut out = vec![0i32; count as usize];
        // SAFETY: vtable thunk; `out` holds `count` writable BOOLs.
        let hr = unsafe {
            (self.dev_vtbl().get_vertex_shader_constant_b)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetPixelShaderConstantB` — see [`Self::get_vertex_shader_constant_b`].
    pub fn get_pixel_shader_constant_b(&self, start: u32, count: u32) -> (i32, Vec<i32>) {
        let mut out = vec![0i32; count as usize];
        // SAFETY: vtable thunk; `out` holds `count` writable BOOLs.
        let hr = unsafe {
            (self.dev_vtbl().get_pixel_shader_constant_b)(
                self.device,
                start,
                out.as_mut_ptr(),
                count,
            )
        };
        (hr, out)
    }

    /// `GetVertexDeclaration` — the raw bound declaration `this` (null if unset).
    #[must_use]
    pub fn vertex_declaration_raw(&self) -> *mut c_void {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().get_vertex_declaration)(self.device, &raw mut out) };
        expect_ok(hr, "GetVertexDeclaration");
        out
    }

    /// `SetMaterial`.
    pub fn set_material(&self, material: &D3DMATERIAL9) -> i32 {
        // SAFETY: vtable thunk; `material` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_material)(
                self.device,
                core::ptr::from_ref(material).cast::<c_void>(),
            )
        }
    }

    /// `GetMaterial`, asserting success.
    #[must_use]
    pub fn material(&self) -> D3DMATERIAL9 {
        // SAFETY: POD struct overwritten by the call before any field is read.
        let mut m = unsafe { core::mem::zeroed::<D3DMATERIAL9>() };
        // SAFETY: vtable thunk; `&mut m` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_material)(
                self.device,
                core::ptr::from_mut(&mut m).cast::<c_void>(),
            )
        };
        expect_ok(hr, "GetMaterial");
        m
    }

    /// `SetLight`.
    pub fn set_light(&self, index: u32, light: &D3DLIGHT9) -> i32 {
        // SAFETY: vtable thunk; `light` is read-only for the call.
        unsafe {
            (self.dev_vtbl().set_light)(
                self.device,
                index,
                core::ptr::from_ref(light).cast::<c_void>(),
            )
        }
    }

    /// `GetLight`, asserting success.
    #[must_use]
    pub fn light(&self, index: u32) -> D3DLIGHT9 {
        // SAFETY: POD struct overwritten by the call before any field is read.
        let mut l = unsafe { core::mem::zeroed::<D3DLIGHT9>() };
        // SAFETY: vtable thunk; `&mut l` is writable.
        let hr = unsafe {
            (self.dev_vtbl().get_light)(
                self.device,
                index,
                core::ptr::from_mut(&mut l).cast::<c_void>(),
            )
        };
        expect_ok(hr, "GetLight");
        l
    }

    /// `LightEnable`.
    pub fn light_enable(&self, index: u32, enable: bool) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().light_enable)(self.device, index, i32::from(enable)) }
    }

    /// `GetLightEnable`, asserting success.
    #[must_use]
    pub fn light_enabled(&self, index: u32) -> bool {
        let mut enabled = 0i32;
        // SAFETY: vtable thunk; `&mut enabled` is writable.
        let hr =
            unsafe { (self.dev_vtbl().get_light_enable)(self.device, index, &raw mut enabled) };
        expect_ok(hr, "GetLightEnable");
        enabled != 0
    }

    /// `CreateStateBlock(type)`, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_state_block(&self, sbt: u32) -> StateBlock<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().create_state_block)(self.device, sbt, &raw mut out) };
        assert_eq!(hr, 0, "CreateStateBlock failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateStateBlock returned null");
        StateBlock::from_raw(out)
    }

    /// `BeginStateBlock`.
    pub fn begin_state_block(&self) -> i32 {
        // SAFETY: vtable thunk; `self.device` is live.
        unsafe { (self.dev_vtbl().begin_state_block)(self.device) }
    }

    /// `EndStateBlock`, asserting success.
    ///
    /// # Panics
    /// Panics if recording was not open or capture fails.
    #[must_use]
    pub fn end_state_block(&self) -> StateBlock<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().end_state_block)(self.device, &raw mut out) };
        assert_eq!(hr, 0, "EndStateBlock failed: 0x{hr:08X}");
        assert!(!out.is_null(), "EndStateBlock returned null");
        StateBlock::from_raw(out)
    }

    /// `CreateQuery(type, null)` — the support probe. Returns the hr.
    pub fn query_supported(&self, query_type: u32) -> i32 {
        // SAFETY: vtable thunk; null out-pointer is the documented probe form.
        unsafe { (self.dev_vtbl().create_query)(self.device, query_type, core::ptr::null_mut()) }
    }

    /// `CreateQuery(type)`. Returns `None` if the type is unsupported.
    #[must_use]
    pub fn create_query(&self, query_type: u32) -> Option<Query<'_>> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().create_query)(self.device, query_type, &raw mut out) };
        if hr != 0 || out.is_null() {
            return None;
        }
        Some(Query::from_raw(out))
    }

    /// `CreateVertexDeclaration` from a `D3DVERTEXELEMENT9` array, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_vertex_declaration(
        &self,
        elements: &[mtld3d_types::D3DVERTEXELEMENT9],
    ) -> VertexDeclaration<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `elements` is read-only, `&mut out` writable.
        let hr = unsafe {
            (self.dev_vtbl().create_vertex_declaration)(
                self.device,
                elements.as_ptr().cast::<c_void>(),
                &raw mut out,
            )
        };
        assert_eq!(hr, 0, "CreateVertexDeclaration failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateVertexDeclaration returned null");
        VertexDeclaration::from_raw(out)
    }

    /// `SetVertexDeclaration(decl)`.
    pub fn set_vertex_declaration(&self, decl: &VertexDeclaration<'_>) -> i32 {
        // SAFETY: vtable thunk; `decl` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_vertex_declaration)(self.device, decl.as_ptr()) }
    }

    /// `SetVertexDeclaration(NULL)` — clears the bound vertex declaration.
    ///
    /// D3D9 also resets the effective FVF to zero when a declaration is cleared.
    pub fn set_vertex_declaration_null(&self) -> i32 {
        // SAFETY: vtable thunk; a null declaration is the documented "unbind".
        unsafe { (self.dev_vtbl().set_vertex_declaration)(self.device, core::ptr::null_mut()) }
    }

    // ── Render targets / depth ──

    /// `CreateRenderTarget` returning the raw hr, for the rejection paths.
    ///
    /// Non-multisampled and non-lockable. A format with no renderable colour
    /// mapping is INVALIDCALL; [`Self::create_render_target`] asserts success.
    /// The sampleable render-target path is `CreateTexture(D3DUSAGE_RENDERTARGET)`.
    pub fn create_render_target_hr(&self, width: u32, height: u32, format: u32) -> i32 {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle allowed.
        unsafe {
            (self.dev_vtbl().create_render_target)(
                self.device,
                width,
                height,
                format,
                0,
                0,
                0,
                &raw mut out,
                core::ptr::null_mut(),
            )
        }
    }

    /// `CreateRenderTarget`, asserting success and returning the surface.
    ///
    /// Use [`Self::create_render_target_hr`] to test the rejection paths
    /// instead.
    ///
    /// # Panics
    /// Panics if the call fails or returns null.
    #[must_use]
    pub fn create_render_target(&self, width: u32, height: u32, format: u32) -> Surface<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_render_target)(
                self.device,
                width,
                height,
                format,
                0,
                0,
                0,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(hr, 0, "CreateRenderTarget failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateRenderTarget returned null");
        Surface::from_raw(out)
    }

    /// `CreateDepthStencilSurface`, asserting success.
    ///
    /// # Panics
    /// Panics if creation fails.
    #[must_use]
    pub fn create_depth_stencil_surface(
        &self,
        width: u32,
        height: u32,
        format: u32,
    ) -> Surface<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_depth_stencil_surface)(
                self.device,
                width,
                height,
                format,
                0,
                0,
                0,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(hr, 0, "CreateDepthStencilSurface failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateDepthStencilSurface returned null");
        Surface::from_raw(out)
    }

    /// `SetRenderTarget(index, surface)`.
    pub fn set_render_target(&self, index: u32, surface: &Surface<'_>) -> i32 {
        // SAFETY: vtable thunk; `surface` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_render_target)(self.device, index, surface.as_ptr()) }
    }

    /// `GetRenderTarget(index)`, asserting success.
    ///
    /// # Panics
    /// Panics if the call fails.
    #[must_use]
    pub fn render_target(&self, index: u32) -> Surface<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().get_render_target)(self.device, index, &raw mut out) };
        assert_eq!(hr, 0, "GetRenderTarget({index}) failed: 0x{hr:08X}");
        assert!(!out.is_null(), "GetRenderTarget returned null");
        Surface::from_raw(out)
    }

    /// `SetDepthStencilSurface(surface)`.
    pub fn set_depth_stencil_surface(&self, surface: &Surface<'_>) -> i32 {
        // SAFETY: vtable thunk; `surface` is a live binding for the call.
        unsafe { (self.dev_vtbl().set_depth_stencil_surface)(self.device, surface.as_ptr()) }
    }

    /// `SetDepthStencilSurface(null)` — unbind the depth-stencil surface.
    pub fn clear_depth_stencil_surface(&self) -> i32 {
        // SAFETY: vtable thunk; null unbinds the depth-stencil surface.
        unsafe { (self.dev_vtbl().set_depth_stencil_surface)(self.device, core::ptr::null_mut()) }
    }

    /// `GetDepthStencilSurface` — `None` if no depth-stencil is bound.
    #[must_use]
    pub fn depth_stencil_surface(&self) -> Option<Surface<'_>> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable.
        let hr = unsafe { (self.dev_vtbl().get_depth_stencil_surface)(self.device, &raw mut out) };
        if hr != 0 || out.is_null() {
            return None;
        }
        Some(Surface::from_raw(out))
    }

    /// `StretchRect` over whole surfaces (null rects). Returns the hr.
    pub fn stretch_rect(&self, src: &Surface<'_>, dst: &Surface<'_>, filter: u32) -> i32 {
        // SAFETY: vtable thunk; both surfaces are live, null rects = whole surface.
        unsafe {
            (self.dev_vtbl().stretch_rect)(
                self.device,
                src.as_ptr(),
                core::ptr::null(),
                dst.as_ptr(),
                core::ptr::null(),
                filter,
            )
        }
    }

    /// `ColorFill` over the whole surface (null rect). Returns the hr.
    ///
    /// Only a `D3DPOOL_DEFAULT` render target or offscreen-plain surface is a
    /// valid destination; anything else is INVALIDCALL.
    pub fn color_fill_hr(&self, surface: &Surface<'_>, color: u32) -> i32 {
        // SAFETY: vtable thunk; null rect = whole surface.
        unsafe {
            (self.dev_vtbl().color_fill)(self.device, surface.as_ptr(), core::ptr::null(), color)
        }
    }

    /// `GetRenderTargetData` — copy a render target into a system-memory surface.
    ///
    /// Returns the hr; a destination outside `D3DPOOL_SYSTEMMEM` is INVALIDCALL.
    pub fn get_render_target_data_hr(&self, rt: &Surface<'_>, dst: &Surface<'_>) -> i32 {
        // SAFETY: vtable thunk; both surfaces are live.
        unsafe { (self.dev_vtbl().get_render_target_data)(self.device, rt.as_ptr(), dst.as_ptr()) }
    }

    /// `GetFrontBufferData` (a documented stub today). Returns the hr.
    pub fn get_front_buffer_data_hr(&self, dst: &Surface<'_>) -> i32 {
        // SAFETY: vtable thunk; `dst` is a live surface.
        unsafe { (self.dev_vtbl().get_front_buffer_data)(self.device, 0, dst.as_ptr()) }
    }

    /// `CreateOffscreenPlainSurface` returning the raw hr, for the rejection paths.
    pub fn create_offscreen_plain_surface_hr(
        &self,
        width: u32,
        height: u32,
        format: u32,
        pool: u32,
    ) -> i32 {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle allowed.
        unsafe {
            (self.dev_vtbl().create_offscreen_plain_surface)(
                self.device,
                width,
                height,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        }
    }

    /// `CreateOffscreenPlainSurface`, asserting success and returning the surface.
    ///
    /// Use [`Self::create_offscreen_plain_surface_hr`] to test the rejection
    /// paths instead.
    ///
    /// # Panics
    /// Panics if the call fails or returns null.
    #[must_use]
    pub fn create_offscreen_plain_surface(
        &self,
        width: u32,
        height: u32,
        format: u32,
        pool: u32,
    ) -> Surface<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable, null shared-handle allowed.
        let hr = unsafe {
            (self.dev_vtbl().create_offscreen_plain_surface)(
                self.device,
                width,
                height,
                format,
                pool,
                &raw mut out,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(hr, 0, "CreateOffscreenPlainSurface failed: 0x{hr:08X}");
        assert!(!out.is_null(), "CreateOffscreenPlainSurface returned null");
        Surface::from_raw(out)
    }

    /// `SetCursorProperties(x_hotspot, y_hotspot, surface)`.
    ///
    /// Returns the hr so callers can exercise both the accept and reject paths.
    pub fn set_cursor_properties_hr(
        &self,
        x_hotspot: u32,
        y_hotspot: u32,
        surface: &Surface<'_>,
    ) -> i32 {
        // SAFETY: vtable thunk; `surface` is a live IDirect3DSurface9.
        unsafe {
            (self.dev_vtbl().set_cursor_properties)(
                self.device,
                x_hotspot,
                y_hotspot,
                surface.as_ptr(),
            )
        }
    }

    /// `ShowCursor(show)`.
    ///
    /// Returns the previous visibility state as reported by the device (BOOL).
    pub fn show_cursor(&self, show: bool) -> i32 {
        // SAFETY: vtable thunk; `self.device` is a live IDirect3DDevice9.
        unsafe { (self.dev_vtbl().show_cursor)(self.device, i32::from(show)) }
    }

    /// Send `msg` synchronously through the device window's wndproc.
    ///
    /// mtld3d subclasses that wndproc at `CreateDevice`, so tests can
    /// synthesize the messages macdrv posts (e.g. `WM_SIZE`) deterministically.
    pub fn send_window_message(&self, msg: u32, wparam: usize, lparam: isize) -> isize {
        win32::send_message(self.hwnd, msg, wparam, lparam)
    }

    /// user32 `GetCursor` — the thread cursor handle the d3d9 cursor module last pushed.
    ///
    /// Zero means none. The harness runs device calls and the window's wndproc
    /// on this thread, so this observes cursor realization directly.
    pub fn thread_cursor(&self) -> usize {
        win32::get_cursor()
    }

    /// user32 `SetCursor` — clobber the thread cursor.
    ///
    /// Simulates the native cursor taking over while the pointer was outside
    /// the window.
    pub fn set_thread_cursor(&self, cursor: usize) -> usize {
        win32::set_cursor(cursor)
    }

    /// `GetBackBuffer(0, index, MONO)`, asserting success.
    ///
    /// # Panics
    /// Panics if the call fails.
    #[must_use]
    pub fn back_buffer(&self, index: u32) -> Surface<'_> {
        let mut out: *mut c_void = core::ptr::null_mut();
        // SAFETY: vtable thunk; `&mut out` is writable; type 0 = D3DBACKBUFFER_TYPE_MONO.
        let hr =
            unsafe { (self.dev_vtbl().get_back_buffer)(self.device, 0, index, 0, &raw mut out) };
        assert_eq!(hr, 0, "GetBackBuffer({index}) failed: 0x{hr:08X}");
        assert!(!out.is_null(), "GetBackBuffer returned null");
        Surface::from_raw(out)
    }

    /// `Reset` to `width`×`height`, preserving the configured formats.
    ///
    /// Updates [`Self::dims`] on success. Returns the hr.
    pub fn reset(&self, width: u32, height: u32) -> i32 {
        let cfg = HarnessConfig {
            width,
            height,
            back_buffer_format: self.back_buffer_format,
            depth_format: self.depth_format,
            visible: false,
        };
        let mut pp = present_params(&cfg, self.hwnd);
        // SAFETY: vtable thunk; `&mut pp` is writable.
        let hr = unsafe { (self.dev_vtbl().reset)(self.device, (&raw mut pp).cast::<c_void>()) };
        if hr == 0 {
            self.width.set(width);
            self.height.set(height);
        }
        hr
    }

    /// `Reset` with caller-built parameters (for malformed-input tests).
    ///
    /// Returns the hr; does not touch [`Self::dims`].
    pub fn reset_params(&self, pp: &mut D3DPRESENT_PARAMETERS) -> i32 {
        // SAFETY: vtable thunk; `pp` is writable for the call.
        unsafe { (self.dev_vtbl().reset)(self.device, core::ptr::from_mut(pp).cast::<c_void>()) }
    }

    // ── Factory (IDirect3D9) queries ──

    /// `IDirect3D9::CheckDeviceType`.
    pub fn check_device_type(
        &self,
        adapter_format: u32,
        backbuffer_format: u32,
        windowed: bool,
    ) -> i32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe {
            (self.factory_vtbl().check_device_type)(
                self.d3d9,
                0,
                D3DDEVTYPE_HAL,
                adapter_format,
                backbuffer_format,
                i32::from(windowed),
            )
        }
    }

    /// `IDirect3D9::CheckDeviceFormat`.
    pub fn check_device_format(
        &self,
        adapter_format: u32,
        usage: u32,
        resource_type: u32,
        check_format: u32,
    ) -> i32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe {
            (self.factory_vtbl().check_device_format)(
                self.d3d9,
                0,
                D3DDEVTYPE_HAL,
                adapter_format,
                usage,
                resource_type,
                check_format,
            )
        }
    }

    /// `IDirect3D9::CheckDeviceFormatConversion`.
    pub fn check_device_format_conversion(&self, source: u32, target: u32) -> i32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe {
            (self.factory_vtbl().check_device_format_conversion)(
                self.d3d9,
                0,
                D3DDEVTYPE_HAL,
                source,
                target,
            )
        }
    }

    /// `IDirect3D9::GetAdapterCount`.
    #[must_use]
    pub fn adapter_count(&self) -> u32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe { (self.factory_vtbl().get_adapter_count)(self.d3d9) }
    }

    /// `IDirect3D9::GetAdapterIdentifier`, asserting success.
    #[must_use]
    pub fn adapter_identifier(&self) -> D3DADAPTER_IDENTIFIER9 {
        // SAFETY: POD struct overwritten by the call before any field is read.
        let mut id = unsafe { core::mem::zeroed::<D3DADAPTER_IDENTIFIER9>() };
        // SAFETY: vtable thunk; `&mut id` is writable.
        let hr =
            unsafe { (self.factory_vtbl().get_adapter_identifier)(self.d3d9, 0, 0, &raw mut id) };
        expect_ok(hr, "GetAdapterIdentifier");
        id
    }

    /// `IDirect3D9::GetAdapterModeCount`.
    #[must_use]
    pub fn adapter_mode_count(&self, format: u32) -> u32 {
        // SAFETY: vtable thunk; `self.d3d9` is live.
        unsafe { (self.factory_vtbl().get_adapter_mode_count)(self.d3d9, 0, format) }
    }

    /// `IDirect3D9::EnumAdapterModes` into a `D3DDISPLAYMODE`. Returns the hr.
    pub fn enum_adapter_modes(
        &self,
        format: u32,
        index: u32,
        mode: &mut mtld3d_types::D3DDISPLAYMODE,
    ) -> i32 {
        // SAFETY: vtable thunk; `mode` is writable for the call.
        unsafe {
            (self.factory_vtbl().enum_adapter_modes)(
                self.d3d9,
                0,
                format,
                index,
                core::ptr::from_mut(mode).cast::<c_void>(),
            )
        }
    }

    /// `IDirect3D9::GetAdapterDisplayMode` into a `D3DDISPLAYMODE`. Returns the hr.
    pub fn adapter_display_mode(&self, mode: &mut mtld3d_types::D3DDISPLAYMODE) -> i32 {
        // SAFETY: vtable thunk; `mode` is writable for the call.
        unsafe {
            (self.factory_vtbl().get_adapter_display_mode)(
                self.d3d9,
                0,
                core::ptr::from_mut(mode).cast::<c_void>(),
            )
        }
    }

    /// `IDirect3D9::GetDeviceCaps`, asserting success.
    #[must_use]
    pub fn device_caps(&self) -> D3DCAPS9 {
        // SAFETY: POD struct overwritten by the call before any field is read.
        let mut caps = unsafe { core::mem::zeroed::<D3DCAPS9>() };
        // SAFETY: vtable thunk; `&mut caps` is writable.
        let hr = unsafe {
            (self.factory_vtbl().get_device_caps)(self.d3d9, 0, D3DDEVTYPE_HAL, &raw mut caps)
        };
        expect_ok(hr, "GetDeviceCaps");
        caps
    }
}

impl Default for Harness {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if !self.device.is_null() {
            // SAFETY: vtable thunk; `self.device` is live and released exactly once.
            unsafe { (self.dev_vtbl().release)(self.device) };
        }
        // SAFETY: vtable thunk; `self.d3d9` is live and released exactly once.
        unsafe { (self.factory_vtbl().release)(self.d3d9) };
        if self.hwnd != 0 {
            win32::destroy_window(self.hwnd);
        }
    }
}

fn present_params(cfg: &HarnessConfig, hwnd: usize) -> D3DPRESENT_PARAMETERS {
    D3DPRESENT_PARAMETERS {
        back_buffer_width: cfg.width,
        back_buffer_height: cfg.height,
        back_buffer_format: cfg.back_buffer_format,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: hwnd,
        windowed: 1,
        enable_auto_depth_stencil: u32::from(cfg.depth_format.is_some()),
        auto_depth_stencil_format: cfg.depth_format.unwrap_or(0),
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    }
}
