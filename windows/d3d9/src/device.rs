use core::{ffi::c_void, ptr::NonNull};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use log::{debug, error, info, trace, warn};
use mtld3d_core::{
    caps,
    convert::{
        self, FfVsLayout, InputSemantic, d3d_to_metal_primitive, fvf_to_elements,
        resolve_attrs_for_ff, resolve_attrs_for_vs, vertex_count,
    },
    ff_state::{FfState, FfVsDirty},
    format::{FormatMapping, compute_mip_count, compute_mip_size, is_dxt_format, map_d3d_format},
    ids::{BufferId, ProgramId, TextureId},
    page_box::PageBox,
    perf::{
        ApiPerfState, ApiTimer, BindSubCategory, CycleAddTimer, CycleSetTimer, DeviceSubCategory,
        KeysGate,
    },
};
use mtld3d_shared::{
    BlitTextureToBufferParams, CreateColorTargetParams, CreateDepthTextureParams,
    DestroyCommandQueueParams, InPtr, InPtrMut, MetalHandle, OutPtr, ValueIn, VtableThis,
    mtl_handle::{
        CAMetalLayerKind, MTLCommandQueueKind, MTLDeviceKind, MTLTextureKind, NSViewKind,
    },
};
use mtld3d_types::{
    D3DCAPS9, D3DCLEAR_STENCIL, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DDEVICE_CREATION_PARAMETERS,
    D3DDISPLAYMODE, D3DFMT_INDEX16, D3DFMT_INDEX32, D3DFMT_UYVY, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPRESENT_PARAMETERS, D3DPRESENTFLAG_LOCKABLE_BACKBUFFER,
    D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST, D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF,
    D3DRS_ALPHATESTENABLE, D3DRS_AMBIENT, D3DRS_AMBIENTMATERIALSOURCE, D3DRS_BLENDFACTOR,
    D3DRS_BLENDOP, D3DRS_BLENDOPALPHA, D3DRS_CCW_STENCILFAIL, D3DRS_CCW_STENCILFUNC,
    D3DRS_CCW_STENCILPASS, D3DRS_CCW_STENCILZFAIL, D3DRS_CLIPPING, D3DRS_COLORVERTEX,
    D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_DEBUGMONITORTOKEN, D3DRS_DEPTHBIAS,
    D3DRS_DESTBLEND, D3DRS_DESTBLENDALPHA, D3DRS_DIFFUSEMATERIALSOURCE,
    D3DRS_EMISSIVEMATERIALSOURCE, D3DRS_FILLMODE, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_MULTISAMPLEANTIALIAS,
    D3DRS_MULTISAMPLEMASK, D3DRS_NORMALDEGREE, D3DRS_NORMALIZENORMALS, D3DRS_PATCHEDGESTYLE,
    D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN, D3DRS_POSITIONDEGREE, D3DRS_RANGEFOGENABLE,
    D3DRS_SCISSORTESTENABLE, D3DRS_SEPARATEALPHABLENDENABLE, D3DRS_SHADEMODE,
    D3DRS_SLOPESCALEDEPTHBIAS, D3DRS_SPECULARENABLE, D3DRS_SPECULARMATERIALSOURCE, D3DRS_SRCBLEND,
    D3DRS_SRCBLENDALPHA, D3DRS_SRGBWRITEENABLE, D3DRS_STENCILENABLE, D3DRS_STENCILFAIL,
    D3DRS_STENCILFUNC, D3DRS_STENCILMASK, D3DRS_STENCILPASS, D3DRS_STENCILREF,
    D3DRS_STENCILWRITEMASK, D3DRS_STENCILZFAIL, D3DRS_TEXTUREFACTOR, D3DRS_TWEENFACTOR,
    D3DRS_TWOSIDEDSTENCILMODE, D3DRS_VERTEXBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DSAMP_MAXMIPLEVEL, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT,
    D3DTSS_BUMPENVLOFFSET, D3DTSS_BUMPENVLSCALE, D3DTSS_BUMPENVMAT00, D3DTSS_BUMPENVMAT01,
    D3DTSS_BUMPENVMAT10, D3DTSS_BUMPENVMAT11, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DONOTCLIP, D3DUSAGE_DYNAMIC, D3DUSAGE_NONSECURE, D3DUSAGE_NPATCHES, D3DUSAGE_POINTS,
    D3DUSAGE_RENDERTARGET, D3DUSAGE_RTPATCHES, D3DUSAGE_SOFTWAREPROCESSING, D3DUSAGE_WRITEONLY,
    D3DVIEWPORT9, Guid, IDirect3DDevice9Vtbl, RENDER_STATE_COUNT, SAMPLER_STATE_COUNT,
    TEXTURE_STAGE_STATE_COUNT, render_state_defaults,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_FAIL, E_NOINTERFACE, E_NOTIMPL, LOG_TARGET,
    bound_buffers::BoundBuffers,
    bound_rt::BoundRt,
    com_ref::{Bound, CachedComPtr},
    cursor::{self, CursorState},
    direct3d9::{depth_format_has_stencil, is_depth_stencil_format},
    draw::{
        AttrSnapshot, ConstSource, CurrentSnapshot, CurrentSnapshotPtr, DepthScissorFlags,
        DepthStencilFlags, DrawOp, IndexSource, PsSource, PsSourcePtr, RenderStatePtr,
        RenderStateSnapshot, ScratchSlice, StageBinding, VertexSource, VsSource, VsSourcePtr,
        arena_alloc_bytes, build_alpha_ref_bytes, bump_packed_stage_bindings,
    },
    encoder::{
        BlitSide, EncoderThread, FrameData, FrameEncoder, FrameInit, Op, StagingWarmupEntry,
        TextureInfo, TextureUploadJob, VbibWarmupEntry,
    },
    index_buffer::{Direct3DIndexBuffer9, IndexBufferCreateInfo},
    null_out,
    pixel_shader::Direct3DPixelShader9,
    shader_bindings::{CONSTANT_ROWS, PS_FLOAT_CONSTANT_LIMIT, ShaderBindings},
    stage_bindings::{STAGE_COUNT, StageBindings, TextureSwapDelta},
    state_block::{RecordingStateBlock, StateOp},
    surface::Direct3DSurface9,
    texture::{
        Direct3DTexture9, SourceImage, TextureCreateInfo, TextureFlags, TextureInner,
        new_uninit_page_box,
    },
    unix_call::unix_call,
    vertex_buffer::{Direct3DVertexBuffer9, VertexBufferCreateInfo},
    vertex_decl::{Direct3DVertexDeclaration9, VertexDeclCreateInfo},
    vertex_shader::Direct3DVertexShader9,
};

/// Sub-target for accepted-`StretchRect` blit traces.
///
/// Sits under `mtld3d::d3d9::*` so `RUST_LOG=mtld3d::d3d9::blit=trace` opts
/// in granularly without flipping the rest of the d3d9 logger.
const BLIT_TRACE_TARGET: &str = "mtld3d::d3d9::blit";

/// Sub-target for the once-per-distinct texture-create diagnostic in `device_create_texture`.
///
/// Permanent probe (zero-cost when off); gated under its own sub-target so
/// a texture investigation can `RUST_LOG=mtld3d::d3d9::tex=trace` without
/// flipping the rest of the d3d9 logger.
const TEX_TRACE_TARGET: &str = "mtld3d::d3d9::tex";

/// Sub-target for the depth-path diagnostic probes.
///
/// Covers depth-stencil surface binds, the per-stage depth-sampler mask, the
/// per-attachment load action. Permanent probes (zero-cost when off); gated
/// under their own sub-target so a depth / shadow-map investigation can
/// `RUST_LOG=mtld3d::d3d9::depth=trace` without flipping the rest of the
/// d3d9 logger. Mirrored as `encoder.rs::DEPTH_TRACE_TARGET`.
const DEPTH_TRACE_TARGET: &str = "mtld3d::d3d9::depth";

static DIRECT3D_DEVICE9_VTBL: IDirect3DDevice9Vtbl = IDirect3DDevice9Vtbl {
    query_interface: device_query_interface,
    add_ref: device_add_ref,
    release: device_release,
    test_cooperative_level: device_test_cooperative_level,
    get_available_texture_mem: device_get_available_texture_mem,
    evict_managed_resources: device_evict_managed_resources,
    get_direct3d: device_get_direct3d,
    get_device_caps: device_get_device_caps,
    get_display_mode: device_get_display_mode,
    get_creation_parameters: device_get_creation_parameters,
    set_cursor_properties: cursor::device_set_cursor_properties,
    set_cursor_position: cursor::device_set_cursor_position,
    show_cursor: cursor::device_show_cursor,
    create_additional_swap_chain: device_create_additional_swap_chain,
    get_swap_chain: device_get_swap_chain,
    get_number_of_swap_chains: device_get_number_of_swap_chains,
    reset: device_reset,
    present: device_present,
    get_back_buffer: device_get_back_buffer,
    get_raster_status: device_get_raster_status,
    set_dialog_box_mode: device_set_dialog_box_mode,
    set_gamma_ramp: device_set_gamma_ramp,
    get_gamma_ramp: device_get_gamma_ramp,
    create_texture: device_create_texture,
    create_volume_texture: device_create_volume_texture,
    create_cube_texture: device_create_cube_texture,
    create_vertex_buffer: device_create_vertex_buffer,
    create_index_buffer: device_create_index_buffer,
    create_render_target: device_create_render_target,
    create_depth_stencil_surface: device_create_depth_stencil_surface,
    update_surface: device_update_surface,
    update_texture: device_update_texture,
    get_render_target_data: device_get_render_target_data,
    get_front_buffer_data: device_get_front_buffer_data,
    stretch_rect: device_stretch_rect,
    color_fill: device_color_fill,
    create_offscreen_plain_surface: device_create_offscreen_plain_surface,
    set_render_target: device_set_render_target,
    get_render_target: device_get_render_target,
    set_depth_stencil_surface: device_set_depth_stencil_surface,
    get_depth_stencil_surface: device_get_depth_stencil_surface,
    begin_scene: device_begin_scene,
    end_scene: device_end_scene,
    clear: device_clear,
    set_transform: device_set_transform,
    get_transform: device_get_transform,
    multiply_transform: device_multiply_transform,
    set_viewport: device_set_viewport,
    get_viewport: device_get_viewport,
    set_material: device_set_material,
    get_material: device_get_material,
    set_light: device_set_light,
    get_light: device_get_light,
    light_enable: device_light_enable,
    get_light_enable: device_get_light_enable,
    set_clip_plane: device_set_clip_plane,
    get_clip_plane: device_get_clip_plane,
    set_render_state: device_set_render_state,
    get_render_state: device_get_render_state,
    create_state_block: device_create_state_block,
    begin_state_block: device_begin_state_block,
    end_state_block: device_end_state_block,
    set_clip_status: device_set_clip_status,
    get_clip_status: device_get_clip_status,
    get_texture: device_get_texture,
    set_texture: device_set_texture,
    get_texture_stage_state: device_get_texture_stage_state,
    set_texture_stage_state: device_set_texture_stage_state,
    get_sampler_state: device_get_sampler_state,
    set_sampler_state: device_set_sampler_state,
    validate_device: device_validate_device,
    set_palette_entries: device_set_palette_entries,
    get_palette_entries: device_get_palette_entries,
    set_current_texture_palette: device_set_current_texture_palette,
    get_current_texture_palette: device_get_current_texture_palette,
    set_scissor_rect: device_set_scissor_rect,
    get_scissor_rect: device_get_scissor_rect,
    set_software_vertex_processing: device_set_software_vertex_processing,
    get_software_vertex_processing: device_get_software_vertex_processing,
    set_npatch_mode: device_set_npatch_mode,
    get_npatch_mode: device_get_npatch_mode,
    draw_primitive: device_draw_primitive,
    draw_indexed_primitive: device_draw_indexed_primitive,
    draw_primitive_up: device_draw_primitive_up,
    draw_indexed_primitive_up: device_draw_indexed_primitive_up,
    process_vertices: device_process_vertices,
    create_vertex_declaration: device_create_vertex_declaration,
    set_vertex_declaration: device_set_vertex_declaration,
    get_vertex_declaration: device_get_vertex_declaration,
    set_fvf: device_set_fvf,
    get_fvf: device_get_fvf,
    create_vertex_shader: device_create_vertex_shader,
    set_vertex_shader: device_set_vertex_shader,
    get_vertex_shader: device_get_vertex_shader,
    set_vertex_shader_constant_f: device_set_vertex_shader_constant_f,
    get_vertex_shader_constant_f: device_get_vertex_shader_constant_f,
    set_vertex_shader_constant_i: device_set_vertex_shader_constant_i,
    get_vertex_shader_constant_i: device_get_vertex_shader_constant_i,
    set_vertex_shader_constant_b: device_set_vertex_shader_constant_b,
    get_vertex_shader_constant_b: device_get_vertex_shader_constant_b,
    set_stream_source: device_set_stream_source,
    get_stream_source: device_get_stream_source,
    set_stream_source_freq: device_set_stream_source_freq,
    get_stream_source_freq: device_get_stream_source_freq,
    set_indices: device_set_indices,
    get_indices: device_get_indices,
    create_pixel_shader: device_create_pixel_shader,
    set_pixel_shader: device_set_pixel_shader,
    get_pixel_shader: device_get_pixel_shader,
    set_pixel_shader_constant_f: device_set_pixel_shader_constant_f,
    get_pixel_shader_constant_f: device_get_pixel_shader_constant_f,
    set_pixel_shader_constant_i: device_set_pixel_shader_constant_i,
    get_pixel_shader_constant_i: device_get_pixel_shader_constant_i,
    set_pixel_shader_constant_b: device_set_pixel_shader_constant_b,
    get_pixel_shader_constant_b: device_get_pixel_shader_constant_b,
    draw_rect_patch: device_draw_rect_patch,
    draw_tri_patch: device_draw_tri_patch,
    delete_patch: device_delete_patch,
    create_query: device_create_query,
};

/// Number of user-clip-plane storage slots.
///
/// D3D9 defines `D3DMAXUSERCLIPPLANES = 32`; the conformance suite probes
/// indices across `0..2*32`, so the store is sized to cover the full probed
/// range with every index addressable.
const CLIP_PLANE_SLOTS: usize = 64;

// ── DeviceInner — non-repr(C) state behind the inner pointer ──

/// One VB/IB backing queued for seq-gated destruction.
///
/// Pushed by the API thread on Lock-rename and on VB/IB release; drained
/// into `FrameData` at `present()` and handed to the encoder for final
/// cleanup.
pub struct PendingVbibRetention {
    pub buffer_id: BufferId,
    pub page_box: PageBox,
    pub last_submit_seq: u64,
}

bitflags::bitflags! {
    /// Assorted per-device boolean state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DeviceFlags: u8 {
        /// Set after the app called `SetDepthStencilSurface(NULL)` — depth is explicitly absent.
        ///
        /// Distinguishes "no override, use the auto depth" from "explicitly
        /// unbound" so the pipeline's depth/stencil-format snapshot matches
        /// the actual render-pass attachment. Cleared by
        /// `reseed_current_frame` (which restores the default bindings).
        const DEPTH_EXPLICITLY_UNBOUND = 1 << 0;
        /// Set between a successful `BeginScene` and its `EndScene`.
        ///
        /// D3D9 pairs them strictly: `BeginScene` while already in a scene and
        /// `EndScene` without an open scene both return `D3DERR_INVALIDCALL`.
        /// Rendering does not otherwise depend on scene state.
        const IN_SCENE = 1 << 1;
    }
}

pub struct DeviceInner {
    // Metal handles / presentation.
    device_handle: MetalHandle<MTLDeviceKind>,
    queue_handle: MetalHandle<MTLCommandQueueKind>,
    view_handle: MetalHandle<NSViewKind>,
    layer_handle: MetalHandle<CAMetalLayerKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
    depth_stencil_handle: MetalHandle<MTLTextureKind>,
    depth_stencil_format: u32,
    /// Scene / depth-binding boolean state (`DEPTH_EXPLICITLY_UNBOUND` / `IN_SCENE`).
    ///
    /// See [`DeviceFlags`].
    flags: DeviceFlags,
    backbuffer_width: u32,
    backbuffer_height: u32,
    fvf: u32,
    /// Currently-bound vertex declaration (null = none).
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline rather than going through the COM vtable's
    /// `AddRef`/`Release` thunks. Separate from FVF: `SetVertexDeclaration`
    /// and `SetFVF` shadow each other and the most recent wins at snapshot
    /// time (`vertex_decl` takes precedence when non-null).
    vertex_decl: CachedComPtr<Direct3DVertexDeclaration9, Bound>,
    /// Implicit vertex declarations synthesised by `SetFVF`, keyed by FVF.
    ///
    /// D3D9 converts a non-zero FVF into a declaration that
    /// `GetVertexDeclaration` returns; the same FVF always maps to the same
    /// cached object. Each entry is held by one `Bound` (private) refcount so
    /// the public refcount a game observes via `GetVertexDeclaration` reflects
    /// only its own `AddRef`s. Released when `DeviceInner` drops (`HashMap`
    /// value `Drop` runs `K::on_drop`).
    fvf_decl_cache: rustc_hash::FxHashMap<u32, CachedComPtr<Direct3DVertexDeclaration9, Bound>>,
    /// `IDirect3D9`* that created this device.
    ///
    /// Kept so `GetDirect3D` can hand back the parent interface (with
    /// `AddRef`) instead of the `D3DERR_INVALIDCALL` any D3D9 title would
    /// treat as a fatal init failure.
    direct3d: u64,
    /// The owning `Direct3DDevice9`* COM wrapper.
    ///
    /// Stamped after the wrapper is boxed in `CreateDevice`. Stored as `u64`
    /// to mirror `direct3d` and leave `DeviceInner`'s auto-traits unchanged.
    /// Resource `GetDevice` thunks return it (`AddRef`'d) so callers — e.g.
    /// the conformance readback path — get a usable device pointer instead of
    /// an uninitialised out-param.
    device_wrapper: u64,
    /// Saved creation parameters, served verbatim by `GetCreationParameters`.
    creation_adapter: u32,
    creation_device_type: u32,
    creation_behavior_flags: u32,
    creation_focus_window: usize,
    /// Normalised present parameters the implicit swapchain reports.
    ///
    /// Served through `GetSwapChain(0)` / `GetPresentParameters`: dimensions
    /// resolved, back-buffer count clamped to >= 1. Refreshed on `Reset`.
    /// `windowed` also gates `CreateAdditionalSwapChain` (no additional
    /// swapchains while the device is fullscreen).
    present_params: D3DPRESENT_PARAMETERS,
    /// Lazily-created implicit swapchain handed out by `GetSwapChain(0)`.
    ///
    /// Stored as `u64` (like [`direct3d`](Self::direct3d)/`device_wrapper`) so
    /// a raw pointer doesn't change `DeviceInner`'s auto-traits. The device
    /// owns it (the app may keep using it after its own `Release`), so the
    /// shell is leaked at teardown like the device wrapper. `0` until the
    /// first `GetSwapChain`.
    implicit_swapchain: u64,

    /// Lazily-created device-owned implicit render target == backbuffer surface.
    ///
    /// `GetRenderTarget(0)` / `GetBackBuffer(0)` / implicit
    /// `GetSwapChain(0).GetBackBuffer(0)` all return this one object. Stored as
    /// `u64` like [`implicit_swapchain`](Self::implicit_swapchain); created at
    /// refcount 0, finalized at device teardown. `0` until first requested.
    implicit_render_target: u64,

    /// Lazily-created device-owned implicit depth-stencil surface (`GetDepthStencilSurface`).
    ///
    /// Same lifecycle as [`implicit_render_target`](Self::implicit_render_target).
    /// `0` until first requested (and never created when the device has no auto
    /// depth-stencil).
    implicit_depth_stencil: u64,

    // Encoder + frame state.
    encoder: EncoderThread,
    /// Lifetime handle for the detached shader-cache prewarm thread.
    ///
    /// Stored here so `device_release` can stop it before any teardown
    /// step — otherwise an in-flight prewarm `CompileShaderLibrary`
    /// thunk would race with `shutdown_cleanup`'s destroy thunks on
    /// the same `MTLDevice`.
    prewarm: crate::shader_prewarm::PrewarmHandle,
    current_frame: FrameData,
    /// Shared with the encoder thread and the unix completion handler.
    ///
    /// Bumped in `present()` before `send_frame`; completion block
    /// `fetch_max`'s it when a frame retires on the GPU.
    coherent_seq: Arc<AtomicU64>,
    /// Texture-upload retirement seq.
    ///
    /// `fetch_max`'d by the *upload* command buffer's completion handler (a
    /// separate CB committed before the draw CB; see
    /// `SubmitFrameParams::upload_coherent_seq_ptr`). Because that CB retires
    /// ~a frame earlier than the draw CB tracked by `coherent_seq`, a texture
    /// mip's staging reads as retired sooner, so a contended whole-mip
    /// `LockRect` can write in place instead of renaming + preserving.
    /// Texture-staging contention reads this; VB/IB stays on `coherent_seq`
    /// (their backings are consumed by draws, which live in the draw CB).
    upload_coherent_seq: Arc<AtomicU64>,
    /// Live VB/IB retained-`PageBox` byte total, shared with the encoder.
    ///
    /// Mirrors `coherent_seq`'s sharing. The encoder `fetch_add`s on intake
    /// and `fetch_sub`s on drain; the API thread reads it in
    /// `alloc_pagebox_with_recovery` to cap retention before a rename burst
    /// balloons PE-heap usage into 32-bit OOM territory.
    vbib_retained_bytes: Arc<AtomicU64>,
    /// Running total of bytes occupied by live `D3DPOOL_DEFAULT` textures.
    ///
    /// Counts RTs + DEFAULT textures, maintained at `register_texture` /
    /// `deregister_texture`. `GetAvailableTextureMem` reports
    /// `VRAM_BUDGET - this`, so the value visibly decreases as the app
    /// allocates GPU resources.
    vram_bytes_used: Arc<AtomicU64>,
    /// Monotonic submit seq.
    ///
    /// Each `present()` bumps this before stamping it onto the outgoing
    /// `FrameData`. Buffers captured in the frame's draws are stamped with the
    /// pre-bump value (i.e. the seq they're visible to the GPU in).
    current_seq: u64,
    /// All API-thread telemetry.
    ///
    /// Per-category timer buckets, Lock / texture counters, prev-present TSC.
    /// See `mtld3d_core::perf` for the field list. Drained into
    /// `FrameData::perf` at `Present`.
    perf: ApiPerfState,
    /// Retention pipeline for VB/IB `PageBox`es whose in-flight frame hasn't yet retired.
    ///
    /// API thread pushes on Lock-rename and Release; drained into `FrameData`
    /// at `present()` and from there into the encoder's
    /// `pending_vbib_retention` for seq-gated destruction of the Metal wrapper
    /// + drop of the Box.
    vbib_retention_pending: Vec<PendingVbibRetention>,
    /// Byte total of `vbib_retention_pending` queued this frame.
    ///
    /// Not yet handed to the encoder (and thus not yet in
    /// `vbib_retained_bytes`). Added to the shared total when reading the
    /// retention cap so the current frame's renames count immediately; reset
    /// to 0 at `stamp_and_swap` when the queue is handed off.
    pending_retention_bytes: u64,
    /// Proactive retention cap in bytes.
    ///
    /// When `vbib_retained_bytes + pending_retention_bytes` reaches this,
    /// `alloc_pagebox_with_recovery` drains (and, if still over,
    /// mid-frame-submits + GPU-waits) before allocating, bounding peak PE-heap
    /// retention.
    retention_cap_bytes: u64,
    render_states: [u32; RENDER_STATE_COUNT],
    /// Per-slot "have we warned about this unsupported RS write yet?" latch.
    ///
    /// Bit-packed one bit per RS index (`[u64; 4]` covers all 210 slots in
    /// 32 B instead of 210). Prevents log spam while still firing once per
    /// slot per device when a caller writes a non-default value to an RS slot
    /// we don't consume. Access via `rs_warn_fired()` / `mark_rs_warn()`.
    rs_warn_fired: [u64; RENDER_STATE_COUNT.div_ceil(64)],
    ff_state: FfState,
    /// Scissor rect set by `SetScissorRect`.
    ///
    /// The encoder thread reads this each draw and emits a Metal
    /// `setScissorRect` command gated on `D3DRS_SCISSORTESTENABLE`.
    /// `(0, 0, 0, 0)` means "unset — use viewport".
    scissor_rect: [u32; 4],
    /// Viewport set by `SetViewport`.
    ///
    /// Served back by `GetViewport`. Width/height also flow to the encoder so
    /// `setViewport` emits with the actual viewport dimensions instead of the
    /// backbuffer size.
    viewport: D3DVIEWPORT9,
    /// User clip planes set by `SetClipPlane`, served back by `GetClipPlane`.
    ///
    /// CPU round-trip only — GPU application is a no-op (the
    /// `D3DRS_CLIPPLANEENABLE` render state is stored but not consumed), so this
    /// has no rendering effect. The index is clamped into range, so an
    /// out-of-range plane aliases the last slot instead of being rejected.
    clip_planes: [[f32; 4]; CLIP_PLANE_SLOTS],

    // Submodule-owned state. Each group's fields are private to its own
    // submodule; only `group()` / `group_mut()` cross the boundary.
    cursor: CursorState,
    bound_rt: BoundRt,
    bound_buffers: BoundBuffers,
    shader_bindings: ShaderBindings,
    stage_bindings: StageBindings,
    /// In-progress `BeginStateBlock` recording.
    ///
    /// `Some(..)` between a successful `BeginStateBlock` and its matching
    /// `EndStateBlock`. While set, every state-change COM setter diverts its
    /// write into the block instead of the live device — spec-correct replay
    /// semantics for `Apply()` on the resulting state block. Null-safe to read
    /// via `recording_state_block()`; mutating through
    /// `recording_state_block_mut()` is how each setter records its op.
    recording_state_block: Option<Box<RecordingStateBlock>>,
    /// `Some(v)` if `IDirect3DDevice9::Reset` changed `PresentationInterval` since the last frame.
    ///
    /// Consumed by `fresh_frame` and applied by the encoder thread on the next
    /// frame's first `nextDrawable`, matching the spec's "next Present" timing.
    pending_display_sync_enabled: Option<bool>,
    /// The colour render-target binding most recently applied via `SetRenderTarget`.
    ///
    /// `None` means the implicit backbuffer default is in effect. The encoder's
    /// per-frame pass state resets to the backbuffer on every fresh frame, but a
    /// D3D9 render-target binding survives an *internal*
    /// `flush_current_frame_blocking` (a mid-frame readback flush is not a
    /// Present). Re-pushed into the fresh frame after such a flush so draws that
    /// follow a `GetRenderTargetData` keep rendering to the bound RT instead of
    /// silently reverting to the backbuffer format.
    last_color_rt_binding: Option<RtBinding>,
    /// Texture id of the autogen render-target currently bound at RT0, if any.
    ///
    /// When RT0 changes away from it the mip chain is regenerated (a render or
    /// clear into an `D3DUSAGE_AUTOGENMIPMAP` texture must refresh the lower
    /// levels).
    cur_autogen_rt_id: Option<TextureId>,
    /// The depth/stencil attachment most recently applied via `SetDepthStencilSurface`.
    ///
    /// Holds binding + `is_sampleable` + `depth_has_stencil`. `None` means the
    /// implicit auto-depth default is in effect. Like `last_color_rt_binding`,
    /// re-pushed after a mid-frame flush: the encoder's per-frame reset
    /// re-attaches the implicit depth-stencil, but a draw issued after
    /// `SetDepthStencilSurface(NULL)` + readback would then carry a depth
    /// attachment the pipeline declares no format for (Metal rejects the
    /// pipeline-vs-framebuffer depth/stencil mismatch and drops the draw).
    last_depth_binding: Option<(DepthBinding, bool, bool)>,
    /// Live `IDirect3DTexture9` objects.
    ///
    /// Populated in `texture_create` after `Box::into_raw`; entries removed in
    /// `texture_release`'s rc→0 path before the inner Box is dropped. Walked
    /// by `evict_managed_resources` to mark per-mip `dirty` flags so the next
    /// bind replays the staging upload — the spec contract for
    /// `IDirect3DDevice9::EvictManagedResources` is "evict from VRAM, runtime
    /// re-uploads on next use," and lazy upload's bind-time flush is exactly
    /// that re-upload trigger. Mutex contention is zero in steady state
    /// (create / release / Evict are all on the API thread, serially).
    live_textures: Mutex<Vec<*mut TextureInner>>,
    /// Per-draw snapshot dirty-bitmask.
    ///
    /// Each bit marks one `CurrentSnapshot` piece as needing rebuild on the
    /// next Draw. Set by every live-state-path Set* method (after the
    /// `recording_state_block_mut` early-return so recording writes don't
    /// touch it). `emit_snapshot_deltas` walks the bits, rebuilds only the
    /// dirty pieces, and clears the flag.
    ///
    /// `stamp_and_swap` sets this to `SnapshotDirty::all()` on frame
    /// rotation so the first draw of each new frame re-emits every
    /// piece — the cached scratch pointers in `snapshot_cache` all
    /// alias into the previous frame's `ScratchArena`, which is about
    /// to drop.
    snapshot_dirty: SnapshotDirty,
    /// Cached `CurrentSnapshot` pieces from the most recent `emit_snapshot_deltas`.
    ///
    /// Each `Op::SetCurrentSnapshot` op shipped to the encoder is built from
    /// this cache: dirty pieces are rebuilt + the cache field is updated; clean
    /// pieces reuse the cached scratch pointer (same per-frame arena, still
    /// valid). Initial state is `default()` (all `None`); the first draw of
    /// every frame starts with `snapshot_dirty == all()` so every field is
    /// freshly populated before the cached state is composed.
    snapshot_cache: CurrentSnapshot,
    /// Cached `bound_texture_mask` from the most recent `STAGES` rebuild.
    ///
    /// Input to FF VS/PS key construction; not part of `CurrentSnapshot` (the
    /// encoder doesn't need it — `emit_draw` reads textures via
    /// `stage_bindings`).
    cached_bound_texture_mask: u8,
    /// Cached `FfVsLayout` from the most recent `VDECL` rebuild.
    ///
    /// Input to FF VS key construction. Same lifecycle as
    /// `cached_bound_texture_mask`.
    cached_ff_vs_layout: FfVsLayout,
    /// Bit `i` set ⇒ the bound vertex declaration provides input register `vi`.
    ///
    /// Recomputed in the VDECL snapshot (so it tracks both decl and VS changes)
    /// and folded into a programmable `VsSource` so a shader reading an
    /// unprovided input compiles a distinct, zero-filled variant.
    cached_vs_provided_mask: u16,
    /// Running high-water mark of `FrameData.ops.len()`.
    ///
    /// Covers every frame this device has rotated through `stamp_and_swap`.
    /// Used to pre-reserve the new frame's ops Vec so steady-state and
    /// post-burst frames never pay a realloc. Monotonically grows;
    /// memory cost = peak × `size_of::<Op>()` (~72 B).
    peak_ops_count: usize,
}

/// Per-RS-index dirty mask.
///
/// The `SetRenderState` thunk feeds `rs_dirty_mask(index)` to
/// `mark_snapshot_dirty` after the `set_render_state` mutator updates the
/// slot (the mutator itself only marks the FF-VS const rows a write touches).
/// Most RS only flip the `RS` bit; the ones listed below also dirty derived
/// snapshot pieces:
///
/// - alpha test/ref → `ALPHA_REF` (bytes), `VARIANT` (`alpha_func` bit),
///   `PS_SOURCE` (PS variant key changes)
/// - fog enable/mode → `FOG_COLOR`, `VARIANT`, `VS_SOURCE`, `VS_CONST`,
///   `PS_SOURCE` (FF VS reads fog table mode + variant)
/// - fog color → `FOG_COLOR` only
/// - fog start/end/density → `FOG_COLOR`, `VS_CONST` (FF VS computes
///   vertex fog)
/// - texture factor → `PS_CONST` (FF PS reads it)
/// - lighting / material source / vertex blend → `VS_SOURCE`,
///   `VS_CONST`, `PS_SOURCE` (FF VS/PS keys + builder read these)
pub const fn rs_dirty_mask(state: u32) -> SnapshotDirty {
    let rs = SnapshotDirty::RS;
    match state {
        D3DRS_ALPHAREF => rs.union(SnapshotDirty::ALPHA_REF),
        D3DRS_ALPHAFUNC | D3DRS_ALPHATESTENABLE => rs
            .union(SnapshotDirty::ALPHA_REF)
            .union(SnapshotDirty::VARIANT)
            .union(SnapshotDirty::PS_SOURCE),
        // DEPTHBIAS rides the fog_data params row (table fog sources the
        // post-bias fragment depth), so the slot-13 bytes must rebuild — the
        // same `FOG_COLOR` dirty set as an explicit fog-colour change.
        D3DRS_FOGCOLOR | D3DRS_DEPTHBIAS => rs.union(SnapshotDirty::FOG_COLOR),
        D3DRS_FOGENABLE | D3DRS_FOGTABLEMODE | D3DRS_FOGVERTEXMODE => rs
            .union(SnapshotDirty::FOG_COLOR)
            .union(SnapshotDirty::VARIANT)
            .union(SnapshotDirty::VS_SOURCE)
            .union(SnapshotDirty::VS_CONST)
            .union(SnapshotDirty::PS_SOURCE),
        D3DRS_FOGSTART | D3DRS_FOGEND | D3DRS_FOGDENSITY => rs
            .union(SnapshotDirty::FOG_COLOR)
            .union(SnapshotDirty::VS_CONST),
        D3DRS_TEXTUREFACTOR => rs.union(SnapshotDirty::PS_CONST),
        // FLAT vs GOURAUD flips the PS `[[flat]]` varying qualifier (VariantKey
        // flat_shade), and SRGBWRITEENABLE toggles the in-shader linear→sRGB OETF
        // (VariantKey srgb_write); both rebuild the PS source + variant key.
        // (SRGBWRITEENABLE also retains its RS bit elsewhere, keeping the
        // pipeline's SRGB_WRITE flag current.)
        D3DRS_SHADEMODE | D3DRS_SRGBWRITEENABLE => rs
            .union(SnapshotDirty::VARIANT)
            .union(SnapshotDirty::PS_SOURCE),
        D3DRS_LIGHTING
        | D3DRS_AMBIENT
        | D3DRS_AMBIENTMATERIALSOURCE
        | D3DRS_DIFFUSEMATERIALSOURCE
        | D3DRS_SPECULARMATERIALSOURCE
        | D3DRS_EMISSIVEMATERIALSOURCE
        | D3DRS_SPECULARENABLE
        | D3DRS_NORMALIZENORMALS
        | D3DRS_COLORVERTEX
        | D3DRS_RANGEFOGENABLE
        | D3DRS_VERTEXBLEND
        | D3DRS_INDEXEDVERTEXBLENDENABLE => rs
            .union(SnapshotDirty::VS_SOURCE)
            .union(SnapshotDirty::VS_CONST)
            .union(SnapshotDirty::PS_SOURCE),
        // Key-only: flips FfVsFlags::LOCAL_VIEWER (specular view-vector
        // model); no constant section reads it.
        D3DRS_LOCALVIEWER => rs.union(SnapshotDirty::VS_SOURCE),
        _ => rs,
    }
}

bitflags::bitflags! {
    /// Per-draw snapshot dirty mask.
    ///
    /// One bit per cached piece in `FrameEncoder::current_snapshot`. See
    /// `SnapshotCache` doc on `DeviceInner::snapshot_dirty` for lifecycle.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SnapshotDirty: u32 {
        /// `RenderStateSnapshot` (~25 RS slots).
        const RS          = 1 << 0;
        /// `[Option<StageBinding>; STAGE_COUNT]` — bound textures and per-stage sampler state.
        ///
        /// `bound_texture_mask` rebuild is folded into this branch (see
        /// `emit_snapshot_deltas`).
        const STAGES      = 1 << 1;
        /// `has_depth` + `has_stencil` on the current render target.
        const RT_DS       = 1 << 3;
        /// Vertex attribute layout (`AttrSnapshot`: attrs slice, stride, vdecl hash).
        const VDECL       = 1 << 4;
        /// Pipeline variant key.
        const VARIANT     = 1 << 5;
        /// VS source (FF key or programmable `vs_id`).
        const VS_SOURCE   = 1 << 6;
        /// PS source.
        const PS_SOURCE   = 1 << 7;
        /// VS constants slot.
        const VS_CONST    = 1 << 8;
        /// PS constants slot.
        const PS_CONST    = 1 << 9;
        /// Alpha-ref bytes (PS slot 14).
        const ALPHA_REF   = 1 << 10;
        /// Fog-color bytes (PS slot 13).
        const FOG_COLOR   = 1 << 11;
        /// Bump-environment matrix bytes (PS slot 12).
        ///
        /// Per-stage `D3DTSS_BUMPENVMAT*` + luminance, consumed by SM1
        /// `texbem`/`texbeml`/`bem`.
        const BUMP_ENV    = 1 << 12;
        /// VS integer-constant file bytes (vertex slot 14).
        ///
        /// `vs_constants_i`, consumed by a VS reading a dynamic (non-`defi`)
        /// integer constant.
        const VS_CONST_I  = 1 << 13;
    }
}

impl DeviceInner {
    pub const fn scissor_rect(&self) -> [u32; 4] {
        self.scissor_rect
    }

    pub const fn set_scissor_rect(&mut self, r: [u32; 4]) {
        self.scissor_rect = r;
    }

    pub const fn viewport(&self) -> D3DVIEWPORT9 {
        self.viewport
    }

    pub fn set_viewport(&mut self, v: D3DVIEWPORT9) {
        self.viewport = v;
        // D3D9 viewport z-range fixup: the far plane forwarded to the encoder
        // is clamped to at least `min_z + 0.001` so a degenerate (`min_z ==
        // max_z`) or inverted (`max_z < min_z`) range collapses to a tiny
        // forward range instead of mapping every fragment to a single depth.
        // `self.viewport` keeps the raw values so GetViewport round-trips
        // unchanged. Ordinary `[min_z, max_z]` ranges (`max_z >= min_z + 0.001`)
        // are left untouched.
        let (x, y, width, height, min_z) = (v.x, v.y, v.width, v.height, v.min_z);
        let max_z = v.max_z.max(v.min_z + 0.001);
        self.push_op(Box::new(move |enc| {
            enc.set_viewport(x, y, width, height, min_z, max_z);
        }));
        // Viewport feeds XYZRHW row 0 (`[vp_w, vp_h, vp_x, vp_y]`). Mark
        // WV — `emit_snapshot_deltas` dispatches the XYZRHW row 0 write
        // unconditionally when `ff_dirty` is non-empty and `key.has_rhw`,
        // so any FfVsDirty bit will refresh row 0. Picking WV keeps the
        // mark conceptually paired with row 0's content slot.
        self.ff_state
            .mark_ff_vs_dirty(mtld3d_core::ff_state::FfVsDirty::WV);
        // Row 0 lives in the FF VS const section, gated by VS_CONST.
        // Internalized here so callers can't forget — a missing mark
        // leaves the next FF draw reading a stale viewport transform.
        let mask = self.ff_aware_mask(SnapshotDirty::VS_CONST);
        self.mark_snapshot_dirty(mask);
    }

    /// Store a user clip plane set via `SetClipPlane`.
    ///
    /// CPU round-trip only — see the `clip_planes` field doc. The index is
    /// clamped into range so an out-of-range plane aliases the last slot
    /// rather than panicking.
    pub fn set_clip_plane(&mut self, index: u32, plane: [f32; 4]) {
        let slot = (index as usize).min(CLIP_PLANE_SLOTS - 1);
        self.clip_planes[slot] = plane;
    }

    /// Read back a user clip plane for `GetClipPlane`.
    ///
    /// Unset slots read back the zero-initialised default.
    pub fn clip_plane(&self, index: u32) -> [f32; 4] {
        let slot = (index as usize).min(CLIP_PLANE_SLOTS - 1);
        self.clip_planes[slot]
    }

    pub const fn ff_state(&self) -> &FfState {
        &self.ff_state
    }

    pub const fn ff_state_mut(&mut self) -> &mut FfState {
        &mut self.ff_state
    }

    pub const fn cursor(&self) -> &CursorState {
        &self.cursor
    }

    pub const fn cursor_mut(&mut self) -> &mut CursorState {
        &mut self.cursor
    }

    pub const fn bound_rt(&self) -> &BoundRt {
        &self.bound_rt
    }

    pub const fn bound_rt_mut(&mut self) -> &mut BoundRt {
        &mut self.bound_rt
    }

    pub const fn bound_buffers(&self) -> &BoundBuffers {
        &self.bound_buffers
    }

    pub const fn bound_buffers_mut(&mut self) -> &mut BoundBuffers {
        &mut self.bound_buffers
    }

    pub const fn shader_bindings(&self) -> &ShaderBindings {
        &self.shader_bindings
    }

    pub const fn shader_bindings_mut(&mut self) -> &mut ShaderBindings {
        &mut self.shader_bindings
    }

    pub const fn vertex_decl(&self) -> *mut Direct3DVertexDeclaration9 {
        self.vertex_decl.raw()
    }

    /// Bind `new` as the current vertex declaration. Pass null to clear.
    ///
    /// `CachedComPtr::adopt` `AddRefs` the new pointer (no-op for null);
    /// assignment Drops the old slot, which Releases the previous refcount.
    ///
    /// Returns whether the bound pointer changed. A bound decl is kept
    /// alive by its slot refcount, so identical pointers mean the same
    /// immutable object (same elements/hash) — callers gate the
    /// expensive VDECL re-resolve on this.
    pub fn replace_vertex_decl(&mut self, new: *mut Direct3DVertexDeclaration9) -> bool {
        let changed = self.vertex_decl.raw() != new;
        // SAFETY: `new` is null or a live IDirect3DVertexDeclaration9 from
        // a Set* thunk / state-block apply; AddRef/Release thunks valid
        // for our lifetime.
        self.vertex_decl = unsafe { CachedComPtr::adopt(new) };
        if changed {
            // VDECL change can flip vs_key.has_rhw (row 0 layout switches
            // between XYZRHW viewport and WV transposed) and
            // vs_key.vertex_blend_count (gates PALETTE rows 95+).
            // `has_rhw` also feeds `lit = !has_rhw && D3DRS_LIGHTING != 0`,
            // so it indirectly changes MATERIAL/LIGHTS extent. Under
            // per-section emit, mark every section that depends on the
            // layout. TT is the one section unaffected (per-stage TTFF
            // gates it, not VDECL).
            self.ff_state.mark_ff_vs_dirty(
                mtld3d_core::ff_state::FfVsDirty::WV
                    | mtld3d_core::ff_state::FfVsDirty::PROJ
                    | mtld3d_core::ff_state::FfVsDirty::FOG
                    | mtld3d_core::ff_state::FfVsDirty::AMBIENT
                    | mtld3d_core::ff_state::FfVsDirty::MATERIAL
                    | mtld3d_core::ff_state::FfVsDirty::LIGHTS
                    | mtld3d_core::ff_state::FfVsDirty::PALETTE,
            );
        }
        changed
    }

    /// Get (or lazily synthesise + cache) the implicit vertex declaration for a non-zero `fvf`.
    ///
    /// The returned pointer is borrowed: the cache keeps the object alive via
    /// a `Bound` (private) refcount, so the public refcount stays zero until a
    /// game `Get`s it. Returns null only if the FVF produced an unbindable
    /// element array (should not happen for a real FVF).
    fn get_or_create_fvf_decl(&mut self, fvf: u32) -> *mut Direct3DVertexDeclaration9 {
        if let Some(cached) = self.fvf_decl_cache.get(&fvf) {
            return cached.raw();
        }
        let (mut elements, _stride) = fvf_to_elements(fvf);
        elements.push(mtld3d_types::D3DDECL_END);
        let device_inner = std::ptr::from_mut::<Self>(self);
        let Some(decl) = Direct3DVertexDeclaration9::new(&VertexDeclCreateInfo {
            device_inner,
            elements: &elements,
        }) else {
            return core::ptr::null_mut();
        };
        let decl_ptr = Box::into_raw(Box::new(decl));
        // The wrapper is born with public refcount 1. Register its device
        // reference like any child, then hand that public reference to the cache
        // as a `Bound` (private) refcount and drop the public one — so the public
        // count reflects only the game's own GetVertexDeclaration AddRefs and
        // the device-forward stays balanced: the
        // register here is matched by the public release below (and re-acquired
        // on a game `Get`).
        // SAFETY: `decl_ptr` is a freshly created, live declaration at refcount 1.
        unsafe { crate::com_ref::com_register_child(decl_ptr) };
        // SAFETY: `decl_ptr` is a freshly-boxed, live wrapper.
        let cached = unsafe { CachedComPtr::<Direct3DVertexDeclaration9, Bound>::adopt(decl_ptr) };
        // SAFETY: `decl_ptr` is live and `vtbl()` returns its installed vtable.
        let release = unsafe { (*decl_ptr).vtbl().release };
        // SAFETY: `release` is the matching Release thunk; the `Bound` ref
        // adopted above keeps the wrapper alive as public drops 1 -> 0.
        unsafe { release(decl_ptr.cast::<c_void>()) };
        self.fvf_decl_cache.insert(fvf, cached);
        decl_ptr
    }

    /// Apply D3D9 `SetFVF` semantics for a non-zero `fvf`.
    ///
    /// Record the FVF and bind its implicit declaration as the current vertex
    /// declaration (the most-recent of `SetFVF` / `SetVertexDeclaration`
    /// wins). `fvf == 0` is a no-op on the binding, matching the driver.
    /// Returns whether the bound declaration changed (callers gate snapshot
    /// dirtying on this).
    pub fn bind_fvf_decl(&mut self, fvf: u32) -> bool {
        if fvf == 0 {
            return false;
        }
        let decl = self.get_or_create_fvf_decl(fvf);
        self.fvf = fvf;
        self.replace_vertex_decl(decl)
    }

    pub const fn stage_bindings(&self) -> &StageBindings {
        &self.stage_bindings
    }

    pub const fn stage_bindings_mut(&mut self) -> &mut StageBindings {
        &mut self.stage_bindings
    }

    pub fn set_fvf_field(&mut self, fvf: u32) {
        self.fvf = fvf;
        // FVF change can flip the same `vs_key` fields that VDECL does
        // (has_rhw, vertex_blend_count). Mirror the conservative mark
        // in `replace_vertex_decl`.
        self.ff_state.mark_ff_vs_dirty(
            mtld3d_core::ff_state::FfVsDirty::WV
                | mtld3d_core::ff_state::FfVsDirty::PROJ
                | mtld3d_core::ff_state::FfVsDirty::FOG
                | mtld3d_core::ff_state::FfVsDirty::AMBIENT
                | mtld3d_core::ff_state::FfVsDirty::MATERIAL
                | mtld3d_core::ff_state::FfVsDirty::LIGHTS
                | mtld3d_core::ff_state::FfVsDirty::PALETTE,
        );
    }

    pub const fn fvf_field(&self) -> u32 {
        self.fvf
    }

    /// In-progress `BeginStateBlock` recording, if any.
    ///
    /// Every state-change COM vtable entry point checks this: when `Some`,
    /// the change is recorded into the block and the live device is left
    /// untouched.
    pub fn recording_state_block_mut(&mut self) -> Option<&mut RecordingStateBlock> {
        self.recording_state_block.as_deref_mut()
    }

    /// Mark all snapshot pieces as dirty.
    ///
    /// Used by `stamp_and_swap` (arena rotation invalidates every cached
    /// scratch pointer), `reset_to_defaults` (every input reset), and
    /// state-block `apply_to` (touches many pieces — coarse is fine).
    pub fn mark_snapshot_dirty_all(&mut self) {
        self.snapshot_dirty.insert(SnapshotDirty::all());
    }

    /// Insert specific dirty bits for a Set* on the live state path.
    ///
    /// Cheaper than `mark_snapshot_dirty_all` — only the listed pieces
    /// get rebuilt on the next draw; clean pieces reuse the cached
    /// scratch pointers in `snapshot_cache`.
    pub fn mark_snapshot_dirty(&mut self, bits: SnapshotDirty) {
        self.snapshot_dirty.insert(bits);
    }

    /// Strip FF-only dirty bits from `mask` if the corresponding shader path is programmable.
    ///
    /// Used by Set* sites whose effect on `VS_SOURCE` / `VS_CONST` /
    /// `PS_SOURCE` / `PS_CONST` is mediated by FF state (transforms, lights,
    /// material, `bound_texture_mask` feeding FF VS/PS keys, etc.) — when the
    /// bound shader is programmable, those pieces don't depend on FF state.
    ///
    /// Do NOT use for Set* sites that change the source path itself
    /// (`SetVertexShader`, `SetPixelShader`) or write directly to the
    /// shader-binding constants (`SetVertexShaderConstantF`,
    /// `SetPixelShaderConstantF`) — those dirties are unconditional.
    pub fn ff_aware_mask(&self, mask: SnapshotDirty) -> SnapshotDirty {
        let mut result = mask;
        // A pre-transformed (POSITIONT/XYZRHW) layout bypasses a bound VS —
        // the draw runs the FF pre-transformed path regardless — so
        // FF-mediated dirties still feed the VS side. `cached_ff_vs_layout`
        // can lag one SetFVF/SetVertexDeclaration (it is rebuilt at snapshot
        // time); those two sites compensate by dirtying VS_CONST/VARIANT
        // unconditionally when RHW-ness may flip.
        if !self.shader_bindings.vertex_shader().is_null() && !self.cached_ff_vs_layout.has_rhw() {
            // Programmable VS bound: ff_state + bound_texture_mask
            // don't feed it, so changes to those don't affect VS source
            // or VS constants slice.
            result.remove(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
        }
        if !self.shader_bindings.pixel_shader().is_null() {
            // Same for programmable PS.
            result.remove(SnapshotDirty::PS_SOURCE | SnapshotDirty::PS_CONST);
        }
        result
    }

    /// Start a new recording.
    ///
    /// `BeginStateBlock`-only path — returns `false` if a recording is
    /// already in progress (D3D9 spec reject).
    pub fn begin_state_block_recording(&mut self) -> bool {
        if self.recording_state_block.is_some() {
            return false;
        }
        self.recording_state_block = Some(Box::new(RecordingStateBlock::new()));
        true
    }

    /// Finish the in-progress recording and hand ownership back to the caller.
    ///
    /// Returns `None` if no recording was active.
    pub const fn end_state_block_recording(&mut self) -> Option<Box<RecordingStateBlock>> {
        self.recording_state_block.take()
    }

    /// True while a `BeginStateBlock` recording is open.
    ///
    /// D3D9 rejects `Apply`/`Capture`/`CreateStateBlock` with `INVALIDCALL`
    /// during recording.
    pub const fn is_state_block_recording(&self) -> bool {
        self.recording_state_block.is_some()
    }

    /// Reconstruct a `&mut DeviceInner` from the opaque `inner: u64` field.
    pub fn from_ptr(ptr: u64) -> &'static mut Self {
        // SAFETY: `ptr` is the `Direct3DDevice9::inner` field — a stable
        // `*mut DeviceInner` produced when the device wrapper was created.
        // The inner outlives the wrapper, so the borrow is valid.
        unsafe { &mut *(ptr as *mut Self) }
    }

    /// Submit seq of the *next* frame that will be sent to the encoder.
    ///
    /// Stamped onto bound VB/IB at Draw snapshot time.
    pub const fn current_seq(&self) -> u64 {
        self.current_seq
    }

    /// Push a VB/IB backing into the retention pipeline.
    ///
    /// Called from `vb_lock` / `ib_lock` rename paths and from VB/IB release
    /// on refcount→0. Drained into `FrameData` at `present()` and from there
    /// into the encoder's retention queue; destruction of the wrapped
    /// `MTLBuffer` + drop of the `PageBox` happens once
    /// `coherent_seq >= last_submit_seq`.
    pub fn queue_vbib_retention(
        &mut self,
        buffer_id: BufferId,
        page_box: PageBox,
        last_submit_seq: u64,
    ) {
        // Count locally so the retention cap sees this frame's renames
        // before the encoder intakes them into `vbib_retained_bytes`.
        // Reset at `stamp_and_swap` when the queue is handed off.
        self.pending_retention_bytes += page_box.len() as u64;
        self.vbib_retention_pending.push(PendingVbibRetention {
            buffer_id,
            page_box,
            last_submit_seq,
        });
    }

    /// Push an inline, op-stream-ordered `Staged` VB/IB dirty-range upload.
    ///
    /// `page_box` is a transient snapshot of the dirtied bytes taken on the
    /// API thread at `Unlock`; the encoder wraps it and uploads `[0, size)`
    /// into the buffer's device buffer at `dst_offset` (renaming first if a
    /// draw earlier in the open pass already read the range — see
    /// `FrameEncoder::apply_stage_upload`). Pushing as an `Op` (rather than
    /// a frame-head drain) is what lets the encoder see the upload in draw
    /// order. Counts into `pending_retention_bytes` so the retention cap
    /// sees the transient before the encoder intakes it. No Metal thunk
    /// runs on the API thread here — just a `PageBox` move + `Vec::push`.
    pub fn push_stage_upload(
        &mut self,
        buffer_id: BufferId,
        page_box: PageBox,
        dst_offset: u32,
        size: u32,
    ) {
        self.pending_retention_bytes += page_box.len() as u64;
        self.push_op_inline(crate::encoder::Op::StageUpload {
            buffer_id,
            page_box,
            dst_offset,
            size,
        });
    }

    /// Pointer to the shared `coherent_seq` atomic.
    ///
    /// Read on Lock to decide VB/IB rename, read on the encoder thread to
    /// drain retention queues, and passed across the PE/Unix boundary so the
    /// submit completion handler can bump it.
    pub const fn coherent_seq_arc(&self) -> &Arc<AtomicU64> {
        &self.coherent_seq
    }

    /// The texture-upload retirement atomic.
    ///
    /// Read on texture `LockRect` (instead of `coherent_seq`) to decide
    /// staging contention. See [`Self::upload_coherent_seq`].
    pub const fn upload_coherent_seq_arc(&self) -> &Arc<AtomicU64> {
        &self.upload_coherent_seq
    }

    /// Build a fresh `FrameData` matching the device's current backbuffer / queue / layer handles.
    ///
    /// Used to seed the replacement frame when swapping at `Present` or at
    /// `flush_current_frame_blocking`. Takes `&mut self` because a pending
    /// `PresentationInterval` change from `device_reset` is consumed here so
    /// the encoder can apply it on the next frame's first `nextDrawable`.
    pub const fn fresh_frame(&mut self) -> FrameData {
        FrameData::new(&FrameInit {
            device_handle: self.device_handle,
            queue_handle: self.queue_handle,
            backbuffer_handle: self.backbuffer_handle,
            layer_handle: self.layer_handle,
            view_handle: self.view_handle,
            backbuffer_width: self.backbuffer_width,
            backbuffer_height: self.backbuffer_height,
            backbuffer_format: mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
            depth_texture: self.depth_stencil_handle,
            depth_has_stencil: depth_format_has_stencil(self.depth_stencil_format),
            apply_display_sync_enabled: self.pending_display_sync_enabled.take(),
        })
    }

    /// Stamp per-frame counters + `submit_seq` onto `frame`, swap it in for `current_frame`.
    ///
    /// Returns the stamped outgoing frame ready to hand to the encoder.
    /// Shared between `Present` and `flush_current_frame_blocking`.
    fn stamp_and_swap(&mut self, new_frame: FrameData, no_present: bool) -> FrameData {
        let mut frame = core::mem::replace(&mut self.current_frame, new_frame);
        // Pre-reserve the new frame's ops Vec to the running peak so
        // it never reallocs in steady-state — and so that a post-burst
        // dip doesn't shrink capacity (causing the next burst to
        // realloc again). High-water mark monotonically grows; memory
        // cost is one Op slot (~72 B) per peak op. The first frame
        // sees `peak_ops_count = 0` and pays the initial doubling;
        // every subsequent frame reuses the peak.
        self.peak_ops_count = self.peak_ops_count.max(frame.ops_len());
        self.current_frame.reserve_ops(self.peak_ops_count);
        // Sample the API→encoder Vec<Op> footprint *before* draining
        // the perf state — capacity is read off the outgoing frame and
        // the realloc counter is taken (drained to 0) from the same
        // frame. Plumbs through `FramePerfPayload` so the encoder
        // thread's `log_frame_summary` can surface them in the
        // `Per-frame allocator footprint` section alongside the
        // encoder-side `cmd_vec` row.
        let op_vec_capacity_bytes = frame.op_vec_capacity_bytes();
        let op_vec_realloc_bytes = frame.take_op_vec_realloc_bytes();
        self.perf.drain_into_payload(frame.perf_mut());
        frame
            .perf_mut()
            .set_op_vec_metrics(op_vec_capacity_bytes, op_vec_realloc_bytes);
        frame.set_vbib_retentions(core::mem::take(&mut self.vbib_retention_pending));
        // `Staged` uploads ride the op stream (inline `Op::StageUpload`),
        // so they were already moved into `frame.ops` at `Unlock`. Handed
        // off — the encoder now owns counting their `PageBox` bytes into
        // the shared `vbib_retained_bytes` at intake.
        self.pending_retention_bytes = 0;
        frame.set_no_present(no_present);

        let this_seq = self.current_seq;
        self.current_seq = self.current_seq.saturating_add(1);
        frame.set_submit_fence(
            this_seq,
            Arc::as_ptr(&self.coherent_seq) as u64,
            Arc::as_ptr(&self.upload_coherent_seq) as u64,
        );
        frame.set_retained_bytes_ptr(Arc::as_ptr(&self.vbib_retained_bytes) as u64);
        // Every cached snapshot pointer in the encoder's CurrentSnapshot
        // aliases into the outgoing frame's `ScratchArena`, which is
        // about to drop after the encoder drains it. Force the API
        // thread to re-emit every Op::Set* on the first draw of the
        // new frame.
        self.snapshot_dirty = SnapshotDirty::all();
        frame
    }

    /// Submit the current frame's accumulated ops synchronously.
    ///
    /// Then continue with a fresh empty frame. Used by `LockRect` on the
    /// backbuffer and `GetRenderTargetData` to ensure the GPU has executed
    /// every draw issued this frame before the readback blit samples the
    /// backbuffer. Present is suppressed for this submission so the drawable
    /// is not consumed.
    pub fn flush_current_frame_blocking(&mut self) {
        let fresh = self.fresh_frame();
        let frame = self.stamp_and_swap(fresh, true);
        self.encoder.mid_frame_submit(frame);
        // The fresh frame's pass state defaults to the backbuffer. A D3D9
        // render-target binding survives an internal flush (this is not a
        // Present), so re-assert it; otherwise a draw issued after a
        // `GetRenderTargetData` readback would silently render to the
        // backbuffer's format instead of the bound RT's.
        // Clone (not Copy) out of the persistent binding: `TextureInfo` and the
        // binding enums are wide aggregates, and this flush path is rare.
        if let Some(info) = self.last_color_rt_binding.clone() {
            self.push_color_rt_binding_op(info);
        }
        if let Some((binding, is_sampleable, has_stencil)) = self.last_depth_binding.clone() {
            self.push_depth_binding_op(binding, is_sampleable, has_stencil);
        }
    }

    /// Push the encoder op that binds `binding` as the depth/stencil attachment.
    ///
    /// Factored out of `device_set_depth_stencil_surface` so
    /// `flush_current_frame_blocking` can re-assert the persistent binding.
    fn push_depth_binding_op(
        &mut self,
        binding: DepthBinding,
        is_sampleable: bool,
        depth_has_stencil: bool,
    ) {
        self.push_op(Box::new(move |enc| {
            let depth_texture = match binding {
                DepthBinding::None => MetalHandle::NULL,
                DepthBinding::Eager(h) => h,
                // SAFETY: `get_or_create_texture` returns a Metal texture
                // handle from the typed `texture_cache` via `.raw()`.
                DepthBinding::Lazy(info) => unsafe {
                    MetalHandle::<MTLTextureKind>::new(enc.get_or_create_texture(&info))
                },
            };
            enc.set_depth_stencil_attachment(depth_texture, is_sampleable, depth_has_stencil);
        }));
    }

    /// Push the encoder op that binds `info` as the colour render target.
    ///
    /// Factored out of `device_set_render_target` so `flush_current_frame_
    /// blocking` can re-assert the persistent binding into the fresh frame.
    fn push_color_rt_binding_op(&mut self, info: RtBinding) {
        self.push_op(Box::new(move |enc| {
            let (handle, w, h, fmt, has_alpha) = match info {
                RtBinding::Backbuffer {
                    handle,
                    width,
                    height,
                } => (
                    handle,
                    width,
                    height,
                    mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
                    // The backbuffer is an alpha-bearing A8R8G8B8 target
                    // (see `PassState::reset_frame`), so its destination-alpha
                    // blend factors resolve unclamped.
                    true,
                ),
                RtBinding::StandaloneColor {
                    handle,
                    format,
                    has_alpha,
                    width,
                    height,
                } => (handle, width, height, format, has_alpha),
                RtBinding::Texture {
                    info,
                    has_alpha,
                    width,
                    height,
                } => {
                    let fmt = info.pixel_format;
                    let h = enc.get_or_create_texture(&info);
                    // SAFETY: `get_or_create_texture` returns a Metal texture
                    // handle from the encoder's typed `texture_cache` via `.raw()`.
                    (
                        unsafe { MetalHandle::<MTLTextureKind>::new(h) },
                        width,
                        height,
                        fmt,
                        has_alpha,
                    )
                }
            };
            enc.set_color_render_target(handle, w, h, fmt, has_alpha);
        }));
    }

    /// Cheap alloc-recovery tier.
    ///
    /// Sends a synchronous `DrainRetiredNow` to the encoder so it drains
    /// retention items whose seq has already retired. No submit, no GPU
    /// wait — only useful when the encoder is sitting on drainable
    /// retention between frames. Returns when drain completes; caller
    /// retries `try_new_uninit` on success.
    pub fn drain_retention_now(&self) {
        self.encoder.drain_retired_now();
    }

    /// Heavy alloc-recovery tier.
    ///
    /// Same submission path as `flush_current_frame_blocking` (no Present,
    /// drawable not consumed), but the encoder additionally waits for GPU
    /// completion of the submitted seq and drains retention before
    /// returning — so on return the global allocator has freed bytes that
    /// include same-frame retentions which `drain_retention_now` couldn't
    /// release.
    pub fn mid_frame_submit_for_alloc(&mut self) {
        let fresh = self.fresh_frame();
        let frame = self.stamp_and_swap(fresh, true);
        self.encoder.mid_frame_submit_for_alloc(frame);
    }

    /// Two-tier fallible-alloc recovery for VB/IB Lock-rename.
    ///
    /// The common path returns immediately when `try_new_uninit` succeeds.
    /// The cheap fallback runs `drain_retention_now` and retries — that
    /// frees retention items whose seq has already retired. The heavy
    /// fallback runs `mid_frame_submit_for_alloc` (commit + GPU wait +
    /// drain) which also releases same-frame retentions whose seq matches
    /// the in-progress frame. If even the heavy tier fails to satisfy the
    /// alloc, we fall through to `new_uninit`'s panic — the address space
    /// is genuinely exhausted and aborting is the right outcome.
    pub fn alloc_pagebox_with_recovery(&mut self, logical_len: usize) -> PageBox {
        // Proactive memory cap. The reactive `try_new_uninit`-failure tiers
        // below only trip when the 32-bit address space is *already*
        // exhausted — far too late, since by then the game is thrashing.
        // So before allocating, if live VB/IB retention is at the cap,
        // drain retired backings (cheap) and, if still over, force a
        // mid-frame submit + GPU-wait so this frame's renames can retire
        // and free. Bounds peak PE-heap retention well below the OOM cliff.
        if self.retention_cap_bytes != 0 {
            let retained =
                self.vbib_retained_bytes.load(Ordering::Acquire) + self.pending_retention_bytes;
            if retained >= self.retention_cap_bytes {
                // Reuses the alloc-recovery counters/row — same drain +
                // mid-frame-submit operations, just proactively triggered
                // by the cap rather than by a hard alloc failure.
                self.perf.bump_alloc_recovery_drain();
                self.drain_retention_now();
                let after =
                    self.vbib_retained_bytes.load(Ordering::Acquire) + self.pending_retention_bytes;
                if after >= self.retention_cap_bytes {
                    self.perf.bump_alloc_recovery_submit();
                    self.mid_frame_submit_for_alloc();
                }
            }
        }
        if let Some(b) = PageBox::try_new_uninit(logical_len) {
            return b;
        }
        self.perf.bump_alloc_recovery_drain();
        self.drain_retention_now();
        if let Some(b) = PageBox::try_new_uninit(logical_len) {
            return b;
        }
        self.perf.bump_alloc_recovery_submit();
        self.mid_frame_submit_for_alloc();
        // Fall through to the panicking path on second failure —
        // either we've recovered enough address space and this
        // succeeds, or we're truly OOM and aborting is correct.
        PageBox::new_uninit(logical_len)
    }

    /// Swap in a fresh frame and send the full op list to the encoder.
    ///
    /// Clears and attachment changes flow through `push_op` closures inside
    /// the frame itself, so no per-Device clear snapshot is needed.
    ///
    /// This is also where the per-frame perf counters are published:
    /// `api_thread_cycles` is stashed into the outgoing frame verbatim;
    /// `present_block_cycles` (the backpressure wait from the *previous*
    /// Present's `send_frame`) is stashed into the incoming fresh frame so
    /// the encoder's next summary can read it.
    pub fn present(&mut self, new_frame: FrameData) {
        let frame = self.stamp_and_swap(new_frame, false);

        // The block we measure belongs to the frame that will next be
        // observed by the encoder — the one we just swapped in. The
        // `CycleSetTimer` writes into that frame's `present_block_cycles`
        // when it drops at end of scope.
        let _stall = CycleSetTimer::start(self.current_frame.perf_mut().present_block_cycles_ptr());
        self.encoder.send_frame(frame);
    }

    pub const fn perf_mut(&mut self) -> &mut ApiPerfState {
        &mut self.perf
    }

    /// Raw pointer to the embedded `ApiPerfState`.
    ///
    /// For `ApiTimer::start` at the top of every COM vtable fn. SAFETY at
    /// the call site: the timer is dropped before the fn returns and the COM
    /// object holds a ref to `DeviceInner` for its entire lifetime.
    pub const fn perf_ptr(&mut self) -> *mut ApiPerfState {
        &raw mut self.perf
    }

    /// Null-tolerant wrapper: returns `null_mut()` for a null device.
    ///
    /// Else the embedded `ApiPerfState` pointer. Used by every resource
    /// `*_timer` helper (`vb_timer`, `tex_timer`, …) that may see a null
    /// `device_inner` on standalone surfaces.
    pub fn perf_ptr_of(dev: *mut Self) -> *mut ApiPerfState {
        if dev.is_null() {
            core::ptr::null_mut()
        } else {
            // SAFETY: caller guarantees a non-null valid device pointer
            // for the duration of the timer (same contract as
            // `from_ptr`).
            unsafe { (*dev).perf_ptr() }
        }
    }

    pub fn shutdown(&mut self) {
        self.encoder.shutdown();
    }

    pub fn push_op(&mut self, op: Box<dyn FnOnce(&mut FrameEncoder) + Send>) {
        self.current_frame.push_op(op);
    }

    /// Forwarder for `FrameData::push_op_inline`.
    ///
    /// Used by the hot draw path to emit `Op::Set*` + `Op::Draw` without
    /// per-op heap alloc.
    pub fn push_op_inline(&mut self, op: crate::encoder::Op) {
        self.current_frame.push_op_inline(op);
    }

    /// Queue an eager `MTLTexture` create on the current frame.
    ///
    /// The encoder drains the queue at `run_frame`'s head into one batched
    /// `CreateTexturesBatch` thunk, so subsequent draw closures hit the
    /// texture cache instead of cache-missing on first bind.
    pub fn push_texture_warmup(&mut self, info: TextureInfo) {
        self.current_frame.push_texture_warmup(info);
    }

    /// Queue an eager VB/IB `MTLBuffer` wrap on the current frame.
    ///
    /// Same drain semantics as `push_texture_warmup`.
    pub fn push_buffer_warmup(&mut self, entry: VbibWarmupEntry) {
        self.current_frame.push_buffer_warmup(entry);
    }

    /// Queue an eager texture-staging `MTLBuffer` wrap.
    ///
    /// Drained after the texture warmup so the parent's `texture_cache`
    /// entry exists.
    pub fn push_staging_warmup(&mut self, entry: StagingWarmupEntry) {
        self.current_frame.push_staging_warmup(entry);
    }

    /// Register a freshly-created `TextureInner` in the live-texture registry.
    ///
    /// So `evict_managed_resources` can iterate live textures. The pointer
    /// is the same `Box::into_raw` result that backs the COM wrapper's
    /// `inner` field. Single-threaded API contract: lock contention is
    /// zero in steady state.
    pub fn register_texture(&self, ti: *mut TextureInner) {
        // SAFETY: `ti` is a freshly-built (or rehydrating) live `TextureInner`.
        let tex = unsafe { &*ti };
        if tex.is_default_pool() {
            self.vram_bytes_used
                .fetch_add(tex.allocated_bytes(), Ordering::AcqRel);
        }
        self.live_textures
            .lock()
            .expect("live_textures mutex poisoned")
            .push(ti);
    }

    /// Drop a `TextureInner` from the live-texture registry.
    ///
    /// Called from `texture_release`'s rc→0 path **before** the inner Box is
    /// freed, so the registry never holds a dangling pointer.
    pub fn deregister_texture(&self, ti: *mut TextureInner) {
        let mut live = self
            .live_textures
            .lock()
            .expect("live_textures mutex poisoned");
        if let Some(pos) = live.iter().position(|&p| p == ti) {
            live.swap_remove(pos);
            // Release the registry lock before the VRAM accounting below — it
            // touches only the atomic, not the texture list.
            drop(live);
            // SAFETY: `ti` is still a live `TextureInner` (deregister runs
            // before the Box is freed); only subtract once, gated on the
            // registry having actually held it.
            let tex = unsafe { &*ti };
            if tex.is_default_pool() {
                self.vram_bytes_used
                    .fetch_sub(tex.allocated_bytes(), Ordering::AcqRel);
            }
        }
    }

    /// `IDirect3DDevice9::EvictManagedResources` body.
    ///
    /// Walks the live-textures registry, marks every previously-uploaded mip
    /// dirty (via `texture::evict_mark_dirty`), and pushes one
    /// `destroy_cached_texture` closure per affected texture. The next
    /// bind-time `flush_dirty_mips` repopulates fresh `MTLTextures` from the
    /// still-alive PE-side staging Arc — exactly the spec contract "evict
    /// from VRAM, runtime re-uploads on next use". Render targets are
    /// filtered out by `evict_mark_dirty`; their cache entries stay intact.
    pub fn evict_managed_resources(&mut self) {
        let live: Vec<*mut TextureInner> = self
            .live_textures
            .lock()
            .expect("live_textures mutex poisoned")
            .clone();
        let mut to_evict: Vec<TextureId> = Vec::new();
        for ti_ptr in live {
            // SAFETY: `ti_ptr` is a snapshot from `live_textures`; entries
            // are removed on `TextureInner` drop, so the pointer is live
            // for the duration of this loop iteration.
            let ti = unsafe { &mut *ti_ptr };
            if let Some(tex_id) = crate::texture::evict_mark_dirty(ti) {
                to_evict.push(tex_id);
            }
        }
        let evicted_count = to_evict.len();
        for tex_id in to_evict {
            self.push_op(Box::new(move |enc: &mut FrameEncoder| {
                enc.destroy_cached_texture(tex_id);
            }));
        }
        mtld3d_shared::log_once_info!(
            target: TEX_TRACE_TARGET,
            "EvictManagedResources: marked {evicted_count} textures dirty (cache eviction queued)"
        );
    }

    /// Finalize the visibility query whose END frame retires at `target_seq`.
    ///
    /// The encoder waits (via `WaitForGpuRetire` thunk → Metal
    /// `waitUntilCompleted`) only when `coherent_seq < target_seq`;
    /// otherwise it just runs intake locally. `target_seq == 0` (END closure
    /// not yet processed: game called `Issue(END)` but not Present) skips the
    /// round-trip entirely so the FLUSH poll loop can return `S_FALSE` fast.
    pub fn encoder_intake_visibility_for(&self, target_seq: u64) {
        if target_seq == 0 {
            return;
        }
        self.encoder.intake_visibility_for(target_seq);
    }

    pub const fn render_state(&self, index: usize) -> u32 {
        self.render_states[index]
    }

    /// Returns whether the stored value actually changed.
    ///
    /// Callers gate `mark_snapshot_dirty` on this: a same-value write
    /// produces a byte-identical `RenderStateSnapshot`/FF key, so re-marking
    /// the snapshot dirty would force an identical rebuild on the next draw.
    pub fn set_render_state(&mut self, index: usize, value: u32) -> bool {
        self.warn_rs_non_default_once(index, value);
        let prev = self.render_states[index];
        self.render_states[index] = value;
        let changed = prev != value;
        if changed {
            // RS-driven FF VS const-buffer rows under per-section
            // emit. The encoder mirror has to catch each change before
            // the next FF draw reads from it. Two flavors here:
            //
            //   (a) RS values that supply the *contents* of a row.
            //       FOGSTART/END/DENSITY feed row 8; AMBIENT feeds row 9.
            //
            //   (b) RS values that change the *extent* the shader reads
            //       OR change whether a section is gated on/off. Because
            //       per-section emit writes only the section that changed,
            //       these must be marked explicitly — nothing else rewrites
            //       the full extent to cover them.
            //
            //       - FOGENABLE/VERTEXMODE/TABLEMODE flip vs_key.fog_mode,
            //         which toggles row 8 between zero-fill and the
            //         actual fog params.
            //       - LIGHTING flips vs_key.lighting_enabled, which
            //         changes both the MATERIAL extent (1 row unlit vs
            //         4-5 rows lit) and whether LIGHTS rows are read.
            //       - SPECULARENABLE flips vs_key.specular_enable,
            //         which adds row 14 (material.power) to the read
            //         extent.
            //       - VERTEXBLEND/INDEXEDVERTEXBLENDENABLE gates the
            //         palette section (rows 95+).
            //
            // Other RS writes don't feed the FF VS const buffer or
            // change which sections the shader reads.
            let bits = match u32::try_from(index).ok() {
                Some(
                    D3DRS_FOGSTART | D3DRS_FOGEND | D3DRS_FOGDENSITY | D3DRS_FOGENABLE
                    | D3DRS_FOGVERTEXMODE | D3DRS_FOGTABLEMODE,
                ) => FfVsDirty::FOG,
                Some(D3DRS_AMBIENT) => FfVsDirty::AMBIENT,
                Some(D3DRS_LIGHTING) => FfVsDirty::MATERIAL | FfVsDirty::LIGHTS,
                Some(D3DRS_SPECULARENABLE) => FfVsDirty::MATERIAL,
                Some(D3DRS_VERTEXBLEND | D3DRS_INDEXEDVERTEXBLENDENABLE) => FfVsDirty::PALETTE,
                _ => FfVsDirty::empty(),
            };
            if !bits.is_empty() {
                self.ff_state.mark_ff_vs_dirty(bits);
            }
        }
        changed
    }

    /// Returns whether the once-per-slot RS warn latch for `index` has fired.
    const fn rs_warn_fired(&self, index: usize) -> bool {
        (self.rs_warn_fired[index / 64] & (1u64 << (index % 64))) != 0
    }

    /// Sets the once-per-slot RS warn latch for `index`.
    const fn mark_rs_warn(&mut self, index: usize) {
        self.rs_warn_fired[index / 64] |= 1u64 << (index % 64);
    }

    fn warn_rs_non_default_once(&mut self, index: usize, value: u32) {
        static RS_DEFAULTS: [u32; RENDER_STATE_COUNT] = render_state_defaults();

        if index >= RENDER_STATE_COUNT {
            return;
        }
        if value == RS_DEFAULTS[index] {
            if mtld3d_core::state_trace::enabled() {
                log::trace!(
                    target: mtld3d_core::state_trace::TARGET,
                    "D3DRS_{index} = {value:#x} (default — write suppressed in warn machinery)"
                );
            }
            return;
        }
        if self.rs_warn_fired(index) {
            return;
        }
        let class = rs_classify(
            u32::try_from(index).expect("D3DRS index fits u32 by RENDER_STATE_COUNT bound"),
        );
        if matches!(class, RsClass::Consumed) {
            if mtld3d_core::state_trace::enabled() {
                let default = RS_DEFAULTS[index];
                log::trace!(
                    target: mtld3d_core::state_trace::TARGET,
                    "D3DRS_{index} Consumed = {value:#x} (default {default:#x})"
                );
            }
            return;
        }
        self.mark_rs_warn(index);
        let default = RS_DEFAULTS[index];
        match class {
            RsClass::Consumed => {} // unreachable given early-return above
            RsClass::PortCandidate(feat) => {
                warn!(
                    target: LOG_TARGET,
                    "D3DRS_{index} = {value:#x} (default {default:#x}) set but {feat} not implemented"
                );
            }
            RsClass::Obsolete(reason) => {
                info!(
                    target: LOG_TARGET,
                    "D3DRS_{index} = {value:#x} (default {default:#x}) no Metal analog — {reason}"
                );
            }
            RsClass::NotImplemented => {
                warn!(
                    target: LOG_TARGET,
                    "D3DRS_{index} = {value:#x} (default {default:#x}) written but not consumed"
                );
            }
        }
    }

    pub const fn render_states(&self) -> &[u32; RENDER_STATE_COUNT] {
        &self.render_states
    }

    /// `IDirect3DDevice9::Reset` analog of the destruction path.
    ///
    /// Returns the device to the state a fresh `CreateDevice` would have
    /// produced, minus the cursor subclass (per-spec, cursor settings survive
    /// Reset) and the silent-write warn latches (those are process-lifetime
    /// telemetry, not device state).
    ///
    /// Caller is responsible for replacing the implicit backbuffer +
    /// depth/stencil `MTLTextures` *before* calling this — the new handles
    /// flow into the next frame via `fresh_frame`, but the viewport push
    /// here references the new dimensions.
    pub fn reset_to_defaults(&mut self) {
        self.bound_rt.teardown();
        // Reset reverts the colour target to the implicit backbuffer and the
        // depth/stencil to the implicit auto-depth default.
        self.last_color_rt_binding = None;
        self.last_depth_binding = None;
        self.bound_buffers.teardown();
        self.stage_bindings
            .reset_to_defaults(&[mtld3d_types::sampler_state_defaults(); STAGE_COUNT]);
        self.replace_vertex_decl(core::ptr::null_mut());
        self.shader_bindings
            .replace_vertex_shader(core::ptr::null_mut());
        self.shader_bindings
            .replace_pixel_shader(core::ptr::null_mut());

        self.fvf = 0;
        self.render_states = render_state_defaults();
        self.ff_state = FfState::new();
        // Reset abandons any open scene; a following EndScene must fail.
        self.flags.remove(DeviceFlags::IN_SCENE);
        // Scissor defaults to the full target, like the viewport reseed below.
        self.scissor_rect = [0, 0, self.backbuffer_width, self.backbuffer_height];

        // Drop any in-flight state-block recording; per spec, Reset
        // invalidates an open Begin/EndStateBlock pair.
        self.recording_state_block = None;

        // Viewport reseed mirrors `set_viewport` — push the op so the
        // encoder's pass-state picks up the default before the first
        // post-Reset draw.
        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: self.backbuffer_width,
            height: self.backbuffer_height,
            min_z: 0.0,
            max_z: 1.0,
        };
        self.set_viewport(viewport);
        // Wipe any cached snapshot — every input was just reset.
        self.snapshot_dirty = SnapshotDirty::all();
    }

    /// Update the implicit backbuffer Metal handle after `device_reset` recreates it.
    ///
    /// The old handle is destroyed by the caller via `DestroyResourcesBulk`
    /// *before* this setter; the next `fresh_frame` stamps the new handle
    /// into the outgoing `FrameData`.
    pub const fn set_backbuffer_handle(&mut self, handle: MetalHandle<MTLTextureKind>) {
        self.backbuffer_handle = handle;
    }

    /// Update the implicit depth/stencil Metal handle after `device_reset` recreates it.
    ///
    /// Same lifecycle as `set_backbuffer_handle`.
    pub const fn set_depth_stencil_handle(&mut self, handle: MetalHandle<MTLTextureKind>) {
        self.depth_stencil_handle = handle;
    }

    /// Update the device's backbuffer dimensions after `device_reset` honours a resize.
    ///
    /// Every other consumer reads dims off `DeviceInner` (viewport
    /// defaults, `GetBackBuffer`, `GetRenderTarget`, `fresh_frame`), so
    /// this single setter propagates everywhere.
    pub const fn set_backbuffer_dims(&mut self, width: u32, height: u32) {
        self.backbuffer_width = width;
        self.backbuffer_height = height;
    }

    /// Queue a `PresentationInterval` change for the next frame's first `nextDrawable`.
    ///
    /// Drained by `fresh_frame`. Spec-compliant timing — a synchronous
    /// layer-property write from the API thread races the encoder's
    /// in-flight submission.
    pub const fn queue_display_sync_change(&mut self, enabled: bool) {
        self.pending_display_sync_enabled = Some(enabled);
    }

    /// Drive the encoder thread to run `reset_cleanup`.
    ///
    /// Drain retention queues + GPU-idle wait so the caller can safely
    /// destroy the implicit backbuffer + depth/stencil `MTLTextures` and
    /// create their replacements. Returns when the encoder has
    /// acknowledged.
    pub fn encoder_reset(&self) {
        self.encoder.reset();
    }

    /// Drop the empty `current_frame` left behind by `flush_current_frame_blocking`.
    ///
    /// Replace it with a fresh one carrying the device's *current*
    /// backbuffer / depth handles. Used by `device_reset` after the
    /// implicit backbuffer + depth textures are recreated: without it,
    /// the next `Present` would send the stale handles the pre-Reset
    /// flush baked into `current_frame` and the unix-side `submit_frame`
    /// would dereference the freed `MTLTextures`.
    pub fn reseed_current_frame(&mut self) {
        self.current_frame = self.fresh_frame();
        // Reseeding restores the default RT/depth bindings, so any prior
        // explicit `SetDepthStencilSurface(NULL)` override no longer applies.
        self.flags.remove(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND);
    }

    /// Apply an implicit backbuffer resize triggered by a chrome-shrink `WM_SIZE`.
    ///
    /// Mirrors `device_reset`'s size-change pipeline (drain → destroy
    /// old textures → adopt new dims → push `drawableSize` → recreate
    /// textures → reseed `current_frame` → re-push default viewport) but
    /// **skips** `reset_to_defaults` — the game didn't request a Reset,
    /// so its render states / textures / vertex bindings must survive.
    /// No-op when dims already match. Caller drives this from the
    /// cursor subclass wndproc on the API thread; encoder is paused
    /// inside `flush_current_frame_blocking` for the destroy/create
    /// span so no in-flight cmdbuf references the freed handles.
    pub fn apply_auto_resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }
        if new_width == self.backbuffer_width && new_height == self.backbuffer_height {
            return;
        }
        debug!(
            target: LOG_TARGET,
            "apply_auto_resize: backbuffer {}x{} → {new_width}x{new_height} (WM_SIZE-driven)",
            self.backbuffer_width, self.backbuffer_height,
        );

        self.flush_current_frame_blocking();
        self.encoder_reset();

        let old_handles: [u64; 2] = [
            self.backbuffer_handle.raw(),
            self.depth_stencil_handle.raw(),
        ];
        let live: Vec<u64> = old_handles.iter().copied().filter(|&h| h != 0).collect();
        if !live.is_empty() {
            let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
                kind: mtld3d_shared::mtl::DestroyKind::Texture,
                pad0: 0,
                handles_ptr: live.as_ptr() as u64,
                count: u32::try_from(live.len()).expect("at most 2 handles"),
                pad1: 0,
            };
            unix_call(&mut destroy);
        }

        self.set_backbuffer_dims(new_width, new_height);

        if !self.layer_handle.is_null() {
            let mut size = mtld3d_shared::SetLayerDrawableSizeParams {
                layer_handle: self.layer_handle,
                width: new_width,
                height: new_height,
            };
            unix_call(&mut size);
        }

        let mut bb_params = mtld3d_shared::CreateBackbufferParams {
            device_handle: self.device_handle,
            width: new_width,
            height: new_height,
            texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut bb_params);
        if status != 0 || bb_params.texture_handle.is_null() {
            error!(
                target: LOG_TARGET,
                "apply_auto_resize: CreateBackbuffer failed (0x{status:08X}) — device unusable",
            );
            self.set_backbuffer_handle(MetalHandle::NULL);
            self.set_depth_stencil_handle(MetalHandle::NULL);
            return;
        }
        self.set_backbuffer_handle(bb_params.texture_handle);

        if self.depth_stencil_format != 0 {
            let Some(pixel_format) =
                mtld3d_core::format::map_d3d_depth_format(self.depth_stencil_format)
            else {
                error!(
                    target: LOG_TARGET,
                    "apply_auto_resize: depth_stencil_format {} has no Metal mapping — depth lost",
                    self.depth_stencil_format,
                );
                self.set_depth_stencil_handle(MetalHandle::NULL);
                return;
            };
            let mut ds_params = CreateDepthTextureParams {
                device_handle: self.device_handle,
                width: new_width,
                height: new_height,
                pixel_format,
                pad0: 0,
                texture_handle: MetalHandle::NULL,
            };
            let status = unix_call(&mut ds_params);
            if status != 0 || ds_params.texture_handle.is_null() {
                error!(
                    target: LOG_TARGET,
                    "apply_auto_resize: CreateDepthTexture failed (0x{status:08X}) — depth lost",
                );
                self.set_depth_stencil_handle(MetalHandle::NULL);
                return;
            }
            self.set_depth_stencil_handle(ds_params.texture_handle);
        } else {
            self.set_depth_stencil_handle(MetalHandle::NULL);
        }

        self.reseed_current_frame();

        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: new_width,
            height: new_height,
            min_z: 0.0,
            max_z: 1.0,
        };
        self.set_viewport(viewport);
        self.scissor_rect = [0, 0, new_width, new_height];
    }
}

// ── IDirect3DDevice9 COM object ──

/// Parameters for `Direct3DDevice9::new`.
///
/// Grouped so the constructor doesn't take a dozen positional arguments.
pub struct DeviceCreateInfo {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub queue_handle: MetalHandle<MTLCommandQueueKind>,
    pub view_handle: MetalHandle<NSViewKind>,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    pub depth_stencil_handle: MetalHandle<MTLTextureKind>,
    pub depth_stencil_format: u32,
    pub backbuffer_width: u32,
    pub backbuffer_height: u32,
    pub encoder: EncoderThread,
    pub prewarm: crate::shader_prewarm::PrewarmHandle,
    pub current_frame: FrameData,
    pub render_states: [u32; RENDER_STATE_COUNT],
    pub sampler_states: [[u32; SAMPLER_STATE_COUNT]; STAGE_COUNT],
    pub direct3d: u64,
    pub creation_adapter: u32,
    pub creation_device_type: u32,
    pub creation_behavior_flags: u32,
    pub creation_focus_window: usize,
    /// Normalised present parameters served by the implicit swapchain and refreshed on `Reset`.
    ///
    /// Dimensions resolved, back-buffer count clamped to >= 1.
    pub present_params: D3DPRESENT_PARAMETERS,
    /// HWND the Metal layer is attached to.
    ///
    /// Either `device_window` or `focus_window` from
    /// `D3DPRESENT_PARAMETERS`. Used by the cursor subclass; may be null
    /// in headless smoke tests.
    pub hwnd: *mut c_void,
    /// Integer multiplier applied to the Win32 HCURSOR bitmap.
    ///
    /// Scales it to match the display's `backingScaleFactor`. Sourced
    /// from `AttachMetalLayerParams.backing_scale` on the unix side,
    /// clamped to `[1, 8]`. 1 is the no-op fast path.
    pub cursor_scale: u32,
}

#[repr(C)]
pub struct Direct3DDevice9 {
    vtbl: *const IDirect3DDevice9Vtbl,
    refcount: u32,
    inner: *mut DeviceInner,
}

impl Direct3DDevice9 {
    pub fn new(info: DeviceCreateInfo) -> Self {
        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: info.backbuffer_width,
            height: info.backbuffer_height,
            min_z: 0.0,
            max_z: 1.0,
        };

        let coherent_seq = Arc::new(AtomicU64::new(0));
        let upload_coherent_seq = Arc::new(AtomicU64::new(0));
        let vbib_retained_bytes = Arc::new(AtomicU64::new(0));

        let inner = Box::into_raw(Box::new(DeviceInner {
            device_handle: info.device_handle,
            queue_handle: info.queue_handle,
            view_handle: info.view_handle,
            layer_handle: info.layer_handle,
            backbuffer_handle: info.backbuffer_handle,
            depth_stencil_handle: info.depth_stencil_handle,
            depth_stencil_format: info.depth_stencil_format,
            flags: DeviceFlags::empty(),
            backbuffer_width: info.backbuffer_width,
            backbuffer_height: info.backbuffer_height,
            fvf: 0,
            vertex_decl: CachedComPtr::null(),
            fvf_decl_cache: rustc_hash::FxHashMap::default(),
            direct3d: info.direct3d,
            device_wrapper: 0,
            creation_adapter: info.creation_adapter,
            creation_device_type: info.creation_device_type,
            creation_behavior_flags: info.creation_behavior_flags,
            creation_focus_window: info.creation_focus_window,
            present_params: info.present_params,
            implicit_swapchain: 0,
            implicit_render_target: 0,
            implicit_depth_stencil: 0,
            encoder: info.encoder,
            prewarm: info.prewarm,
            current_frame: info.current_frame,
            coherent_seq,
            upload_coherent_seq,
            vbib_retained_bytes,
            vram_bytes_used: Arc::new(AtomicU64::new(0)),
            // Start at 1 so `current_seq - 1` never underflows.
            current_seq: 1,
            perf: ApiPerfState::new(),
            vbib_retention_pending: Vec::new(),
            pending_retention_bytes: 0,
            retention_cap_bytes: crate::config::CONFIG.vbib_retention_cap_bytes,
            render_states: info.render_states,
            rs_warn_fired: [0; RENDER_STATE_COUNT.div_ceil(64)],
            ff_state: FfState::new(),
            // D3D9 default scissor rect covers the full backbuffer; like the
            // viewport, SetRenderTarget and Reset re-cover the new target.
            scissor_rect: [0, 0, info.backbuffer_width, info.backbuffer_height],
            viewport,
            clip_planes: [[0.0; 4]; CLIP_PLANE_SLOTS],
            cursor: CursorState::new(info.hwnd, info.cursor_scale),
            bound_rt: BoundRt::new(info.backbuffer_width, info.backbuffer_height),
            bound_buffers: BoundBuffers::new(),
            shader_bindings: ShaderBindings::new(),
            stage_bindings: StageBindings::new(&info.sampler_states),
            recording_state_block: None,
            pending_display_sync_enabled: None,
            last_color_rt_binding: None,
            cur_autogen_rt_id: None,
            last_depth_binding: None,
            live_textures: Mutex::new(Vec::new()),
            snapshot_dirty: SnapshotDirty::all(),
            snapshot_cache: CurrentSnapshot::EMPTY,
            cached_bound_texture_mask: 0,
            cached_ff_vs_layout: FfVsLayout::default(),
            cached_vs_provided_mask: u16::MAX,
            peak_ops_count: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_DEVICE9_VTBL,
            refcount: 1,
            inner,
        }
    }

    pub fn inner(&self) -> &'static mut DeviceInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `device_release` at
        // refcount zero, so it stays live for every live wrapper
        // reference.
        unsafe { &mut *self.inner }
    }

    /// Raw `DeviceInner` pointer.
    ///
    /// Used by resource-wrapper constructors (and `ApiTimer` guards
    /// inside them) that need a stable back-ref without holding a Rust
    /// reference across the resource's lifetime.
    pub const fn inner_ptr(&self) -> *mut DeviceInner {
        self.inner
    }

    /// COM `AddRef` used when a child resource's `GetDevice` hands this device back to the caller.
    ///
    /// Bumps the wrapper refcount directly — D3D9 objects are
    /// single-threaded, so this matches `device_add_ref`'s effect without
    /// routing another module through the vtable thunk.
    pub const fn add_ref_self(&mut self) -> u32 {
        self.refcount += 1;
        self.refcount
    }

    pub fn fvf(&self) -> u32 {
        self.inner().fvf
    }

    /// Mutation goes through `inner()`'s raw-pointer indirection, so `&self` is sufficient.
    pub fn set_fvf(&self, fvf: u32) {
        self.inner().fvf = fvf;
    }
}

impl DeviceInner {
    pub const fn device_handle(&self) -> MetalHandle<MTLDeviceKind> {
        self.device_handle
    }

    pub const fn queue_handle(&self) -> MetalHandle<MTLCommandQueueKind> {
        self.queue_handle
    }

    /// Owning `Direct3DDevice9`* wrapper, or null until `CreateDevice` stamps it.
    ///
    /// Stamped via [`set_device_wrapper`](Self::set_device_wrapper).
    /// Returned (`AddRef`'d) by resource `GetDevice` thunks.
    pub const fn device_wrapper(&self) -> *mut c_void {
        self.device_wrapper as *mut c_void
    }

    /// Stamp the owning wrapper pointer once the COM object is boxed in `CreateDevice`.
    pub fn set_device_wrapper(&mut self, wrapper: *mut c_void) {
        self.device_wrapper = wrapper as u64;
    }

    /// The implicit backbuffer `MTLTexture`.
    ///
    /// The single drawable also backs every swapchain's `GetBackBuffer`
    /// surface.
    pub const fn backbuffer_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.backbuffer_handle
    }

    pub const fn backbuffer_width(&self) -> u32 {
        self.backbuffer_width
    }

    pub const fn backbuffer_height(&self) -> u32 {
        self.backbuffer_height
    }

    /// The device's current default depth-stencil `MTLTexture`.
    ///
    /// Null when the device has no auto depth-stencil. Recreated on
    /// `Reset` / window resize, so the implicit depth-stencil surface
    /// resolves it live each call.
    pub const fn depth_stencil_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.depth_stencil_handle
    }

    /// Whether a depth-stencil surface is currently bound to the device.
    ///
    /// Distinct from "the device has an auto depth-stencil": an explicit
    /// `SetDepthStencilSurface(NULL)` unbinds depth even though the auto
    /// texture still exists, and a custom depth surface bound for an
    /// offscreen render target is reflected in `bound_rt` rather than in the
    /// auto `depth_stencil_handle`. Used by `Clear` to reject
    /// `D3DCLEAR_ZBUFFER`/`_STENCIL` with no depth attachment.
    pub const fn depth_stencil_bound(&self) -> bool {
        if self.flags.contains(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND) {
            return false;
        }
        // A custom depth surface bound via `SetDepthStencilSurface` counts as
        // bound regardless of the auto handle; otherwise the device default
        // auto depth-stencil is in effect iff its handle exists.
        !self.bound_rt.depth_stencil().is_null() || !self.depth_stencil_handle.is_null()
    }

    /// The device's default depth-stencil format (`D3DFMT_*`), or `0` when none.
    pub const fn depth_stencil_format(&self) -> u32 {
        self.depth_stencil_format
    }

    /// Normalised present parameters the implicit swapchain reports.
    pub const fn present_params(&self) -> &D3DPRESENT_PARAMETERS {
        &self.present_params
    }

    /// `true` when the device is fullscreen.
    ///
    /// `CreateAdditionalSwapChain` is rejected in that mode.
    pub const fn is_fullscreen(&self) -> bool {
        self.present_params.windowed == 0
    }

    /// The device's presentation window.
    ///
    /// The `device_window` it was created with, falling back to the focus
    /// window — the default target a `CreateAdditionalSwapChain` request
    /// resolves its dimensions against.
    pub const fn window(&self) -> usize {
        if self.present_params.device_window != 0 {
            self.present_params.device_window
        } else {
            self.creation_focus_window
        }
    }

    /// Get-or-create the device-owned implicit swapchain (`GetSwapChain(0)`), cached as a `u64`.
    ///
    /// Created at refcount 0 — the caller AddRef-forwards it so the first
    /// hand-out bumps the device. The shell is leaked at teardown.
    pub fn get_or_create_implicit_swapchain(
        &mut self,
    ) -> *mut crate::swapchain::Direct3DSwapChain9 {
        if self.implicit_swapchain == 0 {
            let sc = crate::swapchain::Direct3DSwapChain9::new_implicit(
                core::ptr::from_mut(self),
                self.present_params,
            );
            self.implicit_swapchain = Box::into_raw(Box::new(sc)) as u64;
        }
        self.implicit_swapchain as *mut crate::swapchain::Direct3DSwapChain9
    }

    /// Get-or-create the device-owned implicit render target == backbuffer surface.
    ///
    /// Cached as a `u64`. `GetRenderTarget(0)`, `GetBackBuffer(0)` and
    /// the implicit `GetSwapChain(0).GetBackBuffer(0)` all return this one
    /// object (the `pRenderTarget == pBackBuffer` identity the suite
    /// checks). Its container is the implicit swapchain. Created at
    /// refcount 0.
    pub fn get_or_create_implicit_render_target(
        &mut self,
    ) -> *mut crate::surface::Direct3DSurface9 {
        if self.implicit_render_target == 0 {
            let container = self.get_or_create_implicit_swapchain() as u64;
            let surf = crate::surface::Direct3DSurface9::new_implicit_backbuffer(
                core::ptr::from_mut(self),
                container,
            );
            self.implicit_render_target = Box::into_raw(Box::new(surf)) as u64;
        }
        self.implicit_render_target as *mut crate::surface::Direct3DSurface9
    }

    /// Get-or-create the device-owned implicit depth-stencil surface (`GetDepthStencilSurface`).
    ///
    /// Cached as a `u64`. Its container is the device. Returns null when
    /// the device has no auto depth-stencil (the caller maps that to
    /// `D3DERR_NOTFOUND`). Created at refcount 0.
    pub fn get_or_create_implicit_depth_stencil(
        &mut self,
    ) -> *mut crate::surface::Direct3DSurface9 {
        if self.depth_stencil_handle.is_null() {
            return core::ptr::null_mut();
        }
        if self.implicit_depth_stencil == 0 {
            let container = self.device_wrapper;
            let surf = crate::surface::Direct3DSurface9::new_implicit_depth_stencil(
                core::ptr::from_mut(self),
                container,
            );
            self.implicit_depth_stencil = Box::into_raw(Box::new(surf)) as u64;
        }
        self.implicit_depth_stencil as *mut crate::surface::Direct3DSurface9
    }

    /// Refresh the stored present parameters after a successful `Reset`.
    ///
    /// Back-buffer count clamped to >= 1, matching `CreateDevice`.
    pub fn set_present_params(&mut self, mut pp: D3DPRESENT_PARAMETERS) {
        pp.back_buffer_count = pp.back_buffer_count.max(1);
        self.present_params = pp;
        // Keep the cached implicit swapchain (if it has already been handed
        // out) in lockstep: GetSwapChain(0).GetPresentParameters must reflect
        // the post-Reset geometry, not the values captured when it was created.
        if self.implicit_swapchain != 0 {
            let sc = self.implicit_swapchain as *mut crate::swapchain::Direct3DSwapChain9;
            // SAFETY: `implicit_swapchain` is a device-owned `Box::into_raw`,
            // freed only at device teardown, so it is live here.
            unsafe { (*sc).set_present_params(pp) };
        }
    }
}

// ── RtBinding — attachment info captured on API thread for pass break ──
//
// The `SetRenderTarget` closure runs on the encoder thread and needs to
// either (1) create/fetch a Metal texture for a texture-backed RT surface,
// or (2) restore the backbuffer handle for a standalone surface. We
// capture everything it needs at call time since the surface pointer may
// be released before the closure runs.

#[derive(Clone)]
enum RtBinding {
    Backbuffer {
        handle: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
    },
    /// A standalone `CreateRenderTarget` colour surface.
    ///
    /// `parent_texture` is null (so it is not texture-backed) but it
    /// carries its own persistent `metal_color_handle` distinct from the
    /// backbuffer, plus its own format and dimensions. Bound directly —
    /// unlike `Backbuffer`, the format is the surface's actual format, not
    /// the hard-wired backbuffer `Bgra8Unorm`.
    StandaloneColor {
        handle: MetalHandle<MTLTextureKind>,
        format: mtld3d_shared::mtl::PixelFormat,
        /// Whether the surface's D3D format has a real alpha channel.
        ///
        /// Carried separately because the Metal `format` can't distinguish
        /// X8R8G8B8 (no alpha) from A8R8G8B8 (both `Bgra8Unorm`). Feeds
        /// the pipeline snapshot's `COLOR_HAS_ALPHA` bit.
        has_alpha: bool,
        width: u32,
        height: u32,
    },
    Texture {
        info: TextureInfo,
        /// See `StandaloneColor::has_alpha`.
        has_alpha: bool,
        width: u32,
        height: u32,
    },
}

// ── IUnknown implementation (IDirect3DDevice9) ──

/// Constructs an `ApiTimer` keyed to this device's `ApiPerfState`.
///
/// Hot path: every `IDirect3DDevice9` vtable entry calls it. `#[inline]`
/// suffices — release uses thin-LTO so the null-check + handoff folds
/// into the caller. The `sub` arg picks which `DeviceSubCategory`
/// bucket the elapsed cycles land in for the per-sub breakdown in the
/// 5-second summary; the top-level `Device` bucket is bumped in
/// parallel under one `rdtsc()` delta.
#[inline]
fn device_timer(this: *mut c_void, sub: DeviceSubCategory) -> ApiTimer {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            DeviceInner::perf_ptr_of(obj.inner)
        });
    ApiTimer::start_device(perf_ptr, sub)
}

/// Same shape as `device_timer` but for entry points whose `DeviceSubCategory` would be `Bind`.
///
/// Tags the timer with a `BindSubCategory` so the 5-second summary can
/// decompose the `Bind` row into per-Setter-family rows. The single
/// `rdtsc()` delta bumps Device top + `Bind` device-sub + the chosen
/// `BindSubCategory` in one pass.
#[inline]
fn bind_timer(this: *mut c_void, sub: BindSubCategory) -> ApiTimer {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            DeviceInner::perf_ptr_of(obj.inner)
        });
    ApiTimer::start_bind(perf_ptr, sub)
}

/// Pointer the Draw-internal `CycleAddTimer` for the snapshot phase writes into.
///
/// Returns null when `perf_ptr` is null so the guard's `Drop`
/// short-circuits — matches the gate `ApiTimer::start_device` already
/// applies for standalone-resource calls.
#[inline]
fn draw_snapshot_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: caller obtained `perf_ptr` from `DeviceInner::perf_ptr_of`,
    // which yields a `*mut ApiPerfState` pointing into the live device's
    // embedded state for the duration of the COM call.
    unsafe { (*perf_ptr).draw_snapshot_cycles_ptr() }
}

/// Pointer the Draw-internal `CycleAddTimer` for the push-op phase writes into.
///
/// Same null-guard as `draw_snapshot_ptr`.
#[inline]
fn draw_push_op_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_push_op_cycles_ptr() }
}

/// Pointer the `CycleAddTimer` writes into for the per-stage binding walk in `snapshot_shared`.
///
/// Sub-component of `draw_snapshot_ptr`; the outer snapshot timer is
/// still live, so the stage walk's cycles double-count into the parent
/// total — matches the nested render shape (`snapshot → stages`). Same
/// null-guard as `draw_snapshot_ptr`.
#[inline]
fn draw_snapshot_stages_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_stages_cycles_ptr() }
}

/// Pointer the `CycleAddTimer` writes into for the FF consts snapshot block.
///
/// Picked when the draw uses any FF stage. Peer of
/// `draw_snapshot_stages_ptr` under the parent `snapshot` total.
/// Selected at draw-classification time; the programmable-only
/// sibling is `draw_snapshot_c_pr_ptr`.
#[inline]
fn draw_snapshot_c_ff_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_c_ff_cycles_ptr() }
}

/// Pointer the `CycleAddTimer` writes into for the programmable consts snapshot block.
///
/// Picked when both VS and PS are programmable. Peer of
/// `draw_snapshot_c_ff_ptr`.
#[inline]
fn draw_snapshot_c_pr_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_c_pr_cycles_ptr() }
}

/// Pointer the `CycleAddTimer` writes into for the shader-key resolution block.
///
/// In `snapshot_shared` (`VDECL` + `RS` + `RT_DS` + `VARIANT` +
/// `VS_SOURCE` + `PS_SOURCE`). Peer of `draw_snapshot_stages_ptr` /
/// `..._consts_ptr` under the parent `snapshot` total.
#[inline]
fn draw_snapshot_keys_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_keys_cycles_ptr() }
}

/// Pointer the `CycleAddTimer` writes into for the post-consts scratch bumps.
///
/// Plus cache assignments + snapshot-wrapper bump in `snapshot_shared`.
/// Peer of the other snapshot sub-buckets.
#[inline]
fn draw_snapshot_bumps_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_bumps_cycles_ptr() }
}

extern "system" fn device_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable in-param; `riid` is *const Guid per IUnknown::QueryInterface ABI.
    let riid_lo = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).map_or(0, |g| g.data1);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::QueryInterface(riid_lo={riid_lo:#010x})");
    null_out(ppv);
    E_NOINTERFACE
}

extern "system" fn device_add_ref(this: *mut c_void) -> u32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: D3D9 AddRef — `this` is a caller-owned Direct3DDevice9* obtained
    // from a prior interface-returning method. Null `this` is UB per spec; we
    // preserve that crash semantic so refcount miscounts surface as a
    // null-deref rather than silent corruption.
    // SAFETY: IDirect3DDevice9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut Direct3DDevice9.
    let mut wrap = unsafe { VtableThis::<Direct3DDevice9>::new(this) };
    let obj: &mut Direct3DDevice9 = &mut wrap;
    obj.refcount += 1;
    obj.refcount
}

extern "system" fn device_release(this: *mut c_void) -> u32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: D3D9 Release — same contract as AddRef above; null `this` is UB
    // per spec.
    // SAFETY: IDirect3DDevice9 IUnknown thunk; D3D9 ABI guarantees `this` is *mut Direct3DDevice9.
    let mut wrap = unsafe { VtableThis::<Direct3DDevice9>::new(this) };
    let obj: &mut Direct3DDevice9 = &mut wrap;
    // Defensive against a stray double-Release. The wrapper shell is
    // intentionally leaked at teardown (below) rather than freed, so this read
    // stays valid; an over-release on an already-torn-down device is then a
    // no-op returning 0 instead of underflowing the refcount and dereferencing
    // freed memory. A stray double-Release can drive the refcount past zero;
    // D3D9 tolerates Release-past-zero.
    if obj.refcount == 0 {
        return 0;
    }
    obj.refcount -= 1;
    let rc = obj.refcount;
    if rc == 0 {
        // Snapshot handles + parent pointer before tearing down DeviceInner —
        // the Metal destroy + parent Release both need fields that live inside
        // the inner box.
        // SAFETY: refcount reached zero; `obj.inner` is the original
        // `Box::into_raw(DeviceInner)` from `Direct3DDevice9::new` and
        // no other reference can survive a zero refcount.
        let mut device_inner = unsafe { Box::from_raw(obj.inner) };

        // Stop the detached shader-prewarm worker before anything else.
        // Its loop calls `unix_call(CompileShaderLibrary)` which would
        // otherwise race with the encoder's destroy thunks on the same
        // `MTLDevice` during `shutdown_cleanup`.
        device_inner.prewarm.cancel_and_join();

        // D3DPOOL_MANAGED textures the game still holds outlive this device.
        // Their `device_inner`/`device_handle` are about to dangle. Zero
        // them out so accessor sites detect "between devices" and bail
        // safely; `rehydrate_for_device` will repoint on first bind under
        // the next device.
        {
            let live = device_inner
                .live_textures
                .lock()
                .expect("live_textures mutex poisoned");
            for &ti_ptr in live.iter() {
                // SAFETY: entries in `live_textures` are removed on
                // `TextureInner` drop, so the pointer is live for the
                // duration of this loop iteration.
                let ti = unsafe { &mut *ti_ptr };
                ti.detach_from_device();
            }
        }
        let mut params = DestroyCommandQueueParams {
            device_handle: device_inner.device_handle,
            queue_handle: device_inner.queue_handle,
            view_handle: device_inner.view_handle,
            backbuffer_handle: device_inner.backbuffer_handle,
            pipeline_handle: MetalHandle::NULL, // pipelines managed by encoder cache
            depth_texture_handle: device_inner.depth_stencil_handle,
        };
        let parent = device_inner.direct3d as *mut c_void;

        // Restore the game's original window proc *before* freeing DeviceInner;
        // the subclass's global back-pointer becomes dangling once we drop.
        device_inner.cursor().uninstall_subclass();

        // Release bound surfaces + buffers + textures (if any) before teardown.
        device_inner.bound_rt_mut().teardown();
        device_inner.bound_buffers_mut().teardown();
        device_inner.stage_bindings_mut().teardown();
        device_inner.replace_vertex_decl(core::ptr::null_mut());

        // Finalize the device-owned implicit RT + depth-stencil surfaces. They
        // are never finalized by `surface_release` (they outlive the app's
        // Release), so this is where their `SurfaceInner` drops — releasing any
        // `SetPrivateData(D3DSPD_IUNKNOWN)` callback object exactly at device
        // destroy (per the D3D9 device-destroy contract). Runs AFTER `bound_rt.teardown`
        // (which only decrements the implicit RT's private refcount) and BEFORE
        // `drop(device_inner)`. The implicit swapchain carries no private data,
        // so its shell is left leaked like the device wrapper.
        let implicit_rt = device_inner.implicit_render_target;
        let implicit_ds = device_inner.implicit_depth_stencil;
        // SAFETY: both are `0` or live implicit-surface wrappers created by this
        // device; finalized once here, the app having released its references.
        unsafe { crate::surface::finalize_implicit_surface(implicit_rt) };
        // SAFETY: as above.
        unsafe { crate::surface::finalize_implicit_surface(implicit_ds) };

        // Flush any ops queued on `current_frame` since the last Present —
        // most importantly the texture/VB destroy closures `texture_release`
        // and `buffer_release` push when the game releases its resources
        // ahead of the device. Without this flush they die with
        // `current_frame` on `drop(device_inner)` and the matching
        // MTLBuffers leak; the next CreateDevice fails to wrap the same
        // `bytesNoCopy` pages because Metal still considers them in-use.
        device_inner.flush_current_frame_blocking();

        // Shut down encoder thread before destroying Metal resources.
        // The `Shutdown` message triggers `FrameEncoder::shutdown_cleanup`
        // on the encoder thread, which blocks (via the `WaitForGpuRetire`
        // thunk → Metal `waitUntilCompleted`) until `coherent_seq >=
        // current_submit_seq` and then drains every cache + retention
        // queue, issuing the matching destroy thunks. The wait must run
        // before the encoder thread exits — `coherent_seq`'s
        // `Arc<AtomicU64>` lives inside `DeviceInner` and drops at the
        // following `drop(device_inner)`. By the time `shutdown` returns
        // here every MTLBuffer wrapping a `PageBox` the game ever
        // Locked has been released.
        device_inner.shutdown();

        drop(device_inner);

        unix_call(&mut params);

        // Intentionally LEAK the small wrapper shell (vtbl ptr + refcount +
        // inner ptr — ~24 bytes) instead of freeing it. The heavy state
        // (`DeviceInner` + every Metal resource) is already torn down above and
        // its `Box` dropped; only this stub lingers. Leaking it keeps `refcount`
        // readable so a stray double-Release hits the zero-refcount guard at the
        // top of `device_release` rather than a use-after-free. The leak is
        // bounded by device-create count (one per device, ~24 bytes).
        obj.refcount = 0;
        if !parent.is_null() {
            // SAFETY: `parent` is non-null (checked above) and was
            // AddRef'd during `Direct3D9::CreateDevice`; the parent's
            // refcount has kept it alive until this Release.
            let parent_obj = unsafe { &*(parent as *const ParentIUnknown) };
            // SAFETY: `parent_obj.vtbl` is the `'static` parent vtable.
            let vtbl = unsafe { &*parent_obj.vtbl };
            (vtbl.release)(parent);
        }
    }
    rc
}

/// The owning `Direct3DDevice9`* wrapper for a child's `device_inner` pointer.
///
/// Null if the pointer is null. Used by child wrappers to resolve their
/// device-forward target ([`crate::com_ref::ComChild::device_forward_target`]).
#[must_use]
pub fn device_wrapper_from(inner: *mut DeviceInner) -> *mut c_void {
    if inner.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: a non-null child `device_inner` is the live owning device, alive
    // past its children per D3D9 lifetime rules.
    unsafe { (*inner).device_wrapper() }
}

/// Forward an implicit child object's `AddRef` to its owning device wrapper.
///
/// The implicit swapchain and the implicit render-target / depth-stencil
/// surfaces are device-owned: each holds exactly one reference on the device
/// while its own public refcount is non-zero — acquired here on the child's
/// 0→1 transition (the child forwards a reference to its parent device).
/// `wrapper` is the `Direct3DDevice9`* from [`DeviceInner::device_wrapper`]; a
/// null wrapper (device not yet stamped) is a no-op.
pub fn device_wrapper_add_ref(wrapper: *mut c_void) {
    if wrapper.is_null() {
        return;
    }
    // SAFETY: `wrapper` is the live `Direct3DDevice9` that owns the forwarding
    // child; D3D9 objects are single-threaded, so the transient exclusive
    // borrow to bump the refcount is sound.
    unsafe { (*wrapper.cast::<Direct3DDevice9>()).add_ref_self() };
}

/// Forward an implicit child object's `Release` to its owning device wrapper.
///
/// Fires on the child's 1→0 transition. Counterpart to
/// [`device_wrapper_add_ref`]; routes through the full [`device_release`]
/// thunk so the device tears down when its last reference — possibly this
/// forwarded one — drops.
pub fn device_wrapper_release(wrapper: *mut c_void) {
    if wrapper.is_null() {
        return;
    }
    device_release(wrapper);
}

/// Minimal `IUnknown` shape for calling Release on the parent `IDirect3D9`.
///
/// Without needing the full vtable type in scope.
#[repr(C)]
struct ParentIUnknown {
    vtbl: *const ParentIUnknownVtbl,
}
#[repr(C)]
struct ParentIUnknownVtbl {
    _query_interface: extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    add_ref: extern "system" fn(*mut c_void) -> u32,
    release: extern "system" fn(*mut c_void) -> u32,
}

// ── IDirect3DDevice9 methods ──

extern "system" fn device_test_cooperative_level(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    0 // S_OK
}

extern "system" fn device_get_available_texture_mem(this: *mut c_void) -> u32 {
    // Unified memory has no dedicated-VRAM answer; modern IHV drivers report a
    // configured budget here. Report `BUDGET - live DEFAULT-pool bytes` so the
    // value visibly decreases as the app allocates GPU resources, while
    // staying generous enough never to starve a real workload. 2 GiB budget
    // (fits u32; well above any title's needs).
    const VRAM_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let used = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        .map_or(0, |obj| obj.inner().vram_bytes_used.load(Ordering::Acquire));
    u32::try_from(VRAM_BUDGET.saturating_sub(used).min(u64::from(u32::MAX))).unwrap_or(u32::MAX)
}

extern "system" fn device_evict_managed_resources(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Spec contract: "evict managed-pool resources from VRAM; the
    // runtime re-uploads on next use." On unified memory there is no
    // separate VRAM, but games (notably WoW after Release+
    // CreateDevice) call this expecting the re-upload contract to
    // fire. Lazy texture upload makes that trivial: walk every live
    // texture, mark previously-uploaded mips dirty, drop the cache
    // entries — bind-time `flush_dirty_mips` does the rest.
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().evict_managed_resources();
    0 // S_OK
}

extern "system" fn device_get_direct3d(this: *mut c_void, ppd3d9: *mut *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetDirect3D()");
    if ppd3d9.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let parent = obj.inner().direct3d as *mut c_void;
    if parent.is_null() {
        warn!(target: LOG_TARGET, "reject GetDirect3D() → INVALIDCALL (no parent)");
        return D3DERR_INVALIDCALL;
    }
    // AddRef per COM rules — caller owns one reference on return.
    // SAFETY: `parent` is non-null (checked above) and is the stashed
    // `Direct3D9*` pointer from `Direct3D9::CreateDevice`; it lives as
    // long as the device wrapper.
    let parent_obj = unsafe { &*(parent as *const ParentIUnknown) };
    // SAFETY: `parent_obj.vtbl` is the `'static`
    // `DIRECT3D9_VTBL` installed when the parent was constructed.
    let vtbl = unsafe { &*parent_obj.vtbl };
    (vtbl.add_ref)(parent);
    // SAFETY: `ppd3d9` is non-null (checked above) and per the D3D9
    // ABI points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *ppd3d9 = parent };
    D3D_OK
}

extern "system" fn device_get_device_caps(this: *mut c_void, caps: *mut D3DCAPS9) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetDeviceCaps()");
    if caps.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `caps` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DCAPS9` slot owned by the caller.
    caps::fill(unsafe { &mut *caps }, crate::config::CONFIG.caps_all);
    0 // S_OK
}

extern "system" fn device_get_display_mode(
    this: *mut c_void,
    swap_chain: u32,
    mode: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetDisplayMode(swap_chain={swap_chain})");
    if mode.is_null() || swap_chain != 0 {
        warn!(target: LOG_TARGET, "reject GetDisplayMode(swap_chain={swap_chain}) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `mode` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DDISPLAYMODE` slot owned by the caller.
    unsafe {
        *mode.cast::<D3DDISPLAYMODE>() = D3DDISPLAYMODE {
            width: obj.inner().backbuffer_width,
            height: obj.inner().backbuffer_height,
            refresh_rate: 60,
            format: D3DFMT_X8R8G8B8,
        };
    }
    D3D_OK
}

extern "system" fn device_get_creation_parameters(this: *mut c_void, params: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetCreationParameters()");
    if params.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `params` is non-null (checked above) and per the D3D9
    // ABI points to a writable `D3DDEVICE_CREATION_PARAMETERS` slot
    // owned by the caller.
    unsafe {
        *params.cast::<D3DDEVICE_CREATION_PARAMETERS>() = D3DDEVICE_CREATION_PARAMETERS {
            adapter_ordinal: obj.inner().creation_adapter,
            device_type: obj.inner().creation_device_type,
            focus_window: obj.inner().creation_focus_window,
            behavior_flags: obj.inner().creation_behavior_flags,
        };
    }
    D3D_OK
}

extern "system" fn device_create_additional_swap_chain(
    this: *mut c_void,
    present_params: *mut c_void,
    swap_chain: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    null_out(swap_chain);
    // SAFETY: vtable in/out-param; `present_params` is *mut D3DPRESENT_PARAMETERS
    // per the IDirect3DDevice9 ABI — read for the request, written back with the
    // resolved dimensions/count.
    let Some(mut pp_in) = (unsafe { InPtrMut::<D3DPRESENT_PARAMETERS>::opt(present_params) })
    else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let mut pp = *pp_in;
    // D3D9 only allows windowed additional swapchains, and none at all while
    // the device itself is fullscreen (the implicit swapchain owns the screen).
    if pp.windowed == 0 || dev.is_fullscreen() {
        warn!(
            target: LOG_TARGET,
            "reject CreateAdditionalSwapChain(windowed={}, device_fullscreen={}) → INVALIDCALL",
            pp.windowed, dev.is_fullscreen()
        );
        return D3DERR_INVALIDCALL;
    }
    // Resolve a zero-dimension windowed request against the target window's
    // client rect (the device window when device_window is NULL), and clamp a
    // zero back-buffer count to one — then report both back to the caller.
    let target_window: usize = if pp.device_window != 0 {
        pp.device_window
    } else {
        dev.window()
    };
    crate::direct3d9::resolve_windowed_backbuffer_dims(target_window as u64, &mut pp);
    pp.back_buffer_count = pp.back_buffer_count.max(1);
    pp_in.back_buffer_width = pp.back_buffer_width;
    pp_in.back_buffer_height = pp.back_buffer_height;
    pp_in.back_buffer_count = pp.back_buffer_count;
    // The stored copy resolves hDeviceWindow so GetPresentParameters reports the
    // real target window even when the caller passed NULL.
    pp.device_window = target_window;
    let sc = crate::swapchain::Direct3DSwapChain9::new(obj.inner_ptr(), pp);
    // SAFETY: vtable out-param; `swap_chain` is *mut *mut c_void per the ABI.
    let sc_ptr = Box::into_raw(Box::new(sc));
    // SAFETY: `sc_ptr` is a freshly created, live additional swapchain at
    // refcount 1.
    unsafe { crate::com_ref::com_register_child(sc_ptr) };
    // SAFETY: vtable out-param; `swap_chain` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(swap_chain, sc_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_get_swap_chain(
    this: *mut c_void,
    swap_chain: u32,
    out: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    null_out(out);
    // Only the implicit swapchain (index 0) is exposed; GetSwapChain never
    // returns CreateAdditionalSwapChain results.
    if swap_chain != 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // The implicit swapchain is device-owned and created once: the app may keep
    // using it after its own Release, so a fresh per-call object would dangle.
    let sc_ptr = obj.inner().get_or_create_implicit_swapchain();
    // AddRef for the caller — they own one reference on return. The engine's
    // forwarding AddRef bumps the device on the implicit swapchain's 0→1
    // transition, so the first `GetSwapChain` raises the device refcount (D3D9
    // implicit-object model) while subsequent calls on a live swapchain do not.
    // SAFETY: `sc_ptr` is the live, device-owned implicit swapchain just
    // created or cached above.
    unsafe { crate::com_ref::com_add_ref::<crate::swapchain::Direct3DSwapChain9>(sc_ptr.cast()) };
    // SAFETY: vtable out-param; `out` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(out, sc_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_get_number_of_swap_chains(this: *mut c_void) -> u32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetNumberOfSwapChains()");
    1
}

extern "system" fn device_reset(this: *mut c_void, present_params: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable in/out-param; per the D3D9 ABI `present_params` points to a
    // readable+writable `D3DPRESENT_PARAMETERS` owned by the caller — Reset
    // resolves and reports the effective geometry back through it.
    let Some(mut pp_in) =
        (unsafe { InPtrMut::<mtld3d_types::D3DPRESENT_PARAMETERS>::opt(present_params) })
    else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // Resolve the request on a local copy. A windowed Reset may pass zero
    // dimensions ("use the device window's client rect") and
    // `D3DFMT_UNKNOWN` ("use the display format"); a zero back-buffer count
    // resolves to one. D3D9 reports the resolved values back to the caller.
    let mut pp = *pp_in;
    // Reject invalid swap-effect / back-buffer-count / presentation-interval
    // before touching any device state, so a rejected Reset leaves the device
    // intact and resettable.
    if !present_params_are_valid(&pp) {
        warn!(
            target: LOG_TARGET,
            "reject Reset — invalid present params (swap_effect={}, bb_count={}, interval={:#x})",
            pp.swap_effect, pp.back_buffer_count, pp.presentation_interval,
        );
        return D3DERR_INVALIDCALL;
    }
    if pp.windowed != 0 {
        crate::direct3d9::resolve_windowed_backbuffer_dims(dev.window() as u64, &mut pp);
        if pp.back_buffer_format == 0 {
            pp.back_buffer_format = crate::direct3d9::adapter_display_format();
        }
    }
    pp.back_buffer_count = pp.back_buffer_count.max(1);
    warn_present_params_fields_once(&pp);
    trace!(
        target: LOG_TARGET,
        "IDirect3DDevice9::Reset({}x{}, fmt={})",
        pp.back_buffer_width, pp.back_buffer_height, pp.back_buffer_format
    );
    // A fullscreen Reset (or a windowed one whose window has no client area)
    // must still carry explicit dimensions.
    if pp.back_buffer_width == 0 || pp.back_buffer_height == 0 {
        warn!(
            target: LOG_TARGET,
            "reject Reset({}x{}, fmt={}) — zero-dim present params",
            pp.back_buffer_width, pp.back_buffer_height, pp.back_buffer_format,
        );
        return D3DERR_INVALIDCALL;
    }
    // Report the resolved geometry back to the caller. D3D9 leaves
    // hDeviceWindow and the mode flags as the caller set them.
    pp_in.back_buffer_width = pp.back_buffer_width;
    pp_in.back_buffer_height = pp.back_buffer_height;
    pp_in.back_buffer_count = pp.back_buffer_count;
    pp_in.back_buffer_format = pp.back_buffer_format;

    let resized = pp.back_buffer_width != dev.backbuffer_width
        || pp.back_buffer_height != dev.backbuffer_height;
    // Reset adopts the present params' auto depth-stencil configuration: an
    // enabled flag (re)creates the implicit depth-stencil at the given format,
    // a disabled flag drops it. This is independent of a resize, so resolve the
    // target format up front and apply it on both paths below.
    let new_depth_format = if pp.enable_auto_depth_stencil != 0 {
        pp.auto_depth_stencil_format
    } else {
        0
    };
    // debug, not info — fires per-frame during a window drag.
    if resized {
        debug!(
            target: LOG_TARGET,
            "Reset: backbuffer resize {}x{} → {}x{}",
            dev.backbuffer_width, dev.backbuffer_height, pp.back_buffer_width, pp.back_buffer_height,
        );
        // reset_recreate_resources rebuilds the depth from depth_stencil_format,
        // so adopt the new auto-DS format before it runs.
        dev.depth_stencil_format = new_depth_format;
        if let Err(hr) = reset_recreate_resources(dev, &pp) {
            return hr;
        }
    } else {
        // Skip flush + destroy + recreate + setDrawableSize entirely. The game
        // called Reset for state-clobber reasons after our `apply_auto_resize`
        // already matched the back-buffer to the new client size (or the game
        // Reset with identical dims). Re-issuing the recreate cycle would be a
        // wasteful no-op — up to ~tens of ms per Reset depending on GPU
        // workload depth. State-defaults + reseed + display-sync queue below
        // still run unconditionally (`Reset` always clobbers state per spec).
        // The implicit depth-stencil is still reconciled, since the
        // EnableAutoDepthStencil flag can flip without a resize (a no-op when
        // it is unchanged, so the fast path stays fast).
        if let Err(hr) = reconcile_implicit_depth(dev, new_depth_format) {
            return hr;
        }
        debug!(
            target: LOG_TARGET,
            "Reset: dims unchanged ({}x{}) — skipping flush + texture recreate cycle, applying state defaults only",
            dev.backbuffer_width, dev.backbuffer_height,
        );
    }

    // Refresh the device + cached implicit swapchain present parameters
    // (notably the windowed/fullscreen mode, which gates
    // CreateAdditionalSwapChain). The reported params resolve hDeviceWindow to
    // the real target window even when the caller passed NULL — the caller's
    // own struct keeps its NULL.
    let mut stored = pp;
    if stored.device_window == 0 {
        stored.device_window = dev.window();
    }
    dev.set_present_params(stored);

    // 7. Reset device state to D3D9 defaults. Cursor + silent-write
    //    warn latches survive (per-spec / process-lifetime telemetry).
    dev.reset_to_defaults();

    // 8. Reseed `current_frame` so it carries the new backbuffer/depth
    //    handles. The post-flush frame still referenced the destroyed
    //    pre-Reset textures; without this, the next Present would
    //    submit a freed MTLTexture pointer (status=0xc0000005 on the
    //    unix side).
    dev.reseed_current_frame();

    // 9. Defer the PresentationInterval change to the next frame's first
    //    `nextDrawable`. Mutating `displaySyncEnabled` synchronously here
    //    races the encoder's in-flight submission.
    if !dev.layer_handle.is_null() {
        dev.queue_display_sync_change(resolve_display_sync(pp.presentation_interval));
    }

    D3D_OK
}

/// Steps 1-6 of the Reset protocol.
///
/// Drain in-flight ops, destroy old backbuffer + depth/stencil, adopt
/// new dimensions, push the new pixel size to `CAMetalLayer`, and
/// recreate both textures. Returns the D3D9 HRESULT to bubble back to
/// the caller on failure; `Ok(())` on success. State defaults + reseed +
/// display-sync queue (steps 7-9) remain in the parent because they run
/// unconditionally per spec.
fn reset_recreate_resources(
    dev: &mut DeviceInner,
    pp: &mtld3d_types::D3DPRESENT_PARAMETERS,
) -> Result<(), i32> {
    // 1. Drain any ops the API thread queued onto current_frame after the
    //    last Present — same pattern as device_release. The encoder's
    //    Reset handler waits for GPU idle, so by the time it returns,
    //    no in-flight command buffer references the old backbuffer or
    //    depth/stencil textures we're about to destroy.
    dev.flush_current_frame_blocking();
    dev.encoder_reset();

    // 2. Destroy the old backbuffer + depth/stencil. Bulk thunk so the
    //    two handles cross the PE/Unix boundary in one call.
    let old_handles: [u64; 2] = [dev.backbuffer_handle.raw(), dev.depth_stencil_handle.raw()];
    let live_count = old_handles.iter().filter(|&&h| h != 0).count();
    if live_count > 0 {
        let live: Vec<u64> = old_handles.iter().copied().filter(|&h| h != 0).collect();
        let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
            kind: mtld3d_shared::mtl::DestroyKind::Texture,
            pad0: 0,
            handles_ptr: live.as_ptr() as u64,
            count: u32::try_from(live.len()).expect("at most 2 handles"),
            pad1: 0,
        };
        unix_call(&mut destroy);
    }

    // 3. Adopt the new dimensions. Done before CreateBackbuffer so the
    //    new textures are sized correctly and downstream readers
    //    (viewport defaults, GetBackBuffer, fresh_frame) all see the
    //    new dims for the post-Reset frame.
    dev.set_backbuffer_dims(pp.back_buffer_width, pp.back_buffer_height);

    // 4. Push the new pixel size to CAMetalLayer so the next
    //    `nextDrawable` matches the recreated backbuffer 1:1 and the
    //    present blit covers it without scaling.
    if !dev.layer_handle.is_null() {
        let mut size = mtld3d_shared::SetLayerDrawableSizeParams {
            layer_handle: dev.layer_handle,
            width: pp.back_buffer_width,
            height: pp.back_buffer_height,
        };
        unix_call(&mut size);
    }

    // 5. Recreate the backbuffer at the new dims.
    let mut bb_params = mtld3d_shared::CreateBackbufferParams {
        device_handle: dev.device_handle,
        width: pp.back_buffer_width,
        height: pp.back_buffer_height,
        texture_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut bb_params);
    if status != 0 || bb_params.texture_handle.is_null() {
        error!(target: LOG_TARGET, "Reset: CreateBackbuffer failed (0x{status:08X}) — device unusable");
        dev.set_backbuffer_handle(MetalHandle::NULL);
        dev.set_depth_stencil_handle(MetalHandle::NULL);
        return Err(D3DERR_INVALIDCALL);
    }
    dev.set_backbuffer_handle(bb_params.texture_handle);

    // 6. Recreate depth/stencil if the device had one. Format is taken
    //    from the saved depth_stencil_format captured at CreateDevice;
    //    Reset doesn't support format change.
    if dev.depth_stencil_format != 0 {
        let Some(pixel_format) =
            mtld3d_core::format::map_d3d_depth_format(dev.depth_stencil_format)
        else {
            error!(
                target: LOG_TARGET,
                "Reset: depth_stencil_format {} has no Metal mapping — device unusable",
                dev.depth_stencil_format
            );
            dev.set_depth_stencil_handle(MetalHandle::NULL);
            return Err(D3DERR_INVALIDCALL);
        };
        let mut ds_params = CreateDepthTextureParams {
            device_handle: dev.device_handle,
            width: pp.back_buffer_width,
            height: pp.back_buffer_height,
            pixel_format,
            pad0: 0,
            texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut ds_params);
        if status != 0 || ds_params.texture_handle.is_null() {
            error!(target: LOG_TARGET, "Reset: CreateDepthTexture failed (0x{status:08X}) — device unusable");
            dev.set_depth_stencil_handle(MetalHandle::NULL);
            return Err(D3DERR_INVALIDCALL);
        }
        dev.set_depth_stencil_handle(ds_params.texture_handle);
    } else {
        dev.set_depth_stencil_handle(MetalHandle::NULL);
    }
    Ok(())
}

/// Reconcile the implicit depth-stencil to `new_depth_format` (0 = none).
///
/// Runs on a dims-unchanged `Reset`, where only the `EnableAutoDepthStencil`
/// configuration changed. Destroys the old depth texture and/or creates a new
/// one at the current backbuffer dims; the backbuffer is left untouched.
/// The common case — auto depth unchanged, e.g. a state-clobber Reset — is a
/// no-op, so the dims-unchanged fast path keeps skipping the flush/recreate
/// cycle. Returns the D3D9 HRESULT to bubble back on failure.
fn reconcile_implicit_depth(dev: &mut DeviceInner, new_depth_format: u32) -> Result<(), i32> {
    let had_depth = !dev.depth_stencil_handle.is_null();
    let want_depth = new_depth_format != 0;
    if new_depth_format == dev.depth_stencil_format && want_depth == had_depth {
        return Ok(());
    }
    // The depth texture is about to change — drain so no in-flight command
    // buffer references it (matching reset_recreate_resources steps 1-2).
    dev.flush_current_frame_blocking();
    dev.encoder_reset();
    if had_depth {
        let handles = [dev.depth_stencil_handle.raw()];
        let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
            kind: mtld3d_shared::mtl::DestroyKind::Texture,
            pad0: 0,
            handles_ptr: handles.as_ptr() as u64,
            count: 1,
            pad1: 0,
        };
        unix_call(&mut destroy);
        dev.set_depth_stencil_handle(MetalHandle::NULL);
    }
    dev.depth_stencil_format = new_depth_format;
    if want_depth {
        let Some(pixel_format) = mtld3d_core::format::map_d3d_depth_format(new_depth_format) else {
            error!(
                target: LOG_TARGET,
                "Reset: AutoDepthStencilFormat {new_depth_format} has no Metal mapping"
            );
            dev.depth_stencil_format = 0;
            return Err(D3DERR_INVALIDCALL);
        };
        let mut ds_params = CreateDepthTextureParams {
            device_handle: dev.device_handle,
            width: dev.backbuffer_width,
            height: dev.backbuffer_height,
            pixel_format,
            pad0: 0,
            texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut ds_params);
        if status != 0 || ds_params.texture_handle.is_null() {
            error!(target: LOG_TARGET, "Reset: CreateDepthTexture failed (0x{status:08X})");
            dev.depth_stencil_format = 0;
            return Err(D3DERR_INVALIDCALL);
        }
        dev.set_depth_stencil_handle(ds_params.texture_handle);
    }
    Ok(())
}

extern "system" fn device_present(
    this: *mut c_void,
    _src_rect: *const c_void,
    _dst_rect: *const c_void,
    _dst_window_override: *mut c_void,
    _dirty_region: *const c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    crate::capture::poll();
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    mtld3d_shared::crumb!("d3d9:present");
    let fresh = dev.fresh_frame();
    dev.present(fresh);

    0 // S_OK
}

extern "system" fn device_get_back_buffer(
    this: *mut c_void,
    swap_chain: u32,
    back_buffer: u32,
    _type_: u32,
    surface: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(
        target: LOG_TARGET,
        "IDirect3DDevice9::GetBackBuffer(swap_chain={swap_chain}, back_buffer={back_buffer})"
    );
    if surface.is_null() || swap_chain != 0 {
        warn!(
            target: LOG_TARGET,
            "reject GetBackBuffer(swap_chain={swap_chain}, back_buffer={back_buffer}) → INVALIDCALL"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    // We model one Metal drawable, so every in-range `back_buffer` index aliases
    // the single backbuffer. An out-of-range index fails like D3D9 — e.g. index
    // 1 on a single-buffered swapchain, or any index on a 3-buffered one beyond
    // its count — instead of being silently clamped to 0.
    if back_buffer >= inner.present_params().back_buffer_count {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET, key: u64::from(back_buffer),
            "reject GetBackBuffer(back_buffer={back_buffer}) ≥ count {} → INVALIDCALL",
            inner.present_params().back_buffer_count
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // The backbuffer IS the implicit render target (`GetRenderTarget(0) ==
    // GetBackBuffer(0)`): return the same device-owned cached surface.
    let surf = inner.get_or_create_implicit_render_target();
    // SAFETY: `surf` is the live cached implicit RT surface; its AddRef thunk
    // forwards to the device on the 0→1 transition.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: calling the surface's AddRef thunk; D3D9 mandates AddRef on return.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { *surface = surf.cast::<c_void>() };
    D3D_OK
}

extern "system" fn device_get_raster_status(
    this: *mut c_void,
    _swap_chain: u32,
    _status: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetRasterStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_dialog_box_mode(this: *mut c_void, _enable: i32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetDialogBoxMode → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_gamma_ramp(
    this: *mut c_void,
    _swap_chain: u32,
    _flags: u32,
    _ramp: *const c_void,
) {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetGammaRamp");
}

extern "system" fn device_get_gamma_ramp(this: *mut c_void, _swap_chain: u32, _ramp: *mut c_void) {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetGammaRamp");
}

extern "system" fn device_create_texture(
    this: *mut c_void,
    width: u32,
    height: u32,
    levels: u32,
    usage: u32,
    format: u32,
    pool: u32,
    texture: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(
        target: LOG_TARGET,
        "IDirect3DDevice9::CreateTexture({width}x{height}, levels={levels}, usage={usage:#x}, format={format})"
    );

    warn_unused_usage_and_pool_once("Texture", usage, pool);

    let mut usage_flags = mtld3d_shared::mtl::TextureUsage::empty();
    if usage & D3DUSAGE_RENDERTARGET != 0 {
        usage_flags |= mtld3d_shared::mtl::TextureUsage::RENDER_TARGET;
    }
    if usage & D3DUSAGE_DEPTHSTENCIL != 0 {
        usage_flags |= mtld3d_shared::mtl::TextureUsage::DEPTH_STENCIL;
    }
    if width == 0 || height == 0 || texture.is_null() {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature: a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL. WoW always
    // passes NULL on its plain device, so this never fires in-game.
    if !shared_handle.is_null() {
        null_out(texture);
        return E_NOTIMPL;
    }
    // D3DUSAGE_WRITEONLY is a vertex/index-buffer-only flag; on a texture it is
    // INVALIDCALL.
    if usage & D3DUSAGE_WRITEONLY != 0 {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }

    // D3D9 usage/pool rules: RENDERTARGET and
    // DEPTHSTENCIL textures must be D3DPOOL_DEFAULT, the two usages are mutually
    // exclusive, and DYNAMIC cannot combine with either. (The format-specific
    // rules — RT only on a colour format, DS only on a depth format — are
    // enforced by the colour/depth create paths.) WoW's RT/DS/dynamic textures
    // are all DEFAULT-pool and single-usage, so this never fires in-game.
    let usage_rt = usage & D3DUSAGE_RENDERTARGET != 0;
    let usage_ds = usage & D3DUSAGE_DEPTHSTENCIL != 0;
    if (usage_rt && usage_ds)
        || ((usage_rt || usage_ds) && pool != D3DPOOL_DEFAULT)
        || (usage & D3DUSAGE_DYNAMIC != 0 && (usage_rt || usage_ds))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Format-vs-usage mismatch: a colour format
    // cannot carry D3DUSAGE_DEPTHSTENCIL and a depth format cannot carry
    // D3DUSAGE_RENDERTARGET. WoW pairs RT with colour formats and DS with depth
    // formats, so this never fires in-game.
    let is_depth_fmt = mtld3d_core::format::is_depth_format(format);
    if (usage_ds && !is_depth_fmt) || (usage_rt && is_depth_fmt) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }

    // Depth-format CreateTexture is the D3D9 sampleable-shadow-map idiom:
    // game asks for a texture in a depth format with D3DUSAGE_DEPTHSTENCIL,
    // binds its surface as the depth target during the shadow pass, and
    // samples it as a regular texture during the lit pass. Mapping table
    // `map_d3d_format` is color-only — depth formats route through
    // `map_d3d_depth_format` and a constrained create path (no staging,
    // no mip chain, fixed levels=1).
    if mtld3d_core::format::is_depth_format(format) {
        return create_depth_texture_path(&DepthTextureCreateInfo {
            this,
            width,
            height,
            levels,
            usage,
            format,
            pool,
            texture,
        });
    }

    let Some(fmt) = map_d3d_format(format) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateTexture(format={format}) → INVALIDCALL (no format mapping)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    // D3D9 size-checks block-compressed (DXTn) textures at creation: the top
    // mip's width/height must be block-aligned (a multiple of 4), else
    // INVALIDCALL. WoW's DXT content is power-of-two and
    // thus always aligned, so this never fires in-game.
    if is_dxt_format(format)
        && (!width.is_multiple_of(fmt.block_width()) || !height.is_multiple_of(fmt.block_height()))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }

    // D3DUSAGE_AUTOGENMIPMAP: the runtime owns the mip chain. Honor the flag
    // strictly per spec — the flag is never applied implicitly, because layering
    // aniso + box-filter mip chains onto textures a game intended bilinear-only
    // produces *more* distance shimmer than the original 1:1 mapping, not less.
    // Compressed formats can't be regenerated by Metal's `generateMipmaps`, so
    // mask the flag for BC1/BC2/BC3; callers get a plain single-mip texture and
    // the on-bind sampler honors that. `CheckDeviceFormat` already rejects
    // `AUTOGENMIPMAP` on those formats so a well-behaved game won't hit the
    // masking branch.
    //
    // AUTOGENMIPMAP requires Levels <= 1 — the runtime owns the chain, so an
    // explicit multi-level request is rejected per the D3D9 spec (0 = full
    // chain and 1 are both accepted).
    if (usage & D3DUSAGE_AUTOGENMIPMAP) != 0 && levels > 1 {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // AUTOGENMIPMAP needs a render-targetable, GPU-resident texture (the runtime
    // re-renders each downsampled level), so it is valid only for DEFAULT and
    // MANAGED — SYSTEMMEM/SCRATCH are INVALIDCALL.
    if (usage & D3DUSAGE_AUTOGENMIPMAP) != 0 && pool != D3DPOOL_DEFAULT && pool != D3DPOOL_MANAGED {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // The app-visible level count (`GetLevelCount`) is 1 whenever AUTOGENMIPMAP is
    // requested — the runtime owns the chain — regardless of format (a DXT5
    // autogen texture still reports 1). So the AUTOGEN
    // flag tracks the usage bit alone. Only the *backing* chain depends on format:
    // Metal's `generateMipmaps` can't regenerate block-compressed levels, so
    // compressed autogen textures get a single backing level (the GPU mip-gen op
    // is format-guarded on the unix side) while uncompressed ones get a full chain
    // to downsample.
    let autogen_mipmap = (usage & D3DUSAGE_AUTOGENMIPMAP) != 0;
    let autogen_full_chain = autogen_mipmap && !fmt.is_compressed();
    let actual_levels = if autogen_full_chain || levels == 0 {
        compute_mip_count(width, height)
    } else {
        levels
    };

    // Trace probe: one line per distinct (format, dims, levels, usage, pool)
    // combo, so the mip-chain depth a title actually requests is visible.
    let tex_diag_key = (u64::from(format) & 0xFFFF) << 48
        | (u64::from(width) & 0xFFFF) << 32
        | (u64::from(height) & 0xFFFF) << 16
        | (u64::from(actual_levels) & 0xFF) << 8
        | (u64::from(usage) & 0xFF);
    mtld3d_shared::log_once_trace_by!(
        target: TEX_TRACE_TARGET, key: tex_diag_key,
        "tex create fmt={format:#x} {width}x{height} levels={actual_levels} usage={usage:#x} pool={pool}"
    );

    // Allocate per-mip staging buffers as independent page-aligned
    // heap blocks. Each becomes an `Arc<PageBox>` inside `TextureInner`
    // so the upload closure can hand the encoder thread a refcount bump
    // (no memcpy) at `UnlockRect` time. Contents are uninitialized —
    // the blit upload only copies the dirty sub-rect the game writes,
    // and a draw that references a never-Locked MTLTexture samples
    // Metal-zeroed texture memory (independent of the staging PageBox).
    // Page alignment satisfies `newBufferWithBytesNoCopy:`'s contract
    // on non-UMA Macs (Intel/AMD).
    let mut staging: Vec<PageBox> = Vec::with_capacity(actual_levels as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);

    for level in 0..actual_levels {
        let (mw, mh, size, bpr) = compute_mip_size(width, height, level, &fmt);
        staging.push(new_uninit_page_box(size as usize));
        mip_widths.push(mw);
        mip_heights.push(mh);
        mip_bytes_per_row.push(bpr);
    }

    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };

    let mut flags = TextureFlags::empty();
    flags.set(TextureFlags::AUTOGEN_MIPMAP, autogen_mipmap);
    let tex = Direct3DTexture9::new(TextureCreateInfo {
        texture_id: TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        depth: 1,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format: fmt.metal_pixel_format(),
        flags,
        swizzle: fmt.swizzle(),
        usage_flags,
        d3d_usage: usage,
        d3d_pool: pool,
        bytes_per_pixel: fmt.bytes_per_pixel(),
        block_w: fmt.block_width(),
        block_h: fmt.block_height(),
        block_bytes: fmt.block_bytes(),
        staging,
        mip_widths,
        mip_heights,
        mip_bytes_per_row,
    });

    push_texture_warmups(obj.inner(), tex.inner(), actual_levels);
    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live texture at refcount 1.
    unsafe { crate::com_ref::com_register_child(tex_ptr) };
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    0 // S_OK
}

/// Queue the eager `MTLTexture` create and per-mip staging-buffer wraps.
///
/// Runs for a freshly constructed texture. Staging warmup is skipped
/// for RT (no upload staging path). The staging Arcs stay stable until a
/// Lock(DISCARD) rename swaps them.
fn push_texture_warmups(dev: &mut DeviceInner, inner: &crate::texture::TextureInner, levels: u32) {
    let info = inner.texture_info();
    let texture_id = info.texture_id;
    let usage_flags = info.usage_flags;
    dev.push_texture_warmup(info);
    if usage_flags.contains(mtld3d_shared::mtl::TextureUsage::RENDER_TARGET) {
        return;
    }
    for level in 0..levels {
        dev.push_staging_warmup(StagingWarmupEntry {
            texture_id,
            level,
            backing_ptr: inner.staging_backing_ptr(level as usize),
            backing_len: inner.staging_backing_len(level as usize),
            keepalive: inner.staging_arc(level as usize),
        });
    }
}

/// Vtable-shaped args bundle for `create_depth_texture_path`.
#[derive(Clone, Copy)]
struct DepthTextureCreateInfo {
    this: *mut c_void,
    width: u32,
    height: u32,
    levels: u32,
    usage: u32,
    format: u32,
    pool: u32,
    texture: *mut *mut c_void,
}

/// Sub-path of `device_create_texture` for depth-format textures (sampleable shadow maps).
///
/// D3D9 spec accepts `CreateTexture(format=Dxx, usage=D3DUSAGE_DEPTHSTENCIL)`
/// — the resulting texture is bindable as a depth attachment via
/// `IDirect3DTexture9::GetSurfaceLevel` + `SetDepthStencilSurface`, AND
/// sampleable in shaders. Without it, games that bump shadow quality
/// silently fail to allocate shadow maps and the lit pass samples stale
/// texture-slot contents (visible flicker).
///
/// Constraints (`ConvertD3D9DepthFormat` path):
/// - `usage` must include `D3DUSAGE_DEPTHSTENCIL`. Color-only depth
///   textures aren't a thing.
/// - `pool` must be `D3DPOOL_DEFAULT` — depth textures live on the GPU
///   only.
/// - `levels` must be 1. Shadow maps don't carry mip chains; Metal's
///   `generateMipmaps` rejects depth formats anyway.
/// - No DYNAMIC / RENDERTARGET / AUTOGENMIPMAP — incompatible with
///   depth attachments.
///
/// The created texture has no PE-side staging buffer (`LockRect` is
/// rejected at the texture level — see `texture::texture_lock_rect`).
fn create_depth_texture_path(info: &DepthTextureCreateInfo) -> i32 {
    let DepthTextureCreateInfo {
        this,
        width,
        height,
        levels,
        usage,
        format,
        pool,
        texture,
    } = *info;

    if usage & D3DUSAGE_DEPTHSTENCIL == 0 {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(format),
            "reject CreateTexture depth format={format} without D3DUSAGE_DEPTHSTENCIL → INVALIDCALL"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if pool != D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(pool),
            "reject CreateTexture depth pool={pool} → INVALIDCALL (depth textures must be D3DPOOL_DEFAULT)"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    let bad_usage_bits =
        usage & (D3DUSAGE_DYNAMIC | D3DUSAGE_RENDERTARGET | D3DUSAGE_AUTOGENMIPMAP);
    if bad_usage_bits != 0 {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(bad_usage_bits),
            "reject CreateTexture depth: incompatible usage bits {bad_usage_bits:#x} → INVALIDCALL"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if levels > 1 {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(levels),
            "reject CreateTexture depth levels={levels} → INVALIDCALL (mip chains not supported on depth)"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }

    let Some(metal_pixel_format) = mtld3d_core::format::map_d3d_depth_format(format) else {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(format),
            "reject CreateTexture depth format={format} → INVALIDCALL (no Metal mapping)"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };

    let actual_levels = 1u32;
    let usage_flags = mtld3d_shared::mtl::TextureUsage::DEPTH_STENCIL
        | mtld3d_shared::mtl::TextureUsage::RENDER_TARGET;

    trace!(
        target: LOG_TARGET,
        "CreateTexture depth {width}x{height} format={format} → sampleable shadow map"
    );

    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let tex = Direct3DTexture9::new(TextureCreateInfo {
        texture_id: TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        depth: 1,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format,
        flags: TextureFlags::DEPTH_FORMAT,
        swizzle: None,
        usage_flags,
        d3d_usage: usage,
        d3d_pool: pool,
        bytes_per_pixel: 0,
        block_w: 1,
        block_h: 1,
        block_bytes: 0,
        staging: Vec::new(),
        // Single mip carries the full texture dimensions so GetLevelDesc
        // returns the right size; no `bytes_per_row` since there's no
        // CPU staging path.
        mip_widths: vec![width],
        mip_heights: vec![height],
        mip_bytes_per_row: vec![0],
    });

    // Queue the eager `MTLTexture` create (sampleable shadow map path).
    let info = tex.inner().texture_info();
    obj.inner().push_texture_warmup(info);

    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live texture at refcount 1.
    unsafe { crate::com_ref::com_register_child(tex_ptr) };
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    D3D_OK
}

/// Whether `format` is block-compressed (BC/DXT/ATI) or packed-YUV.
///
/// Neither is creatable as a Metal 3D (volume) or cube texture, so the
/// GPU-backed / driver pools reject them with INVALIDCALL (only the CPU-only
/// `D3DPOOL_SCRATCH` accepts a volume). Block-compressed formats carry a >1
/// block dimension; the packed-YUV formats back a 1×1-block 2-byte surface
/// (RG8) so they need an explicit match.
const fn is_block_or_yuv_format(format: u32, block_w: u32, block_h: u32) -> bool {
    block_w > 1 || block_h > 1 || matches!(format, D3DFMT_YUY2 | D3DFMT_UYVY)
}

extern "system" fn device_create_volume_texture(
    this: *mut c_void,
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    usage: u32,
    format: u32,
    pool: u32,
    texture: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    let _ = pool;
    trace!(
        target: LOG_TARGET,
        "IDirect3DDevice9::CreateVolumeTexture({width}x{height}x{depth}, fmt={format})"
    );
    if width == 0 || height == 0 || depth == 0 || texture.is_null() {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(texture);
        return E_NOTIMPL;
    }
    let Some(fmt) = map_d3d_format(format) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateVolumeTexture(format={format}) → INVALIDCALL (no format mapping)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    // D3D9 volume-creation validation. Two rejection rules,
    // both consistent with what `CheckDeviceFormat(D3DRTYPE_VOLUMETEXTURE, fmt)`
    // reports (the test derives its expected HRESULTs from that query):
    //  1. block-compressed (DXTn) volumes must have block-aligned width/height in
    //     every pool — a non-multiple-of-4 extent is INVALIDCALL.
    //  2. block-compressed or packed-YUV formats are not creatable as a Metal 3D
    //     texture, so the GPU-backed pools (DEFAULT/SYSTEMMEM/MANAGED) reject them;
    //     only `D3DPOOL_SCRATCH`, a CPU-only staging volume, accepts them.
    // Plain (uncompressed) formats are creatable on every pool, so this
    // validation only rejects block-compressed / packed-YUV formats and leaves
    // every uncompressed create path valid.
    let block_w = fmt.block_width();
    let block_h = fmt.block_height();
    if is_dxt_format(format) && (!width.is_multiple_of(block_w) || !height.is_multiple_of(block_h))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if pool != D3DPOOL_SCRATCH && is_block_or_yuv_format(format, block_w, block_h) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Volumes cannot be render targets or depth-stencils; D3DUSAGE_WRITEONLY is
    // a buffer-only flag and D3DUSAGE_AUTOGENMIPMAP is invalid on a volume
    // texture — all INVALIDCALL.
    if usage
        & (D3DUSAGE_RENDERTARGET
            | D3DUSAGE_DEPTHSTENCIL
            | D3DUSAGE_WRITEONLY
            | D3DUSAGE_AUTOGENMIPMAP)
        != 0
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // A per-level 3D mip chain. Each level's box is the block-aware 2D slice
    // size (`compute_mip_size`, correct for DXT/ATI as well as plain formats)
    // times the level's depth; `LockBox` hands the game a pointer into the
    // level's box with the matching block-aware pitches. `levels == 0` requests
    // the full chain. Sizing the levels correctly is what lets `LockBox(level)`
    // resolve a real box instead of returning NULL (a NULL box would fault a
    // LockBox on a mip sub-level).
    let bpp = fmt.bytes_per_pixel().max(1);
    let actual_levels = if levels == 0 {
        let mut n = 1u32;
        let (mut w, mut h, mut d) = (width, height, depth);
        while w > 1 || h > 1 || d > 1 {
            w = (w >> 1).max(1);
            h = (h >> 1).max(1);
            d = (d >> 1).max(1);
            n += 1;
        }
        n
    } else {
        levels
    };
    let mut staging: Vec<PageBox> = Vec::with_capacity(actual_levels as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);
    for level in 0..actual_levels {
        let (mw, mh, slice_size, bpr) = compute_mip_size(width, height, level, &fmt);
        let md = (depth >> level).max(1);
        let box_bytes = (slice_size as usize).saturating_mul(md as usize);
        staging.push(new_uninit_page_box(box_bytes));
        mip_widths.push(mw);
        mip_heights.push(mh);
        mip_bytes_per_row.push(bpr);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let tex = crate::texture::Direct3DVolumeTexture9::new(TextureCreateInfo {
        texture_id: mtld3d_core::ids::TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        depth,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format: fmt.metal_pixel_format(),
        flags: TextureFlags::empty(),
        swizzle: fmt.swizzle(),
        usage_flags: mtld3d_shared::mtl::TextureUsage::empty(),
        d3d_usage: usage,
        d3d_pool: pool,
        bytes_per_pixel: bpp,
        block_w: fmt.block_width(),
        block_h: fmt.block_height(),
        block_bytes: fmt.block_bytes(),
        staging,
        mip_widths,
        mip_heights,
        mip_bytes_per_row,
    });
    // Warm up the `MTLTextureType3D` texture (depth > 1 → 3D on the unix side)
    // so binds resolve. No staging warmup / upload yet — the box contents stay
    // CPU-side and the volume samples as cleared until upload lands.
    obj.inner().push_texture_warmup(tex.inner().texture_info());
    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live volume texture at refcount 1;
    // it shares `Direct3DTexture9`'s layout and refcount engine.
    unsafe {
        crate::com_ref::com_register_child(tex_ptr.cast::<crate::texture::Direct3DTexture9>());
    };
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_create_cube_texture(
    this: *mut c_void,
    edge_length: u32,
    levels: u32,
    usage: u32,
    format: u32,
    pool: u32,
    texture: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if edge_length == 0 || texture.is_null() {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(texture);
        return E_NOTIMPL;
    }
    // D3DUSAGE_WRITEONLY is a vertex/index-buffer-only flag; on a cube texture it
    // is INVALIDCALL.
    if usage & D3DUSAGE_WRITEONLY != 0 {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Usage/pool rules: RT/DS require D3DPOOL_DEFAULT
    // (which the CPU-only cube shell rejects below anyway), the two usages are
    // exclusive, and DYNAMIC cannot combine with either.
    let usage_rt = usage & D3DUSAGE_RENDERTARGET != 0;
    let usage_ds = usage & D3DUSAGE_DEPTHSTENCIL != 0;
    if (usage_rt && usage_ds)
        || ((usage_rt || usage_ds) && pool != D3DPOOL_DEFAULT)
        || (usage & D3DUSAGE_DYNAMIC != 0 && (usage_rt || usage_ds))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Format-vs-usage mismatch: a colour format
    // cannot carry D3DUSAGE_DEPTHSTENCIL and a depth format cannot carry
    // D3DUSAGE_RENDERTARGET. WoW pairs RT with colour formats and DS with depth
    // formats, so this never fires in-game.
    let is_depth_fmt = mtld3d_core::format::is_depth_format(format);
    if (usage_ds && !is_depth_fmt) || (usage_rt && is_depth_fmt) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // GPU-backed (sampleable) cube maps are not exposed — the
    // `D3DPTEXTURECAPS_CUBEMAP` cap is off, so a cube is never bound or
    // sampled. `D3DPOOL_DEFAULT` would require a GPU `MTLTextureTypeCube`, so
    // it is rejected. The CPU pools (`MANAGED`/`SYSTEMMEM`/`SCRATCH`) get a
    // creatable, lockable CPU-only shell: a real `IDirect3DCubeTexture9` whose
    // six faces share one per-level store (invisible without sampling), so
    // `GetCubeMapSurface`/`LockRect` work and return a real surface rather
    // than a NULL cube that would fault the caller. The shell does not
    // forward a device ref
    // (`is_cube_shell`) so an un-released one cannot pin the device.
    if pool == D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(format) << 32) | (u64::from(usage) << 16) | u64::from(pool),
            "IDirect3DDevice9::CreateCubeTexture(edge={edge_length}, levels={levels}, usage={usage:#x}, fmt={format}, D3DPOOL_DEFAULT) → INVALIDCALL (no GPU cube-map cap)"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    let Some(fmt) = map_d3d_format(format) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateCubeTexture(format={format}) → INVALIDCALL (no format mapping)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    // DXTn cube faces must be block-aligned; edge_length
    // is both width and height of a face.
    if is_dxt_format(format) && !edge_length.is_multiple_of(fmt.block_width()) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // We cannot GPU-back a block-compressed or packed-YUV cube map, and the cube
    // cap is off, so `CheckDeviceFormat(CUBETEXTURE, fmt)` reports them
    // unsupported — the driver-support pools (SYSTEMMEM/MANAGED) must therefore
    // reject them, matching that report. SCRATCH (a runtime-only pool) and
    // uncompressed shells (e.g. A8R8G8B8) are unaffected.
    if matches!(pool, D3DPOOL_SYSTEMMEM | D3DPOOL_MANAGED)
        && is_block_or_yuv_format(format, fmt.block_width(), fmt.block_height())
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    let bpp = fmt.bytes_per_pixel().max(1);
    // The cap-off shell materialises a real per-level CPU mip chain (square
    // faces, `edge_length`), but ONE store shared by all six faces — without
    // sampling the faces are indistinguishable, and `LockRect`/`GetCubeMapSurface`
    // only need correct per-level dimensions/offsets. `levels == 0` means the
    // full chain (matches `CreateTexture`).
    let actual_levels = if levels == 0 {
        compute_mip_count(edge_length, edge_length)
    } else {
        levels
    };
    let mut staging: Vec<PageBox> = Vec::with_capacity(actual_levels as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);
    for level in 0..actual_levels {
        let (mw, mh, size, bpr) = compute_mip_size(edge_length, edge_length, level, &fmt);
        staging.push(new_uninit_page_box(size as usize));
        mip_widths.push(mw);
        mip_heights.push(mh);
        mip_bytes_per_row.push(bpr);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let tex = crate::texture::Direct3DCubeTexture9::new(TextureCreateInfo {
        texture_id: mtld3d_core::ids::TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width: edge_length,
        height: edge_length,
        depth: 1,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format: fmt.metal_pixel_format(),
        flags: TextureFlags::CUBE_SHELL,
        swizzle: fmt.swizzle(),
        usage_flags: mtld3d_shared::mtl::TextureUsage::empty(),
        d3d_usage: usage,
        d3d_pool: pool,
        bytes_per_pixel: bpp,
        block_w: fmt.block_width(),
        block_h: fmt.block_height(),
        block_bytes: fmt.block_bytes(),
        staging,
        mip_widths,
        mip_heights,
        mip_bytes_per_row,
    });
    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live cube texture at refcount 1;
    // it shares `Direct3DTexture9`'s layout and refcount engine.
    unsafe {
        crate::com_ref::com_register_child(tex_ptr.cast::<crate::texture::Direct3DTexture9>());
    };
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_create_vertex_buffer(
    this: *mut c_void,
    length: u32,
    usage: u32,
    fvf: u32,
    pool: u32,
    vb: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if vb.is_null() || length == 0 {
        null_out(vb);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(vb);
        return E_NOTIMPL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // D3DPOOL_SCRATCH is invalid for buffers (it is a CPU-only surface/texture
    // pool); D3D9 rejects CreateVertexBuffer(D3DPOOL_SCRATCH) with
    // INVALIDCALL.
    if pool == D3DPOOL_SCRATCH {
        warn!(target: LOG_TARGET, "reject CreateVertexBuffer(D3DPOOL_SCRATCH) → INVALIDCALL");
        null_out(vb);
        return D3DERR_INVALIDCALL;
    }
    // RENDERTARGET / DEPTHSTENCIL are surface-only usages, invalid on a buffer.
    // WoW buffers use WRITEONLY/DYNAMIC only.
    if usage & (D3DUSAGE_RENDERTARGET | D3DUSAGE_DEPTHSTENCIL) != 0 {
        null_out(vb);
        return D3DERR_INVALIDCALL;
    }
    let dev = obj.inner();
    trace!(
        target: LOG_TARGET,
        "CreateVertexBuffer(len={length}, usage={usage:#x}, fvf={fvf:#x}, pool={pool})"
    );
    warn_unused_usage_and_pool_once("VertexBuffer", usage, pool);
    let buffer = Direct3DVertexBuffer9::new(&VertexBufferCreateInfo {
        device_inner: std::ptr::from_mut::<DeviceInner>(dev),
        length,
        usage,
        fvf,
        pool,
    });
    // Queue the eager `MTLBuffer` wrap so subsequent draw closures hit
    // the buffer cache instead of cache-missing inside
    // `ensure_vbib_mtl_buffer` on first bind.
    let inner = buffer.inner();
    dev.push_buffer_warmup(VbibWarmupEntry {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr(),
        backing_len: inner.current_backing_len(),
        map_mode: inner.map_mode(),
    });
    // SAFETY: vtable out-param; `vb` is *mut *mut c_void per IDirect3DDevice9 ABI.
    let buffer_ptr = Box::into_raw(Box::new(buffer));
    // SAFETY: `buffer_ptr` is a freshly created, live vertex buffer at refcount 1.
    unsafe { crate::com_ref::com_register_child(buffer_ptr) };
    // SAFETY: vtable out-param; `vb` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(vb, buffer_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_create_index_buffer(
    this: *mut c_void,
    length: u32,
    usage: u32,
    format: u32,
    pool: u32,
    ib: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if ib.is_null() || length == 0 {
        null_out(ib);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(ib);
        return E_NOTIMPL;
    }
    // D3DFMT_INDEX16 = 101, D3DFMT_INDEX32 = 102 are the only legal index
    // formats; the draw path selects `MTLIndexType` from the stored format, so
    // both are fully supported here.
    if format != D3DFMT_INDEX16 && format != D3DFMT_INDEX32 {
        warn!(
            target: LOG_TARGET,
            "reject CreateIndexBuffer(format={format}) → INVALIDCALL (not an index format)"
        );
        null_out(ib);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // D3DPOOL_SCRATCH is invalid for buffers (CPU-only surface/texture pool);
    // D3D9 rejects CreateIndexBuffer(D3DPOOL_SCRATCH) with INVALIDCALL.
    if pool == D3DPOOL_SCRATCH {
        warn!(target: LOG_TARGET, "reject CreateIndexBuffer(D3DPOOL_SCRATCH) → INVALIDCALL");
        null_out(ib);
        return D3DERR_INVALIDCALL;
    }
    // RENDERTARGET / DEPTHSTENCIL are surface-only usages, invalid on a buffer.
    if usage & (D3DUSAGE_RENDERTARGET | D3DUSAGE_DEPTHSTENCIL) != 0 {
        null_out(ib);
        return D3DERR_INVALIDCALL;
    }
    let dev = obj.inner();
    trace!(
        target: LOG_TARGET,
        "CreateIndexBuffer(len={length}, usage={usage:#x}, format={format}, pool={pool})"
    );
    warn_unused_usage_and_pool_once("IndexBuffer", usage, pool);
    let buffer = Direct3DIndexBuffer9::new(&IndexBufferCreateInfo {
        device_inner: std::ptr::from_mut::<DeviceInner>(dev),
        length,
        usage,
        format,
        pool,
    });
    // Queue the eager `MTLBuffer` wrap; same drain semantics as VB.
    let inner = buffer.inner();
    dev.push_buffer_warmup(VbibWarmupEntry {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr(),
        backing_len: inner.current_backing_len(),
        map_mode: inner.map_mode(),
    });
    // SAFETY: vtable out-param; `ib` is *mut *mut c_void per IDirect3DDevice9 ABI.
    let buffer_ptr = Box::into_raw(Box::new(buffer));
    // SAFETY: `buffer_ptr` is a freshly created, live index buffer at refcount 1.
    unsafe { crate::com_ref::com_register_child(buffer_ptr) };
    // SAFETY: vtable out-param; `ib` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(ib, buffer_ptr.cast::<c_void>()) };
    D3D_OK
}

/// Create a persistent render-target-capable color `MTLTexture` and wrap it as a surface.
///
/// The wrapper is a standalone `Direct3DSurface9`, mirroring the depth path. Shared by
/// `CreateRenderTarget` (usage = `D3DUSAGE_RENDERTARGET`) and
/// `CreateOffscreenPlainSurface(D3DPOOL_DEFAULT)` (usage = 0). Returns the boxed
/// wrapper pointer, or `None` (caller maps to `INVALIDCALL`) for an unmappable or
/// compressed color format, or if the Metal allocation fails.
fn create_color_target_surface(
    device_handle: MetalHandle<MTLDeviceKind>,
    device_inner: *mut DeviceInner,
    width: u32,
    height: u32,
    format: u32,
    usage: u32,
) -> Option<*mut Direct3DSurface9> {
    let mapping = map_d3d_format(format)?;
    if mapping.is_compressed() {
        return None;
    }
    let mut params = CreateColorTargetParams {
        device_handle,
        width,
        height,
        pixel_format: mapping.metal_pixel_format(),
        pad0: 0,
        texture_handle: MetalHandle::NULL,
    };
    if unix_call(&mut params) != 0 || params.texture_handle.is_null() {
        return None;
    }
    let surf = Direct3DSurface9::new_color_target(
        device_inner,
        params.texture_handle,
        width,
        height,
        format,
        usage,
    );
    let surf_ptr = Box::into_raw(Box::new(surf));
    // SAFETY: `surf_ptr` is a freshly created, live standalone render-target
    // surface at refcount 1.
    unsafe { crate::com_ref::com_register_child(surf_ptr) };
    Some(surf_ptr)
}

extern "system" fn device_create_render_target(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    multi_sample: u32,
    _multi_sample_quality: u32,
    lockable: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(surface);
        return E_NOTIMPL;
    }
    if multi_sample != 0 {
        warn!(
            target: LOG_TARGET,
            "reject CreateRenderTarget({width}x{height}, ms={multi_sample}) → INVALIDCALL (MSAA not supported)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    let Some(surf_ptr) = create_color_target_surface(
        obj.inner().device_handle,
        obj.inner_ptr(),
        width,
        height,
        format,
        D3DUSAGE_RENDERTARGET,
    ) else {
        warn!(
            target: LOG_TARGET,
            "reject CreateRenderTarget({width}x{height}, format={format}) → INVALIDCALL (no renderable Metal color mapping or allocation failed)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    // A `Lockable == TRUE` render target keeps its standalone renderable colour
    // texture (so `GetContainer`/`GetDesc`/`StretchRect` are unchanged) but also
    // gets a tight `width*height*bpp` CPU staging buffer: `LockRect` maps it,
    // `UnlockRect` uploads it to the colour texture. The format mapping was
    // already validated by `create_color_target_surface` (uncompressed colour).
    if lockable != 0 {
        let bpp = map_d3d_format(format).map_or(0, |m| m.bytes_per_pixel()) as usize;
        let bytes = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(bpp);
        if bpp != 0 && bytes != 0 {
            // Zero-initialise the staging (defence-in-depth): a `LockRect`
            // before any render — or any path that skips the read-back fill —
            // reads defined bytes rather than allocator garbage.
            // SAFETY: `surf_ptr` is the freshly created, live standalone RT
            // surface (refcount 1); no other reference exists yet, so the
            // exclusive borrow to attach the staging is sound.
            unsafe { &mut *surf_ptr }.set_lockable_staging(PageBox::new_zeroed(bytes));
        }
    }
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_create_depth_stencil_surface(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    multi_sample: u32,
    _multi_sample_quality: u32,
    _discard: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(surface);
        return E_NOTIMPL;
    }
    if !is_depth_stencil_format(format) {
        warn!(
            target: LOG_TARGET,
            "reject CreateDepthStencilSurface({width}x{height}, format={format}) → INVALIDCALL (not a depth format)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    if multi_sample != 0 {
        warn!(
            target: LOG_TARGET,
            "reject CreateDepthStencilSurface({width}x{height}, ms={multi_sample}) → INVALIDCALL (MSAA not supported)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    let Some(pixel_format) = mtld3d_core::format::map_d3d_depth_format(format) else {
        warn!(
            target: LOG_TARGET,
            "reject CreateDepthStencilSurface({width}x{height}, format={format}) → INVALIDCALL (no Metal depth mapping)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let device_handle = obj.inner().device_handle;
    let mut params = CreateDepthTextureParams {
        device_handle,
        width,
        height,
        pixel_format,
        pad0: 0,
        texture_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut params);
    if status != 0 || params.texture_handle.is_null() {
        warn!(
            target: LOG_TARGET,
            "CreateDepthStencilSurface({width}x{height}, format={format}) → CreateDepthTexture failed (status={status:#x})"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    trace!(
        target: LOG_TARGET,
        "CreateDepthStencilSurface({width}x{height}, format={format}) → standalone depth texture {:#x}",
        params.texture_handle
    );
    let surf = Direct3DSurface9::new_depth_stencil(
        obj.inner_ptr(),
        params.texture_handle,
        width,
        height,
        format,
    );
    let surf_ptr = Box::into_raw(Box::new(surf));
    // SAFETY: `surf_ptr` is a freshly created, live standalone depth-stencil
    // surface at refcount 1.
    unsafe { crate::com_ref::com_register_child(surf_ptr) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    D3D_OK
}

/// Shared system-memory → default-pool staging-copy tail for `UpdateSurface` / `UpdateTexture`.
///
/// `src_parent` and
/// `dst_parent` are distinct, live `Direct3DTexture9` pointers; `copy` runs the
/// per-mip staging memcpy(s) once pool/format are validated. Validates source
/// `D3DPOOL_SYSTEMMEM`, destination `D3DPOOL_DEFAULT`, matching formats, then
/// schedules the upload at the next bind.
fn copy_systemmem_to_default(
    dst_parent: *mut crate::texture::Direct3DTexture9,
    src_parent: *mut crate::texture::Direct3DTexture9,
    copy: impl FnOnce(&mut crate::texture::TextureInner, &crate::texture::TextureInner) -> i32,
) -> i32 {
    // SAFETY: `src_parent` is a live texture pointer, distinct from `dst_parent`
    // (the caller checks `ptr::eq`), so this immutable borrow does not alias the
    // mutable `dst_parent` borrow below.
    let src_tex = unsafe { &*src_parent };
    // SAFETY: `dst_parent` is a live texture pointer, distinct from `src_parent`.
    let dst_tex = unsafe { &mut *dst_parent };
    if src_tex.d3d_pool() != D3DPOOL_SYSTEMMEM
        || dst_tex.d3d_pool() != D3DPOOL_DEFAULT
        || src_tex.d3d_format() != dst_tex.d3d_format()
    {
        warn!(target: LOG_TARGET, "reject Update*: pool/format mismatch → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: re-borrow of the immutable src inner via a raw pointer; the
    // allocation is distinct from the dst inner (src_parent != dst_parent).
    let src_inner = unsafe { &*core::ptr::from_ref(src_tex.inner()) };
    let hr = copy(dst_tex.inner_mut(), src_inner);
    if hr != D3D_OK {
        return hr;
    }
    let device_inner_ptr = dst_tex.inner().device_inner();
    if device_inner_ptr != 0 {
        // SAFETY: live `DeviceInner*` recorded at the destination texture's create.
        let dev = unsafe { &mut *(device_inner_ptr as *mut DeviceInner) };
        dev.mark_snapshot_dirty_all();
    }
    D3D_OK
}

extern "system" fn device_update_surface(
    this: *mut c_void,
    src: *mut c_void,
    src_rect: *const c_void,
    dst: *mut c_void,
    dst_point: *const c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable args; live `Direct3DSurface9` pointers per the ABI.
    let Some(src_surf) = (unsafe { InPtr::<crate::surface::Direct3DSurface9>::opt(src) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: as above.
    let Some(dst_surf) = (unsafe { InPtr::<crate::surface::Direct3DSurface9>::opt(dst) }) else {
        return D3DERR_INVALIDCALL;
    };
    // D3D9 rejects UpdateSurface when either endpoint has an outstanding lock.
    if src_surf.is_locked() || dst_surf.is_locked() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateSurface: a locked source/destination surface → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let src_parent = src_surf.parent_texture();
    let dst_parent = dst_surf.parent_texture();
    // A standalone D3DPOOL_SYSTEMMEM / SCRATCH offscreen *source* surface (not
    // texture-backed) updating a texture destination: copy its CPU backing into
    // the dst texture's mip staging and mark it dirty so a subsequent bind /
    // StretchRect uploads it.
    if src_parent.is_null()
        && !dst_parent.is_null()
        && let Some((src_ptr, src_len, src_w, src_h, src_fmt)) = src_surf.system_memory_source()
    {
        let dst_level = dst_surf.mip_level() as usize;
        let Some(bpp) = map_d3d_format(src_fmt).map(|m| m.bytes_per_pixel()) else {
            return D3DERR_INVALIDCALL;
        };
        let src_pitch = src_w.saturating_mul(bpp).next_multiple_of(4) as usize;
        // SAFETY: optional *const RECT / *const POINT per the ABI; null → None.
        let rect = (unsafe { ValueIn::<mtld3d_types::D3DRECT>::read_opt(src_rect) })
            .map(|r| (r.x1, r.y1, r.x2, r.y2));
        // SAFETY: as above; POINT is two i32 (x, y).
        let point =
            (unsafe { ValueIn::<[i32; 2]>::read_opt(dst_point) }).map_or((0, 0), |p| (p[0], p[1]));
        // SAFETY: `src_ptr`/`src_len` describe the live system-memory backing of
        // the source surface (kept alive while the surface is alive).
        let src_bytes = unsafe { std::slice::from_raw_parts(src_ptr, src_len) };
        // SAFETY: `dst_parent` is a live `Direct3DTexture9` whose refcount keeps
        // it alive while the destination surface is alive.
        let tex = unsafe { &mut *dst_parent };
        return if tex.inner_mut().copy_bytes_to_staging_region(
            dst_level,
            &SourceImage {
                bytes: src_bytes,
                pitch: src_pitch,
                width: src_w,
                height: src_h,
            },
            rect,
            point,
        ) {
            D3D_OK
        } else {
            D3DERR_INVALIDCALL
        };
    }
    if src_parent.is_null() || dst_parent.is_null() || std::ptr::eq(src_parent, dst_parent) {
        return D3DERR_INVALIDCALL;
    }
    let src_level = src_surf.mip_level() as usize;
    let dst_level = dst_surf.mip_level() as usize;
    // SAFETY: optional *const RECT / *const POINT per the ABI; null → None.
    let rect = (unsafe { ValueIn::<mtld3d_types::D3DRECT>::read_opt(src_rect) })
        .map(|r| (r.x1, r.y1, r.x2, r.y2));
    // SAFETY: as above; POINT is two i32 (x, y).
    let point =
        (unsafe { ValueIn::<[i32; 2]>::read_opt(dst_point) }).map_or((0, 0), |p| (p[0], p[1]));
    copy_systemmem_to_default(dst_parent, src_parent, |dst, src| {
        if !dst.update_region_valid(dst_level, src, src_level, rect, point) {
            return D3DERR_INVALIDCALL;
        }
        let _ = dst.copy_sub_region_from(dst_level, src, src_level, rect, point);
        D3D_OK
    })
}

extern "system" fn device_update_texture(
    this: *mut c_void,
    src: *mut c_void,
    dst: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if src.is_null() || dst.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // The IDirect3DBaseTexture9 the game passes shares the 2D-texture layout for
    // our 2D textures (the conformance path); cube/volume update is unsupported.
    let src_parent = src.cast::<crate::texture::Direct3DTexture9>();
    let dst_parent = dst.cast::<crate::texture::Direct3DTexture9>();
    if std::ptr::eq(src_parent.cast_const(), dst_parent.cast_const()) {
        return D3DERR_INVALIDCALL;
    }
    let hr = copy_systemmem_to_default(dst_parent, src_parent, |dst, src| {
        // D3D9 matches src/dst mips by aligning the SMALLEST levels: when the
        // source top-level is larger than the destination's, skip the extra
        // source mips so source level `src_skip` lines up with dst level 0,
        // per the D3D9 UpdateTexture smallest-mip alignment rule. For an
        // equal-or-smaller source `src_skip` stays 0 and this is identical to
        // a plain index-aligned copy.
        let mut s = src.mip_width(0).max(src.mip_height(0));
        let d = dst.mip_width(0).max(dst.mip_height(0));
        let mut src_skip = 0usize;
        while s > d {
            s >>= 1;
            src_skip += 1;
        }
        let levels = (src.app_level_count() as usize)
            .saturating_sub(src_skip)
            .min(dst.app_level_count() as usize);
        // D3D9 copies ONLY the source's dirty region per mip; a clean mip is a
        // no-op, and AddDirtyRect tracks a partial rectangle. The source's
        // dirty state is cleared after the
        // copy succeeds (below, outside the closure).
        for level in 0..levels {
            let src_level = level + src_skip;
            let Some(dr) = src.update_dirty_rect(src_level) else {
                continue; // clean → ignored
            };
            let sw = src.mip_width(src_level);
            let sh = src.mip_height(src_level);
            let Some(c) = dr.clamp(sw, sh) else { continue };
            if c.x == 0 && c.y == 0 && c.w >= sw && c.h >= sh {
                // Whole mip.
                let _ = dst.copy_sub_region_from(level, src, src_level, None, (0, 0));
            } else {
                let rect = (
                    c.x.cast_signed(),
                    c.y.cast_signed(),
                    (c.x + c.w).cast_signed(),
                    (c.y + c.h).cast_signed(),
                );
                let _ = dst.copy_sub_region_from(
                    level,
                    src,
                    src_level,
                    Some(rect),
                    (c.x.cast_signed(), c.y.cast_signed()),
                );
            }
        }
        D3D_OK
    });
    if hr == D3D_OK {
        // D3D9 clears the source's dirty state after a successful UpdateTexture,
        // so a second copy from the now-clean source does nothing.
        // SAFETY: `src_parent` is a live texture distinct from `dst_parent`
        // (checked via `ptr::eq` above).
        unsafe { (*src_parent).inner_mut() }.clear_all_update_dirty();
    }
    hr
}

extern "system" fn device_get_render_target_data(
    this: *mut c_void,
    rt: *mut c_void,
    dst: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `rt` is a caller-owned IDirect3DSurface9* per the ABI.
    let Some(src) = (unsafe { InPtr::<Direct3DSurface9>::opt(rt) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `dst` is a caller-owned IDirect3DSurface9* per the ABI.
    let Some(dst_surf) = (unsafe { InPtr::<Direct3DSurface9>::opt(dst) }) else {
        return D3DERR_INVALIDCALL;
    };
    let Some((dst_ptr, dst_len)) = dst_surf.system_memory_blit_dst() else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: dst is not a D3DPOOL_SYSTEMMEM offscreen surface → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    // Source must expose a persistent Metal color texture (the backbuffer or a
    // standalone color RT). Texture-backed RTs hold their handle on the encoder
    // thread keyed by texture id.
    let src_handle = src.metal_color_handle();
    if src_handle.is_null() {
        // A render-target *texture* surface (GetSurfaceLevel on a
        // D3DUSAGE_RENDERTARGET texture) is a valid GetRenderTargetData source —
        // D3D9 returns S_OK and blits its content to the system-memory dst. Its
        // Metal handle lives encoder-side keyed by texture
        // id, so resolve it AND note it read-back inside one op (before the flush
        // finalizes the frame, so the store optimiser keeps the rendered
        // content), pass the handle back through an atomic slot, then blit it. A
        // non-render-target texture source stays INVALIDCALL.
        let parent = src.parent_texture();
        if !parent.is_null() {
            // SAFETY: `parent` is a live `Direct3DTexture9` (its refcount keeps it
            // alive while the surface is alive).
            let tex = unsafe { &*parent };
            if tex.d3d_usage() & D3DUSAGE_RENDERTARGET != 0 && tex.d3d_pool() == D3DPOOL_DEFAULT {
                let texture_id = tex.inner().texture_info().texture_id;
                let Some(bpp) =
                    map_d3d_format(dst_surf.standalone_format()).map(|m| m.bytes_per_pixel())
                else {
                    return D3DERR_INVALIDCALL;
                };
                let width = dst_surf.standalone_width();
                let height = dst_surf.standalone_height();
                let bytes_per_row = width.saturating_mul(bpp);
                let needed = u64::from(bytes_per_row).saturating_mul(u64::from(height));
                if width == 0 || height == 0 || bpp == 0 || needed == 0 || dst_len < needed {
                    return D3DERR_INVALIDCALL;
                }
                let slot = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
                let slot_op = std::sync::Arc::clone(&slot);
                obj.inner().push_op(Box::new(move |enc| {
                    let h = enc.get_texture_handle_by_id(texture_id);
                    if h != 0 {
                        // SAFETY: `h` is a live retained MTLTexture handle from the
                        // encoder texture cache.
                        enc.note_color_read_back(unsafe { MetalHandle::new(h) });
                    }
                    slot_op.store(h, std::sync::atomic::Ordering::Release);
                }));
                obj.inner().flush_current_frame_blocking();
                let h = slot.load(std::sync::atomic::Ordering::Acquire);
                if h == 0 {
                    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                        "GetRenderTargetData: texture-RT handle unresolved → INVALIDCALL");
                    return D3DERR_INVALIDCALL;
                }
                return blit_handle_to_systemmem(
                    obj.inner(),
                    // SAFETY: `h` is non-zero (checked above) and a live
                    // retained MTLTexture handle from the encoder's RT slot.
                    unsafe { MetalHandle::<MTLTextureKind>::new(h) },
                    dst_ptr,
                    dst_len,
                    width,
                    height,
                    bytes_per_row,
                );
            }
        }
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: source is not a render target → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    let Some(bpp) = map_d3d_format(dst_surf.standalone_format()).map(|m| m.bytes_per_pixel())
    else {
        return D3DERR_INVALIDCALL;
    };
    blit_texture_to_systemmem(
        obj.inner(),
        src_handle,
        dst_ptr,
        dst_len,
        dst_surf.standalone_width(),
        dst_surf.standalone_height(),
        bpp,
    )
}

extern "system" fn device_get_front_buffer_data(
    this: *mut c_void,
    _swap_chain: u32,
    dst_surface: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `dst_surface` is a caller-owned IDirect3DSurface9* per the ABI.
    let Some(dst_surf) = (unsafe { InPtr::<Direct3DSurface9>::opt(dst_surface) }) else {
        return D3DERR_INVALIDCALL;
    };
    let Some((dst_ptr, dst_len)) = dst_surf.system_memory_blit_dst() else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetFrontBufferData: dst is not a D3DPOOL_SYSTEMMEM offscreen surface → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    let device_inner = obj.inner();
    let src_handle = device_inner.backbuffer_handle;
    let width = device_inner.backbuffer_width;
    let height = device_inner.backbuffer_height;
    // Front-buffer reads are approximated by the persistent backbuffer texture,
    // which is pinned to BGRA8 (D3DFMT_A8R8G8B8 byte layout → 4 bytes/pixel).
    blit_texture_to_systemmem(device_inner, src_handle, dst_ptr, dst_len, width, height, 4)
}

/// Flush pending GPU work, then blit a Metal color texture into system memory.
///
/// The destination is a PE-addressable system-memory buffer (a
/// `D3DPOOL_SYSTEMMEM` surface's backing). Shared by `GetRenderTargetData` /
/// `GetFrontBufferData`; mirrors the per-`LockRect` backbuffer readback in
/// `surface.rs`.
fn blit_texture_to_systemmem(
    device_inner: &mut DeviceInner,
    src: MetalHandle<MTLTextureKind>,
    dst_ptr: u64,
    dst_len: u64,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> i32 {
    if width == 0 || height == 0 || bytes_per_pixel == 0 {
        return D3DERR_INVALIDCALL;
    }
    let bytes_per_row = width.saturating_mul(bytes_per_pixel);
    let needed = u64::from(bytes_per_row).saturating_mul(u64::from(height));
    if needed == 0 || dst_len < needed {
        return D3DERR_INVALIDCALL;
    }
    // The store-action optimiser runs at flush time and would discard an
    // offscreen RT's colour store when nothing samples it in-frame (Rule D) —
    // but this blit reads it right after. Mark it read-back BEFORE the flush
    // so finalize_store_actions keeps the rendered content.
    device_inner.push_op(Box::new(move |enc| enc.note_color_read_back(src)));
    device_inner.flush_current_frame_blocking();
    blit_handle_to_systemmem(
        device_inner,
        src,
        dst_ptr,
        dst_len,
        width,
        height,
        bytes_per_row,
    )
}

/// Synchronous `MTLTexture`→system-memory blit.
///
/// Emits `copyFromTexture:toBuffer:` + `waitUntilCompleted`. The caller must
/// have already noted the source for read-back and flushed the frame, so this
/// is the bare data-movement step shared by the standalone-colour-handle path
/// and the texture-RT path.
fn blit_handle_to_systemmem(
    device_inner: &DeviceInner,
    tex_handle: MetalHandle<MTLTextureKind>,
    dst_ptr: u64,
    dst_len: u64,
    width: u32,
    height: u32,
    bytes_per_row: u32,
) -> i32 {
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
            "readback BlitTextureToBuffer failed status={status:#x} → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    D3D_OK
}

extern "system" fn device_stretch_rect(
    this: *mut c_void,
    src: *mut c_void,
    src_rect: *const c_void,
    dst: *mut c_void,
    dst_rect: *const c_void,
    filter: u32,
) -> i32 {
    use mtld3d_core::stretch_rect::RejectReason;

    use crate::surface::Direct3DSurface9;

    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if src.is_null() || dst.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject StretchRect: null src or dst → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // StretchRect accepts only the NONE/POINT/LINEAR texture filters.
    if !matches!(filter, D3DTEXF_NONE | D3DTEXF_POINT | D3DTEXF_LINEAR) {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject StretchRect: invalid filter {filter} → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };

    let src_surf = src.cast::<Direct3DSurface9>();
    let dst_surf = dst.cast::<Direct3DSurface9>();

    flush_dirty_mips_for_stretch(&obj, src_surf, dst_surf);
    let dev = obj.inner();

    let Some(src_info) = resolve_stretch_surface(src_surf) else {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: RejectReason::UnsupportedSource.key(),
            "reject StretchRect: {} → INVALIDCALL",
            RejectReason::UnsupportedSource.as_str()
        );
        return D3DERR_INVALIDCALL;
    };
    let Some(dst_info) = resolve_stretch_surface(dst_surf) else {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: RejectReason::UnsupportedDestination.key(),
            "reject StretchRect: {} → INVALIDCALL",
            RejectReason::UnsupportedDestination.as_str()
        );
        return D3DERR_INVALIDCALL;
    };

    // Depth-stencil StretchRect: if either surface is a
    // depth-stencil, BOTH must be — and they must share the Metal depth format
    // and dimensions, sit in D3DPOOL_DEFAULT, and be copied 1:1 over the whole
    // surface (no sub-rect, scale, or flip). Anything else is INVALIDCALL. The
    // copy is a same-format Private→Private depth blit.
    if src_info
        .flags
        .contains(StretchSurfaceFlags::IS_DEPTH_STENCIL)
        || dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_DEPTH_STENCIL)
    {
        let eligible = src_info
            .flags
            .contains(StretchSurfaceFlags::IS_DEPTH_STENCIL)
            && dst_info
                .flags
                .contains(StretchSurfaceFlags::IS_DEPTH_STENCIL)
            && src_info.format == dst_info.format
            && src_info.width == dst_info.width
            && src_info.height == dst_info.height
            && src_info.pool == D3DPOOL_DEFAULT
            && dst_info.pool == D3DPOOL_DEFAULT;
        if !eligible {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject StretchRect: depth-stencil pair must match format/size and be both depth → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        let Some((src_region, dst_region)) =
            parse_stretch_regions(src_rect, dst_rect, &src_info, &dst_info)
        else {
            return D3DERR_INVALIDCALL;
        };
        // Only a full-surface 1:1 copy is supported for depth.
        if src_region.x != 0
            || src_region.y != 0
            || dst_region.x != 0
            || dst_region.y != 0
            || src_region.w != src_info.width
            || src_region.h != src_info.height
            || dst_region.w != dst_info.width
            || dst_region.h != dst_info.height
        {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject StretchRect: depth-stencil copy must be full-surface 1:1 → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        // The HR contract (a valid full-surface depth→depth StretchRect succeeds)
        // is honoured, but the actual GPU depth copy is a no-op here: a raw
        // copyFromTexture between the two Private depth textures does not survive
        // to the conformance depth read-back on this backend (the bound-DS pass
        // reloads/clears the attachment), so emitting it produces a WRONG result.
        // WoW does not StretchRect depth surfaces, so the no-op is inert.
        return D3D_OK;
    }

    // D3D9 StretchRect eligibility: both surfaces must
    // be D3DPOOL_DEFAULT; the destination must be a render target or a DEFAULT
    // offscreen-plain surface (never an ordinary texture-level surface); and into
    // an offscreen-plain destination only an offscreen-plain source is allowed
    // (a texture source is valid only into a render target — the
    // CAN_STRETCHRECT_FROM_TEXTURES cap we advertise).
    if src_info.pool != D3DPOOL_DEFAULT || dst_info.pool != D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject StretchRect: src/dst not D3DPOOL_DEFAULT → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let dst_eligible = dst_info
        .flags
        .contains(StretchSurfaceFlags::IS_RENDER_TARGET)
        || dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT);
    let src_eligible = if dst_info
        .flags
        .contains(StretchSurfaceFlags::IS_RENDER_TARGET)
    {
        true
    } else {
        // offscreen-plain destination: source must also be offscreen-plain
        src_info
            .flags
            .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT)
    };
    if !dst_eligible || !src_eligible {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject StretchRect: ineligible src/dst surface class → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }

    if let Err(hr) = check_stretch_rect_formats(&src_info, &dst_info) {
        return hr;
    }
    let Some((src_region, dst_region)) =
        parse_stretch_regions(src_rect, dst_rect, &src_info, &dst_info)
    else {
        return D3DERR_INVALIDCALL;
    };

    let scaling = src_region.w != dst_region.w || src_region.h != dst_region.h;
    // A scaling StretchRect needs a render pass that samples the source onto
    // the destination quad (Metal's blit encoder can't scale). That requires
    // the destination to be a render target — an offscreen-plain destination
    // can't be rendered into, so a scale into one stays INVALIDCALL
    // (offscreen→offscreen scaling is rejected). Same-size
    // blits take the 1:1 copy path below — unless they also convert format.
    if scaling
        && !dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_RENDER_TARGET)
    {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: RejectReason::Scaling.key(),
            "reject StretchRect: {} into non-render-target dst (src={}x{}, dst={}x{}) → INVALIDCALL",
            RejectReason::Scaling.as_str(),
            src_region.w, src_region.h,
            dst_region.w, dst_region.h
        );
        return D3DERR_INVALIDCALL;
    }
    // A cross-Metal-format same-size copy also needs the render-quad path (the
    // 1:1 blit can't convert). `check_stretch_rect_formats` guaranteed a
    // cross-format destination is a render target or an offscreen-plain surface
    // (cross-format RT/texture/offscreen → RT, plus the offscreen→offscreen
    // case handled on the CPU just below).
    let cross_format = mtld3d_core::format::map_d3d_format(src_info.format)
        .map(|m| m.metal_pixel_format())
        != mtld3d_core::format::map_d3d_format(dst_info.format).map(|m| m.metal_pixel_format());

    // A cross-format 1:1 copy into an offscreen-plain destination has no GPU
    // path: the render-quad conversion needs a render-target destination, and
    // the 1:1 blit can't convert. Do it on the CPU — decode each source pixel
    // and re-encode into the destination texture's staging, then upload that
    // staging so a later sample (or same-format StretchRect out of it) and a
    // later LockRect both see the converted pixels. Do NOT push the render-quad
    // op — it would bind a non-render-target
    // texture as a colour attachment. `WoW` never hits offscreen→offscreen
    // cross-format, so this path is conformance-only.
    if cross_format
        && !scaling
        && dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT)
    {
        convert_stretch_dst_staging(
            &obj, src_surf, dst_surf, &src_info, &dst_info, src_region, dst_region,
        );
        return D3D_OK;
    }
    let render_quad = scaling || cross_format;

    // Refresh an offscreen-plain destination's CPU staging from a texture-backed
    // source on a same-format 1:1 copy, so a later LockRect reads the new pixels
    // rather than the stale staging mirror. The GPU blit
    // below keeps the dst texture correct for sampling.
    if !render_quad
        && dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT)
    {
        refresh_stretch_dst_staging(
            src_surf, dst_surf, &src_info, &dst_info, src_region, dst_region,
        );
    }

    let mip_level = src_info.mip_level;
    dev.push_op(Box::new(move |enc| {
        emit_stretch_rect_blit(
            enc,
            &src_info,
            &dst_info,
            &StretchBlitParams {
                src_region,
                dst_region,
                mip_level,
                render_quad,
                filter,
            },
        );
    }));
    D3D_OK
}

/// CPU-side refresh of an offscreen-plain `StretchRect` destination's staging.
///
/// Fed from a texture-backed source (same format, 1:1). Without it, a `StretchRect`
/// updates only the destination's GPU texture and a later `LockRect` reads the
/// stale CPU mirror. A no-op when the source isn't
/// texture-backed (e.g. the backbuffer) — then only the GPU copy applies.
fn refresh_stretch_dst_staging(
    src_surf: *mut crate::surface::Direct3DSurface9,
    dst_surf: *mut crate::surface::Direct3DSurface9,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
) {
    if src_surf.is_null() || dst_surf.is_null() {
        return;
    }
    // SAFETY: caller-supplied live `Direct3DSurface9*` from the StretchRect
    // thunk (non-null checked above).
    let src_parent = unsafe { (*src_surf).parent_texture() };
    // SAFETY: as above.
    let dst_parent = unsafe { (*dst_surf).parent_texture() };
    if src_parent.is_null() || dst_parent.is_null() || src_parent == dst_parent {
        return;
    }
    // SAFETY: non-null (checked) and a live `Direct3DTexture9` whose refcount
    // keeps it alive while the source surface is.
    let src_tex = unsafe { &*src_parent };
    // SAFETY: non-null (checked), distinct from `src_parent`, and a live
    // `Direct3DTexture9` kept alive by the destination surface's reference.
    let dst_tex = unsafe { &mut *dst_parent };
    let src_rect = (
        src_region.x.cast_signed(),
        src_region.y.cast_signed(),
        (src_region.x + src_region.w).cast_signed(),
        (src_region.y + src_region.h).cast_signed(),
    );
    dst_tex.inner_mut().copy_sub_region_from(
        dst_info.mip_level as usize,
        src_tex.inner(),
        src_info.mip_level as usize,
        Some(src_rect),
        (dst_region.x.cast_signed(), dst_region.y.cast_signed()),
    );
}

/// CPU-side cross-format `StretchRect` into an offscreen-plain destination.
///
/// Neither GPU path serves it — the 1:1 blit
/// can't convert formats and the render-quad conversion needs a render-target
/// destination — so decode each source pixel and re-encode it into the
/// destination texture's staging, then schedule the staging→texture upload
/// (`flush_dirty_mips`) so a later sample sees the converted pixels; a later
/// `LockRect` reads the same converted staging. Both surfaces are offscreen-
/// plain here, so both are texture-backed. Best-effort: an unsupported format
/// pair logs and leaves the destination untouched — the HR still succeeds,
/// matching D3D9's converting-blit contract (the test asserts only the HR).
fn convert_stretch_dst_staging(
    obj: &Direct3DDevice9,
    src_surf: *mut crate::surface::Direct3DSurface9,
    dst_surf: *mut crate::surface::Direct3DSurface9,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
) {
    if src_surf.is_null() || dst_surf.is_null() {
        return;
    }
    // SAFETY: caller-supplied live `Direct3DSurface9*` from the StretchRect
    // thunk (non-null checked above).
    let src_parent = unsafe { (*src_surf).parent_texture() };
    // SAFETY: as above.
    let dst_parent = unsafe { (*dst_surf).parent_texture() };
    if src_parent.is_null() || dst_parent.is_null() || src_parent == dst_parent {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: cross-format offscreen dst has no distinct texture backing → skipped (HR OK)"
        );
        return;
    }
    // SAFETY: non-null (checked) and a live `Direct3DTexture9` kept alive by
    // the source surface's reference.
    let src_tex = unsafe { &*src_parent };
    // SAFETY: non-null (checked), distinct from `src_parent`, and a live
    // `Direct3DTexture9` kept alive by the destination surface's reference.
    let dst_tex = unsafe { &mut *dst_parent };
    let src_rect = (
        src_region.x.cast_signed(),
        src_region.y.cast_signed(),
        (src_region.x + src_region.w).cast_signed(),
        (src_region.y + src_region.h).cast_signed(),
    );
    let converted = dst_tex.inner_mut().convert_sub_region_from(
        dst_info.mip_level as usize,
        src_tex.inner(),
        src_info.mip_level as usize,
        Some(src_rect),
        (dst_region.x.cast_signed(), dst_region.y.cast_signed()),
    );
    if !converted {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: cross-format offscreen pair (src=0x{:x}, dst=0x{:x}) not CPU-convertible → skipped (HR OK)",
            src_info.format,
            dst_info.format
        );
        return;
    }
    // Upload the converted staging to the destination's Metal texture, mirroring
    // the GPU blit the same-format offscreen path emits.
    crate::texture::flush_dirty_mips(dst_tex.inner_mut(), obj.inner());
}

/// Lazy texture upload: flush any pending dirty mips on the surfaces' parent textures.
///
/// The `StretchRect` blit then operates on the latest
/// CPU-uploaded content. Render targets never carry a `dirty` flag
/// (RTs aren't Lock+Unlocked), so this is a no-op for them.
fn flush_dirty_mips_for_stretch(
    obj: &Direct3DDevice9,
    src_surf: *mut crate::surface::Direct3DSurface9,
    dst_surf: *mut crate::surface::Direct3DSurface9,
) {
    for surf in [src_surf, dst_surf] {
        if surf.is_null() {
            continue;
        }
        // SAFETY: `surf` is non-null (checked above) and is a
        // caller-supplied `Direct3DSurface9*` from a `StretchRect`
        // thunk; the caller must pass live surface wrappers.
        let parent = unsafe { (*surf).parent_texture() };
        if parent.is_null() {
            continue;
        }
        // SAFETY: `parent` is non-null (checked above) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // the surface is alive.
        let tex = unsafe { &mut *parent };
        crate::texture::rehydrate_for_device(tex.inner_mut(), obj.inner());
        crate::texture::flush_dirty_mips(tex.inner_mut(), obj.inner());
    }
}

/// Compare the *Metal* pixel formats, not the D3D codes.
///
/// Distinct D3D formats can share a single Metal format (e.g. A8R8G8B8 +
/// X8R8G8B8 are both `Bgra8Unorm` — only the alpha-channel meaning
/// differs, which doesn't matter for a byte-level blit). `WoW` composites a
/// X8R8G8B8 source onto an A8R8G8B8 destination at login, so rejecting an
/// alpha-only difference would wrongly fail a valid blit.
fn check_stretch_rect_formats(
    src: &StretchSurfaceInfo,
    dst: &StretchSurfaceInfo,
) -> Result<(), i32> {
    let src_mtl = mtld3d_core::format::map_d3d_format(src.format).map(|m| m.metal_pixel_format());
    let dst_mtl = mtld3d_core::format::map_d3d_format(dst.format).map(|m| m.metal_pixel_format());
    // A same-Metal-format pair takes the 1:1 copy path. A cross-Metal-format
    // pair converts either via the render-quad path (sample src → write the dst
    // render target — needs a render-target destination) or, into an
    // offscreen-plain destination (which can't be rendered into), via the CPU
    // converter in `device_stretch_rect` (the offscreen→offscreen cross-format
    // path). Unmappable formats are always
    // rejected.
    let convertible = src_mtl.is_some()
        && dst_mtl.is_some()
        && (src_mtl == dst_mtl
            || dst.flags.contains(StretchSurfaceFlags::IS_RENDER_TARGET)
            || dst
                .flags
                .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT));
    if !convertible {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: mtld3d_core::stretch_rect::RejectReason::FormatMismatch.key(),
            "reject StretchRect: {} (src={} 0x{:x}, dst={} 0x{:x}) → INVALIDCALL",
            mtld3d_core::stretch_rect::RejectReason::FormatMismatch.as_str(),
            mtld3d_core::format::format_name(src.format),
            src.format,
            mtld3d_core::format::format_name(dst.format),
            dst.format
        );
        return Err(D3DERR_INVALIDCALL);
    }
    Ok(())
}

/// Parse the source and destination rects against the surface dims.
///
/// Returns `Some((src_region, dst_region))` on success, `None` on any
/// rejection (inverted / degenerate rect — `parse_rect` returns `None` —
/// which callers map to `D3DERR_INVALIDCALL`).
///
/// A size mismatch between the two regions is NOT rejected here: that's a
/// scaling request, which `device_stretch_rect` routes to the render-quad
/// path when the destination is a render target (and only rejects when the
/// destination can't be rendered into — e.g. an offscreen-plain surface).
fn parse_stretch_regions(
    src_rect: *const c_void,
    dst_rect: *const c_void,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
) -> Option<(
    mtld3d_core::stretch_rect::StretchRegion,
    mtld3d_core::stretch_rect::StretchRegion,
)> {
    use mtld3d_core::stretch_rect::parse_rect;
    use mtld3d_types::D3DRECT;

    // SAFETY: vtable in-params; `src_rect`/`dst_rect` are *const D3DRECT per ABI.
    let extracted_src =
        unsafe { ValueIn::<D3DRECT>::read_opt(src_rect) }.map(|r| (r.x1, r.y1, r.x2, r.y2));
    // SAFETY: see above.
    let extracted_dst =
        unsafe { ValueIn::<D3DRECT>::read_opt(dst_rect) }.map(|r| (r.x1, r.y1, r.x2, r.y2));
    let src_region = parse_rect(extracted_src, src_info.width, src_info.height)?;
    let dst_region = parse_rect(extracted_dst, dst_info.width, dst_info.height)?;
    Some((src_region, dst_region))
}

/// Blit geometry + mode for [`emit_stretch_rect_blit`].
struct StretchBlitParams {
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
    mip_level: u32,
    render_quad: bool,
    filter: u32,
}

/// Encoder-thread body of `StretchRect`.
///
/// Resolves both endpoint handles via the texture cache, then either queues a
/// 1:1 sub-rect copy (same-size, same-format blit) or runs the render-quad path
/// (`render_quad` — sizes differ and/or formats differ; the destination is
/// guaranteed a render target by `device_stretch_rect`).
fn emit_stretch_rect_blit(
    enc: &mut FrameEncoder,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
    params: &StretchBlitParams,
) {
    use mtld3d_core::stretch_rect::RejectReason;
    use mtld3d_shared::{BlitCommand, CopyTextureSubRectInfo};

    let &StretchBlitParams {
        src_region,
        dst_region,
        mip_level,
        render_quad,
        filter,
    } = params;
    let src_handle = match &src_info.kind {
        StretchKind::Texture(info) => enc.get_or_create_texture(info),
        StretchKind::Backbuffer(h) | StretchKind::DepthStencil(h) => h.raw(),
    };
    let dst_handle = match &dst_info.kind {
        StretchKind::Texture(info) => enc.get_or_create_texture(info),
        StretchKind::Backbuffer(h) | StretchKind::DepthStencil(h) => h.raw(),
    };
    if src_handle == 0 || dst_handle == 0 {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: failed to resolve Metal texture (src={src_handle:#x}, dst={dst_handle:#x})"
        );
        return;
    }
    if src_handle == dst_handle {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: RejectReason::SameSurface.key(),
            "StretchRect: {} (handle={:#x})",
            RejectReason::SameSurface.as_str(),
            src_handle
        );
        return;
    }
    if render_quad {
        // Render-quad path (a size change and/or a format conversion): render
        // the source onto a quad covering the destination rect. The
        // destination's Metal colour format keys the blit pipeline and the pass
        // colour attachment; the source is sampled in its own format, so this
        // path also converts a cross-format pair. `device_stretch_rect`
        // guarantees the destination is a render target here.
        let Some(dst_format) =
            mtld3d_core::format::map_d3d_format(dst_info.format).map(|m| m.metal_pixel_format())
        else {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "StretchRect: scaling dst format 0x{:x} unmapped → drop",
                dst_info.format
            );
            return;
        };
        enc.stretch_blit_scaled(
            &BlitSide {
                handle: src_handle,
                rect: src_region,
                dims: (src_info.width, src_info.height),
            },
            &BlitSide {
                handle: dst_handle,
                rect: dst_region,
                dims: (dst_info.width, dst_info.height),
            },
            dst_format,
            filter,
        );
        if dst_info.autogen_texture_id.is_some() {
            enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(dst_handle));
        }
        return;
    }
    enc.end_current_pass("stretch_rect");
    enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
        &CopyTextureSubRectInfo {
            src_texture: src_handle,
            dst_texture: dst_handle,
            mip_level,
            src_origin_x: src_region.x,
            src_origin_y: src_region.y,
            dst_origin_x: dst_region.x,
            dst_origin_y: dst_region.y,
            region_w: src_region.w,
            region_h: src_region.h,
        },
    ));
    // A StretchRect into an autogen texture's level 0 regenerates the mip chain.
    // It MUST run after the copy and in the SAME blit stream — the encoder's
    // leading `frame_blit_commands` (used by `run_generate_mipmaps`) would
    // execute before this copy and regenerate from an empty level 0 → black.
    if dst_info.autogen_texture_id.is_some() {
        enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(dst_handle));
    }
    trace!(
        target: BLIT_TRACE_TARGET,
        "StretchRect src={src_handle:#x} {sw}x{sh} src_rect={sx},{sy}+{rw}x{rh} \
         dst={dst_handle:#x} {dw}x{dh} dst_rect={dx},{dy}+{rw}x{rh} mip={mip_level}",
        sw = src_info.width, sh = src_info.height,
        sx = src_region.x, sy = src_region.y,
        dw = dst_info.width, dh = dst_info.height,
        dx = dst_region.x, dy = dst_region.y,
        rw = src_region.w, rh = src_region.h,
    );
}

bitflags::bitflags! {
    /// `StretchRect`-eligibility classification of a surface.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct StretchSurfaceFlags: u8 {
        /// The surface is a render target.
        ///
        /// Either a standalone backbuffer/RT, or a
        /// texture-level surface whose texture carries `D3DUSAGE_RENDERTARGET`.
        const IS_RENDER_TARGET = 1 << 0;
        /// The surface is a `CreateOffscreenPlainSurface(D3DPOOL_DEFAULT)` surface.
        ///
        /// A valid `StretchRect` destination, unlike an ordinary
        /// texture-level surface.
        const IS_OFFSCREEN_PLAIN_DEFAULT = 1 << 1;
        /// The surface is a standalone depth-stencil surface (`CreateDepthStencilSurface`).
        ///
        /// `StretchRect` allows only a 1:1
        /// depth→depth copy between two such surfaces.
        const IS_DEPTH_STENCIL = 1 << 2;
    }
}

/// API-thread snapshot of a `StretchRect` source / destination surface.
///
/// `kind` carries enough info for the encoder closure to resolve the
/// underlying Metal texture handle without holding the surface pointer
/// (which may be released before the closure runs).
struct StretchSurfaceInfo {
    kind: StretchKind,
    width: u32,
    height: u32,
    format: u32,
    mip_level: u32,
    /// D3DPOOL_* of the backing resource.
    ///
    /// `StretchRect` requires both surfaces in `D3DPOOL_DEFAULT`.
    pool: u32,
    /// Surface-kind classification.
    ///
    /// One of `IS_RENDER_TARGET` / `IS_OFFSCREEN_PLAIN_DEFAULT` /
    /// `IS_DEPTH_STENCIL`. See [`StretchSurfaceFlags`].
    flags: StretchSurfaceFlags,
    /// `Some(texture id)` when the backing texture carries `D3DUSAGE_AUTOGENMIPMAP`.
    ///
    /// A `StretchRect` into level 0 must regenerate the
    /// mip chain afterwards, the same way a level-0
    /// `UnlockRect` does.
    autogen_texture_id: Option<TextureId>,
}

enum StretchKind {
    Texture(crate::encoder::TextureInfo),
    Backbuffer(MetalHandle<MTLTextureKind>),
    /// A standalone depth-stencil surface's retained `Private` depth texture.
    DepthStencil(MetalHandle<MTLTextureKind>),
}

fn resolve_stretch_surface(
    surf: *mut crate::surface::Direct3DSurface9,
) -> Option<StretchSurfaceInfo> {
    if surf.is_null() {
        return None;
    }
    // SAFETY: `surf` is non-null (checked above) and is the
    // caller-supplied surface pointer from a `StretchRect` thunk; the
    // caller must pass live surface wrappers.
    let s = unsafe { &*surf };
    let parent = s.parent_texture();
    if !parent.is_null() {
        // SAFETY: `parent` is non-null (checked above) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // the surface is alive.
        let tex = unsafe { &*parent };
        let level = s.mip_level();
        let lvl_idx = level as usize;
        let info = tex.inner().texture_info();
        let mut flags = StretchSurfaceFlags::empty();
        flags.set(
            StretchSurfaceFlags::IS_RENDER_TARGET,
            (tex.d3d_usage() & D3DUSAGE_RENDERTARGET) != 0,
        );
        flags.set(
            StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT,
            s.owns_parent_texture(),
        );
        return Some(StretchSurfaceInfo {
            kind: StretchKind::Texture(info),
            width: tex.inner().mip_width(lvl_idx),
            height: tex.inner().mip_height(lvl_idx),
            format: tex.d3d_format(),
            mip_level: level,
            pool: tex.d3d_pool(),
            flags,
            autogen_texture_id: (tex.inner().autogen_mipmap() && level == 0)
                .then(|| tex.texture_id()),
        });
    }
    let color = s.metal_color_handle();
    if !color.is_null() {
        // A standalone colour surface: the implicit backbuffer or a
        // `CreateRenderTarget` surface — both DEFAULT-pool render targets.
        return Some(StretchSurfaceInfo {
            kind: StretchKind::Backbuffer(color),
            width: s.standalone_width(),
            height: s.standalone_height(),
            format: s.standalone_format(),
            mip_level: 0,
            pool: D3DPOOL_DEFAULT,
            flags: StretchSurfaceFlags::IS_RENDER_TARGET,
            autogen_texture_id: None,
        });
    }
    let depth = s.metal_depth_handle();
    if !depth.is_null() {
        // A standalone depth-stencil surface (`CreateDepthStencilSurface`): a
        // DEFAULT-pool `Private` depth texture. StretchRect permits only a 1:1
        // depth→depth copy.
        return Some(StretchSurfaceInfo {
            kind: StretchKind::DepthStencil(depth),
            width: s.standalone_width(),
            height: s.standalone_height(),
            format: s.standalone_format(),
            mip_level: 0,
            pool: D3DPOOL_DEFAULT,
            flags: StretchSurfaceFlags::IS_DEPTH_STENCIL,
            autogen_texture_id: None,
        });
    }
    None
}

extern "system" fn device_color_fill(
    this: *mut c_void,
    surface: *mut c_void,
    rect: *const c_void,
    color: u32,
) -> i32 {
    use crate::surface::Direct3DSurface9;

    let _timer = device_timer(this, DeviceSubCategory::Frame);
    if surface.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `surface` is a live IDirect3DSurface9 per the D3D9 ABI.
    let s = unsafe { &*surface.cast::<Direct3DSurface9>() };
    let parent = s.parent_texture();
    if parent.is_null() {
        // Standalone colour surface (the implicit backbuffer or a
        // CreateRenderTarget surface): fill its live colour MTLTexture directly.
        // Only a whole-surface fill (NULL rect) is supported here — the only case
        // the conformance suite exercises. A colour-less standalone surface
        // (depth-stencil, or a system-memory offscreen-plain surface), or a
        // sub-rect, is INVALIDCALL.
        let color_handle = s.metal_color_handle();
        if color_handle.is_null() || !rect.is_null() {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "ColorFill on a standalone surface without a colour handle (or with a sub-rect) → INVALIDCALL");
            return D3DERR_INVALIDCALL;
        }
        let width = s.standalone_width();
        let height = s.standalone_height();
        let format = s.standalone_format();
        // Encode the fill colour; an unmapped format still succeeds but leaves
        // the surface unfilled (the colour check, not the HR, is what would fail).
        let Some(pixel) = mtld3d_core::convert::d3dcolor_fill_pixel_bytes(color, format) else {
            return D3D_OK;
        };
        let bpp = pixel.len();
        if width == 0 || height == 0 || bpp == 0 {
            return D3D_OK;
        }
        let mut tight = vec![0u8; width as usize * height as usize * bpp];
        for chunk in tight.chunks_exact_mut(bpp) {
            chunk.copy_from_slice(&pixel);
        }
        let handle = color_handle.raw();
        let bpp_u = u32::try_from(bpp).expect("ColorFill bpp fits u32");
        obj.inner().push_op(Box::new(move |enc: &mut FrameEncoder| {
            enc.upload_bytes_to_color_handle(handle, &tight, width, height, bpp_u);
        }));
        return D3D_OK;
    }
    // SAFETY: `parent` non-null (checked); its refcount keeps it alive while
    // the surface is alive.
    let tex = unsafe { &*parent };
    // D3D9: ColorFill on a texture surface is valid for a DEFAULT-pool render
    // target AND for a DEFAULT-pool offscreen-plain surface (which owns its
    // internal texture). Managed / sysmem / scratch and an ordinary DEFAULT
    // texture-level surface are rejected.
    if tex.d3d_pool() != D3DPOOL_DEFAULT
        || (tex.d3d_usage() & D3DUSAGE_RENDERTARGET == 0 && !s.owns_parent_texture())
    {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "ColorFill: surface is not a DEFAULT render target or offscreen-plain → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    let level = s.mip_level();
    let lvl = level as usize;
    let width = tex.inner().mip_width(lvl);
    let height = tex.inner().mip_height(lvl);
    let format = tex.d3d_format();
    // NULL rect fills the whole mip; otherwise fill the requested sub-rect.
    let (origin_x, origin_y, region_w, region_h) = if rect.is_null() {
        (0, 0, width, height)
    } else {
        // SAFETY: vtable in-param; `rect` is *const D3DRECT per the D3D9 ABI.
        let r = unsafe { *rect.cast::<mtld3d_types::D3DRECT>() };
        (
            r.x1.max(0).cast_unsigned(),
            r.y1.max(0).cast_unsigned(),
            (r.x2 - r.x1).max(0).cast_unsigned(),
            (r.y2 - r.y1).max(0).cast_unsigned(),
        )
    };
    // A block-compressed ColorFill sub-rect must land on the block grid;
    // a misaligned rect is INVALIDCALL. Uncompressed formats have block 1×1 so
    // this is inert.
    if !rect.is_null()
        && let Some(fmt) = map_d3d_format(format)
    {
        let (bw, bh) = (fmt.block_width(), fmt.block_height());
        let x2 = origin_x + region_w;
        let y2 = origin_y + region_h;
        if (bw > 1 || bh > 1)
            && (!origin_x.is_multiple_of(bw)
                || !origin_y.is_multiple_of(bh)
                || (!x2.is_multiple_of(bw) && x2 != width)
                || (!y2.is_multiple_of(bh) && y2 != height))
        {
            return D3DERR_INVALIDCALL;
        }
    }
    // Encode the fill colour into the destination format. Unsupported formats
    // still succeed (the colour check, not the ColorFill return, is what fails
    // for those) but leave the surface unfilled.
    let Some(pixel) = mtld3d_core::convert::d3dcolor_fill_pixel_bytes(color, format) else {
        return D3D_OK;
    };
    let bpp = pixel.len();
    if region_w == 0 || region_h == 0 || bpp == 0 {
        return D3D_OK;
    }
    let pitch = region_w as usize * bpp;
    let mut page = PageBox::new_uninit(pitch * region_h as usize);
    for chunk in page.as_mut_slice().chunks_exact_mut(bpp) {
        chunk.copy_from_slice(&pixel);
    }
    // A lockable DEFAULT offscreen-plain surface reads its fill back through
    // LockRect (CPU staging), so mirror the fill into staging. The GPU upload
    // below keeps the internal Metal texture coherent for StretchRect/sampling.
    if s.owns_parent_texture() {
        tex.inner()
            .fill_staging_region(lvl, origin_x, origin_y, region_w, region_h, &pixel);
    }
    let info = tex.inner().texture_info();
    let job = TextureUploadJob {
        info,
        arc: Arc::new(page),
        level,
        origin_x,
        origin_y,
        region_w,
        region_h,
        src_d3d_format: format,
        src_pitch: u32::try_from(pitch).expect("ColorFill row pitch fits u32"),
        bytes_per_pixel: u32::try_from(bpp).expect("ColorFill bpp fits u32"),
        // ColorFill targets a 2D surface — single slice, so the encoder keeps
        // the untouched 2D blit path.
        depth: 1,
        slice_pitch: 0,
    };
    obj.inner().push_op(Box::new(move |enc: &mut FrameEncoder| {
        enc.run_texture_upload(job);
    }));
    D3D_OK
}

extern "system" fn device_create_offscreen_plain_surface(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    pool: u32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // Shared resource handles are a D3D9Ex-only feature — a plain device rejects a
    // non-NULL pSharedHandle with E_NOTIMPL.
    if !shared_handle.is_null() {
        null_out(surface);
        return E_NOTIMPL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    // Raw `*mut DeviceInner` copied out so the DEFAULT branch can call
    // `device_create_texture` (which re-derives its own device borrow) without
    // holding a live borrow from `obj` across the call.
    let device_inner = obj.inner_ptr();
    // D3DPOOL_DEFAULT: a lockable, GPU-resident offscreen surface. Back it with
    // an internal color texture (CPU staging for LockRect + a Metal texture for
    // StretchRect) and hand back its owned level-0 surface — lock/unlock/upload
    // reuse the texture machinery, StretchRect resolves the texture handle. No
    // D3DUSAGE_* (offscreen plain), single mip.
    if pool == D3DPOOL_DEFAULT {
        let mut tex_out: *mut c_void = core::ptr::null_mut();
        let hr = device_create_texture(
            this,
            width,
            height,
            1,
            0,
            format,
            D3DPOOL_DEFAULT,
            &raw mut tex_out,
            core::ptr::null_mut(),
        );
        if hr != D3D_OK || tex_out.is_null() {
            warn!(target: LOG_TARGET,
                "reject CreateOffscreenPlainSurface({width}x{height}, format={format}, DEFAULT) → INVALIDCALL (internal texture create failed)");
            null_out(surface);
            return D3DERR_INVALIDCALL;
        }
        let surf = Direct3DSurface9::new_owned_texture_backed(
            device_inner,
            tex_out.cast::<crate::texture::Direct3DTexture9>(),
        );
        // The internal texture (created just above) forwards the device
        // reference, so this owned surface is NOT registered (no double-count).
        // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
        unsafe { OutPtr::write_opt(surface, Box::into_raw(Box::new(surf)).cast::<c_void>()) };
        return D3D_OK;
    }
    // Otherwise a CPU/system-memory offscreen surface: D3DPOOL_SYSTEMMEM (the
    // destination of GetRenderTargetData / GetFrontBufferData) and D3DPOOL_SCRATCH
    // (e.g. a cursor bitmap) are both lockable
    // system-RAM surfaces, served by the same backing. D3DPOOL_MANAGED offscreen
    // surfaces are not supported.
    if pool != D3DPOOL_SYSTEMMEM && pool != D3DPOOL_SCRATCH {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateOffscreenPlainSurface(pool={pool}) → INVALIDCALL (only D3DPOOL_SYSTEMMEM / D3DPOOL_SCRATCH / D3DPOOL_DEFAULT supported)");
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    let Some(fmt) = map_d3d_format(format) else {
        warn!(target: LOG_TARGET,
            "reject CreateOffscreenPlainSurface({width}x{height}, format={format}) → INVALIDCALL (no format mapping)");
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    // Block-compressed (DXTn) offscreen surfaces must be block-aligned in
    // width/height, like textures.
    if is_dxt_format(format)
        && (!width.is_multiple_of(fmt.block_width()) || !height.is_multiple_of(fmt.block_height()))
    {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // Block-compressed formats (DXT*, ATI*) have no bytes-per-pixel — size the
    // backing by their block grid (`ceil(dim/block) * block_bytes`). Linear
    // formats size as `aligned_pitch * height`, where the pitch rounds the
    // `width * bpp` row stride up to a 4-byte boundary — matching the pitch
    // `systemmem_lock_rect` reports, so the last locked row stays in bounds.
    let bpp = fmt.bytes_per_pixel();
    let bytes = if bpp == 0 {
        let blocks_w = (width as usize).div_ceil(fmt.block_width().max(1) as usize);
        let blocks_h = (height as usize).div_ceil(fmt.block_height().max(1) as usize);
        blocks_w
            .saturating_mul(blocks_h)
            .saturating_mul(fmt.block_bytes() as usize)
    } else {
        let pitch = width.saturating_mul(bpp).next_multiple_of(4);
        (pitch as usize).saturating_mul(height as usize)
    };
    let surf = Direct3DSurface9::new_system_memory(
        obj.inner_ptr(),
        width,
        height,
        format,
        pool,
        PageBox::new_uninit(bytes),
    );
    let surf_ptr = Box::into_raw(Box::new(surf));
    // SAFETY: `surf_ptr` is a freshly created, live system-memory surface at
    // refcount 1.
    unsafe { crate::com_ref::com_register_child(surf_ptr) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_set_render_target(
    this: *mut c_void,
    index: u32,
    surface: *mut c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if index != 0 {
        warn!(
            target: LOG_TARGET,
            "reject SetRenderTarget(index={index}) → INVALIDCALL (only RT0 supported)"
        );
        return D3DERR_INVALIDCALL;
    }
    if surface.is_null() {
        // D3D9 spec: RT0 must remain non-null.
        warn!(target: LOG_TARGET, "reject SetRenderTarget(index=0, null) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }

    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // Pull width/height via GetDesc through the surface vtable so we cover
    // both standalone (CreateRenderTarget) and texture-backed (GetSurfaceLevel
    // on a render-target texture) cases uniformly.
    let surf = surface.cast::<Direct3DSurface9>();
    // A render target created by a DIFFERENT device is INVALIDCALL, and (because
    // the call fails) must leave the currently-bound RT0 untouched. Every
    // surface records its owning device
    // (used by GetDevice); compare it against this device before any mutation.
    // WoW uses a single device, so this never fires in-game.
    // SAFETY: `surf` is non-null (validated above) and points to a live surface.
    if unsafe { (*surf).device_inner() } != obj.inner_ptr() {
        warn!(target: LOG_TARGET, "reject SetRenderTarget: surface owned by a different device → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `surf` is non-null (validated by `surface.is_null()` check
    // earlier in this fn) and points to a live `Direct3DSurface9` whose
    // refcount keeps it alive while bound on the device.
    let vtbl = unsafe { (*surf).vtbl() };
    let mut desc = mtld3d_types::D3DSURFACE_DESC {
        format: 0,
        resource_type: 0,
        usage: 0,
        pool: 0,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        width: 0,
        height: 0,
    };
    // SAFETY: calling the just-loaded `get_desc` thunk through the
    // surface vtable with `surface` as `this` and `desc` as the writable
    // out-pointer.
    if unsafe { (vtbl.get_desc)(surface, &raw mut desc) } != 0 {
        warn!(target: LOG_TARGET, "reject SetRenderTarget: GetDesc failed");
        return D3DERR_INVALIDCALL;
    }
    // The destination must be render-target-capable.
    // GetDesc reports the surface's true usage — the parent texture's
    // D3DUSAGE_RENDERTARGET for a GetSurfaceLevel surface, or RENDERTARGET for
    // the implicit backbuffer / a CreateRenderTarget surface — so every
    // legitimate render-target bind passes.
    if desc.usage & D3DUSAGE_RENDERTARGET == 0 {
        warn!(
            target: LOG_TARGET,
            "reject SetRenderTarget: surface is not a render target (usage={:#x}) → INVALIDCALL",
            desc.usage
        );
        return D3DERR_INVALIDCALL;
    }

    // Capture attachment info for the encoder thread. Texture-backed
    // surfaces (from `GetSurfaceLevel` on a `D3DUSAGE_RENDERTARGET`
    // texture) carry the parent's `TextureInfo` so the encoder can
    // lazily create the Metal texture; standalone surfaces (from
    // `GetBackBuffer` / `GetRenderTarget` on the default RT) fall
    // back to the backbuffer handle.
    // SAFETY: `surf` is the live surface validated above.
    let parent = unsafe { (*surf).parent_texture() };
    // SAFETY: `surf` is the live surface validated above.
    let standalone_color = unsafe { (*surf).metal_color_handle() };
    let info = if parent.is_null() {
        if !standalone_color.is_null() && standalone_color != dev.backbuffer_handle {
            // A standalone `CreateRenderTarget` colour surface: parent-null but
            // it owns a persistent Metal colour texture distinct from the
            // backbuffer. Bind that texture at its own format/size instead of
            // mis-binding the backbuffer, which would desync the pipeline
            // colour format from the real attachment.
            let mapping = map_d3d_format(desc.format);
            let fmt = mapping
                .as_ref()
                .map_or(mtld3d_shared::mtl::PixelFormat::Bgra8Unorm, |m| {
                    m.metal_pixel_format()
                });
            // Unknown formats default to alpha-bearing (the pre-clamp
            // behaviour); a real X8R8G8B8 surface reports `false` here.
            let has_alpha = mapping.as_ref().is_none_or(FormatMapping::has_alpha);
            RtBinding::StandaloneColor {
                handle: standalone_color,
                format: fmt,
                has_alpha,
                width: desc.width,
                height: desc.height,
            }
        } else {
            // The implicit backbuffer render target.
            RtBinding::Backbuffer {
                handle: dev.backbuffer_handle,
                width: dev.backbuffer_width,
                height: dev.backbuffer_height,
            }
        }
    } else {
        // SAFETY: `parent` is non-null (else branch) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // the surface is bound.
        let parent_tex = unsafe { &*parent };
        RtBinding::Texture {
            info: TextureInfo {
                texture_id: parent_tex.texture_id(),
                width: parent_tex.width(),
                height: parent_tex.height(),
                depth: 1,
                levels: parent_tex.levels(),
                pixel_format: parent_tex.metal_pixel_format(),
                has_swizzle: parent_tex.has_swizzle(),
                swizzle: parent_tex.swizzle(),
                usage_flags: parent_tex.usage_flags()
                    | mtld3d_shared::mtl::TextureUsage::RENDER_TARGET,
            },
            // Unknown formats default to alpha-bearing (the pre-clamp
            // behaviour); a real X8R8G8B8 RT texture reports `false`.
            has_alpha: map_d3d_format(desc.format)
                .as_ref()
                .is_none_or(FormatMapping::has_alpha),
            width: desc.width,
            height: desc.height,
        }
    };

    // Autogen render targets: if RT0 is changing away from an
    // `D3DUSAGE_AUTOGENMIPMAP` texture, regenerate its mip chain now (ordered
    // after the render/clear that just modified its level 0). Track the new
    // RT0's autogen id for the next change.
    let new_autogen = if parent.is_null() {
        None
    } else {
        // SAFETY: `parent` is the live parent texture validated above.
        let pt = unsafe { &*parent };
        pt.inner().autogen_mipmap().then(|| pt.texture_id())
    };
    if let Some(old_id) = dev.cur_autogen_rt_id.take()
        && Some(old_id) != new_autogen
    {
        dev.push_op(Box::new(move |enc| {
            enc.run_generate_mipmaps_ordered(old_id);
        }));
    }
    dev.cur_autogen_rt_id = new_autogen;

    dev.bound_rt_mut()
        .replace_render_target(surf, desc.width, desc.height);

    // Remember the applied binding so a mid-frame readback flush can re-assert
    // it into the fresh frame (the encoder's pass state resets to the
    // backbuffer default each frame; a D3D9 RT binding outlives an internal
    // flush — see `last_color_rt_binding`).
    dev.last_color_rt_binding = Some(info.clone());
    dev.push_color_rt_binding_op(info);

    // D3D9 spec: SetRenderTarget resets the viewport to cover the new
    // RT's full dimensions. Games rely on this, and skipping it leaves
    // draws clipped to whatever rect was last set, effectively
    // rendering into a sub-rect of the new attachment.
    dev.set_viewport(mtld3d_types::D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: desc.width,
        height: desc.height,
        min_z: 0.0,
        max_z: 1.0,
    });
    // D3D9 likewise resets the scissor rect to the new RT's full dimensions,
    // overriding any rect set before the switch.
    dev.set_scissor_rect([0, 0, desc.width, desc.height]);
    // RT swap: depth/stencil resolution may change; new RT might also
    // already be bound as a texture on some stage (sampling-from-RT). RS
    // carries the reset scissor; VS_CONST is internalized into `set_viewport`.
    dev.mark_snapshot_dirty(SnapshotDirty::RT_DS | SnapshotDirty::STAGES | SnapshotDirty::RS);
    D3D_OK
}

extern "system" fn device_get_render_target(
    this: *mut c_void,
    index: u32,
    surface: *mut *mut c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if surface.is_null() || index != 0 {
        warn!(target: LOG_TARGET, "reject GetRenderTarget(index={index}) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();

    // If a custom RT was set via SetRenderTarget, hand it back per D3D9 spec
    // (caller releases). Otherwise the default RT is the device-owned implicit
    // backbuffer surface — a single cached object returned by every call (the
    // `pRenderTarget == pBackBuffer` and refcount-0 identity the suite checks),
    // resolving its Metal handle live so StretchRect / LockRect readback see the
    // current backbuffer.
    let bound = inner.bound_rt().render_target();
    let surf = if bound.is_null() {
        inner.get_or_create_implicit_render_target()
    } else {
        bound
    };
    // SAFETY: `surf` is non-null — either the live bound RT (refcount keeps it
    // alive while bound) or the live cached implicit RT. Its AddRef thunk
    // forwards to the device on the implicit surface's 0→1 transition.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: calling the surface's AddRef thunk; D3D9 mandates AddRef on return.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: `surface` is the caller's out-pointer per the D3D9 ABI.
    unsafe { *surface = surf.cast::<c_void>() };
    D3D_OK
}

/// `SetDepthStencilSurface` capture shape, owned by the closure pushed to the encoder thread.
///
/// `Lazy` defers the `MTLTexture` lookup to the encoder so a sampleable
/// shadow map's Metal handle is created (or reused from the cache) on
/// first bind, mirroring how `SetRenderTarget` handles texture-backed
/// render targets. `Eager` is the standalone-surface path
/// (`CreateDepthStencilSurface`) where the handle is known up-front.
#[derive(Clone)]
enum DepthBinding {
    None,
    Eager(MetalHandle<MTLTextureKind>),
    Lazy(TextureInfo),
}

extern "system" fn device_set_depth_stencil_surface(
    this: *mut c_void,
    surface: *mut c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let surf = surface.cast::<Direct3DSurface9>();
    // A non-NULL depth-stencil surface must report D3DUSAGE_DEPTHSTENCIL; NULL
    // unbinds the depth buffer. GetDesc reports the true usage (the parent
    // depth texture's DEPTHSTENCIL for a sampleable shadow map, the implicit
    // DS, or a CreateDepthStencilSurface surface), so WoW's shadow-map and
    // implicit-DS binds all pass. Validate before mutating any device state.
    if !surf.is_null() {
        // SAFETY: `surf` is non-null (checked) and a live `Direct3DSurface9`.
        let vtbl = unsafe { (*surf).vtbl() };
        let mut desc = mtld3d_types::D3DSURFACE_DESC {
            format: 0,
            resource_type: 0,
            usage: 0,
            pool: 0,
            multi_sample_type: 0,
            multi_sample_quality: 0,
            width: 0,
            height: 0,
        };
        // SAFETY: the `get_desc` thunk with `surface` as `this` and `desc` as
        // the writable out-pointer.
        if unsafe { (vtbl.get_desc)(surface, &raw mut desc) } != 0
            || desc.usage & D3DUSAGE_DEPTHSTENCIL == 0
        {
            warn!(
                target: LOG_TARGET,
                "reject SetDepthStencilSurface: surface is not a depth-stencil (usage={:#x}) → INVALIDCALL",
                desc.usage
            );
            return D3DERR_INVALIDCALL;
        }
    }
    dev.bound_rt_mut().replace_depth_stencil(surf);
    // A null surface explicitly removes the depth buffer; track it so the
    // pipeline snapshot reports no depth instead of falling back to the
    // device-default auto depth-stencil.
    dev.flags
        .set(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND, surf.is_null());

    // Pick the actual Metal depth-texture handle to bind. Three shapes:
    // - null surface → unbind depth.
    // - texture-backed depth surface (sampleable shadow map from
    //   `CreateTexture(D24X8, DEPTHSTENCIL)`) → capture the parent's
    //   `TextureInfo` and resolve via `get_or_create_texture` on the
    //   encoder thread, mirroring how `SetRenderTarget` handles RT
    //   textures.
    // - standalone depth surface from `CreateDepthStencilSurface` /
    //   `GetDepthStencilSurface` → eager `metal_depth_handle()` capture.
    let binding = if surf.is_null() {
        DepthBinding::None
    // SAFETY: `surf` is non-null (else branch) and is the
    // caller-supplied `Direct3DSurface9*`; the device's bound-RT
    // tracker keeps the surface alive while bound.
    } else if let Some(info) = unsafe { (*surf).depth_texture_info() } {
        // Trace probe: one line per distinct (parent_texture, mip_level)
        // texture-backed depth bind, so the cascade depth-bind pattern
        // across frames is visible. `dim` cross-references the per-pass
        // `viewport=…` probe: if `viewport` is smaller than `dim` for
        // a cascade depth attachment, the caster's content lands in
        // only a sub-rect and the rest of the texture stays cleared.
        // Zero-cost when `mtld3d::d3d9::depth=trace` isn't enabled.
        // SAFETY: `surf` is non-null (validated by the `let Some(info)
        // = …` arm above) and points to a live surface.
        let mip = unsafe { (*surf).mip_level() };
        let w = info.width;
        let h = info.height;
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: (info.texture_id.raw() << 8) | u64::from(mip),
            "depth: surface bind tex={:#x} mip={} dim={w}x{h}",
            info.texture_id,
            mip
        );
        DepthBinding::Lazy(info)
    } else {
        // SAFETY: `surf` is non-null (else-if branch) and points to a
        // live surface.
        let h = unsafe { (*surf).metal_depth_handle() };
        if h.is_null() {
            error!(
                target: LOG_TARGET,
                "SetDepthStencilSurface: surface {:#x} has no Metal depth backing — binding depth=0",
                surface as usize
            );
        }
        DepthBinding::Eager(h)
    };
    // `is_sampleable` distinguishes a sampleable shadow map
    // (`CreateTexture(D24X8, DEPTHSTENCIL)` — the `DepthBinding::Lazy`
    // path) from a standalone non-sampleable depth surface
    // (`CreateDepthStencilSurface` — the `DepthBinding::Eager` path).
    // The pass machine uses this to keep `Store` unconditionally on
    // sampleable shadow maps (Rule B short-circuit), since they may
    // be sampled in a future frame even if no sample lands this
    // frame — typical of cascade-3 in CSM rotations.
    let is_sampleable = matches!(binding, DepthBinding::Lazy(_));
    // Whether the bound depth attachment is a combined depth+stencil format
    // (D24S8 etc. → the combined Metal texture), so the clear-quad / draw
    // pipelines declare matching depth/stencil attachment formats. Mirrors the
    // snapshot's `depth_format_has_stencil(standalone_format())`. Sampleable
    // (Lazy) depths are the depth-only shadow-map path; `None` has no depth.
    let depth_has_stencil = if matches!(binding, DepthBinding::Eager(_)) && !surf.is_null() {
        // SAFETY: the Eager arm implies `surf` is the live, non-null bound
        // surface (already deref'd above to build the binding).
        depth_format_has_stencil(unsafe { (*surf).standalone_format() })
    } else {
        false
    };
    // Remember the binding so a mid-frame readback flush can re-assert it (the
    // encoder's pass state re-attaches the implicit auto-depth each frame; a
    // D3D9 depth bind — including an explicit unbind — outlives an internal
    // flush, see `last_depth_binding`).
    dev.last_depth_binding = Some((binding.clone(), is_sampleable, depth_has_stencil));
    dev.push_depth_binding_op(binding, is_sampleable, depth_has_stencil);
    // A depth-attachment change ends the current Metal render pass; the next FF
    // draw runs on a fresh encoder, so its FF vertex constants (`vs_c`, buffer
    // 15) must be re-emitted. `SetRenderTarget` gets this for free via its
    // viewport reset (which marks `FfVsDirty` + `VS_CONST`); mirror that here so
    // the FF VS const range is re-pushed after the pass break.
    dev.ff_state
        .mark_ff_vs_dirty(mtld3d_core::ff_state::FfVsDirty::WV);
    let mask = dev.ff_aware_mask(SnapshotDirty::RT_DS | SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    D3D_OK
}

extern "system" fn device_get_depth_stencil_surface(
    this: *mut c_void,
    surface: *mut *mut c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if surface.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // `GetDepthStencilSurface` reflects the currently *bound* depth-stencil, not
    // merely the device's auto depth-stencil: an explicit `SetDepthStencilSurface
    // (NULL)` unbinds it, so report "none bound" (NOTFOUND) even though the auto
    // depth texture still exists.
    let dev = obj.inner();
    if dev.flags.contains(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND) {
        // SAFETY: `surface` is non-null (checked above) and per the D3D9 ABI
        // points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *surface = core::ptr::null_mut() };
        return crate::D3DERR_NOTFOUND;
    }
    // The device-owned implicit depth-stencil surface (cached, refcount-0,
    // container = the device). A single object across calls, depth-backed so the
    // common `GetDepthStencilSurface → … → SetDepthStencilSurface(saved)`
    // save/restore restores the real Metal depth handle (resolved live). Null
    // when the device has no auto depth-stencil.
    let surf = dev.get_or_create_implicit_depth_stencil();
    if surf.is_null() {
        // SAFETY: `surface` is non-null (checked above) and per the D3D9 ABI
        // points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *surface = core::ptr::null_mut() };
        return crate::D3DERR_NOTFOUND;
    }
    // SAFETY: `surf` is the live cached implicit DS surface; its AddRef thunk
    // forwards to the device on the 0→1 transition.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: calling the surface's AddRef thunk; D3D9 mandates AddRef on return.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { *surface = surf.cast::<c_void>() };
    D3D_OK
}

extern "system" fn device_begin_scene(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // D3D9 forbids nested scenes: BeginScene while already in one fails.
    if dev.flags.contains(DeviceFlags::IN_SCENE) {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "BeginScene while already in a scene → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    dev.flags.insert(DeviceFlags::IN_SCENE);
    0 // S_OK
}

extern "system" fn device_end_scene(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // EndScene without a matching BeginScene fails.
    if !dev.flags.contains(DeviceFlags::IN_SCENE) {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "EndScene without BeginScene → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    dev.flags.remove(DeviceFlags::IN_SCENE);
    0 // S_OK
}

/// Read the `count` `D3DRECT`s an `IDirect3DDevice9::Clear` supplies at `rects`.
///
/// Each becomes an `(x1, y1, x2, y2)` tuple. `count == 0` or a null pointer means "no
/// rects" → empty (the caller then clears the whole viewport). Defensive
/// against a null pointer carried with a non-zero count.
fn clear_target_rects(count: u32, rects: *const c_void) -> Vec<(i32, i32, i32, i32)> {
    if count == 0 || rects.is_null() {
        return Vec::new();
    }
    // SAFETY: per the Clear ABI, `rects` points to `count` contiguous, caller-
    // owned `D3DRECT`s when non-null (checked above); read-only access here.
    let slice = unsafe {
        core::slice::from_raw_parts(rects.cast::<mtld3d_types::D3DRECT>(), count as usize)
    };
    slice.iter().map(|r| (r.x1, r.y1, r.x2, r.y2)).collect()
}

/// Intersect two half-open D3D9 rects `(x1, y1, x2, y2)`.
///
/// Returns the overlap,
/// or `None` if disjoint or either is degenerate. Clips a `Clear` rect to the
/// scissor rect when `D3DRS_SCISSORTESTENABLE` is on.
fn intersect_d3d_rects(
    a: (i32, i32, i32, i32),
    b: (i32, i32, i32, i32),
) -> Option<(i32, i32, i32, i32)> {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = a.2.min(b.2);
    let y2 = a.3.min(b.3);
    (x2 > x1 && y2 > y1).then_some((x1, y1, x2, y2))
}

extern "system" fn device_clear(
    this: *mut c_void,
    count: u32,
    rects: *const c_void,
    flags: u32,
    color: u32,
    z: f32,
    _stencil: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // Clearing depth or stencil with no depth-stencil attachment bound is
    // invalid: a prior `SetDepthStencilSurface(NULL)` leaves no surface to
    // clear.
    if flags & (D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL) != 0 && !dev.depth_stencil_bound() {
        return D3DERR_INVALIDCALL;
    }

    if flags & D3DCLEAR_TARGET != 0 {
        // D3DCOLOR is ARGB; unpack to normalized float bits so the encoder
        // can fold them into a Metal `MTLLoadAction::Clear` at pass-begin.
        let rgba = mtld3d_core::convert::d3dcolor_to_rgba_f32(color);
        let r_bits = f32::to_bits(rgba[0]);
        let g_bits = f32::to_bits(rgba[1]);
        let b_bits = f32::to_bits(rgba[2]);
        let a_bits = f32::to_bits(rgba[3]);

        // Clear also honours D3DRS_SCISSORTESTENABLE: when on, every cleared
        // region is additionally clipped to the (non-degenerate) device scissor
        // rect. Resolved on the API thread; the encoder then clips ∩ viewport.
        // `scissor_rect()` is stored as [x, y, width, height]; convert to the
        // half-open `(x1, y1, x2, y2)` the rect intersectors expect.
        let s = dev.scissor_rect(); // [x, y, width, height]
        let scissor_on =
            dev.render_state(D3DRS_SCISSORTESTENABLE as usize) != 0 && s[2] > 0 && s[3] > 0;
        let scissor = (
            s[0].cast_signed(),
            s[1].cast_signed(),
            s[0].saturating_add(s[2]).cast_signed(),
            s[1].saturating_add(s[3]).cast_signed(),
        );

        // D3D9 Clear's pRects/Count semantics:
        //  - pRects == NULL  → clear the whole target (Count ignored). With the
        //    scissor on, whole target ∩ scissor == the scissor rect.
        //  - pRects != NULL, Count == 0 → clear NOTHING (a no-op).
        //  - pRects != NULL, Count >  0 → clear each rect (∩ scissor if on).
        // A combined TARGET|ZBUFFER clear keeps both attachments on the fold
        // path (whole-attachment loadAction); only a colour-ONLY Clear(NULL)
        // honours viewport bounding via a scissored clear-quad, so the depth
        // side is never forced onto the (state-sensitive) clear-quad path.
        let color_only = flags & (D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL) == 0;
        if rects.is_null() {
            if scissor_on {
                let rects = vec![scissor];
                dev.push_op(Box::new(move |enc| {
                    enc.clear_color_rects(r_bits, g_bits, b_bits, a_bits, &rects);
                }));
            } else if color_only {
                dev.push_op(Box::new(move |enc| {
                    enc.clear_color_bounded_to_viewport(r_bits, g_bits, b_bits, a_bits);
                }));
            } else {
                dev.push_op(Box::new(move |enc| {
                    enc.clear_color(r_bits, g_bits, b_bits, a_bits);
                }));
            }
        } else if count > 0 {
            let mut rects = clear_target_rects(count, rects);
            if scissor_on {
                rects.retain_mut(|r| {
                    intersect_d3d_rects(*r, scissor).is_some_and(|clipped| {
                        *r = clipped;
                        true
                    })
                });
            }
            dev.push_op(Box::new(move |enc| {
                enc.clear_color_rects(r_bits, g_bits, b_bits, a_bits, &rects);
            }));
        }
    }

    if flags & D3DCLEAR_ZBUFFER != 0 {
        let value_bits = f32::to_bits(z);
        dev.push_op(Box::new(move |enc| {
            enc.clear_depth(value_bits);
        }));
    }

    if flags & D3DCLEAR_STENCIL != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "Clear: D3DCLEAR_STENCIL flag set but stencil clear not implemented");
    }

    0 // S_OK
}

extern "system" fn device_set_transform(
    this: *mut c_void,
    state: u32,
    matrix: *const c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable in-param; `matrix` is *const D3DMATRIX per ABI.
    let Some(m) = (unsafe { ValueIn::<D3DMATRIX>::read_opt(matrix) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::Transform { state, matrix: m });
        return D3D_OK;
    }
    // Unknown D3DTS_* indices (vertex blending etc.) are silently accepted.
    dev.ff_state_mut().set_transform(state, &m);
    let mut mask = dev.ff_aware_mask(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
    // Active table fog keys its Z-vs-W source on the projection matrix's
    // 4th column (`VariantKey::fog_source_w`), so a PROJECTION write must
    // rebuild the variant. Gated on live table fog so vertex-fog-only games
    // don't churn the variant on every per-frame projection update.
    if state == mtld3d_types::D3DTS_PROJECTION
        && dev.render_states()[D3DRS_FOGENABLE as usize] != 0
        && matches!(dev.render_states()[D3DRS_FOGTABLEMODE as usize], 1..=3)
    {
        mask |= SnapshotDirty::VARIANT | SnapshotDirty::PS_SOURCE;
    }
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_transform(this: *mut c_void, state: u32, matrix: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if matrix.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let m = dev
        .ff_state()
        .transform(state)
        .copied()
        .unwrap_or(D3DMATRIX::IDENTITY);
    // SAFETY: `matrix` is non-null (checked above) and per the D3D9
    // ABI points to a writable `D3DMATRIX` slot owned by the caller.
    unsafe {
        *matrix.cast::<D3DMATRIX>() = m;
    }
    0 // S_OK
}

extern "system" fn device_multiply_transform(
    this: *mut c_void,
    state: u32,
    matrix: *const c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable in-param; `matrix` is *const D3DMATRIX per ABI.
    let Some(rhs) = (unsafe { ValueIn::<D3DMATRIX>::read_opt(matrix) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // MultiplyTransform is NOT a recordable state-block operation in D3D9: even
    // inside a Begin/EndStateBlock it mutates the live device transform
    // immediately and is never captured into the block (GetTransform after
    // EndStateBlock returns the multiplied matrix, and a later Capture/Apply
    // does not restore it). So
    // always apply to live FF state, regardless of recording.
    dev.ff_state_mut().multiply_transform(state, &rhs);
    let mask = dev.ff_aware_mask(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_set_viewport(this: *mut c_void, viewport: *const c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    // SAFETY: vtable in-param; `viewport` is *const D3DVIEWPORT9 per ABI.
    let Some(v) = (unsafe { ValueIn::<D3DVIEWPORT9>::read_opt(viewport) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::Viewport(v));
        return D3D_OK;
    }
    dev.set_viewport(v);
    D3D_OK
}

extern "system" fn device_get_viewport(this: *mut c_void, viewport: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    if viewport.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: vtable out-param; `viewport` is *mut D3DVIEWPORT9 per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(viewport.cast::<D3DVIEWPORT9>(), dev.viewport()) };
    D3D_OK
}

extern "system" fn device_set_material(this: *mut c_void, material: *const c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable in-param; `material` is *const D3DMATERIAL9 per ABI.
    let Some(m) = (unsafe { ValueIn::<D3DMATERIAL9>::read_opt(material) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::Material(m));
        return D3D_OK;
    }
    dev.ff_state_mut().set_material(&m);
    let mask = dev.ff_aware_mask(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_material(this: *mut c_void, material: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if material.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: vtable out-param; `material` is *mut D3DMATERIAL9 per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(material.cast::<D3DMATERIAL9>(), *dev.ff_state().material()) };
    0 // S_OK
}

extern "system" fn device_set_light(this: *mut c_void, index: u32, light: *const c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable in-param; `light` is *const D3DLIGHT9 per ABI.
    let Some(l) = (unsafe { ValueIn::<D3DLIGHT9>::read_opt(light) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::Light { index, light: l });
        return D3D_OK;
    }
    dev.ff_state_mut().set_light_at(index, &l);
    let mask = dev.ff_aware_mask(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_light(this: *mut c_void, index: u32, light: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if light.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // D3D9: GetLight fails (leaving the caller's buffer untouched) for a slot
    // that was never defined via SetLight/LightEnable.
    let Some(l) = dev.ff_state().get_light_at(index) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `light` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DLIGHT9` slot owned by the caller.
    unsafe {
        *light.cast::<D3DLIGHT9>() = l;
    }
    0 // S_OK
}

extern "system" fn device_light_enable(this: *mut c_void, index: u32, enable: i32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let on = enable != 0;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::LightEnable { index, enable: on });
        return D3D_OK;
    }
    dev.ff_state_mut().set_light_enabled_at(index, on);
    let mask = dev.ff_aware_mask(SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_light_enable(this: *mut c_void, index: u32, enable: *mut i32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if enable.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // D3D9: GetLightEnable fails (leaving the caller's BOOL untouched) for a
    // slot that was never defined via SetLight/LightEnable.
    if !dev.ff_state().is_light_defined_at(index) {
        return D3DERR_INVALIDCALL;
    }
    // D3D9 reports the enabled flag as 128 (not 1).
    let enabled = if dev.ff_state().is_light_enabled_at(index) {
        128
    } else {
        0
    };
    // SAFETY: `enable` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `BOOL` (i32) slot owned by the caller.
    unsafe {
        *enable = enabled;
    }
    0 // S_OK
}

extern "system" fn device_set_clip_plane(this: *mut c_void, index: u32, plane: *const f32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if plane.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `plane` is non-null (checked) and per the D3D9 ABI points to the
    // 4 readable f32 plane-equation coefficients (A, B, C, D).
    let coeffs = unsafe { *plane.cast::<[f32; 4]>() };
    obj.inner().set_clip_plane(index, coeffs);
    0 // S_OK
}

extern "system" fn device_get_clip_plane(this: *mut c_void, index: u32, plane: *mut f32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    if plane.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let coeffs = obj.inner().clip_plane(index);
    // SAFETY: `plane` is non-null (checked) and per the D3D9 ABI points to 4
    // writable f32 slots owned by the caller.
    unsafe { *plane.cast::<[f32; 4]>() = coeffs };
    0 // S_OK
}

extern "system" fn device_set_render_state(this: *mut c_void, state: u32, value: u32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::RenderState);
    if (state as usize) >= RENDER_STATE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::RenderState { state, value });
        return D3D_OK;
    }
    // Redundant-set elimination: a write that doesn't change the stored
    // value yields a byte-identical snapshot, so skip the dirty mark.
    let changed = dev.set_render_state(state as usize, value);
    if changed {
        let mask = dev.ff_aware_mask(rs_dirty_mask(state));
        dev.mark_snapshot_dirty(mask);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetRenderState, !changed);
    0 // S_OK
}

extern "system" fn device_get_render_state(this: *mut c_void, state: u32, value: *mut u32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::RenderState);
    if (state as usize) >= RENDER_STATE_COUNT || value.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `u32` slot owned by the caller.
    unsafe { *value = dev.render_state(state as usize) };
    0 // S_OK
}

extern "system" fn device_create_state_block(
    this: *mut c_void,
    type_: u32,
    sb: *mut *mut c_void,
) -> i32 {
    use crate::state_block::Direct3DStateBlock9;
    let _timer = device_timer(this, DeviceSubCategory::StateBlock);
    if sb.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // CreateStateBlock during an open BeginStateBlock recording is INVALIDCALL —
    // reject before creating/registering any block so the device refcount is
    // unchanged.
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    if let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        && obj.inner().is_state_block_recording()
    {
        warn!(target: LOG_TARGET, "reject CreateStateBlock during BeginStateBlock recording → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    match Direct3DStateBlock9::capture(this.cast::<Direct3DDevice9>(), type_) {
        Ok(obj) => {
            // SAFETY: vtable out-param; `sb` is *mut *mut c_void per IDirect3DDevice9 ABI.
            let sb_ptr = Box::into_raw(Box::new(obj));
            // SAFETY: `sb_ptr` is a freshly created, live state block at refcount 1.
            unsafe { crate::com_ref::com_register_child(sb_ptr) };
            // SAFETY: vtable out-param; `sb` is *mut *mut c_void per the ABI.
            unsafe { OutPtr::write_opt(sb, sb_ptr.cast::<c_void>()) };
            D3D_OK
        }
        Err(e) => e,
    }
}

extern "system" fn device_begin_state_block(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::StateBlock);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !dev.begin_state_block_recording() {
        warn!(
            target: LOG_TARGET,
            "BeginStateBlock called while another recording is in progress → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    D3D_OK
}

extern "system" fn device_end_state_block(this: *mut c_void, sb: *mut *mut c_void) -> i32 {
    use crate::state_block::Direct3DStateBlock9;

    let _timer = device_timer(this, DeviceSubCategory::StateBlock);
    if sb.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let obj_ptr = this.cast::<Direct3DDevice9>();
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per
    // IDirect3DDevice9 ABI.
    let obj = unsafe { &mut *obj_ptr };
    let dev = obj.inner();
    let Some(recording) = dev.end_state_block_recording() else {
        warn!(
            target: LOG_TARGET,
            "EndStateBlock without matching BeginStateBlock → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    };
    let block = Direct3DStateBlock9::from_recording(obj_ptr, *recording);
    // SAFETY: vtable out-param; `sb` is *mut *mut c_void per IDirect3DDevice9 ABI.
    let sb_ptr = Box::into_raw(Box::new(block));
    // SAFETY: `sb_ptr` is a freshly created, live state block at refcount 1.
    unsafe { crate::com_ref::com_register_child(sb_ptr) };
    // SAFETY: vtable out-param; `sb` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(sb, sb_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_set_clip_status(this: *mut c_void, _status: *const c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetClipStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_clip_status(this: *mut c_void, _status: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetClipStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_texture(
    this: *mut c_void,
    stage: u32,
    texture: *mut *mut c_void,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Texture);
    if stage >= 8 || texture.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let tex_ptr = dev.stage_bindings().texture(stage as usize);
    if !tex_ptr.is_null() {
        // SAFETY: `tex_ptr` is non-null (checked above) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // bound on the device.
        let add_ref = unsafe { (*tex_ptr).vtbl().add_ref };
        // SAFETY: calling the just-loaded `add_ref` thunk through the
        // texture vtable; D3D9 mandates AddRef on out-pointer returns.
        unsafe { add_ref(tex_ptr.cast::<c_void>()) };
    }
    // SAFETY: `texture` is non-null (checked above) and per the D3D9
    // ABI points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *texture = tex_ptr.cast::<c_void>() };
    0 // S_OK
}

extern "system" fn device_set_texture(this: *mut c_void, stage: u32, texture: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Texture);
    if stage as usize >= STAGE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    let new_tex = texture.cast::<Direct3DTexture9>();

    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `new_tex` is null or a *mut Direct3DTexture9 supplied by
        // the calling game via SetTexture; its AddRef/Release thunks are
        // valid for the lifetime of the recording.
        let tex = unsafe { CachedComPtr::adopt(new_tex) };
        rec.record(StateOp::Texture { stage, tex });
        return D3D_OK;
    }

    let delta = dev
        .stage_bindings_mut()
        .replace_texture(stage as usize, new_tex);
    // STAGES always: the encoder binds the new handle and
    // `snapshot_stage_bindings` re-runs flush_dirty_mips/rehydrate and
    // refreshes `cached_bound_texture_mask`. The FF VS/PS keys depend
    // only on the 8-bit occupancy mask (stages 0..7); the variant only
    // on `depth_sampler_mask` / `volume_sampler_mask` (any slot's
    // depth-format-ness / 3D-ness). A swap that flips none of those
    // rebuilds byte-identical keys, so gate those pieces on the actual
    // deltas. `ff_aware_mask` still strips VS/PS bits for programmable
    // shaders.
    let mut bits = SnapshotDirty::STAGES;
    if delta.intersects(TextureSwapDelta::DEPTH_CHANGED | TextureSwapDelta::VOLUME_CHANGED) {
        bits |= SnapshotDirty::VARIANT;
    }
    let ffkey_rebuilt = (stage as usize) < 8 && delta.contains(TextureSwapDelta::OCCUPANCY_CHANGED);
    if ffkey_rebuilt {
        bits |= SnapshotDirty::VS_SOURCE
            | SnapshotDirty::VS_CONST
            | SnapshotDirty::PS_SOURCE
            | SnapshotDirty::PS_CONST;
    }
    let mask = dev.ff_aware_mask(bits);
    dev.mark_snapshot_dirty(mask);
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetTexture, !ffkey_rebuilt);
    0 // S_OK
}

extern "system" fn device_get_texture_stage_state(
    this: *mut c_void,
    stage: u32,
    type_: u32,
    value: *mut u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::TexStageState);
    if value.is_null() || stage >= 8 || (type_ as usize) >= TEXTURE_STAGE_STATE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `u32` slot owned by the caller.
    unsafe {
        *value = dev
            .ff_state()
            .texture_stage_state(stage as usize, type_ as usize);
    }
    0 // S_OK
}

extern "system" fn device_set_texture_stage_state(
    this: *mut c_void,
    stage: u32,
    type_: u32,
    value: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::TexStageState);
    if stage >= 8 || (type_ as usize) >= TEXTURE_STAGE_STATE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::TextureStageState {
            stage,
            type_,
            value,
        });
        return D3D_OK;
    }
    let changed = dev
        .ff_state_mut()
        .set_texture_stage_state(stage as usize, type_ as usize, value);
    // Redundant-set elimination: a same-value TSS write leaves every FF
    // VS/PS key byte-identical, so skip the rebuild. TSS feeds FF VS
    // layout + FF PS key + variant + constants — ff_aware strips VS/PS
    // bits for programmable shaders.
    if changed {
        let mut mask = dev.ff_aware_mask(
            SnapshotDirty::STAGES
                | SnapshotDirty::VARIANT
                | SnapshotDirty::VS_SOURCE
                | SnapshotDirty::VS_CONST
                | SnapshotDirty::PS_SOURCE
                | SnapshotDirty::PS_CONST,
        );
        // The bump-environment matrix / luminance states feed the SM1
        // texbem PS uniform (slot 12), independent of the FF keys above.
        if matches!(
            type_,
            D3DTSS_BUMPENVMAT00
                | D3DTSS_BUMPENVMAT01
                | D3DTSS_BUMPENVMAT10
                | D3DTSS_BUMPENVMAT11
                | D3DTSS_BUMPENVLSCALE
                | D3DTSS_BUMPENVLOFFSET
        ) {
            mask |= SnapshotDirty::BUMP_ENV;
        }
        dev.mark_snapshot_dirty(mask);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetTextureStageState, !changed);
    0 // S_OK
}

extern "system" fn device_get_sampler_state(
    this: *mut c_void,
    sampler: u32,
    type_: u32,
    value: *mut u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::SamplerState);
    if sampler as usize >= STAGE_COUNT || type_ as usize >= SAMPLER_STATE_COUNT || value.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `u32` slot owned by the caller.
    unsafe {
        *value = dev
            .stage_bindings()
            .sampler_state(sampler as usize, type_ as usize);
    }
    0 // S_OK
}

extern "system" fn device_set_sampler_state(
    this: *mut c_void,
    sampler: u32,
    type_: u32,
    value: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::SamplerState);
    if sampler as usize >= STAGE_COUNT || type_ as usize >= SAMPLER_STATE_COUNT {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::SamplerState {
            sampler,
            type_,
            value,
        });
        return D3D_OK;
    }
    dev.stage_bindings_mut()
        .set_sampler_state(sampler as usize, type_ as usize, value);
    // Sampler state lives inside StageBinding only.
    dev.mark_snapshot_dirty(SnapshotDirty::STAGES);
    0 // S_OK
}

extern "system" fn device_validate_device(this: *mut c_void, num_passes: *mut u32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Metal validates pipeline state at PSO-creation time, and every
    // fixed-function / shader state combination we accept renders in a single
    // pass, so the current device state is always single-pass valid. Report one
    // pass and succeed. Returning INVALIDCALL would
    // wrongly push games onto a multi-pass / capability-fallback path.
    if !num_passes.is_null() {
        // SAFETY: caller-supplied writable `u32` out-param per the D3D9 ABI.
        unsafe { *num_passes = 1 };
    }
    mtld3d_shared::log_once_info!(target: crate::LOG_TARGET, "IDirect3DDevice9::ValidateDevice: single-pass valid under Metal → S_OK (1 pass)");
    D3D_OK
}

extern "system" fn device_set_palette_entries(
    this: *mut c_void,
    _palette: u32,
    entries: *const c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if entries.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // D3D9's palette API is non-functional on modern hardware: the setter
    // succeeds and the palette is simply ignored. One validation remains:
    // without D3DPTEXTURECAPS_ALPHAPALETTE — advertised only under
    // debug.capsAll — every PALETTEENTRY's peFlags must be 0xFF (fully
    // opaque); an alpha-bearing entry
    // is INVALIDCALL. The getters stay INVALIDCALL.
    if !crate::config::CONFIG.caps_all {
        // PALETTEENTRY is { peRed, peGreen, peBlue, peFlags } (4 bytes, peFlags
        // last); the array holds 256 entries per the D3D9 ABI = 1024 bytes.
        // SAFETY: `entries` is non-null (checked) and per the D3D9 ABI points to
        // 256 PALETTEENTRYs.
        let bytes = unsafe { core::slice::from_raw_parts(entries.cast::<u8>(), 256 * 4) };
        // peFlags of each entry are bytes 3, 7, 11, … (every 4th, offset 3).
        if bytes.iter().skip(3).step_by(4).any(|&f| f != 0xFF) {
            return D3DERR_INVALIDCALL;
        }
    }
    mtld3d_shared::log_once_info!(target: crate::LOG_TARGET,
        "IDirect3DDevice9::SetPaletteEntries: palette API non-functional on modern hardware, palette ignored → S_OK");
    D3D_OK
}

extern "system" fn device_get_palette_entries(
    this: *mut c_void,
    _palette: u32,
    _entries: *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetPaletteEntries → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_current_texture_palette(this: *mut c_void, _palette: u32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Non-functional palette API: the setter succeeds, the palette is ignored.
    // GetCurrentTexturePalette stays INVALIDCALL.
    mtld3d_shared::log_once_info!(target: crate::LOG_TARGET,
        "IDirect3DDevice9::SetCurrentTexturePalette: palette API non-functional on modern hardware, ignored → S_OK");
    D3D_OK
}

extern "system" fn device_get_current_texture_palette(
    this: *mut c_void,
    _palette_number: *mut u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetCurrentTexturePalette → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_scissor_rect(this: *mut c_void, rect: *const c_void) -> i32 {
    use mtld3d_types::D3DRECT;
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    // SAFETY: vtable in-param; `rect` is *const D3DRECT per ABI.
    let Some(r) = (unsafe { ValueIn::<D3DRECT>::read_opt(rect) }) else {
        return D3DERR_INVALIDCALL;
    };
    let rect_x = r.x1.max(0).cast_unsigned();
    let rect_y = r.y1.max(0).cast_unsigned();
    let rect_w = (r.x2 - r.x1).max(0).cast_unsigned();
    let rect_h = (r.y2 - r.y1).max(0).cast_unsigned();
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::ScissorRect([rect_x, rect_y, rect_w, rect_h]));
        return D3D_OK;
    }
    dev.set_scissor_rect([rect_x, rect_y, rect_w, rect_h]);
    // scissor_rect is the only piece of RenderStateSnapshot affected.
    dev.mark_snapshot_dirty(SnapshotDirty::RS);
    D3D_OK
}

extern "system" fn device_get_scissor_rect(this: *mut c_void, rect: *mut c_void) -> i32 {
    use mtld3d_types::D3DRECT;
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    if rect.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let [x, y, w, h] = dev.scissor_rect();
    // SAFETY: `rect` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `RECT` (alias for `D3DRECT`) owned by the
    // caller.
    unsafe {
        *rect.cast::<D3DRECT>() = D3DRECT {
            x1: x.cast_signed(),
            y1: y.cast_signed(),
            x2: (x + w).cast_signed(),
            y2: (y + h).cast_signed(),
        };
    }
    D3D_OK
}

extern "system" fn device_set_software_vertex_processing(this: *mut c_void, software: i32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SW vertex processing exists for old CPUs without HW T&L. Running
    // the same pipeline on HW VP produces the same visible result, so
    // accepting the call and using HW is transparent to the game — it
    // gets the draws it expects, just faster. Modern Windows drivers
    // behave the same way. Return S_OK; log once per distinct arg so
    // both the 0 and 1 cases surface.
    mtld3d_shared::log_once_info_by!(
        target: crate::LOG_TARGET,
        key: u64::from(software.cast_unsigned()),
        "IDirect3DDevice9::SetSoftwareVertexProcessing({software}): obsolete, hardware VP is always used"
    );
    D3D_OK
}

extern "system" fn device_get_software_vertex_processing(this: *mut c_void) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DDevice9::GetSoftwareVertexProcessing: obsolete, returning 0 (hardware VP)"
    );
    0
}

extern "system" fn device_set_npatch_mode(this: *mut c_void, segments: f32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SetNPatchMode(0.0) and SetNPatchMode(1.0) both mean "N-patch
    // tessellation disabled" — i.e. default behavior. Games clear this
    // on startup without the intent to subdivide, so an INVALIDCALL was
    // wrong. Silently accept the disable. Warn once only for a real
    // subdivision request (> 1.0), which we can't honor.
    if segments <= 1.0 {
        return D3D_OK;
    }
    mtld3d_shared::log_once_warn!(
        target: crate::LOG_TARGET,
        "stub IDirect3DDevice9::SetNPatchMode({segments}) — N-patch tessellation not implemented, ignoring"
    );
    D3D_OK
}

extern "system" fn device_get_npatch_mode(this: *mut c_void) -> f32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // N-patches are obsolete fixed-function tessellation; every modern
    // driver returns 0.0 (disabled). No port candidate.
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DDevice9::GetNPatchMode: obsolete, returning 0.0"
    );
    0.0
}

/// Whether the device has a usable vertex-layout source for a draw.
///
/// A `Draw*` call needs either an explicitly bound vertex declaration or a
/// non-zero FVF; with neither, the runtime has no way to interpret the
/// vertex stream and the draw is invalid. A non-zero FVF binds its implicit
/// declaration (so `vertex_decl()` is non-null), and binding a declaration
/// directly resets the FVF to zero, so the two conditions are mutually
/// exclusive: the draw is invalid only when both are absent.
const fn has_vertex_layout_source(dev: &DeviceInner) -> bool {
    !dev.vertex_decl().is_null() || dev.fvf_field() != 0
}

extern "system" fn device_draw_primitive(
    this: *mut c_void,
    primitive_type: u32,
    start_vertex: u32,
    primitive_count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return D3DERR_INVALIDCALL;
    }
    let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
        return D3DERR_INVALIDCALL;
    };
    let vtx_count = vertex_count(primitive_type, primitive_count);
    if vtx_count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // Flush any bound buffer that's drawn while still mapped, before the draw
    // snapshot reads it.
    flush_mapped_bound_buffers(obj.inner());
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    let Some(vertex_source) = snapshot_bound_vertex_source(dev) else {
        warn!(target: LOG_TARGET, "DrawPrimitive: no vertex buffer bound");
        return D3DERR_INVALIDCALL;
    };
    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    obj.inner().push_op_inline(Op::Draw(DrawOp {
        metal_prim,
        vertex_source,
        index_source: IndexSource::None {
            start_vertex,
            vertex_count: vtx_count,
        },
    }));
    D3D_OK
}

extern "system" fn device_draw_indexed_primitive(
    this: *mut c_void,
    primitive_type: u32,
    base_vertex_index: i32,
    _min_vertex_index: u32,
    _num_vertices: u32,
    start_index: u32,
    primitive_count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return D3DERR_INVALIDCALL;
    }
    let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
        return D3DERR_INVALIDCALL;
    };
    let index_count = vertex_count(primitive_type, primitive_count);
    if index_count == 0 {
        return D3DERR_INVALIDCALL;
    }

    // Flush any bound buffer that's drawn while still mapped, before the draw
    // snapshot reads it.
    flush_mapped_bound_buffers(obj.inner());
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    let Some(vertex_source) = snapshot_bound_vertex_source(dev) else {
        // D3D9 permits an indexed draw with a valid declaration but NO stream
        // source bound: it returns S_OK (rendering is undefined) rather than
        // INVALIDCALL — unlike the non-indexed DrawPrimitive. With no vertex
        // data there is nothing to
        // render, so skip the draw and report success.
        return D3D_OK;
    };
    let Some(index_source) =
        snapshot_bound_index_source(dev, start_index, index_count, base_vertex_index)
    else {
        warn!(target: LOG_TARGET, "DrawIndexedPrimitive: no index buffer bound");
        return D3DERR_INVALIDCALL;
    };

    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    obj.inner().push_op_inline(Op::Draw(DrawOp {
        metal_prim,
        vertex_source,
        index_source,
    }));
    D3D_OK
}

/// Upload any still-mapped `Staged` VB/IB dirty span before a draw reads it.
///
/// A buffer drawn while locked never reached `Unlock`'s upload, so its latest
/// CPU writes are flushed here. The lock stays open and `dirty` stays set, so
/// `Unlock` still flushes afterwards.
fn flush_mapped_bound_buffers(dev: &mut DeviceInner) {
    let vb = dev.bound_buffers().vertex_buffer();
    let ib = dev.bound_buffers().index_buffer();
    if !vb.is_null() {
        // SAFETY: a bound vertex buffer is a live wrapper while bound.
        unsafe { (*vb).inner_mut() }.flush_staged_if_mapped(dev);
    }
    if !ib.is_null() {
        // SAFETY: a bound index buffer is a live wrapper while bound.
        unsafe { (*ib).inner_mut() }.flush_staged_if_mapped(dev);
    }
}

/// Snapshot the bound vertex buffer into `VertexSource::Bound`.
///
/// Stamps the current submit seq so the retention pipeline keeps the `PageBox`
/// alive until that seq retires. Runs on the API thread.
fn snapshot_bound_vertex_source(dev: &DeviceInner) -> Option<VertexSource> {
    let ptr = dev.bound_buffers().vertex_buffer();
    if ptr.is_null() {
        return None;
    }
    let offset = dev.bound_buffers().vb_offset();
    let stride = dev.bound_buffers().vb_stride();
    let seq = dev.current_seq();
    // SAFETY: `ptr` is non-null (checked above) and points to a live
    // `Direct3DVertexBuffer9` whose refcount keeps it alive while bound
    // on the device.
    let vb = unsafe { &mut *ptr };
    let inner = vb.inner_mut();
    inner.stamp_submit_seq(seq);
    Some(VertexSource::Bound {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr(),
        backing_len: inner.current_backing_len(),
        offset,
        stride,
    })
}

/// Snapshot the bound index buffer into `IndexSource::Bound`.
///
/// Mirrors `snapshot_bound_vertex_source`: stamps the current submit seq and
/// collapses the draw's `start_index` into a byte offset.
fn snapshot_bound_index_source(
    dev: &DeviceInner,
    start_index: u32,
    index_count: u32,
    base_vertex: i32,
) -> Option<IndexSource> {
    let ptr = dev.bound_buffers().index_buffer();
    if ptr.is_null() {
        return None;
    }
    let seq = dev.current_seq();
    // SAFETY: `ptr` is non-null (checked above) and points to a live
    // `Direct3DIndexBuffer9` whose refcount keeps it alive while bound
    // on the device.
    let ib = unsafe { &mut *ptr };
    let inner = ib.inner_mut();
    let (index_type, index_stride): (mtld3d_shared::mtl::IndexType, u32) = match inner.format() {
        D3DFMT_INDEX16 => (mtld3d_shared::mtl::IndexType::UInt16, 2),
        D3DFMT_INDEX32 => (mtld3d_shared::mtl::IndexType::UInt32, 4),
        other => {
            warn!(
                target: LOG_TARGET,
                "DrawIndexedPrimitive: unsupported index format {other}"
            );
            return None;
        }
    };
    inner.stamp_submit_seq(seq);
    Some(IndexSource::Bound {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr(),
        backing_len: inner.current_backing_len(),
        offset: start_index * index_stride,
        index_count,
        index_type,
        base_vertex,
    })
}

extern "system" fn device_draw_primitive_up(
    this: *mut c_void,
    primitive_type: u32,
    primitive_count: u32,
    vertex_data: *const c_void,
    vertex_stride: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return D3DERR_INVALIDCALL;
    }

    // Triangle fan has no Metal primitive: expand the inline fan vertices into a
    // triangle list and emit that. Kept off the (non-fan) hot path below.
    if primitive_type == D3DPT_TRIANGLEFAN {
        if vertex_data.is_null() || primitive_count == 0 {
            return D3DERR_INVALIDCALL;
        }
        let stride = vertex_stride as usize;
        let src_size = (primitive_count as usize + 2) * stride;
        // SAFETY: per the D3D9 ABI the caller guarantees `(primitive_count + 2)`
        // vertices of `vertex_stride` bytes are readable from `vertex_data`.
        let src = unsafe { core::slice::from_raw_parts(vertex_data.cast::<u8>(), src_size) };
        let expanded = convert::expand_triangle_fan(src, stride, primitive_count);
        let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
        let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
        emit_snapshot_deltas(&obj);
        drop(snap);
        let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
        let size = u32::try_from(expanded.len()).expect("triangle-fan UP size fits u32");
        let metal_prim =
            d3d_to_metal_primitive(D3DPT_TRIANGLELIST).expect("triangle list is supported");
        dev.push_op_inline(Op::Draw(DrawOp {
            metal_prim,
            vertex_source: VertexSource::Up {
                bytes: expanded,
                size,
                stride: vertex_stride,
            },
            index_source: IndexSource::None {
                start_vertex: 0,
                vertex_count: primitive_count * 3,
            },
        }));
        // D3D9 resets stream source 0 to (NULL, 0, 0) after DrawPrimitiveUP.
        dev.bound_buffers_mut().reset_stream0();
        return D3D_OK;
    }

    let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
        return D3DERR_INVALIDCALL;
    };
    let vtx_count = vertex_count(primitive_type, primitive_count);
    if vtx_count == 0 || vertex_data.is_null() {
        return D3DERR_INVALIDCALL;
    }

    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));

    let data_size = (vtx_count * vertex_stride) as usize;
    let mut vertex_copy = Vec::with_capacity(data_size);
    // SAFETY: `vertex_data` covers `data_size` bytes per the caller's stride
    // contract; `vertex_copy` was just allocated with matching capacity.
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertex_data.cast::<u8>(),
            vertex_copy.as_mut_ptr(),
            data_size,
        );
    }
    // SAFETY: `data_size <= vertex_copy.capacity()` and the bytes 0..data_size
    // were just initialised by the copy above.
    unsafe { vertex_copy.set_len(data_size) };

    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    dev.push_op_inline(Op::Draw(DrawOp {
        metal_prim,
        vertex_source: VertexSource::Up {
            bytes: vertex_copy,
            size: u32::try_from(data_size).expect("DrawPrimitiveUP data size fits u32"),
            stride: vertex_stride,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: vtx_count,
        },
    }));
    // D3D9 resets stream source 0 to (NULL, 0, 0) after DrawPrimitiveUP.
    dev.bound_buffers_mut().reset_stream0();
    D3D_OK
}

/// Clamp `max_const_used` (reported as `u32` by the parsed shader) to the 256-row mirror.
///
/// It must fit the `u16` field in
/// [`VsSource::Programmable`] / [`PsSource::Programmable`]. Shaders that
/// statically reference more than 256 const rows would already fail
/// validation upstream; this is defence-in-depth so `emit_draw` never
/// underreads the encoder mirror.
fn clamp_const_rows(max_const_used: u32) -> u16 {
    let clamped = (max_const_used as usize).min(CONSTANT_ROWS);
    u16::try_from(clamped).expect("CONSTANT_ROWS = 256 fits u16")
}

/// Bump the new VS const rows into the current-frame scratch arena and push a delta op.
///
/// The `Op::SetVsConstRange` delta keeps the encoder-side mirror
/// in sync with `ShaderBindings::vs_constants`. The encoder
/// snapshots from its own mirror at `emit_draw` time, so the API thread
/// records only the delta rather than bumping
/// `vs_constants[..max_const_used]` per draw.
pub fn propagate_vs_const_delta(dev: &mut DeviceInner, start_register: u32, slice: &[[f32; 4]]) {
    let Some((start_row, rows, data)) = bump_const_delta(dev, start_register, slice) else {
        return;
    };
    dev.push_op_inline(Op::SetVsConstRange {
        start_row,
        rows,
        data,
    });
}

pub fn propagate_ps_const_delta(dev: &mut DeviceInner, start_register: u32, slice: &[[f32; 4]]) {
    let Some((start_row, rows, data)) = bump_const_delta(dev, start_register, slice) else {
        return;
    };
    dev.push_op_inline(Op::SetPsConstRange {
        start_row,
        rows,
        data,
    });
}

/// Shared body for [`propagate_vs_const_delta`] / [`propagate_ps_const_delta`].
///
/// Clamps the (start, count) range to the
/// 256-row mirror, bumps the bytes into the per-frame scratch arena,
/// and returns `(start_row_u16, rows_u16, scratch_slice)` ready to fold
/// into a `Set*ConstRange` op. Returns `None` when the input range is
/// entirely outside the mirror (or empty after clamping); the caller
/// then skips the op push.
fn bump_const_delta(
    dev: &mut DeviceInner,
    start_register: u32,
    slice: &[[f32; 4]],
) -> Option<(u16, u16, ScratchSlice)> {
    let start = start_register as usize;
    if start >= CONSTANT_ROWS || slice.is_empty() {
        return None;
    }
    let rows = (CONSTANT_ROWS - start).min(slice.len());
    if rows == 0 {
        return None;
    }
    // SAFETY: `[f32; 4]` is POD with no padding; reinterpreting the
    // first `rows` entries as `rows * 16` bytes is sound, and the
    // borrow lifetime is local to this function (consumed by
    // `arena_alloc_bytes` below).
    let bytes = unsafe {
        core::slice::from_raw_parts(
            slice.as_ptr().cast::<u8>(),
            rows * core::mem::size_of::<[f32; 4]>(),
        )
    };
    let scratch = dev.current_frame.scratch_mut();
    let data = arena_alloc_bytes(scratch, bytes);
    // start ≤ CONSTANT_ROWS ≤ u16::MAX and rows ≤ CONSTANT_ROWS, so
    // both fit `u16` trivially.
    let start_row = u16::try_from(start).expect("start_row ≤ 256 fits u16");
    let rows_u16 = u16::try_from(rows).expect("rows ≤ 256 fits u16");
    Some((start_row, rows_u16, data))
}

/// Rebuild the dirty pieces of `DeviceInner::snapshot_cache` into a fresh `CurrentSnapshot`.
///
/// The snapshot — a mix of newly-rebuilt and cached scratch
/// pointers — is bumped into the per-frame arena, then pushed as one
/// `Op::SetCurrentSnapshot` op onto `current_frame.ops`. The encoder
/// applies the snapshot wholesale on each Draw.
///
/// Gated on `DeviceInner::snapshot_dirty`: clean draws return
/// immediately (encoder's `current_snapshot` already valid). Dirty
/// draws rebuild ONLY the pieces whose bits fired — clean pieces
/// reuse their cached scratch pointers (same per-frame arena, still
/// valid until `stamp_and_swap` sets `all()`).
fn emit_snapshot_deltas(obj: &Direct3DDevice9) {
    let dirty = obj.inner().snapshot_dirty;
    if dirty.is_empty() {
        return;
    }

    let stages_ptr = draw_snapshot_stages_ptr(DeviceInner::perf_ptr_of(obj.inner));
    let stages_timer = CycleAddTimer::start(stages_ptr);

    // STAGES first: `flush_dirty_mips` inside `snapshot_stage_bindings`
    // pushes upload ops via `dev.push_op`, which mutably borrows
    // `current_frame.ops`. Done before we hold our own `&mut` on
    // scratch/ops below.
    let stage_bindings_arr_opt = if dirty.contains(SnapshotDirty::STAGES) {
        let (arr, ff_mask, packed_mask) = snapshot_stage_bindings(obj.inner());
        obj.inner().cached_bound_texture_mask = ff_mask;
        Some((arr, packed_mask))
    } else {
        None
    };
    drop(stages_timer);

    // `keys_timer` wraps the shader-key resolution block — VDECL, RS,
    // RT_DS, VARIANT, VS_SOURCE, PS_SOURCE. These all run between the
    // stages walk and the consts work; instrumenting them as one bucket
    // attributes per-draw cost that would otherwise fall into the "other"
    // residual. Dropped just before `consts_timer`
    // starts so the buckets don't double-count.
    let keys_timer =
        CycleAddTimer::start(draw_snapshot_keys_ptr(DeviceInner::perf_ptr_of(obj.inner)));
    let dev = obj.inner();

    // VDECL FIRST — its rebuild updates `dev.cached_ff_vs_layout`,
    // which conflicts with the long-lived `dev.render_states()` borrow
    // taken below.
    let vdecl_value = if dirty.contains(SnapshotDirty::VDECL) {
        let bound_vertex_shader = dev.shader_bindings().vertex_shader();
        let fvf = dev.fvf;
        let decl_ptr = dev.vertex_decl();
        let (attrs_vec, stride, vdecl_hash, ff_vs_layout) = if decl_ptr.is_null() {
            let (elements, _fvf_stride) = fvf_to_elements(fvf);
            // `fvf == 0` only when the format came from SetVertexDeclaration
            // (SetFVF always carries D3DFVF_XYZ); a real declaration reads 0
            // for an omitted COLORVERTEX source, FVF falls back to material.
            let layout = convert::ff_vs_layout_from_elements(&elements, fvf == 0);
            // Pre-transformed (POSITIONT/XYZRHW) layouts bypass a bound VS —
            // D3D9 runs the FF pre-transformed path regardless, even when a
            // VS is still bound — so the attrs must resolve for the FF VS too.
            let (attrs, stride) = if bound_vertex_shader.is_null() || layout.has_rhw() {
                resolve_attrs_for_ff(&elements)
            } else {
                // SAFETY: non-null check passed; refcount holds it live.
                let vs_obj = unsafe { &*bound_vertex_shader };
                resolve_attrs_for_vs(&elements, vs_obj.input_semantics())
            };
            (attrs, stride, u64::from(fvf), layout)
        } else {
            // SAFETY: non-null check passed; refcount holds it live.
            let decl = unsafe { &*decl_ptr };
            let elements = decl.inner().elements();
            let layout = convert::ff_vs_layout_from_elements(elements, fvf == 0);
            // See the FVF arm: POSITIONT bypasses a bound VS.
            let (attrs, stride) = if bound_vertex_shader.is_null() || layout.has_rhw() {
                resolve_attrs_for_ff(elements)
            } else {
                // SAFETY: see above.
                let vs_obj = unsafe { &*bound_vertex_shader };
                resolve_attrs_for_vs(elements, vs_obj.input_semantics())
            };
            (attrs, stride, decl.inner().hash(), layout)
        };
        dev.cached_ff_vs_layout = ff_vs_layout;
        // Which VS input registers the declaration backs — folded into a
        // programmable VsSource so a shader reading an unprovided input gets a
        // distinct zero-filled variant. For the FF
        // path the value is unused.
        dev.cached_vs_provided_mask = attrs_vec.iter().fold(0u16, |m, a| {
            if a.attr_index < 16 {
                m | (1u16 << a.attr_index)
            } else {
                m
            }
        });
        Some((attrs_vec, stride, vdecl_hash))
    } else {
        None
    };

    // Now safe to take the long-lived rs borrow. Direct field access (not
    // the `render_states()` method) so the borrow lands on
    // `dev.render_states` only — field-level NLL then lets `dev.ff_state`
    // and `dev.current_frame.scratch_mut()` coexist later in this function
    // even while `rs` is still in scope.
    let rs = &dev.render_states;

    // RS snapshot. Pipeline-relevant bits go in pipeline_rs (shared
    // verbatim with PipelineSnapshot.rs); depth/scissor booleans go
    // in depth_scissor; enum RS narrow to u8.
    let render_state_value = if dirty.contains(SnapshotDirty::RS) {
        use mtld3d_core::pipeline_state::{PipelineRsBits, PipelineRsFlags};

        // D3D9 enum render-state values are spec-bounded: D3DCMP_* in
        // 1..=8, D3DBLEND_* in 1..=19, D3DBLENDOP_* in 1..=5,
        // D3DCULL_* in 1..=3, D3DCOLORWRITEENABLE_* uses 4 bits.
        let to_u8 = |v: u32| u8::try_from(v).expect("D3D9 enum render-state value ≤ u8::MAX");

        let mut prs_flags = PipelineRsFlags::empty();
        prs_flags.set(
            PipelineRsFlags::BLEND_ENABLE,
            rs[D3DRS_ALPHABLENDENABLE as usize] != 0,
        );
        prs_flags.set(
            PipelineRsFlags::SEPARATE_ALPHA_BLEND,
            rs[D3DRS_SEPARATEALPHABLENDENABLE as usize] != 0,
        );
        prs_flags.set(
            PipelineRsFlags::SRGB_WRITE,
            rs[D3DRS_SRGBWRITEENABLE as usize] != 0,
        );
        let pipeline_rs = PipelineRsBits {
            flags: prs_flags,
            src_blend: to_u8(rs[D3DRS_SRCBLEND as usize]),
            dst_blend: to_u8(rs[D3DRS_DESTBLEND as usize]),
            blend_op: to_u8(rs[D3DRS_BLENDOP as usize]),
            src_blend_alpha: to_u8(rs[D3DRS_SRCBLENDALPHA as usize]),
            dst_blend_alpha: to_u8(rs[D3DRS_DESTBLENDALPHA as usize]),
            blend_op_alpha: to_u8(rs[D3DRS_BLENDOPALPHA as usize]),
            color_write_mask: to_u8(rs[D3DRS_COLORWRITEENABLE as usize]),
        };

        let mut depth_scissor = DepthScissorFlags::empty();
        depth_scissor.set(
            DepthScissorFlags::DEPTH_ENABLE,
            rs[D3DRS_ZENABLE as usize] != 0,
        );
        depth_scissor.set(
            DepthScissorFlags::DEPTH_WRITE,
            rs[D3DRS_ZWRITEENABLE as usize] != 0,
        );
        depth_scissor.set(
            DepthScissorFlags::SCISSOR_TEST,
            rs[D3DRS_SCISSORTESTENABLE as usize] != 0,
        );

        let sr = dev.scissor_rect();
        // D3D9 caps scissor coords at MaxTextureWidth/Height (16384) — fits u16.
        let scissor_rect = [
            u16::try_from(sr[0]).expect("D3D9 scissor x ≤ 16384"),
            u16::try_from(sr[1]).expect("D3D9 scissor y ≤ 16384"),
            u16::try_from(sr[2]).expect("D3D9 scissor w ≤ 16384"),
            u16::try_from(sr[3]).expect("D3D9 scissor h ≤ 16384"),
        ];

        Some(RenderStateSnapshot {
            pipeline_rs,
            depth_scissor,
            depth_func: to_u8(rs[D3DRS_ZFUNC as usize]),
            cull_mode: to_u8(rs[D3DRS_CULLMODE as usize]),
            scissor_rect,
            blend_factor: rs[D3DRS_BLENDFACTOR as usize],
            depth_bias: rs[D3DRS_DEPTHBIAS as usize],
            slope_scale_depth_bias: rs[D3DRS_SLOPESCALEDEPTHBIAS as usize],
        })
    } else {
        None
    };

    // RT_DS: depth/stencil presence.
    let depth_stencil_value = if dirty.contains(SnapshotDirty::RT_DS) {
        let bound_ds = dev.bound_rt().depth_stencil();
        let (has_depth, has_stencil) = if !bound_ds.is_null() {
            // SAFETY: non-null check passed; refcount holds it live.
            let fmt = unsafe { (*bound_ds).standalone_format() };
            (true, depth_format_has_stencil(fmt))
        } else if dev.flags.contains(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND) {
            // App called `SetDepthStencilSurface(NULL)`: no depth attachment,
            // so the pipeline must declare no depth/stencil format.
            (false, false)
        } else {
            let h = !dev.depth_stencil_handle.is_null();
            (h, h && depth_format_has_stencil(dev.depth_stencil_format))
        };
        let mut flags = DepthStencilFlags::empty();
        flags.set(DepthStencilFlags::HAS_DEPTH, has_depth);
        flags.set(DepthStencilFlags::HAS_STENCIL, has_stencil);
        Some(flags)
    } else {
        None
    };

    // VARIANT: depends on RS + ff_vs_layout.has_rhw + depth_sampler_mask
    // (current live stage bindings).
    let variant_value = if dirty.contains(SnapshotDirty::VARIANT) {
        let mut variant = dev
            .ff_state()
            .variant_key(rs, dev.cached_ff_vs_layout.has_rhw());
        variant.depth_sampler_mask = dev.stage_bindings().depth_sampler_mask();
        variant.depth_fetch_mask = dev.stage_bindings().depth_fetch_mask();
        variant.volume_sampler_mask = dev.stage_bindings().volume_sampler_mask();
        // D3DTTFF_PROJECTED stages drive an implicit per-pixel projective divide
        // for the ps_1_0..1_3 programmable PS (the SM1 emitter consumes this; FF
        // uses its own FfPsKey mask, ps_1_4 uses DZ/DW, ps_2_0+ ignore TTFF).
        variant.tt_projected_mask = dev.ff_state().tt_projected_mask();
        if variant.depth_sampler_mask != 0 {
            mtld3d_shared::log_once_trace_by!(
                target: DEPTH_TRACE_TARGET,
                key: u64::from(variant.depth_sampler_mask),
                "depth: sampler_mask={:#x} (slots bound to depth-format textures)",
                variant.depth_sampler_mask
            );
        }
        Some(variant)
    } else {
        None
    };

    let bound_vertex_shader = dev.shader_bindings().vertex_shader();
    let bound_pixel_shader = dev.shader_bindings().pixel_shader();
    let bound_mask = dev.cached_bound_texture_mask;

    // VS_SOURCE. A pre-transformed (POSITIONT/XYZRHW) layout bypasses a bound
    // VS: D3D9 runs the FF pre-transformed path regardless of the binding,
    // even when a VS is still bound. The PS side is NOT bypassed — a bound PS
    // still runs.
    let vs_value = if dirty.contains(SnapshotDirty::VS_SOURCE) {
        if bound_vertex_shader.is_null() || dev.cached_ff_vs_layout.has_rhw() {
            let key = dev
                .ff_state()
                .build_vs_key(rs, dev.cached_ff_vs_layout, bound_mask);
            mtld3d_shared::crumb!(
                "ffvs:cap",
                dev.current_seq(),
                u64::from(key.tex_coord_count),
            );
            let max_row_count = dev.ff_state().ff_vs_row_count(&key);
            Some(VsSource::FixedFunction { key, max_row_count })
        } else {
            // SAFETY: non-null check; refcount holds it live.
            let vs_obj = unsafe { &*bound_vertex_shader };
            Some(VsSource::Programmable {
                vs_id: vs_obj.shader_id(),
                max_const_used: clamp_const_rows(vs_obj.max_const_used()),
                uses_rel_const: vs_obj.uses_rel_const(),
                provided_input_mask: dev.cached_vs_provided_mask,
                uses_int_const: vs_obj.uses_int_const(),
            })
        }
    } else {
        None
    };

    // PS_SOURCE.
    let ps_value = if dirty.contains(SnapshotDirty::PS_SOURCE) {
        if bound_pixel_shader.is_null() {
            let key = dev.ff_state().build_ps_key(rs, bound_mask);
            Some(PsSource::FixedFunction { key })
        } else {
            // SAFETY: non-null check; refcount holds it live.
            let ps_obj = unsafe { &*bound_pixel_shader };
            Some(PsSource::Programmable {
                ps_id: ps_obj.shader_id(),
                max_const_used: clamp_const_rows(ps_obj.max_const_used()),
                uses_bump_env: ps_obj.uses_bump_env(),
            })
        }
    } else {
        None
    };

    drop(keys_timer);

    // From here through `drop(consts_timer)` below is the "consts"
    // measurement scope — VS/PS const source build + alpha/fog Vec
    // build + the corresponding 4 scratch bumps. The scope bills to
    // one of two sibling counters chosen per-draw:
    //   c_ff = at least one shader stage is FF
    //   c_pr = both VS and PS are programmable
    // Lets the perf summary attribute residual consts cost between
    // the two classes.
    let any_ff = bound_vertex_shader.is_null() || bound_pixel_shader.is_null();
    let consts_timer = if any_ff {
        CycleAddTimer::start(draw_snapshot_c_ff_ptr(DeviceInner::perf_ptr_of(obj.inner)))
    } else {
        CycleAddTimer::start(draw_snapshot_c_pr_ptr(DeviceInner::perf_ptr_of(obj.inner)))
    };

    // VS_CONST source (Phase 1 — owned/borrowed before scratch borrow).
    //
    // FF VS uses an encoder-side mirror parallel to the programmable
    // path: when any `FfVsDirty` section changed since the last
    // snapshot, rebuild the blob, push as `Op::SetFfVsConstRange`,
    // and clear the dirty mask. The mirror persists across frames; the
    // encoder snapshots `max_row + 1` rows from it into per-frame
    // scratch at `emit_draw` time via `ff_vs_const_scratch`.
    //
    // The cache-hit case (consecutive draws with no FF state changes
    // between them) returns no `vs_const_src` here and emits no delta
    // op — `emit_draw` reuses the previously-bumped scratch slice.
    if dirty.contains(SnapshotDirty::VS_CONST)
        && (bound_vertex_shader.is_null() || dev.cached_ff_vs_layout.has_rhw())
    {
        // FF VS const builder needs the FF key — pull from
        // newly-rebuilt or cached vs. Both halves borrow; we
        // `.clone()` the FF key once on the borrowed-from-cache
        // path because `VsSource` is not Copy and the FF
        // section helpers take `&FfVsKey`. The cache fallback derefs
        // the scratch `VsSourcePtr` lazily (`or_else`): it's only
        // reached when VS_SOURCE wasn't dirty this draw, which never
        // happens on the first draw of a frame (`all()`), so the
        // cached pointer is always current-frame valid before deref.
        let key_ref = vs_value
            .as_ref()
            .or_else(|| dev.snapshot_cache.vs.as_ref().map(VsSourcePtr::as_ref));
        let key = match key_ref {
            Some(VsSource::FixedFunction { key, .. }) => key.clone(),
            _ => dev
                .ff_state()
                .build_vs_key(rs, dev.cached_ff_vs_layout, bound_mask),
        };
        let ff_dirty = dev.ff_state.take_ff_vs_dirty();
        if !ff_dirty.is_empty() {
            // Each set bit emits one `Op::SetFfVsConstRange` for its
            // owning section. The encoder mirror persists across
            // frames; rows untouched by a given emit retain their
            // previously-written values, so unchanged sections don't
            // need to re-bump.
            //
            // SAFETY contract for every section helper below: the
            // returned `*mut u8` points into the per-frame scratch
            // arena and stays alive until end-of-frame. `ScratchSlice`
            // wraps it; the encoder copies the bytes into
            // `ff_vs_constants_mirror` at `apply_ff_vs_const_range`
            // time. Per-draw isolation is preserved by
            // `ff_vs_const_scratch` bumping a fresh slice from the
            // mirror after every apply.
            let push_section =
                |frame: &mut crate::encoder::FrameData, start_row: u16, rows: u16, ptr: *mut u8| {
                    let nn = NonNull::new(ptr).expect("ScratchArena alloc returned non-null");
                    let byte_len = u32::from(rows) * 16;
                    let data = ScratchSlice::from_raw_parts(nn, byte_len);
                    frame.push_op_inline(crate::encoder::Op::SetFfVsConstRange {
                        start_row,
                        rows,
                        data,
                    });
                };

            if key.has_rhw() {
                // XYZRHW: only row 0 (viewport) matters. Other sections
                // are never read by the shader on this path; their
                // dirty bits, if set, are absorbed without emit since
                // `take_ff_vs_dirty` already cleared the mask.
                let v = dev.viewport();
                let to_f32 = |n: u32| {
                    f32::from(u16::try_from(n).expect("D3D9 viewport dim ≤ 16384 fits u16"))
                };
                let viewport = (to_f32(v.x), to_f32(v.y), to_f32(v.width), to_f32(v.height));
                let ptr = FfState::build_xyzrhw_row(viewport, dev.current_frame.scratch_mut());
                push_section(&mut dev.current_frame, 0, 1, ptr);
            } else {
                if ff_dirty.contains(FfVsDirty::WV) {
                    let (s, r, p) = dev
                        .ff_state
                        .build_wv_section(dev.current_frame.scratch_mut());
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::PROJ) {
                    let (s, r, p) = dev
                        .ff_state
                        .build_proj_section(dev.current_frame.scratch_mut());
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::FOG) {
                    let (s, r, p) = FfState::build_fog_section(
                        rs,
                        key.fog_mode,
                        dev.current_frame.scratch_mut(),
                    );
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::AMBIENT) {
                    let (s, r, p) =
                        FfState::build_ambient_section(rs, dev.current_frame.scratch_mut());
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::MATERIAL) {
                    let (s, r, p) = dev
                        .ff_state
                        .build_material_section(&key, dev.current_frame.scratch_mut());
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::LIGHTS)
                    && let Some((s, r, p)) = dev
                        .ff_state
                        .build_lights_section(&key, dev.current_frame.scratch_mut())
                {
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::TT)
                    && let Some((s, r, p)) = dev
                        .ff_state
                        .build_tt_section(dev.current_frame.scratch_mut())
                {
                    push_section(&mut dev.current_frame, s, r, p);
                }
                if ff_dirty.contains(FfVsDirty::PALETTE)
                    && let Some((s, r, p)) = dev
                        .ff_state
                        .build_palette_section(&key, dev.current_frame.scratch_mut())
                {
                    push_section(&mut dev.current_frame, s, r, p);
                }
            }
        }
        // FF VS const bytes are NOT carried in the snapshot cache
        // anymore — the encoder mirror is the source of truth, and
        // `emit_draw` snapshots from it via `enc.ff_vs_const_scratch`.
    }

    // PS_CONST source. Same routing as VS_CONST: programmable PS uses
    // the encoder mirror; only FF runs here.
    let ps_const_src = if dirty.contains(SnapshotDirty::PS_CONST) && bound_pixel_shader.is_null() {
        let vec = dev.ff_state().build_ps_constants(rs);
        Some(ConstSource::Owned(vec))
    } else {
        None
    };

    // ALPHA_REF + FOG_COLOR bytes (variant must be current).
    let alpha_ref_vec = if dirty.contains(SnapshotDirty::ALPHA_REF) {
        let variant = variant_value
            .or(dev.snapshot_cache.variant)
            .unwrap_or_default();
        Some(build_alpha_ref_bytes(variant, dev.ff_state().alpha_ref(rs)))
    } else {
        None
    };
    let fog_color_vec = if dirty.contains(SnapshotDirty::FOG_COLOR) {
        let variant = variant_value
            .or(dev.snapshot_cache.variant)
            .unwrap_or_default();
        Some(mtld3d_core::ff_state::build_fog_color_bytes(rs, variant))
    } else {
        None
    };
    // Bump-environment matrix bytes (PS slot 12). Built only when a bump TSS
    // state changed (rare); the slot is bound at draw time only for a PS that
    // actually uses texbem/texbeml/bem.
    let bump_env_vec = if dirty.contains(SnapshotDirty::BUMP_ENV) {
        Some(dev.ff_state().build_bump_env_bytes())
    } else {
        None
    };
    // VS integer-constant bytes (vertex slot 14). Built when an integer
    // constant changed (rare) or on the first draw of a frame (all-dirty); the
    // slot is bound at draw time only for a VS that reads a dynamic integer
    // constant. Mirrors the bump-env capture above.
    let vs_int_const_vec = if dirty.contains(SnapshotDirty::VS_CONST_I) {
        Some(dev.shader_bindings().vs_constants_i_bytes())
    } else {
        None
    };

    // Phase 2: take scratch + bump dirty pieces + update cache. SAFETY
    // (`Borrowed` ConstSource arms): the (ptr, len) pairs alias
    // `dev.shader_bindings.{vs,ps}_constants[]`, which
    // emit_snapshot_deltas (synchronous API thread) does not allow
    // any `Set*ShaderConstantF` to mutate between source read and
    // copy. Direct field access on `dev.current_frame.scratch` splits
    // the borrow off `dev.snapshot_cache`, letting both be
    // mutated/read in turn.
    let scratch = dev.current_frame.scratch_mut();

    // ── consts_timer SCOPE: VS/PS const + alpha/fog bumps. Matches
    //    baseline `snapshot_shared` scoping so the summary's `consts`
    //    row stays comparable. RS/stages/attrs
    //    bumps + wrapper bump + scalar cache updates fall into
    //    "other" (snapshot total - stages - consts).
    //
    // FF VS const bytes flow through the encoder's
    // `ff_vs_constants_mirror`; the API-side `snapshot_cache.vs_constants`
    // is not consulted for FF (or programmable — that's mirror-only too).
    // Clear the cached pointer so a stale cached entry doesn't leak.
    if dirty.contains(SnapshotDirty::VS_CONST) {
        dev.snapshot_cache.vs_constants = None;
    }
    if dirty.contains(SnapshotDirty::PS_CONST) {
        dev.snapshot_cache.ps_constants = ps_const_src.map(|src| src.alloc_into(scratch));
    }
    if let Some(v) = alpha_ref_vec {
        dev.snapshot_cache.alpha_ref_bytes = Some(arena_alloc_bytes(scratch, &v));
    }
    if let Some(v) = fog_color_vec {
        dev.snapshot_cache.fog_color_bytes = Some(arena_alloc_bytes(scratch, &v));
    }
    if let Some(v) = bump_env_vec {
        dev.snapshot_cache.bump_env_bytes = Some(arena_alloc_bytes(scratch, &v));
    }
    if let Some(v) = vs_int_const_vec {
        dev.snapshot_cache.vs_int_const_bytes = Some(arena_alloc_bytes(scratch, &v));
    }
    drop(consts_timer);
    // ── END consts_timer SCOPE ──

    // `bumps_timer` wraps the remaining phase-2 work: RS / stage_bindings
    // / attrs scratch bumps, scalar cache assignments, and the
    // snapshot-wrapper bump. Closes the prior "other" residual so the
    // sum stages + consts + keys + bumps ≈ snapshot total.
    let bumps_timer =
        CycleAddTimer::start(draw_snapshot_bumps_ptr(DeviceInner::perf_ptr_of(obj.inner)));
    // Non-const bumps + scalar cache updates.
    if let Some(rs_val) = render_state_value {
        // SAFETY: RenderStateSnapshot fields are all primitives /
        // small Copy types with trivial Drop; bytewise scratch copy
        // is sound.
        let ptr = NonNull::new(unsafe { scratch.alloc_from(&rs_val) })
            .expect("ScratchArena returned non-null");
        dev.snapshot_cache.render_state = Some(RenderStatePtr(ptr));
    }
    if let Some((arr, packed_mask)) = stage_bindings_arr_opt {
        // SAFETY: StageBinding fields are TextureInfo (integer / enum
        // / bitflag primitives) + sampler_state [u32; N] — trivial
        // Drop, bytewise scratch copy is sound (same contract as the
        // prior flat-array bump). The packed form only memcpys the
        // bound slots, collapsing the per-draw bump from ~2 KB to
        // ~120-360 B for typical WoW workloads.
        let ptr = unsafe { bump_packed_stage_bindings(scratch, packed_mask, &arr) };
        dev.snapshot_cache.stage_bindings = Some(ptr);
    }
    if let Some((attrs_vec, stride, vdecl_hash)) = vdecl_value {
        let (raw_ptr, len) = scratch.alloc_slice(&attrs_vec);
        let ptr = NonNull::new(raw_ptr).expect("ScratchArena alloc_slice returned non-null");
        dev.snapshot_cache.attrs = Some(AttrSnapshot {
            ptr,
            len,
            stride,
            vdecl_hash,
        });
    }
    if let Some(v) = vs_value {
        // Bump the VS source behind a scratch pointer so the per-draw
        // wrapper memcpy doesn't carry the embedded FfVsKey — done only
        // when VS_SOURCE is dirty (rare post-gating).
        // SAFETY: VsSource is trivial-Drop (FfVsKey + scalars), so the
        // bytewise scratch copy is sound; the pointer lives in the
        // current frame's scratch, consumed by emit_draw before reset.
        let raw = unsafe { scratch.alloc_from(&v) };
        let ptr = NonNull::new(raw).expect("ScratchArena returned non-null");
        dev.snapshot_cache.vs = Some(VsSourcePtr(ptr));
    }
    if let Some(p) = ps_value {
        // SAFETY: PsSource is trivial-Drop (FfPsKey + scalars); same
        // lifetime contract as the VS bump above.
        let raw = unsafe { scratch.alloc_from(&p) };
        let ptr = NonNull::new(raw).expect("ScratchArena returned non-null");
        dev.snapshot_cache.ps = Some(PsSourcePtr(ptr));
    }
    if let Some(v) = variant_value {
        dev.snapshot_cache.variant = Some(v);
    }
    if let Some(ds) = depth_stencil_value {
        dev.snapshot_cache.depth_stencil = ds;
    }

    // Cache is now the assembled snapshot — memcpy it once into
    // scratch as the wrapper for the Op. SAFETY: CurrentSnapshot
    // fields are all `Copy` with trivial Drop (Option<NonNull>,
    // scalar, enum) so the bit-identical scratch copy never needs
    // its own drop run.
    let snap_ptr = unsafe { scratch.alloc_from(&dev.snapshot_cache) };

    let snap_nn = NonNull::new(snap_ptr).expect("ScratchArena returned non-null");
    dev.push_op_inline(Op::SetCurrentSnapshot(CurrentSnapshotPtr(snap_nn)));
    dev.snapshot_dirty = SnapshotDirty::empty();
    drop(bumps_timer);
}

/// Capture bound textures as `TextureInfo` snapshots plus the bound-texture bitmask.
///
/// Uploads are handled independently of draw-time
/// binding capture: `texture_unlock_rect` / `texture_add_dirty_rect` push
/// their own upload closures onto the current frame, so this function no
/// longer touches staging bytes.
fn snapshot_stage_bindings(
    dev: &mut DeviceInner,
) -> ([Option<StageBinding>; STAGE_COUNT], u8, u16) {
    let mut stage_bindings: [Option<StageBinding>; STAGE_COUNT] = core::array::from_fn(|_| None);
    let mut bound_texture_mask: u8 = 0;
    let mut packed_mask: u16 = 0;
    for (stage, slot) in stage_bindings.iter_mut().enumerate() {
        let tex_ptr = dev.stage_bindings().texture(stage);
        if tex_ptr.is_null() {
            continue;
        }
        packed_mask |= 1u16 << stage;
        // FF combiner only consumes stages 0–7 (MaxTextureBlendStages = 8);
        // stages 8–15 are programmable-PS-only and would shift past the
        // u8 mask width.
        if let Ok(stage_u8) = u8::try_from(stage)
            && stage_u8 < 8
        {
            bound_texture_mask |= 1u8 << stage_u8;
        }
        let mut sampler_state = dev.stage_bindings().sampler_states(stage);
        // Lazy texture upload: flush any per-mip `dirty` flags before
        // capturing TextureInfo. Closures pushed by `schedule_upload`
        // precede the Draw closure on the encoder thread, so the
        // upload runs before the bind reads. The `&mut
        // Direct3DTexture9` here doesn't alias `dev: &mut DeviceInner`
        // because the Texture is a separate Box —
        // `dev.stage_bindings.textures[stage]` is a raw `*mut`, not a
        // tracked reference.
        // SAFETY: `tex_ptr` is the bound-texture pointer from
        // `stage_bindings` (caller checks non-null above); the texture
        // is held alive by stage-bindings refcount until rebound.
        let tex = unsafe { &mut *tex_ptr };
        // Cross-device migration handler — must run before flush_dirty_mips
        // so the re-marked dirty bits drive an upload against the new
        // device's encoder + handles.
        crate::texture::rehydrate_for_device(tex.inner_mut(), dev);
        crate::texture::flush_dirty_mips(tex.inner_mut(), dev);
        // A texture's SetLOD raises the effective most-detailed mip. LOD == 0
        // (the common case) is a no-op in both branches.
        let lod = tex.inner().lod();
        if sampler_state[D3DSAMP_MIPFILTER as usize] == D3DTEXF_NONE {
            // mip-OFF: the effective level is the texture LOD alone (MAXMIPLEVEL
            // does not apply). Metal samples level 0 for a non-mipmapped sampler
            // and ignores lodMinClamp, so promote to POINT with MAXMIPLEVEL = LOD
            // — the clamp then pins sampling to the LOD level.
            if lod > 0 {
                sampler_state[D3DSAMP_MIPFILTER as usize] = D3DTEXF_POINT;
                sampler_state[D3DSAMP_MAXMIPLEVEL as usize] = lod;
            }
        } else {
            // mip-ON: the sampler clamps to max(MAXMIPLEVEL, LOD); fold the LOD
            // into MAXMIPLEVEL so the cached sampler's lodMinClamp honours it.
            let max_mip = sampler_state[D3DSAMP_MAXMIPLEVEL as usize];
            sampler_state[D3DSAMP_MAXMIPLEVEL as usize] = max_mip.max(lod);
        }
        *slot = Some(StageBinding {
            texture_id: tex.texture_id(),
            sampler_state,
        });
        // Diag probe: when a depth-format texture lands on a sampler
        // slot the emitter treats it as `depth2d<float>` and emits
        // `sample_compare`. Log its id + D3D / Metal format so a wrong
        // (non-depth) texture on a depth slot, or the
        // D24S8 → Depth32FloatStencil8 promotion, is visible. Once per
        // (stage, texture_id, d3d_format); zero-cost when
        // `mtld3d::d3d9::depth=trace` isn't enabled.
        if tex.is_depth_format() {
            mtld3d_shared::log_once_trace_by!(
                target: DEPTH_TRACE_TARGET,
                key: ((stage as u64) << 56)
                    ^ (tex.texture_id().raw() << 16)
                    ^ u64::from(tex.d3d_format()),
                "depth: slot {} tex={:#x} d3d_fmt={:#x} metal_fmt={:?}",
                stage,
                tex.texture_id(),
                tex.d3d_format(),
                tex.metal_pixel_format()
            );
        }
    }
    (stage_bindings, bound_texture_mask, packed_mask)
}

extern "system" fn device_draw_indexed_primitive_up(
    this: *mut c_void,
    primitive_type: u32,
    min_vertex_index: u32,
    num_vertices: u32,
    primitive_count: u32,
    index_data: *const c_void,
    index_format: u32,
    vertex_data: *const c_void,
    vertex_stride: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return D3DERR_INVALIDCALL;
    }
    if index_data.is_null() || vertex_data.is_null() || primitive_count == 0 {
        return D3DERR_INVALIDCALL;
    }
    let (index_type, index_size): (mtld3d_shared::mtl::IndexType, usize) = match index_format {
        D3DFMT_INDEX16 => (mtld3d_shared::mtl::IndexType::UInt16, 2),
        D3DFMT_INDEX32 => (mtld3d_shared::mtl::IndexType::UInt32, 4),
        other => {
            warn!(
                target: LOG_TARGET,
                "DrawIndexedPrimitiveUP: unsupported index format {other}"
            );
            return D3DERR_INVALIDCALL;
        }
    };

    // Upload enough vertices to cover the indexed range. The inline indices are
    // absolute (base vertex 0), so copy `[0, min_vertex_index + num_vertices)`
    // straight from the user pointer; vertices below `min_vertex_index` are
    // uploaded but unreferenced.
    let vtx_upload = min_vertex_index as usize + num_vertices as usize;
    let vtx_bytes = vtx_upload * vertex_stride as usize;
    let mut vertex_copy = Vec::<u8>::with_capacity(vtx_bytes);
    // SAFETY: per the D3D9 ABI `vertex_data` covers at least
    // `(min_vertex_index + num_vertices) * vertex_stride` bytes; `vertex_copy`
    // was just allocated with matching capacity.
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertex_data.cast::<u8>(),
            vertex_copy.as_mut_ptr(),
            vtx_bytes,
        );
    }
    // SAFETY: the leading `vtx_bytes` were just initialised by the copy above.
    unsafe { vertex_copy.set_len(vtx_bytes) };

    // Build the index stream. Triangle fan has no Metal primitive, so expand the
    // inline index list into a triangle list (indices 0, i+1, i+2) — the same
    // fan expansion the vertex path uses, applied here over the index stride.
    let (metal_prim, index_bytes, index_count) = if primitive_type == D3DPT_TRIANGLEFAN {
        let src_bytes = (primitive_count as usize + 2) * index_size;
        // SAFETY: per the D3D9 ABI `index_data` covers at least
        // `(primitive_count + 2)` indices of `index_size` bytes.
        let src = unsafe { core::slice::from_raw_parts(index_data.cast::<u8>(), src_bytes) };
        let expanded = convert::expand_triangle_fan(src, index_size, primitive_count);
        (
            d3d_to_metal_primitive(D3DPT_TRIANGLELIST).expect("triangle list is supported"),
            expanded,
            primitive_count * 3,
        )
    } else {
        let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
            return D3DERR_INVALIDCALL;
        };
        let index_count = vertex_count(primitive_type, primitive_count);
        if index_count == 0 {
            return D3DERR_INVALIDCALL;
        }
        let idx_bytes = index_count as usize * index_size;
        let mut index_copy = Vec::<u8>::with_capacity(idx_bytes);
        // SAFETY: per the D3D9 ABI `index_data` covers `index_count` indices of
        // `index_size` bytes; `index_copy` was just allocated to match.
        unsafe {
            core::ptr::copy_nonoverlapping(
                index_data.cast::<u8>(),
                index_copy.as_mut_ptr(),
                idx_bytes,
            );
        }
        // SAFETY: the leading `idx_bytes` were just initialised by the copy above.
        unsafe { index_copy.set_len(idx_bytes) };
        (metal_prim, index_copy, index_count)
    };

    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    dev.push_op_inline(Op::Draw(DrawOp {
        metal_prim,
        vertex_source: VertexSource::Up {
            bytes: vertex_copy,
            size: u32::try_from(vtx_bytes).expect("DrawIndexedPrimitiveUP vertex size fits u32"),
            stride: vertex_stride,
        },
        index_source: IndexSource::Up {
            bytes: index_bytes,
            index_count,
            index_type,
        },
    }));
    // D3D9 resets stream source 0 to (NULL, 0, 0) AND the index buffer to NULL
    // after a successful DrawIndexedPrimitiveUP.
    let bound = dev.bound_buffers_mut();
    bound.reset_stream0();
    bound.replace_index_buffer(core::ptr::null_mut());
    D3D_OK
}

extern "system" fn device_process_vertices(
    this: *mut c_void,
    _src_start: u32,
    _dst_index: u32,
    _count: u32,
    _dst_buffer: *mut c_void,
    _decl: *mut c_void,
    _flags: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::ProcessVertices → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_create_vertex_declaration(
    this: *mut c_void,
    elements: *const c_void,
    decl: *mut *mut c_void,
) -> i32 {
    const MAX_ELEMENTS: usize = 64; // generous; D3D9 limit is MAXD3DDECLLENGTH=64

    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if elements.is_null() || decl.is_null() {
        null_out(decl);
        return D3DERR_INVALIDCALL;
    }
    // Read the element array up to D3DDECL_END (stream==0xFF). We don't
    // know the length up front, so walk in place until the terminator.
    let mut len = 0usize;
    loop {
        if len >= MAX_ELEMENTS {
            warn!(target: LOG_TARGET, "CreateVertexDeclaration: no terminator within {MAX_ELEMENTS} elements");
            null_out(decl);
            return D3DERR_INVALIDCALL;
        }
        // SAFETY: `elements + len * size_of::<D3DVERTEXELEMENT9>()` stays within
        // the caller-provided array; the `MAX_ELEMENTS` bound above guards `len`.
        let e_ptr = unsafe { elements.cast::<mtld3d_types::D3DVERTEXELEMENT9>().add(len) };
        // SAFETY: `e_ptr` is a valid, aligned `D3DVERTEXELEMENT9` pointer.
        let e = unsafe { *e_ptr };
        len += 1;
        if e.stream == mtld3d_types::D3DDECL_END_STREAM {
            break;
        }
        // Real-element validation. D3D9 rejects these with E_FAIL (distinct
        // from the INVALIDCALL used for structural problems): element offsets
        // must be DWORD-aligned, and D3DDECLTYPE_UNUSED is only legal in the
        // D3DDECL_END terminator.
        if e.offset % 4 != 0 || e.type_ == mtld3d_types::D3DDECLTYPE_UNUSED {
            null_out(decl);
            return E_FAIL;
        }
    }
    // SAFETY: `elements` is the caller-supplied decl array; the walk
    // above advanced `len` exactly to the `D3DDECL_END` terminator, so
    // `len` `D3DVERTEXELEMENT9` entries are readable.
    let slice = unsafe {
        core::slice::from_raw_parts(elements.cast::<mtld3d_types::D3DVERTEXELEMENT9>(), len)
    };
    // Trace-only probe — surface the decl shape under
    // `RUST_LOG=mtld3d::d3d9::state=trace` for bring-up enumeration. Not a
    // warn: FF VS consumes BLENDWEIGHT/BLENDINDICES correctly when
    // D3DRS_VERTEXBLEND opts in. WoW (and any game doing CPU skinning) just
    // leaves D3DRS_VERTEXBLEND at D3DVBF_DISABLE, in which case these decl
    // elements are declared but unused — neither a bug nor a warn-worthy
    // event.
    if mtld3d_core::state_trace::enabled() {
        for e in slice.iter().take(len.saturating_sub(1)) {
            if e.usage == mtld3d_types::D3DDECLUSAGE_BLENDWEIGHT
                || e.usage == mtld3d_types::D3DDECLUSAGE_BLENDINDICES
            {
                let usage = e.usage;
                let ty = e.type_;
                let stream = e.stream;
                let offset = e.offset;
                log::trace!(
                    target: mtld3d_core::state_trace::TARGET,
                    "vertex decl declares D3DDECLUSAGE_{usage} (type={ty}, stream={stream}, offset={offset})"
                );
            }
        }
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    Direct3DVertexDeclaration9::new(&VertexDeclCreateInfo {
        device_inner: obj.inner_ptr(),
        elements: slice,
    })
    .map_or_else(
        || {
            null_out(decl);
            D3DERR_INVALIDCALL
        },
        |obj| {
            // SAFETY: vtable out-param; `decl` is *mut *mut c_void per IDirect3DDevice9 ABI.
            let decl_ptr = Box::into_raw(Box::new(obj));
            // SAFETY: `decl_ptr` is a freshly created, live declaration at refcount 1.
            unsafe { crate::com_ref::com_register_child(decl_ptr) };
            // SAFETY: vtable out-param; `decl` is *mut *mut c_void per the ABI.
            unsafe { OutPtr::write_opt(decl, decl_ptr.cast::<c_void>()) };
            D3D_OK
        },
    )
}

extern "system" fn device_set_vertex_declaration(this: *mut c_void, decl: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let new = decl.cast::<Direct3DVertexDeclaration9>();
    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `new` is null or a *mut Direct3DVertexDeclaration9 supplied
        // by the calling game via SetVertexDeclaration.
        let adopted = unsafe { CachedComPtr::adopt(new) };
        rec.record(StateOp::VertexDeclaration(adopted));
        return D3D_OK;
    }
    // Redundant-set elimination: re-binding the same decl pointer
    // re-resolves to a byte-identical attrs slice (+ FfVsLayout), so
    // skip the expensive VDECL rebuild. VS_SOURCE/VS_CONST only matter
    // if FF VS bound (FF VS key reads ff_vs_layout).
    let changed = dev.replace_vertex_decl(new);
    // An explicitly-set declaration carries no FVF: GetFVF reports 0 until the
    // next SetFVF re-establishes one. Mirrors the D3D9 runtime resetting the
    // effective FVF when a declaration is bound directly.
    dev.fvf = 0;
    if changed {
        // VS_SOURCE is marked unconditionally (not via `ff_aware_mask`, which
        // drops it for a programmable VS): a declaration change alters which VS
        // input registers are provided (`provided_input_mask`), so the
        // programmable `VsSource` must rebuild to pick up the new mask.
        let mut mask = SnapshotDirty::VDECL
            | SnapshotDirty::VS_SOURCE
            | dev.ff_aware_mask(SnapshotDirty::VS_CONST);
        // Pre-transformed (POSITIONT) declarations bypass a bound VS, and
        // `cached_ff_vs_layout` (which `ff_aware_mask` consults) lags until
        // the next snapshot — when RHW-ness may flip, dirty the FF VS consts
        // (xyzrhw viewport row) and the variant (`fog_mode` keys on RHW)
        // unconditionally.
        let new_rhw = !new.is_null()
            // SAFETY: non-null checked; the slot adopted a ref above.
            && convert::ff_vs_layout_from_elements(unsafe { (*new).inner().elements() }, true)
                .has_rhw();
        if new_rhw || dev.cached_ff_vs_layout.has_rhw() {
            mask |= SnapshotDirty::VS_CONST | SnapshotDirty::VARIANT;
        }
        dev.mark_snapshot_dirty(mask);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetVertexDecl, !changed);
    D3D_OK
}

extern "system" fn device_get_vertex_declaration(this: *mut c_void, decl: *mut *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    if decl.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let current = obj.inner().vertex_decl();
    if !current.is_null() {
        // SAFETY: `current` is non-null (checked above) and points to a
        // live `Direct3DVertexDeclaration9` whose refcount keeps it
        // alive while bound on the device.
        let wrapper = unsafe { &*current };
        let vtbl = wrapper.vtbl();
        // SAFETY: calling the just-loaded `add_ref` thunk through `vtbl`
        // with the wrapper pointer as `this`; D3D9 mandates AddRef on
        // out-pointer returns from getter thunks.
        unsafe { (vtbl.add_ref)(current.cast::<c_void>()) };
    }
    // SAFETY: `decl` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *decl = current.cast::<c_void>() };
    D3D_OK
}

extern "system" fn device_set_fvf(this: *mut c_void, fvf: u32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::Fvf(fvf));
        return D3D_OK;
    }
    // SetFVF(0) is not a valid FVF; the driver treats it as a no-op, leaving
    // the current declaration (explicit, or a prior implicit one) bound.
    if fvf == 0 {
        dev.perf_mut().record_keys_gate(KeysGate::SetFvf, true);
        return 0;
    }
    // A non-zero FVF binds its implicit declaration so GetVertexDeclaration
    // returns it and the draw path resolves the same layout it would from the
    // FVF directly. Redundant-set elimination: re-binding the same cached decl
    // changes nothing, so skip the VDECL rebuild.
    let changed = dev.bind_fvf_decl(fvf);
    if changed {
        // VS_SOURCE is marked unconditionally (not via `ff_aware_mask`, which
        // drops it for a programmable VS): a declaration change alters which VS
        // input registers are provided (`provided_input_mask`), so the
        // programmable `VsSource` must rebuild to pick up the new mask.
        let mut mask = SnapshotDirty::VDECL
            | SnapshotDirty::VS_SOURCE
            | dev.ff_aware_mask(SnapshotDirty::VS_CONST);
        // XYZRHW bypasses a bound VS, and `cached_ff_vs_layout` (which
        // `ff_aware_mask` consults) lags until the next snapshot — when
        // RHW-ness may flip, dirty the FF VS consts (xyzrhw viewport row)
        // and the variant (`fog_mode` keys on RHW) unconditionally.
        let new_rhw = (fvf & mtld3d_types::D3DFVF_POSITION_MASK) == mtld3d_types::D3DFVF_XYZRHW;
        if new_rhw || dev.cached_ff_vs_layout.has_rhw() {
            mask |= SnapshotDirty::VS_CONST | SnapshotDirty::VARIANT;
        }
        dev.mark_snapshot_dirty(mask);
    }
    dev.perf_mut().record_keys_gate(KeysGate::SetFvf, !changed);
    0 // S_OK
}

extern "system" fn device_get_fvf(this: *mut c_void, fvf: *mut u32) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    if fvf.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `fvf` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `u32` slot owned by the caller.
    unsafe { *fvf = obj.inner().fvf };
    0 // S_OK
}

/// Read a DXSO token stream from a caller-supplied pointer until the End opcode (0x0000FFFF).
///
/// Handles comment payloads and instruction token
/// counts so payload bytes that happen to equal 0xFFFF aren't misread as End.
fn read_shader_bytecode(ptr: *const u32) -> Option<Vec<u32>> {
    const MAX_TOKENS: usize = 65536;
    let mut len = 1; // version token
    loop {
        if len >= MAX_TOKENS {
            return None;
        }
        // SAFETY: `ptr + len` stays within the caller-provided token stream;
        // the `MAX_TOKENS` bound above guards `len`.
        let tok_ptr = unsafe { ptr.add(len) };
        // SAFETY: `tok_ptr` is a valid, aligned `u32` pointer.
        let tok = unsafe { *tok_ptr };
        len += 1;
        let opcode = (tok & 0xFFFF) as u16;
        if opcode == 0xFFFF {
            break;
        }
        if opcode == 0xFFFE {
            let payload = ((tok >> 16) & 0x7FFF) as usize;
            len += payload;
        } else {
            let count = ((tok >> 24) & 0xF) as usize;
            len += count;
        }
    }
    // SAFETY: `ptr` is the caller-supplied DXSO token stream; the
    // walk above advanced `len` exactly to the End opcode position,
    // so `len` u32 tokens are readable from `ptr`.
    let bc = unsafe { core::slice::from_raw_parts(ptr, len) };
    Some(bc.to_vec())
}

/// Optional on-disk dump of DXSO bytecode.
///
/// Gated by the `debug.bytecodeDumpDir = <dir>` key in `mtld3d.conf`. Writes
/// `{dir}/{prefix}_{id:x}.dxso` as raw little-endian `u32` tokens (no
/// framing) the first time a shader with that id is seen; subsequent
/// calls with the same id are no-ops. Failures log once and don't
/// abort the caller — this is a forensic shader-capture probe, not a
/// correctness path.
fn maybe_dump_bytecode(prefix: &str, shader_id: ProgramId, bytecode: &[u32]) {
    let dir = &crate::config::CONFIG.bytecode_dump_dir;
    if dir.is_empty() {
        return;
    }
    let path = std::path::PathBuf::from(dir).join(format!("{prefix}_{shader_id:x}.dxso"));
    if path.exists() {
        return;
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "debug.bytecodeDumpDir: create_dir_all({dir}) failed: {e}"
        );
        return;
    }
    let mut bytes = Vec::with_capacity(bytecode.len() * 4);
    for &t in bytecode {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&path, &bytes) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "debug.bytecodeDumpDir: write({}) failed: {e}",
            path.display()
        );
    }
}

extern "system" fn device_create_vertex_shader(
    this: *mut c_void,
    function: *const u32,
    shader: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Null `*ppShader` before any failure return — see `device_create_pixel_shader`:
    // callers (the conformance suite, real apps) ignore the HRESULT and bind the
    // out-param, so an uninitialised slot becomes a wild shader pointer.
    null_out(shader);
    if function.is_null() || shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let Some(bytecode) = read_shader_bytecode(function) else {
        return D3DERR_INVALIDCALL;
    };
    // Dump before parse so unsupported-shader-model attempts (e.g. SM3
    // bytecode under SM2 caps) still land in `debug.bytecodeDumpDir` for
    // offline analysis. `ProgramId::from_tokens` is content-derived, so
    // the id is stable without a successful parse.
    let shader_id = ProgramId::from_tokens(&bytecode);
    maybe_dump_bytecode("vs", shader_id, &bytecode);
    let program = match mtld3d_core::dxso::parse(&bytecode) {
        Ok(p) => p,
        Err(e) => {
            warn!(target: LOG_TARGET, "CreateVertexShader parse failed: {e:?}");
            return D3DERR_INVALIDCALL;
        }
    };
    if program.shader_type != mtld3d_core::dxso::ShaderType::Vertex {
        return D3DERR_INVALIDCALL;
    }
    // Reject bytecode that addresses a constant register past the model's
    // file (vs float file is 256; int/bool files are 16 and exist from
    // vs_2_0 on). The D3D9 validator fails these at create time.
    if program.violates_constant_register_limits() {
        warn!(target: LOG_TARGET, "reject CreateVertexShader: constant register out of range → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    let max_const_used = program.max_const_reg().map_or(0, |m| u32::from(m) + 1);
    let uses_rel_const = program.uses_relative_const_addressing();
    let uses_int_const = program.uses_dynamic_int_constants();
    // Extract the input-register semantics so `snapshot_shared` can resolve
    // a bound vertex declaration's elements → `[[attribute(N)]]` indices
    // without a trip to the encoder thread.
    let input_semantics = extract_input_semantics(&program);
    // `bytecode` drops at end of function scope; the parsed program moves
    // into an op bound for the encoder's program cache, and the wrapper
    // only records identity bits.
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().push_op(Box::new(move |enc| {
        enc.register_program(shader_id, program);
    }));
    let shader_obj = Direct3DVertexShader9::new(
        obj.inner_ptr(),
        shader_id,
        max_const_used,
        uses_rel_const,
        uses_int_const,
        input_semantics,
    );
    let shader_ptr = Box::into_raw(Box::new(shader_obj));
    // SAFETY: `shader_ptr` is a freshly created, live shader at refcount 1.
    unsafe { crate::com_ref::com_register_child(shader_ptr) };
    // SAFETY: vtable out-param; `shader` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(shader, shader_ptr.cast::<c_void>()) };
    0
}

/// Walk the VS's `dcl_*` declarations and collect the input-register semantics.
///
/// Non-Input declarations (samplers in PS, outputs like `oPos`)
/// are filtered out — only `v0..vN` entries land here.
fn extract_input_semantics(program: &mtld3d_core::dxso::DxsoProgram) -> Vec<InputSemantic> {
    program
        .declarations
        .iter()
        .filter_map(|decl| match decl {
            mtld3d_core::dxso::Declaration::Semantic {
                usage,
                usage_index,
                reg,
            } if reg.kind == mtld3d_core::dxso::RegKind::Input => Some(InputSemantic {
                usage: *usage,
                usage_index: u8::try_from(*usage_index)
                    .expect("D3D9 usage_index ≤ 15 (4-bit DXSO field)"),
                register_index: reg.index,
            }),
            _ => None,
        })
        .collect()
}

extern "system" fn device_set_vertex_shader(this: *mut c_void, shader: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let new = shader.cast::<Direct3DVertexShader9>();
    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `new` is null or a *mut Direct3DVertexShader9 supplied by
        // the calling game via SetVertexShader.
        let adopted = unsafe { CachedComPtr::adopt(new) };
        rec.record(StateOp::VertexShader(adopted));
        return D3D_OK;
    }
    // Redundant-set elimination: re-binding the same VS pointer leaves
    // attr-resolution, shader id, uses_rel_const and max_const_used all
    // identical, so skip the rebuild. VDECL: attr-resolution branch
    // depends on whether prog VS is bound. VS_SOURCE/VS_CONST: shader
    // identity + uses_rel_const + max_const_used change.
    let changed = dev.shader_bindings_mut().replace_vertex_shader(new);
    if changed {
        dev.mark_snapshot_dirty(
            SnapshotDirty::VDECL | SnapshotDirty::VS_SOURCE | SnapshotDirty::VS_CONST,
        );
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetVertexShader, !changed);
    0
}

extern "system" fn device_get_vertex_shader(this: *mut c_void, shader: *mut *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    if shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let vs = dev.shader_bindings().vertex_shader();
    if !vs.is_null() {
        // SAFETY: `vs` is non-null (checked) and points to a live
        // Direct3DVertexShader9 kept alive by the device binding.
        let add_ref = unsafe { (*vs).vtbl().add_ref };
        // SAFETY: calling the just-loaded `add_ref` thunk; D3D9 mandates
        // AddRef on out-pointer returns (the caller Releases the result).
        unsafe { add_ref(vs.cast::<c_void>()) };
    }
    // SAFETY: `shader` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *shader = vs.cast::<c_void>() };
    0
}

extern "system" fn device_set_vertex_shader_constant_f(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const f32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null()
        || count == 0
        || !const_window_in_range(start_register, count, CONSTANT_ROWS)
    {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked
    // above); per the D3D9 ABI the caller guarantees `count * 4` `f32`s
    // are readable from `constant_data`.
    let slice =
        unsafe { core::slice::from_raw_parts(constant_data.cast::<[f32; 4]>(), count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantF {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Redundant-set elimination: a write that leaves every mirror row
    // unchanged yields a byte-identical encoder delta, so skip the delta
    // push + dirty mark when nothing changed. WoW re-uploads identical
    // constant rows frequently; this is the ShaderConst analogue of the
    // RenderState / VDECL gates.
    let changed = dev
        .shader_bindings_mut()
        .write_vs_constants(start_register, slice);
    if changed {
        // Propagate the new rows to the encoder-side mirror via a delta op.
        // The encoder applies it before the next `Op::Draw` sees programmable
        // VS const state, so emit_snapshot_deltas does not need to bump
        // `vs_constants` into the API-thread arena (the encoder snapshots
        // from its own mirror at emit_draw time).
        propagate_vs_const_delta(dev, start_register, slice);
        // M2 skinning hot path: per-draw VS const update only needs the
        // VS constants slice re-bumped; everything else stays cached.
        dev.mark_snapshot_dirty(SnapshotDirty::VS_CONST);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetVsConst, !changed);
    0
}

/// True when a `[start, start + count)` constant-register window fits a register file.
///
/// The file is `limit` rows deep. Shared by the `Set*ShaderConstantF`
/// thunks, which reject out-of-range windows with `D3DERR_INVALIDCALL`
/// (D3D9 validates the window rather than silently clamping the write). The
/// sum is widened to `u64` first so a near-`u32::MAX` start register — a
/// signed `-1` passed as the start, or an unbounded `start++` probe sweep
/// that walks past the register file — cannot wrap back into range and spin
/// forever waiting for the rejection that a clamping write never yields.
fn const_window_in_range(start: u32, count: u32, limit: usize) -> bool {
    u64::from(start) + u64::from(count) <= limit as u64
}

/// Copy `mirror[start..]` into `out`, clamping to the mirror's length.
///
/// Shared by every `Get*ShaderConstant*` thunk. Out-of-range tail rows are
/// left as the caller supplied them — the Set/Get pair clamps to the same
/// fixed register file, so an in-range round-trip is exact, which is all the
/// D3D9 ABI promises here.
fn copy_constants_out<T: Copy>(mirror: &[T], start: u32, out: &mut [T]) {
    let start = start as usize;
    let end = (start + out.len()).min(mirror.len());
    if start >= end {
        return;
    }
    out[..end - start].copy_from_slice(&mirror[start..end]);
}

extern "system" fn device_get_vertex_shader_constant_f(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut f32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `f32`s are writable.
    let out = unsafe {
        core::slice::from_raw_parts_mut(constant_data.cast::<[f32; 4]>(), count as usize)
    };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_vertex_shader_constant_i(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are readable.
    let slice =
        unsafe { core::slice::from_raw_parts(constant_data.cast::<[i32; 4]>(), count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantI {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // A VS reading a dynamic integer constant (a `loop`/`rep` counter) consumes
    // these via the `vs_i` buffer (vertex slot 14), captured into the snapshot
    // on change. Boolean constants remain store-only (no shader consumer yet).
    let changed = dev
        .shader_bindings_mut()
        .write_vs_constants_i(start_register, slice);
    if changed {
        dev.mark_snapshot_dirty(SnapshotDirty::VS_CONST_I);
    }
    0
}

extern "system" fn device_get_vertex_shader_constant_i(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_i_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are writable.
    let out = unsafe {
        core::slice::from_raw_parts_mut(constant_data.cast::<[i32; 4]>(), count as usize)
    };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_vertex_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are readable.
    let slice = unsafe { core::slice::from_raw_parts(constant_data, count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantB {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Stored only — see `device_set_vertex_shader_constant_i`.
    dev.shader_bindings_mut()
        .write_vs_constants_b(start_register, slice);
    0
}

extern "system" fn device_get_vertex_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_b_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data, count as usize) };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_stream_source(
    this: *mut c_void,
    stream: u32,
    data: *mut c_void,
    offset: u32,
    stride: u32,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if stream as usize >= crate::bound_buffers::MAX_STREAMS {
        warn!(
            target: LOG_TARGET,
            "reject SetStreamSource(stream={stream}) → INVALIDCALL (exceeds max_streams)"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let vb = data.cast::<Direct3DVertexBuffer9>();
    if let Some(rec) = dev.recording_state_block_mut() {
        if stream == 0 {
            // SAFETY: `vb` is null or a *mut Direct3DVertexBuffer9 supplied by
            // the calling game via SetStreamSource.
            let adopted = unsafe { CachedComPtr::adopt(vb) };
            rec.record(StateOp::StreamSource {
                vb: adopted,
                offset,
                stride,
            });
        }
        // Streams > 0 are not captured by state blocks (single-stream
        // architecture; the multi-stream capture cluster is by design absent).
        return D3D_OK;
    }
    dev.bound_buffers_mut()
        .set_stream(stream as usize, vb, offset, stride);
    D3D_OK
}

extern "system" fn device_get_stream_source(
    this: *mut c_void,
    stream: u32,
    stream_data: *mut *mut c_void,
    offset_in_bytes: *mut u32,
    stride: *mut u32,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if stream_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // Streams 0..MAX round-trip their binding (only stream 0 is rendered); a
    // caller that binds a higher stream and reads it back — relying on the
    // binding outliving its own Release — sees the buffer. An out-of-range
    // stream is unbound → NULL/0 per the "nothing bound" contract (S_OK).
    let (vb_ptr, offset, vb_stride) = if (stream as usize) < crate::bound_buffers::MAX_STREAMS {
        let b = dev.bound_buffers();
        (
            b.stream_vertex_buffer(stream as usize),
            b.stream_offset(stream as usize),
            b.stream_stride(stream as usize),
        )
    } else {
        (std::ptr::null_mut(), 0, 0)
    };
    if !vb_ptr.is_null() {
        // SAFETY: `vb_ptr` is non-null (checked) and points to a live
        // Direct3DVertexBuffer9 kept alive by the device binding.
        let add_ref = unsafe { (*vb_ptr).vtbl().add_ref };
        // SAFETY: calling the just-loaded `add_ref` thunk; D3D9 mandates
        // AddRef on out-pointer returns.
        unsafe { add_ref(vb_ptr.cast::<c_void>()) };
    }
    // SAFETY: `stream_data` is non-null (checked) and the D3D9 ABI guarantees a
    // writable `*mut c_void` slot.
    unsafe { *stream_data = vb_ptr.cast::<c_void>() };
    if !offset_in_bytes.is_null() {
        // SAFETY: caller-supplied writable `u32` slot per the D3D9 ABI.
        unsafe { *offset_in_bytes = offset };
    }
    if !stride.is_null() {
        // SAFETY: caller-supplied writable `u32` slot per the D3D9 ABI.
        unsafe { *stride = vb_stride };
    }
    0 // S_OK
}

extern "system" fn device_set_stream_source_freq(
    this: *mut c_void,
    _stream: u32,
    _setting: u32,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetStreamSourceFreq → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_stream_source_freq(
    this: *mut c_void,
    _stream: u32,
    _setting: *mut u32,
) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetStreamSourceFreq → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_indices(this: *mut c_void, index_data: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let ib = index_data.cast::<Direct3DIndexBuffer9>();
    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `ib` is null or a *mut Direct3DIndexBuffer9 supplied by
        // the calling game via SetIndices.
        let adopted = unsafe { CachedComPtr::adopt(ib) };
        rec.record(StateOp::Indices(adopted));
        return D3D_OK;
    }
    dev.bound_buffers_mut().replace_index_buffer(ib);
    D3D_OK
}

extern "system" fn device_get_indices(this: *mut c_void, index_data: *mut *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if index_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let ib_ptr = dev.bound_buffers().index_buffer();
    if !ib_ptr.is_null() {
        // SAFETY: `ib_ptr` is non-null (checked) and points to a live
        // Direct3DIndexBuffer9 kept alive by the device binding.
        let add_ref = unsafe { (*ib_ptr).vtbl().add_ref };
        // SAFETY: calling the just-loaded `add_ref` thunk; D3D9 mandates
        // AddRef on out-pointer returns.
        unsafe { add_ref(ib_ptr.cast::<c_void>()) };
    }
    // SAFETY: `index_data` is non-null (checked) and the D3D9 ABI guarantees a
    // writable `*mut c_void` slot.
    unsafe { *index_data = ib_ptr.cast::<c_void>() };
    0 // S_OK
}

extern "system" fn device_create_pixel_shader(
    this: *mut c_void,
    function: *const u32,
    shader: *mut *mut c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // D3D9 nulls `*ppShader` on every failure path. Some apps ignore a failed
    // HRESULT and then `SetPixelShader(*ppShader)` regardless — an
    // uninitialised slot is a bogus
    // shader pointer that the bind path adopts (a wild `Bound` write), so null
    // it up front and let success overwrite it.
    null_out(shader);
    if function.is_null() || shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let Some(bytecode) = read_shader_bytecode(function) else {
        return D3DERR_INVALIDCALL;
    };
    // See `device_create_vertex_shader` — dump before parse so failed
    // attempts still get captured.
    let shader_id = ProgramId::from_tokens(&bytecode);
    maybe_dump_bytecode("ps", shader_id, &bytecode);
    let program = match mtld3d_core::dxso::parse(&bytecode) {
        Ok(p) => p,
        Err(e) => {
            warn!(target: LOG_TARGET, "CreatePixelShader parse failed: {e:?}");
            return D3DERR_INVALIDCALL;
        }
    };
    if program.shader_type != mtld3d_core::dxso::ShaderType::Pixel {
        return D3DERR_INVALIDCALL;
    }
    // Reject a pixel shader that declares a `v#` input with the POSITION0 usage —
    // the D3D9 validator (and the native assembler) forbid it; the rasterizer
    // position is `vPos`, not a `v#` input. Higher position indices are valid
    // user semantics.
    if program.has_invalid_pixel_input_decl() {
        warn!(target: LOG_TARGET, "reject CreatePixelShader: POSITION0 on a pixel-shader input register → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // Reject bytecode that addresses a constant register past the model's
    // file (ps float file is 8 / 32 / 224 for ps_1 / ps_2 / ps_3; int/bool
    // files are 16 and exist only from ps_3_0, so any int/bool use in ps_2_0
    // is out of range). The D3D9 validator fails these at create time.
    if program.violates_constant_register_limits() {
        warn!(target: LOG_TARGET, "reject CreatePixelShader: constant register out of range → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    let max_const_used = program.max_const_reg().map_or(0, |m| u32::from(m) + 1);
    let uses_bump_env = program.uses_bump_env();
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().push_op(Box::new(move |enc| {
        enc.register_program(shader_id, program);
    }));
    let shader_obj =
        Direct3DPixelShader9::new(obj.inner_ptr(), shader_id, max_const_used, uses_bump_env);
    let shader_ptr = Box::into_raw(Box::new(shader_obj));
    // SAFETY: `shader_ptr` is a freshly created, live shader at refcount 1.
    unsafe { crate::com_ref::com_register_child(shader_ptr) };
    // SAFETY: vtable out-param; `shader` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(shader, shader_ptr.cast::<c_void>()) };
    0
}

extern "system" fn device_set_pixel_shader(this: *mut c_void, shader: *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let new = shader.cast::<Direct3DPixelShader9>();
    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `new` is null or a *mut Direct3DPixelShader9 supplied by
        // the calling game via SetPixelShader.
        let adopted = unsafe { CachedComPtr::adopt(new) };
        rec.record(StateOp::PixelShader(adopted));
        return D3D_OK;
    }
    // Redundant-set elimination: re-binding the same PS pointer leaves
    // the shader id / max_const_used identical, so skip the rebuild.
    let changed = dev.shader_bindings_mut().replace_pixel_shader(new);
    if changed {
        dev.mark_snapshot_dirty(SnapshotDirty::PS_SOURCE | SnapshotDirty::PS_CONST);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetPixelShader, !changed);
    0
}

extern "system" fn device_get_pixel_shader(this: *mut c_void, shader: *mut *mut c_void) -> i32 {
    let _timer = bind_timer(this, BindSubCategory::Shader);
    if shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let ps = dev.shader_bindings().pixel_shader();
    if !ps.is_null() {
        // SAFETY: `ps` is non-null (checked) and points to a live
        // Direct3DPixelShader9 kept alive by the device binding.
        let add_ref = unsafe { (*ps).vtbl().add_ref };
        // SAFETY: calling the just-loaded `add_ref` thunk; D3D9 mandates
        // AddRef on out-pointer returns (the caller Releases the result).
        unsafe { add_ref(ps.cast::<c_void>()) };
    }
    // SAFETY: `shader` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `*mut c_void` slot owned by the caller.
    unsafe { *shader = ps.cast::<c_void>() };
    0
}

extern "system" fn device_set_pixel_shader_constant_f(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const f32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null()
        || count == 0
        || !const_window_in_range(start_register, count, PS_FLOAT_CONSTANT_LIMIT)
    {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked
    // above); per the D3D9 ABI the caller guarantees `count * 4` `f32`s
    // are readable from `constant_data`.
    let slice =
        unsafe { core::slice::from_raw_parts(constant_data.cast::<[f32; 4]>(), count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::PixelShaderConstantF {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Redundant-set elimination: see `device_set_vertex_shader_constant_f`.
    let changed = dev
        .shader_bindings_mut()
        .write_ps_constants(start_register, slice);
    if changed {
        propagate_ps_const_delta(dev, start_register, slice);
        dev.mark_snapshot_dirty(SnapshotDirty::PS_CONST);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetPsConst, !changed);
    0
}

extern "system" fn device_get_pixel_shader_constant_f(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut f32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `f32`s are writable.
    let out = unsafe {
        core::slice::from_raw_parts_mut(constant_data.cast::<[f32; 4]>(), count as usize)
    };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_pixel_shader_constant_i(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are readable.
    let slice =
        unsafe { core::slice::from_raw_parts(constant_data.cast::<[i32; 4]>(), count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::PixelShaderConstantI {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Stored only — see `device_set_vertex_shader_constant_i`.
    dev.shader_bindings_mut()
        .write_ps_constants_i(start_register, slice);
    0
}

extern "system" fn device_get_pixel_shader_constant_i(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_i_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are writable.
    let out = unsafe {
        core::slice::from_raw_parts_mut(constant_data.cast::<[i32; 4]>(), count as usize)
    };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_pixel_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are readable.
    let slice = unsafe { core::slice::from_raw_parts(constant_data, count as usize) };
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::PixelShaderConstantB {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Stored only — see `device_set_vertex_shader_constant_i`.
    dev.shader_bindings_mut()
        .write_ps_constants_b(start_register, slice);
    0
}

extern "system" fn device_get_pixel_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() || count == 0 {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_b_copy();
    // SAFETY: `constant_data` is non-null and `count != 0` (checked above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data, count as usize) };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_draw_rect_patch(
    this: *mut c_void,
    _handle: u32,
    _num_segs: *const f32,
    _tri_patch_info: *const c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::DrawRectPatch → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_draw_tri_patch(
    this: *mut c_void,
    _handle: u32,
    _num_segs: *const f32,
    _tri_patch_info: *const c_void,
) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::DrawTriPatch → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_delete_patch(this: *mut c_void, _handle: u32) -> i32 {
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::DeletePatch → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_create_query(
    this: *mut c_void,
    type_: u32,
    query: *mut *mut c_void,
) -> i32 {
    use crate::query::{Direct3DQuery9, data_size_for};
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // `query` being null is the D3D9 idiom for "is this query type supported?"
    // (returns S_OK without allocating). Unsupported types are logged so an
    // unhandled query type surfaces.
    let Some(data_size) = data_size_for(type_) else {
        warn!(
            target: LOG_TARGET,
            "reject CreateQuery(type={type_}) → NOTAVAILABLE (unsupported query type)"
        );
        return crate::D3DERR_NOTAVAILABLE;
    };
    if query.is_null() {
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(dev_obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let obj = Direct3DQuery9::new(dev_obj.inner_ptr(), type_, data_size);
    // SAFETY: vtable out-param; `query` is *mut *mut c_void per IDirect3DDevice9 ABI.
    let query_ptr = Box::into_raw(Box::new(obj));
    // SAFETY: `query_ptr` is a freshly created, live query at refcount 1.
    unsafe { crate::com_ref::com_register_child(query_ptr) };
    // SAFETY: vtable out-param; `query` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(query, query_ptr.cast::<c_void>()) };
    D3D_OK
}

// ── Silent-write audit: D3DRS_* classifier ──
// Closes the class of bug where a silently-ignored render-state hides a
// feature gap. Per-slot latches live on `DeviceInner.rs_warn_fired`;
// this table classifies each slot so the warn message is targeted.
// Slots not yet implemented are flagged as port candidates with a
// targeted message; slots that are obsolete or have no Metal analog
// route to `Obsolete`; everything else is `NotImplemented`.

enum RsClass {
    Consumed,
    PortCandidate(&'static str),
    /// Done-by-design no-op.
    ///
    /// Metal has no analog or the feature is
    /// obsolete on every modern Windows driver. Logged at info so the
    /// first write is still visible for triage, but off the warn surface
    /// (these are not port candidates). Mirrors the `log_once_info!`
    /// info-vs-warn cut line: obsolete no-ops are not port candidates.
    Obsolete(&'static str),
    NotImplemented,
}

const fn rs_classify(index: u32) -> RsClass {
    match index {
        // Bucket A — consumed by mtld3d (draw-path snapshot + FF pipeline).
        D3DRS_ZENABLE
        | D3DRS_ZWRITEENABLE
        | D3DRS_ZFUNC
        | D3DRS_ALPHABLENDENABLE
        | D3DRS_SRCBLEND
        | D3DRS_DESTBLEND
        // BLENDOP / BLENDOPALPHA / separate-alpha / *_BLENDALPHA are
        // consumed by pipeline_state::key_from_snapshot (mtld3d-core).
        // The per-field tests there assert the invariant that mutating
        // any of these in the snapshot produces a different
        // PipelineKey — if someone silently drops the value in the
        // builder path, those tests fail.
        | D3DRS_BLENDOP
        | D3DRS_BLENDOPALPHA
        | D3DRS_SEPARATEALPHABLENDENABLE
        | D3DRS_SRCBLENDALPHA
        | D3DRS_DESTBLENDALPHA
        // SRGBWRITEENABLE is consumed by the unix pipeline's sRGB
        // format upgrade (unix/src/metal/pipeline.rs); see the
        // PipelineSnapshot.srgb_write_enable field for the wiring.
        | D3DRS_SRGBWRITEENABLE
        | D3DRS_COLORWRITEENABLE
        | D3DRS_CULLMODE
        | D3DRS_SCISSORTESTENABLE
        | D3DRS_LIGHTING
        | D3DRS_ALPHATESTENABLE
        | D3DRS_ALPHAFUNC
        | D3DRS_ALPHAREF
        | D3DRS_AMBIENT
        | D3DRS_TEXTUREFACTOR
        | D3DRS_FOGENABLE
        | D3DRS_FOGVERTEXMODE
        | D3DRS_FOGCOLOR
        | D3DRS_FOGSTART
        | D3DRS_FOGEND
        | D3DRS_FOGDENSITY
        // COLORVERTEX + DIFFUSE/AMBIENTMATERIALSOURCE feed the DXSO FF emitter's
        // resolve_mat helper; see crates/dxso/src/ff.rs.
        | D3DRS_COLORVERTEX
        | D3DRS_DIFFUSEMATERIALSOURCE
        | D3DRS_AMBIENTMATERIALSOURCE
        // NORMALIZENORMALS is effectively on — ff.rs always normalizes
        // the eye-space normal regardless of the render-state bit.
        | D3DRS_NORMALIZENORMALS
        // SPECULARENABLE gates Blinn-Phong color1 emission in ff.rs;
        // SPECULAR/EMISSIVEMATERIALSOURCE feed the DXSO FF emitter's
        // resolve_mat for the specular / emissive accumulation sites.
        | D3DRS_SPECULARENABLE
        | D3DRS_SPECULARMATERIALSOURCE
        | D3DRS_EMISSIVEMATERIALSOURCE
        // LOCALVIEWER selects the specular view-vector model (per-vertex
        // normalize(-posEye) vs the constant infinite-viewer direction);
        // feeds FfVsFlags::LOCAL_VIEWER.
        | D3DRS_LOCALVIEWER
        // VERTEXBLEND + INDEXEDVERTEXBLENDENABLE feed FfState::build_vs_key →
        // FfVsKey::vertex_blend_count / vertex_blend_indexed → emit_vs blends
        // position + normal across the world-matrix palette. See
        // core/src/ff_state.rs::resolve_vertex_blend_count.
        | D3DRS_VERTEXBLEND
        | D3DRS_INDEXEDVERTEXBLENDENABLE
        // POINTSIZE_MIN / POINTSIZE_MAX are no-ops under the current
        // cap (caps.rs::max_point_size = 1.0; no POINTSPRITE / POINTSCALE
        // support). Intentionally silenced.
        | D3DRS_POINTSIZE_MIN
        | D3DRS_POINTSIZE_MAX
        // CLIPPING is a driver-side hint for frustum clipping — Metal
        // always clips to the viewport, so disabling this state on our
        // side is a no-op by construction, not a missing feature.
        | D3DRS_CLIPPING
        // BLENDFACTOR feeds the per-encoder constant blend color via
        // `Command::set_blend_color`, emitted in `emit_draw` whenever
        // the value differs from the default opaque white.
        | D3DRS_BLENDFACTOR
        // DEPTHBIAS / SLOPESCALEDEPTHBIAS feed Metal's per-encoder
        // rasterizer offset via `Command::set_depth_bias`, emitted
        // unconditionally per draw. Without these, ground-projected
        // decals (shadows, projectors, alpha overlays) z-fight with
        // the surface they sit on.
        | D3DRS_DEPTHBIAS
        | D3DRS_SLOPESCALEDEPTHBIAS => RsClass::Consumed,

        // Bucket B — not yet implemented → port-target candidates.
        D3DRS_STENCILENABLE => RsClass::PortCandidate("stencil test"),
        D3DRS_STENCILFAIL
        | D3DRS_STENCILZFAIL
        | D3DRS_STENCILPASS
        | D3DRS_STENCILFUNC
        | D3DRS_STENCILMASK
        | D3DRS_STENCILWRITEMASK
        | D3DRS_STENCILREF => RsClass::PortCandidate("stencil"),
        D3DRS_TWOSIDEDSTENCILMODE => RsClass::PortCandidate("two-sided stencil"),
        D3DRS_CCW_STENCILFAIL
        | D3DRS_CCW_STENCILZFAIL
        | D3DRS_CCW_STENCILPASS
        | D3DRS_CCW_STENCILFUNC => RsClass::PortCandidate("CCW stencil"),
        D3DRS_FOGTABLEMODE => RsClass::PortCandidate("table fog"),
        D3DRS_RANGEFOGENABLE => RsClass::PortCandidate("range fog"),
        D3DRS_FILLMODE => {
            RsClass::PortCandidate("non-solid fill mode (Metal has no native wireframe)")
        }
        // Bucket D — obsolete / no Metal analog. Info-level (not warn)
        // because the no-op IS the correct behaviour on every modern
        // driver.
        D3DRS_MULTISAMPLEANTIALIAS | D3DRS_MULTISAMPLEMASK => {
            RsClass::Obsolete("MSAA not supported (caps reject multisample formats)")
        }
        D3DRS_PATCHEDGESTYLE | D3DRS_POSITIONDEGREE | D3DRS_NORMALDEGREE => {
            RsClass::Obsolete("N-patch tessellation is obsolete — every modern driver ignores")
        }
        D3DRS_TWEENFACTOR => {
            RsClass::Obsolete("fixed-function vertex tweening is obsolete")
        }
        D3DRS_DEBUGMONITORTOKEN => {
            RsClass::Obsolete("debug-only token with no rendering effect")
        }

        // Bucket C — not implemented (no consumer in mtld3d).
        _ => RsClass::NotImplemented,
    }
}

// ── Silent-field audit: CreateTexture / CreateVertexBuffer / CreateIndexBuffer
// usage bits + pool value.
// Each unhandled usage bit and each non-default pool value gets one warn per
// process (log_once_warn is per-call-site; helper is called from all three
// Create* entry points so the first site to hit a bit wins).

fn warn_unused_usage_and_pool_once(kind: &str, usage: u32, pool: u32) {
    // Usage bits we actually honor at some level.

    // D3DUSAGE_DYNAMIC: no warn — every VB/IB uses a `PageBox` wrapped
    // as a Shared MTLBuffer on first draw (vertex_buffer.rs::vb_lock +
    // encoder.rs::ensure_{vb,ib}_mtl_buffer). Rename-on-DISCARD fires on
    // contention regardless of this flag, so we honour the dynamic
    // contract universally.
    // D3DUSAGE_WRITEONLY: no warn — Metal has no write-combined storage
    // tier. The PageBox backing is StorageModeShared; a Private+staging
    // variant would double uploads and defeat the zero-copy property we
    // care about. Architectural non-feature, not a stub.
    if usage & D3DUSAGE_SOFTWAREPROCESSING != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "Create{kind}: D3DUSAGE_SOFTWAREPROCESSING set but software-vertex-processing not supported"
        );
    }
    if usage & D3DUSAGE_DONOTCLIP != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "Create{kind}: D3DUSAGE_DONOTCLIP set but clip-bypass not honoured");
    }
    if usage & D3DUSAGE_POINTS != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "Create{kind}: D3DUSAGE_POINTS set but point-sprites not implemented"
        );
    }
    if usage & D3DUSAGE_RTPATCHES != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "Create{kind}: D3DUSAGE_RTPATCHES set but rectangular patches not implemented"
        );
    }
    if usage & D3DUSAGE_NPATCHES != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "Create{kind}: D3DUSAGE_NPATCHES set but N-patches not implemented");
    }
    // D3DUSAGE_AUTOGENMIPMAP is honoured for textures (`Create{Texture,
    // CubeTexture,VolumeTexture}` — runtime owns the mip chain via
    // Metal's blit-encoder `generateMipmaps`. Compressed-format requests
    // are dropped silently in `device_create_texture` since Metal
    // refuses to regenerate BC/DXT.
    if usage & D3DUSAGE_NONSECURE != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "Create{kind}: D3DUSAGE_NONSECURE set but non-secure hint ignored");
    }

    // Surface any remaining unknown bits (beyond the union of honored + warned).
    let known = D3DUSAGE_RENDERTARGET
        | D3DUSAGE_DEPTHSTENCIL
        | D3DUSAGE_WRITEONLY
        | D3DUSAGE_SOFTWAREPROCESSING
        | D3DUSAGE_DONOTCLIP
        | D3DUSAGE_POINTS
        | D3DUSAGE_RTPATCHES
        | D3DUSAGE_NPATCHES
        | D3DUSAGE_DYNAMIC
        | D3DUSAGE_AUTOGENMIPMAP
        | D3DUSAGE_NONSECURE;
    let unknown = usage & !known;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "Create{kind}: unknown usage bits {unknown:#x} — update warn_unused_usage_and_pool_once"
        );
    }

    // Pool: we treat everything as GPU-resident. DEFAULT = 0.
    // D3DPOOL_MANAGED: behaviourally equivalent to DEFAULT in our
    // implementation. Metal's StorageModeShared/Managed textures are already
    // CPU+GPU accessible with OS-level paging, so "driver-managed residency"
    // is a non-concept. We also don't implement device loss (see
    // device_reset), so MANAGED's "survives reset" promise is vacuously true
    // for DEFAULT too. Accept and move on.
    match pool {
        0 | D3DPOOL_MANAGED => {}
        D3DPOOL_SYSTEMMEM => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "Create{kind}: D3DPOOL_SYSTEMMEM set but system-memory backing not implemented"
            );
        }
        D3DPOOL_SCRATCH => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "Create{kind}: D3DPOOL_SCRATCH set but scratch pool not implemented"
            );
        }
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "Create{kind}: unknown D3DPOOL={other} — update warn_unused_usage_and_pool_once"
            );
        }
    }
}

// ── Silent-field audit: D3DPRESENT_PARAMETERS ──
// `back_buffer_format` already warns at d3d9_create_device. Warn on
// every other non-default field so the next mismatched present-time
// expectation surfaces on first device creation / reset.

/// Validate the swap-effect / back-buffer-count / presentation-interval fields.
///
/// Checked on a present-parameters block per the D3D9 `CreateDevice`/`Reset`
/// contract. `false` ⇒ the call must return `D3DERR_INVALIDCALL`.
///
/// - Swap effect must be DISCARD(1)/FLIP(2)/COPY(3); `0` and the `D3D9Ex` effects
///   (OVERLAY/FLIPEX/…) are rejected.
/// - COPY allows at most one back buffer.
/// - At most 3 back buffers (a requested 0 resolves to 1).
/// - Presentation interval must be DEFAULT/ONE/TWO/THREE/FOUR/IMMEDIATE.
pub const fn present_params_are_valid(pp: &mtld3d_types::D3DPRESENT_PARAMETERS) -> bool {
    const SWAPEFFECT_DISCARD: u32 = 1;
    const SWAPEFFECT_FLIP: u32 = 2;
    const SWAPEFFECT_COPY: u32 = 3;
    const MAX_BACK_BUFFERS: u32 = 3;
    const INTERVAL_DEFAULT: u32 = 0x0000_0000;
    const INTERVAL_ONE: u32 = 0x0000_0001;
    const INTERVAL_TWO: u32 = 0x0000_0002;
    const INTERVAL_THREE: u32 = 0x0000_0004;
    const INTERVAL_FOUR: u32 = 0x0000_0008;
    const INTERVAL_IMMEDIATE: u32 = 0x8000_0000;

    if !matches!(
        pp.swap_effect,
        SWAPEFFECT_DISCARD | SWAPEFFECT_FLIP | SWAPEFFECT_COPY
    ) {
        return false;
    }
    if pp.swap_effect == SWAPEFFECT_COPY && pp.back_buffer_count > 1 {
        return false;
    }
    if pp.back_buffer_count > MAX_BACK_BUFFERS {
        return false;
    }
    matches!(
        pp.presentation_interval,
        INTERVAL_DEFAULT
            | INTERVAL_ONE
            | INTERVAL_TWO
            | INTERVAL_THREE
            | INTERVAL_FOUR
            | INTERVAL_IMMEDIATE
    )
}

pub fn warn_present_params_fields_once(pp: &mtld3d_types::D3DPRESENT_PARAMETERS) {
    if pp.back_buffer_count > 1 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: back_buffer_count={} requested but only single back-buffer supported",
            pp.back_buffer_count
        );
    }
    if pp.multi_sample_type != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: multi_sample_type={} requested but MSAA not implemented",
            pp.multi_sample_type
        );
    }
    if pp.multi_sample_quality != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: multi_sample_quality={} requested but MSAA not implemented",
            pp.multi_sample_quality
        );
    }
    if pp.swap_effect != 0 && pp.swap_effect != 1 {
        // 0 is invalid per spec; 1 = D3DSWAPEFFECT_DISCARD. FLIP/COPY/etc.
        // would need a real swap chain instead of a single drawable.
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: swap_effect={} requested but only DISCARD (1) implemented",
            pp.swap_effect
        );
    }
    // D3DPRESENTFLAG_LOCKABLE_BACKBUFFER (0x1) is honoured: WoW's portrait
    // path locks the backbuffer + reads it back through BlitTextureToBuffer.
    // Warn once per distinct bit-combo for anything else so D3DPRESENTFLAG_*
    // additions don't collapse into a single stale line.
    let unhandled_present_flags = pp.flags & !D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
    if unhandled_present_flags != 0 {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(unhandled_present_flags),
            "CreateDevice: flags={:#x} set but D3DPRESENTFLAG_* bits {:#x} not honoured",
            pp.flags, unhandled_present_flags
        );
    }
    if pp.full_screen_refresh_rate_in_hz != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: refresh_rate_in_hz={} set but display-mode control not implemented",
            pp.full_screen_refresh_rate_in_hz
        );
    }
    // presentation_interval is honoured at the AttachMetalLayer call site via
    // resolve_display_sync, which fires its own log_once_warn_by! for non-1:1
    // ratios — no separate arm here.
}

/// Map `D3DPRESENT_PARAMETERS::PresentationInterval` to `CAMetalLayer.displaySyncEnabled`.
///
/// The boolean value is sent across the PE/Unix
/// boundary, and unsupported ratios fire a one-shot warn.
///
/// Pure mapping lives in `mtld3d_core::present`; this wrapper layers the
/// project's logging policy on top so the helper itself stays
/// host-testable without pulling in `log` plumbing.
pub fn resolve_display_sync(interval: u32) -> bool {
    use mtld3d_core::present::{DisplaySync, display_sync_for};
    let mapped = display_sync_for(interval);
    if matches!(mapped, DisplaySync::Fallthrough) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(interval),
            "PresentationInterval={interval:#x} not supported — only display-rate (DEFAULT/ONE) and IMMEDIATE are honoured; falling through to display-rate vsync"
        );
    }
    mapped.enabled()
}
