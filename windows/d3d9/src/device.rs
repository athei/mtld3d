use core::{ffi::c_void, mem::MaybeUninit, ops::DerefMut, ptr::NonNull};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

use log::{debug, error, info, trace, warn};
pub use mtld3d_core::encoder_data::{
    BeginVisibilityOp, BindColorOp, BindDepthOp, BindDepthOpFlags, CarryDepthOp, ClearColorOp,
    ClearColorRectsOp, ClearDepthStencilOp, ClearDepthStencilRectsOp, ColorFillOp, DepthBinding,
    DestroyBufferOp, DestroyTextureOp, EndVisibilityOp, GenerateMipmapsOp,
    GenerateMipmapsOrderedOp, NoteColorReadOp, PendingVbibRetention, ReadColorHandleOp,
    ReadDeviceBufferOp, ReadTextureColorHandleOp, ReadTextureHandleOp, ResolveDepthSurfaceOp,
    ResolveDepthTextureOp, ResolveDynamicDepthOp, RetireColorOp, RetireDepthOp, RtBinding,
    SetDumpDrawOp, SetVertexSamplerOp, SetVertexTextureOp, SetViewportOp, StretchBlitOp,
    StretchKind, StretchSurfaceFlags, StretchSurfaceInfo, UnbindExtraColorOp, UpdateColorRegionOp,
    UploadColorOp, UploadResampledOp, UploadTextureAndMipsOp, UploadTextureOp,
    UploadTextureOpFlags,
};

mod extended;
mod frame_dump;
mod mem_watch;
use mtld3d_core::{
    api_lock::{ApiGuard, ApiLock},
    caps,
    config::Mtld3dConfig,
    convert::{
        self, FfVsLayout, InputSemantic, d3d_to_metal_primitive, fvf_to_elements,
        resolve_attrs_for_ff, resolve_attrs_for_vs, vertex_count,
    },
    departed_textures::DepartedTextures,
    dirty_rect::DirtyRect,
    dxso::{MAX_LINKED_INPUTS, VsSamplerKinds, operand_token_count},
    encoder_draw::{
        ApiSnapshotCache, SnapshotAttributes, SnapshotDelta,
        draw_record::{BoundVertices, IndexBuffer, IndexedDraw, NonindexedDraw, StreamRecord},
    },
    ff_state::{FfState, FfVsDirty, TssWriteFeeds, tss_write_feeds},
    format::{
        FormatMapping, compute_mip_count, compute_mip_size, compute_volume_mip_count,
        is_dxt_format, linear_mip_size, linear_row_pitch, map_d3d_format, resolve_mip_levels,
    },
    ids::{BufferId, ProgramId, TextureId},
    page_box::PageBox,
    passes::BackbufferContents,
    perf::{
        ApiPerfState, ApiPerfStorage, ApiTimer, BindSubCategory, CycleAddTimer, CycleSetTimer,
        DeviceSubCategory, KeysGate,
    },
    present::LayerPacing,
    readback::{ReadbackDestination, ReadbackReject, ReadbackSource},
    render_scale::TargetExtent,
    render_state::{RsClass, rs_classify},
    reset_summary::{
        ResetDone, ResetFailed, ResetFailureEffect, ResetMethod, ResetSummary,
        changes_presentation, failure_warns, success_logs_at_info,
    },
    sampler_state::TEXTURE_LOD_SLOT,
    shader_constants::{int_bool_rows, window_in_range},
    snapshot::SnapshotSection,
    streams::validate_stream_freq,
    texture_flags::TextureFlags,
    upload_redirty::RedirtyQueue,
    validate_device::{StageFilterVerdict, stage_filter_verdict},
};
use mtld3d_shared::{
    BlitTextureToBufferParams, CreateColorTargetParams, CreateDepthTextureParams,
    DestroyCommandQueueParams, InPtr, InPtrMut, MetalHandle, OutPtr, SetPresentWaitPolicyParams,
    ValueIn, VtableThis,
    mtl::PresentWaitPolicy,
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, MTLTextureKind, NSViewKind},
    record_handle::DeviceRecordHandle,
};
use mtld3d_types::{
    D3D_MAX_SIMULTANEOUS_RENDERTARGETS, D3DCAPS9, D3DCLEAR_STENCIL, D3DCLEAR_TARGET,
    D3DCLEAR_ZBUFFER, D3DDEVICE_CREATION_PARAMETERS, D3DDISPLAYMODE,
    D3DERR_UNSUPPORTEDTEXTUREFILTER, D3DFMT_ATI1, D3DFMT_INDEX16, D3DFMT_INDEX32, D3DFMT_UYVY,
    D3DFMT_YUY2, D3DGAMMARAMP, D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPRESENT_PARAMETERS,
    D3DPRESENTFLAG_LOCKABLE_BACKBUFFER, D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_AMBIENT,
    D3DRS_AMBIENTMATERIALSOURCE, D3DRS_BLENDFACTOR, D3DRS_BLENDOP, D3DRS_BLENDOPALPHA,
    D3DRS_CLIPPING, D3DRS_CLIPPLANEENABLE, D3DRS_COLORVERTEX, D3DRS_COLORWRITEENABLE,
    D3DRS_COLORWRITEENABLE1, D3DRS_COLORWRITEENABLE2, D3DRS_COLORWRITEENABLE3, D3DRS_CULLMODE,
    D3DRS_DEPTHBIAS, D3DRS_DESTBLEND, D3DRS_DESTBLENDALPHA, D3DRS_DIFFUSEMATERIALSOURCE,
    D3DRS_EMISSIVEMATERIALSOURCE, D3DRS_FILLMODE, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_MULTISAMPLEMASK,
    D3DRS_NORMALIZENORMALS, D3DRS_POINTSCALE_A, D3DRS_POINTSCALE_B, D3DRS_POINTSCALE_C,
    D3DRS_POINTSCALEENABLE, D3DRS_POINTSIZE, D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN,
    D3DRS_POINTSPRITEENABLE, D3DRS_RANGEFOGENABLE, D3DRS_SCISSORTESTENABLE,
    D3DRS_SEPARATEALPHABLENDENABLE, D3DRS_SHADEMODE, D3DRS_SLOPESCALEDEPTHBIAS,
    D3DRS_SPECULARENABLE, D3DRS_SPECULARMATERIALSOURCE, D3DRS_SRCBLEND, D3DRS_SRCBLENDALPHA,
    D3DRS_SRGBWRITEENABLE, D3DRS_STENCILREF, D3DRS_TEXTUREFACTOR, D3DRS_VERTEXBLEND,
    D3DRTYPE_CUBETEXTURE, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_AUTOGENMIPMAP,
    D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DMAP, D3DUSAGE_DONOTCLIP, D3DUSAGE_DYNAMIC, D3DUSAGE_NONSECURE,
    D3DUSAGE_NPATCHES, D3DUSAGE_POINTS, D3DUSAGE_QUERY_FILTER, D3DUSAGE_RENDERTARGET,
    D3DUSAGE_RTPATCHES, D3DUSAGE_SOFTWAREPROCESSING, D3DUSAGE_WRITEONLY, D3DVIEWPORT9, Guid,
    IDirect3DDevice9ExVtbl, IDirect3DDevice9Vtbl, RENDER_STATE_COUNT, SAMPLER_STATE_COUNT,
    render_state_defaults,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, E_FAIL, LOG_TARGET,
    bound_buffers::BoundBuffers,
    bound_rt::{BoundRt, RENDER_TARGET_SLOTS},
    com_ref::{Bound, CachedComPtr},
    cursor::{self, CursorState},
    direct3d9::{depth_format_has_stencil, is_depth_stencil_format},
    draw::{
        DepthScissorFlags, DepthStencilFlags, DrawOp, IndexSource, PsSource, RenderStateSnapshot,
        ScratchSlice, StageBinding, StreamBinding, VertexSource, VsSource, arena_alloc_bytes,
        build_alpha_ref_bytes,
    },
    encoder::{
        ColorFillTarget, EncoderThread, FrameData, FrameInit, SubmitFence, TextureInfo,
        VbibWarmupEntry,
    },
    index_buffer::{Direct3DIndexBuffer9, IndexBufferCreateInfo},
    null_out,
    pixel_shader::Direct3DPixelShader9,
    shader_bindings::{
        BOOL_CONSTANT_COUNT, CONSTANT_ROWS, INT_CONSTANT_ROWS, PS_FLOAT_CONSTANT_LIMIT,
        ShaderBindings,
    },
    stage_bindings::{STAGE_COUNT, StageBindings, TextureSwapDelta},
    state_block::{RecordingStateBlock, StateOp},
    surface::{ColorTargetCreateInfo, Direct3DSurface9, SurfaceMultiSample, SystemMemoryDst},
    texture::{
        CUBE_FACE_COUNT, Direct3DTexture9, SourceImage, TextureCreateInfo, TextureInner,
        take_staging,
    },
    unix_call::unix_call,
    vertex_buffer::{Direct3DVertexBuffer9, VertexBufferCreateInfo},
    vertex_decl::{Direct3DVertexDeclaration9, VertexDeclCreateInfo},
    vertex_shader::Direct3DVertexShader9,
};

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

/// The vtable every device carries, extended or not.
///
/// A plain device answers `E_NOINTERFACE` for `IID_IDirect3DDevice9Ex`, so a
/// caller that follows COM never reaches past the base slots on one.
static DIRECT3D_DEVICE9_VTBL: IDirect3DDevice9ExVtbl = IDirect3DDevice9ExVtbl {
    base: IDirect3DDevice9Vtbl {
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
    },
    set_convolution_mono_kernel: extended::set_convolution_mono_kernel,
    compose_rects: extended::compose_rects,
    present_ex: extended::present_ex,
    get_gpu_thread_priority: extended::get_gpu_thread_priority,
    set_gpu_thread_priority: extended::set_gpu_thread_priority,
    wait_for_vblank: extended::wait_for_vblank,
    check_resource_residency: extended::check_resource_residency,
    set_maximum_frame_latency: extended::set_maximum_frame_latency,
    get_maximum_frame_latency: extended::get_maximum_frame_latency,
    check_device_state: extended::check_device_state,
    create_render_target_ex: extended::create_render_target_ex,
    create_offscreen_plain_surface_ex: extended::create_offscreen_plain_surface_ex,
    create_depth_stencil_surface_ex: extended::create_depth_stencil_surface_ex,
    reset_ex: extended::reset_ex,
    get_display_mode_ex: extended::get_display_mode_ex,
};

/// Number of user-clip-plane storage slots: `D3DCAPS9::MaxUserClipPlanes`.
///
/// D3D9 aliases every index at or past `MaxUserClipPlanes - 1` onto the last
/// slot, for `SetClipPlane` and `GetClipPlane` alike (Wine's
/// `test_clip_planes_limits` probes `0..2*D3DMAXUSERCLIPPLANES`), so the store
/// is exactly the planes the GPU can apply.
const CLIP_PLANE_SLOTS: usize = mtld3d_core::vs_draw::MAX_CLIP_PLANES;

// ── DeviceInner — non-repr(C) state behind the inner pointer ──

bitflags::bitflags! {
    /// Assorted per-device boolean state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DeviceFlags: u8 {
        /// Set after the app called `SetDepthStencilSurface(NULL)` — depth is explicitly absent.
        ///
        /// Distinguishes "no override, use the auto depth" from "explicitly
        /// unbound" so the pipeline's depth/stencil-format snapshot matches
        /// the actual render-pass attachment. Cleared by `reset_to_defaults`
        /// (which restores the default bindings).
        const DEPTH_EXPLICITLY_UNBOUND = 1 << 0;
        /// Set between a successful `BeginScene` and its `EndScene`.
        ///
        /// D3D9 pairs them strictly: `BeginScene` while already in a scene and
        /// `EndScene` without an open scene both return `D3DERR_INVALIDCALL`.
        /// Rendering does not otherwise depend on scene state.
        const IN_SCENE = 1 << 1;
        /// Set when the implicit presentation resources could not be rebuilt.
        ///
        /// `Present` and `TestCooperativeLevel` report
        /// `D3DERR_DEVICENOTRESET` until a later `Reset` succeeds, which is
        /// how an app learns it must retry (after releasing any
        /// `D3DPOOL_DEFAULT` resources that blocked it).
        const NOT_RESET = 1 << 2;
        /// Set once the last `Release` has begun tearing the device down.
        ///
        /// The teardown hands a fullscreen window back before the cursor
        /// subclass comes off, and the restore's `WM_SIZE` reaches the
        /// subclass with the device no longer fullscreen. Answering it
        /// would rebuild a back buffer whose handles the teardown has
        /// already captured and is about to release, so the resize path
        /// checks this and stands down.
        const RELEASING = 1 << 3;
        /// Independent coverage request controlled only by A2M1/A2M0.
        const A2M_ENABLED = 1 << 4;
        /// Created by an interface `Direct3DCreate9Ex` made: a `D3D9Ex` device.
        ///
        /// Seeded at creation and never cleared, `reset_to_defaults` included.
        /// It refuses `D3DPOOL_MANAGED`, reads a system-memory `pSharedHandle`
        /// as user memory, keeps its state and its default-pool resources
        /// across `Reset`, and answers `IID_IDirect3DDevice9Ex`.
        const EXTENDED = 1 << 5;
        /// Set on an extended device once a failed `Reset` has warned, until one succeeds.
        ///
        /// A plain device's rejected or failed `Reset` leaves it waiting for
        /// another (`NOT_RESET`), which is what keeps a game's per-frame
        /// retries at debug. An extended device's rejected `Reset` changes
        /// nothing and owes nothing, so this is what quiets its retries; the
        /// next success clears it and says so at info.
        const RESET_FAILURE_LOGGED = 1 << 6;
    }
}

pub struct DeviceInner {
    // Metal handles / presentation.
    device_handle: MetalHandle<MTLDeviceKind>,
    /// This device's unix-side record, opaque here.
    ///
    /// Names the device on every thunk that acts on it, from a submission to
    /// the destroy that frees the record and the queue behind it.
    record_handle: DeviceRecordHandle,
    view_handle: MetalHandle<NSViewKind>,
    layer_handle: MetalHandle<CAMetalLayerKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `backbuffer_handle`, attached under `D3DRS_SRGBWRITEENABLE`.
    ///
    /// Created with the back buffer and destroyed with it, so the two always
    /// name the same storage.
    backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of the back buffer, NULL when it is single-sampled.
    ///
    /// Every pass on the back buffer attaches this and resolves into
    /// `backbuffer_handle`, which stays the texture Present, `StretchRect`
    /// and `LockRect` read.
    backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `backbuffer_msaa_handle`, NULL whenever that is.
    ///
    /// Attached in the companion's place under `D3DRS_SRGBWRITEENABLE`; the
    /// pass then resolves into `backbuffer_srgb_handle`, which Metal requires
    /// to carry the attachment's pixel format.
    backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    depth_stencil_handle: MetalHandle<MTLTextureKind>,
    depth_stencil_format: u32,
    /// `D3DMULTISAMPLE_TYPE` the swap chain was created with.
    ///
    /// Reported by `GetDesc` on the implicit back buffer and depth-stencil
    /// surfaces, and by the implicit swap chain's present parameters. Also
    /// decides whether `D3DRS_MULTISAMPLEMASK` applies: D3D9 defines the mask
    /// only for the maskable levels.
    backbuffer_multi_sample_type: u32,
    /// `MultiSampleQuality` the swap chain was created with.
    backbuffer_multi_sample_quality: u32,
    /// Sample count the back buffer and the implicit depth surface carry, 1 for none.
    backbuffer_sample_count: u8,
    /// Scene / depth-binding boolean state (`DEPTH_EXPLICITLY_UNBOUND` / `IN_SCENE`).
    ///
    /// See [`DeviceFlags`].
    flags: DeviceFlags,
    /// Window state saved when this device took its window fullscreen.
    ///
    /// `Some` exactly while the device is fullscreen. Restored (and cleared)
    /// by a windowed `Reset` or by device destruction.
    fullscreen: Option<crate::fullscreen::SavedWindow>,
    backbuffer_width: u32,
    backbuffer_height: u32,
    /// Fraction of the logical back buffer that is actually rasterized.
    ///
    /// `backbuffer_width`/`backbuffer_height` above stay logical — the size
    /// D3D9 reports and the space every game-supplied rect lives in. This
    /// converts those to the Metal texture's own resolution at the point a
    /// value becomes a Metal command. Identity unless `render.scale` is set.
    render_scale: mtld3d_core::render_scale::RenderScale,
    /// Currently-bound vertex declaration (null = none).
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline rather than going through the COM vtable's
    /// `AddRef`/`Release` thunks. It also carries the FVF: `SetFVF` binds
    /// the declaration it built for the FVF, which reports that FVF, so the
    /// most recent of `SetFVF` and `SetVertexDeclaration` wins and no other
    /// path can set one without the other.
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
    /// The FVF `bind_fvf_decl` last bound, with its implicit declaration from `fvf_decl_cache`.
    ///
    /// Only compared against the bound pointer, never dereferenced. The cache
    /// holds every entry until the device drops, so no other declaration can
    /// take this address while the pair names it.
    last_fvf_bind: (u32, *mut Direct3DVertexDeclaration9),
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
    /// The device's entry-point lock, `Some` under `D3DCREATE_MULTITHREADED`.
    ///
    /// Leaked at creation rather than owned here: a child `Release` can drive
    /// the device to zero and free this struct while the guard the child's
    /// thunk took is still live, and that guard releases through the lock on
    /// its way out, and a texture keeps entering it after the device is gone
    /// ([`Self::api_lock_ptr`]). One small allocation per flagged device
    /// creation. `None` is the whole of "not multithreaded": every thunk then
    /// takes the no-op guard.
    api_lock: Option<&'static ApiLock>,
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
    current_frame: FrameData,
    /// Shared with the encoder thread and the unix completion handler.
    ///
    /// The frame's submit seq is stamped in `stamp_and_swap`; this atomic
    /// is the *retired* seq, raised on the unix side only, with a `Release`
    /// `fetch_max`, as draw command buffers end in sequence order, each
    /// observed by its own status (from their completion handlers and the
    /// retirement waits), and by a failed submission's cleanup. Every
    /// PE-side reader only `Acquire`-loads it, so a read that misses a
    /// concurrent retirement is a lower bound on GPU progress: it can make a
    /// caller more conservative, never less.
    coherent_seq: Arc<AtomicU64>,
    /// Texture-upload retirement seq.
    ///
    /// Raised as the *upload* command buffers (a separate CB committed before
    /// the draw CB) end in
    /// sequence order, and for a submission without one once no upload
    /// buffer up to it is in flight. Because that CB retires ~a frame earlier
    /// than the draw CB tracked by `coherent_seq`, a texture mip's staging
    /// reads as retired sooner, so a contended whole-mip `LockRect` can write
    /// in place instead of renaming + preserving. Texture-staging contention
    /// reads this alone; the retention drain, VB/IB backings included, reads
    /// the lower of the two.
    upload_coherent_seq: Arc<AtomicU64>,
    /// Highest submit seq whose command buffer the GPU aborted.
    ///
    /// `fetch_max`'d as a draw or upload command buffer that reached
    /// `MTLCommandBufferStatus::Error` is retired, before the retirement
    /// counter passes it. Kept separate from `coherent_seq`
    /// because an aborted command buffer is genuinely finished with its
    /// source memory: withholding the retirement bump instead would pin
    /// every seq-gated queue behind a seq that never retires, and
    /// `wait_for_gpu_idle` would block forever. The encoder reads the pair
    /// to tell an upload it can free from one whose blit was discarded and
    /// has to be re-issued.
    failed_submit_seq: Arc<AtomicU64>,
    /// Live VB/IB retained-`PageBox` byte total, shared with the encoder.
    ///
    /// Mirrors `coherent_seq`'s sharing. The encoder `fetch_add`s on intake
    /// and `fetch_sub`s on drain; the API thread reads it in
    /// `alloc_pagebox_capped` to cap retention before a rename burst
    /// balloons PE-heap usage into 32-bit OOM territory.
    vbib_retained_bytes: Arc<AtomicU64>,
    /// Address-space watch state: this device's present-sampling counter.
    ///
    /// See `mem_watch::MemWatchState`; the thresholds it reports against stay
    /// process-wide because the address space is.
    mem_watch: mem_watch::MemWatchState,
    /// Running total of bytes occupied by live `D3DPOOL_DEFAULT` resources.
    ///
    /// Counts RTs + DEFAULT textures, maintained at `register_texture` /
    /// `deregister_texture`, plus the standalone `CreateRenderTarget` and
    /// `CreateDepthStencilSurface` surfaces, maintained at
    /// `register_standalone_surface` / `deregister_standalone_surface` (they
    /// own a Metal texture with no `TextureInner` behind it).
    /// `GetAvailableTextureMem` reports `VRAM_BUDGET - this`, so the value
    /// visibly decreases as the app allocates GPU resources.
    vram_bytes_used: Arc<AtomicU64>,
    /// Number of `D3DPOOL_DEFAULT` resources and implicit surfaces the app holds a reference to.
    ///
    /// Maintained by the COM engine on each such object's public 0↔1
    /// refcount edge (`ComChild::blocks_reset_while_referenced`); the
    /// device's own bind slots never count. `Reset` fails with
    /// `D3DERR_INVALIDCALL` while it is non-zero, as D3D9 requires.
    outstanding_reset_blockers: AtomicU32,
    /// What `SetMaximumFrameLatency` last stored, which `GetMaximumFrameLatency` reports.
    ///
    /// Reported, not enforced: the encoder's own queue depth bounds how far
    /// the application runs ahead of the GPU.
    max_frame_latency: u32,
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
    perf: ApiPerfStorage,
    /// Byte total of VB/IB retirement commands captured in the current frame.
    ///
    /// Not yet handed to the encoder (and thus not yet in
    /// `vbib_retained_bytes`). Added to the shared total when reading the
    /// retention cap so the current frame's renames count immediately; reset
    /// to 0 at `stamp_and_swap` when the queue is handed off.
    pending_retention_bytes: u64,
    /// Proactive retention cap in bytes.
    ///
    /// When `vbib_retained_bytes + pending_retention_bytes` reaches this,
    /// `alloc_pagebox_capped` drains (and, if still over, mid-frame-submits +
    /// GPU-waits) before allocating, bounding peak PE-heap retention. This is
    /// the only bound on retained bytes; `0` removes it entirely.
    retention_cap_bytes: u64,
    /// The configuration of the `IDirect3D9` that created this device.
    ///
    /// Shared with the encoder thread; the device reads it wherever an option
    /// decides a per-call answer.
    config: Arc<Mtld3dConfig>,
    render_states: [u32; RENDER_STATE_COUNT],
    /// Last numeric POINTSIZE, preserved across A2M and RESZ controls.
    point_size: u32,
    /// Per-slot "have we warned about this unsupported RS write yet?" latch.
    ///
    /// Bit-packed one bit per RS index (`[u64; 4]` covers all 210 slots in
    /// 32 B instead of 210). Prevents log spam while still firing once per
    /// slot per device when a caller writes a non-default value to an RS slot
    /// we don't consume. Access via `rs_warn_fired()` / `mark_rs_warn()`.
    rs_warn_fired: [u64; RENDER_STATE_COUNT.div_ceil(64)],
    ff_state: FfState,
    vs_draw: mtld3d_core::vs_draw::VsDrawState,
    /// `SetScissorRect`'s rect as the game wrote it, which `GetScissorRect` hands back.
    ///
    /// The render-state snapshot narrows it to the region it lets through,
    /// which the encoder emits as `setScissorRect` while
    /// `D3DRS_SCISSORTESTENABLE` is on.
    scissor_rect: D3DRECT,
    /// Viewport set by `SetViewport`.
    ///
    /// Served back by `GetViewport`. Width/height also flow to the encoder so
    /// `setViewport` emits with the actual viewport dimensions instead of the
    /// backbuffer size.
    viewport: D3DVIEWPORT9,
    /// User clip planes set by `SetClipPlane`, served back by `GetClipPlane`.
    ///
    /// The first `vs_draw::MAX_CLIP_PLANES` slots reach the GPU through the
    /// per-draw `VsDraw` uniform whenever `D3DRS_CLIPPLANEENABLE` names them
    /// (and `D3DRS_CLIPPING` is on); the rest are a CPU round-trip for the
    /// conformance suite's out-of-range probes. The index is clamped into
    /// range, so an out-of-range plane aliases the last slot instead of
    /// being rejected.
    clip_planes: [[f32; 4]; CLIP_PLANE_SLOTS],

    // Submodule-owned state. Each group's fields are private to its own
    // submodule; only `group()` / `group_mut()` cross the boundary.
    cursor: CursorState,
    bound_rt: BoundRt,
    bound_buffers: BoundBuffers,
    shader_bindings: ShaderBindings,
    stage_bindings: StageBindings,
    /// Vertex texture fetch slots (`D3DVERTEXTEXTURESAMPLER0..3`).
    ///
    /// Bound-slot refcounts like `StageBindings`; the encoder mirror is
    /// pushed via ops at set time rather than riding the per-draw
    /// snapshot, since vertex textures change orders of magnitude less
    /// often than draws.
    vertex_textures: [CachedComPtr<crate::texture::Direct3DTexture9, Bound>; 4],
    /// Sampler state for the vertex slots, for `GetSamplerState`.
    vertex_sampler_states: [[u32; SAMPLER_STATE_COUNT]; 4],
    /// Texture kind bound at each vertex fetch slot.
    ///
    /// Maintained by `set_vertex_texture_slot` alongside the slot write and
    /// folded into `ProgrammableVsSource::sampler_kinds`, so the vertex
    /// emitter types each `[[texture(n)]]` argument from the texture the
    /// draw binds rather than from the shader's `dcl_*`.
    vertex_texture_kinds: VsSamplerKinds,
    /// The last texture-backed depth-stencil bind, as `(texture, width, height)`.
    ///
    /// Read by the `depth.aliasSameSize` carry: a bind of a *different*
    /// texture with equal dimensions inherits this one's contents (see
    /// the config key's doc). Dimensions are those of the bound mip.
    last_sized_depth: Option<(TextureId, u32, u32)>,
    /// In-progress `BeginStateBlock` recording.
    ///
    /// `Some(..)` between a successful `BeginStateBlock` and its matching
    /// `EndStateBlock`, or a `Reset` that ends it first. While set, every
    /// state-change COM setter diverts its write into the block instead of
    /// the live device: spec-correct replay semantics for `Apply()` on the
    /// resulting state block. Null-safe to read via
    /// `recording_state_block()`; mutating through
    /// `recording_state_block_mut()` is how each setter records its op.
    recording_state_block: Option<Box<RecordingStateBlock>>,
    /// The pacing this device has handed to its layer.
    ///
    /// Seeded with what `AttachMetalLayer` was called with, at device
    /// creation and again at a retarget's fresh attach, and moved to a queued
    /// value once `stamp_and_swap` puts that value on the frame carrying it.
    /// A `Reset` compares its resolved interval against this, so one that
    /// does not move the pacing sends no thunk.
    layer_pacing: LayerPacing,
    /// `Some(v)` if `IDirect3DDevice9::Reset` changed `PresentationInterval` since the last frame.
    ///
    /// Consumed by `stamp_and_swap` and applied by the encoder thread on the
    /// first frame sent after the Reset, before that frame's own
    /// `nextDrawable`, matching the spec's "next Present" timing.
    pending_pacing: Option<LayerPacing>,
    /// The gamma ramp the application last set, identity until it sets one.
    ///
    /// Kept verbatim so `GetGammaRamp` answers with what `SetGammaRamp` was
    /// given, whether or not it reached the present pass: D3D9 stores the
    /// ramp on the swap chain and reports it back unchanged.
    gamma_ramp: Box<D3DGAMMARAMP>,
    /// The present-pass table a queued ramp change carries, if one is queued.
    ///
    /// `Some(Some(table))` applies a table, `Some(None)` removes the one the
    /// layer has, and `None` is the steady state. Consumed by
    /// `stamp_and_swap` and applied by the encoder thread on the first frame
    /// sent afterwards, so the ramp lands on the present that follows the
    /// call, the same timing the queued pacing uses.
    pending_gamma: Option<mtld3d_core::gamma::Change>,
    /// Whether the layer currently carries a table.
    ///
    /// A ramp that changes nothing sends nothing: an identity ramp set twice,
    /// or a windowed device's ramp, leaves this alone.
    gamma_applied: bool,
    /// The colour render-target binding most recently applied via `SetRenderTarget`.
    ///
    /// `None` means the implicit backbuffer default is in effect. The encoder's
    /// per-frame pass state resets to the backbuffer on every fresh frame, but a
    /// D3D9 render-target binding survives an *internal*
    /// `flush_current_frame_blocking` (a mid-frame readback flush is not a
    /// Present). Re-pushed into the fresh frame after such a flush so draws that
    /// follow a `GetRenderTargetData` keep rendering to the bound RT instead of
    /// silently reverting to the backbuffer format.
    ///
    /// Paired with what the bound resource is rasterized at, which the encoder
    /// thread cannot reach the resource to ask for.
    last_color_rt_binding: Option<(RtBinding, mtld3d_core::render_scale::RenderScale)>,
    /// Render targets 1..3 as most recently bound, re-asserted like `last_color_rt_binding`.
    ///
    /// Index `i` holds slot `i + 1`; `None` is an unbound slot.
    last_extra_rt_bindings:
        [Option<(RtBinding, mtld3d_core::render_scale::RenderScale)>; RENDER_TARGET_SLOTS - 1],
    /// Texture id of the autogen render-target bound at each slot, if any.
    ///
    /// When a slot changes away from it the mip chain is regenerated (a
    /// render or clear into an `D3DUSAGE_AUTOGENMIPMAP` texture must refresh
    /// the lower levels).
    cur_autogen_rt_ids: [Option<TextureId>; RENDER_TARGET_SLOTS],
    /// The depth/stencil attachment most recently applied via `SetDepthStencilSurface`.
    ///
    /// Holds binding + `is_sampleable` + `depth_has_stencil`. `None` means the
    /// implicit auto-depth default is in effect. Like `last_color_rt_binding`,
    /// re-pushed after a mid-frame flush: the encoder's per-frame reset
    /// re-attaches the implicit depth-stencil, but a draw issued after
    /// `SetDepthStencilSurface(NULL)` + readback would then carry a depth
    /// attachment the pipeline declares no format for (Metal rejects the
    /// pipeline-vs-framebuffer depth/stencil mismatch and drops the draw).
    /// `(binding, is_sampleable, has_stencil, sample_count)`.
    last_depth_binding: Option<(DepthBinding, bool, bool, u8)>,
    /// Live `IDirect3DTexture9` objects.
    ///
    /// Populated in `texture_create` after `Box::into_raw`; entries removed in
    /// `texture_release`'s rc→0 path before the inner Box is dropped, or when
    /// a texture migrates to another device. Keyed by the immutable texture
    /// id so upload answers resolve only the textures they name. Walked
    /// by `evict_managed_resources` to mark per-mip `dirty` flags so the next
    /// bind replays the staging upload — the spec contract for
    /// `IDirect3DDevice9::EvictManagedResources` is "evict from VRAM, runtime
    /// re-uploads on next use," and lazy upload's bind-time flush is exactly
    /// that re-upload trigger. Mutex contention is zero in steady state:
    /// create, release and Evict run one at a time, on the API thread or
    /// serialised by the device `ApiLock` under `D3DCREATE_MULTITHREADED`.
    live_textures: Mutex<rustc_hash::FxHashMap<TextureId, *mut TextureInner>>,
    /// Textures that moved to another device and still have storage on this one's encoder.
    ///
    /// The move runs under the adopting device's `ApiLock`, which does not
    /// cover this device's `current_frame`, so it files the id here instead of
    /// recording the destroy itself; `stamp_and_swap` drains the list into
    /// the frame it hands over, under this device's own lock. A leaf mutex
    /// like `live_textures`.
    departed_textures: DepartedTextures,
    /// Metal textures of standalone targets released after the device failed.
    ///
    /// A device whose failure is latched sends no frame again, so a retire
    /// operation recorded then would never reach the encoder that destroys
    /// the texture. The handles wait here instead, sRGB twins ahead of the
    /// texture each holds a retain on, and the final `Release` destroys them
    /// once the encoder's shutdown has waited for the GPU.
    retired_while_failed: Vec<u64>,
    /// What the encoder made of each upload, waiting to be acted on.
    ///
    /// The bind-time flush clears a level's dirty bit and takes its pending
    /// rectangle before the job crosses to the encoder thread, and
    /// `UnlockRect` publishes only the rectangle the game locked, so a
    /// decline downstream of the hand-off would otherwise leave the region
    /// unannounced for the rest of the run. The encoder files the
    /// subresource and its rectangle here, along with the subresources whose
    /// upload it did emit and whose staging is waiting on that answer;
    /// [`Self::apply_upload_answers`] drains the queue once a frame, marks
    /// the declined ones dirty again and releases the staging of the rest.
    upload_redirty: Arc<RedirtyQueue>,
    /// Per-draw snapshot dirty-bitmask.
    ///
    /// Each bit marks one `CurrentSnapshot` piece as needing rebuild on the
    /// next Draw. Set by every live-state-path Set* method (after the
    /// `recording_state_block_mut` early-return so recording writes don't
    /// touch it). `emit_snapshot_deltas` walks the bits, rebuilds only the
    /// dirty pieces, and clears the flag.
    ///
    /// Frame rotation sets all dirty bits so native state is initialized before
    /// the first draw in each frame.
    snapshot_dirty: SnapshotDirty,
    /// Owned keys retained only for subsequent API-side dirty-state builders and dumps.
    snapshot_cache: ApiSnapshotCache,
    /// The Ctrl+Shift+P draw-state dump, see `frame_dump`.
    frame_dump: frame_dump::FrameDump,
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
    /// Stream 0's consumed extent from the most recent `VDECL` rebuild.
    ///
    /// Sizes a `Draw*PrimitiveUP` vertex copy: when it is past the draw's
    /// stride, the last vertex's crossing attribute reads past the vertices
    /// the application supplied, which the copy zero-fills. Same lifecycle as
    /// `cached_ff_vs_layout`; the UP draws read it after their snapshot.
    cached_stream0_extent: u32,
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
        // flat_shade), SRGBWRITEENABLE carries `D3DRS_SRGBWRITEENABLE` to the
        // encoder (VariantKey srgb_write) and POINTSPRITEENABLE flips
        // VariantFlags::POINT_SPRITE; all rebuild the PS source + variant key.
        D3DRS_SHADEMODE | D3DRS_SRGBWRITEENABLE | D3DRS_POINTSPRITEENABLE => rs
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
        // Key-only: LOCALVIEWER flips FfVsFlags::LOCAL_VIEWER (specular
        // view-vector model) and POINTSCALEENABLE flips
        // FfVsFlags::POINT_SCALE (eye-distance attenuation); no constant
        // section reads either.
        D3DRS_LOCALVIEWER | D3DRS_POINTSCALEENABLE => rs.union(SnapshotDirty::VS_SOURCE),
        // The point size, its clamp and the scale factors travel in the
        // per-draw VsDraw uniform every vertex shader reads.
        D3DRS_POINTSIZE | D3DRS_POINTSIZE_MIN | D3DRS_POINTSIZE_MAX | D3DRS_POINTSCALE_A
        | D3DRS_POINTSCALE_B | D3DRS_POINTSCALE_C => rs.union(SnapshotDirty::VS_DRAW),
        // The enabled-plane set repacks the VsDraw clip rows and its count
        // keys both vertex-shader sources (the programmable side is added
        // past `ff_aware_mask` by the SetRenderState thunk).
        D3DRS_CLIPPLANEENABLE | D3DRS_CLIPPING => rs
            .union(SnapshotDirty::VS_DRAW)
            .union(SnapshotDirty::VS_SOURCE),
        _ => rs,
    }
}

bitflags::bitflags! {
    /// Per-draw snapshot dirty mask.
    ///
    /// One bit per cached piece of the native `DrawReader` snapshot. See
    /// `SnapshotCache` doc on `DeviceInner::snapshot_dirty` for lifecycle.
    /// Each bit is its [`SnapshotSection`]'s, so the perf summary's
    /// per-section rebuild counters read the mask directly.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SnapshotDirty: u32 {
        /// `RenderStateSnapshot` (~25 RS slots).
        const RS          = SnapshotSection::Rs.bit();
        /// `[Option<StageBinding>; STAGE_COUNT]` — bound textures and per-stage sampler state.
        ///
        /// `bound_texture_mask` rebuild is folded into this branch (see
        /// `emit_snapshot_deltas`).
        const STAGES      = SnapshotSection::Stages.bit();
        /// `has_depth` + `has_stencil` on the current render target.
        const RT_DS       = SnapshotSection::RtDs.bit();
        /// Vertex attribute layout (`AttrSnapshot`: attrs slice, stride, vdecl hash).
        const VDECL       = SnapshotSection::Vdecl.bit();
        /// Pipeline variant key.
        const VARIANT     = SnapshotSection::Variant.bit();
        /// VS source (FF key or programmable `vs_id`).
        const VS_SOURCE   = SnapshotSection::VsSource.bit();
        /// PS source.
        const PS_SOURCE   = SnapshotSection::PsSource.bit();
        /// VS constants slot.
        const VS_CONST    = SnapshotSection::VsConst.bit();
        /// PS constants slot.
        const PS_CONST    = SnapshotSection::PsConst.bit();
        /// Alpha-ref bytes (PS slot 14).
        const ALPHA_REF   = SnapshotSection::AlphaRef.bit();
        /// Fog-color bytes (PS slot 13).
        const FOG_COLOR   = SnapshotSection::FogColor.bit();
        /// Bump-environment matrix bytes (PS slot 12).
        ///
        /// Per-stage `D3DTSS_BUMPENVMAT*` + luminance, consumed by SM1
        /// `texbem`/`texbeml`/`bem`.
        const BUMP_ENV    = SnapshotSection::BumpEnv.bit();
        /// VS integer-constant file bytes (vertex slot 14).
        ///
        /// `vs_constants_i`, consumed by a VS reading a dynamic (non-`defi`)
        /// integer constant.
        const VS_CONST_I  = SnapshotSection::VsConstI.bit();
        /// Per-draw `VsDraw` uniform bytes (point size state).
        ///
        /// `mtld3d_core::vs_draw::VsDrawState::build_bytes` over the point render
        /// states, bound for every draw.
        const VS_DRAW     = SnapshotSection::VsDraw.bit();
        /// VS boolean-constant bitmask (vertex slot 26).
        ///
        /// `vs_constants_b`, consumed by a VS reading a dynamic (non-`defb`)
        /// boolean constant.
        const VS_CONST_B  = SnapshotSection::VsConstB.bit();
        /// PS integer-constant file bytes (fragment slot 11).
        ///
        /// `ps_constants_i`, consumed by a PS reading a dynamic (non-`defi`)
        /// integer constant.
        const PS_CONST_I  = SnapshotSection::PsConstI.bit();
        /// PS boolean-constant bitmask (fragment slot 10).
        ///
        /// `ps_constants_b`, consumed by a PS reading a dynamic (non-`defb`)
        /// boolean constant.
        const PS_CONST_B  = SnapshotSection::PsConstB.bit();
    }
}

// The flags take their bits from `SnapshotSection`; pin a few so a renumbered
// section cannot move a flag unnoticed.
const _: () = {
    assert!(SnapshotDirty::RS.bits() == 1 << 0);
    assert!(SnapshotDirty::STAGES.bits() == 1 << 1);
    assert!(SnapshotDirty::RT_DS.bits() == 1 << 3);
    assert!(SnapshotDirty::VS_SOURCE.bits() == 1 << 6);
    assert!(SnapshotDirty::PS_CONST_B.bits() == 1 << 17);
};

impl DeviceInner {
    /// Read the implicit front buffer into a validated caller-owned surface.
    ///
    /// Both front-buffer entry points use the persistent backbuffer and the
    /// same extent/format policy. The caller holds this device's API lock.
    pub fn read_front_buffer(&mut self, dst_surf: &Direct3DSurface9) -> i32 {
        let Some(dst_desc) = dst_surf.system_memory_blit_dst() else {
            warn_readback_rejected("GetFrontBufferData", ReadbackReject::NotSystemMemory);
            return D3DERR_INVALIDCALL;
        };
        // Front-buffer reads are approximated by the persistent back-buffer
        // texture, whose extent and layout the destination is measured against.
        let source = ReadbackSource {
            width: self.backbuffer_width,
            height: self.backbuffer_height,
            format: mtld3d_core::format::backbuffer_surface_format(
                self.present_params.back_buffer_format,
            ),
        };
        if reject_readback("GetFrontBufferData", &source, &dst_desc).is_some() {
            return D3DERR_INVALIDCALL;
        }
        let hr = blit_texture_to_systemmem(
            self,
            &SystemMemReadback::for_readback(
                self.backbuffer_handle,
                // The back-buffer texture is one image: mip 0, slice 0.
                (0, 0),
                (source.width, source.height),
                &dst_desc,
            ),
        );
        if hr == D3D_OK {
            dst_surf.note_system_memory_written();
        }
        hr
    }

    /// The scale a game-created render target or depth-stencil of this size inherits.
    ///
    /// A surface created at exactly the resolution D3D9 reports for the back
    /// buffer is the game's main view: either it renders the scene there and
    /// blits to the back buffer, or it is the depth buffer paired with it.
    /// Either way it belongs to the same image, so it scales with it — which is
    /// both what makes `render.scale` save anything for a game that does not
    /// draw straight to the back buffer, and what keeps a colour/depth pair the
    /// same size once one of them shrinks.
    ///
    /// Anything sized differently is an intermediate the game picked a
    /// resolution for (a shadow map, a glow chain, a fixed-size scratch
    /// target); its coordinates are its own and must pass through untouched.
    ///
    /// Only render targets and depth-stencils qualify. A surface the game
    /// uploads pixels into has a CPU-side layout that must keep matching what
    /// D3D9 reports, so its caller passes `false`. A lockable render target is
    /// not one of those: it is still bound and drawn into, so it takes the
    /// scale and its staging resamples past its texture, the way the lockable
    /// back buffer's does.
    pub const fn scale_for_created_target(
        &self,
        width: u32,
        height: u32,
        is_target: bool,
    ) -> mtld3d_core::render_scale::RenderScale {
        if is_target && width == self.backbuffer_width && height == self.backbuffer_height {
            self.render_scale
        } else {
            mtld3d_core::render_scale::RenderScale::IDENTITY
        }
    }

    pub const fn scissor_rect(&self) -> D3DRECT {
        self.scissor_rect
    }

    pub const fn set_scissor_rect(&mut self, r: D3DRECT) {
        self.scissor_rect = r;
    }

    pub const fn viewport(&self) -> D3DVIEWPORT9 {
        self.viewport
    }

    pub fn set_viewport(&mut self, v: D3DVIEWPORT9) {
        self.viewport = v;
        // D3D9 itself collapses a degenerate (`min_z == max_z`) or inverted
        // (`max_z < min_z`) range to `min_z .. min_z + 0.001`, so the far
        // plane forwarded to the encoder is clamped to at least
        // `min_z + 0.001`; an inverted range is not honoured on purpose.
        // `self.viewport` keeps the raw values so GetViewport round-trips
        // unchanged. Ordinary `[min_z, max_z]` ranges (`max_z >= min_z + 0.001`)
        // are left untouched.
        // The rect stays in the game's coordinate space all the way to the
        // encoder: `PassState` converts it to render resolution against
        // whichever target is bound when it becomes a Metal command. Keeping it
        // logical here is also what holds the fixed-function XYZRHW row in the
        // game's screen space.
        let (x, y, width, height) = (v.x, v.y, v.width, v.height);
        let min_z = v.min_z;
        let max_z = v.max_z.max(v.min_z + 0.001);
        self.push_control(crate::device::SetViewportOp {
            x,
            y,
            width,
            height,
            min_z,
            max_z,
        });
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
    /// The first `vs_draw::MAX_CLIP_PLANES` feed the per-draw `VsDraw`
    /// uniform when `D3DRS_CLIPPLANEENABLE` names them, so the uniform is
    /// marked for a rebuild. The index is clamped into range so an
    /// out-of-range plane aliases the last slot rather than panicking.
    pub fn set_clip_plane(&mut self, index: u32, plane: [f32; 4]) {
        let slot = (index as usize).min(CLIP_PLANE_SLOTS - 1);
        self.clip_planes[slot] = plane;
        self.mark_snapshot_dirty(SnapshotDirty::VS_DRAW);
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

    fn uses_automatic_fog(&self) -> bool {
        let shader = self.shader_bindings().pixel_shader();
        shader.is_null() || {
            // SAFETY: the non-null shader is kept live by the device binding.
            unsafe { (*shader).uses_automatic_fog() }
        }
    }

    pub const fn vertex_decl(&self) -> *mut Direct3DVertexDeclaration9 {
        self.vertex_decl.raw()
    }

    /// The elements the bound declaration passes to the pixel stage of a pre-transformed draw.
    ///
    /// Empty with no declaration bound, the state in which the FVF is 0.
    fn bound_decl_passthrough(&self) -> [u8; MAX_LINKED_INPUTS] {
        let decl = self.vertex_decl();
        if decl.is_null() {
            return [0; MAX_LINKED_INPUTS];
        }
        // SAFETY: non-null; the bound-slot refcount keeps the declaration live.
        unsafe { &*decl }.inner().passthrough()
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
            fvf,
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
    /// Bind the FVF's implicit declaration, which carries the FVF, as the
    /// current vertex declaration (the most-recent of `SetFVF` /
    /// `SetVertexDeclaration` wins). `fvf == 0` is a no-op on the binding,
    /// matching the driver.
    /// Returns whether the bound declaration changed (callers gate snapshot
    /// dirtying on this), or `None` when the declaration cannot be built,
    /// leaving the binding as it was. A call that would change nothing
    /// returns before the cache lookup (see [`Self::fvf_bind_is_redundant`]).
    #[inline]
    pub fn bind_fvf_decl(&mut self, fvf: u32) -> Option<bool> {
        if fvf == 0 || self.fvf_bind_is_redundant(fvf) {
            return Some(false);
        }
        self.bind_fvf_decl_uncached(fvf)
    }

    /// Whether binding `fvf` would leave the bound declaration as it is.
    ///
    /// True exactly when the bound declaration is `fvf`'s implicit one,
    /// answered without the cache lookup or a dereference: the bound pointer
    /// is compared against the one `bind_fvf_decl` last bound, and the FVF
    /// against the one it bound it for (`last_fvf_bind`). Each FVF has one
    /// cached declaration, so a match on both halves is the declaration the
    /// lookup would produce, and every dirty mark the caller gates on a change
    /// would be skipped anyway. Any other path that binds a declaration (a
    /// `SetVertexDeclaration`, a state-block apply) moves the pointer and
    /// fails the first half.
    #[inline]
    fn fvf_bind_is_redundant(&self, fvf: u32) -> bool {
        let (bound_fvf, bound_decl) = self.last_fvf_bind;
        self.vertex_decl.raw() == bound_decl && bound_fvf == fvf
    }

    /// The [`Self::bind_fvf_decl`] path that looks the declaration up and binds it.
    #[inline(never)]
    fn bind_fvf_decl_uncached(&mut self, fvf: u32) -> Option<bool> {
        let decl = self.get_or_create_fvf_decl(fvf);
        if decl.is_null() {
            return None;
        }
        self.last_fvf_bind = (fvf, decl);
        Some(self.replace_vertex_decl(decl))
    }

    /// Release every vertex fetch slot's bound-texture refcount at device teardown.
    pub fn teardown_vertex_textures(&mut self) {
        self.vertex_textures = [const { CachedComPtr::null() }; 4];
        self.vertex_texture_kinds = VsSamplerKinds::default();
    }

    /// Texture kind bound at each of the four vertex fetch slots.
    ///
    /// Folded into the programmable VS source, hence into the VS library and
    /// disk keys.
    #[must_use]
    pub const fn vertex_texture_kinds(&self) -> VsSamplerKinds {
        self.vertex_texture_kinds
    }

    /// One vertex-slot sampler state value, for `GetSamplerState` and capture.
    #[must_use]
    pub const fn vertex_sampler_state(&self, slot: usize, type_: usize) -> u32 {
        self.vertex_sampler_states[slot][type_]
    }

    /// The texture bound at a vertex fetch slot, for `GetTexture` and capture.
    #[must_use]
    pub const fn vertex_texture(&self, slot: usize) -> *mut crate::texture::Direct3DTexture9 {
        self.vertex_textures[slot].raw()
    }

    /// Whether this device samples `texture` unfiltered whatever its stage's filters say.
    ///
    /// Only the single-precision float formats qualify, and only on a device
    /// without 32-bit float filtering, which answers `D3DUSAGE_QUERY_FILTER`
    /// with no for them. The format test runs first, so the device's answer is
    /// read only for those three formats.
    fn samples_unfiltered(&self, texture: &crate::texture::Direct3DTexture9) -> bool {
        !mtld3d_core::format::supports_usage_query(
            texture.d3d_format(),
            D3DUSAGE_QUERY_FILTER,
            false,
            true,
        ) && !crate::direct3d9::float32_filtering_supported(self.config().deny_float32_filtering)
    }

    /// Store one vertex-slot sampler state and mirror the row to the encoder.
    pub fn set_vertex_sampler_slot_state(&mut self, slot: usize, type_: usize, value: u32) {
        self.vertex_sampler_states[slot][type_] = value;
        self.push_vertex_sampler_row(slot);
    }

    /// Mirror vertex sampler `slot`'s whole state row to the encoder.
    ///
    /// The row carries the filters the bound texture can take: a texture this
    /// device cannot filter is point-sampled, so the row is pushed again when
    /// a bind starts or ends that. It also carries the bound texture's
    /// `SetLOD`, the level a vertex `texldl` counts from, so a bind that
    /// changes it and a `SetLOD` on the bound texture push it again too.
    fn push_vertex_sampler_row(&mut self, slot: usize) {
        let mut state = self.vertex_sampler_states[slot];
        // The spare slot 0 may hold a game's write of state 0; the copy
        // carries the texture LOD there instead.
        state[TEXTURE_LOD_SLOT] = self.vertex_texture_lod(slot);
        if self.vertex_textures[slot]
            .as_ref()
            .is_some_and(|texture| self.samples_unfiltered(texture))
        {
            mtld3d_core::sampler_state::sample_unfiltered(&mut state);
        }
        self.push_control(crate::device::SetVertexSamplerOp {
            slot: u8::try_from(slot).expect("validated attachment or vertex sampler slot"),
            state,
        });
    }

    /// Bind `tex` to vertex texture fetch slot `slot` (0..4).
    ///
    /// Swaps the bound-slot refcount, moves a texture another device last
    /// used over to this one, flushes the texture's dirty mips, and mirrors
    /// the id to the encoder. Later CPU writes dirty the draw
    /// snapshot, which flushes the same slots before the next draw.
    pub fn set_vertex_texture_slot(
        &mut self,
        slot: usize,
        tex: *mut crate::texture::Direct3DTexture9,
    ) {
        let was_unfiltered = self.vertex_textures[slot]
            .as_ref()
            .is_some_and(|texture| self.samples_unfiltered(texture));
        let was_lod = self.vertex_texture_lod(slot);
        // SAFETY: `tex` is null or a live IDirect3DTexture9 supplied by the
        // calling D3D9 vtable thunk; AddRef/Release valid for our lifetime.
        self.vertex_textures[slot] = unsafe { CachedComPtr::adopt(tex) };
        // The kind decides the emitted argument type, so a swap that changes
        // it needs a fresh VS source (hence a fresh library). An unbound slot
        // reads 2D, which is the kind of the black fallback the draw binds.
        let (volume, cube) = if tex.is_null() {
            (false, false)
        } else {
            // SAFETY: non-null per the branch, and live per D3D9 lifetime
            // rules; the kind flags are read through a shared reference.
            let bound = unsafe { &*tex };
            (bound.is_volume(), bound.is_cube())
        };
        let mut kinds = self.vertex_texture_kinds;
        kinds.set_slot(slot, volume, cube);
        if kinds != self.vertex_texture_kinds {
            self.vertex_texture_kinds = kinds;
            self.mark_snapshot_dirty(SnapshotDirty::VS_SOURCE);
        }
        let id = if tex.is_null() {
            None
        } else {
            // SAFETY: non-null per the branch; live per D3D9 lifetime rules.
            let bound = unsafe { &mut *tex };
            // A texture another device last used is moved over first, so its
            // levels upload here and its id names storage on this encoder.
            crate::texture::rehydrate_for_device(bound, self);
            // Vertex texture fetch samples like a stage bind, so it ends a
            // system-memory texture's CPU-only phase the same way.
            promote_cpu_only_texture(self, bound);
            crate::texture::flush_dirty_mips(bound.inner_mut(), self);
            bound.inner_mut().note_gpu_use();
            Some(bound.texture_id())
        };
        self.push_control(crate::device::SetVertexTextureOp {
            slot: u8::try_from(slot).expect("validated attachment or vertex sampler slot"),
            id,
        });
        let unfiltered = self.vertex_textures[slot]
            .as_ref()
            .is_some_and(|texture| self.samples_unfiltered(texture));
        if was_unfiltered || unfiltered || was_lod != self.vertex_texture_lod(slot) {
            self.push_vertex_sampler_row(slot);
        }
    }

    /// The `SetLOD` of the texture bound at vertex slot `slot`, 0 for an empty slot.
    fn vertex_texture_lod(&self, slot: usize) -> u32 {
        self.vertex_textures[slot]
            .as_ref()
            .map_or(0, |texture| texture.inner().lod())
    }

    /// Push the vertex sampler row of every vertex slot `tex` is bound to.
    ///
    /// For a `SetLOD` on `tex`: the rows carry the bound texture's LOD.
    pub fn refresh_vertex_texture_lod(&mut self, tex: *const crate::texture::Direct3DTexture9) {
        for slot in 0..self.vertex_textures.len() {
            if core::ptr::eq(self.vertex_textures[slot].raw(), tex) {
                self.push_vertex_sampler_row(slot);
            }
        }
    }

    pub const fn stage_bindings(&self) -> &StageBindings {
        &self.stage_bindings
    }

    pub const fn stage_bindings_mut(&mut self) -> &mut StageBindings {
        &mut self.stage_bindings
    }

    /// The FVF `GetFVF` reports: the bound declaration's, 0 with none bound.
    ///
    /// Only the declaration `SetFVF` built for an FVF carries one; a
    /// declaration the game created reports 0.
    pub fn fvf(&self) -> u32 {
        let decl = self.vertex_decl.raw();
        if decl.is_null() {
            return 0;
        }
        // SAFETY: non-null; the bound slot's refcount keeps the declaration live.
        unsafe { (*decl).inner().fvf() }
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
    /// Used by `stamp_and_swap` (a new native frame starts with empty state),
    /// `reset_to_defaults` (every input reset), and
    /// state-block `apply_to` (touches many pieces — coarse is fine).
    pub fn mark_snapshot_dirty_all(&mut self) {
        self.snapshot_dirty.insert(SnapshotDirty::all());
    }

    /// Insert specific dirty bits for a Set* on the live state path.
    ///
    /// Cheaper than `mark_snapshot_dirty_all` — only the listed pieces
    /// get rebuilt on the next draw; clean pieces retain the native state.
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
    /// (`SetVertexShader`, `SetPixelShader`): those dirties are unconditional.
    /// Float constant setters emit ordered mirror deltas without snapshot dirties.
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

    /// Dirty bits for a transform or light write to the fixed-function state.
    ///
    /// The write always changes the FF VS constants. It changes the FF VS
    /// source only when `source_moved`: a transform write never does, since
    /// a matrix reaches neither the key nor the row count, and a light thunk
    /// compares [`FfState::vs_source_light_inputs`] across the write, since a
    /// light's parameters alone never reach the key while a light enable or a
    /// light type change can. The result then goes through
    /// [`Self::ff_aware_mask`].
    fn ff_vs_write_mask(&self, source_moved: bool) -> SnapshotDirty {
        let mut mask = SnapshotDirty::VS_CONST;
        if source_moved {
            mask |= SnapshotDirty::VS_SOURCE;
        }
        self.ff_aware_mask(mask)
    }

    /// Dirty bits outside the FF VS source and constants that a write to transform `state` needs.
    ///
    /// `SetTransform` and `MultiplyTransform` both add these to
    /// [`Self::ff_vs_write_mask`], since each rewrites the matrix in place.
    /// Active table fog keys its Z-versus-W source on the projection matrix's
    /// 4th column (`VariantKey::fog_source_w`), so a PROJECTION write rebuilds
    /// the variant; the gate on live table fog keeps vertex-fog-only games
    /// from churning the variant on every per-frame projection update. The
    /// fixed-function clip planes walk back from eye space through the
    /// inverse view the `VsDraw` uniform carries, so a VIEW write rebuilds it.
    fn transform_write_side_dirty(&self, state: u32) -> SnapshotDirty {
        match state {
            mtld3d_types::D3DTS_PROJECTION
                if self.render_states[D3DRS_FOGENABLE as usize] != 0
                    && matches!(self.render_states[D3DRS_FOGTABLEMODE as usize], 1..=3) =>
            {
                SnapshotDirty::VARIANT | SnapshotDirty::PS_SOURCE
            }
            mtld3d_types::D3DTS_VIEW => SnapshotDirty::VS_DRAW,
            _ => SnapshotDirty::empty(),
        }
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
        self.current_frame
            .push_vbib_retention(PendingVbibRetention {
                buffer_id,
                page_box,
                last_submit_seq,
            });
    }

    /// Retire a released VB/IB: its CPU backing when it has one, its device buffer otherwise.
    ///
    /// A buffer that still holds its backing goes through the retention
    /// pipeline, which pairs the backing with the encoder's cache entry. A
    /// `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` buffer gave its backing up after
    /// its upload, so nothing would reach that entry; the destroy op takes its
    /// device buffer out instead, ordered after every draw that bound it.
    pub fn retire_released_buffer(
        &mut self,
        buffer_id: BufferId,
        backing: Option<PageBox>,
        last_submit_seq: u64,
    ) {
        match backing {
            Some(page_box) => self.queue_vbib_retention(buffer_id, page_box, last_submit_seq),
            None => self.push_control(crate::device::DestroyBufferOp { buffer_id }),
        }
    }

    /// Push an inline, op-stream-ordered `Staged` VB/IB dirty-range upload.
    ///
    /// `page_box` is a transient snapshot of the dirtied bytes taken on the
    /// API thread at `Unlock`; the encoder wraps it and uploads `[0, size)`
    /// into the buffer's device buffer at `dst_offset` (renaming first if a
    /// draw earlier in the open pass already read the range, see
    /// `FrameEncoder::apply_stage_upload`). The control record preserves
    /// upload order relative to draws. Counts into `pending_retention_bytes`
    /// so the retention cap sees the transient before the encoder intakes it.
    /// Capture retains the `PageBox` and appends its fixed record without a
    /// Metal thunk on the API thread.
    pub fn push_stage_upload(
        &mut self,
        buffer_id: BufferId,
        page_box: PageBox,
        dst_offset: u32,
        size: u32,
    ) {
        self.pending_retention_bytes += page_box.len() as u64;
        self.push_control(mtld3d_core::encoder_data::StageUploadOp {
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
    /// Device capabilities, as the encoder sees them.
    pub const fn gpu_caps(&self) -> mtld3d_core::gpu_caps::GpuCaps {
        self.encoder.gpu_caps()
    }

    /// The configuration of the `IDirect3D9` that created this device.
    pub fn config(&self) -> &Mtld3dConfig {
        &self.config
    }

    pub const fn upload_coherent_seq_arc(&self) -> &Arc<AtomicU64> {
        &self.upload_coherent_seq
    }

    /// Build a fresh `FrameData` matching the device's current backbuffer / queue / layer handles.
    ///
    /// Used to seed the replacement frame when swapping at `Present` or at
    /// `flush_current_frame_blocking`. A pending `PresentationInterval` change
    /// is not put here but on the frame `stamp_and_swap` hands to the encoder,
    /// so it rides the next submission rather than the one after it.
    pub fn fresh_frame(&self) -> FrameData {
        let mut frame = FrameData::new(&FrameInit {
            device_handle: self.device_handle,
            record_handle: self.record_handle,
            backbuffer_handle: self.backbuffer_handle,
            backbuffer_srgb_handle: self.backbuffer_srgb_handle,
            backbuffer_msaa_handle: self.backbuffer_msaa_handle,
            backbuffer_msaa_srgb_handle: self.backbuffer_msaa_srgb_handle,
            backbuffer_sample_count: self.backbuffer_sample_count,
            layer_handle: self.layer_handle,
            view_handle: self.view_handle,
            // Logical, paired with the scale: `PassState::reset_frame` derives
            // the rasterized extent from the two so the conversion lives in one
            // place rather than at every producer of a frame stamp.
            backbuffer_width: self.backbuffer_width,
            backbuffer_height: self.backbuffer_height,
            backbuffer_format: mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
            render_scale: self.render_scale,
            backbuffer_contents: BackbufferContents::from_swap_effect(
                self.present_params.swap_effect,
            ),
            depth_texture: self.depth_stencil_handle,
            depth_has_stencil: depth_format_has_stencil(self.depth_stencil_format),
        });
        self.encoder.reuse_recording_storage(&mut frame);
        frame
    }

    /// Stamp per-frame counters + `submit_seq` onto `frame`, swap it in for `current_frame`.
    ///
    /// Returns the stamped outgoing frame ready to hand to the encoder and
    /// the `submit_seq` it carries. Shared between `Present` and
    /// `flush_current_frame_blocking`.
    fn stamp_and_swap(&mut self, new_frame: FrameData, no_present: bool) -> (FrameData, u64) {
        // Textures that moved to another device leave this encoder's cache
        // behind every op the outgoing frame recorded.
        for tex_id in self.departed_textures.take() {
            self.push_control(DestroyTextureOp { tex_id });
        }
        let mut frame = core::mem::replace(&mut self.current_frame, new_frame);
        // A `PresentationInterval` a Reset changed rides the first frame that
        // leaves the device after it, which is this one: the encoder applies it
        // before the frame's own `nextDrawable`. Putting it on the replacement
        // instead would hold it back until the Present after the next one.
        if let Some(pacing) = self.pending_pacing.take() {
            // Handing it over commits it: a later Reset can no longer take it
            // back, and this is the pacing the next one compares against.
            self.layer_pacing = pacing;
            frame.set_apply_pacing(Some(pacing));
        }
        // The gamma ramp rides the same frame for the same reason: the
        // encoder applies it before this frame's own present, so the first
        // frame after `SetGammaRamp` is the one that shows it.
        if let Some(change) = self.pending_gamma.take() {
            frame.set_apply_gamma(Some(change));
        }
        // A Ctrl+Shift+P run ends with the frame the closing `Present`
        // submits. A mid-frame flush sends the marked frame out early, so its
        // stop mark moves onto the continuation; the start mark stays with the
        // first piece, and the encoder keeps capturing until it sees the stop.
        if no_present {
            let carried = frame.take_carried_capture_marks(true);
            self.current_frame.mark_gpu_capture(carried);
        }
        // Keep the perf payload fields while command bytes and UP data share the frame arena.
        let op_vec_capacity_bytes = frame.op_vec_capacity_bytes();
        let op_vec_realloc_bytes = frame.take_op_vec_realloc_bytes();
        self.perf_mut()
            .drain_into_payload(frame.perf_mut(), !no_present);
        // The summary runs in `mtld3d.so`, whose `PageBox` counters and pool
        // are not this binary's, so this runtime's traffic rides the payload.
        #[cfg(perf_tracking)]
        {
            let hz = self.encoder.source_clock_hz();
            self.perf_mut().drain_pagebox_traffic(
                frame.perf_mut(),
                &crate::page_box_pool::PAGEBOX_POOL,
                hz,
            );
        }
        frame
            .perf_mut()
            .set_op_vec_metrics(op_vec_capacity_bytes, op_vec_realloc_bytes);
        // `Staged` uploads ride the op stream (inline `Op::StageUpload`),
        // so their page leases were recorded at `Unlock`. The encoder owns
        // counting their `PageBox` bytes into
        // the shared `vbib_retained_bytes` at intake.
        self.pending_retention_bytes = 0;
        frame.set_no_present(no_present);

        let this_seq = self.current_seq;
        self.current_seq = self.current_seq.saturating_add(1);
        frame.set_submit_fence(&SubmitFence {
            submit_seq: this_seq,
            coherent_seq_ptr: Arc::as_ptr(&self.coherent_seq) as u64,
            upload_coherent_seq_ptr: Arc::as_ptr(&self.upload_coherent_seq) as u64,
            failed_submit_seq_ptr: Arc::as_ptr(&self.failed_submit_seq) as u64,
        });
        frame.set_retained_bytes_ptr(Arc::as_ptr(&self.vbib_retained_bytes) as u64);
        // The next frame's packet decodes into a fresh, empty
        // `DrawReader`, and the tokens the outgoing frame's snapshot
        // held borrow that frame's command storage, released once the
        // encoder drains it. Force the API thread to re-emit every
        // snapshot section on the first draw of the new frame.
        self.snapshot_dirty = SnapshotDirty::all();
        self.reassert_saved_bindings();
        (frame, this_seq)
    }

    /// Push the saved render-target and depth bindings into `current_frame`.
    ///
    /// The fresh frame's pass state defaults to the implicit backbuffer +
    /// auto depth-stencil, but a D3D9 render-target or depth binding,
    /// including an explicit `SetDepthStencilSurface(NULL)` unbind, outlives
    /// Present, internal flushes and an automatic back-buffer resize alike.
    /// Re-asserted into the fresh frame, or the pass would carry an attachment
    /// the pipeline (built from the D3D9 snapshot) does not declare.
    fn reassert_saved_bindings(&mut self) {
        // Clone (not Copy) out of the persistent binding: `TextureInfo` and
        // the binding enums are wide aggregates, and frame swaps are rare
        // relative to draws.
        if let Some((info, scale)) = self.last_color_rt_binding.clone() {
            self.push_color_rt_binding_op(0, info, scale);
        }
        for slot in 1..RENDER_TARGET_SLOTS {
            if let Some((info, scale)) = self.last_extra_rt_bindings[slot - 1].clone() {
                self.push_color_rt_binding_op(slot, info, scale);
            }
        }
        if let Some((binding, is_sampleable, has_stencil, sample_count)) =
            self.last_depth_binding.clone()
        {
            self.push_depth_binding_op(binding, is_sampleable, has_stencil, sample_count);
        }
    }

    /// Point the saved bindings that name the implicit surfaces at their replacements.
    ///
    /// A binding saved by `SetRenderTarget` or `SetDepthStencilSurface` holds
    /// the handles and extent of the surface it bound, while the surface
    /// object itself resolves them live from the device. After
    /// `apply_auto_resize` has replaced the back buffer and the implicit
    /// depth surface, a saved back-buffer binding and a saved binding of
    /// `old_depth`, the implicit depth texture the resize retired, take the
    /// device's current handles and extent; every other binding names a
    /// surface the resize did not touch.
    fn retarget_implicit_bindings(&mut self, old_depth: MetalHandle<MTLTextureKind>) {
        let current = RtBinding::Backbuffer {
            handle: self.backbuffer_handle,
            msaa: self.backbuffer_msaa_handle,
            msaa_srgb: self.backbuffer_msaa_srgb_handle,
            sample_count: self.backbuffer_sample_count,
            width: self.backbuffer_width,
            height: self.backbuffer_height,
        };
        for saved in core::iter::once(&mut self.last_color_rt_binding)
            .chain(self.last_extra_rt_bindings.iter_mut())
            .flatten()
        {
            if matches!(saved.0, RtBinding::Backbuffer { .. }) {
                saved.0 = current.clone();
            }
        }
        if old_depth.is_null() {
            return;
        }
        if let Some((DepthBinding::Eager(handle, extent, scale), ..)) =
            self.last_depth_binding.as_mut()
            && *handle == old_depth
        {
            *handle = self.depth_stencil_handle;
            *extent = (
                scale.dimension(self.backbuffer_width),
                scale.dimension(self.backbuffer_height),
            );
        }
    }

    /// Drop what the open frame recorded on a device that can no longer send a frame.
    ///
    /// A failed device keeps recording, uploads included, and nothing would
    /// ever take the frame off it, so each `Present` starts it again from the
    /// saved bindings instead of letting it grow for as long as the
    /// application runs. Nothing in it can reach the GPU: the encoder refuses
    /// every frame of a failed device.
    #[cold]
    fn discard_unsendable_frame(&mut self) {
        let fresh = self.fresh_frame();
        drop(core::mem::replace(&mut self.current_frame, fresh));
        self.reassert_saved_bindings();
    }

    /// Submit the current frame's accumulated ops synchronously.
    ///
    /// Then continue with a fresh empty frame. Used by `LockRect` on the
    /// backbuffer and `GetRenderTargetData` to ensure the GPU has executed
    /// every draw issued this frame before the readback blit samples the
    /// backbuffer. Present is suppressed for this submission so the drawable
    /// is not consumed.
    pub fn flush_current_frame_blocking(&mut self) -> Result<(), i32> {
        self.encoder.status()?;
        self.hurry_presentation()?;
        let fresh = self.fresh_frame();
        let (frame, _) = self.stamp_and_swap(fresh, true);
        self.encoder.mid_frame_submit(frame)
    }

    /// Let the submits in flight copy the present they wait for, ahead of a flush.
    ///
    /// A flush queues behind the encoder, and the encoder may be blocked on a
    /// payload the submit thread holds while it waits for the previous
    /// present to commit, which is a wait on the display. Setting the policy
    /// here, before the flush is sent, wakes that submit into a copy so the
    /// encoder, and then the flush, go on at once. The encoder puts the policy
    /// back when the request that follows ends: the flush arm once its own
    /// submission has committed, the visibility intake once its drain is done.
    /// The one thunk this side issues off the device lifecycle, and only on a
    /// path that is already a synchronous read-back.
    pub fn hurry_presentation(&self) -> Result<(), i32> {
        if self.record_handle.is_null() {
            return Ok(());
        }
        let mut params = SetPresentWaitPolicyParams {
            record_handle: self.record_handle,
            policy: PresentWaitPolicy::SnapshotPending,
            pad0: 0,
        };
        let status = unix_call(&mut params);
        if status != D3D_OK {
            error!(target: LOG_TARGET, "encoder: presentation barrier failed {status:#x}");
            return Err(self
                .encoder
                .record_failure(status, "the presentation barrier failed"));
        }
        Ok(())
    }

    /// Queue the current frame without presenting or waiting for its submission to finish.
    ///
    /// EVENT queries observe GPU retirement separately. They only need to
    /// hand off the open frame, with the same bounded channel backpressure
    /// as Present, while the API starts recording its continuation.
    pub fn flush_current_frame_async(&mut self) -> Result<(), i32> {
        self.encoder.status()?;
        let fresh = self.fresh_frame();
        let (frame, _) = self.stamp_and_swap(fresh, true);
        self.encoder.send_frame(frame)
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
        sample_count: u8,
    ) {
        self.push_control(crate::device::BindDepthOp {
            binding,
            sample_count,
            flags: {
                let mut flags = BindDepthOpFlags::empty();
                flags.set(BindDepthOpFlags::SAMPLEABLE, is_sampleable);
                flags.set(BindDepthOpFlags::HAS_STENCIL, depth_has_stencil);
                flags
            },
        });
    }

    /// Push the encoder op that binds `info` as colour render target `slot` (0..=3).
    ///
    /// Factored out of `device_set_render_target` so `flush_current_frame_
    /// blocking` can re-assert the persistent binding into the fresh frame.
    /// Every variant of `info` carries the size D3D9 reports for the target;
    /// `scale` is what the bound resource itself is rasterized at, taken from
    /// the resource on the API thread because the operation runs on the encoder
    /// thread, which cannot reach it.
    fn push_color_rt_binding_op(
        &mut self,
        slot: usize,
        info: RtBinding,
        scale: mtld3d_core::render_scale::RenderScale,
    ) {
        self.push_control(crate::device::BindColorOp {
            slot: u8::try_from(slot).expect("validated attachment or vertex sampler slot"),
            info,
            scale,
        });
    }

    /// Unbind render target `slot` (1..=3): `SetRenderTarget(slot, NULL)`.
    ///
    /// Regenerates the mip chain of an autogen texture that was bound there,
    /// drops the persistent binding and tells the encoder.
    fn unbind_extra_render_target(&mut self, slot: usize) {
        if let Some(old_id) = self.cur_autogen_rt_ids[slot].take() {
            self.push_control(crate::device::GenerateMipmapsOrderedOp { old_id });
        }
        self.bound_rt_mut()
            .replace_render_target(slot, core::ptr::null_mut(), 0, 0);
        self.last_extra_rt_bindings[slot - 1] = None;
        self.push_control(crate::device::UnbindExtraColorOp {
            slot: u8::try_from(slot).expect("validated extra color slot"),
        });
    }

    /// Cheap retention-cap tier.
    ///
    /// Sends a synchronous `DrainRetiredNow` to the encoder so it drains
    /// retention items whose seq has already retired. No submit, no GPU
    /// wait — only useful when the encoder is sitting on drainable
    /// retention between frames. Returns when the drain completes.
    pub fn drain_retention_now(&self) -> Result<(), i32> {
        self.encoder.drain_retired_now()
    }

    /// Heavy retention-cap tier.
    ///
    /// Same submission path as `flush_current_frame_blocking` (no Present,
    /// drawable not consumed), but the encoder additionally waits for GPU
    /// completion of the submitted seq and drains retention before
    /// returning — so on return the global allocator has freed bytes that
    /// include same-frame retentions which `drain_retention_now` couldn't
    /// release.
    pub fn mid_frame_submit_for_retention(&mut self) -> Result<(), i32> {
        self.encoder.status()?;
        self.hurry_presentation()?;
        let fresh = self.fresh_frame();
        let (frame, _) = self.stamp_and_swap(fresh, true);
        self.encoder.mid_frame_submit_for_retention(frame)
    }

    /// Allocate a rename backing under the VB/IB retention cap.
    ///
    /// Recycle-pool hit, else enforce the cap, else allocate. The cap is
    /// the only mechanism bounding retained bytes: allocation itself is
    /// infallible (see `PageBox::new_uninit`), because on the 32-bit game
    /// process the allocator never fails cleanly: the process thrashes or
    /// dies long before `alloc` returns null, so reacting to a null was
    /// always too late to be the fix. The one exception is the system-memory
    /// copy a resource gets at creation, which a create answers with
    /// `E_OUTOFMEMORY` as D3D9 does; a rename is never a creation.
    pub fn alloc_pagebox_capped(&mut self, logical_len: usize) -> Result<PageBox, i32> {
        self.encoder.status()?;
        // Recycle-pool fast path: a hit is a warm, still-committed box of
        // the same padded size, allocates nothing, and therefore skips the
        // retention-cap check below (which exists to bound allocations).
        let pool = &*crate::page_box_pool::PAGEBOX_POOL;
        if let Some(b) = pool.acquire(logical_len) {
            self.perf.state_mut().bump_vbib_pool_hit();
            return Ok(b);
        }
        if pool.enabled() {
            // Misses count only while the pool is on, so the A/B baseline
            // arm reads hit=0 miss=0 rather than all-miss.
            self.perf.state_mut().bump_vbib_pool_miss();
        }
        // Before allocating, if live VB/IB retention is at the cap, drain
        // retired backings (cheap) and, if still over, force a mid-frame
        // submit + GPU-wait so this frame's renames can retire and free.
        // Bounds peak PE-heap retention well below the OOM cliff.
        if self.retention_cap_bytes != 0 {
            let retained =
                self.vbib_retained_bytes.load(Ordering::Acquire) + self.pending_retention_bytes;
            if retained >= self.retention_cap_bytes {
                self.perf.state_mut().bump_retention_cap_drain();
                self.drain_retention_now()?;
                let after =
                    self.vbib_retained_bytes.load(Ordering::Acquire) + self.pending_retention_bytes;
                if after >= self.retention_cap_bytes {
                    self.perf.state_mut().bump_retention_cap_submit();
                    self.mid_frame_submit_for_retention()?;
                }
            }
        }
        Ok(PageBox::new_uninit(logical_len))
    }

    /// Swap in a fresh frame and send the full op list to the encoder.
    ///
    /// Clears and attachment changes flow through `push_op` operations inside
    /// the frame itself, so no per-Device clear snapshot is needed.
    ///
    /// This is also where the per-frame perf counters are published:
    /// `api_thread_cycles` is stashed into the outgoing frame verbatim;
    /// `present_block_cycles` (the backpressure wait from the *previous*
    /// Present's `send_frame`) is stashed into the incoming fresh frame so
    /// the encoder's next summary can read it.
    pub fn present(&mut self) -> i32 {
        if let Err(hr) = self.encoder.status() {
            self.discard_unsendable_frame();
            return hr;
        }
        // Both `IDirect3DDevice9::Present` and the swap chain's land here, so
        // the diagnostics that run once per frame poll from this point.
        crate::capture::poll();
        // Games can present exclusively through IDirect3DSwapChain9. Cursor
        // scale, native visibility and diagnostics must follow that path too.
        let cursor_scale = self.config().cursor_scale;
        self.cursor_mut().follow_published_scale(cursor_scale);
        self.cursor_mut().sync_native_visibility();
        self.cursor_mut().note_present();
        // Input polling can deliver WM_SIZE and recreate the implicit textures.
        // Build the continuation only after those callbacks have returned.
        if self.needs_reset() {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "Present: input processing left the device requiring Reset");
            return mtld3d_types::D3DERR_DEVICENOTRESET;
        }
        self.apply_upload_answers();
        let new_frame = self.fresh_frame();
        let (frame, seq) = self.stamp_and_swap(new_frame, false);

        // The block we measure belongs to the frame that will next be
        // observed by the encoder — the one we just swapped in. The
        // `CycleSetTimer` writes into that frame's `present_block_cycles`
        // when it drops.
        let stall = CycleSetTimer::start(self.current_frame.perf_mut().present_block_cycles_ptr());
        if let Err(hr) = self.encoder.send_frame(frame) {
            return hr;
        }
        self.frame_dump_present(crate::capture::take_request(), seq);
        drop(stall);
        // Outside the stall, so the address-space walk is not charged to it.
        self.mem_watch_present();
        mtld3d_types::D3D_OK
    }

    pub fn perf_mut(&mut self) -> impl DerefMut<Target = ApiPerfState> + '_ {
        self.perf.state_mut()
    }

    /// Borrow the storage while constructing a timer that retains it.
    ///
    /// # Safety
    /// A non-null device must be live for the returned borrow, with its API
    /// lock held when multithreaded. No counter borrow may span the timer.
    pub unsafe fn perf_storage_of<'a>(dev: *mut Self) -> Option<&'a ApiPerfStorage> {
        // SAFETY: the caller guarantees the device's lifetime and API lock.
        unsafe { dev.as_ref() }.map(|dev| &dev.perf)
    }

    /// Null-tolerant wrapper: returns `null_mut()` for a null device.
    ///
    /// The subtimer finishes before the enclosing API timer, which retains
    /// the separately allocated counter state in a PERF build.
    pub fn perf_ptr_of(dev: *mut Self) -> *mut ApiPerfState {
        if dev.is_null() {
            core::ptr::null_mut()
        } else {
            // SAFETY: caller guarantees a non-null valid device pointer
            // while obtaining the counter pointer. The enclosing API timer
            // retains its storage until the subtimer has finished.
            unsafe { (*dev).perf.as_ptr() }
        }
    }

    pub const fn encoder_runtime(&self) -> u64 {
        self.encoder.runtime()
    }

    pub fn shutdown(&mut self) -> Result<(), i32> {
        self.encoder.shutdown()
    }

    pub fn encoder_status(&self) -> Result<(), i32> {
        self.recording_status()?;
        self.encoder.status()
    }

    /// Stop capture after a known failure without polling native work per draw.
    fn known_encoder_status(&self) -> Result<(), i32> {
        self.recording_status()?;
        self.encoder.known_status()
    }

    /// Stop a bound draw after a failure the device has already latched.
    ///
    /// A capture error still latched only in the frame recorder is reported by the draw's own
    /// record, which returns that error, or by `recording_status` on a path that records
    /// nothing. Either way it maps through `capture_failure`, which latches the device failure.
    fn latched_encoder_status(&self) -> Result<(), i32> {
        self.encoder.known_status()
    }

    /// Report capture failures without polling workers again within the same draw.
    fn recording_status(&self) -> Result<(), i32> {
        if let Some(error) = self.current_frame.recording_error() {
            return Err(self.capture_failure(error));
        }
        Ok(())
    }

    fn capture_failure(&self, error: mtld3d_shared::encoder_wire::WireError) -> i32 {
        let status = match error {
            mtld3d_shared::encoder_wire::WireError::AllocationFailed => mtld3d_types::E_OUTOFMEMORY,
            _ => mtld3d_types::D3DERR_DEVICELOST,
        };
        self.encoder.record_failure(status, "frame capture failed")
    }

    pub fn try_push_control<T: mtld3d_core::encoder_packet::CaptureControl>(
        &mut self,
        value: T,
    ) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        self.current_frame.try_push_control(value)
    }

    pub fn push_control<T: mtld3d_core::encoder_packet::CaptureControl>(&mut self, value: T) {
        let _ = self.current_frame.try_push_control(value);
    }

    pub fn record_draw(&mut self, draw: &DrawOp) {
        self.current_frame.record_draw(draw);
    }

    /// Capture mutable API input once for replay from the shared frame arena.
    ///
    /// # Safety
    ///
    /// Keep the returned token within this frame's operations. No copy of the
    /// token may be read after native replay releases the frame arena.
    pub unsafe fn capture_frame_bytes(&mut self, bytes: &[u8]) -> ScratchSlice {
        // SAFETY: frame ownership keeps this immutable capture live through replay.
        unsafe { arena_alloc_bytes(self.current_frame.scratch_mut(), bytes) }
    }

    /// Queue an eager `MTLTexture` create on the current frame.
    ///
    /// The encoder drains the queue at `run_frame`'s head into one batched
    /// `CreateTexturesBatch` thunk, so subsequent draw operations hit the
    /// texture cache instead of cache-missing on first bind.
    pub fn push_texture_warmup(&mut self, info: &TextureInfo) {
        self.current_frame.push_texture_warmup(info);
    }

    /// Queue an eager VB/IB `MTLBuffer` wrap on the current frame.
    ///
    /// Same drain semantics as `push_texture_warmup`.
    pub fn push_buffer_warmup(&mut self, entry: VbibWarmupEntry) {
        self.current_frame.push_buffer_warmup(entry);
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
            .insert(tex.texture_id(), ti);
    }

    /// Charge a standalone `D3DPOOL_DEFAULT` surface against the VRAM total.
    ///
    /// The surfaces `CreateRenderTarget` and `CreateDepthStencilSurface` hand
    /// out own real Metal textures without a `TextureInner`, so the texture
    /// registry never sees them; without this they would cost nothing in the
    /// figure `GetAvailableTextureMem` reports. `standalone_surface_bytes` is
    /// the same formula [`Self::deregister_standalone_surface`] refunds with,
    /// fed the dimensions, format and sample count the surface reports through
    /// `GetDesc` plus the `render.scale` its Metal textures were created at,
    /// so a multisampled or scaled surface gives back exactly what it took.
    pub fn register_standalone_surface(
        &self,
        width: u32,
        height: u32,
        d3d_format: u32,
        sample_count: u32,
        kind: mtld3d_core::format::StandaloneSurfaceKind,
        render_scale: mtld3d_core::render_scale::RenderScale,
    ) {
        self.vram_bytes_used.fetch_add(
            mtld3d_core::format::standalone_surface_bytes(
                width,
                height,
                d3d_format,
                sample_count,
                kind,
                render_scale,
            ),
            Ordering::AcqRel,
        );
    }

    /// Refund a standalone surface's bytes as it retires its Metal textures.
    ///
    /// Called from the colour and depth retire arms of `finalize_surface`,
    /// which are gated exactly as the two creation sites that charged the
    /// surface and read the scale back off the surface those sites stored it
    /// on, so the total returns to where it started.
    pub fn deregister_standalone_surface(
        &self,
        width: u32,
        height: u32,
        d3d_format: u32,
        sample_count: u32,
        kind: mtld3d_core::format::StandaloneSurfaceKind,
        render_scale: mtld3d_core::render_scale::RenderScale,
    ) {
        self.vram_bytes_used.fetch_sub(
            mtld3d_core::format::standalone_surface_bytes(
                width,
                height,
                d3d_format,
                sample_count,
                kind,
                render_scale,
            ),
            Ordering::AcqRel,
        );
    }

    /// Drop a `TextureInner` from the live-texture registry.
    ///
    /// Called from `texture_release`'s rc→0 path **before** the inner Box is
    /// freed, so the registry never holds a dangling pointer, and from
    /// `texture::rehydrate_for_device` for the device a texture migrates off,
    /// which is the only other way an entry stops belonging here.
    /// File a texture that moved to another device for this device's encoder to drop.
    ///
    /// Called under the adopting device's lock; see
    /// [`mtld3d_core::departed_textures`].
    pub fn note_departed_texture(&self, id: TextureId) {
        self.departed_textures.note(id);
    }

    /// Retire a standalone colour target's Metal textures.
    ///
    /// The encoder destroys them once the GPU is past every pass that names
    /// them. A device whose failure is latched sends no frame again, so the
    /// final `Release` destroys them instead (see `retired_while_failed`).
    pub fn retire_color_target(&mut self, retired: crate::encoder::RetiredColorTarget) {
        if self.encoder.status().is_ok() {
            self.push_control(crate::device::RetireColorOp { retired });
            return;
        }
        self.keep_retired_while_failed(&[
            retired.srgb,
            retired.base,
            retired.msaa_srgb,
            retired.msaa,
        ]);
    }

    /// Retire a standalone depth-stencil target's Metal texture, as [`Self::retire_color_target`].
    pub fn retire_depth_target(&mut self, depth: MetalHandle<MTLTextureKind>) {
        if self.encoder.status().is_ok() {
            self.push_control(crate::device::RetireDepthOp { depth });
            return;
        }
        self.keep_retired_while_failed(&[depth]);
    }

    /// File the non-null `handles` for the final `Release` to destroy.
    #[cold]
    fn keep_retired_while_failed(&mut self, handles: &[MetalHandle<MTLTextureKind>]) {
        self.retired_while_failed.extend(
            handles
                .iter()
                .filter(|handle| !handle.is_null())
                .map(|handle| handle.raw()),
        );
    }

    /// Keep the storage of a texture that moved back before its departure was drained.
    pub fn cancel_departed_texture(&self, id: TextureId) {
        self.departed_textures.cancel(id);
    }

    pub fn deregister_texture(&self, ti: *mut TextureInner) {
        // SAFETY: deregistration runs before the live inner Box is freed or
        // transferred to another device; its texture id never changes.
        let tex = unsafe { &*ti };
        let mut live = self
            .live_textures
            .lock()
            .expect("live_textures mutex poisoned");
        if live.remove(&tex.texture_id()).is_some() {
            // Release the registry lock before the VRAM accounting below;
            // it touches only the atomic, not the texture registry.
            drop(live);
            if tex.is_default_pool() {
                self.vram_bytes_used
                    .fetch_sub(tex.allocated_bytes(), Ordering::AcqRel);
            }
        }
    }

    /// `IDirect3DDevice9::EvictManagedResources` body.
    ///
    /// Walks the live-textures registry, marks every previously-uploaded mip
    /// of a `D3DPOOL_MANAGED` texture dirty (via `texture::evict_mark_dirty`),
    /// and pushes one `destroy_cached_texture` operation per affected texture.
    /// The next bind-time `flush_dirty_mips` repopulates fresh `MTLTextures`
    /// from the still-alive PE-side staging Arc, which is the spec contract
    /// "evict from VRAM, runtime re-uploads on next use". A texture of any
    /// other pool is filtered out by `evict_mark_dirty`, which is where the
    /// reason lives; its cache entry stays intact.
    pub fn evict_managed_resources(&mut self) {
        let live: Vec<*mut TextureInner> = self
            .live_textures
            .lock()
            .expect("live_textures mutex poisoned")
            .values()
            .copied()
            .collect();
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
            self.push_control(crate::device::DestroyTextureOp { tex_id });
        }
        // The replay runs in the stage walk, fragment and vertex slots alike,
        // and the walk runs only when STAGES is dirty. Without the mark, a
        // bound texture's next draw in this frame would find its Metal texture
        // gone and nothing scheduled to recreate it. Eviction is rare, so the
        // mark is not narrowed to the textures a stage holds.
        if evicted_count != 0 {
            self.mark_snapshot_dirty(SnapshotDirty::STAGES);
        }
        mtld3d_shared::log_once_info!(
            target: TEX_TRACE_TARGET,
            "EvictManagedResources: marked {evicted_count} textures dirty (cache eviction queued)"
        );
    }

    /// The queue an upload job reports a failed emission to.
    ///
    /// Cloned onto every `TextureUploadJob`, so a job that emits nothing
    /// names the device whose textures it belongs to without the encoder
    /// having to track which device it is draining for.
    pub fn upload_redirty(&self) -> Arc<RedirtyQueue> {
        Arc::clone(&self.upload_redirty)
    }

    /// Act on what the encoder made of the uploads of the frame just ended.
    ///
    /// Runs once per `Present`, before the frame is stamped: a declined
    /// upload is marked dirty again so the next frame's bind-time flush
    /// re-schedules the region, and a level that was holding its staging for
    /// the emitted answer releases it. The declines are applied first, so a
    /// level whose later upload was declined is dirty by the time its earlier
    /// release is considered and keeps the pages the retry reads. A texture
    /// released between the answer and this drain has left the registry, and
    /// its entries are dropped with it.
    fn apply_upload_answers(&self) {
        if !self.upload_redirty.has_pending() {
            return;
        }
        let declined = self.upload_redirty.take_pending();
        let released = self.upload_redirty.take_released();
        if declined.is_empty() && released.is_empty() {
            return;
        }
        let live = self
            .live_textures
            .lock()
            .expect("live_textures mutex poisoned");
        let mut restored = 0u32;
        for entry in &declined {
            let Some(&ptr) = live.get(&entry.subresource.texture_id) else {
                // Released or migrated textures no longer belong to this device.
                continue;
            };
            // SAFETY: release and migration remove entries under this registry
            // lock before freeing or transferring the inner. API calls are
            // serialized, and redirtying does not re-enter the registry.
            let ti = unsafe { &mut *ptr };
            crate::texture::redirty_declined_upload(
                ti,
                entry.face,
                entry.level as usize,
                entry.rect,
            );
            restored += 1;
        }
        for ack in &released {
            let Some(&ptr) = live.get(&ack.subresource.texture_id) else {
                // Released or migrated textures no longer belong to this device.
                continue;
            };
            // SAFETY: the registry lock keeps the inner live, API calls are
            // serialized, and staging release does not re-enter the registry.
            let ti = unsafe { &mut *ptr };
            crate::texture::release_emitted_staging(ti, ack.level as usize, ack.generation);
        }
        drop(live);
        if !declined.is_empty() {
            mtld3d_shared::log_once_info!(
                target: TEX_TRACE_TARGET,
                "upload redirty: {restored} of {} declined uploads marked dirty again",
                declined.len()
            );
        }
    }

    /// Finalize the visibility query whose END frame retires at `target_seq`.
    ///
    /// The encoder waits (via `WaitForGpuRetire` thunk → Metal
    /// `waitUntilCompleted`) only when `coherent_seq < target_seq`;
    /// otherwise it just runs intake locally. The encoder takes the request
    /// after every frame handed to it before, so the frame `target_seq` names
    /// has been encoded and committed by the time it waits. `target_seq == 0`,
    /// a query never ended, skips the round-trip.
    pub fn encoder_intake_visibility_for(&self, target_seq: u64) -> Result<(), i32> {
        self.encoder.status()?;
        if target_seq == 0 {
            return Ok(());
        }
        self.encoder.intake_visibility_for(target_seq)
    }

    pub const fn render_state(&self, index: usize) -> u32 {
        self.render_states[index]
    }

    /// Resolve the explicit RESZ request without enlarging ordinary render-state calls.
    ///
    /// This rare extension captures texture-transfer records and diagnostic temporaries.
    #[cold]
    #[inline(never)]
    fn resolve_resz_depth(&mut self) {
        let bindings = self.stage_bindings();
        let tex = bindings.texture(0);
        if tex.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "RESZ resolve requested with no stage-0 texture bound — skipped"
            );
        } else {
            // SAFETY: a bound stage holds a live texture reference until it
            // is rebound or released, both on this thread.
            let tex = unsafe { &mut *tex };
            let inner = tex.inner();
            // The resolve is a texture-to-texture copy, so the destination is
            // measured where its Metal texture lives: a depth texture created
            // at the reported back-buffer size is rasterized at
            // `render.scale` of it, exactly like the depth attachment the
            // resolve compares it against.
            let (w, h) = inner.render_extent();
            let id = tex.texture_id();
            if self.frame_dump.active {
                self.frame_dump_event(&format!("RESZ resolve → {id:?} {w}x{h}"));
            }
            let dynamic_depth = inner.d3d_usage() & D3DUSAGE_DYNAMIC != 0 && tex.is_depth_format();
            if dynamic_depth && self.depth_stencil_bound() {
                let inner = tex.inner_mut();
                crate::texture::flush_dirty_mips(inner, self);
                inner.note_gpu_use();
                let info = inner.texture_info();
                inner.mark_subresource_gpu_authoritative(0, 0);
                self.push_control(crate::device::ResolveDynamicDepthOp { id, info });
            } else if !dynamic_depth {
                let format = tex.metal_pixel_format();
                self.push_control(crate::device::ResolveDepthTextureOp { id, w, h, format });
            }
        }
    }

    /// Returns whether raw or derived state changed.
    ///
    /// Callers gate `mark_snapshot_dirty` on this. POINTSIZE also compares
    /// its numeric value and coverage latch: a state-block restore can leave
    /// either different from the component a same-raw write would select.
    ///
    /// A write of the value the slot already holds returns here, before the
    /// call into [`Self::write_render_state`]. The silent-write audit has
    /// nothing to say about it: a stored value is one the device starts or
    /// resets with, all of them defaults or consumed, or it came through the
    /// audit, which latched the slot if the value is not consumed. Only the
    /// state trace lists such a write, so the early return is taken while
    /// the trace is off.
    #[inline]
    pub fn set_render_state(&mut self, index: usize, value: u32) -> bool {
        if index != D3DRS_POINTSIZE as usize
            && self.render_states[index] == value
            && !mtld3d_core::state_trace::enabled()
        {
            return false;
        }
        self.write_render_state(index, value)
    }

    /// The [`Self::set_render_state`] path that audits the write and stores it.
    ///
    /// Out of line so the setter's callers inline only the same-value test.
    #[inline(never)]
    fn write_render_state(&mut self, index: usize, value: u32) -> bool {
        if index == D3DRS_POINTSIZE as usize {
            return self.set_point_size_render_state(value);
        }
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

    /// The POINTSIZE arm of [`Self::write_render_state`], which also decodes the A2M token.
    ///
    /// Out of line so the render-state write stays small.
    #[inline(never)]
    fn set_point_size_render_state(&mut self, value: u32) -> bool {
        self.set_point_size_state(
            value,
            mtld3d_core::multisample::numeric_point_size(value),
            mtld3d_core::multisample::a2m_control(value),
        )
    }

    pub const fn point_size(&self) -> u32 {
        self.point_size
    }

    pub const fn a2m_enabled(&self) -> bool {
        self.flags.contains(DeviceFlags::A2M_ENABLED)
    }

    /// Restore raw POINTSIZE and only the hidden components included by the caller.
    ///
    /// Recorded Capture retains membership separately from the raw value, so
    /// a refreshed numeric DWORD cannot erase a captured latch or vice versa.
    pub fn set_point_size_state(
        &mut self,
        raw: u32,
        numeric: Option<u32>,
        coverage: Option<bool>,
    ) -> bool {
        self.warn_rs_non_default_once(D3DRS_POINTSIZE as usize, raw);
        let mut changed = self.render_states[D3DRS_POINTSIZE as usize] != raw;
        self.render_states[D3DRS_POINTSIZE as usize] = raw;
        if let Some(size) = numeric {
            changed |= self.point_size != size;
            self.point_size = size;
        }
        if let Some(enabled) = coverage {
            changed |= self.a2m_enabled() != enabled;
            self.flags.set(DeviceFlags::A2M_ENABLED, enabled);
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

    /// The silent-write audit of one render-state write.
    ///
    /// The tests stay inline, so the setter can inline this; the latch mark
    /// and every log line live in cold functions.
    #[inline]
    fn warn_rs_non_default_once(&mut self, index: usize, value: u32) {
        static RS_DEFAULTS: [u32; RENDER_STATE_COUNT] = render_state_defaults();

        if index >= RENDER_STATE_COUNT {
            return;
        }
        let default = RS_DEFAULTS[index];
        if value == default {
            if mtld3d_core::state_trace::enabled() {
                Self::trace_default_rs(index, value);
            }
            return;
        }
        if self.rs_warn_fired(index) {
            return;
        }
        let state = u32::try_from(index).expect("D3DRS index fits u32 by RENDER_STATE_COUNT bound");
        if matches!(rs_classify(state, value), RsClass::Consumed) {
            if mtld3d_core::state_trace::enabled() {
                Self::trace_consumed_rs(index, value, default);
            }
            return;
        }
        self.log_unconsumed_rs_once(index, value, default);
    }

    /// Keep trace formatting off the ordinary render-state setter stack.
    #[cold]
    #[inline(never)]
    fn trace_default_rs(index: usize, value: u32) {
        log::trace!(
            target: mtld3d_core::state_trace::TARGET,
            "D3DRS_{index} = {value:#x} (default — write suppressed in warn machinery)"
        );
    }

    /// Keep consumed-state trace formatting off the ordinary setter stack.
    #[cold]
    #[inline(never)]
    fn trace_consumed_rs(index: usize, value: u32, default: u32) {
        log::trace!(
            target: mtld3d_core::state_trace::TARGET,
            "D3DRS_{index} Consumed = {value:#x} (default {default:#x})"
        );
    }

    /// Mark the once-per-slot latch of an unconsumed write, then format its diagnostic.
    #[cold]
    #[inline(never)]
    fn log_unconsumed_rs_once(&mut self, index: usize, value: u32, default: u32) {
        self.mark_rs_warn(index);
        let state = u32::try_from(index).expect("D3DRS index fits u32 by RENDER_STATE_COUNT bound");
        match rs_classify(state, value) {
            RsClass::Consumed => {} // unreachable given early-return above
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
        self.reset_targets_to_implicit();
        self.bound_buffers.teardown();
        self.stage_bindings
            .reset_to_defaults(&[mtld3d_types::sampler_state_defaults(); STAGE_COUNT]);
        // Vertex fetch slots unbind like the fragment stages; the encoder
        // mirror clears with them.
        for slot in 0..self.vertex_textures.len() {
            if !self.vertex_textures[slot].raw().is_null() {
                self.set_vertex_texture_slot(slot, core::ptr::null_mut());
            }
        }
        // The encoder's mirror of the vertex sampler rows learns of the
        // defaults only through the ops, as it does of every other write.
        for slot in 0..self.vertex_sampler_states.len() {
            self.vertex_sampler_states[slot] = mtld3d_types::sampler_state_defaults();
            self.push_vertex_sampler_row(slot);
        }
        self.replace_vertex_decl(core::ptr::null_mut());
        self.shader_bindings
            .replace_vertex_shader(core::ptr::null_mut());
        self.shader_bindings
            .replace_pixel_shader(core::ptr::null_mut());

        self.render_states = render_state_defaults();
        self.point_size = self.render_states[D3DRS_POINTSIZE as usize];
        self.flags.remove(DeviceFlags::A2M_ENABLED);
        self.ff_state = FfState::new();
        self.vs_draw = mtld3d_core::vs_draw::VsDrawState::new();
        // Every user clip plane returns to its zero default; the per-draw
        // uniform that carries them rebuilds with the rest below.
        self.clip_planes = [[0.0; 4]; CLIP_PLANE_SLOTS];
        #[cfg(perf_tracking)]
        self.perf.state_mut().advance_reset_epoch();
        // Reset abandons any open scene; a following EndScene must fail.
        self.flags.remove(DeviceFlags::IN_SCENE);
        // Scissor defaults to the full target, like the viewport reseed below.
        self.scissor_rect = mtld3d_core::render_state::full_target_scissor(
            self.backbuffer_width,
            self.backbuffer_height,
        );

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

    /// Return the targets to the implicit surfaces, as every `Reset` does.
    ///
    /// Render target 0 goes back to the back buffer, 1 to 3 are unbound, and
    /// the depth stencil goes back to the implicit surface, or to none on a
    /// swap chain without one. A plain device's `Reset` then wipes the rest of
    /// its state; an extended device's keeps it.
    fn reset_targets_to_implicit(&mut self) {
        // An autogen texture bound as a render target regenerates its mip
        // chain when it stops being one, as `SetRenderTarget` does on the way
        // off it; queued ahead of the teardown that may release it.
        let autogen = core::mem::replace(&mut self.cur_autogen_rt_ids, [None; RENDER_TARGET_SLOTS]);
        for old_id in autogen.into_iter().flatten() {
            self.push_control(crate::device::GenerateMipmapsOrderedOp { old_id });
        }
        self.bound_rt.teardown();
        // The encoder's frame reset already drops render targets 1..3; the
        // explicit unbind covers the frame in flight.
        self.last_color_rt_binding = None;
        for slot in 1..RENDER_TARGET_SLOTS {
            if self.last_extra_rt_bindings[slot - 1].take().is_some() {
                self.push_control(crate::device::UnbindExtraColorOp {
                    slot: u8::try_from(slot).expect("validated extra color slot"),
                });
            }
        }
        self.last_depth_binding = None;
        self.flags.remove(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND);
    }

    /// What an extended device's `Reset` changes: its targets and the viewport's extent.
    ///
    /// The targets return to the implicit surfaces as on a plain device. The
    /// viewport and the scissor rect cover the new back buffer, and the
    /// viewport keeps its depth range. Render states, bindings, shaders, an
    /// open scene and a recording state block all survive.
    fn rebind_after_extended_reset(&mut self) {
        self.reset_targets_to_implicit();
        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: self.backbuffer_width,
            height: self.backbuffer_height,
            min_z: self.viewport.min_z,
            max_z: self.viewport.max_z,
        };
        self.set_viewport(viewport);
        self.scissor_rect = mtld3d_core::render_state::full_target_scissor(
            self.backbuffer_width,
            self.backbuffer_height,
        );
        self.snapshot_dirty = SnapshotDirty::all();
    }

    /// Update the implicit backbuffer Metal handle after `device_reset` recreates it.
    ///
    /// The old handle is destroyed by the caller via `DestroyResourcesBulk`
    /// *before* this setter; the next `fresh_frame` stamps the new handle
    /// into the outgoing `FrameData`.
    pub const fn set_backbuffer_handle(
        &mut self,
        handle: MetalHandle<MTLTextureKind>,
        srgb_handle: MetalHandle<MTLTextureKind>,
    ) {
        self.backbuffer_handle = handle;
        self.backbuffer_srgb_handle = srgb_handle;
    }

    /// Update the back buffer's multisampled companion alongside its resolve texture.
    ///
    /// NULL when the swap chain is single-sampled. Set in the same step as
    /// [`Self::set_backbuffer_handle`] on every recreate path.
    pub const fn set_backbuffer_msaa_handle(
        &mut self,
        handle: MetalHandle<MTLTextureKind>,
        srgb_handle: MetalHandle<MTLTextureKind>,
    ) {
        self.backbuffer_msaa_handle = handle;
        self.backbuffer_msaa_srgb_handle = srgb_handle;
    }

    /// Adopt the multisample configuration a `Reset` re-specified.
    ///
    /// Set before the back buffer and implicit depth surface are recreated, so
    /// both are made at the new count, and read by `GetDesc` on the implicit
    /// surfaces afterwards.
    pub const fn set_backbuffer_multi_sample(
        &mut self,
        multi_sample_type: u32,
        multi_sample_quality: u32,
        sample_count: u8,
    ) {
        self.backbuffer_multi_sample_type = multi_sample_type;
        self.backbuffer_multi_sample_quality = multi_sample_quality;
        self.backbuffer_sample_count = sample_count;
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

    /// Queue a `PresentationInterval` change unless the layer paces that way already.
    ///
    /// A queued change is drained by `stamp_and_swap` onto the frame it hands
    /// over, which is the spec-compliant timing: a synchronous layer-property
    /// write from the API thread races the encoder's in-flight submission.
    /// The comparison against `layer_pacing` keeps a `Reset` at the pacing
    /// the device already runs on off that path entirely: the thunk, the
    /// store on the attachment record and the main-thread walk it queues
    /// would all write the pacing the layer holds. A flip between `ONE` and a
    /// divided interval keeps the vsync request and moves the ceiling, so the
    /// comparison is over the pair.
    pub const fn queue_pacing_change(&mut self, pacing: LayerPacing) {
        self.pending_pacing = mtld3d_core::present::queued_pacing(self.layer_pacing, pacing);
    }

    /// Drive the encoder thread to run `reset_cleanup`.
    ///
    /// Drain retention queues + GPU-idle wait so the caller can safely
    /// destroy the implicit backbuffer + depth/stencil `MTLTextures` and
    /// create their replacements. `retired_textures` names exactly the
    /// handles the caller destroys once this returns, so the encoder can
    /// forget them: they never reach the retention queue that prunes every
    /// other texture. Returns when the encoder has acknowledged.
    pub fn encoder_reset(&self, retired_textures: &[u64]) -> Result<(), i32> {
        self.encoder.reset(retired_textures)
    }

    /// Drop the `current_frame` left behind by `flush_current_frame_blocking`.
    ///
    /// Replace it with a fresh one carrying the device's *current*
    /// backbuffer / depth handles. Used by `device_reset` and
    /// `apply_auto_resize` after the implicit backbuffer + depth textures
    /// are recreated: without it, the next `Present` would send the stale
    /// handles the flush baked into `current_frame` and the unix-side
    /// `submit_frame` would dereference the freed `MTLTextures`. The dropped
    /// frame holds only the bindings the flush re-asserted, which may name
    /// the retired textures; the caller decides which bindings the fresh
    /// frame gets back.
    pub fn reseed_current_frame(&mut self) {
        // The replaced frame is dropped rather than submitted, so a
        // Ctrl+Shift+P run in progress hands its marks to the fresh one:
        // without that the encoder never sees the run's stop and the capture
        // ends only with the process. Every reseed follows a flush, which
        // already sent the start out with the piece it submitted, so this is
        // the stop.
        let carried = self.current_frame.take_carried_capture_marks(false);
        self.current_frame = self.fresh_frame();
        self.current_frame.mark_gpu_capture(carried);
    }

    /// The window this device took fullscreen, or `None` when it is windowed.
    pub fn fullscreen_window(&self) -> Option<*mut c_void> {
        self.fullscreen
            .as_ref()
            .map(crate::fullscreen::SavedWindow::window)
    }

    /// Take `hwnd` fullscreen: set `mode`, then borderless, covering the monitor.
    ///
    /// The z-order is left alone. The mode-set keeps the client rect and the
    /// back buffer in one space; present scales the back buffer to the
    /// display. A device created with `D3DCREATE_NOWINDOWCHANGES` leaves the
    /// window untouched, which is what the flag asks for, and still sets the
    /// mode. Called on a device that is already fullscreen it moves the
    /// session onto `hwnd`, leaving the window it was on covering the
    /// monitor.
    pub fn enter_fullscreen(
        &mut self,
        hwnd: *mut c_void,
        mode: Option<mtld3d_core::display_mode::ModeRequest>,
    ) {
        self.fullscreen = Some(crate::fullscreen::enter(hwnd, self.manages_window(), mode));
    }

    /// `true` unless the app took window management over itself.
    ///
    /// `D3DCREATE_NOWINDOWCHANGES` is the app saying it owns the device
    /// window's style, rect and visibility.
    pub const fn manages_window(&self) -> bool {
        self.creation_behavior_flags & mtld3d_types::D3DCREATE_NOWINDOWCHANGES == 0
    }

    /// Re-apply mode and window rect for a device that stays fullscreen across a `Reset`.
    ///
    /// No-op for a device that is not fullscreen: the caller decides the
    /// transition, this only carries it out.
    pub fn update_fullscreen(&mut self, mode: Option<mtld3d_core::display_mode::ModeRequest>) {
        if let Some(saved) = self.fullscreen.as_mut() {
            crate::fullscreen::update(saved, mode);
        }
    }

    /// Re-assert the mode and re-cover the monitor after the app is activated.
    ///
    /// The deferred half of the cursor subclass's `WM_ACTIVATEAPP TRUE`
    /// handling, run when the posted message is processed; a no-op when the
    /// device is no longer fullscreen.
    pub fn reactivate_fullscreen(&mut self) {
        if let Some(saved) = self.fullscreen.as_mut() {
            crate::fullscreen::reactivate(saved);
        }
    }

    /// Re-cover the monitor after an external resize of the fullscreen window.
    ///
    /// The deferred half of the cursor subclass's `WM_SIZE` handling: runs
    /// when the posted re-assert message is processed, which is native
    /// D3D9's cadence. Bounded by the guard in `fullscreen::reassert_cover`;
    /// a no-op when the device is no longer fullscreen.
    pub fn reassert_fullscreen_cover(&mut self, new_width: u32, new_height: u32) {
        if let Some(saved) = self.fullscreen.as_mut() {
            crate::fullscreen::reassert_cover(saved, (new_width, new_height));
        }
    }

    /// Give the window back. No-op unless the device is fullscreen.
    pub fn leave_fullscreen(&mut self, reason: &mtld3d_core::fullscreen_log::LeaveReason) {
        if let Some(saved) = self.fullscreen.take() {
            crate::fullscreen::leave(&saved, self.leave_kind(), reason);
        }
    }

    /// How leaving fullscreen puts this device's window back.
    pub const fn leave_kind(&self) -> crate::fullscreen::LeaveKind {
        if self.is_extended() {
            crate::fullscreen::LeaveKind::Extended
        } else {
            crate::fullscreen::LeaveKind::Plain
        }
    }

    /// Apply an implicit backbuffer resize triggered by a chrome-shrink `WM_SIZE`.
    ///
    /// Mirrors `device_reset`'s size-change pipeline (drain → destroy
    /// old textures → adopt new dims → push `drawableSize` → recreate
    /// textures → reseed `current_frame` → re-push default viewport while
    /// render target 0 is the back buffer) but **skips**
    /// `reset_to_defaults`: the game didn't request a Reset, so its render
    /// states / textures / bindings must survive.
    /// No-op when dims already match, and for a fullscreen device: its
    /// logical size is the mode the game requested, decoupled from the
    /// window, and only a `Reset` may change it. Caller drives this from
    /// the cursor subclass wndproc on the API thread; encoder is paused
    /// inside `flush_current_frame_blocking` for the destroy/create
    /// span so no in-flight cmdbuf references the freed handles.
    pub fn apply_auto_resize(&mut self, new_width: u32, new_height: u32) {
        if new_width == 0 || new_height == 0 {
            return;
        }
        if self.flags.contains(DeviceFlags::RELEASING) {
            debug!(
                target: LOG_TARGET,
                "WM_SIZE ({new_width}x{new_height}) during the device's release ignored; the \
                 back buffer is being destroyed, not resized",
            );
            return;
        }
        if self.fullscreen.is_some() {
            debug!(
                target: LOG_TARGET,
                "WM_SIZE ({new_width}x{new_height}) on a fullscreen device ignored; the back \
                 buffer keeps the requested {}x{}",
                self.backbuffer_width, self.backbuffer_height,
            );
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

        if self.flush_current_frame_blocking().is_err() {
            return;
        }
        let old_depth = self.depth_stencil_handle;
        let old_handles: [u64; 5] = [
            self.backbuffer_handle.raw(),
            self.backbuffer_srgb_handle.raw(),
            self.backbuffer_msaa_handle.raw(),
            self.backbuffer_msaa_srgb_handle.raw(),
            self.depth_stencil_handle.raw(),
        ];
        let live: Vec<u64> = old_handles.iter().copied().filter(|&h| h != 0).collect();
        // The same five leave through the same direct destroy a `Reset` uses,
        // so the encoder is told about them for the same reason.
        if self.encoder_reset(&live).is_err() {
            return;
        }
        if !live.is_empty() {
            let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
                kind: mtld3d_shared::mtl::DestroyKind::Texture,
                pad0: 0,
                handles_ptr: live.as_ptr() as u64,
                count: u32::try_from(live.len()).expect("at most 5 handles"),
                pad1: 0,
            };
            unix_call(&mut destroy);
        }

        self.set_backbuffer_dims(new_width, new_height);
        let back_buffer_made = self.recreate_implicit_surfaces(new_width, new_height);
        // Whichever way the recreate went, the textures the flush's frame and
        // the saved bindings named are gone: the reseeded frame and the
        // re-asserted bindings name what it made, or nothing where it failed.
        // The game's own bindings outlive the resize.
        self.reseed_current_frame();
        self.retarget_implicit_bindings(old_depth);
        self.reassert_saved_bindings();

        // The viewport and scissor follow render target 0 when that is the
        // back buffer the resize changed. A game drawing into its own target
        // keeps them: D3D9 changes them only at `SetRenderTarget` and `Reset`.
        // A back buffer that failed leaves them to the `Reset` it requires.
        let rt0_is_back_buffer = self
            .last_color_rt_binding
            .as_ref()
            .is_none_or(|(binding, _)| matches!(binding, RtBinding::Backbuffer { .. }));
        if back_buffer_made && rt0_is_back_buffer {
            // An extended device keeps the viewport's depth range, as its
            // `Reset` does; a plain one returns it to the default range.
            let (min_z, max_z) = if self.is_extended() {
                (self.viewport.min_z, self.viewport.max_z)
            } else {
                (0.0, 1.0)
            };
            self.set_viewport(D3DVIEWPORT9 {
                x: 0,
                y: 0,
                width: new_width,
                height: new_height,
                min_z,
                max_z,
            });
            self.scissor_rect =
                mtld3d_core::render_state::full_target_scissor(new_width, new_height);
        }
    }

    /// Create the back buffer and implicit depth texture `apply_auto_resize` needs.
    ///
    /// The old ones are already destroyed. A back buffer Metal refuses leaves
    /// every implicit handle NULL and the device requiring `Reset`; a depth
    /// texture it refuses leaves the depth handle NULL and the device drawing
    /// without depth. Each failure logs once, at the step that failed.
    /// Returns whether the back buffer was made.
    fn recreate_implicit_surfaces(&mut self, new_width: u32, new_height: u32) -> bool {
        let mut bb_params = mtld3d_shared::CreateBackbufferParams {
            device_handle: self.device_handle,
            record_handle: self.record_handle,
            width: self.render_scale.dimension(new_width),
            height: self.render_scale.dimension(new_height),
            sample_count: u32::from(self.backbuffer_sample_count),
            pad0: 0,
            texture_handle: MetalHandle::NULL,
            srgb_texture_handle: MetalHandle::NULL,
            msaa_texture_handle: MetalHandle::NULL,
            msaa_srgb_texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut bb_params);
        if status != 0 || bb_params.texture_handle.is_null() {
            error!(
                target: LOG_TARGET,
                "apply_auto_resize: CreateBackbuffer failed (0x{status:08X}) for \
                 {new_width}x{new_height} (render {}x{}) samples={} fmt={}; device unusable",
                bb_params.width,
                bb_params.height,
                bb_params.sample_count,
                self.present_params.back_buffer_format,
            );
            self.set_backbuffer_handle(MetalHandle::NULL, MetalHandle::NULL);
            self.set_backbuffer_msaa_handle(MetalHandle::NULL, MetalHandle::NULL);
            self.set_depth_stencil_handle(MetalHandle::NULL);
            self.flags.insert(DeviceFlags::NOT_RESET);
            return false;
        }
        self.set_backbuffer_handle(bb_params.texture_handle, bb_params.srgb_texture_handle);
        self.set_backbuffer_msaa_handle(
            bb_params.msaa_texture_handle,
            bb_params.msaa_srgb_texture_handle,
        );
        // The colour backing now has these dimensions even if rebuilding the
        // depth surface fails. Keep both existing and future swapchain queries
        // consistent with GetDesc on the live backbuffer.
        let mut pp = self.present_params;
        pp.back_buffer_width = new_width;
        pp.back_buffer_height = new_height;
        self.set_present_params(pp);

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
                return true;
            };
            // Render space, matching the colour attachment exactly.
            let mut ds_params = CreateDepthTextureParams {
                device_handle: self.device_handle,
                width: bb_params.width,
                height: bb_params.height,
                pixel_format,
                sample_count: u32::from(self.backbuffer_sample_count),
                texture_handle: MetalHandle::NULL,
            };
            let status = unix_call(&mut ds_params);
            if status != 0 || ds_params.texture_handle.is_null() {
                error!(
                    target: LOG_TARGET,
                    "apply_auto_resize: CreateDepthTexture failed (0x{status:08X}) — depth lost",
                );
                self.set_depth_stencil_handle(MetalHandle::NULL);
                return true;
            }
            self.set_depth_stencil_handle(ds_params.texture_handle);
        } else {
            self.set_depth_stencil_handle(MetalHandle::NULL);
        }
        true
    }
}

/// The implicit surfaces an extended device's `Reset` finds the application holding.
struct HeldImplicitSurfaces {
    back_buffer: Option<crate::surface::HeldSurface>,
    depth: Option<crate::surface::HeldSurface>,
}

// ── IDirect3DDevice9 COM object ──

/// Parameters for `Direct3DDevice9::new`.
///
/// Grouped so the constructor doesn't take a dozen positional arguments.
pub struct DeviceCreateInfo {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub record_handle: DeviceRecordHandle,
    pub view_handle: MetalHandle<NSViewKind>,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    /// The pacing that attach was called with.
    ///
    /// Seeds `DeviceInner::layer_pacing`, so the first `Reset` compares its
    /// interval against the pacing the attach put on the layer.
    pub pacing: LayerPacing,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `backbuffer_handle`; see `DeviceInner`.
    pub backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of the back buffer; see `DeviceInner`.
    pub backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion; see `DeviceInner`.
    pub backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    pub depth_stencil_handle: MetalHandle<MTLTextureKind>,
    pub depth_stencil_format: u32,
    /// Sample count of the back buffer and the implicit depth surface, 1 for none.
    pub backbuffer_sample_count: u8,
    pub backbuffer_width: u32,
    pub backbuffer_height: u32,
    /// Resolved `render.scale`, already forced to identity where unusable.
    pub render_scale: mtld3d_core::render_scale::RenderScale,
    pub encoder: EncoderThread,
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
    /// Whether this device draws its cursor through the unix-side overlay window.
    ///
    /// Resolved at attach from `cursor.software` and the layer mode; fixed for
    /// the device's lifetime.
    pub software_cursor: bool,
    /// The words the unix side publishes into for this device; see `DisplaySinks`.
    ///
    /// Created before the attach that hands their addresses over, owned by
    /// the device's cursor state from here on.
    pub display_sinks: Box<crate::cursor::DisplaySinks>,
    /// Window state saved before a fullscreen `CreateDevice` took the window over.
    ///
    /// `None` for a windowed device. The device holds it for as long as it
    /// stays fullscreen and hands it back to `fullscreen::leave` on the way
    /// out (a windowed `Reset`, or device destruction).
    pub fullscreen: Option<crate::fullscreen::SavedWindow>,
    /// The configuration of the creating `IDirect3D9`.
    pub config: Arc<Mtld3dConfig>,
    /// The flags the device starts with: `DeviceFlags::EXTENDED` for an extended device.
    pub initial_flags: DeviceFlags,
}

#[repr(C)]
pub struct Direct3DDevice9 {
    vtbl: *const IDirect3DDevice9Vtbl,
    refcount: u32,
    inner: *mut DeviceInner,
}

impl Direct3DDevice9 {
    pub fn new(mut info: DeviceCreateInfo) -> Self {
        info.encoder
            .reuse_recording_storage(&mut info.current_frame);
        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: info.backbuffer_width,
            height: info.backbuffer_height,
            min_z: 0.0,
            max_z: 1.0,
        };

        let counters = info.encoder.counters();
        let coherent_seq = counters.coherent_seq();
        let upload_coherent_seq = counters.upload_coherent_seq();
        let failed_submit_seq = counters.failed_submit_seq();
        let vbib_retained_bytes = counters.retained_bytes();

        let inner = Box::into_raw(Box::new(DeviceInner {
            device_handle: info.device_handle,
            record_handle: info.record_handle,
            view_handle: info.view_handle,
            layer_handle: info.layer_handle,
            backbuffer_handle: info.backbuffer_handle,
            backbuffer_srgb_handle: info.backbuffer_srgb_handle,
            backbuffer_msaa_handle: info.backbuffer_msaa_handle,
            backbuffer_msaa_srgb_handle: info.backbuffer_msaa_srgb_handle,
            backbuffer_multi_sample_type: info.present_params.multi_sample_type,
            backbuffer_multi_sample_quality: info.present_params.multi_sample_quality,
            backbuffer_sample_count: info.backbuffer_sample_count,
            depth_stencil_handle: info.depth_stencil_handle,
            depth_stencil_format: info.depth_stencil_format,
            flags: info.initial_flags,
            max_frame_latency: mtld3d_core::extended::DEFAULT_FRAME_LATENCY,
            backbuffer_width: info.backbuffer_width,
            backbuffer_height: info.backbuffer_height,
            render_scale: info.render_scale,
            vertex_decl: CachedComPtr::null(),
            fvf_decl_cache: rustc_hash::FxHashMap::default(),
            last_fvf_bind: (0, core::ptr::null_mut()),
            direct3d: info.direct3d,
            device_wrapper: 0,
            creation_adapter: info.creation_adapter,
            creation_device_type: info.creation_device_type,
            creation_behavior_flags: info.creation_behavior_flags,
            creation_focus_window: info.creation_focus_window,
            api_lock: (info.creation_behavior_flags & mtld3d_types::D3DCREATE_MULTITHREADED != 0)
                .then(|| &*Box::leak(Box::new(ApiLock::new()))),
            present_params: info.present_params,
            implicit_swapchain: 0,
            implicit_render_target: 0,
            implicit_depth_stencil: 0,
            encoder: info.encoder,
            current_frame: info.current_frame,
            coherent_seq,
            upload_coherent_seq,
            failed_submit_seq,
            vbib_retained_bytes,
            mem_watch: mem_watch::MemWatchState::new(),
            vram_bytes_used: Arc::new(AtomicU64::new(0)),
            outstanding_reset_blockers: AtomicU32::new(0),
            // Start at 1 so `current_seq - 1` never underflows.
            current_seq: 1,
            perf: ApiPerfStorage::new(),
            pending_retention_bytes: 0,
            retention_cap_bytes: info.config.vbib_retention_cap_bytes,
            config: info.config,
            point_size: info.render_states[D3DRS_POINTSIZE as usize],
            render_states: info.render_states,
            rs_warn_fired: [0; RENDER_STATE_COUNT.div_ceil(64)],
            ff_state: FfState::new(),
            vs_draw: mtld3d_core::vs_draw::VsDrawState::new(),
            // D3D9 default scissor rect covers the full backbuffer; like the
            // viewport, SetRenderTarget and Reset re-cover the new target.
            scissor_rect: mtld3d_core::render_state::full_target_scissor(
                info.backbuffer_width,
                info.backbuffer_height,
            ),
            viewport,
            clip_planes: [[0.0; 4]; CLIP_PLANE_SLOTS],
            cursor: CursorState::new(
                info.hwnd,
                info.cursor_scale,
                info.software_cursor,
                info.view_handle,
                info.display_sinks,
            ),
            fullscreen: info.fullscreen,
            bound_rt: BoundRt::new(info.backbuffer_width, info.backbuffer_height),
            bound_buffers: BoundBuffers::new(),
            shader_bindings: ShaderBindings::new(),
            stage_bindings: StageBindings::new(&info.sampler_states),
            vertex_textures: [const { CachedComPtr::null() }; 4],
            vertex_sampler_states: [mtld3d_types::sampler_state_defaults(); 4],
            vertex_texture_kinds: VsSamplerKinds::default(),
            last_sized_depth: None,
            recording_state_block: None,
            layer_pacing: info.pacing,
            pending_pacing: None,
            gamma_ramp: Box::new(mtld3d_core::gamma::identity_ramp()),
            pending_gamma: None,
            gamma_applied: false,
            last_color_rt_binding: None,
            last_extra_rt_bindings: [const { None }; RENDER_TARGET_SLOTS - 1],
            cur_autogen_rt_ids: [None; RENDER_TARGET_SLOTS],
            last_depth_binding: None,
            live_textures: Mutex::new(rustc_hash::FxHashMap::default()),
            departed_textures: DepartedTextures::default(),
            retired_while_failed: Vec::new(),
            upload_redirty: Arc::new(RedirtyQueue::new()),
            snapshot_dirty: SnapshotDirty::all(),
            snapshot_cache: ApiSnapshotCache::EMPTY,
            frame_dump: frame_dump::FrameDump::IDLE,
            cached_bound_texture_mask: 0,
            cached_ff_vs_layout: FfVsLayout::default(),
            cached_vs_provided_mask: u16::MAX,
            cached_stream0_extent: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_DEVICE9_VTBL.base,
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

    /// COM `AddRef` on the device wrapper, taken on a child object's behalf.
    ///
    /// Bumps the wrapper refcount directly; access is exclusive (D3D9 objects
    /// are single-threaded, or serialised by the device `ApiLock` under
    /// `D3DCREATE_MULTITHREADED`), so this matches `device_add_ref`'s effect
    /// without routing another module through the vtable thunk.
    pub const fn add_ref_self(&mut self) -> u32 {
        self.refcount += 1;
        self.refcount
    }
}

impl DeviceInner {
    pub const fn device_handle(&self) -> MetalHandle<MTLDeviceKind> {
        self.device_handle
    }

    pub const fn record_handle(&self) -> DeviceRecordHandle {
        self.record_handle
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

    /// The device's API lock as the pointer a texture keeps; null without the multithreaded flag.
    ///
    /// The lock is leaked at creation, so the pointer stays valid after this
    /// device is gone; a `D3DPOOL_MANAGED` texture that outlives the device
    /// keeps entering it.
    pub fn api_lock_ptr(&self) -> *mut ApiLock {
        self.api_lock.map_or(core::ptr::null_mut(), |lock| {
            core::ptr::from_ref(lock).cast_mut()
        })
    }

    /// The implicit backbuffer `MTLTexture`.
    ///
    /// The single drawable also backs every swapchain's `GetBackBuffer`
    /// surface.
    pub const fn backbuffer_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.backbuffer_handle
    }

    /// sRGB twin view of the back buffer, attached under `D3DRS_SRGBWRITEENABLE`.
    #[must_use]
    pub const fn backbuffer_srgb_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.backbuffer_srgb_handle
    }

    /// Multisampled companion of the back buffer, NULL when it is single-sampled.
    #[must_use]
    pub const fn backbuffer_msaa_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.backbuffer_msaa_handle
    }

    /// sRGB twin view of that companion, NULL whenever the companion is.
    #[must_use]
    pub const fn backbuffer_msaa_srgb_handle(&self) -> MetalHandle<MTLTextureKind> {
        self.backbuffer_msaa_srgb_handle
    }

    /// Sample count of the back buffer and the implicit depth surface, 1 for none.
    #[must_use]
    pub const fn backbuffer_sample_count(&self) -> u8 {
        self.backbuffer_sample_count
    }

    /// `D3DMULTISAMPLE_TYPE` the swap chain was created with.
    #[must_use]
    pub const fn backbuffer_multi_sample_type(&self) -> u32 {
        self.backbuffer_multi_sample_type
    }

    /// `MultiSampleQuality` the swap chain was created with.
    #[must_use]
    pub const fn backbuffer_multi_sample_quality(&self) -> u32 {
        self.backbuffer_multi_sample_quality
    }

    pub const fn backbuffer_width(&self) -> u32 {
        self.backbuffer_width
    }

    pub const fn backbuffer_height(&self) -> u32 {
        self.backbuffer_height
    }

    /// The scale the back buffer and everything sized with it is rasterized at.
    ///
    /// Read by the implicit surfaces, whose Metal textures the device recreates
    /// at this scale whenever it recreates them.
    pub const fn render_scale(&self) -> mtld3d_core::render_scale::RenderScale {
        self.render_scale
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

    /// Whether the bound depth-stencil surface carries a stencil plane.
    ///
    /// D3D9 silently ignores `D3DCLEAR_STENCIL` against a depth-only format
    /// (D16 / D24X8 / D32) rather than failing the call, so `Clear` masks the
    /// flag off instead of rejecting it.
    pub fn depth_stencil_has_stencil(&self) -> bool {
        if !self.depth_stencil_bound() {
            return false;
        }
        let bound = self.bound_rt.depth_stencil();
        if bound.is_null() {
            return depth_format_has_stencil(self.depth_stencil_format);
        }
        // SAFETY: non-null check passed; the bound-RT refcount holds it live.
        depth_format_has_stencil(unsafe { (*bound).standalone_format() })
    }

    /// The device's default depth-stencil format (`D3DFMT_*`), or `0` when none.
    pub const fn depth_stencil_format(&self) -> u32 {
        self.depth_stencil_format
    }

    /// Normalised present parameters the implicit swapchain reports.
    pub const fn present_params(&self) -> &D3DPRESENT_PARAMETERS {
        &self.present_params
    }

    /// `true` when presentation must wait for a successful `Reset`.
    pub const fn needs_reset(&self) -> bool {
        self.flags.contains(DeviceFlags::NOT_RESET)
    }

    /// `true` for a device an interface `Direct3DCreate9Ex` made created.
    pub const fn is_extended(&self) -> bool {
        self.flags.contains(DeviceFlags::EXTENDED)
    }

    /// The frame latency `GetMaximumFrameLatency` reports.
    pub const fn max_frame_latency(&self) -> u32 {
        self.max_frame_latency
    }

    /// Store the frame latency `SetMaximumFrameLatency` validated.
    pub const fn set_max_frame_latency(&mut self, latency: u32) {
        self.max_frame_latency = latency;
    }

    /// `true` when the device is fullscreen.
    ///
    /// `CreateAdditionalSwapChain` is rejected in that mode.
    pub const fn is_fullscreen(&self) -> bool {
        self.present_params.windowed == 0
    }

    /// The ramp the application last set, for `GetGammaRamp`.
    pub fn gamma_ramp(&self) -> D3DGAMMARAMP {
        D3DGAMMARAMP {
            red: self.gamma_ramp.red,
            green: self.gamma_ramp.green,
            blue: self.gamma_ramp.blue,
        }
    }

    /// Keep a validated ramp and queue what the present pass should do with it.
    pub fn set_gamma_ramp(&mut self, ramp: &D3DGAMMARAMP) {
        *self.gamma_ramp = D3DGAMMARAMP {
            red: ramp.red,
            green: ramp.green,
            blue: ramp.blue,
        };
        if !self.is_fullscreen() && !mtld3d_core::gamma::is_identity(&self.gamma_ramp) {
            // Native ramps the whole desktop for a windowed device, which is
            // not ours to touch, so the ramp is kept for `GetGammaRamp` and
            // changes nothing on screen. It applies as soon as a `Reset`
            // takes the device fullscreen.
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "SetGammaRamp on a windowed device: the ramp is stored and reported back, \
                 but only a fullscreen device applies one"
            );
        }
        self.queue_gamma();
    }

    /// Queue the table the layer should carry, when it is not the one it has.
    ///
    /// A non-identity ramp on a fullscreen device is queued every time it is
    /// set, since two different ramps both want applying; a ramp that stops
    /// applying is queued once, as a removal.
    fn queue_gamma(&mut self) {
        let apply = self.is_fullscreen() && !mtld3d_core::gamma::is_identity(&self.gamma_ramp);
        if apply {
            self.pending_gamma = Some(mtld3d_core::gamma::Change::Apply(Box::new(
                mtld3d_core::gamma::to_lut(&self.gamma_ramp),
            )));
            self.gamma_applied = true;
        } else if self.gamma_applied {
            self.pending_gamma = Some(mtld3d_core::gamma::Change::Remove);
            self.gamma_applied = false;
        }
    }

    /// Re-queue the ramp for a layer that has just been attached.
    ///
    /// A `Reset` that retargets the device window, or one that moves it
    /// between windowed and fullscreen, leaves the ramp where it was and the
    /// new layer carrying nothing, so what the device holds is sent again
    /// rather than assumed to be live.
    pub fn reapply_gamma(&mut self) {
        self.gamma_applied = false;
        self.pending_gamma = None;
        self.queue_gamma();
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

    /// Capture the implicit surfaces the application holds, for an extended device's `Reset`.
    ///
    /// Empty on a plain device, whose `Reset` refuses while the application
    /// holds either. Taken before the `Reset` adopts its new back-buffer size,
    /// multisample configuration or auto depth-stencil format, so each
    /// capture describes the surface the application holds.
    fn hold_implicit_surfaces(&self) -> HeldImplicitSurfaces {
        if !self.is_extended() {
            return HeldImplicitSurfaces {
                back_buffer: None,
                depth: None,
            };
        }
        HeldImplicitSurfaces {
            // SAFETY: the field is 0 or the device's live cached implicit surface.
            back_buffer: unsafe {
                crate::surface::hold_implicit_surface(self.implicit_render_target)
            },
            // SAFETY: as above.
            depth: unsafe { crate::surface::hold_implicit_surface(self.implicit_depth_stencil) },
        }
    }

    /// Detach the held implicit surfaces `held` names, for an extended device's `Reset`.
    ///
    /// Each becomes a standalone surface that keeps the textures it names,
    /// and the device forgets both the surface and its own handles to those
    /// textures, so nothing it destroys or recreates afterwards reaches them
    /// and the next `GetBackBuffer` or `GetDepthStencilSurface` hands out a
    /// new object.
    fn detach_held_implicit_surfaces(&mut self, held: HeldImplicitSurfaces) {
        if let Some(back_buffer) = held.back_buffer {
            // SAFETY: captured from the live, held cached back buffer, which
            // the device forgets together with its handles just below.
            unsafe { crate::surface::detach_implicit_surface(&back_buffer) };
            self.implicit_render_target = 0;
            self.set_backbuffer_handle(MetalHandle::NULL, MetalHandle::NULL);
            self.set_backbuffer_msaa_handle(MetalHandle::NULL, MetalHandle::NULL);
        }
        if let Some(depth) = held.depth {
            // SAFETY: as above, for the cached depth surface.
            unsafe { crate::surface::detach_implicit_surface(&depth) };
            self.implicit_depth_stencil = 0;
            self.set_depth_stencil_handle(MetalHandle::NULL);
        }
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

    /// Refresh the stored present parameters after Reset or a backbuffer resize.
    ///
    /// Back-buffer count clamped to >= 1, matching `CreateDevice`.
    pub fn set_present_params(&mut self, mut pp: D3DPRESENT_PARAMETERS) {
        pp.back_buffer_count = pp.back_buffer_count.max(1);
        let was_fullscreen = self.is_fullscreen();
        self.present_params = pp;
        // A gamma ramp only applies fullscreen, and a layer the Reset
        // re-attached carries no table, so the ramp the device holds is sent
        // again on both edges rather than assumed live.
        if was_fullscreen != self.is_fullscreen() {
            self.reapply_gamma();
        }
        // Keep the cached implicit swapchain (if it has already been handed
        // out) in lockstep: GetSwapChain(0).GetPresentParameters must reflect
        // the live geometry, not the values captured when it was created.
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
// The `SetRenderTarget` operation runs on the encoder thread and needs to
// either (1) create/fetch a Metal texture for a texture-backed RT surface,
// or (2) restore the backbuffer handle for a standalone surface. We
// capture everything it needs at call time since the surface pointer may
// be released before the operation runs.

// ── IUnknown implementation (IDirect3DDevice9) ──

/// Whether the device a child resource records as its `DeviceInner` is extended.
///
/// # Safety
/// `device_inner` is 0 or the address of a live `DeviceInner`.
pub unsafe fn device_inner_is_extended(device_inner: u64) -> bool {
    // SAFETY: the caller's contract: 0 or a live `DeviceInner`.
    unsafe { (device_inner as *const DeviceInner).as_ref() }.is_some_and(DeviceInner::is_extended)
}

/// Whether the device behind a vtable `this` is extended; `false` for a null `this`.
pub fn device_is_extended(this: *mut c_void) -> bool {
    // SAFETY: vtable thunk argument; `this` is null or *mut Direct3DDevice9 per
    // the IDirect3DDevice9 ABI.
    unsafe { InPtr::<Direct3DDevice9>::opt(this) }.is_some_and(|obj| obj.inner().is_extended())
}

/// Apply the `pSharedHandle` rules to a create, `Err` with the refusal's code.
///
/// `Ok(Some)` is the application's pointer to the level's packed pixels, which
/// only an extended device's single-level system-memory texture or offscreen
/// plain surface reads; `Ok(None)` proceeds as before. A pointer to a null
/// pointer is accepted as user memory with nothing in it, warned once.
fn shared_handle_gate(
    this: *mut c_void,
    kind: &mtld3d_core::extended::CreateKind,
    pool: u32,
    shared_handle: *mut c_void,
    entry_point: &str,
) -> Result<Option<NonNull<u8>>, i32> {
    use mtld3d_core::extended::{SharedHandleVerdict, shared_handle_verdict};
    let extended = device_is_extended(this);
    match shared_handle_verdict(kind, pool, !shared_handle.is_null(), extended) {
        SharedHandleVerdict::Proceed => Ok(None),
        SharedHandleVerdict::UserMemory => {
            // SAFETY: an extended device's user-memory contract: `shared_handle`
            // is non-null (the verdict says so) and points to the application's
            // pixel pointer.
            let memory =
                unsafe { ValueIn::<*mut c_void>::read_opt(shared_handle.cast_const().cast()) };
            let memory = memory.and_then(|memory| NonNull::new(memory.cast::<u8>()));
            if memory.is_none() {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                    "{entry_point}: user memory names a null pointer; the level starts \
                     uninitialised");
            }
            Ok(memory)
        }
        SharedHandleVerdict::Refuse(refusal) => {
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: (create_kind_key(kind) << 48)
                    | (u64::from(pool) << 32)
                    | u64::from(refusal.hresult().cast_unsigned()),
                "reject {entry_point}(pool={pool}) with a pSharedHandle → {:#010x}: {}",
                refusal.hresult(),
                refusal.reason()
            );
            Err(refusal.hresult())
        }
    }
}

/// Whether an extended device refuses `pool` for a create, warned once per kind.
fn managed_pool_refused(
    this: *mut c_void,
    kind: &mtld3d_core::extended::CreateKind,
    pool: u32,
    entry_point: &str,
) -> bool {
    let refused = mtld3d_core::pool::refused_on_extended(pool, device_is_extended(this));
    if refused {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: create_kind_key(kind),
            "reject {entry_point}(D3DPOOL_MANAGED) on an extended device → INVALIDCALL"
        );
    }
    refused
}

/// A distinct log-latch key per create kind.
const fn create_kind_key(kind: &mtld3d_core::extended::CreateKind) -> u64 {
    use mtld3d_core::extended::CreateKind;
    match kind {
        CreateKind::Texture { .. } => 0,
        CreateKind::CubeTexture => 1,
        CreateKind::VolumeTexture => 2,
        CreateKind::VertexBuffer => 3,
        CreateKind::IndexBuffer => 4,
        CreateKind::RenderTarget => 5,
        CreateKind::DepthStencil => 6,
        CreateKind::OffscreenPlain => 7,
    }
}

/// Copy an extended create's user memory into the level that backs it.
///
/// `rows` is the level's packed extent and `pitch` the row pitch of `level`.
/// A level too small for the rows is left as it was, warned once.
fn copy_user_memory(
    memory: NonNull<u8>,
    rows: &mtld3d_core::extended::PackedRows,
    level: &mut [u8],
    pitch: usize,
) {
    // SAFETY: an extended device's user-memory contract: the application's
    // pointer covers the level's tightly packed rows, which `rows` sizes from
    // the create's own extent and format, and it stays valid for the create.
    let src =
        unsafe { core::slice::from_raw_parts(memory.as_ptr().cast_const(), rows.total_bytes()) };
    if !mtld3d_core::extended::copy_packed_rows(src, rows, level, pitch) {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "user memory: {} rows of {} bytes do not fit a level of pitch {pitch} and {} bytes; \
             the level starts uninitialised",
            rows.rows, rows.row_bytes, level.len());
    }
}

/// Hold the device's API lock for the thunk; a no-op unless the app asked for `MULTITHREADED`.
///
/// Every `IDirect3DDevice9` entry point binds this as its first statement,
/// ahead of its `ApiTimer`, so the timer's drop runs under the lock and a
/// wait for the lock counts as API time, which is what the app pays. The
/// shell is leaked at teardown and its inner is not, so a refcount of zero
/// means there is no inner to read: a `Release` past zero, like a null
/// `this`, gets the no-op guard. A texture reaches the lock through the
/// pointer it carries instead ([`crate::com_ref::ComChild::enter_api_lock`]),
/// because a managed one can be called while the device is in its final
/// `Release` or gone. The unflagged fast path is a null test, a refcount
/// load, one pointer chase and a discriminant test.
#[inline]
pub fn device_api_lock(this: *mut c_void) -> ApiGuard {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return ApiGuard::NOOP;
    };
    if obj.refcount == 0 {
        return ApiGuard::NOOP;
    }
    let Some(lock) = obj.inner().api_lock else {
        return ApiGuard::NOOP;
    };
    // SAFETY: the lock was leaked at device creation and lives for the rest
    // of the process, so it outlives the guard.
    unsafe { lock.enter() }
}

/// Constructs an `ApiTimer` keyed to this device's `ApiPerfState`.
///
/// Hot path: every `IDirect3DDevice9` vtable entry calls it, the cursor
/// thunks in `cursor.rs` included. `#[inline]` suffices — release uses
/// thin-LTO so the null-check + handoff folds into the caller. The `sub`
/// arg picks which `DeviceSubCategory` bucket the elapsed cycles land in
/// for the per-sub breakdown in the 2-second summary; the top-level
/// `Device` bucket is bumped in parallel under one `rdtsc()` delta.
#[inline]
pub fn device_timer(this: *mut c_void, sub: DeviceSubCategory) -> ApiTimer {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let storage = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }).and_then(|obj| {
        // SAFETY: the entry point holds the API lock and the device is live at timer entry.
        unsafe { DeviceInner::perf_storage_of(obj.inner) }
    });
    ApiTimer::start_device(storage, sub)
}

/// Same shape as `device_timer` but for entry points whose `DeviceSubCategory` would be `Bind`.
///
/// Tags the timer with a `BindSubCategory` so the 2-second summary can
/// decompose the `Bind` row into per-Setter-family rows. The single
/// `rdtsc()` delta bumps Device top + `Bind` device-sub + the chosen
/// `BindSubCategory` in one pass.
#[inline]
fn bind_timer(this: *mut c_void, sub: BindSubCategory) -> ApiTimer {
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let storage = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }).and_then(|obj| {
        // SAFETY: the entry point holds the API lock and the device is live at timer entry.
        unsafe { DeviceInner::perf_storage_of(obj.inner) }
    });
    ApiTimer::start_bind(storage, sub)
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
    // which yields separately allocated counter storage retained by the
    // enclosing API timer for the duration of the COM call.
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

/// Pointer the `CycleAddTimer` writes into for one section's rebuild block in the snapshot.
///
/// A child of `draw_snapshot_keys_ptr` for the six sections inside the
/// `keys` scope, started only inside the section's dirty branch. Same
/// null-guard as `draw_snapshot_ptr`.
#[inline]
fn draw_snapshot_section_ptr(perf_ptr: *mut ApiPerfState, section: SnapshotSection) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_section_cycles_ptr(section) }
}

/// Pointer the `CycleAddTimer` for the `keys` scope of a sampled draw writes into.
///
/// Null on an unsampled draw (the caller passes a null `perf_ptr`), so the
/// timer reads no clock there. Same null-guard as `draw_snapshot_ptr`.
#[inline]
fn draw_snapshot_keys_sampled_ptr(perf_ptr: *mut ApiPerfState) -> *mut u64 {
    if perf_ptr.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: see `draw_snapshot_ptr`.
    unsafe { (*perf_ptr).draw_snapshot_keys_sampled_cycles_ptr() }
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    let accepted: &[Guid] = if device_is_extended(this) {
        &[
            mtld3d_types::IID_IUNKNOWN,
            mtld3d_types::IID_IDIRECT3DDEVICE9,
            mtld3d_types::IID_IDIRECT3DDEVICE9EX,
        ]
    } else {
        &[
            mtld3d_types::IID_IUNKNOWN,
            mtld3d_types::IID_IDIRECT3DDEVICE9,
        ]
    };
    // SAFETY: vtable thunk; `this`, `riid` and `ppv` are the caller's per the
    // IUnknown::QueryInterface ABI.
    unsafe {
        crate::com_ref::com_query_interface(
            this,
            riid,
            ppv,
            accepted,
            device_add_ref,
            "IDirect3DDevice9",
        )
    }
}

extern "system" fn device_add_ref(this: *mut c_void) -> u32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    // A stray Release of a device an earlier final Release tore down is the
    // no-op the refcount guard below makes it, and is not timed: the timer
    // reads the device's counters, which went with it.
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let live = unsafe { InPtr::<Direct3DDevice9>::opt(this) }.is_some_and(|obj| obj.refcount != 0);
    let _timer = live.then(|| device_timer(this, DeviceSubCategory::Misc));
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

        // From here on a window message that reaches the cursor subclass
        // finds a device being torn down, and the resize it may ask for
        // must not rebuild what the steps below destroy.
        device_inner.flags.insert(DeviceFlags::RELEASING);

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
            for &ti_ptr in live.values() {
                // SAFETY: entries in `live_textures` are removed on
                // `TextureInner` drop, so the pointer is live for the
                // duration of this loop iteration.
                let ti = unsafe { &mut *ti_ptr };
                ti.detach_from_device();
            }
        }
        let mut params = DestroyCommandQueueParams {
            device_handle: device_inner.device_handle,
            record_handle: device_inner.record_handle,
            view_handle: device_inner.view_handle,
            backbuffer_handle: device_inner.backbuffer_handle,
            depth_texture_handle: device_inner.depth_stencil_handle,
        };
        // Neither the back buffer's sRGB twin view, nor its multisampled
        // companion, nor that companion's own twin has a slot on the destroy
        // thunk; all three go out with the bulk release, issued below once the
        // encoder shutdown has waited for the GPU, so a frame the encoder or
        // the submit thread still holds keeps naming live textures. The twins
        // are listed first: each holds a retain on the texture released after
        // it.
        let implicit_handles: Vec<u64> = [
            device_inner.backbuffer_srgb_handle.raw(),
            device_inner.backbuffer_msaa_srgb_handle.raw(),
            device_inner.backbuffer_msaa_handle.raw(),
        ]
        .into_iter()
        .filter(|&handle| handle != 0)
        .collect();
        let parent = device_inner.direct3d as *mut c_void;

        // Hand the window back before the subclass goes: the restore issues a
        // `SetWindowPos`, and the game's own wndproc should see it exactly as
        // it sees any other window change.
        device_inner.leave_fullscreen(&mtld3d_core::fullscreen_log::LeaveReason::Release);

        // Leave the window's subclass *before* freeing DeviceInner; the
        // registry's back-pointer becomes dangling once we drop. The game's
        // procedure comes back with the last device on the window.
        let device_ptr = std::ptr::from_mut::<DeviceInner>(&mut device_inner);
        device_inner.cursor().uninstall_subclass(device_ptr);
        // The HCURSORs the device built die with it, now that no message can
        // realize one of them again.
        device_inner.cursor_mut().destroy_handles();

        // Release bound surfaces + buffers + textures (if any) before teardown,
        // and the objects an open `BeginStateBlock` recording holds: a
        // device-internal reference can be the last one on its object, whose
        // finalization still reaches this device.
        device_inner.recording_state_block = None;
        device_inner.bound_rt_mut().teardown();
        device_inner.bound_buffers_mut().teardown();
        device_inner.stage_bindings_mut().teardown();
        device_inner.teardown_vertex_textures();
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
        // most importantly the texture/VB destroy operations `texture_release`
        // and `buffer_release` push when the game releases its resources
        // ahead of the device. Without this flush they die with
        // `current_frame` on `drop(device_inner)` and the matching
        // MTLBuffers leak; the next CreateDevice fails to wrap the same
        // `bytesNoCopy` pages because Metal still considers them in-use.
        let _flush = device_inner.flush_current_frame_blocking();

        // Shut down encoder thread before destroying Metal resources.
        // The `Shutdown` message triggers `FrameEncoder::shutdown_cleanup`
        // on the encoder thread, which blocks (via the `WaitForGpuRetire`
        // thunk → Metal `waitUntilCompleted`) until `coherent_seq >=
        // current_submit_seq` and then drains every cache + retention
        // queue, issuing the matching destroy thunks. The wait must run
        // before the encoder thread exits — `coherent_seq`'s
        // `Arc<AtomicU64>` lives inside `DeviceInner` and drops at the
        // following `drop(device_inner)`. The same arm first waits for the
        // presenter to consume every queued frame and for its last present
        // to retire, so no present buffer reads the back buffer or the
        // layer once they go. By the time `shutdown` returns here every
        // MTLBuffer wrapping a `PageBox` the game ever Locked has been
        // released.
        if device_inner.shutdown().is_err() {
            // An unacknowledged shutdown may still read device-owned sinks and GPU resources,
            // so the device memory and its command queue stay as they are. The parent's
            // reference is the application's object, which nothing native holds (the
            // encoder decodes its own copy of the configuration), so it is given back.
            error!(target: LOG_TARGET,
                "device release: the encoder did not acknowledge shutdown; the device and its \
                 command queue are left allocated");
            std::mem::forget(device_inner);
            release_parent(parent);
            return rc;
        }

        // Targets released after the device failed never reached the encoder,
        // and the shutdown above has waited for every pass that named them.
        let retired_while_failed = core::mem::take(&mut device_inner.retired_while_failed);
        if !retired_while_failed.is_empty() {
            let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
                kind: mtld3d_shared::mtl::DestroyKind::Texture,
                pad0: 0,
                handles_ptr: retired_while_failed.as_ptr() as u64,
                count: u32::try_from(retired_while_failed.len())
                    .expect("a device's released targets fit u32"),
                pad1: 0,
            };
            unix_call(&mut destroy);
            info!(target: LOG_TARGET,
                "device release: destroyed {} Metal textures of targets released after the \
                 device failed", retired_while_failed.len());
        }
        if !implicit_handles.is_empty() {
            let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
                kind: mtld3d_shared::mtl::DestroyKind::Texture,
                pad0: 0,
                handles_ptr: implicit_handles.as_ptr() as u64,
                count: u32::try_from(implicit_handles.len()).expect("at most 3 handles"),
                pad1: 0,
            };
            unix_call(&mut destroy);
        }
        unix_call(&mut params);

        // Native observers may write the cursor state's boxed display sinks
        // until the destroy thunk unregisters the attachment under its lock.
        // Keep their backing alive until that thunk has returned.
        drop(device_inner);

        // Intentionally LEAK the small wrapper shell (vtbl ptr + refcount +
        // inner ptr — ~24 bytes) instead of freeing it. The heavy state
        // (`DeviceInner` + every Metal resource) is already torn down above and
        // its `Box` dropped; only this stub lingers. Leaking it keeps `refcount`
        // readable so a stray double-Release hits the zero-refcount guard at the
        // top of `device_release` rather than a use-after-free. The leak is
        // bounded by device-create count (one per device, ~24 bytes).
        obj.refcount = 0;
        release_parent(parent);
    }
    rc
}

/// Give back the reference a device holds on the `IDirect3D9` that created it.
///
/// Null for a device whose parent was never recorded.
fn release_parent(parent: *mut c_void) {
    if parent.is_null() {
        return;
    }
    // SAFETY: `parent` is non-null (checked above) and was AddRef'd during
    // `Direct3D9::CreateDevice`; the parent's refcount has kept it alive until
    // this Release.
    let parent_obj = unsafe { &*(parent as *const ParentIUnknown) };
    // SAFETY: `parent_obj.vtbl` is the `'static` parent vtable.
    let vtbl = unsafe { &*parent_obj.vtbl };
    (vtbl.release)(parent);
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

/// Take one reference on a device wrapper on behalf of a child object.
///
/// Two callers. The implicit swapchain and the implicit render-target /
/// depth-stencil surfaces are device-owned: each holds exactly one reference
/// on the device while its own public refcount is non-zero, acquired here on
/// the child's 0→1 transition. `GetDevice` takes one as well, on behalf of the
/// application, which releases it. `wrapper` is the `Direct3DDevice9`* from
/// [`DeviceInner::device_wrapper`]; a null wrapper (device not yet stamped) is
/// a no-op.
pub fn device_wrapper_add_ref(wrapper: *mut c_void) {
    if wrapper.is_null() {
        return;
    }
    // SAFETY: `wrapper` is the live `Direct3DDevice9` that owns the forwarding
    // child; access is exclusive (D3D9 objects are single-threaded, or
    // serialised by the device `ApiLock` under `D3DCREATE_MULTITHREADED`), so
    // the transient exclusive borrow to bump the refcount is sound.
    unsafe { (*wrapper.cast::<Direct3DDevice9>()).add_ref_self() };
}

/// Record that the app acquired (`true`) or dropped (`false`) its reference to a `Reset` blocker.
///
/// Fired by the COM engine on the public 0↔1 edge of a child whose
/// `blocks_reset_while_referenced` is set; `wrapper` is that child's
/// forwarding device (null when the child has none, a no-op).
pub fn device_wrapper_note_reset_blocker(wrapper: *mut c_void, acquired: bool) {
    if wrapper.is_null() {
        return;
    }
    // SAFETY: `wrapper` is the live `Direct3DDevice9` that owns the forwarding
    // child; the counter is atomic, so a shared borrow suffices.
    let counter = unsafe {
        &(*wrapper.cast::<Direct3DDevice9>())
            .inner()
            .outstanding_reset_blockers
    };
    if acquired {
        counter.fetch_add(1, Ordering::AcqRel);
    } else {
        counter.fetch_sub(1, Ordering::AcqRel);
    }
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // No exclusive mode is ever taken, so focus changes never lose the device.
    // The non-OK answers are the failure latch a step the layer could not
    // complete leaves behind (a failed encode, submission or capture) and the
    // DEVICENOTRESET a failed implicit-resource rebuild leaves.
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let object = unsafe { InPtr::<Direct3DDevice9>::opt(this) };
    if let Some(obj) = &object
        && let Err(hr) = obj.inner().encoder_status()
    {
        return hr;
    }
    // An extended device is never left owing a `Reset` by a rejected one, so
    // on it DEVICENOTRESET only reports a back buffer the layer could not
    // rebuild.
    let not_reset = object.is_some_and(|obj| obj.inner().needs_reset());
    if not_reset {
        mtld3d_types::D3DERR_DEVICENOTRESET
    } else {
        D3D_OK
    }
}

extern "system" fn device_get_available_texture_mem(this: *mut c_void) -> u32 {
    // Unified memory has no dedicated-VRAM answer; modern IHV drivers report a
    // configured budget here. Report `BUDGET - live DEFAULT-pool bytes` so the
    // value visibly decreases as the app allocates GPU resources, while
    // staying generous enough never to starve a real workload. 2 GiB budget
    // (fits u32; well above any title's needs).
    //
    // `memory.vramBudgetMB` lowers that ceiling, and defaults to doing so on a
    // 32-bit guest. An engine of this era sizes its streaming pool from what
    // this call reports, so an unrestricted figure invites a title to commit
    // past the process address space; what fails then is not an allocation it
    // could handle but a Metal command buffer, out of memory, whose rendering
    // is discarded.
    const VRAM_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return 0;
    };
    let dev = obj.inner();
    let budget = match dev.config().vram_budget_cap_bytes {
        0 => VRAM_BUDGET,
        cap => VRAM_BUDGET.min(cap),
    };
    let used = dev.vram_bytes_used.load(Ordering::Acquire);
    // An extended device pages its default-pool memory, so its figure does
    // not shrink as the application allocates.
    let available = mtld3d_core::extended::available_texture_mem(budget, used, dev.is_extended());
    // Games size their texture budgets from this call or from DXGI; the
    // one-time line tells which path a title took when its settings menu
    // shows a surprising video-memory figure.
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "IDirect3DDevice9::GetAvailableTextureMem → {} MiB (first call)",
        available >> 20
    );
    available
}

extern "system" fn device_evict_managed_resources(this: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetDeviceCaps()");
    if caps.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `caps` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `D3DCAPS9` slot owned by the caller.
    caps::fill(
        unsafe { &mut *caps },
        obj.inner().config().caps_all,
        crate::direct3d9::sampler_border_supported(),
    );
    0 // S_OK
}

extern "system" fn device_get_display_mode(
    this: *mut c_void,
    swap_chain: u32,
    mode: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
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
        *mode.cast::<D3DDISPLAYMODE>() =
            crate::direct3d9::reported_display_mode(obj.inner().present_params());
    }
    D3D_OK
}

extern "system" fn device_get_creation_parameters(this: *mut c_void, params: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Nowhere to hand the chain to: refused before it is made, since a chain
    // nobody holds would keep its device reference for good.
    if swap_chain.is_null() {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject CreateAdditionalSwapChain(ppSwapChain = NULL) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
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
    crate::direct3d9::resolve_backbuffer_dims(target_window as u64, &mut pp);
    if pp.back_buffer_format == 0 {
        pp.back_buffer_format = crate::direct3d9::adapter_display_format();
    }
    pp.back_buffer_count = pp.back_buffer_count.max(1);
    pp_in.back_buffer_width = pp.back_buffer_width;
    pp_in.back_buffer_height = pp.back_buffer_height;
    pp_in.back_buffer_count = pp.back_buffer_count;
    pp_in.back_buffer_format = pp.back_buffer_format;
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
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(target: LOG_TARGET, "IDirect3DDevice9::GetNumberOfSwapChains()");
    1
}

extern "system" fn device_reset(this: *mut c_void, present_params: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    reset_impl(this, present_params, ResetCall::Reset)
}

/// Which entry point a `Reset` came through.
#[derive(Clone, Copy)]
pub enum ResetCall {
    /// The base `Reset` slot, on either kind of device.
    Reset,
    /// `ResetEx`, with the size of the display mode it named, if any.
    ResetEx { mode: Option<(u32, u32)> },
}

/// What every line one `Reset` writes shares.
struct ResetLog {
    /// `IDirect3DDevice9` on a plain device, `IDirect3DDevice9Ex` on an extended one.
    interface: &'static str,
    method: ResetMethod,
    /// The back-buffer size before the call.
    old_size: (u32, u32),
    /// The device was waiting for a successful `Reset` when the call came.
    recovering: bool,
}

impl ResetLog {
    /// The line for this call with `pp` as resolved so far and its `outcome`.
    const fn summary<'a, O: core::fmt::Display>(
        &self,
        pp: &'a mtld3d_types::D3DPRESENT_PARAMETERS,
        outcome: O,
    ) -> ResetSummary<'a, O> {
        ResetSummary::new(self.interface, self.old_size, pp, outcome).via(self.method)
    }
}

/// The body `Reset` and `ResetEx` share; the caller holds the API lock.
///
/// A plain device's `Reset` refuses while the application references a
/// default-pool resource or an implicit surface, returns every state to its
/// default, and leaves a rejected or failed device requiring another `Reset`.
/// An extended device's skips the reference check, keeps its state, its
/// default-pool resources and an open scene, rebinds only its targets and the
/// viewport extent, and answers a rejected request without changing anything.
fn reset_impl(this: *mut c_void, present_params: *mut c_void, call: ResetCall) -> i32 {
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
    // What the line a Reset writes compares against, and whether the device
    // already waits for a successful Reset: a game retries a failing one
    // every frame, and a success then is a recovery.
    let log = ResetLog {
        interface: if dev.is_extended() {
            "IDirect3DDevice9Ex"
        } else {
            "IDirect3DDevice9"
        },
        method: match call {
            ResetCall::Reset => ResetMethod::Reset,
            ResetCall::ResetEx { .. } => ResetMethod::ResetEx,
        },
        old_size: (dev.backbuffer_width, dev.backbuffer_height),
        recovering: dev.needs_reset(),
    };
    let previous = *dev.present_params();
    // `ResetEx` names a display mode exactly when it asks for fullscreen, of
    // the back buffer's size; one that disagrees is refused before anything
    // else is looked at.
    if let ResetCall::ResetEx { mode } = call
        && !mtld3d_core::extended::reset_ex_mode_valid(
            pp_in.windowed != 0,
            mode,
            (pp_in.back_buffer_width, pp_in.back_buffer_height),
        )
    {
        let pp = *pp_in;
        let reason = match mode {
            Some((width, height)) if pp.windowed != 0 => {
                format!("a windowed request names display mode {width}x{height}")
            }
            Some((width, height)) => format!(
                "display mode {width}x{height} is not the back buffer's {}x{}",
                pp.back_buffer_width, pp.back_buffer_height
            ),
            None => String::from("a fullscreen request names no display mode"),
        };
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!("rejected, {reason}"),
            ResetFailureEffect::Unchanged,
        );
        return D3DERR_INVALIDCALL;
    }
    if let Err(hr) = dev.encoder_status() {
        return hr;
    }

    // Resolve the request on a local copy. A windowed Reset may pass zero
    // dimensions ("use the device window's client rect") and
    // `D3DFMT_UNKNOWN` ("use the display format"); a zero back-buffer count
    // resolves to one. D3D9 reports the resolved values back to the caller.
    let mut pp = *pp_in;
    // Reject invalid swap-effect / back-buffer-count / presentation-interval
    // before touching any device state, so a rejected Reset leaves the device
    // intact and resettable.
    let extended = dev.is_extended();
    if !mtld3d_core::present::present_params_are_valid(&pp, extended) {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!(
                "rejected, the swap effect, back buffer count or presentation interval is invalid"
            ),
            ResetFailureEffect::Unchanged,
        );
        return D3DERR_INVALIDCALL;
    }
    // A Reset with well-formed parameters ends an open `BeginStateBlock`
    // recording whether or not it goes on to succeed, before it looks for
    // outstanding references: the recording dies with the state it was
    // recording against. Every rejection from here on goes through
    // `reject_reset`, which restores the state defaults as well. An extended
    // device keeps the recording, as it keeps the state it records against.
    if !extended {
        dev.recording_state_block = None;
    }
    // A fullscreen request must still be well-formed even though its size is
    // not used: the D3D9 "zero means the client area" rule is windowed-only,
    // so zero dimensions here are a malformed request. Checked before the
    // window moves, so a rejected Reset leaves the window as it was.
    if pp.windowed == 0 && (pp.back_buffer_width == 0 || pp.back_buffer_height == 0) {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!("rejected, a fullscreen request may not carry zero dimensions"),
            ResetFailureEffect::Unchanged,
        );
        return reject_reset(dev);
    }
    // Reset rejects any outstanding app reference to a `D3DPOOL_DEFAULT`
    // resource or an implicit surface: those are backed by the device memory
    // the Reset recreates, and D3D9 makes the app release them first. A state
    // block that holds such a resource keeps it outstanding too, since the
    // resource lives as long as the block; the device's bindings do not count,
    // so they need not be released before the count is read.
    // An extended device keeps its default-pool resources and implicit
    // surfaces alive across `Reset`, so nothing it holds blocks one.
    let blockers = dev.outstanding_reset_blockers.load(Ordering::Acquire);
    if blockers != 0 && !extended {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!(
                "rejected, {blockers} D3DPOOL_DEFAULT resource(s) or implicit surface(s) still \
                 referenced"
            ),
            ResetFailureEffect::Unchanged,
        );
        return reject_reset(dev);
    }
    if pp.windowed != 0 && pp.back_buffer_format == 0 {
        pp.back_buffer_format = crate::direct3d9::adapter_display_format();
    }
    crate::direct3d9::warn_unsupported_backbuffer_format(pp.back_buffer_format);
    // `Reset` re-specifies the swap chain, multisample configuration
    // included. Its validation depends on the format and device caps alone,
    // so reject it before changing the window or display mode.
    let Ok(new_sample_count) = mtld3d_core::multisample::resolve_sample_count(
        pp.multi_sample_type,
        pp.multi_sample_quality,
        pp.back_buffer_format,
        crate::direct3d9::device_caps_flags(),
    )
    .map(|count| u8::try_from(count).expect("sample count ≤ 16 fits u8")) else {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!(
                "rejected, the multisample type and quality are not available on this back-buffer \
                 format"
            ),
            ResetFailureEffect::Unchanged,
        );
        return reject_reset(dev);
    };
    // The auto depth-stencil format answers to the same depth and multisample
    // rules as at `CreateDevice`, also before the window or the mode moves.
    if pp.enable_auto_depth_stencil != 0
        && pp.auto_depth_stencil_format != 0
        && !mtld3d_core::multisample::auto_depth_stencil_accepts(
            pp.auto_depth_stencil_format,
            pp.multi_sample_type,
            pp.multi_sample_quality,
            crate::direct3d9::device_caps_flags(),
        )
    {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!(
                "rejected, the auto depth-stencil format at this multisample type is no \
                 depth-stencil the device offers"
            ),
            ResetFailureEffect::Unchanged,
        );
        return reject_reset(dev);
    }
    let target_window = if pp.device_window == 0 {
        dev.window()
    } else {
        pp.device_window
    };
    let retargeted = target_window != dev.window();
    if !resolve_reset_window_mode(dev, target_window, &mut pp) {
        log_failed_reset(
            dev,
            &log,
            &pp,
            format_args!("rejected, no usable windowed client area"),
            ResetFailureEffect::Unchanged,
        );
        return reject_reset(dev);
    }
    pp.back_buffer_count = pp.back_buffer_count.max(1);
    warn_present_params_fields_once(&pp);

    // Report resolved geometry only after validation. The caller's window
    // and mode flags stay as supplied.
    pp_in.back_buffer_width = pp.back_buffer_width;
    pp_in.back_buffer_height = pp.back_buffer_height;
    pp_in.back_buffer_count = pp.back_buffer_count;
    pp_in.back_buffer_format = pp.back_buffer_format;

    // Captured before the configuration below changes what the held
    // surfaces resolve to.
    let held = dev.hold_implicit_surfaces();
    let detached = (held.back_buffer.is_some(), held.depth.is_some());
    let multi_sample_changed = new_sample_count != dev.backbuffer_sample_count;
    dev.set_backbuffer_multi_sample(
        pp.multi_sample_type,
        pp.multi_sample_quality,
        new_sample_count,
    );
    // A back buffer an earlier `Reset` failed to create takes the recreate
    // path at any size: the dimensions that `Reset` adopted are not a back
    // buffer that exists.
    // An extended device's `Reset` with an implicit surface the application
    // still holds makes the swap chain new textures at any size: the held
    // surface keeps the old ones (`detach_held_implicit_surfaces`).
    // Only the first four are a change in what the device presents with,
    // which the line's level keys on; a held surface alone is not.
    let recreated_for_a_change = pp.back_buffer_width != dev.backbuffer_width
        || pp.back_buffer_height != dev.backbuffer_height
        || multi_sample_changed
        || dev.backbuffer_handle.is_null();
    let resized = recreated_for_a_change || held.back_buffer.is_some() || held.depth.is_some();
    // Reset adopts the present params' auto depth-stencil configuration: an
    // enabled flag (re)creates the implicit depth-stencil at the given format,
    // a disabled flag drops it. This is independent of a resize, so resolve the
    // target format up front and apply it on both paths below.
    let new_depth_format = if pp.enable_auto_depth_stencil != 0 {
        pp.auto_depth_stencil_format
    } else {
        0
    };
    // A `Reset` may name another `hDeviceWindow`. The presentation surface
    // and the window subclass both live on the window the device attached, so
    // they move across before anything is recreated against the new one.
    if retargeted {
        // A pending change from an earlier Reset would otherwise ride the
        // frame the retarget flush below sends, which still names the layer
        // the device is leaving. The fresh attach receives this Reset's
        // pacing directly.
        dev.pending_pacing = None;
        if let Err(hr) = retarget_device_window(dev, &pp, target_window) {
            log_failed_reset(
                dev,
                &log,
                &pp,
                format_args!("failed retargeting the device window ({hr:#010x})"),
                ResetFailureEffect::Unstated,
            );
            return hr;
        }
    }

    if resized {
        // reset_recreate_resources rebuilds the depth from depth_stencil_format,
        // so adopt the new auto-DS format before it runs.
        dev.depth_stencil_format = new_depth_format;
        if let Err(hr) = reset_recreate_resources(dev, &pp, held) {
            log_failed_reset(
                dev,
                &log,
                &pp,
                format_args!("failed recreating the back buffer ({hr:#010x})"),
                ResetFailureEffect::StateKeptResetOwed,
            );
            return fail_reset(dev, hr);
        }
    } else {
        // Skip flush + destroy + recreate + setDrawableSize entirely. The game
        // called Reset for state-clobber reasons after our `apply_auto_resize`
        // already matched the back-buffer to the new client size (or the game
        // Reset with identical dims). Re-issuing the recreate cycle would be a
        // wasteful no-op — up to ~tens of ms per Reset depending on GPU
        // workload depth. State-defaults + reseed still run unconditionally
        // (`Reset` always clobbers state per spec).
        // The implicit depth-stencil is still reconciled, since the
        // EnableAutoDepthStencil flag can flip without a resize (a no-op when
        // it is unchanged, so the fast path stays fast).
        if let Err(hr) = reconcile_implicit_depth(dev, new_depth_format) {
            log_failed_reset(
                dev,
                &log,
                &pp,
                format_args!("failed recreating the auto depth-stencil ({hr:#010x})"),
                ResetFailureEffect::StateKeptResetOwed,
            );
            return fail_reset(dev, hr);
        }
        // Deliver every op queued since the last Present before the reseed
        // below replaces `current_frame`. The queue holds work whose
        // bookkeeping already advanced (texture and Staged-buffer uploads
        // scheduled at bind time cleared their dirty bits when they were
        // queued), so dropping it loses that content on the GPU side for
        // good: the game believes it uploaded and never rewrites it. HL2's
        // cached VGUI text meshes rode exactly this queue through its
        // same-size windowed/fullscreen Reset, which is issue #76's garbled
        // menu text. The resized path flushes inside
        // `reset_recreate_resources`.
        if let Err(hr) = dev.flush_current_frame_blocking() {
            log_failed_reset(
                dev,
                &log,
                &pp,
                format_args!("failed delivering the pending frame ({hr:#010x})"),
                ResetFailureEffect::Unstated,
            );
            return hr;
        }
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

    // 7. Reseed `current_frame` so it carries the new backbuffer/depth
    //    handles. The post-flush frame still referenced the destroyed
    //    pre-Reset textures; without this, the next Present would
    //    submit a freed MTLTexture pointer (status=0xc0000005 on the
    //    unix side). Runs before the state defaults: `reset_to_defaults`
    //    pushes ops (default viewport, unbinds), and pushing them first
    //    would hand them to the reseed to throw away, so this keeps the
    //    order `apply_auto_resize` already uses.
    dev.reseed_current_frame();
    // The fresh attach's layer carries no gamma table. The ramp is queued only
    // now that `current_frame` names that layer: the frames the flushes above
    // sent still named the layer the retarget detached, and a ramp riding one
    // of them would be dropped for want of an attachment.
    if retargeted {
        dev.reapply_gamma();
    }

    // 8. Reset device state to D3D9 defaults. Cursor + silent-write
    //    warn latches survive (per-spec / process-lifetime telemetry). An
    //    extended device keeps its state and rebinds only its targets.
    if extended {
        dev.rebind_after_extended_reset();
    } else {
        dev.reset_to_defaults();
    }
    dev.flags.remove(DeviceFlags::NOT_RESET);

    // 9. On the existing attachment, defer the PresentationInterval change
    //    to a later frame's first `nextDrawable`. Mutating the attachment
    //    synchronously here races the encoder's in-flight submission. A
    //    retarget's fresh attach already latched these presentation params.
    //    A Reset that leaves the interval where it was queues nothing: the
    //    layer would be written the value it holds, and the write costs a
    //    thunk, a store on the attachment record and a main-thread walk.
    if !retargeted && !dev.layer_handle.is_null() {
        dev.queue_pacing_change(resolve_layer_pacing(&pp, &dev.config));
    }

    // Info only for a Reset that changed what the device presents with, that
    // recovered it, or that ends an extended device's run of failed Resets
    // (`success_logs_at_info`). A game may call Reset on every step of a
    // window drag after `apply_auto_resize` already followed the client area,
    // and those same-size calls stay at debug, held surfaces or not.
    let mut done = ResetDone::empty();
    done.set(ResetDone::RESIZED, resized);
    done.set(ResetDone::RECOVERED, log.recovering);
    done.set(ResetDone::EXTENDED, extended);
    done.set(ResetDone::BACK_BUFFER_DETACHED, detached.0);
    done.set(ResetDone::DEPTH_DETACHED, detached.1);
    // A device that was waiting for this Reset says it recovered; one that
    // owed nothing, an extended device past a rejection, says the run ended.
    done.set(
        ResetDone::FAILURES_ENDED,
        !log.recovering && dev.flags.contains(DeviceFlags::RESET_FAILURE_LOGGED),
    );
    dev.flags.remove(DeviceFlags::RESET_FAILURE_LOGGED);
    let changed = recreated_for_a_change || changes_presentation(&previous, &stored);
    let level = if success_logs_at_info(changed, done) {
        log::Level::Info
    } else {
        log::Level::Debug
    };
    log::log!(target: LOG_TARGET, level, "{}", log.summary(&pp, done));
    D3D_OK
}

/// Log a `Reset` that did not complete, with the request as far as it was resolved.
///
/// At warn for the first failure since the device last reset successfully,
/// and at debug after it: a game retries a failing `Reset` every frame, and
/// the first line already said why it fails. A plain device is past its
/// first failure while it waits for a successful `Reset`; a rejection of
/// malformed present parameters leaves it as it was, so that one warns each
/// time unless an earlier failure is still pending. An extended device owes
/// no `Reset` after a rejection, so it marks its first warning instead and
/// keeps later rejections at debug, but a failure that leaves it needing a
/// `Reset` warns regardless (`failure_warns`). Its line ends with `effect`,
/// what the failure left behind.
fn log_failed_reset(
    dev: &mut DeviceInner,
    log: &ResetLog,
    pp: &mtld3d_types::D3DPRESENT_PARAMETERS,
    reason: core::fmt::Arguments<'_>,
    effect: ResetFailureEffect,
) {
    let extended = dev.is_extended();
    let failure_logged = dev.flags.contains(DeviceFlags::RESET_FAILURE_LOGGED);
    let level = if failure_warns(log.recovering, failure_logged, &effect) {
        log::Level::Warn
    } else {
        log::Level::Debug
    };
    let effect = if extended {
        dev.flags.insert(DeviceFlags::RESET_FAILURE_LOGGED);
        effect
    } else {
        ResetFailureEffect::Unstated
    };
    log::log!(
        target: LOG_TARGET,
        level,
        "{}",
        log.summary(pp, ResetFailed::new(reason, effect)),
    );
}

/// Answer a rejected `Reset`: on a plain device, with the device state at its defaults.
///
/// An extended device answers `D3DERR_INVALIDCALL` and changes nothing: it
/// keeps its state, bindings, scene and frame, and owes no `Reset`. On a
/// plain device D3D9 resets the device state before it checks what can make a
/// `Reset` fail, so a rejected one still releases every binding and returns
/// every state to its default, leaving what a successful `Reset` leaves: render
/// target 0 on the back buffer, the depth stencil on the implicit surface,
/// no recording and no open scene. The frame in flight is delivered first,
/// as a same-size `Reset` does, and [`fail_reset`] replaces it, so the
/// attachments it records follow the defaults as well; without that, a draw
/// after the rejection would be built for the back buffer while its pass
/// still carried the targets the application had bound.
fn reject_reset(dev: &mut DeviceInner) -> i32 {
    // An extended device answers a rejected `Reset` without touching its
    // state: it keeps its bindings, its scene and its frame, and needs no
    // further `Reset`.
    if dev.is_extended() {
        return D3DERR_INVALIDCALL;
    }
    if let Err(hr) = dev.flush_current_frame_blocking() {
        dev.flags.insert(DeviceFlags::NOT_RESET);
        return hr;
    }
    fail_reset(dev, D3DERR_INVALIDCALL)
}

/// End a failed `Reset` on a fresh frame, answering `hr`.
///
/// The frame in flight was already delivered by the caller, unless the
/// encoder had already failed, which this then leaves latched. Past the point
/// where `Reset` destroyed the implicit back buffer or depth texture, the
/// frame that delivery left behind and the saved render-target and depth
/// bindings still name the destroyed textures, so the frame is dropped
/// unsent rather than delivered and the defaults clear the saved bindings.
/// The fresh frame names whatever the failed recreate left, NULL where it
/// made nothing. The device reports `D3DERR_DEVICENOTRESET` until a `Reset`
/// succeeds.
fn fail_reset(dev: &mut DeviceInner, hr: i32) -> i32 {
    dev.flags.insert(DeviceFlags::NOT_RESET);
    dev.reseed_current_frame();
    // An extended device keeps its state here too; its targets return to the
    // implicit surfaces, which name whatever the failed recreate left.
    if dev.is_extended() {
        dev.rebind_after_extended_reset();
    } else {
        dev.reset_to_defaults();
    }
    hr
}

/// Resolve geometry after the window transition, undoing a rejected fullscreen exit.
///
/// Only windowed requests can still have zero dimensions here. Leaving
/// fullscreen must precede their client-area query, but the fullscreen
/// session stays owned until that query has produced a usable extent.
fn resolve_reset_window_mode(
    dev: &mut DeviceInner,
    target: usize,
    pp: &mut mtld3d_types::D3DPRESENT_PARAMETERS,
) -> bool {
    if pp.windowed != 0
        && (pp.back_buffer_width == 0 || pp.back_buffer_height == 0)
        && let Some(saved) = dev.fullscreen.as_ref()
    {
        let accepted = crate::fullscreen::try_leave(saved, dev.leave_kind(), || {
            crate::direct3d9::resolve_backbuffer_dims(target as u64, pp);
            pp.back_buffer_width != 0 && pp.back_buffer_height != 0
        });
        if accepted {
            dev.fullscreen = None;
        }
        return accepted;
    }
    apply_reset_window_mode(dev, pp);
    crate::direct3d9::resolve_backbuffer_dims(target as u64, pp);
    pp.back_buffer_width != 0 && pp.back_buffer_height != 0
}

/// Carry out the windowed/fullscreen transition a `Reset` asks for.
///
/// All four combinations are legal and a game picks freely between them:
/// entering fullscreen, leaving it, staying fullscreen (possibly on a new
/// device window), and staying windowed. A refused display mode uses the
/// existing monitor-covering fallback.
fn apply_reset_window_mode(dev: &mut DeviceInner, pp: &mtld3d_types::D3DPRESENT_PARAMETERS) {
    if pp.windowed != 0 {
        dev.leave_fullscreen(&mtld3d_core::fullscreen_log::LeaveReason::WindowedReset);
        return;
    }
    let target = if pp.device_window == 0 {
        dev.window()
    } else {
        pp.device_window
    } as *mut c_void;
    let mode = crate::direct3d9::fullscreen_mode_request(pp);
    if dev.fullscreen_window() == Some(target) {
        dev.update_fullscreen(mode);
        return;
    }
    // A Reset may retarget the device at another window. The session is
    // handed over rather than given back: the window left behind keeps the
    // borderless style and the monitor rect it was put in, since the app that
    // retargeted its device did not ask for a screen to be uncovered behind
    // it, and the new window is taken over exactly as a fullscreen create
    // takes one over, so the eventual leave gives back the window the device
    // presents into.
    dev.enter_fullscreen(target, mode);
}

/// Move the presentation surface and the cursor subclass to a new device window.
///
/// A `Reset` may retarget the device at another window, and D3D9 answers it by
/// re-specifying the swap chain on that window. Here the swap chain is the
/// `CAMetalLayer` attached to the window `CreateDevice` was given, plus the
/// subclass the cursor latches and the auto-resize ride. The old metal view
/// and its attachment record are retired first, so the record the new attach
/// registers is the only one this device owns.
///
/// An attach that fails leaves the device with no presentation surface, the
/// same answer `CreateDevice` gives that failure: the device keeps working and
/// `Present` becomes a no-op.
fn retarget_device_window(
    dev: &mut DeviceInner,
    pp: &mtld3d_types::D3DPRESENT_PARAMETERS,
    hwnd: usize,
) -> Result<(), i32> {
    // Ops already queued name the layer that is about to go, and the encoder
    // waits for GPU idle, so nothing in flight references the view once the
    // detach releases it. No texture is destroyed here, so the encoder is
    // handed no retired handles.
    dev.flush_current_frame_blocking()?;
    dev.encoder_reset(&[])?;
    if !dev.view_handle.is_null() {
        let mut detach = mtld3d_shared::DetachMetalLayerParams {
            view_handle: dev.view_handle,
        };
        unix_call(&mut detach);
    }
    let config = Arc::clone(&dev.config);
    let layer_params = crate::direct3d9::attach_metal_layer(
        hwnd as u64,
        dev.device_handle,
        pp,
        &config,
        dev.cursor.sinks(),
    );
    if layer_params.view_handle.is_null() {
        warn!(
            target: LOG_TARGET,
            "Reset: no Metal layer on the new device window 0x{hwnd:x}; the device presents              nowhere until a later Reset attaches one",
        );
    }
    dev.view_handle = layer_params.view_handle;
    dev.layer_handle = layer_params.layer_handle;
    // The attach above latched this Reset's pacing on the new layer, so that
    // is the value a later Reset on this window compares against.
    dev.layer_pacing = crate::direct3d9::attached_pacing(&layer_params);
    let (cursor_scale, _origin) =
        crate::direct3d9::resolve_cursor_scale(layer_params.backing_scale, config.cursor_scale);
    let dev_ptr = std::ptr::from_mut::<DeviceInner>(dev);
    dev.cursor
        .retarget(hwnd as *mut c_void, layer_params.view_handle, dev_ptr);
    dev.cursor.follow_scale(cursor_scale);
    info!(
        target: LOG_TARGET,
        "Reset: device window retargeted to 0x{hwnd:x} (view {:#x}, layer {:#x})",
        layer_params.view_handle.raw(),
        layer_params.layer_handle.raw(),
    );
    Ok(())
}

/// Steps 1-6 of the Reset protocol.
///
/// Drain in-flight ops, destroy old backbuffer + depth/stencil, adopt
/// new dimensions, push the new pixel size to `CAMetalLayer`, and
/// recreate both textures. Returns the D3D9 HRESULT to bubble back to
/// the caller on failure; `Ok(())` on success. State defaults + reseed +
/// display-sync handling (steps 7-9) remains in the parent because the final
/// pacing step depends on whether this Reset attached a fresh layer.
fn reset_recreate_resources(
    dev: &mut DeviceInner,
    pp: &mtld3d_types::D3DPRESENT_PARAMETERS,
    held: HeldImplicitSurfaces,
) -> Result<(), i32> {
    // 1. Drain any ops the API thread queued onto current_frame after the
    //    last Present — same pattern as device_release. The encoder's
    //    Reset handler waits for GPU idle, so by the time it returns,
    //    no in-flight command buffer references the old backbuffer or
    //    depth/stencil textures we're about to destroy.
    dev.flush_current_frame_blocking()?;
    // 2. Destroy the old backbuffer + depth/stencil. Bulk thunk so the
    //    two handles cross the PE/Unix boundary in one call. The encoder is
    //    handed the same list: these five leave without passing through the
    //    retention queue, which is where every other texture's handle-keyed
    //    records are pruned. A held implicit surface of an extended device
    //    takes its textures over instead, and they stay.
    dev.detach_held_implicit_surfaces(held);
    let old_handles: [u64; 5] = [
        dev.backbuffer_handle.raw(),
        dev.backbuffer_srgb_handle.raw(),
        dev.backbuffer_msaa_handle.raw(),
        dev.backbuffer_msaa_srgb_handle.raw(),
        dev.depth_stencil_handle.raw(),
    ];
    let live: Vec<u64> = old_handles.iter().copied().filter(|&h| h != 0).collect();
    dev.encoder_reset(&live)?;
    if !live.is_empty() {
        let mut destroy = mtld3d_shared::DestroyResourcesBulkParams {
            kind: mtld3d_shared::mtl::DestroyKind::Texture,
            pad0: 0,
            handles_ptr: live.as_ptr() as u64,
            count: u32::try_from(live.len()).expect("at most 5 handles"),
            pad1: 0,
        };
        unix_call(&mut destroy);
    }
    // 3. Adopt the new dimensions. Done before CreateBackbuffer so the
    //    new textures are sized correctly and downstream readers
    //    (viewport defaults, GetBackBuffer, fresh_frame) all see the
    //    new dims for the post-Reset frame.
    dev.set_backbuffer_dims(pp.back_buffer_width, pp.back_buffer_height);

    // 4. Recreate the backbuffer at the new dims, in render space — this is
    //    the texture rasterization writes into. The drawable is not touched:
    //    it is the layer's own surface, sized by Core Animation from the
    //    view, and present resolves whatever difference remains.
    let mut bb_params = mtld3d_shared::CreateBackbufferParams {
        device_handle: dev.device_handle,
        record_handle: dev.record_handle,
        width: dev.render_scale.dimension(pp.back_buffer_width),
        height: dev.render_scale.dimension(pp.back_buffer_height),
        sample_count: u32::from(dev.backbuffer_sample_count),
        pad0: 0,
        texture_handle: MetalHandle::NULL,
        srgb_texture_handle: MetalHandle::NULL,
        msaa_texture_handle: MetalHandle::NULL,
        msaa_srgb_texture_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut bb_params);
    if status != 0 || bb_params.texture_handle.is_null() {
        error!(
            target: LOG_TARGET,
            "Reset: CreateBackbuffer failed (0x{status:08X}) for {}x{} (render {}x{}) samples={} \
             fmt={}; device unusable",
            pp.back_buffer_width,
            pp.back_buffer_height,
            bb_params.width,
            bb_params.height,
            bb_params.sample_count,
            pp.back_buffer_format,
        );
        dev.set_backbuffer_handle(MetalHandle::NULL, MetalHandle::NULL);
        dev.set_backbuffer_msaa_handle(MetalHandle::NULL, MetalHandle::NULL);
        dev.set_depth_stencil_handle(MetalHandle::NULL);
        dev.depth_stencil_format = 0;
        return Err(D3DERR_INVALIDCALL);
    }
    dev.set_backbuffer_handle(bb_params.texture_handle, bb_params.srgb_texture_handle);
    dev.set_backbuffer_msaa_handle(
        bb_params.msaa_texture_handle,
        bb_params.msaa_srgb_texture_handle,
    );

    // 5. Recreate depth/stencil if the device had one. Format is taken
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
            // No implicit depth surface exists now, and the format says so,
            // as `reconcile_implicit_depth` leaves it on the same failure.
            dev.set_depth_stencil_handle(MetalHandle::NULL);
            dev.depth_stencil_format = 0;
            return Err(D3DERR_INVALIDCALL);
        };
        // Render space, matching the colour attachment exactly — Metal
        // rejects a pass whose attachments disagree on size.
        let mut ds_params = CreateDepthTextureParams {
            device_handle: dev.device_handle,
            width: bb_params.width,
            height: bb_params.height,
            pixel_format,
            sample_count: u32::from(dev.backbuffer_sample_count),
            texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut ds_params);
        if status != 0 || ds_params.texture_handle.is_null() {
            error!(target: LOG_TARGET, "Reset: CreateDepthTexture failed (0x{status:08X}) — device unusable");
            dev.set_depth_stencil_handle(MetalHandle::NULL);
            dev.depth_stencil_format = 0;
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
    dev.flush_current_frame_blocking()?;
    // The old surface leaves without passing through the retention queue, so
    // the encoder is told about it here for the same reason the resizing path
    // tells it: nothing else prunes the records keyed on its handle.
    let old_depth = dev.depth_stencil_handle.raw();
    let retired: Vec<u64> = if had_depth {
        vec![old_depth]
    } else {
        Vec::new()
    };
    dev.encoder_reset(&retired)?;
    if had_depth {
        let handles = [old_depth];
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
        // Render space, matching the backbuffer texture rather than the
        // logical dims the device reports.
        let mut ds_params = CreateDepthTextureParams {
            device_handle: dev.device_handle,
            width: dev.render_scale.dimension(dev.backbuffer_width),
            height: dev.render_scale.dimension(dev.backbuffer_height),
            pixel_format,
            sample_count: u32::from(dev.backbuffer_sample_count),
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
    src_rect: *const c_void,
    dst_rect: *const c_void,
    dst_window_override: *mut c_void,
    dirty_region: *const c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    present_impl(
        this,
        src_rect,
        dst_rect,
        dst_window_override,
        dirty_region,
        0,
    )
}

/// The body `Present` and `PresentEx` share; the caller holds the API lock.
///
/// Every argument but `this` is accepted and ignored, each warned once.
fn present_impl(
    this: *mut c_void,
    src_rect: *const c_void,
    dst_rect: *const c_void,
    dst_window_override: *mut c_void,
    dirty_region: *const c_void,
    flags: u32,
) -> i32 {
    warn_ignored_present_arguments(
        src_rect,
        dst_rect,
        !dst_window_override.is_null(),
        dirty_region,
    );
    warn_ignored_present_flags(flags);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if dev.needs_reset() {
        return mtld3d_types::D3DERR_DEVICENOTRESET;
    }

    mtld3d_shared::crumb!("d3d9:present");
    dev.present()
}

/// Warn once for each `D3DPRESENT_*` flag bit a present carries, none of which is honoured.
///
/// The frame presents as it would without the flag: `DONOTWAIT` still waits,
/// `FORCEIMMEDIATE` still follows the presentation interval, and the overlay
/// bits have no overlay to act on. Shared by `PresentEx` and
/// `IDirect3DSwapChain9::Present`.
pub fn warn_ignored_present_flags(flags: u32) {
    for bit in mtld3d_core::extended::present_flag_bits(flags) {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: u64::from(bit),
            "Present: {} ({bit:#x}) is not honoured; the frame presents as it would without it",
            mtld3d_core::extended::present_flag_name(bit)
        );
    }
}

/// Warn once for each optional `Present` argument, none of which is honoured.
///
/// Presentation always shows the whole back buffer across the device window:
/// a source or destination rect, a destination window override and a dirty
/// region are accepted and ignored. Shared by `IDirect3DDevice9::Present` and
/// `IDirect3DSwapChain9::Present`.
#[inline]
pub fn warn_ignored_present_arguments(
    source_rect: *const c_void,
    dest_rect: *const c_void,
    window_override: bool,
    dirty_region: *const c_void,
) {
    if source_rect.is_null() && dest_rect.is_null() && !window_override && dirty_region.is_null() {
        return;
    }
    warn_present_arguments(source_rect, dest_rect, window_override, dirty_region);
}

/// The warnings themselves, out of line: nearly every `Present` passes no optional argument.
#[cold]
#[inline(never)]
fn warn_present_arguments(
    source_rect: *const c_void,
    dest_rect: *const c_void,
    window_override: bool,
    dirty_region: *const c_void,
) {
    if !source_rect.is_null() {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "Present: a source rect is ignored; the whole back buffer is presented");
    }
    if !dest_rect.is_null() {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "Present: a destination rect is ignored; the frame fills the device window");
    }
    if window_override {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "Present: a destination window override is ignored; the frame goes to the device \
             window");
    }
    if !dirty_region.is_null() {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "Present: a dirty region is ignored; the whole back buffer is presented");
    }
}

extern "system" fn device_get_back_buffer(
    this: *mut c_void,
    swap_chain: u32,
    back_buffer: u32,
    _type_: u32,
    surface: *mut *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetRasterStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_dialog_box_mode(this: *mut c_void, _enable: i32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetDialogBoxMode → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_gamma_ramp(
    this: *mut c_void,
    swap_chain: u32,
    flags: u32,
    ramp: *const c_void,
) {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // The call returns nothing, so every rejection below is a warn and a
    // no-op: there is no channel to answer through.
    if ramp.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "SetGammaRamp: null ramp → ignored"
        );
        return;
    }
    if swap_chain != 0 {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "SetGammaRamp: only the implicit swap chain carries a ramp; \
             swap chain {swap_chain} is ignored"
        );
        return;
    }
    if flags & mtld3d_types::D3DSGR_CALIBRATE != 0 {
        // The cap that promises calibration, D3DCAPS2_CANCALIBRATEGAMMA, is
        // not advertised, so applying the ramp as given is the conformant
        // answer to the flag rather than a shortcut.
        mtld3d_shared::log_once_info!(
            target: crate::LOG_TARGET,
            "SetGammaRamp: D3DSGR_CALIBRATE without CANCALIBRATEGAMMA → the ramp is \
             applied uncalibrated"
        );
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return;
    };
    // SAFETY: per the ABI `ramp` points at one readable `D3DGAMMARAMP` for
    // the call; it is non-null per the branch above and read, never retained.
    let ramp = unsafe { &*ramp.cast::<D3DGAMMARAMP>() };
    if !mtld3d_core::gamma::is_usable(ramp) {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "SetGammaRamp: no channel of the ramp rises end to end → ignored, the \
             display keeps the ramp it has"
        );
        return;
    }
    let dev = obj.inner();
    dev.set_gamma_ramp(ramp);
}

extern "system" fn device_get_gamma_ramp(this: *mut c_void, swap_chain: u32, ramp: *mut c_void) {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if ramp.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "GetGammaRamp: null ramp → ignored"
        );
        return;
    }
    if swap_chain != 0 {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "GetGammaRamp: only the implicit swap chain carries a ramp; \
             swap chain {swap_chain} answers nothing"
        );
        return;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return;
    };
    // SAFETY: per the ABI `ramp` points at one writable `D3DGAMMARAMP` for
    // the call; it is non-null per the branch above.
    unsafe {
        core::ptr::write(ramp.cast::<D3DGAMMARAMP>(), obj.inner().gamma_ramp());
    }
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    create_texture_path(&TextureCreateArgs {
        this,
        width,
        height,
        levels,
        usage,
        format,
        pool,
        texture,
        shared_handle,
        offscreen_plain: false,
    })
}

/// Vtable-shaped args bundle for [`create_texture_path`].
#[derive(Clone, Copy)]
struct TextureCreateArgs {
    this: *mut c_void,
    width: u32,
    height: u32,
    levels: u32,
    usage: u32,
    format: u32,
    pool: u32,
    texture: *mut *mut c_void,
    shared_handle: *mut c_void,
    /// The texture backs a `CreateOffscreenPlainSurface` surface.
    ///
    /// Carried in by the caller rather than stamped on the finished texture:
    /// the flag gates the staging-drop loop that runs inside the texture
    /// constructor, and a plain keeps its staging because the game may lock it.
    offscreen_plain: bool,
}

/// Resolve a create's `Levels` argument against the chain its extent allows.
///
/// Shared by the colour, depth, cube and volume create paths, which all take
/// `0` as a request for the full chain and any other value as the level count.
/// A count above the natural chain is capped at it rather than refused: the
/// levels past the chain would each repeat the last one, and D3D9 hands the
/// caller the chain its dimensions do have. Passing such a count through is
/// not an option either, because Metal's descriptor validation refuses a
/// `mipmapLevelCount` above `floor(log2(max_dim)) + 1` with an abort rather
/// than a returned error. The clamp is warned once per distinct
/// `(requested, natural)` pair, naming the entry point the first request came
/// through.
fn resolve_create_levels(entry_point: &str, requested: u32, natural: u32) -> u32 {
    if requested > natural {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(requested) << 32) | u64::from(natural),
            "{entry_point} levels={requested} is past the {natural}-level chain the extent allows; \
             clamped to {natural}"
        );
    }
    resolve_mip_levels(requested, natural)
}

/// Refuse a create whose system-memory copy the process cannot allocate.
///
/// The staging a resource gets at creation is the one allocation that may
/// fail; D3D9 answers such a create `E_OUTOFMEMORY` and hands back no object.
/// What was allocated before the failure has already been dropped by the
/// caller.
fn refuse_unallocatable_staging(entry_point: &str, out: *mut *mut c_void) -> i32 {
    warn!(target: LOG_TARGET,
        "reject {entry_point}: the resource's system-memory copy cannot be allocated → \
         E_OUTOFMEMORY");
    null_out(out);
    mtld3d_types::E_OUTOFMEMORY
}

/// The body of `CreateTexture`, plus the intent the vtable signature cannot carry.
///
/// `CreateOffscreenPlainSurface` backs a `D3DPOOL_DEFAULT` plain with a
/// texture created through here, so both entry points share one create path.
fn create_texture_path(info: &TextureCreateArgs) -> i32 {
    let TextureCreateArgs {
        this,
        width,
        height,
        levels,
        usage,
        format,
        pool,
        texture,
        shared_handle,
        offscreen_plain,
    } = *info;
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
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as user memory or a shared resource.
    let kind = if offscreen_plain {
        mtld3d_core::extended::CreateKind::OffscreenPlain
    } else {
        mtld3d_core::extended::CreateKind::Texture { levels }
    };
    let user_memory = match shared_handle_gate(this, &kind, pool, shared_handle, "CreateTexture") {
        Ok(memory) => memory,
        Err(hr) => {
            null_out(texture);
            return hr;
        }
    };
    if managed_pool_refused(this, &kind, pool, "CreateTexture") {
        null_out(texture);
        return D3DERR_INVALIDCALL;
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

    // Past the extent the device reports, a texture Metal backs is refused:
    // Metal itself aborts the process on a descriptor past its own limit. A
    // system-memory or scratch texture has no Metal texture until it is bound
    // for sampling, and creates at any extent.
    if !mtld3d_core::pool::is_cpu_only(pool) && !caps::texture_extent_fits(width, height) {
        let entry_point = if offscreen_plain {
            "CreateOffscreenPlainSurface"
        } else {
            "CreateTexture"
        };
        mtld3d_shared::log_once_warn_by!(target: LOG_TARGET, key: u64::from(offscreen_plain),
            "reject {entry_point}({width}x{height}, pool={pool}) → NOTAVAILABLE (past the {} \
             texel extent the device reports)", caps::MAX_TEXTURE_DIM);
        null_out(texture);
        return D3DERR_NOTAVAILABLE;
    }

    // Offscreen plain surfaces must be lockable. A GPU-only depth texture
    // cannot provide the staging that this internal surface path requires.
    if is_depth_fmt && offscreen_plain {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject CreateOffscreenPlainSurface depth format → INVALIDCALL (no CPU staging)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }

    // Depth-format CreateTexture is the D3D9 sampleable-shadow-map idiom:
    // game asks for a texture in a depth format with D3DUSAGE_DEPTHSTENCIL,
    // binds its surface as the depth target during the shadow pass, and
    // samples it as a regular texture during the lit pass. Mapping table
    // `map_d3d_format` is color-only — depth formats route through
    // `map_d3d_depth_format` and a create path of their own: no CPU staging,
    // and a mip chain, when one is asked for, that only the GPU can fill.
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

    // Packed 4:2:2 YUV has no sampling decode, so the device answers no to the
    // texture query and only the runtime's CPU-only SCRATCH form creates, the
    // rule cube and volume textures follow too. The DEFAULT-pool offscreen
    // plain `StretchRect` decodes keeps the texture behind it.
    if !offscreen_plain
        && mtld3d_core::stretch_rect::is_packed_yuv(format)
        && pool != D3DPOOL_SCRATCH
    {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(pool) << 32) | u64::from(format),
            "reject CreateTexture(format={}, pool={pool}) → INVALIDCALL (a packed YUV texture is \
             a D3DPOOL_SCRATCH resource only)",
            mtld3d_core::format::format_name(format)
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    let expand_packed16 = obj.inner().config().expand_packed16;
    let Some((fmt, planar_layout)) =
        resolve_texture_storage(format, (width, height), offscreen_plain, expand_packed16)
    else {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    if mtld3d_core::pool::usage_conflicts_with_pool(usage, pool) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(pool) << 32) | u64::from(format),
            "reject CreateTexture(format={format}, DYNAMIC, pool={pool}) → INVALIDCALL"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // A create's usage has to agree with the answer `CheckDeviceFormat` gives
    // for the same format, so a format that is not colour-renderable cannot
    // carry D3DUSAGE_RENDERTARGET; this rejects the caller that skipped the
    // probe. Two groups fail it: the sampling-only formats, block-compressed
    // ones included, which no D3D9 device rendered into, and the packed 16-bit
    // members on a device that expansion-backs them, where rendering into the
    // BGRA8 backing would break Lock/readback fidelity. Neither can be passed
    // through, because Metal's render-pipeline validation refuses a
    // non-renderable colour attachment with an abort rather than a returned
    // error: a lenient create costs the process at the first draw into the
    // texture instead of costing the application one handled failure here.
    if usage_rt && !crate::direct3d9::is_render_target_format_on_device(format, expand_packed16) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(format),
            "reject CreateTexture(format={format}, RENDERTARGET) → INVALIDCALL (format is not colour-renderable on this device)"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
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
    // to downsample. V16U16 retains public usage but uses one level without
    // the internal generation flag, matching its NOAUTOGEN query answer.
    let autogen_requested = usage & D3DUSAGE_AUTOGENMIPMAP != 0;
    let noautogen = autogen_requested && mtld3d_core::format::uses_noautogen_fallback(format);
    let autogen_mipmap = autogen_requested && !noautogen;
    let autogen_full_chain = autogen_mipmap && !fmt.is_compressed();
    let natural_levels = compute_mip_count(width, height);
    let actual_levels = if noautogen {
        1
    } else if autogen_full_chain {
        natural_levels
    } else {
        resolve_create_levels("CreateTexture", levels, natural_levels)
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
    // so the upload operation can hand the encoder thread a refcount bump
    // (no memcpy) at `UnlockRect` time. Contents are uninitialized —
    // the blit upload only copies the dirty sub-rect the game writes,
    // and a draw that references a never-Locked MTLTexture samples
    // Metal-zeroed texture memory (independent of the staging PageBox).
    // Page alignment satisfies `newBufferWithBytesNoCopy:`'s contract
    // on non-UMA Macs (Intel/AMD). Each level pops a parked box of its
    // padded size from the page-box pool before it falls back to the
    // allocator; the device counts both outcomes once per create.
    let mut staging: Vec<PageBox> = Vec::with_capacity(actual_levels as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);

    // The staging is the one allocation of a create that may fail: a level the
    // process has no room for refuses the create with `E_OUTOFMEMORY`.
    let mut out_of_memory = false;
    let (pool_hits, pool_misses) = {
        let mut staging_take = take_staging();
        if let Some(layout) = &planar_layout {
            // One level whose allocation is the planar layout's, not pitch times
            // height: the chroma planes follow the luma rows in the same box, and
            // the lock pitch strides all of them. The per-level arrays keep the
            // logical extent, which is what `GetDesc` and rect validation read.
            match staging_take.try_take(layout.total_bytes()) {
                Some(page) => staging.push(page),
                None => out_of_memory = true,
            }
            mip_widths.push(width);
            mip_heights.push(height);
            mip_bytes_per_row.push(layout.pitch());
            // Every offset a planar lock, upload or decode forms is derived from
            // the layout, so the allocation has to be exactly the layout's size.
            debug_assert!(out_of_memory || staging[0].logical_len() == layout.total_bytes());
            debug_assert_eq!(actual_levels, 1);
        } else {
            for level in 0..actual_levels {
                let (mw, mh, size, bpr) = compute_mip_size(width, height, level, &fmt);
                let Some(page) = staging_take.try_take(size as usize) else {
                    out_of_memory = true;
                    break;
                };
                staging.push(page);
                mip_widths.push(mw);
                mip_heights.push(mh);
                mip_bytes_per_row.push(bpr);
            }
        }
        staging_take.finish()
    };
    if out_of_memory {
        return refuse_unallocatable_staging("CreateTexture", texture);
    }
    // An extended device's single-level system-memory texture starts with the
    // application's pixels, packed row after row, copied into the level once.
    if let Some(memory) = user_memory {
        let rows = mtld3d_core::extended::PackedRows::of_level(
            width,
            height,
            fmt.bytes_per_pixel(),
            (fmt.block_width(), fmt.block_height(), fmt.block_bytes()),
        );
        let pitch = usize::try_from(mip_bytes_per_row[0]).expect("a row pitch fits usize");
        copy_user_memory(memory, &rows, staging[0].as_mut_slice(), pitch);
    }

    obj.inner()
        .perf_mut()
        .add_texture_pool_outcomes(pool_hits, pool_misses);
    let mut flags = TextureFlags::empty();
    flags.set(TextureFlags::AUTOGEN_MIPMAP, autogen_mipmap);
    flags.set(TextureFlags::OFFSCREEN_PLAIN, offscreen_plain);
    // A render target the game created at the reported back-buffer size is its
    // main view and shares the back buffer's scale. A texture it uploads pixels
    // into never does: its staging layout is keyed to the reported size.
    let render_scale =
        obj.inner()
            .scale_for_created_target(width, height, usage & D3DUSAGE_RENDERTARGET != 0);
    let tex = Direct3DTexture9::new(TextureCreateInfo {
        texture_id: TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        render_scale,
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

    push_texture_warmups(obj.inner(), tex.inner());
    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live texture at refcount 1.
    unsafe { crate::com_ref::com_register_child(tex_ptr) };
    // Target textures (render target / depth-stencil usage) are rare and
    // long-lived, and which of them a game keeps, releases, or re-creates
    // decides how cross-pass data flows; log their lifecycle unconditionally.
    if usage & (D3DUSAGE_RENDERTARGET | D3DUSAGE_DEPTHSTENCIL) != 0 {
        // SAFETY: `tex_ptr` is the freshly created, live texture from above.
        let id = unsafe { (*tex_ptr).texture_id() };
        debug!(
            target: LOG_TARGET,
            "target texture created: {id:?} fmt={format:#x} {width}x{height} usage={usage:#x} \
             ptr={tex_ptr:p}"
        );
    }
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    0 // S_OK
}

/// Storage a 2D colour create uses: the Metal mapping, plus the planar layout if it has one.
///
/// Every format but the planar YUV pair answers from the device-aware format
/// lookup. `YV12` and `NV12` are absent from that lookup on purpose, because a
/// planar level is taller than its logical height and no generic consumer of a
/// mapping sizes for that. They are accepted here for the one create that
/// sizes the level from the planar layout, the default-pool offscreen plain,
/// and only at an extent the layout defines. `None` has been warned about and
/// is the caller's `D3DERR_INVALIDCALL`.
fn resolve_texture_storage(
    format: u32,
    (width, height): (u32, u32),
    offscreen_plain: bool,
    expand_packed16: bool,
) -> Option<(
    FormatMapping,
    Option<mtld3d_core::planar_yuv::PlanarYuvLayout>,
)> {
    if !mtld3d_core::stretch_rect::is_planar_yuv(format) {
        let mapping = crate::direct3d9::map_for_device(format, expand_packed16);
        if mapping.is_none() {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "reject CreateTexture(format={format}) → INVALIDCALL (no format mapping)");
        }
        return mapping.map(|fmt| (fmt, None));
    }
    if !offscreen_plain {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateTexture(format={}) → INVALIDCALL (planar YUV exists only as a \
             D3DPOOL_DEFAULT offscreen plain surface)",
            mtld3d_core::format::format_name(format));
        return None;
    }
    let Some(layout) = mtld3d_core::planar_yuv::planar_yuv_layout(format, width, height) else {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(format),
            "reject CreateOffscreenPlainSurface({width}x{height}, format={}) → INVALIDCALL (a \
             YV12 surface needs an even height, and the luma and chroma rows together have to \
             fit the {} texture limit)",
            mtld3d_core::format::format_name(format),
            caps::MAX_TEXTURE_DIM
        );
        return None;
    };
    Some((
        mtld3d_core::format::planar_yuv_storage_mapping(),
        Some(layout),
    ))
}

/// Queue the eager `MTLTexture` creation before any draw can bind it.
///
/// Runs for a freshly constructed texture and when a system-memory texture is
/// promoted at its first sampling bind. Uploads create staging wrappers on first use.
fn push_texture_warmups(dev: &mut DeviceInner, inner: &crate::texture::TextureInner) {
    if inner.is_cpu_only() {
        return;
    }
    let frame = &mut dev.current_frame;
    let recorder = frame
        .recorder
        .get_or_insert_with(mtld3d_core::encoder_packet::FrameRecorder::new);
    recorder.capture_texture_warmup(&mut frame.scratch, &inner.texture_info());
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

/// Create sampleable depth textures, optionally with packed dynamic staging.
///
/// DEPTHSTENCIL textures can attach through their level surfaces. Plain
/// textures are sampling/RESZ resources; their surface usage rejects attachment
/// binds. DYNAMIC admits only D16, D24X8 and D24S8 plain DEFAULT-pool resources
/// and retains packed CPU staging for each mip. Every other depth resource
/// stays GPU-only. Color render-target usage and CPU pools remain rejected.
///
/// Levels follow [`resolve_create_levels`], including full explicit chains.
/// AUTOGEN is the existing NOAUTOGEN one-level fallback, with no native mip
/// generation. Dynamic resources use logical dimensions, even at render.scale;
/// RESZ transfers convert the attachment's physical extent when necessary.
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

    // The lockable depth formats create as a depth-stencil texture, with no
    // `LockRect` of its levels, and not as a plain one: a plain depth texture
    // is a sampling resource the format queries would have to offer.
    if usage & D3DUSAGE_DEPTHSTENCIL == 0 && !is_depth_stencil_format(format) {
        mtld3d_shared::log_once_warn_by!(target: LOG_TARGET, key: u64::from(format),
            "reject CreateTexture plain depth format={format} → INVALIDCALL (unsupported format)");
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
    let dynamic = usage & D3DUSAGE_DYNAMIC != 0;
    let dynamic_supported = mtld3d_core::depth_texture::PackedDepth::from_d3d(format).is_some()
        && usage & D3DUSAGE_DEPTHSTENCIL == 0;
    let bad_usage_bits = usage & D3DUSAGE_RENDERTARGET
        | if dynamic && !dynamic_supported {
            D3DUSAGE_DYNAMIC
        } else {
            0
        };
    if bad_usage_bits != 0 {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(bad_usage_bits),
            "reject CreateTexture depth: incompatible usage bits {bad_usage_bits:#x} → INVALIDCALL"
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

    // A mip chain on a depth texture is an engine's depth pyramid: each level
    // is rendered into through `GetSurfaceLevel(n)` bound as the depth
    // attachment and sampled back at a coarser resolution.
    let actual_levels = if usage & D3DUSAGE_AUTOGENMIPMAP != 0 {
        if levels > 1 {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "reject CreateTexture depth AUTOGENMIPMAP levels={levels} → INVALIDCALL (expected 0 or 1)");
            null_out(texture);
            return D3DERR_INVALIDCALL;
        }
        // NOAUTOGEN is a successful single-level fallback. Leave the internal
        // AUTOGEN_MIPMAP flag unset: no path may generate depth mipmaps.
        1
    } else {
        resolve_create_levels(
            "CreateTexture depth",
            levels,
            compute_mip_count(width, height),
        )
    };
    let usage_flags = mtld3d_shared::mtl::TextureUsage::DEPTH_STENCIL
        | mtld3d_shared::mtl::TextureUsage::RENDER_TARGET;
    // The per-pixel size the mip chain is charged at against the
    // `GetAvailableTextureMem` budget, in the application's own currency: its
    // D3D9 format, not the wider Metal one a 24-bit depth format is promoted
    // to. Every format with a Metal depth mapping has an entry, so the
    // fallback is unreachable.
    let bytes_per_pixel =
        mtld3d_core::format::depth_format_bytes_per_pixel(format).unwrap_or_else(|| {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(format),
                "CreateTexture depth format={format} has a Metal mapping but no byte size; \
                 its levels are charged 0 bytes of texture memory"
            );
            0
        });

    trace!(
        target: LOG_TARGET,
        "CreateTexture depth {width}x{height} levels={actual_levels} format={format} → sampleable shadow map"
    );

    // Per-level dimensions so `GetLevelDesc` reports each mip, and a per-level
    // row pitch so `TextureInner::allocated_bytes` charges the chain on the
    // formula a colour chain is charged on: the host-visible stride of the
    // level's width, not a tight one. Dynamic depth also uses that pitch for
    // packed CPU staging. All three stay
    // in the dimensions the application asked for, which is what `GetLevelDesc`
    // answers with; a chain rasterized at `render.scale` is charged at the
    // extent its Metal levels hold instead, which `allocated_bytes` measures
    // from the scale the texture carries.
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);
    for level in 0..actual_levels {
        let (mw, mh, _, bpr) = linear_mip_size(width, height, level, bytes_per_pixel);
        mip_widths.push(mw);
        mip_heights.push(mh);
        mip_bytes_per_row.push(bpr);
    }

    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // Sized to the back buffer, this is the depth buffer for the main view and
    // has to keep matching the colour attachment it is bound with; any other
    // size is a shadow map with its own resolution.
    let render_scale = if dynamic {
        mtld3d_core::render_scale::RenderScale::IDENTITY
    } else {
        obj.inner().scale_for_created_target(width, height, true)
    };
    // The staging is the one allocation of a create that may fail.
    let staging = if dynamic {
        let staging: Option<Vec<PageBox>> = mip_bytes_per_row
            .iter()
            .zip(&mip_heights)
            .map(|(&pitch, &rows)| {
                PageBox::try_new_zeroed((pitch as usize).saturating_mul(rows as usize))
            })
            .collect();
        let Some(staging) = staging else {
            return refuse_unallocatable_staging("CreateTexture", texture);
        };
        staging
    } else {
        Vec::new()
    };
    let tex = Direct3DTexture9::new(TextureCreateInfo {
        texture_id: TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        render_scale,
        depth: 1,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format,
        flags: TextureFlags::DEPTH_FORMAT,
        swizzle: None,
        usage_flags,
        d3d_usage: usage,
        d3d_pool: pool,
        bytes_per_pixel,
        block_w: 1,
        block_h: 1,
        block_bytes: bytes_per_pixel,
        staging,
        mip_widths,
        mip_heights,
        mip_bytes_per_row,
    });

    // Queue the eager `MTLTexture` create (sampleable shadow map path).
    let info = tex.inner().texture_info();
    obj.inner().push_texture_warmup(&info);

    let tex_ptr = Box::into_raw(Box::new(tex));
    // SAFETY: `tex_ptr` is a freshly created, live texture at refcount 1.
    unsafe { crate::com_ref::com_register_child(tex_ptr) };
    // Mirror of the colour path's "target texture created" line; the depth
    // path returns before that one runs.
    {
        // SAFETY: `tex_ptr` is the freshly created, live texture from above.
        let id = unsafe { (*tex_ptr).texture_id() };
        debug!(
            target: LOG_TARGET,
            "target texture created: {id:?} fmt={format:#x} {width}x{height} usage={usage:#x} \
             ptr={tex_ptr:p}"
        );
    }
    // SAFETY: vtable out-param; `texture` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(texture, tex_ptr.cast::<c_void>()) };
    D3D_OK
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    trace!(
        target: LOG_TARGET,
        "IDirect3DDevice9::CreateVolumeTexture({width}x{height}x{depth}, fmt={format})"
    );
    if width == 0 || height == 0 || depth == 0 || texture.is_null() {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::VolumeTexture,
        pool,
        shared_handle,
        "CreateVolumeTexture",
    ) {
        null_out(texture);
        return hr;
    }
    if managed_pool_refused(
        this,
        &mtld3d_core::extended::CreateKind::VolumeTexture,
        pool,
        "CreateVolumeTexture",
    ) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    let Some(fmt) = crate::direct3d9::map_for_device(format, obj.inner().config().expand_packed16)
    else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateVolumeTexture(format={format}) → INVALIDCALL (no format mapping)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    // DXT base extents must be block aligned in every pool. The GPU-backed
    // pools take what `is_volume_texture_format` advertises, DXT1 to DXT5
    // among the compressed formats; the other compressed and the packed-YUV
    // formats are creatable as SCRATCH volumes only.
    let block_w = fmt.block_width();
    let block_h = fmt.block_height();
    if is_dxt_format(format) && (!width.is_multiple_of(block_w) || !height.is_multiple_of(block_h))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if pool != D3DPOOL_SCRATCH && !mtld3d_core::format::is_volume_texture_format(format) {
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
    if mtld3d_core::pool::usage_conflicts_with_pool(usage, pool) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(pool) << 32) | u64::from(format),
            "reject CreateVolumeTexture(format={format}, DYNAMIC, pool={pool}) → INVALIDCALL"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // A volume more than one slice deep is a 3D Metal texture, held to
    // `MaxVolumeExtent` on every axis; Metal aborts the process past its limit.
    if !mtld3d_core::pool::is_cpu_only(pool) && !caps::volume_extent_fits(width, height, depth) {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject CreateVolumeTexture({width}x{height}x{depth}, pool={pool}) → NOTAVAILABLE \
             (past the extent the device reports)");
        null_out(texture);
        return D3DERR_NOTAVAILABLE;
    }
    // A per-level 3D mip chain. Each level's box is the block-aware 2D slice
    // size (`compute_mip_size`, correct for DXT/ATI as well as plain formats)
    // times the level's depth; `LockBox` hands the game a pointer into the
    // level's box with the matching block-aware pitches. `levels == 0` requests
    // the full chain. Sizing the levels correctly is what lets `LockBox(level)`
    // resolve a real box instead of returning NULL (a NULL box would fault a
    // LockBox on a mip sub-level).
    // Zero stays zero: it marks a compressed layout, and the upload of a level
    // that is one slice deep counts block rows only while the marker survives.
    let bpp = fmt.bytes_per_pixel();
    let actual_levels = resolve_create_levels(
        "CreateVolumeTexture",
        levels,
        compute_volume_mip_count(width, height, depth),
    );
    let mut staging: Vec<PageBox> = Vec::with_capacity(actual_levels as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);
    let (pool_hits, pool_misses) = {
        let mut staging_take = take_staging();
        for level in 0..actual_levels {
            let (mw, mh, slice_size, bpr) = compute_mip_size(width, height, level, &fmt);
            let md = (depth >> level).max(1);
            let box_bytes = (slice_size as usize).saturating_mul(md as usize);
            // The staging is the one allocation of a create that may fail.
            let Some(page) = staging_take.try_take(box_bytes) else {
                break;
            };
            staging.push(page);
            mip_widths.push(mw);
            mip_heights.push(mh);
            mip_bytes_per_row.push(bpr);
        }
        staging_take.finish()
    };
    if staging.len() != actual_levels as usize {
        return refuse_unallocatable_staging("CreateVolumeTexture", texture);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner()
        .perf_mut()
        .add_texture_pool_outcomes(pool_hits, pool_misses);
    let tex = crate::texture::Direct3DVolumeTexture9::new(TextureCreateInfo {
        texture_id: mtld3d_core::ids::TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width,
        height,
        // A volume texture is never a render target, so it never scales.
        render_scale: mtld3d_core::render_scale::RenderScale::IDENTITY,
        depth,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format: fmt.metal_pixel_format(),
        // Its sub-resources are `IDirect3DVolume9` levels, not surfaces, whatever
        // the depth; the container has to free the cached slots as the kind they
        // hold.
        flags: TextureFlags::VOLUME_TEXTURE,
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
    // CPU-side and the volume samples as cleared until upload lands. A
    // system-memory volume gets no Metal texture at all.
    if !mtld3d_core::pool::is_cpu_only(pool) {
        obj.inner().push_texture_warmup(&tex.inner().texture_info());
    }
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if edge_length == 0 || texture.is_null() {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::CubeTexture,
        pool,
        shared_handle,
        "CreateCubeTexture",
    ) {
        null_out(texture);
        return hr;
    }
    if managed_pool_refused(
        this,
        &mtld3d_core::extended::CreateKind::CubeTexture,
        pool,
        "CreateCubeTexture",
    ) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // D3DUSAGE_WRITEONLY is a vertex/index-buffer-only flag; on a cube texture it
    // is INVALIDCALL.
    if usage & D3DUSAGE_WRITEONLY != 0 {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    let mut usage_flags = mtld3d_shared::mtl::TextureUsage::empty();
    if usage & D3DUSAGE_RENDERTARGET != 0 {
        usage_flags.insert(mtld3d_shared::mtl::TextureUsage::RENDER_TARGET);
    }
    // Cube render targets must be DEFAULT-pool color resources. Depth cubes
    // remain unsupported, and DYNAMIC cannot combine with either target usage.
    let usage_rt = usage & D3DUSAGE_RENDERTARGET != 0;
    let usage_ds = usage & D3DUSAGE_DEPTHSTENCIL != 0;
    if usage_ds || (usage_rt && (pool != D3DPOOL_DEFAULT || usage & D3DUSAGE_DYNAMIC != 0)) {
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
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    let expand_packed16 = obj.inner().config().expand_packed16;
    let Some(fmt) = crate::direct3d9::map_for_device(format, expand_packed16) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateCubeTexture(format={format}) → INVALIDCALL (no format mapping)");
        null_out(texture);
        return D3DERR_INVALIDCALL;
    };
    if mtld3d_core::pool::usage_conflicts_with_pool(usage, pool) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: (u64::from(pool) << 32) | u64::from(format),
            "reject CreateCubeTexture(format={format}, DYNAMIC, pool={pool}) → INVALIDCALL"
        );
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // DXTn cube faces must be block-aligned; edge_length
    // is both width and height of a face.
    if is_dxt_format(format) && !edge_length.is_multiple_of(fmt.block_width()) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // ATI1 and packed YUV retain only their CPU SCRATCH resource form. Depth
    // formats remain unsupported, and render-target cubes require a Metal
    // renderable color format on THIS device (the packed 16-bit members drop
    // out where they are expansion-backed).
    if (matches!(format, D3DFMT_ATI1 | D3DFMT_YUY2 | D3DFMT_UYVY) && pool != D3DPOOL_SCRATCH)
        || is_depth_fmt
        || (usage_rt
            && !crate::direct3d9::is_render_target_format_on_device(format, expand_packed16))
    {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if usage & D3DUSAGE_AUTOGENMIPMAP != 0 && levels > 1 {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    if usage & D3DUSAGE_AUTOGENMIPMAP != 0 && !matches!(pool, D3DPOOL_DEFAULT | D3DPOOL_MANAGED) {
        null_out(texture);
        return D3DERR_INVALIDCALL;
    }
    // Same extent rule as `CreateTexture`: a face past the reported extent is
    // refused in the pools Metal backs, before six faces of staging are made.
    if !mtld3d_core::pool::is_cpu_only(pool) && !caps::texture_extent_fits(edge_length, edge_length)
    {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject CreateCubeTexture(edge={edge_length}, pool={pool}) → NOTAVAILABLE (past the \
             {} texel extent the device reports)", caps::MAX_TEXTURE_DIM);
        null_out(texture);
        return D3DERR_NOTAVAILABLE;
    }
    // Zero stays zero: it marks a compressed layout, and a face upload counts
    // block rows and block offsets only while the marker survives creation.
    let bpp = fmt.bytes_per_pixel();
    // `levels == 0` means the full chain. Staging is face-major so the cube
    // sidecar can address `face * levels + level` without another allocation.
    let autogen_requested = usage & D3DUSAGE_AUTOGENMIPMAP != 0;
    let noautogen = autogen_requested && mtld3d_core::format::uses_noautogen_fallback(format);
    let autogen_mipmap = autogen_requested && !noautogen;
    let autogen_full_chain = autogen_mipmap && !fmt.is_compressed();
    let natural_levels = compute_mip_count(edge_length, edge_length);
    let actual_levels = if noautogen {
        1
    } else if autogen_full_chain {
        natural_levels
    } else {
        resolve_create_levels("CreateCubeTexture", levels, natural_levels)
    };
    let mut staging: Vec<PageBox> =
        Vec::with_capacity(actual_levels as usize * CUBE_FACE_COUNT as usize);
    let mut mip_widths = Vec::with_capacity(actual_levels as usize);
    let mut mip_heights = Vec::with_capacity(actual_levels as usize);
    let mut mip_bytes_per_row = Vec::with_capacity(actual_levels as usize);
    for level in 0..actual_levels {
        let (mw, mh, _size, bpr) = compute_mip_size(edge_length, edge_length, level, &fmt);
        mip_widths.push(mw);
        mip_heights.push(mh);
        mip_bytes_per_row.push(bpr);
    }
    let (pool_hits, pool_misses) = {
        let mut staging_take = take_staging();
        // The staging is the one allocation of a create that may fail.
        'faces: for _face in 0..CUBE_FACE_COUNT {
            for level in 0..actual_levels {
                let (_, _, size, _) = compute_mip_size(edge_length, edge_length, level, &fmt);
                let Some(page) = staging_take.try_take(size as usize) else {
                    break 'faces;
                };
                staging.push(page);
            }
        }
        staging_take.finish()
    };
    if staging.len() != actual_levels as usize * CUBE_FACE_COUNT as usize {
        return refuse_unallocatable_staging("CreateCubeTexture", texture);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner()
        .perf_mut()
        .add_texture_pool_outcomes(pool_hits, pool_misses);
    // A cube face is never the main view's colour or depth target, so it keeps
    // whatever edge length the game asked for.
    let mut flags = TextureFlags::CUBE;
    flags.set(TextureFlags::AUTOGEN_MIPMAP, autogen_mipmap);
    let tex = crate::texture::Direct3DCubeTexture9::new(TextureCreateInfo {
        texture_id: mtld3d_core::ids::TextureId::new_unique(),
        device_handle: obj.inner().device_handle,
        device_inner: obj.inner as u64,
        width: edge_length,
        height: edge_length,
        render_scale: mtld3d_core::render_scale::RenderScale::IDENTITY,
        depth: 1,
        levels: actual_levels,
        d3d_format: format,
        metal_pixel_format: fmt.metal_pixel_format(),
        flags,
        swizzle: fmt.swizzle(),
        usage_flags,
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
    if !mtld3d_core::pool::is_cpu_only(pool) {
        obj.inner().push_texture_warmup(&tex.inner().texture_info());
    }
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if vb.is_null() || length == 0 {
        null_out(vb);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::VertexBuffer,
        pool,
        shared_handle,
        "CreateVertexBuffer",
    ) {
        null_out(vb);
        return hr;
    }
    if managed_pool_refused(
        this,
        &mtld3d_core::extended::CreateKind::VertexBuffer,
        pool,
        "CreateVertexBuffer",
    ) {
        null_out(vb);
        return D3DERR_INVALIDCALL;
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
    let Some(buffer) = Direct3DVertexBuffer9::new(&VertexBufferCreateInfo {
        device_inner: std::ptr::from_mut::<DeviceInner>(dev),
        length,
        usage,
        fvf,
        pool,
    }) else {
        return refuse_unallocatable_staging("CreateVertexBuffer", vb);
    };
    // Queue the eager `MTLBuffer` wrap so subsequent draw operations hit
    // the buffer cache instead of cache-missing inside
    // `ensure_vbib_mtl_buffer` on first bind.
    let inner = buffer.inner();
    dev.push_buffer_warmup(VbibWarmupEntry {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr() as u64,
        backing_len: inner.current_backing_len() as u64,
        backing_generation: inner.current_backing_generation(),
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if ib.is_null() || length == 0 {
        null_out(ib);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::IndexBuffer,
        pool,
        shared_handle,
        "CreateIndexBuffer",
    ) {
        null_out(ib);
        return hr;
    }
    if managed_pool_refused(
        this,
        &mtld3d_core::extended::CreateKind::IndexBuffer,
        pool,
        "CreateIndexBuffer",
    ) {
        null_out(ib);
        return D3DERR_INVALIDCALL;
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
    let Some(buffer) = Direct3DIndexBuffer9::new(&IndexBufferCreateInfo {
        device_inner: std::ptr::from_mut::<DeviceInner>(dev),
        length,
        usage,
        format,
        pool,
    }) else {
        return refuse_unallocatable_staging("CreateIndexBuffer", ib);
    };
    // Queue the eager `MTLBuffer` wrap; same drain semantics as VB.
    let inner = buffer.inner();
    dev.push_buffer_warmup(VbibWarmupEntry {
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr() as u64,
        backing_len: inner.current_backing_len() as u64,
        backing_generation: inner.current_backing_generation(),
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

/// Resolve a surface create's `(multi_sample, quality)` against the device.
///
/// The same predicate `CheckDeviceMultiSampleType` answers with, so a game
/// that asked first gets the same verdict at create time. A create says
/// `INVALIDCALL` for every rejection, malformed or merely unavailable: the
/// query is where D3D9 draws that distinction, and the create's contract is
/// the single "these parameters do not describe a surface I can make".
fn resolve_surface_multi_sample(
    multi_sample_type: u32,
    multi_sample_quality: u32,
    format: u32,
    site: &str,
) -> Result<SurfaceMultiSample, i32> {
    let Ok(sample_count) = mtld3d_core::multisample::resolve_sample_count(
        multi_sample_type,
        multi_sample_quality,
        format,
        crate::direct3d9::device_caps_flags(),
    ) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject {site}: multi_sample={multi_sample_type} quality={multi_sample_quality} on format {format} is not creatable on this device → INVALIDCALL"
        );
        return Err(D3DERR_INVALIDCALL);
    };
    Ok(SurfaceMultiSample {
        multi_sample_type,
        multi_sample_quality,
        sample_count: u8::try_from(sample_count).expect("sample count ≤ 16 fits u8"),
    })
}

/// The D3D9 description of one standalone colour surface to create.
///
/// `multi_sample` is the already-resolved `(type, quality)` pair: the sample
/// count plus the D3D9 type the surface reports from `GetDesc`. Above one
/// sample the create also produces the multisampled companion the passes
/// attach.
struct ColorTargetSpec {
    width: u32,
    height: u32,
    format: u32,
    usage: u32,
    multi_sample: SurfaceMultiSample,
}

/// Create a persistent render-target-capable color `MTLTexture` and wrap it as a surface.
///
/// The wrapper is a standalone `Direct3DSurface9`, mirroring the depth path.
/// `CreateRenderTarget` (usage = `D3DUSAGE_RENDERTARGET`) is its only caller
/// today; the usage-0 shape a `D3DPOOL_DEFAULT` offscreen plain would take is
/// answered too, because the two differ only in which format question they ask.
/// Returns the boxed wrapper pointer, or `None` (caller maps to `INVALIDCALL`)
/// for an unmappable or compressed color format, for a format whose usage the
/// device cannot honour, or if the Metal allocation fails.
fn create_color_target_surface(
    device_handle: MetalHandle<MTLDeviceKind>,
    device_inner: *mut DeviceInner,
    spec: &ColorTargetSpec,
    expand_packed16: bool,
) -> Option<*mut Direct3DSurface9> {
    let &ColorTargetSpec {
        width,
        height,
        format,
        usage,
        multi_sample,
    } = spec;
    let mapping = crate::direct3d9::map_for_device(format, expand_packed16)?;
    if mapping.is_compressed() {
        return None;
    }
    // A create's usage has to agree with the answer `CheckDeviceFormat` gives
    // for the same format, and the two usages this fn serves ask different
    // questions of it.
    //
    // With `D3DUSAGE_RENDERTARGET` the question is the device predicate every
    // other render-target create gate uses. Two groups fail it: the
    // sampling-only formats, block-compressed ones included, which no D3D9
    // device rendered into, and the packed 16-bit members on a device that
    // expansion-backs them, where rendering into the BGRA8 backing would break
    // Lock/readback fidelity. Neither can be passed through, because Metal
    // refuses a colour attachment it cannot render into with an abort rather
    // than a returned error, and this create is the last point that can answer
    // the application with a D3D9 error instead. Nothing between here and the
    // draw catches it: `A8` (Metal's `A8Unorm`, which the format tables list as
    // sampleable only) and `L8` both built a render-target surface fine before
    // this gate.
    //
    // Without the usage the surface is only locked, sampled and blitted, so the
    // narrower question stands: a format whose texels are widened on the way
    // into a BGRA8 backing would pair a narrower CPU staging with a 32-bit
    // texture through the upload and readback blits. That is the packed 16-bit
    // family on a device without the native formats, and R8G8B8 on every
    // device.
    if usage & D3DUSAGE_RENDERTARGET != 0 {
        if !crate::direct3d9::is_render_target_format_on_device(format, expand_packed16) {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(format),
                "reject CreateRenderTarget(format={format}) → INVALIDCALL (format is not colour-renderable on this device)"
            );
            return None;
        }
    } else if mtld3d_core::upload_pass::is_expanded_upload(format, mapping.metal_pixel_format()) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(format),
            "reject standalone surface(format={format}) → INVALIDCALL (a format widened on upload is sampling-only)"
        );
        return None;
    }
    // A render target created at the reported back-buffer size is the game's
    // main view and shares its scale; the surface still reports `width`/`height`
    // so `GetDesc` and every coordinate the game supplies stay logical.
    // `D3DUSAGE_RENDERTARGET` gates it: this fn also serves
    // `CreateOffscreenPlainSurface`, whose pixels the game reads and writes at
    // the size it asked for. A lockable render target is not an exception: it
    // is bound and drawn into like any other, and a depth-stencil of the same
    // size scales whatever it is paired with, so a target that kept its
    // reported extent would rasterize against an attachment three quarters its
    // size. Its CPU staging stays at the reported extent instead, and the two
    // resample past each other the way the lockable back buffer's do. The
    // scale is stored on the surface, so every consumer of it reads the answer
    // back rather than re-deriving the rule.
    // SAFETY: `device_inner` is the live owning device, non-null for every
    // caller of this fn (they hold it from the device thunk).
    let device = unsafe { &*device_inner };
    let scale = device.scale_for_created_target(width, height, usage & D3DUSAGE_RENDERTARGET != 0);
    let mut params = CreateColorTargetParams {
        device_handle,
        record_handle: device.record_handle(),
        width: scale.dimension(width),
        height: scale.dimension(height),
        pixel_format: mapping.metal_pixel_format(),
        sample_count: u32::from(multi_sample.sample_count),
        texture_handle: MetalHandle::NULL,
        srgb_texture_handle: MetalHandle::NULL,
        msaa_texture_handle: MetalHandle::NULL,
        msaa_srgb_texture_handle: MetalHandle::NULL,
    };
    if unix_call(&mut params) != 0 || params.texture_handle.is_null() {
        return None;
    }
    let surf = Direct3DSurface9::new_color_target(&ColorTargetCreateInfo {
        device_inner,
        metal_color_handle: params.texture_handle,
        metal_color_srgb_handle: params.srgb_texture_handle,
        metal_msaa_handle: params.msaa_texture_handle,
        metal_msaa_srgb_handle: params.msaa_srgb_texture_handle,
        width,
        height,
        format,
        usage,
        render_scale: scale,
        multi_sample,
    });
    // The surface owns DEFAULT-pool Metal textures no `TextureInner` covers,
    // so charge them here; `finalize_surface`'s colour retire arm refunds
    // them. Above one sample that is the single-sample texture plus the
    // multisampled companion beside it. The charge measures the memory, so it
    // goes in at the extent the textures were created at while `width`/
    // `height` stay the logical pair `GetDesc` reports.
    // SAFETY: `device_inner` is the live owning device, non-null for every
    // caller of this fn.
    unsafe { &*device_inner }.register_standalone_surface(
        width,
        height,
        format,
        u32::from(multi_sample.sample_count),
        mtld3d_core::format::StandaloneSurfaceKind::ColorTarget,
        scale,
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
    multi_sample_quality: u32,
    lockable: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::RenderTarget,
        D3DPOOL_DEFAULT,
        shared_handle,
        "CreateRenderTarget",
    ) {
        null_out(surface);
        return hr;
    }
    let multi_sample = match resolve_surface_multi_sample(
        multi_sample,
        multi_sample_quality,
        format,
        "CreateRenderTarget",
    ) {
        Ok(ms) => ms,
        Err(hr) => {
            null_out(surface);
            return hr;
        }
    };
    // A lockable multisampled render target has no meaning: D3D9 defines the
    // lock against a single-sample surface, and the samples are the point of
    // the multisampled one.
    if lockable != 0 && multi_sample.sample_count > 1 {
        warn!(
            target: LOG_TARGET,
            "reject CreateRenderTarget({width}x{height}, ms={}) → INVALIDCALL (a multisampled render target cannot be lockable)",
            multi_sample.multi_sample_type
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    // A `Lockable == TRUE` render target keeps its standalone renderable colour
    // texture (so `GetContainer`/`GetDesc`/`StretchRect` are unchanged) but also
    // gets a CPU staging buffer: `LockRect` maps it, `UnlockRect` uploads it to
    // the colour texture. It is sized at the same row pitch every host-visible
    // surface store uses, so `LockRect`, the GPU read-back and the DIB a
    // `GetDC` wraps around it all step by the same stride. A format with no CPU
    // byte size sizes it at zero and is rejected by the create just below (a
    // lockable render target is an uncompressed colour format), so every
    // surface that survives the create carries the staging it asked for.
    let staging_bytes = if lockable == 0 {
        0
    } else {
        let bpp = map_d3d_format(format).map_or(0, |m| m.bytes_per_pixel());
        (linear_row_pitch(width, bpp) as usize).saturating_mul(height as usize)
    };
    // Zero-initialise the staging (defence-in-depth): a `LockRect` before any
    // render, or any path that skips the read-back fill, reads defined bytes
    // rather than allocator garbage. It is the one allocation of the create that
    // may fail, and it is made before the colour texture so a refusal leaves
    // nothing to tear down.
    let staging = if staging_bytes == 0 {
        None
    } else {
        let Some(staging) = PageBox::try_new_zeroed(staging_bytes) else {
            return refuse_unallocatable_staging("CreateRenderTarget", surface);
        };
        Some(staging)
    };
    let Some(surf_ptr) = create_color_target_surface(
        obj.inner().device_handle,
        obj.inner_ptr(),
        &ColorTargetSpec {
            width,
            height,
            format,
            usage: D3DUSAGE_RENDERTARGET,
            multi_sample,
        },
        obj.inner().config().expand_packed16,
    ) else {
        warn!(
            target: LOG_TARGET,
            "reject CreateRenderTarget({width}x{height}, format={format}) → INVALIDCALL (no renderable Metal color mapping or allocation failed)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    };
    if let Some(staging) = staging {
        // SAFETY: `surf_ptr` is the freshly created, live standalone RT
        // surface (refcount 1); no other reference exists yet, so the
        // exclusive borrow to attach the staging is sound.
        unsafe { &mut *surf_ptr }.set_lockable_staging(staging);
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
    multi_sample_quality: u32,
    _discard: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as a shared resource, which it does not create.
    if let Err(hr) = shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::DepthStencil,
        D3DPOOL_DEFAULT,
        shared_handle,
        "CreateDepthStencilSurface",
    ) {
        null_out(surface);
        return hr;
    }
    if !is_depth_stencil_format(format) {
        warn!(
            target: LOG_TARGET,
            "reject CreateDepthStencilSurface({width}x{height}, format={format}) → INVALIDCALL (not a depth format)"
        );
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    let multi_sample = match resolve_surface_multi_sample(
        multi_sample,
        multi_sample_quality,
        format,
        "CreateDepthStencilSurface",
    ) {
        Ok(ms) => ms,
        Err(hr) => {
            null_out(surface);
            return hr;
        }
    };
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
    // A depth buffer created at the reported back-buffer size is the one paired
    // with the main view, so it shares the back buffer's scale and stays the
    // same size as the colour attachment it is bound with. The surface still
    // reports `width`/`height`. A differently-sized depth surface is a shadow
    // map or similar and keeps its own resolution.
    let scale = obj.inner().scale_for_created_target(width, height, true);
    let mut params = CreateDepthTextureParams {
        device_handle,
        width: scale.dimension(width),
        height: scale.dimension(height),
        pixel_format,
        sample_count: u32::from(multi_sample.sample_count),
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
        scale,
        multi_sample,
    );
    // Same accounting as a standalone colour target: a real Metal depth
    // texture with no `TextureInner` behind it, charged at the extent it was
    // created at and refunded by `finalize_surface`'s depth retire arm. There
    // is no resolve companion here, so above one sample the one texture is
    // simply that much larger.
    obj.inner().register_standalone_surface(
        width,
        height,
        format,
        u32::from(multi_sample.sample_count),
        mtld3d_core::format::StandaloneSurfaceKind::DepthStencil,
        scale,
    );
    let surf_ptr = Box::into_raw(Box::new(surf));
    // SAFETY: `surf_ptr` is a freshly created, live standalone depth-stencil
    // surface at refcount 1.
    unsafe { crate::com_ref::com_register_child(surf_ptr) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { OutPtr::write_opt(surface, surf_ptr.cast::<c_void>()) };
    D3D_OK
}

/// Publish a staging write to `tex`: at the next draw's re-walk, or now for a render target.
///
/// Bind-time `flush_dirty_mips` only runs while the API thread rebuilds a
/// dirty snapshot, so a write that lands in a texture's staging between two
/// draws over otherwise-clean state has to dirty the snapshot itself or its
/// upload is never scheduled and the next draw samples the old texels. The
/// mark is deliberately coarse and does not ask whether the texture is bound:
/// a redundant snapshot re-emit dedups at the encoder. A render-target texture
/// uploads now instead, in application order
/// (`crate::texture::publish_render_target_write`).
fn publish_staging_write(tex: &mut crate::texture::Direct3DTexture9) {
    let device_inner_ptr = tex.inner().device_inner();
    if device_inner_ptr != 0 {
        // SAFETY: live `DeviceInner*` recorded at the texture's create; the
        // device outlives every texture it owns (textures hold a device
        // refcount via their COM ABI).
        let dev = unsafe { &mut *(device_inner_ptr as *mut DeviceInner) };
        dev.mark_snapshot_dirty_all();
        crate::texture::publish_render_target_write(tex.inner_mut(), dev);
    }
}

/// Shared system-memory → default-pool staging-copy tail for `UpdateSurface` / `UpdateTexture`.
///
/// `src_parent` and
/// `dst_parent` are distinct, live `Direct3DTexture9` pointers; `copy` runs the
/// per-mip staging copies once pool and format are validated. Validates source
/// `D3DPOOL_SYSTEMMEM`, destination `D3DPOOL_DEFAULT`, and a format pair that is
/// either identical or one the CPU converter covers, then publishes the write
/// ([`publish_staging_write`]).
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
    if src_tex.d3d_pool() != D3DPOOL_SYSTEMMEM || dst_tex.d3d_pool() != D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "reject Update*: pool mismatch → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // D3D9 accepts a source and a destination of different formats and
    // converts. The CPU codec covers the simple uncompressed colour formats;
    // a pair outside it (block-compressed, depth, YUV destinations) has no
    // conversion to run, so it is the one mismatch still rejected.
    let (src_fmt, dst_fmt) = (src_tex.d3d_format(), dst_tex.d3d_format());
    if src_fmt != dst_fmt && !mtld3d_core::pixel_convert::can_convert_update(src_fmt, dst_fmt) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject Update*: no conversion for src=0x{src_fmt:x} into dst=0x{dst_fmt:x} → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: re-borrow of the immutable src inner via a raw pointer; the
    // allocation is distinct from the dst inner (src_parent != dst_parent).
    let src_inner = unsafe { &*core::ptr::from_ref(src_tex.inner()) };
    let hr = copy(dst_tex.inner_mut(), src_inner);
    // A copy that failed partway has still written the levels before the one
    // that failed, and those owe their upload like any other.
    publish_staging_write(dst_tex);
    hr
}

extern "system" fn device_update_surface(
    this: *mut c_void,
    src: *mut c_void,
    src_rect: *const c_void,
    dst: *mut c_void,
    dst_point: *const c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable args; live `Direct3DSurface9` pointers per the ABI.
    let Some(src_surf) = (unsafe { InPtr::<crate::surface::Direct3DSurface9>::opt(src) }) else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateSurface: null source surface → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: as above.
    let Some(dst_surf) = (unsafe { InPtr::<crate::surface::Direct3DSurface9>::opt(dst) }) else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateSurface: null destination surface → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    if let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        && obj.inner().frame_dump.active
    {
        obj.inner().frame_dump_event(&format!(
            "UpdateSurface(src={}, dst={})",
            frame_dump::surface_label(src),
            frame_dump::surface_label(dst)
        ));
    }
    // D3D9 rejects UpdateSurface when either endpoint is mapped, and a held
    // device context maps the surface exactly as a `LockRect` does.
    if src_surf.is_locked()
        || dst_surf.is_locked()
        || src_surf.has_open_dc()
        || dst_surf.has_open_dc()
    {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateSurface: a locked or DC-held source/destination surface → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let src_parent = src_surf.parent_texture();
    let dst_parent = dst_surf.parent_texture();
    // A planar YUV surface is a `StretchRect` source and nothing else. Its
    // level is taller than its logical height, which none of the staging
    // copies below size for, so a planar endpoint is refused before any of
    // them can run, whatever the other endpoint is.
    for parent in [src_parent, dst_parent] {
        // SAFETY: a non-null parent is the live `Direct3DTexture9` the surface
        // holds a reference on.
        if let Some(tex) = unsafe { parent.as_ref() }
            && mtld3d_core::stretch_rect::is_planar_yuv(tex.d3d_format())
        {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: planar YUV endpoint ({}) → INVALIDCALL",
                mtld3d_core::format::format_name(tex.d3d_format())
            );
            return D3DERR_INVALIDCALL;
        }
    }
    // A standalone offscreen *source* surface (not texture-backed) updating a
    // texture destination: copy its CPU backing into the dst texture's mip
    // staging and mark it dirty so a subsequent bind / StretchRect uploads it.
    if src_parent.is_null()
        && !dst_parent.is_null()
        && let Some((src_ptr, src_len, src_pitch, src_w, src_h, src_fmt)) =
            src_surf.system_memory_source()
    {
        // An `UpdateSurface` source is a `D3DPOOL_SYSTEMMEM` surface. A
        // `D3DPOOL_SCRATCH` one carries the same CPU backing but is not a
        // device resource, so it is never a valid endpoint of a device copy.
        if src_surf.standalone_pool() != D3DPOOL_SYSTEMMEM {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: standalone source outside D3DPOOL_SYSTEMMEM → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        let dst_level = dst_surf.mip_level() as usize;
        if map_d3d_format(src_fmt).is_none() {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: unmapped source format {} (0x{src_fmt:x}) → INVALIDCALL",
                mtld3d_core::format::format_name(src_fmt)
            );
            return D3DERR_INVALIDCALL;
        }
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
        // An `UpdateSurface` destination is a `D3DPOOL_DEFAULT` surface; a
        // managed or system-memory destination has its own upload path and
        // never takes this one.
        if tex.d3d_pool() != D3DPOOL_DEFAULT {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: destination pool is not D3DPOOL_DEFAULT → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        // D3D9 accepts a source and a destination of different formats and
        // converts. The copy below re-encodes such a pair through the CPU
        // codec, which covers the simple uncompressed colour formats; a pair
        // outside it has no conversion to run, so it is the one mismatch still
        // rejected.
        let dst_fmt = tex.d3d_format();
        if src_fmt != dst_fmt && !mtld3d_core::pixel_convert::can_convert_update(src_fmt, dst_fmt) {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: no conversion for src=0x{src_fmt:x} into dst=0x{dst_fmt:x} → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        // The same region rules as a texture-level source: the rect inside the
        // source, the copy inside the destination level, and block-aligned
        // origins and extents for a block-compressed pair.
        if !tex
            .inner()
            .update_region_valid(dst_level, (src_w, src_h), rect, point)
        {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: standalone source rect/destination point out of bounds → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        let image = SourceImage {
            bytes: src_bytes,
            pitch: src_pitch,
            width: src_w,
            height: src_h,
            format: src_fmt,
        };
        let copied = if let Some(face) = dst_surf.cube_face() {
            tex.inner_mut()
                .update_bytes_to_cube_staging_region(face, dst_level, &image, rect, point)
        } else {
            tex.inner_mut()
                .update_bytes_to_staging_region(dst_level, &image, rect, point)
        };
        if !copied {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: staging copy failed → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        publish_staging_write(tex);
        return D3D_OK;
    }
    // A render-target surface or the back buffer: a colour surface with no
    // texture behind it, whose Metal texture takes the region directly.
    if dst_parent.is_null() && !dst_surf.metal_color_handle().is_null() {
        // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
        let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
            return D3DERR_INVALIDCALL;
        };
        // SAFETY: optional *const RECT / *const POINT per the ABI; null → None.
        let rect = (unsafe { ValueIn::<mtld3d_types::D3DRECT>::read_opt(src_rect) })
            .map(|r| (r.x1, r.y1, r.x2, r.y2));
        // SAFETY: as above; POINT is two i32 (x, y).
        let point =
            (unsafe { ValueIn::<[i32; 2]>::read_opt(dst_point) }).map_or((0, 0), |p| (p[0], p[1]));
        return update_surface_into_color_target(obj.inner(), &src_surf, &dst_surf, rect, point);
    }
    if src_parent.is_null() || dst_parent.is_null() || std::ptr::eq(src_parent, dst_parent) {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateSurface: endpoints are not two distinct textures → INVALIDCALL"
        );
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
        let src_extent = (src.mip_width(src_level), src.mip_height(src_level));
        if !dst.update_region_valid(dst_level, src_extent, rect, point) {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: source rect/destination point out of bounds → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        let Some(image) = src.surface_source_image(src_surf.cube_face(), src_level) else {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: source surface has no staging → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        };
        let copied = if let Some(dst_face) = dst_surf.cube_face() {
            dst.update_bytes_to_cube_staging_region(dst_face, dst_level, &image, rect, point)
        } else {
            dst.update_bytes_to_staging_region(dst_level, &image, rect, point)
        };
        if !copied {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject UpdateSurface: staging copy failed → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        D3D_OK
    })
}

/// `UpdateSurface` into a render-target surface or the back buffer.
///
/// Both are `D3DPOOL_DEFAULT` colour surfaces with no texture behind them, so
/// the region goes to their Metal texture as tightly packed rows, ordered
/// among the application's passes on the encoder thread. The source is a
/// `D3DPOOL_SYSTEMMEM` surface, standalone or a texture level, its region is
/// held to the rules a texture destination applies, and a pair of two
/// formats converts where the update codec covers it, as it does into a
/// texture. A multisampled destination is refused, as D3D9 refuses one.
fn update_surface_into_color_target(
    dev: &mut DeviceInner,
    src_surf: &crate::surface::Direct3DSurface9,
    dst_surf: &crate::surface::Direct3DSurface9,
    src_rect: Option<(i32, i32, i32, i32)>,
    dst_point: (i32, i32),
) -> i32 {
    // Keyed by the reason, so each one is warned once.
    let reject = |key: u64, why: &str| {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: key,
            "reject UpdateSurface into a render target or back buffer: {why} → INVALIDCALL"
        );
        D3DERR_INVALIDCALL
    };
    if dst_surf.multi_sample().sample_count > 1 {
        return reject(1, "the destination is multisampled");
    }
    let src_parent = src_surf.parent_texture();
    let image = if src_parent.is_null() {
        if src_surf.standalone_pool() != D3DPOOL_SYSTEMMEM {
            return reject(2, "the source is not D3DPOOL_SYSTEMMEM");
        }
        let Some((ptr, len, pitch, width, height, format)) = src_surf.system_memory_source() else {
            return reject(3, "the source has no system-memory backing");
        };
        SourceImage {
            // SAFETY: `ptr`/`len` describe the live system-memory backing of
            // the source surface, which outlives this call.
            bytes: unsafe { std::slice::from_raw_parts(ptr, len) },
            pitch,
            width,
            height,
            format,
        }
    } else {
        // SAFETY: a non-null parent is the live `Direct3DTexture9` the source
        // surface holds a reference on.
        let tex = unsafe { &*src_parent };
        if tex.d3d_pool() != D3DPOOL_SYSTEMMEM {
            return reject(4, "the source is not D3DPOOL_SYSTEMMEM");
        }
        let Some(image) = tex
            .inner()
            .surface_source_image(src_surf.cube_face(), src_surf.mip_level() as usize)
        else {
            return reject(5, "the source level has no staging");
        };
        image
    };
    let dst_format = dst_surf.standalone_format();
    let dst_extent = (dst_surf.standalone_width(), dst_surf.standalone_height());
    // The rows go to the Metal texture as they are, so the destination's
    // storage has to be its D3D format's own layout, which every format a
    // render target or back buffer is created in has.
    let Some(dst_mapping) =
        crate::direct3d9::map_for_device(dst_format, dev.config().expand_packed16).filter(|m| {
            !mtld3d_core::upload_pass::is_expanded_upload(dst_format, m.metal_pixel_format())
        })
    else {
        return reject(
            6,
            "the destination format has no storage that takes its rows",
        );
    };
    if image.format != dst_format
        && !mtld3d_core::pixel_convert::can_convert_update(image.format, dst_format)
    {
        return reject(7, "no conversion between the two formats");
    }
    let Some((region, origin)) = mtld3d_core::dirty_rect::update_surface_region(
        src_rect,
        dst_point,
        (image.width, image.height),
        dst_extent,
        (1, 1),
    ) else {
        return reject(8, "the source rect or destination point leaves a surface");
    };
    let bpp = dst_mapping.bytes_per_pixel() as usize;
    let row_bytes = region.w as usize * bpp;
    let mut rows = vec![0u8; row_bytes * region.h as usize];
    if image.format == dst_format {
        for (row, out) in (region.y as usize..).zip(rows.chunks_exact_mut(row_bytes)) {
            let start = row * image.pitch + region.x as usize * bpp;
            let Some(src_row) = image.bytes.get(start..start + row_bytes) else {
                return reject(9, "the source backing is shorter than its rows");
            };
            out.copy_from_slice(src_row);
        }
    } else {
        let convert = mtld3d_core::pixel_convert::ConvertRegion {
            src_x: region.x,
            src_y: region.y,
            dst_x: 0,
            dst_y: 0,
            width: region.w,
            height: region.h,
            src_pitch: image.pitch,
            dst_pitch: row_bytes,
            src_slice_pitch: image.pitch * image.height as usize,
            dst_slice_pitch: rows.len(),
            depth: 1,
        };
        if !mtld3d_core::pixel_convert::convert_region(
            &mut rows,
            dst_format,
            image.bytes,
            image.format,
            &convert,
        ) {
            return reject(10, "the conversion ran past the source backing");
        }
    }
    let scale = dst_surf.render_scale();
    let texture = TargetExtent::whole(scale, dst_extent).texture();
    // SAFETY: the captured token moves directly into this frame's operation.
    let bytes = unsafe { dev.capture_frame_bytes(&rows) };
    dev.push_control(UpdateColorRegionOp {
        target: mtld3d_core::encoder_data::ColorRegionUpdate {
            color_handle: dst_surf.metal_color_handle().raw(),
            format: dst_mapping.metal_pixel_format(),
            origin,
            extent: (region.w, region.h),
            logical: dst_extent,
            texture,
            scale,
            bytes_per_row: u32::try_from(row_bytes).expect("a surface row fits u32"),
        },
        bytes,
    });
    D3D_OK
}

extern "system" fn device_update_texture(
    this: *mut c_void,
    src: *mut c_void,
    dst: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if src.is_null() || dst.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateTexture: null source or destination texture → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // All texture interfaces share this wrapper layout. The type flag below
    // selects the cube path; volumes take the plain path, whose staging copy
    // walks every depth slice of a level.
    let src_parent = src.cast::<crate::texture::Direct3DTexture9>();
    let dst_parent = dst.cast::<crate::texture::Direct3DTexture9>();
    if std::ptr::eq(src_parent.cast_const(), dst_parent.cast_const()) {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateTexture: source and destination are one texture → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    if let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) })
        && obj.inner().frame_dump.active
    {
        // SAFETY: `src_parent` is a non-null live base-texture wrapper.
        let src_id = unsafe { (*src_parent).texture_id() };
        // SAFETY: `dst_parent` is a non-null live base-texture wrapper.
        let dst_id = unsafe { (*dst_parent).texture_id() };
        obj.inner()
            .frame_dump_event(&format!("UpdateTexture(src={src_id:?}, dst={dst_id:?})"));
    }
    // D3D9 pairs the two resources by type: a 2D texture only updates a 2D
    // texture, a cube a cube, a volume a volume. The container reports the
    // type it was created as, which is not the kind of the backing Metal
    // texture: a single-slice volume texture is backed 2D and is still a
    // volume here.
    // SAFETY: both pointers are non-null live base-texture wrappers. All three
    // texture interfaces share the same wrapper layout.
    let src_type = unsafe { (*src_parent).d3d_resource_type() };
    // SAFETY: same invariant as the source pointer above.
    let dst_type = unsafe { (*dst_parent).d3d_resource_type() };
    if src_type != dst_type {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject UpdateTexture: source and destination resource types differ → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let src_is_cube = src_type == D3DRTYPE_CUBETEXTURE;
    if src_is_cube {
        let hr = copy_systemmem_to_default(dst_parent, src_parent, |dst, src| {
            let mut s = src.mip_width(0);
            let d = dst.mip_width(0);
            let mut src_skip = 0usize;
            while s > d {
                s >>= 1;
                src_skip += 1;
            }
            let levels = (src.app_level_count() as usize)
                .saturating_sub(src_skip)
                .min(dst.app_level_count() as usize);
            for face in 0..6 {
                for level in 0..levels {
                    let src_level = level + src_skip;
                    let Some(dr) = src.cube_update_dirty_rect(face, src_level) else {
                        continue;
                    };
                    let sw = src.mip_width(src_level);
                    let sh = src.mip_height(src_level);
                    let Some(c) = dr.clamp(sw, sh) else { continue };
                    if !c.reaches(dst.mip_width(level), dst.mip_height(level)) {
                        continue;
                    }
                    let copied = if c.x == 0 && c.y == 0 && c.w >= sw && c.h >= sh {
                        dst.update_cube_sub_region_from(
                            (face, level),
                            src,
                            face,
                            src_level,
                            None,
                            (0, 0),
                        )
                    } else {
                        let rect = (
                            c.x.cast_signed(),
                            c.y.cast_signed(),
                            (c.x + c.w).cast_signed(),
                            (c.y + c.h).cast_signed(),
                        );
                        dst.update_cube_sub_region_from(
                            (face, level),
                            src,
                            face,
                            src_level,
                            Some(rect),
                            (c.x.cast_signed(), c.y.cast_signed()),
                        )
                    };
                    if !copied {
                        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                            "reject UpdateTexture: cube staging copy failed → INVALIDCALL");
                        return D3DERR_INVALIDCALL;
                    }
                }
            }
            D3D_OK
        });
        if hr == D3D_OK {
            // SAFETY: `src_parent` remains live and distinct from `dst_parent`.
            unsafe { (*src_parent).inner_mut() }.clear_all_cube_update_dirty();
        }
        return hr;
    }
    let hr = copy_systemmem_to_default(dst_parent, src_parent, |dst, src| {
        // D3D9 matches src/dst mips by aligning the SMALLEST levels: when the
        // source top-level is larger than the destination's, skip the extra
        // source mips so source level `src_skip` lines up with dst level 0,
        // per the D3D9 UpdateTexture smallest-mip alignment rule. For an
        // equal-or-smaller source `src_skip` stays 0 and this is identical to
        // a plain index-aligned copy.
        let mut s = src
            .mip_width(0)
            .max(src.mip_height(0))
            .max(src.mip_depth(0));
        let d = dst
            .mip_width(0)
            .max(dst.mip_height(0))
            .max(dst.mip_depth(0));
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
            if !c.reaches(dst.mip_width(level), dst.mip_height(level)) {
                continue;
            }
            let copied = if c.x == 0 && c.y == 0 && c.w >= sw && c.h >= sh {
                // Whole mip.
                dst.update_sub_region_from(level, src, src_level, None, (0, 0))
            } else {
                let rect = (
                    c.x.cast_signed(),
                    c.y.cast_signed(),
                    (c.x + c.w).cast_signed(),
                    (c.y + c.h).cast_signed(),
                );
                dst.update_sub_region_from(
                    level,
                    src,
                    src_level,
                    Some(rect),
                    (c.x.cast_signed(), c.y.cast_signed()),
                )
            };
            if !copied {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "reject UpdateTexture: staging copy failed → INVALIDCALL");
                return D3DERR_INVALIDCALL;
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
        // Eager upload for a destination the game cannot lock: the copy just
        // wrote the whole payload, and flushing now (instead of at first
        // bind) lets the level's staging drop immediately. A texture the
        // game updates but never draws would otherwise hold its copy
        // indefinitely.
        // SAFETY: `dst_parent` is the live destination texture and `this`
        // the device; distinct allocations, both live for this call.
        let dst_ti = unsafe { (*dst_parent).inner_mut() };
        // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
        let dev_obj = unsafe { InPtrMut::<Direct3DDevice9>::opt(this) };
        if dst_ti.d3d_pool() == D3DPOOL_DEFAULT
            && dst_ti.d3d_usage() & D3DUSAGE_DYNAMIC == 0
            && let Some(obj) = dev_obj
        {
            crate::texture::flush_dirty_mips(dst_ti, obj.inner());
        }
    }
    hr
}

/// Reject a read-back whose destination cannot take the source; `None` accepts it.
///
/// One warn per distinct reason, naming the entry point that reached it first,
/// since an application that reads back every frame retries a rejected call
/// every frame.
fn reject_readback(
    entry_point: &str,
    src: &ReadbackSource,
    dst: &SystemMemoryDst,
) -> Option<ReadbackReject> {
    let reason = mtld3d_core::readback::reject_readback_dst(
        src,
        &ReadbackDestination {
            width: dst.width,
            height: dst.height,
            format: dst.format,
            bytes_per_row: dst.bytes_per_row,
            len: dst.len,
        },
    )?;
    warn_readback_rejected(entry_point, reason);
    Some(reason)
}

/// One `log_once_warn_by!` line per (entry point, reason) pair.
fn warn_readback_rejected(entry_point: &str, reason: ReadbackReject) {
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: reason.key(),
        "reject {entry_point}: {} → INVALIDCALL",
        reason.as_str()
    );
}

extern "system" fn device_get_render_target_data(
    this: *mut c_void,
    rt: *mut c_void,
    dst: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
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
    if obj.inner().frame_dump.active {
        obj.inner().frame_dump_event(&format!(
            "GetRenderTargetData(src={}, dst={})",
            frame_dump::surface_label(rt),
            frame_dump::surface_label(dst)
        ));
    }
    // A multisampled source has no per-pixel value to hand back: D3D9 makes
    // the application resolve it with `StretchRect` into a single-sampled
    // surface first, and rejects the call outright.
    if src.multi_sample().sample_count > 1 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: source is multisampled → INVALIDCALL (StretchRect it into a single-sampled surface first)");
        return D3DERR_INVALIDCALL;
    }
    let Some(dst_desc) = dst_surf.system_memory_blit_dst() else {
        warn_readback_rejected("GetRenderTargetData", ReadbackReject::NotSystemMemory);
        return D3DERR_INVALIDCALL;
    };
    // A source with no persistent colour handle is either a render-target
    // texture level, whose handle lives encoder-side, or not a render target
    // at all.
    let src_handle = src.metal_color_handle();
    let hr = if src_handle.is_null() {
        readback_from_texture_rt(&obj, &src, &dst_desc)
    } else {
        if map_d3d_format(src.standalone_format()).is_none() {
            warn_readback_rejected("GetRenderTargetData", ReadbackReject::FormatMismatch);
            return D3DERR_INVALIDCALL;
        }
        let source = ReadbackSource {
            width: src.standalone_width(),
            height: src.standalone_height(),
            format: src.standalone_format(),
        };
        if reject_readback("GetRenderTargetData", &source, &dst_desc).is_some() {
            return D3DERR_INVALIDCALL;
        }
        blit_texture_to_systemmem(
            obj.inner(),
            &SystemMemReadback::for_readback(
                src_handle,
                // A standalone colour surface is one image: mip 0, slice 0.
                (0, 0),
                (source.width, source.height),
                &dst_desc,
            ),
        )
    };
    if hr == D3D_OK {
        dst_surf.note_system_memory_written();
    }
    hr
}

/// `GetRenderTargetData` from a level of a `D3DUSAGE_RENDERTARGET` texture.
///
/// D3D9 accepts one and returns its content; a non-render-target texture
/// source stays `INVALIDCALL`. Its Metal handle lives on the encoder thread
/// keyed by texture id, so one op resolves it *and* notes it read-back before
/// the flush finalizes the frame (the store optimiser would otherwise discard
/// a colour store nothing sampled in-frame), then hands it back through an
/// atomic slot.
fn readback_from_texture_rt(
    dev: &Direct3DDevice9,
    src: &Direct3DSurface9,
    dst: &SystemMemoryDst,
) -> i32 {
    let parent = src.parent_texture();
    if parent.is_null() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: source is not a render target → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: `parent` is a live `Direct3DTexture9` (its refcount keeps it
    // alive while the surface is alive).
    let tex = unsafe { &*parent };
    if tex.d3d_usage() & D3DUSAGE_RENDERTARGET == 0 || tex.d3d_pool() != D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: source is not a render target → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    if map_d3d_format(tex.d3d_format()).is_none() {
        warn_readback_rejected("GetRenderTargetData", ReadbackReject::FormatMismatch);
        return D3DERR_INVALIDCALL;
    }
    // The surface names one subresource of the parent texture: a cube face
    // rides the Metal array slice, a `GetSurfaceLevel` / `GetCubeMapSurface`
    // level the mip.
    let level = src.mip_level();
    let slice = src.cube_face().unwrap_or(0);
    let ti = tex.inner();
    let source = ReadbackSource {
        width: ti.mip_width(level as usize),
        height: ti.mip_height(level as usize),
        format: tex.d3d_format(),
    };
    if reject_readback("GetRenderTargetData", &source, dst).is_some() {
        return D3DERR_INVALIDCALL;
    }
    let texture_id = ti.texture_info().texture_id;
    // The texture's own logical extent, which the mip extent above is measured
    // against. Taken here because the flush below reborrows the texture.
    let full_extent = (ti.mip_width(0), ti.mip_height(0));
    // Past every gate that rejects the call, so the uploads scheduled here are
    // work this read will use. A level of a non-dynamic DEFAULT-pool texture is
    // lockable here, so the source can carry a CPU write no bind has uploaded
    // yet, and the read below reads the Metal texture alone: an upload left for
    // a later bind would land after it and the read would hand back the older
    // GPU content. `flush_dirty_mips_for_gpu_write` does the same for a
    // surface a GPU write claims, but it walks the surface to its parent and
    // rehydrates it first, and this path holds the validated parent already,
    // whose DEFAULT pool leaves the rehydration nothing to do.
    // SAFETY: `parent` is non-null (checked above) and points to a live
    // `Direct3DTexture9` whose refcount keeps it alive while the surface is.
    let parent_inner = unsafe { (*parent).inner_mut() };
    crate::texture::flush_dirty_mips(parent_inner, dev.inner());
    parent_inner.note_gpu_use();
    let slot = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let slot_op = std::sync::Arc::clone(&slot).into();
    dev.inner().push_control(crate::device::ReadColorHandleOp {
        texture_id,
        slot_op,
    });
    if let Err(hr) = dev.inner().flush_current_frame_blocking() {
        return hr;
    }
    let h = slot.load(std::sync::atomic::Ordering::Acquire);
    if h == 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "GetRenderTargetData: texture-RT handle unresolved → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    blit_handle_to_systemmem(
        dev.inner(),
        &SystemMemReadback::for_readback(
            // SAFETY: `h` is non-zero (checked above) and a live retained
            // MTLTexture handle from the encoder texture cache.
            unsafe { MetalHandle::<MTLTextureKind>::new(h) },
            (level, slice),
            full_extent,
            dst,
        ),
    )
}

extern "system" fn device_get_front_buffer_data(
    this: *mut c_void,
    swap_chain: u32,
    dst_surface: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // Only the implicit swapchain is addressed by this device entry point.
    if swap_chain != 0 {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject GetFrontBufferData swapchain={swap_chain} → INVALIDCALL (expected 0)");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `dst_surface` is a caller-owned IDirect3DSurface9* per the ABI.
    let Some(dst_surf) = (unsafe { InPtr::<Direct3DSurface9>::opt(dst_surface) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().read_front_buffer(&dst_surf)
}

/// Flush pending GPU work, then blit a Metal color texture into system memory.
///
/// Shared by `GetRenderTargetData` / `GetFrontBufferData`; mirrors the
/// per-`LockRect` backbuffer readback in `surface.rs`. The destination has
/// already been checked against the source by `reject_readback`.
fn blit_texture_to_systemmem(device_inner: &mut DeviceInner, read: &SystemMemReadback) -> i32 {
    let src = read.tex_handle;
    // This blit reads the RT right after the flush. Mark it read-back BEFORE
    // the flush so the store-action rules treat it as live and never discard
    // its colour store.
    device_inner.push_control(crate::device::NoteColorReadOp { src });
    if let Err(hr) = device_inner.flush_current_frame_blocking() {
        return hr;
    }
    blit_handle_to_systemmem(device_inner, read)
}

/// Parameters for one synchronous read of a Metal texture into system memory.
///
/// The read starts at the texture's origin and covers `width` x `height` of mip
/// `level` in array slice `slice`, so the extent belongs to that mip.
/// `full_width` / `full_height` are the texture's own logical extent instead,
/// which is what the source is measured against on the way out.
pub struct SystemMemReadback {
    pub tex_handle: MetalHandle<MTLTextureKind>,
    /// Page-aligned PE-addressable destination.
    pub dst_ptr: u64,
    /// Page-multiple length of `dst_ptr`.
    pub dst_len: u64,
    pub level: u32,
    /// Array slice the read addresses within the texture.
    ///
    /// A cube face index for a cube-backed source, zero for every other class,
    /// none of which carries a second slice.
    pub slice: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    /// Logical width of the texture the read is measured against.
    ///
    /// Equal to `width` for a mip-0 read. A scaled back buffer's texture is
    /// smaller than this and the unix side resolves it up first; every other
    /// source matches and the resolve is skipped.
    pub full_width: u32,
    /// Logical height of the texture the read is measured against.
    ///
    /// See [`Self::full_width`].
    pub full_height: u32,
}

impl SystemMemReadback {
    /// The read a `GetRenderTargetData` / `GetFrontBufferData` performs.
    ///
    /// Call it only on a source and destination `reject_readback` accepted:
    /// their extents agree there, so the whole image is one region and the
    /// destination's own row stride carries it. `full_width` / `full_height`
    /// are the source texture's logical extent, which `level` is measured
    /// against. `subresource` is the `(level, slice)` the source surface names.
    const fn for_readback(
        tex_handle: MetalHandle<MTLTextureKind>,
        (level, slice): (u32, u32),
        (full_width, full_height): (u32, u32),
        dst: &SystemMemoryDst,
    ) -> Self {
        Self {
            tex_handle,
            dst_ptr: dst.ptr,
            dst_len: dst.len,
            level,
            slice,
            width: dst.width,
            height: dst.height,
            bytes_per_row: dst.bytes_per_row,
            full_width,
            full_height,
        }
    }
}

/// Synchronous `MTLTexture`→system-memory blit.
///
/// Emits `copyFromTexture:toBuffer:` + `waitUntilCompleted`. The caller must
/// have already noted the source for read-back and flushed the frame, so this
/// is the bare data-movement step shared by the standalone-colour-handle path,
/// the texture-RT path and the released-staging refill.
pub fn blit_handle_to_systemmem(device_inner: &DeviceInner, read: &SystemMemReadback) -> i32 {
    let mut params = BlitTextureToBufferParams {
        planes: mtld3d_shared::mtl::ReadbackPlanes::Color,
        stencil_bytes_per_row: 0,
        stencil_offset: 0,
        record_handle: device_inner.record_handle(),
        device_handle: device_inner.device_handle(),
        tex_handle: read.tex_handle,
        dst_ptr: read.dst_ptr,
        dst_len: read.dst_len,
        mip_level: read.level,
        slice: read.slice,
        origin_x: 0,
        origin_y: 0,
        width: read.width,
        height: read.height,
        bytes_per_row: read.bytes_per_row,
        source_width: read.full_width,
        source_height: read.full_height,
        // Render-target colour formats are all uncompressed, so a block row is
        // a pixel row.
        block_height: 1,
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

    let _api = device_api_lock(this);
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

    let dev = obj.inner();
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "StretchRect(src={}, dst={}, filter={filter})",
            frame_dump::surface_label(src),
            frame_dump::surface_label(dst)
        ));
    }

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
    // copy is a same-format Private→Private depth blit, or a depth resolve
    // when the source carries samples the destination does not.
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
        // Sample counts decide the transport. An equal pair is a copy. A
        // multisampled source into a single-sampled destination is the resolve
        // D3D9 defines for reading a multisampled surface. Neither remaining
        // pair has a Metal shape: the blit encoder cannot change the sample
        // count, and the resolve unit only ever reduces to one sample.
        let resolve = src_info.sample_count > 1 && dst_info.sample_count == 1;
        if src_info.sample_count != dst_info.sample_count && !resolve {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "reject StretchRect: depth-stencil sample counts {} → {} are neither equal nor a \
                 resolve → INVALIDCALL",
                src_info.sample_count,
                dst_info.sample_count
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
        // Past every depth-stencil gate, so the endpoints' pending uploads are
        // work this call will use.
        flush_dirty_mips_for_gpu_write(&obj, &[src_surf, dst_surf]);
        if resolve {
            // The samples are reduced by a depth transfer queued after the
            // passes recorded so far, which also enters the source into the
            // load/store model as read and the destination as written. Both
            // endpoints are standalone depth surfaces, the only shape that
            // carries `IS_DEPTH_STENCIL`.
            let (
                StretchKind::DepthStencil(src_handle),
                StretchKind::DepthStencil(dst_handle),
                Some(source_format),
                Some(destination_format),
            ) = (
                &src_info.kind,
                &dst_info.kind,
                mtld3d_core::format::map_d3d_depth_format(src_info.format),
                mtld3d_core::format::map_d3d_depth_format(dst_info.format),
            )
            else {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "reject StretchRect: a depth-stencil endpoint has no depth texture → INVALIDCALL"
                );
                return D3DERR_INVALIDCALL;
            };
            let transfer = crate::encoder::DepthTransfer {
                source: *src_handle,
                source_level: 0,
                source_size: src_info.texture_size,
                source_format,
                source_samples: src_info.sample_count,
                destination: *dst_handle,
                destination_size: dst_info.texture_size,
                destination_format,
            };
            if dev.frame_dump.active {
                dev.frame_dump_event("StretchRect: multisampled depth resolve queued");
            }
            dev.push_control(crate::device::ResolveDepthSurfaceOp { transfer });
            return D3D_OK;
        }
        // Same-format Private→Private depth copy on the 1:1 blit path. The
        // blit is entered into the load/store model like a colour copy: the
        // source counts as read (its last pass keeps its depth store) and the
        // destination counts as blit-written (its next pass loads rather than
        // discards), and a clear still waiting for a pass on either endpoint
        // is materialized before the copy.
        let mip_level = src_info.mip_level;
        if dev.frame_dump.active {
            dev.frame_dump_event("StretchRect: full-surface depth copy queued");
        }
        dev.push_control(crate::device::StretchBlitOp {
            src_info,
            dst_info,
            src_region,
            dst_region,
            mip_level,
            render_quad: false,
            filter,
        });
        return D3D_OK;
    }

    // D3D9 StretchRect eligibility: both surfaces must be D3DPOOL_DEFAULT, and
    // the surface classes must form a pair `class_reject` allows (a texture
    // source is valid into a render target, the CAN_STRETCHRECT_FROM_TEXTURES
    // cap we advertise).
    if src_info.pool != D3DPOOL_DEFAULT || dst_info.pool != D3DPOOL_DEFAULT {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "reject StretchRect: src/dst not D3DPOOL_DEFAULT → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let rects_given = !src_rect.is_null() || !dst_rect.is_null();
    // An extended device also copies between two default-pool surfaces of
    // no eligible class, a texture level that is no render target included,
    // when the call is a whole-surface copy of one format and size.
    let extended_copy = mtld3d_core::stretch_rect::extended_whole_surface_copy(
        dev.is_extended(),
        rects_given,
        (src_info.format, src_info.width, src_info.height),
        (dst_info.format, dst_info.width, dst_info.height),
    );
    if let Some(reason) = mtld3d_core::stretch_rect::class_reject(src_info.flags, dst_info.flags)
        && !extended_copy
    {
        log_pair_reject(
            reason,
            (src, &src_info, src_rect),
            (dst, &dst_info, dst_rect),
            dev.is_extended(),
        );
        return D3DERR_INVALIDCALL;
    }

    let expand_packed16 = dev.config().expand_packed16;
    // The planar YUV formats have no entry in the format lookup, so the
    // generic format gate would refuse them as unmappable. A pair with a
    // planar endpoint is judged by its own route below, once the rects say
    // whether the copy scales.
    let planar_endpoint = mtld3d_core::stretch_rect::is_planar_yuv(src_info.format)
        || mtld3d_core::stretch_rect::is_planar_yuv(dst_info.format);
    if !planar_endpoint
        && let Err(hr) = check_stretch_rect_formats(&src_info, &dst_info, expand_packed16)
    {
        return hr;
    }
    let Some((src_region, dst_region)) =
        parse_stretch_regions(src_rect, dst_rect, &src_info, &dst_info)
    else {
        return D3DERR_INVALIDCALL;
    };

    let scaling = src_region.w != dst_region.w || src_region.h != dst_region.h;
    // D3D9 copies between two rects of one surface only 1:1. A surface is one
    // COM object per subresource, so two levels or two faces of one texture are
    // two surfaces and may stretch. So may an additional swap chain's back
    // buffer and the device's, which alias one texture here but are two
    // surfaces to the application. The back-buffer indices of one swap chain
    // are the exception: this layer hands every index the same surface object
    // over one texture, so a stretch between two of them, which D3D9 allows,
    // is refused here as a stretch inside one surface.
    if scaling && core::ptr::eq(src, dst) {
        log_pair_reject(
            RejectReason::SameSurfaceStretch,
            (src, &src_info, src_rect),
            (dst, &dst_info, dst_rect),
            dev.is_extended(),
        );
        return D3DERR_INVALIDCALL;
    }
    // A planar source decodes through the render quad into a render target and
    // through the CPU converter, 1:1, into an offscreen plain; both are the
    // cross-format branches below. Everything else with a planar endpoint is
    // refused here, ahead of the upload flush, so a rejected call schedules
    // nothing.
    if let mtld3d_core::stretch_rect::PlanarStretch::Reject(reason) =
        mtld3d_core::stretch_rect::planar_stretch_route(
            src_info.format,
            dst_info.format,
            dst_info
                .flags
                .contains(StretchSurfaceFlags::IS_RENDER_TARGET),
            scaling,
        )
    {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: reason.key(),
            "reject StretchRect: {} (src={} 0x{:x}, dst={} 0x{:x}) → INVALIDCALL",
            reason.as_str(),
            mtld3d_core::format::format_name(src_info.format),
            src_info.format,
            mtld3d_core::format::format_name(dst_info.format),
            dst_info.format
        );
        return D3DERR_INVALIDCALL;
    }
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
    // Past every gate that can reject the call, so the endpoints' pending
    // uploads are work this call will use.
    flush_dirty_mips_for_gpu_write(&obj, &[src_surf, dst_surf]);
    // A same-size copy that converts also needs the render-quad path (the 1:1
    // blit can't convert): a cross-Metal-format pair, or a source without
    // alpha into a destination with alpha sharing its storage, whose alpha
    // the quad forces to one while it samples (`BlitDecode::OpaqueAlpha` for
    // an X source; a widened source already stores alpha one).
    // `check_stretch_rect_formats` guaranteed a cross-format destination is a
    // render target or an offscreen-plain surface (cross-format
    // RT/texture/offscreen → RT, plus the offscreen→offscreen case handled on
    // the CPU just below).
    // Device-aware mapping: it must agree with the Metal formats the textures
    // were actually created with (e.g. a packed 16-bit pair that is
    // BGRA8-backed on this device is NOT cross-format).
    // A planar source always converts: its R8 storage holds YUV planes, not
    // the pixels of any colour format, so it never takes the 1:1 blit. It is
    // answered ahead of the lookup, which has no entry for it.
    let cross_format = planar_endpoint || !copies_bytes(&src_info, &dst_info, expand_packed16);

    // A cross-format 1:1 copy into an offscreen-plain destination has no GPU
    // path: the render-quad conversion needs a render-target destination, and
    // the 1:1 blit can't convert. Do it on the CPU: decode each source pixel
    // and re-encode into the destination texture's staging, then upload that
    // staging so a later sample (or same-format StretchRect out of it) and a
    // later LockRect both see the converted pixels. Do NOT push the render-quad
    // op, which would bind a non-render-target texture as a colour attachment.
    // This serves the offscreen pairs of two storages, the YUV decodes, and a
    // source without alpha into a destination with alpha, whose padding the
    // converter reads as alpha one.
    if cross_format
        && !scaling
        && dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT)
    {
        return convert_stretch_dst_staging(
            &obj, src_surf, dst_surf, &src_info, &dst_info, src_region, dst_region,
        );
    }
    let render_quad = scaling || cross_format;

    // Both transports below write only the destination's Metal texture, and a
    // texture-backed destination (an offscreen plain, a render-target texture
    // level or a cube face) reads its pixels back through the CPU staging a
    // `LockRect` or a `GetDC` maps. Claim that subresource for the GPU so the
    // next map reads the copy rather than the staging it left behind. The
    // render quad binds the destination as a colour attachment, so it claims
    // only a destination that carries `D3DUSAGE_RENDERTARGET`: claiming one the
    // quad cannot write would trade stale pixels for uninitialised ones. Every
    // render-quad pair that gets this far has such a destination, the
    // rejections above having answered `INVALIDCALL` (a scale into one that is
    // not a render target) or taken the CPU converter (a cross-format copy into
    // an offscreen plain).
    let dst_renderable = dst_info
        .flags
        .contains(StretchSurfaceFlags::IS_RENDER_TARGET);
    if matches!(dst_info.kind, StretchKind::Texture(_)) && (!render_quad || dst_renderable) {
        claim_dst_subresource_for_gpu(dst_surf, &dst_info);
    }

    let mip_level = src_info.mip_level;
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "StretchRect: {} queued",
            if render_quad {
                "render quad"
            } else {
                "1:1 blit"
            }
        ));
    }
    dev.push_control(crate::device::StretchBlitOp {
        src_info,
        dst_info,
        src_region,
        dst_region,
        mip_level,
        render_quad,
        filter,
    });
    D3D_OK
}

/// Warn once per rule that a `StretchRect` pair broke a rule on its two surfaces.
///
/// The rules are the surface-class matrix and the refusal of a stretch inside
/// one surface. The line names the rule, then for each endpoint its class,
/// usage, format, level extent, mip level and the rect the call passed, and
/// whether the device is extended, so a game log alone says which pair the
/// application asked for and why an extended device's whole-surface copy did
/// not take it. The arguments are formatted only when the line fires, so each
/// rule names the first pair that broke it.
fn log_pair_reject(
    reason: mtld3d_core::stretch_rect::RejectReason,
    src: (*mut c_void, &StretchSurfaceInfo, *const c_void),
    dst: (*mut c_void, &StretchSurfaceInfo, *const c_void),
    extended: bool,
) {
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: reason.key(),
        "reject StretchRect: {} (src={}, dst={}, {} device) → INVALIDCALL",
        reason.as_str(),
        describe_stretch_endpoint(src.0, src.1, src.2),
        describe_stretch_endpoint(dst.0, dst.1, dst.2),
        if extended { "extended" } else { "plain" }
    );
}

/// One `StretchRect` endpoint for a rejection line.
///
/// Reads as `texture level usage=0x0 A8R8G8B8 0x15 256x256 level 0 rect
/// whole`, the rect being the `D3DRECT` the call passed or `whole` for a null
/// one.
fn describe_stretch_endpoint(
    surface: *mut c_void,
    info: &StretchSurfaceInfo,
    rect: *const c_void,
) -> String {
    use crate::surface::Direct3DSurface9;

    // SAFETY: `surface` is the live IDirect3DSurface9 the game passed to this
    // `StretchRect` call.
    let usage = unsafe { InPtr::<Direct3DSurface9>::opt(surface) }.map_or(0, |s| s.d3d_usage());
    // SAFETY: `rect` is the `*const D3DRECT` in-param of the same call.
    let rect = unsafe { ValueIn::<mtld3d_types::D3DRECT>::read_opt(rect) }.map_or_else(
        || String::from("whole"),
        |r| format!("({}, {})-({}, {})", r.x1, r.y1, r.x2, r.y2),
    );
    format!(
        "{} usage=0x{usage:x} {} 0x{:x} {}x{} level {} rect {rect}",
        info.class_name(),
        mtld3d_core::format::format_name(info.format),
        info.format,
        info.width,
        info.height,
        info.mip_level
    )
}

/// Claim a `StretchRect` or `ColorFill` destination subresource for the GPU.
///
/// A texture-backed destination carries CPU staging, and both the 1:1 blit and
/// the render quad write only its Metal texture, so a `LockRect` or a `GetDC`
/// of that subresource has to read it back instead of serving the staging the
/// write never reached. The claim is per (face, level): a cube face's five
/// siblings keep whatever their own staging holds. Every source kind lands
/// here, including one with no CPU staging of its own to copy from. Marking is
/// all this does; the read happens at the next map.
fn claim_dst_subresource_for_gpu(
    dst_surf: *mut crate::surface::Direct3DSurface9,
    dst_info: &StretchSurfaceInfo,
) {
    if dst_surf.is_null() {
        return;
    }
    // SAFETY: caller-supplied live `Direct3DSurface9*` from the StretchRect or
    // ColorFill thunk (non-null checked above).
    let dst_parent = unsafe { (*dst_surf).parent_texture() };
    if dst_parent.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect/ColorFill: texture-backed destination has no parent texture → \
             its staging keeps what it held"
        );
        return;
    }
    // SAFETY: `dst_parent` is non-null (checked) and a live `Direct3DTexture9`
    // kept alive by the destination surface's reference.
    unsafe { &mut *dst_parent }
        .inner_mut()
        .mark_subresource_gpu_authoritative(
            dst_info.slice.unwrap_or(0),
            dst_info.mip_level as usize,
        );
}

/// CPU-side cross-format `StretchRect` into an offscreen-plain destination.
///
/// The 1:1 blit cannot convert formats, and the render-quad conversion needs a
/// render-target destination. Decode each source pixel and re-encode it into the
/// destination texture's staging, then schedule the staging→texture upload
/// (`flush_dirty_mips`) so a later sample sees the converted pixels; a later
/// `LockRect` reads the same converted staging. Both surfaces are offscreen-
/// plain here, so both are texture-backed. Unsupported pairs are rejected
/// before this path schedules an upload or changes destination contents.
fn convert_stretch_dst_staging(
    obj: &Direct3DDevice9,
    src_surf: *mut crate::surface::Direct3DSurface9,
    dst_surf: *mut crate::surface::Direct3DSurface9,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
) -> i32 {
    if src_surf.is_null() || dst_surf.is_null() {
        return D3DERR_INVALIDCALL;
    }
    if !mtld3d_core::pixel_convert::can_convert(src_info.format, dst_info.format) {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "StretchRect: cross-format offscreen pair (src=0x{:x}, dst=0x{:x}) not CPU-convertible → INVALIDCALL",
            src_info.format, dst_info.format);
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: caller-supplied live `Direct3DSurface9*` from the StretchRect
    // thunk (non-null checked above).
    let src_parent = unsafe { (*src_surf).parent_texture() };
    // SAFETY: as above.
    let dst_parent = unsafe { (*dst_surf).parent_texture() };
    if src_parent.is_null() || dst_parent.is_null() || src_parent == dst_parent {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: cross-format offscreen dst has no distinct texture backing → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: non-null (checked), distinct from `dst_parent`, and a live
    // `Direct3DTexture9` kept alive by the source surface's reference.
    let src_tex = unsafe { &mut *src_parent };
    if !src_tex.inner_mut().materialize_subresource_for_cpu_read(
        src_info.slice.unwrap_or(0),
        src_info.mip_level as usize,
    ) {
        return D3DERR_INVALIDCALL;
    }
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
            "StretchRect: cross-format offscreen pair (src=0x{:x}, dst=0x{:x}) staging conversion failed → INVALIDCALL",
            src_info.format,
            dst_info.format
        );
        return D3DERR_INVALIDCALL;
    }
    // Upload the converted staging to the destination's Metal texture, mirroring
    // the GPU blit the same-format offscreen path emits.
    crate::texture::flush_converted_mips(dst_tex.inner_mut(), obj.inner());
    D3D_OK
}

/// Lazy texture upload: flush any pending dirty mips on the surfaces' parent textures.
///
/// A GPU write of a surface (`StretchRect` either way round, `ColorFill` of a
/// render target) reads and writes Metal textures alone, so every upload a
/// surface's texture still owes has to be scheduled ahead of it: an upload runs
/// at the head of the frame it is scheduled in, so one left for a later bind
/// lands on top of the write instead of under it.
///
/// Every caller sits below the gates that return `D3DERR_INVALIDCALL`, so a
/// rejected call schedules no upload for any endpoint.
fn flush_dirty_mips_for_gpu_write(
    obj: &Direct3DDevice9,
    surfaces: &[*mut crate::surface::Direct3DSurface9],
) {
    for &surf in surfaces {
        if surf.is_null() {
            continue;
        }
        // SAFETY: `surf` is non-null (checked above) and is a
        // caller-supplied `Direct3DSurface9*` from a `StretchRect` or
        // `ColorFill` thunk; the caller must pass live surface wrappers.
        let parent = unsafe { (*surf).parent_texture() };
        if parent.is_null() {
            continue;
        }
        // SAFETY: `parent` is non-null (checked above) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // the surface is alive.
        let tex = unsafe { &mut *parent };
        crate::texture::rehydrate_for_device(tex, obj.inner());
        crate::texture::flush_dirty_mips(tex.inner_mut(), obj.inner());
        tex.inner_mut().note_gpu_use();
    }
}

/// Whether the 1:1 blit, a copy of the texture's bytes, is the copy D3D9 defines.
///
/// Compares the *Metal* pixel formats, not the D3D codes: distinct D3D
/// formats can share a single Metal format (e.g. A8R8G8B8 + X8R8G8B8 are both
/// `Bgra8Unorm`, and on a device without the packed 16-bit formats A1R5G5B5
/// and A4R4G4B4 are too). A shared storage does not make every pair a byte
/// copy, though. A source without alpha into a destination with alpha would
/// hand the source's padding to the destination as alpha where D3D9 writes
/// alpha one (`exposes_padding_as_alpha`, judged on the mappings this device
/// created), so that pair converts; the other direction stays a byte copy,
/// since every reader of a destination without alpha ignores the alpha it
/// carries. A packed YUV endpoint lays its bytes out in an order no other
/// format shares (`reinterprets_packed_yuv`), so such a pair converts or is
/// refused like a pair of two storages.
fn copies_bytes(src: &StretchSurfaceInfo, dst: &StretchSurfaceInfo, expand_packed16: bool) -> bool {
    let src_map = crate::direct3d9::map_for_device(src.format, expand_packed16);
    let dst_map = crate::direct3d9::map_for_device(dst.format, expand_packed16);
    let exposes_padding = matches!(
        (&src_map, &dst_map),
        (Some(s), Some(d)) if mtld3d_core::stretch_rect::exposes_padding_as_alpha(s, d)
    );
    src_map.map(|m| m.metal_pixel_format()) == dst_map.map(|m| m.metal_pixel_format())
        && !exposes_padding
        && !mtld3d_core::stretch_rect::reinterprets_packed_yuv(src.format, dst.format)
}

/// Gate a colour `StretchRect` pair on a transport that copies or converts it.
fn check_stretch_rect_formats(
    src: &StretchSurfaceInfo,
    dst: &StretchSurfaceInfo,
    expand_packed16: bool,
) -> Result<(), i32> {
    // Device-aware mapping, so the comparison sees the Metal formats the
    // textures were actually created with on this device.
    let src_mtl = crate::direct3d9::map_for_device(src.format, expand_packed16)
        .map(|m| m.metal_pixel_format());
    let dst_mtl = crate::direct3d9::map_for_device(dst.format, expand_packed16)
        .map(|m| m.metal_pixel_format());
    // A pair `copies_bytes` accepts takes the 1:1 copy path. Any other pair
    // converts either via the render-quad path (sample src → write the dst
    // render target — needs a render-target destination) or, into an
    // offscreen-plain destination (which can't be rendered into), via the CPU
    // converter in `device_stretch_rect` (the offscreen→offscreen cross-format
    // path). An offscreen pair needs an available CPU codec before either
    // endpoint schedules an upload. Unmappable formats are always rejected.
    let convertible = src_mtl.is_some()
        && dst_mtl.is_some()
        && (copies_bytes(src, dst, expand_packed16)
            || dst.flags.contains(StretchSurfaceFlags::IS_RENDER_TARGET)
            || (dst
                .flags
                .contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT)
                && mtld3d_core::pixel_convert::can_convert(src.format, dst.format)));
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
/// rejection (an empty or inverted rect, or one with an edge outside its
/// surface), which callers map to `D3DERR_INVALIDCALL`. A rejection is
/// warned once per reason, naming both rects.
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
    let parsed = parse_rect(extracted_src, src_info.width, src_info.height).and_then(|src| {
        parse_rect(extracted_dst, dst_info.width, dst_info.height).map(|dst| (src, dst))
    });
    match parsed {
        Ok(regions) => Some(regions),
        Err(reason) => {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: reason.key(),
                "reject StretchRect: {} (src rect {extracted_src:?} on {}x{}, dst rect \
                 {extracted_dst:?} on {}x{}) → INVALIDCALL",
                reason.as_str(),
                src_info.width,
                src_info.height,
                dst_info.width,
                dst_info.height
            );
            None
        }
    }
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
        let (width, height) = (
            tex.inner().mip_width(lvl_idx),
            tex.inner().mip_height(lvl_idx),
        );
        let texture_size = TargetExtent::mip_level(
            tex.inner().render_scale(),
            (width, height),
            tex.inner().render_extent(),
            level,
        )
        .texture();
        return Some(StretchSurfaceInfo {
            kind: StretchKind::Texture(info),
            width,
            height,
            texture_size,
            // The texture's own scale, not one re-derived from the device: it
            // is what its Metal levels were created at, it holds for every
            // level rather than only the one that matches the back buffer, and
            // it does not move when the back buffer is resized under it.
            scale: tex.inner().render_scale(),
            format: tex.d3d_format(),
            mip_level: level,
            slice: s.cube_face(),
            pool: tex.d3d_pool(),
            flags,
            autogen_texture_id: (tex.inner().autogen_mipmap() && level == 0)
                .then(|| tex.texture_id()),
            // D3D9 has no multisampled texture: only a surface carries samples.
            msaa: MetalHandle::NULL,
            msaa_srgb: MetalHandle::NULL,
            sample_count: 1,
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
            texture_size: TargetExtent::whole(
                s.render_scale(),
                (s.standalone_width(), s.standalone_height()),
            )
            .texture(),
            // The surface's own scale, fixed when it was created: it is what
            // its Metal texture was allocated at, and it does not move when
            // the back buffer is resized under it.
            scale: s.render_scale(),
            format: s.standalone_format(),
            mip_level: 0,
            slice: None,
            pool: D3DPOOL_DEFAULT,
            flags: StretchSurfaceFlags::IS_RENDER_TARGET,
            autogen_texture_id: None,
            msaa: s.metal_msaa_handle(),
            msaa_srgb: s.metal_msaa_srgb_handle(),
            sample_count: s.multi_sample().sample_count,
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
            texture_size: TargetExtent::whole(
                s.render_scale(),
                (s.standalone_width(), s.standalone_height()),
            )
            .texture(),
            // See the colour branch above: the surface answers with what its
            // own depth texture was created at.
            scale: s.render_scale(),
            format: s.standalone_format(),
            mip_level: 0,
            slice: None,
            pool: D3DPOOL_DEFAULT,
            flags: StretchSurfaceFlags::IS_DEPTH_STENCIL,
            autogen_texture_id: None,
            // A depth surface carries its samples in its own texture: there is
            // no single-sample companion to resolve into, because D3D9 offers
            // no way to sample one. `StretchRect` reads the sample count to
            // tell a plain depth copy from the resolve.
            msaa: MetalHandle::NULL,
            msaa_srgb: MetalHandle::NULL,
            sample_count: s.multi_sample().sample_count,
        });
    }
    None
}

/// Resolve a `ColorFill` rect against the destination mip extent.
///
/// `rect` is the caller's `D3DRECT`; `None` fills the whole mip. Edges are
/// clipped to the surface, so a rect that hangs over an edge fills the part
/// that lands on it, and one that misses entirely returns `None` for the
/// caller to treat as a no-op.
fn color_fill_region(rect: Option<mtld3d_types::D3DRECT>, extent: (u32, u32)) -> Option<DirtyRect> {
    let Some(r) = rect else {
        return DirtyRect::full(extent.0, extent.1).clamp(extent.0, extent.1);
    };
    let x = r.x1.max(0).cast_unsigned();
    let y = r.y1.max(0).cast_unsigned();
    let right = r.x2.max(0).cast_unsigned();
    let bottom = r.y2.max(0).cast_unsigned();
    DirtyRect {
        x,
        y,
        w: right.saturating_sub(x),
        h: bottom.saturating_sub(y),
    }
    .clamp(extent.0, extent.1)
}

/// True when a `ColorFill` region lands on the destination's block grid.
///
/// A block-compressed fill that splits a block is `INVALIDCALL`; an edge that
/// ends on the surface boundary is exempt, because the last block there is
/// partial anyway. Uncompressed formats have a 1x1 grid, so this is inert.
fn color_fill_block_aligned(region: DirtyRect, extent: (u32, u32), block: (u32, u32)) -> bool {
    let (bw, bh) = (block.0.max(1), block.1.max(1));
    if bw == 1 && bh == 1 {
        return true;
    }
    let right = region.x + region.w;
    let bottom = region.y + region.h;
    region.x.is_multiple_of(bw)
        && region.y.is_multiple_of(bh)
        && (right.is_multiple_of(bw) || right == extent.0)
        && (bottom.is_multiple_of(bh) || bottom == extent.1)
}

/// `ColorFill` a render target: paint the fill on the GPU.
///
/// The destination is a colour attachment on this device, so the fill is a
/// one-off render pass through the clear machinery and the API thread writes
/// no pixels at all. A fill into level 0 of an `D3DUSAGE_AUTOGENMIPMAP`
/// texture carries the mip-chain regeneration with it. `info` is consumed
/// because the destination kind travels into the encoder operation, which
/// cannot reach the surface.
fn color_fill_render_target(
    dev: &mut DeviceInner,
    info: StretchSurfaceInfo,
    region: DirtyRect,
    color: u32,
) -> i32 {
    // Device-aware: the clear-quad pipeline's colour format must match the
    // attachment as it was created here (BGRA8 for an expanded 16-bit target).
    let Some(format) = crate::direct3d9::map_for_device(info.format, dev.config().expand_packed16)
        .map(|m| m.metal_pixel_format())
    else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill: render-target format {} unmapped → INVALIDCALL", info.format);
        return D3DERR_INVALIDCALL;
    };
    let [r, g, b, a] = convert::d3dcolor_to_rgba_f32(color);
    let fill = ColorFillTarget {
        // Resolved in the operation below: a texture destination only creates
        // its `MTLTexture` on the encoder thread.
        texture: MetalHandle::NULL,
        logical_size: (info.width, info.height),
        texture_size: info.texture_size,
        format,
        scale: info.scale,
        subresource: (info.slice.unwrap_or(0), info.mip_level),
        rect: (region.x, region.y, region.w, region.h),
        rgba: (r.to_bits(), g.to_bits(), b.to_bits(), a.to_bits()),
        msaa: info.msaa,
        msaa_srgb: info.msaa_srgb,
        sample_count: info.sample_count,
        regenerate_mipmaps: info.autogen_texture_id.is_some(),
    };
    let kind = info.kind;
    dev.push_control(crate::device::ColorFillOp { kind, fill });
    D3D_OK
}

/// `ColorFill` a lockable `D3DPOOL_DEFAULT` offscreen-plain surface.
///
/// Its internal Metal texture is created shader-read-only, so it cannot be a
/// colour attachment, and its read-back is a `LockRect` straight into CPU
/// staging that no path refreshes from the GPU. The fill therefore lands in
/// the staging on this thread and rides the ordinary upload to the texture,
/// exactly like the write half of a `LockRect` / `UnlockRect` pair.
fn color_fill_offscreen_plain(
    dev: &mut DeviceInner,
    parent: *mut Direct3DTexture9,
    info: &StretchSurfaceInfo,
    region: DirtyRect,
    color: u32,
) -> i32 {
    // Encode the fill colour into the destination format. An unmapped format
    // still succeeds but leaves the surface unfilled (the colour check, not
    // the ColorFill return, is what would fail for those).
    let Some(pixel) = convert::d3dcolor_fill_pixel_bytes(color, info.format) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill: no fill encoding for format {} → surface left unfilled", info.format);
        return D3D_OK;
    };
    if pixel.is_empty() {
        return D3D_OK;
    }
    // SAFETY: `parent` is non-null (the caller classified this surface as
    // texture-backed) and its refcount keeps it alive while the surface is
    // alive; access is exclusive: D3D9 objects are single-threaded, or
    // serialised by the device `ApiLock` under `D3DCREATE_MULTITHREADED`.
    let ti = unsafe { &mut *parent }.inner_mut();
    let level = info.mip_level;
    if !ti.fill_staging_region(
        level as usize,
        region.x,
        region.y,
        region.w,
        region.h,
        &pixel,
    ) {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill: staging fill of level {level} fell outside the mip → surface left unfilled");
        return D3D_OK;
    }
    crate::texture::schedule_upload(ti, dev, level, region);
    D3D_OK
}

extern "system" fn device_color_fill(
    this: *mut c_void,
    surface: *mut c_void,
    rect: *const c_void,
    color: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    if surface.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if obj.inner().frame_dump.active {
        obj.inner().frame_dump_event(&format!(
            "ColorFill({}, {color:#010x})",
            frame_dump::surface_label(surface)
        ));
    }
    // SAFETY: vtable in-param; `rect` is *const D3DRECT per the D3D9 ABI.
    let rect = unsafe { ValueIn::<mtld3d_types::D3DRECT>::read_opt(rect) };
    let surf = surface.cast::<Direct3DSurface9>();
    let Some(info) = resolve_stretch_surface(surf) else {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill on a surface with no DEFAULT-pool backing → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    // D3D9: ColorFill takes a DEFAULT-pool render target (a texture level, a
    // `CreateRenderTarget` surface or the back buffer) or a DEFAULT-pool
    // offscreen-plain surface. Managed / sysmem / scratch, a depth-stencil
    // surface and an ordinary DEFAULT texture level are all INVALIDCALL.
    if info.pool != D3DPOOL_DEFAULT
        || !info.flags.intersects(
            StretchSurfaceFlags::IS_RENDER_TARGET | StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT,
        )
    {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill: surface is not a DEFAULT render target or offscreen-plain → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // A planar YUV plain has no fill encoding: a D3DCOLOR names no Y, U, V
    // triple the layer could defend, the same position the packed YUV formats
    // are in. The call succeeds and the surface keeps its bytes. Answered
    // here because the format lookup below has no entry for the format and
    // would warn on every call.
    if mtld3d_core::stretch_rect::is_planar_yuv(info.format) {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "ColorFill: no fill encoding for planar YUV format {} → surface left unfilled",
            mtld3d_core::format::format_name(info.format));
        return D3D_OK;
    }
    let extent = (info.width, info.height);
    let Some(region) = color_fill_region(rect, extent) else {
        return D3D_OK;
    };
    if let Some(fmt) = map_d3d_format(info.format)
        && !color_fill_block_aligned(region, extent, (fmt.block_width(), fmt.block_height()))
    {
        return D3DERR_INVALIDCALL;
    }
    if info.flags.contains(StretchSurfaceFlags::IS_RENDER_TARGET) {
        // The fill runs as a render pass over the destination's Metal texture
        // and never touches its CPU staging, so a texture-backed destination
        // owes the same two things a `StretchRect` destination does: the
        // uploads its levels still have pending go first, past every gate that
        // can reject the call, and the next map reads the fill back rather
        // than the staging it left.
        if matches!(info.kind, StretchKind::Texture(_)) {
            flush_dirty_mips_for_gpu_write(&obj, &[surf]);
            claim_dst_subresource_for_gpu(surf, &info);
        }
        return color_fill_render_target(obj.inner(), info, region, color);
    }
    // SAFETY: an offscreen-plain surface is texture-backed, so
    // `resolve_stretch_surface` classified it off a non-null parent.
    let parent = unsafe { &*surf }.parent_texture();
    color_fill_offscreen_plain(obj.inner(), parent, &info, region, color)
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if surface.is_null() || width == 0 || height == 0 {
        null_out(surface);
        return D3DERR_INVALIDCALL;
    }
    // A plain device refuses a non-null pSharedHandle with E_NOTIMPL; an
    // extended one reads it as user memory for a system-memory surface and as
    // a shared resource, which it does not create, in the default pool.
    let user_memory = match shared_handle_gate(
        this,
        &mtld3d_core::extended::CreateKind::OffscreenPlain,
        pool,
        shared_handle,
        "CreateOffscreenPlainSurface",
    ) {
        Ok(memory) => memory,
        Err(hr) => {
            null_out(surface);
            return hr;
        }
    };
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
        let hr = create_texture_path(&TextureCreateArgs {
            this,
            width,
            height,
            levels: 1,
            usage: 0,
            format,
            pool: D3DPOOL_DEFAULT,
            texture: &raw mut tex_out,
            shared_handle: core::ptr::null_mut(),
            offscreen_plain: true,
        });
        if hr != D3D_OK || tex_out.is_null() {
            warn!(target: LOG_TARGET,
                "reject CreateOffscreenPlainSurface({width}x{height}, format={format}, DEFAULT) → INVALIDCALL (internal texture create failed)");
            null_out(surface);
            return D3DERR_INVALIDCALL;
        }
        let tex_ptr = tex_out.cast::<crate::texture::Direct3DTexture9>();
        let surf = Direct3DSurface9::new_owned_texture_backed(device_inner, tex_ptr);
        // The internal texture (created just above) forwards the device
        // reference, so this owned surface is NOT registered (no double-count).
        // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
        unsafe { OutPtr::write_opt(surface, Box::into_raw(Box::new(surf)).cast::<c_void>()) };
        return D3D_OK;
    }
    // A planar YUV surface holds its chroma planes after its luma rows, so it
    // is half again as large as the pitch-times-height backing the CPU pools
    // below allocate. Only the DEFAULT path above sizes it from the planar
    // layout; a CPU-pool surface would let the application write its chroma
    // past the end of the allocation.
    if mtld3d_core::stretch_rect::is_planar_yuv(format) {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "reject CreateOffscreenPlainSurface(format={}, pool={pool}) → INVALIDCALL (planar \
             YUV surfaces exist in D3DPOOL_DEFAULT only)",
            mtld3d_core::format::format_name(format));
        null_out(surface);
        return D3DERR_INVALIDCALL;
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
        (linear_row_pitch(width, bpp) as usize).saturating_mul(height as usize)
    };
    // The surface's bytes are the one allocation of the create that may fail.
    let Some(mut backing) = PageBox::try_new_uninit(bytes) else {
        return refuse_unallocatable_staging("CreateOffscreenPlainSurface", surface);
    };
    // An extended device's system-memory surface starts with the
    // application's packed pixels, copied once at the lock pitch.
    if let Some(memory) = user_memory {
        let rows = mtld3d_core::extended::PackedRows::of_level(
            width,
            height,
            bpp,
            (fmt.block_width(), fmt.block_height(), fmt.block_bytes()),
        );
        let pitch = if bpp == 0 {
            rows.row_bytes
        } else {
            linear_row_pitch(width, bpp) as usize
        };
        copy_user_memory(memory, &rows, backing.as_mut_slice(), pitch);
    }
    let surf =
        Direct3DSurface9::new_system_memory(obj.inner_ptr(), width, height, format, pool, backing);
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if index >= D3D_MAX_SIMULTANEOUS_RENDERTARGETS {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject SetRenderTarget(index={index}) → INVALIDCALL (four simultaneous render targets)"
        );
        return D3DERR_INVALIDCALL;
    }
    let slot = usize::try_from(index).expect("index < 4 fits usize");

    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "SetRenderTarget({index}, {})",
            frame_dump::surface_label(surface)
        ));
    }

    if surface.is_null() {
        if slot == 0 {
            // D3D9 spec: RT0 must remain non-null.
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "reject SetRenderTarget(index=0, null) → INVALIDCALL"
            );
            return D3DERR_INVALIDCALL;
        }
        // Render targets 1..3 may be unbound; draws then leave the slot alone.
        dev.unbind_extra_render_target(slot);
        return D3D_OK;
    }

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
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject SetRenderTarget: surface owned by a different device → INVALIDCALL"
        );
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
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "reject SetRenderTarget: GetDesc failed");
        return D3DERR_INVALIDCALL;
    }
    // The destination must be render-target-capable.
    // GetDesc reports the surface's true usage — the parent texture's
    // D3DUSAGE_RENDERTARGET for a GetSurfaceLevel surface, or RENDERTARGET for
    // the implicit backbuffer / a CreateRenderTarget surface — so every
    // legitimate render-target bind passes.
    if desc.usage & D3DUSAGE_RENDERTARGET == 0 {
        mtld3d_shared::log_once_warn!(
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
    let surface_ref = unsafe { &*surf };
    let parent = surface_ref.parent_texture();
    // A texture level bound as a target takes any CPU write it still owes
    // ahead of the passes about to draw into it.
    if !parent.is_null() {
        // SAFETY: `parent` is non-null (checked) and points to a live
        // `Direct3DTexture9` whose refcount keeps it alive while the surface
        // is; it is a distinct allocation from the device.
        let texture = unsafe { &mut *parent };
        crate::texture::publish_render_target_write(texture.inner_mut(), dev);
    }
    let standalone_color = surface_ref.metal_color_handle();
    // What the bound resource is rasterized at, asked of the resource itself:
    // the surface for a standalone or implicit target, the parent texture for
    // a texture level (whose Metal levels all carry the texture's scale, not
    // only the one that happens to match the back buffer).
    let scale = if parent.is_null() {
        surface_ref.render_scale()
    } else {
        // SAFETY: `parent` is non-null (checked) and points to a live
        // `Direct3DTexture9` whose refcount keeps it alive while the surface
        // is bound.
        unsafe { &*parent }.inner().render_scale()
    };
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
                srgb: surface_ref.metal_color_srgb_handle(),
                msaa: surface_ref.metal_msaa_handle(),
                msaa_srgb: surface_ref.metal_msaa_srgb_handle(),
                sample_count: surface_ref.multi_sample().sample_count,
                format: fmt,
                has_alpha,
                width: desc.width,
                height: desc.height,
            }
        } else {
            // The implicit backbuffer render target.
            RtBinding::Backbuffer {
                handle: dev.backbuffer_handle,
                msaa: dev.backbuffer_msaa_handle,
                msaa_srgb: dev.backbuffer_msaa_srgb_handle,
                sample_count: dev.backbuffer_sample_count,
                width: dev.backbuffer_width,
                height: dev.backbuffer_height,
            }
        }
    } else {
        // SAFETY: `parent` is non-null (else branch) and points to a
        // live `Direct3DTexture9` whose refcount keeps it alive while
        // the surface is bound.
        let parent_tex = unsafe { &*parent };
        let mut texture_info = parent_tex.inner().texture_info();
        texture_info
            .usage_flags
            .insert(mtld3d_shared::mtl::TextureUsage::RENDER_TARGET);
        RtBinding::Texture {
            info: texture_info,
            // Unknown formats default to alpha-bearing (the pre-clamp
            // behaviour); a real X8R8G8B8 RT texture reports `false`.
            has_alpha: map_d3d_format(desc.format)
                .as_ref()
                .is_none_or(FormatMapping::has_alpha),
            width: desc.width,
            height: desc.height,
            // A 2D texture uses slice zero. Cube-face surfaces carry their
            // face as the Metal array slice while sharing the parent texture.
            slice: surface_ref.cube_face().unwrap_or(0),
            level: surface_ref.mip_level(),
        }
    };

    // Autogen render targets: if the slot is changing away from an
    // `D3DUSAGE_AUTOGENMIPMAP` texture, regenerate its mip chain now (ordered
    // after the render/clear that just modified its level 0). Track the new
    // target's autogen id for the next change.
    let new_autogen = if parent.is_null() {
        None
    } else {
        // SAFETY: `parent` is the live parent texture validated above.
        let pt = unsafe { &*parent };
        pt.inner().autogen_mipmap().then(|| pt.texture_id())
    };
    if let Some(old_id) = dev.cur_autogen_rt_ids[slot].take()
        && Some(old_id) != new_autogen
    {
        dev.push_control(crate::device::GenerateMipmapsOrderedOp { old_id });
    }
    dev.cur_autogen_rt_ids[slot] = new_autogen;

    dev.bound_rt_mut()
        .replace_render_target(slot, surf, desc.width, desc.height);

    // Remember the applied binding so a mid-frame readback flush can re-assert
    // it into the fresh frame (the encoder's pass state resets to the
    // backbuffer default each frame; a D3D9 RT binding outlives an internal
    // flush — see `last_color_rt_binding`).
    if slot != 0 {
        dev.last_extra_rt_bindings[slot - 1] = Some((info.clone(), scale));
        dev.push_color_rt_binding_op(slot, info, scale);
        // Only render target 0 owns the viewport and scissor; the pipeline
        // reads the extra targets' formats on the encoder thread, so no
        // snapshot section goes stale here.
        return D3D_OK;
    }
    dev.last_color_rt_binding = Some((info.clone(), scale));
    dev.push_color_rt_binding_op(0, info, scale);

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
    dev.set_scissor_rect(mtld3d_core::render_state::full_target_scissor(
        desc.width,
        desc.height,
    ));
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if surface.is_null() || index >= D3D_MAX_SIMULTANEOUS_RENDERTARGETS {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject GetRenderTarget(index={index}) → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let slot = usize::try_from(index).expect("index < 4 fits usize");
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
    // current backbuffer. Render targets 1..3 have no default: an unbound slot
    // reports `D3DERR_NOTFOUND` with a null out-pointer.
    let bound = inner.bound_rt().render_target(slot);
    let surf = if bound.is_null() {
        if slot != 0 {
            // SAFETY: `surface` is the caller's out-pointer per the D3D9 ABI.
            unsafe { *surface = core::ptr::null_mut() };
            return crate::D3DERR_NOTFOUND;
        }
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

extern "system" fn device_set_depth_stencil_surface(
    this: *mut c_void,
    surface: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "SetDepthStencilSurface({})",
            frame_dump::surface_label(surface)
        ));
    }
    let surf = surface.cast::<Direct3DSurface9>();
    // A non-NULL depth-stencil surface must report D3DUSAGE_DEPTHSTENCIL; NULL
    // unbinds the depth buffer. GetDesc reports the true usage (the parent
    // depth texture's DEPTHSTENCIL for a sampleable shadow map, the implicit
    // DS, or a CreateDepthStencilSurface surface), so WoW's shadow-map and
    // implicit-DS binds all pass. Validate before mutating any device state.
    // The extent comes back with it, for the standalone-surface bind arm
    // below: that one has no parent `TextureInfo` to read a size out of.
    let reported_size = if surf.is_null() {
        (0, 0)
    } else {
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
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "reject SetDepthStencilSurface: surface is not a depth-stencil (usage={:#x}) → INVALIDCALL",
                desc.usage
            );
            return D3DERR_INVALIDCALL;
        }
        (desc.width, desc.height)
    };
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
        // The depth.aliasSameSize carry: a different texture bound at the
        // same dimensions inherits the previous one's contents, matching
        // the shared physical depth allocation of D3D9-era drivers that
        // engines of that era rely on (bind one handle, sample the other).
        if dev.config().depth_alias_same_size {
            let mip_w = (w >> mip).max(1);
            let mip_h = (h >> mip).max(1);
            if let Some((prev_id, pw, ph)) = dev.last_sized_depth
                && prev_id != info.texture_id
                && (pw, ph) == (mip_w, mip_h)
            {
                let cur_id = info.texture_id;
                if dev.frame_dump.active {
                    dev.frame_dump_event(&format!(
                        "depth-alias carry {prev_id:?} → {cur_id:?} {mip_w}x{mip_h}"
                    ));
                }
                dev.push_control(crate::device::CarryDepthOp {
                    prev_id,
                    cur_id,
                    mip_w,
                    mip_h,
                });
            }
            dev.last_sized_depth = Some((info.texture_id, mip_w, mip_h));
        }
        // The parent texture carries the scale its Metal texture was created
        // at, as it does for a colour target.
        // SAFETY: `depth_texture_info` answered, so `surf` is texture-backed
        // and its parent texture is live while the surface is.
        let parent = unsafe { (*surf).parent_texture() };
        // SAFETY: `parent` is non-null and live (see above).
        let scale = unsafe { &*parent }.inner().render_scale();
        DepthBinding::Lazy(info, mip, scale)
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
        // The Metal texture behind a standalone depth surface was created at
        // the scale the surface carries, so ask the surface rather than assume
        // the extent D3D9 reports. What the pass machine measures a `Clear`'s
        // viewport against, and what a depth snapshot copy has to match, is
        // the texture's real extent.
        // SAFETY: `surf` is non-null (else-if branch) and points to a live
        // surface.
        let scale = unsafe { (*surf).render_scale() };
        DepthBinding::Eager(
            h,
            (
                scale.dimension(reported_size.0),
                scale.dimension(reported_size.1),
            ),
            scale,
        )
    };
    // `is_sampleable` distinguishes a sampleable shadow map
    // (`CreateTexture(D24X8, DEPTHSTENCIL)`) from a depth surface no
    // `SetTexture` can reach (`CreateDepthStencilSurface`, the implicit
    // depth-stencil). The pass machine uses this to keep `Store`
    // unconditionally on sampleable shadow maps (Rule B short-circuit),
    // since they may be sampled in a future frame even if no sample lands
    // this frame, typical of cascade-3 in CSM rotations. The surface
    // answers for itself rather than the bind path answering for it: the
    // flag has to be the same for every bind of one Metal texture, and the
    // pass state asserts as much.
    let is_sampleable = !surf.is_null() && {
        // SAFETY: `surf` is non-null (checked) and a live `Direct3DSurface9`
        // (already deref'd above to build the binding).
        unsafe { (*surf).depth_is_sampleable() }
    };
    // Whether the bound depth attachment is a combined depth+stencil format
    // (D24S8 etc. → the combined Metal texture), so the clear-quad / draw
    // pipelines declare matching depth/stencil attachment formats. Mirrors
    // the snapshot's `depth_format_has_stencil(depth_attachment_format())`,
    // which resolves texture-backed (sampleable) depth surfaces to the parent
    // texture's format — a D24S8 shadow map binds a combined-stencil Metal
    // texture just like a standalone D24S8 surface does.
    let depth_has_stencil = if surf.is_null() {
        false
    } else {
        // SAFETY: `surf` is non-null (checked) and a live `Direct3DSurface9`
        // (already deref'd above to build the binding).
        depth_format_has_stencil(unsafe { (*surf).depth_attachment_format() })
    };
    // Remember the binding so a mid-frame readback flush can re-assert it (the
    // encoder's pass state re-attaches the implicit auto-depth each frame; a
    // D3D9 depth bind — including an explicit unbind — outlives an internal
    // flush, see `last_depth_binding`).
    // A texture-backed depth surface is never multisampled: D3D9 has no
    // multisampled texture, only a multisampled surface.
    let depth_sample_count = if surf.is_null() {
        1
    } else {
        // SAFETY: `surf` is non-null (checked) and a live `Direct3DSurface9`.
        unsafe { (*surf).multi_sample() }.sample_count
    };
    dev.last_depth_binding = Some((
        binding.clone(),
        is_sampleable,
        depth_has_stencil,
        depth_sample_count,
    ));
    dev.push_depth_binding_op(
        binding,
        is_sampleable,
        depth_has_stencil,
        depth_sample_count,
    );
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::RtDs);
    if surface.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // `GetDepthStencilSurface` reflects the currently *bound* depth-stencil, not
    // merely the device's auto depth-stencil. Three shapes, in order:
    // - a surface bound by `SetDepthStencilSurface` is handed back as itself, so
    //   an application comparing the pointer against its own surface matches and
    //   a `saved = Get; Set(other); ...; Set(saved)` sequence rebinds what was
    //   actually bound rather than the auto depth;
    // - an explicit `SetDepthStencilSurface(NULL)` unbinds depth, so report
    //   "none bound" (NOTFOUND) even though the auto depth texture still exists;
    // - otherwise the auto depth-stencil is in effect, and stands for the
    //   device-owned implicit depth-stencil surface (cached, refcount-0,
    //   container = the device). A single object across calls, depth-backed so
    //   the save/restore above restores the real Metal depth handle (resolved
    //   live). Null when the device has no auto depth-stencil.
    let dev = obj.inner();
    let bound = dev.bound_rt().depth_stencil();
    let surf = if bound.is_null() {
        if dev.flags.contains(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND) {
            core::ptr::null_mut()
        } else {
            dev.get_or_create_implicit_depth_stencil()
        }
    } else {
        bound
    };
    if surf.is_null() {
        // SAFETY: `surface` is non-null (checked above) and per the D3D9 ABI
        // points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *surface = core::ptr::null_mut() };
        return crate::D3DERR_NOTFOUND;
    }
    // SAFETY: `surf` is non-null: either the live bound depth-stencil (the
    // bind refcount keeps it alive) or the live cached implicit DS surface,
    // whose AddRef thunk forwards to the device on the 0→1 transition.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: calling the surface's AddRef thunk; D3D9 mandates AddRef on return.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: vtable out-param; `surface` is *mut *mut c_void per IDirect3DDevice9 ABI.
    unsafe { *surface = surf.cast::<c_void>() };
    D3D_OK
}

extern "system" fn device_begin_scene(this: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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

/// The `INVALIDCALL` `Clear` answers for a depth or stencil clear with no depth-stencil surface.
///
/// Out of line so `Clear` does not build the log arguments on its hot path.
#[cold]
#[inline(never)]
fn reject_depth_clear_without_surface(flags: u32) -> i32 {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "Clear of depth or stencil (flags={flags:#x}) with no depth-stencil surface bound → \
         INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

extern "system" fn device_clear(
    this: *mut c_void,
    count: u32,
    rects: *const c_void,
    flags: u32,
    color: u32,
    z: f32,
    stencil: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if dev.frame_dump.active {
        let (rt, ds) = dev.frame_dump_target_labels();
        dev.frame_dump_event(&format!(
            "clear flags={flags:#x} color={color:#010x} z={z} stencil={stencil} rects={count} \
             rt={rt} ds={ds}"
        ));
    }

    // Clearing depth or stencil with no depth-stencil attachment bound is
    // invalid: a prior `SetDepthStencilSurface(NULL)` leaves no surface to
    // clear.
    if flags & (D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL) != 0 && !dev.depth_stencil_bound() {
        return reject_depth_clear_without_surface(flags);
    }

    // Clear also honours D3DRS_SCISSORTESTENABLE: when on, every cleared
    // region is additionally clipped to the device scissor rect, and an empty
    // scissor clears nothing. Resolved on the API thread; the encoder then
    // clips ∩ viewport.
    let scissor_on = dev.render_state(D3DRS_SCISSORTESTENABLE as usize) != 0;
    let scissor = mtld3d_core::render_state::scissor_region(dev.scissor_rect());

    // D3D9 Clear's pRects/Count semantics, shared by every plane:
    //  - pRects == NULL  → clear the whole target (Count ignored). With the
    //    scissor on, whole target ∩ scissor == the scissor rect.
    //  - pRects != NULL, Count == 0 → clear NOTHING (a no-op).
    //  - pRects != NULL, Count >  0 → clear each rect (∩ scissor if on).
    // `None` is the whole target; `Some` is an explicit list, already clipped
    // to the scissor and possibly empty.
    let regions: Option<Vec<(i32, i32, i32, i32)>> = if rects.is_null() {
        scissor_on.then(|| {
            let (x1, y1, x2, y2) = scissor;
            if x2 > x1 && y2 > y1 {
                vec![scissor]
            } else {
                Vec::new()
            }
        })
    } else {
        let mut list = if count > 0 {
            clear_target_rects(count, rects)
        } else {
            Vec::new()
        };
        if scissor_on {
            list.retain_mut(|r| {
                intersect_d3d_rects(*r, scissor).is_some_and(|clipped| {
                    *r = clipped;
                    true
                })
            });
        }
        Some(list)
    };

    if flags & D3DCLEAR_TARGET != 0 {
        // D3DCOLOR is ARGB; unpack to normalized float bits so the encoder
        // can fold them into a Metal `MTLLoadAction::Clear` at pass-begin.
        // D3DRS_SRGBWRITEENABLE applies to the clear colour exactly as it
        // applies to a draw's output, but which of the two encodes it is a
        // property of the attachment the encoder will bind: an sRGB view
        // converts the clear value itself, and only a target without one
        // needs the curve applied to the value on the way in. The bit rides
        // along and the encoder decides.
        let srgb_write = dev.render_state(D3DRS_SRGBWRITEENABLE as usize) != 0;
        let rgba = mtld3d_core::convert::d3dcolor_to_rgba_f32(color);
        let r_bits = f32::to_bits(rgba[0]);
        let g_bits = f32::to_bits(rgba[1]);
        let b_bits = f32::to_bits(rgba[2]);
        let a_bits = f32::to_bits(rgba[3]);

        // `pRects == NULL` still means "the whole target ∩ the viewport", so
        // the encoder decides per attachment whether the viewport covers it:
        // covered folds to a whole-attachment loadAction, a strict sub-region
        // paints a scissored clear-quad. Independent of which other planes
        // this Clear names, exactly as the depth side below is.
        match &regions {
            None => dev.push_control(crate::device::ClearColorOp {
                r_bits,
                g_bits,
                b_bits,
                a_bits,
                srgb_write,
            }),
            Some(list) if list.is_empty() => {}
            Some(list) => {
                let rects = list.clone();
                dev.push_control(crate::device::ClearColorRectsOp {
                    r_bits,
                    g_bits,
                    b_bits,
                    a_bits,
                    srgb_write,
                    rects,
                });
            }
        }
    }

    let z_bits = f32::to_bits(z);
    // D3D9 passes the stencil clear as a DWORD; the attachment is 8-bit.
    let stencil = stencil & mtld3d_core::depth_stencil_state::STENCIL_MASK_BITS;
    let depth_plane = (flags & D3DCLEAR_ZBUFFER != 0).then_some(z_bits);
    let stencil_plane =
        (flags & D3DCLEAR_STENCIL != 0 && dev.depth_stencil_has_stencil()).then_some(stencil);
    match (regions, depth_plane, stencil_plane) {
        (_, None, None) => {}
        // Explicit regions (or the scissor) bound the depth/stencil clear
        // exactly as they bound the colour clear: one scissored quad per rect.
        (Some(list), depth, stencil) => {
            if !list.is_empty() {
                dev.push_control(crate::device::ClearDepthStencilRectsOp {
                    depth,
                    stencil,
                    list,
                });
            }
        }
        // The whole target, which D3D9 still bounds by the viewport. Both
        // planes go in one op so a covering clear of both paints one quad
        // (or folds into one pair of load actions) rather than two:
        // shadow-volume renderers clear depth and stencil together between
        // lights.
        (None, depth, stencil) => {
            dev.push_control(crate::device::ClearDepthStencilOp { depth, stencil });
        }
    }

    0 // S_OK
}

extern "system" fn device_set_transform(
    this: *mut c_void,
    state: u32,
    matrix: *const c_void,
) -> i32 {
    let _api = device_api_lock(this);
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
    // A matrix feeds the FF VS constants only, never its key or row count.
    let mask = dev.ff_vs_write_mask(false) | dev.transform_write_side_dirty(state);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_transform(this: *mut c_void, state: u32, matrix: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    // A matrix feeds the FF VS constants only, never its key or row count.
    let mask = dev.ff_vs_write_mask(false) | dev.transform_write_side_dirty(state);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_set_viewport(this: *mut c_void, viewport: *const c_void) -> i32 {
    let _api = device_api_lock(this);
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
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "SetViewport {},{}+{}x{} z=[{},{}]",
            v.x, v.y, v.width, v.height, v.min_z, v.max_z
        ));
    }
    dev.set_viewport(v);
    D3D_OK
}

extern "system" fn device_get_viewport(this: *mut c_void, viewport: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    // The material feeds the constant sections alone, never the FF VS key.
    dev.ff_state_mut().set_material(&m);
    let mask = dev.ff_aware_mask(SnapshotDirty::VS_CONST);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_material(this: *mut c_void, material: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    let inputs = dev.ff_state().vs_source_light_inputs();
    dev.ff_state_mut().set_light_at(index, &l);
    let mask = dev.ff_vs_write_mask(dev.ff_state().vs_source_light_inputs() != inputs);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_light(this: *mut c_void, index: u32, light: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::FfFixed);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let on = enable != 0;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record_light_enable(index, on);
        return D3D_OK;
    }
    let inputs = dev.ff_state().vs_source_light_inputs();
    dev.ff_state_mut().set_light_enabled_at(index, on);
    let mask = dev.ff_vs_write_mask(dev.ff_state().vs_source_light_inputs() != inputs);
    dev.mark_snapshot_dirty(mask);
    0 // S_OK
}

extern "system" fn device_get_light_enable(this: *mut c_void, index: u32, enable: *mut i32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::ClipPlane {
            index,
            plane: coeffs,
        });
        return D3D_OK;
    }
    dev.set_clip_plane(index, coeffs);
    0 // S_OK
}

extern "system" fn device_get_clip_plane(this: *mut c_void, index: u32, plane: *mut f32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::RenderState);
    // A state no D3D9 render state names (1 to 6, or past `D3DRS_BLENDOPALPHA`)
    // is accepted and ignored.
    if (1..mtld3d_types::D3DRS_ZENABLE).contains(&state) || (state as usize) >= RENDER_STATE_COUNT {
        return ignore_unnamed_render_state(state);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // The 'RESZ' hack: this magic written to POINTSIZE asks the driver to
    // resolve the bound depth-stencil into the depth texture bound at
    // stage 0. Advertised via the RESZ fourcc in CheckDeviceFormat;
    // engines on the matching vendor path use it as their only way to
    // hand scene depth to a second depth consumer.
    if state == D3DRS_POINTSIZE && value == 0x7fa0_5000 {
        dev.resolve_resz_depth();
    }
    if let Some(rec) = dev.recording_state_block_mut() {
        if state == D3DRS_POINTSIZE {
            rec.record(StateOp::PointSize {
                raw: value,
                numeric: mtld3d_core::multisample::numeric_point_size(value),
                coverage: mtld3d_core::multisample::a2m_control(value),
            });
        } else {
            rec.record(StateOp::RenderState { state, value });
        }
        return D3D_OK;
    }
    // Skip the dirty mark only when raw and derived state are unchanged.
    let changed = dev.set_render_state(state as usize, value);
    if changed {
        let mut mask = dev.ff_aware_mask(rs_dirty_mask(state));
        // The user clip plane count is a key input of the programmable VS as
        // well, which `ff_aware_mask` would otherwise strip.
        if matches!(state, D3DRS_CLIPPLANEENABLE | D3DRS_CLIPPING) {
            mask |= SnapshotDirty::VS_SOURCE;
        }
        dev.mark_snapshot_dirty(mask);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetRenderState, !changed);
    0 // S_OK
}

extern "system" fn device_get_render_state(this: *mut c_void, state: u32, value: *mut u32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::RenderState);
    if value.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // States 1 to 6 and past 255 are refused; state 0 and the unnamed states
    // up to 255 read as zero.
    if (1..mtld3d_types::D3DRS_ZENABLE).contains(&state) || state > 255 {
        return reject_unreadable_render_state(state);
    }
    if state == 0 || (state as usize) >= RENDER_STATE_COUNT {
        warn_unnamed_render_state_read(state);
        // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
        // points to a writable `u32` slot owned by the caller.
        unsafe { *value = 0 };
        return D3D_OK;
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

/// The `D3D_OK` `SetRenderState` answers for a state no D3D9 render state names, which it ignores.
///
/// Out of line, like the other index rejects below, so the state and sampler
/// setters do not build the log arguments on their hot path.
#[cold]
#[inline(never)]
fn ignore_unnamed_render_state(state: u32) -> i32 {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "SetRenderState({state}): no render state has that index → ignored, D3D_OK"
    );
    D3D_OK
}

/// The `INVALIDCALL` `GetRenderState` answers for states 1 to 6 and past 255.
#[cold]
#[inline(never)]
fn reject_unreadable_render_state(state: u32) -> i32 {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "GetRenderState({state}): outside the readable render states → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

/// Warn once that `GetRenderState` read zero for state 0 or an unnamed state up to 255.
#[cold]
#[inline(never)]
fn warn_unnamed_render_state_read(state: u32) {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "GetRenderState({state}): no render state has that index → 0, D3D_OK"
    );
}

extern "system" fn device_create_state_block(
    this: *mut c_void,
    type_: u32,
    sb: *mut *mut c_void,
) -> i32 {
    use crate::state_block::Direct3DStateBlock9;
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::StateBlock);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !dev.begin_state_block_recording() {
        // The rejection leaves the open recording alone, per the D3D9
        // contract, so an application that begins a block and never ends it
        // has every later `BeginStateBlock` rejected until the next `Reset`
        // clears the recording. One misstep, not one per call: warn once.
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "BeginStateBlock called while another recording is in progress → INVALIDCALL. \
             A recording left open by an earlier BeginStateBlock rejects every later one \
             until the next Reset."
        );
        return D3DERR_INVALIDCALL;
    }
    D3D_OK
}

extern "system" fn device_end_state_block(this: *mut c_void, sb: *mut *mut c_void) -> i32 {
    use crate::state_block::Direct3DStateBlock9;

    let _api = device_api_lock(this);
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
        mtld3d_shared::log_once_warn!(
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::SetClipStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_clip_status(this: *mut c_void, _status: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetClipStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_texture(
    this: *mut c_void,
    stage: u32,
    texture: *mut *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Texture);
    if texture.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let vertex_slot = vertex_sampler_slot(stage);
    if vertex_slot.is_none() && stage as usize >= STAGE_COUNT {
        warn_invalid_sampler("GetTexture", stage);
        // SAFETY: `texture` is non-null (checked above) and per the D3D9
        // ABI points to a writable `*mut c_void` slot owned by the caller.
        unsafe { *texture = core::ptr::null_mut() };
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let tex_ptr = vertex_slot.map_or_else(
        || dev.stage_bindings().texture(stage as usize),
        |slot| dev.vertex_textures[slot].raw(),
    );
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

/// Map a `D3DVERTEXTEXTURESAMPLER0..3` stage index (257..=260) to a slot.
///
/// `SetTexture` / `Set|GetSamplerState` accept these next to the sixteen
/// fragment stages; everything else in that range stays invalid
/// (`D3DDMAPSAMPLER` = 256 is displacement mapping, unimplemented).
pub const fn vertex_sampler_slot(stage: u32) -> Option<usize> {
    match stage {
        257..=260 => Some((stage - 257) as usize),
        _ => None,
    }
}

/// Warn once per entry point that a texture or sampler call named a stage no sampler has.
///
/// Such a call is accepted and ignored, and a getter reads zero or null: a
/// stage past the sixteen fragment samplers and outside the four vertex ones.
/// `D3DDMAPSAMPLER` is one of them, as in Wine: no displacement-map sampler
/// exists here.
#[cold]
#[inline(never)]
fn warn_invalid_sampler(call: &str, stage: u32) {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}({stage}): no sampler has that stage → ignored, D3D_OK"
    );
}

/// Clamp a texture-stage-state stage and type into the stored table, as D3D9 runtimes do.
///
/// [`mtld3d_core::ff_state::clamp_texture_stage_state`] owns the rule; an
/// index outside the table is logged once per entry point.
fn clamp_texture_stage_state(call: &str, stage: u32, type_: u32) -> (u32, u32) {
    if mtld3d_core::ff_state::texture_stage_state_in_table(stage, type_) {
        return (stage, type_);
    }
    clamp_texture_stage_state_out_of_table(call, stage, type_)
}

/// The out-of-table half of [`clamp_texture_stage_state`]: log once and clamp.
///
/// Out of line so the texture-stage-state calls do not build the log
/// arguments on their hot path.
#[cold]
#[inline(never)]
fn clamp_texture_stage_state_out_of_table(call: &str, stage: u32, type_: u32) -> (u32, u32) {
    let (to_stage, to_type) = mtld3d_core::ff_state::clamp_texture_stage_state(stage, type_);
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}(stage {stage}, type {type_}): outside the table → clamped to stage {to_stage}, \
         type {to_type}"
    );
    (to_stage, to_type)
}

/// The `INVALIDCALL` a sampler-state call answers for a type past `D3DSAMP_DMAPOFFSET`.
#[cold]
#[inline(never)]
fn reject_sampler_state_type(call: &str, type_: u32) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}: no sampler state has type {type_} → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_texture(this: *mut c_void, stage: u32, texture: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Texture);
    let vertex_slot = vertex_sampler_slot(stage);
    if vertex_slot.is_none() && stage as usize >= STAGE_COUNT {
        warn_invalid_sampler("SetTexture", stage);
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    let new_tex = texture.cast::<Direct3DTexture9>();

    if let Some(rec) = dev.recording_state_block_mut() {
        // SAFETY: `new_tex` is null or a *mut Direct3DTexture9 supplied by
        // the calling game via SetTexture, which the reference taken here
        // keeps alive for the lifetime of the recording.
        let tex = unsafe { CachedComPtr::adopt(new_tex) };
        rec.record(StateOp::Texture { stage, tex });
        return D3D_OK;
    }

    if let Some(slot) = vertex_slot {
        dev.set_vertex_texture_slot(slot, new_tex);
        return D3D_OK;
    }
    if rebind_is_redundant(dev, stage as usize, new_tex) {
        dev.perf_mut().record_keys_gate(KeysGate::SetTexture, true);
        return D3D_OK;
    }
    let delta = dev
        .stage_bindings_mut()
        .replace_texture(stage as usize, new_tex);
    // STAGES on every swap: the encoder binds the new handle and
    // `snapshot_stage_bindings` re-runs flush_dirty_mips/rehydrate and
    // refreshes `cached_bound_texture_mask`. The FF VS/PS keys depend
    // only on the 8-bit occupancy mask (stages 0..7); the variant only
    // on `depth_sampler_mask` / `volume_sampler_mask` (any slot's
    // depth-format-ness / 3D-ness). A swap that flips none of those
    // rebuilds byte-identical keys, so gate those pieces on the actual
    // deltas. `ff_aware_mask` still strips VS/PS bits for programmable
    // shaders.
    let mut bits = SnapshotDirty::STAGES;
    if delta.intersects(
        TextureSwapDelta::DEPTH_CHANGED
            | TextureSwapDelta::VOLUME_CHANGED
            | TextureSwapDelta::CUBE_CHANGED
            | TextureSwapDelta::SAMPLING_CHANGED,
    ) {
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

/// Whether binding `tex` at pixel `stage` leaves every input of the stage walk as it is.
///
/// True when the stage already holds `tex` and `tex` is null or attached to
/// `dev`. The slot, its reference and the cached kind and Fetch4 masks are
/// functions of the pointer alone, so rewriting them changes nothing. What the
/// walk reads through the texture (its contents, pending uploads, LOD and
/// residency) is marked dirty on the texture's own device by whichever call
/// changes it, which is `dev` once the texture is attached here. A texture
/// attached to another device comes back through a rehydrate on this device,
/// which the stage walk performs for every bound stage, so that rebind keeps
/// its mark.
fn rebind_is_redundant(
    dev: &DeviceInner,
    stage: usize,
    tex: *mut crate::texture::Direct3DTexture9,
) -> bool {
    if dev.stage_bindings().texture(stage) != tex {
        return false;
    }
    // SAFETY: `tex` is the pointer the stage already holds, so it is null or
    // kept alive by the stage's reference.
    unsafe { tex.as_ref() }.is_none_or(|bound| {
        bound.inner().device_inner() == std::ptr::from_ref::<DeviceInner>(dev) as u64
    })
}

extern "system" fn device_get_texture_stage_state(
    this: *mut c_void,
    stage: u32,
    type_: u32,
    value: *mut u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::TexStageState);
    if value.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let (stage, type_) = clamp_texture_stage_state("GetTextureStageState", stage, type_);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::TexStageState);
    let (stage, type_) = clamp_texture_stage_state("SetTextureStageState", stage, type_);
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
    // VS/PS key byte-identical, so skip the rebuild. Each slot dirties only
    // the piece that reads it. ff_aware strips the VS/PS bits for
    // programmable shaders; the bump uniform is not an FF piece and needs
    // no such filter.
    if changed {
        let mask = match tss_write_feeds(type_) {
            TssWriteFeeds::StageConstant => dev.ff_aware_mask(SnapshotDirty::PS_CONST),
            TssWriteFeeds::BumpEnv => SnapshotDirty::BUMP_ENV,
            TssWriteFeeds::FfPipeline => dev.ff_aware_mask(
                SnapshotDirty::STAGES
                    | SnapshotDirty::VARIANT
                    | SnapshotDirty::VS_SOURCE
                    | SnapshotDirty::VS_CONST
                    | SnapshotDirty::PS_SOURCE
                    | SnapshotDirty::PS_CONST,
            ),
        };
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::SamplerState);
    if value.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let vertex_slot = vertex_sampler_slot(sampler);
    if vertex_slot.is_none() && sampler as usize >= STAGE_COUNT {
        warn_invalid_sampler("GetSamplerState", sampler);
        // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
        // points to a writable `u32` slot owned by the caller.
        unsafe { *value = 0 };
        return D3D_OK;
    }
    if type_ as usize >= SAMPLER_STATE_COUNT {
        return reject_sampler_state_type("GetSamplerState", type_);
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let read = vertex_slot.map_or_else(
        || {
            dev.stage_bindings()
                .sampler_state(sampler as usize, type_ as usize)
        },
        |slot| dev.vertex_sampler_states[slot][type_ as usize],
    );
    // SAFETY: `value` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `u32` slot owned by the caller.
    unsafe {
        *value = read;
    }
    0 // S_OK
}

extern "system" fn device_set_sampler_state(
    this: *mut c_void,
    sampler: u32,
    type_: u32,
    value: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::SamplerState);
    let vertex_slot = vertex_sampler_slot(sampler);
    if vertex_slot.is_none() && sampler as usize >= STAGE_COUNT {
        warn_invalid_sampler("SetSamplerState", sampler);
        return D3D_OK;
    }
    if type_ as usize >= SAMPLER_STATE_COUNT {
        return reject_sampler_state_type("SetSamplerState", type_);
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
            fetch4: if type_ == mtld3d_types::D3DSAMP_MIPMAPLODBIAS {
                mtld3d_core::fetch4::command(value)
            } else {
                None
            },
        });
        return D3D_OK;
    }
    if let Some(slot) = vertex_slot {
        dev.set_vertex_sampler_slot_state(slot, type_ as usize, value);
        return D3D_OK;
    }
    let stage = sampler as usize;
    let old_fetch4 = dev.stage_bindings().fetch4().masks();
    if dev
        .stage_bindings_mut()
        .set_sampler_state(stage, type_ as usize, value)
    {
        if old_fetch4 != dev.stage_bindings().fetch4().masks() {
            dev.mark_snapshot_dirty(SnapshotDirty::VARIANT);
        }
        // Sampler state lives inside StageBinding only.
        dev.mark_snapshot_dirty(SnapshotDirty::STAGES);
    } else if !rebind_is_redundant(dev, stage, dev.stage_bindings().texture(stage)) {
        // A write of the stored value changes no input of the stage walk, as
        // a rebind of the bound texture changes none, unless that texture is
        // attached to another device and needs the walk to come back.
        dev.mark_snapshot_dirty(SnapshotDirty::STAGES);
    }
    0 // S_OK
}

/// The `ValidateDevice` answer of the first sampler stage that rejects its filters.
///
/// Walks the sixteen fragment stages and then the four vertex-texture slots,
/// so the stage reported is the lowest-numbered one that fails. `None` means
/// every stage's filter setup runs, which is the only case that reports a
/// pass count.
fn rejected_filter_stage(dev: &DeviceInner) -> Option<i32> {
    let float32_filtering =
        crate::direct3d9::float32_filtering_supported(dev.config().deny_float32_filtering);
    let native_packed16 = crate::direct3d9::native_packed16_supported(dev.config().expand_packed16);
    let filterable = |texture: *mut Direct3DTexture9| {
        if texture.is_null() {
            return None;
        }
        // SAFETY: a non-null slot holds a live `Direct3DTexture9`; the binding
        // owns a reference for as long as the slot does.
        let format = unsafe { (*texture).d3d_format() };
        Some(mtld3d_core::format::supports_usage_query(
            format,
            D3DUSAGE_QUERY_FILTER,
            float32_filtering,
            native_packed16,
        ))
    };
    let fragment = (0..STAGE_COUNT).map(|stage| {
        (
            dev.stage_bindings().sampler_states(stage),
            dev.stage_bindings().texture(stage),
        )
    });
    let vertex = dev
        .vertex_sampler_states
        .iter()
        .zip(dev.vertex_textures.iter())
        .map(|(states, texture)| (*states, texture.raw()));
    for (sampler, (states, texture)) in fragment.chain(vertex).enumerate() {
        match stage_filter_verdict(&states, filterable(texture)) {
            StageFilterVerdict::Valid => {}
            StageFilterVerdict::FilterDisabled => {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "IDirect3DDevice9::ValidateDevice: sampler {sampler} disables its mag or min \
                     filter → UNSUPPORTEDTEXTUREFILTER"
                );
                return Some(D3DERR_UNSUPPORTEDTEXTUREFILTER);
            }
            StageFilterVerdict::TextureNotFilterable => {
                mtld3d_shared::log_once_warn!(
                    target: crate::LOG_TARGET,
                    "IDirect3DDevice9::ValidateDevice: sampler {sampler} filters a texture the \
                     device point-samples → E_FAIL"
                );
                return Some(E_FAIL);
            }
        }
    }
    None
}

extern "system" fn device_validate_device(this: *mut c_void, num_passes: *mut u32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // A sampler stage whose filters cannot run is the one thing D3D9 reports
    // here, and the pass count is left untouched when it does.
    if let Some(hr) = rejected_filter_stage(obj.inner()) {
        return hr;
    }
    // Metal validates the rest of the pipeline state at PSO-creation time, and
    // every fixed-function / shader state combination we accept renders in a
    // single pass, so the remaining state is always single-pass valid. Report
    // one pass and succeed. Returning INVALIDCALL would wrongly push games onto
    // a multi-pass / capability-fallback path.
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if entries.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // D3D9's palette API is non-functional on modern hardware: the setter
    // succeeds and the palette is simply ignored. One validation remains:
    // without D3DPTEXTURECAPS_ALPHAPALETTE — advertised only under
    // debug.capsAll — every PALETTEENTRY's peFlags must be 0xFF (fully
    // opaque); an alpha-bearing entry
    // is INVALIDCALL. The getters stay INVALIDCALL.
    if !obj.inner().config().caps_all {
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetPaletteEntries → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_current_texture_palette(this: *mut c_void, _palette: u32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::GetCurrentTexturePalette → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_set_scissor_rect(this: *mut c_void, rect: *const c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    // SAFETY: vtable in-param; `rect` is *const D3DRECT per ABI.
    let Some(r) = (unsafe { ValueIn::<D3DRECT>::read_opt(rect) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::ScissorRect(r));
        return D3D_OK;
    }
    if dev.frame_dump.active {
        dev.frame_dump_event(&format!(
            "SetScissorRect ({},{})-({},{})",
            r.x1, r.y1, r.x2, r.y2
        ));
    }
    dev.set_scissor_rect(r);
    // scissor_rect is the only piece of RenderStateSnapshot affected.
    dev.mark_snapshot_dirty(SnapshotDirty::RS);
    D3D_OK
}

extern "system" fn device_get_scissor_rect(this: *mut c_void, rect: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::ViewScissor);
    if rect.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    let scissor = dev.scissor_rect();
    // SAFETY: `rect` is non-null (checked above) and per the D3D9 ABI
    // points to a writable `RECT` (alias for `D3DRECT`) owned by the
    // caller.
    unsafe {
        *rect.cast::<D3DRECT>() = scissor;
    }
    D3D_OK
}

extern "system" fn device_set_software_vertex_processing(this: *mut c_void, software: i32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_info!(
        target: crate::LOG_TARGET,
        "IDirect3DDevice9::GetSoftwareVertexProcessing: obsolete, returning 0 (hardware VP)"
    );
    0
}

extern "system" fn device_set_npatch_mode(this: *mut c_void, segments: f32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
/// declaration, so a bound declaration is the whole test.
const fn has_vertex_layout_source(dev: &DeviceInner) -> bool {
    !dev.vertex_decl().is_null()
}

/// The `log_once_warn_by!` key of a log line shared by several entry points.
///
/// Keyed by the entry point's name, so each caller of a shared cold helper
/// logs its own first occurrence instead of the first caller hiding the rest.
fn caller_log_key(call: &str) -> u64 {
    use core::hash::BuildHasher as _;
    rustc_hash::FxBuildHasher.hash_one(call)
}

/// The `INVALIDCALL` every `Draw*` answers with no vertex declaration and no FVF bound.
#[cold]
#[inline(never)]
fn reject_missing_vertex_layout(call: &str) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call} with neither a vertex declaration nor an FVF bound → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

/// The `INVALIDCALL` a triangle fan answers when its rewrite to a list cannot address it.
#[cold]
#[inline(never)]
fn reject_unaddressable_fan(call: &str, primitive_count: u32) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}: a {primitive_count}-triangle fan has an index past u32 → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

/// The `INVALIDCALL` an inline draw answers for a zero vertex stride.
#[cold]
#[inline(never)]
fn reject_zero_up_stride(call: &str) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call} with a zero vertex stride → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

extern "system" fn device_draw_primitive(
    this: *mut c_void,
    primitive_type: u32,
    start_vertex: u32,
    primitive_count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return reject_missing_vertex_layout("DrawPrimitive");
    }
    // A draw of no primitives is valid and draws nothing.
    if primitive_count == 0 {
        std::hint::cold_path();
        return D3D_OK;
    }
    // Triangle fan has no Metal primitive: rewrite it as a triangle-list index
    // stream over the bound streams. Kept off the (non-fan) hot path below.
    if primitive_type == D3DPT_TRIANGLEFAN {
        std::hint::cold_path();
        // The encoder's shared 16-bit pattern covers every fan a 16-bit index
        // can address and costs nothing per draw; anything longer, or a start
        // vertex past the base-vertex range, gets a generated 32-bit list.
        let index_source = if primitive_count <= convert::FAN_PATTERN_MAX_TRIANGLES
            && i32::try_from(start_vertex).is_ok()
        {
            IndexSource::Fan {
                start_vertex,
                primitive_count,
            }
        } else {
            let Some(fan) = convert::FanRewrite::sequential(start_vertex, primitive_count) else {
                return reject_unaddressable_fan("DrawPrimitive", primitive_count);
            };
            generated_fan_source(dev, &fan)
        };
        return draw_bound_triangle_fan(&obj, index_source, D3DERR_INVALIDCALL);
    }
    let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    };
    let vtx_count = vertex_count(primitive_type, primitive_count);
    if vtx_count == 0 {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    }
    // Flush any bound buffer that's drawn while still mapped, before the draw
    // snapshot reads it.
    flush_mapped_bound_buffers(obj.inner());
    if let Err(hr) = obj.inner().latched_encoder_status() {
        std::hint::cold_path();
        return hr;
    }
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    let Some(vertex_source) = snapshot_bound_vertex_source(dev) else {
        std::hint::cold_path();
        if let Err(hr) = obj.inner().recording_status() {
            return hr;
        }
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "DrawPrimitive: no vertex buffer bound"
        );
        return D3DERR_INVALIDCALL;
    };
    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    let result = obj.inner().current_frame.record_single_stream_draw(
        &NonindexedDraw {
            primitive: metal_prim,
            start_vertex,
            vertex_count: vtx_count,
        },
        &vertex_source,
    );
    result.map_or_else(|error| obj.inner().capture_failure(error), |()| D3D_OK)
}

/// The `IndexSource` for a fan rewritten into an explicit index list.
///
/// The list is written straight into the frame's scratch arena, which is
/// what the unix side reads at replay time, so a fan that cannot ride the
/// encoder's shared pattern still allocates nothing and is copied once.
fn generated_fan_source(dev: &mut DeviceInner, fan: &convert::FanRewrite) -> IndexSource {
    let byte_len = fan.byte_len();
    let ptr = dev
        .current_frame
        .scratch_mut()
        .alloc_uninit_slice::<u8>(byte_len);
    // SAFETY: `alloc_uninit_slice` returned `byte_len` writable bytes in the
    // frame arena, and nothing else holds a reference to that block yet.
    let out = unsafe { core::slice::from_raw_parts_mut(ptr, byte_len) };
    fan.write(out);
    // SAFETY: frame scratch remains immutable and retained through submit replay.
    let data = unsafe {
        ScratchSlice::from_raw_parts(
            NonNull::new(ptr).expect("ScratchArena alloc returned non-null"),
            u32::try_from(byte_len).expect("fan index list fits u32"),
        )
    };
    IndexSource::Generated {
        data,
        index_count: fan.index_count(),
        index_type: fan.index_type(),
        min_vertex: fan.min_vertex(),
        max_vertex: fan.max_vertex(),
    }
}

/// Emit a bound-stream triangle fan as a triangle-list index draw.
///
/// `DrawPrimitive` and `DrawIndexedPrimitive` hand their fans here as an
/// `IndexSource::Fan` (the encoder's shared pattern) or `IndexSource::Generated`
/// (an explicit list); the vertices stay the bound streams, snapshotted
/// exactly as for any other bound draw. `no_vertex_buffer_hr` is what the
/// caller returns when no named stream has a buffer (the two entry points
/// differ there, see `device_draw_indexed_primitive`).
fn draw_bound_triangle_fan(
    obj: &Direct3DDevice9,
    index_source: IndexSource,
    no_vertex_buffer_hr: i32,
) -> i32 {
    // Flush any bound buffer that's drawn while still mapped, before the draw
    // snapshot reads it.
    flush_mapped_bound_buffers(obj.inner());
    if let Err(hr) = obj.inner().known_encoder_status() {
        return hr;
    }
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    let Some((first, extra)) = snapshot_bound_streams::<StreamBinding>(obj.inner()) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "triangle fan: no vertex buffer bound"
        );
        return no_vertex_buffer_hr;
    };
    let stream0_freq = obj.inner().bound_buffers().stream_freq(0);
    emit_snapshot_deltas(obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    obj.inner().record_draw(&DrawOp {
        metal_prim: mtld3d_shared::mtl::PrimitiveType::Triangle,
        vertex_source: VertexSource::Bound {
            first,
            extra,
            stream0_freq,
        },
        index_source,
    });
    obj.inner()
        .recording_status()
        .map_or_else(|hr| hr, |()| D3D_OK)
}

/// Complete an index buffer's CPU mirror from its device buffer.
///
/// A `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` index buffer keeps no CPU copy of
/// its bytes, and Metal has no triangle-fan primitive, so the rewrite below
/// still needs the application's indices. The device buffer holds them in
/// `StorageModePrivate` storage at an address the 32-bit PE cannot
/// dereference, so they come back as a GPU copy into fresh PE pages plus a
/// mid-frame submit that waits for it. The backing that lands is pinned, so
/// however many fans a buffer draws it stalls at most once.
///
/// `false`, with a warn, when the copy could not be made; the caller drops
/// the draw rather than rewriting a fan out of zeroed pages.
fn materialise_index_backing(dev: &mut DeviceInner, ib: *mut Direct3DIndexBuffer9) -> bool {
    // SAFETY: `ib` is non-null and points to a live `Direct3DIndexBuffer9`
    // whose bound-slot reference keeps it alive for this call.
    let inner = unsafe { &mut *ib }.inner_mut();
    // A fan can be drawn before Unlock. Its readback must follow the mapped
    // range's upload, just as it follows every already-queued Unlock upload.
    inner.flush_staged_if_mapped(dev);
    // A whole-buffer mapped write already completes the CPU mirror.
    if !inner.backing_needs_readback() {
        return true;
    }
    let buffer_id = inner.buffer_id();
    let mut page_box = PageBox::new_zeroed(inner.length() as usize);
    let dst_ptr = page_box.as_mut_ptr() as u64;
    let dst_len = page_box.len() as u64;
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "DrawIndexedPrimitive(D3DPT_TRIANGLEFAN) from an index buffer whose CPU copy was released: \
         reading the indices back off the GPU costs one mid-frame submit and one GPU wait per buffer"
    );
    let read = Arc::new(AtomicU32::new(0));
    let done = Arc::clone(&read).into();
    dev.push_control(crate::device::ReadDeviceBufferOp {
        done,
        buffer_id,
        dst_ptr,
        dst_len,
    });
    // Submits the frame the copy rides and waits for the GPU to finish it,
    // which is what makes the destination pages readable here.
    if dev.mid_frame_submit_for_retention().is_err() {
        // Admission may have succeeded even when the synchronous acknowledgment was lost.
        std::mem::forget(page_box);
        return false;
    }
    if read.load(Ordering::Acquire) == 0 {
        warn!(
            target: LOG_TARGET,
            "DrawIndexedPrimitive: triangle fan could not read its indices back off the GPU"
        );
        return false;
    }
    // SAFETY: `ib` stays live for this call (see above), and the mutable
    // reference taken at the top of this function has been dropped.
    unsafe { &mut *ib }.inner_mut().adopt_device_copy(page_box);
    true
}

/// Rewrite the fan a `DrawIndexedPrimitive` addresses in the bound index buffer.
///
/// Reads the application's indices straight from the buffer's CPU-side backing,
/// which is current under both map modes (the `Direct` box is the GPU memory
/// itself, the `Staged` box is the copy every Lock writes). A buffer that
/// released or partially restored its copy has it read back off the GPU first.
/// `None`, with a warn, when nothing is bound, the format is unknown, the
/// readback fails, or the draw reads past the buffer.
fn bound_index_fan(
    dev: &mut DeviceInner,
    start_index: u32,
    base_vertex: i32,
    primitive_count: u32,
) -> Option<IndexSource> {
    let ptr = dev.bound_buffers().index_buffer();
    if ptr.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "DrawIndexedPrimitive: no index buffer bound"
        );
        return None;
    }
    let (format, needs_readback) = {
        // SAFETY: `ptr` is non-null (checked above) and points to a live
        // `Direct3DIndexBuffer9` whose refcount keeps it alive while bound.
        let inner = unsafe { &*ptr }.inner();
        (inner.format(), inner.backing_needs_readback())
    };
    let index_size: u64 = match format {
        D3DFMT_INDEX16 => 2,
        D3DFMT_INDEX32 => 4,
        other => {
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: u64::from(other),
                "DrawIndexedPrimitive: unsupported index format {other}"
            );
            return None;
        }
    };
    if needs_readback && !materialise_index_backing(dev, ptr) {
        return None;
    }
    // SAFETY: `ptr` is non-null (checked above) and points to a live
    // `Direct3DIndexBuffer9` whose refcount keeps it alive while bound.
    let inner = unsafe { &*ptr }.inner();
    let first = u64::from(start_index) * index_size;
    let len = (u64::from(primitive_count) + 2) * index_size;
    if first + len > inner.current_backing_len() as u64 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "DrawIndexedPrimitive: triangle fan reads past the index buffer (start {start_index}, {primitive_count} primitives)"
        );
        return None;
    }
    let base = usize::try_from(inner.current_backing_ptr() as u64 + first).ok()?;
    let len = usize::try_from(len).ok()?;
    // SAFETY: `[first, first + len)` lies inside the live backing box (checked
    // above); the API thread owns CPU access to it while the buffer is bound.
    let src = unsafe { core::slice::from_raw_parts(base as *const u8, len) };
    let fan = convert::FanRewrite::indexed(
        src,
        usize::try_from(index_size).ok()?,
        base_vertex,
        primitive_count,
    )?;
    Some(generated_fan_source(dev, &fan))
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return reject_missing_vertex_layout("DrawIndexedPrimitive");
    }
    // A draw of no primitives is valid and draws nothing.
    if primitive_count == 0 {
        std::hint::cold_path();
        return D3D_OK;
    }
    // Triangle fan has no Metal primitive: rewrite the addressed indices as a
    // triangle list over the bound streams. Kept off the (non-fan) hot path.
    if primitive_type == D3DPT_TRIANGLEFAN {
        std::hint::cold_path();
        let Some(index_source) =
            bound_index_fan(dev, start_index, base_vertex_index, primitive_count)
        else {
            return D3DERR_INVALIDCALL;
        };
        // A valid declaration with no stream bound is S_OK for indexed draws,
        // same as the non-fan path below.
        return draw_bound_triangle_fan(&obj, index_source, D3D_OK);
    }
    let Some(metal_prim) = d3d_to_metal_primitive(primitive_type) else {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    };
    let index_count = vertex_count(primitive_type, primitive_count);
    if index_count == 0 {
        std::hint::cold_path();
        return D3DERR_INVALIDCALL;
    }

    // Flush any bound buffer that's drawn while still mapped, before the draw
    // snapshot reads it.
    flush_mapped_bound_buffers(obj.inner());
    if let Err(hr) = obj.inner().latched_encoder_status() {
        std::hint::cold_path();
        return hr;
    }
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    let Some(vertex_source) = snapshot_bound_vertex_source(dev) else {
        std::hint::cold_path();
        // D3D9 permits an indexed draw with a valid declaration but NO stream
        // source bound: it returns S_OK (rendering is undefined) rather than
        // INVALIDCALL — unlike the non-indexed DrawPrimitive. With no vertex
        // data there is nothing to
        // render, so skip the draw and report success.
        return obj
            .inner()
            .recording_status()
            .map_or_else(|hr| hr, |()| D3D_OK);
    };
    let Some(index_source) = snapshot_bound_index_source(dev, start_index) else {
        std::hint::cold_path();
        if let Err(hr) = obj.inner().recording_status() {
            return hr;
        }
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "DrawIndexedPrimitive: no index buffer bound"
        );
        return D3DERR_INVALIDCALL;
    };

    emit_snapshot_deltas(&obj);
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    let result = obj.inner().current_frame.record_single_stream_draw(
        &IndexedDraw {
            primitive: metal_prim,
            base_vertex: base_vertex_index,
            index_count,
            index: &index_source,
        },
        &vertex_source,
    );
    result.map_or_else(|error| obj.inner().capture_failure(error), |()| D3D_OK)
}

/// Upload any still-mapped `Staged` VB/IB dirty span before a draw reads it.
///
/// A buffer drawn while locked never reached `Unlock`'s upload, so its latest
/// CPU writes are flushed here. The lock stays open and `dirty` stays set, so
/// `Unlock` still flushes afterwards.
///
/// Only the streams the bound declaration reads are visited, the same set
/// [`snapshot_bound_streams`] captures: a buffer bound to a stream the draw
/// does not read is not read by it either.
#[inline]
fn flush_mapped_bound_buffers(dev: &mut DeviceInner) {
    let mut mask = declared_stream_mask(dev);
    while mask != 0 {
        let stream = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let vb = dev.bound_buffers().stream_vertex_buffer(stream);
        if !vb.is_null() {
            // SAFETY: a bound vertex buffer is a live wrapper while bound.
            unsafe { (*vb).inner_mut() }.flush_staged_if_mapped(dev);
        }
    }
    let ib = dev.bound_buffers().index_buffer();
    if !ib.is_null() {
        // SAFETY: a bound index buffer is a live wrapper while bound.
        unsafe { (*ib).inner_mut() }.flush_staged_if_mapped(dev);
    }
}

/// Snapshot the bound vertex streams the declaration reads for borrowed capture.
///
/// Only streams the bound declaration names are snapshotted (an implicit
/// FVF declaration names stream 0 alone), so a single-stream draw pays for
/// one binding. Each bound stream is stamped with the current submit seq so
/// the retention pipeline keeps its `PageBox` alive until that seq retires; a
/// named stream with nothing bound is left out and reads zeros at draw time.
/// The first stream is captured in its fixed record layout, like the index
/// buffer. `None` when no named stream has a buffer. Runs on the API thread.
/// Kept out of line so the draw receives the record as one contiguous value.
#[inline(never)]
fn snapshot_bound_vertex_source(dev: &DeviceInner) -> Option<BoundVertices> {
    let (first, extra) = snapshot_bound_streams::<StreamRecord>(dev)?;
    Some(BoundVertices {
        first,
        extra,
        stream0_freq: dev.bound_buffers().stream_freq(0),
    })
}

/// The vertex streams a draw reads: the bound declaration's, or stream 0 alone with none bound.
fn declared_stream_mask(dev: &DeviceInner) -> u16 {
    let decl_ptr = dev.vertex_decl();
    if decl_ptr.is_null() {
        1
    } else {
        // SAFETY: non-null; the device slot's refcount keeps the declaration
        // alive while bound.
        unsafe { &*decl_ptr }.inner().stream_mask()
    }
}

/// Snapshot the first bound declared stream and any further ones, as for a bound draw.
///
/// Generic over the first stream's form, so each caller converts it where it is captured.
fn snapshot_bound_streams<First: From<StreamBinding>>(
    dev: &DeviceInner,
) -> Option<(First, mtld3d_core::draw_data::ExtraStreams)> {
    let bound = dev.bound_buffers();
    let mut mask = declared_stream_mask(dev);
    if mask == 0 {
        return None;
    }
    let seq = dev.current_seq();
    let mut first: Option<First> = None;
    let mut extra = Vec::new();
    while mask != 0 {
        let stream = mask.trailing_zeros();
        mask &= mask - 1;
        let Some(binding) = snapshot_stream_binding(bound, stream, seq) else {
            continue;
        };
        if first.is_none() {
            first = Some(First::from(binding));
        } else {
            extra.push(binding);
        }
    }
    Some((
        first?,
        if extra.is_empty() {
            mtld3d_core::draw_data::ExtraStreams::Empty
        } else {
            mtld3d_core::draw_data::ExtraStreams::Owned(extra.into_boxed_slice())
        },
    ))
}

/// Snapshot a declared stream if bound, stamping its buffer with `seq`.
fn snapshot_stream_binding(bound: &BoundBuffers, stream: u32, seq: u64) -> Option<StreamBinding> {
    let s = stream as usize;
    let ptr = bound.stream_vertex_buffer(s);
    // SAFETY: a non-null slot holds a live buffer by its bound refcount. The
    // device API lock serializes access; an unbound declared stream is skipped.
    let vb = unsafe { ptr.as_mut() }?;
    let inner = vb.inner_mut();
    inner.stamp_submit_seq(seq);
    Some(StreamBinding {
        stream: u8::try_from(stream).expect("stream index below MAX_STREAMS"),
        buffer_id: inner.buffer_id(),
        backing_ptr: inner.current_backing_ptr(),
        backing_len: inner.current_backing_len(),
        backing_generation: inner.current_backing_generation(),
        offset: bound.stream_offset(s),
        stride: bound.stream_stride(s),
        freq: bound.stream_freq(s),
    })
}

/// Snapshot the bound index buffer directly into its fixed draw record.
///
/// Mirrors `snapshot_bound_vertex_source`: stamps the current submit seq and
/// collapses the draw's `start_index` into a byte offset.
fn snapshot_bound_index_source(dev: &DeviceInner, start_index: u32) -> Option<IndexBuffer> {
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
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: u64::from(other),
                "DrawIndexedPrimitive: unsupported index format {other}"
            );
            return None;
        }
    };
    inner.stamp_submit_seq(seq);
    Some(IndexBuffer {
        buffer: inner.buffer_id().raw(),
        address: inner.current_backing_ptr() as u64,
        length: inner.current_backing_len() as u64,
        generation: inner.current_backing_generation(),
        offset: start_index * index_stride,
        kind: index_type as u8,
        reserved: [0; 3],
    })
}

/// Copy `len` bytes of a `Draw*PrimitiveUP` vertex stream out of the user pointer.
///
/// The bytes travel to the encoder inside the draw op, so they are copied
/// once here rather than read from the application's memory later.
///
/// # Safety
///
/// `vertex_data` must be readable for `len` bytes for the duration of the
/// call, which the D3D9 ABI makes the caller's contract.
unsafe fn copy_up_vertices(
    dev: &mut DeviceInner,
    vertex_data: *const c_void,
    len: usize,
) -> ScratchSlice {
    // SAFETY: the caller guarantees `len` readable bytes at `vertex_data`.
    let source = unsafe { core::slice::from_raw_parts(vertex_data.cast::<u8>(), len) };
    // SAFETY: current_frame retains immutable scratch through native submit replay.
    unsafe { arena_alloc_bytes(dev.current_frame.scratch_mut(), source) }
}

/// Copy a UP vertex stream whose last crossing attribute ends past its vertices.
///
/// The `count * stride` bytes D3D9 promises behind `vertex_data` are copied
/// and zero-filled up to that attribute's end, `(count - 1) * stride +
/// extent`, which is what the attribute then reads. Returns the copy and its
/// size, or `None` when the size does not fit `u32`, which no vertex count a
/// draw can carry reaches. Only a draw whose stream-0 extent is past its
/// stride calls this; every other UP draw copies with [`copy_up_vertices`].
///
/// # Safety
///
/// `vertex_data` must be readable for `count * stride` bytes for the
/// duration of the call, which the D3D9 ABI makes the caller's contract.
#[cold]
#[inline(never)]
unsafe fn copy_up_crossing_vertices(
    dev: &mut DeviceInner,
    vertex_data: *const c_void,
    count: u32,
    stride: u32,
) -> Option<(ScratchSlice, u32)> {
    let extent = dev.cached_stream0_extent;
    let Some(size) = mtld3d_core::streams::inline_vertex_span(count, stride, extent) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "Draw*PrimitiveUP: {count} vertices of stride {stride} with a crossing attribute \
             ending at {extent} need more than 4 GiB → INVALIDCALL"
        );
        return None;
    };
    let supplied = count as usize * stride as usize;
    // SAFETY: the caller guarantees `count * stride` readable bytes at `vertex_data`.
    let source = unsafe { core::slice::from_raw_parts(vertex_data.cast::<u8>(), supplied) };
    // SAFETY: current_frame retains immutable scratch through native submit replay.
    let copy = unsafe {
        mtld3d_core::draw_data::arena_alloc_zero_padded(
            dev.current_frame.scratch_mut(),
            source,
            size as usize,
        )
    };
    Some((copy, size))
}

extern "system" fn device_draw_primitive_up(
    this: *mut c_void,
    primitive_type: u32,
    primitive_count: u32,
    vertex_data: *const c_void,
    vertex_stride: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if vertex_stride == 0 {
        return reject_zero_up_stride("DrawPrimitiveUP");
    }
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return reject_missing_vertex_layout("DrawPrimitiveUP");
    }
    // A draw of no primitives is valid and draws nothing.
    if primitive_count == 0 {
        return D3D_OK;
    }

    // Triangle fan has no Metal primitive: the fan's vertices go up as they
    // are and an index list makes the triangles. Kept off the (non-fan) hot
    // path below.
    if primitive_type == D3DPT_TRIANGLEFAN {
        if vertex_data.is_null() {
            return D3DERR_INVALIDCALL;
        }
        // The encoder's shared 16-bit pattern is relative to the fan's first
        // vertex, which the inline stream starts at, so it covers every fan a
        // 16-bit index can address; anything longer gets a generated list.
        let index_source = if primitive_count <= convert::FAN_PATTERN_MAX_TRIANGLES {
            IndexSource::Fan {
                start_vertex: 0,
                primitive_count,
            }
        } else {
            let Some(fan) = convert::FanRewrite::sequential(0, primitive_count) else {
                return reject_unaddressable_fan("DrawPrimitiveUP", primitive_count);
            };
            generated_fan_source(dev, &fan)
        };
        let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
        let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
        emit_snapshot_deltas(&obj);
        drop(snap);
        // After the snapshot, which resolves the declaration the copy is sized by.
        let (vertex_copy, fan_size) = if dev.cached_stream0_extent <= vertex_stride {
            let fan_bytes = (primitive_count as usize + 2) * vertex_stride as usize;
            // SAFETY: per the D3D9 ABI the caller guarantees `(primitive_count + 2)`
            // vertices of `vertex_stride` bytes are readable from `vertex_data`.
            let vertex_copy = unsafe { copy_up_vertices(dev, vertex_data, fan_bytes) };
            (
                vertex_copy,
                u32::try_from(fan_bytes).expect("triangle-fan UP size fits u32"),
            )
        } else {
            let fan_vertices = primitive_count.saturating_add(2);
            // SAFETY: per the D3D9 ABI the caller guarantees the fan's vertices
            // of `vertex_stride` bytes are readable; the copy reads no more.
            let copy =
                unsafe { copy_up_crossing_vertices(dev, vertex_data, fan_vertices, vertex_stride) };
            let Some(copy) = copy else {
                return D3DERR_INVALIDCALL;
            };
            copy
        };
        let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
        let metal_prim =
            d3d_to_metal_primitive(D3DPT_TRIANGLELIST).expect("triangle list is supported");
        dev.record_draw(&DrawOp {
            metal_prim,
            vertex_source: VertexSource::Up {
                bytes: vertex_copy,
                size: fan_size,
                stride: vertex_stride,
            },
            index_source,
        });
        // D3D9 resets stream source 0 to (NULL, 0, 0) after DrawPrimitiveUP.
        dev.bound_buffers_mut().reset_stream0();
        return obj
            .inner()
            .encoder_status()
            .map_or_else(|hr| hr, |()| D3D_OK);
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

    emit_snapshot_deltas(&obj);
    // After the snapshot, which resolves the declaration the copy is sized by.
    let (vertex_copy, data_size) = if dev.cached_stream0_extent <= vertex_stride {
        let data_size = (vtx_count * vertex_stride) as usize;
        // SAFETY: `vertex_data` covers `data_size` bytes per the caller's stride
        // contract.
        let vertex_copy = unsafe { copy_up_vertices(dev, vertex_data, data_size) };
        (
            vertex_copy,
            u32::try_from(data_size).expect("DrawPrimitiveUP data size fits u32"),
        )
    } else {
        // SAFETY: per the D3D9 ABI `vertex_data` covers `vtx_count` vertices of
        // `vertex_stride` bytes; the copy reads no more.
        let copy = unsafe { copy_up_crossing_vertices(dev, vertex_data, vtx_count, vertex_stride) };
        let Some(copy) = copy else {
            return D3DERR_INVALIDCALL;
        };
        copy
    };
    drop(snap);
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    dev.record_draw(&DrawOp {
        metal_prim,
        vertex_source: VertexSource::Up {
            bytes: vertex_copy,
            size: data_size,
            stride: vertex_stride,
        },
        index_source: IndexSource::None {
            start_vertex: 0,
            vertex_count: vtx_count,
        },
    });
    // D3D9 resets stream source 0 to (NULL, 0, 0) after DrawPrimitiveUP.
    dev.bound_buffers_mut().reset_stream0();
    obj.inner()
        .encoder_status()
        .map_or_else(|hr| hr, |()| D3D_OK)
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
    propagate_const_delta(
        dev,
        mtld3d_shared::encoder_protocol::EncoderOpcode::SetVsConstRange,
        start_register,
        slice,
    );
}

pub fn propagate_ps_const_delta(dev: &mut DeviceInner, start_register: u32, slice: &[[f32; 4]]) {
    propagate_const_delta(
        dev,
        mtld3d_shared::encoder_protocol::EncoderOpcode::SetPsConstRange,
        start_register,
        slice,
    );
}

fn propagate_const_delta(
    dev: &mut DeviceInner,
    opcode: mtld3d_shared::encoder_protocol::EncoderOpcode,
    start_register: u32,
    slice: &[[f32; 4]],
) {
    let start = start_register as usize;
    if start >= CONSTANT_ROWS || slice.is_empty() {
        return;
    }
    let rows = (CONSTANT_ROWS - start).min(slice.len());
    let start_row = u16::try_from(start).expect("start row is within constant mirror");
    dev.current_frame
        .record_constant_source(opcode, start_row, &slice[..rows]);
}

/// Record changed state directly from the dirty-state builders.
///
/// Clean draws retain the native encoder's current snapshot. Dirty draws emit
/// only the changed fields, then the following Draw consumes the native state.
fn emit_snapshot_deltas(obj: &Direct3DDevice9) {
    let dirty = obj.inner().snapshot_dirty;
    if dirty.is_empty() {
        // A draw with the same state as its predecessor still gets its dump
        // line and its trace marker; the cached snapshot is its state.
        if obj.inner().frame_dump.active {
            obj.inner().frame_dump_draw();
        }
        return;
    }
    let sampled = obj.inner().perf_mut().record_snapshot_rebuild(dirty.bits());

    let stages_ptr = draw_snapshot_stages_ptr(DeviceInner::perf_ptr_of(obj.inner));
    let stages_timer = CycleAddTimer::start(stages_ptr);

    // STAGES first: `flush_dirty_mips` inside `snapshot_stage_bindings`
    // pushes upload ops via `dev.push_op`, which mutably borrows
    // `current_frame.ops`. Done before we hold our own `&mut` on
    // scratch/ops below.
    let stage_bindings_arr_opt = if dirty.contains(SnapshotDirty::STAGES) {
        let dev = obj.inner();
        // Vertex bindings ride encoder ops, but their CPU writes dirty the
        // snapshot just like fragment writes. Flush before queuing this draw.
        for slot in 0..dev.vertex_textures.len() {
            let tex = dev.vertex_texture(slot);
            if tex.is_null() {
                continue;
            }
            // SAFETY: the bound-slot refcount keeps this separate texture
            // allocation live; the device API lock serialises its access.
            let bound = unsafe { &mut *tex };
            // Bound here, then used on another device: back it comes, as a
            // fragment stage's texture does in the stage walk.
            crate::texture::rehydrate_for_device(bound, dev);
            crate::texture::flush_dirty_mips(bound.inner_mut(), dev);
            bound.inner_mut().note_gpu_use();
        }
        let (arr, ff_mask, packed_mask) = snapshot_stage_bindings(dev);
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
    // starts so the buckets don't double-count. Each section's own timer
    // runs inside its dirty branch, a child of `keys_timer`, and only on a
    // sampled draw: on the others its target is null and it reads no clock.
    // The sampled draws also time the whole scope into a slot of their own,
    // so the part of `keys` outside the sections is taken on the same draws
    // as the sections, with the same timer cost in both.
    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let section_perf_ptr = if sampled {
        perf_ptr
    } else {
        core::ptr::null_mut()
    };
    let keys_timer = CycleAddTimer::start(draw_snapshot_keys_ptr(perf_ptr));
    let keys_sampled_timer = CycleAddTimer::start(draw_snapshot_keys_sampled_ptr(section_perf_ptr));
    let dev = obj.inner();

    // VDECL FIRST — its rebuild updates `dev.cached_ff_vs_layout`,
    // which conflicts with the long-lived `dev.render_states()` borrow
    // taken below.
    let vdecl_value = if dirty.contains(SnapshotDirty::VDECL) {
        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::Vdecl,
        ));
        let bound_vertex_shader = dev.shader_bindings().vertex_shader();
        let fvf = dev.fvf();
        let decl_ptr = dev.vertex_decl();
        let (resolved, vdecl_hash, ff_vs_layout) = if decl_ptr.is_null() {
            let (elements, _fvf_stride) = fvf_to_elements(fvf);
            let layout = convert::ff_vs_layout_from_elements(&elements);
            // Pre-transformed (POSITIONT/XYZRHW) layouts bypass a bound VS —
            // D3D9 runs the FF pre-transformed path regardless, even when a
            // VS is still bound — so the attrs must resolve for the FF VS too.
            let resolved = if bound_vertex_shader.is_null() || layout.has_rhw() {
                resolve_attrs_for_ff(&elements, &[0; MAX_LINKED_INPUTS])
            } else {
                // SAFETY: non-null check passed; refcount holds it live.
                let vs_obj = unsafe { &*bound_vertex_shader };
                resolve_attrs_for_vs(&elements, vs_obj.input_semantics())
            };
            (resolved, u64::from(fvf), layout)
        } else {
            // SAFETY: non-null check passed; refcount holds it live.
            let decl = unsafe { &*decl_ptr };
            let elements = decl.inner().elements();
            let layout = convert::ff_vs_layout_from_elements(elements);
            // See the FVF arm: POSITIONT bypasses a bound VS.
            let resolved = if bound_vertex_shader.is_null() || layout.has_rhw() {
                resolve_attrs_for_ff(elements, &decl.inner().passthrough())
            } else {
                // SAFETY: see above.
                let vs_obj = unsafe { &*bound_vertex_shader };
                resolve_attrs_for_vs(elements, vs_obj.input_semantics())
            };
            (resolved, decl.inner().hash(), layout)
        };
        dev.cached_ff_vs_layout = ff_vs_layout;
        dev.cached_stream0_extent = resolved.extents[0];
        // Which VS input registers the declaration backs — folded into a
        // programmable VsSource so a shader reading an unprovided input gets a
        // distinct zero-filled variant. For the FF
        // path the value is unused.
        dev.cached_vs_provided_mask = resolved.attrs.iter().fold(0u16, |m, a| {
            if a.attr_index < 16 {
                m | (1u16 << a.attr_index)
            } else {
                m
            }
        });
        Some((resolved, vdecl_hash))
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

        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::Rs,
        ));
        // `SetRenderState` stores whatever DWORD the game passed, so an
        // enum state is narrowed through `render_state::enum_value`: the
        // byte when the value is inside that state's enum space, the D3D9
        // default when it is not.
        let enum_rs = |state: u32| mtld3d_core::render_state::enum_value(rs, state);

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
            PipelineRsFlags::ALPHA_TO_COVERAGE,
            mtld3d_core::multisample::alpha_to_coverage_requested(rs, dev.a2m_enabled()),
        );
        let pipeline_rs = PipelineRsBits {
            flags: prs_flags,
            src_blend: enum_rs(D3DRS_SRCBLEND),
            dst_blend: enum_rs(D3DRS_DESTBLEND),
            blend_op: enum_rs(D3DRS_BLENDOP),
            src_blend_alpha: enum_rs(D3DRS_SRCBLENDALPHA),
            dst_blend_alpha: enum_rs(D3DRS_DESTBLENDALPHA),
            blend_op_alpha: enum_rs(D3DRS_BLENDOPALPHA),
            color_write_mask: enum_rs(D3DRS_COLORWRITEENABLE),
            color_write_mask_ext: [
                enum_rs(D3DRS_COLORWRITEENABLE1),
                enum_rs(D3DRS_COLORWRITEENABLE2),
                enum_rs(D3DRS_COLORWRITEENABLE3),
            ],
        };

        let depth_stencil_state = mtld3d_core::depth_stencil_state::snapshot_from_state(rs);
        let mut depth_scissor = DepthScissorFlags::empty();
        depth_scissor.set(
            DepthScissorFlags::DEPTH_ENABLE,
            depth_stencil_state.depth_enable != 0,
        );
        depth_scissor.set(
            DepthScissorFlags::DEPTH_WRITE,
            depth_stencil_state.depth_write != 0,
        );
        depth_scissor.set(
            DepthScissorFlags::SCISSOR_TEST,
            rs[D3DRS_SCISSORTESTENABLE as usize] != 0,
        );

        // `SetScissorRect` stores any RECT; the snapshot carries the region
        // it lets through, saturated to `u16`.
        let scissor_rect = mtld3d_core::render_state::scissor_snapshot_rect(dev.scissor_rect());

        // `D3DRS_MULTISAMPLEMASK` only means anything against a maskable
        // multisampled target, and which target is bound is API-thread
        // knowledge; `SetRenderTarget` re-dirties this section, so the
        // narrowing cannot go stale.
        let rt0 = dev.bound_rt().render_target(0);
        let rt0_multi_sample = if rt0.is_null() {
            SurfaceMultiSample {
                multi_sample_type: dev.backbuffer_multi_sample_type(),
                multi_sample_quality: dev.backbuffer_multi_sample_quality(),
                sample_count: dev.backbuffer_sample_count(),
            }
        } else {
            // SAFETY: non-null check passed; the bound-slot refcount holds
            // the surface live.
            unsafe { (*rt0).multi_sample() }
        };
        Some(RenderStateSnapshot {
            reserved: 0,
            pipeline_rs,
            depth_scissor,
            depth_stencil_state,
            cull_mode: enum_rs(D3DRS_CULLMODE),
            fill_mode: enum_rs(D3DRS_FILLMODE),
            scissor_rect,
            blend_factor: rs[D3DRS_BLENDFACTOR as usize],
            depth_bias: rs[D3DRS_DEPTHBIAS as usize],
            slope_scale_depth_bias: rs[D3DRS_SLOPESCALEDEPTHBIAS as usize],
            stencil_ref: rs[D3DRS_STENCILREF as usize],
            sample_mask: mtld3d_core::multisample::effective_sample_mask(
                rs[D3DRS_MULTISAMPLEMASK as usize],
                rt0_multi_sample.sample_count,
                rt0_multi_sample.multi_sample_type,
            ),
        })
    } else {
        None
    };

    // RT_DS: depth/stencil presence.
    let depth_stencil_value = if dirty.contains(SnapshotDirty::RT_DS) {
        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::RtDs,
        ));
        let bound_ds = dev.bound_rt().depth_stencil();
        let (has_depth, has_stencil) = if !bound_ds.is_null() {
            // SAFETY: non-null check passed; refcount holds it live.
            let fmt = unsafe { (*bound_ds).depth_attachment_format() };
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
        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::Variant,
        ));
        let mut variant = dev.ff_state().variant_key(
            rs,
            dev.cached_ff_vs_layout.has_rhw(),
            dev.uses_automatic_fog(),
        );
        variant.depth_sampler_mask = dev.stage_bindings().depth_sampler_mask();
        variant.depth_fetch_mask = dev.stage_bindings().depth_fetch_mask();
        variant.raw_depth_red_mask = dev.stage_bindings().fetch4().raw_red_mask();
        (variant.fetch4_mask, variant.fetch4_alpha_mask) = dev.stage_bindings().fetch4().masks();
        variant.volume_sampler_mask = dev.stage_bindings().volume_sampler_mask();
        variant.cube_sampler_mask = dev.stage_bindings().cube_sampler_mask();
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
        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::VsSource,
        ));
        if bound_vertex_shader.is_null() || dev.cached_ff_vs_layout.has_rhw() {
            let key = dev.ff_state().build_vs_key(
                rs,
                dev.cached_ff_vs_layout,
                bound_mask,
                dev.bound_decl_passthrough(),
            );
            mtld3d_shared::crumb!(
                "ffvs:cap",
                dev.current_seq(),
                u64::from(key.tex_coord_count),
            );
            let max_row_count = dev.ff_state().ff_vs_row_count(&key);
            Some(VsSource::FixedFunction(
                mtld3d_core::draw_data::FixedVsSource {
                    key,
                    max_row_count,
                    reserved: [0; 6],
                },
            ))
        } else {
            // SAFETY: non-null check; refcount holds it live.
            let vs_obj = unsafe { &*bound_vertex_shader };
            Some(VsSource::Programmable(
                mtld3d_core::draw_data::ProgrammableVsSource {
                    vs_id: vs_obj.shader_id(),
                    max_const_used: clamp_const_rows(vs_obj.max_const_used()),

                    provided_input_mask: dev.cached_vs_provided_mask,

                    clip_plane_count: mtld3d_core::vs_draw::clip_plane_count(rs),
                    sampler_kinds: dev.vertex_texture_kinds(),
                    reserved: [0; 7],

                    flags: (if vs_obj.uses_rel_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::RELATIVE
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if vs_obj.uses_int_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::INTEGER
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if vs_obj.uses_bool_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::BOOLEAN
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }),
                },
            ))
        }
    } else {
        None
    };

    // PS_SOURCE.
    let ps_value = if dirty.contains(SnapshotDirty::PS_SOURCE) {
        let _section = CycleAddTimer::start(draw_snapshot_section_ptr(
            section_perf_ptr,
            SnapshotSection::PsSource,
        ));
        if bound_pixel_shader.is_null() {
            let key = dev.ff_state().build_ps_key(rs, bound_mask);
            let sampled_stage_mask = key.sampled_stage_mask();
            let constant_rows = key.constant_rows();
            Some(PsSource::FixedFunction(
                mtld3d_core::draw_data::FixedPsSource {
                    key,
                    sampled_stage_mask,
                    constant_rows,

                    reserved: [0; 3],
                },
            ))
        } else {
            // SAFETY: non-null check; refcount holds it live.
            let ps_obj = unsafe { &*bound_pixel_shader };
            Some(PsSource::Programmable(
                mtld3d_core::draw_data::ProgrammablePsSource {
                    ps_id: ps_obj.shader_id(),
                    max_const_used: clamp_const_rows(ps_obj.max_const_used()),

                    color_out_mask: ps_obj.color_out_mask(),

                    flags: (if ps_obj.uses_int_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::INTEGER
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if ps_obj.uses_bool_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::BOOLEAN
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if ps_obj.uses_bump_env() {
                        mtld3d_core::draw_data::ShaderSourceFlags::BUMP_ENV
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if ps_obj.uses_rel_const() {
                        mtld3d_core::draw_data::ShaderSourceFlags::RELATIVE
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }) | (if ps_obj.reads_linked_inputs() {
                        mtld3d_core::draw_data::ShaderSourceFlags::LINKED_INPUTS
                    } else {
                        mtld3d_core::draw_data::ShaderSourceFlags::empty()
                    }),

                    reserved: [0; 4],
                },
            ))
        }
    } else {
        None
    };

    drop(keys_sampled_timer);
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
        // FF VS constants borrow the new source or the owned cached key.
        // The existing section builders clone only the FF key they consume.
        let key_ref = vs_value.as_ref().or(dev.snapshot_cache.vs.as_ref());
        let key = match key_ref {
            Some(VsSource::FixedFunction(value)) => value.key.clone(),
            _ => dev.ff_state().build_vs_key(
                rs,
                dev.cached_ff_vs_layout,
                bound_mask,
                dev.bound_decl_passthrough(),
            ),
        };
        let ff_dirty = dev.ff_state.take_ff_vs_dirty();
        if !ff_dirty.is_empty() {
            if key.has_rhw() {
                // A pretransformed layout reads row 0 alone, so every other
                // section stays marked for the next draw that reads it. WV is
                // the one this row overwrites, and a change back to a
                // transformed layout marks it again.
                dev.ff_state
                    .mark_ff_vs_dirty(ff_dirty.difference(FfVsDirty::WV));
                let v = dev.viewport();
                let to_f32 = |n: u32| {
                    f32::from(u16::try_from(n).expect("D3D9 viewport dim ≤ 16384 fits u16"))
                };
                let viewport = (to_f32(v.x), to_f32(v.y), to_f32(v.width), to_f32(v.height));
                dev.current_frame
                    .record_ff_vs_destination(0, 1, |destination| {
                        FfState::fill_xyzrhw_row(viewport, destination);
                    });
            } else {
                if ff_dirty.contains(FfVsDirty::WV) {
                    dev.current_frame
                        .record_ff_vs_destination(0, 4, |destination| {
                            dev.ff_state.fill_wv_section(destination);
                        });
                }
                if ff_dirty.contains(FfVsDirty::PROJ) {
                    dev.current_frame
                        .record_ff_vs_destination(4, 4, |destination| {
                            dev.ff_state.fill_proj_section(destination);
                        });
                }
                if ff_dirty.contains(FfVsDirty::FOG) {
                    dev.current_frame
                        .record_ff_vs_destination(8, 1, |destination| {
                            FfState::fill_fog_section(rs, key.fog_mode, destination);
                        });
                }
                if ff_dirty.contains(FfVsDirty::AMBIENT) {
                    dev.current_frame
                        .record_ff_vs_destination(9, 1, |destination| {
                            FfState::fill_ambient_section(rs, destination);
                        });
                }
                if ff_dirty.contains(FfVsDirty::MATERIAL) {
                    let rows = FfState::material_section_rows(&key);
                    dev.current_frame
                        .record_ff_vs_destination(10, rows, |destination| {
                            dev.ff_state.fill_material_section(&key, destination);
                        });
                }
                if ff_dirty.contains(FfVsDirty::LIGHTS) {
                    let rows = FfState::lights_section_rows(&key);
                    if rows != 0 {
                        dev.current_frame
                            .record_ff_vs_destination(15, rows, |destination| {
                                dev.ff_state.fill_lights_section(&key, destination);
                            });
                    }
                }
                if ff_dirty.contains(FfVsDirty::TT) {
                    let rows = dev.ff_state.tt_section_rows();
                    if rows != 0 {
                        dev.current_frame
                            .record_ff_vs_destination(63, rows, |destination| {
                                dev.ff_state.fill_tt_section(destination);
                            });
                    }
                }
                if ff_dirty.contains(FfVsDirty::PALETTE) {
                    let rows = dev.ff_state.palette_section_rows(&key);
                    if rows != 0 {
                        dev.current_frame.record_ff_vs_destination(
                            mtld3d_core::dxso::FF_VS_PALETTE_BASE_ROW,
                            rows,
                            |destination| dev.ff_state.fill_palette_section(&key, destination),
                        );
                    }
                }
            }
        }
        // FF VS const bytes are NOT carried in the snapshot cache
        // anymore — the encoder mirror is the source of truth, and
        // `emit_draw` snapshots from it via `enc.ff_vs_const_scratch`.
    }

    // PS_CONST source. Same routing as VS_CONST: programmable PS uses
    // the encoder mirror; only FF runs here.
    let ps_rows = if dirty.contains(SnapshotDirty::PS_CONST) && bound_pixel_shader.is_null() {
        match ps_value.as_ref().or(dev.snapshot_cache.ps.as_ref()) {
            Some(PsSource::FixedFunction(value)) => value.constant_rows,
            _ => 0,
        }
    } else {
        0
    };
    let ps_stage_constants =
        if dirty.contains(SnapshotDirty::PS_CONST) && bound_pixel_shader.is_null() && ps_rows > 1 {
            let ptr =
                dev.ff_state
                    .build_ps_stage_constants(rs, ps_rows, dev.current_frame.scratch_mut());
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            Some(unsafe {
                ScratchSlice::from_raw_parts(
                    NonNull::new(ptr).expect("stage constants scratch is non-null"),
                    u32::from(ps_rows) * 16,
                )
            })
        } else {
            None
        };
    let ps_const_buf = if dirty.contains(SnapshotDirty::PS_CONST)
        && bound_pixel_shader.is_null()
        && ps_rows <= 1
    {
        Some(dev.ff_state().build_ps_constants(rs))
    } else {
        None
    };

    // ALPHA_REF + FOG_COLOR bytes (variant must be current).
    let alpha_ref_buf = if dirty.contains(SnapshotDirty::ALPHA_REF) {
        let variant = variant_value
            .or(dev.snapshot_cache.variant)
            .unwrap_or_default();
        Some(build_alpha_ref_bytes(variant, dev.ff_state().alpha_ref(rs)))
    } else {
        None
    };
    let fog_color_buf = if dirty.contains(SnapshotDirty::FOG_COLOR) {
        let variant = variant_value
            .or(dev.snapshot_cache.variant)
            .unwrap_or_default();
        Some(mtld3d_core::ff_state::build_fog_color_bytes(rs, variant))
    } else {
        None
    };
    // Per-draw VsDraw uniform (point size, inverse view, clip planes), read
    // by every vertex shader.
    let vs_draw_buf = if dirty.contains(SnapshotDirty::VS_DRAW) {
        let view = dev
            .ff_state()
            .transform(mtld3d_types::D3DTS_VIEW)
            .copied()
            .unwrap_or(D3DMATRIX::IDENTITY);
        let point_size = dev.point_size();
        #[cfg(perf_tracking)]
        {
            let bytes = dev
                .vs_draw
                .build_bytes(rs, point_size, &view, &dev.clip_planes);
            dev.perf
                .state_mut()
                .record_inverse_view(dev.vs_draw.last_use());
            Some(bytes)
        }
        #[cfg(not(perf_tracking))]
        Some(
            dev.vs_draw
                .build_bytes(rs, point_size, &view, &dev.clip_planes),
        )
    } else {
        None
    };
    // Bump-environment matrix bytes (PS slot 12). Built only when a bump TSS
    // state changed (rare); the slot is bound at draw time only for a PS that
    // actually uses texbem/texbeml/bem.
    let bump_env_buf = if dirty.contains(SnapshotDirty::BUMP_ENV) {
        Some(dev.ff_state().build_bump_env_bytes())
    } else {
        None
    };
    // VS integer-constant bytes (vertex slot 14). Built when an integer
    // constant changed (rare) or on the first draw of a frame (all-dirty); the
    // slot is bound at draw time only for a VS that reads a dynamic integer
    // constant. Mirrors the bump-env capture above.
    let vs_int_const_buf = if dirty.contains(SnapshotDirty::VS_CONST_I) {
        Some(dev.shader_bindings().vs_constants_i_bytes())
    } else {
        None
    };
    // VS boolean-constant bitmask (vertex slot 26), same lifecycle as the
    // integer file above.
    let vs_bool_const_buf = if dirty.contains(SnapshotDirty::VS_CONST_B) {
        Some(dev.shader_bindings().vs_constants_b_bits().to_ne_bytes())
    } else {
        None
    };
    // PS integer / boolean constant files (fragment slots 11 / 10), the
    // fragment-side twins with the same lifecycle.
    let ps_int_const_buf = if dirty.contains(SnapshotDirty::PS_CONST_I) {
        Some(dev.shader_bindings().ps_constants_i_bytes())
    } else {
        None
    };
    let ps_bool_const_buf = if dirty.contains(SnapshotDirty::PS_CONST_B) {
        Some(dev.shader_bindings().ps_constants_b_bits().to_ne_bytes())
    } else {
        None
    };

    // Capture each changed structural value directly from its builder output.
    // Only uniform bytes need an initial copy into retained guest scratch.
    let mut byte_bindings;
    let bytes = if dirty.intersects(
        SnapshotDirty::VS_CONST
            | SnapshotDirty::PS_CONST
            | SnapshotDirty::ALPHA_REF
            | SnapshotDirty::FOG_COLOR
            | SnapshotDirty::BUMP_ENV
            | SnapshotDirty::VS_CONST_I
            | SnapshotDirty::VS_CONST_B
            | SnapshotDirty::PS_CONST_I
            | SnapshotDirty::PS_CONST_B
            | SnapshotDirty::VS_DRAW,
    ) {
        let scratch = dev.current_frame.scratch_mut();
        byte_bindings = [None; 10];
        if dirty.contains(SnapshotDirty::VS_CONST) {
            byte_bindings[0] = Some(None);
        }
        if dirty.contains(SnapshotDirty::PS_CONST) {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[1] = Some(
                ps_stage_constants
                    .or_else(|| ps_const_buf.map(|b| unsafe { arena_alloc_bytes(scratch, &b) })),
            );
        }
        if let Some((buf, len)) = alpha_ref_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[2] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf[..len]) }));
        }
        if let Some((buf, len)) = fog_color_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[3] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf[..len]) }));
        }
        if let Some(buf) = bump_env_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[4] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        if let Some(buf) = vs_int_const_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[5] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        if let Some(buf) = vs_bool_const_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[6] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        if let Some(buf) = ps_int_const_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[7] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        if let Some(buf) = ps_bool_const_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[8] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        if let Some(buf) = vs_draw_buf {
            // SAFETY: frame scratch remains immutable and retained through submit replay.
            byte_bindings[9] = Some(Some(unsafe { arena_alloc_bytes(scratch, &buf) }));
        }
        Some(&byte_bindings)
    } else {
        None
    };
    drop(consts_timer);
    let bumps_timer =
        CycleAddTimer::start(draw_snapshot_bumps_ptr(DeviceInner::perf_ptr_of(obj.inner)));
    let stages = stage_bindings_arr_opt.as_ref().map(|(packed, mask)| {
        // SAFETY: snapshot_stage_bindings initialized exactly popcount(mask)
        // entries. The borrowed prefix is consumed synchronously by capture.
        let prefix = unsafe {
            core::slice::from_raw_parts(
                packed.as_ptr().cast::<StageBinding>(),
                mask.count_ones() as usize,
            )
        };
        (*mask, prefix)
    });
    let delta = SnapshotDelta {
        render_state: render_state_value.as_ref(),
        stages,
        attrs: vdecl_value
            .as_ref()
            .map(|(resolved, hash)| SnapshotAttributes {
                attrs: &resolved.attrs,
                extents: &resolved.extents,
                used_streams: resolved.used_streams,
                vdecl_hash: *hash,
            }),
        vs: vs_value.as_ref().map(VsSource::as_view),
        ps: ps_value.as_ref().map(PsSource::as_view),
        variant: variant_value,
        bytes,
        depth_stencil: depth_stencil_value,
    };
    // Failure stays latched in the frame recorder and rejects the entire frame.
    dev.current_frame.record_snapshot_delta(&delta);
    if let Some(value) = vs_value {
        dev.snapshot_cache.vs = Some(value);
    }
    if let Some(value) = ps_value {
        dev.snapshot_cache.ps = Some(value);
    }
    if let Some(value) = variant_value {
        dev.snapshot_cache.variant = Some(value);
    }
    if let Some(value) = depth_stencil_value {
        dev.snapshot_cache.depth_stencil = value;
    }
    dev.snapshot_dirty = SnapshotDirty::empty();
    if dev.frame_dump.active {
        dev.frame_dump_draw();
    }
    drop(bumps_timer);
}

/// Give a system-memory texture the Metal texture its pool withheld.
///
/// A `D3DPOOL_SYSTEMMEM` or `D3DPOOL_SCRATCH` texture is created with staging
/// and nothing else, so one an application only locks, copies from, or reads
/// back into never reaches the GPU. D3D9 samples such a texture when it is
/// bound at a texture stage, so that bind is where the allocation has to
/// happen; the levels already written are marked dirty, and the flush that
/// follows the bind uploads them. A no-op for every texture that already has
/// a Metal texture, which is every texture in a GPU-resident pool.
fn promote_cpu_only_texture(dev: &mut DeviceInner, tex: &mut crate::texture::Direct3DTexture9) {
    if !crate::texture::promote_to_gpu(tex.inner_mut()) {
        return;
    }
    let id = tex.texture_id();
    // Not a gap: this is what D3D9 does. The line is a residency probe, so a
    // title that pays video memory for a pool meant to avoid it is visible.
    mtld3d_shared::log_once_info_by!(
        target: LOG_TARGET,
        key: id.raw(),
        "texture {:#x} (D3DPOOL={}) sampled from a texture stage: allocating the Metal texture \
         its system-memory pool had withheld",
        id.raw(),
        tex.inner().d3d_pool()
    );
    push_texture_warmups(dev, tex.inner());
}

/// Capture bound textures as a packed [`StageBinding`] prefix plus the bound-stage masks.
///
/// Walks only the set bits of `StageBindings::bound_mask` and writes the
/// captured bindings as an initialised prefix of the returned array: the
/// first `popcount(packed_mask)` slots, in ascending stage order, exactly
/// the layout `bump_packed_stage_bindings` copies into scratch. Returns
/// `(packed prefix, FF bound-texture mask (stages 0–7), packed_mask)`.
///
/// Writable texture unlocks dirty the draw snapshot. This walk schedules
/// their pending uploads before capturing the bindings the draw consumes.
fn snapshot_stage_bindings(
    dev: &mut DeviceInner,
) -> ([MaybeUninit<StageBinding>; STAGE_COUNT], u8, u16) {
    let mut packed: [MaybeUninit<StageBinding>; STAGE_COUNT] =
        [const { MaybeUninit::uninit() }; STAGE_COUNT];
    let mut bound_texture_mask: u8 = 0;
    let mut out_idx: usize = 0;
    let mut packed_mask = dev.stage_bindings().bound_mask();
    let mut remaining = packed_mask;
    while remaining != 0 {
        let stage = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let tex_ptr = dev.stage_bindings().texture(stage);
        if tex_ptr.is_null() {
            // `bound_mask` promises a non-null slot; a mismatch means the
            // incremental mask update drifted. Drop the bit so the packed
            // count stays in agreement with the entries actually written.
            debug_assert!(false, "bound_mask bit {stage} set for a null texture slot");
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "stage {stage}: bound_mask set but texture slot is null; skipping"
            );
            packed_mask &= !(1u16 << stage);
            continue;
        }
        // The `&mut Direct3DTexture9` here doesn't alias `dev: &mut
        // DeviceInner` because the texture is a separate Box:
        // `dev.stage_bindings.textures[stage]` is a raw `*mut`, not a tracked
        // reference.
        // SAFETY: `tex_ptr` is the bound-texture pointer from
        // `stage_bindings` (checked non-null above); the texture is held
        // alive by the stage-bindings refcount until rebound.
        let tex = unsafe { &mut *tex_ptr };
        // A system-memory texture bound for sampling stops being CPU-only
        // here, before the flush below can find a level to upload.
        promote_cpu_only_texture(dev, tex);
        // FF combiner only consumes stages 0–7 (MaxTextureBlendStages = 8);
        // stages 8–15 are programmable-PS-only and would shift past the
        // u8 mask width.
        if stage < 8 {
            bound_texture_mask |= 1u8 << stage;
        }
        let mut sampler_state = dev.stage_bindings().sampler_states(stage);
        if dev.samples_unfiltered(tex) {
            mtld3d_core::sampler_state::sample_unfiltered(&mut sampler_state);
        }
        // Lazy texture upload: flush any per-mip `dirty` flags before
        // capturing TextureInfo. Operations pushed by `schedule_upload`
        // precede the Draw operation on the encoder thread, so the
        // upload runs before the bind reads.
        // Cross-device migration handler — must run before flush_dirty_mips
        // so the re-marked dirty bits drive an upload against the new
        // device's encoder + handles.
        crate::texture::rehydrate_for_device(tex, dev);
        crate::texture::flush_dirty_mips(tex.inner_mut(), dev);
        tex.inner_mut().note_gpu_use();
        // A texture's SetLOD is its most detailed level. The sampler
        // translation and the explicit-LOD rows read it from the copy's spare
        // slot; LOD 0 (the common case) leaves both as the state alone.
        sampler_state[TEXTURE_LOD_SLOT] = tex.inner().lod();
        packed[out_idx].write(StageBinding {
            texture_id: tex.texture_id(),
            sampler_state,
        });
        out_idx += 1;
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
    debug_assert_eq!(
        out_idx,
        packed_mask.count_ones() as usize,
        "packed prefix length must equal the popcount of the returned mask"
    );
    (packed, bound_texture_mask, packed_mask)
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if vertex_stride == 0 {
        return reject_zero_up_stride("DrawIndexedPrimitiveUP");
    }
    let dev = obj.inner();
    if !has_vertex_layout_source(dev) {
        return reject_missing_vertex_layout("DrawIndexedPrimitiveUP");
    }
    // A draw of no primitives is valid and draws nothing.
    if primitive_count == 0 {
        return D3D_OK;
    }
    if index_data.is_null() || vertex_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let (index_type, index_size): (mtld3d_shared::mtl::IndexType, usize) = match index_format {
        D3DFMT_INDEX16 => (mtld3d_shared::mtl::IndexType::UInt16, 2),
        D3DFMT_INDEX32 => (mtld3d_shared::mtl::IndexType::UInt32, 4),
        other => {
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: u64::from(other),
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

    // Build the index stream. Triangle fan has no Metal primitive, so the
    // inline indices are gathered into a triangle list (fan vertices
    // 0, i+1, i+2) written straight into the frame arena.
    let (metal_prim, index_source) = if primitive_type == D3DPT_TRIANGLEFAN {
        let src_bytes = (primitive_count as usize + 2) * index_size;
        // SAFETY: per the D3D9 ABI `index_data` covers at least
        // `(primitive_count + 2)` indices of `index_size` bytes.
        let src = unsafe { core::slice::from_raw_parts(index_data.cast::<u8>(), src_bytes) };
        // Inline indices are absolute, so no base vertex folds in.
        let Some(fan) = convert::FanRewrite::indexed(src, index_size, 0, primitive_count) else {
            return reject_unaddressable_fan("DrawIndexedPrimitiveUP", primitive_count);
        };
        (
            d3d_to_metal_primitive(D3DPT_TRIANGLELIST).expect("triangle list is supported"),
            generated_fan_source(dev, &fan),
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
        // SAFETY: the D3D9 caller supplies `idx_bytes` readable index data;
        // current_frame retains the captured immutable bytes through replay.
        let index_copy = unsafe { copy_up_vertices(dev, index_data, idx_bytes) };

        (
            metal_prim,
            IndexSource::Up {
                bytes: index_copy,
                index_count,
                index_type,
            },
        )
    };

    let perf_ptr = DeviceInner::perf_ptr_of(obj.inner);
    let snap = CycleAddTimer::start(draw_snapshot_ptr(perf_ptr));
    emit_snapshot_deltas(&obj);
    drop(snap);
    // After the snapshot, which resolves the declaration the copy is sized by.
    let (vertex_copy, vtx_size) = if dev.cached_stream0_extent <= vertex_stride {
        // SAFETY: per the D3D9 ABI `vertex_data` covers at least
        // `(min_vertex_index + num_vertices) * vertex_stride` bytes.
        let vertex_copy = unsafe { copy_up_vertices(dev, vertex_data, vtx_bytes) };
        (
            vertex_copy,
            u32::try_from(vtx_bytes).expect("DrawIndexedPrimitiveUP vertex size fits u32"),
        )
    } else {
        let count = min_vertex_index.saturating_add(num_vertices);
        // SAFETY: per the D3D9 ABI `vertex_data` covers `min_vertex_index +
        // num_vertices` vertices of `vertex_stride` bytes; the copy reads no more.
        let copy = unsafe { copy_up_crossing_vertices(dev, vertex_data, count, vertex_stride) };
        let Some(copy) = copy else {
            return D3DERR_INVALIDCALL;
        };
        copy
    };
    let _push = CycleAddTimer::start(draw_push_op_ptr(perf_ptr));
    dev.record_draw(&DrawOp {
        metal_prim,
        vertex_source: VertexSource::Up {
            bytes: vertex_copy,
            size: vtx_size,
            stride: vertex_stride,
        },
        index_source,
    });
    // D3D9 resets stream source 0 to (NULL, 0, 0) AND the index buffer to NULL
    // after a successful DrawIndexedPrimitiveUP.
    let bound = dev.bound_buffers_mut();
    bound.reset_stream0();
    bound.replace_index_buffer(core::ptr::null_mut());
    obj.inner()
        .encoder_status()
        .map_or_else(|hr| hr, |()| D3D_OK)
}

extern "system" fn device_process_vertices(
    this: *mut c_void,
    src_start: u32,
    dst_index: u32,
    count: u32,
    dst_buffer: *mut c_void,
    decl: *mut c_void,
    _flags: u32,
) -> i32 {
    use crate::vertex_buffer::Direct3DVertexBuffer9;

    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Draws);
    if dst_buffer.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // A vertex declaration source (as opposed to the current FVF) is not
    // implemented; software vertex processing is a conformance-only path.
    if !decl.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "ProcessVertices with an explicit vertex declaration is unimplemented → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // The source is stream 0 read through the device's current FVF.
    let src_vb = dev.bound_buffers().stream_vertex_buffer(0);
    if src_vb.is_null() {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "ProcessVertices with no stream-0 vertex buffer bound → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let src_fvf = dev.fvf();
    let src_stream_offset = dev.bound_buffers().stream_offset(0);
    let src_stride = dev.bound_buffers().stream_stride(0);
    let wvp = dev.ff_state().world_view_projection();
    let viewport = dev.viewport();

    // SAFETY: `src_vb` is a live bound `Direct3DVertexBuffer9` (non-null
    // checked); the bound-slot reference keeps it alive for this call.
    let src_inner = unsafe { (*src_vb).inner() };
    if src_inner.backing_is_released() {
        // A default-pool D3DUSAGE_WRITEONLY buffer keeps its bytes on the GPU
        // alone. D3DUSAGE_SOFTWAREPROCESSING is the usage a title declares for
        // a buffer it feeds to ProcessVertices, and that declaration keeps the
        // CPU copy, so a source without it has nothing left to read.
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "ProcessVertices from a D3DUSAGE_WRITEONLY source without D3DUSAGE_SOFTWAREPROCESSING: no CPU copy to read → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    let src_bytes = src_inner.backing();
    let first = (src_stream_offset + src_start.saturating_mul(src_stride)) as usize;
    if first > src_bytes.len() {
        return D3DERR_INVALIDCALL;
    }

    // SAFETY: `dst_buffer` is a live `Direct3DVertexBuffer9*` per the D3D9 ABI;
    // it is distinct from `src_vb` (the source is a bound stream, the
    // destination a caller-owned buffer) so the two borrows do not alias.
    let dst_obj = unsafe { &mut *dst_buffer.cast::<Direct3DVertexBuffer9>() };
    let dst_fvf = dst_obj.inner().fvf();

    let processed = mtld3d_core::process_vertices::process_vertices(
        &mtld3d_core::process_vertices::ProcessVerticesRequest {
            src: &src_bytes[first..],
            src_stride,
            src_fvf,
            dst_fvf,
            count,
            wvp,
            viewport,
        },
    );
    let Some(processed) = processed else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "ProcessVertices: destination FVF {dst_fvf:#x} has no transformed position or the \
             source is too short → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    };
    if mtld3d_core::process_vertices::dst_wants_shaded_output(dst_fvf)
        && dev.render_state(D3DRS_LIGHTING as usize) != 0
    {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "ProcessVertices does not run software lighting; the destination colour is the \
             source colour passed through"
        );
    }
    let (_, dst_stride) = mtld3d_core::convert::fvf_to_elements(dst_fvf);
    let dst_offset = (dst_index.saturating_mul(dst_stride)) as usize;
    dst_obj
        .inner_mut()
        .write_processed(dst_offset, &processed, dev);
    dev.encoder_status().map_or_else(|hr| hr, |()| D3D_OK)
}

extern "system" fn device_create_vertex_declaration(
    this: *mut c_void,
    elements: *const c_void,
    decl: *mut *mut c_void,
) -> i32 {
    const MAX_ELEMENTS: usize = 64; // generous; D3D9 limit is MAXD3DDECLLENGTH=64

    let _api = device_api_lock(this);
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
        fvf: 0,
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
    let _api = device_api_lock(this);
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
    // The FVF is the declaration's (`DeviceInner::fvf`): the one `SetFVF`
    // built for an FVF keeps reporting that FVF when it is bound here, and a
    // declaration the game created reports 0, so the pointer decides both.
    let changed = dev.replace_vertex_decl(new);
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
            && convert::vertex_decl_has_rhw(unsafe { (*new).inner().elements() });
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
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Shader);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    if dev.is_state_block_recording() {
        // A recorded SetFVF is a change of declaration, to the one built for
        // the FVF, so a `Capture` of the block refreshes it from the bound
        // declaration like a recorded SetVertexDeclaration. SetFVF(0) changes
        // nothing and records nothing.
        if fvf != 0 {
            let decl = dev.get_or_create_fvf_decl(fvf);
            if decl.is_null() {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "recorded SetFVF({fvf:#x}): no declaration for the FVF → DRIVERINTERNALERROR, nothing recorded"
                );
                return mtld3d_types::D3DERR_DRIVERINTERNALERROR;
            }
            if let Some(rec) = dev.recording_state_block_mut() {
                // SAFETY: `decl` is a live declaration the FVF cache holds
                // until the device drops.
                let adopted = unsafe { CachedComPtr::adopt(decl) };
                rec.record(StateOp::VertexDeclaration(adopted));
            }
        }
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
    // changes nothing, so skip the VDECL rebuild; `bind_fvf_decl` answers
    // that without the cache lookup when the FVF and its declaration are
    // already bound. An FVF whose declaration cannot be built leaves the
    // bound declaration as it is, as in Wine.
    let Some(changed) = dev.bind_fvf_decl(fvf) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "SetFVF({fvf:#x}): no declaration for the FVF → DRIVERINTERNALERROR, binding kept"
        );
        return mtld3d_types::D3DERR_DRIVERINTERNALERROR;
    };
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
    let _api = device_api_lock(this);
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
    unsafe { *fvf = obj.inner().fvf() };
    0 // S_OK
}

/// Read a DXSO token stream from a caller-supplied pointer until the End opcode (0x0000FFFF).
///
/// Skips comment payloads and each instruction's operand run, so a payload or
/// immediate word that happens to equal 0xFFFF is not misread as End. The
/// operand run is counted the way the model requires, which for SM1 is not a
/// field but a scan: see [`operand_token_count`]. Getting that wrong here does
/// not merely mis-measure, it walks past the End token and keeps reading
/// whatever follows the shader in the caller's address space.
///
/// The version word is validated rather than skipped: the walk cannot pick its
/// counting rule without the model, and a stream whose first token is not a
/// VS/PS version word has no length to find.
fn read_shader_bytecode(ptr: *const u32) -> Option<Vec<u32>> {
    const MAX_TOKENS: usize = 65536;
    // SAFETY: both callers reject a null `pFunction` before reaching here, and
    // a non-null token stream is at least its version word.
    let version = unsafe { *ptr };
    match (version >> 16) & 0xFFFF {
        0xFFFE | 0xFFFF => {}
        _ => {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "shader creation: first token 0x{version:08X} is not a vertex or pixel shader version word → INVALIDCALL"
            );
            return None;
        }
    }
    let major = ((version >> 8) & 0xFF) as u8;
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
            len += operand_token_count(major, tok, |n| {
                if len + n >= MAX_TOKENS {
                    return None;
                }
                // SAFETY: same stream as the read above, under the same
                // `MAX_TOKENS` bound.
                let peek_ptr = unsafe { ptr.add(len + n) };
                // SAFETY: the bound above keeps `len + n` inside the same
                // stream the loop is already reading.
                Some(unsafe { *peek_ptr })
            });
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
fn maybe_dump_bytecode(dir: &str, prefix: &str, shader_id: ProgramId, bytecode: &[u32]) {
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    null_out(shader);
    if function.is_null() || shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let Some(bytecode) = read_shader_bytecode(function) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; this is a live Direct3DDevice9 per the D3D9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mut created = match create_native_shader(
        obj.inner(),
        &bytecode,
        mtld3d_shared::shader_create::ShaderStage::Vertex,
    ) {
        Ok(created) => created,
        Err(result) => return result,
    };
    let mut const_usage = crate::vertex_shader::VsConstUsage::empty();
    const_usage.set(
        crate::vertex_shader::VsConstUsage::USES_REL_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::RELATIVE_CONST),
    );
    const_usage.set(
        crate::vertex_shader::VsConstUsage::USES_INT_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::INT_CONST),
    );
    const_usage.set(
        crate::vertex_shader::VsConstUsage::USES_BOOL_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::BOOL_CONST),
    );
    if let Err(result) = created.adopt(obj.inner()) {
        return result;
    }
    let shader_obj = Direct3DVertexShader9::new(
        obj.inner_ptr(),
        created.shader_id,
        created.max_const_used,
        const_usage,
        core::mem::take(&mut created.input_semantics),
        Arc::from(bytecode),
    );
    let shader_ptr = Box::into_raw(Box::new(shader_obj));
    // SAFETY: shader_ptr is a freshly created live shader at refcount one.
    unsafe { crate::com_ref::com_register_child(shader_ptr) };
    // SAFETY: the D3D9 ABI supplies a writable shader out pointer, checked non-null above.
    unsafe { OutPtr::write_opt(shader, shader_ptr.cast::<c_void>()) };
    D3D_OK
}

/// Parsed-program ownership until an ordered frame operation accepts it.
struct CreatedNativeShader {
    runtime: u64,
    registration: u64,
    shader_id: ProgramId,
    max_const_used: u32,
    usage: mtld3d_shared::shader_create::ShaderUsage,
    color_out_mask: u8,
    input_semantics: Vec<InputSemantic>,
}

impl CreatedNativeShader {
    fn adopt(&mut self, dev: &mut DeviceInner) -> Result<(), i32> {
        dev.try_push_control(mtld3d_core::encoder_data::AdoptProgramOp {
            registration: self.registration,
        })
        .map_err(|error| {
            error!(target: LOG_TARGET, "shader creation: cannot record adoption: {error:?}");
            mtld3d_types::E_OUTOFMEMORY
        })?;
        // The frame now retains the registration through rejection or submission.
        self.registration = 0;
        Ok(())
    }
}

impl Drop for CreatedNativeShader {
    fn drop(&mut self) {
        if self.registration != 0 {
            let mut cancel = mtld3d_shared::shader_create::CancelShaderProgramParams {
                runtime: self.runtime,
                registration: self.registration,
            };
            if unix_call(&mut cancel) != D3D_OK {
                error!(target: LOG_TARGET, "shader creation: pending-program cancellation failed");
            }
        }
    }
}

fn create_native_shader(
    dev: &DeviceInner,
    bytecode: &[u32],
    stage: mtld3d_shared::shader_create::ShaderStage,
) -> Result<CreatedNativeShader, i32> {
    use mtld3d_shared::shader_create::{
        CreateShaderProgramParams, ShaderInputSemantic, ShaderStage, ShaderUsage,
    };

    if !dev.config().bytecode_dump_dir.is_empty() {
        let prefix = match stage {
            ShaderStage::Vertex => "vs",
            ShaderStage::Pixel => "ps",
        };
        // Rejected programs are dumped too, before native validation.
        maybe_dump_bytecode(
            &dev.config().bytecode_dump_dir,
            prefix,
            ProgramId::from_tokens(bytecode),
            bytecode,
        );
    }
    // Each semantic requires at least one token, so this bounds repeated declarations too.
    let capacity = if stage == ShaderStage::Vertex {
        bytecode.len()
    } else {
        0
    };
    let mut semantics = Vec::<MaybeUninit<ShaderInputSemantic>>::new();
    semantics
        .try_reserve_exact(capacity)
        .map_err(|_| mtld3d_types::E_OUTOFMEMORY)?;
    semantics.resize_with(capacity, MaybeUninit::uninit);
    let mut params = CreateShaderProgramParams {
        runtime: dev.encoder_runtime(),
        tokens_ptr: bytecode.as_ptr() as u64,
        semantics_ptr: semantics.as_mut_ptr() as u64,
        stage,
        token_count: u32::try_from(bytecode.len()).expect("shader token walk is bounded by 65536"),
        semantic_capacity: u32::try_from(capacity)
            .expect("semantic capacity is bounded by token count"),
        result: D3DERR_INVALIDCALL,
        program_id: 0,
        registration: 0,
        max_const_used: 0,
        semantic_count: 0,
        usage: ShaderUsage::empty(),
        color_out_mask: 0,
        padding: [0; 6],
    };
    let status = unix_call(&mut params);
    if status != D3D_OK || params.result != D3D_OK {
        return Err(if status == D3D_OK {
            params.result
        } else {
            status
        });
    }
    let mut created = CreatedNativeShader {
        runtime: params.runtime,
        registration: params.registration,
        shader_id: ProgramId::from_shader_reply(params.program_id),
        max_const_used: params.max_const_used,
        usage: params.usage,
        color_out_mask: params.color_out_mask,
        input_semantics: Vec::new(),
    };
    let count = params.semantic_count as usize;
    if count > semantics.len() || created.registration == 0 {
        error!(target: LOG_TARGET, "shader creation: invalid native metadata reply");
        return Err(D3DERR_INVALIDCALL);
    }
    created
        .input_semantics
        .try_reserve_exact(count)
        .map_err(|_| mtld3d_types::E_OUTOFMEMORY)?;
    for semantic in &semantics[..count] {
        // SAFETY: successful native validation initialized exactly semantic_count output slots.
        let semantic = unsafe { semantic.assume_init_ref() };
        created.input_semantics.push(InputSemantic {
            usage: semantic.usage,
            usage_index: semantic.usage_index,
            register_index: semantic.register_index,
        });
    }
    Ok(created)
}

extern "system" fn device_set_vertex_shader(this: *mut c_void, shader: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    if !window_in_range(start_register, count, CONSTANT_ROWS) {
        return reject_constant_window(
            "SetVertexShaderConstantF",
            start_register,
            count,
            CONSTANT_ROWS,
        );
    }
    // A zero count names an empty window, which D3D9 accepts and writes nothing into.
    if count == 0 {
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` non-zero (both
    // settled above); per the D3D9 ABI the caller guarantees the bounded floats
    // are initialized and readable without mutation in one allocation for
    // this call. The byte size fits `isize::MAX`; all `f32` bits are valid.
    let constants = unsafe {
        mtld3d_shared::slice_from_caller(constant_data.cast::<[f32; 4]>(), count as usize)
    };
    let slice: &[[f32; 4]] = &constants;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantF {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // Redundant-set elimination: a write that leaves every mirror row
    // unchanged yields a byte-identical encoder delta, so skip the delta
    // push when nothing changed. WoW re-uploads identical
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
        // The delta invalidates the encoder's constant scratch cache directly.
        // Programmable float registers do not change the captured snapshot.
        propagate_vs_const_delta(dev, start_register, slice);
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetVsConst, !changed);
    0
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

/// How many registers an integer or boolean constant call moves, or the `HRESULT` refusing it.
///
/// D3D9 refuses a null array and a start at or past the `rows`-deep register
/// file whatever the count, accepts a zero count as a call that moves
/// nothing, and clamps a count that runs past the file to the registers that
/// remain, so the caller copy is bounded before an unaligned input can
/// allocate. Shared by the eight `Set`/`Get` integer and boolean thunks.
fn int_bool_constant_rows(
    call: &str,
    has_data: bool,
    start: u32,
    count: u32,
    rows: usize,
) -> Result<usize, i32> {
    if !has_data {
        return Err(D3DERR_INVALIDCALL);
    }
    int_bool_rows(start, count, rows).ok_or_else(|| reject_constant_start(call, start, rows))
}

/// The `INVALIDCALL` an integer or boolean constant call answers for a start past the file.
///
/// Out of line so the setters do not build the log arguments on their hot path.
#[cold]
#[inline(never)]
fn reject_constant_start(call: &str, start: u32, rows: usize) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}: start register {start} past the {rows}-register file → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

/// The `INVALIDCALL` a float constant setter or getter answers for a window past the file.
///
/// Out of line so the four calls do not build the log arguments on their hot path.
#[cold]
#[inline(never)]
fn reject_constant_window(call: &str, start: u32, count: u32, rows: usize) -> i32 {
    mtld3d_shared::log_once_warn_by!(
        target: LOG_TARGET,
        key: caller_log_key(call),
        "{call}: registers {start}+{count} run past the {rows}-register file → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

extern "system" fn device_get_vertex_shader_constant_f(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut f32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    if !window_in_range(start_register, count, CONSTANT_ROWS) {
        return reject_constant_window(
            "GetVertexShaderConstantF",
            start_register,
            count,
            CONSTANT_ROWS,
        );
    }
    // A zero count reads nothing, which D3D9 accepts.
    if count == 0 {
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_copy();
    // SAFETY: `constant_data` is non-null and `count` non-zero (both settled above);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "SetVertexShaderConstantI",
        !constant_data.is_null(),
        start_register,
        count,
        INT_CONSTANT_ROWS,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` is bounded above;
    // per the D3D9 ABI, the bounded rows are initialized and readable
    // without mutation in one allocation for this call. The byte size
    // fits `isize::MAX`, and all `i32` bit patterns are valid.
    let constants =
        unsafe { mtld3d_shared::slice_from_caller(constant_data.cast::<[i32; 4]>(), count) };
    let slice: &[[i32; 4]] = &constants;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantI {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // A VS reading a dynamic integer constant (a `loop`/`rep` counter) consumes
    // these via the `vs_i` buffer (vertex slot 14), captured into the snapshot
    // on change. Boolean constants take the same route as a bitmask.
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "GetVertexShaderConstantI",
        !constant_data.is_null(),
        start_register,
        count,
        INT_CONSTANT_ROWS,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_i_copy();
    // SAFETY: `constant_data` is non-null and `count` is non-zero and clamped to the file (above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data.cast::<[i32; 4]>(), count) };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_vertex_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "SetVertexShaderConstantB",
        !constant_data.is_null(),
        start_register,
        count,
        BOOL_CONSTANT_COUNT,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` is bounded above;
    // per the D3D9 ABI, the bounded BOOLs are initialized and readable
    // without mutation in one allocation for this call. The byte size
    // fits `isize::MAX`, and all `i32` bit patterns are valid.
    let constants = unsafe { mtld3d_shared::slice_from_caller(constant_data, count) };
    let slice = &*constants;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::VertexShaderConstantB {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    // A VS reading a dynamic boolean constant (a static `if b0`) consumes
    // these as the `vs_b` bitmask (vertex slot 26), captured into the
    // snapshot on change.
    let changed = dev
        .shader_bindings_mut()
        .write_vs_constants_b(start_register, slice);
    if changed {
        dev.mark_snapshot_dirty(SnapshotDirty::VS_CONST_B);
    }
    0
}

extern "system" fn device_get_vertex_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "GetVertexShaderConstantB",
        !constant_data.is_null(),
        start_register,
        count,
        BOOL_CONSTANT_COUNT,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().vs_constants_b_copy();
    // SAFETY: `constant_data` is non-null and `count` is non-zero and clamped to the file (above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data, count) };
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if stream >= mtld3d_types::MAX_STREAMS {
        mtld3d_shared::log_once_warn!(
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
        // SAFETY: `vb` is null or a *mut Direct3DVertexBuffer9 supplied by
        // the calling game via SetStreamSource.
        let adopted = unsafe { CachedComPtr::adopt(vb) };
        rec.record(StateOp::StreamSource {
            stream,
            vb: adopted,
            offset,
            stride,
        });
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
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if stream_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // Streams 0..MAX round-trip their binding; a caller that binds a stream
    // and reads it back — relying on the binding outliving its own Release —
    // sees the buffer. An out-of-range stream is unbound → NULL/0 per the
    // "nothing bound" contract (S_OK).
    let (vb_ptr, offset, vb_stride) = if stream < mtld3d_types::MAX_STREAMS {
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
    stream: u32,
    setting: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if let Err(reason) = validate_stream_freq(stream, setting) {
        // A rejected call leaves the stored frequency untouched.
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject SetStreamSourceFreq(stream={stream}, setting={setting:#x}) → INVALIDCALL ({reason:?})"
        );
        return D3DERR_INVALIDCALL;
    }
    let dev = obj.inner();
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::StreamSourceFreq { stream, setting });
        return D3D_OK;
    }
    dev.bound_buffers_mut()
        .set_stream_freq(stream as usize, setting);
    D3D_OK
}

extern "system" fn device_get_stream_source_freq(
    this: *mut c_void,
    stream: u32,
    setting: *mut u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = bind_timer(this, BindSubCategory::Buffer);
    if setting.is_null() || stream >= mtld3d_types::MAX_STREAMS {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // The raw word round-trips, flags included.
    let value = obj.inner().bound_buffers().stream_freq(stream as usize);
    // SAFETY: `setting` is non-null (checked) and per the D3D9 ABI points to a
    // writable `u32` slot owned by the caller.
    unsafe { *setting = value };
    D3D_OK
}

extern "system" fn device_set_indices(this: *mut c_void, index_data: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    null_out(shader);
    if function.is_null() || shader.is_null() {
        return D3DERR_INVALIDCALL;
    }
    let Some(bytecode) = read_shader_bytecode(function) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: vtable thunk; this is a live Direct3DDevice9 per the D3D9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mut created = match create_native_shader(
        obj.inner(),
        &bytecode,
        mtld3d_shared::shader_create::ShaderStage::Pixel,
    ) {
        Ok(created) => created,
        Err(result) => return result,
    };
    let mut usage = crate::pixel_shader::PsUsage::empty();
    usage.set(
        crate::pixel_shader::PsUsage::AUTOMATIC_FOG,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::AUTOMATIC_FOG),
    );
    usage.set(
        crate::pixel_shader::PsUsage::USES_BUMP_ENV,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::BUMP_ENV),
    );
    usage.set(
        crate::pixel_shader::PsUsage::USES_INT_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::INT_CONST),
    );
    usage.set(
        crate::pixel_shader::PsUsage::USES_BOOL_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::BOOL_CONST),
    );
    usage.set(
        crate::pixel_shader::PsUsage::USES_REL_CONST,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::RELATIVE_CONST),
    );
    usage.set(
        crate::pixel_shader::PsUsage::READS_LINKED_INPUTS,
        created
            .usage
            .contains(mtld3d_shared::shader_create::ShaderUsage::LINKED_INPUTS),
    );
    if let Err(result) = created.adopt(obj.inner()) {
        return result;
    }
    let shader_obj = Direct3DPixelShader9::new(
        obj.inner_ptr(),
        created.shader_id,
        created.max_const_used,
        usage,
        created.color_out_mask,
        Arc::from(bytecode),
    );
    let shader_ptr = Box::into_raw(Box::new(shader_obj));
    // SAFETY: shader_ptr is a freshly created live shader at refcount one.
    unsafe { crate::com_ref::com_register_child(shader_ptr) };
    // SAFETY: the D3D9 ABI supplies a writable shader out pointer, checked non-null above.
    unsafe { OutPtr::write_opt(shader, shader_ptr.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn device_set_pixel_shader(this: *mut c_void, shader: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let automatic_fog = dev.uses_automatic_fog();
    let changed = dev.shader_bindings_mut().replace_pixel_shader(new);
    if changed {
        dev.mark_snapshot_dirty(SnapshotDirty::PS_SOURCE | SnapshotDirty::PS_CONST);
        if automatic_fog != dev.uses_automatic_fog() {
            // SM3 owns its fog. Rebinding a legacy shader restores both its
            // fog specialization and the uniform bytes suppressed for SM3.
            dev.mark_snapshot_dirty(SnapshotDirty::VARIANT | SnapshotDirty::FOG_COLOR);
        }
    }
    dev.perf_mut()
        .record_keys_gate(KeysGate::SetPixelShader, !changed);
    0
}

extern "system" fn device_get_pixel_shader(this: *mut c_void, shader: *mut *mut c_void) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    if !window_in_range(start_register, count, PS_FLOAT_CONSTANT_LIMIT) {
        return reject_constant_window(
            "SetPixelShaderConstantF",
            start_register,
            count,
            PS_FLOAT_CONSTANT_LIMIT,
        );
    }
    // A zero count names an empty window, which D3D9 accepts and writes nothing into.
    if count == 0 {
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` non-zero (both
    // settled above); per the D3D9 ABI the caller guarantees the bounded floats
    // are initialized and readable without mutation in one allocation for
    // this call. The byte size fits `isize::MAX`; all `f32` bits are valid.
    let constants = unsafe {
        mtld3d_shared::slice_from_caller(constant_data.cast::<[f32; 4]>(), count as usize)
    };
    let slice: &[[f32; 4]] = &constants;
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    if constant_data.is_null() {
        return D3DERR_INVALIDCALL;
    }
    if !window_in_range(start_register, count, PS_FLOAT_CONSTANT_LIMIT) {
        return reject_constant_window(
            "GetPixelShaderConstantF",
            start_register,
            count,
            PS_FLOAT_CONSTANT_LIMIT,
        );
    }
    // A zero count reads nothing, which D3D9 accepts.
    if count == 0 {
        return D3D_OK;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_copy();
    // SAFETY: `constant_data` is non-null and `count` non-zero (both settled above);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "SetPixelShaderConstantI",
        !constant_data.is_null(),
        start_register,
        count,
        INT_CONSTANT_ROWS,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` is bounded above;
    // per the D3D9 ABI, the bounded rows are initialized and readable
    // without mutation in one allocation for this call. The byte size
    // fits `isize::MAX`, and all `i32` bit patterns are valid.
    let constants =
        unsafe { mtld3d_shared::slice_from_caller(constant_data.cast::<[i32; 4]>(), count) };
    let slice: &[[i32; 4]] = &constants;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::PixelShaderConstantI {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    if dev
        .shader_bindings_mut()
        .write_ps_constants_i(start_register, slice)
    {
        dev.mark_snapshot_dirty(SnapshotDirty::PS_CONST_I);
    }
    0
}

extern "system" fn device_get_pixel_shader_constant_i(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "GetPixelShaderConstantI",
        !constant_data.is_null(),
        start_register,
        count,
        INT_CONSTANT_ROWS,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_i_copy();
    // SAFETY: `constant_data` is non-null and `count` is non-zero and clamped to the file (above);
    // per the D3D9 ABI the caller guarantees `count * 4` `i32`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data.cast::<[i32; 4]>(), count) };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_set_pixel_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *const i32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "SetPixelShaderConstantB",
        !constant_data.is_null(),
        start_register,
        count,
        BOOL_CONSTANT_COUNT,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();
    // SAFETY: `constant_data` is non-null and `count` is bounded above;
    // per the D3D9 ABI, the bounded BOOLs are initialized and readable
    // without mutation in one allocation for this call. The byte size
    // fits `isize::MAX`, and all `i32` bit patterns are valid.
    let constants = unsafe { mtld3d_shared::slice_from_caller(constant_data, count) };
    let slice = &*constants;
    if let Some(rec) = dev.recording_state_block_mut() {
        rec.record(StateOp::PixelShaderConstantB {
            start: start_register,
            values: slice.to_vec(),
        });
        return D3D_OK;
    }
    if dev
        .shader_bindings_mut()
        .write_ps_constants_b(start_register, slice)
    {
        dev.mark_snapshot_dirty(SnapshotDirty::PS_CONST_B);
    }
    0
}

extern "system" fn device_get_pixel_shader_constant_b(
    this: *mut c_void,
    start_register: u32,
    constant_data: *mut i32,
    count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::ShaderConst);
    let count = match int_bool_constant_rows(
        "GetPixelShaderConstantB",
        !constant_data.is_null(),
        start_register,
        count,
        BOOL_CONSTANT_COUNT,
    ) {
        Ok(0) => return D3D_OK,
        Ok(rows) => rows,
        Err(hr) => return hr,
    };
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let mirror = obj.inner().shader_bindings().ps_constants_b_copy();
    // SAFETY: `constant_data` is non-null and `count` is non-zero and clamped to the file (above);
    // per the D3D9 ABI the caller guarantees `count` `BOOL`s are writable.
    let out = unsafe { core::slice::from_raw_parts_mut(constant_data, count) };
    copy_constants_out(&mirror, start_register, out);
    0
}

extern "system" fn device_draw_rect_patch(
    this: *mut c_void,
    _handle: u32,
    _num_segs: *const f32,
    _tri_patch_info: *const c_void,
) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DDevice9::DrawTriPatch → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn device_delete_patch(this: *mut c_void, _handle: u32) -> i32 {
    let _api = device_api_lock(this);
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
    let _api = device_api_lock(this);
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
    // D3DUSAGE_DMAP marks a displacement map for the N-patch tessellator,
    // which no modern driver runs either; the texture is an ordinary texture.
    if usage & D3DUSAGE_DMAP != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "Create{kind}: D3DUSAGE_DMAP set but displacement mapping not implemented");
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
        | D3DUSAGE_NONSECURE
        | D3DUSAGE_DMAP;
    let unknown = usage & !known;
    if unknown != 0 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "Create{kind}: unknown usage bits {unknown:#x} — update warn_unused_usage_and_pool_once"
        );
    }

    // Pool. DEFAULT = 0.
    // D3DPOOL_MANAGED: behaviourally equivalent to DEFAULT in our
    // implementation. Metal's StorageModeShared/Managed textures are already
    // CPU+GPU accessible with OS-level paging, so "driver-managed residency"
    // is a non-concept. We also don't implement device loss (see
    // device_reset), so MANAGED's "survives reset" promise is vacuously true
    // for DEFAULT too. Accept and move on.
    // D3DPOOL_SYSTEMMEM / D3DPOOL_SCRATCH on a texture: a system-memory
    // resource, created with staging and no Metal texture
    // (`TextureFlags::CPU_ONLY`).
    // D3DPOOL_SYSTEMMEM on a vertex / index buffer: D3D9 draws from one
    // directly, so the buffer keeps its `MTLBuffer` and behaves like a
    // default-pool buffer. D3DPOOL_SCRATCH never reaches here for a buffer;
    // both Create* entry points reject it first.
    match pool {
        0 | D3DPOOL_MANAGED | D3DPOOL_SYSTEMMEM | D3DPOOL_SCRATCH => {}
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

pub fn warn_present_params_fields_once(pp: &mtld3d_types::D3DPRESENT_PARAMETERS) {
    if pp.back_buffer_count > 1 {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: back_buffer_count={} requested but only single back-buffer supported",
            pp.back_buffer_count
        );
    }
    if pp.swap_effect != 0 && pp.swap_effect != 1 {
        // 0 is invalid per spec; 1 = D3DSWAPEFFECT_DISCARD. FLIP and COPY
        // keep the one back buffer's contents across Present
        // (`BackbufferContents::Preserved`); FLIP's buffer rotation would need
        // a real swap chain instead of a single drawable.
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: swap_effect={} requested: the back buffer keeps its contents \
             across Present but is not rotated",
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
        // A fullscreen device takes a monitor-covering window rather than
        // setting a display mode, so there is no mode for a refresh rate to
        // select. Presentation pacing follows the panel, capped by
        // `present.maxFps`.
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "CreateDevice: full_screen_refresh_rate_in_hz={} requested but the display mode is \
             never changed — presentation follows the panel's own rate",
            pp.full_screen_refresh_rate_in_hz
        );
    }
    // presentation_interval is honoured at the AttachMetalLayer call site via
    // resolve_layer_pacing, which fires its own log_once_warn_by! for a value
    // that names no interval, so there is no separate arm here.
}

/// Map `D3DPRESENT_PARAMETERS::PresentationInterval` to the pacing the layer is handed.
///
/// `ONE` and `DEFAULT` ask for display-rate vsync and `IMMEDIATE` for none.
/// `TWO`, `THREE` and `FOUR` ask for vsync under a ceiling of the refresh
/// rate over two, three and four, folded with `present.maxFps` into the one
/// ceiling that crosses the PE/Unix boundary. The rate divided is the one
/// `GetDisplayMode` reports for `pp`, which is the one the application saw.
///
/// Pure mapping lives in `mtld3d_core::present`; this wrapper supplies the
/// reported mode and layers the project's logging policy on top so the
/// helper itself stays host-testable without pulling in `log` plumbing.
pub fn resolve_layer_pacing(
    pp: &mtld3d_types::D3DPRESENT_PARAMETERS,
    cfg: &Mtld3dConfig,
) -> LayerPacing {
    use mtld3d_core::present::{DisplaySync, display_sync_for, interval_divisor, layer_pacing_for};
    let interval = pp.presentation_interval;
    if matches!(display_sync_for(interval), DisplaySync::Fallthrough) {
        mtld3d_shared::log_once_warn_by!(
            target: crate::LOG_TARGET,
            key: u64::from(interval),
            "PresentationInterval={interval:#x} names no interval; falling through to display-rate vsync"
        );
    }
    // The reported mode of a windowed device is a Win32 query, so only an
    // interval that divides the rate asks for it.
    let refresh_hz = if interval_divisor(interval) > 1 {
        let refresh_hz = crate::direct3d9::reported_display_mode(pp).refresh_rate;
        if refresh_hz == 0 {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "PresentationInterval={interval:#x}: the reported display mode carries no refresh \
                 rate to divide; pacing at display-rate vsync"
            );
        }
        refresh_hz
    } else {
        0
    };
    layer_pacing_for(interval, refresh_hz, cfg.present_max_fps)
}
