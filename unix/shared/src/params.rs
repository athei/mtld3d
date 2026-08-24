use super::{
    Thunk, Thunks,
    mtl::{
        AddressMode, BlendFactor, BlendOperation, BufferKind, ClearQuadFlags, ColorSpacePolicy,
        ColorWriteMask, CompareFunc, DestroyKind, LoadAction, MinMagFilter, MipFilter, PixelFormat,
        StageTag, StencilOp, StorageMode, StoreAction, Swizzle, TextureUsage, VertexFormat,
        VertexStepFunction,
    },
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLCommandQueueKind, MTLDepthStencilStateKind,
        MTLDeviceKind, MTLFunctionKind, MTLLibraryKind, MTLRenderPipelineStateKind,
        MTLSamplerStateKind, MTLTextureKind, MetalHandle, NSViewKind,
    },
};

// ── Wire-layout guards ──
//
// Checked at compile time on EVERY target this crate is built for — the two
// PE arches (i686 + x86_64 `*-pc-windows-msvc`) AND both unix `.so` arches
// (x86_64 and aarch64 Apple, which share these `repr(C)` layouts). The
// whole PE↔unix thunk protocol assumes a `repr(C)` `u64` is 8-byte aligned on
// all of them; if a 32-bit target ever aligned `u64` to 4, every struct with a
// `u64` after an odd run of 4-byte fields would shift and the unix handler
// would write out-params past the PE caller's (often stack-allocated) struct —
// smashing the PE return address. This is the wow64-divergence the host-only
// `#[test]` size checks could never catch. The self-contained probe proves the
// alignment property; the per-struct asserts pin the device-lifecycle layouts.
const _: () = {
    #[repr(C)]
    struct U64After4 {
        a: u32,
        b: u64,
    }
    // 8 (not 4) ⇒ repr(C) u64 is 8-aligned on this target.
    assert!(core::mem::offset_of!(U64After4, b) == 8);
    // A real wire struct whose `u64 id` sits after device_handle(8) + three
    // 4-byte fields: offset 24 ⇒ u64 8-aligned; would be 20 if 4-aligned.
    assert!(core::mem::size_of::<StencilFaceParams>() == 16);
    assert!(core::mem::offset_of!(CreateDepthStencilStateParams, front) == 24);
    assert!(core::mem::offset_of!(CreateDepthStencilStateParams, id) == 64);
    assert!(core::mem::size_of::<CreateDepthStencilStateParams>() == 80);

    // Device create / render / destroy structs: align must be 8 and size
    // identical on all targets.
    assert!(core::mem::align_of::<CreateCommandQueueParams>() == 8);
    assert!(core::mem::size_of::<CreateCommandQueueParams>() == 24);
    assert!(core::mem::size_of::<AttachMetalLayerParams>() == 64);
    assert!(core::mem::size_of::<CreateBackbufferParams>() == 32);
    assert!(core::mem::size_of::<DestroyCommandQueueParams>() == 48);
    assert!(core::mem::size_of::<SubmitFrameParams>() == 96);
    assert!(core::mem::size_of::<PassDescriptor>() == 168);
};

/// One-shot "register `env_logger` on the unix side" thunk.
///
/// Fired once from d3d9.dll's `init_logger` on DLL load, before any other
/// thunk that might want to log. No payload — the `reserved` field keeps the
/// struct non-zero-sized so the pointer handed across the boundary is
/// distinct.
#[repr(C, align(8))]
pub struct InitLoggerParams {
    // Keeps the struct non-zero-sized so the pointer handed across the
    // PE/Unix boundary is distinct. Constructed by name across crates,
    // hence pub.
    pub reserved: u64,
}

impl Thunk for InitLoggerParams {
    const CODE: u32 = Thunks::InitLogger as u32;
}

/// One formatted log line from the PE-side logger, for the unix stderr.
///
/// `ptr`/`len` describe a byte slice the PE side keeps alive for the call.
#[repr(C, align(8))]
pub struct WriteLogParams {
    pub ptr: u64, // in: *const u8
    pub len: u32, // in: byte count
    pub pad0: u32,
}

impl Thunk for WriteLogParams {
    const CODE: u32 = Thunks::WriteLog as u32;
}

#[repr(C, align(8))]
pub struct GetDeviceInfoParams {
    pub name_ptr: u64,
    pub name_buf_len: u64,
    pub name_len: u64,    // out
    pub registry_id: u64, // out
}

impl Thunk for GetDeviceInfoParams {
    const CODE: u32 = Thunks::GetDeviceInfo as u32;
}

#[repr(C, align(8))]
pub struct CreateCommandQueueParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // out
    pub queue_handle: MetalHandle<MTLCommandQueueKind>, // out
    /// 0 / non-zero boolean: `MTLDevice.hasUnifiedMemory`.
    ///
    /// False on Intel/AMD non-UMA Macs; the storage-mode policy in
    /// `mtld3d-core::storage_policy` switches CPU-visible buffers to
    /// `Managed` and the encoder enqueues `didModifyRange:` calls when
    /// this is 0. (Textures are always `Private`.)
    pub unified_memory: u32, // out
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`.
    ///
    /// 16 on Apple Silicon, 256 on AMD/Intel (Mac2). Threaded into
    /// `pad_source_stride` so blit-staging `bytes_per_row` rounds to this
    /// floor.
    pub min_linear_texture_align: u32, // out
}

impl Thunk for CreateCommandQueueParams {
    const CODE: u32 = Thunks::CreateCommandQueue as u32;
}

#[repr(C, align(8))]
pub struct AttachMetalLayerParams {
    pub hwnd: u64,                                   // in
    pub device_handle: MetalHandle<MTLDeviceKind>,   // in: from CreateCommandQueue
    pub width: u32,                                  // in: backbuffer width
    pub height: u32,                                 // in: backbuffer height
    pub view_handle: MetalHandle<NSViewKind>,        // out: macdrv_metal_view (for cleanup)
    pub layer_handle: MetalHandle<CAMetalLayerKind>, // out
    /// `NSWindow.backingScaleFactor` for the attached window.
    ///
    /// Rounded to an integer and clamped to `[1, 8]`. Consumed by the
    /// PE-side cursor upscaler so a retina display gets a
    /// proportionally-sized HCURSOR bitmap (Wine's Win32 cursor path
    /// doesn't participate in the OS's retina upscale).
    pub backing_scale: u32, // out
    /// The vsync request from `D3DPRESENT_PARAMETERS::PresentationInterval`.
    ///
    /// Mapped through `mtld3d_core::present::display_sync_for` on the PE
    /// side: 0 = vsync off (CAMetalLayer.displaySyncEnabled = false),
    /// non-zero = on.
    pub display_sync_enabled: u32, // in
    /// `color.hdr.enable` from `mtld3d.conf`.
    ///
    /// Non-zero = allow the HDR present pipeline when the display also has
    /// EDR headroom, zero = force the SDR path. Resolved PE-side from
    /// `CONFIG.hdr_enable`; unix side feeds it to `resolve_hdr_active`.
    pub hdr_enable: u32, // in
    /// `color.space` from `mtld3d.conf`.
    ///
    /// `Passthrough` (the default, today's behaviour) tags the layer with
    /// the display's own `CGColorSpace` — D3D9's untagged values land at
    /// the panel's native primaries. `Accurate` overrides that with the
    /// sRGB family for both SDR and HDR paths so guest art reads with its
    /// designer-intended hues. PE side reads this from
    /// `CONFIG.color_space`.
    pub color_space: ColorSpacePolicy, // in
    /// `present.maxFps` from `mtld3d.conf`: frame-rate ceiling in Hz, `0` = uncapped.
    ///
    /// Combined with the vsync request into the present-throttle
    /// duration — the lower rate wins. PE side reads this from
    /// `CONFIG.present_max_fps`.
    pub max_fps: u32, // in
    /// Whether this GPU can run a `MetalFX` spatial upscale.
    ///
    /// Non-zero lets the PE side size the drawable to the window and leave
    /// the resample to `MetalFX`. Zero means the drawable must keep the back
    /// buffer's size so present stays a 1:1 copy and Core Animation scales
    /// the layer instead — the pre-`MetalFX` behaviour, and the only correct
    /// fallback since nothing else on the unix side can resize a frame.
    pub metalfx_available: u32, // out
}

impl Thunk for AttachMetalLayerParams {
    const CODE: u32 = Thunks::AttachMetalLayer as u32;
}

/// Update `CAMetalLayer.displaySyncEnabled` on an already-attached layer.
///
/// Used by the D3D9 Reset path to honour a runtime change of
/// `D3DPRESENT_PARAMETERS::PresentationInterval`.
#[repr(C, align(8))]
pub struct SetDisplaySyncEnabledParams {
    pub layer_handle: MetalHandle<CAMetalLayerKind>, // in
    pub display_sync_enabled: u32,                   // in: 0 = off, !=0 = on
    /// `present.maxFps` from `mtld3d.conf`: frame-rate ceiling in Hz, `0` = uncapped.
    ///
    /// Re-sent on every Reset so the throttle recomputation keeps
    /// honouring the cap.
    pub max_fps: u32, // in
}

impl Thunk for SetDisplaySyncEnabledParams {
    const CODE: u32 = Thunks::SetDisplaySyncEnabled as u32;
}

/// Block the caller until the GPU has retired the cmdbuf with `submit_seq >= target_seq`.
///
/// Then bump `coherent_seq` so subsequent `Acquire` loaders observe the
/// advance synchronously.
///
/// Used by `wait_for_gpu_idle` (Reset / OOM recovery / shutdown) and by the
/// occlusion-query FLUSH path to convert spin loops into a kernel sleep on
/// Metal's `MTLCommandBuffer::waitUntilCompleted`.
#[repr(C, align(8))]
pub struct WaitForGpuRetireParams {
    pub target_seq: u64,       // in
    pub coherent_seq_ptr: u64, // in: PE-side AtomicU64 backing
}

impl Thunk for WaitForGpuRetireParams {
    const CODE: u32 = Thunks::WaitForGpuRetire as u32;
}

/// Begin a Metal GPU frame capture writing a `.gputrace` document to disk.
///
/// The path is hard-coded to `/tmp/mtld3d_capture.gputrace`.
///
/// Apple requires the process to have launched with `MTL_CAPTURE_ENABLED=1`;
/// without it the unix-side handler logs a warn and returns without
/// capturing. Triggered from the encoder thread when the API thread sets the
/// `CAPTURE_REQUESTED` flag (F12 hotkey in `device_present`).
#[repr(C, align(8))]
pub struct StartGpuCaptureParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in: capture-object
}

impl Thunk for StartGpuCaptureParams {
    const CODE: u32 = Thunks::StartGpuCapture as u32;
}

/// End the in-progress Metal GPU frame capture.
///
/// Idempotent on the unix side (no-op if no capture was started).
#[repr(C, align(8))]
pub struct StopGpuCaptureParams {
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u64,
}

impl Thunk for StopGpuCaptureParams {
    const CODE: u32 = Thunks::StopGpuCapture as u32;
}

/// Read the process-wide page-fault counters via `getrusage(RUSAGE_SELF)`.
///
/// Sampled by the encoder thread once per perf summary window (PERF=1
/// builds only) so the summary can report a fault-rate delta. Minor
/// faults are the first-touch zero-fill signal the `PageBox` churn
/// investigation watches; major faults ride along for free. Process-wide:
/// every thread's faults land in the same counters.
#[repr(C, align(8))]
pub struct GetTaskFaultsParams {
    pub minor_faults: u64, // out: cumulative ru_minflt since process start
    pub major_faults: u64, // out: cumulative ru_majflt since process start
}

impl Thunk for GetTaskFaultsParams {
    const CODE: u32 = Thunks::GetTaskFaults as u32;
}

#[repr(C, align(8))]
pub struct DestroyCommandQueueParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub queue_handle: MetalHandle<MTLCommandQueueKind>, // in
    pub view_handle: MetalHandle<NSViewKind>,      // in (NULL = none)
    pub backbuffer_handle: MetalHandle<MTLTextureKind>, // in (NULL = none)
    pub pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>, // in (NULL = none)
    pub depth_texture_handle: MetalHandle<MTLTextureKind>, // in (NULL = none)
}

impl Thunk for DestroyCommandQueueParams {
    const CODE: u32 = Thunks::DestroyCommandQueue as u32;
}

#[repr(C, align(8))]
pub struct CreateBackbufferParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    /// Frame queue the creation-time clear is encoded on.
    ///
    /// A new `MTLTexture` has undefined contents, and the back buffer is
    /// presentable before the application's first draw or clear reaches it.
    /// Encoding the clear on the frame queue makes commit order the fence:
    /// every later frame command buffer observes a black back buffer.
    pub queue_handle: MetalHandle<MTLCommandQueueKind>, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateBackbufferParams {
    const CODE: u32 = Thunks::CreateBackbuffer as u32;
}

/// Vertex attribute descriptor, one per Metal vertex input attribute.
///
/// Packed as an array pointed to by `CreateRenderPipelineParams::vertex_attrs_ptr`.
#[repr(C, align(4))]
#[derive(Clone, Copy)]
pub struct VertexAttrDesc {
    pub attr_index: u32,      // in: Metal attribute slot ([[attribute(N)]])
    pub buffer_index: u32,    // in: Metal buffer slot
    pub offset: u32,          // in: byte offset within the buffer
    pub format: VertexFormat, // in
}

/// One vertex buffer layout of a render pipeline: a D3D9 stream the draw reads.
///
/// Packed as an array pointed to by `CreateRenderPipelineParams::vertex_layouts_ptr`,
/// one entry per stream that contributes an attribute. `stride` is never 0
/// (Metal rejects it for every step function). `step_rate` is the instances
/// per advance for `PerInstance`, 1 for `PerVertex`, 0 for `Constant`.
#[repr(C, align(4))]
pub struct VertexBufferLayoutDesc {
    pub buffer_index: u32, // in: Metal vertex buffer slot (= D3D9 stream)
    pub stride: u32,       // in: bytes per step
    pub step_function: VertexStepFunction, // in
    pub step_rate: u32,    // in
}

#[repr(C, align(8))]
pub struct CreateRenderPipelineParams {
    pub device_handle: MetalHandle<MTLDeviceKind>,  // in
    pub vs_fn_handle: MetalHandle<MTLFunctionKind>, // in
    pub ps_fn_handle: MetalHandle<MTLFunctionKind>, // in
    pub vertex_attrs_ptr: u64,                      // in: *const VertexAttrDesc
    pub vertex_layouts_ptr: u64,                    // in: *const VertexBufferLayoutDesc
    pub vertex_attr_count: u32,                     // in
    pub vertex_layout_count: u32,                   // in
    pub blend_enable: u32,                          // in: non-zero = enabled
    pub src_blend: BlendFactor,                     // in: source RGB
    pub dst_blend: BlendFactor,                     // in: dest RGB
    pub blend_op: BlendOperation,                   // in: RGB blend op (D3DRS_BLENDOP)
    pub src_blend_alpha: BlendFactor, // in: source alpha (only if separate_alpha_blend_enable)
    pub dst_blend_alpha: BlendFactor, // in: dest alpha (only if separate_alpha_blend_enable)
    pub blend_op_alpha: BlendOperation, // in: alpha blend op (D3DRS_BLENDOPALPHA)
    pub separate_alpha_blend_enable: u32, // in: non-zero = use *_alpha fields; else mirror RGB
    pub srgb_write_enable: u32,       // in: non-zero = upgrade color_format to its sRGB twin
    pub color_write_mask: ColorWriteMask, // in
    pub has_depth: u32,               // in: non-zero = pipeline declares depth attachment
    pub has_stencil: u32, // in: non-zero = depth attachment format carries stencil (D24S8/D24FS8)
    pub color_format: PixelFormat, // in: colorAttachments[0]
    pub has_color_output: u32, /* in: non-zero = pipeline declares a color attachment; zero =
                           * no color attachment, descriptor leaves colorAttachments[0]
                           * default (pixelFormat=Invalid). Set zero by the pass-state
                           * machine for cascade caster passes where every draw has
                           * color_write_mask=0 (eliminates Apple "Unused Texture"). */
    pub extra_present_mask: u32, // in: bit i = colorAttachments[i + 1] is declared
    pub extra: [ExtraColorAttachmentParams; 3], // in: colorAttachments[1..=3]
    pub pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>, // out
}

/// One extra colour attachment (`colorAttachments[1..=3]`) of a render pipeline.
///
/// The blend operations are shared with attachment 0 (D3D9 has one blend
/// state); the factors are resolved per attachment because the
/// destination-alpha clamp depends on whether that target's D3D format has an
/// alpha channel. Ignored unless the matching `extra_present_mask` bit is set.
#[repr(C)]
pub struct ExtraColorAttachmentParams {
    pub format: PixelFormat,          // in
    pub write_mask: ColorWriteMask,   // in
    pub src_blend: BlendFactor,       // in: source RGB
    pub dst_blend: BlendFactor,       // in: dest RGB
    pub src_blend_alpha: BlendFactor, // in: source alpha (already effective)
    pub dst_blend_alpha: BlendFactor, // in: dest alpha (already effective)
}

impl Thunk for CreateRenderPipelineParams {
    const CODE: u32 = Thunks::CreateRenderPipeline as u32;
}

/// Lazy create-or-fetch of the per-format-combo "clear-quad" pipeline.
///
/// Used to honour D3D9's viewport-clipped mid-pass Clear semantics on
/// Metal — instead of ending the encoder and starting a new one with
/// `loadAction = Clear` (which clears the full attachment and wipes
/// prior in-pass draws), the PE side binds this pipeline, sets scissor
/// to the viewport, pushes the clear value via `setVertexBytes`, and
/// draws a single fullscreen triangle that writes the constant depth
/// (or color) only inside the scissor rect.
///
/// One pipeline per `(depth_format, color_format, flags)` combo (where
/// `flags` carries `HAS_COLOR` / `HAS_DEPTH` / `HAS_STENCIL`), cached
/// unix-side for process lifetime in a
/// `HashMap<key, MTLRenderPipelineState*>`. A workload whose
/// cascade-depth tile atlases all share one combo (`Depth32Float`,
/// no color) caps the cache at a single entry.
///
/// The same VS / PS pair handles both depth-only and depth+color clears
/// via the `HAS_COLOR` flag. The pipeline's depth-write side is gated by
/// the depth-stencil state the PE emits separately
/// (`get_or_create_depth_stencil(1, 1, ALWAYS)`); the color side is gated
/// by `HAS_COLOR` and the matching `color_format`.
#[repr(C, align(8))]
pub struct EnsureClearQuadPipelineParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub depth_format: PixelFormat,                 // in (ignored when HAS_DEPTH unset)
    pub color_format: PixelFormat,                 // in (ignored when HAS_COLOR unset)
    pub flags: ClearQuadFlags,                     // in: HAS_COLOR | HAS_DEPTH | HAS_STENCIL
    /// Render targets 1..3 of the pass the quad draws into: bit `i` = slot `i + 1` bound.
    ///
    /// A colour quad writes its clear colour to every declared slot; a
    /// depth quad declares them with an empty write mask, as it does for
    /// slot 0 under `COLOR_FORMAT_NO_WRITE`. Zero on a single-target pass.
    pub extra_present_mask: u32, // in
    pub extra_formats: [PixelFormat; 3],           // in (ignored where the mask bit is clear)
    pub pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>, // out
}

impl Thunk for EnsureClearQuadPipelineParams {
    const CODE: u32 = Thunks::EnsureClearQuadPipeline as u32;
}

/// Lazy create-or-fetch of the per-destination-format "blit" pipeline.
///
/// Used by a *scaling* `IDirect3DDevice9::StretchRect`.
///
/// Metal's `MTLBlitCommandEncoder` can only do 1:1 copies, so a `StretchRect`
/// whose source and destination rects differ in size is translated into a
/// render pass that samples the source texture onto a fullscreen-NDC quad
/// covering the destination rect (the PE side sets viewport + scissor to the
/// destination rect; the source rect is mapped to `[0,1]` texcoords via a
/// `setVertexBytes` transform). This pipeline is the VS/PS pair for that quad.
///
/// One pipeline per destination `color_format` (the source is bound as a
/// fragment texture, not declared in the pipeline), cached unix-side for
/// process lifetime in a `HashMap<color_format, MTLRenderPipelineState*>`.
/// Mirrors `EnsureClearQuadPipelineParams`. No depth attachment: the blit
/// quad never writes depth, and the PE side opens the destination pass with
/// `SetDepthStencilSurface(NULL)` so no depth format is declared.
#[repr(C, align(8))]
pub struct EnsureBlitPipelineParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub color_format: PixelFormat,                 // in: destination colour format
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32, // align next field to 8
    pub pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>, // out
}

impl Thunk for EnsureBlitPipelineParams {
    const CODE: u32 = Thunks::EnsureBlitPipeline as u32;
}

/// Compile one stage's MSL source into an `MTLLibrary` and resolve its single entry point.
///
/// A pipeline can mix an `MTLFunction` from a VS library with one from a PS
/// library — Metal links by stage-in/stage-out layout at pipeline creation.
///
/// `entry_ptr` / `entry_len` are the UTF-8 entry-point name to look up via
/// `newFunctionWithName:`; the same string must appear in the function
/// definition inside the MSL at `msl_ptr`. Per-shader-id names
/// (`mtld3d_vs_ff_5f3a0001`, `mtld3d_ps_sm3_a2b1c4d8`, …) make Xcode's
/// pipeline-state inspector show distinct labels per shader rather than
/// collapsing every pipeline to "`mtld3d_vs`" / "`mtld3d_ps`".
#[repr(C, align(8))]
pub struct CompileShaderLibraryParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub msl_ptr: u64,                              // in: *const u8 (UTF-8 MSL source)
    pub msl_len: u32,                              // in: byte length
    pub stage_tag: StageTag,                       // in
    pub entry_ptr: u64,                            // in: *const u8 (UTF-8 entry-point name)
    pub entry_len: u32,                            // in: byte length
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,                                   // align next u64
    pub library_handle: MetalHandle<MTLLibraryKind>, // out
    pub fn_handle: MetalHandle<MTLFunctionKind>,     // out
}

impl Thunk for CompileShaderLibraryParams {
    const CODE: u32 = Thunks::CompileShaderLibrary as u32;
}

/// One render pass inside a `SubmitFrame` submission.
///
/// Carries the attachments plus load actions for the Metal render pass
/// descriptor and the slice of commands to replay inside it. An array of
/// these describes the full frame: the unix side creates one
/// `MTLRenderCommandEncoder` per pass and replays
/// `commands_ptr[0..command_count]` between `begin` and `endEncoding`.
///
/// `leading_blits_ptr` / `leading_blits_count` describe blits that run
/// inside an `MTLBlitCommandEncoder` *before* this pass's render
/// encoder. Used by `StretchRect` (texture-to-texture copy) so a blit
/// that lands between two D3D9 draws is ordered against both the source
/// pass's draws and the next pass's draws — the global
/// `SubmitFrameParams.blit_commands_ptr` runs at frame start and would
/// mis-order a mid-frame blit. A pass with `color_texture == 0` and
/// `command_count == 0` is a "blit-only" trailing pass synthesised when
/// `StretchRect` lands after the last draw of the frame.
///
/// Fields are ordered u64s-first then u32s so the natural struct layout
/// is padding-free; size is 168 bytes on both 32- and 64-bit PE.
#[repr(C, align(8))]
pub struct PassDescriptor {
    pub color_texture: MetalHandle<MTLTextureKind>, // in
    pub depth_texture: MetalHandle<MTLTextureKind>, // in (NULL = none)
    pub commands_ptr: u64,                          // in: *const Command
    pub visibility_result_buffer: MetalHandle<MTLBufferKind>, // in (NULL = no visibility tracking)
    pub leading_blits_ptr: u64,                     // in: *const BlitCommand (0 = none)
    pub color_load_action: LoadAction,              // in
    pub color_store_action: StoreAction,            // in
    pub clear_r: u32,                               // in: f32 bits
    pub clear_g: u32,                               // in: f32 bits
    pub clear_b: u32,                               // in: f32 bits
    pub clear_a: u32,                               // in: f32 bits
    pub depth_load_action: LoadAction,              // in
    /// Applies to both the depth attachment and the stencil attachment.
    ///
    /// The stencil half is live only when the depth texture is
    /// `Depth32Float_Stencil8`, since mtld3d uses the combined format.
    /// The unix side mirrors this value to both `setStoreAction:` calls.
    pub depth_store_action: StoreAction, // in
    pub depth_clear_value: u32,                     // in: f32 bits (default 1.0)
    /// Load action for the stencil half of a combined depth/stencil texture.
    ///
    /// Independent of `depth_load_action` because D3D9 clears the two planes
    /// separately: `Clear(D3DCLEAR_STENCIL)` without `D3DCLEAR_ZBUFFER` has to
    /// reset stencil while carrying depth forward. Ignored when the depth
    /// texture's format has no stencil plane.
    pub stencil_load_action: LoadAction, // in
    pub stencil_clear_value: u32,                   // in: 0..=255
    pub command_count: u32,                         // in
    pub leading_blits_count: u32,                   // in
    /// Leading-blit and color-subresource flags.
    ///
    /// Bit 0 is whether the leading-blit list needs an encoder. Bits 1..3
    /// carry the color attachment slice, and bits 4..7 carry its mip level.
    /// Ordinary 2D level-zero passes therefore retain their previous 0/1
    /// value and the descriptor keeps its size.
    pub pass_flags: u32, // in
    /// Render targets 1..3 (`colorAttachments[1..=3]`); `texture` null = unbound.
    ///
    /// They share `clear_r..clear_a` with attachment 0 (a D3D9 `Clear` has
    /// one colour for every target) and carry their own load/store actions.
    pub extra_color: [ExtraColorDesc; 3], // in
}

impl PassDescriptor {
    const LEADING_BLITS_NEED_ENCODER: u32 = 1;
    const COLOR_SLICE_SHIFT: u32 = 1;
    const COLOR_LEVEL_SHIFT: u32 = 4;

    /// Pack leading-blit and color-subresource state.
    #[must_use]
    pub const fn pack_flags(needs_encoder: bool, color_slice: u32, color_level: u32) -> u32 {
        (if needs_encoder { 1 } else { 0 })
            | ((color_slice & 0x7) << Self::COLOR_SLICE_SHIFT)
            | ((color_level & 0xf) << Self::COLOR_LEVEL_SHIFT)
    }

    /// Whether the leading-blit list contains an encoder-bound command.
    #[must_use]
    pub const fn leading_blits_need_encoder(&self) -> bool {
        self.pass_flags & Self::LEADING_BLITS_NEED_ENCODER != 0
    }

    /// Color attachment array slice.
    #[must_use]
    pub const fn color_slice(&self) -> u32 {
        (self.pass_flags >> Self::COLOR_SLICE_SHIFT) & 0x7
    }

    /// Color attachment mip level.
    #[must_use]
    pub const fn color_level(&self) -> u32 {
        (self.pass_flags >> Self::COLOR_LEVEL_SHIFT) & 0xf
    }
}

/// One of render targets 1..3 on a [`PassDescriptor`].
///
/// `subresource` packs the array slice in bits 0..7 and the mip level in
/// bits 8..15, the same packing the PE side keeps for attachment 0 before it
/// folds it into `pass_flags`. 24 bytes, 8-aligned through `texture`.
#[repr(C)]
pub struct ExtraColorDesc {
    pub texture: MetalHandle<MTLTextureKind>, // in (NULL = unbound)
    pub subresource: u32,                     // in: slice | (level << 8)
    pub load_action: LoadAction,              // in
    pub store_action: StoreAction,            // in
    pub reserved: u32,
}

impl ExtraColorDesc {
    /// The unbound attachment.
    pub const NONE: Self = Self {
        texture: MetalHandle::NULL,
        subresource: 0,
        load_action: LoadAction::DontCare,
        store_action: StoreAction::DontCare,
        reserved: 0,
    };

    #[must_use]
    pub const fn is_bound(&self) -> bool {
        !self.texture.is_null()
    }

    /// Array slice of the attachment.
    #[must_use]
    pub const fn slice(&self) -> u32 {
        self.subresource & 0xff
    }

    /// Mip level of the attachment.
    #[must_use]
    pub const fn level(&self) -> u32 {
        self.subresource >> 8
    }
}

/// Self-contained frame submission.
///
/// Carries one or more render passes plus the optional present blit, so
/// `SetRenderTarget` / mid-frame `Clear` / depth-stencil changes can break
/// the flat command stream into separate Metal encoders.
#[repr(C, align(8))]
pub struct SubmitFrameParams {
    pub queue_handle: MetalHandle<MTLCommandQueueKind>, // in
    // Leading blit pass. Replayed inside a single
    // `MTLBlitCommandEncoder` before any render pass. 0-count =
    // skip.
    pub blit_commands_ptr: u64,  // in: *const BlitCommand
    pub blit_command_count: u32, // in
    /// 0 / non-zero: same gate as `PassDescriptor::leading_blits_need_encoder`.
    ///
    /// Applied to the frame-leading blit list.
    pub blit_commands_need_encoder: u32, // in
    // Render pass list
    pub passes_ptr: u64, // in: *const PassDescriptor
    pub pass_count: u32, // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad1: u32,
    // Present (NULL = skip)
    pub present_layer: MetalHandle<CAMetalLayerKind>, // in (NULL = no present)
    pub present_texture: MetalHandle<MTLTextureKind>, // in: blit to drawable
    // Submit-seq fencing. The submit `addCompletedHandler` block
    // `fetch_max`es `submit_seq` into `*(coherent_seq_ptr as
    // *const AtomicU64)` with Release ordering once the frame retires
    // on the GPU, so the PE-side texture + VB/IB retention drains can
    // release backings / MTLBuffers. `coherent_seq_ptr` is 0 on the
    // very first submit (no previous frame).
    pub submit_seq: u64,       // in
    pub coherent_seq_ptr: u64, // in: *const AtomicU64 (PE heap, stable)
    /// Texture-upload completion fence.
    ///
    /// When non-zero, the texture-upload (frame-leading) blits are encoded
    /// into their OWN command buffer committed *before* the draw CB; that
    /// CB's `addCompletedHandler` `fetch_max`es `submit_seq` into
    /// `*(upload_coherent_seq_ptr as *const AtomicU64)`. Because the queue
    /// is in-order the uploads still finish before any same-frame draw
    /// samples them, but this CB retires ~a frame earlier than the draw
    /// CB — so the next frame's texture `LockRect` sees the staging retired
    /// and skips the synchronous preserve memcpy. Every submitted frame
    /// carries the real pointer; 0 (a defensive null guard) falls back to
    /// encoding the leading blits on the draw CB. Distinct from
    /// `coherent_seq_ptr`, which tracks full-frame (draw) retirement for
    /// VB/IB.
    pub upload_coherent_seq_ptr: u64, // in: *const AtomicU64 (PE heap, stable)
    /// Nanoseconds spent in `nextDrawable()`, 0 outside a `PERF=1` build.
    ///
    /// Nanoseconds, not cycles, because this is the one duration that crosses
    /// the boundary: each side calibrates its own counter, and an arm64 `.so`
    /// reads `CNTVCT_EL0` while the PE side reads an emulated `rdtsc` at a
    /// different rate, so a raw cycle count would be scaled by the wrong Hz on
    /// arrival. Measured by `perf::NanosSetTimer`; the PE side converts it into
    /// its own cycles with `tsc::ns_to_cycles` before folding it into perf.
    pub drawable_wait_ns: u64, // out
    /// `NSView*` the layer was attached to.
    ///
    /// `submit_frame` walks `view → window → screen` each present to read
    /// the screen's *dynamic*
    /// `maximumExtendedDynamicRangeColorComponentValue` (the BT.2446 target
    /// peak each frame). NULL if no layer was attached (no present this
    /// frame). Used only on the HDR branch, gated unix-side by
    /// `HDR_BOOTSTRAP_PEAK_BITS > 1.0`.
    pub present_view: MetalHandle<NSViewKind>, // in
}

impl Thunk for SubmitFrameParams {
    const CODE: u32 = Thunks::SubmitFrame as u32;
}

#[repr(C, align(8))]
pub struct CreateDepthTextureParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub pixel_format: PixelFormat, // in (resolved via mtld3d_core::format::map_d3d_depth_format)
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateDepthTextureParams {
    const CODE: u32 = Thunks::CreateDepthTexture as u32;
}

#[repr(C, align(8))]
pub struct CreateColorTargetParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub pixel_format: PixelFormat, // in (resolved via mtld3d_core::format::map_d3d_format)
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateColorTargetParams {
    const CODE: u32 = Thunks::CreateColorTarget as u32;
}

#[repr(C, align(8))]
pub struct CreateDepthStencilStateParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub depth_test_enable: u32,                    // in: non-zero = enabled
    pub depth_write_enable: u32,                   // in: non-zero = enabled
    pub depth_compare_func: CompareFunc,           // in
    pub stencil_test_enable: u32,                  // in: non-zero = enabled
    pub front: StencilFaceParams,                  // in
    pub back: StencilFaceParams,                   // in
    pub stencil_read_mask: u32,                    // in
    pub stencil_write_mask: u32,                   // in
    pub id: u64,                                   // in: caller-defined label tag
    pub state_handle: MetalHandle<MTLDepthStencilStateKind>, // out
}

/// One face of the stencil test, mirroring `MTLStencilDescriptor`.
///
/// Embedded twice in `CreateDepthStencilStateParams`. D3D9 addresses the back
/// face through the separate `D3DRS_CCW_STENCIL*` states, which apply only
/// while `D3DRS_TWOSIDEDSTENCILMODE` is set; the PE side resolves that and
/// sends both faces populated either way.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFaceParams {
    pub compare_func: CompareFunc,
    pub stencil_fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

impl Thunk for CreateDepthStencilStateParams {
    const CODE: u32 = Thunks::CreateDepthStencilState as u32;
}

/// Per-element descriptor inside `CreateTexturesBatchParams::descs_ptr`.
///
/// One entry per `MTLTexture` to create. The unix side iterates the slice
/// and writes each resulting handle into the matching slot of
/// `handles_out_ptr`. No `device_handle` or output handle field here — both
/// live on the batch struct.
#[repr(C, align(8))]
pub struct TextureCreateDesc {
    pub tex_id: u64,               // in: mtld3d TextureId for Xcode capture labeling
    pub width: u32,                // in
    pub height: u32,               // in
    pub depth: u32,                // in: 1 for 2D textures, >1 → MTLTextureType3D (volume)
    pub levels: u32,               // in: mip level count
    pub pixel_format: PixelFormat, // in
    pub storage_mode: StorageMode, // in
    pub flags: crate::mtl::TextureCreateFlags, // in: swizzle and texture shape
    pub swizzle_r: Swizzle,        // in: R channel
    pub swizzle_g: Swizzle,        // in: G channel
    pub swizzle_b: Swizzle,        // in: B channel
    pub swizzle_a: Swizzle,        // in: A channel
    pub usage_flags: TextureUsage, // in
}

/// Batched `MTLTexture` create.
///
/// One PE↔Unix crossing creates `count` textures from the descriptor
/// array. `handles_out_ptr` points at a caller-owned `[u64; count]` buffer;
/// each slot receives the resulting `MTLTexture*` (zero on per-element
/// failure). Both arrays must be 8-byte aligned and stable for the
/// duration of the call — the unix side dereferences these pointers,
/// so the backing storage cannot move until `unix_call` returns.
///
/// Single-create call sites use `count = 1` against a one-element array.
#[repr(C, align(8))]
pub struct CreateTexturesBatchParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub count: u32,                                // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub descs_ptr: u64,       // in: *const TextureCreateDesc, len=count
    pub handles_out_ptr: u64, // out: *mut MetalHandle<MTLTextureKind>, len=count (NULL on failure)
}

impl Thunk for CreateTexturesBatchParams {
    const CODE: u32 = Thunks::CreateTexturesBatch as u32;
}

#[repr(C, align(8))]
pub struct CreateSamplerStateParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub id: u64,                                   // in: caller-defined label tag
    pub min_filter: MinMagFilter,                  // in
    pub mag_filter: MinMagFilter,                  // in
    pub mip_filter: MipFilter,                     // in
    pub address_u: AddressMode,                    // in
    pub address_v: AddressMode,                    // in
    pub address_w: AddressMode,                    // in
    pub max_anisotropy: u32,                       // in
    /// `D3DSAMP_MAXMIPLEVEL` → Metal's `setLodMinClamp`.
    ///
    /// D3D9's MAXMIPLEVEL is confusingly the *minimum* fine mip index the
    /// sampler may select (i.e. "don't sample mips finer than N"). Stored
    /// as f32 bits; 0 means default (no clamp). Zero bit-pattern of f32 is
    /// 0.0, which is also the natural default — clamping to "at least
    /// mip 0" is a no-op.
    pub lod_min_clamp: u32, // in: f32 bits
    /// Upper bound on selected mip LOD.
    ///
    /// Stored as f32 bits. A fixed `1000.0` matches the D3D9 convention —
    /// effectively "no upper clamp", with Metal naturally capping at the
    /// texture's actual mip count. Metal's default is `FLT_MAX`, so
    /// `1000.0` is just an explicit ceiling that makes the field's intent
    /// visible.
    pub lod_max_clamp: u32, // in: f32 bits
    /// 0 / non-zero: when set, the sampler is created with `compareFunction = LessEqual`.
    ///
    /// MSL `sample_compare(...)` against a `depth2d<float>` then returns
    /// the D3D9 hardware-shadow PCF result (1 = lit, 0 = shadowed) the
    /// terrain shadow shaders depend on. Set by the encoder for any
    /// sampler bound to a depth-format slot (`depth_sampler_mask` bit
    /// set); separate cache entry from the non-compare variant of the
    /// same D3D9 sampler state.
    pub is_compare: u32,
    pub sampler_handle: MetalHandle<MTLSamplerStateKind>, // out
}

impl Thunk for CreateSamplerStateParams {
    const CODE: u32 = Thunks::CreateSamplerState as u32;
}

/// Per-element descriptor inside `CreateBuffersBatchParams::descs_ptr`.
///
/// One entry per `MTLBuffer` to wrap. Each `backing_ptr` is caller-owned,
/// page-aligned, and stays in PE-addressable memory; the unix side wraps
/// it with `newBufferWithBytesNoCopy` (deallocator nil — PE retains
/// ownership). The backing is sourced PE-side because i386 PE pointers
/// cannot dereference into the unix heap above 4 GiB; allocating on the
/// PE side keeps the address in the low 32-bit range.
#[repr(C, align(8))]
pub struct BufferCreateDesc {
    pub backing_ptr: u64,          // in: *mut u8, caller-allocated, page-aligned
    pub length: u64,               // in: buffer size in bytes (page multiple)
    pub id: u64,                   // in: caller-defined id, formatted into MTLBuffer label
    pub storage_mode: StorageMode, // in: Private not supported for newBufferWithBytesNoCopy
    pub kind: BufferKind,          // in: role of the buffer, formatted into MTLBuffer label
}

/// Batched `MTLBuffer` wrap.
///
/// One PE↔Unix crossing wraps `count` PE-owned memory regions as
/// `MTLBuffer`s. Same shape as `CreateTexturesBatchParams`: caller-owned
/// `[BufferCreateDesc; count]` in and `[u64; count]` out, both 8-byte
/// aligned and stable for the duration of the call.
///
/// Single-create call sites use `count = 1` against a one-element array.
#[repr(C, align(8))]
pub struct CreateBuffersBatchParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub count: u32,                                // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub descs_ptr: u64,       // in: *const BufferCreateDesc, len=count
    pub handles_out_ptr: u64, // out: *mut MetalHandle<MTLBufferKind>, len=count (NULL on failure)
}

impl Thunk for CreateBuffersBatchParams {
    const CODE: u32 = Thunks::CreateBuffersBatch as u32;
}

/// Bulk MTL handle release.
///
/// PE side collects handles of one `DestroyKind` into a stable-backed
/// `&[u64]` (stack array for one handle, `Vec` for many) and the unix
/// dispatcher iterates the slice, dropping each handle's `Retained` to
/// decrement its objc refcount. Used at encoder shutdown (entire caches
/// released in 7 calls) and at any live mid-frame teardown that drops more
/// than a single handle.
#[repr(C, align(8))]
pub struct DestroyResourcesBulkParams {
    pub kind: DestroyKind, // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub handles_ptr: u64, // in: *const u64, stable for the duration of the call
    pub count: u32,       // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad1: u32,
}

impl Thunk for DestroyResourcesBulkParams {
    const CODE: u32 = Thunks::DestroyResourcesBulk as u32;
}

/// Synchronous texture→buffer readback.
///
/// The PE caller allocates a page-aligned PE-addressable heap block via
/// `PageBox`, passes its raw pointer as `dst_ptr` + `dst_len`. The unix side
/// wraps that memory as an `MTLBuffer` via `newBufferWithBytesNoCopy:`,
/// records a one-shot command buffer that blits `(origin_x, origin_y, width,
/// height)` of the source texture at `mip_level` into the buffer at
/// `bytes_per_row` stride, commits, and `waitUntilCompleted`. On return
/// `dst_ptr` contains the readback pixels. The caller holds onto the backing
/// until `UnlockRect`.
///
/// In-order queue execution makes it safe to call immediately after a
/// `MidFrameSubmit`: this command buffer cannot start until the
/// previously-submitted render command buffer has finished.
#[repr(C, align(8))]
pub struct BlitTextureToBufferParams {
    pub queue_handle: MetalHandle<MTLCommandQueueKind>, // in
    pub device_handle: MetalHandle<MTLDeviceKind>,      // in (for newBufferWithBytesNoCopy)
    pub tex_handle: MetalHandle<MTLTextureKind>,        // in
    pub dst_ptr: u64,       // in: page-aligned PE-addressable destination
    pub dst_len: u64,       // in: page-multiple length of dst_ptr
    pub mip_level: u32,     // in
    pub origin_x: u32,      // in
    pub origin_y: u32,      // in
    pub width: u32,         // in
    pub height: u32,        // in
    pub bytes_per_row: u32, // in: destination row stride
    /// Full width of the image `origin_*` / `width` / `height` are measured in.
    ///
    /// The *logical* resolution: under `render.scale` the source texture is
    /// rasterized smaller than what D3D9 reports, and readback has to hand the
    /// caller the resolution it asked for. When this differs from the texture's
    /// own width the unix side resolves the frame to this size through `MetalFX`
    /// before reading, so the pixels match what the display shows. Equal to the
    /// texture's width at the default scale, which makes the resolve a no-op.
    pub source_width: u32, // in
    /// Full height of the image the coordinates are measured in.
    ///
    /// See [`Self::source_width`].
    pub source_height: u32, // in
}

impl Thunk for BlitTextureToBufferParams {
    const CODE: u32 = Thunks::BlitTextureToBuffer as u32;
}

#[cfg(test)]
mod tests;
