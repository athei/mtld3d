use super::{
    Thunk, Thunks,
    mtl::{
        AddressMode, BlendFactor, BlendOperation, BufferKind, ClearQuadFlags, ColorSpacePolicy,
        ColorWriteMask, CompareFunc, DestroyKind, LoadAction, MinMagFilter, MipFilter, PixelFormat,
        StageTag, StorageMode, StoreAction, Swizzle, TextureUsage, VertexFormat,
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
// PE arches (i686 + x86_64 `*-pc-windows-msvc`) AND the x86_64 unix `.so`. The
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
    assert!(core::mem::offset_of!(CreateDepthStencilStateParams, id) == 24);
    assert!(core::mem::size_of::<CreateDepthStencilStateParams>() == 40);

    // Device create / render / destroy structs: align must be 8 and size
    // identical on all targets.
    assert!(core::mem::align_of::<CreateCommandQueueParams>() == 8);
    assert!(core::mem::size_of::<CreateCommandQueueParams>() == 24);
    assert!(core::mem::size_of::<AttachMetalLayerParams>() == 64);
    assert!(core::mem::size_of::<CreateBackbufferParams>() == 24);
    assert!(core::mem::size_of::<DestroyCommandQueueParams>() == 48);
    assert!(core::mem::size_of::<SubmitFrameParams>() == 96);
    assert!(core::mem::size_of::<PassDescriptor>() == 88);
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
    // Keeps the struct repr(C, align(8)) layout deterministic across PE
    // and Unix linkage units. Constructed by name from d3d9, hence pub.
    pub pad0: u32,
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

/// Update `CAMetalLayer.drawableSize` on an already-attached layer.
///
/// Used by the D3D9 Reset path to honour a window resize: the rendering
/// surface's pixel dimensions must match the new backbuffer texture so the
/// `present` blit covers the drawable 1:1.
#[repr(C, align(8))]
pub struct SetLayerDrawableSizeParams {
    pub layer_handle: MetalHandle<CAMetalLayerKind>, // in
    pub width: u32,                                  // in: pixels
    pub height: u32,                                 // in: pixels
}

impl Thunk for SetLayerDrawableSizeParams {
    const CODE: u32 = Thunks::SetLayerDrawableSize as u32;
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

/// Query the primary display's pixel size and refresh rate via `NSScreen`.
///
/// Used at `Direct3DCreate9` time to build a realistic `EnumAdapterModes`
/// table around the host's actual desktop size. macOS doesn't do D3D9-style
/// mode-setting (`CAMetalLayer` renders at whatever size we ask, the
/// `WindowServer` composites it onto the actual desktop), so the values are
/// purely advisory — they shape the game's UI dropdown, not the actual
/// rendering surface.
#[repr(C, align(8))]
pub struct GetPrimaryDisplayModeParams {
    pub width: u32,      // out: pixels (NSScreen.frame.size.width, rounded)
    pub height: u32,     // out: pixels
    pub refresh_hz: u32, // out: NSScreen.maximumFramesPerSecond, or 0 if unknown
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
}

impl Thunk for GetPrimaryDisplayModeParams {
    const CODE: u32 = Thunks::GetPrimaryDisplayMode as u32;
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
    pub device_handle: MetalHandle<MTLDeviceKind>,   // in
    pub width: u32,                                  // in
    pub height: u32,                                 // in
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

#[repr(C, align(8))]
pub struct CreateRenderPipelineParams {
    pub device_handle: MetalHandle<MTLDeviceKind>,  // in
    pub vs_fn_handle: MetalHandle<MTLFunctionKind>, // in
    pub ps_fn_handle: MetalHandle<MTLFunctionKind>, // in
    pub vertex_attrs_ptr: u64,                      // in: *const VertexAttrDesc
    pub vertex_attr_count: u32,                     // in
    pub vertex_stride: u32,                         // in: bytes per vertex on buffer 0
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
    pub pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>, // out
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
/// is padding-free; size is 88 bytes on both 32- and 64-bit PE.
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
    pub command_count: u32,                         // in
    pub leading_blits_count: u32,                   // in
    /// 0 / non-zero: whether the leading-blit list contains any encoder-bound command.
    ///
    /// Encoder-bound = the CopyBuffer/Texture variants. The unix
    /// dispatcher uses this to skip `MTLBlitCommandEncoder` creation on
    /// pure-`NotifyBufferDidModifyRange` lists; saves a per-pass scan of
    /// the blit slice on the encoder thread.
    pub leading_blits_need_encoder: u32, // in
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
    pub drawable_wait_tsc: u64, // out: TSC cycles spent in nextDrawable()
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
    pub id: u64,                                   // in: caller-defined label tag
    pub state_handle: MetalHandle<MTLDepthStencilStateKind>, // out
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
    pub has_swizzle: u32,          // in: non-zero = use swizzle fields
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
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
}

impl Thunk for BlitTextureToBufferParams {
    const CODE: u32 = Thunks::BlitTextureToBuffer as u32;
}

#[cfg(test)]
mod tests {
    use super::{
        BufferCreateDesc, CreateBuffersBatchParams, CreateTexturesBatchParams,
        DestroyResourcesBulkParams, PassDescriptor, SubmitFrameParams, TextureCreateDesc,
    };

    #[test]
    fn buffer_param_layouts_match_wow64() {
        // All thunk params must be 8-byte aligned and contain only u32/u64
        // fields so 32-bit PE and 64-bit Unix agree on layout.
        assert_eq!(core::mem::align_of::<CreateBuffersBatchParams>(), 8);
        assert_eq!(core::mem::align_of::<BufferCreateDesc>(), 8);
        assert_eq!(core::mem::align_of::<DestroyResourcesBulkParams>(), 8);

        // Sizes: sum of fields with repr(C, align(8)) padding:
        //   CreateBuffersBatchParams   = 8 + 4 + 4 + 8 + 8     = 32
        //   BufferCreateDesc           = 8 + 8 + 8 + 4 + 4     = 32
        //   DestroyResourcesBulkParams = 4 + 4 + 8 + 4 + 4     = 24
        assert_eq!(core::mem::size_of::<CreateBuffersBatchParams>(), 32);
        assert_eq!(core::mem::size_of::<BufferCreateDesc>(), 32);
        assert_eq!(core::mem::size_of::<DestroyResourcesBulkParams>(), 24);
    }

    #[test]
    fn attach_metal_layer_layout() {
        use super::AttachMetalLayerParams;
        // 2*u64 + 2*u32 + 2*u64 + 2*u32 + 1*u32 + 1*ColorSpacePolicy
        // + 2*u32 = 16 + 8 + 16 + 8 + 4 + 4 + 8 = 64 (explicit pad0
        // keeps the size a multiple of the align-8).
        assert_eq!(core::mem::align_of::<AttachMetalLayerParams>(), 8);
        assert_eq!(core::mem::size_of::<AttachMetalLayerParams>(), 64);
    }

    #[test]
    fn set_display_sync_enabled_layout() {
        use super::SetDisplaySyncEnabledParams;
        // u64 + u32 + u32 = 8 + 4 + 4 = 16
        assert_eq!(core::mem::align_of::<SetDisplaySyncEnabledParams>(), 8);
        assert_eq!(core::mem::size_of::<SetDisplaySyncEnabledParams>(), 16);
    }

    #[test]
    fn set_layer_drawable_size_layout() {
        use super::SetLayerDrawableSizeParams;
        // u64 + u32 + u32 = 8 + 4 + 4 = 16
        assert_eq!(core::mem::align_of::<SetLayerDrawableSizeParams>(), 8);
        assert_eq!(core::mem::size_of::<SetLayerDrawableSizeParams>(), 16);
    }

    #[test]
    fn wait_for_gpu_retire_layout() {
        use super::WaitForGpuRetireParams;
        // 2 * u64 = 16
        assert_eq!(core::mem::align_of::<WaitForGpuRetireParams>(), 8);
        assert_eq!(core::mem::size_of::<WaitForGpuRetireParams>(), 16);
    }

    #[test]
    fn frame_param_layouts_match_wow64() {
        assert_eq!(core::mem::align_of::<PassDescriptor>(), 8);
        assert_eq!(core::mem::align_of::<SubmitFrameParams>(), 8);
        assert_eq!(core::mem::align_of::<CreateTexturesBatchParams>(), 8);
        assert_eq!(core::mem::align_of::<TextureCreateDesc>(), 8);

        // PassDescriptor: 5 * u64 + 12 * u32 = 40 + 48 = 88, already 8-aligned.
        assert_eq!(core::mem::size_of::<PassDescriptor>(), 88);

        // SubmitFrameParams:
        //   8 queue_handle
        //   + 8 blit_commands_ptr + 4 blit_command_count + 4 blit_commands_need_encoder
        //   + 8 passes_ptr + 4 pass_count + 4 _pad1
        //   + 8 present_layer + 8 present_texture
        //   + 8 submit_seq + 8 coherent_seq_ptr + 8 upload_coherent_seq_ptr
        //   + 8 drawable_wait_tsc + 8 present_view
        //   = 96
        assert_eq!(core::mem::size_of::<SubmitFrameParams>(), 96);

        // CreateTexturesBatchParams:
        //   8 device_handle + 4 count + 4 _pad0 + 8 descs_ptr + 8 handles_out_ptr = 32
        assert_eq!(core::mem::size_of::<CreateTexturesBatchParams>(), 32);

        // TextureCreateDesc:
        //   8 tex_id
        //   + 4 width + 4 height + 4 depth + 4 levels (16)
        //   + 4 pixel_format + 4 storage_mode + 4 has_swizzle + 4 swizzle_r (16)
        //   + 4 swizzle_g + 4 swizzle_b + 4 swizzle_a + 4 usage_flags (16)
        //   = 56
        assert_eq!(core::mem::size_of::<TextureCreateDesc>(), 56);
    }
}
