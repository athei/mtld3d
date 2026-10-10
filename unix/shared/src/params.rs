use super::{
    Thunk, Thunks,
    mtl::{
        BufferKind, ColorSpacePolicy, CursorOverlayFlags, DestroyKind, DeviceCapsFlags, LoadAction,
        PixelFormat, PresentDebugFlags, PresentWaitPolicy, SoftwareCursorPolicy, StorageMode,
        StoreAction, Swizzle, TextureUsage, VertexFormat, VertexStepFunction,
    },
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLDeviceKind, MTLTextureKind, MetalHandle, NSViewKind,
    },
    record_handle::DeviceRecordHandle,
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
    // Device create / render / destroy structs: align must be 8 and size
    // identical on all targets.
    assert!(core::mem::align_of::<CreateCommandQueueParams>() == 8);
    assert!(core::mem::size_of::<CreateCommandQueueParams>() == 40);
    assert!(core::mem::offset_of!(CreateCommandQueueParams, present_debug) == 36);
    assert!(core::mem::size_of::<AttachMetalLayerParams>() == 88);
    assert!(core::mem::size_of::<DetachMetalLayerParams>() == 8);
    assert!(core::mem::size_of::<CreateBackbufferParams>() == 64);
    assert!(core::mem::size_of::<DestroyCommandQueueParams>() == 40);
    assert!(core::mem::size_of::<SetCursorOverlayParams>() == 56);
    assert!(core::mem::size_of::<PassDescriptor>() == 208);
};

/// One-shot "register `env_logger` on the unix side" thunk.
///
/// Fired once from d3d9.dll on load, before other thunks can log.
/// The UTF-8 filter comes from the PE process environment and is borrowed for the call.
///
/// `early_dir` and `early_stem` name the default log location, the unix path
/// of `mtld3d-logs` beside the executable and the executable's stem, both
/// UTF-8 without a terminator and borrowed for the call. A crash report that
/// arrives before `OpenLog` opens `<early_dir>/<early_stem>-<pid>.log` there,
/// since the process may die before `Direct3DCreate9` names the real one. A
/// zero length means the PE side could not derive it.
#[repr(C, align(8))]
pub struct InitLoggerParams {
    pub filter_ptr: u64,
    pub early_dir_ptr: u64,  // in: *const u8
    pub early_stem_ptr: u64, // in: *const u8
    pub filter_len: u32,
    pub early_dir_len: u32,  // in: byte count
    pub early_stem_len: u32, // in: byte count
    pub pad0: u32,
}

const _: () = {
    assert!(size_of::<InitLoggerParams>() == 40);
    assert!(align_of::<InitLoggerParams>() == 8);
    assert!(core::mem::offset_of!(InitLoggerParams, filter_ptr) == 0);
    assert!(core::mem::offset_of!(InitLoggerParams, early_dir_ptr) == 8);
    assert!(core::mem::offset_of!(InitLoggerParams, early_stem_ptr) == 16);
    assert!(core::mem::offset_of!(InitLoggerParams, filter_len) == 24);
    assert!(core::mem::offset_of!(InitLoggerParams, early_dir_len) == 28);
    assert!(core::mem::offset_of!(InitLoggerParams, early_stem_len) == 32);
    assert!(core::mem::offset_of!(InitLoggerParams, pad0) == 36);
};

impl Thunk for InitLoggerParams {
    const CODE: u32 = Thunks::InitLogger as u32;
}

/// What a line `WriteLog` carries is, which decides where it may go.
#[repr(u32)]
pub enum LogLineKind {
    /// An ordinary line: it waits in the backlog until `OpenLog` names the file.
    Ordinary = 0,
    /// A line of a fault report the process may still recover from.
    ///
    /// Written as an ordinary line; it marks the process, so an exit before
    /// `OpenLog` saves the backlog to the early location.
    FaultReport = 1,
    /// A line of a terminal fault's report: before `OpenLog` it opens the early location.
    CrashReport = 2,
}

/// One formatted log line from the PE-side logger, for the process's log file.
///
/// `ptr`/`len` describe a byte slice the PE side keeps alive for the call.
#[repr(C, align(8))]
pub struct WriteLogParams {
    pub ptr: u64,          // in: *const u8
    pub len: u32,          // in: byte count
    pub kind: LogLineKind, // in
}

const _: () = {
    assert!(size_of::<WriteLogParams>() == 16);
    assert!(align_of::<WriteLogParams>() == 8);
    assert!(core::mem::offset_of!(WriteLogParams, len) == 8);
    assert!(core::mem::offset_of!(WriteLogParams, kind) == 12);
};

impl Thunk for WriteLogParams {
    const CODE: u32 = Thunks::WriteLog as u32;
}

/// Where this process's log file and GPU traces go, sent once per process.
///
/// `Direct3DCreate9` resolves `mtld3d.conf` long after `InitLogger` fired
/// from `DllMain`, so the location travels separately. `dir` is the unix
/// path of the log directory, `stem` the executable's file name without its
/// extension, both UTF-8 without a terminator and kept alive by the PE side
/// for the call. The unix side names the process itself, by its host pid:
/// the Windows pid the PE side sees is Wine's, and Wine hands the first
/// process of every fresh wineserver the same one. It opens
/// `<dir>/<stem>-<pid>.log` on the first line it writes and numbers the
/// traces `<dir>/<stem>-<pid>-<n>.gputrace`.
///
/// `main_thread_checker` rides along because this is the one thunk that
/// fires once per process with the resolved configuration in hand: set, the
/// unix side loads Apple's Main Thread Checker before it opens the log, so
/// every `AppKit` call the layer makes from then on is checked.
#[repr(C, align(8))]
pub struct OpenLogParams {
    pub dir_ptr: u64,             // in: *const u8
    pub stem_ptr: u64,            // in: *const u8
    pub dir_len: u32,             // in: byte count
    pub stem_len: u32,            // in: byte count
    pub main_thread_checker: u32, // in: 1 loads Apple's Main Thread Checker
    pub pad0: u32,
}

impl Thunk for OpenLogParams {
    const CODE: u32 = Thunks::OpenLog as u32;
}

#[repr(C, align(8))]
pub struct GetDeviceInfoParams {
    pub name_ptr: u64,
    pub name_buf_len: u64,
    pub name_len: u64,    // out
    pub registry_id: u64, // out
    /// Out: the device's boolean capability bits.
    ///
    /// See `DeviceCapsFlags` for the members; the PE side caches the whole
    /// answer once per process and derives every device-conditional cap from
    /// it.
    pub caps: DeviceCapsFlags,
    pub pad0: u32,
}

impl Thunk for GetDeviceInfoParams {
    const CODE: u32 = Thunks::GetDeviceInfo as u32;
}

#[repr(C, align(8))]
pub struct CreateCommandQueueParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // out
    /// The device's unix-side record: its command queue and presentation state.
    ///
    /// Opaque to the PE side, which keeps it on `DeviceInner` and names the
    /// device with it on every later thunk. `DestroyCommandQueue` returns it
    /// and frees the record.
    pub record_handle: DeviceRecordHandle, // out
    /// 0 / non-zero boolean: `MTLDevice.hasUnifiedMemory`.
    ///
    /// False on Intel/AMD non-UMA Macs; the storage-mode policy in
    /// `mtld3d-core::storage_policy` switches CPU-visible buffers to
    /// `Managed` and the encoder enqueues `didModifyRange:` calls when
    /// this is 0. (Textures are always `Private`.) The PE side may
    /// force the non-UMA answer through `intel.managedMemory`.
    pub unified_memory: u32, // out
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`.
    ///
    /// 16 on Apple Silicon, 256 on AMD/Intel (Mac2). Threaded into
    /// `pad_source_stride` so blit-staging `bytes_per_row` rounds to this
    /// floor. The PE side may raise it to the Mac2 value through
    /// `intel.linearAlign256`.
    pub min_linear_texture_align: u32, // out
    /// `debug.presentGateFile` as a unix path, `0` = no gate.
    ///
    /// While the named file exists, the queue's presenter parks before
    /// acquiring a drawable. The bytes are valid for the call; the unix
    /// side copies them into the presenter state it creates for this queue.
    pub gate_file_ptr: u64, // in: *const u8
    pub gate_file_len: u32,                        // in: byte count
    /// The queue's presenter switches, `debug.presentOccluded` among them.
    pub present_debug: PresentDebugFlags, // in
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
    /// Wine's retina factor for the attached window: 2 in retina mode, else 1.
    ///
    /// The Wine metal layer's `contentsScale`, rounded and clamped to
    /// `[1, 8]`. Consumed by the PE-side cursor upscaler: in retina mode the
    /// game draws at physical pixels and its cursor comes out at half a
    /// point per pixel, while in non-retina mode macOS already doubles
    /// everything the game draws, the cursor included.
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
    /// the interface's `hdr_enable`; unix side feeds it to `resolve_hdr_active`.
    pub hdr_enable: u32, // in
    /// `color.space` from `mtld3d.conf`.
    ///
    /// `Passthrough` (the default, today's behaviour) tags the layer with
    /// the display's own `CGColorSpace` — D3D9's untagged values land at
    /// the panel's native primaries. `Accurate` overrides that with the
    /// sRGB family for both SDR and HDR paths so guest art reads with its
    /// designer-intended hues. PE side reads this from
    /// the interface's `color_space`.
    pub color_space: ColorSpacePolicy, // in
    /// The effective frame-rate ceiling in Hz, `0` = uncapped.
    ///
    /// The lower of `present.maxFps` from `mtld3d.conf` and the ceiling a
    /// divided `PresentationInterval` sets (the reported refresh rate over
    /// two, three or four), resolved on the PE side. Combined with the vsync
    /// request into the present-throttle duration, where the lower rate wins.
    pub max_fps: u32, // in
    /// Whether this GPU can run a `MetalFX` spatial upscale.
    ///
    /// Non-zero lets the PE side size the drawable to the window and leave
    /// the resample to `MetalFX`. Zero means the drawable must keep the back
    /// buffer's size so present stays a 1:1 copy and Core Animation scales
    /// the layer instead — the pre-`MetalFX` behaviour, and the only correct
    /// fallback since nothing else on the unix side can resize a frame.
    pub metalfx_available: u32, // out
    /// Address of a PE-side `AtomicU32` that receives a changed `backing_scale`.
    ///
    /// `backing_scale` above answers for the layer as attach found it. The
    /// unix side stores a new value here whenever its display-follow
    /// reconciliation derives one, so the PE-side cursor upscale picks it up
    /// without a second thunk. The word lives in a heap box the device owns
    /// and keeps at one address for its lifetime; the unix side records the
    /// address on the attachment record it creates for this view and writes
    /// through it only while that record is registered, and
    /// `DestroyCommandQueue` unregisters the record before the box is
    /// dropped. `0` disables the republish.
    pub backing_scale_ptr: u64, // in: *const AtomicU32 (device-owned box, stable address)
    /// `cursor.software` from `mtld3d.conf`.
    ///
    /// Resolved on the unix side against the layer mode attach picked, since
    /// `Auto` means "on when the present path is HDR" and only the unix side
    /// knows that. PE side reads this from the interface's `cursor_software`.
    pub software_cursor: SoftwareCursorPolicy, // in
    /// Whether this device draws its cursor through the overlay window.
    ///
    /// Non-zero = the PE side keeps the Win32 cursor blank over the client
    /// area and ships cursor bitmaps and visibility through
    /// `SetCursorOverlay`; zero = the hardware HCURSOR path. Resolved once per
    /// device at attach and held for its lifetime.
    pub software_cursor_active: u32, // out
    /// Address of a PE-side `AtomicU32` the unix side sets to ask for a cursor re-apply.
    ///
    /// Set to non-zero when the pointer comes back after another process held
    /// it (a system tool such as the screenshot crosshair), which leaves that
    /// process's cursor on screen. Wine re-applies its cursor only on a handle
    /// change, so the PE side answers a set flag with its null-then-set kick
    /// at the next `WM_SETCURSOR` or `ShowCursor(TRUE)`, taking the flag back
    /// to zero. Every live device's word is set, since the kick is idempotent
    /// and the pointer's return concerns each of them. Same backing contract
    /// as `backing_scale_ptr`. `0` disables it.
    pub cursor_kick_ptr: u64, // in: *const AtomicU32 (device-owned box, stable address)
}

impl Thunk for AttachMetalLayerParams {
    const CODE: u32 = Thunks::AttachMetalLayer as u32;
}

/// Retire one metal view: its attachment record first, then the view itself.
///
/// The counterpart of `AttachMetalLayer` for a device that keeps running. A
/// `Reset` naming another `hDeviceWindow` sends this for the view it is
/// leaving and attaches a fresh one on the new window, so the device presents
/// into the window its presentation parameters name. Device teardown does the
/// same work inside `DestroyCommandQueue`, which owns the queue and the
/// device as well and has to fence between the two halves.
///
/// A view with no attachment record, and the null handle a device that never
/// attached carries, are both no-ops.
#[repr(C, align(8))]
pub struct DetachMetalLayerParams {
    pub view_handle: MetalHandle<NSViewKind>, // in
}

impl Thunk for DetachMetalLayerParams {
    const CODE: u32 = Thunks::DetachMetalLayer as u32;
}

/// Wanted state of the software cursor overlay: which sprite, and whether it shows.
///
/// Sent from the API thread on every `ShowCursor` and `SetCursorProperties`
/// while the software cursor is active, and on `ShowCursor` transitions
/// with `CursorOverlayFlags::HARDWARE` set (visibility only, `hash` 0) while
/// the hardware cursor is. `hash` names the sprite (never `0` otherwise);
/// `pixels_ptr` carries its BGRA bytes the first time the PE side sends a hash
/// and is `0` afterwards, the unix side keeping every sprite it has been
/// handed. Sprite pixels are already upscaled by `scale` (pixels per point),
/// the hotspot is in sprite pixels. The handler stores the state and queues one
/// main-thread apply; it never blocks on `AppKit`, which is what lets this
/// thunk run on the API thread.
#[repr(C, align(8))]
pub struct SetCursorOverlayParams {
    pub hash: u64,                 // in: sprite identity, never 0
    pub pixels_ptr: u64,           // in: *const u8 BGRA rows, 0 = sprite already uploaded
    pub pixels_len: u32,           // in: byte count = width * height * 4
    pub width: u32,                // in: sprite width in pixels
    pub height: u32,               // in: sprite height in pixels
    pub x_hotspot: u32,            // in: in sprite pixels
    pub y_hotspot: u32,            // in: in sprite pixels
    pub scale: u32,                // in: sprite pixels per point, 1..=8
    pub flags: CursorOverlayFlags, // in
    pub pad0: u32,
    /// The metal view of the device that speaks: its `AttachMetalLayer` `view_handle`.
    ///
    /// Names the attachment record whose window the overlay is drawn over
    /// and whose layer mode, colorspace, headroom and backing scale the
    /// sprite is rendered for. The overlay follows the device whose call
    /// arrived most recently, since `SetCursorProperties` and `ShowCursor`
    /// are per-device calls. A view no attachment record names is rejected.
    pub view_handle: MetalHandle<NSViewKind>, // in
}

impl Thunk for SetCursorOverlayParams {
    const CODE: u32 = Thunks::SetCursorOverlay as u32;
}

/// Set how a present-bearing submit on `record_handle` treats a pending present.
///
/// The PE-side barrier that waits for its in-flight submits sets
/// `SnapshotPending` first and `WaitForCommit` after, so no submit it waits
/// for can itself wait on the display; a synchronous flush sets it from the
/// API thread before it queues behind the encoder, and the encoder's flush
/// arm puts it back. See `PresentWaitPolicy`.
#[repr(C, align(8))]
pub struct SetPresentWaitPolicyParams {
    pub record_handle: DeviceRecordHandle, // in
    pub policy: PresentWaitPolicy,         // in
    pub pad0: u32,
}

impl Thunk for SetPresentWaitPolicyParams {
    const CODE: u32 = Thunks::SetPresentWaitPolicy as u32;
}

#[repr(C, align(8))]
pub struct DestroyCommandQueueParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub record_handle: DeviceRecordHandle,         // in
    pub view_handle: MetalHandle<NSViewKind>,      // in (NULL = none)
    pub backbuffer_handle: MetalHandle<MTLTextureKind>, // in (NULL = none)
    pub depth_texture_handle: MetalHandle<MTLTextureKind>, // in (NULL = none)
}

impl Thunk for DestroyCommandQueueParams {
    const CODE: u32 = Thunks::DestroyCommandQueue as u32;
}

#[repr(C, align(8))]
pub struct CreateBackbufferParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    /// The device whose frame queue the creation-time clear is encoded on.
    ///
    /// A new `MTLTexture` has undefined contents, and the back buffer is
    /// presentable before the application's first draw or clear reaches it.
    /// Encoding the clear on the frame queue makes commit order the fence:
    /// every later frame command buffer observes a black back buffer.
    pub record_handle: DeviceRecordHandle, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    /// Multisample count of the back buffer, 1 for none.
    ///
    /// Above 1 the thunk creates a second, multisampled texture beside the
    /// single-sample one and returns it in `msaa_texture_handle`. The
    /// single-sample texture stays the one Present, `StretchRect` and
    /// `LockRect` read; the multisampled one is the colour attachment every
    /// pass renders into and resolves from.
    pub sample_count: u32, // in
    // allow: FFI struct padding; pub for cross-crate field-init.
    pub pad0: u32,
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Eagerly-created sRGB twin view of `texture_handle`.
    ///
    /// The back buffer is `Bgra8Unorm`, which has an sRGB counterpart, so
    /// this is never NULL on success. The render pass attaches it in place
    /// of the base texture under `D3DRS_SRGBWRITEENABLE`, which is what
    /// gives the encode its D3D9 position, after the blender. Destroyed
    /// with the base texture.
    pub srgb_texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Multisampled companion of `texture_handle`, NULL when `sample_count` is 1.
    pub msaa_texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Eagerly-created sRGB twin view of `msaa_texture_handle`.
    ///
    /// A multisampled pass writing sRGB attaches this and resolves into
    /// `srgb_texture_handle`: Metal requires the resolve texture to carry the
    /// attachment's pixel format, so the two views have to agree. NULL
    /// whenever `msaa_texture_handle` is.
    pub msaa_srgb_texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateBackbufferParams {
    const CODE: u32 = Thunks::CreateBackbuffer as u32;
}

/// Vertex attribute descriptor, one per Metal vertex input attribute.
///
/// Borrowed by native pipeline creation and captured in shader input records.
#[repr(C, align(4))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VertexAttrDesc {
    pub attr_index: u32,      // in: Metal attribute slot ([[attribute(N)]])
    pub buffer_index: u32,    // in: Metal buffer slot
    pub offset: u32,          // in: byte offset within the buffer
    pub format: VertexFormat, // in
}

/// One vertex buffer layout of a render pipeline: a Metal vertex buffer slot the draw reads.
///
/// Borrowed by native pipeline creation,
/// one entry per Metal vertex buffer slot that feeds an attribute. `stride` is never 0
/// (Metal rejects it for every step function). `step_rate` is the instances
/// per advance for `PerInstance`, 1 for `PerVertex`, 0 for `Constant`.
#[repr(C, align(4))]
pub struct VertexBufferLayoutDesc {
    pub buffer_index: u32, // in: Metal vertex buffer slot (stream n at n, or a crossing slot)
    pub stride: u32,       // in: bytes per step
    pub step_function: VertexStepFunction, // in
    pub step_rate: u32,    // in
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
/// pass's draws and the next pass's draws. The frame-leading blit slice
/// runs at frame start and would
/// mis-order a mid-frame blit. A pass with `color_texture == 0` and
/// `command_count == 0` is a "blit-only" trailing pass synthesised when
/// `StretchRect` lands after the last draw of the frame.
///
/// Fields are ordered u64s-first then an even number of u32s, so the layout
/// carries no implicit padding on either PE architecture.
#[repr(C, align(8))]
pub struct PassDescriptor {
    pub color_texture: MetalHandle<MTLTextureKind>, // in
    /// Single-sample texture `color_texture` resolves into, NULL when it is not multisampled.
    ///
    /// Non-NULL only alongside a `color_store_action` that carries a resolve;
    /// it takes the same slice and level as the colour attachment, because it
    /// is the same D3D9 surface seen without multisampling.
    pub color_resolve_texture: MetalHandle<MTLTextureKind>, // in (NULL = no resolve)
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
    /// Store action for the depth plane alone.
    ///
    /// The stencil plane of a combined texture takes `stencil_store_action`.
    pub depth_store_action: StoreAction, // in
    pub depth_clear_value: u32,                     // in: f32 bits (default 1.0)
    /// Load action for the stencil half of a combined depth/stencil texture.
    ///
    /// Independent of `depth_load_action` because D3D9 clears the two planes
    /// separately: `Clear(D3DCLEAR_STENCIL)` without `D3DCLEAR_ZBUFFER` has to
    /// reset stencil while carrying depth forward. Ignored when the depth
    /// texture's format has no stencil plane.
    pub stencil_load_action: LoadAction, // in
    /// Store action for the stencil half of a combined depth/stencil texture.
    ///
    /// Independent of `depth_store_action`: the two planes of a
    /// `Depth32Float_Stencil8` attachment each take their own store action,
    /// so a plane nothing reads later is discarded while the other is kept.
    /// Ignored when the depth texture's format has no stencil plane.
    pub stencil_store_action: StoreAction, // in
    pub stencil_clear_value: u32,                   // in: 0..=255
    pub command_count: u32,                         // in
    pub leading_blits_count: u32,                   // in
    /// Leading-blit and color-subresource flags.
    ///
    /// Bit 0 is whether the leading-blit list needs an encoder. Bits 1..11
    /// carry the color attachment slice, wide enough for every depth plane of
    /// a volume, bits 12..15 its mip level and bits 16..19 the depth
    /// attachment's mip level. Ordinary 2D level-zero passes therefore retain
    /// their previous 0/1 value and the descriptor keeps its size.
    pub pass_flags: u32, // in
    /// Keeps the `u32` run even, so `extra_color` needs no implicit padding.
    pub reserved: u32,
    /// Render targets 1..3 (`colorAttachments[1..=3]`); `texture` null = unbound.
    ///
    /// They share `clear_r..clear_a` with attachment 0 (a D3D9 `Clear` has
    /// one colour for every target) and carry their own load/store actions.
    /// The shared colour is set whenever any attachment clears, including a
    /// pass whose attachment 0 is stripped or does not clear.
    pub extra_color: [ExtraColorDesc; 3], // in
}

impl PassDescriptor {
    const LEADING_BLITS_NEED_ENCODER: u32 = 1;
    const COLOR_SLICE_SHIFT: u32 = 1;
    const COLOR_LEVEL_SHIFT: u32 = 12;
    const DEPTH_LEVEL_SHIFT: u32 = 16;
    /// Largest color attachment slice `pass_flags` carries.
    ///
    /// Eleven bits: an upload pass addresses each depth plane of a volume,
    /// and `D3DCAPS9::MaxVolumeExtent` advertises 2048.
    pub const MAX_COLOR_SLICE: u32 = 0x7ff;

    /// Pack leading-blit, color-subresource and depth-level state.
    #[must_use]
    pub const fn pack_flags(
        needs_encoder: bool,
        color_slice: u32,
        color_level: u32,
        depth_level: u32,
    ) -> u32 {
        (if needs_encoder { 1 } else { 0 })
            | ((color_slice & Self::MAX_COLOR_SLICE) << Self::COLOR_SLICE_SHIFT)
            | ((color_level & 0xf) << Self::COLOR_LEVEL_SHIFT)
            | ((depth_level & 0xf) << Self::DEPTH_LEVEL_SHIFT)
    }

    /// Depth attachment mip level.
    #[must_use]
    pub const fn depth_level(&self) -> u32 {
        (self.pass_flags >> Self::DEPTH_LEVEL_SHIFT) & 0xf
    }

    /// Whether the leading-blit list contains an encoder-bound command.
    #[must_use]
    pub const fn leading_blits_need_encoder(&self) -> bool {
        self.pass_flags & Self::LEADING_BLITS_NEED_ENCODER != 0
    }

    /// Color attachment array slice.
    #[must_use]
    pub const fn color_slice(&self) -> u32 {
        (self.pass_flags >> Self::COLOR_SLICE_SHIFT) & Self::MAX_COLOR_SLICE
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
/// bits 8..15: an extra attachment is a render target, whose slice is a cube
/// face at most. 24 bytes, 8-aligned through `texture`.
#[repr(C)]
pub struct ExtraColorDesc {
    pub texture: MetalHandle<MTLTextureKind>, // in (NULL = unbound)
    /// See [`PassDescriptor::color_resolve_texture`].
    pub resolve_texture: MetalHandle<MTLTextureKind>, // in (NULL = no resolve)
    pub subresource: u32,                     // in: slice | (level << 8)
    pub load_action: LoadAction,              // in
    pub store_action: StoreAction,            // in
    pub reserved: u32,
}

impl ExtraColorDesc {
    /// The unbound attachment.
    pub const NONE: Self = Self {
        texture: MetalHandle::NULL,
        resolve_texture: MetalHandle::NULL,
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

#[repr(C, align(8))]
pub struct CreateDepthTextureParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub pixel_format: PixelFormat, // in (resolved via mtld3d_core::format::map_d3d_depth_format)
    /// Multisample count of the depth attachment, 1 for none.
    ///
    /// D3D9 offers no way to read a multisampled depth surface, so there is
    /// no resolve companion here: the texture the thunk returns is itself the
    /// multisampled one, and it must match the colour attachment's count.
    pub sample_count: u32, // in
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateDepthTextureParams {
    const CODE: u32 = Thunks::CreateDepthTexture as u32;
}

#[repr(C, align(8))]
pub struct CreateColorTargetParams {
    pub device_handle: MetalHandle<MTLDeviceKind>, // in
    /// The device whose frame queue the creation-time clear is encoded on.
    pub record_handle: DeviceRecordHandle, // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub pixel_format: PixelFormat, // in (resolved via mtld3d_core::format::map_d3d_format)
    /// Multisample count of the render target, 1 for none.
    ///
    /// Above 1 the thunk creates a multisampled companion beside the
    /// single-sample texture and returns it in `msaa_texture_handle`; see
    /// [`CreateBackbufferParams::sample_count`].
    pub sample_count: u32, // in
    pub texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Eagerly-created sRGB twin view of `texture_handle`.
    ///
    /// NULL when the format has no sRGB counterpart. Same role as
    /// `CreateBackbufferParams::srgb_texture_handle`.
    pub srgb_texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Multisampled companion of `texture_handle`, NULL when `sample_count` is 1.
    pub msaa_texture_handle: MetalHandle<MTLTextureKind>, // out
    /// Eagerly-created sRGB twin view of `msaa_texture_handle`.
    ///
    /// Same role as `CreateBackbufferParams::msaa_srgb_texture_handle`.
    pub msaa_srgb_texture_handle: MetalHandle<MTLTextureKind>, // out
}

impl Thunk for CreateColorTargetParams {
    const CODE: u32 = Thunks::CreateColorTarget as u32;
}

/// One native texture creation description.
///
/// One entry per `MTLTexture` to create. The unix side iterates the slice
/// and writes each resulting handle into the matching slot of
/// `views_out_ptr`. No `device_handle` or output handle field here: both
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

/// One native buffer creation description.
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
/// height)` of the source texture at `slice` / `mip_level` into the buffer at
/// `bytes_per_row` stride, commits, and `waitUntilCompleted`. On return
/// `dst_ptr` contains the readback pixels. The caller holds onto the backing
/// until `UnlockRect`.
///
/// In-order queue execution makes it safe to call immediately after a
/// `MidFrameSubmit`: this command buffer cannot start until the
/// previously-submitted render command buffer has finished.
#[repr(C, align(8))]
pub struct BlitTextureToBufferParams {
    /// Native layout, including both planes in one submission when requested.
    pub planes: crate::mtl::ReadbackPlanes,
    /// Row pitch of the optional stencil plane.
    pub stencil_bytes_per_row: u32,
    /// Byte offset of stencil within the same page-aligned destination allocation.
    pub stencil_offset: u64,
    pub record_handle: DeviceRecordHandle,         // in
    pub device_handle: MetalHandle<MTLDeviceKind>, // in (for newBufferWithBytesNoCopy)
    pub tex_handle: MetalHandle<MTLTextureKind>,   // in
    pub dst_ptr: u64,                              // in: page-aligned PE-addressable destination
    pub dst_len: u64,                              // in: page-multiple length of dst_ptr
    pub mip_level: u32,                            // in
    /// Source array slice: a cube face index, zero for every other texture.
    pub slice: u32, // in
    pub origin_x: u32,                             // in
    pub origin_y: u32,                             // in
    pub width: u32,                                // in
    pub height: u32,                               // in
    pub bytes_per_row: u32,                        // in: destination row stride
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
    /// Block height of the source format: 1 uncompressed, 4 for the BC family.
    ///
    /// `bytes_per_row` is the stride of one *block* row, so a slice is
    /// `ceil(height / block_height)` rows rather than `height`. The unix side
    /// has no format table and derives the blit's `bytesPerImage` from this
    /// through [`crate::blit_geometry::bytes_per_image`].
    pub block_height: u32, // in
}

impl Thunk for BlitTextureToBufferParams {
    const CODE: u32 = Thunks::BlitTextureToBuffer as u32;
}

#[cfg(test)]
mod tests;
