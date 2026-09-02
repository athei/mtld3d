use std::{
    collections::{VecDeque, hash_map::Entry},
    fs::{File, OpenOptions},
    io::Write as _,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use log::{Level, debug, error, log_enabled, trace};
use mtld3d_core::{
    buffer_rename::BufferMapMode,
    convert::{FAN_PATTERN_MAX_TRIANGLES, fan_pattern_bytes, fill_fan_pattern_u16},
    depth_stencil_state::{DepthStencilSnapshot, key_from_snapshot, params_from_snapshot},
    dxso::{
        DxsoProgram, FfPsKey, FfVsKey, LOG_TARGET as MSL_TRACE_TARGET, VariantKey, VsSamplerKinds,
        declared_ps_samplers, emit_ps_ff_named, emit_ps_programmable_named, emit_vs_ff_named,
        emit_vs_programmable_named,
    },
    format::map_d3d_format,
    gpu_caps::GpuCaps,
    ids::{BufferId, DepthStencilKey, ProgramId, SamplerKey, TextureId},
    page_box::PageBox,
    passes::{
        ColorClearOutcome, ColorLoad, DepthClearOutcome, DepthLoad, DepthResolve, ExtraColorSlot,
        LastBoundCache, Pass, PassState, StencilClearOutcome, StencilLoad,
        StoreAction as PassStoreAction, UploadPassTarget,
    },
    perf::{
        CacheSizes, EncoderPerfState, FramePerfPayload, FrameSummaryContext, OpSub, OpSubDetail,
        PairShaderId, PairStatsSample, TaskFaults, perf_enabled,
    },
    pipeline_state::{self, PipelineBuildInputs, PipelineKey, PipelineSnapshot},
    render_scale::RenderScale,
    sampler_state,
    scratch::ScratchArena,
    shader_cache::{self, CachedKind},
    shader_compile_stats::{self, BurstTracker, CompileBucket},
    storage_policy::buffer_storage_mode,
    stretch_rect::StretchRegion,
    upload_pass::UploadDecode,
    upload_recovery::{UploadFate, UploadRecoveryQueue},
    visibility::{
        MAX_SLOTS, RetiredVisibilityBuffer, SLOT_BYTES, VisibilityQueryCore, VisibilityQueryState,
    },
};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, BufferCreateDesc, Command, CommandType,
    CompileShaderLibraryParams, CopyBufferToBufferInfo, CopyBufferToTextureInfo,
    CreateBuffersBatchParams, CreateTextureSliceViewParams, CreateTexturesBatchParams,
    DestroyResourcesBulkParams, EnsureBlitPipelineParams, EnsureClearQuadPipelineParams,
    ExtraColorDesc, GetTaskFaultsParams, MetalHandle, PassDescriptor, SetDisplaySyncEnabledParams,
    SubmitFrameParams, TextureCreateDesc, VertexAttrDesc, WaitForGpuRetireParams,
    mtl::{
        BufferKind, ClearQuadFlags, CullMode, DepthResolveFilter, DestroyKind, LoadAction,
        PixelFormat, PrimitiveType, QuadPipelineKind, StageTag, StorageMode, StoreAction, Swizzle,
        TextureCreateFlags, TextureUsage, VisibilityResultMode,
    },
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLCommandQueueKind, MTLDepthStencilStateKind,
        MTLDeviceKind, MTLFunctionKind, MTLRenderPipelineStateKind, MTLSamplerStateKind,
        MTLTextureKind, NSViewKind,
    },
    tsc::{ns_to_cycles, rdtsc, secs_to_cycles},
};
use mtld3d_types::{D3DSAMP_MIPMAPLODBIAS, SAMPLER_STATE_COUNT};
// Fast non-cryptographic hasher for the per-draw resource caches below
// (texture/lib/pipeline/sampler/buffer/...). Keys are small trusted integers
// or fixed structs; SipHash's DoS resistance buys nothing here and its
// per-probe cost shows up in the encoder `resolve`/`binds`/`samplers` phases.
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    LOG_TARGET,
    device::PendingVbibRetention,
    draw::{
        self, CurrentSnapshotPtr, DrawOp, IndexSource, PsKey, PsSource, ScratchSlice, ShaderRef,
        VertexSource, VsSource,
    },
    shader_bindings::CONSTANT_ROWS,
    unix_call::unix_call,
};

/// Sub-target for the per-draw breadcrumb emitted by `FrameEncoder::maybe_emit_draw_trace`.
///
/// Sits under `mtld3d::d3d9::*` so `RUST_LOG=mtld3d::d3d9::draw=trace`
/// opts in granularly without flipping the rest of the d3d9 logger. MSL
/// dumps reuse `mtld3d_core::dxso::LOG_TARGET` (re-exported above as
/// `MSL_TRACE_TARGET`) so the emitter and its output share one knob.
const DRAW_TRACE_TARGET: &str = "mtld3d::d3d9::draw";

/// Sub-target for the once-per-distinct sampler-state diagnostic.
///
/// Emitted from `get_or_create_sampler`. Permanent probe (zero-cost when
/// off); gated under its own sub-target so a sampler-state investigation
/// can `RUST_LOG=mtld3d::d3d9::sampler=trace` without flipping the
/// per-draw breadcrumb's flood.
const SAMPLER_TRACE_TARGET: &str = "mtld3d::d3d9::sampler";

/// Sub-target for the depth-path diagnostic probes.
///
/// Here, the per-render-pass depth-attachment load action emitted from
/// `submit`. Permanent probe (zero-cost when off);
/// `RUST_LOG=mtld3d::d3d9::depth=trace` opts in. Mirrored as
/// `device.rs::DEPTH_TRACE_TARGET`.
const DEPTH_TRACE_TARGET: &str = "mtld3d::d3d9::depth";

/// Sub-target for the `StretchRect` blit-path diagnostic.
///
/// Mirrors `device.rs::BLIT_TRACE_TARGET`; the scaling-`StretchRect`
/// render path lives on the encoder thread so its trace is emitted from
/// here. `RUST_LOG=mtld3d::d3d9::blit=trace` opts in.
const BLIT_TRACE_TARGET: &str = "mtld3d::d3d9::blit";

type EncoderFn = Box<dyn FnOnce(&mut FrameEncoder) + Send>;

/// Number of [`FramePayload`]s allowed to exist.
///
/// One is being built while up to one is in flight on the submit thread;
/// a third request blocks on the return channel. This bounds render-ahead
/// to ≤1 frame (the encoder can be at most one finalize ahead of the
/// submit stage) and caps the pooled buffer memory at two payload sets.
const SUBMIT_PAYLOAD_CAP: u32 = 2;

/// `u16` view of [`CONSTANT_ROWS`] for the populated-rows watermark arithmetic.
///
/// Used by `apply_{vs,ps}_const_range`. Defined as a `u16` literal and
/// cross-checked against `CONSTANT_ROWS` below, so a change to the row
/// count is a compile error rather than a silent truncating `as` cast.
const CONSTANT_ROWS_U16: u16 = 256;
const _: () = assert!(CONSTANT_ROWS == CONSTANT_ROWS_U16 as usize);

/// Wire size of an `f32` clear-depth scratch entry.
///
/// Used by `emit_clear_quad_*` to size the `setVertexBytes` command
/// without a runtime `.len() as u32` cast (the value is a compile-time
/// constant of the depth path's IEEE-754 little-endian encoding).
const F32_BYTE_LEN: u32 = 4;

/// Wire size of the `float4` clear-color scratch entry.
///
/// For the color-quad fragment shader's `[[buffer(0)]]` uniform.
const RGBA_BYTE_LEN: u32 = 16;

/// Discriminated union over the work the API thread queues for the encoder.
///
/// The hot per-draw path uses `SetCurrentSnapshot` (one push per dirty
/// draw) + `Draw` — both inline, no per-op heap allocation. `Closure` is
/// the escape hatch for the long tail of non-draw work (RT swap, blit,
/// clear, upload, present, mid-frame submit, …).
///
/// See `windows/core/src/scratch.rs` for why hot payloads (snapshots,
/// const ranges, stage bindings) are pointers into the per-frame arena
/// rather than `Box<T>`.
pub enum Op {
    /// Replace `FrameEncoder.current_snapshot` wholesale with the scratch-allocated snapshot.
    ///
    /// Pushed once per dirty draw — every field is populated by
    /// `emit_snapshot_deltas`, so the encoder just memcpys the pointee
    /// into its `current_snapshot`.
    SetCurrentSnapshot(CurrentSnapshotPtr),
    /// Apply a delta into the encoder-side VS programmable constant mirror.
    ///
    /// `data` is a scratch-allocated `[u8]` of `rows × 16` bytes starting
    /// at row `start_row`. Pushed by `SetVertexShaderConstantF` (and
    /// state-block-apply sites) on the API thread; consumed by `run_frame`
    /// which copies the bytes into `FrameEncoder::vs_constants_mirror`.
    SetVsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    SetPsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    /// Section delta into the FF VS const mirror.
    ///
    /// Pushed once per dirty `FfVsDirty` section from
    /// `emit_ff_vs_section_deltas`. Structurally identical to
    /// `SetVsConstRange` but routes to a separate mirror because FF and
    /// programmable VS feed different content into the same shader slot
    /// (slot 0 c-bank).
    SetFfVsConstRange {
        start_row: u16,
        rows: u16,
        data: ScratchSlice,
    },
    /// Issue a draw using the current snapshot.
    Draw(DrawOp),
    /// Long-tail escape hatch: arbitrary closure for non-draw work.
    Closure(EncoderFn),
    /// Inline op-stream-ordered `Staged` VB/IB upload.
    ///
    /// Carries the transient `PageBox` snapshot of the bytes the game
    /// wrote between `Lock` and `Unlock` (taken on the API thread — no
    /// Metal thunk there). The encoder wraps it as a `bytesNoCopy` blit
    /// source and copies its range into the buffer's persistent `Private`
    /// device buffer via `frame_blit_commands` (a leading phase, before
    /// any draw). If a draw earlier this frame already read an overlapping
    /// region, the encoder first renames the device buffer so the earlier
    /// draw keeps its bytes — see [`FrameEncoder::apply_stage_upload`].
    StageUpload {
        buffer_id: BufferId,
        page_box: PageBox,
        dst_offset: u32,
        size: u32,
    },
}

/// Render target 0 as the encoder binds it.
///
/// A parameter bag rather than eight positional arguments. `logical_size` is
/// the extent D3D9 reports and `scale` what it is rasterized at; `msaa_texture`
/// is the multisampled companion the pass attaches, NULL for a single-sampled
/// target, and `sample_count` its count.
pub struct ColorRtBinding {
    pub texture: MetalHandle<MTLTextureKind>,
    pub msaa_texture: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `msaa_texture`, NULL whenever that is.
    pub msaa_srgb_texture: MetalHandle<MTLTextureKind>,
    pub sample_count: u8,
    pub logical_size: (u32, u32),
    pub format: PixelFormat,
    pub has_alpha: bool,
    pub scale: RenderScale,
    /// `(slice, level)` of the attachment.
    pub subresource: (u32, u32),
}

/// One side (source or destination) of a scaled blit, for [`FrameEncoder::stretch_blit_scaled`].
pub struct BlitSide {
    pub handle: u64,
    pub rect: mtld3d_core::stretch_rect::StretchRegion,
    pub dims: (u32, u32),
    /// Mip level of `handle` the blit reads or writes.
    pub mip: u32,
    /// Array slice of `handle` the blit reads or writes.
    ///
    /// `Some(face)` when the endpoint is one face of a cube map, `None` when
    /// its backing texture holds a single slice. The destination attaches the
    /// face as the colour attachment's slice; the source is sampled through a
    /// 2D view of it, since a `texturecube` binding would read face 0.
    pub slice: Option<u32>,
    /// Multisampled companion of `handle`, NULL when there is none.
    ///
    /// Read on the destination side only: the quad renders into the
    /// multisampled attachment and the pass resolves into `handle`. A
    /// multisampled *source* is read through `handle`, which the resolve has
    /// already filled.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion, NULL whenever the companion is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the destination, 1 when it is single-sampled.
    pub sample_count: u8,
}

/// One back-buffer `ReleaseDC` write-back that has to change size on the way in.
///
/// `GetDC` hands the DIB out at the extent D3D9 reports, so under a
/// `render.scale` below 100% the page GDI drew into is larger than the texture
/// it belongs in. Built by `surface.rs` on the API thread, which is where the
/// device's scale and the back buffer's extent are both reachable.
pub struct ResampledUpload {
    /// Destination colour `MTLTexture`.
    pub color_handle: u64,
    /// Metal format of the destination, and so of the staging source too.
    pub format: PixelFormat,
    /// Extent the `tight` rows describe, which is the extent D3D9 reports.
    pub logical: (u32, u32),
    /// Extent of the destination texture, at or below `logical`.
    pub texture: (u32, u32),
    /// Bytes per pixel of `format`.
    pub bytes_per_pixel: u32,
    /// Multisampled companion of the destination, null when single-sampled.
    pub msaa: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `msaa`, null whenever `msaa` is.
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
    /// Sample count of the destination; 1 without a companion.
    pub sample_count: u8,
}

/// One `ColorFill` against a render-target texture, resolved on the API thread.
///
/// Built by `device_color_fill` and consumed by
/// [`FrameEncoder::color_fill_target`] on the encoder thread, which cannot
/// reach the device to re-derive the destination's scale or extent.
pub struct ColorFillTarget {
    /// Destination `MTLTexture`.
    pub texture: MetalHandle<MTLTextureKind>,
    /// Mip extent as D3D9 reports it; `scale` converts it to the texture's own.
    pub logical_size: (u32, u32),
    /// Metal format of the destination as it was created on this device.
    pub format: PixelFormat,
    /// What the destination is rasterized at relative to `logical_size`.
    pub scale: RenderScale,
    /// `(array slice, mip level)` of the destination subresource.
    pub subresource: (u32, u32),
    /// Fill rect in D3D9 coordinates, as `(x, y, width, height)`.
    pub rect: (u32, u32, u32, u32),
    /// Fill colour, one `f32::to_bits` per channel in RGBA order.
    pub rgba: (u32, u32, u32, u32),
    /// True when the destination is level 0 of a `D3DUSAGE_AUTOGENMIPMAP` texture.
    ///
    /// The runtime owns that texture's mip chain, so the fill is followed by a
    /// regeneration from the level it just painted.
    pub regenerate_mipmaps: bool,
}

/// Parameter bag for `FrameEncoder::run_texture_upload`.
///
/// Built by `texture::schedule_upload` on the API thread and consumed by
/// the upload closure on the encoder thread; keeps the encoder method
/// signature to a single argument.
pub struct TextureUploadJob {
    pub info: TextureInfo,
    pub arc: Arc<PageBox>,
    pub level: u32,
    /// Destination array slice.
    ///
    /// Zero for ordinary textures; cube uploads use the face index.
    pub destination_slice: u32,
    /// Index in the texture's staging-buffer cache.
    ///
    /// Equal to `level` for ordinary textures and `face * levels + level` for
    /// cubes.
    pub staging_index: usize,
    pub origin_x: u32,
    pub origin_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    pub src_d3d_format: u32,
    pub src_pitch: u32,
    pub bytes_per_pixel: u32,
    /// Slice count for this mip.
    ///
    /// 1 for a 2D texture, `(depth >> level)` (≥1) for a volume (3D)
    /// texture. Selects the 2D vs volume blit path in `run_texture_upload`.
    pub depth: u32,
    /// Byte stride between slices (the box slice pitch).
    ///
    /// Only read by the volume blit path; the 2D path derives its
    /// single-slice `bytes_per_image` from the region's block-row count
    /// instead.
    pub slice_pitch: u32,
}

/// Per-mip `MTLBuffer` wrapper for texture staging.
///
/// `handle = 0` means "not yet created". `backing_ptr` tracks which
/// PE-heap Box this `MTLBuffer` wraps — if the PE-side staging Arc is
/// replaced (which happens under the DISCARD-contended /
/// default-contended paths), the next upload sees a different `as_ptr()`
/// and we re-create the wrapper to target the fresh backing.
///
/// `keepalive` holds the PE-side `Arc<PageBox>` for the entire
/// lifetime of `handle`'s `MTLBuffer` wrapper. `texture_release` drops
/// the `TextureInner.staging` Arc synchronously on the API thread, so
/// without our own clone the `MTLBuffer` would wrap freed pages
/// between "queue for destroy" and the eventual bulk-destroy after
/// GPU retire. The clone moves into the matching
/// `PendingResourceRetention.staging_arc` when the slot is parked.
#[derive(Clone, Default)]
pub struct MipStagingBuffer {
    pub handle: MetalHandle<MTLBufferKind>,
    pub backing_ptr: u64,
    pub length: u64,
    pub keepalive: Option<Arc<PageBox>>,
}

/// A device-shared `AtomicU64` counter reached across the encoder boundary by its raw address.
///
/// The API side keeps the counter in an `Arc<AtomicU64>` that outlives every
/// encoder; the encoder stores only the `u64` address (also forwarded verbatim
/// across the PE/Unix boundary) and recovers a typed handle at each access, so
/// the raw-pointer deref lives behind one contract instead of being repeated at
/// every call site.
#[repr(transparent)]
struct SharedCounter(*const AtomicU64);

impl SharedCounter {
    /// # Safety
    ///
    /// `raw` must be the non-zero address of a live `AtomicU64` owned by an
    /// `Arc` that outlives the returned handle — the device-side counter `Arc`,
    /// whose raw pointer the API thread seeds into the frame.
    const unsafe fn new(raw: u64) -> Self {
        Self(raw as *const AtomicU64)
    }

    fn load(&self, order: Ordering) -> u64 {
        // SAFETY: `SharedCounter::new`'s contract — `self.0` is a live
        // `AtomicU64` for the handle's lifetime.
        unsafe { &*self.0 }.load(order)
    }

    fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        // SAFETY: `SharedCounter::new`'s contract.
        unsafe { &*self.0 }.fetch_add(val, order)
    }

    fn fetch_sub(&self, val: u64, order: Ordering) -> u64 {
        // SAFETY: `SharedCounter::new`'s contract.
        unsafe { &*self.0 }.fetch_sub(val, order)
    }
}

/// One entry of `FrameEncoder::sampler_resolve_memo`.
///
/// The raw D3D9 sampler-state words a stage last resolved, and the
/// `MTLSamplerState` handle that resolve produced. Compared wholesale
/// (14 words + the compare flag) — cheaper than rebuilding the
/// snapshot + `SamplerKey` and probing `sampler_cache` on every draw.
struct SamplerResolveMemo {
    state: [u32; SAMPLER_STATE_COUNT],
    is_compare: bool,
    handle: u64,
}

/// Per-texture encoder-thread state.
///
/// Owns the `MTLTexture` handle and one `MTLBuffer` wrapper per mip that
/// wraps the PE-heap staging `PageBox` via `newBufferWithBytesNoCopy`.
/// `mip_staging_buffers` is sized to the texture's `levels` count but
/// entries stay unpopulated (`handle == 0`) until the first upload for
/// that mip.
/// A scratch texture staging a `StretchRect` whose two endpoints are one texture.
///
/// Sized to the largest region asked of it so far, so a game that scrolls the
/// same surface every frame allocates once. Keyed by the source handle in
/// `stretch_scratch`.
struct StretchScratch {
    handle: MetalHandle<MTLTextureKind>,
    width: u32,
    height: u32,
    format: PixelFormat,
}

/// A scratch copy of a depth attachment handed to draws that sample it.
struct DepthSnapshot {
    handle: MetalHandle<MTLTextureKind>,
    width: u32,
    height: u32,
    format: mtld3d_shared::mtl::PixelFormat,
    /// `depth_write_epoch` the copy reflects.
    epoch: u64,
}

pub struct TextureGpuState {
    pub mtl_texture: MetalHandle<MTLTextureKind>,
    /// Eager sRGB twin view of `mtl_texture` (NULL when the format has none).
    ///
    /// Bound instead of the base handle when the sampling stage has
    /// `D3DSAMP_SRGBTEXTURE=1`, so the hardware performs the sRGB→linear
    /// decode. Same storage as the base texture — uploads and blits keep
    /// targeting `mtl_texture` and are visible through this view.
    pub mtl_texture_srgb: MetalHandle<MTLTextureKind>,
    pub mip_staging_buffers: Vec<MipStagingBuffer>,
}

/// Frame-lifetime retention for blit-source PE-heap staging.
///
/// Each entry keeps one `Arc<PageBox>` alive from blit-encode time
/// through GPU retirement of the owning command buffer, keyed by the
/// frame's `submit_seq`. Drained FIFO in `begin_frame` once the seq is ≤
/// `coherent_seq`.
struct PendingBlitArc {
    submit_seq: u64,
    arc: Arc<PageBox>,
}

impl PendingBlitArc {
    const fn new(submit_seq: u64, arc: Arc<PageBox>) -> Self {
        Self { submit_seq, arc }
    }

    /// Strong-count probe used by the reclaim loop's debug checks.
    ///
    /// Also ensures the `arc` field stays `#[warn(dead_code)]`-clean
    /// — the field's real job is to keep the Box alive until drop,
    /// which rustc doesn't count as a "read".
    fn strong_count(&self) -> usize {
        Arc::strong_count(&self.arc)
    }

    /// Byte length of the retained staging Box.
    ///
    /// Used by the reclaim loop to decrement `tex_staging_retained_bytes`
    /// by the exact amount the matching submit-time push added.
    fn byte_len(&self) -> usize {
        self.arc.len()
    }
}

/// Inputs every slice of one texture upload's passes shares.
///
/// `mip_size` is the destination mip's extent, `level` its mip index;
/// `src_pitch` is the staging row stride in bytes and `decode` the source
/// layout the upload quad's fragment function reads it with.
struct UploadPassInputs {
    pipeline: u64,
    depth_state: u64,
    staging_buffer_handle: u64,
    texture_handle: u64,
    format: PixelFormat,
    level: u32,
    mip_size: (u32, u32),
    src_pitch: u32,
    decode: UploadDecode,
}

/// Texture metadata captured from the API thread for deferred Metal creation.
#[derive(Clone)]
pub struct TextureInfo {
    pub texture_id: TextureId,
    /// `D3DFMT_*` the game created the texture with.
    ///
    /// Kept alongside `pixel_format` because the pair is what decides how an
    /// upload reaches the texture: the same `Bgra8Unorm` backs an
    /// `A8R8G8B8` source verbatim and a packed 16-bit source through the
    /// widening upload pass.
    pub d3d_format: u32,
    pub width: u32,
    pub height: u32,
    /// Slice count: 1 for 2D textures, >1 for a volume (3D) texture.
    pub depth: u32,
    pub levels: u32,
    pub pixel_format: PixelFormat,
    pub create_flags: TextureCreateFlags,
    pub swizzle: [Swizzle; 4],
    /// `TextureUsage` bits passed through to the unix side.
    ///
    /// The Metal texture is allocated with `RenderTarget` usage when the
    /// D3D9 texture was created with `D3DUSAGE_RENDERTARGET`.
    pub usage_flags: TextureUsage,
}

/// Resolved handles for a compiled per-stage MSL library.
///
/// Each library contains a single entry point (`mtld3d_vs` for VS libraries,
/// `mtld3d_ps` for PS libraries). Both handles are retained so encoder shutdown
/// can release them — pipelines hold strong refs to the function, the function
/// holds a strong ref to its library; we destroy functions before libraries so
/// the refcount graph drains leaf-first.
#[derive(Clone, Copy)]
pub struct StageLibHandles {
    pub library: MetalHandle<mtld3d_shared::mtl_handle::MTLLibraryKind>,
    pub func: MetalHandle<MTLFunctionKind>,
}

// ── FrameEncoder — persistent context that closures execute against ──
//
// Persists across frames on the encoder thread. `begin_frame()` resets
// per-frame state (commands, scratch) while preserving caches.

/// Owns every per-frame buffer the unix `SubmitFrame` thunk reads via raw pointer.
///
/// The pointers in `SubmitFrameParams` and in each `PassDescriptor` alias into
/// `scratch`, `passes`, `descriptors`, `frame_blit_commands`, and
/// `trailing_blits`, so the whole payload must stay alive and unmutated for the
/// full duration of that thunk. It is detached from the encoder at submit
/// (`finalize_submit`) by O(1) `Vec`/arena swaps — the heap behind each field
/// never moves, so the raw pointers stay valid wherever the payload travels —
/// and recycled afterwards (`reclaim_payload`) so steady-state frames allocate
/// nothing here. In `Async` mode the payload crosses to the dedicated submit
/// thread; in `Sync` mode the thunk runs inline on the encoder thread.
#[derive(Default)]
struct FramePayload {
    /// Per-frame shader-constant / `DrawPrimitiveUP` scratch.
    ///
    /// Pointers to its chunks are embedded in `Command`s inside `passes`.
    scratch: ScratchArena,
    /// The frame's finalized passes, each owning its `commands` and `leading_blits`.
    ///
    /// Taken from `PassState`. `descriptors` point into these.
    passes: Vec<Pass>,
    /// One `PassDescriptor` per pass (plus an optional trailing blit-only pass).
    ///
    /// `SubmitFrameParams.passes_ptr` aliases this vec's backing.
    descriptors: Vec<PassDescriptor>,
    /// Frame-leading blits (texture uploads, GPU preserves, notifies).
    ///
    /// `SubmitFrameParams.blit_commands_ptr` aliases this vec's backing.
    frame_blit_commands: Vec<BlitCommand>,
    /// `StretchRect` blits queued after the last draw of the frame.
    ///
    /// Carried by the synthetic trailing `PassDescriptor`.
    trailing_blits: Vec<BlitCommand>,
}

/// How `submit` runs the `SubmitFrame` thunk for one frame.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SubmitMode {
    /// Hand the finalized payload to the dedicated submit thread and return immediately.
    ///
    /// Overlaps the unix command-walk + present with the next frame's
    /// build. The normal Present path.
    Async,
    /// Run the `SubmitFrame` thunk inline on the encoder thread and block until it returns.
    ///
    /// Used after a submit-thread barrier for the rare paths that need the
    /// command buffer committed before they proceed (mid-frame readback,
    /// GPU capture, device reset, shutdown).
    Sync,
}

/// One frame's finalized work handed to the submit thread.
///
/// Sent through the work channel by value (no extra `Box`): the per-frame
/// handoff stays alloc-free, and the channel slot carries the struct inline.
struct SubmitPacket {
    params: SubmitFrameParams,
    payload: FramePayload,
    /// The frame's `FrameData`, kept alive until the replay finishes.
    ///
    /// Several per-draw fragment-bytes Commands (fog color, alpha ref, FF
    /// pixel constants) point into `FrameData::scratch` — bumped by the API
    /// thread, not copied into the encoder's payload-isolated scratch — and
    /// the unix-side replay reads those pointers at encode time. Dropped on
    /// the submit thread only after `execute_submit` returns. (Inline draw
    /// data — UP vertices, VS/PS const slices — is copied into the payload's
    /// scratch and isn't affected.)
    frame: Box<FrameData>,
}

/// A finished frame coming back from the submit thread.
///
/// The payload (for recycling) plus the unix-side status and the
/// drawable-wait time the `SubmitFrame` thunk measured.
struct ReturnedPayload {
    payload: FramePayload,
    status: i32,
    /// `nextDrawable` wait in nanoseconds, as the unix side measured it.
    ///
    /// Nanoseconds because the two sides do not share a cycle counter; it
    /// becomes our cycles via `ns_to_cycles` when folded into perf.
    drawable_wait_ns: u64,
    /// Total submit-thread CPU for `execute_submit`.
    ///
    /// Covers the unix command-walk, present, and commit — including the
    /// `drawable_wait_ns` portion. Folded into perf so the summary can
    /// show the submit thread's own cost; `submit_exec - drawable_wait` is
    /// the encode+commit CPU.
    submit_exec_tsc: u64,
}

/// The dedicated submit thread.
///
/// Drains `SubmitFrame` work items, issues the thunk (the unix
/// command-walk + `nextDrawable` + present + commit — the part that would
/// otherwise block the encoder thread), and returns each payload for
/// recycling. Exits when the encoder drops the work channel at teardown
/// (`recv` returns `Err`).
fn submit_thread_main(
    work_rx: &mpsc::Receiver<SubmitPacket>,
    return_tx: &mpsc::Sender<ReturnedPayload>,
) {
    mtld3d_shared::crumb::init();
    while let Ok(packet) = work_rx.recv() {
        mtld3d_shared::crumb!("phase:SubmitExec");
        let SubmitPacket {
            params,
            payload,
            frame,
        } = packet;
        let mut submit_exec_tsc: u64 = 0;
        let (payload, status, drawable_wait_ns) = {
            let _exec = mtld3d_core::perf::CycleSetTimer::start(&raw mut submit_exec_tsc);
            execute_submit(params, payload)
        };
        // The replay copied every `FrameData::scratch`-resident byte into
        // the command buffer at encode time, so the frame can drop now.
        drop(frame);
        if return_tx
            .send(ReturnedPayload {
                payload,
                status,
                drawable_wait_ns,
                submit_exec_tsc,
            })
            .is_err()
        {
            break;
        }
    }
}

bitflags::bitflags! {
    /// Assorted per-`FrameEncoder` boolean state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FrameEncoderFlags: u8 {
        /// Latched whenever an *encoder-bound* blit command is pushed into `frame_blit_commands`.
        ///
        /// Encoder-bound covers any CopyBuffer/Texture variant.
        /// `NotifyBufferDidModifyRange` does NOT flip it because the unix
        /// dispatcher calls that one outside any encoder. Read at submit to
        /// fill `SubmitFrameParams.blit_commands_need_encoder`, so the unix
        /// side can skip `MTLBlitCommandEncoder` creation on pure-notify
        /// frames. Reset in `begin_frame` alongside the Vec clear.
        const BLIT_CMDS_NEED_ENCODER = 1 << 0;
        /// Set once the pre-warm payload has been ingested.
        ///
        /// Gates lazy file opening on first miss-compile so no records are
        /// written before pre-warm validates / wipes the file's header.
        const CACHE_READY = 1 << 1;
        /// Latched when the disk cache is disabled at startup (`shaderCache.enable = false`).
        ///
        /// Also latched when any open/write failure makes further attempts
        /// pointless.
        const CACHE_DISABLED = 1 << 2;
    }
}

/// The sampler slots a programmable shader declares.
///
/// `mask` has a bit set for every stage the shader declares a sampler for. The
/// declaration decides only which slots exist: both emitters type each one
/// from the texture bound to it, so the draw path takes the kind of the
/// opaque-black fallback it binds to an unbound slot from the same live
/// bindings rather than from here.
#[derive(Clone, Copy, Default)]
pub struct PsSamplerDecls {
    mask: u16,
}

/// One vertex-sampler slot's binding, mirrored from the device.
struct VertexTexBinding {
    /// Bound texture, or `None` for an empty slot.
    texture_id: Option<mtld3d_core::ids::TextureId>,
    sampler_state: [u32; SAMPLER_STATE_COUNT],
}

impl Default for VertexTexBinding {
    fn default() -> Self {
        Self {
            texture_id: None,
            sampler_state: mtld3d_types::sampler_state_defaults(),
        }
    }
}

impl PsSamplerDecls {
    /// Collect the declared samplers from a parsed program (empty for a VS).
    ///
    /// Uses `declared_ps_samplers`, the same source the emitter builds the
    /// fragment-function signature from, so the bind side cannot drift from it.
    /// Stages at or past `STAGE_COUNT` are ignored (a D3D9 PS declares s0..s15).
    fn from_program(program: &DxsoProgram) -> Self {
        let mut decls = Self::default();
        for &slot in declared_ps_samplers(program).keys() {
            let slot = slot as usize;
            if slot >= crate::stage_bindings::STAGE_COUNT {
                continue;
            }
            decls.mask |= 1u16 << slot;
        }
        decls
    }

    /// Slots this shader declares a sampler for but `bound_mask` leaves unbound.
    #[must_use]
    pub const fn unbound(self, bound_mask: u16) -> u16 {
        self.mask & !bound_mask
    }

    /// Every slot this shader declares a sampler for.
    ///
    /// The draw path binds a texture and a sampler only inside this mask: a
    /// stage the game bound a texture to that the shader declares no sampler
    /// for is a binding no draw reads.
    #[must_use]
    pub const fn mask(self) -> u16 {
        self.mask
    }
}

pub struct FrameEncoder {
    /// Pass-management state (passes, pending clears, current attachments).
    ///
    /// See `mtld3d_core::passes::PassState`.
    pass_state: PassState,
    /// Per-pass last-bound state cache.
    ///
    /// Skips redundant fragment-sampler / fragment-texture / pipeline /
    /// depth-stencil / cull-mode emissions for draws that share state with
    /// the previous draw in the same Metal render encoder. Reset on every
    /// new-pass entry from `begin_render_pass_if_needed`.
    last_bound: LastBoundCache,
    /// Per-frame scratch arena for API→encoder copies.
    ///
    /// Shader constants and `DrawPrimitiveUP` inline vertices. A chunked
    /// bump: existing chunks never move, so pointers handed out earlier in
    /// the frame stay valid for the unix-side read during `SubmitFrame`. A
    /// single growing `Vec<u8>` would reallocate and silently invalidate
    /// those pointers. `clear()` at `begin_frame` retains the hot chunk, so
    /// steady-state frames allocate 0 chunks here.
    scratch: ScratchArena,
    /// Leading blit commands accumulated during the frame.
    ///
    /// Texture uploads, GPU-side preserves, non-UMA `didModifyRange:`
    /// notifies. Replayed inside a single `MTLBlitCommandEncoder` before
    /// any render pass. Stable backing for
    /// `SubmitFrameParams.blit_commands_ptr`.
    frame_blit_commands: Vec<BlitCommand>,
    /// Assorted encoder booleans (`BLIT_CMDS_NEED_ENCODER` / `CACHE_READY` / `CACHE_DISABLED`).
    ///
    /// See [`FrameEncoderFlags`].
    flags: FrameEncoderFlags,
    /// Free-list of recycled [`FramePayload`]s.
    ///
    /// `finalize_submit` pops one (or default-allocates) to swap the live
    /// per-frame buffers into; `reclaim_payload` clears a finished payload
    /// and pushes it back. `Sync` mode holds one entry (the payload returns
    /// within the same frame); `Async` mode lets a second be in flight on the
    /// submit thread, which bounds the pool's steady-state size to ~2.
    payload_pool: Vec<FramePayload>,
    /// Work channel to the dedicated submit thread (`Async` mode).
    ///
    /// Dropping it at encoder teardown is what tells the submit thread to
    /// exit.
    submit_work_tx: mpsc::SyncSender<SubmitPacket>,
    /// Finished payloads coming back from the submit thread for recycling.
    submit_return_rx: mpsc::Receiver<ReturnedPayload>,
    /// Packets sent to the submit thread but not yet returned.
    ///
    /// The barrier (`drain_submit_thread`) blocks until this reaches zero.
    submit_in_flight: u32,
    /// How many `FramePayload`s have been created, capped at [`SUBMIT_PAYLOAD_CAP`].
    ///
    /// Once at the cap, `acquire_clean_payload` blocks on the return
    /// channel instead of allocating a new one.
    submit_payloads_total: u32,
    /// Most recent `SubmitFrame` status, folded in when a payload returns.
    ///
    /// The `Async` per-frame perf summary reports this (lagged ≤1 frame).
    last_submit_status: i32,
    /// Whether the previous submit was a mid-frame flush (`NO_PRESENT`).
    ///
    /// Set in `finalize_submit` from the frame's flags, read in the next
    /// `begin_frame` so `PassState::reset_frame` keeps the seen-rt sets when
    /// the D3D9 frame did not actually end at the flush. Encoder-thread only.
    prev_submit_no_present: bool,
    /// Arc clones of staging `PageBoxes` referenced by blits emitted this frame.
    ///
    /// Moved into `pending_blit_retention` at submit time with the frame's
    /// `submit_seq`; drained from `pending_blit_retention` in `begin_frame`
    /// once `coherent_seq` catches up.
    current_blit_retention: Vec<Arc<PageBox>>,
    /// Prior frames' retained staging Arcs, keyed by `submit_seq`.
    ///
    /// Entries drop when their seq is ≤ the latest `coherent_seq`.
    pending_blit_retention: VecDeque<PendingBlitArc>,
    /// Pointer to the shared `coherent_seq` atomic, copied from `FrameData` in `begin_frame`.
    ///
    /// Read on the encoder thread to drain `pending_blit_retention`. 0
    /// means "not yet seeded" — the very first frame has no retention to
    /// drain.
    coherent_seq_ptr: u64,
    /// Pointer to the shared `failed_submit_seq` atomic, copied from `FrameData` in `begin_frame`.
    ///
    /// Read next to `coherent_seq_ptr` when the upload-recovery queues
    /// settle: a seq that retired at or below this one had its command
    /// buffer discarded, so its upload has to be re-issued rather than
    /// freed. 0 means "not yet seeded".
    failed_seq_ptr: u64,
    /// Pointer to the shared `upload_coherent_seq` atomic, copied from `FrameData`.
    ///
    /// The upload command buffer is the one that actually carries an
    /// upload's copy, and its own completion handler is the only thing that
    /// ever moves this counter. `coherent_seq` can be hand-advanced by
    /// `wait_for_gpu_retire` before that handler has run, so an upload is
    /// only settled once both counters have reached its seq. 0 means "not
    /// yet seeded", or the defensive path where the leading blits rode the
    /// draw command buffer instead.
    upload_coherent_seq_ptr: u64,
    /// `Staged` VB/IB dirty-range uploads the GPU has not acknowledged yet.
    ///
    /// Each entry owns the transient `bytesNoCopy` wrapper and the
    /// PE-heap snapshot the blit reads, so settling one either frees both
    /// or re-emits the copy from them. Replaces what used to be a plain
    /// `PendingResourceRetention` entry: destroying at a seq and replaying
    /// at a seq are different policies, and one front-gated loop cannot
    /// hold both.
    pending_stage_uploads: UploadRecoveryQueue<StagedUploadRetry>,
    /// Texture mip uploads the GPU has not acknowledged yet.
    ///
    /// Holds the whole `TextureUploadJob`, whose `staging_arc` is a clone
    /// of the texture's own persistent staging, so a replay costs one blit
    /// and no extra memory.
    pending_texture_uploads: UploadRecoveryQueue<TextureUploadJob>,
    /// Pointer to the shared `vbib_retained_bytes` atomic (device-owned).
    ///
    /// Copied from `FrameData` in `begin_frame`. `fetch_add`'d when a
    /// `PageBox` enters retention and `fetch_sub`'d when one drains, so
    /// the API thread can cap retention. 0 means "not yet seeded".
    retained_bytes_ptr: u64,
    /// Submit seq for the frame currently being encoded.
    ///
    /// Stashed here in `begin_frame` so VB/IB wrap helpers can record
    /// "this frame used the cache entry" on the cache entry for retention
    /// keying.
    current_submit_seq: u64,

    // Per-frame config (seeded by begin_frame).
    backbuffer_width: u32,
    backbuffer_height: u32,
    /// Size and pixel format of the bound depth attachment.
    ///
    /// Read by `depth_snapshot_for_sampling` to size its copy.
    depth_attachment_desc: (u32, u32, mtld3d_shared::mtl::PixelFormat),
    /// Bumped by every depth-writing draw and every depth clear.
    ///
    /// A snapshot taken under an older value is stale.
    depth_write_epoch: u64,
    /// Scratch copies of depth attachments that draws sampled while bound, by source handle.
    depth_snapshots: FxHashMap<u64, DepthSnapshot>,
    /// Scratch textures staging same-texture `StretchRect` copies, by source handle.
    stretch_scratch: FxHashMap<u64, StretchScratch>,

    /// Every per-frame and rolling telemetry field.
    ///
    /// TSC buckets, per-category API timers, Lock / wrap / destroy
    /// counters, and the 5-second `PerfWindow` aggregator. See
    /// `mtld3d_core::perf` for the full field list.
    perf: EncoderPerfState,

    /// Set of `(rt_handle, vs_id, ps_id)` tuples already logged by `maybe_log_pass_shader`.
    ///
    /// Lets one `debug` run emit one line per unique triple. Keyed on the
    /// Metal texture handle (not RT size) so distinct render targets that
    /// happen to share dimensions stay distinguishable. Lives on
    /// `FrameEncoder` rather than the perf struct because the log itself
    /// is a shader-debug aid (target `mtld3d::d3d9`), not perf telemetry.
    pass_shader_log_fired: FxHashSet<(MetalHandle<MTLTextureKind>, PairShaderId, PairShaderId)>,

    /// Captured at encoder spawn from `MTLDevice` queries.
    ///
    /// Drives storage-mode policy (`Shared` vs `Managed`), texture-buffer
    /// alignment, and the `didModifyRange:` enqueue gate.
    gpu_caps: GpuCaps,

    // Persistent caches (survive across frames)
    device_handle: MetalHandle<MTLDeviceKind>,
    depth_stencil_cache: FxHashMap<DepthStencilKey, MetalHandle<MTLDepthStencilStateKind>>,
    pipeline_cache: FxHashMap<PipelineKey, MetalHandle<MTLRenderPipelineStateKind>>,
    /// Per-format-combo "clear-quad" pipeline handles.
    ///
    /// One entry per `(depth_format, color_format, has_color, has_stencil)`
    /// combo. Used by the mid-pass `Clear` translation path to emit a
    /// scissored fullscreen triangle that writes the constant clear value
    /// as depth (and optionally color), preserving D3D9's viewport-clipped
    /// Clear semantics on Metal. A typical shadow-cascade caster pass lands
    /// at a single combo (`Depth32Float`, no color); the cache caps at a
    /// handful of entries across games. Process-lifetime — the underlying
    /// `MTLRenderPipelineState`s leak for the unix process lifetime in the
    /// unix-side cache.
    clear_quad_pipeline_cache: FxHashMap<ClearQuadKey, MetalHandle<MTLRenderPipelineStateKind>>,
    /// Per-destination-format "blit-quad" pipeline handles.
    ///
    /// One entry per `(destination colour format, pass sample count)`. Used by the scaling
    /// `StretchRect` path (`stretch_blit_scaled`) to render the source
    /// texture onto a quad covering the destination rect — Metal's blit
    /// encoder can't scale. Process-lifetime, same posture as
    /// `clear_quad_pipeline_cache`.
    blit_pipeline_cache: FxHashMap<(PixelFormat, u8), MetalHandle<MTLRenderPipelineStateKind>>,
    /// Per-destination-format "upload-quad" pipeline handles.
    ///
    /// One entry per destination colour `PixelFormat`. Used by the GPU
    /// texture-upload pass, which reads the staging slab as a fragment
    /// buffer argument. Process-lifetime, same posture as
    /// `blit_pipeline_cache`.
    upload_pipeline_cache: FxHashMap<PixelFormat, MetalHandle<MTLRenderPipelineStateKind>>,
    /// Textures whose content the GPU upload pass wrote this frame.
    ///
    /// Those writes land in render passes spliced into the head of the
    /// frame, after the leading blit stream, so an `AUTOGENMIPMAP` regen
    /// that follows one has to run in the ordered blit stream instead of the
    /// leading one or it would read a stale level 0.
    upload_pass_textures: FxHashSet<TextureId>,
    /// Reusable command buffer for one texture-upload pass.
    ///
    /// Held on the encoder so the six commands an upload pass carries cost
    /// no allocation per upload; `emit_upload_pass` takes it, fills it, hands
    /// the slice to `PassState`, and puts it back.
    upload_pass_commands: Vec<Command>,
    /// Scratch texture the scaled back-buffer `ReleaseDC` write-back stages through.
    ///
    /// `NULL` until a `ReleaseDC` on a lockable back buffer has to change size
    /// on the way in (see [`FrameEncoder::upload_bytes_resampled`]). One
    /// texture, replaced when [`Self::dc_write_back_scratch_key`] stops
    /// matching, because the extent it is built for is the back buffer's own
    /// and that changes only at `Reset`.
    dc_write_back_scratch: MetalHandle<MTLTextureKind>,
    /// `(width, height, format)` [`Self::dc_write_back_scratch`] was built for.
    dc_write_back_scratch_key: (u32, u32, PixelFormat),
    /// `with-color-handle → no-color-handle` side-map.
    ///
    /// Populated by `get_or_create_pipeline` whenever a draw arrives with
    /// `color_write_mask == 0`: both pipeline variants are built (cached in
    /// `pipeline_cache` under their respective keys), and the no-color
    /// sibling of the emitted handle is recorded here. Consumed at submit
    /// time by `PassState::strip_color_from_no_color_draw_passes` (Rule H)
    /// to retroactively rewrite the pass's `SetRenderPipelineState`
    /// commands. Process-lifetime (the `pipeline_cache` itself is
    /// process-lifetime, so the handles never dangle); no per-frame clear.
    no_color_pipeline_alt: FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>>,
    /// Single-entry "L0" memo in front of `pipeline_cache`.
    ///
    /// `(last with-color snapshot → its handle)`. Consecutive draws
    /// overwhelmingly reuse the same pipeline (identical shaders + vdecl +
    /// blend + RT), so an equal snapshot returns the handle without
    /// rebuilding the `PipelineKey` (its D3D→Metal translations) or
    /// probing the cache. Holds a `PipelineSnapshot` (all-`Copy` fields,
    /// no borrowed/arena pointer) + the `u64` handle, so it persists
    /// across frames safely — `pipeline_cache` never evicts, so a
    /// snapshot→handle mapping stays valid for the process lifetime. Only
    /// successful (non-null) resolves are stored; failures fall through to
    /// the unchanged resolve path.
    last_pipeline_memo: Option<(PipelineSnapshot, u64)>,
    program_cache: FxHashMap<ProgramId, Box<DxsoProgram>>,
    /// Per-PS declared sampler slots + types, computed once at registration.
    ///
    /// Read on every programmable draw to bind an opaque-black fallback to any
    /// declared sampler the game left unbound. Only PS programs get an entry;
    /// VS programs (no samplers) fall through to the empty default.
    prog_sampler_decls: FxHashMap<ProgramId, PsSamplerDecls>,
    /// Compiled `MTLLibrary` handles keyed by content hash (`disk_key`).
    ///
    /// One entry per unique shader source; a single shader compiled
    /// for multiple `VsKey` / `PsKey` variants shares the same entry
    /// because variants either don't change MSL (VS) or do change it
    /// (PS) — and either way the `disk_key` derivation matches the MSL
    /// the shader will produce. Pre-warm ingest and live miss-compile
    /// both populate it; lookups happen by `disk_key`. No longer the
    /// per-draw lookup path — that goes through the source-keyed indices
    /// below; `lib_cache` is now the warm-load landing zone + disk-write
    /// index, consulted only on an index miss (≈ once per shader).
    lib_cache: FxHashMap<u64, StageLibHandles>,
    /// Per-draw shader-library lookup, keyed on the shader-identity struct.
    ///
    /// `FxHash` + exact `Eq`, probed by borrow — no per-draw content hash,
    /// no clone. One pair of maps per stage; VS keys exclude `variant`
    /// (variants share one `MTLLibrary`) but carry the user clip plane count
    /// (a programmable VS compiles one library per count), PS keys fold the
    /// variant in. The Xxh3
    /// `disk_key` is computed only on a miss here, to bridge `lib_cache`
    /// (warm-load) and address the on-disk cache.
    ff_vs_libs: FxHashMap<FfVsKey, StageLibHandles>,
    prog_vs_libs: FxHashMap<(ProgramId, u16, u8, VsSamplerKinds), StageLibHandles>,
    ff_ps_libs: FxHashMap<FfPsKey, FxHashMap<VariantKey, StageLibHandles>>,
    prog_ps_libs: FxHashMap<(ProgramId, VariantKey), StageLibHandles>,
    texture_cache: FxHashMap<TextureId, TextureGpuState>,
    sampler_cache: FxHashMap<SamplerKey, MetalHandle<MTLSamplerStateKind>>,
    /// Per-stage memo of the last sampler resolve, keyed on the raw D3D9 sampler-state words.
    ///
    /// A hit skips the snapshot + key build AND the `sampler_cache` probe
    /// (`get_or_create_sampler` runs per bound stage per draw, and sampler
    /// state almost never changes between consecutive draws). Never
    /// invalidated: `sampler_cache` entries live until encoder shutdown,
    /// so a memoized handle can't dangle.
    sampler_resolve_memo: [Option<SamplerResolveMemo>; crate::stage_bindings::STAGE_COUNT],
    /// Vertex texture fetch slots 0..3, mirrored from the device via ops.
    ///
    /// `SetTexture` / `SetSamplerState` on `D3DVERTEXTEXTURESAMPLER0..3`
    /// push updates; `emit_draw` binds the slots a programmable VS
    /// declares samplers for. Kept off the per-draw snapshot: vertex
    /// textures change orders of magnitude less often than draws.
    vertex_tex_bindings: [VertexTexBinding; mtld3d_core::passes::VERTEX_SAMPLER_SLOTS],
    /// Lazy `MTLBuffer` wrappers for bound VBs / IBs, keyed by their process-unique `BufferId`.
    ///
    /// One entry per live backing; on Lock-rename the API thread pushes
    /// the old `PageBox` into `pending_resource_retention`, `begin_frame`
    /// merges that with the cache's `MTLBuffer` handle, and the drain
    /// destroys both once the GPU retires the frame that last bound it.
    buffer_cache: FxHashMap<BufferId, BufferGpuState>,
    /// `MTLBuffer` wrappers + `MTLTextures` + their `PageBox` backings.
    ///
    /// Waiting for their submit seq to retire on the GPU. Drained at
    /// `begin_frame`. See `PendingResourceRetention` for the producer
    /// list.
    pending_resource_retention: VecDeque<PendingResourceRetention>,
    /// The shared triangle-fan index pattern every `IndexSource::Fan` draw binds.
    fan_index_buffer: FanIndexBuffer,
    /// D3D9 occlusion-query state.
    ///
    /// Per-frame slot allocator, shared visibility-buffer pool,
    /// active-query counter, pending finalize list. Reset per-frame via
    /// `visibility.reset_frame()` after queries retiring on the GPU have
    /// been finalized.
    visibility: VisibilityQueryState,
    /// Append-only writer for `mtld3d_shaders.bin`.
    ///
    /// `None` until the pre-warm thread signals readiness via the
    /// dedicated prewarm channel (or the disk cache is permanently
    /// disabled). After that, every cache-miss compile in
    /// `resolve_*_library` appends one record.
    cache_writer: Option<File>,
    /// Debounce state for the live `shaders: N compiled in Tms (…)` burst log.
    ///
    /// Polled once per frame from `run_frame`; emits when
    /// `shader_compile_stats::current_counts` has been stable + nonzero
    /// for ≥1 second of TSC cycles.
    compile_burst: BurstTracker,
    /// Pointer to the most recently shipped `CurrentSnapshot`.
    ///
    /// Lives in the per-frame `ScratchArena`. Set by
    /// `Op::SetCurrentSnapshot` in the dispatch loop; read by `emit_draw`
    /// via lifetime-laundered deref. Reset to `None` at the head of
    /// `run_frame` so stale pointers from a prior frame's arena can't
    /// dangle into the new frame's op stream — the API thread re-emits a
    /// fresh `Op::SetCurrentSnapshot` on the first draw of every new frame
    /// (`stamp_and_swap` sets `SnapshotDirty::all()`).
    current_snapshot: Option<CurrentSnapshotPtr>,

    /// Encoder-thread mirror of the programmable VS constant array.
    ///
    /// Kept in sync with `ShaderBindings::vs_constants` (API thread) via
    /// `Op::SetVsConstRange` delta ops. Boxed to keep `FrameEncoder` small
    /// despite the 4 KB array. Lifetime spans the encoder thread; persists
    /// across frames just like the API mirror.
    vs_constants_mirror: Box<[[f32; 4]; CONSTANT_ROWS]>,
    ps_constants_mirror: Box<[[f32; 4]; CONSTANT_ROWS]>,
    /// High-watermark of populated rows in each mirror.
    ///
    /// Mirrors `ShaderBindings::{vs,ps}_constants_populated_rows`. Used
    /// when a shader binds with `uses_rel_const` to bind the full
    /// populated prefix.
    vs_constants_populated_rows: u16,
    ps_constants_populated_rows: u16,
    /// Per-pass cache for the programmable VS const slice.
    ///
    /// Bumped into the current frame's `ScratchArena`. `emit_draw` reuses
    /// the cached slice across consecutive draws when the encoder-side
    /// mirror hasn't been touched by a `SetVsConstRange` op and the bound
    /// shader's `rows_to_bind` is unchanged. Cleared at `begin_frame`
    /// because the pointee lives in the previous frame's arena (about to
    /// drop). The cache is also invalidated whenever a delta op is applied
    /// (mirror content changed) or `rows_to_bind` changes.
    vs_const_scratch_cache: Option<(ScratchSlice, u16)>,
    ps_const_scratch_cache: Option<(ScratchSlice, u16)>,
    /// Encoder-thread mirror of the FF VS const buffer.
    ///
    /// Kept in sync with `FfState`-derived section deltas via
    /// `Op::SetFfVsConstRange`. Parallel to `vs_constants_mirror`
    /// (programmable). No populated-rows watermark needed — every FF draw
    /// carries `max_row + 1` via `VsSource::FixedFunction.max_row`.
    ff_vs_constants_mirror: Box<[[f32; 4]; CONSTANT_ROWS]>,
    /// Per-pass cache for the FF VS const slice bumped into the current frame's `ScratchArena`.
    ///
    /// Mirrors `vs_const_scratch_cache` semantics: cleared at
    /// `begin_frame` AND on every `apply_ff_vs_const_range` (delta op
    /// changes the mirror → stale cache). Distinct slices per "mirror
    /// epoch" between deltas guarantee the per-draw isolation invariant
    /// Metal's submit-time setVertexBytes copy depends on.
    ff_vs_const_scratch_cache: Option<(ScratchSlice, u16)>,
}

/// Shared body for `apply_{vs,ps}_const_range`.
///
/// Reads `rows × 16` bytes from `data` and writes them into
/// `mirror[start_row..]`. Out-of-range inputs are clamped (the API thread
/// should have clamped already; this is a defence-in-depth check). `tag`
/// is logged on the rare clamp path to make a bug observable without
/// spamming.
fn apply_const_range_into(
    mirror: &mut [[f32; 4]; CONSTANT_ROWS],
    start_row: u16,
    rows: u16,
    data: ScratchSlice,
    tag: &'static str,
) {
    if rows == 0 {
        return;
    }
    let start = usize::from(start_row);
    if start >= CONSTANT_ROWS {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "{tag}: start_row {start} out of range");
        return;
    }
    let end = (start + usize::from(rows)).min(CONSTANT_ROWS);
    let need_bytes = (end - start) * core::mem::size_of::<[f32; 4]>();
    let bytes = data.as_slice();
    if bytes.len() < need_bytes {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "{tag}: data {} < need {need_bytes}",
            bytes.len()
        );
        return;
    }
    // SAFETY: `start < CONSTANT_ROWS` per the early-return above; offset
    // stays inside the same allocated `[[f32; 4]; CONSTANT_ROWS]` array.
    let dst_row = unsafe { mirror.as_mut_ptr().add(start) };
    // SAFETY: `bytes.len() >= need_bytes` was just bounds-checked.
    // `dst_row` points at `mirror[start]`, covering exactly `need_bytes`
    // contiguous bytes of `[f32; 4]` rows up to `end`. POD bytewise copy.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst_row.cast::<u8>(), need_bytes);
    }
}

/// Cache key for the per-format-combo clear-quad pipeline.
///
/// Used by `emit_clear_quad_depth_stencil_inner` / `emit_clear_quad_color_inner`.
/// Mirrors `EnsureClearQuadPipelineParams` (modulo `device_handle`) so the
/// PE-side cache and the unix-side cache agree on what counts as a
/// distinct pipeline.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct ClearQuadKey {
    depth_format: PixelFormat,
    color_format: PixelFormat,
    flags: ClearQuadFlags,
    /// Render targets 1..3 of the pass, alpha bits cleared (the quad blends nothing).
    extra: mtld3d_core::pipeline_state::ExtraColorAttachments,
    /// Sample count of the pass the quad draws into; Metal requires the match.
    sample_count: u8,
}

/// Where a mid-pass clear quad lands, as the depth and stencil clear chains report it.
///
/// `clear_depth_stencil` compares the two reports to decide whether one quad
/// can serve both planes.
struct ClearQuadTarget {
    viewport: (u32, u32, u32, u32),
    has_color: bool,
    color_format: PixelFormat,
}

impl ClearQuadTarget {
    fn same_as(&self, other: &Self) -> bool {
        self.viewport == other.viewport
            && self.has_color == other.has_color
            && self.color_format == other.color_format
    }
}

/// Cached `MTLBuffer` wrapper for one live VB/IB `PageBox`.
struct BufferGpuState {
    /// `Direct` buffers: the `bytesNoCopy` wrapper over the CPU backing the GPU reads directly.
    ///
    /// `Staged` buffers: `NULL` (the GPU never reads the CPU staging —
    /// see `device_buffer`).
    mtl_buffer: MetalHandle<MTLBufferKind>,
    /// `Staged` buffers only: the persistent `StorageModePrivate` device buffer that draws bind.
    ///
    /// Written by the staging-upload blit. `NULL` for `Direct`, and for a
    /// `Staged` buffer whose warmup create failed (the placeholder shape,
    /// recreated lazily on the buffer's next upload or draw).
    device_buffer: MetalHandle<MTLBufferKind>,
    /// `true` for a non-DYNAMIC buffer on the separate-staging upload path.
    ///
    /// `false` for the zero-copy `Direct` path.
    is_staged: bool,
    backing_ptr: u64,
    length: u64,
    /// Identity of the backing allocation the wrapper was created over.
    ///
    /// A cache hit needs it to match alongside the address: the allocator
    /// can hand a freed backing's address to a later allocation, and the
    /// `bytesNoCopy` wrapper pins the dead allocation's pages, so an
    /// address-only match pairs GPU reads with pages the CPU no longer
    /// writes (issue #76's garbled text). Meaningless for `Staged` (0).
    backing_generation: u64,
    /// Max submit seq this wrapper has been bound into a Draw for.
    ///
    /// Used when the cache entry is evicted to retention.
    last_submit_seq: u64,
}

/// The encoder's shared 16-bit triangle-fan index pattern.
///
/// `convert::fill_fan_pattern_u16` in a PE `PageBox` wrapped as an
/// `MTLBuffer`, grown to the longest fan drawn so far. A grown-out pattern
/// goes through `pending_resource_retention` like any other buffer: an
/// earlier draw this frame may still reference it.
struct FanIndexBuffer {
    backing: Option<PageBox>,
    handle: MetalHandle<MTLBufferKind>,
    /// Triangles the pattern currently covers.
    triangles: u32,
}

impl FanIndexBuffer {
    const EMPTY: Self = Self {
        backing: None,
        handle: MetalHandle::NULL,
        triangles: 0,
    };
}

/// One deferred Metal-handle retention entry owned by the encoder thread.
///
/// On drain: `destroy_resources_bulk(kind, &[handle])` if `handle != 0`,
/// then drop `page_box` if present. Producers:
///
/// 1. API-thread VB/IB Lock-rename (`intake_vbib_retention`):
///    `Buffer` + handle + `page_box`.
/// 2. Encoder-side VB/IB mid-frame cache swap
///    (`ensure_vbib_mtl_buffer_impl`): `Buffer` + handle, `page_box = None`.
///    The new backing is live in the replacement cache entry; the old
///    backing was queued separately at Lock-rename time.
/// 3. Encoder-side texture-staging mid-frame cache swap
///    (`get_or_create_staging_buffer`): `Buffer` + handle,
///    `page_box = None`. The staging `Box` is kept alive via
///    `pending_blit_retention` (Arc clones).
/// 4. Visibility-buffer pool over-cap eviction (`submit` path, via
///    `VisibilityQueryState::retire_current_buffer`): `Buffer` +
///    handle + `page_box`, `seq = release_seq` of the evicted buffer.
///    `newBufferWithBytesNoCopy:` over the evicted `PageBox`; drain must
///    destroy the wrapper before the backing drops.
/// 5. `repack_blit_source_padded` transient: `Buffer` + handle +
///    `page_box`.
/// 6. `destroy_cached_texture` (refcount → 0): `Texture` + the cached
///    `MTLTexture` handle, plus `Buffer` entries for each mip staging
///    wrapper. Both kinds get queued together so any `BlitCommand`
///    pushed earlier in this frame referencing them outlives this
///    frame's submit. Destroying these synchronously races against
///    the in-flight blit replay on Intel/AMD (Bronze driver) where
///    Metal recycles the freed address as the wrong type.
/// 7. `fan_index_buffer` growth: `Buffer` + the grown-out pattern's
///    handle + `page_box`, `seq = current_submit_seq`, since a draw
///    earlier this frame may still bind it.
/// 8. `readback_device_buffer` destination wrapper: `Buffer` + handle,
///    `page_box = None`. The PE pages under it belong to the index
///    buffer that asked for the read and outlive the wrapper.
struct PendingResourceRetention {
    kind: DestroyKind,
    handle: u64,
    /// Owned `PageBox` carried via unique ownership transfer.
    ///
    /// VB/IB rename, visibility eviction, padded-blit transient. Released
    /// when this entry drops at drain time, after the wrapping `MTLBuffer`
    /// is destroyed.
    page_box: Option<PageBox>,
    /// Shared `Arc<PageBox>` keepalive used by texture-staging entries.
    ///
    /// The PE-side staging is `Vec<Arc<PageBox>>` on `TextureInner` —
    /// `texture_release` drops the original Arc synchronously on the
    /// API thread, so the staging `MTLBuffer` cache slot must hold its
    /// own clone to outlive that drop. This field carries that clone
    /// from `MipStagingBuffer.keepalive` into the retention queue when
    /// the slot is parked.
    staging_arc: Option<Arc<PageBox>>,
    seq: u64,
    /// `true` when the entry comes from the texture lifecycle.
    ///
    /// The sites are `MTLTexture` destroy, mip-staging `MTLBuffer`
    /// wrapper destroy on rename, padded-blit transient wrapper, and the
    /// single-slice view a scaling blit out of a cube face binds. At
    /// drain time the destroy is attributed to the textures `destroys`
    /// row instead of VB/IB. Default `false` covers VB/IB rename/intake
    /// and visibility-pool eviction — those stay on the VB/IB row.
    from_texture: bool,
}

/// Everything a replay of one `Staged` VB/IB upload needs.
///
/// The payload of a `pending_stage_uploads` entry. Both the transient
/// `bytesNoCopy` wrapper and the PE-heap snapshot it wraps stay alive
/// until the GPU acknowledges the copy, so a replay is one more
/// buffer-to-buffer blit from the same source: the buffer's persistent CPU
/// staging may have moved on since, and the bytes this upload owed are the
/// ones in `page_box`. Freed the other way round (wrapper destroyed, then
/// backing dropped) because Metal holds a raw pointer into the box.
struct StagedUploadRetry {
    buffer_id: BufferId,
    transient: MetalHandle<MTLBufferKind>,
    page_box: PageBox,
    dst_offset: u32,
    size: u32,
}

/// Out-parameter for `drain_retention_and_wait`.
///
/// Holds the `PageBox`/`Arc<PageBox>` backings of every drained
/// `PendingResourceRetention` entry so they outlive the caller's
/// `destroy_resources_bulk` calls. Order matters at drop time: the
/// wrapping `MTLBuffer` (created via `bytesNoCopy`) must be released
/// by Metal before its backing memory drops, or the buffer holds a
/// dangling pointer.
#[derive(Default)]
struct HeldBackings {
    pageboxes: Vec<PageBox>,
    staging_arcs: Vec<Arc<PageBox>>,
}

/// Intersect a D3D9 `RECT` `(x1, y1, x2, y2)` with the viewport `(x, y, w, h)`.
///
/// The rect is half-open, top-left origin. Returns the overlap as
/// `(x, y, w, h)`, or `None` if the rect is inverted/degenerate or the
/// overlap is empty. Used by `clear_color_rects` to turn each `Clear`
/// pRect into a clip-to-viewport scissor region.
fn clip_rect_to_viewport(
    rect: (i32, i32, i32, i32),
    vp: (u32, u32, u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let (rx1, ry1, rx2, ry2) = rect;
    if rx2 <= rx1 || ry2 <= ry1 {
        return None;
    }
    let (vx, vy, vw, vh) = vp;
    let vx2 = vx.saturating_add(vw);
    let vy2 = vy.saturating_add(vh);
    let x1 = rx1.max(0).cast_unsigned().max(vx);
    let y1 = ry1.max(0).cast_unsigned().max(vy);
    let x2 = rx2.max(0).cast_unsigned().min(vx2);
    let y2 = ry2.max(0).cast_unsigned().min(vy2);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    Some((x1, y1, x2 - x1, y2 - y1))
}

/// The Metal textures a standalone colour surface owns, for retirement.
///
/// A surface can carry up to four: the single-sample texture the D3D9
/// surface's identity is, its sRGB twin view, the multisampled companion the
/// passes attach, and that companion's own twin. Grouped so
/// [`FrameEncoder::retire_color_target`] takes one argument per surface
/// rather than one per view.
pub struct RetiredColorTarget {
    pub base: MetalHandle<MTLTextureKind>,
    pub srgb: MetalHandle<MTLTextureKind>,
    pub msaa: MetalHandle<MTLTextureKind>,
    pub msaa_srgb: MetalHandle<MTLTextureKind>,
}

impl FrameEncoder {
    fn new(gpu_caps: GpuCaps) -> Self {
        // Spawn the dedicated submit thread. It issues the `SubmitFrame`
        // thunk for `Async` frames so the unix command-walk + present
        // overlaps the encoder's next build. The work channel is cap-1 so
        // the encoder can queue at most one packet ahead of an in-progress
        // submit; the return channel is unbounded so the submit thread
        // never blocks handing payloads back.
        let (submit_work_tx, submit_work_rx) = mpsc::sync_channel::<SubmitPacket>(1);
        let (submit_return_tx, submit_return_rx) = mpsc::channel::<ReturnedPayload>();
        // The join handle is dropped (thread detached): like the encoder
        // thread it is never joined — Wine can report STATUS_INVALID_HANDLE
        // for the Win32 handle on long sessions — and it exits on its own
        // when the work channel closes at teardown.
        thread::Builder::new()
            .name("mtld3d-submit".into())
            .spawn(move || submit_thread_main(&submit_work_rx, &submit_return_tx))
            .expect("mtld3d: failed to spawn submit thread");
        Self {
            pass_state: PassState::new(),
            last_bound: LastBoundCache::new(),
            scratch: ScratchArena::new(),
            frame_blit_commands: Vec::new(),
            flags: if shader_cache_enabled() {
                FrameEncoderFlags::empty()
            } else {
                FrameEncoderFlags::CACHE_DISABLED
            },
            payload_pool: Vec::new(),
            submit_work_tx,
            submit_return_rx,
            submit_in_flight: 0,
            submit_payloads_total: 0,
            last_submit_status: 0,
            prev_submit_no_present: false,
            current_blit_retention: Vec::new(),
            pending_blit_retention: VecDeque::new(),
            coherent_seq_ptr: 0,
            failed_seq_ptr: 0,
            upload_coherent_seq_ptr: 0,
            pending_stage_uploads: UploadRecoveryQueue::new(),
            pending_texture_uploads: UploadRecoveryQueue::new(),
            retained_bytes_ptr: 0,
            current_submit_seq: 0,
            backbuffer_width: 0,
            backbuffer_height: 0,
            depth_attachment_desc: (0, 0, mtld3d_shared::mtl::PixelFormat::Depth32Float),
            depth_write_epoch: 0,
            depth_snapshots: FxHashMap::default(),
            stretch_scratch: FxHashMap::default(),
            perf: EncoderPerfState::new(),
            pass_shader_log_fired: FxHashSet::default(),
            gpu_caps,
            device_handle: MetalHandle::NULL,
            depth_stencil_cache: FxHashMap::default(),
            pipeline_cache: FxHashMap::default(),
            clear_quad_pipeline_cache: FxHashMap::default(),
            blit_pipeline_cache: FxHashMap::default(),
            upload_pipeline_cache: FxHashMap::default(),
            upload_pass_textures: FxHashSet::default(),
            upload_pass_commands: Vec::new(),
            dc_write_back_scratch: MetalHandle::NULL,
            dc_write_back_scratch_key: (0, 0, PixelFormat::Bgra8Unorm),
            no_color_pipeline_alt: FxHashMap::default(),
            last_pipeline_memo: None,
            program_cache: FxHashMap::default(),
            prog_sampler_decls: FxHashMap::default(),
            lib_cache: FxHashMap::default(),
            ff_vs_libs: FxHashMap::default(),
            prog_vs_libs: FxHashMap::default(),
            ff_ps_libs: FxHashMap::default(),
            prog_ps_libs: FxHashMap::default(),
            texture_cache: FxHashMap::default(),
            sampler_cache: FxHashMap::default(),
            sampler_resolve_memo: core::array::from_fn(|_| None),
            vertex_tex_bindings: core::array::from_fn(|_| VertexTexBinding::default()),
            buffer_cache: FxHashMap::default(),
            pending_resource_retention: VecDeque::new(),
            fan_index_buffer: FanIndexBuffer::EMPTY,
            visibility: VisibilityQueryState::new(),
            cache_writer: None,
            compile_burst: BurstTracker::new(),
            current_snapshot: None,
            vs_constants_mirror: Box::new([[0.0; 4]; CONSTANT_ROWS]),
            ps_constants_mirror: Box::new([[0.0; 4]; CONSTANT_ROWS]),
            vs_constants_populated_rows: 0,
            ps_constants_populated_rows: 0,
            vs_const_scratch_cache: None,
            ps_const_scratch_cache: None,
            ff_vs_constants_mirror: Box::new([[0.0; 4]; CONSTANT_ROWS]),
            ff_vs_const_scratch_cache: None,
        }
    }

    /// Pointer accessor for the encoder's current snapshot.
    ///
    /// Returns the raw scratch pointer so callers can launder the
    /// lifetime (the pointee lives in the per-frame arena, distinct
    /// from `self`).
    pub const fn current_snapshot_ptr(&self) -> Option<CurrentSnapshotPtr> {
        self.current_snapshot
    }

    /// Issue one batched `CreateTexturesBatch` thunk.
    ///
    /// Caller owns `descs`, `handles_out` and `srgb_handles_out`; all three
    /// slices must outlive the call because the unix side dereferences
    /// their pointers during the thunk. On success `handles_out[i]` carries
    /// the handle for `descs[i]` and `srgb_handles_out[i]` its eager sRGB
    /// twin view (NULL when the format has none); on per-element failure
    /// the slots stay at their initial value (caller passes zeros).
    fn batch_create_textures(
        &self,
        descs: &[TextureCreateDesc],
        handles_out: &mut [MetalHandle<MTLTextureKind>],
        srgb_handles_out: &mut [MetalHandle<MTLTextureKind>],
    ) -> i32 {
        debug_assert_eq!(descs.len(), handles_out.len());
        debug_assert_eq!(descs.len(), srgb_handles_out.len());
        if descs.is_empty() {
            return 0;
        }
        let count =
            u32::try_from(descs.len()).expect("batch_create_textures: descs.len() exceeds u32");
        let mut params = CreateTexturesBatchParams {
            device_handle: self.device_handle,
            count,
            pad0: 0,
            descs_ptr: descs.as_ptr() as u64,
            handles_out_ptr: handles_out.as_mut_ptr() as u64,
            srgb_handles_out_ptr: srgb_handles_out.as_mut_ptr() as u64,
        };
        unix_call(&mut params)
    }

    /// Issue one batched `CreateBuffersBatch` thunk.
    ///
    /// Same wire-backing rules as `batch_create_textures`.
    fn batch_create_buffers(
        &self,
        descs: &[BufferCreateDesc],
        handles_out: &mut [MetalHandle<MTLBufferKind>],
    ) -> i32 {
        debug_assert_eq!(descs.len(), handles_out.len());
        if descs.is_empty() {
            return 0;
        }
        let count =
            u32::try_from(descs.len()).expect("batch_create_buffers: descs.len() exceeds u32");
        let mut params = CreateBuffersBatchParams {
            device_handle: self.device_handle,
            count,
            pad0: 0,
            descs_ptr: descs.as_ptr() as u64,
            handles_out_ptr: handles_out.as_mut_ptr() as u64,
        };
        unix_call(&mut params)
    }

    /// Pack a `TextureInfo` snapshot into the per-element wire descriptor.
    ///
    /// The single source for both the load-phase warmup batch
    /// (`drain_texture_warmups`) and the one-off lazy fallback
    /// (`get_or_create_texture`), so both emit byte-identical descriptors.
    fn texture_desc_from_info(&self, info: &TextureInfo) -> TextureCreateDesc {
        // Every texture created here is `Private`. Nothing CPU-writes a
        // texture directly: render targets are GPU output, and all uploads
        // (including the A4R4G4B4 / R5G6B5 / A1R5G5B5 → BGRA8 expansion path)
        // go through `copyFromBuffer:toTexture:` blits, whose destination can
        // be Private. There is deliberately no CPU-timeline `replaceRegion`
        // path — it would race a texture sampled by an in-flight frame — so no
        // texture needs a CPU-writable mode. Only the staging *buffers* (blit
        // sources) follow `buffer_storage_mode`.
        let storage_mode = StorageMode::Private;
        // Uploads that cannot ride a blit copy (a packed 16-bit source
        // widened to BGRA8, or a mip whose row pitch is under the linear
        // texture alignment) are written by a render pass, which needs the
        // destination to be an attachment. The predicate is a superset of
        // what the upload path selects per upload, so an upload never finds a
        // texture without the usage.
        let mut usage_flags = info.usage_flags;
        if mtld3d_core::upload_pass::needs_render_target(
            info.d3d_format,
            info.pixel_format,
            info.width,
            info.levels,
            self.gpu_caps.min_linear_texture_align,
        ) {
            usage_flags |= TextureUsage::RENDER_TARGET;
        }
        TextureCreateDesc {
            tex_id: info.texture_id.raw(),
            width: info.width,
            height: info.height,
            depth: info.depth,
            levels: info.levels,
            pixel_format: info.pixel_format,
            storage_mode,
            flags: info.create_flags,
            swizzle_r: info.swizzle[0],
            swizzle_g: info.swizzle[1],
            swizzle_b: info.swizzle[2],
            swizzle_a: info.swizzle[3],
            usage_flags,
        }
    }

    const fn texture_staging_slot_count(info: &TextureInfo) -> usize {
        let faces = if info.create_flags.contains(TextureCreateFlags::TYPE_CUBE) {
            6
        } else {
            1
        };
        info.levels as usize * faces
    }

    /// Drain the API-thread-queued texture warmups into one batched `CreateTexturesBatch` thunk.
    ///
    /// Called at the head of `run_frame` before the op loop, so
    /// subsequent draw closures hit the cache instead of cache-missing
    /// on first bind.
    ///
    /// Cache-collision case: a `TextureId` already in `texture_cache`
    /// (e.g. rehydration ran the lazy path between push and drain) gets
    /// its freshly-created handle queued for seq-gated destroy. The
    /// existing cache entry stays untouched.
    fn drain_texture_warmups(&mut self, infos: Vec<TextureInfo>) {
        if infos.is_empty() {
            return;
        }
        let descs: Vec<TextureCreateDesc> = infos
            .iter()
            .map(|info| self.texture_desc_from_info(info))
            .collect();
        let mut handles = vec![MetalHandle::<MTLTextureKind>::NULL; descs.len()];
        let mut srgb_handles = vec![MetalHandle::<MTLTextureKind>::NULL; descs.len()];
        let status = self.batch_create_textures(&descs, &mut handles, &mut srgb_handles);
        if status != 0 {
            error!(
                target: LOG_TARGET,
                "drain_texture_warmups: CreateTexturesBatch status={status:#x} (count={})",
                infos.len()
            );
        }
        let current_seq = self.current_submit_seq;
        for ((info, handle), srgb_handle) in infos.into_iter().zip(handles).zip(srgb_handles) {
            if handle.is_null() {
                continue;
            }
            match self.texture_cache.entry(info.texture_id) {
                Entry::Vacant(v) => {
                    v.insert(TextureGpuState {
                        mtl_texture: handle,
                        mtl_texture_srgb: srgb_handle,
                        mip_staging_buffers: vec![
                            MipStagingBuffer::default();
                            Self::texture_staging_slot_count(&info)
                        ],
                    });
                    self.pass_state.register_srgb_twin(srgb_handle, handle);
                }
                Entry::Occupied(_) => {
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "drain_texture_warmups: cache collision for tex_id, queueing orphan handle for retire"
                    );
                    for orphan in [handle, srgb_handle] {
                        if orphan.is_null() {
                            continue;
                        }
                        self.pending_resource_retention
                            .push_back(PendingResourceRetention {
                                kind: DestroyKind::Texture,
                                handle: orphan.raw(),
                                page_box: None,
                                staging_arc: None,
                                seq: current_seq,
                                from_texture: false,
                            });
                    }
                }
            }
        }
    }

    /// Drain the API-thread-queued VB/IB warmups into one batched `CreateBuffersBatch` thunk.
    ///
    /// Called at the head of `run_frame` alongside
    /// `drain_texture_warmups`.
    ///
    /// Only the load-phase create case is queued here (initial
    /// `CreateVertexBuffer` / `CreateIndexBuffer`); mid-frame
    /// Lock(DISCARD) renames stay on the lazy path inside
    /// `ensure_vbib_mtl_buffer` to avoid mid-frame cache collision
    /// (an old-backing draw closure would otherwise mismatch the
    /// freshly-installed new wrapper and trigger redundant churn).
    ///
    /// A failed `Direct` create leaves no cache entry: the ensure tail is
    /// a complete lazy retry for that shape. A failed `Staged` create must
    /// NOT do the same: with no entry the ensure tail would rebuild the
    /// buffer as `Direct` over its CPU staging while the PE side keeps
    /// sending `Op::StageUpload`s, silently switching buffer models. It
    /// instead inserts a placeholder entry (`is_staged: true`, NULL
    /// `device_buffer`) so staged-ness survives; the device buffer is
    /// recreated lazily by `apply_stage_upload` / `ensure_vbib_mtl_buffer`
    /// on the buffer's next touch.
    fn drain_buffer_warmups(&mut self, warmups: Vec<VbibWarmupEntry>) {
        if warmups.is_empty() {
            return;
        }
        let storage_mode = buffer_storage_mode(self.gpu_caps.unified_memory);
        // `Staged` buffers get a `StorageModePrivate` device buffer (no
        // CPU backing — `backing_ptr` ignored unix-side); `Direct` buffers
        // get the `bytesNoCopy` wrap over their CPU backing. Mixed kinds in
        // one batch are fine — the unix handler branches per descriptor.
        let descs: Vec<BufferCreateDesc> = warmups
            .iter()
            .map(|w| {
                let staged = matches!(w.map_mode, BufferMapMode::Staged);
                BufferCreateDesc {
                    backing_ptr: if staged { 0 } else { w.backing_ptr },
                    length: w.backing_len,
                    id: w.buffer_id.raw(),
                    storage_mode,
                    kind: if staged {
                        BufferKind::VbIbDevice
                    } else {
                        BufferKind::VbIb
                    },
                }
            })
            .collect();
        let mut handles = vec![MetalHandle::<MTLBufferKind>::NULL; descs.len()];
        let status = self.batch_create_buffers(&descs, &mut handles);
        if status != 0 {
            error!(
                target: LOG_TARGET,
                "drain_buffer_warmups: CreateBuffersBatch status={status:#x} (count={})",
                warmups.len()
            );
        }
        let current_seq = self.current_submit_seq;
        for (warmup, handle) in warmups.into_iter().zip(handles) {
            let staged = matches!(warmup.map_mode, BufferMapMode::Staged);
            if handle.is_null() {
                error!(
                    target: LOG_TARGET,
                    "drain_buffer_warmups: CreateBuffer failed \
                     (id={:#x}, len={}, staged={staged})",
                    warmup.buffer_id.raw(),
                    warmup.backing_len,
                );
                if staged {
                    // Keep the staged identity alive: the device buffer is
                    // recreated lazily on the buffer's next touch. Occupied
                    // is unreachable (ids are minted once and warmups are
                    // pushed once per Create), and there is no handle to
                    // orphan here anyway.
                    if let Entry::Vacant(v) = self.buffer_cache.entry(warmup.buffer_id) {
                        v.insert(BufferGpuState {
                            mtl_buffer: MetalHandle::NULL,
                            device_buffer: MetalHandle::NULL,
                            is_staged: true,
                            backing_ptr: 0,
                            length: warmup.backing_len,
                            backing_generation: 0,
                            last_submit_seq: current_seq,
                        });
                    }
                }
                continue;
            }
            match self.buffer_cache.entry(warmup.buffer_id) {
                Entry::Vacant(v) => {
                    v.insert(BufferGpuState {
                        mtl_buffer: if staged { MetalHandle::NULL } else { handle },
                        device_buffer: if staged { handle } else { MetalHandle::NULL },
                        is_staged: staged,
                        backing_ptr: if staged { 0 } else { warmup.backing_ptr },
                        length: warmup.backing_len,
                        backing_generation: if staged { 0 } else { warmup.backing_generation },
                        last_submit_seq: current_seq,
                    });
                    // `Direct`: fresh `bytesNoCopy` wrapper — notify the
                    // GPU about every byte the CPU may have written since
                    // the backing was allocated (no-op on UMA). `Staged`:
                    // the device buffer is `Private`, never CPU-written,
                    // so no notify — its contents arrive via upload blits.
                    if !staged {
                        self.enqueue_notify_buffer_did_modify_range(
                            handle.raw(),
                            0,
                            warmup.backing_len,
                        );
                    }
                }
                Entry::Occupied(_) => {
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "drain_buffer_warmups: cache collision for buffer_id, queueing orphan handle for retire"
                    );
                    self.pending_resource_retention
                        .push_back(PendingResourceRetention {
                            kind: DestroyKind::Buffer,
                            handle: handle.raw(),
                            page_box: None,
                            staging_arc: None,
                            seq: current_seq,
                            from_texture: false,
                        });
                }
            }
        }
    }

    /// Apply one inline `Staged` VB/IB upload in op-stream order.
    ///
    /// The transient `page_box` snapshots the bytes the game wrote between
    /// `Lock` and `Unlock` (taken on the API thread, so a later frame's
    /// writes to the persistent CPU staging can't corrupt the in-flight
    /// copy). We wrap it as a `Shared` `bytesNoCopy` blit source and copy
    /// its range into the buffer's persistent `Private` device buffer via
    /// `frame_blit_commands` (a leading phase, before any draw), then
    /// retire the transient once this frame's submit retires.
    ///
    /// RENAME-AT-OVERLAP: if a draw earlier this frame already read a
    /// region this upload overwrites, writing the upload into the live
    /// device buffer would corrupt that earlier draw (they share one
    /// buffer — and the blit lands frame-head, before every pass). Instead
    /// we allocate a FRESH device buffer,
    /// preserve the old contents into it (full device→device copy), write
    /// the upload there, and rebind it for later draws — the earlier draws
    /// keep the old buffer (per-draw snapshot). All three blits land in
    /// the leading phase precisely because the fresh buffer is read by no
    /// earlier draw, so NO render-pass split is needed: the TBDR-correct
    /// equivalent of a `D3DLOCK_DISCARD` buffer rename. Overlaps are rare (measured
    /// ~0.07/frame), so the extra device-buffer churn is negligible and
    /// bounded by the same seq-gated retire as every other VB/IB rename.
    ///
    /// A cache entry whose `device_buffer` is NULL is the failed-warmup
    /// placeholder; the device buffer is recreated here before the upload
    /// is applied, which makes a warmup create failure fully recoverable.
    fn apply_stage_upload(
        &mut self,
        buffer_id: BufferId,
        page_box: PageBox,
        dst_offset: u32,
        size: u32,
    ) {
        let _t =
            mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::StageUpload));
        let current_seq = self.current_submit_seq;
        let storage_mode = buffer_storage_mode(self.gpu_caps.unified_memory);

        // Resolve the buffer's current device buffer + length, gating its
        // eventual destroy past this frame's upload write.
        let Some((device_handle, length)) = self
            .buffer_cache
            .get_mut(&buffer_id)
            .filter(|s| s.is_staged)
            .map(|s| {
                if current_seq > s.last_submit_seq {
                    s.last_submit_seq = current_seq;
                }
                (s.device_buffer.raw(), s.length)
            })
        else {
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: buffer_id.raw(),
                "apply_stage_upload: no cache entry for buffer_id {:#x}, dropping upload",
                buffer_id.raw()
            );
            return;
        };

        // A NULL device buffer is the placeholder `drain_buffer_warmups`
        // leaves behind when the warmup create failed: recreate it here so
        // the upload lands. Warmups drain before the op loop, so the first
        // post-failure upload passes through this recreate and nothing is
        // lost once Metal allocates again. Both failure returns above and
        // below sit before `add_retained_bytes`, so the dropped `page_box`
        // never skews the retention cap. This must also stay ahead of the
        // transient create below: returning after it would leak a live
        // transient wrapper.
        let device_handle = if device_handle == 0 {
            let Some(fresh) = self.alloc_fresh_device_buffer(buffer_id, length) else {
                error!(
                    target: LOG_TARGET,
                    "apply_stage_upload: device buffer recreate failed \
                     (id={buffer_id:#x}), dropping upload ({size} bytes at {dst_offset})",
                );
                return;
            };
            if let Some(s) = self.buffer_cache.get_mut(&buffer_id) {
                s.device_buffer = fresh;
            }
            mtld3d_shared::log_once_info_by!(
                target: LOG_TARGET,
                key: buffer_id.raw(),
                "apply_stage_upload: recreated device buffer for buffer_id {:#x} \
                 after a failed warmup create",
                buffer_id.raw()
            );
            fresh.raw()
        } else {
            device_handle
        };

        // Wrap the transient snapshot as a `Shared` `bytesNoCopy` blit
        // source. The CPU just wrote it, so notify the GPU on non-UMA
        // before the blit reads it (no-op on UMA).
        let desc = BufferCreateDesc {
            backing_ptr: page_box.as_ptr() as u64,
            length: page_box.len() as u64,
            id: buffer_id.raw(),
            storage_mode,
            kind: BufferKind::VbIb,
        };
        let mut transient = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut transient),
        );
        if status != 0 || transient.is_null() {
            error!(
                target: LOG_TARGET,
                "apply_stage_upload: transient CreateBuffer failed (status={status:#x}, len={})",
                page_box.len(),
            );
            return;
        }
        self.enqueue_notify_buffer_did_modify_range(transient.raw(), 0, u64::from(size));

        // Does this upload overwrite a region a draw already read this
        // frame? If so, rename rather than corrupt that draw (the blit
        // lands frame-head, before every pass).
        let end = dst_offset.saturating_add(size);
        let overlap = self
            .pass_state
            .drawn_range_overlaps(buffer_id.raw(), dst_offset, end);

        let dst_handle = if overlap {
            if let Some(fresh) = self.alloc_fresh_device_buffer(buffer_id, length) {
                // Preserve the old contents into the fresh buffer (full
                // device→device copy), then rebind it for later draws and
                // retire the old buffer once this frame's GPU read retires.
                self.frame_blit_commands
                    .push(BlitCommand::copy_buffer_to_buffer(
                        &CopyBufferToBufferInfo {
                            src_buffer: device_handle,
                            dst_buffer: fresh.raw(),
                            src_offset: 0,
                            dst_offset: 0,
                            byte_size: length,
                        },
                    ));
                if let Some(s) = self.buffer_cache.get_mut(&buffer_id) {
                    s.device_buffer = fresh;
                }
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Buffer,
                        handle: device_handle,
                        page_box: None,
                        staging_arc: None,
                        seq: current_seq,
                        from_texture: false,
                    });
                // The fresh buffer has been read by no draw yet.
                self.pass_state.clear_drawn_range(buffer_id.raw());
                self.perf.bump_vbib_mid_pass_reorder();
                fresh.raw()
            } else {
                // Alloc failed — fall back to overwriting the live buffer.
                // One draw may glitch this frame, but dropping the upload
                // would persist stale geometry instead.
                device_handle
            }
        } else {
            device_handle
        };

        // Apply the dirty-range upload to the (possibly fresh) device buffer.
        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_buffer(
                &CopyBufferToBufferInfo {
                    src_buffer: transient.raw(),
                    dst_buffer: dst_handle,
                    src_offset: 0,
                    dst_offset: u64::from(dst_offset),
                    byte_size: u64::from(size),
                },
            ));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        self.perf.bump_vbib_staging_upload();

        // Hold the transient wrapper + backing until this frame's submit is
        // *acknowledged*, not merely retired: an aborted command buffer
        // discards the blit above, and the recovery queue is what re-emits
        // it from these same bytes. Account the CPU bytes into the shared
        // retention total like every queued `PageBox`.
        self.perf.bump_vbib_retained_add(page_box.len());
        self.add_retained_bytes(page_box.len());
        self.pending_stage_uploads.push(
            buffer_id.raw(),
            current_seq,
            StagedUploadRetry {
                buffer_id,
                transient,
                page_box,
                dst_offset,
                size,
            },
        );
    }

    /// Re-emit one discarded `Staged` VB/IB upload into this frame's leading blits.
    ///
    /// The transient `MTLBuffer` and its backing are still alive (holding
    /// them is the recovery queue's whole job), so the replay is the same
    /// buffer-to-buffer copy aimed at the buffer's *current* device buffer.
    /// No second `didModifyRange` notify: nothing has written the transient
    /// since the original upload notified it. No rename-at-overlap check
    /// either, because this runs from `begin_frame` after
    /// `PassState::reset_frame`, so no draw of this frame has read the
    /// destination yet.
    ///
    /// Returns `false` when the buffer no longer has a `Staged` device
    /// buffer, which means the game released it and the lost upload has
    /// nowhere left to land. A live entry with a NULL `device_buffer`
    /// (the failed-warmup placeholder) also returns `false`. That state
    /// should be unreachable (a pending upload implies a successful
    /// `apply_stage_upload`, which implies a device buffer nothing nulls
    /// back out), but the placeholder makes it constructible, and a blit
    /// into buffer 0 is the one outcome worth a guard.
    fn reissue_stage_upload(&mut self, retry: &StagedUploadRetry) -> bool {
        let current_seq = self.current_submit_seq;
        let Some(dst_handle) = self
            .buffer_cache
            .get_mut(&retry.buffer_id)
            .filter(|s| s.is_staged && !s.device_buffer.is_null())
            .map(|s| {
                if current_seq > s.last_submit_seq {
                    s.last_submit_seq = current_seq;
                }
                s.device_buffer.raw()
            })
        else {
            return false;
        };
        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_buffer(
                &CopyBufferToBufferInfo {
                    src_buffer: retry.transient.raw(),
                    dst_buffer: dst_handle,
                    src_offset: 0,
                    dst_offset: u64::from(retry.dst_offset),
                    byte_size: u64::from(retry.size),
                },
            ));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        self.perf.bump_vbib_staging_upload();
        true
    }

    /// Allocate a fresh `StorageModePrivate` device buffer for a `Staged` VB/IB.
    ///
    /// Backs a rename-at-overlap, and the lazy recreate after a failed
    /// warmup create. Returns `None` on create failure; each caller has its
    /// own fallback (overwrite the live buffer, drop the upload, drop the
    /// draw).
    fn alloc_fresh_device_buffer(
        &self,
        buffer_id: BufferId,
        length: u64,
    ) -> Option<MetalHandle<MTLBufferKind>> {
        let desc = BufferCreateDesc {
            backing_ptr: 0,
            length,
            id: buffer_id.raw(),
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::VbIbDevice,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "alloc_fresh_device_buffer: CreateBuffer failed (status={status:#x}, len={length})"
            );
            return None;
        }
        Some(handle)
    }

    /// Drain API-thread-queued texture-staging warmups into one batched `CreateBuffersBatch` thunk.
    ///
    /// Must run after `drain_texture_warmups` — each entry's handle
    /// slots into the matching `TextureGpuState` already inserted by
    /// texture drain.
    fn drain_staging_warmups(&mut self, warmups: Vec<StagingWarmupEntry>) {
        if warmups.is_empty() {
            return;
        }
        let storage_mode = buffer_storage_mode(self.gpu_caps.unified_memory);
        let descs: Vec<BufferCreateDesc> = warmups
            .iter()
            .map(|w| BufferCreateDesc {
                backing_ptr: w.backing_ptr,
                length: w.backing_len,
                id: w.texture_id.raw(),
                storage_mode,
                kind: BufferKind::TexStaging,
            })
            .collect();
        let mut handles = vec![MetalHandle::<MTLBufferKind>::NULL; descs.len()];
        let status = self.batch_create_buffers(&descs, &mut handles);
        if status != 0 {
            error!(
                target: LOG_TARGET,
                "drain_staging_warmups: CreateBuffersBatch status={status:#x} (count={})",
                warmups.len()
            );
        }
        let current_seq = self.current_submit_seq;
        for (warmup, handle) in warmups.into_iter().zip(handles) {
            if handle.is_null() {
                continue;
            }
            let level = warmup.level as usize;
            let Some(state) = self.texture_cache.get_mut(&warmup.texture_id) else {
                // Texture warmup must have failed; orphan the staging
                // wrapper to keep refcounts straight. The Arc keepalive
                // travels with the retention entry so the wrapper
                // outlives the page-backing it was created against.
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "drain_staging_warmups: parent texture missing from cache, orphaning staging handle"
                );
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Buffer,
                        handle: handle.raw(),
                        page_box: None,
                        staging_arc: Some(warmup.keepalive),
                        seq: current_seq,
                        from_texture: true,
                    });
                continue;
            };
            // Slot vacant by construction (we just installed it with
            // `MipStagingBuffer::default()` in `drain_texture_warmups`).
            // Anything else means a lazy create raced us — orphan.
            if state.mip_staging_buffers[level].handle.is_null() {
                state.mip_staging_buffers[level] = MipStagingBuffer {
                    handle,
                    backing_ptr: warmup.backing_ptr,
                    length: warmup.backing_len,
                    keepalive: Some(warmup.keepalive),
                };
            } else {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "drain_staging_warmups: staging slot already populated, orphaning fresh handle"
                );
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Buffer,
                        handle: handle.raw(),
                        page_box: None,
                        staging_arc: Some(warmup.keepalive),
                        seq: current_seq,
                        from_texture: true,
                    });
            }
        }
    }

    fn begin_frame(&mut self, frame: &FrameData) {
        self.scratch.clear();
        // Cached const-slice pointers alias the previous frame's
        // arena which is about to be cleared / reused. Drop them so
        // emit_draw re-bumps on the first dirty draw of the new frame.
        self.vs_const_scratch_cache = None;
        self.ps_const_scratch_cache = None;
        // FF VS scratch cache pointed into the previous frame's arena
        // (about to drop). Drop the cached slice; next FF draw re-bumps
        // from the persistent mirror.
        self.ff_vs_const_scratch_cache = None;
        self.frame_blit_commands.clear();
        self.upload_pass_textures.clear();
        self.flags.remove(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        mtld3d_shared::crumb!("phase:BfRecl");
        self.reclaim_retired_blit_retention();
        if frame.coherent_seq_ptr != 0 {
            self.coherent_seq_ptr = frame.coherent_seq_ptr;
        }
        if frame.retained_bytes_ptr != 0 {
            self.retained_bytes_ptr = frame.retained_bytes_ptr;
        }
        if frame.failed_submit_seq_ptr != 0 {
            self.failed_seq_ptr = frame.failed_submit_seq_ptr;
        }
        if frame.upload_coherent_seq_ptr != 0 {
            self.upload_coherent_seq_ptr = frame.upload_coherent_seq_ptr;
        }
        self.current_submit_seq = frame.submit_seq;
        self.backbuffer_width = frame.backbuffer_width;
        self.backbuffer_height = frame.backbuffer_height;
        self.device_handle = frame.device_handle;
        self.perf.begin_frame(frame.perf());
        // Drain VB/IB retention entries whose seq has retired on the
        // GPU. Intake of *this* frame's entries is deferred to
        // `intake_vbib_retentions`, called after the op loop in
        // `run_frame` — doing it here would remove a cache entry whose
        // backing a same-frame draw closure still references (via a
        // pre-Lock snapshot), forcing `ensure_vb` to rebuild and
        // re-destroy an MTLBuffer wrapper within one frame.
        mtld3d_shared::crumb!("phase:BfDrain");
        self.drain_retired_resource_retention();
        mtld3d_shared::crumb!("phase:BfVisIn");
        self.intake_visibility();
        mtld3d_shared::crumb!("phase:BfVisRst");
        self.visibility.reset_frame();
        mtld3d_shared::crumb!("phase:BfPassRst");
        // Keep the seen-rt sets when the previous submit was a mid-frame flush
        // (the D3D9 frame did not end there); `finalize_submit` consumes the
        // flag by the time this reads it.
        self.pass_state
            .reset_frame(&mtld3d_core::passes::FrameReset {
                backbuffer: frame.backbuffer_handle,
                backbuffer_srgb: frame.backbuffer_srgb_handle,
                backbuffer_msaa: frame.backbuffer_msaa_handle,
                backbuffer_msaa_srgb: frame.backbuffer_msaa_srgb_handle,
                backbuffer_sample_count: frame.backbuffer_sample_count,
                backbuffer_size: (frame.backbuffer_width, frame.backbuffer_height),
                backbuffer_format: frame.backbuffer_format,
                depth_texture: frame.depth_texture,
                // The frame's default depth attachment is created at the
                // rasterized back-buffer size so it matches the colour one
                // exactly, and `Clear` measures the viewport against that.
                depth_size: if frame.depth_texture.is_null() {
                    (0, 0)
                } else {
                    (
                        frame.render_scale.dimension(frame.backbuffer_width),
                        frame.render_scale.dimension(frame.backbuffer_height),
                    )
                },
                depth_has_stencil: frame.flags.contains(FrameDataFlags::DEPTH_HAS_STENCIL),
                render_scale: frame.render_scale,
                continues_frame: self.prev_submit_no_present,
            });
        // After `reset_frame`: a replayed upload is a frame-leading blit,
        // and the rename-at-overlap bookkeeping it must not trip on still
        // holds the previous frame's draws until the reset above runs.
        mtld3d_shared::crumb!("phase:BfUpRec");
        self.settle_pending_uploads();
        mtld3d_shared::crumb!("phase:BfDone");
    }

    /// Finalize visibility queries whose Issue(END) frame has retired on the GPU.
    ///
    /// Then release retired buffers back into the pool's free list.
    /// Delegates to `VisibilityQueryState::intake_completed` for the
    /// sum + pool release. Called from `begin_frame` each frame, plus
    /// on-demand via the `IntakeVisibility` message when an app polls
    /// `GetData(D3DGETDATA_FLUSH)` between frames.
    pub fn intake_visibility(&mut self) {
        let coherent = if self.coherent_seq_ptr == 0 {
            0
        } else {
            // SAFETY: `coherent_seq_ptr` is a PE-heap `Arc<AtomicU64>` raw
            // pointer kept alive by the device-side `Arc`; nonzero here
            // means the encoder has been wired up and the Arc is still
            // live.
            unsafe { SharedCounter::new(self.coherent_seq_ptr) }.load(Ordering::Acquire)
        };
        self.visibility.intake_completed(coherent);
    }

    /// Cross-field adapter for `EncoderPerfState::log_frame_summary`.
    ///
    /// It reaches the pass list (on `self.pass_state`) and the cache
    /// sizes (on `self.*_cache`). Lives on `FrameEncoder` so the
    /// disjoint-field borrow between `&mut self.perf` and
    /// `&self.pass_state` / `&self.*_cache` is obvious to the borrow
    /// checker — splitting via `self.perf.log_frame_summary(self.pass_state…)`
    /// from an outside caller would not compile.
    ///
    /// Emit the per-frame perf summary. Reads the frame's `passes` and
    /// `scratch` from the just-submitted `payload` rather than from
    /// `self`: `finalize_submit` has already swapped the live arena and
    /// taken the passes out of `self` into the payload, so the payload
    /// is where this frame's state now lives. `submit_cycles` /
    /// `drawable_wait` / `status` are settled by the time this runs, and
    /// the payload is recycled only afterwards — reproducing the
    /// pre-split ordering exactly.
    fn log_perf_summary(&mut self, payload: &FramePayload, ctx: &FrameSummaryContext, status: i32) {
        let caches = self.cache_sizes(&payload.scratch);
        let cmd_vec_realloc_bytes = self.pass_state.take_cmd_vec_realloc_bytes();
        // One getrusage unix_call per 5 s window, only when the summary is
        // both enabled and about to emit; every other frame passes None.
        let task_faults = (perf_enabled() && self.perf.window_due()).then(|| {
            let mut p = GetTaskFaultsParams {
                minor_faults: 0,
                major_faults: 0,
            };
            let _ = unix_call(&mut p);
            TaskFaults {
                minor: p.minor_faults,
                major: p.major_faults,
            }
        });
        self.perf.log_frame_summary(
            &caches,
            &payload.passes,
            ctx,
            status,
            cmd_vec_realloc_bytes,
            task_faults,
        );
    }

    /// Cache-length snapshot handed to `EncoderPerfState::log_frame_summary`.
    ///
    /// Walks every cache `HashMap` exactly once; cheap even at debug
    /// log levels because `HashMap::len()` is O(1). `scratch` is passed in
    /// (the just-submitted payload's filled arena) since `self.scratch` is
    /// already the clean arena swapped in for the next frame.
    fn cache_sizes(&self, scratch: &ScratchArena) -> CacheSizes {
        CacheSizes {
            textures: self.texture_cache.len(),
            pipelines: self.pipeline_cache.len(),
            samplers: self.sampler_cache.len(),
            programs: self.program_cache.len(),
            libs: self.lib_cache.len(),
            depth_states: self.depth_stencil_cache.len(),
            scratch_small_blocks: scratch.small_chunk_count(),
            scratch_oversized_blocks: scratch.oversized_chunk_count(),
            scratch_bytes: scratch.capacity_bytes(),
            cmd_vec_capacity_bytes: self.pass_state.cmd_vec_capacity_bytes(),
            pending_blit_retention_depth: self.pending_blit_retention.len(),
            pending_resource_retention_depth: self.pending_resource_retention.len(),
            pagebox_pool_bytes: crate::page_box_pool::PAGEBOX_POOL.pooled_bytes() as u64,
        }
    }

    /// Acquire a clean [`FramePayload`] to swap this frame's buffers into.
    ///
    /// Reuses a recycled one if available; otherwise allocates a fresh one
    /// until [`SUBMIT_PAYLOAD_CAP`] exist, after which it blocks on the
    /// return channel — the backpressure that bounds render-ahead to ≤1
    /// frame.
    fn acquire_clean_payload(&mut self) -> FramePayload {
        self.drain_returned_payloads();
        if let Some(payload) = self.payload_pool.pop() {
            return payload;
        }
        if self.submit_payloads_total < SUBMIT_PAYLOAD_CAP {
            self.submit_payloads_total += 1;
            return FramePayload::default();
        }
        // At the cap with an empty pool → every payload is in flight. Block
        // until the submit thread hands one back. This wait is the encoder's
        // backpressure stall (the submit thread is the pacing stage, usually
        // GPU/present-bound); time it separately so it isn't billed as
        // encoder CPU.
        mtld3d_shared::crumb!("phase:SubmitBackpr");
        let mut stall_tsc: u64 = 0;
        let returned = {
            let _stall = mtld3d_core::perf::CycleSetTimer::start(&raw mut stall_tsc);
            self.submit_return_rx
                .recv()
                .expect("submit thread alive while frames are in flight")
        };
        self.perf.add_submit_stall_cycles(stall_tsc);
        self.reclaim_returned(returned);
        self.payload_pool
            .pop()
            .expect("reclaim_returned refilled the pool")
    }

    /// Non-blocking reclaim of any payloads the submit thread has finished.
    ///
    /// Recycles their buffers and folds back their status /
    /// drawable-wait. Called at frame head so the command-vec pool is
    /// warm before the op loop, and inside `acquire_clean_payload`.
    fn drain_returned_payloads(&mut self) {
        while let Ok(returned) = self.submit_return_rx.try_recv() {
            self.reclaim_returned(returned);
        }
    }

    /// Fold one returned frame back in.
    ///
    /// Decrement the in-flight count, latch its status + drawable-wait
    /// for the next `Async` summary, log on failure, and recycle the
    /// payload's buffers.
    fn reclaim_returned(&mut self, returned: ReturnedPayload) {
        self.submit_in_flight = self.submit_in_flight.saturating_sub(1);
        self.last_submit_status = returned.status;
        // The unix side measures this one in nanoseconds (its counter is not
        // ours), so it converts to our cycles here, where every other perf
        // bucket is denominated.
        self.perf
            .set_drawable_wait_cycles(ns_to_cycles(returned.drawable_wait_ns));
        self.perf.set_submit_exec_cycles(returned.submit_exec_tsc);
        if returned.status != 0 {
            error!(
                target: LOG_TARGET,
                "encoder: SubmitFrame failed (status={:#x})",
                returned.status,
            );
        }
        reclaim_payload(self, returned.payload);
    }

    /// Hand a finalized packet to the submit thread (`Async` mode).
    ///
    /// Blocks only if the cap-1 work channel is full, i.e. a prior
    /// submit is still in progress — the other half of the render-ahead
    /// backpressure.
    fn dispatch_submit(&mut self, packet: SubmitPacket) {
        self.submit_in_flight += 1;
        if self.submit_work_tx.send(packet).is_err() {
            // Submit thread is gone (only possible post-shutdown). Undo the
            // count so a later barrier doesn't wait forever.
            self.submit_in_flight = self.submit_in_flight.saturating_sub(1);
        }
    }

    /// Barrier: block until every in-flight async submit has been issued and its payload returned.
    ///
    /// Recycles each. After this, no `SubmitFrame` runs on the submit
    /// thread, so a synchronous submit / GPU wait / capture / reset can
    /// proceed with correct ordering.
    fn drain_submit_thread(&mut self) {
        while self.submit_in_flight > 0 {
            let returned = self
                .submit_return_rx
                .recv()
                .expect("submit thread alive while frames are in flight");
            self.reclaim_returned(returned);
        }
    }

    /// Tag the current pass with "this draw wants to write color".
    ///
    /// Applied iff `D3DRS_COLORWRITEENABLE != 0`. Forwarded into
    /// `PassState` so Rule H can strip the color attachment from passes
    /// where every draw closed with `mask == 0`. Opens a pass first if
    /// none is live.
    pub fn note_draw_color_write_mask(&mut self, mask: u32) {
        self.pass_state.note_draw_color_write_mask(mask);
    }

    pub fn emit_command(&mut self, cmd: Command) {
        // Pass-boundary re-arm for active occlusion queries: if a new
        // pass is about to open while at least one query is active,
        // bump to a fresh slot and emit a Counting-mode set *before*
        // the user command, so Metal continues accumulating into a
        // new slot on the new pass. Skip when `cmd` is itself a
        // SetVisibilityResultMode (that's the Begin/End path
        // allocating their own slot).
        if self.pass_state.current_pass_closed()
            && self.visibility.active_count() > 0
            && cmd.cmd != CommandType::SetVisibilityResultMode as u32
            && !self.visibility.exhausted_this_frame()
        {
            if let Some(slot) = self.visibility.bump_slot() {
                let re_arm = Command::set_visibility_result_mode(
                    VisibilityResultMode::Counting,
                    slot * SLOT_BYTES,
                );
                self.pass_state.emit_command(re_arm);
            } else {
                self.mark_visibility_exhausted();
            }
        }
        self.pass_state.emit_command(cmd);
    }

    /// Arm a visibility query.
    ///
    /// Captures the current frame's `submit_seq` as BEGIN and emits a
    /// Counting-mode command onto the current pass. Ensures a per-frame
    /// visibility buffer exists, allocating or pulling from the pool on
    /// first call in the frame.
    pub fn begin_visibility_query(&mut self, core: &Arc<VisibilityQueryCore>) {
        if self.visibility.exhausted_this_frame() {
            core.begin(self.current_submit_seq, 0);
            self.visibility.inc_active();
            return;
        }
        if !self.ensure_visibility_buffer() {
            self.mark_visibility_exhausted();
            core.begin(self.current_submit_seq, 0);
            self.visibility.inc_active();
            return;
        }
        let Some(slot) = self.visibility.bump_slot() else {
            self.mark_visibility_exhausted();
            core.begin(self.current_submit_seq, 0);
            self.visibility.inc_active();
            return;
        };
        core.begin(self.current_submit_seq, slot);
        self.visibility.inc_active();
        let cmd =
            Command::set_visibility_result_mode(VisibilityResultMode::Counting, slot * SLOT_BYTES);
        // `emit_command` opens a fresh Metal pass when the prior one was
        // closed — e.g. a `SetRenderTarget` immediately before `Issue(BEGIN)`,
        // which closes the pass so the visibility mode is the first command of
        // a new encoder. Like the clear-quad paths, reset the per-draw
        // `last_bound` dedup across that encoder boundary so the following draw
        // re-emits its pipeline + bindings: the fresh encoder starts with none,
        // and `emit_draw`'s own reset would no-op here since this call already
        // opened the pass (a draw with no pipeline bound faults in Metal).
        let passes_before = self.pass_state.passes().len();
        self.pass_state.emit_command(cmd);
        self.reset_last_bound_if_pass_opened(passes_before);
    }

    /// Close a visibility query.
    ///
    /// Bumps to a fresh slot so summation sees a half-open `[begin, end)`
    /// range, transitions the Metal encoder to Disabled (or re-arms to
    /// Counting if other queries are still active), and queues the core
    /// onto the pending list to be finalized once the GPU has retired
    /// this frame.
    pub fn end_visibility_query(&mut self, core: Arc<VisibilityQueryCore>) {
        let submit_seq = self.current_submit_seq;
        if self.visibility.exhausted_this_frame() {
            // Match the safe-fallback span: end == begin, sum = 0 (but
            // the fallback path below will finalize to u32::MAX at
            // intake, not sum the buffer).
            core.end(submit_seq, core.offset_begin());
            self.visibility.dec_active();
            self.visibility.push_pending(submit_seq, core);
            return;
        }
        let Some(slot) = self.visibility.bump_slot() else {
            self.mark_visibility_exhausted();
            core.end(submit_seq, core.offset_begin());
            self.visibility.dec_active();
            self.visibility.push_pending(submit_seq, core);
            return;
        };
        core.end(submit_seq, slot);
        self.visibility.dec_active();
        let mode = if self.visibility.active_count() == 0 {
            VisibilityResultMode::Disabled
        } else {
            VisibilityResultMode::Counting
        };
        let cmd = Command::set_visibility_result_mode(mode, slot * SLOT_BYTES);
        // Symmetric with `begin_visibility_query`: if this mode-set opens a
        // fresh pass, reset `last_bound` so any later draw in the frame re-emits
        // its bindings across the encoder boundary.
        let passes_before = self.pass_state.passes().len();
        self.pass_state.emit_command(cmd);
        self.reset_last_bound_if_pass_opened(passes_before);
        self.visibility.push_pending(submit_seq, core);
    }

    /// Reserve a visibility buffer for the current frame.
    ///
    /// Returns true on success, false if buffer allocation failed
    /// (caller should mark the frame exhausted and finalize queries with
    /// `u32::MAX`).
    fn ensure_visibility_buffer(&mut self) -> bool {
        if !self.visibility.current_buffer_handle().is_null() {
            return true;
        }
        // Pool-acquired buffer first — reuses a PageBox + Metal wrapper.
        if let Some(mut reused) = self.visibility.try_acquire_reusable() {
            // Zero the backing so prior-frame counter values don't
            // leak into slots the GPU didn't touch this frame.
            zero_page_box(reused.backing_mut());
            self.visibility.install_current_buffer(reused);
            return true;
        }
        // Pool empty: allocate a fresh PageBox + CreateBuffer.
        let mut backing = PageBox::new_zeroed((MAX_SLOTS * SLOT_BYTES) as usize);
        let backing_ptr = backing.as_mut_ptr() as u64;
        let length = backing.len() as u64;
        let desc = BufferCreateDesc {
            backing_ptr,
            length,
            id: 0,
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::Visibility,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "ensure_visibility_buffer: CreateBuffer failed (status={status:#x})"
            );
            return false;
        }
        let fresh = RetiredVisibilityBuffer::new(backing, handle, 0);
        self.visibility.install_current_buffer(fresh);
        true
    }

    /// The shared fan index buffer, covering at least `primitive_count` triangles.
    ///
    /// Grows geometrically (capped at the 16-bit pattern's reach) so a scene
    /// of ever-longer fans allocates a logarithmic number of times. Returns
    /// 0 when Metal refuses the allocation.
    pub fn fan_index_buffer(&mut self, primitive_count: u32) -> u64 {
        const FIRST_TRIANGLES: u32 = 256;
        if !self.fan_index_buffer.handle.is_null()
            && self.fan_index_buffer.triangles >= primitive_count
        {
            return self.fan_index_buffer.handle.raw();
        }
        let triangles = primitive_count
            .max(self.fan_index_buffer.triangles.saturating_mul(2))
            .clamp(FIRST_TRIANGLES, FAN_PATTERN_MAX_TRIANGLES);
        let mut backing = PageBox::new_zeroed(fan_pattern_bytes(triangles));
        fill_fan_pattern_u16(backing.as_mut_slice(), triangles);
        let length = backing.len() as u64;
        let desc = BufferCreateDesc {
            backing_ptr: backing.as_mut_ptr() as u64,
            length,
            id: 0,
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::VbIb,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "fan_index_buffer: CreateBuffer failed (triangles={triangles}, status={status:#x})"
            );
            return 0;
        }
        // The pattern was written by the CPU before the wrap; on managed
        // storage the GPU has to be told (no-op on UMA).
        self.enqueue_notify_buffer_did_modify_range(handle.raw(), 0, length);
        let grown_out = core::mem::replace(
            &mut self.fan_index_buffer,
            FanIndexBuffer {
                backing: Some(backing),
                handle,
                triangles,
            },
        );
        if !grown_out.handle.is_null() {
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Buffer,
                    handle: grown_out.handle.raw(),
                    page_box: grown_out.backing,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: false,
                });
        }
        handle.raw()
    }

    fn mark_visibility_exhausted(&mut self) {
        self.visibility.mark_exhausted();
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "visibility-query slot budget exhausted for this frame \
             — overflowing queries finalize to u32::MAX"
        );
    }

    pub fn end_current_pass(&mut self, caller: &'static str) {
        self.pass_state.end_current_pass(caller);
    }

    /// Index of the currently-open pass within the frame.
    ///
    /// Proxies `PassState::current_pass_index` for the
    /// `mtld3d::d3d9::decal` trace probe in `emit_draw`.
    #[must_use]
    pub const fn current_pass_index(&self) -> usize {
        self.pass_state.current_pass_index()
    }

    /// Metal handle of the currently-bound depth attachment.
    ///
    /// Proxies `PassState::current_depth_texture` for the
    /// `mtld3d::d3d9::caster` trace probe in `emit_draw`.
    /// Record what the depth attachment just bound looks like.
    ///
    /// A snapshot copy of it (see [`Self::depth_snapshot_for_sampling`]) has
    /// to match in size and format, and the encoder's texture cache keeps
    /// only handles.
    pub const fn set_depth_attachment_desc(
        &mut self,
        width: u32,
        height: u32,
        format: mtld3d_shared::mtl::PixelFormat,
    ) {
        self.depth_attachment_desc = (width, height, format);
    }

    /// Note that the bound depth attachment is about to be written.
    ///
    /// Depth-writing draws and depth clears call this; a snapshot taken
    /// before the bump no longer reflects the attachment.
    pub const fn bump_depth_write_epoch(&mut self) {
        self.depth_write_epoch += 1;
    }

    /// Metal handle of the snapshot copy kept for a depth attachment, or 0.
    ///
    /// Diagnostics only: the frame dump reads the copy back, since that is
    /// what depth-sampling draws actually saw.
    #[must_use]
    pub fn depth_snapshot_handle(&self, src_raw: u64) -> u64 {
        self.depth_snapshots
            .get(&src_raw)
            .map_or(0, |s| s.handle.raw())
    }

    /// Resolve the bound depth attachment into `dst` (the RESZ hack).
    ///
    /// The magic `SetRenderState(POINTSIZE, 0x7fa05000)` asks for the
    /// current depth-stencil contents in the texture bound at stage 0.
    /// From a single-sampled depth surface that is a full-surface depth
    /// blit queued ahead of the next pass; from a multisampled one the
    /// samples have to be resolved, which Metal offers on a render pass
    /// rather than on the blit encoder. The destination keeps its own
    /// contents when no depth attachment is bound (the resolve is then a
    /// no-op, as on hardware).
    pub fn resolve_depth_to_texture(&mut self, dst: u64, dst_w: u32, dst_h: u32) {
        let src = self.pass_state.current_depth_texture();
        if src.is_null() || dst == 0 {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "RESZ resolve without a bound depth attachment or destination texture — skipped"
            );
            return;
        }
        let (width, height, _) = self.depth_attachment_desc;
        if width != dst_w || height != dst_h {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "RESZ resolve size mismatch: depth {width}x{height} vs destination \
                 {dst_w}x{dst_h} — skipped"
            );
            return;
        }
        if self.pass_state.current_depth_sample_count() > 1 {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "RESZ resolve: resolving the bound {}x multisampled depth attachment \
                 ({width}x{height}) into the stage-0 texture",
                self.pass_state.current_depth_sample_count()
            );
            // SAFETY: `dst` is a Metal texture handle the encoder's typed
            // cache produced through `.raw()` and checked non-zero above.
            let destination = unsafe { MetalHandle::<MTLTextureKind>::new(dst) };
            self.pass_state
                .resolve_depth_attachment(destination, DepthResolveFilter::Sample0);
            return;
        }
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "RESZ resolve: copying the bound depth attachment ({width}x{height}) into the \
             stage-0 texture"
        );
        self.end_current_pass("resz");
        self.pass_state.push_pending_leading_blit(BlitCommand {
            cmd: BlitCommandType::CopyTextureToTexture as u32,
            mip_level: 0,
            src_handle: src.raw(),
            dst_handle: dst,
            src_offset: 0,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: width,
            region_h: height,
            dst_offset: 0,
            byte_size: 0,
            depth: 1,
            bytes_per_image: 0,
            dst_mip_level: 0,
            dst_slice: 0,
            src_slice: 0,
        });
    }

    /// Resolve one multisampled depth surface into a single-sampled one.
    ///
    /// The multisample arm of the depth-to-depth `StretchRect`: D3D9 resolves
    /// the samples on a copy that leaves a multisampled surface, and Metal
    /// takes a depth resolve on a render pass rather than on the blit encoder,
    /// which refuses a sample-count change outright. Both handles address the
    /// whole surface at the same extent and depth format, which the caller has
    /// established. Sample zero is the reduction: D3D9 defines no filter for
    /// this and it is what the hardware the copy was written for delivers.
    pub fn resolve_depth_surface(
        &mut self,
        src: MetalHandle<MTLTextureKind>,
        dst: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
    ) {
        let resolved = self.pass_state.resolve_depth_texture(&DepthResolve {
            source: src,
            level: 0,
            size: (width, height),
            destination: dst,
            filter: DepthResolveFilter::Sample0,
            source_is_sampleable: false,
        });
        if resolved {
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "StretchRect: resolving a multisampled {width}x{height} depth surface \
                 into a single-sampled one"
            );
        } else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "StretchRect: depth resolve skipped, source or destination texture unresolved \
                 (src={:#x}, dst={:#x})",
                src.raw(),
                dst.raw()
            );
        }
    }

    /// Copy one depth texture's contents into another (`depth.aliasSameSize`).
    ///
    /// The bind-time carry for engines that expect equal-size depth-stencil
    /// surfaces to share one physical allocation: the destination is about to
    /// be bound as the depth attachment and must open on the source's
    /// contents. Same shape as the RESZ resolve: close the pass, queue a
    /// full-surface blit ahead of the next one (which registers the
    /// destination as blit-written, so its first-use load stays `Load`).
    pub fn carry_depth_contents(&mut self, src_id: TextureId, dst_id: TextureId, w: u32, h: u32) {
        let src = self.get_texture_handle_by_id(src_id);
        let dst = self.get_texture_handle_by_id(dst_id);
        if src == 0 || dst == 0 || src == dst {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "depth.aliasSameSize: carry skipped (unresolved handle or identical textures)"
            );
            return;
        }
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "depth.aliasSameSize: carrying {w}x{h} depth contents across a same-size bind"
        );
        self.end_current_pass("depth-alias");
        self.pass_state.push_pending_leading_blit(BlitCommand {
            cmd: BlitCommandType::CopyTextureToTexture as u32,
            mip_level: 0,
            src_handle: src,
            dst_handle: dst,
            src_offset: 0,
            bytes_per_row: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: w,
            region_h: h,
            dst_offset: 0,
            byte_size: 0,
            depth: 1,
            bytes_per_image: 0,
            dst_mip_level: 0,
            dst_slice: 0,
            src_slice: 0,
        });
    }

    /// A readable copy of the bound depth attachment, for a draw that samples it.
    ///
    /// Metal forbids reading a texture that is an attachment of the running
    /// pass, and Apple GPUs return garbage rather than the depth. D3D9 allows
    /// it (a deferred renderer binds its INTZ scene depth for the depth test
    /// and samples it for position reconstruction in the same draws), with
    /// the values as of the last write. So: close the pass, queue a blit that
    /// copies the attachment into a scratch depth texture of the same size
    /// and format, and hand that copy out. The copy stays valid until a
    /// depth write or clear bumps the epoch, so a run of light-volume draws
    /// costs one copy. Returns 0 when no depth attachment is bound or the
    /// scratch texture cannot be created.
    pub fn depth_snapshot_for_sampling(&mut self) -> u64 {
        let src = self.pass_state.current_depth_texture();
        if src.is_null() {
            return 0;
        }
        let (width, height, format) = self.depth_attachment_desc;
        if width == 0 || height == 0 {
            return 0;
        }
        let epoch = self.depth_write_epoch;
        let stale_handle = match self.depth_snapshots.get(&src.raw()) {
            Some(snap) if snap.width == width && snap.height == height && snap.format == format => {
                None
            }
            Some(snap) => Some(snap.handle),
            None => None,
        };
        if let Some(old) = stale_handle {
            // Same source handle, different geometry: the texture behind the
            // handle was recreated. Retire the old copy behind the GPU.
            self.depth_snapshots.remove(&src.raw());
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: old.raw(),
                    page_box: None,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: false,
                });
        }
        if !self.depth_snapshots.contains_key(&src.raw()) {
            // The generic texture path: a depth format with DEPTH_STENCIL |
            // RENDER_TARGET usage comes back RenderTarget | ShaderRead, which
            // the copy needs on both ends (blit destination, then sampled).
            let desc = TextureCreateDesc {
                tex_id: src.raw(),
                width,
                height,
                depth: 1,
                levels: 1,
                pixel_format: format,
                storage_mode: StorageMode::Private,
                flags: TextureCreateFlags::empty(),
                swizzle_r: mtld3d_shared::mtl::Swizzle::Red,
                swizzle_g: mtld3d_shared::mtl::Swizzle::Green,
                swizzle_b: mtld3d_shared::mtl::Swizzle::Blue,
                swizzle_a: mtld3d_shared::mtl::Swizzle::Alpha,
                usage_flags: TextureUsage::DEPTH_STENCIL | TextureUsage::RENDER_TARGET,
            };
            let mut handles = [MetalHandle::<MTLTextureKind>::NULL];
            // Depth formats never have an sRGB twin; the slot stays NULL.
            let mut srgb_handles = [MetalHandle::<MTLTextureKind>::NULL];
            let status = self.batch_create_textures(&[desc], &mut handles, &mut srgb_handles);
            if status != 0 || handles[0].is_null() {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "depth snapshot: creating a {width}x{height} {format:?} copy failed \
                     ({status:#x}); the draw samples the live attachment"
                );
                return 0;
            }
            mtld3d_shared::log_once_info!(
                target: LOG_TARGET,
                "depth snapshot: a draw samples the bound depth attachment; copying it \
                 ({width}x{height} {format:?}) before such draws"
            );
            self.depth_snapshots.insert(
                src.raw(),
                DepthSnapshot {
                    handle: handles[0],
                    width,
                    height,
                    format,
                    epoch: epoch.wrapping_sub(1),
                },
            );
        }
        let (dst, needs_copy) = {
            let snap = self
                .depth_snapshots
                .get_mut(&src.raw())
                .expect("inserted above");
            let needs_copy = snap.epoch != epoch;
            snap.epoch = epoch;
            (snap.handle, needs_copy)
        };
        if needs_copy {
            self.end_current_pass("depth_snapshot");
            self.pass_state.push_pending_leading_blit(BlitCommand {
                cmd: BlitCommandType::CopyTextureToTexture as u32,
                mip_level: 0,
                src_handle: src.raw(),
                dst_handle: dst.raw(),
                src_offset: 0,
                bytes_per_row: 0,
                origin_x: 0,
                origin_y: 0,
                region_w: width,
                region_h: height,
                dst_offset: 0,
                byte_size: 0,
                depth: 1,
                bytes_per_image: 0,
                dst_mip_level: 0,
                dst_slice: 0,
                src_slice: 0,
            });
        }
        dst.raw()
    }

    /// Scratch texture staging a `StretchRect` whose two endpoints are one texture.
    ///
    /// D3D9 reads the whole source region before it writes any of the
    /// destination, so an overlapping or scaled copy inside one texture needs
    /// somewhere to hold the source first. The scratch is kept per source
    /// handle and grown to the largest region ever asked of it, so a game
    /// scrolling the same surface every frame allocates once. `src_handle` is
    /// the copy's one texture, used as the cache key. Returns 0 when
    /// `None` when the texture cannot be created; the caller then drops the copy
    /// and says so. The returned dimensions are the scratch's own, which the
    /// scaled route needs to build its texcoord transform.
    pub fn stretch_scratch_texture(
        &mut self,
        src_handle: u64,
        size: (u32, u32),
        format: PixelFormat,
    ) -> Option<(u64, u32, u32)> {
        let (want_w, want_h) = size;
        if src_handle == 0 || want_w == 0 || want_h == 0 {
            return None;
        }
        let stale = match self.stretch_scratch.get(&src_handle) {
            Some(scratch)
                if scratch.format == format
                    && scratch.width >= want_w
                    && scratch.height >= want_h =>
            {
                return Some((scratch.handle.raw(), scratch.width, scratch.height));
            }
            Some(scratch) => Some((scratch.handle, scratch.width, scratch.height)),
            None => None,
        };
        // Grow rather than shrink: a later call with the earlier geometry then
        // hits the cache instead of trading one allocation for another.
        let (width, height) =
            stale.map_or((want_w, want_h), |(_, w, h)| (w.max(want_w), h.max(want_h)));
        if let Some((old, _, _)) = stale {
            self.stretch_scratch.remove(&src_handle);
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: old.raw(),
                    page_box: None,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: false,
                });
        }
        // `ShaderRead` comes free with every texture the unix side creates,
        // which is all the scaled route needs: the scratch is a blit
        // destination and then either a blit source or a sampled source.
        let desc = TextureCreateDesc {
            tex_id: src_handle,
            width,
            height,
            depth: 1,
            levels: 1,
            pixel_format: format,
            storage_mode: StorageMode::Private,
            flags: TextureCreateFlags::empty(),
            swizzle_r: Swizzle::Red,
            swizzle_g: Swizzle::Green,
            swizzle_b: Swizzle::Blue,
            swizzle_a: Swizzle::Alpha,
            usage_flags: TextureUsage::empty(),
        };
        let mut handles = [MetalHandle::<MTLTextureKind>::NULL];
        // The scratch is never sampled through an sRGB view; the slot stays NULL.
        let mut srgb_handles = [MetalHandle::<MTLTextureKind>::NULL];
        let status = self.batch_create_textures(&[desc], &mut handles, &mut srgb_handles);
        if status != 0 || handles[0].is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "StretchRect: creating a {width}x{height} {format:?} scratch for a copy inside \
                 one texture failed ({status:#x}); the copy is dropped"
            );
            return None;
        }
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "StretchRect: copying between two rects of one surface; staging through a \
             {width}x{height} {format:?} scratch"
        );
        self.stretch_scratch.insert(
            src_handle,
            StretchScratch {
                handle: handles[0],
                width,
                height,
                format,
            },
        );
        Some((handles[0].raw(), width, height))
    }

    #[must_use]
    pub const fn current_depth_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.pass_state.current_depth_texture()
    }

    /// Whether the live pass carries the bound depth attachment.
    ///
    /// Proxies [`PassState::pass_binds_depth`] for the draw and clear-quad
    /// paths, which build the pipelines that have to agree with the pass on
    /// whether a depth and a stencil format are declared.
    #[must_use]
    pub const fn pass_binds_depth(&self) -> bool {
        self.pass_state.pass_binds_depth()
    }

    /// Record a caster draw against the currently-bound cascade depth handle.
    ///
    /// Called from `draw.rs::emit_draw`. The `PassState` implementation
    /// self-filters to known-sampleable handles via
    /// `seen_sampleable_depth_textures`.
    pub fn note_caster_draw(&mut self, depth_tex: MetalHandle<MTLTextureKind>) {
        self.pass_state.note_caster_draw(depth_tex);
    }

    /// `true` when `depth_tex` is a live handle bound as a sampleable shadow map this session.
    ///
    /// Proxies `PassState::is_depth_handle_sampleable` for the
    /// `mtld3d::d3d9::caster`/`cascade` trace probes, which classify a draw by
    /// the texture it targets. `current_depth_is_sampleable()` answers a
    /// different question (whether the surface bound right now is a shadow
    /// map) and so reads false for every draw against the scene depth.
    #[must_use]
    pub fn is_depth_handle_sampleable(&self, depth_tex: MetalHandle<MTLTextureKind>) -> bool {
        self.pass_state.is_depth_handle_sampleable(depth_tex)
    }

    /// Queue a `StretchRect` blit to run before the *next* pass.
    ///
    /// Caller must `flush_pending_clears()` and `end_current_pass()` first so
    /// the blit is correctly ordered after a `Clear` still waiting for a pass
    /// and between the just-ended pass's draws and the next pass's draws. If
    /// no further pass opens this frame, `submit` synthesises a trailing
    /// blit-only `PassDescriptor` to drain it.
    pub fn push_stretch_rect_blit(&mut self, blit: BlitCommand) {
        self.pass_state.push_pending_leading_blit(blit);
    }

    /// Materialize any pending clears as a pass on the current attachments.
    ///
    /// A `Clear` issued with no pass open waits for the next pass's load
    /// action; a blit queued in between would otherwise run before it and
    /// either read the pre-clear source or be wiped by the clear.
    pub fn flush_pending_clears(&mut self) {
        self.pass_state.flush_pending_clears();
    }

    /// Bind render target 0, with its subresource, alpha bit and multisample companion.
    pub fn set_color_render_target(&mut self, binding: &ColorRtBinding) {
        let (width, height) = binding.logical_size;
        self.pass_state.set_color_render_target_subresource(
            binding.texture,
            width,
            height,
            binding.format,
            binding.scale,
            binding.subresource,
        );
        // Kept in lockstep with the format: the Metal pixel format alone can't
        // distinguish X8R8G8B8 (no alpha) from A8R8G8B8 (both `Bgra8Unorm`).
        self.pass_state.set_color_rt_has_alpha(binding.has_alpha);
        // Likewise in lockstep: the setter above clears the companion, so a
        // single-sampled target can never inherit the previous one's.
        self.pass_state.set_color_msaa(
            binding.msaa_texture,
            binding.msaa_srgb_texture,
            binding.sample_count,
        );
    }

    /// Metal pixel format of the currently bound color RT.
    ///
    /// Read at draw time to key the pipeline cache on RT format so
    /// multiple passes against different formats don't share a pipeline.
    pub const fn current_color_format(&self) -> PixelFormat {
        self.pass_state.current_color_format()
    }

    /// Register a live sRGB twin view so a colour target bound later can attach it.
    pub fn register_srgb_twin(
        &mut self,
        twin: MetalHandle<MTLTextureKind>,
        base: MetalHandle<MTLTextureKind>,
    ) {
        self.pass_state.register_srgb_twin(twin, base);
    }

    /// Park a standalone colour target's textures on the retention queue.
    ///
    /// Called when the surface that owns them finalizes. Each view goes ahead
    /// of the texture it was made from, since it holds a retain on it, and
    /// the sRGB twin's registration is dropped with it so no later binding
    /// can resolve a view whose storage is gone. Every destroy is gated on
    /// the current submit seq, since a pass or blit already encoded this
    /// frame may still name any of the handles.
    pub fn retire_color_target(&mut self, target: &RetiredColorTarget) {
        self.pass_state.unregister_srgb_twin(target.srgb);
        let seq = self.current_submit_seq;
        for handle in [target.srgb, target.base, target.msaa_srgb, target.msaa] {
            if handle.is_null() {
                continue;
            }
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: handle.raw(),
                    page_box: None,
                    staging_arc: None,
                    seq,
                    from_texture: true,
                });
        }
    }

    /// Park a standalone depth-stencil target's texture on the retention queue.
    ///
    /// Called when the surface that owns it finalizes. The destroy is gated
    /// on the current submit seq, since a pass already encoded this frame may
    /// still attach the handle, and the pass state forgets the handle here so
    /// nothing binds or classifies it once the storage is gone.
    pub fn retire_depth_target(&mut self, depth: MetalHandle<MTLTextureKind>) {
        if depth.is_null() {
            return;
        }
        self.pass_state.retire_depth_texture(depth);
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Texture,
                handle: depth.raw(),
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });
    }

    /// Apply `D3DRS_SRGBWRITEENABLE` as the draw or `Clear` about to run sees it.
    pub fn set_srgb_write_enabled(&mut self, enabled: bool) {
        self.pass_state.set_srgb_write_enabled(enabled);
    }

    /// Whether the pass binds sRGB views, so the hardware encodes post-blend.
    ///
    /// Read at draw time: when it is set the pixel shader must NOT also
    /// apply the OETF, or the colour is encoded twice.
    #[must_use]
    pub const fn color_attachment_is_srgb(&self) -> bool {
        self.pass_state.pass_srgb_write()
    }

    /// Sample count of the currently bound color RT, 1 when single-sampled.
    ///
    /// Read at draw time into the pipeline key: Metal requires a pipeline's
    /// `rasterSampleCount` to match the pass's attachments.
    pub const fn current_color_sample_count(&self) -> u8 {
        self.pass_state.current_color_sample_count()
    }

    /// Whether the currently bound color RT's D3D format has a real alpha channel.
    ///
    /// Read at draw time into the pipeline snapshot's `COLOR_HAS_ALPHA`
    /// bit so destination-alpha blend factors clamp on alpha-less
    /// targets (X8R8G8B8).
    pub const fn current_color_rt_has_alpha(&self) -> bool {
        self.pass_state.current_color_rt_has_alpha()
    }

    /// Render targets 1..3 as the next pass attaches them, for the pipeline key.
    pub const fn current_extra_color_attachments(
        &self,
    ) -> mtld3d_core::pipeline_state::ExtraColorAttachments {
        self.pass_state.extra_color_attachments()
    }

    /// Bind or unbind render target `slot` (1..=3).
    ///
    /// `binding.logical_size` is the D3D9-reported extent and `binding.scale`
    /// what it is rasterized at, as for render target 0.
    pub fn set_extra_color_render_target(&mut self, slot: usize, binding: Option<ExtraColorSlot>) {
        self.pass_state.set_extra_color_render_target(slot, binding);
    }

    /// Render targets 1..3 as a clear-quad pipeline must declare them.
    ///
    /// The quad blends nothing, so the alpha bits are dropped to keep the
    /// pipeline key canonical.
    const fn clear_quad_extra_targets(&self) -> mtld3d_core::pipeline_state::ExtraColorAttachments {
        let mut extra = self.pass_state.extra_color_attachments();
        extra.has_alpha_mask = 0;
        extra
    }

    /// Run `f` once per colour target that is bound but outside the pass, with it bound alone.
    ///
    /// A render target 1..3 sized unlike target 0 is attached to no pass
    /// (the D3D9 rule), yet `Clear` still reaches it. Each such target gets
    /// the single-target clear treatment in turn, with depth unbound for the
    /// scoped pass (a depth attachment smaller than the target would clip
    /// the clear). The device's binding set comes back exactly as it was,
    /// alpha bits and extras included. Never runs when every bound target
    /// matches target 0, so the common multi-target shape costs no pass
    /// break here.
    fn clear_targets_outside_pass(&mut self, mut f: impl FnMut(&mut Self)) {
        let saved = self.pass_state.take_color_attachments();
        let prev_depth = self.pass_state.current_depth_texture();
        let prev_depth_size = self.pass_state.current_depth_size();
        let prev_depth_sampleable = self.pass_state.current_depth_is_sampleable();
        let prev_depth_has_stencil = self.pass_state.current_depth_has_stencil();
        let prev_depth_sample_count = self.pass_state.current_depth_sample_count();
        for slot in 1..4usize {
            if saved.extra_matches_rt0(slot) {
                continue;
            }
            let Some(target) = saved.slot(slot) else {
                continue;
            };
            self.pass_state.set_color_render_target_subresource(
                target.texture,
                target.logical_size.0,
                target.logical_size.1,
                target.format,
                target.scale,
                (target.subresource & 0xff, target.subresource >> 8),
            );
            self.pass_state.set_color_rt_has_alpha(target.has_alpha);
            self.pass_state.set_color_msaa(
                target.msaa_texture,
                target.msaa_srgb_texture,
                target.sample_count,
            );
            self.pass_state
                .set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
            f(self);
            self.end_current_pass("color_target_clear");
        }
        self.pass_state.set_depth_stencil_attachment(
            prev_depth,
            prev_depth_size,
            prev_depth_sampleable,
            prev_depth_has_stencil,
        );
        // The setter above reset the count, so it travels back with the
        // handle; without it a multisampled depth attachment would come back
        // declared single-sampled and be dropped at the next pass open.
        self.pass_state
            .set_depth_sample_count(prev_depth_sample_count);
        self.pass_state.restore_color_attachments(saved);
    }

    /// Current viewport `(x, y, w, h)` in pixels, with the `ensure_pass_open` fallback.
    ///
    /// Falls back to the bound RT size when the game never set a
    /// viewport. Read at draw time to derive the half-pixel `pos_fixup`
    /// uniform (VS slot 13).
    pub fn effective_viewport(&self) -> (u32, u32, u32, u32) {
        self.pass_state.effective_viewport()
    }

    /// Scale between the D3D9-reported space and the bound target's own.
    ///
    /// `render.scale` while the back buffer is bound, the identity otherwise.
    /// Read at draw time for the `pos_fixup` uniform, whose `.w` lane converts
    /// the point size from the logical pixels D3D9 states it in to the render
    /// pixels Metal rasterizes with.
    pub const fn target_scale(&self) -> RenderScale {
        self.pass_state.target_scale()
    }

    /// Mark a colour texture as read back this session.
    ///
    /// See `PassState::note_color_read_back`. The store-action optimiser
    /// then keeps its rendered content for a post-frame
    /// `GetRenderTargetData` blit.
    pub fn note_color_read_back(&mut self, handle: MetalHandle<MTLTextureKind>) {
        self.pass_state.note_color_read_back(handle);
    }

    /// Resolve `handle` now, because a blit is about to read it.
    ///
    /// See `PassState::note_msaa_read`. A no-op for a single-sampled target.
    pub fn note_msaa_read(&mut self, handle: MetalHandle<MTLTextureKind>) {
        self.pass_state.note_msaa_read(handle);
    }

    /// Bind mip `level` of `texture`, extent `size`, as the depth/stencil attachment.
    pub fn set_depth_stencil_attachment_level(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        level: u32,
        size: (u32, u32),
        is_sampleable: bool,
        has_stencil: bool,
    ) {
        self.pass_state.set_depth_stencil_attachment_level(
            texture,
            level,
            size,
            is_sampleable,
            has_stencil,
        );
    }

    /// Declare the bound depth attachment's sample count.
    ///
    /// Called in lockstep with `set_depth_stencil_attachment_level`, which
    /// resets it to 1.
    pub const fn set_depth_sample_count(&mut self, sample_count: u8) {
        self.pass_state.set_depth_sample_count(sample_count);
    }

    /// Apply a whole-target colour `Clear`.
    ///
    /// `srgb_write` is `D3DRS_SRGBWRITEENABLE` at the `Clear` call. It is
    /// resolved into the value here rather than on the API thread because
    /// only the pass state knows whether the attachment about to be bound
    /// is an sRGB view that converts the clear value itself.
    pub fn clear_color(&mut self, r: u32, g: u32, b: u32, a: u32, srgb_write: bool) {
        self.pass_state.set_srgb_write_enabled(srgb_write);
        let (r, g, b, a) = self.resolved_clear_rgba(r, g, b, a, srgb_write);
        let passes_before = self.pass_state.passes().len();
        match self.pass_state.clear_color(r, g, b, a) {
            ColorClearOutcome::Folded => {}
            ColorClearOutcome::EmitQuad {
                rgba,
                viewport,
                color_format,
            } => {
                self.reset_last_bound_if_pass_opened(passes_before);
                self.emit_clear_quad_color_inner(rgba, viewport, color_format);
            }
        }
        // A target bound outside the pass (sized unlike target 0) is owed the
        // clear too; neither the fold nor the quad above reached it.
        if self.pass_state.has_extra_color_targets_outside_pass() {
            self.clear_targets_outside_pass(|enc| enc.clear_color(r, g, b, a, srgb_write));
        }
    }

    /// The clear colour as the bound attachment needs it stored.
    ///
    /// A pass that binds sRGB views takes the linear value and lets Metal
    /// encode it, exactly as it encodes a draw's blended output. A target
    /// with no sRGB view is written raw, so the curve is applied here: the
    /// same encode the pixel-shader OETF variant performs for a draw.
    fn resolved_clear_rgba(
        &self,
        r: u32,
        g: u32,
        b: u32,
        a: u32,
        srgb_write: bool,
    ) -> (u32, u32, u32, u32) {
        if !srgb_write || self.pass_state.pass_srgb_write() {
            return (r, g, b, a);
        }
        let encoded = mtld3d_core::convert::linear_to_srgb_rgba([
            f32::from_bits(r),
            f32::from_bits(g),
            f32::from_bits(b),
            f32::from_bits(a),
        ]);
        (
            encoded[0].to_bits(),
            encoded[1].to_bits(),
            encoded[2].to_bits(),
            encoded[3].to_bits(),
        )
    }

    /// `Clear(pRects = NULL)` for colour: D3D9 bounds it to the current viewport ∩ RT.
    ///
    /// A viewport that covers the whole attachment folds to a fast
    /// full-attachment `loadAction = Clear`; a strict sub-region instead
    /// emits one scissored clear-quad over the viewport so pixels
    /// outside it keep their prior content.
    ///
    /// Every whole-target colour `Clear` comes here, combined with a depth or
    /// stencil plane or not. The bound is per plane and per attachment;
    /// [`Self::clear_depth_stencil_bounded_to_viewport`] answers the same
    /// question for the depth-stencil side.
    pub fn clear_color_bounded_to_viewport(
        &mut self,
        r: u32,
        g: u32,
        b: u32,
        a: u32,
        srgb_write: bool,
    ) {
        if self.pass_state.viewport_covers_color_attachment() {
            self.clear_color(r, g, b, a, srgb_write);
        } else {
            let (vpx, vpy, vpw, vph) = self.pass_state.effective_viewport();
            let rect = (
                vpx.cast_signed(),
                vpy.cast_signed(),
                vpx.saturating_add(vpw).cast_signed(),
                vpy.saturating_add(vph).cast_signed(),
            );
            // Derived from `effective_viewport`, so already in the bound
            // texture's space — goes to the resolved entry point, not the
            // converting one.
            self.clear_color_rects_resolved(r, g, b, a, srgb_write, &[rect]);
            // A target outside the pass bounds the clear to its own extent.
            if self.pass_state.has_extra_color_targets_outside_pass() {
                self.clear_targets_outside_pass(|enc| {
                    enc.clear_color_bounded_to_viewport(r, g, b, a, srgb_write);
                });
            }
        }
    }

    /// `Clear` with explicit `pRects`: clip each rect to the current viewport.
    ///
    /// Emit one scissored colour clear-quad per surviving region.
    /// Inverted / degenerate / fully-clipped-out rects are dropped
    /// silently. Routes through `PassState::begin_region_color_clear` so
    /// the render pass/encoder is open before any `drawPrimitives` —
    /// never a NULL encoder. `(r,g,b,a)` are f32 bits, as for
    /// `clear_color`.
    pub fn clear_color_rects(
        &mut self,
        r: u32,
        g: u32,
        b: u32,
        a: u32,
        srgb_write: bool,
        rects: &[(i32, i32, i32, i32)],
    ) {
        // `rects` are the game's own; the viewport they clip against is already
        // the bound texture's, so convert before clipping rather than after, or
        // the intersection is taken between two different spaces.
        let scale = self.pass_state.target_scale();
        if scale.is_identity() {
            self.clear_color_rects_resolved(r, g, b, a, srgb_write, rects);
        } else {
            let scaled: Vec<(i32, i32, i32, i32)> =
                rects.iter().map(|&rc| scale.rect_edges_i32(rc)).collect();
            self.clear_color_rects_resolved(r, g, b, a, srgb_write, &scaled);
        }
        // A target outside the pass clips the rects against its own viewport
        // and converts them at its own scale.
        if self.pass_state.has_extra_color_targets_outside_pass() {
            self.clear_targets_outside_pass(|enc| {
                enc.clear_color_rects(r, g, b, a, srgb_write, rects);
            });
        }
    }

    /// `clear_color_rects` for rects already in the bound texture's space.
    ///
    /// Split out so a caller that derived its rect from `effective_viewport`
    /// (itself already converted) cannot scale it a second time.
    fn clear_color_rects_resolved(
        &mut self,
        r: u32,
        g: u32,
        b: u32,
        a: u32,
        srgb_write: bool,
        rects: &[(i32, i32, i32, i32)],
    ) {
        self.pass_state.set_srgb_write_enabled(srgb_write);
        let (r, g, b, a) = self.resolved_clear_rgba(r, g, b, a, srgb_write);
        let vp = self.pass_state.effective_viewport();
        let regions: Vec<(u32, u32, u32, u32)> = rects
            .iter()
            .filter_map(|&rc| clip_rect_to_viewport(rc, vp))
            .collect();
        if regions.is_empty() {
            return;
        }
        let passes_before = self.pass_state.passes().len();
        let color_format = self.pass_state.begin_region_color_clear();
        self.reset_last_bound_if_pass_opened(passes_before);
        for region in regions {
            self.emit_clear_quad_color_inner((r, g, b, a), region, color_format);
        }
    }

    /// Depth and/or stencil `Clear` with explicit `pRects`: clip each rect to the viewport.
    ///
    /// The depth-stencil mirror of `clear_color_rects`: one scissored
    /// clear-quad per surviving region, through
    /// `PassState::begin_region_depth_stencil_clear` so the pass is open
    /// before any draw and pixels outside the rects keep their content.
    /// `rects` are the game's own; they are converted to the bound texture's
    /// space before clipping. `depth`/`stencil` carry the f32 bits / the
    /// masked stencil value of the planes being cleared.
    pub fn clear_depth_stencil_rects(
        &mut self,
        depth: Option<u32>,
        stencil: Option<u32>,
        rects: &[(i32, i32, i32, i32)],
    ) {
        let scale = self.pass_state.target_scale();
        if scale.is_identity() {
            self.clear_depth_stencil_rects_resolved(depth, stencil, rects);
        } else {
            let scaled: Vec<(i32, i32, i32, i32)> =
                rects.iter().map(|&rc| scale.rect_edges_i32(rc)).collect();
            self.clear_depth_stencil_rects_resolved(depth, stencil, &scaled);
        }
    }

    /// `clear_depth_stencil_rects` for rects already in the bound texture's space.
    ///
    /// The depth-stencil mirror of `clear_color_rects_resolved`, split out for
    /// the same reason: a caller that derived its rect from
    /// `effective_viewport` (itself already converted) must not scale it a
    /// second time.
    fn clear_depth_stencil_rects_resolved(
        &mut self,
        depth: Option<u32>,
        stencil: Option<u32>,
        rects: &[(i32, i32, i32, i32)],
    ) {
        self.bump_depth_write_epoch();
        let vp = self.pass_state.effective_viewport();
        let regions: Vec<(u32, u32, u32, u32)> = rects
            .iter()
            .filter_map(|&rc| clip_rect_to_viewport(rc, vp))
            .collect();
        if regions.is_empty() {
            return;
        }
        let passes_before = self.pass_state.passes().len();
        let Some((has_color, color_format)) = self.pass_state.begin_region_depth_stencil_clear()
        else {
            return;
        };
        self.reset_last_bound_if_pass_opened(passes_before);
        for region in regions {
            self.emit_clear_quad_depth_stencil_inner(
                depth,
                stencil,
                region,
                has_color,
                color_format,
            );
        }
    }

    /// `Clear(pRects = NULL)` for depth and/or stencil, bounded to the viewport ∩ DS.
    ///
    /// The depth-stencil mirror of [`Self::clear_color_bounded_to_viewport`]:
    /// a viewport covering the whole depth attachment folds to a fast
    /// full-attachment `loadAction = Clear`, and a strict sub-region emits one
    /// scissored clear-quad over the viewport so depth and stencil outside it
    /// keep their prior values. Coverage is asked of the depth attachment's
    /// own extent rather than the colour one: a depth-only pass has no colour
    /// attachment to measure against, and D3D9 permits a depth surface larger
    /// than render target 0.
    ///
    /// `depth`/`stencil` carry the f32 bits / the masked stencil value of the
    /// planes being cleared, as for `clear_depth_stencil_rects`. Both planes
    /// arrive in one call so a covering clear of both paints one quad rather
    /// than two: shadow-volume renderers clear depth and stencil together
    /// between lights.
    pub fn clear_depth_stencil_bounded_to_viewport(
        &mut self,
        depth: Option<u32>,
        stencil: Option<u32>,
    ) {
        if depth.is_none() && stencil.is_none() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "viewport-bounded depth-stencil clear with neither plane; skipped"
            );
            return;
        }
        if self.pass_state.viewport_covers_depth_attachment() {
            match (depth, stencil) {
                (Some(depth), Some(stencil)) => self.clear_depth_stencil(depth, stencil),
                (Some(depth), None) => self.clear_depth(depth),
                (None, Some(stencil)) => self.clear_stencil(stencil),
                // Rejected above. The arm exists for exhaustiveness, not
                // because it is reachable.
                (None, None) => {}
            }
            return;
        }
        let (vpx, vpy, vpw, vph) = self.pass_state.effective_viewport();
        let rect = (
            vpx.cast_signed(),
            vpy.cast_signed(),
            vpx.saturating_add(vpw).cast_signed(),
            vpy.saturating_add(vph).cast_signed(),
        );
        // Derived from `effective_viewport`, so already in the bound texture's
        // space: it goes to the resolved entry point, not the converting one.
        self.clear_depth_stencil_rects_resolved(depth, stencil, &[rect]);
    }

    pub fn clear_depth(&mut self, value: u32) {
        self.bump_depth_write_epoch();
        let passes_before = self.pass_state.passes().len();
        match self.pass_state.clear_depth(value) {
            DepthClearOutcome::Folded | DepthClearOutcome::NoOp => {}
            DepthClearOutcome::EmitQuad {
                value,
                viewport,
                has_color,
                color_format,
            } => {
                self.reset_last_bound_if_pass_opened(passes_before);
                self.emit_clear_quad_depth_stencil_inner(
                    Some(value),
                    None,
                    viewport,
                    has_color,
                    color_format,
                );
            }
        }
    }

    pub fn clear_stencil(&mut self, value: u32) {
        self.bump_depth_write_epoch();
        let passes_before = self.pass_state.passes().len();
        match self.pass_state.clear_stencil(value) {
            StencilClearOutcome::Folded | StencilClearOutcome::NoOp => {}
            StencilClearOutcome::EmitQuad {
                value,
                viewport,
                has_color,
                color_format,
            } => {
                self.reset_last_bound_if_pass_opened(passes_before);
                self.emit_clear_quad_depth_stencil_inner(
                    None,
                    Some(value),
                    viewport,
                    has_color,
                    color_format,
                );
            }
        }
    }

    /// `Clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL)`: both planes, one quad where a quad is due.
    ///
    /// Asks the depth chain and then the stencil chain. Neither call changes
    /// the state the other reads, except through a single `ensure_pass_open`,
    /// after which the stencil chain takes the branch that sees that same
    /// open pass. So the two answers pair up: both fold, or both paint the
    /// same rect, or both find nothing to clear; one draw writes both planes.
    /// Shadow-volume renderers clear both planes between lights, so the
    /// two-quad shape would double the clear draws on exactly that workload.
    /// The single-plane fallback below only guards the pairing; it is not
    /// expected to run.
    pub fn clear_depth_stencil(&mut self, depth: u32, stencil: u32) {
        self.bump_depth_write_epoch();
        let passes_before = self.pass_state.passes().len();
        let depth_outcome = self.pass_state.clear_depth(depth);
        let stencil_outcome = self.pass_state.clear_stencil(stencil);
        let depth_quad = match depth_outcome {
            DepthClearOutcome::Folded | DepthClearOutcome::NoOp => None,
            DepthClearOutcome::EmitQuad {
                value,
                viewport,
                has_color,
                color_format,
            } => Some((
                value,
                ClearQuadTarget {
                    viewport,
                    has_color,
                    color_format,
                },
            )),
        };
        let stencil_quad = match stencil_outcome {
            StencilClearOutcome::Folded | StencilClearOutcome::NoOp => None,
            StencilClearOutcome::EmitQuad {
                value,
                viewport,
                has_color,
                color_format,
            } => Some((
                value,
                ClearQuadTarget {
                    viewport,
                    has_color,
                    color_format,
                },
            )),
        };
        debug_assert!(
            stencil_quad.is_none() || depth_quad.is_some(),
            "depth folded while stencil painted"
        );
        debug_assert!(
            depth_quad.is_none() || !matches!(stencil_outcome, StencilClearOutcome::Folded),
            "depth painted while stencil folded"
        );
        if depth_quad.is_none() && stencil_quad.is_none() {
            return;
        }
        self.reset_last_bound_if_pass_opened(passes_before);
        match (depth_quad, stencil_quad) {
            (Some((depth, at)), Some((stencil, stencil_at))) if at.same_as(&stencil_at) => {
                self.emit_clear_quad_depth_stencil_inner(
                    Some(depth),
                    Some(stencil),
                    at.viewport,
                    at.has_color,
                    at.color_format,
                );
            }
            (depth_quad, stencil_quad) => {
                if let Some((depth, at)) = depth_quad {
                    self.emit_clear_quad_depth_stencil_inner(
                        Some(depth),
                        None,
                        at.viewport,
                        at.has_color,
                        at.color_format,
                    );
                }
                if let Some((stencil, at)) = stencil_quad {
                    self.emit_clear_quad_depth_stencil_inner(
                        None,
                        Some(stencil),
                        at.viewport,
                        at.has_color,
                        at.color_format,
                    );
                }
            }
        }
    }

    /// Flush `last_bound` when a `PassState` call opened a fresh Metal encoder.
    ///
    /// `PassState::clear_{color,depth,stencil}` and the visibility mode-sets
    /// open the new pass themselves (`ensure_pass_open`, with `loadAction =
    /// Load` to preserve prior tiles), but they can't reach the
    /// `FrameEncoder`-owned `last_bound`, so unlike
    /// `begin_render_pass_if_needed` the per-draw dedup would carry stale
    /// bindings across the encoder boundary. The new encoder starts with no
    /// bindings, so the next draw must re-emit everything, including the FF
    /// VS constants at buffer 15, whose content-based dedup otherwise
    /// suppresses the re-bind when the constants are unchanged from the
    /// prior pass (e.g. a sample pass after `SetDepthStencilSurface(NULL)`
    /// with the same viewport).
    ///
    /// A fresh pass is detected by the pass count growing, not by whether
    /// the pass was closed beforehand: a clear under a counting visibility
    /// query ends the open pass and opens a new one within one call.
    fn reset_last_bound_if_pass_opened(&mut self, passes_before: usize) {
        if self.pass_state.passes().len() != passes_before {
            self.reset_last_bound_for_fresh_encoder();
        }
    }

    /// Flush `last_bound` for an encoder known to have just opened.
    fn reset_last_bound_for_fresh_encoder(&mut self) {
        self.last_bound.reset();
        // Keep the debug-build emitted-command shadow in lockstep with the
        // cache so the in-sync assertion shares the same fresh-encoder
        // baseline (no bindings yet).
        #[cfg(debug_assertions)]
        self.pass_state.debug_reset_emitted();
    }

    /// Debug-build invariant on the per-draw dedup cache (`last_bound`).
    ///
    /// Assert it still matches what was actually emitted onto the
    /// encoder before a draw consumes it. Catches a cached-slot bind
    /// that bypassed its `_changed` gate (the clear-quad desync class).
    /// Compiled out of release builds.
    #[cfg(debug_assertions)]
    pub fn debug_assert_cache_in_sync(&self) {
        self.last_bound
            .debug_assert_in_sync(self.pass_state.debug_emitted());
    }

    /// Lazy create-or-fetch of the `(depth_format, color_format, flags)` clear-quad pipeline.
    ///
    /// Returns 0 if the unix-side pipeline creation fails (MSL compile
    /// error or Metal pipeline-create error). The clear-quad emit path
    /// guards on `handle != 0` and falls back to the legacy pass-break
    /// behaviour when 0 — rendering keeps working, but viewport-scoped
    /// mid-pass clears degrade to full-attachment clears for that frame,
    /// with a once-per-process warn.
    fn get_or_create_clear_quad_pipeline(&mut self, key: ClearQuadKey) -> u64 {
        if let Some(&handle) = self.clear_quad_pipeline_cache.get(&key) {
            return handle.raw();
        }
        let mut params = EnsureClearQuadPipelineParams {
            device_handle: self.device_handle,
            depth_format: key.depth_format,
            color_format: key.color_format,
            flags: key.flags,
            extra_present_mask: u32::from(key.extra.present_mask),
            extra_formats: key.extra.formats,
            sample_count: u32::from(key.sample_count),
            pipeline_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut params);
        let pipeline = params.pipeline_handle;
        if status != 0 || pipeline.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "clear-quad: EnsureClearQuadPipeline failed status={status:#x} → fallback to pass-break Clear (WoW tile-atlas shadows will regress)"
            );
            self.clear_quad_pipeline_cache
                .insert(key, MetalHandle::NULL);
            return 0;
        }
        self.clear_quad_pipeline_cache.insert(key, pipeline);
        if key.flags.contains(ClearQuadFlags::COLOR_FORMAT_NO_WRITE) {
            // This depth clear-quad declares the pass's color format (write
            // mask off) so it binds against a color-retaining pass. If Rule H
            // later strips that color attachment (cascade caster passes), the
            // SetPSO must be rewritten to a depth-only sibling — build it now
            // and map color→sibling in `no_color_pipeline_alt`. (The recursive
            // build self-maps the sibling, satisfying Rule H's resolvable check
            // for the unstripped case too.)
            let sibling_key = ClearQuadKey {
                flags: key.flags - ClearQuadFlags::COLOR_FORMAT_NO_WRITE,
                extra: mtld3d_core::pipeline_state::ExtraColorAttachments::NONE,
                ..key
            };
            let _ = self.get_or_create_clear_quad_pipeline(sibling_key);
            if let Some(&sibling) = self.clear_quad_pipeline_cache.get(&sibling_key)
                && !sibling.is_null()
            {
                self.no_color_pipeline_alt.insert(pipeline.raw(), sibling);
            }
        } else if !key.flags.contains(ClearQuadFlags::HAS_COLOR) {
            // Depth-only clear-quad pipelines are, by construction, no-color.
            // Self-mapping the handle in `no_color_pipeline_alt` lets Rule H's
            // resolvable check (passes.rs) succeed when a cascade caster pass
            // contains mid-pass depth clear-quads alongside zero-mask caster
            // draws — rewriting `SetRenderPipelineState` to the same handle is
            // a no-op, and the depth-only pipeline binds cleanly against a
            // depth-only render-pass descriptor. Color clear-quads (`HAS_COLOR`,
            // which writes color via the fragment function) must not be
            // self-mapped: their pipeline declares a color output and would fail
            // Metal's pipeline-vs-RP format validation against a stripped
            // (depth-only) descriptor.
            self.no_color_pipeline_alt.insert(pipeline.raw(), pipeline);
        }
        pipeline.raw()
    }

    /// Lazy create-or-fetch of the per-destination-format "blit-quad" pipeline.
    ///
    /// Used by the scaling `StretchRect` path. Returns 0 on a unix-side
    /// compile / pipeline-create failure; `stretch_blit_scaled` guards
    /// on `!= 0` and aborts the scale (the 1:1 path is unaffected).
    fn get_or_create_blit_pipeline(&mut self, color_format: PixelFormat, sample_count: u8) -> u64 {
        let key = (color_format, sample_count);
        if let Some(&handle) = self.blit_pipeline_cache.get(&key) {
            return handle.raw();
        }
        let mut params = EnsureBlitPipelineParams {
            device_handle: self.device_handle,
            color_format,
            quad_kind: QuadPipelineKind::StretchBlit,
            sample_count: u32::from(sample_count),
            pipeline_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut params);
        let pipeline = params.pipeline_handle;
        if status != 0 || pipeline.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "blit-quad: EnsureBlitPipeline failed status={status:#x} → scaling StretchRect dropped"
            );
            self.blit_pipeline_cache.insert(key, MetalHandle::NULL);
            return 0;
        }
        self.blit_pipeline_cache.insert(key, pipeline);
        pipeline.raw()
    }

    /// Lazy create-or-fetch of the per-destination-format "upload-quad" pipeline.
    ///
    /// Used by the GPU texture-upload pass. Returns 0 on a unix-side compile
    /// / pipeline-create failure; the caller then falls back to the blit
    /// upload where the source layout allows one.
    fn get_or_create_upload_pipeline(&mut self, color_format: PixelFormat) -> u64 {
        if let Some(&handle) = self.upload_pipeline_cache.get(&color_format) {
            return handle.raw();
        }
        let mut params = EnsureBlitPipelineParams {
            device_handle: self.device_handle,
            color_format,
            quad_kind: QuadPipelineKind::TextureUpload,
            // A D3D9 texture cannot be multisampled, so the upload pass is
            // always single-sampled.
            sample_count: 1,
            pipeline_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut params);
        let pipeline = params.pipeline_handle;
        if status != 0 || pipeline.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "upload-quad: EnsureBlitPipeline failed status={status:#x} → texture upload pass dropped"
            );
            self.upload_pipeline_cache
                .insert(color_format, MetalHandle::NULL);
            return 0;
        }
        self.upload_pipeline_cache.insert(color_format, pipeline);
        pipeline.raw()
    }

    /// Clamp-addressed sampler for the scaling-`StretchRect` blit.
    ///
    /// Built, or fetched from the sampler cache. The D3D9 `filter`
    /// selects POINT (`D3DTEXF_NONE` / `D3DTEXF_POINT`) or LINEAR
    /// (`D3DTEXF_LINEAR`) min/mag; mip is `NONE` (the source is always
    /// sampled at its bound mip level — there is no mip chain to
    /// traverse during a `StretchRect`); the address mode is CLAMP so a
    /// scale that samples exactly the rect edges never wraps in from the
    /// opposite side.
    fn get_or_create_blit_sampler(&mut self, filter: u32) -> u64 {
        // D3DTEXF_NONE on a StretchRect means "no filtering" → point sample.
        let min_mag = if filter == mtld3d_types::D3DTEXF_LINEAR {
            mtld3d_types::D3DTEXF_LINEAR
        } else {
            mtld3d_types::D3DTEXF_POINT
        };
        let mut ss = [0u32; mtld3d_types::SAMPLER_STATE_COUNT];
        ss[mtld3d_types::D3DSAMP_MINFILTER as usize] = min_mag;
        ss[mtld3d_types::D3DSAMP_MAGFILTER as usize] = min_mag;
        // The blit shader samples an explicit source level; a point mip filter
        // makes that level exact (without one the texture's level 0 is read
        // regardless of the explicit level).
        ss[mtld3d_types::D3DSAMP_MIPFILTER as usize] = mtld3d_types::D3DTEXF_POINT;
        ss[mtld3d_types::D3DSAMP_ADDRESSU as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        ss[mtld3d_types::D3DSAMP_ADDRESSV as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        ss[mtld3d_types::D3DSAMP_ADDRESSW as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        self.get_or_create_sampler(0, &ss, false, false)
    }

    /// A 2D view of one array slice of `handle`, retired with the frame that binds it.
    ///
    /// The scaling-`StretchRect` fragment function samples a `texture2d`, so a
    /// cube-map source is bound through a view of the single face it
    /// addresses. The view is a fresh Metal object: it goes on the retention
    /// queue at the current submit seq, so it outlives the replay of the pass
    /// that binds it and is destroyed once the GPU has retired that frame.
    /// Returns 0 when the unix side cannot create it, which drops the blit.
    fn slice_view_for_frame(&mut self, handle: u64, slice: u32) -> u64 {
        let mut params = CreateTextureSliceViewParams {
            // SAFETY: `handle` is a live Metal texture address resolved by the
            // caller from the texture cache, non-zero per its own guard.
            texture_handle: unsafe { MetalHandle::<MTLTextureKind>::new(handle) },
            view_handle: MetalHandle::NULL,
            slice,
            pad0: 0,
        };
        let status = unix_call(&mut params);
        let view = params.view_handle;
        if status != 0 || view.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "blit-quad: CreateTextureSliceView failed status={status:#x}, a scaling \
                 StretchRect out of a cube face is dropped"
            );
            return 0;
        }
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Texture,
                handle: view.raw(),
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });
        view.raw()
    }

    /// Scaling `StretchRect`: render the source texture onto a quad covering the destination rect.
    ///
    /// Metal's blit encoder can only do 1:1 copies, so a size-mismatch
    /// `StretchRect` is translated into a one-off render pass on the
    /// destination texture.
    ///
    /// `src_dims` / `dst_dims` are the source / destination mip-level pixel
    /// dimensions; `src_rect` / `dst_rect` are the (already-clamped) sub-rects.
    /// `dst_format` is the destination's Metal colour format (drives the
    /// pipeline cache + the pass colour attachment); `decode` is the
    /// source-side decode the fragment function applies (as-is, or one of the
    /// packed YUV formats, which it converts to RGB while sampling); `filter`
    /// is the D3D9 `D3DTEXF_*` value (POINT / LINEAR).
    ///
    /// The destination pass opens with `loadAction = Load` (or `DontCare` when
    /// the dst rect covers the whole attachment — both correct, the quad
    /// overwrites exactly the scissor rect) so content outside the dst rect is
    /// preserved. The prior render-target / depth / viewport binding is saved
    /// and restored around the pass, so a `StretchRect` mid-frame doesn't
    /// perturb the device's current RT. `note_color_read_back` marks the dst so a
    /// post-frame `GetRenderTargetData` keeps the rendered content (the store
    /// optimiser would otherwise discard a last-use non-backbuffer colour).
    pub fn stretch_blit_scaled(
        &mut self,
        src: &BlitSide,
        dst: &BlitSide,
        dst_format: PixelFormat,
        decode: mtld3d_core::stretch_rect::BlitDecode,
        filter: u32,
    ) {
        let &BlitSide {
            handle: src_handle,
            rect: src_rect,
            dims: src_dims,
            mip: src_mip,
            slice: src_slice,
            ..
        } = src;
        let &BlitSide {
            handle: dst_handle,
            rect: dst_rect,
            dims: dst_dims,
            mip: dst_mip,
            slice: dst_slice,
            ..
        } = dst;
        if dst_handle == 0 || src_handle == 0 {
            return;
        }
        // SAFETY: both handles are live Metal texture addresses resolved by
        // the caller (`get_or_create_texture` / a standalone colour handle),
        // non-zero per the guard above.
        let dst_tex = unsafe { MetalHandle::<MTLTextureKind>::new(dst_handle) };
        // `StretchRect` copies pixels verbatim, so no render state reaches it
        // and `D3DRS_SRGBWRITEENABLE` must not pick sRGB views for the
        // destination pass: encoding the copy would change the pixels it is
        // supposed to reproduce. One decision therefore drives both halves:
        // the pass attaches the destination's own format and the quad's
        // pipeline declares that same format. It is applied before the
        // destination is bound, so `ensure_pass_open` freezes the same
        // choice, and the next draw or `Clear` re-applies the game's state.
        self.pass_state.set_srgb_write_enabled(false);
        let pipeline = self.get_or_create_blit_pipeline(dst_format, dst.sample_count.max(1));
        if pipeline == 0 {
            return;
        }
        let sampler = self.get_or_create_blit_sampler(filter);
        if sampler == 0 {
            return;
        }

        // The fragment function declares `texture2d<float>`, so a cube-map
        // source reaches it through a 2D view of the face the call named;
        // binding the cube itself is a `texturecube` binding that reads face 0.
        let src_bind = match src_slice {
            Some(face) => {
                let view = self.slice_view_for_frame(src_handle, face);
                if view == 0 {
                    return;
                }
                view
            }
            None => src_handle,
        };

        // Source-rect → [0,1] texcoord transform, applied per-vertex in the
        // blit VS: `texcoord = q * scale + offset`, where `q` is the quad's
        // normalised coord in [0,1] (top-left origin). `scale` maps the unit
        // quad onto the source rect's *size* (normalised to the source
        // texture) and `offset` shifts it to the rect's *origin* — so q=(0,0)
        // samples the rect's top-left texel and q=(1,1) its bottom-right.
        // D3D9 surface dimensions and clamped sub-rect coords are ≤16384, so
        // the `u32 → u16 → f32` conversion is exact (well inside f32's 23-bit
        // mantissa). `saturating` on the (unreachable) >u16 case keeps the
        // conversion total without an `as`-cast precision-loss lint.
        let to_f = |v: u32| f32::from(u16::try_from(v).unwrap_or(u16::MAX));
        let (sw, sh) = (to_f(src_dims.0).max(1.0), to_f(src_dims.1).max(1.0));
        let scale_x = to_f(src_rect.w) / sw;
        let scale_y = to_f(src_rect.h) / sh;
        let offset_x = to_f(src_rect.x) / sw;
        let offset_y = to_f(src_rect.y) / sh;
        let mut xform = [0u8; 16];
        for (i, v) in [scale_x, scale_y, offset_x, offset_y].iter().enumerate() {
            xform[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let xform_ptr = self.scratch.alloc(&xform);
        // The source level, as the float the blit PS passes to `level()`; mip
        // counts are tiny, so the conversion is exact. `.y` carries the source
        // decode (0 = sample as-is, 1 = YUY2, 2 = UYVY) so one pipeline per
        // destination format serves every source format.
        let mut src_level = [0u8; 16];
        src_level[..4].copy_from_slice(&to_f(src_mip).to_le_bytes());
        src_level[4..8].copy_from_slice(&decode.uniform().to_le_bytes());
        let src_level_ptr = self.scratch.alloc(&src_level);

        // Save the device's current attachments + viewport so the one-off
        // destination pass doesn't perturb the live render target. The colour
        // set comes back verbatim, scale and extra targets included: the
        // blit's own binds run in already-converted coordinates and so declare
        // the identity, which would otherwise leak onto the device's target.
        let saved_color = self.pass_state.take_color_attachments();
        let prev_depth = self.pass_state.current_depth_texture();
        let prev_depth_size = self.pass_state.current_depth_size();
        let prev_depth_sampleable = self.pass_state.current_depth_is_sampleable();
        let prev_depth_has_stencil = self.pass_state.current_depth_has_stencil();
        let prev_depth_sample_count = self.pass_state.current_depth_sample_count();
        let prev_viewport = self.pass_state.viewport();
        let (prev_min_z, prev_max_z) = self.pass_state.viewport_depth_range();

        // Bind the destination as the colour RT with no depth attachment, then
        // open a Load pass scoped to the destination rect via the viewport.
        // `set_color_render_target` / `set_depth_stencil_attachment` end the
        // current pass for us, so the quad never draws on a stale encoder.
        // `dst_dims` and `dst_rect` are already in the destination texture's own
        // space (the caller converted them), so this binding declares the
        // identity rather than converting a second time.
        self.pass_state.set_color_render_target_subresource(
            dst_tex,
            dst_dims.0,
            dst_dims.1,
            dst_format,
            RenderScale::IDENTITY,
            (dst_slice.unwrap_or(0), dst_mip),
        );
        // A multisampled destination is written through its companion and
        // resolved into `dst_tex` at pass end, which is what every later read
        // of the surface looks at.
        self.pass_state
            .set_color_msaa(dst.msaa, dst.msaa_srgb, dst.sample_count);
        self.pass_state
            .set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
        self.pass_state
            .set_viewport(dst_rect.x, dst_rect.y, dst_rect.w, dst_rect.h, 0.0, 1.0);
        // A pipeline whose colour format differs from the bound attachment's
        // is undefined behaviour with the validation layer off, so pin the
        // two together where the binding is finished.
        debug_assert_eq!(
            self.pass_state.current_color_format(),
            dst_format,
            "blit-quad pipeline format must equal the pass's attachment format"
        );
        self.pass_state.ensure_pass_open();
        // The destination's content survives the readback that drives the
        // conformance check (and any real `GetRenderTargetData`).
        self.pass_state.note_color_read_back(dst_tex);
        // A fresh Metal encoder always opens here (the colour RT changed, which
        // ends any prior pass), so flush the per-draw dedup so every binding
        // below is actually emitted (the clear-quad cross-pass rule).
        self.reset_last_bound_for_fresh_encoder();

        let depth_state = self.get_or_create_depth_stencil(&DepthStencilSnapshot::inert());
        if self.last_bound.pipeline_changed(pipeline) {
            self.pass_state
                .emit_command(Command::set_render_pipeline_state(pipeline));
        }
        if self.last_bound.depth_stencil_changed(depth_state) {
            self.pass_state
                .emit_command(Command::set_depth_stencil_state(depth_state));
        }
        self.emit_scissor_rect_resolved((dst_rect.x, dst_rect.y, dst_rect.w, dst_rect.h));
        // Bind the source texture + sampler at fragment slot 0, and the
        // texcoord transform at vertex bytes slot 0.
        if self.last_bound.fragment_texture_changed(0, src_bind) {
            self.pass_state
                .emit_command(Command::set_fragment_texture(src_bind, 0));
        }
        if self.last_bound.fragment_sampler_changed(0, sampler) {
            self.pass_state
                .emit_command(Command::set_fragment_sampler_state(sampler, 0));
        }
        self.pass_state
            .emit_command(Command::set_vertex_bytes_at(xform_ptr, RGBA_BYTE_LEN, 0));
        self.pass_state.emit_command(Command::set_fragment_bytes_at(
            src_level_ptr,
            RGBA_BYTE_LEN,
            0,
        ));
        // Inline slot-0 vertex bind clobbers any real bound VB; drop the cache
        // so a subsequent bound draw re-emits its `setVertexBuffer`.
        self.last_bound.invalidate_vertex_buffer();
        self.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        self.end_current_pass("stretch_blit_scaled");

        // Restore the device's previous attachments + viewport.
        self.pass_state.restore_color_attachments(saved_color);
        self.pass_state.set_depth_stencil_attachment(
            prev_depth,
            prev_depth_size,
            prev_depth_sampleable,
            prev_depth_has_stencil,
        );
        // The setter above reset the count, so it travels back with the
        // handle; without it a multisampled depth attachment would come back
        // declared single-sampled and be dropped at the next pass open.
        self.pass_state
            .set_depth_sample_count(prev_depth_sample_count);
        let (pvx, pvy, pvw, pvh) = prev_viewport;
        self.pass_state
            .set_viewport(pvx, pvy, pvw, pvh, prev_min_z, prev_max_z);

        trace!(
            target: BLIT_TRACE_TARGET,
            "StretchRect SCALE src={src_handle:#x} {sw}x{sh} src_rect={sx},{sy}+{srw}x{srh} \
             dst={dst_handle:#x} {dw}x{dh} dst_rect={dx},{dy}+{drw}x{drh} filter={filter}",
            sw = src_dims.0, sh = src_dims.1,
            sx = src_rect.x, sy = src_rect.y, srw = src_rect.w, srh = src_rect.h,
            dw = dst_dims.0, dh = dst_dims.1,
            dx = dst_rect.x, dy = dst_rect.y, drw = dst_rect.w, drh = dst_rect.h,
        );
    }

    /// `ColorFill` a render target: paint the fill colour over `fill.rect`.
    ///
    /// The destination is bound as a one-off colour attachment with no depth
    /// and the ordinary clear machinery paints it, so a whole-surface fill
    /// folds into `loadAction = Clear` and a sub-rect becomes one clear-quad
    /// scissored to the rect. The device's own attachments and viewport are
    /// saved and restored around the pass, exactly as `stretch_blit_scaled`
    /// does, so a `ColorFill` mid-frame does not perturb the bound target.
    ///
    /// Being a pass rather than a blit also puts the fill in stream order:
    /// a fill issued after this frame's draws lands after them.
    ///
    /// `note_color_read_back` marks the destination so the store-action
    /// optimiser keeps the fill for whatever reads it later (a `StretchRect`
    /// source, a `GetRenderTargetData`, the next frame).
    pub fn color_fill_target(&mut self, fill: &ColorFillTarget) {
        let (rx, ry, rw, rh) = fill.rect;
        if fill.texture.is_null() || rw == 0 || rh == 0 {
            return;
        }
        // Save the device's current attachments + viewport so the one-off
        // destination pass doesn't perturb the live render target. The colour
        // set comes back verbatim, scale and extra targets included.
        let saved_color = self.pass_state.take_color_attachments();
        let prev_depth = self.pass_state.current_depth_texture();
        let prev_depth_size = self.pass_state.current_depth_size();
        let prev_depth_sampleable = self.pass_state.current_depth_is_sampleable();
        let prev_depth_has_stencil = self.pass_state.current_depth_has_stencil();
        let prev_viewport = self.pass_state.viewport();
        let (prev_min_z, prev_max_z) = self.pass_state.viewport_depth_range();

        // Bind the destination alone, then scope the fill with the viewport:
        // `clear_color_bounded_to_viewport` folds a viewport that covers the
        // attachment into the load action and scissors a quad to it otherwise.
        self.pass_state.set_color_render_target_subresource(
            fill.texture,
            fill.logical_size.0,
            fill.logical_size.1,
            fill.format,
            fill.scale,
            fill.subresource,
        );
        self.pass_state
            .set_depth_stencil_attachment(MetalHandle::NULL, (0, 0), false, false);
        self.pass_state.set_viewport(rx, ry, rw, rh, 0.0, 1.0);
        self.pass_state.note_color_read_back(fill.texture);
        // `ColorFill` writes the colour bytes verbatim, so the fill pass
        // attaches the base view and applies no encode whatever
        // `D3DRS_SRGBWRITEENABLE` the game left set. The next draw or `Clear`
        // re-applies the game's state to the pass it opens.
        self.pass_state.set_srgb_write_enabled(false);
        let (r, g, b, a) = fill.rgba;
        self.clear_color_bounded_to_viewport(r, g, b, a, false);
        // A folded fill is still only a pending clear; materialise it here so
        // it lands on this destination rather than on the restored one.
        self.pass_state.ensure_pass_open();
        self.end_current_pass("color_fill");
        // An autogen destination regenerates from the level the fill painted.
        // The blit rides the ordered stream right behind the fill's own pass,
        // so it reads the filled level 0 rather than leading the frame the way
        // the upload path's `run_generate_mipmaps` does.
        if fill.regenerate_mipmaps {
            self.push_stretch_rect_blit(BlitCommand::generate_mipmaps(fill.texture.raw()));
        }

        // Restore the device's previous attachments + viewport.
        self.pass_state.restore_color_attachments(saved_color);
        self.pass_state.set_depth_stencil_attachment(
            prev_depth,
            prev_depth_size,
            prev_depth_sampleable,
            prev_depth_has_stencil,
        );
        let (pvx, pvy, pvw, pvh) = prev_viewport;
        self.pass_state
            .set_viewport(pvx, pvy, pvw, pvh, prev_min_z, prev_max_z);

        trace!(
            target: BLIT_TRACE_TARGET,
            "ColorFill dst={dst:#x} {lw}x{lh} rect={rx},{ry}+{rw}x{rh} level={level}",
            dst = fill.texture.raw(),
            lw = fill.logical_size.0,
            lh = fill.logical_size.1,
            level = fill.subresource.1,
        );
    }

    /// Emit the clear-quad sequence for a mid-pass depth, stencil, or depth+stencil `Clear`.
    ///
    /// Sequence: pipeline → DSS → stencil reference (stencil clears only) →
    /// scissor → `SetVertexBytesAt(slot=0, &z)` → `DrawPrimitives (Triangle,
    /// 0, 3)`. Pipeline/DSS/reference/scissor are routed through
    /// `LastBoundCache` so back-to-back clear-quads and
    /// clear-quad-then-redraw both dedup (and the cache stays in sync with
    /// the encoder's actual bound state). The 3-vertex VS uses `vertex_id`
    /// to synthesise a fullscreen triangle covering `[-1, 1]^2` in clip
    /// space; the scissor constrains writes to the D3D9 viewport rect.
    ///
    /// Which planes the quad writes is decided by the depth-stencil state
    /// alone, so the same pipeline serves all three shapes. The constant `z`
    /// becomes the depth value where depth is requested. MSL cannot export a
    /// stencil value, so the stencil value rides the encoder as the stencil
    /// reference, which a `Replace` operation on every outcome writes to each
    /// covered fragment. `Clear(ZBUFFER | STENCIL)` therefore costs one draw.
    fn emit_clear_quad_depth_stencil_inner(
        &mut self,
        depth: Option<u32>,
        stencil: Option<u32>,
        viewport: (u32, u32, u32, u32),
        has_color: bool,
        color_format: PixelFormat,
    ) {
        let snapshot = match (depth.is_some(), stencil.is_some()) {
            (true, true) => DepthStencilSnapshot::depth_stencil_overwrite(),
            (true, false) => DepthStencilSnapshot::depth_overwrite(),
            (false, true) => DepthStencilSnapshot::stencil_overwrite(),
            (false, false) => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "clear quad requested with neither plane; skipped"
                );
                return;
            }
        };
        // Hardcoded for now: every depth attachment mtld3d emits is
        // `Depth32Float` (D24X8 / D24 / D32 / D16) or
        // `Depth32FloatStencil8` (D24S8). Shadow-cascade caster passes
        // land on the no-stencil variant. Future games hitting D24S8 mid-
        // pass Clear will need format plumbing from the depth-attach
        // site; this is a TODO with a graceful Metal-reject fallback
        // via the `handle == 0` check below.
        //
        // A depth-only clear writes no color, but Metal still validates the
        // pipeline's color format against the bound attachment. Two cases:
        //   - The live pass has NO color attachment (a Rule-H-stripped cascade
        //     caster pass, or a depth-only pass): use the no-color pipeline.
        //   - The live pass STILL has a color attachment (a smaller depth-stencil
        //     bound under a larger colour RT, or a caster pass before Rule H
        //     decides whether to strip): the pipeline
        //     must declare that color format with a zero write mask
        //     (`COLOR_FORMAT_NO_WRITE`), or Metal rejects the bind (and it is
        //     heap-corrupting UB with the layer off). `get_or_create_clear_quad_
        //     pipeline` also builds the depth-only sibling and maps
        //     color→sibling in `no_color_pipeline_alt`, so if Rule H later
        //     strips this pass's color the SetPSO is rewritten to the sibling.
        // Declare a stencil plane iff the bound depth attachment is a combined
        // depth+stencil texture (D24S8 etc. → `Depth32Float_Stencil8`); the
        // unix builder switches the depth format to the combined one when
        // `HAS_STENCIL` is set. Mismatching the pass's depth format is a Metal
        // validation failure / heap-corrupting UB.
        let mut flags = ClearQuadFlags::HAS_DEPTH;
        flags.set(
            ClearQuadFlags::HAS_STENCIL,
            self.pass_state.current_depth_has_stencil(),
        );
        flags.set(ClearQuadFlags::COLOR_FORMAT_NO_WRITE, has_color);
        let key = ClearQuadKey {
            depth_format: PixelFormat::Depth32Float,
            color_format: if has_color {
                color_format
            } else {
                PixelFormat::Bgra8Unorm
            },
            flags,
            sample_count: self.pass_state.current_color_sample_count(),
            extra: if has_color {
                self.clear_quad_extra_targets()
            } else {
                mtld3d_core::pipeline_state::ExtraColorAttachments::NONE
            },
        };
        let pipeline = self.get_or_create_clear_quad_pipeline(key);
        if pipeline == 0 {
            if let Some(value) = depth {
                self.pass_state.clear_depth_legacy_break(value);
            }
            if let Some(value) = stencil {
                self.pass_state.clear_stencil_legacy_break(value);
            }
            return;
        }
        let depth_state = self.get_or_create_depth_stencil(&snapshot);
        // `Clear`'s Z is a raw depth value: D3D9's `MinZ`/`MaxZ` scale a
        // transformed vertex's z, not a clear. The quad writes its value as
        // the vertex's clip-space z, so Metal's viewport depth transform would
        // remap it under a partitioned depth range (a sky / world / weapon
        // split, and D3D9 accepts an inverted `MinZ > MaxZ` too). Emit
        // the raw range for the draw and hand the game's own range back
        // straight after, so nothing downstream sees the bracket. Skipped
        // where the range is already raw, which is the overwhelmingly common
        // case, and where no depth plane is being written at all.
        let saved_range = self.pass_state.viewport_depth_range();
        let raw_range = (0.0f32, 1.0f32);
        let bracket_depth_range = depth.is_some()
            && (saved_range.0.to_bits(), saved_range.1.to_bits())
                != (raw_range.0.to_bits(), raw_range.1.to_bits());
        if bracket_depth_range {
            self.pass_state
                .set_emitted_depth_range(raw_range.0, raw_range.1);
        }
        // A stencil-only clear writes no depth, but the vertex stage still
        // consumes a constant z at slot 0; any value inside the clip range
        // will do.
        let z_bytes = depth.map_or(0.0f32, f32::from_bits).to_le_bytes();
        let z_ptr = self.scratch.alloc(&z_bytes);
        let (vx, vy, vw, vh) = viewport;
        if self.last_bound.pipeline_changed(pipeline) {
            self.pass_state
                .emit_command(Command::set_render_pipeline_state(pipeline));
        }
        if self.last_bound.depth_stencil_changed(depth_state) {
            self.pass_state
                .emit_command(Command::set_depth_stencil_state(depth_state));
        }
        if let Some(value) = stencil
            && self.last_bound.stencil_reference_changed(value)
        {
            self.pass_state
                .emit_command(Command::set_stencil_reference(value));
        }
        self.emit_scissor_rect_resolved((vx, vy, vw, vh));
        // The quad is one counter-clockwise triangle, back-facing under
        // Metal's default clockwise front face, so the cull mode the last
        // draw left behind (D3D's default CULL_CCW is cull-back) would drop
        // it whole. Go through the dedup cache so the next draw re-emits
        // its own mode.
        if self.last_bound.cull_mode_changed(CullMode::None) {
            self.pass_state
                .emit_command(Command::set_cull_mode(CullMode::None));
        }
        self.pass_state
            .emit_command(Command::set_vertex_bytes_at(z_ptr, F32_BYTE_LEN, 0));
        // Inline slot-0 bind clobbers the real Metal vertex-buffer binding;
        // drop the cached bound-VB so the next bound draw re-emits its
        // `setVertexBuffer` instead of reading this constant-z payload.
        self.last_bound.invalidate_vertex_buffer();
        // All clear-quad state is bound; assert the dedup cache matches the
        // encoder before the draw consumes it.
        #[cfg(debug_assertions)]
        self.debug_assert_cache_in_sync();
        self.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        if bracket_depth_range {
            self.pass_state
                .set_emitted_depth_range(saved_range.0, saved_range.1);
        }
    }

    /// Color-clear mirror of `emit_clear_quad_depth_inner`.
    ///
    /// Same shape; writes the constant RGBA via `setFragmentBytes` instead
    /// of a constant depth.
    fn emit_clear_quad_color_inner(
        &mut self,
        rgba: (u32, u32, u32, u32),
        viewport: (u32, u32, u32, u32),
        color_format: PixelFormat,
    ) {
        // Bracket the emitted commands in a color-clear-quad block so
        // Rule H can tell synthetic clear-quad writes apart from real
        // color-writing draws. When every other draw in the pass has
        // `COLORWRITEENABLE == 0`, Rule H strips the color attachment
        // AND drains this block — both are dead work once the
        // attachment is gone (the clear-quad pipeline declares a
        // color output and would otherwise fail Metal's pipeline-vs-RP
        // format validation against the depth-only descriptor).
        let block_start = self.pass_state.open_color_clear_quad_block();
        // A color clear-quad must declare a depth attachment ONLY when the live
        // pass has one. On a no-depth pass (an explicit
        // `SetDepthStencilSurface(NULL)`, or a depth surface the pass drops
        // for disagreeing with render target 0 on sample count) a pipeline
        // that declares depth is rejected by Metal ("depth attachment
        // pixelFormat must be Invalid, as no texture is set"), so gate
        // `HAS_DEPTH` on the attachment the pass actually takes.
        let mut flags = ClearQuadFlags::HAS_COLOR;
        let has_depth = self.pass_binds_depth();
        flags.set(ClearQuadFlags::HAS_DEPTH, has_depth);
        // Match the bound depth attachment's stencil-ness (see the depth
        // clear-quad above) so the pipeline's depth/stencil formats agree with
        // the pass — only meaningful when a depth attachment is present.
        flags.set(
            ClearQuadFlags::HAS_STENCIL,
            has_depth && self.pass_state.current_depth_has_stencil(),
        );
        let key = ClearQuadKey {
            depth_format: PixelFormat::Depth32Float,
            color_format,
            flags,
            extra: self.clear_quad_extra_targets(),
            sample_count: self.pass_state.current_color_sample_count(),
        };
        let pipeline = self.get_or_create_clear_quad_pipeline(key);
        if pipeline == 0 {
            // Open/close pair must be balanced even on the legacy
            // fallback path so a future `emit_clear_quad_color_inner`
            // doesn't see a stale start offset on the same pass.
            self.pass_state.close_color_clear_quad_block(block_start);
            self.pass_state
                .clear_color_legacy_break(rgba.0, rgba.1, rgba.2, rgba.3);
            return;
        }
        // Color clear doesn't write depth: bind a no-write depth-stencil
        // state so a transient color clear over an in-use depth
        // attachment doesn't perturb depth values.
        let depth_state = self.get_or_create_depth_stencil(&DepthStencilSnapshot::inert());
        // Color: write rgba as float4 via setFragmentBytes. The caller
        // (`device_clear` → `clear_color`/`clear_color_rects`) passes each
        // channel as f32 BITS, exactly like the folded load-action clear
        // (unix `command.rs` reads `f32::from_bits(pass.clear_*)` for the
        // MTLClearColor), so decode the same way — NOT as a D3DCOLOR byte.
        // Stable backing via scratch.
        let component = f32::from_bits;
        let rgba_f = [
            component(rgba.0),
            component(rgba.1),
            component(rgba.2),
            component(rgba.3),
        ];
        let mut rgba_bytes = [0u8; 16];
        for (i, v) in rgba_f.iter().enumerate() {
            rgba_bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        // Depth: zero so it doesn't write to depth (mask=0 + always works,
        // but Metal needs *some* z, so 0.0 is harmless).
        let z_bytes = 0f32.to_le_bytes();
        let z_ptr = self.scratch.alloc(&z_bytes);
        let rgba_ptr = self.scratch.alloc(&rgba_bytes);
        let (vx, vy, vw, vh) = viewport;
        if self.last_bound.pipeline_changed(pipeline) {
            self.pass_state
                .emit_command(Command::set_render_pipeline_state(pipeline));
        }
        if self.last_bound.depth_stencil_changed(depth_state) {
            self.pass_state
                .emit_command(Command::set_depth_stencil_state(depth_state));
        }
        self.emit_scissor_rect_resolved((vx, vy, vw, vh));
        // The quad is one counter-clockwise triangle, back-facing under
        // Metal's default clockwise front face, so the cull mode the last
        // draw left behind (D3D's default CULL_CCW is cull-back) would drop
        // it whole. Go through the dedup cache so the next draw re-emits
        // its own mode.
        if self.last_bound.cull_mode_changed(CullMode::None) {
            self.pass_state
                .emit_command(Command::set_cull_mode(CullMode::None));
        }
        self.pass_state
            .emit_command(Command::set_vertex_bytes_at(z_ptr, F32_BYTE_LEN, 0));
        // Inline slot-0 bind clobbers the real Metal vertex-buffer binding;
        // drop the cached bound-VB so the next bound draw re-emits its
        // `setVertexBuffer` instead of reading this constant-z payload.
        self.last_bound.invalidate_vertex_buffer();
        self.pass_state
            .emit_command(Command::set_fragment_bytes_at(rgba_ptr, RGBA_BYTE_LEN, 0));
        // All clear-quad state is bound; assert the dedup cache matches the
        // encoder before the draw consumes it.
        #[cfg(debug_assertions)]
        self.debug_assert_cache_in_sync();
        self.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        self.pass_state.close_color_clear_quad_block(block_start);
    }

    /// Copy data into the scratch arena and return a pointer to it.
    ///
    /// The pointer is valid for the lifetime of this frame's encoding.
    pub fn alloc_scratch(&mut self, data: &[u8]) -> u64 {
        self.scratch.alloc(data)
    }

    /// Apply an `Op::SetVsConstRange` delta to the encoder-side VS mirror.
    ///
    /// Reads `rows × 16` bytes from `data` (a scratch-allocated slice from
    /// the previous-frame arena's API-thread tail), copies them into
    /// `vs_constants_mirror[start_row..]`, advances the populated-rows
    /// watermark, and invalidates the per-pass scratch cache so the next
    /// dirty draw re-bumps.
    fn apply_vs_const_range(&mut self, start_row: u16, rows: u16, data: ScratchSlice) {
        apply_const_range_into(
            self.vs_constants_mirror.as_mut(),
            start_row,
            rows,
            data,
            "vs_const_range",
        );
        let watermark = start_row.saturating_add(rows).min(CONSTANT_ROWS_U16);
        if watermark > self.vs_constants_populated_rows {
            self.vs_constants_populated_rows = watermark;
        }
        self.vs_const_scratch_cache = None;
    }

    fn apply_ps_const_range(&mut self, start_row: u16, rows: u16, data: ScratchSlice) {
        apply_const_range_into(
            self.ps_constants_mirror.as_mut(),
            start_row,
            rows,
            data,
            "ps_const_range",
        );
        let watermark = start_row.saturating_add(rows).min(CONSTANT_ROWS_U16);
        if watermark > self.ps_constants_populated_rows {
            self.ps_constants_populated_rows = watermark;
        }
        self.ps_const_scratch_cache = None;
    }

    /// Snapshot `rows` rows from the VS constant mirror into the per-frame scratch arena.
    ///
    /// Returns the previously-cached slice instead if the mirror hasn't
    /// changed and `rows` matches. Returned `ScratchSlice` is what gets
    /// passed to `Command::set_vertex_bytes_at` from `emit_draw`.
    pub fn vs_const_scratch(&mut self, rows: u16) -> ScratchSlice {
        if rows == 0 {
            return ScratchSlice::EMPTY;
        }
        if let Some((slice, cached_rows)) = self.vs_const_scratch_cache
            && cached_rows == rows
        {
            return slice;
        }
        let byte_len = usize::from(rows) * core::mem::size_of::<[f32; 4]>();
        // SAFETY: `[f32; 4]` is POD; the borrow is `&[u8]` of `rows * 16`
        // bytes which lies fully within `vs_constants_mirror`.
        let bytes = unsafe {
            core::slice::from_raw_parts(self.vs_constants_mirror.as_ptr().cast::<u8>(), byte_len)
        };
        let slice = draw::arena_alloc_bytes(&mut self.scratch, bytes);
        self.vs_const_scratch_cache = Some((slice, rows));
        slice
    }

    pub fn ps_const_scratch(&mut self, rows: u16) -> ScratchSlice {
        if rows == 0 {
            return ScratchSlice::EMPTY;
        }
        if let Some((slice, cached_rows)) = self.ps_const_scratch_cache
            && cached_rows == rows
        {
            return slice;
        }
        let byte_len = usize::from(rows) * core::mem::size_of::<[f32; 4]>();
        // SAFETY: see [`Self::vs_const_scratch`].
        let bytes = unsafe {
            core::slice::from_raw_parts(self.ps_constants_mirror.as_ptr().cast::<u8>(), byte_len)
        };
        let slice = draw::arena_alloc_bytes(&mut self.scratch, bytes);
        self.ps_const_scratch_cache = Some((slice, rows));
        slice
    }

    /// Apply an `Op::SetFfVsConstRange` delta to the FF VS mirror.
    ///
    /// Parallel to `apply_vs_const_range` but routes to the FF mirror.
    /// **Always** invalidates `ff_vs_const_scratch_cache` so the next
    /// draw bumps a fresh slice — preserves the per-draw isolation
    /// invariant Metal's submit-time setVertexBytes copy depends on.
    fn apply_ff_vs_const_range(&mut self, start_row: u16, rows: u16, data: ScratchSlice) {
        apply_const_range_into(
            self.ff_vs_constants_mirror.as_mut(),
            start_row,
            rows,
            data,
            "ff_vs_const_range",
        );
        self.ff_vs_const_scratch_cache = None;
    }

    /// Snapshot `rows` rows from the FF VS constant mirror into the per-frame scratch arena.
    ///
    /// Cached across consecutive draws within one "mirror epoch" — every
    /// `apply_ff_vs_const_range` invalidates the cache so the next draw
    /// gets fresh bytes. **Never** returns a pointer into the mirror
    /// itself; always bumps to scratch.
    pub fn ff_vs_const_scratch(&mut self, rows: u16) -> ScratchSlice {
        if rows == 0 {
            return ScratchSlice::EMPTY;
        }
        if let Some((slice, cached_rows)) = self.ff_vs_const_scratch_cache
            && cached_rows == rows
        {
            return slice;
        }
        let byte_len = usize::from(rows) * core::mem::size_of::<[f32; 4]>();
        // SAFETY: see [`Self::vs_const_scratch`]. `[f32; 4]` is POD and
        // the byte_len lies fully within `ff_vs_constants_mirror`.
        let bytes = unsafe {
            core::slice::from_raw_parts(self.ff_vs_constants_mirror.as_ptr().cast::<u8>(), byte_len)
        };
        let slice = draw::arena_alloc_bytes(&mut self.scratch, bytes);
        self.ff_vs_const_scratch_cache = Some((slice, rows));
        slice
    }

    /// Populated-row high-watermark of the encoder-side VS mirror.
    ///
    /// The maximum `start_row + rows` seen across every
    /// `Op::SetVsConstRange` applied. `emit_draw` uses this for shaders
    /// that bind constants via relative addressing (`c[a0.x + N]`), where
    /// the static-analysis bound from `max_const_used` would truncate. PS
    /// has no equivalent because D3D9 PS doesn't support relative-addressed
    /// constants in any profile we ship.
    pub const fn vs_constants_populated_rows(&self) -> u16 {
        self.vs_constants_populated_rows
    }

    /// Ensure a pass is live for the next draw.
    ///
    /// Retained as the draw-site entry point for `emit_draw`; delegates
    /// into `PassState`. When a new pass actually opens, flushes
    /// `last_bound` so the per-draw dedup in `emit_draw` re-emits the full
    /// state on the first draw of the new Metal render encoder.
    pub fn begin_render_pass_if_needed(&mut self) {
        let passes_before = self.pass_state.passes().len();
        self.pass_state.ensure_pass_open();
        self.reset_last_bound_if_pass_opened(passes_before);
    }

    /// Record that the draw being emitted read `[offset, offset + size)` from VB/IB `id`.
    ///
    /// Size 0 = to end of buffer. Feeds rename-at-overlap (and the
    /// `reorder` perf counter). Call in op order (after the bind) so a
    /// later overlapping staging upload sees it.
    pub fn note_buffer_draw_range(&mut self, id: u64, offset: u32, size: u32, logical_len: u32) {
        self.pass_state
            .note_draw_range(id, offset, size, logical_len);
    }

    /// Mutable access to the per-pass last-bound state cache.
    ///
    /// Used by `emit_draw` to skip redundant `set*` commands when the value
    /// hasn't changed since the previous draw in the current pass.
    pub const fn last_bound(&mut self) -> &mut LastBoundCache {
        &mut self.last_bound
    }

    /// Raw pointer to the `OpSub` slot of the encoder's per-frame perf accumulator.
    ///
    /// For the per-draw phase timers in `emit_draw`
    /// (`CycleAddTimer::start(enc.op_sub_cycles_ptr(sub))`). The timer holds
    /// only this pointer — no borrow of `self` — so the measured region
    /// reborrows `self` freely and `Drop` folds the cycles in at scope end
    /// (including on draw-drop `return` paths). Returns null when perf
    /// tracking is off, which makes the timer a no-op.
    pub const fn op_sub_cycles_ptr(&mut self, sub: OpSub) -> *mut u64 {
        self.perf.op_sub_cycles_ptr(sub)
    }

    /// Raw pointer to an [`OpSubDetail`] slot.
    ///
    /// The second-level child timers nested inside the `resolve`/`binds`
    /// parent timers in `emit_draw`. Same no-borrow / null-when-off
    /// contract as `op_sub_cycles_ptr`.
    pub const fn op_sub_detail_ptr(&mut self, detail: OpSubDetail) -> *mut u64 {
        self.perf.op_sub_detail_ptr(detail)
    }

    pub fn set_viewport(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        min_z: f32,
        max_z: f32,
    ) {
        self.pass_state
            .set_viewport(x, y, width, height, min_z, max_z);
    }

    pub fn emit_scissor(&mut self, test_enable: bool, rect: [u32; 4]) {
        let resolved = self.pass_state.resolved_scissor_rect(test_enable, rect);
        self.emit_scissor_rect_resolved(resolved);
    }

    fn emit_scissor_rect_resolved(&mut self, rect: (u32, u32, u32, u32)) {
        if self.last_bound.scissor_rect_changed(rect) {
            let (x, y, w, h) = rect;
            self.pass_state
                .emit_command(Command::set_scissor_rect(x, y, w, h));
        }
    }

    // ── D3D9→Metal translation + caching (runs on encoder thread) ──

    /// Look up or create an `MTLDepthStencilState` for the given D3D9 state.
    pub fn get_or_create_depth_stencil(&mut self, snapshot: &DepthStencilSnapshot) -> u64 {
        let key = key_from_snapshot(snapshot);
        if let Some(&handle) = self.depth_stencil_cache.get(&key) {
            return handle.raw();
        }

        let mut params = params_from_snapshot(snapshot, key, self.device_handle);
        let status = unix_call(&mut params);
        let state = params.state_handle;
        if status != 0 || state.is_null() {
            error!(target: LOG_TARGET, "encoder: CreateDepthStencilState failed");
            return 0;
        }
        self.depth_stencil_cache.insert(key, state);
        state.raw()
    }

    /// Install a parsed `DxsoProgram` under its content-hash id.
    ///
    /// Called from a closure pushed by `CreateVertexShader` /
    /// `CreatePixelShader`, so programs arrive on the encoder thread before
    /// the first draw that could reference them. Idempotent — a second
    /// register for the same id (identical bytecode re-create) is a no-op.
    pub fn register_program(&mut self, shader_id: ProgramId, program: DxsoProgram) {
        // Precompute the declared sampler slots so the draw path never scans the
        // program. A PS with no samplers, and every VS, stores the empty default.
        self.prog_sampler_decls
            .entry(shader_id)
            .or_insert_with(|| PsSamplerDecls::from_program(&program));
        self.program_cache
            .entry(shader_id)
            .or_insert_with(|| Box::new(program));
    }

    /// The declared sampler slots + types for a programmable pixel shader.
    ///
    /// Empty for an unregistered id (the draw path then binds no fallback).
    /// Also serves vertex shaders: `register_program` collects
    /// `Declaration::Sampler` entries for every program, so a `vs_3_0`
    /// using vertex texture fetch reports its slots here too.
    pub fn ps_declared_samplers(&self, ps_id: ProgramId) -> PsSamplerDecls {
        self.prog_sampler_decls
            .get(&ps_id)
            .copied()
            .unwrap_or_default()
    }

    /// Update one mirrored vertex texture slot (`SetTexture` on 257..=260).
    pub const fn set_vertex_texture_binding(
        &mut self,
        slot: usize,
        id: Option<mtld3d_core::ids::TextureId>,
    ) {
        self.vertex_tex_bindings[slot].texture_id = id;
    }

    /// Update one mirrored vertex sampler state (`SetSamplerState` on 257..=260).
    pub const fn set_vertex_sampler_binding(
        &mut self,
        slot: usize,
        state: [u32; SAMPLER_STATE_COUNT],
    ) {
        self.vertex_tex_bindings[slot].sampler_state = state;
    }

    /// One mirrored vertex slot: `(texture id, sampler state)` by value.
    #[must_use]
    pub const fn vertex_binding(
        &self,
        slot: usize,
    ) -> (
        Option<mtld3d_core::ids::TextureId>,
        [u32; SAMPLER_STATE_COUNT],
    ) {
        (
            self.vertex_tex_bindings[slot].texture_id,
            self.vertex_tex_bindings[slot].sampler_state,
        )
    }

    /// Absorb the pre-warm thread's compiled MSL → `MTLLibrary` handles into `lib_cache`.
    ///
    /// Each entry serves subsequent live miss lookups keyed by the same
    /// `disk_key`. Called once from `encoder_thread_main` after the
    /// dedicated prewarm channel resolves, *before* any `EncoderMessage` is
    /// processed; the call also flips `cache_ready`, allowing subsequent
    /// miss-compiles to append records to `mtld3d_shaders.bin` — unless
    /// `writes_disabled` is set, in which case `cache_disabled` latches so
    /// the rest of the session skips the open/append entirely.
    pub fn ingest_warm_cache(
        &mut self,
        entries: Vec<(u64, StageLibHandles)>,
        writes_disabled: bool,
    ) {
        for (key, handles) in entries {
            self.lib_cache.insert(key, handles);
        }
        self.flags.insert(FrameEncoderFlags::CACHE_READY);
        if writes_disabled {
            self.flags.insert(FrameEncoderFlags::CACHE_DISABLED);
        }
    }

    /// Total distinct shader-cache entries known to this encoder.
    ///
    /// Used by `maybe_emit_compile_summary` for the burst log's
    /// `… N total)` field — the source of truth is the cache itself,
    /// no separate counter.
    fn shader_cache_total(&self) -> u32 {
        u32::try_from(self.lib_cache.len()).unwrap_or(u32::MAX)
    }

    /// Emit the live `shaders: N compiled in Tms (...)` line once a burst has gone idle.
    ///
    /// Polled once per frame from `run_frame`. Debounce uses TSC cycles
    /// (calibrated via `tsc_hz()` in `core/src/tsc.rs`) so the per-frame
    /// poll cost stays in the few-cycle range — no `Instant::now()`
    /// syscall.
    pub fn maybe_emit_compile_summary(&mut self) {
        let counts = shader_compile_stats::current_counts();
        let idle = secs_to_cycles(1);
        if !self.compile_burst.poll(counts, rdtsc(), idle) {
            return;
        }
        let snap = shader_compile_stats::drain();
        let total = self.shader_cache_total();
        log::info!(
            target: LOG_TARGET,
            "{}",
            shader_compile_stats::format_summary(&snap, "compiled", total),
        );
    }

    /// Append one freshly-compiled MSL record to `mtld3d_shaders.bin`.
    ///
    /// Best-effort: any I/O failure latches `cache_disabled` so the rest of
    /// the session stops trying.
    fn cache_write_record(&mut self, kind: CachedKind, key: u64, msl: &str) {
        if self.flags.contains(FrameEncoderFlags::CACHE_DISABLED)
            || !self.flags.contains(FrameEncoderFlags::CACHE_READY)
        {
            return;
        }
        if self.cache_writer.is_none() {
            match open_or_create_cache_file() {
                Ok(file) => self.cache_writer = Some(file),
                Err(e) => {
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "shader_cache: open mtld3d_shaders.bin failed → cache disabled: {e}"
                    );
                    self.flags.insert(FrameEncoderFlags::CACHE_DISABLED);
                    return;
                }
            }
        }
        // Upper bound: chunk header + uncompressed MSL. The actual
        // frame written is zstd-compressed and therefore smaller, but
        // this avoids any reallocation on the hot path.
        let mut buf = Vec::with_capacity(shader_cache::CHUNK_HEADER_LEN + msl.len());
        shader_cache::write_record(
            &mut buf,
            &shader_cache::CacheEntry {
                kind,
                key,
                msl: msl.to_owned(),
            },
        );
        if let Some(file) = self.cache_writer.as_mut()
            && let Err(e) = file.write_all(&buf)
        {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "shader_cache: write mtld3d_shaders.bin failed → cache disabled: {e}"
            );
            self.flags.insert(FrameEncoderFlags::CACHE_DISABLED);
            self.cache_writer = None;
        }
    }

    /// Resolve the VS library for a draw.
    ///
    /// Hot path: borrow-probe the source-keyed index (`ff_vs_libs` /
    /// `prog_vs_libs`) — `FxHash` + exact `Eq`, no per-draw content hash,
    /// no clone. VS variants share one `MTLLibrary`, so the index key
    /// excludes `variant`. On a miss (≈ once per shader) the cold path
    /// computes the `disk_key`. Returns `None` if no program was registered
    /// or emit/compile fails.
    pub fn resolve_vs_library(&mut self, source: &VsSource) -> Option<StageLibHandles> {
        match source {
            VsSource::FixedFunction { key, .. } => {
                if let Some(&handles) = self.ff_vs_libs.get(key) {
                    return Some(handles);
                }
            }
            VsSource::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => {
                if let Some(&handles) = self.prog_vs_libs.get(&(
                    *vs_id,
                    *provided_input_mask,
                    *clip_plane_count,
                    *sampler_kinds,
                )) {
                    return Some(handles);
                }
            }
        }
        let handles = self.resolve_vs_library_cold(source)?;
        match source {
            VsSource::FixedFunction { key, .. } => {
                self.ff_vs_libs.insert(key.clone(), handles);
            }
            VsSource::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => {
                self.prog_vs_libs.insert(
                    (
                        *vs_id,
                        *provided_input_mask,
                        *clip_plane_count,
                        *sampler_kinds,
                    ),
                    handles,
                );
            }
        }
        Some(handles)
    }

    /// Cold path of [`resolve_vs_library`] — index miss.
    ///
    /// Computes the Xxh3 `disk_key` (the only content hash, ~once per
    /// shader), bridges the warm-loaded disk-keyed `lib_cache`, else
    /// emits + compiles + writes the on-disk cache. The `disk_key` is the
    /// on-disk content identity; every `VsKey` variant of a shader maps
    /// to it.
    fn resolve_vs_library_cold(&mut self, source: &VsSource) -> Option<StageLibHandles> {
        let disk_key = source.disk_key();
        if let Some(&handles) = self.lib_cache.get(&disk_key) {
            return Some(handles);
        }
        let kind = match source {
            VsSource::Programmable { vs_id, .. } => {
                let major = self.program_cache.get(vs_id).map_or(0, |p| p.major);
                CachedKind::from_programmable(major, false)
            }
            VsSource::FixedFunction { .. } => Some(CachedKind::FfVs),
        };
        let entry_name = vs_entry_name(source, &self.program_cache, disk_key);
        let started = Instant::now();
        let (msl, bucket) = match source {
            VsSource::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => {
                let Some(program) = self.program_cache.get(vs_id) else {
                    error!(target: LOG_TARGET, "VS {vs_id:#x} missing from program_cache");
                    return None;
                };
                let bucket = CompileBucket::from_sm_major(program.major);
                let msl = match emit_vs_programmable_named(
                    program,
                    &entry_name,
                    *provided_input_mask,
                    *clip_plane_count,
                    *sampler_kinds,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        error!(target: LOG_TARGET, "emit_vs_programmable failed: {e:?}");
                        return None;
                    }
                };
                (msl, bucket)
            }
            VsSource::FixedFunction { key, .. } => {
                mtld3d_shared::crumb!(
                    "ffvs:emit",
                    self.current_submit_seq,
                    u64::from(key.tex_coord_count),
                );
                (emit_vs_ff_named(key, &entry_name), Some(CompileBucket::Ff))
            }
        };
        if log_enabled!(target: MSL_TRACE_TARGET, Level::Trace) {
            let tag = shader_source_tag_vs(source);
            trace!(target: MSL_TRACE_TARGET, "── VS MSL {tag} ──\n{msl}\n── /VS MSL {tag} ──");
        }
        let handles =
            compile_stage_library(self.device_handle, StageTag::Vertex, &msl, &entry_name)?;
        if let Some(b) = bucket {
            shader_compile_stats::record(b, started.elapsed());
        }
        if let Some(kind) = kind {
            self.cache_write_record(kind, disk_key, &msl);
        }
        self.lib_cache.insert(disk_key, handles);
        Some(handles)
    }

    /// Resolve the PS library for a draw.
    ///
    /// Hot path: borrow-probe the source-keyed index. PS MSL depends on
    /// `variant`, so the key folds it in — `ff_ps_libs` nests
    /// `FfPsKey → variant → handles` (borrow the `FfPsKey`, no clone),
    /// `prog_ps_libs` uses a `(ProgramId, VariantKey)` `Copy` tuple. On a
    /// miss the cold path computes the `disk_key`.
    pub fn resolve_ps_library(
        &mut self,
        source: &PsSource,
        variant: VariantKey,
    ) -> Option<StageLibHandles> {
        match source {
            PsSource::FixedFunction { key, .. } => {
                if let Some(&handles) = self.ff_ps_libs.get(key).and_then(|m| m.get(&variant)) {
                    return Some(handles);
                }
            }
            PsSource::Programmable { ps_id, .. } => {
                if let Some(&handles) = self.prog_ps_libs.get(&(*ps_id, variant)) {
                    return Some(handles);
                }
            }
        }
        let handles = self.resolve_ps_library_cold(source, variant)?;
        match source {
            PsSource::FixedFunction { key, .. } => {
                self.ff_ps_libs
                    .entry(key.clone())
                    .or_default()
                    .insert(variant, handles);
            }
            PsSource::Programmable { ps_id, .. } => {
                self.prog_ps_libs.insert((*ps_id, variant), handles);
            }
        }
        Some(handles)
    }

    /// Cold path of [`resolve_ps_library`] — index miss.
    ///
    /// Mirror of `resolve_vs_library_cold`; the `disk_key` folds in
    /// `variant`.
    fn resolve_ps_library_cold(
        &mut self,
        source: &PsSource,
        variant: VariantKey,
    ) -> Option<StageLibHandles> {
        let disk_key = source.disk_key(variant);
        if let Some(&handles) = self.lib_cache.get(&disk_key) {
            return Some(handles);
        }
        let kind = match source {
            PsSource::Programmable { ps_id, .. } => {
                let major = self.program_cache.get(ps_id).map_or(0, |p| p.major);
                CachedKind::from_programmable(major, true)
            }
            PsSource::FixedFunction { .. } => Some(CachedKind::FfPs),
        };
        let entry_name = ps_entry_name(source, &self.program_cache, disk_key);
        let started = Instant::now();
        let (msl, bucket) = match source {
            PsSource::Programmable { ps_id, .. } => {
                let Some(program) = self.program_cache.get(ps_id) else {
                    error!(target: LOG_TARGET, "PS {ps_id:#x} missing from program_cache");
                    return None;
                };
                let bucket = CompileBucket::from_sm_major(program.major);
                let msl = match emit_ps_programmable_named(program, variant, &entry_name) {
                    Ok(s) => s,
                    Err(e) => {
                        error!(target: LOG_TARGET, "emit_ps_programmable failed: {e:?}");
                        return None;
                    }
                };
                (msl, bucket)
            }
            PsSource::FixedFunction { key, .. } => (
                emit_ps_ff_named(key, variant, &entry_name),
                Some(CompileBucket::Ff),
            ),
        };
        if log_enabled!(target: MSL_TRACE_TARGET, Level::Trace) {
            let tag = shader_source_tag_ps(source, variant);
            trace!(target: MSL_TRACE_TARGET, "── PS MSL {tag} ──\n{msl}\n── /PS MSL {tag} ──");
        }
        let handles =
            compile_stage_library(self.device_handle, StageTag::Fragment, &msl, &entry_name)?;
        if let Some(b) = bucket {
            shader_compile_stats::record(b, started.elapsed());
        }
        if let Some(kind) = kind {
            self.cache_write_record(kind, disk_key, &msl);
        }
        self.lib_cache.insert(disk_key, handles);
        Some(handles)
    }

    /// One-shot `debug!` per unique `(rt_handle, vs_key, ps_key)` seen by `emit_draw`.
    ///
    /// Dedup is keyed on the Metal texture handle, not on size, so distinct
    /// render targets that share dimensions stay distinguishable; size is
    /// included in the message for grep convenience. Logs under
    /// `mtld3d::d3d9` (this is a shader-debug aid, not perf telemetry). For
    /// programmable PS the trailing `ps_cs=` carries the raw bytecode
    /// content hash so the printed line greps directly against
    /// `debug.bytecodeDumpDir`'s `ps_<hash>.dxso` filename — distinct from
    /// `ps_tag`'s variant-folded library hash. VS variants share one
    /// `MTLLibrary`, so `vs_tag`'s hash already matches the bytecode
    /// filename and no `vs_cs=` is needed.
    pub fn maybe_log_pass_shader(
        &mut self,
        shaders: ShaderRef,
        stage_bindings: &crate::draw::StageBindingsPtr,
    ) {
        if !log_enabled!(target: LOG_TARGET, Level::Debug) {
            return;
        }
        // Build the keys here (after the gate) so the hot path pays nothing.
        let vs_key = shaders.vs.key(shaders.variant);
        let ps_key = shaders.ps.key(shaders.variant);
        let rt_handle = self.pass_state.current_color_texture();
        let vs_pid = vs_key.pair_id();
        let ps_pid = ps_key.pair_id();
        if self
            .pass_shader_log_fired
            .insert((rt_handle, vs_pid, ps_pid))
        {
            let (w, h) = self.pass_state.current_color_size();
            let vs_tag = vs_pid.tag();
            let ps_tag = ps_pid.tag();
            let ps_cs = match &ps_key {
                PsKey::Programmable { ps_id, .. } => format!("  ps_cs={:#x}", ps_id.raw()),
                PsKey::FixedFunction { .. } => String::new(),
            };
            // Bound tex_ids per stage: a PS hash read off a GPU capture greps
            // straight to the bound texture identities, which are the same ids
            // carried on the Metal object labels. No second capture is needed
            // to correlate the two.
            let bound: String = stage_bindings
                .iter()
                .map(|(stage, sb)| format!("s{stage}={:#x}", sb.texture_id.raw()))
                .collect::<Vec<_>>()
                .join(" ");
            debug!(
                target: LOG_TARGET,
                "pass RT {rt_handle:#x} {w}x{h} uses VS {vs_tag}  PS {ps_tag}{ps_cs}  bound=[{bound}]"
            );
        }
    }

    /// Per-draw breadcrumb used to pinpoint a misbehaving draw.
    ///
    /// Matched against captured `.dxso` shaders and Metal texture handles.
    /// Disabled unless `RUST_LOG=mtld3d::d3d9::draw=trace`. Floods on
    /// purpose — scope this target only when investigating a specific bug.
    pub fn maybe_emit_draw_trace(
        &self,
        shaders: ShaderRef,
        metal_prim: PrimitiveType,
        vertex_source: &VertexSource,
        index_source: &IndexSource,
        stride: u32,
    ) {
        if !log_enabled!(target: DRAW_TRACE_TARGET, Level::Trace) {
            return;
        }
        let vs_key = shaders.vs.key(shaders.variant);
        let ps_key = shaders.ps.key(shaders.variant);
        let rt = self.pass_state.current_color_texture();
        let (w, h) = self.pass_state.current_color_size();
        let (vp_x, vp_y, vp_w, vp_h) = self.pass_state.viewport();
        let vs_tag = vs_key.pair_id().tag();
        let ps_tag = ps_key.pair_id().tag();
        let ps_cs = match &ps_key {
            PsKey::Programmable { ps_id, .. } => format!(" ps_cs={:#x}", ps_id.raw()),
            PsKey::FixedFunction { .. } => String::new(),
        };
        let vb = match vertex_source {
            VertexSource::Up { size, .. } => format!("vb=UP({size})"),
            VertexSource::Bound { first, extra, .. } => format!(
                "vb={:#x}+{} streams={}",
                first.buffer_id,
                first.offset,
                extra.len() + 1
            ),
        };
        let idx = match index_source {
            IndexSource::None {
                start_vertex,
                vertex_count,
            } => format!("verts={vertex_count}@{start_vertex}"),
            IndexSource::Bound {
                buffer_id,
                offset,
                index_count,
                base_vertex,
                ..
            } => format!("ib={buffer_id:#x}+{offset} idx={index_count} basevtx={base_vertex}"),
            IndexSource::Up {
                index_count,
                index_type,
                ..
            } => format!("ib=UP idx={index_count} {index_type:?}"),
            IndexSource::Fan {
                start_vertex,
                primitive_count,
            } => format!("ib=fan-pattern tris={primitive_count} verts@{start_vertex}"),
            IndexSource::Generated {
                index_count,
                index_type,
                min_vertex,
                max_vertex,
                ..
            } => {
                format!("ib=fan idx={index_count} {index_type:?} verts={min_vertex}..={max_vertex}")
            }
        };
        trace!(
            target: DRAW_TRACE_TARGET,
            "draw rt={rt:#x} {w}x{h} prim={metal_prim:?} \
             vp={vp_x},{vp_y}+{vp_w}x{vp_h} \
             VS {vs_tag} PS {ps_tag}{ps_cs} \
             {vb} stride={stride} {idx}"
        );
    }

    /// Per-draw shader-pair telemetry.
    ///
    /// Gated on `pair_stats_enabled()` (`mtld3d::d3d9::passes=trace` off —
    /// the common case) so the cold path skips even the map insert. The
    /// `PairShaderId`s — including their `disk_key` content hash — are built
    /// *after* the gate from the sources, so the hot path pays nothing (this
    /// is no longer on the per-draw cache lookup path).
    /// Count a triangle-fan draw that took the generated-index slow path.
    pub const fn bump_fan_generated(&mut self) {
        self.perf.bump_fan_generated();
    }

    pub fn bump_pair_stats(
        &mut self,
        shaders: ShaderRef,
        verts: u32,
        alpha_func: u8,
        cull_mode: u32,
    ) {
        if !mtld3d_core::perf::pair_stats_enabled() {
            return;
        }
        let vs_pid = shaders.vs.key(shaders.variant).pair_id();
        let ps_pid = shaders.ps.key(shaders.variant).pair_id();
        let (w, h) = self.pass_state.current_color_size();
        self.perf.bump_pair_stats(PairStatsSample {
            rt_w: w,
            rt_h: h,
            vs: vs_pid,
            ps: ps_pid,
            verts,
            alpha_func,
            cull_mode,
        });
    }

    /// Look up or create an `MTLRenderPipelineState` for the given pipeline state snapshot.
    ///
    /// Translation from D3D9 state to Metal enums happens in
    /// `mtld3d_core::pipeline_state` — the per-field invariant test there
    /// guards against "classified Consumed but value silently dropped".
    pub fn get_or_create_pipeline(
        &mut self,
        snapshot: &PipelineSnapshot,
        vertex_attrs: &[VertexAttrDesc],
    ) -> u64 {
        self.perf.bump_pipeline_memo_call();
        // L0 memo: a draw whose pipeline snapshot is identical to the
        // previous one returns the cached handle without rebuilding the
        // `PipelineKey` (its D3D→Metal translations) or probing
        // `pipeline_cache`. It also skips the no-color twin's second resolve
        // below: a hit means the populating miss already ran it, and
        // `no_color_pipeline_alt` is process-lifetime, so the side-map entry
        // is still present. Only successful resolves are memoised, so a
        // failing snapshot still flows through the unchanged path (and keeps
        // its existing per-draw error/retry behaviour). The `match` copies
        // the handle out so the memo borrow ends before the `&mut perf` bump.
        let memo_hit = match &self.last_pipeline_memo {
            Some((prev, handle)) if *prev == *snapshot => Some(*handle),
            _ => None,
        };
        if let Some(handle) = memo_hit {
            self.perf.bump_pipeline_memo_hit();
            return handle;
        }
        let with_color = self.resolve_pipeline(snapshot, vertex_attrs);
        // Dual-build for zero-mask draws: build the matching no-color
        // variant up-front so pass-finalisation (Rule H) can swap to it
        // retroactively if every draw in the pass had `mask == 0`.
        // Building both is cheap — cache hit on the second call after
        // the first frame; CreateRenderPipeline thunk on cold-miss.
        if !with_color.is_null() && snapshot.writes_no_color() && snapshot.has_color_output() {
            // No-color twin: same identity except the attach flag (and no
            // render targets 1..3, which Rule H strips together with target
            // 0). Explicit `.clone()` because PipelineSnapshot is no longer
            // Copy; fires once per unique pipeline (cache hit thereafter).
            let mut alt = snapshot.clone();
            alt.attach
                .remove(mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_COLOR_OUTPUT);
            alt.extra = mtld3d_core::pipeline_state::ExtraColorAttachments::NONE;
            let no_color = self.resolve_pipeline(&alt, vertex_attrs);
            if !no_color.is_null() {
                self.no_color_pipeline_alt
                    .insert(with_color.raw(), no_color);
            }
        }
        if !with_color.is_null() {
            self.last_pipeline_memo = Some((snapshot.clone(), with_color.raw()));
        }
        with_color.raw()
    }

    fn resolve_pipeline(
        &mut self,
        snapshot: &PipelineSnapshot,
        vertex_attrs: &[VertexAttrDesc],
    ) -> MetalHandle<MTLRenderPipelineStateKind> {
        let key = pipeline_state::key_from_snapshot(snapshot);
        if let Some(&handle) = self.pipeline_cache.get(&key) {
            return handle;
        }
        // One wire layout per used stream; lives on this frame until the
        // synchronous thunk below has read it.
        let vertex_layouts = pipeline_state::vertex_layouts_from_snapshot(snapshot);
        let mut params = pipeline_state::params_from_snapshot(&PipelineBuildInputs {
            snapshot,
            vertex_attrs,
            vertex_layouts: &vertex_layouts,
            device_handle: self.device_handle,
        });
        let status = unix_call(&mut params);
        let pipeline = params.pipeline_handle;
        if status != 0 || pipeline.is_null() {
            error!(target: LOG_TARGET, "encoder: CreateRenderPipeline failed");
            return MetalHandle::NULL;
        }
        self.pipeline_cache.insert(key, pipeline);
        pipeline
    }

    /// Look up the Metal texture handle for a previously-warmed-up `TextureId`.
    ///
    /// Returns 0 on cache miss (with a `log_once_warn`).
    ///
    /// Per-draw bind path. Relies on the invariant that every texture
    /// that can be set as a stage binding has had `push_texture_warmup`
    /// called on the API thread before the draw, so the cache entry
    /// exists by the time `run_frame` drains warmups (which it does
    /// before processing any ops). Maintained by `device_create_texture`,
    /// `device_create_shadow_texture`, and `texture::rehydrate_for_device`.
    pub fn get_texture_handle_by_id(&self, texture_id: mtld3d_core::ids::TextureId) -> u64 {
        if let Some(state) = self.texture_cache.get(&texture_id) {
            return state.mtl_texture.raw();
        }
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: texture_id.raw(),
            "encoder: texture {:#x} bound but missing from cache — warmup ordering bug",
            texture_id.raw()
        );
        0
    }

    /// `get_texture_handle_by_id` for a stage sampling with `D3DSAMP_SRGBTEXTURE=1`.
    ///
    /// Returns the eager sRGB twin view so the hardware decodes
    /// sRGB→linear at sample time. A texture whose format has no sRGB
    /// encoding falls back to the base handle and is sampled linear —
    /// the same silent no-op real D3D9 hardware performs — with a
    /// once-per-texture info line so the fallback is observable.
    pub fn get_texture_handle_by_id_srgb(&self, texture_id: mtld3d_core::ids::TextureId) -> u64 {
        if let Some(state) = self.texture_cache.get(&texture_id) {
            if !state.mtl_texture_srgb.is_null() {
                return state.mtl_texture_srgb.raw();
            }
            mtld3d_shared::log_once_info_by!(
                target: LOG_TARGET,
                key: texture_id.raw(),
                "encoder: texture {:#x} sampled with D3DSAMP_SRGBTEXTURE=1 but its format has \
                 no sRGB twin — sampled linear (matches hardware D3D9)",
                texture_id.raw()
            );
            return state.mtl_texture.raw();
        }
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: texture_id.raw(),
            "encoder: texture {:#x} bound but missing from cache — warmup ordering bug",
            texture_id.raw()
        );
        0
    }

    /// Look up or create an `MTLTexture` for the given texture ID (deferred creation).
    ///
    /// Cache hit returns immediately; cache miss goes through a one-element
    /// batched `CreateTexturesBatch` thunk — same wire path used by
    /// `drain_texture_warmups` when the API thread queued the texture at
    /// `CreateTexture` time.
    pub fn get_or_create_texture(&mut self, info: &TextureInfo) -> u64 {
        let texture_id = info.texture_id;
        let staging_slots = Self::texture_staging_slot_count(info);
        if let Some(state) = self.texture_cache.get(&texture_id) {
            return state.mtl_texture.raw();
        }

        let desc = self.texture_desc_from_info(info);
        let mut handle = MetalHandle::<MTLTextureKind>::NULL;
        let mut srgb_handle = MetalHandle::<MTLTextureKind>::NULL;
        let status = self.batch_create_textures(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
            core::slice::from_mut(&mut srgb_handle),
        );
        if status != 0 || handle.is_null() {
            error!(target: LOG_TARGET, "encoder: CreateTexture failed");
            return 0;
        }
        self.texture_cache.insert(
            texture_id,
            TextureGpuState {
                mtl_texture: handle,
                mtl_texture_srgb: srgb_handle,
                mip_staging_buffers: vec![MipStagingBuffer::default(); staging_slots],
            },
        );
        self.pass_state.register_srgb_twin(srgb_handle, handle);
        handle.raw()
    }

    /// Wrap the bound VB or IB `PageBox` in a Shared `MTLBuffer` lazily on first Draw post-rename.
    ///
    /// Subsequent Draws within the same-backing window hit the cache. The
    /// rename itself is handled via `intake_vbib_retention` at the start of
    /// the subsequent frame. A `Staged` entry left as a failed-warmup
    /// placeholder gets its device buffer recreated here before the draw
    /// binds it.
    pub fn ensure_vbib_mtl_buffer(
        &mut self,
        buffer_id: BufferId,
        backing_ptr: u64,
        backing_len: u64,
        backing_generation: u64,
    ) -> u64 {
        let current_seq = self.current_submit_seq;
        if let Some(state) = self.buffer_cache.get_mut(&buffer_id) {
            if state.is_staged {
                // Draws bind the persistent `Private` device buffer; the
                // `backing_ptr`/`backing_len` args describe the CPU staging
                // and are irrelevant here. No notify — the device buffer's
                // contents arrive via the staging-upload blit, not CPU
                // writes. Track the draw seq so release-retention gates the
                // device buffer's destroy past this frame's GPU read.
                if current_seq > state.last_submit_seq {
                    state.last_submit_seq = current_seq;
                }
                let device = state.device_buffer;
                let length = state.length;
                if !device.is_null() {
                    return device.raw();
                }
                // Failed-warmup placeholder: recreate the device buffer so
                // the entry heals and later uploads take the fast path. Its
                // contents are undefined until the next upload (any upload
                // dropped while Metal kept failing is gone), matching what
                // D3D9 promises for a buffer the game never wrote. On
                // repeat failure return 0; the draw sites drop the draw
                // and log it.
                let Some(fresh) = self.alloc_fresh_device_buffer(buffer_id, length) else {
                    return 0;
                };
                if let Some(s) = self.buffer_cache.get_mut(&buffer_id) {
                    s.device_buffer = fresh;
                }
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: buffer_id.raw(),
                    "ensure_vbib_mtl_buffer: recreated device buffer for buffer_id {:#x} \
                     at draw time, contents undefined until the next upload",
                    buffer_id.raw()
                );
                return fresh.raw();
            }
            if state.backing_ptr == backing_ptr
                && state.length == backing_len
                && state.backing_generation == backing_generation
            {
                let mtl_buffer = state.mtl_buffer;
                if current_seq > state.last_submit_seq {
                    // First bind of this buffer this frame — assume the
                    // CPU may have written via Lock/Unlock since the
                    // previous frame's notify, so notify the full range
                    // before the GPU reads. NOOVERWRITE Lock keeps the
                    // backing stable (cache hit) but still mutates bytes,
                    // so cache-hit alone isn't enough to skip the notify.
                    state.last_submit_seq = current_seq;
                    self.enqueue_notify_buffer_did_modify_range(mtl_buffer.raw(), 0, backing_len);
                }
                return mtl_buffer.raw();
            }
            // Backing changed mid-frame for the same `BufferId` — the
            // expected pattern is `Draw; Lock(DISCARD|default); Draw`
            // inside a single frame, where Draw1's closure snapshotted
            // the old backing and Draw2's closure the new one. Defer
            // the stale wrapper's destroy via the retention queue
            // gated on the current submit seq — destroying
            // synchronously would free an MTLBuffer that earlier
            // closures in this frame still reference in their
            // `SetVertexBuffer` / `SetFragmentBuffer` commands, which
            // the unix-side `encode_pass` replays at submit time.
            let stale = self.buffer_cache.remove(&buffer_id).expect("just checked");
            if !stale.mtl_buffer.is_null() {
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Buffer,
                        handle: stale.mtl_buffer.raw(),
                        page_box: None,
                        staging_arc: None,
                        seq: current_seq,
                        from_texture: false,
                    });
            }
        }
        let desc = BufferCreateDesc {
            backing_ptr,
            length: backing_len,
            id: buffer_id.raw(),
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::VbIb,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "ensure_vbib_mtl_buffer: CreateBuffer failed \
                 (id={buffer_id:#x}, backing={backing_ptr:#x}, len={backing_len}, status={status:#x})",
            );
            return 0;
        }
        self.buffer_cache.insert(
            buffer_id,
            BufferGpuState {
                mtl_buffer: handle,
                device_buffer: MetalHandle::NULL,
                is_staged: false,
                backing_ptr,
                length: backing_len,
                backing_generation,
                last_submit_seq: current_seq,
            },
        );
        // Fresh wrapper around new (or renamed) backing — notify the
        // GPU about every byte the CPU may have written since the
        // backing was allocated. No-op on UMA via the helper's gate.
        self.enqueue_notify_buffer_did_modify_range(handle.raw(), 0, backing_len);
        handle.raw()
    }

    /// Copy a `Staged` VB/IB's device buffer into caller-owned PE memory.
    ///
    /// The device buffer is `StorageModePrivate` at an address Metal chose,
    /// which the 32-bit PE cannot dereference, so the only route back to the
    /// CPU is a GPU copy into a `Shared` wrapper over PE pages. The
    /// destination is `Shared` on every device rather than following the
    /// storage policy: a `Managed` one holds the GPU's write in VRAM until a
    /// synchronize, and this copy exists to be read on the CPU.
    ///
    /// The caller owns `dst_ptr`, keeps it alive past this frame's submit,
    /// and must wait for GPU completion of that submit before reading it.
    /// `false`, with a log line, when the buffer has no device buffer to
    /// read: the caller then has no indices and drops the draw.
    pub fn readback_device_buffer(
        &mut self,
        buffer_id: BufferId,
        dst_ptr: u64,
        dst_len: u64,
    ) -> bool {
        let Some((src, length)) = self
            .buffer_cache
            .get(&buffer_id)
            .filter(|s| s.is_staged && !s.device_buffer.is_null())
            .map(|s| (s.device_buffer.raw(), s.length))
        else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "readback_device_buffer: no Staged device buffer behind buffer_id {:#x}, nothing to read",
                buffer_id.raw());
            return false;
        };
        let desc = BufferCreateDesc {
            backing_ptr: dst_ptr,
            length: dst_len,
            id: buffer_id.raw(),
            storage_mode: StorageMode::Shared,
            kind: BufferKind::VbIb,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "readback_device_buffer: CreateBuffer failed \
                 (id={buffer_id:#x}, len={dst_len}, status={status:#x})",
            );
            return false;
        }
        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_buffer(
                &CopyBufferToBufferInfo {
                    src_buffer: src,
                    dst_buffer: handle.raw(),
                    src_offset: 0,
                    dst_offset: 0,
                    byte_size: length.min(dst_len),
                },
            ));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        // The wrapper is this frame's alone. Retention gates its destroy on
        // the submit that carries the copy, the same gate every other
        // mid-frame wrapper rides.
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Buffer,
                handle: handle.raw(),
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: false,
            });
        true
    }

    /// Append a `NotifyBufferDidModifyRange` to `frame_blit_commands`.
    ///
    /// The unix dispatcher will call `[buffer didModifyRange:]` before the
    /// next GPU read. Short-circuits on UMA — Apple Silicon uses `Shared`
    /// storage where the GPU sees CPU writes coherently, no notify needed.
    /// Crucially this does **not** flip `frame_blit_commands_need_encoder`,
    /// so a frame whose only blit activity is notifies skips
    /// `MTLBlitCommandEncoder` creation on the unix side.
    fn enqueue_notify_buffer_did_modify_range(
        &mut self,
        mtl_buffer: u64,
        offset: u64,
        length: u64,
    ) {
        if self.gpu_caps.unified_memory || mtl_buffer == 0 || length == 0 {
            return;
        }
        self.frame_blit_commands
            .push(BlitCommand::notify_buffer_did_modify_range(
                mtl_buffer, offset, length,
            ));
    }

    /// Drain every API-thread VB/IB retention entry for this frame.
    ///
    /// Entries move into the encoder's `pending_resource_retention`. Called
    /// *after* the op loop in `run_frame`, not at `begin_frame` — by then,
    /// any same-frame draw closure that still referenced the old backing
    /// has run and populated the cache with its own wrapper (via
    /// `ensure_vb`'s hit path), and the subsequent switch to the new
    /// backing has already queued the stale wrapper via the mid-frame
    /// rename path. Running intake here means the cache entry we match on
    /// is the one that's genuinely retired, not one that's about to be
    /// re-created in the same frame.
    fn intake_vbib_retentions(&mut self, frame: &mut FrameData) {
        for entry in core::mem::take(&mut frame.vbib_retentions) {
            self.intake_vbib_retention(entry);
        }
    }

    /// Mirror a `bump_vbib_retained_add` into the device-shared atomic.
    ///
    /// The API thread's retention cap then sees live bytes. No-op before
    /// the first frame seeds `retained_bytes_ptr`.
    fn add_retained_bytes(&self, bytes: usize) {
        if self.retained_bytes_ptr != 0 {
            // SAFETY: `retained_bytes_ptr` is a PE-heap `Arc<AtomicU64>`
            // raw pointer from `FrameData`, valid for the device's
            // lifetime (mirrors `coherent_seq_ptr`).
            unsafe { SharedCounter::new(self.retained_bytes_ptr) }
                .fetch_add(bytes as u64, Ordering::AcqRel);
        }
    }

    /// Mirror a `bump_vbib_retained_sub` into the device-shared atomic.
    fn sub_retained_bytes(&self, bytes: usize) {
        if self.retained_bytes_ptr != 0 {
            // SAFETY: see `add_retained_bytes`.
            unsafe { SharedCounter::new(self.retained_bytes_ptr) }
                .fetch_sub(bytes as u64, Ordering::AcqRel);
        }
    }

    /// Intake one API-thread VB/IB retention entry.
    ///
    /// Pairs its `PageBox` with the cache's `MTLBuffer` (if any) and queues
    /// the pair for seq-gated destruction. The cache entry is removed when
    /// its `backing_ptr` matches the retained box — that's the path that
    /// destroys the `MTLBuffer` wrapper. When the cache already holds a
    /// newer backing (mid-frame rename happened inside `ensure_vb`), the
    /// wrapper was already queued there, so only the `PageBox` is attached
    /// here.
    fn intake_vbib_retention(&mut self, entry: PendingVbibRetention) {
        let PendingVbibRetention {
            buffer_id,
            page_box,
            last_submit_seq,
        } = entry;
        let backing_ptr = page_box.as_ptr() as u64;
        let (mtl_buffer, seq) = match self.buffer_cache.get(&buffer_id) {
            // `Staged`: the retained `page_box` is the CPU staging (no GPU
            // wrapper); the thing to destroy is the persistent `Private`
            // device buffer. A `Staged` buffer only ever queues retention
            // on release, so removing the entry here is correct.
            Some(state) if state.is_staged => {
                let removed = self.buffer_cache.remove(&buffer_id).expect("just checked");
                (
                    removed.device_buffer,
                    removed.last_submit_seq.max(last_submit_seq),
                )
            }
            Some(state)
                if state.backing_ptr == backing_ptr
                    && state.backing_generation == page_box.generation() =>
            {
                let removed = self.buffer_cache.remove(&buffer_id).expect("just checked");
                (
                    removed.mtl_buffer,
                    removed.last_submit_seq.max(last_submit_seq),
                )
            }
            // The address matches but the allocation does not: the entry
            // wraps a later backing at the retained one's address, and
            // taking it would destroy a wrapper draws still bind. The
            // ownership rules make this unreachable (a retained box is alive
            // and so cannot share an address with a live one); the check
            // keeps that a local fact rather than a lifetime argument.
            Some(state) if state.backing_ptr == backing_ptr => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "intake_vbib_retention: buffer {:#x} retired backing {backing_ptr:#x} \
                     generation {} while the cache wraps generation {} at that address; \
                     the cache entry stays",
                    buffer_id.raw(),
                    page_box.generation(),
                    state.backing_generation
                );
                (MetalHandle::NULL, last_submit_seq)
            }
            _ => (MetalHandle::NULL, last_submit_seq),
        };
        self.perf.bump_vbib_retained_add(page_box.len());
        self.add_retained_bytes(page_box.len());
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Buffer,
                handle: mtl_buffer.raw(),
                page_box: Some(page_box),
                staging_arc: None,
                seq,
                from_texture: false,
            });
    }

    /// Release or replay every upload the GPU has finished with.
    ///
    /// Reads the retirement counter and the aborted-submit counter as a
    /// pair. `coherent_seq` is loaded first: both unix-side stores are
    /// `Release` with the failure recorded before the retirement, so
    /// observing a retirement guarantees the matching failure is visible.
    /// An upload whose seq retired without a failure at or after it is
    /// released; one whose command buffer aborted is re-emitted into this
    /// frame's leading blits and re-queued under this frame's seq.
    ///
    /// Called at the end of `begin_frame`, after `PassState::reset_frame`:
    /// the replays are frame-leading blits, and the rename-at-overlap
    /// bookkeeping they would otherwise consult still holds the previous
    /// frame's draws until that reset runs.
    fn settle_pending_uploads(&mut self) {
        if self.pending_stage_uploads.is_empty() && self.pending_texture_uploads.is_empty() {
            return;
        }
        let Some((settled, failed)) = self.upload_gate() else {
            return;
        };
        self.settle_stage_uploads(settled, failed);
        self.settle_texture_uploads(settled, failed);
    }

    /// The `(settled_seq, failed_seq)` pair the upload-recovery queues gate on.
    ///
    /// `settled_seq` is the lower of the two retirement counters: `coherent_seq`
    /// for the draw command buffer and `upload_coherent_seq` for the upload
    /// command buffer that actually carries the copies. Both matter because
    /// `wait_for_gpu_retire` advances `coherent_seq` by hand so its caller sees
    /// the advance synchronously, and that hand-advance can outrun the upload
    /// handler which is the only thing that records an aborted upload. Taking
    /// the minimum means an entry is never freed before the handler that would
    /// have condemned it has reported in. `None` before the encoder is wired
    /// up; the upload counter is skipped on the defensive path where the
    /// leading blits rode the draw command buffer.
    ///
    /// `coherent_seq` is read first: the unix side records a failure before
    /// bumping either retirement counter and both stores are `Release`, so a
    /// retirement observed here implies the matching failure is visible.
    fn upload_gate(&self) -> Option<(u64, u64)> {
        if self.coherent_seq_ptr == 0 {
            return None;
        }
        // SAFETY: `coherent_seq_ptr` is a PE-heap `Arc<AtomicU64>` raw
        // pointer kept alive by the device-side `Arc`; nonzero here
        // (checked above) means the encoder has been wired up.
        let coherent = unsafe { SharedCounter::new(self.coherent_seq_ptr) }.load(Ordering::Acquire);
        let settled = if self.upload_coherent_seq_ptr == 0 {
            coherent
        } else {
            // SAFETY: same contract as `coherent_seq_ptr`: a device-owned
            // `Arc<AtomicU64>` outliving every frame.
            let upload =
                unsafe { SharedCounter::new(self.upload_coherent_seq_ptr) }.load(Ordering::Acquire);
            coherent.min(upload)
        };
        let failed = if self.failed_seq_ptr == 0 {
            0
        } else {
            // SAFETY: same contract as `coherent_seq_ptr`.
            unsafe { SharedCounter::new(self.failed_seq_ptr) }.load(Ordering::Acquire)
        };
        Some((settled, failed))
    }

    /// Settle the `Staged` VB/IB half of the upload recovery.
    ///
    /// Frees a released entry the way `drain_retired_resource_retention`
    /// does (wrapper destroyed in one bulk thunk, then the backing offered
    /// to the page-box pool), and subtracts its bytes from the shared
    /// retention total exactly once, so a replayed entry (whose bytes stay
    /// live) is never double-counted in either direction.
    fn settle_stage_uploads(&mut self, settled: u64, failed: u64) {
        if self.pending_stage_uploads.is_empty() {
            return;
        }
        let reissue_seq = self.current_submit_seq;
        let mut freed: Vec<StagedUploadRetry> = Vec::new();
        for (fate, entry) in self.pending_stage_uploads.settle(settled, failed) {
            match fate {
                UploadFate::Reissue if self.reissue_stage_upload(entry.payload()) => {
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: failed,
                        "settle_stage_uploads: re-issuing VB/IB uploads discarded by the \
                         aborted submit at seq {failed}; without this the geometry they \
                         carried would stay stale for the rest of the run",
                    );
                    self.pending_stage_uploads.requeue(entry, reissue_seq);
                    continue;
                }
                UploadFate::Abandoned => {
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: entry.key(),
                        "settle_stage_uploads: dropping the upload for buffer {:#x} after \
                         {} aborted submits; its geometry stays stale until the game locks \
                         that range again",
                        entry.key(),
                        entry.attempts(),
                    );
                }
                // Acknowledged, or re-issue found no destination left
                // (the game released the buffer): free it either way.
                UploadFate::Released | UploadFate::Reissue => {}
            }
            freed.push(entry.into_payload());
        }
        self.free_stage_upload_transients(freed);
    }

    /// Free the transient wrapper + PE-heap backing of settled `Staged` VB/IB uploads.
    ///
    /// Destroy order mirrors `drain_retired_resource_retention`: every
    /// `MTLBuffer` wrapper goes in one bulk thunk, and only then do the
    /// backings drop, because Metal holds a `bytesNoCopy` pointer into them
    /// until the wrapper is released. The bytes leave the shared retention
    /// total here, exactly once per entry.
    fn free_stage_upload_transients(&mut self, retries: Vec<StagedUploadRetry>) {
        if retries.is_empty() {
            return;
        }
        let mut wrappers: Vec<u64> = Vec::new();
        let mut backings: Vec<PageBox> = Vec::new();
        for retry in retries {
            if !retry.transient.is_null() {
                wrappers.push(retry.transient.raw());
                self.perf.bump_buffer_destroy();
            }
            self.perf.bump_vbib_retained_sub(retry.page_box.len());
            self.sub_retained_bytes(retry.page_box.len());
            backings.push(retry.page_box);
        }
        destroy_resources_bulk(DestroyKind::Buffer, &wrappers);
        let pool = &*crate::page_box_pool::PAGEBOX_POOL;
        for pb in backings {
            let len = pb.len();
            if pool.recycle(pb).is_none() {
                self.perf.bump_pagebox_pool_recycled(len);
            }
        }
    }

    /// Free every upload the GPU acknowledged, without replaying anything.
    ///
    /// The retention-cap relief drains (`DrainRetiredNow` and the mid-frame
    /// submit) run outside `begin_frame`, so `frame_blit_commands` belongs
    /// to no frame there and a replay pushed into it would be cleared
    /// unnoticed at the next `begin_frame`. They take only the
    /// acknowledged prefix; anything owing a replay waits for the next
    /// `begin_frame`, one frame later, which is the right trade on a path
    /// that only runs after a GPU abort.
    ///
    /// This is what keeps `memory.vbibRetentionCapMB` relief working: a
    /// `Staged` upload's snapshot is counted in the shared retained-bytes
    /// total, so it has to be freeable from the drain the API thread
    /// triggers when it hits the cap.
    fn release_acknowledged_uploads(&mut self) {
        if self.pending_stage_uploads.is_empty() && self.pending_texture_uploads.is_empty() {
            return;
        }
        let Some((settled, failed)) = self.upload_gate() else {
            return;
        };
        let freed: Vec<StagedUploadRetry> = self
            .pending_stage_uploads
            .release_acknowledged(settled, failed)
            .into_iter()
            .map(mtld3d_core::upload_recovery::PendingUpload::into_payload)
            .collect();
        self.free_stage_upload_transients(freed);
        // Texture jobs own only a staging Arc clone, so dropping them here
        // is the whole release.
        drop(
            self.pending_texture_uploads
                .release_acknowledged(settled, failed),
        );
    }

    /// Settle the texture half of the upload recovery.
    ///
    /// Cheaper than the VB/IB half: the job holds a clone of the texture's
    /// own staging `Arc` rather than a private snapshot, so a released entry
    /// frees only the clone and a replay re-reads the same pages the
    /// original upload did (the cached per-mip `MTLBuffer` still wraps
    /// them, so it is a cache hit). When the PE side has since swapped that
    /// `Arc` for a fresh box, the newer upload's own entry orders after this
    /// one under the queue's per-key rule, same as the VB/IB half.
    fn settle_texture_uploads(&mut self, settled: u64, failed: u64) {
        if self.pending_texture_uploads.is_empty() {
            return;
        }
        let reissue_seq = self.current_submit_seq;
        for (fate, entry) in self.pending_texture_uploads.settle(settled, failed) {
            match fate {
                UploadFate::Reissue if self.reissue_texture_upload(entry.payload()) => {
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: failed,
                        "settle_texture_uploads: re-issuing texture uploads discarded by the \
                         aborted submit at seq {failed}; without this their mips would keep \
                         whatever was in the texture before",
                    );
                    self.pending_texture_uploads.requeue(entry, reissue_seq);
                }
                UploadFate::Abandoned => {
                    mtld3d_shared::log_once_warn_by!(
                        target: LOG_TARGET,
                        key: entry.key(),
                        "settle_texture_uploads: dropping the upload for texture {:#x} after \
                         {} aborted submits; that mip keeps its previous contents",
                        entry.key(),
                        entry.attempts(),
                    );
                }
                // Acknowledged, or the game released the texture so the
                // replay had no destination: dropping the job releases the
                // staging Arc clone, which is all the entry owns.
                UploadFate::Released | UploadFate::Reissue => {}
            }
        }
    }

    /// Re-emit one discarded texture mip upload into this frame's leading blits.
    ///
    /// Skips the sampled-this-frame rename `run_texture_upload` does: this
    /// runs from `begin_frame` after `PassState::reset_frame`, so no draw of
    /// this frame has sampled the destination yet. Returns `false` when the
    /// texture is gone from the cache (the game released it) or when the
    /// blit path declined to emit anything, both of which make the upload
    /// moot.
    fn reissue_texture_upload(&mut self, job: &TextureUploadJob) -> bool {
        let Some(handle) = self
            .texture_cache
            .get(&job.info.texture_id)
            .map(|state| state.mtl_texture.raw())
            .filter(|handle| *handle != 0)
        else {
            return false;
        };
        if job.depth > 1 {
            self.run_volume_upload_blit(job, handle)
        } else {
            self.run_texture_upload_blit(job, handle)
        }
    }

    /// Prune the pass state's handle-keyed records for a texture being destroyed.
    ///
    /// The retention drains are the one point where an `MTLTexture` handle
    /// stops naming this resource: the GPU has retired every submission that
    /// referenced it, so no pass under construction can name it either, and
    /// the address is about to become available to the next allocation.
    fn retire_texture_handle(&mut self, handle: u64) {
        // SAFETY: a `DestroyKind::Texture` retention entry carries the `.raw()`
        // of a `MetalHandle<MTLTextureKind>`, so the value is an `MTLTexture`
        // handle. `unregister_texture` only hashes it.
        self.pass_state
            .unregister_texture(unsafe { MetalHandle::<MTLTextureKind>::new(handle) });
    }

    /// Drain resource-retention entries whose seq has retired on the GPU.
    ///
    /// Partitions popped entries by `DestroyKind`, destroys each kind's
    /// handles in one bulk thunk, then drops any `PageBox` backings. Drop
    /// order matters: the wrapper destroy fires before the `PageBox` drops
    /// so Metal releases its `bytesNoCopy` pointer before the backing pages
    /// return to the allocator. Safe to call with a 0 `coherent_seq_ptr`
    /// (no-op before first frame).
    fn drain_retired_resource_retention(&mut self) {
        if self.coherent_seq_ptr == 0 {
            return;
        }
        // SAFETY: `coherent_seq_ptr` is a PE-heap `Arc<AtomicU64>` raw
        // pointer kept alive by the device-side `Arc`; nonzero here
        // (checked above) means the encoder has been wired up.
        let coh = unsafe { SharedCounter::new(self.coherent_seq_ptr) }.load(Ordering::Acquire);
        let mut buffers: Vec<u64> = Vec::new();
        let mut textures: Vec<u64> = Vec::new();
        let mut drained: Vec<PendingResourceRetention> = Vec::new();
        while let Some(front) = self.pending_resource_retention.front() {
            if front.seq > coh {
                break;
            }
            let entry = self
                .pending_resource_retention
                .pop_front()
                .expect("checked front");
            if entry.handle != 0 {
                match entry.kind {
                    DestroyKind::Buffer => {
                        buffers.push(entry.handle);
                        // Attribute to the originating subsystem so
                        // each section's `destroys` row reflects only
                        // its own activity. Texture-staging wrapper
                        // destroys (rename + padded + cached_texture
                        // teardown) flow through the same retention
                        // queue as VB/IB, but the perf split mirrors
                        // where the work was scheduled.
                        if entry.from_texture {
                            self.perf.bump_texture_destroy();
                        } else {
                            self.perf.bump_buffer_destroy();
                        }
                    }
                    DestroyKind::Texture => {
                        textures.push(entry.handle);
                        self.perf.bump_texture_destroy();
                        self.retire_texture_handle(entry.handle);
                    }
                    other => {
                        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                            "drain_retired_resource_retention: unexpected kind {other:?} \
                             — bulk-destroying as single-element call",
                        );
                        destroy_resources_bulk(other, &[entry.handle]);
                    }
                }
            }
            if let Some(ref pb) = entry.page_box {
                self.perf.bump_vbib_retained_sub(pb.len());
                self.sub_retained_bytes(pb.len());
            }
            drained.push(entry);
        }
        destroy_resources_bulk(DestroyKind::Buffer, &buffers);
        destroy_resources_bulk(DestroyKind::Texture, &textures);
        // PageBoxes inside `drained` are released here, after every
        // wrapper destroy thunk has returned. VB/IB boxes are offered to
        // the recycle pool first so the next same-size Lock-rename gets
        // warm, still-committed pages; texture padded-staging boxes and
        // pool rejects (disabled, oversize, cap reached) drop to the
        // allocator exactly as before.
        let pool = &*crate::page_box_pool::PAGEBOX_POOL;
        for mut entry in drained {
            let Some(pb) = entry.page_box.take() else {
                continue;
            };
            if entry.from_texture {
                continue;
            }
            let len = pb.len();
            if pool.recycle(pb).is_none() {
                self.perf.bump_pagebox_pool_recycled(len);
            }
        }
    }

    /// Drain `pending_blit_retention` entries whose `submit_seq` has been retired by the GPU.
    ///
    /// Arcs drop here, bringing staging Box refcounts back to 1 (sole
    /// owner: the texture's `TextureInner`).
    fn reclaim_retired_blit_retention(&mut self) {
        if self.coherent_seq_ptr == 0 {
            return;
        }
        // SAFETY: `coherent_seq_ptr` is the PE-heap `Arc<AtomicU64>`
        // pointer the device shares with the encoder. The Arc outlives
        // every frame referencing it, so the read is well-defined.
        let coh = unsafe { SharedCounter::new(self.coherent_seq_ptr) }.load(Ordering::Acquire);
        while let Some(front) = self.pending_blit_retention.front() {
            if front.submit_seq > coh {
                break;
            }
            // Arc drops here — staging Box refcount falls back to 1
            // (the texture's TextureInner remains the sole owner).
            let entry = self
                .pending_blit_retention
                .pop_front()
                .expect("checked front");
            self.perf.bump_tex_staging_retained_sub(entry.byte_len());
            debug_assert!(
                entry.strong_count() >= 1,
                "pending blit Arc already orphaned"
            );
        }
    }

    /// Lazily wrap a PE-heap staging Box in a Shared `MTLBuffer`.
    ///
    /// Subsequent blits can then read from it. `backing_ptr` and `length`
    /// describe the Box; the cached wrapper is reused until the backing
    /// changes (e.g. the texture's DISCARD/default-contended paths replace
    /// the Arc with a fresh Box), at which point the old wrapper is
    /// destroyed and a fresh one created.
    fn get_or_create_staging_buffer(
        &mut self,
        texture_id: TextureId,
        level: usize,
        keepalive: &Arc<PageBox>,
    ) -> u64 {
        let backing_ptr = keepalive.as_ptr() as u64;
        let length = keepalive.len() as u64;
        let (slot_handle, slot_matches) = {
            let Some(state) = self.texture_cache.get(&texture_id) else {
                error!(
                    target: LOG_TARGET,
                    "get_or_create_staging_buffer: texture_id not in cache — MTLTexture must be \
                     created before its staging buffer",
                );
                return 0;
            };
            if level >= state.mip_staging_buffers.len() {
                error!(
                    target: LOG_TARGET,
                    "get_or_create_staging_buffer: level {level} out of range (levels={})",
                    state.mip_staging_buffers.len(),
                );
                return 0;
            }
            let slot = &state.mip_staging_buffers[level];
            (
                slot.handle,
                !slot.handle.is_null() && slot.backing_ptr == backing_ptr && slot.length == length,
            )
        };
        if slot_matches {
            return slot_handle.raw();
        }
        // Either no wrapper yet, or the PE-side staging Box was
        // re-allocated (different pointer or size). Defer the stale
        // wrapper's destroy via the retention queue gated on the
        // current submit seq — blits emitted earlier in this frame
        // reference `slot.handle` in `frame_blit_commands`, which the
        // unix-side `encode_leading_blits` replays at submit time. A
        // synchronous destroy would free it under them. The stale
        // slot's `keepalive` Arc travels with the retention entry so
        // the wrapper outlives the backing it was wrapping.
        if !slot_handle.is_null() {
            let current_seq = self.current_submit_seq;
            let stale = {
                let state = self
                    .texture_cache
                    .get_mut(&texture_id)
                    .expect("texture_id present — checked above");
                core::mem::take(&mut state.mip_staging_buffers[level])
            };
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Buffer,
                    handle: stale.handle.raw(),
                    page_box: None,
                    staging_arc: stale.keepalive,
                    seq: current_seq,
                    from_texture: true,
                });
        }
        let desc = BufferCreateDesc {
            backing_ptr,
            length,
            id: texture_id.raw(),
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::TexStaging,
        };
        let mut handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        let Some(state) = self.texture_cache.get_mut(&texture_id) else {
            error!(
                target: LOG_TARGET,
                "get_or_create_staging_buffer: texture_id vanished from cache mid-call",
            );
            return 0;
        };
        if status != 0 || handle.is_null() {
            error!(
                target: LOG_TARGET,
                "get_or_create_staging_buffer: CreateBuffer failed \
                 (texture_id={texture_id:#x}, level={level}, length={length})",
            );
            state.mip_staging_buffers[level] = MipStagingBuffer::default();
            return 0;
        }
        state.mip_staging_buffers[level] = MipStagingBuffer {
            handle,
            backing_ptr,
            length,
            keepalive: Some(Arc::clone(keepalive)),
        };
        handle.raw()
    }

    /// Full upload flow for one dirty sub-rect of a texture mip.
    ///
    /// Every path wraps the staging Box in a Shared `MTLBuffer` (lazy on
    /// first upload, cached per mip); the blit paths emit a
    /// `BlitCopyBufferToTexture` into `frame_blit_commands`. The
    /// `job.arc` clone is retained in `current_blit_retention` so the
    /// Box stays alive until the GPU retires the frame — blit reads
    /// happen at command-buffer execution time, long after this
    /// function returns.
    ///
    /// The uploads a blit cannot express (A4R4G4B4 / R5G6B5 / A1R5G5B5 →
    /// BGRA8, and any mip whose row pitch is under the linear texture
    /// alignment) are written by a render pass over the same wrapped
    /// staging instead, spliced into the head of the frame. Every upload is
    /// on the command stream — there is deliberately no CPU-timeline
    /// `replaceRegion` path, which would race a texture referenced by an
    /// in-flight frame.
    pub fn run_texture_upload(&mut self, job: TextureUploadJob) {
        let mut handle = self.get_or_create_texture(&job.info);
        if handle == 0 {
            error!(target: LOG_TARGET, "run_texture_upload: texture handle creation failed");
            return;
        }
        // Per-draw texture versioning: this blit lands in the frame-head
        // leading phase, so if a draw earlier this frame already sampled
        // the texture, writing into the live MTLTexture would rewrite
        // what that draw reads (its per-draw D3D9 state would collapse
        // to frame-final). Rename instead — later draws resolve the
        // fresh handle, the earlier draw keeps the old one.
        //
        // SAFETY: `handle` is the non-null MTLTexture handle just
        // returned by the cache.
        let sampled = self
            .pass_state
            .texture_sampled_this_frame(unsafe { MetalHandle::new(handle) });
        if sampled {
            if job
                .info
                .create_flags
                .contains(TextureCreateFlags::TYPE_CUBE)
            {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: job.info.texture_id.raw(),
                    "run_texture_upload: cube texture {:#x} uploaded after being sampled this frame; per-draw cube versioning is not implemented",
                    job.info.texture_id.raw(),
                );
            } else if job.depth > 1 {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: job.info.texture_id.raw(),
                    "run_texture_upload: volume texture {:#x} uploaded after being sampled \
                     this frame — per-draw versioning not implemented for volumes, earlier \
                     draws will sample the newer content",
                    job.info.texture_id.raw(),
                );
            } else {
                handle = self.rename_sampled_texture(&job, handle);
                if handle == 0 {
                    return;
                }
            }
        }
        // Volume (3D) textures take a dedicated full-box path; 2D textures
        // keep the original hot-path blit untouched.
        let emitted = if job.depth > 1 {
            self.run_volume_upload_blit(&job, handle)
        } else {
            self.run_texture_upload_blit(&job, handle)
        };
        if emitted {
            // Keep the job so an aborted upload command buffer can replay
            // it. It only holds a clone of the texture's own persistent
            // staging Arc, so the memory cost is the clone.
            self.pending_texture_uploads.push(
                job.info.texture_id.raw(),
                self.current_submit_seq,
                job,
            );
        }
    }

    /// Redirect an upload that hit an already-sampled texture to a fresh `MTLTexture`.
    ///
    /// Rename-at-overlap, the texture analogue of `apply_stage_upload`'s
    /// device-buffer rename. Mips the upload does not fully rewrite are
    /// carried over with `copyFromTexture` blits; they append to
    /// `frame_blit_commands` *before* the caller's upload blit, so the
    /// stream order is: earlier uploads → old, copies old → fresh, this
    /// upload → fresh. The dominant case — a single-mip texture with a
    /// full-mip upload — carries nothing over and costs only the texture
    /// allocation. The old handle stays alive via seq-gated retention until
    /// this frame's draws retire.
    ///
    /// Returns the fresh handle, the old handle on allocation failure
    /// (mirrors the buffer rename's fallback: one draw may glitch this
    /// frame, but dropping the upload would persist stale content), or
    /// 0 only if the caller should abort.
    fn rename_sampled_texture(&mut self, job: &TextureUploadJob, old_handle: u64) -> u64 {
        let info = &job.info;
        let desc = self.texture_desc_from_info(info);
        let mut fresh = MetalHandle::<MTLTextureKind>::NULL;
        let mut fresh_srgb = MetalHandle::<MTLTextureKind>::NULL;
        let status = self.batch_create_textures(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut fresh),
            core::slice::from_mut(&mut fresh_srgb),
        );
        if status != 0 || fresh.is_null() {
            error!(
                target: LOG_TARGET,
                "rename_sampled_texture: fresh CreateTexture failed — uploading into the \
                 live texture (one already-emitted draw may sample too-new content this frame)"
            );
            return old_handle;
        }

        // Carry over every mip this upload does not fully rewrite. The
        // upload's own mip is skipped when the job covers it entirely
        // (the standard whole-mip Lock path); a partial-rect job needs
        // the old content underneath.
        let mip_w = (info.width.max(1) >> job.level).max(1);
        let mip_h = (info.height.max(1) >> job.level).max(1);
        let full_cover = job.origin_x == 0
            && job.origin_y == 0
            && job.region_w >= mip_w
            && job.region_h >= mip_h;
        for level in 0..info.levels {
            if level == job.level && full_cover {
                continue;
            }
            let lw = (info.width.max(1) >> level).max(1);
            let lh = (info.height.max(1) >> level).max(1);
            self.frame_blit_commands
                .push(BlitCommand::copy_texture_to_texture_full_mip(
                    old_handle,
                    fresh.raw(),
                    level,
                    lw,
                    lh,
                ));
        }
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);

        // Later draws resolve the fresh handle; the per-mip staging
        // wrappers key on the PE-side backing and are unaffected. The sRGB
        // twin views the storage, so it renames in lock-step: the old twin
        // retires with the old texture and the fresh one takes its slot.
        let mut old_srgb = MetalHandle::<MTLTextureKind>::NULL;
        if let Some(state) = self.texture_cache.get_mut(&info.texture_id) {
            state.mtl_texture = fresh;
            old_srgb = state.mtl_texture_srgb;
            state.mtl_texture_srgb = fresh_srgb;
        }
        self.pass_state.unregister_srgb_twin(old_srgb);
        self.pass_state.register_srgb_twin(fresh_srgb, fresh);
        // The old texture (and its twin view) is read by this frame's
        // already-emitted draws — destroy only after the frame's GPU work
        // retires.
        for old in [old_handle, old_srgb.raw()] {
            if old == 0 {
                continue;
            }
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: old,
                    page_box: None,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: true,
                });
        }
        self.perf.bump_texture_gpu_rename();
        fresh.raw()
    }

    /// `D3DUSAGE_AUTOGENMIPMAP` path: regenerate mips 1..N from the just-uploaded mip 0.
    ///
    /// Called on the encoder thread from the closure pushed by
    /// `texture::schedule_upload` (after upload), and from
    /// `IDirect3DBaseTexture9::GenerateMipSubLevels` (explicit game
    /// trigger). The blit is appended to `frame_blit_commands` right after
    /// the mip-0 `CopyBufferToTexture`, so the unix side replays
    /// `generateMipmapsForTexture` inside the frame's own shared
    /// leading-blit encoder, no per-texture command buffer. A level 0 the
    /// GPU upload pass wrote instead is not in that stream, so it diverts to
    /// the ordered form.
    pub fn run_generate_mipmaps(&mut self, texture_id: TextureId) {
        // A GPU upload pass writes level 0 from a render pass at the head of
        // the frame, which is *after* the leading blit stream; the regen has
        // to follow it in the ordered stream instead.
        if self.upload_pass_textures.contains(&texture_id) {
            self.run_generate_mipmaps_ordered(texture_id);
            return;
        }
        let Some(state) = self.texture_cache.get(&texture_id) else {
            // Texture has no MTL backing yet (no draw has bound it) —
            // mipgen will run on the upload that precedes the first
            // draw, so skipping here is fine.
            return;
        };
        if state.mtl_texture.is_null() {
            return;
        }
        self.frame_blit_commands
            .push(BlitCommand::generate_mipmaps(state.mtl_texture.raw()));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
    }

    /// Regenerate an autogen texture's mip chain in the *ordered* stretch-rect blit stream.
    ///
    /// After the current render pass, not the leading `frame_blit_commands`.
    /// Used when the level-0 modification was itself an ordered op — a
    /// `StretchRect` copy or a render/clear into the texture as a render
    /// target — so the regen must follow it rather than lead the frame.
    pub fn run_generate_mipmaps_ordered(&mut self, texture_id: TextureId) {
        let Some(state) = self.texture_cache.get(&texture_id) else {
            return;
        };
        if state.mtl_texture.is_null() {
            return;
        }
        let handle = state.mtl_texture.raw();
        // A render target cleared (or drawn) without a following draw leaves the
        // clear stashed as a pending load-action; materialize it onto the (still
        // current) attachment first so the regen reads the cleared level 0.
        self.pass_state.flush_pending_clears();
        self.pass_state.note_texture_read(state.mtl_texture);
        self.end_current_pass("autogen_rt_regen");
        self.push_stretch_rect_blit(BlitCommand::generate_mipmaps(handle));
    }

    /// Blit-based 2D upload.
    ///
    /// A job the GPU upload pass takes diverts to it up front; the rest
    /// reuse the per-mip staging `MTLBuffer` (wrapping the game's staging
    /// `PageBox`) and emit a `BlitCopyBufferToTexture` against the frame's
    /// leading blit pass.
    fn run_texture_upload_blit(&mut self, job: &TextureUploadJob, texture_handle: u64) -> bool {
        if let Some(outcome) = self.try_texture_upload_pass(job, texture_handle) {
            return outcome;
        }
        let _t = mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::TexRaw));
        let backing_length = job.arc.len() as u64;
        if backing_length == 0 {
            return false;
        }

        // Compute the blit descriptor against the staging buffer's
        // src_pitch stride. The format's block height is carried through
        // alongside `info` because a Metal blit is measured in block rows,
        // not pixel rows: it turns `region_h` into the row count the GPU
        // actually reads, both for the alignment-pad repack below and for
        // the slice size the copy is given.
        let staging_buffer_handle =
            self.get_or_create_staging_buffer(job.info.texture_id, job.staging_index, &job.arc);
        if staging_buffer_handle == 0 {
            return false;
        }

        let (info, block_height) = if job.bytes_per_pixel == 0 {
            // Compressed (BC1/2/3). Sub-rect must land on the block
            // grid; otherwise fall back to a full-mip blit from the
            // start of the staging buffer. Both variants are correct
            // because the staging preserves every byte the game wrote.
            let fmt = map_d3d_format(job.src_d3d_format)
                .expect("compressed format already mapped at CreateTexture");
            let bw = fmt.block_width();
            let bh = fmt.block_height();
            let bb = fmt.block_bytes();
            let mip_w = (job.info.width.max(1) >> job.level).max(1);
            let mip_h = (job.info.height.max(1) >> job.level).max(1);
            let aligned = job.origin_x.is_multiple_of(bw)
                && job.origin_y.is_multiple_of(bh)
                && (job.region_w.is_multiple_of(bw) || job.origin_x + job.region_w == mip_w)
                && (job.region_h.is_multiple_of(bh) || job.origin_y + job.region_h == mip_h);
            if aligned {
                let block_x = job.origin_x / bw;
                let block_y = job.origin_y / bh;
                let buffer_offset = u64::from(block_y) * u64::from(job.src_pitch)
                    + u64::from(block_x) * u64::from(bb);
                let info = CopyBufferToTextureInfo {
                    buffer_handle: staging_buffer_handle,
                    buffer_offset,
                    bytes_per_row: job.src_pitch,
                    texture_handle,
                    destination_slice: job.destination_slice,
                    mip_level: job.level,
                    origin_x: job.origin_x,
                    origin_y: job.origin_y,
                    region_w: job.region_w,
                    region_h: job.region_h,
                    depth: 1,
                    bytes_per_image: 0,
                };
                (info, bh)
            } else {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "run_texture_upload_blit: compressed sub-rect ({}+{},{}+{}) unaligned to {}×{} block grid → full-mip fallback",
                    job.origin_x,
                    job.region_w,
                    job.origin_y,
                    job.region_h,
                    bw,
                    bh,
                );
                let info = CopyBufferToTextureInfo {
                    buffer_handle: staging_buffer_handle,
                    buffer_offset: 0,
                    bytes_per_row: job.src_pitch,
                    texture_handle,
                    destination_slice: job.destination_slice,
                    mip_level: job.level,
                    origin_x: 0,
                    origin_y: 0,
                    region_w: mip_w,
                    region_h: mip_h,
                    depth: 1,
                    bytes_per_image: 0,
                };
                (info, bh)
            }
        } else {
            // Uncompressed path. Sub-rect offset is
            // origin_y * pitch + origin_x * bpp bytes into the Box.
            let buffer_offset = u64::from(job.origin_y) * u64::from(job.src_pitch)
                + u64::from(job.origin_x) * u64::from(job.bytes_per_pixel);
            let info = CopyBufferToTextureInfo {
                buffer_handle: staging_buffer_handle,
                buffer_offset,
                bytes_per_row: job.src_pitch,
                texture_handle,
                destination_slice: job.destination_slice,
                mip_level: job.level,
                origin_x: job.origin_x,
                origin_y: job.origin_y,
                region_w: job.region_w,
                region_h: job.region_h,
                depth: 1,
                bytes_per_image: 0,
            };
            (info, 1)
        };
        let num_blit_rows = mtld3d_shared::blit_geometry::block_rows(info.region_h, block_height);

        // `copyFromBuffer:toTexture:` requires `sourceBytesPerRow` to
        // be ≥ `device.minimumLinearTextureAlignmentForPixelFormat`
        // (16 on Apple Silicon, 256 on Mac2). Bottom-of-chain mips
        // (BC1 1×1 = 8 bytes, BGRA8 1×1 = 4 bytes, …) trip it. Apple
        // Silicon happens to tolerate the violation today but the
        // behaviour is officially undefined. Repack the affected rows
        // into a transient padded MTLBuffer and aim the blit there.
        let info = if info.bytes_per_row < self.gpu_caps.min_linear_texture_align {
            match self.repack_blit_source_padded(&job.arc, &info, num_blit_rows) {
                Some(padded_info) => padded_info,
                None => return false,
            }
        } else {
            // Notify the staging MTLBuffer (no-op on UMA). The padded
            // path notifies the transient buffer instead.
            self.enqueue_notify_buffer_did_modify_range(staging_buffer_handle, 0, backing_length);
            info
        };

        // Single-slice (`depth == 1`) copy: `bytes_per_image` is the slice's
        // block-row count times the *final* (post-padding) row stride. For a
        // compressed level that is `block_height` times smaller than the
        // pixel-row product.
        let info = CopyBufferToTextureInfo {
            bytes_per_image: mtld3d_shared::blit_geometry::bytes_per_image(
                info.bytes_per_row,
                info.region_h,
                block_height,
            ),
            ..info
        };
        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_texture(&info));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        // Counts every successful blit-path upload (padded subset
        // included) — the total texture uploads per frame.
        self.perf.bump_texture_blit_upload();
        // Retain the staging Box for the GPU's view of this frame —
        // even on the padded path the source bytes were just copied
        // out, but keeping the Arc alive is harmless and uniform.
        // `pending_blit_retention` releases this clone once
        // `coherent_seq >= submit_seq`; the caller's own clone in
        // `pending_texture_uploads` outlives it by however long the
        // upload takes to be acknowledged.
        self.current_blit_retention.push(Arc::clone(&job.arc));
        true
    }

    /// Volume (3D) full-box upload.
    ///
    /// Copies the level's whole staging box (`depth` contiguous slices,
    /// each `slice_pitch` bytes) into the 3D `MTLTexture`. Kept separate
    /// from `run_texture_upload_blit` so the 2D hot path is untouched;
    /// volumes always re-upload the whole box on Unlock (the staging
    /// retains every byte the game wrote, so a full-box copy subsumes any
    /// sub-box lock), which keeps the origin / sub-rect bookkeeping
    /// trivial.
    ///
    /// `lock_box` sizes a slice as `row_pitch * ceil(mip_h / block_h)` with
    /// no inter-slice gap, so the slices are contiguous in the box — a
    /// single `depth`-slice `copyFromBuffer` with `bytesPerImage =
    /// slice_pitch` reads them all. When `row_pitch` is below Metal's
    /// `minimumLinearTextureAlignmentForPixelFormat`, a format the upload
    /// pass cannot write (compressed, or carrying a sampler swizzle) has
    /// every row across every slice repacked to the padded stride (the rows
    /// being contiguous makes this a single `region_rows * depth` repack),
    /// and `bytes_per_image` widens to `padded_pitch * region_rows`.
    fn run_volume_upload_blit(&mut self, job: &TextureUploadJob, texture_handle: u64) -> bool {
        if let Some(outcome) = self.try_texture_upload_pass(job, texture_handle) {
            return outcome;
        }
        let _t = mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::TexRaw));
        let backing_length = job.arc.len() as u64;
        if backing_length == 0 {
            return false;
        }
        let src_pitch = job.src_pitch;
        let slice_pitch = job.slice_pitch;
        let depth = job.depth.max(1);
        // Rows per slice (block-rows for compressed): `slice_pitch` is
        // exactly `src_pitch * block_rows`, so recover it by division.
        let region_rows = slice_pitch.checked_div(src_pitch).unwrap_or(0);
        if region_rows == 0 {
            return false;
        }
        let mip_w = (job.info.width.max(1) >> job.level).max(1);
        let mip_h = (job.info.height.max(1) >> job.level).max(1);

        let staging_buffer_handle =
            self.get_or_create_staging_buffer(job.info.texture_id, job.staging_index, &job.arc);
        if staging_buffer_handle == 0 {
            return false;
        }

        let info = CopyBufferToTextureInfo {
            buffer_handle: staging_buffer_handle,
            buffer_offset: 0,
            bytes_per_row: src_pitch,
            texture_handle,
            destination_slice: 0,
            mip_level: job.level,
            origin_x: 0,
            origin_y: 0,
            region_w: mip_w,
            region_h: mip_h,
            depth,
            bytes_per_image: slice_pitch,
        };

        // Same `minimumLinearTextureAlignmentForPixelFormat` requirement as
        // the 2D path. Repack every row across every slice — the slices are
        // contiguous, so `region_rows * depth` covers the whole box — and
        // widen the slice stride to the padded row stride.
        let info = if info.bytes_per_row < self.gpu_caps.min_linear_texture_align {
            let total_rows = region_rows.saturating_mul(depth);
            match self.repack_blit_source_padded(&job.arc, &info, total_rows) {
                Some(mut padded_info) => {
                    padded_info.bytes_per_image =
                        padded_info.bytes_per_row.saturating_mul(region_rows);
                    padded_info
                }
                None => return false,
            }
        } else {
            self.enqueue_notify_buffer_did_modify_range(staging_buffer_handle, 0, backing_length);
            info
        };

        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_texture(&info));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        self.perf.bump_texture_blit_upload();
        self.current_blit_retention.push(Arc::clone(&job.arc));
        true
    }

    /// Which upload-pass decode this job takes, or `None` for the blit path.
    ///
    /// An expansion has no blit form and always takes the pass. A verbatim
    /// copy takes it only when the staging row pitch is under Metal's linear
    /// texture alignment, which is what a blit copy cannot accept; above it
    /// the blit is the cheaper write.
    fn upload_pass_decode(&self, job: &TextureUploadJob) -> Option<UploadDecode> {
        let decode =
            mtld3d_core::upload_pass::upload_decode(job.src_d3d_format, job.info.pixel_format)?;
        (mtld3d_core::upload_pass::is_expansion(decode)
            || job.src_pitch < self.gpu_caps.min_linear_texture_align)
            .then_some(decode)
    }

    /// Run the upload as a GPU pass when it takes one, reporting whether the caller is done.
    ///
    /// `None` means the job belongs on the blit path: either it never took
    /// the pass, or the pass declined a verbatim copy (a pipeline-create
    /// failure) and the blit's CPU repack can still write it. `Some` is the
    /// result the caller returns; a declined expansion is `Some(false)`,
    /// because no blit can widen those texels.
    fn try_texture_upload_pass(
        &mut self,
        job: &TextureUploadJob,
        texture_handle: u64,
    ) -> Option<bool> {
        let decode = self.upload_pass_decode(job)?;
        if self.run_texture_upload_pass(job, texture_handle, decode) {
            return Some(true);
        }
        if mtld3d_core::upload_pass::is_expansion(decode) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "run_texture_upload: the upload pass declined a texel-widening expansion; no \
                 blit can widen those texels, so the mip keeps its previous contents",
            );
            return Some(false);
        }
        None
    }

    /// GPU upload pass: write one dirty region into the texture with a render quad.
    ///
    /// Serves the two upload shapes `copyFromBuffer:toTexture:` cannot take.
    /// A source narrower than the `Bgra8Unorm` texture behind it has no
    /// verbatim copy at all: a packed 16-bit format on a device without the
    /// native formats, or 24-bit R8G8B8 anywhere. The widening happens in the
    /// fragment function, which writes D3D channel order and forces alpha
    /// opaque for the formats that store none. A mip whose row pitch
    /// is below `min_linear_texture_align` has no legal blit source pitch:
    /// the fragment function addresses the staging by texel, so the pitch is
    /// just a multiplier.
    ///
    /// The staging keeps its D3D layout (Lock semantics and upload-abort
    /// replay both read it) and is wrapped in the same cached per-mip
    /// `MTLBuffer` the blit path uses. Handles the 2D dirty-rect shape and
    /// a cube face in one pass, and the volume whole-box shape
    /// (`job.depth > 1`) in one pass per slice.
    fn run_texture_upload_pass(
        &mut self,
        job: &TextureUploadJob,
        texture_handle: u64,
        decode: UploadDecode,
    ) -> bool {
        let _t = mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::TexRaw));
        let backing_length = job.arc.len() as u64;
        if backing_length == 0 || job.bytes_per_pixel != decode.bytes_per_texel() {
            return false;
        }
        let pipeline = self.get_or_create_upload_pipeline(job.info.pixel_format);
        if pipeline == 0 {
            return false;
        }
        let staging_buffer_handle =
            self.get_or_create_staging_buffer(job.info.texture_id, job.staging_index, &job.arc);
        if staging_buffer_handle == 0 {
            return false;
        }
        // Non-UMA: the game wrote these pages on the CPU. The notify rides
        // the frame-head blit stream, which runs before every pass.
        self.enqueue_notify_buffer_did_modify_range(staging_buffer_handle, 0, backing_length);

        let mip_w = (job.info.width.max(1) >> job.level).max(1);
        let mip_h = (job.info.height.max(1) >> job.level).max(1);
        let depth = job.depth.max(1);
        let emit = UploadPassInputs {
            pipeline,
            depth_state: self.get_or_create_depth_stencil(&DepthStencilSnapshot::inert()),
            staging_buffer_handle,
            texture_handle,
            format: job.info.pixel_format,
            level: job.level,
            mip_size: (mip_w, mip_h),
            src_pitch: job.src_pitch,
            decode,
        };
        if depth > 1 {
            // Volumes re-upload the whole box on Unlock and their slices are
            // contiguous in it, so slice `s` starts `s * slice_pitch` in.
            for slice in 0..depth {
                self.emit_upload_pass(
                    &emit,
                    slice,
                    (0, 0, mip_w, mip_h),
                    slice.saturating_mul(job.slice_pitch),
                );
            }
        } else {
            // Clamp the dirty rect into the mip: the viewport is not clamped
            // on the unix side the way the scissor is.
            let w = job.region_w.min(mip_w.saturating_sub(job.origin_x));
            let h = job.region_h.min(mip_h.saturating_sub(job.origin_y));
            self.emit_upload_pass(
                &emit,
                job.destination_slice,
                (job.origin_x, job.origin_y, w, h),
                0,
            );
        }

        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "texture upload pass in use: uploads a blit cannot express (texel widening, \
             row pitch under the {}-byte linear texture alignment) render into the destination",
            self.gpu_caps.min_linear_texture_align,
        );
        self.upload_pass_textures.insert(job.info.texture_id);
        self.perf.bump_texture_blit_upload();
        self.perf.bump_texture_expand_upload();
        // The pass reads the staging at command-buffer execution time, long
        // after this returns; hold the Box for the GPU's view of the frame.
        self.current_blit_retention.push(Arc::clone(&job.arc));
        true
    }

    /// Splice one slice's upload pass into the frame.
    ///
    /// `rect` is the dirty region in destination texels, `base_offset` the
    /// byte offset of the slice inside the staging slab. A 2D upload writes
    /// the rect at the coordinates it already occupies in the staging, so
    /// the fragment function derives its source address from the destination
    /// position alone and `base_offset` is zero; a volume slice carries its
    /// own base.
    fn emit_upload_pass(
        &mut self,
        emit: &UploadPassInputs,
        slice: u32,
        rect: (u32, u32, u32, u32),
        base_offset: u32,
    ) {
        let (x, y, w, h) = rect;
        if w == 0 || h == 0 {
            return;
        }
        let mut args = [0u8; RGBA_BYTE_LEN as usize];
        for (i, v) in [
            base_offset,
            emit.src_pitch,
            emit.decode.wire(),
            emit.decode.bytes_per_texel(),
        ]
        .iter()
        .enumerate()
        {
            args[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let args_ptr = self.scratch.alloc(&args);
        let mut cmds = core::mem::take(&mut self.upload_pass_commands);
        cmds.clear();
        cmds.push(Command::set_render_pipeline_state(emit.pipeline));
        cmds.push(Command::set_depth_stencil_state(emit.depth_state));
        cmds.push(Command::set_scissor_rect(x, y, w, h));
        cmds.push(Command::set_fragment_bytes_at(args_ptr, RGBA_BYTE_LEN, 0));
        cmds.push(Command::set_fragment_buffer(
            emit.staging_buffer_handle,
            0,
            1,
        ));
        cmds.push(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        let target = UploadPassTarget {
            // SAFETY: `texture_handle` is the live MTLTexture address the
            // caller resolved out of the texture cache.
            texture: unsafe { MetalHandle::<MTLTextureKind>::new(emit.texture_handle) },
            subresource: (slice, emit.level),
            size: emit.mip_size,
            format: emit.format,
            rect,
        };
        self.pass_state.push_upload_pass(&target, &cmds);
        self.upload_pass_commands = cmds;
    }

    /// Repack `num_blit_rows` source rows from `staging` into a transient `PageBox`.
    ///
    /// Source rows sit at `info.buffer_offset` / `info.bytes_per_row`
    /// stride; the transient box's row stride is
    /// `gpu_caps.min_linear_texture_align`. Wraps that `PageBox` in a fresh
    /// `MTLBuffer`, queues both for retire, and returns an updated `info`
    /// aimed at the new buffer. Returns `None` only on `CreateBuffer`
    /// failure.
    ///
    /// Why this exists: the staging `PageBox` is sized to D3D's
    /// per-mip pitch, which for tiny mips (1×1 BGRA8 = 4 bytes,
    /// 1-block BC1 = 8 bytes) is below
    /// `minimumLinearTextureAlignmentForPixelFormat:`. The Metal blit
    /// spec says behaviour is undefined in that case. `ASi` tolerates
    /// it today, Mac2 won't.
    fn repack_blit_source_padded(
        &mut self,
        staging: &Arc<PageBox>,
        info: &CopyBufferToTextureInfo,
        num_blit_rows: u32,
    ) -> Option<CopyBufferToTextureInfo> {
        let src_pitch = info.bytes_per_row as usize;
        let padded_pitch = self.gpu_caps.min_linear_texture_align as usize;
        debug_assert!(padded_pitch > src_pitch);

        // Snap buffer_offset to the start of its row; the within-row
        // offset (origin_x * bpp / block_x * block_bytes) is preserved
        // verbatim into the padded layout since each padded row begins
        // with a verbatim copy of the source row.
        let abs_offset =
            usize::try_from(info.buffer_offset).expect("buffer offset fits host address space");
        let start_row = abs_offset / src_pitch;
        let intra_row_offset = abs_offset - start_row * src_pitch;

        let padded_size = padded_pitch
            .checked_mul(num_blit_rows as usize)
            .expect("padded blit-source size overflow");
        let mut padded = PageBox::new_uninit(padded_size);
        // SAFETY: `start_row * src_pitch` stays within the staging slab per
        // the caller's row-bound contract.
        let src_base = unsafe { staging.as_ptr().add(start_row * src_pitch) };
        let dst_base = padded.as_mut_ptr();
        for row in 0..num_blit_rows as usize {
            // SAFETY: `src_base + row * src_pitch` covers `src_pitch` bytes
            // within the staging slab; `dst_base + row * padded_pitch`
            // covers `padded_pitch >= src_pitch` bytes within the just-
            // allocated `padded` slab. Source and dest are disjoint slabs.
            let src_row = unsafe { src_base.add(row * src_pitch) };
            // SAFETY: dst offset stays within `padded_size`.
            let dst_row = unsafe { dst_base.add(row * padded_pitch) };
            // SAFETY: both pointers and the byte count are valid as above.
            unsafe { core::ptr::copy_nonoverlapping(src_row, dst_row, src_pitch) };
        }

        let padded_ptr = padded.as_ptr() as u64;
        let padded_len = padded.len() as u64;
        let desc = BufferCreateDesc {
            backing_ptr: padded_ptr,
            length: padded_len,
            id: 0,
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::Repack,
        };
        let mut padded_handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut padded_handle),
        );
        if status != 0 || padded_handle.is_null() {
            error!(
                target: LOG_TARGET,
                "repack_blit_source_padded: CreateBuffer failed (status={status:#x}, \
                 src_pitch={src_pitch}, padded_pitch={padded_pitch}, num_blit_rows={num_blit_rows}, \
                 padded_size={padded_size}, padded_len={}, padded_ptr={padded_ptr:#x})",
                padded.len(),
            );
            return None;
        }

        // Notify the transient MTLBuffer for non-UMA. The PageBox just
        // got its full padded region written by the memcpy above.
        self.enqueue_notify_buffer_did_modify_range(padded_handle.raw(), 0, padded_len);

        // Hand both the wrapper and the PageBox to the frame's
        // retention queue — destroy fires after the GPU retires the
        // submit_seq we'll stamp in `submit`. Order matters: the
        // wrapper must drop first so Metal releases its
        // `bytesNoCopy` pointer before the PageBox dealloc returns
        // pages to the allocator.
        // Account the PageBox into `vbib_retained_bytes` so the
        // drain's matching `_sub` doesn't silently underreport (the
        // counter is named for VB/IB but tracks every PageBox sitting
        // in the shared retention queue). Bump the operation count
        // separately so the perf summary makes the padding-path
        // frequency visible.
        self.perf.bump_vbib_retained_add(padded.len());
        self.add_retained_bytes(padded.len());
        self.perf.bump_texture_blit_padded_upload();
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Buffer,
                handle: padded_handle.raw(),
                page_box: Some(padded),
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });

        Some(CopyBufferToTextureInfo {
            buffer_handle: padded_handle.raw(),
            buffer_offset: intra_row_offset as u64,
            bytes_per_row: u32::try_from(padded_pitch)
                .expect("Metal min_linear_texture_align fits u32"),
            ..*info
        })
    }

    /// Upload `rows` into the standalone colour `MTLTexture` `color_handle`.
    ///
    /// `rows` is `src_stride * height` bytes of the source's own rows; this is
    /// the `UnlockRect` half of a lockable render target (`CreateRenderTarget`
    /// with `Lockable == TRUE`), whose staging carries the row pitch every
    /// host-visible surface store uses. Copies the rows into a fresh
    /// page-aligned `PageBox` (padding each row up to
    /// `min_linear_texture_align` if the source stride is below it), wraps
    /// that in a transient `MTLBuffer`, appends a `CopyBufferToTexture` to the
    /// frame's leading blit pass, and retires both after the GPU retires this
    /// frame. The bytes are *copied* here (the caller's staging is not aliased
    /// across the API/encoder boundary).
    pub fn upload_bytes_to_color_handle(
        &mut self,
        color_handle: u64,
        rows: &[u8],
        width: u32,
        height: u32,
        src_stride: u32,
    ) {
        if color_handle == 0 || width == 0 || height == 0 || src_stride == 0 {
            return;
        }
        let src_stride = src_stride as usize;
        // `copyFromBuffer:toTexture:` requires `sourceBytesPerRow` ≥
        // `minimumLinearTextureAlignmentForPixelFormat:`; pad narrow rows up.
        let padded_stride = src_stride.max(self.gpu_caps.min_linear_texture_align as usize);
        let Some(padded_size) = padded_stride.checked_mul(height as usize) else {
            error!(target: LOG_TARGET, "upload_bytes_to_color_handle: staging size overflow");
            return;
        };
        if rows.len() < src_stride.saturating_mul(height as usize) {
            error!(
                target: LOG_TARGET,
                "upload_bytes_to_color_handle: source slice {} shorter than {src_stride}*{height}",
                rows.len(),
            );
            return;
        }
        // `bytesNoCopy` needs page-aligned backing, so the rows must land in a
        // `PageBox` (re-packing the source rows into the padded stride).
        let mut staging = PageBox::new_uninit(padded_size);
        let dst_base = staging.as_mut_ptr();
        let src_base = rows.as_ptr();
        for row in 0..height as usize {
            // SAFETY: the source row `[row*src_stride, +src_stride)` is in
            // bounds (`rows.len() >= src_stride * height`, checked above).
            let src_row = unsafe { src_base.add(row * src_stride) };
            // SAFETY: the dest row `[row*padded_stride, +src_stride)` is
            // within `staging` (`padded_stride >= src_stride`, alloc has
            // `padded_size = padded_stride * height` bytes).
            let dst_row = unsafe { dst_base.add(row * padded_stride) };
            // SAFETY: both pointers and the byte count are valid per above, and
            // `rows` / `staging` are distinct allocations (disjoint copy).
            unsafe { core::ptr::copy_nonoverlapping(src_row, dst_row, src_stride) };
        }

        let staging_len = staging.len() as u64;
        let desc = BufferCreateDesc {
            backing_ptr: staging.as_ptr() as u64,
            length: staging_len,
            id: 0,
            storage_mode: buffer_storage_mode(self.gpu_caps.unified_memory),
            kind: BufferKind::Repack,
        };
        let mut staging_handle = MetalHandle::<MTLBufferKind>::NULL;
        let status = self.batch_create_buffers(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut staging_handle),
        );
        if status != 0 || staging_handle.is_null() {
            error!(
                target: LOG_TARGET,
                "upload_bytes_to_color_handle: CreateBuffer failed (status={status:#x}, len={staging_len})",
            );
            return;
        }
        // Non-UMA: the CPU just wrote the staging slab; notify before the blit.
        self.enqueue_notify_buffer_did_modify_range(staging_handle.raw(), 0, staging_len);

        let bytes_per_row = u32::try_from(padded_stride).expect("padded stride fits u32");
        let info = CopyBufferToTextureInfo {
            buffer_handle: staging_handle.raw(),
            buffer_offset: 0,
            bytes_per_row,
            texture_handle: color_handle,
            destination_slice: 0,
            mip_level: 0,
            origin_x: 0,
            origin_y: 0,
            region_w: width,
            region_h: height,
            // Single-slice 2D copy: `bytes_per_image == bytes_per_row *
            // region_h`, matching the blit's pre-existing implicit value.
            depth: 1,
            bytes_per_image: bytes_per_row.saturating_mul(height),
        };
        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_texture(&info));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        self.perf.bump_texture_blit_upload();

        // Retire the wrapper + PageBox after the GPU retires this frame — the
        // blit reads them at command-buffer execution. Wrapper first so Metal
        // releases its `bytesNoCopy` pointer before the PageBox frees.
        self.perf.bump_vbib_retained_add(staging.len());
        self.add_retained_bytes(staging.len());
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Buffer,
                handle: staging_handle.raw(),
                page_box: Some(staging),
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });
    }

    /// Upload `tight` at its own extent, then resample it into a smaller colour texture.
    ///
    /// The resizing counterpart of [`Self::upload_bytes_to_color_handle`], for
    /// the back buffer's `ReleaseDC` write-back under a `render.scale` below
    /// 100%. The rows land in a scratch texture at the extent they describe,
    /// which the blit-quad pipeline then samples across the destination with a
    /// linear filter, the same resample a scaling `StretchRect` runs. The
    /// upload rides the frame's leading blit pass and the quad is a render pass
    /// after it, so the two are ordered without a barrier of their own.
    ///
    /// Declines, once, when the scratch cannot be created: the destination
    /// keeps the pixels the GPU already holds, which is what an unresampled
    /// direct copy could not have given it either.
    pub fn upload_bytes_resampled(&mut self, target: &ResampledUpload, tight: &[u8]) {
        let (src_w, src_h) = target.logical;
        let (dst_w, dst_h) = target.texture;
        if target.color_handle == 0 || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return;
        }
        let scratch = self.ensure_dc_write_back_scratch(src_w, src_h, target.format);
        if scratch == 0 {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "ReleaseDC write-back: no {src_w}x{src_h} scratch texture to resample \
                 through, GDI's drawing is dropped"
            );
            return;
        }
        // `tight` is packed at exactly `src_w` pixels per row.
        self.upload_bytes_to_color_handle(
            scratch,
            tight,
            src_w,
            src_h,
            src_w * target.bytes_per_pixel,
        );
        let src = BlitSide {
            handle: scratch,
            rect: StretchRegion {
                x: 0,
                y: 0,
                w: src_w,
                h: src_h,
            },
            dims: (src_w, src_h),
            mip: 0,
            slice: None,
            msaa: MetalHandle::NULL,
            msaa_srgb: MetalHandle::NULL,
            sample_count: 1,
        };
        let dst = BlitSide {
            handle: target.color_handle,
            rect: StretchRegion {
                x: 0,
                y: 0,
                w: dst_w,
                h: dst_h,
            },
            dims: (dst_w, dst_h),
            mip: 0,
            slice: None,
            msaa: target.msaa,
            msaa_srgb: target.msaa_srgb,
            sample_count: target.sample_count,
        };
        self.stretch_blit_scaled(
            &src,
            &dst,
            target.format,
            mtld3d_core::stretch_rect::BlitDecode::None,
            mtld3d_types::D3DTEXF_LINEAR,
        );
    }

    /// Get, or build, the scratch texture [`Self::upload_bytes_resampled`] stages through.
    ///
    /// Returns 0 when Metal declines the texture. One slot rather than a map:
    /// the only extent asked for is the back buffer's reported one, which
    /// changes at `Reset` and never per frame, so a replacement retires the
    /// previous texture on the seq-gated queue instead of accumulating entries.
    fn ensure_dc_write_back_scratch(
        &mut self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> u64 {
        if self.dc_write_back_scratch_key == (width, height, format)
            && !self.dc_write_back_scratch.is_null()
        {
            return self.dc_write_back_scratch.raw();
        }
        let desc = TextureCreateDesc {
            tex_id: 0,
            width,
            height,
            depth: 1,
            levels: 1,
            pixel_format: format,
            storage_mode: StorageMode::Private,
            flags: TextureCreateFlags::empty(),
            swizzle_r: Swizzle::Red,
            swizzle_g: Swizzle::Green,
            swizzle_b: Swizzle::Blue,
            swizzle_a: Swizzle::Alpha,
            // Sampled by the blit quad and written by a buffer copy; neither
            // needs the render-target usage bit, which the unix side adds only
            // on request.
            usage_flags: TextureUsage::empty(),
        };
        let mut handle = MetalHandle::<MTLTextureKind>::NULL;
        let mut srgb_handle = MetalHandle::<MTLTextureKind>::NULL;
        let status = self.batch_create_textures(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
            core::slice::from_mut(&mut srgb_handle),
        );
        if status != 0 || handle.is_null() {
            return 0;
        }
        if !srgb_handle.is_null() {
            // The quad samples the linear view; the eager twin has no reader
            // here, so retire it rather than leave it registered.
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: srgb_handle.raw(),
                    page_box: None,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: true,
                });
        }
        if !self.dc_write_back_scratch.is_null() {
            self.pending_resource_retention
                .push_back(PendingResourceRetention {
                    kind: DestroyKind::Texture,
                    handle: self.dc_write_back_scratch.raw(),
                    page_box: None,
                    staging_arc: None,
                    seq: self.current_submit_seq,
                    from_texture: true,
                });
        }
        self.dc_write_back_scratch = handle;
        self.dc_write_back_scratch_key = (width, height, format);
        handle.raw()
    }

    /// Remove a texture from the cache and park its Metal handles on the retention queue.
    ///
    /// The `MTLTexture` + every per-mip staging `MTLBuffer` wrapper go on
    /// `pending_resource_retention` gated on the current submit seq. Called
    /// from `texture_release` when the D3D9 refcount hits 0. Synchronous
    /// destroy would race against `BlitCommand`s pushed earlier in this frame
    /// that still reference these handles in `dst_handle` / `src_handle`; the
    /// drain destroys them only after `coherent_seq >= seq`. The
    /// `texture_destroys` counter is bumped at drain time, not here, so
    /// it tracks "actually destroyed", not "scheduled".
    pub fn destroy_cached_texture(&mut self, texture_id: TextureId) {
        if let Some(state) = self.texture_cache.remove(&texture_id) {
            let seq = self.current_submit_seq;
            let mtl_texture = state.mtl_texture;
            self.pass_state.unregister_srgb_twin(state.mtl_texture_srgb);
            // `into_iter` so each slot's `keepalive` Arc moves into the
            // retention entry — the `MTLBuffer` wrapper must outlive
            // the page-backing it wraps via `bytesNoCopy`.
            for s in state.mip_staging_buffers {
                if !s.handle.is_null() {
                    self.pending_resource_retention
                        .push_back(PendingResourceRetention {
                            kind: DestroyKind::Buffer,
                            handle: s.handle.raw(),
                            page_box: None,
                            staging_arc: s.keepalive,
                            seq,
                            from_texture: true,
                        });
                }
            }
            if !mtl_texture.is_null() {
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Texture,
                        handle: mtl_texture.raw(),
                        page_box: None,
                        staging_arc: None,
                        seq,
                        from_texture: true,
                    });
            }
            if !state.mtl_texture_srgb.is_null() {
                self.pending_resource_retention
                    .push_back(PendingResourceRetention {
                        kind: DestroyKind::Texture,
                        handle: state.mtl_texture_srgb.raw(),
                        page_box: None,
                        staging_arc: None,
                        seq,
                        from_texture: true,
                    });
            }
        }
    }

    /// Look up or create an `MTLSamplerState` for the given D3D9 sampler state.
    ///
    /// Key + params both come from `mtld3d_core::sampler_state` so the static
    /// invariant "key ⊇ consumed fields" holds by construction.
    ///
    /// `is_compare` flips the sampler into the D3D9 hardware-shadow PCF
    /// variant: same min/mag/mip/address state but `compareFunction =
    /// LessEqual` on the descriptor, distinct cache entry, used when the
    /// matching texture slot is bound to a depth-format texture
    /// (sampleable shadow map). The MSL emitter pairs this with a
    /// `sample_compare` call site keyed on the same `depth_sampler_mask`.
    pub fn get_or_create_sampler(
        &mut self,
        stage: u32,
        sampler_state: &[u32; SAMPLER_STATE_COUNT],
        is_compare: bool,
        force_point: bool,
    ) -> u64 {
        if !force_point
            && let Some(Some(memo)) = self.sampler_resolve_memo.get(stage as usize)
            && memo.is_compare == is_compare
            && memo.state == *sampler_state
        {
            return memo.handle;
        }
        let mut snapshot = sampler_state::snapshot_from_state(sampler_state, is_compare);
        if force_point {
            // Raw depth fetch: Apple GPUs cannot filter Depth32Float, and a
            // linear sample of it returns garbage rather than depth. D3D9
            // games set LINEAR on everything, so the slot's sampler is forced
            // to point; the shader reads exact stored depths, which is what
            // position reconstruction wants anyway. Comparison samplers stay
            // as configured: linear there is hardware PCF, which Apple GPUs
            // do support.
            snapshot.min_filter = mtld3d_types::D3DTEXF_POINT;
            snapshot.mag_filter = mtld3d_types::D3DTEXF_POINT;
            snapshot.mip_filter = mtld3d_types::D3DTEXF_NONE;
            snapshot.max_anisotropy = 1;
        }
        let key = sampler_state::key_from_snapshot(&snapshot);
        let lodbias_raw = sampler_state[D3DSAMP_MIPMAPLODBIAS as usize];
        let dedup = (u64::from(stage) << 56) ^ (u64::from(lodbias_raw) << 24) ^ key.raw();
        mtld3d_shared::log_once_trace_by!(
            target: SAMPLER_TRACE_TARGET, key: dedup,
            "sampler diag stage={stage} key={key:#x} cmp={cmp} srgb={srgb} min={min} mag={mag} mip={mip} addrU={au} addrV={av} addrW={aw} aniso={aniso} maxmip={mml} lodbias=0x{lb:08x}({lf:.3})",
            cmp = u8::from(is_compare),
            srgb = u8::from(snapshot.flags.contains(sampler_state::SamplerFlags::SRGB_TEXTURE)),
            min = snapshot.min_filter,
            mag = snapshot.mag_filter,
            mip = snapshot.mip_filter,
            au = snapshot.address_u, av = snapshot.address_v, aw = snapshot.address_w,
            aniso = snapshot.max_anisotropy,
            mml = snapshot.max_mip_level,
            lb = lodbias_raw, lf = f32::from_bits(lodbias_raw),
        );
        if let Some(&handle) = self.sampler_cache.get(&key) {
            if !force_point {
                self.memoize_sampler_resolve(stage, sampler_state, is_compare, handle.raw());
            }
            return handle.raw();
        }
        let mut params = sampler_state::params_from_snapshot(&snapshot, key, self.device_handle);
        let status = unix_call(&mut params);
        let sampler = params.sampler_handle;
        if status != 0 || sampler.is_null() {
            error!(target: LOG_TARGET, "encoder: CreateSamplerState failed");
            return 0;
        }
        self.sampler_cache.insert(key, sampler);
        if !force_point {
            self.memoize_sampler_resolve(stage, sampler_state, is_compare, sampler.raw());
        }
        sampler.raw()
    }

    /// Stash a successful sampler resolve in the per-stage memo.
    ///
    /// Failed creates (handle 0) never land here, so they keep retrying.
    fn memoize_sampler_resolve(
        &mut self,
        stage: u32,
        sampler_state: &[u32; SAMPLER_STATE_COUNT],
        is_compare: bool,
        handle: u64,
    ) {
        if let Some(slot) = self.sampler_resolve_memo.get_mut(stage as usize) {
            *slot = Some(SamplerResolveMemo {
                state: *sampler_state,
                is_compare,
                handle,
            });
        }
    }

    /// Drain every cache + retention queue, releasing the MTL handles via bulk-destroy thunks.
    ///
    /// Called from the encoder thread on `EncoderMessage::Shutdown` *before*
    /// the loop exits — the `Arc<AtomicU64>` backing `coherent_seq` lives
    /// inside `DeviceInner` and is freed by the API thread once
    /// `device_inner.shutdown()` joins our thread; we must finish reading it
    /// before returning.
    fn shutdown_cleanup(&mut self) {
        mtld3d_shared::crumb!("phase:SdEnter");
        // 1. Collect live-cache handles into local Vecs. Pure-Rust walks
        //    overlap the GPU's final command buffers finishing up.
        let mut buffers: Vec<u64> = Vec::new();
        let mut textures: Vec<u64> = Vec::new();

        for state in self.buffer_cache.values() {
            if !state.mtl_buffer.is_null() {
                buffers.push(state.mtl_buffer.raw());
            }
        }
        for state in self.texture_cache.values() {
            for slot in &state.mip_staging_buffers {
                if !slot.handle.is_null() {
                    buffers.push(slot.handle.raw());
                }
            }
            if !state.mtl_texture.is_null() {
                textures.push(state.mtl_texture.raw());
            }
            if !state.mtl_texture_srgb.is_null() {
                textures.push(state.mtl_texture_srgb.raw());
            }
        }

        let pipelines: Vec<u64> = self.pipeline_cache.values().map(|h| h.raw()).collect();
        let libraries: Vec<u64> = self
            .lib_cache
            .values()
            .filter_map(|h| (!h.library.is_null()).then_some(h.library.raw()))
            .collect();
        let functions: Vec<u64> = self
            .lib_cache
            .values()
            .filter_map(|h| (!h.func.is_null()).then_some(h.func.raw()))
            .collect();
        let samplers: Vec<u64> = self.sampler_cache.values().map(|h| h.raw()).collect();
        let depth_states: Vec<u64> = self.depth_stencil_cache.values().map(|h| h.raw()).collect();

        // 2. Drain retention + GPU-idle wait. Shared with reset_cleanup.
        //    The visibility-pool drain hands us `(PageBox, handle, seq)`
        //    triples — `held` outlives the bulk destroys below so
        //    `MTLBuffer`s never outlive their `bytesNoCopy` backings
        //    (owned `PageBox` and `Arc<PageBox>` keepalive both). Drained
        //    Texture-kind retention entries merge into the `textures`
        //    Vec collected from the live cache above.
        mtld3d_shared::crumb!("phase:SdDrain");
        let held = self.drain_retention_and_wait(&mut buffers, &mut textures);

        // 3. Bulk destroys for live caches. Pipelines reference functions,
        //    which reference libraries — destroy leaf-first.
        mtld3d_shared::crumb!("phase:SdBufs");
        destroy_resources_bulk(DestroyKind::Buffer, &buffers);
        mtld3d_shared::crumb!("phase:SdTexs");
        destroy_resources_bulk(DestroyKind::Texture, &textures);
        mtld3d_shared::crumb!("phase:SdPipes");
        destroy_resources_bulk(DestroyKind::RenderPipeline, &pipelines);
        mtld3d_shared::crumb!("phase:SdFns");
        destroy_resources_bulk(DestroyKind::ShaderFunction, &functions);
        mtld3d_shared::crumb!("phase:SdLibs");
        destroy_resources_bulk(DestroyKind::ShaderLibrary, &libraries);
        mtld3d_shared::crumb!("phase:SdSamps");
        destroy_resources_bulk(DestroyKind::SamplerState, &samplers);
        mtld3d_shared::crumb!("phase:SdDStates");
        destroy_resources_bulk(DestroyKind::DepthStencilState, &depth_states);

        // 4. Drop held backings + clear blit retention NOW that all
        //    wrapping MTLBuffers are released. Order matters: the
        //    staging memory backs MTLBuffers via `bytesNoCopy`, so the
        //    wrapper must die first or the buffer holds a dangling
        //    pointer.
        mtld3d_shared::crumb!("phase:SdBack");
        drop(held);
        self.pending_blit_retention.clear();
        self.current_blit_retention.clear();

        // 5. Clear the cache HashMaps so any stray frame message that
        //    races us (defensive; shouldn't happen) sees empty caches.
        //    Dropping the `texture_cache` HashMap also drops every
        //    surviving `MipStagingBuffer.keepalive` Arc, returning the
        //    pages to snmalloc (or to the OS if it was the last ref).
        self.buffer_cache.clear();
        self.texture_cache.clear();
        self.pipeline_cache.clear();
        self.lib_cache.clear();
        // Non-owning indices into the libraries destroyed above via `lib_cache`
        // — just drop the handle copies.
        self.ff_vs_libs.clear();
        self.prog_vs_libs.clear();
        self.ff_ps_libs.clear();
        self.prog_ps_libs.clear();
        self.sampler_cache.clear();
        self.depth_stencil_cache.clear();
        self.program_cache.clear();

        // 6. Close the disk shader cache writer; File's Drop flushes.
        self.cache_writer = None;
        mtld3d_shared::crumb!("phase:SdDone");
    }

    /// Drain retention queues, wait for GPU idle, leave live caches alone.
    ///
    /// Used by `EncoderMessage::Reset` (`device_reset` path) and shared with
    /// `shutdown_cleanup`.
    ///
    /// Reset replaces only the implicit backbuffer + depth/stencil — every
    /// game-created resource (textures, VBs, IBs, shaders) survives, so
    /// the encoder's caches that mirror them must survive too. Only the
    /// per-frame retention queues need draining: their `MTLBuffers` were
    /// already slated for release once the GPU finished, and Reset's GPU
    /// idle wait is exactly that signal.
    fn reset_cleanup(&mut self) {
        let mut buffers: Vec<u64> = Vec::new();
        let mut textures: Vec<u64> = Vec::new();
        let held = self.drain_retention_and_wait(&mut buffers, &mut textures);
        destroy_resources_bulk(DestroyKind::Buffer, &buffers);
        destroy_resources_bulk(DestroyKind::Texture, &textures);
        drop(held);
        self.pending_blit_retention.clear();
        self.current_blit_retention.clear();
    }

    /// Drain resource + visibility retention into the caller's `buffers` / `textures` Vecs.
    ///
    /// They merge with the live-cache handles the caller already collected.
    /// Returns the held backings (both `PageBox` and `Arc<PageBox>`
    /// variants), then `wait_for_gpu_idle`. Does NOT touch the
    /// `pending_blit_retention` / `current_blit_retention` Arcs (those must
    /// outlive the bulk destroy of the staging `MTLBuffers` that wrap them
    /// via `bytesNoCopy`). Caller drops the returned `HeldBackings` after
    /// `destroy_resources_bulk`.
    fn drain_retention_and_wait(
        &mut self,
        buffers: &mut Vec<u64>,
        textures: &mut Vec<u64>,
    ) -> HeldBackings {
        let mut held = HeldBackings::default();
        // Un-acknowledged uploads go the same way as retention: the GPU is
        // about to be idle, so there is nothing left to replay into. Their
        // bytes leave the shared retention total here, which is the only
        // place besides `settle_stage_uploads` that subtracts them.
        for entry in self.pending_stage_uploads.drain_all() {
            let retry = entry.into_payload();
            if !retry.transient.is_null() {
                buffers.push(retry.transient.raw());
            }
            self.perf.bump_vbib_retained_sub(retry.page_box.len());
            self.sub_retained_bytes(retry.page_box.len());
            held.pageboxes.push(retry.page_box);
        }
        // Texture jobs own only a clone of the texture's staging Arc; park
        // it with the other staging keepalives so it outlives the bulk
        // destroy of the `MTLBuffer`s that wrap those pages.
        for entry in self.pending_texture_uploads.drain_all() {
            held.staging_arcs.push(entry.into_payload().arc);
        }
        while let Some(entry) = self.pending_resource_retention.pop_front() {
            if entry.handle != 0 {
                match entry.kind {
                    DestroyKind::Buffer => buffers.push(entry.handle),
                    DestroyKind::Texture => {
                        textures.push(entry.handle);
                        self.retire_texture_handle(entry.handle);
                    }
                    other => {
                        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                            "drain_retention_and_wait: unexpected kind {other:?} \
                             — bulk-destroying as single-element call",
                        );
                        destroy_resources_bulk(other, &[entry.handle]);
                    }
                }
            }
            if let Some(pb) = entry.page_box {
                held.pageboxes.push(pb);
            }
            if let Some(arc) = entry.staging_arc {
                held.staging_arcs.push(arc);
            }
        }
        for vis_buf in self.visibility.drain_all_buffers() {
            let (page_box, handle, _seq) = vis_buf.into_parts();
            if !handle.is_null() {
                buffers.push(handle.raw());
            }
            held.pageboxes.push(page_box);
        }
        let fan = core::mem::replace(&mut self.fan_index_buffer, FanIndexBuffer::EMPTY);
        if !fan.handle.is_null() {
            buffers.push(fan.handle.raw());
        }
        if let Some(page_box) = fan.backing {
            held.pageboxes.push(page_box);
        }
        self.wait_for_gpu_idle();
        held
    }

    /// Block until `coherent_seq >= current_submit_seq`.
    ///
    /// Parks on the unix-side `WaitForGpuRetire` thunk, which calls Metal's
    /// `MTLCommandBuffer::waitUntilCompleted` on the registered cmdbuf for
    /// `current_submit_seq`. Skips immediately when the encoder hasn't yet
    /// been given a `coherent_seq` pointer or hasn't submitted any frame —
    /// both states arrive together in `begin_frame`.
    fn wait_for_gpu_idle(&self) {
        if self.coherent_seq_ptr == 0 || self.current_submit_seq == 0 {
            return;
        }
        let mut params = WaitForGpuRetireParams {
            target_seq: self.current_submit_seq,
            coherent_seq_ptr: self.coherent_seq_ptr,
            failed_submit_seq_ptr: self.failed_seq_ptr,
        };
        let _ = unix_call(&mut params);
    }
}

// ── FrameData — bundle sent from API thread to encoder thread ──
//
// Carries per-frame handles + the op list. Clear state no longer lives here
// — `Clear()` pushes an op that calls `FrameEncoder::clear_color` /
// `clear_depth` directly, which means a mid-frame clear can break the
// current pass and seed the next pass's load action.

/// Warmup entry for a `MTLBuffer` wrap pre-registered by the API thread.
///
/// Registered at `CreateVertexBuffer` / `CreateIndexBuffer` time. The encoder
/// thread drains the queue into one batched `CreateBuffersBatch` thunk at the
/// head of `run_frame`, before the op loop — so subsequent draw closures
/// hit the `buffer_cache` instead of cache-missing on first bind.
#[derive(Clone, Copy)]
pub struct VbibWarmupEntry {
    pub buffer_id: BufferId,
    pub backing_ptr: u64,
    pub backing_len: u64,
    /// The backing allocation's identity (see `PageBox::generation`).
    pub backing_generation: u64,
    /// Decides the create path.
    ///
    /// `Direct` → one `bytesNoCopy` wrapper (today's zero-copy bind);
    /// `Staged` → a `StorageModePrivate` device buffer (the draw-bind target
    /// written by staging-upload blits).
    pub map_mode: BufferMapMode,
}

/// Warmup entry for a texture mip's staging `MTLBuffer` wrap.
///
/// Pushed at `CreateTexture` time alongside the `TextureInfo` warmup; drained
/// after `drain_texture_warmups` so the `texture_cache` entry exists. The
/// resulting handle lands in `TextureGpuState::mip_staging_buffers[level]`,
/// so the first `UnlockRect`-driven upload's `get_or_create_staging_buffer`
/// hits the cache instead of cache-missing.
///
/// Skipped for RT / depth / expansion-path textures — their staging
/// backing is never wrapped as a cached `MTLBuffer` here (RT/depth have no
/// upload staging path; the expansion path builds its own transient staging
/// buffer per upload before blitting).
///
/// `keepalive` holds the PE-side `Arc<PageBox>` so the staging
/// allocation outlives the API thread's `texture_release` (which drops
/// the original Arc from `TextureInner.staging` synchronously) and is
/// still valid when `drain_staging_warmups` wraps it via
/// `newBufferWithBytesNoCopy:`. After drain the clone moves into
/// `MipStagingBuffer.keepalive`.
pub struct StagingWarmupEntry {
    pub texture_id: TextureId,
    pub level: u32,
    pub backing_ptr: u64,
    pub backing_len: u64,
    pub keepalive: Arc<PageBox>,
}

bitflags::bitflags! {
    /// Per-frame boolean state on [`FrameData`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FrameDataFlags: u8 {
        /// `depth_texture` is a combined depth+stencil format.
        ///
        /// Forwarded to `PassState::reset_frame` so clear-quad pipelines
        /// match the pass.
        const DEPTH_HAS_STENCIL = 1 << 0;
        /// This frame is a mid-frame checkpoint rather than a user `Present`.
        ///
        /// The triggers are `LockRect` on the backbuffer and
        /// `GetRenderTargetData`. `submit()` honours the flag by zeroing the
        /// present-layer fields in `SubmitFrameParams` so the Metal side skips
        /// `nextDrawable` and the backbuffer→drawable blit. The command buffer
        /// still commits, so in-order queue execution makes the backbuffer
        /// texture safe to read from the subsequent readback-blit command
        /// buffer.
        const NO_PRESENT = 1 << 1;
    }
}

pub struct FrameData {
    ops: Vec<Op>,
    /// Texture creates pushed by the API thread at `IDirect3DDevice9::CreateTexture` time.
    ///
    /// Drained into one batched `CreateTexturesBatch` thunk at the head of
    /// `run_frame` so the `MTLTexture` exists before any draw closure
    /// references it.
    pending_texture_warmups: Vec<TextureInfo>,
    /// VB/IB wraps pushed at `CreateVertexBuffer` / `CreateIndexBuffer` time.
    ///
    /// Drained alongside the texture warmups via one batched
    /// `CreateBuffersBatch` thunk.
    pending_buffer_warmups: Vec<VbibWarmupEntry>,
    /// Per-mip staging `MTLBuffer` wraps pushed alongside texture warmups.
    ///
    /// One per mip for non-RT / non-depth / non-expansion textures. Drained
    /// after `drain_texture_warmups` so the `texture_cache` slot already
    /// exists.
    pending_staging_warmups: Vec<StagingWarmupEntry>,
    device_handle: MetalHandle<MTLDeviceKind>,
    queue_handle: MetalHandle<MTLCommandQueueKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of the back buffer; see `FrameInit`.
    backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of `backbuffer_handle`, NULL when there is none.
    backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of `backbuffer_msaa_handle`; see `FrameInit`.
    backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Sample count the frame's back buffer and default depth surface carry.
    backbuffer_sample_count: u8,
    layer_handle: MetalHandle<CAMetalLayerKind>,
    /// `NSView*` the layer was attached to.
    ///
    /// Forwarded to `SubmitFrameParams.present_view` so the unix side can
    /// follow the screen the window is on: its *dynamic* EDR headroom each
    /// present, and its EDR capability across a display change. Which layer
    /// configuration that resolves to is decided unix-side.
    view_handle: MetalHandle<NSViewKind>,
    /// Logical back-buffer width, the resolution D3D9 reports.
    ///
    /// `render_scale` of this is the rasterized extent.
    backbuffer_width: u32,
    /// Logical back-buffer height. See `backbuffer_width`.
    backbuffer_height: u32,
    /// Fraction of the logical resolution the back buffer is rasterized at.
    ///
    /// Handed to `PassState::reset_frame`, which reconciles the two spaces.
    render_scale: RenderScale,
    /// Metal pixel format of the backbuffer.
    ///
    /// Always `Bgra8Unorm` in mtld3d today — `unix/unix/src/metal/texture.rs`
    /// always creates the backbuffer as `BGRA8Unorm`. Seeded into
    /// `PassState::reset_frame` so the initial pass's pipeline cache key has
    /// the right format before any `SetRenderTarget`.
    backbuffer_format: PixelFormat,
    depth_texture: MetalHandle<MTLTextureKind>,
    /// Per-frame boolean state (`DEPTH_HAS_STENCIL` / `NO_PRESENT`).
    ///
    /// See [`FrameDataFlags`].
    flags: FrameDataFlags,
    /// All per-frame telemetry drained from `ApiPerfState` by `DeviceInner::present`.
    ///
    /// Plus the `present_block_cycles` field set on the *next* frame right
    /// after `send_frame` returns. See `mtld3d_core::perf::FramePerfPayload`.
    perf: FramePerfPayload,
    /// Per-frame VB/IB backings + submit seqs queued for GPU-retire-gated destruction.
    ///
    /// Destroyed on the encoder thread; consumed in `begin_frame`.
    vbib_retentions: Vec<PendingVbibRetention>,
    /// Monotonic submit seq stamped by `DeviceInner::present` before the encoder handoff.
    ///
    /// Carried into `SubmitFrameParams` so the unix `addCompletedHandler`
    /// knows which seq to broadcast.
    submit_seq: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` coherent-seq.
    ///
    /// Stays valid for the device's lifetime (Arc is dropped after all frames
    /// drain). The completion block stores the retired seq via this
    /// pointer with Release ordering.
    coherent_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` upload-coherent-seq.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. Forwarded verbatim
    /// into `SubmitFrameParams::upload_coherent_seq_ptr`; non-zero tells
    /// the unix side to split the frame-leading blits into their own,
    /// earlier-retiring command buffer. 0 only before the frame is
    /// stamped (`FrameData::new` default); every submitted frame carries
    /// the real pointer.
    upload_coherent_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` failed-submit seq.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. Forwarded verbatim
    /// into `SubmitFrameParams::failed_submit_seq_ptr`, which both
    /// completion handlers `fetch_max` when their command buffer aborts.
    /// 0 only before the frame is stamped.
    failed_submit_seq_ptr: u64,
    /// Raw pointer to the device's `Arc<AtomicU64>` VB/IB retained-bytes total.
    ///
    /// Same lifetime guarantee as `coherent_seq_ptr`. The encoder
    /// `fetch_add`/`fetch_sub`s it as `PageBox`es enter/leave retention.
    retained_bytes_ptr: u64,
    /// `Some(v)` if `IDirect3DDevice9::Reset` changed `PresentationInterval` since the last frame.
    ///
    /// The encoder applies it via `SetDisplaySyncEnabledParams` at the top of
    /// `run_frame` so the new vsync state takes effect on this frame's
    /// `nextDrawable`, matching the spec's "next Present" timing rather than
    /// the previous behaviour of mutating the layer property synchronously
    /// from the API thread mid-frame.
    apply_display_sync_enabled: Option<bool>,
    /// API-thread bump arena.
    ///
    /// Used by `snapshot_shared` to allocate per-draw VS/PS constants +
    /// alpha-ref + fog-color bytes without per-draw `Vec::to_vec()` heap
    /// traffic — pointers handed across the channel via `ScratchSlice` stay
    /// valid until this `FrameData` is dropped after the encoder finishes
    /// draining `ops`. Separate from `FrameEncoder::scratch` (which the
    /// encoder thread uses for clear-pass constants etc.) so no two threads
    /// ever write the same arena.
    scratch: ScratchArena,
    /// Running per-frame total of bytes `Vec::push` memcpys when `ops` doubles its capacity.
    ///
    /// The API→encoder bridge counterpart to
    /// `PassState::cmd_vec_realloc_bytes`. `push_op` / `push_op_inline`
    /// check `len == capacity` before push and add
    /// `capacity × size_of::<Op>()` here on equality (= the bytes the
    /// imminent realloc memcpys). `peak_ops_count` in `DeviceInner` reserves
    /// the new frame's `ops` at the running peak, so steady-state should land
    /// at 0 — non-zero signals a new variant or workload that perturbed the
    /// peak.
    op_vec_realloc_bytes: u64,
}

/// The four GPU-fencing values `DeviceInner::stamp_and_swap` puts on a frame.
///
/// One bag rather than four positional `u64`s, because three of them are
/// raw pointers to device-owned atomics and swapping two at a call site
/// would compile silently. Every pointer is an `Arc<AtomicU64>` address
/// that stays valid for the device's lifetime.
pub struct SubmitFence {
    /// Monotonic seq of the frame being handed to the encoder.
    pub submit_seq: u64,
    /// Draw-command-buffer retirement counter.
    pub coherent_seq_ptr: u64,
    /// Upload-command-buffer retirement counter.
    pub upload_coherent_seq_ptr: u64,
    /// Highest seq whose command buffer the GPU aborted.
    pub failed_submit_seq_ptr: u64,
}

/// Parameter bag for `FrameData::new`.
///
/// Grouped so the constructor signature stays under clippy's
/// `too_many_arguments` threshold — pattern borrowed from `DeviceCreateInfo` /
/// `TextureCreateInfo`.
pub struct FrameInit {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub queue_handle: MetalHandle<MTLCommandQueueKind>,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of the back buffer, attached under `D3DRS_SRGBWRITEENABLE`.
    pub backbuffer_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Multisampled companion of the back buffer, NULL when it is single-sampled.
    pub backbuffer_msaa_handle: MetalHandle<MTLTextureKind>,
    /// sRGB twin view of that companion, attached under `D3DRS_SRGBWRITEENABLE`.
    pub backbuffer_msaa_srgb_handle: MetalHandle<MTLTextureKind>,
    /// Sample count of the back buffer and the frame's default depth surface.
    pub backbuffer_sample_count: u8,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    pub view_handle: MetalHandle<NSViewKind>,
    /// The frame's **logical** back-buffer width, the one D3D9 reports.
    ///
    /// `render_scale` converts it to the rasterized extent; keeping the pair
    /// separate means every consumer states which space it wants.
    pub backbuffer_width: u32,
    /// The frame's **logical** back-buffer height. See `backbuffer_width`.
    pub backbuffer_height: u32,
    pub backbuffer_format: PixelFormat,
    /// Fraction of the logical resolution the back buffer is rasterized at.
    ///
    /// Forwarded to `PassState::reset_frame`, which is the single place the
    /// logical and render coordinate spaces are reconciled.
    pub render_scale: RenderScale,
    pub depth_texture: MetalHandle<MTLTextureKind>,
    /// `true` when the frame's default depth attachment is a combined depth+stencil format.
    ///
    /// That format is `Depth32Float_Stencil8`. Drives the clear-quad
    /// pipelines' depth/stencil attachment formats so they match the pass.
    pub depth_has_stencil: bool,
    /// `Some(v)` from `device_reset` to defer a `PresentationInterval` change.
    ///
    /// The change lands on this frame's first `nextDrawable`. `None` for
    /// normal frames.
    pub apply_display_sync_enabled: Option<bool>,
}

impl FrameData {
    pub const fn new(init: &FrameInit) -> Self {
        Self {
            ops: Vec::new(),
            pending_texture_warmups: Vec::new(),
            pending_buffer_warmups: Vec::new(),
            pending_staging_warmups: Vec::new(),
            device_handle: init.device_handle,
            queue_handle: init.queue_handle,
            backbuffer_handle: init.backbuffer_handle,
            backbuffer_srgb_handle: init.backbuffer_srgb_handle,
            backbuffer_msaa_handle: init.backbuffer_msaa_handle,
            backbuffer_msaa_srgb_handle: init.backbuffer_msaa_srgb_handle,
            backbuffer_sample_count: init.backbuffer_sample_count,
            layer_handle: init.layer_handle,
            view_handle: init.view_handle,
            backbuffer_width: init.backbuffer_width,
            backbuffer_height: init.backbuffer_height,
            backbuffer_format: init.backbuffer_format,
            render_scale: init.render_scale,
            depth_texture: init.depth_texture,
            flags: if init.depth_has_stencil {
                FrameDataFlags::DEPTH_HAS_STENCIL
            } else {
                FrameDataFlags::empty()
            },
            perf: FramePerfPayload::new(),
            vbib_retentions: Vec::new(),
            submit_seq: 0,
            coherent_seq_ptr: 0,
            upload_coherent_seq_ptr: 0,
            failed_submit_seq_ptr: 0,
            retained_bytes_ptr: 0,
            apply_display_sync_enabled: init.apply_display_sync_enabled,
            scratch: ScratchArena::new(),
            op_vec_realloc_bytes: 0,
        }
    }

    /// Mutable handle to the API-thread bump arena.
    ///
    /// Called by `snapshot_shared` to copy VS/PS constants + alpha-ref +
    /// fog-color bytes once per draw without `Vec::to_vec()`. The returned
    /// arena is cleared on `FrameData` drop (i.e. after the encoder finishes
    /// the frame), so pointers stay valid for the entire op-replay window.
    pub const fn scratch_mut(&mut self) -> &mut ScratchArena {
        &mut self.scratch
    }

    /// Number of ops queued in this frame so far.
    ///
    /// `stamp_and_swap` reads this on the outgoing frame to pre-size the
    /// incoming frame's ops Vec, eliminating the per-frame Vec doubling
    /// burden (which was statistically landing on Draw `push_op` calls after
    /// `Set*ConstRange` ops bumped the per-frame total by ~50%).
    pub const fn ops_len(&self) -> usize {
        self.ops.len()
    }

    /// Pre-reserve `count` elements of capacity in the ops Vec.
    ///
    /// Called from `stamp_and_swap` with the previous frame's
    /// `ops_len()` so the new frame fills without any realloc in the
    /// common case where frame-to-frame op count is stable.
    pub fn reserve_ops(&mut self, count: usize) {
        self.ops.reserve(count);
    }

    /// Byte layout of the frame's back-buffer texture.
    ///
    /// Read by `GetFrontBufferData` to check its destination against the image
    /// it copies: the back buffer is created `Bgra8Unorm` whatever format the
    /// swap chain was asked for, so this is the layout, not the declared
    /// `D3DFMT_*`.
    pub const fn backbuffer_format(&self) -> PixelFormat {
        self.backbuffer_format
    }

    pub const fn perf(&self) -> &FramePerfPayload {
        &self.perf
    }

    pub const fn perf_mut(&mut self) -> &mut FramePerfPayload {
        &mut self.perf
    }

    pub const fn set_no_present(&mut self, no_present: bool) {
        // const fn: bitflags `.set()` isn't const, so union/difference (which
        // are) toggle the bit.
        self.flags = if no_present {
            self.flags.union(FrameDataFlags::NO_PRESENT)
        } else {
            self.flags.difference(FrameDataFlags::NO_PRESENT)
        };
    }

    pub const fn set_submit_fence(&mut self, fence: &SubmitFence) {
        self.submit_seq = fence.submit_seq;
        self.coherent_seq_ptr = fence.coherent_seq_ptr;
        self.upload_coherent_seq_ptr = fence.upload_coherent_seq_ptr;
        self.failed_submit_seq_ptr = fence.failed_submit_seq_ptr;
    }

    pub const fn set_retained_bytes_ptr(&mut self, ptr: u64) {
        self.retained_bytes_ptr = ptr;
    }

    pub fn push_op(&mut self, op: Box<dyn FnOnce(&mut FrameEncoder) + Send>) {
        self.account_op_vec_realloc();
        self.ops.push(Op::Closure(op));
    }

    /// Push an `Op` variant directly.
    ///
    /// Used by the hot draw path (`emit_snapshot_deltas` + `Op::Draw`) so it
    /// can emit inline state-delta + draw variants without the per-op
    /// `Box<dyn FnOnce>` allocation `push_op` adds for closure-shaped work.
    pub fn push_op_inline(&mut self, op: Op) {
        self.account_op_vec_realloc();
        self.ops.push(op);
    }

    /// Add the old capacity's bytes to the per-frame counter before `Vec::push` reallocs.
    ///
    /// The imminent push is the one that trips `Vec::push`'s
    /// double-and-memcpy. Hot-path: a single `len == capacity`
    /// compare when no realloc fires. Mirrors the `emit_command`
    /// pattern in `PassState`.
    #[inline]
    const fn account_op_vec_realloc(&mut self) {
        if self.ops.len() == self.ops.capacity() {
            let bytes = (self.ops.capacity() as u64).saturating_mul(size_of::<Op>() as u64);
            self.op_vec_realloc_bytes = self.op_vec_realloc_bytes.saturating_add(bytes);
        }
    }

    /// Resident `Vec<Op>` capacity in bytes.
    ///
    /// Read by `stamp_and_swap` to seed the outgoing frame's
    /// `FramePerfPayload` so the `op_vec size` row in the per-frame allocator
    /// footprint reflects steady-state footprint paired with the realloc
    /// churn.
    pub const fn op_vec_capacity_bytes(&self) -> u64 {
        (self.ops.capacity() as u64).saturating_mul(size_of::<Op>() as u64)
    }

    /// Drain the per-frame `Vec<Op>` realloc-byte counter into the caller and zero it.
    ///
    /// Called once per frame from `stamp_and_swap` so the outgoing frame
    /// ships its realloc total to the encoder via `FramePerfPayload`.
    pub const fn take_op_vec_realloc_bytes(&mut self) -> u64 {
        core::mem::replace(&mut self.op_vec_realloc_bytes, 0)
    }

    /// Queue a texture for eager `MTLTexture` creation at the head of the next `run_frame`.
    ///
    /// Called from `IDirect3DDevice9::CreateTexture` on the API thread.
    pub fn push_texture_warmup(&mut self, info: TextureInfo) {
        self.pending_texture_warmups.push(info);
    }

    /// Queue a VB/IB for eager `MTLBuffer` wrap at the head of the next `run_frame`.
    ///
    /// Called from `CreateVertexBuffer` / `CreateIndexBuffer` on the API
    /// thread.
    pub fn push_buffer_warmup(&mut self, entry: VbibWarmupEntry) {
        self.pending_buffer_warmups.push(entry);
    }

    /// Queue a texture-staging `MTLBuffer` wrap.
    ///
    /// Called per mip from `IDirect3DDevice9::CreateTexture` on the API
    /// thread for textures that go through the blit-upload path.
    pub fn push_staging_warmup(&mut self, entry: StagingWarmupEntry) {
        self.pending_staging_warmups.push(entry);
    }

    pub fn set_vbib_retentions(&mut self, retentions: Vec<PendingVbibRetention>) {
        self.vbib_retentions = retentions;
    }
}

// ── EncoderThread ──

pub struct EncoderThread {
    sender: mpsc::SyncSender<EncoderMessage>,
    prewarm_tx: mpsc::SyncSender<PrewarmPayload>,
    handle: Option<thread::JoinHandle<()>>,
    /// The device capabilities the encoder was spawned with.
    ///
    /// Kept here so API-thread callers that have to predict an encoder
    /// decision (which upload path a mip takes, and therefore which command
    /// buffer reads its staging) read the same values the encoder does.
    gpu_caps: GpuCaps,
}

/// Pre-warm completion payload.
///
/// Carried on a dedicated one-shot channel rather than wrapped in
/// `EncoderMessage`, so the encoder thread can block specifically on
/// the prewarm result *before* touching the `Frame` queue. Without this
/// split the API thread could push a `Frame` into the (cap = 1)
/// `EncoderMessage` channel ahead of the prewarm's completion message,
/// causing the encoder to compile a shader from scratch that the
/// prewarm is concurrently compiling from disk.
struct PrewarmPayload {
    entries: Vec<(u64, StageLibHandles)>,
    writes_disabled: bool,
}

/// One-shot sender used by the shader pre-warm thread.
///
/// Wraps the dedicated `PrewarmPayload` channel so callers outside this
/// module never see the encoder-private type.
pub struct PrewarmSender(mpsc::SyncSender<PrewarmPayload>);

impl PrewarmSender {
    /// Normal completion.
    ///
    /// Ship pre-warmed handles (empty vec for a cold start) and let the
    /// encoder open the cache for append.
    pub fn send(self, entries: Vec<(u64, StageLibHandles)>) {
        let _ = self.0.send(PrewarmPayload {
            entries,
            writes_disabled: false,
        });
    }

    /// The cache file is unusable for this session.
    ///
    /// Prewarm couldn't read it but the file exists, so the encoder must
    /// not append (it would corrupt the existing content past the foreign
    /// bytes). `cache_ready` still flips so the encoder progresses out of
    /// its "prewarm not done" gate; `cache_disabled` latches so writes
    /// stay off for the rest of the session.
    pub fn send_disabled(self) {
        let _ = self.0.send(PrewarmPayload {
            entries: Vec::new(),
            writes_disabled: true,
        });
    }
}

impl EncoderThread {
    pub fn spawn(gpu_caps: GpuCaps) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<EncoderMessage>(1);
        let (prewarm_tx, prewarm_rx) = mpsc::sync_channel::<PrewarmPayload>(1);
        let handle = thread::Builder::new()
            .name("mtld3d-encoder".into())
            .spawn(move || encoder_thread_main(&receiver, &prewarm_rx, gpu_caps))
            .expect("mtld3d: failed to spawn encoder thread");
        Self {
            sender,
            prewarm_tx,
            handle: Some(handle),
            gpu_caps,
        }
    }

    /// The device capabilities this encoder translates against.
    #[must_use]
    pub const fn gpu_caps(&self) -> GpuCaps {
        self.gpu_caps
    }

    pub fn send_frame(&self, frame: FrameData) {
        let _ = self.sender.send(EncoderMessage::Frame(Box::new(frame)));
    }

    /// Submit the passed frame synchronously.
    ///
    /// The encoder thread runs ops → submit → completion, and this call
    /// blocks until the `SubmitFrame` thunk has returned (i.e. the command
    /// buffer has committed). Used by `LockRect` on the backbuffer and
    /// `GetRenderTargetData` — callers follow up with a readback-blit that
    /// reads the freshly submitted backbuffer texture, relying on Metal's
    /// in-order queue execution to order the readback after this
    /// submission.
    pub fn mid_frame_submit(&self, frame: FrameData) {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let _ = self.sender.send(EncoderMessage::MidFrameSubmit {
            frame: Box::new(frame),
            done: done_tx,
        });
        let _ = done_rx.recv();
    }

    /// Heavy retention-cap tier.
    ///
    /// Like `mid_frame_submit`, but the encoder also waits for GPU
    /// completion of the submitted seq and drains retention before
    /// signalling — when this returns the global allocator has the freed
    /// bytes back. Cost: ~1-2 ms of GPU completion + drain. Used only when
    /// `drain_retired_now` already failed to get retention under the cap.
    pub fn mid_frame_submit_for_retention(&self, frame: FrameData) {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let _ = self
            .sender
            .send(EncoderMessage::MidFrameSubmitForRetention {
                frame: Box::new(frame),
                done: done_tx,
            });
        let _ = done_rx.recv();
    }

    /// Cheap retention-cap tier.
    ///
    /// Encoder runs only `drain_retired_resource_retention`; no submit,
    /// no GPU wait. Frees retention items whose seq has already retired
    /// but haven't been drained because the encoder hasn't hit
    /// `begin_frame` since their seq retired. Cost: one encoder
    /// round-trip (~tens of µs).
    pub fn drain_retired_now(&self) {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let _ = self.sender.send(EncoderMessage::DrainRetiredNow(done_tx));
        let _ = done_rx.recv();
    }

    /// Drive the encoder thread to finalize visibility queries up to `target_seq`.
    ///
    /// The encoder waits (via the `WaitForGpuRetire` thunk → Metal
    /// `waitUntilCompleted`) only when `coherent_seq < target_seq`;
    /// otherwise it just runs `intake_visibility` and returns. Used by
    /// `IDirect3DQuery9::GetData(D3DGETDATA_FLUSH)`. `target_seq == 0`
    /// skips the wait (END closure not yet processed). Routing through
    /// the encoder is required so channel order guarantees the cmdbuf
    /// containing END is already submitted on the unix side by the time
    /// the wait fires.
    pub fn intake_visibility_for(&self, target_seq: u64) {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let _ = self.sender.send(EncoderMessage::IntakeVisibilityFor {
            target_seq,
            done: done_tx,
        });
        let _ = done_rx.recv();
    }

    /// Detached sender used by the shader pre-warm thread to deliver its completion payload.
    ///
    /// The encoder thread blocks on this channel before processing any
    /// `EncoderMessage`, so the prewarm always populates `lib_cache` first
    /// and live miss-compiles never race duplicate disk-cached entries.
    /// Empty payload is the "cold launch, file is fresh, you may start
    /// writing" signal that flips `cache_ready`.
    pub fn prewarm_sender(&self) -> PrewarmSender {
        PrewarmSender(self.prewarm_tx.clone())
    }

    pub fn shutdown(&mut self) {
        let _ = self.sender.send(EncoderMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            // Don't call `handle.join()`. On long sessions Wine reports
            // `STATUS_INVALID_HANDLE` for the encoder thread's Win32
            // handle (server/thread.c:1141 `wait_on_handles` ->
            // `get_handle_obj` NULL -> "os error 6"); std's
            // `JoinHandle::join` panics on `WAIT_FAILED`, and
            // `panic = "abort"` makes `catch_unwind` a no-op. Mirror
            // `shader_prewarm::cancel_and_join`: poll the std `Packet`
            // strong count (handle-independent userspace atomic) and
            // drop the handle without touching `WaitForSingleObject`.
            while !handle.is_finished() {
                thread::sleep(Duration::from_millis(1));
            }
            drop(handle);
        }
    }

    /// Drive `FrameEncoder::reset_cleanup` on the encoder thread and block until it acknowledges.
    ///
    /// Used by `device_reset` between destroying the old backbuffer/depth
    /// and creating their replacements: the cleanup waits for GPU idle so
    /// no in-flight command buffer references the textures we're about to
    /// destroy.
    pub fn reset(&self) {
        let (ack_tx, ack_rx) = mpsc::sync_channel(0);
        let _ = self.sender.send(EncoderMessage::Reset { ack: ack_tx });
        let _ = ack_rx.recv();
    }
}

enum EncoderMessage {
    Frame(Box<FrameData>),
    /// Synchronous variant of `Frame` for mid-frame checkpoints.
    ///
    /// The checkpoints are backbuffer `LockRect` and `GetRenderTargetData`.
    /// Identical processing; the encoder signals `done` after `submit`
    /// returns so the API-thread caller knows the command buffer is
    /// committed.
    MidFrameSubmit {
        frame: Box<FrameData>,
        done: mpsc::SyncSender<()>,
    },
    /// Heavy retention-cap tier.
    ///
    /// Run the frame, spin until our seq retires on the GPU (so
    /// `coherent_seq` covers our submission), then drain
    /// `pending_resource_retention` so freed bytes return to the global
    /// allocator before signalling done. Used by VB/IB `Lock`-rename when
    /// retained bytes are still over the cap after the cheap-tier
    /// `DrainRetiredNow`.
    MidFrameSubmitForRetention {
        frame: Box<FrameData>,
        done: mpsc::SyncSender<()>,
    },
    /// Cheap retention-cap tier.
    ///
    /// Just drain `pending_resource_retention` against the current
    /// `coherent_seq` — frees only items already retired. Useful when the
    /// encoder hasn't auto-drained between frames and parked retention is
    /// sitting freeable.
    DrainRetiredNow(mpsc::SyncSender<()>),
    /// Finalize visibility queries up to `target_seq`.
    ///
    /// Used by `Query9::GetData(D3DGETDATA_FLUSH)` to drain queries the
    /// app is polling as a GPU fence between frames. The encoder blocks
    /// (via `WaitForGpuRetire` thunk → Metal `waitUntilCompleted`) only
    /// when `coherent_seq < target_seq` — otherwise it just runs intake
    /// locally. `target_seq == 0` means the END closure has not been
    /// processed yet (game called `Issue(END)` but not Present); skip the
    /// wait, run intake, return. Channel order guarantees the cmdbuf
    /// carrying END is in the unix-side `PENDING_CMDBUFS` registry by the
    /// time this handler runs.
    IntakeVisibilityFor {
        target_seq: u64,
        done: mpsc::SyncSender<()>,
    },
    /// Drain retention + GPU-idle wait without breaking the message loop.
    ///
    /// `device_reset` follows up with `DestroyResourcesBulk` for the old
    /// backbuffer/depth and `CreateBackbuffer` for their replacements; the
    /// encoder keeps running afterward with the new handles arriving via
    /// the next `FrameData`.
    Reset {
        ack: mpsc::SyncSender<()>,
    },
    Shutdown,
}

/// Compile one stage's MSL into an `MTLLibrary` via the unix-side `CompileShaderLibrary` thunk.
///
/// `entry` must match the function name in the MSL source (the unix side
/// passes it to `newFunctionWithName:`). Returns `None` on UTF-8 / Metal
/// compile failure.
pub fn compile_stage_library(
    device_handle: MetalHandle<MTLDeviceKind>,
    stage_tag: StageTag,
    msl: &str,
    entry: &str,
) -> Option<StageLibHandles> {
    let mut params = CompileShaderLibraryParams {
        device_handle,
        msl_ptr: msl.as_ptr() as u64,
        msl_len: u32::try_from(msl.len()).expect("MSL source ≤ u32::MAX bytes"),
        stage_tag,
        entry_ptr: entry.as_ptr() as u64,
        entry_len: u32::try_from(entry.len()).expect("entry name ≤ u32::MAX bytes"),
        pad0: 0,
        library_handle: MetalHandle::NULL,
        fn_handle: MetalHandle::NULL,
    };
    let status = unix_call(&mut params);
    if status != 0 || params.library_handle.is_null() || params.fn_handle.is_null() {
        error!(target: LOG_TARGET, "encoder: CompileShaderLibrary failed (stage={stage_tag:?}, entry={entry})");
        return None;
    }
    Some(StageLibHandles {
        library: params.library_handle,
        func: params.fn_handle,
    })
}

/// Issue a single bulk-destroy thunk.
///
/// Caller hands us a slice of MTL handles of one `DestroyKind`; the
/// slice's backing must outlive this call (stack array, `Vec`, or
/// `Box<[u64]>` — anything stable). Empty slices short-circuit before
/// touching the FFI boundary.
fn destroy_resources_bulk(kind: DestroyKind, handles: &[u64]) {
    if handles.is_empty() {
        return;
    }
    let mut params = DestroyResourcesBulkParams {
        kind,
        pad0: 0,
        handles_ptr: handles.as_ptr() as u64,
        count: u32::try_from(handles.len()).expect("bulk-destroy count fits u32"),
        pad1: 0,
    };
    unix_call(&mut params);
}

// ── Encoder thread main loop ──

fn encoder_thread_main(
    receiver: &mpsc::Receiver<EncoderMessage>,
    prewarm_rx: &mpsc::Receiver<PrewarmPayload>,
    gpu_caps: GpuCaps,
) {
    if !gpu_caps.unified_memory {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "non-UMA Mac detected: hasUnifiedMemory=false, min_linear_texture_align={} \
             — repack_blit_source_padded path active for tiny mips",
            gpu_caps.min_linear_texture_align,
        );
    }
    let mut enc = FrameEncoder::new(gpu_caps);
    let mut frame_counter: u64 = 0;
    // Idempotent — also called from `lib.rs::init_logger` during
    // DllMain so the file is already mapped by the time we get here.
    mtld3d_shared::crumb::init();

    // Block on the pre-warm payload before draining any `EncoderMessage`.
    // While we're parked here the API thread can push one Frame into
    // the (cap = 1) channel and then stalls on its second `send_frame`,
    // so at most one Frame is buffered ahead of prewarm completion — and
    // we never process it until `lib_cache` is populated. Live
    // miss-compiles can therefore never duplicate a shader the prewarm
    // is about to deliver. The Err arm covers a prewarm-thread panic:
    // drop the warm cache, leave writes enabled, fall through so
    // subsequent `Shutdown` can still drain.
    if let Ok(payload) = prewarm_rx.recv() {
        enc.ingest_warm_cache(payload.entries, payload.writes_disabled);
    } else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: pre-warm channel closed without payload → starting cold"
        );
        enc.ingest_warm_cache(Vec::new(), false);
    }

    loop {
        match receiver.recv() {
            Ok(EncoderMessage::Frame(frame)) => {
                frame_counter += 1;
                mtld3d_shared::crumb!("phase:RecvFrame");
                let capture = crate::capture::take_request();
                if capture {
                    // Capture must bracket the actual `SubmitFrame` thunk,
                    // which `Async` runs on the submit thread. Drain the
                    // submit thread so prior frames are committed, then run
                    // this frame synchronously so Start/Stop wrap its
                    // inline execute on the encoder thread.
                    enc.drain_submit_thread();
                    let mut p = mtld3d_shared::StartGpuCaptureParams {
                        device_handle: enc.device_handle,
                    };
                    let _ = unix_call(&mut p);
                    run_frame(&mut enc, frame, frame_counter, SubmitMode::Sync);
                    let mut p = mtld3d_shared::StopGpuCaptureParams { pad0: 0 };
                    let _ = unix_call(&mut p);
                } else {
                    run_frame(&mut enc, frame, frame_counter, SubmitMode::Async);
                }
            }
            Ok(EncoderMessage::MidFrameSubmit { frame, done }) => {
                frame_counter += 1;
                mtld3d_shared::crumb!("phase:RecvMid");
                // Readback reads the backbuffer after this submit and relies
                // on Metal's in-order queue, so every prior async frame must
                // be committed first.
                enc.drain_submit_thread();
                run_frame(&mut enc, frame, frame_counter, SubmitMode::Sync);
                let _ = done.send(());
            }
            Ok(EncoderMessage::MidFrameSubmitForRetention { frame, done }) => {
                frame_counter += 1;
                mtld3d_shared::crumb!("phase:RecvMidRet");
                enc.drain_submit_thread();
                run_frame(&mut enc, frame, frame_counter, SubmitMode::Sync);
                // Wait for our just-submitted seq to retire on the GPU
                // so `coherent_seq` covers it; then drain so the freed
                // bytes are back in the global allocator before the
                // API thread allocates.
                enc.wait_for_gpu_idle();
                enc.drain_retired_resource_retention();
                enc.release_acknowledged_uploads();
                let _ = done.send(());
            }
            Ok(EncoderMessage::DrainRetiredNow(done)) => {
                mtld3d_shared::crumb!("phase:RecvDrain");
                // Cheap tier: no barrier needed. A resource retired at seq N
                // whose async submit is still in flight has seq > coherent
                // (coherent only advances on GPU completion of committed
                // work), so the seq-gated drain can't free it early.
                enc.drain_retired_resource_retention();
                enc.release_acknowledged_uploads();
                let _ = done.send(());
            }
            Ok(EncoderMessage::IntakeVisibilityFor { target_seq, done }) => {
                mtld3d_shared::crumb!("phase:RecvVisIn");
                mtld3d_shared::crumb!("vis:drainbeg", target_seq);
                // The cmdbuf carrying the END query must be committed (in the
                // unix-side PENDING_CMDBUFS registry) before WaitForGpuRetire,
                // so drain any in-flight async submits first.
                enc.drain_submit_thread();
                mtld3d_shared::crumb!("vis:drainend", target_seq);
                if target_seq != 0 && enc.coherent_seq_ptr != 0 {
                    // SAFETY: `coherent_seq_ptr` is a PE-heap
                    // `Arc<AtomicU64>` raw pointer kept alive by the
                    // device-side `Arc`; nonzero here means the
                    // encoder has been wired up.
                    let coh =
                        unsafe { SharedCounter::new(enc.coherent_seq_ptr) }.load(Ordering::Acquire);
                    if coh < target_seq {
                        let mut params = WaitForGpuRetireParams {
                            target_seq,
                            coherent_seq_ptr: enc.coherent_seq_ptr,
                            failed_submit_seq_ptr: enc.failed_seq_ptr,
                        };
                        mtld3d_shared::crumb!("vis:retirebeg", target_seq, coh);
                        let _ = unix_call(&mut params);
                        mtld3d_shared::crumb!("vis:retireend", target_seq);
                    }
                }
                enc.intake_visibility();
                let _ = done.send(());
            }
            Ok(EncoderMessage::Reset { ack }) => {
                mtld3d_shared::crumb!("phase:RecvReset");
                // Commit every in-flight async frame before the reset tears
                // down / recreates the backbuffer + depth they reference.
                enc.drain_submit_thread();
                enc.reset_cleanup();
                let _ = ack.send(());
            }
            Ok(EncoderMessage::Shutdown) | Err(_) => {
                mtld3d_shared::crumb!("phase:RecvSd");
                // Commit every in-flight async frame before destroying
                // resources the submit thread may still be reading. The
                // thread itself exits when `enc` (and its work-channel
                // sender) drops on return from this function.
                enc.drain_submit_thread();
                enc.shutdown_cleanup();
                break;
            }
        }
    }
}

/// Drain one frame's ops, submit the resulting command buffer, and log.
///
/// Shared between `EncoderMessage::Frame` (normal Present, `Async`) and the
/// rare readback / capture / reset paths (`Sync`, after a submit-thread
/// barrier).
fn run_frame(enc: &mut FrameEncoder, mut frame: Box<FrameData>, fc: u64, mode: SubmitMode) {
    if let Some(enabled) = frame.apply_display_sync_enabled.take()
        && !frame.layer_handle.is_null()
    {
        let mut params = SetDisplaySyncEnabledParams {
            layer_handle: frame.layer_handle,
            display_sync_enabled: u32::from(enabled),
            max_fps: crate::config::CONFIG.present_max_fps,
        };
        unix_call(&mut params);
    }
    mtld3d_shared::crumb!("phase:BfEnter");
    // Reset encoder-side state cache before draining ops: the pointer
    // it holds aliases into the *previous* frame's ScratchArena, which
    // is about to drop. The API thread always re-emits a fresh
    // SetCurrentSnapshot on the first draw of a new frame (driven by
    // SnapshotDirty::all() after arena rotation).
    enc.current_snapshot = None;
    enc.begin_frame(&frame);
    // Reclaim any payloads the submit thread finished so their command vecs
    // are back in the pool before this frame's op loop, and the returned
    // drawable-wait / status land in this frame's `Async` summary. Runs
    // after `begin_frame` so its `drawable_wait` reset doesn't clobber the
    // folded value.
    enc.drain_returned_payloads();
    // Eager Metal-object creation: drain API-thread-queued texture +
    // VB/IB warmups into batched thunks before any draw closure runs.
    // The closures snapshotted these resources at their D3D9-Create
    // time; running drain first means subsequent ops hit `texture_cache`
    // / `buffer_cache` and skip the per-resource lazy thunk crossing.
    enc.drain_texture_warmups(core::mem::take(&mut frame.pending_texture_warmups));
    enc.drain_buffer_warmups(core::mem::take(&mut frame.pending_buffer_warmups));
    // `Staged` VB/IB dirty-range uploads are no longer drained here — they
    // are inline `Op::StageUpload`s processed in op-stream order (so the
    // encoder can rename-at-overlap), handled in the op loop below.
    // `drain_buffer_warmups` above still creates the device buffers first.
    // Staging buffers slot into texture_cache entries created above —
    // must run after `drain_texture_warmups`.
    enc.drain_staging_warmups(core::mem::take(&mut frame.pending_staging_warmups));
    let ops = core::mem::take(&mut frame.ops);
    mtld3d_shared::crumb!("phase:OpLoop");
    {
        let _ops = mtld3d_core::perf::CycleSetTimer::start(enc.perf.op_cycles_ptr());
        for (idx, op) in ops.into_iter().enumerate() {
            let idx_u32 = u32::try_from(idx).expect("per-frame op count fits u32");
            mtld3d_shared::crumb!("enc_op", fc, u64::from(idx_u32));
            match op {
                Op::SetCurrentSnapshot(p) => enc.current_snapshot = Some(p),
                Op::SetVsConstRange {
                    start_row,
                    rows,
                    data,
                } => {
                    let _t = mtld3d_core::perf::CycleAddTimer::start(
                        enc.op_sub_cycles_ptr(OpSub::ConstRange),
                    );
                    enc.apply_vs_const_range(start_row, rows, data);
                }
                Op::SetPsConstRange {
                    start_row,
                    rows,
                    data,
                } => {
                    let _t = mtld3d_core::perf::CycleAddTimer::start(
                        enc.op_sub_cycles_ptr(OpSub::ConstRange),
                    );
                    enc.apply_ps_const_range(start_row, rows, data);
                }
                Op::SetFfVsConstRange {
                    start_row,
                    rows,
                    data,
                } => {
                    let _t = mtld3d_core::perf::CycleAddTimer::start(
                        enc.op_sub_cycles_ptr(OpSub::ConstRange),
                    );
                    enc.apply_ff_vs_const_range(start_row, rows, data);
                }
                Op::Draw(d) => draw::emit_draw(enc, d),
                Op::Closure(f) => f(enc),
                Op::StageUpload {
                    buffer_id,
                    page_box,
                    dst_offset,
                    size,
                } => enc.apply_stage_upload(buffer_id, page_box, dst_offset, size),
            }
        }
    }
    mtld3d_shared::crumb!("phase:OpLoopDn");
    enc.intake_vbib_retentions(&mut frame);
    mtld3d_shared::crumb!("phase:IntakeVbib");
    submit(enc, frame, mode);
    mtld3d_shared::crumb!("phase:Submit");
    mtld3d_shared::crumb!("phase:FrameDone");
}

/// Finalize the frame, issue the `SubmitFrame` thunk, and recycle the payload.
///
/// Split into three seams so the submit stage can run on its own thread:
///   * [`finalize_submit`] — encoder-thread work: close passes, apply the
///     load/store rules, build descriptors, and swap the per-frame buffers
///     out of the encoder into an owned [`FramePayload`] + `params`.
///   * [`execute_submit`] — the `unix_call(SubmitFrame)` itself; reads the
///     payload's pointers, returns it for recycling.
///   * [`reclaim_payload`] — drain the passes' command vecs back into the
///     pool and return the cleared buffers to `payload_pool`.
///
/// In `Async` mode `execute_submit` runs on the dedicated submit thread and
/// the payload is recycled when it returns; in `Sync` mode all three run
/// inline on the encoder thread.
fn submit(enc: &mut FrameEncoder, frame: Box<FrameData>, mode: SubmitMode) {
    match mode {
        SubmitMode::Async => submit_async(enc, frame),
        SubmitMode::Sync => submit_sync(enc, frame),
    }
}

/// Build a frame summary context from the frame's backbuffer attachment.
const fn frame_summary_ctx(frame: &FrameData) -> FrameSummaryContext {
    FrameSummaryContext {
        backbuffer_handle: frame.backbuffer_handle,
        depth_texture: frame.depth_texture,
        backbuffer_width: frame.backbuffer_width,
        backbuffer_height: frame.backbuffer_height,
    }
}

/// Async Present path.
///
/// Finalize the frame on the encoder thread, emit the per-frame summary
/// from the still-live payload (status / drawable-wait are the most recent
/// submit's — lagged ≤1 frame), then hand the payload to the submit thread
/// and return so the next frame's build overlaps the `SubmitFrame` thunk.
/// The `submit_cycles` timer captures only the encoder-side finalize (plus
/// any backpressure wait inside `acquire_clean_payload`); the unix
/// command-walk + present is no longer on this thread.
fn submit_async(enc: &mut FrameEncoder, frame: Box<FrameData>) {
    let (params, payload) = {
        let _submit = mtld3d_core::perf::CycleSetTimer::start(enc.perf.submit_cycles_ptr());
        finalize_submit(enc, &frame)
    };
    let status = enc.last_submit_status;
    let ctx = frame_summary_ctx(&frame);
    enc.log_perf_summary(&payload, &ctx, status);
    enc.maybe_emit_compile_summary();
    // `frame` rides along so its scratch (which several Commands point into)
    // outlives the deferred replay; the submit thread drops it afterwards.
    enc.dispatch_submit(SubmitPacket {
        params,
        payload,
        frame,
    });
}

/// Synchronous submit: run the `SubmitFrame` thunk inline and block until it commits.
///
/// Used after a `drain_submit_thread` barrier for the rare paths that need
/// the command buffer committed before they proceed. The `submit_cycles`
/// timer wraps finalize + execute so the per-frame summary (emitted after,
/// from the still-live payload) reads a settled value; the payload is
/// recycled only once the summary has read its passes / scratch.
fn submit_sync(enc: &mut FrameEncoder, frame: Box<FrameData>) {
    let (payload, status) = {
        let _submit = mtld3d_core::perf::CycleSetTimer::start(enc.perf.submit_cycles_ptr());
        let (params, payload) = finalize_submit(enc, &frame);
        let (payload, status, drawable_wait_ns) = execute_submit(params, payload);
        enc.perf
            .set_drawable_wait_cycles(ns_to_cycles(drawable_wait_ns));
        (payload, status)
    };
    enc.last_submit_status = status;
    if status != 0 {
        error!(
            target: LOG_TARGET,
            "encoder: SubmitFrame failed (status={status:#x}, passes={}, present_tex={:#x})",
            payload.descriptors.len(),
            frame.backbuffer_handle,
        );
    }
    let ctx = frame_summary_ctx(&frame);
    enc.log_perf_summary(&payload, &ctx, status);
    enc.maybe_emit_compile_summary();
    reclaim_payload(enc, payload);
    // `execute_submit` ran inline, so the replay is done reading
    // `frame`'s scratch; drop it (explicit for symmetry with the async path).
    drop(frame);
}

/// Encoder-thread half of submit.
///
/// Close passes, run the load/store rules, build the `PassDescriptor`s,
/// and detach the frame's read payload from the encoder. Returns the
/// `params` (with raw pointers aliasing into the payload) plus the owned
/// [`FramePayload`] that backs them.
fn finalize_submit(enc: &mut FrameEncoder, frame: &FrameData) -> (SubmitFrameParams, FramePayload) {
    // If the game called `Clear()` without any subsequent draw this frame
    // (or after the last draw), the pending clear still needs to
    // materialize so the RT actually gets cleared this frame. Then close
    // whatever is open.
    enc.pass_state.flush_pending_clears();
    enc.end_current_pass("submit");

    // A readback flush is not a frame end: the frame continues and any colour
    // or depth target may still be read back or drawn into, so the last-use
    // rules (D colour, B depth/stencil) are suppressed. Remember it so the
    // next `begin_frame` keeps the seen-rt sets for the continuation's Rule A.
    let no_present = frame.flags.contains(FrameDataFlags::NO_PRESENT);
    enc.prev_submit_no_present = no_present;
    apply_pass_rules(enc, no_present);
    log_cascade_frame_summary(enc);

    // StretchRect blits queued after the last draw of the frame have no
    // follow-up pass to attach to. Drain them into a stable backing
    // (the payload's `trailing_blits`) so a synthetic blit-only
    // `PassDescriptor` (color_texture=0, command_count=0) below can carry
    // the pointer.
    let trailing_blits = enc.pass_state.take_pending_leading_blits();
    // Take the finalized passes out of `PassState` so they (and the
    // `commands` the descriptors point into) can outlive this frame's
    // encoder state. `apply_pass_rules` above has already rewritten them
    // in place, so the descriptors built from the taken vec are final.
    let passes = enc.pass_state.take_finished_passes();

    let visibility_buffer_handle = enc.visibility.current_buffer_handle();
    let mut descriptors: Vec<PassDescriptor> = passes
        .iter()
        .map(|p| pass_to_descriptor(p, visibility_buffer_handle))
        .collect();
    if !trailing_blits.is_empty() {
        descriptors.push(trailing_blit_descriptor(&trailing_blits));
    }

    // Swap the encoder's live per-frame buffers into a recycled payload and
    // install a clean set, so the next frame can start building while this
    // one is submitted. Every move here is an O(1) `Vec`/arena header swap;
    // the heap behind `scratch` / `frame_blit_commands` is untouched, so
    // the raw pointers built into `params` below stay valid.
    let mut payload = enc.acquire_clean_payload();
    core::mem::swap(&mut payload.scratch, &mut enc.scratch);
    core::mem::swap(
        &mut payload.frame_blit_commands,
        &mut enc.frame_blit_commands,
    );
    payload.passes = passes;
    payload.descriptors = descriptors;
    payload.trailing_blits = trailing_blits;

    let params = SubmitFrameParams {
        queue_handle: frame.queue_handle,
        blit_commands_ptr: if payload.frame_blit_commands.is_empty() {
            0
        } else {
            payload.frame_blit_commands.as_ptr() as u64
        },
        blit_command_count: u32::try_from(payload.frame_blit_commands.len())
            .expect("frame blit count fits u32"),
        blit_commands_need_encoder: u32::from(
            enc.flags
                .contains(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER),
        ),
        passes_ptr: payload.descriptors.as_ptr() as u64,
        pass_count: u32::try_from(payload.descriptors.len()).expect("pass count fits u32"),
        pad1: 0,
        present_layer: if frame.flags.contains(FrameDataFlags::NO_PRESENT) {
            MetalHandle::NULL
        } else {
            frame.layer_handle
        },
        present_texture: if frame.flags.contains(FrameDataFlags::NO_PRESENT) {
            MetalHandle::NULL
        } else {
            frame.backbuffer_handle
        },
        submit_seq: frame.submit_seq,
        coherent_seq_ptr: frame.coherent_seq_ptr,
        upload_coherent_seq_ptr: frame.upload_coherent_seq_ptr,
        failed_submit_seq_ptr: frame.failed_submit_seq_ptr,
        drawable_wait_ns: 0,
        present_view: if frame.flags.contains(FrameDataFlags::NO_PRESENT) {
            MetalHandle::NULL
        } else {
            frame.view_handle
        },
    };

    // Retention bookkeeping is keyed by `submit_seq` and only needs the
    // staging Arcs to stay alive until `coherent_seq` catches up — moving
    // them from `current_blit_retention` into `pending_blit_retention`
    // keeps them alive regardless of which thread later runs the blits, so
    // this is safe to do here before handing the payload off.
    retire_visibility_buffer(enc, frame.submit_seq);
    retire_blit_arcs(enc, frame.submit_seq);

    (params, payload)
}

/// Issue the `SubmitFrame` thunk for one finalized frame.
///
/// `params` carries raw pointers aliasing into `payload`; both are taken
/// by value so the payload stays alive for the whole thunk, then handed
/// back for recycling along with the unix status and the drawable-wait
/// nanoseconds the thunk writes into `params`. This is the only part of submit
/// that runs on the dedicated submit thread in `Async` mode.
fn execute_submit(
    mut params: SubmitFrameParams,
    payload: FramePayload,
) -> (FramePayload, i32, u64) {
    let status = unix_call(&mut params);
    (payload, status, params.drawable_wait_ns)
}

/// Recycle a finished payload.
///
/// Drain its passes' `commands` vecs back into the `PassState` pool, clear
/// the buffers (retaining their heap), and return the set to
/// `payload_pool` for the next frame's `finalize_submit`.
fn reclaim_payload(enc: &mut FrameEncoder, mut payload: FramePayload) {
    enc.pass_state.recycle_passes(&mut payload.passes);
    payload.descriptors.clear();
    payload.frame_blit_commands.clear();
    payload.trailing_blits.clear();
    payload.scratch.clear();
    enc.payload_pool.push(payload);
}

/// Convert one finalised `Pass` into a `PassDescriptor` payload for the unix-side replay.
///
/// The visibility buffer is attached only on passes that emit a `Counting`
/// command — binding it unconditionally makes Metal track the buffer in
/// the pass's resource residency set + CB dependency graph even when no
/// counter is written, and under `MTL_DEBUG_LAYER=1` the validator retains
/// per-pass tracking state proportional to pass count × frames (observed
/// as ~200 MiB/s growth in the Metal HUD). The flag is latched at
/// `emit_command` time in `passes.rs`, so this predicate is O(1) per pass.
fn pass_to_descriptor(
    p: &Pass,
    visibility_buffer_handle: MetalHandle<MTLBufferKind>,
) -> PassDescriptor {
    let (color_load_action, clear_r, clear_g, clear_b, clear_a) = match p.color_load() {
        ColorLoad::Load => (LoadAction::Load, 0, 0, 0, 0),
        ColorLoad::Clear { r, g, b, a } => (LoadAction::Clear, r, g, b, a),
        ColorLoad::DontCare => (LoadAction::DontCare, 0, 0, 0, 0),
    };
    let (depth_load_action, depth_clear_value) = match p.depth_load() {
        DepthLoad::Load => (LoadAction::Load, f32::to_bits(1.0)),
        DepthLoad::Clear { value } => (LoadAction::Clear, value),
        DepthLoad::DontCare => (LoadAction::DontCare, f32::to_bits(1.0)),
    };
    let (stencil_load_action, stencil_clear_value) = match p.stencil_load() {
        StencilLoad::Load => (LoadAction::Load, 0),
        StencilLoad::Clear { value } => (LoadAction::Clear, value),
        StencilLoad::DontCare => (LoadAction::DontCare, 0),
    };
    let mut color_store_action = match p.color_store() {
        PassStoreAction::Store => StoreAction::Store,
        PassStoreAction::DontCare => StoreAction::DontCare,
    };
    if !p.color_resolve_texture().is_null() {
        color_store_action = color_store_action.with_resolve();
    }
    let mut depth_store_action = match p.depth_store() {
        PassStoreAction::Store => StoreAction::Store,
        PassStoreAction::DontCare => StoreAction::DontCare,
    };
    if !p.depth_resolve_texture().is_null() {
        depth_store_action = depth_store_action.with_resolve();
    }
    log_pass_depth_attach(p);
    let leading = p.leading_blits();
    let visibility_result_buffer =
        if !visibility_buffer_handle.is_null() && p.has_counting_visibility() {
            visibility_buffer_handle
        } else {
            MetalHandle::NULL
        };
    PassDescriptor {
        // The attachment is the multisampled companion where the target has
        // one, and the sRGB twin view of whichever texture that is whenever
        // the pass encodes on write; every load/store rule above still
        // reasons about the base handle, which is the same Metal texture.
        color_texture: p.color_attachment_texture(),
        color_resolve_texture: p.color_resolve_texture(),
        depth_texture: p.depth_texture(),
        depth_resolve_texture: p.depth_resolve_texture(),
        commands_ptr: p.commands().as_ptr() as u64,
        visibility_result_buffer,
        leading_blits_ptr: if leading.is_empty() {
            0
        } else {
            leading.as_ptr() as u64
        },
        color_load_action,
        color_store_action,
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        depth_load_action,
        depth_store_action,
        depth_clear_value,
        stencil_load_action,
        stencil_clear_value,
        command_count: u32::try_from(p.commands().len()).expect("per-pass command count fits u32"),
        leading_blits_count: u32::try_from(leading.len())
            .expect("per-pass leading blit count fits u32"),
        // Per-pass leading blits today are only StretchRect CopyTexture
        // commands (notifies go in the frame-level list), so any
        // non-empty leading list needs the encoder. If a future caller
        // threads notifies through here, switch to a tracked flag on
        // `Pass`.
        pass_flags: PassDescriptor::pack_flags(
            !leading.is_empty(),
            p.color_slice(),
            p.color_level(),
            p.depth_level(),
        ),
        depth_resolve_filter: p.depth_resolve_filter(),
        pad0: 0,
        extra_color: core::array::from_fn(|i| {
            let a = &p.extra_color()[i];
            if !a.is_bound() {
                return ExtraColorDesc::NONE;
            }
            let store = match a.store() {
                PassStoreAction::Store => StoreAction::Store,
                PassStoreAction::DontCare => StoreAction::DontCare,
            };
            ExtraColorDesc {
                texture: a.attachment_texture(),
                resolve_texture: a.resolve_texture(),
                subresource: a.slice() | (a.level() << 8),
                load_action: match a.load() {
                    ColorLoad::Load => LoadAction::Load,
                    ColorLoad::Clear { .. } => LoadAction::Clear,
                    ColorLoad::DontCare => LoadAction::DontCare,
                },
                store_action: if a.resolve_texture().is_null() {
                    store
                } else {
                    store.with_resolve()
                },
                reserved: 0,
            }
        }),
    }
}

/// Diag probe: per-attachment load action + viewport.
///
/// A depth texture that only ever appears as `DepthLoad::Load` makes the
/// pass load undefined Private-storage memory — a shadow map that is never
/// cleared reads as garbage depth. The viewport is the smoking gun for
/// cascade caster passes whose D3D9 `SetViewport` doesn't cover the full
/// attachment: content lands only in the sub-rect, leaving the rest
/// cleared, and shadows appear/disappear as world positions project in/out
/// of that sub-rect. Once per `(depth_texture, viewport, color_size)`;
/// zero-cost when `mtld3d::d3d9::depth=trace` isn't enabled.
fn log_pass_depth_attach(p: &Pass) {
    if p.depth_texture().is_null() {
        return;
    }
    let (vpx, vpy, vpw, vph) = p.viewport();
    let (cw, ch) = p.color_size();
    let vp_key =
        (u64::from(vpx) << 48) ^ (u64::from(vpy) << 32) ^ (u64::from(vpw) << 16) ^ u64::from(vph);
    mtld3d_shared::log_once_trace_by!(
        target: DEPTH_TRACE_TARGET,
        key: p.depth_texture().raw().rotate_left(13) ^ vp_key,
        "depth: pass attach={:#x} load={:?} viewport=({vpx},{vpy},{vpw}x{vph}) color_size={cw}x{ch}",
        p.depth_texture(),
        p.depth_load()
    );
}

/// Synthetic blit-only `PassDescriptor`.
///
/// Carries trailing `StretchRect` blits queued after the last draw of the
/// frame. No color/depth attachments, no commands; the unix side spins an
/// encoder only because `CopyTextureToTexture` needs one.
fn trailing_blit_descriptor(trailing_blits: &[BlitCommand]) -> PassDescriptor {
    PassDescriptor {
        color_texture: MetalHandle::NULL,
        color_resolve_texture: MetalHandle::NULL,
        depth_texture: MetalHandle::NULL,
        depth_resolve_texture: MetalHandle::NULL,
        commands_ptr: 0,
        visibility_result_buffer: MetalHandle::NULL,
        leading_blits_ptr: trailing_blits.as_ptr() as u64,
        color_load_action: LoadAction::DontCare,
        color_store_action: StoreAction::DontCare,
        clear_r: 0,
        clear_g: 0,
        clear_b: 0,
        clear_a: 0,
        depth_load_action: LoadAction::DontCare,
        depth_store_action: StoreAction::DontCare,
        depth_clear_value: 0,
        stencil_load_action: LoadAction::DontCare,
        stencil_clear_value: 0,
        command_count: 0,
        leading_blits_count: u32::try_from(trailing_blits.len())
            .expect("trailing blit count fits u32"),
        pass_flags: PassDescriptor::pack_flags(true, 0, 0, 0),
        depth_resolve_filter: DepthResolveFilter::Sample0,
        pad0: 0,
        extra_color: [ExtraColorDesc::NONE; 3],
    }
}

/// Apply the load/store optimiser rules in dependency order.
///
/// Rule E (coalesce) runs first so the load/store finalisers see the
/// merged pass list. Rule A reverts eager `Load=DontCare` whose attachment
/// is sampled later this frame; Rules B/C set store actions on stable load
/// actions. Rule G strips dead color attachments from clear-only passes
/// (kills Apple's "Unused Texture" Insight on the cascade placeholder).
/// Rule H strips color from passes-with-draws where every draw had
/// `color_write_mask=0` (caster passes), rewriting `SetRenderPipelineState`
/// to the no-color variant so Metal's RP-format validation stays happy.
/// Rule F drops clear-only passes that nothing observes; must run after
/// Rule G so the cull picks up the strip.
fn apply_pass_rules(enc: &mut FrameEncoder, frame_continues: bool) {
    enc.pass_state.coalesce_clear_only_passes();
    enc.pass_state.finalize_load_actions();
    enc.pass_state.finalize_store_actions(frame_continues);
    enc.pass_state.strip_dead_color_in_clear_only_passes();
    enc.pass_state
        .strip_color_from_no_color_draw_passes(&enc.no_color_pipeline_alt);
    enc.pass_state.cull_dead_clear_only_passes();
}

/// Per-frame cascade summary probe.
///
/// One row per frame listing every cascade depth handle that either (a)
/// received caster writes or (b) was bound as a fragment-sample target
/// this frame, with the counts for each. Built to localise tree
/// self-shadow flicker: a cascade with `samples>0 caster=0` is the smoking
/// gun — receiver sampled this cascade with no fresh caster content this
/// frame, falling back to whatever stale content survived from earlier (or
/// to the cleared 1.0 if the double-buffer sibling was also dry). Opt in
/// with `RUST_LOG=mtld3d::d3d9::cascade=trace`. Counter sites inside
/// `PassState` gate their own writes on the same `cascade=trace` target,
/// so the per-frame maps stay empty when the probe is off. No drain needed
/// in the off path — empty maps cost nothing to leave behind, and
/// `reset_frame` clears them on the next frame as a belt-and-braces
/// safety.
fn log_cascade_frame_summary(enc: &mut FrameEncoder) {
    if !log::log_enabled!(target: "mtld3d::d3d9::cascade", log::Level::Trace) {
        return;
    }
    let (frame_seq, rows) = enc.pass_state.take_cascade_frame_summary();
    if rows.is_empty() {
        return;
    }
    let mut buf = String::with_capacity(rows.len() * 48);
    for (tex, caster, samples) in &rows {
        let _ = std::fmt::Write::write_fmt(
            &mut buf,
            format_args!(" 0x{tex:x}[w={caster},r={samples}]"),
        );
    }
    log::trace!(
        target: "mtld3d::d3d9::cascade",
        "cascade-frame seq={frame_seq}{buf}",
    );
}

/// Move this frame's visibility buffer (if any was reserved) into the pool's retired list.
///
/// The list is keyed by `submit_seq`. It becomes reusable once the GPU
/// retires the frame (`coherent_seq` catches up), which both releases the
/// buffer *and* unblocks `intake_completed` so pending queries matched
/// against this seq can be summed.
///
/// If the pool is over cap, the oldest retired entry is evicted. Route
/// it through `pending_resource_retention` so the drain path destroys
/// the `MTLBuffer` wrapper before the `PageBox` drops — Metal still
/// holds a `bytesNoCopy` pointer into the backing until `DestroyBuffer`
/// fires.
fn retire_visibility_buffer(enc: &mut FrameEncoder, submit_seq: u64) {
    let Some(evicted) = enc.visibility.retire_current_buffer(submit_seq) else {
        return;
    };
    let (page_box, mtl_buffer, release_seq) = evicted.into_parts();
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "visibility buffer pool over cap — evicting oldest entry \
         (seq={release_seq}); routing through pending_resource_retention so \
         DestroyBuffer fires before the PageBox drops"
    );
    enc.perf.bump_vbib_retained_add(page_box.len());
    enc.add_retained_bytes(page_box.len());
    enc.pending_resource_retention
        .push_back(PendingResourceRetention {
            kind: DestroyKind::Buffer,
            handle: mtl_buffer.raw(),
            page_box: Some(page_box),
            staging_arc: None,
            seq: release_seq,
            from_texture: false,
        });
}

/// Move this frame's blit-source Arc retentions into the pending queue.
///
/// Keyed by the frame's `submit_seq`. They're released when `coherent_seq`
/// reaches `submit_seq` — checked next `begin_frame`. Called from
/// `finalize_submit`, before the thunk is issued: the move into
/// `pending_blit_retention` is what keeps the Arcs alive across the blit
/// encode + commit path, whichever thread runs it.
fn retire_blit_arcs(enc: &mut FrameEncoder, submit_seq: u64) {
    for arc in enc.current_blit_retention.drain(..) {
        enc.perf.bump_tex_staging_retained_add(arc.len());
        enc.pending_blit_retention
            .push_back(PendingBlitArc::new(submit_seq, arc));
    }
}

/// Banner tag for a compiled shader in MSL trace dumps.
///
/// The hash is the on-disk shader-cache `disk_key` so the banner
/// identifier matches the pass×shader log line, the truncated hex in the
/// Xcode pipeline label (`mtld3d_vs_*_<8hex>`), and (for programmable
/// shaders) the `debug.bytecodeDumpDir` `vs_<hash>.dxso` filename.
fn shader_source_tag_vs(source: &VsSource) -> String {
    match source {
        VsSource::Programmable {
            vs_id,
            provided_input_mask,
            clip_plane_count,
            sampler_kinds,
            ..
        } => {
            format!(
                "prog {:#x}",
                draw::vs_source_disk_key_programmable(
                    *vs_id,
                    *provided_input_mask,
                    *clip_plane_count,
                    *sampler_kinds
                )
            )
        }
        VsSource::FixedFunction { key, .. } => {
            format!("ff {:#x}", draw::vs_source_disk_key_ff(key))
        }
    }
}

fn shader_source_tag_ps(source: &PsSource, variant: VariantKey) -> String {
    match source {
        PsSource::Programmable { ps_id, .. } => format!(
            "prog {:#x}",
            draw::ps_source_disk_key_programmable(*ps_id, variant)
        ),
        PsSource::FixedFunction { key, .. } => {
            format!("ff {:#x}", draw::ps_source_disk_key_ff(key, variant))
        }
    }
}

/// Resolve the `CachedKind` for a live-path VS source.
///
/// The entry name is derived from it via `CachedKind::entry_name`. Falls
/// back to `Sm2Vs` for programmable shaders with an out-of-range major
/// (the live path will fail compile elsewhere; this just keeps the name
/// well-formed).
fn vs_entry_name(
    source: &VsSource,
    program_cache: &FxHashMap<ProgramId, Box<DxsoProgram>>,
    disk_key: u64,
) -> String {
    let kind = match source {
        VsSource::Programmable { vs_id, .. } => {
            let major = program_cache.get(vs_id).map_or(2, |p| p.major);
            CachedKind::from_programmable(major, false).unwrap_or(CachedKind::Sm2Vs)
        }
        VsSource::FixedFunction { .. } => CachedKind::FfVs,
    };
    kind.entry_name(disk_key)
}

fn ps_entry_name(
    source: &PsSource,
    program_cache: &FxHashMap<ProgramId, Box<DxsoProgram>>,
    disk_key: u64,
) -> String {
    let kind = match source {
        PsSource::Programmable { ps_id, .. } => {
            let major = program_cache.get(ps_id).map_or(2, |p| p.major);
            CachedKind::from_programmable(major, true).unwrap_or(CachedKind::Sm2Ps)
        }
        PsSource::FixedFunction { .. } => CachedKind::FfPs,
    };
    kind.entry_name(disk_key)
}

/// Zero the full backing of a `PageBox`.
///
/// Called when a visibility buffer is pulled off the pool for reuse —
/// Metal only writes u64 counters to slots it touches under Counting mode,
/// so stale values in slots we bump but the GPU never enters Counting for
/// would leak across frames without this.
fn zero_page_box(backing: &mut PageBox) {
    backing.as_mut_slice().fill(0);
}

// ── Shader disk cache helpers ──

/// Resolve `<host-exe-dir>/mtld3d_shaders.bin` once per process.
///
/// `None` when `current_exe()` fails or has no parent (unusual setups;
/// the caller treats this as "cache disabled").
pub fn shader_cache_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(parent.join("mtld3d_shaders.bin"))
}

/// Mirror of the `shaderCache.enable` config key.
///
/// Read once at `FrameEncoder::new` and once at pre-warm spawn; users set
/// `shaderCache.enable = false` in `mtld3d.conf` to bypass disk caching
/// for both. Default: `true`.
pub fn shader_cache_enabled() -> bool {
    crate::config::CONFIG.shader_cache_enable
}

/// Open the cache file in append mode, creating it (and writing the 16-byte header) if absent.
///
/// Caller invokes lazily on first miss-compile, after the pre-warm thread
/// has already validated the file's schema, so a non-empty file we
/// encounter here is guaranteed to already start with a valid header.
fn open_or_create_cache_file() -> std::io::Result<File> {
    let Some(path) = shader_cache_path() else {
        return Err(std::io::Error::other("shader_cache_path unavailable"));
    };
    let exists = path.exists();
    let mut f = OpenOptions::new().append(true).create(true).open(&path)?;
    if !exists {
        let mut hdr = Vec::with_capacity(shader_cache::HEADER_LEN);
        shader_cache::write_header(&mut hdr);
        f.write_all(&hdr)?;
    }
    Ok(f)
}
