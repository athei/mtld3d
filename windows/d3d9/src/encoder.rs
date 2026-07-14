use std::{
    collections::{HashSet, VecDeque, hash_map::Entry},
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
    convert::d3d_to_metal_cmp,
    dxso::{
        DxsoProgram, FfPsKey, FfVsKey, LOG_TARGET as MSL_TRACE_TARGET, VariantKey,
        emit_ps_ff_named, emit_ps_programmable_named, emit_vs_ff_named, emit_vs_programmable_named,
    },
    format::map_d3d_format,
    gpu_caps::GpuCaps,
    ids::{BufferId, DepthStencilKey, ProgramId, SamplerKey, TextureId},
    page_box::PageBox,
    passes::{
        ColorClearOutcome, ColorLoad, DepthClearOutcome, DepthLoad, LastBoundCache, Pass,
        PassState, StoreAction as PassStoreAction,
    },
    perf::{
        CacheSizes, EncoderPerfState, FramePerfPayload, FrameSummaryContext, OpSub, OpSubDetail,
        PairShaderId, PairStatsSample,
    },
    pipeline_state::{self, PipelineBuildInputs, PipelineKey, PipelineSnapshot},
    sampler_state,
    scratch::ScratchArena,
    shader_cache::{self, CachedKind},
    shader_compile_stats::{self, BurstTracker, CompileBucket},
    storage_policy::buffer_storage_mode,
    visibility::{
        MAX_SLOTS, RetiredVisibilityBuffer, SLOT_BYTES, VisibilityQueryCore, VisibilityQueryState,
    },
};
use mtld3d_shared::{
    BlitCommand, BufferCreateDesc, Command, CommandType, CompileShaderLibraryParams,
    CopyBufferToBufferInfo, CopyBufferToTextureInfo, CreateBuffersBatchParams,
    CreateDepthStencilStateParams, CreateTexturesBatchParams, DestroyResourcesBulkParams,
    EnsureBlitPipelineParams, EnsureClearQuadPipelineParams, MetalHandle, PassDescriptor,
    SetDisplaySyncEnabledParams, SubmitFrameParams, TextureCreateDesc, VertexAttrDesc,
    WaitForGpuRetireParams,
    mtl::{
        BufferKind, ClearQuadFlags, DestroyKind, LoadAction, PixelFormat, PrimitiveType, StageTag,
        StorageMode, StoreAction, Swizzle, TextureUsage, VisibilityResultMode,
    },
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLCommandQueueKind, MTLDepthStencilStateKind,
        MTLDeviceKind, MTLFunctionKind, MTLRenderPipelineStateKind, MTLSamplerStateKind,
        MTLTextureKind, NSViewKind,
    },
    tsc::{rdtsc, secs_to_cycles},
};
use mtld3d_types::{D3DCMP_ALWAYS, D3DSAMP_MIPMAPLODBIAS, SAMPLER_STATE_COUNT};
// Fast non-cryptographic hasher for the per-draw resource caches below
// (texture/lib/pipeline/sampler/buffer/...). Keys are small trusted integers
// or fixed structs; SipHash's DoS resistance buys nothing here and its
// per-probe cost shows up in the encoder `resolve`/`binds`/`samplers` phases.
use rustc_hash::FxHashMap;

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

/// One side (source or destination) of a scaled blit, for [`FrameEncoder::stretch_blit_scaled`].
pub struct BlitSide {
    pub handle: u64,
    pub rect: mtld3d_core::stretch_rect::StretchRegion,
    pub dims: (u32, u32),
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
    /// single-slice `bytes_per_image` from `src_pitch * region_h` instead.
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
pub struct TextureGpuState {
    pub mtl_texture: MetalHandle<MTLTextureKind>,
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

/// Texture metadata captured from the API thread for deferred Metal creation.
#[derive(Clone)]
pub struct TextureInfo {
    pub texture_id: TextureId,
    pub width: u32,
    pub height: u32,
    /// Slice count: 1 for 2D textures, >1 for a volume (3D) texture.
    pub depth: u32,
    pub levels: u32,
    pub pixel_format: PixelFormat,
    pub has_swizzle: u32,
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
/// drawable-wait cycles the `SubmitFrame` thunk measured.
struct ReturnedPayload {
    payload: FramePayload,
    status: i32,
    drawable_wait_tsc: u64,
    /// Total submit-thread CPU for `execute_submit`.
    ///
    /// Covers the unix command-walk, present, and commit — including the
    /// `drawable_wait_tsc` portion. Folded into perf so the summary can
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
        let (payload, status, drawable_wait_tsc) = {
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
                drawable_wait_tsc,
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
    pass_shader_log_fired: HashSet<(MetalHandle<MTLTextureKind>, PairShaderId, PairShaderId)>,

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
    /// One entry per destination colour `PixelFormat`. Used by the scaling
    /// `StretchRect` path (`stretch_blit_scaled`) to render the source
    /// texture onto a quad covering the destination rect — Metal's blit
    /// encoder can't scale. Process-lifetime, same posture as
    /// `clear_quad_pipeline_cache`.
    blit_pipeline_cache: FxHashMap<PixelFormat, MetalHandle<MTLRenderPipelineStateKind>>,
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
    /// (variants share one `MTLLibrary`), PS keys fold it in. The Xxh3
    /// `disk_key` is computed only on a miss here, to bridge `lib_cache`
    /// (warm-load) and address the on-disk cache.
    ff_vs_libs: FxHashMap<FfVsKey, StageLibHandles>,
    prog_vs_libs: FxHashMap<(ProgramId, u16), StageLibHandles>,
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
/// Used by `emit_clear_quad_depth_inner` / `emit_clear_quad_color_inner`.
/// Mirrors `EnsureClearQuadPipelineParams` (modulo `device_handle`) so the
/// PE-side cache and the unix-side cache agree on what counts as a
/// distinct pipeline.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct ClearQuadKey {
    depth_format: PixelFormat,
    color_format: PixelFormat,
    flags: ClearQuadFlags,
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
    /// Written by the staging-upload blit. `NULL` for `Direct`.
    device_buffer: MetalHandle<MTLBufferKind>,
    /// `true` for a non-DYNAMIC buffer on the separate-staging upload path.
    ///
    /// `false` for the zero-copy `Direct` path.
    is_staged: bool,
    backing_ptr: u64,
    length: u64,
    /// Max submit seq this wrapper has been bound into a Draw for.
    ///
    /// Used when the cache entry is evicted to retention.
    last_submit_seq: u64,
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
    /// wrapper destroy on rename, and padded-blit transient wrapper. At
    /// drain time the destroy is attributed to the textures `destroys`
    /// row instead of VB/IB. Default `false` covers VB/IB rename/intake
    /// and visibility-pool eviction — those stay on the VB/IB row.
    from_texture: bool,
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
            current_blit_retention: Vec::new(),
            pending_blit_retention: VecDeque::new(),
            coherent_seq_ptr: 0,
            retained_bytes_ptr: 0,
            current_submit_seq: 0,
            backbuffer_width: 0,
            backbuffer_height: 0,
            perf: EncoderPerfState::new(),
            pass_shader_log_fired: HashSet::new(),
            gpu_caps,
            device_handle: MetalHandle::NULL,
            depth_stencil_cache: FxHashMap::default(),
            pipeline_cache: FxHashMap::default(),
            clear_quad_pipeline_cache: FxHashMap::default(),
            blit_pipeline_cache: FxHashMap::default(),
            no_color_pipeline_alt: FxHashMap::default(),
            last_pipeline_memo: None,
            program_cache: FxHashMap::default(),
            lib_cache: FxHashMap::default(),
            ff_vs_libs: FxHashMap::default(),
            prog_vs_libs: FxHashMap::default(),
            ff_ps_libs: FxHashMap::default(),
            prog_ps_libs: FxHashMap::default(),
            texture_cache: FxHashMap::default(),
            sampler_cache: FxHashMap::default(),
            sampler_resolve_memo: core::array::from_fn(|_| None),
            buffer_cache: FxHashMap::default(),
            pending_resource_retention: VecDeque::new(),
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
    /// Caller owns `descs` and `handles_out`; both slices must outlive
    /// the call because the unix side dereferences both pointers during
    /// the thunk. On success `handles_out[i]` carries the handle for
    /// `descs[i]`; on per-element failure the slot stays at its
    /// initial value (caller passes zeros).
    fn batch_create_textures(
        &self,
        descs: &[TextureCreateDesc],
        handles_out: &mut [MetalHandle<MTLTextureKind>],
    ) -> i32 {
        debug_assert_eq!(descs.len(), handles_out.len());
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
    const fn texture_desc_from_info(info: &TextureInfo) -> TextureCreateDesc {
        // Every texture created here is `Private`. Nothing CPU-writes a
        // texture directly: render targets are GPU output, and all uploads
        // (including the A4R4G4B4 / R5G6B5 / A1R5G5B5 → BGRA8 expansion path)
        // go through `copyFromBuffer:toTexture:` blits, whose destination can
        // be Private. There is deliberately no CPU-timeline `replaceRegion`
        // path — it would race a texture sampled by an in-flight frame — so no
        // texture needs a CPU-writable mode. Only the staging *buffers* (blit
        // sources) follow `buffer_storage_mode`.
        let storage_mode = StorageMode::Private;
        TextureCreateDesc {
            tex_id: info.texture_id.raw(),
            width: info.width,
            height: info.height,
            depth: info.depth,
            levels: info.levels,
            pixel_format: info.pixel_format,
            storage_mode,
            has_swizzle: info.has_swizzle,
            swizzle_r: info.swizzle[0],
            swizzle_g: info.swizzle[1],
            swizzle_b: info.swizzle[2],
            swizzle_a: info.swizzle[3],
            usage_flags: info.usage_flags,
        }
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
        let descs: Vec<TextureCreateDesc> =
            infos.iter().map(Self::texture_desc_from_info).collect();
        let mut handles = vec![MetalHandle::<MTLTextureKind>::NULL; descs.len()];
        let status = self.batch_create_textures(&descs, &mut handles);
        if status != 0 {
            error!(
                target: LOG_TARGET,
                "drain_texture_warmups: CreateTexturesBatch status={status:#x} (count={})",
                infos.len()
            );
        }
        let current_seq = self.current_submit_seq;
        for (info, handle) in infos.into_iter().zip(handles) {
            if handle.is_null() {
                continue;
            }
            match self.texture_cache.entry(info.texture_id) {
                Entry::Vacant(v) => {
                    v.insert(TextureGpuState {
                        mtl_texture: handle,
                        mip_staging_buffers: vec![
                            MipStagingBuffer::default();
                            info.levels as usize
                        ],
                    });
                }
                Entry::Occupied(_) => {
                    mtld3d_shared::log_once_warn!(
                        target: LOG_TARGET,
                        "drain_texture_warmups: cache collision for tex_id, queueing orphan handle for retire"
                    );
                    self.pending_resource_retention
                        .push_back(PendingResourceRetention {
                            kind: DestroyKind::Texture,
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
            if handle.is_null() {
                continue;
            }
            let staged = matches!(warmup.map_mode, BufferMapMode::Staged);
            match self.buffer_cache.entry(warmup.buffer_id) {
                Entry::Vacant(v) => {
                    v.insert(BufferGpuState {
                        mtl_buffer: if staged { MetalHandle::NULL } else { handle },
                        device_buffer: if staged { handle } else { MetalHandle::NULL },
                        is_staged: staged,
                        backing_ptr: if staged { 0 } else { warmup.backing_ptr },
                        length: warmup.backing_len,
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
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "apply_stage_upload: no Staged device buffer for buffer_id — dropping upload"
            );
            return;
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

        // Retire the transient wrapper + backing once this frame's submit
        // retires (the blit reads it by then). Account the CPU bytes into
        // the shared retention total like every queued `PageBox`.
        self.perf.bump_vbib_retained_add(page_box.len());
        self.add_retained_bytes(page_box.len());
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Buffer,
                handle: transient.raw(),
                page_box: Some(page_box),
                staging_arc: None,
                seq: current_seq,
                from_texture: false,
            });
    }

    /// Allocate a fresh `StorageModePrivate` device buffer for a `Staged` VB/IB.
    ///
    /// Backs a rename-at-overlap. Returns `None` on create failure
    /// (caller falls back to overwriting the live buffer).
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
        self.flags.remove(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        mtld3d_shared::crumb!("phase:BfRecl");
        self.reclaim_retired_blit_retention();
        if frame.coherent_seq_ptr != 0 {
            self.coherent_seq_ptr = frame.coherent_seq_ptr;
        }
        if frame.retained_bytes_ptr != 0 {
            self.retained_bytes_ptr = frame.retained_bytes_ptr;
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
        self.pass_state.reset_frame(
            frame.backbuffer_handle,
            (frame.backbuffer_width, frame.backbuffer_height),
            frame.backbuffer_format,
            frame.depth_texture,
            frame.flags.contains(FrameDataFlags::DEPTH_HAS_STENCIL),
        );
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
        self.perf
            .log_frame_summary(&caches, &payload.passes, ctx, status, cmd_vec_realloc_bytes);
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
        self.perf
            .set_drawable_wait_cycles(returned.drawable_wait_tsc);
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
        let was_closed = self.pass_state.current_pass_closed();
        self.pass_state.emit_command(cmd);
        self.reset_last_bound_on_fresh_pass(was_closed);
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
        let was_closed = self.pass_state.current_pass_closed();
        self.pass_state.emit_command(cmd);
        self.reset_last_bound_on_fresh_pass(was_closed);
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
    #[must_use]
    pub const fn current_depth_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.pass_state.current_depth_texture()
    }

    /// Record a caster draw against the currently-bound cascade depth handle.
    ///
    /// Called from `draw.rs::emit_draw`. The `PassState` implementation
    /// self-filters to known-sampleable handles via
    /// `seen_sampleable_depth_textures`.
    pub fn note_caster_draw(&mut self, depth_tex: MetalHandle<MTLTextureKind>) {
        self.pass_state.note_caster_draw(depth_tex);
    }

    /// `true` when `depth_tex` was bound as a sampleable shadow map at any point this session.
    ///
    /// Proxies `PassState::is_depth_handle_sampleable` for the
    /// `mtld3d::d3d9::caster`/`cascade` trace probes — needed because
    /// `current_depth_is_sampleable()` can be incorrectly false after
    /// a `GetDepthStencilSurface` save/restore cycle.
    #[must_use]
    pub fn is_depth_handle_sampleable(&self, depth_tex: MetalHandle<MTLTextureKind>) -> bool {
        self.pass_state.is_depth_handle_sampleable(depth_tex)
    }

    /// Queue a `StretchRect` blit to run before the *next* pass.
    ///
    /// Caller must `end_current_pass()` first so the blit is correctly
    /// ordered between the just-ended pass's draws and the next pass's
    /// draws. If no further pass opens this frame, `submit` synthesises
    /// a trailing blit-only `PassDescriptor` to drain it.
    pub fn push_stretch_rect_blit(&mut self, blit: BlitCommand) {
        self.pass_state.push_pending_leading_blit(blit);
    }

    pub fn set_color_render_target(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
        format: PixelFormat,
        has_alpha: bool,
    ) {
        self.pass_state
            .set_color_render_target(texture, width, height, format);
        // Kept in lockstep with the format: the Metal pixel format alone can't
        // distinguish X8R8G8B8 (no alpha) from A8R8G8B8 (both `Bgra8Unorm`).
        self.pass_state.set_color_rt_has_alpha(has_alpha);
    }

    /// Metal pixel format of the currently bound color RT.
    ///
    /// Read at draw time to key the pipeline cache on RT format so
    /// multiple passes against different formats don't share a pipeline.
    pub const fn current_color_format(&self) -> PixelFormat {
        self.pass_state.current_color_format()
    }

    /// Whether the currently bound color RT's D3D format has a real alpha channel.
    ///
    /// Read at draw time into the pipeline snapshot's `COLOR_HAS_ALPHA`
    /// bit so destination-alpha blend factors clamp on alpha-less
    /// targets (X8R8G8B8).
    pub const fn current_color_rt_has_alpha(&self) -> bool {
        self.pass_state.current_color_rt_has_alpha()
    }

    /// Current viewport `(x, y, w, h)` in pixels, with the `ensure_pass_open` fallback.
    ///
    /// Falls back to the bound RT size when the game never set a
    /// viewport. Read at draw time to derive the half-pixel `pos_fixup`
    /// uniform (VS slot 13).
    pub const fn effective_viewport(&self) -> (u32, u32, u32, u32) {
        self.pass_state.effective_viewport()
    }

    /// Mark a colour texture as read back this session.
    ///
    /// See `PassState::note_color_read_back`. The store-action optimiser
    /// then keeps its rendered content for a post-frame
    /// `GetRenderTargetData` blit.
    pub fn note_color_read_back(&mut self, handle: MetalHandle<MTLTextureKind>) {
        self.pass_state.note_color_read_back(handle);
    }

    pub fn set_depth_stencil_attachment(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        is_sampleable: bool,
        has_stencil: bool,
    ) {
        self.pass_state
            .set_depth_stencil_attachment(texture, is_sampleable, has_stencil);
    }

    pub fn clear_color(&mut self, r: u32, g: u32, b: u32, a: u32) {
        let was_closed = self.pass_state.current_pass_closed();
        match self.pass_state.clear_color(r, g, b, a) {
            ColorClearOutcome::Folded => {}
            ColorClearOutcome::EmitQuad {
                rgba,
                viewport,
                color_format,
            } => {
                self.reset_last_bound_on_fresh_pass(was_closed);
                self.emit_clear_quad_color_inner(rgba, viewport, color_format);
            }
        }
    }

    /// `Clear(pRects = NULL)` for colour only — D3D9 bounds it to the current viewport ∩ RT.
    ///
    /// A viewport that covers the whole attachment folds to a fast
    /// full-attachment `loadAction = Clear`; a strict sub-region instead
    /// emits one scissored clear-quad over the viewport so pixels
    /// outside it keep their prior content.
    ///
    /// Only used for colour-only clears — a combined `TARGET|ZBUFFER`
    /// clear stays on the fold path (`clear_color` + `clear_depth`) so
    /// the depth side is not forced onto the clear-quad path.
    pub fn clear_color_bounded_to_viewport(&mut self, r: u32, g: u32, b: u32, a: u32) {
        if self.pass_state.viewport_covers_color_attachment() {
            self.clear_color(r, g, b, a);
        } else {
            let (vpx, vpy, vpw, vph) = self.pass_state.effective_viewport();
            let rect = (
                vpx.cast_signed(),
                vpy.cast_signed(),
                vpx.saturating_add(vpw).cast_signed(),
                vpy.saturating_add(vph).cast_signed(),
            );
            self.clear_color_rects(r, g, b, a, &[rect]);
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
        rects: &[(i32, i32, i32, i32)],
    ) {
        let vp = self.pass_state.effective_viewport();
        let regions: Vec<(u32, u32, u32, u32)> = rects
            .iter()
            .filter_map(|&rc| clip_rect_to_viewport(rc, vp))
            .collect();
        if regions.is_empty() {
            return;
        }
        let was_closed = self.pass_state.current_pass_closed();
        let color_format = self.pass_state.begin_region_color_clear();
        self.reset_last_bound_on_fresh_pass(was_closed);
        for region in regions {
            self.emit_clear_quad_color_inner((r, g, b, a), region, color_format);
        }
    }

    pub fn clear_depth(&mut self, value: u32) {
        let was_closed = self.pass_state.current_pass_closed();
        match self.pass_state.clear_depth(value) {
            DepthClearOutcome::Folded => {}
            DepthClearOutcome::EmitQuad {
                value,
                viewport,
                has_color,
                color_format,
            } => {
                self.reset_last_bound_on_fresh_pass(was_closed);
                self.emit_clear_quad_depth_inner(value, viewport, has_color, color_format);
            }
        }
    }

    /// Flush `last_bound` when a cross-pass clear-quad opened a fresh Metal encoder.
    ///
    /// `PassState::clear_{color,depth}` opens the new pass itself
    /// (`ensure_pass_open`, with `loadAction = Load` to preserve prior
    /// tiles), but it can't reach the `FrameEncoder`-owned `last_bound`,
    /// so unlike `begin_render_pass_if_needed` the per-draw dedup would
    /// carry stale bindings across the encoder boundary. The new encoder
    /// starts with no bindings, so the next draw must re-emit everything
    /// — including the FF VS constants at buffer 15, whose content-based
    /// dedup otherwise suppresses the re-bind when the constants are
    /// unchanged from the prior pass (e.g. a sample pass after
    /// `SetDepthStencilSurface(NULL)` with the same viewport).
    fn reset_last_bound_on_fresh_pass(&mut self, was_closed: bool) {
        if was_closed {
            self.last_bound.reset();
            // Keep the debug-build emitted-command shadow in lockstep with the
            // cache so the in-sync assertion shares the same fresh-encoder
            // baseline (no bindings yet).
            #[cfg(debug_assertions)]
            self.pass_state.debug_reset_emitted();
        }
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
    fn get_or_create_blit_pipeline(&mut self, color_format: PixelFormat) -> u64 {
        if let Some(&handle) = self.blit_pipeline_cache.get(&color_format) {
            return handle.raw();
        }
        let mut params = EnsureBlitPipelineParams {
            device_handle: self.device_handle,
            color_format,
            pad0: 0,
            pipeline_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut params);
        let pipeline = params.pipeline_handle;
        if status != 0 || pipeline.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "blit-quad: EnsureBlitPipeline failed status={status:#x} → scaling StretchRect dropped"
            );
            self.blit_pipeline_cache
                .insert(color_format, MetalHandle::NULL);
            return 0;
        }
        self.blit_pipeline_cache.insert(color_format, pipeline);
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
        ss[mtld3d_types::D3DSAMP_MIPFILTER as usize] = mtld3d_types::D3DTEXF_NONE;
        ss[mtld3d_types::D3DSAMP_ADDRESSU as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        ss[mtld3d_types::D3DSAMP_ADDRESSV as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        ss[mtld3d_types::D3DSAMP_ADDRESSW as usize] = mtld3d_types::D3DTADDRESS_CLAMP;
        self.get_or_create_sampler(0, &ss, false)
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
    /// pipeline cache + the pass colour attachment); `filter` is the D3D9
    /// `D3DTEXF_*` value (POINT / LINEAR).
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
        filter: u32,
    ) {
        let &BlitSide {
            handle: src_handle,
            rect: src_rect,
            dims: src_dims,
        } = src;
        let &BlitSide {
            handle: dst_handle,
            rect: dst_rect,
            dims: dst_dims,
        } = dst;
        if dst_handle == 0 || src_handle == 0 {
            return;
        }
        // SAFETY: both handles are live Metal texture addresses resolved by
        // the caller (`get_or_create_texture` / a standalone colour handle),
        // non-zero per the guard above.
        let dst_tex = unsafe { MetalHandle::<MTLTextureKind>::new(dst_handle) };
        let pipeline = self.get_or_create_blit_pipeline(dst_format);
        if pipeline == 0 {
            return;
        }
        let sampler = self.get_or_create_blit_sampler(filter);
        if sampler == 0 {
            return;
        }

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

        // Save the device's current attachments + viewport so the one-off
        // destination pass doesn't perturb the live render target.
        let prev_color = self.pass_state.current_color_texture();
        let prev_color_size = self.pass_state.current_color_size();
        let prev_color_format = self.pass_state.current_color_format();
        let prev_depth = self.pass_state.current_depth_texture();
        let prev_depth_sampleable = self.pass_state.current_depth_is_sampleable();
        let prev_depth_has_stencil = self.pass_state.current_depth_has_stencil();
        let prev_viewport = self.pass_state.viewport();
        let (prev_min_z, prev_max_z) = self.pass_state.viewport_depth_range();

        // Bind the destination as the colour RT with no depth attachment, then
        // open a Load pass scoped to the destination rect via the viewport.
        // `set_color_render_target` / `set_depth_stencil_attachment` end the
        // current pass for us, so the quad never draws on a stale encoder.
        self.pass_state
            .set_color_render_target(dst_tex, dst_dims.0, dst_dims.1, dst_format);
        self.pass_state
            .set_depth_stencil_attachment(MetalHandle::NULL, false, false);
        self.pass_state
            .set_viewport(dst_rect.x, dst_rect.y, dst_rect.w, dst_rect.h, 0.0, 1.0);
        self.pass_state.ensure_pass_open();
        // The destination's content survives the readback that drives the
        // conformance check (and any real `GetRenderTargetData`).
        self.pass_state.note_color_read_back(dst_tex);
        // A fresh Metal encoder always opens here (the colour RT changed, which
        // ends any prior pass), so flush the per-draw dedup so every binding
        // below is actually emitted (the clear-quad cross-pass rule).
        self.reset_last_bound_on_fresh_pass(true);

        let depth_state = self.get_or_create_depth_stencil(0, 0, D3DCMP_ALWAYS);
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
        if self.last_bound.fragment_texture_changed(0, src_handle) {
            self.pass_state
                .emit_command(Command::set_fragment_texture(src_handle, 0));
        }
        if self.last_bound.fragment_sampler_changed(0, sampler) {
            self.pass_state
                .emit_command(Command::set_fragment_sampler_state(sampler, 0));
        }
        self.pass_state
            .emit_command(Command::set_vertex_bytes_at(xform_ptr, RGBA_BYTE_LEN, 0));
        // Inline slot-0 vertex bind clobbers any real bound VB; drop the cache
        // so a subsequent bound draw re-emits its `setVertexBuffer`.
        self.last_bound.invalidate_vertex_buffer();
        self.pass_state
            .emit_command(Command::draw_primitives(PrimitiveType::Triangle, 0, 3));
        self.end_current_pass("stretch_blit_scaled");

        // Restore the device's previous attachments + viewport.
        self.pass_state.set_color_render_target(
            prev_color,
            prev_color_size.0,
            prev_color_size.1,
            prev_color_format,
        );
        self.pass_state.set_depth_stencil_attachment(
            prev_depth,
            prev_depth_sampleable,
            prev_depth_has_stencil,
        );
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

    /// Emit the per-tile clear-quad sequence for a depth-only (or depth+color) mid-pass `Clear`.
    ///
    /// Sequence: pipeline → DSS → scissor → `SetVertexBytesAt(slot=0, &z)`
    /// → `DrawPrimitives (Triangle, 0, 3)`. Pipeline/DSS/scissor are
    /// routed through `LastBoundCache` so back-to-back clear-quads and
    /// clear-quad-then-redraw both dedup (and the cache stays in sync
    /// with the encoder's actual bound state). The 3-vertex VS uses
    /// `vertex_id` to synthesise a fullscreen triangle covering
    /// `[-1, 1]^2` in clip space; the scissor constrains writes to the
    /// D3D9 viewport rect; the constant `z` becomes the depth value the
    /// depth-stencil state writes to the depth attachment.
    fn emit_clear_quad_depth_inner(
        &mut self,
        value: u32,
        viewport: (u32, u32, u32, u32),
        has_color: bool,
        color_format: PixelFormat,
    ) {
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
        };
        let pipeline = self.get_or_create_clear_quad_pipeline(key);
        if pipeline == 0 {
            self.pass_state.clear_depth_legacy_break(value);
            return;
        }
        let depth_state = self.get_or_create_depth_stencil(1, 1, D3DCMP_ALWAYS);
        let z_bytes = f32::from_bits(value).to_le_bytes();
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
        self.emit_scissor_rect_resolved((vx, vy, vw, vh));
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
        // pass has one. On a no-depth pass (e.g. after an explicit
        // `SetDepthStencilSurface(NULL)`) a pipeline that declares depth is
        // rejected by Metal ("depth attachment pixelFormat must be Invalid, as
        // no texture is set"), so gate `HAS_DEPTH` on the bound attachment.
        let mut flags = ClearQuadFlags::HAS_COLOR;
        let has_depth = !self.current_depth_texture().is_null();
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
        let depth_state = self.get_or_create_depth_stencil(0, 0, D3DCMP_ALWAYS);
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
        let was_closed = self.pass_state.current_pass_closed();
        self.pass_state.ensure_pass_open();
        if was_closed {
            self.last_bound.reset();
            // Reset the debug-build emitted-command shadow in lockstep so the
            // in-sync assertion shares the same fresh-encoder baseline.
            #[cfg(debug_assertions)]
            self.pass_state.debug_reset_emitted();
        }
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

    /// Look up or create an `MTLDepthStencilState` for the given D3D9 params.
    ///
    /// `depth_func` is a D3DCMP_* value, translated to `MTLCompareFunction` here.
    pub fn get_or_create_depth_stencil(
        &mut self,
        depth_enable: u32,
        depth_write: u32,
        depth_func: u32,
    ) -> u64 {
        let key = DepthStencilKey::from_state(depth_enable, depth_write, depth_func);
        if let Some(&handle) = self.depth_stencil_cache.get(&key) {
            return handle.raw();
        }

        let metal_cmp = d3d_to_metal_cmp(depth_func);
        let mut params = CreateDepthStencilStateParams {
            device_handle: self.device_handle,
            depth_test_enable: depth_enable,
            depth_write_enable: depth_write,
            depth_compare_func: metal_cmp,
            id: key.raw(),
            state_handle: MetalHandle::NULL,
        };
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
        self.program_cache
            .entry(shader_id)
            .or_insert_with(|| Box::new(program));
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
                ..
            } => {
                if let Some(&handles) = self.prog_vs_libs.get(&(*vs_id, *provided_input_mask)) {
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
                ..
            } => {
                self.prog_vs_libs
                    .insert((*vs_id, *provided_input_mask), handles);
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
                ..
            } => {
                let Some(program) = self.program_cache.get(vs_id) else {
                    error!(target: LOG_TARGET, "VS {vs_id:#x} missing from program_cache");
                    return None;
                };
                let bucket = CompileBucket::from_sm_major(program.major);
                let msl =
                    match emit_vs_programmable_named(program, &entry_name, *provided_input_mask) {
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
            PsSource::FixedFunction { key } => {
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
            PsSource::FixedFunction { key } => {
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
            PsSource::FixedFunction { key } => (
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
            VertexSource::Bound {
                buffer_id, offset, ..
            } => format!("vb={buffer_id:#x}+{offset}"),
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
        if !with_color.is_null() && snapshot.rs.color_write_mask == 0 && snapshot.has_color_output()
        {
            // No-color twin: same identity except the attach flag.
            // Explicit `.clone()` because PipelineSnapshot is no longer
            // Copy; fires once per unique pipeline (cache hit thereafter).
            let mut alt = snapshot.clone();
            alt.attach
                .remove(mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_COLOR_OUTPUT);
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
        let mut params = pipeline_state::params_from_snapshot(&PipelineBuildInputs {
            snapshot,
            vertex_attrs,
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

    /// Look up or create an `MTLTexture` for the given texture ID (deferred creation).
    ///
    /// Cache hit returns immediately; cache miss goes through a one-element
    /// batched `CreateTexturesBatch` thunk — same wire path used by
    /// `drain_texture_warmups` when the API thread queued the texture at
    /// `CreateTexture` time.
    pub fn get_or_create_texture(&mut self, info: &TextureInfo) -> u64 {
        let texture_id = info.texture_id;
        let levels = info.levels;
        if let Some(state) = self.texture_cache.get(&texture_id) {
            return state.mtl_texture.raw();
        }

        let desc = Self::texture_desc_from_info(info);
        let mut handle = MetalHandle::<MTLTextureKind>::NULL;
        let status = self.batch_create_textures(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut handle),
        );
        if status != 0 || handle.is_null() {
            error!(target: LOG_TARGET, "encoder: CreateTexture failed");
            return 0;
        }
        self.texture_cache.insert(
            texture_id,
            TextureGpuState {
                mtl_texture: handle,
                mip_staging_buffers: vec![MipStagingBuffer::default(); levels as usize],
            },
        );
        handle.raw()
    }

    /// Wrap the bound VB or IB `PageBox` in a Shared `MTLBuffer` lazily on first Draw post-rename.
    ///
    /// Subsequent Draws within the same-backing window hit the cache. The
    /// rename itself is handled via `intake_vbib_retention` at the start of
    /// the subsequent frame.
    pub fn ensure_vbib_mtl_buffer(
        &mut self,
        buffer_id: BufferId,
        backing_ptr: u64,
        backing_len: u64,
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
                return state.device_buffer.raw();
            }
            if state.backing_ptr == backing_ptr && state.length == backing_len {
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
                last_submit_seq: current_seq,
            },
        );
        // Fresh wrapper around new (or renamed) backing — notify the
        // GPU about every byte the CPU may have written since the
        // backing was allocated. No-op on UMA via the helper's gate.
        self.enqueue_notify_buffer_did_modify_range(handle.raw(), 0, backing_len);
        handle.raw()
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
            Some(state) if state.backing_ptr == backing_ptr => {
                let removed = self.buffer_cache.remove(&buffer_id).expect("just checked");
                (
                    removed.mtl_buffer,
                    removed.last_submit_seq.max(last_submit_seq),
                )
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
        // PageBoxes inside `drained` drop here, after every wrapper
        // destroy thunk has returned.
        drop(drained);
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
    /// The non-expansion paths wrap the staging Box in a Shared
    /// `MTLBuffer` (lazy on first upload, cached per mip) and emit a
    /// `BlitCopyBufferToTexture` into `frame_blit_commands`. The
    /// `job.arc` clone is retained in `current_blit_retention` so the
    /// Box stays alive until the GPU retires the frame — blit reads
    /// happen at command-buffer execution time, long after this
    /// function returns.
    ///
    /// The expansion path (A4R4G4B4 / R5G6B5 / A1R5G5B5 → BGRA8) also
    /// blits: it expands into a fresh page-aligned staging `PageBox`,
    /// then emits the same `copyFromBuffer:toTexture:`. Every upload is
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
            if job.depth > 1 {
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
        if job.depth > 1 {
            self.run_volume_upload_blit(job, handle);
        } else {
            self.run_texture_upload_blit(job, handle);
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
        let desc = Self::texture_desc_from_info(info);
        let mut fresh = MetalHandle::<MTLTextureKind>::NULL;
        let status = self.batch_create_textures(
            core::slice::from_ref(&desc),
            core::slice::from_mut(&mut fresh),
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
        // wrappers key on the PE-side backing and are unaffected.
        if let Some(state) = self.texture_cache.get_mut(&info.texture_id) {
            state.mtl_texture = fresh;
        }
        // The old texture is read by this frame's already-emitted draws —
        // destroy only after the frame's GPU work retires.
        self.pending_resource_retention
            .push_back(PendingResourceRetention {
                kind: DestroyKind::Texture,
                handle: old_handle,
                page_box: None,
                staging_arc: None,
                seq: self.current_submit_seq,
                from_texture: true,
            });
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
    /// leading-blit encoder — no per-texture command buffer.
    pub fn run_generate_mipmaps(&mut self, texture_id: TextureId) {
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
        self.end_current_pass("autogen_rt_regen");
        self.push_stretch_rect_blit(BlitCommand::generate_mipmaps(handle));
    }

    /// Blit-based upload for non-expansion formats.
    ///
    /// Reuses the per-mip staging `MTLBuffer` (wrapping the game's staging
    /// `PageBox`) and emits a `BlitCopyBufferToTexture` against the frame's
    /// leading blit pass.
    fn run_texture_upload_blit(&mut self, job: TextureUploadJob, texture_handle: u64) {
        let _t = mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::TexRaw));
        let backing_length = job.arc.len() as u64;
        if backing_length == 0 {
            return;
        }

        // Compute the blit descriptor against the staging buffer's
        // src_pitch stride. `num_blit_rows` is the row count the GPU
        // will actually read — pixel rows for uncompressed, block rows
        // (rounded up) for compressed. Carried through alongside `info`
        // so the alignment-pad branch below knows how many source rows
        // to repack.
        let staging_buffer_handle =
            self.get_or_create_staging_buffer(job.info.texture_id, job.level as usize, &job.arc);
        if staging_buffer_handle == 0 {
            return;
        }

        let (info, num_blit_rows) = if job.bytes_per_pixel == 0 {
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
                    mip_level: job.level,
                    origin_x: job.origin_x,
                    origin_y: job.origin_y,
                    region_w: job.region_w,
                    region_h: job.region_h,
                    depth: 1,
                    bytes_per_image: 0,
                };
                (info, job.region_h.div_ceil(bh))
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
                    mip_level: job.level,
                    origin_x: 0,
                    origin_y: 0,
                    region_w: mip_w,
                    region_h: mip_h,
                    depth: 1,
                    bytes_per_image: 0,
                };
                (info, mip_h.div_ceil(bh))
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
                mip_level: job.level,
                origin_x: job.origin_x,
                origin_y: job.origin_y,
                region_w: job.region_w,
                region_h: job.region_h,
                depth: 1,
                bytes_per_image: 0,
            };
            (info, job.region_h)
        };

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
                None => return,
            }
        } else {
            // Notify the staging MTLBuffer (no-op on UMA). The padded
            // path notifies the transient buffer instead.
            self.enqueue_notify_buffer_did_modify_range(staging_buffer_handle, 0, backing_length);
            info
        };

        // Single-slice (`depth == 1`) copy: `bytes_per_image` is
        // `bytes_per_row * region_h` computed off the *final*
        // (post-padding) row stride — the exact value the unix blit
        // derived implicitly before the field existed, so the 2D wire is
        // byte-identical.
        let info = CopyBufferToTextureInfo {
            bytes_per_image: info.bytes_per_row.saturating_mul(info.region_h),
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
        // Refcount drops back to 1 when `pending_blit_retention`
        // releases this Arc after `coherent_seq >= submit_seq`.
        self.current_blit_retention.push(job.arc);
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
    /// `minimumLinearTextureAlignmentForPixelFormat`, every row across
    /// every slice is repacked to the padded stride (the rows being
    /// contiguous makes this a single `region_rows * depth` repack), and
    /// `bytes_per_image` widens to `padded_pitch * region_rows`.
    fn run_volume_upload_blit(&mut self, job: TextureUploadJob, texture_handle: u64) {
        let _t = mtld3d_core::perf::CycleAddTimer::start(self.op_sub_cycles_ptr(OpSub::TexRaw));
        let backing_length = job.arc.len() as u64;
        if backing_length == 0 {
            return;
        }
        let src_pitch = job.src_pitch;
        let slice_pitch = job.slice_pitch;
        let depth = job.depth.max(1);
        // Rows per slice (block-rows for compressed): `slice_pitch` is
        // exactly `src_pitch * block_rows`, so recover it by division.
        let region_rows = slice_pitch.checked_div(src_pitch).unwrap_or(0);
        if region_rows == 0 {
            return;
        }
        let mip_w = (job.info.width.max(1) >> job.level).max(1);
        let mip_h = (job.info.height.max(1) >> job.level).max(1);

        let staging_buffer_handle =
            self.get_or_create_staging_buffer(job.info.texture_id, job.level as usize, &job.arc);
        if staging_buffer_handle == 0 {
            return;
        }

        let info = CopyBufferToTextureInfo {
            buffer_handle: staging_buffer_handle,
            buffer_offset: 0,
            bytes_per_row: src_pitch,
            texture_handle,
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
                None => return,
            }
        } else {
            self.enqueue_notify_buffer_did_modify_range(staging_buffer_handle, 0, backing_length);
            info
        };

        self.frame_blit_commands
            .push(BlitCommand::copy_buffer_to_texture(&info));
        self.flags.insert(FrameEncoderFlags::BLIT_CMDS_NEED_ENCODER);
        self.perf.bump_texture_blit_upload();
        self.current_blit_retention.push(job.arc);
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

    /// Upload `tight` into the standalone colour `MTLTexture` `color_handle`.
    ///
    /// `tight` is `width * height * bpp` bytes of tight-packed rows; this is
    /// the `UnlockRect` half of a lockable render target (`CreateRenderTarget`
    /// with `Lockable == TRUE`). Copies the rows into a fresh page-aligned
    /// `PageBox` (padding each row up to `min_linear_texture_align` if the
    /// tight stride is below it), wrap that in a transient `MTLBuffer`, append
    /// a `CopyBufferToTexture` to the frame's leading blit pass, and retire
    /// both after the GPU retires this frame. The bytes are *copied* here (the
    /// caller's staging is not aliased across the API/encoder boundary).
    pub fn upload_bytes_to_color_handle(
        &mut self,
        color_handle: u64,
        tight: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) {
        if color_handle == 0 || width == 0 || height == 0 || bytes_per_pixel == 0 {
            return;
        }
        let tight_stride = (width as usize) * (bytes_per_pixel as usize);
        // `copyFromBuffer:toTexture:` requires `sourceBytesPerRow` ≥
        // `minimumLinearTextureAlignmentForPixelFormat:`; pad narrow rows up.
        let padded_stride = tight_stride.max(self.gpu_caps.min_linear_texture_align as usize);
        let Some(padded_size) = padded_stride.checked_mul(height as usize) else {
            error!(target: LOG_TARGET, "upload_bytes_to_color_handle: staging size overflow");
            return;
        };
        if tight.len() < tight_stride.saturating_mul(height as usize) {
            error!(
                target: LOG_TARGET,
                "upload_bytes_to_color_handle: source slice {} shorter than {tight_stride}*{height}",
                tight.len(),
            );
            return;
        }
        // `bytesNoCopy` needs page-aligned backing, so the rows must land in a
        // `PageBox` (re-packing the tight source into the padded stride).
        let mut staging = PageBox::new_uninit(padded_size);
        let dst_base = staging.as_mut_ptr();
        let src_base = tight.as_ptr();
        for row in 0..height as usize {
            // SAFETY: the source row `[row*tight_stride, +tight_stride)` is in
            // bounds (`tight.len() >= tight_stride * height`, checked above).
            let src_row = unsafe { src_base.add(row * tight_stride) };
            // SAFETY: the dest row `[row*padded_stride, +tight_stride)` is
            // within `staging` (`padded_stride >= tight_stride`, alloc has
            // `padded_size = padded_stride * height` bytes).
            let dst_row = unsafe { dst_base.add(row * padded_stride) };
            // SAFETY: both pointers and the byte count are valid per above, and
            // `tight` / `staging` are distinct allocations (disjoint copy).
            unsafe { core::ptr::copy_nonoverlapping(src_row, dst_row, tight_stride) };
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
    ) -> u64 {
        if let Some(Some(memo)) = self.sampler_resolve_memo.get(stage as usize)
            && memo.is_compare == is_compare
            && memo.state == *sampler_state
        {
            return memo.handle;
        }
        let snapshot = sampler_state::snapshot_from_state(sampler_state, is_compare);
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
            self.memoize_sampler_resolve(stage, sampler_state, is_compare, handle.raw());
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
        self.memoize_sampler_resolve(stage, sampler_state, is_compare, sampler.raw());
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
        for entry in self.pending_resource_retention.drain(..) {
            if entry.handle != 0 {
                match entry.kind {
                    DestroyKind::Buffer => buffers.push(entry.handle),
                    DestroyKind::Texture => textures.push(entry.handle),
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
    layer_handle: MetalHandle<CAMetalLayerKind>,
    /// `NSView*` the layer was attached to.
    ///
    /// Forwarded to `SubmitFrameParams.present_view` so the unix side can
    /// read the screen's *dynamic* EDR headroom each present (only used when
    /// HDR is active, which is decided unix-side and gated by the
    /// `macdrv::hdr_active()` global).
    view_handle: MetalHandle<NSViewKind>,
    backbuffer_width: u32,
    backbuffer_height: u32,
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

/// Parameter bag for `FrameData::new`.
///
/// Grouped so the constructor signature stays under clippy's
/// `too_many_arguments` threshold — pattern borrowed from `DeviceCreateInfo` /
/// `TextureCreateInfo`.
pub struct FrameInit {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub queue_handle: MetalHandle<MTLCommandQueueKind>,
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    pub layer_handle: MetalHandle<CAMetalLayerKind>,
    pub view_handle: MetalHandle<NSViewKind>,
    pub backbuffer_width: u32,
    pub backbuffer_height: u32,
    pub backbuffer_format: PixelFormat,
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
            layer_handle: init.layer_handle,
            view_handle: init.view_handle,
            backbuffer_width: init.backbuffer_width,
            backbuffer_height: init.backbuffer_height,
            backbuffer_format: init.backbuffer_format,
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

    pub const fn set_submit_fence(
        &mut self,
        submit_seq: u64,
        coherent_seq_ptr: u64,
        upload_coherent_seq_ptr: u64,
    ) {
        self.submit_seq = submit_seq;
        self.coherent_seq_ptr = coherent_seq_ptr;
        self.upload_coherent_seq_ptr = upload_coherent_seq_ptr;
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
        }
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

    /// Heavy alloc-recovery tier.
    ///
    /// Like `mid_frame_submit`, but the encoder also waits for GPU
    /// completion of the submitted seq and drains retention before
    /// signalling — when this returns the global allocator has the freed
    /// bytes back. Cost: ~1-2 ms of GPU completion + drain. Used only as
    /// a fallback when `drain_retired_now` already failed to free enough.
    pub fn mid_frame_submit_for_alloc(&self, frame: FrameData) {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        let _ = self.sender.send(EncoderMessage::MidFrameSubmitForAlloc {
            frame: Box::new(frame),
            done: done_tx,
        });
        let _ = done_rx.recv();
    }

    /// Cheap alloc-recovery tier.
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
    /// Heavy alloc-recovery tier.
    ///
    /// Run the frame, spin until our seq retires on the GPU (so
    /// `coherent_seq` covers our submission), then drain
    /// `pending_resource_retention` so freed bytes return to the global
    /// allocator before signalling done. Used by VB/IB `Lock`-rename when
    /// `try_new_uninit` returns null and the cheap-tier `DrainRetiredNow`
    /// already failed.
    MidFrameSubmitForAlloc {
        frame: Box<FrameData>,
        done: mpsc::SyncSender<()>,
    },
    /// Cheap alloc-recovery tier.
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
            Ok(EncoderMessage::MidFrameSubmitForAlloc { frame, done }) => {
                frame_counter += 1;
                mtld3d_shared::crumb!("phase:RecvMidAl");
                enc.drain_submit_thread();
                run_frame(&mut enc, frame, frame_counter, SubmitMode::Sync);
                // Wait for our just-submitted seq to retire on the GPU
                // so `coherent_seq` covers it; then drain so the freed
                // bytes are back in the global allocator before the
                // API thread retries `try_new_uninit`.
                enc.wait_for_gpu_idle();
                enc.drain_retired_resource_retention();
                let _ = done.send(());
            }
            Ok(EncoderMessage::DrainRetiredNow(done)) => {
                mtld3d_shared::crumb!("phase:RecvDrain");
                // Cheap tier: no barrier needed. A resource retired at seq N
                // whose async submit is still in flight has seq > coherent
                // (coherent only advances on GPU completion of committed
                // work), so the seq-gated drain can't free it early.
                enc.drain_retired_resource_retention();
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
        let (payload, status, drawable_wait_tsc) = execute_submit(params, payload);
        enc.perf.set_drawable_wait_cycles(drawable_wait_tsc);
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

    apply_pass_rules(enc);
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
        drawable_wait_tsc: 0,
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
/// cycles the thunk writes into `params`. This is the only part of submit
/// that runs on the dedicated submit thread in `Async` mode.
fn execute_submit(
    mut params: SubmitFrameParams,
    payload: FramePayload,
) -> (FramePayload, i32, u64) {
    let status = unix_call(&mut params);
    (payload, status, params.drawable_wait_tsc)
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
    let color_store_action = match p.color_store() {
        PassStoreAction::Store => StoreAction::Store,
        PassStoreAction::DontCare => StoreAction::DontCare,
    };
    let depth_store_action = match p.depth_store() {
        PassStoreAction::Store => StoreAction::Store,
        PassStoreAction::DontCare => StoreAction::DontCare,
    };
    log_pass_depth_attach(p);
    let leading = p.leading_blits();
    let visibility_result_buffer =
        if !visibility_buffer_handle.is_null() && p.has_counting_visibility() {
            visibility_buffer_handle
        } else {
            MetalHandle::NULL
        };
    PassDescriptor {
        color_texture: p.color_texture(),
        depth_texture: p.depth_texture(),
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
        command_count: u32::try_from(p.commands().len()).expect("per-pass command count fits u32"),
        leading_blits_count: u32::try_from(leading.len())
            .expect("per-pass leading blit count fits u32"),
        // Per-pass leading blits today are only StretchRect CopyTexture
        // commands (notifies go in the frame-level list), so any
        // non-empty leading list needs the encoder. If a future caller
        // threads notifies through here, switch to a tracked flag on
        // `Pass`.
        leading_blits_need_encoder: u32::from(!leading.is_empty()),
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
        depth_texture: MetalHandle::NULL,
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
        command_count: 0,
        leading_blits_count: u32::try_from(trailing_blits.len())
            .expect("trailing blit count fits u32"),
        leading_blits_need_encoder: 1,
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
fn apply_pass_rules(enc: &mut FrameEncoder) {
    enc.pass_state.coalesce_clear_only_passes();
    enc.pass_state.finalize_load_actions();
    enc.pass_state.finalize_store_actions();
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
            ..
        } => {
            format!(
                "prog {:#x}",
                draw::vs_source_disk_key_programmable(*vs_id, *provided_input_mask)
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
        PsSource::FixedFunction { key } => {
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
