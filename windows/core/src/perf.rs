//! Performance-counter and telemetry plumbing.
//!
//! Every TSC timer, per-frame counter, and the 5-second `info!` summary
//! lives here so it is visibly separate from the D3D9 / Metal runtime
//! state. None of the fields here exist for the game — they exist for the
//! developer who built with `PERF=1` (the summary then prints by default
//! at `info`; silence with `RUST_LOG=mtld3d::perf=warn`).
//!
//! Structure:
//! * `ApiCategory` + `ApiTimer`  — RAII guard that buckets TSC cycles
//!   by COM vtable category on every D3D9 entry point.
//! * `ApiPerfState` — embedded on `DeviceInner` (the API thread).
//!   Every counter the API thread bumps lands here.
//! * `FramePerfPayload` — embedded on `FrameData`. Copy of the
//!   API-thread counters that crosses the API→encoder channel.
//! * `EncoderPerfState` — embedded on `FrameEncoder`. Per-frame
//!   encoder counters + the rolling `PerfWindow` aggregator.
//! * `PerfWindow` + `FrameSample` (private) — 5-second rolling window
//!   that aggregates per-frame samples for the summary log. Time
//!   counters are emitted as per-frame averages (ms/frame); event
//!   counters as raw window totals; depth snapshots as decimal
//!   averages across the frame count; peak counters as per-frame
//!   max values observed over the window (distinguishes transient
//!   spikes from sustained cost).
//!
//! Nothing in this module touches COM, `raw-dylib`, or `DeviceInner`.
//! `ApiTimer` holds an `*mut ApiPerfState` so d3d9 is the only side
//! that knows about `DeviceInner` — callers compute the perf pointer
//! from their device pointer at construction time.

#[cfg(perf_tracking)]
use std::{collections::HashMap, fmt::Write as _, sync::LazyLock};

#[cfg(perf_tracking)]
use log::{info, trace};
/// Re-exports so existing `mtld3d_core::perf::*` paths in d3d9 keep working.
///
/// The gate + state-agnostic timers + latch fn moved to
/// [`mtld3d_shared::perf`].
pub use mtld3d_shared::perf::{
    CycleAddTimer, CycleSetTimer, init_tracking_enabled, pair_stats_enabled, perf_enabled,
};
#[cfg(perf_tracking)]
use mtld3d_shared::{
    CommandType,
    tsc::{cycles_to_ms, rdtsc, secs_to_cycles},
};
use mtld3d_shared::{MetalHandle, mtl_handle::MTLTextureKind};
// Brings `OpSub::COUNT` (the `strum::EnumCount` associated const) into scope
// for the `[_; OpSub::COUNT]` arrays below. Only referenced from
// perf-tracking code, so the import is elided under `not(perf_tracking)`.
#[cfg(perf_tracking)]
use strum::EnumCount;

use super::passes::Pass;
#[cfg(perf_tracking)]
use super::passes::{ColorLoad, DepthLoad};

/// Window length for the averaged `mtld3d::perf=debug` summary.
///
/// The per-pass / present-texture / per-pair detail rides on a separate
/// switch — `mtld3d::d3d9::passes=trace` — so the default
/// `RUST_LOG=info` never sees either.
pub const SUMMARY_INTERVAL_SECS: u64 = 5;

/// Perf telemetry has its own `log` target so the 5-second summary can be silenced.
///
/// Silencing it does not lose other COM-layer logging —
/// `RUST_LOG=mtld3d::perf=warn` mutes only this module.
#[cfg(perf_tracking)]
const LOG_TARGET: &str = "mtld3d::perf";

/// Pass / workload-shape detail lives on the existing `mtld3d::d3d9::passes` diag target.
///
/// Per-pass dump, `present_texture=…` audit line, and per-RT pair stats
/// sit alongside the per-event pass-break / pass-open probes in
/// `windows/core/src/passes.rs`. The per-frame `log_frame_summary` dump
/// just rides the same switch.
#[cfg(perf_tracking)]
const PASSES_TARGET: &str = "mtld3d::d3d9::passes";

/// Which COM wrapper type owns the vtable fn wrapping an `ApiTimer`.
///
/// Arranged so `as usize` indexes directly into the accumulator arrays
/// on `ApiPerfState`.
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum ApiCategory {
    Device = 0,
    VertexBuffer,
    IndexBuffer,
    Texture,
    Surface,
    Query,
    StateBlock,
    VertexDecl,
    VertexShader,
    PixelShader,
}

/// Sub-bucket inside the `Device` `ApiCategory`.
///
/// Every `extern "system"` `IDirect3DDevice9` entry point tags its
/// `ApiTimer` with a `DeviceSubCategory` so the 5-second summary can
/// decompose the `Device` row into "where did the cycles actually go" —
/// draws vs the per-batch state-setter storms (`RenderState` /
/// `TexStageState` / `SamplerState` / `ShaderConst`) vs binds vs
/// frame boundaries vs state-block recording vs the long tail.
///
/// `Misc` is the catch-all so the sub-buckets sum exactly to
/// `api_cycles_by_category[Device]` — assertable from tests.
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum DeviceSubCategory {
    Draws = 0,
    RenderState,
    TexStageState,
    SamplerState,
    ShaderConst,
    Bind,
    Frame,
    StateBlock,
    Misc,
}

/// Sub-bucket inside the `Bind` `DeviceSubCategory`.
///
/// Every `IDirect3DDevice9` entry point whose `device_timer` tag is
/// `Bind` instead uses `bind_timer` and supplies a `BindSubCategory`,
/// so the 5-second summary can decompose the 0.15 ms/frame `Bind` row
/// into "which Setter family ate the cycles": resource bindings
/// (`Texture` / `Buffer` / `Shader`), render-target swaps (`RtDs`),
/// fixed-function state (`FfFixed`), and viewport/scissor (`ViewScissor`).
///
/// The 6 buckets sum exactly to `device_sub_cycles[Bind]` — every Bind
/// method belongs to exactly one bucket, no `Misc` escape hatch.
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum BindSubCategory {
    /// `Set/GetTexture`.
    Texture = 0,
    /// `Set/GetStreamSource`, `Set/GetStreamSourceFreq`, `Set/GetIndices`.
    Buffer,
    /// `Set/GetVertexDeclaration`, `Set/GetVertexShader`, `Set/GetPixelShader`, `Set/GetFVF`.
    Shader,
    /// `Set/GetRenderTarget`, `Set/GetDepthStencilSurface`.
    ///
    /// The three methods already pushing a Closure Op directly to the
    /// encoder.
    RtDs,
    /// Fixed-function state setters: transform, material, light, clip plane.
    ///
    /// `Set/Get/MultiplyTransform`, `Set/GetMaterial`, `Set/GetLight`,
    /// `LightEnable/GetLightEnable`, `Set/GetClipPlane`. Small struct
    /// writes into `FfState`; sparser than resource swaps.
    FfFixed,
    /// `Set/GetViewport`, `Set/GetScissorRect`.
    ViewScissor,
}

/// Per-draw snapshot dirty-mark gate whose redundant-skip rate the perf summary reports.
///
/// Each variant indexes the `keys_gate_calls` / `keys_gate_skips` arrays:
/// every live (non-state-block) call to the setter bumps `calls`, and the
/// subset that left the FF VS/PS keys unchanged — and so skipped the
/// snapshot rebuild billed to the `keys` bucket — bumps `skips`. Defined
/// unconditionally so the COM thunks can name a variant regardless of
/// `cfg(perf_tracking)`.
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum KeysGate {
    /// `SetTexture`.
    ///
    /// Skip when the new texture leaves the slot occupancy mask and
    /// depth-format-ness unchanged.
    SetTexture = 0,
    /// `SetRenderState` — skip when the written value is unchanged.
    SetRenderState,
    /// `SetTextureStageState` — skip when the written value is unchanged.
    SetTextureStageState,
    /// `SetFVF` — skip when the FVF value is unchanged (gates the VDECL attrs re-resolve).
    SetFvf,
    /// `SetVertexDeclaration` — skip when the bound decl pointer is unchanged.
    SetVertexDecl,
    /// `SetVertexShader` — skip when the bound VS pointer is unchanged.
    SetVertexShader,
    /// `SetPixelShader` — skip when the bound PS pointer is unchanged.
    SetPixelShader,
    /// `SetVertexShaderConstantF`.
    ///
    /// Skip when every written constant row is byte-identical to the
    /// mirror.
    SetVsConst,
    /// `SetPixelShaderConstantF`.
    ///
    /// Skip when every written constant row is byte-identical to the
    /// mirror.
    SetPsConst,
}

/// Per-draw phase of the encoder op loop (the "Closures (op)" bucket).
///
/// Each variant indexes the `op_sub_cycles` accumulator so the 5-second
/// summary can decompose `emit_draw`'s per-draw cost — the way
/// `DeviceSubCategory` decomposes the API-thread `Device` row. The six
/// phases tile `emit_draw` end to end; whatever the timers don't cover (the
/// non-`Draw` ops and the trailing per-pair stat bump) shows up as the
/// rendered `resid` row. Defined unconditionally so `emit_draw` can name a
/// variant regardless of `cfg(perf_tracking)`.
///
/// `COUNT` comes from `strum::EnumCount` (fully-qualified derive path so it
/// resolves under `not(perf_tracking)` too, where the trait import is
/// elided); add a variant and every `[_; OpSub::COUNT]` array tracks it.
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum OpSub {
    /// Snapshot deref + const-scratch + bound-texture handle lookups.
    ///
    /// Plus the VS/PS shader-library cache lookups.
    Resolve = 0,
    /// Pipeline-state key build + cache lookup, depth-stencil state, cull.
    Pipeline,
    /// Pass open + scissor + pipeline/depth/cull/depth-bias binds.
    State,
    /// Decal/caster diagnostic probe (gated; ~0 at default log level).
    Probe,
    /// Per-stage texture + sampler-state resolve and bind.
    Samplers,
    /// Shader-constant slot binds + VB/IB wrap + the draw command.
    Binds,
    // The variants below are NON-draw ops. They are timed inside their
    // worker methods (not `emit_draw`), so they decompose what used to fall
    // entirely into the Closures `resid` row. They never overlap each other
    // or the draw phases above, so `resid` = op total − sum(all variants)
    // stays a true "everything else" (RT swap, clear, present, snapshot
    // memcpy, resource lifecycle, noise).
    /// Texture upload: `bytesNoCopy` staging → `MTLTexture` blit (`run_texture_upload_blit`).
    TexRaw,
    /// Inline op-stream-ordered VB/IB `StageUpload` apply (`apply_stage_upload`).
    StageUpload,
    /// VS/PS/FF programmable + FF const-range delta memcpy into the encoder mirrors.
    ///
    /// Applied by `apply_*_const_range`.
    ConstRange,
}

/// Second-level breakdown of the two dominant [`OpSub`] phases.
///
/// `Resolve` and `Binds` each render as nested children so the summary can
/// attribute the per-draw cost that remains after `FxHash` removed the
/// cache-probe cost (FF const copies, key hashing, the `LastBoundCache`
/// `memcmp`, VB/IB wrapping, draw emit). `R*` are children of `Resolve`, `B*`
/// of `Binds`; the parent total minus its children renders as a per-parent
/// `resid`. Same zero-cost-when-off contract as [`OpSub`].
#[derive(Clone, Copy, Debug, strum::EnumCount)]
#[repr(u32)]
pub enum OpSubDetail {
    /// `Resolve`: VS/PS/FF shader-constant copy into scratch + the inline slices.
    RConsts = 0,
    /// `Resolve`: the `debug.skipShaders` skip-set check.
    ///
    /// The per-draw key clone + `pair_id` hashing it once measured moved
    /// off the hot path into the source-keyed cache lookup; this reads ~0
    /// unless skip is armed.
    RKeys,
    /// `Resolve`: bound-texture handle gather + VS/PS library cache lookups.
    RLookup,
    /// `Binds`: shader-constant slot `*_changed` `memcmp` + `set_*_bytes`.
    BCbind,
    /// `Binds`: vertex-source bind — VB `MTLBuffer` wrap + notify + draw-range.
    BVbib,
    /// `Binds`: index-buffer wrap + the `draw{,_indexed}_primitives` command.
    BDraw,
}

/// RAII guard that measures TSC cycles spent inside a D3D9 COM vtable entry point.
///
/// Adds the cycles to the owning device's `ApiPerfState`.
///
/// Constructed at the top of every `extern "system"` fn; `Drop` handles every
/// return path (happy, early-INVALIDCALL, panic unwind) without
/// per-return-site bookkeeping.
///
/// Null `perf_ptr` *or* perf-tracking disabled both skip the rdtsc and
/// writeback — standalone resources (e.g. backbuffer surfaces from
/// `GetBackBuffer`) stay safe, and the per-call cost is one Relaxed
/// load + branch when the user isn't running with
/// `RUST_LOG=mtld3d::perf=debug`.
///
/// Under `cfg(not(perf_tracking))` this collapses to a unit struct with no
/// `Drop` — the entire timer disappears at compile time via LLVM DCE on
/// every `let _timer = device_timer(...)` call site.
#[cfg(perf_tracking)]
pub struct ApiTimer {
    start: u64,
    perf_ptr: *mut ApiPerfState,
    category: ApiCategory,
    /// Set on `Device`-category timers built via `start_device`.
    ///
    /// Bumps both the top-level `Device` bucket and a per-sub-category
    /// breakdown so the summary can show where Device cycles concentrate.
    device_sub: Option<DeviceSubCategory>,
    /// Set on `Bind`-sub-category timers built via `start_bind`.
    ///
    /// Bumps the same Device top + `device_sub_cycles[Bind]` as a regular
    /// `start_device(Bind)` timer would, plus the per-`BindSubCategory`
    /// breakdown row. Always implies `device_sub == Some(Bind)`.
    bind_sub: Option<BindSubCategory>,
    /// The parent timer's `active_child_cycles` accumulator.
    ///
    /// Saved at `start` and restored (plus this timer's own `elapsed`) at
    /// `Drop`. Powers exclusive-time accounting: a vtable entry point that
    /// delegates into another entry point (e.g. `IDirect3DSurface9::LockRect`
    /// → the texture `LockRect` thunk) must not double-count the nested
    /// interval into both top-level buckets. Each bucket accrues only its
    /// own *self* time. See [`ApiPerfState::active_child_cycles`].
    saved_child_cycles: u64,
    enabled: bool,
}

#[cfg(not(perf_tracking))]
pub struct ApiTimer;

impl ApiTimer {
    /// Enter a child timing scope.
    ///
    /// Save the parent's child-cycle accumulator, zero it so this timer
    /// measures only *its own* children, and bump the nesting depth.
    /// Returns the saved parent value for the timer to stash. No-op
    /// (returns 0) when disabled.
    #[cfg(perf_tracking)]
    #[inline]
    fn enter_scope(perf_ptr: *mut ApiPerfState, enabled: bool) -> u64 {
        if !enabled {
            return 0;
        }
        // SAFETY: the API thread is single-threaded; this transient
        // `&mut` never overlaps another live borrow — parent timers
        // touch the perf state only inside their own `start`/`Drop`,
        // which are strictly nested around this call.
        let perf = unsafe { &mut *perf_ptr };
        let saved = perf.active_child_cycles;
        perf.active_child_cycles = 0;
        perf.timer_depth = perf.timer_depth.saturating_add(1);
        saved
    }

    #[cfg(perf_tracking)]
    pub fn start(perf_ptr: *mut ApiPerfState, category: ApiCategory) -> Self {
        let enabled = !perf_ptr.is_null() && perf_enabled();
        let saved_child_cycles = Self::enter_scope(perf_ptr, enabled);
        let start = if enabled { rdtsc() } else { 0 };
        Self {
            start,
            perf_ptr,
            category,
            device_sub: None,
            bind_sub: None,
            saved_child_cycles,
            enabled,
        }
    }

    #[cfg(not(perf_tracking))]
    #[inline]
    #[must_use]
    pub const fn start(_perf_ptr: *mut ApiPerfState, _category: ApiCategory) -> Self {
        Self
    }

    /// Like `start`, but tags the timer with a `DeviceSubCategory`.
    ///
    /// On `Drop`, the elapsed cycles are added to both the top-level
    /// `Device` bucket and the matching sub-bucket under a single
    /// `rdtsc()` delta. Used by every `IDirect3DDevice9` vtable thunk.
    #[cfg(perf_tracking)]
    pub fn start_device(perf_ptr: *mut ApiPerfState, sub: DeviceSubCategory) -> Self {
        let enabled = !perf_ptr.is_null() && perf_enabled();
        let saved_child_cycles = Self::enter_scope(perf_ptr, enabled);
        let start = if enabled { rdtsc() } else { 0 };
        Self {
            start,
            perf_ptr,
            category: ApiCategory::Device,
            device_sub: Some(sub),
            bind_sub: None,
            saved_child_cycles,
            enabled,
        }
    }

    #[cfg(not(perf_tracking))]
    #[inline]
    #[must_use]
    pub const fn start_device(_perf_ptr: *mut ApiPerfState, _sub: DeviceSubCategory) -> Self {
        Self
    }

    /// Like `start_device(Bind)`, but additionally tags the timer with a `BindSubCategory`.
    ///
    /// The tag lets the summary decompose the `Bind` row. On `Drop`, a
    /// single `rdtsc()` delta bumps the Device top bucket,
    /// `device_sub_cycles[Bind]`, AND `bind_sub_cycles[sub]` in one pass.
    /// Used by `IDirect3DDevice9::Set*` / `Get*` thunks whose
    /// `DeviceSubCategory` would otherwise be `Bind`.
    #[cfg(perf_tracking)]
    pub fn start_bind(perf_ptr: *mut ApiPerfState, sub: BindSubCategory) -> Self {
        let enabled = !perf_ptr.is_null() && perf_enabled();
        let saved_child_cycles = Self::enter_scope(perf_ptr, enabled);
        let start = if enabled { rdtsc() } else { 0 };
        Self {
            start,
            perf_ptr,
            category: ApiCategory::Device,
            device_sub: Some(DeviceSubCategory::Bind),
            bind_sub: Some(sub),
            saved_child_cycles,
            enabled,
        }
    }

    #[cfg(not(perf_tracking))]
    #[inline]
    #[must_use]
    pub const fn start_bind(_perf_ptr: *mut ApiPerfState, _sub: BindSubCategory) -> Self {
        Self
    }
}

/// Pure arithmetic of [`ApiTimer`]'s exclusive-time `Drop`.
///
/// Inputs: `elapsed` (this timer's raw TSC delta), `children` (cycles its
/// nested timers accumulated), `saved` (the parent accumulator stashed
/// at entry), and `depth` (the nesting depth *after* decrementing this
/// timer). Returns `(self_time, restored_accumulator)`:
/// - `self_time = elapsed − children`, saturating (clock noise can make
///   a nested delta momentarily exceed the parent's; clamp to 0 rather
///   than wrap). This is the value booked to this timer's bucket.
/// - `restored_accumulator` hands this timer's full `elapsed` up to the
///   parent (`saved + elapsed`), or resets to 0 at the outermost level
///   (`depth == 0`) so the accumulator is 0 whenever no timer runs.
///
/// Invariant across a balanced nest: the sum of every timer's
/// `self_time` equals the outermost timer's `elapsed` (no double-count).
#[cfg(perf_tracking)]
#[inline]
const fn exclusive_exit(elapsed: u64, children: u64, saved: u64, depth: u32) -> (u64, u64) {
    let self_time = elapsed.saturating_sub(children);
    let restored = if depth == 0 {
        0
    } else {
        saved.saturating_add(elapsed)
    };
    (self_time, restored)
}

#[cfg(perf_tracking)]
impl Drop for ApiTimer {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let elapsed = rdtsc() - self.start;
        // SAFETY: the owning extern fn holds the timer for its own
        // duration; the COM object (and therefore the `ApiPerfState`
        // it points into) cannot be freed while the game is inside
        // one of its methods.
        let perf = unsafe { &mut *self.perf_ptr };
        // Exclusive (self) time: subtract the cycles consumed by nested
        // timers (delegated D3D9 entry points) so their interval lands
        // in their own bucket, not double-counted into ours too. Then
        // hand our full `elapsed` up to the parent's accumulator —
        // unless we are the outermost timer, in which case reset to 0
        // so the accumulator is always 0 at rest (no residue across
        // top-level calls). See [`exclusive_exit`].
        let my_children = perf.active_child_cycles;
        perf.timer_depth = perf.timer_depth.saturating_sub(1);
        let (self_time, restored) = exclusive_exit(
            elapsed,
            my_children,
            self.saved_child_cycles,
            perf.timer_depth,
        );
        perf.active_child_cycles = restored;
        if let Some(bind) = self.bind_sub {
            // `start_bind` always pairs with `device_sub == Bind`; the
            // helper bumps Device top + Bind device-sub + bind-sub all
            // under the single rdtsc delta.
            perf.add_bind_cycles(bind, self_time);
        } else if let Some(sub) = self.device_sub {
            perf.add_device_cycles(sub, self_time);
        } else {
            perf.add_api_cycles(self.category, self_time);
        }
    }
}

// Empty Drop under `not(perf_tracking)` so any call site that holds the
// timer via `let _timer = ...` keeps consistent type semantics across
// cfgs. LLVM DCEs the empty drop body after inlining.
#[cfg(not(perf_tracking))]
impl Drop for ApiTimer {
    fn drop(&mut self) {}
}

/// API-thread-bumped per-frame counters — the set drained at `Present`.
///
/// One home for the counters that flow unchanged through the whole
/// pipeline: bumped on the API thread (`ApiPerfState`), moved across the
/// channel (`FramePerfPayload`), seeded encoder-side (`EncoderPerfState`),
/// and snapshotted for the window (`FrameSample`). Embedding this struct
/// in each stage turns the per-field hand-copies in `drain_into_payload` /
/// `begin_frame` / the `FrameSample` build into single struct moves, and a
/// new counter is one field here instead of one in every stage.
#[cfg(perf_tracking)]
#[derive(Clone, Copy)]
struct FrameCounters {
    /// TSC cycles this frame, bucketed by `ApiCategory`.
    ///
    /// Accumulated on the API thread inside every D3D9 COM vtable entry
    /// point.
    api_cycles_by_category: [u64; ApiCategory::COUNT],
    /// Per-category call counts for the same frame window.
    api_call_counts_by_category: [u32; ApiCategory::COUNT],
    /// `PageBox` allocations from VB/IB Lock-rename this frame.
    vb_rename: u32,
    ib_rename: u32,
    /// Subset of `vb_rename` / `ib_rename` that took the no-preserve branch.
    ///
    /// Either explicit `D3DLOCK_DISCARD`, or whole-buffer
    /// `D3DUSAGE_WRITEONLY` contended (game writes every byte; old
    /// contents need not survive). Partitions every rename together
    /// with the preserve counter: `rename = discards + preserve_cpu` is
    /// an invariant.
    vb_discards: u32,
    ib_discards: u32,
    /// Count of VB/IB Lock calls this frame that took the synchronous CPU-memcpy preserve path.
    ///
    /// Whole-buffer non-WRITEONLY contended Locks (the game might read
    /// the entire buffer through the Lock pointer). The bump sits next to
    /// the memcpy itself, so a non-zero count means the preserve actually
    /// ran — not merely that a rename was planned.
    vbib_preserve_cpu: u32,
    /// VB/IB Lock-rename hits where `try_new_uninit` returned null.
    ///
    /// The cheap `DrainRetiredNow` recovery freed enough for the retry to
    /// succeed.
    alloc_recovery_drain: u32,
    /// VB/IB Lock-rename hits where `try_new_uninit` returned null and the cheap drain failed.
    ///
    /// The cheap drain didn't free enough, so the heavy
    /// `MidFrameSubmitForAlloc` recovery (commit + GPU wait + drain)
    /// was needed. ~1-2 ms per fire — surfaced separately so a
    /// burst frame's worth of these is visible in the perf summary.
    alloc_recovery_submit: u32,
    /// Count of texture `LockRect` calls that allocated a fresh uninit staging Box this frame.
    ///
    /// The `FreshBox` branch of `decide_lock_action`.
    texture_renames: u32,
    /// Subset of `texture_renames` that took the no-preserve branch.
    ///
    /// Either explicit `D3DLOCK_DISCARD`, or a whole-mip
    /// `D3DUSAGE_DYNAMIC` contended Lock (the texture-side
    /// "no readback expected" hint; `WRITEONLY` is the analog for
    /// VB/IB but not documented for `CreateTexture`).
    texture_discards: u32,
    /// Count of texture `LockRect` calls this frame that synchronously preserved the staging Box.
    ///
    /// The non-DISCARD non-READONLY non-DYNAMIC contended-rename path
    /// **synchronously memcpied** the old staging Box forward on
    /// the API thread. No GPU-blit variant exists because texture
    /// staging is PE-side `Box<[u8]>`, not an `MTLBuffer` — a GPU blit
    /// can't populate it, and `MTLTexture` handles are never swapped on
    /// rename, so there's nothing for a `copyFromTexture:toTexture:`
    /// path to preserve.
    texture_preserve_cpu: u32,
    /// Count of `AddDirtyRect` calls this frame (`texture_add_dirty_rect` thunk).
    ///
    /// With `texture_add_dirty_partial` and
    /// `texture_add_dirty_area_bp` this characterizes whether the game
    /// declares a changed sub-region we could use to shrink the whole-mip
    /// preserve memcpy into a dirty-rect snapshot upload.
    texture_add_dirty_calls: u32,
    /// Subset of `texture_add_dirty_calls` whose rect is a *usable* sub-region.
    ///
    /// Strictly narrower than the level-0 surface. A `Some(rect)`
    /// covering the whole mip, or a `None`/whole-mip call, does not count.
    texture_add_dirty_partial: u32,
    /// Running sum of per-call dirty-rect area in basis points of the level-0 mip area.
    ///
    /// Whole-mip / `None` = 10000. Divided by `texture_add_dirty_calls`
    /// at render to report average coverage.
    texture_add_dirty_area_bp: u32,
    /// TSC cycles the API thread spent blocked in `IDirect3DQuery9::GetData(D3DGETDATA_FLUSH)`.
    ///
    /// Waiting on Metal's `MTLCommandBuffer::waitUntilCompleted` —
    /// accumulated by `CycleAddTimer` because a single frame can contain
    /// multiple FLUSH polls that each block.
    query_wait_cycles: u64,
    /// Per-sub-category cycle bucket inside the `Device` `ApiCategory`.
    ///
    /// Bumped by `ApiTimer::start_device` on every `IDirect3DDevice9`
    /// vtable thunk; sums to `api_cycles_by_category[Device]` within
    /// the same frame.
    device_sub_cycles: [u64; DeviceSubCategory::COUNT],
    /// Companion call-count array for `device_sub_cycles`.
    ///
    /// One bump per Device entry per call, regardless of which sub-bucket
    /// fires. Drives the per-sub-row `( N calls)` aux cell in the summary.
    device_sub_calls: [u32; DeviceSubCategory::COUNT],
    /// Per-sub-category cycle bucket inside `Bind`.
    ///
    /// Bumped by `ApiTimer::start_bind` on every `IDirect3DDevice9`
    /// Bind-family vtable thunk; sums to `device_sub_cycles[Bind]` within
    /// the same frame. Decomposes the Bind row into resource swaps vs
    /// RT/DS vs FF-state vs viewport/scissor so the next optimisation can
    /// target the dominant Setter family.
    bind_sub_cycles: [u64; BindSubCategory::COUNT],
    /// Companion call-count array for `bind_sub_cycles`.
    bind_sub_calls: [u32; BindSubCategory::COUNT],
    /// Per-[`KeysGate`] live setter-call count this frame.
    ///
    /// Denominator for the redundant-skip rate in the perf summary.
    keys_gate_calls: [u32; KeysGate::COUNT],
    /// Subset of `keys_gate_calls` that left the FF VS/PS keys unchanged.
    ///
    /// So they skipped the snapshot rebuild (the `keys` bucket work).
    keys_gate_skips: [u32; KeysGate::COUNT],
    /// TSC cycles inside the Draw methods spent in the snapshot phase.
    ///
    /// The phase: `snapshot_bound_vertex_source`,
    /// `snapshot_bound_index_source`, `snapshot_shared`. Accumulated by
    /// `CycleAddTimer` so all Draw entry points share a single bucket.
    draw_snapshot_cycles: u64,
    /// Sub-component of `draw_snapshot_cycles` covering the per-stage binding walk.
    ///
    /// Just the walk inside `snapshot_shared` (`snapshot_stage_bindings` —
    /// 8 stages × {texture, sampler state, TSS slots} with lazy upload
    /// dispatch).
    draw_snapshot_stages_cycles: u64,
    /// Sub-component of `draw_snapshot_cycles` covering the consts snapshot for FF draws.
    ///
    /// The block runs for draws where at least one shader stage is
    /// Fixed-Function (i.e. `bound_vs.is_null() || bound_ps.is_null()`).
    /// `c_ff + c_pr` sums to what the old `consts` row did.
    draw_snapshot_c_ff_cycles: u64,
    /// Sub-component of `draw_snapshot_cycles` covering the consts snapshot block.
    ///
    /// The programmable peer of `draw_snapshot_c_ff_cycles`: the block
    /// runs for draws where both VS and PS are programmable.
    draw_snapshot_c_pr_cycles: u64,
    /// Sub-component of `draw_snapshot_cycles` covering the shader-key resolution block.
    ///
    /// In `snapshot_shared`: `VDECL` → attrs/stride, `RS` snapshot struct
    /// build, `RT_DS` resolve, `VARIANT` key, `VS_SOURCE` (FF key build
    /// or programmable `shader_id`), and `PS_SOURCE`.
    draw_snapshot_keys_cycles: u64,
    /// Sub-component of `draw_snapshot_cycles` covering the post-consts bumps.
    ///
    /// The scratch bumps + cache assignments + snapshot-wrapper bump in
    /// `snapshot_shared` (the work after `drop(consts_timer)`).
    draw_snapshot_bumps_cycles: u64,
    /// TSC cycles inside the Draw methods spent in the push-op phase.
    ///
    /// `Box::new` of the `emit_draw` closure plus `push_op` append onto
    /// the current frame's op list. Closure-build cost.
    draw_push_op_cycles: u64,
}

#[cfg(perf_tracking)]
impl Default for FrameCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl FrameCounters {
    const fn new() -> Self {
        Self {
            api_cycles_by_category: [0; ApiCategory::COUNT],
            api_call_counts_by_category: [0; ApiCategory::COUNT],
            vb_rename: 0,
            ib_rename: 0,
            vb_discards: 0,
            ib_discards: 0,
            vbib_preserve_cpu: 0,
            alloc_recovery_drain: 0,
            alloc_recovery_submit: 0,
            texture_renames: 0,
            texture_discards: 0,
            texture_preserve_cpu: 0,
            texture_add_dirty_calls: 0,
            texture_add_dirty_partial: 0,
            texture_add_dirty_area_bp: 0,
            query_wait_cycles: 0,
            device_sub_cycles: [0; DeviceSubCategory::COUNT],
            device_sub_calls: [0; DeviceSubCategory::COUNT],
            bind_sub_cycles: [0; BindSubCategory::COUNT],
            bind_sub_calls: [0; BindSubCategory::COUNT],
            keys_gate_calls: [0; KeysGate::COUNT],
            keys_gate_skips: [0; KeysGate::COUNT],
            draw_snapshot_cycles: 0,
            draw_snapshot_stages_cycles: 0,
            draw_snapshot_c_ff_cycles: 0,
            draw_snapshot_c_pr_cycles: 0,
            draw_snapshot_keys_cycles: 0,
            draw_snapshot_bumps_cycles: 0,
            draw_push_op_cycles: 0,
        }
    }
}

/// Payload-injected per-frame timing.
///
/// Set on the channel payload (not bumped on the API thread), then
/// carried encoder-side into the sample.
#[cfg(perf_tracking)]
#[derive(Clone, Copy)]
struct FrameTiming {
    /// TSC cycles the API thread spent blocked on `sync_channel(1).send()`.
    ///
    /// Inside `Present`, waiting for the encoder thread to drain the
    /// previous frame — the backpressure signal.
    present_block_cycles: u64,
    /// TSC delta between the previous `Present` entry and this frame's `Present` entry.
    ///
    /// The API thread's wall-clock frame period. Zero on the very first
    /// frame (no predecessor).
    frame_total_cycles: u64,
    /// `FrameData.ops.capacity() * size_of::<Op>()` sampled at `stamp_and_swap`.
    ///
    /// The API→encoder `Vec<Op>`'s resident footprint.
    op_vec_capacity_bytes: u64,
    /// Per-frame total drained from `FrameData::take_op_vec_realloc_bytes()` at `stamp_and_swap`.
    ///
    /// Bytes Rust's `Vec::push` memcpy'd when `FrameData.ops` doubled.
    op_vec_realloc_bytes: u64,
}

#[cfg(perf_tracking)]
impl Default for FrameTiming {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl FrameTiming {
    const fn new() -> Self {
        Self {
            present_block_cycles: 0,
            frame_total_cycles: 0,
            op_vec_capacity_bytes: 0,
            op_vec_realloc_bytes: 0,
        }
    }
}

/// Encoder-bumped per-frame counters that flow into the sample.
///
/// Reset each `begin_frame` via `Default` (one move replaces the
/// per-field zeroing). Running totals and the `per_pair_stats` map stay
/// on `EncoderPerfState` because they persist across frames / are
/// non-`Copy`.
#[cfg(perf_tracking)]
#[derive(Clone, Copy)]
struct EncoderFrameCounters {
    /// Destroys of non-`from_texture` `MTLBuffer` wrappers on retention drain.
    ///
    /// The drain sources: VB/IB cache renames, Lock-rename / Release
    /// intake, visibility-pool eviction.
    buffer_destroys: u32,
    /// Destroys of texture-staging `MTLBuffer` wrappers on retention drain.
    ///
    /// Kept apart from `buffer_destroys` so each section's `destroys` row
    /// reflects only its own work.
    texture_destroys: u32,
    /// Per-frame count of `Staged` VB/IB dirty-range uploads the encoder emitted.
    ///
    /// `copy_buffer_to_buffer` staging→device in `apply_stage_upload`.
    /// High here with `rename ≈ 0` confirms world-geometry locks upload
    /// dirty ranges instead of renaming.
    vbib_staging_uploads: u32,
    /// Per-frame count of `Staged` VB/IB uploads that hit rename-at-overlap.
    ///
    /// The dirty range overlapped a region a draw already read earlier
    /// this frame, so `apply_stage_upload` RENAMED the device buffer to
    /// preserve that draw's bytes. The rare path; the common case is a
    /// cheap in-place upload.
    vbib_mid_pass_reorders: u32,
    /// Per-frame total texture uploads.
    ///
    /// Count of `run_texture_upload_blit` invocations that emitted a
    /// `copy_buffer_to_texture` command.
    texture_blit_uploads: u32,
    /// Per-frame count of `run_texture_upload_blit` invocations taking the padded-staging path.
    ///
    /// Taken because the source row pitch was below
    /// `gpu_caps.min_linear_texture_align`. Subset of
    /// `texture_blit_uploads`.
    texture_blit_padded_uploads: u32,
    /// Per-frame count of texture rename-at-overlap.
    ///
    /// An upload whose target texture was already sampled by a draw
    /// earlier this frame, so the encoder renamed the `MTLTexture` (fresh
    /// handle for later draws, earlier draws keep the old one) instead of
    /// letting the frame-head blit rewrite what the earlier draw reads.
    /// The texture analogue of `vbib_mid_pass_reorders`.
    texture_gpu_renames: u32,
    /// Total TSC cycles the encoder spent replaying this frame's op list.
    op_cycles: u64,
    /// Per-[`OpSub`] decomposition of `op_cycles`.
    ///
    /// Accumulated by the per-draw `CycleAddTimer`s inside `emit_draw`.
    op_sub_cycles: [u64; OpSub::COUNT],
    /// Second-level breakdown of the `Resolve`/`Binds` phases — see [`OpSubDetail`].
    ///
    /// Same lifecycle as `op_sub_cycles`.
    op_sub_detail: [u64; OpSubDetail::COUNT],
    /// Pipeline-resolve memo effectiveness.
    ///
    /// `calls` counts every `get_or_create_pipeline` invocation; `hits`
    /// the subset the single-entry snapshot memo served without
    /// rebuilding the key.
    pipeline_memo_hits: u32,
    pipeline_memo_calls: u32,
    /// Encoder-thread submit cost.
    ///
    /// Finalize (close passes, build descriptors, swap buffers, hand off)
    /// plus any backpressure stall waiting for the submit thread.
    submit_cycles: u64,
    /// Submit-thread `nextDrawable` (GPU + compositor) wait.
    ///
    /// Measured on the unix side and folded back when the payload
    /// returns. Lagged ≤1 frame under async.
    drawable_wait_cycles: u64,
    /// Submit-thread total for `execute_submit`, folded back on payload return.
    ///
    /// Command-walk + present + commit, incl. `drawable_wait_cycles`.
    /// `submit_exec - drawable_wait` is the encode+commit CPU.
    submit_exec_cycles: u64,
    /// Encoder backpressure stall.
    ///
    /// Time spent blocked in `acquire_clean_payload` waiting for the
    /// submit thread to return a payload. Part of the encoder thread's
    /// wall time but NOT encoder CPU.
    submit_stall_cycles: u64,
}

#[cfg(perf_tracking)]
impl Default for EncoderFrameCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl EncoderFrameCounters {
    const fn new() -> Self {
        Self {
            buffer_destroys: 0,
            texture_destroys: 0,
            vbib_staging_uploads: 0,
            vbib_mid_pass_reorders: 0,
            texture_blit_uploads: 0,
            texture_blit_padded_uploads: 0,
            texture_gpu_renames: 0,
            op_cycles: 0,
            op_sub_cycles: [0; OpSub::COUNT],
            op_sub_detail: [0; OpSubDetail::COUNT],
            pipeline_memo_hits: 0,
            pipeline_memo_calls: 0,
            submit_cycles: 0,
            drawable_wait_cycles: 0,
            submit_exec_cycles: 0,
            submit_stall_cycles: 0,
        }
    }
}

/// Per-frame API-thread state. Embedded on `DeviceInner`.
///
/// Under `cfg(not(perf_tracking))` this collapses to a unit struct; all
/// `bump_*` / `add_*` / `*_cycles_ptr` methods become `const fn` no-ops
/// (pointer accessors return `null_mut()`, which the shared-side timer
/// stubs ignore). The counter state and the entire `drain_into_payload`
/// path are then compile-time-elided.
#[cfg(perf_tracking)]
pub struct ApiPerfState {
    /// The API-thread-bumped per-frame counters (see [`FrameCounters`]).
    ///
    /// `drain_into_payload` moves this wholesale into the payload and
    /// leaves a zeroed `FrameCounters` behind.
    counters: FrameCounters,
    /// TSC at the start of the previous `device_present`.
    ///
    /// Used to derive the wall-clock frame period. API-thread
    /// bookkeeping — never drained.
    prev_present_rdtsc: u64,
    /// Cycles consumed by nested timers of the currently-innermost running [`ApiTimer`].
    ///
    /// Drives exclusive (self) time accounting: a delegating entry point
    /// (e.g. `IDirect3DSurface9::LockRect` forwarding to the texture
    /// `LockRect` thunk) subtracts this from its raw `elapsed` so the
    /// nested interval is booked only to the callee's bucket. Always 0
    /// when no timer is running (`timer_depth == 0`). Never drained — it
    /// self-balances via the timer save/restore protocol.
    active_child_cycles: u64,
    /// Nesting depth of live `ApiTimer`s on the API thread.
    ///
    /// Used only to detect the outermost timer's `Drop` so
    /// `active_child_cycles` can be reset to 0 at rest (prevents
    /// top-level residue).
    timer_depth: u32,
}

#[cfg(perf_tracking)]
impl Default for ApiPerfState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl ApiPerfState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counters: FrameCounters::new(),
            prev_present_rdtsc: 0,
            active_child_cycles: 0,
            timer_depth: 0,
        }
    }

    /// Pointer the `CycleAddTimer` writes into.
    ///
    /// Naming mirrors the `present_block_cycles_ptr` accessor — d3d9
    /// query.rs is the only consumer.
    pub const fn query_wait_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.query_wait_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the Draw-internal snapshot phase.
    ///
    /// Accumulated (multiple calls per frame), one shared bucket for all
    /// Draw methods.
    pub const fn draw_snapshot_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the Draw-internal push-op phase.
    ///
    /// `Box::new` + `push_op`.
    pub const fn draw_push_op_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_push_op_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for `snapshot_shared`'s per-stage binding walk.
    ///
    /// Sub-component of `draw_snapshot_cycles_ptr`; both timers are live
    /// simultaneously — the stage walk's cycles are double-counted into
    /// the outer `snapshot` total, which is the desired display semantic
    /// (`snapshot` is the parent, `stages` is shown nested under it).
    pub const fn draw_snapshot_stages_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_stages_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the const-snapshot block on FF draws.
    ///
    /// Peer of `draw_snapshot_stages_cycles_ptr` under the parent
    /// `snapshot` total. Selected at draw-classification time alongside
    /// its programmable sibling [`draw_snapshot_c_pr_cycles_ptr`].
    pub const fn draw_snapshot_c_ff_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_c_ff_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the const-snapshot block.
    ///
    /// Selected when both VS and PS are programmable. Peer of
    /// [`draw_snapshot_c_ff_cycles_ptr`].
    pub const fn draw_snapshot_c_pr_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_c_pr_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the shader-key resolution block.
    ///
    /// The block lives in `snapshot_shared`. Peer of `stages`/`consts`
    /// under the parent `snapshot` total.
    pub const fn draw_snapshot_keys_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_keys_cycles
    }

    /// Pointer the `CycleAddTimer` writes into for the post-consts block in `snapshot_shared`.
    ///
    /// The scratch-bump + cache-assignment work. Peer of
    /// `stages`/`consts`/`keys` under the parent `snapshot` total.
    pub const fn draw_snapshot_bumps_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.counters.draw_snapshot_bumps_cycles
    }

    /// Add cycles + one call to the per-category accumulator.
    pub const fn add_api_cycles(&mut self, category: ApiCategory, cycles: u64) {
        let idx = category as usize;
        let cyc = &mut self.counters.api_cycles_by_category[idx];
        *cyc = cyc.saturating_add(cycles);
        let cnt = &mut self.counters.api_call_counts_by_category[idx];
        *cnt = cnt.saturating_add(1);
    }

    /// Add cycles + one call to the top-level `Device` bucket and the per-sub-category breakdown.
    ///
    /// Called from `ApiTimer::start_device` on every `IDirect3DDevice9`
    /// vtable thunk's `Drop`.
    pub const fn add_device_cycles(&mut self, sub: DeviceSubCategory, cycles: u64) {
        let top = &mut self.counters.api_cycles_by_category[ApiCategory::Device as usize];
        *top = top.saturating_add(cycles);
        let top_cnt = &mut self.counters.api_call_counts_by_category[ApiCategory::Device as usize];
        *top_cnt = top_cnt.saturating_add(1);

        let idx = sub as usize;
        let cyc = &mut self.counters.device_sub_cycles[idx];
        *cyc = cyc.saturating_add(cycles);
        let cnt = &mut self.counters.device_sub_calls[idx];
        *cnt = cnt.saturating_add(1);
    }

    /// Add cycles + one call to the Device top, `Bind`, and per-`BindSubCategory` buckets.
    ///
    /// All three land under one `rdtsc()` delta supplied by
    /// `ApiTimer::start_bind`. Sub-bucket sums equal the parent Bind row
    /// by construction (every Bind method uses `bind_timer`; no `Misc`
    /// escape).
    pub const fn add_bind_cycles(&mut self, sub: BindSubCategory, cycles: u64) {
        let top = &mut self.counters.api_cycles_by_category[ApiCategory::Device as usize];
        *top = top.saturating_add(cycles);
        let top_cnt = &mut self.counters.api_call_counts_by_category[ApiCategory::Device as usize];
        *top_cnt = top_cnt.saturating_add(1);

        let dev_idx = DeviceSubCategory::Bind as usize;
        let dev_cyc = &mut self.counters.device_sub_cycles[dev_idx];
        *dev_cyc = dev_cyc.saturating_add(cycles);
        let dev_cnt = &mut self.counters.device_sub_calls[dev_idx];
        *dev_cnt = dev_cnt.saturating_add(1);

        let idx = sub as usize;
        let cyc = &mut self.counters.bind_sub_cycles[idx];
        *cyc = cyc.saturating_add(cycles);
        let cnt = &mut self.counters.bind_sub_calls[idx];
        *cnt = cnt.saturating_add(1);
    }

    /// Record one live call to a [`KeysGate`] setter, plus whether it skipped the FF-key rebuild.
    ///
    /// Skipped means the value/mask was unchanged. The `skips / calls`
    /// ratio per gate surfaces the redundant-work elision rate in the
    /// perf summary.
    pub const fn record_keys_gate(&mut self, gate: KeysGate, skipped: bool) {
        let idx = gate as usize;
        let calls = &mut self.counters.keys_gate_calls[idx];
        *calls = calls.saturating_add(1);
        if skipped {
            let skips = &mut self.counters.keys_gate_skips[idx];
            *skips = skips.saturating_add(1);
        }
    }

    pub const fn bump_vb_rename(&mut self) {
        self.counters.vb_rename = self.counters.vb_rename.saturating_add(1);
    }

    pub const fn bump_ib_rename(&mut self) {
        self.counters.ib_rename = self.counters.ib_rename.saturating_add(1);
    }

    /// Whole-buffer non-WRITEONLY contended Lock.
    ///
    /// Fresh `PageBox` + inline `memcpy` from the old box (game might
    /// read the whole buffer through the Lock pointer). Rare; bump exists
    /// so the `preserve` row stays informative if it ever fires.
    pub const fn bump_vbib_preserve_cpu(&mut self) {
        self.counters.vbib_preserve_cpu = self.counters.vbib_preserve_cpu.saturating_add(1);
    }

    /// Bump on a successful `DrainRetiredNow` recovery.
    ///
    /// The cheap tier was enough for the retry to allocate.
    pub const fn bump_alloc_recovery_drain(&mut self) {
        self.counters.alloc_recovery_drain = self.counters.alloc_recovery_drain.saturating_add(1);
    }

    /// Bump on a `MidFrameSubmitForAlloc` recovery — heavy tier, includes a GPU completion wait.
    ///
    /// Each fire costs ~1-2 ms.
    pub const fn bump_alloc_recovery_submit(&mut self) {
        self.counters.alloc_recovery_submit = self.counters.alloc_recovery_submit.saturating_add(1);
    }

    pub const fn bump_vb_discard(&mut self) {
        self.counters.vb_discards = self.counters.vb_discards.saturating_add(1);
    }

    pub const fn bump_ib_discard(&mut self) {
        self.counters.ib_discards = self.counters.ib_discards.saturating_add(1);
    }

    /// Bump the CPU-memcpy preserve counter.
    ///
    /// Called from the non-WRITEONLY non-DISCARD non-READONLY contended
    /// texture `LockRect` branch.
    pub const fn bump_texture_preserve_cpu(&mut self) {
        self.counters.texture_preserve_cpu = self.counters.texture_preserve_cpu.saturating_add(1);
    }

    /// Feed the `AddDirtyRect` probe: one call this frame.
    ///
    /// `partial` is true when the rect is a usable sub-region (narrower
    /// than the level-0 surface); `area_bp` is its area in basis points of
    /// the mip (whole-mip / `None` = 10000). Surfaces in the
    /// `AddDirtyRect` row of the Resources(textures) section.
    pub const fn bump_texture_add_dirty_rect(&mut self, partial: bool, area_bp: u32) {
        self.counters.texture_add_dirty_calls =
            self.counters.texture_add_dirty_calls.saturating_add(1);
        if partial {
            self.counters.texture_add_dirty_partial =
                self.counters.texture_add_dirty_partial.saturating_add(1);
        }
        self.counters.texture_add_dirty_area_bp = self
            .counters
            .texture_add_dirty_area_bp
            .saturating_add(area_bp);
    }

    pub const fn bump_texture_rename(&mut self) {
        self.counters.texture_renames = self.counters.texture_renames.saturating_add(1);
    }

    /// Subset of `bump_texture_rename` that took the no-preserve branch.
    ///
    /// Explicit `D3DLOCK_DISCARD` or whole-mip `D3DUSAGE_DYNAMIC`
    /// contended.
    pub const fn bump_texture_discard(&mut self) {
        self.counters.texture_discards = self.counters.texture_discards.saturating_add(1);
    }

    /// Drain this frame's API-thread counters into the outgoing payload, then zero self.
    ///
    /// Also samples `rdtsc()` and computes `frame_total_cycles` as the
    /// delta from the previous call — returns 0 on the very first frame
    /// (no predecessor).
    pub fn drain_into_payload(&mut self, payload: &mut FramePerfPayload) {
        let now = rdtsc();
        let prev = core::mem::replace(&mut self.prev_present_rdtsc, now);
        // Move this frame's counters into the payload, leaving a zeroed
        // `FrameCounters` behind (`mem::take`). `present_block` / `op_vec_*`
        // are written to `payload.timing` by their own setters after this
        // drain, so moving only `counters` + `frame_total_cycles` here never
        // clobbers them.
        payload.counters = core::mem::take(&mut self.counters);
        payload.timing.frame_total_cycles = if prev == 0 { 0 } else { now - prev };
    }
}

/// ZST twin of [`ApiPerfState`] for `cfg(not(perf_tracking))`.
///
/// Every method is a `const fn` no-op so `let _t = device_timer(...)` and
/// every `dev.perf_mut().bump_*()` call site compile away under thin-LTO.
#[cfg(not(perf_tracking))]
#[derive(Default)]
pub struct ApiPerfState;

#[cfg(not(perf_tracking))]
impl ApiPerfState {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    pub const fn query_wait_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_push_op_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_stages_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_c_ff_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_c_pr_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_keys_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn draw_snapshot_bumps_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }

    #[inline]
    pub const fn add_api_cycles(&mut self, _category: ApiCategory, _cycles: u64) {}
    #[inline]
    pub const fn add_device_cycles(&mut self, _sub: DeviceSubCategory, _cycles: u64) {}
    #[inline]
    pub const fn add_bind_cycles(&mut self, _sub: BindSubCategory, _cycles: u64) {}

    #[inline]
    pub const fn record_keys_gate(&mut self, _gate: KeysGate, _skipped: bool) {}

    #[inline]
    pub const fn bump_vb_rename(&mut self) {}
    #[inline]
    pub const fn bump_ib_rename(&mut self) {}
    #[inline]
    pub const fn bump_vbib_preserve_cpu(&mut self) {}
    #[inline]
    pub const fn bump_alloc_recovery_drain(&mut self) {}
    #[inline]
    pub const fn bump_alloc_recovery_submit(&mut self) {}
    #[inline]
    pub const fn bump_vb_discard(&mut self) {}
    #[inline]
    pub const fn bump_ib_discard(&mut self) {}
    #[inline]
    pub const fn bump_texture_preserve_cpu(&mut self) {}
    #[inline]
    pub const fn bump_texture_add_dirty_rect(&mut self, _partial: bool, _area_bp: u32) {}
    #[inline]
    pub const fn bump_texture_rename(&mut self) {}
    #[inline]
    pub const fn bump_texture_discard(&mut self) {}

    #[inline]
    pub const fn drain_into_payload(&mut self, _payload: &mut FramePerfPayload) {}
}

/// Payload that crosses the API→encoder channel inside `FrameData`.
///
/// Every field mirrors an `ApiPerfState` counter drained at
/// `Present` time, plus the two wall-clock fields set around the
/// `sync_channel(1)` backpressure wait.
///
/// Under `cfg(not(perf_tracking))` this is a unit struct; all setters
/// become `const fn` no-ops.
#[cfg(perf_tracking)]
pub struct FramePerfPayload {
    /// The API-thread counters moved across the channel (see [`FrameCounters`]).
    ///
    /// Seeded by `drain_into_payload`'s wholesale move.
    counters: FrameCounters,
    /// Payload-injected per-frame timing (see [`FrameTiming`]).
    ///
    /// The backpressure stall, the wall-clock frame period, and the op-vec
    /// footprint — set by the setters below and `drain_into_payload`, not
    /// bumped on the API thread.
    timing: FrameTiming,
}

#[cfg(perf_tracking)]
impl Default for FramePerfPayload {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl FramePerfPayload {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counters: FrameCounters::new(),
            timing: FrameTiming::new(),
        }
    }

    /// Stamp the API→encoder `Vec<Op>` footprint metrics into the payload.
    ///
    /// At `stamp_and_swap` time: called once per frame from
    /// `DeviceInner::stamp_and_swap` after the running `ApiPerfState` has
    /// been drained into the payload.
    pub const fn set_op_vec_metrics(&mut self, capacity_bytes: u64, realloc_bytes: u64) {
        self.timing.op_vec_capacity_bytes = capacity_bytes;
        self.timing.op_vec_realloc_bytes = realloc_bytes;
    }

    pub const fn set_present_block_cycles(&mut self, cycles: u64) {
        self.timing.present_block_cycles = cycles;
    }

    /// Pointer the API thread's `CycleSetTimer` writes into.
    ///
    /// Set once per `Present` to capture the `send_frame` backpressure
    /// wait. Mirrors `query_wait_cycles_ptr`.
    pub const fn present_block_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.timing.present_block_cycles
    }
}

/// ZST twin of [`FramePerfPayload`] for `cfg(not(perf_tracking))`.
#[cfg(not(perf_tracking))]
#[derive(Default)]
pub struct FramePerfPayload;

#[cfg(not(perf_tracking))]
impl FramePerfPayload {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    pub const fn set_op_vec_metrics(&mut self, _capacity_bytes: u64, _realloc_bytes: u64) {}
    #[inline]
    pub const fn set_present_block_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn present_block_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
}

/// Identity tag for a shader at the encoder side.
///
/// Used as a map key for the per-pair stats and as the formatted tag in
/// trace dumps.
///
/// The module in d3d9 that constructs these fills `hash` from
/// `ProgramId.raw()` when `is_programmable`, or from a `DefaultHasher` over
/// the fixed-function key otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairShaderId {
    pub is_programmable: bool,
    pub hash: u64,
}

impl PairShaderId {
    /// "prog 0x…" for programmable shaders, "ff 0x…" for fixed-function.
    ///
    /// The same tag shape the encoder's trace dumps use for a shader source.
    #[must_use]
    pub fn tag(&self) -> String {
        let kind = if self.is_programmable { "prog" } else { "ff" };
        format!("{kind} {:#x}", self.hash)
    }
}

/// Parameter bag for `EncoderPerfState::bump_pair_stats`.
///
/// Grouped so the call signature stays under clippy's
/// `too_many_arguments` threshold — pattern matches `DeviceCreateInfo` /
/// `FrameInit`.
#[derive(Clone, Copy)]
pub struct PairStatsSample {
    pub rt_w: u32,
    pub rt_h: u32,
    pub vs: PairShaderId,
    pub ps: PairShaderId,
    pub verts: u32,
    pub alpha_func: u8,
    pub cull_mode: u32,
}

/// Cache-length snapshot handed to `EncoderPerfState::log_frame_summary`.
///
/// Sourced from `FrameEncoder::cache_sizes()` so perf doesn't need to
/// know about the per-cache types.
pub struct CacheSizes {
    pub textures: usize,
    pub pipelines: usize,
    pub samplers: usize,
    pub programs: usize,
    pub libs: usize,
    pub depth_states: usize,
    /// Small bump-packed scratch chunks (`ScratchArena::small_chunk_count`).
    ///
    /// Counted separately from `scratch_oversized_blocks` so the diag row
    /// can show which path dominates the arena footprint.
    pub scratch_small_blocks: u32,
    /// Oversized one-shot scratch chunks (`ScratchArena::oversized_chunk_count`).
    ///
    /// Expected to be 0 in d3d9 — non-zero is the signal that some payload
    /// is exceeding the chunk size and motivating its own chunk per frame.
    pub scratch_oversized_blocks: u32,
    pub scratch_bytes: u64,
    /// Live `Vec<Command>` capacity in bytes across every pass at frame end.
    ///
    /// Sourced from `PassState::cmd_vec_capacity_bytes`. The pool recycles
    /// these vectors across frames, so this is resident footprint — paired
    /// with `cmd_vec_realloc_bytes` on `FrameSample` (steady-state size
    /// vs. growth churn).
    pub cmd_vec_capacity_bytes: u64,
    pub pending_blit_retention_depth: usize,
    pub pending_resource_retention_depth: usize,
}

/// Frame-identity bits needed by `log_frame_summary` for the trace-level per-pass dump.
///
/// The handles are compared against each pass's color/depth attachments to
/// annotate which pass renders to the backbuffer vs off-screen.
pub struct FrameSummaryContext {
    pub backbuffer_handle: MetalHandle<MTLTextureKind>,
    pub depth_texture: MetalHandle<MTLTextureKind>,
    pub backbuffer_width: u32,
    pub backbuffer_height: u32,
}

/// Per-frame encoder-thread state. Embedded on `FrameEncoder`.
///
/// Under `cfg(not(perf_tracking))` this collapses to a unit struct; all
/// methods (`begin_frame`, `set_*`, `*_cycles_ptr`, `bump_*`,
/// `bump_pair_stats`, `log_frame_summary`) become `const fn` no-ops or
/// return `null_mut()`. Together with the [`PerfWindow`] / [`Summary`]
/// items below it (also cfg-gated) the entire 5-second summary pipeline
/// is compile-time-elided.
#[cfg(perf_tracking)]
pub struct EncoderPerfState {
    /// Rolling aggregator for the 5-second `info!` summary.
    perf_window: PerfWindow,

    /// API-thread counters seeded from `FramePerfPayload` in `begin_frame`.
    ///
    /// See [`FrameCounters`]. Read-only encoder-side — forwarded into the
    /// `FrameSample`.
    counters: FrameCounters,
    /// Payload-injected per-frame timing seeded from the payload in `begin_frame`.
    ///
    /// See [`FrameTiming`].
    timing: FrameTiming,
    /// Encoder-bumped per-frame counters (see [`EncoderFrameCounters`]).
    ///
    /// Reset wholesale each `begin_frame`; `drawable_wait` / `submit_exec`
    /// are folded back from the submit thread *after* that reset (so a
    /// frame with nothing returned yet reports 0 rather than stale data).
    enc: EncoderFrameCounters,

    per_pair_stats: HashMap<(u32, u32, PairShaderId, PairShaderId), PerPairStats>,

    // ── Running totals (live across frames) ──
    /// Byte total of live `PageBoxes` in the encoder's retention queue.
    ///
    /// Despite the `vbib_` prefix the queue (`pending_vbib_retention`)
    /// holds every `PageBox` parked for retire — VB/IB rename intake,
    /// visibility-pool eviction, and texture-blit padded staging all
    /// share it. Mutated by `bump_vbib_retained_add` / `_sub` at each
    /// producer / drain site. Not reset per frame; sampled at end of
    /// frame as a peak-memory proxy.
    vbib_retained_bytes: usize,
    /// Byte total of live texture-staging `Box<[u8]>` Arcs.
    ///
    /// Held in the encoder's `pending_blit_retention`. Mutated by
    /// `bump_tex_staging_retained_add` / `_sub` when Arcs move into the
    /// pending queue at submit time and when drained entries are popped.
    /// Not reset per frame; sampled at end of frame as a peak-memory
    /// proxy, symmetric to `vbib_retained_bytes`.
    tex_staging_retained_bytes: usize,
}

#[cfg(perf_tracking)]
impl Default for EncoderPerfState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(perf_tracking)]
impl EncoderPerfState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            perf_window: PerfWindow::new(),
            counters: FrameCounters::new(),
            timing: FrameTiming::new(),
            enc: EncoderFrameCounters::new(),
            per_pair_stats: HashMap::new(),
            vbib_retained_bytes: 0,
            tex_staging_retained_bytes: 0,
        }
    }

    /// Seed per-frame encoder counters from the incoming payload.
    ///
    /// Resets every per-frame encoder-side counter. Live totals
    /// (`vbib_retained_bytes`, `tex_staging_retained_bytes`) persist.
    pub fn begin_frame(&mut self, payload: &FramePerfPayload) {
        // Seed the API-thread counters + payload timing wholesale.
        self.counters = payload.counters;
        self.timing = payload.timing;
        // Reset every encoder-bumped per-frame counter in one move.
        // `drawable_wait_cycles` / `submit_exec_cycles` live in `enc` and are
        // folded back from the submit thread when a payload returns (after
        // this reset, in `drain_returned_payloads`); zeroing them here means a
        // frame with nothing returned yet reports 0 rather than stale data.
        self.enc = EncoderFrameCounters::default();
        self.per_pair_stats.clear();
    }

    pub const fn set_op_cycles(&mut self, cycles: u64) {
        self.enc.op_cycles = cycles;
    }

    pub const fn set_submit_cycles(&mut self, cycles: u64) {
        self.enc.submit_cycles = cycles;
    }

    /// Pointer the encoder thread's `CycleSetTimer` writes into for the per-frame ops loop.
    ///
    /// Used by `run_frame` to bracket the closure-replay cycles inline.
    pub const fn op_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.enc.op_cycles
    }

    /// Pointer the per-draw `CycleAddTimer` for op-loop phase `sub` writes into.
    ///
    /// One slot of `op_sub_cycles`; the six phases decompose `op_cycles`
    /// for the summary's "Closures (op)" sub-tree.
    pub const fn op_sub_cycles_ptr(&mut self, sub: OpSub) -> *mut u64 {
        &raw mut self.enc.op_sub_cycles[sub as usize]
    }

    /// Pointer the per-draw `CycleAddTimer` for an [`OpSubDetail`] child writes into.
    ///
    /// Nested inside the `Resolve`/`Binds` parent timers in `emit_draw`.
    pub const fn op_sub_detail_ptr(&mut self, detail: OpSubDetail) -> *mut u64 {
        &raw mut self.enc.op_sub_detail[detail as usize]
    }

    /// Count one `get_or_create_pipeline` call — the memo hit-rate denominator.
    pub const fn bump_pipeline_memo_call(&mut self) {
        self.enc.pipeline_memo_calls = self.enc.pipeline_memo_calls.saturating_add(1);
    }

    /// Count one pipeline-resolve memo hit (snapshot matched the previous draw).
    pub const fn bump_pipeline_memo_hit(&mut self) {
        self.enc.pipeline_memo_hits = self.enc.pipeline_memo_hits.saturating_add(1);
    }

    /// Pointer the encoder thread's `CycleSetTimer` writes into for the per-frame submit.
    ///
    /// Covers `drawable_wait` + commit. Bracketed by `run_frame`.
    pub const fn submit_cycles_ptr(&mut self) -> *mut u64 {
        &raw mut self.enc.submit_cycles
    }

    pub const fn set_drawable_wait_cycles(&mut self, cycles: u64) {
        self.enc.drawable_wait_cycles = cycles;
    }

    pub const fn set_submit_exec_cycles(&mut self, cycles: u64) {
        self.enc.submit_exec_cycles = cycles;
    }

    pub const fn add_submit_stall_cycles(&mut self, cycles: u64) {
        self.enc.submit_stall_cycles = self.enc.submit_stall_cycles.saturating_add(cycles);
    }

    /// Bumped when the retention drain destroys an `MTLBuffer` wrapper.
    ///
    /// Fires for wrappers whose `from_texture` flag is `false`: VB/IB
    /// cache renames (`ensure_vbib_mtl_buffer_impl`), VB/IB Lock-rename /
    /// Release intake (`intake_vbib_retention`), visibility-pool eviction.
    /// Texture-staging wrapper destroys flow through the same retention
    /// queue but increment `texture_destroys` instead, so each section's
    /// `destroys` row reflects only its own work.
    pub const fn bump_buffer_destroy(&mut self) {
        self.enc.buffer_destroys = self.enc.buffer_destroys.saturating_add(1);
    }

    /// Count one `Staged` VB/IB dirty-range upload blit.
    ///
    /// Pushed into `frame_blit_commands` by `apply_stage_upload`.
    pub const fn bump_vbib_staging_upload(&mut self) {
        self.enc.vbib_staging_uploads = self.enc.vbib_staging_uploads.saturating_add(1);
    }

    /// Count one VB/IB rename-at-overlap.
    ///
    /// A `Staged` upload that overwrote a region a draw already read
    /// earlier this frame, so the device buffer was renamed to preserve
    /// that draw's bytes.
    pub const fn bump_vbib_mid_pass_reorder(&mut self) {
        self.enc.vbib_mid_pass_reorders = self.enc.vbib_mid_pass_reorders.saturating_add(1);
    }

    pub const fn bump_vbib_retained_add(&mut self, bytes: usize) {
        self.vbib_retained_bytes = self.vbib_retained_bytes.saturating_add(bytes);
    }

    pub const fn bump_vbib_retained_sub(&mut self, bytes: usize) {
        self.vbib_retained_bytes = self.vbib_retained_bytes.saturating_sub(bytes);
    }

    pub const fn bump_tex_staging_retained_add(&mut self, bytes: usize) {
        self.tex_staging_retained_bytes = self.tex_staging_retained_bytes.saturating_add(bytes);
    }

    pub const fn bump_tex_staging_retained_sub(&mut self, bytes: usize) {
        self.tex_staging_retained_bytes = self.tex_staging_retained_bytes.saturating_sub(bytes);
    }

    pub const fn bump_texture_destroy(&mut self) {
        self.enc.texture_destroys = self.enc.texture_destroys.saturating_add(1);
    }

    /// Bumped once per successful `run_texture_upload_blit` invocation.
    ///
    /// Successful means it emits a `copy_buffer_to_texture` command.
    /// Includes the padded-staging sub-path; the padded counter is a
    /// strict subset of this one.
    pub const fn bump_texture_blit_upload(&mut self) {
        self.enc.texture_blit_uploads = self.enc.texture_blit_uploads.saturating_add(1);
    }

    pub const fn bump_texture_blit_padded_upload(&mut self) {
        self.enc.texture_blit_padded_uploads =
            self.enc.texture_blit_padded_uploads.saturating_add(1);
    }

    /// Count one texture rename-at-overlap.
    ///
    /// An upload into a texture a draw already sampled this frame,
    /// redirected to a fresh `MTLTexture` so the earlier draw keeps its
    /// per-draw content.
    pub const fn bump_texture_gpu_rename(&mut self) {
        self.enc.texture_gpu_renames = self.enc.texture_gpu_renames.saturating_add(1);
    }

    /// Accumulate one draw's stats.
    ///
    /// Gated on `mtld3d::d3d9::passes=trace` via the cached
    /// `pair_stats_enabled` flag.
    pub fn bump_pair_stats(&mut self, sample: PairStatsSample) {
        if !pair_stats_enabled() {
            return;
        }
        let PairStatsSample {
            rt_w,
            rt_h,
            vs,
            ps,
            verts,
            alpha_func,
            cull_mode,
        } = sample;
        let entry = self.per_pair_stats.entry((rt_w, rt_h, vs, ps)).or_default();
        entry.draws = entry.draws.saturating_add(1);
        entry.verts = entry.verts.saturating_add(verts);
        entry.alpha_func = alpha_func;
        entry.cull_mode = cull_mode;
    }

    /// Accumulate this frame into the rolling 5-second window.
    ///
    /// Once the window has spanned `SUMMARY_INTERVAL_SECS`, emit the
    /// averaged `info!` summary on `mtld3d::perf`. The per-pass breakdown,
    /// the `present_texture=…` audit line, and the per-RT pair dump are
    /// emitted on the separate `mtld3d::d3d9::passes=trace` switch — they
    /// are pass / workload shape, not perf metrics.
    ///
    /// # Panics
    ///
    /// Panics if pass / command counts exceed `u32::MAX`. Unreachable —
    /// real frames cap at a few thousand passes.
    pub fn log_frame_summary(
        &mut self,
        caches: &CacheSizes,
        passes: &[Pass],
        ctx: &FrameSummaryContext,
        submit_status: i32,
        cmd_vec_realloc_bytes: u64,
    ) {
        // `mtld3d::perf=debug` gates both the averaged summary and the
        // per-call ApiTimer / CycleSet / CycleAdd cycle accounting —
        // both read from the latched `perf_enabled()` flag so this
        // per-frame check is a single Relaxed load. The pass /
        // present-texture / per-pair detail is gated independently on
        // `mtld3d::d3d9::passes=trace` via the cached
        // `pair_stats_enabled` flag.
        let want_stats = perf_enabled();
        let want_passes = pair_stats_enabled();
        if !want_stats && !want_passes {
            return;
        }
        let draw_prim = CommandType::DrawPrimitives as u32;
        let draw_idx = CommandType::DrawIndexedPrimitives as u32;
        let set_pipeline = CommandType::SetRenderPipelineState as u32;
        let set_frag_tex = CommandType::SetFragmentTexture as u32;

        let mut total_draws: u32 = 0;
        let mut total_commands: u32 = 0;
        for p in passes {
            total_commands +=
                u32::try_from(p.commands().len()).expect("per-pass command count fits u32");
            for cmd in p.commands() {
                if cmd.cmd == draw_prim || cmd.cmd == draw_idx {
                    total_draws += 1;
                }
            }
        }
        let enc_cycles = self.enc.op_cycles + self.enc.submit_cycles;
        // Raw Device bucket from `ApiTimer` includes the entire
        // `device_present` body — which contains the `send_frame`
        // backpressure wait. The display nests `Present stall` as a
        // subtimer under `Device`, so the raw value is what we want
        // per-bucket. `api_cyc` = raw sum (includes the stall).
        // `api_work = api_cyc − present_block` is kept for the
        // bottleneck classifier and the `buckets:` summary line,
        // where "API CPU" should mean non-blocking CPU.
        let api_by = self.counters.api_cycles_by_category;
        let api_total: u64 = api_by.iter().sum();
        let outside_d3d9 = self.timing.frame_total_cycles.saturating_sub(api_total);
        let api_work_cyc = api_total.saturating_sub(self.timing.present_block_cycles);

        let sample = FrameSample {
            // The three per-frame counter groups flow into the sample as
            // single moves (the API counters, payload timing, and encoder
            // counters were all assembled on `self` over the frame).
            counters: self.counters,
            timing: self.timing,
            enc: self.enc,
            // Cache-length / pass snapshot pulled from `caches` + the pass list.
            passes: u32::try_from(passes.len()).expect("pass count fits u32"),
            commands: total_commands,
            draws: total_draws,
            scratch_small_blocks: caches.scratch_small_blocks,
            scratch_oversized_blocks: caches.scratch_oversized_blocks,
            scratch_bytes: caches.scratch_bytes,
            cmd_vec_capacity_bytes: caches.cmd_vec_capacity_bytes,
            vbib_retention_depth: caches.pending_resource_retention_depth,
            vbib_retained_bytes: self.vbib_retained_bytes,
            pending_blit_retention_depth: caches.pending_blit_retention_depth,
            tex_staging_retained_bytes: self.tex_staging_retained_bytes,
            cmd_vec_realloc_bytes,
            // Derived this frame from the counters above.
            outside_d3d9,
            api_cyc: api_total,
            api_work: api_work_cyc,
            enc_cyc: enc_cycles,
            submit_status,
        };
        self.perf_window.accumulate(&sample);

        let window_cycles = rdtsc().saturating_sub(self.perf_window.started_tsc);
        if window_cycles < secs_to_cycles(SUMMARY_INTERVAL_SECS) {
            return;
        }

        if want_stats {
            let window_secs = cycles_to_ms(window_cycles) / 1e3;
            let rendered = Summary::render(&self.perf_window, caches, window_secs);
            info!(target: LOG_TARGET, "{rendered}");
        }

        if want_passes {
            for (i, p) in passes.iter().enumerate() {
                let (cw, ch) = p.color_size();
                let target = if p.color_texture() == ctx.backbuffer_handle {
                    format!("bb={:#x}", p.color_texture())
                } else {
                    format!("rt={:#x}", p.color_texture())
                };
                let color_load = match p.color_load() {
                    ColorLoad::Load => "load".to_string(),
                    ColorLoad::Clear { r, g, b, a } => format!(
                        "clear({:.2},{:.2},{:.2},{:.2})",
                        f32::from_bits(r),
                        f32::from_bits(g),
                        f32::from_bits(b),
                        f32::from_bits(a),
                    ),
                    ColorLoad::DontCare => "dontcare".to_string(),
                };
                let depth_tag = if p.depth_texture() == ctx.depth_texture {
                    // Pass uses the device's default depth. If the color
                    // attachment is not the backbuffer, that's almost
                    // certainly wrong — sizes won't match and Metal will
                    // reject the pass.
                    if p.color_texture() == ctx.backbuffer_handle {
                        format!("depth={:#x}", p.depth_texture())
                    } else {
                        format!("depth=!{:#x}(default)", p.depth_texture())
                    }
                } else {
                    format!("depth={:#x}", p.depth_texture())
                };
                let depth_load = match p.depth_load() {
                    DepthLoad::Load => "load".to_string(),
                    DepthLoad::Clear { value } => format!("clear({:.3})", f32::from_bits(value)),
                    DepthLoad::DontCare => "dontcare".to_string(),
                };
                let mut pass_draws: u32 = 0;
                let mut pass_pipelines: u32 = 0;
                let mut pass_frag_textures: u32 = 0;
                for cmd in p.commands() {
                    if cmd.cmd == draw_prim || cmd.cmd == draw_idx {
                        pass_draws += 1;
                    } else if cmd.cmd == set_pipeline {
                        pass_pipelines += 1;
                    } else if cmd.cmd == set_frag_tex {
                        pass_frag_textures += 1;
                    }
                }
                trace!(
                    target: PASSES_TARGET,
                    "  pass {i}: {target} {cw}x{ch} color={color_load} {depth_tag} depth_load={depth_load} cmds={cmds} draws={draws} pipelines={pipes} frag_textures={tex}",
                    cmds = p.commands().len(),
                    draws = pass_draws,
                    pipes = pass_pipelines,
                    tex = pass_frag_textures,
                );
            }

            let expected_final_bb = passes
                .last()
                .is_some_and(|p| p.color_texture() == ctx.backbuffer_handle);
            trace!(
                target: PASSES_TARGET,
                "  present_texture={pt:#x} bb={bb:#x} {size} expected_final_bb_pass={check}",
                pt = ctx.backbuffer_handle,
                bb = ctx.backbuffer_handle,
                size = format_args!("{}x{}", ctx.backbuffer_width, ctx.backbuffer_height),
                check = if expected_final_bb { "yes" } else { "NO" },
            );

            // Per-(RT w×h, vs, ps) draw stats. Filtered to 4096×2048
            // because that's a common large off-screen RT size worth
            // surfacing; wider filters would flood output.
            let mut rows: Vec<(u32, u32, String, String, PerPairStats)> = self
                .per_pair_stats
                .iter()
                .filter(|((w, h, _, _), _)| *w == 4096 && *h == 2048)
                .map(|((w, h, vs, ps), s)| (*w, *h, vs.tag(), ps.tag(), *s))
                .collect();
            rows.sort_by(|a, b| (a.0, a.1, &a.2, &a.3).cmp(&(b.0, b.1, &b.2, &b.3)));
            for (w, h, vs_tag, ps_tag, stats) in &rows {
                trace!(
                    target: PASSES_TARGET,
                    "  pair RT {w}x{h} VS {vs_tag} PS {ps_tag} draws={d} verts={v} alpha_func={af} cull_mode={cm}",
                    d = stats.draws,
                    v = stats.verts,
                    af = stats.alpha_func,
                    cm = stats.cull_mode,
                );
            }
        }

        self.perf_window.reset();
    }
}

/// ZST twin of [`EncoderPerfState`] for `cfg(not(perf_tracking))`.
#[cfg(not(perf_tracking))]
#[derive(Default)]
pub struct EncoderPerfState;

#[cfg(not(perf_tracking))]
impl EncoderPerfState {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    pub const fn begin_frame(&mut self, _payload: &FramePerfPayload) {}
    #[inline]
    pub const fn set_op_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn set_submit_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn op_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn op_sub_cycles_ptr(&mut self, _sub: OpSub) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn op_sub_detail_ptr(&mut self, _detail: OpSubDetail) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn bump_pipeline_memo_call(&mut self) {}
    #[inline]
    pub const fn bump_pipeline_memo_hit(&mut self) {}
    #[inline]
    pub const fn submit_cycles_ptr(&mut self) -> *mut u64 {
        core::ptr::null_mut()
    }
    #[inline]
    pub const fn set_drawable_wait_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn set_submit_exec_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn add_submit_stall_cycles(&mut self, _cycles: u64) {}
    #[inline]
    pub const fn bump_buffer_destroy(&mut self) {}
    #[inline]
    pub const fn bump_vbib_staging_upload(&mut self) {}
    #[inline]
    pub const fn bump_vbib_mid_pass_reorder(&mut self) {}
    #[inline]
    pub const fn bump_vbib_retained_add(&mut self, _bytes: usize) {}
    #[inline]
    pub const fn bump_vbib_retained_sub(&mut self, _bytes: usize) {}
    #[inline]
    pub const fn bump_tex_staging_retained_add(&mut self, _bytes: usize) {}
    #[inline]
    pub const fn bump_tex_staging_retained_sub(&mut self, _bytes: usize) {}
    #[inline]
    pub const fn bump_texture_destroy(&mut self) {}
    #[inline]
    pub const fn bump_texture_blit_upload(&mut self) {}
    #[inline]
    pub const fn bump_texture_blit_padded_upload(&mut self) {}
    #[inline]
    pub const fn bump_texture_gpu_rename(&mut self) {}
    #[inline]
    pub const fn bump_pair_stats(&mut self, _sample: PairStatsSample) {}
    #[inline]
    pub const fn log_frame_summary(
        &mut self,
        _caches: &CacheSizes,
        _passes: &[Pass],
        _ctx: &FrameSummaryContext,
        _submit_status: i32,
        _cmd_vec_realloc_bytes: u64,
    ) {
    }
}

/// Per-frame draw stats accumulated for one (`rt_w`, `rt_h`, vs, ps) tuple.
///
/// `alpha_func` / `cull_mode` capture the *last* draw of the tuple in the
/// frame — stable per pair in practice.
#[cfg(perf_tracking)]
#[derive(Default, Clone, Copy)]
struct PerPairStats {
    draws: u32,
    verts: u32,
    alpha_func: u8,
    cull_mode: u32,
}

/// Snapshot of one frame's counters, fed into `PerfWindow::accumulate`.
///
/// Separate from `EncoderPerfState` so `log_frame_summary` can compute
/// derived fields (`enc_cyc`, `api_work`, `outside_d3d9`) once.
#[cfg(perf_tracking)]
struct FrameSample {
    /// API-thread counters for this frame (see [`FrameCounters`]).
    ///
    /// Moved in from `EncoderPerfState` so a new counter added there
    /// appears here automatically.
    counters: FrameCounters,
    /// Payload-injected timing for this frame (see [`FrameTiming`]).
    timing: FrameTiming,
    /// Encoder-bumped counters for this frame (see [`EncoderFrameCounters`]).
    enc: EncoderFrameCounters,

    // ── Cache-length / pass snapshot (from `CacheSizes` + the pass list) ──
    passes: u32,
    commands: u32,
    draws: u32,
    scratch_small_blocks: u32,
    scratch_oversized_blocks: u32,
    scratch_bytes: u64,
    /// Resident `Vec<Command>` capacity bytes summed across every pass at end-of-frame.
    ///
    /// Steady-state size of the encoder's command storage (the pool keeps
    /// these around between frames). Paired with `cmd_vec_realloc_bytes`
    /// below to separate footprint from churn.
    cmd_vec_capacity_bytes: u64,
    vbib_retention_depth: usize,
    vbib_retained_bytes: usize,
    pending_blit_retention_depth: usize,
    tex_staging_retained_bytes: usize,
    /// Per-frame total of bytes memcpy'd by `Pass::commands` `Vec` doublings.
    ///
    /// Sourced from `PassState::take_cmd_vec_realloc_bytes` once per
    /// frame. Paired with `cmd_vec_capacity_bytes` above so the diag row
    /// shows growth churn next to resident footprint; a working pool
    /// drives this near zero in steady state.
    cmd_vec_realloc_bytes: u64,

    // ── Derived this frame from the counters above ──
    /// `frame_total − Σ api_cycles_by_category`.
    ///
    /// The game-code time the frame spent outside any D3D9 entry point.
    outside_d3d9: u64,
    /// Σ `api_cycles_by_category` (raw, includes the `Present` send stall).
    api_cyc: u64,
    /// `api_cyc − present_block` — non-blocking API-thread CPU.
    api_work: u64,
    /// Encoder-thread CPU = `op_cycles + submit_cycles`.
    enc_cyc: u64,
    submit_status: i32,
}

/// One windowed metric: a running window **sum** and a per-frame **peak**.
///
/// `add` folds one frame's value into both halves; `peak` updates only the
/// peak (for derived quantities that have no meaningful window sum, e.g.
/// the `*_leftover` residuals). Render reads `.sum` (often `/ frames` for
/// an average) and `.max` (worst single frame). A metric that only ever
/// uses one half simply leaves the other at 0 — uniform storage beats a
/// third "sum-only" / "peak-only" type. Both halves are `u64` so the
/// window sum never overflows the PE-side 32-bit `usize` (the documented
/// retained-byte overflow class).
#[cfg(perf_tracking)]
#[derive(Clone, Copy, Default)]
struct Stat {
    sum: u64,
    max: u64,
}

#[cfg(perf_tracking)]
impl Stat {
    /// Fold one frame: add to the window sum (saturating) and raise the peak.
    fn add(&mut self, v: u64) {
        self.sum = self.sum.saturating_add(v);
        self.max = self.max.max(v);
    }
    /// Raise only the per-frame peak — for derived quantities with no sum.
    fn peak(&mut self, v: u64) {
        self.max = self.max.max(v);
    }
}

/// Rolling 5-second window that folds per-frame counters into a [`Stat`] per metric.
///
/// Each [`Stat`] is a window sum + per-frame peak. On emit, time sums
/// divide by `frames` to give ms/frame, event sums surface as raw window
/// totals, and depth sums divide by `frames` as decimal f64 averages;
/// peaks surface as the worst single frame. `started_tsc` anchors the
/// window via `rdtsc()` deltas so the rate limit is wall-clock (not
/// frame-count) based.
#[cfg(perf_tracking)]
#[derive(Default)]
struct PerfWindow {
    started_tsc: u64,
    frames: u32,
    passes: Stat,
    commands: Stat,
    draws: Stat,
    /// Live small (bump-packed) scratch chunks.
    ///
    /// Cleared every `begin_frame`, so the peak is a single-frame peak.
    scratch_small_blocks: Stat,
    /// Live oversized scratch chunks.
    ///
    /// Expected 0 in d3d9; a non-zero peak flags a payload exceeding the
    /// chunk size.
    scratch_oversized_blocks: Stat,
    /// Live scratch-arena bytes (capacity held at frame emit).
    scratch_bytes: Stat,
    /// Resident `Vec<Command>` footprint across all passes (sum) and the busiest frame (peak).
    ///
    /// The peak is monotonic since the pool keeps the vectors. Paired with
    /// `cmd_vec_realloc_bytes` (footprint vs churn).
    cmd_vec_capacity_bytes: Stat,
    /// Resident `FrameData.ops` `Vec<Op>` footprint sampled at `stamp_and_swap`.
    ///
    /// Paired with `op_vec_realloc_bytes`.
    op_vec_capacity_bytes: Stat,
    frame_total: Stat,
    outside_d3d9: Stat,
    /// `api_cyc` = Σ of all raw `ApiCategory` buckets.
    ///
    /// The peak includes the `send_frame` stall baked into Device. Shown
    /// as the "D3D9 calls" row.
    api_cyc: Stat,
    api_work: Stat,
    present_block: Stat,
    query_wait: Stat,
    enc_cyc: Stat,
    enc_work: Stat,
    op_cyc: Stat,
    /// Each [`OpSub`] phase — decomposes `op_cyc` for the "Closures (op)" tree.
    op_sub: [Stat; OpSub::COUNT],
    /// Each [`OpSubDetail`] child of `Resolve`/`Binds`.
    op_sub_detail: [Stat; OpSubDetail::COUNT],
    /// Pipeline-resolve memo hits / calls — rendered as a hit rate (sum only).
    pipeline_memo_hits: Stat,
    pipeline_memo_calls: Stat,
    submit_cyc: Stat,
    drawable_wait: Stat,
    submit_exec: Stat,
    submit_stall: Stat,
    /// Per-`ApiCategory` bucket: window sum + per-frame peak.
    ///
    /// The peak surfaces a category that spikes (Device, Texture, …) on a
    /// hitch frame instead of hiding inside the aggregate `api_work`.
    api_by: [Stat; ApiCategory::COUNT],
    /// Per-`ApiCategory` call counts (sum only).
    calls_by: [Stat; ApiCategory::COUNT],
    /// Per-`DeviceSubCategory` bucket: sum + per-frame peak.
    device_sub_by: [Stat; DeviceSubCategory::COUNT],
    /// Per-`DeviceSubCategory` call counts (sum only).
    device_sub_calls_by: [Stat; DeviceSubCategory::COUNT],
    /// Per-`BindSubCategory` bucket: sum + per-frame peak.
    bind_sub_by: [Stat; BindSubCategory::COUNT],
    /// Per-`BindSubCategory` call counts (sum only).
    bind_sub_calls_by: [Stat; BindSubCategory::COUNT],
    /// Per-[`KeysGate`] live setter calls / skips (sum only).
    keys_gate_calls_by: [Stat; KeysGate::COUNT],
    keys_gate_skips_by: [Stat; KeysGate::COUNT],
    draw_snapshot: Stat,
    draw_snapshot_stages: Stat,
    draw_snapshot_c_ff: Stat,
    draw_snapshot_c_pr: Stat,
    /// `VDECL/RS/RT_DS/VARIANT/VS_SOURCE/PS_SOURCE` block inside `snapshot_shared`.
    draw_snapshot_keys: Stat,
    /// Post-consts scratch bumps + cache assignments + snapshot-wrapper bump.
    draw_snapshot_bumps: Stat,
    /// Peak only: the unaccounted residual of the `draw_snapshot` breakdown.
    ///
    /// `draw_snapshot − stages − c_ff − c_pr − keys − bumps`. Near zero
    /// expected; computed per-frame because
    /// peak-of-difference ≠ difference-of-peaks.
    draw_snapshot_leftover: Stat,
    draw_push_op: Stat,
    vb_rename: Stat,
    ib_rename: Stat,
    vb_discards: Stat,
    ib_discards: Stat,
    vbib_preserve_cpu: Stat,
    /// `Staged` VB/IB dirty-range upload blits.
    ///
    /// The separate-staging path's headline counter (replaces renames for
    /// non-DYNAMIC world geometry).
    vbib_staging_uploads: Stat,
    /// Inline staging uploads overwriting a region drawn earlier in the open pass.
    ///
    /// The deciding number for a hybrid upload model.
    vbib_mid_pass_reorders: Stat,
    alloc_recovery_drain: Stat,
    /// Heavy-tier recoveries (~1-2 ms each).
    ///
    /// The peak surfaces a single frame with > 1 as a stutter signal.
    alloc_recovery_submit: Stat,
    buffer_destroys: Stat,
    /// Retention depth: window sum (averaged) + entry-count peak.
    ///
    /// Paired with the bytes peak, lets readers estimate per-buffer size.
    vbib_retention_depth: Stat,
    vbib_retained_bytes: Stat,
    texture_renames: Stat,
    texture_discards: Stat,
    /// No GPU analog (texture staging is PE-side, `MTLTexture` handles aren't swapped on rename).
    texture_preserve_cpu: Stat,
    /// `AddDirtyRect` probe (see `FrameCounters::texture_add_dirty_*`).
    ///
    /// `area_bp` sum / `calls` sum at render gives average mip coverage.
    texture_add_dirty_calls: Stat,
    texture_add_dirty_partial: Stat,
    texture_add_dirty_area_bp: Stat,
    texture_destroys: Stat,
    texture_blit_uploads: Stat,
    texture_blit_padded_uploads: Stat,
    texture_gpu_renames: Stat,
    pending_blit_retention_depth: Stat,
    tex_staging_retained_bytes: Stat,
    /// Bytes memcpy'd by `Pass::commands` Vec doublings.
    ///
    /// The `realloc=` half of the `cmd_vec` row (sum across the window +
    /// outlier peak).
    cmd_vec_realloc_bytes: Stat,
    /// Bytes memcpy'd by `FrameData.ops` Vec doublings on the API thread.
    ///
    /// The `realloc=` half of the `op_vec` row (sum + outlier peak).
    op_vec_realloc_bytes: Stat,
    /// Peak only: `device_sub_by[Frame] − present_block`, the non-stall Frame sub-bucket.
    ///
    /// Present body, `Clear`, `Begin/EndScene`, `ColorFill`.
    frame_other: Stat,
    /// Peak only: `op_cyc − Σ op_sub`, the uninstrumented op-loop residual.
    op_leftover: Stat,
    /// Peak only: `Resolve`/`Binds` parent-minus-children residuals.
    resolve_leftover: Stat,
    binds_leftover: Stat,
    /// Peak only: `submit_cyc − submit_stall` (the finalize CPU).
    finalize: Stat,
    /// Peak only: `submit_exec − drawable_wait`.
    ///
    /// Encode+commit CPU, excluding the `nextDrawable` GPU wait.
    encode_commit: Stat,
    /// Peak only: `vb_rename + ib_rename` on any single frame.
    vbib_rename: Stat,
    last_submit_status: i32,
}

#[cfg(perf_tracking)]
impl PerfWindow {
    /// All-zero window.
    ///
    /// Every [`Stat`] (and the three plain fields) defaults to 0, so a new
    /// window is just `Default`.
    fn new() -> Self {
        Self::default()
    }

    fn accumulate(&mut self, s: &FrameSample) {
        if self.started_tsc == 0 {
            self.started_tsc = rdtsc();
        }
        self.frames += 1;
        // Paired metrics: `Stat::add` folds the value into both the window
        // sum and the per-frame peak. Sum-only metrics also land in `.max`,
        // but render never reads that half for them.
        self.passes.add(u64::from(s.passes));
        self.commands.add(u64::from(s.commands));
        self.draws.add(u64::from(s.draws));
        self.scratch_small_blocks
            .add(u64::from(s.scratch_small_blocks));
        self.scratch_oversized_blocks
            .add(u64::from(s.scratch_oversized_blocks));
        self.scratch_bytes.add(s.scratch_bytes);
        self.cmd_vec_capacity_bytes.add(s.cmd_vec_capacity_bytes);
        self.op_vec_capacity_bytes
            .add(s.timing.op_vec_capacity_bytes);
        self.frame_total.add(s.timing.frame_total_cycles);
        self.outside_d3d9.add(s.outside_d3d9);
        self.api_cyc.add(s.api_cyc);
        self.api_work.add(s.api_work);
        self.present_block.add(s.timing.present_block_cycles);
        self.query_wait.add(s.counters.query_wait_cycles);
        self.enc_cyc.add(s.enc_cyc);
        // Encoder-thread CPU = op + finalize. `enc_cyc` (= op + submit_cyc)
        // also contains the backpressure stall; subtract it so `enc_work`
        // reports real work, not time spent blocked on the submit thread.
        // `drawable_wait` is submit-thread time and was never in `enc_cyc`.
        self.enc_work
            .add(s.enc_cyc.saturating_sub(s.enc.submit_stall_cycles));
        self.op_cyc.add(s.enc.op_cycles);
        for i in 0..OpSub::COUNT {
            self.op_sub[i].add(s.enc.op_sub_cycles[i]);
        }
        for i in 0..OpSubDetail::COUNT {
            self.op_sub_detail[i].add(s.enc.op_sub_detail[i]);
        }
        self.pipeline_memo_hits
            .add(u64::from(s.enc.pipeline_memo_hits));
        self.pipeline_memo_calls
            .add(u64::from(s.enc.pipeline_memo_calls));
        self.submit_cyc.add(s.enc.submit_cycles);
        self.submit_stall.add(s.enc.submit_stall_cycles);
        self.drawable_wait.add(s.enc.drawable_wait_cycles);
        self.submit_exec.add(s.enc.submit_exec_cycles);
        for i in 0..ApiCategory::COUNT {
            self.api_by[i].add(s.counters.api_cycles_by_category[i]);
            self.calls_by[i].add(u64::from(s.counters.api_call_counts_by_category[i]));
        }
        for i in 0..DeviceSubCategory::COUNT {
            self.device_sub_by[i].add(s.counters.device_sub_cycles[i]);
            self.device_sub_calls_by[i].add(u64::from(s.counters.device_sub_calls[i]));
        }
        for i in 0..BindSubCategory::COUNT {
            self.bind_sub_by[i].add(s.counters.bind_sub_cycles[i]);
            self.bind_sub_calls_by[i].add(u64::from(s.counters.bind_sub_calls[i]));
        }
        for i in 0..KeysGate::COUNT {
            self.keys_gate_calls_by[i].add(u64::from(s.counters.keys_gate_calls[i]));
            self.keys_gate_skips_by[i].add(u64::from(s.counters.keys_gate_skips[i]));
        }
        self.draw_snapshot.add(s.counters.draw_snapshot_cycles);
        self.draw_snapshot_stages
            .add(s.counters.draw_snapshot_stages_cycles);
        self.draw_snapshot_c_ff
            .add(s.counters.draw_snapshot_c_ff_cycles);
        self.draw_snapshot_c_pr
            .add(s.counters.draw_snapshot_c_pr_cycles);
        self.draw_snapshot_keys
            .add(s.counters.draw_snapshot_keys_cycles);
        self.draw_snapshot_bumps
            .add(s.counters.draw_snapshot_bumps_cycles);
        self.draw_push_op.add(s.counters.draw_push_op_cycles);
        self.vb_rename.add(u64::from(s.counters.vb_rename));
        self.ib_rename.add(u64::from(s.counters.ib_rename));
        self.vb_discards.add(u64::from(s.counters.vb_discards));
        self.ib_discards.add(u64::from(s.counters.ib_discards));
        self.vbib_preserve_cpu
            .add(u64::from(s.counters.vbib_preserve_cpu));
        self.vbib_staging_uploads
            .add(u64::from(s.enc.vbib_staging_uploads));
        self.vbib_mid_pass_reorders
            .add(u64::from(s.enc.vbib_mid_pass_reorders));
        self.alloc_recovery_drain
            .add(u64::from(s.counters.alloc_recovery_drain));
        self.alloc_recovery_submit
            .add(u64::from(s.counters.alloc_recovery_submit));
        self.buffer_destroys.add(u64::from(s.enc.buffer_destroys));
        self.vbib_retention_depth.add(s.vbib_retention_depth as u64);
        self.vbib_retained_bytes.add(s.vbib_retained_bytes as u64);
        self.texture_renames
            .add(u64::from(s.counters.texture_renames));
        self.texture_discards
            .add(u64::from(s.counters.texture_discards));
        self.texture_preserve_cpu
            .add(u64::from(s.counters.texture_preserve_cpu));
        self.texture_add_dirty_calls
            .add(u64::from(s.counters.texture_add_dirty_calls));
        self.texture_add_dirty_partial
            .add(u64::from(s.counters.texture_add_dirty_partial));
        self.texture_add_dirty_area_bp
            .add(u64::from(s.counters.texture_add_dirty_area_bp));
        self.texture_destroys.add(u64::from(s.enc.texture_destroys));
        self.texture_blit_uploads
            .add(u64::from(s.enc.texture_blit_uploads));
        self.texture_blit_padded_uploads
            .add(u64::from(s.enc.texture_blit_padded_uploads));
        self.texture_gpu_renames
            .add(u64::from(s.enc.texture_gpu_renames));
        self.pending_blit_retention_depth
            .add(s.pending_blit_retention_depth as u64);
        self.tex_staging_retained_bytes
            .add(s.tex_staging_retained_bytes as u64);
        self.cmd_vec_realloc_bytes.add(s.cmd_vec_realloc_bytes);
        self.op_vec_realloc_bytes.add(s.timing.op_vec_realloc_bytes);

        // Derived peaks (no window sum): each is a per-frame quantity, so
        // peak-of-difference ≠ difference-of-peaks — compute per frame.
        let op_sub_sum: u64 = s.enc.op_sub_cycles.iter().sum();
        self.op_leftover
            .peak(s.enc.op_cycles.saturating_sub(op_sub_sum));
        // Per-parent residual peak: parent phase minus its three children.
        let r_children = s.enc.op_sub_detail[OpSubDetail::RConsts as usize]
            + s.enc.op_sub_detail[OpSubDetail::RKeys as usize]
            + s.enc.op_sub_detail[OpSubDetail::RLookup as usize];
        let b_children = s.enc.op_sub_detail[OpSubDetail::BCbind as usize]
            + s.enc.op_sub_detail[OpSubDetail::BVbib as usize]
            + s.enc.op_sub_detail[OpSubDetail::BDraw as usize];
        self.resolve_leftover
            .peak(s.enc.op_sub_cycles[OpSub::Resolve as usize].saturating_sub(r_children));
        self.binds_leftover
            .peak(s.enc.op_sub_cycles[OpSub::Binds as usize].saturating_sub(b_children));
        // Snapshot leftover — work outside every named sub-timer. Near-zero
        // expected (measurement noise + inter-scope branch overhead); a
        // sustained value flags uninstrumented work warranting a sub-timer.
        let snapshot_leftover = s
            .counters
            .draw_snapshot_cycles
            .saturating_sub(s.counters.draw_snapshot_stages_cycles)
            .saturating_sub(s.counters.draw_snapshot_c_ff_cycles)
            .saturating_sub(s.counters.draw_snapshot_c_pr_cycles)
            .saturating_sub(s.counters.draw_snapshot_keys_cycles)
            .saturating_sub(s.counters.draw_snapshot_bumps_cycles);
        self.draw_snapshot_leftover.peak(snapshot_leftover);
        let frame_other = s.counters.device_sub_cycles[DeviceSubCategory::Frame as usize]
            .saturating_sub(s.timing.present_block_cycles);
        self.frame_other.peak(frame_other);
        self.finalize.peak(
            s.enc
                .submit_cycles
                .saturating_sub(s.enc.submit_stall_cycles),
        );
        self.encode_commit.peak(
            s.enc
                .submit_exec_cycles
                .saturating_sub(s.enc.drawable_wait_cycles),
        );
        self.vbib_rename
            .peak(u64::from(s.counters.vb_rename) + u64::from(s.counters.ib_rename));
        self.last_submit_status = s.submit_status;
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Cached decision for emitting ANSI escape sequences in the 5-second summary.
///
/// `NO_COLOR=1` wins (<https://no-color.org>); `CLICOLOR_FORCE=1` wins next;
/// otherwise ANSI is on by default. Auto detection on stderr is not used
/// because the PE side runs under Wine, where `is_terminal` on stderr returns
/// false even when macOS fd 2 is a real TTY. Users who pipe the log to a file
/// set `NO_COLOR=1` to get plain text.
#[cfg(perf_tracking)]
fn ansi_enabled() -> bool {
    static CACHE: LazyLock<bool> = LazyLock::new(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if std::env::var_os("CLICOLOR_FORCE").is_some() {
            return true;
        }
        true
    });
    *CACHE
}

#[cfg(perf_tracking)]
struct Style {
    bold: &'static str,
    dim: &'static str,
    green: &'static str,
    yellow: &'static str,
    red: &'static str,
    reset: &'static str,
}

#[cfg(perf_tracking)]
impl Style {
    const fn from_flag(ansi: bool) -> Self {
        if ansi {
            Self {
                bold: "\x1b[1m",
                dim: "\x1b[2m",
                green: "\x1b[32m",
                yellow: "\x1b[33m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        } else {
            Self {
                bold: "",
                dim: "",
                green: "",
                yellow: "",
                red: "",
                reset: "",
            }
        }
    }
}

/// Classification of which side of the pipeline is pacing the frame.
///
/// Derived from the four terminal time buckets (`api_work`, `outside_d3d9`,
/// encoder CPU, GPU wait) plus the API-thread present-stall.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg(perf_tracking)]
enum Bottleneck {
    ApiD3d9,
    ApiGame,
    EncoderCpu,
    EncoderGpu,
    Balanced,
}

#[cfg(perf_tracking)]
impl Bottleneck {
    /// All inputs in ms/frame averages.
    ///
    /// `present_block > 15 % of frame_total` wins first: that's the signal
    /// that the API thread is stalling on the backpressure channel, so the
    /// encoder pipeline is strictly longer than the API thread and the
    /// bottleneck lives there — attribute by whichever of encoder-CPU vs
    /// GPU-wait dominates. Otherwise the API thread is the pacing side and
    /// we ask whether D3D9 or game code on the API thread is the dominant
    /// consumer. `Balanced` only fires when no bucket clearly leads (all
    /// four within 20 % of the per-bucket mean).
    fn classify(
        frame_total: f64,
        api_work: f64,
        outside: f64,
        enc_cpu: f64,
        gpu_wait: f64,
        present_block: f64,
    ) -> Self {
        if frame_total > 0.0 && present_block > 0.15 * frame_total {
            return if gpu_wait > enc_cpu {
                Self::EncoderGpu
            } else {
                Self::EncoderCpu
            };
        }
        if frame_total > 0.0 {
            let mean = frame_total / 4.0;
            let band = 0.20 * frame_total;
            let all_within = [api_work, outside, enc_cpu, gpu_wait]
                .iter()
                .all(|&b| (b - mean).abs() < band);
            if all_within {
                return Self::Balanced;
            }
        }
        if api_work > outside {
            Self::ApiD3d9
        } else {
            Self::ApiGame
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ApiD3d9 => "API (D3D9)",
            Self::ApiGame => "API (game)",
            Self::EncoderCpu => "ENCODER (CPU)",
            Self::EncoderGpu => "ENCODER (GPU)",
            Self::Balanced => "balanced",
        }
    }

    const fn color(self, s: &Style) -> &str {
        match self {
            Self::ApiD3d9 | Self::ApiGame => s.green,
            Self::EncoderCpu => s.yellow,
            Self::EncoderGpu => s.red,
            Self::Balanced => s.dim,
        }
    }
}

/// Multi-line renderer for the 5-second summary.
///
/// Pure: takes a `PerfWindow` snapshot plus the live cache sizes and produces
/// a `String`. Kept off `log_frame_summary` so it's host-testable under
/// `cargo test -p mtld3d-core --target x86_64-apple-darwin`.
#[cfg(perf_tracking)]
struct Summary<'a> {
    w: &'a PerfWindow,
    caches: &'a CacheSizes,
    window_secs: f64,
    frames: u32,
    s: Style,
}

/// Column grid shared by every row in the summary.
///
/// Each cell starts at a fixed column; missing cells leave their column empty
/// without shifting later cells. Sized so the widest plausible contents fit:
/// 8-digit call counts, MB retention pairs, 20-char descriptions.
///
/// ```text
/// col  0..22  label + tree chars   ("│  ├─ Device")
/// col 22..31  ms value             ("  1.72 ms", right-aligned)
/// col 34..52  aux                  ("(  8 678 846)" / "( 42.2 %)")
/// col 54..74  description (dim)    ("encoder backpressure")
/// col 76..89  peak (dim)           ("peak  2.20 ms")
/// ```
#[cfg(perf_tracking)]
const LABEL_W: usize = 22;
#[cfg(perf_tracking)]
const AUX_COL: usize = 34;
#[cfg(perf_tracking)]
const DESC_COL: usize = 54;
#[cfg(perf_tracking)]
const PEAK_COL: usize = 76;

/// Column where the comment cell starts on Resources rows.
///
/// The shape differs from tree rows: Resources uses a (label, window,
/// peak/frame, comment) grid without tree chars.
///
/// ```text
/// col  0..12  label                ("fresh     ")
/// col 12..34  window cell          ("VB=335    IB=18")
/// col 36..62  peak/frame cell      ("peak depth=6   peak 5.8 MB")
/// col 64..    comment (dim)
/// ```
#[cfg(perf_tracking)]
const RES_WINDOW_COL: usize = 12;
#[cfg(perf_tracking)]
const RES_PEAK_COL: usize = 36;
#[cfg(perf_tracking)]
const RES_COMMENT_COL: usize = 64;

/// Compute and format a `(N ns/draw)` aux cell.
///
/// For a row whose cycle accumulator scales with draw count.
///
/// `cycles` is the window-total cycles for the row; `draws` is the
/// window-total draw count. Returns `None` when there are no draws
/// (login / static menus / empty scene) so non-camera captures don't
/// show a divide-by-zero or noise. Integer ns is the natural format
/// for the magnitudes we see (50-3000 ns/draw) and avoids `tsc_hz`
/// calibration jitter at the LSD that a fractional format would
/// expose (the golden test would be flaky).
#[cfg(perf_tracking)]
fn ns_per_draw_aux(cycles: u64, draws: u64) -> Option<String> {
    if draws == 0 {
        return None;
    }
    let ns = cycles_to_ms(cycles) * 1e6 / mtld3d_shared::tsc::u64_to_f64_exact(draws);
    Some(format!("({ns:>4.0} ns/draw)"))
}

/// Compute and format a `(N ns/call)` aux cell for a per-Set device sub-row.
///
/// The row's cycle total `cycles` was accumulated over `calls` setter
/// invocations. Mirrors [`ns_per_draw_aux`] but divides by the row's own
/// call count, so `RenderState` / `TexStageState` / `SamplerState` /
/// `ShaderConst` / `Bind` report a per-call cost directly comparable across
/// captures with different call volumes. Returns `None` when `calls == 0`
/// so the row falls back to the plain `(N)` count.
#[cfg(perf_tracking)]
fn ns_per_call_aux(cycles: u64, calls: u64) -> Option<String> {
    if calls == 0 {
        return None;
    }
    let ns = cycles_to_ms(cycles) * 1e6 / mtld3d_shared::tsc::u64_to_f64_exact(calls);
    Some(format!("({ns:>4.0} ns/call)"))
}

/// Format a (avg, peak) pair of KB values as short strings that share the same unit.
///
/// Above 1024 KB, both switch to MB so the two numbers are directly
/// comparable at a glance. Returns `(avg_str, peak_str)`.
#[cfg(perf_tracking)]
fn format_kb_pair(avg_kb: f64, peak_kb: f64) -> (String, String) {
    let max = avg_kb.max(peak_kb);
    if max >= 1024.0 {
        (
            format!("{:.1} MB", avg_kb / 1024.0),
            format!("{:.1} MB", peak_kb / 1024.0),
        )
    } else {
        (format!("{avg_kb:.0} KB"), format!("{peak_kb:.0} KB"))
    }
}

/// Pad `out`'s visible cursor up to `target` by appending ASCII spaces.
///
/// If the cursor is already past the target, appends a single space
/// separator so the next cell never butts up against the previous one.
/// `cursor == target` emits nothing — the previous cell ended exactly at
/// the column we want and the next cell starts immediately. Tracks a
/// display-column cursor separately from `out.len()` so multibyte tree
/// chars (`│` / `├─` / `└─`) and ANSI escapes stay out of the width
/// accounting.
#[cfg(perf_tracking)]
fn pad_spaces(out: &mut String, cursor: &mut usize, target: usize) {
    if *cursor > target {
        out.push(' ');
        *cursor += 1;
        return;
    }
    while *cursor < target {
        out.push(' ');
        *cursor += 1;
    }
}

/// Column-grid row for the API-thread / Encoder-thread / Frame-total / Outside-d3d9 trees.
///
/// Each cell lands at a fixed column driven by the `LABEL_W` / `AUX_COL` /
/// `DESC_COL` / `PEAK_COL` constants; missing cells render as blank space
/// without shifting later cells.
#[cfg(perf_tracking)]
struct Row<'a> {
    /// Label with tree chars baked in.
    ///
    /// Plain text, no ANSI. Measured as display columns via
    /// `chars().count()`.
    label: &'a str,
    /// Wrap the label with `Style.bold` when true.
    ///
    /// Styling is applied after width measurement, so ANSI escapes never
    /// widen the cell.
    bold_label: bool,
    /// ms value, right-aligned 6.2 with a trailing `" ms"`.
    ///
    /// `None` leaves the cell blank — caller explicitly omits ms to signal
    /// "this row has no time reading."
    ms: Option<f64>,
    /// Free-text aux cell (percent, parenthesised call count, …).
    ///
    /// Left-aligned starting at `AUX_COL`.
    aux: Option<String>,
    /// Dim-styled description ("encoder backpressure", "game code").
    ///
    /// Starts at `DESC_COL`.
    desc: Option<&'a str>,
    /// Dim-styled peak cell formatted as `"peak {val:>5.2} ms"` at `PEAK_COL`.
    ///
    /// The constant-width format keeps the cell's right edge stable
    /// regardless of magnitude.
    peak: Option<f64>,
}

/// Emit one grid row into `out`.
///
/// Every cell is placed at its fixed column using `pad_spaces`; ANSI
/// escapes wrap cell contents after width is measured so the visible
/// layout matches ansi-off.
#[cfg(perf_tracking)]
fn write_row(out: &mut String, s: &Style, row: &Row<'_>) {
    let mut cursor: usize = 0;
    if row.bold_label {
        out.push_str(s.bold);
        out.push_str(row.label);
        out.push_str(s.reset);
    } else {
        out.push_str(row.label);
    }
    cursor += row.label.chars().count();

    if let Some(ms) = row.ms {
        pad_spaces(out, &mut cursor, LABEL_W);
        let cell = format!("{ms:>6.2} ms");
        out.push_str(&cell);
        cursor += cell.chars().count();
    }

    if let Some(aux) = row.aux.as_deref() {
        pad_spaces(out, &mut cursor, AUX_COL);
        out.push_str(aux);
        cursor += aux.chars().count();
    }

    if let Some(desc) = row.desc {
        pad_spaces(out, &mut cursor, DESC_COL);
        out.push_str(s.dim);
        out.push_str(desc);
        out.push_str(s.reset);
        cursor += desc.chars().count();
    }

    if let Some(peak) = row.peak {
        pad_spaces(out, &mut cursor, PEAK_COL);
        let cell = format!("peak {peak:>5.2} ms");
        out.push_str(s.dim);
        out.push_str(&cell);
        out.push_str(s.reset);
        cursor += cell.chars().count();
    }

    let _ = cursor;
    out.push('\n');
}

#[cfg(perf_tracking)]
impl<'a> Summary<'a> {
    fn render(w: &'a PerfWindow, caches: &'a CacheSizes, window_secs: f64) -> String {
        Self::render_with_ansi(w, caches, window_secs, ansi_enabled())
    }

    fn render_with_ansi(
        w: &'a PerfWindow,
        caches: &'a CacheSizes,
        window_secs: f64,
        ansi: bool,
    ) -> String {
        let this = Self {
            w,
            caches,
            window_secs,
            frames: w.frames.max(1),
            s: Style::from_flag(ansi),
        };
        this.build()
    }

    fn build(&self) -> String {
        let w = self.w;
        let f = u64::from(self.frames);

        let frame_total_ms = cycles_to_ms(w.frame_total.sum / f);
        let api_work_ms = cycles_to_ms(w.api_work.sum / f);
        let api_cyc_ms = cycles_to_ms(w.api_cyc.sum / f);
        let outside_ms = cycles_to_ms(w.outside_d3d9.sum / f);
        let present_ms = cycles_to_ms(w.present_block.sum / f);
        let query_wait_ms = cycles_to_ms(w.query_wait.sum / f);
        let enc_cyc_ms = cycles_to_ms(w.enc_cyc.sum / f);
        let op_ms = cycles_to_ms(w.op_cyc.sum / f);
        let stall_ms = cycles_to_ms(w.submit_stall.sum / f);
        // Finalize = submit-cycle minus the backpressure stall (the stall is
        // shown as its own encoder-thread row, not encoder work).
        let finalize_ms = cycles_to_ms(w.submit_cyc.sum.saturating_sub(w.submit_stall.sum) / f);
        let dw_ms = cycles_to_ms(w.drawable_wait.sum / f);
        let submit_exec_ms = cycles_to_ms(w.submit_exec.sum / f);
        let enc_work_ms = cycles_to_ms(w.enc_work.sum / f);
        // Submit-thread CPU = total execute minus the GPU/compositor wait.
        let encode_commit_ms = (submit_exec_ms - dw_ms).max(0.0);

        let bn = Bottleneck::classify(
            frame_total_ms,
            api_work_ms,
            outside_ms,
            enc_work_ms,
            dw_ms,
            present_ms,
        );

        let mut out = String::with_capacity(2048);
        self.write_header(&mut out, bn, frame_total_ms);
        self.write_bucket_summary(
            &mut out,
            api_work_ms,
            outside_ms,
            enc_work_ms,
            encode_commit_ms,
            dw_ms,
        );
        self.write_api_thread(
            &mut out,
            frame_total_ms,
            api_cyc_ms,
            present_ms,
            query_wait_ms,
            outside_ms,
        );
        self.write_encoder_thread(&mut out, enc_cyc_ms, op_ms, finalize_ms, stall_ms);
        self.write_submit_thread(&mut out, submit_exec_ms, encode_commit_ms, dw_ms);
        self.write_frame_total(&mut out, frame_total_ms);
        self.write_resources_vbib(&mut out);
        self.write_resources_textures(&mut out);
        self.write_caches(&mut out);
        self.write_commands_passes(&mut out);
        self.write_keys_gating(&mut out);
        self.write_alloc_footprint(&mut out);

        if out.ends_with('\n') {
            out.pop();
        }
        out
    }

    fn write_header(&self, out: &mut String, bn: Bottleneck, _frame_total_ms: f64) {
        let s = &self.s;
        let _ = writeln!(
            out,
            "{b}── perf  window={wsec:.2}s  frames={frames}  bottleneck={c}{label}{r}{b} ──{r}",
            b = s.bold,
            r = s.reset,
            c = bn.color(s),
            wsec = self.window_secs,
            frames = self.w.frames,
            label = bn.label(),
        );
    }

    fn write_bucket_summary(
        &self,
        out: &mut String,
        api_work_ms: f64,
        outside_ms: f64,
        enc_cpu_ms: f64,
        submit_cpu_ms: f64,
        dw_ms: f64,
    ) {
        let s = &self.s;
        let _ = writeln!(
            out,
            "{d}buckets:{r} api_d3d9={api:.2}  api_outside={ots:.2}  enc_work={ec:.2}  submit_work={sc:.2}  gpu_wait={gw:.2}  {d}(ms/frame, avg){r}",
            d = s.dim,
            r = s.reset,
            api = api_work_ms,
            ots = outside_ms,
            ec = enc_cpu_ms,
            sc = submit_cpu_ms,
            gw = dw_ms,
        );
    }

    fn write_api_thread(
        &self,
        out: &mut String,
        frame_total_ms: f64,
        api_cyc_ms: f64,
        present_ms: f64,
        query_wait_ms: f64,
        outside_ms: f64,
    ) {
        let w = self.w;
        let f = u64::from(self.frames);
        let s = &self.s;
        let pct = |x: f64| {
            if frame_total_ms > 0.0 {
                x / frame_total_ms * 100.0
            } else {
                0.0
            }
        };

        let _ = writeln!(out);
        // Top-level parent: no aux, no desc, only ms + peak.
        write_row(
            out,
            s,
            &Row {
                label: "API thread",
                bold_label: true,
                ms: Some(frame_total_ms),
                aux: None,
                desc: None,
                peak: Some(cycles_to_ms(w.frame_total.max)),
            },
        );

        let total_calls: u64 = w.calls_by.iter().map(|c| c.sum).sum();
        // `D3D9 calls` reports `api_cyc` (sum of raw buckets). The
        // Device bucket's raw value includes `send_frame` backpressure
        // wait, which nests below as a `Present stall` subtimer of
        // Device — that way every row sums cleanly into its parent.
        let d3d9_calls_desc = format!("{total_calls} calls");
        write_row(
            out,
            s,
            &Row {
                label: "├─ D3D9 calls",
                bold_label: false,
                ms: Some(api_cyc_ms),
                aux: Some(format!("({pct:>5.1} %)", pct = pct(api_cyc_ms))),
                desc: Some(&d3d9_calls_desc),
                peak: Some(cycles_to_ms(w.api_cyc.max)),
            },
        );

        let cat_rows: &[(&str, ApiCategory)] = &[
            ("Device", ApiCategory::Device),
            ("VertexBuffer", ApiCategory::VertexBuffer),
            ("IndexBuffer", ApiCategory::IndexBuffer),
            ("Texture", ApiCategory::Texture),
            ("Surface", ApiCategory::Surface),
            ("Query", ApiCategory::Query),
            ("StateBlock", ApiCategory::StateBlock),
            ("VertexDecl", ApiCategory::VertexDecl),
            ("VertexShader", ApiCategory::VertexShader),
            ("PixelShader", ApiCategory::PixelShader),
        ];
        for (i, (name, cat)) in cat_rows.iter().enumerate() {
            let branch = if i + 1 == cat_rows.len() {
                "└─"
            } else {
                "├─"
            };
            let label = format!("│  {branch} {name}");
            let avg_ms = cycles_to_ms(w.api_by[*cat as usize].sum / f);
            let calls = w.calls_by[*cat as usize].sum;
            write_row(
                out,
                s,
                &Row {
                    label: &label,
                    bold_label: false,
                    ms: Some(avg_ms),
                    aux: Some(format!("({calls:>10})")),
                    desc: None,
                    peak: Some(cycles_to_ms(w.api_by[*cat as usize].max)),
                },
            );
            // Decompose `Device` into its named sub-buckets so the
            // reader can see which class of vtable entries is hot
            // — draws vs the per-batch state-setter storms vs frame
            // boundaries vs the long tail. `Present stall` nests
            // under `Frame` because the stall lives inside the
            // `device_present` body. The sub-buckets sum cleanly to
            // the parent `Device` row.
            if *cat as usize == ApiCategory::Device as usize {
                self.write_device_sub_rows(out, present_ms);
            }
            // Nest `Wait for GPU` under `Query`: the raw Query bucket
            // accumulates every GetData / Issue / GetType call; the
            // FLUSH-on-Pending wait (Metal `waitUntilCompleted`) is
            // measured separately so the numbers split cleanly into
            // (cached-read CPU) + (kernel sleep on the GPU fence).
            if *cat as usize == ApiCategory::Query as usize {
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  └─ Wait for GPU",
                        bold_label: false,
                        ms: Some(query_wait_ms),
                        aux: None,
                        desc: Some("waitUntilCompleted"),
                        peak: Some(cycles_to_ms(w.query_wait.max)),
                    },
                );
            }
        }

        write_row(
            out,
            s,
            &Row {
                label: "└─ Outside d3d9",
                bold_label: false,
                ms: Some(outside_ms),
                aux: Some(format!("({pct:>5.1} %)", pct = pct(outside_ms))),
                desc: Some("game code"),
                peak: Some(cycles_to_ms(w.outside_d3d9.max)),
            },
        );
    }

    /// Render the per-`DeviceSubCategory` rows nested under the top-level `Device` row.
    ///
    /// Lets the reader see which sub-bucket (`Draws` / `RenderState` /
    /// …) spiked even when the parent average looks benign. `Frame`
    /// carries `Present stall` as a child; `Draws` carries `snapshot`
    /// + `push_op` children.
    fn write_device_sub_rows(&self, out: &mut String, present_ms: f64) {
        let w = self.w;
        let f = u64::from(self.frames);
        let s = &self.s;
        let sub_rows: &[(&str, DeviceSubCategory)] = &[
            ("Frame", DeviceSubCategory::Frame),
            ("Draws", DeviceSubCategory::Draws),
            ("RenderState", DeviceSubCategory::RenderState),
            ("TexStageState", DeviceSubCategory::TexStageState),
            ("SamplerState", DeviceSubCategory::SamplerState),
            ("ShaderConst", DeviceSubCategory::ShaderConst),
            ("Bind", DeviceSubCategory::Bind),
            ("StateBlock", DeviceSubCategory::StateBlock),
            ("Misc", DeviceSubCategory::Misc),
        ];
        let draws_total = w.draws.sum;
        for (i, (name, sub)) in sub_rows.iter().enumerate() {
            let branch = if i + 1 == sub_rows.len() {
                "└─"
            } else {
                "├─"
            };
            let idx = *sub as usize;
            let label = format!("│  │  {branch} {name}");
            let avg_ms = cycles_to_ms(w.device_sub_by[idx].sum / f);
            let calls = w.device_sub_calls_by[idx].sum;
            // Draws reports `(N ns/draw)` (each call IS one draw); the
            // per-Set rows report `(N ns/call)` against their own call
            // count so the magnitude stays comparable across captures
            // with different call volumes. Frame / StateBlock / Misc keep
            // the plain `(N)` count, and any zero-denominator row falls
            // back to it too.
            let aux = match sub {
                DeviceSubCategory::Draws => ns_per_draw_aux(w.device_sub_by[idx].sum, draws_total),
                DeviceSubCategory::RenderState
                | DeviceSubCategory::TexStageState
                | DeviceSubCategory::SamplerState
                | DeviceSubCategory::ShaderConst
                | DeviceSubCategory::Bind => ns_per_call_aux(w.device_sub_by[idx].sum, calls),
                _ => None,
            };
            let aux_cell = aux.unwrap_or_else(|| format!("({calls:>10})"));
            write_row(
                out,
                s,
                &Row {
                    label: &label,
                    bold_label: false,
                    ms: Some(avg_ms),
                    aux: Some(aux_cell),
                    desc: None,
                    peak: Some(cycles_to_ms(w.device_sub_by[idx].max)),
                },
            );
            // `Present stall` is measured by `CycleSetTimer` inside
            // `device_present`'s body — which sits in the `Frame`
            // sub-bucket. Honest nesting (Frame contains Present stall)
            // matches the measurement hierarchy.
            if matches!(sub, DeviceSubCategory::Frame) {
                // Named "Send stall" (not "Present stall") because the
                // 10-char depth-4 label budget (LABEL_W − 12-char tree
                // prefix) forces brevity, and the wait is on the
                // `send_frame` sync_channel call inside Present — the
                // shorter name is also more precise.
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  ├─ Send stall",
                        bold_label: false,
                        ms: Some(present_ms),
                        aux: None,
                        desc: Some("encoder backpressure"),
                        peak: Some(cycles_to_ms(w.present_block.max)),
                    },
                );
                // Make Frame's children sum to Frame: the remainder
                // (Frame total − Send stall) is the non-blocking body
                // work of Present + Clear + BeginScene/EndScene +
                // ColorFill. Derived by subtraction; peak tracked
                // per-frame in `accumulate`.
                let frame_total = w.device_sub_by[DeviceSubCategory::Frame as usize].sum;
                let frame_other_ms =
                    cycles_to_ms(frame_total.saturating_sub(w.present_block.sum) / f);
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  └─ other",
                        bold_label: false,
                        ms: Some(frame_other_ms),
                        aux: None,
                        desc: Some("non-blocking body"),
                        peak: Some(cycles_to_ms(w.frame_other.max)),
                    },
                );
            }
            if matches!(sub, DeviceSubCategory::Draws) {
                let snap_ms = cycles_to_ms(w.draw_snapshot.sum / f);
                let stages_ms = cycles_to_ms(w.draw_snapshot_stages.sum / f);
                let c_ff_ms = cycles_to_ms(w.draw_snapshot_c_ff.sum / f);
                let c_pr_ms = cycles_to_ms(w.draw_snapshot_c_pr.sum / f);
                let keys_ms = cycles_to_ms(w.draw_snapshot_keys.sum / f);
                let bumps_ms = cycles_to_ms(w.draw_snapshot_bumps.sum / f);
                let push_ms = cycles_to_ms(w.draw_push_op.sum / f);
                // Leftover = snapshot − (stages + c_ff + c_pr + keys + bumps).
                // Should be near-zero (measurement noise + branch overhead
                // between scopes). A sustained non-trivial value flags
                // uninstrumented work inside `snapshot_shared` and is the
                // signal to add a new sub-timer.
                let snap_leftover_cyc = w
                    .draw_snapshot
                    .sum
                    .saturating_sub(w.draw_snapshot_stages.sum)
                    .saturating_sub(w.draw_snapshot_c_ff.sum)
                    .saturating_sub(w.draw_snapshot_c_pr.sum)
                    .saturating_sub(w.draw_snapshot_keys.sum)
                    .saturating_sub(w.draw_snapshot_bumps.sum);
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  ├─ snapshot",
                        bold_label: false,
                        ms: Some(snap_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot.sum, draws_total),
                        desc: Some("read+stamp state"),
                        peak: Some(cycles_to_ms(w.draw_snapshot.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  ├─ stages",
                        bold_label: false,
                        ms: Some(stages_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot_stages.sum, draws_total),
                        desc: Some("tex/samp/TSS walk"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_stages.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  ├─ c_ff",
                        bold_label: false,
                        ms: Some(c_ff_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot_c_ff.sum, draws_total),
                        desc: Some("FF VS+PS const build"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_c_ff.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  ├─ c_pr",
                        bold_label: false,
                        ms: Some(c_pr_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot_c_pr.sum, draws_total),
                        desc: Some("programmable consts"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_c_pr.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  ├─ keys",
                        bold_label: false,
                        ms: Some(keys_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot_keys.sum, draws_total),
                        desc: Some("VDECL+RS+variant+srcs"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_keys.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  ├─ bumps",
                        bold_label: false,
                        ms: Some(bumps_ms),
                        aux: ns_per_draw_aux(w.draw_snapshot_bumps.sum, draws_total),
                        desc: Some("scratch+cache+wrapper"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_bumps.max)),
                    },
                );
                // Residual after every named sub-timer — see
                // `PerfWindow::draw_snapshot_leftover` for what a non-zero
                // value here means.
                let snap_leftover_ms = cycles_to_ms(snap_leftover_cyc / f);
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  │  └─ resid",
                        bold_label: false,
                        ms: Some(snap_leftover_ms),
                        aux: ns_per_draw_aux(snap_leftover_cyc, draws_total),
                        desc: Some("uninstrumented noise"),
                        peak: Some(cycles_to_ms(w.draw_snapshot_leftover.max)),
                    },
                );
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  │  └─ push_op",
                        bold_label: false,
                        ms: Some(push_ms),
                        aux: ns_per_draw_aux(w.draw_push_op.sum, draws_total),
                        desc: Some("inline Op::Draw push"),
                        peak: Some(cycles_to_ms(w.draw_push_op.max)),
                    },
                );
            }
            if matches!(sub, DeviceSubCategory::Bind) {
                self.write_bind_sub_rows(out);
            }
        }
    }

    /// Render the per-`BindSubCategory` rows nested under the `Bind` `DeviceSubCategory`.
    ///
    /// Sums to the parent `Bind` row by construction — every
    /// `IDirect3DDevice9` Bind-family method uses `bind_timer`, so there is
    /// no `Misc` escape.
    fn write_bind_sub_rows(&self, out: &mut String) {
        let w = self.w;
        let f = u64::from(self.frames);
        let s = &self.s;
        // Display labels fit the LABEL_W=22 budget at depth 4
        // (12 chars of `│  │  │  ├─ ` tree prefix + ≤10 chars name).
        // `VpScissor` is the trimmed display form of
        // `BindSubCategory::ViewScissor`. Descriptions stay under the
        // 22-char `PEAK_COL − DESC_COL` budget.
        let rows: &[(&str, BindSubCategory, &str)] = &[
            ("Texture", BindSubCategory::Texture, "Set/GetTexture"),
            ("Buffer", BindSubCategory::Buffer, "VB/IB/StreamFreq"),
            ("Shader", BindSubCategory::Shader, "VDecl/VS/PS/FVF"),
            ("RtDs", BindSubCategory::RtDs, "RT + DepthStencil"),
            ("FfFixed", BindSubCategory::FfFixed, "xform/material/light"),
            (
                "VpScissor",
                BindSubCategory::ViewScissor,
                "viewport + scissor",
            ),
        ];
        for (i, (name, sub, desc)) in rows.iter().enumerate() {
            let branch = if i + 1 == rows.len() {
                "└─"
            } else {
                "├─"
            };
            let idx = *sub as usize;
            let label = format!("│  │  │  {branch} {name}");
            let avg_ms = cycles_to_ms(w.bind_sub_by[idx].sum / f);
            let calls = w.bind_sub_calls_by[idx].sum;
            write_row(
                out,
                s,
                &Row {
                    label: &label,
                    bold_label: false,
                    ms: Some(avg_ms),
                    aux: Some(format!("({calls:>10})")),
                    desc: Some(desc),
                    peak: Some(cycles_to_ms(w.bind_sub_by[idx].max)),
                },
            );
        }
    }

    fn write_encoder_thread(
        &self,
        out: &mut String,
        enc_cyc_ms: f64,
        op_ms: f64,
        finalize_ms: f64,
        stall_ms: f64,
    ) {
        let w = self.w;
        let s = &self.s;
        let enc_pct = |x: f64| {
            if enc_cyc_ms > 0.0 {
                x / enc_cyc_ms * 100.0
            } else {
                0.0
            }
        };

        let _ = writeln!(out);
        write_row(
            out,
            s,
            &Row {
                label: "Encoder thread",
                bold_label: true,
                ms: Some(enc_cyc_ms),
                aux: None,
                desc: None,
                peak: None,
            },
        );
        write_row(
            out,
            s,
            &Row {
                label: "├─ Closures (op)",
                bold_label: false,
                ms: Some(op_ms),
                aux: Some(format!("({pct:>5.1} %)", pct = enc_pct(op_ms))),
                desc: Some("D3D9→Metal translate"),
                peak: Some(cycles_to_ms(w.op_cyc.max)),
            },
        );
        // Decompose "Closures (op)" into the six per-draw phases measured
        // inside `emit_draw` (mirrors the API thread's `snapshot` sub-tree),
        // plus a `resid` for the non-`Draw` ops and inter-timer noise.
        let f = u64::from(self.frames);
        let draws_total = w.draws.sum;
        let op_sub_rows: [(&str, &str); OpSub::COUNT] = [
            ("│  ├─ resolve", "tex + shader libs"),
            ("│  ├─ pipeline", "pipeline+depth+cull"),
            ("│  ├─ state", "pass open + binds"),
            ("│  ├─ probe", "decal/caster diag"),
            ("│  ├─ samplers", "texture+sampler bind"),
            ("│  ├─ binds", "consts + VB/IB + draw"),
            ("│  ├─ tex_raw", "texture blit upload"),
            ("│  ├─ stage_up", "VB/IB staged upload"),
            ("│  ├─ const_rng", "VS/PS/FF const copy"),
        ];
        for (i, &(label, desc)) in op_sub_rows.iter().enumerate() {
            write_row(
                out,
                s,
                &Row {
                    label,
                    bold_label: false,
                    ms: Some(cycles_to_ms(w.op_sub[i].sum / f)),
                    aux: ns_per_draw_aux(w.op_sub[i].sum, draws_total),
                    desc: Some(desc),
                    peak: Some(cycles_to_ms(w.op_sub[i].max)),
                },
            );
            // `resolve` and `binds` get a nested second level (the two
            // dominant phases); each child row + a parent-minus-children
            // `resid`, mirroring the API `snapshot` sub-tree.
            let children: &[(&str, &str, usize)] = if i == OpSub::Resolve as usize {
                &[
                    (
                        "│  │  ├─ consts",
                        "VS/PS/FF const copy",
                        OpSubDetail::RConsts as usize,
                    ),
                    (
                        "│  │  ├─ skip",
                        "skip-set check",
                        OpSubDetail::RKeys as usize,
                    ),
                    (
                        "│  │  ├─ lookup",
                        "tex + lib probes",
                        OpSubDetail::RLookup as usize,
                    ),
                ]
            } else if i == OpSub::Binds as usize {
                &[
                    (
                        "│  │  ├─ cbind",
                        "const memcmp+bytes",
                        OpSubDetail::BCbind as usize,
                    ),
                    (
                        "│  │  ├─ vbib",
                        "VB/IB wrap + notify",
                        OpSubDetail::BVbib as usize,
                    ),
                    (
                        "│  │  ├─ draw",
                        "draw cmd emit",
                        OpSubDetail::BDraw as usize,
                    ),
                ]
            } else {
                &[]
            };
            if !children.is_empty() {
                let mut child_sum = 0u64;
                for &(clabel, cdesc, di) in children {
                    child_sum += w.op_sub_detail[di].sum;
                    write_row(
                        out,
                        s,
                        &Row {
                            label: clabel,
                            bold_label: false,
                            ms: Some(cycles_to_ms(w.op_sub_detail[di].sum / f)),
                            aux: ns_per_draw_aux(w.op_sub_detail[di].sum, draws_total),
                            desc: Some(cdesc),
                            peak: Some(cycles_to_ms(w.op_sub_detail[di].max)),
                        },
                    );
                }
                let leftover = w.op_sub[i].sum.saturating_sub(child_sum);
                let leftover_peak = if i == OpSub::Resolve as usize {
                    w.resolve_leftover.max
                } else {
                    w.binds_leftover.max
                };
                write_row(
                    out,
                    s,
                    &Row {
                        label: "│  │  └─ resid",
                        bold_label: false,
                        ms: Some(cycles_to_ms(leftover / f)),
                        aux: ns_per_draw_aux(leftover, draws_total),
                        desc: Some("deref + glue"),
                        peak: Some(cycles_to_ms(leftover_peak)),
                    },
                );
            }
        }
        let op_leftover_cyc = w
            .op_cyc
            .sum
            .saturating_sub(w.op_sub.iter().map(|p| p.sum).sum::<u64>());
        write_row(
            out,
            s,
            &Row {
                label: "│  └─ resid",
                bold_label: false,
                ms: Some(cycles_to_ms(op_leftover_cyc / f)),
                aux: ns_per_draw_aux(op_leftover_cyc, draws_total),
                desc: Some("other ops + noise"),
                peak: Some(cycles_to_ms(w.op_leftover.max)),
            },
        );
        write_row(
            out,
            s,
            &Row {
                label: "├─ Finalize",
                bold_label: false,
                ms: Some(finalize_ms),
                aux: Some(format!("({pct:>5.1} %)", pct = enc_pct(finalize_ms))),
                desc: Some("passes + descriptors"),
                peak: Some(cycles_to_ms(w.finalize.max)),
            },
        );
        write_row(
            out,
            s,
            &Row {
                label: "└─ Submit stall",
                bold_label: false,
                ms: Some(stall_ms),
                aux: Some(format!("({pct:>5.1} %)", pct = enc_pct(stall_ms))),
                desc: Some("submit backpressure"),
                peak: Some(cycles_to_ms(w.submit_stall.max)),
            },
        );
    }

    /// The dedicated submit thread.
    ///
    /// Issues the `SubmitFrame` thunk (the unix command-walk → Metal calls,
    /// then present + commit) off the encoder thread. `Drawable wait` (GPU +
    /// compositor) is part of the execute and is broken out; `Encode+commit`
    /// is the submit thread's own CPU. Reported lagged ≤1 frame (folded back
    /// when a payload returns).
    fn write_submit_thread(
        &self,
        out: &mut String,
        submit_exec_ms: f64,
        encode_commit_ms: f64,
        dw_ms: f64,
    ) {
        let w = self.w;
        let s = &self.s;
        let _ = writeln!(out);
        write_row(
            out,
            s,
            &Row {
                label: "Submit thread",
                bold_label: true,
                ms: Some(submit_exec_ms),
                aux: None,
                desc: None,
                peak: Some(cycles_to_ms(w.submit_exec.max)),
            },
        );
        write_row(
            out,
            s,
            &Row {
                label: "├─ Encode+commit",
                bold_label: false,
                ms: Some(encode_commit_ms),
                aux: None,
                desc: Some("command-walk → Metal"),
                peak: Some(cycles_to_ms(w.encode_commit.max)),
            },
        );
        write_row(
            out,
            s,
            &Row {
                label: "└─ Drawable wait",
                bold_label: false,
                ms: Some(dw_ms),
                aux: None,
                desc: Some("nextDrawable GPU+comp"),
                peak: Some(cycles_to_ms(w.drawable_wait.max)),
            },
        );
    }

    fn write_frame_total(&self, out: &mut String, frame_total_ms: f64) {
        let w = self.w;
        let s = &self.s;
        let _ = writeln!(out);
        write_row(
            out,
            s,
            &Row {
                label: "Frame total",
                bold_label: true,
                ms: Some(frame_total_ms),
                aux: None,
                desc: None,
                peak: Some(cycles_to_ms(w.frame_total.max)),
            },
        );
        let _ = writeln!(
            out,
            "{d}submit_status={status:#x}   (API, Encoder, Submit run in parallel; frame_total ≥ max(api_cpu, enc_cpu, submit_cpu + gpu_wait)){r}",
            d = s.dim,
            r = s.reset,
            status = w.last_submit_status,
        );
    }

    fn write_resources_vbib(&self, out: &mut String) {
        use mtld3d_shared::tsc::u64_to_f64_exact;
        let w = self.w;
        let s = &self.s;
        let f = f64::from(self.frames);
        let vbib_ret_depth = u64_to_f64_exact(w.vbib_retention_depth.sum) / f;
        let vbib_ret_kb = u64_to_f64_exact(w.vbib_retained_bytes.sum) / f / 1024.0;
        let vbib_ret_peak_kb = u64_to_f64_exact(w.vbib_retained_bytes.max) / 1024.0;

        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Resources (VB/IB){r}  {d}— window totals; depth/retention averaged{r}",
            b = s.bold,
            r = s.reset,
            d = s.dim,
        );

        self.res_row(
            out,
            "rename",
            &format!(
                "VB={vb:<6} IB={ib}",
                vb = w.vb_rename.sum,
                ib = w.ib_rename.sum,
            ),
            Some(&format!(
                "peak/frame VB={pk_vb:<3} IB={pk_ib}",
                pk_vb = w.vb_rename.max,
                pk_ib = w.ib_rename.max,
            )),
            "API: PageBox alloc on contended Lock(DISCARD) or whole-buffer Lock",
        );
        // `rename = discards + preserve` invariant — the child rows
        // partition every rename by which preserve branch fired.
        self.res_row(
            out,
            "  discards",
            &format!(
                "VB={vb:<6} IB={ib}",
                vb = w.vb_discards.sum,
                ib = w.ib_discards.sum,
            ),
            None,
            "API: rename, no preserve (DISCARD or whole-buffer WRITEONLY)",
        );
        self.res_row(
            out,
            "  preserve",
            &format!("{c}", c = w.vbib_preserve_cpu.sum),
            None,
            "API: rename + sync memcpy (whole-buffer non-WRITEONLY contended — game may read back)",
        );
        self.res_row(
            out,
            "staging up",
            &format!("{up}", up = w.vbib_staging_uploads.sum),
            None,
            "encoder: Staged (non-DYNAMIC) dirty-range upload blits — separate-staging path; high here with rename≈0 is the goal",
        );
        self.res_row(
            out,
            "reorder",
            &format!("{r}", r = w.vbib_mid_pass_reorders.sum),
            None,
            "encoder: rename-at-overlap (upload hit a just-drawn region; rare)",
        );
        self.res_row(
            out,
            "destroys",
            &format!("{dx}", dx = w.buffer_destroys.sum),
            None,
            "encoder: MTLBuffer wrappers freed (VB/IB cache renames, Lock-rename intake, visibility-pool eviction)",
        );
        self.res_row(
            out,
            "alloc rcv",
            &format!(
                "drain={d} submit={s}",
                d = w.alloc_recovery_drain.sum,
                s = w.alloc_recovery_submit.sum,
            ),
            Some(&format!(
                "peak/frame submit={pk}",
                pk = w.alloc_recovery_submit.max,
            )),
            "API: VB/IB retention recovery — proactive cap + alloc-fail (drain=cheap, submit=GPU wait)",
        );
        let (avg_fmt, peak_fmt) = format_kb_pair(vbib_ret_kb, vbib_ret_peak_kb);
        self.res_row(
            out,
            "retention",
            &format!(
                "depth={vbib_ret_depth:>4.1}  {avg_fmt} avg"
            ),
            // One "peak" — the cell is already the peak column; the unit
            // on the byte value disambiguates it from the depth count.
            // Keeps the row short enough that the description stays in its
            // column even at 4-digit depths / GB byte peaks.
            Some(&format!(
                "peak depth={pk_depth:<3} {pk}",
                pk_depth = w.vbib_retention_depth.max,
                pk = peak_fmt,
            )),
            "encoder: shared PageBox queue (VB/IB renames + texture-blit padded staging + visibility pool)",
        );
    }

    fn write_resources_textures(&self, out: &mut String) {
        use mtld3d_shared::tsc::u64_to_f64_exact;
        let w = self.w;
        let s = &self.s;
        let f = f64::from(self.frames);
        let blit_ret = u64_to_f64_exact(w.pending_blit_retention_depth.sum) / f;
        let tex_ret_kb = u64_to_f64_exact(w.tex_staging_retained_bytes.sum) / f / 1024.0;
        let tex_ret_peak_kb = u64_to_f64_exact(w.tex_staging_retained_bytes.max) / 1024.0;

        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Resources (textures){r}  {d}— same layout as VB/IB; n/a rows omitted{r}",
            b = s.bold,
            r = s.reset,
            d = s.dim,
        );
        self.res_row(
            out,
            "rename",
            &format!("{t}", t = w.texture_renames.sum),
            None,
            "API: fresh staging Arc on contended LockRect",
        );
        // `rename = discards + preserve` — the two child rows partition
        // every texture rename by which preserve branch fired.
        self.res_row(
            out,
            "  discards",
            &format!("{d}", d = w.texture_discards.sum),
            None,
            "API: rename, no preserve (DISCARD or D3DUSAGE_DYNAMIC)",
        );
        self.res_row(
            out,
            "  preserve",
            &format!("{c}", c = w.texture_preserve_cpu.sum),
            Some(&format!(
                "peak/frame {pk_cpu}",
                pk_cpu = w.texture_preserve_cpu.max,
            )),
            "API: rename + sync memcpy (non-DISCARD non-DYNAMIC contended)",
        );
        // `uploads = raw + padded` — every Unlock takes one of the two
        // paths. `raw` is the cheap blit (cached `bytesNoCopy` wrapper
        // around the game's PageBox). `padded` is a blit too, but its
        // source had to be repacked into a transient widened-stride buffer
        // (extra alloc + memcpy + sync `CreateBuffer` thunk).
        let raw_blits = w
            .texture_blit_uploads
            .sum
            .saturating_sub(w.texture_blit_padded_uploads.sum);
        self.res_row(
            out,
            "uploads",
            &format!("{t}", t = w.texture_blit_uploads.sum),
            None,
            "encoder: total texture uploads (raw + padded)",
        );
        self.res_row(
            out,
            "  raw",
            &format!("{raw_blits}"),
            None,
            "encoder: blit; source = cached bytesNoCopy wrapper (cheap)",
        );
        self.res_row(
            out,
            "  padded",
            &format!("{p}", p = w.texture_blit_padded_uploads.sum),
            None,
            "encoder: blit; source repacked into transient buffer (alloc + memcpy + extra unix_call)",
        );
        self.res_row(
            out,
            "reorder",
            &format!("{r}", r = w.texture_gpu_renames.sum),
            None,
            "encoder: MTLTexture rename-at-overlap (upload hit a texture sampled earlier this frame)",
        );
        self.res_row(
            out,
            "destroys",
            &format!("{dx}", dx = w.texture_destroys.sum),
            None,
            "encoder: MTLTexture freed + texture-staging MTLBuffer wrappers freed (rename + padded + texture release)",
        );
        let (tex_avg_fmt, tex_peak_fmt) = format_kb_pair(tex_ret_kb, tex_ret_peak_kb);
        self.res_row(
            out,
            "retention",
            &format!("depth={blit_ret:>4.1}  {tex_avg_fmt} avg"),
            Some(&format!("peak {tex_peak_fmt}")),
            "encoder: blit source staging Arcs (separate from VB/IB retention; MTLTexture handles in destroys)",
        );
        // AddDirtyRect probe: does the game declare a changed sub-region we
        // could use to shrink the whole-mip preserve into a dirty-rect
        // snapshot upload? `partial` counts rects narrower than the mip;
        // coverage is the mean dirty-region area as a % of the mip.
        let adddirty_cover = if w.texture_add_dirty_calls.sum > 0 {
            u64_to_f64_exact(w.texture_add_dirty_area_bp.sum)
                / u64_to_f64_exact(w.texture_add_dirty_calls.sum)
                / 100.0
        } else {
            0.0
        };
        self.res_row(
            out,
            "dirtyrect",
            &format!("{c}", c = w.texture_add_dirty_calls.sum),
            Some(&format!(
                "{p} partial, {adddirty_cover:.0}% mip",
                p = w.texture_add_dirty_partial.sum,
            )),
            "API: AddDirtyRect calls; partial = sub-region narrower than mip (feeds dirty-rect snapshot upload decision)",
        );
    }

    /// Render a Resources row on the four-column grid.
    ///
    /// Label at col 0, window cell at `RES_WINDOW_COL`, optional peak/frame
    /// cell at `RES_PEAK_COL`, dim comment at `RES_COMMENT_COL`. Rows that
    /// have no per-frame peak (discards, wraps, destroys, plain counter
    /// rows) pass `peak = None` and the cell is left blank.
    fn res_row(
        &self,
        out: &mut String,
        label: &str,
        window: &str,
        peak: Option<&str>,
        comment: &str,
    ) {
        let s = &self.s;
        let mut cursor: usize = 0;
        out.push_str(label);
        cursor += label.chars().count();
        pad_spaces(out, &mut cursor, RES_WINDOW_COL);
        out.push_str(window);
        cursor += window.chars().count();
        if let Some(pk) = peak {
            pad_spaces(out, &mut cursor, RES_PEAK_COL);
            out.push_str(pk);
            cursor += pk.chars().count();
        }
        // Pad to the comment column, but guarantee a 2-space gap so a
        // value that reaches/overflows the column (e.g. a long
        // `retention … peak NNNN.N MB`) can't butt against the
        // description.
        let comment_col = RES_COMMENT_COL.max(cursor + 2);
        pad_spaces(out, &mut cursor, comment_col);
        out.push_str(s.dim);
        out.push_str(comment);
        out.push_str(s.reset);
        out.push('\n');
    }

    fn write_caches(&self, out: &mut String) {
        let s = &self.s;
        let c = self.caches;
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Caches{r} (live sizes at summary emit)",
            b = s.bold,
            r = s.reset,
        );
        let _ = writeln!(
            out,
            "  textures={t:<6} pipelines={p:<6} samplers={sa:<5} programs={pr}",
            t = c.textures,
            p = c.pipelines,
            sa = c.samplers,
            pr = c.programs,
        );
        let _ = writeln!(
            out,
            "  libs={v:<7}  depth_states={ds}",
            v = c.libs,
            ds = c.depth_states,
        );
    }

    fn write_commands_passes(&self, out: &mut String) {
        use mtld3d_shared::tsc::u64_to_f64_exact;
        let w = self.w;
        let s = &self.s;
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Commands / passes{r} (raw window totals)",
            b = s.bold,
            r = s.reset,
        );
        let _ = writeln!(
            out,
            "  passes={p:<9} commands={c:<11} draws={d}",
            p = w.passes.sum,
            c = w.commands.sum,
            d = w.draws.sum,
        );
        // Pipeline-resolve memo hit rate — share of get_or_create_pipeline
        // calls the single-entry snapshot memo served without a key build +
        // cache probe. Higher = more per-draw resolve work elided.
        let memo_hits = w.pipeline_memo_hits.sum;
        let memo_calls = w.pipeline_memo_calls.sum;
        let memo_pct = if memo_calls > 0 {
            u64_to_f64_exact(memo_hits) / u64_to_f64_exact(memo_calls) * 100.0
        } else {
            0.0
        };
        let _ = writeln!(
            out,
            "  pipeline memo  {memo_hits} / {memo_calls}  ({memo_pct:.1}%)  consecutive-draw resolve elided",
        );
    }

    /// Per-[`KeysGate`] redundant-skip rate.
    ///
    /// Of all live setter calls, how many left the FF VS/PS keys unchanged
    /// and so skipped the snapshot rebuild billed to the `keys` bucket. A
    /// scene-invariant read on how much per-draw key work the dirty-mark
    /// gating elides — the per-draw `keys` ns figure alone moves with scene
    /// composition, so this ratio is the direct measure of the optimisation
    /// firing.
    fn write_keys_gating(&self, out: &mut String) {
        use mtld3d_shared::tsc::u64_to_f64_exact;
        const LABELS: [&str; KeysGate::COUNT] = [
            "SetTexture",
            "SetRenderState",
            "SetTexStageState",
            "SetFvf",
            "SetVertexDecl",
            "SetVertexShader",
            "SetPixelShader",
            "SetVsConst",
            "SetPsConst",
        ];
        let w = self.w;
        let s = &self.s;
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Keys gating{r}  {d}— redundant snapshot dirty-marks elided (skips/calls); higher = more `keys` work avoided{r}",
            b = s.bold,
            r = s.reset,
            d = s.dim,
        );
        for (i, label) in LABELS.iter().enumerate() {
            let calls = w.keys_gate_calls_by[i].sum;
            let skips = w.keys_gate_skips_by[i].sum;
            let pct = if calls > 0 {
                u64_to_f64_exact(skips) / u64_to_f64_exact(calls) * 100.0
            } else {
                0.0
            };
            let _ = writeln!(out, "  {label:<16} {skips:>9} / {calls:<9} ({pct:5.1}%)");
        }
    }

    fn write_alloc_footprint(&self, out: &mut String) {
        use mtld3d_shared::tsc::u64_to_f64_exact;
        let w = self.w;
        let s = &self.s;
        let f = f64::from(self.frames);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{b}Per-frame allocator footprint{r}  scratch as small/oversized blocks; op_vec + cmd_vec each split into size + realloc",
            b = s.bold,
            r = s.reset,
        );
        // Scratch arena: per-frame live (cleared at begin_frame). Avg
        // block counts carry one decimal (sub-frame precision matters when
        // averaging across a 5 s window); peak counts are integers.
        let scratch_avg_small = u64_to_f64_exact(w.scratch_small_blocks.sum) / f;
        let scratch_avg_over = u64_to_f64_exact(w.scratch_oversized_blocks.sum) / f;
        let scratch_avg_kb = u64_to_f64_exact(w.scratch_bytes.sum) / f / 1024.0;
        let scratch_peak_kb = u64_to_f64_exact(w.scratch_bytes.max) / 1024.0;
        let (scratch_avg_fmt, scratch_peak_fmt) = format_kb_pair(scratch_avg_kb, scratch_peak_kb);
        self.res_row(
            out,
            "scratch",
            &format!("{scratch_avg_small:>4.1}/{scratch_avg_over:<3.1}  {scratch_avg_fmt} avg"),
            Some(&format!(
                "peak {ps}/{po:<3} peak {pk}",
                ps = w.scratch_small_blocks.max,
                po = w.scratch_oversized_blocks.max,
                pk = scratch_peak_fmt,
            )),
            "per-frame bump arena (VS/PS constants + UP vertices); cleared at begin_frame",
        );
        // op_vec — `FrameData.ops: Vec<Op>` shipped from the API
        // thread to the encoder via `sync_channel(1)`. The encoder
        // drains every Op in turn, translates D3D9 state to Metal,
        // and emits the `Vec<Command>` that becomes the cmd_vec
        // below.
        self.res_row(
            out,
            "op_vec",
            "",
            None,
            "API→encoder: Vec<Op> shipped via sync_channel(1); encoder drains and translates each Op",
        );
        let op_cap_avg_kb = u64_to_f64_exact(w.op_vec_capacity_bytes.sum) / f / 1024.0;
        let op_cap_peak_kb = u64_to_f64_exact(w.op_vec_capacity_bytes.max) / 1024.0;
        let (op_cap_avg_fmt, op_cap_peak_fmt) = format_kb_pair(op_cap_avg_kb, op_cap_peak_kb);
        self.res_row(
            out,
            "  size",
            &format!("{op_cap_avg_fmt} avg"),
            Some(&format!("peak {op_cap_peak_fmt}")),
            "high-water-reserved Vec<Op> capacity per frame (peak_ops_count)",
        );
        let op_realloc_avg_kb = u64_to_f64_exact(w.op_vec_realloc_bytes.sum) / f / 1024.0;
        let op_realloc_peak_kb = u64_to_f64_exact(w.op_vec_realloc_bytes.max) / 1024.0;
        let (op_realloc_avg_fmt, op_realloc_peak_fmt) =
            format_kb_pair(op_realloc_avg_kb, op_realloc_peak_kb);
        self.res_row(
            out,
            "  realloc",
            &format!("{op_realloc_avg_fmt} avg"),
            Some(&format!("peak {op_realloc_peak_fmt}")),
            "Vec<Op> doubling memcpy on push_op (target ≈ 0 with peak_ops_count reserve)",
        );
        // cmd_vec — `Pass::commands: Vec<Command>` built on the
        // encoder thread and shipped to unix via
        // `SubmitCommandBuffer`. Unix walks the Command[] and
        // dispatches each entry against a `MTL*Encoder` method.
        self.res_row(
            out,
            "cmd_vec",
            "",
            None,
            "encoder→unix: Vec<Command> shipped via SubmitCommandBuffer; unix dispatches each Command to a Metal encoder",
        );
        let cap_avg_kb = u64_to_f64_exact(w.cmd_vec_capacity_bytes.sum) / f / 1024.0;
        let cap_peak_kb = u64_to_f64_exact(w.cmd_vec_capacity_bytes.max) / 1024.0;
        let (cap_avg_fmt, cap_peak_fmt) = format_kb_pair(cap_avg_kb, cap_peak_kb);
        self.res_row(
            out,
            "  size",
            &format!("{cap_avg_fmt} avg"),
            Some(&format!("peak {cap_peak_fmt}")),
            "pool-resident Vec<Command> capacity across frames (recycled, not freed)",
        );
        let realloc_avg_kb = u64_to_f64_exact(w.cmd_vec_realloc_bytes.sum) / f / 1024.0;
        let realloc_peak_kb = u64_to_f64_exact(w.cmd_vec_realloc_bytes.max) / 1024.0;
        let (realloc_avg_fmt, realloc_peak_fmt) = format_kb_pair(realloc_avg_kb, realloc_peak_kb);
        self.res_row(
            out,
            "  realloc",
            &format!("{realloc_avg_fmt} avg"),
            Some(&format!("peak {realloc_peak_fmt}")),
            "Vec<Command> doubling memcpy on emit_command (target ≈ 0 with Pass::commands pool)",
        );
    }
}

#[cfg(all(test, perf_tracking))]
mod tests {
    use super::*;

    const fn sample(enc_cyc: u64, drawable_wait: u64) -> FrameSample {
        FrameSample {
            counters: FrameCounters::new(),
            timing: FrameTiming::new(),
            enc: EncoderFrameCounters {
                drawable_wait_cycles: drawable_wait,
                ..EncoderFrameCounters::new()
            },
            passes: 0,
            commands: 0,
            draws: 0,
            scratch_small_blocks: 0,
            scratch_oversized_blocks: 0,
            scratch_bytes: 0,
            cmd_vec_capacity_bytes: 0,
            vbib_retention_depth: 0,
            vbib_retained_bytes: 0,
            pending_blit_retention_depth: 0,
            tex_staging_retained_bytes: 0,
            cmd_vec_realloc_bytes: 0,
            outside_d3d9: 0,
            api_cyc: 0,
            api_work: 0,
            enc_cyc,
            submit_status: 0,
        }
    }

    /// Encoder-thread CPU (`enc_work`) is just `enc_cyc` (op + finalize).
    ///
    /// `drawable_wait` is submit-thread time tracked independently — it is
    /// NOT subtracted from the encoder bucket (the submit moved off-thread).
    /// Verify `PerfWindow::accumulate` keeps the two separate.
    #[test]
    fn perf_window_enc_work_identity() {
        let mut w = PerfWindow::new();
        w.accumulate(&sample(1000, 400));
        w.accumulate(&sample(1500, 800));
        assert_eq!(w.enc_cyc.sum, 2500);
        assert_eq!(w.drawable_wait.sum, 1200);
        assert_eq!(w.enc_work.sum, 2500);
        assert_eq!(w.enc_work.sum, w.enc_cyc.sum);
    }

    /// A large submit-thread `drawable_wait` never reduces the encoder-thread CPU bucket.
    ///
    /// They are independent threads now.
    #[test]
    fn perf_window_enc_work_independent_of_drawable_wait() {
        let mut w = PerfWindow::new();
        w.accumulate(&sample(100, 5000));
        assert_eq!(w.enc_work.sum, 100);
        assert_eq!(w.drawable_wait.sum, 5000);
    }

    /// `ApiPerfState::drain_into_payload` moves counters to the payload and zeroes the source.
    ///
    /// First-frame `frame_total` must be 0 (no predecessor TSC yet);
    /// subsequent drains report a delta.
    #[test]
    fn api_perf_drain_moves_and_resets() {
        let mut api = ApiPerfState::new();
        api.add_api_cycles(ApiCategory::VertexBuffer, 1234);
        api.bump_vb_rename();
        api.bump_vb_rename();
        api.bump_vbib_preserve_cpu();

        let mut p = FramePerfPayload::new();
        api.drain_into_payload(&mut p);

        assert_eq!(
            p.counters.api_cycles_by_category[ApiCategory::VertexBuffer as usize],
            1234
        );
        assert_eq!(
            p.counters.api_call_counts_by_category[ApiCategory::VertexBuffer as usize],
            1
        );
        assert_eq!(p.counters.vb_rename, 2);
        assert_eq!(p.counters.vbib_preserve_cpu, 1);
        assert_eq!(
            p.timing.frame_total_cycles, 0,
            "first frame has no predecessor"
        );

        // Source must be zeroed.
        assert_eq!(
            api.counters.api_cycles_by_category[ApiCategory::VertexBuffer as usize],
            0
        );
        assert_eq!(api.counters.vb_rename, 0);
        assert_eq!(api.counters.vbib_preserve_cpu, 0);

        // Second drain should report a real frame_total (tsc moved).
        let mut p2 = FramePerfPayload::new();
        api.drain_into_payload(&mut p2);
        assert!(
            p2.timing.frame_total_cycles > 0,
            "second drain must see a non-zero TSC delta"
        );
    }

    /// `EncoderPerfState::begin_frame` seeds per-frame encoder counters from the payload.
    ///
    /// Resets the encoder-side counters without clobbering running totals
    /// (`vbib_retained_bytes`).
    #[test]
    fn encoder_begin_frame_seeds_from_payload() {
        let mut enc = EncoderPerfState::new();
        enc.bump_vbib_retained_add(4096);
        enc.bump_buffer_destroy();
        enc.bump_texture_destroy();

        let mut payload = FramePerfPayload::new();
        payload.counters.vb_rename = 7;
        payload.set_present_block_cycles(42);
        payload.timing.frame_total_cycles = 1000;

        enc.begin_frame(&payload);

        assert_eq!(enc.counters.vb_rename, 7);
        assert_eq!(enc.timing.present_block_cycles, 42);
        assert_eq!(enc.timing.frame_total_cycles, 1000);
        assert_eq!(
            enc.enc.buffer_destroys, 0,
            "encoder-side counter reset per frame"
        );
        assert_eq!(enc.enc.texture_destroys, 0);
        assert_eq!(
            enc.vbib_retained_bytes, 4096,
            "running totals survive begin_frame"
        );
    }

    /// Encoder stall is the first signal.
    ///
    /// If the API thread is blocked on the backpressure channel for > 15% of
    /// the frame, classify by which encoder sub-bucket dominates.
    #[test]
    fn bottleneck_encoder_gpu_when_drawable_wait_dominates() {
        // frame=10ms, present_block=2ms (20%), gpu_wait=4ms, enc_cpu=1ms.
        let bn = Bottleneck::classify(10.0, 0.5, 0.5, 1.0, 4.0, 2.0);
        assert_eq!(bn, Bottleneck::EncoderGpu);
    }

    #[test]
    fn bottleneck_encoder_cpu_when_enc_cpu_dominates() {
        // frame=10ms, present_block=3ms (30%), enc_cpu=5ms, gpu_wait=0.5ms.
        let bn = Bottleneck::classify(10.0, 1.0, 1.0, 5.0, 0.5, 3.0);
        assert_eq!(bn, Bottleneck::EncoderCpu);
    }

    /// API thread keeps up with the encoder (present stall is small).
    ///
    /// The pacing side is the API thread, so the tie-break is D3D9 vs game
    /// code.
    #[test]
    fn bottleneck_api_d3d9_when_api_work_leads() {
        // present_block tiny, api_work > outside.
        let bn = Bottleneck::classify(10.0, 6.0, 3.0, 0.5, 0.3, 0.2);
        assert_eq!(bn, Bottleneck::ApiD3d9);
    }

    #[test]
    fn bottleneck_api_game_when_outside_leads() {
        let bn = Bottleneck::classify(10.0, 2.0, 7.0, 0.5, 0.3, 0.2);
        assert_eq!(bn, Bottleneck::ApiGame);
    }

    /// `Balanced` only when all four buckets are within 20 % of the per-bucket mean.
    ///
    /// No clear winner.
    #[test]
    fn bottleneck_balanced_when_buckets_even() {
        // frame=10ms, each bucket ≈ 2.5ms, present_block below encoder
        // threshold.
        let bn = Bottleneck::classify(10.0, 2.5, 2.5, 2.5, 2.5, 0.5);
        assert_eq!(bn, Bottleneck::Balanced);
    }

    /// ANSI on and ANSI off must emit the same visible layout.
    ///
    /// Stripping CSI escapes from the ANSI output must yield the exact
    /// plain output — if that fails, escapes are leaking into cells
    /// whose widths are format-padded.
    #[test]
    fn summary_ansi_off_matches_stripped_ansi_on() {
        let w = sample_window();
        let caches = sample_caches();
        let plain = Summary::render_with_ansi(&w, &caches, 5.01, false);
        let colored = Summary::render_with_ansi(&w, &caches, 5.01, true);
        let stripped = strip_ansi(&colored);
        assert_eq!(
            plain, stripped,
            "ANSI off output must equal the ANSI-on output with escapes stripped"
        );
    }

    /// Golden-string snapshot pinning the column grid.
    ///
    /// Every cell in the layout lands at a fixed column (`LABEL_W`,
    /// `AUX_COL`, `DESC_COL`, `PEAK_COL` for tree rows; `RES_WINDOW_COL` /
    /// `RES_PEAK_COL` / `RES_COMMENT_COL` for Resources rows). If a future
    /// edit drifts any cell, this fails with a diff that points at the exact
    /// row/col.
    #[test]
    fn summary_golden_layout() {
        let w = sample_window();
        let caches = sample_caches();
        let got = Summary::render_with_ansi(&w, &caches, 5.01, false);
        let want = concat!(
            "── perf  window=5.01s  frames=1  bottleneck=ENCODER (GPU) ──\n",
            "buckets: api_d3d9=2.80  api_outside=3.00  enc_work=1.50  submit_work=0.10  gpu_wait=6.00  (ms/frame, avg)\n",
            "\n",
            "API thread             10.00 ms                                             peak 10.00 ms\n",
            "├─ D3D9 calls           4.00 ms   ( 40.0 %)           243 calls             peak  4.00 ms\n",
            "│  ├─ Device            0.30 ms   (       123)                              peak  0.30 ms\n",
            "│  │  ├─ Frame          0.04 ms   (         3)                              peak  0.04 ms\n",
            "│  │  │  ├─ Send stall  3.20 ms                       encoder backpressure  peak  3.20 ms\n",
            "│  │  │  └─ other       0.00 ms                       non-blocking body     peak  0.00 ms\n",
            "│  │  ├─ Draws          0.12 ms   (1200 ns/draw)                            peak  0.12 ms\n",
            "│  │  │  ├─ snapshot    0.09 ms   ( 900 ns/draw)      read+stamp state      peak  0.09 ms\n",
            "│  │  │  │  ├─ stages   0.02 ms   ( 200 ns/draw)      tex/samp/TSS walk     peak  0.02 ms\n",
            "│  │  │  │  ├─ c_ff     0.02 ms   ( 200 ns/draw)      FF VS+PS const build  peak  0.02 ms\n",
            "│  │  │  │  ├─ c_pr     0.01 ms   ( 100 ns/draw)      programmable consts   peak  0.01 ms\n",
            "│  │  │  │  ├─ keys     0.02 ms   ( 200 ns/draw)      VDECL+RS+variant+srcs peak  0.02 ms\n",
            "│  │  │  │  ├─ bumps    0.01 ms   ( 100 ns/draw)      scratch+cache+wrapper peak  0.01 ms\n",
            "│  │  │  │  └─ resid    0.01 ms   ( 100 ns/draw)      uninstrumented noise  peak  0.01 ms\n",
            "│  │  │  └─ push_op     0.02 ms   ( 200 ns/draw)      inline Op::Draw push  peak  0.02 ms\n",
            "│  │  ├─ RenderState    0.06 ms   (2000 ns/call)                            peak  0.06 ms\n",
            "│  │  ├─ TexStageState  0.03 ms   (1667 ns/call)                            peak  0.03 ms\n",
            "│  │  ├─ SamplerState   0.02 ms   (1667 ns/call)                            peak  0.02 ms\n",
            "│  │  ├─ ShaderConst    0.02 ms   (2500 ns/call)                            peak  0.02 ms\n",
            "│  │  ├─ Bind           0.01 ms   (1667 ns/call)                            peak  0.01 ms\n",
            "│  │  │  ├─ Texture     0.00 ms   (         2)        Set/GetTexture        peak  0.00 ms\n",
            "│  │  │  ├─ Buffer      0.00 ms   (         1)        VB/IB/StreamFreq      peak  0.00 ms\n",
            "│  │  │  ├─ Shader      0.00 ms   (         1)        VDecl/VS/PS/FVF       peak  0.00 ms\n",
            "│  │  │  ├─ RtDs        0.00 ms   (         1)        RT + DepthStencil     peak  0.00 ms\n",
            "│  │  │  ├─ FfFixed     0.00 ms   (         1)        xform/material/light  peak  0.00 ms\n",
            "│  │  │  └─ VpScissor   0.00 ms   (         0)        viewport + scissor    peak  0.00 ms\n",
            "│  │  ├─ StateBlock     0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  │  └─ Misc           0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  ├─ VertexBuffer      0.20 ms   (        88)                              peak  0.20 ms\n",
            "│  ├─ IndexBuffer       0.08 ms   (        32)                              peak  0.08 ms\n",
            "│  ├─ Texture           0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  ├─ Surface           0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  ├─ Query             0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  │  └─ Wait for GPU   0.00 ms                       waitUntilCompleted    peak  0.00 ms\n",
            "│  ├─ StateBlock        0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  ├─ VertexDecl        0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  ├─ VertexShader      0.00 ms   (         0)                              peak  0.00 ms\n",
            "│  └─ PixelShader       0.00 ms   (         0)                              peak  0.00 ms\n",
            "└─ Outside d3d9         3.00 ms   ( 30.0 %)           game code             peak  3.00 ms\n",
            "\n",
            "Encoder thread          1.70 ms\n",
            "├─ Closures (op)        1.40 ms   ( 82.4 %)           D3D9→Metal translate  peak  1.40 ms\n",
            "│  ├─ resolve           0.30 ms   (3000 ns/draw)      tex + shader libs     peak  0.30 ms\n",
            "│  │  ├─ consts         0.15 ms   (1500 ns/draw)      VS/PS/FF const copy   peak  0.15 ms\n",
            "│  │  ├─ skip           0.09 ms   ( 900 ns/draw)      skip-set check        peak  0.09 ms\n",
            "│  │  ├─ lookup         0.03 ms   ( 300 ns/draw)      tex + lib probes      peak  0.03 ms\n",
            "│  │  └─ resid          0.03 ms   ( 300 ns/draw)      deref + glue          peak  0.03 ms\n",
            "│  ├─ pipeline          0.40 ms   (4000 ns/draw)      pipeline+depth+cull   peak  0.40 ms\n",
            "│  ├─ state             0.10 ms   (1000 ns/draw)      pass open + binds     peak  0.10 ms\n",
            "│  ├─ probe             0.20 ms   (2000 ns/draw)      decal/caster diag     peak  0.20 ms\n",
            "│  ├─ samplers          0.15 ms   (1500 ns/draw)      texture+sampler bind  peak  0.15 ms\n",
            "│  ├─ binds             0.20 ms   (2000 ns/draw)      consts + VB/IB + draw peak  0.20 ms\n",
            "│  │  ├─ cbind          0.12 ms   (1200 ns/draw)      const memcmp+bytes    peak  0.12 ms\n",
            "│  │  ├─ vbib           0.05 ms   ( 500 ns/draw)      VB/IB wrap + notify   peak  0.05 ms\n",
            "│  │  ├─ draw           0.02 ms   ( 200 ns/draw)      draw cmd emit         peak  0.02 ms\n",
            "│  │  └─ resid          0.01 ms   ( 100 ns/draw)      deref + glue          peak  0.01 ms\n",
            "│  ├─ tex_raw           0.01 ms   ( 100 ns/draw)      texture blit upload   peak  0.01 ms\n",
            "│  ├─ stage_up          0.01 ms   ( 100 ns/draw)      VB/IB staged upload   peak  0.01 ms\n",
            "│  ├─ const_rng         0.01 ms   ( 100 ns/draw)      VS/PS/FF const copy   peak  0.01 ms\n",
            "│  └─ resid             0.02 ms   ( 200 ns/draw)      other ops + noise     peak  0.02 ms\n",
            "├─ Finalize             0.10 ms   (  5.9 %)           passes + descriptors  peak  0.10 ms\n",
            "└─ Submit stall         0.20 ms   ( 11.8 %)           submit backpressure   peak  0.20 ms\n",
            "\n",
            "Submit thread           6.10 ms                                             peak  6.10 ms\n",
            "├─ Encode+commit        0.10 ms                       command-walk → Metal  peak  0.10 ms\n",
            "└─ Drawable wait        6.00 ms                       nextDrawable GPU+comp peak  6.00 ms\n",
            "\n",
            "Frame total            10.00 ms                                             peak 10.00 ms\n",
            "submit_status=0x0   (API, Encoder, Submit run in parallel; frame_total ≥ max(api_cpu, enc_cpu, submit_cpu + gpu_wait))\n",
            "\n",
            "Resources (VB/IB)  — window totals; depth/retention averaged\n",
            "rename      VB=12     IB=3          peak/frame VB=12  IB=3      API: PageBox alloc on contended Lock(DISCARD) or whole-buffer Lock\n",
            "  discards  VB=10     IB=3                                      API: rename, no preserve (DISCARD or whole-buffer WRITEONLY)\n",
            "  preserve  2                                                   API: rename + sync memcpy (whole-buffer non-WRITEONLY contended — game may read back)\n",
            "staging up  0                                                   encoder: Staged (non-DYNAMIC) dirty-range upload blits — separate-staging path; high here with rename≈0 is the goal\n",
            "reorder     0                                                   encoder: rename-at-overlap (upload hit a just-drawn region; rare)\n",
            "destroys    1                                                   encoder: MTLBuffer wrappers freed (VB/IB cache renames, Lock-rename intake, visibility-pool eviction)\n",
            "alloc rcv   drain=2 submit=1        peak/frame submit=1         API: VB/IB retention recovery — proactive cap + alloc-fail (drain=cheap, submit=GPU wait)\n",
            "retention   depth= 6.0  3.5 MB avg  peak depth=6   3.5 MB       encoder: shared PageBox queue (VB/IB renames + texture-blit padded staging + visibility pool)\n",
            "\n",
            "Resources (textures)  — same layout as VB/IB; n/a rows omitted\n",
            "rename      2                                                   API: fresh staging Arc on contended LockRect\n",
            "  discards  1                                                   API: rename, no preserve (DISCARD or D3DUSAGE_DYNAMIC)\n",
            "  preserve  1                       peak/frame 1                API: rename + sync memcpy (non-DISCARD non-DYNAMIC contended)\n",
            "uploads     2                                                   encoder: total texture uploads (raw + padded)\n",
            "  raw       2                                                   encoder: blit; source = cached bytesNoCopy wrapper (cheap)\n",
            "  padded    0                                                   encoder: blit; source repacked into transient buffer (alloc + memcpy + extra unix_call)\n",
            "reorder     1                                                   encoder: MTLTexture rename-at-overlap (upload hit a texture sampled earlier this frame)\n",
            "destroys    1                                                   encoder: MTLTexture freed + texture-staging MTLBuffer wrappers freed (rename + padded + texture release)\n",
            "retention   depth= 0.0  0 KB avg    peak 0 KB                   encoder: blit source staging Arcs (separate from VB/IB retention; MTLTexture handles in destroys)\n",
            "dirtyrect   4                       3 partial, 25% mip          API: AddDirtyRect calls; partial = sub-region narrower than mip (feeds dirty-rect snapshot upload decision)\n",
            "\n",
            "Caches (live sizes at summary emit)\n",
            "  textures=48     pipelines=12     samplers=6     programs=8\n",
            "  libs=8        depth_states=4\n",
            "\n",
            "Commands / passes (raw window totals)\n",
            "  passes=4         commands=140         draws=100\n",
            "  pipeline memo  97 / 100  (97.0%)  consecutive-draw resolve elided\n",
            "\n",
            "Keys gating  — redundant snapshot dirty-marks elided (skips/calls); higher = more `keys` work avoided\n",
            "  SetTexture               0 / 0         (  0.0%)\n",
            "  SetRenderState           0 / 0         (  0.0%)\n",
            "  SetTexStageState         0 / 0         (  0.0%)\n",
            "  SetFvf                   0 / 0         (  0.0%)\n",
            "  SetVertexDecl            0 / 0         (  0.0%)\n",
            "  SetVertexShader          0 / 0         (  0.0%)\n",
            "  SetPixelShader           0 / 0         (  0.0%)\n",
            "  SetVsConst               0 / 0         (  0.0%)\n",
            "  SetPsConst               0 / 0         (  0.0%)\n",
            "\n",
            "Per-frame allocator footprint  scratch as small/oversized blocks; op_vec + cmd_vec each split into size + realloc\n",
            "scratch     24.0/0.0  2.0 MB avg    peak 24/0   peak 2.0 MB     per-frame bump arena (VS/PS constants + UP vertices); cleared at begin_frame\n",
            "op_vec                                                          API→encoder: Vec<Op> shipped via sync_channel(1); encoder drains and translates each Op\n",
            "  size      72 KB avg               peak 72 KB                  high-water-reserved Vec<Op> capacity per frame (peak_ops_count)\n",
            "  realloc   32 KB avg               peak 32 KB                  Vec<Op> doubling memcpy on push_op (target ≈ 0 with peak_ops_count reserve)\n",
            "cmd_vec                                                         encoder→unix: Vec<Command> shipped via SubmitCommandBuffer; unix dispatches each Command to a Metal encoder\n",
            "  size      64 KB avg               peak 64 KB                  pool-resident Vec<Command> capacity across frames (recycled, not freed)\n",
            "  realloc   192 KB avg              peak 192 KB                 Vec<Command> doubling memcpy on emit_command (target ≈ 0 with Pass::commands pool)",
        );
        assert_eq!(got, want, "perf summary drifted — diff above");
    }

    /// Sanity snapshot of the rendered summary.
    ///
    /// The summary contains the section headers and the bottleneck label for
    /// a GPU-bound frame. Keeps the layout from silently losing a section in
    /// future refactors without locking the entire multi-line string.
    #[test]
    fn summary_contains_expected_sections() {
        let w = sample_window();
        let caches = sample_caches();
        let out = Summary::render_with_ansi(&w, &caches, 5.01, false);
        for expected in [
            "── perf  window=5.01s",
            "bottleneck=ENCODER (GPU)",
            "API thread",
            "├─ D3D9 calls",
            "├─ Send stall",
            "└─ Outside d3d9",
            "Encoder thread",
            "├─ Closures (op)",
            "├─ Finalize",
            "└─ Submit stall",
            "Submit thread",
            "├─ Encode+commit",
            "└─ Drawable wait",
            "Frame total",
            "submit_status=0x0",
            "Resources (VB/IB)",
            "Resources (textures)",
            "Caches",
            "Commands / passes",
            "Keys gating",
        ] {
            assert!(
                out.contains(expected),
                "summary missing {expected:?}:\n{out}"
            );
        }
    }

    fn sample_window() -> PerfWindow {
        // Construct a window populated with a single synthetic frame
        // whose shape classifies as ENCODER (GPU): drawable_wait
        // dominates, present_block is large.
        let mut w = PerfWindow::new();
        let mut cats = [0u64; ApiCategory::COUNT];
        let mut calls = [0u32; ApiCategory::COUNT];
        cats[ApiCategory::Device as usize] = 300_000;
        cats[ApiCategory::VertexBuffer as usize] = 200_000;
        cats[ApiCategory::IndexBuffer as usize] = 80_000;
        calls[ApiCategory::Device as usize] = 123;
        calls[ApiCategory::VertexBuffer as usize] = 88;
        calls[ApiCategory::IndexBuffer as usize] = 32;
        // Decompose Device into sub-buckets; must sum to cats[Device]
        // = 300_000 to mirror the production invariant. Calls sum to
        // calls[Device] = 123.
        let mut dsub = [0u64; DeviceSubCategory::COUNT];
        let mut dcalls = [0u32; DeviceSubCategory::COUNT];
        // Values picked so `cycles / tsc_hz * 1e3` rounds cleanly at
        // `{:.2}` even with the small jitter in runtime calibration —
        // multiples of 10_000 cycles only. Sum = 300_000 to match
        // `cats[Device]`.
        dsub[DeviceSubCategory::Frame as usize] = 40_000;
        dsub[DeviceSubCategory::Draws as usize] = 120_000;
        dsub[DeviceSubCategory::RenderState as usize] = 60_000;
        dsub[DeviceSubCategory::TexStageState as usize] = 30_000;
        dsub[DeviceSubCategory::SamplerState as usize] = 20_000;
        dsub[DeviceSubCategory::ShaderConst as usize] = 20_000;
        dsub[DeviceSubCategory::Bind as usize] = 10_000;
        dsub[DeviceSubCategory::StateBlock as usize] = 0;
        dsub[DeviceSubCategory::Misc as usize] = 0;
        dcalls[DeviceSubCategory::Frame as usize] = 3;
        dcalls[DeviceSubCategory::Draws as usize] = 100;
        dcalls[DeviceSubCategory::RenderState as usize] = 30;
        dcalls[DeviceSubCategory::TexStageState as usize] = 18;
        dcalls[DeviceSubCategory::SamplerState as usize] = 12;
        dcalls[DeviceSubCategory::ShaderConst as usize] = 8;
        dcalls[DeviceSubCategory::Bind as usize] = 6;
        dcalls[DeviceSubCategory::StateBlock as usize] = 0;
        dcalls[DeviceSubCategory::Misc as usize] = 0;
        // Decompose the Bind device-sub (10_000 cyc, 6 calls) into
        // BindSubCategory rows. Sums must match the parent exactly —
        // every BindSubCategory site uses `bind_timer`, no escape.
        // Values are multiples of 1_000 cycles to round cleanly at
        // `{:.2}` under runtime tsc_hz calibration.
        let mut bsub = [0u64; BindSubCategory::COUNT];
        let mut bcalls = [0u32; BindSubCategory::COUNT];
        bsub[BindSubCategory::Texture as usize] = 4_000;
        bsub[BindSubCategory::Buffer as usize] = 2_000;
        bsub[BindSubCategory::Shader as usize] = 2_000;
        bsub[BindSubCategory::RtDs as usize] = 1_000;
        bsub[BindSubCategory::FfFixed as usize] = 1_000;
        bsub[BindSubCategory::ViewScissor as usize] = 0;
        bcalls[BindSubCategory::Texture as usize] = 2;
        bcalls[BindSubCategory::Buffer as usize] = 1;
        bcalls[BindSubCategory::Shader as usize] = 1;
        bcalls[BindSubCategory::RtDs as usize] = 1;
        bcalls[BindSubCategory::FfFixed as usize] = 1;
        bcalls[BindSubCategory::ViewScissor as usize] = 0;
        let s = FrameSample {
            counters: FrameCounters {
                api_cycles_by_category: cats,
                api_call_counts_by_category: calls,
                vb_rename: 12,
                ib_rename: 3,
                vb_discards: 10,
                ib_discards: 3,
                // Two whole-buffer non-WRITEONLY contended Locks took the
                // CPU-memcpy preserve path. Surfaces in the `preserve` row.
                // `rename = discards + preserve_cpu` holds (12 = 10 + 2 for VB).
                vbib_preserve_cpu: 2,
                // Two cheap-tier recoveries (drain) and one heavy
                // (submit) — exercises the row formatting on both halves.
                alloc_recovery_drain: 2,
                alloc_recovery_submit: 1,
                // 2 renames: 1 was DISCARD/WRITEONLY (no preserve),
                // 1 needed CPU memcpy preserve. Invariant
                // `rename = discards + preserve_cpu` holds.
                texture_renames: 2,
                texture_discards: 1,
                texture_preserve_cpu: 1,
                // AddDirtyRect probe fixture: 4 calls, 3 with a usable
                // sub-region; area sum 10000 bp ⇒ mean coverage 25% of the mip.
                texture_add_dirty_calls: 4,
                texture_add_dirty_partial: 3,
                texture_add_dirty_area_bp: 10_000,
                query_wait_cycles: 0,
                device_sub_cycles: dsub,
                device_sub_calls: dcalls,
                bind_sub_cycles: bsub,
                bind_sub_calls: bcalls,
                keys_gate_calls: [0; KeysGate::COUNT],
                keys_gate_skips: [0; KeysGate::COUNT],
                // snapshot dominates the Draws bucket; split inside it
                // is stages 20 + c_ff 20 + c_pr 10 + keys 20 + bumps 10
                // + leftover 10 = 90. push_op trails. Every component is
                // a multiple of 10_000 so `cycles / tsc_hz * 1e3` rounds
                // identically across calibration jitter (the underlying
                // tsc_hz wobble of ±few ppm only flips rounding for
                // values like 5_000 or 25_000 that fall on the {:.2}
                // boundary).
                draw_snapshot_cycles: 90_000,
                draw_snapshot_stages_cycles: 20_000,
                draw_snapshot_c_ff_cycles: 20_000,
                draw_snapshot_c_pr_cycles: 10_000,
                draw_snapshot_keys_cycles: 20_000,
                draw_snapshot_bumps_cycles: 10_000,
                draw_push_op_cycles: 20_000,
            },
            timing: FrameTiming {
                present_block_cycles: 3_200_000,
                frame_total_cycles: 10_000_000,
                // peak_ops_count × size_of::<Op>() — peak ~1000 ops at
                // ~72 B/Op rounds to ~72 KB. Pick 72 KB exactly so the
                // golden assertion pins the row and `format_kb_pair`
                // renders cleanly.
                op_vec_capacity_bytes: 72 * 1024,
                // Steady-state target is 0; pick a small non-zero value
                // so the realloc row's number column is exercised by the
                // golden assertion.
                op_vec_realloc_bytes: 32 * 1024,
            },
            enc: EncoderFrameCounters {
                buffer_destroys: 1,
                texture_destroys: 1,
                vbib_staging_uploads: 0,
                vbib_mid_pass_reorders: 0,
                texture_blit_uploads: 2,
                texture_blit_padded_uploads: 0,
                texture_gpu_renames: 1,
                op_cycles: 1_400_000,
                // Decompose op_cyc 1.40M into the nine phases (six draw phases sum
                // 1.35M + three non-draw phases sum 0.03M = 1.38M) so the golden
                // pins each "Closures (op)" sub-row; resid = 0.02M. Multiples of
                // 10_000 cyc round cleanly at {:.2} under tsc jitter.
                op_sub_cycles: [
                    300_000, 400_000, 100_000, 200_000, 150_000, 200_000, 10_000, 10_000, 10_000,
                ],
                // resolve(300k) split 150/90/30 → resid 30k; binds(200k) split
                // 120/50/20 → resid 10k. Multiples of 10k round cleanly at {:.2}.
                op_sub_detail: [150_000, 90_000, 30_000, 120_000, 50_000, 20_000],
                // 97 of 100 pipeline resolves served from the memo → 97.0%.
                pipeline_memo_hits: 97,
                pipeline_memo_calls: 100,
                submit_cycles: 300_000,
                // Submit thread: total execute 6.1M = encode+commit 0.1M +
                // drawable wait 6.0M.
                drawable_wait_cycles: 6_000_000,
                submit_exec_cycles: 6_100_000,
                // 0.2M backpressure stall → Finalize 0.10M, stall 0.20M.
                submit_stall_cycles: 200_000,
            },
            passes: 4,
            commands: 140,
            draws: 100,
            scratch_small_blocks: 24,
            scratch_oversized_blocks: 0,
            scratch_bytes: 2 * 1024 * 1024,
            // ~140 commands at 32 B each ≈ 4.4 KB live; round to one
            // 64 KB pool entry to keep the resident-footprint cell
            // non-zero in the golden assertion.
            cmd_vec_capacity_bytes: 64 * 1024,
            // Retention in MB range exercises the format_kb_pair MB
            // branch and the longest peak cell the Resources grid
            // ever renders — the golden assertion below pins the
            // column gap so future edits don't re-regress the butting
            // bug where "peak N.N MBlive queue..." ran together.
            vbib_retention_depth: 6,
            vbib_retained_bytes: 3_670_016,
            pending_blit_retention_depth: 0,
            tex_staging_retained_bytes: 0,
            cmd_vec_realloc_bytes: 196_608,
            // Encoder thread = op (Closures) + finalize. The unix
            // command-walk + present moved to the submit thread.
            outside_d3d9: 3_000_000,
            api_cyc: 4_000_000,
            api_work: 2_800_000,
            enc_cyc: 1_700_000,
            submit_status: 0,
        };
        w.accumulate(&s);
        w
    }

    const fn sample_caches() -> CacheSizes {
        CacheSizes {
            textures: 48,
            pipelines: 12,
            samplers: 6,
            programs: 8,
            libs: 8,
            depth_states: 4,
            scratch_small_blocks: 24,
            scratch_oversized_blocks: 0,
            scratch_bytes: 2 * 1024 * 1024,
            cmd_vec_capacity_bytes: 64 * 1024,
            pending_blit_retention_depth: 0,
            pending_resource_retention_depth: 1,
        }
    }

    /// Strip ANSI CSI sequences (ESC `[` … final-byte) from a string.
    ///
    /// Implemented locally so the test suite doesn't grow a
    /// `strip-ansi-escapes` dep.
    fn strip_ansi(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                i += 2;
                while i < bytes.len() {
                    let c = bytes[i];
                    i += 1;
                    if (0x40..=0x7e).contains(&c) {
                        break;
                    }
                }
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8(out).expect("ANSI-strip produced invalid UTF-8")
    }

    // Exclusive-time accounting: a nested timer's interval must land in
    // its own bucket only, never double-counted into the delegating
    // parent's bucket too (the Surface→Texture LockRect case). The
    // invariant is `Σ self_time == outermost elapsed`.
    #[test]
    fn exclusive_exit_single_timer_books_full_elapsed() {
        // No children, outermost (depth 0 after): self == elapsed,
        // accumulator resets to 0.
        let (self_time, restored) = exclusive_exit(200, 0, 0, 0);
        assert_eq!(self_time, 200);
        assert_eq!(restored, 0);
    }

    #[test]
    fn exclusive_exit_nested_pair_no_double_count() {
        // Outer (Surface) elapsed 200 wraps inner (Texture) elapsed 50.
        // Inner drops first: depth back to 1 (parent still live), no
        // children of its own → self 50, hands its 50 up to the parent
        // accumulator (saved 0 + 50).
        let (inner_self, inner_restored) = exclusive_exit(50, 0, 0, 1);
        assert_eq!(inner_self, 50);
        assert_eq!(inner_restored, 50);
        // Outer drops: its children == the 50 the inner handed up;
        // self 150; outermost → reset to 0.
        let (outer_self, outer_restored) = exclusive_exit(200, inner_restored, 0, 0);
        assert_eq!(outer_self, 150);
        assert_eq!(outer_restored, 0);
        // No double-count: the two self-times partition the outer span.
        assert_eq!(inner_self + outer_self, 200);
    }

    #[test]
    fn exclusive_exit_two_siblings_under_parent() {
        // Parent (elapsed 100) contains two sequential children (30, 40).
        // c1 drops, hands 30 up (parent acc 0+30).
        let (c1_self, acc) = exclusive_exit(30, 0, 0, 1);
        // c2 starts with the parent acc saved (30), drops, hands 40 up.
        let (c2_self, acc) = exclusive_exit(40, 0, acc, 1);
        // Parent drops: children == 70; self == 30; outermost resets.
        let (p_self, p_restored) = exclusive_exit(100, acc, 0, 0);
        assert_eq!((c1_self, c2_self, p_self), (30, 40, 30));
        assert_eq!(p_restored, 0);
        assert_eq!(c1_self + c2_self + p_self, 100);
    }

    #[test]
    fn exclusive_exit_saturates_when_children_exceed_elapsed() {
        // TSC noise can make a nested delta momentarily exceed the
        // parent's measured span; self_time clamps to 0, never wraps.
        let (self_time, _) = exclusive_exit(10, 25, 0, 1);
        assert_eq!(self_time, 0);
    }
}
