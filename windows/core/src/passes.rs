//! Pure-Rust render-pass state machine used by the PE-side `FrameEncoder`.
//!
//! Holds the `passes: Vec<Pass>` plus the bookkeeping for pass breaks on
//! `SetRenderTarget`, `SetDepthStencilSurface`, and mid-frame `Clear`.

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasher,
};

use log::{Level, log_enabled, trace};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, CommandType, MetalHandle,
    mtl::{CullMode, PixelFormat, VisibilityResultMode},
    mtl_handle::{MTLRenderPipelineStateKind, MTLTextureKind},
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::dirty_range::DirtyRange;

/// Compile-time gate for Rule A (first-use `DontCare`).
///
/// Flip to `false` for a single-line hotfix if a temporal-blending game
/// surfaces that reads prior-frame contents on first use of frame N.
const ENABLE_FIRST_USE_DONTCARE: bool = true;

/// Compile-time gate for Rule B (last-use depth/stencil `DontCare`).
///
/// Flip to `false` if a game using `INTZ`-style late-frame depth-readback
/// surfaces. D3D9 spec already says depth contents are undefined across
/// `Present`, so this is conformant for any game that respects the spec.
const ENABLE_LAST_USE_DEPTH_DONTCARE: bool = true;

/// Compile-time gate for Rule C (color `Store=DontCare`).
///
/// Applies when the next pass that touches the same color rt begins with
/// a full-attachment clear. D3D9's `Clear()` always covers the full
/// attachment regardless of viewport or scissor, so the prior Store is
/// provably redundant whenever the next pass's
/// `color_load == ColorLoad::Clear { .. }`. Mirrors Rule B but keyed on
/// color and predicated on the next-pass clear instead of
/// last-occurrence. Flip to `false` if a game starts a pass with `Clear`
/// but expects to read the underlying rt contents in some way mtld3d
/// doesn't model (no such case is known).
const ENABLE_NEXT_CLEAR_COLOR_DONTCARE: bool = true;

/// Compile-time gate for Rule F (cull clear-only passes with dead Stores).
///
/// A pass qualifies when its every-attachment-Store ends up `DontCare`
/// after Rules B/C/D run. Such a pass has zero observable effect: no
/// draws, no leading blits, and nothing reaches VRAM. Runs at the very
/// end of the pass-finalisation pipeline so it sees the post-rule Store
/// actions. Cheap correctness guard: passes with leading blits stay
/// (the blits are real work scheduled before the encoder).
const ENABLE_CULL_DEAD_CLEAR_PASSES: bool = true;

/// Compile-time gate for Rule D (last-use non-backbuffer color `Store=DontCare`).
///
/// Symmetric to Rule B for the color attachment, with the backbuffer
/// explicitly exempted because Present consumes its content from VRAM
/// after submit and we have no in-pass visibility into that consumer.
/// Sampler-aware via `seen_sampled_textures`. Eliminates the multi-MB
/// writeback of a cascade color attachment — that color is a
/// placeholder for depth-only caster draws and is never sampled. Flip
/// to `false` if a game samples a non-backbuffer color rt
/// across the Present boundary in a way mtld3d's single-frame
/// `seen_sampled_textures` can't capture.
const ENABLE_LAST_USE_COLOR_DONTCARE: bool = true;

/// Compile-time gate for Rule G — strip the color attachment from a clear-only pass.
///
/// Fires when the pass's color side is provably wasted
/// (`color_store == DontCare` after Rules C/D, no draws, no leading
/// blits). The pass becomes a *depth-only* Metal render pass with no
/// `colorAttachments[0]` binding. Eliminates Apple's "Unused Texture"
/// Insight on the cascade-color placeholder that per-cascade
/// depth-clear sub-passes would otherwise attach. Requires unix-side
/// `encode_pass` to handle `color_texture == 0` with
/// `command_count > 0`.
const ENABLE_STRIP_DEAD_COLOR_IN_CLEAR_ONLY: bool = true;

/// Compile-time gate for Rule H — strip the color attachment from a pass-with-draws.
///
/// Fires when every draw in the pass runs with `D3DRS_COLORWRITEENABLE == 0`.
/// Symmetric to Rule G but for passes that contain draws (Rule G only
/// covers clear-only passes). Predicate: `color_writes_observed == false`,
/// `color_texture != 0`, `depth_texture != 0` (Metal needs ≥1 attachment),
/// and at least one draw command (otherwise Rule G already handled it).
/// The rule also rewrites the pass's `SetRenderPipelineState` commands to
/// bind a matching no-color pipeline variant — the caller passes a
/// `with_color_handle → no_color_handle` side-map populated at draw time
/// from `FrameEncoder::no_color_pipeline_alt`. Eliminates Apple's "Unused
/// Texture" Insight on the cascade-color texture across cascade caster
/// passes. Flip to `false` if a game surfaces relying on color writes
/// against a masked-everywhere attachment (no such case is known — D3D9
/// spec is unambiguous).
const ENABLE_NO_COLOR_PASS_FOR_DRAWS: bool = true;

/// Sub-target for one-line-per-event pass-break / pass-open trace probes.
///
/// Gated by `RUST_LOG=mtld3d::d3d9::passes=trace`; the helpers below short-circuit
/// to a single `log_enabled!` call when the target isn't active.
const TRACE_TARGET: &str = "mtld3d::d3d9::passes";

/// Depth-path probes.
///
/// `RUST_LOG=mtld3d::d3d9::depth=trace` opts in; this is the same
/// sub-target the encoder + device modules use for
/// `depth: pass attach=…`, `depth: surface bind tex=…`, and
/// `depth: slot N …`. Re-used here so the per-`Clear` decision (Quad
/// vs. Folded-amend vs. Folded-pending vs. visibility-fallback) shows
/// up next to those probes — pre-fix vs. post-fix the count of each
/// branch firing tells you immediately which Clear shape the game is
/// using and whether the clear-quad path is reached.
const DEPTH_TRACE_TARGET: &str = "mtld3d::d3d9::depth";

/// Matches `STAGE_COUNT = 16` (PS3.0 allows s0–s15) used by `StageBindingsPtr` in the d3d9 crate.
///
/// The 4th CSM cascade shadow texture on the receiver path lands at
/// slot 8.
pub const LAST_BOUND_MAX_STAGES: usize = 16;

/// Cap on `command_vec_pool` size.
///
/// A 5-pass frame is the typical shape; 16 absorbs every realistic
/// pass-count spike without parking unused capacity forever.
const MAX_CMD_VEC_POOL: usize = 16;

/// Cascade-summary probe target.
///
/// Used both to gate the per-frame summary log line at submit time AND
/// to skip the per-draw / per-bind counter increments in
/// `note_caster_draw` and `emit_command` when the probe is off. Without
/// the gate at the increment sites the probe would have non-zero cost at
/// default `RUST_LOG` (one `HashMap` entry per caster draw + per sample
/// bind), violating the zero-cost-when-off discipline that all mtld3d
/// diag probes follow.
const CASCADE_PROBE_TARGET: &str = "mtld3d::d3d9::cascade";

/// Pack a `(x, y, w, h)` viewport rect into a single u64.
///
/// Keeps the per-(texture, viewport) `log_once_trace_by!` keys for the
/// clear-quad probes deduping at the right grain.
const fn pack_viewport_key(vp: (u32, u32, u32, u32)) -> u64 {
    let (x, y, w, h) = vp;
    ((x as u64) << 48) ^ ((y as u64) << 32) ^ ((w as u64) << 16) ^ (h as u64)
}

/// How the next render-pass should load its color attachment.
///
/// `Load` preserves whatever the previous pass wrote; `Clear` replaces
/// it with the stored RGBA bits (f32 bits each); `DontCare` leaves
/// tile memory uninitialized at pass start (used on first-frame-use
/// of an rt whose prior contents are undefined or about to be fully
/// overwritten).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorLoad {
    Load,
    Clear { r: u32, g: u32, b: u32, a: u32 },
    DontCare,
}

/// How the next render-pass should load its depth attachment.
///
/// `Load` carries the previous pass's depth buffer forward; `Clear`
/// resets it to `value` (stored as f32 bits); `DontCare` leaves tile
/// memory uninitialized at pass start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthLoad {
    Load,
    Clear { value: u32 },
    DontCare,
}

/// How the render-pass should store its attachment at pass end.
///
/// `Store` writes tile memory back to device memory; `DontCare`
/// discards it. Used on the last pass with a given depth attachment
/// in a frame (depth never crosses Present, Rule B) and on color
/// attachments whose next consumer this frame begins with a full-
/// attachment `Clear` (Rule C).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreAction {
    Store,
    DontCare,
}

/// Result of `PassState::clear_depth` / `clear_color`.
///
/// `Folded` is the fast path: pass either had no work yet
/// (Clear amended into the load action in place) or was closed (Clear
/// stashed in `pending_*_clear` for the next pass-open). The caller
/// has nothing more to do.
///
/// `EmitQuad` means the pass already had draws when the Clear
/// arrived. Ending the pass and starting a fresh one with
/// `loadAction = Clear` is wrong on Metal — Metal's load-Clear is
/// full-attachment, ignoring viewport, and would wipe the prior
/// draws (e.g. each tile of a shared shadow tile atlas wipes the
/// previously rendered tiles). Instead the caller — the
/// `FrameEncoder` layer that owns the per-format clear-quad pipeline
/// cache — emits a fullscreen-triangle draw inside the current
/// encoder, scissored to the D3D9 viewport, that writes the constant
/// clear value as depth (and color when `has_color`). The pass
/// stays open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthClearOutcome {
    Folded,
    EmitQuad {
        value: u32,
        viewport: (u32, u32, u32, u32),
        has_color: bool,
        color_format: PixelFormat,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorClearOutcome {
    Folded,
    EmitQuad {
        rgba: (u32, u32, u32, u32),
        viewport: (u32, u32, u32, u32),
        color_format: PixelFormat,
    },
}

/// One Metal render pass.
///
/// Each pass maps to a single `MTLRenderCommandEncoder` on the unix side.
/// Attachments are frozen at pass open; further changes (`SetRenderTarget`,
/// mid-frame Clear, depth change) end the current pass and open a new one.
pub struct Pass {
    color_texture: MetalHandle<MTLTextureKind>,
    color_size: (u32, u32),
    color_format: PixelFormat,
    color_load: ColorLoad,
    /// Defaults to `Store` at pass open.
    ///
    /// `PassState::finalize_store_actions` flips to `DontCare` at submit
    /// time when the very next pass this frame touching the same color
    /// texture begins with a full-attachment `Clear` (Rule C) — the prior
    /// contents are provably overwritten. The last pass per rt in the
    /// frame is naturally exempt (no next pass), so backbuffer Present
    /// and persistent rt contents survive.
    color_store: StoreAction,
    depth_texture: MetalHandle<MTLTextureKind>,
    depth_load: DepthLoad,
    /// Defaults to `Store`.
    ///
    /// Flipped to `DontCare` by `finalize_store_actions` on the *last*
    /// pass with each depth texture in the frame — depth/stencil contents
    /// are undefined across `Present` per D3D9 spec, so the final flush
    /// back to device memory is wasted. The unix side mirrors this to the
    /// stencil attachment when the texture is `Depth32Float_Stencil8`.
    depth_store: StoreAction,
    viewport: (u32, u32, u32, u32),
    commands: Vec<Command>,
    /// Blits replayed inside an `MTLBlitCommandEncoder` *before* this pass's render encoder begins.
    ///
    /// Drained from `PassState::pending_leading_blits` at pass open. Used
    /// by `StretchRect` so a texture-to-texture copy that happens between
    /// two D3D9 draws lands in correct order with both passes (the
    /// global `frame_blit_commands` runs at frame start and would mis-
    /// order a mid-frame `StretchRect` against the source pass's draws).
    leading_blits: Vec<BlitCommand>,
    /// Latched `true` by a `SetVisibilityResultMode(Counting, …)` command in this pass.
    ///
    /// Emitted into the pass via `PassState::emit_command`. The submit
    /// path uses this to decide whether to attach the frame's visibility
    /// result buffer to this pass's render-pass descriptor. Passes with
    /// only `Disabled` (trailing END with no further active queries) or
    /// no visibility command at all keep the attachment cleared, which
    /// avoids Metal tracking the buffer in the pass's resource residency
    /// set and keeps the `MTL_DEBUG_LAYER` validator from retaining
    /// per-pass tracking state for it.
    has_counting_visibility: bool,
    /// `true` when the depth attachment is a sampleable shadow map.
    ///
    /// Created via `CreateTexture(D24X8, USAGE_DEPTHSTENCIL)`. Rule B
    /// short-circuits on this flag: any sampleable depth keeps `Store`
    /// regardless of whether it's been sampled this session yet —
    /// avoids the bootstrap-frame gap where a cascade sampled only
    /// every Nth frame loses content on the intervening frames.
    depth_is_sampleable: bool,
    /// Latched `true` as soon as any draw arrives at the pass with `D3DRS_COLORWRITEENABLE != 0`.
    ///
    /// Default `false` at pass-open. When the pass closes with this still
    /// `false` AND at least one real (non-clear-quad) draw was emitted,
    /// Rule H (`strip_color_from_no_color_draw_passes`) strips the color
    /// attachment and rewrites the pass's `SetRenderPipelineState`
    /// commands to bind the matching no-color pipeline variant —
    /// eliminating Apple's "Unused Texture" warning on cascade caster
    /// passes where every draw runs with color writes masked off but
    /// the bound pipeline still declares a color output.
    color_writes_observed: bool,
    /// `[start, end)` command-index ranges of color clear-quad blocks emitted into this pass.
    ///
    /// Recorded by `PassState::open_color_clear_quad_block` /
    /// `close_color_clear_quad_block` from the encoder's
    /// `emit_clear_quad_color_inner`.
    ///
    /// Rule H ignores commands inside these ranges when deciding
    /// whether a "real" color-writing draw is present, and removes the
    /// ranges entirely when it strips the color attachment — the
    /// color clear-quad pipeline declares a color output and would
    /// fail Metal's pipeline-vs-RP format validation against a
    /// stripped (depth-only) descriptor, and its writes are dead work
    /// anyway once the attachment is gone.
    color_clear_quad_ranges: Vec<(usize, usize)>,
}

impl Pass {
    #[must_use]
    pub const fn color_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.color_texture
    }
    #[must_use]
    pub const fn color_size(&self) -> (u32, u32) {
        self.color_size
    }
    /// Metal pixel format of the pass's color attachment.
    ///
    /// Included in `PipelineKey` so cache hits distinguish pipelines by
    /// rt format — a pipeline built for `BGRA8Unorm` would be rejected by
    /// Metal if bound against an rt with a different format.
    #[must_use]
    pub const fn color_format(&self) -> PixelFormat {
        self.color_format
    }
    #[must_use]
    pub const fn depth_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.depth_texture
    }
    #[must_use]
    pub const fn color_load(&self) -> ColorLoad {
        self.color_load
    }
    #[must_use]
    pub const fn color_store(&self) -> StoreAction {
        self.color_store
    }
    #[must_use]
    pub const fn depth_load(&self) -> DepthLoad {
        self.depth_load
    }
    #[must_use]
    pub const fn depth_store(&self) -> StoreAction {
        self.depth_store
    }
    /// `(origin_x, origin_y, width, height)` in pixels.
    ///
    /// `x, y` are non-zero when the game sub-rects the render target via
    /// `SetViewport` — essential for XYZRHW-relative UI draws.
    #[must_use]
    pub const fn viewport(&self) -> (u32, u32, u32, u32) {
        self.viewport
    }
    #[must_use]
    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    #[must_use]
    pub fn leading_blits(&self) -> &[BlitCommand] {
        &self.leading_blits
    }

    #[must_use]
    pub const fn has_counting_visibility(&self) -> bool {
        self.has_counting_visibility
    }

    #[must_use]
    pub const fn color_writes_observed(&self) -> bool {
        self.color_writes_observed
    }

    #[must_use]
    pub fn color_clear_quad_ranges(&self) -> &[(usize, usize)] {
        &self.color_clear_quad_ranges
    }
}

/// Per-pass record of the byte range each VB/IB was read from by draws.
///
/// Scoped to the currently-open render pass, keyed by `BufferId` raw.
///
/// Load-bearing for the rename-at-overlap upload model: when an inline
/// `Staged` upload overwrites a region a draw already read *this frame*,
/// applying it to the live device buffer would corrupt that earlier draw
/// (they share one buffer), so the encoder renames instead. `overlaps`
/// drives that decision; the `reorder` perf counter rides on the same
/// signal.
///
/// Tracking is per-FRAME, not per-pass: the upload blits emit into the
/// frame-head leading phase (before *every* pass), so an upload that
/// overwrites a region read by a draw in an earlier, already-closed pass
/// would corrupt it just the same — the tracker must remember draws across
/// pass boundaries. Cleared at frame start (`reset_frame`) and per-buffer
/// on a rename (the fresh buffer has no draws yet).
#[derive(Default)]
struct DrawnRangeTracker {
    // FxHash, not SipHash: `note` runs a `.entry` per draw (twice per
    // indexed draw), the same per-draw probe frequency as the encoder's
    // resource caches.
    ranges: FxHashMap<u64, DirtyRange>,
}

impl DrawnRangeTracker {
    fn new() -> Self {
        Self::default()
    }

    /// Conjoin `[offset, offset + size)` into the range drawn from buffer `id` this pass.
    ///
    /// A `size` of 0 runs to the end of the buffer.
    fn note(&mut self, id: u64, offset: u32, size: u32, logical_len: u32) {
        self.ranges
            .entry(id)
            .or_default()
            .conjoin(offset, size, logical_len);
    }

    /// True if buffer `id` was drawn this pass from a range overlapping the half-open `[off, end)`.
    fn overlaps(&self, id: u64, off: u32, end: u32) -> bool {
        self.ranges.get(&id).is_some_and(|r| r.overlaps(off, end))
    }

    /// Forget buffer `id`'s drawn range — called after a rename.
    ///
    /// The fresh device buffer has been read by no draw yet.
    fn clear_buffer(&mut self, id: u64) {
        self.ranges.remove(&id);
    }

    fn clear(&mut self) {
        self.ranges.clear();
    }
}

bitflags::bitflags! {
    /// Descriptor bits for the attachments currently bound on `PassState`.
    ///
    /// Packed into a u8 instead of three separate `bool` fields; read via the
    /// `current_*` accessors and folded onto each `Pass`/pipeline snapshot at
    /// draw time.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct CurrentAttachmentFlags: u8 {
        /// Whether the bound colour RT's D3D format has a real alpha channel.
        ///
        /// Tracked alongside `current_color_format` because the Metal pixel
        /// format alone can't tell X8R8G8B8 (no alpha) from A8R8G8B8 (both are
        /// `Bgra8Unorm`). Read at draw time into the pipeline snapshot's
        /// `COLOR_HAS_ALPHA` bit so destination-alpha blend factors clamp on
        /// alpha-less targets. Updated in lockstep with the format by
        /// `set_color_rt_has_alpha` (called from the encoder's colour-RT bind);
        /// `reset_frame` seeds it set for the alpha-bearing backbuffer.
        const COLOR_HAS_ALPHA = 1 << 0;
        /// Set when the currently bound depth attachment came from the sampleable-depth path.
        ///
        /// That path is `CreateTexture(D24X8, USAGE_DEPTHSTENCIL)`
        /// — i.e. a sampleable shadow map. Clear for standalone
        /// `CreateDepthStencilSurface` targets that can never be sampled.
        /// Folded onto the `Pass` at `ensure_pass_open` so Rule B
        /// (last-use depth `DontCare`) can short-circuit on it without
        /// relying on the per-session `seen_sampled_textures` set (which
        /// has a bootstrap-frame gap for cascades sampled rarely).
        const DEPTH_SAMPLEABLE = 1 << 1;
        /// Set when the bound depth attachment's D3D format carries a stencil plane.
        ///
        /// D24S8 / D24FS8 / D15S1 / D24X4S4 all map to the combined Metal
        /// `Depth32Float_Stencil8` texture. The clear-quad pipelines must declare
        /// the matching depth/stencil attachment formats or Metal's
        /// pipeline-vs-render-pass validation rejects them (undefined behaviour /
        /// heap corruption with the layer off).
        const DEPTH_HAS_STENCIL = 1 << 2;
    }
}

/// Pass-management state machine.
///
/// Owned by the encoder thread's `FrameEncoder`; every frame begins with
/// `reset_frame` and ends with `end_current_pass` followed by draining
/// `passes()` into the submit thunk.
pub struct PassState {
    passes: Vec<Pass>,
    current_pass_closed: bool,

    current_color_texture: MetalHandle<MTLTextureKind>,
    current_color_size: (u32, u32),
    current_color_format: PixelFormat,
    current_depth_texture: MetalHandle<MTLTextureKind>,
    /// Descriptor bits for the currently bound colour/depth attachments.
    ///
    /// `COLOR_HAS_ALPHA` / `DEPTH_SAMPLEABLE` / `DEPTH_HAS_STENCIL`. See
    /// `CurrentAttachmentFlags` for the per-bit semantics.
    current_attachments: CurrentAttachmentFlags,

    pending_color_clear: Option<(u32, u32, u32, u32)>,
    pending_depth_clear: Option<u32>,

    /// Sticky across frames — games call `SetViewport` once and expect it to persist.
    ///
    /// When width/height are zero (uninitialized) we fall back to
    /// `(0, 0, color_size.0, color_size.1)` at pass-begin.
    viewport_x: u32,
    viewport_y: u32,
    viewport_width: u32,
    viewport_height: u32,
    /// D3D9's per-viewport depth range.
    ///
    /// Default `(0.0, 1.0)` matches Metal's default and the D3DVIEWPORT9
    /// uninitialized state; games that partition depth (sky / world /
    /// weapon) override these.
    viewport_min_z: f32,
    viewport_max_z: f32,
    /// The viewport last *emitted* onto the open pass's encoder.
    ///
    /// Held as `(x, y, w, h, min_z_bits, max_z_bits)` (z-range kept as
    /// raw bits for exact equality). Seeded by `ensure_pass_open` to the
    /// first command it pushes; `set_viewport` skips a mid-pass re-emit
    /// that matches it. A fresh `MTLRenderCommandEncoder` carries no
    /// viewport state, so this resets to `None` at each pass open via the
    /// seed.
    last_emitted_viewport: Option<(u32, u32, u32, u32, u32, u32)>,

    /// Blits queued by `StretchRect` between two passes.
    ///
    /// Drained into the next pass's `leading_blits` at
    /// `ensure_pass_open`. If the frame ends with no follow-up pass,
    /// `submit` synthesises a trailing blit-only pass so the queued blits
    /// still run.
    pending_leading_blits: Vec<BlitCommand>,

    /// Color-attachment textures that have already been used as an rt in this frame.
    ///
    /// Inserted at `ensure_pass_open`. First-use opens the door to
    /// `ColorLoad::DontCare` (Rule A) — subsequent uses default to `Load`
    /// so accumulated draws survive across pass breaks. Reset each frame
    /// in `reset_frame`. Capacity hint matches a typical frame shape
    /// (backbuffer + a few CSM ping-pong RTs).
    seen_color_rts: HashSet<MetalHandle<MTLTextureKind>>,
    /// Depth-attachment textures that have already been used as a depth rt in this frame.
    ///
    /// Same semantics as `seen_color_rts`.
    seen_depth_rts: HashSet<MetalHandle<MTLTextureKind>>,
    /// The swap-chain backbuffer texture for this frame, captured in `reset_frame`.
    ///
    /// Rule D (last-use color `Store=DontCare`) exempts this handle so
    /// Present can still read the pixels from VRAM after submit.
    backbuffer_texture: MetalHandle<MTLTextureKind>,
    /// Texture handles ever bound as a fragment sampler input in any pass this frame.
    ///
    /// Populated in `emit_command` from `SetFragmentTexture` commands.
    /// Consumed by Rule A (`ensure_pass_open`) and
    /// `finalize_load_actions` to skip / revert `LoadAction::DontCare` on
    /// attachments whose content a sampler reads elsewhere in the frame,
    /// and by `finalize_store_actions` to skip `StoreAction::DontCare` on
    /// the same. Closes a hole in the original load/store optimiser that
    /// discarded CSM cascade content between the caster pass that wrote
    /// it and the scene pass that sampled it. Reset each frame in
    /// `reset_frame`.
    seen_sampled_textures: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// Texture handles bound as a fragment sampler input so far THIS frame, in op-stream order.
    ///
    /// Populated at the `emit_command` funnel beside
    /// `seen_sampled_textures`; unlike that session-wide set, this one is
    /// cleared every `reset_frame`.
    ///
    /// Load-bearing for texture rename-at-overlap: upload blits land in
    /// the frame-head leading phase (before *every* pass), so an upload
    /// into a texture a draw already sampled this frame would rewrite
    /// what that earlier draw reads — the per-draw D3D9 texture state
    /// would collapse to frame-final. The encoder consults
    /// [`Self::texture_sampled_this_frame`] at upload time and renames
    /// the `MTLTexture` instead (fresh handle for later draws, earlier
    /// draws keep the old one). Handle-keyed on purpose: the fresh
    /// handle has been sampled by no earlier draw, so a rename needs no
    /// explicit clear here.
    frame_sampled_textures: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// Session-wide set of texture handles that were EVER bound as a sampleable depth attachment.
    ///
    /// The bind runs via
    /// `set_depth_stencil_attachment(_, is_sampleable=true)`. The
    /// `mtld3d::d3d9::cascade=trace` end-of-frame summary uses this to
    /// classify fragment-sample binds: a `SetFragmentTexture` of a
    /// handle in this set is a cascade-depth read, and is counted
    /// into `frame_cascade_samples`. Persistent across frames
    /// (textures are stable resource identities).
    seen_sampleable_depth_textures: HashSet<MetalHandle<MTLTextureKind>>,
    /// Per-frame counter: how many caster draws targeted each cascade depth handle.
    ///
    /// Counts the draws made this frame. Incremented in `note_caster_draw`,
    /// drained + cleared by `take_cascade_frame_summary`.
    frame_caster_writes: HashMap<MetalHandle<MTLTextureKind>, u32>,
    /// Per-frame counter: how many `SetFragmentTexture` binds of a known cascade depth handle.
    ///
    /// Counts the binds emitted this frame. Incremented in `emit_command`
    /// when the bound texture is in `seen_sampleable_depth_textures`.
    /// Drained by `take_cascade_frame_summary`.
    frame_cascade_samples: HashMap<MetalHandle<MTLTextureKind>, u32>,
    /// Monotonic per-frame counter for the cascade-summary log line.
    ///
    /// Distinct from `submit_seq` (encoder-thread): incremented in
    /// `reset_frame`.
    frame_seq: u64,
    /// Running per-frame total of bytes Metal will memcpy as a result of `Vec` doublings.
    ///
    /// The doublings are `Vec<Command>::push` growing `Pass::commands`.
    /// Incremented in `emit_command` when `len == capacity` before push (the
    /// doubling is about to fire); the increment is the *old* capacity in
    /// bytes, which is exactly what `realloc` has to copy from the old buffer
    /// to the new one. Drained by `take_cmd_vec_realloc_bytes` once per frame
    /// and rolled into the perf summary as `cmd_realloc`. Reset in
    /// `reset_frame` as a safety net for the case where the drain is somehow
    /// skipped.
    cmd_vec_realloc_bytes: u64,
    /// Free-list of `Vec<Command>`s recycled across frames.
    ///
    /// `reset_frame` drains each retired `Pass`'s `commands` into the pool
    /// with capacity preserved; the next frame's `ensure_pass_open` pops one
    /// instead of freshly `Vec::with_capacity(64)`. Once warmed, the pool's
    /// Vecs converge on the steady-state high-water capacity per pass and no
    /// further `Vec::push` doublings fire. Capped at `MAX_CMD_VEC_POOL` (~16
    /// entries) so a one-off heavy frame can't grow the pool unboundedly.
    command_vec_pool: Vec<Vec<Command>>,

    /// Per-pass VB/IB read-range tracker driving rename-at-overlap.
    ///
    /// Also feeds the `reorder` perf counter.
    drawn_ranges: DrawnRangeTracker,

    /// Debug-build mirror of what was actually emitted onto the current Metal encoder.
    ///
    /// Diffed against the encoder's `LastBoundCache` before every draw to
    /// catch cache↔encoder desyncs. See [`DebugBoundShadow`].
    #[cfg(debug_assertions)]
    debug_emitted: DebugBoundShadow,
}

impl PassState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            passes: Vec::with_capacity(4),
            current_pass_closed: true,
            current_color_texture: MetalHandle::NULL,
            current_color_size: (0, 0),
            // Placeholder; `reset_frame` always overwrites this before any
            // pass opens. Chose the dominant backbuffer format rather than
            // adding an `Unknown` variant to `PixelFormat` that would pollute
            // every exhaustive match downstream.
            current_color_format: PixelFormat::Bgra8Unorm,
            current_depth_texture: MetalHandle::NULL,
            // Placeholder; `reset_frame` reseeds these for the backbuffer, and
            // every `SetRenderTarget` bind overwrites `COLOR_HAS_ALPHA` via
            // `set_color_rt_has_alpha`. The dominant backbuffer is alpha-bearing
            // (`COLOR_HAS_ALPHA` set), non-sampleable, no stencil.
            current_attachments: CurrentAttachmentFlags::COLOR_HAS_ALPHA,
            pending_color_clear: None,
            pending_depth_clear: None,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: 0,
            viewport_height: 0,
            viewport_min_z: 0.0,
            viewport_max_z: 1.0,
            last_emitted_viewport: None,
            pending_leading_blits: Vec::new(),
            seen_color_rts: HashSet::with_capacity(4),
            seen_depth_rts: HashSet::with_capacity(2),
            backbuffer_texture: MetalHandle::NULL,
            seen_sampled_textures: FxHashSet::with_capacity_and_hasher(8, FxBuildHasher),
            frame_sampled_textures: FxHashSet::with_capacity_and_hasher(64, FxBuildHasher),
            seen_sampleable_depth_textures: HashSet::with_capacity(8),
            frame_caster_writes: HashMap::with_capacity(8),
            frame_cascade_samples: HashMap::with_capacity(8),
            frame_seq: 0,
            cmd_vec_realloc_bytes: 0,
            command_vec_pool: Vec::with_capacity(MAX_CMD_VEC_POOL),
            drawn_ranges: DrawnRangeTracker::new(),
            #[cfg(debug_assertions)]
            debug_emitted: DebugBoundShadow::default(),
        }
    }

    /// Reset per-frame state.
    ///
    /// Seeds the default attachments (frame's backbuffer + depth) and clears
    /// any leftover pending clears. Does not touch the sticky viewport — that
    /// survives across frames.
    pub fn reset_frame(
        &mut self,
        backbuffer: MetalHandle<MTLTextureKind>,
        backbuffer_size: (u32, u32),
        backbuffer_format: PixelFormat,
        depth_texture: MetalHandle<MTLTextureKind>,
        depth_has_stencil: bool,
    ) {
        // Recycle each retired pass's `commands` Vec back into the pool
        // (capacity preserved, length zeroed). Once warm, the pool's
        // Vecs carry the steady-state high-water capacity and the next
        // frame's `ensure_pass_open` reuses them — eliminating the
        // `Vec::with_capacity(64)` → many doublings cycle that the
        // `cmd_realloc` perf row measures. Cap so a one-off frame with
        // many passes (rare) can't park unused capacity forever.
        for pass in self.passes.drain(..) {
            if self.command_vec_pool.len() >= MAX_CMD_VEC_POOL {
                break;
            }
            let mut cmds = pass.commands;
            cmds.clear();
            self.command_vec_pool.push(cmds);
        }
        // Belt-and-braces in case the break-on-cap fired mid-drain.
        self.passes.clear();
        self.current_pass_closed = true;
        // No pass open → no viewport emitted yet; the next pass's open
        // reseeds this. (`set_viewport` only reads it inside an open pass.)
        self.last_emitted_viewport = None;
        self.current_color_texture = backbuffer;
        self.current_color_size = backbuffer_size;
        self.current_color_format = backbuffer_format;
        // The backbuffer is an alpha-bearing (`Bgra8Unorm` / A8R8G8B8) target,
        // so destination-alpha blend factors resolve unclamped — byte-identical
        // to the pre-`COLOR_HAS_ALPHA` behaviour. A sub-frame `SetRenderTarget`
        // to an X8 surface overrides this via `set_color_rt_has_alpha`.
        self.current_attachments
            .insert(CurrentAttachmentFlags::COLOR_HAS_ALPHA);
        self.current_depth_texture = depth_texture;
        // The frame's default depth target is the standalone backbuffer
        // depth surface from `CreateDepthStencilSurface` — not
        // sampleable. Sub-frame `set_depth_stencil_attachment` calls
        // override this flag when WoW binds a sampleable shadow map.
        self.current_attachments
            .remove(CurrentAttachmentFlags::DEPTH_SAMPLEABLE);
        self.current_attachments
            .set(CurrentAttachmentFlags::DEPTH_HAS_STENCIL, depth_has_stencil);
        self.backbuffer_texture = backbuffer;
        self.pending_color_clear = None;
        self.pending_depth_clear = None;
        self.pending_leading_blits.clear();
        self.seen_color_rts.clear();
        self.seen_depth_rts.clear();
        self.frame_caster_writes.clear();
        self.frame_cascade_samples.clear();
        self.frame_sampled_textures.clear();
        self.drawn_ranges.clear();
        self.frame_seq = self.frame_seq.wrapping_add(1);
        // Safety net: `take_cmd_vec_realloc_bytes` should already have
        // drained this at end-of-frame. Zero again so a missed drain
        // doesn't carry stale bytes into the next frame's accounting.
        self.cmd_vec_realloc_bytes = 0;
        // Do NOT clear `seen_sampled_textures`. Per-frame reset would
        // break double-buffered cascade textures (shadow cascades):
        // caster writes to cascade-A in frame N, receiver samples
        // cascade-A in frame N+1. Rule B at frame-N finalize would
        // see cascade-A "not sampled this frame" and flip
        // `depth_store=DontCare`, letting Metal discard the depth
        // content at pass-end — wiping the cascade content the
        // receiver needs next frame. Rule A's first-use `DontCare`
        // check on the load side has the same hazard. Tracking
        // "ever sampled" across frames keeps both rules
        // conservative for cross-frame-referenced textures at the
        // cost of a Store/Load on first-frame-use, which is the
        // correct trade.
        //
        // Memory cost: bounded by the number of distinct texture
        // handles ever used as a sampler input over the session
        // (~100s for WoW). Cleared only at device-reset (when the
        // game might destroy and reissue textures with the same
        // handles).
    }

    #[must_use]
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }

    /// Take the frame's finished passes, leaving an empty (capacity-retained) vec behind.
    ///
    /// The caller owns the passes for the duration of the submit stage — the
    /// unix side reads each pass's `commands` via raw pointer — then returns
    /// them through [`Self::recycle_passes`] so the command vecs re-enter the
    /// pool. This is the seam that lets the finished passes outlive this
    /// `PassState` while the next frame starts building; the synchronous
    /// recycling `reset_frame` does inline still covers the path where passes
    /// were never taken out (it then sees an empty vec).
    pub fn take_finished_passes(&mut self) -> Vec<Pass> {
        core::mem::take(&mut self.passes)
    }

    /// Drain finished passes' `commands` vecs back into the recycle pool.
    ///
    /// Capacity preserved, length zeroed, capped at `MAX_CMD_VEC_POOL`. The
    /// counterpart to [`Self::take_finished_passes`]: once the submit stage is
    /// done reading a taken pass list, this returns its command vecs so the
    /// next frame's `ensure_pass_open` reuses them instead of freshly
    /// allocating. `drain(..)` always empties `passes` (retaining its
    /// capacity); the cap only bounds how many vecs the pool parks.
    pub fn recycle_passes(&mut self, passes: &mut Vec<Pass>) {
        for pass in passes.drain(..) {
            if self.command_vec_pool.len() >= MAX_CMD_VEC_POOL {
                continue;
            }
            let mut cmds = pass.commands;
            cmds.clear();
            self.command_vec_pool.push(cmds);
        }
    }

    /// Index of the currently-open pass within the frame (zero-based).
    ///
    /// Callers downstream of `emit_command` are guaranteed a pass is
    /// open, so the value equals `passes.len() - 1`. Used by the
    /// `mtld3d::d3d9::decal` trace probe so a single trace line tells
    /// whether two draws share an `MTLRenderCommandEncoder`.
    /// `saturating_sub` keeps the value sane if called before the
    /// first pass opens (returns 0).
    #[must_use]
    pub const fn current_pass_index(&self) -> usize {
        self.passes.len().saturating_sub(1)
    }

    #[must_use]
    pub const fn current_pass_closed(&self) -> bool {
        self.current_pass_closed
    }

    /// Record that a draw this frame read `[offset, offset + size)` from VB/IB `id`.
    ///
    /// A `size` of 0 means to the end of the buffer. Feeds rename-at-overlap
    /// via [`Self::drawn_range_overlaps`].
    pub fn note_draw_range(&mut self, id: u64, offset: u32, size: u32, logical_len: u32) {
        self.drawn_ranges.note(id, offset, size, logical_len);
    }

    /// True if buffer `id` was drawn this frame from a range overlapping half-open `[off, end)`.
    ///
    /// I.e. a staging upload to that range would land (frame-head) out of
    /// order relative to a draw that already read it, so the device buffer
    /// must be renamed.
    #[must_use]
    pub fn drawn_range_overlaps(&self, id: u64, off: u32, end: u32) -> bool {
        self.drawn_ranges.overlaps(id, off, end)
    }

    /// Forget buffer `id`'s drawn range.
    ///
    /// Called after the encoder renames its device buffer, since the fresh
    /// buffer has no draws yet.
    pub fn clear_drawn_range(&mut self, id: u64) {
        self.drawn_ranges.clear_buffer(id);
    }

    #[must_use]
    pub const fn current_color_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.current_color_texture
    }

    #[must_use]
    pub const fn current_depth_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.current_depth_texture
    }

    /// `true` when the bound depth attachment is a combined depth+stencil Metal format.
    ///
    /// The combined format is `Depth32Float_Stencil8`. The clear-quad
    /// pipelines key on this so their declared depth/stencil attachment
    /// formats match the pass.
    #[must_use]
    pub const fn current_depth_has_stencil(&self) -> bool {
        self.current_attachments
            .contains(CurrentAttachmentFlags::DEPTH_HAS_STENCIL)
    }

    #[must_use]
    pub const fn current_depth_is_sampleable(&self) -> bool {
        self.current_attachments
            .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE)
    }

    /// `true` when `depth_tex` was bound as a sampleable shadow map at any point this session.
    ///
    /// Built for diagnostic probes that want to identify cascade-depth writes
    /// regardless of the current `is_sampleable` flag, which can be
    /// incorrectly `false` after a `GetDepthStencilSurface` save/restore cycle
    /// on a cascade surface (see `note_caster_draw` doc).
    #[must_use]
    pub fn is_depth_handle_sampleable(&self, depth_tex: MetalHandle<MTLTextureKind>) -> bool {
        self.seen_sampleable_depth_textures.contains(&depth_tex)
    }

    #[must_use]
    pub const fn current_color_format(&self) -> PixelFormat {
        self.current_color_format
    }

    /// Set whether the currently bound colour RT's D3D format has a real alpha channel.
    ///
    /// Called by the encoder's colour-RT bind in lockstep with
    /// `set_color_render_target` so the two never desync — the Metal pixel
    /// format alone can't distinguish X8R8G8B8 (no alpha) from A8R8G8B8.
    pub fn set_color_rt_has_alpha(&mut self, has_alpha: bool) {
        self.current_attachments
            .set(CurrentAttachmentFlags::COLOR_HAS_ALPHA, has_alpha);
    }

    /// Whether the currently bound colour RT's D3D format has a real alpha channel.
    ///
    /// Read at draw time into the pipeline snapshot's `COLOR_HAS_ALPHA` bit.
    #[must_use]
    pub const fn current_color_rt_has_alpha(&self) -> bool {
        self.current_attachments
            .contains(CurrentAttachmentFlags::COLOR_HAS_ALPHA)
    }

    /// Record that a colour texture is read back this session.
    ///
    /// Read back by something the in-frame load/store analysis can't see — a
    /// `GetRenderTargetData` blit runs *after* the frame's
    /// `finalize_store_actions`, so without this hint Rule D (last-use
    /// non-backbuffer colour `Store=DontCare`) would discard the rendered
    /// content and the readback would observe a cleared/garbage surface.
    /// Treated exactly like a sampled texture, which already exempts the
    /// colour store (Rules C/D).
    pub fn note_color_read_back(&mut self, handle: MetalHandle<MTLTextureKind>) {
        if !handle.is_null() {
            self.seen_sampled_textures.insert(handle);
        }
    }

    /// True when `handle` was bound as a fragment sampler input by an earlier draw this frame.
    ///
    /// Drives texture rename-at-overlap: an upload into such a texture must go
    /// to a fresh `MTLTexture` (the upload blit executes frame-head, before
    /// the draw that already sampled the old content). Stream-exact by
    /// construction — a texture uploaded before its first sample this frame is
    /// absent and correctly skips the rename.
    #[must_use]
    pub fn texture_sampled_this_frame(&self, handle: MetalHandle<MTLTextureKind>) -> bool {
        self.frame_sampled_textures.contains(&handle)
    }

    #[must_use]
    pub const fn current_color_size(&self) -> (u32, u32) {
        self.current_color_size
    }

    #[must_use]
    pub const fn pending_color_clear(&self) -> Option<(u32, u32, u32, u32)> {
        self.pending_color_clear
    }

    #[must_use]
    pub const fn pending_depth_clear(&self) -> Option<u32> {
        self.pending_depth_clear
    }

    #[must_use]
    pub const fn viewport(&self) -> (u32, u32, u32, u32) {
        (
            self.viewport_x,
            self.viewport_y,
            self.viewport_width,
            self.viewport_height,
        )
    }

    /// The viewport's depth-range near/far (`D3DVIEWPORT9.MinZ`/`MaxZ`).
    ///
    /// Exposed so a save/restore around a one-off pass (the scaling
    /// `StretchRect` render path) can preserve the game's depth range
    /// rather than clobber it to the default `[0, 1]`.
    #[must_use]
    pub const fn viewport_depth_range(&self) -> (f32, f32) {
        (self.viewport_min_z, self.viewport_max_z)
    }

    /// Viewport with the `ensure_pass_open` fallback.
    ///
    /// When width or height is zero (game never called `SetViewport`),
    /// substitute the current rt's size at origin. Used by pass-open viewport
    /// emission and by `emit_scissor` so both see the same rect. Exposed so
    /// the encoder's clear-quad emit path can resolve the same scissor as the
    /// rest of the pass machine.
    #[must_use]
    pub const fn effective_viewport(&self) -> (u32, u32, u32, u32) {
        if self.viewport_width != 0 && self.viewport_height != 0 {
            (
                self.viewport_x,
                self.viewport_y,
                self.viewport_width,
                self.viewport_height,
            )
        } else {
            (0, 0, self.current_color_size.0, self.current_color_size.1)
        }
    }

    /// True when the current viewport covers (or exceeds) the whole bound color attachment.
    ///
    /// I.e. a `Clear(NULL rects)` need not be viewport-bounded and can fold to
    /// a fast full-attachment `loadAction = Clear`. False only for a strict
    /// sub-region viewport (origin off (0,0) or smaller than the attachment),
    /// where the clear must be scissored to the viewport. With no color
    /// attachment bound there is nothing to bound, so fold.
    #[must_use]
    pub const fn viewport_covers_color_attachment(&self) -> bool {
        if self.current_color_texture.is_null() {
            return true;
        }
        let (vpx, vpy, vpw, vph) = self.effective_viewport();
        vpx == 0 && vpy == 0 && vpw >= self.current_color_size.0 && vph >= self.current_color_size.1
    }

    /// Tag the current pass with "color writes happened" iff `mask != 0`.
    ///
    /// Called by the PE encoder right before emitting the per-draw
    /// `SetRenderPipelineState` so the pass closes with an accurate
    /// "every draw had `COLORWRITEENABLE == 0`" signal for Rule H. Opens
    /// a pass first if none is live (mirrors the `emit_command` contract).
    pub fn note_draw_color_write_mask(&mut self, mask: u32) {
        self.ensure_pass_open();
        if mask != 0
            && let Some(pass) = self.passes.last_mut()
        {
            pass.color_writes_observed = true;
        }
    }

    /// Note a draw targeting the given depth handle.
    ///
    /// Increments the per-frame caster-writes counter iff the handle was ever
    /// bound as a sampleable shadow map this session — i.e. it's a known
    /// cascade texture. Filtering on the persistent
    /// `seen_sampleable_depth_textures` set rather than the per-binding
    /// `current_depth_is_sampleable` flag is intentional:
    /// `GetDepthStencilSurface` returns a surface with `parent_texture: null`,
    /// so a save/restore cycle of a cascade depth surface lands in the `Eager`
    /// branch of `device_set_depth_stencil_surface` with `is_sampleable=false`
    /// — but the underlying Metal handle is the same cascade we marked
    /// earlier. Caller can therefore unconditionally call this with the
    /// current depth handle; non-cascade binds filter out here.
    pub fn note_caster_draw(&mut self, depth_tex: MetalHandle<MTLTextureKind>) {
        if !log_enabled!(target: CASCADE_PROBE_TARGET, Level::Trace) {
            return;
        }
        if depth_tex.is_null() || !self.seen_sampleable_depth_textures.contains(&depth_tex) {
            return;
        }
        *self.frame_caster_writes.entry(depth_tex).or_insert(0) += 1;
    }

    /// Drain the per-frame cascade summary.
    ///
    /// Returns `(frame_seq, [(cascade_tex, caster_writes, sample_binds)])`
    /// covering every cascade-depth handle that received caster writes AND
    /// every cascade-depth handle that was sampled this frame (union).
    /// Counters are cleared.
    ///
    /// The union shape matters: a cascade with `caster_writes=0 AND
    /// sample_binds>0` is the smoking gun for "receiver sampled a
    /// cascade with no fresh caster content this frame".
    #[must_use]
    pub fn take_cascade_frame_summary(
        &mut self,
    ) -> (u64, Vec<(MetalHandle<MTLTextureKind>, u32, u32)>) {
        let mut keys: HashSet<MetalHandle<MTLTextureKind>> = HashSet::with_capacity(
            self.frame_caster_writes.len() + self.frame_cascade_samples.len(),
        );
        keys.extend(self.frame_caster_writes.keys().copied());
        keys.extend(self.frame_cascade_samples.keys().copied());
        let mut rows: Vec<(MetalHandle<MTLTextureKind>, u32, u32)> = keys
            .into_iter()
            .map(|tex| {
                (
                    tex,
                    self.frame_caster_writes.get(&tex).copied().unwrap_or(0),
                    self.frame_cascade_samples.get(&tex).copied().unwrap_or(0),
                )
            })
            .collect();
        rows.sort_by_key(|(tex, _, _)| tex.raw());
        self.frame_caster_writes.clear();
        self.frame_cascade_samples.clear();
        (self.frame_seq, rows)
    }

    /// Capture the command index where a color clear-quad block is about to be emitted.
    ///
    /// Returns the start index for the caller to thread into
    /// `close_color_clear_quad_block` after the clear-quad's `emit_command`
    /// calls.
    ///
    /// Deliberately does NOT tag `color_writes_observed`: a clear-quad's
    /// output is a fixed RGBA over a viewport, and if the pass closes
    /// with no other color-writing draws, Rule H drops the block along
    /// with the color attachment (both are dead work). Opens a pass
    /// first if none is live (mirrors the `emit_command` contract).
    pub fn open_color_clear_quad_block(&mut self) -> usize {
        self.ensure_pass_open();
        self.passes.last().map_or(0, |p| p.commands.len())
    }

    /// Record the command range covered by the just-emitted color clear-quad.
    ///
    /// Caller passes the value returned by the matching
    /// `open_color_clear_quad_block` call. Zero-length ranges (caller emitted
    /// no commands between the open/close pair) are ignored.
    pub fn close_color_clear_quad_block(&mut self, start: usize) {
        if let Some(pass) = self.passes.last_mut() {
            let end = pass.commands.len();
            if end > start {
                pass.color_clear_quad_ranges.push((start, end));
            }
        }
    }

    pub fn emit_command(&mut self, cmd: Command) {
        self.ensure_pass_open();
        // Mirror every pushed command into the debug shadow at the single
        // funnel, so `FrameEncoder::debug_assert_cache_in_sync` can catch a
        // cached-slot emit that bypassed its `LastBoundCache` gate.
        #[cfg(debug_assertions)]
        self.debug_emitted.record(&cmd);
        if cmd.cmd == CommandType::SetFragmentTexture as u32 && cmd.param_b != 0 {
            // SAFETY: SetFragmentTexture's param_b holds a non-null MTLTexture
            // handle, packed from the encoder's typed cache via .raw().
            let tex = unsafe { MetalHandle::<MTLTextureKind>::new(cmd.param_b) };
            self.seen_sampled_textures.insert(tex);
            self.frame_sampled_textures.insert(tex);
            // Cascade-sample counter: gated on the probe target so the
            // HashMap inc is skipped at default `RUST_LOG`. The map
            // stays empty when off; `take_cascade_frame_summary` then
            // returns an empty Vec and the encoder-side summary block
            // short-circuits without further work.
            if log_enabled!(target: CASCADE_PROBE_TARGET, Level::Trace)
                && self.seen_sampleable_depth_textures.contains(&tex)
            {
                *self.frame_cascade_samples.entry(tex).or_insert(0) += 1;
            }
        }
        let mut realloc_bytes: u64 = 0;
        if let Some(pass) = self.passes.last_mut() {
            if cmd.cmd == CommandType::SetVisibilityResultMode as u32
                && cmd.param_a == VisibilityResultMode::Counting as u32
            {
                pass.has_counting_visibility = true;
            }
            // Detect Vec::push doubling: the realloc memcpys
            // `capacity * size_of::<Command>()` bytes from the old
            // buffer to the new one. The running total is the churn
            // figure the perf summary reports as `cmd_realloc`; once
            // `command_vec_pool` is warm it settles at zero.
            if pass.commands.len() == pass.commands.capacity() {
                let bytes = pass
                    .commands
                    .capacity()
                    .saturating_mul(size_of::<Command>());
                realloc_bytes = bytes as u64;
            }
            pass.commands.push(cmd);
        }
        if realloc_bytes != 0 {
            self.cmd_vec_realloc_bytes = self.cmd_vec_realloc_bytes.saturating_add(realloc_bytes);
        }
    }

    /// Debug-build accessor for the emitted-command shadow.
    ///
    /// Diffed against the encoder's `LastBoundCache` before each draw.
    #[cfg(debug_assertions)]
    #[must_use]
    pub const fn debug_emitted(&self) -> &DebugBoundShadow {
        &self.debug_emitted
    }

    /// Forget the emitted-command shadow.
    ///
    /// Call in lockstep with `LastBoundCache::reset` whenever a fresh Metal
    /// encoder opens, so the shadow and cache share the same "nothing bound
    /// yet" baseline.
    #[cfg(debug_assertions)]
    pub fn debug_reset_emitted(&mut self) {
        self.debug_emitted = DebugBoundShadow::default();
    }

    /// Drain and return the running per-frame total of bytes Metal will memcpy.
    ///
    /// The bytes are the memcpy cost of `Pass::commands` `Vec` doublings.
    /// Called once per frame from the encoder's `log_perf_summary`. Zeroes the
    /// field so the next frame starts fresh.
    pub const fn take_cmd_vec_realloc_bytes(&mut self) -> u64 {
        let bytes = self.cmd_vec_realloc_bytes;
        self.cmd_vec_realloc_bytes = 0;
        bytes
    }

    /// Sum `Vec::capacity() * size_of::<Command>()` across every pass's command buffer.
    ///
    /// Summed at end-of-frame. The pool recycles these vectors across frames
    /// so capacity is the resident-memory footprint, not a per-frame
    /// allocation cost. Paired with `take_cmd_vec_realloc_bytes` so the diag
    /// row can show steady-state size alongside growth churn.
    #[must_use]
    pub fn cmd_vec_capacity_bytes(&self) -> u64 {
        let elem = core::mem::size_of::<Command>() as u64;
        self.passes
            .iter()
            .map(|p| p.commands.capacity() as u64 * elem)
            .sum()
    }

    /// Ensure a pass is live for the next command.
    ///
    /// Opens a new `Pass` if the previous one was closed (or if this is the
    /// first command of the frame), consuming any pending clears and emitting
    /// the current viewport as the first command of the new pass.
    ///
    /// Rule A — first-use `DontCare`: when an attachment has not been
    /// seen yet this frame AND there is no pending clear AND no queued
    /// leading-blit writes the same attachment, the load action is
    /// `DontCare` instead of `Load`. Saves the TBDR tile-fill cost on
    /// passes that will fully overwrite undefined contents anyway.
    pub fn ensure_pass_open(&mut self) {
        if !self.current_pass_closed && !self.passes.is_empty() {
            return;
        }
        let (vpx, vpy, vpw, vph) = self.effective_viewport();
        let leading_blits = core::mem::take(&mut self.pending_leading_blits);

        // Rule A (FIRST_USE_DONTCARE) is only safe when the new pass
        // will WRITE the entire attachment — otherwise `DontCare` lets
        // Metal trash the un-rendered region. Sub-rect viewports (e.g.
        // a shared shadow cascade tile atlas, where one frame
        // renders a few 683x683 tiles into a 2048x2048 atlas while
        // expecting the other tiles from the previous frame to
        // survive) need `Load` so prior content carries forward — real
        // D3D9 drivers preserve depth content across frames; MTLD3D
        // must match.
        let viewport_covers_attachment = vpx == 0
            && vpy == 0
            && vpw == self.current_color_size.0
            && vph == self.current_color_size.1;
        let color_load = match self.pending_color_clear.take() {
            Some((r, g, b, a)) => ColorLoad::Clear { r, g, b, a },
            None if ENABLE_FIRST_USE_DONTCARE
                && viewport_covers_attachment
                && !self.current_color_texture.is_null()
                && !self.seen_color_rts.contains(&self.current_color_texture)
                && !self
                    .seen_sampled_textures
                    .contains(&self.current_color_texture)
                && !blit_list_writes(&leading_blits, self.current_color_texture) =>
            {
                ColorLoad::DontCare
            }
            None => ColorLoad::Load,
        };
        let depth_load = match self.pending_depth_clear.take() {
            Some(value) => DepthLoad::Clear { value },
            None if ENABLE_FIRST_USE_DONTCARE
                && viewport_covers_attachment
                && !self.current_depth_texture.is_null()
                && !self
                    .current_attachments
                    .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE)
                && !self.seen_depth_rts.contains(&self.current_depth_texture)
                && !self
                    .seen_sampled_textures
                    .contains(&self.current_depth_texture)
                && !blit_list_writes(&leading_blits, self.current_depth_texture) =>
            {
                DepthLoad::DontCare
            }
            None => DepthLoad::Load,
        };
        if !self.current_color_texture.is_null() {
            self.seen_color_rts.insert(self.current_color_texture);
        }
        if !self.current_depth_texture.is_null() {
            self.seen_depth_rts.insert(self.current_depth_texture);
        }

        // Reuse a `Vec<Command>` recycled from a previous frame's pass
        // (capacity preserved by `reset_frame`); fall back to a small
        // fresh allocation on the cold-start frame or after the pool
        // has been drained by a high-pass-count frame.
        let commands = self
            .command_vec_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(64));
        let mut pass = Pass {
            color_texture: self.current_color_texture,
            color_size: self.current_color_size,
            color_format: self.current_color_format,
            color_load,
            color_store: StoreAction::Store,
            depth_texture: self.current_depth_texture,
            depth_load,
            depth_store: StoreAction::Store,
            viewport: (vpx, vpy, vpw, vph),
            commands,
            leading_blits,
            has_counting_visibility: false,
            depth_is_sampleable: self
                .current_attachments
                .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE),
            color_writes_observed: false,
            color_clear_quad_ranges: Vec::new(),
        };
        pass.commands.push(Command::set_viewport(
            vpx,
            vpy,
            vpw,
            vph,
            self.viewport_min_z,
            self.viewport_max_z,
        ));
        // Seed the dedup with the viewport just emitted as this encoder's
        // first command, so a mid-pass `set_viewport` with the same value
        // (games re-set an unchanged viewport every frame) is skipped.
        self.last_emitted_viewport = Some((
            vpx,
            vpy,
            vpw,
            vph,
            self.viewport_min_z.to_bits(),
            self.viewport_max_z.to_bits(),
        ));
        self.passes.push(pass);
        self.current_pass_closed = false;
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            let idx = self.passes.len() - 1;
            trace!(
                target: TRACE_TARGET,
                "pass-open  idx={idx} color={:#x} depth={:#x} \
                 size={}x{} color_load={:?} depth_load={:?} viewport={vpx},{vpy}+{vpw}x{vph}",
                self.current_color_texture,
                self.current_depth_texture,
                self.current_color_size.0,
                self.current_color_size.1,
                color_load,
                depth_load,
            );
        }
    }

    /// Queue a blit to run before the *next* pass that opens.
    ///
    /// Caller should `end_current_pass()` immediately before pushing so that
    /// any in-flight render encoder closes first — the queued blit then orders
    /// correctly between the just-ended pass's draws and the next pass's
    /// draws. If no further pass opens this frame, `submit` drains the queue
    /// into a synthetic trailing blit-only pass via
    /// `take_pending_leading_blits`.
    pub fn push_pending_leading_blit(&mut self, blit: BlitCommand) {
        self.pending_leading_blits.push(blit);
    }

    /// Drain any leading blits queued after the last pass ended.
    ///
    /// Used by `submit` to synthesise a trailing blit-only pass when a
    /// `StretchRect` lands after the final draw of the frame.
    pub fn take_pending_leading_blits(&mut self) -> Vec<BlitCommand> {
        core::mem::take(&mut self.pending_leading_blits)
    }

    /// Close the current render pass.
    ///
    /// The next `emit_command` / `ensure_pass_open` opens a fresh pass using
    /// the attachments and pending clears in effect at that point. `caller` is
    /// a static identifier (e.g. `"set_color_rt"`, `"stretch_rect"`) emitted
    /// into the `mtld3d::d3d9::passes` trace probe so a frame log shows which
    /// trigger drove each pass break.
    pub fn end_current_pass(&mut self, caller: &'static str) {
        if !self.passes.is_empty() && !self.current_pass_closed {
            self.current_pass_closed = true;
            if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                let idx = self.passes.len() - 1;
                let last = &self.passes[idx];
                let draws = last
                    .commands
                    .iter()
                    .filter(|c| {
                        c.cmd == CommandType::DrawPrimitives as u32
                            || c.cmd == CommandType::DrawIndexedPrimitives as u32
                    })
                    .count();
                trace!(
                    target: TRACE_TARGET,
                    "pass-close idx={idx} caller={caller} color={:#x} depth={:#x} cmds={} draws={draws}",
                    last.color_texture,
                    last.depth_texture,
                    last.commands.len()
                );
            }
        }
    }

    /// Materialize any pending clears as a standalone pass on the current attachments.
    ///
    /// D3D9 semantics: `Clear()` applies to whichever rt is bound at call
    /// time; if the game then changes rt (or calls Present without drawing),
    /// the original target must still be cleared. This is a no-op when there
    /// are no pending clears.
    pub fn flush_pending_clears(&mut self) {
        if self.pending_color_clear.is_some() || self.pending_depth_clear.is_some() {
            self.ensure_pass_open();
            self.end_current_pass("flush_pending_clears");
        }
    }

    /// Rebind the color attachment for the next pass.
    ///
    /// No-op if the new texture matches the current one (games often re-assert
    /// the backbuffer between scenes).
    ///
    /// Only flushes pending clears when a *color* clear is pending: the
    /// color attachment is about to change, so the pending color clear
    /// must materialise on the outgoing rt (D3D9's
    /// Clear-then-SetRenderTarget ordering). A companion pending depth
    /// clear gets folded into the same materialised pass.
    ///
    /// If only a depth clear is pending (color clear is None), leave
    /// both pending and skip the flush — the depth attachment is
    /// unchanged across this setter, so the depth clear is still
    /// associated with the right surface and applies to the next
    /// user-issued pass. Without this gate the typical cascade-init
    /// sequence `SetRT(C) → Clear(TARGET) → SetDST(D) → Clear(ZBUFFER)
    /// → Draw` produced a spurious 1-cmd clear-only pass at the
    /// `SetDST` site.
    pub fn set_color_render_target(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) {
        if self.current_color_texture == texture {
            self.current_color_size = (width, height);
            self.current_color_format = format;
            return;
        }
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            trace!(
                target: TRACE_TARGET,
                "pass-break trigger=set_color_rt prev={:#x} new={:#x} new_size={width}x{height}",
                self.current_color_texture,
                texture,
            );
        }
        if self.pending_color_clear.is_some() {
            self.flush_pending_clears();
        }
        self.end_current_pass("set_color_rt");
        self.current_color_texture = texture;
        self.current_color_size = (width, height);
        self.current_color_format = format;
    }

    /// Rebind the depth/stencil attachment for the next pass.
    ///
    /// Mirrors `set_color_render_target`: only flushes pending clears when a
    /// pending *depth* clear exists (depth attachment is about to change). A
    /// solo pending color clear stays pending for the unchanged color
    /// attachment.
    pub fn set_depth_stencil_attachment(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        is_sampleable: bool,
        has_stencil: bool,
    ) {
        if is_sampleable && !texture.is_null() {
            self.seen_sampleable_depth_textures.insert(texture);
        }
        // `has_stencil` is a property of the bound texture's format, so a
        // repeat bind of the same texture carries the same value — fold it
        // before the no-change early-out below.
        self.current_attachments
            .set(CurrentAttachmentFlags::DEPTH_HAS_STENCIL, has_stencil);
        if self.current_depth_texture == texture
            && self
                .current_attachments
                .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE)
                == is_sampleable
        {
            return;
        }
        self.current_attachments
            .set(CurrentAttachmentFlags::DEPTH_SAMPLEABLE, is_sampleable);
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            trace!(
                target: TRACE_TARGET,
                "pass-break trigger=set_depth_attach prev={:#x} new={:#x}",
                self.current_depth_texture,
                texture,
            );
        }
        if self.pending_depth_clear.is_some() {
            self.flush_pending_clears();
        }
        self.end_current_pass("set_depth_attach");
        self.current_depth_texture = texture;
    }

    /// Apply a color clear.
    ///
    /// If the current pass already has draws, end it so the clear applies to
    /// the next pass's load action. If the current pass is open but has only
    /// the initial viewport, amend its load action in place. Otherwise stash
    /// as pending for the next pass-begin.
    pub fn clear_color(&mut self, r: u32, g: u32, b: u32, a: u32) -> ColorClearOutcome {
        let color_texture = self.current_color_texture;
        if self.current_pass_has_work() {
            // Pass has draws — translate the D3D9 viewport-clipped Clear
            // by asking the caller to emit a scissored clear-quad inside
            // this encoder. Visibility-counting passes (occlusion queries
            // active) fall back to the legacy pass-break: see comment in
            // `clear_depth`.
            if self.current_pass_has_counting_visibility() {
                mtld3d_shared::log_once_trace_by!(
                    target: DEPTH_TRACE_TARGET,
                    key: color_texture.raw(),
                    "clear-quad color: visibility-active → legacy pass-break (tex={color_texture:#x})"
                );
                self.end_current_pass("clear_color_vis_fallback");
            } else {
                let vp = self.effective_viewport();
                mtld3d_shared::log_once_trace_by!(
                    target: DEPTH_TRACE_TARGET,
                    key: color_texture.raw().rotate_left(13) ^ pack_viewport_key(vp),
                    "clear-quad color: EmitQuad tex={color_texture:#x} viewport=({},{},{}x{})",
                    vp.0, vp.1, vp.2, vp.3
                );
                return ColorClearOutcome::EmitQuad {
                    rgba: (r, g, b, a),
                    viewport: vp,
                    color_format: self.current_color_format,
                };
            }
        }
        // Cross-pass case: the color texture already received content
        // earlier this frame. Folding into a fresh pass's load action
        // would let Metal's full-attachment `loadAction = Clear` wipe
        // every prior tile. Open the pass with `Load` (preserving
        // content) and emit a scissored clear-quad instead. Only
        // applies when the new viewport is meaningful (non-zero size)
        // and a color texture is bound.
        if !color_texture.is_null()
            && self.seen_color_rts.contains(&color_texture)
            && self.viewport_width > 0
            && self.viewport_height > 0
        {
            let vp = self.effective_viewport();
            // Open the pass first (or take the existing one). When
            // already open with no work, rewrite the load action to
            // `Load` so the clear-quad's scissored write is the only
            // thing that lands in this tile; the previous tile's
            // content survives outside the scissor rect.
            self.ensure_pass_open();
            if let Some(pass) = self.passes.last_mut()
                && matches!(pass.color_load, ColorLoad::Clear { .. })
            {
                pass.color_load = ColorLoad::Load;
            }
            mtld3d_shared::log_once_trace_by!(
                target: DEPTH_TRACE_TARGET,
                key: color_texture.raw().rotate_left(29) ^ pack_viewport_key(vp),
                "clear-quad color: EmitQuad(cross-pass) tex={color_texture:#x} viewport=({},{},{}x{}) — preserved via Load action",
                vp.0, vp.1, vp.2, vp.3
            );
            return ColorClearOutcome::EmitQuad {
                rgba: (r, g, b, a),
                viewport: vp,
                color_format: self.current_color_format,
            };
        }
        if !self.current_pass_closed
            && let Some(pass) = self.passes.last_mut()
        {
            pass.color_load = ColorLoad::Clear { r, g, b, a };
            self.pending_color_clear = None;
            mtld3d_shared::log_once_trace_by!(
                target: DEPTH_TRACE_TARGET,
                key: color_texture.raw(),
                "clear-quad color: Folded(amend) tex={color_texture:#x} (first Clear in pass — load action set)"
            );
            return ColorClearOutcome::Folded;
        }
        self.pending_color_clear = Some((r, g, b, a));
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: color_texture.raw().rotate_left(7),
            "clear-quad color: Folded(pending) tex={color_texture:#x} (no pass open — stashed for next ensure_pass_open)"
        );
        ColorClearOutcome::Folded
    }

    /// Open (or reuse) the colour pass for a `Clear` with explicit `pRects` sub-regions.
    ///
    /// Prior tile content is preserved. A rect-clear can NEVER fold into
    /// a full-attachment `loadAction = Clear` (that wipes pixels outside
    /// the rects), so — exactly like `clear_color`'s cross-pass branch —
    /// open the pass with `Load` and rewrite a pending whole-attachment
    /// Clear to `Load`. The caller then emits one scissored clear-quad
    /// per clipped rect via `emit_clear_quad_color_inner`, reusing the
    /// proven clear-quad path (so there is no fresh draw-without-encoder
    /// hazard). Returns the bound colour format for the quad pipeline
    /// key.
    pub fn begin_region_color_clear(&mut self) -> PixelFormat {
        // A clear-quad is a draw; under an active occlusion query, break the
        // pass first so the synthetic draw can't pollute the visibility count
        // (mirrors `clear_color`'s visibility fallback).
        if self.current_pass_has_counting_visibility() {
            self.end_current_pass("region_color_clear_vis");
        }
        // A pending whole-RT colour clear (a prior `Clear(NULL)` not yet
        // realised) MUST land under the rect quads — per the D3D9 spec,
        // `Clear(NULL, white)` then `Clear(rects, red)` yields white
        // everywhere outside the rects. `ensure_pass_open` turns that pending
        // clear into `loadAction = Clear`; keep it so the whole RT clears
        // first, then the rect quads overwrite the rects. Only when there is NO
        // pending clear (the RT already holds drawn/loaded content to preserve)
        // do we rewrite a `Clear` load to `Load`.
        //
        // Crucially, only touch the load action when WE freshly opened the pass.
        // If a pass is already open — e.g. a sequence of region clears in one
        // frame like `Clear(NULL,green)` then `Clear(rect,red)` under a scissor
        // — its load action is already committed (and may carry an earlier
        // realised whole-RT Clear); rewriting it to Load here would drop that
        // clear and the prior frame's content would load through instead.
        let was_closed = self.current_pass_closed();
        let had_pending_clear = self.pending_color_clear.is_some();
        self.ensure_pass_open();
        if was_closed
            && !had_pending_clear
            && let Some(pass) = self.passes.last_mut()
            && matches!(pass.color_load, ColorLoad::Clear { .. })
        {
            pass.color_load = ColorLoad::Load;
        }
        self.current_color_format
    }

    /// Apply a depth clear.
    ///
    /// Mirrors `clear_color` semantics for the depth attachment's load
    /// action. Routes through one of four paths, checked in order:
    ///
    /// 1. Active pass with draws → emit a scissored clear-quad (or fall
    ///    back to pass-break under visibility counting).
    /// 2. Cross-pass — depth texture already received content this frame
    ///    → open a Load-action pass + emit a clear-quad to avoid wiping
    ///    prior tiles.
    /// 3. Open pass with no draws yet → amend its load action to Clear.
    /// 4. No open pass → stash as `pending_depth_clear`.
    pub fn clear_depth(&mut self, value: u32) -> DepthClearOutcome {
        let depth_texture = self.current_depth_texture;
        if self.current_pass_has_work()
            && let Some(outcome) = self.clear_depth_in_active_pass(value, depth_texture)
        {
            return outcome;
        }
        if let Some(outcome) = self.clear_depth_cross_pass(value, depth_texture) {
            return outcome;
        }
        if let Some(outcome) = self.clear_depth_amend_open(value, depth_texture) {
            return outcome;
        }
        self.clear_depth_stash_pending(value, depth_texture)
    }

    /// Active-pass branch.
    ///
    /// Returns `Some(EmitQuad)` on the normal path or `None` if a
    /// visibility-counting query forced the legacy pass-break fallback
    /// (caller falls through to the cross-pass / amend chain).
    ///
    /// Falling through to `end_current_pass` here would open a new
    /// encoder with `loadAction = Clear`, which on Metal clears the
    /// WHOLE depth attachment regardless of viewport — wiping prior
    /// tile draws under a shared shadow-atlas pattern.
    /// `FrameEncoder::clear_depth` paints the constant clear value via
    /// a scissored fullscreen quad inside the live encoder instead.
    ///
    /// Visibility-active exception: a clear-quad's draw would falsely
    /// increment the fragment counter, so the legacy pass-break is
    /// retained until full save/restore of the
    /// `SetVisibilityResultMode` offset lands.
    fn clear_depth_in_active_pass(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> Option<DepthClearOutcome> {
        if self.current_pass_has_counting_visibility() {
            mtld3d_shared::log_once_trace_by!(
                target: DEPTH_TRACE_TARGET,
                key: depth_texture.raw(),
                "clear-quad depth: visibility-active → legacy pass-break (tex={depth_texture:#x})"
            );
            self.end_current_pass("clear_depth_vis_fallback");
            return None;
        }
        let vp = self.effective_viewport();
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: depth_texture.raw().rotate_left(13) ^ pack_viewport_key(vp),
            "clear-quad depth: EmitQuad tex={depth_texture:#x} viewport=({},{},{}x{}) value={:?}",
            vp.0, vp.1, vp.2, vp.3, f32::from_bits(value)
        );
        Some(DepthClearOutcome::EmitQuad {
            value,
            viewport: vp,
            has_color: !self.current_color_texture.is_null(),
            color_format: self.current_color_format,
        })
    }

    /// Cross-pass branch.
    ///
    /// The depth texture already received content earlier this frame.
    /// Folding into a fresh pass's load action would let Metal's
    /// full-attachment `loadAction = Clear` wipe every prior tile — the
    /// failure mode for a shared shadow cascade-atlas. Open the pass
    /// with `Load` (preserving content) and emit a scissored clear-quad
    /// instead. Only applies when the new viewport is meaningful
    /// (non-zero size) and a depth texture is bound.
    fn clear_depth_cross_pass(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> Option<DepthClearOutcome> {
        if depth_texture.is_null()
            || !self.seen_depth_rts.contains(&depth_texture)
            || self.viewport_width == 0
            || self.viewport_height == 0
        {
            return None;
        }
        let vp = self.effective_viewport();
        self.ensure_pass_open();
        if let Some(pass) = self.passes.last_mut()
            && matches!(pass.depth_load, DepthLoad::Clear { .. })
        {
            pass.depth_load = DepthLoad::Load;
        }
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: depth_texture.raw().rotate_left(29) ^ pack_viewport_key(vp),
            "clear-quad depth: EmitQuad(cross-pass) tex={depth_texture:#x} viewport=({},{},{}x{}) value={:?} — preserved via Load action",
            vp.0, vp.1, vp.2, vp.3, f32::from_bits(value)
        );
        Some(DepthClearOutcome::EmitQuad {
            value,
            viewport: vp,
            has_color: !self.current_color_texture.is_null(),
            color_format: self.current_color_format,
        })
    }

    /// Amend branch.
    ///
    /// If a pass is open with no draws yet, set its depth load action to
    /// `Clear` and clear any pending fallback.
    fn clear_depth_amend_open(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> Option<DepthClearOutcome> {
        if self.current_pass_closed {
            return None;
        }
        let pass = self.passes.last_mut()?;
        pass.depth_load = DepthLoad::Clear { value };
        self.pending_depth_clear = None;
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: depth_texture.raw(),
            "clear-quad depth: Folded(amend) tex={depth_texture:#x} (first Clear in pass — load action set)"
        );
        Some(DepthClearOutcome::Folded)
    }

    /// Stash branch.
    ///
    /// No open pass to amend, no cross-pass case to quad-clear — record
    /// the clear as pending so the next `ensure_pass_open` opens the
    /// pass with `loadAction = Clear`.
    fn clear_depth_stash_pending(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> DepthClearOutcome {
        self.pending_depth_clear = Some(value);
        mtld3d_shared::log_once_trace_by!(
            target: DEPTH_TRACE_TARGET,
            key: depth_texture.raw().rotate_left(7),
            "clear-quad depth: Folded(pending) tex={depth_texture:#x} (no pass open — stashed for next ensure_pass_open)"
        );
        DepthClearOutcome::Folded
    }

    fn current_pass_has_counting_visibility(&self) -> bool {
        if self.current_pass_closed {
            return false;
        }
        self.passes
            .last()
            .is_some_and(|p| p.has_counting_visibility)
    }

    /// Legacy "end pass on Clear" fallback for when the clear-quad pipeline create fails.
    ///
    /// Used by the encoder layer. Restores the pre-clear-quad
    /// behaviour: end the current pass, then either amend the next
    /// pass's load action (if a fresh pass is already opened later in
    /// the frame) or stash as `pending_depth_clear` so the next
    /// pass-open consumes it.
    pub fn clear_depth_legacy_break(&mut self, value: u32) {
        self.end_current_pass("clear_depth_legacy_fallback");
        self.pending_depth_clear = Some(value);
    }

    /// Color mirror of `clear_depth_legacy_break`.
    pub fn clear_color_legacy_break(&mut self, r: u32, g: u32, b: u32, a: u32) {
        self.end_current_pass("clear_color_legacy_fallback");
        self.pending_color_clear = Some((r, g, b, a));
    }

    /// Resolve the `(x, y, w, h)` rect that `emit_scissor` would emit for the given inputs.
    ///
    /// Exposed so the encoder wrapper can dedup against the *resolved*
    /// rect — when scissor test is disabled, the rect falls back to the
    /// current viewport, which can change mid-pass.
    #[must_use]
    pub const fn resolved_scissor_rect(
        &self,
        test_enable: bool,
        rect: [u32; 4],
    ) -> (u32, u32, u32, u32) {
        if test_enable && rect[2] != 0 && rect[3] != 0 {
            (rect[0], rect[1], rect[2], rect[3])
        } else {
            self.effective_viewport()
        }
    }

    /// Test-only direct emit of `setScissorRect`.
    ///
    /// Production code goes through `FrameEncoder::emit_scissor`
    /// (`encoder.rs`), which calls `resolved_scissor_rect` for the rect
    /// math and routes the emit through `LastBoundCache` for dedup. A
    /// bypass here would let a caller silently re-introduce
    /// cache-vs-encoder drift the clear-quad `LastBoundCache` routing
    /// already closes.
    #[cfg(test)]
    fn emit_scissor(&mut self, test_enable: bool, rect: [u32; 4]) {
        let (x, y, w, h) = self.resolved_scissor_rect(test_enable, rect);
        self.emit_command(Command::set_scissor_rect(x, y, w, h));
    }

    /// Update the tracked viewport.
    ///
    /// If the render pass is already open, also emit a `setViewport`
    /// command so later draws see the change.
    pub fn set_viewport(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        min_z: f32,
        max_z: f32,
    ) {
        self.viewport_x = x;
        self.viewport_y = y;
        self.viewport_width = width;
        self.viewport_height = height;
        self.viewport_min_z = min_z;
        self.viewport_max_z = max_z;
        // Re-emit only on an actual change. A fresh `set_viewport` whose
        // value matches what was last emitted on this encoder would be a
        // redundant Metal bind (Xcode's "bound … when it was already
        // bound"); the z-range is part of the key, compared by bits so a
        // depth-range-only change (sky / weapon) still re-emits.
        let key = (x, y, width, height, min_z.to_bits(), max_z.to_bits());
        if !self.current_pass_closed
            && self.last_emitted_viewport != Some(key)
            && let Some(pass) = self.passes.last_mut()
        {
            pass.commands
                .push(Command::set_viewport(x, y, width, height, min_z, max_z));
            pass.viewport = (x, y, width, height);
            self.last_emitted_viewport = Some(key);
        }
    }

    /// Rule G — strip the color attachment from clear-only passes.
    ///
    /// Fires when the pass's color side is provably wasted
    /// (`color_store == DontCare`, no draws, no leading blits). Result:
    /// depth-only Metal render pass on the unix side. Eliminates Apple's
    /// "Unused Texture" Insight on the cascade-color placeholder in
    /// cascade-init clear-only sub-passes.
    ///
    /// Must run after `finalize_store_actions` so the Store decisions
    /// are stable, but before `cull_dead_clear_only_passes` so the
    /// cull's `color_writes` check sees `color_texture == 0` on
    /// stripped passes.
    pub fn strip_dead_color_in_clear_only_passes(&mut self) {
        if !ENABLE_STRIP_DEAD_COLOR_IN_CLEAR_ONLY {
            return;
        }
        for pass in &mut self.passes {
            let has_draw = pass.commands.iter().any(|c| {
                c.cmd == CommandType::DrawPrimitives as u32
                    || c.cmd == CommandType::DrawIndexedPrimitives as u32
            });
            if has_draw || !pass.leading_blits.is_empty() {
                continue;
            }
            if !pass.color_texture.is_null()
                && matches!(pass.color_store, StoreAction::DontCare)
                && !pass.depth_texture.is_null()
            {
                let stripped = pass.color_texture;
                pass.color_texture = MetalHandle::NULL;
                // Once the color attachment is gone, the load/store
                // actions are moot for the unix side; reset them to
                // their unused defaults so a stale `Clear` doesn't
                // mislead readers.
                pass.color_load = ColorLoad::DontCare;
                pass.color_store = StoreAction::DontCare;
                if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                    trace!(
                        target: TRACE_TARGET,
                        "pass-strip color={stripped:#x} → depth-only (clear-only pass)",
                    );
                }
            }
        }
    }

    /// Rule H — strip the color attachment from passes-with-draws.
    ///
    /// Applies where every real (non-clear-quad) draw ran with
    /// `D3DRS_COLORWRITEENABLE == 0`. Symmetric to Rule G but for the
    /// with-draws case. The pass's `SetRenderPipelineState` commands
    /// have their `param_b` rewritten from the original (with-color)
    /// pipeline handle to the matching no-color variant via `alt`, the
    /// color clear-quad blocks (if any) are removed entirely, and the
    /// color attachment is dropped — the unix `encode_pass` already
    /// supports the `color_texture == 0 && depth_texture != 0` shape
    /// (Rule G is the existing precedent).
    ///
    /// Color clear-quad blocks are walked separately: their pipelines
    /// declare a color output (they have to, to write the clear value)
    /// and would fail Metal's pipeline-vs-RP format validation against
    /// the stripped descriptor. Removing them is sound here because
    /// once the color attachment is gone, the clear-quad's writes are
    /// dead anyway — the pass is now depth-only and the cascade-color
    /// VRAM is never read.
    ///
    /// If the side-map is missing an entry for a non-clear-quad `SetPSO`
    /// inside a candidate pass, abort the strip for that pass (single
    /// `log_once` warning) — means a zero-mask draw skipped the
    /// dual-build path in `FrameEncoder::get_or_create_pipeline`, which
    /// would be a correctness bug elsewhere.
    ///
    /// Must run after `strip_dead_color_in_clear_only_passes` (Rule G)
    /// so clear-only passes are already handled, and before
    /// `cull_dead_clear_only_passes` (Rule F) — though Rule F won't
    /// touch the pass anyway because it still has draws.
    pub fn strip_color_from_no_color_draw_passes<S: BuildHasher>(
        &mut self,
        alt: &HashMap<u64, MetalHandle<MTLRenderPipelineStateKind>, S>,
    ) {
        if !ENABLE_NO_COLOR_PASS_FOR_DRAWS {
            return;
        }
        for pass in &mut self.passes {
            if pass.color_writes_observed
                || pass.color_texture.is_null()
                || pass.depth_texture.is_null()
                // A folded color Clear (`color_load == Clear`) is a real color
                // write even with no draw to tag `color_writes_observed` — e.g.
                // a backbuffer color Clear that shares a pass with a cross-pass
                // depth clear-quad. Stripping color here would discard that
                // clear; a later Load pass would then read black.
                || matches!(pass.color_load, ColorLoad::Clear { .. })
            {
                continue;
            }
            // Local copy so we can mutate `pass.commands` below while
            // still classifying indices.
            let cq_ranges = pass.color_clear_quad_ranges.clone();
            let in_clear_quad =
                |idx: usize| -> bool { cq_ranges.iter().any(|(s, e)| idx >= *s && idx < *e) };
            // A "real" draw is a draw command outside every clear-quad
            // block. A pass with only clear-quad blocks is somebody
            // else's territory (Rule F / Rule G).
            let has_real_draw = pass.commands.iter().enumerate().any(|(idx, c)| {
                !in_clear_quad(idx)
                    && (c.cmd == CommandType::DrawPrimitives as u32
                        || c.cmd == CommandType::DrawIndexedPrimitives as u32)
            });
            if !has_real_draw {
                continue;
            }
            // Confirm every non-clear-quad SetPSO has a no-color sibling
            // in the side-map. Clear-quad SetPSOs are exempt because the
            // block is about to be removed.
            let all_resolvable = pass.commands.iter().enumerate().all(|(idx, c)| {
                c.cmd != CommandType::SetRenderPipelineState as u32
                    || in_clear_quad(idx)
                    || alt.contains_key(&c.param_b)
            });
            if !all_resolvable {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "strip_color_from_no_color_draw_passes: side-map miss → keeping color attachment");
                continue;
            }
            // Rewrite non-clear-quad SetPSO handles to the no-color
            // variant. Clear-quad SetPSOs are about to be removed
            // wholesale, so leave them alone here.
            for (idx, c) in pass.commands.iter_mut().enumerate() {
                if in_clear_quad(idx) {
                    continue;
                }
                if c.cmd == CommandType::SetRenderPipelineState as u32
                    && let Some(&no_color) = alt.get(&c.param_b)
                {
                    c.param_b = no_color.raw();
                }
            }
            // Remove clear-quad blocks in reverse order so earlier
            // ranges' indices stay valid as we drain.
            let dropped_cmds: usize = cq_ranges.iter().map(|(s, e)| e - s).sum();
            for (start, end) in cq_ranges.iter().rev() {
                pass.commands.drain(*start..*end);
            }
            pass.color_clear_quad_ranges.clear();
            let stripped = pass.color_texture;
            pass.color_texture = MetalHandle::NULL;
            pass.color_load = ColorLoad::DontCare;
            pass.color_store = StoreAction::DontCare;
            if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                trace!(
                    target: TRACE_TARGET,
                    "pass-strip color={stripped:#x} → depth-only \
                     (all draws color_write_mask=0; dropped {dropped_cmds} clear-quad cmds)",
                );
            }
        }
    }

    /// Rule F — cull clear-only passes that perform no observable work.
    ///
    /// Runs after Rules B/C/D finalise. A pass with zero draw commands,
    /// no leading blits, and every attachment's Store flipped to
    /// `DontCare` writes nothing to VRAM and exists purely as encoder
    /// overhead; drop it. Typical case: a cascade init clear-only pass
    /// for a depth texture that is never sampled this frame, so Rule B
    /// flipped depth Store=DontCare on top of Rule C already flipping
    /// the color side.
    ///
    /// Must run after `finalize_load_actions` / `finalize_store_actions`
    /// so the Store decisions are stable.
    pub fn cull_dead_clear_only_passes(&mut self) {
        if !ENABLE_CULL_DEAD_CLEAR_PASSES {
            return;
        }
        let before = self.passes.len();
        self.passes.retain(|p| {
            let has_draw = p.commands.iter().any(|c| {
                c.cmd == CommandType::DrawPrimitives as u32
                    || c.cmd == CommandType::DrawIndexedPrimitives as u32
            });
            if has_draw || !p.leading_blits.is_empty() {
                return true;
            }
            let color_writes =
                !p.color_texture.is_null() && matches!(p.color_store, StoreAction::Store);
            let depth_writes =
                !p.depth_texture.is_null() && matches!(p.depth_store, StoreAction::Store);
            color_writes || depth_writes
        });
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            let dropped = before - self.passes.len();
            if dropped > 0 {
                trace!(
                    target: TRACE_TARGET,
                    "pass-cull dropped={dropped} dead clear-only passes",
                );
            }
        }
    }

    /// Rule E — coalesce clear-only passes into the load action of the next pass.
    ///
    /// The merge target is the next pass that attaches the same texture.
    /// `WoW`'s frame pattern commonly does `Clear(target) → SetRT(other)
    /// → … → SetRT(target) → Draw`, which currently produces a spurious
    /// 1-cmd clear-only pass at the `SetRT(other)` site that just clears
    /// the original target in isolation (with a Load on whatever else
    /// was attached). Folding that Clear into the next pass on the same
    /// target removes the spurious pass entirely.
    ///
    /// A merge is safe iff no intervening pass reads the target (as a
    /// fragment sampler input or as a blit source). If anything in
    /// between *would* observe the cleared content, the clear-only
    /// pass must materialise where it was originally placed.
    ///
    /// "Clear-only" means the pass has zero `DrawPrimitives` /
    /// `DrawIndexedPrimitives` commands; any setviewport / setscissor /
    /// setpipeline / setBlendColor that the encoder pushed without a
    /// subsequent draw still counts as clear-only here.
    pub fn coalesce_clear_only_passes(&mut self) {
        let mut i = 0;
        while i < self.passes.len() {
            let p = &self.passes[i];
            let has_draw = p.commands.iter().any(|c| {
                c.cmd == CommandType::DrawPrimitives as u32
                    || c.cmd == CommandType::DrawIndexedPrimitives as u32
            });
            let needs_color = !has_draw && matches!(p.color_load, ColorLoad::Clear { .. });
            let needs_depth = !has_draw && matches!(p.depth_load, DepthLoad::Clear { .. });
            if !needs_color && !needs_depth {
                i += 1;
                continue;
            }
            let target_color = p.color_texture;
            let target_depth = p.depth_texture;
            let color_load = p.color_load;
            let depth_load = p.depth_load;
            // Pass has only Clear load actions; no real draws / state.
            // Look ahead for a merge target. Both attachments (if Clear)
            // must match the target pass's attachments AND that pass
            // must currently be Loading them (so the move is observable
            // and lossless). Bail on any intervening read of either.
            let target_idx = self.find_clear_merge_target(
                i,
                target_color,
                target_depth,
                needs_color,
                needs_depth,
            );
            if let Some(t) = target_idx {
                if needs_color {
                    self.passes[t].color_load = color_load;
                }
                if needs_depth {
                    self.passes[t].depth_load = depth_load;
                }
                if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                    trace!(
                        target: TRACE_TARGET,
                        "pass-coalesce drop idx={i} (clear-only) → fold into idx={t} color={target_color:#x} depth={target_depth:#x}",
                    );
                }
                self.passes.remove(i);
                // Don't increment i — what was at i+1 is now at i.
            } else {
                i += 1;
            }
        }
    }

    /// Walk `passes[start+1..]` looking for the first pass that reattaches the target.
    ///
    /// The target color/depth must come back with `Load` so we can move
    /// Rule E's Clear into it. Bail on any intervening pass that reads
    /// the target as a fragment sampler input, as a blit source, or
    /// attaches it with `Clear` itself (that pass already overwrites
    /// whatever we'd move).
    fn find_clear_merge_target(
        &self,
        start: usize,
        target_color: MetalHandle<MTLTextureKind>,
        target_depth: MetalHandle<MTLTextureKind>,
        needs_color: bool,
        needs_depth: bool,
    ) -> Option<usize> {
        for j in (start + 1)..self.passes.len() {
            let cand = &self.passes[j];
            // Intervening read on a side we care about kills the merge.
            if needs_color && pass_reads_texture(cand, target_color) {
                return None;
            }
            if needs_depth && pass_reads_texture(cand, target_depth) {
                return None;
            }
            // Intervening Clear on the same attachment supersedes ours.
            if needs_color
                && cand.color_texture == target_color
                && matches!(cand.color_load, ColorLoad::Clear { .. })
            {
                return None;
            }
            if needs_depth
                && cand.depth_texture == target_depth
                && matches!(cand.depth_load, DepthLoad::Clear { .. })
            {
                return None;
            }
            // Match: same attachments, currently loading.
            let color_ok = !needs_color
                || (cand.color_texture == target_color
                    && matches!(cand.color_load, ColorLoad::Load));
            let depth_ok = !needs_depth
                || (cand.depth_texture == target_depth
                    && matches!(cand.depth_load, DepthLoad::Load));
            if color_ok && depth_ok {
                return Some(j);
            }
            // This pass consumes (Loads) one of the to-be-cleared attachments
            // but is NOT a full merge target (the other side doesn't match).
            // Folding the combined Clear into a later pass would let it
            // leapfrog this consumer, which then loads uninitialised content
            // (a render-to-texture pass that depth-tests against the auto-DS
            // sits between the clear-only pass and the final backbuffer pass).
            // Bail so the
            // clear-only pass materialises and this consumer loads the real
            // cleared content. WoW's pattern is unaffected — its first
            // Load pass matches BOTH sides and returns above.
            let consumes_color = needs_color
                && cand.color_texture == target_color
                && matches!(cand.color_load, ColorLoad::Load);
            let consumes_depth = needs_depth
                && cand.depth_texture == target_depth
                && matches!(cand.depth_load, DepthLoad::Load);
            if consumes_color || consumes_depth {
                return None;
            }
        }
        None
    }

    /// Rule A correction — revert `Load = DontCare` on attachments a fragment sampler reads.
    ///
    /// The revert fires whenever the attachment's content is read
    /// elsewhere in this frame. `ensure_pass_open` decides the load
    /// action eagerly without lookahead, so a pass that attaches a
    /// texture first AND lacks a pending clear gets `DontCare`; if a
    /// later pass then samples that texture (CSM cascade rendered then
    /// sampled by the scene PS), the sampler reads tile memory that was
    /// never loaded. Conservative: reverts even when the sampler bind
    /// happened earlier in the frame than the attachment (sampler
    /// already completed against VRAM), trading one tile load for
    /// safety.
    pub fn finalize_load_actions(&mut self) {
        if !ENABLE_FIRST_USE_DONTCARE {
            return;
        }
        for pass in &mut self.passes {
            if matches!(pass.color_load, ColorLoad::DontCare)
                && self.seen_sampled_textures.contains(&pass.color_texture)
            {
                pass.color_load = ColorLoad::Load;
                if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                    trace!(
                        target: TRACE_TARGET,
                        "pass-load color={:#x} DontCare → Load (sampled this frame)",
                        pass.color_texture,
                    );
                }
            }
            if matches!(pass.depth_load, DepthLoad::DontCare)
                && self.seen_sampled_textures.contains(&pass.depth_texture)
            {
                pass.depth_load = DepthLoad::Load;
                if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                    trace!(
                        target: TRACE_TARGET,
                        "pass-load depth={:#x} DontCare → Load (sampled this frame)",
                        pass.depth_texture,
                    );
                }
            }
        }
    }

    /// Rule B — flip `depth_store` to `DontCare` on each depth attachment's *last* pass.
    ///
    /// Scoped to this frame. D3D9 spec says depth/stencil contents are
    /// undefined across `Present`, so the final flush back to device
    /// memory is wasted bandwidth on TBDR.
    /// Also flips `color_store` to `DontCare` on a pass whose very
    /// next consumer of the same color rt this frame begins with a
    /// full-attachment `Clear` (Rule C) — the next pass's `Clear`
    /// provably overwrites the prior contents, so storing them is
    /// wasted bandwidth.
    ///
    /// Both rules skip the flip when the texture is bound as a fragment
    /// sampler somewhere in the frame (`seen_sampled_textures`): the
    /// sampler reads VRAM at draw time, so `DontCare` would discard the
    /// content it expects (CSM cascade written here, sampled in the
    /// scene pass).
    ///
    /// Called once at frame submit, after `end_current_pass`, before
    /// the unix-side thunk is dispatched.
    ///
    /// Each rule is one reverse walk over `passes`:
    /// - Rule B: the first pass we see with a given `depth_texture` is
    ///   the last in forward order; flip and mark handled.
    /// - Rule C: maintain `next_color_use: HashMap<u64, usize>` from
    ///   color texture to the most-recently-seen pass (i.e. the next
    ///   in forward order). For pass `i`, if `next_color_use[i.color]`
    ///   resolves and that next pass's `color_load` is `Clear`, flip
    ///   `i.color_store`. Then update the map with `i`.
    pub fn finalize_store_actions(&mut self) {
        if ENABLE_LAST_USE_DEPTH_DONTCARE {
            let mut handled: HashSet<MetalHandle<MTLTextureKind>> =
                HashSet::with_capacity(self.seen_depth_rts.len());
            for pass in self.passes.iter_mut().rev() {
                if pass.depth_texture.is_null() {
                    continue;
                }
                if handled.insert(pass.depth_texture) {
                    if pass.depth_is_sampleable {
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store depth={:#x} → keep Store (sampleable shadow map)",
                                pass.depth_texture,
                            );
                        }
                    } else if self.seen_sampled_textures.contains(&pass.depth_texture) {
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store depth={:#x} → keep Store (ever sampled)",
                                pass.depth_texture,
                            );
                        }
                    } else {
                        pass.depth_store = StoreAction::DontCare;
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store depth={:#x} → DontCare (last-use)",
                                pass.depth_texture,
                            );
                        }
                    }
                }
            }
        }
        if ENABLE_NEXT_CLEAR_COLOR_DONTCARE {
            let mut next_color_use: HashMap<MetalHandle<MTLTextureKind>, usize> =
                HashMap::with_capacity(self.seen_color_rts.len());
            for i in (0..self.passes.len()).rev() {
                let rt = self.passes[i].color_texture;
                if !rt.is_null()
                    && let Some(&next) = next_color_use.get(&rt)
                    && matches!(self.passes[next].color_load, ColorLoad::Clear { .. })
                    && !self.seen_sampled_textures.contains(&rt)
                {
                    self.passes[i].color_store = StoreAction::DontCare;
                    if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                        trace!(
                            target: TRACE_TARGET,
                            "pass-store idx={i} color={rt:#x} → DontCare (next-clear at idx={next})",
                        );
                    }
                }
                next_color_use.insert(rt, i);
            }
        }
        if ENABLE_LAST_USE_COLOR_DONTCARE {
            let mut handled: HashSet<MetalHandle<MTLTextureKind>> =
                HashSet::with_capacity(self.seen_color_rts.len());
            for pass in self.passes.iter_mut().rev() {
                let rt = pass.color_texture;
                if rt.is_null() || rt == self.backbuffer_texture {
                    continue;
                }
                if handled.insert(rt) && !matches!(pass.color_store, StoreAction::DontCare) {
                    if self.seen_sampled_textures.contains(&rt) {
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store color={rt:#x} → keep Store (sampled this frame)",
                            );
                        }
                    } else {
                        pass.color_store = StoreAction::DontCare;
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store color={rt:#x} → DontCare (last-use, non-backbuffer)",
                            );
                        }
                    }
                }
            }
        }
    }

    fn current_pass_has_work(&self) -> bool {
        if self.current_pass_closed {
            return false;
        }
        self.passes.last().is_some_and(|p| p.commands.len() > 1)
    }
}

/// True if `pass` would observe the contents of `target_handle`.
///
/// Either as a fragment-sampler input inside the pass (the typical
/// case) or as a leading blit's source texture. Used by
/// `coalesce_clear_only_passes` to decide whether moving a Clear past
/// this pass is safe: if the pass reads the pre-Clear contents, the
/// merge changes observable behaviour and is rejected.
///
/// `target_handle == 0` is treated as "no read" since 0 is the unset
/// sentinel for texture handles.
fn pass_reads_texture(pass: &Pass, target_handle: MetalHandle<MTLTextureKind>) -> bool {
    if target_handle.is_null() {
        return false;
    }
    let target_raw = target_handle.raw();
    let sampler_reads = pass
        .commands
        .iter()
        .any(|c| c.cmd == CommandType::SetFragmentTexture as u32 && c.param_b == target_raw);
    if sampler_reads {
        return true;
    }
    pass.leading_blits.iter().any(|b| {
        let reads_src = matches!(
            BlitCommandType::from_repr(b.cmd),
            Some(BlitCommandType::CopyTextureToTexture | BlitCommandType::GenerateMipmaps)
        );
        reads_src && b.src_handle == target_raw
    })
}

/// True if any blit in `blits` writes to texture `target_handle`.
///
/// Used at pass-open to disqualify `LoadAction::DontCare` on an
/// attachment that just got a leading blit's output (`StretchRect`'s
/// typical pattern: copy A → B, then render onto B; the next pass MUST
/// `Load` to preserve the blit's contents).
///
/// `NotifyBufferDidModifyRange` and `CopyBufferToBuffer` carry buffer
/// handles in `src_handle`/`dst_handle` — never texture handles — so
/// they're safely filtered out by the type-mismatch on the handle
/// value (texture handles are disjoint from buffer handles in Metal).
/// The exhaustive match makes any new `BlitCommandType` a compile
/// error here, forcing the author to classify it.
fn blit_list_writes(blits: &[BlitCommand], target_handle: MetalHandle<MTLTextureKind>) -> bool {
    if target_handle.is_null() {
        return false;
    }
    let target_raw = target_handle.raw();
    // allow: identical bodies (`=> true`) on the texture-writing arms and
    // the `None` arm are intentional — the exhaustive match on `Some(...)`
    // turns "new BlitCommandType variant" into a compile error here,
    // forcing the author to classify it. Collapsing to `Some(_) | None
    // => true` would silently default new variants to "texture-writing",
    // defeating that.
    blits.iter().any(|b| {
        // Unknown variants on the wire → conservatively assume "writes texture".
        let Some(ty) = BlitCommandType::from_repr(b.cmd) else {
            return b.dst_handle == target_raw;
        };
        // Exhaustive match keeps the compile-error-on-new-variant safety:
        // any new BlitCommandType forces an author here to classify it.
        let writes_texture = match ty {
            BlitCommandType::CopyBufferToTexture
            | BlitCommandType::CopyTextureToTexture
            | BlitCommandType::GenerateMipmaps => true,
            BlitCommandType::CopyBufferToBuffer | BlitCommandType::NotifyBufferDidModifyRange => {
                false
            }
        };
        writes_texture && b.dst_handle == target_raw
    })
}

impl Default for PassState {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-render-pass last-bound state cache.
///
/// Skips redundant `setFragmentSamplerState` / `setFragmentTexture` /
/// `setRenderPipelineState` / `setDepthStencilState` / `setCullMode`
/// emissions when the value matches what was last bound on the same
/// `MTLRenderCommandEncoder`. State persists across draws within a Metal
/// render encoder, so the cache is sound as long as `reset` is called on
/// every new-pass entry.
///
/// `0` is the unset sentinel for the `u64` handles — Metal object pointers
/// are never zero, so the first emission of a real handle always reports
/// "changed". `cull_mode` uses `Option<CullMode>` because `CullMode::None`
/// (value 0) is a valid binding distinct from "not yet bound".
pub struct LastBoundCache {
    fragment_samplers: [u64; LAST_BOUND_MAX_STAGES],
    fragment_textures: [u64; LAST_BOUND_MAX_STAGES],
    pipeline: u64,
    depth_stencil: u64,
    cull_mode: Option<CullMode>,
    /// VS slot 15 — programmable / FF vertex constant buffer.
    vs_constants: Vec<u8>,
    /// VS slot 13 — half-pixel rasterization fixup `(1/vp_w, -1/vp_h, 0, 0)`.
    ///
    /// Re-bound only when the viewport dims change (rare), so the per-draw
    /// cost is a length-then-memcmp against 16 bytes.
    vs_pos_fixup: Vec<u8>,
    /// PS slot 15 — programmable / FF pixel constant buffer.
    ps_constants: Vec<u8>,
    /// PS slot 14 — alpha-test reference float, when alpha test is enabled.
    ps_alpha_ref: Vec<u8>,
    /// PS slot 13 — fog colour vec4, when fog is enabled.
    ps_fog_color: Vec<u8>,
    /// PS slot 12 — per-stage bump-environment matrix.
    ///
    /// Set when the bound PS uses `texbem`/`texbeml`/`bem`.
    ps_bump_env: Vec<u8>,
    /// VS slot 0 — bound `MTLBuffer` handle + byte offset.
    ///
    /// `(0, _)` is the unset sentinel (Metal buffer handles are never
    /// zero).
    vertex_buffer: (u64, u32),
    /// Resolved `(x, y, w, h)` scissor rect.
    ///
    /// `None` is the unset sentinel — a brand-new render encoder has no
    /// scissor bound, so the first `emit_scissor` on a new pass must
    /// always go through.
    scissor_rect: Option<(u32, u32, u32, u32)>,
    /// `D3DRS_BLENDFACTOR` as a `D3DCOLOR` u32.
    ///
    /// `0xFFFF_FFFF` is the Metal default (opaque white) and the value
    /// at fresh-pass entry, so the per-draw conditional in `emit_draw`
    /// (which already skips default values) continues to skip the first
    /// default-value draw of each pass.
    blend_color: u32,
    /// `D3DRS_DEPTHBIAS` + `D3DRS_SLOPESCALEDEPTHBIAS`.
    ///
    /// Post the `d3d_depth_bias_to_metal` conversion + the
    /// implicit-decal-bias heuristic. Stored as raw bits so the
    /// comparison is exact (no NaN ambiguity) and the slot has a
    /// definite "not yet bound" sentinel — `(0, 0)` matches Metal's
    /// fresh-encoder default.
    depth_bias_bits: (u32, u32),
    /// Depth-clip mode: `true` = Clip (Metal's fresh-encoder default), `false` = Clamp.
    ///
    /// Driven per-draw by "depth test active" (ZENABLE + a depth
    /// attachment) — see `Command::set_depth_clip_mode`.
    depth_clip: bool,
}

impl LastBoundCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fragment_samplers: [0; LAST_BOUND_MAX_STAGES],
            fragment_textures: [0; LAST_BOUND_MAX_STAGES],
            pipeline: 0,
            depth_stencil: 0,
            cull_mode: None,
            vs_constants: Vec::new(),
            vs_pos_fixup: Vec::new(),
            ps_constants: Vec::new(),
            ps_alpha_ref: Vec::new(),
            ps_fog_color: Vec::new(),
            ps_bump_env: Vec::new(),
            vertex_buffer: (0, 0),
            scissor_rect: None,
            blend_color: 0xFFFF_FFFF,
            depth_bias_bits: (0, 0),
            depth_clip: true,
        }
    }

    /// Forget every binding.
    ///
    /// Call on new-pass entry — Metal resets state across `endEncoding`
    /// / fresh `renderCommandEncoder` boundaries. Byte-blob slots keep
    /// their backing allocation via `Vec::clear`, so steady-state passes
    /// don't reallocate.
    pub fn reset(&mut self) {
        self.fragment_samplers = [0; LAST_BOUND_MAX_STAGES];
        self.fragment_textures = [0; LAST_BOUND_MAX_STAGES];
        self.pipeline = 0;
        self.depth_stencil = 0;
        self.cull_mode = None;
        self.vs_constants.clear();
        self.vs_pos_fixup.clear();
        self.ps_constants.clear();
        self.ps_alpha_ref.clear();
        self.ps_fog_color.clear();
        self.ps_bump_env.clear();
        self.vertex_buffer = (0, 0);
        self.scissor_rect = None;
        self.blend_color = 0xFFFF_FFFF;
        self.depth_bias_bits = (0, 0);
        self.depth_clip = true;
    }

    #[inline]
    pub const fn fragment_sampler_changed(&mut self, stage: u32, handle: u64) -> bool {
        let slot = &mut self.fragment_samplers[stage as usize];
        if *slot == handle {
            false
        } else {
            *slot = handle;
            true
        }
    }

    #[inline]
    pub const fn fragment_texture_changed(&mut self, stage: u32, handle: u64) -> bool {
        let slot = &mut self.fragment_textures[stage as usize];
        if *slot == handle {
            false
        } else {
            *slot = handle;
            true
        }
    }

    #[inline]
    pub const fn pipeline_changed(&mut self, handle: u64) -> bool {
        if self.pipeline == handle {
            false
        } else {
            self.pipeline = handle;
            true
        }
    }

    #[inline]
    pub const fn depth_stencil_changed(&mut self, handle: u64) -> bool {
        if self.depth_stencil == handle {
            false
        } else {
            self.depth_stencil = handle;
            true
        }
    }

    #[inline]
    pub const fn cull_mode_changed(&mut self, mode: CullMode) -> bool {
        // `Option::eq` / `PartialEq` aren't const-stable for `Option<CullMode>`,
        // so destructure manually and compare via the `u32` repr.
        if let Some(prev) = self.cull_mode
            && prev as u32 == mode as u32
        {
            return false;
        }
        self.cull_mode = Some(mode);
        true
    }

    #[inline]
    pub const fn vertex_buffer_changed(&mut self, handle: u64, offset: u32) -> bool {
        if self.vertex_buffer.0 == handle && self.vertex_buffer.1 == offset {
            false
        } else {
            self.vertex_buffer = (handle, offset);
            true
        }
    }

    /// Forget the bound slot-0 vertex buffer.
    ///
    /// Forces the next `vertex_buffer_changed` to report a change. Call
    /// after binding slot 0 with inline bytes
    /// (`setVertexBytes(..., index 0)`): that clobbers the real Metal
    /// vertex-buffer binding while leaving this cache pointing at the
    /// previously bound buffer, so without this a following bound draw
    /// with the same `(handle, offset)` would skip its `setVertexBuffer`
    /// and read the inline payload as vertices. Resets to the `(0, _)`
    /// unset sentinel (Metal buffer handles are never zero).
    #[inline]
    pub const fn invalidate_vertex_buffer(&mut self) {
        self.vertex_buffer = (0, 0);
    }

    #[inline]
    pub const fn scissor_rect_changed(&mut self, rect: (u32, u32, u32, u32)) -> bool {
        // `PartialEq` on tuples isn't const-stable; destructure manually.
        if let Some(prev) = self.scissor_rect
            && prev.0 == rect.0
            && prev.1 == rect.1
            && prev.2 == rect.2
            && prev.3 == rect.3
        {
            return false;
        }
        self.scissor_rect = Some(rect);
        true
    }

    #[inline]
    pub const fn blend_color_changed(&mut self, d3dcolor: u32) -> bool {
        if self.blend_color == d3dcolor {
            false
        } else {
            self.blend_color = d3dcolor;
            true
        }
    }

    #[inline]
    pub const fn depth_bias_changed(&mut self, depth_bias: f32, slope_scale: f32) -> bool {
        let bits = (depth_bias.to_bits(), slope_scale.to_bits());
        if self.depth_bias_bits.0 == bits.0 && self.depth_bias_bits.1 == bits.1 {
            false
        } else {
            self.depth_bias_bits = bits;
            true
        }
    }

    #[inline]
    pub const fn depth_clip_changed(&mut self, clip: bool) -> bool {
        if self.depth_clip == clip {
            false
        } else {
            self.depth_clip = clip;
            true
        }
    }

    #[inline]
    pub fn vs_constants_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.vs_constants, bytes)
    }

    #[inline]
    pub fn vs_pos_fixup_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.vs_pos_fixup, bytes)
    }

    #[inline]
    pub fn ps_constants_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.ps_constants, bytes)
    }

    #[inline]
    pub fn ps_alpha_ref_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.ps_alpha_ref, bytes)
    }

    #[inline]
    pub fn ps_fog_color_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.ps_fog_color, bytes)
    }

    #[inline]
    pub fn ps_bump_env_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.ps_bump_env, bytes)
    }
}

impl Default for LastBoundCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` and updates `cache` iff `bytes` differs from `cache`.
///
/// `Vec<u8> == [u8]` is a length-then-memcmp; the update path retains the
/// Vec's capacity so the typical "constants change once, then stick" pattern
/// allocates exactly once per (slot, pass) pair.
fn update_inline_bytes(cache: &mut Vec<u8>, bytes: &[u8]) -> bool {
    if cache.as_slice() == bytes {
        false
    } else {
        cache.clear();
        cache.extend_from_slice(bytes);
        true
    }
}

/// Debug-only mirror of what was last emitted onto the current encoder.
///
/// Tracks each cache-covered slot whose `Command` embeds a directly comparable
/// value. Updated at the single command funnel (`PassState::emit_command`) and
/// diffed against [`LastBoundCache`] before every draw via
/// [`LastBoundCache::debug_assert_in_sync`].
///
/// A correct gated emit calls `<slot>_changed(v)` (advancing the cache to `v`)
/// immediately before pushing `set_<slot>(v)` (advancing this shadow to `v`),
/// so cache and shadow agree at every draw. A *bypass* — a `set_<slot>` pushed
/// without its `_changed` gate — advances the shadow while the cache stays
/// stale, and the next `debug_assert_in_sync` catches it. A clear-quad emitted
/// mid-pass is the usual source of such a bypass, since it binds pipeline /
/// depth-stencil / scissor / vertex-buffer state outside the per-draw gates.
///
/// `blend_color` (the command carries four `f32` lanes; the cache a packed
/// `D3DCOLOR`) and the four inline-bytes slots (the command carries a pointer +
/// length, not the bytes the cache holds) are deliberately not mirrored — they
/// are emitted solely from `emit_draw`, never a clear-quad, so the clear-quad
/// bypass surface (pipeline / depth-stencil / scissor / vertex buffer) stays
/// fully covered. Multi-field slots keep their command's *packed* `param_*`
/// form so decoding never needs a truncating cast; `debug_assert_in_sync`
/// re-packs the cache side with widening casts only.
#[cfg(debug_assertions)]
#[derive(Default)]
pub struct DebugBoundShadow {
    fragment_samplers: [u64; LAST_BOUND_MAX_STAGES],
    fragment_textures: [u64; LAST_BOUND_MAX_STAGES],
    pipeline: u64,
    depth_stencil: u64,
    /// Raw `CullMode` discriminant (`Command::param_a`).
    cull_mode: Option<u32>,
    /// `(handle, offset)`, `offset` kept as the command's `u64` param.
    vertex_buffer: (u64, u64),
    /// Raw `(param_a, param_b, param_c)` of `Command::set_scissor_rect`.
    scissor_rect: Option<(u32, u64, u64)>,
    /// Raw `(param_a, param_b)` of `Command::set_depth_bias`.
    depth_bias: (u32, u64),
}

#[cfg(debug_assertions)]
impl DebugBoundShadow {
    /// Mirror a just-pushed `Command` into its slot.
    ///
    /// Untracked command types (viewport, draws, blend color, fragment
    /// bytes, inline vertex bytes at a non-zero slot, visibility) are
    /// ignored.
    const fn record(&mut self, cmd: &Command) {
        let t = cmd.cmd;
        if t == CommandType::SetRenderPipelineState as u32 {
            self.pipeline = cmd.param_b;
        } else if t == CommandType::SetDepthStencilState as u32 {
            self.depth_stencil = cmd.param_b;
        } else if t == CommandType::SetCullMode as u32 {
            self.cull_mode = Some(cmd.param_a);
        } else if t == CommandType::SetFragmentTexture as u32 {
            self.fragment_textures[cmd.param_a as usize] = cmd.param_b;
        } else if t == CommandType::SetFragmentSamplerState as u32 {
            self.fragment_samplers[cmd.param_a as usize] = cmd.param_b;
        } else if t == CommandType::SetScissorRect as u32 {
            self.scissor_rect = Some((cmd.param_a, cmd.param_b, cmd.param_c));
        } else if t == CommandType::SetVertexBuffer as u32 {
            // The cache tracks slot 0 only.
            if cmd.param_a == 0 {
                self.vertex_buffer = (cmd.param_b, cmd.param_c);
            }
        } else if t == CommandType::SetDepthBias as u32 {
            self.depth_bias = (cmd.param_a, cmd.param_b);
        } else if (t == CommandType::SetVertexBytes as u32
            || t == CommandType::SetVertexBytesAt as u32)
            && cmd.param_a == 0
        {
            // Inline slot-0 bind clobbers the real Metal vertex buffer; mirror
            // `LastBoundCache::invalidate_vertex_buffer` so both forget it.
            self.vertex_buffer = (0, 0);
        }
    }
}

#[cfg(debug_assertions)]
impl LastBoundCache {
    /// Assert every mirrored slot matches what was actually emitted onto the encoder (`shadow`).
    ///
    /// Debug-build only; called before each draw from `FrameEncoder`.
    ///
    /// # Panics
    ///
    /// Panics on a cache↔encoder desync — a `set_*` that bypassed its
    /// `_changed` gate, or a gate that advanced the cache to a value the
    /// matching emit didn't carry. That panic is the guard doing its job.
    pub fn debug_assert_in_sync(&self, shadow: &DebugBoundShadow) {
        assert_eq!(
            self.pipeline, shadow.pipeline,
            "pipeline cache desync (cache vs encoder-emitted)"
        );
        assert_eq!(
            self.depth_stencil, shadow.depth_stencil,
            "depth-stencil cache desync (cache vs encoder-emitted)"
        );
        assert_eq!(
            self.cull_mode.map(|c| c as u32),
            shadow.cull_mode,
            "cull-mode cache desync (cache vs encoder-emitted)"
        );
        assert_eq!(
            (self.vertex_buffer.0, u64::from(self.vertex_buffer.1)),
            shadow.vertex_buffer,
            "vertex-buffer cache desync (cache vs encoder-emitted)"
        );
        assert_eq!(
            self.scissor_rect.map(|(x, y, w, h)| (
                x,
                u64::from(y),
                (u64::from(w) << 32) | u64::from(h)
            )),
            shadow.scissor_rect,
            "scissor cache desync (cache vs encoder-emitted)"
        );
        assert_eq!(
            (self.depth_bias_bits.0, u64::from(self.depth_bias_bits.1)),
            shadow.depth_bias,
            "depth-bias cache desync (cache vs encoder-emitted)"
        );
        for (stage, (&cache_h, &emitted_h)) in self
            .fragment_textures
            .iter()
            .zip(&shadow.fragment_textures)
            .enumerate()
        {
            assert_eq!(
                cache_h, emitted_h,
                "fragment-texture[{stage}] cache desync (cache vs encoder-emitted)"
            );
        }
        for (stage, (&cache_h, &emitted_h)) in self
            .fragment_samplers
            .iter()
            .zip(&shadow.fragment_samplers)
            .enumerate()
        {
            assert_eq!(
                cache_h, emitted_h,
                "fragment-sampler[{stage}] cache desync (cache vs encoder-emitted)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use mtld3d_shared::CommandType;

    use super::*;

    fn tex(raw: u64) -> MetalHandle<MTLTextureKind> {
        // SAFETY: tests; opaque values never dereferenced.
        unsafe { MetalHandle::new(raw) }
    }

    fn pso(raw: u64) -> MetalHandle<MTLRenderPipelineStateKind> {
        // SAFETY: tests; opaque values never dereferenced.
        unsafe { MetalHandle::new(raw) }
    }

    const BB_SIZE: (u32, u32) = (640, 480);
    const BB_FORMAT: PixelFormat = PixelFormat::Bgra8Unorm;
    const RT_FORMAT: PixelFormat = PixelFormat::Bgra8Unorm;

    fn backbuffer() -> MetalHandle<MTLTextureKind> {
        tex(0x1000)
    }
    fn depth() -> MetalHandle<MTLTextureKind> {
        tex(0x2000)
    }

    fn fresh() -> PassState {
        let mut s = PassState::new();
        s.reset_frame(backbuffer(), BB_SIZE, BB_FORMAT, depth(), false);
        s
    }

    fn dummy_draw() -> Command {
        // Any non-viewport command serves as a "draw" marker for bookkeeping
        // tests — the state machine only counts commands, not their kind.
        Command::draw_primitives(mtld3d_shared::mtl::PrimitiveType::Triangle, 0, 3)
    }

    fn unpack_scissor(cmd: &Command) -> (u32, u32, u32, u32) {
        assert_eq!(cmd.cmd, CommandType::SetScissorRect as u32);
        let x = cmd.param_a;
        // param_b/c are wire payload encoded in u64 — extract low/high u32 halves.
        let y = u32::try_from(cmd.param_b & 0xFFFF_FFFF).expect("low 32 bits fit u32");
        let w = u32::try_from(cmd.param_c >> 32).expect("high 32 bits fit u32");
        let h = u32::try_from(cmd.param_c & 0xFFFF_FFFF).expect("low 32 bits fit u32");
        (x, y, w, h)
    }

    #[test]
    fn frame_sampled_textures_tracks_fragment_binds_in_stream_order() {
        let mut s = fresh();
        let atlas = tex(0x7E10);
        // Not sampled before any draw emitted a bind — an upload landing
        // here must NOT rename (no earlier draw reads the old content).
        assert!(!s.texture_sampled_this_frame(atlas));
        s.emit_command(Command::set_fragment_texture(atlas.raw(), 0));
        assert!(s.texture_sampled_this_frame(atlas));
        // Unrelated handle stays unsampled (a renamed-fresh texture
        // relies on exactly this).
        assert!(!s.texture_sampled_this_frame(tex(0x7E20)));
    }

    #[test]
    fn frame_sampled_textures_clears_on_reset_frame() {
        let mut s = fresh();
        let atlas = tex(0x7E10);
        s.emit_command(Command::set_fragment_texture(atlas.raw(), 0));
        assert!(s.texture_sampled_this_frame(atlas));
        s.reset_frame(backbuffer(), BB_SIZE, BB_FORMAT, depth(), false);
        // Per-frame set resets — a next-frame upload before the first
        // sample goes to the live texture again.
        assert!(!s.texture_sampled_this_frame(atlas));
    }

    #[test]
    fn frame_sampled_textures_ignores_null_bind() {
        let mut s = fresh();
        s.emit_command(Command::set_fragment_texture(0, 0));
        assert!(!s.texture_sampled_this_frame(tex(0)));
    }

    #[test]
    fn inline_slot0_bind_forces_next_bound_vertex_buffer_reemit() {
        let mut cache = LastBoundCache::new();
        // First bind of a real VB handle reports a change and caches it.
        assert!(cache.vertex_buffer_changed(0xDEAD, 0));
        // A redundant rebind of the same (handle, offset) would be skipped.
        assert!(!cache.vertex_buffer_changed(0xDEAD, 0));
        // An inline slot-0 bind (setVertexBytes) clobbers the Metal binding;
        // invalidating the cache must force the next bound draw to re-emit
        // even though it targets the same (handle, offset).
        cache.invalidate_vertex_buffer();
        assert!(cache.vertex_buffer_changed(0xDEAD, 0));
    }

    #[test]
    fn begin_frame_starts_no_pass() {
        let s = fresh();
        assert!(s.passes().is_empty());
        assert!(s.current_pass_closed());
    }

    #[test]
    fn first_command_opens_pass() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        let pass = &s.passes()[0];
        assert_eq!(pass.color_texture(), backbuffer());
        assert_eq!(pass.depth_texture(), depth());
        assert_eq!(pass.viewport(), (0, 0, BB_SIZE.0, BB_SIZE.1));
        // Rule A: first use of the backbuffer + depth this frame, no
        // pending clear ⇒ DontCare. Prior contents are undefined per
        // D3D9 spec.
        assert_eq!(pass.color_load(), ColorLoad::DontCare);
        assert_eq!(pass.depth_load(), DepthLoad::DontCare);
        // First command is the implicit viewport, second is our draw.
        assert_eq!(pass.commands().len(), 2);
    }

    #[test]
    fn set_render_target_ends_pass_on_diff() {
        let rt = tex(0x3000);
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].color_texture(), backbuffer());
        assert_eq!(s.passes()[1].color_texture(), rt);
        // With no explicit viewport set, the new pass falls back to the new
        // attachment size — matches D3D9 semantics where SetRenderTarget
        // implicitly resizes the viewport to the new target.
        assert_eq!(s.passes()[1].viewport(), (0, 0, 256, 256));
    }

    #[test]
    fn set_render_target_same_handle_no_break() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
    }

    #[test]
    fn mid_pass_color_clear_returns_emit_quad_outcome() {
        // D3D9's `Clear` is viewport-clipped and can fire mid-render.
        // Metal has no in-encoder Clear primitive, so a mid-pass Clear
        // returns `ColorClearOutcome::EmitQuad` so the encoder layer
        // (which owns the clear-quad pipeline cache) can emit a
        // scissored fullscreen-triangle draw. The pass does NOT break:
        // breaking on Clear would open a new encoder with
        // `loadAction = Clear` which wipes the full attachment under
        // Metal's full-attachment Clear semantics, deleting all prior
        // tile draws (the failure mode for sub-rect Clears into a
        // shared shadow/tile atlas).
        let mut s = fresh();
        s.emit_command(dummy_draw());
        let outcome = s.clear_color(1, 2, 3, 4);
        s.emit_command(dummy_draw());
        assert!(matches!(outcome, ColorClearOutcome::EmitQuad { .. }));
        assert_eq!(
            s.passes().len(),
            1,
            "pass should not break on mid-pass Clear; encoder emits a clear-quad inline"
        );
    }

    #[test]
    fn clear_before_any_draw_merges_into_first_pass() {
        let mut s = fresh();
        s.clear_color(5, 6, 7, 8);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        assert_eq!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 5,
                g: 6,
                b: 7,
                a: 8
            }
        );
    }

    #[test]
    fn clear_amends_empty_pass_in_place() {
        // Pass open with only the viewport command → Clear amends the load
        // action directly instead of ending the pass.
        let mut s = fresh();
        s.ensure_pass_open();
        assert_eq!(s.passes().len(), 1);
        s.clear_color(9, 9, 9, 9);
        assert_eq!(s.passes().len(), 1, "empty pass should not be broken");
        assert_eq!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 9,
                g: 9,
                b: 9,
                a: 9
            }
        );
    }

    #[test]
    fn depth_change_triggers_pass_break() {
        let other_depth = tex(0x4000);
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_depth_stencil_attachment(other_depth, false, false);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].depth_texture(), depth());
        assert_eq!(s.passes()[1].depth_texture(), other_depth);
    }

    #[test]
    fn viewport_applied_to_new_pass_start() {
        let rt = tex(0x3000);
        let mut s = fresh();
        s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 128, 128, RT_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        // Both passes use the 320x240 viewport (sticky). The first command
        // of each pass is the viewport set.
        assert_eq!(s.passes()[0].viewport(), (0, 0, 320, 240));
        assert_eq!(s.passes()[1].viewport(), (0, 0, 320, 240));
    }

    #[test]
    fn first_use_each_rt_is_dontcare() {
        let rt = tex(0x3000);
        // Rule A — every rt's first use in a frame, with no pending
        // clear, gets DontCare. Backbuffer is first-use in pass A;
        // rt is first-use in pass B. Depth is shared so it's
        // first-use in A and re-use (Load) in B.
        let mut s = fresh();
        s.emit_command(dummy_draw()); // pass A on backbuffer() + depth()
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw()); // pass B on rt + depth()
        assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
        assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
        assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
        // depth() is seen-already (pass A used it), so Load this time.
        assert_eq!(s.passes()[1].depth_load(), DepthLoad::Load);
    }

    #[test]
    fn mid_pass_depth_clear_returns_emit_quad_outcome() {
        // Depth mirror of `mid_pass_color_clear_returns_emit_quad_outcome`.
        let mut s = fresh();
        s.emit_command(dummy_draw());
        let z = f32::to_bits(0.5);
        let outcome = s.clear_depth(z);
        s.emit_command(dummy_draw());
        assert!(matches!(outcome, DepthClearOutcome::EmitQuad { value, .. } if value == z));
        assert_eq!(s.passes().len(), 1);
    }

    #[test]
    fn reset_frame_drops_pending_clears() {
        let mut s = fresh();
        s.clear_color(1, 2, 3, 4);
        assert!(s.pending_color_clear().is_some());
        s.reset_frame(backbuffer(), BB_SIZE, BB_FORMAT, depth(), false);
        assert!(s.pending_color_clear().is_none());
        assert!(s.passes().is_empty());
    }

    #[test]
    fn clear_then_rt_switch_materializes_old_target() {
        let rt = tex(0x3000);
        // D3D9 semantic: Clear applies to the bound rt at call time. If the
        // game clears the rt and switches target without drawing, the old
        // rt must still receive the clear.
        let mut s = fresh();
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.clear_color(1, 2, 3, 4);
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        // Old rt got the clear
        assert_eq!(s.passes()[0].color_texture(), rt);
        assert_eq!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 1,
                g: 2,
                b: 3,
                a: 4
            }
        );
        // Backbuffer pass is first-use this frame (the synthesised
        // clear pass ran on rt, not on backbuffer()), no pending clear ⇒
        // Rule A flips to DontCare.
        assert_eq!(s.passes()[1].color_texture(), backbuffer());
        assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
    }

    #[test]
    fn flush_pending_clears_is_noop_when_empty() {
        let mut s = fresh();
        s.flush_pending_clears();
        assert!(s.passes().is_empty());
    }

    #[test]
    fn flush_pending_clears_materializes_pass() {
        let mut s = fresh();
        s.clear_color(7, 8, 9, 10);
        s.flush_pending_clears();
        assert_eq!(s.passes().len(), 1);
        assert_eq!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 7,
                g: 8,
                b: 9,
                a: 10
            }
        );
        // Pass is closed so a subsequent draw opens a new pass.
        assert!(s.current_pass_closed());
    }

    #[test]
    fn multiple_rt_swaps_produce_multiple_passes() {
        let rt = tex(0x3000);
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 3);
        assert_eq!(s.passes()[0].color_texture(), backbuffer());
        assert_eq!(s.passes()[1].color_texture(), rt);
        assert_eq!(s.passes()[2].color_texture(), backbuffer());
    }

    #[test]
    fn color_format_propagates_per_pass() {
        const OTHER_FORMAT: PixelFormat = PixelFormat::Rgba16Float;
        let rt = tex(0x3000);
        // Format the pass opens with is what was current at pass-open
        // time. Pipelines created during each pass key on this value,
        // so distinct rt formats must yield distinct Pass.color_format.
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, OTHER_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes()[0].color_format(), BB_FORMAT);
        assert_eq!(s.passes()[1].color_format(), OTHER_FORMAT);
        assert_eq!(s.current_color_format(), OTHER_FORMAT);
    }

    #[test]
    fn emit_scissor_enabled_uses_game_rect() {
        let mut s = fresh();
        s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
        s.emit_scissor(true, [10, 20, 200, 150]);
        let cmds = s.passes()[0].commands();
        // [0] = implicit viewport, [1] = our scissor
        assert_eq!(unpack_scissor(&cmds[1]), (10, 20, 200, 150));
    }

    #[test]
    fn emit_scissor_disabled_falls_back_to_viewport() {
        let mut s = fresh();
        s.set_viewport(5, 7, 320, 240, 0.0, 1.0);
        // test_enable = false → stored rect ignored, viewport used
        s.emit_scissor(false, [10, 20, 200, 150]);
        let cmds = s.passes()[0].commands();
        assert_eq!(unpack_scissor(&cmds[1]), (5, 7, 320, 240));
    }

    #[test]
    fn emit_scissor_zero_rect_falls_back_to_viewport() {
        let mut s = fresh();
        s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
        // SetScissorRect was never called (scissor_rect = [0; 4]) but the
        // game turned the test on anyway → fall back to viewport so Metal
        // doesn't clip to an empty rect.
        s.emit_scissor(true, [0, 0, 0, 0]);
        let cmds = s.passes()[0].commands();
        assert_eq!(unpack_scissor(&cmds[1]), (0, 0, 640, 480));
    }

    #[test]
    fn emit_scissor_reemit_updates_per_draw() {
        // Our architecture re-emits scissor every draw (no dirty
        // tracking). Two draws with different states produce two commands.
        let mut s = fresh();
        s.set_viewport(0, 0, 640, 480, 0.0, 1.0);
        s.emit_scissor(true, [10, 20, 200, 150]);
        s.emit_scissor(false, [0, 0, 0, 0]);
        let cmds = s.passes()[0].commands();
        // [0] viewport, [1] first scissor, [2] second scissor
        assert_eq!(unpack_scissor(&cmds[1]), (10, 20, 200, 150));
        assert_eq!(unpack_scissor(&cmds[2]), (0, 0, 640, 480));
    }

    #[test]
    fn emit_scissor_without_viewport_uses_rt_size() {
        // No SetViewport call → PassState falls back to the color-size
        // fallback at pass-open (also used as viewport fallback).
        let mut s = fresh();
        s.emit_scissor(false, [10, 20, 30, 40]);
        let cmds = s.passes()[0].commands();
        assert_eq!(unpack_scissor(&cmds[1]), (0, 0, BB_SIZE.0, BB_SIZE.1));
    }

    #[test]
    fn set_viewport_dedups_redundant_reemit_within_pass() {
        let mut s = fresh();
        // Opens the pass; the pass-open viewport (RT-size fallback,
        // depth range 0..1) is the first command.
        s.emit_command(dummy_draw());
        let n0 = s.passes()[0].commands().len();
        // Re-setting the value the pass already opened with is a no-op —
        // re-emitting it is the Xcode "already bound" redundant bind.
        s.set_viewport(0, 0, BB_SIZE.0, BB_SIZE.1, 0.0, 1.0);
        assert_eq!(
            s.passes()[0].commands().len(),
            n0,
            "redundant viewport must not re-emit",
        );
        // A genuine x/y/w/h change re-emits once.
        s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
        assert_eq!(s.passes()[0].commands().len(), n0 + 1);
        // Re-setting that same value is again a no-op.
        s.set_viewport(0, 0, 320, 240, 0.0, 1.0);
        assert_eq!(s.passes()[0].commands().len(), n0 + 1);
        // A depth-range-only change (same x/y/w/h) must still re-emit —
        // the z-range is part of the bind.
        s.set_viewport(0, 0, 320, 240, 0.0, 0.5);
        assert_eq!(
            s.passes()[0].commands().len(),
            n0 + 2,
            "depth-range-only change must re-emit",
        );
    }

    fn dummy_blit() -> BlitCommand {
        BlitCommand::copy_texture_to_texture_full_mip(0xAA, 0xBB, 0, 64, 64)
    }

    #[test]
    fn pending_blit_drains_into_next_pass() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.push_pending_leading_blit(dummy_blit());
        // Next pass open inherits the queued blit.
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].leading_blits().len(), 0);
        assert_eq!(s.passes()[1].leading_blits().len(), 1);
        // Pending queue is empty after the drain.
        let mut s2 = s;
        assert!(s2.take_pending_leading_blits().is_empty());
    }

    #[test]
    fn trailing_pending_blit_survives_via_take() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.push_pending_leading_blit(dummy_blit());
        // No follow-up draw — pending blit stays in the queue for
        // `submit` to drain into a synthetic blit-only pass.
        let trailing = s.take_pending_leading_blits();
        assert_eq!(trailing.len(), 1);
    }

    #[test]
    fn fresh_pass_has_no_counting_visibility() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        assert!(!s.passes()[0].has_counting_visibility());
    }

    #[test]
    fn counting_visibility_latches_flag() {
        let mut s = fresh();
        s.emit_command(Command::set_visibility_result_mode(
            VisibilityResultMode::Counting,
            0,
        ));
        assert!(s.passes()[0].has_counting_visibility());
    }

    #[test]
    fn disabled_only_does_not_flip_flag() {
        // End-of-query tail: the encoder emits `Disabled` with
        // `active_count == 0`. No counter is written in this pass, so
        // the buffer must not be attached.
        let mut s = fresh();
        s.emit_command(Command::set_visibility_result_mode(
            VisibilityResultMode::Disabled,
            0,
        ));
        assert!(!s.passes()[0].has_counting_visibility());
    }

    #[test]
    fn non_visibility_commands_do_not_flip_flag() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.emit_command(Command::set_cull_mode(mtld3d_shared::mtl::CullMode::None));
        assert!(!s.passes()[0].has_counting_visibility());
    }

    #[test]
    fn counting_then_disabled_stays_latched() {
        // BEGIN then END within one pass: Counting arms, Disabled
        // closes — the counter was written, so the flag must stay set
        // for the submit path to keep the buffer attached.
        let mut s = fresh();
        s.emit_command(Command::set_visibility_result_mode(
            VisibilityResultMode::Counting,
            0,
        ));
        s.emit_command(Command::set_visibility_result_mode(
            VisibilityResultMode::Disabled,
            8,
        ));
        assert!(s.passes()[0].has_counting_visibility());
    }

    #[test]
    fn pass_break_clears_flag_for_new_pass() {
        let rt = tex(0x3000);
        // A Counting pass followed by a rendertarget switch must not
        // bleed the flag into the next pass — each pass tracks its
        // own attachments independently.
        let mut s = fresh();
        s.emit_command(Command::set_visibility_result_mode(
            VisibilityResultMode::Counting,
            0,
        ));
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        assert!(s.passes()[0].has_counting_visibility());
        assert!(!s.passes()[1].has_counting_visibility());
    }

    // ── LastBoundCache ──

    #[test]
    fn last_bound_first_call_reports_changed() {
        let mut c = LastBoundCache::new();
        assert!(c.fragment_sampler_changed(0, 0xAAAA));
        assert!(c.fragment_texture_changed(0, 0xBBBB));
        assert!(c.pipeline_changed(0xCCCC));
        assert!(c.depth_stencil_changed(0xDDDD));
        assert!(c.cull_mode_changed(CullMode::Back));
    }

    #[test]
    fn last_bound_repeat_value_is_unchanged() {
        let mut c = LastBoundCache::new();
        c.fragment_sampler_changed(2, 0xAAAA);
        assert!(!c.fragment_sampler_changed(2, 0xAAAA));
        c.fragment_texture_changed(2, 0xBBBB);
        assert!(!c.fragment_texture_changed(2, 0xBBBB));
        c.pipeline_changed(0xCCCC);
        assert!(!c.pipeline_changed(0xCCCC));
        c.depth_stencil_changed(0xDDDD);
        assert!(!c.depth_stencil_changed(0xDDDD));
        c.cull_mode_changed(CullMode::Front);
        assert!(!c.cull_mode_changed(CullMode::Front));
    }

    #[test]
    fn last_bound_different_value_is_changed() {
        let mut c = LastBoundCache::new();
        c.fragment_sampler_changed(0, 0xAAAA);
        assert!(c.fragment_sampler_changed(0, 0xBEEF));
        c.cull_mode_changed(CullMode::None);
        assert!(c.cull_mode_changed(CullMode::Back));
    }

    #[test]
    fn last_bound_cull_none_distinct_from_unset() {
        // CullMode::None is value 0, but a freshly-reset cache must still
        // report "changed" on the first call — otherwise the first draw of
        // a pass that wants None cull would silently inherit whatever
        // Metal's default is. The Option<CullMode> sentinel guards this.
        let mut c = LastBoundCache::new();
        assert!(c.cull_mode_changed(CullMode::None));
        assert!(!c.cull_mode_changed(CullMode::None));
    }

    #[test]
    fn last_bound_stages_are_independent() {
        let mut c = LastBoundCache::new();
        c.fragment_sampler_changed(3, 0xAAAA);
        c.fragment_texture_changed(3, 0xBBBB);
        assert!(c.fragment_sampler_changed(4, 0xAAAA));
        assert!(c.fragment_texture_changed(4, 0xBBBB));
        assert!(!c.fragment_sampler_changed(3, 0xAAAA));
        assert!(!c.fragment_texture_changed(3, 0xBBBB));
    }

    #[test]
    fn last_bound_reset_clears_everything() {
        let mut c = LastBoundCache::new();
        c.fragment_sampler_changed(0, 0xAAAA);
        c.fragment_texture_changed(1, 0xBBBB);
        c.pipeline_changed(0xCCCC);
        c.depth_stencil_changed(0xDDDD);
        c.cull_mode_changed(CullMode::Back);
        c.vs_constants_changed(&[1, 2, 3, 4]);
        c.ps_constants_changed(&[5, 6, 7, 8]);
        c.ps_alpha_ref_changed(&[9, 10, 11, 12]);
        c.ps_fog_color_changed(&[13, 14, 15, 16]);
        c.vertex_buffer_changed(0xEEEE, 32);
        c.scissor_rect_changed((1, 2, 3, 4));
        c.blend_color_changed(0xFF11_2233);
        c.reset();
        assert!(c.fragment_sampler_changed(0, 0xAAAA));
        assert!(c.fragment_texture_changed(1, 0xBBBB));
        assert!(c.pipeline_changed(0xCCCC));
        assert!(c.depth_stencil_changed(0xDDDD));
        assert!(c.cull_mode_changed(CullMode::Back));
        assert!(c.vs_constants_changed(&[1, 2, 3, 4]));
        assert!(c.ps_constants_changed(&[5, 6, 7, 8]));
        assert!(c.ps_alpha_ref_changed(&[9, 10, 11, 12]));
        assert!(c.ps_fog_color_changed(&[13, 14, 15, 16]));
        assert!(c.vertex_buffer_changed(0xEEEE, 32));
        assert!(c.scissor_rect_changed((1, 2, 3, 4)));
        assert!(c.blend_color_changed(0xFF11_2233));
    }

    #[test]
    fn last_bound_inline_bytes_dedup() {
        let mut c = LastBoundCache::new();
        assert!(c.ps_constants_changed(&[1, 2, 3, 4]));
        assert!(!c.ps_constants_changed(&[1, 2, 3, 4]));
        assert!(c.ps_constants_changed(&[1, 2, 3, 5]));
        assert!(c.ps_constants_changed(&[1, 2, 3])); // length change
        assert!(!c.ps_constants_changed(&[1, 2, 3]));
    }

    #[test]
    fn last_bound_inline_bytes_slots_are_independent() {
        let mut c = LastBoundCache::new();
        c.vs_constants_changed(&[1; 16]);
        c.ps_constants_changed(&[2; 16]);
        c.ps_alpha_ref_changed(&[3; 4]);
        c.ps_fog_color_changed(&[4; 16]);
        // Identical content in a different slot must still report changed
        // (slot 13 hasn't seen this payload yet).
        assert!(!c.vs_constants_changed(&[1; 16]));
        assert!(!c.ps_constants_changed(&[2; 16]));
        assert!(!c.ps_alpha_ref_changed(&[3; 4]));
        assert!(!c.ps_fog_color_changed(&[4; 16]));
    }

    #[test]
    fn last_bound_inline_bytes_reset_keeps_capacity() {
        let mut c = LastBoundCache::new();
        c.ps_constants_changed(&[0xAB; 256]);
        let cap_before = c.ps_constants.capacity();
        c.reset();
        assert_eq!(c.ps_constants.len(), 0);
        assert_eq!(c.ps_constants.capacity(), cap_before);
    }

    #[test]
    fn last_bound_vertex_buffer_dedup() {
        let mut c = LastBoundCache::new();
        assert!(c.vertex_buffer_changed(0xAAAA, 0));
        assert!(!c.vertex_buffer_changed(0xAAAA, 0));
        // Same handle, different offset → changed.
        assert!(c.vertex_buffer_changed(0xAAAA, 64));
        // Same offset, different handle → changed.
        assert!(c.vertex_buffer_changed(0xBBBB, 64));
    }

    #[test]
    fn last_bound_scissor_dedup() {
        let mut c = LastBoundCache::new();
        // First emit always goes through — fresh encoder has no scissor.
        assert!(c.scissor_rect_changed((0, 0, 640, 480)));
        assert!(!c.scissor_rect_changed((0, 0, 640, 480)));
        // Any tuple field different → changed.
        assert!(c.scissor_rect_changed((10, 0, 640, 480)));
        assert!(c.scissor_rect_changed((10, 20, 640, 480)));
        assert!(c.scissor_rect_changed((10, 20, 700, 480)));
        assert!(c.scissor_rect_changed((10, 20, 700, 500)));
        // Reset → first emit goes through again even with the same rect.
        let rect = (10, 20, 700, 500);
        assert!(!c.scissor_rect_changed(rect));
        c.reset();
        assert!(c.scissor_rect_changed(rect));
    }

    #[test]
    fn last_bound_blend_color_dedup() {
        let mut c = LastBoundCache::new();
        // Default `0xFFFF_FFFF` is the post-reset value; matches Metal's
        // own default, so a first-call with the default reports unchanged.
        assert!(!c.blend_color_changed(0xFFFF_FFFF));
        assert!(c.blend_color_changed(0xFF80_2040));
        assert!(!c.blend_color_changed(0xFF80_2040));
        assert!(c.blend_color_changed(0xFFFF_FFFF));
    }

    #[test]
    fn clear_quad_pipeline_change_forces_caster_reemit() {
        // The CSM cascade-atlas shadow-flicker case: all four cascades
        // render in one pass, each preceded by a mid-pass depth clear-quad. If
        // the clear-quad routes its own pipeline/DSS through the cache (as it
        // must), a later caster with the SAME pipeline/DSS as a prior cascade
        // is forced to re-emit — it does not stale-skip and inherit the
        // clear-quad's always-compare depth state.
        let mut c = LastBoundCache::new();
        let (p_caster, d_caster) = (0xCA57, 0x0D55);
        let (p_clear, d_clear) = (0xC1EA, 0xC1DD);
        // Cascade 0 caster binds its pipeline + depth-stencil.
        assert!(c.pipeline_changed(p_caster));
        assert!(c.depth_stencil_changed(d_caster));
        // Mid-pass clear-quad advances the cache to its own state.
        assert!(c.pipeline_changed(p_clear));
        assert!(c.depth_stencil_changed(d_clear));
        // Cascade 1 caster: identical to cascade 0 → must re-emit, not skip.
        assert!(c.pipeline_changed(p_caster));
        assert!(c.depth_stencil_changed(d_caster));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "pipeline cache desync")]
    fn debug_guard_catches_pipeline_bypass() {
        // Reproduce the clear-quad bug class: a pipeline emitted DIRECTLY onto
        // the encoder without going through `pipeline_changed`. The shadow
        // records the emit; the cache stays stale; the in-sync check fires.
        let cache = LastBoundCache::new();
        let mut shadow = DebugBoundShadow::default();
        shadow.record(&Command::set_render_pipeline_state(0xDEAD_BEEF));
        cache.debug_assert_in_sync(&shadow);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_guard_in_sync_across_every_tracked_slot() {
        // Each tracked slot driven through the real gate→emit→shadow cycle must
        // leave cache and shadow agreeing — proving the decode/re-pack round
        // trips (scissor packing, depth-bias bits, vertex-buffer offset
        // widening, cull discriminant) and that the guard is free of false
        // positives on correct usage. A fresh cache reports every first bind as
        // changed, so each gate must return `true`.
        let mut cache = LastBoundCache::new();
        let mut shadow = DebugBoundShadow::default();

        assert!(cache.pipeline_changed(0x9001));
        shadow.record(&Command::set_render_pipeline_state(0x9001));
        assert!(cache.depth_stencil_changed(0x9002));
        shadow.record(&Command::set_depth_stencil_state(0x9002));
        assert!(cache.cull_mode_changed(CullMode::Back));
        shadow.record(&Command::set_cull_mode(CullMode::Back));
        assert!(cache.fragment_texture_changed(3, 0x7E10));
        shadow.record(&Command::set_fragment_texture(0x7E10, 3));
        assert!(cache.fragment_sampler_changed(3, 0x5A77));
        shadow.record(&Command::set_fragment_sampler_state(0x5A77, 3));
        assert!(cache.vertex_buffer_changed(0xBEEF, 0x40));
        shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x40, 0));
        assert!(cache.scissor_rect_changed((7, 9, 1024, 768)));
        shadow.record(&Command::set_scissor_rect(7, 9, 1024, 768));
        assert!(cache.depth_bias_changed(-1e-4, -1.5));
        shadow.record(&Command::set_depth_bias(-1e-4, -1.5));

        // Every gate fired and recorded its matching emit: cache == encoder.
        cache.debug_assert_in_sync(&shadow);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_guard_in_sync_after_inline_slot0_invalidate() {
        // An inline slot-0 bind (UP geometry / clear-quad) clobbers the real
        // vertex buffer. The cache invalidates; the shadow must mirror that or
        // the next in-sync check false-positives.
        let mut cache = LastBoundCache::new();
        let mut shadow = DebugBoundShadow::default();

        assert!(cache.vertex_buffer_changed(0xBEEF, 0x10));
        shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
        cache.debug_assert_in_sync(&shadow);

        // Inline slot-0 bind, then invalidate — the encoder's emit order.
        shadow.record(&Command::set_vertex_bytes_at(0xA000, 4, 0));
        cache.invalidate_vertex_buffer();
        cache.debug_assert_in_sync(&shadow);

        // The next bound draw re-binds the same buffer and stays in sync.
        assert!(cache.vertex_buffer_changed(0xBEEF, 0x10));
        shadow.record(&Command::set_vertex_buffer(0xBEEF, 0x10, 0));
        cache.debug_assert_in_sync(&shadow);
    }

    // ── Rule A: first-use LoadAction::DontCare ────────────────────

    #[test]
    fn rule_a_fresh_frame_clear_color_only_keeps_clear() {
        // Pending color clear must beat the first-use DontCare branch;
        // depth still falls through to DontCare since no depth clear
        // was issued.
        let mut s = fresh();
        s.clear_color(1, 2, 3, 4);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        assert_eq!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 1,
                g: 2,
                b: 3,
                a: 4
            }
        );
        assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    }

    #[test]
    fn rule_a_same_rt_after_pass_break_is_load() {
        let rt = tex(0x3000);
        // backbuffer() → rt → backbuffer(): the third pass re-uses
        // backbuffer(), which was already seen in pass 0, so it gets
        // Load (Rule A would let DontCare slip otherwise).
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 3);
        assert_eq!(s.passes()[2].color_texture(), backbuffer());
        assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
    }

    #[test]
    fn rule_a_reset_frame_re_arms_dontcare() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
        // Next frame: same backbuffer is "first use again" because the
        // seen set was reset.
        s.reset_frame(backbuffer(), BB_SIZE, BB_FORMAT, depth(), false);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].color_load(), ColorLoad::DontCare);
        assert_eq!(s.passes()[0].depth_load(), DepthLoad::DontCare);
    }

    #[test]
    fn rule_a_leading_blit_to_rt_forces_load() {
        let rt_src = tex(0x3000);
        let rt_dst = tex(0x4000);
        // StretchRect lands between two passes and writes to the
        // pass's destination rt. The blit's output must survive into
        // the pass — first-use DontCare would discard it.
        let mut s = fresh();
        // First pass on backbuffer() so rt_dst is first-use when it opens.
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt_dst, 256, 256, RT_FORMAT);
        s.push_pending_leading_blit(BlitCommand::copy_texture_to_texture_full_mip(
            rt_src.raw(),
            rt_dst.raw(),
            0,
            256,
            256,
        ));
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[1].color_texture(), rt_dst);
        assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
    }

    // ── Rule B: last-use depth/stencil StoreAction::DontCare ──────

    #[test]
    fn rule_b_single_pass_depth_store_is_dontcare() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].depth_store(), StoreAction::DontCare);
        // Color store is left as Store — the HDR present pass or next
        // frame's reads still need it.
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_b_three_passes_same_depth_only_last_is_dontcare() {
        let rt_a = tex(0x3000);
        let rt_b = tex(0x4000);
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt_a, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt_b, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 3);
        // All three share depth(); only the last pass's depth_store is DontCare.
        assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
    }

    #[test]
    fn rule_b_alternating_depth_each_gets_last_use_dontcare() {
        let d1 = depth();
        let d2 = tex(0x9000);
        // Two depth textures alternating: d1, d2, d1, d2. Last d1 is
        // pass 2; last d2 is pass 3. Both should be DontCare; the
        // earlier passes (0, 1) keep Store.
        let mut s = fresh();
        // Pass 0: backbuffer() + d1
        s.emit_command(dummy_draw());
        // Pass 1: backbuffer() + d2
        s.set_depth_stencil_attachment(d2, false, false);
        s.emit_command(dummy_draw());
        // Pass 2: backbuffer() + d1
        s.set_depth_stencil_attachment(d1, false, false);
        s.emit_command(dummy_draw());
        // Pass 3: backbuffer() + d2
        s.set_depth_stencil_attachment(d2, false, false);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 4);
        assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
        assert_eq!(s.passes()[3].depth_store(), StoreAction::DontCare);
    }

    // ── Rule C: next-pass-clear color StoreAction::DontCare ───────

    #[test]
    fn rule_c_single_pass_color_store_is_store() {
        // No "next pass" → final pass's color contents must survive
        // (backbuffer Present and persistent RTs read them).
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_c_distinct_rts_no_next_use_keep_store() {
        let rt = tex(0x3000);
        // Pass 0 backbuffer(), pass 1 rt, pass 2 backbuffer() — neither rt
        // is followed by another pass with the SAME color rt (backbuffer()'s
        // re-use at pass 2 has color_load=Load, not Clear). Rule C does
        // not fire for any pass here. Rule D, however, flips pass 1's
        // rt (non-backbuffer, last use, not sampled) to DontCare.
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
        assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
        assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_c_next_pass_clears_same_rt_flips_store() {
        let rt = tex(0x3000);
        // Pass 0 backbuffer(), pass 1 rt with clear, pass 2 backbuffer() with
        // clear. backbuffer() pass 0's next use is pass 2 (clear) → Rule C
        // flips. rt pass 1 has no next use → Rule C keeps Store, then
        // Rule D (non-backbuffer last-use) flips to DontCare.
        // backbuffer() pass 2 is the last pass for backbuffer() → exempt
        // from Rule D, keeps Store (Present reads it).
        let mut s = fresh();
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.clear_color(1, 2, 3, 4);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.clear_color(5, 6, 7, 8);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 3);
        assert_eq!(s.passes()[0].color_texture(), backbuffer());
        assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
        assert_eq!(s.passes()[1].color_texture(), rt);
        assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
        assert_eq!(s.passes()[2].color_texture(), backbuffer());
        assert_eq!(s.passes()[2].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_c_next_pass_loads_same_rt_keeps_store() {
        let rt = tex(0x3000);
        // Pass 0 backbuffer(), pass 1 backbuffer() (no clear → Load). Pass 0's
        // contents must survive — pass 1 reads them via Load.
        let mut s = fresh();
        s.emit_command(dummy_draw());
        // Force a pass break with no pending clear: bounce rt then back.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 3);
        assert_eq!(s.passes()[0].color_texture(), backbuffer());
        assert_eq!(s.passes()[2].color_texture(), backbuffer());
        // Pass 2's color_load is Load (no clear was queued), so pass 0
        // MUST keep its store.
        assert_eq!(s.passes()[2].color_load(), ColorLoad::Load);
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_c_csm_cluster_intra_frame_stores_drop() {
        let rt_a = tex(0x3000);
        let rt_b = tex(0x4000);
        // WoW CSM-style frame shape: scene-on-backbuffer → rt_a clear →
        // rt_b clear → rt_a clear → UI-on-backbuffer (loads). Each
        // intra-frame cascade pass's color store is redundant because
        // the next pass touching the same rt begins with Clear. The
        // first backbuffer() pass's store stays because the UI pass loads.
        let mut s = fresh();
        // Pass 0: scene on backbuffer()
        s.emit_command(dummy_draw());
        // Pass 1: rt_a cascade A1, cleared on entry
        s.set_color_render_target(rt_a, 1024, 512, RT_FORMAT);
        s.clear_color(0, 0, 0, 0);
        s.emit_command(dummy_draw());
        // Pass 2: rt_b cascade B1, cleared on entry
        s.set_color_render_target(rt_b, 1024, 512, RT_FORMAT);
        s.clear_color(0, 0, 0, 0);
        s.emit_command(dummy_draw());
        // Pass 3: rt_a cascade A2, cleared on entry
        s.set_color_render_target(rt_a, 1024, 512, RT_FORMAT);
        s.clear_color(0, 0, 0, 0);
        s.emit_command(dummy_draw());
        // Pass 4: UI on backbuffer() (loads scene)
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 5);
        // Pass 0 backbuffer() → next backbuffer() use is pass 4, which Loads → keep Store.
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
        // Pass 1 rt_a → next rt_a use is pass 3 Clear → Rule C flips.
        assert_eq!(s.passes()[1].color_store(), StoreAction::DontCare);
        // Pass 2 rt_b → no next rt_b use → Rule D (non-backbuffer last-use)
        // flips to DontCare since rt_b is never sampled.
        assert_eq!(s.passes()[2].color_store(), StoreAction::DontCare);
        // Pass 3 rt_a → no next rt_a use → Rule D flips.
        assert_eq!(s.passes()[3].color_store(), StoreAction::DontCare);
        // Pass 4 backbuffer() → last in frame, exempt from Rule D (Present reads it).
        assert_eq!(s.passes()[4].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_c_color_walk_independent_of_depth_walk() {
        // Two passes share backbuffer() + depth(), second pass starts with a
        // color clear. Rule C flips pass 0's color store, Rule B flips
        // pass 1's depth store, and pass 0's depth keeps Store (Rule B
        // only flips the LAST pass per depth texture).
        let rt = tex(0x3000);
        let mut s = fresh();
        s.emit_command(dummy_draw());
        // Force a pass break with a pending color clear on backbuffer().
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.clear_color(1, 1, 1, 1);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 3);
        // Pass 0 backbuffer() → next backbuffer() is pass 2 Clear → flip color.
        assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
        // All three share depth(); only pass 2's depth_store flips.
        assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[1].depth_store(), StoreAction::Store);
        assert_eq!(s.passes()[2].depth_store(), StoreAction::DontCare);
    }

    // ── Sampler-aware exemptions (CSM sampling) ──────────────────────

    #[test]
    fn rule_b_keeps_store_when_depth_sampled_later() {
        let cascade_depth = tex(0x9000);
        let cascade_color = tex(0x3000);
        // Cascade sampling: cascade depth is written in pass 0, sampled
        // by the scene PS in pass 1. Rule B must NOT flip pass 0's
        // depth_store to DontCare or the scene's `sample_compare` reads
        // tile memory that was never preserved to VRAM.
        let mut s = fresh();
        // Pass 0: cascade caster pass — write into cascade_depth.
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.set_depth_stencil_attachment(cascade_depth, false, false);
        s.clear_depth(f32::to_bits(1.0));
        s.emit_command(dummy_draw());
        // Pass 1: scene pass — different rt+depth, sample cascade_depth.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.set_depth_stencil_attachment(depth(), false, false);
        s.emit_command(Command::set_fragment_texture(cascade_depth.raw(), 4));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_load_actions();
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 2);
        // Pass 0 is the last (only) pass that depth-attaches cascade_depth;
        // without sampler awareness Rule B would flip Store→DontCare and
        // discard the caster depth before the scene PS samples it.
        assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
        assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
        // Pass 1's depth (the scene depth) is never sampled this frame,
        // so the normal Rule B optimisation still applies there.
        assert_eq!(s.passes()[1].depth_texture(), depth());
        assert_eq!(s.passes()[1].depth_store(), StoreAction::DontCare);
    }

    #[test]
    fn rule_a_reverts_dontcare_when_color_sampled_later() {
        let rt = tex(0x4000);
        // Pass 0 first-attaches a fresh color rt (Rule A: Load=DontCare),
        // pass 1 samples that same rt as a fragment texture. The eager
        // DontCare must be reverted at finalize so the sampler reads the
        // pass-0 content, not undefined tile memory.
        let mut s = fresh();
        // Bounce off backbuffer() first so the next pass-open is first-use of rt.
        s.emit_command(dummy_draw());
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        // First-use ⇒ Rule A flipped Load=DontCare eagerly.
        assert_eq!(s.passes()[1].color_load(), ColorLoad::DontCare);
        // Bounce back to backbuffer() and sample rt.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_load_actions();
        // finalize_load_actions must revert pass 1's eager DontCare.
        assert_eq!(s.passes()[1].color_load(), ColorLoad::Load);
    }

    // ── Rule G: depth-only strip for clear-only passes ────────────

    #[test]
    fn rule_g_strips_color_from_clear_only_pass_with_wasted_color() {
        let cascade_color = tex(0x3000);
        let cascade_d0 = tex(0x9000);
        let cascade_d1 = tex(0x9100);
        // Cascade-init clear-only pass: cascade_color (Clear), depth
        // sampled by scene (so depth Store stays Store via Rule B).
        // Rule C flips color Store=DontCare because the next pass on
        // cascade_color also begins with Clear. Rule G should then
        // strip the color attachment so the pass becomes depth-only.
        let mut s = fresh();
        // Pass 0: cascade-color + cascade_d0, clear-only (no draws).
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.set_depth_stencil_attachment(cascade_d0, false, false);
        s.clear_color(1, 2, 3, 4);
        s.clear_depth(f32::to_bits(1.0));
        // Pass 1: same cascade_color but different depth. cascade_d0
        // is sampled in the scene pass later.
        s.set_depth_stencil_attachment(cascade_d1, false, false);
        s.clear_color(1, 2, 3, 4);
        s.clear_depth(f32::to_bits(1.0));
        s.emit_command(dummy_draw());
        // Scene pass samples cascade_d0 so its Store must stay.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.set_depth_stencil_attachment(depth(), false, false);
        s.emit_command(Command::set_fragment_texture(cascade_d0.raw(), 4));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.coalesce_clear_only_passes();
        s.finalize_load_actions();
        s.finalize_store_actions();
        s.strip_dead_color_in_clear_only_passes();
        s.cull_dead_clear_only_passes();
        // The cascade-d0 clear-only pass should now be depth-only:
        // color_texture stripped, depth_texture preserved.
        let stripped = s
            .passes()
            .iter()
            .find(|p| p.depth_texture() == cascade_d0)
            .expect("cascade_d0 pass must remain");
        assert_eq!(
            stripped.color_texture(),
            MetalHandle::NULL,
            "color stripped"
        );
        assert_eq!(stripped.depth_store(), StoreAction::Store);
    }

    // ── Rule F: dead clear-only pass culling ──────────────────────

    #[test]
    fn rule_f_culls_pass_where_both_stores_become_dontcare() {
        let cascade_color = tex(0x3000);
        let cascade_depth = tex(0x9000);
        // Pass 0: cascade_color (Clear) + cascade_depth (Clear), no
        // draws. cascade_depth is NEVER sampled this frame, so Rule B
        // flips depth Store=DontCare. cascade_color is non-backbuffer,
        // not sampled, last-use → Rule D flips color Store=DontCare.
        // Both Stores DontCare + no draws + no blits → Rule F culls.
        let mut s = fresh();
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.set_depth_stencil_attachment(cascade_depth, false, false);
        s.clear_color(1, 2, 3, 4);
        s.clear_depth(f32::to_bits(1.0));
        // No draws, no blits — pure clear-only pass.
        // Switch back to BB so this is the last cascade frame use.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.set_depth_stencil_attachment(depth(), false, false);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.coalesce_clear_only_passes();
        s.finalize_load_actions();
        s.finalize_store_actions();
        s.cull_dead_clear_only_passes();
        // The cascade clear-only pass should be gone; only the BB
        // scene pass remains.
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].color_texture(), backbuffer());
    }

    #[test]
    fn rule_f_keeps_pass_where_depth_is_sampled() {
        let cascade_color = tex(0x3000);
        let cascade_depth = tex(0x9000);
        // Same as above but cascade_depth IS sampled by the scene pass
        // — Rule B keeps its Store=Store, so the cascade pass still
        // performs observable work (depth clear lands in VRAM for the
        // sampler). Rule F must NOT cull.
        let mut s = fresh();
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.set_depth_stencil_attachment(cascade_depth, false, false);
        s.clear_color(1, 2, 3, 4);
        s.clear_depth(f32::to_bits(1.0));
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.set_depth_stencil_attachment(depth(), false, false);
        s.emit_command(Command::set_fragment_texture(cascade_depth.raw(), 4));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.coalesce_clear_only_passes();
        s.finalize_load_actions();
        s.finalize_store_actions();
        s.cull_dead_clear_only_passes();
        // Cascade clear-only pass stays — depth Store must commit to
        // VRAM for the scene's sample_compare to read it.
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
        assert_eq!(s.passes()[0].depth_store(), StoreAction::Store);
    }

    // ── Rule E: clear-only pass coalescing ────────────────────────

    #[test]
    fn rule_e_bb_clear_coalesces_into_scene_pass() {
        let other_rt = tex(0x3000);
        // The canonical WoW frame pattern that produced spurious BB
        // clear passes: Clear(BB) → SetRT(other) → … → SetRT(BB) →
        // Draw. The clear-only BB pass should fold into the scene
        // pass's color_load.
        let mut s = fresh();
        s.clear_color(7, 7, 7, 7);
        // Switch rt — currently materialises a spurious BB clear pass.
        s.set_color_render_target(other_rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        // Come back to BB and draw.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.coalesce_clear_only_passes();
        // Three passes pre-coalesce: BB clear-only, other_rt draw, BB
        // draw. Post-coalesce: two — other_rt, BB-with-Clear-load.
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].color_texture(), other_rt);
        assert_eq!(s.passes()[1].color_texture(), backbuffer());
        assert!(matches!(
            s.passes()[1].color_load(),
            ColorLoad::Clear {
                r: 7,
                g: 7,
                b: 7,
                a: 7
            }
        ));
    }

    #[test]
    fn rule_e_aborts_when_intervening_pass_samples_target() {
        let rt = tex(0x4000);
        // If something between the clear-only pass and the candidate
        // merge target SAMPLES the texture, moving the Clear past it
        // would change the read; the merge must be rejected.
        let mut s = fresh();
        // Pass 0: clear-only on rt.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.clear_color(1, 2, 3, 4);
        // Force the pending clear to materialise by hopping rt
        // (combined flush).
        s.set_color_render_target(tex(0x5000), 256, 256, RT_FORMAT);
        s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
        s.emit_command(dummy_draw());
        // Re-attach rt and draw. Without the read at 0x5000 this would
        // be a valid merge target, but the intervening sample disables it.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let before = s.passes().len();
        s.coalesce_clear_only_passes();
        // Coalesce must not delete the clear-only pass.
        assert_eq!(s.passes().len(), before);
        // rt's clear-only pass is still there with its Clear load action.
        let cleared = s
            .passes()
            .iter()
            .find(|p| p.color_texture() == rt && matches!(p.color_load(), ColorLoad::Clear { .. }))
            .expect("clear-only rt pass must remain");
        let cmds = cleared.commands();
        let has_draw = cmds.iter().any(|c| {
            c.cmd == mtld3d_shared::CommandType::DrawPrimitives as u32
                || c.cmd == mtld3d_shared::CommandType::DrawIndexedPrimitives as u32
        });
        assert!(!has_draw);
    }

    #[test]
    fn rule_d_non_backbuffer_color_last_use_is_dontcare() {
        let cascade_color = tex(0x3000);
        // CSM cascade color is a placeholder for the depth-only caster
        // pass and is never sampled. Rule D must flip its Store to
        // DontCare so the 16 MB writeback doesn't hit VRAM. Backbuffer
        // color in the next pass must stay Store (Present consumes it).
        let mut s = fresh();
        // Pass 0: cascade caster pass — color is junk, depth gets work.
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.emit_command(dummy_draw());
        // Pass 1: scene pass on backbuffer.
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].color_texture(), cascade_color);
        assert_eq!(s.passes()[0].color_store(), StoreAction::DontCare);
        // Backbuffer Present needs the pixels — exempt from Rule D.
        assert_eq!(s.passes()[1].color_texture(), backbuffer());
        assert_eq!(s.passes()[1].color_store(), StoreAction::Store);
    }

    #[test]
    fn rule_d_keeps_store_when_color_sampled_later() {
        let rt = tex(0x4000);
        // A non-backbuffer color rt that IS sampled by a later pass
        // must preserve its content; Rule D must NOT flip Store to
        // DontCare for it.
        let mut s = fresh();
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 2);
        assert_eq!(s.passes()[0].color_texture(), rt);
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    }

    #[test]
    fn cascade_init_sequence_collapses_to_one_pass() {
        let cascade_color = tex(0x3000);
        let cascade_depth = tex(0x9000);
        // WoW's typical cascade-init sequence is
        //   SetRT(C) → Clear(TARGET) → SetDST(D) → Clear(ZBUFFER) → Draw.
        // The pending color clear when SetDST fires applies to the
        // *unchanged* color rt C, so it must survive the depth-attach
        // switch and combine with the depth clear on the next pass.
        // Without the split flush, this would produce a spurious
        // 1-cmd clear-only pass for C with the still-old depth.
        let mut s = fresh();
        s.set_color_render_target(cascade_color, 2048, 2048, RT_FORMAT);
        s.clear_color(1, 2, 3, 4);
        s.set_depth_stencil_attachment(cascade_depth, false, false);
        s.clear_depth(f32::to_bits(1.0));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        // One pass — no spurious clear-only pass dropped between the two
        // clears.
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].color_texture(), cascade_color);
        assert_eq!(s.passes()[0].depth_texture(), cascade_depth);
        // Both clears land on the single pass's load actions.
        assert!(matches!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 1,
                g: 2,
                b: 3,
                a: 4
            }
        ));
        assert!(matches!(
            s.passes()[0].depth_load(),
            DepthLoad::Clear { .. }
        ));
    }

    #[test]
    fn pending_color_clear_survives_depth_attach_change() {
        let d2 = tex(0x9000);
        // Narrow assertion: when only the depth attachment changes and
        // a color clear is pending, the clear stays pending (does not
        // materialise into a spurious pass).
        let mut s = fresh();
        s.clear_color(7, 7, 7, 7);
        s.set_depth_stencil_attachment(d2, false, false);
        // No draws yet — the pending color clear should still be
        // pending on the same (unchanged) color rt.
        assert!(s.passes().is_empty());
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        assert!(matches!(
            s.passes()[0].color_load(),
            ColorLoad::Clear {
                r: 7,
                g: 7,
                b: 7,
                a: 7
            }
        ));
    }

    #[test]
    fn rule_c_skips_color_store_dontcare_when_sampled_between() {
        let rt = tex(0x5000);
        // Pass 0 writes rt, pass 1 samples rt, pass 2 re-clears rt.
        // Rule C would naively flip pass 0's color_store to DontCare
        // because the next consumer (pass 2) begins with Clear — but
        // pass 1 in between samples rt, so the content must survive to
        // VRAM. Sampler-aware Rule C keeps pass 0 Store.
        let mut s = fresh();
        // Pass 0: write to rt.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.emit_command(dummy_draw());
        // Pass 1: sample rt into backbuffer().
        s.set_color_render_target(backbuffer(), BB_SIZE.0, BB_SIZE.1, BB_FORMAT);
        s.emit_command(Command::set_fragment_texture(rt.raw(), 0));
        s.emit_command(dummy_draw());
        // Pass 2: clear+rewrite rt.
        s.set_color_render_target(rt, 256, 256, RT_FORMAT);
        s.clear_color(0, 0, 0, 0);
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        s.finalize_load_actions();
        s.finalize_store_actions();
        assert_eq!(s.passes().len(), 3);
        assert_eq!(s.passes()[0].color_texture(), rt);
        // Without the sampler check, pass 0's color_store would be
        // DontCare (next consumer at pass 2 begins with Clear).
        // Sampler-aware Rule C keeps Store because pass 1 reads rt.
        assert_eq!(s.passes()[0].color_store(), StoreAction::Store);
    }

    // ── Rule H — strip color from passes-with-draws where every draw
    // ── ran with COLORWRITEENABLE = 0. Side-map of with-color → no-color
    // ── pipeline handles is supplied by the caller (built by the
    // ── FrameEncoder at draw time from zero-mask snapshots).

    const PSO_WITH: u64 = 0xAAAA_1111;
    const PSO_NO_COLOR: u64 = 0xBBBB_2222;

    fn set_pso(handle: u64) -> Command {
        Command::set_render_pipeline_state(handle)
    }

    #[test]
    fn rule_h_strips_color_when_all_draws_have_writemask_zero() {
        let mut s = fresh();
        // Five zero-mask draws into the backbuffer + depth pass.
        for _ in 0..5 {
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_WITH));
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");
        let mut alt = HashMap::new();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.strip_color_from_no_color_draw_passes(&alt);
        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            MetalHandle::NULL,
            "color attachment stripped"
        );
        assert_eq!(pass.color_load(), ColorLoad::DontCare);
        assert_eq!(pass.color_store(), StoreAction::DontCare);
        // Every SetPSO in the pass now binds the no-color variant.
        let pso_handles: Vec<u64> = pass
            .commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b)
            .collect();
        assert!(!pso_handles.is_empty(), "test setup emitted SetPSO");
        assert!(
            pso_handles.iter().all(|h| *h == PSO_NO_COLOR),
            "every SetPSO rewritten: {pso_handles:?}"
        );
    }

    #[test]
    fn rule_h_keeps_color_when_any_draw_writes_color() {
        let mut s = fresh();
        for _ in 0..4 {
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_WITH));
            s.emit_command(dummy_draw());
        }
        // One non-zero-mask draw flips the pass's tag.
        s.note_draw_color_write_mask(0xF);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let mut alt = HashMap::new();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.strip_color_from_no_color_draw_passes(&alt);
        let pass = &s.passes()[0];
        assert_eq!(pass.color_texture(), backbuffer(), "color attachment kept");
        assert!(pass.color_writes_observed());
        // SetPSO handles preserved unchanged.
        assert!(
            pass.commands()
                .iter()
                .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
                .all(|c| c.param_b == PSO_WITH),
            "no rewrite on color-writing pass"
        );
    }

    #[test]
    fn rule_h_skipped_without_depth_attachment() {
        let mut s = fresh();
        // Detach depth so the candidate pass has color but no depth —
        // stripping color would produce an encoder with zero
        // attachments, which Metal rejects.
        s.set_depth_stencil_attachment(MetalHandle::NULL, false, false);
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let mut alt = HashMap::new();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.strip_color_from_no_color_draw_passes(&alt);
        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            backbuffer(),
            "no-depth pass must keep color"
        );
    }

    #[test]
    fn rule_h_skipped_for_clear_only_pass() {
        // A pass with zero draws is Rule G's territory, not Rule H's.
        // Rule H must leave it alone so finalize-time invariants hold.
        let mut s = fresh();
        s.clear_color(0, 0, 0, 0);
        s.flush_pending_clears();
        let alt: HashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = HashMap::new();
        s.strip_color_from_no_color_draw_passes(&alt);
        let pass = &s.passes()[0];
        // Color still attached — Rule H bailed because the pass had
        // no draw commands.
        assert_eq!(pass.color_texture(), backbuffer());
    }

    #[test]
    fn rule_h_aborts_strip_on_missing_alt_handle() {
        // A zero-mask draw bound PSO_WITH but the side-map is empty
        // (would mean the FrameEncoder skipped the dual-build path —
        // an upstream bug). The rule must keep the color attachment
        // intact rather than bind a with-color pipeline against a
        // depth-only render pass descriptor.
        let mut s = fresh();
        s.note_draw_color_write_mask(0);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");
        let alt: HashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = HashMap::new();
        s.strip_color_from_no_color_draw_passes(&alt);
        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            backbuffer(),
            "missing alt-handle → no strip"
        );
        assert_eq!(
            pass.commands()
                .iter()
                .find(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
                .map(|c| c.param_b),
            Some(PSO_WITH),
            "no rewrite on aborted strip"
        );
    }

    #[test]
    fn rule_h_strips_color_with_self_mapped_depth_clear_quad() {
        const PSO_CASTER: u64 = 0xAAAA_1111;
        const PSO_CASTER_NO_COLOR: u64 = 0xBBBB_2222;
        const PSO_CLEAR_QUAD_DEPTH: u64 = 0xCCCC_3333;
        // Cascade caster pass: per-tile depth clear-quad SetPSO +
        // zero-mask caster SetPSO + draws. The depth clear-quad
        // pipeline is built `has_color: false` and self-maps in
        // `no_color_pipeline_alt` (encoder.rs); Rule H must strip
        // color cleanly without firing the side-map-miss warn.

        let mut s = fresh();
        for _ in 0..3 {
            s.emit_command(set_pso(PSO_CLEAR_QUAD_DEPTH));
            s.emit_command(dummy_draw());
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_CASTER));
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");

        let mut alt = HashMap::new();
        alt.insert(PSO_CASTER, pso(PSO_CASTER_NO_COLOR));
        alt.insert(PSO_CLEAR_QUAD_DEPTH, pso(PSO_CLEAR_QUAD_DEPTH));
        s.strip_color_from_no_color_draw_passes(&alt);

        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            MetalHandle::NULL,
            "color attachment stripped"
        );
        let pso_handles: Vec<u64> = pass
            .commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b)
            .collect();
        assert!(
            pso_handles.contains(&PSO_CASTER_NO_COLOR),
            "caster rewritten to no-color sibling: {pso_handles:?}"
        );
        assert!(
            pso_handles.contains(&PSO_CLEAR_QUAD_DEPTH),
            "self-mapped depth clear-quad preserved: {pso_handles:?}"
        );
        assert!(
            !pso_handles.contains(&PSO_CASTER),
            "caster with-color handle replaced: {pso_handles:?}"
        );
    }

    #[test]
    fn rule_h_strips_color_and_clear_quad_when_only_zero_mask_draws_plus_clear_quad() {
        const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
        // Cascade caster pass shape: WoW issued mid-pass `Clear` on
        // the cascade color atlas (e.g. per-tile clear), which the
        // encoder folded into a cross-pass color clear-quad. The rest
        // of the pass is zero-mask caster draws. Rule H must strip
        // the color attachment AND drain the clear-quad's commands so
        // the resulting depth-only descriptor doesn't try to bind a
        // color-output clear-quad pipeline.
        let mut s = fresh();
        // Color clear-quad block — 6 commands, none of which should
        // tag `color_writes_observed`.
        let start = s.open_color_clear_quad_block();
        s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
        s.emit_command(dummy_draw());
        s.close_color_clear_quad_block(start);
        // Two zero-mask caster draws.
        for _ in 0..2 {
            s.note_draw_color_write_mask(0);
            s.emit_command(set_pso(PSO_WITH));
            s.emit_command(dummy_draw());
        }
        s.end_current_pass("test");
        // Only the real caster needs a side-map entry; clear-quad
        // PSOs are removed wholesale and don't need to resolve.
        let mut alt = HashMap::new();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.strip_color_from_no_color_draw_passes(&alt);

        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            MetalHandle::NULL,
            "color attachment stripped"
        );
        assert_eq!(pass.color_load(), ColorLoad::DontCare);
        assert_eq!(pass.color_store(), StoreAction::DontCare);
        assert!(
            pass.color_clear_quad_ranges().is_empty(),
            "clear-quad ranges drained after strip"
        );
        // Caster SetPSO rewritten; clear-quad SetPSO gone.
        let pso_handles: Vec<u64> = pass
            .commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b)
            .collect();
        assert!(
            !pso_handles.contains(&PSO_CLEAR_QUAD_COLOR),
            "color clear-quad SetPSO removed: {pso_handles:?}"
        );
        assert!(
            !pso_handles.contains(&PSO_WITH),
            "caster with-color handle replaced: {pso_handles:?}"
        );
        assert!(
            pso_handles.iter().all(|h| *h == PSO_NO_COLOR),
            "every surviving SetPSO is the no-color variant: {pso_handles:?}"
        );
    }

    #[test]
    fn rule_h_keeps_color_clear_quad_when_real_color_writing_draw_present() {
        const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
        // Same shape as above but one real draw writes color
        // (`COLORWRITEENABLE != 0`). The clear-quad output is now
        // load-bearing for that draw's blend, so Rule H must skip the
        // pass entirely — both the attachment AND the clear-quad
        // commands must survive untouched.
        let mut s = fresh();
        let start = s.open_color_clear_quad_block();
        s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
        s.emit_command(dummy_draw());
        s.close_color_clear_quad_block(start);
        // One real color-writing draw.
        s.note_draw_color_write_mask(0xF);
        s.emit_command(set_pso(PSO_WITH));
        s.emit_command(dummy_draw());
        s.end_current_pass("test");

        let mut alt = HashMap::new();
        alt.insert(PSO_WITH, pso(PSO_NO_COLOR));
        s.strip_color_from_no_color_draw_passes(&alt);

        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            backbuffer(),
            "real color-writing draw keeps the attachment"
        );
        assert!(pass.color_writes_observed());
        assert_eq!(
            pass.color_clear_quad_ranges().len(),
            1,
            "clear-quad range preserved"
        );
        let pso_handles: Vec<u64> = pass
            .commands()
            .iter()
            .filter(|c| c.cmd == CommandType::SetRenderPipelineState as u32)
            .map(|c| c.param_b)
            .collect();
        assert!(
            pso_handles.contains(&PSO_CLEAR_QUAD_COLOR),
            "clear-quad SetPSO preserved: {pso_handles:?}"
        );
        assert!(
            pso_handles.contains(&PSO_WITH),
            "real-draw SetPSO not rewritten: {pso_handles:?}"
        );
    }

    #[test]
    fn rule_h_skipped_when_pass_has_only_color_clear_quad_no_real_draws() {
        const PSO_CLEAR_QUAD_COLOR: u64 = 0xCAFE_BABE;
        // A pass with ONLY a color clear-quad and no real draw is not
        // Rule H's territory — it leaves the pass intact (Rule F /
        // Rule G handle the clear-only shape elsewhere). The
        // clear-quad's color writes are kept; if they're wasted,
        // upstream rules cull the pass.
        let mut s = fresh();
        let start = s.open_color_clear_quad_block();
        s.emit_command(set_pso(PSO_CLEAR_QUAD_COLOR));
        s.emit_command(dummy_draw());
        s.close_color_clear_quad_block(start);
        s.end_current_pass("test");

        let alt: HashMap<u64, MetalHandle<MTLRenderPipelineStateKind>> = HashMap::new();
        s.strip_color_from_no_color_draw_passes(&alt);

        let pass = &s.passes()[0];
        assert_eq!(
            pass.color_texture(),
            backbuffer(),
            "clear-quad-only pass left alone by Rule H"
        );
        assert_eq!(pass.color_clear_quad_ranges().len(), 1);
    }

    // ── Clear-quad mid-pass Clear translation ─────────────────────

    /// A shared shadow tile-atlas pattern.
    ///
    /// Open a single pass on a cascade depth texture, then for each of N
    /// tiles emit `set_viewport(tile_N) + clear_depth(1.0) + draw`.
    /// Under Metal's full-attachment Clear semantics, breaking a pass
    /// per tile would emit N separate passes each `loadAction = Clear`,
    /// wiping the prior tile's draws; the clear-quad path instead keeps
    /// one pass open and returns N `EmitQuad` outcomes the encoder
    /// layer translates into scissored fullscreen-triangle draws.
    #[test]
    fn wow_tile_atlas_clears_emit_inline_quad_not_pass_break() {
        const TILE_COUNT: u32 = 9;
        let mut s = fresh();
        // Establish a pass open on the depth attachment with one draw,
        // so the first per-tile Clear arrives at a "has work" pass.
        s.emit_command(dummy_draw());
        let z = f32::to_bits(1.0);
        let mut quad_outcomes: u32 = 0;
        for tile in 0..TILE_COUNT {
            let x = (tile % 3) * 683;
            let y = (tile / 3) * 683;
            s.set_viewport(x, y, 683, 683, 0.0, 1.0);
            match s.clear_depth(z) {
                DepthClearOutcome::EmitQuad {
                    value, viewport, ..
                } => {
                    assert_eq!(value, z);
                    assert_eq!(viewport, (x, y, 683, 683));
                    quad_outcomes += 1;
                }
                DepthClearOutcome::Folded => {
                    panic!("tile {tile} clear should have returned EmitQuad");
                }
            }
            s.emit_command(dummy_draw());
        }
        assert_eq!(
            quad_outcomes, TILE_COUNT,
            "every tile-clear must emit a quad outcome"
        );
        assert_eq!(
            s.passes().len(),
            1,
            "single pass should survive the entire tile sequence"
        );
    }

    /// Color mirror of the depth tile-atlas test.
    #[test]
    fn wow_color_clear_mid_pass_returns_emit_quad_per_tile() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        let outcome = s.clear_color(0x11, 0x22, 0x33, 0x44);
        assert!(matches!(
            outcome,
            ColorClearOutcome::EmitQuad {
                rgba: (0x11, 0x22, 0x33, 0x44),
                ..
            }
        ));
        assert_eq!(s.passes().len(), 1);
    }

    /// First Clear in a pass still folds into the pass's load action.
    ///
    /// The pass has only the implicit viewport command, no draws —
    /// Metal's `loadAction = Clear` is the cheap path here. Quad
    /// emission only kicks in once real work has been added.
    #[test]
    fn first_depth_clear_in_pass_folds_into_load_action() {
        let mut s = fresh();
        s.ensure_pass_open();
        let z = f32::to_bits(1.0);
        let outcome = s.clear_depth(z);
        assert_eq!(outcome, DepthClearOutcome::Folded);
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
    }

    /// Two mid-pass Clears with different values both emit their own quad outcome.
    ///
    /// The encoder will materialise both with their distinct depths in
    /// the same encoder.
    #[test]
    fn distinct_depth_clear_values_in_same_pass_each_emit_quad() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        let z1 = f32::to_bits(0.5);
        let z2 = f32::to_bits(0.75);
        let o1 = s.clear_depth(z1);
        s.emit_command(dummy_draw());
        let o2 = s.clear_depth(z2);
        s.emit_command(dummy_draw());
        assert!(matches!(o1, DepthClearOutcome::EmitQuad { value, .. } if value == z1));
        assert!(matches!(o2, DepthClearOutcome::EmitQuad { value, .. } if value == z2));
        assert_eq!(s.passes().len(), 1);
    }

    /// Cross-pass case: a tile sequence where each tile is its own pass.
    ///
    /// A fresh `SetRenderTarget` between tiles breaks the pass. First
    /// tile's `Clear` lands as `pending_depth_clear` and the pass opens
    /// with `loadAction = Clear` for the full attachment (correct —
    /// first use of the texture). Second tile's `Clear` hits a CLOSED
    /// pass on a depth texture *already seen* this frame — folding into
    /// a fresh `loadAction = Clear` would let Metal wipe the first
    /// tile's draws. The fix opens the second pass with
    /// `loadAction = Load` and returns `EmitQuad` so the encoder layer
    /// emits a scissored clear-quad inside the new pass.
    #[test]
    fn cross_pass_depth_clear_uses_load_plus_quad() {
        let mut s = fresh();
        // Tile 0: open the first pass on depth(); Clear folds into load
        // action; a draw lands in the pass; we end the pass (e.g. a
        // SetRenderTarget switch).
        let z = f32::to_bits(1.0);
        s.set_viewport(0, 0, 683, 683, 0.0, 1.0);
        assert_eq!(s.clear_depth(z), DepthClearOutcome::Folded);
        s.emit_command(dummy_draw());
        assert_eq!(s.passes().len(), 1);
        assert_eq!(s.passes()[0].depth_load(), DepthLoad::Clear { value: z });
        s.end_current_pass("test_color_rt_switch");

        // Tile 1: Clear arrives on the same depth(). Depth is already
        // in seen_depth_rts → cross-pass case fires. Pass opens with
        // load=Load (preserving tile 0's content) and the outcome is
        // EmitQuad so the encoder emits a scissored clear-quad.
        s.set_viewport(683, 0, 683, 683, 0.0, 1.0);
        let outcome = s.clear_depth(z);
        assert!(
            matches!(
                outcome,
                DepthClearOutcome::EmitQuad { value, viewport, .. }
                    if value == z && viewport == (683, 0, 683, 683)
            ),
            "cross-pass clear must return EmitQuad, got {outcome:?}",
        );
        assert_eq!(s.passes().len(), 2);
        assert_eq!(
            s.passes()[1].depth_load(),
            DepthLoad::Load,
            "tile 1 pass must use Load so Metal preserves tile 0",
        );
    }

    /// Sampleable shadow maps must keep `Store` even when not sampled in this frame.
    ///
    /// The receiver may sample them on a future frame (cascade-3
    /// rotations etc.). The `is_sampleable` flag on
    /// `set_depth_stencil_attachment` covers the bootstrap-frame gap
    /// that the persist-`seen_sampled` fix alone can't close for
    /// rarely-sampled cascades.
    #[test]
    fn sampleable_depth_keeps_store_even_when_never_sampled() {
        let cascade_depth = tex(0xCAFE_5000);
        let mut s = fresh();
        s.set_depth_stencil_attachment(cascade_depth, /* is_sampleable */ true, false);
        s.emit_command(dummy_draw());
        s.finalize_store_actions();
        let cascade_pass = s
            .passes()
            .iter()
            .find(|p| p.depth_texture() == cascade_depth)
            .expect("cascade pass present");
        assert_eq!(
            cascade_pass.depth_store(),
            StoreAction::Store,
            "sampleable depth must keep Store even without a sample in seen_sampled",
        );
    }

    /// Non-sampleable depth still gets the Rule B optimization when no sample lands on it.
    ///
    /// Non-sampleable means a standalone `CreateDepthStencilSurface`,
    /// e.g. the backbuffer's z; the sample has to be absent for this
    /// frame. Guards against the sampleable-flag fix accidentally
    /// over-conservatively keeping Store on every depth attachment.
    #[test]
    fn non_sampleable_depth_still_gets_rule_b_dontcare() {
        let rt_depth = tex(0xCAFE_6000);
        let mut s = fresh();
        s.set_depth_stencil_attachment(
            rt_depth, /* is_sampleable */ false, /* has_stencil */ false,
        );
        s.emit_command(dummy_draw());
        s.finalize_store_actions();
        let rt_pass = s
            .passes()
            .iter()
            .find(|p| p.depth_texture() == rt_depth)
            .expect("rt pass present");
        assert_eq!(
            rt_pass.depth_store(),
            StoreAction::DontCare,
            "non-sampleable depth never sampled → Rule B optimization preserved",
        );
    }

    /// Visibility-counting passes fall back to the legacy pass-break path.
    ///
    /// Emitting a clear-quad mid-pass would falsely increment the
    /// per-pass fragment counter; until proper save/restore of
    /// `SetVisibilityResultMode` lands, the safe behaviour is to end
    /// the pass on Clear-with-work as before.
    #[test]
    fn clear_depth_with_visibility_query_active_falls_back_to_pass_break() {
        let mut s = fresh();
        s.emit_command(dummy_draw());
        // Activate visibility counting on the current pass.
        s.emit_command(Command::set_visibility_result_mode(
            mtld3d_shared::mtl::VisibilityResultMode::Counting,
            0,
        ));
        let z = f32::to_bits(1.0);
        let outcome = s.clear_depth(z);
        assert_eq!(
            outcome,
            DepthClearOutcome::Folded,
            "visibility-active Clear must fall back to legacy pass-break (not EmitQuad)"
        );
    }
}
