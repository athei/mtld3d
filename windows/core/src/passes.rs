//! Pure-Rust render-pass state machine used by the PE-side `FrameEncoder`.
//!
//! Holds the `passes: Vec<Pass>` plus the bookkeeping for pass breaks on
//! `SetRenderTarget`, `SetDepthStencilSurface`, and mid-frame `Clear`.

use log::{Level, log_enabled, trace};
use mtld3d_shared::{
    BlitCommand, BlitCommandType, Command, CommandType, MetalHandle,
    mtl::{CullMode, PixelFormat, VERTEX_STREAM_SLOTS, VisibilityResultMode},
    mtl_handle::{MTLRenderPipelineStateKind, MTLTextureKind},
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::{
    dirty_range::DirtyRange, pipeline_state::ExtraColorAttachments, render_scale::RenderScale,
};

/// What a clear-only pass carries, and the attachments it must land on.
struct ClearMerge {
    color: MetalHandle<MTLTextureKind>,
    color_subresource: u32,
    /// Render targets 1..3 as `(texture, subresource)`; a null texture is an unbound slot.
    ///
    /// A merge target must carry exactly this set, slot for slot.
    extra: [(MetalHandle<MTLTextureKind>, u32); 3],
    depth: MetalHandle<MTLTextureKind>,
    needs_color: bool,
    needs_depth: bool,
    needs_stencil: bool,
}

/// Compile-time gate for Rule A (first-use `DontCare`).
///
/// Flip to `false` for a single-line hotfix if a temporal-blending game
/// surfaces that reads prior-frame contents on first use of frame N.
const ENABLE_FIRST_USE_DONTCARE: bool = true;

/// Compile-time gate for Rule A on the stencil plane (first-use `DontCare`).
///
/// The stencil plane shares the depth texture, so its first use in a frame
/// takes the same `DontCare` the depth plane takes, under the same
/// predicate. Stencil written in frame N and tested in frame N+1 without a
/// clear in between was already undefined before this rule: the stencil
/// store mirrors the depth store, so whenever Rule B discards depth at frame
/// end the stencil content goes with it. Flip to `false` if a game surfaces
/// that carries stencil across `Present` and a frame-start `Load` turns out
/// to matter.
const ENABLE_FIRST_USE_STENCIL_DONTCARE: bool = true;

/// Compile-time gate for Rule B (last-use depth/stencil `DontCare`).
///
/// Flip to `false` if a game using `INTZ`-style late-frame depth-readback
/// surfaces. D3D9 spec already says depth contents are undefined across
/// `Present`, so this is conformant for any game that respects the spec.
const ENABLE_LAST_USE_DEPTH_DONTCARE: bool = true;

/// Compile-time gate for Rule C (color `Store=DontCare`).
///
/// Applies when the next pass that touches the same color rt begins with
/// a full-attachment clear, i.e. its `color_load == ColorLoad::Clear
/// { .. }`. That load action is reached only by a `Clear` the encoder
/// judged to cover the whole attachment; a `Clear` bounded by a sub-rect
/// viewport, a scissor or `pRects` paints a quad over a `Load` instead and
/// is therefore not a Rule C opportunity. Mirrors Rule B but keyed on
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

/// Vertex texture fetch slots (`vs_3_0` s0..s3, `D3DVERTEXTEXTURESAMPLER0..3`).
pub const VERTEX_SAMPLER_SLOTS: usize = 4;

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

/// How the next render-pass should load its stencil attachment.
///
/// Separate from `DepthLoad` because the two planes of a combined
/// `Depth32Float_Stencil8` attachment take independent load actions:
/// `Clear(D3DCLEAR_STENCIL)` without `D3DCLEAR_ZBUFFER` resets stencil while
/// carrying depth forward, and a stencil clear value is an integer rather
/// than an f32 bit pattern. `DontCare` is Rule A's first-use discard and
/// nothing else: games carry stencil across passes within a frame, so a
/// later pass in the same frame always loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilLoad {
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
/// stays open. `NoOp` means there was nothing to clear: no depth-stencil
/// texture is attached, or the viewport has no area, and D3D9 clears nothing
/// in either case. The pass state is left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthClearOutcome {
    Folded,
    NoOp,
    EmitQuad {
        value: u32,
        viewport: (u32, u32, u32, u32),
        has_color: bool,
        color_format: PixelFormat,
    },
}

/// What `clear_stencil` decided.
///
/// `Folded` means the clear became the next pass's `loadAction`; `EmitQuad`
/// means the caller paints a scissored quad instead, because folding would
/// clear the whole attachment and wipe tiles the frame already drew. `NoOp`
/// means there was nothing to clear: no depth-stencil texture is attached, or
/// the viewport has no area, and D3D9 clears nothing in either case. The pass
/// state is left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilClearOutcome {
    Folded,
    NoOp,
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

/// Render target 1..3 as currently bound, before any pass has frozen it.
///
/// `size` is the texture extent the pass will see and `logical_size` the one
/// D3D9 reports; `scale` relates the two exactly as for render target 0. A
/// slot is bound when `texture` is non-null, and takes part in a pass only
/// when `size` equals render target 0's (the D3D9 rule: draws reach targets
/// whose extent matches the first one; a mismatched target is still cleared).
#[derive(Debug, PartialEq, Eq)]
pub struct ExtraColorSlot {
    pub texture: MetalHandle<MTLTextureKind>,
    /// `slice | (level << 8)`, as on [`Pass`].
    pub subresource: u32,
    pub size: (u32, u32),
    pub logical_size: (u32, u32),
    pub format: PixelFormat,
    pub scale: RenderScale,
    /// Whether the target's D3D format has a real alpha channel.
    pub has_alpha: bool,
}

impl ExtraColorSlot {
    /// The unbound slot.
    pub const NONE: Self = Self {
        texture: MetalHandle::NULL,
        subresource: 0,
        size: (0, 0),
        logical_size: (0, 0),
        format: PixelFormat::Bgra8Unorm,
        scale: RenderScale::IDENTITY,
        has_alpha: false,
    };

    #[must_use]
    pub const fn is_bound(&self) -> bool {
        !self.texture.is_null()
    }
}

/// One of render targets 1..3 as frozen onto a [`Pass`].
///
/// Unbound when `texture` is null. The load and store actions follow the
/// same rules as render target 0's, evaluated per attachment.
#[derive(Debug, PartialEq, Eq)]
pub struct PassColorAttachment {
    texture: MetalHandle<MTLTextureKind>,
    subresource: u32,
    size: (u32, u32),
    format: PixelFormat,
    load: ColorLoad,
    store: StoreAction,
}

impl PassColorAttachment {
    /// The unbound attachment.
    pub const NONE: Self = Self {
        texture: MetalHandle::NULL,
        subresource: 0,
        size: (0, 0),
        format: PixelFormat::Bgra8Unorm,
        load: ColorLoad::DontCare,
        store: StoreAction::DontCare,
    };

    #[must_use]
    pub const fn is_bound(&self) -> bool {
        !self.texture.is_null()
    }
    #[must_use]
    pub const fn texture(&self) -> MetalHandle<MTLTextureKind> {
        self.texture
    }
    #[must_use]
    pub const fn slice(&self) -> u32 {
        self.subresource & 0xff
    }
    #[must_use]
    pub const fn level(&self) -> u32 {
        self.subresource >> 8
    }
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }
    #[must_use]
    pub const fn load(&self) -> ColorLoad {
        self.load
    }
    #[must_use]
    pub const fn store(&self) -> StoreAction {
        self.store
    }
}

/// One bound colour attachment of a [`Pass`] as the store rules see it.
struct BoundColorAttachment {
    /// 0 = render target 0, 1..=3 = extras.
    slot: usize,
    texture: MetalHandle<MTLTextureKind>,
    subresource: u32,
    store: StoreAction,
}

impl BoundColorAttachment {
    const NONE: Self = Self {
        slot: 0,
        texture: MetalHandle::NULL,
        subresource: 0,
        store: StoreAction::DontCare,
    };
}

/// The bound colour attachments of one pass, at most four, in slot order.
///
/// A fixed array rather than a `Vec` because the store rules build one per
/// pass per frame.
struct BoundColorAttachments {
    items: [BoundColorAttachment; 4],
    len: usize,
}

impl BoundColorAttachments {
    fn iter(&self) -> impl Iterator<Item = &BoundColorAttachment> {
        self.items[..self.len].iter()
    }
}

/// The full colour binding set of the device, taken off the pass state.
///
/// `PassState::take_color_attachments` hands it out so a caller can bind
/// targets of its own for a scoped pass and then put the device's binding
/// back exactly, extras and alpha bit included.
pub struct SavedColorAttachments {
    texture: MetalHandle<MTLTextureKind>,
    slice: u32,
    level: u32,
    logical_size: (u32, u32),
    format: PixelFormat,
    scale: RenderScale,
    has_alpha: bool,
    extra: [ExtraColorSlot; 3],
}

impl SavedColorAttachments {
    /// Render target `slot` (0..=3) as a bindable [`ExtraColorSlot`], if bound.
    ///
    /// Slot 0 is always bound. The returned value carries the logical size
    /// and scale, so binding it as render target 0 reproduces the device's
    /// coordinate space for that target.
    #[must_use]
    pub fn slot(&self, slot: usize) -> Option<ExtraColorSlot> {
        if slot == 0 {
            return Some(ExtraColorSlot {
                texture: self.texture,
                subresource: self.slice | (self.level << 8),
                size: (
                    self.scale.dimension(self.logical_size.0),
                    self.scale.dimension(self.logical_size.1),
                ),
                logical_size: self.logical_size,
                format: self.format,
                scale: self.scale,
                has_alpha: self.has_alpha,
            });
        }
        let extra = &self.extra[slot - 1];
        extra.is_bound().then_some(ExtraColorSlot {
            texture: extra.texture,
            subresource: extra.subresource,
            size: extra.size,
            logical_size: extra.logical_size,
            format: extra.format,
            scale: extra.scale,
            has_alpha: extra.has_alpha,
        })
    }

    /// Whether extra slot `slot` (1..=3) matches render target 0's extent.
    #[must_use]
    pub fn extra_matches_rt0(&self, slot: usize) -> bool {
        let rt0 = (
            self.scale.dimension(self.logical_size.0),
            self.scale.dimension(self.logical_size.1),
        );
        let extra = &self.extra[slot - 1];
        extra.is_bound() && extra.size == rt0
    }
}

/// One Metal render pass.
///
/// Each pass maps to a single `MTLRenderCommandEncoder` on the unix side.
/// Attachments are frozen at pass open; further changes (`SetRenderTarget`,
/// mid-frame Clear, depth change) end the current pass and open a new one.
pub struct Pass {
    color_texture: MetalHandle<MTLTextureKind>,
    color_subresource: u32,
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
    /// Mip level of `depth_texture` the pass renders depth into.
    depth_level: u32,
    depth_load: DepthLoad,
    stencil_load: StencilLoad,
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
    /// Render targets 1..3; all unbound on a single-target pass.
    extra_color: [PassColorAttachment; 3],
}

impl Pass {
    #[must_use]
    pub const fn color_texture(&self) -> MetalHandle<MTLTextureKind> {
        self.color_texture
    }
    /// Render targets 1..3 of this pass, unbound entries included.
    #[must_use]
    pub const fn extra_color(&self) -> &[PassColorAttachment; 3] {
        &self.extra_color
    }
    /// Bit `i` set ⇒ render target `i + 1` is bound on this pass.
    #[must_use]
    pub fn extra_present_mask(&self) -> u8 {
        self.extra_color
            .iter()
            .enumerate()
            .fold(0, |m, (i, a)| if a.is_bound() { m | (1 << i) } else { m })
    }
    /// Every bound colour attachment on the pass, with its absolute slot.
    ///
    /// Render target 0 (slot 0) first, then the bound extras (slots 1..=3).
    /// Lets the load/store rules treat the attachments uniformly; the slot
    /// feeds [`Self::color_load_of`] / [`Self::set_color_store_of`].
    fn bound_color_attachments(&self) -> BoundColorAttachments {
        let mut list = BoundColorAttachments {
            items: [BoundColorAttachment::NONE; 4],
            len: 0,
        };
        if !self.color_texture.is_null() {
            list.items[0] = BoundColorAttachment {
                slot: 0,
                texture: self.color_texture,
                subresource: self.color_subresource,
                store: self.color_store,
            };
            list.len = 1;
        }
        for (i, a) in self.extra_color.iter().enumerate() {
            if a.is_bound() {
                list.items[list.len] = BoundColorAttachment {
                    slot: i + 1,
                    texture: a.texture,
                    subresource: a.subresource,
                    store: a.store,
                };
                list.len += 1;
            }
        }
        list
    }
    /// Load action of attachment `slot` (0 = render target 0, 1..=3 = extras).
    const fn color_load_of(&self, slot: usize) -> ColorLoad {
        if slot == 0 {
            self.color_load
        } else {
            self.extra_color[slot - 1].load
        }
    }
    /// Set the store action of attachment `slot` (0 = render target 0, 1..=3 = extras).
    const fn set_color_store_of(&mut self, slot: usize, store: StoreAction) {
        if slot == 0 {
            self.color_store = store;
        } else {
            self.extra_color[slot - 1].store = store;
        }
    }
    #[must_use]
    pub const fn color_slice(&self) -> u32 {
        self.color_subresource & 0xff
    }
    #[must_use]
    pub const fn color_level(&self) -> u32 {
        self.color_subresource >> 8
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
    /// Mip level of `depth_texture` the pass renders depth into.
    #[must_use]
    pub const fn depth_level(&self) -> u32 {
        self.depth_level
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
    pub const fn stencil_load(&self) -> StencilLoad {
        self.stencil_load
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

/// The per-frame inputs `PassState::reset_frame` seeds a new frame from.
///
/// A parameter struct rather than a long argument list: the frame's
/// backbuffer identity, its logical size and format, the depth surface and
/// whether it carries stencil, the back-buffer render scale, and whether this
/// frame continues one that a mid-frame flush interrupted (see
/// [`PassState::reset_frame`]).
pub struct FrameReset {
    pub backbuffer: MetalHandle<MTLTextureKind>,
    /// Logical back-buffer size, the resolution D3D9 reports.
    pub backbuffer_size: (u32, u32),
    pub backbuffer_format: PixelFormat,
    pub depth_texture: MetalHandle<MTLTextureKind>,
    /// Extent of `depth_texture` in its own space; `(0, 0)` when there is none.
    ///
    /// The frame's default depth attachment is created at the rasterized back
    /// buffer's size, so this is `render_scale` of `backbuffer_size`. Passed in
    /// rather than derived here for the same reason `depth_has_stencil` is:
    /// how the attachment was made is the caller's knowledge, not the pass
    /// machine's.
    pub depth_size: (u32, u32),
    pub depth_has_stencil: bool,
    pub render_scale: RenderScale,
    /// `true` when the previous submit was a mid-frame flush, not a `Present`.
    pub continues_frame: bool,
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
    current_color_subresource: u32,
    current_color_size: (u32, u32),
    current_color_format: PixelFormat,
    /// Render targets 1..3 as bound by the device.
    ///
    /// Every entry is `ExtraColorSlot::NONE` on the single-target path,
    /// which keeps `current_extra_present_mask` at zero and every
    /// multi-target branch cold.
    current_extra_color: [ExtraColorSlot; 3],
    /// Bit `i` set ⇒ `current_extra_color[i]` is bound AND matches render target 0's extent.
    ///
    /// Recomputed whenever a colour binding changes; read per draw to key
    /// the pipeline and the PS variant, so the size rule is evaluated once
    /// per bind rather than once per draw.
    current_extra_present_mask: u8,
    /// `current_extra_color` as the pipeline key sees it, rebuilt on every colour bind.
    ///
    /// Cached so a draw copies 16 bytes instead of walking the three slots.
    current_extra_attachments: ExtraColorAttachments,
    current_depth_texture: MetalHandle<MTLTextureKind>,
    /// Mip level of `current_depth_texture` bound as the depth attachment.
    current_depth_level: u32,
    /// Extent of the bound depth attachment's mip level, in its own space.
    ///
    /// `(0, 0)` when nothing is attached. Held beside the handle the way
    /// `current_color_size` is held beside `current_color_texture`, so
    /// `viewport_covers_depth_attachment` can answer whether a whole-target
    /// depth `Clear` may fold into the pass's load action.
    current_depth_size: (u32, u32),
    /// Descriptor bits for the currently bound colour/depth attachments.
    ///
    /// `COLOR_HAS_ALPHA` / `DEPTH_SAMPLEABLE` / `DEPTH_HAS_STENCIL`. See
    /// `CurrentAttachmentFlags` for the per-bit semantics.
    current_attachments: CurrentAttachmentFlags,

    pending_color_clear: Option<(u32, u32, u32, u32)>,
    pending_depth_clear: Option<u32>,
    pending_stencil_clear: Option<u32>,

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

    /// Color-attachment textures that have already been used as an rt this D3D9 frame.
    ///
    /// Inserted at `ensure_pass_open`. First-use opens the door to
    /// `ColorLoad::DontCare` (Rule A) — subsequent uses default to `Load`
    /// so accumulated draws survive across pass breaks. Consulted only by the
    /// load/store rules (Rule A first-use, the finalisers), which reason about
    /// the whole D3D9 frame, so a mid-frame flush does *not* clear it (the
    /// content it tracks stays in VRAM across the flush). Reset on a real
    /// `Present`. Capacity hint matches a typical frame shape (backbuffer + a
    /// few CSM ping-pong RTs).
    seen_color_rts: FxHashSet<(MetalHandle<MTLTextureKind>, u32)>,
    /// Depth-attachment textures that have already been used as a depth rt this D3D9 frame.
    ///
    /// Same semantics as `seen_color_rts`.
    seen_depth_rts: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// Colour rts that received content in the *current submission segment*.
    ///
    /// Reset on every `reset_frame`, mid-frame flush included, unlike
    /// [`Self::seen_color_rts`]. The clear paths key on this: a cross-pass
    /// `Clear` paints a scissored quad (preserving prior tiles) only when the
    /// target already holds content Metal's full-attachment `loadAction =
    /// Clear` would wipe *within this segment*. After a flush every attachment
    /// is safely stored to VRAM, so a fresh full clear is correct and must fold
    /// — a shared shadow-atlas ping-pong lives inside one segment, so the
    /// preserve-tiles case still fires there.
    seen_color_rts_segment: FxHashSet<(MetalHandle<MTLTextureKind>, u32)>,
    /// Depth rts that received content in the current submission segment.
    ///
    /// Segment-scoped twin of [`Self::seen_depth_rts`]; drives the depth and
    /// stencil cross-pass clear decisions for the same reason.
    seen_depth_rts_segment: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// Textures a queued blit writes this frame (`StretchRect` destinations, mipmap regens).
    ///
    /// Inserted by `push_pending_leading_blit`, which sees every ordered blit
    /// before it is drained into some pass's `leading_blits`. Rule A consults
    /// it: the blit that wrote the texture may sit in an earlier pass's
    /// leading list, not the one that first attaches the texture, so the
    /// attachment's own `leading_blits` are not enough to know that its
    /// content is live. Reset each frame in `reset_frame`.
    blit_written_rts: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// The swap-chain backbuffer texture for this frame, captured in `reset_frame`.
    ///
    /// Rule D (last-use color `Store=DontCare`) exempts this handle so
    /// Present can still read the pixels from VRAM after submit. Also the
    /// left-hand side of [`Self::target_scale`]'s comparison: it is what makes
    /// "is the back buffer bound" a handle identity rather than something the
    /// D3D9 layer has to infer and pass down.
    backbuffer_texture: MetalHandle<MTLTextureKind>,
    /// Fraction of the logical resolution the back buffer is rasterized at.
    ///
    /// Seeded per frame from `FrameData`. Applies to the back buffer alone: a
    /// game-created render target is exactly the size the game asked for, so
    /// coordinates aimed at one are already in that texture's space. See
    /// [`Self::target_scale`].
    render_scale: RenderScale,
    /// The scale of the *currently bound* colour attachment.
    ///
    /// The back buffer's own scale while it is bound, and whatever
    /// `set_color_render_target` was told for a game-created target: one sized
    /// to the back buffer shares its scale, anything else is the identity.
    /// Read by [`Self::target_scale`], which every coordinate conversion goes
    /// through, so the rule lives in exactly one field.
    current_color_scale: RenderScale,
    /// The bound colour attachment's size as D3D9 reports it.
    ///
    /// `current_color_size` is this times [`Self::current_color_scale`]. Held
    /// separately rather than divided back out, because the scale rounds and a
    /// round trip through it would not be exact — and because the encoder's
    /// scoped `StretchRect` pass has to restore the device's binding precisely.
    current_color_logical_size: (u32, u32),
    /// The frame's logical resolution, the one D3D9 reports.
    ///
    /// `backbuffer_texture` is `render_scale` of this. Held alongside the scale
    /// because the pair is what defines the two coordinate spaces; the size a
    /// game-created render target is measured against is this one, not the
    /// rasterized extent.
    backbuffer_logical_size: (u32, u32),
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
    /// sRGB twin view → base texture, for every live twin the encoder created.
    ///
    /// A draw sampling with `D3DSAMP_SRGBTEXTURE=1` binds the twin handle,
    /// but every identity question this module answers — "was this texture
    /// sampled this frame" (rename-at-overlap), "does a later pass read this
    /// attachment" (Clear coalescing, store-action Rules C/D) — is asked with
    /// the base handle. The `emit_command` funnel resolves sampled twins to
    /// their base for the sampled sets, and `pass_reads_texture` consults the
    /// map for its command scans. Maintained by the encoder's texture
    /// create / rename / destroy paths.
    srgb_twin_to_base: FxHashMap<MetalHandle<MTLTextureKind>, MetalHandle<MTLTextureKind>>,
    /// Session-wide set of texture handles that were EVER bound as a sampleable depth attachment.
    ///
    /// The bind runs via
    /// `set_depth_stencil_attachment(_, is_sampleable=true)`. The
    /// `mtld3d::d3d9::cascade=trace` end-of-frame summary uses this to
    /// classify fragment-sample binds: a `SetFragmentTexture` of a
    /// handle in this set is a cascade-depth read, and is counted
    /// into `frame_cascade_samples`. Persistent across frames
    /// (textures are stable resource identities).
    seen_sampleable_depth_textures: FxHashSet<MetalHandle<MTLTextureKind>>,
    /// Per-frame counter: how many caster draws targeted each cascade depth handle.
    ///
    /// Counts the draws made this frame. Incremented in `note_caster_draw`,
    /// drained + cleared by `take_cascade_frame_summary`.
    frame_caster_writes: FxHashMap<MetalHandle<MTLTextureKind>, u32>,
    /// Per-frame counter: how many `SetFragmentTexture` binds of a known cascade depth handle.
    ///
    /// Counts the binds emitted this frame. Incremented in `emit_command`
    /// when the bound texture is in `seen_sampleable_depth_textures`.
    /// Drained by `take_cascade_frame_summary`.
    frame_cascade_samples: FxHashMap<MetalHandle<MTLTextureKind>, u32>,
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
            current_color_subresource: 0,
            current_color_size: (0, 0),
            // Placeholder; `reset_frame` always overwrites this before any
            // pass opens. Chose the dominant backbuffer format rather than
            // adding an `Unknown` variant to `PixelFormat` that would pollute
            // every exhaustive match downstream.
            current_color_format: PixelFormat::Bgra8Unorm,
            current_extra_color: [ExtraColorSlot::NONE; 3],
            current_extra_present_mask: 0,
            current_extra_attachments: ExtraColorAttachments::NONE,
            current_depth_texture: MetalHandle::NULL,
            current_depth_level: 0,
            current_depth_size: (0, 0),
            // Placeholder; `reset_frame` reseeds these for the backbuffer, and
            // every `SetRenderTarget` bind overwrites `COLOR_HAS_ALPHA` via
            // `set_color_rt_has_alpha`. The dominant backbuffer is alpha-bearing
            // (`COLOR_HAS_ALPHA` set), non-sampleable, no stencil.
            current_attachments: CurrentAttachmentFlags::COLOR_HAS_ALPHA,
            pending_color_clear: None,
            pending_depth_clear: None,
            pending_stencil_clear: None,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: 0,
            viewport_height: 0,
            viewport_min_z: 0.0,
            viewport_max_z: 1.0,
            last_emitted_viewport: None,
            pending_leading_blits: Vec::new(),
            seen_color_rts: FxHashSet::with_capacity_and_hasher(4, FxBuildHasher),
            seen_depth_rts: FxHashSet::with_capacity_and_hasher(2, FxBuildHasher),
            seen_color_rts_segment: FxHashSet::with_capacity_and_hasher(4, FxBuildHasher),
            seen_depth_rts_segment: FxHashSet::with_capacity_and_hasher(2, FxBuildHasher),
            blit_written_rts: FxHashSet::with_capacity_and_hasher(2, FxBuildHasher),
            backbuffer_texture: MetalHandle::NULL,
            // Placeholder; `reset_frame` reseeds it from the frame stamp.
            // Identity means a `PassState` that never saw a frame cannot
            // perturb a coordinate.
            render_scale: RenderScale::IDENTITY,
            current_color_scale: RenderScale::IDENTITY,
            current_color_logical_size: (0, 0),
            backbuffer_logical_size: (0, 0),
            seen_sampled_textures: FxHashSet::with_capacity_and_hasher(8, FxBuildHasher),
            frame_sampled_textures: FxHashSet::with_capacity_and_hasher(64, FxBuildHasher),
            srgb_twin_to_base: FxHashMap::with_capacity_and_hasher(8, FxBuildHasher),
            seen_sampleable_depth_textures: FxHashSet::with_capacity_and_hasher(8, FxBuildHasher),
            frame_caster_writes: FxHashMap::with_capacity_and_hasher(8, FxBuildHasher),
            frame_cascade_samples: FxHashMap::with_capacity_and_hasher(8, FxBuildHasher),
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
    ///
    /// `backbuffer_size` is **logical**, the resolution D3D9 reports;
    /// `render_scale` converts it to the size of the texture actually bound.
    /// Callers stay in the game's coordinate space and this is the one place
    /// the two are reconciled.
    ///
    /// `continues_frame` is `true` when the previous submit was a mid-frame
    /// flush (a readback / retention drain, `NO_PRESENT`) rather than a
    /// `Present`. The D3D9 frame the game is drawing did not end there, so the
    /// render targets and depth surface it already wrote keep their content in
    /// VRAM. The per-frame "seen" sets are kept across the boundary so Rule A
    /// loads those attachments on their first use in the continuation instead
    /// of discarding them with `DontCare` (the store side is handled by
    /// `finalize_store_actions` skipping Rules B and D on the flush).
    pub fn reset_frame(&mut self, reset: &FrameReset) {
        let &FrameReset {
            backbuffer,
            backbuffer_size,
            backbuffer_format,
            depth_texture,
            depth_size,
            depth_has_stencil,
            render_scale,
            continues_frame,
        } = reset;
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
        self.render_scale = render_scale;
        self.current_color_scale = render_scale;
        self.current_color_logical_size = backbuffer_size;
        self.backbuffer_logical_size = backbuffer_size;
        self.current_color_texture = backbuffer;
        self.current_color_subresource = 0;
        self.current_color_size = (
            render_scale.dimension(backbuffer_size.0),
            render_scale.dimension(backbuffer_size.1),
        );
        self.current_color_format = backbuffer_format;
        // The device re-asserts any extra render targets it holds into the
        // fresh frame, exactly as it does render target 0.
        self.current_extra_color = [ExtraColorSlot::NONE; 3];
        self.current_extra_present_mask = 0;
        self.current_extra_attachments = ExtraColorAttachments::NONE;
        // The backbuffer is an alpha-bearing (`Bgra8Unorm` / A8R8G8B8) target,
        // so destination-alpha blend factors resolve unclamped — byte-identical
        // to the pre-`COLOR_HAS_ALPHA` behaviour. A sub-frame `SetRenderTarget`
        // to an X8 surface overrides this via `set_color_rt_has_alpha`.
        self.current_attachments
            .insert(CurrentAttachmentFlags::COLOR_HAS_ALPHA);
        self.current_depth_texture = depth_texture;
        self.current_depth_level = 0;
        self.current_depth_size = depth_size;
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
        self.pending_stencil_clear = None;
        self.pending_leading_blits.clear();
        // Keep the frame-scoped seen-rt sets across a mid-frame flush: the D3D9
        // frame continues, so the targets already drawn keep their VRAM content
        // and their first use in the continuation must Load, not `DontCare`. On
        // a real `Present` (`continues_frame` false) the frame ended and every
        // target starts fresh. The segment-scoped sets always reset: after the
        // flush every attachment is stored, so a fresh full clear is correct
        // and must fold rather than paint a scissored quad.
        if !continues_frame {
            self.seen_color_rts.clear();
            self.seen_depth_rts.clear();
        }
        self.seen_color_rts_segment.clear();
        self.seen_depth_rts_segment.clear();
        self.blit_written_rts.clear();
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

    /// Extent of the bound depth attachment's mip level, `(0, 0)` when unbound.
    ///
    /// Exposed so a save/restore around a one-off pass (a scoped
    /// `StretchRect` or extra-target clear) rebinds the attachment with the
    /// size it came in with.
    #[must_use]
    pub const fn current_depth_size(&self) -> (u32, u32) {
        self.current_depth_size
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

    /// Render targets 1..3 as the next pass will attach them, for the pipeline key.
    #[must_use]
    pub const fn extra_color_attachments(&self) -> ExtraColorAttachments {
        self.current_extra_attachments
    }

    /// Bit `i` set ⇒ render target `i + 1` takes part in the next pass.
    #[must_use]
    pub const fn extra_present_mask(&self) -> u8 {
        self.current_extra_present_mask
    }

    /// `true` when any of render targets 1..3 is bound, whether or not it matches target 0.
    #[must_use]
    pub fn has_extra_color_targets(&self) -> bool {
        self.current_extra_color
            .iter()
            .any(ExtraColorSlot::is_bound)
    }

    /// `true` when a render target 1..3 is bound but sized unlike target 0.
    ///
    /// Such a target is attached to no pass and is still owed every
    /// `Clear`; the encoder clears it on its own.
    #[must_use]
    pub fn has_extra_color_targets_outside_pass(&self) -> bool {
        self.current_extra_color
            .iter()
            .enumerate()
            .any(|(i, slot)| slot.is_bound() && self.current_extra_present_mask & (1 << i) == 0)
    }

    /// Bind or unbind render target `slot` (1..=3) for the next pass.
    ///
    /// Mirrors `set_color_render_target_subresource`: a rebind of the same
    /// texture and subresource refreshes the size and format in place, any
    /// other change materialises a pending colour clear and ends the pass.
    /// `slot.logical_size` is the D3D9-reported extent; the stored `size` is
    /// what `slot.scale` makes of it.
    pub fn set_extra_color_render_target(&mut self, slot: usize, binding: Option<ExtraColorSlot>) {
        let mut binding = binding.unwrap_or(ExtraColorSlot::NONE);
        binding.size = (
            binding.scale.dimension(binding.logical_size.0),
            binding.scale.dimension(binding.logical_size.1),
        );
        let index = slot - 1;
        let current = &self.current_extra_color[index];
        if current.texture == binding.texture && current.subresource == binding.subresource {
            self.current_extra_color[index] = binding;
            self.recompute_extra_present_mask();
            return;
        }
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            trace!(
                target: TRACE_TARGET,
                "pass-break trigger=set_color_rt{slot} prev={:#x} new={:#x} new_size={}x{}",
                current.texture,
                binding.texture,
                binding.size.0,
                binding.size.1,
            );
        }
        if self.pending_color_clear.is_some() {
            self.flush_pending_clears();
        }
        self.end_current_pass("set_color_rt_extra");
        self.current_extra_color[index] = binding;
        self.recompute_extra_present_mask();
    }

    /// Take the whole colour binding set off the state, leaving render target 0 alone bound.
    ///
    /// Pairs with [`Self::restore_color_attachments`]. Ends the current pass
    /// when extras were bound, since a pass records its attachment set at
    /// open and the caller is about to bind targets of its own.
    pub fn take_color_attachments(&mut self) -> SavedColorAttachments {
        let saved = SavedColorAttachments {
            texture: self.current_color_texture,
            slice: self.current_color_subresource & 0xff,
            level: self.current_color_subresource >> 8,
            logical_size: self.current_color_logical_size,
            format: self.current_color_format,
            scale: self.current_color_scale,
            has_alpha: self.current_color_rt_has_alpha(),
            extra: core::mem::replace(&mut self.current_extra_color, [ExtraColorSlot::NONE; 3]),
        };
        if saved.extra.iter().any(ExtraColorSlot::is_bound) {
            if self.pending_color_clear.is_some() {
                self.flush_pending_clears();
            }
            self.end_current_pass("take_color_attachments");
        }
        self.recompute_extra_present_mask();
        saved
    }

    /// Put back a binding set taken by [`Self::take_color_attachments`].
    ///
    /// Render target 0 goes through the ordinary setter (which ends the pass
    /// when it differs from what is bound); the extras end it when they
    /// differ from the current, usually empty, set.
    pub fn restore_color_attachments(&mut self, saved: SavedColorAttachments) {
        self.set_color_render_target_subresource(
            saved.texture,
            saved.logical_size.0,
            saved.logical_size.1,
            saved.format,
            saved.scale,
            (saved.slice, saved.level),
        );
        self.set_color_rt_has_alpha(saved.has_alpha);
        if self.current_extra_color != saved.extra {
            if self.pending_color_clear.is_some() {
                self.flush_pending_clears();
            }
            self.end_current_pass("restore_color_attachments");
            self.current_extra_color = saved.extra;
        }
        self.recompute_extra_present_mask();
    }

    /// Re-evaluate which extras take part in a pass: bound and sized like target 0.
    ///
    /// A mismatched target warns once per texture; draws skip it (the D3D9
    /// multiple-render-target rule) while `Clear` still reaches it through
    /// the per-target path.
    fn recompute_extra_present_mask(&mut self) {
        let mut mask = 0u8;
        for (i, slot) in self.current_extra_color.iter().enumerate() {
            if !slot.is_bound() {
                continue;
            }
            if slot.size == self.current_color_size {
                mask |= 1 << i;
            } else {
                mtld3d_shared::log_once_warn_by!(
                    target: crate::LOG_TARGET,
                    key: slot.texture.raw(),
                    "render target {} is {}x{} but render target 0 is {}x{}: draws skip it, clears \
                     still reach it",
                    i + 1,
                    slot.size.0,
                    slot.size.1,
                    self.current_color_size.0,
                    self.current_color_size.1,
                );
            }
        }
        self.current_extra_present_mask = mask;
        let mut has_alpha_mask = 0u8;
        for (i, slot) in self.current_extra_color.iter().enumerate() {
            if slot.has_alpha {
                has_alpha_mask |= 1 << i;
            }
        }
        self.current_extra_attachments = ExtraColorAttachments {
            formats: core::array::from_fn(|i| self.current_extra_color[i].format),
            present_mask: mask,
            has_alpha_mask: has_alpha_mask & mask,
        };
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
        self.note_texture_read(handle);
    }

    /// Record an attachment read that the render-command stream cannot see.
    pub fn note_texture_read(&mut self, handle: MetalHandle<MTLTextureKind>) {
        if !handle.is_null() {
            self.seen_sampled_textures.insert(handle);
        }
    }

    /// Register a live sRGB twin view for base-handle identity resolution.
    ///
    /// Called by the encoder whenever a texture create hands back a twin;
    /// see the `srgb_twin_to_base` field for what the mapping protects.
    pub fn register_srgb_twin(
        &mut self,
        twin: MetalHandle<MTLTextureKind>,
        base: MetalHandle<MTLTextureKind>,
    ) {
        if !twin.is_null() && !base.is_null() {
            self.srgb_twin_to_base.insert(twin, base);
        }
    }

    /// Drop a twin registration when its texture is destroyed or renamed.
    pub fn unregister_srgb_twin(&mut self, twin: MetalHandle<MTLTextureKind>) {
        if !twin.is_null() {
            self.srgb_twin_to_base.remove(&twin);
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
    pub fn effective_viewport(&self) -> (u32, u32, u32, u32) {
        if self.viewport_width != 0 && self.viewport_height != 0 {
            // The stored viewport is the game's own; convert against whatever
            // is bound *now* rather than baking the scale in at `set_viewport`,
            // so a viewport that outlives a render-target change is read in the
            // space of the target it is actually clipping.
            self.target_scale().rect(
                self.viewport_x,
                self.viewport_y,
                self.viewport_width,
                self.viewport_height,
            )
        } else {
            // Already the bound texture's own size, so no conversion.
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
    pub fn viewport_covers_color_attachment(&self) -> bool {
        if self.current_color_texture.is_null() {
            return true;
        }
        let (vpx, vpy, vpw, vph) = self.effective_viewport();
        vpx == 0 && vpy == 0 && vpw >= self.current_color_size.0 && vph >= self.current_color_size.1
    }

    /// True when the current viewport covers (or exceeds) the whole bound depth attachment.
    ///
    /// The depth-stencil mirror of [`Self::viewport_covers_color_attachment`],
    /// answering the same question for a `Clear(NULL rects)` of the depth
    /// and/or stencil plane: cover means the clear may fold to a fast
    /// full-attachment `loadAction = Clear`, and a strict sub-region viewport
    /// means it must be scissored to that region. Greater-or-equal for the
    /// same reason: D3D9 clips the viewport to the attachment, so an oversized
    /// viewport still covers. With no depth attachment bound there is nothing
    /// to bound, so fold.
    #[must_use]
    pub fn viewport_covers_depth_attachment(&self) -> bool {
        if self.current_depth_texture.is_null() {
            return true;
        }
        let (vpx, vpy, vpw, vph) = self.effective_viewport();
        vpx == 0 && vpy == 0 && vpw >= self.current_depth_size.0 && vph >= self.current_depth_size.1
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
        let mut keys: FxHashSet<MetalHandle<MTLTextureKind>> = FxHashSet::with_capacity_and_hasher(
            self.frame_caster_writes.len() + self.frame_cascade_samples.len(),
            FxBuildHasher,
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
            // An sRGB twin bind reads its base texture's storage — record the
            // base too so rename-at-overlap and the store-action rules see the
            // read under the handle they key on.
            if let Some(&base) = self.srgb_twin_to_base.get(&tex) {
                self.seen_sampled_textures.insert(base);
                self.frame_sampled_textures.insert(base);
            }
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
        //
        // Each plane is judged against its own attachment's extent. D3D9 only
        // requires the depth-stencil surface to be at least as large as the
        // render target, so a viewport that covers render target 0 exactly can
        // still leave a larger depth surface partly un-rendered.
        let viewport_covers_color_extent = vpx == 0
            && vpy == 0
            && vpw == self.current_color_size.0
            && vph == self.current_color_size.1;
        let pending_color_clear = self.pending_color_clear.take();
        // Shared by render target 0 and every extra: a pending clear lands on
        // all of them (D3D9 clears every bound target), and the Rule A
        // first-use predicate is evaluated per attachment.
        let color_load_for =
            |texture: MetalHandle<MTLTextureKind>, subresource: u32| match pending_color_clear {
                Some((r, g, b, a)) => ColorLoad::Clear { r, g, b, a },
                None if ENABLE_FIRST_USE_DONTCARE
                    && viewport_covers_color_extent
                    && !texture.is_null()
                    && !self.seen_color_rts.contains(&(texture, subresource))
                    && !self.seen_sampled_textures.contains(&texture)
                    && !self.blit_written_rts.contains(&texture) =>
                {
                    ColorLoad::DontCare
                }
                None => ColorLoad::Load,
            };
        let color_load = color_load_for(self.current_color_texture, self.current_color_subresource);
        let extra_color: [PassColorAttachment; 3] = core::array::from_fn(|i| {
            let slot = &self.current_extra_color[i];
            if self.current_extra_present_mask & (1 << i) == 0 {
                return PassColorAttachment::NONE;
            }
            PassColorAttachment {
                texture: slot.texture,
                subresource: slot.subresource,
                size: slot.size,
                format: slot.format,
                load: color_load_for(slot.texture, slot.subresource),
                store: StoreAction::Store,
            }
        });
        // The depth texture's first use this frame under a viewport that
        // covers it: the Rule A predicate, shared by the depth plane and the
        // stencil plane because both live in that one texture. Coverage is
        // measured against the depth attachment's own extent, greater-or-equal
        // because D3D9 clips the viewport to the render target, so an oversized
        // viewport still covers everything the pass can write.
        let depth_first_use = self.viewport_covers_depth_attachment()
            && !self.current_depth_texture.is_null()
            && !self
                .current_attachments
                .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE)
            && !self.seen_depth_rts.contains(&self.current_depth_texture)
            && !self
                .seen_sampled_textures
                .contains(&self.current_depth_texture)
            && !self.blit_written_rts.contains(&self.current_depth_texture);
        let depth_load = match self.pending_depth_clear.take() {
            Some(value) => DepthLoad::Clear { value },
            None if ENABLE_FIRST_USE_DONTCARE && depth_first_use => DepthLoad::DontCare,
            None => DepthLoad::Load,
        };
        let stencil_load = match self.pending_stencil_clear.take() {
            Some(value) => StencilLoad::Clear { value },
            None if ENABLE_FIRST_USE_STENCIL_DONTCARE && depth_first_use => StencilLoad::DontCare,
            None => StencilLoad::Load,
        };
        if !self.current_color_texture.is_null() {
            let key = (self.current_color_texture, self.current_color_subresource);
            self.seen_color_rts.insert(key);
            self.seen_color_rts_segment.insert(key);
        }
        for attachment in extra_color.iter().filter(|a| a.is_bound()) {
            let key = (attachment.texture, attachment.subresource);
            self.seen_color_rts.insert(key);
            self.seen_color_rts_segment.insert(key);
        }
        if !self.current_depth_texture.is_null() {
            self.seen_depth_rts.insert(self.current_depth_texture);
            self.seen_depth_rts_segment
                .insert(self.current_depth_texture);
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
            color_subresource: self.current_color_subresource,
            color_size: self.current_color_size,
            color_format: self.current_color_format,
            color_load,
            color_store: StoreAction::Store,
            depth_texture: self.current_depth_texture,
            depth_level: self.current_depth_level,
            depth_load,
            stencil_load,
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
            extra_color,
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
                 size={}x{} color_load={:?} depth_load={:?} viewport={vpx},{vpy}+{vpw}x{vph} \
                 extra={:#x}",
                self.current_color_texture,
                self.current_depth_texture,
                self.current_color_size.0,
                self.current_color_size.1,
                color_load,
                depth_load,
                self.current_extra_present_mask,
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
    ///
    /// The blit is also entered into the read/write model the load/store rules
    /// reason over. A texture-to-texture copy reads its source from device
    /// memory after every pass that wrote it, so the source counts as read
    /// (`seen_sampled_textures`, which Rules B/C/D consult before discarding a
    /// store). The destination of any texture-writing blit goes into
    /// `blit_written_rts` so Rule A loads it instead of discarding the copy.
    pub fn push_pending_leading_blit(&mut self, blit: BlitCommand) {
        if BlitCommandType::from_repr(blit.cmd) == Some(BlitCommandType::CopyTextureToTexture)
            && blit.src_handle != 0
        {
            // SAFETY: a texture copy carries a non-null MTLTexture handle in
            // `src_handle`, packed from the encoder's typed cache via `.raw()`.
            let src = unsafe { MetalHandle::<MTLTextureKind>::new(blit.src_handle) };
            self.note_texture_read(src);
        }
        if let Some(dst) = blit_written_texture(&blit) {
            self.blit_written_rts.insert(dst);
        }
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
        if self.pending_color_clear.is_some()
            || self.pending_depth_clear.is_some()
            || self.pending_stencil_clear.is_some()
        {
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
        scale: RenderScale,
    ) {
        self.set_color_render_target_subresource(texture, width, height, format, scale, (0, 0));
    }

    /// Rebind a color attachment slice and mip level for the next pass.
    pub fn set_color_render_target_subresource(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        width: u32,
        height: u32,
        format: PixelFormat,
        scale: RenderScale,
        subresource: (u32, u32),
    ) {
        // `width`/`height` arrive logical, the size D3D9 reports for this
        // target; `scale` says what it is actually rasterized at.
        // `current_color_size` is the real texture extent.
        self.current_color_scale = scale;
        self.current_color_logical_size = (width, height);
        self.warn_if_scale_wasted(width, height, scale);
        let (width, height) = (scale.dimension(width), scale.dimension(height));
        let (slice, level) = subresource;
        let packed_subresource = slice | (level << 8);
        if self.current_color_texture == texture
            && self.current_color_subresource == packed_subresource
        {
            self.current_color_size = (width, height);
            self.current_color_format = format;
            if self.has_extra_color_targets() {
                self.recompute_extra_present_mask();
            }
            return;
        }
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            trace!(
                target: TRACE_TARGET,
                "pass-break trigger=set_color_rt prev={:#x} new={:#x} slice={slice} level={level} new_size={width}x{height}",
                self.current_color_texture,
                texture,
            );
        }
        if self.pending_color_clear.is_some() {
            self.flush_pending_clears();
        }
        self.end_current_pass("set_color_rt");
        self.current_color_texture = texture;
        self.current_color_subresource = packed_subresource;
        self.current_color_size = (width, height);
        self.current_color_format = format;
        if self.has_extra_color_targets() {
            self.recompute_extra_present_mask();
        }
    }

    /// Rebind the depth/stencil attachment for the next pass.
    ///
    /// Mirrors `set_color_render_target`: only flushes pending clears when a
    /// pending *depth* clear exists (depth attachment is about to change). A
    /// solo pending color clear stays pending for the unchanged color
    /// attachment. `size` is the attachment's extent in its own space, `(0, 0)`
    /// for an unbind.
    pub fn set_depth_stencil_attachment(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        size: (u32, u32),
        is_sampleable: bool,
        has_stencil: bool,
    ) {
        self.set_depth_stencil_attachment_level(texture, 0, size, is_sampleable, has_stencil);
    }

    /// Bind mip `level` of `texture` as the depth/stencil attachment.
    ///
    /// A different level of the same texture is a different attachment and
    /// ends the pass the way a different texture does. `size` is that level's
    /// own extent, not level 0's: it is what
    /// [`Self::viewport_covers_depth_attachment`] measures the viewport
    /// against.
    pub fn set_depth_stencil_attachment_level(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        level: u32,
        size: (u32, u32),
        is_sampleable: bool,
        has_stencil: bool,
    ) {
        if is_sampleable && !texture.is_null() {
            self.seen_sampleable_depth_textures.insert(texture);
        }
        // A cascade depth surface saved and restored through
        // `GetDepthStencilSurface` comes back via the `Eager` bind path with
        // `is_sampleable = false` — the returned surface carries
        // `parent_texture = null` — even though the underlying Metal texture is
        // the same sampleable shadow map. Resolve the flag against the
        // session-wide "ever sampleable" set so such a rebind neither breaks
        // the pass (an encoder close/open, Load+Store of every attachment) nor
        // clears Rule B's keep-Store exemption for the cascade.
        let is_sampleable = is_sampleable
            || (!texture.is_null() && self.seen_sampleable_depth_textures.contains(&texture));
        // `has_stencil` is a property of the bound texture's format, so a
        // repeat bind of the same texture carries the same value — fold it
        // before the no-change early-out below.
        self.current_attachments
            .set(CurrentAttachmentFlags::DEPTH_HAS_STENCIL, has_stencil);
        if self.current_depth_texture == texture
            && self.current_depth_level == level
            && self
                .current_attachments
                .contains(CurrentAttachmentFlags::DEPTH_SAMPLEABLE)
                == is_sampleable
        {
            // Same handle and level, so the same extent: a repeat bind carries
            // nothing new for the size either.
            return;
        }
        self.current_attachments
            .set(CurrentAttachmentFlags::DEPTH_SAMPLEABLE, is_sampleable);
        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
            trace!(
                target: TRACE_TARGET,
                "pass-break trigger=set_depth_attach prev={:#x}:{} new={:#x}:{}",
                self.current_depth_texture,
                self.current_depth_level,
                texture,
                level,
            );
        }
        if self.pending_depth_clear.is_some() || self.pending_stencil_clear.is_some() {
            self.flush_pending_clears();
        }
        self.end_current_pass("set_depth_attach");
        self.current_depth_texture = texture;
        self.current_depth_level = level;
        self.current_depth_size = size;
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
            // this encoder (the quad writes every colour target of the
            // pass). Visibility-counting passes (occlusion queries active)
            // fall back to the legacy pass-break: see comment in
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
        // Any colour target of the pass, not just render target 0: the
        // clear-quad writes all of them, so one seen target makes the quad
        // the only safe path for the whole set.
        if self.any_bound_color_target_seen() && self.viewport_width > 0 && self.viewport_height > 0
        {
            let vp = self.effective_viewport();
            // Open the pass first (or take the existing one). When
            // already open with no work, rewrite the load action to
            // `Load` so the clear-quad's scissored write is the only
            // thing that lands in this tile; the previous tile's
            // content survives outside the scissor rect.
            self.ensure_pass_open();
            if let Some(pass) = self.passes.last_mut() {
                if matches!(pass.color_load, ColorLoad::Clear { .. }) {
                    pass.color_load = ColorLoad::Load;
                }
                for attachment in pass.extra_color.iter_mut().filter(|a| a.is_bound()) {
                    if matches!(attachment.load, ColorLoad::Clear { .. }) {
                        attachment.load = ColorLoad::Load;
                    }
                }
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
            for attachment in pass.extra_color.iter_mut().filter(|a| a.is_bound()) {
                attachment.load = ColorLoad::Clear { r, g, b, a };
            }
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
        // first, then the rect quads overwrite the rects. Only when there is
        // NO pending clear must the freshly opened pass load as `Load`: the
        // rect quads write only the rects, so every other pixel shows the
        // load action's result. That covers Rule A's first-use `DontCare` too
        // — a region clear as the frame's first touch of the attachment
        // otherwise presents undefined tile memory outside the rects.
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
        {
            if matches!(
                pass.color_load,
                ColorLoad::Clear { .. } | ColorLoad::DontCare
            ) {
                pass.color_load = ColorLoad::Load;
            }
            for attachment in pass.extra_color.iter_mut().filter(|a| a.is_bound()) {
                if matches!(
                    attachment.load,
                    ColorLoad::Clear { .. } | ColorLoad::DontCare
                ) {
                    attachment.load = ColorLoad::Load;
                }
            }
        }
        self.current_color_format
    }

    /// Open (or reuse) the pass for a depth/stencil `Clear` with explicit `pRects` sub-regions.
    ///
    /// The depth/stencil mirror of [`Self::begin_region_color_clear`]: a
    /// rect-clear can never fold into a whole-attachment `loadAction =
    /// Clear`, so a freshly opened pass loads both planes unless a pending
    /// whole-attachment clear is due to land under the rect quads (which
    /// `ensure_pass_open` has just turned into the load action, and which
    /// must stay). A pass that was already open keeps its committed load
    /// actions. The caller then paints one scissored clear-quad per clipped
    /// rect. Returns whether the pass carries a colour attachment and its
    /// format, which the quad pipeline key needs; `None` when no
    /// depth-stencil is bound (nothing to clear).
    pub fn begin_region_depth_stencil_clear(&mut self) -> Option<(bool, PixelFormat)> {
        if self.current_depth_texture.is_null() {
            return None;
        }
        if self.current_pass_has_counting_visibility() {
            self.end_current_pass("region_depth_clear_vis");
        }
        let was_closed = self.current_pass_closed();
        let had_pending_depth = self.pending_depth_clear.is_some();
        let had_pending_stencil = self.pending_stencil_clear.is_some();
        self.ensure_pass_open();
        if was_closed && let Some(pass) = self.passes.last_mut() {
            if !had_pending_depth
                && matches!(
                    pass.depth_load,
                    DepthLoad::Clear { .. } | DepthLoad::DontCare
                )
            {
                pass.depth_load = DepthLoad::Load;
            }
            if !had_pending_stencil
                && matches!(
                    pass.stencil_load,
                    StencilLoad::Clear { .. } | StencilLoad::DontCare
                )
            {
                pass.stencil_load = StencilLoad::Load;
            }
        }
        Some((
            !self.current_color_texture.is_null(),
            self.current_color_format,
        ))
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
        if depth_texture.is_null() {
            // Nothing is attached to clear. Folding would carry the clear
            // onto whatever texture the next pass attaches.
            return DepthClearOutcome::NoOp;
        }
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
    /// (caller falls through to the cross-pass / amend chain). A zero-area
    /// viewport is an explicit `NoOp`: D3D9 clears nothing, and a zero-size
    /// quad would only cost the state switches around it.
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
        if vp.2 == 0 || vp.3 == 0 {
            return Some(DepthClearOutcome::NoOp);
        }
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

    /// Apply a stencil clear.
    ///
    /// Mirrors `clear_depth`: fold into the next pass's `loadAction` where
    /// that is observationally identical, and paint a scissored quad where it
    /// is not. Metal's `loadAction = Clear` covers the whole attachment
    /// regardless of viewport, so a mid-frame re-clear of a plane the frame
    /// already drew into has to be a quad or it wipes those tiles. Depth keeps
    /// its own load action throughout, so a stencil-only clear never disturbs
    /// the depth plane the two share.
    pub fn clear_stencil(&mut self, value: u32) -> StencilClearOutcome {
        let depth_texture = self.current_depth_texture;
        if depth_texture.is_null() {
            // Nothing is attached to clear. Folding would carry the clear
            // onto whatever texture the next pass attaches.
            return StencilClearOutcome::NoOp;
        }
        if self.current_pass_has_work()
            && let Some(outcome) = self.clear_stencil_in_active_pass(value)
        {
            return outcome;
        }
        if let Some(outcome) = self.clear_stencil_cross_pass(value, depth_texture) {
            return outcome;
        }
        if let Some(outcome) = self.clear_stencil_amend_open(value) {
            return outcome;
        }
        self.clear_stencil_stash_pending(value)
    }

    /// Active-pass branch: paint into the live encoder.
    ///
    /// Returns `None` only when a visibility-counting query is armed, since
    /// the quad's own fragments would inflate the occlusion counter; that path
    /// ends the pass first, so the folding chain sees a closed pass and cannot
    /// amend one that already holds draws. A zero-area viewport is an explicit
    /// `NoOp`: D3D9 clears nothing, and falling through would fold a
    /// full-attachment clear into this pass, ahead of its recorded draws.
    fn clear_stencil_in_active_pass(&mut self, value: u32) -> Option<StencilClearOutcome> {
        if self.current_pass_has_counting_visibility() {
            self.end_current_pass("clear_stencil_vis_fallback");
            return None;
        }
        let vp = self.effective_viewport();
        if vp.2 == 0 || vp.3 == 0 {
            return Some(StencilClearOutcome::NoOp);
        }
        Some(StencilClearOutcome::EmitQuad {
            value,
            viewport: vp,
            has_color: !self.current_color_texture.is_null(),
            color_format: self.current_color_format,
        })
    }

    /// Cross-pass branch: the plane already carries content from this frame.
    ///
    /// Open the pass with `Load` so the earlier tiles survive, and let the
    /// caller paint only the cleared region.
    fn clear_stencil_cross_pass(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> Option<StencilClearOutcome> {
        if !self.seen_depth_rts_segment.contains(&depth_texture)
            || self.viewport_width == 0
            || self.viewport_height == 0
        {
            return None;
        }
        let vp = self.effective_viewport();
        if vp.2 == 0 || vp.3 == 0 {
            // The game's viewport rounds to nothing at render resolution.
            // Opening a pass for a zero-size quad would only cost an encoder.
            return Some(StencilClearOutcome::NoOp);
        }
        self.ensure_pass_open();
        if let Some(pass) = self.passes.last_mut()
            && matches!(pass.stencil_load, StencilLoad::Clear { .. })
        {
            pass.stencil_load = StencilLoad::Load;
        }
        Some(StencilClearOutcome::EmitQuad {
            value,
            viewport: vp,
            has_color: !self.current_color_texture.is_null(),
            color_format: self.current_color_format,
        })
    }

    /// Amend branch: a pass is open with no draws, so its load action is free.
    ///
    /// A pass that already holds draws is never amended: its load action
    /// runs before those draws, so the clear would land ahead of them.
    fn clear_stencil_amend_open(&mut self, value: u32) -> Option<StencilClearOutcome> {
        if self.current_pass_closed || self.current_pass_has_work() {
            return None;
        }
        let pass = self.passes.last_mut()?;
        pass.stencil_load = StencilLoad::Clear { value };
        self.pending_stencil_clear = None;
        Some(StencilClearOutcome::Folded)
    }

    /// Stash branch: no pass to amend, so the next `ensure_pass_open` takes it.
    const fn clear_stencil_stash_pending(&mut self, value: u32) -> StencilClearOutcome {
        self.pending_stencil_clear = Some(value);
        StencilClearOutcome::Folded
    }

    /// Fallback for when the clear-quad pipeline cannot be built.
    ///
    /// Ends the pass so the next one carries the clear in its load action.
    pub fn clear_stencil_legacy_break(&mut self, value: u32) {
        self.end_current_pass("clear_stencil_legacy_fallback");
        self.pending_stencil_clear = Some(value);
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
        if !self.seen_depth_rts_segment.contains(&depth_texture)
            || self.viewport_width == 0
            || self.viewport_height == 0
        {
            return None;
        }
        let vp = self.effective_viewport();
        if vp.2 == 0 || vp.3 == 0 {
            // The game's viewport rounds to nothing at render resolution.
            // Opening a pass for a zero-size quad would only cost an encoder.
            return Some(DepthClearOutcome::NoOp);
        }
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
    /// `Clear` and clear any pending fallback. A pass that already holds
    /// draws is never amended: its load action runs before those draws, so
    /// the clear would land ahead of them.
    fn clear_depth_amend_open(
        &mut self,
        value: u32,
        depth_texture: MetalHandle<MTLTextureKind>,
    ) -> Option<DepthClearOutcome> {
        if self.current_pass_closed || self.current_pass_has_work() {
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

    /// `true` when rt 0 or a bound extra already has content in this submission segment.
    ///
    /// Segment-scoped, not frame-scoped: a full `loadAction = Clear` only wipes
    /// content in the current encoder chain, so a clear on a target last drawn
    /// before a mid-frame flush (already stored to VRAM) correctly folds to a
    /// full clear rather than a scissored quad.
    fn any_bound_color_target_seen(&self) -> bool {
        let rt0_seen = !self.current_color_texture.is_null()
            && self
                .seen_color_rts_segment
                .contains(&(self.current_color_texture, self.current_color_subresource));
        rt0_seen
            || self
                .current_extra_color
                .iter()
                .enumerate()
                .any(|(i, slot)| {
                    self.current_extra_present_mask & (1 << i) != 0
                        && self
                            .seen_color_rts_segment
                            .contains(&(slot.texture, slot.subresource))
                })
    }

    /// Color mirror of `clear_depth_legacy_break`.
    pub fn clear_color_legacy_break(&mut self, r: u32, g: u32, b: u32, a: u32) {
        self.end_current_pass("clear_color_legacy_fallback");
        self.pending_color_clear = Some((r, g, b, a));
    }

    /// Warn once when a near-full-screen target renders at full resolution anyway.
    ///
    /// A game-created target inherits the back buffer's scale only when it was
    /// created at exactly the reported back-buffer size. One that is merely
    /// *close* to it — a scene target rounded to a power of two, say — misses
    /// that test and still costs full price, so `render.scale` buys much less
    /// than the setting implies. Silence there reads as "the knob did nothing",
    /// which is the one failure mode a user cannot diagnose from the output.
    ///
    /// Kept free of false positives by construction: it needs a non-default
    /// scale, a target that did *not* inherit it, and coverage of most of the
    /// back buffer on *both* axes. Shadow maps, glow chains and every other
    /// sub-size intermediate stay quiet. Fires once per process.
    fn warn_if_scale_wasted(&self, width: u32, height: u32, scale: RenderScale) {
        /// Percent of each back-buffer axis a target must cover to count as full-screen.
        ///
        /// Below this it is an intermediate, not the scene.
        const FULL_SCREEN_PERCENT: u64 = 90;

        if self.render_scale.is_identity() || !scale.is_identity() {
            return;
        }
        let (bw, bh) = self.backbuffer_logical_size;
        let covers = |extent: u32, full: u32| {
            full != 0 && u64::from(extent) * 100 >= u64::from(full) * FULL_SCREEN_PERCENT
        };
        if covers(width, bw) && covers(height, bh) {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "render.scale = {}% saves less than it looks like here: the game renders into its \
                 own {width}x{height} target, which is close to the {bw}x{bh} back buffer but not \
                 equal to it, so it keeps its own size and rasterizes at full resolution",
                self.render_scale.percent(),
            );
        }
    }

    /// The scale to apply to a game-supplied coordinate for the *currently bound* target.
    ///
    /// `render.scale` shrinks the back buffer alone, so this is the back
    /// buffer's scale while it is bound and the identity otherwise. Keying on
    /// handle identity (rather than on whether the D3D9 layer happens to hold a
    /// null render-target pointer) is what makes the rule hold through an
    /// explicit `SetRenderTarget` back to the back buffer.
    #[must_use]
    pub const fn target_scale(&self) -> RenderScale {
        self.current_color_scale
    }

    /// The bound colour attachment's size as D3D9 reports it, with its scale.
    ///
    /// Pairs with [`Self::target_scale`] for a caller that binds a target of
    /// its own and must put the device's binding back exactly as it was.
    #[must_use]
    pub const fn current_color_logical_size(&self) -> (u32, u32) {
        self.current_color_logical_size
    }

    /// Resolve the `(x, y, w, h)` rect that `emit_scissor` would emit for the given inputs.
    ///
    /// Exposed so the encoder wrapper can dedup against the *resolved*
    /// rect — when scissor test is disabled, the rect falls back to the
    /// current viewport, which can change mid-pass.
    ///
    /// `rect` arrives in the game's coordinate space and comes back in the
    /// bound texture's, so the dedup upstream compares post-conversion rects.
    #[must_use]
    pub fn resolved_scissor_rect(&self, test_enable: bool, rect: [u32; 4]) -> (u32, u32, u32, u32) {
        if test_enable && rect[2] != 0 && rect[3] != 0 {
            self.target_scale().rect(rect[0], rect[1], rect[2], rect[3])
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
        // The fields above keep the game's own numbers so a later render-target
        // change re-reads them in the new target's space; only what reaches
        // Metal is converted.
        let rect = self.target_scale().rect(x, y, width, height);
        self.emit_viewport_if_changed(rect, min_z, max_z);
    }

    /// Override just the depth range Metal holds, keeping the viewport rect.
    ///
    /// `Clear` writes a raw depth value that D3D9's `MinZ`/`MaxZ` do not
    /// touch, but the clear quad writes its value as the vertex's clip-space
    /// z, which Metal's viewport transform would remap. The clear-quad emit
    /// path therefore brackets its draw with `[0, 1]` and then the game's own
    /// range. Only the emitted range moves: the sticky viewport rect and the
    /// coordinate space it is read in stay exactly as the game left them,
    /// which a `set_viewport` round trip could not promise (it takes the
    /// game's rect, and a game that never called `SetViewport` has none).
    pub fn set_emitted_depth_range(&mut self, min_z: f32, max_z: f32) {
        self.viewport_min_z = min_z;
        self.viewport_max_z = max_z;
        let rect = self.effective_viewport();
        self.emit_viewport_if_changed(rect, min_z, max_z);
    }

    /// Push `setViewport` onto the open pass unless the encoder already holds it.
    ///
    /// Re-emit only on an actual change. A fresh `set_viewport` whose value
    /// matches what was last emitted on this encoder would be a redundant
    /// Metal bind (Xcode's "bound … when it was already bound"); the z-range
    /// is part of the key, compared by bits so a depth-range-only change (sky
    /// / weapon, or the clear quad's bracket) still re-emits. `rect` is
    /// already in the bound texture's space, which is what the encoder holds.
    fn emit_viewport_if_changed(&mut self, rect: (u32, u32, u32, u32), min_z: f32, max_z: f32) {
        let (sx, sy, sw, sh) = rect;
        let key = (sx, sy, sw, sh, min_z.to_bits(), max_z.to_bits());
        if !self.current_pass_closed
            && self.last_emitted_viewport != Some(key)
            && let Some(pass) = self.passes.last_mut()
        {
            pass.commands
                .push(Command::set_viewport(sx, sy, sw, sh, min_z, max_z));
            pass.viewport = (sx, sy, sw, sh);
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
            for attachment in pass.extra_color.iter_mut().filter(|a| a.is_bound()) {
                if matches!(attachment.store, StoreAction::DontCare) {
                    if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                        trace!(
                            target: TRACE_TARGET,
                            "pass-strip color={:#x} → dropped (clear-only pass, extra target)",
                            attachment.texture,
                        );
                    }
                    *attachment = PassColorAttachment::NONE;
                }
            }
            if !pass.color_texture.is_null()
                && matches!(pass.color_store, StoreAction::DontCare)
                && !pass.depth_texture.is_null()
            {
                let stripped = pass.color_texture;
                pass.color_texture = MetalHandle::NULL;
                pass.color_subresource = 0;
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
    pub fn strip_color_from_no_color_draw_passes(
        &mut self,
        alt: &FxHashMap<u64, MetalHandle<MTLRenderPipelineStateKind>>,
    ) {
        if !ENABLE_NO_COLOR_PASS_FOR_DRAWS {
            return;
        }
        // Colour textures a later pass that keeps its colour attachment opens
        // with `Load`: a colour clear-quad in an earlier pass is content they
        // observe. Walked in reverse so a later pass that is itself stripped
        // never counts as an observer.
        let mut loaded_later: FxHashSet<MetalHandle<MTLTextureKind>> =
            FxHashSet::with_capacity_and_hasher(4, FxBuildHasher);
        for i in (0..self.passes.len()).rev() {
            let pass = &self.passes[i];
            let record_loads = |loaded_later: &mut FxHashSet<MetalHandle<MTLTextureKind>>| {
                for attachment in pass.bound_color_attachments().iter() {
                    if matches!(pass.color_load_of(attachment.slot), ColorLoad::Load) {
                        loaded_later.insert(attachment.texture);
                    }
                }
            };
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
                record_loads(&mut loaded_later);
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
                record_loads(&mut loaded_later);
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
                record_loads(&mut loaded_later);
                continue;
            }
            // The mask-0 draws write no colour, so dropping the attachment
            // loses nothing of theirs. A colour clear-quad does write: it can
            // only go when no one observes the colour texture afterwards. The
            // back buffer is presented, a texture read back or sampled is
            // seen, a later blit or sampler read sees it, and a later pass
            // that keeps its colour attachment and loads it sees it (the
            // cross-pass colour clear of a later frame region, for one).
            if !cq_ranges.is_empty() {
                let observed_later = pass.bound_color_attachments().iter().any(|attachment| {
                    let tex = attachment.texture;
                    tex == self.backbuffer_texture
                        || self.seen_sampled_textures.contains(&tex)
                        || loaded_later.contains(&tex)
                        || self.passes[i + 1..]
                            .iter()
                            .any(|later| pass_reads_texture(later, tex, &self.srgb_twin_to_base))
                });
                if observed_later {
                    record_loads(&mut loaded_later);
                    continue;
                }
            }
            let pass = &mut self.passes[i];
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
            pass.color_subresource = 0;
            pass.color_load = ColorLoad::DontCare;
            pass.color_store = StoreAction::DontCare;
            // The no-colour twin declares no colour attachment at all, so
            // render targets 1..3 go with target 0.
            pass.extra_color = [PassColorAttachment::NONE; 3];
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
            let color_writes = p
                .bound_color_attachments()
                .iter()
                .any(|a| matches!(a.store, StoreAction::Store));
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
    /// subsequent draw still counts as clear-only here. A pass carrying
    /// leading blits is never a candidate: the blits are real work that the
    /// merge would drop along with the pass.
    pub fn coalesce_clear_only_passes(&mut self) {
        let mut i = 0;
        while i < self.passes.len() {
            let p = &self.passes[i];
            let has_draw = !p.leading_blits.is_empty()
                || p.commands.iter().any(|c| {
                    c.cmd == CommandType::DrawPrimitives as u32
                        || c.cmd == CommandType::DrawIndexedPrimitives as u32
                });
            // Any colour target of the pass with a Clear makes the colour
            // side a candidate; the whole set then moves together.
            let needs_color = !has_draw
                && (matches!(p.color_load, ColorLoad::Clear { .. })
                    || p.extra_color
                        .iter()
                        .any(|a| a.is_bound() && matches!(a.load, ColorLoad::Clear { .. })));
            let needs_depth = !has_draw && matches!(p.depth_load, DepthLoad::Clear { .. });
            let needs_stencil = !has_draw
                && !p.depth_texture.is_null()
                && matches!(p.stencil_load, StencilLoad::Clear { .. });
            if !needs_color && !needs_depth && !needs_stencil {
                i += 1;
                continue;
            }
            let target_color = p.color_texture;
            let target_color_subresource = p.color_subresource;
            let target_extra: [(MetalHandle<MTLTextureKind>, u32); 3] =
                core::array::from_fn(|k| (p.extra_color[k].texture, p.extra_color[k].subresource));
            let target_depth = p.depth_texture;
            let color_load = p.color_load;
            let extra_loads: [ColorLoad; 3] = core::array::from_fn(|k| p.extra_color[k].load);
            let depth_load = p.depth_load;
            let stencil_load = p.stencil_load;
            // Pass has only Clear load actions; no real draws / state.
            // Look ahead for a merge target. Both attachments (if Clear)
            // must match the target pass's attachments AND that pass
            // must currently be Loading them (so the move is observable
            // and lossless). Bail on any intervening read of either.
            let target_idx = self.find_clear_merge_target(
                i,
                &ClearMerge {
                    color: target_color,
                    color_subresource: target_color_subresource,
                    extra: target_extra,
                    depth: target_depth,
                    needs_color,
                    needs_depth,
                    needs_stencil,
                },
            );
            if let Some(t) = target_idx {
                if needs_color {
                    self.passes[t].color_load = color_load;
                    for (k, load) in extra_loads.iter().enumerate() {
                        if self.passes[t].extra_color[k].is_bound() {
                            self.passes[t].extra_color[k].load = *load;
                        }
                    }
                }
                if needs_depth {
                    self.passes[t].depth_load = depth_load;
                }
                if needs_stencil {
                    self.passes[t].stencil_load = stencil_load;
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
    /// whatever we'd move), and on any intervening leading blit that
    /// writes the target: the copy landed after the clear, so a clear moved
    /// past it would wipe it.
    fn find_clear_merge_target(&self, start: usize, want: &ClearMerge) -> Option<usize> {
        let ClearMerge {
            color: target_color,
            color_subresource: target_color_subresource,
            extra: target_extra,
            depth: target_depth,
            needs_color,
            needs_depth,
            needs_stencil,
        } = *want;
        for j in (start + 1)..self.passes.len() {
            let cand = &self.passes[j];
            // Intervening read on a side we care about kills the merge.
            if needs_color && pass_reads_texture(cand, target_color, &self.srgb_twin_to_base) {
                return None;
            }
            if needs_color
                && target_extra.iter().any(|&(tex, _)| {
                    !tex.is_null() && pass_reads_texture(cand, tex, &self.srgb_twin_to_base)
                })
            {
                return None;
            }
            // Intervening blit write on a side we care about kills the merge
            // too: the copy is ordered after our clear.
            if needs_color
                && (blit_list_writes(&cand.leading_blits, target_color)
                    || target_extra
                        .iter()
                        .any(|&(tex, _)| blit_list_writes(&cand.leading_blits, tex)))
            {
                return None;
            }
            if (needs_depth || needs_stencil) && blit_list_writes(&cand.leading_blits, target_depth)
            {
                return None;
            }
            // Render targets 1..3 travel as a set: only a candidate with
            // exactly this set can take the colour clears. One with a
            // different set that attaches any of our extra textures, clearing
            // or loading them, either supersedes or consumes the clear; one
            // that touches none of them is simply not the target.
            let cand_extra: [(MetalHandle<MTLTextureKind>, u32); 3] = core::array::from_fn(|k| {
                (cand.extra_color[k].texture, cand.extra_color[k].subresource)
            });
            let same_set = cand_extra == target_extra;
            if needs_color {
                let touches_extra = target_extra.iter().any(|&(tex, _)| {
                    !tex.is_null()
                        && (cand.color_texture == tex
                            || cand.extra_color.iter().any(|a| a.texture == tex))
                });
                if !same_set && touches_extra {
                    return None;
                }
                if same_set
                    && cand
                        .extra_color
                        .iter()
                        .any(|a| a.is_bound() && !matches!(a.load, ColorLoad::Load))
                {
                    // Same set, but an extra is cleared or first-used there:
                    // that pass supersedes the move for the whole set.
                    return None;
                }
            }
            if (needs_depth || needs_stencil)
                && pass_reads_texture(cand, target_depth, &self.srgb_twin_to_base)
            {
                return None;
            }
            // Intervening Clear on the same attachment supersedes ours.
            if needs_color
                && cand.color_texture == target_color
                && cand.color_subresource == target_color_subresource
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
            if needs_stencil
                && cand.depth_texture == target_depth
                && matches!(cand.stencil_load, StencilLoad::Clear { .. })
            {
                return None;
            }
            // Match: same attachments, currently loading.
            let color_ok = !needs_color
                || (same_set
                    && cand.color_texture == target_color
                    && cand.color_subresource == target_color_subresource
                    && matches!(cand.color_load, ColorLoad::Load));
            let depth_ok = !needs_depth
                || (cand.depth_texture == target_depth
                    && matches!(cand.depth_load, DepthLoad::Load));
            // A `DontCare` candidate cannot occur here: the clear-only pass
            // was this texture's first use of the frame, so every later pass
            // on it opened with `Load` or its own `Clear`.
            let stencil_ok = !needs_stencil
                || (cand.depth_texture == target_depth
                    && matches!(cand.stencil_load, StencilLoad::Load));
            if color_ok && depth_ok && stencil_ok {
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
                && cand.color_subresource == target_color_subresource
                && matches!(cand.color_load, ColorLoad::Load);
            let consumes_depth = needs_depth
                && cand.depth_texture == target_depth
                && matches!(cand.depth_load, DepthLoad::Load);
            let consumes_stencil = needs_stencil
                && cand.depth_texture == target_depth
                && matches!(cand.stencil_load, StencilLoad::Load);
            if consumes_color || consumes_depth || consumes_stencil {
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
        if !ENABLE_FIRST_USE_DONTCARE && !ENABLE_FIRST_USE_STENCIL_DONTCARE {
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
            for attachment in pass.extra_color.iter_mut().filter(|a| a.is_bound()) {
                if matches!(attachment.load, ColorLoad::DontCare)
                    && self.seen_sampled_textures.contains(&attachment.texture)
                {
                    attachment.load = ColorLoad::Load;
                    if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                        trace!(
                            target: TRACE_TARGET,
                            "pass-load color={:#x} DontCare → Load (sampled this frame)",
                            attachment.texture,
                        );
                    }
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
            if matches!(pass.stencil_load, StencilLoad::DontCare)
                && self.seen_sampled_textures.contains(&pass.depth_texture)
            {
                pass.stencil_load = StencilLoad::Load;
                if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                    trace!(
                        target: TRACE_TARGET,
                        "pass-load stencil={:#x} DontCare → Load (sampled this frame)",
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
    ///
    /// `frame_continues` marks a mid-frame flush (a readback or retention drain, not
    /// `Present`): the D3D9 frame keeps going afterwards, so a colour target may still be
    /// read back or drawn into and a depth surface may still be tested against. Both last-use
    /// rules are therefore suppressed — Rule D (colour) and Rule B (depth/stencil) would
    /// discard content the continuation still needs. Rule C (next-clear) still runs, since a
    /// pass that a later pass *in this submission* clears is provably overwritten regardless
    /// of whether the frame ends here.
    pub fn finalize_store_actions(&mut self, frame_continues: bool) {
        if ENABLE_LAST_USE_DEPTH_DONTCARE && !frame_continues {
            let mut handled: FxHashSet<MetalHandle<MTLTextureKind>> =
                FxHashSet::with_capacity_and_hasher(self.seen_depth_rts.len(), FxBuildHasher);
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
            // Value: `(pass index, attachment slot)` of the next use in
            // forward order, so the load action consulted is the one of the
            // slot the texture is bound to there.
            let mut next_color_use: FxHashMap<(MetalHandle<MTLTextureKind>, u32), (usize, usize)> =
                FxHashMap::with_capacity_and_hasher(self.seen_color_rts.len(), FxBuildHasher);
            for i in (0..self.passes.len()).rev() {
                let attachments = self.passes[i].bound_color_attachments();
                for attachment in attachments.iter() {
                    let (slot, rt) = (attachment.slot, attachment.texture);
                    let key = (rt, attachment.subresource);
                    if let Some(&(next, next_slot)) = next_color_use.get(&key)
                        && matches!(
                            self.passes[next].color_load_of(next_slot),
                            ColorLoad::Clear { .. }
                        )
                        && !self.seen_sampled_textures.contains(&rt)
                    {
                        self.passes[i].set_color_store_of(slot, StoreAction::DontCare);
                        if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                            trace!(
                                target: TRACE_TARGET,
                                "pass-store idx={i} color={rt:#x} → DontCare (next-clear at idx={next})",
                            );
                        }
                    }
                    next_color_use.insert(key, (i, slot));
                }
            }
        }
        if ENABLE_LAST_USE_COLOR_DONTCARE && !frame_continues {
            let mut handled: FxHashSet<(MetalHandle<MTLTextureKind>, u32)> =
                FxHashSet::with_capacity_and_hasher(self.seen_color_rts.len(), FxBuildHasher);
            let backbuffer = self.backbuffer_texture;
            for pass in self.passes.iter_mut().rev() {
                let attachments = pass.bound_color_attachments();
                for attachment in attachments.iter() {
                    let (slot, rt, sub) =
                        (attachment.slot, attachment.texture, attachment.subresource);
                    if rt == backbuffer {
                        continue;
                    }
                    if handled.insert((rt, sub))
                        && !matches!(attachment.store, StoreAction::DontCare)
                    {
                        if self.seen_sampled_textures.contains(&rt) {
                            if log_enabled!(target: TRACE_TARGET, Level::Trace) {
                                trace!(
                                    target: TRACE_TARGET,
                                    "pass-store color={rt:#x} → keep Store (sampled this frame)",
                                );
                            }
                        } else {
                            pass.set_color_store_of(slot, StoreAction::DontCare);
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
fn pass_reads_texture(
    pass: &Pass,
    target_handle: MetalHandle<MTLTextureKind>,
    srgb_twin_to_base: &FxHashMap<MetalHandle<MTLTextureKind>, MetalHandle<MTLTextureKind>>,
) -> bool {
    if target_handle.is_null() {
        return false;
    }
    let target_raw = target_handle.raw();
    let sampler_reads = pass.commands.iter().any(|c| {
        if c.cmd != CommandType::SetFragmentTexture as u32 {
            return false;
        }
        if c.param_b == target_raw {
            return true;
        }
        // A bind of the target's sRGB twin view reads the same storage.
        // SAFETY: SetFragmentTexture's param_b holds a non-null MTLTexture
        // handle, packed from the encoder's typed cache via .raw().
        !srgb_twin_to_base.is_empty()
            && srgb_twin_to_base.get(&unsafe { MetalHandle::new(c.param_b) })
                == Some(&target_handle)
    });
    if sampler_reads {
        return true;
    }
    pass.leading_blits
        .iter()
        .any(|b| match BlitCommandType::from_repr(b.cmd) {
            Some(BlitCommandType::CopyTextureToTexture) => b.src_handle == target_raw,
            Some(BlitCommandType::GenerateMipmaps) => b.dst_handle == target_raw,
            _ => false,
        })
}

/// The texture a blit writes, if it writes one.
///
/// `NotifyBufferDidModifyRange` and `CopyBufferToBuffer` carry buffer
/// handles in `src_handle`/`dst_handle`, never texture handles, so they
/// write no texture. An unknown variant on the wire is conservatively
/// treated as texture-writing. The exhaustive match makes any new
/// `BlitCommandType` a compile error here, forcing the author to classify it.
const fn blit_written_texture(blit: &BlitCommand) -> Option<MetalHandle<MTLTextureKind>> {
    let writes_texture = match BlitCommandType::from_repr(blit.cmd) {
        Some(
            BlitCommandType::CopyBufferToTexture
            | BlitCommandType::CopyTextureToTexture
            | BlitCommandType::GenerateMipmaps,
        )
        | None => true,
        Some(BlitCommandType::CopyBufferToBuffer | BlitCommandType::NotifyBufferDidModifyRange) => {
            false
        }
    };
    if !writes_texture || blit.dst_handle == 0 {
        return None;
    }
    // SAFETY: a texture-writing blit carries a non-null MTLTexture handle in
    // `dst_handle`, packed from the encoder's typed cache via `.raw()`.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(blit.dst_handle) })
}

/// True if any blit in `blits` writes to texture `target_handle`.
///
/// Used by Rule E to refuse moving a clear past a pass whose leading blits
/// write the cleared target (`StretchRect`'s typical pattern: copy A → B,
/// then render onto B; a clear folded into that render pass would wipe the
/// copy).
fn blit_list_writes(blits: &[BlitCommand], target_handle: MetalHandle<MTLTextureKind>) -> bool {
    if target_handle.is_null() {
        return false;
    }
    blits
        .iter()
        .any(|b| blit_written_texture(b) == Some(target_handle))
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
    /// Vertex texture fetch slots 0..3.
    ///
    /// `MTLTexture` / `MTLSamplerState` handles bound on the vertex
    /// stage; `0` is the unset sentinel.
    vertex_textures: [u64; VERTEX_SAMPLER_SLOTS],
    vertex_samplers: [u64; VERTEX_SAMPLER_SLOTS],
    pipeline: u64,
    depth_stencil: u64,
    stencil_reference: u32,
    cull_mode: Option<CullMode>,
    /// VS float constant slot — programmable / FF vertex constant buffer.
    vs_constants: Vec<u8>,
    /// VS pos-fixup slot — half-pixel rasterization fixup `(1/vp_w, -1/vp_h, 0, 0)`.
    ///
    /// Re-bound only when the viewport dims change (rare), so the per-draw
    /// cost is a length-then-memcmp against 16 bytes.
    vs_pos_fixup: Vec<u8>,
    /// VS draw slot — the per-draw `VsDraw` uniform (point and clip state).
    ///
    /// Re-bound only when a point state, a clip plane or the view matrix
    /// changes, so the per-draw cost is a length-then-memcmp against
    /// `vs_draw::VS_DRAW_BYTES`.
    vs_draw: Vec<u8>,
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
    /// PS LOD-bias slot: per-sampler-slot `D3DSAMP_MIPMAPLODBIAS`.
    ///
    /// Set only while a bound stage carries a non-zero bias.
    ps_lod_bias: Vec<u8>,
    /// Vertex stream slots 0..16 — bound `MTLBuffer` handle + byte offset each.
    ///
    /// Indexed by D3D9 stream, which is the Metal vertex buffer slot. `(0, _)`
    /// is the unset sentinel (Metal buffer handles are never zero).
    vertex_buffers: [(u64, u32); VERTEX_STREAM_SLOTS as usize],
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
}

impl LastBoundCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fragment_samplers: [0; LAST_BOUND_MAX_STAGES],
            fragment_textures: [0; LAST_BOUND_MAX_STAGES],
            vertex_textures: [0; VERTEX_SAMPLER_SLOTS],
            vertex_samplers: [0; VERTEX_SAMPLER_SLOTS],
            pipeline: 0,
            depth_stencil: 0,
            stencil_reference: 0,
            cull_mode: None,
            vs_constants: Vec::new(),
            vs_pos_fixup: Vec::new(),
            vs_draw: Vec::new(),
            ps_constants: Vec::new(),
            ps_alpha_ref: Vec::new(),
            ps_fog_color: Vec::new(),
            ps_bump_env: Vec::new(),
            ps_lod_bias: Vec::new(),
            vertex_buffers: [(0, 0); VERTEX_STREAM_SLOTS as usize],
            scissor_rect: None,
            blend_color: 0xFFFF_FFFF,
            depth_bias_bits: (0, 0),
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
        self.vertex_textures = [0; VERTEX_SAMPLER_SLOTS];
        self.vertex_samplers = [0; VERTEX_SAMPLER_SLOTS];
        self.pipeline = 0;
        self.depth_stencil = 0;
        self.stencil_reference = 0;
        self.cull_mode = None;
        self.vs_constants.clear();
        self.vs_pos_fixup.clear();
        self.vs_draw.clear();
        self.ps_constants.clear();
        self.ps_alpha_ref.clear();
        self.ps_fog_color.clear();
        self.ps_bump_env.clear();
        self.ps_lod_bias.clear();
        self.vertex_buffers = [(0, 0); VERTEX_STREAM_SLOTS as usize];
        self.scissor_rect = None;
        self.blend_color = 0xFFFF_FFFF;
        self.depth_bias_bits = (0, 0);
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
    pub const fn vertex_texture_changed(&mut self, slot: u32, handle: u64) -> bool {
        let s = &mut self.vertex_textures[slot as usize];
        if *s == handle {
            false
        } else {
            *s = handle;
            true
        }
    }

    #[inline]
    pub const fn vertex_sampler_changed(&mut self, slot: u32, handle: u64) -> bool {
        let s = &mut self.vertex_samplers[slot as usize];
        if *s == handle {
            false
        } else {
            *s = handle;
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

    /// Whether vertex stream `slot` needs a `setVertexBuffer` for `(handle, offset)`.
    ///
    /// Records the binding when it does. `slot` is the D3D9 stream index,
    /// below [`VERTEX_STREAM_SLOTS`].
    #[inline]
    pub const fn vertex_buffer_changed(&mut self, slot: u32, handle: u64, offset: u32) -> bool {
        let cur = &mut self.vertex_buffers[slot as usize];
        if cur.0 == handle && cur.1 == offset {
            false
        } else {
            *cur = (handle, offset);
            true
        }
    }

    /// Forget the vertex buffer bound at stream slot 0.
    ///
    /// Forces the next `vertex_buffer_changed(0, ..)` to report a change.
    /// Call after binding slot 0 with inline bytes
    /// (`setVertexBytes(..., index 0)`): that clobbers the real Metal
    /// vertex-buffer binding while leaving this cache pointing at the
    /// previously bound buffer, so without this a following bound draw
    /// with the same `(handle, offset)` would skip its `setVertexBuffer`
    /// and read the inline payload as vertices. Resets to the `(0, _)`
    /// unset sentinel (Metal buffer handles are never zero).
    #[inline]
    pub const fn invalidate_vertex_buffer(&mut self) {
        self.invalidate_vertex_buffer_slot(0);
    }

    /// Forget the vertex buffer bound at stream `slot`.
    ///
    /// The per-slot form of [`Self::invalidate_vertex_buffer`], for a stream
    /// the draw path fed inline zero bytes because nothing was bound to it.
    #[inline]
    pub const fn invalidate_vertex_buffer_slot(&mut self, slot: u32) {
        self.vertex_buffers[slot as usize] = (0, 0);
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
    pub const fn stencil_reference_changed(&mut self, value: u32) -> bool {
        if self.stencil_reference == value {
            false
        } else {
            self.stencil_reference = value;
            true
        }
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
    pub fn vs_constants_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.vs_constants, bytes)
    }

    #[inline]
    pub fn vs_pos_fixup_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.vs_pos_fixup, bytes)
    }

    #[inline]
    pub fn vs_draw_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.vs_draw, bytes)
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

    #[inline]
    pub fn ps_lod_bias_changed(&mut self, bytes: &[u8]) -> bool {
        update_inline_bytes(&mut self.ps_lod_bias, bytes)
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
/// Last-bound sentinel a null-texture bind records for its texture slot.
///
/// A `SetFragmentNullTexture` command binds the shared opaque-black texture, not
/// a game texture, so the per-slot dedup stores a reserved value — never a Metal
/// handle pointer, and distinct per kind so a slot's declared type changing
/// re-emits. Shared by the draw path and the in-sync shadow so the two agree.
#[must_use]
pub const fn null_texture_tex_sentinel(kind: u64) -> u64 {
    u64::MAX - kind
}

/// Last-bound sentinel for the default sampler a null-texture bind installs.
///
/// Reserved, never a Metal sampler pointer; recorded so a later real sampler
/// bind to the slot is not deduped away.
pub const NULL_TEXTURE_SAMPLER_SENTINEL: u64 = u64::MAX - 8;

#[cfg(debug_assertions)]
#[derive(Default)]
pub struct DebugBoundShadow {
    fragment_samplers: [u64; LAST_BOUND_MAX_STAGES],
    fragment_textures: [u64; LAST_BOUND_MAX_STAGES],
    pipeline: u64,
    depth_stencil: u64,
    /// Raw `CullMode` discriminant (`Command::param_a`).
    cull_mode: Option<u32>,
    /// Per vertex stream slot `(handle, offset)`, `offset` kept as the command's `u64` param.
    vertex_buffers: [(u64, u64); VERTEX_STREAM_SLOTS as usize],
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
    /// bytes, inline vertex bytes at a uniform slot, visibility) are
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
        } else if t == CommandType::SetFragmentNullTexture as u32 {
            // Binds the opaque-black texture + default sampler; mirror the same
            // sentinels the draw path records so the cache and shadow agree.
            self.fragment_textures[cmd.param_a as usize] = null_texture_tex_sentinel(cmd.param_b);
            self.fragment_samplers[cmd.param_a as usize] = NULL_TEXTURE_SAMPLER_SENTINEL;
        } else if t == CommandType::SetScissorRect as u32 {
            self.scissor_rect = Some((cmd.param_a, cmd.param_b, cmd.param_c));
        } else if t == CommandType::SetVertexBuffer as u32 {
            // The cache tracks the vertex stream slots; the uniform slots
            // above them are never bound through `SetVertexBuffer`.
            if cmd.param_a < VERTEX_STREAM_SLOTS {
                self.vertex_buffers[cmd.param_a as usize] = (cmd.param_b, cmd.param_c);
            }
        } else if t == CommandType::SetDepthBias as u32 {
            self.depth_bias = (cmd.param_a, cmd.param_b);
        } else if (t == CommandType::SetVertexBytes as u32
            || t == CommandType::SetVertexBytesAt as u32)
            && cmd.param_a < VERTEX_STREAM_SLOTS
        {
            // An inline bind at a stream slot clobbers the real Metal vertex
            // buffer there; mirror `LastBoundCache::invalidate_vertex_buffer_slot`
            // so both forget it.
            self.vertex_buffers[cmd.param_a as usize] = (0, 0);
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
        for (slot, (&(cache_h, cache_off), &emitted)) in self
            .vertex_buffers
            .iter()
            .zip(&shadow.vertex_buffers)
            .enumerate()
        {
            assert_eq!(
                (cache_h, u64::from(cache_off)),
                emitted,
                "vertex-buffer[{slot}] cache desync (cache vs encoder-emitted)"
            );
        }
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
mod tests;
