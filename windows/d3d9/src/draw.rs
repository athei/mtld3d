//! Shared draw-path closure body for all four `IDirect3DDevice9::Draw*` entry points.
//!
//! Those four are `DrawPrimitive`, `DrawIndexedPrimitive`, `DrawPrimitiveUP`,
//! `DrawIndexedPrimitiveUP`. Each entry point is a thin wrapper that
//! snapshots D3D9 state into a `DrawContext` on the API thread and hands the
//! context to `emit_draw` on the encoder thread.

use core::ptr::NonNull;

use log::{Level, log_enabled};
use mtld3d_core::{
    convert::{
        DecalHeuristicInputs, IMPLICIT_DECAL_BIAS_RAW, IMPLICIT_DECAL_SLOPE_SCALE,
        d3d_depth_bias_to_metal, d3d_to_metal_cull, looks_like_decal,
    },
    depth_stencil_state::{DepthStencilSnapshot, STENCIL_MASK_BITS},
    dirty_range::{indexed_vb_range_lower_bound, nonindexed_vb_range},
    dxso::{
        FfPsKey, FfVsKey, TextureType, VariantFlags, VariantKey, VsSamplerKinds, bound_sampler_type,
    },
    ids::{BufferId, ProgramId},
    passes::{NULL_TEXTURE_SAMPLER_SENTINEL, null_texture_tex_sentinel},
    perf::{CycleAddTimer, OpSub, OpSubDetail, PairShaderId},
    pipeline_state::{PipelineSnapshot, StreamLayout},
    scratch::ScratchArena,
    shader_cache,
    streams::{instance_count, instanced_stream_read_bytes, is_instance_data, stream_step},
    vs_draw::{MAX_CLIP_PLANES, VS_DRAW_BYTES, build_vs_draw_bytes},
};
use mtld3d_shared::{
    Command, NullTextureKind, VertexAttrDesc,
    mtl::{
        IndexType, PS_BOOL_CONST_SLOT, PS_INT_CONST_SLOT, PS_LOD_BIAS_SLOT, PrimitiveType,
        VS_BOOL_CONST_SLOT, VS_DRAW_SLOT, VS_FLOAT_CONST_SLOT, VS_INT_CONST_SLOT,
        VS_POS_FIXUP_SLOT, VertexStepFunction,
    },
};
use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_NEVER, D3DMATRIX, MAX_STREAMS, SAMPLER_STATE_COUNT, render_state_defaults,
};

/// `VsDraw` bytes for the default render states.
///
/// The fallback bind when a snapshot reaches `emit_draw` without its own
/// (never expected; warned once).
static VS_DRAW_DEFAULT: std::sync::LazyLock<[u8; VS_DRAW_BYTES]> = std::sync::LazyLock::new(|| {
    build_vs_draw_bytes(
        &render_state_defaults(),
        &D3DMATRIX::IDENTITY,
        &[[0.0; 4]; MAX_CLIP_PLANES],
    )
});

use super::{encoder::FrameEncoder, stage_bindings::STAGE_COUNT};

/// Sub-target for the per-`(VS, PS, decision)` diagnostic from the implicit-decal-bias site below.
///
/// Sits under `mtld3d::d3d9::*` like the other diag probes;
/// `RUST_LOG=mtld3d::d3d9::decal=trace` opts in without flipping the
/// broader d3d9 logger.
const DECAL_TRACE_TARGET: &str = "mtld3d::d3d9::decal";

/// Per-unique-caster trace target.
///
/// One row per `(depth_tex, vs_hash, ps_hash, alpha_func, alpha_ref_bits,
/// depth_write, blend_enable, cull_mode)` tuple for draws that target a
/// sampleable depth attachment (cascade shadow map). Opt in with
/// `RUST_LOG=mtld3d::d3d9::caster=trace`; `log_once_trace_by!` keeps cost
/// at one cached atomic load when not enabled. Built for diffing
/// caster-pipeline state between two GPU captures when the visual shadow
/// flickers across runs.
const CASTER_TRACE_TARGET: &str = "mtld3d::d3d9::caster";

/// Where the vertex data comes from for this draw.
pub enum VertexSource {
    /// `DrawPrimitiveUP` / `DrawIndexedPrimitiveUP`: inline user pointer.
    ///
    /// Copied into an encoder scratch buffer per draw. `stride` is the
    /// application's `VertexStreamZeroStride` — the true per-vertex span,
    /// which can exceed the vertex declaration's min-extent when the app's
    /// vertex struct carries padding past the declared elements (e.g. a
    /// `FLOAT1` TEXCOORD element over a `float texcoord[4]` field). The
    /// pipeline's vertex layout must step by this stride, not the declaration
    /// extent, or every vertex past the first is fetched from the wrong offset
    /// and the primitive degenerates.
    Up {
        bytes: Vec<u8>,
        size: u32,
        stride: u32,
    },
    /// `DrawPrimitive` / `DrawIndexedPrimitive`: the bound `IDirect3DVertexBuffer9` streams.
    ///
    /// One [`StreamBinding`] per stream the declaration reads that has a
    /// buffer bound; a read stream with nothing bound is absent and feeds
    /// zeros at draw time. `first` is the lowest such stream and `extra` the
    /// rest, so the single-stream case carries no heap allocation (an empty
    /// boxed slice does not allocate).
    Bound {
        first: StreamBinding,
        extra: Box<[StreamBinding]>,
        /// Raw `SetStreamSourceFreq` word of stream 0.
        ///
        /// The instance count of an indexed draw comes from here even when
        /// stream 0 is not among the bound streams.
        stream0_freq: u32,
    },
}

/// One bound vertex stream of a draw.
///
/// `buffer_id` keys the encoder's `MTLBuffer` wrap cache; `backing_ptr` /
/// `backing_len` describe the `PageBox` to wrap if we need a fresh
/// `MTLBuffer`; `offset` is the game's `SetStreamSource` byte offset.
/// `stride` is the game's `SetStreamSource` stride — the true per-vertex
/// span, which can exceed the vertex declaration's extent when the
/// application's vertex struct carries fields past the declared elements.
pub struct StreamBinding {
    /// D3D9 stream index, which is also the Metal vertex buffer slot.
    pub stream: u8,
    pub buffer_id: BufferId,
    pub backing_ptr: u64,
    pub backing_len: u64,
    /// The backing allocation's identity (see `PageBox::generation`).
    pub backing_generation: u64,
    pub offset: u32,
    pub stride: u32,
    /// Raw `SetStreamSourceFreq` word of this stream (flags + count).
    pub freq: u32,
}

/// How one stream is fed for a draw.
enum StreamFeed<'a> {
    /// Stream 0 of a UP draw: inline bytes at the call's stride.
    Inline { stride: u32 },
    /// A bound vertex buffer.
    Buffer(&'a StreamBinding),
    /// Read by the declaration, nothing bound: zeros.
    Null,
}

impl VertexSource {
    /// How `stream` is fed, for a stream the declaration reads.
    fn feed(&self, stream: u32) -> StreamFeed<'_> {
        match self {
            Self::Up { stride, .. } => {
                if stream == 0 {
                    StreamFeed::Inline { stride: *stride }
                } else {
                    StreamFeed::Null
                }
            }
            Self::Bound { first, extra, .. } => core::iter::once(first)
                .chain(extra.iter())
                .find(|b| u32::from(b.stream) == stream)
                .map_or(StreamFeed::Null, StreamFeed::Buffer),
        }
    }

    /// Every bound stream of the draw, lowest first.
    fn bindings(&self) -> impl Iterator<Item = &StreamBinding> {
        let (first, extra) = match self {
            Self::Up { .. } => (None, &[][..]),
            Self::Bound { first, extra, .. } => (Some(first), &extra[..]),
        };
        first.into_iter().chain(extra.iter())
    }
}

/// The stride a stream's vertex buffer layout steps by.
///
/// The application's `SetStreamSource` stride wins when it covers the
/// extent of the declaration elements the shader consumes on that stream
/// (it can exceed it when the vertex struct carries fields past them). A
/// zero stride means the application left it to the declaration. A non-zero
/// stride smaller than the consumed extent means the shader reads an
/// attribute past the end of each vertex — Metal rejects such a pipeline,
/// so the layout is widened to the extent with a warning; the affected
/// draw fetches wrong data either way.
fn layout_stride(app_stride: u32, extent: u32) -> u32 {
    if app_stride == 0 {
        return extent;
    }
    if app_stride < extent {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "stream stride {app_stride} below the consumed declaration extent {extent}; layout widened to the extent"
        );
        return extent;
    }
    app_stride
}

/// Zero bytes fed to a stream the declaration reads but nothing is bound to.
///
/// Bound inline at the stream's slot under a `Constant` layout, so every
/// vertex and instance reads offset 0 and sees zeros, the value a D3D9 vertex
/// reads from an unbound stream. Sized to the `setVertexBytes` limit; a
/// declaration whose extent on one stream exceeds it is not something a
/// 16-element declaration can produce.
static NULL_STREAM_ZEROS: [u8; 4096] = [0; 4096];

/// The vertex buffer layouts of a draw, one per stream the declaration reads.
///
/// Unread streams stay [`StreamLayout::UNUSED`] so the pipeline key is
/// canonical.
fn stream_layouts(
    source: &VertexSource,
    attrs: &AttrSnapshot,
) -> [StreamLayout; MAX_STREAMS as usize] {
    let mut layouts = [StreamLayout::UNUSED; MAX_STREAMS as usize];
    let mut used = attrs.used_streams;
    while used != 0 {
        let stream = used.trailing_zeros();
        used &= used - 1;
        let extent = attrs.extents[stream as usize];
        layouts[stream as usize] = match source.feed(stream) {
            StreamFeed::Inline { stride } => StreamLayout {
                stride: layout_stride(stride, extent),
                step: VertexStepFunction::PerVertex,
                step_rate: 1,
            },
            StreamFeed::Buffer(b) => {
                let (step, step_rate) = stream_step(b.freq);
                StreamLayout {
                    stride: layout_stride(b.stride, extent),
                    step,
                    step_rate,
                }
            }
            StreamFeed::Null => StreamLayout {
                stride: extent,
                step: VertexStepFunction::Constant,
                step_rate: 0,
            },
        };
    }
    layouts
}

/// Where the index data comes from (or whether the draw is non-indexed).
pub enum IndexSource {
    /// Non-indexed draw (`DrawPrimitive` / `DrawPrimitiveUP`).
    None {
        /// First vertex to read from the vertex buffer (always 0 for UP).
        start_vertex: u32,
        vertex_count: u32,
    },
    /// Indexed draw from a bound `IDirect3DIndexBuffer9` (`DrawIndexedPrimitive`).
    ///
    /// Mirrors `VertexSource::Bound` but carries index-stream metadata.
    Bound {
        buffer_id: BufferId,
        backing_ptr: u64,
        backing_len: u64,
        /// The backing allocation's identity (see `PageBox::generation`).
        backing_generation: u64,
        offset: u32,
        index_count: u32,
        index_type: IndexType,
        /// `BaseVertexIndex` from `DrawIndexedPrimitive`.
        ///
        /// Added to every vertex index fetched via the index buffer.
        /// Signed — D3D9 explicitly allows negative values.
        base_vertex: i32,
    },
    /// Non-indexed triangle fan (`DrawPrimitive` / `DrawPrimitiveUP`).
    ///
    /// Metal has no fan primitive. Every non-indexed fan is a prefix of the
    /// same index pattern, `(0, i+1, i+2)` relative to its first vertex, so
    /// the encoder keeps one shared 16-bit buffer of that pattern
    /// (`FrameEncoder::fan_index_buffer`) and the draw passes `start_vertex`
    /// as Metal's base vertex: nothing is generated or staged per draw, and
    /// the vertices are the caller's, bound or inline, untouched. Fans the
    /// 16-bit pattern cannot address take the `Generated` path.
    Fan {
        start_vertex: u32,
        primitive_count: u32,
    },
    /// Indexed draw over a triangle-list index list built in the frame arena.
    ///
    /// Triangle fans whose indices the shared pattern cannot serve: both
    /// indexed entry points, and non-indexed fans past the pattern's reach.
    /// The API thread writes the rewritten list (`convert::FanRewrite`)
    /// straight into `FrameData::scratch`, so the draw stages `index_count`
    /// indices and allocates nothing. The indices are absolute, and
    /// `min_vertex..=max_vertex` is the vertex span they reference, which
    /// gives the exact VB read range when the vertices are bound buffers.
    Generated {
        data: ScratchSlice,
        index_count: u32,
        index_type: IndexType,
        min_vertex: u32,
        max_vertex: u32,
    },
    /// Indexed draw from an inline (user-pointer) index stream (`DrawIndexedPrimitiveUP`).
    ///
    /// The bytes are copied per draw and uploaded via the encoder scratch
    /// arena; the unix side wraps them in a transient `MTLBuffer`. The
    /// indices are absolute (base vertex 0), paired with `VertexSource::Up`.
    Up {
        bytes: Vec<u8>,
        index_count: u32,
        index_type: IndexType,
    },
}

pub struct StageBinding {
    pub texture_id: mtld3d_core::ids::TextureId,
    pub sampler_state: [u32; SAMPLER_STATE_COUNT],
}

// Per-draw stage payload — N of these (popcount of bound-mask) ship in
// every snapshot the API thread bumps into scratch. Keep the budget
// visible so unrelated additions to sampler_state surface here as a
// compile error instead of as silent per-draw bandwidth.
//
// 8 B (TextureId) + 56 B (`[u32; 14]` sampler_state) = 64 B. The encoder
// resolves the Metal texture handle via its `texture_cache` keyed by
// `texture_id` — the full `TextureInfo` is no longer on the per-draw
// path. Cross-device migration (`texture::rehydrate_for_device`) pushes
// a warmup so the cache is populated before the rehydrated texture's
// next bind runs.
const _: () = {
    assert!(
        core::mem::size_of::<StageBinding>() <= 64,
        "StageBinding > 64 B — recheck sampler_state layout"
    );
    // The LOD-bias uniform is indexed by sampler slot, so it needs a row per
    // stage the encoder can bind a texture to.
    assert!(crate::stage_bindings::STAGE_COUNT <= mtld3d_core::sampler_state::LOD_BIAS_SLOTS);
};

/// Per-draw varying parameters.
///
/// Pushed onto the encoder ops Vec as `Op::Draw`; consumed by `emit_draw`
/// which combines them with the encoder's `CurrentSnapshot` to issue the
/// actual GPU commands. State shared with the previous draw (RS, textures,
/// constants, etc.) lives in `CurrentSnapshot` and is updated via separate
/// `Op::Set*` ops only when a dirty bit fires.
pub struct DrawOp {
    pub metal_prim: PrimitiveType,
    pub vertex_source: VertexSource,
    pub index_source: IndexSource,
}

/// Cached vertex-attribute layout.
///
/// A pointer into the frame's scratch plus the metadata `emit_draw` needs
/// to pipeline-key against. Updated via `Op::SetVertexAttrs` when the
/// vertex declaration or FVF changes; reused across draws otherwise.
///
/// `Copy` (pointer + scalars + the per-stream extents) is structurally
/// needed: the `Option<AttrSnapshot>` field on `CurrentSnapshot` is read
/// by-value through a borrowed snapshot (`snap.attrs.expect(...)`). Rust
/// requires `Clone` whenever `Copy` is derived, so `Clone` rides
/// along despite no explicit `.clone()` callers.
#[derive(Clone, Copy)]
pub struct AttrSnapshot {
    pub ptr: NonNull<VertexAttrDesc>,
    pub len: u32,
    /// Per stream, the declaration's `max(offset + size)`; see `ResolvedAttrs`.
    pub extents: [u32; MAX_STREAMS as usize],
    /// Bit `s` set: stream `s` feeds a consumed attribute.
    pub used_streams: u16,
    pub vdecl_hash: u64,
}

// SAFETY: AttrSnapshot.ptr aliases bytes in the per-frame ScratchArena
// owned by the FrameData currently being processed by the encoder.
// CurrentSnapshot lives on FrameEncoder (encoder-thread-only). Send is
// permitted but never actually crossed.
unsafe impl Send for AttrSnapshot {}

impl AttrSnapshot {
    pub const fn as_slice(&self) -> &[VertexAttrDesc] {
        // SAFETY: per type invariant the (ptr, len) refer to a live
        // slice in the current frame's ScratchArena.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }
}

/// Cached pointer to a scratch-allocated `CurrentSnapshot`.
///
/// Wrapped in a newtype so it can be `Copy` + `Send` while making the
/// unsafe deref site explicit at the read.
#[derive(Clone, Copy)]
pub struct CurrentSnapshotPtr(pub NonNull<CurrentSnapshot>);

// SAFETY: see `AttrSnapshot`. The CurrentSnapshot struct lives in the
// current frame's ScratchArena owned by the encoder thread for the
// duration of `run_frame`.
unsafe impl Send for CurrentSnapshotPtr {}

impl CurrentSnapshotPtr {
    /// Raw `*mut CurrentSnapshot` for lifetime-laundered reads inside `emit_draw`.
    ///
    /// Direct `as_ref` is intentionally not provided — `as_ref` returns
    /// `&CurrentSnapshot` whose lifetime is tied to `self`, which in turn
    /// lives on `FrameEncoder` and prevents the usual `&mut enc` reborrows.
    pub const fn as_ptr(&self) -> *mut CurrentSnapshot {
        self.0.as_ptr()
    }
}

/// Cached pointer to a scratch-allocated `RenderStateSnapshot`.
///
/// Wrapped in a newtype so it can be `Copy` while making the unsafe deref
/// site explicit at the read.
#[derive(Clone, Copy)]
pub struct RenderStatePtr(pub NonNull<RenderStateSnapshot>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for RenderStatePtr {}

impl RenderStatePtr {
    pub const fn as_ref(&self) -> &RenderStateSnapshot {
        // SAFETY: per type invariant the pointer refers to a live
        // value in the current frame's ScratchArena.
        unsafe { self.0.as_ref() }
    }
}

/// Cached pointer to a scratch-allocated [`VsSource`].
///
/// Wrapped in a Copy newtype so the per-draw `CurrentSnapshot` carries an
/// 8-byte pointer instead of the ~48-byte enum (which embeds `FfVsKey`);
/// the source is bumped into scratch only when `VS_SOURCE` is dirty.
#[derive(Clone, Copy)]
pub struct VsSourcePtr(pub NonNull<VsSource>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for VsSourcePtr {}

impl VsSourcePtr {
    pub const fn as_ref(&self) -> &VsSource {
        // SAFETY: per type invariant the pointer refers to a live
        // value in the current frame's ScratchArena.
        unsafe { self.0.as_ref() }
    }
}

/// Cached pointer to a scratch-allocated [`PsSource`].
///
/// Same rationale as [`VsSourcePtr`] — keeps the ~56-byte `FfPsKey` out of
/// the per-draw wrapper memcpy.
#[derive(Clone, Copy)]
pub struct PsSourcePtr(pub NonNull<PsSource>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for PsSourcePtr {}

impl PsSourcePtr {
    pub const fn as_ref(&self) -> &PsSource {
        // SAFETY: per type invariant the pointer refers to a live
        // value in the current frame's ScratchArena.
        unsafe { self.0.as_ref() }
    }
}

/// Cached pointer to a scratch-allocated, mask-packed stage-bindings payload.
///
/// The pointee is `[StageBinding; mask.count_ones()]` — only the bound
/// slots are bumped. `mask` bit `b` set means stage `b` is bound; the
/// bindings array stores them in ascending bit-order.
///
/// Replaced the prior `Option<StageBinding>; 16]` flat-array pointee
/// to collapse the per-draw scratch bump from ~2 KB to
/// `2 + popcount × ~116 B` (~120-360 B for typical d3d9 draws that
/// bind 1-3 stages).
#[derive(Clone, Copy)]
pub struct StageBindingsPtr {
    pub mask: u16,
    pub bindings: NonNull<StageBinding>,
}

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for StageBindingsPtr {}

impl StageBindingsPtr {
    /// Iterator over `(stage_index, &StageBinding)` pairs in ascending stage order.
    ///
    /// Skips unbound stages — callers that built handle arrays indexed by
    /// stage must still seed those defaults (`[0; STAGE_COUNT]`) before
    /// iterating.
    pub const fn iter(&self) -> StageBindingsIter<'_> {
        StageBindingsIter {
            mask: self.mask,
            base: self.bindings,
            next_packed_idx: 0,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Iterator over the bound stages of a `StageBindingsPtr`.
///
/// Yields `(stage_index, &StageBinding)` in ascending bit-order of the
/// mask. Lifetime is tied to the source `StageBindingsPtr` borrow.
///
/// `stage_index` is `u32` to match the natural width of
/// `u16::trailing_zeros()`; callers needing `usize` for indexing use
/// `stage as usize`, which is a widening cast on every supported
/// target.
pub struct StageBindingsIter<'a> {
    mask: u16,
    base: NonNull<StageBinding>,
    next_packed_idx: u32,
    _marker: core::marker::PhantomData<&'a StageBinding>,
}

impl<'a> Iterator for StageBindingsIter<'a> {
    type Item = (u32, &'a StageBinding);

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        }
        let stage = self.mask.trailing_zeros();
        self.mask &= self.mask - 1;
        // SAFETY: the packed payload contains `popcount(original_mask)`
        // bindings in ascending bit-order; `next_packed_idx` starts at
        // 0 and is incremented once per yielded slot, so it never
        // overshoots the array length.
        let slot_ptr = unsafe { self.base.as_ptr().add(self.next_packed_idx as usize) };
        // SAFETY: the pointee lives in the current frame's
        // ScratchArena per `StageBindingsPtr`'s type invariant; the
        // returned reference's lifetime is bound to `'a` via the
        // iterator's PhantomData marker.
        let binding =
            unsafe { slot_ptr.as_ref() }.expect("packed StageBinding pointer is non-null");
        self.next_packed_idx += 1;
        Some((stage, binding))
    }
}

/// Bump-allocate an already-packed `StageBinding` prefix into `scratch`.
///
/// Returns a `StageBindingsPtr` referencing the
/// `[StageBinding; popcount(mask)]` payload. `packed` must agree with
/// `mask`: entry `i` is the binding for the `i`-th set bit, ascending
/// (the layout `snapshot_stage_bindings` produces); a debug-assert
/// verifies the length.
///
/// Mask 0 (no bound stages) returns a `dangling()` pointer; readers
/// see an iter that immediately returns `None`, so the pointer is
/// never dereferenced.
///
/// # Safety
///
/// `StageBinding` is currently not `Copy`, but its fields are all
/// integer/enum/bitflag primitives with trivial `Drop`. The bytewise
/// scratch copy never has its own drop run; callers must ensure no
/// field gains a non-trivial `Drop`.
pub unsafe fn bump_packed_stage_bindings(
    scratch: &mut ScratchArena,
    mask: u16,
    packed: &[StageBinding],
) -> StageBindingsPtr {
    debug_assert_eq!(
        packed.len(),
        mask.count_ones() as usize,
        "packed prefix length must equal popcount(mask)"
    );
    if mask == 0 {
        return StageBindingsPtr {
            mask: 0,
            bindings: NonNull::dangling(),
        };
    }
    let dst = scratch.alloc_uninit_slice::<StageBinding>(packed.len());
    // SAFETY: `alloc_uninit_slice` reserved `packed.len()` consecutive
    // `StageBinding` slots at `dst`; source and destination are disjoint
    // allocations. The bytewise copy is sound per this function's
    // trivial-Drop contract.
    unsafe { core::ptr::copy_nonoverlapping(packed.as_ptr(), dst, packed.len()) };
    StageBindingsPtr {
        mask,
        bindings: NonNull::new(dst).expect("ScratchArena returned non-null"),
    }
}

bitflags::bitflags! {
    /// Resolved depth/stencil presence for the current render target.
    ///
    /// Lives in `CurrentSnapshot.depth_stencil`; refreshed by the
    /// API thread when the render-target or depth-stencil-surface
    /// changes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DepthStencilFlags: u8 {
        const HAS_DEPTH = 1 << 0;
        const HAS_STENCIL = 1 << 1;
    }
}

/// Encoder-thread state representing what's currently "bound" for `emit_draw`.
///
/// Lives in the per-frame `ScratchArena`; `FrameEncoder` holds an
/// `Option<CurrentSnapshotPtr>` that the `Op::SetCurrentSnapshot` op
/// updates. `emit_draw` borrows `&CurrentSnapshot` once via lifetime
/// laundering and reads fields directly — no struct copies on the encoder
/// side.
///
/// Intentionally NOT `Copy` / `Clone` — accidental whole-struct copies
/// would be a per-draw pessimisation. The large FF keys (`FfVsKey` /
/// `FfPsKey`) live behind `VsSourcePtr` / `PsSourcePtr` scratch pointers
/// rather than inline, so the wrapper that gets memcpy'd per draw is
/// ~160 B (pointers + scalars) and the FF source is only bumped when
/// `VS_SOURCE` / `PS_SOURCE` is dirty.
pub struct CurrentSnapshot {
    pub render_state: Option<RenderStatePtr>,
    pub stage_bindings: Option<StageBindingsPtr>,
    pub attrs: Option<AttrSnapshot>,
    pub vs: Option<VsSourcePtr>,
    pub ps: Option<PsSourcePtr>,
    pub variant: Option<VariantKey>,
    pub vs_constants: Option<ScratchSlice>,
    pub ps_constants: Option<ScratchSlice>,
    pub alpha_ref_bytes: Option<ScratchSlice>,
    pub fog_color_bytes: Option<ScratchSlice>,
    /// Per-stage bump-environment matrix + luminance (PS slot 12).
    ///
    /// Consumed by SM1 `texbem`/`texbeml`/`bem`. Bound only for a PS that
    /// uses one of those ops (`PsSource::Programmable::uses_bump_env`).
    pub bump_env_bytes: Option<ScratchSlice>,
    /// VS integer-constant file (vertex slot 14).
    ///
    /// Bound only for a VS that reads a dynamic integer constant
    /// (`VsSource::Programmable::uses_int_const`).
    pub vs_int_const_bytes: Option<ScratchSlice>,
    /// VS boolean-constant bitmask (vertex slot 26).
    ///
    /// Bound only for a VS that reads a dynamic boolean constant
    /// (`VsSource::Programmable::uses_bool_const`).
    pub vs_bool_const_bytes: Option<ScratchSlice>,
    /// PS integer-constant file (fragment slot 11).
    ///
    /// Bound only for a PS that reads a dynamic integer constant
    /// (`PsSource::Programmable::uses_int_const`).
    pub ps_int_const_bytes: Option<ScratchSlice>,
    /// PS boolean-constant bitmask (fragment slot 10).
    ///
    /// Bound only for a PS that reads a dynamic boolean constant
    /// (`PsSource::Programmable::uses_bool_const`).
    pub ps_bool_const_bytes: Option<ScratchSlice>,
    /// Per-draw `VsDraw` uniform (`mtld3d_core::vs_draw`): point size state.
    ///
    /// Every vertex shader reads it, so it is bound for every draw and
    /// rebuilt when a point render state changes (`SnapshotDirty::VS_DRAW`).
    pub vs_draw_bytes: Option<ScratchSlice>,
    pub depth_stencil: DepthStencilFlags,
}

impl CurrentSnapshot {
    /// Initial all-`None` state.
    ///
    /// Used to seed `DeviceInner::snapshot_cache` before any rebuild has
    /// populated the fields.
    pub const EMPTY: Self = Self {
        render_state: None,
        stage_bindings: None,
        attrs: None,
        vs: None,
        ps: None,
        variant: None,
        vs_constants: None,
        ps_constants: None,
        alpha_ref_bytes: None,
        fog_color_bytes: None,
        bump_env_bytes: None,
        vs_int_const_bytes: None,
        vs_bool_const_bytes: None,
        ps_int_const_bytes: None,
        ps_bool_const_bytes: None,
        vs_draw_bytes: None,
        depth_stencil: DepthStencilFlags::empty(),
    };
}

/// Content-addressed identity for a compiled VS library.
///
/// Same content hash or same FF key → same cache entry, so
/// destroy/recreate of identical shader objects hits the cache.
///
/// Neither `Copy` nor `Clone`: at 42+ B for the FF variant, accidental
/// whole-struct reads (e.g. pattern-binding by value) were silent memcpys.
/// `vs.key(variant)` builds an owned key from a borrowed `VsSource` by
/// cloning the embedded `FfVsKey`, so the enum itself never needs a derive.
#[derive(PartialEq, Eq, Hash)]
pub enum VsKey {
    Programmable {
        vs_id: ProgramId,
        variant: VariantKey,
        /// Part of the key so a missing-attribute variant gets its own compiled library.
        ///
        /// See `VsSource::Programmable::provided_input_mask`.
        provided_input_mask: u16,
        /// See `VsSource::Programmable::clip_plane_count`.
        clip_plane_count: u8,
        /// See `VsSource::Programmable::sampler_kinds`.
        sampler_kinds: VsSamplerKinds,
    },
    FixedFunction {
        ff: FfVsKey,
        variant: VariantKey,
    },
}

/// Content-addressed identity for a compiled PS library.
///
/// Same shape as `VsKey` but for the pixel stage. Same Copy / Clone
/// rationale.
#[derive(PartialEq, Eq, Hash)]
pub enum PsKey {
    Programmable {
        ps_id: ProgramId,
        variant: VariantKey,
    },
    FixedFunction {
        ff: FfPsKey,
        variant: VariantKey,
    },
}

impl VsKey {
    /// On-disk shader-cache identifier.
    ///
    /// Exactly the value `FrameEncoder::resolve_vs_library` keys `lib_cache`
    /// on, so the same u64 also drives `CachedKind::entry_name` (Xcode
    /// pipeline label) and `debug.bytecodeDumpDir`'s `vs_<hash>.dxso`
    /// filename.
    pub fn disk_key(&self) -> u64 {
        match self {
            Self::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => vs_source_disk_key_programmable(
                *vs_id,
                *provided_input_mask,
                *clip_plane_count,
                *sampler_kinds,
            ),
            Self::FixedFunction { ff, .. } => vs_source_disk_key_ff(ff),
        }
    }

    /// Type-erased identity for the perf module's per-pair stats and pass-shader dedup set.
    ///
    /// `hash` is the on-disk shader-cache key (`disk_key`), so the value
    /// printed in the pass×shader log (`pass RT … VS prog/ff 0x…`) matches
    /// the truncated hex baked into the Metal entry-point name
    /// (`mtld3d_vs_*_<8hex>`) shown by Xcode's Pipeline State inspector and
    /// Frame Capture timeline.
    pub fn pair_id(&self) -> PairShaderId {
        PairShaderId {
            is_programmable: matches!(self, Self::Programmable { .. }),
            hash: self.disk_key(),
        }
    }
}

impl PsKey {
    pub fn disk_key(&self) -> u64 {
        match self {
            Self::Programmable { ps_id, variant } => {
                ps_source_disk_key_programmable(*ps_id, *variant)
            }
            Self::FixedFunction { ff, variant } => ps_source_disk_key_ff(ff, *variant),
        }
    }

    pub fn pair_id(&self) -> PairShaderId {
        PairShaderId {
            is_programmable: matches!(self, Self::Programmable { .. }),
            hash: self.disk_key(),
        }
    }
}

/// The black-fallback texture of the kind an emitted sampler argument carries.
///
/// Both stages type their `[[texture(n)]]` arguments from the texture bound to
/// the slot, so a slot the draw cannot resolve to a real texture has to fall
/// back to a black texture of that same kind or Metal rejects the binding.
const fn null_texture_kind(ty: TextureType) -> NullTextureKind {
    match ty {
        TextureType::Texture3D => NullTextureKind::Texture3D,
        TextureType::TextureCube => NullTextureKind::TextureCube,
        TextureType::Texture2D | TextureType::Unknown => NullTextureKind::Texture2D,
    }
}

/// Single source of truth for the VS programmable `disk_key`.
///
/// The provided-input mask IS folded in: a shader reading an unprovided
/// input emits different MSL (the input becomes `float4(0)` and `VertexIn`
/// drops the attribute), so each `(vs_id, mask)` compiles a distinct
/// `MTLLibrary`. So is the user clip plane count: it sizes the
/// `[[clip_distance]]` output and the distances the epilogue computes. So is
/// the vertex-fetch sampler kind mask: it types the `[[texture(n)]]` arguments
/// and picks the sample coordinate swizzle. Variant bits still don't change VS
/// MSL, so they are intentionally left out.
pub fn vs_source_disk_key_programmable(
    vs_id: ProgramId,
    provided_input_mask: u16,
    clip_plane_count: u8,
    sampler_kinds: VsSamplerKinds,
) -> u64 {
    shader_cache::ff_key_hash(&(
        vs_id.raw(),
        provided_input_mask,
        clip_plane_count,
        sampler_kinds,
    ))
}

pub fn vs_source_disk_key_ff(ff: &FfVsKey) -> u64 {
    shader_cache::ff_key_hash(ff)
}

/// Single source of truth for the PS `disk_key`.
///
/// Variant bits ARE folded in: PS variants (alpha-test mode, fog mode)
/// produce different MSL, so each `(source, variant)` compiles to a
/// distinct `MTLLibrary`.
pub fn ps_source_disk_key_programmable(ps_id: ProgramId, variant: VariantKey) -> u64 {
    shader_cache::ff_key_hash(&(ps_id.raw(), variant))
}

pub fn ps_source_disk_key_ff(ff: &FfPsKey, variant: VariantKey) -> u64 {
    shader_cache::ff_key_hash(&(ff, variant))
}

/// Source for the VS stage of a draw.
///
/// `Programmable` carries only the `shader_id`; the parsed `DxsoProgram`
/// lives in the encoder's `program_cache`, populated by the
/// `register_program` op pushed at `CreateVertexShader`. `FixedFunction`
/// carries the FF key.
///
/// Neither `Copy` nor `Clone`: the `FixedFunction` variant carries a 38 B
/// `FfVsKey` and `VsSource` is stored by value in `CurrentSnapshot`. `Copy`
/// would have turned every `let vs = snap.vs.unwrap()` into a silent ~38 B
/// memcpy off scratch. Duplication into scratch goes through
/// `ScratchArena::alloc_from` (a bytewise copy with no `Clone` bound), and
/// `VsSource::key` clones only the embedded `FfVsKey`.
pub enum VsSource {
    Programmable {
        vs_id: ProgramId,
        /// `max_const_used` from the bound shader (rows of `c[]` referenced by static analysis).
        ///
        /// Carried in the snapshot so the encoder can snapshot exactly
        /// `rows × 16` bytes out of its VS const mirror at `emit_draw` time
        /// without going through the encoder-side `program_cache` lookup.
        /// Capped at 256.
        max_const_used: u16,
        /// `true` when the bound shader reads `c[a0.x+N]` (relative addressing).
        ///
        /// Static analysis can't bound the index, so the encoder must bind
        /// the full populated prefix
        /// (`FrameEncoder::vs_constants_populated_rows`) rather than
        /// `max_const_used`.
        uses_rel_const: bool,
        /// Bit `i` set ⇒ VS input register `vi` is provided by the bound vertex declaration.
        ///
        /// Folds into the VS library + disk keys so a shader reading an
        /// unprovided input (read as `float4(0)`) compiles a distinct
        /// variant. All-ones for a fully-provided decl, so real workloads
        /// keep a single variant.
        provided_input_mask: u16,
        /// Shader reads a dynamic integer constant → bind the integer-constant buffer.
        ///
        /// Dynamic here means a non-`defi` `iN`, e.g. a `loop`/`rep` counter
        /// fed by `SetVertexShaderConstantI`; the buffer goes to vertex slot
        /// 14. False for the vast majority of shaders, which then pay no
        /// slot-14 bind.
        uses_int_const: bool,
        /// Whether the VS reads a dynamic boolean constant.
        ///
        /// A non-`defb` `bN`, typically a static `if` condition fed by
        /// `SetVertexShaderConstantB`; the bitmask goes to vertex slot 26.
        uses_bool_const: bool,
        /// User clip planes the draw applies (`vs_draw::clip_plane_count`), 0..=6.
        ///
        /// Folds into the VS library + disk keys: the shader declares one
        /// `[[clip_distance]]` lane per plane and computes it in the
        /// epilogue, so each count is a distinct variant. Zero for every
        /// draw that never enables a plane, which keeps the common case at
        /// one variant.
        clip_plane_count: u8,
        /// Texture kind bound at each of the four vertex texture fetch slots.
        ///
        /// Folds into the VS library + disk keys: the emitter types each
        /// `[[texture(n)]]` argument and its sample coordinate swizzle from
        /// the bound kind rather than from the shader's `dcl_*`, because
        /// Metal type-checks the binding against the signature. Zero for
        /// every shader without vertex texture fetch and for the ordinary
        /// case of a 2D texture in every slot, which keeps one variant.
        sampler_kinds: VsSamplerKinds,
    },
    FixedFunction {
        key: FfVsKey,
        /// Highest row of the FF VS const blob the shader reads, plus 1.
        ///
        /// I.e. the number of rows to snapshot from the encoder's
        /// `ff_vs_constants_mirror` and bind via `setVertexBytes`. Computed
        /// on the API thread from `key` + `FfState` masks at snapshot time;
        /// the same derivation lives inside `build_vs_constants` but is
        /// replicated on the source so `emit_draw` can read it without
        /// re-walking the masks.
        max_row_count: u16,
    },
}

impl VsSource {
    /// Build the `VsKey` (cache lookup identity) for a draw.
    ///
    /// Takes `self` as the source and `variant` as the variant. Clones the
    /// embedded `FfVsKey` once per construction; the resulting `VsKey` owns
    /// the FF key and lives until cache-insert or trace logging consumes it.
    pub fn key(&self, variant: VariantKey) -> VsKey {
        match self {
            Self::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => VsKey::Programmable {
                vs_id: *vs_id,
                variant,
                provided_input_mask: *provided_input_mask,
                clip_plane_count: *clip_plane_count,
                sampler_kinds: *sampler_kinds,
            },
            Self::FixedFunction { key, .. } => VsKey::FixedFunction {
                ff: key.clone(),
                variant,
            },
        }
    }

    /// On-disk content-hash key for this source (variant-independent).
    ///
    /// VS variants share one library. Computed only on a cache miss /
    /// gated-diagnostic path, never per draw. Mirrors `VsKey::disk_key`.
    pub fn disk_key(&self) -> u64 {
        match self {
            Self::Programmable {
                vs_id,
                provided_input_mask,
                clip_plane_count,
                sampler_kinds,
                ..
            } => vs_source_disk_key_programmable(
                *vs_id,
                *provided_input_mask,
                *clip_plane_count,
                *sampler_kinds,
            ),
            Self::FixedFunction { key, .. } => vs_source_disk_key_ff(key),
        }
    }
}

/// Source for the PS stage of a draw.
///
/// Symmetric to `VsSource`; same Copy / Clone rationale.
pub enum PsSource {
    Programmable {
        ps_id: ProgramId,
        /// See [`VsSource::Programmable::max_const_used`].
        max_const_used: u16,
        /// Shader uses `texbem`/`texbeml`/`bem` → bind the bump-environment uniform.
        ///
        /// The per-stage uniform goes to PS slot 12. False for the vast
        /// majority of shaders, which then pay no slot-12 bind.
        uses_bump_env: bool,
        /// Whether the PS reads a dynamic integer constant.
        ///
        /// A non-`defi` `iN`, typically a `rep`/`loop` counter fed by
        /// `SetPixelShaderConstantI`; the file goes to fragment slot 11.
        uses_int_const: bool,
        /// Whether the PS reads a dynamic boolean constant.
        ///
        /// A non-`defb` `bN`, typically a static `if` condition fed by
        /// `SetPixelShaderConstantB`; the bitmask goes to fragment slot 10.
        uses_bool_const: bool,
        /// Bit `i` set ⇒ the bytecode writes `oCi`.
        ///
        /// Feeds the pipeline key so a render target the shader never
        /// writes gets an empty write mask and keeps its contents.
        color_out_mask: u8,
    },
    FixedFunction {
        key: FfPsKey,
        /// Stages the emitted shader declares a texture and sampler for.
        ///
        /// `FfPsKey::sampled_stage_mask`, resolved on the API thread so
        /// `emit_draw` never re-walks the stage array. The draw binds a
        /// texture only inside the mask; a stage outside it is one the
        /// combiner cascade never samples.
        sampled_stage_mask: u16,
        /// Whether the shader reads the fixed-function pixel constant buffer.
        ///
        /// `FfPsKey::reads_texture_factor`. Its one row is
        /// `D3DRS_TEXTUREFACTOR`, so a key that references neither
        /// `D3DTA_TFACTOR` nor `D3DTOP_BLENDFACTORALPHA` binds no constants.
        reads_texture_factor: bool,
    },
}

impl PsSource {
    pub fn key(&self, variant: VariantKey) -> PsKey {
        match self {
            Self::Programmable { ps_id, .. } => PsKey::Programmable {
                ps_id: *ps_id,
                variant,
            },
            Self::FixedFunction { key, .. } => PsKey::FixedFunction {
                ff: key.clone(),
                variant,
            },
        }
    }

    /// On-disk content-hash key for this source + `variant` (PS MSL depends on the variant).
    ///
    /// Computed only on a cache miss / gated-diagnostic path, never per
    /// draw. Mirrors `PsKey::disk_key`.
    pub fn disk_key(&self, variant: VariantKey) -> u64 {
        match self {
            Self::Programmable { ps_id, .. } => ps_source_disk_key_programmable(*ps_id, variant),
            Self::FixedFunction { key, .. } => ps_source_disk_key_ff(key, variant),
        }
    }
}

/// The shader identity for one draw — the VS/PS sources + `variant`, which travel together.
///
/// Passed as a unit to the gated diagnostic / telemetry consumers
/// (`maybe_log_pass_shader`, `maybe_emit_draw_trace`, `bump_pair_stats`) so
/// each builds its `VsKey`/`PsKey` / `PairShaderId` internally only when its
/// gate is open — never on the hot path. `Copy` (two references + a small
/// `VariantKey`).
#[derive(Clone, Copy)]
pub struct ShaderRef<'a> {
    pub vs: &'a VsSource,
    pub ps: &'a PsSource,
    pub variant: VariantKey,
}

/// Owning-by-reference slice into the current `FrameData::scratch` bump arena.
///
/// Replaces `Vec<u8>` for per-draw constants (VS / PS / alpha-ref /
/// fog-color). The API thread bump-allocates the bytes via
/// [`arena_alloc_bytes`] and captures the resulting `ScratchSlice` in
/// the draw closure; the encoder thread reads the bytes via
/// [`ScratchSlice::as_slice`] or hands the raw `(ptr, len)` to a Metal
/// `set_*_bytes_at` command.
///
/// # Safety invariants
///
/// 1. The pointed-to bytes live in the `ScratchArena` owned by the
///    same `FrameData` whose `ops` vec holds the closure containing
///    this `ScratchSlice`.
/// 2. That `FrameData` is not dropped (and its arena not cleared)
///    until the encoder thread has finished running every op-closure
///    on it, so the pointer is valid for the entire window between
///    API-side construction and encoder-side consumption.
/// 3. Each closure is `FnOnce`; the encoder runs it once and drops it
///    before dropping the owning `FrameData`. The slice is never used
///    after the closure exits.
#[derive(Clone, Copy)]
pub struct ScratchSlice {
    ptr: NonNull<u8>,
    len: u32,
}

// SAFETY: `ScratchSlice` is a logically-owning pointer into a frame-
// local arena. It crosses the API→encoder channel inside a draw closure;
// once the encoder receives the closure it has exclusive access to the
// owning `FrameData` (and thus the arena), so concurrent mutation
// through this pointer is impossible by construction.
unsafe impl Send for ScratchSlice {}

impl ScratchSlice {
    pub const EMPTY: Self = Self {
        ptr: NonNull::<u8>::dangling(),
        len: 0,
    };

    /// Construct a `ScratchSlice` from a raw pointer + length.
    ///
    /// Both come back from `ScratchArena::alloc_uninit_slice`. Caller
    /// asserts the pointer is live in a per-frame arena (drops with
    /// `FrameData`) and that the referenced bytes are initialised.
    #[must_use]
    pub const fn from_raw_parts(ptr: NonNull<u8>, len: u32) -> Self {
        Self { ptr, len }
    }

    /// Raw pointer + byte count suitable for the encoder's `set_*_bytes_at` commands.
    ///
    /// Pointer stays valid until the owning `FrameData` is dropped (see
    /// type-level invariants).
    pub fn as_raw(&self) -> (u64, u32) {
        (self.ptr.as_ptr() as u64, self.len)
    }

    /// Slice view for encoder-side byte reads.
    ///
    /// Lifetime of the returned slice is tied to `&self`, so the borrow
    /// checker prevents stash past the closure body that holds `Self`.
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: per type-level invariants, `ptr` points to `len`
        // bytes in a live arena whenever a `ScratchSlice` is in scope.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }
}

/// Copy `bytes` into the given `FrameData::scratch` arena and return a `ScratchSlice` view.
///
/// Empty inputs short-circuit to [`ScratchSlice::EMPTY`] so the encoder
/// skips the bind cleanly.
///
/// # Panics
///
/// Panics if `bytes.len()` exceeds `u32::MAX`. Unreachable —
/// per-draw constant buffers are bounded by the 8 KB VS+PS budget.
pub fn arena_alloc_bytes(scratch: &mut ScratchArena, bytes: &[u8]) -> ScratchSlice {
    if bytes.is_empty() {
        return ScratchSlice::EMPTY;
    }
    let ptr = scratch.alloc(bytes);
    let len = u32::try_from(bytes.len()).expect("constants slice fits u32");
    let nn = NonNull::new(ptr as *mut u8).expect("ScratchArena::alloc returned non-null");
    ScratchSlice { ptr: nn, len }
}

bitflags::bitflags! {
    /// Boolean RS bits that DON'T affect pipeline identity (depth test/write, scissor enable).
    ///
    /// Bits that DO affect pipeline identity live in `PipelineRsFlags`
    /// inside `PipelineRsBits.flags`. Split this way so the cache key
    /// (`PipelineSnapshot`) only hashes pipeline-relevant bits.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DepthScissorFlags: u8 {
        const DEPTH_ENABLE = 1 << 0;
        const DEPTH_WRITE = 1 << 1;
        const SCISSOR_TEST = 1 << 2;
    }
}

/// Render-state snapshot carried from the API thread to the encoder.
///
/// Two flag bytes split by purpose: `pipeline_rs` carries the bits
/// that participate in `MTLRenderPipelineState` identity (blend, color
/// write, sRGB) and is consumed directly by `PipelineSnapshot.rs` —
/// one field copy at draw time, no per-field repack. `depth_scissor`
/// carries the remaining boolean RS that the encoder reads at draw
/// time without going through the pipeline cache.
///
/// D3D9 enum-valued RS (`D3DCMP_*`, `D3DCULL_*`) are `u8`. The
/// blend/color-write fields live inside `pipeline_rs`. `scissor_rect`
/// is `[u16; 4]` (D3D9 max texture/RT dim is 16384). `blend_factor` /
/// `depth_bias` / `slope_scale_depth_bias` keep `u32` (D3DCOLOR or
/// f32 bit pattern).
pub struct RenderStateSnapshot {
    pub pipeline_rs: mtld3d_core::pipeline_state::PipelineRsBits,
    pub depth_scissor: DepthScissorFlags,
    /// Depth + stencil test, the sole input to the `MTLDepthStencilState` cache.
    ///
    /// `depth_scissor`'s `DEPTH_ENABLE` / `DEPTH_WRITE` bits are derived from
    /// this same snapshot, so the two cannot drift apart.
    pub depth_stencil_state: DepthStencilSnapshot,
    pub cull_mode: u8,
    pub scissor_rect: [u16; 4],
    /// Constant RGBA referenced by `MTLBlendFactor::BlendColor` / `OneMinusBlendColor`.
    ///
    /// Stored as the raw D3DCOLOR (ARGB byte order); decoded to four f32
    /// lanes inside `emit_draw` and emitted only when distinct from the
    /// default 0xFFFFFFFF.
    pub blend_factor: u32,
    /// Raw `D3DRS_DEPTHBIAS` bit pattern (f32 stored in the state DWORD).
    ///
    /// Decoded inside `emit_draw` via
    /// `mtld3d_core::convert::d3d_depth_bias_to_metal`.
    pub depth_bias: u32,
    /// Raw `D3DRS_SLOPESCALEDEPTHBIAS` bit pattern.
    pub slope_scale_depth_bias: u32,
    /// Raw `D3DRS_STENCILREF`, narrowed to `STENCIL_MASK_BITS` at emit.
    ///
    /// Kept out of `depth_stencil` on purpose: Metal carries the reference
    /// on the encoder, so folding it into the state key would mint one
    /// `MTLDepthStencilState` per reference value.
    pub stencil_ref: u32,
    /// `D3DRS_MULTISAMPLEMASK` narrowed against the bound render target.
    ///
    /// `mtld3d_core::multisample::SAMPLE_MASK_ALL` when the state has no
    /// effect, which is every single-sampled draw. Resolved here rather than
    /// on the encoder thread because it needs the target's
    /// `D3DMULTISAMPLE_TYPE`, and the render-state section is re-snapshotted
    /// whenever the render target changes.
    pub sample_mask: u8,
}

impl RenderStateSnapshot {
    #[inline]
    #[must_use]
    pub const fn depth_enable(&self) -> bool {
        self.depth_scissor.contains(DepthScissorFlags::DEPTH_ENABLE)
    }
    #[inline]
    #[must_use]
    pub const fn depth_write(&self) -> bool {
        self.depth_scissor.contains(DepthScissorFlags::DEPTH_WRITE)
    }
    #[inline]
    #[must_use]
    pub const fn scissor_test_enable(&self) -> bool {
        self.depth_scissor.contains(DepthScissorFlags::SCISSOR_TEST)
    }
    #[inline]
    #[must_use]
    pub const fn blend_enable(&self) -> bool {
        self.pipeline_rs.blend_enable()
    }
}

/// Serialize the D3D9 alpha-reference float for upload to the PS slot-14 buffer.
///
/// Returns `(buffer, len)` with `len == 0` when alpha test is off, so
/// `emit_draw` can skip the bind. A fixed buffer (not a `Vec`) so the
/// per-dirty-draw build never touches the allocator.
pub const fn build_alpha_ref_bytes(variant: VariantKey, alpha_ref: f32) -> ([u8; 4], usize) {
    if variant.alpha_func == 0 || variant.alpha_func == 8 {
        return ([0u8; 4], 0);
    }
    (alpha_ref.to_le_bytes(), 4)
}

/// `debug.skipShaders = hex,hex,…` shader-identity bisection probe.
///
/// Each value is a `pair_id().hash` u64 in hex (exactly the value the
/// per-pass debug log prints as `VS ff 0xN` / `VS prog 0xN` / `PS ff
/// 0xN` / `PS prog 0xN`). A draw is skipped when *either* its VS
/// `pair_id` hash or its PS `pair_id` hash matches any value in the
/// set. Stable across frames, unlike index-based skip — the right tool
/// when draw counts vary frame-to-frame.
fn skip_shader_hashes() -> &'static [u64] {
    &crate::config::CONFIG.skip_shaders
}

/// Encoder-thread draw dispatch.
///
/// Pulls the cumulative state from `enc.current_snapshot` (updated by
/// `Op::Set*` ops earlier in the op stream) and combines it with `draw`'s
/// per-call varying parameters (primitive type + vertex/index source).
/// Consumes `draw` so the captured `VertexSource::Up` `Vec` drops at the
/// end of the frame.
pub fn emit_draw(enc: &mut FrameEncoder, draw: DrawOp) {
    let DrawOp {
        metal_prim,
        vertex_source,
        index_source,
    } = draw;
    // Per-draw cost breakdown: six `CycleAddTimer` scopes tile `emit_draw`
    // end to end so the perf summary's "Closures (op)" row decomposes into
    // resolve / pipeline / state / probe / samplers / binds. Each guard holds
    // only a raw counter pointer (no borrow of `enc`), so the measured region
    // reborrows `enc` freely; the explicit `drop` closes one phase before the
    // next begins, and a draw-drop `return` folds the open phase in on the way
    // out. All no-ops unless perf tracking is on.
    let t_resolve = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Resolve));
    // Lifetime-launder the scratch-resident snapshot ptr off `enc` so
    // the rest of emit_draw can freely reborrow `&mut enc`. SAFETY:
    // the pointee lives in `FrameData::scratch` which the encoder
    // owns for the full op-drain duration; the pointer was set by
    // `Op::SetCurrentSnapshot` earlier in this frame's op stream.
    let snap_ptr = enc
        .current_snapshot_ptr()
        .expect("emit_draw: SetCurrentSnapshot not seen")
        .as_ptr();
    // SAFETY: snap_ptr is non-null (NonNull invariant) and points to
    // a live CurrentSnapshot in FrameData::scratch. The pointee
    // outlives the entire op-drain loop in run_frame, well past every
    // `enc` reborrow below.
    let snap: &CurrentSnapshot = unsafe { &*snap_ptr };
    // Every Option must be Some by the time a Draw runs — the API
    // thread populates every field before pushing SetCurrentSnapshot.
    let render_state: &RenderStateSnapshot = snap
        .render_state
        .as_ref()
        .expect("emit_draw: render_state not populated")
        .as_ref();
    let stage_bindings: &StageBindingsPtr = snap
        .stage_bindings
        .as_ref()
        .expect("emit_draw: stage_bindings not populated");
    let attrs = snap.attrs.expect("emit_draw: attrs not populated");
    let vs: &VsSource = snap
        .vs
        .as_ref()
        .expect("emit_draw: vs not populated")
        .as_ref();
    let ps: &PsSource = snap
        .ps
        .as_ref()
        .expect("emit_draw: ps not populated")
        .as_ref();
    let variant = snap.variant.expect("emit_draw: variant not populated");
    // Stages the bound pixel shader declares a sampler for. Every texture,
    // sampler and per-slot bias bind below is confined to this mask: a stage
    // the game bound a texture to that the shader never samples has no
    // argument in the emitted function, so binding it only adds encoder work.
    let ps_sampled_mask = match ps {
        PsSource::Programmable { ps_id, .. } => enc.ps_declared_samplers(*ps_id).mask(),
        PsSource::FixedFunction {
            sampled_stage_mask, ..
        } => *sampled_stage_mask,
    };
    // `D3DSAMP_MIPMAPLODBIAS` has no Metal sampler equivalent, so the bias
    // reaches the GPU as a fragment uniform the sample sites read. Resolving
    // it here keeps every draw that leaves the state at its zero default on
    // the unbiased shader with nothing extra bound. A bias on a stage this
    // shader does not sample reaches no sample site, so it neither mints the
    // biased variant nor binds the table.
    let mut lod_bias = [0.0f32; mtld3d_core::sampler_state::LOD_BIAS_SLOTS];
    let mut any_lod_bias = false;
    for (stage_u32, b) in stage_bindings.iter() {
        if ps_sampled_mask & (1u16 << stage_u32) == 0 {
            continue;
        }
        let bias = mtld3d_core::sampler_state::lod_bias(&b.sampler_state);
        if mtld3d_core::sampler_state::lod_bias_active(bias) {
            lod_bias[stage_u32 as usize] = bias;
            any_lod_bias = true;
        }
    }
    // `D3DRS_SRGBWRITEENABLE` picks the pass's colour attachment views, so it
    // has to reach the pass state before anything this draw emits opens a
    // pass. A change ends the current one, since the views are frozen at
    // pass open.
    enc.set_srgb_write_enabled(variant.flags.contains(VariantFlags::SRGB_WRITE));
    // The render pass decides which colour outputs the PS may export, and
    // that is only known here on the encoder thread. Patch a PS-only copy so
    // the VS key, which shares `variant`, is untouched; the FF PS writes one
    // output and keeps the default so its library index never fragments.
    let extra_attachments = enc.current_extra_color_attachments();
    let (mut ps_variant, ps_color_out_mask) = match ps {
        PsSource::Programmable { color_out_mask, .. } => (
            VariantKey {
                color_out_mask: extra_attachments.present_mask << 1,
                ..variant
            },
            *color_out_mask,
        ),
        PsSource::FixedFunction { .. } => (variant, 1),
    };
    // Point sprites only exist on point primitives: the API thread raises the
    // flag from `D3DRS_POINTSPRITEENABLE` alone, so every other primitive
    // drops it here and keeps its non-sprite library.
    if metal_prim != PrimitiveType::Point {
        ps_variant.flags.remove(VariantFlags::POINT_SPRITE);
    }
    // With sRGB attachment views bound, Metal applies the linear → sRGB OETF
    // after the blender, which is the D3D9 order. The in-shader encode is the
    // fallback for a colour target with no sRGB Metal view, and it must not
    // run as well or the colour is encoded twice.
    if enc.color_attachment_is_srgb() {
        ps_variant.flags.remove(VariantFlags::SRGB_WRITE);
    }
    // A fragment function declaring a depth output against a pass with no
    // depth attachment is a Metal pipeline error, so a programmable PS drops
    // its depth export when no depth buffer is bound (D3D9 discards the
    // write). The FF PS never writes depth and keeps the default key.
    if matches!(ps, PsSource::Programmable { .. }) {
        ps_variant.flags.set(
            VariantFlags::NO_DEPTH_ATTACHMENT,
            !snap.depth_stencil.contains(DepthStencilFlags::HAS_DEPTH),
        );
    }
    // Both emitters honour the bias, so the flag rides on the shared PS key.
    ps_variant.flags.set(VariantFlags::LOD_BIAS, any_lod_bias);
    // `D3DRS_MULTISAMPLEMASK`: Metal has no pipeline-state sample mask, so a
    // narrowed mask becomes a `[[sample_mask]]` output in a pixel-shader
    // variant. The API thread already resolved the state against the bound
    // target, so a full mask (the default) never mints a variant.
    if render_state.sample_mask != mtld3d_core::multisample::SAMPLE_MASK_ALL {
        ps_variant.flags.insert(VariantFlags::SAMPLE_MASK);
        ps_variant.sample_mask = render_state.sample_mask;
    }
    // Programmable VS/PS: snapshot from the encoder-side mirror (kept
    // in sync via `Op::Set{Vs,Ps}ConstRange` deltas). FF: symmetric —
    // snapshot from `ff_vs_constants_mirror` (kept in sync via
    // `Op::SetFfVsConstRange`). Each path's `*_const_scratch` bumps
    // fresh scratch bytes per "mirror epoch" so Metal's submit-time
    // `setVertexBytes` copy sees a stable per-draw payload; a shared
    // mirror would let a later draw's constants bleed into an in-flight
    // one.
    let t_consts = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RConsts));
    let vs_constants = match vs {
        VsSource::Programmable {
            max_const_used,
            uses_rel_const,
            ..
        } => {
            let rows = if *uses_rel_const {
                enc.vs_constants_populated_rows()
            } else {
                *max_const_used
            };
            enc.vs_const_scratch(rows)
        }
        VsSource::FixedFunction { max_row_count, .. } => enc.ff_vs_const_scratch(*max_row_count),
    };
    let ps_constants = match ps {
        PsSource::Programmable { max_const_used, .. } => enc.ps_const_scratch(*max_const_used),
        PsSource::FixedFunction {
            reads_texture_factor: true,
            ..
        } => snap.ps_constants.unwrap_or(ScratchSlice::EMPTY),
        PsSource::FixedFunction { .. } => ScratchSlice::EMPTY,
    };
    let alpha_ref_slice = snap.alpha_ref_bytes.unwrap_or(ScratchSlice::EMPTY);
    let fog_color_slice = snap.fog_color_bytes.unwrap_or(ScratchSlice::EMPTY);
    // SM1 texbem bump-env uniform (slot 12). Only the bound PS knowing it uses
    // a bem-family op pulls the slice — every other draw skips it entirely.
    let ps_uses_bump_env = matches!(
        ps,
        PsSource::Programmable {
            uses_bump_env: true,
            ..
        }
    );
    let bump_env_slice = if ps_uses_bump_env {
        snap.bump_env_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // VS integer constants (vertex slot 14). Only a VS that reads a dynamic
    // integer constant pulls the slice; every other draw skips it entirely.
    let vs_uses_int_const = matches!(
        vs,
        VsSource::Programmable {
            uses_int_const: true,
            ..
        }
    );
    let vs_int_const_slice = if vs_uses_int_const {
        snap.vs_int_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // VS boolean constants (vertex slot 26), gated the same way.
    let vs_uses_bool_const = matches!(
        vs,
        VsSource::Programmable {
            uses_bool_const: true,
            ..
        }
    );
    let vs_bool_const_slice = if vs_uses_bool_const {
        snap.vs_bool_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // PS integer / boolean constants (fragment slots 11 / 10), gated by the
    // bound PS the same way.
    let ps_uses_int_const = matches!(
        ps,
        PsSource::Programmable {
            uses_int_const: true,
            ..
        }
    );
    let ps_int_const_slice = if ps_uses_int_const {
        snap.ps_int_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    let ps_uses_bool_const = matches!(
        ps,
        PsSource::Programmable {
            uses_bool_const: true,
            ..
        }
    );
    let ps_bool_const_slice = if ps_uses_bool_const {
        snap.ps_bool_const_bytes.unwrap_or(ScratchSlice::EMPTY)
    } else {
        ScratchSlice::EMPTY
    };
    // The snapshot records what the app bound; the pass records what Metal
    // will accept. A depth surface that disagrees with render target 0 on
    // sample count is dropped at pass open, and a pipeline built for that pass
    // must declare neither a depth nor a stencil format or Metal rejects the
    // draw.
    let pass_binds_depth = enc.pass_binds_depth();
    let has_depth = pass_binds_depth && snap.depth_stencil.contains(DepthStencilFlags::HAS_DEPTH);
    let has_stencil =
        pass_binds_depth && snap.depth_stencil.contains(DepthStencilFlags::HAS_STENCIL);
    drop(t_consts);

    // 1. Lazily create each bound Metal texture; collect handles. Upload
    //    closures for dirty mips were pushed at `UnlockRect` time and run
    //    earlier in this frame's op list, so by the time we get here the
    //    texture contents are already in place.
    let t_lookup = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RLookup));
    let mut stage_texture_handles: [u64; STAGE_COUNT] = [0; STAGE_COUNT];
    for (stage, b) in stage_bindings.iter() {
        if ps_sampled_mask & (1u16 << stage) == 0 {
            // No sampler argument at this slot. Resolving the handle anyway
            // would put a texture no draw reads into the alias check below,
            // where a stage that happens to hold the pass's depth attachment
            // costs a depth copy.
            continue;
        }
        // D3DSAMP_SRGBTEXTURE binds the texture's eager sRGB twin view so
        // the hardware decodes sRGB→linear at sample time. The handle value
        // itself carries the choice, so the `last_bound` dedup below re-emits
        // the bind whenever a stage flips the state on an unchanged texture.
        stage_texture_handles[stage as usize] =
            if mtld3d_core::sampler_state::srgb_texture_enabled(&b.sampler_state) {
                enc.get_texture_handle_by_id_srgb(b.texture_id)
            } else {
                enc.get_texture_handle_by_id(b.texture_id)
            };
    }
    // A draw that samples the bound depth attachment reads a copy of it: Metal
    // forbids reading an attachment of the running pass. D3D9 permits the
    // bind (scene depth as both depth test and position source) with the
    // values as of the last write, which is what the copy holds.
    let depth_attachment = enc.current_depth_texture();
    if !depth_attachment.is_null() && stage_texture_handles.contains(&depth_attachment.raw()) {
        let snapshot = enc.depth_snapshot_for_sampling();
        if snapshot != 0 {
            for handle in &mut stage_texture_handles {
                if *handle == depth_attachment.raw() {
                    *handle = snapshot;
                }
            }
        }
    }
    if !depth_attachment.is_null() && render_state.depth_enable() && render_state.depth_write() {
        enc.bump_depth_write_epoch();
    }
    drop(t_lookup);

    // 2. Resolve the VS and PS libraries independently. The encoder owns
    //    the parsed programs (populated by register_program ops at
    //    CreateShader); `resolve_*_library` handles cache hit/miss + emit
    //    + compile behind the scenes.
    // `debug.skipShaders` bisection: drop this draw if either stage's content
    // hash is in the skip set. The hash (`disk_key`) is computed only when the
    // set is armed — empty in normal play — so the hot path pays nothing.
    let t_keys = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RKeys));
    let skip_set = skip_shader_hashes();
    if !skip_set.is_empty() {
        let (vs_h, ps_h) = (vs.disk_key(), ps.disk_key(ps_variant));
        // Also match the raw content-hash program ids: they are what the
        // frame dump prints per draw and what names the dumped bytecode
        // files, so a dump line's id can go straight into the skip list
        // without a debug-log run to harvest variant-mixed disk keys.
        let vs_raw = match vs {
            VsSource::Programmable { vs_id, .. } => vs_id.raw(),
            VsSource::FixedFunction { .. } => 0,
        };
        let ps_raw = match ps {
            PsSource::Programmable { ps_id, .. } => ps_id.raw(),
            PsSource::FixedFunction { .. } => 0,
        };
        if skip_set.contains(&vs_h)
            || skip_set.contains(&ps_h)
            || (vs_raw != 0 && skip_set.contains(&vs_raw))
            || (ps_raw != 0 && skip_set.contains(&ps_raw))
        {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: vs_h ^ ps_h,
                "debug.skipShaders: dropping draw with VS {vs_h:#x}/{vs_raw:#x} \
                 PS {ps_h:#x}/{ps_raw:#x}"
            );
            return;
        }
    }
    drop(t_keys);
    // 2. Resolve the VS and PS libraries. The hot path is a borrow-probe of
    //    the source-keyed index (no per-draw content hash, no clone); the
    //    `disk_key` Xxh3 + warm-cache bridge + compile happen lazily inside,
    //    only on a miss (~once per shader).
    let t_lookup = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::RLookup));
    let Some(vs_handles) = enc.resolve_vs_library(vs) else {
        let dk = vs.disk_key();
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: resolve_vs_library returned None");
        mtld3d_shared::log_once_trace_by!(
            target: crate::LOG_TARGET,
            key: dk,
            "drop: VS {dk:#x} did not resolve",
        );
        return;
    };
    let Some(ps_handles) = enc.resolve_ps_library(ps, ps_variant) else {
        let dk = ps.disk_key(ps_variant);
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: resolve_ps_library returned None");
        mtld3d_shared::log_once_trace_by!(
            target: crate::LOG_TARGET,
            key: dk,
            "drop: PS {dk:#x} did not resolve",
        );
        return;
    };
    drop(t_lookup);
    let shaders = ShaderRef { vs, ps, variant };
    enc.maybe_log_pass_shader(shaders, stage_bindings);
    // One vertex buffer layout per stream the declaration reads: stride from
    // the binding (or the declaration extent), step function from the
    // stream's `SetStreamSourceFreq`, a constant zero feed where nothing is
    // bound. Part of the pipeline identity.
    let layouts = stream_layouts(&vertex_source, &attrs);
    enc.maybe_emit_draw_trace(
        shaders,
        metal_prim,
        &vertex_source,
        &index_source,
        layouts[0].stride,
    );
    drop(t_resolve);

    let t_pipeline = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Pipeline));
    let attrs_ref = attrs.as_slice();
    // Instances of an indexed draw: stream 0's frequency count, but only when
    // a stream this draw reads is per-instance; non-indexed draws never
    // instance (D3D9 ignores the frequency state for them).
    let instances = match (&vertex_source, &index_source) {
        (_, IndexSource::None { .. } | IndexSource::Fan { .. }) | (VertexSource::Up { .. }, _) => 1,
        (VertexSource::Bound { stream0_freq, .. }, _) => {
            let any_instanced = vertex_source
                .bindings()
                .any(|b| attrs.used_streams & (1 << b.stream) != 0 && is_instance_data(b.freq));
            instance_count(*stream0_freq, any_instanced)
        }
    };
    let vdecl_hash = attrs.vdecl_hash;
    let vs_const_bytes = vs_constants.as_slice();
    let ps_const_bytes = ps_constants.as_slice();
    let alpha_ref_bytes = alpha_ref_slice.as_slice();
    let fog_color_bytes = fog_color_slice.as_slice();
    let bump_env_bytes = bump_env_slice.as_slice();

    // 3. Pipeline + depth state + cull.
    let color_format = enc.current_color_format();
    let mut attach = mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_COLOR_OUTPUT;
    attach.set(
        mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_DEPTH,
        has_depth,
    );
    attach.set(
        mtld3d_core::pipeline_state::PipelineAttachFlags::HAS_STENCIL,
        has_stencil,
    );
    // Carry the bound RT's D3D "has alpha" bit so destination-alpha blend
    // factors clamp on alpha-less targets (X8R8G8B8 shares `Bgra8Unorm` with
    // A8R8G8B8, so the color format alone can't distinguish them).
    attach.set(
        mtld3d_core::pipeline_state::PipelineAttachFlags::COLOR_HAS_ALPHA,
        enc.current_color_rt_has_alpha(),
    );
    let pipeline_snapshot = PipelineSnapshot {
        vs_fn: vs_handles.func,
        ps_fn: ps_handles.func,
        vdecl_hash,
        stream_layouts: layouts,
        color_format,
        attach,
        rs: render_state.pipeline_rs,
        extra: extra_attachments,
        ps_color_out_mask,
        sample_count: enc.current_color_sample_count(),
    };
    let pipeline = enc.get_or_create_pipeline(&pipeline_snapshot, attrs_ref);
    if pipeline == 0 {
        // Pipeline build failed — e.g. a vertex-declaration/shader attribute
        // mismatch (a shader reads `v0` the bound decl never supplies) or a
        // shader that did not compile. Drop the draw, mirroring the VS/PS
        // resolve-failure drops above: a render pass that issues
        // `drawPrimitives` with no pipeline bound is undefined in Metal and
        // faults hard at submit (a process-killing SIGSEGV with no recovery),
        // so the draw must never be emitted.
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "draw dropped: pipeline creation failed (no pipeline bound)"
        );
        return;
    }
    let depth_state = if has_depth {
        let snapshot = render_state
            .depth_stencil_state
            .gated_on_stencil_attachment(has_stencil);
        enc.get_or_create_depth_stencil(&snapshot)
    } else {
        0
    };
    let metal_cull = d3d_to_metal_cull(u32::from(render_state.cull_mode));
    drop(t_pipeline);

    let t_state = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::State));
    enc.begin_render_pass_if_needed();
    // Tag the pass with "this draw wants to write color" iff
    // COLORWRITEENABLE is non-zero. When every draw in the pass closes
    // with this still false, Rule H strips the color attachment + swaps
    // the bound pipeline to the no-color variant. Must run after
    // begin_render_pass_if_needed so the tag lands on the right pass.
    enc.note_draw_color_write_mask(u32::from(!pipeline_snapshot.writes_no_color()));
    enc.emit_scissor(
        render_state.scissor_test_enable(),
        render_state.scissor_rect.map(u32::from),
    );
    if enc.last_bound().pipeline_changed(pipeline) {
        enc.emit_command(Command::set_render_pipeline_state(pipeline));
    }
    if depth_state != 0 && enc.last_bound().depth_stencil_changed(depth_state) {
        enc.emit_command(Command::set_depth_stencil_state(depth_state));
    }
    if enc.last_bound().cull_mode_changed(metal_cull) {
        enc.emit_command(Command::set_cull_mode(metal_cull));
    }

    // D3DRS_DEPTHBIAS / D3DRS_SLOPESCALEDEPTHBIAS drive Metal's
    // per-encoder rasterizer offset. Routed through `LastBoundCache`
    // so the per-encoder bias only re-binds when the resolved
    // (bias, slope-scale) pair actually changes — the cache itself
    // is what prevents the "leaked from a previous draw" failure
    // mode (every emit updates the slot; every change re-emits).
    //
    // Implicit decal bias: when the render-state pattern matches a
    // typical alpha-blended decal (depth-test on, depth-write off,
    // alpha-blend on, game's DEPTHBIAS == 0) AND the game hasn't
    // already supplied a slope-scale, push the polygon slightly
    // toward the camera so it reliably wins ZTest against the
    // underlying surface. A ground-projected decal whose VS and the
    // underlying surface's VS use different WV translation columns
    // (e.g. a decal in object-space verts × `world_decal · view`
    // versus terrain in world-space verts × `view`) produces ULP-level
    // different eye-space depths for the same world point even when
    // the math is bit-identical — no shader-invariance trick can
    // bridge that on Apple Silicon, so the bias is the load-bearing
    // fix.
    //
    // Magnitude: negative pushes toward camera (D3D9 depth: 0 = near,
    // 1 = far). Small enough that genuine geometry an order of
    // magnitude further from the surface still composites correctly;
    // large enough to swamp ULP-level noise from divergent FP rounding
    // between pipelines (~10s of ULPs at the depth-buffer's
    // precision).
    let decal_inputs = DecalHeuristicInputs {
        depth_enable: u32::from(render_state.depth_enable()),
        depth_write: u32::from(render_state.depth_write()),
        blend_enable: u32::from(render_state.blend_enable()),
        raw_depth_bias: render_state.depth_bias,
        raw_slope_scale: render_state.slope_scale_depth_bias,
    };
    let decal_fires = looks_like_decal(decal_inputs);
    let raw_bias = if decal_fires {
        IMPLICIT_DECAL_BIAS_RAW
    } else {
        render_state.depth_bias
    };
    let depth_bias = d3d_depth_bias_to_metal(raw_bias);
    // Slope-scale: when the decal heuristic fires, layer
    // `IMPLICIT_DECAL_SLOPE_SCALE` on top of the absolute bias.
    // `looks_like_decal` already requires the game's own slope-scale to be
    // zero, so `decal_fires` alone implies "the game hasn't supplied one".
    // Metal applies `m × slopeScale + r × bias`, so this term is
    // free for flat surfaces (m ≈ 0) and does the heavy lifting
    // at grazing angles where the structural eye-space delta
    // exceeds the absolute budget — see the constant's doc for
    // the rationale.
    let slope_scale = if decal_fires {
        IMPLICIT_DECAL_SLOPE_SCALE
    } else {
        f32::from_bits(render_state.slope_scale_depth_bias)
    };
    if enc.last_bound().depth_bias_changed(depth_bias, slope_scale) {
        enc.emit_command(Command::set_depth_bias(depth_bias, slope_scale));
    }

    // D3D9 depth-clamps (skips z-clip on) pre-transformed (XYZRHW) geometry
    // while the depth test is inactive; everything else z-clips. Both
    // conjuncts are load-bearing (the D3D9 depth-clamp rule: depth-clamp ⇔
    // depth test inactive AND geometry pre-transformed):
    //  - Case 1: RHW quads spanning z in [-0.5, 1.5] with no depth surface
    //    (or ZENABLE off) draw in full — Metal's always-clip default discards
    //    their outer columns;
    //  - Case 2: the SAME ZENABLE=FALSE state with a regular VS quad still
    //    z-clips (so "depth test inactive" alone is wrong);
    //  - Case 3: RHW with the test LIVE stays clipped under any
    //    D3DRS_CLIPPING value (so "RHW" alone is wrong — clamping every
    //    RHW draw unconditionally would wrongly bypass clipping here).
    // The RHW-with-bound-VS bypass resolves pre-transformed draws to a
    // FixedFunction source, so the FF key's has_rhw covers every RHW draw.
    // Realized as a VS-side clamp selected per-draw through `pos_fixup.z`
    // (emitted below with the half-pixel fixup) rather than
    // `MTLDepthClipMode::Clamp`: encoder-level clamp is not honoured by
    // every Metal device (a GitHub runner's paravirtual GPU clips
    // regardless), while the clamp in the FF vertex shader behaves
    // identically on all of them.
    let position_transformed = matches!(vs, VsSource::FixedFunction { key, .. } if key.has_rhw());
    let depth_clamp_z = position_transformed && !(has_depth && render_state.depth_enable());
    drop(t_state);

    // Diagnostic probe: per-(VS, PS, state) decal + caster trace rows. Zero
    // cost at the default log level via the explicit `log_enabled!` gate
    // below — it skips the whole key build, not just the trace emit.
    // `RUST_LOG=mtld3d::d3d9::decal=trace` / `…::caster=trace` opt in.
    let t_probe = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Probe));
    // `note_caster_draw` self-gates on `mtld3d::d3d9::cascade` and the
    // session's sampleable-depth set, so it stays unconditional (one cached
    // atomic load when that probe is off) — this keeps the cascade summary's
    // caster-write counters correct regardless of the trace gate below.
    let depth_tex = enc.current_depth_texture();
    enc.note_caster_draw(depth_tex);
    // Everything else here only feeds the decal/caster trace rows. Gate the
    // whole key+message build on those two targets so a default-log-level
    // draw pays nothing — in particular it skips the `pair_id` content hash
    // the keys would otherwise recompute (an Xxh3 for FF / programmable-PS).
    if log_enabled!(target: DECAL_TRACE_TARGET, Level::Trace)
        || log_enabled!(target: CASTER_TRACE_TARGET, Level::Trace)
    {
        // Gated path: the content hashes are computed here (only when a trace
        // target is on), not on the hot path.
        let vs_hash = vs.disk_key();
        let ps_hash = ps.disk_key(variant);
        let pair_key = vs_hash ^ ps_hash.rotate_left(1);
        // Bake the discriminating render-state bits into the dedup
        // key so a shader pair re-used in distinct (ZW, AB)
        // configurations produces one trace row per configuration
        // instead of collapsing.
        let state_bits = (u64::from(decal_fires) << 2)
            | (u64::from(render_state.depth_write()) << 1)
            | u64::from(render_state.blend_enable());
        let probe_key = pair_key ^ (state_bits << 60);
        let pass_idx = enc.current_pass_index();
        let alpha_func = variant.alpha_func;
        mtld3d_shared::log_once_trace_by!(
            target: DECAL_TRACE_TARGET,
            key: probe_key,
            "decal: pass={pass_idx} VS prog {vs_hash:#018x} PS prog {ps_hash:#018x} \
             rs[Z={z} ZW={zw} AB={ab} bias={bias:#010x} slope={slope:#010x}] \
             blend[src={src} dst={dst} op={op}] at={alpha_func} \
             decal_fires={decal_fires} applied_raw={raw_bias:#010x} \
             applied_metal={depth_bias:.3} slope_metal={slope_scale:.3}",
            z = u32::from(render_state.depth_enable()),
            zw = u32::from(render_state.depth_write()),
            ab = u32::from(render_state.blend_enable()),
            bias = render_state.depth_bias,
            slope = render_state.slope_scale_depth_bias,
            src = render_state.pipeline_rs.src_blend,
            dst = render_state.pipeline_rs.dst_blend,
            op = render_state.pipeline_rs.blend_op,
        );

        // Caster probe: one row per unique caster-draw signature on a
        // sampleable shadow map (cascade depth attachment). Built to
        // diff caster pipeline state between two captures when tree
        // self-shadow flickers. Combines alpha-test (the hypothesised
        // failure mode for foliage casters), depth-write / blend, and
        // bias to flag any frame-to-frame drift. Self-filters on the
        // session's sampleable-depth set — the same handles
        // `note_caster_draw` above counts — so it fires only on draws
        // into cascade textures, not the main scene depth.
        if !depth_tex.is_null() && enc.is_depth_handle_sampleable(depth_tex) {
            // 32-bit alpha-ref f32 mantissa-truncated into the key; full
            // bits go into the message so legitimate frame-to-frame
            // ref changes show as distinct rows.
            let alpha_ref_bits: u32 = alpha_ref_bytes
                .get(..4)
                .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            let caster_state_bits = (u64::from(alpha_func) << 56)
                | (u64::from(render_state.depth_write()) << 55)
                | (u64::from(render_state.blend_enable()) << 54)
                | (u64::from(render_state.cull_mode & 0x3) << 52)
                | u64::from(alpha_ref_bits);
            let caster_key = pair_key ^ caster_state_bits ^ depth_tex.raw().rotate_left(8);
            mtld3d_shared::log_once_trace_by!(
                target: CASTER_TRACE_TARGET,
                key: caster_key,
                "caster: depth=0x{dt:x} VS prog {vs_hash:#018x} PS prog {ps_hash:#018x} \
                 at={alpha_func} aref={aref:#010x} zw={zw} ze={ze} ab={ab} \
                 cull={cull} bias={bias:#010x} slope={slope:#010x}",
                dt = depth_tex,
                aref = alpha_ref_bits,
                ze = u32::from(render_state.depth_enable()),
                zw = u32::from(render_state.depth_write()),
                ab = u32::from(render_state.blend_enable()),
                cull = render_state.cull_mode,
                bias = render_state.depth_bias,
                slope = render_state.slope_scale_depth_bias,
            );
        }
    }
    drop(t_probe);

    let t_samplers = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Samplers));
    // D3DRS_STENCILREF drives Metal's per-encoder stencil reference. Only
    // meaningful while the stencil test is on, and unlike the blend color it
    // is emitted on every change (including back to zero) so a pass that
    // lowers the reference mid-encoder doesn't keep comparing against the
    // previous one.
    let stencil_ref = render_state.stencil_ref & STENCIL_MASK_BITS;
    if render_state.depth_stencil_state.stencil_enable != 0
        && has_stencil
        && enc.last_bound().stencil_reference_changed(stencil_ref)
    {
        enc.emit_command(Command::set_stencil_reference(stencil_ref));
    }

    // D3DRS_BLENDFACTOR drives Metal's per-encoder constant blend
    // color. Default is 0xFFFFFFFF (opaque white), which is also the
    // Metal default — only emit the command when the game has overridden
    // it, so the per-pass command stream stays minimal for the common
    // case where no constant-color blend is used.
    if render_state.blend_factor != 0xFFFF_FFFF
        && enc
            .last_bound()
            .blend_color_changed(render_state.blend_factor)
    {
        let [r, g, b, a] = mtld3d_core::convert::d3dcolor_to_rgba_f32(render_state.blend_factor);
        enc.emit_command(Command::set_blend_color(r, g, b, a));
    }

    // 4. Texture + sampler binds, one pair per valid stage. Depth-bound
    //    slots (sampleable shadow maps) need the `compareFunction =
    //    LessEqual` sampler variant so MSL `sample_compare` returns the
    //    D3D9 hardware-shadow PCF result; the bit mirrors the emitter's
    //    `depth_sampler_mask` so the call site and the sampler state
    //    can't drift.
    let depth_mask = variant.depth_sampler_mask;
    // Raw-fetch depth slots (INTZ/DF24/DF16) are depth textures but are read
    // with a plain `.sample()`, which requires a NON-comparison sampler — so
    // exclude them from the compare-sampler set.
    let fetch_mask = variant.depth_fetch_mask;
    let mut bound_mask: u16 = 0;
    for (stage_u32, b) in stage_bindings.iter() {
        if ps_sampled_mask & (1u16 << stage_u32) == 0 {
            continue;
        }
        let handle = stage_texture_handles[stage_u32 as usize];
        if handle == 0 {
            mtld3d_shared::log_once_warn_by!(target: crate::LOG_TARGET,
                key: b.texture_id.raw(),
                "draw: stage {stage_u32} bound to {:?} but its texture handle is 0 — bind \
                 skipped, an unbound declared sampler reads opaque black",
                b.texture_id
            );
            continue;
        }
        let bit = 1u16 << stage_u32;
        bound_mask |= bit;
        let is_compare = (depth_mask & bit) != 0 && (fetch_mask & bit) == 0;
        let is_fetch = (fetch_mask & bit) != 0;
        let sampler = enc.get_or_create_sampler(stage_u32, &b.sampler_state, is_compare, is_fetch);
        if enc.last_bound().fragment_texture_changed(stage_u32, handle) {
            enc.emit_command(Command::set_fragment_texture(handle, stage_u32));
        }
        if enc
            .last_bound()
            .fragment_sampler_changed(stage_u32, sampler)
        {
            enc.emit_command(Command::set_fragment_sampler_state(sampler, stage_u32));
        }
    }

    // A pixel shader may declare a sampler the game bound no texture to. D3D9
    // requires that sample to read opaque black, and Metal requires every
    // declared `[[texture(n)]]` argument to be bound, so bind the shared 1×1
    // black texture (of the declared type) plus a default sampler to each such
    // slot. Only the programmable path can declare-without-binding; the FF PS
    // only declares samplers for stages it actually samples a bound texture on.
    if let PsSource::Programmable { ps_id, .. } = ps {
        let decls = enc.ps_declared_samplers(*ps_id);
        let mut unbound = decls.unbound(bound_mask);
        while unbound != 0 {
            let stage = unbound.trailing_zeros();
            unbound &= unbound - 1;
            // The fragment function types every slot from the bound texture,
            // not from its `dcl_<dim>`, so the fallback follows the same rule:
            // an unbound slot has no mask bit set and takes the 2D black
            // texture whatever the shader declared.
            let slot = u16::try_from(stage).expect("declared sampler slot is below STAGE_COUNT");
            let kind = null_texture_kind(bound_sampler_type(variant, slot));
            let tex_sentinel = null_texture_tex_sentinel(kind as u64);
            if enc
                .last_bound()
                .fragment_texture_changed(stage, tex_sentinel)
            {
                enc.emit_command(Command::set_fragment_null_texture(kind, stage));
            }
            // The null command also binds a default sampler; record a sentinel
            // so a later real sampler bind to this slot is not deduped away.
            enc.last_bound()
                .fragment_sampler_changed(stage, NULL_TEXTURE_SAMPLER_SENTINEL);
        }
    }
    // Vertex texture fetch: bind each slot the VS declares a sampler for
    // (`vs_3_0`, at most four). The bindings mirror `SetTexture` /
    // `SetSamplerState` on `D3DVERTEXTEXTURESAMPLER0..3` and live on the
    // encoder rather than the per-draw snapshot; a declared slot the game
    // never bound gets the shared black fallback, as on the fragment side.
    if let VsSource::Programmable {
        vs_id,
        sampler_kinds,
        ..
    } = vs
    {
        let decls = enc.ps_declared_samplers(*vs_id);
        let mut mask = decls.unbound(0) & 0xF;
        while mask != 0 {
            let slot = mask.trailing_zeros();
            mask &= mask - 1;
            let (id, ss) = enc.vertex_binding(slot as usize);
            let handle = id.map_or(0, |id| {
                if mtld3d_core::sampler_state::srgb_texture_enabled(&ss) {
                    enc.get_texture_handle_by_id_srgb(id)
                } else {
                    enc.get_texture_handle_by_id(id)
                }
            });
            if handle == 0 {
                // The emitter types the argument from `sampler_kinds`, so the
                // black fallback must carry the same kind: an unbound slot is
                // 2D there, and a slot whose texture has no handle yet keeps
                // the kind the key already compiled for.
                let kind = null_texture_kind(
                    sampler_kinds
                        .kind(u16::try_from(slot).expect("vertex fetch slot is below four")),
                );
                let tex_sentinel = null_texture_tex_sentinel(kind as u64);
                if enc.last_bound().vertex_texture_changed(slot, tex_sentinel) {
                    enc.emit_command(Command::set_vertex_null_texture(kind, slot));
                }
                enc.last_bound()
                    .vertex_sampler_changed(slot, NULL_TEXTURE_SAMPLER_SENTINEL);
            } else {
                // Slot indices 16..19: past the fragment-stage memo range,
                // so vertex samplers never collide with a stage's memo entry.
                let sampler = enc.get_or_create_sampler(16 + slot, &ss, false, false);
                if enc.last_bound().vertex_texture_changed(slot, handle) {
                    enc.emit_command(Command::set_vertex_texture(handle, slot));
                }
                if sampler != 0 && enc.last_bound().vertex_sampler_changed(slot, sampler) {
                    enc.emit_command(Command::set_vertex_sampler_state(sampler, slot));
                }
            }
        }
    }
    drop(t_samplers);

    let t_binds = CycleAddTimer::start(enc.op_sub_cycles_ptr(OpSub::Binds));
    let t_cbind = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BCbind));
    // 5. Shader constants (VS float slot / PS slot 15), alpha-ref float (PS
    //    slot 14), fog color (PS slot 13). Dedup via inline-bytes cache: when
    //    the same constants re-bind draw-after-draw (FF pass with one CB
    //    update at the head, or shadow-cast pass with shared light constants)
    //    we skip the `setBytes` command. The bytes already live in the owning
    //    `FrameData::scratch` arena (bump-allocated on the API thread)
    //    so we pass their `(ptr, len)` straight to the Metal command
    //    without re-copying through the encoder's scratch.
    if !vs_const_bytes.is_empty() && enc.last_bound().vs_constants_changed(vs_const_bytes) {
        let (p, n) = vs_constants.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_FLOAT_CONST_SLOT));
    }
    // Half-pixel rasterization fixup (VS pos-fixup slot). Every DXSO/FF vertex
    // shader declares `constant float4 &pos_fixup` there and shifts
    // clip-space position half a pixel right/down so on-boundary geometry
    // lands on the D3D9 window→NDC reference.
    // `(1/vp_w, -1/vp_h, depth_clamp_z, render_scale)` from the live viewport;
    // the `.z` lane selects the FF RHW epilogue's depth clamp (the D3D9
    // depth-clamp rule, see `depth_clamp_z` above) and the `.w` lane carries
    // render pixels per logical pixel, which the point-size epilogue applies
    // to a size D3D9 states in the logical space. The viewport dims are
    // already in the bound target's space, so `.xy` needs no conversion.
    // Deduped so it only re-emits when the viewport dims, the clamp predicate
    // or the bound target's scale change (rare).
    let (_, _, vp_w, vp_h) = enc.effective_viewport();
    // Viewport dims fit u16 in practice; convert without an `as`-cast
    // precision-loss lint (same idiom as `encoder.rs`).
    let to_f = |v: u32| f32::from(u16::try_from(v).unwrap_or(u16::MAX));
    let pos_fixup: [f32; 4] = [
        1.0 / to_f(vp_w.max(1)),
        -1.0 / to_f(vp_h.max(1)),
        f32::from(u8::from(depth_clamp_z)),
        enc.target_scale().factor(),
    ];
    // SAFETY: `[f32; 4]` is POD with no padding; reinterpreting the array as
    // 16 contiguous bytes is sound and the borrow is local to this scope.
    let pos_fixup_bytes =
        unsafe { core::slice::from_raw_parts(pos_fixup.as_ptr().cast::<u8>(), 16) };
    if enc.last_bound().vs_pos_fixup_changed(pos_fixup_bytes) {
        let ptr = enc.alloc_scratch(pos_fixup_bytes);
        enc.emit_command(Command::set_vertex_bytes_at(ptr, 16, VS_POS_FIXUP_SLOT));
    }
    // Per-draw `VsDraw` uniform (point size state). Every vertex shader
    // declares it, so it must be bound before the first draw of a pass; the
    // API thread rebuilds the bytes only when a point render state changes
    // and the dedup skips the bind when they match the last ones emitted.
    let vs_draw = snap.vs_draw_bytes.unwrap_or(ScratchSlice::EMPTY);
    let vs_draw_bytes = vs_draw.as_slice();
    if vs_draw_bytes.is_empty() {
        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
            "draw: VsDraw uniform not populated; binding the render-state defaults"
        );
        if enc.last_bound().vs_draw_changed(&*VS_DRAW_DEFAULT) {
            let ptr = enc.alloc_scratch(&*VS_DRAW_DEFAULT);
            enc.emit_command(Command::set_vertex_bytes_at(
                ptr,
                u32::try_from(VS_DRAW_BYTES).expect("32 fits u32"),
                VS_DRAW_SLOT,
            ));
        }
    } else if enc.last_bound().vs_draw_changed(vs_draw_bytes) {
        let (p, n) = vs_draw.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_DRAW_SLOT));
    }
    if !ps_const_bytes.is_empty() && enc.last_bound().ps_constants_changed(ps_const_bytes) {
        let (p, n) = ps_constants.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 15));
    }
    // The alpha-reference scalar (slot 14) is read only by a comparison that
    // has a reference to compare against: the two constant results compile to
    // `true` / `false` and leave the argument untouched, and a disabled test
    // emits none. Asking the shader rather than trusting the byte buffer to be
    // empty keeps the bind in step with what the emitter produced.
    let alpha_func = u32::from(ps_variant.alpha_func);
    let alpha_ref_read =
        alpha_func != 0 && alpha_func != D3DCMP_ALWAYS && alpha_func != D3DCMP_NEVER;
    if alpha_ref_read
        && !alpha_ref_bytes.is_empty()
        && enc.last_bound().ps_alpha_ref_changed(alpha_ref_bytes)
    {
        let (p, n) = alpha_ref_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 14));
    }
    if !fog_color_bytes.is_empty() && enc.last_bound().ps_fog_color_changed(fog_color_bytes) {
        let (p, n) = fog_color_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 13));
    }
    if !bump_env_bytes.is_empty() && enc.last_bound().ps_bump_env_changed(bump_env_bytes) {
        let (p, n) = bump_env_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, 12));
    }
    // Per-slot LOD bias. Bound only for a draw whose shader declares the
    // uniform; the binding then persists on the encoder, so a later biased
    // draw carrying the same table skips the re-bind.
    if any_lod_bias {
        let bias_bytes = mtld3d_core::sampler_state::build_lod_bias_bytes(&lod_bias);
        if enc.last_bound().ps_lod_bias_changed(&bias_bytes) {
            let ptr = enc.alloc_scratch(&bias_bytes);
            enc.emit_command(Command::set_fragment_bytes_at(
                ptr,
                u32::try_from(mtld3d_core::sampler_state::LOD_BIAS_BYTES)
                    .expect("LOD bias uniform is 256 bytes"),
                PS_LOD_BIAS_SLOT,
            ));
        }
    }
    // VS integer constants — bound only for the rare shader that reads a
    // dynamic integer constant. Re-bound unconditionally (no dedup): such
    // draws are infrequent and the payload is a fixed 256 B.
    if !vs_int_const_slice.as_slice().is_empty() {
        let (p, n) = vs_int_const_slice.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_INT_CONST_SLOT));
    }
    // VS boolean constants: a 4-byte bitmask, same rare-draw policy.
    if !vs_bool_const_slice.as_slice().is_empty() {
        let (p, n) = vs_bool_const_slice.as_raw();
        enc.emit_command(Command::set_vertex_bytes_at(p, n, VS_BOOL_CONST_SLOT));
    }
    // PS integer / boolean constants: the fragment-side twins, same policy.
    if !ps_int_const_slice.as_slice().is_empty() {
        let (p, n) = ps_int_const_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, PS_INT_CONST_SLOT));
    }
    if !ps_bool_const_slice.as_slice().is_empty() {
        let (p, n) = ps_bool_const_slice.as_raw();
        enc.emit_command(Command::set_fragment_bytes_at(p, n, PS_BOOL_CONST_SLOT));
    }
    drop(t_cbind);

    // 6. Bind the vertex streams. Wraps each bound VB's `PageBox` in an
    //    MTLBuffer lazily — the cache hits after the first draw post-rename
    //    and churns only when the game renames.
    let t_vbib = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BVbib));
    match &vertex_source {
        VertexSource::Up { bytes, size, .. } => {
            let scratch_ptr = enc.alloc_scratch(bytes);
            enc.emit_command(Command::set_vertex_bytes(scratch_ptr, *size, 0));
            // Inline slot-0 bind clobbers the real Metal vertex-buffer
            // binding; drop the cached bound-VB so the next bound draw
            // re-emits its `setVertexBuffer` instead of reading these bytes.
            enc.last_bound().invalidate_vertex_buffer();
        }
        VertexSource::Bound { .. } => {
            for b in vertex_source.bindings() {
                let slot = u32::from(b.stream);
                let layout = layouts[b.stream as usize];
                if !layout.is_used() {
                    // Bound but not read by the declaration's consumed
                    // attributes: nothing to bind.
                    continue;
                }
                let buffer_handle = enc.ensure_vbib_mtl_buffer(
                    b.buffer_id,
                    b.backing_ptr,
                    b.backing_len,
                    b.backing_generation,
                );
                if buffer_handle == 0 {
                    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: ensure_vbib_mtl_buffer returned 0 for VB");
                    mtld3d_shared::log_once_trace_by!(
                        target: crate::LOG_TARGET,
                        key: b.buffer_id.raw(),
                        "drop: VB buffer {:#x} (stream {slot}) wrap failed",
                        b.buffer_id.raw(),
                    );
                    return;
                }
                if enc
                    .last_bound()
                    .vertex_buffer_changed(slot, buffer_handle, b.offset)
                {
                    enc.emit_command(Command::set_vertex_buffer(buffer_handle, b.offset, slot));
                }
                // Record this draw's VB read range so a later overlapping
                // staging upload renames instead of corrupting this draw.
                // Tighten it past the whole-buffer fallback so a disjoint
                // later upload to the same buffer doesn't force a needless
                // rename: non-indexed draws give the exact vertex span;
                // indexed draws tighten only the lower bound by `base_vertex`
                // (the upper bound needs the max index value, which we don't
                // scan, so it stays at end-of-buffer); a per-instance stream
                // reads one element per `step_rate` instances. All may
                // over-cover but never under-cover; overflow falls back to the
                // whole tail. `None` = the draw reads nothing → record nothing.
                let logical_len = u32::try_from(b.backing_len).unwrap_or(u32::MAX);
                let read_range = match (layout.step, &index_source) {
                    (
                        VertexStepFunction::PerVertex,
                        IndexSource::None {
                            start_vertex,
                            vertex_count,
                        },
                    ) => nonindexed_vb_range(b.offset, layout.stride, *start_vertex, *vertex_count),
                    (
                        VertexStepFunction::PerVertex,
                        IndexSource::Bound {
                            base_vertex,
                            index_count,
                            ..
                        },
                    ) => indexed_vb_range_lower_bound(
                        b.offset,
                        layout.stride,
                        *base_vertex,
                        *index_count,
                    ),
                    // A fan over the shared pattern reads `primitive_count + 2`
                    // vertices from its start, like a non-indexed draw.
                    (
                        VertexStepFunction::PerVertex,
                        IndexSource::Fan {
                            start_vertex,
                            primitive_count,
                        },
                    ) => nonindexed_vb_range(
                        b.offset,
                        layout.stride,
                        *start_vertex,
                        primitive_count.saturating_add(2),
                    ),
                    // A generated index list knows exactly which vertices it
                    // references, so the range is as tight as a non-indexed
                    // draw's.
                    (
                        VertexStepFunction::PerVertex,
                        IndexSource::Generated {
                            min_vertex,
                            max_vertex,
                            ..
                        },
                    ) => nonindexed_vb_range(
                        b.offset,
                        layout.stride,
                        *min_vertex,
                        max_vertex - min_vertex + 1,
                    ),
                    // `Up` indices only ever pair with `VertexSource::Up`,
                    // never a bound VB, so this arm is unreachable in
                    // practice; record no read range.
                    (VertexStepFunction::PerVertex, IndexSource::Up { .. }) => None,
                    (VertexStepFunction::PerInstance | VertexStepFunction::Constant, _) => Some((
                        b.offset,
                        instanced_stream_read_bytes(
                            instances,
                            layout.step,
                            layout.step_rate,
                            layout.stride,
                        ),
                    )),
                };
                if let Some((range_off, range_size)) = read_range {
                    enc.note_buffer_draw_range(
                        b.buffer_id.raw(),
                        range_off,
                        range_size,
                        logical_len,
                    );
                }
            }
        }
    }
    // Streams the declaration reads with nothing bound: feed zeros inline
    // under the constant layout built above. The inline bind clobbers that
    // slot's real Metal binding, so forget it in the cache too.
    let mut null_streams = attrs.used_streams;
    while null_streams != 0 {
        let stream = null_streams.trailing_zeros();
        null_streams &= null_streams - 1;
        if !matches!(vertex_source.feed(stream), StreamFeed::Null) {
            continue;
        }
        let extent = layouts[stream as usize].stride;
        let len = u32::try_from(NULL_STREAM_ZEROS.len()).expect("4 KiB fits u32");
        if extent > len {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "stream {stream} is unbound and its declaration extent {extent} exceeds the zero feed; clamped"
            );
        }
        enc.emit_command(Command::set_vertex_bytes_at(
            NULL_STREAM_ZEROS.as_ptr() as u64,
            extent.min(len),
            stream,
        ));
        enc.last_bound().invalidate_vertex_buffer_slot(stream);
    }
    drop(t_vbib);

    // 7. Emit the draw call.
    // Debug-build invariant: the per-draw dedup cache must match what was
    // actually emitted onto the encoder before the draw consumes it — catches
    // any cached-slot bind that bypassed its `last_bound` gate.
    #[cfg(debug_assertions)]
    enc.debug_assert_cache_in_sync();
    let t_draw = CycleAddTimer::start(enc.op_sub_detail_ptr(OpSubDetail::BDraw));
    // The generated-index fan is the slow path (per-draw rewrite plus a
    // transient index buffer); the PERF grid counts it as a tripwire.
    if matches!(index_source, IndexSource::Generated { .. }) {
        enc.bump_fan_generated();
    }
    let verts = match index_source {
        IndexSource::None {
            start_vertex,
            vertex_count,
        } => {
            enc.emit_command(Command::draw_primitives(
                metal_prim,
                start_vertex,
                vertex_count,
            ));
            vertex_count
        }
        IndexSource::Bound {
            buffer_id,
            backing_ptr,
            backing_len,
            backing_generation,
            offset,
            index_count,
            index_type,
            base_vertex,
        } => {
            let buffer_handle =
                enc.ensure_vbib_mtl_buffer(buffer_id, backing_ptr, backing_len, backing_generation);
            if buffer_handle == 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "draw dropped: ensure_vbib_mtl_buffer returned 0 for IB");
                mtld3d_shared::log_once_trace_by!(
                    target: crate::LOG_TARGET,
                    key: buffer_id.raw(),
                    "drop: IB buffer {:#x} wrap failed",
                    buffer_id.raw(),
                );
                return;
            }
            // Record this draw's IB read range so a later overlapping
            // staging upload renames instead of corrupting this draw.
            // Exact — `[offset, offset + index_count × index_size)`.
            let index_size: u32 = match index_type {
                IndexType::UInt16 => 2,
                IndexType::UInt32 => 4,
            };
            let read_bytes = index_count.saturating_mul(index_size);
            let logical_len = u32::try_from(backing_len).unwrap_or(u32::MAX);
            enc.note_buffer_draw_range(buffer_id.raw(), offset, read_bytes, logical_len);
            enc.emit_command(Command::draw_indexed_primitives(
                metal_prim,
                index_count,
                index_type,
                buffer_handle,
                offset,
                base_vertex,
                instances,
            ));
            index_count
        }
        IndexSource::Fan {
            start_vertex,
            primitive_count,
        } => {
            // The shared pattern is relative to the fan's first vertex;
            // `start_vertex` becomes Metal's base vertex. The device routes
            // only fans the pattern can address here, so the narrowing and
            // the buffer are expected to succeed; both failures drop the
            // draw loudly rather than draw the wrong triangles.
            let Ok(base_vertex) = i32::try_from(start_vertex) else {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "draw dropped: triangle fan start vertex {start_vertex} exceeds the base vertex range");
                return;
            };
            let buffer_handle = enc.fan_index_buffer(primitive_count);
            if buffer_handle == 0 {
                mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                    "draw dropped: the shared triangle fan index buffer could not be created");
                return;
            }
            let index_count = primitive_count * 3;
            enc.emit_command(Command::draw_indexed_primitives(
                metal_prim,
                index_count,
                IndexType::UInt16,
                buffer_handle,
                0,
                base_vertex,
                instances,
            ));
            index_count
        }
        IndexSource::Up {
            bytes,
            index_count,
            index_type,
        } => {
            // Inline index stream: stage the bytes in the per-frame scratch
            // arena and let the unix side wrap them in a transient MTLBuffer
            // (Metal has no inline-index draw form). The vertices were bound
            // above from `VertexSource::Up`.
            let scratch_ptr = enc.alloc_scratch(&bytes);
            let byte_len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
            enc.emit_command(Command::draw_indexed_primitives_up(
                metal_prim,
                index_count,
                index_type,
                scratch_ptr,
                byte_len,
                instances,
            ));
            index_count
        }
        IndexSource::Generated {
            data,
            index_count,
            index_type,
            ..
        } => {
            // The list is already in the frame arena the unix side reads at
            // replay time, so it goes to the same inline-index draw form
            // without a second copy. The vertices were bound above: the
            // caller's buffers, or `VertexSource::Up` bytes.
            let (index_ptr, byte_len) = data.as_raw();
            enc.emit_command(Command::draw_indexed_primitives_up(
                metal_prim,
                index_count,
                index_type,
                index_ptr,
                byte_len,
                instances,
            ));
            index_count
        }
    };
    drop(t_draw);
    drop(t_binds);

    enc.bump_pair_stats(
        shaders,
        verts,
        variant.alpha_func,
        u32::from(render_state.cull_mode),
    );
}
