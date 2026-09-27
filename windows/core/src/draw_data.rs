//! Captured draw state shared by API recording and native encoding.
//!
//! These are local Rust owners and borrowed arena tokens, not the PE/Unix wire layout.

use core::ptr::NonNull;

use mtld3d_shared::{
    NullTextureKind, VertexAttrDesc,
    mtl::{IndexType, PrimitiveType, VertexStepFunction},
};
use mtld3d_types::{MAX_STREAMS, SAMPLER_STATE_COUNT};

pub use crate::shader_cache::{ps_source_disk_key_programmable, vs_source_disk_key_programmable};
use crate::{
    depth_stencil_state::DepthStencilSnapshot,
    dxso::{FfPsKey, FfVsKey, TextureType, VariantKey, VsSamplerKinds},
    ids::{BufferId, ProgramId},
    perf::PairShaderId,
    pipeline_state::StreamLayout,
    scratch::ScratchArena,
    shader_cache,
    streams::{bound_stream_layout, layout_stride},
};

/// Pixel sampler stages represented by captured stage bindings.
pub const STAGE_COUNT: usize = 16;

/// Rows in each captured floating-point shader constant file.
pub const CONSTANT_ROWS: usize = 256;

/// Where the vertex data comes from for this draw.
pub enum VertexSource {
    /// `DrawPrimitiveUP` / `DrawIndexedPrimitiveUP`: inline user pointer.
    ///
    /// Captured once into retained API frame scratch per draw. `stride` is the
    /// application's `VertexStreamZeroStride` — the true per-vertex span,
    /// which can exceed the vertex declaration's min-extent when the app's
    /// vertex struct carries padding past the declared elements (e.g. a
    /// `FLOAT1` TEXCOORD element over a `float texcoord[4]` field). The
    /// pipeline's vertex layout must step by this stride, not the declaration
    /// extent, or every vertex past the first is fetched from the wrong offset
    /// and the primitive degenerates.
    Up {
        bytes: ScratchSlice,
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
/// CPU addresses and lengths keep their native width until the wire boundary.
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
    pub backing_ptr: usize,
    pub backing_len: usize,
    /// The backing allocation's identity (see `PageBox::generation`).
    pub backing_generation: u64,
    pub offset: u32,
    pub stride: u32,
    /// Raw `SetStreamSourceFreq` word of this stream (flags + count).
    pub freq: u32,
}

/// How one stream is fed for a draw.
pub enum StreamFeed<'a> {
    /// Stream 0 of a UP draw: inline bytes at the call's stride.
    Inline { stride: u32 },
    /// A bound vertex buffer.
    Buffer(&'a StreamBinding),
    /// Read by the declaration, nothing bound: zeros.
    Null,
}

impl VertexSource {
    /// How `stream` is fed, for a stream the declaration reads.
    pub fn feed(&self, stream: u32) -> StreamFeed<'_> {
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
    pub fn bindings(&self) -> impl Iterator<Item = &StreamBinding> {
        let (first, extra) = match self {
            Self::Up { .. } => (None, &[][..]),
            Self::Bound { first, extra, .. } => (Some(first), &extra[..]),
        };
        first.into_iter().chain(extra.iter())
    }
}

/// Zero bytes fed to a stream the declaration reads but nothing is bound to.
///
/// Bound inline at the stream's slot under a `Constant` layout, so every
/// vertex and instance reads offset 0 and sees zeros, the value a D3D9 vertex
/// reads from an unbound stream. Sized to the `setVertexBytes` limit; a
/// declaration whose extent on one stream exceeds it is not something a
/// 16-element declaration can produce.
pub static NULL_STREAM_ZEROS: [u8; 4096] = [0; 4096];

/// The vertex buffer layouts of a draw, one per stream the declaration reads.
///
/// Unread streams stay [`StreamLayout::UNUSED`] so the pipeline key is
/// canonical.
#[must_use]
pub fn stream_layouts(
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
            StreamFeed::Buffer(b) => bound_stream_layout(b.stride, extent, b.freq),
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
        backing_ptr: usize,
        backing_len: usize,
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
    /// The API captures bytes once into retained frame scratch; native replay
    /// reads that span directly when populating its Metal upload ring. The
    /// indices are absolute (base vertex 0), paired with `VertexSource::Up`.
    Up {
        bytes: ScratchSlice,
        index_count: u32,
        index_type: IndexType,
    },
}

pub struct StageBinding {
    pub texture_id: crate::ids::TextureId,
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
    assert!(STAGE_COUNT <= crate::sampler_state::LOD_BIAS_SLOTS);
};

/// Per-draw varying parameters.
///
/// Carried by `Op::Draw` or `Op::DrawWithSnapshot`; consumed by `emit_draw`
/// which combines them with the encoder's `CurrentSnapshot` to issue the
/// actual GPU commands. State shared with the previous draw (RS, textures,
/// constants, etc.) lives in `CurrentSnapshot` and is updated via separate
/// draw snapshots only when a dirty bit fires.
pub struct DrawOp {
    pub metal_prim: PrimitiveType,
    pub vertex_source: VertexSource,
    pub index_source: IndexSource,
}

// CPU backing addresses stay native-width in the API-to-encoder queue.
// Widening them before the wire boundary adds two unused words per buffer
// to every op slot in a 32-bit process.
#[cfg(target_pointer_width = "32")]
const _: () = {
    assert!(core::mem::size_of::<StreamBinding>() <= 40);
    assert!(core::mem::size_of::<IndexSource>() <= 40);
    assert!(core::mem::size_of::<DrawOp>() <= 104);
};

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
    ptr: NonNull<VertexAttrDesc>,
    len: u32,
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
    /// Bind initialized attributes to a frame-retained arena token.
    ///
    /// # Safety
    ///
    /// `ptr` addresses `len` initialized attributes. Keep them immutable and
    /// allocated until every copy of the returned token is forgotten.
    #[must_use]
    pub const unsafe fn new(
        ptr: NonNull<VertexAttrDesc>,
        len: u32,
        extents: [u32; MAX_STREAMS as usize],
        used_streams: u16,
        vdecl_hash: u64,
    ) -> Self {
        Self {
            ptr,
            len,
            extents,
            used_streams,
            vdecl_hash,
        }
    }

    #[must_use]
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
pub struct CurrentSnapshotPtr(NonNull<CurrentSnapshot>);

// SAFETY: see `AttrSnapshot`. The CurrentSnapshot struct lives in the
// current frame's ScratchArena owned by the encoder thread for the
// duration of `run_frame`.
unsafe impl Send for CurrentSnapshotPtr {}

impl CurrentSnapshotPtr {
    /// Bind a snapshot to its frame-retained arena.
    ///
    /// # Safety
    ///
    /// `ptr` addresses an initialized `CurrentSnapshot`. Keep it and all referenced
    /// storage immutable and allocated until every copy of this token is forgotten.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<CurrentSnapshot>) -> Self {
        Self(ptr)
    }

    /// Raw `*mut CurrentSnapshot` for lifetime-laundered reads inside `emit_draw`.
    ///
    /// Direct `as_ref` is intentionally not provided — `as_ref` returns
    /// `&CurrentSnapshot` whose lifetime is tied to `self`, which in turn
    /// lives on `FrameEncoder` and prevents the usual `&mut enc` reborrows.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut CurrentSnapshot {
        self.0.as_ptr()
    }
}

/// Cached pointer to a scratch-allocated `RenderStateSnapshot`.
///
/// Wrapped in a newtype so it can be `Copy` while making the unsafe deref
/// site explicit at the read.
#[derive(Clone, Copy)]
pub struct RenderStatePtr(NonNull<RenderStateSnapshot>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for RenderStatePtr {}

impl RenderStatePtr {
    /// Bind a snapshot to its frame-retained arena.
    ///
    /// # Safety
    ///
    /// `ptr` addresses an initialized `RenderStateSnapshot`. Keep it and all referenced
    /// storage immutable and allocated until every copy of this token is forgotten.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<RenderStateSnapshot>) -> Self {
        Self(ptr)
    }

    #[must_use]
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
pub struct VsSourcePtr(NonNull<VsSource>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for VsSourcePtr {}

impl VsSourcePtr {
    /// Bind a snapshot to its frame-retained arena.
    ///
    /// # Safety
    ///
    /// `ptr` addresses an initialized `VsSource`. Keep it and all referenced
    /// storage immutable and allocated until every copy of this token is forgotten.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<VsSource>) -> Self {
        Self(ptr)
    }

    #[must_use]
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
pub struct PsSourcePtr(NonNull<PsSource>);

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for PsSourcePtr {}

impl PsSourcePtr {
    /// Bind a snapshot to its frame-retained arena.
    ///
    /// # Safety
    ///
    /// `ptr` addresses an initialized `PsSource`. Keep it and all referenced
    /// storage immutable and allocated until every copy of this token is forgotten.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<PsSource>) -> Self {
        Self(ptr)
    }

    #[must_use]
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
    mask: u16,
    bindings: NonNull<StageBinding>,
}

// SAFETY: see `AttrSnapshot`.
unsafe impl Send for StageBindingsPtr {}

impl StageBindingsPtr {
    /// Reference an initialized packed stage array without copying it.
    ///
    /// # Safety
    /// `bindings` must point to `mask.count_ones()` initialized `StageBinding`s,
    /// ordered by ascending set bit. Keep that storage immutable and allocated
    /// until every copy of this token is forgotten. An empty mask permits a
    /// dangling pointer because iteration never dereferences it.
    #[must_use]
    pub const unsafe fn from_raw_parts(mask: u16, bindings: NonNull<StageBinding>) -> Self {
        Self { mask, bindings }
    }

    #[must_use]
    pub const fn mask(&self) -> u16 {
        self.mask
    }

    /// Iterator over `(stage_index, &StageBinding)` pairs in ascending stage order.
    ///
    /// Skips unbound stages — callers that built handle arrays indexed by
    /// stage must still seed those defaults (`[0; STAGE_COUNT]`) before
    /// iterating.
    #[must_use]
    pub const fn iter(&self) -> StageBindingsIter<'_> {
        StageBindingsIter {
            mask: self.mask,
            base: self.bindings,
            next_packed_idx: 0,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a> IntoIterator for &'a StageBindingsPtr {
    type Item = (u32, &'a StageBinding);
    type IntoIter = StageBindingsIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
/// field gains a non-trivial `Drop`. Keep the arena and copied values immutable
/// and allocated until every copy of the returned token is forgotten.
/// `packed.len()` must equal `mask.count_ones()`.
///
/// # Panics
///
/// Panics if the packed length disagrees with the mask in a debug build.
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
/// `Option<CurrentSnapshotPtr>` that a draw carrying changed state
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
    /// Per-draw `VsDraw` uniform (`crate::vs_draw`): point size state.
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
    #[must_use]
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
    #[must_use]
    pub fn pair_id(&self) -> PairShaderId {
        PairShaderId {
            is_programmable: matches!(self, Self::Programmable { .. }),
            hash: self.disk_key(),
        }
    }
}

impl PsKey {
    #[must_use]
    pub fn disk_key(&self) -> u64 {
        match self {
            Self::Programmable { ps_id, variant } => {
                ps_source_disk_key_programmable(*ps_id, *variant)
            }
            Self::FixedFunction { ff, variant } => ps_source_disk_key_ff(ff, *variant),
        }
    }

    #[must_use]
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
#[must_use]
pub const fn null_texture_kind(ty: TextureType) -> NullTextureKind {
    match ty {
        TextureType::Texture3D => NullTextureKind::Texture3D,
        TextureType::TextureCube => NullTextureKind::TextureCube,
        TextureType::Texture2D | TextureType::Unknown => NullTextureKind::Texture2D,
    }
}

#[must_use]
pub fn vs_source_disk_key_ff(ff: &FfVsKey) -> u64 {
    shader_cache::ff_key_hash(ff)
}

#[must_use]
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
    #[must_use]
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
    #[must_use]
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
        /// Used fragment constant rows, cached from `FfPsKey::constant_rows`.
        ///
        /// Zero skips slot 15; one is texture factor only. Larger extents
        /// include stage constants without enlarging the snapshot or key.
        constant_rows: u8,
    },
}

impl PsSource {
    #[must_use]
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
    #[must_use]
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

/// Immutable slice token into the current frame or encoder scratch arena.
///
/// Replaces `Vec<u8>` for per-draw constants (VS / PS / alpha-ref /
/// fog-color). The API thread captures tokens into its frame arena; the
/// encoder snapshots constant mirrors into its own arena. Both allocate via
/// [`arena_alloc_bytes`]. The encoder reads through [`ScratchSlice::as_slice`]
/// or hands the raw `(ptr, len)` to a Metal `set_*_bytes_at` command.
///
/// # Safety invariants
///
/// 1. The bytes are immutable and live in the current `FrameData` or
///    `FrameEncoder` scratch arena. Tokens do not extend the arena's lifetime.
/// 2. Binding caches forget these tokens on every fresh render encoder and
///    before frame scratch is cleared or transferred to a submit payload.
/// 3. Commands may retain the raw pointer until submission completes. The
///    submitted frame and payload own both arenas for that entire interval.
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
    /// asserts the pointer is live in the owning frame or encoder arena
    /// and that the referenced bytes are initialised and immutable.
    ///
    /// # Safety
    ///
    /// Keep all referenced bytes allocated, initialized and immutable until
    /// every copy of the returned token and every derived reference is forgotten.
    #[must_use]
    pub const unsafe fn from_raw_parts(ptr: NonNull<u8>, len: u32) -> Self {
        Self { ptr, len }
    }

    /// Raw pointer + byte count suitable for the encoder's `set_*_bytes_at` commands.
    ///
    /// Pointer stays valid while the owning frame or payload retains its
    /// arena unchanged (see type-level invariants).
    #[must_use]
    pub fn as_raw(&self) -> (u64, u32) {
        (self.ptr.as_ptr() as u64, self.len)
    }

    /// Slice view for encoder-side byte reads.
    ///
    /// The caller keeps the owning arena live and unchanged for the borrow;
    /// the token itself does not enforce the arena's lifetime.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: per type-level invariants, `ptr` points to `len`
        // bytes in a live arena whenever a `ScratchSlice` is in scope.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }
}

impl AsRef<[u8]> for ScratchSlice {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Copy `bytes` into the given frame or encoder arena and return a `ScratchSlice` view.
///
/// Empty inputs short-circuit to [`ScratchSlice::EMPTY`] so the encoder
/// skips the bind cleanly.
///
/// # Panics
///
/// Panics if `bytes.len()` exceeds `u32::MAX`. Unreachable —
/// per-draw constant buffers are bounded by the 8 KB VS+PS budget.
///
/// # Safety
///
/// Keep the arena and allocated bytes immutable and live until every copy of
/// the returned token and every derived reference is forgotten.
pub unsafe fn arena_alloc_bytes(scratch: &mut ScratchArena, bytes: &[u8]) -> ScratchSlice {
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
    pub pipeline_rs: crate::pipeline_state::PipelineRsBits,
    pub depth_scissor: DepthScissorFlags,
    /// Depth + stencil test, the sole input to the `MTLDepthStencilState` cache.
    ///
    /// `depth_scissor`'s `DEPTH_ENABLE` / `DEPTH_WRITE` bits are derived from
    /// this same snapshot, so the two cannot drift apart.
    pub depth_stencil_state: DepthStencilSnapshot,
    pub cull_mode: u8,
    pub fill_mode: u8,
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
    /// `crate::convert::d3d_depth_bias_to_clip`.
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
    /// `crate::multisample::SAMPLE_MASK_ALL` when the state has no
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
#[must_use]
pub const fn build_alpha_ref_bytes(variant: VariantKey, alpha_ref: f32) -> ([u8; 4], usize) {
    if variant.alpha_func == 0 || variant.alpha_func == 8 {
        return ([0u8; 4], 0);
    }
    (alpha_ref.to_le_bytes(), 4)
}
