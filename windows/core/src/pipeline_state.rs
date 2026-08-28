//! Single source of truth for D3D9 → Metal pipeline-state translation.
//!
//! Two functions consume the same input (`PipelineSnapshot`) and produce
//! the two outputs that must stay in lockstep — the pipeline-cache
//! `PipelineKey` and the wire-format `CreateRenderPipelineParams`. Anything
//! that can change the Metal pipeline **must** appear in both. Per-field
//! unit tests below assert the static invariant: "mutating one snapshot
//! field produces a different key". If the audit claims a D3D state is
//! Consumed but that value isn't keyed, the cache collides and draws
//! silently get the wrong pipeline — e.g. a `D3DRS_BLENDOP` that is consumed on
//! the unix side but absent from the key would collapse every blend op onto a
//! single cached pipeline.

use mtld3d_shared::{
    CreateRenderPipelineParams, ExtraColorAttachmentParams, MetalHandle, VertexAttrDesc,
    VertexBufferLayoutDesc,
    mtl::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, VertexStepFunction},
    mtl_handle::MTLFunctionKind,
};
use mtld3d_types::MAX_STREAMS;

use crate::convert::{d3d_to_metal_blend_op, d3d_to_metal_blend_rt, d3d_to_metal_write_mask};

bitflags::bitflags! {
    /// Boolean RS bits that affect pipeline identity.
    ///
    /// Shared between `PipelineSnapshot` (the pipeline cache key) and the
    /// d3d9 layer's `RenderStateSnapshot` (per-draw RS capture). Packed
    /// into a u8; each bit mirrors a D3D9 BOOL render state.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct PipelineRsFlags: u8 {
        const BLEND_ENABLE = 1 << 0;
        const SEPARATE_ALPHA_BLEND = 1 << 1;
    }
}

bitflags::bitflags! {
    /// Booleans on `PipelineSnapshot` that aren't part of `PipelineRsBits`.
    ///
    /// Attachment shape — depth/stencil presence on the bound RT, and
    /// whether the pipeline declares a color attachment. Packed into
    /// a u8 instead of three separate `bool` fields.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct PipelineAttachFlags: u8 {
        /// Bound RT has a depth attachment.
        const HAS_DEPTH = 1 << 0;
        /// Bound RT's depth attachment also carries stencil.
        const HAS_STENCIL = 1 << 1;
        /// Pipeline declares a color attachment.
        ///
        /// False for cascade caster passes where every draw has
        /// `color_write_mask == 0` so the pass runs depth-only.
        const HAS_COLOR_OUTPUT = 1 << 2;
        /// Bound color RT's D3D format has a real alpha channel.
        ///
        /// Drives the destination-alpha blend-factor clamp: when clear
        /// (e.g. X8R8G8B8, which shares `Bgra8Unorm` with A8R8G8B8)
        /// `D3DBLEND_DESTALPHA` / `INVDESTALPHA` resolve to One / Zero
        /// instead of sampling the physically-stored X byte. Set from
        /// `map_d3d_format(fmt).has_alpha()` for the bound RT.
        const COLOR_HAS_ALPHA = 1 << 3;
    }
}

/// Pipeline-identity-affecting render-state bits.
///
/// Shared between `PipelineSnapshot` (cache key) and the d3d9 layer's
/// `RenderStateSnapshot` (per-draw capture). Carries only the RS that
/// gets baked into the compiled `MTLRenderPipelineState` — blend
/// state, color-write mask, sRGB write. NOT included: depth state
/// (`MTLDepthStencilState` is a separate cache), cull / scissor /
/// blend-factor / depth-bias (per-encoder runtime state set via Metal
/// command API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineRsBits {
    pub flags: PipelineRsFlags,
    /// `D3DRS_SRCBLEND` (raw D3DBLEND value, fits u8: 1..=19).
    pub src_blend: u8,
    /// `D3DRS_DESTBLEND` (raw D3DBLEND value).
    pub dst_blend: u8,
    /// `D3DRS_BLENDOP` (raw D3DBLENDOP value, 1..=5).
    pub blend_op: u8,
    /// `D3DRS_SRCBLENDALPHA`.
    ///
    /// Only active when `flags.contains(SEPARATE_ALPHA_BLEND)`; otherwise
    /// alpha mirrors `src_blend` per D3D9 spec.
    pub src_blend_alpha: u8,
    /// `D3DRS_DESTBLENDALPHA`. Same activation rule.
    pub dst_blend_alpha: u8,
    /// `D3DRS_BLENDOPALPHA`. Same activation rule.
    pub blend_op_alpha: u8,
    /// `D3DRS_COLORWRITEENABLE` (4 D3DCOLORWRITEENABLE_* bits).
    pub color_write_mask: u8,
    /// `D3DRS_COLORWRITEENABLE1..3`, the write masks of render targets 1..3.
    ///
    /// Index `i` holds the mask for target `i + 1`. Only consulted for
    /// targets present in the pass; an absent target contributes a zero
    /// mask to the key so single-target draws never fragment the cache on
    /// these states.
    pub color_write_mask_ext: [u8; 3],
}

/// Colour attachments 1..3 of the render pass a draw lands in.
///
/// Render target 0 stays on [`PipelineSnapshot`] itself; this carries the
/// extra simultaneous render targets. `present_mask` bit `i` (0..3) says slot
/// `i + 1` is bound in the pass, `formats[i]` is its Metal format (ignored
/// when absent) and `has_alpha_mask` bit `i` mirrors `COLOR_HAS_ALPHA` for it.
/// `Copy` because the encoder copies it out of the pass state once per draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExtraColorAttachments {
    pub formats: [PixelFormat; 3],
    pub present_mask: u8,
    pub has_alpha_mask: u8,
}

impl ExtraColorAttachments {
    /// No extra attachment: the single-render-target shape.
    pub const NONE: Self = Self {
        formats: [PixelFormat::Bgra8Unorm; 3],
        present_mask: 0,
        has_alpha_mask: 0,
    };

    #[inline]
    #[must_use]
    pub const fn is_present(&self, extra_index: usize) -> bool {
        self.present_mask & (1 << extra_index) != 0
    }

    #[inline]
    #[must_use]
    pub const fn has_alpha(&self, extra_index: usize) -> bool {
        self.has_alpha_mask & (1 << extra_index) != 0
    }
}

impl Default for ExtraColorAttachments {
    fn default() -> Self {
        Self::NONE
    }
}

impl PipelineRsBits {
    #[inline]
    #[must_use]
    pub const fn blend_enable(&self) -> bool {
        self.flags.contains(PipelineRsFlags::BLEND_ENABLE)
    }
    #[inline]
    #[must_use]
    pub const fn separate_alpha_blend_enable(&self) -> bool {
        self.flags.contains(PipelineRsFlags::SEPARATE_ALPHA_BLEND)
    }
}

/// Input describing one draw's pipeline state.
///
/// Raw-D3D (where a translation helper exists) plus pre-translated
/// (where the value is already Metal-shaped). All future pipeline-keyed
/// state gets a field here.
///
/// Not `Copy`: at 48 B this is wide enough that accidental whole-struct
/// reads should be compile errors. `emit_draw` builds one snapshot per
/// draw and passes it by reference to `key_from_snapshot` and
/// `get_or_create_pipeline`. `Clone` stays for the rare explicit
/// duplication path (currently just the no-color twin in
/// `get_or_create_pipeline`).
///
/// `PartialEq`/`Eq` back the encoder's single-entry resolve memo: comparing
/// two 48 B snapshots is cheaper than rebuilding the [`PipelineKey`] (its
/// D3D→Metal translations + the cache probe), and equality implies an
/// identical key — [`key_from_snapshot`] is a pure function of the snapshot —
/// so the memo can return the cached handle directly.
#[derive(Clone, PartialEq, Eq)]
pub struct PipelineSnapshot {
    pub vs_fn: MetalHandle<MTLFunctionKind>,
    pub ps_fn: MetalHandle<MTLFunctionKind>,
    pub vdecl_hash: u64,
    /// Vertex buffer layout per D3D9 stream, indexed by stream.
    ///
    /// Canonical: a stream the draw does not read is
    /// [`StreamLayout::UNUSED`], so two draws that differ only in streams
    /// neither reads share a pipeline.
    pub stream_layouts: [StreamLayout; MAX_STREAMS as usize],
    pub color_format: PixelFormat,
    /// Attachment-shape flags: `HAS_DEPTH`, `HAS_STENCIL`, `HAS_COLOR_OUTPUT`.
    ///
    /// Packed instead of three bool fields.
    pub attach: PipelineAttachFlags,
    /// Blend / color-write / sRGB RS.
    ///
    /// The subset of D3D9 RS that affects `MTLRenderPipelineState`
    /// identity. d3d9 layer's `RenderStateSnapshot` carries an identical
    /// `PipelineRsBits` substruct so per-draw construction is one field
    /// copy.
    pub rs: PipelineRsBits,
    /// Render targets 1..3 bound in the pass.
    pub extra: ExtraColorAttachments,
    /// Bit `i` set ⇒ the bound pixel shader writes `oCi`.
    ///
    /// An extra target the shader never writes gets an empty write mask so
    /// its contents survive the draw (Metal leaves an unwritten colour
    /// output undefined). Fixed-function and SM1 shaders report bit 0.
    pub ps_color_out_mask: u8,
}

impl PipelineSnapshot {
    /// Effective Metal write mask of extra target `extra_index` (slot `extra_index + 1`).
    ///
    /// Empty when the target is absent from the pass or the shader does not
    /// write it; otherwise the D3D9 `COLORWRITEENABLEn` mask.
    fn extra_write_mask(&self, extra_index: usize) -> ColorWriteMask {
        let written = self.ps_color_out_mask & (1 << (extra_index + 1)) != 0;
        if self.extra.is_present(extra_index) && written {
            d3d_to_metal_write_mask(u32::from(self.rs.color_write_mask_ext[extra_index]))
        } else {
            ColorWriteMask::empty()
        }
    }

    /// `true` when no colour target of the pass receives a write from this draw.
    ///
    /// Render target 0 by its D3D9 mask, targets 1..3 by their effective
    /// mask (present, written by the shader, non-zero `COLORWRITEENABLEn`).
    /// Rule H builds the no-colour pipeline twin for such draws.
    #[must_use]
    pub fn writes_no_color(&self) -> bool {
        self.rs.color_write_mask == 0 && (0..3).all(|i| self.extra_write_mask(i).is_empty())
    }

    /// Metal format keyed for extra target `extra_index`, normalised when absent.
    const fn extra_format(&self, extra_index: usize) -> PixelFormat {
        if self.extra.is_present(extra_index) {
            self.extra.formats[extra_index]
        } else {
            ExtraColorAttachments::NONE.formats[extra_index]
        }
    }

    /// Blend factors for extra target `extra_index`, with its own alpha clamp.
    ///
    /// `(src, dst, src_alpha, dst_alpha)`; the blend ops are shared with
    /// target 0 (D3D9 has one blend state).
    fn extra_blend_factors(
        &self,
        extra_index: usize,
    ) -> (BlendFactor, BlendFactor, BlendFactor, BlendFactor) {
        let has_alpha = self.extra.is_present(extra_index) && self.extra.has_alpha(extra_index);
        let (src_a, dst_a, _) = effective_alpha_blend(self);
        (
            d3d_to_metal_blend_rt(u32::from(self.rs.src_blend), has_alpha),
            d3d_to_metal_blend_rt(u32::from(self.rs.dst_blend), has_alpha),
            d3d_to_metal_blend_rt(src_a, has_alpha),
            d3d_to_metal_blend_rt(dst_a, has_alpha),
        )
    }

    #[inline]
    #[must_use]
    pub const fn has_depth(&self) -> bool {
        self.attach.contains(PipelineAttachFlags::HAS_DEPTH)
    }
    #[inline]
    #[must_use]
    pub const fn has_stencil(&self) -> bool {
        self.attach.contains(PipelineAttachFlags::HAS_STENCIL)
    }
    #[inline]
    #[must_use]
    pub const fn has_color_output(&self) -> bool {
        self.attach.contains(PipelineAttachFlags::HAS_COLOR_OUTPUT)
    }
    /// Whether the bound colour RT's D3D format carries a real alpha channel.
    ///
    /// Feeds [`d3d_to_metal_blend_rt`] so destination-alpha blend factors
    /// clamp on alpha-less targets (X8R8G8B8). Its effect flows into both the
    /// key and the wire params via the remapped factors, so no extra key field
    /// is needed to keep X8 and A8 pipelines distinct.
    #[inline]
    #[must_use]
    pub const fn color_has_alpha(&self) -> bool {
        self.attach.contains(PipelineAttachFlags::COLOR_HAS_ALPHA)
    }
}

/// Cache key.
///
/// Opaque outside this module — the only consumer is the pipeline cache's
/// `HashMap<PipelineKey, u64>`, which uses the derived `Hash + Eq` on the
/// struct as a whole. Keeping fields private makes the per-field invariant
/// test (below) the sole contract between this module and every D3D9 state
/// that influences pipeline identity.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    vs_fn: MetalHandle<MTLFunctionKind>,
    ps_fn: MetalHandle<MTLFunctionKind>,
    vdecl_hash: u64,
    stream_layouts: [StreamLayout; MAX_STREAMS as usize],
    blend_enable: u32,
    src_blend: BlendFactor,
    dst_blend: BlendFactor,
    blend_op: BlendOperation,
    src_blend_alpha: BlendFactor,
    dst_blend_alpha: BlendFactor,
    blend_op_alpha: BlendOperation,
    separate_alpha_blend_enable: u32,
    color_write_mask: ColorWriteMask,
    has_depth: u32,
    has_stencil: u32,
    color_format: PixelFormat,
    has_color_output: u32,
    /// Render targets 1..3: presence, format, effective write mask, alpha clamp.
    ///
    /// The alpha bit stands in for the per-target blend factors: they differ
    /// from target 0's only through the destination-alpha clamp, which is a
    /// pure function of this bit, so keying the bit keys the factors.
    extra_present_mask: u8,
    extra_has_alpha_mask: u8,
    extra_formats: [PixelFormat; 3],
    extra_write_masks: [ColorWriteMask; 3],
}

/// Per-draw thunk-params builder input.
///
/// Adds the slice reference the wire-format struct needs
/// (`vertex_attrs_ptr` + count).
///
/// Separated from `PipelineSnapshot` so the snapshot can be borrowed
/// through this wrapper without dragging the lifetime of
/// `vertex_attrs` into the underlying type.
pub struct PipelineBuildInputs<'a> {
    pub snapshot: &'a PipelineSnapshot,
    pub vertex_attrs: &'a [VertexAttrDesc],
    /// The wire form of `snapshot.stream_layouts`, used streams only.
    ///
    /// Built by [`vertex_layouts_from_snapshot`]; the slice outlives the
    /// synchronous `CreateRenderPipeline` thunk that reads it by pointer.
    pub vertex_layouts: &'a [VertexBufferLayoutDesc],
    pub device_handle: MetalHandle<mtld3d_shared::mtl_handle::MTLDeviceKind>,
}

/// One vertex buffer layout of a pipeline: how Metal steps through a D3D9 stream.
///
/// Part of the pipeline identity: a stream bound with a different stride or
/// a different `SetStreamSourceFreq` needs a different vertex descriptor.
/// `Copy` because the snapshot carries an array of them and the draw path
/// builds that array by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamLayout {
    /// Bytes per step; never 0 for a used stream (Metal rejects it).
    pub stride: u32,
    pub step: VertexStepFunction,
    /// Instances per advance (`PerInstance`); 1 for `PerVertex`, 0 for `Constant`.
    pub step_rate: u32,
}

impl StreamLayout {
    /// The canonical value of a stream the draw does not read.
    pub const UNUSED: Self = Self {
        stride: 0,
        step: VertexStepFunction::PerVertex,
        step_rate: 0,
    };

    #[inline]
    #[must_use]
    pub const fn is_used(&self) -> bool {
        self.stride != 0
    }
}

/// The wire layouts of a snapshot, one per used stream.
///
/// Allocates; called only on a pipeline-cache miss, never per draw.
#[must_use]
pub fn vertex_layouts_from_snapshot(s: &PipelineSnapshot) -> Vec<VertexBufferLayoutDesc> {
    // `0u32..` yields the stream index as a u32 without a fallible width
    // conversion.
    (0u32..)
        .zip(s.stream_layouts.iter())
        .filter(|(_, l)| l.is_used())
        .map(|(stream, l)| VertexBufferLayoutDesc {
            buffer_index: stream,
            stride: l.stride,
            step_function: l.step,
            step_rate: l.step_rate,
        })
        .collect()
}

#[must_use]
pub fn key_from_snapshot(s: &PipelineSnapshot) -> PipelineKey {
    let (src_a, dst_a, op_a) = effective_alpha_blend(s);
    PipelineKey {
        vs_fn: s.vs_fn,
        ps_fn: s.ps_fn,
        vdecl_hash: s.vdecl_hash,
        stream_layouts: s.stream_layouts,
        blend_enable: u32::from(s.rs.blend_enable()),
        src_blend: d3d_to_metal_blend_rt(u32::from(s.rs.src_blend), s.color_has_alpha()),
        dst_blend: d3d_to_metal_blend_rt(u32::from(s.rs.dst_blend), s.color_has_alpha()),
        blend_op: d3d_to_metal_blend_op(u32::from(s.rs.blend_op)),
        src_blend_alpha: d3d_to_metal_blend_rt(src_a, s.color_has_alpha()),
        dst_blend_alpha: d3d_to_metal_blend_rt(dst_a, s.color_has_alpha()),
        blend_op_alpha: d3d_to_metal_blend_op(op_a),
        separate_alpha_blend_enable: u32::from(s.rs.separate_alpha_blend_enable()),
        color_write_mask: d3d_to_metal_write_mask(u32::from(s.rs.color_write_mask)),
        has_depth: u32::from(s.has_depth()),
        has_stencil: u32::from(s.has_stencil()),
        color_format: s.color_format,
        has_color_output: u32::from(s.has_color_output()),
        extra_present_mask: s.extra.present_mask,
        // Absent slots drop their alpha bit so the key stays canonical.
        extra_has_alpha_mask: s.extra.has_alpha_mask & s.extra.present_mask,
        extra_formats: core::array::from_fn(|i| s.extra_format(i)),
        extra_write_masks: core::array::from_fn(|i| s.extra_write_mask(i)),
    }
}

/// Build the `CreateRenderPipelineParams` wire struct from a pipeline snapshot.
///
/// # Panics
///
/// Panics if `inputs.vertex_attrs.len()` or `inputs.vertex_layouts.len()`
/// exceeds `u32::MAX` (unreachable — D3D9 caps both at 16).
#[must_use]
pub fn params_from_snapshot(inputs: &PipelineBuildInputs<'_>) -> CreateRenderPipelineParams {
    let s = inputs.snapshot;
    let (src_a, dst_a, op_a) = effective_alpha_blend(s);
    let vertex_attr_count =
        u32::try_from(inputs.vertex_attrs.len()).expect("vertex attr count ≤ D3D9 max 16");
    let vertex_layout_count =
        u32::try_from(inputs.vertex_layouts.len()).expect("vertex layout count ≤ MaxStreams");
    CreateRenderPipelineParams {
        device_handle: inputs.device_handle,
        vs_fn_handle: s.vs_fn,
        ps_fn_handle: s.ps_fn,
        vertex_attrs_ptr: inputs.vertex_attrs.as_ptr() as u64,
        vertex_layouts_ptr: inputs.vertex_layouts.as_ptr() as u64,
        vertex_attr_count,
        vertex_layout_count,
        blend_enable: u32::from(s.rs.blend_enable()),
        src_blend: d3d_to_metal_blend_rt(u32::from(s.rs.src_blend), s.color_has_alpha()),
        dst_blend: d3d_to_metal_blend_rt(u32::from(s.rs.dst_blend), s.color_has_alpha()),
        blend_op: d3d_to_metal_blend_op(u32::from(s.rs.blend_op)),
        src_blend_alpha: d3d_to_metal_blend_rt(src_a, s.color_has_alpha()),
        dst_blend_alpha: d3d_to_metal_blend_rt(dst_a, s.color_has_alpha()),
        blend_op_alpha: d3d_to_metal_blend_op(op_a),
        separate_alpha_blend_enable: u32::from(s.rs.separate_alpha_blend_enable()),
        color_write_mask: d3d_to_metal_write_mask(u32::from(s.rs.color_write_mask)),
        has_depth: u32::from(s.has_depth()),
        has_stencil: u32::from(s.has_stencil()),
        color_format: s.color_format,
        has_color_output: u32::from(s.has_color_output()),
        extra_present_mask: u32::from(s.extra.present_mask),
        extra: core::array::from_fn(|i| {
            let (src_blend, dst_blend, src_blend_alpha, dst_blend_alpha) = s.extra_blend_factors(i);
            ExtraColorAttachmentParams {
                format: s.extra_format(i),
                write_mask: s.extra_write_mask(i),
                src_blend,
                dst_blend,
                src_blend_alpha,
                dst_blend_alpha,
            }
        }),
        pipeline_handle: MetalHandle::NULL,
    }
}

/// D3D9 spec: the alpha-side blend factors / op are conditional.
///
/// They only take effect when `D3DRS_SEPARATEALPHABLENDENABLE` is TRUE.
/// Otherwise the RGB values apply to alpha too. Resolve here once so both
/// the key and the thunk params see the same effective alpha state.
fn effective_alpha_blend(s: &PipelineSnapshot) -> (u32, u32, u32) {
    if s.rs.separate_alpha_blend_enable() {
        (
            u32::from(s.rs.src_blend_alpha),
            u32::from(s.rs.dst_blend_alpha),
            u32::from(s.rs.blend_op_alpha),
        )
    } else {
        (
            u32::from(s.rs.src_blend),
            u32::from(s.rs.dst_blend),
            u32::from(s.rs.blend_op),
        )
    }
}

#[cfg(test)]
mod tests;
