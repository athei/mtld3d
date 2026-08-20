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
        const SRGB_WRITE = 1 << 2;
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
    #[inline]
    #[must_use]
    pub const fn srgb_write_enable(&self) -> bool {
        self.flags.contains(PipelineRsFlags::SRGB_WRITE)
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
    srgb_write_enable: u32,
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
        srgb_write_enable: u32::from(s.rs.srgb_write_enable()),
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
        srgb_write_enable: u32::from(s.rs.srgb_write_enable()),
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
mod tests {
    use super::*;

    /// Default snapshot with sane non-zero values.
    ///
    /// So the tests below exercise "change this field to something
    /// different" rather than "change this field from zero" — the latter
    /// can false-positive when a raw-D3D value falls through to a
    /// fallback.
    /// Stream 0 only, per-vertex at `stride` bytes.
    fn stream0(stride: u32) -> [StreamLayout; MAX_STREAMS as usize] {
        let mut layouts = [StreamLayout::UNUSED; MAX_STREAMS as usize];
        layouts[0] = StreamLayout {
            stride,
            step: VertexStepFunction::PerVertex,
            step_rate: 1,
        };
        layouts
    }

    fn base() -> PipelineSnapshot {
        PipelineSnapshot {
            // SAFETY: tests; opaque values never dereferenced.
            vs_fn: unsafe { MetalHandle::new(0x1000) },
            // SAFETY: tests; opaque values never dereferenced.
            ps_fn: unsafe { MetalHandle::new(0x2000) },
            vdecl_hash: 0x3000,
            stream_layouts: stream0(32),
            color_format: PixelFormat::Bgra8Unorm,
            // Bgra8Unorm here models an A8R8G8B8 RT, so the default (has-alpha)
            // blend path is exercised — destination-alpha factors pass through
            // unclamped, byte-identical to the pre-`COLOR_HAS_ALPHA` behaviour.
            attach: PipelineAttachFlags::HAS_DEPTH
                | PipelineAttachFlags::HAS_COLOR_OUTPUT
                | PipelineAttachFlags::COLOR_HAS_ALPHA,
            rs: PipelineRsBits {
                flags: PipelineRsFlags::BLEND_ENABLE,
                src_blend: 5,       // D3DBLEND_SRCALPHA
                dst_blend: 6,       // D3DBLEND_INVSRCALPHA
                blend_op: 1,        // D3DBLENDOP_ADD
                src_blend_alpha: 2, // D3DBLEND_ONE
                dst_blend_alpha: 1, // D3DBLEND_ZERO
                blend_op_alpha: 1,  // D3DBLENDOP_ADD
                color_write_mask: 0xF,
                color_write_mask_ext: [0xF; 3],
            },
            extra: ExtraColorAttachments::NONE,
            ps_color_out_mask: 0b1,
        }
    }

    /// `base()` with render target 1 bound as an `R8Unorm` target the PS writes.
    fn with_rt1() -> PipelineSnapshot {
        let mut s = base();
        s.extra = ExtraColorAttachments {
            formats: [PixelFormat::R8Unorm; 3],
            present_mask: 0b001,
            has_alpha_mask: 0b001,
        };
        s.ps_color_out_mask = 0b11;
        s
    }

    #[test]
    fn extra_targets_key_presence_format_mask_and_alpha() {
        let k1 = key_from_snapshot(&with_rt1());
        assert_ne!(key_from_snapshot(&base()), k1, "presence");
        let mutate = |f: fn(&mut PipelineSnapshot)| {
            let mut s = with_rt1();
            f(&mut s);
            key_from_snapshot(&s)
        };
        assert_ne!(
            k1,
            mutate(|s| s.extra.formats[0] = PixelFormat::Bgra8Unorm),
            "format"
        );
        assert_ne!(
            k1,
            mutate(|s| s.rs.color_write_mask_ext[0] = 0x1),
            "write mask"
        );
        assert_ne!(k1, mutate(|s| s.extra.has_alpha_mask = 0), "has_alpha");
        // Slot 2 is absent: its format, mask and alpha bit are normalised
        // away so a single-target draw never fragments on them.
        assert_eq!(
            k1,
            mutate(|s| s.extra.formats[1] = PixelFormat::Bgra8Unorm),
            "absent format"
        );
        assert_eq!(
            k1,
            mutate(|s| s.rs.color_write_mask_ext[1] = 0x1),
            "absent mask"
        );
        assert_eq!(
            k1,
            mutate(|s| s.extra.has_alpha_mask |= 0b010),
            "absent alpha"
        );
    }

    #[test]
    fn unwritten_extra_target_gets_an_empty_write_mask() {
        // A target the shader never writes must keep its contents: it keys
        // and binds exactly like an RS mask of zero.
        let mut unwritten = with_rt1();
        unwritten.ps_color_out_mask = 0b01;
        let mut masked = with_rt1();
        masked.rs.color_write_mask_ext[0] = 0;
        assert_eq!(key_from_snapshot(&unwritten), key_from_snapshot(&masked));
        assert_eq!(
            key_from_snapshot(&unwritten).extra_write_masks[0],
            ColorWriteMask::empty()
        );
        // Target 0 keeps the RS mask regardless of the written bit.
        assert_eq!(
            key_from_snapshot(&unwritten).color_write_mask,
            ColorWriteMask::ALL
        );
    }

    #[test]
    fn extra_target_blend_factors_clamp_on_their_own_alpha() {
        let mut s = with_rt1();
        s.rs.src_blend = 7; // D3DBLEND_DESTALPHA
        s.rs.dst_blend = 8; // D3DBLEND_INVDESTALPHA
        s.extra.has_alpha_mask = 0; // RT1 is alpha-less, RT0 keeps alpha
        let attrs: [VertexAttrDesc; 0] = [];
        let layouts = vertex_layouts_from_snapshot(&s);
        // SAFETY: tests; opaque values never dereferenced.
        let dev = unsafe { MetalHandle::new(0xDEAD) };
        let p = params_from_snapshot(&PipelineBuildInputs {
            snapshot: &s,
            vertex_attrs: &attrs,
            vertex_layouts: &layouts,
            device_handle: dev,
        });
        assert_eq!(p.src_blend, BlendFactor::DestinationAlpha);
        assert_eq!(p.extra_present_mask, 0b001);
        assert_eq!(p.extra[0].src_blend, BlendFactor::One);
        assert_eq!(p.extra[0].dst_blend, BlendFactor::Zero);
        assert_eq!(p.extra[0].format, PixelFormat::R8Unorm);
        assert_eq!(p.extra[0].write_mask, ColorWriteMask::ALL);
        assert_eq!(p.extra[1].write_mask, ColorWriteMask::empty());
    }

    /// Per-field static invariant check.
    ///
    /// Mutating one snapshot field must produce a different `PipelineKey`.
    /// If this test fails for a field, the pipeline cache is colliding and
    /// draws with that field differing silently share a pipeline — the
    /// exact bug class this module exists to prevent.
    ///
    /// Each assertion pairs the base value with a second value chosen
    /// so the translation helper produces a *different* Metal enum (not
    /// the fallback).
    #[test]
    fn key_changes_on_every_field() {
        let k0 = key_from_snapshot(&base());
        let mutate = |f: fn(&mut PipelineSnapshot)| {
            let mut s = base();
            f(&mut s);
            key_from_snapshot(&s)
        };

        assert_ne!(
            k0,
            // SAFETY: tests; opaque values never dereferenced.
            mutate(|s| s.vs_fn = unsafe { MetalHandle::new(0xFACE) }),
            "vs_fn"
        );
        assert_ne!(
            k0,
            // SAFETY: tests; opaque values never dereferenced.
            mutate(|s| s.ps_fn = unsafe { MetalHandle::new(0xFACE) }),
            "ps_fn"
        );
        assert_ne!(k0, mutate(|s| s.vdecl_hash = 0xFACE), "vdecl_hash");
        assert_ne!(
            k0,
            mutate(|s| s.stream_layouts[0].stride = 64),
            "stream 0 stride"
        );
        assert_ne!(
            k0,
            mutate(|s| s.stream_layouts[1] = StreamLayout {
                stride: 12,
                step: VertexStepFunction::PerVertex,
                step_rate: 1,
            }),
            "stream 1 present"
        );
        assert_ne!(
            k0,
            mutate(|s| s.stream_layouts[0].step = VertexStepFunction::PerInstance),
            "stream 0 step function"
        );
        assert_ne!(
            k0,
            mutate(|s| s.stream_layouts[0].step_rate = 2),
            "stream 0 step rate"
        );
        assert_ne!(
            k0,
            mutate(|s| s.color_format = PixelFormat::Rgba16Float),
            "color_format"
        );
        assert_ne!(
            k0,
            mutate(|s| s.attach.remove(PipelineAttachFlags::HAS_DEPTH)),
            "has_depth"
        );
        assert_ne!(
            k0,
            mutate(|s| s.attach.insert(PipelineAttachFlags::HAS_STENCIL)),
            "has_stencil"
        );
        assert_ne!(
            k0,
            mutate(|s| s.rs.flags.remove(PipelineRsFlags::BLEND_ENABLE)),
            "blend_enable"
        );
        assert_ne!(k0, mutate(|s| s.rs.src_blend = 2), "src_blend"); // → One
        assert_ne!(k0, mutate(|s| s.rs.dst_blend = 2), "dst_blend"); // → One
        assert_ne!(k0, mutate(|s| s.rs.blend_op = 5), "blend_op"); // → Max
        assert_ne!(
            k0,
            mutate(|s| s.rs.color_write_mask = 0x1),
            "color_write_mask"
        );
        assert_ne!(
            k0,
            mutate(|s| s.rs.flags.insert(PipelineRsFlags::SRGB_WRITE)),
            "srgb_write_enable"
        );
        assert_ne!(
            k0,
            mutate(|s| s.attach.remove(PipelineAttachFlags::HAS_COLOR_OUTPUT)),
            "has_color_output"
        );

        // Separate-alpha path: enabling it changes the effective alpha
        // factors even though the per-alpha fields were already set.
        assert_ne!(
            k0,
            mutate(|s| s.rs.flags.insert(PipelineRsFlags::SEPARATE_ALPHA_BLEND)),
            "separate_alpha_blend_enable"
        );

        // When separate-alpha IS enabled, mutating an alpha field must
        // change the key; when it's NOT enabled, alpha fields mirror
        // RGB and mutating them is a no-op (correct — nothing to key).
        let mut s_sep = base();
        s_sep.rs.flags.insert(PipelineRsFlags::SEPARATE_ALPHA_BLEND);
        let k_sep = key_from_snapshot(&s_sep);
        let mutate_sep = |f: fn(&mut PipelineSnapshot)| {
            let mut s = s_sep.clone();
            f(&mut s);
            key_from_snapshot(&s)
        };
        assert_ne!(
            k_sep,
            mutate_sep(|s| s.rs.src_blend_alpha = 5),
            "src_blend_alpha under sep-alpha"
        ); // → SourceAlpha
        assert_ne!(
            k_sep,
            mutate_sep(|s| s.rs.dst_blend_alpha = 5),
            "dst_blend_alpha under sep-alpha"
        );
        assert_ne!(
            k_sep,
            mutate_sep(|s| s.rs.blend_op_alpha = 5),
            "blend_op_alpha under sep-alpha"
        ); // → Max
    }

    /// On an alpha-less RT (X8R8G8B8), `D3DBLEND_DESTALPHA` / `INVDESTALPHA` clamp to One / Zero.
    ///
    /// On an alpha-bearing RT they pass through as `DestinationAlpha` /
    /// `OneMinusDestinationAlpha`. The clamp flows through the remapped
    /// factors into the key, so the two RTs hash distinctly with no
    /// dedicated key field.
    #[test]
    fn destination_alpha_clamps_on_no_alpha_rt() {
        let mut with_alpha = base();
        with_alpha.rs.src_blend = 7; // D3DBLEND_DESTALPHA
        with_alpha.rs.dst_blend = 8; // D3DBLEND_INVDESTALPHA
        let k_alpha = key_from_snapshot(&with_alpha);
        assert_eq!(k_alpha.src_blend, BlendFactor::DestinationAlpha);
        assert_eq!(k_alpha.dst_blend, BlendFactor::OneMinusDestinationAlpha);

        let mut no_alpha = with_alpha;
        no_alpha.attach.remove(PipelineAttachFlags::COLOR_HAS_ALPHA);
        let k_no_alpha = key_from_snapshot(&no_alpha);
        assert_eq!(k_no_alpha.src_blend, BlendFactor::One);
        assert_eq!(k_no_alpha.dst_blend, BlendFactor::Zero);

        // X8 and A8 pipelines must not collide in the cache.
        assert_ne!(k_alpha, k_no_alpha, "X8 vs A8 destalpha pipeline key");

        // Non-destination-alpha factors are unaffected by the RT alpha bit.
        let mut src_alpha = base();
        src_alpha.rs.src_blend = 5; // D3DBLEND_SRCALPHA
        let k_src = key_from_snapshot(&src_alpha);
        let mut src_alpha_no_a = src_alpha;
        src_alpha_no_a
            .attach
            .remove(PipelineAttachFlags::COLOR_HAS_ALPHA);
        assert_eq!(k_src, key_from_snapshot(&src_alpha_no_a));
    }

    #[test]
    fn params_match_key_on_default_snapshot() {
        // Sanity: params_from_snapshot is not smuggling different
        // values than key_from_snapshot. Any downstream divergence on
        // these fields would be a silent bug.
        let s = base();
        let k = key_from_snapshot(&s);
        let attrs: [VertexAttrDesc; 0] = [];
        let layouts = vertex_layouts_from_snapshot(&s);
        // SAFETY: tests; opaque values never dereferenced.
        let dev = unsafe { MetalHandle::new(0xDEAD) };
        let p = params_from_snapshot(&PipelineBuildInputs {
            snapshot: &s,
            vertex_attrs: &attrs,
            vertex_layouts: &layouts,
            device_handle: dev,
        });
        assert_eq!(p.device_handle, dev);
        assert_eq!(p.vertex_layout_count, 1);
        assert_eq!(layouts[0].buffer_index, 0);
        assert_eq!(layouts[0].stride, 32);
        assert_eq!(layouts[0].step_function, VertexStepFunction::PerVertex);
        assert_eq!(layouts[0].step_rate, 1);
        assert_eq!(p.vs_fn_handle, k.vs_fn);
        assert_eq!(p.ps_fn_handle, k.ps_fn);
        assert_eq!(p.src_blend, k.src_blend);
        assert_eq!(p.dst_blend, k.dst_blend);
        assert_eq!(p.blend_op, k.blend_op);
        assert_eq!(p.src_blend_alpha, k.src_blend_alpha);
        assert_eq!(p.dst_blend_alpha, k.dst_blend_alpha);
        assert_eq!(p.blend_op_alpha, k.blend_op_alpha);
        assert_eq!(p.separate_alpha_blend_enable, k.separate_alpha_blend_enable);
        assert_eq!(p.srgb_write_enable, k.srgb_write_enable);
        assert_eq!(p.color_write_mask, k.color_write_mask);
        assert_eq!(p.has_depth, k.has_depth);
        assert_eq!(p.has_stencil, k.has_stencil);
        assert_eq!(p.color_format, k.color_format);
        assert_eq!(p.has_color_output, k.has_color_output);
        assert_eq!(p.extra_present_mask, u32::from(k.extra_present_mask));
        for i in 0..3 {
            assert_eq!(p.extra[i].format, k.extra_formats[i]);
            assert_eq!(p.extra[i].write_mask, k.extra_write_masks[i]);
        }
    }

    #[test]
    fn wire_layouts_carry_used_streams_with_their_slot() {
        // Streams 0 and 2 used, stream 1 not: two wire entries, each at the
        // Metal slot of its D3D9 stream, with the per-instance step intact.
        let mut s = base();
        s.stream_layouts[2] = StreamLayout {
            stride: 16,
            step: VertexStepFunction::PerInstance,
            step_rate: 3,
        };
        let layouts = vertex_layouts_from_snapshot(&s);
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[1].buffer_index, 2);
        assert_eq!(layouts[1].stride, 16);
        assert_eq!(layouts[1].step_function, VertexStepFunction::PerInstance);
        assert_eq!(layouts[1].step_rate, 3);
    }
}
