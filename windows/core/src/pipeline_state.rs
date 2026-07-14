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
    CreateRenderPipelineParams, MetalHandle, VertexAttrDesc,
    mtl::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat},
    mtl_handle::MTLFunctionKind,
};

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
    pub vertex_stride: u32,
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
}

impl PipelineSnapshot {
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
    vertex_stride: u32,
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
    pub device_handle: MetalHandle<mtld3d_shared::mtl_handle::MTLDeviceKind>,
}

#[must_use]
pub fn key_from_snapshot(s: &PipelineSnapshot) -> PipelineKey {
    let (src_a, dst_a, op_a) = effective_alpha_blend(s);
    PipelineKey {
        vs_fn: s.vs_fn,
        ps_fn: s.ps_fn,
        vdecl_hash: s.vdecl_hash,
        vertex_stride: s.vertex_stride,
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
    }
}

/// Build the `CreateRenderPipelineParams` wire struct from a pipeline snapshot.
///
/// # Panics
///
/// Panics if `inputs.vertex_attrs.len()` exceeds `u32::MAX` (unreachable —
/// D3D9 caps the count at 16).
#[must_use]
pub fn params_from_snapshot(inputs: &PipelineBuildInputs<'_>) -> CreateRenderPipelineParams {
    let s = inputs.snapshot;
    let (src_a, dst_a, op_a) = effective_alpha_blend(s);
    let vertex_attr_count =
        u32::try_from(inputs.vertex_attrs.len()).expect("vertex attr count ≤ D3D9 max 16");
    CreateRenderPipelineParams {
        device_handle: inputs.device_handle,
        vs_fn_handle: s.vs_fn,
        ps_fn_handle: s.ps_fn,
        vertex_attrs_ptr: inputs.vertex_attrs.as_ptr() as u64,
        vertex_attr_count,
        vertex_stride: s.vertex_stride,
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
    fn base() -> PipelineSnapshot {
        PipelineSnapshot {
            // SAFETY: tests; opaque values never dereferenced.
            vs_fn: unsafe { MetalHandle::new(0x1000) },
            // SAFETY: tests; opaque values never dereferenced.
            ps_fn: unsafe { MetalHandle::new(0x2000) },
            vdecl_hash: 0x3000,
            vertex_stride: 32,
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
            },
        }
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
        assert_ne!(k0, mutate(|s| s.vertex_stride = 64), "vertex_stride");
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
        // SAFETY: tests; opaque values never dereferenced.
        let dev = unsafe { MetalHandle::new(0xDEAD) };
        let p = params_from_snapshot(&PipelineBuildInputs {
            snapshot: &s,
            vertex_attrs: &attrs,
            device_handle: dev,
        });
        assert_eq!(p.device_handle, dev);
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
    }
}
