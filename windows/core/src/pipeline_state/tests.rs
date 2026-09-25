//! Unit tests for D3D9 to Metal pipeline-state translation.
//!
//! One test mutates a default `PipelineSnapshot` field by field and asserts each change produces
//! a different `PipelineKey`, so unlike draws cannot share a cached pipeline. Others cover
//! normalisation and the wire format: an absent extra target drops out of the key, an extra target
//! the shader never writes gets an empty write mask while target 0 keeps its render-state mask,
//! destination-alpha factors clamp on an alpha-less target, and the wire params match the key.
//! Blend factors left over while blending is off, and declarations that resolve to the same
//! vertex attributes, collapse onto one key.

use mtld3d_shared::mtl::VertexFormat;
use mtld3d_types::{
    D3DBLEND_DESTALPHA, D3DBLEND_INVDESTALPHA, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE,
    D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DBLENDOP_ADD, D3DBLENDOP_REVSUBTRACT, D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD,
    D3DFVF_TEX1, D3DFVF_XYZ, D3DVERTEXELEMENT9,
};

use super::*;
use crate::convert::{fvf_to_elements, hash_elements, resolve_attrs_for_ff};

/// A position-shaped attribute: register 0, stream 0, offset 0, three floats.
const FLOAT3_AT_0: VertexAttrDesc = VertexAttrDesc {
    attr_index: 0,
    buffer_index: 0,
    offset: 0,
    format: VertexFormat::Float3,
};

/// Key of `s` for a draw reading [`FLOAT3_AT_0`] alone.
fn key_of(s: &PipelineSnapshot) -> PipelineKey {
    key_from_snapshot(s, &[FLOAT3_AT_0])
}

/// D3D enum constant at the snapshot's narrow width.
fn narrow(v: u32) -> u8 {
    u8::try_from(v).expect("D3D9 enum render-state value ≤ u8::MAX")
}

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
            src_blend: narrow(D3DBLEND_SRCALPHA),
            dst_blend: narrow(D3DBLEND_INVSRCALPHA),
            blend_op: narrow(D3DBLENDOP_ADD),
            src_blend_alpha: narrow(D3DBLEND_ONE),
            dst_blend_alpha: narrow(D3DBLEND_ZERO),
            blend_op_alpha: narrow(D3DBLENDOP_ADD),
            color_write_mask: 0xF,
            color_write_mask_ext: [0xF; 3],
        },
        extra: ExtraColorAttachments::NONE,
        ps_color_out_mask: 0b1,
        sample_count: 1,
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
    let k1 = key_of(&with_rt1());
    assert_ne!(key_of(&base()), k1, "presence");
    let mutate = |f: fn(&mut PipelineSnapshot)| {
        let mut s = with_rt1();
        f(&mut s);
        key_of(&s)
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
    assert_eq!(key_of(&unwritten), key_of(&masked));
    assert_eq!(
        key_of(&unwritten).extra_write_masks[0],
        ColorWriteMask::empty()
    );
    // Target 0 keeps the RS mask regardless of the written bit.
    assert_eq!(key_of(&unwritten).color_write_mask, ColorWriteMask::ALL);
}

#[test]
fn extra_target_blend_factors_clamp_on_their_own_alpha() {
    let mut s = with_rt1();
    s.rs.src_blend = narrow(D3DBLEND_DESTALPHA);
    s.rs.dst_blend = narrow(D3DBLEND_INVDESTALPHA);
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
    let k0 = key_of(&base());
    let mutate = |f: fn(&mut PipelineSnapshot)| {
        let mut s = base();
        f(&mut s);
        key_of(&s)
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
    assert_ne!(k0, key_from_snapshot(&base(), &[]), "vertex attributes");
    assert_eq!(
        k0,
        mutate(|s| s.vdecl_hash = 0xFACE),
        "vdecl_hash is not keyed"
    );
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
        mutate(|s| s.attach.remove(PipelineAttachFlags::HAS_COLOR_OUTPUT)),
        "has_color_output"
    );
    assert_ne!(k0, mutate(|s| s.sample_count = 4), "sample_count");

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
    let k_sep = key_of(&s_sep);
    let mutate_sep = |f: fn(&mut PipelineSnapshot)| {
        let mut s = s_sep.clone();
        f(&mut s);
        key_of(&s)
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
    with_alpha.rs.src_blend = narrow(D3DBLEND_DESTALPHA);
    with_alpha.rs.dst_blend = narrow(D3DBLEND_INVDESTALPHA);
    let k_alpha = key_of(&with_alpha);
    assert_eq!(k_alpha.src_blend, BlendFactor::DestinationAlpha);
    assert_eq!(k_alpha.dst_blend, BlendFactor::OneMinusDestinationAlpha);

    let mut no_alpha = with_alpha;
    no_alpha.attach.remove(PipelineAttachFlags::COLOR_HAS_ALPHA);
    let k_no_alpha = key_of(&no_alpha);
    assert_eq!(k_no_alpha.src_blend, BlendFactor::One);
    assert_eq!(k_no_alpha.dst_blend, BlendFactor::Zero);

    // X8 and A8 pipelines must not collide in the cache.
    assert_ne!(k_alpha, k_no_alpha, "X8 vs A8 destalpha pipeline key");

    // Non-destination-alpha factors are unaffected by the RT alpha bit.
    let mut src_alpha = base();
    src_alpha.rs.src_blend = narrow(D3DBLEND_SRCALPHA);
    let k_src = key_of(&src_alpha);
    let mut src_alpha_no_a = src_alpha;
    src_alpha_no_a
        .attach
        .remove(PipelineAttachFlags::COLOR_HAS_ALPHA);
    assert_eq!(k_src, key_of(&src_alpha_no_a));
}

#[test]
fn params_match_key_on_default_snapshot() {
    // Sanity: params_from_snapshot is not smuggling different
    // values than key_from_snapshot. Any downstream divergence on
    // these fields would be a silent bug.
    let s = base();
    let k = key_of(&s);
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

#[test]
fn alpha_to_coverage_keys_only_multisampled_pipelines_and_reaches_wire() {
    let mut off = base();
    let mut on = base();
    on.rs.flags.insert(PipelineRsFlags::ALPHA_TO_COVERAGE);
    assert_eq!(key_of(&on), key_of(&off));
    for count in [2, 4, 8] {
        off.sample_count = count;
        on.sample_count = count;
        on.attach = off.attach;
        assert_ne!(key_of(&on), key_of(&off));
        for snapshot in [&off, &on] {
            let key = key_of(snapshot);
            let params = params_from_snapshot(&PipelineBuildInputs {
                snapshot,
                vertex_attrs: &[],
                vertex_layouts: &[],
                device_handle: MetalHandle::NULL,
            });
            assert_eq!(params.alpha_to_coverage != 0, key.alpha_to_coverage);
        }
        on.attach.remove(PipelineAttachFlags::HAS_COLOR_OUTPUT);
        assert!(
            key_of(&on).alpha_to_coverage,
            "depth-only sibling keeps coverage"
        );
    }
}

#[test]
fn render_target_0_is_written_only_under_a_mask_and_an_oc0_write() {
    let mut rs = base().rs;
    for (mask, ps_color_out_mask, written) in [
        (0xF, 0b01, true),
        (0xF, 0b10, false),
        (0x0, 0b01, false),
        (0x0, 0b10, false),
    ] {
        rs.color_write_mask = mask;
        assert_eq!(
            rs.writes_rt0(ps_color_out_mask),
            written,
            "mask {mask:#x}, shader outputs {ps_color_out_mask:#b}"
        );
    }
}

#[test]
fn removing_the_colour_output_matches_a_pass_without_colour_attachments() {
    let mut snapshot = base();
    snapshot.extra.present_mask = 0b1;
    snapshot.remove_color_output();
    assert!(!snapshot.has_color_output());
    assert_eq!(snapshot.extra, ExtraColorAttachments::NONE);
}

/// Wire params of `s` with no vertex input, for comparing blend fields.
fn params_of(s: &PipelineSnapshot) -> CreateRenderPipelineParams {
    params_from_snapshot(&PipelineBuildInputs {
        snapshot: s,
        vertex_attrs: &[],
        vertex_layouts: &[],
        device_handle: MetalHandle::NULL,
    })
}

/// The blend fields of the wire params, target 0 then targets 1..3.
fn blend_fields(p: &CreateRenderPipelineParams) -> Vec<u32> {
    let mut fields = vec![
        p.blend_enable,
        p.src_blend as u32,
        p.dst_blend as u32,
        p.blend_op as u32,
        p.src_blend_alpha as u32,
        p.dst_blend_alpha as u32,
        p.blend_op_alpha as u32,
        p.separate_alpha_blend_enable,
    ];
    for extra in &p.extra {
        fields.extend([
            extra.src_blend as u32,
            extra.dst_blend as u32,
            extra.src_blend_alpha as u32,
            extra.dst_blend_alpha as u32,
        ]);
    }
    fields
}

#[test]
fn blend_off_ignores_stale_factors_in_key_and_params() {
    let mut plain = with_rt1();
    plain.rs.flags.remove(PipelineRsFlags::BLEND_ENABLE);
    let mut stale = plain.clone();
    stale.rs.src_blend = narrow(D3DBLEND_DESTALPHA);
    stale.rs.dst_blend = narrow(D3DBLEND_INVDESTALPHA);
    stale.rs.blend_op = narrow(D3DBLENDOP_REVSUBTRACT);
    stale.rs.src_blend_alpha = narrow(D3DBLEND_SRCALPHA);
    stale.rs.dst_blend_alpha = narrow(D3DBLEND_INVSRCALPHA);
    stale.rs.blend_op_alpha = narrow(D3DBLENDOP_REVSUBTRACT);
    stale.rs.flags.insert(PipelineRsFlags::SEPARATE_ALPHA_BLEND);
    // The per-target alpha clamp only changes blend factors.
    stale.extra.has_alpha_mask = 0;
    stale.attach.remove(PipelineAttachFlags::COLOR_HAS_ALPHA);
    assert_eq!(key_of(&plain), key_of(&stale));
    assert_eq!(
        blend_fields(&params_of(&plain)),
        blend_fields(&params_of(&stale))
    );
    let p = params_of(&stale);
    assert_eq!(p.blend_enable, 0);
    assert_eq!(p.src_blend, BlendFactor::One);
    assert_eq!(p.dst_blend, BlendFactor::Zero);
    assert_eq!(p.blend_op, BlendOperation::Add);
    assert_eq!(p.src_blend_alpha, BlendFactor::One);
    assert_eq!(p.dst_blend_alpha, BlendFactor::Zero);
    assert_eq!(p.blend_op_alpha, BlendOperation::Add);
    assert_eq!(p.separate_alpha_blend_enable, 0);
    assert_eq!(p.extra[0].src_blend, BlendFactor::One);
    assert_eq!(p.extra[0].dst_blend, BlendFactor::Zero);
}

#[test]
fn blend_on_keys_every_factor_difference() {
    let on = with_rt1();
    let k = key_of(&on);
    let mutate = |f: fn(&mut PipelineSnapshot)| {
        let mut s = with_rt1();
        f(&mut s);
        key_of(&s)
    };
    assert_ne!(k, mutate(|s| s.rs.src_blend = narrow(D3DBLEND_ONE)), "src");
    assert_ne!(k, mutate(|s| s.rs.dst_blend = narrow(D3DBLEND_ZERO)), "dst");
    assert_ne!(
        k,
        mutate(|s| s.rs.blend_op = narrow(D3DBLENDOP_REVSUBTRACT)),
        "op"
    );
    assert_ne!(
        k,
        mutate(|s| {
            s.rs.flags.insert(PipelineRsFlags::SEPARATE_ALPHA_BLEND);
            s.rs.src_blend_alpha = narrow(D3DBLEND_SRCALPHA);
        }),
        "separate alpha"
    );
    let mut dest_alpha = with_rt1();
    dest_alpha.rs.src_blend = narrow(D3DBLEND_DESTALPHA);
    let mut dest_alpha_rt1_no_alpha = dest_alpha.clone();
    dest_alpha_rt1_no_alpha.extra.has_alpha_mask = 0;
    assert_ne!(
        key_of(&dest_alpha),
        key_of(&dest_alpha_rt1_no_alpha),
        "render target 1 alpha clamp"
    );
    assert_ne!(
        blend_fields(&params_of(&dest_alpha)),
        blend_fields(&params_of(&dest_alpha_rt1_no_alpha))
    );
}

/// Key of `base()` for `elements`, resolved under the fixed-function convention.
fn key_for_decl(elements: &[D3DVERTEXELEMENT9], vdecl_hash: u64) -> PipelineKey {
    let mut s = base();
    s.vdecl_hash = vdecl_hash;
    key_from_snapshot(&s, &resolve_attrs_for_ff(elements).attrs)
}

const fn element(offset: u16, type_: u8, usage: u8) -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    }
}

#[test]
fn declarations_resolving_to_the_same_attributes_share_a_key() {
    // An FVF and the declaration spelling out the same elements.
    let (fvf_elements, _) = fvf_to_elements(D3DFVF_XYZ | D3DFVF_TEX1);
    let decl = [
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        element(12, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
    ];
    let fvf_hash = u64::from(D3DFVF_XYZ | D3DFVF_TEX1);
    assert_ne!(fvf_hash, hash_elements(&decl));
    let from_decl = key_for_decl(&decl, hash_elements(&decl));
    assert_eq!(key_for_decl(&fvf_elements, fvf_hash), from_decl);

    // A texcoord moved to another offset, and one with another format, do not.
    let moved = [
        decl[0],
        element(16, D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_TEXCOORD),
    ];
    let widened = [
        decl[0],
        element(12, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_TEXCOORD),
    ];
    let with_normal = [
        decl[0],
        decl[1],
        element(20, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL),
    ];
    for (other, what) in [
        (&moved[..], "offset"),
        (&widened[..], "format"),
        (&with_normal[..], "extra attribute"),
    ] {
        assert_ne!(
            from_decl,
            key_for_decl(other, hash_elements(other)),
            "{what}"
        );
    }
}

#[test]
fn attrs_hash_covers_every_attribute_field() {
    let attr = FLOAT3_AT_0;
    let h = VertexAttrsHash::from_attrs(&[attr]);
    for (changed, what) in [
        (
            VertexAttrDesc {
                attr_index: 1,
                ..attr
            },
            "attr_index",
        ),
        (
            VertexAttrDesc {
                buffer_index: 1,
                ..attr
            },
            "buffer_index",
        ),
        (VertexAttrDesc { offset: 4, ..attr }, "offset"),
        (
            VertexAttrDesc {
                format: VertexFormat::Float2,
                ..attr
            },
            "format",
        ),
    ] {
        assert_ne!(h, VertexAttrsHash::from_attrs(&[changed]), "{what}");
    }
    assert_ne!(h, VertexAttrsHash::from_attrs(&[attr, attr]), "count");
}
