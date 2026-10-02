//! Native render-pipeline creation against the fixed-function vertex layout.
//!
//! Pins that every vertex format a `BLENDINDICES` element can carry, from an
//! FVF or a declaration, builds a pipeline with the FF VS that reads it.

use mtld3d_core::{
    convert::{
        decl_type_to_metal_format, ff_vs_layout_from_elements, fvf_to_elements,
        resolve_attrs_for_ff,
    },
    dxso::{VariantKey, emit_ps_ff_named, emit_vs_ff_named},
    ff_state::FfState,
    pipeline_state::{
        ExtraColorAttachmentDescription, PipelineAttachFlags, PipelineDescription, PipelineRsFlags,
    },
};
use mtld3d_shared::{
    MetalHandle, VertexBufferLayoutDesc,
    mtl::{BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, StageTag, VertexStepFunction},
    mtl_handle::MTLDeviceKind,
    perf::{PipelineTimings, ShaderTimings},
};
use mtld3d_types::{
    D3DDECLMETHOD_DEFAULT, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT1, D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4, D3DDECLTYPE_FLOAT16_2, D3DDECLTYPE_FLOAT16_4,
    D3DDECLTYPE_SHORT2, D3DDECLTYPE_SHORT2N, D3DDECLTYPE_SHORT4, D3DDECLTYPE_SHORT4N,
    D3DDECLTYPE_UBYTE4, D3DDECLTYPE_UBYTE4N, D3DDECLTYPE_USHORT2N, D3DDECLTYPE_USHORT4N,
    D3DDECLUSAGE_BLENDINDICES, D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_POSITION,
    D3DFVF_LASTBETA_D3DCOLOR, D3DFVF_LASTBETA_UBYTE4, D3DFVF_XYZB2, D3DFVF_XYZB3, D3DFVF_XYZB5,
    D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_VERTEXBLEND, D3DVBF_1WEIGHTS,
    D3DVERTEXELEMENT9, render_state_defaults,
};
use objc2::rc::Retained;
use objc2_metal::MTLCreateSystemDefaultDevice;

use super::{create_render_pipeline, destroy_render_pipeline};
use crate::metal::{
    compile_shader_library, destroy_function, destroy_library, handle::IntoRetained,
};

const fn element(offset: u16, type_: u8, usage: u8) -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: D3DDECLMETHOD_DEFAULT,
        usage,
        usage_index: 0,
    }
}

/// Build the FF pipeline a blended draw of `elements` uses; `true` when Metal accepts it.
fn ff_blend_pipeline_builds(
    device: MetalHandle<MTLDeviceKind>,
    elements: &[D3DVERTEXELEMENT9],
    indexed: bool,
) -> bool {
    let mut states = render_state_defaults();
    states[D3DRS_LIGHTING as usize] = 0;
    states[D3DRS_VERTEXBLEND as usize] = D3DVBF_1WEIGHTS;
    states[D3DRS_INDEXEDVERTEXBLENDENABLE as usize] = u32::from(indexed);
    let ff = FfState::new();
    let layout = ff_vs_layout_from_elements(elements, true);
    let vs_key = ff.build_vs_key(&states, layout, 0);
    assert!(vs_key.vertex_blend_count > 0 && vs_key.declared_indices());
    let vs_msl = emit_vs_ff_named(&vs_key, "blend_probe_vs");
    let ps_msl = emit_ps_ff_named(
        &ff.build_ps_key(&states, 0),
        VariantKey::default(),
        "blend_probe_ps",
    );
    let mut shader_timings = ShaderTimings::new();
    let (vs_lib, vs_fn) = compile_shader_library(
        device,
        &vs_msl,
        StageTag::Vertex,
        "blend_probe_vs",
        &mut shader_timings,
    )
    .unwrap_or_else(|| panic!("FF VS compiles\n{vs_msl}"));
    let (ps_lib, ps_fn) = compile_shader_library(
        device,
        &ps_msl,
        StageTag::Fragment,
        "blend_probe_ps",
        &mut shader_timings,
    )
    .unwrap_or_else(|| panic!("FF PS compiles\n{ps_msl}"));
    let resolved = resolve_attrs_for_ff(elements);
    let layouts = [VertexBufferLayoutDesc {
        buffer_index: 0,
        stride: resolved.extents[0],
        step_function: VertexStepFunction::PerVertex,
        step_rate: 1,
    }];
    let extra = || ExtraColorAttachmentDescription {
        format: PixelFormat::Bgra8Unorm,
        write_mask: ColorWriteMask::ALL,
        src_blend: BlendFactor::One,
        dst_blend: BlendFactor::Zero,
        src_blend_alpha: BlendFactor::One,
        dst_blend_alpha: BlendFactor::Zero,
    };
    let description = PipelineDescription {
        vs_fn_handle: vs_fn,
        ps_fn_handle: ps_fn,
        vertex_attrs: &resolved.attrs,
        vertex_layouts: &layouts,
        flags: PipelineRsFlags::empty(),
        attach: PipelineAttachFlags::HAS_COLOR_OUTPUT,
        src_blend: BlendFactor::One,
        dst_blend: BlendFactor::Zero,
        blend_op: BlendOperation::Add,
        src_blend_alpha: BlendFactor::One,
        dst_blend_alpha: BlendFactor::Zero,
        blend_op_alpha: BlendOperation::Add,
        color_write_mask: ColorWriteMask::ALL,
        color_format: PixelFormat::Bgra8Unorm,
        extra_present_mask: 0,
        sample_count: 1,
        extra: [extra(), extra(), extra()],
    };
    let pipeline = create_render_pipeline(
        &device.into_retained().expect("device handle"),
        &description,
        &mut PipelineTimings::new(),
    );
    let built = pipeline.is_some();
    if let Some(pipeline) = pipeline {
        destroy_render_pipeline(pipeline.raw());
    }
    destroy_function(vs_fn.raw());
    destroy_function(ps_fn.raw());
    destroy_library(vs_lib.raw());
    destroy_library(ps_lib.raw());
    built
}

#[test]
fn every_ff_blend_index_format_builds_a_pipeline() {
    mtld3d_shared::init_logger();
    let device = MTLCreateSystemDefaultDevice().expect("Metal device for the pipeline test");
    // SAFETY: the device's retain stays alive until every call below returns.
    let handle = unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) };
    let mut cases: Vec<(String, Vec<D3DVERTEXELEMENT9>)> = [
        (
            "XYZB2 | LASTBETA_UBYTE4",
            D3DFVF_XYZB2 | D3DFVF_LASTBETA_UBYTE4,
        ),
        (
            "XYZB2 | LASTBETA_D3DCOLOR",
            D3DFVF_XYZB2 | D3DFVF_LASTBETA_D3DCOLOR,
        ),
        (
            "XYZB3 | LASTBETA_D3DCOLOR",
            D3DFVF_XYZB3 | D3DFVF_LASTBETA_D3DCOLOR,
        ),
        ("XYZB5", D3DFVF_XYZB5),
    ]
    .into_iter()
    .map(|(name, fvf)| (format!("FVF {name}"), fvf_to_elements(fvf).0))
    .collect();
    for ty in [
        D3DDECLTYPE_FLOAT1,
        D3DDECLTYPE_FLOAT2,
        D3DDECLTYPE_FLOAT3,
        D3DDECLTYPE_FLOAT4,
        D3DDECLTYPE_D3DCOLOR,
        D3DDECLTYPE_UBYTE4,
        D3DDECLTYPE_SHORT2,
        D3DDECLTYPE_SHORT4,
        D3DDECLTYPE_UBYTE4N,
        D3DDECLTYPE_SHORT2N,
        D3DDECLTYPE_SHORT4N,
        D3DDECLTYPE_USHORT2N,
        D3DDECLTYPE_USHORT4N,
        D3DDECLTYPE_FLOAT16_2,
        D3DDECLTYPE_FLOAT16_4,
    ] {
        assert_ne!(decl_type_to_metal_format(ty).1, 0, "type {ty} has a format");
        cases.push((
            format!("declared BLENDINDICES type {ty}"),
            vec![
                element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
                element(12, D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDWEIGHT),
                element(16, ty, D3DDECLUSAGE_BLENDINDICES),
            ],
        ));
    }
    let mut refused = Vec::new();
    for (name, elements) in &cases {
        for indexed in [false, true] {
            if !ff_blend_pipeline_builds(handle, elements, indexed) {
                refused.push(format!("{name}, indexed={indexed}"));
            }
        }
    }
    assert!(
        refused.is_empty(),
        "Metal refused the FF pipeline for: {refused:#?}"
    );
}
