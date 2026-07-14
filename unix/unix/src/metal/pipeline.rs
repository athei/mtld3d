use log::{debug, error};
use mtld3d_shared::{
    CreateRenderPipelineParams, MetalHandle, VertexAttrDesc,
    mtl::{BlendOperation, VertexFormat},
    mtl_handle::MTLRenderPipelineStateKind,
};
use objc2::rc::Retained;
use objc2_metal::{
    MTLBlendOperation, MTLColorWriteMask, MTLDevice, MTLFunction, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};

use super::texture::{mtl_blend_factor, mtl_pixel_format};
use crate::{
    LOG_TARGET,
    metal::handle::{IntoRetained, ReleaseRetain},
};

/// Build an `MTLRenderPipelineState` from pre-resolved vertex/fragment functions.
///
/// The vertex descriptor is caller-supplied. Shader compilation happens
/// upstream in `compile_shader_library`; this function deals purely with
/// pipeline state. `vertex_attrs` is reconstructed at the handler boundary
/// because the FFI struct carries only a raw pointer + length.
pub fn create_render_pipeline(
    params: &CreateRenderPipelineParams,
    vertex_attrs: &[VertexAttrDesc],
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let device = params.device_handle.into_retained()?;
    let vertex_function = params.vs_fn_handle.into_retained()?;
    let fragment_function = params.ps_fn_handle.into_retained()?;

    // Vertex descriptor: one attribute entry per supplied VertexAttrDesc; a single
    // buffer layout at index 0 (multi-stream support lands with vertex declarations).
    //
    // The six `unsafe` blocks below are all objc2 typed bindings whose
    // arguments (`attr_index`, `offset`, `buffer_index`, `stride`,
    // subscript 0) are caller-validated for the vertex descriptor capacity.
    let vertex_desc = MTLVertexDescriptor::new();
    let attrs = vertex_desc.attributes();
    for a in vertex_attrs {
        // SAFETY: objc2 typed binding; subscript follows Metal's
        // unbounded attribute-array semantics.
        let entry = unsafe { attrs.objectAtIndexedSubscript(a.attr_index as usize) };
        entry.setFormat(mtl_vertex_format(a.format));
        // SAFETY: objc2 typed binding; pure accessor passthrough.
        unsafe { entry.setOffset(a.offset as usize) };
        // SAFETY: objc2 typed binding; pure accessor passthrough.
        unsafe { entry.setBufferIndex(a.buffer_index as usize) };
    }
    let buffer_layouts = vertex_desc.layouts();
    // SAFETY: objc2 typed binding; subscript 0 is always valid on the
    // buffer-layout array.
    let buffer0 = unsafe { buffer_layouts.objectAtIndexedSubscript(0) };
    // SAFETY: objc2 typed binding; pure accessor passthrough.
    unsafe { buffer0.setStride(params.vertex_stride as usize) };
    buffer0.setStepFunction(MTLVertexStepFunction::PerVertex);

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vertex_function));
    desc.setFragmentFunction(Some(&fragment_function));
    desc.setVertexDescriptor(Some(&vertex_desc));

    // Pipeline-state label = `<vs_name> + <ps_name>`. Surfaces in Xcode's
    // Frame Capture timeline + pipeline-state list views as the per-shader
    // identifier; without this every pipeline shows the generic Metal
    // default. The function names already carry kind+id (e.g.
    // `mtld3d_vs_ff_5f3a0001`), so the concatenation is unambiguous.
    {
        let vs_name = vertex_function.name();
        let ps_name = fragment_function.name();
        let suffix = if params.has_color_output == 0 {
            " no-color"
        } else {
            ""
        };
        let label = objc2_foundation::NSString::from_str(&format!("{vs_name} + {ps_name}{suffix}"));
        desc.setLabel(Some(&label));
    }

    if params.has_color_output != 0 {
        // SAFETY: `colorAttachments()` returns a non-null descriptor array;
        // subscript 0 is always valid.
        let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        // The pipeline's color format MUST equal the render pass's bound
        // texture format — Metal validates this and the mismatch is undefined
        // behaviour with the layer off (it corrupts the heap, surfacing as a
        // crash in an unrelated later teardown). The render pass binds the RT
        // texture at its own (`color_format`) format, so the pipeline matches
        // it. `D3DRS_SRGBWRITEENABLE` is intentionally NOT honoured by swapping
        // to the sRGB twin format here: the D3D9 semantics ("encode the shader
        // output linear→sRGB on write") need an sRGB render target or
        // attachment view, but we bind the plain texture — swapping only the
        // pipeline desynced the formats. Correct sRGB-write encoding (a
        // pixel-shader OETF variant) is the proper follow-up; until then
        // `SRGBWRITEENABLE` writes linear instead of corrupting the heap.
        color0.setPixelFormat(mtl_pixel_format(params.color_format));

        let mask = MTLColorWriteMask::from_bits_truncate(params.color_write_mask.bits() as usize);
        color0.setWriteMask(mask);

        if params.blend_enable != 0 {
            // D3D9 spec: separate alpha factors/ops only apply when
            // D3DRS_SEPARATEALPHABLENDENABLE is TRUE. Otherwise the RGB
            // values mirror onto alpha. The PE side pre-resolves this in
            // `pipeline_state::effective_alpha_blend` — by the time the
            // thunk arrives the alpha fields already carry the correct
            // effective values, so we just apply them unconditionally.
            color0.setBlendingEnabled(true);
            color0.setSourceRGBBlendFactor(mtl_blend_factor(params.src_blend));
            color0.setDestinationRGBBlendFactor(mtl_blend_factor(params.dst_blend));
            color0.setRgbBlendOperation(mtl_blend_op(params.blend_op));
            color0.setSourceAlphaBlendFactor(mtl_blend_factor(params.src_blend_alpha));
            color0.setDestinationAlphaBlendFactor(mtl_blend_factor(params.dst_blend_alpha));
            color0.setAlphaBlendOperation(mtl_blend_op(params.blend_op_alpha));
        }
    }

    if params.has_depth != 0 {
        // Match the framebuffer: depth-only targets (D24X8 / D16 / D32 → Depth32Float)
        // must leave stencilAttachmentPixelFormat at the Invalid default, or Metal
        // rejects the pipeline at draw time. Only D24S8 / D24FS8 pair depth+stencil.
        if params.has_stencil != 0 {
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
            desc.setStencilAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
        } else {
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
        }
    }

    let pipeline = match device.newRenderPipelineStateWithDescriptor_error(&desc) {
        Ok(pso) => pso,
        Err(e) => {
            error!(target: LOG_TARGET, "pipeline creation failed: {e}");
            return None;
        }
    };

    debug!(
        target: LOG_TARGET,
        "created render pipeline (attrs={}, stride={}, blend={}, depth={})",
        vertex_attrs.len(),
        params.vertex_stride,
        params.blend_enable,
        params.has_depth != 0
    );
    // SAFETY: `Retained::into_raw` transfers the retain into the typed handle.
    Some(unsafe {
        MetalHandle::<MTLRenderPipelineStateKind>::new(Retained::into_raw(pipeline) as u64)
    })
}

const fn mtl_blend_op(op: BlendOperation) -> MTLBlendOperation {
    match op {
        BlendOperation::Add => MTLBlendOperation::Add,
        BlendOperation::Subtract => MTLBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOperation::Min => MTLBlendOperation::Min,
        BlendOperation::Max => MTLBlendOperation::Max,
    }
}

const fn mtl_vertex_format(wire: VertexFormat) -> MTLVertexFormat {
    match wire {
        VertexFormat::Invalid => MTLVertexFormat::Invalid,
        VertexFormat::UChar4 => MTLVertexFormat::UChar4,
        VertexFormat::UChar4Normalized => MTLVertexFormat::UChar4Normalized,
        VertexFormat::UChar4NormalizedBgra => MTLVertexFormat::UChar4Normalized_BGRA,
        VertexFormat::Short2 => MTLVertexFormat::Short2,
        VertexFormat::Short4 => MTLVertexFormat::Short4,
        VertexFormat::UShort2Normalized => MTLVertexFormat::UShort2Normalized,
        VertexFormat::UShort4Normalized => MTLVertexFormat::UShort4Normalized,
        VertexFormat::Short2Normalized => MTLVertexFormat::Short2Normalized,
        VertexFormat::Short4Normalized => MTLVertexFormat::Short4Normalized,
        VertexFormat::Half2 => MTLVertexFormat::Half2,
        VertexFormat::Half4 => MTLVertexFormat::Half4,
        VertexFormat::Float => MTLVertexFormat::Float,
        VertexFormat::Float2 => MTLVertexFormat::Float2,
        VertexFormat::Float3 => MTLVertexFormat::Float3,
        VertexFormat::Float4 => MTLVertexFormat::Float4,
    }
}

/// Releases an `MTLRenderPipelineState` from a raw handle.
pub fn destroy_render_pipeline(pipeline_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has flushed the GPU and dropped
    // its only copy of `pipeline_handle`.
    let handle = unsafe { MetalHandle::<MTLRenderPipelineStateKind>::new(pipeline_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}
