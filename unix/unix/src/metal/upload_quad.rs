//! "Upload-quad" pipeline that writes one texture upload from its staging buffer.
//!
//! A `copyFromBuffer:toTexture:` blit constrains the source row stride: Metal
//! requires it to be at least
//! `minimumLinearTextureAlignmentForPixelFormat:` (16 bytes on Apple
//! Silicon), and it can only move bytes verbatim, so a 3 bpp `R8G8B8` source
//! cannot be handed to a `Bgra8Unorm` texture at all. Both constraints
//! disappear when the
//! staging slab is read as a shader argument instead: the fragment function
//! addresses it by texel with the D3D9 row pitch, and decodes the texel on
//! the way to the destination.
//!
//! The PE side opens a render pass on the destination mip (one pass per cube
//! face / volume slice), scopes it to the dirty rect with the viewport and
//! scissor, binds the staging `MTLBuffer` plus a `uint4` of upload
//! parameters, and draws a single fullscreen triangle.
//!
//! This module is the VS/PS pair + per-destination-format pipeline for that
//! quad, mirroring `blit` and `clear_quad`: one cached `MTLLibrary` +
//! functions, and a `FxHashMap<color_format, MTLRenderPipelineState*>` of
//! pipelines. Pipelines + library are process-lifetime (leaked via
//! `Retained::into_raw`).

use std::sync::{Mutex, OnceLock};

use mtld3d_shared::{
    MetalHandle,
    mtl::PixelFormat,
    mtl_handle::{MTLFunctionKind, MTLRenderPipelineStateKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
    MTLRenderPipelineDescriptor,
};
use rustc_hash::FxHashMap;

use super::texture::mtl_pixel_format;
use crate::{
    LOG_TARGET,
    metal::handle::{IntoRetained, keep_first},
};

/// MSL source for the upload-quad library.
///
/// Vertex stage `mtld3d_upload_vs` synthesises a fullscreen triangle from
/// `vertex_id` (0..3). Metal maps NDC onto the viewport, which the PE side
/// set to the dirty rect, so the rasterized region is exactly that rect and
/// `[[position]]` arrives in destination-texel coordinates.
///
/// Fragment stage `mtld3d_upload_ps` turns that position into a byte address
/// in the staging slab bound at buffer slot 1: `base + y * pitch + x * bpp`,
/// where `args` at slot 0 carries `(base, pitch, decode, bpp)`. A 2D upload
/// writes the dirty rect at the same coordinates it occupies in the staging,
/// so `base` is zero there and non-zero only for a volume slice. Bytes are
/// read one at a time, which is what removes every alignment constraint on
/// the pitch and on the base offset.
///
/// `decode` selects the source layout. The 24-bit layout stores a whole byte
/// per channel and only gains an opaque alpha; the copy decode hands the
/// bytes through in the destination's channel order.
/// `mtld3d_core::upload_pass::UploadDecode` owns the numbering.
const UPLOAD_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;

vertex float4 mtld3d_upload_vs(uint vid [[vertex_id]]) {
    // vid=0 -> q=(0,0), vid=1 -> q=(2,0), vid=2 -> q=(0,2); the visible
    // region q in [0,1]^2 covers the whole viewport.
    float2 q = float2(float((vid << 1u) & 2u), float(vid & 2u));
    return float4(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0, 0.0, 1.0);
}

fragment float4 mtld3d_upload_ps(
    float4 pos [[position]],
    constant uint4 &args [[buffer(0)]],
    device const uchar *src [[buffer(1)]]
) {
    uint2 texel = uint2(pos.xy);
    uint decode = args.z;
    uint bpp = args.w;
    uint addr = args.x + texel.y * args.y + texel.x * bpp;
    if (decode == 3u) {
        // Verbatim copy of an already-native layout: four bytes per texel in
        // BGRA memory order, which is what the destination stores, so the
        // fragment output puts them back in RGBA logical order.
        return float4(
            float(src[addr + 2u]),
            float(src[addr + 1u]),
            float(src[addr]),
            float(src[addr + 3u])
        ) / 255.0;
    }
    // R8G8B8 (decode 5): three bytes per texel in B, G, R memory order, one
    // unorm byte each, with no stored alpha (opaque).
    return float4(
        float(src[addr + 2u]),
        float(src[addr + 1u]),
        float(src[addr]),
        255.0
    ) / 255.0;
}
";

/// VS entry-point name. Must match the `vertex` function in `UPLOAD_MSL`.
const VS_NAME: &str = "mtld3d_upload_vs";
/// PS entry-point name. Must match the `fragment` function in `UPLOAD_MSL`.
const PS_NAME: &str = "mtld3d_upload_ps";

struct UploadCache {
    vs_fn: MetalHandle<MTLFunctionKind>,
    ps_fn: MetalHandle<MTLFunctionKind>,
    pipelines: Mutex<FxHashMap<PixelFormat, MetalHandle<MTLRenderPipelineStateKind>>>,
}

static CACHE: OnceLock<Option<UploadCache>> = OnceLock::new();

/// Lazy create-or-fetch of the upload-quad pipeline for the requested destination format.
///
/// Returns `None` on any compile / pipeline-create failure; the PE side then
/// falls back to the blit upload path for that texture.
pub fn ensure_upload_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    color_format: PixelFormat,
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let cache = CACHE
        .get_or_init(|| build_library_and_functions(device))
        .as_ref()?;

    {
        let pipelines = cache.pipelines.lock().ok()?;
        if let Some(&handle) = pipelines.get(&color_format) {
            return Some(handle);
        }
    }

    let handle = build_pipeline(device, cache, color_format)?;
    let mut pipelines = cache.pipelines.lock().ok()?;
    // SAFETY: `build_pipeline` handed `handle` the only retain on its
    // pipeline, and nothing else copied it.
    Some(unsafe { keep_first(&mut pipelines, color_format, handle) })
}

fn build_library_and_functions(device: &ProtocolObject<dyn MTLDevice>) -> Option<UploadCache> {
    let source = NSString::from_str(UPLOAD_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Pin `mathMode = Fast` for parity with the other inline compile sites
    // here. The upload shader's only arithmetic is integer channel widening
    // plus one divide by a constant, so the math mode cannot move a result.
    options.setMathMode(MTLMathMode::Fast);
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "upload-quad: MSL compilation failed: {e}"
            );
            return None;
        }
    };
    library.setLabel(Some(&NSString::from_str("mtld3d-upload-quad")));

    let vs = library.newFunctionWithName(&NSString::from_str(VS_NAME))?;
    let ps = library.newFunctionWithName(&NSString::from_str(PS_NAME))?;

    // Leak the function handles for process lifetime so `build_pipeline` can
    // re-derive the `MTLFunction`s per pipeline-build (same posture as
    // `blit` / `clear_quad`). The library stays alive via the function refs;
    // the function refs stay alive via the pipeline states once built.
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let vs_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(vs) as u64) };
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let ps_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(ps) as u64) };
    drop(library);

    Some(UploadCache {
        vs_fn: vs_handle,
        ps_fn: ps_handle,
        pipelines: Mutex::new(FxHashMap::default()),
    })
}

fn build_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &UploadCache,
    color_format: PixelFormat,
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let vs = cache.vs_fn.into_retained()?;
    let ps = cache.ps_fn.into_retained()?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs));
    desc.setFragmentFunction(Some(&ps));
    // SAFETY: `colorAttachments()` returns a non-null
    // `MTLRenderPipelineColorAttachmentDescriptorArray`; subscript 0 is always
    // valid.
    let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setPixelFormat(mtl_pixel_format(color_format));
    // No depth attachment: the upload quad never writes depth and the PE side
    // opens the destination pass with no depth texture bound.

    let label = format!("mtld3d-upload-quad c={color_format:?}");
    desc.setLabel(Some(&NSString::from_str(&label)));

    let pipeline = match device.newRenderPipelineStateWithDescriptor_error(&desc) {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "upload-quad: pipeline creation failed ({label}): {e}"
            );
            return None;
        }
    };
    drop(vs);
    drop(ps);
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    Some(unsafe {
        MetalHandle::<MTLRenderPipelineStateKind>::new(Retained::into_raw(pipeline) as u64)
    })
}
