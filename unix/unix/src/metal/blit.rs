//! "Blit-quad" pipeline used to translate a *scaling* D3D9 `StretchRect`.
//!
//! D3D9's `IDirect3DDevice9::StretchRect` can resize the source onto the
//! destination (a filtered up/down-scale). Metal's `MTLBlitCommandEncoder`
//! only does 1:1 copies, so a size-mismatch `StretchRect` can't ride the blit
//! path the 1:1 case uses. Instead the PE side opens a render pass on the
//! destination texture (`loadAction = Load` to preserve content outside the
//! destination rect), binds the source texture + a sampler, sets the Metal
//! viewport + scissor to the destination rect, pushes the source→destination
//! texcoord transform via `setVertexBytes`, and draws a single fullscreen
//! triangle that samples the source across the destination rect.
//!
//! This module is the VS/PS pair + per-destination-format pipeline for that
//! quad. It mirrors `clear_quad.rs`: one cached `MTLLibrary` + functions,
//! and a `HashMap<color_format, MTLRenderPipelineState*>` of pipelines.
//! Pipelines + library are process-lifetime (leaked via `Retained::into_raw`)
//! — same posture as `clear_quad` / `present`.

use std::sync::{Mutex, OnceLock};

use mtld3d_shared::{
    EnsureBlitPipelineParams, MetalHandle,
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
use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// MSL source for the blit-quad library.
///
/// Vertex stage `mtld3d_blit_vs` synthesises a fullscreen triangle from
/// `vertex_id` (0..3). The visible region covers exactly the Metal viewport
/// (which the PE side sets to the destination rect). It computes a normalised
/// quad coordinate `q` in `[0,1]^2` (top-left origin: `q=(0,0)` at the
/// viewport's top-left, matching Metal's framebuffer + texture-sample origin),
/// then maps it across the source rect via the `xform` transform from
/// `setVertexBytes` slot 0: `texcoord = q * xform.xy + xform.zw`, where
/// `xform.xy = srcRect.size / srcTexSize` and `xform.zw = srcRect.origin /
/// srcTexSize`. So `q=(0,0)` samples the source rect's top-left texel and
/// `q=(1,1)` its bottom-right — a 1:1-orientation up/down-scale, no flip.
///
/// Fragment stage `mtld3d_blit_ps` samples the bound source texture (fragment
/// texture slot 0) with the bound sampler (slot 0) at `texcoord` and returns
/// the colour. The PE side chooses a POINT or LINEAR sampler from the D3D9
/// filter. Its `float4` uniform carries the source mip level in `.x` and the
/// source decode in `.y`: a packed YUV source (`YUY2` / `UYVY`, backed by an
/// RG8 texture) is fetched per macropixel and converted to RGB, unfiltered.
const BLIT_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;

struct BlitVsOut {
    float4 position [[position]];
    float2 texcoord;
};

vertex BlitVsOut mtld3d_blit_vs(
    uint vid [[vertex_id]],
    constant float4 &xform [[buffer(0)]]
) {
    // Fullscreen triangle, normalised quad coord q (top-left origin):
    //   vid=0 -> q=(0,0), vid=1 -> q=(2,0), vid=2 -> q=(0,2).
    // Visible region q in [0,1]^2 covers the whole Metal viewport.
    float2 q = float2(float((vid << 1u) & 2u), float(vid & 2u));
    // Clip position: q.x in [0,2] -> x in [-1,3]; q.y=0 (top) -> y=+1,
    // q.y=2 -> y=-3. Metal NDC +Y maps to the top of the framebuffer, so
    // q=(0,0) lands at the viewport's top-left.
    float2 pos = float2(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0);
    BlitVsOut out;
    out.position = float4(pos, 0.0, 1.0);
    // Map the quad coord across the source rect (normalised to the source
    // texture size) so the destination samples exactly srcRect.
    out.texcoord = q * xform.xy + xform.zw;
    return out;
}

fragment float4 mtld3d_blit_ps(
    BlitVsOut in [[stage_in]],
    texture2d<float> src [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float4 &src_level [[buffer(0)]]
) {
    // src_level.x is the source mip level (the sampler's point mip filter
    // makes the explicit level exact); src_level.y is the source decode,
    // 0 = sample as-is, 1 = YUY2, 2 = UYVY (mtld3d_core BlitDecode).
    uint decode = uint(src_level.y);
    if (decode == 0u) {
        return src.sample(samp, in.texcoord, level(src_level.x));
    }
    // Packed 4:2:2 YUV backed by an RG8 texture: one texel per pixel, the
    // even/odd texel pair is one macropixel. YUY2 texels are (Y0,U) (Y1,V),
    // UYVY texels are (U,Y0) (V,Y1). Luma comes from the pixel's own texel,
    // chroma from its pair, all fetched unfiltered: a linear sample across
    // the pair would mix U into V.
    uint lvl = uint(src_level.x);
    float2 size = float2(max(src.get_width(lvl), 1u), max(src.get_height(lvl), 1u));
    uint2 texel = uint2(clamp(in.texcoord * size, float2(0.0), size - 1.0));
    uint even_x = texel.x & ~1u;
    uint odd_x = min(even_x + 1u, uint(size.x) - 1u);
    float2 own = src.read(texel, lvl).rg;
    float2 even = src.read(uint2(even_x, texel.y), lvl).rg;
    float2 odd = src.read(uint2(odd_x, texel.y), lvl).rg;
    bool yuy2 = decode == 1u;
    float y = yuy2 ? own.r : own.g;
    float u = yuy2 ? even.g : even.r;
    float v = yuy2 ? odd.g : odd.r;
    // Reduced-range Y'CbCr to RGB (BT.601 coefficients, luma scaled from
    // [16, 235], chroma centred on 128); the CPU twin is
    // mtld3d_core::stretch_rect::yuv_to_rgb8, keep them in step.
    float l = (y - 0.063) * 1.164;
    float cu = u - 0.5;
    float cv = v - 0.5;
    float3 rgb = float3(l + 1.596 * cv, l - 0.392 * cu - 0.813 * cv, l + 2.017 * cu);
    return float4(saturate(rgb), 1.0);
}
";

/// VS entry-point name. Must match the `vertex` function in `BLIT_MSL`.
const VS_NAME: &str = "mtld3d_blit_vs";
/// PS entry-point name. Must match the `fragment` function in `BLIT_MSL`.
const PS_NAME: &str = "mtld3d_blit_ps";

struct BlitCache {
    vs_fn: MetalHandle<MTLFunctionKind>,
    ps_fn: MetalHandle<MTLFunctionKind>,
    /// Keyed by `(destination colour format, pass sample count)`.
    pipelines: Mutex<FxHashMap<(PixelFormat, u32), MetalHandle<MTLRenderPipelineStateKind>>>,
}

static CACHE: OnceLock<Option<BlitCache>> = OnceLock::new();

/// Lazy create-or-fetch of the blit-quad pipeline for the requested destination colour format.
///
/// Returns `None` on any compile / pipeline-create failure; the PE side then
/// leaves the size-mismatch `StretchRect` rejected (the 1:1 path is
/// unaffected).
pub fn ensure_blit_pipeline(
    params: &EnsureBlitPipelineParams,
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let device = params.device_handle.into_retained()?;

    let cache = CACHE
        .get_or_init(|| build_library_and_functions(&device))
        .as_ref()?;

    let key = (params.color_format, params.sample_count.max(1));
    {
        let pipelines = cache.pipelines.lock().ok()?;
        if let Some(&handle) = pipelines.get(&key) {
            return Some(handle);
        }
    }

    let handle = build_pipeline(&device, cache, key)?;
    let mut pipelines = cache.pipelines.lock().ok()?;
    Some(*pipelines.entry(key).or_insert(handle))
}

fn build_library_and_functions(device: &ProtocolObject<dyn MTLDevice>) -> Option<BlitCache> {
    let source = NSString::from_str(BLIT_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Pin `mathMode = Fast` for parity with the other inline compile sites
    // here; the blit shader does a single texture sample so the math mode is
    // immaterial to correctness.
    options.setMathMode(MTLMathMode::Fast);
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "blit-quad: MSL compilation failed: {e}"
            );
            return None;
        }
    };
    library.setLabel(Some(&NSString::from_str("mtld3d-blit-quad")));

    let vs = library.newFunctionWithName(&NSString::from_str(VS_NAME))?;
    let ps = library.newFunctionWithName(&NSString::from_str(PS_NAME))?;

    // Leak the function handles for process lifetime so `build_pipeline` can
    // re-derive the `MTLFunction`s per pipeline-build (same posture as
    // `clear_quad`). The library stays alive via the function refs; the
    // function refs stay alive via the pipeline states once built.
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let vs_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(vs) as u64) };
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let ps_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(ps) as u64) };
    drop(library);

    Some(BlitCache {
        vs_fn: vs_handle,
        ps_fn: ps_handle,
        pipelines: Mutex::new(FxHashMap::default()),
    })
}

fn build_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &BlitCache,
    key: (PixelFormat, u32),
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let (color_format, sample_count) = key;
    let vs = cache.vs_fn.into_retained()?;
    let ps = cache.ps_fn.into_retained()?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs));
    desc.setFragmentFunction(Some(&ps));
    desc.setRasterSampleCount(sample_count as usize);
    // SAFETY: `colorAttachments()` returns a non-null
    // `MTLRenderPipelineColorAttachmentDescriptorArray`; subscript 0 is always
    // valid.
    let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setPixelFormat(mtl_pixel_format(color_format));
    // No depth attachment: the blit quad never writes depth and the PE side
    // opens the destination pass with no depth texture bound. Declaring a
    // depth format here would make Metal reject the pipeline against the
    // depth-less render pass.

    let label = format!("mtld3d-blit-quad c={color_format:?} s={sample_count}");
    desc.setLabel(Some(&NSString::from_str(&label)));

    let pipeline = match device.newRenderPipelineStateWithDescriptor_error(&desc) {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "blit-quad: pipeline creation failed ({label}): {e}"
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

#[cfg(test)]
mod tests;
