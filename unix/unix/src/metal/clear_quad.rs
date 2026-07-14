//! "Clear-quad" pipeline used to translate D3D9 mid-pass `Clear` calls.
//!
//! D3D9's `IDirect3DDevice9::Clear` is viewport-clipped and can fire at
//! any point during rendering. Metal has no equivalent — clears can
//! only happen as `MTLRenderPassDescriptor.loadAction = Clear` at
//! encoder-begin, and that action clears the **entire** attachment,
//! ignoring viewport.
//!
//! Ending the current encoder and starting a new one with
//! `loadAction = Clear` handles the common "one Clear at pass
//! start" case, but is wrong for atlas-tile patterns such as a
//! shared shadow-cascade tile atlas, where each per-tile sub-rect
//! `Clear` would wipe the whole texture and delete every prior
//! tile's caster draws.
//!
//! Instead this path emits a fullscreen-triangle draw inside the
//! current encoder, scissored to the D3D9 viewport, whose VS writes
//! the caller-supplied clear value as constant depth (and the FS
//! writes `clear_color` to the bound color attachment when one
//! exists). The per-format pipeline lives here and is created lazily
//! on first use.
//!
//! Cache shape: `(depth_format, color_format, flags)` →
//! `MTLRenderPipelineState*`, where `flags` carries `HAS_COLOR` /
//! `HAS_DEPTH` / `HAS_STENCIL`. Shadow-cascade tile
//! casters land at one combo (`Depth32Float`, no color); games with a
//! richer mid-pass `Clear` pattern grow the cache by a handful of
//! entries. Pipelines + library are process-lifetime (leaked via
//! `Retained::into_raw`) — same posture as `hdr_present`.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use mtld3d_shared::{
    EnsureClearQuadPipelineParams, MetalHandle,
    mtl::{ClearQuadFlags, PixelFormat},
    mtl_handle::{MTLFunctionKind, MTLRenderPipelineStateKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLPixelFormat,
    MTLRenderPipelineDescriptor,
};

use super::texture::mtl_pixel_format;
use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// MSL source for the clear-quad library.
///
/// Vertex stage `mtld3d_clear_quad_vs` synthesises a fullscreen triangle
/// from `vertex_id` (0..3) and writes the caller-supplied clear value
/// from `setVertexBytes` at buffer slot 0 as the position's `z`. The
/// triangle's vertices span `[-1, 3]^2` in clip-space so its visible
/// region covers exactly `[-1, 1]^2` — the scissor (set by the PE side
/// to the D3D9 viewport rect) constrains writes to the requested
/// sub-rect of the attachment.
///
/// Fragment stage `mtld3d_clear_quad_ps_color` writes a caller-supplied
/// clear color from `setFragmentBytes` at buffer slot 0 to the bound
/// color attachment. The depth-write side of the clear is gated by the
/// PE-side depth-stencil state (`depth_test = always, depth_write =
/// on`) which writes the VS's `z` to the depth attachment uniformly
/// across the scissored fragments.
///
/// `mtld3d_clear_quad_ps_void` is the depth-only variant — Metal allows
/// pipelines with no fragment function when the color attachment is
/// absent. Shadow-cascade tile casters use this variant; the
/// with-color variant exists for future games that mid-pass Clear
/// color too.
const CLEAR_QUAD_MSL: &str = r"
#include <metal_stdlib>
using namespace metal;

struct ClearVsOut {
    float4 position [[position]];
};

vertex ClearVsOut mtld3d_clear_quad_vs(
    uint vid [[vertex_id]],
    constant float &z [[buffer(0)]]
) {
    // Fullscreen triangle: vid=0 (-1,-1), vid=1 (3,-1), vid=2 (-1,3).
    float2 p = float2(float((vid << 1u) & 2u) * 2.0 - 1.0,
                      float(vid & 2u) * 2.0 - 1.0);
    ClearVsOut out;
    out.position = float4(p, z, 1.0);
    return out;
}

fragment float4 mtld3d_clear_quad_ps_color(
    constant float4 &rgba [[buffer(0)]]
) {
    return rgba;
}
";

/// VS entry-point name.
///
/// Must match the `vertex` function declaration in `CLEAR_QUAD_MSL`.
const VS_NAME: &str = "mtld3d_clear_quad_vs";
/// PS entry-point name for the color-writing variant.
const PS_COLOR_NAME: &str = "mtld3d_clear_quad_ps_color";

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct ClearQuadKey {
    depth_format: PixelFormat,
    color_format: PixelFormat,
    flags: ClearQuadFlags,
}

struct ClearQuadCache {
    vs_fn: MetalHandle<MTLFunctionKind>,
    ps_color_fn: MetalHandle<MTLFunctionKind>,
    pipelines: Mutex<HashMap<ClearQuadKey, MetalHandle<MTLRenderPipelineStateKind>>>,
}

static CACHE: OnceLock<Option<ClearQuadCache>> = OnceLock::new();

/// Lazy create-or-fetch of the clear-quad pipeline for the requested format combo.
///
/// Returns `None` on any compile / pipeline-create failure; the PE side
/// then falls back to the legacy `end_current_pass` path (sub-rect Clear
/// semantics regress, but the game still renders).
pub fn ensure_clear_quad_pipeline(
    params: &EnsureClearQuadPipelineParams,
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let device = params.device_handle.into_retained()?;

    let cache = CACHE
        .get_or_init(|| build_library_and_functions(&device))
        .as_ref()?;

    let key = ClearQuadKey {
        depth_format: params.depth_format,
        color_format: params.color_format,
        flags: params.flags,
    };

    {
        let pipelines = cache.pipelines.lock().ok()?;
        if let Some(&handle) = pipelines.get(&key) {
            return Some(handle);
        }
    }

    let handle = build_pipeline(&device, cache, &key)?;
    let mut pipelines = cache.pipelines.lock().ok()?;
    Some(*pipelines.entry(key).or_insert(handle))
}

fn build_library_and_functions(device: &ProtocolObject<dyn MTLDevice>) -> Option<ClearQuadCache> {
    let source = NSString::from_str(CLEAR_QUAD_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Pin `mathMode = Fast` (default for MSL ≤ 3.1, `Relaxed` for ≥ 3.2 —
    // future-proof against an MSL bump). The clear-quad shader does a
    // constant-fed `out.color = u.color` write; the math-mode pin matters
    // only as documentation parity with the other compile sites here.
    options.setMathMode(MTLMathMode::Fast);
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "clear-quad: MSL compilation failed: {e}"
            );
            return None;
        }
    };
    library.setLabel(Some(&NSString::from_str("mtld3d-clear-quad")));

    let vs = library.newFunctionWithName(&NSString::from_str(VS_NAME))?;
    let ps_color = library.newFunctionWithName(&NSString::from_str(PS_COLOR_NAME))?;

    // Library is kept alive by the function refs; the function refs
    // are kept alive by the pipeline states once they're built.
    // Intentionally leak the function handles for process lifetime
    // so build_pipeline can re-derive `&ProtocolObject<dyn
    // MTLFunction>` from them per pipeline-build.
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let vs_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(vs) as u64) };
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    let ps_color_handle =
        unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(ps_color) as u64) };
    drop(library);

    Some(ClearQuadCache {
        vs_fn: vs_handle,
        ps_color_fn: ps_color_handle,
        pipelines: Mutex::new(HashMap::new()),
    })
}

fn build_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &ClearQuadCache,
    key: &ClearQuadKey,
) -> Option<MetalHandle<MTLRenderPipelineStateKind>> {
    let vs = cache.vs_fn.into_retained()?;
    let has_color = key.flags.contains(ClearQuadFlags::HAS_COLOR);
    let has_depth = key.flags.contains(ClearQuadFlags::HAS_DEPTH);
    let has_stencil = key.flags.contains(ClearQuadFlags::HAS_STENCIL);

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs));

    if has_color {
        let ps_color = cache.ps_color_fn.into_retained()?;
        desc.setFragmentFunction(Some(&ps_color));
        // SAFETY: `colorAttachments()` returns a non-null `MTLRenderPipeline-
        // ColorAttachmentDescriptorArray`; subscript 0 is always valid.
        let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
        color0.setPixelFormat(mtl_pixel_format(key.color_format));
        drop(ps_color);
    } else {
        // Depth-only pipeline: no fragment function. Metal accepts a
        // `MTLRenderPipelineDescriptor` with `fragmentFunction = nil`
        // when there are no color outputs, and the VS's z-write
        // alone drives the depth attachment under the
        // `depth_write = on` depth-stencil state.
        desc.setFragmentFunction(None);
        if key.flags.contains(ClearQuadFlags::COLOR_FORMAT_NO_WRITE) {
            // The pass still has a colour attachment bound, so the pipeline
            // must declare its format (Metal validates pipeline-vs-pass colour
            // format even for a draw that writes no colour) — but with a zero
            // write mask, since this is a depth-only clear. No fragment
            // function is needed: nothing is written.
            // SAFETY: `colorAttachments()` returns a non-null descriptor array;
            // subscript 0 is always valid.
            let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
            color0.setPixelFormat(mtl_pixel_format(key.color_format));
            color0.setWriteMask(objc2_metal::MTLColorWriteMask::empty());
        }
    }

    // Depth attachment formats — mirror `pipeline::create_render_pipeline`'s
    // rule: only set the stencil pixel format when the depth format actually
    // carries stencil bits; mismatched stencil format makes Metal reject the
    // pipeline at draw time. When the pass has NO depth attachment (`HAS_DEPTH`
    // unset — e.g. a color clear-quad after `SetDepthStencilSurface(NULL)`),
    // leave Metal's default `.invalid`: declaring a depth format against a
    // depth-less render pass is rejected ("depth attachment pixelFormat must be
    // Invalid, as no texture is set").
    if has_stencil {
        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
        desc.setStencilAttachmentPixelFormat(MTLPixelFormat::Depth32Float_Stencil8);
    } else if has_depth {
        desc.setDepthAttachmentPixelFormat(mtl_pixel_format(key.depth_format));
    }

    let label = format!(
        "mtld3d-clear-quad d={} c={:?}{}{}",
        if has_depth {
            format!("{:?}", key.depth_format)
        } else {
            "none".to_owned()
        },
        if has_color {
            key.color_format
        } else {
            PixelFormat::Bgra8Unorm
        },
        if has_color { "" } else { " no-color" },
        if has_stencil { " +stencil" } else { "" },
    );
    desc.setLabel(Some(&NSString::from_str(&label)));

    let pipeline = match device.newRenderPipelineStateWithDescriptor_error(&desc) {
        Ok(p) => p,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "clear-quad: pipeline creation failed ({label}): {e}"
            );
            return None;
        }
    };
    drop(vs);
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    Some(unsafe {
        MetalHandle::<MTLRenderPipelineStateKind>::new(Retained::into_raw(pipeline) as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the inline MSL compiles on the host `MTLDevice`.
    ///
    /// Catches shader typos at unit-test time so a regression never ships
    /// to the game first.
    #[test]
    fn clear_quad_msl_compiles_under_metal() {
        use objc2_metal::MTLCreateSystemDefaultDevice;
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
            return;
        };
        let source = NSString::from_str(CLEAR_QUAD_MSL);
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_4);
        options.setMathMode(MTLMathMode::Fast);
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .expect("clear-quad MSL must compile");
        let _vs = library
            .newFunctionWithName(&NSString::from_str(VS_NAME))
            .expect("VS entry must exist");
        let _ps = library
            .newFunctionWithName(&NSString::from_str(PS_COLOR_NAME))
            .expect("PS color entry must exist");
    }
}
