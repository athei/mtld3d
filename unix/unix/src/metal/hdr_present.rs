//! Present-time render pass for HDR output.
//!
//! When the `CAMetalLayer` is configured for EDR (`RGBA16Float` +
//! `kCGColorSpaceExtendedLinear*` + `wantsExtendedDynamicRange = true`)
//! the present step can no longer be a straight blit from the game's
//! `BGRA8` backbuffer to the drawable: the linear colorspace expects
//! linear float values, not the gamma-encoded bytes the game wrote.
//! This module supplies the render pass that bridges them — a single
//! full-screen triangle whose fragment shader samples the backbuffer,
//! applies the sRGB → linear EOTF, and performs SDR→HDR inverse tone
//! mapping in **`ICtCp`** (BT.2100 perceptual color space) before writing.
//!
//! The `ICtCp` variant of BT.2446 Method A operates on the I (intensity)
//! channel of `ICtCp` while leaving Ct/Cp (chroma) untouched, so
//! saturated content (spells, fire, sunset) keeps its chroma when
//! lifted into HDR brightness instead of desaturating toward white.
//!
//! Output values are in linear BT.709/sRGB primaries — the same
//! primaries the source backbuffer uses. The display-class-matched
//! layer colorspace (`ExtendedLinearSRGB` / `ExtendedLinearDisplayP3` /
//! `ExtendedLinearITUR_2020`, picked at attach time in
//! `macdrv::configure_metal_layer_inner`) tells macOS what primaries
//! those values are in; macOS gamut-converts to the panel as needed.
//!
//! The library + pipeline-state are created once per process on first
//! HDR-active submit (via `OnceLock`) and reused thereafter. Resources
//! intentionally leak on shutdown — they're process-lifetime objects
//! alongside the device and command queue.

use core::ptr::NonNull;
use std::sync::OnceLock;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLDevice, MTLFunction, MTLLanguageVersion, MTLLibrary, MTLMathMode,
    MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
};

use crate::LOG_TARGET;

/// MSL source for the present-pass library.
///
/// One library, one shared vertex stage, **two fragment entry points**:
///
/// - `mtld3d_hdr_present_ps_passthrough` — sample the source backbuffer,
///   apply the sRGB → linear EOTF, return. Selected on the CPU side
///   when the panel reports no EDR headroom this frame (`peak <= 1.0`),
///   either because macOS hasn't promoted the screen yet or because
///   brightness/thermal state physically rules it out. Output is in
///   linear sRGB/BT.709 primaries; the display-class-matched
///   `kCGColorSpaceExtendedLinear*` layer tag lets macOS gamut-convert
///   to the panel. BT.2446-A is **not** identity at `L_hdr = L_sdr`
///   (the inverse mapping under-corrects), so we pick this pipeline
///   rather than running BT.2446 with a no-op intent.
///
/// - `mtld3d_hdr_present_ps_bt2446` — sample, sRGB → linear EOTF, then
///   **ITU-R BT.2446 Method A** SDR→HDR inverse tone mapping operated
///   in `ICtCp` (BT.2100 perceptual color space). The inverse curve
///   runs on the **I (intensity)** channel only; Ct, Cp (chroma) stay
///   put. That keeps saturated content (spell effects, fire, sunset)
///   from desaturating toward white as it's lifted into HDR
///   brightness — the chroma-preserving property of operating in
///   `ICtCp` rather than on luminance alone. Output is in linear BT.709
///   primaries; `1.0` = SDR-paper-white = 100 nits, values exceed 1.0
///   for HDR.
///
/// Vertex stage: synthesise a single oversized triangle covering the
/// full viewport from `vertex_id` alone — no vertex buffer required.
/// The standard fullscreen-triangle trick saves the edge-overlap
/// rasterisation cost of a two-triangle quad. Shared between the two
/// fragment entry points.
///
/// `L_SDR` is pinned to `100.0` because Apple's compositor anchors
/// `1.0`-in-the-drawable to 100 nits and reports
/// `maximumPotentialExtendedDynamicRangeColorComponentValue` as a
/// multiplier of that same 100-nit reference. Any other `L_SDR` would
/// put the BT.2446-A normalization out of phase with the compositor.
///
/// `P_SDR` is the constant `pSDR` from ITU-R BT.2446 §6.1.1 at
/// `L_SDR=100`; precomputed so the compiler folds it.
///
/// Both fragment stages use the **accurate piecewise sRGB EOTF** (not
/// the `pow(x, 2.2)` shortcut — at EDR brightness the 2 % midtone
/// error of the shortcut is visible).
///
/// Ported from the `ICtCp` branch of `Bt2446A` in Lilium's `ReShade` HDR
/// shaders (`Shaders/lilium__include/inverse_tone_mappers.fxh`). The
/// BT.709→LMS / LMS→BT.709 matrices and the LMS-PQ↔ICtCp matrices come
/// from BT.2100 (transitively published in Lilium's `colour_space.fxh`).
/// Stripped of the `InputNitsFactor`, `GammaIn/Out`, and `BT2020 + PQ
/// encode` output steps that don't apply when we feed linearised sRGB
/// and output linear sRGB (the layer's `ExtendedLinear*` tag handles
/// gamut + OETF for the actual display).
///
/// MSL language version pinned to 2.4 for parity with the rest of mtld3d
/// (`shader.rs`) — keeps the same library working on Intel/AMD Macs.
const PRESENT_MSL: &str = include_str!("hdr_present.msl");

/// Cached present-pass resources keyed on the device.
///
/// mtld3d has one `MTLDevice` per process so a global `OnceLock` is the
/// right grain; the fields are raw `u64` handles so the type is trivially
/// `Send + Sync`. Handles leak at process exit — these are process-lifetime
/// objects, the same as the device and command queue.
///
/// Two pipeline states share one MSL library: the CPU side picks
/// `pipeline_passthrough` when `peak <= 1.0` (panel reports no EDR
/// headroom this frame) and `pipeline_bt2446` otherwise. Both compile
/// once at first HDR-active submit.
#[derive(Clone, Copy)]
pub struct HdrPresentResources {
    pub pipeline_passthrough: u64, // MTLRenderPipelineState*
    pub pipeline_bt2446: u64,      // MTLRenderPipelineState*
}

static HDR_RESOURCES: OnceLock<HdrPresentResources> = OnceLock::new();

/// Lazily compile + cache the present-pass library + pipeline.
///
/// Called from `submit_frame` on the encoder thread when HDR is active.
/// The first call compiles MSL (~1–2 ms); subsequent calls are pointer
/// loads.
///
/// Returns `None` (with a once-warn at the failure site) if MSL
/// compilation or pipeline creation fails — `submit_frame` falls back
/// to the SDR blit-present path so the game still renders, just
/// without the EDR boost.
pub fn ensure_resources(device: &ProtocolObject<dyn MTLDevice>) -> Option<HdrPresentResources> {
    if let Some(r) = HDR_RESOURCES.get() {
        return Some(*r);
    }
    let resources = create(device)?;
    Some(*HDR_RESOURCES.get_or_init(|| resources))
}

fn create(device: &ProtocolObject<dyn MTLDevice>) -> Option<HdrPresentResources> {
    let source = NSString::from_str(PRESENT_MSL);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // `mathMode` defaults to `Fast` for MSL ≤ 3.1 (Apple's back-compat with
    // the deprecated `fastMathEnabled = true` default) and `Relaxed` for
    // MSL ≥ 3.2. Pin explicitly so a future MSL bump doesn't silently
    // halve transcendental throughput: the present pass is sRGB EOTF and
    // PQ/ICtCp math, none of which needs IEEE-precise edge handling
    // (existing `max(x, 0)` / `max(x, 1e-20)` clamps already guard the
    // domain). Applies to both the passthrough and BT.2446 fragment
    // entry points; the VS is a positional fullscreen triangle with no
    // invariance concerns.
    options.setMathMode(MTLMathMode::Fast);

    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "hdr present: MSL compilation failed: {e}"
            );
            return None;
        }
    };
    {
        let label = NSString::from_str("mtld3d-hdr-present");
        library.setLabel(Some(&label));
    }

    let vs_name = NSString::from_str("mtld3d_hdr_present_vs");
    let ps_passthrough_name = NSString::from_str("mtld3d_hdr_present_ps_passthrough");
    let ps_bt2446_name = NSString::from_str("mtld3d_hdr_present_ps_bt2446");
    let vs = library.newFunctionWithName(&vs_name)?;
    let ps_passthrough = library.newFunctionWithName(&ps_passthrough_name)?;
    let ps_bt2446 = library.newFunctionWithName(&ps_bt2446_name)?;

    let pipeline_passthrough = build_pipeline(
        device,
        &vs,
        &ps_passthrough,
        "mtld3d-present-pipeline-hdr-passthrough",
    )?;
    let pipeline_bt2446 = build_pipeline(
        device,
        &vs,
        &ps_bt2446,
        "mtld3d-present-pipeline-hdr-bt2446",
    )?;

    // Library and functions are kept alive by the pipeline states
    // (Metal copies what it needs at pipeline-state creation time).
    // The pipeline handles themselves leak for process lifetime via
    // `Retained::into_raw`.
    let _ = library;
    let _ = vs;
    let _ = ps_passthrough;
    let _ = ps_bt2446;

    let pipeline_passthrough_handle = Retained::into_raw(pipeline_passthrough) as u64;
    let pipeline_bt2446_handle = Retained::into_raw(pipeline_bt2446) as u64;
    // Sanity: a raw pointer cast through `Retained::into_raw` can't be
    // null, but proving that to the type system requires the
    // conversion below; the `NonNull` is purely a debug-time guard
    // against a future API change.
    debug_assert!(NonNull::new(pipeline_passthrough_handle as *mut u8).is_some());
    debug_assert!(NonNull::new(pipeline_bt2446_handle as *mut u8).is_some());

    Some(HdrPresentResources {
        pipeline_passthrough: pipeline_passthrough_handle,
        pipeline_bt2446: pipeline_bt2446_handle,
    })
}

fn build_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vs: &ProtocolObject<dyn MTLFunction>,
    ps: &ProtocolObject<dyn MTLFunction>,
    label: &str,
) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(vs));
    desc.setFragmentFunction(Some(ps));
    // No vertex descriptor: the VS synthesises positions from
    // `vertex_id`; Metal requires *some* vertex input slot, but with no
    // attributes declared and no buffer bound, it's a no-op.
    // SAFETY: `colorAttachments()` returns a non-null descriptor array;
    // subscript 0 is always valid.
    let color0 = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setPixelFormat(MTLPixelFormat::RGBA16Float);
    {
        let label = NSString::from_str(label);
        desc.setLabel(Some(&label));
    }

    match device.newRenderPipelineStateWithDescriptor_error(&desc) {
        Ok(p) => Some(p),
        Err(e) => {
            log::error!(
                target: LOG_TARGET,
                "hdr present: pipeline creation failed ({label}): {e}"
            );
            None
        }
    }
}
