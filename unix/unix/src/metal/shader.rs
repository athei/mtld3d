use log::error;
use mtld3d_shared::{
    MetalHandle,
    mtl::StageTag,
    mtl_handle::{MTLDeviceKind, MTLFunctionKind, MTLLibraryKind},
};
use objc2::rc::Retained;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode};

use crate::{
    LOG_TARGET,
    metal::handle::{IntoRetained, ReleaseRetain},
};

/// Compile MSL source text into a library and resolve a single entry point.
///
/// `entry` is the function name to look up via `newFunctionWithName:` and
/// must match the `vertex Varyings <entry>(...)` / `fragment ... <entry>(...)`
/// declaration in the MSL. A render pipeline can mix functions from
/// independently compiled VS and PS libraries, so each stage is its own
/// library.
///
/// MSL pinned to 2.4 so the same compiled libraries work on Intel/AMD
/// Macs. Metal 3.x is Apple-Silicon-only; without the pin, Metal picks
/// a default that depends on the host (≥ 3.0 on macOS 14+) and Intel
/// drivers reject the source. Our DXSO emitter doesn't use any 3.x
/// features, so the pin is loss-free.
pub fn compile_shader_library(
    device_handle: MetalHandle<MTLDeviceKind>,
    msl: &str,
    stage_tag: StageTag,
    entry: &str,
) -> Option<(MetalHandle<MTLLibraryKind>, MetalHandle<MTLFunctionKind>)> {
    let device = device_handle.into_retained()?;

    let source = objc2_foundation::NSString::from_str(msl);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Documented prerequisite for `[[position, invariant]]` to take effect:
    // without it, AMD-Mac Metal compilers reorder FMA chains independently
    // per shader and ground-projected decals z-fight terrain even when the
    // emitter produces byte-identical FMA chains across VS pairs. Apple
    // Silicon honours invariance without the flag; AMD-Mac Metal compilers
    // require it, so `[[position, invariant]]` is set explicitly here.
    options.setPreserveInvariance(true);
    // Fast Math allows the compiler to reassociate FP ops, which on
    // AMD-Mac defeats `preserveInvariance` even when each shader's
    // emitted MSL is byte-identical: AMD's compiler reassociates within
    // the explicit `fma()` chain and the cross-shader byte-equivalence
    // collapses. Apple Silicon's compiler doesn't reassociate under
    // `preserveInvariance` even with Fast Math on, so Safe is the
    // safe default on the VS path that depends on cross-pipeline
    // position invariance. PS keeps Fast — needs the transcendental
    // throughput and has no invariance contract. Pinned explicitly
    // rather than relying on the language-version default (Fast for
    // MSL ≤ 3.1, Relaxed for ≥ 3.2 — future-proof).
    match stage_tag {
        StageTag::Vertex => options.setMathMode(MTLMathMode::Safe),
        StageTag::Fragment => options.setMathMode(MTLMathMode::Fast),
    }
    let library = match device.newLibraryWithSource_options_error(&source, Some(&options)) {
        Ok(lib) => lib,
        Err(e) => {
            error!(target: LOG_TARGET, "shader compilation failed: {e}");
            return None;
        }
    };

    let name = objc2_foundation::NSString::from_str(entry);
    // `setLabel` makes the per-shader entry name surface in Xcode views
    // that show the library/object identity rather than the function-source
    // text — at minimum the GPU-trace timeline's pipeline-state list and
    // the resource browser. Same string as the function name keeps
    // captures self-consistent across all Xcode tabs.
    library.setLabel(Some(&name));
    let func = library.newFunctionWithName(&name)?;

    // SAFETY: `Retained::into_raw` transfers the canonical retain into the
    // typed library handle.
    let lib_handle =
        unsafe { MetalHandle::<MTLLibraryKind>::new(Retained::into_raw(library) as u64) };
    // SAFETY: `Retained::into_raw` transfers the canonical retain into the
    // typed function handle.
    let fn_handle = unsafe { MetalHandle::<MTLFunctionKind>::new(Retained::into_raw(func) as u64) };
    Some((lib_handle, fn_handle))
}

/// Release a Metal library handle.
pub fn destroy_library(library_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has dropped its only copy of `library_handle`.
    let handle = unsafe { MetalHandle::<MTLLibraryKind>::new(library_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}

/// Release a Metal function handle (resolved from a library entry point).
pub fn destroy_function(fn_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has dropped its only copy of `fn_handle`.
    let handle = unsafe { MetalHandle::<MTLFunctionKind>::new(fn_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}
