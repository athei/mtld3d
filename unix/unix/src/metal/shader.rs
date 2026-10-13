use log::error;
use mtld3d_shared::{
    MetalHandle,
    mtl_handle::{MTLDeviceKind, MTLFunctionKind, MTLLibraryKind},
};
use objc2::rc::Retained;
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode};

use crate::{
    LOG_TARGET,
    metal::handle::{IntoRetained, ReleaseRetain},
};

#[cfg(test)]
mod tests;

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
    entry: &str,
    timings: &mut mtld3d_shared::perf::ShaderTimings,
) -> Option<(MetalHandle<MTLLibraryKind>, MetalHandle<MTLFunctionKind>)> {
    timings.reset();
    let preparation = mtld3d_shared::perf::NanosSetTimer::start(&raw mut timings.preparation_ns);
    let device = device_handle.into_retained()?;

    let source = objc2_foundation::NSString::from_str(msl);
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    // Documented prerequisite for `[[position, invariant]]` to take effect.
    // It keeps the position identical across pipelines that compute it the
    // same way, such as a depth prepass and the pass that tests against it.
    options.setPreserveInvariance(true);
    // Fast for both stages. With `preserveInvariance`, Fast only adds the
    // per-op nnan, ninf, nsz, arcp and afn flags in the frontend and leaves
    // reassociation and contraction to the backend under the invariance
    // contract (the `invariance-late-*` function attributes), so
    // `[[position, invariant]]` holds and the vertex stage needs no Safe
    // mode. Pinned explicitly rather than relying on the language-version
    // default (Fast for MSL 3.1 and older, Relaxed from 3.2).
    options.setMathMode(MTLMathMode::Fast);
    drop(preparation);
    let compile = mtld3d_shared::perf::NanosSetTimer::start(&raw mut timings.library_ns);
    let compiled = device.newLibraryWithSource_options_error(&source, Some(&options));
    drop(compile);
    let library = match compiled {
        Ok(lib) => lib,
        Err(e) => {
            error!(target: LOG_TARGET, "shader compilation failed: {e}");
            return None;
        }
    };

    let function = mtld3d_shared::perf::NanosSetTimer::start(&raw mut timings.function_ns);
    let name = objc2_foundation::NSString::from_str(entry);
    // `setLabel` makes the per-shader entry name surface in Xcode views
    // that show the library/object identity rather than the function-source
    // text — at minimum the GPU-trace timeline's pipeline-state list and
    // the resource browser. Same string as the function name keeps
    // captures self-consistent across all Xcode tabs.
    library.setLabel(Some(&name));
    let resolved = library.newFunctionWithName(&name);
    drop(function);
    let func = resolved?;

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
