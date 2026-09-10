use mtld3d_shared::{MetalHandle, mtl::StageTag, perf::ShaderTimings};
use objc2::rc::Retained;
use objc2_metal::MTLCreateSystemDefaultDevice;

use super::{compile_shader_library, destroy_function, destroy_library};

fn poisoned_timings() -> ShaderTimings {
    if !mtld3d_shared::perf::perf_enabled() {
        return ShaderTimings::new();
    }
    ShaderTimings {
        preparation_ns: u64::MAX,
        library_ns: u64::MAX,
        function_ns: u64::MAX,
    }
}

#[test]
fn shader_timings_reset_unreached_phases_and_preserve_results() {
    mtld3d_shared::init_logger();
    mtld3d_shared::perf::init_tracking_enabled();
    let mut timings = poisoned_timings();
    assert!(
        compile_shader_library(MetalHandle::NULL, "", StageTag::Vertex, "", &mut timings).is_none()
    );
    assert_eq!(timings.library_ns, 0);
    assert_eq!(timings.function_ns, 0);
    assert_ne!(timings.preparation_ns, u64::MAX);

    let device = MTLCreateSystemDefaultDevice().expect("Metal device for shader timing test");
    // SAFETY: the device's retain stays alive until all three calls return.
    let handle = unsafe { MetalHandle::new(Retained::as_ptr(&device) as u64) };
    let msl = "#include <metal_stdlib>\nusing namespace metal;\nvertex float4 probe(uint id [[vertex_id]]) { return float4(float(id), 0, 0, 1); }";
    for (source, entry, success, function_reached) in [
        (msl, "probe", true, true),
        (msl, "missing", false, true),
        ("invalid MSL", "probe", false, false),
    ] {
        timings = poisoned_timings();
        let result = compile_shader_library(handle, source, StageTag::Vertex, entry, &mut timings);
        assert_eq!(result.is_some(), success);
        assert_ne!(timings.preparation_ns, u64::MAX);
        assert_ne!(timings.library_ns, u64::MAX);
        assert_ne!(timings.function_ns, u64::MAX);
        if !function_reached {
            assert_eq!(timings.function_ns, 0);
        }
        if mtld3d_shared::perf::perf_enabled() {
            assert!(timings.preparation_ns > 0);
            assert!(timings.library_ns > 0);
            assert_eq!(timings.function_ns > 0, function_reached);
        } else {
            assert_eq!(
                (
                    timings.preparation_ns,
                    timings.library_ns,
                    timings.function_ns
                ),
                (0, 0, 0)
            );
        }
        if let Some((library, function)) = result {
            destroy_function(function.raw());
            destroy_library(library.raw());
        }
    }
}
