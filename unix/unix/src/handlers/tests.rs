use super::{core_foundation_image, identity, kCFRunLoopCommonModes};

#[test]
fn core_foundation_identity_names_the_framework() {
    let framework = core_foundation_image().expect("linked CoreFoundation image");
    let path = framework.path().expect("loaded framework path");
    assert_eq!(path.file_name().unwrap(), "CoreFoundation");
    assert!(path.to_str().unwrap().contains("CoreFoundation.framework/"));
    let uuid = framework.uuid().expect("loaded framework UUID");
    assert_eq!(uuid.len(), 36);
    let own_symbol = (identity::image_id as *const ()).cast();
    // SAFETY: the image executing this function remains loaded during the call.
    let own = unsafe { identity::LoadedImage::for_symbol(own_symbol) }.unwrap();
    assert_ne!(framework.base(), own.base());
    assert_ne!(Some(uuid), own.uuid());
    assert_eq!(own.uuid(), identity::image_id().as_deref());
    assert!((&raw const kCFRunLoopCommonModes) as usize >= framework.base());
    println!(
        "CoreFoundation path={} base={:#x} uuid={uuid}",
        path.display(),
        framework.base()
    );
    println!("caller base={:#x} uuid={}", own.base(), own.uuid().unwrap());
}

#[test]
fn rejected_shader_thunk_clears_timing_outputs() {
    let mut params = mtld3d_shared::CompileShaderLibraryParams {
        device_handle: mtld3d_shared::MetalHandle::NULL,
        msl_ptr: 0,
        msl_len: 0,
        stage_tag: mtld3d_shared::mtl::StageTag::Vertex,
        entry_ptr: 0,
        entry_len: 0,
        pad0: 0,
        library_handle: mtld3d_shared::MetalHandle::NULL,
        fn_handle: mtld3d_shared::MetalHandle::NULL,
        timings: mtld3d_shared::perf::ShaderTimings {
            preparation_ns: u64::MAX,
            library_ns: u64::MAX,
            function_ns: u64::MAX,
        },
    };
    let result = super::compile_shader_library_handler((&raw mut params).cast());
    assert_ne!(result, 0);
    assert_eq!(
        (
            params.timings.preparation_ns,
            params.timings.library_ns,
            params.timings.function_ns
        ),
        (0, 0, 0)
    );
}
