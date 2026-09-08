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
