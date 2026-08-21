use super::*;

const _: () = {
    assert!(core::mem::size_of::<MetalHandle<MTLDeviceKind>>() == core::mem::size_of::<u64>());
    assert!(core::mem::size_of::<MetalHandle<MTLTextureKind>>() == core::mem::size_of::<u64>());
    assert!(
        core::mem::align_of::<MetalHandle<MTLDeviceKind>>() == core::mem::align_of::<u64>()
    );
};

#[test]
fn null_is_null() {
    assert!(MetalHandle::<MTLDeviceKind>::NULL.is_null());
    assert_eq!(MetalHandle::<MTLDeviceKind>::NULL.raw(), 0);
}

#[test]
fn non_null_round_trip() {
    // SAFETY: in tests we never dereference; the value is opaque.
    let h = unsafe { MetalHandle::<MTLDeviceKind>::new(0x1234_5678_9abc_def0) };
    assert!(!h.is_null());
    assert_eq!(h.raw(), 0x1234_5678_9abc_def0);
}

#[test]
fn default_is_null() {
    let h: MetalHandle<MTLDeviceKind> = MetalHandle::default();
    assert!(h.is_null());
}

#[test]
fn lower_hex_matches_raw() {
    // SAFETY: opaque value, not dereferenced.
    let h = unsafe { MetalHandle::<MTLTextureKind>::new(0xdead_beef) };
    assert_eq!(format!("{h:#x}"), "0xdeadbeef");
}
