use objc2_metal::{MTLCreateSystemDefaultDevice, MTLResourceOptions};

use super::*;

/// A null buffer handle borrows nothing, exactly as it retains nothing.
#[test]
fn borrow_retained_filters_the_null_handle() {
    let handle = MetalHandle::<MTLBufferKind>::NULL;
    // SAFETY: the null handle addresses no object, so no retain has to
    // outlive the (absent) reference the call returns.
    assert!(unsafe { handle.borrow_retained() }.is_none());
    assert!(handle.into_retained().is_none());
}

/// The borrow reads through the canonical retain instead of taking one.
#[test]
fn borrow_retained_addresses_the_object_without_a_refcount_bump() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let buffer = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("Metal buffer");
    let canonical =
        Retained::into_raw(ProtocolObject::<dyn MTLBuffer>::from_retained(buffer)) as u64;
    // SAFETY: `canonical` is the address of the retain `Retained::into_raw`
    // just gave up, so the handle stands for a live `id<MTLBuffer>`.
    let handle = unsafe { MetalHandle::<MTLBufferKind>::new(canonical) };

    // SAFETY: the canonical retain above is released only at the end of this
    // test, after the borrow and every read through it.
    let borrowed = unsafe { handle.borrow_retained() }.expect("non-null handle borrows");
    let before = borrowed.retainCount();
    assert_eq!(core::ptr::from_ref(borrowed) as u64, canonical);
    // SAFETY: as the borrow above.
    let second = unsafe { handle.borrow_retained() }.expect("non-null handle borrows");
    assert_eq!(second.retainCount(), before);

    let retained = handle.into_retained().expect("non-null handle retains");
    assert_eq!(retained.retainCount(), before + 1);
    drop(retained);
    assert_eq!(borrowed.retainCount(), before);

    // SAFETY: the handle holds the canonical retain and no copy of it is used
    // after this call; `borrowed` and `second` are dead here.
    unsafe { handle.release_retain() };
}
