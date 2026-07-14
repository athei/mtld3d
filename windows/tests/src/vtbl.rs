//! COM vtable dereference helper, shared by the harness and every resource wrapper.
//!
//! Concentrated here so the `*this -> &Vtbl` reinterpretation has a single
//! audited home.

use core::ffi::c_void;

/// Read the `'static` vtable reference from a COM `this` pointer.
///
/// # Safety
/// `p` must be a live COM object whose first pointer-sized field is a non-null
/// pointer to a static `T`-typed vtable (the D3D9 ABI guarantees this for every
/// interface mtld3d returns).
pub unsafe fn deref_vtbl<T>(p: *mut c_void) -> &'static T {
    // SAFETY: per contract, the first pointer-sized field of `p` is the vtable
    // pointer.
    let vtbl_ptr = unsafe { *p.cast::<*const T>() };
    // SAFETY: per contract, `vtbl_ptr` points to a static `T` vtable that
    // outlives any use.
    unsafe { &*vtbl_ptr }
}
