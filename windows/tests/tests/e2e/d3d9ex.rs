//! `Direct3DCreate9Ex` resolves by name and reports `D3D9Ex` as unavailable.
//!
//! A runtime probe that only resolves the name, the way a title's
//! compatibility check tells a Vista-era d3d9 from an older one, must find
//! the export; a caller that goes on to call it must get the documented
//! failure of a runtime without `D3D9Ex`, `D3DERR_NOTAVAILABLE` with the out
//! slot nulled, so it takes its plain-D3D9 fallback instead of reading a
//! stale pointer. A null out slot is `D3DERR_INVALIDCALL`.
//!
//! No shared harness: the point is the export table, and the harness's
//! `raw-dylib` link would make a missing export a link error rather than a
//! test failure.

use core::ffi::{c_char, c_void};

use mtld3d_types::{D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DSDK_VERSION};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

type CreateExFn = unsafe extern "system" fn(u32, *mut *mut c_void) -> i32;

#[test]
fn direct3d_create9_ex_resolves_and_reports_not_available() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");

    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let addr = unsafe { GetProcAddress(lib, c"Direct3DCreate9Ex".as_ptr()) };
    assert!(!addr.is_null(), "GetProcAddress(Direct3DCreate9Ex)");
    // SAFETY: the export has the documented `Direct3DCreate9Ex` signature.
    let create_ex: CreateExFn = unsafe { core::mem::transmute(addr) };

    // A poisoned slot: a caller that ignores the HRESULT must read null, not
    // whatever it held before the call.
    let mut out: *mut c_void = core::ptr::dangling_mut::<c_void>();
    // SAFETY: the resolved export with the SDK version and a live out slot.
    let hr = unsafe { create_ex(D3DSDK_VERSION, &raw mut out) };
    assert_eq!(hr, D3DERR_NOTAVAILABLE, "D3D9Ex is reported unavailable");
    assert!(out.is_null(), "the out slot is nulled on failure");

    // SAFETY: the resolved export; a null out slot is a documented invalid call.
    let hr = unsafe { create_ex(D3DSDK_VERSION, core::ptr::null_mut()) };
    assert_eq!(hr, D3DERR_INVALIDCALL, "a null out slot is an invalid call");

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}
