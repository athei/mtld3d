//! `HRESULT` assertion helpers shared by the harness and resource wrappers.
//!
//! Centralising the panic sites here keeps every getter free of inline
//! `assert!`, so the panic contract is documented in one place.

use core::ffi::c_void;

/// Assert a D3D9 `HRESULT` is `D3D_OK`.
///
/// # Panics
/// Panics with `what` and the hex `HRESULT` when `hr != 0`.
#[track_caller]
pub fn expect_ok(hr: i32, what: &str) {
    assert!(hr == 0, "{what} failed: 0x{hr:08X}");
}

/// Assert a creation call returned `D3D_OK` and a non-null object.
///
/// # Panics
/// Panics when `hr != 0` or `ptr` is null.
#[track_caller]
pub fn expect_created(hr: i32, ptr: *mut c_void, what: &str) {
    assert!(hr == 0, "{what} failed: 0x{hr:08X}");
    assert!(!ptr.is_null(), "{what} returned null");
}
