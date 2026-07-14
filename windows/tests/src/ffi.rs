//! Raw `d3d9.dll` imports used by the harness.
//!
//! The library is resolved at load time via `raw-dylib`; on i386 the exports
//! are undecorated `stdcall`, on x64 they are plain. Wine resolves the DLL
//! from the prefix `make install` populates.

use core::ffi::c_void;

#[cfg_attr(
    target_arch = "x86",
    link(name = "d3d9", kind = "raw-dylib", import_name_type = "undecorated")
)]
#[cfg_attr(target_arch = "x86_64", link(name = "d3d9", kind = "raw-dylib"))]
unsafe extern "system" {
    pub fn Direct3DCreate9(sdk_version: u32) -> *mut c_void;
}
