//! Win32 types used by the `IDirect3DDevice9::*Cursor*` implementation.
//!
//! These are repr(C) binary-layout mirrors of `windows.h` definitions used
//! across the FFI boundary to `user32.dll` / `gdi32.dll`. Only the fields
//! actually touched by the cursor-building code are declared.

use core::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct POINT {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
pub struct ICONINFO {
    /// `FALSE` = cursor, `TRUE` = icon.
    pub f_icon: i32,
    pub x_hotspot: u32,
    pub y_hotspot: u32,
    /// AND mask bitmap (monochrome).
    pub hbm_mask: *mut c_void,
    /// Color bitmap.
    pub hbm_color: *mut c_void,
}
