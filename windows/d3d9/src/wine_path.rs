//! The unix path of a DOS path, through Wine's `wine_get_unix_file_name`.
//!
//! A path the PE side resolves (the log directory, the presenter gate) is
//! handed to the unix side as the unix path it can open, since the DOS path
//! means nothing there. Wine's kernel32 carries the mapping as an extension
//! the SDK import library does not export, so it is resolved by name at run
//! time.

use core::ffi::{c_char, c_void};
use std::{os::windows::ffi::OsStrExt, path::Path};

unsafe extern "system" {
    fn GetProcessHeap() -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, mem: *mut c_void) -> i32;
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, proc_name: *const u8) -> *mut c_void;
}

/// Wine's kernel32 extension: the unix path of a DOS path, heap-allocated.
///
/// Resolved by name at run time, since the SDK import library we link
/// against does not carry it. Wine declares it `CDECL`, not `WINAPI`: on
/// i686 the caller pops the argument, and a stdcall type here corrupts the
/// caller's stack.
type WineGetUnixFileName = unsafe extern "C" fn(*const u16) -> *mut c_char;

/// The unix path of `dos`, or `None` when Wine cannot map it.
pub fn unix_path(dos: &Path) -> Option<String> {
    // SAFETY: kernel32 is loaded for the life of the process; the name is
    // NUL-terminated.
    let kernel32 = unsafe { GetModuleHandleA(c"kernel32.dll".as_ptr().cast::<u8>()) };
    if kernel32.is_null() {
        return None;
    }
    // SAFETY: a live module handle and a NUL-terminated export name.
    let proc =
        unsafe { GetProcAddress(kernel32, c"wine_get_unix_file_name".as_ptr().cast::<u8>()) };
    if proc.is_null() {
        return None;
    }
    // SAFETY: the export has this signature in every Wine that carries it.
    let get_unix_file_name: WineGetUnixFileName = unsafe { core::mem::transmute(proc) };
    let wide: Vec<u16> = dos
        .as_os_str()
        .encode_wide()
        .chain(core::iter::once(0))
        .collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 string live for the call;
    // the result is a NUL-terminated string on the process heap or null.
    let raw = unsafe { get_unix_file_name(wide.as_ptr()) };
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` is a valid NUL-terminated C string until freed below.
    let path = unsafe { core::ffi::CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: the process heap handle is a stable kernel32 pseudo-handle.
    let heap = unsafe { GetProcessHeap() };
    // SAFETY: `raw` came from the process heap per the kernel32 contract.
    let _ = unsafe { HeapFree(heap, 0, raw.cast::<c_void>()) };
    Some(path)
}
