//! The client area of a device window.

use core::ffi::c_void;

use super::fullscreen::Rect;
use crate::LOG_TARGET;

#[link(name = "user32")]
unsafe extern "system" {
    fn GetClientRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
}

/// Client-area pixel dimensions of `hwnd`, or `None` when the call fails or the rect is empty.
///
/// The single `GetClientRect` boundary is concentrated here so the call site
/// stays unsafe-free. A null window is the caller's own case (no window to
/// read) and passes silently; the two other ways to `None`, a window user32
/// cannot read and a window with no client area, are warned about once per
/// window, since either leaves the requested size standing where the caller
/// expected the window's.
pub fn client_size(hwnd: *mut c_void) -> Option<(u32, u32)> {
    if hwnd.is_null() {
        return None;
    }
    let window = hwnd as usize as u64;
    let mut rect = Rect::EMPTY;
    // SAFETY: GetClientRect accepts any HWND and writes a RECT through the
    // out pointer; `rect` is an owned local, so non-null + aligned + writable
    // holds. A bad HWND yields a zero return, handled below.
    let ok = unsafe { GetClientRect(hwnd, &raw mut rect) };
    if ok == 0 {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: window,
            "GetClientRect({window:#x}) failed: the window is gone or not this process's; the \
             requested back-buffer size stands",
        );
        return None;
    }
    let w = u32::try_from(rect.width()).unwrap_or(0);
    let h = u32::try_from(rect.height()).unwrap_or(0);
    if w == 0 || h == 0 {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: window,
            "window {window:#x} has an empty client area ({w}x{h}); the requested back-buffer \
             size stands",
        );
        return None;
    }
    Some((w, h))
}
