//! The host view a native device presents into, read through `AppKit`.
//!
//! The application hands an `NSView*` where Wine would hand an `HWND`. Each
//! query runs on the main thread, through the unix side's synchronous door.

use core::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};

use mtld3d_shared::mtl::ClientSurfaceRequest;
use objc2::{MainThreadMarker, rc::Retained, runtime::AnyObject};
use objc2_app_kit::NSView;

use crate::LOG_TARGET;

/// Whether the device window is an `NSView`, the one kind of window a native device presents into.
///
/// Checked before `CreateDevice`, `Reset` or `CreateAdditionalSwapChain`
/// sizes a back buffer from it: the view calls that follow would abort the
/// process on any other object, an `NSWindow` among them. A null window is
/// the caller's to answer. A refusal is logged.
pub fn is_device_window(window: *mut c_void) -> bool {
    if window.is_null() {
        return true;
    }
    // The address crosses to the main thread as an integer, where the object
    // is retained, checked and released.
    let address = window as usize;
    let mut refused = None;
    mtld3d_unix::run_on_main_thread(|| {
        let _mtm =
            MainThreadMarker::new().expect("the device window is checked on the main thread");
        // SAFETY: a native application passes an Objective-C object as the
        // device window and keeps it alive across the call that names it.
        let Some(object) = (unsafe { Retained::retain(address as *mut AnyObject) }) else {
            return;
        };
        if object.downcast_ref::<NSView>().is_none() {
            refused = Some(object.class().name().to_string_lossy().into_owned());
        }
    });
    let Some(class) = refused else {
        return true;
    };
    log::warn!(
        target: LOG_TARGET,
        "device window {window:p} is a {class}, not an NSView → INVALIDCALL",
    );
    false
}

/// The view's bounds in backing pixels, or `None` for a null view or an empty one.
///
/// The back buffer a windowed device sizes from it is in pixels, as the
/// drawable the unix side presents into is. A `resize` word loses its
/// pending resize in the same main-thread call, so a resize the read saw is
/// not applied again and one after it stays pending.
pub fn client_size(window: *mut c_void, resize: Option<&AtomicU64>) -> Option<(u32, u32)> {
    if window.is_null() {
        return None;
    }
    // The address crosses to the main thread as an integer: the view is only
    // turned back into an object there.
    let address = window as usize;
    let mut size = None;
    mtld3d_unix::run_on_main_thread(|| {
        let _mtm = MainThreadMarker::new().expect("client_size reads the view on the main thread");
        // SAFETY: a native application passes an `NSView*` as the device
        // window, which it keeps alive across the call that names it.
        let Some(view) = (unsafe { Retained::retain(address as *mut NSView) }) else {
            return;
        };
        if let Some(word) = resize {
            word.fetch_and(!ClientSurfaceRequest::RESIZE, Ordering::AcqRel);
        }
        size = Some(mtld3d_unix::view_pixels(&view).unwrap_or((0, 0)));
    });
    let (width, height) = size?;
    if width == 0 || height == 0 {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: address as u64,
            "view {address:#x} has an empty client area ({width}x{height}); the requested \
             back-buffer size stands",
        );
        return None;
    }
    Some((width, height))
}
