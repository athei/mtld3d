//! The metal layer's own `nextDrawable`, and the Wine client surface it would have presented.
//!
//! The layer Wine creates for a metal view is a `WineMetalLayer`, winemac's
//! `CAMetalLayer` subclass whose only change is a `nextDrawable` override:
//! before it acquires the drawable it reads `superview` and `window` of the
//! view the layer backs and, when the view sits in a client surface's cocoa
//! view, posts that surface's present to the window's event queue, which
//! makes Wine show that cocoa view and hide the window's other one. The
//! presenter calls `nextDrawable` on its own thread, so every present into a
//! visible window made that override walk `AppKit` off the main thread.
//!
//! So attach turns the layer back into the plain `CAMetalLayer` it extends
//! ([`bypass_present_hook`]), and the one thing the override did for the
//! layer is done here instead, on the thread that asks: the client surface
//! the view sits in is read on the main thread ([`client_surface_of`]),
//! retained, and presented through win32u's own `client_surface_present`
//! when a device attaches to it and when a device leaves a window another
//! live device still presents into. Wine showed the surface of whichever
//! device presented last; this shows the surface of the newest device
//! attached to the window.
//!
//! The surface functions are win32u's exports, called on the API thread,
//! which is a Wine thread: `client_surface_present` takes win32u's surface
//! lock and then the Mac driver's window data, the order every other caller
//! of it takes them in.

use core::ffi::{CStr, c_void};

use log::info;
use objc2::{
    ClassType, Encode, MainThreadMarker,
    rc::Retained,
    runtime::{AnyClass, AnyObject},
};
use objc2_quartz_core::CAMetalLayer;

use super::MACDRV_LIB;
use crate::LOG_TARGET;

/// winemac's `CAMetalLayer` subclass with the `nextDrawable` override.
const HOOK_LAYER_CLASS: &CStr = c"WineMetalLayer";

/// winemac's class for a client surface's cocoa view.
const CLIENT_VIEW_CLASS: &CStr = c"WineContentView";

/// The cocoa view's field that names the client surface it shows.
const CLIENT_SURFACE_IVAR: &CStr = c"d3dmetal_client_surface";

/// What [`bypass_present_hook`] found the layer to be.
#[derive(Debug, PartialEq, Eq)]
pub enum LayerClass {
    /// The layer was the hook class and is a plain `CAMetalLayer` now.
    Bypassed,
    /// The layer already presents through `CAMetalLayer`'s own `nextDrawable`.
    Plain,
    /// The layer is some other subclass, which is left as it is.
    Foreign,
}

/// One of win32u's client surface entry points.
type SurfaceFn = unsafe extern "C" fn(*mut c_void);

/// Turn a `WineMetalLayer` back into the `CAMetalLayer` it extends. **Main thread only.**
///
/// Only a layer whose class is exactly the hook class changes: that class
/// adds no instance variables and overrides nothing but `nextDrawable`, so
/// the object is laid out as its superclass and only the method lookup
/// moves. A layer of any other class, such as one a key-value observer has
/// subclassed, keeps its class and is reported as [`LayerClass::Foreign`].
/// A Wine without the hook class has nothing to bypass.
pub fn bypass_present_hook(layer: *mut c_void, _mtm: MainThreadMarker) -> LayerClass {
    let Some(hook) = AnyClass::get(HOOK_LAYER_CLASS) else {
        return LayerClass::Plain;
    };
    // SAFETY: `layer` is the layer of a metal view the caller holds, which
    // Wine retains for the view's lifetime; the retain keeps it for this call.
    let Some(layer) = (unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }) else {
        return LayerClass::Foreign;
    };
    bypass_hook_class(&layer, hook)
}

/// The client surface the view's superview shows, `0` when it shows none. **Main thread only.**
///
/// Read from the field winemac keeps on a client surface's cocoa view, the
/// same field its `nextDrawable` override reads. `0` when the view has no
/// superview, the superview is no client surface's cocoa view, or this Wine
/// keeps no such field.
pub fn client_surface_of(view: *mut c_void, _mtm: MainThreadMarker) -> usize {
    let Some(host) = AnyClass::get(CLIENT_VIEW_CLASS) else {
        return 0;
    };
    // SAFETY: `view` is a metal view the caller holds retained; the retain
    // taken here keeps it alive for the walk.
    let Some(view) = (unsafe { Retained::retain(view.cast::<objc2_app_kit::NSView>()) }) else {
        return 0;
    };
    // SAFETY: objc2 typed binding; `superview` is unsafe only because the
    // view hierarchy may change under a caller off the main thread, and the
    // marker puts this read on the main thread, where that hierarchy changes.
    let Some(superview) = (unsafe { view.superview() }) else {
        return 0;
    };
    let object: &AnyObject = superview.as_ref();
    surface_field(object, host)
}

/// Take a reference on `surface` for the attachment record, `0` passing through.
///
/// Returns `surface` when the reference was taken, and `0` when there is
/// nothing to hold or this Wine exports no client surface functions.
pub fn retain(surface: usize) -> usize {
    if surface == 0 {
        return 0;
    }
    let Some(add_ref) = surface_fn(c"client_surface_add_ref") else {
        return 0;
    };
    // SAFETY: `surface` was read from a live cocoa view's field on the main
    // thread just before, and the window data list Wine keeps for the window
    // holds a reference on it until the window is destroyed.
    unsafe { add_ref(surface as *mut c_void) };
    surface
}

/// Have Wine show `surface` as its window's client view; `0` does nothing.
///
/// The caller holds a reference on `surface`. A surface whose window is gone
/// is one Wine has detached, and presenting it does nothing.
pub fn present(surface: usize) {
    if surface == 0 {
        return;
    }
    let Some(present) = surface_fn(c"client_surface_present") else {
        return;
    };
    // SAFETY: the caller holds a reference on `surface`, so it is a live
    // client surface; win32u checks it is still attached under its lock.
    unsafe { present(surface as *mut c_void) };
    info!(target: LOG_TARGET, "present: Wine shows client surface {surface:#x}");
}

/// Give back a reference [`retain`] took; `0` does nothing.
pub fn release(surface: usize) {
    if surface == 0 {
        return;
    }
    let Some(release) = surface_fn(c"client_surface_release") else {
        return;
    };
    // SAFETY: the caller hands over the reference `retain` took.
    unsafe { release(surface as *mut c_void) };
}

/// [`bypass_present_hook`] against a hook class the caller names.
fn bypass_hook_class(layer: &CAMetalLayer, hook: &AnyClass) -> LayerClass {
    let object: &AnyObject = layer.as_ref();
    let class = object.class();
    if class == CAMetalLayer::class() {
        return LayerClass::Plain;
    }
    if class != hook {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: the metal layer is a {:?}, not {HOOK_LAYER_CLASS:?}; it keeps its class and \
             its nextDrawable",
            class.name(),
        );
        return LayerClass::Foreign;
    }
    // SAFETY: the object's class is exactly `hook`, a subclass of
    // `CAMetalLayer` that adds no instance variables, so the object is laid
    // out as a `CAMetalLayer` and changing its class changes only which
    // `nextDrawable` it answers with.
    unsafe {
        objc2::ffi::object_setClass(
            core::ptr::from_ref(object).cast_mut(),
            CAMetalLayer::class(),
        )
    };
    LayerClass::Bypassed
}

/// The client surface pointer `object` keeps in its field, when it is a `host`.
fn surface_field(object: &AnyObject, host: &AnyClass) -> usize {
    let mut class = Some(object.class());
    while let Some(current) = class {
        if current == host {
            break;
        }
        class = current.superclass();
    }
    if class.is_none() {
        return 0;
    }
    let Some(ivar) = host.instance_variable(CLIENT_SURFACE_IVAR) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: this Wine's {CLIENT_VIEW_CLASS:?} keeps no {CLIENT_SURFACE_IVAR:?}; a \
             window another device's surface hid stays on that surface",
        );
        return 0;
    };
    let encoding = ivar.type_encoding().to_str().unwrap_or_default();
    if !<*mut c_void>::ENCODING.equivalent_to_str(encoding) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: {CLIENT_VIEW_CLASS:?}.{CLIENT_SURFACE_IVAR:?} is encoded {encoding:?}, not \
             a pointer; it is not read",
        );
        return 0;
    }
    // SAFETY: `object` is a `host`, which declares the field, and its
    // encoding is a pointer's; winemac writes it on the main thread only,
    // where the caller reads it.
    let surface = unsafe { *ivar.load::<*mut c_void>(object) };
    surface as usize
}

/// Resolve one of win32u's client surface functions, `None` with a warning when it is missing.
fn surface_fn(name: &CStr) -> Option<SurfaceFn> {
    // SAFETY: the named export is a win32u function taking one client
    // surface pointer and returning nothing, the type `SurfaceFn` declares.
    let symbol = unsafe { MACDRV_LIB.get::<SurfaceFn>(name.to_bytes_with_nul()) };
    if let Ok(symbol) = symbol {
        return Some(*symbol);
    }
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "present: this Wine exports no {name:?}; a window another device's surface hid stays \
         on that surface",
    );
    None
}

#[cfg(test)]
mod tests;
