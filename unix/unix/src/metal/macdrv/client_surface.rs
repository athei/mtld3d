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
//! retained, and presented through win32u's own `client_surface_present`,
//! which also sizes the surface's cocoa view to the window's client rect.
//! That happens once when a device attaches, and after that only once the
//! surface's own device has submitted a frame: at that device's first
//! submitted frame, when a device leaves a window another live device still
//! presents into, and at the next `Present` that submits a frame after the
//! client view was hidden or the window's client area changed (the device's
//! `PresentClientSurface` request, see `client_view_watch`). Some Wines hide
//! the client view whenever they flush the window surface, and some leave the
//! surface out of the updates a window move or resize makes, counting on the
//! override for both. A winemac that keeps a client view shown across window
//! surface flushes does so once the surface was presented while its view was
//! hidden, or presented again after the present that attached it; a surface
//! whose device has not drawn is therefore not presented again, since its
//! view would then stay over the window surface with nothing in its layer.
//! Wine showed the surface of whichever device presented last; this shows the
//! surface of the newest device attached to the window.
//!
//! The surface functions are win32u's exports, called on the API thread,
//! which is a Wine thread: `client_surface_present` takes win32u's surface
//! lock and then the Mac driver's window data, the order every other caller
//! of it takes them in.

use core::ffi::{CStr, c_void};

use log::{debug, info};
use mtld3d_shared::mtl::ClientSurfaceRequest;
use objc2::{
    ClassType, Encode, MainThreadMarker,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Ivar},
};
use objc2_quartz_core::CAMetalLayer;

use super::MACDRV_LIB;
use crate::LOG_TARGET;

/// winemac's `CAMetalLayer` subclass with the `nextDrawable` override.
const HOOK_LAYER_CLASS: &CStr = c"WineMetalLayer";

/// winemac's class for a client surface's cocoa view.
const CLIENT_VIEW_CLASS: &CStr = c"WineContentView";

/// The names Wines give the cocoa view's field that names the client surface it shows.
///
/// `CrossOver`'s winemac uses the first name, and the `WineAndAqua` fork,
/// which carries the same field for another Metal layer, the second. The
/// first one the class declares is read.
const CLIENT_SURFACE_IVARS: [&CStr; 2] = [c"d3dmetal_client_surface", c"dxmt_client_surface"];

/// What [`bypass_present_hook`] found the layer to be.
#[derive(Debug, PartialEq, Eq)]
pub enum LayerClass {
    /// The layer was the hook class and is a plain `CAMetalLayer` now.
    Bypassed,
    /// The layer already presents through `CAMetalLayer`'s own `nextDrawable`.
    Plain,
    /// The layer is some other subclass, which is left as it is.
    Foreign,
    /// The layer is the hook class and keeps it: nothing here can present its client surface.
    Hooked,
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
///
/// The override is what has Wine show the client surface, and size it to the
/// window's client area, as the layer presents; once it is gone, the attach
/// and the device's later requests do that through win32u. So the layer keeps
/// the override ([`LayerClass::Hooked`]) when they cannot: `can_present` is
/// false when this Wine lacks win32u's surface exports, and a cocoa view class
/// that declares none of [`CLIENT_SURFACE_IVARS`] names no surface to present.
pub fn bypass_present_hook(
    layer: *mut c_void,
    can_present: bool,
    _mtm: MainThreadMarker,
) -> LayerClass {
    let Some(hook) = AnyClass::get(HOOK_LAYER_CLASS) else {
        return LayerClass::Plain;
    };
    if !can_replace_hook(can_present, AnyClass::get(CLIENT_VIEW_CLASS)) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: the metal layer keeps {HOOK_LAYER_CLASS:?}'s nextDrawable, which walks the \
             view off the main thread: this Wine gives no way to show and size the client surface \
             without it",
        );
        return LayerClass::Hooked;
    }
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

/// The client surface the client view `client_view` shows, `0` when none. **Main thread only.**
///
/// `client_view` is the cocoa view of a client surface Wine created, read
/// from a `get_win_data` record the caller still holds.
pub fn client_view_surface(client_view: *mut c_void, _mtm: MainThreadMarker) -> usize {
    let Some(host) = AnyClass::get(CLIENT_VIEW_CLASS) else {
        return 0;
    };
    // SAFETY: `client_view` is the cocoa view of the `get_win_data` record
    // the caller holds, which keeps the view in its window until released;
    // the retain covers the read.
    let Some(client_view) = (unsafe { Retained::retain(client_view.cast::<AnyObject>()) }) else {
        return 0;
    };
    surface_field(&client_view, host)
}

/// win32u's client surface reference and present calls, resolved together.
///
/// Resolved before any lock is taken, so a caller that takes a reference
/// under the attachment registry lock makes one atomic increment there and
/// no symbol lookup.
pub struct SurfaceCalls {
    add_ref: SurfaceFn,
    present: SurfaceFn,
    release: SurfaceFn,
}

impl SurfaceCalls {
    /// The three exports, or `None` with a warning on a Wine that lacks one.
    pub fn load() -> Option<Self> {
        Some(Self {
            add_ref: surface_fn(c"client_surface_add_ref")?,
            present: surface_fn(c"client_surface_present")?,
            release: surface_fn(c"client_surface_release")?,
        })
    }

    /// Take a reference on `surface`, handing it back; `0` passes through.
    pub fn retain(&self, surface: usize) -> usize {
        if surface != 0 {
            // SAFETY: `surface` is one a live attachment record holds a
            // reference on, or one attach read from a client view's field on
            // the main thread just before. Wine's own reference on the latter
            // is held by the window's data and dropped only by the driver's
            // `DestroyWindow`, which takes the window data lock first. A view
            // created or moved in through `get_win_data` is retained while
            // the caller holds that lock, so the surface is alive. A kept
            // view reused in place goes through no `get_win_data`, which
            // would create and show a client surface of its own, so there the
            // precondition is the D3D9 one that the device window stays valid
            // for the `CreateDevice` or `Reset` that attaches to it: a window
            // its owner destroys while another thread attaches a device to it
            // can free that surface between the read and this reference.
            unsafe { (self.add_ref)(surface as *mut c_void) };
        }
        surface
    }

    /// Have Wine show `surface` as its window's client view, for `cause`; `0` does nothing.
    ///
    /// The caller holds a reference on `surface`. A surface whose window is
    /// gone is one Wine has detached, and presenting it does nothing. `cause`
    /// names the event for the log line.
    pub fn present(&self, surface: usize, cause: &str) {
        if surface == 0 {
            return;
        }
        self.call_present(surface);
        info!(target: LOG_TARGET, "present: Wine shows client surface {surface:#x} ({cause})");
    }

    /// [`Self::present`] for a device's request, which can come every frame, logged quietly.
    ///
    /// win32u updates the surface's frame from the window's client rect before
    /// it presents, which is what a resize needs on a Wine that leaves the
    /// surface out of its own updates.
    pub fn present_requested(&self, surface: usize, reasons: ClientSurfaceRequest) {
        if surface == 0 {
            return;
        }
        self.call_present(surface);
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: Wine shows client surface {surface:#x} again on the device's request \
             ({reasons:?}); later requests are logged at debug",
        );
        debug!(
            target: LOG_TARGET,
            "present: Wine shows client surface {surface:#x} again ({reasons:?})",
        );
    }

    /// win32u's `client_surface_present` on a surface the caller holds a reference on.
    fn call_present(&self, surface: usize) {
        // SAFETY: the caller holds a reference on `surface`, so it is a live
        // client surface; win32u checks it is still attached under its lock.
        unsafe { (self.present)(surface as *mut c_void) };
    }

    /// Give back a reference [`Self::retain`] took; `0` does nothing.
    pub fn release(&self, surface: usize) {
        if surface != 0 {
            // SAFETY: the caller hands over a reference `retain` took.
            unsafe { (self.release)(surface as *mut c_void) };
        }
    }
}

/// Whether anything can replace the hook: win32u's calls and a surface field on `host`.
///
/// `can_present` says the surface exports loaded; `host` is the cocoa view
/// class, `None` on a Wine without one.
fn can_replace_hook(can_present: bool, host: Option<&'static AnyClass>) -> bool {
    can_present && host.and_then(surface_ivar).is_some()
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
fn surface_field(object: &AnyObject, host: &'static AnyClass) -> usize {
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
    let Some(ivar) = surface_ivar(host) else {
        return 0;
    };
    // SAFETY: `object` is a `host`, which declares the field, and its
    // encoding is a pointer's. winemac writes the field once, on the Wine
    // thread inside the `get_win_data` that creates the cocoa view, before
    // that call returns; every read here runs on the main thread after a
    // synchronous hop the same or a later attach made after that call, so
    // the write is ordered before the read, and a later `get_win_data` for
    // the window writes a new cocoa view's field, not this one.
    let surface = unsafe { *ivar.load::<*mut c_void>(object) };
    surface as usize
}

/// The field of `host` that names the client surface, under the first name `host` declares.
///
/// `None`, logged once, when `host` declares none of them or declares one
/// that is not a pointer.
fn surface_ivar(host: &'static AnyClass) -> Option<&'static Ivar> {
    let Some((name, ivar)) = CLIENT_SURFACE_IVARS
        .iter()
        .find_map(|name| host.instance_variable(name).map(|ivar| (*name, ivar)))
    else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: this Wine's {:?} keeps none of {CLIENT_SURFACE_IVARS:?}, so no client \
             surface is read from it",
            host.name(),
        );
        return None;
    };
    let encoding = ivar.type_encoding().to_str().unwrap_or_default();
    if !<*mut c_void>::ENCODING.equivalent_to_str(encoding) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: {:?}.{name:?} is encoded {encoding:?}, not a pointer; it is not read",
            host.name(),
        );
        return None;
    }
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "present: this Wine's {:?} names its client surface in {name:?}",
        host.name(),
    );
    Some(ivar)
}

/// Resolve one of win32u's client surface functions, `None` with a warning when it is missing.
fn surface_fn(name: &CStr) -> Option<SurfaceFn> {
    // SAFETY: the named export is a win32u function taking one client
    // surface pointer and returning nothing, the type `SurfaceFn` declares.
    let symbol = unsafe { MACDRV_LIB.get::<SurfaceFn>(name.to_bytes_with_nul()) };
    if let Ok(symbol) = symbol {
        return Some(*symbol);
    }
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "present: this Wine exports no {name:?}; a window another device's surface hid stays \
         on that surface, and a metal view's client surface is not shown by the layer",
    );
    None
}

#[cfg(test)]
mod tests;
