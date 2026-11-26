//! The native host: the application hands over its own `NSView` and the layer lives inside it.
//!
//! The `hwnd` of an attach request carries the host view. The attach adds a
//! subview of mtld3d's own, backed by a `CAMetalLayer`, that fills the host
//! view and resizes with it, then takes the shared steps the Wine host takes.
//! A frame change of the subview asks the device to follow the view's size at
//! its next present.

use core::{ffi::c_void, ptr::NonNull};
use std::sync::{Arc, Once};

use block2::RcBlock;
use log::{debug, info};
use mtld3d_shared::{
    MetalHandle,
    mtl::ClientSurfaceRequest,
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, NSViewKind},
};
use objc2::{
    MainThreadMarker, MainThreadOnly,
    rc::{Retained, autoreleasepool},
    runtime::{AnyObject, NSObjectProtocol, ProtocolObject},
};
use objc2_app_kit::{
    NSApplicationDidChangeScreenParametersNotification, NSAutoresizingMaskOptions, NSView,
    NSViewFrameDidChangeNotification,
};
use objc2_foundation::{NSNotification, NSNotificationCenter};
use objc2_quartz_core::CAMetalLayer;

use super::{
    Attachment, attachment, finish_attach, refresh_all_on_main, register_layer,
    run_on_main_thread_async, run_on_main_thread_sync, screen_configuration_changed,
    view_display_caps, view_pixels,
};
use crate::{
    LOG_TARGET,
    metal::layer::{DisplayCaps, LayerAttachRequest},
};

/// The process and its signal handlers are the host's.
pub const INSTALLS_CRASH_HANDLER: bool = false;

/// Attach a `CAMetalLayer` to the host view `request.hwnd` names, in a subview that fills it.
///
/// The view handle names the subview, which the attachment record is
/// registered under. `None` for a null view or device.
pub fn attach_metal_layer(
    device_handle: MetalHandle<MTLDeviceKind>,
    request: &LayerAttachRequest,
) -> Option<(
    MetalHandle<NSViewKind>,
    MetalHandle<CAMetalLayerKind>,
    DisplayCaps,
)> {
    if request.hwnd == 0 || device_handle.is_null() {
        return None;
    }
    let host = usize::try_from(request.hwnd).expect("a 64-bit host addresses every view pointer");
    let mut subview = None;
    let mut hint = None;
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("a metal subview is made on the main thread");
        subview = add_metal_subview(host, mtm);
        if let Some((view, _)) = subview {
            hint = Some(view_display_caps(view as *mut c_void, mtm));
        }
    });
    let (view, layer) = subview?;
    let hint = hint.expect("synchronous display lookup completed");
    info!(
        target: LOG_TARGET,
        "present: layer {layer:#x} of view {view:#x} presents into host view {host:#x}",
    );
    let registered = register_layer(request, view, layer, 0, &hint);
    info!(
        target: LOG_TARGET,
        "cursor: no software overlay in a native host (cursor.software = {:?}); the host's \
         cursor is left alone",
        request.software_cursor,
    );
    Some(finish_attach(
        device_handle,
        request,
        hint,
        registered,
        false,
    ))
}

/// The client surface calls a native host has: none, so [`SurfaceCalls::load`] never answers one.
#[derive(Clone, Copy)]
pub enum SurfaceCalls {}

impl SurfaceCalls {
    /// No Wine client surface API in a native host.
    pub const fn load() -> Option<Self> {
        None
    }

    /// Never called: no value of this type exists.
    pub const fn retain(self, _surface: usize) -> usize {
        match self {}
    }

    /// Never called: no value of this type exists.
    pub const fn present(self, _surface: usize, _cause: &str) {
        match self {}
    }

    /// Never called: no value of this type exists.
    pub const fn present_requested(self, _surface: usize, _reasons: ClientSurfaceRequest) {
        match self {}
    }

    /// Never called: no value of this type exists.
    pub const fn release(self, _surface: usize) {
        match self {}
    }
}

/// Take the subview out of its host view and release both. Runs on the API thread.
///
/// `record` is the attachment record the view served, which names the host
/// view; a view no record names is released alone.
pub fn retire_metal_view(view_handle: MetalHandle<NSViewKind>, record: Option<&Attachment>) {
    if view_handle.is_null() {
        return;
    }
    let view =
        usize::try_from(view_handle.raw()).expect("a 64-bit host addresses every view pointer");
    let host = record.filter(|att| att.view() == view).map_or(0, |att| {
        usize::try_from(att.hwnd()).expect("a host view address fits usize")
    });
    if host == 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: metal view {view:#x} retired without its record; the host view it was \
             added to keeps the retain the attach took",
        );
    }
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("a metal subview is retired on the main thread");
        remove_metal_subview(view, host, mtm);
    });
    debug!(target: LOG_TARGET, "present: metal view {view:#x} retired from host view {host:#x}");
}

/// Watch the subview's frame for the record `att`, so a resize asks the device to follow it.
///
/// **Main thread only.** The record holds the boxed observer token in the
/// slot the Wine host keeps its client view watch in.
pub fn watch_view(att: &Arc<Attachment>, view: &NSView, _mtm: MainThreadMarker) {
    view.setPostsFrameChangedNotifications(true);
    let view_addr = att.view();
    let block = RcBlock::new(move |notification: NonNull<NSNotification>| {
        autoreleasepool(|_| {
            // SAFETY: AppKit posts the notification on the main thread and
            // keeps it alive for the block's call.
            let notification = unsafe { notification.as_ref() };
            let pixels = notification
                .object()
                .and_then(|object| object.downcast::<NSView>().ok())
                .and_then(|view| view_pixels(&view));
            let asked = attachment::request_client_surface(
                view_addr,
                ClientSurfaceRequest::CLIENT_AREA_CHANGED,
                false,
                pixels,
            );
            debug!(
                target: LOG_TARGET,
                "present: metal view {view_addr:#x} re-framed to {pixels:?} pixels; asked the device to follow={asked}",
            );
        });
    });
    let center = NSNotificationCenter::defaultCenter();
    // SAFETY: AppKit-exported notification-name constant, valid for the process lifetime.
    let name = unsafe { NSViewFrameDidChangeNotification };
    // SAFETY: objc2 typed binding; the observed object is the live subview,
    // the block captures an address only, and the token is removed at detach.
    let token = unsafe {
        center.addObserverForName_object_queue_usingBlock(Some(name), Some(view), None, &block)
    };
    let previous = att.replace_client_view_watch(Box::into_raw(Box::new(token)) as usize);
    remove_frame_watch(previous);
}

/// Set the layer's contents scale from the window the host view is in now. **Main thread only.**
///
/// The layer is mtld3d's own here, so nothing else sets it; the shared
/// reconciliation then republishes the backing scale it implies. `true` when
/// the scale changed: the view keeps its frame in points, so no frame change
/// tells the device its size in pixels moved, and the caller asks for it.
pub fn follow_host_backing(att: &Arc<Attachment>, view: &NSView, mtm: MainThreadMarker) -> bool {
    let Some(window) = view.window() else {
        return false;
    };
    let Some(layer) = attachment::retain_layer(att, mtm) else {
        return false;
    };
    let scale = window.backingScaleFactor();
    if layer.contentsScale().to_bits() == scale.to_bits() {
        return false;
    }
    layer.setContentsScale(scale);
    debug!(
        target: LOG_TARGET,
        "present: layer {:#x} follows backing scale {scale}",
        att.layer(),
    );
    true
}

/// Reconcile the layers on a real screen configuration change. **Main thread only.**
///
/// A plain observer, so the host's own application delegate keeps its
/// registration. Installed once and leaked for the process lifetime.
pub fn observe_screen_parameters(_mtm: MainThreadMarker) {
    /// The observer is installed.
    ///
    /// The resource is process-wide: one `NSApp` and one notification, for
    /// every attachment the observer reconciles.
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        let block = RcBlock::new(move |_notification: NonNull<NSNotification>| {
            autoreleasepool(|_| {
                let mtm = MainThreadMarker::new()
                    .expect("the screen-parameter notification is posted on the main thread");
                if screen_configuration_changed(mtm) {
                    refresh_all_on_main();
                }
            });
        });
        // SAFETY: AppKit-exported notification-name constant, valid for the process lifetime.
        let name = unsafe { NSApplicationDidChangeScreenParametersNotification };
        // SAFETY: objc2 typed binding; the block captures nothing, and the
        // token is leaked so the observer is never removed.
        let token = unsafe {
            NSNotificationCenter::defaultCenter().addObserverForName_object_queue_usingBlock(
                Some(name),
                None,
                None,
                &block,
            )
        };
        core::mem::forget(token);
    });
}

/// A native host's window layers are the host's: nothing is cleared under a covering view.
pub const fn clear_covered_window_layers(_view: &NSView, _mtm: MainThreadMarker) {}

/// No host main-thread route: the native host's work goes through the main queue.
pub const fn run_through_host_main_thread(
    _work: extern "C" fn(*mut c_void),
    _ctx: *mut c_void,
) -> bool {
    false
}

/// Remove the frame watch `att` holds, on the main thread, without waiting for it.
pub fn unwatch_view(att: &Attachment) {
    let address = att.take_client_view_watch();
    if address == 0 {
        return;
    }
    run_on_main_thread_async(move || remove_frame_watch(address));
}

/// Make the metal subview inside `host` and return it and its layer. **Main thread only.**
///
/// The subview and the host view are each retained once for the attachment;
/// the layer is the subview's own. `None` for a null host view.
fn add_metal_subview(host: usize, mtm: MainThreadMarker) -> Option<(usize, usize)> {
    // SAFETY: a native application passes an `NSView*` as the device window
    // and keeps it alive while the device holds it; the retain taken here is
    // the attachment's, given back at retire.
    let host = unsafe { Retained::retain(host as *mut NSView) }?;
    let layer = CAMetalLayer::new();
    if let Some(window) = host.window() {
        layer.setContentsScale(window.backingScaleFactor());
    }
    let view = NSView::initWithFrame(NSView::alloc(mtm), host.bounds());
    view.setAutoresizingMask(
        NSAutoresizingMaskOptions::ViewWidthSizable | NSAutoresizingMaskOptions::ViewHeightSizable,
    );
    view.setLayer(Some(&layer));
    view.setWantsLayer(true);
    host.addSubview(&view);
    let layer = Retained::as_ptr(&layer) as usize;
    let view = Retained::into_raw(view) as usize;
    // The host view's retain is the attachment's from here on.
    let _ = Retained::into_raw(host);
    Some((view, layer))
}

/// Take `view` out of its host view and release it, and the host view when `host` names one.
fn remove_metal_subview(view: usize, host: usize, _mtm: MainThreadMarker) {
    // SAFETY: `view` is the address `add_metal_subview` leaked one retain of,
    // taken back exactly once, here.
    let view = unsafe { Retained::from_raw(view as *mut NSView) };
    if let Some(view) = view {
        view.removeFromSuperview();
    }
    if host != 0 {
        // SAFETY: `host` is the host view the attach retained once for the
        // attachment the retiring record served; that retain ends here.
        drop(unsafe { Retained::from_raw(host as *mut NSView) });
    }
}

/// Remove and release the frame observer at `address`; `0` does nothing.
fn remove_frame_watch(address: usize) {
    if address == 0 {
        return;
    }
    // SAFETY: `address` came out of `Box::into_raw` in `watch_view` and was
    // taken out of its record by exactly one swap, so this is the box's owner.
    let token =
        unsafe { Box::from_raw(address as *mut Retained<ProtocolObject<dyn NSObjectProtocol>>) };
    let observer: &AnyObject = (**token).as_ref();
    // SAFETY: objc2 typed binding; the token is an observer the default
    // center handed out, removed once.
    unsafe { NSNotificationCenter::defaultCenter().removeObserver(observer) };
}
