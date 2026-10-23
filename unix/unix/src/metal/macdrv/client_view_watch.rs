//! The watch on the client view a metal view sits in: hidden means the device asks to be shown.
//!
//! Wine shows a window's client surface by unhiding its cocoa view and hiding
//! the window's other one. Some Wines also hide it whenever they flush the
//! window surface, which a resize, a move into native fullscreen or any GDI
//! drawing into the window does, and count on winemac's `nextDrawable`
//! override to show it again on the next present. The layer presents
//! without that override (`client_surface`), so a key-value observer on the
//! client view's `hidden` stands in for it: when Wine hides the view of the
//! newest device on a window, the observer sets that device's request word,
//! and the device's next `Present` that submits a frame has Wine show its
//! surface again.
//!
//! The observer and the client view are retained by a box whose address the
//! attachment record carries from the attach that installs the watch to the
//! teardown that removes it, so the view outlives the registration and the
//! observer outlives every notification. Both are created, registered,
//! removed and released on the main thread; the record only holds the address.

use core::ffi::c_void;
use std::sync::Arc;

use log::debug;
use mtld3d_shared::mtl::ClientSurfaceRequest;
use objc2::{
    MainThreadMarker, MainThreadOnly, define_class, extern_methods,
    rc::Retained,
    runtime::{AnyObject, NSObject},
};
use objc2_app_kit::NSView;
use objc2_foundation::{
    NSKeyValueObservingOptions, NSObjectNSKeyValueObserverRegistration, NSString,
};

use super::attachment::{self, Attachment};
use crate::LOG_TARGET;

/// The key path the observer watches.
const HIDDEN_KEY: &str = "hidden";

define_class!(
    /// Observes one client view's `hidden` for the device whose metal view sits in it.
    ///
    /// The registration's context is the metal view's address, the key the
    /// device's attachment record is registered under; it is only ever used to
    /// look that record up, never dereferenced.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "Mtld3dClientViewObserver"]
    struct ClientViewObserver;

    impl ClientViewObserver {
        #[unsafe(method(observeValueForKeyPath:ofObject:change:context:))]
        fn observe_value(
            &self,
            _key_path: Option<&NSString>,
            object: Option<&AnyObject>,
            _change: Option<&AnyObject>,
            context: *mut c_void,
        ) {
            client_view_changed(object, context.addr());
        }
    }
);

impl ClientViewObserver {
    extern_methods!(
        #[unsafe(method(new))]
        #[unsafe(method_family = new)]
        fn new(mtm: MainThreadMarker) -> Retained<Self>;
    );
}

/// What the attachment record holds the address of while a watch is installed.
struct ClientViewWatch {
    client_view: Retained<NSView>,
    observer: Retained<ClientViewObserver>,
    /// The metal view's address the registration names, for the removal's log line.
    view: usize,
}

/// Watch the client view `view` sits in for the record `att`. **Main thread only.**
///
/// A view outside any client view, which a view mid-teardown can be, is not
/// watched, logged once. A watch the record still held is removed. A view
/// already hidden when the watch is installed asks at once, since Wine can
/// hide it between the attach's present and the registration.
pub fn watch(att: &Arc<Attachment>, view: &NSView, mtm: MainThreadMarker) {
    // SAFETY: objc2 typed binding; `superview` is unsafe only because the
    // view hierarchy may change under a caller off the main thread, and the
    // marker puts this read on the main thread, where that hierarchy changes.
    let Some(client_view) = (unsafe { view.superview() }) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: metal view {:#x} has no superview at attach, so nothing watches whether \
             Wine hides its client view",
            att.view(),
        );
        return;
    };
    let observer = ClientViewObserver::new(mtm);
    let key = NSString::from_str(HIDDEN_KEY);
    // SAFETY: `observer` implements `observeValueForKeyPath:ofObject:change:context:`,
    // and the box below keeps it alive until `unwatch` removes it from the
    // view, which the same box keeps alive; the context is an integer the
    // observer uses only as a registry key and never dereferences.
    unsafe {
        client_view.addObserver_forKeyPath_options_context(
            &observer,
            &key,
            NSKeyValueObservingOptions::empty(),
            core::ptr::without_provenance_mut(att.view()),
        );
    };
    let hidden_at_install = client_view.isHidden();
    debug!(
        target: LOG_TARGET,
        "present: watching client view {:#x} of view {:#x} (hidden={hidden_at_install})",
        Retained::as_ptr(&client_view) as usize,
        att.view(),
    );
    let watch = Box::new(ClientViewWatch {
        client_view,
        observer,
        view: att.view(),
    });
    let previous = att.replace_client_view_watch(Box::into_raw(watch) as usize);
    remove(previous, mtm);
    // Wine may have hidden the view between the attach's present and the
    // registration above, and no notification announces that one.
    if hidden_at_install {
        let asked =
            attachment::request_client_surface(att.view(), ClientSurfaceRequest::VIEW_HIDDEN);
        debug!(
            target: LOG_TARGET,
            "present: client view of view {:#x} was already hidden at install; asked={asked}",
            att.view(),
        );
    }
}

/// Remove the watch `att` holds, on the main thread, without waiting for it.
///
/// Called on the API thread after teardown unregistered the record, so a
/// notification that arrives before the removal finds no record and asks
/// nothing. The hop is the asynchronous one, the only one a thread winemac
/// may be waiting on is allowed to take.
pub fn unwatch(att: &Attachment) {
    let address = att.take_client_view_watch();
    if address == 0 {
        return;
    }
    super::run_on_main_thread_async(move || {
        let mtm =
            MainThreadMarker::new().expect("a client view watch is removed on the main thread");
        remove(address, mtm);
    });
}

/// Unregister and release the watch at `address`; `0` does nothing. **Main thread only.**
fn remove(address: usize, _mtm: MainThreadMarker) {
    if address == 0 {
        return;
    }
    // SAFETY: `address` came out of `Box::into_raw` in `watch` and was taken
    // out of its record by exactly one swap, so this is the box's one owner.
    let watch = unsafe { Box::from_raw(address as *mut ClientViewWatch) };
    let key = NSString::from_str(HIDDEN_KEY);
    // SAFETY: the observer was registered on this view for this key path by
    // `watch`, and the box kept both alive since.
    unsafe {
        watch
            .client_view
            .removeObserver_forKeyPath(&watch.observer, &key);
    }
    debug!(
        target: LOG_TARGET,
        "present: client view watch of view {:#x} removed",
        watch.view,
    );
}

/// A watched client view's `hidden` changed; ask for the device's surface when it is hidden now.
fn client_view_changed(object: Option<&AnyObject>, view: usize) {
    if MainThreadMarker::new().is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: a client view was hidden or shown off the main thread; the device is asked \
             to show its client surface without reading which",
        );
        attachment::request_client_surface(view, ClientSurfaceRequest::VIEW_HIDDEN);
        return;
    }
    let Some(client_view) = object.and_then(AnyObject::downcast_ref::<NSView>) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: a client view watch was told of a change on no view → ignored",
        );
        return;
    };
    let hidden = client_view.isHidden();
    let asked =
        hidden && attachment::request_client_surface(view, ClientSurfaceRequest::VIEW_HIDDEN);
    debug!(
        target: LOG_TARGET,
        "present: client view {:#x} of view {view:#x} hidden={hidden}{}",
        core::ptr::from_ref(client_view) as usize,
        if asked { "; the device is asked to show its client surface again" } else { "" },
    );
}
