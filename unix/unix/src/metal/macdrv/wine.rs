//! The Wine host: a device's metal view comes from winemac and lives in Wine's client view.
//!
//! Everything here reaches winemac through the `macdrv_functions` table it
//! publishes. The shared half of the attach is the parent module's, which the
//! native host calls the same way.

use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
};
use std::sync::{Arc, LazyLock, Mutex};

use block2::{Block, RcBlock};
use libloading::os::unix::Library;
use log::{debug, error, info, log_enabled};
use mtld3d_shared::{
    MetalHandle,
    mtl::ClientSurfaceRequest,
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, NSViewKind},
};
use objc2::{
    MainThreadMarker, MainThreadOnly, extern_class, extern_methods,
    rc::{Retained, autoreleasepool},
    runtime::AnyClass,
};
use objc2_app_kit::{NSView, NSWindow};
use objc2_core_foundation::CGRect;

use super::{
    Attachment, LayerMode, attachment, finish_attach, refresh_all_on_main, register_layer,
    run_on_main_thread_sync, screen_configuration_changed, view_display_caps,
};
use crate::{
    LOG_TARGET,
    metal::layer::{DisplayCaps, LayerAttachRequest},
};

#[path = "wine/client_surface.rs"]
mod client_surface;
#[path = "wine/client_view_watch.rs"]
mod client_view_watch;
#[path = "wine/delegate_forward.rs"]
mod delegate_forward;

pub use client_surface::SurfaceCalls;
pub use client_view_watch::{unwatch as unwatch_view, watch as watch_view};

pub use self::install_screen_params_filter as observe_screen_parameters;

/// Count of `NSApplicationDidChangeScreenParametersNotification` deliveries.
static SCREEN_PARAM_CHANGES: AtomicU64 = AtomicU64::new(0);

/// Wine's `NSApplication` delegate, retained for the process lifetime.
///
/// Set once by [`install_screen_params_filter`]; zero until then, and for
/// good when the delegate does not implement the method, since then there is
/// nothing to forward to. Held as an address because the observer block that
/// reads it must not capture a `!Send` `Retained`, and the delegate is only
/// ever touched on the main thread, where the notification is posted.
static WINE_APP_DELEGATE_PTR: AtomicUsize = AtomicUsize::new(0);

/// The process is Wine's, whose faults the crash handler knows how to hand back.
pub const INSTALLS_CRASH_HANDLER: bool = true;

/// Leave the layer's contents scale to Wine, which sets it from its retina mode.
///
/// `false`: nothing changed here that the device's window procedure does not
/// report itself.
pub const fn follow_host_backing(
    _att: &Arc<Attachment>,
    _view: &NSView,
    _mtm: MainThreadMarker,
) -> bool {
    false
}

type GetWinDataFn = unsafe extern "C" fn(*mut c_void) -> *mut MacdrvWinData;
type ReleaseWinDataFn = unsafe extern "C" fn(*mut MacdrvWinData);
type CreateMetalViewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
type GetMetalLayerFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type ReleaseMetalViewFn = unsafe extern "C" fn(*mut c_void);
type GetCocoaWindowFn = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;

/// Process-wide handle to the dynamic-symbol table, resolved once.
///
/// `libloading::os::unix::Library::this()` mirrors the
/// `dlopen(NULL, …)` / `RTLD_DEFAULT` symbol space, which is where the
/// `macdrv_functions` table Wine publishes is reachable from. That table
/// is the only macdrv symbol winemac's unix library exports, so it is the
/// only one this handle is asked for. On the unix backend the `this()`
/// constructor is safe (no file is loaded; the handle lives for the
/// process lifetime).
static MACDRV_LIB: LazyLock<Library> = LazyLock::new(Library::this);

/// Run `work(ctx)` on the main thread through Wine's `OnMainThread`; whether it ran.
///
/// `false` when this Wine publishes no usable entry, or when its entry
/// returned without running the work.
pub fn run_through_host_main_thread(work: extern "C" fn(*mut c_void), ctx: *mut c_void) -> bool {
    /// Whether this Wine's `OnMainThread` returned without running a hop's work.
    ///
    /// A machine fact about the loaded Wine, latched by the first hop that
    /// sees it and never cleared: `CrossOver`'s arm64 build publishes the entry
    /// as a stub that logs and drops the block, and every later hop goes
    /// straight to the main queue rather than ask it again.
    static WINE_HOP_DROPS_WORK: AtomicBool = AtomicBool::new(false);
    if !WINE_HOP_DROPS_WORK.load(Ordering::Relaxed)
        && let Some(on_main_thread) = wine_on_main_thread()
    {
        if run_through_wine(on_main_thread, work, ctx) {
            return true;
        }
        WINE_HOP_DROPS_WORK.store(true, Ordering::Relaxed);
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "main-thread hop: this Wine's OnMainThread returned without running the work → \
             libdispatch's main queue, which winemac does not drain while it waits for a query",
        );
    }
    false
}

/// Wine's `OnMainThread`, read from the `macdrv_functions` table; `None` without one.
fn wine_on_main_thread() -> Option<unsafe extern "C" fn(&Block<dyn Fn()>)> {
    let table = macdrv_functions()?;
    // SAFETY: the entry is Wine's `void (*)(dispatch_block_t)` stored as
    // `*mut c_void` per its C ABI, and a `dispatch_block_t` is a pointer to a
    // block; a null entry reads as `None`.
    unsafe {
        core::mem::transmute::<*mut c_void, Option<unsafe extern "C" fn(&Block<dyn Fn()>)>>(
            table.on_main_thread,
        )
    }
}

/// Run `work(ctx)` on the main thread through Wine's `OnMainThread`; whether it ran.
///
/// Wine's own entry runs the block on the main thread before it returns. A
/// stub entry returns without running it, and then the answer is `false`
/// and the work is the caller's to run another way. The block claims the run
/// on a state it shares with this call before it reaches `ctx`, and this call
/// claims it the other way before it answers `false`, so a block an entry
/// kept and runs later finds the run taken and touches nothing of the
/// caller's. The block carries `ctx` as an address rather than a borrow for
/// the same reason: Wine keeps a copy of the block until its request loop
/// lets it go, which can be after this returns.
fn run_through_wine(
    on_main_thread: unsafe extern "C" fn(&Block<dyn Fn()>),
    work: extern "C" fn(*mut c_void),
    ctx: *mut c_void,
) -> bool {
    const PENDING: u8 = 0;
    const RUNNING: u8 = 1;
    const DONE: u8 = 2;
    const ABANDONED: u8 = 3;
    let state = Arc::new(AtomicU8::new(PENDING));
    let block_state = Arc::clone(&state);
    let ctx_addr = ctx.expose_provenance();
    let block = RcBlock::new(move || {
        if block_state
            .compare_exchange(PENDING, RUNNING, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            work(core::ptr::with_exposed_provenance_mut(ctx_addr));
            block_state.store(DONE, Ordering::Release);
        }
    });
    // SAFETY: the entry takes a block and returns; `ctx` is live until this
    // call returns, and the block reaches it only while this call waits
    // below. The caller does not call this from the main thread, where a
    // synchronous wait could never end, as it does not for `dispatch_sync_f`
    // either.
    unsafe { on_main_thread(&block) };
    loop {
        match state.compare_exchange(PENDING, ABANDONED, Ordering::Acquire, Ordering::Acquire) {
            Ok(_) => return false,
            Err(DONE) => return true,
            // An entry that runs the block on its own schedule is running it
            // now; `ctx` stays live until it is done.
            Err(_) => std::thread::yield_now(),
        }
    }
}

/// Resolves HWND → `CAMetalLayer` via Wine's macdrv.
///
/// Returns (`view_handle`, `layer_handle`, `display_caps`).
/// Display-caps field:
/// - `backing_scale` is the Wine metal layer's `contentsScale` rounded +
///   clamped to `[1, 8]`: 2 when the prefix runs in retina mode, else 1.
///
/// Side effect: registers an attachment record for the view, keyed by the
/// view address the PE side gets back, holding the layer, the pacing, the
/// backing scale, both user settings and whether the layer was configured
/// for HDR (the display has EDR potential and `hdr_enable` is set), so the
/// per-present poll can follow this window onto another display and
/// re-derive everything that display decides, for this device alone.
pub fn attach_metal_layer(
    device_handle: MetalHandle<MTLDeviceKind>,
    request: &LayerAttachRequest,
) -> Option<(
    MetalHandle<NSViewKind>,
    MetalHandle<CAMetalLayerKind>,
    DisplayCaps,
)> {
    let hwnd = request.hwnd;
    if hwnd == 0 || device_handle.is_null() {
        return None;
    }

    let funcs = MacdrvFuncs::load()?;
    // A view kept from this window's previous device is taken as it is; one
    // kept from a window that is gone is moved in; only a window with
    // neither gets a view from Wine.
    let kept = kept_metal_view(&funcs, hwnd);
    let hosted = matches!(kept, Some(KeptView::Hosted { .. }));
    let surface_calls = client_surface::SurfaceCalls::load();
    // A view that went through `get_win_data` comes back with its client
    // surface already retained, under the window data lock that keeps the
    // surface alive; a hosted view's is read and retained below.
    let (view, layer, retained) = match kept {
        Some(KeptView::Hosted { view, layer }) => (view as *mut c_void, layer as *mut c_void, None),
        Some(KeptView::Orphan {
            view,
            layer,
            from_hwnd,
        }) => adopt_metal_view(
            &funcs,
            hwnd,
            device_handle,
            &KeptLayer {
                view,
                layer,
                from_hwnd,
            },
            surface_calls.as_ref(),
        )?,
        None => create_metal_view(&funcs, hwnd, device_handle, surface_calls.as_ref())?,
    };
    // AppKit owns the view's window and screen relationships on the main
    // thread; the view is retained for as long as the device holds it. The
    // same hop takes the layer off winemac's `nextDrawable` override, which
    // walks the view from the presenter thread, reads a hosted view's
    // client surface, which the override would have presented, takes the
    // fade out of the window's order-out, so the layer can move on at once
    // when the window goes, and shows a kept view the park hid, before the
    // present below shows its client view.
    let mut hint = None;
    let mut layer_class = client_surface::LayerClass::Plain;
    let mut surface = 0;
    let can_present = surface_calls.is_some();
    run_on_main_thread_sync(|| {
        let mtm = objc2::MainThreadMarker::new().expect("display lookup runs on the main thread");
        show_metal_view(view, mtm);
        hint = Some(view_display_caps(view, mtm));
        layer_class = client_surface::bypass_present_hook(layer, can_present, mtm);
        if !order_out_without_fade(view, mtm) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "present: a metal view is in no window at attach; the window it joins later \
                 keeps its order-out fade until the view is parked",
            );
        }
        if retained.is_none() {
            surface = client_surface::client_surface_of(view, mtm);
        }
    });
    let hint = hint.expect("synchronous display lookup completed");
    let surface = retained.unwrap_or_else(|| {
        surface_calls
            .as_ref()
            .map_or(0, |calls| calls.retain(surface))
    });
    info!(
        target: LOG_TARGET,
        "present: layer {:#x} of view {:#x} presents through CAMetalLayer ({layer_class:?}); \
         client surface {surface:#x}",
        layer as usize,
        view as usize,
    );
    if hosted {
        info!(
            target: LOG_TARGET,
            "present: metal view {:#x} kept from the previous device on window {hwnd:#x} is \
             attached again, layer and all",
            view as usize,
        );
    }
    // A record still registered for this view is one whose teardown never
    // ran; the new record replaces it, and the reference it held on its
    // client surface goes with it.
    if let Some(stale) = attachment::unregister(view as usize) {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: view {:#x} attached again without a teardown between; the earlier \
             record is dropped",
            view as usize,
        );
        client_view_watch::unwatch(&stale);
        if let Some(calls) = &surface_calls {
            calls.release(stale.client_surface());
        }
    }
    // The cursor scale follows Wine's retina mode, which the layer carries as
    // its contents scale, not the display's own factor: in non-retina mode
    // macOS already doubles everything the game draws, its cursor included.
    let registered = register_layer(request, view as usize, layer as usize, surface, &hint);
    // A new view's surface is the one Wine shows already; a kept view's may
    // have been hidden for a later surface of the same window since. This
    // present also has win32u set the surface's frame and superview. The
    // device's first submitted frame presents the surface again, which is
    // the present after which Wine keeps the client view shown across its
    // window surface flushes; until then no other device's call presents it.
    if let Some(calls) = &surface_calls {
        calls.present(surface, "attach");
    }
    // Every device with a surface asks after its first frame, whatever its
    // rank, so the unix side learns it has drawn
    // ([`Attachment::note_frame_submitted`]).
    if surface != 0 {
        attachment::request_client_surface(
            view as usize,
            ClientSurfaceRequest::ATTACHED,
            false,
            None,
        );
    }
    // `Auto` turns the software cursor on under HDR, where the hardware
    // cursor reads dim. The overlay window costs the game window its
    // direct-to-display path in either mode.
    let software_cursor = request.software_cursor;
    let software_cursor_active = software_cursor.resolve(registered.mode == LayerMode::Hdr);
    info!(
        target: LOG_TARGET,
        "cursor: software overlay {} (cursor.software = {software_cursor:?}, layer {:?})",
        if software_cursor_active { "on" } else { "off" },
        registered.mode,
    );
    Some(finish_attach(
        device_handle,
        request,
        hint,
        registered,
        software_cursor_active,
    ))
}

/// Create a metal view and its layer on `hwnd`'s client view through Wine.
///
/// Wine creates a client surface for the window on every `get_win_data`,
/// with a cocoa view the metal view is then created in, and makes that
/// surface the one it shows. `None`, with the failure logged, when any step
/// hands back null.
fn create_metal_view(
    funcs: &MacdrvFuncs,
    hwnd: u64,
    device_handle: MetalHandle<MTLDeviceKind>,
    calls: Option<&client_surface::SurfaceCalls>,
) -> Option<(*mut c_void, *mut c_void, Option<usize>)> {
    let win_data = get_win_data(funcs, hwnd)?;
    // SAFETY: `win_data` is the live record `get_win_data` handed back,
    // valid until `release_win_data`.
    let client_view = unsafe { (*win_data).client_cocoa_view };
    let result = wine_metal_view(funcs, client_view, device_handle)
        .map(|(view, layer)| (view, layer, Some(retain_client_surface(client_view, calls))));
    release_win_data(funcs, win_data);
    result
}

/// Read and retain the client surface `client_view` shows, `0` when there is none.
///
/// Called while the caller holds the window data `get_win_data` locked: the
/// window's data holds Wine's own reference on the surface, and the one
/// place that drops it, the driver's `DestroyWindow`, takes that lock first,
/// so the surface stays alive from the read to the reference taken here.
/// The reference is taken on the calling thread, a Wine thread; the surface
/// is presented only after the lock is released, since presenting takes
/// win32u's surface lock and then the window data, the order a concurrent
/// `detach_client_surfaces` already holds the first of.
fn retain_client_surface(
    client_view: *mut c_void,
    calls: Option<&client_surface::SurfaceCalls>,
) -> usize {
    let Some(calls) = calls else {
        return 0;
    };
    if client_view.is_null() {
        return 0;
    }
    let mut surface = 0;
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("the client surface is read on the main thread");
        surface = client_surface::client_view_surface(client_view, mtm);
    });
    calls.retain(surface)
}

/// Move a kept view, `layer` and all, into `hwnd`'s window for the device attaching there.
///
/// Wine creates the client surface for the window as it does for a new
/// view, and the kept view takes the place in that surface's cocoa view a
/// new one gets ([`adopt_view_into_client`]). The surface's own frame,
/// superview and unhide requests sit on Wine's main-thread queue when
/// `get_win_data` returns; that queue runs in order and Wine's synchronous
/// calls ride it, so the `macdrv_view_get_metal_layer` round trip here
/// returns after they have run, and is at the same time the check that the
/// view still carries the parked layer. A view that does not, or a surface
/// with no cocoa view, is released, and the window gets a view of its own
/// as if nothing had been kept; only a window Wine has no record for fails
/// the attach.
fn adopt_metal_view(
    funcs: &MacdrvFuncs,
    hwnd: u64,
    device_handle: MetalHandle<MTLDeviceKind>,
    kept: &KeptLayer,
    calls: Option<&client_surface::SurfaceCalls>,
) -> Option<(*mut c_void, *mut c_void, Option<usize>)> {
    let &KeptLayer {
        view,
        layer,
        from_hwnd,
    } = kept;
    let Some(win_data) = get_win_data(funcs, hwnd) else {
        release_metal_view(view);
        return None;
    };
    // SAFETY: `win_data` is the live record `get_win_data` handed back,
    // valid until `release_win_data`.
    let client_view = unsafe { (*win_data).client_cocoa_view };
    // SAFETY: extern "C" Wine entry point; `view` is the kept view, which the
    // park handed to this caller still retained.
    let carried = unsafe { (funcs.macdrv_view_get_metal_layer)(view as *mut c_void) } as usize;
    let mut moved = false;
    if client_view.is_null() {
        error!(
            target: LOG_TARGET,
            "present: Wine's client surface for window {hwnd:#x} has no cocoa view; the metal \
             view {view:#x} kept from window {from_hwnd:#x} is released",
        );
    } else if carried != layer {
        error!(
            target: LOG_TARGET,
            "present: metal view {view:#x} kept from window {from_hwnd:#x} carries layer \
             {carried:#x}, not the kept layer {layer:#x}; released, and window {hwnd:#x} gets a \
             view of its own",
        );
    } else {
        run_on_main_thread_sync(|| {
            let mtm = MainThreadMarker::new().expect("the kept view is moved on the main thread");
            moved = adopt_view_into_client(view, client_view as usize, mtm);
        });
        if !moved {
            error!(
                target: LOG_TARGET,
                "present: metal view {view:#x} kept from window {from_hwnd:#x} could not be \
                 moved into window {hwnd:#x}; released, and the window gets a view of its own",
            );
        }
    }
    let result = if moved {
        info!(
            target: LOG_TARGET,
            "present: metal view {view:#x} kept from window {from_hwnd:#x}, whose window is \
             gone, is moved into window {hwnd:#x}, layer and all",
        );
        Some((view as *mut c_void, layer as *mut c_void))
    } else {
        release_metal_view(view);
        if client_view.is_null() {
            None
        } else {
            wine_metal_view(funcs, client_view, device_handle)
        }
    }
    .map(|(view, layer)| (view, layer, Some(retain_client_surface(client_view, calls))));
    release_win_data(funcs, win_data);
    result
}

/// Give a kept view the place in `client_view` a new metal view gets. **Main thread only.**
///
/// What `newMetalViewWithDevice` does for a view it creates: the client
/// view's bounds as the frame, the client view resizing its subviews (off
/// on one until then), the view below every other subview, and the window's
/// `windowDidDrawContent`, after which Wine treats the window as having
/// content. The view is taken out of whatever still holds it first: a window
/// Wine has closed may keep its client view until a later request disposes
/// it. `false` when either view is not there.
fn adopt_view_into_client(view: usize, client_view: usize, mtm: MainThreadMarker) -> bool {
    use objc2_app_kit::NSWindowOrderingMode;

    let Some(view) = retain_parked_view(view, mtm) else {
        return false;
    };
    // SAFETY: `client_view` is the cocoa view of the client surface Wine
    // created for the attaching window, read from the `get_win_data` record
    // the caller holds until its `release_win_data`; the retain taken here
    // covers the calls below.
    let Some(client) = (unsafe { Retained::retain(client_view as *mut NSView) }) else {
        return false;
    };
    view.removeFromSuperview();
    view.setFrame(client.bounds());
    client.setAutoresizesSubviews(true);
    client.addSubview_positioned_relativeTo(&view, NSWindowOrderingMode::Below, None);
    if let Some(window) = client.window() {
        window_did_draw_content(&window, mtm);
    } else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: a kept metal view was moved into a client view that is in no window yet; \
             the window is not told it has content",
        );
    }
    true
}

extern_class!(
    /// winemac's window class, told when a moved-in metal view gives it content.
    ///
    /// Declared here because no binding crate carries Wine's classes.
    /// The content notification matches `newMetalViewWithDevice`, and
    /// `closing` distinguishes a destroyed host from a live child window's
    /// host. The class is looked up by name before use, and a window that
    /// is not one of the driver's is left alone.
    #[unsafe(super(NSWindow))]
    #[thread_kind = MainThreadOnly]
    #[name = "WineWindow"]
    struct WineWindow;
);

impl WineWindow {
    extern_methods!(
        #[unsafe(method(windowDidDrawContent))]
        #[unsafe(method_family = none)]
        fn window_did_draw_content(&self);

        #[unsafe(method(closing))]
        #[unsafe(method_family = none)]
        fn closing(&self) -> bool;
    );
}

/// Whether the running driver has a `WineWindow` class at all.
static HAS_WINE_WINDOW: LazyLock<bool> = LazyLock::new(|| AnyClass::get(c"WineWindow").is_some());

/// Tell a Wine window it has content, as a new metal view's creation does. **Main thread only.**
fn window_did_draw_content(window: &NSWindow, _mtm: MainThreadMarker) {
    if !*HAS_WINE_WINDOW {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: this driver has no WineWindow class; a window given a kept metal view is \
             not told it has content",
        );
        return;
    }
    if let Some(window) = window.downcast_ref::<WineWindow>() {
        window.window_did_draw_content();
    } else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: the window given a kept metal view is not a WineWindow; it is not told it \
             has content",
        );
    }
}

/// Have the window `view` sits in order out without `AppKit`'s fade. **Main thread only.**
///
/// A window with the default animation behaviour fades when it is ordered
/// out or closed, and `AppKit` finishes ordering it out only when the fade
/// ends, about a quarter of a second later. A layer the window showed that
/// moves into another window before then, as a kept view does when an
/// application destroys its device window and creates the next one at once
/// ([`adopt_metal_view`]), never has a drawable displayed in its new window:
/// every presented handler reports a `presentedTime` of 0 while the window
/// reads visible. Without the fade the order-out is immediate and the moved
/// layer displays. Attach sets it, and the park again for a view that was in
/// no window then; the window keeps it after its device leaves, since it
/// still holds the kept view. `false` when the view is in no window.
fn order_out_without_fade(view: *mut c_void, _mtm: MainThreadMarker) -> bool {
    use objc2_app_kit::NSWindowAnimationBehavior;

    if view.is_null() {
        return false;
    }
    // SAFETY: the view is a retained one attach or the park holds, created
    // by Wine or kept since; the retain taken here covers the walk to its
    // window.
    let Some(view) = (unsafe { Retained::retain(view.cast::<NSView>()) }) else {
        return false;
    };
    let Some(window) = view.window() else {
        return false;
    };
    if window.animationBehavior() != NSWindowAnimationBehavior::None {
        window.setAnimationBehavior(NSWindowAnimationBehavior::None);
    }
    true
}

/// Wine's record for `hwnd`, with a client surface created for the window on every call.
///
/// `None`, logged, when Wine has no record for the handle. Given back with
/// [`release_win_data`].
fn get_win_data(funcs: &MacdrvFuncs, hwnd: u64) -> Option<*mut MacdrvWinData> {
    // SAFETY: `get_win_data` is the table entry resolved at load; `hwnd` is
    // the PE-supplied window handle (non-zero per the check in the caller).
    let win_data = unsafe { (funcs.get_win_data)(hwnd as *mut c_void) };
    if win_data.is_null() {
        error!(target: LOG_TARGET, "get_win_data returned null for hwnd 0x{hwnd:x}");
        return None;
    }
    Some(win_data)
}

/// Give back the record [`get_win_data`] handed out.
fn release_win_data(funcs: &MacdrvFuncs, win_data: *mut MacdrvWinData) {
    // SAFETY: `release_win_data` matches the `get_win_data` that handed out
    // `win_data`, the live pointer returned there.
    unsafe { (funcs.release_win_data)(win_data) };
}

/// A metal view and its layer Wine creates in `client_view`, or `None` with the failure logged.
fn wine_metal_view(
    funcs: &MacdrvFuncs,
    client_view: *mut c_void,
    device_handle: MetalHandle<MTLDeviceKind>,
) -> Option<(*mut c_void, *mut c_void)> {
    // SAFETY: `macdrv_view_create_metal_view` is the table entry resolved at
    // load; `client_view` is the Cocoa view read from a live `get_win_data`
    // record.
    let view = unsafe {
        (funcs.macdrv_view_create_metal_view)(client_view, device_handle.raw() as *mut c_void)
    };
    if view.is_null() {
        error!(target: LOG_TARGET, "macdrv_view_create_metal_view returned null");
        return None;
    }
    // SAFETY: `macdrv_view_get_metal_layer` is the table entry resolved at
    // load; `view` is non-null per the check above.
    let layer = unsafe { (funcs.macdrv_view_get_metal_layer)(view) };
    if layer.is_null() {
        error!(target: LOG_TARGET, "macdrv_view_get_metal_layer returned null; view released");
        // SAFETY: extern "C" Wine entry point; `view` is the retained view
        // `macdrv_view_create_metal_view` handed out above, which nothing else
        // holds, so this gives back its only reference.
        unsafe { (funcs.macdrv_view_release_metal_view)(view) };
        return None;
    }
    Some((view, layer))
}

/// A kept view and its layer moving from the window `from_hwnd` into another.
struct KeptLayer {
    view: usize,
    layer: usize,
    from_hwnd: u64,
}

/// What the park holds for a device attaching to a window, owned by the caller from here on.
enum KeptView {
    /// The view kept for the handle, still in that handle's window; reused as it is.
    Hosted { view: usize, layer: usize },
    /// A kept view whose own handle has no window any more; moved into the attaching one.
    Orphan {
        view: usize,
        layer: usize,
        from_hwnd: u64,
    },
}

/// The kept view for a device attaching to `hwnd`, if the park holds one for it.
///
/// The view kept for `hwnd` itself comes first: still hosted by the handle's
/// window it is reused, and taking it goes through none of Wine's calls, so
/// the client surface it sits in stays the one Wine shows for the window
/// (one Wine hid for a later surface of the same window is shown again by
/// the attach, which presents the surface the view sits in). No longer
/// hosted (the handle reused by a new window, or its window gone) it is
/// moved into the window the handle has now. With
/// none kept for `hwnd`, the newest kept view whose own handle has no window
/// any more is moved in; a kept view whose handle still has a window is that
/// window's, on screen or not, and stays.
///
/// A slot is taken or retained under the park lock before its view is touched,
/// since another device's retire can displace a slot and release its view at
/// any time. The park lock is never held across a Wine call or main-thread hop.
/// Runs on the API thread: `macdrv_get_cocoa_window` takes Wine's window-data
/// lock, which the driver holds across a synchronous main-thread request
/// while it destroys a window, so it is never called from the main thread.
fn kept_metal_view(funcs: &MacdrvFuncs, hwnd: u64) -> Option<KeptView> {
    let own = PARKED_METAL_VIEW
        .lock()
        .expect("metal view park mutex poisoned")
        .take_for(hwnd);
    if let Some((view, layer)) = own {
        return Some(if kept_view_hosted(funcs, hwnd, view) {
            KeptView::Hosted { view, layer }
        } else {
            KeptView::Orphan {
                view,
                layer,
                from_hwnd: hwnd,
            }
        });
    }
    // A Wine whose table cannot say whose window is gone (no
    // `macdrv_get_cocoa_window`) gets no view moved between handles.
    let get_cocoa_window = funcs.macdrv_get_cocoa_window?;
    let newest_first = PARKED_METAL_VIEW
        .lock()
        .expect("metal view park mutex poisoned")
        .slots_newest_first();
    let orphan = pick_orphan(&newest_first, |kept| {
        cocoa_window_of(get_cocoa_window, kept.hwnd) == 0 && kept_view_host_gone(kept)
    })?;
    let taken = PARKED_METAL_VIEW
        .lock()
        .expect("metal view park mutex poisoned")
        .take_kept(orphan.view, orphan.seq)?;
    Some(KeptView::Orphan {
        view: taken.view,
        layer: taken.layer,
        from_hwnd: taken.hwnd,
    })
}

/// The newest kept view whose own handle has no window any more, among `newest_first`.
///
/// `window_gone` checks both the handle and the view's host: a child handle
/// has no Cocoa window of its own even while its view sits in a live parent.
/// A view in a live host, on screen or not, is not picked.
fn pick_orphan(
    newest_first: &[KeptMetalView],
    window_gone: impl Fn(&KeptMetalView) -> bool,
) -> Option<&KeptMetalView> {
    newest_first.iter().find(|kept| window_gone(kept))
}

/// Whether a still-parked view is detached or belongs to a closing Wine window.
///
/// A child HWND has no Cocoa window of its own, so the handle lookup alone
/// cannot establish that its view is orphaned. Retain under the park lock
/// on the main thread, then release the lock before walking `AppKit`. A slot
/// taken or replaced since the snapshot is skipped without touching its
/// address; `take_kept` checks it again before transferring ownership.
fn kept_view_host_gone(kept: &KeptMetalView) -> bool {
    let mut gone = false;
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("kept view host check runs on the main thread");
        let view = {
            let park = PARKED_METAL_VIEW
                .lock()
                .expect("metal view park mutex poisoned");
            park.slots
                .iter()
                .find(|slot| slot.view == kept.view && slot.seq == kept.seq)
                .and_then(|slot| retain_parked_view(slot.view, mtm))
        };
        let Some(view) = view else {
            // Another attachment took or displaced this parking.
            return;
        };
        let Some(window) = view.window() else {
            gone = true;
            return;
        };
        if *HAS_WINE_WINDOW && let Some(window) = window.downcast_ref::<WineWindow>() {
            gone = window.closing();
        } else {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "present: a kept metal view has an unknown window class; left in its host",
            );
        }
    });
    gone
}

/// Whether the view kept for `hwnd` still sits in the Cocoa window that handle has now.
///
/// A destroyed handle has no window; one a new window inherited answers with
/// that window, which is not the one the kept view sits in.
fn kept_view_hosted(funcs: &MacdrvFuncs, hwnd: u64, view: usize) -> bool {
    let cocoa_window = funcs
        .macdrv_get_cocoa_window
        .map_or(0, |get| cocoa_window_of(get, hwnd));
    if cocoa_window == 0 {
        return false;
    }
    let mut hosted = false;
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("kept view check runs on the main thread");
        hosted = kept_view_is_in_window(view, cocoa_window, mtm);
    });
    hosted
}

/// The Cocoa window `hwnd` has now, `0` for a handle without one. API thread only.
fn cocoa_window_of(get: GetCocoaWindowFn, hwnd: u64) -> usize {
    // SAFETY: extern "C" Wine entry point; `hwnd` is a window handle the PE
    // side supplied, and the second argument says the window need not be on
    // screen.
    unsafe { get(hwnd as *mut c_void, 0) as usize }
}

/// Whether the kept view sits in the Cocoa window at `cocoa_window`. **Main thread only.**
fn kept_view_is_in_window(view: usize, cocoa_window: usize, mtm: MainThreadMarker) -> bool {
    retain_parked_view(view, mtm)
        .and_then(|view| view.window())
        .is_some_and(|window| Retained::as_ptr(&window) as usize == cocoa_window)
}

/// Retain a park-owned view from its address. **Main thread only.**
///
/// The one place a kept view is resurrected from the address the park held:
/// the park holds the retain `macdrv_view_create_metal_view` handed out for
/// the view and nothing releases it before the park's own release, and the
/// caller either owns the taken slot or holds the park lock while retaining
/// a slot it verified is still current. The retain covers the caller's walk.
fn retain_parked_view(view: usize, _mtm: MainThreadMarker) -> Option<Retained<NSView>> {
    // SAFETY: as the doc says; the retain taken here covers the caller's use.
    unsafe { Retained::retain(view as *mut NSView) }
}

/// The metal views retired devices left on their windows, kept for the next device on each.
///
/// Wine creates a client surface, a cocoa view of its own, for every
/// `get_win_data`, and a metal view in it for every
/// `macdrv_view_create_metal_view`, so a device that goes through Wine gets a
/// new `CAMetalLayer` every time. Keeping the retired view and handing it to
/// the next device on that window, without going through Wine at all, is
/// what keeps the layer. The layer matters for Metal's per-layer frame
/// metrics, the GPU time the Metal HUD shows: a `CAMetalLayer` created later
/// in the process reports no GPU time for its frames until it has presented
/// more of them than any layer before it did, which at a game's frame rate
/// is the rest of the session. Replacing the command queue behind an
/// unchanged layer leaves the metric intact; the layer is the identity it
/// follows.
///
/// [`KEPT_METAL_VIEWS`] views are kept, one per window, and parking past
/// that releases the oldest. Parking also drops the layer's drawable pool, so
/// a kept layer holds the one surface it last displayed rather than a pool at
/// the window's size, and hides the view, so its window no longer shows the
/// departed device's last frame ([`prepare_parked_view`]); the next device's
/// attach shows it again. A device attaching to
/// another window leaves a kept view for a device that comes back to its
/// own, as long as that window is there. A kept view whose window is gone
/// (an application that destroys its device window between two devices and
/// creates a new one) is moved into the next window that attaches with none
/// of its own ([`adopt_metal_view`]), so the layer follows the application
/// across that as well.
static PARKED_METAL_VIEW: Mutex<MetalViewPark> = Mutex::new(MetalViewPark::new());

/// Retire a device's metal view: kept for its window's next device, see [`PARKED_METAL_VIEW`].
///
/// `record` is the attachment record the view served, which names the
/// window; a view no record names is released instead. Runs on the API
/// thread; the release of a displaced view hops to the main thread inside
/// Wine.
pub fn retire_metal_view(view_handle: MetalHandle<NSViewKind>, record: Option<&Attachment>) {
    if view_handle.is_null() {
        return;
    }
    let view =
        usize::try_from(view_handle.raw()).expect("a 64-bit host addresses every view pointer");
    let Some(record) = record.filter(|att| att.hwnd() != 0 && att.view() == view) else {
        release_metal_view(view);
        return;
    };
    // Before the park publishes the view, not after: a device that took it
    // back in between would present into the shrunk drawable until its own
    // attach put the layer's geometry back, and the park's hide could land
    // after that attach showed the view.
    prepare_parked_view(view, record.layer());
    let displaced = PARKED_METAL_VIEW
        .lock()
        .expect("metal view park mutex poisoned")
        .park(record.hwnd(), view, record.layer());
    debug!(
        target: LOG_TARGET,
        "present: metal view {view:#x} kept for the next device on window {:#x}, hidden until \
         then",
        record.hwnd(),
    );
    if let Some(displaced) = displaced {
        release_metal_view(displaced);
    }
}

/// Drop the drawable pool of a view about to be parked and hide the view.
///
/// A layer's pool is its window's size, `maximumDrawableCount` deep: at
/// 3456x2234 `RGBA16Float` that is about 62 MB a drawable, held while no
/// device presents through it. `CAMetalLayer` has no call that empties the
/// pool, and neither lowering `maximumDrawableCount` nor releasing the last
/// drawable releases anything of it; writing `drawableSize` does, which is
/// why [`crate::metal::layer::sync_drawable_size`] compares before it writes. So the park writes
/// the smallest drawable Metal takes, and attach restores the layer's own
/// geometry through [`configure_metal_layer`], as it does for a view Wine
/// just created. What survives the write is the surface the layer last
/// displayed.
///
/// The view is hidden ([`hide_parked_view`]), so the window stops showing
/// the departed device's last frame; it stays in its client view and its
/// window, and attach shows it again ([`show_metal_view`]).
///
/// Synchronous, because the caller is about to publish the view to the park:
/// all of this has to land before another device can take the view back.
///
/// The same hop takes the fade out of the order-out of the window the view
/// still sits in ([`order_out_without_fade`]), for a view attach found in no
/// window.
fn prepare_parked_view(view: usize, layer: usize) {
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("a view is parked on the main thread");
        shrink_layer_drawable(layer, mtm);
        order_out_without_fade(view as *mut c_void, mtm);
        hide_parked_view(view, mtm);
    });
}

/// Hide a view about to be parked; its window draws its surface again. **Main thread only.**
///
/// Hidden, the view leaves its client view, which draws nothing of its own,
/// showing Wine's window surface under it rather than the departed device's
/// last frame, and a winemac that keeps a client view shown across window
/// surface flushes while a metal view presents through it treats a hidden one
/// as absent, so the next flush hides the client view as it does any other.
/// The view keeps its place in the client view and the window, so the window
/// still walks it when the retina mode changes, and nothing changes for the
/// layer but its visibility: it keeps its identity and its place in the
/// layer tree, the state the kept layer's per-layer frame metrics were
/// checked in.
///
/// [`clear_covered_window_layers`] may have taken the surface image off the
/// window's content view while the view covered the window, and Wine sets it
/// again only on its next flush. Marking the content view for display has
/// Wine's `updateLayer` put the image back now, so a window between two
/// devices shows its surface rather than nothing.
fn hide_parked_view(view: usize, _mtm: MainThreadMarker) {
    // SAFETY: the view is the retiring device's, still retained by the
    // handle `macdrv_view_create_metal_view` handed out, since the caller has
    // not parked or released it yet; the retain taken here covers the calls.
    let Some(view) = (unsafe { Retained::retain(view as *mut NSView) }) else {
        return;
    };
    view.setHidden(true);
    if let Some(content) = view.window().and_then(|window| window.contentView()) {
        content.setNeedsDisplay(true);
    }
}

/// Show a metal view a device attaches to, which the park hid. **Main thread only.**
///
/// A view Wine just created is shown already, and showing it again changes
/// nothing. Runs before the attach presents the view's client surface, so
/// the client view Wine shows again holds a shown metal view.
fn show_metal_view(view: *mut c_void, _mtm: MainThreadMarker) {
    // SAFETY: the view is the one attach holds, created by Wine just now or
    // kept for this device and handed over by the park, retained either way;
    // the retain taken here covers the call.
    let Some(view) = (unsafe { Retained::retain(view.cast::<NSView>()) }) else {
        return;
    };
    if view.isHidden() {
        view.setHidden(false);
    }
}

/// Write the smallest drawable size `layer` accepts. **Main thread only.**
fn shrink_layer_drawable(layer: usize, _mtm: MainThreadMarker) {
    use objc2_core_foundation::CGSize;
    use objc2_quartz_core::CAMetalLayer;

    /// The smallest drawable Metal accepts; a parked layer vends none of them.
    const PARKED_DRAWABLE_SIZE: CGSize = CGSize {
        width: 1.0,
        height: 1.0,
    };

    // SAFETY: the layer address is the one the retiring device's attachment
    // record held, and Wine retains the `CAMetalLayer` for its metal view's
    // lifetime; the view is still retained here, since the caller has not
    // parked or released it yet, so the address names a live layer and the
    // retain taken here covers the write.
    let Some(layer) = (unsafe { Retained::retain(layer as *mut CAMetalLayer) }) else {
        return;
    };
    layer.setDrawableSize(PARKED_DRAWABLE_SIZE);
}

/// How many retired metal views are kept at once.
///
/// A game has its device window and at most one more that a `Reset`
/// retargets it to, and one that destroys its device window between two
/// devices leaves one kept view per switch, taken by the next window; the
/// end-to-end suite runs four devices at once, and a smaller park would hand
/// its recreations new layers by accident of timing. Each kept view holds
/// the one surface its layer last displayed; the pool behind it goes at the
/// park ([`prepare_parked_view`]).
const KEPT_METAL_VIEWS: usize = 4;

/// One kept metal view: raw addresses and the window it served.
struct KeptMetalView {
    /// The `HWND` the view was attached to, `0` in an empty slot.
    hwnd: u64,
    /// Raw `WineMetalView*`, `0` in an empty slot.
    view: usize,
    /// Raw `CAMetalLayer*` of that view.
    layer: usize,
    /// When it was parked, for choosing the oldest to displace.
    seq: u64,
}

impl KeptMetalView {
    const EMPTY: Self = Self {
        hwnd: 0,
        view: 0,
        layer: 0,
        seq: 0,
    };
}

/// The metal views kept across their devices' teardown, one per window, oldest displaced first.
///
/// Raw addresses and window handles; [`retire_metal_view`] and
/// [`kept_metal_view`] do the retaining and releasing around it.
struct MetalViewPark {
    slots: [KeptMetalView; KEPT_METAL_VIEWS],
    next_seq: u64,
}

impl MetalViewPark {
    const fn new() -> Self {
        Self {
            slots: [KeptMetalView::EMPTY; KEPT_METAL_VIEWS],
            next_seq: 1,
        }
    }

    /// Keep `view` and its `layer` for the next device on `hwnd`, returning the view it displaces.
    ///
    /// A view already kept for `hwnd` is displaced by the new one; with every
    /// slot taken, the oldest kept view is. `None` when nothing is displaced,
    /// which includes `view` being the one kept for `hwnd` already.
    fn park(&mut self, hwnd: u64, view: usize, layer: usize) -> Option<usize> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let kept = KeptMetalView {
            hwnd,
            view,
            layer,
            seq,
        };
        let slot = self
            .slots
            .iter()
            .position(|slot| slot.view != 0 && slot.hwnd == hwnd)
            .or_else(|| self.slots.iter().position(|slot| slot.view == 0))
            .or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, slot)| slot.seq)
                    .map(|(index, _)| index)
            })
            .expect("the park has at least one slot");
        let displaced = self.slots[slot].view;
        self.slots[slot] = kept;
        (displaced != 0 && displaced != view).then_some(displaced)
    }

    /// Take the view and layer kept for a device attaching to `hwnd`, if there is one.
    fn take_for(&mut self, hwnd: u64) -> Option<(usize, usize)> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.view != 0 && slot.hwnd == hwnd)?;
        let kept = (slot.view, slot.layer);
        *slot = KeptMetalView::EMPTY;
        Some(kept)
    }

    /// The kept views, newest first, as copies for a caller that takes one by [`Self::take_kept`].
    ///
    /// Copies rather than the slots, since the caller looks at them with the
    /// park unlocked; what it picks it takes by view and parking, so a slot
    /// that changed under it is not taken.
    fn slots_newest_first(&self) -> Vec<KeptMetalView> {
        let mut kept: Vec<KeptMetalView> = self
            .slots
            .iter()
            .filter(|slot| slot.view != 0)
            .map(|slot| KeptMetalView {
                hwnd: slot.hwnd,
                view: slot.view,
                layer: slot.layer,
                seq: slot.seq,
            })
            .collect();
        kept.sort_by_key(|kept| core::cmp::Reverse(kept.seq));
        kept
    }

    /// Take `view` out of the park if the slot still holds the parking `seq` names.
    ///
    /// A slot displaced and filled again since the caller looked carries a
    /// later `seq`, and its view is then someone else's: `None`, and the
    /// caller has nothing.
    fn take_kept(&mut self, view: usize, seq: u64) -> Option<KeptMetalView> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.view != 0 && slot.view == view && slot.seq == seq)?;
        Some(core::mem::replace(slot, KeptMetalView::EMPTY))
    }
}

/// Release a metal view through Wine, which removes it from its window on the main thread.
fn release_metal_view(view: usize) {
    // A process whose Wine publishes no usable table never created a view
    // through one either, and the load has already said what was wrong with
    // it. The release entry itself is one of the entries that load checks,
    // so the pointer below is a function.
    let Some(funcs) = MacdrvFuncs::load() else {
        return;
    };
    // SAFETY: extern "C" Wine entry point; takes the view pointer by
    // value, and `view` is the address `macdrv_view_create_metal_view`
    // handed out, still retained because nothing released it before.
    unsafe { (funcs.macdrv_view_release_metal_view)(view as *mut c_void) };
}

/// The window data the table's `get_win_data` hands back, as far as we read it.
///
/// This is the record that entry is defined to return, whose third and
/// fourth fields are the window's own Cocoa view and the client view the
/// call creates. It is not winemac's internal `struct macdrv_win_data`,
/// which has the client view third and a rect where the fourth field is
/// here, and which no Wine exports a way to reach.
#[repr(C)]
struct MacdrvWinData {
    hwnd: *mut c_void,
    cocoa_window: *mut c_void,
    cocoa_view: *mut c_void,
    client_cocoa_view: *mut c_void,
}

/// Subset of macdrv function table entries.
///
/// Matching the struct field order in Wine's `macdrv_functions_t`.
#[repr(C)]
struct MacdrvFunctionsTable {
    macdrv_init_display_devices: *mut c_void,
    get_win_data: *mut c_void,
    release_win_data: *mut c_void,
    macdrv_get_cocoa_window: *mut c_void,
    macdrv_create_metal_device: *mut c_void,
    macdrv_release_metal_device: *mut c_void,
    macdrv_view_create_metal_view: *mut c_void,
    macdrv_view_get_metal_layer: *mut c_void,
    macdrv_view_release_metal_view: *mut c_void,
    on_main_thread: *mut c_void,
}

struct MacdrvFuncs {
    get_win_data: GetWinDataFn,
    release_win_data: ReleaseWinDataFn,
    macdrv_view_create_metal_view: CreateMetalViewFn,
    macdrv_view_get_metal_layer: GetMetalLayerFn,
    macdrv_view_release_metal_view: ReleaseMetalViewFn,
    /// `macdrv_get_cocoa_window`, `None` when the table's entry is null.
    ///
    /// Answers which Cocoa window an `HWND` has right now, without the client
    /// surface `get_win_data` creates, and none for a destroyed handle: a
    /// kept view is reused inside that window, and moved to another handle's
    /// window only once its own has none. Called on the API thread only,
    /// since Wine holds its window-data lock across a synchronous main-thread
    /// request while it destroys a window.
    macdrv_get_cocoa_window: Option<GetCocoaWindowFn>,
}

/// The `macdrv_functions` table, or `None` on a Wine that publishes none.
///
/// The table is the whole of the layer's access to winemac: it is the only
/// macdrv symbol the driver's unix library exports, and its `get_win_data`
/// is the only one that answers with a [`MacdrvWinData`]. A Wine without it
/// gives no window a Metal layer, which is what the warning here says.
fn macdrv_functions() -> Option<&'static MacdrvFunctionsTable> {
    // SAFETY: `macdrv_functions` is a Wine-published process-lifetime
    // static; `Symbol<*const T>` derefs to the loaded pointer value.
    let Ok(table_sym) =
        (unsafe { MACDRV_LIB.get::<*const MacdrvFunctionsTable>(b"macdrv_functions\0") })
    else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: this Wine publishes no macdrv_functions table; no window can be given a \
             Metal layer",
        );
        return None;
    };
    // SAFETY: Wine guarantees the address is non-null and the table outlives
    // the process.
    Some(unsafe { &**table_sym })
}

/// The first entry the layer calls that `table` leaves null, by name.
///
/// A null entry among the ones the layer calls says the table is not the
/// one this layer was written against, and nothing else in it can be
/// trusted to be either, so the caller gives up rather than call four of
/// the five. `macdrv_get_cocoa_window` is not among them: it answers which
/// Cocoa window an `HWND` has now, and without that answer a kept view goes
/// back into a client surface of its own handle's window and is never moved
/// to another handle's. `None` when every entry the layer calls is filled.
fn first_null_required_entry(table: &MacdrvFunctionsTable) -> Option<&'static str> {
    [
        ("get_win_data", table.get_win_data),
        ("release_win_data", table.release_win_data),
        (
            "macdrv_view_create_metal_view",
            table.macdrv_view_create_metal_view,
        ),
        (
            "macdrv_view_get_metal_layer",
            table.macdrv_view_get_metal_layer,
        ),
        (
            "macdrv_view_release_metal_view",
            table.macdrv_view_release_metal_view,
        ),
    ]
    .into_iter()
    .find_map(|(name, entry)| entry.is_null().then_some(name))
}

impl MacdrvFuncs {
    /// The table's entries as typed function pointers, or `None` with the reason logged.
    ///
    /// This is where the table is checked, and the only place: every entry
    /// the layer calls is read as a pointer and refused if it is null, so
    /// what the rest of the file holds is typed pointers no caller can find
    /// missing. `macdrv_get_cocoa_window` is the one entry a table may leave
    /// null, and it alone stays an `Option`.
    fn load() -> Option<Self> {
        let table = macdrv_functions()?;
        if let Some(entry) = first_null_required_entry(table) {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "present: this Wine's macdrv_functions table leaves {entry} null; no window can \
                 be given a Metal layer",
            );
            return None;
        }
        Some(Self {
            // SAFETY: table entry is a fn pointer stored as `*mut c_void`
            // per Wine's C ABI, non-null per the check above; transmute
            // reinterprets to the typed fn.
            get_win_data: unsafe {
                core::mem::transmute::<*mut c_void, GetWinDataFn>(table.get_win_data)
            },
            // SAFETY: as above.
            release_win_data: unsafe {
                core::mem::transmute::<*mut c_void, ReleaseWinDataFn>(table.release_win_data)
            },
            // SAFETY: as above.
            macdrv_view_create_metal_view: unsafe {
                core::mem::transmute::<*mut c_void, CreateMetalViewFn>(
                    table.macdrv_view_create_metal_view,
                )
            },
            // SAFETY: as above.
            macdrv_view_get_metal_layer: unsafe {
                core::mem::transmute::<*mut c_void, GetMetalLayerFn>(
                    table.macdrv_view_get_metal_layer,
                )
            },
            // SAFETY: as above.
            macdrv_view_release_metal_view: unsafe {
                core::mem::transmute::<*mut c_void, ReleaseMetalViewFn>(
                    table.macdrv_view_release_metal_view,
                )
            },
            // SAFETY: as above; this entry may be null, and a null one
            // reads as `None`.
            macdrv_get_cocoa_window: unsafe {
                core::mem::transmute::<*mut c_void, Option<GetCocoaWindowFn>>(
                    table.macdrv_get_cocoa_window,
                )
            },
        })
    }
}

/// Whether the screen-parameter filter is in place. **Main thread only.**
///
/// Set by the attempt that found a delegate and installed the handler, and
/// only by that one, whether it took the notification over from the delegate
/// or left a delegate without `applicationDidChangeScreenParameters:` alone.
/// An attempt that runs before `NSApp` has a delegate installs nothing, so it
/// leaves this clear and the next attach or headroom refresh tries again.
/// Read and written on the main thread alone, which is what `Relaxed` rests on.
static SCREEN_PARAMS_FILTER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// What an attempt to install the screen-parameter filter does.
#[derive(Debug, PartialEq, Eq)]
enum ScreenParamsFilterStep {
    /// The notification is already ours, so the attempt is a no-op.
    AlreadyOurs,
    /// `NSApp` has no delegate to take the notification over from yet.
    AwaitDelegate,
    /// Observe the name ourselves, unregistering a delegate that implements the method.
    TakeOver,
}

/// Decide what an install attempt does from the state it found.
///
/// Only a take-over marks the filter installed. Wine installs its application
/// delegate while it brings the application up, which can land after the first
/// `CreateDevice`, so an attempt that finds none has to leave the decision
/// open: marking it done there would spend the process's one attempt on a
/// delegate that does not exist yet and leave the storm unfiltered for the
/// rest of the run.
const fn screen_params_filter_step(installed: bool, has_delegate: bool) -> ScreenParamsFilterStep {
    match (installed, has_delegate) {
        (true, _) => ScreenParamsFilterStep::AlreadyOurs,
        (false, false) => ScreenParamsFilterStep::AwaitDelegate,
        (false, true) => ScreenParamsFilterStep::TakeOver,
    }
}

/// Take over `NSApplicationDidChangeScreenParametersNotification` from Wine. **Main thread only.**
///
/// macOS posts this notification not only for display topology or mode
/// changes but for every step of an EDR headroom ramp, and the headroom
/// follows ambient light, thermal state and on-screen content, so with the
/// HDR layer attached it arrives at up to the refresh rate. Wine's macdrv
/// answers each one as a display-mode change: a window-level pass here, a
/// full display re-enumeration in the desktop process and a display-cache
/// invalidation everywhere, all of it on the main thread that `SetCapture`
/// and `SetCursorPos` wait on synchronously. That wait is what made every
/// mouse press and release frame run 1 to 3 ms long.
///
/// `AppKit` wires the notification to the delegate's
/// `applicationDidChangeScreenParameters:` through the default notification
/// center, so unregistering the delegate for this one name and observing it
/// ourselves lets us forward only the deliveries whose
/// [`ScreenConfiguration`] differs from the last one forwarded. Everything
/// Wine does in its handler still happens on real changes, and nothing at
/// all happens on a headroom step. The desktop process never loads mtld3d,
/// so its own copy of the storm is out of reach here; that half is a Wine
/// patch. Attempted from every attach and every headroom refresh until one
/// lands, because `NSApp` gains its delegate when Wine finishes bringing the
/// application up and that can be after the first `CreateDevice`; an attempt
/// with no delegate to take the notification over from installs nothing and
/// leaves the next one to try again. The observer token is leaked for the
/// process lifetime like the occlusion observer, and it has to outlive any one
/// device: taking the notification over unregisters Wine's delegate for this
/// name, so removing our observer at teardown would leave nobody forwarding
/// it at all. What teardown retires instead is the attachment record, and
/// the handler walks only the records that are still live.
///
/// Whether the delegate implements `applicationDidChangeScreenParameters:`
/// is checked once, here, on the delegate being taken over from, since that
/// is the one object the handler ever forwards to. `AppKit` subscribes a
/// delegate to the notification only when it implements the method, so a
/// delegate without it is left alone and nothing is forwarded; the handler is
/// installed all the same and still reconciles every layer on a real change.
/// A delegate installed later is subscribed by `AppKit` itself and receives
/// the notification unfiltered, which a check per notification would not
/// change either.
pub fn install_screen_params_filter(mtm: objc2::MainThreadMarker) {
    use core::ptr::NonNull;

    use block2::RcBlock;
    use objc2_app_kit::{
        NSApplication, NSApplicationDidChangeScreenParametersNotification, NSScreen,
    };
    use objc2_foundation::{NSNotification, NSNotificationCenter};

    let installed = SCREEN_PARAMS_FILTER_INSTALLED.load(Ordering::Relaxed);
    // The delegate lookup is skipped once the filter is in place, so every
    // attach and every headroom refresh after that costs one relaxed load.
    let delegate = if installed {
        None
    } else {
        NSApplication::sharedApplication(mtm).delegate()
    };
    match screen_params_filter_step(installed, delegate.is_some()) {
        ScreenParamsFilterStep::AlreadyOurs => return,
        ScreenParamsFilterStep::AwaitDelegate => {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "present: NSApp has no delegate yet; screen-parameter notifications are not \
                 filtered and every EDR headroom step costs a Wine display re-enumeration \
                 until an attach or a headroom refresh finds one",
            );
            return;
        }
        ScreenParamsFilterStep::TakeOver => {}
    }
    let delegate = delegate.expect("TakeOver is reached only when the delegate is present");
    let center = NSNotificationCenter::defaultCenter();
    // SAFETY: AppKit-exported notification-name constant, valid for the
    // process lifetime.
    let name = unsafe { NSApplicationDidChangeScreenParametersNotification };
    // SAFETY: `ProtocolObject` is a transparent wrapper over `AnyObject`,
    // so the pointer reinterprets losslessly for the call below.
    let observer = unsafe { &*Retained::as_ptr(&delegate).cast::<objc2::runtime::AnyObject>() };
    let forwards = delegate_forward::delegate_handles_screen_parameters(observer);
    if forwards {
        // SAFETY: objc2 typed binding; the delegate is a live observer of the
        // center (AppKit registered it), and removing a registration that
        // does not exist is a documented no-op.
        unsafe { center.removeObserver_name_object(observer, Some(name), None) };
        WINE_APP_DELEGATE_PTR.store(Retained::as_ptr(&delegate) as usize, Ordering::Release);
        // Leaked on purpose: the delegate is Wine's application controller
        // and lives as long as the process.
        core::mem::forget(delegate);
    } else {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: Wine's application delegate does not implement \
             applicationDidChangeScreenParameters:, so AppKit delivers it no screen-parameter \
             notification and there is nothing to filter or forward; the notification is \
             observed only to reconcile the layers on a real change",
        );
    }

    let block = RcBlock::new(move |notification: NonNull<NSNotification>| {
        autoreleasepool(|_| {
            let count = SCREEN_PARAM_CHANGES.fetch_add(1, Ordering::Relaxed) + 1;
            let mtm = MainThreadMarker::new()
                .expect("the screen-parameter notification is posted on the main thread");
            let changed = screen_configuration_changed(mtm);
            if log_enabled!(target: LOG_TARGET, log::Level::Debug) {
                let headroom = NSScreen::mainScreen(mtm)
                    .map_or(0.0, |s| s.maximumExtendedDynamicRangeColorComponentValue());
                debug!(
                    target: LOG_TARGET,
                    "screen params changed #{count}: headroom={headroom:.3} {}",
                    if changed { "configuration changed" } else { "unchanged, filtered" },
                );
            }
            if !changed {
                return;
            }
            // A real topology or mode change is the moment a display was
            // attached, removed or reconfigured, so reconcile every layer now
            // rather than waiting out the present-counted poll interval.
            refresh_all_on_main();
            let delegate_ptr = WINE_APP_DELEGATE_PTR.load(Ordering::Acquire);
            if delegate_ptr == 0 {
                return;
            }
            // SAFETY: the pointer was taken from a `Retained` that is leaked
            // above, so the delegate outlives this block, and `ProtocolObject`
            // is a transparent wrapper over `AnyObject`.
            let delegate = unsafe { &*(delegate_ptr as *const objc2::runtime::AnyObject) };
            // SAFETY: the notification pointer is valid for the handler's duration.
            let notification = unsafe { notification.as_ref() };
            // SAFETY: the block runs on the main thread, which is the
            // delegate's thread, and `notification` is the one being delivered.
            let forwarded =
                unsafe { delegate_forward::forward_screen_parameters(delegate, notification) };
            if let Err(thrown) = forwarded {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "present: Wine's applicationDidChangeScreenParameters: threw {thrown}; \
                     caught so the process survives, the rest of Wine's handler did not run",
                );
                debug!(target: LOG_TARGET, "screen params #{count}: Wine's handler threw {thrown}");
            }
        });
    });
    // SAFETY: objc2 typed binding; the center copies the block, and the
    // token is leaked below so the observer is never removed.
    let token = unsafe {
        center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
    };
    core::mem::forget(token);
    SCREEN_PARAMS_FILTER_INSTALLED.store(true, Ordering::Relaxed);
    if forwards {
        info!(
            target: LOG_TARGET,
            "present: filtering NSApplicationDidChangeScreenParametersNotification for Wine \
             (forwarded only when screen geometry, scale or the main display mode changed)",
        );
    }
}

/// Clear what the window's layers paint under a metal view that covers it. **Main thread only.**
///
/// Two layers of a winemac window paint beneath the metal view. The frame
/// view's layer carries the window's background colour, which `AppKit` puts
/// there because Wine's content view is not opaque, and the content view's
/// layer holds Wine's window surface image as its `contents`, set again on
/// every surface flush. Either one can keep the game's layer in the
/// compositor instead of going direct to the display, as seen on an M1 Max
/// (an M5 Max goes direct with both in place). When the metal view is shown
/// and its visible rectangle covers the frame view to within a sixty-fourth
/// of a point, far below a device pixel, both are entirely hidden under its
/// opaque layer, so clearing them changes no pixel on screen. A window the
/// view does not cover shows them and keeps them, and so does a window whose
/// metal view is hidden, as Wine hides the client view when another client
/// surface takes over or, on some Wines, on every surface flush: the surface
/// image is then all the window shows. Nothing is restored here: Wine sets
/// the surface image again on its next flush, which a resize makes, and the
/// background colour on the window's next show, and the next refresh clears
/// whichever property `AppKit` or Wine has set again while the cover holds.
/// The park, which hides the view, has the content view draw its surface
/// image again at once (the Wine host's `hide_parked_view`); the image covers what the
/// view covered, so the background colour stays cleared.
pub fn clear_covered_window_layers(view: &NSView, _mtm: MainThreadMarker) {
    // A view between windows has none; the next pass finds the new one.
    let Some(window) = view.window() else {
        return;
    };
    // A hidden view leaves the surface image as all the window shows, so it
    // stays; the next pass after Wine shows the view again clears it.
    if view.isHiddenOrHasHiddenAncestor() {
        return;
    }
    let content = window.contentView();
    // SAFETY: objc2 typed binding; `superview` is unsafe only because the
    // view hierarchy may change under a caller off the main thread, and the
    // marker puts this read on the main thread, where that hierarchy changes.
    let frame_view = content
        .as_deref()
        .and_then(|content| unsafe { content.superview() });
    let (Some(content), Some(frame_view)) = (content, frame_view) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: the metal view's window has no content view or frame view; what the \
             window paints under the game's layer is left as it is",
        );
        return;
    };
    let covered = rect_covers(
        view.convertRect_toView(view.visibleRect(), None),
        frame_view.convertRect_toView(frame_view.bounds(), None),
    );
    if !covered {
        return;
    }
    let (Some(content_layer), Some(frame_layer)) = (content.layer(), frame_view.layer()) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: the metal view's window has a content or frame view without a layer; what \
             the window paints under the game's layer is left as it is",
        );
        return;
    };
    if frame_layer.backgroundColor().is_some() {
        frame_layer.setBackgroundColor(None);
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: the metal view covers its window; cleared the frame view's background \
             colour beneath it so the game's layer can go direct to the display",
        );
    }
    // SAFETY: objc2 typed binding; `contents` is unsafe only because the
    // object it returns is untyped, and it is only tested for presence here.
    if unsafe { content_layer.contents() }.is_some() {
        // SAFETY: `nil` is a valid `contents` for any layer and means no image.
        unsafe { content_layer.setContents(None) };
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: the metal view covers its window; cleared the content view's surface \
             image beneath it so the game's layer can go direct to the display",
        );
    }
}

/// Whether `inner` covers `outer`, both in one coordinate space, to 1/64 point on every edge.
///
/// Wine's view frames come from integer Win32 rectangles, at most halved,
/// so the conversion between view spaces is exact in `f64`; the tolerance
/// only absorbs float noise and stays far below a device pixel, so no
/// visible strip of the window counts as covered.
fn rect_covers(inner: CGRect, outer: CGRect) -> bool {
    const SLACK: f64 = 1.0 / 64.0;
    let (inner_min, inner_max) = (inner.min(), inner.max());
    let (outer_min, outer_max) = (outer.min(), outer.max());
    inner_min.x <= outer_min.x + SLACK
        && inner_min.y <= outer_min.y + SLACK
        && inner_max.x >= outer_max.x - SLACK
        && inner_max.y >= outer_max.y - SLACK
}

#[cfg(test)]
#[path = "wine/tests.rs"]
mod tests;
