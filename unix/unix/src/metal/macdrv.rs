use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::sync::{Arc, LazyLock, Mutex};

use libloading::os::unix::Library;
use log::{debug, error, info, log_enabled};
use mtld3d_shared::{
    MetalHandle,
    mtl::{ColorSpacePolicy, SoftwareCursorPolicy},
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, NSViewKind},
};
use objc2::{
    MainThreadMarker, MainThreadOnly, extern_class, extern_methods,
    rc::{Retained, autoreleasepool},
    runtime::{AnyClass, NSObjectProtocol, ProtocolObject},
};
use objc2_app_kit::{NSView, NSWindow};
use objc2_core_graphics::{CGColor, CGColorSpace};

use crate::{LOG_TARGET, metal::handle::IntoRetained};

pub mod attachment;
mod cursor_overlay;

use attachment::{AttachFlags, AttachLatches, Attachment};
pub use cursor_overlay::{poll_capture_from_present, set_cursor_overlay};

/// Retire the attachment record `view_handle` names. **Device teardown only.**
///
/// The teardown path then retires that metal view, keeping it for the next
/// device on that window or on one that replaces it, or releasing it
/// ([`retire_metal_view`]), so the view,
/// its layer and its window may not be reached through the record from the
/// moment this runs. Unregistering the record first is what keeps the
/// process-lifetime screen-parameter, occlusion and pointer observers from
/// walking a view that is about to be released: every dereference they make
/// goes through the registry, which either retains the object while the
/// record is live or finds it gone. Everything the display
/// decided for that window goes with the record, and the PE-side sinks it
/// published into are never written again, since the device is about to drop
/// them.
///
/// Only the record of this view is retired, and it is handed back so the
/// view's retirement knows the window it served. A device that never
/// attached finds no record, and another device's record is untouched.
pub fn detach_metal_layer(view_handle: MetalHandle<NSViewKind>) -> Option<Arc<Attachment>> {
    let view_addr =
        usize::try_from(view_handle.raw()).expect("a 64-bit host addresses every view pointer");
    let att = attachment::unregister(view_addr)?;
    cursor_overlay::detach(&att);
    debug!(
        target: LOG_TARGET,
        "present: detached view {:#x} (layer {:#x}); its display state is retired",
        att.view(),
        att.layer(),
    );
    Some(att)
}

/// Tell macOS this process is doing continuous, latency-critical, user-interactive work.
///
/// A game, rather than idle UI work — so it stays out of App Nap and the
/// timer/display-update throttling that can let the compositor stop cycling
/// an otherwise-visible `CAMetalLayer`, i.e. the "the window is visible but
/// nothing reaches the screen" stall.
///
/// `UserInteractive` = `UserInitiated | LatencyCritical`: the strongest
/// "real-time foreground app" declaration. `IdleDisplaySleepDisabled`
/// keeps the panel awake during play so the screen never dims mid-scene.
/// The `NSProcessInfo` activity must outlive every present, so the token
/// is intentionally leaked (dropping it calls `endActivity` and throttling
/// resumes; the OS reclaims it at process exit). Called from the library
/// init thunk, which is *not* guaranteed to fire exactly once (its sibling
/// `init_logger` relies on `env_logger`'s idempotent `try_init` for the same
/// reason); the function-scoped `Once` latches the begin so repeat init
/// calls don't each leak another activity.
pub fn declare_latency_critical_activity() {
    use std::sync::Once;

    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let options =
            NSActivityOptions::UserInteractive | NSActivityOptions::IdleDisplaySleepDisabled;
        let reason = NSString::from_str("mtld3d: continuous latency-critical game rendering");
        let token: Retained<ProtocolObject<dyn NSObjectProtocol>> =
            NSProcessInfo::processInfo().beginActivityWithOptions_reason(options, &reason);
        core::mem::forget(token);
        info!(
            target: LOG_TARGET,
            "present: declared NSActivityUserInteractive (latency-critical; no App Nap / idle throttling / display sleep) for continuous rendering",
        );
    });
}

/// Begin tracking the record's window occlusion so presents can skip `nextDrawable`.
///
/// `submit_frame` skips the present while the window is fully
/// covered/minimised. When a window is occluded the compositor stops
/// recycling its drawables, so `nextDrawable` would block its full
/// `allowsNextDrawableTimeout` for nothing on screen and back-pressure the
/// whole pipeline up to the guest's render loop. Records the window pointer
/// on `att` and seeds its occlusion from the current state, then installs
/// the (process-lifetime) observer once. Runs the `AppKit` work on the main
/// thread: `NSView`/`NSWindow` access and the notification center are
/// main-thread affairs, mirroring [`configure_metal_layer`]'s posture. The
/// record is registered before this runs, so the observer can already find
/// it.
fn install_occlusion_tracking(att: &Arc<Attachment>) {
    use objc2_app_kit::NSWindowOcclusionState;

    let att = Arc::clone(att);
    run_on_main_thread_sync(move || {
        let mtm =
            MainThreadMarker::new().expect("install_occlusion_tracking runs on the main thread");
        let window = attachment::retain_view(&att, mtm).and_then(|view| view.window());
        let Some(window) = window else {
            // No host window yet: assume visible so a present is never
            // wrongly suppressed; the observer corrects it on the first
            // state change.
            att.set_window(0);
            att.set_window_occluded(false);
            return;
        };
        att.set_window(Retained::as_ptr(&window) as usize);
        let occluded = !window
            .occlusionState()
            .contains(NSWindowOcclusionState::Visible);
        att.set_window_occluded(occluded);
        install_occlusion_observer_once();
        install_screen_params_filter(mtm);
        cursor_overlay::install_pointer_watch(mtm);
    });
}

/// Count of `NSApplicationDidChangeScreenParametersNotification` deliveries.
static SCREEN_PARAM_CHANGES: AtomicU64 = AtomicU64::new(0);

/// Wine's `NSApplication` delegate, retained for the process lifetime.
///
/// Set once by [`install_screen_params_filter`]; zero until then. Held
/// as an address because the observer block that reads it must not capture
/// a `!Send` `Retained`, and the delegate is only ever touched on the main
/// thread, where the notification is posted.
static WINE_APP_DELEGATE_PTR: AtomicUsize = AtomicUsize::new(0);

/// One screen's contribution to the configuration snapshot.
///
/// Floats are kept as bit patterns so the comparison is exact and the
/// struct stays `Eq`.
#[derive(PartialEq, Eq)]
struct ScreenEntry {
    frame: [u64; 4],
    visible_frame: [u64; 4],
    scale: u64,
}

/// The screen configuration Wine's view of the displays depends on.
///
/// The per-screen geometry and scale, plus the main display's CG mode
/// (size, refresh rate, IO mode id), which is what macdrv reports through
/// `EnumDisplaySettings`. A refresh-rate-only change on a secondary display
/// is the one real change this misses; it is picked up on the next change
/// that moves any geometry.
#[derive(PartialEq, Eq)]
struct ScreenConfiguration {
    screens: Vec<ScreenEntry>,
    main_mode: (usize, usize, u64, i32),
}

/// Snapshot the current screen configuration. **Main thread only.**
fn current_screen_configuration(mtm: objc2::MainThreadMarker) -> ScreenConfiguration {
    use objc2_app_kit::NSScreen;
    use objc2_core_graphics::{CGDisplayCopyDisplayMode, CGDisplayMode, CGMainDisplayID};

    let rect_bits = |r: objc2_foundation::NSRect| {
        [
            r.origin.x.to_bits(),
            r.origin.y.to_bits(),
            r.size.width.to_bits(),
            r.size.height.to_bits(),
        ]
    };
    let screens = NSScreen::screens(mtm)
        .iter()
        .map(|s| ScreenEntry {
            frame: rect_bits(s.frame()),
            visible_frame: rect_bits(s.visibleFrame()),
            scale: s.backingScaleFactor().to_bits(),
        })
        .collect();
    let main_mode = CGDisplayCopyDisplayMode(CGMainDisplayID()).map_or((0, 0, 0, 0), |m| {
        (
            CGDisplayMode::width(Some(&m)),
            CGDisplayMode::height(Some(&m)),
            CGDisplayMode::refresh_rate(Some(&m)).to_bits(),
            CGDisplayMode::io_display_mode_id(Some(&m)),
        )
    });
    ScreenConfiguration { screens, main_mode }
}

/// Last configuration forwarded to Wine. **Main thread only.**
static LAST_SCREEN_CONFIGURATION: Mutex<Option<ScreenConfiguration>> = Mutex::new(None);

/// Whether the screen-parameter filter is in place. **Main thread only.**
///
/// Set by the attempt that took the notification over, and only by that one.
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
    /// Unregister Wine's delegate for the name and observe it ourselves.
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
fn install_screen_params_filter(mtm: objc2::MainThreadMarker) {
    use core::ptr::NonNull;

    use block2::RcBlock;
    use objc2::runtime::ProtocolObject;
    use objc2_app_kit::{
        NSApplication, NSApplicationDelegate, NSApplicationDidChangeScreenParametersNotification,
        NSScreen,
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
    // SAFETY: objc2 typed binding; the delegate is a live observer of the
    // center (AppKit registered it), and removing a registration that
    // does not exist is a documented no-op.
    unsafe { center.removeObserver_name_object(observer, Some(name), None) };
    WINE_APP_DELEGATE_PTR.store(Retained::as_ptr(&delegate) as usize, Ordering::Release);
    // Leaked on purpose: the delegate is Wine's application controller
    // and lives as long as the process.
    core::mem::forget(delegate);

    let block = RcBlock::new(move |notification: NonNull<NSNotification>| {
        autoreleasepool(|_| {
            let count = SCREEN_PARAM_CHANGES.fetch_add(1, Ordering::Relaxed) + 1;
            let mtm = MainThreadMarker::new()
                .expect("the screen-parameter notification is posted on the main thread");
            let config = current_screen_configuration(mtm);
            let changed = {
                let mut last = LAST_SCREEN_CONFIGURATION
                    .lock()
                    .expect("screen-configuration mutex poisoned");
                let changed = last.as_ref() != Some(&config);
                if changed {
                    *last = Some(config);
                }
                changed
            };
            if log_enabled!(target: LOG_TARGET, log::Level::Debug) {
                let headroom = NSScreen::mainScreen(mtm)
                    .map_or(0.0, |s| s.maximumExtendedDynamicRangeColorComponentValue());
                debug!(
                    target: LOG_TARGET,
                    "screen params changed #{count}: headroom={headroom:.3} {}",
                    if changed { "configuration changed, forwarded to Wine" } else { "filtered" },
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
            // above, so the delegate outlives this block, and the call is on
            // the main thread, which is the delegate's thread.
            let delegate =
                unsafe { &*(delegate_ptr as *const ProtocolObject<dyn NSApplicationDelegate>) };
            // SAFETY: the notification pointer is valid for the handler's
            // duration; Wine implements this optional delegate method.
            unsafe { delegate.applicationDidChangeScreenParameters(notification.as_ref()) };
        });
    });
    // SAFETY: objc2 typed binding; the center copies the block, and the
    // token is leaked below so the observer is never removed.
    let token = unsafe {
        center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
    };
    core::mem::forget(token);
    SCREEN_PARAMS_FILTER_INSTALLED.store(true, Ordering::Relaxed);
    info!(
        target: LOG_TARGET,
        "present: filtering NSApplicationDidChangeScreenParametersNotification for Wine \
         (forwarded only when screen geometry, scale or the main display mode changed)",
    );
}

/// Install the `NSWindowDidChangeOcclusionState` observer exactly once.
///
/// Scoped to all windows (`object: None`) and matched in the block against
/// the window of every live attachment record, so a single leaked observer
/// survives device and window churn, and two views on one `NSWindow` both
/// follow it. The token is intentionally leaked for the process lifetime,
/// the same posture as the `NSProcessInfo` activity in
/// [`declare_latency_critical_activity`].
fn install_occlusion_observer_once() {
    use core::ptr::NonNull;
    use std::sync::Once;

    use block2::RcBlock;
    use objc2_app_kit::{NSWindowDidChangeOcclusionStateNotification, NSWindowOcclusionState};
    use objc2_foundation::{NSNotification, NSNotificationCenter};

    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        // The block captures no non-`'static` / non-`Send` state — it reads
        // only module statics plus the live notification object — satisfying
        // `addObserverForName:object:queue:usingBlock:`'s sendable-block
        // contract.
        let block = RcBlock::new(move |notification: NonNull<NSNotification>| {
            autoreleasepool(|_| {
                // SAFETY: AppKit hands a valid `NSNotification` for the call.
                let notification = unsafe { notification.as_ref() };
                let Some(object) = notification.object() else {
                    return;
                };
                let object_ptr = Retained::as_ptr(&object) as usize;
                let ours: Vec<Arc<Attachment>> = attachment::live()
                    .into_iter()
                    .filter(|att| object_ptr != 0 && att.window() == object_ptr)
                    .collect();
                if ours.is_empty() {
                    return;
                }
                // SAFETY: `object` is the live window that posted the notification;
                // its pointer matches a window an attach found, so it is one of
                // our `NSWindow`s, and it stays retained for this call. Occlusion
                // notifications are delivered on the main thread, where the
                // `occlusionState` read is valid.
                let window = unsafe { &*(object_ptr as *const NSWindow) };
                let occluded = !window.occlusionState().contains(NSWindowOcclusionState::Visible);
                for att in &ours {
                    att.set_window_occluded(occluded);
                }
            });
        });

        let center = NSNotificationCenter::defaultCenter();
        // SAFETY: AppKit-exported notification-name constant.
        let name = unsafe { NSWindowDidChangeOcclusionStateNotification };
        // SAFETY: `name` is a valid notification name; `object: None` observes
        // all windows (filtered by the bound-window check in the block); `queue:
        // None` delivers synchronously on the posting (main) thread; the block
        // captures no non-`Send` state. The returned token is leaked below.
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(Some(name), None, None, &block)
        };
        core::mem::forget(token);
        info!(
            target: LOG_TARGET,
            "present: installed NSWindowDidChangeOcclusionState observer (occluded presents skip nextDrawable)",
        );
    });
}

/// Present-throttle request resolved PE-side.
///
/// The guest's vsync ask (`D3DPRESENT_PARAMETERS::PresentationInterval`
/// mapped through `display_sync_for`) plus the user's `present.maxFps`
/// ceiling from `mtld3d.conf` (`0` = uncapped). Bundled so the attach/Reset
/// entry points stay inside clippy's `too_many_arguments` threshold.
pub struct PresentPacing {
    /// `true` for DEFAULT/ONE presentation intervals, `false` for IMMEDIATE.
    ///
    /// Caps presents at the panel ceiling when set.
    pub vsync_requested: bool,
    /// User frame-rate ceiling in Hz; `0` = uncapped.
    ///
    /// When both this and vsync are active the lower rate wins.
    pub max_fps: u32,
}

/// Derive the present-throttle duration from the panel ceiling and the PE-side pacing request.
///
/// A vsync request (DEFAULT/ONE) contributes `1 / panel_max_hz`, capping
/// presents at the panel ceiling; on `ProMotion` the system fills the gap
/// with adaptive cadence below that. A non-zero `max_fps` contributes
/// `1 / max_fps` regardless of the vsync state. The throttle takes the
/// longer of the two durations, so the lower frame rate always wins; when
/// neither contributes (IMMEDIATE + uncapped, or a zero / unknown
/// `panel_max_hz` with no user cap) the result is `0.0` for unthrottled
/// free-run.
fn min_present_duration(panel_max_hz: f64, pacing: &PresentPacing) -> f64 {
    let vsync_duration = if pacing.vsync_requested && panel_max_hz > 0.0 {
        1.0 / panel_max_hz
    } else {
        0.0
    };
    let cap_duration = if pacing.max_fps > 0 {
        1.0 / f64::from(pacing.max_fps)
    } else {
        0.0
    };
    vsync_duration.max(cap_duration)
}

/// Fold a [`PresentPacing`] into the one word an attachment record holds for it.
///
/// The vsync request is the low bit and the frame cap rides above it, so the
/// pair is written and read as a unit.
fn pack_pacing(pacing: &PresentPacing) -> u64 {
    (u64::from(pacing.max_fps) << 1) | u64::from(pacing.vsync_requested)
}

/// Read back what [`pack_pacing`] wrote.
fn unpack_pacing(bits: u64) -> PresentPacing {
    let max_fps = u32::try_from((bits >> 1) & u64::from(u32::MAX))
        .expect("masked to u32::MAX on the line above");
    PresentPacing {
        vsync_requested: bits & 1 != 0,
        max_fps,
    }
}

/// The present-throttle duration to apply when the panel under the window changed.
///
/// `Some(seconds)` when what [`min_present_duration`] derives differs from
/// the duration the present site is using, `None` while the two agree, which
/// is every poll of a session that stays on one display. The comparison is
/// bit-exact because both sides come out of the same derivation, so equal
/// inputs give an identical pattern and only a real change moves it.
fn min_present_duration_change(
    applied_seconds: f64,
    panel_max_hz: f64,
    pacing: &PresentPacing,
) -> Option<f64> {
    let target = min_present_duration(panel_max_hz, pacing);
    (target.to_bits() != applied_seconds.to_bits()).then_some(target)
}

/// Round and clamp the Wine layer's `contentsScale` into the range the PE side takes.
///
/// winemac sets the metal layer's `contentsScale` to 2 in retina mode and 1
/// otherwise, whatever the display's own factor: that is the number of Wine
/// pixels per point, and so how much smaller than a point the game's cursor
/// pixels come out. The clamp bounds the HCURSOR upscaler downstream, which
/// asserts `[1, 8]`.
fn backing_scale_from(contents_scale: f64) -> u32 {
    bounded_cast::f64_to_u32_saturating(contents_scale.round()).clamp(1, 8)
}

/// The cursor scale to publish when the layer no longer matches it.
///
/// `Some(scale)` when the layer asks for a different factor than the one
/// last published, `None` while they agree.
fn backing_scale_change(applied: u32, contents_scale: f64) -> Option<u32> {
    let target = backing_scale_from(contents_scale);
    (target != applied).then_some(target)
}

/// The `contentsScale` of Wine's metal layer: 2 in retina mode, 1 otherwise.
///
/// A plain property read; the value is set once by winemac when it creates
/// the layer and changed only by a retina-mode switch.
fn layer_contents_scale(layer: *mut c_void) -> f64 {
    // SAFETY: `layer` is the `CAMetalLayer` pointer winemac handed out, alive
    // for as long as the metal view it belongs to.
    let layer = unsafe { Retained::retain(layer.cast::<objc2_quartz_core::CAMetalLayer>()) };
    layer.map_or(1.0, |layer| layer.contentsScale())
}

/// A screen's refresh ceiling in Hz, from `NSScreen.maximumFramesPerSecond`.
///
/// 60 on most external displays, 120 on a `ProMotion` panel. `0.0` when the
/// screen reports nothing usable (older macOS, a virtualised display), which
/// [`min_present_duration`] reads as "no vsync throttle".
fn screen_max_hz(screen: &objc2_app_kit::NSScreen) -> f64 {
    let clamped = screen.maximumFramesPerSecond().clamp(0, 1000);
    let as_u32 = u32::try_from(clamped).expect("clamped above to [0, 1000]");
    f64::from(as_u32)
}

/// Everything the PE side's `AttachMetalLayer` request carries in.
///
/// Bundled so the entry point stays inside clippy's `too_many_arguments`
/// threshold, the same reason [`PresentPacing`] exists.
pub struct LayerAttachRequest {
    /// The guest window the layer is attached to.
    pub hwnd: u64,
    /// Back-buffer width, for the geometry log line.
    pub width: u32,
    /// Back-buffer height, for the geometry log line.
    pub height: u32,
    /// The guest's vsync ask plus the user's frame-rate ceiling.
    pub pacing: PresentPacing,
    /// `color.hdr.enable` from `mtld3d.conf`.
    pub hdr_enable: bool,
    /// `color.space` from `mtld3d.conf`.
    pub color_space: ColorSpacePolicy,
    /// Address of the PE-side `AtomicU32` that receives a changed backing scale.
    ///
    /// Recorded on the attachment record and written only while that record
    /// is live. `0` leaves the display-follow path with nothing to publish
    /// into, which is what a headless smoke test that never built one looks
    /// like.
    pub backing_scale_sink_ptr: u64,
    /// Where a cursor re-apply request is written on the PE side, `0` = nowhere.
    pub cursor_kick_sink_ptr: u64,
    /// `cursor.software`, resolved here against the layer mode attach picks.
    pub software_cursor: SoftwareCursorPolicy,
}

/// What attach answers the PE side about the display and the cursor.
#[derive(Clone, Copy)]
pub struct DisplayCaps {
    /// The Wine layer's `contentsScale` rounded + clamped to `[1, 8]`: 2 in retina mode, else 1.
    pub backing_scale: u32,
    /// Whether the device draws its cursor through the overlay window.
    ///
    /// `cursor.software` resolved against the layer mode; `Auto` follows HDR.
    pub software_cursor_active: bool,
}

/// Which of the two `CAMetalLayer` configurations a display asks for.
///
/// `Sdr` is `BGRA8Unorm` with a standard-range colorspace and no EDR opt-in;
/// `Hdr` is `RGBA16Float` with an extended-linear colorspace and
/// `wantsExtendedDynamicRangeContent`. The three properties are one decision:
/// a float surface tagged with a non-linear profile double-EOTFs and goes
/// dark, and an extended-linear profile without the opt-in clamps at SDR
/// paper white.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerMode {
    Sdr,
    Hdr,
}

/// Layer colorspace + layer-mode decision bundled together.
///
/// Keeps `configure_metal_layer` inside clippy's `too_many_arguments`
/// threshold. `mode` drives the SDR-vs-HDR branch; `native_colorspace`
/// is the screen's profile (SDR feeds through `copy_with_standard_range`,
/// HDR through `extended_linearized`); `screen_name` is the logging key
/// for fallback warns; `screen_profile_name` is the user-facing profile
/// string surfaced in the post-config log line.
struct LayerColorConfig {
    mode: LayerMode,
    color_space: ColorSpacePolicy,
    native_colorspace: Option<Retained<CGColorSpace>>,
    screen_name: Option<String>,
    screen_profile_name: Option<String>,
}

/// Borrowed view of `LayerColorConfig` for the main-thread callee.
#[derive(Clone, Copy)]
struct LayerColorRefs<'a> {
    mode: LayerMode,
    color_space: ColorSpacePolicy,
    native_colorspace: Option<&'a CGColorSpace>,
    screen_name: Option<&'a str>,
    screen_profile_name: Option<&'a str>,
}

bitflags::bitflags! {
    /// Diagnostic colorspace classification of the bound screen's profile.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ColorspaceFlags: u8 {
        /// `CGColorSpaceIsHDR` on the screen's profile.
        ///
        /// Diagnostic only — asymmetry between `edr_potential <= 1.0` and
        /// this being set flags the case of an HDR-capable display macOS's
        /// EDR pipeline isn't managing.
        const IS_HDR = 1 << 0;
        /// `CGColorSpaceIsWideGamutRGB` on the screen's profile.
        ///
        /// Diagnostic only — paired with the post-config gamut label for
        /// sanity.
        const IS_WIDE_GAMUT = 1 << 1;
    }
}

/// Bundle of `NSScreen`-derived properties used at attach time.
///
/// All of them drive layer configuration unix-side; what travels back to the
/// PE side comes from the layer, not the screen.
struct DisplayHint {
    /// `maximumPotentialExtendedDynamicRangeColorComponentValue` — static panel ceiling.
    ///
    /// Drives the SDR-vs-HDR layer-config decision.
    edr_potential: f64,
    /// `NSScreen.localizedName` for logging.
    screen_name: Option<String>,
    /// `NSScreen.colorSpace.CGColorSpace` — the display's own profile.
    ///
    /// SDR feeds this through `CGColorSpaceCreateCopyWithStandardRange`;
    /// HDR feeds it through `CGColorSpaceCreateExtendedLinearized`.
    native_colorspace: Option<Retained<CGColorSpace>>,
    /// `NSColorSpace.localizedName` — user-facing string.
    ///
    /// Like `"Color LCD"`, `"Display P3"`, `"sRGB IEC61966-2.1"`. Set for
    /// the post-config log line so the actual screen profile shows up
    /// in user reports. `CGColorSpace::name()` returns `None` for
    /// calibrated panel profiles, so we go via `NSColorSpace` instead.
    screen_profile_name: Option<String>,
    /// Diagnostic colorspace classification of the screen's profile (`IS_HDR` / `IS_WIDE_GAMUT`).
    ///
    /// See [`ColorspaceFlags`].
    colorspace_flags: ColorspaceFlags,
    /// `NSScreen.maximumFramesPerSecond` for the bound view's panel.
    ///
    /// `0.0` if `NSScreen` reported no usable value (older macOS /
    /// virtualised display). Drives the present-throttle duration computed
    /// at attach.
    panel_max_hz: f64,
}

type GetWinDataFn = unsafe extern "C" fn(*mut c_void) -> *mut MacdrvWinData;
type ReleaseWinDataFn = unsafe extern "C" fn(*mut MacdrvWinData);
type CreateMetalViewFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
type GetMetalLayerFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type ReleaseMetalViewFn = unsafe extern "C" fn(*mut c_void);
type GetCocoaWindowFn = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;

unsafe extern "C" {
    /// libdispatch's main-queue singleton, exported by libSystem as `_dispatch_main_q`.
    ///
    /// `dispatch_get_main_queue()` is a C macro that resolves to
    /// `&_dispatch_main_q`; taking the address here matches that expansion.
    static _dispatch_main_q: c_void;
    /// libdispatch's function-pointer `dispatch_sync`.
    ///
    /// Synchronous dispatch to a queue without needing an Obj-C block — the
    /// `(ctx, work_fn)` pair carries the closure state. Standard libSystem
    /// export.
    fn dispatch_sync_f(queue: *mut c_void, ctx: *mut c_void, work: extern "C" fn(*mut c_void));
    /// libdispatch's function-pointer `dispatch_async`.
    ///
    /// The asynchronous twin of `dispatch_sync_f`. Used where the presenting
    /// thread needs main-thread work done but must not wait for it: waiting
    /// would put the main run loop in the frame's critical path and deadlock
    /// outright if the main thread is itself blocked on us.
    fn dispatch_async_f(queue: *mut c_void, ctx: *mut c_void, work: extern "C" fn(*mut c_void));
}

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

/// Run a closure synchronously on `AppKit`'s main thread (libdispatch's main queue).
///
/// Waits for completion. Apple documents compositor-impacting `CALayer`
/// setters — `wantsExtendedDynamicRangeContent`, `colorspace`, `pixelFormat` —
/// as needing to take effect inside a `CATransaction` commit, which by
/// convention runs on the main thread's run loop. Setting these properties
/// from a non-main thread sets the model layer but leaves the *rendered*
/// state stale until the next main-thread commit; the `WindowServer` EDR-mode
/// arbiter may sample the layer's state between our off-main write and that
/// commit and see `wantsEDR=false`, preventing the screen from ever promoting
/// to EDR. Wine itself wraps `macdrv_view_create_metal_view` in
/// `OnMainThread` (`dlls/winemac.drv/cocoa_window.m`); mtld3d mirrors that
/// posture for its own layer configuration.
///
/// The closure runs in an autorelease pool of its own, so the objects it
/// autoreleases drain when it returns rather than at the end of whatever
/// pool the main thread's caller holds open around its own work; winemac
/// runs its request loop inside one, and an object released into it lives
/// until that loop's iteration ends.
///
/// `panic = "abort"` in our profile means the closure's panic aborts
/// the process — no unwinding across the `extern "C"` boundary, no UB.
fn run_on_main_thread_sync<F: FnOnce()>(f: F) {
    struct CallCtx<F> {
        f: Option<F>,
    }
    extern "C" fn thunk<F: FnOnce()>(ctx: *mut c_void) {
        // SAFETY: `ctx` is the `&mut CallCtx<F>` we just handed to
        // `dispatch_sync_f`; libdispatch passes it through to the
        // worker function unchanged.
        let ctx = unsafe { &mut *(ctx.cast::<CallCtx<F>>()) };
        if let Some(f) = ctx.f.take() {
            autoreleasepool(|_| f());
        }
    }
    let mut ctx = CallCtx { f: Some(f) };
    // SAFETY: `_dispatch_main_q` is libSystem's main-queue singleton —
    // a valid `dispatch_queue_t` for the process lifetime. `&mut ctx`
    // is valid until this function returns, and `dispatch_sync_f` is
    // synchronous, so the thunk runs before we drop `ctx`.
    unsafe {
        let main_q = (&raw const _dispatch_main_q).cast_mut().cast::<c_void>();
        dispatch_sync_f(main_q, (&raw mut ctx).cast::<c_void>(), thunk::<F>);
    }
}

/// Run a closure on `AppKit`'s main thread without waiting for it.
///
/// The asynchronous twin of [`run_on_main_thread_sync`], for callers that must
/// not put the main run loop in their critical path: the thunk that carries the
/// software cursor's state runs on the API thread, the display reconciliation
/// runs on the submit thread's cadence, and a Reset's re-pacing runs on the
/// encoder thread. The closure is boxed and handed to libdispatch, which runs
/// it once on the main queue and frees it, in an autorelease pool of its own
/// for the reason the synchronous twin gives.
///
/// This is also the only hop a thread may take while another thread can be
/// waiting on it. winemac's main thread waits on Wine threads in a private
/// run-loop mode that does not drain the main queue, and the thread it waits
/// on can be the API thread inside `Present`, holding the device lock and
/// blocked on the encoder's channel; a synchronous hop from the encoder
/// thread would then close a cycle that the asynchronous one cannot.
fn run_on_main_thread_async<F: FnOnce() + Send + 'static>(f: F) {
    extern "C" fn thunk<F: FnOnce()>(ctx: *mut c_void) {
        // SAFETY: `ctx` is the `Box<F>` leaked below; libdispatch hands it to
        // the work function exactly once, so taking it back here is the one
        // and only owner.
        let f = unsafe { Box::from_raw(ctx.cast::<F>()) };
        autoreleasepool(|_| f());
    }
    let ctx = Box::into_raw(Box::new(f));
    // SAFETY: `_dispatch_main_q` is libSystem's main-queue singleton, a valid
    // `dispatch_queue_t` for the process lifetime; `ctx` is a heap allocation
    // owned by the queued block until `thunk` consumes it.
    unsafe {
        let main_q = (&raw const _dispatch_main_q).cast_mut().cast::<c_void>();
        dispatch_async_f(main_q, ctx.cast::<c_void>(), thunk::<F>);
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
    request: LayerAttachRequest,
) -> Option<(
    MetalHandle<NSViewKind>,
    MetalHandle<CAMetalLayerKind>,
    DisplayCaps,
)> {
    let LayerAttachRequest {
        hwnd,
        width,
        height,
        pacing,
        hdr_enable,
        color_space,
        backing_scale_sink_ptr,
        cursor_kick_sink_ptr,
        software_cursor,
    } = request;
    if hwnd == 0 || device_handle.is_null() {
        return None;
    }

    let funcs = MacdrvFuncs::load()?;
    // A view kept from this window's previous device is taken as it is; one
    // kept from a window that is gone is moved in; only a window with
    // neither gets a view from Wine.
    let kept = kept_metal_view(&funcs, hwnd);
    let hosted = matches!(kept, Some(KeptView::Hosted { .. }));
    let (view, layer) = match kept {
        Some(KeptView::Hosted { view, layer }) => (view as *mut c_void, layer as *mut c_void),
        Some(KeptView::Orphan {
            view,
            layer,
            from_hwnd,
        }) => adopt_metal_view(&funcs, hwnd, device_handle, view, layer, from_hwnd)?,
        None => create_metal_view(&funcs, hwnd, device_handle)?,
    };
    // AppKit owns the view's window and screen relationships on the main
    // thread; the view is retained for as long as the device holds it.
    let mut hint = None;
    run_on_main_thread_sync(|| {
        let mtm = objc2::MainThreadMarker::new().expect("display lookup runs on the main thread");
        hint = Some(view_display_caps(view, mtm));
    });
    let hint = hint.expect("synchronous display lookup completed");
    if hosted {
        info!(
            target: LOG_TARGET,
            "present: metal view {:#x} kept from the previous device on window {hwnd:#x} is \
             attached again, layer and all",
            view as usize,
        );
    }
    // The cursor scale follows Wine's retina mode, which the layer
    // carries as its contents scale, not the display's own factor: in
    // non-retina mode macOS already doubles everything the game draws,
    // its cursor included.
    let backing_scale = backing_scale_from(layer_contents_scale(layer));
    // Decide HDR vs SDR layer configuration from the panel's
    // static potential + the user's `color.hdr.enable` setting. The
    // result is the configuration the layer now carries; the user
    // gate stays unix-side from here on.
    let mode = resolve_layer_mode(
        hint.edr_potential,
        hint.screen_name.as_deref(),
        hint.colorspace_flags.contains(ColorspaceFlags::IS_HDR),
        hint.colorspace_flags
            .contains(ColorspaceFlags::IS_WIDE_GAMUT),
        hdr_enable,
    );
    // The record holds the layer and everything the display-follow
    // path needs to reconfigure it for a screen that was not attached
    // yet: the same gate and colorspace policy attach applied, the
    // guest's vsync ask, the user's frame cap and the scale the PE
    // side is already using.
    let mut flags = AttachFlags::empty();
    flags.set(AttachFlags::HDR_ENABLE_REQUESTED, hdr_enable);
    flags.set(AttachFlags::HDR_ACTIVE, mode == LayerMode::Hdr);
    let att = attachment::register(
        view as usize,
        layer as usize,
        &AttachLatches {
            hwnd,
            flags,
            color_space,
            pacing_bits: pack_pacing(&pacing),
            backing_scale,
            backing_scale_sink: usize::try_from(backing_scale_sink_ptr)
                .expect("PE wire pointer fits host address space (unix is 64-bit)"),
            cursor_kick_sink: usize::try_from(cursor_kick_sink_ptr)
                .expect("PE wire pointer fits host address space (unix is 64-bit)"),
        },
    );
    attachment::publish_backing_scale(&att, backing_scale);
    // The software cursor rides the same decision: the overlay window
    // is a compositing cost an EDR layer already pays.
    let software_cursor_active = software_cursor.resolve(mode == LayerMode::Hdr);
    info!(
        target: LOG_TARGET,
        "cursor: software overlay {} (cursor.software = {software_cursor:?}, layer {mode:?})",
        if software_cursor_active { "on" } else { "off" },
    );
    let min_present_duration = configure_metal_layer(
        layer,
        device_handle.raw(),
        width,
        height,
        &pacing,
        hint.panel_max_hz,
        LayerColorConfig {
            mode,
            color_space,
            native_colorspace: hint.native_colorspace,
            screen_name: hint.screen_name,
            screen_profile_name: hint.screen_profile_name,
        },
    );
    att.set_min_present_duration(min_present_duration);
    // Start occlusion tracking for this window so presents skip the
    // `nextDrawable` timeout while it is fully covered/minimised.
    install_occlusion_tracking(&att);
    // SAFETY: the view is the one `macdrv_view_create_metal_view` handed
    // out retained, either just now or for this window's previous device,
    // kept since. The PE side holds the handle until the matching retire.
    let view_handle = unsafe { MetalHandle::<NSViewKind>::new(view as u64) };
    // SAFETY: as the comment above; Wine retains the `CAMetalLayer` for the
    // view's lifetime.
    let layer_handle = unsafe { MetalHandle::<CAMetalLayerKind>::new(layer as u64) };
    let caps = DisplayCaps {
        backing_scale,
        software_cursor_active,
    };
    Some((view_handle, layer_handle, caps))
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
) -> Option<(*mut c_void, *mut c_void)> {
    let win_data = get_win_data(funcs, hwnd)?;
    // SAFETY: `win_data` is the live record `get_win_data` handed back,
    // valid until `release_win_data`.
    let client_view = unsafe { (*win_data).client_cocoa_view };
    let result = wine_metal_view(funcs, client_view, device_handle);
    release_win_data(funcs, win_data);
    result
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
    view: usize,
    layer: usize,
    from_hwnd: u64,
) -> Option<(*mut c_void, *mut c_void)> {
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
    };
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
        error!(target: LOG_TARGET, "macdrv_view_get_metal_layer returned null");
        return None;
    }
    Some((view, layer))
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
/// the first `nextDrawable` of the kept layer, which Wine reports as that
/// surface's present). No longer hosted (the handle reused by a new window,
/// or its window gone) it is moved into the window the handle has now. With
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

/// Apply a runtime change to the guest's vsync request.
///
/// The D3D9 Reset path honouring a `D3DPRESENT_PARAMETERS::PresentationInterval`
/// flip, run by the encoder thread on the first frame after the Reset. The
/// layer's `displaySyncEnabled` stays `false` from attach time onward; what
/// changes is the present-throttle duration on the layer's attachment record,
/// consulted at the present site. The encoder thread latches the new pacing
/// on the record (the PE side re-sends the `present.maxFps` cap so it survives
/// Resets) and queues the same main-thread reconciliation the display-follow
/// poll uses, which walks to the window's own screen and re-derives the
/// throttle from the latched pacing ([`follow_screen_present_throttle`]). So
/// the panel is the one under the window rather than the main screen, nothing
/// here touches `AppKit` off the main thread, and the new throttle is live
/// within one present. A device's Reset re-paces that device alone.
///
/// The hop is the asynchronous one, and has to be: the encoder thread is what
/// the API thread waits on from inside `Present`, and winemac's main thread can
/// be waiting on that API thread in turn ([`run_on_main_thread_async`]). The
/// registry lock is released by the lookup before the hop is queued.
pub fn set_display_sync_enabled(
    layer_handle: MetalHandle<CAMetalLayerKind>,
    pacing: &PresentPacing,
) {
    let layer_addr =
        usize::try_from(layer_handle.raw()).expect("a 64-bit host addresses every layer pointer");
    let Some(att) = attachment::find_by_layer(layer_addr) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: SetDisplaySyncEnabled for layer {layer_addr:#x} with no attachment record; \
             the present throttle is unchanged",
        );
        return;
    };
    att.set_pacing_bits(pack_pacing(pacing));
    // The counterpart of the attach line, so a log says which present the
    // guest's interval change reached the layer on. One line per Reset that
    // changed the interval, off the encoder thread, not per frame.
    info!(
        target: LOG_TARGET,
        "present: layer {layer_addr:#x} re-paced (vsync {}, maxFps {})",
        if pacing.vsync_requested { "on" } else { "off" },
        pacing.max_fps,
    );
    if att.request_refresh() {
        run_on_main_thread_async(move || refresh_attachment_on_main(&att));
    }
}

/// Host-time seconds (`CFTimeInterval`) to nanoseconds, saturating.
///
/// `presentedTime` is a `CACurrentMediaTime`-based host time; a session's
/// uptime in nanoseconds sits far below `u64::MAX`.
pub fn host_seconds_to_ns(secs: f64) -> u64 {
    bounded_cast::f64_to_u64_saturating(secs * 1e9)
}

/// Numeric casts where the cast lints fire but the bounds are established by the caller.
///
/// Grouping them under one mod-level allow collapses what would otherwise be
/// four per-site allows into one. Each fn is the raw cast — callers document
/// the bound that justifies it.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod bounded_cast {
    /// Saturating `f64 → u32`.
    ///
    /// NaN/negative → 0, ≥ `u32::MAX` → `u32::MAX`; all other inputs land in
    /// `(0.0, u32::MAX)` where the cast is exact.
    pub fn f64_to_u32_saturating(v: f64) -> u32 {
        if !v.is_finite() || v <= 0.0 {
            return 0;
        }
        if v >= f64::from(u32::MAX) {
            return u32::MAX;
        }
        v as u32
    }

    /// Saturating `f64 → u64`.
    ///
    /// NaN/negative → 0, ≥ `u64::MAX` → `u64::MAX`; all other inputs land in
    /// `(0.0, u64::MAX)` where the cast is exact to f64 precision.
    pub fn f64_to_u64_saturating(v: f64) -> u64 {
        if !v.is_finite() || v <= 0.0 {
            return 0;
        }
        if v >= u64::MAX as f64 {
            return u64::MAX;
        }
        v as u64
    }

    /// `f64 → f32` narrowing.
    ///
    /// Caller establishes the bound where mantissa loss is acceptable.
    pub const fn f64_to_f32(v: f64) -> f32 {
        v as f32
    }

    /// `i32 → f32` narrowing.
    ///
    /// Caller establishes `|v|` is well inside the f32 mantissa (< 2^24) so
    /// the cast is exact.
    pub const fn i32_to_f32(v: i32) -> f32 {
        v as f32
    }
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
/// that releases the oldest. Parking also drops the layer's drawable pool
/// ([`shrink_parked_layer`]), so a kept layer holds the one surface it still
/// displays rather than a pool at the window's size. A device attaching to
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
    // attach put the layer's geometry back.
    shrink_parked_layer(record.layer());
    let displaced = PARKED_METAL_VIEW
        .lock()
        .expect("metal view park mutex poisoned")
        .park(record.hwnd(), view, record.layer());
    debug!(
        target: LOG_TARGET,
        "present: metal view {view:#x} kept for the next device on window {:#x}",
        record.hwnd(),
    );
    if let Some(displaced) = displaced {
        release_metal_view(displaced);
    }
}

/// Drop the drawable pool of a layer about to be parked, keeping the frame it shows.
///
/// A layer's pool is its window's size, `maximumDrawableCount` deep: at
/// 3456x2234 `RGBA16Float` that is about 62 MB a drawable, held while no
/// device presents through it. `CAMetalLayer` has no call that empties the
/// pool, and neither lowering `maximumDrawableCount` nor releasing the last
/// drawable releases anything of it; writing `drawableSize` does, which is
/// why [`sync_drawable_size`] compares before it writes. So the park writes
/// the smallest drawable Metal takes, and attach restores the layer's own
/// geometry through [`configure_metal_layer`], as it does for a view Wine
/// just created. What survives the write is the surface the layer
/// displays, which is the frame the window shows until its next device
/// presents.
///
/// Synchronous, because the caller is about to publish the view to the park:
/// the write has to land before another device can take the view back.
fn shrink_parked_layer(layer: usize) {
    run_on_main_thread_sync(|| {
        let mtm = MainThreadMarker::new().expect("the parked layer is shrunk on the main thread");
        shrink_layer_drawable(layer, mtm);
    });
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
/// the one surface its layer still displays; the pool behind it goes at the
/// park ([`shrink_parked_layer`]).
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

/// Gather the `NSScreen` properties of the screen the bound view lives on.
///
/// Reads the screen's `colorSpace`, EDR potential, refresh ceiling and
/// `localizedName`; returns them as a [`DisplayHint`]. All of them drive
/// layer configuration unix-side; the cursor scale the PE side consumes comes
/// from the Wine layer instead (see [`layer_contents_scale`]).
///
/// The colorspace flows through to `configure_metal_layer_inner` and drives
/// the layer's `colorspace` property — SDR uses it directly (identity = max
/// vibrance per display), HDR classifies it into an extended-linear variant.
fn view_display_caps(view: *mut c_void, mtm: objc2::MainThreadMarker) -> DisplayHint {
    use objc2_app_kit::NSScreen;

    // Prefer the NSScreen attached to the view's window so
    // multi-monitor setups with mixed scales pick the right display;
    // fall back to `+[NSScreen mainScreen]`.
    let view_obj = if view.is_null() {
        None
    } else {
        // SAFETY: Wine's macdrv hands us a retained NSView*; `Retained::retain`
        // bumps the count for the duration of the property walk and drops
        // when this Option goes out of scope.
        unsafe { Retained::retain(view.cast::<NSView>()) }
    };
    let screen = view_obj
        .as_deref()
        .and_then(NSView::window)
        .and_then(|w| w.screen())
        .or_else(|| NSScreen::mainScreen(mtm));

    // `maximumPotentialExtendedDynamicRangeColorComponentValue` is
    // the static panel ceiling (vs the dynamic `maximum…` which moves
    // with brightness / thermals and is polled per-frame in
    // `submit_frame`). The static value drives the one-shot SDR-vs-HDR
    // layer configuration decision at attach.
    // Construct the per-screen bundle inline. The map_or default
    // covers the no-screen path (view==null, mainScreen() None) —
    // potential falls back to 1.0 (no EDR), no colorspace, no profile
    // name, both diagnostic flags off.
    let (
        edr_potential,
        screen_name,
        native_colorspace,
        screen_profile_name,
        colorspace_is_hdr,
        colorspace_is_wide_gamut,
    ) = screen
        .as_deref()
        .map_or((1.0_f64, None, None, None, false, false), |s| {
            // The CGColorSpace (for layer setColorspace) and its gamut
            // label, plus is_hdr/is_wide_gamut flags for the
            // HDR-tagged-but-no-EDR asymmetry diagnostic.
            let (cg_cs, profile_name) = screen_color_profile(s);
            let is_hdr = cg_cs.as_deref().is_some_and(CGColorSpace::is_hdr);
            let is_wide = cg_cs
                .as_deref()
                .is_some_and(CGColorSpace::is_wide_gamut_rgb);
            (
                s.maximumPotentialExtendedDynamicRangeColorComponentValue(),
                Some(s.localizedName().to_string()),
                cg_cs,
                profile_name,
                is_hdr,
                is_wide,
            )
        });

    // The panel ceiling drives the present-throttle duration computed at
    // attach; a display move re-derives it from the same helper.
    let panel_max_hz = screen.as_deref().map_or(0.0_f64, screen_max_hz);
    let mut colorspace_flags = ColorspaceFlags::empty();
    colorspace_flags.set(ColorspaceFlags::IS_HDR, colorspace_is_hdr);
    colorspace_flags.set(ColorspaceFlags::IS_WIDE_GAMUT, colorspace_is_wide_gamut);
    DisplayHint {
        edr_potential,
        screen_name,
        native_colorspace,
        screen_profile_name,
        colorspace_flags,
        panel_max_hz,
    }
}

/// The layer configuration a screen's EDR ceiling and the user's setting ask for.
///
/// `potential` is `maximumPotentialExtendedDynamicRangeColorComponentValue`,
/// the *static* panel ceiling rather than the live headroom: a panel that can
/// reach EDR keeps the HDR layer through a brightness dip or a thermal
/// throttle, and one that cannot never gets it. A non-finite or `<= 1.0`
/// ceiling resolves to `Sdr`, and so does `color.hdr.enable = false` whatever
/// the panel reports.
const fn layer_mode_for(potential: f64, hdr_enable: bool) -> LayerMode {
    if hdr_enable && potential > 1.0 && potential.is_finite() {
        LayerMode::Hdr
    } else {
        LayerMode::Sdr
    }
}

/// The configuration to re-apply when a screen no longer matches the layer.
///
/// `Some(mode)` when the applied configuration disagrees with what the screen
/// asks for, `None` while the two already match, which is every poll of a
/// session that stays on one display.
fn layer_mode_change(applied: LayerMode, potential: f64, hdr_enable: bool) -> Option<LayerMode> {
    let target = layer_mode_for(potential, hdr_enable);
    (target != applied).then_some(target)
}

/// Decide the layer configuration at attach time, and say why in the log.
///
/// Wraps [`layer_mode_for`] with one info line per attach naming the screen,
/// so multi-monitor reports can be triaged. The actual per-frame BT.2446
/// target is the live dynamic headroom polled in `submit_frame`, not a
/// function of `potential`.
fn resolve_layer_mode(
    potential: f64,
    screen_name: Option<&str>,
    colorspace_is_hdr: bool,
    colorspace_is_wide_gamut: bool,
    hdr_enable: bool,
) -> LayerMode {
    let screen = screen_name.unwrap_or("(unknown screen)");
    // Diagnostic suffix shared across all three branches. `cs_hdr=true`
    // alongside `potential=1.0` is the asymmetric case: the display is
    // tagged HDR (PQ/HLG) but macOS isn't engaging EDR. There's no
    // software fix for that case (WindowServer owns the pipeline);
    // logging it makes the failure mode visible in user reports.
    let cs = format!("cs_hdr={colorspace_is_hdr} cs_wide={colorspace_is_wide_gamut}");
    let mode = layer_mode_for(potential, hdr_enable);
    if !hdr_enable {
        info!(
            target: LOG_TARGET,
            "hdr: disabled via mtld3d.conf color.hdr.enable=false on '{screen}' (potential={potential:.2}× {cs})",
        );
    } else if mode == LayerMode::Sdr {
        info!(
            target: LOG_TARGET,
            "hdr: '{screen}' has no EDR headroom (potential={potential:.2}× {cs}), running SDR",
        );
    } else {
        info!(
            target: LOG_TARGET,
            "hdr: '{screen}' reports {potential:.2}× peak headroom ({cs}) — HDR active, present peak follows live headroom",
        );
    }
    mode
}

/// Reconcile every live attachment against its display. **Main thread only.**
///
/// The screen-parameter filter calls this on a real topology or mode change,
/// when every window may have landed on another panel.
fn refresh_all_on_main() {
    for att in attachment::live() {
        refresh_attachment_on_main(&att);
    }
}

/// Read the live EDR headroom of the record's window and publish it. **Main thread only.**
///
/// Walks `NSView.window → NSWindow.screen` and reads the screen's dynamic
/// `maximumExtendedDynamicRangeColorComponentValue`, which is the walk that
/// must not happen anywhere else: the first two are main-thread-only
/// objects that the main thread rebuilds across a window or display change.
/// The value is the panel's currently-available headroom, which on a Mac is
/// `panel_peak_nits / current_paper_white_nits`; it drops as the user raises
/// display brightness, and under thermal load, and `submit_frame` clamps
/// the BT.2446-A target peak to it because macOS global-scales
/// over-headroom EDR (crushes midtones) rather than soft-knee compressing
/// the top. Logs the drift line here too, for the same reason, since naming
/// the screen means walking to it again.
///
/// The walk ends on whichever screen the window is on *now*, so it is also
/// where everything the display decides is reconciled against that screen
/// for this record: the layer's own configuration
/// ([`follow_screen_layer_mode`]), the present throttle
/// ([`follow_screen_present_throttle`]) and the backing scale the PE side
/// consumes ([`follow_layer_backing_scale`]). A record that was retired
/// between the queueing and the run finds no view and does nothing.
fn refresh_attachment_on_main(att: &Arc<Attachment>) {
    use objc2_app_kit::NSScreen;

    att.end_headroom_refresh();
    let mtm = MainThreadMarker::new().expect("refresh_attachment_on_main runs on the main thread");
    let Some(view_obj) = attachment::retain_view(att, mtm) else {
        return;
    };
    // The attach that registered this record may have run before Wine had an
    // application delegate, which leaves the notification unfiltered. This is
    // the recurring main-thread pass, so it is where the install is retried;
    // once it has landed the retry is one relaxed load.
    install_screen_params_filter(mtm);
    let screen = view_obj
        .window()
        .and_then(|w| w.screen())
        .or_else(|| NSScreen::mainScreen(mtm));
    let headroom = screen.as_deref().map_or(
        1.0,
        NSScreen::maximumExtendedDynamicRangeColorComponentValue,
    );
    let headroom = if headroom.is_finite() && headroom >= 1.0 {
        // EDR headroom is at most ~16x in practice (Apple Reference Display
        // peaks at 16x); f32 mantissa loss is one ULP at the 1x to 4x range,
        // negligible for the Metal shader's float peak uniform.
        bounded_cast::f64_to_f32(headroom)
    } else {
        1.0
    };
    att.set_headroom(headroom);
    // Only reconcile against a screen we actually reached. A window mid-move
    // between displays reports none, and the layer is better left as it is
    // than reconfigured twice against a screen the window is leaving.
    if let Some(screen) = screen.as_deref() {
        follow_screen_layer_mode(att, screen, mtm);
        follow_screen_present_throttle(att, screen);
        follow_layer_backing_scale(att, mtm);
    }
    log_headroom_change_if_any(att, headroom, &view_obj, mtm);
    cursor_overlay::reconcile_on_main();
}

/// Emit one `info!` line when the live headroom drifts more than 5% from the last logged value.
///
/// **Main thread only**, because naming the screen walks the same
/// main-thread-only `NSView.window → NSWindow.screen` chain the reading
/// itself does. Called from [`refresh_attachment_on_main`].
///
/// A record's first call always logs so the refresh baseline is distinct
/// from the attach line. Subsequent within-±5% calls are silent, which gives
/// the user a way to verify the per-frame clamp is doing what it claims
/// without flooding the console during sub-percent oscillation. Names the
/// screen the view is currently on so a stuck-at-1.0 run tells us *which*
/// display is reporting no headroom.
fn log_headroom_change_if_any(
    att: &Arc<Attachment>,
    current_headroom: f32,
    view: &objc2_app_kit::NSView,
    mtm: MainThreadMarker,
) {
    let last = att.last_logged_headroom();
    let should_log = last.is_none_or(|last| ((current_headroom - last).abs() / last) > 0.05);
    if !should_log {
        return;
    }
    let last = last.unwrap_or(0.0);
    att.set_last_logged_headroom(current_headroom);
    let screen = view_screen_name(view, mtm);
    let screen_ref = screen.as_deref().unwrap_or("(unknown screen)");
    info!(
        target: LOG_TARGET,
        "hdr: '{screen_ref}' headroom {current_headroom:.2}× (was {last:.2}×)",
    );
}

/// Look up `NSScreen.localizedName` for the screen the view's window is currently on.
///
/// Mirrors the screen-lookup walk in [`refresh_attachment_on_main`] so the
/// logged screen identity matches the screen whose headroom we just read.
/// **Main thread only**, for the same reason that one is. Returns `None`
/// if the view has no window or no screen association yet.
fn view_screen_name(view: &objc2_app_kit::NSView, mtm: MainThreadMarker) -> Option<String> {
    use objc2_app_kit::NSScreen;

    let screen = view
        .window()
        .and_then(|w| w.screen())
        .or_else(|| NSScreen::mainScreen(mtm))?;
    Some(screen.localizedName().to_string())
}

/// Configure a freshly attached layer on the main thread; returns the present throttle.
///
/// The throttle is [`min_present_duration`] of the panel under the window
/// and the guest's pacing, in seconds; the caller stores it on the
/// attachment record for the present site to consult.
fn configure_metal_layer(
    layer: *mut c_void,
    device_handle: u64,
    width: u32,
    height: u32,
    pacing: &PresentPacing,
    panel_max_hz: f64,
    color: LayerColorConfig,
) -> f64 {
    // Hop to AppKit's main thread for the entire CALayer configuration
    // block. `wantsExtendedDynamicRangeContent`, `colorspace`, and
    // `pixelFormat` must land in a CATransaction commit observable by
    // the WindowServer EDR-mode arbiter *before* the first present;
    // setting them on the encoder thread (the original caller) only
    // updates the model layer and leaves an intermittent race where
    // the arbiter samples the layer in its old non-EDR state and the
    // panel never promotes. Wine itself wraps `macdrv_view_create_metal_view`
    // in `OnMainThread`, so the layer is *created* on the main thread —
    // we just have to keep our configuration on the same thread.
    //
    // Synchronous dispatch (not async) so `attach_metal_layer` is
    // guaranteed to return with the layer fully configured: subsequent
    // `submit_frame` calls on the encoder thread see committed state.
    //
    // Raw pointers (`layer`, `device_handle`) are `!Send` in Rust but
    // libdispatch crosses the thread boundary by-value bit-for-bit;
    // they're safe to use on the main thread inside the closure
    // because Wine retains the underlying Obj-C objects for the
    // lifetime of the view. Cast the layer pointer through `u64` to
    // strip the `!Send` so the `move` closure compiles; reconstruct
    // on the main thread. Sound on x86_64 — pointers fit in `u64`
    // losslessly.
    let layer_addr = layer as u64;
    run_on_main_thread_sync(move || {
        configure_metal_layer_inner(
            layer_addr as *mut c_void,
            device_handle,
            width,
            height,
            LayerColorRefs {
                mode: color.mode,
                color_space: color.color_space,
                native_colorspace: color.native_colorspace.as_deref(),
                screen_name: color.screen_name.as_deref(),
                screen_profile_name: color.screen_profile_name.as_deref(),
            },
        );
    });
    min_present_duration(panel_max_hz, pacing)
}

fn configure_metal_layer_inner(
    layer: *mut c_void,
    device_handle: u64,
    width: u32,
    height: u32,
    color: LayerColorRefs<'_>,
) {
    use objc2_quartz_core::{CAMetalLayer, kCAGravityResizeAspect};

    // Cast the raw `*mut c_void` from Wine's macdrv into typed
    // `Retained<CAMetalLayer>`. Using typed objc2 setters means a
    // typo in a selector name (e.g.
    // `setWantsExtendedDynamicRange` vs `…RangeContent`) becomes a
    // compile error rather than a runtime `unrecognized selector` crash.
    //
    // SAFETY: `layer` is the `CAMetalLayer` pointer wine macdrv handed us;
    // `Retained::retain` bumps the refcount via standard Cocoa semantics.
    let Some(layer) = (unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }) else {
        return;
    };
    // SAFETY: device_handle is a previously-retained MTLDevice address.
    let device = unsafe { MetalHandle::<MTLDeviceKind>::new(device_handle) }.into_retained();

    // layer.device = MTLDevice
    layer.setDevice(device.as_deref());
    let cs_label = apply_layer_color(&layer, color);
    // Games are fullscreen-style — no alpha blending with desktop.
    layer.setOpaque(true);
    // Gravity decides what Core Animation does when the drawable is not the
    // size of the layer's backing store. We never leave it that way on
    // purpose: `drawableSize` stays at its default (the layer's own
    // `bounds × contentsScale`) and present resolves the back buffer onto
    // it, so the composite pass is a 1:1 copy and this setting is inert.
    // It is here for the frames where a resize has changed the layer but
    // our next drawable has not caught up yet: aspect-fit centres the
    // stale frame rather than distorting it, and the bars are the layer's
    // own background, which is why it gets an explicit opaque black.
    //
    // SAFETY: `kCAGravityResizeAspect` is a CoreAnimation string constant
    // with static storage duration — reading it is a load of an immutable
    // global the framework initialised before `main`.
    let gravity = unsafe { kCAGravityResizeAspect };
    layer.setContentsGravity(gravity);
    let backdrop = CGColor::new_generic_gray(0.0, 1.0);
    layer.setBackgroundColor(Some(&backdrop));
    // `framebufferOnly = false` is slower than `true`, but required for
    // guest compat: D3D9 games commonly GetBackBuffer + StretchRect,
    // lock, or read the backbuffer.
    layer.setFramebufferOnly(false);
    // We always disable Metal-side vsync and instead throttle presents
    // via `presentDrawable:afterMinimumDuration:` set to `1/panel_max_hz`
    // when the guest asked for vsync. On a fixed-Hz panel that matches
    // the old "snap to vblank" cadence; on a ProMotion panel the system
    // adapts the panel rate down to whatever sub-max cadence the API
    // thread sustains under the cap (transparent VRR) — fractional
    // production rates land at their actual rate instead of being
    // rounded down to the next vsync divisor. PE-side
    // `D3DPRESENT_INTERVAL_*` mapping (`display_sync_for`): DEFAULT/ONE
    // → vsync requested, IMMEDIATE → free-run. Non-1:1 ratios still
    // fall through to vsync-requested with a one-shot warn at the call
    // site. The user's `present.maxFps` ceiling rides the same
    // throttle: the lower rate wins, and it also bounds the
    // IMMEDIATE free-run. The duration itself is derived by the caller and
    // kept on the attachment record.
    layer.setDisplaySyncEnabled(false);
    // 3 explicit drawables; 2 starves at 120 Hz under jitter.
    layer.setMaximumDrawableCount(3);
    // Default true; surface stalls surface as errors, not hangs.
    layer.setAllowsNextDrawableTimeout(true);
    // Default false; no AppKit surface sync needed.
    layer.setPresentsWithTransaction(false);
    // The drawable is the layer's own backing store, never the guest's
    // back-buffer size: a drawable the layer has to rescale into its
    // backing store is a second resample, after whatever present already
    // did, with a phase we do not control. Owning the resample ourselves is
    // what keeps the frame on the pixel grid the screen actually has.
    // Present re-syncs this before every `nextDrawable`; the push here is
    // so the first frame does not have to.
    sync_drawable_size(&layer);
    //
    // Confirm the install. `colorspace` is the label the SDR/HDR
    // applier picked at install time — distinguishes "screen profile
    // (standard-range)" from "kCGColorSpaceSRGB (fallback)" etc. Many
    // calibrated panel profiles have no `CGColorSpaceCopyName` value,
    // so we don't query the layer back here — the applier's label is
    // the source of truth.
    let pf = layer.pixelFormat();
    let wants = layer.wantsExtendedDynamicRangeContent();
    info!(
        target: LOG_TARGET,
        "present: pixelFormat={pf:?} wantsEDR={wants} colorspace={cs_label}",
    );
    log_layer_geometry(&layer, width, height);
    // Keep `device_handle` alive via local — the original was a raw
    // pointer parameter; the local `device` retained it briefly.
    drop(device);
}

/// Report the layer's geometry, and warn when present will have to resample.
///
/// The four numbers that decide whether a frame reaches the screen on the
/// pixel grid it was drawn on: what the guest asked for, the layer's bounds
/// in points, its `contentsScale`, and the drawable size those two imply.
/// Reading them costs one log line at attach and turns "the image looks
/// shifted" into a question with an answer in it.
///
/// The second line fires when the guest's back buffer is not the native
/// drawable size, which is exactly the condition under which present
/// resamples. That is a normal thing for it to do (`render.scale` asks for
/// it deliberately, and D3D9 windowed present stretches a back buffer into
/// a client area of a different size), so it stays informational; what it
/// rules out is resampling every frame *silently*.
fn log_layer_geometry(layer: &objc2_quartz_core::CAMetalLayer, width: u32, height: u32) {
    let bounds = layer.bounds();
    let scale = layer.contentsScale();
    let drawable = layer.drawableSize();
    info!(
        target: LOG_TARGET,
        "present: guest {width}x{height}, layer {:.0}x{:.0}pt @{scale:.2}x, drawable {:.0}x{:.0}",
        bounds.size.width, bounds.size.height, drawable.width, drawable.height,
    );
    let (native_w, native_h) = natural_drawable_size(layer);
    if native_w != width || native_h != height {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: back buffer {width}x{height} onto a {native_w}x{native_h} drawable, \
             so present resamples every frame",
        );
    }
}

/// The drawable size the layer's own geometry asks for: `bounds × contentsScale`.
///
/// `(0, 0)` while the view has no frame yet, which the callers treat as "no
/// answer" rather than a size.
fn natural_drawable_size(layer: &objc2_quartz_core::CAMetalLayer) -> (u32, u32) {
    let bounds = layer.bounds();
    let scale = layer.contentsScale();
    (
        bounded_cast::f64_to_u32_saturating((bounds.size.width * scale).round()),
        bounded_cast::f64_to_u32_saturating((bounds.size.height * scale).round()),
    )
}

/// Point `drawableSize` at the layer's own backing store, and say whether it moved.
///
/// `CAMetalLayer` documents `drawableSize` as defaulting to `bounds ×
/// contentsScale`, but it captures that once and does **not** follow the
/// layer afterwards: a freshly created wine metal view reports a real
/// `bounds` beside a `0x0` `drawableSize`. So the value has to be pushed,
/// and pushed again whenever the window resizes.
///
/// Present calls this before every `nextDrawable`, which is what keeps the
/// drawable equal to the backing store without waiting for a `WM_SIZE` to
/// make its way through the guest. Pushing is not free (the layer drops its
/// drawable pool), hence the compare first. Degenerate geometry is left
/// alone rather than written as a zero size Metal would reject.
///
/// Reading `bounds`/`contentsScale` off the main thread races an in-flight
/// `AppKit` resize; the cost of losing that race is one frame at the previous
/// size, corrected on the next present.
pub fn sync_drawable_size(layer: &objc2_quartz_core::CAMetalLayer) {
    use objc2_core_foundation::CGSize;

    let (native_w, native_h) = natural_drawable_size(layer);
    if native_w == 0 || native_h == 0 {
        return;
    }
    let current = layer.drawableSize();
    let width = f64::from(native_w);
    let height = f64::from(native_h);
    if (current.width - width).abs() <= 0.0 && (current.height - height).abs() <= 0.0 {
        return;
    }
    layer.setDrawableSize(CGSize { width, height });
}

/// Apply the layer's colour configuration, and report the colorspace label it picked.
///
/// Pixel format, colorspace, EDR opt-in and layer name are one decision (see
/// [`LayerMode`]), so they are written together and from one place: attach
/// configures a fresh layer through here, and the display-follow path
/// re-applies the other configuration through the same code when the window
/// lands on a display of the other class.
///
/// The colorspace policy comes from `mtld3d.conf::color.space`.
/// `Passthrough` (the default) tags the screen's own profile (SDR via
/// `copy_with_standard_range`, HDR via `extended_linearized`), so D3D9 values
/// land at the panel's native primaries, max vibrance per display.
/// `Accurate` tags the sRGB family for both paths instead; D3D9 art is
/// overwhelmingly authored against sRGB primaries, so an sRGB-tagged layer
/// lets Core Animation colour-manage to the panel and render
/// designer-intended hues rather than the display's gamut stretch.
///
/// **Main thread only** — these are the compositor-observed setters that have
/// to land inside a main-thread `CATransaction` commit, or the `WindowServer`
/// EDR-mode arbiter can sample the layer between the write and the commit.
fn apply_layer_color(layer: &objc2_quartz_core::CAMetalLayer, color: LayerColorRefs<'_>) -> String {
    use objc2_foundation::NSString;
    use objc2_metal::MTLPixelFormat;

    let LayerColorRefs {
        mode,
        color_space,
        native_colorspace,
        screen_name,
        screen_profile_name,
    } = color;
    let hdr = mode == LayerMode::Hdr;
    // The HDR surface gives the present pass linear float pixels that the
    // compositor maps directly to the panel's EDR headroom.
    layer.setPixelFormat(if hdr {
        MTLPixelFormat::RGBA16Float
    } else {
        MTLPixelFormat::BGRA8Unorm
    });
    let cs_label = match (mode, color_space) {
        (LayerMode::Hdr, ColorSpacePolicy::Passthrough) => apply_hdr_colorspace_passthrough(
            layer,
            native_colorspace,
            screen_name,
            screen_profile_name,
        ),
        (LayerMode::Hdr, ColorSpacePolicy::Accurate) => apply_hdr_colorspace_accurate(layer),
        (LayerMode::Sdr, ColorSpacePolicy::Passthrough) => apply_sdr_colorspace_passthrough(
            layer,
            native_colorspace,
            screen_name,
            screen_profile_name,
        ),
        (LayerMode::Sdr, ColorSpacePolicy::Accurate) => apply_sdr_colorspace_accurate(layer),
    };
    // EDR opt-in. macOS only routes the layer's contents through the panel's
    // HDR headroom when this is set; without it the panel clamps to SDR
    // paper-white even if the surface format and colorspace are HDR-capable.
    // Written on both paths so an HDR layer that moves onto an SDR display
    // gives the opt-in back rather than keeping a claim it cannot honour.
    layer.setWantsExtendedDynamicRangeContent(hdr);
    // Label the layer so Xcode GPU captures show `mtld3d-layer-hdr` vs
    // `mtld3d-layer-sdr` — useful when triaging HDR-specific bugs.
    layer.setName(Some(&NSString::from_str(if hdr {
        "mtld3d-layer-hdr"
    } else {
        "mtld3d-layer-sdr"
    })));
    cs_label
}

/// The screen-profile pair the layer colorspace appliers need.
///
/// `CGColorSpace` for `setColorspace`, plus the user-facing profile label the
/// post-configuration log line carries. The label is classified from the ICC
/// primaries where they are readable, because a profile's description can be
/// renamed in `ColorSync` Utility while the chromaticities are the physical
/// thing the panel renders into; it falls back to `NSColorSpace.localizedName`.
fn screen_color_profile(
    screen: &objc2_app_kit::NSScreen,
) -> (Option<Retained<CGColorSpace>>, Option<String>) {
    let ns_cs = screen.colorSpace();
    let cg_cs = ns_cs.as_ref().and_then(|n| n.CGColorSpace());
    let profile_name = cg_cs
        .as_deref()
        .and_then(classify_icc_gamut)
        .map(ToOwned::to_owned)
        .or_else(|| {
            ns_cs
                .as_ref()
                .and_then(|n| n.localizedName().map(|s| s.to_string()))
        });
    (cg_cs, profile_name)
}

/// Follow the record's window onto a screen of the other EDR class. **Main thread only.**
///
/// A session can move between displays: the window is dragged to another
/// screen, an external monitor is attached or unplugged, or the machine is
/// docked. Without this the layer keeps whatever the screen at attach time
/// asked for, which leaves an HDR layer driving an SDR panel (macOS
/// gamut-compresses it, so it looks plausible rather than broken) or an SDR
/// layer on a panel with headroom to spare.
///
/// The decision is the screen's *static* EDR ceiling, never the live
/// headroom: the live value drops with brightness and thermal state, and
/// re-formatting the layer on a brightness step would drop the drawable pool
/// for a value the present shader already tracks per frame. Reconfiguring is
/// therefore rare, and each one gets a log line.
fn follow_screen_layer_mode(
    att: &Arc<Attachment>,
    screen: &objc2_app_kit::NSScreen,
    mtm: MainThreadMarker,
) {
    let applied = if att.hdr_active() {
        LayerMode::Hdr
    } else {
        LayerMode::Sdr
    };
    let potential = screen.maximumPotentialExtendedDynamicRangeColorComponentValue();
    // The layer already matches this screen, which is every poll of a session
    // that stays on one display.
    let Some(mode) = layer_mode_change(applied, potential, att.hdr_enable_requested()) else {
        return;
    };
    // A record retired since the refresh was queued leaves no layer to
    // reconcile.
    let Some(layer) = attachment::retain_layer(att, mtm) else {
        return;
    };
    let color_space = att.color_space();
    let (native_colorspace, screen_profile_name) = screen_color_profile(screen);
    let screen_name = screen.localizedName().to_string();
    let cs_label = apply_layer_color(
        &layer,
        LayerColorRefs {
            mode,
            color_space,
            native_colorspace: native_colorspace.as_deref(),
            screen_name: Some(&screen_name),
            screen_profile_name: screen_profile_name.as_deref(),
        },
    );
    att.set_hdr_active(mode == LayerMode::Hdr);
    let pf = layer.pixelFormat();
    let wants = layer.wantsExtendedDynamicRangeContent();
    info!(
        target: LOG_TARGET,
        "hdr: window moved onto '{screen_name}' (potential={potential:.2}×), layer reconfigured: \
         pixelFormat={pf:?} wantsEDR={wants} colorspace={cs_label}",
    );
}

/// Re-derive the present throttle for the screen the window is on. **Main thread only.**
///
/// The throttle is a function of the panel's refresh ceiling, so a window
/// dragged from a 120 Hz panel onto a 60 Hz one keeps presenting at an 8.3 ms
/// floor until this runs, and the reverse leaves a 16.6 ms floor on a panel
/// that could show twice as many frames. The guest's vsync request and the
/// user's frame cap are not what moved, so they come from the pacing latched
/// at attach and at every Reset.
///
/// A session that stays on one display derives the duration it already has,
/// and nothing is stored or logged.
fn follow_screen_present_throttle(att: &Arc<Attachment>, screen: &objc2_app_kit::NSScreen) {
    let pacing = unpack_pacing(att.pacing_bits());
    let panel_max_hz = screen_max_hz(screen);
    let applied = att.min_present_duration_sec();
    let Some(seconds) = min_present_duration_change(applied, panel_max_hz, &pacing) else {
        return;
    };
    att.set_min_present_duration(seconds);
    info!(
        target: LOG_TARGET,
        "present: '{}' tops out at {panel_max_hz:.0} Hz, minimum present duration \
         {applied:.5}s -> {seconds:.5}s (vsync={} maxFps={})",
        screen.localizedName(),
        pacing.vsync_requested,
        pacing.max_fps,
    );
}

/// Re-derive the cursor scale from the Wine layer's `contentsScale`. **Main thread only.**
///
/// The PE side drives the cursor upscale from it. winemac sets the scale from
/// its retina mode, so this only ever changes with that mode, but the read is
/// one property and rides the same reconciliation as everything else the
/// display decides.
///
/// A session whose mode stays put reads back the scale it already published,
/// and nothing is stored or logged.
fn follow_layer_backing_scale(att: &Arc<Attachment>, mtm: MainThreadMarker) {
    let Some(layer) = attachment::retain_layer(att, mtm) else {
        return;
    };
    let applied = att.backing_scale();
    let Some(scale) = backing_scale_change(applied, layer.contentsScale()) else {
        return;
    };
    att.set_backing_scale(scale);
    attachment::publish_backing_scale(att, scale);
    info!(
        target: LOG_TARGET,
        "present: the Wine layer's contents scale is {scale}x (was {applied}x), cursor scale republished to the guest",
    );
}

/// Set the SDR layer colorspace under the `Passthrough` policy.
///
/// Uses Apple's `CGColorSpaceCreateCopyWithStandardRange` on the screen's
/// profile — for SDR (non-extended) source profiles that's effectively
/// identity, for HDR/PQ source profiles (TV in HDR mode reporting
/// `kCGColorSpaceITUR_2100_PQ`) it returns the gamma-encoded SDR counterpart
/// so we never tag a BGRA8 layer with a PQ profile (which would EOTF-double
/// and go dark). Falls back to `kCGColorSpaceSRGB` only when no screen
/// profile is reachable at all.
fn apply_sdr_colorspace_passthrough(
    layer: &objc2_quartz_core::CAMetalLayer,
    native_colorspace: Option<&CGColorSpace>,
    screen_name: Option<&str>,
    screen_profile_name: Option<&str>,
) -> String {
    if let Some(cs) = native_colorspace {
        // `copy_with_standard_range` is the Apple-supplied "give me the
        // SDR-range equivalent of this profile" function — handles
        // calibrated panel profiles, PQ→SDR demotion, and named
        // profiles uniformly. No name matching, no heuristics.
        let sdr_cs = cs.copy_with_standard_range();
        layer.setColorspace(Some(&sdr_cs));
        return format!(
            "'{}' (standard-range)",
            screen_profile_name.unwrap_or("<unnamed screen profile>"),
        );
    }
    // No screen profile reachable — fall back to color-managed sRGB
    // with a loud warn so the user's log identifies the degenerate path.
    match screen_name {
        None => mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: SDR colorspace fallback to sRGB — NSView pointer was null at \
             AttachMetalLayer. On Apple wide-gamut panels colors will look less punchy than expected.",
        ),
        Some(name) => mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: hash_screen_key(name),
            "present: SDR colorspace fallback to sRGB on '{name}' — \
             NSScreen.colorSpace was unavailable.",
        ),
    }
    // SAFETY: `kCGColorSpaceSRGB` is a process-lifetime CoreGraphics
    // extern static; Apple guarantees it's valid for the entire process
    // lifetime.
    let srgb_name = unsafe { objc2_core_graphics::kCGColorSpaceSRGB };
    let Some(cs) = CGColorSpace::with_name(Some(srgb_name)) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: CGColorSpaceCreateWithName(kCGColorSpaceSRGB) returned nil — \
             layer keeps default colorspace (washout on wide-gamut displays).",
        );
        return "(setColorspace failed)".to_owned();
    };
    layer.setColorspace(Some(&cs));
    "kCGColorSpaceSRGB (fallback)".to_owned()
}

/// Set the SDR layer colorspace under the `Accurate` policy.
///
/// Tag the layer with plain `kCGColorSpaceSRGB` regardless of the display
/// profile. `CoreAnimation` then colour-manages the sRGB-tagged surface
/// onto the panel's gamut at composite time, so guest assets authored
/// against sRGB render with their designer-intended hues. No screen
/// profile reachable is not a degenerate path here — the result is
/// exactly what the user asked for either way.
fn apply_sdr_colorspace_accurate(layer: &objc2_quartz_core::CAMetalLayer) -> String {
    // SAFETY: `kCGColorSpaceSRGB` is a process-lifetime CoreGraphics
    // extern static; Apple guarantees it's valid for the entire process
    // lifetime.
    let srgb_name = unsafe { objc2_core_graphics::kCGColorSpaceSRGB };
    let Some(cs) = CGColorSpace::with_name(Some(srgb_name)) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: color.space=accurate but CGColorSpaceCreateWithName(kCGColorSpaceSRGB) returned nil — \
             layer keeps default colorspace.",
        );
        return "(setColorspace failed)".to_owned();
    };
    layer.setColorspace(Some(&cs));
    "kCGColorSpaceSRGB (accurate)".to_owned()
}

/// Set the HDR layer colorspace under the `Passthrough` policy.
///
/// Uses Apple's `CGColorSpaceCreateExtendedLinearized` on the screen's
/// profile — constructs the correct extended-linear variant whether
/// the input is a calibrated panel profile, a named `kCG*` profile,
/// or a PQ/HLG HDR profile. No name matching. Falls back to
/// `kCGColorSpaceExtendedLinearDisplayP3` when the linearisation API
/// can't produce one (rare; some non-RGB profiles).
fn apply_hdr_colorspace_passthrough(
    layer: &objc2_quartz_core::CAMetalLayer,
    native_colorspace: Option<&CGColorSpace>,
    screen_name: Option<&str>,
    screen_profile_name: Option<&str>,
) -> String {
    if let Some(cs) = native_colorspace
        && let Some(hdr_cs) = cs.extended_linearized()
    {
        layer.setColorspace(Some(&hdr_cs));
        return format!(
            "'{}' (extended-linearized)",
            screen_profile_name.unwrap_or("<unnamed screen profile>"),
        );
    }
    // Either no screen profile or `extended_linearized` returned None
    // (e.g. non-RGB source) — fall back to the previous default.
    match screen_name {
        None => mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: HDR colorspace fallback to ExtendedLinearDisplayP3 — no NSView/screen profile reachable.",
        ),
        Some(name) => mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: hash_screen_key(name),
            "present: HDR colorspace fallback to ExtendedLinearDisplayP3 on '{name}' — \
             CGColorSpaceCreateExtendedLinearized could not produce an extended-linear variant.",
        ),
    }
    // SAFETY: `kCGColorSpaceExtendedLinearDisplayP3` is a
    // process-lifetime CoreGraphics extern static.
    let p3_name = unsafe { objc2_core_graphics::kCGColorSpaceExtendedLinearDisplayP3 };
    let Some(cs) = CGColorSpace::with_name(Some(p3_name)) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearDisplayP3) returned nil — \
             layer keeps default colorspace (no HDR boost).",
        );
        return "(setColorspace failed)".to_owned();
    };
    layer.setColorspace(Some(&cs));
    "kCGColorSpaceExtendedLinearDisplayP3 (fallback)".to_owned()
}

/// Set the HDR layer colorspace under the `Accurate` policy.
///
/// Tag the layer with `kCGColorSpaceExtendedLinearSRGB` regardless of the
/// display profile. The extended-linear variant is mandatory for the
/// `RGBA16Float` surface (a non-linear profile on a float surface
/// double-EOTFs and goes dark); pairing it with sRGB primaries means
/// the HDR present pass produces colour-managed sRGB output that the
/// compositor maps to the panel's actual gamut.
fn apply_hdr_colorspace_accurate(layer: &objc2_quartz_core::CAMetalLayer) -> String {
    // SAFETY: `kCGColorSpaceExtendedLinearSRGB` is a process-lifetime
    // CoreGraphics extern static; Apple guarantees it's valid for the
    // entire process lifetime.
    let name = unsafe { objc2_core_graphics::kCGColorSpaceExtendedLinearSRGB };
    let Some(cs) = CGColorSpace::with_name(Some(name)) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: color.space=accurate but CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB) returned nil — \
             layer keeps default colorspace (no HDR boost).",
        );
        return "(setColorspace failed)".to_owned();
    };
    layer.setColorspace(Some(&cs));
    "kCGColorSpaceExtendedLinearSRGB (accurate)".to_owned()
}

/// Stable u64 hash of a screen name for `log_once_warn_by!` key.
///
/// FNV-1a — small, no std-hash variability, distinct names rarely
/// collide. We only need uniqueness across a handful of screens per
/// process.
fn hash_screen_key(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Classify the screen's gamut from its ICC profile primaries.
///
/// Returns a static name (`"sRGB"`, `"DisplayP3"`, `"BT.2020"`) when
/// the primaries match one of the standard families within tolerance,
/// `Some("Wide RGB (unknown gamut)")` when the red x sits outside the
/// known buckets, `None` when ICC data isn't available or the profile
/// doesn't carry `rXYZ`/`gXYZ`/`bXYZ` primary tags.
///
/// Why classify from primaries instead of from the profile
/// description: macOS users can rename their display profile in
/// `ColorSync` Utility, and the description string also varies by
/// preset ("Apple XDR Display (P3-1600 nits)" vs "Color LCD" vs
/// vendor-specific names). The chromaticities are the actual physical
/// thing — that's what determines the rendered colors regardless of
/// what the profile is *called*.
fn classify_icc_gamut(cs: &CGColorSpace) -> Option<&'static str> {
    let data = CGColorSpace::icc_data(Some(cs))?;
    // SAFETY: We hold the `CFRetained<CFData>` for the duration of
    // this function; `CFData` is immutable from our point of use.
    let bytes = unsafe { data.as_bytes_unchecked() };
    let (red_x, green_y) = icc_red_x_green_y(bytes)?;
    Some(gamut_from_chromaticities(red_x, green_y))
}

/// Walk an ICC profile's tag table for `rXYZ` and `gXYZ` primary tags.
///
/// Parse the `XYZType` payload (signed 16.16 fixed-point), and convert to xy
/// chromaticity. Returns `(red_x, green_y)` — the two most distinguishing
/// coordinates across sRGB / P3 / BT.2020.
fn icc_red_x_green_y(bytes: &[u8]) -> Option<(f32, f32)> {
    // ICC header is 128 bytes, then 4-byte tag count, then 12-byte
    // tag entries (signature[4] + offset[4] + size[4]).
    if bytes.len() < 132 {
        return None;
    }
    let tag_count = u32::from_be_bytes(bytes[128..132].try_into().ok()?) as usize;
    let tag_table_start: usize = 132;
    let tag_table_end = tag_table_start.checked_add(tag_count.checked_mul(12)?)?;
    if tag_table_end > bytes.len() {
        return None;
    }
    let mut red_xyz: Option<(f32, f32, f32)> = None;
    let mut green_xyz: Option<(f32, f32, f32)> = None;
    for i in 0..tag_count {
        let entry = tag_table_start + i * 12;
        let sig = &bytes[entry..entry + 4];
        if sig != b"rXYZ" && sig != b"gXYZ" {
            continue;
        }
        let offset = u32::from_be_bytes(bytes[entry + 4..entry + 8].try_into().ok()?) as usize;
        let size = u32::from_be_bytes(bytes[entry + 8..entry + 12].try_into().ok()?) as usize;
        let end = offset.checked_add(size)?;
        if end > bytes.len() {
            return None;
        }
        let xyz = parse_xyz_tag(&bytes[offset..end])?;
        if sig == b"rXYZ" {
            red_xyz = Some(xyz);
        } else {
            green_xyz = Some(xyz);
        }
    }
    let (rx, ry, rz) = red_xyz?;
    let (gx, gy, gz) = green_xyz?;
    let r_sum = rx + ry + rz;
    let g_sum = gx + gy + gz;
    if r_sum.abs() < 1e-6 || g_sum.abs() < 1e-6 {
        return None;
    }
    Some((rx / r_sum, gy / g_sum))
}

/// ICC `XYZType`: signature(4 = 'XYZ ') + reserved(4) + at least one 12-byte `XYZNumber`.
///
/// Each `XYZNumber` is 3× s15Fixed16Number, big-endian signed 16.16.
fn parse_xyz_tag(data: &[u8]) -> Option<(f32, f32, f32)> {
    if data.len() < 20 {
        return None;
    }
    // s15Fixed16 inputs come from a panel's primary chromaticities,
    // small magnitudes (|XYZ| < 2) — f32 precision is more than enough.
    let s15fixed16 = |slice: &[u8]| -> f32 {
        let raw = i32::from_be_bytes(slice.try_into().expect("4 bytes"));
        bounded_cast::i32_to_f32(raw) / 65536.0
    };
    let x = s15fixed16(&data[8..12]);
    let y = s15fixed16(&data[12..16]);
    let z = s15fixed16(&data[16..20]);
    Some((x, y, z))
}

/// Classify (`red_x`, `green_y`) chromaticities into a known gamut family.
///
/// Tolerances cover both the D65 reference values and the D50-PCS-adapted
/// values ICC profiles actually store. Standard primaries:
/// - sRGB / BT.709:  R=(0.640, 0.330), G=(0.300, 0.600)
/// - `DisplayP3`:      R=(0.680, 0.320), G=(0.265, 0.690)
/// - BT.2020/2100:   R=(0.708, 0.292), G=(0.170, 0.797)
fn gamut_from_chromaticities(red_x: f32, green_y: f32) -> &'static str {
    if red_x > 0.69 || green_y > 0.74 {
        "BT.2020"
    } else if red_x > 0.65 || green_y > 0.64 {
        "DisplayP3"
    } else if red_x > 0.55 {
        "sRGB"
    } else {
        "Wide RGB (unknown gamut)"
    }
}

#[cfg(test)]
mod tests;
