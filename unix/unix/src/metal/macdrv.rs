use core::ffi::c_void;
use std::sync::{Arc, Mutex};

use log::{debug, info};
use mtld3d_shared::{
    MetalHandle,
    mtl::{ClientSurfaceRequest, ColorSpacePolicy},
    mtl_handle::{CAMetalLayerKind, MTLDeviceKind, NSViewKind},
    render_size::{ScreenSizes, scaled_down_to_panel},
};
use objc2::{
    MainThreadMarker,
    rc::{Retained, autoreleasepool},
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_app_kit::{NSView, NSWindow};
use objc2_core_graphics::CGColorSpace;

use super::{
    bounded_cast,
    layer::{
        DisplayCaps, LayerAttachRequest, PresentPacing, ScreenPixels, apply_shared_settings,
        backing_pixels, min_present_duration, natural_drawable_size, pack_pacing, unpack_pacing,
    },
};
use crate::{LOG_TARGET, metal::handle::IntoRetained};

pub mod attachment;
mod cursor_overlay;
#[cfg_attr(not(native_d3d), path = "macdrv/wine.rs")]
#[cfg_attr(native_d3d, path = "macdrv/native.rs")]
mod host;

use attachment::{AttachFlags, AttachLatches, Attachment};
pub use cursor_overlay::{poll_from_present, set_cursor_overlay};
pub use host::{INSTALLS_CRASH_HANDLER, attach_metal_layer, retire_metal_view};

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
/// attached finds no record, and another device's record is untouched,
/// except that Wine is told to show the client surface of the newest device
/// still attached to the same window, once that device has submitted a
/// frame. Runs on the API thread, a Wine thread, which win32u's client
/// surface calls need.
pub fn detach_metal_layer(view_handle: MetalHandle<NSViewKind>) -> Option<Arc<Attachment>> {
    let view_addr =
        usize::try_from(view_handle.raw()).expect("a 64-bit host addresses every view pointer");
    let att = attachment::unregister(view_addr)?;
    host::unwatch_view(&att);
    cursor_overlay::detach(&att);
    super::gamma::detach(&att);
    // Another device still presenting into this window gets its surface
    // shown again, rather than the window keeping this view's last frame. A
    // device that has not drawn yet presents its own surface at its first
    // frame instead.
    if let Some(calls) = host::SurfaceCalls::load() {
        match attachment::retain_newest_surface_on(att.hwnd(), |surface| calls.retain(surface)) {
            attachment::NewestSurface::Retained(sibling) => {
                calls.present(sibling, "the device left on the window");
                calls.release(sibling);
            }
            attachment::NewestSurface::NotDrawn(sibling) => debug!(
                target: LOG_TARGET,
                "present: client surface {sibling:#x}, the newest left on window {:#x}, waits \
                 for its device's first frame",
                att.hwnd(),
            ),
            attachment::NewestSurface::None => {}
        }
        calls.release(att.client_surface());
    }
    debug!(
        target: LOG_TARGET,
        "present: detached view {:#x} (layer {:#x}); its display state is retired",
        att.view(),
        att.layer(),
    );
    Some(att)
}

/// Have Wine show and size the client surface of the newest device on a view's window.
///
/// The device whose metal view is `view_handle` asks at a `Present` that
/// submitted a frame: its first one after the attach, and the next one after
/// its client view was hidden or its window's client area changed
/// (`reasons`). The call is what marks the device as having drawn. win32u's
/// `client_surface_present` updates the surface's cocoa view to the window's
/// client rect and makes it the one Wine shows, the two things winemac's
/// `nextDrawable` override did on every present. The surface is the newest
/// device's on the record's window, as at a teardown, and only once that
/// device has drawn: its own first frame presents it otherwise. Runs on the
/// API thread, a Wine thread, which win32u's calls need; a view no record
/// names asks for nothing, logged once.
pub fn present_client_surface(view_handle: MetalHandle<NSViewKind>, reasons: ClientSurfaceRequest) {
    let view_addr =
        usize::try_from(view_handle.raw()).expect("a 64-bit host addresses every view pointer");
    let Some(att) = attachment::find(view_addr) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "present: a client surface request names view {view_addr:#x}, which has no \
             attachment record → ignored",
        );
        return;
    };
    att.note_frame_submitted();
    let Some(calls) = host::SurfaceCalls::load() else {
        return;
    };
    match attachment::retain_newest_surface_on(att.hwnd(), |surface| calls.retain(surface)) {
        attachment::NewestSurface::Retained(surface) => {
            calls.present_requested(surface, reasons);
            calls.release(surface);
        }
        attachment::NewestSurface::NotDrawn(surface) => debug!(
            target: LOG_TARGET,
            "present: client surface {surface:#x}, the newest on window {:#x}, waits for its \
             device's first frame ({reasons:?})",
            att.hwnd(),
        ),
        attachment::NewestSurface::None => {}
    }
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

/// Begin tracking the record's window occlusion and its client view's `hidden`.
///
/// `submit_frame` skips the present while the window is fully
/// covered/minimised. When a window is occluded the compositor stops
/// recycling its drawables, so `nextDrawable` would block its full
/// `allowsNextDrawableTimeout` for nothing on screen and back-pressure the
/// whole pipeline up to the guest's render loop. Records the window pointer
/// on `att` and seeds its occlusion from the current state
/// ([`follow_view_window`], which every reconciliation runs again), and
/// installs the (process-lifetime) observers once. The client view the metal
/// view sits in gets a watch of its own (the host's `watch_view`), which
/// asks the device to have Wine show its surface again when Wine hides it.
/// Runs the `AppKit` work on the main thread: `NSView`/`NSWindow` access and
/// the notification center are main-thread affairs, mirroring
/// [`configure_metal_layer`]'s posture. The record is registered before this
/// runs, so the observers can already find it.
fn install_window_tracking(att: &Arc<Attachment>) {
    let att = Arc::clone(att);
    run_on_main_thread_sync(move || {
        let mtm = MainThreadMarker::new().expect("install_window_tracking runs on the main thread");
        if let Some(view) = attachment::retain_view(&att, mtm) {
            host::watch_view(&att, &view, mtm);
            follow_view_window(&att, &view);
        }
        // Installed whether or not the view has a window yet: a reconciliation
        // points the record at the window the view gets later, and the
        // observer has to be there to match it from then on.
        install_occlusion_observer_once();
        host::observe_screen_parameters(mtm);
        cursor_overlay::install_pointer_watch(mtm);
    });
}

/// A layer an attach registered, with what the rest of the attach reads from it.
struct RegisteredLayer {
    att: Arc<Attachment>,
    /// The cursor scale, from the layer's contents scale ([`backing_scale_from`]).
    backing_scale: u32,
    /// The screen's frame at the layer's own scale, which present caps the drawable against.
    backing: Option<(u32, u32)>,
    /// The configuration the layer gets, decided once here.
    mode: LayerMode,
}

/// Register the attachment record of `view` and `layer`, from the screen attach found.
///
/// The record holds the layer and everything the display-follow path needs
/// to reconfigure it for a screen that was not attached yet: the HDR gate
/// and colorspace policy, the guest's vsync ask, the user's frame cap and the
/// scale the PE side is already using. Both hosts call it.
fn register_layer(
    request: &LayerAttachRequest,
    view: usize,
    layer: usize,
    client_surface: usize,
    hint: &DisplayHint,
) -> RegisteredLayer {
    let contents_scale = layer_contents_scale(layer as *mut c_void);
    let backing_scale = backing_scale_from(contents_scale);
    let backing = backing_pixels(hint.frame, contents_scale);
    // From the panel's static potential and `color.hdr.enable`; the user
    // gate stays unix-side from here on.
    let mode = resolve_layer_mode(
        hint.edr_potential,
        hint.screen_name.as_deref(),
        hint.colorspace_flags.contains(ColorspaceFlags::IS_HDR),
        hint.colorspace_flags
            .contains(ColorspaceFlags::IS_WIDE_GAMUT),
        request.hdr_enable,
    );
    let mut flags = AttachFlags::empty();
    flags.set(AttachFlags::HDR_ENABLE_REQUESTED, request.hdr_enable);
    flags.set(AttachFlags::HDR_ACTIVE, mode == LayerMode::Hdr);
    let sink = |ptr: u64| {
        usize::try_from(ptr).expect("PE wire pointer fits host address space (unix is 64-bit)")
    };
    let att = attachment::register(
        view,
        layer,
        &AttachLatches {
            hwnd: request.hwnd,
            flags,
            color_space: request.color_space,
            pacing_bits: pack_pacing(&request.pacing),
            backing_scale,
            backing_scale_sink: sink(request.backing_scale_sink_ptr),
            cursor_kick_sink: sink(request.cursor_kick_sink_ptr),
            client_surface_request_sink: sink(request.client_surface_request_sink_ptr),
            screen_sizes_sink: sink(request.screen_sizes_sink_ptr),
            client_surface,
        },
    );
    RegisteredLayer {
        att,
        backing_scale,
        backing,
        mode,
    }
}

/// Finish an attach: publish to the device, configure the layer and track its window.
///
/// Both hosts call it once the record is registered. The handles name the
/// record's view and layer, which the host keeps alive until the matching
/// retire.
fn finish_attach(
    device_handle: MetalHandle<MTLDeviceKind>,
    request: &LayerAttachRequest,
    hint: DisplayHint,
    registered: RegisteredLayer,
    software_cursor_active: bool,
) -> (
    MetalHandle<NSViewKind>,
    MetalHandle<CAMetalLayerKind>,
    DisplayCaps,
) {
    let RegisteredLayer {
        att,
        backing_scale,
        backing,
        mode,
    } = registered;
    attachment::publish_backing_scale(&att, backing_scale);
    // The screen attach found, so the device's word holds it before the
    // first resize; the display-follow pass republishes it on the main
    // thread when the window reaches another screen.
    attachment::publish_screen_sizes(
        &att,
        &ScreenSizes {
            panel: hint.panel,
            backing,
        },
    );
    let min_present_duration = configure_metal_layer(
        att.layer() as *mut c_void,
        device_handle.raw(),
        request.width,
        request.height,
        &request.pacing,
        hint.panel_max_hz,
        LayerColorConfig {
            mode,
            color_space: request.color_space,
            native_colorspace: hint.native_colorspace,
            screen_name: hint.screen_name,
            screen_profile_name: hint.screen_profile_name,
        },
    );
    att.set_min_present_duration(min_present_duration);
    // One line per attach: the pace a present into this window is held to,
    // by the display when the window is shown and by the presenter when not.
    info!(
        target: LOG_TARGET,
        "present: minimum present duration {min_present_duration:.5}s (panel {:.0} Hz)",
        hint.panel_max_hz,
    );
    // Occlusion tracking, so presents skip the `nextDrawable` timeout while
    // the window is fully covered or minimised, and the host's view watch.
    install_window_tracking(&att);
    // SAFETY: the host retained the view for the attach and keeps it until
    // the matching retire; the PE side holds the handle no longer.
    let view_handle = unsafe { MetalHandle::<NSViewKind>::new(att.view() as u64) };
    // SAFETY: the view hosts the layer, so the layer lives as long as it.
    let layer_handle = unsafe { MetalHandle::<CAMetalLayerKind>::new(att.layer() as u64) };
    let caps = DisplayCaps {
        backing_scale,
        software_cursor_active,
        panel: hint.panel,
        backing,
    };
    (view_handle, layer_handle, caps)
}

/// Point `att` at the window `view` is in now, and seed its occlusion. **Main thread only.**
///
/// The window is not fixed at attach: Wine's Mac driver moves a client view,
/// and the metal view in it, into another Cocoa window when the device
/// window becomes a child (its own Cocoa window is destroyed) or moves to
/// another toplevel. The occlusion observer matches notifications by the
/// window the record holds, so a record left on the old window would miss
/// every change of the new one. A view in no window clears the pointer and
/// counts as not occluded, so a present is never suppressed on a guess; the
/// next reconciliation that finds a window seeds it. The same window keeps
/// the value the observer last wrote.
fn follow_view_window(att: &Attachment, view: &objc2_app_kit::NSView) {
    use objc2_app_kit::NSWindowOcclusionState;

    let window = view.window();
    let current = window
        .as_ref()
        .map_or(0, |window| Retained::as_ptr(window) as usize);
    let stored = att.window();
    let Some(current) = window_change(stored, current) else {
        return;
    };
    let occluded = window.as_ref().is_some_and(|window| {
        !window
            .occlusionState()
            .contains(NSWindowOcclusionState::Visible)
    });
    att.follow_window(current, occluded);
    info!(
        target: LOG_TARGET,
        "present: metal view {:#x} of window {:#x} follows Cocoa window {current:#x}, was \
         {stored:#x} (occluded {occluded})",
        att.view(),
        att.hwnd(),
    );
}

/// The window a record has to follow its view into, `None` while it holds that window already.
///
/// `stored` is the window address the record holds and `current` the one the
/// view sits in now, `0` for none. Compared by address, as
/// the Wine host's `kept_view_is_in_window` compares a kept view's window.
const fn window_change(stored: usize, current: usize) -> Option<usize> {
    if stored == current {
        None
    } else {
        Some(current)
    }
}

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

/// Whether the screen configuration differs from the one seen last, which it then becomes.
///
/// **Main thread only.** The hosts' screen-parameter observers ask it, so a
/// headroom step, which leaves the configuration as it was, reconciles nothing.
fn screen_configuration_changed(mtm: objc2::MainThreadMarker) -> bool {
    // Last configuration a screen-parameter observer saw.
    static LAST_SCREEN_CONFIGURATION: Mutex<Option<ScreenConfiguration>> = Mutex::new(None);

    let config = current_screen_configuration(mtm);
    let mut last = LAST_SCREEN_CONFIGURATION
        .lock()
        .expect("screen-configuration mutex poisoned");
    let changed = last.as_ref() != Some(&config);
    if changed {
        *last = Some(config);
    }
    changed
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

/// Round and clamp the layer's `contentsScale` into the range the PE side takes.
///
/// Under Wine, winemac sets the metal layer's `contentsScale` to 2 in retina
/// mode and 1 otherwise, whatever the display's own factor: that is the
/// number of Wine pixels per point, and so how much smaller than a point the game's cursor
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

/// The `contentsScale` of the device's metal layer.
///
/// A plain property read. Under Wine it is 2 in retina mode and 1 otherwise,
/// set by winemac; the native host sets it from the window's backing factor
/// (its `follow_host_backing`).
fn layer_contents_scale(layer: *mut c_void) -> f64 {
    // SAFETY: `layer` is the `CAMetalLayer` pointer the host handed out, alive
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
/// All but the panel and the frame drive layer configuration unix-side; those
/// two travel back to the PE side, which sizes a back buffer larger than the
/// panel by them (`render.panelFit`).
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
    /// The native pixel size of the screen's panel, `None` = unknown.
    panel: Option<(u32, u32)>,
    /// The screen's frame in points, `None` = unknown.
    frame: Option<(u32, u32)>,
}

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

/// Run a closure synchronously on `AppKit`'s main thread.
///
/// Waits for completion, through Wine's own `OnMainThread` (the
/// `on_main_thread` entry of the `macdrv_functions` table) where the table
/// has it, and through libdispatch's main queue otherwise. Wine's door is the
/// one to take: winemac's main thread, while it waits for a Wine thread to
/// answer a query (a resize, the min/max info, the pasteboard), runs only a
/// private run-loop mode that never drains the main queue, so a bare
/// `dispatch_sync` from that Wine thread stalls both until the query times
/// out and fails. `OnMainThread` queues the work on Wine's request source,
/// which that mode does run, and while the calling thread waits it answers
/// the queries addressed to it, as every synchronous request Wine itself
/// makes does. A Wine whose entry returns without running the work
/// (`CrossOver`'s arm64 build publishes it as a stub) gets the main queue
/// instead, for that hop and every later one, and so does the native host,
/// which has no such table.
///
/// Apple documents compositor-impacting `CALayer` setters
/// (`wantsExtendedDynamicRangeContent`, `colorspace`, `pixelFormat`)
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
/// A caller already on the main thread, a native host's among them, runs the
/// closure in place.
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
        // `run_through_wine` or `dispatch_sync_f`, which pass it through to
        // the worker function unchanged while the caller waits.
        let ctx = unsafe { &mut *(ctx.cast::<CallCtx<F>>()) };
        if let Some(f) = ctx.f.take() {
            autoreleasepool(|_| f());
        }
    }
    // A caller on the main thread runs the work in place: a hop to the queue
    // or the request source that thread drains could never end.
    if MainThreadMarker::new().is_some() {
        autoreleasepool(|_| f());
        return;
    }
    let mut ctx = CallCtx { f: Some(f) };
    let ctx_ptr = (&raw mut ctx).cast::<c_void>();
    if host::run_through_host_main_thread(thunk::<F>, ctx_ptr) {
        return;
    }
    // SAFETY: `_dispatch_main_q` is libSystem's main-queue singleton —
    // a valid `dispatch_queue_t` for the process lifetime. `&mut ctx`
    // is valid until this function returns, and `dispatch_sync_f` is
    // synchronous, so the thunk runs before we drop `ctx`.
    unsafe {
        let main_q = (&raw const _dispatch_main_q).cast_mut().cast::<c_void>();
        dispatch_sync_f(main_q, ctx_ptr, thunk::<F>);
    }
}

/// `run_on_main_thread_sync` for another crate: the closure may change threads.
pub fn run_on_main_thread<F: FnOnce() + Send>(f: F) {
    run_on_main_thread_sync(f);
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

/// Gather the `NSScreen` properties of the screen the bound view lives on.
///
/// Reads the screen's `colorSpace`, EDR potential, refresh ceiling and
/// `localizedName`; returns them as a [`DisplayHint`]. All of them drive
/// layer configuration unix-side; the cursor scale the PE side consumes comes
/// from the layer instead (see [`layer_contents_scale`]).
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
        // SAFETY: the host hands us a retained NSView*; `Retained::retain`
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
    let panel = screen
        .as_deref()
        .and_then(screen_display_id)
        .and_then(native_panel_pixels);
    let frame = screen.as_deref().and_then(|s| {
        let size = s.frame().size;
        let frame = (
            bounded_cast::f64_to_u32_saturating(size.width.round()),
            bounded_cast::f64_to_u32_saturating(size.height.round()),
        );
        (frame.0 != 0 && frame.1 != 0).then_some(frame)
    });
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
        panel,
        frame,
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
/// when every window may have landed on another panel. A display id can name
/// another panel after a reconfiguration, so each record's cached panel
/// lookup is forgotten first.
fn refresh_all_on_main() {
    for att in attachment::live() {
        att.set_panel_display(0);
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
/// The walk starts from whichever window the view is in *now*, which the
/// record follows ([`follow_view_window`]), and ends on whichever screen that
/// window is on now, so it is also where everything the display decides is
/// reconciled against that screen for this record: the pixel sizes that cap
/// the drawable ([`follow_screen_pixels`]), the layer's own
/// configuration ([`follow_screen_layer_mode`]), the present throttle
/// ([`follow_screen_present_throttle`]), the backing scale the PE side
/// consumes ([`follow_layer_backing_scale`]) and the window layers a covering
/// metal view hides (the host's `clear_covered_window_layers`). A record that was
/// retired between the queueing and the run finds no view and does nothing.
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
    host::observe_screen_parameters(mtm);
    follow_view_window(att, &view_obj);
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
        // The scale goes first, so the screen sizes published next are at the
        // layer's new scale when the resize asked for below reads them.
        let rescaled = host::follow_host_backing(att, &view_obj, mtm);
        follow_screen_pixels(att, screen, mtm);
        follow_screen_layer_mode(att, screen, mtm);
        follow_screen_present_throttle(att, screen);
        follow_layer_backing_scale(att, mtm);
        if rescaled {
            let asked = attachment::request_client_surface(
                att.view(),
                ClientSurfaceRequest::CLIENT_AREA_CHANGED,
                false,
                view_pixels(&view_obj),
            );
            debug!(target: LOG_TARGET, "present: rescaled; asked the device to follow={asked}");
        }
        host::clear_covered_window_layers(&view_obj, mtm);
    }
    log_headroom_change_if_any(att, headroom, &view_obj, mtm);
    cursor_overlay::reconcile_on_main();
}

/// `view`'s bounds in backing pixels, `None` for an empty view.
pub fn view_pixels(view: &NSView) -> Option<(u32, u32)> {
    let backing = view.convertRectToBacking(view.bounds());
    let pixels = (
        bounded_cast::f64_to_u32_saturating(backing.size.width.round()),
        bounded_cast::f64_to_u32_saturating(backing.size.height.round()),
    );
    Some(pixels).filter(|&(width, height)| width != 0 && height != 0)
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
    // because the host retains the underlying Obj-C objects for the
    // lifetime of the view. Cast the layer pointer through `u64` to
    // strip the `!Send` so the `move` closure compiles; reconstruct
    // on the main thread. Sound on a 64-bit host: pointers fit in `u64`
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
    use objc2_quartz_core::CAMetalLayer;

    // Cast the raw `*mut c_void` from the host into typed
    // `Retained<CAMetalLayer>`. Using typed objc2 setters means a
    // typo in a selector name (e.g.
    // `setWantsExtendedDynamicRange` vs `…RangeContent`) becomes a
    // compile error rather than a runtime `unrecognized selector` crash.
    //
    // SAFETY: `layer` is the `CAMetalLayer` pointer the host handed us;
    // `Retained::retain` bumps the refcount via standard Cocoa semantics.
    let Some(layer) = (unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }) else {
        return;
    };
    // SAFETY: device_handle is a previously-retained MTLDevice address.
    let device = unsafe { MetalHandle::<MTLDeviceKind>::new(device_handle) }.into_retained();

    // layer.device = MTLDevice
    layer.setDevice(device.as_deref());
    let cs_label = apply_layer_color(&layer, color);
    apply_shared_settings(&layer);
    crate::hud_state::show_row(&layer);
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

/// Re-derive the cursor scale from the layer's `contentsScale`. **Main thread only.**
///
/// The PE side drives the cursor upscale from it. Under Wine, winemac sets
/// the scale from its retina mode, so it only ever changes with that mode;
/// the read is
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
        "present: the layer's contents scale is {scale}x (was {applied}x), cursor scale republished to the guest",
    );
}

/// Publish the panel size and frame of `screen`, which cap the drawable. **Main thread only.**
///
/// The frame is in points; the presenter takes it to the layer's pixels at
/// the `contentsScale` it reads per present. The panel lookup reruns only
/// when the window's display id changes or a reconfiguration forgot it
/// ([`refresh_all_on_main`]). A screen with no display id or no native mode
/// publishes no panel, and the drawable then keeps the layer's own size.
///
/// The same pass sets the layer's magnification filter: linear while the
/// display scales the backing store down to the panel, where a capped
/// drawable is magnified back over the layer, and nearest otherwise, as
/// winemac creates the layer, so a layer at scale 1 on a 2x screen is
/// doubled pixel for pixel.
fn follow_screen_pixels(
    att: &Arc<Attachment>,
    screen: &objc2_app_kit::NSScreen,
    mtm: MainThreadMarker,
) {
    use objc2_quartz_core::{kCAFilterLinear, kCAFilterNearest};

    let Some(layer) = attachment::retain_layer(att, mtm) else {
        return;
    };
    let frame = screen.frame().size;
    let frame = (
        bounded_cast::f64_to_u32_saturating(frame.width.round()),
        bounded_cast::f64_to_u32_saturating(frame.height.round()),
    );
    let display = screen_display_id(screen);
    let panel = match display {
        Some(display) if display == att.panel_display() => att.screen_pixels().panel,
        Some(display) => {
            att.set_panel_display(display);
            native_panel_pixels(display)
        }
        None => {
            att.set_panel_display(0);
            None
        }
    };
    let pixels = ScreenPixels {
        panel,
        frame: Some(frame).filter(|&(width, height)| width != 0 && height != 0),
    };
    if att.set_screen_pixels(&pixels) {
        debug!(
            target: LOG_TARGET,
            "present: '{}' panel {:?}, frame {:?} points",
            screen.localizedName(),
            pixels.panel,
            pixels.frame,
        );
    }
    let backing = backing_pixels(pixels.frame, layer.contentsScale());
    // The device sizes its next back buffer against these (`render.panelFit`);
    // published from this main-thread pass whenever it runs, so a window that
    // reached another screen is sized against that screen at its next resize
    // or `Reset`.
    attachment::publish_screen_sizes(
        att,
        &ScreenSizes {
            panel: pixels.panel,
            backing,
        },
    );
    let scaled_down = scaled_down_to_panel(backing, pixels.panel);
    let filter = if scaled_down {
        // SAFETY: a CoreAnimation string constant with static storage duration.
        unsafe { kCAFilterLinear }
    } else {
        // SAFETY: as above.
        unsafe { kCAFilterNearest }
    };
    if *layer.magnificationFilter() != *filter {
        layer.setMagnificationFilter(filter);
        debug!(
            target: LOG_TARGET,
            "present: layer magnification filter {filter} (screen scaled down to the panel: \
             {scaled_down})",
        );
    }
}

/// The `CGDirectDisplayID` of the display behind `screen`, if it names one.
fn screen_display_id(screen: &objc2_app_kit::NSScreen) -> Option<u32> {
    let number = screen
        .deviceDescription()
        .objectForKey(&objc2_foundation::NSString::from_str("NSScreenNumber"))?;
    // `NSScreenNumber` is documented as an `NSNumber` holding the `CGDirectDisplayID`.
    let display = number
        .downcast_ref::<objc2_foundation::NSNumber>()?
        .unsignedIntValue();
    (display != 0).then_some(display)
}

/// The native pixel size of the main display's panel, if it reports one.
///
/// The main display is winemac's primary adapter, `\\.\DISPLAY1`. Only
/// CoreGraphics is read, which any thread may call, so this runs on the
/// calling thread.
pub fn main_display_panel_pixels() -> Option<(u32, u32)> {
    native_panel_pixels(objc2_core_graphics::CGMainDisplayID())
}

/// The native pixel size of `display`, if it reports one.
///
/// A display flags one native mode per refresh rate, and with the duplicate
/// low-resolution modes listed also the scaled modes whose pixels are the
/// panel's, so the largest flagged pixel size is taken. Some external 5K
/// displays flag a 3840x2160 mode under a 5120x2880 panel; whether they flag
/// a 5120x2880 one in the duplicate list too is not known, and if not, a
/// drawable there is capped at 3840x2160 whenever the frame fits it.
fn native_panel_pixels(display: u32) -> Option<(u32, u32)> {
    use objc2_core_foundation::{CFBoolean, CFDictionary, CFString};
    use objc2_core_graphics::{
        CGDisplayCopyAllDisplayModes, CGDisplayMode, kCGDisplayShowDuplicateLowResolutionModes,
    };

    /// `kDisplayModeNativeFlag` (`IOKit` `IOGraphicsTypes.h`), which no `objc2` crate binds.
    const DISPLAY_MODE_NATIVE_FLAG: u32 = 0x0200_0000;

    // SAFETY: `kCGDisplayShowDuplicateLowResolutionModes` is a CoreGraphics
    // string constant with static storage duration.
    let key = unsafe { kCGDisplayShowDuplicateLowResolutionModes };
    let options = CFDictionary::<CFString, CFBoolean>::from_slices(&[key], &[CFBoolean::new(true)]);
    // SAFETY: the options map that key to a `CFBoolean`, the type CoreGraphics
    // documents for it; the display id came from `AppKit` or CoreGraphics.
    let modes = unsafe { CGDisplayCopyAllDisplayModes(display, Some(options.as_ref())) }?;
    // SAFETY: CoreGraphics documents every element of the array as a `CGDisplayMode`.
    let modes = unsafe { modes.cast_unchecked::<CGDisplayMode>() };
    modes
        .iter()
        .filter(|mode| CGDisplayMode::io_flags(Some(mode)) & DISPLAY_MODE_NATIVE_FLAG != 0)
        .filter_map(|mode| {
            let width = u32::try_from(CGDisplayMode::pixel_width(Some(&mode))).ok()?;
            let height = u32::try_from(CGDisplayMode::pixel_height(Some(&mode))).ok()?;
            (width != 0 && height != 0).then_some((width, height))
        })
        .max_by_key(|&(width, height)| u64::from(width) * u64::from(height))
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
