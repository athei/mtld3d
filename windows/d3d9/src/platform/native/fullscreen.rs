//! Display modes from CoreGraphics, and the fullscreen session the native build does not offer.
//!
//! The modes are the main display's own, in pixels. Exclusive fullscreen is
//! refused before the session functions here are reached
//! ([`super::exclusive_fullscreen`]), so they leave every view alone.

use core::ffi::c_void;

use mtld3d_core::{display_mode::ModeRequest, fullscreen_log::LeaveReason};
use mtld3d_unix::f64_to_u32_saturating;
use objc2_core_foundation::{CFArray, CFRetained};
use objc2_core_graphics::{
    CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode, CGDisplayMode, CGMainDisplayID,
};

use crate::{LOG_TARGET, device::LeaveKind, direct3d9::DisplayModeInfo};

/// The depth every mode is reported at: macOS no longer reports one per mode.
const BITS_PER_PEL: u32 = 32;

/// The fullscreen session a device holds; never entered in the native build.
pub struct SavedWindow {
    view: *mut c_void,
}

impl SavedWindow {
    /// The view the session was entered on.
    pub const fn window(&self) -> *mut c_void {
        self.view
    }
}

/// The main display's current mode, or `None` when CoreGraphics reports none.
pub fn current_display_mode() -> Option<DisplayModeInfo> {
    let Some(mode) = CGDisplayCopyDisplayMode(CGMainDisplayID()) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "CGDisplayCopyDisplayMode answered no mode for the main display"
        );
        return None;
    };
    Some(mode_info(&mode))
}

/// Every mode the main display lists, in CoreGraphics' order; the caller dedupes.
pub fn enumerate_display_modes() -> Vec<DisplayModeInfo> {
    // SAFETY: no options dictionary is passed, so there is no generic to get wrong.
    let Some(modes) = (unsafe { CGDisplayCopyAllDisplayModes(CGMainDisplayID(), None) }) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "CGDisplayCopyAllDisplayModes answered no list for the main display"
        );
        return Vec::new();
    };
    // SAFETY: CoreGraphics documents the array's elements as `CGDisplayModeRef`.
    let modes = unsafe { CFRetained::cast_unchecked::<CFArray<CGDisplayMode>>(modes) };
    modes.iter().map(|mode| mode_info(&mode)).collect()
}

/// No monitor handle exists natively; `GetAdapterMonitor` answers null.
pub fn primary_monitor() -> *mut c_void {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "GetAdapterMonitor: the native build has no HMONITOR → NULL"
    );
    core::ptr::null_mut()
}

/// Nothing scales a native view's coordinates by a system DPI, so there is nothing to warn of.
pub const fn warn_if_dpi_scaled(_view: *mut c_void) {}

/// A native view is never sized by a maximize the device would have to follow.
pub const fn is_maximized(_view: *mut c_void) -> bool {
    false
}

/// No window to activate: the host owns its windows.
pub const fn activate_focus_window(_focus: *mut c_void) {}

/// Record a session on `view` without touching it; the shared guard keeps this unreached.
pub fn enter(view: *mut c_void, _manage_window: bool, _mode: Option<ModeRequest>) -> SavedWindow {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "fullscreen requested on view {view:p} in the native build, which offers none; the view \
         is left as it is"
    );
    SavedWindow { view }
}

/// Nothing to re-apply: the session touched nothing.
pub fn update(saved: &SavedWindow, _mode: Option<ModeRequest>) {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "fullscreen kept across Reset on view {:p} in the native build, which offers none",
        saved.view,
    );
}

/// Nothing to restore: the session touched nothing.
pub fn leave(saved: &SavedWindow, _kind: LeaveKind, reason: &LeaveReason) {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "fullscreen left ({reason}) on view {:p}; the native build touched nothing to restore",
        saved.view,
    );
}

/// Leave at once, as there is nothing to put back; `accept` decides.
pub fn try_leave(_saved: &SavedWindow, _kind: LeaveKind, accept: impl FnOnce() -> bool) -> bool {
    accept()
}

/// One CoreGraphics mode as the adapter mode table takes it, in pixels.
fn mode_info(mode: &CGDisplayMode) -> DisplayModeInfo {
    let mode = Some(mode);
    DisplayModeInfo {
        width: u32::try_from(CGDisplayMode::pixel_width(mode)).unwrap_or(u32::MAX),
        height: u32::try_from(CGDisplayMode::pixel_height(mode)).unwrap_or(u32::MAX),
        refresh_hz: f64_to_u32_saturating(CGDisplayMode::refresh_rate(mode).round()),
        bits_per_pel: BITS_PER_PEL,
    }
}
