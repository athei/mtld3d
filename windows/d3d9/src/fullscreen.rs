//! Display mode and device-window management for a fullscreen device.
//!
//! A fullscreen `CreateDevice` sets the display mode the game asked for
//! through user32, exactly as native D3D9 does, then strips the window's
//! decoration and stretches it over the monitor rect. What it deliberately
//! does **not** do is touch the window's z-order.
//!
//! The mode-set is what keeps the game's coordinate spaces in agreement. A
//! game sizes its rendering and its mouse handling from the mode it picked;
//! with the mode set, the client rect, `GetSystemMetrics`, `GetCursorPos` and
//! every mouse message answer in that mode, so its clicks land where its UI
//! is drawn. (Until 2026-08 the back buffer honored the mode under a
//! monitor-sized window, which left the mouse in monitor space: Half-Life 2's
//! menu missed by the ratio of the two at every sub-native mode.)
//!
//! On Wine the mode-set is meant to be virtual. With `EmulateModeset` on
//! (`HKCU\Software\Wine\X11 Driver`, read by win32u whatever the driver, once
//! per process at start; the launcher and the test prefix set it), win32u
//! leaves the physical display alone, records the mode as the virtual
//! desktop, scales every window rect onto the physical monitor before the
//! driver sees it (uniformly and centred, so a mode of another aspect is
//! letterboxed and the menu bar stays, since the window no longer covers the
//! screen) and maps the pointer back into the mode. The `NSWindow` and the
//! `CAMetalLayer` in it stay display-sized, the back buffer is the mode, and
//! present scales one to the other (`MetalFX` when enlarging), the same
//! resample `render.scale` rides. Without the key Wine's mac driver hands the
//! request to `CGDisplaySetDisplayMode` and the whole desktop switches
//! resolution, which is native behaviour but not what anyone wants on a Mac.
//!
//! Only a settable mode is set, one in the list win32u validates against;
//! the adapter mode table in `direct3d9` is seeded from that list so the two
//! agree by construction, and games enumerate a bounded subset of it. A request that is no display
//! mode at all (native would reject it) follows the client rect instead, and
//! so does a maximized window, where the window manager sizes the window;
//! `render.scale` decides how many pixels are rasterized in those cases. A
//! mode-set user32 refuses anyway falls back to the monitor-covering window
//! with the back buffer at the request, mouse in monitor space, with a
//! warning.
//!
//! The rest of the mode contract follows native: when a fullscreen device
//! loses focus or leaves fullscreen, [`restore_registry_mode`] puts the
//! user's desktop back, and a re-activation sets the mode and re-covers the
//! monitor again. A `CDS_FULLSCREEN` mode-set is also undone by explorer when
//! the process exits, so a crash cannot strand the desktop in the game's mode.
//!
//! The z-order is left to the window manager. Raising the window to the
//! topmost level deadlocks Wine's mac driver: it re-derives the Cocoa window's
//! level and parent while holding winemac's per-window lock and hops to the
//! main thread to do it, so a focus event arriving meanwhile re-enters
//! `NtUserSetWindowPos` on another thread and both stall. See
//! `apply_fullscreen_window`.
//!
//! A device created with `D3DCREATE_NOWINDOWCHANGES` opts out of the window
//! half. The flag hands window management to the app, so the device leaves
//! the window's style, rect and visibility exactly as it found them, in
//! fullscreen as in windowed mode, and never re-covers the monitor behind
//! the app's back. The mode is still set and restored: neither is a window
//! change.
//!
//! Only the primary display is driven, so a device whose window lives on a
//! secondary monitor is covered by the wrong rect. Matching the window's
//! current monitor is a follow-up.

use core::ffi::c_void;

use log::{debug, warn};
use mtld3d_core::display_mode::{ModeRequest, mode_set_attempts};

/// Sub-target shared with the display-enumeration probes in `direct3d9`.
const LOG_TARGET: &str = "mtld3d::d3d9::display";

// ── Win32 FFI ──

/// Win32 `RECT`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// Win32 `POINT`.
#[repr(C)]
struct Point {
    x: i32,
    y: i32,
}

/// Win32 `MONITORINFO`.
#[repr(C)]
struct MonitorInfo {
    cb_size: u32,
    rc_monitor: Rect,
    rc_work: Rect,
    dw_flags: u32,
}

/// `MONITORINFO`'s ABI size, which the API reads back out of `cb_size`.
const MONITOR_INFO_SIZE: u32 = 40;
const _: () = assert!(size_of::<MonitorInfo>() == MONITOR_INFO_SIZE as usize);

/// Win32 `DEVMODEW`, display-device shape (printer fields collapsed).
///
/// Every member is 2- or 4-byte, so the layout is identical on the i686 and
/// `x86_64` Win32 ABIs (no 8-byte members to shift alignment; see the
/// wire-invariant rule on out-param structs). The union in the C header
/// (`dmPosition` + orientation/fixed-output vs. the printer fields) is
/// declared as its display arm, which is what `EnumDisplaySettingsW` fills.
#[repr(C)]
struct DevModeW {
    device_name: [u16; 32],
    spec_version: u16,
    driver_version: u16,
    size: u16,
    driver_extra: u16,
    fields: u32,
    position_x: i32,
    position_y: i32,
    display_orientation: u32,
    display_fixed_output: u32,
    color: i16,
    duplex: i16,
    y_resolution: i16,
    tt_option: i16,
    collate: i16,
    form_name: [u16; 32],
    log_pixels: u16,
    bits_per_pel: u32,
    pels_width: u32,
    pels_height: u32,
    display_flags: u32,
    display_frequency: u32,
    icm_method: u32,
    icm_intent: u32,
    media_type: u32,
    dither_type: u32,
    reserved1: u32,
    reserved2: u32,
    panning_width: u32,
    panning_height: u32,
}

/// `DEVMODEW`'s ABI size, which the API reads back out of `size`.
const DEV_MODE_SIZE: u16 = 220;
const _: () = assert!(size_of::<DevModeW>() == DEV_MODE_SIZE as usize);

/// `EnumDisplaySettingsW` mode selector for the current display mode.
const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
/// `EnumDisplaySettingsW` mode selector for the registry display mode.
const ENUM_REGISTRY_SETTINGS: u32 = 0xFFFF_FFFE;

/// `DEVMODEW.dmFields` bit: `dmBitsPerPel` is set.
const DM_BITSPERPEL: u32 = 0x0004_0000;
/// `DEVMODEW.dmFields` bit: `dmPelsWidth` is set.
const DM_PELSWIDTH: u32 = 0x0008_0000;
/// `DEVMODEW.dmFields` bit: `dmPelsHeight` is set.
const DM_PELSHEIGHT: u32 = 0x0010_0000;
/// `DEVMODEW.dmFields` bit: `dmDisplayFrequency` is set.
const DM_DISPLAYFREQUENCY: u32 = 0x0040_0000;
/// `ChangeDisplaySettingsW` flag: a temporary mode, not written to the registry.
const CDS_FULLSCREEN: u32 = 0x0000_0004;
/// `ChangeDisplaySettingsW` return value for a mode that was applied.
const DISP_CHANGE_SUCCESSFUL: i32 = 0;

/// The mode-list ceiling for [`enumerate_display_modes`].
///
/// Wine's virtual list is a few dozen entries; a driver enumerating without
/// end is a bug this cap turns into a truncated list rather than a hang.
const MAX_ENUMERATED_MODES: u32 = 4096;

/// One display mode as `EnumDisplaySettingsW` reports it.
///
/// `refresh_hz` may be 0 when the driver doesn't report one.
#[derive(Clone, Copy)]
pub struct DisplayModeInfo {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
    pub bits_per_pel: u32,
}

impl DisplayModeInfo {
    const fn from_devmode(dm: &DevModeW) -> Self {
        Self {
            width: dm.pels_width,
            height: dm.pels_height,
            refresh_hz: dm.display_frequency,
            bits_per_pel: dm.bits_per_pel,
        }
    }
}

/// A zeroed `DEVMODEW` with `size` set, ready for `EnumDisplaySettingsW`.
const fn empty_devmode() -> DevModeW {
    // SAFETY: `DevModeW` is all-integer POD, so the all-zero bit pattern is
    // a valid value.
    let mut dm: DevModeW = unsafe { core::mem::zeroed() };
    dm.size = DEV_MODE_SIZE;
    dm
}

/// One `EnumDisplaySettingsW` query on the primary display.
fn enum_display_settings(mode_num: u32, dm: &mut DevModeW) -> bool {
    // SAFETY: null device name selects the primary display; `dm` is a
    // writable `DEVMODEW` with `size` set per the API contract.
    unsafe { EnumDisplaySettingsW(core::ptr::null(), mode_num, dm) != 0 }
}

/// One `ChangeDisplaySettingsW` call on the primary display.
///
/// `None` applies the registry mode, per the API contract.
fn change_display_settings(mode: Option<&mut DevModeW>, flags: u32) -> i32 {
    let dm = mode.map_or(core::ptr::null_mut(), |dm| &raw mut *dm);
    // SAFETY: `dm` is null or an owned `DEVMODEW` with `size` set; user32
    // reads it and writes nothing back.
    unsafe { ChangeDisplaySettingsW(dm, flags) }
}

/// The current or registry display mode of the primary display.
///
/// `mode_num` is `ENUM_CURRENT_SETTINGS` / `ENUM_REGISTRY_SETTINGS`.
fn query_display_mode(mode_num: u32) -> Option<DisplayModeInfo> {
    let mut dm = empty_devmode();
    let ok = enum_display_settings(mode_num, &mut dm);
    if !ok || dm.pels_width == 0 || dm.pels_height == 0 {
        warn!(
            target: LOG_TARGET,
            "EnumDisplaySettingsW({mode_num:#x}) failed (ok={ok}, {}x{})",
            dm.pels_width,
            dm.pels_height
        );
        return None;
    }
    Some(DisplayModeInfo::from_devmode(&dm))
}

/// The primary display's current mode in the Win32 coordinate space.
///
/// This is the same view win32u validates `ChangeDisplaySettingsW` against
/// and derives `GetMonitorInfoW` from, so callers deriving D3D9 display
/// geometry from it agree with the window-management side by construction;
/// reading `NSScreen` instead gave a second source of truth that disagreed
/// on displays where the two scale differently (a CI runner's virtual
/// display reports 2048x1536 through Win32 but 1024x768 through `NSScreen`).
pub fn current_display_mode() -> Option<DisplayModeInfo> {
    query_display_mode(ENUM_CURRENT_SETTINGS)
}

/// Every display mode the primary display enumerates, in Win32's order.
///
/// The list win32u validates `ChangeDisplaySettingsW` against, so a mode
/// taken from here is one a fullscreen device can set. Under Wine's
/// `EmulateModeset` it is a fixed bank of common sizes plus the display's own
/// modes, at every depth Win32 offers; the caller dedupes.
pub fn enumerate_display_modes() -> Vec<DisplayModeInfo> {
    let mut modes = Vec::new();
    let mut dm = empty_devmode();
    for mode_num in 0..MAX_ENUMERATED_MODES {
        if !enum_display_settings(mode_num, &mut dm) {
            return modes;
        }
        modes.push(DisplayModeInfo::from_devmode(&dm));
    }
    warn!(
        target: LOG_TARGET,
        "EnumDisplaySettingsW enumerated {MAX_ENUMERATED_MODES} modes without ending; list truncated"
    );
    modes
}

/// Set `request` as the primary display's mode, `true` on success.
///
/// Compare-first: a mode that is already current is left alone, which keeps
/// a fullscreen `Reset` at the same mode and a re-activation free of a
/// `WM_DISPLAYCHANGE` broadcast. Each attempt from `mode_set_attempts` is a
/// `CDS_FULLSCREEN` change (temporary, so the registry mode stays the
/// desktop's for [`restore_registry_mode`]). Failure is reported once; the
/// caller keeps the monitor-covering window as the fallback.
fn set_display_mode(request: ModeRequest) -> bool {
    let (cur_w, cur_h) =
        query_display_mode(ENUM_CURRENT_SETTINGS).map_or((0, 0), |mode| (mode.width, mode.height));
    if (cur_w, cur_h) == (request.width, request.height) {
        debug!(target: LOG_TARGET, "display mode {cur_w}x{cur_h} already current");
        return true;
    }
    let mut ret = DISP_CHANGE_SUCCESSFUL;
    for attempt in mode_set_attempts(request) {
        let mut dm = empty_devmode();
        dm.fields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL;
        dm.pels_width = attempt.width;
        dm.pels_height = attempt.height;
        dm.bits_per_pel = 32;
        if attempt.refresh_hz != 0 {
            dm.fields |= DM_DISPLAYFREQUENCY;
            dm.display_frequency = attempt.refresh_hz;
        }
        ret = change_display_settings(Some(&mut dm), CDS_FULLSCREEN);
        if ret == DISP_CHANGE_SUCCESSFUL {
            debug!(
                target: LOG_TARGET,
                "set display mode {}x{}@{}Hz (was {cur_w}x{cur_h})",
                attempt.width, attempt.height, attempt.refresh_hz,
            );
            return true;
        }
        debug!(
            target: LOG_TARGET,
            "ChangeDisplaySettingsW({}x{}@{}Hz, CDS_FULLSCREEN) returned {ret}",
            attempt.width, attempt.height, attempt.refresh_hz,
        );
    }
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "ChangeDisplaySettingsW({}x{}) failed (ret={ret}); the window covers the monitor \
         instead and present scales the frame, so mouse coordinates stay in monitor space",
        request.width, request.height,
    );
    false
}

/// Put the desktop back to the registry display mode when it differs.
///
/// The D3D9 contract restores the registry mode when a fullscreen device
/// loses focus (app deactivation) and when it leaves fullscreen (windowed
/// `Reset`, final release), whether the device set the mode or the app did
/// through user32. A windowed device never triggers this. The compare-first
/// guard keeps the nothing-changed case free of a spurious
/// `WM_DISPLAYCHANGE` broadcast; the refresh rate is ignored in the
/// comparison because the registry view may report 0 where the current view
/// reports the real rate.
pub fn restore_registry_mode() {
    let Some(cur) = query_display_mode(ENUM_CURRENT_SETTINGS) else {
        return;
    };
    let Some(reg) = query_display_mode(ENUM_REGISTRY_SETTINGS) else {
        return;
    };
    let (cur_w, cur_h, reg_w, reg_h) = (cur.width, cur.height, reg.width, reg.height);
    if (cur_w, cur_h) == (reg_w, reg_h) {
        return;
    }
    let ret = change_display_settings(None, 0);
    if ret == DISP_CHANGE_SUCCESSFUL {
        debug!(
            target: LOG_TARGET,
            "restored registry display mode {reg_w}x{reg_h} (was {cur_w}x{cur_h})"
        );
    } else {
        warn!(
            target: LOG_TARGET,
            "ChangeDisplaySettingsW(NULL) failed restoring {reg_w}x{reg_h} from {cur_w}x{cur_h}: ret={ret}"
        );
    }
}

#[link(name = "user32")]
unsafe extern "system" {
    fn ChangeDisplaySettingsW(dev_mode: *mut DevModeW, flags: u32) -> i32;
    fn EnumDisplaySettingsW(device_name: *const u16, mode_num: u32, dev_mode: *mut DevModeW)
    -> i32;
    fn GetMonitorInfoW(monitor: *mut c_void, info: *mut MonitorInfo) -> i32;
    fn GetWindowLongW(hwnd: *mut c_void, index: i32) -> i32;
    fn GetWindowRect(hwnd: *mut c_void, rect: *mut Rect) -> i32;
    fn IsWindow(hwnd: *mut c_void) -> i32;
    fn IsZoomed(hwnd: *mut c_void) -> i32;
    fn MonitorFromPoint(point: Point, flags: u32) -> *mut c_void;
    fn SetWindowPos(
        hwnd: *mut c_void,
        insert_after: *mut c_void,
        x: i32,
        y: i32,
        cx: i32,
        cy: i32,
        flags: u32,
    ) -> i32;
}

// Win32 LONG is 32-bit; `SetWindowLongPtrW` only exists on 64-bit Windows,
// while 32-bit user32 exports `SetWindowLongW` (the header is a #define
// alias). Declare a single Rust-side `SetWindowLongPtrW` symbol per arch
// and route the 32-bit one through `#[link_name = "SetWindowLongW"]` so
// the call site stays uniform. Window styles ride the same entry point:
// they are 32-bit values either way, and both Windows and Wine truncate.
#[cfg(target_pointer_width = "64")]
#[link(name = "user32")]
unsafe extern "system" {
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, new_long: isize) -> isize;
}

#[cfg(target_pointer_width = "32")]
#[link(name = "user32")]
unsafe extern "system" {
    #[link_name = "SetWindowLongW"]
    fn SetWindowLongPtrW(hwnd: *mut c_void, index: i32, new_long: isize) -> isize;
}

/// Set one of a window's `GWL_*` / `GWLP_*` longs, returning the previous value.
pub fn set_window_long_ptr(hwnd: *mut c_void, index: i32, new: isize) -> isize {
    // SAFETY: SetWindowLongPtrW (or SetWindowLongW on 32-bit) accepts any
    // HWND and isize; documented to return the previous value.
    unsafe { SetWindowLongPtrW(hwnd, index, new) }
}

// ── Constants ──

const GWL_STYLE: i32 = -16;
const GWL_EXSTYLE: i32 = -20;

const WS_POPUP: u32 = 0x8000_0000;
const WS_SYSMENU: u32 = 0x0008_0000;
const WS_CAPTION: u32 = 0x00C0_0000;
const WS_THICKFRAME: u32 = 0x0004_0000;
const WS_VISIBLE: u32 = 0x1000_0000;
const WS_EX_WINDOWEDGE: u32 = 0x0000_0100;
const WS_EX_CLIENTEDGE: u32 = 0x0000_0200;

const SWP_NOSIZE: u32 = 0x0001;
const SWP_NOMOVE: u32 = 0x0002;
const SWP_NOZORDER: u32 = 0x0004;
const SWP_NOACTIVATE: u32 = 0x0010;
const SWP_SHOWWINDOW: u32 = 0x0040;
const SWP_FRAMECHANGED: u32 = 0x0020;

const MONITOR_DEFAULTTOPRIMARY: u32 = 0x0000_0001;

// ── Safe wrappers ──
//
// One wrapper per Win32 entry point so call sites stay unsafe-free, per
// CONVENTIONS.md §13 "Don't sprinkle — concentrate".

/// The primary display's `HMONITOR`.
///
/// Resolved from the desktop origin, which is the primary monitor by
/// definition, with the primary as the fallback if (0, 0) is somehow
/// uncovered. Every rect this module works with has to come from the same
/// monitor, so they all resolve through here.
pub fn primary_monitor() -> *mut c_void {
    // SAFETY: MonitorFromPoint takes a POINT by value plus a flags DWORD and
    // returns an HMONITOR; the arguments are plain scalars.
    unsafe { MonitorFromPoint(Point { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) }
}

/// The primary monitor's rect in virtual-desktop coordinates.
///
/// This is the rect a fullscreen device's window is stretched over. While a
/// display mode is set it is that mode's rect, which win32u maps onto the
/// physical monitor; between mode-sets it is the desktop's.
pub fn primary_monitor_rect() -> Option<Rect> {
    let monitor = primary_monitor();
    let mut info = MonitorInfo {
        cb_size: MONITOR_INFO_SIZE,
        rc_monitor: Rect::EMPTY,
        rc_work: Rect::EMPTY,
        dw_flags: 0,
    };
    // SAFETY: `monitor` is a live HMONITOR and `info` is an owned local with
    // `cb_size` set, which is what GetMonitorInfoW validates.
    let ok = unsafe { GetMonitorInfoW(monitor, &raw mut info) };
    (ok != 0).then_some(info.rc_monitor)
}

fn window_long(hwnd: *mut c_void, index: i32) -> u32 {
    // SAFETY: GetWindowLongW accepts any HWND and a documented index;
    // returns 0 for an invalid window.
    unsafe { GetWindowLongW(hwnd, index) }.cast_unsigned()
}

fn set_window_long(hwnd: *mut c_void, index: i32, value: u32) {
    let long = isize::try_from(value.cast_signed()).expect("Win32 LONG fits isize");
    set_window_long_ptr(hwnd, index, long);
}

fn window_rect(hwnd: *mut c_void) -> Option<Rect> {
    let mut rect = Rect::EMPTY;
    // SAFETY: GetWindowRect accepts any HWND and writes a RECT through the
    // out pointer, which is an owned local. A bad HWND yields a zero return.
    let ok = unsafe { GetWindowRect(hwnd, &raw mut rect) };
    (ok != 0).then_some(rect)
}

fn set_window_pos(hwnd: *mut c_void, insert_after: *mut c_void, rect: Rect, flags: u32) {
    // SAFETY: SetWindowPos accepts any HWND plus a z-order pseudo-handle and
    // an arbitrary rect; failure is a window-manager concern, not a
    // memory-safety one.
    unsafe {
        SetWindowPos(
            hwnd,
            insert_after,
            rect.left,
            rect.top,
            rect.width(),
            rect.height(),
            flags,
        );
    }
}

fn is_window(hwnd: *mut c_void) -> bool {
    // SAFETY: IsWindow accepts any pointer-shaped handle, valid or not —
    // that is the entire point of the call.
    unsafe { IsWindow(hwnd) != 0 }
}

/// `true` when `hwnd` is maximized (`WS_MAXIMIZE`).
///
/// A maximized window is sized by the window manager, not the game, so it is
/// the one windowed case where the back buffer follows the client rect and
/// the requested resolution is ignored.
pub fn is_maximized(hwnd: *mut c_void) -> bool {
    // SAFETY: IsZoomed accepts any HWND and returns zero for one that is not
    // a window.
    unsafe { IsZoomed(hwnd) != 0 }
}

// ── Geometry helpers ──

impl Rect {
    /// The zero rect, used for out-params and for a window whose rect is unknown.
    pub const EMPTY: Self = Self {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };

    /// Width in pixels, saturating so an inverted rect yields zero.
    pub const fn width(&self) -> i32 {
        self.right.saturating_sub(self.left)
    }

    /// Height in pixels, saturating so an inverted rect yields zero.
    pub const fn height(&self) -> i32 {
        self.bottom.saturating_sub(self.top)
    }
}

// ── Re-entrancy latch ──

/// Nesting depth of mtld3d-driven window moves, process-wide.
///
/// Every `SetWindowPos` below bounces a synchronous `WM_SIZE` back through the
/// cursor subclass, which would otherwise rebuild the device's back buffer
/// from inside our own window management.
///
/// It is global rather than per-device because the message is delivered to
/// whichever device is subclassed on that window, and that is not necessarily
/// the device performing the move: a second `CreateDevice` on a window that
/// already has one moves the window before its own device exists, so the
/// bounce lands on the *first* device. A per-device flag cannot see that, and
/// the stale device would rebuild its resources re-entrantly.
///
/// A counter, not a flag: leaving and re-entering fullscreen for a retarget
/// nests, and a plain flag would clear on the inner exit.
static DRIVING_WINDOW: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// `true` while any mtld3d window move is in flight.
///
/// Read by the cursor subclass to skip the auto-resize for a `WM_SIZE` we
/// caused ourselves.
pub fn driving_window() -> bool {
    DRIVING_WINDOW.load(core::sync::atomic::Ordering::Relaxed) != 0
}

/// Holds [`DRIVING_WINDOW`] raised for the duration of a window move.
struct DrivingGuard;

impl DrivingGuard {
    fn new() -> Self {
        DRIVING_WINDOW.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Self
    }
}

impl Drop for DrivingGuard {
    fn drop(&mut self) {
        DRIVING_WINDOW.fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
    }
}

// ── Fullscreen state ──

/// Window state saved before the device took its window fullscreen.
///
/// Held by the device for the lifetime of the fullscreen mode so [`leave`]
/// can put the window back exactly as the game left it. The `HWND` travels
/// with it: a `Reset` may hand the device a different window, and the one to
/// restore is always the one we took over.
pub struct SavedWindow {
    hwnd: *mut c_void,
    style: u32,
    exstyle: u32,
    /// The pre-fullscreen window rect, in the desktop mode's coordinates.
    ///
    /// Captured before the mode-set and restored after the registry mode is
    /// back, so both ends of the round trip are in the same space.
    rect: Rect,
    /// Ping-pong guard for [`reassert_cover`], one per fullscreen session.
    guard: mtld3d_core::fullscreen_resize::ExternalResizeGuard,
    /// The display mode this session set, re-asserted on activation.
    ///
    /// `None` when the request was no settable mode, or when user32 refused
    /// it and the device fell back to the monitor-covering window.
    mode: Option<ModeRequest>,
    /// `false` under `D3DCREATE_NOWINDOWCHANGES`: the window is the app's.
    manage_window: bool,
}

impl SavedWindow {
    /// The window this state was captured from.
    pub const fn window(&self) -> *mut c_void {
        self.hwnd
    }
}

/// `true` when the app asked the device to keep its hands off the window.
///
/// `D3DCREATE_NOWINDOWCHANGES` makes window management the app's job, so a
/// fullscreen device neither styles, moves, shows nor hides the window it
/// presents into, and never re-covers the monitor behind the app's back. The
/// note is logged once because a fullscreen device whose window stays small
/// and decorated looks like a defect from the outside.
fn window_changes_suppressed(saved: &SavedWindow) -> bool {
    if saved.manage_window {
        return false;
    }
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "D3DCREATE_NOWINDOWCHANGES: the device window keeps its style, rect and visibility",
    );
    true
}

/// Take `hwnd` fullscreen: set `mode`, then no decoration, covering the monitor.
///
/// The window's pre-fullscreen style and rect are captured first so [`leave`]
/// can put it back, in the desktop mode's coordinates since the mode-set
/// comes after. The z-order is deliberately untouched; see the module docs.
/// With `manage_window` false (`D3DCREATE_NOWINDOWCHANGES`) the window half
/// is skipped: the state is captured, the window is left to the app, and
/// every later window transition is a no-op. The mode is set either way.
pub fn enter(hwnd: *mut c_void, manage_window: bool, mode: Option<ModeRequest>) -> SavedWindow {
    // Held across the mode-set too: win32u broadcasts `WM_DISPLAYCHANGE`
    // synchronously from inside it, and a game handler that answers by
    // resizing its window bounces a `WM_SIZE` the subclass would otherwise
    // auto-resize on; the window is placed right after anyway.
    let _driving = DrivingGuard::new();
    let mut saved = SavedWindow {
        hwnd,
        style: window_long(hwnd, GWL_STYLE),
        exstyle: window_long(hwnd, GWL_EXSTYLE),
        rect: window_rect(hwnd).unwrap_or(Rect::EMPTY),
        guard: mtld3d_core::fullscreen_resize::ExternalResizeGuard::new(),
        mode: None,
        manage_window,
    };
    saved.mode = mode.filter(|request| set_display_mode(*request));
    if !window_changes_suppressed(&saved) {
        apply_fullscreen_window(hwnd, &saved);
    }
    saved
}

/// Re-apply mode and window rect for a device that stayed fullscreen across a `Reset`.
///
/// The saved window state is the one captured on the way in and is left
/// alone: a Reset between two fullscreen present params must still restore
/// the *pre-fullscreen* window when the device finally leaves. The mode
/// follows the new request (compare-first, so a same-mode Reset sets
/// nothing). The re-assert guard refills: a game-driven Reset is a fresh
/// session.
pub fn update(saved: &mut SavedWindow, mode: Option<ModeRequest>) {
    let _driving = DrivingGuard::new();
    saved.mode = mode.filter(|request| set_display_mode(*request));
    if window_changes_suppressed(saved) {
        return;
    }
    saved.guard.reset();
    apply_fullscreen_window(saved.hwnd, saved);
}

/// Re-assert the mode and the monitor rect after the app is activated again.
///
/// The focus-regain half of the mode contract: deactivation put the registry
/// mode back (see the cursor subclass's `WM_ACTIVATEAPP` handling), so the
/// device's mode is set again and the window re-covers the monitor, whose
/// rect is the mode's once more. Nothing to do for a device that set no
/// mode, except re-covering a managed window.
pub fn reactivate(saved: &mut SavedWindow) {
    let _driving = DrivingGuard::new();
    let mode_set = saved.mode.is_some_and(set_display_mode);
    if window_changes_suppressed(saved) {
        return;
    }
    saved.guard.reset();
    apply_fullscreen_window(saved.hwnd, saved);
    debug!(
        target: LOG_TARGET,
        "fullscreen device reactivated: display mode {}, window re-covered",
        if mode_set { "re-asserted" } else { "not ours to set" },
    );
}

/// Answer an external resize of a fullscreen device's window.
///
/// The window is supposed to cover the monitor, whose rect is the mode's
/// while one is set; games that manage their own window resize it to their
/// mode's outer rect after a Reset, a few pixels off. Re-apply the monitor
/// rect, bounded by the [`ExternalResizeGuard`] so a window manager that
/// keeps clamping the window back gets a one-shot warning instead of a fight.
///
/// [`ExternalResizeGuard`]: mtld3d_core::fullscreen_resize::ExternalResizeGuard
pub fn reassert_cover(saved: &mut SavedWindow, incoming: (u32, u32)) {
    use mtld3d_core::fullscreen_resize::ExternalResizeAction;

    if window_changes_suppressed(saved) {
        return;
    }
    let Some(rect) = primary_monitor_rect() else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "GetMonitorInfo failed; fullscreen window not re-asserted after an external resize",
        );
        return;
    };
    let monitor = (rect.width().cast_unsigned(), rect.height().cast_unsigned());
    match saved.guard.decide(incoming, monitor) {
        ExternalResizeAction::Covered => {}
        ExternalResizeAction::Reassert => {
            debug!(
                target: LOG_TARGET,
                "external resize to {}x{} on a fullscreen window; re-asserting the monitor rect",
                incoming.0, incoming.1,
            );
            apply_fullscreen_window(saved.hwnd, saved);
        }
        ExternalResizeAction::Suppressed => {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "the window manager keeps re-sizing the fullscreen window ({}x{}); leaving it, \
                 the back buffer keeps its size and present scales the frame",
                incoming.0, incoming.1,
            );
        }
    }
}

/// A windowed style made fullscreen: no decoration, but still managed.
///
/// `WS_POPUP | WS_SYSMENU` is what keeps the window in the window manager's
/// hands (and therefore able to take keyboard focus) while the caption and
/// the resize frame go away.
const fn fullscreen_style(style: u32) -> u32 {
    (style | WS_POPUP | WS_SYSMENU) & !(WS_CAPTION | WS_THICKFRAME)
}

/// A windowed extended style made fullscreen: the decoration edges go away.
const fn fullscreen_exstyle(exstyle: u32) -> u32 {
    exstyle & !(WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE)
}

/// Strip the decoration and stretch `hwnd` over the primary monitor.
///
/// Runs after any mode-set: win32u recomputes a window's physical rect only
/// on its next `SetWindowPos`, so this is also what puts the `NSWindow` onto
/// the physical monitor once the mode is in place.
fn apply_fullscreen_window(hwnd: *mut c_void, saved: &SavedWindow) {
    let _driving = DrivingGuard::new();
    let style = fullscreen_style(saved.style);
    let exstyle = fullscreen_exstyle(saved.exstyle);
    set_window_long(hwnd, GWL_STYLE, style);
    set_window_long(hwnd, GWL_EXSTYLE, exstyle);

    let Some(rect) = primary_monitor_rect() else {
        warn!(target: LOG_TARGET, "GetMonitorInfo failed — device window not resized to the monitor");
        return;
    };
    // Deliberately `SWP_NOZORDER` rather than `HWND_TOPMOST`. Raising a
    // window to the topmost level makes Wine's mac driver re-derive the Cocoa
    // window's level and parent (`set_cocoa_window_properties` →
    // `macdrv_set_cocoa_parent_window`), which hops to the main thread via
    // `OnMainThread` while still holding winemac's per-window lock. A focus
    // event arriving in that window re-enters `NtUserSetWindowPos` on another
    // thread, blocks on the same lock, and the process deadlocks; observed
    // reproducibly in the d3d9 `visual` conformance subtest.
    //
    // Nothing is lost: a borderless window covering the monitor already looks
    // fullscreen, and leaving the z-order alone keeps the window manager in
    // charge of stacking, which is the better macOS citizen anyway.
    set_window_pos(
        hwnd,
        core::ptr::null_mut(),
        rect,
        SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER,
    );
    debug!(
        target: LOG_TARGET,
        "fullscreen window {}x{} at ({}, {}), style {style:#010x}/{exstyle:#010x}",
        rect.width(), rect.height(), rect.left, rect.top,
    );
}

/// Restore the display mode and the window state captured by [`enter`].
pub fn leave(saved: &SavedWindow) {
    // Leaving fullscreen (windowed `Reset` or device destruction) puts the
    // registry display mode back first, matching native D3D9's order (mode
    // restore, then window restore), so the saved rect lands in the space it
    // was captured in. No-op when the mode is already the desktop's.
    restore_registry_mode();
    if window_changes_suppressed(saved) {
        return;
    }
    let _driving = DrivingGuard::new();
    let hwnd = saved.hwnd;
    if !is_window(hwnd) {
        return;
    }

    // WS_VISIBLE changed because *we* changed it, so it is excluded from both
    // the comparison and the restored value: the window keeps whatever
    // visibility it has now. The z-order is never touched on the way in, so
    // there is nothing to put back.
    let style = window_long(hwnd, GWL_STYLE);
    let exstyle = window_long(hwnd, GWL_EXSTYLE);
    let ours = style & !WS_VISIBLE == fullscreen_style(saved.style) & !WS_VISIBLE
        && exstyle == fullscreen_exstyle(saved.exstyle);
    // Only put the style back if the game did not touch it while fullscreen.
    // Titles that swap styles themselves before Reset-ing to windowed expect
    // their own value to survive, so an app-modified style stays.
    if ours {
        set_window_long(
            hwnd,
            GWL_STYLE,
            saved.style & !WS_VISIBLE | style & WS_VISIBLE,
        );
        set_window_long(hwnd, GWL_EXSTYLE, saved.exstyle);
    }

    let show = if style & WS_VISIBLE == 0 {
        0
    } else {
        SWP_SHOWWINDOW
    };
    // A window whose rect we never managed to read stays where it is rather
    // than being teleported to a zero rect at the desktop origin.
    let geometry = if saved.rect.width() == 0 || saved.rect.height() == 0 {
        SWP_NOMOVE | SWP_NOSIZE
    } else {
        0
    };
    set_window_pos(
        hwnd,
        core::ptr::null_mut(),
        saved.rect,
        SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER | show | geometry,
    );
    debug!(
        target: LOG_TARGET,
        "restored windowed rect {}x{} at ({}, {})",
        saved.rect.width(), saved.rect.height(), saved.rect.left, saved.rect.top,
    );
}
