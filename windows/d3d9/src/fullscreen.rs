//! Device-window management for a fullscreen device.
//!
//! A fullscreen `CreateDevice` strips the window's decoration and stretches it
//! over the monitor rect. What it deliberately does **not** do is change the
//! display mode or the window's z-order.
//!
//! Switching the desktop mode is the letter of the D3D9 contract, but on macOS
//! it is the wrong trade. Wine's mac driver hands the request to
//! `CGDisplaySetDisplayMode` unless `EmulateModeset` is set, so a fullscreen
//! game reconfigures the user's screen, rearranges every other window, and
//! costs a second mode change on the way out — and making the image correct
//! would depend on a registry key the user has to know about.
//!
//! A monitor-covering borderless window buys the same property the mode-set
//! was there for. The guest's coordinate space matches the window's client
//! rect, so `GetClientRect`, mouse input and our back-buffer geometry stay in
//! agreement, and the resolution the game asked for is simply ignored: the
//! back buffer follows the client rect and `render.scale` decides how many
//! pixels are actually rasterized. That makes fullscreen and a maximized
//! window the same path.
//!
//! The z-order is left to the window manager. Raising the window to the
//! topmost level deadlocks Wine's mac driver: it re-derives the Cocoa window's
//! level and parent while holding winemac's per-window lock and hops to the
//! main thread to do it, so a focus event arriving meanwhile re-enters
//! `NtUserSetWindowPos` on another thread and both stall. See
//! `apply_fullscreen_window`.
//!
//! Only the primary display is driven, so a device whose window lives on a
//! secondary monitor is covered by the wrong rect. Matching the window's
//! current monitor is a follow-up.

use core::ffi::c_void;

use log::{debug, warn};

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

#[link(name = "user32")]
unsafe extern "system" {
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
/// This is the rect a fullscreen device's window is stretched over, and
/// therefore the size its back buffer resolves to.
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
/// A maximized window is sized by the window manager, not the game, so it gets
/// the same treatment as a fullscreen one: the back buffer follows the client
/// rect and the requested resolution is ignored.
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
    rect: Rect,
}

impl SavedWindow {
    /// The window this state was captured from.
    pub const fn window(&self) -> *mut c_void {
        self.hwnd
    }
}

/// Take `hwnd` fullscreen: no decoration, covering the monitor.
///
/// The window's pre-fullscreen style and rect are captured first so [`leave`]
/// can put it back. The display mode is deliberately untouched, and so is the
/// z-order; see the module docs.
pub fn enter(hwnd: *mut c_void) -> SavedWindow {
    let saved = SavedWindow {
        hwnd,
        style: window_long(hwnd, GWL_STYLE),
        exstyle: window_long(hwnd, GWL_EXSTYLE),
        rect: window_rect(hwnd).unwrap_or(Rect::EMPTY),
    };
    apply_fullscreen_window(hwnd, &saved);
    saved
}

/// Re-apply the window rect for a fullscreen device that stayed fullscreen.
///
/// The saved window state is the one captured on the way in and is left alone
/// — a Reset between two fullscreen present params must still restore the
/// *pre-fullscreen* window when the device finally leaves.
pub fn update(saved: &SavedWindow) {
    apply_fullscreen_window(saved.hwnd, saved);
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

/// Restore the window state captured by [`enter`].
pub fn leave(saved: &SavedWindow) {
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
