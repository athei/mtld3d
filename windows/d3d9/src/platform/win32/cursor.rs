//! The Win32 side of the D3D9 cursor: a hardware HCURSOR, or a blank one under the overlay.
//!
//! It does the Win32 work of the `SetCursor*` / `ShowCursor` entry points in
//! `device/cursor.rs`, whose `CursorState` keeps the D3D9 rules and holds a
//! [`SystemCursor`], and holds `cursor_wnd_proc`. Games that rely on the
//! Win32 cursor (hiding the OS pointer while they render their own sprite over
//! it) get the behaviour they expect.
//!
//! **Per-window back-pointers.** `CallWindowProcW` can't pass per-device user
//! data through, so the subclass maps each window's `HWND` to its owning
//! `DeviceInner` in `DEVICE_INSTANCES`. Two devices may exist at once, and a
//! single global back-pointer would be wrong the moment they do, so the lookup
//! is keyed by the window the message arrived on.
//!
//! **Software mode.** With `cursor.software` resolved on, the game's cursor is
//! drawn by the unix side in an overlay window, which `CursorState` feeds, and
//! the Win32 cursor this module realizes is a blank HCURSOR, so the
//! `WindowServer` cursor plane never toggles on our account.

use core::{ffi::c_void, fmt, ptr::null_mut};
use std::{
    sync::{LazyLock, Mutex},
    time::Instant,
};

use log::{Level, debug, error, log_enabled, trace, warn};
use mtld3d_core::{cursor::BitmapLayout, window_subclass::WindowSubclasses};
use mtld3d_shared::{bounded_cache::BoundedCache, mtl::CURSOR_SPRITE_CACHE_ENTRIES};
use mtld3d_types::{CURSOR_SHOWING, CURSORINFO, ICONINFO, POINT};

use super::fullscreen::set_window_long_ptr;
use crate::device::{
    DeviceInner,
    cursor::{LOG_TARGET, elapsed_us, scale_cursor_pixels},
};

// ── Win32 FFI ──

#[link(name = "user32")]
unsafe extern "system" {
    fn SetCursor(cursor: *mut c_void) -> *mut c_void;
    fn SetCursorPos(x: i32, y: i32) -> i32;
    fn GetCursorPos(p: *mut POINT) -> i32;
    fn CreateIconIndirect(info: *const ICONINFO) -> *mut c_void;
    fn GetWindowThreadProcessId(hwnd: *mut c_void, process_id: *mut u32) -> u32;
    fn CallWindowProcW(
        prev_proc: *mut c_void,
        hwnd: *mut c_void,
        msg: u32,
        wp: usize,
        lp: isize,
    ) -> isize;
    fn DefWindowProcW(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize;
    fn PostMessageW(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> i32;
    fn GetCursor() -> *mut c_void;
    fn GetCursorInfo(info: *mut CURSORINFO) -> i32;
    fn GetForegroundWindow() -> *mut c_void;
    fn DestroyCursor(cursor: *mut c_void) -> i32;
    fn LoadCursorW(instance: *mut c_void, name: *const u16) -> *mut c_void;
}

// `GetClassLongPtrW` only exists on 64-bit Windows; 32-bit user32 exports
// `GetClassLongW`, and the header aliases one to the other. One Rust-side
// symbol per arch keeps the call site uniform.
#[cfg(target_pointer_width = "64")]
#[link(name = "user32")]
unsafe extern "system" {
    fn GetClassLongPtrW(hwnd: *mut c_void, index: i32) -> usize;
}

#[cfg(target_pointer_width = "32")]
#[link(name = "user32")]
unsafe extern "system" {
    #[link_name = "GetClassLongW"]
    fn GetClassLongPtrW(hwnd: *mut c_void, index: i32) -> usize;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThreadId() -> u32;
    fn GetLastError() -> u32;
}

// Safe wrappers around the Win32 calls used by this module — each Win32
// function is wrapped once so call sites are unsafe-free per CONVENTIONS.md
// §13 "Don't sprinkle — concentrate".

fn set_cursor(handle: *mut c_void) {
    // SAFETY: SetCursor accepts null and any valid HCURSOR; documented to
    // be thread-safe and side-effect-free besides updating the cursor.
    unsafe {
        SetCursor(handle);
    }
}

/// The calling thread's current cursor, null for none.
fn get_cursor() -> *mut c_void {
    // SAFETY: GetCursor takes no arguments and cannot fail.
    unsafe { GetCursor() }
}

/// Free `handle`; `false` when user32 refused, which the caller reports.
fn destroy_cursor(handle: *mut c_void) -> bool {
    // SAFETY: DestroyCursor accepts any handle value; an invalid one fails
    // with FALSE and touches nothing.
    unsafe { DestroyCursor(handle) != 0 }
}

/// The cursor `hwnd`'s window class carries, null when it has none or the window is gone.
fn class_cursor(hwnd: *mut c_void) -> *mut c_void {
    // SAFETY: GetClassLongPtrW (GetClassLongW on 32-bit) accepts any HWND
    // and index; an invalid window answers 0.
    unsafe { GetClassLongPtrW(hwnd, GCLP_HCURSOR) as *mut c_void }
}

/// The system arrow, a shared cursor user32 owns.
fn load_arrow_cursor() -> *mut c_void {
    // SAFETY: LoadCursorW with a null module loads a predefined cursor;
    // `IDC_ARROW` is passed as MAKEINTRESOURCE, an integer in place of the
    // name pointer, which the call is documented to accept.
    unsafe { LoadCursorW(null_mut(), IDC_ARROW as *const u16) }
}

/// `set_cursor` plus its wall time in microseconds.
///
/// The Win32 cursor calls can stall the calling thread on Wine's Cocoa
/// main thread: `SetCursorPos` and an idle `GetCursorPos` (no pointer
/// change for 100 ms) go through a synchronous `OnMainThread`, and
/// `SetCursor` is a wineserver round trip. Every cursor log line carries
/// the microseconds its calls took, so a frame hitch around a cursor
/// transition can be attributed to, or cleared of, these calls from a
/// `RUST_LOG=mtld3d::d3d9::cursor=debug` log alone.
fn timed_set_cursor(handle: *mut c_void) -> u64 {
    let started = Instant::now();
    set_cursor(handle);
    elapsed_us(started)
}

fn set_cursor_pos(x: i32, y: i32) {
    // SAFETY: SetCursorPos accepts any i32 pair; failure (return 0) is a
    // game-side concern, not a memory-safety issue.
    unsafe {
        SetCursorPos(x, y);
    }
}

fn get_cursor_pos() -> Option<POINT> {
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: GetCursorPos writes a POINT through `&mut p`; pointer comes
    // from an owned local, so non-null + aligned + writable holds.
    let ok = unsafe { GetCursorPos(&raw mut p) };
    (ok != 0).then_some(p)
}

/// Win32 thread id of the caller.
///
/// Cursor realization only reaches the Mac driver when it runs on the thread
/// that owns the cursor window: wine's `set_cursor` server request notifies
/// the driver from `update_desktop_cursor_handle` only when the calling
/// thread's input is the cursor window's input. A game that drives
/// `ShowCursor` / `SetCursorProperties` from a worker thread therefore
/// updates nothing on screen, which is invisible without this id in the log.
pub fn current_thread_id() -> u32 {
    // SAFETY: GetCurrentThreadId takes no arguments and cannot fail.
    unsafe { GetCurrentThreadId() }
}

/// Win32 thread id that owns `hwnd`, or `0` if the window is gone.
fn window_thread_id(hwnd: *mut c_void) -> u32 {
    // SAFETY: GetWindowThreadProcessId accepts any HWND and a null
    // process-id out-pointer; returns 0 for an invalid window.
    unsafe { GetWindowThreadProcessId(hwnd, null_mut()) }
}

fn def_window_proc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    // SAFETY: DefWindowProcW is the documented Win32 fallback — accepts any
    // HWND/msg/wp/lp tuple.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

fn post_message(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) {
    // SAFETY: PostMessageW accepts any HWND and message tuple; posting to a
    // window that is destroyed before delivery just drops the message.
    unsafe {
        PostMessageW(hwnd, msg, wp, lp);
    }
}

fn call_window_proc(
    prev_proc: *mut c_void,
    hwnd: *mut c_void,
    msg: u32,
    wp: usize,
    lp: isize,
) -> isize {
    // SAFETY: CallWindowProcW forwards to a documented WNDPROC; the
    // subclass-install path stored `prev_proc` from a prior
    // GetWindowLongPtrW call.
    unsafe { CallWindowProcW(prev_proc, hwnd, msg, wp, lp) }
}

fn create_icon_indirect(info: &ICONINFO) -> *mut c_void {
    // SAFETY: ICONINFO is passed by ref so the pointer is non-null +
    // properly aligned; CreateIconIndirect returns null on failure (caller
    // checks).
    unsafe { CreateIconIndirect(&raw const *info) }
}

fn delete_object(obj: *mut c_void) -> i32 {
    // SAFETY: DeleteObject accepts null + any GDI object handle; returns 0
    // on failure (caller logs).
    unsafe { DeleteObject(obj) }
}

/// The calling thread's Win32 last-error code.
///
/// Read right after the call that failed: every Win32 call, `DeleteObject`
/// included, may overwrite it.
fn last_error() -> u32 {
    // SAFETY: GetLastError reads the calling thread's own error slot and
    // takes no arguments.
    unsafe { GetLastError() }
}

fn create_bitmap_packed(
    width: i32,
    height: i32,
    planes: u32,
    bpp: u32,
    bits: *const c_void,
) -> *mut c_void {
    // SAFETY: CreateBitmap copies `bits` as WORD-aligned DDB scanlines. The
    // caller supplies every row in that layout: 32 bpp colour rows need no
    // padding, and the 1 bpp mask is padded to 16 bits.
    unsafe { CreateBitmap(width, height, planes, bpp, bits) }
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn CreateBitmap(
        width: i32,
        height: i32,
        planes: u32,
        bpp: u32,
        bits: *const c_void,
    ) -> *mut c_void;
    fn DeleteObject(object: *mut c_void) -> i32;
}

// ── Constants ──

const GWLP_WNDPROC: i32 = -4;
const GCLP_HCURSOR: i32 = -12;
/// `IDC_ARROW`, the standard arrow's `MAKEINTRESOURCE` id.
const IDC_ARROW: usize = 32512;
const WM_SETCURSOR: u32 = 0x0020;
const WM_ACTIVATE: u32 = 0x0006;
const WM_ACTIVATEAPP: u32 = 0x001C;
const WM_SIZE: u32 = 0x0005;
const WM_DISPLAYCHANGE: u32 = 0x007E;
const WM_WINDOWPOSCHANGED: u32 = 0x0047;

/// Byte offset of `flags` in the Win32 `WINDOWPOS` a `WM_WINDOWPOSCHANGED` points at.
///
/// Two window handles, then `x`, `y`, `cx` and `cy` as 32-bit ints, on both
/// PE arches; only `flags` is read, so the struct is not declared.
const WINDOWPOS_FLAGS_OFFSET: usize = 2 * size_of::<*mut c_void>() + 4 * size_of::<i32>();
const WA_INACTIVE: u32 = 0;
const HTCLIENT: usize = 1;

/// Private message: re-cover the monitor after an external fullscreen resize.
///
/// Posted (never sent) from the `WM_SIZE` handler, so the restore runs when
/// the game next pumps messages. That is native D3D9's cadence: the rect an
/// app sets on its own fullscreen device window survives the `MoveWindow`
/// call itself and is put back when window events are processed.
/// `lParam` carries the client size like `WM_SIZE`.
/// `WM_APP` range, consumed by the subclass and never forwarded.
const WM_APP_REASSERT_FULLSCREEN: u32 = 0x8000 + 0x03D9;

/// Private message: re-set the display mode and re-cover after an activation.
///
/// Posted (never sent) from the `WM_ACTIVATEAPP TRUE` handler. That message
/// can arrive synchronously inside the device's own `SetWindowPos` while it
/// enters fullscreen, and inside a game's `Reset`; deferring the re-assert to
/// the next pump keeps it out of both. `WM_APP` range, consumed by the
/// subclass and never forwarded.
const WM_APP_REACTIVATE_FULLSCREEN: u32 = 0x8000 + 0x03DA;

// ── Per-window subclass back-pointers ──

/// The subclassed windows (`HWND` as `usize`), each with the `DeviceInner`s registered on it.
///
/// Devices are held as raw pointers widened to `usize`. A device registers in
/// `SystemCursor::attach` during `CreateDevice` and a retarget, and leaves in
/// `SystemCursor::detach` during release and a retarget.
/// `cursor_wnd_proc` looks up its own `hwnd` here to find the device a message
/// goes to and the procedure to forward to: `CallWindowProcW` can't pass
/// per-device user data, and a window procedure has no key but the `HWND`.
/// Two devices may share one window, so only the first one replaces the
/// procedure and the last one to leave puts it back
/// ([`mtld3d_core::window_subclass`]). Each window's entry also counts the
/// calls of `cursor_wnd_proc` running on it, which only the window procedure
/// knows of and which belong to the window rather than to a device, and names
/// the device to tell when the outermost one returns (`HookCall`).
static DEVICE_INSTANCES: LazyLock<Mutex<WindowSubclasses>> =
    LazyLock::new(|| Mutex::new(WindowSubclasses::default()));

// ── SystemCursor ──

/// Frame-hitch attribution around cursor transitions.
///
/// `SystemCursor::note_present` feeds it once per `Present`. It keeps a
/// running typical Present-to-Present interval and, when one interval exceeds
/// 1.5x of it by at least `HITCH_MIN_EXCESS_US`, logs a single debug line
/// that ties the hitched frame to the
/// last show/hide transition, to the wall time the Win32 cursor calls
/// consumed since the previous `Present`, and to the `WM_SETCURSOR`
/// re-asserts in that window. A clean game thread on a hitched frame (zero
/// cursor microseconds, no other calls) points at the present side instead.
struct HitchProbe {
    last_present: Option<Instant>,
    /// Exponential running average of the Present interval, microseconds.
    ///
    /// Zero until warm, so the first frames never trip the threshold.
    interval_ewma_us: u64,
    /// Instant of the last `ShowCursor` transition and whether it showed.
    last_transition: Option<(Instant, bool)>,
    /// Wall time of every Win32 cursor call since the last `Present`, µs.
    calls_us_since_present: u64,
    /// `WM_SETCURSOR` re-asserts consumed since the last `Present`.
    setcursor_msgs_since_present: u32,
}

impl HitchProbe {
    const fn new() -> Self {
        Self {
            last_present: None,
            interval_ewma_us: 0,
            last_transition: None,
            calls_us_since_present: 0,
            setcursor_msgs_since_present: 0,
        }
    }
}

/// Minimum excess over the typical interval for a hitch, µs.
///
/// Applied on top of a 1.5x ratio, so jitter on a fast panel stays quiet
/// while one dropped refresh at 120 Hz (8.3 ms to 16.6 ms) registers.
const HITCH_MIN_EXCESS_US: u64 = 3_000;
/// Intervals above this are pauses (alt-tab, loading), not frame hitches.
const HITCH_MAX_INTERVAL_US: u64 = 500_000;

/// A cursor image the platform built, opaque to shared code.
///
/// Formats with `{:p}` as the `HCURSOR` it wraps; null for no cursor.
pub struct CursorImage(*mut c_void);

impl fmt::Pointer for CursorImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.0, f)
    }
}

/// The Win32 half of a device's cursor, one field of `CursorState`.
///
/// Fields are private to this module, so the Win32 cursor invariants stay
/// here.
pub struct SystemCursor {
    hwnd: *mut c_void,
    /// The HCURSOR realized while the D3D cursor is shown.
    ///
    /// The game's bitmap built by `build_hcursor` in hardware mode; the blank
    /// cursor in software mode, where the overlay window draws the bitmap.
    /// Null until the first accepted `SetCursorProperties`, which is also
    /// what the wndproc takes as "no cursor surface set".
    handle: *mut c_void,
    /// Cursor handle or hash changed since the last paint.
    ///
    /// The next `WM_SETCURSOR` / paint cycle needs to re-realise.
    dirty: bool,
    /// The HCURSORs built from the game's bitmaps (hardware mode), by content hash.
    ///
    /// Bounded: the least recently realized handle is destroyed when a new
    /// bitmap arrives at capacity, unless it is still in use, in which case
    /// it waits in `retired`.
    cache: BoundedCache<u64, *mut c_void>,
    /// Evicted handles that were the realized or thread cursor at eviction time.
    ///
    /// Destroyed by the next eviction that finds them out of use, or at release.
    retired: Vec<*mut c_void>,
    probe: HitchProbe,
}

impl SystemCursor {
    pub const fn new(hwnd: *mut c_void) -> Self {
        Self {
            hwnd,
            handle: null_mut(),
            dirty: true,
            cache: BoundedCache::new(CURSOR_SPRITE_CACHE_ENTRIES),
            retired: Vec::new(),
            probe: HitchProbe::new(),
        }
    }

    /// The device window the cursor follows.
    pub const fn window(&self) -> *mut c_void {
        self.hwnd
    }

    /// Follow another device window; the caller detaches from the old one first.
    pub const fn set_window(&mut self, hwnd: *mut c_void) {
        self.hwnd = hwnd;
    }

    /// Whether the foreground window's own cursor is hidden; `None` when it cannot be read.
    ///
    /// `false` while the device window is not the foreground window.
    pub fn native_hidden(&self) -> Option<bool> {
        // SAFETY: GetForegroundWindow takes no arguments and returns an opaque HWND.
        let foreground = unsafe { GetForegroundWindow() };
        let hidden = if !self.hwnd.is_null() && foreground == self.hwnd {
            let mut info = CURSORINFO {
                cb_size: u32::try_from(core::mem::size_of::<CURSORINFO>())
                    .expect("CURSORINFO size fits u32"),
                flags: 0,
                cursor: null_mut(),
                screen_pos: POINT { x: 0, y: 0 },
            };
            // SAFETY: initialized ABI-sized output with cb_size set as user32 requires.
            if unsafe { GetCursorInfo(&raw mut info) } == 0 {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET, "GetCursorInfo failed; native visibility unchanged");
                return None;
            }
            info.flags & CURSOR_SHOWING == 0 || info.cursor.is_null()
        } else {
            false
        };
        Some(hidden)
    }

    /// The blank cursor software mode realizes, built on first use and made current.
    ///
    /// `None` when GDI refused to build it, which the caller reports the way a
    /// failed `build_hcursor` is reported.
    pub fn blank(&mut self) -> Option<CursorImage> {
        if self.handle.is_null() {
            self.handle = build_blank_hcursor()?;
        }
        Some(CursorImage(self.handle))
    }

    /// Pick the image for the game's bitmap at `scale` (hardware mode), and its log outcome.
    ///
    /// `unchanged` keeps the current image; otherwise the image cached under
    /// `hash` is reused or a new one built. `None` when the build fails.
    ///
    /// # Safety
    ///
    /// `src` points at `layout.height` rows `pitch` bytes apart, each holding
    /// `layout.width` 4-byte pixels, readable for the duration of the call.
    pub unsafe fn select(
        &mut self,
        hash: u64,
        unchanged: bool,
        layout: &BitmapLayout,
        src: *const u8,
        pitch: usize,
        scale: u32,
    ) -> Option<(CursorImage, &'static str)> {
        let &BitmapLayout {
            width,
            height,
            x_hotspot,
            y_hotspot,
        } = layout;
        if unchanged {
            Some((CursorImage(self.handle), "unchanged"))
        } else if let Some(h) = self.cache.get(&hash).copied() {
            Some((CursorImage(h), "cache-hit"))
        } else {
            let Some(h) = build_hcursor(width, height, pitch, src, x_hotspot, y_hotspot, scale)
            else {
                warn!(
                    target: LOG_TARGET,
                    "reject SetCursorProperties: build_hcursor failed (hash={hash:#018x} {width}x{height} scale={scale})",
                );
                return None;
            };
            self.remember_handle(hash, h);
            Some((CursorImage(h), "built-fresh"))
        }
    }

    /// Rebuild the current pointer from its tight `pixels` at `scale`, realized while `visible`.
    ///
    /// Reuses the image cached under `hash`. `false` when the build fails,
    /// which leaves the current image in place.
    pub fn rebuild(
        &mut self,
        hash: u64,
        layout: &BitmapLayout,
        pixels: &[u8],
        scale: u32,
        visible: bool,
    ) -> bool {
        let pitch = layout.width as usize * 4;
        let mut handle = self.cache.get(&hash).copied();
        if handle.is_none() {
            handle = build_hcursor(
                layout.width,
                layout.height,
                pitch,
                pixels.as_ptr(),
                layout.x_hotspot,
                layout.y_hotspot,
                scale,
            );
            if let Some(h) = handle {
                self.remember_handle(hash, h);
            }
        }
        let Some(handle) = handle else {
            return false;
        };
        self.handle = handle;
        if visible {
            let us = timed_set_cursor(handle);
            self.charge_call_us(us);
        }
        true
    }

    /// Make `image` the current one, without realizing it.
    pub const fn set_image(&mut self, image: &CursorImage) {
        self.handle = image.0;
    }

    /// The current image when `shown`, no cursor otherwise.
    pub const fn image(&self, shown: bool) -> CursorImage {
        CursorImage(if shown { self.handle } else { null_mut() })
    }

    /// Realize `image` on screen, returning the microseconds the call took.
    pub fn apply(&mut self, image: &CursorImage) -> u64 {
        let us = timed_set_cursor(image.0);
        self.charge_call_us(us);
        us
    }

    /// Realize no cursor, so the next realize reaches the display as a change.
    pub fn kick() {
        set_cursor(null_mut());
    }

    /// Re-assert the cursor after a resize, when `reassert` says the device owns it.
    ///
    /// Marks the cursor dirty, so the next `WM_SETCURSOR` runs the
    /// null-then-set kick, and re-asserts the HCURSOR, since a drag resize
    /// leaves macOS's resize cursor on screen until something replaces it. A
    /// hidden hardware cursor is not re-asserted: while it is hidden the game
    /// owns the Win32 cursor. The software blank is, as `ShowCursor` keeps it
    /// selected in both states.
    pub fn note_resize(&mut self, reassert: bool) {
        self.dirty = true;
        if !self.handle.is_null() && reassert {
            set_cursor(self.handle);
        }
    }

    /// The number of images the cache holds.
    pub const fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Keep a built HCURSOR under its hash, destroying what the bound pushes out.
    ///
    /// An evicted handle that is the realized cursor or the thread's current
    /// cursor is not freed under it (user32 frees a cursor even while it is
    /// current, leaving the thread pointing at a freed handle); it waits in
    /// `retired` for a later eviction or the release path.
    fn remember_handle(&mut self, hash: u64, handle: *mut c_void) {
        if let Some((_, evicted)) = self.cache.insert(hash, handle) {
            self.retired.push(evicted);
        }
        let in_use = [self.handle, get_cursor()];
        self.retired.retain(|&retired| {
            if in_use.contains(&retired) {
                return true;
            }
            if !destroy_cursor(retired) {
                error!(
                    target: LOG_TARGET,
                    "remember_handle: DestroyCursor({retired:p}) failed for an evicted cursor",
                );
            }
            false
        });
    }

    /// Record a `ShowCursor` transition to `shown` for the hitch probe.
    pub fn note_transition(&mut self, shown: bool) {
        self.probe.last_transition = Some((Instant::now(), shown));
    }

    /// Fold one `Present` into the hitch probe; see `HitchProbe`.
    ///
    /// Called once per device or swap-chain `Present`. Everything it
    /// produces is a debug or trace line, so with the cursor target below
    /// debug the whole body is one level check; with it on, two `Instant`
    /// reads per frame and the log line only on a hitched frame.
    pub fn note_present(&mut self) {
        let probe = &mut self.probe;
        let calls_us = core::mem::take(&mut probe.calls_us_since_present);
        let msgs = core::mem::take(&mut probe.setcursor_msgs_since_present);
        if !log_enabled!(target: LOG_TARGET, Level::Debug) {
            return;
        }
        let now = Instant::now();
        let Some(last) = probe.last_present.replace(now) else {
            return;
        };
        let interval_us = elapsed_us(last);
        // Per-frame timeline at trace: with `mtld3d::d3d9::cursor=trace`
        // every Present gets a row, so a blip too small for the hitch rule
        // below (one late refresh under VRR pacing) is still on record next
        // to the transition line it follows.
        trace!(
            target: LOG_TARGET,
            "present: interval_us={interval_us} cursor_calls_us={calls_us} wm_setcursor={msgs}",
        );
        if interval_us > HITCH_MAX_INTERVAL_US {
            return;
        }
        let typical_us = probe.interval_ewma_us;
        // EWMA with a 1/16 weight; seeded by the first interval.
        probe.interval_ewma_us = if typical_us == 0 {
            interval_us
        } else {
            typical_us - typical_us / 16 + interval_us / 16
        };
        let hitch = typical_us != 0
            && interval_us * 2 > typical_us * 3
            && interval_us > typical_us + HITCH_MIN_EXCESS_US;
        if !hitch {
            return;
        }
        // `since_transition_ms` is -1 when no transition happened yet.
        let (transition, since_transition_ms) =
            probe
                .last_transition
                .map_or(("none", -1i64), |(at, shown)| {
                    let since = i64::try_from(elapsed_us(at) / 1000).unwrap_or(i64::MAX);
                    (if shown { "show" } else { "hide" }, since)
                });
        debug!(
            target: LOG_TARGET,
            "frame hitch: present interval {interval_us} us (typical {typical_us} us) last_transition={transition} since_transition_ms={since_transition_ms} cursor_calls_since_present_us={calls_us} wm_setcursor_since_present={msgs} tid={}",
            current_thread_id(),
        );
    }

    /// Charge `us` of Win32 cursor-call wall time to the current frame.
    const fn charge_call_us(&mut self, us: u64) {
        self.probe.calls_us_since_present = self.probe.calls_us_since_present.saturating_add(us);
    }

    /// Subclass the game's hwnd so `WM_SETCURSOR` / `WM_ACTIVATE` route through `cursor_wnd_proc`.
    ///
    /// `scale` only reaches the log line.
    ///
    /// Registers `dev_ptr` in `DEVICE_INSTANCES` under this window's `HWND`, so
    /// the subclass can resolve the device from the window a message arrived
    /// on. Only the first device on a window replaces its procedure; a device
    /// joining a window another live device already subclasses shares that
    /// hook, and messages keep going to the first device until it leaves.
    /// No-op if `hwnd` was never captured or `dev_ptr` is null.
    pub fn attach(&self, dev_ptr: *mut DeviceInner, scale: u32) {
        if self.hwnd.is_null() || dev_ptr.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "install_subclass: skipped (hwnd or dev_ptr is null) — \
                 game-side cursor messages will NOT route through mtld3d; \
                 SetCursorProperties/ShowCursor become no-ops in-game"
            );
            return;
        }
        let mut prev_wndproc = 0;
        // The lock is held across the procedure swap so that a device leaving
        // the window on another thread cannot restore the game's procedure in
        // between, and the hook never takes itself for the procedure it wraps.
        // Replacing `GWLP_WNDPROC` sends the window no message.
        let installed = DEVICE_INSTANCES
            .lock()
            .expect("device-instances mutex poisoned")
            .register(self.hwnd as usize, dev_ptr as usize, || {
                prev_wndproc = set_window_long_ptr(
                    self.hwnd,
                    GWLP_WNDPROC,
                    cursor_wnd_proc as *const () as isize,
                )
                .cast_unsigned();
                prev_wndproc
            });
        debug!(
            target: LOG_TARGET,
            "install_subclass: hwnd={:p} dev={:p} installed={installed} prev_wndproc={prev_wndproc:#x} scale={} window_tid={} caller_tid={}",
            self.hwnd,
            dev_ptr,
            scale,
            window_thread_id(self.hwnd),
            current_thread_id(),
        );
    }

    /// Unregister `dev_ptr` from this window, restoring the game's wndproc when no device is left.
    ///
    /// Call from the device-release path before freeing `DeviceInner`, and
    /// before a retarget moves the device to another window.
    pub fn detach(&self, dev_ptr: *mut DeviceInner) {
        let restored = DEVICE_INSTANCES
            .lock()
            .expect("device-instances mutex poisoned")
            .unregister(self.hwnd as usize, dev_ptr as usize, |original| {
                if original != 0 {
                    set_window_long_ptr(self.hwnd, GWLP_WNDPROC, original.cast_signed());
                }
            });
        debug!(
            target: LOG_TARGET,
            "uninstall_subclass: hwnd={:p} dev={dev_ptr:p} restored={restored} cache_entries={}",
            self.hwnd,
            self.cache.len(),
        );
    }

    /// Destroy every HCURSOR this device built.
    ///
    /// Call from the device-release path after `detach`, so no
    /// window message realizes one of them again. The handles are the
    /// device's alone: `CreateIconIndirect` copies the bitmaps, and user32
    /// frees a cursor even while it is the thread's current one, which
    /// would leave the thread pointing at a freed handle. So when the thread
    /// cursor is one of ours, replace it before freeing it. `keep_hidden`
    /// names a hidden software cursor, which leaves its blank handle selected,
    /// so null replaces it to preserve the hide. Otherwise restore the
    /// window's class cursor, or the system arrow when the class has none or
    /// the window is already gone.
    /// Once the replacement has happened every refused `DestroyCursor` is a
    /// real failure, and is reported as one.
    pub fn destroy_images(&mut self, keep_hidden: bool) {
        let mut handles: Vec<*mut c_void> = self
            .cache
            .drain()
            .map(|(_, h)| h)
            .chain(self.retired.drain(..))
            .collect();
        let current = core::mem::replace(&mut self.handle, null_mut());
        if !current.is_null() && !handles.contains(&current) {
            handles.push(current);
        }
        if handles.is_empty() {
            return;
        }
        let thread_cursor = get_cursor();
        let restored = if handles.contains(&thread_cursor) {
            let preserve_hide = keep_hidden;
            let mut replacement = if self.hwnd.is_null() {
                null_mut()
            } else {
                class_cursor(self.hwnd)
            };
            if preserve_hide {
                replacement = null_mut();
            } else if replacement.is_null() {
                replacement = load_arrow_cursor();
            }
            set_cursor(replacement);
            replacement
        } else {
            null_mut()
        };
        let mut failed = 0usize;
        for &handle in &handles {
            if !destroy_cursor(handle) {
                failed += 1;
                error!(
                    target: LOG_TARGET,
                    "destroy_handles: DestroyCursor({handle:p}) failed (thread_cursor={thread_cursor:p})",
                );
            }
        }
        debug!(
            target: LOG_TARGET,
            "destroy_handles: hwnd={:p} destroyed={} failed={failed} thread_cursor={thread_cursor:p} restored={restored:p}",
            self.hwnd,
            handles.len() - failed,
        );
    }

    /// `SetCursorPosition`: move the pointer to `(x, y)` unless it is already there.
    pub fn set_position(&mut self, x: i32, y: i32) {
        // The pre-check earns its keep: a `SetCursorPos` to the current position
        // still queues a `WM_MOUSEMOVE`, so skipping it keeps the game's message
        // queue quiet. Its cost is the idle `GetCursorPos` main-thread hop
        // described on `timed_set_cursor`, hence the timing.
        let started = Instant::now();
        let current = get_cursor_pos();
        let get_us = elapsed_us(started);
        let suppress = current.is_some_and(|p| p.x == x && p.y == y);
        if suppress {
            self.charge_call_us(get_us);
            debug!(target: LOG_TARGET, "SetCursorPosition: noop ({x},{y}) get_us={get_us}");
            return;
        }
        let started = Instant::now();
        set_cursor_pos(x, y);
        let set_us = elapsed_us(started);
        self.charge_call_us(get_us + set_us);
        if let Some(p) = current {
            debug!(
                target: LOG_TARGET,
                "SetCursorPosition: ({},{}) → ({x},{y}) dx={} dy={} get_us={get_us} set_us={set_us}",
                p.x, p.y, x - p.x, y - p.y,
            );
        } else {
            debug!(
                target: LOG_TARGET,
                "SetCursorPosition: → ({x},{y}) got_current=none get_us={get_us} set_us={set_us}",
            );
        }
    }
}

// ── Window-proc subclass ──

/// The subclass window procedure: runs on the window thread, outside the device `ApiLock`.
///
/// A thunk holding the lock inside `Reset` or a fullscreen transition sends
/// this thread synchronous messages (`SetWindowPos`, `ShowWindow`); a window
/// thread blocked here on the lock would wait on that thunk while the thunk
/// waits on it. Native D3D9's `D3DCREATE_MULTITHREADED` critical section has
/// the same hole, and applications keep the window thread out of D3D calls
/// during `Reset`, so this is parity rather than a gap. What runs unlocked:
/// the cursor latches on `WM_SETCURSOR` and `WM_ACTIVATE*`, the client
/// surface request every device on the window gets on a `WM_WINDOWPOSCHANGED`
/// that can move its client area, the fullscreen
/// window lookup, the auto-resize on `WM_SIZE`, which flushes the current
/// frame, the registry mode restore on deactivation, and the fullscreen
/// re-cover and reactivation posted as `WM_APP_REASSERT_FULLSCREEN` and
/// `WM_APP_REACTIVATE_FULLSCREEN`, which set the display mode and move the
/// window. The device pointer is read under the `DEVICE_INSTANCES` mutex and
/// used after it is released; the device's final `Release` removes its
/// registration (the last device on the window also restores the original
/// procedure) before it frees anything, so only a message already past the
/// lookup can meet a device being released. A deferred fullscreen repair that
/// one device posted and another device on the window receives finds that
/// device's own state, and its guards make it a no-op there. The resize hide
/// suppression is the one write made with the mutex held: the call counted
/// on the window ends in `HookCall`'s drop, which disarms the device named
/// for it while the mutex keeps that device registered. A
/// user resize or activation change on this thread while another thread
/// draws, resets or releases the device under the flag is the residual, and
/// native D3D9's window hook leaves the same one.
extern "system" fn cursor_wnd_proc(hwnd: *mut c_void, msg: u32, wp: usize, lp: isize) -> isize {
    // Resolve the device for *this* window, and the procedure the hook
    // replaced. A window may still be subclassed briefly after its last
    // device left (or never have been registered); fall back to the default
    // proc rather than deref a missing/stale device. The call is counted on
    // the window until `_call` drops, on every return below.
    let route = DEVICE_INSTANCES
        .lock()
        .expect("device-instances mutex poisoned")
        .enter(hwnd as usize);
    let Some((device, original_wndproc)) = route else {
        return def_window_proc(hwnd, msg, wp, lp);
    };
    let _call = HookCall {
        window: hwnd as usize,
    };
    let dev_ptr = device as *mut DeviceInner;
    let original_wndproc = original_wndproc as *mut c_void;

    // Seen before the routing below, which sends a move the device makes
    // itself to the default procedure: an extended leave tells the
    // application's own moves of the window from the driver's.
    if msg == WM_WINDOWPOSCHANGED && lp != 0 {
        // SAFETY: a `WM_WINDOWPOSCHANGED` carries a pointer to the
        // `WINDOWPOS` the move describes, valid for the call, and `flags` is
        // the `UINT` at `WINDOWPOS_FLAGS_OFFSET` in it.
        let flags_at = unsafe { (lp as *const u8).add(WINDOWPOS_FLAGS_OFFSET) };
        // SAFETY: `flags_at` points at the `WINDOWPOS`'s `UINT` `flags`, read
        // unaligned since the pointer came through an integer.
        let flags = unsafe { flags_at.cast::<u32>().read_unaligned() };
        super::fullscreen::note_window_pos_changed(hwnd, flags);
        // Win32's client rect is final by now, which a notification Cocoa
        // posts for its own resize is not: Wine applies that resize to the
        // window afterwards, on this thread. The next `Present` has Wine size
        // the client surface to it, which some Wines leave to the present.
        // Every device on the window is asked, not only the one the message
        // routes to: whichever presents next has Wine present the newest
        // device's surface.
        if mtld3d_core::window_pos::moves_client_area(flags) {
            let instances = DEVICE_INSTANCES
                .lock()
                .expect("device-instances mutex poisoned");
            for &device in instances.devices_on(hwnd as usize) {
                // SAFETY: `device` is registered on this window in
                // `DEVICE_INSTANCES`, whose lock is held here, and a device's
                // release unregisters it under that lock before it frees
                // anything, so the device is alive for the atomic OR below.
                let device = unsafe { &*(device as *const DeviceInner) };
                device.cursor().request_client_surface();
            }
        }
    }

    // While the device itself is moving the window through a fullscreen
    // transition (mode-set, cover, restore), every message but the mode
    // change goes to the default proc instead of the game, as native D3D9
    // does: a device filters the messages its own window management sends.
    // Games answer `WM_SIZE` by calling `Reset`, which would recurse into
    // the transition that sent it.
    if super::fullscreen::driving_window(hwnd) && msg != WM_DISPLAYCHANGE {
        return def_window_proc(hwnd, msg, wp, lp);
    }

    if msg == WM_SETCURSOR {
        // SAFETY: `dev_ptr` is the device this window's `DEVICE_INSTANCES`
        // entry routes to, registered by `install_subclass` and removed only
        // by `uninstall_subclass`, which runs before the device is freed.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        let hit_test = lp.cast_unsigned() & 0xFFFF;
        if hit_test == HTCLIENT && !cur.system().handle.is_null() && cur.visible() {
            // We own the win32 cursor ONLY while the D3D cursor is
            // visible. Consuming then takes over DefWindowProc's duty: the
            // displayed cursor is whatever the last SetCursor call pushed, and
            // entering the window does NOT re-apply it — so every consumed
            // pass must push, or a pointer re-entering the client area keeps
            // the stale (often none) cursor. While the D3D cursor is hidden
            // the message is FORWARDED below (native d3d9 never intercepts
            // WM_SETCURSOR): the game shows its own cursor — WoW's login
            // screen never calls ShowCursor(TRUE) and relies on exactly that.
            // The unix side asks for the same kick when the pointer comes
            // back from another process, which left its own cursor behind.
            // Taken unconditionally: a kick left behind an already dirty
            // pass would fire again on the next, correct pass.
            let kicked = cur.take_cursor_kick();
            let sys = cur.system_mut();
            let was_dirty = sys.dirty || kicked;
            let started = Instant::now();
            if was_dirty {
                sys.dirty = false;
                // Null-then-set forces macdrv to drop a lingering native
                // cursor (e.g. macOS's resize cursor after a drag).
                set_cursor(null_mut());
            }
            set_cursor(sys.handle);
            let set_cursor_us = elapsed_us(started);
            sys.charge_call_us(set_cursor_us);
            sys.probe.setcursor_msgs_since_present =
                sys.probe.setcursor_msgs_since_present.saturating_add(1);
            if was_dirty {
                debug!(
                    target: LOG_TARGET,
                    "wndproc WM_SETCURSOR: hit_test={hit_test:#x} (HTCLIENT) dirty_was=true → re-asserted handle={:p} set_cursor_us={set_cursor_us} tid={} → consumed",
                    sys.handle,
                    current_thread_id(),
                );
            } else if log_enabled!(target: LOG_TARGET, Level::Trace) {
                trace!(
                    target: LOG_TARGET,
                    "wndproc WM_SETCURSOR: hit_test={hit_test:#x} (HTCLIENT) dirty_was=false handle={:p} set_cursor_us={set_cursor_us} tid={} → consumed",
                    sys.handle,
                    current_thread_id(),
                );
            }
            return 1;
        }
        if log_enabled!(target: LOG_TARGET, Level::Trace) {
            trace!(
                target: LOG_TARGET,
                "wndproc WM_SETCURSOR: hit_test={hit_test:#x} visible={} handle={:p} tid={} → forwarded",
                cur.visible(), cur.system().handle, current_thread_id(),
            );
        }
    } else if msg == WM_ACTIVATE {
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        let activate_state = u32::try_from(wp & 0xFFFF).expect("16-bit value fits u32");
        let activating = activate_state != WA_INACTIVE;
        if activating {
            cur.system_mut().dirty = true;
        }
        debug!(
            target: LOG_TARGET,
            "wndproc WM_ACTIVATE: state={activate_state} activating={activating} dirty_now={} tid={}",
            cur.system().dirty, current_thread_id(),
        );
    } else if msg == WM_ACTIVATEAPP {
        // The D3D9 fullscreen mode contract, both halves. Deactivation puts
        // the registry display mode back, as native D3D9 does; the window
        // stays where it is (no minimise, no device loss). Activation re-sets
        // the device's mode and re-covers the monitor, deferred to the next
        // pump (see `WM_APP_REACTIVATE_FULLSCREEN`). `WM_ACTIVATEAPP` reaches
        // every top-level window of the thread, so the subclassed device
        // window sees it even when the focus window was the active one.
        // The deactivation takes the window, not the device: its restore
        // broadcasts `WM_DISPLAYCHANGE` synchronously, which can re-enter
        // this procedure and borrow the device mutably.
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let fullscreen = unsafe { (*dev_ptr).fullscreen_window() };
        if let Some(window) = fullscreen
            && wp == 0
        {
            super::fullscreen::deactivate(window);
        } else if fullscreen.is_some() {
            debug!(
                target: LOG_TARGET,
                "wndproc WM_ACTIVATEAPP TRUE on a fullscreen device; re-assert posted tid={}",
                current_thread_id(),
            );
            post_message(hwnd, WM_APP_REACTIVATE_FULLSCREEN, 0, 0);
        }
    } else if msg == WM_SIZE {
        // Implicit client-area resize from Wine's macdrv (e.g. macOS
        // shrunk the visible rect after we attached the layer because
        // chrome / dock take some pixels). lParam's low / high words
        // are the new client width / height in pixels — trigger an
        // auto-resize so a windowed back buffer keeps matching the
        // client area the game sees.
        //
        // Skipped while mtld3d is the one moving this window: a fullscreen
        // transition's own `SetWindowPos` bounces back here, and that
        // path already resolved the back-buffer size. The latch is keyed
        // by the window because the bounce is delivered to whichever
        // device is subclassed on it, which need not be the device doing
        // the move, while a move of another device's window says nothing
        // about this one. A fullscreen device never follows the
        // window: its logical size is the requested mode, and an external
        // shrink is answered by re-covering the monitor — deferred through
        // a posted message, because native leaves the app-set rect in
        // place until window events are processed (test_window_position).
        let lp_bits = lp.cast_unsigned();
        let new_width = u32::try_from(lp_bits & 0xFFFF).expect("16-bit value fits u32");
        let new_height = u32::try_from((lp_bits >> 16) & 0xFFFF).expect("16-bit value fits u32");
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for
        // the lifetime of the subclass.
        let dev = unsafe { &mut *dev_ptr };
        if new_width != 0 && new_height != 0 && !super::fullscreen::driving_window(hwnd) {
            if dev.fullscreen_window().is_some() {
                post_message(hwnd, WM_APP_REASSERT_FULLSCREEN, 0, lp);
            } else {
                dev.apply_auto_resize(new_width, new_height);
            }
        }
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let cur = unsafe { (*dev_ptr).cursor_mut() };
        let suppress = cur.enter_resize(wp);
        if suppress {
            DEVICE_INSTANCES
                .lock()
                .expect("device-instances mutex poisoned")
                .hold_until_dispatch_ends(hwnd as usize, device);
        }
        debug!(
            target: LOG_TARGET,
            "wndproc WM_SIZE: type={wp} {new_width}x{new_height} → game hides suppressed until \
             this dispatch returns={suppress} visible={} handle={:p} tid={}",
            cur.visible(), cur.system().handle, current_thread_id(),
        );
    } else if msg == WM_APP_REASSERT_FULLSCREEN {
        // Deferred half of the WM_SIZE branch above. Our own message, so it
        // is consumed here rather than forwarded to the game. lParam is the
        // client size the WM_SIZE carried; the guard re-checks it against
        // the monitor, so a stale post after the window recovered (or after
        // the device left fullscreen) is a no-op.
        let lp_bits = lp.cast_unsigned();
        let new_width = u32::try_from(lp_bits & 0xFFFF).expect("16-bit value fits u32");
        let new_height = u32::try_from((lp_bits >> 16) & 0xFFFF).expect("16-bit value fits u32");
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let dev = unsafe { &mut *dev_ptr };
        dev.reassert_fullscreen_cover(new_width, new_height);
        return 0;
    } else if msg == WM_APP_REACTIVATE_FULLSCREEN {
        // Deferred half of the WM_ACTIVATEAPP branch above. Our own message,
        // consumed here; a stale post after the device left fullscreen is a
        // no-op, and a same-mode re-assert sets nothing (compare-first).
        // SAFETY: see WM_SETCURSOR branch — `dev_ptr` is live for the
        // lifetime of the subclass.
        let dev = unsafe { &mut *dev_ptr };
        dev.reactivate_fullscreen();
        return 0;
    }

    let _app_call = super::fullscreen::app_procedure_call(hwnd);
    call_window_proc(original_wndproc, hwnd, msg, wp, lp)
}

/// One call of `cursor_wnd_proc` the window's registration counted, ended when it drops.
///
/// Dropped after the game's procedure returned, on every return path. The
/// outermost call on a window ends the message dispatch every nested message
/// ran inside; a resize that suppressed the game's hides in it ends the
/// suppression there. Wine sends a restore from minimised as one
/// `WM_SYSCOMMAND`, and the `WM_SIZE` and the activation that follows it run
/// inside it, so a game that hides from any handler that runs after the
/// `WM_SIZE` in that dispatch is covered. The drop writes the device's
/// cursor flags under the `DEVICE_INSTANCES` lock, and sends no message and
/// takes no other lock there, so the lock stays a leaf.
struct HookCall {
    window: usize,
}

impl Drop for HookCall {
    fn drop(&mut self) {
        let disarmed = {
            let mut instances = DEVICE_INSTANCES
                .lock()
                .expect("device-instances mutex poisoned");
            instances.leave(self.window).is_some_and(|device| {
                // SAFETY: `leave` names only a device still registered on the
                // window, and the `DEVICE_INSTANCES` lock is held across the
                // use: a device's release unregisters it under that lock
                // before it frees anything. The reference is the whole device,
                // as in every branch of `cursor_wnd_proc`: a `WM_SIZE` that a
                // thunk's own window call dispatched on this thread runs while
                // that thunk holds the device too, and only the cursor flags
                // are written here.
                let device = unsafe { &mut *(device as *mut DeviceInner) };
                device.cursor_mut().end_resize_dispatch()
            })
        };
        if disarmed {
            debug!(
                target: LOG_TARGET,
                "wndproc: the dispatch that resized the window returned; game hides honoured again tid={}",
                current_thread_id(),
            );
        }
    }
}

// ── Helpers ──

/// Build the transparent HCURSOR software mode realizes.
///
/// All-zero colour bits under an all-ones AND mask: the mask keeps every screen
/// pixel and the colour adds nothing, a cursor of nothing. A cursor rather
/// than a null push, so the `WindowServer` cursor plane stays where it is; the
/// image the pointer shows is the overlay's business.
fn build_blank_hcursor() -> Option<*mut c_void> {
    const SIDE: usize = 32;
    let color = vec![0u32; SIDE * SIDE];
    // 1 bpp WORD-aligned rows: 32 pixels are 4 bytes per row.
    let mask = [0xFFu8; SIDE * 4];
    let cursor = create_cursor_from_bits(SIDE, SIDE, &color, &mask, (0, 0), "build_blank_hcursor")?;
    debug!(target: LOG_TARGET, "build_blank_hcursor: ok handle={cursor:p}");
    Some(cursor)
}

/// Create an HCURSOR from tight BGRA colour pixels and a 1 bpp AND mask.
///
/// The two DDBs are created, handed to `CreateIconIndirect` (which copies
/// them) and deleted again; every Win32 failure is logged under `what` and
/// yields `None`.
///
/// # Panics
///
/// If `width` or `height` does not fit an `i32`; cursor extents never
/// approach that.
fn create_cursor_from_bits(
    width: usize,
    height: usize,
    color: &[u32],
    mask: &[u8],
    (x_hotspot, y_hotspot): (u32, u32),
    what: &str,
) -> Option<*mut c_void> {
    let bitmap_width = i32::try_from(width).expect("cursor width fits i32");
    let bitmap_height = i32::try_from(height).expect("cursor height fits i32");
    let color_bitmap = create_bitmap_packed(
        bitmap_width,
        bitmap_height,
        1,
        32,
        color.as_ptr().cast::<c_void>(),
    );
    let mask_bitmap = create_bitmap_packed(
        bitmap_width,
        bitmap_height,
        1,
        1,
        mask.as_ptr().cast::<c_void>(),
    );
    if color_bitmap.is_null() || mask_bitmap.is_null() {
        let err = last_error();
        error!(
            target: LOG_TARGET,
            "{what}: CreateBitmap failed (color={color_bitmap:p} mask={mask_bitmap:p}) \
             {width}x{height}, GetLastError={err:#x}",
        );
        if !color_bitmap.is_null() {
            delete_object(color_bitmap);
        }
        if !mask_bitmap.is_null() {
            delete_object(mask_bitmap);
        }
        return None;
    }
    let info = ICONINFO {
        f_icon: 0, // cursor
        x_hotspot,
        y_hotspot,
        hbm_mask: mask_bitmap,
        hbm_color: color_bitmap,
    };
    let cursor = create_icon_indirect(&info);
    let err = last_error();
    // CreateIconIndirect copies the bitmaps; we own the originals.
    delete_object(color_bitmap);
    delete_object(mask_bitmap);
    if cursor.is_null() {
        error!(
            target: LOG_TARGET,
            "{what}: CreateIconIndirect returned null ({width}x{height}, GetLastError={err:#x})"
        );
        None
    } else {
        Some(cursor)
    }
}

/// Build a Win32 HCURSOR from a BGRA bitmap.
///
/// Returns `None` on any Win32 failure. Source pixels are upscaled by
/// `scale` so the cursor matches the display's `backingScaleFactor`: 2×
/// uses xBR (`xbr` crate), and every other factor falls back to
/// nearest-neighbor, since `xbr` implements no other factor. `scale == 1`
/// is the identity path. Hotspot is multiplied by `scale`.
///
/// # Panics
///
/// Pixel allocations can fail if the process exhausts memory. Dimension and
/// hotspot arithmetic are validated before reading the locked surface.
fn build_hcursor(
    width: u32,
    height: u32,
    pitch: usize,
    src: *const u8,
    x_hotspot: u32,
    y_hotspot: u32,
    scale: u32,
) -> Option<*mut c_void> {
    let layout = BitmapLayout {
        width,
        height,
        x_hotspot,
        y_hotspot,
    };
    let Some(scaled) = layout.scaled(scale) else {
        warn!(target: LOG_TARGET, "build_hcursor: bitmap or hotspot scaling overflows");
        return None;
    };
    let w = width as usize;
    let h = height as usize;
    let scale = scale as usize;

    // Copy the locked surface into a tight w*h BGRA buffer (handles
    // pitch != w*4). The upscalers operate on tight buffers. Use
    // `read_unaligned` per pixel so the u8→u32 cast is alignment-agnostic;
    // D3DLOCKED_RECT guarantees u32 alignment in practice but threading
    // that through the type system is more friction than the unaligned
    // read costs.
    let mut src_pixels = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            // SAFETY: `src` is the `D3DLOCKED_RECT.bits` from the
            // locked cursor surface; `pitch * h` plus the in-row
            // offset `x * 4` stays within the locked region (the
            // surface is `w*h*4` BGRA bytes with the given pitch).
            let pixel_ptr = unsafe { src.add(y * pitch + x * 4) };
            // SAFETY: `pixel_ptr` points at 4 readable bytes within
            // the locked region; `read_unaligned` makes no alignment
            // assumption.
            src_pixels[y * w + x] = unsafe { core::ptr::read_unaligned(pixel_ptr.cast::<u32>()) };
        }
    }

    // Probe the source before the dispatch consumes `src_pixels`. The mask
    // decision keys on the *source* so the upscaler's output can't sneakily
    // flip the fallback (it can't today — the upscaler never invents alpha
    // that wasn't in the input — but the check is cheap and future-proof).
    let any_alpha = src_pixels.iter().any(|&px| (px >> 24) != 0);

    let (sw, sh, pixels, path) = scale_cursor_pixels(scale, w, h, src_pixels);
    let and_mask = derive_and_mask(&pixels, sw, sh, any_alpha);

    let hotspot = (scaled.x_hotspot, scaled.y_hotspot);
    let cursor = create_cursor_from_bits(sw, sh, &pixels, &and_mask, hotspot, "build_hcursor")?;
    debug!(
        target: LOG_TARGET,
        "build_hcursor: ok handle={cursor:p} src={width}x{height} → {sw}x{sh} path={path} any_alpha={any_alpha} hotspot=({},{})→({},{})",
        x_hotspot, y_hotspot, hotspot.0, hotspot.1,
    );
    Some(cursor)
}

/// Derive an AND mask from upscaled alpha.
///
/// The DDB mask is 1-bit-per-pixel with WORD-aligned rows; a bit of 1 means
/// "transparent, show screen", a bit of 0 means "opaque, use color".
/// Keying on the upscaled alpha lines the upscaler's smoothed edges up with
/// what the color bitmap actually shows on Wine's mono-cursor path (kicks in
/// whenever `create_alpha_bitmap` in user32 fails to find any alpha via
/// `GetDIBits` on the DDB we hand it). Wine's alpha-blend path ignores
/// the mask entirely, so this is harmless there.
///
/// Some cursors carry alpha=0 across the whole surface; deriving a
/// mask would leave the cursor fully transparent, so `any_alpha=false`
/// returns an all-zeros mask.
fn derive_and_mask(pixels: &[u32], sw: usize, sh: usize, any_alpha: bool) -> Vec<u8> {
    let mask_stride = sw.div_ceil(16) * 2;
    let mut and_mask = vec![0u8; mask_stride * sh];
    if any_alpha {
        for y in 0..sh {
            let row = &pixels[y * sw..(y + 1) * sw];
            for (x, &px) in row.iter().enumerate() {
                if (px >> 24) == 0 {
                    and_mask[y * mask_stride + x / 8] |= 1u8 << (7 - (x & 7));
                }
            }
        }
    }
    and_mask
}
