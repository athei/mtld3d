//! Minimal hand-rolled Win32 bindings + the window plumbing every harness needs.
//!
//! Kept to bare `extern "system"` blocks (house style — no `windows` crate
//! dependency); only the calls the tests exercise are declared.

use core::ffi::{c_char, c_void};
use std::sync::{Mutex, Once, PoisonError};

#[link(name = "user32")]
unsafe extern "system" {
    fn RegisterClassExA(wc: *const WndClassExA) -> u16;
    fn CreateWindowExA(
        ex_style: u32,
        class_name: *const c_char,
        window_name: *const c_char,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: usize,
        menu: usize,
        instance: usize,
        param: *const c_void,
    ) -> usize;
    fn DestroyWindow(hwnd: usize) -> i32;
    fn DefWindowProcA(hwnd: usize, msg: u32, wparam: usize, lparam: isize) -> isize;
    fn PeekMessageA(
        msg: *mut Msg,
        hwnd: usize,
        filter_min: u32,
        filter_max: u32,
        remove: u32,
    ) -> i32;
    fn TranslateMessage(msg: *const Msg) -> i32;
    fn DispatchMessageA(msg: *const Msg) -> isize;
    fn PostQuitMessage(exit_code: i32);
    fn LoadCursorA(instance: usize, cursor_name: *const c_char) -> usize;
    fn SendMessageA(hwnd: usize, msg: u32, wparam: usize, lparam: isize) -> isize;
    fn GetCursor() -> usize;
    fn SetCursor(cursor: usize) -> usize;
    fn GetWindowRect(hwnd: usize, rect: *mut Rect) -> i32;
    fn GetClientRect(hwnd: usize, rect: *mut Rect) -> i32;
    fn GetWindowLongA(hwnd: usize, index: i32) -> i32;
    fn GetSystemMetrics(index: i32) -> i32;
    fn EnumDisplaySettingsW(device_name: *const u16, mode_num: u32, dev_mode: *mut DevModeW)
    -> i32;
    fn SetWindowPos(
        hwnd: usize,
        insert_after: usize,
        x: i32,
        y: i32,
        cx: i32,
        cy: i32,
        flags: u32,
    ) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleA(name: *const c_char) -> usize;
    fn GetCurrentProcess() -> *mut c_void;
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    fn GetPixel(hdc: usize, x: i32, y: i32) -> u32;
    fn SetPixel(hdc: usize, x: i32, y: i32, color: u32) -> u32;
}

static FAILURE_EXIT_HOOK: Once = Once::new();

/// Exit code a test process ends with once an assertion has failed.
///
/// The value libtest itself exits with after a failed run.
const TEST_FAILURE_EXIT_CODE: u32 = 101;

/// Make a failed assertion end the test process with a failing exit code.
///
/// mtld3d's `d3d9.dll` terminates the process from its `DLL_PROCESS_DETACH`
/// once a device has been created (it cannot survive snmalloc's thread-local
/// teardown on Wine's 1 MB main-thread stack), and that `TerminateProcess`
/// exits with code 0 whatever libtest was exiting with, so a failing test
/// binary read as passing. The hook keeps the default hook's report, which
/// names the failing test (libtest runs each test on a thread named after
/// it), and then terminates with libtest's failure code right away, before
/// the detach path can overwrite it. The tests of the suite share the
/// process, so the ones in flight go down with it: the e2e runner marks the
/// named test failed and runs the rest again in a fresh process.
pub fn install_failure_exit_hook() {
    FAILURE_EXIT_HOOK.call_once(|| {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            default_hook(info);
            // SAFETY: Win32 GetCurrentProcess returns a pseudo-handle for
            // the current process.
            let process = unsafe { GetCurrentProcess() };
            // SAFETY: the current process's pseudo-handle; the documented
            // self-terminate form.
            unsafe { TerminateProcess(process, TEST_FAILURE_EXIT_CODE) };
        }));
    });
}

#[repr(C)]
struct WndClassExA {
    size: u32,
    style: u32,
    wnd_proc: unsafe extern "system" fn(usize, u32, usize, isize) -> isize,
    cls_extra: i32,
    wnd_extra: i32,
    instance: usize,
    icon: usize,
    cursor: usize,
    background: usize,
    menu_name: *const c_char,
    class_name: *const c_char,
    icon_sm: usize,
}

/// Win32 `MSG`. Public so `Harness` can own one for its pump loop.
#[repr(C)]
pub struct Msg {
    hwnd: usize,
    message: u32,
    wparam: usize,
    lparam: isize,
    time: u32,
    pt_x: i32,
    pt_y: i32,
}

const WM_DESTROY: u32 = 0x0002;
const WM_QUIT: u32 = 0x0012;
const CW_USEDEFAULT: i32 = 0x8000_0000_u32.cast_signed();
/// `WS_OVERLAPPEDWINDOW` — a normal framed window, initially hidden.
const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
/// `WS_VISIBLE` — the window is shown.
pub const WS_VISIBLE: u32 = 0x1000_0000;
/// `WS_POPUP` — no frame; what a fullscreen device window becomes.
pub const WS_POPUP: u32 = 0x8000_0000;
/// `WS_CAPTION` — title bar; dropped for a fullscreen device window.
pub const WS_CAPTION: u32 = 0x00C0_0000;
/// `WS_EX_TOPMOST` — above every non-topmost window.
pub const WS_EX_TOPMOST: u32 = 0x0000_0008;
/// `IDC_ARROW` standard cursor id (`MAKEINTRESOURCE(32512)`).
const IDC_ARROW: usize = 32512;

static REGISTER_CLASS: Once = Once::new();
const CLASS_NAME: &core::ffi::CStr = c"mtld3d_test_window";

extern "system" fn wnd_proc(hwnd: usize, msg: u32, wparam: usize, lparam: isize) -> isize {
    if msg == WM_DESTROY {
        // SAFETY: Win32 message-loop thunk with no preconditions.
        unsafe { PostQuitMessage(0) };
        return 0;
    }
    // SAFETY: Win32 message-loop thunk; the loader-supplied args are forwarded
    // verbatim to the default handler.
    unsafe { DefWindowProcA(hwnd, msg, wparam, lparam) }
}

fn register_class() {
    REGISTER_CLASS.call_once(|| {
        // SAFETY: Win32 thunk; null module name returns the current process
        // instance handle.
        let instance = unsafe { GetModuleHandleA(core::ptr::null()) };
        // SAFETY: Win32 thunk; `IDC_ARROW` is a standard predefined cursor id.
        // Without a class cursor Wine's macOS driver hides the pointer over the
        // client area.
        let cursor = unsafe { LoadCursorA(0, IDC_ARROW as *const c_char) };
        let size =
            u32::try_from(core::mem::size_of::<WndClassExA>()).expect("WNDCLASSEX size fits u32");
        let wc = WndClassExA {
            size,
            style: 0,
            wnd_proc,
            cls_extra: 0,
            wnd_extra: 0,
            instance,
            icon: 0,
            cursor,
            background: 0,
            menu_name: core::ptr::null(),
            class_name: CLASS_NAME.as_ptr(),
            icon_sm: 0,
        };
        // SAFETY: Win32 thunk; `&wc` is a fully-populated WNDCLASSEX valid for
        // the duration of the call.
        let atom = unsafe { RegisterClassExA(&raw const wc) };
        assert!(atom != 0, "RegisterClassExA failed");
    });
}

/// Create a test window of `width`×`height`.
///
/// Registers the shared window class once per process. `visible` controls
/// `WS_VISIBLE` — hidden is preferred for parallel headless runs; the macdrv
/// Metal layer still attaches because Wine creates the cocoa view when the
/// HWND is created, not when it is shown.
///
/// # Panics
///
/// Panics if `CreateWindowExA` fails.
pub fn create_window(width: i32, height: i32, visible: bool) -> usize {
    register_class();
    // SAFETY: Win32 thunk; null module name returns the current process handle.
    let instance = unsafe { GetModuleHandleA(core::ptr::null()) };
    let style = WS_OVERLAPPEDWINDOW | if visible { WS_VISIBLE } else { 0 };
    // SAFETY: Win32 thunk; the class atom is registered above, the c-strings are
    // valid for the call, and `instance` is this process's module handle.
    let hwnd = unsafe {
        CreateWindowExA(
            0,
            CLASS_NAME.as_ptr(),
            c"mtld3d test".as_ptr(),
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            width,
            height,
            0,
            0,
            instance,
            core::ptr::null(),
        )
    };
    assert!(hwnd != 0, "CreateWindowExA failed");
    hwnd
}

/// `SendMessageA` — synchronous dispatch, bypassing the queue.
///
/// The call runs straight through the window's (possibly subclassed) wndproc.
/// Lets tests synthesize the macdrv-posted messages (e.g. `WM_SIZE`)
/// deterministically.
pub fn send_message(hwnd: usize, msg: u32, wparam: usize, lparam: isize) -> isize {
    // SAFETY: Win32 thunk; `hwnd` is a window this process created.
    unsafe { SendMessageA(hwnd, msg, wparam, lparam) }
}

/// `GetCursor` — the calling thread's current cursor handle.
///
/// Reports what the last `SetCursor` on this thread pushed; 0 = none.
pub fn get_cursor() -> usize {
    // SAFETY: Win32 thunk with no preconditions.
    unsafe { GetCursor() }
}

/// `SetCursor` — overwrite the thread cursor.
///
/// Lets tests simulate an external clobber (native cursor taking over while
/// the pointer was outside).
pub fn set_cursor(cursor: usize) -> usize {
    // SAFETY: Win32 thunk; 0 (no cursor) is a valid argument.
    unsafe { SetCursor(cursor) }
}

/// Win32 `RECT`, as reported by `Harness::window_rect`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

/// `WM_ACTIVATEAPP` — the app-level activation message a fullscreen device answers.
pub const WM_ACTIVATEAPP: u32 = 0x001C;

/// `GWL_STYLE` — a window's style bits.
pub const GWL_STYLE: i32 = -16;
/// `GWL_EXSTYLE` — a window's extended style bits.
pub const GWL_EXSTYLE: i32 = -20;

/// `GetWindowRect` — a window's outer rect in screen coordinates.
///
/// # Panics
///
/// Panics if the call fails, which for a window this process owns means the
/// handle is already destroyed.
pub fn window_rect(hwnd: usize) -> Rect {
    let mut rect = Rect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // SAFETY: Win32 thunk; `hwnd` is a window this process created and `rect`
    // is an owned local.
    let ok = unsafe { GetWindowRect(hwnd, &raw mut rect) };
    assert!(ok != 0, "GetWindowRect failed");
    rect
}

/// `GetWindowLongA` — one of a window's `GWL_*` longs, as a bit mask.
pub fn window_long(hwnd: usize, index: i32) -> u32 {
    // SAFETY: Win32 thunk; `hwnd` is a window this process created and the
    // index is one of the documented constants.
    unsafe { GetWindowLongA(hwnd, index) }.cast_unsigned()
}

/// `GetClientRect` — a window's client area, origin (0, 0).
///
/// # Panics
///
/// Panics if the call fails, which for a window this process owns means the
/// handle is already destroyed.
pub fn client_rect(hwnd: usize) -> Rect {
    let mut rect = Rect {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    };
    // SAFETY: Win32 thunk; `hwnd` is a window this process created and `rect`
    // is an owned local.
    let ok = unsafe { GetClientRect(hwnd, &raw mut rect) };
    assert!(ok != 0, "GetClientRect failed");
    rect
}

/// Win32 `DEVMODEW`, display-device shape, as `EnumDisplaySettingsW` fills it.
///
/// Only the two resolution fields are read; everything before them has to be
/// laid out exactly so they land where the API writes them.
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

/// The primary display's current mode (`EnumDisplaySettingsW(ENUM_CURRENT_SETTINGS)`).
///
/// This is the mode a fullscreen device sets and puts back, read through the
/// same user32 entry point Wine's own display tests use.
///
/// # Panics
///
/// Panics if the query fails, which means the prefix has no display at all.
pub fn current_display_mode() -> (u32, u32) {
    const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;
    // SAFETY: `DevModeW` is all-integer POD, so the all-zero bit pattern is a
    // valid value.
    let mut dm: DevModeW = unsafe { core::mem::zeroed() };
    dm.size = DEV_MODE_SIZE;
    // SAFETY: Win32 thunk; a null device name selects the primary display
    // and `dm` is an owned local with `size` set per the API contract.
    let ok = unsafe { EnumDisplaySettingsW(core::ptr::null(), ENUM_CURRENT_SETTINGS, &raw mut dm) };
    assert!(
        ok != 0,
        "EnumDisplaySettingsW(ENUM_CURRENT_SETTINGS) failed"
    );
    (dm.pels_width, dm.pels_height)
}

/// Every size the primary display enumerates through this binary's own user32 import.
///
/// Walks indices from 0 until user32 says the list ends, once per mode, so
/// a size appears as many times as user32 lists it (once per depth and
/// rate). Read through the test binary's import, which is the one d3d9
/// redirects for the process's main module.
#[must_use]
pub fn enumerate_display_sizes() -> Vec<(u32, u32)> {
    let mut sizes = Vec::new();
    for mode_num in 0..u32::MAX {
        // SAFETY: `DevModeW` is all-integer POD, so the all-zero bit pattern is a
        // valid value.
        let mut dm: DevModeW = unsafe { core::mem::zeroed() };
        dm.size = DEV_MODE_SIZE;
        // SAFETY: Win32 thunk; a null device name selects the primary display
        // and `dm` is an owned local with `size` set per the API contract.
        let ok = unsafe { EnumDisplaySettingsW(core::ptr::null(), mode_num, &raw mut dm) };
        if ok == 0 {
            break;
        }
        sizes.push((dm.pels_width, dm.pels_height));
    }
    sizes
}

/// The primary display's current resolution (`SM_CXSCREEN` / `SM_CYSCREEN`).
pub fn screen_size() -> (u32, u32) {
    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;
    // SAFETY: Win32 thunk; both indices are documented constants.
    let width = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    // SAFETY: as above.
    let height = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    (
        u32::try_from(width).expect("screen width is positive"),
        u32::try_from(height).expect("screen height is positive"),
    )
}

/// `SetWindowPos` without z-order or activation changes.
///
/// This is the app-side move a game makes when it manages its own window;
/// tests use it to simulate an external resize of a device window.
///
/// # Panics
///
/// Panics if the call fails, which for a window this process owns means the
/// handle is already destroyed.
pub fn set_window_pos(hwnd: usize, x: i32, y: i32, width: i32, height: i32) {
    const SWP_NOZORDER: u32 = 0x0004;
    const SWP_NOACTIVATE: u32 = 0x0010;
    // SAFETY: Win32 thunk; `hwnd` is a window this process created and the
    // geometry is plain scalars.
    let ok = unsafe { SetWindowPos(hwnd, 0, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE) };
    assert!(ok != 0, "SetWindowPos failed");
}

/// Read one pixel of a device context as a `COLORREF` (`0x00BBGGRR`).
///
/// For an `IDirect3DSurface9::GetDC` memory DC, this is how a test observes
/// the pixels GDI sees through the DIB the DC wraps.
#[must_use]
pub fn dc_get_pixel(hdc: usize, x: i32, y: i32) -> u32 {
    // SAFETY: GDI thunk; `hdc` is a live device context and the coordinates
    // are plain scalars (an out-of-range one returns CLR_INVALID).
    unsafe { GetPixel(hdc, x, y) }
}

/// Paint one pixel of a device context, `color` a `COLORREF` (`0x00BBGGRR`).
///
/// Returns the colour GDI actually stored, which for a DIB of a
/// lower-precision format is the nearest representable one.
pub fn dc_set_pixel(hdc: usize, x: i32, y: i32, color: u32) -> u32 {
    // SAFETY: GDI thunk; `hdc` is a live device context and the coordinates
    // and colour are plain scalars.
    unsafe { SetPixel(hdc, x, y, color) }
}

/// `DestroyWindow`, one call at a time across the process.
///
/// Wine's Mac driver tears a window's client surfaces down under two locks
/// taken in opposite orders on two of its paths: `detach_client_surfaces`
/// holds the surface list and asks for the window data, `macdrv_DestroyWindow`
/// holds the window data and releases a surface. Two threads destroying
/// their windows at once can therefore deadlock inside the driver; one at a
/// time, they cannot.
static DESTROY_WINDOW: Mutex<()> = Mutex::new(());

/// Destroy a window created by [`create_window`].
pub fn destroy_window(hwnd: usize) {
    let _one_at_a_time = DESTROY_WINDOW
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    // SAFETY: Win32 thunk; `hwnd` is a window this process created.
    let ret = unsafe { DestroyWindow(hwnd) };
    assert!(ret != 0, "DestroyWindow failed");
}

/// Drain the message queue. Returns `false` once `WM_QUIT` is seen.
pub fn pump_messages(msg: &mut Msg) -> bool {
    // SAFETY: Win32 thunk; `msg` is a valid &mut MSG, hwnd 0 pumps the thread queue.
    while unsafe { PeekMessageA(msg, 0, 0, 0, 1) } != 0 {
        if msg.message == WM_QUIT {
            return false;
        }
        // SAFETY: Win32 thunk; both calls only read the populated `msg`.
        unsafe { TranslateMessage(msg) };
        // SAFETY: Win32 thunk; both calls only read the populated `msg`.
        unsafe { DispatchMessageA(msg) };
    }
    msg.message != WM_QUIT
}

/// A zeroed `MSG` for the pump loop.
///
/// The fields are overwritten by `PeekMessageA` before any are read.
#[must_use]
pub const fn zeroed_msg() -> Msg {
    Msg {
        hwnd: 0,
        message: 0,
        wparam: 0,
        lparam: 0,
        time: 0,
        pt_x: 0,
        pt_y: 0,
    }
}
