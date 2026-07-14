//! Minimal hand-rolled Win32 bindings + the window plumbing every harness needs.
//!
//! Kept to bare `extern "system"` blocks (house style — no `windows` crate
//! dependency); only the calls the tests exercise are declared.

use core::ffi::{c_char, c_void};
use std::sync::Once;

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
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleA(name: *const c_char) -> usize;
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
const WS_VISIBLE: u32 = 0x1000_0000;
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

/// Destroy a window created by [`create_window`].
pub fn destroy_window(hwnd: usize) {
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
