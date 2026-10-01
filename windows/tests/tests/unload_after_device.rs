//! A `FreeLibrary` of `d3d9.dll` after `CreateDevice` leaves the process running.
//!
//! A launcher or a settings tool loads `d3d9.dll`, creates a device to probe
//! the caps or the modes, releases it, frees the library and carries on. The
//! first `CreateDevice` pins the image, so that `FreeLibrary` leaves it
//! mapped, and the `DLL_PROCESS_DETACH` that ends a process which created a
//! device only comes at the process's exit. The test runs the sequence twice
//! in one process, the second time as the game that starts after the probe,
//! asserts after each `FreeLibrary` that the module is still mapped, and then
//! ends the process with a status of its own. A detach that ended the process
//! at a `FreeLibrary` would end it while the test runs, before any result or
//! declaration, and the runner in `unix/e2e` fails a test whose process ends
//! that way. The declared status is the second check: the runner fails the
//! test unless the process ended with exactly that code, which proves the
//! exit-status hook and the `TerminateProcess` at the real exit survived both
//! `FreeLibrary` cycles.
//!
//! It is a binary of its own because the pin outlives the test, and
//! `unload.rs` needs an image no device ever pinned. It does not use the
//! shared harness either: that links `d3d9.dll` through `raw-dylib`, and a
//! static import keeps the module mapped whether or not it is pinned.

use core::ffi::{c_char, c_void};

use mtld3d_types::{
    D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DDEVTYPE_HAL, D3DFMT_X8R8G8B8, D3DPRESENT_PARAMETERS,
    D3DSDK_VERSION, D3DSWAPEFFECT_DISCARD, IDirect3D9Vtbl,
};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn ExitProcess(exit_code: u32) -> !;
}

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
        instance: *mut c_void,
        param: *const c_void,
    ) -> usize;
    fn DestroyWindow(hwnd: usize) -> i32;
    fn DefWindowProcA(hwnd: usize, msg: u32, wparam: usize, lparam: isize) -> isize;
}

type Direct3DCreate9Fn = unsafe extern "system" fn(u32) -> *mut c_void;
type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
type WndProc = unsafe extern "system" fn(usize, u32, usize, isize) -> isize;

/// `WNDCLASSEXA`, the window class this binary registers for its device windows.
#[repr(C)]
struct WndClassExA {
    size: u32,
    style: u32,
    wnd_proc: WndProc,
    cls_extra: i32,
    wnd_extra: i32,
    instance: *mut c_void,
    icon: usize,
    cursor: usize,
    background: usize,
    menu_name: *const c_char,
    class_name: *const c_char,
    icon_sm: usize,
}

/// `WS_POPUP`: no frame, and without `WS_VISIBLE` the window stays hidden.
const WS_POPUP: u32 = 0x8000_0000;
/// `IUnknown::Release` vtable slot.
const RELEASE_SLOT: usize = 2;
/// The device window's size, which the back buffer takes.
const WINDOW_SIZE: u32 = 64;
const CLASS_NAME: &core::ffi::CStr = c"mtld3d_unload_after_device";

/// The status this process ends with; no other exit in the suite uses it.
const EXIT_STATUS: u32 = 43;

/// The libtest name of the test below, which the runner reads back off stdout.
const TEST_NAME: &str = "free_library_after_a_device_leaves_the_process_running";

#[test]
fn free_library_after_a_device_leaves_the_process_running() {
    register_class();
    for cycle in ["the probe", "the game after it"] {
        load_create_device_and_free(cycle);
        // SAFETY: plain kernel32 lookup by name.
        let still = unsafe { GetModuleHandleA(c"d3d9.dll".as_ptr()) };
        assert!(
            !still.is_null(),
            "d3d9.dll was unmapped by the FreeLibrary after {cycle}, so the first CreateDevice did not pin it"
        );
    }
    // A leading newline closes libtest's open `test <name> ... ` line so the
    // marker starts one of its own.
    println!("\n[e2e] test {TEST_NAME} ends this process with exit code {EXIT_STATUS}");
    // SAFETY: kernel32's documented process exit, called with a status; it
    // does not return.
    unsafe { ExitProcess(EXIT_STATUS) }
}

/// One launch of the library: load, a device on a hidden window, release everything, free.
fn load_create_device_and_free(cycle: &str) {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll) for {cycle}");
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let create = unsafe { GetProcAddress(lib, c"Direct3DCreate9".as_ptr()) };
    assert!(
        !create.is_null(),
        "GetProcAddress(Direct3DCreate9) for {cycle}"
    );
    // SAFETY: the export is `Direct3DCreate9` with the documented signature.
    let create: Direct3DCreate9Fn = unsafe { core::mem::transmute(create) };
    // SAFETY: calling the resolved export with the SDK version it accepts.
    let d3d9 = unsafe { create(D3DSDK_VERSION) };
    assert!(!d3d9.is_null(), "Direct3DCreate9 returned null for {cycle}");

    let hwnd = create_window();
    let mut pp = D3DPRESENT_PARAMETERS {
        back_buffer_width: WINDOW_SIZE,
        back_buffer_height: WINDOW_SIZE,
        back_buffer_format: D3DFMT_X8R8G8B8,
        back_buffer_count: 1,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        swap_effect: D3DSWAPEFFECT_DISCARD,
        device_window: hwnd,
        windowed: 1,
        enable_auto_depth_stencil: 0,
        auto_depth_stencil_format: 0,
        flags: 0,
        full_screen_refresh_rate_in_hz: 0,
        presentation_interval: 0,
    };
    // SAFETY: a COM object's first word is its vtable pointer, and `d3d9` is
    // a live `IDirect3D9`.
    let vtbl = unsafe { *d3d9.cast::<*const IDirect3D9Vtbl>() };
    // SAFETY: the vtable of a live `IDirect3D9` outlives the interface.
    let vtbl = unsafe { &*vtbl };
    let mut device: *mut c_void = core::ptr::null_mut();
    // SAFETY: `d3d9` is live, `pp` and `device` are writable for the call,
    // and a null focus window is permitted with a device window set.
    let hr = unsafe {
        (vtbl.create_device)(
            d3d9,
            0,
            D3DDEVTYPE_HAL,
            core::ptr::null_mut(),
            D3DCREATE_HARDWARE_VERTEXPROCESSING,
            (&raw mut pp).cast::<c_void>(),
            &raw mut device,
        )
    };
    assert_eq!(hr, 0, "CreateDevice for {cycle} failed: 0x{hr:08X}");
    assert!(
        !device.is_null(),
        "CreateDevice for {cycle} returned a null device"
    );

    // SAFETY: `device` is the live device created above, and this cycle owns
    // its only reference.
    let remaining = unsafe { release(device) };
    assert_eq!(
        remaining, 0,
        "Release of the only device reference for {cycle}"
    );
    // SAFETY: the window this cycle created, no longer used by any device.
    let destroyed = unsafe { DestroyWindow(hwnd) };
    assert_ne!(destroyed, 0, "DestroyWindow for {cycle}");
    // SAFETY: `d3d9` is the live interface created above, and this cycle owns
    // its only reference.
    let remaining = unsafe { release(d3d9) };
    assert_eq!(
        remaining, 0,
        "Release of the only IDirect3D9 reference for {cycle}"
    );
    // SAFETY: balancing the LoadLibrary above.
    let freed = unsafe { FreeLibrary(lib) };
    assert_ne!(freed, 0, "FreeLibrary(d3d9.dll) after {cycle}");
}

/// Release one reference on a live COM object; the count it returns.
///
/// # Safety
///
/// `object` is a live COM object and the caller owns the reference this
/// gives back.
unsafe fn release(object: *mut c_void) -> u32 {
    // SAFETY: per the contract `object` is a live COM object, whose first
    // word is its vtable pointer.
    let vtable = unsafe { *object.cast::<*const ReleaseFn>() };
    // SAFETY: every COM vtable has at least the three IUnknown slots.
    let slot = unsafe { vtable.add(RELEASE_SLOT) };
    // SAFETY: slot 2 of a COM vtable is Release, with the signature declared above.
    let release: ReleaseFn = unsafe { *slot };
    // SAFETY: per the contract the caller owns the reference this gives back.
    unsafe { release(object) }
}

fn register_class() {
    // SAFETY: plain kernel32 call; a null name is this executable.
    let instance = unsafe { GetModuleHandleA(core::ptr::null()) };
    let wc = WndClassExA {
        size: u32::try_from(size_of::<WndClassExA>()).expect("WNDCLASSEXA size fits u32"),
        style: 0,
        wnd_proc: DefWindowProcA,
        cls_extra: 0,
        wnd_extra: 0,
        instance,
        icon: 0,
        cursor: 0,
        background: 0,
        menu_name: core::ptr::null(),
        class_name: CLASS_NAME.as_ptr(),
        icon_sm: 0,
    };
    // SAFETY: `wc` is a fully populated WNDCLASSEXA, valid for the call.
    let atom = unsafe { RegisterClassExA(&raw const wc) };
    assert_ne!(atom, 0, "RegisterClassExA");
}

fn create_window() -> usize {
    // SAFETY: plain kernel32 call; a null name is this executable.
    let instance = unsafe { GetModuleHandleA(core::ptr::null()) };
    let size = i32::try_from(WINDOW_SIZE).expect("window size fits i32");
    // SAFETY: the class is registered by `register_class`, the strings are
    // NUL-terminated and live for the call, and `instance` is this executable.
    let hwnd = unsafe {
        CreateWindowExA(
            0,
            CLASS_NAME.as_ptr(),
            c"mtld3d unload after device".as_ptr(),
            WS_POPUP,
            0,
            0,
            size,
            size,
            0,
            0,
            instance,
            core::ptr::null(),
        )
    };
    assert_ne!(hwnd, 0, "CreateWindowExA");
    hwnd
}
