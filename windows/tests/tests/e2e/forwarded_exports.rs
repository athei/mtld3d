//! Every name Windows' `d3d9.dll` exports, bar the 9On12 factories, resolves by name.
//!
//! A proxy `d3d9.dll` in a game folder (a mod loader) exports the names
//! Windows' d3d9.dll has and forwards each through the address
//! `GetProcAddress` gave it from the real one, with no null check. A name the
//! layer does not export becomes a jump to address 0 as soon as anything calls
//! the proxy's copy: `d3dx9_43.dll` looks up `DebugSetMute` and calls it when
//! found. The test resolves every forwarded name, calls the debug and
//! software-pipeline stubs, and on i386 checks that `DebugSetMute` returns
//! without popping the argument `d3dx9` pushes and pops itself.
//! `Direct3DCreate9On12` and `Direct3DCreate9On12Ex` stay unexported, as on a
//! d3d9.dll without the 9On12 bridge (`docs/STATUS.md`).
//!
//! No shared harness: the point is the export table, as in `d3dperf.rs`.

use core::ffi::{CStr, c_char, c_void};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

type NoArgFn = unsafe extern "system" fn();
type NoArgI32Fn = unsafe extern "system" fn() -> i32;
type ShimFn = unsafe extern "system" fn(u32) -> i32;
type PsgpErrorFn = unsafe extern "system" fn(*mut c_void, u32, u32);
type PsgpSampleTextureFn =
    unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32, *mut c_void);

/// The names Windows' d3d9.dll exports that a forwarding proxy resolves from the layer.
const FORWARDED: [&CStr; 15] = [
    c"D3DPERF_BeginEvent",
    c"D3DPERF_EndEvent",
    c"D3DPERF_GetStatus",
    c"D3DPERF_QueryRepeatFrame",
    c"D3DPERF_SetMarker",
    c"D3DPERF_SetOptions",
    c"D3DPERF_SetRegion",
    c"DebugSetLevel",
    c"DebugSetMute",
    c"Direct3D9EnableMaximizedWindowedModeShim",
    c"Direct3DCreate9",
    c"Direct3DCreate9Ex",
    c"Direct3DShaderValidatorCreate9",
    c"PSGPError",
    c"PSGPSampleTexture",
];

fn resolve(lib: *mut c_void, name: &CStr) -> *mut c_void {
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    unsafe { GetProcAddress(lib, name.as_ptr()) }
}

#[test]
fn every_forwarded_export_resolves_and_the_stubs_return() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");

    let missing: Vec<&str> = FORWARDED
        .iter()
        .filter(|name| resolve(lib, name).is_null())
        .map(|name| name.to_str().unwrap_or("?"))
        .collect();
    assert!(
        missing.is_empty(),
        "exports that do not resolve: {missing:?}"
    );
    for name in [c"Direct3DCreate9On12", c"Direct3DCreate9On12Ex"] {
        assert!(
            resolve(lib, name).is_null(),
            "{} is exported, but there is no 9On12 bridge behind it",
            name.to_str().unwrap_or("?")
        );
    }

    // SAFETY: each export has the d3d9 signature for its name.
    let mute: NoArgFn = unsafe { core::mem::transmute(resolve(lib, c"DebugSetMute")) };
    // SAFETY: as above.
    let level: NoArgI32Fn = unsafe { core::mem::transmute(resolve(lib, c"DebugSetLevel")) };
    // SAFETY: as above.
    let shim: ShimFn =
        unsafe { core::mem::transmute(resolve(lib, c"Direct3D9EnableMaximizedWindowedModeShim")) };
    // SAFETY: as above.
    let psgp_error: PsgpErrorFn = unsafe { core::mem::transmute(resolve(lib, c"PSGPError")) };
    // SAFETY: as above.
    let psgp_sample: PsgpSampleTextureFn =
        unsafe { core::mem::transmute(resolve(lib, c"PSGPSampleTexture")) };

    // SAFETY: the resolved argument-less export.
    unsafe { mute() };
    // SAFETY: as above.
    assert_eq!(unsafe { level() }, 0, "DebugSetLevel: no debug runtime");
    // SAFETY: the resolved export with its one flag argument.
    let shim_answer = unsafe { shim(1) };
    assert_eq!(shim_answer, 0, "Direct3D9EnableMaximizedWindowedModeShim");
    // SAFETY: the stubs read none of their pointers.
    unsafe { psgp_error(core::ptr::null_mut(), 0, 0) };
    // SAFETY: as above.
    unsafe {
        psgp_sample(
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
            0,
            core::ptr::null_mut(),
        );
    };

    #[cfg(target_arch = "x86")]
    assert_eq!(
        stack_drift_with_one_pushed_argument(resolve(lib, c"DebugSetMute")),
        0,
        "DebugSetMute popped the argument its caller pushed"
    );

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}

/// The bytes `esp` moved across a call of `f` that pushes one argument and pops it after.
///
/// `d3dx9_43.dll` calls `DebugSetMute` this way (`push esi; call eax; pop
/// ecx`), so a callee that popped the argument itself would leave the caller's
/// stack four bytes off.
#[cfg(target_arch = "x86")]
fn stack_drift_with_one_pushed_argument(f: *mut c_void) -> usize {
    let drift: usize;
    // SAFETY: `f` is the resolved `DebugSetMute`, which takes no arguments and
    // touches nothing the caller owns; `edi` is callee-saved, so it holds the
    // stack pointer across the call.
    unsafe {
        core::arch::asm!(
            "mov edi, esp",
            "push 0",
            "call {f}",
            "add esp, 4",
            "sub edi, esp",
            f = in(reg) f,
            out("edi") drift,
            clobber_abi("system"),
        );
    }
    drift
}
