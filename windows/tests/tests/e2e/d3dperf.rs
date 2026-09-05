//! The `D3DPERF_*` exports resolve by name and act like a d3d9 with no profiler.
//!
//! Engines resolve the PIX marker family from `d3d9.dll` in one table and treat
//! a `NULL` entry as a broken Direct3D: an in-game overlay SDK refuses to
//! initialise when any of the seven is missing, while the game itself renders
//! fine. The test resolves every one the way an engine does (`GetProcAddress`
//! on the loaded module, not a static import) and pins the no-profiler
//! contract: no nesting, no repeat-frame request, no status bits.
//!
//! No shared harness: the point is the export table, and the harness's
//! `raw-dylib` link would make a missing export a link error rather than a
//! test failure.

use core::ffi::{c_char, c_void};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

type EventFn = unsafe extern "system" fn(u32, *const u16) -> i32;
type MarkerFn = unsafe extern "system" fn(u32, *const u16);
type NoArgI32Fn = unsafe extern "system" fn() -> i32;
type NoArgU32Fn = unsafe extern "system" fn() -> u32;
type SetOptionsFn = unsafe extern "system" fn(u32);

const D3DCOLOR_WHITE: u32 = 0xFFFF_FFFF;
const NAME: [u16; 4] = [b'p' as u16, b'i' as u16, b'x' as u16, 0];

fn resolve(lib: *mut c_void, name: &'static core::ffi::CStr) -> *mut c_void {
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let addr = unsafe { GetProcAddress(lib, name.as_ptr()) };
    assert!(
        !addr.is_null(),
        "GetProcAddress({})",
        name.to_str().unwrap_or("?")
    );
    addr
}

#[test]
fn d3dperf_family_resolves_and_reports_no_profiler() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");

    // Every name in the table the game walks; a missing one fails here rather
    // than as a NULL the game trips over later.
    let names = [
        c"D3DPERF_BeginEvent",
        c"D3DPERF_EndEvent",
        c"D3DPERF_SetMarker",
        c"D3DPERF_SetRegion",
        c"D3DPERF_QueryRepeatFrame",
        c"D3DPERF_SetOptions",
        c"D3DPERF_GetStatus",
    ];
    let addrs = names.map(|name| resolve(lib, name));

    // SAFETY: each export has the documented d3d9 signature for its name.
    let begin: EventFn = unsafe { core::mem::transmute(addrs[0]) };
    // SAFETY: as above.
    let end: NoArgI32Fn = unsafe { core::mem::transmute(addrs[1]) };
    // SAFETY: as above.
    let set_marker: MarkerFn = unsafe { core::mem::transmute(addrs[2]) };
    // SAFETY: as above.
    let set_region: MarkerFn = unsafe { core::mem::transmute(addrs[3]) };
    // SAFETY: as above.
    let query_repeat_frame: NoArgI32Fn = unsafe { core::mem::transmute(addrs[4]) };
    // SAFETY: as above.
    let set_options: SetOptionsFn = unsafe { core::mem::transmute(addrs[5]) };
    // SAFETY: as above.
    let get_status: NoArgU32Fn = unsafe { core::mem::transmute(addrs[6]) };

    // A nested begin/end pair: without a profiler no nesting is reported.
    // SAFETY: the resolved export with a live NUL-terminated wide name.
    let outer = unsafe { begin(D3DCOLOR_WHITE, NAME.as_ptr()) };
    // SAFETY: as above.
    let inner = unsafe { begin(D3DCOLOR_WHITE, NAME.as_ptr()) };
    // SAFETY: the resolved argument-less export.
    let after_inner = unsafe { end() };
    // SAFETY: as above.
    let after_outer = unsafe { end() };
    assert_eq!(
        (outer, inner, after_inner, after_outer),
        (0, 0, 0, 0),
        "no profiler: no nesting is reported"
    );

    // SAFETY: the resolved export with a live NUL-terminated wide name.
    unsafe { set_marker(D3DCOLOR_WHITE, NAME.as_ptr()) };
    // SAFETY: as above.
    unsafe { set_region(D3DCOLOR_WHITE, NAME.as_ptr()) };
    // SAFETY: the resolved export with the documented flags argument.
    unsafe { set_options(0) };
    // SAFETY: the resolved argument-less export.
    let repeat = unsafe { query_repeat_frame() };
    assert_eq!(repeat, 0, "no profiler asks for a frame replay");
    // SAFETY: as above.
    let status = unsafe { get_status() };
    assert_eq!(status, 0, "no profiler is attached");

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
}
