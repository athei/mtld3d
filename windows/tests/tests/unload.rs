//! `FreeLibrary` must leave no pointer into the unmapped `d3d9.dll` behind.
//!
//! Launchers and benchmarks load `d3d9.dll`, probe `Direct3DCreate9` and the
//! caps, and free it again; the process then carries on. Its next exception
//! (MFC and the C++ runtime throw as control flow) walks the vectored-handler
//! chain, and a handler still registered from the unloaded image faults on
//! instruction fetch. That fault raises the next exception, which reaches the
//! same stale handler, and the main thread's stack is gone within
//! milliseconds. 3DMark05 died exactly like this after its capability check.
//!
//! The test replays the sequence: load, create, release, free, then raise a
//! continuable exception with a resuming handler of its own appended at the
//! END of the chain, so anything left at the front by the unloaded DLL runs
//! first. With a stale handler the process dies; without one `RaiseException`
//! returns and the probe flag is set.
//!
//! It deliberately does not use the shared harness: that links `d3d9.dll`
//! through `raw-dylib`, and a static import keeps the module mapped no matter
//! how often it is freed.

use core::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
    fn AddVectoredExceptionHandler(first: u32, handler: VectoredHandler) -> *mut c_void;
    fn RemoveVectoredExceptionHandler(handle: *mut c_void) -> u32;
    fn RaiseException(code: u32, flags: u32, n_args: u32, args: *const usize);
}

type VectoredHandler = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;
type Direct3DCreate9Fn = unsafe extern "system" fn(u32) -> *mut c_void;
type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;

#[repr(C)]
struct ExceptionRecord {
    code: u32,
    flags: u32,
    nested: *mut Self,
    address: *mut c_void,
    n_params: u32,
    information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    record: *mut ExceptionRecord,
    context: *mut c_void,
}

const D3D_SDK_VERSION: u32 = 32;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
/// A continuable, application-defined code: nothing else in the process raises it.
const PROBE_CODE: u32 = 0xE0C0_FFEE;
/// `IUnknown::Release` vtable slot.
const RELEASE_SLOT: usize = 2;

static PROBE_SEEN: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn resume_probe(ep: *mut ExceptionPointers) -> i32 {
    // SAFETY: the dispatcher hands a valid EXCEPTION_POINTERS for the call.
    let record = unsafe { (*ep).record };
    // SAFETY: the record pointer is valid for the handler's duration.
    if unsafe { (*record).code } != PROBE_CODE {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    PROBE_SEEN.store(true, Ordering::Release);
    EXCEPTION_CONTINUE_EXECUTION
}

#[test]
fn exception_after_free_library_survives() {
    // SAFETY: plain kernel32 call with a NUL-terminated name.
    let lib = unsafe { LoadLibraryA(c"d3d9.dll".as_ptr()) };
    assert!(!lib.is_null(), "LoadLibrary(d3d9.dll)");
    // SAFETY: `lib` is a live module handle and the name is NUL-terminated.
    let create = unsafe { GetProcAddress(lib, c"Direct3DCreate9".as_ptr()) };
    assert!(!create.is_null(), "GetProcAddress(Direct3DCreate9)");
    // SAFETY: the export is `Direct3DCreate9` with the documented signature.
    let create: Direct3DCreate9Fn = unsafe { core::mem::transmute(create) };
    // SAFETY: calling the resolved export with the SDK version it accepts.
    let d3d9 = unsafe { create(D3D_SDK_VERSION) };
    assert!(!d3d9.is_null(), "Direct3DCreate9 returned null");
    // SAFETY: a COM object's first word is its vtable pointer.
    let vtable = unsafe { *d3d9.cast::<*const ReleaseFn>() };
    // SAFETY: every COM vtable has at least the three IUnknown slots.
    let slot = unsafe { vtable.add(RELEASE_SLOT) };
    // SAFETY: slot 2 of a COM vtable is Release, with the signature declared above.
    let release: ReleaseFn = unsafe { *slot };
    // SAFETY: releasing the one reference `Direct3DCreate9` handed out.
    let remaining = unsafe { release(d3d9) };
    assert_eq!(remaining, 0, "Release of the only IDirect3D9 reference");

    // SAFETY: balancing the LoadLibrary above.
    assert_ne!(unsafe { FreeLibrary(lib) }, 0, "FreeLibrary(d3d9.dll)");
    // SAFETY: plain kernel32 lookup by name.
    let still = unsafe { GetModuleHandleA(c"d3d9.dll".as_ptr()) };
    assert!(
        still.is_null(),
        "d3d9.dll is still mapped after FreeLibrary, so this test cannot prove anything"
    );

    // SAFETY: appending (first = 0) a handler that stays valid for the test.
    let handle = unsafe { AddVectoredExceptionHandler(0, resume_probe) };
    assert!(!handle.is_null(), "AddVectoredExceptionHandler");
    // SAFETY: a continuable exception (flags 0) with no parameters; the handler
    // above resumes execution, so the call returns.
    unsafe { RaiseException(PROBE_CODE, 0, 0, core::ptr::null()) };
    assert!(
        PROBE_SEEN.load(Ordering::Acquire),
        "the resuming handler never ran"
    );
    // SAFETY: `handle` came from AddVectoredExceptionHandler above.
    assert_ne!(unsafe { RemoveVectoredExceptionHandler(handle) }, 0);
}
