//! The `x86_64` DLL's own `memcpy`, `memmove`, `memset` and `memcmp`.
//!
//! Defined here, every call the compiler emits for one of them, from the Rust
//! and from the C and C++ linked into the image, resolves inside the image
//! instead of to `vcruntime140.dll`, and `mtld3d_core::guest_mem` routes it:
//! to the routines there under the x64 emulator, to the CRT's everywhere
//! else. The i386 DLL keeps the CRT's, since an i386 process never runs an
//! ARM64X CRT, and so does the ARM64X build, whose own code is native.
//!
//! The first call decides the route, and it can come before `DllMain`, from
//! the CRT's startup or a TLS callback. That is safe: the loader binds this
//! image's imports before it runs any of its code, the lookups the decision
//! makes (two `GetModuleHandleA`, four `GetProcAddress`) are allowed under the
//! loader lock, and the emulator and `ucrtbase.dll` are loaded before this
//! image is.

use core::{
    ffi::{CStr, c_void},
    ptr::NonNull,
};

use log::{info, warn};
use mtld3d_core::guest_mem::{CrtMem, LOCAL_COPY_MAX, LatchedRoute, MemHost, MemRoute};

use crate::LOG_TARGET;

/// Where this image's four routines go.
///
/// A machine fact, latched once by the first call and immutable after:
/// whether the x64 emulator runs this process, and where the CRT's routines
/// are. The C routines have no object to reach, and the answer is the same
/// for every device.
static ROUTE: MemRoute = MemRoute::new();

unsafe extern "system" {
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, proc_name: *const u8) -> *mut c_void;
}

/// The C `memcpy`.
///
/// # Safety
///
/// As C's: `src` must be valid for `n` bytes of reads, `dst` for `n` bytes of
/// writes, and the two ranges must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    // SAFETY: the caller's contract is `memcpy`'s, which the route's is.
    unsafe { ROUTE.memcpy(probe_host, dst, src, n) }
}

/// The C `memmove`.
///
/// # Safety
///
/// As C's: `src` must be valid for `n` bytes of reads and `dst` for `n` bytes
/// of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memmove(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
    // SAFETY: the caller's contract is `memmove`'s, which the route's is.
    unsafe { ROUTE.memmove(probe_host, dst, src, n) }
}

/// The C `memset`.
///
/// # Safety
///
/// As C's: `dst` must be valid for `n` bytes of writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(dst: *mut c_void, value: i32, n: usize) -> *mut c_void {
    // SAFETY: the caller's contract is `memset`'s, which the route's is.
    unsafe { ROUTE.memset(probe_host, dst, value, n) }
}

/// The C `memcmp`.
///
/// # Safety
///
/// As C's: `a` and `b` must each be valid for `n` bytes of reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcmp(a: *const c_void, b: *const c_void, n: usize) -> i32 {
    // SAFETY: the caller's contract is `memcmp`'s, which the route's is.
    unsafe { ROUTE.memcmp(probe_host, a, b, n) }
}

/// Logs where the four routines go, deciding it if no call has yet.
pub fn log_route() {
    match ROUTE.latch(probe_host) {
        LatchedRoute::Emulated => info!(
            target: LOG_TARGET,
            "memory routines: in-image under the x64 emulator, the CRT's for copies over {LOCAL_COPY_MAX} bytes"
        ),
        LatchedRoute::Crt => info!(target: LOG_TARGET, "memory routines: the CRT's"),
        LatchedRoute::NoCrt => warn!(
            target: LOG_TARGET,
            "memory routines: ucrtbase.dll's routines not found, in-image for every call"
        ),
        LatchedRoute::Undecided => {
            info!(target: LOG_TARGET, "memory routines: another thread is deciding the route");
        }
    }
}

/// Whether the x64 emulator runs this process, and where the CRT's routines are.
///
/// `xtajit64.dll` is the x64 emulator an arm64 Windows or Wine loads into
/// every x64 process at startup, and into no other.
fn probe_host() -> MemHost {
    // SAFETY: the name is NUL-terminated, and the lookup only reads it.
    let emulator = unsafe { GetModuleHandleA(c"xtajit64.dll".as_ptr().cast::<u8>()) };
    MemHost::new(!emulator.is_null(), crt_routines())
}

/// The entry points of `ucrtbase.dll`'s four routines, if it exports them all.
fn crt_routines() -> Option<CrtMem> {
    // SAFETY: the name is NUL-terminated, and the lookup only reads it.
    let ucrt = unsafe { GetModuleHandleA(c"ucrtbase.dll".as_ptr().cast::<u8>()) };
    if ucrt.is_null() {
        return None;
    }
    let export = |name: &CStr| {
        // SAFETY: `ucrt` is the handle of a loaded module, and `name` is
        // NUL-terminated.
        NonNull::new(unsafe { GetProcAddress(ucrt, name.as_ptr().cast::<u8>()) })
    };
    let copy = export(c"memcpy")?;
    let copy_overlapping = export(c"memmove")?;
    let fill = export(c"memset")?;
    let compare = export(c"memcmp")?;
    // SAFETY: each address is ucrtbase's export of the routine it is passed
    // as, with the C signature. ucrtbase is a load-time dependency of this
    // image, so it stays loaded for as long as the image's routines run.
    Some(unsafe { CrtMem::new(copy, copy_overlapping, fill, compare) })
}
