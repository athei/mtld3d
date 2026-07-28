use core::ffi::c_void;

use log::{error, info};
use mtld3d_shared::identity;

const LOG_TARGET: &str = "mtld3d::shim";

// Wine unixlib symbols from winecrt0's unix_lib.o.
// __wine_unixlib_handle: set by loader after __wine_init_unix_call succeeds.
// __wine_unix_call_dispatcher: function pointer, initially a lazy-init stub,
// patched by Wine loader to the real dispatcher.
unsafe extern "C" {
    static __wine_unixlib_handle: u64;
    static __wine_unix_call_dispatcher: unsafe extern "system" fn(u64, u32, *mut c_void) -> i32;
}

unsafe extern "system" {
    fn __wine_init_unix_call() -> i32;
    fn DisableThreadLibraryCalls(lib_module: *mut c_void) -> i32;
}

#[unsafe(export_name = "DllMain")]
pub extern "system" fn dll_main(instance: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    if reason != 1 {
        return 1;
    }

    mtld3d_shared::init_logger();
    attach_process(instance)
}

#[unsafe(no_mangle)]
pub extern "C" fn mtld3d_unix_call(code: u32, args: *mut c_void) -> i32 {
    dispatch_unix_call(code, args)
}

/// `DLL_PROCESS_ATTACH` body.
///
/// Kept as a private helper so the exported `dll_main` stub stays safe —
/// `not_unsafe_ptr_arg_deref` only checks `pub` functions, so the
/// pointer-passing unsafe work lives here.
fn attach_process(instance: *mut c_void) -> i32 {
    // SAFETY: `instance` is the HINSTANCE the loader passed to DllMain; Win32 accepts it as-is.
    unsafe { DisableThreadLibraryCalls(instance) };

    // SAFETY: Wine-published thunk; init exactly once on PROCESS_ATTACH per the unix-call ABI.
    let status = unsafe { __wine_init_unix_call() };
    if status != 0 {
        error!(target: LOG_TARGET, "__wine_init_unix_call failed");
        return 0;
    }

    log_identity(instance);
    1
}

/// Name this build in the log, alongside the unix-call handshake it completes.
///
/// [`identity::BUILD`] says which release the source came from; the image ID is
/// the PDB GUID the linker assigned, which names this exact binary and picks
/// the `.pdb` that symbolicates it out of the release's debug archive.
fn log_identity(instance: *mut c_void) {
    // SAFETY: `instance` is the HINSTANCE the loader passed to DllMain, so this
    // image is mapped in full.
    let id = unsafe { identity::image_id(instance) };
    let id = id.as_deref().unwrap_or("no-image-id");
    let build = identity::BUILD;
    info!(target: LOG_TARGET, "mtld3d.dll {build} {id}, unix call initialized");
}

/// Forwards to Wine's unix-call dispatcher.
///
/// Private helper for the same reason as `attach_process` — keeps the
/// exported `mtld3d_unix_call` free of raw-pointer unsafe work that clippy
/// would otherwise flag.
fn dispatch_unix_call(code: u32, args: *mut c_void) -> i32 {
    // SAFETY: Wine-published dispatcher fn-pointer + static unixlib handle; `args` is opaque to us.
    unsafe { (__wine_unix_call_dispatcher)(__wine_unixlib_handle, code, args) }
}
