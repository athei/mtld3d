mod bound_buffers;
mod bound_rt;
mod capture;
mod com_ref;
mod config;
mod device;
mod direct3d9;
mod draw;
mod encoder;
mod index_buffer;
mod log_location;
mod page_box_pool;
mod pixel_shader;
mod platform;
mod private_data;
mod query;
mod shader_bindings;
mod stage_bindings;
mod state_block;
mod surface;
mod swapchain;
mod texture;
mod unix_call;
mod vertex_buffer;
mod vertex_decl;
mod vertex_shader;

use core::ffi::c_void;
use std::sync::Arc;

// HRESULT codes live in `mtld3d_types` (shared with the integration-test
// harness); re-exported under the crate root so every in-crate
// `use super::{D3D_OK, …}` path stays valid.
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DERR_NOTFOUND, E_FAIL, E_NOINTERFACE,
    S_FALSE,
};

use crate::direct3d9::Direct3D9;

/// Single source of truth for the `log` crate target, so callers don't hard-code the string.
///
/// Every `log!(target: LOG_TARGET, ...)` site in the crate uses it. Lives
/// under the project-wide `mtld3d::*` root so `RUST_LOG=mtld3d=...` flips the
/// whole project; `RUST_LOG=mtld3d::d3d9=...` scopes to the COM layer.
const LOG_TARGET: &str = "mtld3d::d3d9";

extern "system" fn create9(_sdk_version: u32) -> *mut c_void {
    create_interface(false).cast::<c_void>()
}

extern "system" fn create9_ex(_sdk_version: u32, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject Direct3DCreate9Ex: null out slot → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    write_out(out, create_interface(true).cast::<c_void>());
    log::info!(target: LOG_TARGET, "Direct3DCreate9Ex: extended interface created");
    D3D_OK
}

/// Create the interface both factory exports hand out, `extended` for `Direct3DCreate9Ex`.
fn create_interface(extended: bool) -> *mut Direct3D9 {
    // Resolves `mtld3d.conf` for this interface and logs the option set. The
    // log location it names reaches the unix side before the logging thread
    // starts, so every line queued since `DllMain` lands in the file; a later
    // interface names it again and the unix side keeps the file it has.
    let cfg = config::load();
    log_location::open(&cfg);
    // The page-box pool is one per process; the interface resolved most
    // recently sizes it.
    page_box_pool::PAGEBOX_POOL
        .set_cap(usize::try_from(cfg.pagebox_pool_cap_bytes).unwrap_or(usize::MAX));
    let interface = Box::into_raw(Box::new(Direct3D9::new(Arc::new(cfg), extended)));
    // The first entry point outside `DllMain`: the logging thread can start
    // here (DllMain runs under the loader lock and must not spawn threads).
    // It runs until the last interface is released, holding this image. Last,
    // so the lines the interface's creation logged are among those `acquire`
    // forwards before it returns.
    platform::acquire_logger();
    interface
}

/// Null a COM `**out` parameter before returning a failing HRESULT.
///
/// Callers that ignore the HRESULT and read the out-pointer get `null`
/// instead of stack garbage, which would otherwise read as a bogus COM
/// `this` pointer.
fn null_out(out: *mut *mut c_void) {
    write_out(out, core::ptr::null_mut());
}

/// Write a COM `**out` parameter, the success counterpart of [`null_out`].
///
/// A private helper so the exported entry point that calls it stays a safe
/// function; `out` is null or a writable slot per the COM ABI.
fn write_out(out: *mut *mut c_void, value: *mut c_void) {
    // SAFETY: per the COM ABI `out` is null, which `write_opt` skips, or a
    // writable `*mut c_void` slot owned by the caller.
    unsafe { mtld3d_shared::OutPtr::write_opt(out, value) };
}
