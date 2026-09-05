mod bound_buffers;
mod bound_rt;
mod capture;
mod com_ref;
mod config;
mod crash;
#[cfg(mtld3d_crumb)]
mod crumb_allocator;
mod cursor;
mod device;
mod direct3d9;
mod draw;
mod encoder;
mod fullscreen;
mod index_buffer;
mod log_sink;
mod mode_list_hook;
mod page_box_pool;
mod pixel_shader;
mod private_data;
mod query;
mod shader_bindings;
mod shader_prewarm;
mod shader_validator;
mod stage_bindings;
mod state_block;
mod surface;
mod swapchain;
mod texture;
mod unix_call;
mod vertex_buffer;
mod vertex_decl;
mod vertex_shader;

use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};
use std::sync::Arc;

use mtld3d_shared::{InitLoggerParams, identity};
// HRESULT codes live in `mtld3d_types` (shared with the integration-test
// harness); re-exported under the crate root so every in-crate
// `use super::{D3D_OK, …}` path stays valid.
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DERR_NOTFOUND, E_FAIL, E_NOINTERFACE,
    E_NOTIMPL,
};

use crate::{direct3d9::Direct3D9, unix_call::unix_call};

const DLL_PROCESS_ATTACH: u32 = 1;
const DLL_PROCESS_DETACH: u32 = 0;

/// Single source of truth for the `log` crate target, so callers don't hard-code the string.
///
/// Every `log!(target: LOG_TARGET, ...)` site in the crate uses it. Lives
/// under the project-wide `mtld3d::*` root so `RUST_LOG=mtld3d=...` flips the
/// whole project; `RUST_LOG=mtld3d::d3d9=...` scopes to the COM layer.
const LOG_TARGET: &str = "mtld3d::d3d9";

// The baseline this replaces is not a modern libc malloc. With no global
// allocator, i686-pc-windows-msvc routes every allocation to HeapAlloc,
// which under Wine is RtlAllocateHeap: one shared heap lock across the
// API and encoder threads, and a per-block virtual mapping for large
// blocks, paid from emulated x86 crossing the PE/unix boundary. snmalloc
// serves from per-thread slabs and never calls HeapAlloc at any size —
// it takes address space from VirtualAlloc directly.
//
// Its lock-free remote-free ring also batches cross-thread frees, though
// not for PageBox: the ring's budget is 16 KiB and a PageBox is at least
// one 16 KiB page, so every box posts on its own. The batching earns its
// keep on the smaller traffic crossing the same boundary (Arc control
// blocks, boxed closures, FrameData internals).
//
// mimalloc is not usable here: its cross-thread free path faults on
// 16 KiB-aligned PageBox allocations.
//
// Under `cfg(mtld3d_crumb)` the allocator is swapped for
// `crumb_allocator::CrumbAllocator` — a thin `SnMalloc` wrapper that
// records PageBox-shape alloc/dealloc events into the shared crash
// breadcrumb. Production builds get plain `SnMalloc` with zero overhead.
#[cfg(not(mtld3d_crumb))]
#[global_allocator]
static ALLOCATOR: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;
#[cfg(mtld3d_crumb)]
#[global_allocator]
static ALLOCATOR: crate::crumb_allocator::CrumbAllocator = crate::crumb_allocator::CrumbAllocator;

/// Set to true when the game calls `IDirect3D9::CreateDevice`.
///
/// Latched on entry to the call, before argument validation, so a rejected
/// `CreateDevice` still counts as the game having reached for a device.
///
/// Used by `DllMain`'s `DLL_PROCESS_DETACH` handler to discriminate real
/// shutdown from the loader's early `FreeLibrary` probe — Wine sets
/// `reserved=NULL` for both, so MSDN's "reserved != NULL means process exit"
/// contract is unusable. `Direct3DCreate9` is too early a signal: launcher/mod
/// DLLs commonly probe-call it to verify the export resolves, then
/// `FreeLibrary`. `CreateDevice` requires a real HWND + presentation params, so
/// reaching it implies actual use.
pub static USED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" {
    fn DisableThreadLibraryCalls(lib_module: *mut c_void) -> i32;
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
}

#[unsafe(export_name = "DllMain")]
pub extern "system" fn dll_main(instance: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    if reason == DLL_PROCESS_DETACH && USED.load(Ordering::Relaxed) {
        // Process is exiting (or the game FreeLibrary'd us after CreateDevice).
        // Skip every remaining destructor on the calling thread — snmalloc's
        // C++ thread_local teardown walks pools deeply enough to overflow
        // the 1 MB Wine main-thread stack and Wine then aborts exception
        // dispatch, hanging the process. TerminateProcess is the only call
        // that exits with code 0 while skipping DLL_PROCESS_DETACH and TLS
        // callbacks; ExitProcess / std::process::exit run them, abort uses
        // fast-fail. The USED flag discriminates against the loader's early
        // FreeLibrary probe — Wine's `_reserved` arg is NULL for both
        // the probe and real exit, so the MSDN contract is unusable here.
        // SAFETY: Win32 GetCurrentProcess returns a pseudo-handle for the
        // current process; passing it to TerminateProcess is the documented
        // self-exit form.
        let proc = unsafe { GetCurrentProcess() };
        // SAFETY: pseudo-handle to current process; exit code 0.
        unsafe { TerminateProcess(proc, 0) };
    }
    if reason == DLL_PROCESS_DETACH {
        // A FreeLibrary the process survives: take the process-wide pointers
        // into this image down with it.
        crash::uninstall();
        mode_list_hook::uninstall();
    }
    if reason != DLL_PROCESS_ATTACH {
        return 1;
    }
    init_logger(instance);
    attach_process(instance);
    1
}

#[unsafe(export_name = "Direct3DCreate9")]
#[must_use]
pub extern "system" fn direct3d_create9(_sdk_version: u32) -> *mut c_void {
    // Resolves `mtld3d.conf` for this interface and logs the option set. The
    // log location it names reaches the unix side before the logging thread
    // starts, so every line queued since `DllMain` lands in the file; a later
    // interface names it again and the unix side keeps the file it has.
    let cfg = config::load();
    log_sink::open(&cfg);
    // The first entry point outside `DllMain`: the logging thread can start
    // here (DllMain runs under the loader lock and must not spawn threads).
    log_sink::start();
    // The page-box pool is one per process; the interface resolved most
    // recently sizes it.
    page_box_pool::PAGEBOX_POOL
        .set_cap(usize::try_from(cfg.pagebox_pool_cap_bytes).unwrap_or(usize::MAX));
    Box::into_raw(Box::new(Direct3D9::new(Arc::new(cfg)))).cast::<c_void>()
}

#[unsafe(export_name = "Direct3DShaderValidatorCreate9")]
#[must_use]
pub extern "system" fn direct3d_shader_validator_create9() -> *mut c_void {
    shader_validator::create()
}

// The `D3DPERF_*` family: PIX event markers a game emits around its draw
// groups. Without a profiler attached the real d3d9.dll does nothing and
// reports no nesting, no repeat-frame request and no attached tool, which is
// the complete behaviour here too. They are exported because engines resolve
// the whole family by name in one table and treat a missing entry as a broken
// d3d9.dll: an in-game overlay SDK refuses to initialise when any of them
// resolves to null, even though the game itself renders fine.

/// `D3DPERF_BeginEvent`: opens a PIX event; returns the nesting level.
///
/// Logged once so a game that emits PIX markers is visible in triage; the
/// markers are not forwarded to a Metal capture.
#[unsafe(export_name = "D3DPERF_BeginEvent")]
pub extern "system" fn d3dperf_begin_event(_color: u32, _name: *const u16) -> i32 {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "D3DPERF_BeginEvent: PIX event markers not forwarded (no profiler)"
    );
    0
}

/// `D3DPERF_EndEvent`: closes a PIX event; returns the nesting level.
#[unsafe(export_name = "D3DPERF_EndEvent")]
#[must_use]
pub const extern "system" fn d3dperf_end_event() -> i32 {
    0
}

/// `D3DPERF_SetMarker`: a single PIX marker, not forwarded.
#[unsafe(export_name = "D3DPERF_SetMarker")]
pub const extern "system" fn d3dperf_set_marker(_color: u32, _name: *const u16) {}

/// `D3DPERF_SetRegion`: a PIX region marker, not forwarded.
#[unsafe(export_name = "D3DPERF_SetRegion")]
pub const extern "system" fn d3dperf_set_region(_color: u32, _name: *const u16) {}

/// `D3DPERF_QueryRepeatFrame`: `FALSE`, no profiler asks for a frame replay.
#[unsafe(export_name = "D3DPERF_QueryRepeatFrame")]
#[must_use]
pub const extern "system" fn d3dperf_query_repeat_frame() -> i32 {
    0
}

/// `D3DPERF_SetOptions`: profiler permission flags, nothing to apply them to.
#[unsafe(export_name = "D3DPERF_SetOptions")]
pub const extern "system" fn d3dperf_set_options(_options: u32) {}

/// `D3DPERF_GetStatus`: `0`, no profiler attached.
#[unsafe(export_name = "D3DPERF_GetStatus")]
#[must_use]
pub const extern "system" fn d3dperf_get_status() -> u32 {
    0
}

// Wires up the PE-side `env_logger` for this cdylib, then fires a
// one-shot `InitLogger` thunk so the unix .so registers its own
// (each cdylib has its own `log` crate statics). Runs from DllMain
// after mtld3d.dll's DllMain has already wired up the unix-call
// dispatcher — DLL load ordering is guaranteed by d3d9.dll's implicit
// import of `mtld3d_unix_call` from mtld3d.dll.
fn init_logger(instance: *mut c_void) {
    mtld3d_shared::init_logger_to(Box::new(log_sink::Sink));
    log_identity(instance);
    // Latch the d3d9-side perf-tracking gate (`PERF_TRACKING_ENABLED`)
    // from `RUST_LOG`. Per-cdylib because each cdylib has its own
    // `log` statics; the unix side latches its own copy in
    // `init_logger_handler`.
    mtld3d_core::perf::init_tracking_enabled();
    mtld3d_core::state_trace::init_enabled();
    // Map the shared crash crumb (cfg-gated no-op in production) and
    // install the always-on VEH-based crash handler. Both sides write
    // into the same `/tmp/mtld3d-crumb.bin` file so PE+unix events
    // interleave by seq.
    mtld3d_shared::crumb::init();
    crash::install(instance);
    mtld3d_shared::crumb::set_write_sink(log_sink::write_raw);
    let mut params = InitLoggerParams { reserved: 0 };
    unix_call(&mut params);
}

/// Name this build in the log, as the first line the logger emits.
///
/// [`identity::BUILD`] says which release the source came from; the image ID is
/// the PDB GUID the linker assigned, which names this exact binary and picks
/// the `.pdb` that symbolicates it out of the release's debug archive.
fn log_identity(instance: *mut c_void) {
    // SAFETY: `instance` is the HMODULE the loader passed to `DllMain` during
    // `DLL_PROCESS_ATTACH`, so this image is mapped in full.
    let id = unsafe { identity::image_id(instance) };
    let id = id.as_deref().unwrap_or("no-image-id");
    let build = identity::BUILD;
    let base = instance as usize;
    log::info!(target: LOG_TARGET, "d3d9.dll {build} {id} loaded at {base:#x}");
}

/// Null a COM `**out` parameter before returning a failing HRESULT.
///
/// Callers that ignore the HRESULT and read the out-pointer get `null`
/// instead of stack garbage, which would otherwise read as a bogus COM
/// `this` pointer.
fn null_out(out: *mut *mut c_void) {
    if !out.is_null() {
        // SAFETY: `out` is non-null and per the COM ABI points to a writable
        // `*mut c_void` slot owned by the caller.
        unsafe { *out = core::ptr::null_mut() };
    }
}

/// `DLL_PROCESS_ATTACH` body.
///
/// Private helper so the exported `dll_main` stub stays safe — clippy's
/// `not_unsafe_ptr_arg_deref` only checks `pub` functions.
/// `DisableThreadLibraryCalls` so per-thread `DllMain` notifications don't
/// fire.
fn attach_process(instance: *mut c_void) {
    // SAFETY: `instance` is the HMODULE passed by the loader to `DllMain`
    // during `DLL_PROCESS_ATTACH`; Win32 `DisableThreadLibraryCalls` is
    // safe to call from `DLL_PROCESS_ATTACH` with that module handle.
    unsafe { DisableThreadLibraryCalls(instance) };
    mode_list_hook::install();
}
