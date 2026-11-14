//! The Win32 side of the platform: what `d3d9.dll` needs from Windows and from Wine.
//!
//! It holds `DllMain` and the image's life (the pin, the exit hooks, the
//! crash handler), the PE logger, the exports with their hook prologue, and
//! every user32, gdi32, kernel32 and advapi32 import. The items re-exported at
//! the bottom are the interface the shared code calls; the native platform
//! defines the same ones.
//!
//! Each child module is named by a `path` attribute: this file is reached
//! through one itself, so a plain `mod` would look for its children beside it
//! rather than under `win32/`.

use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, Ordering},
};

use mtld3d_shared::{InitLoggerParams, identity};

use self::crash::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExA,
};
// The modules below name the crate-root items through `super::`, as they did
// when they sat at the crate root.
use crate::{
    D3D_OK, E_NOINTERFACE, LOG_TARGET, create9, create9_ex, null_out,
    unix_call::{self, unix_call},
};

#[macro_use]
#[path = "win32/hookable.rs"]
mod hookable;

#[path = "win32/adapter.rs"]
mod adapter;
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
#[path = "win32/arm64_crt.rs"]
mod arm64_crt;
#[path = "win32/capture.rs"]
mod capture;
#[path = "win32/crash.rs"]
mod crash;
#[cfg(mtld3d_crumb)]
#[path = "win32/crumb_allocator.rs"]
mod crumb_allocator;
#[path = "win32/cursor.rs"]
pub mod cursor;
#[path = "win32/exit_code_hook.rs"]
mod exit_code_hook;
#[path = "win32/fpu.rs"]
mod fpu;
#[path = "win32/fullscreen.rs"]
pub mod fullscreen;
#[cfg(target_arch = "x86_64")]
#[path = "win32/guest_mem.rs"]
mod guest_mem;
#[path = "win32/import_patch.rs"]
mod import_patch;
#[path = "win32/log_sink.rs"]
mod log_sink;
#[path = "win32/mode_list_hook.rs"]
mod mode_list_hook;
#[path = "win32/shader_validator.rs"]
mod shader_validator;
#[path = "win32/surface_dc.rs"]
mod surface_dc;
#[path = "win32/window.rs"]
mod window;
#[path = "win32/wine_path.rs"]
mod wine_path;

/// Whether this build's process can run out of address space: a 32-bit one can.
pub const ADDRESS_SPACE_WALKS: bool = cfg!(target_pointer_width = "32");

const DLL_PROCESS_ATTACH: u32 = 1;
const DLL_PROCESS_DETACH: u32 = 0;
/// The reason a thread's exit gives a TLS callback, which only the ARM64X halves install.
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
const DLL_THREAD_DETACH: u32 = 3;

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
static ALLOCATOR: crumb_allocator::CrumbAllocator = crumb_allocator::CrumbAllocator;

/// Set to true when the game calls `IDirect3D9::CreateDevice`.
///
/// Latched on entry to the call, before argument validation, so a rejected
/// `CreateDevice` still counts as the game having reached for a device. The
/// first latch also pins this image ([`pin_image`]), so from then on no
/// `FreeLibrary` unloads it and the only `DLL_PROCESS_DETACH` still to come is
/// the one at process exit.
///
/// `DllMain`'s detach handler reads it to choose between ending the process
/// and unhooking for an unload. Wine's loader passes `lpvReserved` as 0 when
/// the detach comes from `FreeLibrary` and as 1 when it comes from process
/// exit, the contract Windows documents, but the pin already makes every
/// detach after a device a process exit, so the flag alone decides.
/// `Direct3DCreate9` is too early a signal: launcher and mod DLLs commonly
/// probe-call it to verify the export resolves, then `FreeLibrary`, and that
/// unload has to stay real. `CreateDevice` requires a real HWND and
/// presentation parameters, so reaching it implies actual use.
pub static USED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" {
    fn DisableThreadLibraryCalls(lib_module: *mut c_void) -> i32;
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
}

#[link(name = "mtld3d", kind = "raw-dylib")]
unsafe extern "C" {
    fn mtld3d_unix_call(code: u32, args: *mut c_void) -> i32;
}

pub use adapter::adapter_luid;
pub use capture::capture_hotkey_pressed;
pub use crash::{address_space_map, free_space};
pub use fpu::apply_device_control_word;
pub use fullscreen::{
    current_display_mode, enumerate_display_modes, is_maximized, primary_monitor, system_dpi,
    warn_if_dpi_scaled,
};
pub use log_sink::{
    acquire as acquire_logger, caller_holds_loader_lock, release as release_logger,
};
pub use surface_dc::{SurfaceDc, SurfaceDcSource, create_surface_dc, destroy_surface_dc};
pub use window::client_size;
pub use wine_path::unix_path;

/// Run the unix side's handler for `code` through `mtld3d.dll`'s dispatcher.
///
/// # Safety
///
/// `args` points at a live params struct of the thunk `code` names; the
/// handler casts it back to that type.
pub unsafe fn dispatch_unix_call(code: u32, args: *mut c_void) -> i32 {
    // SAFETY: the caller's contract is the dispatcher's.
    unsafe { mtld3d_unix_call(code, args) }
}

/// Note that the game reached for a device, pinning this image the first time.
///
/// Called on entry to `CreateDevice`, before argument validation; see
/// [`USED`] for why a rejected create counts.
pub fn note_create_device() {
    if !USED.swap(true, Ordering::Relaxed) {
        pin_image();
    }
}

/// The version resource of the main image, the executable an app profile matches on.
pub fn main_image_version_blob() -> Option<Vec<u16>> {
    // A null module name asks for the main image, which is the executable the
    // profile matches on rather than this DLL.
    // SAFETY: the call takes a null name and returns the main image's handle.
    let main = unsafe { GetModuleHandleA(core::ptr::null()) };
    // SAFETY: `main` is a handle the loader owns, so the image behind it is
    // mapped for the life of the process.
    unsafe { identity::version_blob(main) }
}

#[unsafe(export_name = "DllMain")]
pub extern "system" fn dll_main(instance: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    if reason == DLL_PROCESS_DETACH && USED.load(Ordering::Relaxed) {
        // The process is exiting. A device was created, which pinned the
        // image, so no `FreeLibrary` reaches this detach unless the pin
        // failed, which was logged; only process exit does (Wine passes
        // `_reserved` as 1 here and as 0 for the detach of a
        // `FreeLibrary`). Skip every remaining destructor on the calling
        // thread: snmalloc's C++ thread_local teardown walks pools deeply
        // enough to overflow the 1 MB Wine main-thread stack and Wine then
        // aborts exception dispatch, hanging the process. TerminateProcess
        // is the only call that skips DLL_PROCESS_DETACH and TLS callbacks
        // while naming an exit code; ExitProcess / std::process::exit run
        // them, abort uses fast-fail. Its code is the one the unix side of
        // Wine exits with, so it carries the status the process asked to
        // exit with rather than a zero that would hide a failing run from a
        // unix parent.
        let status = exit_code_hook::status();
        // SAFETY: Win32 GetCurrentProcess returns a pseudo-handle for the
        // current process; passing it to TerminateProcess is the documented
        // self-exit form.
        let proc = unsafe { GetCurrentProcess() };
        // SAFETY: pseudo-handle to current process, with the status the
        // process asked to exit with.
        unsafe { TerminateProcess(proc, status) };
    }
    if reason == DLL_PROCESS_DETACH {
        // A FreeLibrary before any device, which the process survives (or
        // process exit before any device): take the process-wide pointers
        // into this image down with it.
        crash::uninstall();
        mode_list_hook::uninstall();
        exit_code_hook::uninstall();
    }
    if reason != DLL_PROCESS_ATTACH {
        return 1;
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    arm64_crt::attach();
    init_logger(instance);
    attach_process(instance);
    1
}

/// Keep this image loaded until the process ends, called once at the first `CreateDevice`.
///
/// A pinned module ignores `FreeLibrary`, so a launcher that creates a
/// device, releases it and frees `d3d9.dll` keeps running with the image
/// mapped, and the next `LoadLibrary` finds it already there. Unloading for
/// real after a device existed would run the allocator's thread-local
/// teardown on the caller's thread and leave process-wide registrations
/// (window subclasses, notification observers, Metal completion handlers)
/// pointing into unmapped code. The pin waits for a device rather than
/// happening at load, because a load that only probes `Direct3DCreate9` and
/// frees the library again has to unload for real.
pub fn pin_image() {
    let mut module: *mut c_void = core::ptr::null_mut();
    // SAFETY: kernel32 export; `USED` is a static of this image, so its
    // address names the module, and `out` is a writable local.
    let ok = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            (&raw const USED).cast::<u8>(),
            &raw mut module,
        )
    };
    if ok == 0 || module.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "CreateDevice: GetModuleHandleEx could not pin d3d9.dll; a later FreeLibrary that unloads it ends the process"
        );
    }
}

// The exports of the d3d9 API. Each is an entry with a fixed prologue that hook
// engines relocate (`platform/win32/hookable.rs`), and jumps to the function after `=`.

hookable_export! {
    /// `Direct3DCreate9`: an `IDirect3D9`.
    #[must_use]
    "Direct3DCreate9" => fn direct3d_create9(sdk_version: u32) -> *mut c_void = create9;
}

hookable_export! {
    /// `Direct3DCreate9Ex`: an `IDirect3D9Ex`, the factory of extended devices.
    ///
    /// The interface is the one `Direct3DCreate9` makes with its extended flag
    /// set: it answers `IID_IDirect3D9Ex`, and every device it creates, through
    /// `CreateDeviceEx` or plain `CreateDevice`, is an extended device. A null
    /// out slot is `D3DERR_INVALIDCALL` and creates nothing.
    "Direct3DCreate9Ex" => fn direct3d_create9_ex(sdk_version: u32, out: *mut *mut c_void) -> i32 = create9_ex;
}

hookable_export! {
    /// `Direct3DShaderValidatorCreate9`: the shader validator interface.
    #[must_use]
    "Direct3DShaderValidatorCreate9" => fn direct3d_shader_validator_create9() -> *mut c_void = shader_validator_create9;
}

// The `D3DPERF_*` family: PIX event markers a game emits around its draw
// groups. Without a profiler attached the real d3d9.dll does nothing and
// reports no nesting, no repeat-frame request and no attached tool, which is
// the complete behaviour here too. They are exported because engines resolve
// the whole family by name in one table and treat a missing entry as a broken
// d3d9.dll: an in-game overlay SDK refuses to initialise when any of them
// resolves to null, even though the game itself renders fine.

hookable_export! {
    /// `D3DPERF_BeginEvent`: opens a PIX event; returns the nesting level.
    ///
    /// Logged once so a game that emits PIX markers is visible in triage; the
    /// markers are not forwarded to a Metal capture.
    "D3DPERF_BeginEvent" => fn d3dperf_begin_event(color: u32, name: *const u16) -> i32 = perf_begin_event;
}

hookable_export! {
    /// `D3DPERF_EndEvent`: closes a PIX event; returns the nesting level.
    #[must_use]
    "D3DPERF_EndEvent" => const fn d3dperf_end_event() -> i32 = perf_end_event;
}

hookable_export! {
    /// `D3DPERF_SetMarker`: a single PIX marker, not forwarded.
    "D3DPERF_SetMarker" => const fn d3dperf_set_marker(color: u32, name: *const u16) = perf_set_marker;
}

hookable_export! {
    /// `D3DPERF_SetRegion`: a PIX region marker, not forwarded.
    "D3DPERF_SetRegion" => const fn d3dperf_set_region(color: u32, name: *const u16) = perf_set_region;
}

hookable_export! {
    /// `D3DPERF_QueryRepeatFrame`: `FALSE`, no profiler asks for a frame replay.
    #[must_use]
    "D3DPERF_QueryRepeatFrame" => const fn d3dperf_query_repeat_frame() -> i32 = perf_query_repeat_frame;
}

hookable_export! {
    /// `D3DPERF_SetOptions`: profiler permission flags, nothing to apply them to.
    "D3DPERF_SetOptions" => const fn d3dperf_set_options(options: u32) = perf_set_options;
}

hookable_export! {
    /// `D3DPERF_GetStatus`: `0`, no profiler attached.
    #[must_use]
    "D3DPERF_GetStatus" => const fn d3dperf_get_status() -> u32 = perf_get_status;
}

// The rest of the names Windows' d3d9.dll exports, apart from the 9On12
// factories. Callers resolve them by name: `d3dx9_43.dll` looks up
// `DebugSetMute` and calls it when found, and a proxy `d3d9.dll` in a game
// folder exports every name and forwards each through the address it got from
// this image with no null check, so a missing export becomes a jump to address
// 0 the moment anything calls the proxy's copy. Each keeps the Windows
// signature, so on i386 it removes as many argument bytes as its callers
// expect. None has anything to do here: there is no debug runtime and no
// processor-specific vertex pipeline.

hookable_export! {
    /// `DebugSetMute`: silences the debug runtime's output; there is none here.
    ///
    /// Takes no arguments, so on i386 it returns with a plain `ret`: `d3dx9`
    /// pushes one argument and pops it itself after the call.
    "DebugSetMute" => fn debug_set_mute_export() = debug_set_mute;
}

hookable_export! {
    /// `DebugSetLevel`: the debug runtime's output level; `0`, there is no debug runtime.
    #[must_use]
    "DebugSetLevel" => fn debug_set_level_export() -> i32 = debug_set_level;
}

hookable_export! {
    /// `Direct3D9EnableMaximizedWindowedModeShim`: an application-compatibility switch; `0`.
    #[must_use]
    "Direct3D9EnableMaximizedWindowedModeShim" => fn direct3d9_enable_maximized_windowed_mode_shim(enable: u32) -> i32 = enable_maximized_windowed_mode_shim;
}

hookable_export! {
    /// `Direct3D9ForceHybridEnumeration`: the unnamed export at ordinal 16; does nothing.
    ///
    /// Windows exports it by ordinal only, and `d3d9.def` marks it `NONAME`
    /// likewise. Its name says it forces the enumeration of hybrid-graphics
    /// adapters; there is one adapter here, so there is nothing to switch.
    "Direct3D9ForceHybridEnumeration" => fn direct3d9_force_hybrid_enumeration(hybrid: u32) = force_hybrid_enumeration;
}

hookable_export! {
    /// `PSGPError`: an error report from a processor-specific vertex pipeline; ignored.
    "PSGPError" => fn psgp_error_export(vertices: *mut c_void, error: u32, value: u32) = psgp_error;
}

hookable_export! {
    /// `PSGPSampleTexture`: a texture fetch for a processor-specific vertex pipeline; does nothing.
    "PSGPSampleTexture" => fn psgp_sample_texture_export(
        vertices: *mut c_void,
        stage: u32,
        coords: *mut c_void,
        count: u32,
        result: *mut c_void,
    ) = psgp_sample_texture;
}

extern "system" fn shader_validator_create9() -> *mut c_void {
    shader_validator::create()
}

extern "system" fn perf_begin_event(_color: u32, _name: *const u16) -> i32 {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "D3DPERF_BeginEvent: PIX event markers not forwarded (no profiler)"
    );
    0
}

const extern "system" fn perf_end_event() -> i32 {
    0
}

const extern "system" fn perf_set_marker(_color: u32, _name: *const u16) {}

const extern "system" fn perf_set_region(_color: u32, _name: *const u16) {}

const extern "system" fn perf_query_repeat_frame() -> i32 {
    0
}

const extern "system" fn perf_set_options(_options: u32) {}

const extern "system" fn perf_get_status() -> u32 {
    0
}

extern "system" fn debug_set_mute() {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "DebugSetMute: no debug runtime to mute, no-op"
    );
}

extern "system" fn debug_set_level() -> i32 {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "DebugSetLevel: no debug runtime → 0"
    );
    0
}

extern "system" fn enable_maximized_windowed_mode_shim(_enable: u32) -> i32 {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "Direct3D9EnableMaximizedWindowedModeShim: no shim to switch → 0"
    );
    0
}

extern "system" fn force_hybrid_enumeration(_hybrid: u32) {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "Direct3D9ForceHybridEnumeration: one adapter, nothing to switch, no-op"
    );
}

extern "system" fn psgp_error(_vertices: *mut c_void, _error: u32, _value: u32) {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub PSGPError: no processor-specific vertex pipeline, ignored"
    );
}

extern "system" fn psgp_sample_texture(
    _vertices: *mut c_void,
    _stage: u32,
    _coords: *mut c_void,
    _count: u32,
    _result: *mut c_void,
) {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub PSGPSampleTexture: no processor-specific vertex pipeline, nothing sampled"
    );
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
    mtld3d_shared::crumb::set_write_sink(log_sink::write_fault);
    let filter = std::env::var("RUST_LOG").unwrap_or_default();
    // Where a crash report goes if one comes before `Direct3DCreate9`
    // names the log location; empty when it cannot be derived.
    let (early_dir, early_stem) = crate::log_location::early_location().unwrap_or_else(|| {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "log file: no default location at load, a crash before Direct3DCreate9 reports to stderr"
        );
        (String::new(), String::new())
    });
    let mut params = InitLoggerParams {
        filter_ptr: filter.as_ptr() as u64,
        early_dir_ptr: early_dir.as_ptr() as u64,
        early_stem_ptr: early_stem.as_ptr() as u64,
        filter_len: u32::try_from(filter.len()).unwrap_or(0),
        early_dir_len: u32::try_from(early_dir.len()).unwrap_or(0),
        early_stem_len: u32::try_from(early_stem.len()).unwrap_or(0),
        pad0: 0,
    };
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
    let arch = identity::ARCH;
    let base = instance as usize;
    log::info!(target: LOG_TARGET, "d3d9.dll {build} {id} loaded at {base:#x} as {arch}");
    #[cfg(target_arch = "x86_64")]
    guest_mem::log_route();
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
    exit_code_hook::install();
}
