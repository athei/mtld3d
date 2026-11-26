//! The native side of the platform: `d3d9` linked with the unix side into one host library.
//!
//! The first `Direct3DCreate9` or `Direct3DCreate9Ex` does what `DllMain` does
//! under Wine, once. The host owns the process's signals, so no crash handler
//! is installed here. Where the native build does less, the items below answer
//! as the shared code's neutral case.
//!
//! This file is reached through a `path` attribute, so its children need one too.

use core::ffi::c_void;
use std::{path::Path, sync::Once};

use mtld3d_core::address_space::FreeSpace;
use mtld3d_shared::identity;

use crate::{LOG_TARGET, create9, create9_ex, surface::SurfaceDcSource};

#[path = "native/cursor.rs"]
pub mod cursor;
#[path = "native/fullscreen.rs"]
pub mod fullscreen;
#[path = "native/view.rs"]
mod view;

/// A native process is 64-bit and has no address space to run out of.
pub const ADDRESS_SPACE_WALKS: bool = false;

pub use fullscreen::{
    current_display_mode, enumerate_display_modes, is_maximized, primary_monitor,
    warn_if_dpi_scaled,
};
pub use view::{client_size, is_device_window};

/// `Direct3DCreate9`: an `IDirect3D9`, after the process's one-time setup.
#[must_use]
#[unsafe(export_name = "Direct3DCreate9")]
pub extern "system" fn direct3d_create9(sdk_version: u32) -> *mut c_void {
    init_once();
    create9(sdk_version)
}

/// `Direct3DCreate9Ex`: an `IDirect3D9Ex`, after the process's one-time setup.
#[unsafe(export_name = "Direct3DCreate9Ex")]
pub extern "system" fn direct3d_create9_ex(sdk_version: u32, out: *mut *mut c_void) -> i32 {
    init_once();
    create9_ex(sdk_version, out)
}

/// Run the unix side's handler for `code`, linked into the same library.
///
/// # Safety
///
/// `args` points at a live params struct of the thunk `code` names; the
/// handler casts it back to that type.
pub unsafe fn dispatch_unix_call(code: u32, args: *mut c_void) -> i32 {
    let handler =
        mtld3d_unix::DISPATCH_TABLE[usize::try_from(code).expect("a u32 thunk code fits usize")];
    // SAFETY: the caller's contract is the handler's.
    unsafe { handler(args) }
}

/// Nothing to pin: the host links the library and never unloads it under a device.
pub const fn note_create_device() {}

/// No PE version resource: a native host matches no app profile.
pub const fn main_image_version_blob() -> Option<Vec<u16>> {
    None
}

/// Host paths are already the unix side's, so a UTF-8 path is passed on as it is.
pub fn unix_path(path: &Path) -> Option<String> {
    path.to_str().map(str::to_owned)
}

/// The capture chord is not read in the native build, which is logged once.
pub fn capture_hotkey_pressed() -> bool {
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "capture: the hotkey chord is not read in the native build"
    );
    false
}

/// No display driver names the adapter: the LUID is zero.
pub const fn adapter_luid() -> (u32, i32) {
    (0, 0)
}

/// An arm64 thread has no x87 control word to set.
pub const fn apply_device_control_word(_behavior_flags: u32) {}

/// Exclusive fullscreen is not offered: `CreateDevice` and `Reset` refuse `Windowed = FALSE`.
pub const fn exclusive_fullscreen() -> bool {
    false
}

/// No log thread starts: the unix side's logger writes every line itself.
pub const fn acquire_logger() {}

/// No log thread to stop.
pub const fn release_logger() {}

/// A native build has no loader lock.
pub const fn caller_holds_loader_lock() -> bool {
    false
}

/// No regions: the native build does not walk its address space.
pub const fn free_space() -> FreeSpace {
    FreeSpace::new()
}

/// The address-space map a native build logs.
pub fn address_space_map() -> String {
    String::from("not walked in the native build")
}

/// The device context `GetDC` hands out; the native build never holds one.
#[derive(Clone, Copy)]
pub struct SurfaceDc {
    dc: *mut c_void,
}

impl SurfaceDc {
    /// No DC held, the only value the native build makes.
    pub const NULL: Self = Self {
        dc: core::ptr::null_mut(),
    };

    /// The `HDC` `GetDC` hands the application.
    pub const fn handle(&self) -> *mut c_void {
        self.dc
    }
}

/// `GetDC` has no GDI to wrap the pixels in, so it is refused, which is logged once.
pub fn surface_dcs() -> bool {
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "IDirect3DSurface9::GetDC: the native build has no GDI device context → INVALIDCALL"
    );
    false
}

/// No DC; [`surface_dcs`] refuses `GetDC` before it gets here, so this only logs at debug.
///
/// # Safety
///
/// None beyond the Win32 implementation's contract: `source` is only logged.
pub unsafe fn create_surface_dc(source: &SurfaceDcSource) -> Option<SurfaceDc> {
    log::debug!(
        target: LOG_TARGET,
        "IDirect3DSurface9::GetDC ({}x{}, format {}, pitch {}, pixels at {:p}): no device \
         context in the native build",
        source.width,
        source.height,
        source.format,
        source.pitch,
        source.memory,
    );
    None
}

/// Nothing to destroy: no DC is ever created.
pub const fn destroy_surface_dc(_held: SurfaceDc) {}

/// Install the logger and name this build, once per process.
///
/// Both sides share one `log` crate in this build, so the logger the unix
/// side's `InitLogger` installs serves `d3d9` too.
fn init_once() {
    /// The one-time setup has run.
    ///
    /// The resource is process-wide: one logger and one set of trace gates.
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        crate::log_location::init_logger();
        mtld3d_core::perf::init_tracking_enabled();
        mtld3d_core::state_trace::init_enabled();
        let id = identity::image_id();
        let id = id.as_deref().unwrap_or("no-image-id");
        log::info!(
            target: LOG_TARGET,
            "d3d9 {} {id} linked into the host as {}",
            identity::BUILD,
            identity::ARCH,
        );
    });
}
