//! The [`Mtld3dConfig`] an `IDirect3D9` resolves at `Direct3DCreate9`.
//!
//! Three lookups feed it. The built-in [`AppProfile`] for this application, if
//! it has one, comes from the executable's name plus the version resource of the
//! image the loader mapped. The optional `mtld3d.conf` is
//! `std::env::current_exe()` -> strip basename -> join `mtld3d.conf`; a missing
//! file is fine. The `MTLD3D_CONFIG` env var, when set, beats both. See
//! `mtld3d.conf` at the repo root for the user-facing sample with documented
//! keys and defaults.
//!
//! Resolved at the top of `Direct3DCreate9`, once per interface, so option
//! resolution and the per-key info log fire before the interface answers
//! anything; the interface owns the result and hands it to each device it
//! creates.

use std::{ffi::c_void, path::PathBuf, ptr};

use log::info;
use mtld3d_core::{
    app_profile::{AppIdentity, AppProfile},
    config::{Mtld3dConfig, log_options, parse},
};
use mtld3d_shared::identity::version_blob;

use crate::LOG_TARGET;

unsafe extern "system" {
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
}

/// Resolve the configuration for a new `IDirect3D9`.
///
/// Reads the profile, the file and the environment afresh each time, so a
/// process that creates several interfaces gives each the options in force
/// when it was created.
pub fn load() -> Mtld3dConfig {
    let env_override = std::env::var("MTLD3D_CONFIG")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let profile = app_profile();
    let file_src = read_conf_file();
    if let Some(env) = &env_override {
        info!(target: LOG_TARGET, "mtld3d.conf: applying MTLD3D_CONFIG overrides: {env}");
    }
    let cfg = parse(
        profile,
        file_src.as_deref().unwrap_or(""),
        env_override.as_deref(),
    );
    log_options(&cfg);
    cfg
}

/// The built-in profile for the executable this library was loaded into.
fn app_profile() -> Option<&'static AppProfile> {
    let exe = std::env::current_exe().ok()?;
    let name = exe.file_name()?.to_string_lossy().into_owned();
    // A null module name asks for the main image, which is the executable the
    // profile matches on rather than this DLL.
    // SAFETY: the call takes a null name and returns the main image's handle.
    let main = unsafe { GetModuleHandleA(ptr::null()) };
    // SAFETY: `main` is a handle the loader owns, so the image behind it is
    // mapped for the life of the process.
    let blob = unsafe { version_blob(main) };
    mtld3d_core::app_profile::lookup(&AppIdentity::new(name, blob.as_deref()))
}

fn read_conf_file() -> Option<String> {
    let Some(path) = conf_path() else {
        info!(
            target: LOG_TARGET,
            "mtld3d.conf: current_exe() unavailable — using defaults"
        );
        return None;
    };
    match std::fs::read_to_string(&path) {
        Ok(src) => {
            info!(target: LOG_TARGET, "mtld3d.conf: loaded from {}", path.display());
            Some(src)
        }
        Err(e) => {
            info!(
                target: LOG_TARGET,
                "mtld3d.conf: not loaded from {} ({e}) — using defaults",
                path.display()
            );
            None
        }
    }
}

fn conf_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(parent.join("mtld3d.conf"))
}
