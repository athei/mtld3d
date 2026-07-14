//! Process-wide [`Mtld3dConfig`] resolved once on first access via [`LazyLock`].
//!
//! The lookup runs `std::env::current_exe()` → strip basename → join
//! `mtld3d.conf`; a missing file is fine and yields defaults. See
//! `mtld3d.conf` at the repo root for the user-facing sample with
//! documented keys and defaults.
//!
//! Touched once at the top of `Direct3DCreate9` so option resolution
//! and the per-key info log fire early in the process lifecycle.

use std::{path::PathBuf, sync::LazyLock};

use log::info;
use mtld3d_core::config::{Mtld3dConfig, log_options, parse};

use crate::LOG_TARGET;

/// Resolved config.
///
/// Read with `&*CONFIG` — `LazyLock` is preferred over `OnceLock` when
/// the initializer takes no captured args.
pub static CONFIG: LazyLock<Mtld3dConfig> = LazyLock::new(load);

fn load() -> Mtld3dConfig {
    let env_override = std::env::var("MTLD3D_CONFIG")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let file_src = read_conf_file();
    if env_override.is_some() {
        info!(target: LOG_TARGET, "mtld3d.conf: applying MTLD3D_CONFIG overrides");
    }
    let cfg = parse(file_src.as_deref().unwrap_or(""), env_override.as_deref());
    log_options(&cfg);
    cfg
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
