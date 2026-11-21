//! Where the process's log file goes, named to the unix side once per interface.

use std::path::Path;

use log::warn;
use mtld3d_shared::OpenLogParams;

use crate::{LOG_TARGET, unix_call::unix_call};

/// The log directory beside the executable when `log.dir` is empty.
const DEFAULT_DIR: &str = "mtld3d-logs";

/// Name the process's log location to the unix side.
///
/// The directory is `log.dir` resolved against the executable's directory
/// (an absolute value stands as is), or `mtld3d-logs` beside the executable
/// when the key is empty; it is created here, the file itself by the unix
/// side with the first line it writes. When the directory cannot be created
/// or has no unix path, the thunk still goes out, empty, so the unix side
/// releases its backlog to stderr instead of holding it forever. The same
/// thunk carries `debug.mainThreadChecker`, the one resolved option the unix
/// side acts on once per process.
pub fn open(cfg: &mtld3d_core::config::Mtld3dConfig) {
    let location = log_location(cfg);
    let (unix_dir, stem) = match &location {
        Ok(pair) => (pair.0.as_str(), pair.1.as_str()),
        Err(why) => {
            warn!(target: LOG_TARGET, "log file: {why}, logging to stderr");
            ("", "")
        }
    };
    let (Ok(dir_len), Ok(stem_len)) = (u32::try_from(unix_dir.len()), u32::try_from(stem.len()))
    else {
        return;
    };
    let mut params = OpenLogParams {
        dir_ptr: unix_dir.as_ptr() as usize as u64,
        stem_ptr: stem.as_ptr() as usize as u64,
        dir_len,
        stem_len,
        main_thread_checker: u32::from(cfg.main_thread_checker),
        pad0: 0,
    };
    unix_call(&mut params);
}

/// The default log location as a unix path, plus the executable's stem, or `None`.
///
/// Derived at load, before `mtld3d.conf` is read, so `log.dir` plays no
/// part: this is where a crash report goes when it comes before
/// `Direct3DCreate9` names the location. Nothing is created here; the unix
/// side creates the directory only if a crash report needs it. The
/// executable's directory is mapped rather than the log directory itself,
/// since a directory that does not exist yet may have no unix path.
pub fn early_location() -> Option<(String, String)> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let unix_exe_dir = crate::platform::unix_path(exe_dir)?;
    let stem = exe.file_stem().map_or_else(
        || String::from("mtld3d"),
        |s| s.to_string_lossy().into_owned(),
    );
    let separator = if unix_exe_dir.ends_with('/') { "" } else { "/" };
    Some((format!("{unix_exe_dir}{separator}{DEFAULT_DIR}"), stem))
}

/// The log directory as a unix path plus the executable's stem, or why not.
fn log_location(cfg: &mtld3d_core::config::Mtld3dConfig) -> Result<(String, String), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe() unavailable ({e})"))?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let dir = if cfg.log_dir.is_empty() {
        exe_dir.join(DEFAULT_DIR)
    } else {
        exe_dir.join(&cfg.log_dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {} ({e})", dir.display()))?;
    let stem = exe.file_stem().map_or_else(
        || String::from("mtld3d"),
        |s| s.to_string_lossy().into_owned(),
    );
    let unix_dir = crate::platform::unix_path(&dir)
        .ok_or_else(|| format!("no unix path for {}", dir.display()))?;
    Ok((unix_dir, stem))
}
