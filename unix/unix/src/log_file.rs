//! The process's log file, and the directory its GPU traces share.
//!
//! Every line of both sides ends here: the unix `env_logger` target
//! ([`FileSink`]), the `WriteLog` handler that carries d3d9.dll's lines, and
//! the crash paths ([`write_raw`], [`write_bytes`], on the raw descriptor
//! because a signal handler may not take the mutex). The PE side names the
//! directory through the `OpenLog` thunk once `mtld3d.conf` is resolved at
//! `Direct3DCreate9`; the unix logger has been running since `DllMain`'s
//! `InitLogger` by then, so the lines of that gap wait in a backlog and go
//! out first. The d3d9.dll side needs no backlog of its own: its log thread
//! starts after `OpenLog`, so its queue still holds every line from `DllMain`.
//!
//! The file is created on the first line written after the location is
//! known, not when it is named: a process that logs nothing (`RUST_LOG=off`,
//! the conformance runner) leaves no empty file behind. A location that
//! cannot be created or opened falls back to stderr with one line saying so.

use core::ffi::c_void;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicI32, AtomicU32, Ordering},
    },
};

/// Bytes the backlog keeps while the location is pending.
///
/// The gap between `InitLogger` and `OpenLog` holds a handful of lines; the
/// cap only matters for a process that loads d3d9.dll and never creates a
/// device, where the backlog would otherwise grow for the process lifetime.
const BACKLOG_CAP: usize = 1024 * 1024;

/// Where a line goes.
enum Sink {
    /// Location not known yet: keep the line for the file that will come.
    ///
    /// `truncated` records that the cap dropped lines, so the file can say
    /// its start is incomplete.
    Pending { backlog: Vec<u8>, truncated: bool },
    /// Location known, file not created yet: the first line creates it.
    Lazy {
        path: PathBuf,
        backlog: Vec<u8>,
        truncated: bool,
    },
    /// The open log file.
    Open(File),
    /// The location failed; stderr is the fallback.
    Stderr,
}

static SINK: Mutex<Sink> = Mutex::new(Sink::Pending {
    backlog: Vec::new(),
    truncated: false,
});

/// The open file's descriptor for the lock-free crash paths; `-1` when closed.
static LOG_FD: AtomicI32 = AtomicI32::new(-1);

/// Where GPU traces go: the log directory plus the process prefix.
static TRACE_BASE: Mutex<Option<(PathBuf, String, u32)>> = Mutex::new(None);

/// Traces written by this process so far, for numbering the next one.
static TRACE_COUNT: AtomicU32 = AtomicU32::new(0);

/// Name the log location: `<dir>/<stem>-<pid>.log`, created on the first line.
///
/// Returns the path the file will have. Nothing is created here; a location
/// that turns out unusable is reported by the first write, which then falls
/// back to stderr.
pub fn open(dir: &str, stem: &str, pid: u32) -> PathBuf {
    let dir = PathBuf::from(dir);
    let path = dir.join(mtld3d_shared::log_paths::log_file_name(stem, pid));
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    let (backlog, truncated) = match core::mem::replace(&mut *guard, Sink::Stderr) {
        Sink::Pending { backlog, truncated }
        | Sink::Lazy {
            backlog, truncated, ..
        } => (backlog, truncated),
        // A second `Direct3DCreate9` names the location again: the file
        // already open keeps everything, and stderr has no backlog.
        Sink::Open(file) => {
            *guard = Sink::Open(file);
            return path;
        }
        Sink::Stderr => (Vec::new(), false),
    };
    *guard = Sink::Lazy {
        path: path.clone(),
        backlog,
        truncated,
    };
    drop(guard);
    *TRACE_BASE.lock().expect("trace base mutex poisoned") = Some((dir, stem.to_owned(), pid));
    path
}

/// Give up on a file: the backlog and everything after it go to stderr.
pub fn fall_back_to_stderr() {
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    let spill = match core::mem::replace(&mut *guard, Sink::Stderr) {
        Sink::Pending { backlog, .. } | Sink::Lazy { backlog, .. } => Some(backlog),
        Sink::Open(file) => {
            *guard = Sink::Open(file);
            None
        }
        Sink::Stderr => None,
    };
    drop(guard);
    if let Some(backlog) = spill {
        let _ = std::io::stderr().lock().write_all(&backlog);
    }
}

/// Create the file at `path` and write the backlog, or explain why not.
fn create(path: &PathBuf, backlog: &[u8], truncated: bool) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if truncated {
        let _ =
            file.write_all(b"[mtld3d::unix] log file: lines before the location were dropped\n");
    }
    file.write_all(backlog)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(file)
}

/// Write one line: into the backlog, the file, or stderr, per the state.
///
/// The file is unbuffered, so one call is one `write(2)` and nothing waits
/// in userspace when the process dies; a failed write falls back to stderr.
pub fn write_all(bytes: &[u8]) {
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    // Whatever stderr gets is written after the lock is released.
    let spill: Option<Vec<u8>> = match &mut *guard {
        Sink::Pending { backlog, truncated } => {
            if backlog.len() + bytes.len() <= BACKLOG_CAP {
                backlog.extend_from_slice(bytes);
            } else {
                *truncated = true;
            }
            None
        }
        Sink::Lazy {
            path,
            backlog,
            truncated,
        } => match create(path, backlog, *truncated) {
            Ok(mut file) => {
                let ok = file.write_all(bytes).is_ok();
                LOG_FD.store(file.as_raw_fd(), Ordering::Release);
                *guard = if ok { Sink::Open(file) } else { Sink::Stderr };
                None
            }
            Err(e) => {
                let mut out = format!("[mtld3d::unix] log file: {e}; logging to stderr instead\n")
                    .into_bytes();
                out.extend_from_slice(backlog);
                out.extend_from_slice(bytes);
                *guard = Sink::Stderr;
                Some(out)
            }
        },
        Sink::Open(file) => {
            if file.write_all(bytes).is_ok() {
                None
            } else {
                LOG_FD.store(-1, Ordering::Release);
                *guard = Sink::Stderr;
                Some(bytes.to_vec())
            }
        }
        Sink::Stderr => Some(bytes.to_vec()),
    };
    drop(guard);
    if let Some(out) = spill {
        let _ = std::io::stderr().lock().write_all(&out);
    }
}

/// The descriptor a signal handler writes: the log file's, or stderr's.
#[must_use]
pub fn raw_fd() -> i32 {
    let fd = LOG_FD.load(Ordering::Acquire);
    if fd < 0 { 2 } else { fd }
}

/// Append `len` bytes at `ptr` from a signal handler.
///
/// Reads the descriptor without the mutex, so it is async-signal-safe like
/// the `write(2)` on stderr it replaces; a line written before the file
/// exists goes to stderr, since creating the file is not signal-safe.
///
/// # Safety
///
/// `ptr` must point at `len` readable bytes.
pub unsafe fn write_raw(ptr: *const c_void, len: usize) {
    // SAFETY: write(2) is async-signal-safe; the caller vouches for `ptr`/`len`.
    unsafe {
        let _ = libc::write(raw_fd(), ptr, len);
    }
}

/// [`write_raw`] over a slice, the shape the crumb dump sink takes.
pub fn write_bytes(bytes: &[u8]) {
    // SAFETY: the slice is `len` readable bytes at `ptr` for the call.
    unsafe { write_raw(bytes.as_ptr().cast::<c_void>(), bytes.len()) };
}

/// The path of the next GPU trace, next to the log: `<dir>/<stem>-<pid>-<n>.gputrace`.
///
/// `None` before the location is known, when the caller keeps its own
/// default. The directory is created here because the capture writes into
/// it directly.
pub fn next_trace_path() -> Option<PathBuf> {
    let (dir, stem, pid) = TRACE_BASE
        .lock()
        .expect("trace base mutex poisoned")
        .as_ref()
        .cloned()?;
    let index = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let path = dir.join(mtld3d_shared::log_paths::trace_file_name(&stem, pid, index));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!(
            target: crate::LOG_TARGET,
            "cannot create the log directory {} for a GPU trace: {e}",
            dir.display()
        );
        return None;
    }
    Some(path)
}

/// The unix side's `env_logger` target.
pub struct FileSink;

impl Write for FileSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        write_all(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
