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
//!
//! A crash report cannot wait for `OpenLog`: the process may not live to
//! send it. `InitLogger` therefore names an early location, the default one
//! (`mtld3d-logs` beside the executable, before `mtld3d.conf` can move it),
//! and a crash report that arrives while the location is still pending opens
//! that file and writes the backlog ahead of itself ([`write_crash`] from an
//! ordinary thread, [`crash_fd`] from a signal handler). Without an early
//! location the backlog and the report go to stderr. The file a crash report
//! opened stays the process's log when `OpenLog` follows, wherever
//! `log.dir` points, so the lines before and after the report stay together.
//!
//! The directory keeps the [`KEEP`] newest logs and the [`KEEP`] newest
//! traces: creating a log or a trace first removes the oldest of its kind
//! past that count, so a game launched every day does not pile up files.

use core::ffi::c_void;
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io::Write,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    sync::{
        Mutex, MutexGuard, TryLockError,
        atomic::{AtomicI32, AtomicU32, Ordering},
    },
    time::SystemTime,
};

use mtld3d_core::crash_report::{CrashContext, CrashRoute, SinkState, crash_route};

/// Logs, and separately traces, the directory keeps; older ones go.
const KEEP: usize = 10;

/// Bytes the backlog keeps while the location is pending.
///
/// The gap between `InitLogger` and `OpenLog` holds a handful of lines; the
/// cap only matters for a process that loads d3d9.dll and never creates a
/// device, where the backlog would otherwise grow for the process lifetime.
const BACKLOG_CAP: usize = 1024 * 1024;

/// The first line of a file whose backlog lost lines to the cap.
const TRUNCATED_NOTE: &[u8] = b"[mtld3d::unix] log file: lines before the location were dropped\n";

/// Where a line goes.
enum Sink {
    /// Location not known yet: keep the line for the file that will come.
    ///
    /// `truncated` records that the cap dropped lines, so the file can say
    /// its start is incomplete; `early` is where a crash report goes meanwhile.
    Pending {
        backlog: Vec<u8>,
        truncated: bool,
        early: Option<EarlyLocation>,
    },
    /// Location known, file not created yet: the first line creates it.
    ///
    /// `early` says a crash report named it, ahead of `OpenLog`.
    Lazy {
        path: PathBuf,
        backlog: Vec<u8>,
        truncated: bool,
        early: bool,
    },
    /// The open log file.
    ///
    /// `early` is its path when a crash report opened it before `OpenLog`.
    Open { file: File, early: Option<PathBuf> },
    /// The location failed; stderr is the fallback.
    Stderr,
}

/// The default log location, named by `InitLogger` for a crash report that comes before `OpenLog`.
///
/// Built whole in advance: a signal handler creates the directory and the
/// file through `mkdir(2)` and `open(2)` on the two C strings and moves
/// `path` into the sink, so it allocates nothing and frees nothing.
struct EarlyLocation {
    dir: CString,
    file: CString,
    path: PathBuf,
}

static SINK: Mutex<Sink> = Mutex::new(Sink::Pending {
    backlog: Vec::new(),
    truncated: false,
    early: None,
});

/// The open file's descriptor for the lock-free crash paths; `-1` when closed.
static LOG_FD: AtomicI32 = AtomicI32::new(-1);

/// Where GPU traces go: the log directory plus the process prefix.
static TRACE_BASE: Mutex<Option<(PathBuf, String, u32)>> = Mutex::new(None);

/// Traces written by this process so far, for numbering the next one.
static TRACE_COUNT: AtomicU32 = AtomicU32::new(0);

/// Name the log location: `<dir>/<stem>-<pid>.log`, created on the first line.
///
/// `pid` is this process's host pid, the one `ps` shows. The PE side's own
/// pid would not do: it is Wine's process id, and a fresh wineserver hands
/// its first process the same one every launch, so every run of a game would
/// append to one file and the retention below would never see a second.
///
/// Returns the path the lines go to. Nothing is created here; a location
/// that turns out unusable is reported by the first write, which then falls
/// back to stderr. A file a crash report already opened at the early
/// location stays the log, and its path is the one returned.
pub fn open(dir: &str, stem: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = PathBuf::from(dir);
    let path = dir.join(mtld3d_shared::log_paths::log_file_name(stem, pid));
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    let (backlog, truncated) = match core::mem::replace(&mut *guard, Sink::Stderr) {
        Sink::Pending {
            backlog, truncated, ..
        }
        | Sink::Lazy {
            backlog, truncated, ..
        } => (backlog, truncated),
        // A second `Direct3DCreate9` names the location again: the file
        // already open keeps everything, and stderr has no backlog.
        Sink::Open { file, early } => {
            *guard = Sink::Open { file, early: None };
            drop(guard);
            let Some(early) = early else {
                return path;
            };
            // The first naming after a crash report opened the early file.
            *TRACE_BASE.lock().expect("trace base mutex poisoned") =
                Some((dir, stem.to_owned(), pid));
            if early != path {
                log::info!(
                    target: crate::LOG_TARGET,
                    "log file: a crash report before Direct3DCreate9 opened {}, which stays the log instead of {}",
                    early.display(),
                    path.display()
                );
            }
            return early;
        }
        Sink::Stderr => (Vec::new(), false),
    };
    *guard = Sink::Lazy {
        path: path.clone(),
        backlog,
        truncated,
        early: false,
    };
    drop(guard);
    *TRACE_BASE.lock().expect("trace base mutex poisoned") = Some((dir, stem.to_owned(), pid));
    path
}

/// Name the early location, `<dir>/<stem>-<pid>.log`, for a crash report before `OpenLog`.
///
/// `dir` is the default log directory as the PE side derives it at load,
/// before `mtld3d.conf` is read. Kept only while the location is pending;
/// nothing is created until a crash report needs it.
pub fn set_early_location(dir: &str, stem: &str) {
    let path = Path::new(dir).join(mtld3d_shared::log_paths::log_file_name(
        stem,
        std::process::id(),
    ));
    let (Ok(dir_c), Ok(file_c)) = (CString::new(dir), CString::new(path.as_os_str().as_bytes()))
    else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "log file: the early location holds a NUL byte, a crash before Direct3DCreate9 logs to stderr"
        );
        return;
    };
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    if let Sink::Pending { early, .. } = &mut *guard {
        *early = Some(EarlyLocation {
            dir: dir_c,
            file: file_c,
            path,
        });
    }
}

/// Give up on a file: the backlog and everything after it go to stderr.
pub fn fall_back_to_stderr() {
    let mut guard = SINK.lock().expect("log file mutex poisoned");
    let spill = match core::mem::replace(&mut *guard, Sink::Stderr) {
        Sink::Pending { backlog, .. } | Sink::Lazy { backlog, .. } => Some(backlog),
        Sink::Open { file, early } => {
            *guard = Sink::Open { file, early };
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
///
/// Makes room first: the oldest logs beyond [`KEEP`] minus this one go.
fn create(path: &PathBuf, backlog: &[u8], truncated: bool) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        prune(parent, "log", KEEP - 1);
    }
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if truncated {
        let _ = file.write_all(TRUNCATED_NOTE);
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
    let spill = write_locked(&mut guard, bytes);
    drop(guard);
    write_stderr(spill);
}

/// Write one line of a crash report from an ordinary thread.
///
/// The PE side's exception handler and panic hook reach this through
/// `WriteLog`. While the location is pending the report opens the early one,
/// with the backlog ahead of it, or goes to stderr with the backlog when
/// there is none; afterwards it is written as any line is. The sink is only
/// tried, never waited for: a fault Wine raised out of a unix call can leave
/// it held by the faulting thread, and the report then goes to the
/// descriptor the sink has.
pub fn write_crash(bytes: &[u8]) {
    let mut guard = match SINK.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => {
            write_bytes(bytes);
            return;
        }
    };
    let spill = match crash_route(&sink_state(&guard), &CrashContext::Thread) {
        CrashRoute::EarlyFile => {
            *guard = match core::mem::replace(&mut *guard, Sink::Stderr) {
                Sink::Pending {
                    backlog,
                    truncated,
                    early: Some(early),
                } => Sink::Lazy {
                    path: early.path,
                    backlog,
                    truncated,
                    early: true,
                },
                other => other,
            };
            write_locked(&mut guard, bytes)
        }
        CrashRoute::Stderr => {
            let mut out = Vec::new();
            match core::mem::replace(&mut *guard, Sink::Stderr) {
                Sink::Pending {
                    backlog, truncated, ..
                } => {
                    if truncated {
                        out.extend_from_slice(TRUNCATED_NOTE);
                    }
                    out.extend_from_slice(&backlog);
                }
                other => *guard = other,
            }
            out.extend_from_slice(bytes);
            Some(out)
        }
        CrashRoute::Sink => write_locked(&mut guard, bytes),
        CrashRoute::Descriptor => {
            drop(guard);
            write_bytes(bytes);
            return;
        }
    };
    drop(guard);
    write_stderr(spill);
}

/// The descriptor a signal handler's crash report goes to, opened first if need be.
///
/// While the location is pending, the early location is created with
/// `mkdir(2)` and `open(2)` and the backlog is written into it, so the
/// report follows the lines that led up to it; without one the backlog goes
/// to stderr ahead of the report. Async-signal-safe: the sink is only tried,
/// since the faulting thread may hold it, and its backlog and early location
/// are moved out and leaked rather than freed, since the allocator's lock
/// may be held too. Every later call, and [`raw_fd`], answers the descriptor
/// this one settled on.
pub fn crash_fd() -> i32 {
    if LOG_FD.load(Ordering::Acquire) >= 0 {
        return raw_fd();
    }
    let mut guard = match SINK.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return raw_fd(),
    };
    match crash_route(&sink_state(&guard), &CrashContext::Signal) {
        CrashRoute::EarlyFile => open_early_from_signal(&mut guard),
        CrashRoute::Stderr => {
            spill_from_signal(&mut guard);
            raw_fd()
        }
        CrashRoute::Sink | CrashRoute::Descriptor => raw_fd(),
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
    prune(&dir, "gputrace", KEEP - 1);
    Some(path)
}

/// What the sink knows, for [`crash_route`].
const fn sink_state(sink: &Sink) -> SinkState {
    match sink {
        Sink::Pending { early, .. } => SinkState::Pending {
            early_location: early.is_some(),
        },
        Sink::Lazy { .. } => SinkState::Named,
        Sink::Open { .. } => SinkState::Open,
        Sink::Stderr => SinkState::Stderr,
    }
}

/// Write one line under the sink's lock, and hand back what stderr gets instead.
///
/// Whatever stderr gets is written by the caller after the lock is released.
fn write_locked(sink: &mut Sink, bytes: &[u8]) -> Option<Vec<u8>> {
    match sink {
        Sink::Pending {
            backlog, truncated, ..
        } => {
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
            early,
        } => match create(path, backlog, *truncated) {
            Ok(mut file) => {
                let ok = file.write_all(bytes).is_ok();
                LOG_FD.store(file.as_raw_fd(), Ordering::Release);
                let early = early.then(|| path.clone());
                *sink = if ok {
                    Sink::Open { file, early }
                } else {
                    Sink::Stderr
                };
                None
            }
            Err(e) => {
                let mut out = format!("[mtld3d::unix] log file: {e}; logging to stderr instead\n")
                    .into_bytes();
                out.extend_from_slice(backlog);
                out.extend_from_slice(bytes);
                *sink = Sink::Stderr;
                Some(out)
            }
        },
        Sink::Open { file, .. } => {
            if file.write_all(bytes).is_ok() {
                None
            } else {
                LOG_FD.store(-1, Ordering::Release);
                *sink = Sink::Stderr;
                Some(bytes.to_vec())
            }
        }
        Sink::Stderr => Some(bytes.to_vec()),
    }
}

/// Write what [`write_locked`] handed back to stderr.
fn write_stderr(spill: Option<Vec<u8>>) {
    if let Some(out) = spill {
        let _ = std::io::stderr().lock().write_all(&out);
    }
}

/// Open the early location from a signal handler and write the backlog into it.
///
/// Answers the file's descriptor, or stderr's when the file cannot be
/// opened, in which case the backlog goes there instead. Nothing here
/// allocates or frees: the C strings were built when the location was
/// named, the path moves into the sink, and the backlog and the C strings
/// are leaked.
fn open_early_from_signal(sink: &mut MutexGuard<'_, Sink>) -> i32 {
    const CANNOT_OPEN: &[u8] =
        b"[mtld3d::unix] log file: cannot open the early location, logging to stderr: ";

    let (backlog, truncated, early) = match core::mem::replace(&mut **sink, Sink::Stderr) {
        Sink::Pending {
            backlog,
            truncated,
            early: Some(early),
        } => (backlog, truncated, early),
        other => {
            **sink = other;
            return raw_fd();
        }
    };
    // SAFETY: mkdir(2) is async-signal-safe and `dir` is NUL-terminated; an
    // existing directory is the expected failure and changes nothing.
    unsafe { libc::mkdir(early.dir.as_ptr(), 0o755) };
    // SAFETY: open(2) is async-signal-safe and `file` is NUL-terminated.
    let fd = unsafe {
        libc::open(
            early.file.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND | libc::O_CLOEXEC,
            0o644,
        )
    };
    let target = if fd < 0 {
        write_fd(2, CANNOT_OPEN);
        write_fd(2, early.file.as_bytes());
        write_fd(2, b"\n");
        2
    } else {
        fd
    };
    if truncated {
        write_fd(target, TRUNCATED_NOTE);
    }
    write_fd(target, &backlog);
    let EarlyLocation { dir, file, path } = early;
    core::mem::forget(backlog);
    core::mem::forget(dir);
    core::mem::forget(file);
    if fd < 0 {
        // The sink stays on stderr, where the backlog went.
        core::mem::forget(path);
        return raw_fd();
    }
    // SAFETY: `fd` is the descriptor `open` just returned, owned by nothing
    // else; the `File` the sink keeps is its only owner from here on.
    let file = unsafe { File::from_raw_fd(fd) };
    **sink = Sink::Open {
        file,
        early: Some(path),
    };
    LOG_FD.store(fd, Ordering::Release);
    fd
}

/// Send a pending backlog to stderr from a signal handler; later lines follow it there.
///
/// The backlog and the early location are leaked rather than freed.
fn spill_from_signal(sink: &mut MutexGuard<'_, Sink>) {
    match core::mem::replace(&mut **sink, Sink::Stderr) {
        Sink::Pending {
            backlog,
            truncated,
            early,
        } => {
            if truncated {
                write_fd(2, TRUNCATED_NOTE);
            }
            write_fd(2, &backlog);
            core::mem::forget(backlog);
            core::mem::forget(early);
        }
        other => **sink = other,
    }
}

/// Write all of `bytes` to `fd` with `write(2)`, retrying short writes; async-signal-safe.
fn write_fd(fd: i32, bytes: &[u8]) {
    let mut rest = bytes;
    while !rest.is_empty() {
        // SAFETY: write(2) is async-signal-safe; `rest` is readable for its length.
        let written = unsafe { libc::write(fd, rest.as_ptr().cast::<c_void>(), rest.len()) };
        let Ok(written) = usize::try_from(written) else {
            return;
        };
        if written == 0 {
            return;
        }
        rest = &rest[written.min(rest.len())..];
    }
}

/// Remove the oldest entries in `dir` with extension `ext` beyond the newest `keep`.
///
/// Age is the modification time; entries whose metadata cannot be read are
/// left alone. A trace is a directory bundle, a log a file; both go whole.
///
/// Nothing here logs, on purpose: the log-file creation calls this under
/// the sink's mutex, and a log line from inside would re-enter the sink and
/// deadlock. A removal that fails (two processes sharing one directory can
/// race for the same file) is simply tried again at the next creation.
fn prune(dir: &Path, ext: &str, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut aged: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == ext))
        .filter_map(|p| {
            let modified = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((modified, p))
        })
        .collect();
    if aged.len() <= keep {
        return;
    }
    // Newest last; a tie keeps the order stable by name.
    aged.sort();
    let doomed = aged.len() - keep;
    for (_, path) in aged.into_iter().take(doomed) {
        let _ = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
    }
}

#[cfg(test)]
mod tests;

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
