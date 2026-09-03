//! The PE-side logger's sink: a queue drained by one logging thread.
//!
//! `env_logger` hands every formatted line to [`Sink::write`], which only
//! pushes the bytes onto an unbounded channel: no unix call and no blocking,
//! so the API and encoder threads pay an allocation and a queue push per line
//! and nothing else. One logging thread, started by [`start`] from the first
//! `Direct3DCreate9` (never from `DllMain`, which runs under the loader lock),
//! drains the queue and forwards each line through the `WriteLog` thunk into
//! the process's log file, which the unix side owns. Lines logged before the
//! thread exists wait in the queue; the queue keeps their order.
//!
//! [`open`] names that file's location first: the directory `log.dir` picks,
//! or `mtld3d-logs` next to the executable, as a unix path the unix side can
//! create, plus the executable's stem and the pid that make the file name.

use core::ffi::{c_char, c_void};
use std::{
    os::windows::ffi::OsStrExt,
    path::Path,
    sync::{
        LazyLock, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
};

use log::warn;
use mtld3d_shared::{OpenLogParams, WriteLogParams};

use crate::{LOG_TARGET, unix_call::unix_call};

unsafe extern "system" {
    fn GetProcessHeap() -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, mem: *mut c_void) -> i32;
    fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, proc_name: *const u8) -> *mut c_void;
}

/// Wine's kernel32 extension: the unix path of a DOS path, heap-allocated.
///
/// Resolved by name at run time, since the SDK import library we link
/// against does not carry it. Wine declares it `CDECL`, not `WINAPI`: on
/// i686 the caller pops the argument, and a stdcall type here corrupts the
/// caller's stack.
type WineGetUnixFileName = unsafe extern "C" fn(*const u16) -> *mut c_char;

/// Name the process's log location to the unix side.
///
/// The directory is `log.dir` resolved against the executable's directory
/// (an absolute value stands as is), or `mtld3d-logs` beside the executable
/// when the key is empty; it is created here, the file itself by the unix
/// side with the first line it writes. When the directory cannot be created
/// or has no unix path, the thunk still goes out, empty, so the unix side
/// releases its backlog to stderr instead of holding it forever.
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
        pid: std::process::id(),
        pad0: 0,
    };
    unix_call(&mut params);
}

/// The log directory as a unix path plus the executable's stem, or why not.
fn log_location(cfg: &mtld3d_core::config::Mtld3dConfig) -> Result<(String, String), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe() unavailable ({e})"))?;
    let exe_dir = exe.parent().unwrap_or_else(|| Path::new("."));
    let dir = if cfg.log_dir.is_empty() {
        exe_dir.join("mtld3d-logs")
    } else {
        exe_dir.join(&cfg.log_dir)
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {} ({e})", dir.display()))?;
    let stem = exe.file_stem().map_or_else(
        || String::from("mtld3d"),
        |s| s.to_string_lossy().into_owned(),
    );
    let unix_dir = unix_path(&dir).ok_or_else(|| format!("no unix path for {}", dir.display()))?;
    Ok((unix_dir, stem))
}

/// The unix path of a DOS path, through Wine's `wine_get_unix_file_name`.
fn unix_path(dos: &Path) -> Option<String> {
    // SAFETY: kernel32 is loaded for the life of the process; the name is
    // NUL-terminated.
    let kernel32 = unsafe { GetModuleHandleA(c"kernel32.dll".as_ptr().cast::<u8>()) };
    if kernel32.is_null() {
        return None;
    }
    // SAFETY: a live module handle and a NUL-terminated export name.
    let proc =
        unsafe { GetProcAddress(kernel32, c"wine_get_unix_file_name".as_ptr().cast::<u8>()) };
    if proc.is_null() {
        return None;
    }
    // SAFETY: the export has this signature in every Wine that carries it.
    let get_unix_file_name: WineGetUnixFileName = unsafe { core::mem::transmute(proc) };
    let wide: Vec<u16> = dos
        .as_os_str()
        .encode_wide()
        .chain(core::iter::once(0))
        .collect();
    // SAFETY: `wide` is a NUL-terminated UTF-16 string live for the call;
    // the result is a NUL-terminated string on the process heap or null.
    let raw = unsafe { get_unix_file_name(wide.as_ptr()) };
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` is a valid NUL-terminated C string until freed below.
    let path = unsafe { core::ffi::CStr::from_ptr(raw) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: the process heap handle is a stable kernel32 pseudo-handle.
    let heap = unsafe { GetProcessHeap() };
    // SAFETY: `raw` came from the process heap per the kernel32 contract.
    let _ = unsafe { HeapFree(heap, 0, raw.cast::<c_void>()) };
    Some(path)
}

struct Queue {
    tx: Sender<Vec<u8>>,
    /// Handed to the logging thread by [`start`]; `None` once it has been.
    rx: Mutex<Option<Receiver<Vec<u8>>>>,
}

static QUEUE: LazyLock<Queue> = LazyLock::new(|| {
    let (tx, rx) = channel();
    Queue {
        tx,
        rx: Mutex::new(Some(rx)),
    }
});

/// The `env_logger` target: queues the line for the logging thread.
pub struct Sink;

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // A closed receiver means the logging thread is gone; the line is
        // dropped rather than the caller failing.
        let _ = QUEUE.tx.send(buf.to_vec());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Start the logging thread; a no-op after the first call.
///
/// Called from `Direct3DCreate9`, the first entry point a game reaches outside
/// `DllMain`. The handle is dropped: the thread is detached and lives for the
/// process.
pub fn start() {
    let rx = QUEUE
        .rx
        .lock()
        .expect("log queue receiver lock poisoned")
        .take();
    let Some(rx) = rx else {
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("mtld3d-log".into())
        .spawn(move || {
            for line in rx {
                forward(&line);
            }
        });
    // On a spawn failure the closure, and the receiver with it, is dropped:
    // every later push fails and the line is discarded, which is the most
    // the logger can do without a thread of its own.
    drop(spawned);
}

/// Hand `line` to the unix side right now, on the calling thread.
///
/// For the crash path only: a fault or panic handler cannot rely on the
/// logging thread ever running again, so it thunks synchronously instead
/// of queueing. Everything else goes through [`Sink`].
pub fn write_raw(line: &[u8]) {
    forward(line);
}

/// One formatted line across the boundary; the unix side writes it to the log file.
fn forward(line: &[u8]) {
    let Ok(len) = u32::try_from(line.len()) else {
        return;
    };
    let mut params = WriteLogParams {
        ptr: line.as_ptr() as usize as u64,
        len,
        pad0: 0,
    };
    unix_call(&mut params);
}
