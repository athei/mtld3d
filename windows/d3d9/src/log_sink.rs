//! The PE-side logger's sink: a queue drained by one logging thread.
//!
//! `env_logger` hands every formatted line to [`Sink::write`], which copies
//! the bytes into an unbounded channel without a Unix call or a queue-capacity
//! wait. PE callers pay formatting, a line allocation and a queue push. Native
//! encoder workers use the Unix logger directly instead of this sink.
//! One logging thread drains the queue and forwards each
//! line through the `WriteLog` thunk into the process's log file, which the
//! unix side owns. Lines logged while no thread runs wait in the queue; the
//! queue keeps their order.
//!
//! The thread lives from the first `IDirect3D9` to the last: [`acquire`],
//! from `Direct3DCreate9` (never from `DllMain`, which runs under the loader
//! lock), starts it, and [`release`], from the last interface's drop, stops
//! it and waits for it to be gone. The thread holds a reference on
//! `d3d9.dll` for its lifetime and leaves through `FreeLibraryAndExitThread`,
//! so a `FreeLibrary` cannot unmap the image under a running thread, and one
//! that follows the last `Release` finds no thread of ours in the image.
//! That exit needs the loader lock, so a last `Release` whose caller holds it
//! (another DLL's `DLL_PROCESS_DETACH`, a TLS callback) sends the stop and
//! returns without waiting; the thread's own reference keeps the image mapped
//! until it is gone.
//!
//! [`open`] names that file's location first: the directory `log.dir` picks,
//! or `mtld3d-logs` next to the executable, as a unix path the unix side can
//! create, plus the executable's stem for the file name. The pid in the name
//! is the unix side's own: the one this side sees is Wine's process id, which
//! repeats from one launch to the next.
//!
//! A crash cannot wait for that. [`early_location`] derives the default
//! location at load for `InitLogger`, and [`write_crash`], the path of a
//! fault that ends the process, marks its lines as a crash report, which
//! opens that location on the unix side when nothing has named one yet. A
//! fault that may still be handled goes through [`write_fault`] instead,
//! which opens nothing. Both first hand over the lines still queued for a
//! logging thread that is not running, so they reach the log ahead of the
//! report rather than dying in the queue.

use core::ffi::c_void;
use std::{
    os::windows::io::AsRawHandle,
    path::Path,
    sync::{
        LazyLock, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
};

use log::warn;
use mtld3d_shared::{LogLineKind, OpenLogParams, WriteLogParams, log_once_warn};

use crate::{
    LOG_TARGET,
    crash::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GetModuleHandleExA},
    unix_call::unix_call,
};

unsafe extern "system" {
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn FreeLibraryAndExitThread(module: *mut c_void, exit_code: u32) -> !;
    fn GetThreadId(thread: *mut c_void) -> u32;
    fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> *mut c_void;
    fn WaitForSingleObject(handle: *mut c_void, millis: u32) -> u32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn RtlGetCurrentPeb() -> *mut c_void;
    fn RtlIsCriticalSectionLockedByThread(section: *mut c_void) -> i32;
}

/// The log directory beside the executable when `log.dir` is empty.
const DEFAULT_DIR: &str = "mtld3d-logs";

/// `OpenThread` access right that allows waiting for the thread's end.
const SYNCHRONIZE: u32 = 0x0010_0000;
/// `WaitForSingleObject` timeout that never expires.
const INFINITE: u32 = 0xFFFF_FFFF;
/// `WaitForSingleObject` result for a signaled object.
const WAIT_OBJECT_0: u32 = 0;
/// Offset of `PEB.LoaderLock`, the loader's `RTL_CRITICAL_SECTION*`, in a 32-bit PEB.
#[cfg(target_pointer_width = "32")]
const PEB_LOADER_LOCK_OFFSET: usize = 0xa0;
/// Offset of `PEB.LoaderLock`, the loader's `RTL_CRITICAL_SECTION*`, in a 64-bit PEB.
#[cfg(target_pointer_width = "64")]
const PEB_LOADER_LOCK_OFFSET: usize = 0x110;

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
    let unix_dir = crate::wine_path::unix_path(&dir)
        .ok_or_else(|| format!("no unix path for {}", dir.display()))?;
    Ok((unix_dir, stem))
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
    let unix_exe_dir = crate::wine_path::unix_path(exe_dir)?;
    let stem = exe.file_stem().map_or_else(
        || String::from("mtld3d"),
        |s| s.to_string_lossy().into_owned(),
    );
    let separator = if unix_exe_dir.ends_with('/') { "" } else { "/" };
    Some((format!("{unix_exe_dir}{separator}{DEFAULT_DIR}"), stem))
}

/// What travels to the logging thread.
enum Message {
    /// One formatted line.
    Line(Vec<u8>),
    /// Drain what is queued, park the receiver, exit.
    Stop,
}

struct Queue {
    tx: Sender<Message>,
    /// The receiver, parked while no logging thread runs.
    ///
    /// The thread takes it at start and puts it back before it exits.
    rx: Mutex<Option<Receiver<Message>>>,
    worker: Mutex<Worker>,
}

/// The logging thread and the interfaces that keep it running.
struct Worker {
    /// Live `IDirect3D9` interfaces; the thread runs while this is non-zero.
    interfaces: u32,
    /// The OS thread id of the running logging thread.
    thread_id: Option<u32>,
}

static QUEUE: LazyLock<Queue> = LazyLock::new(|| {
    let (tx, rx) = channel();
    Queue {
        tx,
        rx: Mutex::new(Some(rx)),
        worker: Mutex::new(Worker {
            interfaces: 0,
            thread_id: None,
        }),
    }
});

/// The `env_logger` target: queues the line for the logging thread.
pub struct Sink;

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // A closed receiver means the logging thread failed to start; the
        // line is dropped rather than the caller failing.
        let _ = QUEUE.tx.send(Message::Line(buf.to_vec()));
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// One more `IDirect3D9` exists: start the logging thread if none runs.
///
/// Called from `Direct3DCreate9`, the first entry point a game reaches outside
/// `DllMain`. The thread takes a reference on this image before it starts
/// and gives it back only as it exits, so no `FreeLibrary` can unmap the
/// image while it runs.
pub fn acquire() {
    let mut worker = QUEUE.worker.lock().expect("log worker lock poisoned");
    worker.interfaces += 1;
    if worker.thread_id.is_none() {
        start(&mut worker);
    }
}

/// One `IDirect3D9` is gone: with the last one, stop the logging thread and wait for it.
///
/// Called from the interface's drop. The wait returns only when the thread
/// has exited, which is the one observation that proves no code of this
/// image will run on it again; a `FreeLibrary` after that can unmap the
/// image. Everything queued before the stop is written first. The worker
/// lock is held throughout, so an interface created meanwhile starts its
/// thread only once this one is gone.
pub fn release() {
    let mut worker = QUEUE.worker.lock().expect("log worker lock poisoned");
    worker.interfaces = worker.interfaces.saturating_sub(1);
    if worker.interfaces == 0 {
        stop(&mut worker);
    }
}

/// Spawn the logging thread with a reference on this image, and record its id.
fn start(worker: &mut Worker) {
    let rx = QUEUE
        .rx
        .lock()
        .expect("log queue receiver lock poisoned")
        .take();
    let Some(rx) = rx else {
        // The previous thread has not parked the receiver: its exit could not
        // be waited for, and it still holds the queue.
        log_once_warn!(
            target: LOG_TARGET,
            "log thread: the previous thread still holds the queue, not starting another"
        );
        return;
    };
    let Some(module) = add_image_reference() else {
        *QUEUE.rx.lock().expect("log queue receiver lock poisoned") = Some(rx);
        return;
    };
    let module_addr = module as usize;
    let spawned = std::thread::Builder::new()
        .name("mtld3d-log".into())
        .spawn(move || drain(rx, module_addr));
    let handle = match spawned {
        Ok(handle) => handle,
        Err(e) => {
            // The closure, and the receiver with it, is gone: every later
            // push fails and the line is discarded, which is the most the
            // logger can do without a thread of its own.
            log_once_warn!(target: LOG_TARGET, "log thread: spawn failed ({e}), lines are dropped");
            // SAFETY: balancing the reference `add_image_reference` took for
            // the thread that never ran.
            unsafe { FreeLibrary(module) };
            return;
        }
    };
    // SAFETY: a fresh, owned thread handle; the id it names is stable for
    // the thread's lifetime.
    let id = unsafe { GetThreadId(handle.as_raw_handle()) };
    // The handle is not kept: `stop` opens a fresh one when it waits, and a
    // handle held for a whole session is not reliable.
    drop(handle);
    if id == 0 {
        log_once_warn!(target: LOG_TARGET, "log thread: GetThreadId failed, its exit cannot be waited for");
    }
    worker.thread_id = (id != 0).then_some(id);
}

/// Send the stop and wait for the logging thread to be gone.
///
/// Except when the calling thread holds the loader lock: the logging thread
/// leaves through `FreeLibraryAndExitThread`, whose unload and thread detach
/// both take that lock, so it cannot be gone before the caller lets the lock
/// go. Then the stop goes out without the wait. The thread drains what is
/// queued and exits once the lock is free, and its own reference on the image
/// keeps it mapped until then; an interface created before the thread has put
/// the queue back starts no thread of its own, and its lines wait for the next
/// one.
fn stop(worker: &mut Worker) {
    let Some(thread_id) = worker.thread_id.take() else {
        return;
    };
    if caller_holds_loader_lock() {
        log::debug!(
            target: LOG_TARGET,
            "log thread: the last release holds the loader lock; stopping without waiting"
        );
        let _ = QUEUE.tx.send(Message::Stop);
        return;
    }
    // A fresh handle, opened while the thread is certainly alive (the stop
    // is not sent yet), so the id cannot name another thread.
    // SAFETY: plain kernel32 call; a failure returns null.
    let handle = unsafe { OpenThread(SYNCHRONIZE, 0, thread_id) };
    let _ = QUEUE.tx.send(Message::Stop);
    if handle.is_null() {
        log_once_warn!(
            target: LOG_TARGET,
            "log thread: OpenThread failed, not waiting for its exit; the image stays mapped until it is gone"
        );
        return;
    }
    // SAFETY: `handle` is the live SYNCHRONIZE handle opened above.
    let waited = unsafe { WaitForSingleObject(handle, INFINITE) };
    // SAFETY: closing the handle opened above, exactly once.
    unsafe { CloseHandle(handle) };
    if waited != WAIT_OBJECT_0 {
        log_once_warn!(
            target: LOG_TARGET,
            "log thread: the wait for its exit failed ({waited:#x}); the image stays mapped until it is gone"
        );
    }
}

/// Whether the calling thread holds the loader lock, read off the process's PEB.
pub fn caller_holds_loader_lock() -> bool {
    // SAFETY: ntdll export; answers this process's PEB, which lives as long as
    // the process.
    let peb = unsafe { RtlGetCurrentPeb() };
    if peb.is_null() {
        return false;
    }
    // SAFETY: the offset is `PEB.LoaderLock`'s on this architecture, inside the PEB.
    let field = unsafe { peb.cast::<u8>().add(PEB_LOADER_LOCK_OFFSET) };
    // SAFETY: the field is a pointer the loader set at process start, readable
    // for the process's lifetime; `read_unaligned` asks nothing of its alignment.
    let section = unsafe { field.cast::<*mut c_void>().read_unaligned() };
    if section.is_null() {
        return false;
    }
    // SAFETY: ntdll export; `section` is the loader's live critical section.
    unsafe { RtlIsCriticalSectionLockedByThread(section) != 0 }
}

/// The logging thread: forward every line until the stop, then leave with the image reference.
///
/// `module_addr` is the `HMODULE` of this image as an address, so the closure
/// that carries it into the thread is `Send`. The receiver goes back to the
/// queue for the next thread before the exit; the exit itself never returns
/// into this image.
fn drain(rx: Receiver<Message>, module_addr: usize) {
    while let Ok(Message::Line(line)) = rx.recv() {
        forward(&line, LogLineKind::Ordinary);
    }
    *QUEUE.rx.lock().expect("log queue receiver lock poisoned") = Some(rx);
    // SAFETY: `module_addr` is the HMODULE `add_image_reference` returned,
    // whose reference this thread owns; the call releases it and ends the
    // thread without returning.
    unsafe { FreeLibraryAndExitThread(module_addr as *mut c_void, 0) }
}

/// Take one reference on this image for a logging thread, or say why not.
fn add_image_reference() -> Option<*mut c_void> {
    let mut module: *mut c_void = core::ptr::null_mut();
    // SAFETY: kernel32 export; `QUEUE` is a static of this image, so its
    // address names the module, and the flag without UNCHANGED_REFCOUNT
    // adds one reference the thread gives back as it exits.
    let ok = unsafe {
        GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            (&raw const QUEUE).cast::<u8>(),
            &raw mut module,
        )
    };
    if ok == 0 || module.is_null() {
        log_once_warn!(target: LOG_TARGET, "log thread: GetModuleHandleEx failed, not started; lines wait in the queue");
        return None;
    }
    Some(module)
}

/// Hand a line of a terminal fault's report to the unix side right now, on the calling thread.
///
/// For the crash path only: a fault or panic handler cannot rely on the
/// logging thread ever running again, so it thunks synchronously instead
/// of queueing. The lines queued while no logging thread runs (every line
/// before `Direct3DCreate9`) go first, so the report follows them. The line
/// travels as a crash report, which opens the early log location when none
/// is named yet. Everything else goes through [`Sink`].
pub fn write_crash(line: &[u8]) {
    forward_parked();
    forward(line, LogLineKind::CrashReport);
}

/// Hand a line of a first-chance fault report to the unix side right now, on the calling thread.
///
/// As [`write_crash`], except that the fault may still be handled: the line
/// goes where any line goes and opens no log. The crumb dump's sink.
pub fn write_fault(line: &[u8]) {
    forward_parked();
    forward(line, LogLineKind::FaultReport);
}

/// Forward the lines waiting in the queue while no logging thread runs.
///
/// The receiver is parked in the queue then, and only tried: a crash on a
/// thread that holds it, or one the logging thread races for, leaves the
/// lines where they are rather than waiting. Each line is leaked once it is
/// forwarded rather than freed, since this runs inside the exception
/// handler, whose faulting thread may hold the allocator.
fn forward_parked() {
    let Ok(parked) = QUEUE.rx.try_lock() else {
        return;
    };
    let Some(rx) = parked.as_ref() else {
        return;
    };
    while let Ok(message) = rx.try_recv() {
        if let Message::Line(line) = message {
            forward(&line, LogLineKind::Ordinary);
            core::mem::forget(line);
        }
    }
}

/// One formatted line across the boundary; the unix side writes it to the log file.
fn forward(line: &[u8], kind: LogLineKind) {
    let Ok(len) = u32::try_from(line.len()) else {
        return;
    };
    let mut params = WriteLogParams {
        ptr: line.as_ptr() as usize as u64,
        len,
        kind,
    };
    unix_call(&mut params);
}
