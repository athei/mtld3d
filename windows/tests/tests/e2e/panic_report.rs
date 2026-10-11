//! A panic inside `d3d9.dll` leaves its report in the process's log file.
//!
//! A game a launcher spawned has no usable standard streams, so the line std's
//! default panic hook prints to stderr, the thread, the location and the
//! message, is lost unless the layer writes it into its own log as well. The
//! layer is built with `panic = "abort"`, so the process ends inside the hook
//! and the line has to be in the file before the hook returns.
//!
//! The panic comes from `IDirect3D9::AddRef` with a null `this`: a null
//! `this` is undefined behaviour per the D3D9 ABI, and the layer's `IUnknown`
//! thunks panic on it by design so a refcount bug surfaces at the call. It
//! panics in every build profile, which makes it a fixed way to reach the
//! panic hook from an application call. It runs in a child process of its
//! own, which the panic ends.

use core::ffi::c_void;

use mtld3d_tests::{Harness, spawn_scoped};
use mtld3d_types::IDirect3D9Vtbl;

use super::device::{running_as, spawn_private_log_child};

/// The copy of the test executable that panics inside the layer.
const CHILD_NAME: &str = "panic-in-layer.exe";

/// The prefix of the layer's panic line in its log, ahead of std's own wording.
const LOG_PANIC_PREFIX: &str = "[mtld3d::d3d9] PANIC: ";

/// The line that opens the crumb trail, which the panic line must precede.
const CRUMB_TRAIL_LINE: &str = "[mtld3d::d3d9] PANIC - dumping crumb trail:";

/// What the child prints on stdout ahead of its worker's name.
const WORKER_NAME_LABEL: &str = "panic worker name: ";

/// A panic in an entry point writes std's panic line and message into the process log.
///
/// The child calls `IDirect3D9::AddRef` with a null `this` on a named worker
/// thread and dies. Its stderr carries the default hook's report,
/// `thread '<name>' (<id>) panicked at <file>:<line>:<column>:` and the
/// message on the next line, from `d3d9.dll`'s own copy of std, which never
/// saw the name the test executable's std gave the worker and so prints
/// `<unnamed>`. The log must carry the same two lines behind the layer's
/// prefix, ahead of the crumb trail, with the worker's name in place of
/// `<unnamed>`: the thread's Win32 description, which std sets when it names
/// a thread, carries it across images. So the thread name, the thread id,
/// the location and the message all reach a log a remote report attaches.
#[test]
fn a_panic_in_an_entry_point_writes_its_message_to_the_process_log() {
    if running_as(CHILD_NAME) {
        panic_in_layer_workload();
        return;
    }
    let (output, dir) = spawn_private_log_child(
        CHILD_NAME,
        "panic_report::a_panic_in_an_entry_point_writes_its_message_to_the_process_log",
        "warn",
        "",
    );
    let log = private_log(&dir);
    std::fs::remove_dir_all(&dir).expect("remove the child's private directory");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "the child outlived a null-this AddRef: {stderr}"
    );
    let worker = stdout
        .lines()
        .find_map(|line| line.strip_prefix(WORKER_NAME_LABEL))
        .unwrap_or_else(|| panic!("the child named no worker: {stdout}"));

    let lines: Vec<&str> = stderr.lines().collect();
    let header_at = lines
        .iter()
        .position(|line| line.starts_with("thread '") && line.contains(" panicked at "))
        .unwrap_or_else(|| panic!("no panic report on the child's stderr: {stderr}"));
    let message = lines
        .get(header_at + 1)
        .unwrap_or_else(|| panic!("the panic report on stderr has no message: {stderr}"));
    assert!(
        message.contains("this=null"),
        "the panic is the null-this one: {message}"
    );
    // `thread '<name>' (<id>) panicked at <location>:`, the name dropped:
    // the id, the location and everything after them stay as stderr has them.
    let after_name = lines[header_at]
        .strip_prefix("thread '")
        .and_then(|rest| rest.split_once("' ("))
        .map_or_else(
            || panic!("an unexpected panic header: {}", lines[header_at]),
            |(_, rest)| rest,
        );

    let log_lines: Vec<&str> = log.lines().collect();
    let expected = format!("{LOG_PANIC_PREFIX}thread '{worker}' ({after_name}");
    let logged_at = log_lines
        .iter()
        .position(|line| *line == expected)
        .unwrap_or_else(|| panic!("the log lacks {expected:?}:\n{log}"));
    assert_eq!(
        log_lines.get(logged_at + 1),
        Some(message),
        "the message follows the panic line in the log:\n{log}"
    );
    let trail_at = log_lines
        .iter()
        .position(|line| *line == CRUMB_TRAIL_LINE)
        .unwrap_or_else(|| panic!("the log lacks the crumb trail:\n{log}"));
    assert!(
        logged_at < trail_at,
        "the panic line precedes the crumb trail:\n{log}"
    );
}

/// Call `IDirect3D9::AddRef` with a null `this` on a named worker, which ends the process.
///
/// The worker is named after the test and prints its name on stdout first,
/// so the parent knows what the log should call it. Should the call return,
/// the worker ends and the test passes in the child, which the parent reads
/// as the panic not happening.
fn panic_in_layer_workload() {
    let h = Harness::factory_only();
    let factory = h.factory() as usize;
    std::thread::scope(|scope| {
        spawn_scoped(scope, || {
            let current = std::thread::current();
            let name = current.name().expect("a test's worker carries a name");
            println!("{WORKER_NAME_LABEL}{name}");
            let interface = factory as *mut c_void;
            // SAFETY: `interface` is the live `IDirect3D9` the harness created,
            // whose first word is its vtable.
            let vtbl_ptr = unsafe { *interface.cast::<*const IDirect3D9Vtbl>() };
            // SAFETY: the vtable of a live interface is a `'static` table.
            let vtbl = unsafe { &*vtbl_ptr };
            // SAFETY: vtable thunk called with a null `this`, which the
            // layer's `IUnknown` thunks answer with a panic.
            unsafe { (vtbl.add_ref)(core::ptr::null_mut()) }
        })
        .join()
        .expect("the worker returns or the process ends");
    });
}

/// The process log the child left in its private directory, empty when it left none.
fn private_log(dir: &std::path::Path) -> String {
    let entries = std::fs::read_dir(dir).expect("read the child's private directory");
    let logs: Vec<_> = entries
        .map(|entry| entry.expect("read a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
        .collect();
    assert!(
        logs.len() <= 1,
        "one process log beside the child: {logs:?}"
    );
    logs.first().map_or_else(String::new, |log| {
        std::fs::read_to_string(log).expect("read the process log")
    })
}
