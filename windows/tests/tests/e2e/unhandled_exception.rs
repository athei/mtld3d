//! An exception nothing handles, raised before the process creates an interface.
//!
//! `d3d9.dll` installs a top-level unhandled-exception filter from `DllMain`,
//! which reports such an exception as a crash and opens the default log. It is
//! best effort, and this process shows its limit: the startup code MSVC
//! links into every executable it builds runs after a statically imported
//! d3d9.dll's `DllMain`, sets a filter of its own and, from Visual Studio
//! 2015 on, does not chain to the one it replaces. A game built the same way
//! loses the filter too.
//! What the test pins is what then happens: the C runtime's filter is the one
//! on top, the exception still ends the process with its own code, and the
//! first-chance report, which opens nothing, leaves no stray log behind.
//!
//! The first-chance report the unix signal handler writes for a hardware
//! fault goes to the process's unix stderr, which Wine does not route into a
//! child's redirected standard error, so it is not read here; the unix unit
//! tests pin that route. The filter's own report and its routing are pinned
//! by the `crash_report` unit tests in `mtld3d-core`.

use core::ffi::{c_char, c_void};

use mtld3d_tests::run_child;

use super::device::running_as;

/// The name the dying child runs under.
const CHILD_NAME: &str = "unhandled-exception.exe";

/// `SetErrorMode` flag: no fault dialog and no debugger for an unhandled exception.
const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;

/// `EXCEPTION_ACCESS_VIOLATION`, the code the child raises.
const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;

/// The child's line naming the top-level filter and where d3d9.dll lies.
const FILTER_LINE: &str = "[child] top-level filter in d3d9.dll:";

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetErrorMode(mode: u32) -> u32;
    fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const c_void);
    fn SetUnhandledExceptionFilter(filter: *mut c_void) -> *mut c_void;
    fn GetModuleHandleExA(flags: u32, name: *const c_char, module: *mut *mut c_void) -> i32;
    fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
}

/// `GetModuleHandleEx` flags: look up by address, take no reference.
const MODULE_FROM_ADDRESS_UNCHANGED: u32 = 0x4 | 0x2;

/// An unhandled exception before `Direct3DCreate9` in an MSVC-built process leaves no log.
///
/// The child reports which module the top-level filter at the time of the
/// exception lies in, which is not d3d9.dll, then raises an access violation
/// nothing handles with the debugger turned off. It ends with the exception's
/// code, and no `mtld3d-logs` directory appears beside its copy of the
/// executable: the vectored handler's first-chance report waited in the
/// backlog, as a fault the process might have recovered from.
#[test]
fn an_unhandled_exception_before_direct3dcreate9_in_a_c_runtime_process_leaves_no_log() {
    if running_as(CHILD_NAME) {
        // SAFETY: kernel32 export; the filter read is put straight back.
        let top = unsafe { SetUnhandledExceptionFilter(core::ptr::null_mut()) };
        // SAFETY: as above, restoring what was read.
        unsafe { SetUnhandledExceptionFilter(top) };
        let mut owner: *mut c_void = core::ptr::null_mut();
        // SAFETY: kernel32 export; `top` is only a lookup key.
        unsafe { GetModuleHandleExA(MODULE_FROM_ADDRESS_UNCHANGED, top.cast(), &raw mut owner) };
        // SAFETY: kernel32 export with a NUL-terminated name.
        let d3d9 = unsafe { GetModuleHandleA(c"d3d9.dll".as_ptr()) };
        println!(
            "{FILTER_LINE} {} (filter {top:p}, its module {owner:p}, d3d9.dll {d3d9:p})",
            !d3d9.is_null() && owner == d3d9
        );
        // SAFETY: kernel32 export; changes this child's error mode only.
        unsafe { SetErrorMode(SEM_NOGPFAULTERRORBOX) };
        // SAFETY: kernel32 export; the exception is the point of the child,
        // and no argument array is passed.
        unsafe { RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, core::ptr::null()) };
        unreachable!("an unhandled exception ends the process");
    }

    let exe = std::env::current_exe().expect("resolve test executable");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock follows Unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "{}-{}-{stamp}",
        CHILD_NAME.trim_end_matches(".exe"),
        std::process::id()
    ));
    std::fs::create_dir(&dir).expect("create the child's private directory");
    let child = dir.join(CHILD_NAME);
    std::fs::copy(&exe, &child).expect("copy the test executable");
    let mut command = std::process::Command::new(&child);
    command.args([
        "--exact",
        "unhandled_exception::an_unhandled_exception_before_direct3dcreate9_in_a_c_runtime_process_leaves_no_log",
        "--nocapture",
    ]);
    // Info records on both sides, so a log that did appear would hold lines.
    command.envs([("RUST_LOG", "info"), ("__CX_UNIX_RUST_LOG", "info")]);
    let output = run_child(&mut command, "").expect("run the dying child");
    let report = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let logs = dir.join("mtld3d-logs").exists();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        output.status.code().map(i32::cast_unsigned),
        Some(EXCEPTION_ACCESS_VIOLATION),
        "the child ends with the exception's code:\n{report}"
    );
    let filter = report
        .lines()
        .find(|line| line.starts_with(FILTER_LINE))
        .unwrap_or_else(|| panic!("the child named no filter:\n{report}"));
    assert!(
        filter.starts_with(&format!("{FILTER_LINE} false")),
        "the C runtime's filter replaced d3d9.dll's: {filter}"
    );
    assert!(
        !logs,
        "a first-chance report opened a log before Direct3DCreate9:\n{report}"
    );
}
