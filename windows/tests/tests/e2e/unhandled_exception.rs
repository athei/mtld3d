//! An exception nothing handles, raised before the process creates an interface.
//!
//! `d3d9.dll` installs a top-level unhandled-exception filter at load. A
//! process that dies of an exception before its first `Direct3DCreate9` has
//! no log location named yet, so the filter's report opens the default one,
//! `mtld3d-logs` beside the executable, with every line logged so far ahead
//! of it. The child here loads the layer through the suite's static import,
//! turns off the debugger Wine would start, and raises an access violation
//! nothing catches; the parent reads the log the child left beside its copy
//! of the executable.

use core::ffi::c_void;

use mtld3d_tests::run_child;

use super::device::running_as;

/// The name the dying child runs under.
const CHILD_NAME: &str = "unhandled-exception.exe";

/// `SetErrorMode` flag: no fault dialog and no debugger for an unhandled exception.
const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;

/// `EXCEPTION_ACCESS_VIOLATION`, the code the child raises.
const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetErrorMode(mode: u32) -> u32;
    fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const c_void);
    fn SetUnhandledExceptionFilter(filter: *mut c_void) -> *mut c_void;
    fn GetModuleHandleA(name: *const core::ffi::c_char) -> *mut c_void;
}

/// The report of an unhandled exception before `Direct3DCreate9` lands in the default log.
///
/// The child's log holds the layer's identity line, the vectored handler's
/// first-chance line for the exception, and after it the filter's terminal
/// line, which refers to that report rather than repeating it. The child
/// ends with the exception's code, the filter having passed it on.
#[test]
fn an_unhandled_exception_before_direct3dcreate9_reaches_the_default_log() {
    if running_as(CHILD_NAME) {
        // Which filter is on top, and where d3d9.dll is, for a failure to name.
        // SAFETY: kernel32 export; the filter read is put straight back.
        let top = unsafe { SetUnhandledExceptionFilter(core::ptr::null_mut()) };
        // SAFETY: as above, restoring what was read.
        unsafe { SetUnhandledExceptionFilter(top) };
        // SAFETY: kernel32 export with a NUL-terminated name.
        let d3d9 = unsafe { GetModuleHandleA(c"d3d9.dll".as_ptr()) };
        println!("[child] top-level filter {top:p}, d3d9.dll at {d3d9:p}");
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
        "unhandled_exception::an_unhandled_exception_before_direct3dcreate9_reaches_the_default_log",
        "--nocapture",
    ]);
    // The identity lines are info records, on both sides.
    command.envs([("RUST_LOG", "info"), ("__CX_UNIX_RUST_LOG", "info")]);
    let output = run_child(&mut command, "").expect("run the dying child");
    let stderr = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let logs: Vec<_> = std::fs::read_dir(dir.join("mtld3d-logs"))
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "log"))
                .collect()
        })
        .unwrap_or_default();
    let log = logs
        .first()
        .map(|path| std::fs::read_to_string(path).expect("read the child's log"));
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        output.status.code().map(i32::cast_unsigned),
        Some(EXCEPTION_ACCESS_VIOLATION),
        "the child ends with the exception's code; stderr:\n{stderr}"
    );
    assert_eq!(logs.len(), 1, "one log beside the child; stderr:\n{stderr}");
    let log = log.expect("the child's log");
    let identity = log
        .find("d3d9.dll v")
        .unwrap_or_else(|| panic!("no identity line:\n{log}"));
    let first_chance = log
        .find("fault outside d3d9.dll: code=0x00000000c0000005")
        .unwrap_or_else(|| panic!("no first-chance line:\n{log}"));
    let unhandled = log
        .find("unhandled exception: code=0x00000000c0000005")
        .unwrap_or_else(|| panic!("no unhandled-exception line:\n{log}"));
    assert!(identity < first_chance && first_chance < unhandled, "{log}");
    assert!(log.contains("reported above at first chance"), "{log}");
    assert_eq!(
        log.matches("fault outside d3d9.dll").count(),
        1,
        "the exception is reported once in full:\n{log}"
    );
}
