//! The startup lines reach the log of a process that ends right after `Direct3DCreate9`.
//!
//! A probe that only asks which d3d9 it got, such as an installer's or a
//! launcher's, calls `Direct3DCreate9` and ends at once. The layer's lines
//! from `DllMain` on, its build identity first, wait in `d3d9.dll`'s log queue
//! until `Direct3DCreate9`, and a process that ended before the logging
//! thread drained them lost them all. The child here ends through
//! `ExitProcess` the moment its interface exists, without releasing it, and
//! the parent reads the log it left.
//!
//! Each of the three startup lines also names the arch its half runs as:
//! `i686` for a 32-bit process, `x86_64` or, for the ARM64X build, `arm64ec`
//! for an x64 one, `aarch64` for a native ARM64 one, and the unix library its
//! own arch, followed by `, under Rosetta 2` when Rosetta translates it.

use mtld3d_tests::Harness;

use super::device::{running_as, spawn_private_log_child};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn ExitProcess(exit_code: u32) -> !;
}

/// The copy of the test executable that ends right after `Direct3DCreate9`.
const CHILD_NAME: &str = "exit-after-create.exe";

/// The arches the PE halves may name in a process of this test binary's arch.
const PE_ARCHES: &[&str] = if cfg!(target_arch = "x86") {
    &["i686"]
} else if cfg!(target_arch = "x86_64") {
    // The x86_64 build, or the ARM64X build's ARM64EC half on an arm64 Wine.
    &["x86_64", "arm64ec"]
} else {
    &["aarch64"]
};

/// The three startup lines are in the log of a process that ends at once.
#[test]
fn a_process_ending_right_after_direct3dcreate9_keeps_its_startup_lines() {
    if running_as(CHILD_NAME) {
        exit_after_create_workload();
        return;
    }
    let (output, dir) = spawn_private_log_child(
        CHILD_NAME,
        "startup_log::a_process_ending_right_after_direct3dcreate9_keeps_its_startup_lines",
        "info",
        "",
    );
    let log = private_log(&dir);
    std::fs::remove_dir_all(&dir).expect("remove the child's private directory");
    assert!(
        output.status.success(),
        "the child failed before it ended: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let d3d9 = find_line(&log, "] d3d9.dll ", " loaded at ");
    let shim = find_line(&log, "] mtld3d.dll ", ", unix call initialized as ");
    let unix = find_line(&log, "] mtld3d.so ", " initialized as ");
    let d3d9_arch = d3d9.rsplit_once(" as ").map_or("", |(_, arch)| arch);
    let shim_arch = shim.rsplit_once(" as ").map_or("", |(_, arch)| arch);
    assert!(
        PE_ARCHES.contains(&d3d9_arch),
        "d3d9.dll names {d3d9_arch:?}, not one of {PE_ARCHES:?}: {d3d9}"
    );
    assert_eq!(
        shim_arch, d3d9_arch,
        "mtld3d.dll runs as the same half as d3d9.dll: {shim}"
    );
    let (_, execution) = unix
        .split_once(" initialized as ")
        .expect("found by that text");
    assert!(
        ["aarch64", "x86_64", "x86_64, under Rosetta 2"].contains(&execution),
        "mtld3d.so names its arch, and Rosetta 2 when it runs under it: {unix}"
    );
}

/// Create the interface and end the process at once, with nothing released.
fn exit_after_create_workload() {
    let harness = Harness::factory_only();
    assert!(
        !harness.factory().is_null(),
        "Direct3DCreate9 returned null"
    );
    // SAFETY: plain kernel32 call ending the process; nothing runs after it.
    unsafe { ExitProcess(0) }
}

/// The first line of `log` holding `stamp` and then `after`; fails the test without one.
fn find_line<'a>(log: &'a str, stamp: &str, after: &str) -> &'a str {
    log.lines()
        .find(|line| {
            line.split_once(stamp)
                .is_some_and(|(_, rest)| rest.contains(after))
        })
        .unwrap_or_else(|| panic!("the log lacks a `{stamp}...{after}` line:\n{log}"))
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
