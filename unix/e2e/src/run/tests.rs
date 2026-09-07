//! The process deadlines: `/bin/sh` plays Wine, a script plays the test binary.

use std::{
    fs,
    io::{BufRead, BufReader, Read},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use super::{ExitKind, ProcessGroup, run};

const DRIVER_HANG: &str = "Caused GPU Hang Error \
    (00000003:kIOAccelCommandBufferCallbackErrorHang)";

/// A script in a fresh directory under the target dir, run as `sh <script>`.
fn script(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mtld3d-e2e-run-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{name}.sh"));
    fs::write(&path, body).expect("script");
    path
}

fn run_script(path: &Path, timeout: Duration) -> (super::Exit, Duration, Vec<String>) {
    let mut lines = Vec::new();
    let started = Instant::now();
    let exit = run(&PathBuf::from("/bin/sh"), path, &[], timeout, &mut |line| {
        lines.push(line.to_owned());
    })
    .expect("spawn sh");
    (exit, started.elapsed(), lines)
}

#[test]
fn a_clean_exit_reports_its_code_and_stderr() {
    let path = script("clean", "echo one\necho two >&2\nexit 3\n");
    let (exit, _, lines) = run_script(&path, Duration::from_secs(5));
    assert_eq!(exit.kind, ExitKind::Code(3));
    assert_eq!(lines, ["one"]);
    assert_eq!(exit.stderr, "two\n");
    assert!(!exit.gpu_hang);
}

#[test]
fn a_gpu_hang_split_across_stderr_writes_stops_the_process() {
    let path = script(
        "gpu-hang",
        "printf 'Caused GPU Hang Error (00000003:kIOAccelCommandBuffer' >&2\n\
         sleep 0.1\n\
         printf 'CallbackErrorHang)\\n' >&2\n\
         sleep 30\n",
    );
    let started = Instant::now();
    let (exit, elapsed, _) = run_script(&path, Duration::from_secs(5));
    assert!(exit.gpu_hang);
    assert_eq!(exit.stderr, format!("{DRIVER_HANG}\n"));
    assert!(
        elapsed < Duration::from_secs(2),
        "took {elapsed:?}: the runner waited instead of stopping the process"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn a_gpu_hang_racing_with_a_clean_exit_still_counts() {
    let path = script(
        "gpu-hang-clean",
        &format!("printf '%s\\n' '{DRIVER_HANG}' >&2\nexit 0\n"),
    );
    let (exit, _, _) = run_script(&path, Duration::from_secs(5));
    assert!(exit.gpu_hang);
    assert_eq!(exit.stderr, format!("{DRIVER_HANG}\n"));
}

#[test]
fn unrelated_gpu_errors_are_not_hang_reports() {
    for stderr in [
        "0000:err:d3d9: GPU hang is not what this line reports",
        "command buffer failed with Internal Error",
        "kIOAccelCommandBufferCallbackErrorNotPermitted",
    ] {
        assert!(!super::is_gpu_hang_report(stderr), "{stderr}");
    }
}

#[test]
fn silence_on_stdout_is_killed_after_the_timeout() {
    let path = script("silent", "echo start\nsleep 30\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::TimedOut(timeout));
    assert_eq!(lines, ["start"]);
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn a_process_that_closes_stdout_and_never_exits_is_killed_and_reported_hung() {
    let path = script("hung", "echo last\nexec 1>&-\nexec 2>&-\nsleep 30\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::Hung(timeout));
    assert_eq!(lines, ["last"]);
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn a_killed_process_whose_survivor_holds_stderr_costs_the_grace_not_the_timeout() {
    // The script goes silent, so the watchdog kills its group; the perl
    // child moved itself to a group of its own first and keeps stderr open
    // past the kill, as Wine's debugger does after a crash.
    let path = script(
        "escaped",
        "echo start\necho early >&2\n(perl -e 'setpgrp(0,0); sleep 30' >/dev/null) &\nsleep 30\n",
    );
    let timeout = Duration::from_secs(2);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::TimedOut(timeout));
    assert_eq!(lines, ["start"]);
    assert!(
        exit.stderr.starts_with("early\n"),
        "stderr: {:?}",
        exit.stderr
    );
    assert!(
        exit.stderr.contains("stderr not collected in full"),
        "stderr: {:?}",
        exit.stderr
    );
    // One timeout for the silence, the grace for stderr, and slack; the old
    // shape paid the timeout twice.
    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "took {elapsed:?}"
    );
}

#[test]
fn a_descendant_holding_stderr_does_not_park_the_run() {
    // The script exits at once; its background child keeps stderr open.
    let path = script("holder", "echo done\n(sleep 30 >/dev/null) &\nexit 0\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::Code(0));
    assert_eq!(lines, ["done"]);
    assert!(
        exit.stderr.contains("stderr not collected in full"),
        "stderr: {:?}",
        exit.stderr
    );
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn observing_a_killed_leader_stops_a_descendant_missed_by_the_first_signal() {
    let path = script("missed-child", "sleep 30 &\necho ready\nwait\n");
    let mut child = Command::new("/bin/sh")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().expect("stdout piped"))
        .read_line(&mut ready)
        .expect("read readiness");
    assert_eq!(ready, "ready\n");
    // The child exists before the parent is killed. Signalling only the
    // leader models a group signal whose snapshot missed the new child,
    // without requiring the signal to race a particular fork instruction.
    child.kill().expect("kill group leader");
    let pid = child.id();
    let mut stderr = child.stderr.take().expect("stderr piped");
    let mut group = ProcessGroup { child: Some(child) };
    group.wait_killed().expect("observe group leader");
    assert_waitable(pid);
    let (closed, eof) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        closed
            .send(stderr.read_to_end(&mut bytes))
            .expect("report EOF");
    });
    let collected = eof.recv_timeout(Duration::from_secs(1));
    // Clean up the fixture even when the assertion below fails.
    group.kill().expect("clean up group");
    reader.join().expect("stderr reader");
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert!(!status.success());
    assert!(
        collected.is_ok(),
        "descendant kept stderr open: {collected:?}"
    );
    assert_eq!(collected.expect("EOF delivered").expect("read stderr"), 0);
}

#[test]
fn observing_a_clean_exit_keeps_the_leader_waitable() {
    let child = Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup { child: Some(child) };
    let gpu_hang = std::sync::atomic::AtomicBool::new(false);
    let observed = super::wait_bounded(&mut group, Duration::from_secs(2), &gpu_hang)
        .expect("observe group leader");
    assert!(matches!(observed, super::Wait::Exited));
    assert_waitable(pid);
    assert!(
        group
            .observe_exit(libc::WNOHANG)
            .expect("observe exit again")
    );
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert_eq!(status.code(), Some(7));
    assert_reaped(pid);
}

#[test]
fn a_killed_leader_stays_waitable_until_the_final_signal() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup { child: Some(child) };
    assert!(
        !group
            .observe_exit(libc::WNOHANG)
            .expect("observe running child")
    );
    group.kill().expect("first group signal");
    group.wait_killed().expect("observe exit and signal again");
    assert_waitable(pid);
    assert!(
        group
            .observe_exit(libc::WNOHANG)
            .expect("observe exit again")
    );
    group.kill().expect("final group signal");
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert!(!status.success());
    assert_reaped(pid);
}

#[test]
fn losing_the_waitable_child_disables_group_signals() {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    // Model an external waiter consuming the child, without reusing any ID.
    child.wait().expect("external reap");
    let mut group = ProcessGroup { child: Some(child) };
    let error = group
        .kill()
        .expect_err("must not signal after external reap");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
    assert!(group.child.is_none());
    let error = group.kill().expect_err("ownership stays revoked");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
}

#[test]
fn dropping_an_unfinished_group_stops_and_reaps_its_leader() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    drop(ProcessGroup { child: Some(child) });
    assert_reaped(pid);
}

#[test]
fn an_unexpected_wait_error_retains_ownership_for_cleanup() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup { child: Some(child) };
    let error = group
        .observe_exit(libc::c_int::MAX)
        .expect_err("invalid wait options");
    assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
    assert!(group.child.is_some());
    drop(group);
    assert_reaped(pid);
}

fn assert_waitable(pid: u32) {
    let info = observe_fixture(pid).expect("exit observation must not reap the leader");
    assert_eq!(info.si_pid.unsigned_abs(), pid);
}

fn assert_reaped(pid: u32) {
    match observe_fixture(pid) {
        Ok(_) => panic!("final reap left a waitable child"),
        Err(error) => assert_eq!(error.raw_os_error(), Some(libc::ECHILD)),
    }
}

fn observe_fixture(pid: u32) -> std::io::Result<libc::siginfo_t> {
    // SAFETY: siginfo_t contains integers and raw pointers, all valid when zeroed.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: info is writable and P_PID selects only this fixture's direct child.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &raw mut info,
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    if result == 0 {
        Ok(info)
    } else {
        Err(std::io::Error::last_os_error())
    }
}
