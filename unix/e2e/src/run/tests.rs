//! The process deadlines: `/bin/sh` plays Wine, a script plays the test binary.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::{ExitKind, run};

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
