//! The process deadlines: `/bin/sh` plays Wine, a script plays the test binary.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::{ExitKind, run};

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
