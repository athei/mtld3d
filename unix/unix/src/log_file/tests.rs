//! Naming and retention of the log directory.
//!
//! `open` names the file after the host pid, the only process id that differs from one launch
//! to the next. `prune` keeps the newest `keep` entries of one extension by modification time
//! and removes the rest, whether file (a log) or directory (a trace bundle), and leaves the
//! other extension alone. The modification times are set explicitly so the order does not
//! depend on how fast the files were created.
//!
//! The crash-report tests run in a re-executed child, since the sink is process-wide: a terminal
//! report written before `OpenLog` opens the early location with the backlog ahead of it and
//! keeps it as the log, or takes the backlog to stderr when there is no early location. A
//! first-chance report opens nothing.

use std::{
    fs::{self, File},
    path::PathBuf,
    time::{Duration, SystemTime},
};

use super::{fall_back_to_stderr, open, prune};

/// Set in the re-executed child that writes a crash report before `OpenLog`; names the early dir.
const CRASH_SELFTEST_ENV: &str = "MTLD3D_LOG_CRASH_SELFTEST";

/// A line logged before the crash report, which must reach the same place first.
const BACKLOG_LINE: &str = "[selftest] logged before the location was named";

/// The crash report's line, as the PE side's exception handler sends it.
const REPORT_LINE: &str = "[selftest] fault outside d3d9.dll";

/// A fresh directory under the system temp dir, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("mtld3d-prune-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        Self(dir)
    }

    /// A file named `name` whose modification time is `age` seconds before now.
    fn file(&self, name: &str, age: u64) {
        let path = self.0.join(name);
        let file = File::create(&path).expect("scratch file");
        file.set_modified(SystemTime::now() - Duration::from_secs(age))
            .expect("set mtime");
    }

    /// A directory named `name` (a trace bundle) `age` seconds old.
    fn bundle(&self, name: &str, age: u64) {
        let path = self.0.join(name);
        fs::create_dir(&path).expect("scratch bundle");
        File::create(path.join("payload")).expect("bundle payload");
        File::open(&path)
            .expect("open bundle")
            .set_modified(SystemTime::now() - Duration::from_secs(age))
            .expect("set mtime");
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .expect("read scratch")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn keeps_the_newest_logs_and_removes_the_oldest() {
    let dir = Scratch::new("logs");
    for age in 1..=5 {
        dir.file(&format!("game-{age}.log"), age * 10);
    }
    prune(&dir.0, "log", 3);
    assert_eq!(dir.names(), ["game-1.log", "game-2.log", "game-3.log"]);
}

#[test]
fn leaves_the_other_extension_alone() {
    let dir = Scratch::new("mixed");
    dir.file("game-1.log", 30);
    dir.file("game-2.log", 20);
    dir.bundle("game-1-1.gputrace", 40);
    dir.bundle("game-1-2.gputrace", 10);
    prune(&dir.0, "log", 1);
    assert_eq!(
        dir.names(),
        ["game-1-1.gputrace", "game-1-2.gputrace", "game-2.log"]
    );
}

#[test]
fn removes_a_trace_bundle_whole() {
    let dir = Scratch::new("traces");
    dir.bundle("game-1-1.gputrace", 30);
    dir.bundle("game-1-2.gputrace", 20);
    dir.bundle("game-1-3.gputrace", 10);
    prune(&dir.0, "gputrace", 2);
    assert_eq!(dir.names(), ["game-1-2.gputrace", "game-1-3.gputrace"]);
}

#[test]
fn nothing_to_prune_below_the_cap() {
    let dir = Scratch::new("few");
    dir.file("game-1.log", 10);
    prune(&dir.0, "log", 3);
    assert_eq!(dir.names(), ["game-1.log"]);
}

#[test]
fn names_the_log_after_the_host_pid() {
    let dir = Scratch::new("name");
    let path = open(&dir.0.to_string_lossy(), "game");
    // Back to stderr before any line creates the file: the sink is process-wide.
    fall_back_to_stderr();
    let expected = dir.0.join(format!("game-{}.log", std::process::id()));
    assert_eq!(path, expected);
    assert!(
        dir.names().is_empty(),
        "naming the location creates nothing"
    );
}

/// Re-execute `test` with the crash child's variable set to `dir`; returns its pid and output.
fn crash_child(test: &str, dir: &str) -> (u32, std::process::Output) {
    let child = std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test, "--nocapture"])
        .env(CRASH_SELFTEST_ENV, dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("re-exec the test binary");
    let pid = child.id();
    (pid, child.wait_with_output().expect("wait for the child"))
}

/// A crash report through `WriteLog` before `OpenLog` opens the early location.
///
/// The backlog goes in ahead of the report, the directory is created on the
/// way, and the file stays the log when `OpenLog` then names another
/// directory: the lines after it land next to the report, not elsewhere.
#[test]
fn a_crash_report_before_the_location_opens_the_early_one_and_keeps_it() {
    const TEST: &str =
        "log_file::tests::a_crash_report_before_the_location_opens_the_early_one_and_keeps_it";
    if let Ok(dir) = std::env::var(CRASH_SELFTEST_ENV) {
        super::set_early_location(&format!("{dir}/early"), "game");
        super::write_all(format!("{BACKLOG_LINE}\n").as_bytes());
        super::write_crash(format!("{REPORT_LINE}\n").as_bytes());
        let named = super::open(&format!("{dir}/configured"), "game");
        super::write_all(format!("[selftest] named {}\n", named.display()).as_bytes());
        return;
    }

    let scratch = Scratch::new("crash-early");
    let (pid, out) = crash_child(TEST, &scratch.0.to_string_lossy());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let early = scratch.0.join("early").join(format!("game-{pid}.log"));
    let log = fs::read_to_string(&early).unwrap_or_else(|e| {
        panic!(
            "no early log at {} ({e}); stderr:\n{stderr}",
            early.display()
        )
    });
    let backlog = log.find(BACKLOG_LINE).expect("the backlog line");
    let report = log.find(REPORT_LINE).expect("the report line");
    assert!(backlog < report, "{log}");
    // `OpenLog` answered the file the lines really go to, and they went there.
    assert!(
        log.contains(&format!("[selftest] named {}", early.display())),
        "{log}"
    );
    assert!(
        !scratch.0.join("configured").exists(),
        "the configured directory stays unused"
    );
    assert!(!stderr.contains(REPORT_LINE), "{stderr}");
}

/// Without an early location, a crash report before `OpenLog` takes the backlog to stderr.
#[test]
fn a_crash_report_without_an_early_location_goes_to_stderr_with_the_backlog() {
    const TEST: &str =
        "log_file::tests::a_crash_report_without_an_early_location_goes_to_stderr_with_the_backlog";
    if std::env::var_os(CRASH_SELFTEST_ENV).is_some() {
        super::write_all(format!("{BACKLOG_LINE}\n").as_bytes());
        super::write_crash(format!("{REPORT_LINE}\n").as_bytes());
        return;
    }

    let (_, out) = crash_child(TEST, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let backlog = stderr
        .find(BACKLOG_LINE)
        .expect("the backlog line on stderr");
    let report = stderr.find(REPORT_LINE).expect("the report line on stderr");
    assert!(backlog < report, "{stderr}");
}

/// A first-chance fault report before `OpenLog` opens nothing.
///
/// The report goes where any line goes, the backlog, so no file exists while
/// the process lives, and none appears when it exits.
#[test]
fn a_fault_report_before_the_location_opens_nothing() {
    const TEST: &str = "log_file::tests::a_fault_report_before_the_location_opens_nothing";
    if let Ok(dir) = std::env::var(CRASH_SELFTEST_ENV) {
        super::set_early_location(&format!("{dir}/early"), "game");
        super::write_all(format!("{BACKLOG_LINE}\n").as_bytes());
        super::write_fault(format!("{REPORT_LINE}\n").as_bytes());
        assert!(
            !std::path::Path::new(&format!("{dir}/early")).exists(),
            "a first-chance report created the early location"
        );
        std::process::exit(0);
    }

    let scratch = Scratch::new("fault-first-chance");
    let (_, out) = crash_child(TEST, &scratch.0.to_string_lossy());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        !scratch.0.join("early").exists(),
        "a first-chance report created the early location"
    );
    assert!(!stderr.contains(REPORT_LINE), "{stderr}");
}
