//! Naming and retention of the log directory.
//!
//! `open` names the file after the host pid, the only process id that differs from one launch
//! to the next. `prune` keeps the newest `keep` entries of one extension by modification time
//! and removes the rest, whether file (a log) or directory (a trace bundle), and leaves the
//! other extension alone. The modification times are set explicitly so the order does not
//! depend on how fast the files were created.

use std::{
    fs::{self, File},
    path::PathBuf,
    time::{Duration, SystemTime},
};

use super::{fall_back_to_stderr, open, prune};

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
