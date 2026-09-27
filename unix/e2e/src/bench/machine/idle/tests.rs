//! The idle gate's streak, deadline, unavailable samples and retained evidence.

use std::{fs, path::PathBuf, process::Command, time::Duration};

use super::{Decision, QuietStreak, wait_with};
use crate::bench::machine::{Sample, file_name};

fn quiet() -> Sample {
    Sample::parse("kernel_task 1.0\nforeign 2.0\n")
}

fn busy() -> Sample {
    Sample::parse("kernel_task 1.0\nforeign 51.0\n")
}

#[test]
fn busy_and_unavailable_samples_break_the_quiet_streak() {
    let mut streak = QuietStreak::new(Duration::from_secs(20));
    for (at, sample) in [
        Some(quiet()),
        Some(quiet()),
        Some(busy()),
        Some(quiet()),
        None,
        Some(quiet()),
        Some(quiet()),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(matches!(
            streak.observe(Duration::from_secs(at as u64), sample.as_ref()),
            Decision::Waiting(_)
        ));
    }
    assert!(matches!(
        streak.observe(Duration::from_secs(8), Some(&quiet())),
        Decision::Ready
    ));
}

#[test]
fn a_third_quiet_sample_must_finish_before_the_deadline() {
    for last in [Duration::from_secs(2), Duration::from_millis(2001)] {
        let mut streak = QuietStreak::new(Duration::from_secs(2));
        for elapsed in [Duration::from_millis(500), Duration::from_secs(1)] {
            assert!(matches!(
                streak.observe(elapsed, Some(&quiet())),
                Decision::Waiting(_)
            ));
        }
        assert!(matches!(
            streak.observe(last, Some(&quiet())),
            Decision::TimedOut
        ));
    }
}

/// A unique test directory under the primary checkout's ignored evidence directory.
fn evidence(name: &str) -> PathBuf {
    let output = Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("resolve primary checkout");
    assert!(output.status.success());
    let common = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
    let root = common.parent().expect("git directory has a parent");
    root.join(".codex/evidence/bench-idle-gate")
        .join(format!("test-{}-{name}", std::process::id()))
}

#[test]
fn every_attempt_is_kept_and_the_machine_file_is_the_final_accepted_sample() {
    let root = evidence("ready");
    let mut samples = [
        Some(quiet()),
        Some(busy()),
        None,
        Some(quiet()),
        Some(quiet()),
        Some(quiet()),
    ]
    .into_iter()
    .enumerate();
    wait_with(&root, "host", Duration::from_secs(10), || {
        let (at, sample) = samples.next().expect("gate stopped at the accepted sample");
        (Duration::from_millis((at as u64 + 1) * 500), sample)
    })
    .unwrap();
    let attempts = root.join("idle-host");
    assert_eq!(fs::read_dir(&attempts).unwrap().count(), 6);
    let rejected = fs::read_to_string(attempts.join("sample-000002.txt")).unwrap();
    assert!(rejected.contains("status busy"));
    let unavailable = fs::read_to_string(attempts.join("sample-000003.txt")).unwrap();
    assert!(unavailable.contains("status unavailable"));
    let accepted = fs::read_to_string(root.join(file_name("host"))).unwrap();
    assert_eq!(Sample::parse(&accepted), quiet());
    let outcome = fs::read_to_string(root.join("idle-host.txt")).unwrap();
    assert!(outcome.contains("status ready\nattempts 6\nwait_ms 3000\n"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn timeout_keeps_its_samples_without_creating_a_machine_file_or_verdict() {
    let root = evidence("timeout");
    let mut number = 0;
    let error = wait_with(&root, "e2e", Duration::from_secs(2), || {
        number += 1;
        (Duration::from_millis(number * 500), None)
    })
    .unwrap_err();
    assert!(error.contains("no performance verdict"));
    assert_eq!(fs::read_dir(root.join("idle-e2e")).unwrap().count(), 4);
    assert!(!root.join(file_name("e2e")).exists());
    assert!(!root.join("report.txt").exists());
    let outcome = fs::read_to_string(root.join("idle-e2e.txt")).unwrap();
    assert!(outcome.contains("status timeout\nattempts 4\nwait_ms 2000\n"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn evidence_write_failure_prevents_a_ready_verdict() {
    let root = evidence("write-error");
    fs::create_dir_all(&root).unwrap();
    // A regular file cannot hold the attempts directory.
    let blocker = root.join("not-a-directory");
    fs::write(&blocker, "existing evidence\n").unwrap();
    let error = wait_with(&blocker, "host", Duration::from_secs(10), || {
        panic!("no sample may pass when its evidence cannot be written")
    })
    .unwrap_err();
    assert!(error.contains("idle-host"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn absent_kernel_measurement_is_explicit_in_idle_evidence() {
    let root = evidence("kernel-unavailable");
    let mut number = 0;
    wait_with(&root, "host", Duration::from_secs(10), || {
        number += 1;
        (
            Duration::from_millis(number * 500),
            Some(Sample::parse("foreign 2.0\n")),
        )
    })
    .unwrap();
    let attempt = fs::read_to_string(root.join("idle-host/sample-000003.txt")).unwrap();
    assert!(attempt.contains("status quiet\nkernel_task unavailable\nforeign 2.0\n"));
    let accepted = fs::read_to_string(root.join(file_name("host"))).unwrap();
    assert_eq!(Sample::parse(&accepted).kernel_task, None);
    fs::remove_dir_all(root).unwrap();
}
