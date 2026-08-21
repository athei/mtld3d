//! Unit tests for the subtest output scanner.
//!
//! Recorded fixtures and hand-written snippets pin the counting rules: repeats
//! at one location accumulate, `flaky` and `todo` markers are tallied apart from
//! real failures, and a panic keeps its location and message. Crash detection
//! gets its own cases: a signal kill, a host-side abort with no failures and a
//! missing end-of-run summary all count, but a marker quoted in a failure does not.

use super::parse_subtest_output;
use crate::model::Site;

fn site(file: &str, line: u32) -> Site {
    Site {
        file: file.to_owned(),
        line,
    }
}

#[test]
fn device_fixture_counts_looped_and_distinct_sites_and_crash() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/device.txt"), false);
    assert!(out.crash, "the Unhandled exception line marks a crash");
    assert_eq!(out.sites.get(&site("device.c", 125)), Some(&3));
    assert_eq!(out.sites.get(&site("device.c", 792)), Some(&1));
    assert_eq!(out.sites.get(&site("device.c", 816)), Some(&1));
    assert_eq!(out.sites.len(), 3);
}

#[test]
fn visual_fixture_multiple_sites_and_crash() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/visual.txt"), false);
    assert!(out.crash);
    assert_eq!(out.sites.len(), 3);
    assert_eq!(out.sites.get(&site("visual.c", 9100)), Some(&1));
}

#[test]
fn stateblock_fixture_zero_failures_but_objc_abort_is_a_crash() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/stateblock.txt"), false);
    assert!(out.sites.is_empty());
    assert!(
        out.crash,
        "libc++abi/NSException abort is a crash even with zero failures"
    );
}

#[test]
fn d3d9ex_fixture_one_failure_no_crash() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/d3d9ex.txt"), false);
    assert_eq!(out.sites.get(&site("d3d9ex.c", 55)), Some(&1));
    assert!(!out.crash);
}

#[test]
fn clean_fixture_no_failures_no_crash() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/clean.txt"), false);
    assert!(out.sites.is_empty());
    assert!(!out.crash);
}

#[test]
fn signal_death_marks_crash_even_when_output_is_clean() {
    let out = parse_subtest_output(include_str!("../../tests/fixtures/clean.txt"), true);
    assert!(out.crash);
}

#[test]
fn rust_panic_is_captured_with_location_and_message() {
    let out = parse_subtest_output(
        "visual.c:5299: Test failed: Got hr 0x8876086c.\n\
         [mtld3d::d3d9] PANIC - dumping crumb trail:\n\
         thread '<unnamed>' (1712) panicked at d3d9/src/device.rs:1080:22:\n\
         misaligned pointer dereference: address must be a multiple of 0x8 but is 0x2340001\n\
         [mtld3d::unix] FATAL: SIGSEGV fault=0x0\n",
        false,
    );
    assert!(out.crash, "a panic is always a crash");
    let panic = out.panic.expect("panic location captured");
    assert_eq!(
        panic,
        "panicked at d3d9/src/device.rs:1080:22: — misaligned pointer dereference: \
         address must be a multiple of 0x8 but is 0x2340001"
    );
    // The failure site before the panic is still counted.
    assert_eq!(out.sites.get(&site("visual.c", 5299)), Some(&1));
}

#[test]
fn flaky_and_todo_marked_lines_are_tallied_separately_not_as_failures() {
    let out = parse_subtest_output(
        "device.c:5368: Test failed: Got 1.\n\
         device.c:5406: Test marked flaky: Didn't receive MOUSEMOVE 7 (0, 0).\n\
         device.c:5406: Test marked flaky: Didn't receive MOUSEMOVE 7 (0, 0).\n\
         visual.c:15668: Test marked todo: Got unexpected colour 0x00fefe00.\n\
         0104:device: 9 tests executed (1 marked as todo, 0 as flaky, 1 failures), 0 skipped.\n",
        false,
    );
    // The real failure is counted; neither marked line inflates `sites`.
    assert_eq!(out.sites.get(&site("device.c", 5368)), Some(&1));
    assert_eq!(out.sites.len(), 1);
    assert!(!out.crash);
    // The marked lines land in their own maps for report visibility.
    assert_eq!(out.flaky_marked.get(&site("device.c", 5406)), Some(&2));
    assert_eq!(out.todo_marked.get(&site("visual.c", 15668)), Some(&1));
}

#[test]
fn output_without_the_end_of_run_summary_is_a_crash() {
    // A process that dies of an unhandled Win32 exception exits without a
    // signal and prints nothing under `WINEDEBUG=-all`; the only trace is
    // the missing summary line.
    let out = parse_subtest_output(
        "device.c:6210: Test failed: Failed to get volume container, hr 0x80004002.\n",
        false,
    );
    assert!(out.crash, "no summary line means the run was cut short");
    assert_eq!(out.sites.get(&site("device.c", 6210)), Some(&1));
    let out = parse_subtest_output(
        "device.c:6210: Test failed: Failed to get volume container, hr 0x80004002.\n\
         03e4:device: 159651 tests executed (102 marked as todo, 1 as flaky, 1 failures), 23 skipped.\n",
        false,
    );
    assert!(
        !out.crash,
        "the summary line marks a run that reached its end"
    );
}

#[test]
fn crash_marker_inside_a_failure_message_does_not_false_positive() {
    let line = "device.c:1: Test failed: expected no SIGSEGV but the handler saw one\n\
                0104:device: 1 tests executed (0 marked as todo, 0 as flaky, 1 failures), 0 skipped.\n";
    let out = parse_subtest_output(line, false);
    assert_eq!(out.sites.get(&site("device.c", 1)), Some(&1));
    assert!(
        !out.crash,
        "a marker inside a Test failed message must not set the crash bit"
    );
}
