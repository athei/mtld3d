//! Unit tests for the libtest output reader.
//!
//! The lines are what a test binary prints under Wine: the threaded and the
//! single-threaded forms of a result, a start line a dying process leaves
//! behind, a test's own output landing inside a line, the summary counts,
//! and the panic report that names the failing test.

use super::{
    Event, Outcome, Parser, Summary, failure_report, listed_tests, panic_report, panicked_test,
    panicked_tests,
};

fn finished(name: &str, outcome: Outcome) -> Event {
    Event::Finished {
        name: name.to_owned(),
        outcome,
    }
}

#[test]
fn a_whole_result_line_finishes_a_test() {
    let mut parser = Parser::default();
    assert_eq!(
        parser.line("test textures::lock_a8r8g8b8 ... ok"),
        Some(finished("textures::lock_a8r8g8b8", Outcome::Ok))
    );
    assert_eq!(
        parser.line("test device::reset ... FAILED"),
        Some(finished("device::reset", Outcome::Failed))
    );
    assert_eq!(
        parser.line("test msaa::resolve ... ignored, no 4x"),
        Some(finished("msaa::resolve", Outcome::Ignored))
    );
    assert_eq!(parser.line("running 3 tests"), None);
    assert_eq!(parser.line(""), None);
}

#[test]
fn a_start_line_is_closed_by_the_next_bare_outcome() {
    let mut parser = Parser::default();
    assert_eq!(
        parser.line("test draw::quad ... "),
        Some(Event::Started("draw::quad".to_owned()))
    );
    assert_eq!(parser.line("ok"), Some(finished("draw::quad", Outcome::Ok)));
    assert_eq!(parser.line("ok"), None, "nothing pending any more");
}

#[test]
fn a_print_inside_a_start_line_still_starts_the_test() {
    let mut parser = Parser::default();
    assert_eq!(
        parser.line("test draw::quad ... device created"),
        Some(Event::Started("draw::quad".to_owned()))
    );
    assert_eq!(
        parser.line("FAILED"),
        Some(finished("draw::quad", Outcome::Failed))
    );
}

#[test]
fn the_summary_counts_are_read() {
    let mut parser = Parser::default();
    let line = "test result: FAILED. 3 passed; 1 failed; 2 ignored; 0 measured; 5 filtered out; finished in 0.23s";
    assert_eq!(
        parser.line(line),
        Some(Event::Summary(Summary {
            passed: 3,
            failed: 1,
            ignored: 2,
            filtered_out: 5,
        }))
    );
}

#[test]
fn the_panic_report_names_the_test_thread() {
    let stderr = "0024:fixme:d3d9:something\n\
        thread 'device::reset_bad_dims' panicked at tests/e2e/device.rs:70:5:\n\
        assertion `left == right` failed\n\
        \x20 left: 0\n\
        \x20right: 1\n\
        note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n\
        wine: something else\n";
    assert_eq!(
        panicked_test("thread 'device::reset_bad_dims' panicked at x.rs:1:1:"),
        Some("device::reset_bad_dims")
    );
    assert_eq!(
        panicked_test("thread 'device::reset_bad_dims' (756) panicked at x.rs:1:1:"),
        Some("device::reset_bad_dims"),
        "the thread id the hook prints between name and verb"
    );
    assert_eq!(panicked_test("thread 'main' panicked at x.rs:1:1:"), None);
    assert_eq!(panicked_test("thread 'x' something else"), None);
    assert_eq!(panicked_tests(stderr), ["device::reset_bad_dims"]);
    let report = panic_report(stderr, "device::reset_bad_dims").unwrap();
    assert!(report.starts_with("thread 'device::reset_bad_dims' panicked"));
    assert!(report.ends_with("right: 1"), "{report}");
    assert_eq!(panic_report(stderr, "other"), None);
}

#[test]
fn the_failures_section_is_read_per_test() {
    let stdout = "test a ... FAILED\ntest b ... FAILED\n\nfailures:\n\n\
        ---- a stdout ----\nfirst message\nsecond line\n\n\
        ---- b stdout ----\nother\n\n\nfailures:\n    a\n    b\n";
    assert_eq!(
        failure_report(stdout, "a").as_deref(),
        Some("first message\nsecond line")
    );
    assert_eq!(failure_report(stdout, "b").as_deref(), Some("other"));
    assert_eq!(failure_report(stdout, "c"), None);
}

#[test]
fn the_list_output_yields_the_names() {
    assert_eq!(
        listed_tests("a::b: test\nc: test\n\n2 tests, 0 benchmarks\n"),
        ["a::b", "c"]
    );
}
