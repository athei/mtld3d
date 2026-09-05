//! Unit tests for the attribution loop, over a scripted launcher.
//!
//! Each script is what one process would print and how it would end; the
//! tests pin how many processes a binary costs and which test each ending
//! is charged to: a clean run costs one, a panic names its test and the
//! rest run again, a crash under several threads runs the in-flight set on
//! one thread to find its test, a hang is the test whose start line has no
//! outcome, a binary that dies before any test fails whole, and fail-fast
//! stops after the first failure with the rest reported unrun.

use std::{collections::VecDeque, time::Duration};

use super::{Launcher, ProcessEnd, Report, TestResult, Verdict, run_binary};
use crate::{libtest::Parser, run::ExitKind};

/// One scripted process: its stdout, stderr, and how it ends.
struct Script {
    stdout: &'static str,
    stderr: &'static str,
    kind: ExitKind,
}

struct Scripted {
    tests: Vec<&'static str>,
    scripts: VecDeque<Script>,
    /// `(names, threads)` of every process launched.
    launched: Vec<(Option<Vec<String>>, u32)>,
}

impl Scripted {
    fn new(tests: &[&'static str], scripts: Vec<Script>) -> Self {
        Self {
            tests: tests.to_vec(),
            scripts: scripts.into(),
            launched: Vec::new(),
        }
    }
}

impl Launcher for Scripted {
    fn run(
        &mut self,
        names: Option<&[String]>,
        threads: u32,
        on_event: &mut dyn FnMut(crate::libtest::Event),
    ) -> Result<ProcessEnd, String> {
        self.launched.push((names.map(<[String]>::to_vec), threads));
        let script = self.scripts.pop_front().expect("a script per process");
        let mut parser = Parser::default();
        for line in script.stdout.lines() {
            if let Some(event) = parser.line(line) {
                on_event(event);
            }
        }
        Ok(ProcessEnd {
            kind: script.kind,
            stdout: script.stdout.to_owned(),
            stderr: script.stderr.to_owned(),
        })
    }

    fn list(&mut self) -> Result<Vec<String>, String> {
        Ok(self.tests.iter().map(|t| (*t).to_owned()).collect())
    }
}

/// Collects results and notes.
#[derive(Default)]
struct Log {
    results: Vec<TestResult>,
    notes: Vec<String>,
}

impl Report for Log {
    fn result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    fn note(&mut self, note: &str) {
        self.notes.push(note.to_owned());
    }
}

fn collect(launcher: &mut Scripted, threads: u32, fail_fast: bool) -> (Vec<TestResult>, u32) {
    let mut log = Log::default();
    let run = run_binary(launcher, None, threads, fail_fast, &mut log).unwrap();
    (log.results, run.processes)
}

fn verdicts(results: &[TestResult]) -> Vec<(&str, &str)> {
    results
        .iter()
        .map(|r| {
            let verdict = match r.verdict {
                Verdict::Passed => "pass",
                Verdict::Failed(_) => "fail",
                Verdict::Ignored => "ignored",
                Verdict::NotRun => "not run",
            };
            (r.name.as_str(), verdict)
        })
        .collect()
}

#[test]
fn a_clean_run_costs_one_process_and_never_lists() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two", "b::three"],
        vec![Script {
            stdout: "running 3 tests\ntest a::one ... ok\ntest a::two ... ignored\ntest b::three ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
            stderr: "",
            kind: ExitKind::Code(0),
        }],
    );
    let (results, processes) = collect(&mut launcher, 4, true);
    assert_eq!(processes, 1);
    assert_eq!(
        verdicts(&results),
        [
            ("a::one", "pass"),
            ("a::two", "ignored"),
            ("b::three", "pass")
        ]
    );
    assert_eq!(launcher.launched, [(None, 4)]);
}

#[test]
fn a_panic_names_its_test_and_the_rest_run_again() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two", "b::three"],
        vec![
            Script {
                stdout: "running 3 tests\ntest a::one ... ok\n",
                stderr: "thread 'a::two' panicked at x.rs:1:1:\nassertion failed: it\n",
                kind: ExitKind::Code(101),
            },
            Script {
                stdout: "running 1 test\ntest b::three ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: "",
                kind: ExitKind::Code(0),
            },
        ],
    );
    let (results, processes) = collect(&mut launcher, 4, false);
    assert_eq!(processes, 2);
    assert_eq!(
        verdicts(&results),
        [("a::one", "pass"), ("a::two", "fail"), ("b::three", "pass")]
    );
    assert!(
        matches!(&results[1].verdict, Verdict::Failed(r) if r.contains("assertion failed: it"))
    );
    assert_eq!(
        launcher.launched[1],
        (Some(vec!["b::three".to_owned()]), 4),
        "only what was left runs again, still at full width"
    );
}

#[test]
fn an_unnamed_crash_under_threads_is_attributed_on_one_thread() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two", "b::three"],
        vec![
            Script {
                stdout: "running 3 tests\ntest a::one ... ok\n",
                stderr: "wine: Unhandled page fault\n",
                kind: ExitKind::Code(5),
            },
            Script {
                stdout: "running 2 tests\ntest a::two ... ok\ntest b::three ... ",
                stderr: "wine: Unhandled page fault\n",
                kind: ExitKind::Code(5),
            },
        ],
    );
    let (results, processes) = collect(&mut launcher, 4, false);
    assert_eq!(processes, 2);
    assert_eq!(
        verdicts(&results),
        [("a::one", "pass"), ("a::two", "pass"), ("b::three", "fail")]
    );
    assert_eq!(
        launcher.launched[1],
        (Some(vec!["a::two".to_owned(), "b::three".to_owned()]), 1)
    );
    assert!(matches!(&results[2].verdict, Verdict::Failed(r) if r.contains("exit code 5")));
}

#[test]
fn a_hang_is_the_test_whose_start_line_has_no_outcome() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![
            Script {
                stdout: "running 2 tests\ntest a::one ... ",
                stderr: "",
                kind: ExitKind::TimedOut(Duration::from_secs(5)),
            },
            Script {
                stdout: "running 1 test\ntest a::two ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: "",
                kind: ExitKind::Code(0),
            },
        ],
    );
    let (results, processes) = collect(&mut launcher, 1, false);
    assert_eq!(processes, 2);
    assert_eq!(verdicts(&results), [("a::one", "fail"), ("a::two", "pass")]);
    assert!(matches!(&results[0].verdict, Verdict::Failed(r) if r.contains("no output for 5 s")));
}

#[test]
fn a_binary_that_dies_before_any_test_fails_whole_without_a_retry_loop() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![
            Script {
                stdout: "",
                stderr: "wine: could not load d3d9.dll\n",
                kind: ExitKind::Code(1),
            },
            Script {
                stdout: "",
                stderr: "wine: could not load d3d9.dll\n",
                kind: ExitKind::Code(1),
            },
        ],
    );
    let (results, processes) = collect(&mut launcher, 4, false);
    assert_eq!(processes, 2, "once at width, once on one thread to be sure");
    assert_eq!(verdicts(&results), [("a::one", "fail"), ("a::two", "fail")]);
    assert!(
        matches!(&results[0].verdict, Verdict::Failed(r) if r.contains("before running any test"))
    );
}

#[test]
fn fail_fast_stops_after_the_first_failure_and_reports_the_rest_unrun() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two", "b::three"],
        vec![Script {
            stdout: "running 3 tests\ntest a::one ... ",
            stderr: "thread 'a::one' panicked at x.rs:1:1:\nno\n",
            kind: ExitKind::Code(101),
        }],
    );
    let (results, processes) = collect(&mut launcher, 1, true);
    assert_eq!(processes, 1);
    assert_eq!(
        verdicts(&results),
        [
            ("a::one", "fail"),
            ("a::two", "not run"),
            ("b::three", "not run")
        ]
    );
}

#[test]
fn a_failure_libtest_survived_is_read_from_its_report() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![Script {
            stdout: "running 2 tests\ntest a::one ... FAILED\ntest a::two ... ok\n\nfailures:\n\n---- a::one stdout ----\nleft != right\n\n\nfailures:\n    a::one\n\ntest result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
            stderr: "",
            kind: ExitKind::Code(101),
        }],
    );
    let (results, processes) = collect(&mut launcher, 1, false);
    assert_eq!(
        processes, 1,
        "everything was accounted for, nothing runs again"
    );
    assert_eq!(verdicts(&results), [("a::one", "fail"), ("a::two", "pass")]);
    assert!(matches!(&results[0].verdict, Verdict::Failed(r) if r == "left != right"));
}

#[test]
fn a_selected_name_the_binary_does_not_know_is_a_failure() {
    let mut launcher = Scripted::new(
        &["a::one"],
        vec![Script {
            stdout: "running 1 test\ntest a::one ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.1s\n",
            stderr: "",
            kind: ExitKind::Code(0),
        }],
    );
    let selection = Some(vec!["a::one".to_owned(), "a::gone".to_owned()]);
    let mut log = Log::default();
    let run = run_binary(&mut launcher, selection, 1, false, &mut log).unwrap();
    assert_eq!(run.processes, 1);
    assert_eq!(
        verdicts(&log.results),
        [("a::one", "pass"), ("a::gone", "fail")]
    );
}

#[test]
fn an_unclean_exit_after_a_full_tally_costs_no_list_and_no_process() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![Script {
            stdout: "running 2 tests\ntest a::one ... ok\ntest a::two ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
            stderr: "",
            kind: ExitKind::Code(3),
        }],
    );
    let mut log = Log::default();
    let run = run_binary(&mut launcher, None, 4, true, &mut log).unwrap();
    assert_eq!(run.processes, 1);
    assert!(!run.failed);
    assert_eq!(
        verdicts(&log.results),
        [("a::one", "pass"), ("a::two", "pass")]
    );
    assert_eq!(log.notes.len(), 1);
    assert!(log.notes[0].contains("exit code 3"), "{:?}", log.notes);
}
