//! Unit tests for the attribution loop, over a scripted launcher.
//!
//! Each script is what one process would print and how it would end; the
//! tests pin how many processes a binary costs and which test each ending
//! is charged to: a clean run costs one, a panic names its test and the
//! rest run again, a crash under several threads runs the in-flight set on
//! one thread to find its test, a hang is the test whose start line has no
//! outcome, a binary that dies before any test fails whole, fail-fast stops
//! after the first failure with the rest reported unrun, and a test that
//! declares the code it ends its process with passes only on that code,
//! and a dead process keeps its whole stderr in a file the note names.
//!
//! Two more pin what a death nothing on stderr accounts for still says: the
//! note names the tests that had named themselves and not finished, and
//! only those run again one at a time, and the layer's own log of the
//! process is quoted with them.

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};

use super::{Launcher, ProcessEnd, Report, TestResult, Verdict, run_binary};
use crate::{
    binary::{keep_stderr, layer_tail},
    libtest::Parser,
    run::ExitKind,
};

/// The pid every scripted process ends under, so a test knows the file's name.
const PID: u32 = 4242;

/// One scripted process: its stdout, stderr, the layer's log of it, and how it ends.
struct Script {
    stdout: &'static str,
    stderr: String,
    /// What the layer wrote into its own log file; `None` = it wrote none.
    layer: Option<&'static str>,
    kind: ExitKind,
}

struct Scripted {
    tests: Vec<&'static str>,
    scripts: VecDeque<Script>,
    /// `(names, threads)` of every process launched.
    launched: Vec<(Option<Vec<String>>, u32)>,
    /// A directory of this launcher's own, so tests running at once do not share one.
    log_dir: PathBuf,
    /// The layer log of the process that ended last, as the script gave it.
    layer: Option<&'static str>,
}

impl Scripted {
    fn new(tests: &[&'static str], scripts: Vec<Script>) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let log_dir = std::env::temp_dir().join(format!(
            "mtld3d-e2e-attr-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        Self {
            tests: tests.to_vec(),
            scripts: scripts.into(),
            launched: Vec::new(),
            log_dir,
            layer: None,
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
        self.layer = script.layer;
        let mut parser = Parser::default();
        for line in script.stdout.lines() {
            if let Some(event) = parser.line(line) {
                on_event(event);
            }
        }
        Ok(ProcessEnd {
            pid: PID,
            kind: script.kind,
            stdout: script.stdout.to_owned(),
            stderr: script.stderr,
        })
    }

    fn list(&mut self) -> Result<Vec<String>, String> {
        Ok(self.tests.iter().map(|t| (*t).to_owned()).collect())
    }

    fn keep_stderr(&self, pid: u32, stderr: &str) -> Result<PathBuf, String> {
        keep_stderr(&self.log_dir, "scripted", pid, stderr)
    }

    fn layer_log(&self, pid: u32) -> Result<(PathBuf, String), String> {
        let path = self.log_dir.join(format!("scripted-{pid}.log"));
        let log = self
            .layer
            .ok_or_else(|| format!("{}: no such file", path.display()))?;
        Ok((path, layer_tail(log)))
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
            stderr: String::new(),
            layer: None,
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
                stderr: "thread 'a::two' panicked at x.rs:1:1:\nassertion failed: it\n".to_owned(),
                layer: None,
                kind: ExitKind::Code(101),
            },
            Script {
                stdout: "running 1 test\ntest b::three ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: String::new(),
                layer: None,
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
                stderr: "wine: Unhandled page fault\n".to_owned(),
                layer: None,
                kind: ExitKind::Code(5),
            },
            Script {
                stdout: "running 2 tests\ntest a::two ... ok\ntest b::three ... ",
                stderr: "wine: Unhandled page fault\n".to_owned(),
                layer: None,
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
                stderr: String::new(),
                layer: None,
                kind: ExitKind::TimedOut(Duration::from_secs(5)),
            },
            Script {
                stdout: "running 1 test\ntest a::two ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: String::new(),
                layer: None,
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
                stderr: "wine: could not load d3d9.dll\n".to_owned(),
                layer: None,
                kind: ExitKind::Code(1),
            },
            Script {
                stdout: "",
                stderr: "wine: could not load d3d9.dll\n".to_owned(),
                layer: None,
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
            stderr: "thread 'a::one' panicked at x.rs:1:1:\nno\n".to_owned(),
            layer: None,
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
            stderr: String::new(),
            layer: None,
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
            stderr: String::new(),
            layer: None,
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
            stderr: String::new(),
            layer: None,
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

#[test]
fn a_declared_exit_code_the_process_ends_with_passes_its_test() {
    let mut launcher = Scripted::new(
        &["a::ends"],
        vec![Script {
            stdout: "running 1 test\ntest a::ends ... \n[e2e] test a::ends ends this process with exit code 42\n",
            stderr: String::new(),
            layer: None,
            kind: ExitKind::Code(42),
        }],
    );
    let mut log = Log::default();
    let run = run_binary(&mut launcher, None, 1, true, &mut log).unwrap();
    assert_eq!(run.processes, 1);
    assert!(!run.failed);
    assert_eq!(verdicts(&log.results), [("a::ends", "pass")]);
    assert!(log.notes.is_empty(), "{:?}", log.notes);
}

#[test]
fn a_declared_exit_code_the_process_misses_fails_its_test() {
    let mut launcher = Scripted::new(
        &["a::ends"],
        vec![Script {
            stdout: "running 1 test\ntest a::ends ... [e2e] test a::ends ends this process with exit code 42\n",
            stderr: String::new(),
            layer: None,
            kind: ExitKind::Code(0),
        }],
    );
    let mut log = Log::default();
    let run = run_binary(&mut launcher, None, 1, true, &mut log).unwrap();
    assert_eq!(run.processes, 1);
    assert!(run.failed);
    assert_eq!(verdicts(&log.results), [("a::ends", "fail")]);
}

#[test]
fn a_dead_process_keeps_its_whole_stderr_and_the_note_names_the_file() {
    let chatter: String = (0..40)
        .map(|i| format!("fixme:dbghelp:elf_search_auxv can't find symbol {i}\n"))
        .collect::<Vec<_>>()
        .concat();
    let stderr = format!("IOSurfaceClientCreate failed\n{chatter}wine = 929;\n");
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![Script {
            stdout: "running 2 tests\ntest a::one ... ok\n",
            stderr: stderr.clone(),
            layer: None,
            kind: ExitKind::Signal(11),
        }],
    );
    let mut log = Log::default();
    let run = run_binary(&mut launcher, None, 1, true, &mut log).unwrap();
    assert_eq!(run.processes, 1);
    let path = launcher.log_dir.join(format!("scripted-{PID}.stderr"));
    let note = log.notes.first().expect("a note about the dead process");
    assert!(note.contains(&path.display().to_string()), "{note}");
    assert!(note.contains("IOSurfaceClientCreate failed"), "{note}");
    assert!(note.contains("wine = 929;"), "{note}");
    assert!(!note.contains("fixme:dbghelp"), "{note}");
    assert_eq!(
        std::fs::read_to_string(&path).expect("the kept stderr"),
        stderr
    );
}

#[test]
fn the_note_names_what_was_in_flight_and_only_those_run_one_at_a_time() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two", "a::three", "a::four"],
        vec![
            Script {
                stdout: "running 4 tests\n[e2e] running a::one\n[e2e] running a::two\n[e2e] running a::three\ntest a::one ... ok\n",
                stderr: String::new(),
                layer: None,
                kind: ExitKind::Code(1),
            },
            Script {
                stdout: "running 2 tests\ntest a::two ... ok\ntest a::three ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: String::new(),
                layer: None,
                kind: ExitKind::Code(0),
            },
            Script {
                stdout: "running 1 test\ntest a::four ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: String::new(),
                layer: None,
                kind: ExitKind::Code(0),
            },
        ],
    );
    let mut log = Log::default();
    let run = run_binary(&mut launcher, None, 4, false, &mut log).unwrap();
    assert_eq!(run.processes, 3);
    assert!(!run.failed);
    assert!(
        log.notes[0].contains("in flight when it ended: a::two, a::three"),
        "{:?}",
        log.notes
    );
    assert_eq!(
        launcher.launched[1],
        (Some(vec!["a::two".to_owned(), "a::three".to_owned()]), 1),
        "only the tests that were running go one at a time"
    );
    assert_eq!(
        launcher.launched[2],
        (Some(vec!["a::four".to_owned()]), 4),
        "the tests the narrowed round set aside run at the caller's width"
    );
}

#[test]
fn a_death_stderr_is_silent_about_is_quoted_from_the_layers_own_log() {
    let mut launcher = Scripted::new(
        &["a::one", "a::two"],
        vec![
            Script {
                stdout: "running 2 tests\n[e2e] running a::one\n[e2e] running a::two\n",
                stderr: String::new(),
                layer: Some(
                    "ordinary line\n[mtld3d::unix] FATAL: SIGSEGV fault=0x0\n[mtld3d::unix] thread=mtld3d-encoder\n",
                ),
                kind: ExitKind::Code(1),
            },
            Script {
                stdout: "running 2 tests\ntest a::one ... ok\ntest a::two ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n",
                stderr: String::new(),
                layer: None,
                kind: ExitKind::Code(0),
            },
        ],
    );
    let mut log = Log::default();
    run_binary(&mut launcher, None, 2, false, &mut log).unwrap();
    let note = &log.notes[0];
    assert!(note.contains("exit code 1"), "{note}");
    assert!(note.contains("[mtld3d::unix] FATAL: SIGSEGV"), "{note}");
    assert!(note.contains("thread=mtld3d-encoder"), "{note}");
    assert!(
        note.contains(
            &launcher
                .log_dir
                .join(format!("scripted-{PID}.log"))
                .display()
                .to_string()
        ),
        "{note}"
    );
    assert!(!note.contains("ordinary line"), "{note}");
}
