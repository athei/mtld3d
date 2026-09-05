//! Run one binary's tests to completion and attribute what its processes report.
//!
//! One process runs the whole selection at first. A process that ends with
//! tests unaccounted for (the harness's panic hook ends it at the first
//! failed assertion, a crash takes it down, the watchdog kills a hang)
//! costs one result and one more process: the test the end is attributed
//! to is marked failed, and the rest run again from a fresh process. When
//! several tests were in flight and nothing names the culprit, the in-flight
//! set runs once more on one thread, where libtest prints each test's name
//! before running it. Every round strictly shrinks what is left, so the loop
//! ends.

use std::collections::BTreeSet;

use crate::{
    binary::stderr_tail,
    libtest::{self, Event, Outcome, Summary},
    run::ExitKind,
};

/// How a process of the binary ended, with everything it printed.
pub struct ProcessEnd {
    pub kind: ExitKind,
    pub stdout: String,
    pub stderr: String,
}

/// Runs the processes of one test binary.
pub trait Launcher {
    /// Run `names` (`None` = every test) on `threads` test threads, streaming stdout events.
    ///
    /// # Errors
    ///
    /// Returns a message when the process cannot be spawned.
    fn run(
        &mut self,
        names: Option<&[String]>,
        threads: u32,
        on_event: &mut dyn FnMut(Event),
    ) -> Result<ProcessEnd, String>;

    /// Every test the binary carries, in libtest's order.
    ///
    /// # Errors
    ///
    /// Returns a message when the binary cannot list itself.
    fn list(&mut self) -> Result<Vec<String>, String>;
}

/// What became of one test.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Passed,
    /// With the report: the panic message, libtest's failure block, or how the process ended.
    Failed(String),
    Ignored,
    /// Left unrun after an earlier failure stopped the run.
    NotRun,
}

/// One test's result, by its libtest path.
#[derive(Debug, PartialEq, Eq)]
pub struct TestResult {
    pub name: String,
    pub verdict: Verdict,
}

/// What running a binary cost and whether anything in it failed.
pub struct BinaryRun {
    pub processes: u32,
    pub failed: bool,
}

/// Where the results and the notes about a binary's processes go.
pub trait Report {
    /// One test's verdict.
    fn result(&mut self, result: TestResult);
    /// Something about a process worth a line in the run's output.
    ///
    /// How a process ended when that was not clean, and what the runner did
    /// about it.
    fn note(&mut self, note: &str);
}

/// One process's worth of results and the state the attribution needs.
struct Round {
    finished: Vec<(String, Outcome)>,
    started: Option<String>,
    summary: Option<Summary>,
}

/// Run `selection` (`None` = the whole binary) on `threads`, handing every result to `report`.
///
/// With `fail_fast`, the first failure ends the binary's run and the tests
/// that were left are reported [`Verdict::NotRun`].
///
/// # Errors
///
/// Returns a message when a process cannot be spawned, or the binary cannot
/// list its tests after a process ended with some unaccounted for.
pub fn run_binary(
    launcher: &mut dyn Launcher,
    selection: Option<Vec<String>>,
    threads: u32,
    fail_fast: bool,
    report: &mut dyn Report,
) -> Result<BinaryRun, String> {
    let mut remaining = selection;
    let mut threads = threads;
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut run = BinaryRun {
        processes: 0,
        failed: false,
    };
    loop {
        if remaining.as_ref().is_some_and(Vec::is_empty) {
            break;
        }
        run.processes += 1;
        let mut round = Round {
            finished: Vec::new(),
            started: None,
            summary: None,
        };
        let end = launcher.run(remaining.as_deref(), threads, &mut |event| match event {
            Event::Started(name) => round.started = Some(name),
            Event::Finished { name, outcome } => {
                round.started = None;
                round.finished.push((name, outcome));
            }
            Event::Summary(summary) => round.summary = Some(summary),
        })?;
        let ran_something = !round.finished.is_empty();
        let reported = round.finished.len();
        for (name, outcome) in round.finished {
            let verdict = match outcome {
                Outcome::Ok => Verdict::Passed,
                Outcome::Ignored => Verdict::Ignored,
                Outcome::Failed => {
                    run.failed = true;
                    Verdict::Failed(
                        libtest::failure_report(&end.stdout, &name)
                            .unwrap_or_else(|| "libtest reported FAILED".to_owned()),
                    )
                }
            };
            done.insert(name.clone());
            report.result(TestResult { name, verdict });
        }

        let clean = end.kind == ExitKind::Code(0) && round.summary.is_some();
        let complete = remaining
            .as_ref()
            .is_none_or(|names| names.iter().all(|name| done.contains(name)));
        if clean && complete {
            break;
        }
        if clean {
            // libtest ran to the end without these: they are not tests it knows.
            for name in remaining.take().into_iter().flatten() {
                if done.insert(name.clone()) {
                    run.failed = true;
                    report.result(TestResult {
                        name,
                        verdict: Verdict::Failed(
                            "the binary ran to completion without running this test".to_owned(),
                        ),
                    });
                }
            }
            break;
        }
        let reason = end.kind.describe();
        // libtest's own tally, when it got that far, says whether anything
        // was still in flight without a `--list`.
        let tallied = round.summary.as_ref().is_some_and(|s| {
            s.passed + s.failed + s.ignored == u32::try_from(reported).unwrap_or(u32::MAX)
        });
        if tallied && complete {
            report.note(&format!(
                "the process ended with {reason} after reporting every test; nothing to run again"
            ));
            break;
        }

        // The process ended with tests unaccounted for: which were in flight?
        let listed = match remaining.take() {
            Some(names) => names,
            None => launcher.list()?,
        };
        let mut in_flight: Vec<String> = listed
            .into_iter()
            .filter(|name| !done.contains(name))
            .collect();
        if in_flight.is_empty() {
            break;
        }
        report.note(&format!(
            "the process ended with {reason}; {} of its tests unaccounted for; its last lines:\n{}",
            in_flight.len(),
            stderr_tail(&end.stderr)
        ));
        let named = libtest::panicked_tests(&end.stderr)
            .into_iter()
            .find(|name| in_flight.contains(name));
        if named.is_none() && threads == 1 && !ran_something && round.started.is_none() {
            // One thread, nothing ran, nothing names a test: the binary itself
            // is broken, and running it again would only say so again.
            run.failed = true;
            let detail = format!(
                "the process ended ({reason}) before running any test\n{}",
                stderr_tail(&end.stderr)
            );
            for name in in_flight {
                done.insert(name.clone());
                report.result(TestResult {
                    name,
                    verdict: Verdict::Failed(detail.clone()),
                });
            }
            break;
        }
        let victim = named.clone().or_else(|| {
            (threads == 1 || in_flight.len() == 1).then(|| {
                round
                    .started
                    .filter(|name| in_flight.contains(name))
                    .unwrap_or_else(|| in_flight[0].clone())
            })
        });
        if let Some(name) = victim {
            let detail = named
                .as_ref()
                .and_then(|_| libtest::panic_report(&end.stderr, &name))
                .unwrap_or_else(|| {
                    format!(
                        "the process ended ({reason}) while this test ran\n{}",
                        stderr_tail(&end.stderr)
                    )
                });
            in_flight.retain(|other| *other != name);
            done.insert(name.clone());
            run.failed = true;
            report.result(TestResult {
                name,
                verdict: Verdict::Failed(detail),
            });
        } else {
            // Several tests were in flight and nothing names one: run them one
            // at a time, where the start line does.
            threads = 1;
            report.note("nothing names the test it ended in; running those again one at a time");
        }
        if fail_fast && run.failed {
            for name in in_flight {
                report.result(TestResult {
                    name,
                    verdict: Verdict::NotRun,
                });
            }
            break;
        }
        if !in_flight.is_empty() {
            report.note(&format!(
                "running the {} tests left in a fresh process",
                in_flight.len()
            ));
        }
        remaining = Some(in_flight);
    }
    Ok(run)
}

#[cfg(test)]
mod tests;
