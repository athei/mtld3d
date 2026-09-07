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
//!
//! What was in flight is not libtest's to say once more than one test runs
//! at a time: it names a test only when the test finishes. So a test names
//! itself when it starts (see [`announced`]), and a process that ends with
//! several unaccounted for is narrowed to the tests that had named
//! themselves and not finished. Those run one at a time, where a start line
//! names the culprit; the rest wait for a process of their own at the width
//! the caller asked for, since nothing so far says they were involved. The
//! announcements are on stderr, where they cannot land inside a result
//! line libtest is writing to stdout.
//!
//! A process that ends cleanly is still checked against libtest's own
//! account. Its `test result:` line counts the results it printed, and a
//! tally short of that count means a result never reached the runner
//! whatever the reason. The tests with no outcome are then named, out of
//! the selection or out of `--list`, and run again in a fresh process, so
//! the run reports on every test it was asked for rather than on a smaller
//! suite.
//!
//! The layer keeps an account of its own. Its log file, not the process's
//! stderr, is where its crash report goes, so a note about a process that
//! ended unaccounted for quotes both, and the runner moves that log out of
//! the layer's retention (the newest ten logs, which the runs that follow
//! soon exceed) so the account is still there when someone reads the note.
//!
//! One kind of test ends its process on purpose: it declares its name and
//! the code it is about to exit with on stdout (see [`declared_exit`]), and
//! the exit code is then the whole assertion, since libtest never gets to
//! report a result. A binary that carries such a test carries nothing else.

use std::{collections::BTreeSet, mem, path::PathBuf};

use crate::{
    binary::{LayerLog, stderr_tail},
    libtest::{self, Event, Outcome, Summary},
    run::ExitKind,
};

/// The stdout marker of a test that ends the process it runs in.
///
/// What follows it is the test's name, then [`ENDS_PROCESS_CODE`] and the
/// exit code. libtest leaves its own `test <name> ... ` line open while a
/// test runs, so the marker can land in the middle of a line and is searched
/// for rather than matched at the start.
const ENDS_PROCESS: &str = "[e2e] test ";
/// What separates the test's name from its exit code in the marker.
const ENDS_PROCESS_CODE: &str = " ends this process with exit code ";

/// The stderr marker of a test that has started; what follows it is its name.
///
/// Printed by the test binary's shared harness on the thread libtest named
/// after the test, and on stderr so that it can never land inside the
/// result line libtest is writing to stdout. Searched for rather than
/// matched at the start, since Wine's own chatter shares the stream.
const RUNNING: &str = "[e2e] running ";

/// How a process of the binary ended, with everything it printed.
pub struct ProcessEnd {
    pub pid: u32,
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

    /// Keep the whole stderr of the process `pid`, and name the file it went to.
    ///
    /// A report shows only [`stderr_tail`], and a failure that starts
    /// outside the test binary prints what it was long before that tail:
    /// the file is where that first line survives.
    ///
    /// # Errors
    ///
    /// Returns the reason when the file cannot be written.
    fn keep_stderr(&self, pid: u32, stderr: &str) -> Result<PathBuf, String>;

    /// Keep the layer's own log of the process `pid`, and read its account of the end.
    ///
    /// The layer writes every line, its crash report included, into this
    /// file and never to the pipes the runner reads, so a process the layer
    /// ended says nothing on stderr and the file is the only account of it.
    /// The layer also removes all but its newest ten logs as later processes
    /// create theirs, so the file is moved to a name that retention never
    /// matches, beside the kept stderr.
    ///
    /// # Errors
    ///
    /// Returns the reason when the file cannot be read, which for a process
    /// that logged nothing is that it was never created.
    fn keep_layer_log(&self, pid: u32) -> Result<LayerLog, String>;
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
    let jobs = threads;
    let mut remaining = selection;
    let mut threads = threads;
    let mut deferred: Vec<String> = Vec::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut run = BinaryRun {
        processes: 0,
        failed: false,
    };
    // The inner loop runs one binary's rounds; leaving it means the rounds
    // are done, and the outer one picks up whatever a narrowed round set
    // aside and runs it at the width the caller asked for.
    'binary: loop {
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
            let reported = u32::try_from(round.finished.len()).unwrap_or(u32::MAX);
            let before = done.len();
            let declared = declared_exit(&end.stdout);
            let ends_itself = declared.is_some();
            if let Some((name, code)) = declared {
                let verdict = if end.kind == ExitKind::Code(code) {
                    Verdict::Passed
                } else {
                    run.failed = true;
                    Verdict::Failed(format!(
                        "the test ends this process with exit code {code}; it ended with {}",
                        end.kind.describe()
                    ))
                };
                done.insert(name.clone());
                report.result(TestResult { name, verdict });
            }
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

            let clean = ends_itself || (end.kind == ExitKind::Code(0) && round.summary.is_some());
            let counted = round.summary.as_ref().map_or(0, Summary::counted);
            // libtest's own tally, when it got that far, says whether anything
            // was still in flight without a `--list`.
            let tallied = round.summary.is_some() && counted == reported;
            // A test that ends its own process is its whole account; libtest
            // never reaches its summary line, so there is nothing to check.
            let accounted = ends_itself || tallied;
            let complete = remaining
                .as_ref()
                .is_none_or(|names| names.iter().all(|name| done.contains(name)));
            if clean && accounted && complete {
                break;
            }
            if clean {
                // A clean end that does not add up: libtest printed results
                // the runner never read, or ran to completion without a test
                // it was given. Either way the names with no outcome are what
                // the run still owes, and only naming them costs a `--list`.
                let narrowed = remaining.is_some();
                let asked = match remaining.take() {
                    Some(names) => names,
                    None => launcher.list()?,
                };
                let owed: Vec<String> = asked
                    .into_iter()
                    .filter(|name| !done.contains(name))
                    .collect();
                if owed.is_empty() {
                    break;
                }
                if accounted {
                    // libtest ran to the end without these: they are not tests it knows.
                    run.failed = true;
                    for name in owed {
                        done.insert(name.clone());
                        report.result(TestResult {
                            name,
                            verdict: Verdict::Failed(
                                "the binary ran to completion without running this test".to_owned(),
                            ),
                        });
                    }
                    break;
                }
                report.note(&format!(
                    "libtest counted {counted} results and the runner read {reported}; \
                     no outcome for: {}",
                    owed.join(", ")
                ));
                if narrowed && done.len() == before {
                    // This round was already the one that ran nothing but
                    // these, and it reported none of them: another would end
                    // the same way.
                    run.failed = true;
                    for name in owed {
                        done.insert(name.clone());
                        report.result(TestResult {
                            name,
                            verdict: Verdict::Failed(
                                "libtest counted a result for this test that never reached the runner"
                                    .to_owned(),
                            ),
                        });
                    }
                    break;
                }
                report.note(&format!(
                    "running the {} tests with no outcome in a fresh process",
                    owed.len()
                ));
                remaining = Some(owed);
                continue;
            }
            let reason = end.kind.describe();
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
            let kept = kept_stderr(launcher.keep_stderr(end.pid, &end.stderr));
            // The tests that had named themselves and never reported a result:
            // exactly what was running on the threads when the process ended.
            let mut running: Vec<String> = announced(&end.stderr)
                .into_iter()
                .filter(|name| in_flight.contains(name))
                .collect();
            report.note(&format!(
                "the process ended with {reason}; {} of its tests unaccounted for\n\
                 {}{kept}; its last lines:\n{}\n{}",
                in_flight.len(),
                in_flight_line(&running),
                stderr_tail(&end.stderr),
                kept_layer_log(launcher.keep_layer_log(end.pid))
            ));
            let named = libtest::panicked_tests(&end.stderr)
                .into_iter()
                .find(|name| in_flight.contains(name));
            if named.is_none() && threads == 1 && !ran_something && round.started.is_none() {
                // One thread, nothing ran, nothing names a test: the binary itself
                // is broken, and running it again would only say so again.
                run.failed = true;
                let detail = format!(
                    "the process ended ({reason}) before running any test; {kept}\n{}",
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
                            "the process ended ({reason}) while this test ran; {kept}\n{}",
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
                // at a time, where the start line does. The tests that named
                // themselves are that set, so only they run that way; the rest
                // are nothing to do with this end and wait for a process of
                // their own at the caller's width.
                threads = 1;
                if !running.is_empty() {
                    deferred.extend(
                        in_flight
                            .iter()
                            .filter(|name| !running.contains(name))
                            .cloned(),
                    );
                    in_flight = mem::take(&mut running);
                }
                report
                    .note("nothing names the test it ended in; running those again one at a time");
            }
            if fail_fast && run.failed {
                for name in in_flight.into_iter().chain(mem::take(&mut deferred)) {
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
        if deferred.is_empty() {
            break 'binary;
        }
        report.note(&format!(
            "running the {} tests the narrowed round set aside",
            deferred.len()
        ));
        remaining = Some(mem::take(&mut deferred));
        threads = jobs;
    }
    Ok(run)
}

/// The tests that named themselves on `stderr`, each once, in the order they first did.
///
/// A test names itself on every thread it creates a device from, and its
/// worker threads carry its name, so one test can announce several times.
fn announced(stderr: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for line in stderr.lines() {
        let Some((_, name)) = line.split_once(RUNNING) else {
            continue;
        };
        let name = name.trim();
        if !names.iter().any(|seen| seen == name) {
            names.push(name.to_owned());
        }
    }
    names
}

/// The report line naming what was in flight, or nothing when no test named itself.
fn in_flight_line(running: &[String]) -> String {
    if running.is_empty() {
        return String::new();
    }
    format!("in flight when it ended: {}\n", running.join(", "))
}

/// Where the layer's own log of a dead process is kept and what it says about the end.
fn kept_layer_log(kept: Result<LayerLog, String>) -> String {
    match kept {
        Ok(LayerLog {
            path,
            tail,
            not_kept: None,
        }) => format!("the layer's own log is kept as {}:\n{tail}", path.display()),
        Ok(LayerLog {
            path,
            tail,
            not_kept: Some(why),
        }) => format!(
            "the layer's own log is still {}, which its retention will remove ({why}):\n{tail}",
            path.display()
        ),
        Err(why) => format!("the layer wrote no log of it ({why})"),
    }
}

/// Where a dead process's whole stderr went, or why it could not be kept.
fn kept_stderr(kept: Result<PathBuf, String>) -> String {
    match kept {
        Ok(path) => format!("its full stderr is in {}", path.display()),
        Err(why) => format!("its full stderr could not be kept ({why})"),
    }
}

/// The test that ended its own process and the exit code it declared.
///
/// A process that ends inside a test reports nothing about it: libtest never
/// prints its result, and the code the process ended with is all that is
/// left to judge it by. So the test names itself in the marker, because
/// nothing else does once the process is gone.
fn declared_exit(stdout: &str) -> Option<(String, i32)> {
    stdout.lines().find_map(|line| {
        let (name, code) = line
            .split_once(ENDS_PROCESS)?
            .1
            .split_once(ENDS_PROCESS_CODE)?;
        Some((name.to_owned(), code.trim().parse().ok()?))
    })
}

#[cfg(test)]
mod tests;
