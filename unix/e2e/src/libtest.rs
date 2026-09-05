//! What libtest prints, read back line by line.
//!
//! A test binary's stdout is libtest's report: `test <name> ... <outcome>`
//! per test and a `test result:` summary at the end. With one test thread
//! libtest writes `test <name> ... ` before it runs the test and the outcome
//! after, so a process that dies mid-test leaves the name of the test it was
//! running as an unfinished line; with more threads every line is written on
//! completion. The tests run with `--nocapture`, so a test's own prints can
//! land between the two halves of a line; the parser treats a `test` line
//! without an outcome as a start and a bare outcome line as the finish of
//! the last started test.
//!
//! stderr carries the panic reports: the default hook writes
//! `thread '<name>' panicked at ...`, and libtest names every test thread
//! after its test, so the report names the test that failed even when the
//! process ends before libtest can say so.

/// One line of a test binary's stdout that the runner acts on.
#[derive(Debug, PartialEq, Eq)]
pub enum Event {
    /// `test <name> ... ` with nothing after it: the test has started.
    Started(String),
    /// `test <name> ... <outcome>`, or a bare outcome closing a started test.
    Finished {
        /// The test's libtest path (`textures::lock_a8r8g8b8`).
        name: String,
        /// What libtest said about it.
        outcome: Outcome,
    },
    /// `test result: ...`, the counts libtest tallied for the process.
    Summary(Summary),
}

/// How libtest reported one test.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Failed,
    Ignored,
}

/// The counts on libtest's `test result:` line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub passed: u32,
    pub failed: u32,
    pub ignored: u32,
    pub filtered_out: u32,
}

/// Line-by-line reader of one process's stdout.
#[derive(Default)]
pub struct Parser {
    /// The test whose `test <name> ... ` line has no outcome yet.
    pending: Option<String>,
}

impl Parser {
    /// The event `line` carries, if any.
    pub fn line(&mut self, line: &str) -> Option<Event> {
        if let Some(counts) = line.strip_prefix("test result: ") {
            return Some(Event::Summary(parse_summary(counts)));
        }
        if let Some((name, tail)) = line
            .strip_prefix("test ")
            .and_then(|rest| rest.split_once(" ... "))
        {
            let name = name.to_owned();
            if let Some(outcome) = parse_outcome(tail) {
                self.pending = None;
                return Some(Event::Finished { name, outcome });
            }
            self.pending = Some(name.clone());
            return Some(Event::Started(name));
        }
        let outcome = parse_outcome(line.trim())?;
        let name = self.pending.take()?;
        Some(Event::Finished { name, outcome })
    }
}

fn parse_outcome(tail: &str) -> Option<Outcome> {
    match tail.trim_end() {
        "ok" => Some(Outcome::Ok),
        "FAILED" => Some(Outcome::Failed),
        tail if tail.starts_with("ignored") => Some(Outcome::Ignored),
        _ => None,
    }
}

/// The counts out of `ok. 3 passed; 1 failed; 0 ignored; 0 measured; 2 filtered out; ...`.
fn parse_summary(counts: &str) -> Summary {
    let counts = counts.split_once(". ").map_or(counts, |(_, rest)| rest);
    let mut summary = Summary::default();
    for entry in counts.split("; ") {
        let Some((count, what)) = entry.trim().split_once(' ') else {
            continue;
        };
        let Ok(count) = count.parse::<u32>() else {
            continue;
        };
        match what {
            "passed" => summary.passed = count,
            "failed" => summary.failed = count,
            "ignored" => summary.ignored = count,
            "filtered out" => summary.filtered_out = count,
            _ => {}
        }
    }
    summary
}

/// The test a `thread '<name>' panicked at` line on stderr names.
///
/// The hook may put the thread's id between the name and the verb
/// (`thread 'a::b' (756) panicked at`). `None` for any other line, and for
/// the main thread, which runs no test.
#[must_use]
pub fn panicked_test(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("thread '")?;
    let (name, rest) = rest.split_once('\'')?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('(').map_or(rest, |id| {
        id.split_once(')')
            .map_or(rest, |(_, after)| after.trim_start())
    });
    rest.starts_with("panicked at")
        .then_some(name)
        .filter(|name| *name != "main")
}

/// Every test the panic reports on `stderr` name, in order.
#[must_use]
pub fn panicked_tests(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter_map(panicked_test)
        .map(str::to_owned)
        .collect()
}

/// The panic report for `name` on `stderr`: its `panicked at` line and the message after it.
///
/// The message runs to the first blank line or libtest's backtrace note.
#[must_use]
pub fn panic_report(stderr: &str, name: &str) -> Option<String> {
    let mut lines = stderr.lines();
    let first = lines.find(|line| panicked_test(line) == Some(name))?;
    let mut report = vec![first];
    report.extend(
        lines.take_while(|line| !line.trim().is_empty() && !line.starts_with("note: run with")),
    );
    Some(report.join("\n"))
}

/// What libtest printed for a failed test in its `failures:` section on `stdout`.
///
/// The block after `---- <name> stdout ----`, up to the next block or the
/// `failures:` list; this is where a failure the process survived is
/// explained, with the test's own captured output.
#[must_use]
pub fn failure_report(stdout: &str, name: &str) -> Option<String> {
    let header = format!("---- {name} stdout ----");
    let mut lines = stdout.lines().skip_while(|line| *line != header);
    lines.next()?;
    let body: Vec<&str> = lines
        .take_while(|line| !line.starts_with("---- ") && *line != "failures:")
        .collect();
    let body = body.join("\n");
    let body = body.trim();
    (!body.is_empty()).then(|| body.to_owned())
}

/// The names `--list` printed: every `<name>: test` line.
#[must_use]
pub fn listed_tests(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests;
