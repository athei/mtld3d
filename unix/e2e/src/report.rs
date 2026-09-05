//! The run's report: one line per test as it lands, and a tally that never undercounts.

use crate::attribute::{Report, TestResult, Verdict};

/// The tally across every binary.
#[derive(Default)]
pub struct Tally {
    pub passed: u32,
    pub failed: u32,
    pub ignored: u32,
    pub not_run: u32,
    pub processes: u32,
    /// `(id, report)` of every failed test, for the list at the end.
    pub failures: Vec<(String, String)>,
}

/// The tally of one binary, printing each line under the binary's name.
pub struct BinaryReport<'a> {
    pub binary: &'a str,
    pub tally: &'a mut Tally,
}

impl Report for BinaryReport<'_> {
    fn result(&mut self, result: TestResult) {
        self.tally
            .record(&crate::select::test_id(self.binary, &result.name), result);
    }

    fn note(&mut self, note: &str) {
        let mut lines = note.lines();
        println!("NOTE {}: {}", self.binary, lines.next().unwrap_or_default());
        for line in lines {
            println!("     {line}");
        }
    }
}

impl Tally {
    /// Count `result` under `id` and print its line.
    pub fn record(&mut self, id: &str, result: TestResult) {
        match result.verdict {
            Verdict::Passed => {
                self.passed += 1;
                println!("PASS {id}");
            }
            Verdict::Ignored => {
                self.ignored += 1;
                println!("SKIP {id} (ignored)");
            }
            Verdict::NotRun => {
                self.not_run += 1;
                println!("SKIP {id} (not run after a failure)");
            }
            Verdict::Failed(report) => {
                self.failed += 1;
                println!("FAIL {id}");
                for line in report.lines() {
                    println!("     {line}");
                }
                self.failures.push((id.to_owned(), report));
            }
        }
    }

    /// Every test counted, whatever became of it.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.passed + self.failed + self.ignored + self.not_run
    }

    /// Whether the run is red: a failure, or tests left unrun by one.
    #[must_use]
    pub const fn is_red(&self) -> bool {
        self.failed > 0 || self.not_run > 0
    }

    /// The summary lines, and the failed ids again where they are easy to find.
    pub fn print_summary(&self) {
        println!();
        println!(
            "Summary: {} tests: {} passed, {} failed, {} ignored, {} not run; {} processes",
            self.total(),
            self.passed,
            self.failed,
            self.ignored,
            self.not_run,
            self.processes
        );
        if !self.failures.is_empty() {
            println!("FAILED:");
            for (id, _) in &self.failures {
                println!("    {id}");
            }
        }
    }
}
