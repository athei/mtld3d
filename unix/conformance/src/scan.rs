//! Parse one subtest's captured output into a [`SubtestResult`].

use std::collections::BTreeMap;

use crate::model::{Site, SubtestResult};

/// The substring that marks a Wine test-framework assertion failure.
///
/// `<file>.c:<line>: Test failed: <message>`
const FAILURE_MARKER: &str = ": Test failed:";

/// A failure inside a Wine `flaky` macro prints `Test marked flaky:` (not `Test failed:`).
///
/// The framework keeps it out of the exit status unless
/// `WINETEST_REPORT_FLAKY` is set (which we never set). We tally these into a
/// separate map so they stay non-gating but visible in the repeat-mode report.
const FLAKY_MARKER: &str = ": Test marked flaky:";

/// A failure inside a Wine `todo` block prints `Test marked todo:`.
///
/// An expected-to-fail-on-Wine assertion. Like [`FLAKY_MARKER`], tallied
/// separately and never gated.
const TODO_MARKER: &str = ": Test marked todo:";

/// Substrings that mark a crash/abort in the captured output.
///
/// This is a superset of the old shell runner's set
/// (`SIGSEGV`/`FATAL`/`Unhandled exception`): the documented `stateblock`
/// failure is a host-side Metal `abort()` (a zero-dimension
/// `MTLTextureDescriptor`), not a Windows SEH — so the `d3d9.dll` `FATAL`
/// banner never fires and the signal / Obj-C abort markers below are what
/// actually catch it. Combined with the `signaled` flag in
/// [`parse_subtest_output`], an abort that prints nothing parseable is still
/// recorded.
const CRASH_MARKERS: &[&str] = &[
    "Unhandled exception",
    "SIGSEGV",
    "SIGABRT",
    "SIGILL",
    "SIGBUS",
    "FATAL",
    "libc++abi",
    "terminating with uncaught exception",
    "NSException",
];

/// Marks a Rust panic in `d3d9.dll` / the unix `.so`.
///
/// Handled specially (not in [`CRASH_MARKERS`]) so the panic *location* on the
/// line — and the message on the following line — are lifted out for the
/// report, instead of just setting the crash bit. A panic aborts the whole
/// process, so it is always a crash.
const PANIC_MARKER: &str = "panicked at";

/// The end-of-run summary every Wine test prints, whatever its outcome.
///
/// `<name>: N tests executed (M marked as todo, F failures), S skipped.`
/// Its absence means the process ended before the framework did: an
/// unhandled Win32 exception exits the process with a status code rather than
/// a signal, prints nothing under `WINEDEBUG=-all`, and leaves every later
/// test unrun. Counting such a run as a clean one would record a truncated
/// baseline.
const SUMMARY_MARKER: &str = " tests executed (";

/// Scan combined stdout+stderr for failing sites and a crash bit.
///
/// `signaled` is set when the spawned process died by a fatal signal — a
/// crash signal independent of stdout. Crash markers are only honoured on lines
/// that are *not* themselves failure lines, so a failure message that happens to
/// quote a signal name cannot false-positive the crash bit. Output without the
/// end-of-run summary is a crash too, whatever else it holds.
#[must_use]
pub fn parse_subtest_output(output: &str, signaled: bool) -> SubtestResult {
    let mut sites: BTreeMap<Site, u32> = BTreeMap::new();
    let mut flaky_marked: BTreeMap<Site, u32> = BTreeMap::new();
    let mut todo_marked: BTreeMap<Site, u32> = BTreeMap::new();
    let mut crash = signaled || !output.contains(SUMMARY_MARKER);
    let mut panic: Option<String> = None;
    // The Rust panic message sits on the line *after* the `panicked at` header
    // (`panicked at <loc>:\n<message>`); set when the header is seen so the next
    // line is appended to the captured location.
    let mut want_panic_msg = false;
    for line in output.lines() {
        if want_panic_msg {
            want_panic_msg = false;
            if let Some(p) = panic.as_mut() {
                let msg = line.trim();
                if !msg.is_empty() {
                    p.push_str(" — ");
                    p.push_str(msg);
                }
            }
        }
        if let Some(idx) = line.find(FAILURE_MARKER) {
            *sites.entry(site_from_prefix(&line[..idx])).or_insert(0) += 1;
            continue;
        }
        if let Some(idx) = line.find(FLAKY_MARKER) {
            *flaky_marked
                .entry(site_from_prefix(&line[..idx]))
                .or_insert(0) += 1;
            continue;
        }
        if let Some(idx) = line.find(TODO_MARKER) {
            *todo_marked
                .entry(site_from_prefix(&line[..idx]))
                .or_insert(0) += 1;
            continue;
        }
        if let Some(idx) = line.find(PANIC_MARKER) {
            crash = true;
            if panic.is_none() {
                // Keep only `panicked at <loc>:` — drop the thread-name prefix.
                panic = Some(line[idx..].trim().to_owned());
                want_panic_msg = true;
            }
            continue;
        }
        if CRASH_MARKERS.iter().any(|marker| line.contains(marker)) {
            crash = true;
        }
    }
    SubtestResult {
        crash,
        sites,
        panic,
        flaky_marked,
        todo_marked,
    }
}

/// Recover `<file>.c:<line>` from the text preceding [`FAILURE_MARKER`].
///
/// A prefix whose trailing token is not an integer line number is kept whole
/// with `line = 0` rather than dropped — an unrecognised shape must still
/// surface as a counted failure.
fn site_from_prefix(prefix: &str) -> Site {
    if let Some((file, line)) = prefix.rsplit_once(':')
        && let Ok(line) = line.trim().parse::<u32>()
    {
        return Site {
            file: file.trim().to_owned(),
            line,
        };
    }
    Site {
        file: prefix.trim().to_owned(),
        line: 0,
    }
}

#[cfg(test)]
mod tests;
