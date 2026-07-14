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

/// Scan combined stdout+stderr for failing sites and a crash bit.
///
/// `signaled` is set when the spawned process died by a fatal signal — a
/// crash signal independent of stdout. Crash markers are only honoured on lines
/// that are *not* themselves failure lines, so a failure message that happens to
/// quote a signal name cannot false-positive the crash bit.
#[must_use]
pub fn parse_subtest_output(output: &str, signaled: bool) -> SubtestResult {
    let mut sites: BTreeMap<Site, u32> = BTreeMap::new();
    let mut flaky_marked: BTreeMap<Site, u32> = BTreeMap::new();
    let mut todo_marked: BTreeMap<Site, u32> = BTreeMap::new();
    let mut crash = signaled;
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
mod tests {
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
        let out = parse_subtest_output(include_str!("../tests/fixtures/device.txt"), false);
        assert!(out.crash, "the Unhandled exception line marks a crash");
        assert_eq!(out.sites.get(&site("device.c", 125)), Some(&3));
        assert_eq!(out.sites.get(&site("device.c", 792)), Some(&1));
        assert_eq!(out.sites.get(&site("device.c", 816)), Some(&1));
        assert_eq!(out.sites.len(), 3);
    }

    #[test]
    fn visual_fixture_multiple_sites_and_crash() {
        let out = parse_subtest_output(include_str!("../tests/fixtures/visual.txt"), false);
        assert!(out.crash);
        assert_eq!(out.sites.len(), 3);
        assert_eq!(out.sites.get(&site("visual.c", 9100)), Some(&1));
    }

    #[test]
    fn stateblock_fixture_zero_failures_but_objc_abort_is_a_crash() {
        let out = parse_subtest_output(include_str!("../tests/fixtures/stateblock.txt"), false);
        assert!(out.sites.is_empty());
        assert!(
            out.crash,
            "libc++abi/NSException abort is a crash even with zero failures"
        );
    }

    #[test]
    fn d3d9ex_fixture_one_failure_no_crash() {
        let out = parse_subtest_output(include_str!("../tests/fixtures/d3d9ex.txt"), false);
        assert_eq!(out.sites.get(&site("d3d9ex.c", 55)), Some(&1));
        assert!(!out.crash);
    }

    #[test]
    fn clean_fixture_no_failures_no_crash() {
        let out = parse_subtest_output(include_str!("../tests/fixtures/clean.txt"), false);
        assert!(out.sites.is_empty());
        assert!(!out.crash);
    }

    #[test]
    fn signal_death_marks_crash_even_when_output_is_clean() {
        let out = parse_subtest_output(include_str!("../tests/fixtures/clean.txt"), true);
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
             visual.c:15668: Test marked todo: Got unexpected colour 0x00fefe00.\n",
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
    fn crash_marker_inside_a_failure_message_does_not_false_positive() {
        let line = "device.c:1: Test failed: expected no SIGSEGV but the handler saw one\n";
        let out = parse_subtest_output(line, false);
        assert_eq!(out.sites.get(&site("device.c", 1)), Some(&1));
        assert!(
            !out.crash,
            "a marker inside a Test failed message must not set the crash bit"
        );
    }
}
