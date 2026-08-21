//! Unit tests for the baseline diff and its gate verdicts.
//!
//! Synthetic baselines and run results drive every branch: a count up, a new
//! site or a fresh crash regress, while a count down or a vanished crash gate
//! as a stale baseline rather than passing quietly. The tolerance classes get
//! their own cases, `flaky` in both directions and `ceiling` only below the
//! pin, each applying only to sites already in the baseline.

use std::collections::BTreeMap;

use super::diff;
use crate::{
    classify::Classification,
    model::{Arch, Baseline, Site, Subtest, SubtestBaseline, SubtestResult},
    triage::DocSite,
};

fn key() -> (Arch, Subtest) {
    (Arch::I686, Subtest::Device)
}

fn site(line: u32) -> Site {
    Site {
        file: "device.c".to_owned(),
        line,
    }
}

fn baseline_with(sites: &[(u32, u32)], crash: bool) -> Baseline {
    let mut sub = SubtestBaseline {
        crash,
        sites: BTreeMap::new(),
    };
    for &(line, count) in sites {
        sub.sites.insert(site(line), count);
    }
    let mut baseline = Baseline::default();
    baseline.entries.insert(key(), sub);
    baseline
}

fn classes_with(entries: &[(u32, Classification)]) -> BTreeMap<Site, DocSite> {
    entries
        .iter()
        .map(|&(line, class)| {
            (
                site(line),
                DocSite {
                    class,
                    cluster: "device.c/test".to_owned(),
                },
            )
        })
        .collect()
}

fn current_with(sites: &[(u32, u32)], crash: bool) -> BTreeMap<(Arch, Subtest), SubtestResult> {
    let mut s = BTreeMap::new();
    for &(line, count) in sites {
        s.insert(site(line), count);
    }
    let mut map = BTreeMap::new();
    map.insert(
        key(),
        SubtestResult {
            crash,
            sites: s,
            panic: None,
            ..Default::default()
        },
    );
    map
}

#[test]
fn crash_with_panic_surfaces_the_location() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[(1, Classification::Real)]);
    let mut cur = current_with(&[(1, 5)], true);
    cur.get_mut(&key()).unwrap().panic =
        Some("panicked at d3d9/src/device.rs:1080:22 — misaligned pointer dereference".into());
    let report = diff(&base, &classes, &cur);
    assert!(report.regressed, "new crash is a regression");
    assert!(
        report
            .text
            .contains("rust panicked at d3d9/src/device.rs:1080:22"),
        "{}",
        report.text
    );
}

#[test]
fn count_up_is_regression() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[(1, Classification::Real)]);
    let cur = current_with(&[(1, 6)], false);
    assert!(diff(&base, &classes, &cur).regressed);
}

#[test]
fn new_site_is_regression() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[(1, Classification::Real)]);
    let cur = current_with(&[(1, 5), (2, 1)], false);
    assert!(diff(&base, &classes, &cur).regressed);
}

#[test]
fn new_crash_is_regression() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[(1, Classification::Real)]);
    let cur = current_with(&[(1, 5)], true);
    assert!(diff(&base, &classes, &cur).regressed);
}

#[test]
fn count_down_and_crash_gone_gate_as_stale_baseline() {
    // An improvement the baseline doesn't record is a stale baseline —
    // tolerating it would widen the budget a later regression hides in.
    let base = baseline_with(&[(1, 5)], true);
    let classes = classes_with(&[(1, Classification::Real)]);
    let cur = current_with(&[(1, 3)], false);
    let report = diff(&base, &classes, &cur);
    assert!(
        !report.regressed,
        "stale is not a regression: {}",
        report.text
    );
    assert!(report.stale, "count down must gate: {}", report.text);
    assert!(report.text.contains("STALE BASELINE"), "{}", report.text);
}

#[test]
fn ceiling_site_below_the_pin_is_tolerated() {
    // A ceiling pin is a cross-environment maximum: an environment where
    // the site reads lower (or zero) must neither gate nor demand a
    // re-record.
    let base = baseline_with(&[(2234, 1)], false);
    let classes = classes_with(&[(2234, Classification::Ceiling)]);
    let cur = current_with(&[(2234, 0)], false);
    let report = diff(&base, &classes, &cur);
    assert!(!report.regressed, "{}", report.text);
    assert!(
        !report.stale,
        "below a ceiling pin is not stale: {}",
        report.text
    );
    assert!(
        report.text.contains("ceiling (below the pin"),
        "{}",
        report.text
    );
}

#[test]
fn ceiling_site_above_the_pin_is_a_regression() {
    let base = baseline_with(&[(2234, 1)], false);
    let classes = classes_with(&[(2234, Classification::Ceiling)]);
    let cur = current_with(&[(2234, 2)], false);
    let report = diff(&base, &classes, &cur);
    assert!(
        report.regressed,
        "above a ceiling pin gates: {}",
        report.text
    );
}

#[test]
fn new_site_still_regresses_even_with_a_ceiling_class_entry() {
    // Like flaky, the ceiling pin applies only to sites already in the
    // baseline.
    let base = baseline_with(&[(2234, 1)], false);
    let classes = classes_with(&[
        (2234, Classification::Ceiling),
        (9999, Classification::Ceiling),
    ]);
    let cur = current_with(&[(2234, 1), (9999, 1)], false);
    assert!(diff(&base, &classes, &cur).regressed);
}

#[test]
fn unchanged_is_ok() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[(1, Classification::Real)]);
    let cur = current_with(&[(1, 5)], false);
    assert!(!diff(&base, &classes, &cur).regressed);
}

#[test]
fn flaky_site_count_up_is_tolerated_not_a_regression() {
    let base = baseline_with(&[(5368, 1)], false);
    let classes = classes_with(&[(5368, Classification::Flaky)]);
    let cur = current_with(&[(5368, 2)], false);
    let report = diff(&base, &classes, &cur);
    assert!(!report.regressed, "a flaky site flutter must not gate");
    assert!(report.text.contains("flaky (count up"), "{}", report.text);
    assert!(report.text.contains("ok"), "{}", report.text);
}

#[test]
fn flaky_site_count_down_is_tolerated_not_an_improvement() {
    let base = baseline_with(&[(5368, 1)], false);
    let classes = classes_with(&[(5368, Classification::Flaky)]);
    let cur = current_with(&[(5368, 0)], false);
    let report = diff(&base, &classes, &cur);
    assert!(!report.regressed);
    // A flaky flutter down is noise, not a celebrated improvement.
    assert!(report.text.contains("flaky (count down"), "{}", report.text);
    assert!(
        !report.text.contains("improved"),
        "flaky flutter must not read as an improvement: {}",
        report.text
    );
}

#[test]
fn new_site_still_regresses_even_with_a_flaky_class_entry() {
    // The flaky pin applies only to sites already in the baseline; a
    // brand-new failing site regresses even if prose for it exists.
    let base = baseline_with(&[(5368, 1)], false);
    let classes = classes_with(&[(5368, Classification::Flaky), (9999, Classification::Flaky)]);
    let cur = current_with(&[(5368, 1), (9999, 1)], false);
    assert!(diff(&base, &classes, &cur).regressed);
}

#[test]
fn baseline_site_without_a_class_entry_is_flagged_untriaged() {
    let base = baseline_with(&[(1, 5)], false);
    let classes = classes_with(&[]);
    let cur = current_with(&[(1, 5)], false);
    let report = diff(&base, &classes, &cur);
    assert!(!report.regressed);
    assert!(report.text.contains("untriaged"), "{}", report.text);
}
