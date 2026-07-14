//! Diff a fresh run against the baseline and render a human report.
//!
//! Gate contract (preserved from the shell runner): the process exits non-zero
//! only on a *regression* — a site's count went up, a new failing site
//! appeared, or a subtest started crashing. Improvements and persisted
//! untriaged sites are reported but do not fail the gate.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

use crate::{
    classify::Classification,
    model::{Arch, Baseline, Site, Subtest, SubtestBaseline, SubtestResult},
};

/// The diff outcome: whether any regression was found, plus the rendered report.
pub struct Report {
    pub regressed: bool,
    pub text: String,
}

/// Compare `current` against `baseline` for every `(arch, subtest)`.
#[must_use]
pub fn diff(baseline: &Baseline, current: &BTreeMap<(Arch, Subtest), SubtestResult>) -> Report {
    let mut text = String::new();
    let mut regressed = false;
    for arch in Arch::ALL {
        for subtest in Subtest::ALL {
            let key = (arch, subtest);
            let Some(cur) = current.get(&key) else {
                continue;
            };
            let base = baseline.entries.get(&key);
            let base_failed = base.map_or(0, total_failed);
            let cur_failed: u32 = cur.sites.values().sum();
            let base_crash = base.is_some_and(|b| b.crash);

            let mut details: Vec<String> = Vec::new();
            let mut sub_regressed = false;
            let mut sub_improved = false;

            let mut locs: BTreeSet<&Site> = BTreeSet::new();
            if let Some(b) = base {
                locs.extend(b.sites.keys());
            }
            locs.extend(cur.sites.keys());
            for site in locs {
                let bc = base.and_then(|b| b.sites.get(site)).map_or(0, |e| e.count);
                let cc = cur.sites.get(site).copied().unwrap_or(0);
                // A site a human pinned `flaky` fails non-deterministically on the
                // identical binary; its count is not load-bearing, so a delta in
                // *either* direction is a tolerated flutter, not a verdict — it
                // sets neither `sub_regressed` nor `sub_improved`. (A brand-new
                // un-pinned site has `bc == 0` and no class, so it still
                // regresses.)
                let flaky = site_class(base, site) == Some(Classification::Flaky);
                if cc > bc {
                    if flaky {
                        details.push(format!(
                            "  {site}  {bc} -> {cc}  flaky (count up, tolerated)"
                        ));
                    } else {
                        sub_regressed = true;
                        let label = if bc == 0 {
                            "REGRESSION (new failing site, untriaged)"
                        } else {
                            "REGRESSION (count up)"
                        };
                        details.push(format!("  {site}  {bc} -> {cc}  {label}"));
                    }
                } else if cc < bc {
                    if flaky {
                        details.push(format!(
                            "  {site}  {bc} -> {cc}  flaky (count down, tolerated)"
                        ));
                    } else {
                        sub_improved = true;
                        let label = if cc == 0 {
                            "improvement (site gone)"
                        } else {
                            "improvement (count down)"
                        };
                        details.push(format!("  {site}  {bc} -> {cc}  {label}"));
                    }
                } else if bc > 0 && site_class(base, site) == Some(Classification::Untriaged) {
                    details.push(format!(
                        "  {site}  {cc}  untriaged - classify in baseline.txt"
                    ));
                }
            }

            if cur.crash && !base_crash {
                sub_regressed = true;
                details.push("  crash  0 -> 1  REGRESSION (new crash)".to_owned());
            } else if !cur.crash && base_crash {
                sub_improved = true;
                details.push("  crash  1 -> 0  improvement (crash gone)".to_owned());
            }

            if cur.crash && cur_failed != base_failed {
                details.push(
                    "  note: subtest crashed - counts cover only failures before truncation"
                        .to_owned(),
                );
            }

            if cur.crash
                && let Some(panic) = &cur.panic
            {
                details.push(format!("  note: rust {panic}"));
            }

            let status = if sub_regressed {
                "REGRESSION"
            } else if sub_improved {
                "improved"
            } else {
                "ok"
            };
            regressed |= sub_regressed;
            let _ = writeln!(
                text,
                "{arch}/{subtest}  baseline(failed={base_failed} crash={}) current(failed={cur_failed} crash={}) {status}",
                u8::from(base_crash),
                u8::from(cur.crash)
            );
            for detail in &details {
                text.push_str(detail);
                text.push('\n');
            }
        }
    }
    Report { regressed, text }
}

fn total_failed(sub: &SubtestBaseline) -> u32 {
    sub.sites.values().map(|e| e.count).sum()
}

/// The baseline classification recorded for `site`.
///
/// `None` if the site is not in the baseline (a brand-new failure).
fn site_class(base: Option<&SubtestBaseline>, site: &Site) -> Option<Classification> {
    base.and_then(|b| b.sites.get(site))
        .map(|entry| entry.class)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::diff;
    use crate::{
        classify::Classification,
        model::{Arch, Baseline, Site, SiteEntry, Subtest, SubtestBaseline, SubtestResult},
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

    fn baseline_with(sites: &[(u32, u32, Classification)], crash: bool) -> Baseline {
        let mut sub = SubtestBaseline {
            crash,
            sites: BTreeMap::new(),
        };
        for &(line, count, class) in sites {
            sub.sites.insert(site(line), SiteEntry { count, class });
        }
        let mut baseline = Baseline::default();
        baseline.entries.insert(key(), sub);
        baseline
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
        let base = baseline_with(&[(1, 5, Classification::Real)], false);
        let mut cur = current_with(&[(1, 5)], true);
        cur.get_mut(&key()).unwrap().panic =
            Some("panicked at d3d9/src/device.rs:1080:22 — misaligned pointer dereference".into());
        let report = diff(&base, &cur);
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
        let base = baseline_with(&[(1, 5, Classification::Real)], false);
        let cur = current_with(&[(1, 6)], false);
        assert!(diff(&base, &cur).regressed);
    }

    #[test]
    fn new_site_is_regression() {
        let base = baseline_with(&[(1, 5, Classification::Real)], false);
        let cur = current_with(&[(1, 5), (2, 1)], false);
        assert!(diff(&base, &cur).regressed);
    }

    #[test]
    fn new_crash_is_regression() {
        let base = baseline_with(&[(1, 5, Classification::Real)], false);
        let cur = current_with(&[(1, 5)], true);
        assert!(diff(&base, &cur).regressed);
    }

    #[test]
    fn count_down_and_crash_gone_are_improvements_not_regressions() {
        let base = baseline_with(&[(1, 5, Classification::Real)], true);
        let cur = current_with(&[(1, 3)], false);
        let report = diff(&base, &cur);
        assert!(!report.regressed);
        assert!(report.text.contains("improved"), "{}", report.text);
    }

    #[test]
    fn unchanged_is_ok() {
        let base = baseline_with(&[(1, 5, Classification::Real)], false);
        let cur = current_with(&[(1, 5)], false);
        assert!(!diff(&base, &cur).regressed);
    }

    #[test]
    fn flaky_site_count_up_is_tolerated_not_a_regression() {
        let base = baseline_with(&[(5368, 1, Classification::Flaky)], false);
        let cur = current_with(&[(5368, 2)], false);
        let report = diff(&base, &cur);
        assert!(!report.regressed, "a flaky site flutter must not gate");
        assert!(report.text.contains("flaky (count up"), "{}", report.text);
        assert!(report.text.contains("ok"), "{}", report.text);
    }

    #[test]
    fn flaky_site_count_down_is_tolerated_not_an_improvement() {
        let base = baseline_with(&[(5368, 1, Classification::Flaky)], false);
        let cur = current_with(&[(5368, 0)], false);
        let report = diff(&base, &cur);
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
    fn new_site_still_regresses_even_alongside_a_flaky_site() {
        // `Flaky` only applies to sites already pinned; a brand-new failing site
        // is still an untriaged regression.
        let base = baseline_with(&[(5368, 1, Classification::Flaky)], false);
        let cur = current_with(&[(5368, 1), (9999, 1)], false);
        assert!(diff(&base, &cur).regressed);
    }

    #[test]
    fn persisted_untriaged_site_is_flagged() {
        let base = baseline_with(&[(1, 5, Classification::Untriaged)], false);
        let cur = current_with(&[(1, 5)], false);
        let report = diff(&base, &cur);
        assert!(!report.regressed);
        assert!(report.text.contains("untriaged"), "{}", report.text);
    }
}
