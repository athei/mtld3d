//! Diff a fresh run against the baseline and render a human report.
//!
//! Gate contract: the process exits non-zero on a *regression* — a site's
//! count went up, a new failing site appeared, or a subtest started
//! crashing — and equally on a *stale baseline* — a site's count went down
//! or a crash disappeared without the baseline being re-recorded. Gating
//! improvements keeps baseline.txt in lockstep with reality; a tolerated
//! improvement would silently widen the budget a later regression can hide
//! in. The two tolerance classes are `flaky` (count not load-bearing, either
//! direction) and `ceiling` (the pin is a cross-environment maximum; only
//! upward movement gates). Persisted untriaged sites are reported but do not
//! fail the gate (the triage sync test owns that).

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
};

use crate::{
    classify::Classification,
    model::{Arch, Baseline, Site, Subtest, SubtestBaseline, SubtestResult},
    triage::DocSite,
};

/// The diff outcome: whether the gate fails, and why, plus the rendered report.
pub struct Report {
    /// A site's count went up, a new site appeared, or a subtest crashed.
    pub regressed: bool,
    /// The baseline overstates reality: a count went down or a crash cleared.
    ///
    /// Fails the gate like a regression, but the fix is `make
    /// conformance-baseline`, not a code hunt.
    pub stale: bool,
    pub text: String,
}

/// Compare `current` against `baseline` for every `(arch, subtest)`.
///
/// `classes` is the per-site triage loaded from CONFORMANCE.md — the diff
/// consults it for the flaky tolerance and to flag baseline sites that have
/// no prose entry yet.
#[must_use]
pub fn diff(
    baseline: &Baseline,
    classes: &BTreeMap<Site, DocSite>,
    current: &BTreeMap<(Arch, Subtest), SubtestResult>,
) -> Report {
    let mut text = String::new();
    let mut regressed = false;
    let mut stale = false;
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
            let mut sub_stale = false;

            let mut locs: BTreeSet<&Site> = BTreeSet::new();
            if let Some(b) = base {
                locs.extend(b.sites.keys());
            }
            locs.extend(cur.sites.keys());
            for site in locs {
                let in_baseline = base.is_some_and(|b| b.sites.contains_key(site));
                let bc = base.and_then(|b| b.sites.get(site)).copied().unwrap_or(0);
                let cc = cur.sites.get(site).copied().unwrap_or(0);
                // A site a human pinned `flaky` fails non-deterministically on the
                // identical binary; its count is not load-bearing, so a delta in
                // *either* direction is a tolerated flutter, not a verdict — it
                // sets neither `sub_regressed` nor `sub_stale`. A `ceiling` pin
                // is a cross-environment maximum: below it is tolerated, above
                // it regresses. Both pins only apply to sites already in the
                // baseline: a brand-new site regresses even if prose for it
                // already exists.
                let class = classes.get(site).map(|doc| doc.class);
                let flaky = in_baseline && class == Some(Classification::Flaky);
                let ceiling = in_baseline && class == Some(Classification::Ceiling);
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
                    } else if ceiling {
                        details.push(format!(
                            "  {site}  {bc} -> {cc}  ceiling (below the pin, tolerated)"
                        ));
                    } else {
                        sub_stale = true;
                        let label = if cc == 0 {
                            "STALE BASELINE (site gone)"
                        } else {
                            "STALE BASELINE (count down)"
                        };
                        details.push(format!("  {site}  {bc} -> {cc}  {label}"));
                    }
                } else if bc > 0 && matches!(class, None | Some(Classification::Untriaged)) {
                    details.push(format!(
                        "  {site}  {cc}  untriaged - add to CONFORMANCE.md per-cluster section"
                    ));
                }
            }

            if cur.crash && !base_crash {
                sub_regressed = true;
                details.push("  crash  0 -> 1  REGRESSION (new crash)".to_owned());
            } else if !cur.crash && base_crash {
                sub_stale = true;
                details.push("  crash  1 -> 0  STALE BASELINE (crash gone)".to_owned());
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
            } else if sub_stale {
                "STALE BASELINE"
            } else {
                "ok"
            };
            regressed |= sub_regressed;
            stale |= sub_stale;
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
    Report {
        regressed,
        stale,
        text,
    }
}

fn total_failed(sub: &SubtestBaseline) -> u32 {
    sub.sites.values().sum()
}

#[cfg(test)]
mod tests;
