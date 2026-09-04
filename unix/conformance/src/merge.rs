//! `--update-baseline` merge: fold a fresh run into the prior baseline.
//!
//! With classifications living in CONFORMANCE.md (see [`crate::triage`]),
//! a re-baseline is a plain re-record of counts — there is no class state to
//! carry. The prior baseline is still consulted to report what changed: which
//! sites are new (they need a CONFORMANCE.md entry before `make test` goes
//! green again) and which were dropped (their prose entries are now stale and
//! must be removed).
//!
//! One run covers one leg (an architecture under one variant), so the merge is
//! *leg-scoped*: the fresh results replace that leg's entries and every other
//! leg is carried over untouched. Re-recording the whole file means running each leg's
//! baseline leg once, in either order.

use std::collections::BTreeMap;

use crate::model::{Baseline, Leg, Site, Subtest, SubtestBaseline, SubtestResult};

/// What a re-baseline changed, for the human-facing summary.
#[derive(Default)]
pub struct MergeSummary {
    /// Sites that were already in the prior baseline.
    pub carried: usize,
    /// Newly-appeared sites — each needs a CONFORMANCE.md cluster entry.
    pub new_sites: Vec<Site>,
    /// Prior sites that no longer fail — their prose entries are now stale.
    ///
    /// Each of these still has a CONFORMANCE.md cluster entry that must be
    /// removed for `make test` to go green again.
    pub dropped_sites: Vec<Site>,
}

/// Build a new baseline from `fresh` results for `leg`.
///
/// Entries the prior baseline holds for any *other* leg survive verbatim: a
/// run only ever measured its own arch under its own variant, so dropping the
/// rest would record a regression-free score for tests that never ran.
#[must_use]
pub fn merge(
    prior: &Baseline,
    leg: Leg,
    fresh: &BTreeMap<(Leg, Subtest), SubtestResult>,
    wine_version: String,
) -> (Baseline, MergeSummary) {
    let mut next = Baseline {
        wine_version,
        entries: prior
            .entries
            .iter()
            .filter(|((l, _), _)| *l != leg)
            .map(|(&key, sub)| {
                (
                    key,
                    SubtestBaseline {
                        crash: sub.crash,
                        sites: sub.sites.clone(),
                    },
                )
            })
            .collect(),
    };
    let mut summary = MergeSummary::default();
    for (&key, result) in fresh {
        let prior_sub = prior.entries.get(&key);
        for site in result.sites.keys() {
            if prior_sub.is_some_and(|sub| sub.sites.contains_key(site)) {
                summary.carried += 1;
            } else {
                summary.new_sites.push(site.clone());
            }
        }
        if let Some(sub) = prior_sub {
            summary.dropped_sites.extend(
                sub.sites
                    .keys()
                    .filter(|site| !result.sites.contains_key(*site))
                    .cloned(),
            );
        }
        next.entries.insert(
            key,
            SubtestBaseline {
                crash: result.crash,
                sites: result.sites.clone(),
            },
        );
    }
    summary.new_sites.dedup();
    summary.dropped_sites.dedup();
    (next, summary)
}

#[cfg(test)]
mod tests;
