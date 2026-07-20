//! `--update-baseline` merge: fold a fresh run into the prior baseline.
//!
//! With classifications living in CONFORMANCE.md (see [`crate::triage`]),
//! a re-baseline is a plain re-record of counts — there is no class state to
//! carry. The prior baseline is still consulted to report what changed: which
//! sites are new (they need a CONFORMANCE.md entry before `make test` goes
//! green again) and which were dropped (their prose entries are now stale and
//! must be removed).

use std::collections::BTreeMap;

use crate::model::{Arch, Baseline, Site, Subtest, SubtestBaseline, SubtestResult};

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

/// Build a new baseline from `fresh` results.
#[must_use]
pub fn merge(
    prior: &Baseline,
    fresh: &BTreeMap<(Arch, Subtest), SubtestResult>,
    wine_version: String,
) -> (Baseline, MergeSummary) {
    let mut next = Baseline {
        wine_version,
        entries: BTreeMap::new(),
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
mod tests {
    use std::collections::BTreeMap;

    use super::merge;
    use crate::model::{Arch, Baseline, Site, Subtest, SubtestBaseline, SubtestResult};

    fn site(line: u32) -> Site {
        Site {
            file: "device.c".to_owned(),
            line,
        }
    }

    #[test]
    fn records_counts_and_reports_new_and_dropped_sites() {
        let key = (Arch::I686, Subtest::Device);

        let mut prior_sub = SubtestBaseline {
            crash: false,
            sites: BTreeMap::new(),
        };
        prior_sub.sites.insert(site(1), 5);
        prior_sub.sites.insert(site(2), 3);
        let mut prior = Baseline {
            wine_version: "old".to_owned(),
            entries: BTreeMap::new(),
        };
        prior.entries.insert(key, prior_sub);

        let mut fresh_sites = BTreeMap::new();
        fresh_sites.insert(site(1), 7u32); // still fails, count up
        fresh_sites.insert(site(3), 2u32); // new site
        let mut fresh = BTreeMap::new();
        fresh.insert(
            key,
            SubtestResult {
                crash: true,
                sites: fresh_sites,
                panic: None,
                ..Default::default()
            },
        );

        let (next, summary) = merge(&prior, &fresh, "new".to_owned());
        let sub = &next.entries[&key];
        assert!(sub.crash);
        assert_eq!(sub.sites[&site(1)], 7); // refreshed
        assert_eq!(sub.sites[&site(3)], 2); // recorded
        assert!(!sub.sites.contains_key(&site(2))); // dropped
        assert_eq!(summary.carried, 1);
        assert_eq!(summary.new_sites, vec![site(3)]);
        assert_eq!(summary.dropped_sites, vec![site(2)]);
        assert_eq!(next.wine_version, "new");
    }
}
