//! `--update-baseline` merge: fold a fresh run into the prior baseline.
//!
//! This is the anti-drift mechanism. A re-baseline preserves the
//! human-assigned [`Classification`] for any site that still fails, marks
//! newly-appeared sites `Untriaged`, and forgets sites that no longer fail — so
//! triage survives across runs while genuinely new failures stay loud.

use std::collections::BTreeMap;

use crate::{
    classify::Classification,
    model::{Arch, Baseline, SiteEntry, Subtest, SubtestBaseline, SubtestResult},
};

/// Outcome counts for the human-facing summary after a re-baseline.
#[derive(Default)]
pub struct MergeSummary {
    /// Sites whose classification carried over from the prior baseline.
    pub carried: usize,
    /// Newly-appeared sites recorded as `Untriaged`.
    ///
    /// Includes prior `Untriaged` sites that still fail and remain untriaged.
    pub untriaged: usize,
    /// Prior sites that no longer fail and were dropped.
    pub dropped: usize,
}

/// Build a new baseline from `fresh` results.
///
/// Carries classifications from `prior` for surviving sites and marks new
/// sites `Untriaged`.
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
        let mut sites = BTreeMap::new();
        for (site, &count) in &result.sites {
            let class = prior_sub
                .and_then(|sub| sub.sites.get(site))
                .map_or(Classification::Untriaged, |entry| entry.class);
            if class == Classification::Untriaged {
                summary.untriaged += 1;
            } else {
                summary.carried += 1;
            }
            sites.insert(site.clone(), SiteEntry { count, class });
        }
        if let Some(sub) = prior_sub {
            summary.dropped += sub
                .sites
                .keys()
                .filter(|site| !result.sites.contains_key(*site))
                .count();
        }
        next.entries.insert(
            key,
            SubtestBaseline {
                crash: result.crash,
                sites,
            },
        );
    }
    (next, summary)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::merge;
    use crate::{
        classify::Classification,
        model::{Arch, Baseline, Site, SiteEntry, Subtest, SubtestBaseline, SubtestResult},
    };

    fn site(line: u32) -> Site {
        Site {
            file: "device.c".to_owned(),
            line,
        }
    }

    #[test]
    fn carries_class_marks_new_untriaged_and_drops_gone() {
        let key = (Arch::I686, Subtest::Device);

        let mut prior_sub = SubtestBaseline {
            crash: false,
            sites: BTreeMap::new(),
        };
        prior_sub.sites.insert(
            site(1),
            SiteEntry {
                count: 5,
                class: Classification::Real,
            },
        );
        prior_sub.sites.insert(
            site(2),
            SiteEntry {
                count: 3,
                class: Classification::Caps,
            },
        );
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
        assert_eq!(sub.sites[&site(1)].class, Classification::Real); // carried
        assert_eq!(sub.sites[&site(1)].count, 7); // refreshed
        assert_eq!(sub.sites[&site(3)].class, Classification::Untriaged); // new
        assert!(!sub.sites.contains_key(&site(2))); // dropped
        assert_eq!(summary.carried, 1);
        assert_eq!(summary.untriaged, 1);
        assert_eq!(summary.dropped, 1);
        assert_eq!(next.wine_version, "new");
    }
}
