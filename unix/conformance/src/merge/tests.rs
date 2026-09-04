//! Unit tests for the leg-scoped baseline merge.
//!
//! A re-baseline is a plain re-record of counts, so what matters is the change
//! report: already-failing sites are carried, fresh ones land in `new_sites`,
//! and sites that stopped failing land in `dropped_sites`. The leg scope is
//! pinned too: a run measures one architecture under one variant, so every
//! other leg's entries survive untouched and contribute nothing to the summary.

use std::collections::BTreeMap;

use super::merge;
use crate::model::{
    Arch, Baseline, Gpu, Leg, Site, Subtest, SubtestBaseline, SubtestResult, Variant,
};

const I686: Leg = Leg {
    arch: Arch::I686,
    variant: Variant::Native,
    gpu: Gpu::Apple,
};

const X64_INTEL: Leg = Leg {
    arch: Arch::X64,
    variant: Variant::Intel,
    gpu: Gpu::Apple,
};

fn site(line: u32) -> Site {
    Site {
        file: "device.c".to_owned(),
        line,
    }
}

#[test]
fn records_counts_and_reports_new_and_dropped_sites() {
    let key = (I686, Subtest::Device);

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

    let (next, summary) = merge(&prior, I686, &fresh, "new".to_owned());
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

#[test]
fn a_single_leg_update_keeps_the_other_legs() {
    let mine = (I686, Subtest::Device);
    let theirs = (X64_INTEL, Subtest::Visual);

    let mut other_sites = BTreeMap::new();
    other_sites.insert(site(9), 4u32);
    let mut prior = Baseline {
        wine_version: "old".to_owned(),
        entries: BTreeMap::new(),
    };
    prior.entries.insert(
        theirs,
        SubtestBaseline {
            crash: true,
            sites: other_sites,
        },
    );

    let mut fresh = BTreeMap::new();
    fresh.insert(mine, SubtestResult::default());

    let (next, summary) = merge(&prior, I686, &fresh, "new".to_owned());
    let kept = &next.entries[&theirs];
    assert!(kept.crash);
    assert_eq!(kept.sites[&site(9)], 4);
    assert!(next.entries.contains_key(&mine));
    // The untouched leg is carried, not re-measured: it reports nothing.
    assert_eq!(summary.carried, 0);
    assert!(summary.new_sites.is_empty());
    assert!(summary.dropped_sites.is_empty());
}
