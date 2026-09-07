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

#[test]
fn an_observed_skip_keeps_the_prior_pin_without_inventing_a_failure() {
    let key = (I686, Subtest::Device);
    let mut prior = Baseline::default();
    prior.entries.insert(
        key,
        SubtestBaseline {
            crash: false,
            sites: BTreeMap::from([(site(6780), 1), (site(5975), 2)]),
        },
    );
    let output = "device.c:6706: Tests skipped: Test loop took too long (100 ms), skipping large query tests.\n\
        device: 58913 tests executed (75 marked as todo, 0 as flaky, 786 failures), 21 skipped.\n";
    let result = crate::scan::parse_subtest_output(output, false);
    assert!(result.sites.is_empty(), "a skip is not a measured failure");
    let fresh = BTreeMap::from([(key, result)]);
    let (next, summary) = merge(&prior, I686, &fresh, "new".to_owned());
    assert_eq!(next.entries[&key].sites, BTreeMap::from([(site(6780), 1)]));
    assert_eq!(summary.skipped_sites, vec![site(6780)]);
    assert_eq!(summary.carried, 0, "no failures were measured");
    assert_eq!(summary.dropped_sites, vec![site(5975)]);

    let (new, summary) = merge(&Baseline::default(), I686, &fresh, "new".to_owned());
    assert!(new.entries[&key].sites.is_empty(), "no pin to retain");
    assert!(summary.skipped_sites.is_empty());

    let output = format!(
        "{output}device.c:6780: Test failed: Got unexpected query result.\n\
        device.c:6780: Test failed: Got unexpected query result.\n"
    );
    let fresh = BTreeMap::from([(key, crate::scan::parse_subtest_output(&output, false))]);
    let (next, summary) = merge(&prior, I686, &fresh, "new".to_owned());
    assert_eq!(next.entries[&key].sites[&site(6780)], 2);
    assert!(
        summary.skipped_sites.is_empty(),
        "observed failures take precedence"
    );
}
