//! The anti-drift gate between `baseline.txt` and `CONFORMANCE.md`.
//!
//! Classifications live only in the prose and failure counts live only in the baseline, so
//! nothing keeps the two in step but this check: it loads both from the crate's asset
//! directory and asserts they name exactly the same sites. A baseline site with no `Sites:`
//! entry is untriaged work, a documented site that no longer fails is stale prose, and
//! either direction fails the test run with the offending sites listed.

use std::{collections::BTreeSet, path::Path};

use super::load;
use crate::model::{Baseline, Site};

/// The anti-drift gate: both files must cover exactly the same sites.
///
/// A baseline site with no prose entry is untriaged work-in-progress; a
/// prose entry for a site that no longer fails is stale documentation.
/// Either direction fails `make test` until the prose is fixed.
#[test]
fn conformance_md_covers_exactly_the_baseline_sites() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline_text =
        std::fs::read_to_string(dir.join("baseline.txt")).expect("baseline.txt must be readable");
    let baseline = Baseline::from_text(&baseline_text).expect("baseline.txt must parse");
    let doc = load(dir).expect("CONFORMANCE.md Sites: lines must parse");

    let baseline_sites: BTreeSet<&Site> = baseline
        .entries
        .values()
        .flat_map(|sub| sub.sites.keys())
        .collect();

    let mut problems = Vec::new();
    for site in &baseline_sites {
        if !doc.contains_key(*site) {
            problems.push(format!(
                "{site} is in baseline.txt but has no Sites: entry in \
                 CONFORMANCE.md — untriaged; add it to its cluster with a rationale"
            ));
        }
    }
    for (site, doc_site) in &doc {
        if !baseline_sites.contains(site) {
            problems.push(format!(
                "{site} (class={}, {}) is documented in CONFORMANCE.md but \
                 absent from baseline.txt — stale prose",
                doc_site.class, doc_site.cluster
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "CONFORMANCE.md and baseline.txt have diverged:\n  {}",
        problems.join("\n  ")
    );
}
