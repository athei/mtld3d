//! The anti-drift gates around `CONFORMANCE.md`.
//!
//! Classifications live only in the prose and failure counts live only in the baseline, so
//! nothing keeps the two in step but this check: it loads both from the crate's asset
//! directory and asserts they name exactly the same sites. A baseline site with no `Sites:`
//! entry is untriaged work, a documented site that no longer fails is stale prose, and
//! either direction fails the test run with the offending sites listed.
//!
//! The document's own "Current classifications" sentence is the second gate, and the one
//! nothing else covers: it is prose a human writes over numbers the `Sites:` tokens already
//! carry, so a reclassification that forgets it leaves the document stating counts nobody
//! measured. `check_doc_summary` recounts the tokens and fails with the sentence to copy in.

use std::{collections::BTreeSet, path::Path};

use super::{load, parse_doc_sites};
use crate::{
    classify::Classification,
    model::{Baseline, Site},
};

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

/// Every class a `Sites:` token can carry, in the order the summary sentence names them.
///
/// `crash` comes last because the sentence does not name it: crash state is machine-recorded
/// in `baseline.txt`, so a rebuilt sentence grows a `crash` term only if a token carries one.
const SUMMARY_CLASSES: [Classification; 7] = [
    Classification::Real,
    Classification::Expected,
    Classification::Caps,
    Classification::Ceiling,
    Classification::Flaky,
    Classification::Untriaged,
    Classification::Crash,
];

/// What the "Current classifications" sentence of `CONFORMANCE.md` claims.
///
/// `preamble` is everything ahead of the final colon, whitespace-normalised, so a
/// rebuilt sentence keeps the document's own wording and the date it carries.
struct DocSummary {
    preamble: String,
    /// The classes the sentence names, in the order it names them.
    counts: Vec<(Classification, usize)>,
    total: usize,
}

/// The second anti-drift gate: the prose summary against the `Sites:` tokens.
///
/// The sentence is hand-written over data the tokens already carry, so a
/// reclassification that forgets it leaves the document stating counts nobody
/// measured (it read 133 `expected`, 4 `caps`, 24 `ceiling` for two months
/// against tokens reading 131, 1, 22). Recounting it here fails `make test`
/// instead.
#[test]
fn conformance_md_summary_matches_the_sites_tokens() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(dir.join("CONFORMANCE.md"))
        .expect("CONFORMANCE.md must be readable");
    if let Err(problem) = check_doc_summary(&text) {
        panic!("{problem}");
    }
}

#[test]
fn a_summary_that_matches_its_tokens_passes() {
    assert_eq!(
        check_doc_summary(&synthetic_doc("1 `real`, 2 `expected`, 3")),
        Ok(())
    );
}

#[test]
fn a_summary_one_count_off_fails_and_prints_the_sentence_to_copy() {
    let problem = check_doc_summary(&synthetic_doc("1 `real`, 3 `expected`, 3"))
        .expect_err("a wrong per-class count must fail the gate");
    assert!(
        problem.contains("3 `expected` where the tokens read 2"),
        "the failure must name the disagreeing term: {problem}"
    );
    assert!(
        problem.contains(
            "Current classifications, counted from the `Sites:` tokens below on 2026-09-22: \
             1 `real`, 2 `expected`, 3 unique sites in all."
        ),
        "the failure must print the sentence to copy in: {problem}"
    );
}

#[test]
fn a_summary_with_a_wrong_total_fails() {
    let problem = check_doc_summary(&synthetic_doc("1 `real`, 2 `expected`, 4"))
        .expect_err("a wrong unique-site total must fail the gate");
    assert!(
        problem.contains("4 unique sites where the tokens read 3"),
        "the failure must name the total: {problem}"
    );
}

#[test]
fn a_class_the_summary_leaves_out_fails_once_it_has_sites() {
    let problem = check_doc_summary(&synthetic_doc("1 `real`, 3"))
        .expect_err("a class with sites and no term must fail the gate");
    assert!(
        problem.contains("0 `expected` where the tokens read 2"),
        "the failure must name the missing class: {problem}"
    );
    assert!(
        problem.contains("1 `real`, 2 `expected`, 3 unique sites in all."),
        "the rebuilt sentence must carry the missing term: {problem}"
    );
}

#[test]
fn a_document_without_the_summary_sentence_fails() {
    let doc = synthetic_doc("1 `real`, 2 `expected`, 3");
    let (before, after) = doc
        .split_once("Current classifications")
        .expect("the fixture states it");
    let problem = check_doc_summary(&format!("{before}Nothing here{after}"))
        .expect_err("a missing summary sentence must fail the gate");
    assert!(
        problem.contains("no \"Current classifications\" sentence"),
        "the failure must say the sentence is missing: {problem}"
    );
}

#[test]
fn a_quoted_mention_of_the_phrase_does_not_shadow_the_real_sentence() {
    let doc = synthetic_doc("1 `real`, 2 `expected`, 3");
    let (before, sentence) = doc
        .split_once("Current classifications")
        .expect("the fixture states it");
    let mention = "A second test recounts those tokens against the \"Current classifications\" \
                   sentence below and fails with the sentence to copy in when they disagree.\n\n";
    let doc = format!("{before}{mention}Current classifications{sentence}");
    assert_eq!(check_doc_summary(&doc), Ok(()));
}

/// A three-site document whose summary sentence is `counts`.
fn synthetic_doc(counts: &str) -> String {
    format!(
        "## Per-cluster classification\n\n\
         Current classifications, counted from the `Sites:` tokens below on 2026-09-22: \
         {counts} unique sites in all.\n\n\
         ### device.c/test_wndproc\n\nSites: 100=real 200=expected\n\n\
         ### visual.c/z_range_test\n\nSites: 300=expected\n"
    )
}

/// Check the "Current classifications" sentence against the document's own `Sites:` tokens.
///
/// # Errors
///
/// Errors when the sentence is absent or malformed, and when any per-class count or the
/// unique-site total disagrees with the tokens. The message carries the sentence the
/// tokens support, so the fix is a copy.
fn check_doc_summary(text: &str) -> Result<(), String> {
    let sites = parse_doc_sites(text)?;
    let summary = parse_doc_summary(text)?;
    let counted = |class: Classification| sites.values().filter(|doc| doc.class == class).count();

    // The rebuilt sentence keeps the terms the document names, in its order, and
    // gains a term only for a class that has sites and no term of its own.
    let mut expected: Vec<(Classification, usize)> = summary
        .counts
        .iter()
        .map(|&(class, _)| (class, counted(class)))
        .collect();
    for class in SUMMARY_CLASSES {
        let sites_of_class = counted(class);
        if sites_of_class > 0 && !summary.counts.iter().any(|&(named, _)| named == class) {
            expected.push((class, sites_of_class));
        }
    }

    let mut problems = Vec::new();
    for &(class, actual) in &expected {
        let claimed = summary
            .counts
            .iter()
            .find(|&&(named, _)| named == class)
            .map_or(0, |&(_, claimed)| claimed);
        if claimed != actual {
            problems.push(format!(
                "{claimed} `{class}` where the tokens read {actual}"
            ));
        }
    }
    if summary.total != sites.len() {
        problems.push(format!(
            "{} unique sites where the tokens read {}",
            summary.total,
            sites.len()
        ));
    }
    if problems.is_empty() {
        return Ok(());
    }

    let terms: Vec<String> = expected
        .iter()
        .map(|(class, count)| format!("{count} `{class}`"))
        .collect();
    Err(format!(
        "CONFORMANCE.md: the \"Current classifications\" sentence disagrees with the Sites: \
         tokens below it: it states {}.\nThe sentence the tokens support, to copy in, rewrap \
         to the file's width, and date to the day of the recount:\n{}: {}, {} unique sites in \
         all.",
        problems.join("; "),
        summary.preamble,
        terms.join(", "),
        sites.len()
    ))
}

/// Find the "Current classifications" sentence and parse it.
///
/// The phrase occurs more than once: the document also mentions it when describing
/// this check, so the opening phrase alone is not an anchor. Every occurrence is
/// tried and the first that parses as a list of counts is the sentence, which leaves
/// a mention (no counts follow it) unable to shadow the real one however it wraps.
///
/// # Errors
///
/// Errors when the phrase occurs nowhere, and otherwise with the first occurrence's
/// problem when no occurrence carries counts.
fn parse_doc_summary(text: &str) -> Result<DocSummary, String> {
    let opening = "Current classifications";
    let mut problem = None;
    for (at, _) in text.match_indices(opening) {
        match parse_summary_sentence(&text[at..]) {
            Ok(summary) => return Ok(summary),
            Err(reason) => {
                if problem.is_none() {
                    problem = Some(reason);
                }
            }
        }
    }
    Err(problem.unwrap_or_else(|| {
        format!("CONFORMANCE.md: no {opening:?} sentence over the Sites: tokens")
    }))
}

/// Parse one candidate sentence: its terms, its total, and its wording.
///
/// The sentence is hard-wrapped prose, so it is read from its opening phrase to the
/// first full stop and whitespace-normalised. Its terms follow the last colon, which
/// puts the `Sites:` of the preamble safely ahead of them.
///
/// # Errors
///
/// Errors when the sentence is unterminated, states no counts at all, states a term
/// that is neither a count of a class nor the unique-site total, names a class twice,
/// or has no total.
fn parse_summary_sentence(rest: &str) -> Result<DocSummary, String> {
    let end = rest.find('.').ok_or_else(|| {
        format!(
            "CONFORMANCE.md: {:?} never ends",
            rest.lines().next().unwrap_or(rest)
        )
    })?;
    let sentence = rest[..end].split_whitespace().collect::<Vec<_>>().join(" ");
    let (preamble, terms) = sentence
        .rsplit_once(':')
        .ok_or_else(|| format!("CONFORMANCE.md: {sentence:?} states no counts"))?;

    let mut counts: Vec<(Classification, usize)> = Vec::new();
    let mut total = None;
    for term in terms.split(',') {
        let term = term.trim();
        let (count, what) = term
            .split_once(' ')
            .ok_or_else(|| format!("CONFORMANCE.md: malformed summary term {term:?}"))?;
        let count: usize = count
            .parse()
            .map_err(|_| format!("CONFORMANCE.md: bad count in summary term {term:?}"))?;
        if let Some(name) = what.strip_prefix('`').and_then(|w| w.strip_suffix('`')) {
            let class: Classification = name
                .parse()
                .map_err(|e| format!("CONFORMANCE.md: summary term {term:?}: {e}"))?;
            if counts.iter().any(|&(named, _)| named == class) {
                return Err(format!("CONFORMANCE.md: the summary names `{class}` twice"));
            }
            counts.push((class, count));
        } else if what.starts_with("unique sites") {
            total = Some(count);
        } else {
            return Err(format!(
                "CONFORMANCE.md: summary term {term:?} is neither a count of a class nor the \
                 unique-site total"
            ));
        }
    }
    if counts.is_empty() {
        return Err(format!("CONFORMANCE.md: {sentence:?} counts no class"));
    }
    let total =
        total.ok_or_else(|| format!("CONFORMANCE.md: {sentence:?} states no unique-site total"))?;
    Ok(DocSummary {
        preamble: preamble.to_owned(),
        counts,
        total,
    })
}
