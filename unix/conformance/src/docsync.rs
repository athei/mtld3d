//! Baseline ↔ CONFORMANCE.md consistency check.
//!
//! `baseline.txt` holds the per-site classification data; the "Per-cluster
//! classification" section of `CONFORMANCE.md` holds the rationale prose, with
//! each cluster block declaring its sites as `<line>=<class>` tokens on
//! `Sites:` lines under a `### <file>.c/<test_function>` heading. The two files
//! drift silently without a check — clusters were historically kept "for
//! history" after their sites were fixed, and new sites landed in the baseline
//! with no prose at all. The test here fails `make test` on any divergence:
//! a baseline site missing from the prose, a prose site missing from the
//! baseline, a duplicate site declaration, a class mismatch, or a site whose
//! class differs between architectures.

use std::collections::BTreeMap;

use crate::model::{Baseline, Site};

/// One site declaration from a CONFORMANCE.md `Sites:` line.
struct DocSite {
    class: String,
    cluster: String,
}

/// Parse `Sites:` declarations from the per-cluster section of CONFORMANCE.md.
///
/// Returns the declared class per site, attributed to the enclosing
/// `### <file>.c/<fn>` heading. Headings without a `.c/` (prose headers like
/// "### device.c clusters") do not start a cluster. Errors on `Sites:` outside
/// a cluster, a malformed token, or a duplicate site.
fn parse_doc_sites(text: &str) -> Result<BTreeMap<Site, DocSite>, String> {
    let section = text
        .split_once("## Per-cluster classification")
        .ok_or("CONFORMANCE.md: missing '## Per-cluster classification' section")?
        .1;
    let mut sites = BTreeMap::new();
    let mut cluster: Option<(String, String)> = None;
    for (idx, line) in section.lines().enumerate() {
        if let Some(heading) = line.strip_prefix("### ") {
            cluster = heading
                .split_once(".c/")
                .map(|(file, func)| (format!("{file}.c"), func.trim().to_owned()));
            continue;
        }
        let Some(tokens) = line.strip_prefix("Sites:") else {
            continue;
        };
        let (file, func) = cluster
            .as_ref()
            .ok_or_else(|| format!("CONFORMANCE.md: 'Sites:' outside a cluster (+{idx})"))?;
        for token in tokens.split_whitespace() {
            let (line_no, class) = token
                .split_once('=')
                .ok_or_else(|| format!("CONFORMANCE.md: malformed site token '{token}'"))?;
            let line: u32 = line_no
                .parse()
                .map_err(|_| format!("CONFORMANCE.md: bad line number in '{token}'"))?;
            let site = Site {
                file: file.clone(),
                line,
            };
            let doc = DocSite {
                class: class.to_owned(),
                cluster: format!("{file}/{func}"),
            };
            if sites.insert(site, doc).is_some() {
                return Err(format!("CONFORMANCE.md: duplicate site {file}:{line}"));
            }
        }
    }
    Ok(sites)
}

/// Flatten the baseline to one class per site.
///
/// Errors on a site whose class differs between architectures — the same
/// divergence has the same nature everywhere, so a per-arch tag split is
/// always a triage mistake.
fn baseline_site_classes(baseline: &Baseline) -> Result<BTreeMap<Site, String>, String> {
    let mut classes: BTreeMap<Site, String> = BTreeMap::new();
    for sub in baseline.entries.values() {
        for (site, entry) in &sub.sites {
            let class = entry.class.to_string();
            match classes.get(site) {
                None => {
                    classes.insert(site.clone(), class);
                }
                Some(existing) if *existing == class => {}
                Some(existing) => {
                    return Err(format!(
                        "baseline: {site} tagged '{existing}' and '{class}' on different arches"
                    ));
                }
            }
        }
    }
    Ok(classes)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{baseline_site_classes, parse_doc_sites};
    use crate::model::Baseline;

    #[test]
    fn conformance_md_matches_baseline() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let baseline_text =
            fs::read_to_string(dir.join("baseline.txt")).expect("baseline.txt must be readable");
        let doc_text = fs::read_to_string(dir.join("CONFORMANCE.md"))
            .expect("CONFORMANCE.md must be readable");

        let baseline = Baseline::from_text(&baseline_text).expect("baseline.txt must parse");
        let classes = baseline_site_classes(&baseline).expect("consistent classes across arches");
        let doc = parse_doc_sites(&doc_text).expect("CONFORMANCE.md Sites: lines must parse");

        let mut problems = Vec::new();
        for (site, class) in &classes {
            match doc.get(site) {
                None => problems.push(format!(
                    "{site} (class={class}) is in baseline.txt but has no \
                     Sites: entry in CONFORMANCE.md"
                )),
                Some(doc_site) if doc_site.class != *class => problems.push(format!(
                    "{site} is '{class}' in baseline.txt but '{}' under {} in CONFORMANCE.md",
                    doc_site.class, doc_site.cluster
                )),
                Some(_) => {}
            }
        }
        for (site, doc_site) in &doc {
            if !classes.contains_key(site) {
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
}
