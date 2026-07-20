//! Site classifications loaded from CONFORMANCE.md's per-cluster section.
//!
//! Classifications are human-assigned and belong with their rationale prose,
//! so CONFORMANCE.md is their single authoritative home: each cluster block
//! under "## Per-cluster classification" declares its sites as
//! `<line>=<class>` tokens on `Sites:` lines beneath a
//! `### <file>.c/<test_function>` heading. `baseline.txt` deliberately holds
//! only machine-recorded counts — a re-baseline never needs to touch prose,
//! and a site's class can never disagree between two files because it exists
//! in exactly one. A site present in the baseline with no entry here is
//! untriaged by definition; the unit test below fails `make test` until the
//! prose is written, and the stale direction (prose for a site that no longer
//! fails) fails it too.

use std::{collections::BTreeMap, path::Path};

use crate::{classify::Classification, model::Site};

/// One site's triage: its class and the cluster heading it was declared under.
pub struct DocSite {
    pub class: Classification,
    /// `<file>/<test_function>`, for messages pointing back at the prose.
    pub cluster: String,
}

/// Load the per-site classifications from `CONFORMANCE.md` in `assets`.
///
/// # Errors
///
/// Errors if the file is unreadable, the per-cluster section is missing, a
/// `Sites:` line appears outside a cluster heading, a token is malformed, a
/// class is unknown, or a site is declared twice.
pub fn load(assets: &Path) -> Result<BTreeMap<Site, DocSite>, String> {
    let path = assets.join("CONFORMANCE.md");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    parse_doc_sites(&text)
}

/// Parse `Sites:` declarations from the per-cluster section of CONFORMANCE.md.
///
/// Headings without a `.c/` (prose headers like "### device.c clusters") do
/// not start a cluster.
fn parse_doc_sites(text: &str) -> Result<BTreeMap<Site, DocSite>, String> {
    let section = text
        .split_once("## Per-cluster classification")
        .ok_or("CONFORMANCE.md: missing '## Per-cluster classification' section")?
        .1;
    let mut sites = BTreeMap::new();
    let mut cluster: Option<(String, String)> = None;
    for line in section.lines() {
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
            .ok_or_else(|| format!("CONFORMANCE.md: 'Sites:' outside a cluster: {line:?}"))?;
        for token in tokens.split_whitespace() {
            let (line_no, class) = token
                .split_once('=')
                .ok_or_else(|| format!("CONFORMANCE.md: malformed site token '{token}'"))?;
            let line: u32 = line_no
                .parse()
                .map_err(|_| format!("CONFORMANCE.md: bad line number in '{token}'"))?;
            let class: Classification = class
                .parse()
                .map_err(|e| format!("CONFORMANCE.md: {file}:{line}: {e}"))?;
            let site = Site {
                file: file.clone(),
                line,
            };
            let doc = DocSite {
                class,
                cluster: format!("{file}/{func}"),
            };
            if sites.insert(site, doc).is_some() {
                return Err(format!("CONFORMANCE.md: duplicate site {file}:{line}"));
            }
        }
    }
    Ok(sites)
}

#[cfg(test)]
mod tests {
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
        let baseline_text = std::fs::read_to_string(dir.join("baseline.txt"))
            .expect("baseline.txt must be readable");
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
}
