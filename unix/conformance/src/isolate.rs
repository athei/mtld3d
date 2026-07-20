//! Repeat-mode flap characterization (`--repeat N`).
//!
//! Wine's `d3d9_test.exe` cannot run a single test *function* — `START_TEST`
//! registers only the subtest/file name, so `argv[1]` selects `device`/`visual`/…
//! and runs the whole subtest. To characterize a *non-deterministic* site (one
//! the gate flips on a clean binary) the only stock lever is to run the whole
//! subtest repeatedly and watch which sites move. This module runs each selected
//! `(arch, subtest)` N times and prints a per-site flap report: a site that
//! fired in every run at the same count is deterministic; one that fired in only
//! some runs, or at a varying count, is flaky. The output is the evidence for
//! tagging a site `flaky` in `CONFORMANCE.md` (see [`crate::triage`]). This is
//! measurement, not a gate — it never sets a non-zero exit code.

use std::{collections::BTreeMap, fmt::Write as _, path::Path};

use crate::{
    model::{Arch, Site, Subtest},
    run,
};

/// Per-site flap statistics over N runs of one `(arch, subtest)`.
#[derive(Default)]
struct SiteFlap {
    /// Runs in which the site failed at least once (count > 0).
    fired_runs: u32,
    /// Observed non-zero hit count → number of runs that showed exactly it.
    counts: BTreeMap<u32, u32>,
}

impl SiteFlap {
    /// A site is stable iff it fired in *every* run at a single, constant count.
    fn is_stable(&self, runs: u32) -> bool {
        self.fired_runs == runs && self.counts.len() == 1
    }

    /// Render the observed count distribution as `value×runs` parts.
    ///
    /// Folds in the implicit zeros (runs where the site did not fire at all).
    fn distribution(&self, runs: u32) -> String {
        let mut parts: Vec<String> = Vec::new();
        let zeros = runs - self.fired_runs;
        if zeros > 0 {
            parts.push(format!("0×{zeros}"));
        }
        for (&count, &n) in &self.counts {
            parts.push(format!("{count}×{n}"));
        }
        parts.join(", ")
    }
}

/// What was observed across N runs of one `(arch, subtest)`.
#[derive(Default)]
struct Aggregate {
    runs: u32,
    /// Runs that crashed/aborted/timed out (the per-subtest crash bit).
    crash_runs: u32,
    /// Failing (gating) sites and their flap stats.
    sites: BTreeMap<Site, SiteFlap>,
    /// Total occurrences of Wine's own `flaky`-macro-marked failures (non-gating) across all runs.
    ///
    /// Reported for visibility next to our gating sites.
    flaky_marked: BTreeMap<Site, u32>,
}

/// Run every selected `(arch, subtest)` `repeat` times and print a flap report.
///
/// # Errors
///
/// Propagates a spawn/wait error from [`run::run_subtest`] (a missing per-arch
/// `d3d9_test.exe`, or `wine` failing to launch).
pub fn run_flap(
    wine: &Path,
    wine_build: &Path,
    arches: &[Arch],
    subtests: &[Subtest],
    repeat: u32,
) -> Result<(), String> {
    println!(
        "flap characterization: {repeat} run(s) per combo (counts shown as value×runs; \
         a site is FLAPS unless it fired in every run at one constant count)\n"
    );
    for &arch in arches {
        for &subtest in subtests {
            let agg = characterize(wine, wine_build, arch, subtest, repeat)?;
            print!("{}", render(arch, subtest, &agg));
        }
    }
    Ok(())
}

/// Run one `(arch, subtest)` `repeat` times, folding each run into an [`Aggregate`].
fn characterize(
    wine: &Path,
    wine_build: &Path,
    arch: Arch,
    subtest: Subtest,
    repeat: u32,
) -> Result<Aggregate, String> {
    let mut agg = Aggregate::default();
    for i in 1..=repeat {
        // Liveness to stderr — a full subtest can take many seconds.
        eprintln!("  [{arch}/{subtest}] run {i}/{repeat}…");
        let result = run::run_subtest(wine, wine_build, arch, subtest)?;
        agg.runs += 1;
        if result.crash {
            agg.crash_runs += 1;
        }
        for (site, &count) in &result.sites {
            let flap = agg.sites.entry(site.clone()).or_default();
            flap.fired_runs += 1;
            *flap.counts.entry(count).or_insert(0) += 1;
        }
        for (site, &c) in &result.flaky_marked {
            *agg.flaky_marked.entry(site.clone()).or_insert(0) += c;
        }
    }
    Ok(agg)
}

/// Render one combo's flap report.
///
/// The header, then flapping sites individually (most-unstable info first),
/// then a compact summary of the stable sites, then any upstream
/// `flaky`-marked sites.
fn render(arch: Arch, subtest: Subtest, agg: &Aggregate) -> String {
    let runs = agg.runs;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{arch}/{subtest}  N={runs}  crash {}/{runs}",
        agg.crash_runs
    );

    let mut stable: Vec<&Site> = Vec::new();
    for (site, flap) in &agg.sites {
        if flap.is_stable(runs) {
            stable.push(site);
        } else {
            let _ = writeln!(
                out,
                "  {site}  fired {}/{runs}  counts {{{}}}  <- FLAPS",
                flap.fired_runs,
                flap.distribution(runs)
            );
        }
    }

    if !stable.is_empty() {
        let locs: Vec<String> = stable.iter().map(ToString::to_string).collect();
        let _ = writeln!(
            out,
            "  {} stable site(s) (fired {runs}/{runs} at a constant count): {}",
            stable.len(),
            locs.join(", ")
        );
    }

    if !agg.flaky_marked.is_empty() {
        let parts: Vec<String> = agg
            .flaky_marked
            .iter()
            .map(|(site, n)| format!("{site}×{n}"))
            .collect();
        let _ = writeln!(
            out,
            "  upstream flaky-marked (non-gating): {}",
            parts.join(", ")
        );
    }

    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::{Aggregate, SiteFlap, render};
    use crate::model::{Arch, Site, Subtest};

    fn site(line: u32) -> Site {
        Site {
            file: "device.c".to_owned(),
            line,
        }
    }

    /// A flap from a per-run count series, where `0` means "did not fire".
    fn flap_from(per_run: &[u32]) -> SiteFlap {
        let mut f = SiteFlap::default();
        for &c in per_run {
            if c > 0 {
                f.fired_runs += 1;
                *f.counts.entry(c).or_insert(0) += 1;
            }
        }
        f
    }

    #[test]
    fn stable_site_fires_every_run_at_constant_count() {
        let flap = flap_from(&[3, 3, 3, 3, 3]);
        assert!(flap.is_stable(5));
        assert_eq!(flap.distribution(5), "3×5");
    }

    #[test]
    fn flapping_site_is_unstable_and_shows_zeros() {
        let flap = flap_from(&[0, 1, 0, 1, 1]); // fired in 3 of 5 runs
        assert!(!flap.is_stable(5));
        assert_eq!(flap.distribution(5), "0×2, 1×3");
    }

    #[test]
    fn report_separates_flapping_from_stable_and_lists_upstream_flaky() {
        let mut agg = Aggregate {
            runs: 5,
            ..Default::default()
        };
        agg.sites.insert(site(5368), flap_from(&[0, 1, 0, 1, 1]));
        agg.sites.insert(site(6516), flap_from(&[3, 3, 3, 3, 3]));
        agg.flaky_marked.insert(site(5406), 37);

        let text = render(Arch::I686, Subtest::Device, &agg);
        assert!(
            text.contains("device.c:5368") && text.contains("<- FLAPS"),
            "{text}"
        );
        assert!(text.contains("1 stable site"), "{text}");
        assert!(text.contains("device.c:6516"), "{text}");
        assert!(
            text.contains("upstream flaky-marked") && text.contains("device.c:5406×37"),
            "{text}"
        );
    }
}
