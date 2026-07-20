//! Wine d3d9 conformance runner for mtld3d.
//!
//! Runs Wine's upstream `d3d9_test.exe` (built in a local Wine build tree,
//! located via `--wine-build`) against our installed builtin `d3d9.dll`, one
//! subtest at a time per arch, and either diffs the result against the
//! checked-in `baseline.txt` or re-records it. See the `CONFORMANCE.md`
//! alongside this crate for the triage prose.
//!
//! This is intentionally NOT a pass/fail gate of zero failures — many subtests
//! fail by design given our documented stub/limitation list. It is a
//! tracked-score tool that exits non-zero only on a *regression* vs the
//! baseline.

mod classify;
mod cli;
mod diff;
#[cfg(test)]
mod docsync;
mod isolate;
mod merge;
mod model;
mod run;
mod scan;

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    path::{Path, PathBuf},
    process::ExitCode,
};

use crate::model::{Arch, Baseline, Subtest, SubtestResult};

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("conformance: {msg}");
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<ExitCode, String> {
    let config = cli::parse_args(std::env::args().skip(1), std::env::var("WINE_BUILD").ok())?;
    let wine = run::wine_binary()?;
    let wine_version = run::wine_version(&wine);
    println!("Wine: {wine_version}");

    // `--only`/`--arch` narrow the Cartesian product; absent = the full set.
    let arches: Vec<Arch> = config.arch.map_or_else(|| Arch::ALL.to_vec(), |a| vec![a]);
    let subtests: Vec<Subtest> = config
        .only
        .map_or_else(|| Subtest::ALL.to_vec(), |s| vec![s]);

    // `--repeat N>1` is characterization, not a gate: run each selected combo N
    // times and print a flap report, then exit 0 regardless of what fluttered.
    if config.repeat > 1 {
        isolate::run_flap(&wine, &config.wine_build, &arches, &subtests, config.repeat)?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut current: BTreeMap<(Arch, Subtest), SubtestResult> = BTreeMap::new();
    for &arch in &arches {
        for &subtest in &subtests {
            let result = run::run_subtest(&wine, &config.wine_build, arch, subtest)?;
            current.insert((arch, subtest), result);
        }
    }

    // Assets (baseline.txt) live in the crate directory by default; --assets
    // overrides for out-of-tree use.
    let assets = config
        .assets
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let baseline_path = assets.join("baseline.txt");

    if config.update {
        // Re-baselining must never be blocked by an unparseable prior (a
        // legacy-format or corrupt file): at worst we lose classification
        // carry-over for this run, which the human re-seeds.
        let prior = load_optional(&baseline_path).unwrap_or_else(|e| {
            eprintln!("warning: ignoring unparseable prior baseline ({e}); classifications reset to untriaged");
            Baseline::default()
        });
        let (next, summary) = merge::merge(&prior, &current, wine_version);
        std::fs::write(&baseline_path, next.to_text())
            .map_err(|e| format!("writing {}: {e}", baseline_path.display()))?;
        println!(
            "wrote baseline ({}): {} carried, {} untriaged, {} dropped",
            next.wine_version, summary.carried, summary.untriaged, summary.dropped
        );
        return Ok(ExitCode::SUCCESS);
    }

    if !baseline_path.is_file() {
        println!("no baseline.txt - current results (run 'make conformance-baseline' to record):");
        print!("{}", render_current(&current));
        return Ok(ExitCode::SUCCESS);
    }

    let baseline = load_optional(&baseline_path)?;
    if !baseline.wine_version.is_empty() && baseline.wine_version != wine_version {
        eprintln!(
            "warning: baseline taken against {}, running {} - file:line sites may have drifted; re-baseline expected",
            baseline.wine_version, wine_version
        );
    }
    let report = diff::diff(&baseline, &current);
    print!("{}", report.text);
    if report.regressed {
        println!("conformance: REGRESSIONS detected");
        Ok(ExitCode::from(1))
    } else {
        println!("conformance: no regressions vs baseline");
        Ok(ExitCode::SUCCESS)
    }
}

/// Load a baseline if the file exists, else an empty one.
fn load_optional(path: &Path) -> Result<Baseline, String> {
    if path.is_file() {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        Baseline::from_text(&text)
    } else {
        Ok(Baseline::default())
    }
}

/// Render fresh results as a plain per-subtest summary (no baseline to diff).
fn render_current(current: &BTreeMap<(Arch, Subtest), SubtestResult>) -> String {
    let mut out = String::new();
    for arch in Arch::ALL {
        for subtest in Subtest::ALL {
            let Some(cur) = current.get(&(arch, subtest)) else {
                continue;
            };
            let failed: u32 = cur.sites.values().sum();
            let _ = writeln!(
                out,
                "{arch}/{subtest}  failed={failed} crash={}",
                u8::from(cur.crash)
            );
            for (site, count) in &cur.sites {
                let _ = writeln!(out, "  {site}  {count}");
            }
        }
    }
    out
}
