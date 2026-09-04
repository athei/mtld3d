//! Wine d3d9 conformance runner for mtld3d.
//!
//! Runs Wine's upstream `d3d9_test.exe` (handed in as `--exe`, alongside the
//! loader as `--wine`) against our installed builtin `d3d9.dll`, one subtest at
//! a time, and either diffs the result against the checked-in `baseline.txt` or
//! re-records it. See the `CONFORMANCE.md` alongside this crate for the triage
//! prose.
//!
//! One invocation covers one architecture's test binary, so the caller decides
//! what to run and where it lives; nothing here resolves a Wine path.
//!
//! This is intentionally NOT a pass/fail gate of zero failures — many subtests
//! fail by design given our documented stub/limitation list. It is a
//! tracked-score tool that exits non-zero only on a *regression* vs the
//! baseline.

mod classify;
mod cli;
mod diff;
mod isolate;
mod merge;
mod model;
mod run;
mod scan;
mod triage;

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    path::{Path, PathBuf},
    process::ExitCode,
};

use crate::model::{Baseline, Gpu, Leg, Subtest, SubtestResult};

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
    let config = cli::parse_args(std::env::args().skip(1))?;
    let wine_version = run::wine_version(&config.wine);
    let leg = Leg {
        arch: config.arch,
        variant: config.variant,
        gpu: Gpu::host(),
    };
    println!("Wine: {wine_version} ({leg})");

    // `--only` narrows the subtest set; absent = all four.
    let subtests: Vec<Subtest> = config
        .only
        .map_or_else(|| Subtest::ALL.to_vec(), |s| vec![s]);

    // `--repeat N>1` is characterization, not a gate: run each selected subtest N
    // times and print a flap report, then exit 0 regardless of what fluttered.
    if config.repeat > 1 {
        isolate::run_flap(&config.wine, &config.exe, leg, &subtests, config.repeat)?;
        return Ok(ExitCode::SUCCESS);
    }

    let mut current: BTreeMap<(Leg, Subtest), SubtestResult> = BTreeMap::new();
    let mut validation_errors = 0;
    for &subtest in &subtests {
        let run = run::run_subtest(&config.wine, &config.exe, leg, subtest)?;
        validation_errors += run.validation_errors;
        current.insert((leg, subtest), run.result);
    }

    // Assets (baseline.txt) live in the crate directory by default; --assets
    // overrides for out-of-tree use.
    let assets = config
        .assets
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let baseline_path = assets.join("baseline.txt");

    if config.update {
        // Re-baselining must never be blocked by an unparseable prior (a
        // legacy-format or corrupt file): the prior is consulted only to
        // report which sites are new or gone.
        let prior = load_optional(&baseline_path).unwrap_or_else(|e| {
            eprintln!("warning: ignoring unparseable prior baseline ({e}); new/dropped reporting incomplete");
            Baseline::default()
        });
        let (next, summary) = merge::merge(&prior, leg, &current, wine_version);
        std::fs::write(&baseline_path, next.to_text())
            .map_err(|e| format!("writing {}: {e}", baseline_path.display()))?;
        println!(
            "wrote baseline ({}): {} carried, {} new, {} dropped",
            next.wine_version,
            summary.carried,
            summary.new_sites.len(),
            summary.dropped_sites.len()
        );
        // Point the human at the exact CONFORMANCE.md edits the re-baseline
        // requires; `make test` (the triage sync test) stays red until done.
        // Loaded tolerantly: the doc may be mid-edit during a re-baseline.
        let classes = triage::load(&assets).unwrap_or_default();
        for site in &summary.new_sites {
            println!("  new: {site} - add to its CONFORMANCE.md cluster with a rationale");
        }
        for site in &summary.dropped_sites {
            let cluster = classes
                .get(site)
                .map_or_else(|| "not documented".to_owned(), |doc| doc.cluster.clone());
            println!("  dropped: {site} - remove its CONFORMANCE.md entry ({cluster})");
        }
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
    let classes = triage::load(&assets)?;
    let report = diff::diff(&baseline, &classes, &current);
    print!("{}", report.text);
    Ok(verdict(&report, validation_errors))
}

/// Turn a run's diff report and its Metal-validation error count into an exit code.
///
/// Three independent verdicts, each fatal on its own: a regression vs the
/// baseline, a baseline that overstates reality, and any Metal API-validation
/// error message the leg logged. The validation gate is orthogonal to the
/// per-site counts: a leg that starts misusing Metal while every count holds
/// still has to fail.
fn verdict(report: &diff::Report, validation_errors: usize) -> ExitCode {
    let validation_failed = run::validation_gate_failed(validation_errors);
    if validation_failed {
        println!(
            "conformance: METAL VALIDATION - {validation_errors} error message(s) logged; every \
             metal-validation: line above, with the detail indented under it, is API misuse to fix"
        );
    }
    if report.regressed {
        println!("conformance: REGRESSIONS detected");
    } else if report.stale {
        println!(
            "conformance: STALE BASELINE - some sites read below their pins; run `make conformance-baseline` and re-triage"
        );
    } else if !validation_failed {
        println!("conformance: no regressions vs baseline");
    }
    if validation_failed || report.regressed || report.stale {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
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
fn render_current(current: &BTreeMap<(Leg, Subtest), SubtestResult>) -> String {
    let mut out = String::new();
    for leg in Leg::ALL {
        for subtest in Subtest::ALL {
            let Some(cur) = current.get(&(leg, subtest)) else {
                continue;
            };
            let failed: u32 = cur.sites.values().sum();
            let _ = writeln!(
                out,
                "{leg}/{subtest}  failed={failed} crash={}",
                u8::from(cur.crash)
            );
            for (site, count) in &cur.sites {
                let _ = writeln!(out, "  {site}  {count}");
            }
        }
    }
    out
}
