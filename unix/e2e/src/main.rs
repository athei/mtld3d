//! End-to-end test runner for mtld3d: the suite's test binaries under Wine, one process each.
//!
//! Each binary the Makefile hands in runs once, all of its tests on the
//! number of threads `--jobs` names, and only a failure, a crash or a hang
//! costs another process (see `attribute`). Every path and knob is an
//! argument; the environment is inherited whole, so the caller owns
//! `MTLD3D_CONFIG` and the Wine variables.
//!
//! Exit code 0 when every selected test passed, 1 when any failed or was
//! left unrun by a failure, 2 when the runner itself could not do its job.

mod attribute;
mod binary;
mod cli;
mod libtest;
mod report;
mod run;
mod select;

use std::process::ExitCode;

use crate::{
    attribute::Launcher as _,
    binary::{WineLauncher, binary_name},
    report::{BinaryReport, Tally},
    select::{selected, test_id},
};

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("e2e: {msg}");
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<ExitCode, String> {
    let config = cli::parse_args(std::env::args().skip(1))?;
    let mut tally = Tally::default();
    let mut stopped_at: Option<String> = None;
    for exe in &config.exes {
        let name = binary_name(exe);
        if stopped_at.is_some() {
            println!("SKIP {name}:: (not run after a failure)");
            continue;
        }
        let mut launcher = WineLauncher::new(&config.wine, exe, config.timeout, Box::new(|_| {}));
        let selection = if config.filter.is_empty() {
            None
        } else {
            let names: Vec<String> = launcher
                .list()?
                .into_iter()
                .filter(|test| selected(&test_id(&name, test), &config.filter))
                .collect();
            if names.is_empty() {
                continue;
            }
            Some(names)
        };
        let run = attribute::run_binary(
            &mut launcher,
            selection,
            config.jobs,
            config.fail_fast,
            &mut BinaryReport {
                binary: &name,
                tally: &mut tally,
            },
        )?;
        tally.processes += run.processes;
        if run.failed && config.fail_fast {
            stopped_at = Some(name);
        }
    }
    if tally.total() == 0 {
        if config.filter.is_empty() {
            return Err("no binary ran any test".to_owned());
        }
        println!("no test matches the filter; nothing to run");
    }
    tally.print_summary();
    if let Some(name) = stopped_at {
        println!("stopped after the first failure (in {name}); --no-fail-fast runs everything");
    }
    Ok(if tally.is_red() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}
