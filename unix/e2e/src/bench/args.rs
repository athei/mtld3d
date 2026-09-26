//! Command-line arguments of the two benchmark subcommands.

use std::{path::PathBuf, time::Duration};

use super::{
    ab::{AbConfig, HostBench, LegSpec},
    compare::Options,
};

/// How long a benchmark process may go without a line when `--timeout` is not given.
const DEFAULT_TIMEOUT: Duration = Duration::from_mins(5);

/// How many rounds a run has when `--runs` is not given.
const DEFAULT_RUNS: u32 = 5;

/// The fewest rounds a run may have: under three, a median is one pair or the mean of two.
const MIN_RUNS: u32 = 3;

/// A parsed `bench-compare` invocation.
#[derive(Debug)]
pub struct CompareConfig {
    /// The A/B directory to judge.
    pub dir: PathBuf,
    pub options: Options,
    /// `--report`: where the report is written besides stdout.
    pub report: Option<PathBuf>,
}

/// Parse `bench-compare <ab_dir> [--accept a,b] [--report <file>] [--allow-same-image]`.
///
/// # Errors
///
/// Returns a message on an unknown flag, a flag without its value, or a
/// missing or second directory.
pub fn parse_compare(mut args: impl Iterator<Item = String>) -> Result<CompareConfig, String> {
    let mut dir: Option<PathBuf> = None;
    let mut options = Options::default();
    let mut report = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--accept" => options.accept.extend(accept_list(&value(&mut args, &arg)?)),
            "--report" => report = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--allow-same-image" => options.allow_same_image = true,
            flag if flag.starts_with("--") => return Err(format!("unknown argument {flag:?}")),
            _ if dir.is_some() => return Err(format!("a second A/B directory {arg:?}")),
            _ => dir = Some(PathBuf::from(arg)),
        }
    }
    let dir = dir.ok_or_else(|| "bench-compare needs the A/B directory".to_owned())?;
    Ok(CompareConfig {
        dir,
        options,
        report,
    })
}

/// Parse a `bench-ab` invocation.
///
/// Mandatory: `--out <dir>`, and for each leg (`base`, `cand`) the Wine
/// loader `--<leg>-wine <path>`, the prefix `--<leg>-prefix <dir>` and the
/// layer stamp the leg's runs must report, `--<leg>-stamp <stamp>`; then
/// `--` and the test binaries. Optional: `--runs <n>` (default 5, at least 3), `--bench
/// <patterns>` (whitespace-separated, repeatable; none means every
/// benchmark), `--config <MTLD3D_CONFIG>`, `--timeout <secs>` (default 300),
/// `--accept a,b`, `--report <file>` and `--allow-same-image` (the legs are
/// one commit from a clean tree) as for `bench-compare`. `--base-host <exe>`
/// and `--cand-host <exe>`, given together, add the host emitter benchmark,
/// each leg running its own tree's `emit_corpus`, and `--host-corpus
/// <file>` (repeatable) adds a shader cache for both of them to read.
/// `--corpus-dir <dir>` names the staged shader caches every end-to-end run
/// sees as `corpus` in its log directory, where the cold-start benchmark
/// looks for them.
///
/// # Errors
///
/// Returns a message on an unknown flag, a flag without its value or with
/// one that does not parse, or a mandatory argument missing.
pub fn parse_ab(mut args: impl Iterator<Item = String>) -> Result<AbConfig, String> {
    let mut base = PartialLeg::default();
    let mut cand = PartialLeg::default();
    let mut out: Option<PathBuf> = None;
    let mut runs = DEFAULT_RUNS;
    let mut benches = Vec::new();
    let mut config = String::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut options = Options::default();
    let mut report = None;
    let mut exes = Vec::new();
    let (mut base_host, mut cand_host) = (None, None);
    let mut corpora = Vec::new();
    let mut corpus_dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base-host" => base_host = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--cand-host" => cand_host = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--host-corpus" => corpora.push(PathBuf::from(value(&mut args, &arg)?)),
            "--corpus-dir" => corpus_dir = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--base-wine" => base.wine = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--base-prefix" => base.prefix = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--base-stamp" => base.stamp = Some(value(&mut args, &arg)?),
            "--cand-wine" => cand.wine = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--cand-prefix" => cand.prefix = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--cand-stamp" => cand.stamp = Some(value(&mut args, &arg)?),
            "--out" => out = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--runs" => {
                runs = count(&value(&mut args, &arg)?, &arg)?;
                if runs < MIN_RUNS {
                    return Err(format!(
                        "--runs must be at least {MIN_RUNS}: fewer rounds give the median and \
                         the MAD nothing to work with"
                    ));
                }
            }
            "--bench" => {
                benches.extend(
                    value(&mut args, &arg)?
                        .split_whitespace()
                        .map(str::to_owned),
                );
            }
            "--config" => config = value(&mut args, &arg)?,
            "--timeout" => {
                timeout = Duration::from_secs(u64::from(count(&value(&mut args, &arg)?, &arg)?));
            }
            "--accept" => options.accept.extend(accept_list(&value(&mut args, &arg)?)),
            "--report" => report = Some(PathBuf::from(value(&mut args, &arg)?)),
            "--allow-same-image" => options.allow_same_image = true,
            "--" => exes.extend(args.by_ref().map(PathBuf::from)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let out = out.ok_or_else(|| "missing --out <A/B directory>".to_owned())?;
    if exes.is_empty() {
        return Err("no test binary given after --".to_owned());
    }
    let host = match (base_host, cand_host) {
        (Some(base), Some(cand)) => Some(HostBench {
            base,
            cand,
            corpora,
        }),
        (None, None) if corpora.is_empty() => None,
        (None, None) => {
            return Err("--host-corpus without --base-host and --cand-host".to_owned());
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "--base-host and --cand-host go together: each leg runs its own emitter".to_owned(),
            );
        }
    };
    Ok(AbConfig {
        base: base.finish("base")?,
        cand: cand.finish("cand")?,
        exes,
        benches,
        runs,
        out,
        config,
        timeout,
        options,
        report,
        host,
        corpus_dir,
    })
}

/// One leg's flags as they arrive, each possibly still missing.
#[derive(Default)]
struct PartialLeg {
    wine: Option<PathBuf>,
    prefix: Option<PathBuf>,
    stamp: Option<String>,
}

impl PartialLeg {
    /// The leg, or which of its flags is missing.
    fn finish(self, leg: &str) -> Result<LegSpec, String> {
        let missing = |flag: &str| format!("missing --{leg}-{flag}");
        Ok(LegSpec {
            wine: self.wine.ok_or_else(|| missing("wine <path>"))?,
            prefix: self.prefix.ok_or_else(|| missing("prefix <dir>"))?,
            stamp: self.stamp.ok_or_else(|| missing("stamp <layer stamp>"))?,
        })
    }
}

/// The value after `flag`.
fn value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} needs a value"))
}

/// A count of at least one.
fn count(value: &str, flag: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{flag} must be a count >= 1, not {value:?}"))
}

/// The names in a comma-separated `--accept` list.
fn accept_list(value: &str) -> impl Iterator<Item = String> + '_ {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests;
