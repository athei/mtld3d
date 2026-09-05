//! Command-line argument parsing for the e2e runner.

use std::{path::PathBuf, time::Duration};

/// Parsed invocation options.
#[derive(Debug)]
pub struct Config {
    /// `--wine`: the Wine loader every test binary is spawned with.
    pub wine: PathBuf,
    /// `--jobs`: how many tests a binary runs at once (`--test-threads`).
    pub jobs: u32,
    /// `--timeout`: how long a process may go without reporting a result.
    pub timeout: Duration,
    /// `--no-fail-fast` absent: stop starting processes after the first failure.
    pub fail_fast: bool,
    /// `--filter`: substrings a test id has to contain one of; empty = every test.
    pub filter: Vec<String>,
    /// The test binaries, after `--`.
    pub exes: Vec<PathBuf>,
}

/// Parse CLI args (excluding `argv[0]`).
///
/// Recognised: `--wine <path>`, `--jobs <N>`, `--timeout <secs>`,
/// `--no-fail-fast`, `--filter <patterns>` (whitespace-separated), then
/// `--` and the test binaries. `--wine` and at least one binary are
/// mandatory; `--jobs` defaults to 1 and `--timeout` to 60 seconds.
///
/// # Errors
///
/// Returns a message on an unknown flag, a flag missing or mis-parsing its
/// value, a missing `--wine`, or no binary.
pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Config, String> {
    let mut wine: Option<PathBuf> = None;
    let mut jobs = 1;
    let mut timeout = Duration::from_mins(1);
    let mut fail_fast = true;
    let mut filter = Vec::new();
    let mut exes = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wine" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--wine needs a path".to_owned())?;
                wine = Some(PathBuf::from(value));
            }
            "--jobs" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--jobs needs a count".to_owned())?;
                jobs = value
                    .parse::<u32>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("--jobs must be a count >= 1, not {value:?}"))?;
            }
            "--timeout" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--timeout needs seconds".to_owned())?;
                let secs = value
                    .parse::<u64>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("--timeout must be seconds >= 1, not {value:?}"))?;
                timeout = Duration::from_secs(secs);
            }
            "--no-fail-fast" => fail_fast = false,
            "--filter" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--filter needs patterns".to_owned())?;
                filter.extend(value.split_whitespace().map(str::to_owned));
            }
            "--" => {
                exes.extend(args.by_ref().map(PathBuf::from));
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let wine = wine.ok_or_else(|| "missing --wine <path to the wine loader>".to_owned())?;
    if exes.is_empty() {
        return Err("no test binary given after --".to_owned());
    }
    Ok(Config {
        wine,
        jobs,
        timeout,
        fail_fast,
        filter,
        exes,
    })
}

#[cfg(test)]
mod tests;
