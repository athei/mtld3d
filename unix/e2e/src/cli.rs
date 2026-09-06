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
    /// `--log-dir`: where a dead process's stderr is kept; `None` = beside the test binary.
    pub log_dir: Option<PathBuf>,
    /// The test binaries, after `--`.
    pub exes: Vec<PathBuf>,
}

/// Parse CLI args (excluding `argv[0]`).
///
/// Recognised: `--wine <path>`, `--jobs <N>`, `--timeout <secs>`,
/// `--no-fail-fast`, `--filter <patterns>` (whitespace-separated),
/// `--log-dir <path>`, then `--` and the test binaries. `--wine` and at
/// least one binary are mandatory; `--jobs` defaults to 1, `--timeout` to
/// 60 seconds, and the log directory to the one the layer writes its own
/// per-process logs to, `mtld3d-logs` beside the test binary.
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
    let mut log_dir: Option<PathBuf> = None;
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
            "--log-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--log-dir needs a path".to_owned())?;
                log_dir = Some(PathBuf::from(value));
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
        log_dir,
        exes,
    })
}

#[cfg(test)]
mod tests;
