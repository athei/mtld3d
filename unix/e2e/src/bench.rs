//! A/B benchmarking of two builds of the layer: run them interleaved, then judge the numbers.
//!
//! `bench-ab` runs the `#[ignore]`d benchmarks of the candidate's test binary
//! against two installs of the layer, the base and the candidate, each under
//! a Wine tree and a prefix of its own. The binary links `d3d9` by name and
//! depends on nothing of the layer's, so the one binary drives both legs and
//! the workload is the same on either side. Every run is a fresh process
//! running exactly one benchmark, and the legs alternate which goes first
//! from round to round, so a machine that drifts (thermals, a background
//! job) moves both legs alike instead of charging the drift to one of them.
//!
//! When both trees carry it, the host emitter benchmark runs in the same
//! rounds, each leg its own tree's `emit_corpus`, since it is host code the
//! candidate's binary cannot stand in for; its files say `meta kind host`,
//! and `compare` checks them apart from the layer's.
//!
//! A finished A/B directory holds `<leg>/<round>/bench-<name>.metrics`, the
//! leg `base` or `cand` and the round `0..N`. `bench-compare` reads that
//! directory back and pairs the two legs round by round, and `bench-ab` ends
//! by doing the same. A run of one clean commit against itself also leaves
//! a `same-image-allowed` file there (see `compare::check_builds`), so a later
//! `bench-compare` judges it the way `bench-ab` did, and a `wine.txt` naming
//! the one Wine both legs ran, which the report quotes.
//!
//! Exit code 0 when nothing regressed, 1 when something did, and 2 when the
//! run or the analysis could not be trusted: a failed benchmark, a build
//! that is not the one the leg expected, a malformed metrics file, rounds
//! that do not pair up.

use std::process::ExitCode;

mod ab;
mod args;
mod compare;
mod metrics;
mod stats;

/// The subcommand that runs an A/B comparison.
pub const AB: &str = "bench-ab";

/// The subcommand that judges a finished A/B directory.
pub const COMPARE: &str = "bench-compare";

/// The file in an A/B directory that allows both legs one `d3d9.dll` image.
const SAME_IMAGE_FILE: &str = "same-image-allowed";

/// The file in an A/B directory that names the one Wine both legs ran.
const WINE_FILE: &str = "wine.txt";

/// One of the two builds an A/B run compares, and the directory its runs write into.
#[derive(Debug, PartialEq, Eq)]
pub enum Leg {
    /// The reference build, `BASE` in the Makefile.
    Base,
    /// The build under test, the current checkout.
    Cand,
}

impl Leg {
    /// The leg's directory name under the A/B directory.
    #[must_use]
    pub const fn dir(&self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Cand => "cand",
        }
    }
}

/// Run `bench-ab` with the arguments after the subcommand's name.
///
/// # Errors
///
/// Returns a message when the arguments are wrong or the run cannot be
/// trusted; the caller exits with code 2.
pub fn ab_main(args: impl Iterator<Item = String>) -> Result<ExitCode, String> {
    let config = args::parse_ab(args)?;
    ab::run(&config)
}

/// Run `bench-compare` with the arguments after the subcommand's name.
///
/// # Errors
///
/// Returns a message when the arguments are wrong or the directory cannot
/// be judged; the caller exits with code 2.
pub fn compare_main(args: impl Iterator<Item = String>) -> Result<ExitCode, String> {
    let config = args::parse_compare(args)?;
    compare::judge_dir(&config.dir, &config.options, config.report.as_deref())
}
