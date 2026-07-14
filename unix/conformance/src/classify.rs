//! The per-site failure classification.
//!
//! A human assigns one of these to every failing site in `baseline.txt`; the
//! `--update-baseline` merge then carries the tag forward across re-baselines
//! (see [`crate::merge`]). The tags are explained for readers in the
//! `CONFORMANCE.md` alongside this crate.

use std::{fmt, str::FromStr};

/// Why a given site fails. Stored per site in the baseline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Classification {
    /// A genuine defect we intend to fix.
    Real,
    /// The test takes a different expected-value branch because our caps omit a bit.
    ///
    /// Our pixels/values are correct, so this is not a defect.
    Caps,
    /// We deliberately do not implement this (e.g. `D3D9Ex`).
    ///
    /// The failure is the documented, by-design outcome.
    Expected,
    /// A site attributed to a crash/abort path.
    Crash,
    /// Environmental / non-deterministic.
    ///
    /// The site fails (or its count moves) run-to-run on the *identical*
    /// binary, driven by the host (window-manager event timing, Retina backing
    /// scale, GPU format-support tie-breaks), not by our DLL. The hit count is
    /// therefore not load-bearing, so a count change at a `Flaky` site does not
    /// gate (see [`crate::diff`]). Confirm a site is genuinely flaky with the
    /// runner's repeat mode (`--repeat`) before tagging it — this tolerance
    /// masks real regressions at the exact site.
    Flaky,
    /// Newly appeared; a human has not yet triaged it.
    Untriaged,
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Real => "real",
            Self::Caps => "caps",
            Self::Expected => "expected",
            Self::Crash => "crash",
            Self::Flaky => "flaky",
            Self::Untriaged => "untriaged",
        })
    }
}

impl FromStr for Classification {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "real" => Ok(Self::Real),
            "caps" => Ok(Self::Caps),
            "expected" => Ok(Self::Expected),
            "crash" => Ok(Self::Crash),
            "flaky" => Ok(Self::Flaky),
            "untriaged" => Ok(Self::Untriaged),
            other => Err(format!("unknown classification {other:?}")),
        }
    }
}
