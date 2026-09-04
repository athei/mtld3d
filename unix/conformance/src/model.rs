//! Data model for the conformance baseline plus its text (de)serializer.
//!
//! The on-disk `baseline.txt` is the single source of truth for *which* Wine
//! `d3d9_test.exe` assertions fail (per `file:line`, with a hit count) and how
//! often. *Why* a site fails — its classification — lives with the rationale
//! prose in CONFORMANCE.md and is loaded by [`crate::triage`]; keeping the two
//! concerns in separate files means the machine can freely rewrite counts on a
//! re-baseline without ever touching human prose. The format is deliberately a
//! hand-parsed, diff-friendly text file — see [`Baseline::to_text`] for the
//! exact shape.

use std::{
    collections::BTreeMap,
    fmt::{self, Write as _},
    str::FromStr,
};

/// The PE architectures the suite runs against, in baseline-output order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Arch {
    I686,
    X64,
}

/// Which device answers the run was measured under, in baseline-output order.
///
/// `Native` runs against the device's own capabilities; `Intel` forces every
/// `intel.*` config key on, so the suite sees the answers an Intel/AMD Mac
/// gives whatever the machine underneath.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Variant {
    Native,
    Intel,
}

/// One runner process's identity: the architecture it ran and the device answers it ran under.
///
/// The baseline is keyed by leg and subtest, so a measurement under the Intel
/// answers never overwrites the native one for the same architecture.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Leg {
    pub arch: Arch,
    pub variant: Variant,
}

/// The four `d3d9_test.exe` subtests, in baseline-output order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Subtest {
    Device,
    Visual,
    Stateblock,
    D3d9Ex,
}

/// A single failing assertion location, e.g. `device.c:792`.
///
/// `file` keeps its source extension (`device.c`); `line` is the source line.
/// `(file, line)` is the stable identity of a Wine test failure — the message
/// text varies with runtime values, the location does not.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Site {
    pub file: String,
    pub line: u32,
}

/// The recorded baseline for one `(leg, subtest)`.
///
/// The crash bit plus every failing site's hit count. Counts only — the
/// classification for a site lives in CONFORMANCE.md's per-cluster section
/// (see [`crate::triage`]), so each datum has exactly one authoritative home:
/// machine-recorded counts here, human-assigned classes with their rationale
/// prose there.
#[derive(PartialEq, Eq, Debug, Default)]
pub struct SubtestBaseline {
    pub crash: bool,
    pub sites: BTreeMap<Site, u32>,
}

/// The full checked-in baseline.
///
/// The Wine version it was taken against plus one [`SubtestBaseline`] per
/// `(leg, subtest)`.
#[derive(PartialEq, Eq, Debug, Default)]
pub struct Baseline {
    pub wine_version: String,
    pub entries: BTreeMap<(Leg, Subtest), SubtestBaseline>,
}

/// A fresh run's result for one `(leg, subtest)`.
///
/// The crash bit plus the per-site hit counts, before any classification is
/// assigned.
#[derive(PartialEq, Eq, Debug, Default)]
pub struct SubtestResult {
    pub crash: bool,
    pub sites: BTreeMap<Site, u32>,
    /// First Rust panic surfaced in the captured output.
    ///
    /// Formatted `panicked at <file>:<line> — <message>`, or `None` if the
    /// crash was not a panic.
    ///
    /// A panic on a worker thread aborts the whole `d3d9_test.exe` process,
    /// and our crash handler then prints a *misleading* `FATAL: SIGSEGV in
    /// wine` banner on top of the abort — so without lifting the panic line
    /// out of the noise the gate would report only an opaque "subtest
    /// crashed". Carrying the panic location turns a flaky abort into a
    /// pinpointed `file:line`.
    pub panic: Option<String>,
    /// Per-site count of assertions Wine itself wrapped in its `flaky` macro.
    ///
    /// Printed as `<file>.c:<line>: Test marked flaky: …`. These are kept
    /// *separate* from `sites` — they never gate (the upstream test author
    /// already declared them non-deterministic) — but recording them gives the
    /// repeat-mode flap report visibility into upstream-flagged jitter
    /// alongside our own.
    pub flaky_marked: BTreeMap<Site, u32>,
    /// Per-site count of assertions inside a Wine `todo` block.
    ///
    /// Printed as `Test marked todo:` — expected-to-fail-on-Wine markers. Like
    /// `flaky_marked`, kept out of `sites` and non-gating; recorded only for
    /// report visibility.
    pub todo_marked: BTreeMap<Site, u32>,
}

impl Variant {
    /// The `MTLD3D_CONFIG` entries the variant adds to the runner's pinned set.
    ///
    /// Empty for `Native`. `Intel` turns on every `intel.*` key, which is
    /// how a real Intel/AMD Mac answers, so the run measures the whole
    /// family at once rather than one key at a time.
    #[must_use]
    pub const fn config_entries(self) -> &'static str {
        match self {
            Self::Native => "",
            Self::Intel => {
                ";intel.expandPacked16=true;intel.denyFloat32Filtering=true;\
                 intel.managedMemory=true;intel.linearAlign256=true"
            }
        }
    }
}

impl Leg {
    /// Every leg, in baseline-output order: both variants of each architecture.
    pub const ALL: [Self; 4] = [
        Self {
            arch: Arch::I686,
            variant: Variant::Native,
        },
        Self {
            arch: Arch::I686,
            variant: Variant::Intel,
        },
        Self {
            arch: Arch::X64,
            variant: Variant::Native,
        },
        Self {
            arch: Arch::X64,
            variant: Variant::Intel,
        },
    ];
}

impl Subtest {
    /// Every subtest, in baseline-output order.
    pub const ALL: [Self; 4] = [Self::Device, Self::Visual, Self::Stateblock, Self::D3d9Ex];

    /// The argument passed to `d3d9_test.exe` to select this subtest.
    #[must_use]
    pub const fn arg(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Visual => "visual",
            Self::Stateblock => "stateblock",
            Self::D3d9Ex => "d3d9ex",
        }
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::I686 => "i686",
            Self::X64 => "x86_64",
        })
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Native => "native",
            Self::Intel => "intel",
        })
    }
}

/// `i686` for the native leg, `i686+intel` for the Intel one.
///
/// The native form is the bare architecture so a baseline recorded before
/// variants existed reads unchanged.
impl fmt::Display for Leg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.variant {
            Variant::Native => write!(f, "{}", self.arch),
            Variant::Intel => write!(f, "{}+{}", self.arch, self.variant),
        }
    }
}

impl fmt::Display for Subtest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.arg())
    }
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

impl FromStr for Arch {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i686" => Ok(Self::I686),
            "x86_64" => Ok(Self::X64),
            other => Err(format!("unknown arch {other:?}")),
        }
    }
}

impl FromStr for Variant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "native" => Ok(Self::Native),
            "intel" => Ok(Self::Intel),
            other => Err(format!("unknown variant {other:?}")),
        }
    }
}

impl FromStr for Leg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (arch, variant) = match s.split_once('+') {
            Some((arch, variant)) => (arch.parse::<Arch>()?, variant.parse::<Variant>()?),
            None => (s.parse::<Arch>()?, Variant::Native),
        };
        Ok(Self { arch, variant })
    }
}

impl FromStr for Subtest {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "device" => Ok(Self::Device),
            "visual" => Ok(Self::Visual),
            "stateblock" => Ok(Self::Stateblock),
            "d3d9ex" => Ok(Self::D3d9Ex),
            other => Err(format!("unknown subtest {other:?}")),
        }
    }
}

impl Baseline {
    /// Serialize to the on-disk text format.
    ///
    /// Output is deterministic: the `BTreeMap`s iterate in `Leg`/`Subtest`
    /// declaration order and sites sort by `(file, line)`, so re-serializing an
    /// unchanged model is byte-identical.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("# mtld3d d3d9 conformance baseline — per-site failure counts.\n");
        let _ = writeln!(out, "# Wine: {}", self.wine_version);
        out.push_str(
            "# Regenerate with 'make conformance-baseline'. Classifications + triage prose\n",
        );
        out.push_str(
            "# live in CONFORMANCE.md ('Per-cluster classification'); a runner unit test\n",
        );
        out.push_str("# keeps the two files covering the same sites.\n");
        out.push_str("# Format: \"[arch/subtest] crash=<0|1>\" header, then indented\n");
        out.push_str("#         \"  <file>.c:<line> count=<n>\". An arch of the form\n");
        out.push_str("#         \"<arch>+intel\" is the run under the intel.* config keys.\n");
        out.push('\n');
        for (&(leg, subtest), sub) in &self.entries {
            let _ = writeln!(out, "[{leg}/{subtest}] crash={}", u8::from(sub.crash));
            for (site, count) in &sub.sites {
                let _ = writeln!(out, "  {site} count={count}");
            }
        }
        out
    }

    /// Parse the on-disk text format.
    ///
    /// # Errors
    ///
    /// Returns a `baseline:<line>: …` message on a malformed header, a site line
    /// before any header, an unparseable leg/subtest/count, or a line that is
    /// neither a comment, a header, nor an indented site.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut baseline = Self::default();
        let mut current: Option<(Leg, Subtest)> = None;
        for (idx, raw) in text.lines().enumerate() {
            let lineno = idx + 1;
            if raw.trim().is_empty() {
                continue;
            }
            if let Some(rest) = raw.strip_prefix('#') {
                if let Some(ver) = rest.trim_start().strip_prefix("Wine:") {
                    ver.trim().clone_into(&mut baseline.wine_version);
                }
                continue;
            }
            if raw.starts_with('[') {
                let (key, sub) =
                    parse_header(raw).map_err(|e| format!("baseline:{lineno}: {e}"))?;
                baseline.entries.insert(key, sub);
                current = Some(key);
                continue;
            }
            if raw.starts_with([' ', '\t']) {
                let key = current
                    .ok_or_else(|| format!("baseline:{lineno}: site line before any header"))?;
                let (site, entry) =
                    parse_site(raw.trim()).map_err(|e| format!("baseline:{lineno}: {e}"))?;
                baseline
                    .entries
                    .get_mut(&key)
                    .expect("current key was inserted when the header was parsed")
                    .sites
                    .insert(site, entry);
                continue;
            }
            return Err(format!("baseline:{lineno}: unexpected line {raw:?}"));
        }
        Ok(baseline)
    }
}

fn parse_header(line: &str) -> Result<((Leg, Subtest), SubtestBaseline), String> {
    let close = line
        .find(']')
        .ok_or_else(|| format!("malformed header (no ']'): {line:?}"))?;
    let inside = &line[1..close];
    let (leg_str, subtest_str) = inside
        .split_once('/')
        .ok_or_else(|| format!("malformed header (no '/'): {line:?}"))?;
    let leg = leg_str.parse::<Leg>()?;
    let subtest = subtest_str.parse::<Subtest>()?;
    let crash_tok = line[close + 1..].trim();
    let crash = match crash_tok.strip_prefix("crash=") {
        Some("0") => false,
        Some("1") => true,
        _ => return Err(format!("malformed header (expected 'crash=0|1'): {line:?}")),
    };
    Ok((
        (leg, subtest),
        SubtestBaseline {
            crash,
            sites: BTreeMap::new(),
        },
    ))
}

fn parse_site(line: &str) -> Result<(Site, u32), String> {
    let mut toks = line.split_whitespace();
    let loc = toks
        .next()
        .ok_or_else(|| format!("empty site line: {line:?}"))?;
    let count_tok = toks
        .next()
        .ok_or_else(|| format!("site line missing count: {line:?}"))?;
    if let Some(extra) = toks.next() {
        return Err(format!(
            "unexpected trailing token {extra:?} (classes moved to CONFORMANCE.md): {line:?}"
        ));
    }
    let (file, line_str) = loc
        .rsplit_once(':')
        .ok_or_else(|| format!("site location missing ':': {loc:?}"))?;
    let line_no = line_str
        .parse::<u32>()
        .map_err(|_| format!("site line number not an integer: {line_str:?}"))?;
    let count = count_tok
        .strip_prefix("count=")
        .ok_or_else(|| format!("expected 'count=<n>': {count_tok:?}"))?
        .parse::<u32>()
        .map_err(|_| format!("count not an integer: {count_tok:?}"))?;
    Ok((
        Site {
            file: file.to_owned(),
            line: line_no,
        },
        count,
    ))
}

#[cfg(test)]
mod tests;
