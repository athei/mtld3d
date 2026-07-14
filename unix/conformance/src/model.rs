//! Data model for the conformance baseline plus its text (de)serializer.
//!
//! The on-disk `baseline.txt` is the single source of truth for *which* Wine
//! `d3d9_test.exe` assertions fail (per `file:line`, with a hit count) and *why*
//! (a per-site [`Classification`] that a human assigns and that survives
//! re-baselining). The format is deliberately a hand-parsed, diff-friendly text
//! file — see [`Baseline::to_text`] for the exact shape.

use std::{
    collections::BTreeMap,
    fmt::{self, Write as _},
    str::FromStr,
};

use crate::classify::Classification;

/// The PE architectures the suite runs against, in baseline-output order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Arch {
    I686,
    X64,
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

/// A baseline entry for one failing site: how many times it fired and its tag.
#[derive(PartialEq, Eq, Debug)]
pub struct SiteEntry {
    pub count: u32,
    pub class: Classification,
}

/// The recorded baseline for one `(arch, subtest)`.
///
/// The crash bit plus every failing site keyed by location.
#[derive(PartialEq, Eq, Debug, Default)]
pub struct SubtestBaseline {
    pub crash: bool,
    pub sites: BTreeMap<Site, SiteEntry>,
}

/// The full checked-in baseline.
///
/// The Wine version it was taken against plus one [`SubtestBaseline`] per
/// `(arch, subtest)`.
#[derive(PartialEq, Eq, Debug, Default)]
pub struct Baseline {
    pub wine_version: String,
    pub entries: BTreeMap<(Arch, Subtest), SubtestBaseline>,
}

/// A fresh run's result for one `(arch, subtest)`.
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

impl Arch {
    /// Every architecture, in baseline-output order.
    pub const ALL: [Self; 2] = [Self::I686, Self::X64];

    /// The Wine build-tree subdirectory holding this arch's `d3d9_test.exe`.
    #[must_use]
    pub const fn wine_target_dir(self) -> &'static str {
        match self {
            Self::I686 => "i386-windows",
            Self::X64 => "x86_64-windows",
        }
    }
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
    /// Output is deterministic: the `BTreeMap`s iterate in `Arch`/`Subtest`
    /// declaration order and sites sort by `(file, line)`, so re-serializing an
    /// unchanged model is byte-identical.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# mtld3d d3d9 conformance baseline — per-site failure counts + classification.\n",
        );
        let _ = writeln!(out, "# Wine: {}", self.wine_version);
        out.push_str(
            "# Regenerate with 'make conformance-baseline'. Triage prose in CONFORMANCE.md.\n",
        );
        out.push_str("# Format: \"[arch/subtest] crash=<0|1>\" header, then indented\n");
        out.push_str("#         \"  <file>.c:<line> count=<n> class=<real|caps|expected|crash|flaky|untriaged>\".\n");
        out.push('\n');
        for (&(arch, subtest), sub) in &self.entries {
            let _ = writeln!(out, "[{arch}/{subtest}] crash={}", u8::from(sub.crash));
            for (site, entry) in &sub.sites {
                let _ = writeln!(out, "  {site} count={} class={}", entry.count, entry.class);
            }
        }
        out
    }

    /// Parse the on-disk text format.
    ///
    /// # Errors
    ///
    /// Returns a `baseline:<line>: …` message on a malformed header, a site line
    /// before any header, an unparseable arch/subtest/count/classification, or a
    /// line that is neither a comment, a header, nor an indented site.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut baseline = Self::default();
        let mut current: Option<(Arch, Subtest)> = None;
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

fn parse_header(line: &str) -> Result<((Arch, Subtest), SubtestBaseline), String> {
    let close = line
        .find(']')
        .ok_or_else(|| format!("malformed header (no ']'): {line:?}"))?;
    let inside = &line[1..close];
    let (arch_str, subtest_str) = inside
        .split_once('/')
        .ok_or_else(|| format!("malformed header (no '/'): {line:?}"))?;
    let arch = arch_str.parse::<Arch>()?;
    let subtest = subtest_str.parse::<Subtest>()?;
    let crash_tok = line[close + 1..].trim();
    let crash = match crash_tok.strip_prefix("crash=") {
        Some("0") => false,
        Some("1") => true,
        _ => return Err(format!("malformed header (expected 'crash=0|1'): {line:?}")),
    };
    Ok((
        (arch, subtest),
        SubtestBaseline {
            crash,
            sites: BTreeMap::new(),
        },
    ))
}

fn parse_site(line: &str) -> Result<(Site, SiteEntry), String> {
    let mut toks = line.split_whitespace();
    let loc = toks
        .next()
        .ok_or_else(|| format!("empty site line: {line:?}"))?;
    let count_tok = toks
        .next()
        .ok_or_else(|| format!("site line missing count: {line:?}"))?;
    let class_tok = toks
        .next()
        .ok_or_else(|| format!("site line missing class: {line:?}"))?;
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
    let class = class_tok
        .strip_prefix("class=")
        .ok_or_else(|| format!("expected 'class=<tag>': {class_tok:?}"))?
        .parse::<Classification>()?;
    Ok((
        Site {
            file: file.to_owned(),
            line: line_no,
        },
        SiteEntry { count, class },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Arch, Baseline, Site, SiteEntry, Subtest, SubtestBaseline};
    use crate::classify::Classification;

    fn sample() -> Baseline {
        let mut device = SubtestBaseline {
            crash: true,
            sites: BTreeMap::new(),
        };
        device.sites.insert(
            Site {
                file: "device.c".to_owned(),
                line: 125,
            },
            SiteEntry {
                count: 37,
                class: Classification::Real,
            },
        );
        device.sites.insert(
            Site {
                file: "device.c".to_owned(),
                line: 792,
            },
            SiteEntry {
                count: 18,
                class: Classification::Real,
            },
        );
        let mut d3d9ex = SubtestBaseline {
            crash: false,
            sites: BTreeMap::new(),
        };
        d3d9ex.sites.insert(
            Site {
                file: "d3d9ex.c".to_owned(),
                line: 55,
            },
            SiteEntry {
                count: 1,
                class: Classification::Expected,
            },
        );
        let mut entries = BTreeMap::new();
        entries.insert((Arch::I686, Subtest::Device), device);
        entries.insert((Arch::X64, Subtest::D3d9Ex), d3d9ex);
        Baseline {
            wine_version: "wine-11.0".to_owned(),
            entries,
        }
    }

    #[test]
    fn model_text_roundtrip_is_canonical() {
        let baseline = sample();
        let text = baseline.to_text();
        assert_eq!(Baseline::from_text(&text).unwrap(), baseline);
        // Re-serializing parsed canonical text is byte-identical.
        assert_eq!(Baseline::from_text(&text).unwrap().to_text(), text);
    }

    #[test]
    fn parse_reads_wine_version() {
        let baseline = Baseline::from_text("# Wine: wine-11.0-6\n[i686/device] crash=0\n").unwrap();
        assert_eq!(baseline.wine_version, "wine-11.0-6");
    }

    #[test]
    fn parse_rejects_site_before_header() {
        let err = Baseline::from_text("  device.c:1 count=1 class=real\n").unwrap_err();
        assert!(err.contains("before any header"), "{err}");
    }

    #[test]
    fn parse_rejects_unknown_class() {
        let err = Baseline::from_text("[i686/device] crash=0\n  device.c:1 count=1 class=bogus\n")
            .unwrap_err();
        assert!(err.contains("unknown classification"), "{err}");
    }

    #[test]
    fn parse_rejects_bad_crash_token() {
        let err = Baseline::from_text("[i686/device] crash=maybe\n").unwrap_err();
        assert!(err.contains("crash=0|1"), "{err}");
    }
}
