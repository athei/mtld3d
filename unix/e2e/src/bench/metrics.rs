//! The `bench-<name>.metrics` file every benchmark writes, read back line by line.
//!
//! Four kinds of line, each opening with its keyword and the benchmark's name:
//!
//! ```text
//! meta <bench> <key> <value...>
//! metric <bench> <name> <value> <unit> <direction> <class>
//! shape <bench> pass <i> <W>x<H> draws=<n> ...
//! # comment
//! ```
//!
//! A meta value is the rest of its line. The keys the comparison reads are
//! `layer` (the layer's release stamp as its log names it), `layer_image`
//! (the image ID of the `d3d9.dll` that ran), `arch`, `profile` and
//! `debug_assertions`; `config` and any other key are kept and not read. A
//! metric's direction says which way is better, and its class says how a
//! change in it is judged (see `compare`). The metric line is a contract with
//! the benchmarks, so anything it does not name is an error that points at
//! the file and line, never a line skipped: a unit or a class misread here
//! would judge a number by the wrong rule without saying so. A `shape` line
//! is kept as the text after its benchmark's name: an A/B comparison reads
//! nothing from it, and `bench-shape`, which compares it with a game's frame,
//! parses it (see `dump`).

use std::{collections::BTreeMap, fs, path::Path};

/// The meta keys whose value may not be empty.
const META_REQUIRED_VALUE: [&str; 5] = [
    "layer",
    "layer_image",
    "arch",
    "profile",
    "debug_assertions",
];

/// The prefix of a metrics file's name, before the benchmark's name.
const FILE_PREFIX: &str = "bench-";

/// The extension of a metrics file.
const FILE_EXT: &str = ".metrics";

/// The unit a metric's value is in.
#[derive(Debug, PartialEq, Eq)]
pub enum Unit {
    Ms,
    Us,
    Ns,
    Count,
    Mib,
    Bytes,
    Ratio,
}

impl Unit {
    /// The unit as the file spells it.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ms => "ms",
            Self::Us => "us",
            Self::Ns => "ns",
            Self::Count => "count",
            Self::Mib => "mib",
            Self::Bytes => "bytes",
            Self::Ratio => "ratio",
        }
    }

    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "ms" => Self::Ms,
            "us" => Self::Us,
            "ns" => Self::Ns,
            "count" => Self::Count,
            "mib" => Self::Mib,
            "bytes" => Self::Bytes,
            "ratio" => Self::Ratio,
            _ => return None,
        })
    }

    /// Four MiB in this unit, the least a `bytes` metric has to move by; `None` for other units.
    #[must_use]
    pub const fn four_mib(&self) -> Option<f64> {
        match self {
            Self::Mib => Some(4.0),
            Self::Bytes => Some(4.0 * 1024.0 * 1024.0),
            Self::Ms | Self::Us | Self::Ns | Self::Count | Self::Ratio => None,
        }
    }
}

/// Which way a metric is better.
#[derive(Debug, PartialEq, Eq)]
pub enum Direction {
    /// Smaller is better: a frame time, a memory footprint.
    Lower,
    /// Larger is better: a throughput.
    Higher,
}

impl Direction {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "lower" => Some(Self::Lower),
            "higher" => Some(Self::Higher),
            _ => None,
        }
    }
}

/// How a change in a metric is judged.
#[derive(Debug, PartialEq, Eq)]
pub enum Class {
    /// A time: the per-pair ratio against a noise floor.
    Time,
    /// A count of rare events: the per-pair difference.
    Spikes,
    /// A memory figure: the ratio, and an absolute floor on top.
    Bytes,
    /// A count the workload fixes: any difference is a change.
    Exact,
    /// A figure the machine moves run to run: judged like a time.
    Noisy,
    /// Reported, never judged.
    Info,
}

impl Class {
    /// The class as the file spells it.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Time => "time",
            Self::Spikes => "spikes",
            Self::Bytes => "bytes",
            Self::Exact => "exact",
            Self::Noisy => "noisy",
            Self::Info => "info",
        }
    }

    fn parse(word: &str) -> Option<Self> {
        Some(match word {
            "time" => Self::Time,
            "spikes" => Self::Spikes,
            "bytes" => Self::Bytes,
            "exact" => Self::Exact,
            "noisy" => Self::Noisy,
            "info" => Self::Info,
            _ => return None,
        })
    }
}

/// One `metric` line.
#[derive(Debug, PartialEq)]
pub struct Metric {
    pub value: f64,
    pub unit: Unit,
    pub direction: Direction,
    pub class: Class,
}

impl Metric {
    /// Whether `other` is defined the same way: unit, direction and class.
    #[must_use]
    pub fn same_definition(&self, other: &Self) -> bool {
        self.unit == other.unit && self.direction == other.direction && self.class == other.class
    }

    /// The definition as the file spells it, for a message: `ms lower time`.
    #[must_use]
    pub fn definition(&self) -> String {
        let direction = match self.direction {
            Direction::Lower => "lower",
            Direction::Higher => "higher",
        };
        format!("{} {direction} {}", self.unit.as_str(), self.class.as_str())
    }
}

/// Everything one metrics file says.
#[derive(Debug, Default)]
pub struct MetricsFile {
    /// The `meta` lines, by key.
    pub meta: BTreeMap<String, String>,
    /// The `metric` lines, by name.
    pub metrics: BTreeMap<String, Metric>,
    /// The `shape` lines in file order, each the text after the benchmark's name.
    pub shape: Vec<String>,
}

/// The benchmark a metrics file's name names: `bench-<name>.metrics`.
#[must_use]
pub fn bench_of(file_name: &str) -> Option<&str> {
    file_name
        .strip_prefix(FILE_PREFIX)?
        .strip_suffix(FILE_EXT)
        .filter(|name| !name.is_empty())
}

/// Read and parse the metrics file at `path`.
///
/// # Errors
///
/// Returns `<path>: <reason>` when the file cannot be read or its name is
/// not `bench-<name>.metrics`, and `<path>:<line>: <reason>` for a line the
/// format does not allow.
pub fn read(path: &Path) -> Result<MetricsFile, String> {
    let bench = path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(bench_of)
        .ok_or_else(|| format!("{}: not a bench-<name>.metrics file", path.display()))?;
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text, bench).map_err(|(line, reason)| format!("{}:{line}: {reason}", path.display()))
}

/// Parse the text of the metrics file of benchmark `bench`.
///
/// # Errors
///
/// Returns the 1-based line number and the reason for the first line the
/// format does not allow.
pub fn parse(text: &str, bench: &str) -> Result<MetricsFile, (usize, String)> {
    let mut file = MetricsFile::default();
    for (index, line) in text.lines().enumerate() {
        parse_line(line, bench, &mut file).map_err(|reason| (index + 1, reason))?;
    }
    Ok(file)
}

/// Add what one line says to `file`.
fn parse_line(line: &str, bench: &str, file: &mut MetricsFile) -> Result<(), String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(());
    }
    let (fields, rest) = split_fields(trimmed, 3);
    let (Some(&kind), Some(&named)) = (fields.first(), fields.get(1)) else {
        return Err(format!("{trimmed:?} is not a meta, metric or shape line"));
    };
    if !matches!(kind, "meta" | "metric" | "shape") {
        return Err(format!(
            "unknown line kind {kind:?}: a line is meta, metric, shape or a # comment"
        ));
    }
    if named != bench {
        return Err(format!(
            "the line names benchmark {named:?}, the file is {FILE_PREFIX}{bench}{FILE_EXT}"
        ));
    }
    match kind {
        "meta" => {
            let Some(&key) = fields.get(2) else {
                return Err("meta line without a key".to_owned());
            };
            parse_meta(key, rest, file)
        }
        "metric" => parse_metric(trimmed, file),
        _ => {
            file.shape.push(
                fields
                    .get(2)
                    .map_or_else(String::new, |first| format!("{first} {rest}"))
                    .trim_end()
                    .to_owned(),
            );
            Ok(())
        }
    }
}

/// Record one `meta <bench> <key> <value...>` line.
fn parse_meta(key: &str, value: &str, file: &mut MetricsFile) -> Result<(), String> {
    if value.is_empty() && META_REQUIRED_VALUE.contains(&key) {
        return Err(format!("meta {key} has no value"));
    }
    if file.meta.insert(key.to_owned(), value.to_owned()).is_some() {
        return Err(format!("meta {key} appears twice"));
    }
    Ok(())
}

/// Record one `metric <bench> <name> <value> <unit> <direction> <class>` line.
fn parse_metric(line: &str, file: &mut MetricsFile) -> Result<(), String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let [_, _, name, value, unit, direction, class] = fields.as_slice() else {
        return Err(format!(
            "a metric line has 7 fields (metric <bench> <name> <value> <unit> <direction> \
             <class>), this one has {}",
            fields.len()
        ));
    };
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.')
    {
        return Err(format!(
            "metric name {name:?} is not made of a-z, 0-9, '_' and '.'"
        ));
    }
    let value = value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("metric {name}: value {value:?} is not a finite number"))?;
    let unit = Unit::parse(unit).ok_or_else(|| {
        format!("metric {name}: unknown unit {unit:?}; the units are ms, us, ns, count, mib, bytes, ratio")
    })?;
    let direction = Direction::parse(direction).ok_or_else(|| {
        format!("metric {name}: unknown direction {direction:?}; it is lower or higher")
    })?;
    let class = Class::parse(class).ok_or_else(|| {
        format!(
            "metric {name}: unknown class {class:?}; the classes are time, spikes, bytes, exact, \
             noisy, info"
        )
    })?;
    if class == Class::Bytes && unit.four_mib().is_none() {
        return Err(format!(
            "metric {name}: class bytes needs unit mib or bytes, not {}",
            unit.as_str()
        ));
    }
    let metric = Metric {
        value,
        unit,
        direction,
        class,
    };
    if file.metrics.insert((*name).to_owned(), metric).is_some() {
        return Err(format!("metric {name} appears twice"));
    }
    Ok(())
}

/// The first `count` whitespace-separated fields of `line`, and the rest of it trimmed.
fn split_fields(line: &str, count: usize) -> (Vec<&str>, &str) {
    let mut fields = Vec::with_capacity(count);
    let mut rest = line.trim_start();
    while fields.len() < count && !rest.is_empty() {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        fields.push(&rest[..end]);
        rest = rest[end..].trim_start();
    }
    (fields, rest.trim_end())
}

#[cfg(test)]
mod tests;
