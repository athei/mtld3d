//! A perf window's length, read back out of a layer log line.
//!
//! The benchmarks align their measured frames to the layer's perf windows,
//! and `make bench-ab` runs the candidate's benchmark binary against a base
//! build whose windows may be of another length (5 s before the interval
//! became 2 s), so the length a run's windows have is read from the log it
//! writes rather than from [`super::SUMMARY_INTERVAL_SECS`]. Two lines carry
//! it: the machine-read `perf-kv v1 window_s=<s> ...` line, and the grid's
//! header `── perf  window=<s>s  frames=...`, which every build with perf
//! windows writes, including those older than the `perf-kv` line. Built in
//! every profile: the benchmark binary reads logs of `PERF=1` builds whatever
//! its own profile.

/// What opens the pairs of the `perf-kv` line, after the logger's prefix.
const KV_TAG: &str = "perf-kv v1 ";

/// What precedes the window's length on the grid's header line.
const HEADER_TAG: &str = "── perf  window=";

/// The seconds of the window a `perf-kv` line or a grid header names; `None` for any other line.
///
/// The `perf-kv` line's `window_s=` wins when a line holds both, which
/// none does. A length that is not a positive finite number is no length.
#[must_use]
pub fn window_secs(line: &str) -> Option<f64> {
    let text = if let Some((_, pairs)) = line.split_once(KV_TAG) {
        pairs
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix("window_s="))?
    } else {
        let (_, rest) = line.split_once(HEADER_TAG)?;
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        &rest[..end]
    };
    text.parse::<f64>()
        .ok()
        .filter(|secs| secs.is_finite() && *secs > 0.0)
}

#[cfg(test)]
mod tests;
