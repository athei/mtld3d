//! Which tests a run selects.

/// A test's id in the runner's reports and filters: `<binary>::<libtest path>`.
#[must_use]
pub fn test_id(binary: &str, name: &str) -> String {
    format!("{binary}::{name}")
}

/// Whether `id` is selected by `patterns`.
///
/// Any pattern that is a substring of the id selects it; no pattern at all
/// selects everything.
#[must_use]
pub fn selected(id: &str, patterns: &[String]) -> bool {
    patterns.is_empty() || patterns.iter().any(|pattern| id.contains(pattern.as_str()))
}

/// Whether `id` is left out by `patterns`, the runner's `--skip`.
///
/// Any pattern that is a substring of the id leaves it out; no pattern at all
/// leaves out nothing. It applies after [`selected`], so a skip narrows what a
/// filter chose.
#[must_use]
pub fn skipped(id: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| id.contains(pattern.as_str()))
}

#[cfg(test)]
mod tests;
