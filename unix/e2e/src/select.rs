//! Which tests a run selects.

/// A test's id in the runner's reports and filters: `<binary>::<libtest path>`.
#[must_use]
pub fn test_id(binary: &str, name: &str) -> String {
    format!("{binary}::{name}")
}

/// Whether `id` is selected by `patterns`: any substring matches, and no pattern selects everything.
#[must_use]
pub fn selected(id: &str, patterns: &[String]) -> bool {
    patterns.is_empty() || patterns.iter().any(|pattern| id.contains(pattern.as_str()))
}

#[cfg(test)]
mod tests;
