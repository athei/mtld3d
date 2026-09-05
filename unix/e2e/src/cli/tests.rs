//! Unit tests for the runner's argument parser.

use std::time::Duration;

use super::parse_args;

fn args(tokens: &[&str]) -> std::vec::IntoIter<String> {
    tokens
        .iter()
        .map(|t| (*t).to_owned())
        .collect::<Vec<_>>()
        .into_iter()
}

#[test]
fn defaults_fill_in_behind_the_mandatory_pair() {
    let config = parse_args(args(&["--wine", "/w", "--", "/a.exe", "/b.exe"])).unwrap();
    assert_eq!(config.wine.to_str(), Some("/w"));
    assert_eq!(config.jobs, 1);
    assert_eq!(config.timeout, Duration::from_mins(1));
    assert!(config.fail_fast);
    assert!(config.filter.is_empty());
    assert_eq!(config.exes.len(), 2);
}

#[test]
fn every_flag_is_read() {
    let config = parse_args(args(&[
        "--wine",
        "/w",
        "--jobs",
        "4",
        "--timeout",
        "240",
        "--no-fail-fast",
        "--filter",
        "msaa:: stencil",
        "--",
        "/a.exe",
    ]))
    .unwrap();
    assert_eq!(config.jobs, 4);
    assert_eq!(config.timeout, Duration::from_mins(4));
    assert!(!config.fail_fast);
    assert_eq!(config.filter, ["msaa::", "stencil"]);
}

#[test]
fn missing_and_malformed_inputs_name_themselves() {
    let err = |tokens: &[&str]| parse_args(args(tokens)).unwrap_err();
    assert!(err(&["--", "/a.exe"]).contains("--wine"));
    assert!(err(&["--wine", "/w"]).contains("no test binary"));
    assert!(err(&["--wine", "/w", "--jobs", "0", "--", "/a"]).contains("--jobs"));
    assert!(err(&["--wine", "/w", "--timeout", "x", "--", "/a"]).contains("--timeout"));
    assert!(err(&["--wine", "/w", "--bogus", "--", "/a"]).contains("--bogus"));
}
