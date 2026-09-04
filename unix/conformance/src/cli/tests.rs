//! Unit tests for the conformance runner's argument parser.
//!
//! One invocation runs one test binary, so `--wine`, `--exe` and `--arch` are
//! mandatory and each missing one must name itself in the error. The rest pin
//! the guard rails: unknown and retired flags fail loudly instead of being
//! ignored, `--repeat` refuses zero, and a filtered `--update-baseline` is
//! rejected before it can drop unselected subtests from the baseline.

use super::parse_args;
use crate::model::{Arch, Subtest, Variant};

fn args(tokens: &[&str]) -> std::vec::IntoIter<String> {
    tokens
        .iter()
        .map(|t| (*t).to_owned())
        .collect::<Vec<_>>()
        .into_iter()
}

/// The mandatory trio, for tests exercising some other flag.
fn base(extra: &[&str]) -> std::vec::IntoIter<String> {
    let mut tokens = vec!["--wine", "/w", "--exe", "/e", "--arch", "i686"];
    tokens.extend_from_slice(extra);
    args(&tokens)
}

#[test]
fn parses_flags_and_update() {
    let config = parse_args(base(&["--update-baseline", "--assets", "/a"])).unwrap();
    assert!(config.update);
    assert_eq!(config.wine.to_str(), Some("/w"));
    assert_eq!(config.exe.to_str(), Some("/e"));
    assert_eq!(config.arch, Arch::I686);
    assert_eq!(
        config.assets.as_deref().and_then(std::path::Path::to_str),
        Some("/a")
    );
}

#[test]
fn variant_defaults_to_native_and_parses_intel() {
    let config = parse_args(base(&[])).unwrap();
    assert_eq!(config.variant, Variant::Native);
    let config = parse_args(base(&["--variant", "intel"])).unwrap();
    assert_eq!(config.variant, Variant::Intel);
    let err = parse_args(base(&["--variant", "amd"])).unwrap_err();
    assert!(err.contains("unknown variant"), "{err}");
}

#[test]
fn assets_is_none_when_absent() {
    let config = parse_args(base(&[])).unwrap();
    assert!(config.assets.is_none());
}

#[test]
fn each_mandatory_flag_is_required() {
    let err = parse_args(args(&["--exe", "/e", "--arch", "i686"])).unwrap_err();
    assert!(err.contains("--wine"), "{err}");
    let err = parse_args(args(&["--wine", "/w", "--arch", "i686"])).unwrap_err();
    assert!(err.contains("--exe"), "{err}");
    let err = parse_args(args(&["--wine", "/w", "--exe", "/e"])).unwrap_err();
    assert!(err.contains("--arch"), "{err}");
}

/// The retired flag must fail loudly, not be silently ignored.
#[test]
fn wine_build_is_no_longer_a_flag() {
    let err = parse_args(base(&["--wine-build", "/wb"])).unwrap_err();
    assert!(err.contains("unknown argument"), "{err}");
}

#[test]
fn unknown_flag_errors() {
    let err = parse_args(base(&["--nope"])).unwrap_err();
    assert!(err.contains("unknown argument"), "{err}");
}

#[test]
fn only_defaults_to_unset_and_repeat_one() {
    let config = parse_args(base(&[])).unwrap();
    assert!(config.only.is_none());
    assert_eq!(config.repeat, 1);
}

#[test]
fn parses_only_and_repeat() {
    let config = parse_args(base(&["--only", "device", "--repeat", "20"])).unwrap();
    assert_eq!(config.only, Some(Subtest::Device));
    assert_eq!(config.repeat, 20);
}

#[test]
fn bad_subtest_and_arch_error() {
    let err = parse_args(base(&["--only", "nope"])).unwrap_err();
    assert!(err.contains("unknown subtest"), "{err}");
    let err = parse_args(args(&["--wine", "/w", "--exe", "/e", "--arch", "arm"])).unwrap_err();
    assert!(err.contains("unknown arch"), "{err}");
}

#[test]
fn repeat_zero_errors() {
    let err = parse_args(base(&["--repeat", "0"])).unwrap_err();
    assert!(err.contains(">= 1"), "{err}");
}

#[test]
fn update_baseline_rejects_filters() {
    let err = parse_args(base(&["--update-baseline", "--only", "device"])).unwrap_err();
    assert!(err.contains("cannot be combined"), "{err}");
}
