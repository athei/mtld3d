//! Unit tests for `RUST_LOG` filter composition.
//!
//! Logger init keeps an `info` baseline and appends whatever the user asked for, relying
//! on `env_logger` giving the last matching spec priority so a bare level or a per-module
//! override still takes effect. These cases pin the composed string for an unset or empty
//! variable, a bare level, root and sub-namespace targets, and a multi-spec passthrough.

use super::resolved_log_filter;

#[test]
fn unset_defaults_to_info() {
    assert_eq!(resolved_log_filter(None), "info");
}

#[test]
fn empty_string_defaults_to_info() {
    assert_eq!(resolved_log_filter(Some("")), "info");
}

#[test]
fn bare_level_wins_via_last_spec() {
    assert_eq!(resolved_log_filter(Some("warn")), "info,warn");
}

#[test]
fn root_spec_composes() {
    assert_eq!(resolved_log_filter(Some("mtld3d=warn")), "info,mtld3d=warn");
}

#[test]
fn sub_namespace_override_restores_baseline() {
    assert_eq!(
        resolved_log_filter(Some("mtld3d::perf=debug")),
        "info,mtld3d::perf=debug"
    );
}

#[test]
fn multi_spec_passthrough() {
    assert_eq!(
        resolved_log_filter(Some("mtld3d=warn,mtld3d::dxso=trace")),
        "info,mtld3d=warn,mtld3d::dxso=trace",
    );
}
