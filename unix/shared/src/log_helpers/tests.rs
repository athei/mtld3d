//! Unit tests for the log-once macros and their per-key latch.
//!
//! `first_seen` mirrors the first-seen check the `_by` macros run, so dedup semantics can
//! be asserted without installing a `log` sink: a repeated key stays silent, distinct keys
//! (including packed multi-field ones) each get their line. The second test never asserts
//! anything, it only has to compile, which catches regressions in macro argument parsing
//! across the call shapes the tree actually uses.

use std::{collections::BTreeSet, sync::Mutex};

use super::first_seen;

#[test]
fn first_seen_dedups_repeat_keys_and_separates_distinct() {
    let seen: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
    assert!(first_seen(&seen, 42));
    assert!(!first_seen(&seen, 42));
    assert!(first_seen(&seen, 43));
    assert!(!first_seen(&seen, 43));
    // Packed two-u8 key, matching the resolve_attrs_for_ff site.
    let pack = |u: u8, i: u8| (u64::from(u) << 8) | u64::from(i);
    assert!(first_seen(&seen, pack(1, 0))); // BLENDWEIGHT
    assert!(first_seen(&seen, pack(2, 0))); // BLENDINDICES
    assert!(!first_seen(&seen, pack(1, 0)));
}

// Compile-only: the macros must accept the call shapes the codebase
// uses. If macro arg parsing regresses, this fails to build.
#[test]
fn macros_compile_with_typical_shapes() {
    crate::log_once_warn!(target: "t", "no args");
    crate::log_once_warn!(target: "t", "fmt {} {}", 1, 2);
    crate::log_once_warn_by!(target: "t", key: 0u64, "no args");
    crate::log_once_warn_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
    let state: u32 = 257;
    crate::log_once_warn_by!(
        target: "t",
        key: u64::from(state),
        "SetTransform: D3DTS_{state} not honoured — value dropped"
    );
    crate::log_once_trace_by!(target: "t", key: 0u64, "no args");
    crate::log_once_trace_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
    crate::log_once_trace_by!(
        target: "t",
        key: u64::from(state),
        "drop: VS {state:#x} did not resolve"
    );
    crate::log_once_debug_by!(target: "t", key: 0u64, "no args");
    crate::log_once_debug_by!(target: "t", key: 7u64, "fmt {} {}", 1, 2);
    crate::log_once_debug_by!(
        target: "t",
        key: u64::from(state),
        "CheckDeviceFormat OK {state:#x}"
    );
    crate::log_once_info!(target: "t", "no args");
    crate::log_once_info!(target: "t", "fmt {} {}", 1, 2);
    crate::log_once_info_by!(target: "t", key: 0u64, "no args");
    crate::log_once_info_by!(
        target: "t",
        key: u64::from(state),
        "SetSoftwareVertexProcessing({state}): obsolete"
    );
}
