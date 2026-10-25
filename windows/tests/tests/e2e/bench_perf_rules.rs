//! The classes the benchmark writer gives the layer's `perf-kv` keys, without a device.

use super::bench::{Class, FrameWork, perf_rule};

/// The class [`perf_rule`] gives `key` over frames of `work`.
fn class_of(key: &str, work: &FrameWork) -> Class {
    perf_rule(key, work)
        .unwrap_or_else(|| panic!("{key} has a known suffix"))
        .class
}

#[test]
fn sampled_snapshot_counts_are_info_and_unsampled_rebuilds_stay_noisy() {
    for work in [FrameWork::Fixed, FrameWork::Varying] {
        for key in [
            "draw_snapshot_sampled_draws_total",
            "draw_snapshot_sampled_rebuild_ps_source_total",
        ] {
            assert!(matches!(class_of(key, &work), Class::Info), "{key}");
        }
        for key in [
            "draw_snapshot_rebuild_draws_total",
            "draw_snapshot_rebuild_ps_source_total",
        ] {
            assert!(matches!(class_of(key, &work), Class::Noisy), "{key}");
        }
    }
}
