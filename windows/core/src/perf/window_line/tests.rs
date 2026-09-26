//! Unit tests for reading a window's length from the layer log's perf lines.

use super::*;

/// Whether `line` names a window of `secs` seconds, to within a microsecond.
fn names(line: &str, secs: f64) -> bool {
    window_secs(line).is_some_and(|read| (read - secs).abs() < 1e-6)
}

#[test]
fn the_perf_kv_line_names_its_window() {
    let line = "[2026-09-26T06:36:30Z INFO  mtld3d::perf] perf-kv v1 window_s=2.004 frames=2500 \
                frame_ms=0.800";
    assert!(names(line, 2.004));
}

#[test]
fn the_grid_header_names_its_window_in_builds_without_the_perf_kv_line() {
    let plain = "[2026-09-26T06:36:30Z INFO  mtld3d::perf] encoder=ThreadId(4) ── perf  \
                 window=5.00s  frames=6381  bottleneck=API (D3D9) ──";
    assert!(names(plain, 5.0));
    let bold = "encoder=ThreadId(4) \u{1b}[1m── perf  window=2.01s  frames=1  bottleneck=ENCODER";
    assert!(names(bold, 2.01));
}

#[test]
fn other_lines_and_broken_lengths_name_no_window() {
    for line in [
        "[2026-09-26T06:36:30Z INFO  mtld3d::perf] pagebox-pool cumulative: hit=1",
        "perf-kv v1 frames=3 frame_ms=0.8",
        "perf-kv v1 window_s=abc frames=3",
        "── perf  window=s  frames=1",
        "perf-kv v1 window_s=0.000 frames=0",
    ] {
        assert!(window_secs(line).is_none(), "{line}");
    }
}

#[cfg(perf_tracking)]
#[test]
fn the_lines_the_layer_writes_read_back() {
    let kv = super::super::KvLine::new(2.004, 3).finish();
    assert!(names(&kv, 2.004));
}
