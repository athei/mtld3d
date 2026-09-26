//! Unit tests for the metrics file parser.

use super::*;

const GOOD: &str = "\
# written by the benchmark
meta frame_shape layer v0.11.0-3-g66e4114
meta frame_shape layer_image 1A2B3C4D-0000-0000-0000-000000000001
meta frame_shape arch x86
meta frame_shape profile production
meta frame_shape debug_assertions off
meta frame_shape config shaderCache.enable=false;log.dir=Z:/tmp/x y

metric frame_shape frame.p50 16.667 ms lower time
metric frame_shape api.p50 2 ms lower noisy
metric frame_shape mem.end.committed_mib 187.5 mib lower bytes
metric frame_shape perf.draws_pf 1234 count lower exact
metric frame_shape frame.max 40.1 ms lower info
metric frame_shape fps 60 count higher time
shape frame_shape pass 0 1920x1080 draws=12 ff_vs=0 ff_ps=0 tex_per_draw=1.50
";

/// The reason `parse` gives for the one-line file `line` of benchmark `b`.
fn error_of(line: &str) -> String {
    let (at, reason) = parse(line, "b").expect_err("the line is rejected");
    assert_eq!(at, 1);
    reason
}

#[test]
fn a_good_file_reads_every_meta_and_metric() {
    let file = parse(GOOD, "frame_shape").unwrap();
    assert_eq!(file.meta["layer"], "v0.11.0-3-g66e4114");
    assert_eq!(file.meta["profile"], "production");
    assert_eq!(
        file.meta["config"], "shaderCache.enable=false;log.dir=Z:/tmp/x y",
        "a meta value is the rest of the line, spaces included"
    );
    assert_eq!(file.metrics.len(), 6);
    let p50 = &file.metrics["frame.p50"];
    assert_eq!(p50.value.to_bits(), 16.667_f64.to_bits());
    assert_eq!(p50.unit, Unit::Ms);
    assert_eq!(p50.direction, Direction::Lower);
    assert_eq!(p50.class, Class::Time);
    assert_eq!(file.metrics["fps"].direction, Direction::Higher);
    assert_eq!(file.metrics["perf.draws_pf"].class, Class::Exact);
    assert_eq!(file.metrics["mem.end.committed_mib"].unit, Unit::Mib);
    assert_eq!(
        file.shape,
        ["pass 0 1920x1080 draws=12 ff_vs=0 ff_ps=0 tex_per_draw=1.50"],
        "a shape line is kept as the text after the benchmark's name"
    );
}

#[test]
fn unknown_meta_keys_are_kept_and_not_an_error() {
    let file = parse("meta b future_key some value\n", "b").unwrap();
    assert_eq!(file.meta["future_key"], "some value");
}

#[test]
fn a_required_meta_value_may_not_be_empty() {
    assert!(error_of("meta b layer").contains("meta layer has no value"));
    assert!(parse("meta b config\n", "b").unwrap().meta["config"].is_empty());
}

#[test]
fn a_meta_key_twice_is_an_error() {
    let text = "meta b arch x86\nmeta b arch x86_64\n";
    let (at, reason) = parse(text, "b").unwrap_err();
    assert_eq!(at, 2);
    assert!(reason.contains("twice"), "{reason}");
}

#[test]
fn every_bad_metric_field_is_named() {
    let cases = [
        ("metric b frame.p50 1 ms lower", "7 fields"),
        ("metric b frame.p50 1 ms lower time extra", "7 fields"),
        ("metric b Frame.p50 1 ms lower time", "not made of"),
        ("metric b frame-p50 1 ms lower time", "not made of"),
        (
            "metric b frame.p50 fast ms lower time",
            "not a finite number",
        ),
        (
            "metric b frame.p50 inf ms lower time",
            "not a finite number",
        ),
        (
            "metric b frame.p50 NaN ms lower time",
            "not a finite number",
        ),
        (
            "metric b frame.p50 1 sec lower time",
            "unknown unit \"sec\"",
        ),
        (
            "metric b frame.p50 1 ms smaller time",
            "unknown direction \"smaller\"",
        ),
        (
            "metric b frame.p50 1 ms lower timing",
            "unknown class \"timing\"",
        ),
        (
            "metric b mem.x 1 count lower bytes",
            "class bytes needs unit mib or bytes",
        ),
    ];
    for (line, expected) in cases {
        let reason = error_of(line);
        assert!(reason.contains(expected), "{line}: {reason}");
    }
}

#[test]
fn a_metric_twice_is_an_error() {
    let text = "metric b x 1 ms lower time\nmetric b x 2 ms lower time\n";
    let (at, reason) = parse(text, "b").unwrap_err();
    assert_eq!(at, 2);
    assert!(reason.contains("twice"), "{reason}");
}

#[test]
fn a_line_naming_another_benchmark_is_an_error() {
    assert!(error_of("metric other x 1 ms lower time").contains("names benchmark \"other\""));
}

#[test]
fn an_unknown_line_kind_is_an_error() {
    assert!(error_of("metrics b x 1 ms lower time").contains("unknown line kind"));
    assert!(error_of("garbage").contains("not a meta, metric or shape line"));
}

#[test]
fn comments_blank_lines_and_shapes_are_skipped() {
    let file = parse("\n   \n# note\nshape b pass 0 64x64 draws=1\n", "b").unwrap();
    assert!(file.meta.is_empty());
    assert!(file.metrics.is_empty());
}

#[test]
fn the_benchmark_comes_from_the_file_name() {
    assert_eq!(bench_of("bench-frame_shape.metrics"), Some("frame_shape"));
    assert_eq!(bench_of("bench-.metrics"), None);
    assert_eq!(bench_of("bench-frame_shape.txt"), None);
    assert_eq!(bench_of("frame_shape.metrics"), None);
}

#[test]
fn a_read_error_names_the_file_and_the_line() {
    let dir = std::env::temp_dir().join(format!("mtld3d-metrics-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench-b.metrics");
    fs::write(&path, "meta b arch x86\nmetric b x 1 ms lower slow\n").unwrap();
    let reason = read(&path).unwrap_err();
    fs::remove_dir_all(&dir).unwrap();
    assert!(
        reason.starts_with(&format!("{}:2: ", path.display())),
        "{reason}"
    );
    assert!(reason.contains("unknown class"), "{reason}");
}
