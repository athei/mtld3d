//! Unit tests for the A/B verdicts and the checks that come before them.

use super::*;

/// A metric definition as a metrics file spells it after the value.
fn def(spec: &str) -> Metric {
    let text = format!("metric b m 0 {spec}");
    let file = metrics::parse(&text, "b").expect("the definition parses");
    file.metrics.into_values().next().expect("one metric")
}

fn verdict_of(name: &str, spec: &str, base: &[f64], cand: &[f64]) -> Verdict {
    judge(name, &def(spec), base, cand, false).verdict
}

#[test]
fn an_a_a_run_within_the_noise_is_neutral() {
    let base = [16.0, 16.3, 15.8, 16.1, 16.2];
    let cand = [16.2, 15.9, 16.1, 16.0, 16.3];
    assert_eq!(
        verdict_of("frame.p50", "ms lower time", &base, &cand),
        Verdict::Neutral
    );
}

#[test]
fn a_noisy_a_a_run_raises_its_own_threshold() {
    // Ratios 1.05, 0.95, 1.15, 1.10, 1.06: median 1.06, above the 3 % floor,
    // and four of five pairs worse, but the MAD of 0.04 puts 3 sigma at 18 %.
    let base = [100.0; 5];
    let cand = [105.0, 95.0, 115.0, 110.0, 106.0];
    assert_eq!(
        verdict_of("frame.mean", "ms lower time", &base, &cand),
        Verdict::Neutral
    );
    assert_eq!(
        verdict_of("api.p50", "ms lower noisy", &base, &cand),
        Verdict::Neutral
    );
}

#[test]
fn a_clear_five_percent_median_regression_fails() {
    let base = [10.0, 10.1, 9.9, 10.0, 10.05];
    let cand: Vec<f64> = base.iter().map(|b| b * 1.05).collect();
    let row = judge("frame.p50", &def("ms lower time"), &base, &cand, false);
    assert_eq!(row.verdict, Verdict::Regression);
    assert!(row.verdict.fails());
    assert_eq!(row.change, "+5.00%");
}

#[test]
fn a_tail_percentile_has_the_wider_floor() {
    let base = [10.0; 5];
    let cand = [10.5; 5];
    assert_eq!(
        verdict_of("frame.p99", "ms lower time", &base, &cand),
        Verdict::Neutral
    );
    let cand = [11.0; 5];
    assert_eq!(
        verdict_of("frame.p99", "ms lower time", &base, &cand),
        Verdict::Regression
    );
}

#[test]
fn a_regression_needs_four_pairs_in_five_worse() {
    let base = [100.0; 5];
    let four = [106.0, 106.0, 106.0, 106.0, 99.0];
    assert_eq!(
        verdict_of("frame.p50", "ms lower time", &base, &four),
        Verdict::Regression
    );
    // Same median, but only three pairs worse.
    let three = [106.0, 106.0, 106.0, 99.0, 99.0];
    assert_eq!(
        verdict_of("frame.p50", "ms lower time", &base, &three),
        Verdict::Neutral
    );
}

#[test]
fn an_improvement_is_the_mirror_image() {
    let base = [10.0; 5];
    let cand = [9.0; 5];
    let row = judge("frame.p50", &def("ms lower time"), &base, &cand, false);
    assert_eq!(row.verdict, Verdict::Improvement);
    assert!(!row.verdict.fails());
}

#[test]
fn a_higher_is_better_metric_is_inverted() {
    let base = [100.0; 5];
    assert_eq!(
        verdict_of("fps", "count higher time", &base, &[95.0; 5]),
        Verdict::Regression
    );
    assert_eq!(
        verdict_of("fps", "count higher time", &base, &[106.0; 5]),
        Verdict::Improvement
    );
    let row = judge("fps", &def("count higher time"), &base, &[95.0; 5], false);
    assert!(
        row.change.starts_with('+'),
        "a worse change reads +: {}",
        row.change
    );
}

#[test]
fn a_bytes_metric_also_has_to_move_four_mib() {
    let spec = "mib lower bytes";
    assert_eq!(
        verdict_of("mem.end", spec, &[100.0; 5], &[105.0; 5]),
        Verdict::Regression
    );
    // Five percent of 10 MiB is half a MiB: allocator noise.
    assert_eq!(
        verdict_of("mem.end", spec, &[10.0; 5], &[10.5; 5]),
        Verdict::Neutral
    );
    let bytes = 100.0 * 1024.0 * 1024.0;
    assert_eq!(
        verdict_of(
            "mem.bytes",
            "bytes lower bytes",
            &[bytes; 5],
            &[bytes * 1.05; 5]
        ),
        Verdict::Regression
    );
    assert_eq!(
        verdict_of("mem.end", spec, &[105.0; 5], &[100.0; 5]),
        Verdict::Improvement
    );
}

#[test]
fn spikes_are_judged_by_the_median_difference() {
    let spec = "count lower spikes";
    let base = [0.0, 1.0, 0.0, 0.0, 1.0];
    // Differences 5, 3, 6, 5, 4: median 5 against max(2, 3 * MAD 1).
    assert_eq!(
        verdict_of("frame.spikes", spec, &base, &[5.0, 4.0, 6.0, 5.0, 5.0]),
        Verdict::Regression
    );
    // Differences 1, 1, 0, 1, 0: median 1, under the floor of 2.
    assert_eq!(
        verdict_of("frame.spikes", spec, &base, &[1.0, 2.0, 0.0, 1.0, 1.0]),
        Verdict::Neutral
    );
    assert_eq!(
        verdict_of("frame.spikes", spec, &[5.0; 5], &[0.0; 5]),
        Verdict::Improvement
    );
}

#[test]
fn an_exact_change_fails_unless_accepted() {
    let spec = def("count lower exact");
    let base = [100.0; 5];
    assert_eq!(
        judge("perf.draws_pf", &spec, &base, &base, false).verdict,
        Verdict::Neutral
    );

    let worse = judge("perf.draws_pf", &spec, &base, &[101.0; 5], false);
    assert_eq!(
        worse.verdict,
        Verdict::Changed {
            worse: true,
            accepted: false
        }
    );
    assert!(worse.verdict.fails());
    assert_eq!(worse.noise, "5/5 pairs differ");

    let better = judge("perf.draws_pf", &spec, &base, &[99.0; 5], false);
    assert_eq!(
        better.verdict,
        Verdict::Changed {
            worse: false,
            accepted: false
        }
    );
    assert!(
        better.verdict.fails(),
        "a change the workload fixes fails either way"
    );

    let accepted = judge("perf.draws_pf", &spec, &base, &[101.0; 5], true);
    assert!(!accepted.verdict.fails());
    assert_eq!(accepted.verdict.label(), "changed, accepted");
}

#[test]
fn one_differing_pair_is_an_exact_change() {
    let spec = def("count lower exact");
    let row = judge(
        "perf.passes_pf",
        &spec,
        &[4.0; 5],
        &[4.0, 4.0, 5.0, 4.0, 4.0],
        false,
    );
    assert!(row.verdict.fails());
    assert_eq!(row.noise, "1/5 pairs differ");
}

#[test]
fn an_info_metric_is_reported_and_never_judged() {
    let row = judge(
        "frame.max",
        &def("ms lower info"),
        &[10.0; 5],
        &[50.0; 5],
        false,
    );
    assert_eq!(row.verdict, Verdict::Info);
    assert!(!row.verdict.fails());
    assert_eq!(row.base, "10 ms");
    assert_eq!(row.cand, "50 ms");
}

#[test]
fn a_zero_base_is_infinitely_worse_and_two_zeros_are_equal() {
    let spec = "ms lower time";
    assert_eq!(
        verdict_of("frame.p50", spec, &[0.0; 5], &[0.0; 5]),
        Verdict::Neutral
    );
    assert_eq!(
        verdict_of("frame.p50", spec, &[0.0; 5], &[1.0; 5]),
        Verdict::Regression
    );
}

/// An A/B directory under the system's temporary directory, removed when dropped.
struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("mtld3d-bench-compare-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    /// Write `bench-<bench>.metrics` into `<leg>/<round>` with the given meta and metrics.
    fn write(
        &self,
        leg: &str,
        round: usize,
        bench: &str,
        meta: &[(&str, &str)],
        metrics: &[(&str, f64, &str)],
    ) {
        let dir = self.root.join(leg).join(round.to_string());
        fs::create_dir_all(&dir).unwrap();
        let mut text = String::new();
        for (key, value) in meta {
            let _ = writeln!(text, "meta {bench} {key} {value}");
        }
        for (name, value, spec) in metrics {
            let _ = writeln!(text, "metric {bench} {name} {value} {spec}");
        }
        fs::write(dir.join(format!("bench-{bench}.metrics")), text).unwrap();
    }

    /// Write a well-formed run of `rounds` rounds for both legs, `cand` scaling frame.p50.
    fn standard(&self, rounds: usize, scale: f64) {
        for round in 0..rounds {
            let jitter = [0.0, 0.05, -0.05, 0.02, -0.02][round % 5];
            self.write(
                "base",
                round,
                "frame_shape",
                &meta("v0.11.0-3-g66e4114", "AAAA"),
                &[
                    ("frame.p50", 10.0 + jitter, "ms lower time"),
                    ("perf.draws_pf", 500.0, "count lower exact"),
                ],
            );
            self.write(
                "cand",
                round,
                "frame_shape",
                &meta("v0.11.0-3-g66e4114", "BBBB"),
                &[
                    ("frame.p50", (10.0 + jitter) * scale, "ms lower time"),
                    ("perf.draws_pf", 500.0, "count lower exact"),
                ],
            );
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn meta<'a>(layer: &'a str, image: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![
        ("layer", layer),
        ("layer_image", image),
        ("arch", "x86"),
        ("profile", "production"),
        ("debug_assertions", "off"),
        ("config", "shaderCache.enable=false"),
    ]
}

fn error_of(fixture: &Fixture) -> String {
    evaluate(&fixture.root, &Options::default()).expect_err("the directory is rejected")
}

#[test]
fn a_regression_in_a_directory_exits_one() {
    let fixture = Fixture::new("regression");
    fixture.standard(5, 1.05);
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(comparison.failed());
    let report = comparison.render();
    assert!(report.contains("REGRESSION"), "{report}");
    assert!(report.contains("bench-compare: FAIL"), "{report}");
    assert_eq!(
        judge_dir(&fixture.root, &Options::default(), None).unwrap(),
        ExitCode::from(1)
    );
}

#[test]
fn a_neutral_directory_exits_zero_and_writes_its_report() {
    let fixture = Fixture::new("neutral");
    fixture.standard(5, 1.0);
    let report = fixture.root.join("out").join("report.txt");
    assert_eq!(
        judge_dir(&fixture.root, &Options::default(), Some(&report)).unwrap(),
        ExitCode::SUCCESS
    );
    let text = fs::read_to_string(&report).unwrap();
    assert!(text.contains("bench-compare: PASS"), "{text}");
    assert!(text.contains("== frame_shape"), "{text}");
}

#[test]
fn different_profiles_cannot_be_compared() {
    let fixture = Fixture::new("profile");
    fixture.standard(3, 1.0);
    let mut release = meta("v0.11.0-3-g66e4114", "BBBB");
    release[3] = ("profile", "release");
    fixture.write(
        "cand",
        1,
        "frame_shape",
        &release,
        &[
            ("frame.p50", 10.0, "ms lower time"),
            ("perf.draws_pf", 500.0, "count lower exact"),
        ],
    );
    let reason = error_of(&fixture);
    assert!(reason.contains("did not run one build"), "{reason}");

    let fixture = Fixture::new("profile-legs");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "AAAA"),
            &[("x", 1.0, "ms lower time")],
        );
        let mut debug = meta("v2", "BBBB");
        debug[4] = ("debug_assertions", "on");
        fixture.write("cand", round, "b", &debug, &[("x", 1.0, "ms lower time")]);
    }
    let reason = error_of(&fixture);
    assert!(reason.contains("meta debug_assertions"), "{reason}");
}

#[test]
fn a_leg_that_changed_its_layer_mid_run_is_rejected() {
    let fixture = Fixture::new("layer");
    fixture.standard(3, 1.0);
    fixture.write(
        "base",
        2,
        "frame_shape",
        &meta("v0.11.0-4-gdeadbee", "AAAA"),
        &[
            ("frame.p50", 10.0, "ms lower time"),
            ("perf.draws_pf", 500.0, "count lower exact"),
        ],
    );
    let reason = error_of(&fixture);
    assert!(
        reason.contains("the base leg did not run one build: meta layer"),
        "{reason}"
    );
}

#[test]
fn one_image_in_both_legs_is_rejected_and_one_stamp_is_not() {
    let fixture = Fixture::new("image");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v1", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("both legs loaded d3d9.dll image SAME"),
        "{reason}"
    );

    // The standard fixture has one stamp and two images: a dirty candidate on its base commit.
    let fixture = Fixture::new("stamp");
    fixture.standard(3, 1.0);
    assert!(evaluate(&fixture.root, &Options::default()).is_ok());
}

#[test]
fn an_unknown_image_is_a_note_not_an_error() {
    let fixture = Fixture::new("unknown-image");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "unknown"),
            &[("x", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v2", "unknown"),
            &[("x", 1.0, "ms lower time")],
        );
    }
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note.contains("no image ID"))
    );
}

#[test]
fn a_missing_meta_line_is_rejected() {
    let fixture = Fixture::new("no-meta");
    fixture.standard(2, 1.0);
    fixture.write(
        "cand",
        0,
        "frame_shape",
        &[("layer", "v1")],
        &[
            ("frame.p50", 10.0, "ms lower time"),
            ("perf.draws_pf", 500.0, "count lower exact"),
        ],
    );
    let reason = error_of(&fixture);
    assert!(reason.contains("no meta layer_image line"), "{reason}");
}

#[test]
fn legs_with_different_round_counts_are_rejected() {
    let fixture = Fixture::new("rounds");
    fixture.standard(3, 1.0);
    fs::remove_dir_all(fixture.root.join("cand").join("2")).unwrap();
    let reason = error_of(&fixture);
    assert!(reason.contains("mismatched rounds"), "{reason}");
    assert!(reason.contains("base has 3, cand has 2"), "{reason}");
}

#[test]
fn rounds_with_a_gap_are_rejected() {
    let fixture = Fixture::new("gap");
    fixture.standard(3, 1.0);
    fs::rename(
        fixture.root.join("base").join("2"),
        fixture.root.join("base").join("7"),
    )
    .unwrap();
    let reason = error_of(&fixture);
    assert!(
        reason.contains("the rounds are 0, 1, 7, not 0..3"),
        "{reason}"
    );
}

#[test]
fn a_benchmark_missing_from_one_round_is_rejected() {
    let fixture = Fixture::new("missing-bench");
    fixture.standard(3, 1.0);
    for round in [0, 2] {
        fixture.write(
            "cand",
            round,
            "other",
            &meta("v0.11.0-3-g66e4114", "BBBB"),
            &[("x", 1.0, "ms lower time")],
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("cand/1 has no bench-other.metrics"),
        "{reason}"
    );
}

#[test]
fn a_metric_that_comes_and_goes_within_a_leg_is_rejected() {
    let fixture = Fixture::new("flaky-metric");
    fixture.standard(3, 1.0);
    fixture.write(
        "base",
        1,
        "frame_shape",
        &meta("v0.11.0-3-g66e4114", "AAAA"),
        &[("frame.p50", 10.0, "ms lower time")],
    );
    let reason = error_of(&fixture);
    assert!(
        reason.contains("metric perf.draws_pf is in one of"),
        "{reason}"
    );
}

#[test]
fn a_metric_redefined_between_legs_is_rejected() {
    let fixture = Fixture::new("redefined");
    for round in 0..2 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "AAAA"),
            &[("x", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v2", "BBBB"),
            &[("x", 1.0, "us lower time")],
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("metric x is ms lower time in base and us lower time in cand"),
        "{reason}"
    );
}

#[test]
fn added_and_removed_metrics_and_benchmarks_are_listed_not_failed() {
    let fixture = Fixture::new("added");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "AAAA"),
            &[("x", 1.0, "ms lower time"), ("gone", 2.0, "ms lower time")],
        );
        fixture.write(
            "base",
            round,
            "old",
            &meta("v1", "AAAA"),
            &[("y", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v2", "BBBB"),
            &[
                ("x", 1.0, "ms lower time"),
                ("new", 3.0, "count lower exact"),
            ],
        );
    }
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(!comparison.failed());
    let b = comparison
        .benches
        .iter()
        .find(|bench| bench.bench == "b")
        .unwrap();
    let verdict = |name: &str| {
        &b.rows
            .iter()
            .find(|row| row.metric == name)
            .unwrap()
            .verdict
    };
    assert_eq!(*verdict("gone"), Verdict::Removed);
    assert_eq!(*verdict("new"), Verdict::Added);
    let old = comparison
        .benches
        .iter()
        .find(|bench| bench.bench == "old")
        .unwrap();
    assert_eq!(old.only, Some(Leg::Base));
    let summary = comparison.summary();
    assert!(summary.contains("1 added, 2 removed"), "{summary}");
}

#[test]
fn accepting_an_exact_change_by_name_passes_the_directory() {
    let fixture = Fixture::new("accept");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "AAAA"),
            &[("perf.draws_pf", 500.0, "count lower exact")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v2", "BBBB"),
            &[("perf.draws_pf", 510.0, "count lower exact")],
        );
    }
    assert!(
        evaluate(&fixture.root, &Options::default())
            .unwrap()
            .failed()
    );
    let options = Options {
        accept: vec!["perf.draws_pf".to_owned(), "perf.unused".to_owned()],
        ..Options::default()
    };
    let comparison = evaluate(&fixture.root, &options).unwrap();
    assert!(!comparison.failed());
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note.contains("--accept perf.unused")),
        "{:?}",
        comparison.notes
    );
}

#[test]
fn a_malformed_metrics_file_is_an_error_naming_its_line() {
    let fixture = Fixture::new("malformed");
    fixture.standard(2, 1.0);
    let path = fixture
        .root
        .join("cand")
        .join("1")
        .join("bench-frame_shape.metrics");
    fs::write(&path, "metric frame_shape frame.p50 1 ms lower slow\n").unwrap();
    let reason = error_of(&fixture);
    assert!(
        reason.starts_with(&format!("{}:1: ", path.display())),
        "{reason}"
    );
}

#[test]
fn a_clean_a_a_run_may_load_one_image_in_both_legs() {
    let fixture = Fixture::new("aa-image");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v1", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
    }
    let allowed = Options {
        allow_same_image: true,
        ..Options::default()
    };
    let comparison = evaluate(&fixture.root, &allowed).unwrap();
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note == "legs loaded identical binaries (A/A)"),
        "{:?}",
        comparison.notes
    );
    // The file `bench-ab` leaves allows it for a later `bench-compare` too.
    assert!(evaluate(&fixture.root, &Options::default()).is_err());
    fs::write(fixture.root.join(SAME_IMAGE_FILE), "").unwrap();
    assert!(evaluate(&fixture.root, &Options::default()).is_ok());
}

#[test]
fn one_image_under_two_stamps_stays_an_error_even_when_allowed() {
    let fixture = Fixture::new("aa-stamps");
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "b",
            &meta("v1", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
        fixture.write(
            "cand",
            round,
            "b",
            &meta("v2", "SAME"),
            &[("x", 1.0, "ms lower time")],
        );
    }
    let allowed = Options {
        allow_same_image: true,
        ..Options::default()
    };
    let reason = evaluate(&fixture.root, &allowed).unwrap_err();
    assert!(
        reason.contains("both legs loaded d3d9.dll image SAME"),
        "{reason}"
    );
}
