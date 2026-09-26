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
fn a_clear_difference_on_a_zero_base_regresses_and_two_zeros_are_equal() {
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
fn a_metric_that_comes_and_goes_within_a_leg_is_reported_not_judged() {
    // Round 1 of the base leaves perf.draws_pf out, as a window without a
    // fault sample leaves out perf.faults_major_pf.
    let fixture = Fixture::new("flaky-metric");
    fixture.standard(3, 1.0);
    fixture.write(
        "base",
        1,
        "frame_shape",
        &meta("v0.11.0-3-g66e4114", "AAAA"),
        &[("frame.p50", 10.0, "ms lower time")],
    );
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    let row = comparison
        .rows()
        .find(|row| row.metric == "perf.draws_pf")
        .expect("the row is reported");
    assert_eq!(row.verdict, Verdict::Incomplete);
    assert!(!row.verdict.fails());
    assert_eq!(row.change, "in 2 of 3 base rounds, 3 of 3 cand rounds");
    assert!(!comparison.failed());
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note.contains("frame_shape: perf.draws_pf is in some rounds")),
        "{:?}",
        comparison.notes
    );
    assert!(
        comparison.summary().contains("1 incomplete"),
        "{}",
        comparison.summary()
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
fn added_and_removed_metrics_are_listed_not_failed() {
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
    let rows = &comparison.benches[0].rows;
    let verdict = |name: &str| &rows.iter().find(|row| row.metric == name).unwrap().verdict;
    assert_eq!(*verdict("gone"), Verdict::Removed);
    assert_eq!(*verdict("new"), Verdict::Added);
    let summary = comparison.summary();
    assert!(summary.contains("1 added, 1 removed"), "{summary}");
}

#[test]
fn a_benchmark_only_one_leg_ran_is_an_incomplete_run() {
    let fixture = Fixture::new("one-leg-bench");
    fixture.standard(3, 1.0);
    for round in 0..3 {
        fixture.write(
            "base",
            round,
            "old",
            &meta("v0.11.0-3-g66e4114", "AAAA"),
            &[("y", 1.0, "ms lower time")],
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("incomplete run: benchmark old ran only in the base leg"),
        "{reason}"
    );
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

#[test]
fn a_zero_base_is_judged_by_the_absolute_difference() {
    let spec = "ms lower time";
    // Under the 0.1 ms floor: reported, not failed.
    let row = judge("frame.stall", &def(spec), &[0.0; 5], &[0.05; 5], false);
    assert_eq!(row.verdict, Verdict::Neutral);
    assert!(row.change.contains("zero base"), "{}", row.change);
    // Past it, on every pair.
    assert_eq!(
        verdict_of("frame.stall", spec, &[0.0; 5], &[0.5; 5]),
        Verdict::Regression
    );
    assert_eq!(
        verdict_of("frame.stall", spec, &[0.0; 5], &[0.0; 5]),
        Verdict::Neutral
    );
    // The floor is in the metric's unit.
    assert_eq!(
        verdict_of("frame.stall", "us lower time", &[0.0; 5], &[50.0; 5]),
        Verdict::Neutral
    );
    assert_eq!(
        verdict_of("frame.stall", "us lower time", &[0.0; 5], &[500.0; 5]),
        Verdict::Regression
    );
}

#[test]
fn a_zero_base_pair_leaves_no_nan_in_the_verdict() {
    // One base round of zero: its ratio is infinite, the sigma of the rest is finite.
    let base = [0.0, 10.0, 10.0, 10.0, 10.0];
    let row = judge("frame.p50", &def("ms lower time"), &base, &[10.0; 5], false);
    assert_eq!(row.verdict, Verdict::Neutral);
    assert!(!row.noise.contains("NaN"), "{}", row.noise);
    assert!(!row.change.contains("NaN"), "{}", row.change);
    let base = [0.0, 0.0, 0.0, 10.0, 10.0];
    let row = judge("frame.p50", &def("ms lower time"), &base, &[10.0; 5], false);
    assert!(!row.noise.contains("NaN"), "{}", row.noise);
}

#[test]
fn a_bytes_metric_keeps_the_narrow_floor_whatever_its_name() {
    // A 5 % rise of 100 MiB: under the 8 % tail floor, over the 3 % one.
    let spec = "mib lower bytes";
    assert_eq!(
        verdict_of("mem.p99", spec, &[100.0; 5], &[105.0; 5]),
        Verdict::Regression
    );
    assert_eq!(
        verdict_of("frame.p99", "ms lower time", &[100.0; 5], &[105.0; 5]),
        Verdict::Neutral
    );
}

/// The standard meta with `layer_unix_image` set to `unix`.
fn meta_unix<'a>(layer: &'a str, image: &'a str, unix: &'a str) -> Vec<(&'a str, &'a str)> {
    let mut meta = meta(layer, image);
    meta.push(("layer_unix_image", unix));
    meta
}

#[test]
fn the_unix_image_follows_the_d3d9_image_rules() {
    let write = |fixture: &Fixture, base: &[(&str, &str)], cand: &[(&str, &str)]| {
        for round in 0..3 {
            fixture.write("base", round, "b", base, &[("x", 1.0, "ms lower time")]);
            fixture.write("cand", round, "b", cand, &[("x", 1.0, "ms lower time")]);
        }
    };
    let fixture = Fixture::new("unix-image-ok");
    write(
        &fixture,
        &meta_unix("v1", "AAAA", "U1"),
        &meta_unix("v1", "BBBB", "U2"),
    );
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(
        comparison
            .header
            .iter()
            .any(|line| line.contains("mtld3d.so U1 / U2"))
    );

    let fixture = Fixture::new("unix-image-same");
    write(
        &fixture,
        &meta_unix("v1", "AAAA", "U1"),
        &meta_unix("v1", "BBBB", "U1"),
    );
    let reason = error_of(&fixture);
    assert!(
        reason.contains("both legs loaded mtld3d.so image U1"),
        "{reason}"
    );
    let allowed = Options {
        allow_same_image: true,
        ..Options::default()
    };
    assert!(
        evaluate(&fixture.root, &allowed).is_ok(),
        "an A/A run may load one image"
    );

    let fixture = Fixture::new("unix-image-one-leg");
    write(
        &fixture,
        &meta_unix("v1", "AAAA", "U1"),
        &meta("v1", "BBBB"),
    );
    let reason = error_of(&fixture);
    assert!(
        reason.contains("meta layer_unix_image is in one leg's metrics only"),
        "{reason}"
    );

    let fixture = Fixture::new("unix-image-changed");
    write(
        &fixture,
        &meta_unix("v1", "AAAA", "U1"),
        &meta_unix("v1", "BBBB", "U2"),
    );
    fixture.write(
        "cand",
        2,
        "b",
        &meta_unix("v1", "BBBB", "U3"),
        &[("x", 1.0, "ms lower time")],
    );
    let reason = error_of(&fixture);
    assert!(
        reason.contains("the cand leg did not run one build: meta layer_unix_image"),
        "{reason}"
    );
}

/// The meta of a host benchmark file: its own arch and profile, and its binary's image.
fn meta_host<'a>(layer: &'a str, image: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![
        ("kind", "host"),
        ("layer", layer),
        ("host_image", image),
        ("arch", "aarch64"),
        ("profile", "production"),
        ("debug_assertions", "off"),
    ]
}

/// Three rounds of `frame_shape` beside a host benchmark with the given base and cand meta.
fn with_host(tag: &str, base: &[(&str, &str)], cand: &[(&str, &str)]) -> Fixture {
    let fixture = Fixture::new(tag);
    fixture.standard(3, 1.0);
    for round in 0..3 {
        let metrics = [
            ("emit.us_per_shader", 3.0, "us lower time"),
            ("msl.bytes_total", 4096.0, "bytes lower exact"),
        ];
        fixture.write("base", round, "host_emit_synthetic_ff", base, &metrics);
        fixture.write("cand", round, "host_emit_synthetic_ff", cand, &metrics);
    }
    fixture
}

#[test]
fn host_benchmarks_are_checked_apart_from_the_layers_benchmarks() {
    let fixture = with_host(
        "host-ok",
        &meta_host("v0.11.0-3-g66e4114", "H1"),
        &meta_host("v0.11.0-3-g66e4114", "H2"),
    );
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(!comparison.failed(), "{}", comparison.render());
    let report = comparison.render();
    assert!(
        report.contains("images (base / cand): d3d9.dll AAAA / BBBB"),
        "{report}"
    );
    assert!(
        report.contains(
            "host benchmarks: base v0.11.0-3-g66e4114   cand v0.11.0-3-g66e4114; \
                         images (base / cand): emit_corpus H1 / H2; production profile, \
                         debug assertions off, aarch64"
        ),
        "{report}"
    );
    assert!(report.contains("msl.bytes_total"), "{report}");
}

#[test]
fn one_host_image_in_both_legs_is_a_note_not_an_error() {
    let fixture = with_host("host-same", &meta_host("v1", "H1"), &meta_host("v1", "H1"));
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note.contains("both legs ran emit_corpus image H1")),
        "{:?}",
        comparison.notes
    );
}

#[test]
fn host_builds_must_match_across_legs_and_appear_in_both() {
    let mut other = meta_host("v1", "H2");
    other[4] = ("profile", "release");
    let fixture = with_host("host-profile", &meta_host("v1", "H1"), &other);
    let reason = error_of(&fixture);
    assert!(
        reason.contains("meta profile is \"production\" in base"),
        "{reason}"
    );

    let fixture = Fixture::new("host-one-leg");
    fixture.standard(3, 1.0);
    for round in 0..3 {
        let metrics = [("emit.us_per_shader", 3.0, "us lower time")];
        fixture.write(
            "cand",
            round,
            "host_emit_x",
            &meta_host("v1", "H1"),
            &metrics,
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("the cand leg has host benchmark files and the base leg none"),
        "{reason}"
    );
}

#[test]
fn a_file_of_an_unknown_kind_is_rejected() {
    let mut odd = meta_host("v1", "H1");
    odd[0] = ("kind", "gpu");
    let fixture = with_host("host-kind", &odd, &odd);
    let reason = error_of(&fixture);
    assert!(reason.contains("meta kind \"gpu\" is no kind"), "{reason}");
}

#[test]
fn a_cache_corpus_only_one_leg_could_read_is_skipped_with_a_note() {
    let fixture = with_host(
        "host-corpus",
        &meta_host("v1", "H1"),
        &meta_host("v1", "H2"),
    );
    let mut corpus = meta_host("v1", "H1");
    corpus.push(("corpus", "game"));
    for round in 0..3 {
        let metrics = [("parse_emit.us_per_shader", 9.0, "us lower time")];
        fixture.write("base", round, "host_emit_game", &corpus, &metrics);
        let ready = [("ready.p50", 9.0, "ms lower time")];
        fixture.write(
            "base",
            round,
            "cold_start_game",
            &corpus_meta_layer(),
            &ready,
        );
    }
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(!comparison.failed(), "{}", comparison.render());
    assert!(
        comparison
            .benches
            .iter()
            .all(|bench| bench.bench != "host_emit_game"),
        "{}",
        comparison.render()
    );
    for bench in ["host_emit_game", "cold_start_game"] {
        assert!(
            comparison
                .notes
                .iter()
                .any(|note| note.starts_with(&format!("{bench} skipped: only the base leg"))),
            "{:?}",
            comparison.notes
        );
    }
}

#[test]
fn a_synthetic_host_corpus_in_one_leg_is_still_an_incomplete_run() {
    let fixture = Fixture::new("host-synthetic-one-leg");
    fixture.standard(3, 1.0);
    for round in 0..3 {
        let metrics = [("emit.us_per_shader", 3.0, "us lower time")];
        for leg in ["base", "cand"] {
            fixture.write(
                leg,
                round,
                "host_emit_synthetic_ff",
                &meta_host("v1", "H1"),
                &metrics,
            );
        }
        fixture.write(
            "cand",
            round,
            "host_emit_synthetic_sm",
            &meta_host("v1", "H1"),
            &metrics,
        );
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("host_emit_synthetic_sm ran only in the cand leg"),
        "{reason}"
    );
}

/// The standard layer meta of a cold-start file over the cache `game`.
fn corpus_meta_layer() -> Vec<(&'static str, &'static str)> {
    let mut meta = meta("v0.11.0-3-g66e4114", "AAAA");
    meta.push(("corpus", "game"));
    meta
}

/// Three rounds of `frame_shape` whose workload meta `key` is `base` and `cand` in the two legs.
fn with_workload(tag: &str, key: &'static str, base: &'static str, cand: &'static str) -> Fixture {
    let fixture = Fixture::new(tag);
    for round in 0..3 {
        for (leg, image, value) in [("base", "AAAA", base), ("cand", "BBBB", cand)] {
            let mut meta = meta("v0.11.0-3-g66e4114", image);
            meta.push((key, value));
            meta.push(("tsc_hz", if round == 1 { "1000" } else { "999" }));
            let metrics = [("frame.p50", 10.0, "ms lower time")];
            fixture.write(leg, round, "frame_shape", &meta, &metrics);
        }
    }
    fixture
}

#[test]
fn a_workload_meta_that_differs_between_the_legs_is_rejected() {
    let reason = error_of(&with_workload(
        "depth-path",
        "depth_path",
        "d24x8",
        "dxt_standin",
    ));
    assert!(reason.contains("meta depth_path is \"d24x8\""), "{reason}");
    assert!(reason.contains("\"dxt_standin\""), "{reason}");
    assert!(
        reason.contains("the legs ran different workloads"),
        "{reason}"
    );

    let fixture = with_workload(
        "entries",
        "config_entries",
        "none",
        "query.eventImmediate=true",
    );
    let reason = error_of(&fixture);
    assert!(reason.contains("meta config_entries"), "{reason}");
}

#[test]
fn run_meta_may_differ_and_a_matching_workload_passes() {
    let fixture = with_workload("same-workload", "depth_path", "d24x8", "d24x8");
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(!comparison.failed(), "{}", comparison.render());
}

#[test]
fn a_workload_meta_that_changes_within_a_leg_is_rejected() {
    let fixture = with_workload("within", "depth_path", "d24x8", "d24x8");
    let mut meta = meta("v0.11.0-3-g66e4114", "AAAA");
    meta.push(("depth_path", "dxt_standin"));
    let metrics = [("frame.p50", 10.0, "ms lower time")];
    fixture.write("base", 2, "frame_shape", &meta, &metrics);
    let reason = error_of(&fixture);
    assert!(
        reason.contains("the base leg changed it between rounds"),
        "{reason}"
    );
}

/// A two-submission pass trace whose first pass stores its colour target, or discards it.
fn shape_log(color_store_dropped: bool) -> String {
    let prefix = "[2026-09-26T00:00:00Z TRACE mtld3d::d3d9::passes]";
    let open = format!(
        "{prefix} pass-open  idx=0 color=0x10 srgb=0x0 depth=0x0 size=64x64 color_load=Load \
         depth_load=DontCare viewport=0,0+64x64 extra=0x0\n"
    );
    let store = if color_store_dropped {
        format!("{prefix} pass-store idx=0 color=0x10 → DontCare (last-use)\n")
    } else {
        String::new()
    };
    format!(
        "{open}{prefix} pass-close idx=0 caller=submit color=0x10 depth=0x0 cmds=3 draws=1\n{store}{open}"
    )
}

#[test]
fn a_changed_pass_shape_fails_the_directory_unless_accepted() {
    let fixture = Fixture::new("shape");
    fixture.standard(3, 1.0);
    fixture.declare_shape();
    for (leg, dropped) in [("base", false), ("cand", true)] {
        fixture.shape_run(leg, &shape_log(dropped));
    }
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(comparison.failed());
    let report = comparison.render();
    assert!(
        report.contains("  pass #0: color_store store -> dontcare (last-use)\nSHAPE CHANGE"),
        "{report}"
    );
    assert!(
        comparison
            .summary()
            .ends_with("1 of 1 shapes changed (0 accepted)"),
        "{}",
        comparison.summary()
    );

    let options = Options {
        accept: vec!["shape:e2e.frame_shape".to_owned()],
        ..Options::default()
    };
    let comparison = evaluate(&fixture.root, &options).unwrap();
    assert!(!comparison.failed(), "{}", comparison.render());
    assert!(
        !comparison
            .notes
            .iter()
            .any(|note| note.contains("--accept")),
        "a shape name is not reported as an unmatched metric: {:?}",
        comparison.notes
    );
}

impl Fixture {
    /// Add a `shape` line to every round's `bench-frame_shape.metrics`.
    fn declare_shape(&self) {
        for leg in ["base", "cand"] {
            for round in fs::read_dir(self.root.join(leg)).unwrap() {
                let path = round.unwrap().path().join("bench-frame_shape.metrics");
                let mut text = fs::read_to_string(&path).unwrap();
                text.push_str(
                    "shape frame_shape pass 0 64x64 draws=1 ff_vs=0 ff_ps=0 tex_per_draw=0\n",
                );
                fs::write(&path, text).unwrap();
            }
        }
    }

    /// Write `leg`'s shape run of `frame_shape` with `log` as its layer log.
    fn shape_run(&self, leg: &str, log: &str) {
        let dir = self.root.join(leg).join("shape").join("e2e.frame_shape");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("e2e-7.log"), log).unwrap();
        fs::write(
            dir.join("bench-frame_shape.metrics"),
            "meta frame_shape layer v1\n",
        )
        .unwrap();
    }
}

#[test]
fn only_a_benchmark_that_declares_its_frame_needs_a_shape_run() {
    let fixture = Fixture::new("shape-declared");
    fixture.standard(3, 1.0);
    fixture.declare_shape();
    for round in 0..3 {
        for (leg, image) in [("base", "AAAA"), ("cand", "BBBB")] {
            fixture.write(
                leg,
                round,
                "stutter",
                &meta("v0.11.0-3-g66e4114", image),
                &[("x", 1.0, "ms lower time")],
            );
        }
    }
    for leg in ["base", "cand"] {
        fixture.shape_run(leg, &shape_log(false));
    }
    let comparison = evaluate(&fixture.root, &Options::default()).unwrap();
    assert!(!comparison.failed(), "{}", comparison.render());
    assert!(
        comparison
            .notes
            .iter()
            .any(|note| note == "stutter: no shape run; its metrics declare no shape lines"),
        "{:?}",
        comparison.notes
    );

    for leg in ["base", "cand"] {
        let dir = fixture.root.join(leg).join("shape");
        fs::rename(dir.join("e2e.frame_shape"), dir.join("e2e.other")).unwrap();
        fs::rename(
            dir.join("e2e.other").join("bench-frame_shape.metrics"),
            dir.join("e2e.other").join("bench-other.metrics"),
        )
        .unwrap();
    }
    let reason = error_of(&fixture);
    assert!(
        reason.contains("benchmark frame_shape declares shape lines and has no shape run"),
        "{reason}"
    );
}

#[test]
fn a_time_of_a_few_printed_steps_needs_more_than_one_step_to_move() {
    // The A/A case: perf.submit_commit_ms read 0.005 ms in the base and
    // 0.004 ms in the candidate on every pair, -20 % at a sigma of zero,
    // one step of the perf-kv line's three decimals apart.
    let spec = "ms lower time";
    let row = judge(
        "perf.submit_commit_ms",
        &def(spec),
        &[0.005; 5],
        &[0.004; 5],
        false,
    );
    assert_eq!(row.verdict, Verdict::Neutral);
    assert!(
        row.change.contains("within five steps of 0.001"),
        "{}",
        row.change
    );
    assert_eq!(
        verdict_of("perf.submit_commit_ms", spec, &[0.004; 5], &[0.005; 5]),
        Verdict::Neutral
    );
    // Exactly five steps is within it, whatever the subtraction rounds to.
    assert_eq!(
        verdict_of("perf.submit_commit_ms", spec, &[0.006; 5], &[0.011; 5]),
        Verdict::Neutral
    );
    // Past the floor both ways.
    assert_eq!(
        verdict_of("perf.submit_commit_ms", spec, &[0.005; 5], &[0.011; 5]),
        Verdict::Regression
    );
    assert_eq!(
        verdict_of("perf.submit_commit_ms", spec, &[0.011; 5], &[0.005; 5]),
        Verdict::Improvement
    );
    // A benchmark's own time has four decimals, so its floor is a tenth of that.
    assert_eq!(
        verdict_of("frame.p50", spec, &[0.1340; 5], &[0.1400; 5]),
        Verdict::Regression
    );
}

#[test]
fn the_time_step_follows_the_unit_and_the_source() {
    assert_eq!(time_step("perf.api_outside_ms", &Unit::Ms), Some(0.001));
    assert_eq!(time_step("frame.p50", &Unit::Ms), Some(0.0001));
    assert_eq!(time_step("emit.us_per_shader", &Unit::Us), Some(0.001));
    assert_eq!(time_step("ns_per_call.draw_clean", &Unit::Ns), Some(0.1));
    assert_eq!(time_step("fps", &Unit::Count), None);
    assert!(!past_floor(0.011 - 0.006, 0.001));
    assert!(past_floor(0.012 - 0.006, 0.001));
    assert!(!past_floor(-0.005, 0.001));
    assert!(past_floor(-0.006, 0.001));
}
