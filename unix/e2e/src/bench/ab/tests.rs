//! Unit tests for the A/B run order and the checks made after each run.

use super::*;

fn spec(stamp: &str) -> LegSpec {
    LegSpec {
        wine: PathBuf::from("/wine"),
        prefix: PathBuf::from("/prefix"),
        stamp: stamp.to_owned(),
    }
}

fn file(text: &str) -> MetricsFile {
    metrics::parse(text, "b").expect("the fixture parses")
}

#[test]
fn the_leg_that_goes_first_alternates_by_round() {
    let order = schedule(2, 3);
    let expected = [
        (0, 0, Leg::Base),
        (0, 0, Leg::Cand),
        (0, 1, Leg::Cand),
        (0, 1, Leg::Base),
        (0, 2, Leg::Base),
        (0, 2, Leg::Cand),
        (1, 0, Leg::Base),
        (1, 0, Leg::Cand),
        (1, 1, Leg::Cand),
        (1, 1, Leg::Base),
        (1, 2, Leg::Base),
        (1, 2, Leg::Cand),
    ];
    assert_eq!(order, expected);
}

#[test]
fn nothing_to_run_is_an_empty_schedule() {
    assert!(schedule(0, 5).is_empty());
}

#[test]
fn the_run_directory_is_appended_as_the_last_log_dir() {
    let dir = Path::new("/ab/base/0");
    assert_eq!(
        run_config("shaderCache.enable=false;log.dir=Z:/elsewhere", dir),
        "shaderCache.enable=false;log.dir=Z:/elsewhere;log.dir=Z:/ab/base/0"
    );
    assert_eq!(run_config("", dir), "log.dir=Z:/ab/base/0");
}

#[test]
fn a_run_must_report_its_legs_stamp_exactly() {
    let path = Path::new("/ab/base/0/bench-b.metrics");
    let good = file("meta b layer v0.11.0-3-g66e4114\n");
    assert!(check_stamp(path, &good, &spec("v0.11.0-3-g66e4114")).is_ok());

    let reason = check_stamp(path, &good, &spec("v0.11.0")).unwrap_err();
    assert!(
        reason.contains("loaded layer v0.11.0-3-g66e4114, the leg installed v0.11.0"),
        "{reason}"
    );

    let reason = check_stamp(path, &file("meta b arch x86\n"), &spec("v1")).unwrap_err();
    assert!(reason.contains("no meta layer line"), "{reason}");
}

#[test]
fn progress_shows_the_median_frame_when_there_is_one() {
    let path = Path::new("/ab/cand/1/bench-b.metrics");
    let with = file("metric b frame.p50 16.6667 ms lower time\n");
    assert_eq!(progress(path, &with), "b: frame.p50 16.667 ms");
    assert_eq!(progress(path, &file("")), "b: no frame.p50");
}
