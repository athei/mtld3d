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

/// A fake Wine tree in a temporary directory: a `wine` that prints `version`, and a `wineserver`.
fn fake_wine(tag: &str, version: &str, server: &str) -> LegSpec {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = std::env::temp_dir().join(format!("mtld3d-bench-ab-{}-{tag}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let wine = dir.join("wine");
    fs::write(&wine, format!("#!/bin/sh\necho '{version}'\n")).unwrap();
    fs::set_permissions(&wine, fs::Permissions::from_mode(0o755)).unwrap();
    fs::write(dir.join("wineserver"), server).unwrap();
    LegSpec {
        wine,
        prefix: dir.join("prefix"),
        stamp: "v1".to_owned(),
    }
}

#[test]
fn both_legs_have_to_run_one_wine() {
    let base = fake_wine("base", "wine-10.0", "server-a");
    let same = fake_wine("same", "wine-10.0", "server-a");
    assert_eq!(
        check_wine(&base, &same).unwrap(),
        "wine-10.0, one wineserver"
    );

    let other_version = fake_wine("version", "wine-9.0", "server-a");
    let reason = check_wine(&base, &other_version).unwrap_err();
    assert!(
        reason.contains("different Wines: base wine-10.0"),
        "{reason}"
    );

    let other_server = fake_wine("server", "wine-10.0", "server-b");
    let reason = check_wine(&base, &other_server).unwrap_err();
    assert!(reason.contains("different wineservers"), "{reason}");

    for spec in [base, same, other_version, other_server] {
        let _ = fs::remove_dir_all(spec.wine.parent().unwrap());
    }
}
