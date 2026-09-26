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
fn the_shape_runs_follow_the_rounds_and_the_leg_that_goes_first_alternates_by_round() {
    let timed = |bench, round, leg| Step::Timed { bench, round, leg };
    let shape = |bench, leg| Step::Shape { bench, leg };
    let order = schedule(2, 3);
    let expected = [
        timed(0, 0, Leg::Base),
        timed(0, 0, Leg::Cand),
        timed(0, 1, Leg::Cand),
        timed(0, 1, Leg::Base),
        timed(0, 2, Leg::Base),
        timed(0, 2, Leg::Cand),
        shape(0, Leg::Base),
        shape(0, Leg::Cand),
        timed(1, 0, Leg::Base),
        timed(1, 0, Leg::Cand),
        timed(1, 1, Leg::Cand),
        timed(1, 1, Leg::Base),
        timed(1, 2, Leg::Base),
        timed(1, 2, Leg::Cand),
        shape(1, Leg::Base),
        shape(1, Leg::Cand),
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

#[test]
fn progress_falls_back_to_the_first_time_metric() {
    let path = Path::new("/ab/cand/1/bench-b.metrics");
    let host = file(
        "metric b shaders 10 count higher info\nmetric b emit.us_per_shader 12.5 us lower time\n",
    );
    assert_eq!(progress(path, &host), "b: emit.us_per_shader 12.500 us");
}

#[test]
fn the_host_benchmark_writes_into_the_round_and_reads_every_corpus() {
    let args = host_args(
        Path::new("/ab/base/0"),
        &[PathBuf::from("/a.bin"), PathBuf::from("/b c.bin")],
    );
    assert_eq!(args, ["--metrics", "/ab/base/0", "/a.bin", "/b c.bin"]);
    assert_eq!(host_args(Path::new("/d"), &[]), ["--metrics", "/d"]);
}

/// A fake `emit_corpus` in a temporary directory running `script` with its arguments.
fn fake_host(tag: &str, script: &str) -> (PathBuf, PathBuf) {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = std::env::temp_dir().join(format!("mtld3d-bench-host-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("emit_corpus");
    fs::write(&exe, format!("#!/bin/sh\n{script}\n")).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
    (exe, dir.join("round"))
}

#[test]
fn a_host_run_reads_back_the_files_it_wrote() {
    let (exe, round) = fake_host(
        "ok",
        "[ \"$1\" = --metrics ] || exit 3\necho timing\n\
         printf 'meta host_emit_x kind host\\nmeta host_emit_x layer v1\\n\
         metric host_emit_x emit.us_per_shader 3.5 us lower time\\n' > \"$2/bench-host_emit_x.metrics\"",
    );
    let written = run_host(&exe, &[], &round, Duration::from_secs(10)).unwrap();
    assert_eq!(written.len(), 1);
    assert_eq!(written[0].1.meta["kind"], "host");
    assert!(check_stamp(&written[0].0, &written[0].1, &spec("v1")).is_ok());
    let log = fs::read_to_string(round.join(HOST_LOG)).unwrap();
    assert_eq!(log, "timing\n");
    let _ = fs::remove_dir_all(exe.parent().unwrap());
}

#[test]
fn a_host_run_that_fails_hangs_or_writes_nothing_is_an_error() {
    let (exe, round) = fake_host("fail", "echo 'emit failed' >&2\nexit 1");
    let reason = run_host(&exe, &[], &round, Duration::from_secs(10)).unwrap_err();
    assert!(reason.contains("ended with"), "{reason}");
    assert!(reason.contains("emit failed"), "{reason}");
    let _ = fs::remove_dir_all(exe.parent().unwrap());

    let (exe, round) = fake_host("silent", "exit 0");
    let reason = run_host(&exe, &[], &round, Duration::from_secs(10)).unwrap_err();
    assert!(reason.contains("wrote no bench-<name>.metrics"), "{reason}");
    let _ = fs::remove_dir_all(exe.parent().unwrap());

    let (exe, round) = fake_host("hang", "exec sleep 30");
    let reason = run_host(&exe, &[], &round, Duration::from_millis(200)).unwrap_err();
    assert!(reason.contains("ran longer than"), "{reason}");
    let _ = fs::remove_dir_all(exe.parent().unwrap());
}

#[test]
fn every_run_directory_links_the_one_staged_corpus() {
    let root = std::env::temp_dir().join(format!("mtld3d-bench-corpus-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let staged = root.join("corpus");
    fs::create_dir_all(staged.join("game")).unwrap();
    fs::write(staged.join("game").join("mtld3d_shaders.bin"), b"cache").unwrap();
    let round = root.join("base").join("0");
    fs::create_dir_all(&round).unwrap();
    link_corpus(&staged, &round).unwrap();
    link_corpus(&staged, &round).unwrap();
    assert_eq!(
        fs::read(round.join("corpus").join("game").join("mtld3d_shaders.bin")).unwrap(),
        b"cache"
    );

    let other = root.join("cand").join("0");
    fs::create_dir_all(other.join("corpus")).unwrap();
    let reason = link_corpus(&staged, &other).unwrap_err();
    assert!(
        reason.contains("is not a link to the staged caches"),
        "{reason}"
    );
    let _ = fs::remove_dir_all(&root);
}
