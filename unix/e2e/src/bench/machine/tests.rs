//! Unit tests for the machine samples: the `ps` filter, the file format and the busy rule.

use super::*;

const LISTING: &str = "\
 97.3  4100 /Applications/Some Game.app/Contents/MacOS/Some Game
 60.0  4200 /Users/alex/.wine-isolated/sdk/bin/wineserver
 45.0  4300 C:\\windows\\system32\\winedevice.exe
 30.0    42 /Users/alex/Developer/mtld3d/unix/target/production/mtld3d-e2e
 12.5   350 /System/Library/CoreServices/WindowServer
  4.0   512 /usr/sbin/mds_stores
  1.0   900 /bin/ps
  0.5   901 /usr/libexec/logd
";

#[test]
fn the_top_leaves_out_the_runs_own_processes() {
    let top = foreign_top(LISTING, 42);
    let pids: Vec<u32> = top.iter().map(|process| process.pid).collect();
    assert_eq!(pids, [4100, 350, 512]);
    assert_eq!(
        top[0].command,
        "/Applications/Some Game.app/Contents/MacOS/Some Game"
    );
}

#[test]
fn a_sample_reads_back_from_its_file() {
    let sample = Sample {
        load1: Some(5.25),
        top: foreign_top(LISTING, 42),
    };
    let text = sample.text();
    assert!(text.starts_with("load1 5.25\ntop 97.3 4100 /Applications/Some Game.app"));
    assert_eq!(Sample::parse(&text), sample);
}

#[test]
fn a_round_is_busy_on_a_high_load_or_a_heavy_foreign_process() {
    let quiet = Sample {
        load1: Some(2.4),
        top: vec![Process {
            cpu: 12.5,
            pid: 350,
            command: "WindowServer".to_owned(),
        }],
    };
    assert_eq!(quiet.busy(), None);
    let loaded = Sample {
        load1: Some(4.5),
        ..Sample::default()
    };
    assert_eq!(
        loaded.busy().as_deref(),
        Some("1-minute load 4.50 (over 4.0)")
    );
    let game = Sample::parse("load1 3.00\ntop 97.3 4100 Some Game\n");
    assert_eq!(
        game.busy().as_deref(),
        Some("Some Game (pid 4100) at 97 % of a core")
    );
}

#[test]
fn a_directory_warns_for_each_busy_round() {
    let root = std::env::temp_dir().join(format!("mtld3d-bench-machine-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let game = Sample::parse("load1 6.00\ntop 97.3 4100 Some Game\n");
    keep(&root.join("cand").join("1"), MACHINE_FILES[0].0, &game).unwrap();
    keep(
        &root.join("base").join("0"),
        MACHINE_FILES[0].0,
        &Sample::parse("load1 1.00\n"),
    )
    .unwrap();
    keep(&root.join("base").join("0"), MACHINE_FILES[1].0, &game).unwrap();
    let found = warnings(&root, &["base", "cand"], 2);
    let _ = fs::remove_dir_all(&root);
    let busy = "1-minute load 6.00 (over 4.0); Some Game (pid 4100) at 97 % of a core";
    assert_eq!(
        found,
        [
            format!(
                "base round 1 of the host emitter started on a busy machine: {busy}; run it again"
            ),
            format!(
                "cand round 2 of the benchmarks started on a busy machine: {busy}; run it again"
            ),
        ]
    );
}
