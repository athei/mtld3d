//! Unit tests for the machine samples: which processes are foreign, the file, the busy rule.

use super::*;

/// The legs' Wine installs in the listing below.
fn legs() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/w/base/.wine-isolated/sdk"),
        PathBuf::from("/w/cand/.wine-isolated/sdk"),
    ]
}

/// A `<cpu%> <pid> <ppid> <command>` listing of the run and what else runs.
///
/// The run (42, under cargo 41 under make 40 under the terminal's zsh 39),
/// its `ps` (900), the legs' Wine sessions, another Wine's Steam, a native
/// game, Metal's compiler and the window server, and the kernel.
const LISTING: &str = "\
 97.3  4100     1 /Applications/Some Game.app/Contents/MacOS/Some Game
 89.0  5100  5000 C:\\Program Files (x86)\\Steam\\bin\\cef\\cef.win7x64\\steamwebhelper.exe
 70.0     0     0 kernel_task
 60.0  4200     1 /w/cand/.wine-isolated/sdk/bin/wineserver
 45.0  4300     1 C:\\windows\\system32\\winedevice.exe
 40.0   600     1 /System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService
 30.0    42    41 /w/cand/unix/target/production/mtld3d-e2e
 20.0    41    40 cargo
 15.0    40    39 make
  3.0    39     1 -zsh
 12.5   350     1 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer
  4.0   512     1 /usr/sbin/mds_stores
  2.0   513     1 /usr/libexec/trustd
  1.0   900    42 /bin/ps
  0.5   901     1 /usr/libexec/logd
";

#[test]
fn only_the_runs_own_wine_is_its_own_and_every_other_process_is_foreign() {
    // The leg's resident winedevice maps its image from the candidate's SDK;
    // Steam's helper maps it from another Wine.
    let mut asked = Vec::new();
    let classified = classify(LISTING, 42, &legs(), |pid| {
        asked.push(pid);
        Some(pid == 4300)
    });
    let pids: Vec<u32> = classified.top.iter().map(|process| process.pid).collect();
    assert_eq!(pids, [4100, 5100, 39]);
    assert_eq!(classified.kernel_task, Some(70.0));
    // Every foreign process is summed, the terminal's shell above the run's
    // make and mds_stores and trustd past the top included; logd, under a
    // percent, is not counted.
    assert!((classified.foreign - (97.3 + 89.0 + 3.0 + 4.0 + 2.0)).abs() < 1e-9);
    // Only the processes that look like Wine are asked about, and not the
    // legs' own wineserver, whose executable already names a leg.
    assert_eq!(asked, [5100, 4300]);
}

#[test]
fn a_sample_reads_back_from_its_file() {
    let classified = classify(LISTING, 42, &legs(), |pid| Some(pid == 4300));
    let sample = Sample {
        load1: Some(5.25),
        kernel_task: classified.kernel_task,
        foreign: 195.5,
        top: classified.top,
    };
    let text = sample.text();
    assert!(text.starts_with(
        "load1 5.25\nkernel_task 70.0\nforeign 195.5\ntop 97.3 4100 /Applications/Some Game.app"
    ));
    assert_eq!(Sample::parse(&text), sample);
}

#[test]
fn a_round_is_busy_on_a_heavy_foreign_process_their_sum_or_a_throttling_kernel() {
    // The load average lags a minute behind the run's own work: recorded, never a reason.
    let quiet =
        Sample::parse("load1 6.40\nkernel_task 3.0\nforeign 20.0\ntop 12.5 350 mds_stores\n");
    assert_eq!(quiet.busy(), None);
    let many = Sample::parse("foreign 55.0\ntop 20.0 1 a\ntop 20.0 2 b\ntop 15.0 3 c\n");
    assert_eq!(
        many.busy().as_deref(),
        Some("other processes at 55 % of a core together")
    );
    let hot = Sample::parse("kernel_task 70.0\n");
    assert_eq!(
        hot.busy().as_deref(),
        Some("kernel_task at 70 % of a core (macOS holding the CPUs back for heat)")
    );
    let game = Sample::parse("foreign 89.0\ntop 89.0 5100 C:\\steamwebhelper.exe --type=gpu\n");
    assert_eq!(
        game.busy().as_deref(),
        Some("C:\\steamwebhelper.exe --type=gpu (pid 5100) at 89 % of a core")
    );
}

#[test]
fn a_directory_warns_for_each_busy_round_and_process() {
    let root = std::env::temp_dir().join(format!("mtld3d-bench-machine-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let game = Sample::parse("load1 6.00\nforeign 97.3\ntop 97.3 4100 Some Game\n");
    keep(&root.join("cand").join("1"), "e2e", &game).unwrap();
    keep(
        &root.join("base").join("0"),
        "e2e",
        &Sample::parse("load1 9.00\n"),
    )
    .unwrap();
    keep(&root.join("base").join("0"), "host", &game).unwrap();
    let found = warnings(&root, &["base", "cand"], 2);
    let _ = fs::remove_dir_all(&root);
    let busy = "Some Game (pid 4100) at 97 % of a core";
    assert_eq!(
        found,
        [
            format!("base round 1 (host) started on a busy machine: {busy}; run it again"),
            format!("cand round 2 (e2e) started on a busy machine: {busy}; run it again"),
        ]
    );
}

#[test]
fn a_cpu_time_reads_in_every_form_ps_prints() {
    let near = |time: &str, seconds: f64| {
        cpu_seconds(time).is_some_and(|read| (read - seconds).abs() < 1e-9)
    };
    assert!(near("0:01.25", 1.25));
    assert!(near("12:34.50", 754.5));
    assert!(near("1:02:03.00", 3723.0));
    assert!(near("2-01:00:00.00", 176_400.0));
    assert!(cpu_seconds("n/a").is_none());
}

#[test]
fn a_wine_process_lsof_cannot_read_is_dropped_not_foreign() {
    // Steam's helper exited before lsof looked (or lsof failed): no image.
    let classified = classify(LISTING, 42, &legs(), |pid| (pid == 4300).then_some(true));
    let pids: Vec<u32> = classified.top.iter().map(|process| process.pid).collect();
    assert_eq!(pids, [4100, 39, 512]);
}

#[test]
fn an_ancestor_cycle_ends_the_walk() {
    // A listing whose parents point at each other must still classify.
    let listing = " 30.0 42 41 mtld3d-e2e\n 20.0 41 40 sh\n 20.0 40 41 make\n 26.0 7 1 other\n";
    let classified = classify(listing, 42, &legs(), |_| None);
    let pids: Vec<u32> = classified.top.iter().map(|process| process.pid).collect();
    assert_eq!(pids, [7]);
}

/// A second CPU-time read of `pid`, child of 1, at `seconds`.
fn later(pid: u32, seconds: f64) -> CpuTime {
    CpuTime {
        pid,
        ppid: 1,
        seconds,
        command: format!("p{pid}"),
    }
}

#[test]
fn shares_divide_the_time_used_by_the_time_measured() {
    let before = BTreeMap::from([(10, 5.0), (11, 1.0), (12, 9.0), (13, 3.0)]);
    let after = [
        later(10, 5.25),
        // 11 was reused by a process younger than the first read.
        later(11, 0.1),
        // 14 started between the reads.
        later(14, 0.2),
        later(13, 3.0),
    ];
    // 12 exited between the reads and is not listed.
    // Measured at 625 ms, not the nominal INTERVAL: 0.25 s used is 40 %.
    let listing = shares(&before, &after, Duration::from_millis(625));
    assert_eq!(
        listing,
        "40.0 10 1 p10\n32.0 14 1 p14\n0.0 11 1 p11\n0.0 13 1 p13\n"
    );
    assert_eq!(
        cputime_rows("  10     1   0:05.25 /bin/p10\nbad line\n").len(),
        1
    );
}
