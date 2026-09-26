//! Unit tests for the machine samples: which processes are foreign, the file, the busy rule.

use super::*;

/// The legs' Wine installs in the listing below.
fn legs() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/w/base/.wine-isolated/sdk"),
        PathBuf::from("/w/cand/.wine-isolated/sdk"),
    ]
}

/// A `ps -r -o pcpu=,pid=,ppid=,comm=` listing: the run (42, under cargo 41 under make 40),
/// its `ps` (900), the legs' Wine sessions, another Wine's Steam, a native game, Metal's
/// compiler and the window server, and the kernel.
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
 12.5   350     1 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer
  4.0   512     1 /usr/sbin/mds_stores
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
        pid == 4300
    });
    let pids: Vec<u32> = classified.top.iter().map(|process| process.pid).collect();
    assert_eq!(pids, [4100, 5100, 512]);
    assert_eq!(classified.kernel_task, Some(70.0));
    // Only the processes that look like Wine are asked about, and not the
    // legs' own wineserver, whose executable already names a leg.
    assert_eq!(asked, [5100, 4300]);
}

#[test]
fn a_sample_reads_back_from_its_file() {
    let sample = Sample {
        load1: Some(5.25),
        kernel_task: Some(70.0),
        top: classify(LISTING, 42, &legs(), |pid| pid == 4300).top,
    };
    let text = sample.text();
    assert!(
        text.starts_with("load1 5.25\nkernel_task 70.0\ntop 97.3 4100 /Applications/Some Game.app")
    );
    assert_eq!(Sample::parse(&text), sample);
}

#[test]
fn a_round_is_busy_on_a_high_load_a_throttling_kernel_or_a_heavy_foreign_process() {
    let quiet = Sample::parse("load1 2.40\nkernel_task 3.0\ntop 12.5 350 mds_stores\n");
    assert_eq!(quiet.busy(), None);
    let loaded = Sample::parse("load1 4.50\n");
    assert_eq!(
        loaded.busy().as_deref(),
        Some("1-minute load 4.50 (over 4.0)")
    );
    let hot = Sample::parse("kernel_task 70.0\n");
    assert_eq!(
        hot.busy().as_deref(),
        Some("kernel_task at 70 % of a core (macOS holding the CPUs back for heat)")
    );
    let game = Sample::parse("load1 3.00\ntop 89.0 5100 C:\\steamwebhelper.exe --type=gpu\n");
    assert_eq!(
        game.busy().as_deref(),
        Some("C:\\steamwebhelper.exe --type=gpu (pid 5100) at 89 % of a core")
    );
}

#[test]
fn a_directory_warns_for_each_busy_round_and_process() {
    let root = std::env::temp_dir().join(format!("mtld3d-bench-machine-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let game = Sample::parse("load1 6.00\ntop 97.3 4100 Some Game\n");
    keep(&root.join("cand").join("1"), "e2e", &game).unwrap();
    keep(
        &root.join("base").join("0"),
        "e2e",
        &Sample::parse("load1 1.00\n"),
    )
    .unwrap();
    keep(&root.join("base").join("0"), "host", &game).unwrap();
    let found = warnings(&root, &["base", "cand"], 2);
    let _ = fs::remove_dir_all(&root);
    let busy = "1-minute load 6.00 (over 4.0); Some Game (pid 4100) at 97 % of a core";
    assert_eq!(
        found,
        [
            format!("base round 1 (host) started on a busy machine: {busy}; run it again"),
            format!("cand round 2 (e2e) started on a busy machine: {busy}; run it again"),
        ]
    );
}
