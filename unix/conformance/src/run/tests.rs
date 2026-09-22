//! Unit tests for the Metal-validation gate and the GPU-hang stop.
//!
//! The recogniser picks the layer's error messages out of a subtest's stderr,
//! keeps the detail lines under each one, collapses volatile numbers so a
//! repeated message reports once, and leaves ordinary Wine and Metal chatter
//! alone. The gate then fails a leg for any message that survives: after the
//! warning filter there is no benign one left.
//!
//! The spawn tests let `/bin/sh` play Wine and a script play the test binary.
//! The driver's hang line has to stop the subtest within seconds, not at its
//! budget, and has to mark the run so its counts never reach a baseline. The
//! raw log has to end with how the process ended, since a run without the
//! framework's summary is a crash whose shape only that line tells, and the
//! process has to be handed a `log.dir` beside its raw output so its log file
//! reaches whoever reads the raw dir. A process that runs out of its budget
//! has to be sampled before the kill, with the sample kept beside the raw log
//! and named on its `TIMED OUT` line: the kill leaves no other account of
//! where the process was. Its `TIMED OUT` line names the wineserver sample
//! too, and that file says why it holds no stacks when no server was looked
//! for. A pipe a process outside the subtest's group still
//! holds after the subtest is gone must not park the run: the collection ends
//! at its grace with what arrived, and the raw log says it was cut short.
//!
//! Which server to sample is decided on a process table and a working
//! directory, so the decision is tested on both rather than on the machine's
//! own processes: a name alone must never pick another checkout's server, and
//! a server counts as this prefix's only when the directory it works from is
//! the one that prefix's device and inode name.

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use super::{
    DEFAULT_TIMEOUT, Launch, cwd_is_server_dir, is_gpu_hang_line, normalize_numbers, run_subtest,
    server_dir_for, validation_errors, validation_gate_failed, wineserver_pids,
};
use crate::model::{Arch, Gpu, Leg, Subtest, Variant};

/// The two draw errors the mismatched-sampler-type case logged, verbatim.
const MISMATCHED_SAMPLER: &str = "\
0000:err:d3d9: something unrelated\n\
2026-08-28 11:59:31.778 wine[27403:71732] Draw Errors Validation\n\
Fragment Function(mtld3d_ps_sm3_29b0ab41fc8da9b1): incorrect type of texture (MTLTextureType2D) bound at Texture binding at index 0 (expect MTLTextureType3D) for s0[0].\n\
Fragment Function(mtld3d_ps_sm3_fab23e4e59af0b3b): incorrect type of texture (MTLTextureType3D) bound at Texture binding at index 0 (expect MTLTextureType2D) for s0[0].\n";

/// The rejected sampler descriptor a paravirtual GPU logged, verbatim.
const REJECTED_SAMPLER_DESCRIPTOR: &str = "\
2026-08-28 11:59:28.959 wine[27403:71131] Metal API Validation Enabled\n\
2026-08-28 11:59:31.778 wine[27403:71732] Sampler Descriptor Validation\n\
MTLSamplerAddressModeMirrorClampToEdge is not supported on this device\n";

#[test]
fn draw_errors_are_detail_of_one_message() {
    let seen = validation_errors(MISMATCHED_SAMPLER);
    assert_eq!(seen.len(), 1, "one report, not one per line: {seen:?}");
    let msg = seen.iter().next().unwrap();
    assert_eq!(msg.lines().count(), 3, "banner and both draw errors: {msg}");
    assert!(validation_gate_failed(seen.len()));
}

#[test]
fn the_rejected_property_is_kept() {
    let seen = validation_errors(REJECTED_SAMPLER_DESCRIPTOR);
    assert_eq!(seen.len(), 1, "{seen:?}");
    let msg = seen.iter().next().unwrap();
    assert!(
        msg.contains("Sampler Descriptor Validation")
            && msg.contains("MTLSamplerAddressModeMirrorClampToEdge is not supported"),
        "the header alone does not say which property was rejected: {msg}"
    );
}

#[test]
fn a_message_ends_at_the_next_writer() {
    let stderr = "\
2026-08-28 11:59:31.778 wine[27403:71732] Sampler Descriptor Validation\n\
MTLSamplerAddressModeMirrorClampToEdge is not supported on this device\n\
0000:err:d3d9: unrelated, and not this message's detail\n\
2026-08-28 11:59:32.001 wine[27403:71732] Buffer Validation\n\
Insufficient buffer size at buffer binding at index 2\n";
    let seen = validation_errors(stderr);
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert!(
        seen.iter().all(|msg| !msg.contains("unrelated")),
        "a Wine channel line was swallowed as detail: {seen:?}"
    );
}

#[test]
fn one_error_repeated_per_draw_reports_once() {
    let line = "2026-08-28 11:59:31.778 wine[27403:71732] Buffer Validation: \
                Insufficient buffer size at buffer binding at index 2";
    let stderr = format!("{line} 0x600002a1c000\n{line} 0x600002b40480\n");
    assert_eq!(validation_errors(&stderr).len(), 1);
}

#[test]
fn ordinary_output_logs_nothing() {
    let stderr = "\
        d3d9_test.exe: 1234 tests executed (0 marked as todo, 0 failures), 3 skipped.\n\
        visual.c:9100: Test failed: Got unexpected colour 0x00000000.\n\
        Metal API Validation Enabled\n";
    assert!(validation_errors(stderr).is_empty());
    assert!(!validation_gate_failed(0));
}

#[test]
fn numbers_and_addresses_collapse() {
    assert_eq!(
        normalize_numbers("texture at index 3 (0xdeadbeef) must be <= 16"),
        "texture at index N (0xN) must be <= N"
    );
}

/// The two lines the paravirtual GPU's driver printed on the Intel CI image, verbatim.
const DRIVER_HANG: &str = "2026-09-05 05:23:06.572 wine[86111:201735] Execution of the command \
    buffer was aborted due to an error during execution. Caused GPU Hang Error \
    (00000003:kIOAccelCommandBufferCallbackErrorHang)";
const DRIVER_IGNORED: &str = "2026-09-05 05:23:06.667 wine[86111:201735] Execution of the \
    command buffer was aborted due to an error during execution. Ignored (for causing \
    prior/excessive GPU errors) (00000004:kIOAccelCommandBufferCallbackErrorSubmissionsIgnored)";

#[test]
fn the_drivers_hang_lines_are_recognised_and_nothing_else_is() {
    assert!(is_gpu_hang_line(DRIVER_HANG));
    assert!(is_gpu_hang_line(DRIVER_IGNORED));
    for line in [
        "2026-08-28 11:59:28.959 wine[27403:71131] Metal API Validation Enabled",
        "visual.c:9100: Test failed: Got unexpected colour 0x00000000.",
        "0000:err:d3d9: GPU hang is not what this line reports",
    ] {
        assert!(!is_gpu_hang_line(line), "{line}");
    }
}

/// A script in a fresh directory under the temp dir, run as `sh <script> <subtest>`.
///
/// `/bin/sh` plays Wine and the script plays `d3d9_test.exe`; `run_subtest`
/// only asks that the exe be a file. A script that has to outlive its lines
/// sleeps in a child of its own, which holds the pipes past the shell: the
/// kill has to take the group, or the run would wait for that child.
fn script(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mtld3d-conformance-run-{}-{name}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("d3d9_test.sh");
    fs::write(&path, body).expect("script");
    path
}

/// The spawn a script stands in for, keeping nothing.
fn launch(exe: PathBuf) -> Launch {
    Launch {
        wine: PathBuf::from("/bin/sh"),
        exe,
        log: "off".to_owned(),
        raw_dir: None,
        timeout: DEFAULT_TIMEOUT,
        wineserver: None,
    }
}

/// The spawn a script stands in for, keeping its raw output in the script's directory.
fn launch_kept(exe: PathBuf) -> Launch {
    let dir = exe.parent().expect("script dir").to_path_buf();
    Launch {
        raw_dir: Some(dir),
        ..launch(exe)
    }
}

const LEG: Leg = Leg {
    arch: Arch::I686,
    variant: Variant::Native,
    gpu: Gpu::Apple,
};

/// The framework's summary line, so a run reads as complete.
const SUMMARY: &str = "device: 10 tests executed (0 marked as todo, 1 failures), 0 skipped.";

/// The bound the spawn tests hold the runner to: well under any subtest budget.
const PROMPT: Duration = Duration::from_secs(5);

#[test]
fn a_subtest_that_hangs_the_gpu_is_stopped_at_once() {
    let exe = script(
        "hang",
        &format!("echo 'visual.c:100: Test failed: x'\necho '{DRIVER_HANG}' >&2\nsleep 30\n"),
    );
    let started = Instant::now();
    let run = run_subtest(&launch(exe), LEG, Subtest::Visual, None).expect("spawn sh");
    let elapsed = started.elapsed();
    assert!(run.gpu_hang);
    assert!(run.result.crash, "a leg cut short is not a clean count");
    assert!(
        elapsed < PROMPT,
        "took {elapsed:?}: the runner waited instead of killing"
    );
}

#[test]
fn a_hang_line_before_a_clean_exit_still_counts() {
    let exe = script(
        "hang-exit",
        &format!(
            "echo '{DRIVER_IGNORED}' >&2\necho 'visual: 10 tests executed (0 marked as todo, 0 failures), 0 skipped.'\nexit 0\n"
        ),
    );
    let run = run_subtest(&launch(exe), LEG, Subtest::Visual, None).expect("spawn sh");
    assert!(run.gpu_hang);
    assert!(
        run.result.crash,
        "a process that survived its hang still read zeros; its counts must not reach the baseline"
    );
}

#[test]
fn a_run_records_its_exit_code_and_keeps_its_log_beside_the_raw_output() {
    let exe = script(
        "exit-code",
        &format!(
            "echo 'device.c:10: Test failed: x'; echo '{SUMMARY}'; echo \"$MTLD3D_CONFIG\"; exit 5\n"
        ),
    );
    let launch = launch_kept(exe);
    let dir = launch.raw_dir.clone().expect("kept");
    let run = run_subtest(&launch, LEG, Subtest::Device, None).expect("spawn sh");
    assert!(!run.result.crash);

    let raw = fs::read_to_string(dir.join("i686-device.log")).expect("raw log kept");
    assert!(
        raw.trim_end()
            .ends_with("[conformance] subtest exited: code 5"),
        "{raw}"
    );
    // The process is told the same directory as a Windows path, one
    // directory per run, so the layer's retention never prunes across runs.
    let expected = format!("log.dir=Z:{}", dir.join("i686-device").display());
    assert!(raw.contains(&expected), "{raw}");
}

#[test]
fn a_run_ended_by_a_signal_records_the_number_and_is_a_crash() {
    let exe = script(
        "signal",
        "echo 'device.c:10: Test failed: x'; kill -SEGV $$\n",
    );
    let launch = launch_kept(exe);
    let dir = launch.raw_dir.clone().expect("kept");
    let run = run_subtest(&launch, LEG, Subtest::Device, None).expect("spawn sh");
    assert!(run.result.crash);
    let raw = fs::read_to_string(dir.join("i686-device.log")).expect("raw log kept");
    assert!(
        raw.trim_end()
            .ends_with("[conformance] subtest exited: signal 11"),
        "{raw}"
    );
}

#[test]
fn a_subtest_past_its_budget_is_sampled_before_the_kill() {
    let exe = script("budget", "echo 'device.c:10: Test failed: x'\nsleep 30\n");
    let launch = Launch {
        timeout: Duration::from_secs(1),
        ..launch_kept(exe)
    };
    let dir = launch.raw_dir.clone().expect("kept");
    let run = run_subtest(&launch, LEG, Subtest::Device, None).expect("spawn sh");
    assert!(run.result.crash, "a process killed at its budget is a hang");

    let raw = fs::read_to_string(dir.join("i686-device.log")).expect("raw log kept");
    let trailer = raw.trim_end().lines().last().expect("trailer");
    assert!(
        trailer.starts_with("[conformance] subtest TIMED OUT after 1s and was killed")
            && trailer.contains("i686-device.sample.txt")
            && trailer.ends_with("i686-device.wineserver-sample.txt"),
        "{trailer}"
    );
    let sample = fs::read_to_string(dir.join("i686-device.sample.txt"))
        .expect("sample kept beside the raw log");
    assert!(
        sample.contains("Call graph:"),
        "the sample holds no stacks of the process: {sample}"
    );
    // This spawn names no wineserver, so the second file says that rather
    // than being absent: a reader must not have to tell a server that was
    // never looked for from one that was and could not be found.
    let server = fs::read_to_string(dir.join("i686-device.wineserver-sample.txt"))
        .expect("wineserver sample kept beside the process's");
    assert!(
        server.contains("no wineserver was sampled"),
        "the wineserver sample says nothing about why it holds no stacks: {server}"
    );
}

/// A process table in the shape `ps -Ao pid=,comm=` prints, two prefixes deep.
///
/// Two checkouts each run a server out of their own isolated clone of a Wine
/// SDK, a third install has one of its own, and the loader of the first
/// checkout is running too.
const PROCESS_TABLE: &str = "  431 /usr/sbin/cfprefsd
 43903 /work/one/.wine-isolated/sdk/bin/wineserver
 44438 /work/two/.wine-isolated/sdk/bin/wineserver
 49324 /Applications/Other.app/Contents/Resources/wine/bin/wineserver
 50001 /work/one/.wine-isolated/sdk/bin/wine
";

#[test]
fn only_the_named_wineserver_binary_matches() {
    assert_eq!(
        wineserver_pids(
            PROCESS_TABLE,
            Path::new("/work/one/.wine-isolated/sdk/bin/wineserver")
        ),
        vec![43903],
        "another checkout's server, another install's, and the loader itself          all carry the same name"
    );
    assert!(
        wineserver_pids(PROCESS_TABLE, Path::new("/work/three/sdk/bin/wineserver")).is_empty(),
        "a binary nothing was exec'd from matches nothing"
    );
}

#[test]
fn the_server_directory_names_the_prefix_by_device_and_inode() {
    assert_eq!(
        server_dir_for(0x0100_0011, 0x08ac_256f),
        "server-1000011-8ac256f"
    );
}

#[test]
fn a_server_is_its_own_prefix_only_when_its_directory_says_so() {
    let cwd = |dir: &str| format!("p43903\nfcwd\nn/private/tmp/.wine-501/{dir}\n");
    assert!(cwd_is_server_dir(
        &cwd("server-1000011-8ac256f"),
        "server-1000011-8ac256f"
    ));
    assert!(
        !cwd_is_server_dir(&cwd("server-1000011-360f506"), "server-1000011-8ac256f"),
        "another prefix's server directory is not this prefix's"
    );
    assert!(
        !cwd_is_server_dir("", "server-1000011-8ac256f"),
        "a process whose directory could not be read is not a match"
    );
}

/// How long the held-pipe test lets the whole call take.
///
/// The collection is a second per pipe, so a run that stops waiting is back
/// in about two; the rest is room for a loaded machine. The holder lives
/// several times longer, so a run that waits for the pipe misses the bound
/// rather than passing slowly.
const HELD_PIPE_BOUND: Duration = Duration::from_secs(10);

/// How long the held-pipe test's holder keeps the pipes open, in seconds.
const HELD_PIPE_HOLD: &str = "30";

#[test]
fn a_pipe_held_past_the_exit_does_not_park_the_run() {
    let holder_pid = std::env::temp_dir().join(format!(
        "mtld3d-conformance-held-pipe-{}.pid",
        std::process::id()
    ));
    // `set -m` gives the background job a process group of its own, which is
    // where `explorer.exe /desktop` puts itself: it inherits the subtest's
    // pipes and neither the budget nor a group kill reaches it.
    let exe = script(
        "held-pipe",
        &format!(
            "echo 'device.c:10: Test failed: x'\necho '{SUMMARY}'\nset -m\nsleep \
             {HELD_PIPE_HOLD} &\necho $! > '{}'\nexit 0\n",
            holder_pid.display()
        ),
    );
    let launch = launch_kept(exe);
    let dir = launch.raw_dir.clone().expect("kept");
    let started = Instant::now();
    let run = run_subtest(&launch, LEG, Subtest::Device, None).expect("spawn sh");
    let elapsed = started.elapsed();

    let holder = fs::read_to_string(&holder_pid).expect("the holder wrote its pid");
    let holder: i32 = holder.trim().parse().expect("a pid");
    // SAFETY: kill(2) touches no memory of ours, and the holder is not this
    // process's child, so an id that has already gone is reported as ESRCH
    // rather than acted on.
    unsafe {
        let _ = libc::kill(holder, libc::SIGKILL);
    }
    let _ = fs::remove_file(&holder_pid);

    assert!(
        elapsed < HELD_PIPE_BOUND,
        "took {elapsed:?}: the run waited for a pipe the subtest no longer holds"
    );
    assert_eq!(
        run.result.sites.values().sum::<u32>(),
        1,
        "the output that arrived before the grace is still parsed in full: {:?}",
        run.result.sites
    );
    assert!(
        !run.result.crash,
        "the subtest printed its summary and exited"
    );
    let raw = fs::read_to_string(dir.join("i686-device.log")).expect("raw log kept");
    assert!(
        raw.contains("[conformance] output cut short after"),
        "a collection that stopped early has to say so: {raw}"
    );
}

#[test]
fn a_repeat_run_keeps_one_raw_log_per_attempt() {
    let exe = script("repeat", &format!("echo '{SUMMARY}'\n"));
    let launch = launch_kept(exe);
    let dir = launch.raw_dir.clone().expect("kept");
    for attempt in 1..=2 {
        run_subtest(&launch, LEG, Subtest::Visual, Some(attempt)).expect("spawn sh");
    }
    assert!(dir.join("i686-visual-1.log").is_file());
    assert!(dir.join("i686-visual-2.log").is_file());
    assert!(!dir.join("i686-visual.log").exists());
}

#[test]
fn nothing_is_kept_and_no_log_dir_is_named_without_a_raw_dir() {
    let exe = script(
        "no-raw-dir",
        &format!("echo \"$MTLD3D_CONFIG\"; echo '{SUMMARY}'\n"),
    );
    let dir = exe.parent().expect("script dir").to_path_buf();
    run_subtest(&launch(exe), LEG, Subtest::Device, None).expect("spawn sh");
    assert!(!dir.join("i686-device.log").exists());
}
