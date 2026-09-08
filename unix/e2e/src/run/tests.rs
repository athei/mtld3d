//! The process deadlines: `/bin/sh` plays Wine, a script plays the test binary.

use std::{
    fs,
    io::{BufRead, BufReader, Read},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use super::{ExitKind, ProcessGroup, run};

const DRIVER_HANG: &str = "Caused GPU Hang Error \
    (00000003:kIOAccelCommandBufferCallbackErrorHang)";

/// A script in a fresh directory under the target dir, run as `sh <script>`.
fn script(name: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mtld3d-e2e-run-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(format!("{name}.sh"));
    fs::write(&path, body).expect("script");
    path
}

fn run_script(path: &Path, timeout: Duration) -> (super::Exit, Duration, Vec<String>) {
    let mut lines = Vec::new();
    let started = Instant::now();
    let exit = run(&PathBuf::from("/bin/sh"), path, &[], timeout, &mut |line| {
        lines.push(line.to_owned());
    })
    .expect("spawn sh");
    (exit, started.elapsed(), lines)
}

#[test]
fn a_clean_exit_reports_its_code_and_stderr() {
    let path = script("clean", "echo one\necho two >&2\nexit 3\n");
    let (exit, _, lines) = run_script(&path, Duration::from_secs(5));
    assert_eq!(exit.kind, ExitKind::Code(3));
    assert_eq!(lines, ["one"]);
    assert_eq!(exit.stderr, "two\n");
    assert!(!exit.gpu_hang);
}

#[test]
fn a_gpu_hang_split_across_stderr_writes_stops_the_process() {
    let path = script(
        "gpu-hang",
        "printf 'Caused GPU Hang Error (00000003:kIOAccelCommandBuffer' >&2\n\
         sleep 0.1\n\
         printf 'CallbackErrorHang)\\n' >&2\n\
         sleep 30\n",
    );
    let started = Instant::now();
    let (exit, elapsed, _) = run_script(&path, Duration::from_secs(5));
    assert!(exit.gpu_hang);
    assert_eq!(exit.stderr, format!("{DRIVER_HANG}\n"));
    assert!(
        elapsed < Duration::from_secs(2),
        "took {elapsed:?}: the runner waited instead of stopping the process"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn a_gpu_hang_racing_with_a_clean_exit_still_counts() {
    let path = script(
        "gpu-hang-clean",
        &format!("printf '%s\\n' '{DRIVER_HANG}' >&2\nexit 0\n"),
    );
    let (exit, _, _) = run_script(&path, Duration::from_secs(5));
    assert!(exit.gpu_hang);
    assert_eq!(exit.stderr, format!("{DRIVER_HANG}\n"));
}

#[test]
fn unrelated_gpu_errors_are_not_hang_reports() {
    for stderr in [
        "0000:err:d3d9: GPU hang is not what this line reports",
        "command buffer failed with Internal Error",
        "kIOAccelCommandBufferCallbackErrorNotPermitted",
    ] {
        assert!(!super::is_gpu_hang_report(stderr), "{stderr}");
    }
}

#[test]
fn silence_on_stdout_is_killed_after_the_timeout() {
    let path = script("silent", "echo start\nsleep 30\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::TimedOut(timeout));
    assert_eq!(lines, ["start"]);
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn a_process_that_closes_stdout_and_never_exits_is_killed_and_reported_hung() {
    let path = script("hung", "echo last\nexec 1>&-\nexec 2>&-\nsleep 30\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::Hung(timeout));
    assert_eq!(lines, ["last"]);
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn a_killed_process_whose_survivor_holds_stderr_costs_the_grace_not_the_timeout() {
    // The script goes silent, so the watchdog kills its group; the perl
    // child moved itself to a group of its own first and keeps stderr open
    // past the kill, as Wine's debugger does after a crash.
    let path = script(
        "escaped",
        "echo start\necho early >&2\n(perl -e 'setpgrp(0,0); sleep 30' >/dev/null) &\nsleep 30\n",
    );
    let timeout = Duration::from_secs(2);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::TimedOut(timeout));
    assert_eq!(lines, ["start"]);
    assert!(
        exit.stderr.starts_with("early\n"),
        "stderr: {:?}",
        exit.stderr
    );
    assert!(
        exit.stderr.contains("stderr not collected in full"),
        "stderr: {:?}",
        exit.stderr
    );
    // One timeout for the silence, the grace for stderr, and slack; the old
    // shape paid the timeout twice.
    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "took {elapsed:?}"
    );
}

#[test]
fn a_descendant_holding_stderr_does_not_park_the_run() {
    // The script exits at once; its background child keeps stderr open.
    let path = script("holder", "echo done\n(sleep 30 >/dev/null) &\nexit 0\n");
    let timeout = Duration::from_millis(500);
    let (exit, elapsed, lines) = run_script(&path, timeout);
    assert_eq!(exit.kind, ExitKind::Code(0));
    assert_eq!(lines, ["done"]);
    assert!(
        exit.stderr.contains("stderr not collected in full"),
        "stderr: {:?}",
        exit.stderr
    );
    assert!(elapsed < Duration::from_secs(5), "took {elapsed:?}");
}

#[test]
fn observing_a_killed_leader_stops_a_descendant_missed_by_the_first_signal() {
    let path = script("missed-child", "sleep 30 &\necho ready\nwait\n");
    let mut child = Command::new("/bin/sh")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let mut ready = String::new();
    BufReader::new(child.stdout.take().expect("stdout piped"))
        .read_line(&mut ready)
        .expect("read readiness");
    assert_eq!(ready, "ready\n");
    // The child exists before the parent is killed. Signalling only the
    // leader models a group signal whose snapshot missed the new child,
    // without requiring the signal to race a particular fork instruction.
    child.kill().expect("kill group leader");
    let pid = child.id();
    let mut stderr = child.stderr.take().expect("stderr piped");
    let mut group = ProcessGroup::new(child);
    group.wait_killed().expect("observe group leader");
    assert_waitable(pid);
    let (closed, eof) = mpsc::channel();
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        closed
            .send(stderr.read_to_end(&mut bytes))
            .expect("report EOF");
    });
    let collected = eof.recv_timeout(Duration::from_secs(1));
    // Clean up the fixture even when the assertion below fails.
    group.kill().expect("clean up group");
    reader.join().expect("stderr reader");
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert!(!status.success());
    assert!(
        collected.is_ok(),
        "descendant kept stderr open: {collected:?}"
    );
    assert_eq!(collected.expect("EOF delivered").expect("read stderr"), 0);
}

#[test]
fn observing_a_clean_exit_keeps_the_leader_waitable() {
    let child = Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup::new(child);
    let gpu_hang = std::sync::atomic::AtomicBool::new(false);
    let observed = super::wait_bounded(&mut group, Duration::from_secs(2), &gpu_hang)
        .expect("observe group leader");
    assert!(matches!(observed, super::Wait::Exited));
    assert_waitable(pid);
    assert!(
        group
            .observe_exit(libc::WNOHANG)
            .expect("observe exit again")
    );
    assert_waitable(pid);
    super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::ZERO,
        || group.observe_exit(libc::WNOHANG),
    )
    .expect("EPERM accepts an already waitable leader");
    group
        .wait_killed()
        .expect("signal after observing clean exit");
    assert_waitable(pid);
    group.kill().expect("final signal before reap");
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert_eq!(status.code(), Some(7));
    assert_reaped(pid);
}

#[test]
fn a_killed_leader_stays_waitable_until_the_final_signal() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup::new(child);
    assert!(
        !group
            .observe_exit(libc::WNOHANG)
            .expect("observe running child")
    );
    group.kill().expect("first group signal");
    group.wait_killed().expect("observe exit and signal again");
    assert_waitable(pid);
    assert!(
        group
            .observe_exit(libc::WNOHANG)
            .expect("observe exit again")
    );
    group.kill().expect("final group signal");
    assert_waitable(pid);
    let status = group.reap().expect("reap group leader");
    assert!(!status.success());
    assert_reaped(pid);
}

#[test]
fn losing_the_waitable_child_disables_group_signals() {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    // Model an external waiter consuming the child, without reusing any ID.
    child.wait().expect("external reap");
    let mut group = ProcessGroup::new(child);
    let error = super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::from_mins(1),
        || group.observe_exit(libc::WNOHANG),
    )
    .expect_err("EPERM observation must revoke ownership after external reap");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
    assert!(group.child.is_none());
    let error = group.kill().expect_err("ownership stays revoked");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
    let error = group
        .signal_group()
        .expect_err("no signal without ownership");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
}

#[test]
fn dropping_an_unfinished_group_stops_and_reaps_its_leader() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    drop(ProcessGroup::new(child));
    assert_reaped(pid);
}

#[test]
fn an_unexpected_wait_error_retains_ownership_for_cleanup() {
    let child = Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .expect("spawn group leader");
    let pid = child.id();
    let mut group = ProcessGroup::new(child);
    let error = group
        .observe_exit(libc::c_int::MAX)
        .expect_err("invalid wait options");
    assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
    assert!(group.child.is_some());
    drop(group);
    assert_reaped(pid);
}

#[test]
fn a_group_permission_error_waits_for_delayed_exit_observation() {
    let mut observations = [false, false, true].into_iter();
    super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::from_mins(1),
        || Ok(observations.next().expect("stop observing once exited")),
    )
    .expect("an exiting leader becomes waitable after EPERM");
    assert_eq!(observations.next(), None);
}

#[test]
fn a_group_permission_error_survives_the_exit_observation_deadline() {
    let mut observations = 0;
    let error = super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::ZERO,
        || {
            observations += 1;
            Ok(false)
        },
    )
    .expect_err("a still-running leader does not excuse EPERM");
    assert_eq!(error.raw_os_error(), Some(libc::EPERM));
    assert_eq!(observations, 1);
}

#[test]
fn a_group_permission_error_propagates_wait_errors_without_retrying() {
    for errno in [libc::ECHILD, libc::EIO] {
        let mut observations = 0;
        let error = super::resolve_group_signal_error(
            std::io::Error::from_raw_os_error(libc::EPERM),
            Duration::from_mins(1),
            || {
                observations += 1;
                if observations == 1 {
                    Ok(false)
                } else {
                    Err(std::io::Error::from_raw_os_error(errno))
                }
            },
        )
        .expect_err("wait errors must propagate");
        assert_eq!(error.raw_os_error(), Some(errno));
        assert_eq!(observations, 2);
    }
}

#[test]
fn other_group_signal_errors_do_not_observe_exit() {
    for errno in [libc::EINVAL, libc::EACCES, libc::EINTR] {
        let error = super::resolve_group_signal_error(
            std::io::Error::from_raw_os_error(errno),
            Duration::from_mins(1),
            || panic!("only EPERM needs exit observation"),
        )
        .expect_err("unrelated signal errors must propagate immediately");
        assert_eq!(error.raw_os_error(), Some(errno));
    }
    super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::ESRCH),
        Duration::from_mins(1),
        || panic!("an absent group needs no exit observation"),
    )
    .expect("ESRCH still accepts an absent group");
}

fn assert_waitable(pid: u32) {
    let info = observe_fixture(pid).expect("exit observation must not reap the leader");
    assert_eq!(info.si_pid.unsigned_abs(), pid);
}

fn assert_reaped(pid: u32) {
    match observe_fixture(pid) {
        Ok(_) => panic!("final reap left a waitable child"),
        Err(error) => assert_eq!(error.raw_os_error(), Some(libc::ECHILD)),
    }
}

fn observe_fixture(pid: u32) -> std::io::Result<libc::siginfo_t> {
    // SAFETY: siginfo_t contains integers and raw pointers, all valid when zeroed.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: info is writable and P_PID selects only this fixture's direct child.
    let result = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &raw mut info,
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    if result == 0 {
        Ok(info)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Private syscall substitutions, never exposed by the runner CLI.
pub struct Faults {
    pub signal: Option<i32>,
    pub wait: Option<i32>,
    pub wait_interruptions: u8,
    pub reap_interruptions: u8,
    pub reap: Option<i32>,
}

impl Faults {
    pub const fn new() -> Self {
        Self {
            signal: None,
            wait: None,
            wait_interruptions: 0,
            reap_interruptions: 0,
            reap: None,
        }
    }
}

/// Invoked only by the parent test, so fatal cleanup cannot end the test runner.
#[test]
fn fatal_cleanup_fixture() {
    let Ok(directory) = std::env::var("E2E_CLEANUP_FIXTURE") else {
        return;
    };
    let mode = std::env::var("E2E_CLEANUP_MODE").expect("fixture mode");
    let adoption = Path::new(&directory).join("adoption");
    let child = Command::new("/usr/bin/perl")
        .args([
            "-e",
            "select undef,undef,undef,$ARGV[1]; open my $f, '>', $ARGV[0] or die $!; print $f qq($$ ),getppid(); close $f;",
        ])
        .arg(adoption)
        .arg(if mode.starts_with("reap-") { "0.02" } else { "3" })
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .expect("spawn finite owned child");
    fs::write(Path::new(&directory).join("pid"), child.id().to_string()).expect("record child");
    let mut group = ProcessGroup::new(child);
    group.cleanup_deadline = Some(Instant::now() + Duration::from_millis(100));
    group.faults.signal = Some(libc::EPERM);
    match mode.as_str() {
        "drop" => drop(group),
        "kill" => {
            let error = group.kill().expect_err("injected signal denial");
            eprintln!("initial kill: {error}");
            drop(group);
        }
        "wait" => {
            group.faults.wait = Some(libc::EIO);
            drop(group);
        }
        "interrupt" => {
            group.faults.wait = Some(libc::EINTR);
            drop(group);
        }
        "reap-error" | "reap-interrupt" => {
            group.wait_killed().expect("observe finite child");
            group.faults.reap = Some(if mode == "reap-error" {
                libc::EIO
            } else {
                libc::EINTR
            });
            assert_waitable(group.child.as_ref().expect("owned child").id());
            let error = group.reap().expect_err("injected consuming wait failure");
            panic!("reap returned after failed cleanup: {error}");
        }
        "reap" => {
            group.faults.reap = Some(libc::EIO);
            let error = group.reap().expect_err("live child cannot be reaped");
            panic!("reap returned after failed cleanup: {error}");
        }
        "spawn" => {
            let error = super::collect(group, Duration::from_millis(10), &mut |_| {}, |_| {
                Err(std::io::Error::from_raw_os_error(libc::EAGAIN))
            })
            .err();
            panic!("reader spawn returned after failed cleanup: {error:?}");
        }
        _ => panic!("unknown fixture mode"),
    }
}

#[test]
fn persistent_signal_denial_ends_the_runner_and_transfers_reaping_to_the_os() {
    check_fatal_cleanup("drop");
}

#[test]
fn failed_kill_does_not_restart_the_cleanup_budget() {
    check_fatal_cleanup("kill");
}

#[test]
fn persistent_wait_errors_do_not_block_cleanup() {
    check_fatal_cleanup("wait");
}

#[test]
fn interrupted_waits_do_not_restart_the_cleanup_budget() {
    check_fatal_cleanup("interrupt");
}

#[test]
fn consuming_reap_cannot_wait_for_a_live_unsignalable_child() {
    check_fatal_cleanup("reap");
}

#[test]
fn reader_spawn_failure_keeps_a_live_child_owned_until_fatal_exit() {
    check_fatal_cleanup("spawn");
}

fn check_fatal_cleanup(mode: &str) {
    let path = script(&format!("fatal-{mode}"), "");
    let directory = path.parent().expect("fixture directory");
    for name in ["pid", "adoption"] {
        match fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("reset fixture: {error}"),
        }
    }
    let started = Instant::now();
    let mut fixture = Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "run::tests::fatal_cleanup_fixture",
            "--nocapture",
        ])
        .env("E2E_CLEANUP_FIXTURE", directory)
        .env("E2E_CLEANUP_MODE", mode)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .expect("spawn fixture runner");
    let status = loop {
        if let Some(status) = fixture.try_wait().expect("poll fixture runner") {
            break status;
        }
        if started.elapsed() > Duration::from_secs(6) {
            fixture.kill().expect("stop owned fixture runner");
            break fixture.wait().expect("reap owned fixture runner");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let elapsed = started.elapsed();
    let mut stderr = String::new();
    fixture
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_string(&mut stderr)
        .expect("read diagnostic");
    let pid: i32 = fs::read_to_string(directory.join("pid"))
        .expect("child identity")
        .parse()
        .expect("pid");
    // The finite child requires no signal after the runner exits. Observe both
    // its adoption report and its disappearance before retiring this fixture.
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        // SAFETY: signal zero only queries existence; no process is signalled.
        let present = unsafe { libc::kill(pid, 0) };
        if present == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "finite fixture child {pid} did not retire"
        );
        thread::sleep(Duration::from_millis(10));
    }
    let adoption = fs::read_to_string(directory.join("adoption")).expect("child adoption report");
    eprintln!("{mode}: runner {status} after {elapsed:?}; child {pid} and group retired");
    assert_eq!(status.code(), Some(2), "{mode}: {status}; {stderr}");
    assert!(
        elapsed < Duration::from_secs(1),
        "{mode}: took {elapsed:?}; {stderr}"
    );
    assert!(
        stderr.contains(&format!("fatal cleanup for process {pid}")),
        "{stderr}"
    );
    assert!(stderr.contains("a survivor may remain"), "{stderr}");
    if !mode.starts_with("reap-") {
        assert_eq!(
            adoption,
            format!("{pid} 1"),
            "child was not adopted by launchd"
        );
    }
    if mode == "spawn" {
        assert!(stderr.contains("start stdout reader"), "{stderr}");
        assert!(
            stderr.contains(&std::io::Error::from_raw_os_error(libc::EAGAIN).to_string()),
            "{stderr}"
        );
    }
    if mode == "wait" || mode == "reap-error" {
        assert!(stderr.contains("Input/output error"), "{stderr}");
    }
    // SAFETY: signal zero queries group existence without signalling a member.
    assert_eq!(unsafe { libc::kill(-pid, 0) }, -1, "fixture group remains");
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    eprintln!("{mode}: infrastructure exit in {elapsed:?}; child {pid} and group retired");
}

#[test]
fn failure_to_start_either_reader_stops_and_reaps_the_owned_child() {
    for fail_at in [0, 1] {
        let child = Command::new("/bin/sleep")
            .arg("30")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .expect("spawn owned child");
        let pid = child.id();
        let mut starts = 0;
        let mut readers = Vec::new();
        let error = super::collect(
            ProcessGroup::new(child),
            Duration::from_secs(1),
            &mut |_| {},
            |reader| {
                if starts == fail_at {
                    Err(std::io::Error::from_raw_os_error(libc::EAGAIN))
                } else {
                    starts += 1;
                    let handle = thread::Builder::new().spawn(reader)?;
                    // collect intentionally detaches the handle. This watcher is
                    // owned by the test and verifies the started reader completes.
                    let (done, completed) = mpsc::channel();
                    readers.push(completed);
                    thread::Builder::new().spawn(move || {
                        handle.join().expect("reader");
                        done.send(()).expect("report reader completion");
                    })
                }
            },
        )
        .err()
        .expect("reader spawn fails");
        assert!(
            error.contains(if fail_at == 0 {
                "stdout reader"
            } else {
                "stderr reader"
            }),
            "{error}"
        );
        assert_reaped(pid);
        for reader in readers {
            reader
                .recv_timeout(Duration::from_secs(1))
                .expect("started reader retired");
        }
    }
}

#[test]
fn a_persistent_consuming_wait_error_ends_the_cli_without_releasing_ownership() {
    check_fatal_cleanup("reap-error");
}

#[test]
fn consuming_wait_echild_revokes_ownership_without_another_group_signal() {
    let mut child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .expect("spawn child");
    let pid = child.id();
    child.wait().expect("external reap");
    let mut group = ProcessGroup::new(child);
    let error = group
        .try_reap()
        .expect_err("external waiter consumed status");
    assert_eq!(error.raw_os_error(), Some(libc::ECHILD));
    assert!(group.child.is_none());
    assert_eq!(
        group
            .signal_group()
            .expect_err("signals revoked")
            .raw_os_error(),
        Some(libc::ECHILD)
    );
    assert_reaped(pid);
}

#[test]
fn a_consuming_wait_interruption_retains_the_child_and_the_deadline() {
    let child = Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .expect("spawn child");
    let pid = child.id();
    let mut group = ProcessGroup::new(child);
    group.wait_killed().expect("observe exit");
    let deadline = group.cleanup_deadline;
    group.faults.reap = Some(libc::EINTR);
    let error = group
        .try_reap()
        .expect_err("injected interrupted consuming wait");
    assert_eq!(error.raw_os_error(), Some(libc::EINTR));
    assert_waitable(pid);
    assert_eq!(group.cleanup_deadline, deadline);
    group.faults.reap = None;
    group.reap().expect("final signal then reap");
    assert_reaped(pid);
}

#[test]
fn stderr_chunks_do_not_restart_the_drain_deadline() {
    let (sender, chunks) = mpsc::channel();
    let producer = thread::spawn(move || {
        for _ in 0..100 {
            if sender.send(b"still alive\n".to_vec()).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    let started = Instant::now();
    let stderr = super::drain_stderr(&chunks, Duration::from_millis(30));
    let elapsed = started.elapsed();
    drop(chunks);
    producer.join().expect("producer retired");
    assert!(!stderr.complete);
    assert!(stderr.text.starts_with("still alive"));
    assert!(elapsed < Duration::from_millis(500), "took {elapsed:?}");
}

#[test]
fn transient_wait_interruptions_preserve_successful_wait_kill_and_reap() {
    let child = Command::new("/bin/sh")
        .args(["-c", "exit 7"])
        .process_group(0)
        .spawn()
        .expect("spawn finite child");
    let pid = child.id();
    let mut group = ProcessGroup::new(child);
    group.faults.wait_interruptions = 1;
    let result = super::wait_bounded(
        &mut group,
        Duration::from_secs(1),
        &std::sync::atomic::AtomicBool::new(false),
    )
    .expect("normal wait retries an interruption");
    assert!(matches!(result, super::Wait::Exited));
    assert_eq!(group.faults.wait_interruptions, 0);
    assert_waitable(pid);
    group.faults.wait_interruptions = 1;
    group.kill().expect("kill retries an interruption");
    assert_eq!(group.faults.wait_interruptions, 0);
    group.faults.reap_interruptions = 1;
    assert_eq!(
        group.reap().expect("reap retries an interruption").code(),
        Some(7)
    );
    assert_reaped(pid);
}

#[test]
fn permission_grace_retries_interrupted_observation_without_resetting_time() {
    let mut observations = 0;
    super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::from_secs(1),
        || {
            observations += 1;
            if observations == 1 {
                Err(std::io::Error::from_raw_os_error(libc::EINTR))
            } else {
                Ok(true)
            }
        },
    )
    .expect("transient interrupted observation");
    assert_eq!(observations, 2);
    let started = Instant::now();
    let error = super::resolve_group_signal_error(
        std::io::Error::from_raw_os_error(libc::EPERM),
        Duration::from_millis(20),
        || Err(std::io::Error::from_raw_os_error(libc::EINTR)),
    )
    .expect_err("persistent interruptions expire with original denial");
    assert_eq!(error.raw_os_error(), Some(libc::EPERM));
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn persistent_consuming_wait_interruptions_expire_with_the_original_deadline() {
    check_fatal_cleanup("reap-interrupt");
}
