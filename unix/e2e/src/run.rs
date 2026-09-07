//! Spawn one test process under Wine and stream what it prints.
//!
//! stdout is handed back line by line as it arrives, because the tests of
//! the whole suite run in one process and a report has to show progress
//! while they run, and because a process that stops reporting is the only
//! sign of a hung test: the watchdog kills it once no line has arrived for
//! the caller's timeout. stderr is collected as it arrives too, but handed
//! back whole once the process has ended; it carries the panic reports,
//! Wine's own diagnostics, and the driver report that stops a process as
//! soon as it says the GPU hung.
//!
//! The same timeout bounds a process that closes stdout and then never
//! exits: it is killed once the timeout passes with no exit. What comes
//! after the process is gone gets only a short grace: a descendant that
//! still holds stderr (Wine's debugger walking a crashed process's DWARF is
//! one) has nothing of the test's left to say, so the run drains what has
//! arrived, kills the group once more and moves on, rather than paying the
//! timeout a second time for an end of stderr that may never come. The
//! child runs in a process group of its own so the kill takes every process
//! Wine forked under it, and never the wineserver the caller booted for the
//! whole run, which belongs to the caller's group.

use std::{
    io::{BufRead, BufReader},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

/// How a process ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitKind {
    /// It exited on its own with this code.
    Code(i32),
    /// The kernel ended it with this signal.
    Signal(i32),
    /// The watchdog killed it after this long without a line on stdout.
    TimedOut(Duration),
    /// It closed stdout and was killed after this long without exiting.
    Hung(Duration),
}

impl ExitKind {
    /// The reason in a report: `exit code 5`, `signal 11`, `no output for 60 s`.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Code(code) => format!("exit code {code}"),
            Self::Signal(signal) => format!("signal {signal}"),
            Self::TimedOut(after) => format!("no output for {} s", after.as_secs()),
            Self::Hung(after) => {
                format!("no exit for {} s after its last line", after.as_secs())
            }
        }
    }
}

/// A process's end: its pid, how it ended, and everything it wrote to stderr.
pub struct Exit {
    pub pid: u32,
    pub kind: ExitKind,
    pub stderr: String,
    /// The driver reported a GPU hang before this process ended.
    pub gpu_hang: bool,
}

/// Whether output reports one of the driver's hung-GPU codes.
pub fn is_gpu_hang_report(text: &str) -> bool {
    GPU_HANG_MARKERS.iter().any(|marker| text.contains(marker))
}

/// The driver's codes for a hung GPU.
///
/// The conformance runner keeps the same two markers. The runners are
/// separate binaries with no runner-support crate between them, while
/// `mtld3d-shared` is the runtime's wire-format crate.
const GPU_HANG_MARKERS: [&str; 2] = [
    "kIOAccelCommandBufferCallbackErrorHang",
    "kIOAccelCommandBufferCallbackErrorSubmissionsIgnored",
];

/// How often the bounded wait for the process's exit looks again.
const EXIT_POLL: Duration = Duration::from_millis(20);

/// How long stderr may stay open after the process has ended.
///
/// A pipe every holder has died on closes within milliseconds of the kill;
/// one still open past this has a survivor on it, whose output is not the
/// test's.
const STDERR_GRACE: Duration = Duration::from_secs(1);

/// Run `wine <exe> <args...>` in the exe's directory, streaming stdout lines to `on_line`.
///
/// Killed, and reported as [`ExitKind::TimedOut`], once `timeout` passes
/// without a stdout line; killed, and reported as [`ExitKind::Hung`], once
/// it passes after stdout closed without the process exiting. stderr is
/// what arrived before the end plus [`STDERR_GRACE`] after it. The
/// environment is inherited whole: the caller owns `MTLD3D_CONFIG` and the
/// Wine variables.
///
/// # Errors
///
/// Returns a message when the process cannot be spawned or waited for.
pub fn run(
    wine: &Path,
    exe: &Path,
    args: &[String],
    timeout: Duration,
    on_line: &mut dyn FnMut(&str),
) -> Result<Exit, String> {
    let cwd = exe.parent().unwrap_or_else(|| Path::new("."));
    let child = Command::new(wine)
        .arg(exe)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| format!("failed to spawn {} {}: {e}", wine.display(), exe.display()))?;

    let mut group = ProcessGroup { child: Some(child) };
    let child = group.child.as_mut().expect("just spawned group leader");
    let pid = child.id();
    let stdout = child.stdout.take().ok_or("stdout not piped")?;
    let stderr = child.stderr.take().ok_or("stderr not piped")?;
    let (lines, received) = mpsc::channel::<String>();
    // Neither reader is joined: each ends when its pipe does, and a pipe a
    // killed process tree held open ends with the kill.
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines.send(line).is_err() {
                break;
            }
        }
    });
    // stderr comes over in chunks so that what the process wrote before it
    // died is in hand the moment it is gone; the sender dropping is the EOF.
    let gpu_hang = Arc::new(AtomicBool::new(false));
    let (stderr_tx, stderr_rx) = mpsc::channel::<Vec<u8>>();
    let stderr_gpu_hang = Arc::clone(&gpu_hang);
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = Vec::new();
        while let Ok(1..) = reader.read_until(b'\n', &mut line) {
            if is_gpu_hang_report(&String::from_utf8_lossy(&line)) {
                stderr_gpu_hang.store(true, Ordering::Relaxed);
            }
            if stderr_tx.send(std::mem::take(&mut line)).is_err() {
                break;
            }
        }
    });

    let mut timed_out = false;
    let mut reported_gpu_hang = false;
    let mut last_line = Instant::now();
    loop {
        if gpu_hang.load(Ordering::Relaxed) {
            group
                .kill()
                .map_err(|e| format!("stop {} failed: {e}", exe.display()))?;
            reported_gpu_hang = true;
            break;
        }
        let since_line = last_line.elapsed();
        if since_line >= timeout {
            group
                .kill()
                .map_err(|e| format!("stop {} failed: {e}", exe.display()))?;
            timed_out = true;
            break;
        }
        let until_timeout = timeout
            .checked_sub(since_line)
            .expect("elapsed time checked against the timeout");
        match received.recv_timeout(EXIT_POLL.min(until_timeout)) {
            Ok(line) => {
                on_line(&line);
                last_line = Instant::now();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let hung = if reported_gpu_hang || timed_out {
        group
            .wait_killed()
            .map_err(|e| format!("wait on {} after stop failed: {e}", exe.display()))?;
        false
    } else {
        match wait_bounded(&mut group, timeout, &gpu_hang)
            .map_err(|e| format!("wait on {} failed: {e}", exe.display()))?
        {
            Wait::Exited => false,
            Wait::GpuHang => {
                group
                    .kill()
                    .map_err(|e| format!("stop {} failed: {e}", exe.display()))?;
                reported_gpu_hang = true;
                group
                    .wait_killed()
                    .map_err(|e| format!("wait on {} after stop failed: {e}", exe.display()))?;
                false
            }
            Wait::TimedOut => {
                group
                    .kill()
                    .map_err(|e| format!("stop {} failed: {e}", exe.display()))?;
                group
                    .wait_killed()
                    .map_err(|e| format!("wait on {} failed: {e}", exe.display()))?;
                true
            }
        }
    };
    let mut stderr = drain_stderr(&stderr_rx, STDERR_GRACE);
    if !stderr.complete {
        // Something the kill did not reach still holds the pipe: a process
        // that put itself in another group. Say so, and try the kill once
        // more for whatever did land in the group since.
        group
            .kill()
            .map_err(|e| format!("stop {} failed: {e}", exe.display()))?;
        stderr.text.push_str(
            "[e2e] stderr not collected in full: the process tree held it open past the end\n",
        );
    }
    let status = group
        .reap()
        .map_err(|e| format!("reap {} failed: {e}", exe.display()))?;
    let stderr = stderr.text;
    let kind = if timed_out {
        ExitKind::TimedOut(timeout)
    } else if hung {
        ExitKind::Hung(timeout)
    } else if let Some(signal) = status.signal() {
        ExitKind::Signal(signal)
    } else {
        ExitKind::Code(status.code().unwrap_or(-1))
    };
    let gpu_hang = reported_gpu_hang || gpu_hang.load(Ordering::Relaxed);
    Ok(Exit {
        pid,
        kind,
        stderr,
        gpu_hang,
    })
}

/// Everything stderr delivered, and whether its end was seen.
struct Stderr {
    text: String,
    complete: bool,
}

/// Collect stderr chunks until the pipe closes or `grace` passes without one.
fn drain_stderr(chunks: &mpsc::Receiver<Vec<u8>>, grace: Duration) -> Stderr {
    let mut buf = Vec::new();
    loop {
        match chunks.recv_timeout(grace) {
            Ok(chunk) => buf.extend_from_slice(&chunk),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Stderr {
                    text: String::from_utf8_lossy(&buf).into_owned(),
                    complete: true,
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Stderr {
                    text: String::from_utf8_lossy(&buf).into_owned(),
                    complete: false,
                };
            }
        }
    }
}

/// A group whose leader stays unreaped until all group signals have been sent.
///
/// On macOS an unreaped child reserves its PID and process-group membership.
/// Only this owner waits for the child; observing exit with WNOWAIT retains
/// that reservation. ECHILD revokes ownership instead of trusting a numeric ID.
struct ProcessGroup {
    child: Option<Child>,
}

impl ProcessGroup {
    /// Observe exit without releasing the leader's PID or group membership.
    fn observe_exit(&mut self, options: libc::c_int) -> std::io::Result<bool> {
        let child = self
            .child
            .as_ref()
            .ok_or_else(|| std::io::Error::from_raw_os_error(libc::ECHILD))?;
        loop {
            // SAFETY: siginfo_t contains integers and raw pointers, valid when zeroed.
            // macOS leaves it untouched for WNOHANG when the child is still running.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            // SAFETY: info is writable and P_PID selects this owner's direct child.
            // WNOWAIT observes exit without consuming the child's waitable status.
            let result = unsafe {
                libc::waitid(
                    libc::P_PID,
                    child.id(),
                    &raw mut info,
                    libc::WEXITED | libc::WNOWAIT | options,
                )
            };
            if result == 0 {
                return Ok(info.si_pid != 0);
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            if error.raw_os_error() == Some(libc::ECHILD) {
                self.child.take();
            }
            return Err(error);
        }
    }

    /// Verify that the leader is still ours before signalling its group.
    fn kill(&mut self) -> std::io::Result<()> {
        self.observe_exit(libc::WNOHANG)?;
        self.signal_group()
    }

    /// Stop descendants a group signal's fork snapshot may have missed.
    fn wait_killed(&mut self) -> std::io::Result<()> {
        self.observe_exit(0)?;
        // The leader has exited, so its in-flight fork has finished. WNOWAIT
        // retains its group identity through this second signal and stderr drain.
        self.signal_group()
    }

    /// Release the reserved identity, consuming the ability to signal the group.
    fn reap(mut self) -> std::io::Result<ExitStatus> {
        let result = self
            .child
            .as_mut()
            .ok_or_else(|| std::io::Error::from_raw_os_error(libc::ECHILD))?
            .wait();
        if result.is_ok()
            || result
                .as_ref()
                .is_err_and(|e| e.raw_os_error() == Some(libc::ECHILD))
        {
            self.child.take();
        }
        result
    }

    /// Signal the group while its leader remains owned and unreaped.
    fn signal_group(&mut self) -> std::io::Result<()> {
        let child = self
            .child
            .as_ref()
            .ok_or_else(|| std::io::Error::from_raw_os_error(libc::ECHILD))?;
        let pid = libc::pid_t::try_from(child.id()).expect("child PID fits pid_t");
        // SAFETY: the child spawned as its own group leader and remains unreaped,
        // so its PID reserves this group's identity until the consuming reap.
        let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        // macOS skips zombies when signalling a group and returns EPERM when
        // no live member could be signalled. The retained zombie alone therefore
        // produces EPERM too. This does not prove every descendant exited;
        // an open stderr pipe still reports a survivor.
        if error.raw_os_error() == Some(libc::ESRCH)
            || (error.raw_os_error() == Some(libc::EPERM) && self.observe_exit(libc::WNOHANG)?)
        {
            return Ok(());
        }
        Err(error)
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.child.is_none() {
            return;
        }
        if let Err(error) = self.observe_exit(libc::WNOHANG) {
            eprintln!("[e2e] observe process during cleanup failed: {error}");
        }
        // ECHILD means another waiter consumed the identity. Never signal it.
        if self.child.is_none() {
            return;
        }
        if let Err(error) = self.signal_group() {
            eprintln!("[e2e] stop process group during cleanup failed: {error}");
        }
        if let Err(error) = self.wait_killed() {
            eprintln!("[e2e] wait for stopped process during cleanup failed: {error}");
        }
        if let Some(mut child) = self.child.take()
            && let Err(error) = child.wait()
        {
            eprintln!("[e2e] reap process during cleanup failed: {error}");
        }
    }
}

/// The result of waiting for a process after its stdout closed.
enum Wait {
    Exited,
    GpuHang,
    TimedOut,
}

/// Wait for exit, a driver hang report, or `timeout` after stdout closed.
fn wait_bounded(
    group: &mut ProcessGroup,
    timeout: Duration,
    gpu_hang: &AtomicBool,
) -> std::io::Result<Wait> {
    let deadline = Instant::now() + timeout;
    loop {
        if group.observe_exit(libc::WNOHANG)? {
            return Ok(Wait::Exited);
        }
        if gpu_hang.load(Ordering::Relaxed) {
            return Ok(Wait::GpuHang);
        }
        if Instant::now() >= deadline {
            return Ok(Wait::TimedOut);
        }
        thread::sleep(EXIT_POLL);
    }
}

#[cfg(test)]
mod tests;
