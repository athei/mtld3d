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
    let mut child = Command::new(wine)
        .arg(exe)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| format!("failed to spawn {} {}: {e}", wine.display(), exe.display()))?;

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
            kill_group(&child);
            reported_gpu_hang = true;
            break;
        }
        let since_line = last_line.elapsed();
        if since_line >= timeout {
            kill_group(&child);
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
    let (status, hung) = if reported_gpu_hang {
        let status = child
            .wait()
            .map_err(|e| format!("wait on {} after GPU hang failed: {e}", exe.display()))?;
        (status, false)
    } else {
        match wait_bounded(&mut child, timeout, &gpu_hang) {
            Wait::Exited(status) => (status, false),
            Wait::GpuHang => {
                kill_group(&child);
                reported_gpu_hang = true;
                let status = child
                    .wait()
                    .map_err(|e| format!("wait on {} after GPU hang failed: {e}", exe.display()))?;
                (status, false)
            }
            Wait::TimedOut => {
                kill_group(&child);
                let status = child
                    .wait()
                    .map_err(|e| format!("wait on {} failed: {e}", exe.display()))?;
                (status, true)
            }
        }
    };
    let mut stderr = drain_stderr(&stderr_rx, STDERR_GRACE);
    if !stderr.complete {
        // Something the kill did not reach still holds the pipe: a process
        // that put itself in another group. Say so, and try the kill once
        // more for whatever did land in the group since.
        kill_group(&child);
        stderr.text.push_str(
            "[e2e] stderr not collected in full: the process tree held it open past the end\n",
        );
    }
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

/// The result of waiting for a process after its stdout closed.
enum Wait {
    Exited(ExitStatus),
    GpuHang,
    TimedOut,
}

/// Wait for exit, a driver hang report, or `timeout` after stdout closed.
fn wait_bounded(child: &mut Child, timeout: Duration, gpu_hang: &AtomicBool) -> Wait {
    let deadline = Instant::now() + timeout;
    loop {
        // A wait that fails is treated as a process that will not exit: the
        // caller kills the group and waits again, and that wait reports.
        if let Ok(Some(status)) = child.try_wait() {
            return Wait::Exited(status);
        }
        if gpu_hang.load(Ordering::Relaxed) {
            return Wait::GpuHang;
        }
        if Instant::now() >= deadline {
            return Wait::TimedOut;
        }
        thread::sleep(EXIT_POLL);
    }
}

/// SIGKILL the child's process group: the child and everything it forked.
///
/// The child is its own group leader (`process_group(0)` at spawn), so the
/// group id is its pid. A group that is already gone is not an error.
fn kill_group(child: &Child) {
    let Ok(pid) = i32::try_from(child.id()) else {
        return;
    };
    // SAFETY: kill(2) with a negative pid signals the group; it touches no
    // memory of ours and a stale id is reported as ESRCH, not acted on.
    unsafe {
        let _ = libc::kill(-pid, libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests;
