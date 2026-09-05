//! Spawn one test process under Wine and stream what it prints.
//!
//! stdout is handed back line by line as it arrives, because the tests of
//! the whole suite run in one process and a report has to show progress
//! while they run, and because a process that stops reporting is the only
//! sign of a hung test: the watchdog kills it once no line has arrived for
//! the caller's timeout. stderr is collected whole; it carries the panic
//! reports and Wine's own diagnostics, read after the process has ended.
//!
//! The same timeout bounds the end of the process: one that closes stdout
//! and then never exits, or leaves a descendant holding stderr open, is
//! killed once the timeout passes with no exit or no end of stderr. The
//! child runs in a process group of its own so the kill takes every process
//! Wine forked under it, and never the wineserver the caller booted for the
//! whole run, which belongs to the caller's group.

use std::{
    io::{BufRead, BufReader, Read},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
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

/// A process's end: how, and everything it wrote to stderr.
pub struct Exit {
    pub kind: ExitKind,
    pub stderr: String,
}

/// How often the bounded wait for the process's exit looks again.
const EXIT_POLL: Duration = Duration::from_millis(20);

/// Run `wine <exe> <args...>` in the exe's directory, streaming stdout lines to `on_line`.
///
/// Killed, and reported as [`ExitKind::TimedOut`], once `timeout` passes
/// without a stdout line; killed, and reported as [`ExitKind::Hung`], once
/// it passes after stdout closed without the process exiting. The
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

    let stdout = child.stdout.take().ok_or("stdout not piped")?;
    let mut stderr = child.stderr.take().ok_or("stderr not piped")?;
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
    let (stderr_tx, stderr_rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        let _ = stderr_tx.send(String::from_utf8_lossy(&buf).into_owned());
    });

    let mut timed_out = false;
    loop {
        match received.recv_timeout(timeout) {
            Ok(line) => on_line(&line),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                kill_group(&child);
                timed_out = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let (status, hung) = if let Some(status) = wait_bounded(&mut child, timeout) {
        (status, false)
    } else {
        kill_group(&child);
        let status = child
            .wait()
            .map_err(|e| format!("wait on {} failed: {e}", exe.display()))?;
        (status, true)
    };
    let stderr = stderr_rx.recv_timeout(timeout).unwrap_or_else(|_| {
        kill_group(&child);
        let mut stderr = stderr_rx.recv_timeout(EXIT_POLL).unwrap_or_default();
        stderr.push_str(
            "[e2e] stderr not collected in full: the process tree held it open past the timeout\n",
        );
        stderr
    });
    let kind = if timed_out {
        ExitKind::TimedOut(timeout)
    } else if hung {
        ExitKind::Hung(timeout)
    } else if let Some(signal) = status.signal() {
        ExitKind::Signal(signal)
    } else {
        ExitKind::Code(status.code().unwrap_or(-1))
    };
    Ok(Exit { kind, stderr })
}

/// The process's exit status if it exits within `timeout`, `None` if it does not.
fn wait_bounded(child: &mut Child, timeout: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        // A wait that fails is treated as a process that will not exit: the
        // caller kills the group and waits again, and that wait reports.
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status);
        }
        if Instant::now() >= deadline {
            return None;
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
