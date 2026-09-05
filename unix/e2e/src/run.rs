//! Spawn one test process under Wine and stream what it prints.
//!
//! stdout is handed back line by line as it arrives, because the tests of
//! the whole suite run in one process and a report has to show progress
//! while they run, and because a process that stops reporting is the only
//! sign of a hung test: the watchdog kills it once no line has arrived for
//! the caller's timeout. stderr is collected whole; it carries the panic
//! reports and Wine's own diagnostics, read after the process has ended.

use std::{
    io::{BufRead, BufReader, Read},
    os::unix::process::ExitStatusExt,
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::Duration,
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
}

impl ExitKind {
    /// The reason in a report: `exit code 5`, `signal 11`, `no output for 60 s`.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Code(code) => format!("exit code {code}"),
            Self::Signal(signal) => format!("signal {signal}"),
            Self::TimedOut(after) => format!("no output for {} s", after.as_secs()),
        }
    }
}

/// A process's end: how, and everything it wrote to stderr.
pub struct Exit {
    pub kind: ExitKind,
    pub stderr: String,
}

/// Run `wine <exe> <args...>` in the exe's directory, streaming stdout lines to `on_line`.
///
/// Killed, and reported as [`ExitKind::TimedOut`], once `timeout` passes
/// without a stdout line. The environment is inherited whole: the caller
/// owns `MTLD3D_CONFIG` and the Wine variables.
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
        .spawn()
        .map_err(|e| format!("failed to spawn {} {}: {e}", wine.display(), exe.display()))?;

    let stdout = child.stdout.take().ok_or("stdout not piped")?;
    let mut stderr = child.stderr.take().ok_or("stderr not piped")?;
    let (lines, received) = mpsc::channel::<String>();
    let stdout_reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines.send(line).is_err() {
                break;
            }
        }
    });
    let stderr_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    });

    let mut timed_out = false;
    loop {
        match received.recv_timeout(timeout) {
            Ok(line) => on_line(&line),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                timed_out = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child
        .wait()
        .map_err(|e| format!("wait on {} failed: {e}", exe.display()))?;
    let _ = stdout_reader.join();
    let stderr = stderr_reader.join().unwrap_or_default();
    let kind = if timed_out {
        ExitKind::TimedOut(timeout)
    } else if let Some(signal) = status.signal() {
        ExitKind::Signal(signal)
    } else {
        ExitKind::Code(status.code().unwrap_or(-1))
    };
    Ok(Exit { kind, stderr })
}
