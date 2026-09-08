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

/// One budget for termination, exit observation and final reap.
const CLEANUP_GRACE: Duration = Duration::from_secs(2);

/// How long EPERM may wait for an owned group leader to become waitable.
const SIGNAL_EXIT_GRACE: Duration = Duration::from_secs(1);

/// How often an EPERM exit observation retries during its grace.
const SIGNAL_EXIT_POLL: Duration = Duration::from_millis(1);

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

    collect(ProcessGroup::new(child), timeout, on_line, |reader| {
        thread::Builder::new().spawn(reader)
    })
}

/// Collect one owned process, retaining cleanup ownership if a reader cannot start.
fn collect(
    mut group: ProcessGroup,
    timeout: Duration,
    on_line: &mut dyn FnMut(&str),
    mut spawn_reader: impl FnMut(Box<dyn FnOnce() + Send>) -> std::io::Result<thread::JoinHandle<()>>,
) -> Result<Exit, String> {
    let child = group.child.as_mut().expect("just spawned group leader");
    let pid = child.id();
    let stdout = child.stdout.take().ok_or("stdout not piped")?;
    let stderr = child.stderr.take().ok_or("stderr not piped")?;
    let (lines, received) = mpsc::channel::<String>();
    // Neither reader is joined: each ends when its pipe does, and a pipe a
    // killed process tree held open ends with the kill. Fatal cleanup ends
    // the entire runner, including readers blocked on a surviving child.
    spawn_reader(Box::new(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines.send(line).is_err() {
                break;
            }
        }
    }))
    .map_err(|e| {
        let message = format!("start stdout reader for {pid} failed: {e}");
        // Cleanup can terminate this CLI before the error reaches main.
        eprintln!("[e2e] {message}");
        message
    })?;
    // stderr comes over in chunks so that what the process wrote before it
    // died is in hand the moment it is gone; the sender dropping is the EOF.
    let gpu_hang = Arc::new(AtomicBool::new(false));
    let (stderr_tx, stderr_rx) = mpsc::channel::<Vec<u8>>();
    let stderr_gpu_hang = Arc::clone(&gpu_hang);
    spawn_reader(Box::new(move || {
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
    }))
    .map_err(|e| {
        let message = format!("start stderr reader for {pid} failed: {e}");
        // Cleanup can terminate this CLI before the error reaches main.
        eprintln!("[e2e] {message}");
        message
    })?;

    let mut timed_out = false;
    let mut reported_gpu_hang = false;
    let mut last_line = Instant::now();
    loop {
        if gpu_hang.load(Ordering::Relaxed) {
            group
                .kill()
                .map_err(|e| format!("stop {pid} failed: {e}"))?;
            reported_gpu_hang = true;
            break;
        }
        let since_line = last_line.elapsed();
        if since_line >= timeout {
            group
                .kill()
                .map_err(|e| format!("stop {pid} failed: {e}"))?;
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
            .map_err(|e| format!("wait on {pid} after stop failed: {e}"))?;
        false
    } else {
        match wait_bounded(&mut group, timeout, &gpu_hang)
            .map_err(|e| format!("wait on {pid} failed: {e}"))?
        {
            Wait::Exited => false,
            Wait::GpuHang => {
                group
                    .kill()
                    .map_err(|e| format!("stop {pid} failed: {e}"))?;
                reported_gpu_hang = true;
                group
                    .wait_killed()
                    .map_err(|e| format!("wait on {pid} after stop failed: {e}"))?;
                false
            }
            Wait::TimedOut => {
                group
                    .kill()
                    .map_err(|e| format!("stop {pid} failed: {e}"))?;
                group
                    .wait_killed()
                    .map_err(|e| format!("wait on {pid} failed: {e}"))?;
                true
            }
        }
    };
    let stderr_grace = group.cleanup_deadline.map_or(STDERR_GRACE, |deadline| {
        STDERR_GRACE.min(deadline.saturating_duration_since(Instant::now()))
    });
    let mut stderr = drain_stderr(&stderr_rx, stderr_grace);
    if !stderr.complete {
        // Something the kill did not reach still holds the pipe: a process
        // that put itself in another group. Say so, and try the kill once
        // more for whatever did land in the group since.
        group
            .kill()
            .map_err(|e| format!("stop {pid} failed: {e}"))?;
        stderr.text.push_str(
            "[e2e] stderr not collected in full: the process tree held it open past the end\n",
        );
    }
    let status = group
        .reap()
        .map_err(|e| format!("reap {pid} failed: {e}"))?;
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

/// Collect stderr chunks until the pipe closes or the total `grace` expires.
fn drain_stderr(chunks: &mpsc::Receiver<Vec<u8>>, grace: Duration) -> Stderr {
    let mut buf = Vec::new();
    let deadline = Instant::now() + grace;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let next = if remaining.is_zero() {
            Err(mpsc::RecvTimeoutError::Timeout)
        } else {
            chunks.recv_timeout(remaining)
        };
        match next {
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
    cleanup_deadline: Option<Instant>,
    #[cfg(test)]
    faults: tests::Faults,
}

impl ProcessGroup {
    const fn new(child: Child) -> Self {
        Self {
            child: Some(child),
            cleanup_deadline: None,
            #[cfg(test)]
            faults: tests::Faults::new(),
        }
    }

    /// Start the budget once, including retries made by Drop after an error.
    fn cleanup_deadline(&mut self) -> Instant {
        *self
            .cleanup_deadline
            .get_or_insert_with(|| Instant::now() + CLEANUP_GRACE)
    }

    /// Observe exit without releasing the leader's PID or group membership.
    fn observe_exit(&mut self, options: libc::c_int) -> std::io::Result<bool> {
        let child = self
            .child
            .as_ref()
            .ok_or_else(|| std::io::Error::from_raw_os_error(libc::ECHILD))?;
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
        #[cfg(test)]
        if self.faults.wait_interruptions > 0 {
            self.faults.wait_interruptions -= 1;
            return Err(std::io::Error::from_raw_os_error(libc::EINTR));
        }
        #[cfg(test)]
        let result = if self.faults.wait.is_some() {
            -1
        } else {
            result
        };
        if result == 0 {
            return Ok(info.si_pid != 0);
        }
        let error = std::io::Error::last_os_error();
        #[cfg(test)]
        let error = self
            .faults
            .wait
            .map_or(error, std::io::Error::from_raw_os_error);
        if error.raw_os_error() == Some(libc::ECHILD) {
            self.child.take();
        }
        Err(error)
    }

    /// Verify that the leader is still ours before signalling its group.
    fn kill(&mut self) -> std::io::Result<()> {
        let deadline = self.cleanup_deadline();
        loop {
            match self.observe_exit(libc::WNOHANG) {
                Ok(_) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                    cleanup_pause(deadline)?;
                }
                Err(error) => return Err(error),
            }
        }
        self.signal_group()
    }

    /// Stop descendants a group signal's fork snapshot may have missed.
    fn wait_killed(&mut self) -> std::io::Result<()> {
        let deadline = self.cleanup_deadline();
        loop {
            match self.observe_exit(libc::WNOHANG) {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
            cleanup_pause(deadline)?;
        }
        // The leader has exited, so its in-flight fork has finished. WNOWAIT
        // retains its group identity through this second signal and stderr drain.
        self.signal_group()
    }

    /// Send the final group signal before consuming the reserved identity.
    fn reap(mut self) -> std::io::Result<ExitStatus> {
        let deadline = self.cleanup_deadline();
        self.wait_killed()?;
        loop {
            match self.try_reap() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
            cleanup_pause(deadline)?;
        }
    }

    /// A nonblocking consuming wait; no group signal may follow success or ECHILD.
    fn try_reap(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let child = self
            .child
            .as_ref()
            .ok_or_else(|| std::io::Error::from_raw_os_error(libc::ECHILD))?;
        let pid = libc::pid_t::try_from(child.id()).expect("child PID fits pid_t");
        #[cfg(test)]
        let injected = if self.faults.reap_interruptions > 0 {
            self.faults.reap_interruptions -= 1;
            Some(libc::EINTR)
        } else {
            self.faults.reap
        };
        #[cfg(not(test))]
        let injected = None;
        let result = injected.map_or_else(
            || {
                let mut status = 0;
                // SAFETY: status is writable and pid selects only this owner's child.
                // WNOHANG prevents blocking. EINTR returns to the bounded caller.
                let waited = unsafe { libc::waitpid(pid, &raw mut status, libc::WNOHANG) };
                if waited == pid {
                    Ok(Some(ExitStatus::from_raw(status)))
                } else if waited == 0 {
                    Ok(None)
                } else {
                    Err(std::io::Error::last_os_error())
                }
            },
            |errno| Err(std::io::Error::from_raw_os_error(errno)),
        );
        if matches!(result, Ok(Some(_)))
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
        #[cfg(not(test))]
        let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
        #[cfg(test)]
        let result = if self.faults.signal.is_some() {
            -1
        } else {
            // SAFETY: the same owned, unreaped leader reserves this group.
            unsafe { libc::kill(-pid, libc::SIGKILL) }
        };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        #[cfg(test)]
        let error = self
            .faults
            .signal
            .map_or(error, std::io::Error::from_raw_os_error);
        let grace = self.cleanup_deadline.map_or(SIGNAL_EXIT_GRACE, |deadline| {
            SIGNAL_EXIT_GRACE.min(deadline.saturating_duration_since(Instant::now()))
        });
        resolve_group_signal_error(error, grace, || self.observe_exit(libc::WNOHANG))
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.child.is_none() {
            return;
        }
        let deadline = self.cleanup_deadline();
        if let Err(error) = self.observe_exit(libc::WNOHANG) {
            eprintln!("[e2e] observe process during cleanup failed: {error}");
        }
        if self.child.is_none() {
            return;
        }
        if let Err(error) = self.signal_group() {
            eprintln!("[e2e] stop process group during cleanup failed: {error}");
        }
        loop {
            if self.child.is_none() {
                return;
            }
            match self.wait_killed().and_then(|()| self.try_reap()) {
                Ok(Some(_)) => return,
                Ok(None) => {}
                Err(error) => {
                    eprintln!("[e2e] wait/reap process during cleanup failed: {error}");
                    if self.child.is_none() {
                        return;
                    }
                    // Retain ownership on genuine errors. Retrying the same
                    // failed syscall cannot extend the cleanup budget.
                    if error.kind() != std::io::ErrorKind::Interrupted {
                        self.fatal_cleanup(&error);
                    }
                }
            }
            if let Err(error) = cleanup_pause(deadline) {
                self.fatal_cleanup(&error);
            }
        }
    }
}

impl ProcessGroup {
    /// End this CLI while the OS can still adopt its unresolved direct child.
    fn fatal_cleanup(&self, error: &std::io::Error) -> ! {
        let pid = self
            .child
            .as_ref()
            .expect("unresolved child stays owned")
            .id();
        eprintln!(
            "[e2e] fatal cleanup for process {pid}: {error}; exit/reap not established; \
             a survivor may remain; exiting runner with code 2, macOS will adopt and eventually reap it"
        );
        // SAFETY: this is the CLI's terminal infrastructure failure. _exit ends
        // every reader thread without destructors or atexit handlers. The kernel
        // reparents any remaining children; it does not promise to kill them.
        unsafe { libc::_exit(2) }
    }
}

fn cleanup_timeout() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "process cleanup deadline expired",
    )
}

fn cleanup_pause(deadline: Instant) -> std::io::Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(cleanup_timeout());
    }
    thread::sleep(EXIT_POLL.min(remaining));
    Ok(())
}

/// Resolve a group signal error without consuming the leader's waitable status.
fn resolve_group_signal_error(
    error: std::io::Error,
    grace: Duration,
    mut observe_exit: impl FnMut() -> std::io::Result<bool>,
) -> std::io::Result<()> {
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    if error.raw_os_error() != Some(libc::EPERM) {
        return Err(error);
    }
    // macOS can stop finding an exiting group member before waitid reports
    // its exit, and skips retained zombies too. Accept EPERM only after exit
    // is observed without reaping. This does not prove descendants exited;
    // an open stderr pipe still reports a survivor.
    let deadline = Instant::now() + grace;
    loop {
        match observe_exit() {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(wait_error) if wait_error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(wait_error) => return Err(wait_error),
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(error);
        }
        thread::sleep(SIGNAL_EXIT_POLL.min(remaining));
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
        match group.observe_exit(libc::WNOHANG) {
            Ok(true) => return Ok(Wait::Exited),
            Ok(false) => {}
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
        if gpu_hang.load(Ordering::Relaxed) {
            return Ok(Wait::GpuHang);
        }
        if Instant::now() >= deadline {
            return Ok(Wait::TimedOut);
        }
        thread::sleep(EXIT_POLL.min(deadline.saturating_duration_since(Instant::now())));
    }
}

#[cfg(test)]
mod tests;
