//! Spawn Wine's `d3d9_test.exe` for one `(leg, subtest)` and interpret it.
//!
//! Both paths, the loader and the test binary, come from the caller
//! (`--wine`/`--exe`). This module resolves nothing itself: it knows no Wine
//! directory layout and reads no environment for one, so whoever invokes the
//! runner owns where a Wine install keeps its loader and its test binaries.

use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs,
    io::{BufRead, BufReader, Read},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    model::{Leg, Subtest, SubtestResult},
    scan,
};

/// The per-subtest wall-clock budget a run has unless its caller sets another.
///
/// A subtest that exceeds it is sampled, killed and reported as a crash rather
/// than blocking the whole run forever: a real reimplementation bug can
/// deadlock `d3d9_test.exe` (e.g. a refcount-forward edge that spins on a GPU
/// wait). The normal subtests finish in seconds. The caller reads
/// `MTLD3D_CONFORMANCE_TIMEOUT_SECS` into [`Launch::timeout`], see
/// [`timeout_from_env`].
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);
const HEADLESS_DLL_OVERRIDES: &str = "mscoree,mshtml=";

/// How long the sample of a timed-out process may take.
///
/// `sample` reads the process for [`SAMPLE_SECONDS`] and then symbolicates,
/// which on a translated Wine process with a few dozen threads has taken ten
/// seconds. A sampler still running past this is killed and the sample says
/// so: the process it was meant to explain must not park the run a second
/// time.
const SAMPLE_BUDGET: Duration = Duration::from_secs(60);

/// How long `sample` watches the process before symbolicating.
const SAMPLE_SECONDS: &str = "2";

/// How long a pipe may stay open after the process the runner spawned is gone.
///
/// A pipe every holder has died on ends within milliseconds of the last of
/// them; one still open past this has a survivor on it whose output is not
/// the subtest's. Wine starts `explorer.exe /desktop` for the prefix's first
/// process that needs a desktop, it inherits the subtest's pipes, and it puts
/// itself in a process group of its own, so neither the budget nor the group
/// kill reaches it and the end of those pipes is as far away as the next
/// process of that prefix leaves it. Each pipe gets its own grace, so the
/// wait after the process is gone is at most twice this.
const DRAIN_GRACE: Duration = Duration::from_secs(1);

/// The driver's codes for a hung GPU, as Metal prints them to stderr.
///
/// The first names the command buffer that hung the GPU, the second every
/// one the driver ignores afterwards for the process's earlier errors. Both
/// mean the same for the counts: every read after the line comes off a GPU
/// that runs nothing, so the subtest is stopped the moment either arrives
/// and the leg ends there (see [`GPU_HANG_EXIT`]).
const GPU_HANG_MARKERS: [&str; 2] = [
    "kIOAccelCommandBufferCallbackErrorHang",
    "kIOAccelCommandBufferCallbackErrorSubmissionsIgnored",
];

/// The runner's exit code for a leg a GPU hang cut short.
///
/// Distinct from a regression (1) and a usage or spawn error (2): the leg
/// has no verdict. On a hosted runner the GPU stays hung for the rest of the
/// machine's life, so the answer is a fresh machine, not another attempt.
pub const GPU_HANG_EXIT: u8 = 3;

/// How many Metal API-validation error messages a leg may log and still pass.
///
/// Zero. The layer's warnings are filtered out (see [`run_subtest`]), so every
/// message that survives is real API misuse and has to read as a regression
/// rather than as noise. The expectation lives beside the reporting code
/// because `baseline.txt` is machine-owned per-site counts whose parser
/// rejects anything else.
const MAX_VALIDATION_ERRORS: usize = 0;

/// What every spawn of the run shares.
///
/// Built once by the caller from the command line and the environment, so
/// the spawn itself reads neither: a test can hand it a shell script as the
/// loader and a scratch directory as the raw dir.
pub struct Launch {
    /// The Wine loader.
    pub wine: PathBuf,
    /// The `d3d9_test.exe` to run.
    pub exe: PathBuf,
    /// The `RUST_LOG` filter the test process runs under.
    ///
    /// `off` for a gating run: the counts are the measurement and our log is
    /// noise there. A repeat run raises it to see what the layer did before
    /// a process ended without its summary.
    pub log: String,
    /// Where each subtest's raw output and log file go, when they are kept.
    ///
    /// `None` keeps nothing. Set, it is made absolute, since the test process
    /// is handed the same directory as a Windows path.
    pub raw_dir: Option<PathBuf>,
    /// The wall-clock budget of one subtest.
    ///
    /// A process still running at the end of it is sampled, then killed with
    /// its group, and the subtest reads as a crash. [`DEFAULT_TIMEOUT`] unless
    /// the caller has another: the spawn reads no environment.
    pub timeout: Duration,
}

/// How many detail lines one validation message keeps.
///
/// Metal's reports run to a handful of them; the cap keeps a stderr that
/// stopped looking like one from pasting a whole subtest into a single
/// message. A message that hits it ends in an ellipsis.
const MAX_DETAIL_LINES: usize = 8;

/// One subtest's outcome.
///
/// The parsed per-site result, plus how many distinct Metal API-validation
/// error messages the run logged for the caller to gate on.
pub struct SubtestRun {
    /// Failing sites, the crash bit and the marked-failure tallies.
    pub result: SubtestResult,
    /// Distinct Metal API-validation error messages the subtest logged.
    pub validation_errors: usize,
    /// The GPU hung under the subtest, which was stopped on the driver's line.
    ///
    /// Every later subtest would run on a GPU that ignores its command
    /// buffers, so the leg's counts past this point mean nothing: the caller
    /// stops the leg and exits with [`GPU_HANG_EXIT`].
    pub gpu_hang: bool,
}

/// Whether the Metal-validation error messages a run logged fail its leg.
#[must_use]
pub const fn validation_gate_failed(errors: usize) -> bool {
    errors > MAX_VALIDATION_ERRORS
}

/// The subtest budget the environment asks for, else [`DEFAULT_TIMEOUT`].
///
/// `MTLD3D_CONFORMANCE_TIMEOUT_SECS`, a positive number of seconds; anything
/// else reads as unset. Read once by the caller and handed to every spawn
/// through [`Launch::timeout`].
#[must_use]
pub fn timeout_from_env() -> Duration {
    std::env::var("MTLD3D_CONFORMANCE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map_or(DEFAULT_TIMEOUT, Duration::from_secs)
}

/// `wine --version`, or `"unknown"` if it can't be determined.
#[must_use]
pub fn wine_version(wine: &Path) -> String {
    Command::new(wine)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Run one subtest in its own process and interpret its output.
///
/// Each subtest is a separate `wine` invocation so a crash in one cannot poison
/// another's counts. The Metal-debug/log/Wine-debug environment is overridden
/// (not inherited) so a validation abort can't mask the failure counts — the
/// same overrides the shell runner used.
///
/// `leg` selects nothing about the binary (the caller already picked it); its
/// variant adds the config entries the run is measured under, and the whole
/// leg is the label the results, raw logs and validation lines are recorded
/// under. `attempt` is the run's number in a repeat run, which keeps every
/// run's raw output; `None` is the one run of a gating leg.
///
/// stderr is read as it arrives, and the driver's GPU-hang line kills the
/// process at once rather than after its budget: what the subtest reads from
/// then on is zeros off a GPU that runs nothing, and the wait would only make
/// the same verdict cost minutes.
///
/// The child runs in a process group of its own, and a kill, for the hang or
/// for the budget, takes that group. It does not take every descendant: a
/// process that puts itself in a group of its own is outside it, keeps the
/// pipes it inherited, and can hold them long past the subtest, whether the
/// subtest exited on its own or was killed. So the collection of what the
/// pipes hold is bounded too ([`DRAIN_GRACE`]), the subtest is judged on what
/// arrived before that, and the raw log says so when the collection was cut
/// short. The wineserver the caller booted for the whole run is in the
/// caller's group and is never touched.
///
/// A process still running at `launch.timeout` is sampled before its group is
/// killed, and the sample is kept beside the raw output, or printed when
/// nothing is kept. The raw log of a hang ends in its `TIMED OUT` line and
/// says nothing about where the process was; the sample is that account.
///
/// # Errors
///
/// Returns a message when `exe` is not a file or `wine` fails to spawn.
pub fn run_subtest(
    launch: &Launch,
    leg: Leg,
    subtest: Subtest,
    attempt: Option<u32>,
) -> Result<SubtestRun, String> {
    if !launch.exe.is_file() {
        return Err(format!(
            "test exe not found: {}; a Wine SDK bundle carries these under \
             lib/wine/tests, so re-bundle if yours predates them",
            launch.exe.display()
        ));
    }
    let raw = launch
        .raw_dir
        .as_deref()
        .map(|dir| RawTarget::new(dir, leg, subtest, attempt));
    // Metal API validation is left ON (`nslog` mode) so every conformance run
    // surfaces Metal misuse (format/attachment/binding mismatches, oversized
    // inline binds, …). `nslog` *logs* validation failures to stderr instead of
    // aborting, so it cannot mask the per-site counts the way `error`/`abort`
    // mode would — the historical reason the layer was disabled here.
    //
    // The layer's *warnings* are ignored. They are performance hints, not
    // misuse: a resource bound to an encoder no draw went on to read, a state
    // setter overwritten before the next draw. A leg emits thousands of them
    // (one `setVisibilityResultMode` pair per occlusion query, one binding per
    // shader that stops reading a slot), all deduplicated to a handful of
    // lines that read exactly like the error lines and bury them. Only errors
    // are reported, so a new validation line means a new misuse.
    let mut child = Command::new(&launch.wine)
        .arg(&launch.exe)
        .arg(subtest.arg())
        .env("MTL_DEBUG_LAYER", "1")
        .env("MTL_DEBUG_LAYER_ERROR_MODE", "nslog")
        .env("MTL_DEBUG_LAYER_WARNING_MODE", "ignore")
        .env("MTL_HUD_ENABLED", "0")
        .env("WINEDEBUG", "-all")
        .env("WINEDLLOVERRIDES", HEADLESS_DLL_OVERRIDES)
        .env("WINEMSYNC", "1")
        .env("RUST_LOG", &launch.log)
        .env("MTLD3D_CONFIG", config_entries(leg, raw.as_ref()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", launch.wine.display()))?;

    // Drain stdout/stderr on their own threads so a full pipe buffer can't
    // wedge the child while we poll for the timeout.
    let out_chunks = drain_on_thread(child.stdout.take().expect("stdout piped"));
    let hung = Arc::new(AtomicBool::new(false));
    let err_chunks = drain_stderr_on_thread(
        child.stderr.take().expect("stderr piped"),
        Arc::clone(&hung),
    );

    let timeout = launch.timeout;
    let start = Instant::now();
    // The sample of a process that ran out of budget, taken before its kill.
    let mut timed_out: Option<String> = None;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("wait on {} failed: {e}", launch.wine.display()))?
        {
            break status;
        }
        if hung.load(Ordering::Relaxed) {
            kill_group(&child);
            break child
                .wait()
                .map_err(|e| format!("reap of hung {} failed: {e}", launch.wine.display()))?;
        }
        if start.elapsed() >= timeout {
            timed_out = Some(sample_process(child.id()));
            kill_group(&child);
            break child
                .wait()
                .map_err(|e| format!("reap of timed-out {} failed: {e}", launch.wine.display()))?;
        }
        thread::sleep(Duration::from_millis(50));
    };
    // Each pipe from here on is a bounded collection, not a wait for its end:
    // the process is gone and what still holds a pipe is not it.
    let stdout = collect_pipe(&out_chunks, Instant::now() + DRAIN_GRACE);
    let stderr = collect_pipe(&err_chunks, Instant::now() + DRAIN_GRACE);
    // Read after the collection, so a line the process printed just before
    // ending on its own counts too: its later reads were zeros all the same.
    let gpu_hang = hung.load(Ordering::Relaxed);

    // Surface Metal API-validation failures (the layer runs in `nslog` mode, so
    // these are logged rather than aborting). Deduplicated, address/number
    // normalised, prefixed with the subtest. The count is what gates the leg:
    // the per-site pass/fail counts never capture Metal misuse.
    let validation_errors =
        report_validation_errors(leg, subtest, &String::from_utf8_lossy(&stderr.bytes));

    // A timeout is a hang: treat it like a fatal signal so it surfaces as a
    // crash (and a regression vs a clean baseline) rather than a silent count.
    // A GPU hang is one too: the counts stop meaning anything at its line.
    let signaled = timed_out.is_some() || gpu_hang || status.signal().is_some();
    let mut combined = String::from_utf8_lossy(&stdout.bytes).into_owned();
    combined.push_str(&String::from_utf8_lossy(&stderr.bytes));
    if !stdout.complete || !stderr.complete {
        report_cut_short(leg, subtest, &mut combined);
    }
    if let Some(sample) = &timed_out {
        let _ = write!(
            combined,
            "\n[conformance] subtest TIMED OUT after {}s and was killed{}\n",
            timeout.as_secs(),
            raw.as_ref().map_or_else(String::new, |raw| {
                format!(
                    "; the sample taken before the kill is {}",
                    raw.sample_file_name()
                )
            })
        );
        report_timeout(leg, subtest, timeout, sample, raw.as_ref());
    } else {
        let _ = write!(combined, "\n{}\n", exit_trailer(status));
    }
    if gpu_hang {
        let after = start.elapsed().as_secs();
        eprintln!(
            "  [{leg}/{subtest}] GPU hang: the driver reported it after {after}s; the subtest \
             was stopped and the leg ends here"
        );
        let _ = write!(
            combined,
            "\n[conformance] GPU HANG reported by the driver after {after}s; the subtest was \
             stopped and the leg ends here\n"
        );
    }

    // Optionally persist the full raw subtest output (every `Test failed:`
    // assertion message + the Metal-validation lines) for offline triage. The
    // normal run reduces this to per-site counts and drops the text; the actual
    // vs. expected values it carries are what distinguish a real defect from an
    // accepted pixel/caps difference. A write failure is reported but never
    // fails the run.
    if let Some(raw) = &raw {
        raw.save(&combined);
    }

    Ok(SubtestRun {
        result: scan::parse_subtest_output(&combined, signaled),
        validation_errors,
        gpu_hang,
    })
}

/// Read a pipe on a thread of its own, handing each chunk over as it arrives.
///
/// A pipe nobody reads fills, and the writer blocks on it: a child polled for
/// its budget, or a sampler polled for its own, must never wait on the poller.
/// The thread is never joined, because the end of the pipe is not the caller's
/// to wait for: it ends when the last holder of the write end closes it, and
/// the chunks it still sends after the caller has collected go nowhere, since
/// the send fails once the receiver is dropped.
fn drain_on_thread(mut reader: impl Read + Send + 'static) -> mpsc::Receiver<Vec<u8>> {
    let (chunks, received) = mpsc::channel();
    thread::spawn(move || {
        let mut buf = [0_u8; 8192];
        while let Ok(read @ 1..) = reader.read(&mut buf) {
            if chunks.send(buf[..read].to_vec()).is_err() {
                return;
            }
        }
    });
    received
}

/// Everything a pipe delivered inside the grace, and whether its end was seen.
struct Drained {
    bytes: Vec<u8>,
    complete: bool,
}

/// Collect a pipe's chunks until it ends or `deadline` passes.
///
/// A pipe that ends delivers every chunk it sent before the caller sees the
/// disconnect, so the bound only ever cuts off what has not arrived.
fn collect_pipe(chunks: &mpsc::Receiver<Vec<u8>>, deadline: Instant) -> Drained {
    let mut bytes = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let next = if remaining.is_zero() {
            Err(mpsc::RecvTimeoutError::Timeout)
        } else {
            chunks.recv_timeout(remaining)
        };
        match next {
            Ok(chunk) => bytes.extend_from_slice(&chunk),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Drained {
                    bytes,
                    complete: true,
                };
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                return Drained {
                    bytes,
                    complete: false,
                };
            }
        }
    }
}

/// Say, on stderr and in the raw log, that the output is not all of it.
///
/// The counts are still the counts of what arrived, so the verdict is
/// unchanged; what a reader must not have to guess is that the end of the
/// output is missing rather than absent.
fn report_cut_short(leg: Leg, subtest: Subtest, combined: &mut String) {
    let after = DRAIN_GRACE.as_secs();
    eprintln!(
        "  [{leg}/{subtest}] output cut short: something outside the subtest's process group \
         still held its pipes {after}s after it ended"
    );
    let _ = write!(
        combined,
        "\n[conformance] output cut short after {after}s: something outside the subtest's \
         process group still held its pipes open, so the end of this output is missing\n"
    );
}

/// A `sample` of the process, taken while it still runs.
///
/// The kill that follows leaves a raw log ending in `TIMED OUT` and the
/// process's own log silent on a thread parked in a syscall, so the sample is
/// the one account of where a hang was. The tool's stderr and how it ended
/// stay in the text when it fails, so a process it could not read is reported
/// rather than dropped, and a sampler still running at [`SAMPLE_BUDGET`] is
/// killed and the text says so.
fn sample_process(pid: u32) -> String {
    let pid = pid.to_string();
    let mut child = match Command::new("sample")
        .args([pid.as_str(), SAMPLE_SECONDS, "-mayDie"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return format!("[conformance] sample could not be started: {e}\n"),
    };
    let out_chunks = drain_on_thread(child.stdout.take().expect("stdout piped"));
    let err_chunks = drain_on_thread(child.stderr.take().expect("stderr piped"));
    let started = Instant::now();
    let ended = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if started.elapsed() < SAMPLE_BUDGET => {
                thread::sleep(Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("was killed after {}s", SAMPLE_BUDGET.as_secs()));
            }
            Err(e) => break Err(format!("could not be waited for: {e}")),
        }
    };
    let out = collect_pipe(&out_chunks, Instant::now() + DRAIN_GRACE);
    let err = collect_pipe(&err_chunks, Instant::now() + DRAIN_GRACE);
    let mut text = String::from_utf8_lossy(&out.bytes).into_owned();
    let stderr = String::from_utf8_lossy(&err.bytes).into_owned();
    if !out.complete || !err.complete {
        let _ = write!(
            text,
            "\n[conformance] the sample's own output was cut short after {}s\n",
            DRAIN_GRACE.as_secs()
        );
    }
    match ended {
        Ok(status) if status.success() => {}
        Ok(status) => {
            let _ = write!(
                text,
                "\n[conformance] sample ended with {status}:\n{stderr}"
            );
        }
        Err(how) => {
            let _ = write!(text, "\n[conformance] sample {how}:\n{stderr}");
        }
    }
    text
}

/// Say on stderr that the subtest ran out of budget, and keep its sample.
///
/// With a raw dir the sample is a file beside the raw log and the line names
/// it. Without one nothing is kept, so the sample itself follows the line.
fn report_timeout(
    leg: Leg,
    subtest: Subtest,
    timeout: Duration,
    sample: &str,
    raw: Option<&RawTarget>,
) {
    let after = timeout.as_secs();
    if let Some(raw) = raw {
        raw.save_sample(sample);
        eprintln!(
            "  [{leg}/{subtest}] TIMED OUT after {after}s; the process was sampled before the \
             kill: {}",
            raw.dir.join(raw.sample_file_name()).display()
        );
    } else {
        eprintln!(
            "  [{leg}/{subtest}] TIMED OUT after {after}s; the process was sampled before the \
             kill (set MTLD3D_CONFORMANCE_RAW_DIR to keep the sample as a file):"
        );
        eprint!("{sample}");
    }
}

/// Drain stderr on a thread of its own, raising `hung` on the GPU-hang line.
///
/// A chunk is a line, so the line is seen while the process still runs and
/// the poll loop can kill it on the flag instead of waiting out the budget.
/// Every byte is handed over, invalid UTF-8 included: the check reads a lossy
/// copy of each line and the drain never stops on one.
fn drain_stderr_on_thread(
    stderr: impl Read + Send + 'static,
    hung: Arc<AtomicBool>,
) -> mpsc::Receiver<Vec<u8>> {
    let (chunks, received) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = Vec::new();
        while let Ok(1..) = reader.read_until(b'\n', &mut line) {
            if is_gpu_hang_line(&String::from_utf8_lossy(&line)) {
                hung.store(true, Ordering::Relaxed);
            }
            if chunks.send(std::mem::take(&mut line)).is_err() {
                return;
            }
        }
    });
    received
}

/// Whether a stderr line is the driver reporting a hung GPU.
fn is_gpu_hang_line(line: &str) -> bool {
    GPU_HANG_MARKERS.iter().any(|marker| line.contains(marker))
}

/// SIGKILL the child's process group: the child and everything it forked.
///
/// The child is its own group leader (`process_group(0)` at spawn), so the
/// group id is its pid. A group that is already gone is not an error. The
/// e2e runner keeps the same function for the same reason; the two runners
/// are separate binaries with no crate between them for a process helper.
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

/// How the process ended, as the raw log's last line.
///
/// A process that ends without the framework's summary is a crash whatever
/// else its output holds, and this line is what tells the shapes apart. Wine
/// ends a process with an unhandled Win32 exception through the exception
/// code, of which unix keeps the low byte, so an access violation
/// (`0xC0000005`) reads as `code 5`. A run that reached its summary exits
/// with the framework's failure count capped at 255, which the layer carries
/// through the `TerminateProcess` its detach ends the process with. A signal
/// is the number alone (11 `SIGSEGV`, 10 `SIGBUS`, 6 `SIGABRT`, 9 `SIGKILL`):
/// the scanner's crash markers are signal names, and a name here would turn
/// a fault the process survived into a crash.
fn exit_trailer(status: ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("[conformance] subtest exited: code {code}"),
        (None, Some(signal)) => format!("[conformance] subtest exited: signal {signal}"),
        (None, None) => "[conformance] subtest exited: unknown status".to_owned(),
    }
}

/// The `MTLD3D_CONFIG` a subtest runs under.
///
/// `shaderCache.enable=false`: disable the persistent on-disk shader cache
/// (`mtld3d_shaders.bin`) for every conformance run so the DLL compiles
/// shaders fresh each run — a change to the shader translator (or a
/// `SHADER_CACHE_SCHEMA` bump) is always reflected without having to delete a
/// stale cache by hand.
///
/// `color.hdr.enable=false`: the shipped default is on, but it resolves off
/// the running machine's panel, so an EDR Mac would present through the
/// tone-mapping shader while another machine blits. The baseline has to mean
/// the same thing on every machine that runs it, so pin the SDR path here.
///
/// The leg's variant appends its own entries: the `intel` variant turns every
/// `intel.*` key on so the whole suite runs under the answers an Intel/AMD Mac
/// gives. A kept run adds `log.dir`, so the process's log file lands beside
/// its raw output.
fn config_entries(leg: Leg, raw: Option<&RawTarget>) -> String {
    let mut entries = format!(
        "shaderCache.enable=false;color.hdr.enable=false{}",
        leg.variant.config_entries()
    );
    if let Some(raw) = raw {
        let _ = write!(entries, ";log.dir={}", raw.log_dir_dos());
    }
    entries
}

/// Where one subtest's raw output and its process's log file go.
///
/// `<dir>/<leg>-<subtest>[-<attempt>].log` for the output and a directory of
/// the same stem for the log file, one per process, so the layer's retention
/// of ten files per directory never prunes one run's log to make room for
/// another's. A timed-out process's sample is `<stem>.sample.txt` beside them.
struct RawTarget {
    dir: PathBuf,
    stem: String,
}

impl RawTarget {
    fn new(dir: &Path, leg: Leg, subtest: Subtest, attempt: Option<u32>) -> Self {
        let stem = attempt.map_or_else(
            || format!("{leg}-{subtest}"),
            |n| format!("{leg}-{subtest}-{n}"),
        );
        Self {
            dir: dir.to_path_buf(),
            stem,
        }
    }

    /// The log directory as the test process names it.
    ///
    /// The layer reads `log.dir` on the Windows side, where the unix root is
    /// drive `Z:`, the same convention the e2e legs use for their `LOG_DIR`.
    fn log_dir_dos(&self) -> String {
        format!("Z:{}", self.dir.join(&self.stem).display())
    }

    /// The file the sample of a timed-out process is kept as, beside the raw log.
    fn sample_file_name(&self) -> String {
        format!("{}.sample.txt", self.stem)
    }

    /// Persist the raw output; a failure is reported and never fails the run.
    fn save(&self, combined: &str) {
        self.write(&format!("{}.log", self.stem), combined);
    }

    /// Persist the sample of a timed-out process; a failure is reported and never fails the run.
    fn save_sample(&self, sample: &str) {
        self.write(&self.sample_file_name(), sample);
    }

    fn write(&self, file_name: &str, text: &str) {
        if let Err(e) = fs::create_dir_all(&self.dir) {
            eprintln!(
                "  [conformance] could not create raw dir {}: {e}",
                self.dir.display()
            );
            return;
        }
        let path = self.dir.join(file_name);
        if let Err(e) = fs::write(&path, text) {
            eprintln!("  [conformance] could not write {}: {e}", path.display());
        }
    }
}

/// Print a deduplicated, number-normalised summary of any Metal API-validation messages.
///
/// The subtest logged them rather than aborting — the layer runs in `nslog`
/// mode. A message prints as its opening line plus its detail lines, indented
/// under it: the opening line names the check that fired (`Sampler Descriptor
/// Validation`) and the detail names what failed it. Returns how many distinct
/// messages were printed, which is what the leg gates on.
fn report_validation_errors(leg: Leg, subtest: Subtest, stderr: &str) -> usize {
    let seen = validation_errors(stderr);
    for msg in &seen {
        let mut lines = msg.lines();
        if let Some(header) = lines.next() {
            eprintln!("  [{leg}/{subtest}] metal-validation: {header}");
        }
        for detail in lines {
            eprintln!("      {detail}");
        }
    }
    seen.len()
}

/// The distinct Metal API-validation error messages in a subtest's stderr.
///
/// A recognised line opens a message and the lines under it are its detail,
/// which is where the layer names the property it rejected. Volatile addresses
/// and counts collapse to `N` so a repeated message reports once. The layer's
/// warnings never reach here (the run switches them off), so every match is
/// misuse.
fn validation_errors(stderr: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut i = 0;
    while i < lines.len() {
        if !opens_validation_message(lines[i].trim()) {
            i += 1;
            continue;
        }
        let mut msg = normalize_numbers(lines[i].trim());
        i += 1;
        let mut detail = 0;
        while i < lines.len() && continues_validation_message(lines[i]) {
            if detail < MAX_DETAIL_LINES {
                msg.push('\n');
                msg.push_str(&normalize_numbers(lines[i].trim()));
            } else if detail == MAX_DETAIL_LINES {
                msg.push_str("\n…");
            }
            detail += 1;
            i += 1;
        }
        seen.insert(msg);
    }
    seen
}

/// Whether a line opens a Metal API-validation message.
///
/// The layer heads a multi-line report with the check that fired
/// (`… Validation`) and also emits stand-alone error lines carrying no such
/// header, so both shapes open one. Its start-up notice is not an error.
fn opens_validation_message(line: &str) -> bool {
    line.contains("does not match")
        || line.contains("is missing from")
        || line.contains("must be <=")
        || line.contains("incorrect type of texture")
        || line.contains("Insufficient")
        || line.contains("exceeds the limit")
        || (line.contains(" Validation") && !line.contains("Validation Enabled"))
}

/// Whether a line is detail under the validation message above it.
///
/// The layer writes its detail lines unadorned, so a message runs until
/// something the run can attribute to another writer: the next `NSLog` line,
/// a Wine channel line, or a blank line.
fn continues_validation_message(line: &str) -> bool {
    let line = line.trim();
    !line.is_empty() && !is_nslog_line(line) && !is_wine_channel_line(line)
}

/// Whether a line carries an `NSLog` timestamp (`YYYY-MM-DD HH:MM:SS.mmm …`).
fn is_nslog_line(line: &str) -> bool {
    let b = line.as_bytes();
    b.len() > 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
}

/// Whether a line is Wine's own channel output (`0000:err:d3d9:…`).
fn is_wine_channel_line(line: &str) -> bool {
    let Some((id, rest)) = line.split_once(':') else {
        return false;
    };
    id.len() >= 4
        && id.bytes().all(|b| b.is_ascii_hexdigit())
        && matches!(
            rest.split(':').next(),
            Some("err" | "warn" | "fixme" | "trace")
        )
}

/// Collapse hex literals (`0x…`) and decimal runs to `N`.
///
/// Volatile addresses and counts then don't defeat deduplication.
fn normalize_numbers(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'0' && i + 1 < bytes.len() && (bytes[i + 1] | 0x20) == b'x' {
            i += 2;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                i += 1;
            }
            out.push_str("0xN");
        } else if c.is_ascii_digit() {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push('N');
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests;
