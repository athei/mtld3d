//! Spawn Wine's `d3d9_test.exe` for one `(leg, subtest)` and interpret it.
//!
//! Both paths, the loader and the test binary, come from the caller
//! (`--wine`/`--exe`). This module resolves nothing itself: it knows no Wine
//! directory layout and reads no environment for one, so whoever invokes the
//! runner owns where a Wine install keeps its loader and its test binaries.

use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::{
    model::{Leg, Subtest, SubtestResult},
    scan,
};

/// Per-subtest wall-clock budget.
///
/// A subtest that exceeds it is killed and reported as a crash rather than
/// blocking the whole run forever — a real reimplementation bug can deadlock
/// `d3d9_test.exe` (e.g. a refcount-forward edge that spins on a GPU wait).
/// Overridable via `MTLD3D_CONFORMANCE_TIMEOUT_SECS`; the normal subtests
/// finish in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 180;
const HEADLESS_DLL_OVERRIDES: &str = "mscoree,mshtml=";

/// How many Metal API-validation error messages a leg may log and still pass.
///
/// Zero. The layer's warnings are filtered out (see [`run_subtest`]), so every
/// message that survives is real API misuse and has to read as a regression
/// rather than as noise. The expectation lives beside the reporting code
/// because `baseline.txt` is machine-owned per-site counts whose parser
/// rejects anything else.
const MAX_VALIDATION_ERRORS: usize = 0;

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
}

/// Whether the Metal-validation error messages a run logged fail its leg.
#[must_use]
pub const fn validation_gate_failed(errors: usize) -> bool {
    errors > MAX_VALIDATION_ERRORS
}

fn subtest_timeout() -> Duration {
    let secs = std::env::var("MTLD3D_CONFORMANCE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    Duration::from_secs(secs)
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
/// under.
///
/// # Errors
///
/// Returns a message when `exe` is not a file or `wine` fails to spawn.
pub fn run_subtest(
    wine: &Path,
    exe: &Path,
    leg: Leg,
    subtest: Subtest,
) -> Result<SubtestRun, String> {
    if !exe.is_file() {
        return Err(format!(
            "test exe not found: {}; a Wine SDK bundle carries these under \
             lib/wine/tests, so re-bundle if yours predates them",
            exe.display()
        ));
    }
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
    let mut child = Command::new(wine)
        .arg(exe)
        .arg(subtest.arg())
        .env("MTL_DEBUG_LAYER", "1")
        .env("MTL_DEBUG_LAYER_ERROR_MODE", "nslog")
        .env("MTL_DEBUG_LAYER_WARNING_MODE", "ignore")
        .env("MTL_HUD_ENABLED", "0")
        .env("WINEDEBUG", "-all")
        .env("WINEDLLOVERRIDES", HEADLESS_DLL_OVERRIDES)
        .env("WINEMSYNC", "1")
        .env("RUST_LOG", "off")
        // `shaderCache.enable=false`: disable the persistent on-disk shader
        // cache (`mtld3d_shaders.bin`) for every conformance run so the DLL
        // compiles shaders fresh each run — a change to the shader translator
        // (or a SHADER_CACHE_SCHEMA bump) is always reflected without having to
        // delete a stale cache by hand.
        //
        // `color.hdr.enable=false`: the shipped default is on, but it resolves
        // off the running machine's panel, so an EDR Mac would present through
        // the tone-mapping shader while another machine blits. The baseline has
        // to mean the same thing on every machine that runs it, so pin the SDR
        // path here.
        //
        // The leg's variant appends its own entries: the `intel` variant turns
        // every `intel.*` key on so the whole suite runs under the answers an
        // Intel/AMD Mac gives.
        .env(
            "MTLD3D_CONFIG",
            format!(
                "shaderCache.enable=false;color.hdr.enable=false{}",
                leg.variant.config_entries()
            ),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", wine.display()))?;

    // Drain stdout/stderr on their own threads so a full pipe buffer can't
    // wedge the child while we poll for the timeout.
    let mut child_stdout = child.stdout.take().expect("stdout piped");
    let mut child_stderr = child.stderr.take().expect("stderr piped");
    let out_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = child_stdout.read_to_end(&mut buf);
        buf
    });
    let err_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = child_stderr.read_to_end(&mut buf);
        buf
    });

    let timeout = subtest_timeout();
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|e| format!("wait on {} failed: {e}", wine.display()))?
        {
            break status;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            timed_out = true;
            break child
                .wait()
                .map_err(|e| format!("reap of timed-out {} failed: {e}", wine.display()))?;
        }
        thread::sleep(Duration::from_millis(50));
    };
    let stdout = out_reader.join().unwrap_or_default();
    let stderr = err_reader.join().unwrap_or_default();

    // Surface Metal API-validation failures (the layer runs in `nslog` mode, so
    // these are logged rather than aborting). Deduplicated, address/number
    // normalised, prefixed with the subtest. The count is what gates the leg:
    // the per-site pass/fail counts never capture Metal misuse.
    let validation_errors =
        report_validation_errors(leg, subtest, &String::from_utf8_lossy(&stderr));

    // A timeout is a hang — treat it like a fatal signal so it surfaces as a
    // crash (and a regression vs a clean baseline) rather than a silent count.
    let signaled = timed_out || status.signal().is_some();
    let mut combined = String::from_utf8_lossy(&stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&stderr));
    if timed_out {
        use std::fmt::Write as _;
        let _ = write!(
            combined,
            "\n[conformance] subtest TIMED OUT after {}s and was killed\n",
            timeout.as_secs()
        );
    }

    // Optionally persist the full raw subtest output (every `Test failed:`
    // assertion message + the Metal-validation lines) for offline triage. The
    // normal run reduces this to per-site counts and drops the text; the actual
    // vs. expected values it carries are what distinguish a real defect from an
    // accepted pixel/caps difference. Off unless `MTLD3D_CONFORMANCE_RAW_DIR` is
    // set; a write failure is reported but never fails the run.
    save_raw_output(leg, subtest, &combined);

    Ok(SubtestRun {
        result: scan::parse_subtest_output(&combined, signaled),
        validation_errors,
    })
}

/// Persist a subtest's raw output to `$MTLD3D_CONFORMANCE_RAW_DIR/<leg>-<subtest>.log`.
///
/// Only when that variable is set. A no-op (and silent) when it is unset.
fn save_raw_output(leg: Leg, subtest: Subtest, combined: &str) {
    let Ok(dir) = std::env::var("MTLD3D_CONFORMANCE_RAW_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!(
            "  [conformance] could not create raw dir {}: {e}",
            dir.display()
        );
        return;
    }
    let path = dir.join(format!("{leg}-{subtest}.log"));
    if let Err(e) = fs::write(&path, combined) {
        eprintln!(
            "  [conformance] could not write raw log {}: {e}",
            path.display()
        );
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
