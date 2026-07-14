//! Spawn Wine's `d3d9_test.exe` for one `(arch, subtest)` and interpret it.
//!
//! The wine loader is located via the ambient `WINE_SDK` environment variable
//! (the global Wine install that `make install` populated with our builtin
//! `d3d9.dll`). `WINE_SDK` is mandatory for the whole Makefile, so the runner
//! reads it from the environment rather than taking it as a conformance flag —
//! the only conformance-specific input is the Wine build tree (`--wine-build`).

use std::{
    fs,
    io::Read,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::{
    model::{Arch, Subtest, SubtestResult},
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

fn subtest_timeout() -> Duration {
    let secs = std::env::var("MTLD3D_CONFORMANCE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Resolve the wine loader from `WINE_SDK`.
///
/// # Errors
///
/// Returns a message when `WINE_SDK` is unset or its `bin/wine` is absent.
pub fn wine_binary() -> Result<PathBuf, String> {
    let sdk = std::env::var("WINE_SDK").map_err(|_| {
        "WINE_SDK is not set (the global Wine install holding our builtin d3d9.dll)".to_owned()
    })?;
    let wine = PathBuf::from(sdk).join("bin/wine");
    if !wine.is_file() {
        return Err(format!("wine loader not found at {}", wine.display()));
    }
    Ok(wine)
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
/// # Errors
///
/// Returns a message when the per-arch `d3d9_test.exe` is missing or `wine`
/// fails to spawn.
pub fn run_subtest(
    wine: &Path,
    wine_build: &Path,
    arch: Arch,
    subtest: Subtest,
) -> Result<SubtestResult, String> {
    let exe = wine_build
        .join("dlls/d3d9/tests")
        .join(arch.wine_target_dir())
        .join("d3d9_test.exe");
    if !exe.is_file() {
        return Err(format!(
            "test exe not found: {} — build the Wine d3d9 tests first",
            exe.display()
        ));
    }
    // Metal API validation is left ON (`nslog` mode) so every conformance run
    // surfaces Metal misuse (format/attachment/binding mismatches, oversized
    // inline binds, …). `nslog` *logs* validation failures to stderr instead of
    // aborting, so it cannot mask the per-site counts the way `error`/`abort`
    // mode would — the historical reason the layer was disabled here.
    let mut child = Command::new(wine)
        .arg(&exe)
        .arg(subtest.arg())
        .env("MTL_DEBUG_LAYER", "1")
        .env("MTL_DEBUG_LAYER_ERROR_MODE", "nslog")
        .env("MTL_DEBUG_LAYER_WARNING_MODE", "nslog")
        .env("MTL_HUD_ENABLED", "0")
        .env("WINEDEBUG", "-all")
        .env("WINEMSYNC", "1")
        .env("RUST_LOG", "off")
        // `shaderCache.enable=false`: disable the persistent on-disk shader
        // cache (`mtld3d_shaders.bin`) for every conformance run so the DLL
        // compiles shaders fresh each run — a change to the shader translator
        // (or a SHADER_CACHE_SCHEMA bump) is always reflected without having to
        // delete a stale cache by hand.
        //
        // `query.flushImmediate=false`: restore the spec-correct *blocking*
        // `GetData(D3DGETDATA_FLUSH)` for occlusion queries. The runtime default
        // (`true`) returns a permissive stub immediately for API-thread
        // throughput, but conformance wants the real GPU pixel count, so flip it
        // off here (occlusion-only; EVENT/TIMESTAMP `GetData` are unaffected).
        .env(
            "MTLD3D_CONFIG",
            "shaderCache.enable=false;query.flushImmediate=false",
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
    // normalised, prefixed with the subtest — a standing watch for Metal misuse
    // that the per-site pass/fail counts don't capture.
    report_validation_errors(arch, subtest, &String::from_utf8_lossy(&stderr));

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
    save_raw_output(arch, subtest, &combined);

    Ok(scan::parse_subtest_output(&combined, signaled))
}

/// Persist a subtest's raw output to `$MTLD3D_CONFORMANCE_RAW_DIR/<arch>-<subtest>.log`.
///
/// Only when that variable is set. A no-op (and silent) when it is unset.
fn save_raw_output(arch: Arch, subtest: Subtest, combined: &str) {
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
    let path = dir.join(format!("{arch}-{subtest}.log"));
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
/// mode. Volatile addresses and counts collapse to `N` so a repeated error
/// reports once.
fn report_validation_errors(arch: Arch, subtest: Subtest, stderr: &str) {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for line in stderr.lines() {
        let l = line.trim();
        let is_validation = l.contains("does not match")
            || l.contains("is missing from")
            || l.contains("must be <=")
            || l.contains("incorrect type of texture")
            || l.contains("Insufficient")
            || l.contains("exceeds the limit")
            || (l.contains(" Validation") && !l.contains("Validation Enabled"));
        if is_validation {
            seen.insert(normalize_numbers(l));
        }
    }
    for msg in &seen {
        eprintln!("  [{arch}/{subtest}] metal-validation: {msg}");
    }
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
