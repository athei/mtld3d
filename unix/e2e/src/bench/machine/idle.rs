//! Wait for consecutive quiet machine samples without changing how busy is judged.

use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use super::{INTERVAL, Sample, keep, measured_sample, sample};

/// Consecutive measured samples required before either leg starts a timed process.
const QUIET_SAMPLES: u8 = 3;

/// How often a continuing wait reports progress.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Record the prestart machine sample, optionally waiting for three quiet samples first.
///
/// Zero `timeout` preserves advisory sampling. Otherwise every attempt is
/// kept under `idle-<what>`, including unavailable samples and quiet samples
/// whose streak a later busy sample breaks. The accepted sample also goes
/// into the ordinary machine file the report reads. A timeout preserves
/// completed rounds and returns an error before another process can start.
///
/// # Errors
///
/// Returns a message when an artifact cannot be written or the machine did
/// not produce three consecutive quiet samples before the timeout.
pub fn keep_when_ready(
    dir: &Path,
    what: &str,
    legs: &[PathBuf],
    timeout: Duration,
) -> Result<(), String> {
    if timeout.is_zero() {
        return keep(dir, what, &sample(legs));
    }
    let started = Instant::now();
    wait_with(dir, what, timeout, || {
        let taken = measured_sample(legs);
        if taken.is_none() {
            // A failed listing can return immediately. Keep failures bounded
            // by the same sampling cadence rather than spinning on `ps`.
            thread::sleep(INTERVAL);
        }
        (started.elapsed(), taken)
    })
}

/// The quiet streak, independent of the operating system and the evidence files.
struct QuietStreak {
    consecutive: u8,
    timeout: Duration,
}

impl QuietStreak {
    const fn new(timeout: Duration) -> Self {
        Self {
            consecutive: 0,
            timeout,
        }
    }

    /// Judge a sample after it completed, so a late third quiet sample cannot pass.
    fn observe(&mut self, elapsed: Duration, sample: Option<&Sample>) -> Decision {
        let reason = sample.map_or_else(
            || Some("measured CPU sample unavailable".to_owned()),
            Sample::busy,
        );
        if reason.is_some() {
            self.consecutive = 0;
        } else {
            self.consecutive += 1;
        }
        if elapsed >= self.timeout {
            return Decision::TimedOut;
        }
        if self.consecutive >= QUIET_SAMPLES {
            Decision::Ready
        } else {
            Decision::Waiting(
                reason.unwrap_or_else(|| {
                    format!("quiet sample {}/{QUIET_SAMPLES}", self.consecutive)
                }),
            )
        }
    }
}

enum Decision {
    Waiting(String),
    Ready,
    TimedOut,
}

/// Run the gate with an injected sampler, retaining each attempt before deciding what follows.
fn wait_with(
    dir: &Path,
    what: &str,
    timeout: Duration,
    mut take: impl FnMut() -> (Duration, Option<Sample>),
) -> Result<(), String> {
    let attempts = dir.join(format!("idle-{what}"));
    fs::create_dir_all(&attempts).map_err(|e| format!("{}: {e}", attempts.display()))?;
    eprintln!(
        "bench-ab: waiting up to {} s for {QUIET_SAMPLES} quiet samples before {} ({what})",
        timeout.as_secs(),
        dir.display()
    );
    let mut streak = QuietStreak::new(timeout);
    let mut number = 0u64;
    let mut progress = Duration::ZERO;
    loop {
        let (elapsed, sample) = take();
        number += 1;
        let status = sample.as_ref().map_or_else(
            || "unavailable measured CPU sample".to_owned(),
            |sample| {
                sample
                    .busy()
                    .map_or_else(|| "quiet".to_owned(), |r| format!("busy {r}"))
            },
        );
        let kernel = if sample
            .as_ref()
            .is_some_and(|sample| sample.kernel_task.is_none())
        {
            "kernel_task unavailable\n"
        } else {
            ""
        };
        let text = format!(
            "attempt {number}\nelapsed_ms {}\nstatus {status}\n{kernel}{}",
            elapsed.as_millis(),
            sample.as_ref().map_or_else(String::new, Sample::text),
        );
        let path = attempts.join(format!("sample-{number:06}.txt"));
        fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        match streak.observe(elapsed, sample.as_ref()) {
            Decision::Ready => {
                let sample = sample.expect("only a valid quiet sample can finish the streak");
                keep(dir, what, &sample)?;
                keep_outcome(dir, what, "ready", number, elapsed)?;
                eprintln!(
                    "bench-ab: machine ready after {:.3} s and {number} samples ({what})",
                    elapsed.as_secs_f64()
                );
                return Ok(());
            }
            Decision::TimedOut => {
                keep_outcome(dir, what, "timeout", number, elapsed)?;
                return Err(format!(
                    "idle wait timed out after {:.3} s before {} ({what}); no performance \
                     verdict, completed rounds and samples remain in {}",
                    elapsed.as_secs_f64(),
                    dir.display(),
                    attempts.display()
                ));
            }
            Decision::Waiting(reason) => {
                if elapsed.saturating_sub(progress) >= PROGRESS_INTERVAL {
                    eprintln!(
                        "bench-ab: still waiting after {:.3} s ({what}): {reason}",
                        elapsed.as_secs_f64()
                    );
                    progress = elapsed;
                }
            }
        }
    }
}

/// Write the terminal gate outcome separately from the compatible machine sample.
fn keep_outcome(
    dir: &Path,
    what: &str,
    status: &str,
    attempts: u64,
    elapsed: Duration,
) -> Result<(), String> {
    let path = dir.join(format!("idle-{what}.txt"));
    let text = format!(
        "status {status}\nattempts {attempts}\nwait_ms {}\nquiet_samples {QUIET_SAMPLES}\n",
        elapsed.as_millis()
    );
    fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests;
