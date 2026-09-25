//! Per-bucket counters for shader compiles.
//!
//! One `CompileStats` lives on each `FrameEncoder`, so the numbers a device
//! reports are its own: a process with two devices compiles on two encoder
//! threads, and a shared counter would have each of them draining the
//! other's work into its summary line.
//!
//! `record()` accumulates count + busy-compile time per bucket, called from
//! the cold path of the encoder's own shader resolution. `poll_drain()` runs
//! once per frame and answers with a `Snapshot` once the burst has been
//! stable and nonzero for ≥1 second, which is when the encoder emits its
//! `shaders: N compiled in Tms (FF: …, SMx: …, M total)` info line. The
//! wall-clock source stays with the caller (`rdtsc()` + `secs_to_cycles(1)`)
//! so this module is plain fields and pure functions: no thread, no mutex,
//! no atomics, no `OnceLock`.

use std::{fmt::Write as _, time::Duration};

use super::LOG_TARGET;

const BUCKET_COUNT: usize = 4;
const ORDER: [CompileBucket; BUCKET_COUNT] = [
    CompileBucket::Ff,
    CompileBucket::Sm1,
    CompileBucket::Sm2,
    CompileBucket::Sm3,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileBucket {
    Ff,
    Sm1,
    Sm2,
    Sm3,
}

impl CompileBucket {
    /// Map a DXSO shader-model major version to its bucket.
    ///
    /// Valid majors are 1, 2, 3 — anything else is DX10+ territory and
    /// should never reach a d3d9 caller; out-of-range values return
    /// `None` after a one-shot warn so the count stays exact.
    pub fn from_sm_major(major: u8) -> Option<Self> {
        match major {
            1 => Some(Self::Sm1),
            2 => Some(Self::Sm2),
            3 => Some(Self::Sm3),
            other => {
                mtld3d_shared::log_once_warn_by!(
                    target: LOG_TARGET,
                    key: u64::from(other),
                    "shader_compile_stats: unrecognised SM major {other} → not counted"
                );
                None
            }
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Ff => 0,
            Self::Sm1 => 1,
            Self::Sm2 => 2,
            Self::Sm3 => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Ff => "FF",
            Self::Sm1 => "SM1",
            Self::Sm2 => "SM2",
            Self::Sm3 => "SM3",
        }
    }
}

/// One encoder's compile counters and the debounce that decides when to emit.
///
/// Owned by the `FrameEncoder` that does the compiling, so every value it
/// reports was produced by its own device. Single-threaded by construction:
/// the encoder thread both records and drains, which is why the counters are
/// plain integers.
pub struct CompileStats {
    counts: [u32; BUCKET_COUNT],
    duration_ns: [u64; BUCKET_COUNT],
    /// Libraries a worker thread built with no draw waiting on them, since the last drain.
    async_compiled: u32,
    /// Draws left out of their frame because a build they needed was still in flight.
    draws_skipped: u32,
    burst: BurstTracker,
}

impl Default for CompileStats {
    fn default() -> Self {
        Self::new()
    }
}

impl CompileStats {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; BUCKET_COUNT],
            duration_ns: [0; BUCKET_COUNT],
            async_compiled: 0,
            draws_skipped: 0,
            burst: BurstTracker::new(),
        }
    }

    /// Count one library a worker built while no draw waited for it.
    pub const fn record_async_compile(&mut self) {
        self.async_compiled = self.async_compiled.saturating_add(1);
    }

    /// Count one draw left out of its frame for a build still in flight.
    pub const fn record_skipped_draw(&mut self) {
        self.draws_skipped = self.draws_skipped.saturating_add(1);
    }

    /// Take the asynchronous-build counts, leaving both at zero.
    ///
    /// Called when [`Self::poll_drain`] answers, so the counts ride the same
    /// debounced summary line as the compiles they belong to.
    pub const fn take_async(&mut self) -> AsyncCounts {
        let counts = AsyncCounts {
            compiled: self.async_compiled,
            skipped: self.draws_skipped,
        };
        self.async_compiled = 0;
        self.draws_skipped = 0;
        counts
    }

    /// Record one finished compile.
    ///
    /// Two adds and no allocation, on the cold path of a shader resolution
    /// that just compiled.
    pub fn record(&mut self, bucket: CompileBucket, elapsed: Duration) {
        let i = bucket.index();
        self.counts[i] = self.counts[i].saturating_add(1);
        // u128 nanos → u64: saturates at ~584 years; single compile fits trivially.
        let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.duration_ns[i] = self.duration_ns[i].saturating_add(ns);
    }

    /// Drain the counters once the burst has been idle for `idle_cycles`.
    ///
    /// Called once per frame with the current TSC reading. `None` means the
    /// burst is still growing, has not gone quiet yet, or there is nothing
    /// to report; `Some` leaves every bucket back at zero.
    pub fn poll_drain(&mut self, now_tsc: u64, idle_cycles: u64) -> Option<Snapshot> {
        if !self.burst.poll(self.counts, now_tsc, idle_cycles) {
            return None;
        }
        let snap = Snapshot {
            counts: self.counts,
            duration_ns: self.duration_ns,
        };
        self.counts = [0; BUCKET_COUNT];
        self.duration_ns = [0; BUCKET_COUNT];
        Some(snap)
    }
}

/// The asynchronous-build side of one summary, from [`CompileStats::take_async`].
pub struct AsyncCounts {
    /// Libraries a worker built while no draw waited for them.
    pub compiled: u32,
    /// Draws left out of their frame while a build they needed was in flight.
    pub skipped: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub counts: [u32; BUCKET_COUNT],
    pub duration_ns: [u64; BUCKET_COUNT],
}

/// Pure debounce state for "burst has gone idle for ≥ `idle_cycles`".
///
/// Half of [`CompileStats`], which owns it. Tracking with `rdtsc` cycles
/// instead of `Instant::now()` so the per-frame poll cost stays in the
/// few-cycle range.
struct BurstTracker {
    last_seen: [u32; BUCKET_COUNT],
    last_change_tsc: u64,
    armed: bool,
}

impl BurstTracker {
    const fn new() -> Self {
        Self {
            last_seen: [0; BUCKET_COUNT],
            last_change_tsc: 0,
            armed: false,
        }
    }

    /// Returns `true` once the current burst has gone idle for `idle_cycles`.
    ///
    /// Signalling the owner should drain and emit. Resets internal state on
    /// emit so subsequent compiles start a fresh burst.
    fn poll(&mut self, current: [u32; BUCKET_COUNT], now_tsc: u64, idle_cycles: u64) -> bool {
        if current.iter().all(|&c| c == 0) {
            // No work to emit. Clear armed state so a future burst
            // starts clean even if the previous one was reset by an
            // external drain.
            self.armed = false;
            self.last_seen = [0; BUCKET_COUNT];
            return false;
        }
        if !self.armed || current != self.last_seen {
            self.last_seen = current;
            self.last_change_tsc = now_tsc;
            self.armed = true;
            return false;
        }
        if now_tsc.saturating_sub(self.last_change_tsc) < idle_cycles {
            return false;
        }
        // Idle long enough: caller will drain. Disarm so we don't
        // re-emit the (post-drain) zero state immediately.
        self.last_seen = [0; BUCKET_COUNT];
        self.last_change_tsc = 0;
        self.armed = false;
        true
    }
}

#[must_use]
pub fn format_summary(snap: &Snapshot, verb: &str, total: u32) -> String {
    let total_count: u32 = snap.counts.iter().sum();
    let total_ms: u64 = snap.duration_ns.iter().sum::<u64>() / 1_000_000;
    // Fixed-column layout so consecutive lines stack readably.
    // Always show every bucket (FF, SM1, SM2, SM3) — hiding zeros made
    // the line widths jitter every emit. Widths cover the realistic
    // ranges (counts up to 9999, ms up to 9999, total up to 99999);
    // larger values still print, just past the column edge.
    let mut out = format!("shaders: {total_count:>4} {verb} in {total_ms:>4}ms (");
    let mut first = true;
    for bucket in &ORDER {
        if !first {
            out.push_str(", ");
        }
        first = false;
        let _ = write!(
            out,
            "{}: {:>3}",
            bucket.label(),
            snap.counts[bucket.index()]
        );
    }
    let _ = write!(out, ", {total:>5} total)");
    out
}

/// The `, N compiled async, M draws skipped` tail of a summary, empty when both are zero.
#[must_use]
pub fn format_async_suffix(counts: &AsyncCounts) -> String {
    if counts.compiled == 0 && counts.skipped == 0 {
        return String::new();
    }
    format!(
        ", {} compiled async, {} draws skipped",
        counts.compiled, counts.skipped
    )
}

#[cfg(test)]
mod tests;
