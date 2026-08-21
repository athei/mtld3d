//! Per-bucket counters for shader compiles.
//!
//! The encoder thread polls `current_counts()` once per frame and emits a
//! `shaders: N compiled in Tms (FF: …, SMx: …, M total)` info line once a
//! burst has gone idle for ≥1 second. The wall-clock check lives on the
//! encoder itself (using `rdtsc()` + `secs_to_cycles(1)`) so this module is
//! pure atomics + pure functions: no thread, no mutex, no `OnceLock`.
//!
//! `record()` accumulates count + busy-compile time per bucket. The
//! encoder reads via `current_counts()` (non-draining) on every frame
//! to detect "burst is still growing" vs "burst has stalled"; when
//! it decides to emit, it calls `drain()` which atomic-swaps each
//! bucket back to zero.

use std::{
    fmt::Write as _,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
    time::Duration,
};

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

/// Record one finished compile.
///
/// Pure atomic bumps — no thread spawn, no allocation. The encoder
/// thread observes the counts on its next frame via `current_counts()`.
pub fn record(bucket: CompileBucket, elapsed: Duration) {
    let i = bucket.index();
    BUCKETS[i].count.fetch_add(1, Ordering::Relaxed);
    // u128 nanos → u64: saturates at ~584 years; single compile fits trivially.
    BUCKETS[i].duration_ns.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// Non-draining snapshot of the per-bucket count atomics.
///
/// Used by the encoder's burst-debounce check on every frame.
#[must_use]
pub fn current_counts() -> [u32; BUCKET_COUNT] {
    std::array::from_fn(|i| BUCKETS[i].count.load(Ordering::Relaxed))
}

/// Atomic-swap each bucket back to zero and return the drained values as a `Snapshot`.
///
/// Called by the encoder once it's decided the burst has stalled and
/// is ready to emit.
pub fn drain() -> Snapshot {
    let mut snap = Snapshot {
        counts: [0; BUCKET_COUNT],
        duration_ns: [0; BUCKET_COUNT],
    };
    for (i, b) in BUCKETS.iter().enumerate() {
        snap.counts[i] = b.count.swap(0, Ordering::Relaxed);
        snap.duration_ns[i] = b.duration_ns.swap(0, Ordering::Relaxed);
    }
    snap
}

struct BucketCounters {
    count: AtomicU32,
    duration_ns: AtomicU64,
}

static BUCKETS: [BucketCounters; BUCKET_COUNT] = [const {
    BucketCounters {
        count: AtomicU32::new(0),
        duration_ns: AtomicU64::new(0),
    }
}; BUCKET_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub counts: [u32; BUCKET_COUNT],
    pub duration_ns: [u64; BUCKET_COUNT],
}

/// Pure debounce state for "burst has gone idle for ≥ `idle_cycles`".
///
/// Lives on the encoder thread alongside its other per-frame state.
/// Tracking with `rdtsc` cycles instead of `Instant::now()` so the
/// per-frame poll cost stays in the few-cycle range.
pub struct BurstTracker {
    last_seen: [u32; BUCKET_COUNT],
    last_change_tsc: u64,
    armed: bool,
}

impl Default for BurstTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BurstTracker {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_seen: [0; BUCKET_COUNT],
            last_change_tsc: 0,
            armed: false,
        }
    }

    /// Returns `true` once the current burst has gone idle for `idle_cycles`.
    ///
    /// Signalling the caller should `drain()` and emit. Resets internal
    /// state on emit so subsequent compiles start a fresh burst.
    pub fn poll(&mut self, current: [u32; BUCKET_COUNT], now_tsc: u64, idle_cycles: u64) -> bool {
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

#[cfg(test)]
mod tests;
