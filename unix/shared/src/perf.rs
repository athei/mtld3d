//! Cross-linkage-unit perf-tracking primitives.
//!
//! Houses the runtime gate (`PERF_TRACKING_ENABLED` cached from
//! `RUST_LOG=mtld3d::perf=info`) and the two state-agnostic RAII timers
//! (`CycleSetTimer`, `CycleAddTimer`) — both used by `d3d9.dll` PE-side
//! AND `mtld3d.so` unix-side. Each cdylib that links `mtld3d-shared`
//! statically gets its own static instance, which matches each cdylib's
//! own `env_logger` filter cache.
//!
//! State-dependent perf (e.g. `ApiTimer` which buckets cycles into
//! `ApiPerfState`, the per-frame summary aggregator `PerfWindow`, the
//! `Summary` renderer) lives in `mtld3d-core::perf` because it's tied
//! to PE-side D3D9 state and has no unix-side analog.
//!
//! ## Compile-time gate
//!
//! Under `cfg(not(perf_tracking))` (i.e. `make` without `PERF=1`), the
//! gate helpers become `const fn` returning `false`, both timer structs
//! become unit structs with `const fn` no-op constructors and no `Drop`,
//! and the statics + the latch fn vanish. LLVM then dead-code-eliminates
//! every `let _t = CycleSetTimer::start(...)` call site to literal zero
//! bytes after thin-LTO inlining.

#[cfg(perf_tracking)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(perf_tracking)]
use log::{Level, log_enabled};

#[cfg(perf_tracking)]
use crate::tsc::rdtsc;

/// Cached `log_enabled!(target: "mtld3d::perf", Level::Info)` result.
///
/// Latched once at logger init via `init_tracking_enabled`. Read on the
/// hot path of every `ApiTimer` / `CycleSetTimer` / `CycleAddTimer`
/// construction — a single `Relaxed` atomic load instead of the
/// `env_logger` filter walk that `log_enabled!` would run per call.
///
/// Latched at `Info` so `PERF=1` builds print the 5-second summary by
/// default — the gate already requires an opt-in build, so the runtime
/// cost is paid only by users who explicitly asked for the dashboard.
/// Silence with `RUST_LOG=mtld3d::perf=warn`.
#[cfg(perf_tracking)]
static PERF_TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Cached `log_enabled!(target: "mtld3d::d3d9::passes", Level::Trace)` result.
///
/// Drives `bump_pair_stats` and the per-pass / present-texture /
/// per-pair dump emitted from `log_frame_summary`. Lives next to the
/// perf gate because the pair dump is computed from the same per-frame
/// state the perf window aggregates.
#[cfg(perf_tracking)]
static PAIR_STATS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Latch `PERF_TRACKING_ENABLED` and `PAIR_STATS_ENABLED` from `RUST_LOG`.
///
/// Call once per cdylib after `env_logger::try_init` — each cdylib has its
/// own `log` statics, so the cache is per-runtime. `d3d9.dll` calls this
/// from `init_logger`; `mtld3d.so` calls it from `init_logger_handler`.
#[cfg(perf_tracking)]
pub fn init_tracking_enabled() {
    let on = log_enabled!(target: "mtld3d::perf", Level::Info);
    PERF_TRACKING_ENABLED.store(on, Ordering::Relaxed);
    let passes_on = log_enabled!(target: "mtld3d::d3d9::passes", Level::Trace);
    PAIR_STATS_ENABLED.store(passes_on, Ordering::Relaxed);
}

/// Compile-time no-op when `cfg(not(perf_tracking))`.
///
/// The entire perf infrastructure was elided at build time, so there is
/// nothing to latch.
#[cfg(not(perf_tracking))]
#[inline]
pub const fn init_tracking_enabled() {}

/// Caller-side `ApiTimer` / `CycleSetTimer` / `CycleAddTimer` gate.
///
/// Under `cfg(perf_tracking)`: `Relaxed` `AtomicBool` load + branch
/// (~1 ns per call when perf is off at runtime), avoiding both per-call
/// rdtsc and the `env_logger` filter walk that bare `log_enabled!` would do.
///
/// Under `cfg(not(perf_tracking))`: `const fn` returning `false`, letting
/// LLVM DCE every `if perf_enabled() { … }` branch at build time.
#[cfg(perf_tracking)]
#[inline]
pub fn perf_enabled() -> bool {
    PERF_TRACKING_ENABLED.load(Ordering::Relaxed)
}

#[cfg(not(perf_tracking))]
#[inline]
#[must_use]
pub const fn perf_enabled() -> bool {
    false
}

/// Gate for `bump_pair_stats` and the `mtld3d::d3d9::passes=trace` dump.
///
/// Same shape as [`perf_enabled`]: one cached `Relaxed` load under
/// `cfg(perf_tracking)`, a `const fn` returning `false` otherwise.
#[cfg(perf_tracking)]
#[inline]
pub fn pair_stats_enabled() -> bool {
    PAIR_STATS_ENABLED.load(Ordering::Relaxed)
}

#[cfg(not(perf_tracking))]
#[inline]
#[must_use]
pub const fn pair_stats_enabled() -> bool {
    false
}

/// RAII guard for once-per-frame measurements that **overwrite** a `*mut u64` field.
///
/// Present stall, encoder op cycles, encoder submit cycles, unix-side
/// `drawable_wait`.
///
/// Null-check + perf-enabled gate so callers don't hand-roll
/// `let t0 = rdtsc(); …; *target = rdtsc() - t0;`.
#[cfg(perf_tracking)]
pub struct CycleSetTimer {
    start: u64,
    target: *mut u64,
    enabled: bool,
}

#[cfg(not(perf_tracking))]
pub struct CycleSetTimer;

impl CycleSetTimer {
    #[cfg(perf_tracking)]
    pub fn start(target: *mut u64) -> Self {
        let enabled = !target.is_null() && perf_enabled();
        let start = if enabled { rdtsc() } else { 0 };
        Self {
            start,
            target,
            enabled,
        }
    }

    #[cfg(not(perf_tracking))]
    #[inline]
    #[must_use]
    pub const fn start(_target: *mut u64) -> Self {
        Self
    }
}

#[cfg(perf_tracking)]
impl Drop for CycleSetTimer {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let elapsed = rdtsc() - self.start;
        // SAFETY: `target` was bound to a `&mut u64` field of `DeviceInner`
        // for this timer's lifetime via the `start` constructor; the borrow
        // outlives this Drop.
        unsafe {
            *self.target = elapsed;
        }
    }
}

// Empty Drop under `not(perf_tracking)` so call sites doing
// `drop(timer)` to bracket a sub-scope don't fire `clippy::drop_non_drop`.
// LLVM DCEs the empty drop body after inlining.
#[cfg(not(perf_tracking))]
impl Drop for CycleSetTimer {
    fn drop(&mut self) {}
}

/// RAII guard for sub-scope measurements that accumulate into a `*mut u64`.
///
/// Used by the visibility-query `waitUntilCompleted` block inside an outer
/// `ApiTimer`, plus the Draw-internal snapshot/`push_op` breakdown timers.
/// Same null-check + perf-enabled gate as [`CycleSetTimer`].
#[cfg(perf_tracking)]
pub struct CycleAddTimer {
    start: u64,
    target: *mut u64,
    enabled: bool,
}

#[cfg(not(perf_tracking))]
pub struct CycleAddTimer;

impl CycleAddTimer {
    #[cfg(perf_tracking)]
    pub fn start(target: *mut u64) -> Self {
        let enabled = !target.is_null() && perf_enabled();
        let start = if enabled { rdtsc() } else { 0 };
        Self {
            start,
            target,
            enabled,
        }
    }

    #[cfg(not(perf_tracking))]
    #[inline]
    #[must_use]
    pub const fn start(_target: *mut u64) -> Self {
        Self
    }
}

#[cfg(perf_tracking)]
impl Drop for CycleAddTimer {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        let elapsed = rdtsc() - self.start;
        // SAFETY: caller-provided `target` points into a counter slot that
        // lives for the timer's lifetime (start construction ties `target`
        // to the borrow on the owning struct).
        let current = unsafe { *self.target };
        // SAFETY: same invariant as the load above.
        unsafe { *self.target = current.wrapping_add(elapsed) };
    }
}

// Empty Drop under `not(perf_tracking)` so call sites doing
// `drop(timer)` to bracket a sub-scope don't fire `clippy::drop_non_drop`.
// LLVM DCEs the empty drop body after inlining.
#[cfg(not(perf_tracking))]
impl Drop for CycleAddTimer {
    fn drop(&mut self) {}
}
