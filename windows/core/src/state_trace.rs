//! Latched gate for the `mtld3d::d3d9::state=trace` diagnostic sub-target.
//!
//! State-trace probes fire at the early-return-on-`Consumed` sites in
//! `warn_rs_non_default_once` / `warn_tss_non_default_once` /
//! `warn_samp_non_default_once`. They enumerate every state write the game
//! actually performs, including ones classified `Consumed` (which the warn
//! helpers themselves silently swallow). Useful for bring-up: "is `WoW`
//! actually setting `D3DRS_BLENDFACTOR` / `D3DRS_DEPTHBIAS` / etc.?".
//!
//! Same shape as `perf::PERF_TRACKING_ENABLED`: latch the `log_enabled!`
//! result once at logger init into a `static AtomicBool`, then the per-call
//! cost is one `Relaxed` load + branch instead of an `env_logger` filter
//! walk on every state write.

use std::sync::atomic::{AtomicBool, Ordering};

use log::{Level, log_enabled};

pub const TARGET: &str = "mtld3d::d3d9::state";

static STATE_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Latch `STATE_TRACE_ENABLED` from `RUST_LOG`.
///
/// Call once per cdylib after `env_logger::try_init`. d3d9.dll calls this
/// alongside `perf::init_tracking_enabled` in `init_logger`.
pub fn init_enabled() {
    let on = log_enabled!(target: TARGET, Level::Trace);
    STATE_TRACE_ENABLED.store(on, Ordering::Relaxed);
}

/// Hot-path query — one `Relaxed` load.
///
/// Call before formatting any state-trace argument so the format work
/// doesn't happen when the target is off.
#[inline]
pub fn enabled() -> bool {
    STATE_TRACE_ENABLED.load(Ordering::Relaxed)
}
