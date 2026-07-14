//! TSC-based timing helper.
//!
//! Compiles to a single `rdtsc` instruction on both PE targets (i386 and
//! `x86_64`), bypassing any Wine-side `QueryPerformanceCounter` translation.
//! Under Rosetta 2 on Apple Silicon the instruction is trapped to the ARM
//! generic timer; cost stays near-native (~10–30 cycles). Native host
//! (aarch64) test/lint builds — which never ship — read `CNTVCT_EL0`
//! directly instead of going through Rosetta.
//!
//! Returns raw TSC cycles, not nanoseconds. TSC is invariant on Nehalem+
//! so a single runtime calibration against `Instant` yields a
//! process-lifetime Hz value. Under Rosetta the emulated ARM generic
//! timer is likewise fixed-frequency (~945 MHz in practice, NOT CPU
//! clock), which is why the calibration is runtime-determined instead of
//! hard-coded.

use std::{
    sync::LazyLock,
    thread,
    time::{Duration, Instant},
};

use log::info;

/// Calibration message rides on the perf log target.
///
/// TSC is a perf-only primitive across this workspace.
const LOG_TARGET: &str = "mtld3d::perf";

/// Read the timestamp / cycle counter.
///
/// Single-instruction primitive on the x86 PE targets (`rdtsc`); `#[inline]`
/// lets a measurement bracket compile to a pair of reads in release. Native
/// host (aarch64) test/lint builds read the ARM generic timer `CNTVCT_EL0`
/// instead — shipped Wine code always runs an x86 path.
#[inline]
#[must_use]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC is unprivileged on every x86_64 mode we run in
        // (Wine PE process). No preconditions; returns the cycle counter.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(target_arch = "x86")]
    {
        // SAFETY: RDTSC is unprivileged on every x86 mode we run in
        // (Wine PE process). No preconditions; returns the cycle counter.
        unsafe { core::arch::x86::_rdtsc() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Native host (test/lint) builds only — shipped Wine code always runs
        // an x86 path above. CNTVCT_EL0 is the ARM generic-timer counter, the
        // aarch64 analogue of the TSC; `calibrate()` derives its Hz at runtime.
        let cnt: u64;
        // SAFETY: CNTVCT_EL0 is readable from EL0 on macOS arm64 (the kernel
        // enables EL0VCTEN); the read has no preconditions and no side effects.
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt, options(nomem, nostack, preserves_flags));
        }
        cnt
    }
}

/// Return the calibrated TSC frequency in Hz.
///
/// The first call blocks for the 50 ms calibration sleep; subsequent calls
/// are a single atomic load. Intended to be warmed from a background thread
/// at `DllMain` so the encoder-thread hot path never pays the sleep.
///
/// Used by both perf telemetry and the shader-compile-burst debouncer, so
/// stays unconditional even under `cfg(not(perf_tracking))`.
#[must_use]
pub fn tsc_hz() -> u64 {
    *TSC_HZ
}

/// Convert a TSC cycle count to milliseconds (f64, for display).
///
/// Used only by the perf summary renderer, so cfg-gated.
#[cfg(perf_tracking)]
#[inline]
#[must_use]
pub fn cycles_to_ms(cycles: u64) -> f64 {
    u64_to_f64_exact(cycles) * 1e3 / u64_to_f64_exact(tsc_hz())
}

/// Lossless u64 → f64 reconstruction via a hi/lo u32 split, scale, and add.
///
/// Each `f64::from(u32)` is exact (u32 ⊂ f64 mantissa); `2^32` is exactly
/// representable. The scale-and-add composes them without the truncating
/// `u64 as f64` direct cast that clippy rejects. Values exceeding 2^53
/// still incur precision loss in the final add, but no
/// `cast_precision_loss` lint fires because every cast is exact.
///
/// # Panics
///
/// The two `try_from` calls are statically unreachable: `v >> 32` and
/// `v & 0xFFFF_FFFF` both fit u32 by construction.
#[inline]
#[must_use]
pub fn u64_to_f64_exact(v: u64) -> f64 {
    let hi = u32::try_from(v >> 32).expect("u64 >> 32 fits u32");
    let lo = u32::try_from(v & 0xFFFF_FFFF).expect("u64 masked to 32 bits fits u32");
    // The scaled-high term is hoisted into its own binding rather than written
    // as `hi * 2^32 + lo`: the single-expression form tempts clippy's
    // `suboptimal_flops` toward `mul_add`, which is an `fmaf` LIBCALL on the
    // no-FMA i686 baseline. `hi * 2^32` is exact (power-of-two scale), so the
    // separate add's single rounding still matches the fused form bit-for-bit.
    let hi_scaled = f64::from(hi) * 4_294_967_296.0;
    hi_scaled + f64::from(lo)
}

/// Lossless `usize` → f64 via [`u64_to_f64_exact`].
///
/// `usize` is ≤ u64 on every platform we target, so the widening cast is
/// exact.
///
/// # Panics
///
/// Unreachable on supported targets (`x86_64-apple-darwin`,
/// `aarch64-apple-darwin`, `i686-pc-windows-msvc`, `x86_64-pc-windows-msvc`)
/// — usize is at most 64 bits everywhere.
#[inline]
#[must_use]
pub fn usize_to_f64_exact(v: usize) -> f64 {
    u64_to_f64_exact(u64::try_from(v).expect("usize fits u64 on all supported targets"))
}

/// Return the TSC cycle count that corresponds to the given wall-clock duration in seconds.
///
/// Used to build rate-limit thresholds.
#[inline]
#[must_use]
pub fn secs_to_cycles(seconds: u64) -> u64 {
    tsc_hz().saturating_mul(seconds)
}

static TSC_HZ: LazyLock<u64> = LazyLock::new(calibrate);

fn calibrate() -> u64 {
    const SLEEP: Duration = Duration::from_millis(50);
    let t0 = Instant::now();
    let c0 = rdtsc();
    thread::sleep(SLEEP);
    let c1 = rdtsc();
    let elapsed = t0.elapsed();
    let elapsed_us =
        u64::try_from(elapsed.as_micros()).expect("calibration sleep is 50 ms — elapsed fits u64");
    // Integer math: ticks/sec = ticks * 1_000_000 / elapsed_µs. SLEEP ≥1 µs by
    // construction, so the divisor is non-zero.
    let hz = (c1 - c0).saturating_mul(1_000_000) / elapsed_us.max(1);
    let mhz = u64_to_f64_exact(hz) / 1e6;
    info!(
        target: LOG_TARGET,
        "tsc calibrated: {hz} Hz ({mhz:.2} MHz) over {:.1} ms",
        elapsed.as_secs_f64() * 1e3,
    );
    hz
}
