//! Unit tests for the shader-compile counters and their burst debounce.
//!
//! `CompileBucket::from_sm_major` is pinned on both sides: majors 1/2/3 map to `Sm1`/`Sm2`/`Sm3`,
//! anything else yields `None`. `format_summary` is checked against exact strings so the
//! fixed-column layout stays stable across emits: all four buckets print, zeros included, and the
//! verb substitutes in place. Two tests round-trip `record` and `poll_drain`, one of them over two
//! `CompileStats` at once, and five walk `BurstTracker` through the zero state, arm, growth reset,
//! fire and disarm. No test needs a lock: the counters are owned, not shared.

use super::*;

/// Enough idle cycles for `poll_drain` to answer, given the fixed readings below.
const IDLE: u64 = 1000;

#[test]
fn from_sm_major_maps_valid_versions() {
    assert_eq!(CompileBucket::from_sm_major(1), Some(CompileBucket::Sm1));
    assert_eq!(CompileBucket::from_sm_major(2), Some(CompileBucket::Sm2));
    assert_eq!(CompileBucket::from_sm_major(3), Some(CompileBucket::Sm3));
}

#[test]
fn from_sm_major_rejects_out_of_range() {
    assert_eq!(CompileBucket::from_sm_major(0), None);
    assert_eq!(CompileBucket::from_sm_major(4), None);
    assert_eq!(CompileBucket::from_sm_major(255), None);
}

#[test]
fn format_multi_bucket_orders_ff_sm1_sm2_sm3() {
    let snap = Snapshot {
        counts: [12, 0, 30, 0],
        duration_ns: [200_000_000, 0, 1_034_000_000, 0],
    };
    assert_eq!(
        format_summary(&snap, "compiled", 156),
        "shaders:   42 compiled in 1234ms (FF:  12, SM1:   0, SM2:  30, SM3:   0,   156 total)"
    );
}

#[test]
fn format_single_bucket_still_names_it() {
    let snap = Snapshot {
        counts: [0, 0, 4, 0],
        duration_ns: [0, 0, 88_000_000, 0],
    };
    assert_eq!(
        format_summary(&snap, "compiled", 157),
        "shaders:    4 compiled in   88ms (FF:   0, SM1:   0, SM2:   4, SM3:   0,   157 total)"
    );
}

#[test]
fn format_all_buckets_present() {
    let snap = Snapshot {
        counts: [1, 2, 3, 4],
        duration_ns: [1_000_000, 2_000_000, 3_000_000, 4_000_000],
    };
    assert_eq!(
        format_summary(&snap, "compiled", 10),
        "shaders:   10 compiled in   10ms (FF:   1, SM1:   2, SM2:   3, SM3:   4,    10 total)"
    );
}

#[test]
fn format_pre_warmed_verb_substitutes_in_place() {
    let snap = Snapshot {
        counts: [12, 0, 30, 0],
        duration_ns: [200_000_000, 0, 1_034_000_000, 0],
    };
    assert_eq!(
        format_summary(&snap, "pre-warmed", 42),
        "shaders:   42 pre-warmed in 1234ms (FF:  12, SM1:   0, SM2:  30, SM3:   0,    42 total)"
    );
}

#[test]
fn record_drain_round_trip() {
    let mut stats = CompileStats::new();

    stats.record(CompileBucket::Sm2, Duration::from_millis(10));
    stats.record(CompileBucket::Sm2, Duration::from_millis(15));
    stats.record(CompileBucket::Ff, Duration::from_millis(2));

    // Still growing as far as the debounce is concerned: no answer yet.
    assert!(stats.poll_drain(100, IDLE).is_none());

    let snap = stats
        .poll_drain(100 + IDLE + 1, IDLE)
        .expect("burst idle long enough to emit");
    assert_eq!(snap.counts, [1, 0, 2, 0]);
    assert_eq!(snap.duration_ns, [2_000_000, 0, 25_000_000, 0]);

    // Drained: nothing left to report.
    assert!(stats.poll_drain(100 + 2 * IDLE + 2, IDLE).is_none());
}

#[test]
fn two_encoders_do_not_see_each_others_compiles() {
    let mut first = CompileStats::new();
    let mut second = CompileStats::new();

    first.record(CompileBucket::Sm2, Duration::from_millis(10));
    second.record(CompileBucket::Ff, Duration::from_millis(4));
    second.record(CompileBucket::Sm3, Duration::from_millis(6));

    let first_snap = {
        assert!(first.poll_drain(100, IDLE).is_none());
        first
            .poll_drain(100 + IDLE + 1, IDLE)
            .expect("first encoder's burst went idle")
    };
    assert_eq!(first_snap.counts, [0, 0, 1, 0], "only its own compile");
    assert_eq!(first_snap.duration_ns, [0, 0, 10_000_000, 0]);

    // Draining the first leaves the second's counts untouched.
    let second_snap = {
        assert!(second.poll_drain(100, IDLE).is_none());
        second
            .poll_drain(100 + IDLE + 1, IDLE)
            .expect("second encoder's burst went idle")
    };
    assert_eq!(second_snap.counts, [1, 0, 0, 1]);
    assert_eq!(second_snap.duration_ns, [4_000_000, 0, 0, 6_000_000]);
}

#[test]
fn burst_tracker_does_not_fire_while_zero() {
    let mut t = BurstTracker::new();
    assert!(!t.poll([0, 0, 0, 0], 0, 1000));
    assert!(!t.poll([0, 0, 0, 0], 5000, 1000));
}

#[test]
fn burst_tracker_arms_on_first_nonzero_then_holds_off_briefly() {
    let mut t = BurstTracker::new();
    // First nonzero observation arms the timer.
    assert!(!t.poll([1, 0, 0, 0], 100, 1000));
    // Same counts, only 500 cycles later — still within idle window.
    assert!(!t.poll([1, 0, 0, 0], 600, 1000));
}

#[test]
fn burst_tracker_fires_after_idle_threshold() {
    let mut t = BurstTracker::new();
    assert!(!t.poll([1, 0, 0, 0], 100, 1000));
    assert!(t.poll([1, 0, 0, 0], 1101, 1000));
}

#[test]
fn burst_tracker_growing_counts_reset_idle_window() {
    let mut t = BurstTracker::new();
    assert!(!t.poll([1, 0, 0, 0], 100, 1000));
    // Counts grew → reset.
    assert!(!t.poll([2, 0, 0, 0], 600, 1000));
    // 500 cycles after reset — still inside.
    assert!(!t.poll([2, 0, 0, 0], 1100, 1000));
    // Now past the idle threshold.
    assert!(t.poll([2, 0, 0, 0], 1700, 1000));
}

#[test]
fn burst_tracker_disarms_after_emit() {
    let mut t = BurstTracker::new();
    assert!(!t.poll([1, 0, 0, 0], 100, 1000));
    assert!(t.poll([1, 0, 0, 0], 1500, 1000));
    // After the caller drains, the next observation will be [0;4].
    assert!(!t.poll([0, 0, 0, 0], 2000, 1000));
    // Subsequent burst arms freshly.
    assert!(!t.poll([1, 0, 0, 0], 3000, 1000));
    assert!(t.poll([1, 0, 0, 0], 4500, 1000));
}

/// The asynchronous tail prints only when there is something to say, and draining zeroes it.
#[test]
fn async_counts_print_once_and_drain_to_zero() {
    let mut stats = CompileStats::new();
    assert_eq!(format_async_suffix(&stats.take_async()), "");
    stats.record_async_compile();
    stats.record_async_compile();
    stats.record_skipped_draw();
    assert_eq!(
        format_async_suffix(&stats.take_async()),
        ", 2 compiled async, 1 draws skipped"
    );
    assert_eq!(format_async_suffix(&stats.take_async()), "");
    stats.record_skipped_draw();
    assert_eq!(
        format_async_suffix(&stats.take_async()),
        ", 0 compiled async, 1 draws skipped",
        "a skip alone is never silent"
    );
}
