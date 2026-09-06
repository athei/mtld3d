//! Compare snapshot-token binding with the existing owned-byte binding cache.
//!
//! Run in release mode on the native host target. Each row reports the median
//! nanoseconds per binding check from five samples; it excludes API setters,
//! scratch allocation, command emission and GPU submission.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use mtld3d_core::passes::{LastBoundCache, SnapshotBytesCache};

const ITERATIONS: u32 = 2_000_000;

fn sample(mut update: impl FnMut(u32) -> bool) -> Duration {
    let start = Instant::now();
    for index in 0..ITERATIONS {
        black_box(update(black_box(index)));
    }
    start.elapsed()
}

fn compare(size: usize, distinct: bool, changed: bool, reset_interval: u32) {
    let first = vec![1; size];
    let mut second = vec![1; size];
    if changed {
        second[size - 1] = 2;
    }
    let input = |index| {
        black_box(if distinct && index % 2 == 1 {
            second.as_slice()
        } else {
            first.as_slice()
        })
    };
    let mut owned_times = Vec::new();
    let mut snapshot_times = Vec::new();
    for round in 0..5 {
        let mut owned = LastBoundCache::new();
        let mut snapshot = SnapshotBytesCache::new();
        let mut owned_run = || {
            sample(|index| {
                if reset_interval != 0 && index % reset_interval == 0 {
                    owned.vs_draw_changed(&[]);
                }
                // This small-uniform slot uses the same owned-byte helper as the
                // former VS/PS caches and accepts arbitrary snapshot lengths.
                owned.vs_draw_changed(input(index))
            })
        };
        let mut snapshot_run = || {
            sample(|index| {
                if reset_interval != 0 && index % reset_interval == 0 {
                    snapshot.reset();
                }
                snapshot.changed(input(index))
            })
        };
        let (owned_time, snapshot_time) = if round % 2 == 0 {
            (owned_run(), snapshot_run())
        } else {
            let snapshot_time = snapshot_run();
            (owned_run(), snapshot_time)
        };
        owned_times.push(owned_time);
        snapshot_times.push(snapshot_time);
    }
    owned_times.sort_unstable();
    snapshot_times.sort_unstable();
    let ns = |duration: Duration| duration.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    println!(
        "{size:4} distinct={distinct} changed={changed} reset={reset_interval:3} owned={:.2} ns snapshot={:.2} ns",
        ns(owned_times[2]),
        ns(snapshot_times[2])
    );
}

fn main() {
    for size in [16, 512, 4096] {
        compare(size, false, false, 0);
        compare(size, true, false, 0);
        compare(size, true, true, 0);
        compare(size, false, false, 128);
    }
}
