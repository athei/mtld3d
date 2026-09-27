//! Frame-clock bookkeeping without a device or GPU workload.

use std::time::Duration;

use super::bench::{FrameClock, FrameStats, TscClock};

#[test]
fn elapsed_matches_recorded_intervals_and_present_boundaries() {
    let mut clock = FrameClock::start(0);
    assert_eq!(clock.elapsed(), Duration::ZERO);
    // Zero saturates against any starting timestamp and establishes a
    // deterministic baseline without reading another clock.
    clock.record_present(0, 0);
    let mut frames = vec![Duration::ZERO];
    let mut work = vec![Duration::ZERO];
    // Irregular intervals, a repeated timestamp and a backwards timestamp
    // preserve the existing saturating differences and next-frame baseline.
    let timestamps = [
        (400, 1_000),
        (1_100, 4_000),
        (4_000, 4_000),
        (3_000, 3_500),
        (4_500, 6_000),
    ];
    let frame_ticks = [1_000, 3_000, 0, 0, 2_500];
    let work_ticks = [400, 100, 0, 0, 1_000];
    for ((called, now), (frame, api)) in timestamps
        .into_iter()
        .zip(frame_ticks.into_iter().zip(work_ticks))
    {
        clock.record_present(called, now);
        frames.push(TscClock::duration(frame));
        work.push(TscClock::duration(api));
        assert_eq!(clock.frames(), frames.len());
        assert_eq!(clock.elapsed(), frames.iter().sum::<Duration>());
        assert_stats(&clock.stats(), &FrameStats::of(&frames));
        assert_stats(&clock.work_stats(), &FrameStats::of(&work));
    }
}

fn assert_stats(actual: &FrameStats, expected: &FrameStats) {
    assert_eq!(actual.frames, expected.frames);
    assert_eq!(actual.mean, expected.mean);
    assert_eq!(actual.p50, expected.p50);
    assert_eq!(actual.p99, expected.p99);
    assert_eq!(actual.max, expected.max);
}
