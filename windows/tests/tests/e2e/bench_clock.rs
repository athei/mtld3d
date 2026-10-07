//! Frame-clock bookkeeping without a device or GPU workload.

use std::time::Duration;

use super::bench::{FRAME_SPIKE_FLOOR, FrameClock, FrameStats, TscClock};

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

#[test]
fn frame_spikes_below_the_floor_are_not_counted() {
    // The v0.12.0 candidate's `shader_stutter_k1` frames: a 31 us median,
    // with its slowest frame at 0.47 ms, over twice the median but under
    // the floor.
    let mut times = vec![Duration::from_micros(31); 180];
    times.extend([Duration::from_micros(62) + Duration::from_nanos(1); 16]);
    times.extend([Duration::from_micros(470); 4]);
    let (spikes, limit) = FrameClock::spikes_of(&times);
    assert_eq!(limit, FRAME_SPIKE_FLOOR);
    assert_eq!(spikes, 0);
}

#[test]
fn frame_spikes_over_the_floor_and_twice_the_median_are_counted() {
    // A short median: the floor is the limit, and a frame over it counts.
    let mut times = vec![Duration::from_micros(31); 198];
    times.extend([
        FRAME_SPIKE_FLOOR,
        FRAME_SPIKE_FLOOR + Duration::from_nanos(1),
    ]);
    assert_eq!(FrameClock::spikes_of(&times), (1, FRAME_SPIKE_FLOOR));

    // A median of a few milliseconds: twice it is the limit, as before the floor.
    let median = Duration::from_micros(11_490);
    let mut times = vec![median; 199];
    times.push(Duration::from_micros(46_435));
    assert_eq!(FrameClock::spikes_of(&times), (1, median * 2));
}

fn assert_stats(actual: &FrameStats, expected: &FrameStats) {
    assert_eq!(actual.frames, expected.frames);
    assert_eq!(actual.mean, expected.mean);
    assert_eq!(actual.p50, expected.p50);
    assert_eq!(actual.p99, expected.p99);
    assert_eq!(actual.max, expected.max);
}
