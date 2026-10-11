use super::*;

/// `auto` with no request enforces the D3D9 default of three.
#[test]
fn auto_without_a_request_enforces_the_default() {
    assert_eq!(enforced_frame_latency(None, 0), DEFAULT_FRAME_LATENCY);
    assert_eq!(FrameLatency::new(0).enforced(), 3);
}

/// `auto` enforces the game's request as given, across the whole accepted range.
#[test]
fn auto_enforces_the_games_request() {
    for requested in [1, 2, 3, 4, 10, MAX_FRAME_LATENCY] {
        assert_eq!(enforced_frame_latency(Some(requested), 0), requested);
    }
}

/// A configured number replaces a larger request.
#[test]
fn a_configured_number_overrides_a_larger_request() {
    assert_eq!(enforced_frame_latency(Some(10), 2), 2);
    assert_eq!(enforced_frame_latency(None, 1), 1, "and the default");
}

/// A configured number replaces a smaller request.
#[test]
fn a_configured_number_overrides_a_smaller_request() {
    assert_eq!(enforced_frame_latency(Some(1), 5), 5);
    assert_eq!(enforced_frame_latency(None, 8), 8, "and the default");
}

/// The device reports the game's request, not the configured override.
#[test]
fn the_reported_value_is_the_games() {
    let mut latency = FrameLatency::new(2);
    assert_eq!(latency.reported(), 3, "the default before any request");
    assert!(
        latency.set(10),
        "the first request on a device is announced"
    );
    assert_eq!(latency.reported(), 10);
    assert_eq!(latency.configured(), 2);
    assert_eq!(latency.enforced(), 2);
    assert!(!latency.set(1), "later requests are not");
    assert_eq!(latency.reported(), 1);
}

/// With the default, `Present(k)` waits for frame `k - 3`, and nothing before three presents.
#[test]
fn the_default_waits_for_the_present_three_back() {
    let mut latency = FrameLatency::new(0);
    assert_eq!(latency.record_present(1), None);
    assert_eq!(latency.record_present(2), None);
    assert_eq!(latency.record_present(3), None);
    assert_eq!(latency.record_present(4), Some(1));
    assert_eq!(latency.record_present(5), Some(2));
}

/// The deepest latency reaches back thirty presents.
#[test]
fn the_deepest_latency_waits_thirty_presents_back() {
    let mut latency = FrameLatency::new(0);
    latency.set(MAX_FRAME_LATENCY);
    for seq in 1..=30 {
        assert_eq!(latency.record_present(seq), None, "present {seq}");
    }
    assert_eq!(latency.record_present(31), Some(1));
    assert_eq!(latency.record_present(32), Some(2));
}

/// Mid-frame flushes take sequences but are no frames, so they never become a target.
#[test]
fn flush_sequences_between_presents_are_never_targets() {
    let mut latency = FrameLatency::new(0);
    for (seq, target) in [
        (3, None),
        (7, None),
        (8, None),
        (12, Some(3)),
        (20, Some(7)),
    ] {
        assert_eq!(latency.record_present(seq), target, "present at {seq}");
    }
}

/// Lowering the latency targets a nearer present at once, raising it a farther one.
#[test]
fn a_changed_latency_takes_effect_at_the_next_present() {
    let mut latency = FrameLatency::new(0);
    for seq in 1..=4 {
        latency.record_present(seq);
    }
    latency.set(1);
    assert_eq!(latency.record_present(5), Some(4), "one back");
    latency.set(2);
    assert_eq!(latency.record_present(6), Some(4), "two back");
    latency.set(4);
    assert_eq!(latency.record_present(7), Some(3), "four back");
}

/// A frame finishes only once both its draw work and any display-paced present are done.
#[test]
fn a_frame_finishes_when_both_counters_reach_it() {
    assert!(!frame_finished(5, 4, 9), "the draw buffer still runs");
    assert!(
        !frame_finished(5, 9, 4),
        "the paced present is not on screen yet"
    );
    assert!(frame_finished(5, 5, 5));
    assert!(frame_finished(5, 9, 6));
}
