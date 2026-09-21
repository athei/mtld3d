//! Unit tests for the present-interval to layer-pacing mapping.
//!
//! Every interval D3D9 names but `IMMEDIATE` asks for vsync, and a bit pattern that names
//! none takes the `Fallthrough` arm, which still runs at display rate but asks the caller
//! to warn. The polarity assertions guard `enabled()`, where flipping a single arm would
//! silently drop vsync for a whole class of intervals. The ceiling tests pin how a divided
//! interval, the mode's refresh rate and `present.maxFps` fold into one number.

use super::{
    DisplaySync, LayerPacing, display_sync_for, effective_max_fps, interval_divisor,
    layer_pacing_for, present_interval as pi, queued_pacing,
};

const fn pacing(display_sync: bool, max_fps: u32) -> LayerPacing {
    LayerPacing {
        display_sync,
        max_fps,
    }
}

#[test]
fn default_and_one_enable_vsync() {
    assert_eq!(display_sync_for(pi::DEFAULT), DisplaySync::On);
    assert_eq!(display_sync_for(pi::ONE), DisplaySync::On);
}

#[test]
fn immediate_disables_vsync() {
    assert_eq!(display_sync_for(pi::IMMEDIATE), DisplaySync::Off);
}

#[test]
fn divided_intervals_enable_vsync_without_a_warning() {
    assert_eq!(display_sync_for(pi::TWO), DisplaySync::On);
    assert_eq!(display_sync_for(pi::THREE), DisplaySync::On);
    assert_eq!(display_sync_for(pi::FOUR), DisplaySync::On);
}

#[test]
fn unknown_bits_fall_through() {
    assert_eq!(display_sync_for(0x1234_5678), DisplaySync::Fallthrough);
    assert!(display_sync_for(0x1234_5678).enabled());
    // Two interval bits at once name no interval.
    assert_eq!(
        display_sync_for(pi::ONE | pi::TWO),
        DisplaySync::Fallthrough
    );
}

#[test]
fn enabled_polarity() {
    assert!(DisplaySync::On.enabled());
    assert!(!DisplaySync::Off.enabled());
    assert!(DisplaySync::Fallthrough.enabled());
}

#[test]
fn the_divisor_is_the_interval_count_not_its_bit_value() {
    assert_eq!(interval_divisor(pi::DEFAULT), 1);
    assert_eq!(interval_divisor(pi::ONE), 1);
    assert_eq!(interval_divisor(pi::TWO), 2);
    assert_eq!(interval_divisor(pi::THREE), 3);
    assert_eq!(interval_divisor(pi::FOUR), 4);
    assert_eq!(interval_divisor(pi::IMMEDIATE), 1);
    assert_eq!(interval_divisor(0x1234_5678), 1);
}

#[test]
fn a_divided_interval_is_a_ceiling_of_the_refresh_rate_over_n() {
    assert_eq!(effective_max_fps(pi::TWO, 60, 0), 30);
    assert_eq!(effective_max_fps(pi::THREE, 60, 0), 20);
    assert_eq!(effective_max_fps(pi::FOUR, 60, 0), 15);
    assert_eq!(effective_max_fps(pi::TWO, 120, 0), 60);
    assert_eq!(effective_max_fps(pi::THREE, 120, 0), 40);
    assert_eq!(effective_max_fps(pi::FOUR, 120, 0), 30);
}

#[test]
fn a_fractional_quotient_rounds_up() {
    // 75 / 2 = 37.5: a 38 Hz ceiling is a minimum duration just under two
    // refresh periods, so the present lands on the second one.
    assert_eq!(effective_max_fps(pi::TWO, 75, 0), 38);
    assert_eq!(effective_max_fps(pi::FOUR, 75, 0), 19);
    assert_eq!(effective_max_fps(pi::THREE, 144, 0), 48);
    assert_eq!(effective_max_fps(pi::FOUR, 1, 0), 1);
}

#[test]
fn the_lower_of_the_interval_and_the_configured_ceiling_wins() {
    assert_eq!(effective_max_fps(pi::TWO, 60, 24), 24);
    assert_eq!(effective_max_fps(pi::TWO, 60, 30), 30);
    assert_eq!(effective_max_fps(pi::TWO, 60, 144), 30);
    assert_eq!(effective_max_fps(pi::FOUR, 120, 60), 30);
}

#[test]
fn an_undivided_interval_leaves_the_configured_ceiling_alone() {
    for interval in [pi::DEFAULT, pi::ONE, pi::IMMEDIATE, 0x1234_5678] {
        assert_eq!(effective_max_fps(interval, 60, 0), 0);
        assert_eq!(effective_max_fps(interval, 60, 45), 45);
        assert_eq!(effective_max_fps(interval, 0, 45), 45);
    }
}

#[test]
fn an_unknown_refresh_rate_divides_nothing() {
    assert_eq!(effective_max_fps(pi::TWO, 0, 0), 0);
    assert_eq!(effective_max_fps(pi::TWO, 0, 50), 50);
}

#[test]
fn layer_pacing_pairs_the_vsync_request_with_the_ceiling() {
    assert_eq!(layer_pacing_for(pi::DEFAULT, 60, 0), pacing(true, 0));
    assert_eq!(layer_pacing_for(pi::ONE, 60, 90), pacing(true, 90));
    assert_eq!(layer_pacing_for(pi::TWO, 60, 0), pacing(true, 30));
    assert_eq!(layer_pacing_for(pi::THREE, 60, 0), pacing(true, 20));
    assert_eq!(layer_pacing_for(pi::FOUR, 60, 10), pacing(true, 10));
    assert_eq!(layer_pacing_for(pi::IMMEDIATE, 60, 0), pacing(false, 0));
    assert_eq!(layer_pacing_for(pi::IMMEDIATE, 60, 72), pacing(false, 72));
}

#[test]
fn a_reset_at_the_pacing_the_layer_holds_queues_nothing() {
    assert_eq!(queued_pacing(pacing(true, 0), pacing(true, 0)), None);
    assert_eq!(queued_pacing(pacing(false, 0), pacing(false, 0)), None);
    assert_eq!(queued_pacing(pacing(true, 30), pacing(true, 30)), None);
}

#[test]
fn a_reset_that_moves_the_pacing_queues_the_new_value() {
    assert_eq!(
        queued_pacing(pacing(true, 0), pacing(false, 0)),
        Some(pacing(false, 0))
    );
    assert_eq!(
        queued_pacing(pacing(false, 0), pacing(true, 0)),
        Some(pacing(true, 0))
    );
}

#[test]
fn a_reset_that_moves_only_the_ceiling_queues_the_new_value() {
    // ONE to TWO keeps the vsync request and moves the ceiling.
    assert_eq!(
        queued_pacing(pacing(true, 0), pacing(true, 30)),
        Some(pacing(true, 30))
    );
    assert_eq!(
        queued_pacing(pacing(true, 30), pacing(true, 0)),
        Some(pacing(true, 0))
    );
}

#[test]
fn the_answer_is_the_whole_queue_not_a_delta() {
    // Asked twice against a layer that has not moved, the second answer
    // empties the queue the first filled rather than leaving it to be
    // written back as the value the layer never left.
    let held = pacing(true, 0);
    assert_eq!(
        queued_pacing(held, pacing(false, 0)),
        Some(pacing(false, 0))
    );
    assert_eq!(queued_pacing(held, pacing(true, 0)), None);
}

#[test]
fn capture_marks_bracket_the_run() {
    use super::capture_marks;
    assert_eq!(capture_marks(1, 3), (true, false));
    assert_eq!(capture_marks(2, 3), (false, false));
    assert_eq!(capture_marks(3, 3), (false, true));
}

#[test]
fn capture_marks_one_frame_run_starts_and_stops() {
    use super::capture_marks;
    assert_eq!(capture_marks(1, 1), (true, true));
}

#[test]
fn capture_marks_outside_the_run_carry_nothing() {
    use super::capture_marks;
    assert_eq!(capture_marks(0, 3), (false, false));
    assert_eq!(capture_marks(4, 3), (false, false));
}

#[test]
fn a_submitted_frame_hands_on_the_stop_and_keeps_the_start() {
    use super::carried_capture_marks;
    assert_eq!(carried_capture_marks((true, false), true), (false, false));
    assert_eq!(carried_capture_marks((false, true), true), (false, true));
    assert_eq!(carried_capture_marks((true, true), true), (false, true));
}

#[test]
fn a_dropped_frame_hands_on_every_mark_it_holds() {
    use super::carried_capture_marks;
    assert_eq!(carried_capture_marks((false, true), false), (false, true));
    assert_eq!(carried_capture_marks((true, true), false), (true, true));
    assert_eq!(carried_capture_marks((true, false), false), (true, false));
}

#[test]
fn an_unmarked_frame_hands_on_nothing() {
    use super::carried_capture_marks;
    assert_eq!(carried_capture_marks((false, false), true), (false, false));
    assert_eq!(carried_capture_marks((false, false), false), (false, false));
}
