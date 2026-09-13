//! Unit tests for the present-interval to display-sync mapping.
//!
//! `D3DPRESENT_INTERVAL_DEFAULT` and `ONE` enable vsync, `IMMEDIATE` disables it, and every
//! other value (the non-1:1 ratios and any unknown bit pattern) takes the `Fallthrough` arm,
//! which still runs at display rate but asks the caller to warn. The polarity assertions
//! guard `enabled()`, where flipping a single arm would silently drop vsync for a whole
//! class of intervals.

use super::{DisplaySync, display_sync_for, present_interval as pi};

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
fn non_unit_ratios_fall_through_to_display_rate() {
    assert_eq!(display_sync_for(pi::TWO), DisplaySync::Fallthrough);
    assert_eq!(display_sync_for(pi::THREE), DisplaySync::Fallthrough);
    assert_eq!(display_sync_for(pi::FOUR), DisplaySync::Fallthrough);
    assert!(display_sync_for(pi::TWO).enabled());
    assert!(display_sync_for(pi::THREE).enabled());
    assert!(display_sync_for(pi::FOUR).enabled());
}

#[test]
fn unknown_bits_fall_through() {
    assert_eq!(display_sync_for(0x1234_5678), DisplaySync::Fallthrough);
    assert!(display_sync_for(0x1234_5678).enabled());
}

#[test]
fn enabled_polarity() {
    assert!(DisplaySync::On.enabled());
    assert!(!DisplaySync::Off.enabled());
    assert!(DisplaySync::Fallthrough.enabled());
}

#[test]
fn a_reset_at_the_pacing_the_layer_holds_queues_nothing() {
    use super::queued_display_sync;
    assert_eq!(queued_display_sync(true, true), None);
    assert_eq!(queued_display_sync(false, false), None);
}

#[test]
fn a_reset_that_moves_the_pacing_queues_the_new_value() {
    use super::queued_display_sync;
    assert_eq!(queued_display_sync(true, false), Some(false));
    assert_eq!(queued_display_sync(false, true), Some(true));
}

#[test]
fn the_answer_is_the_whole_queue_not_a_delta() {
    use super::queued_display_sync;
    // Asked twice against a layer that has not moved, the second answer
    // empties the queue the first filled rather than leaving it to be
    // written back as the value the layer never left.
    let held = true;
    assert_eq!(queued_display_sync(held, false), Some(false));
    assert_eq!(queued_display_sync(held, true), None);
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
