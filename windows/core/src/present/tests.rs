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
