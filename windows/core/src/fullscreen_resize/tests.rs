//! Unit tests for the external-resize policy of a fullscreen window.
//!
//! `ExternalResizeGuard` is pure state, so its ping-pong rules are pinned here
//! against a fixed monitor rect: a covered window is left alone, each distinct
//! clamp size earns exactly one re-assert and its repeat is suppressed, and
//! covering the monitor or calling `reset` refills the budget. The budget caps
//! the alternating-clamp pathology instead of re-asserting every frame.
//! Covering allows for the window manager's rounding, so the rules are pinned
//! against a rect a pixel larger than the monitor as well as an exact one.

use super::{
    COVER_SLACK, ExternalResizeAction, ExternalResizeGuard, REASSERT_BUDGET, covers_monitor,
};

const MONITOR: (u32, u32) = (3456, 2234);

#[test]
fn a_covered_window_is_left_alone() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
}

#[test]
fn a_window_manager_rounding_up_still_counts_as_covered() {
    let mut g = ExternalResizeGuard::new();
    // The rect a window manager answers with when a monitor dimension does
    // not land on its coordinate grid. It covers the display, so re-asserting
    // the monitor rect would only repeat that round trip.
    let rounded = (MONITOR.0, MONITOR.1 + 1);
    assert_eq!(g.decide(rounded, MONITOR), ExternalResizeAction::Covered);
    assert_eq!(g.decide(rounded, MONITOR), ExternalResizeAction::Covered);
}

#[test]
fn covering_the_monitor_takes_a_size_from_the_monitor_up_to_the_slack() {
    let (width, height) = MONITOR;
    let slack = COVER_SLACK;
    assert!(covers_monitor(MONITOR, MONITOR));
    assert!(covers_monitor((width + slack, height + slack), MONITOR));
    // One pixel short on either axis leaves the desktop showing.
    assert!(!covers_monitor((width - 1, height), MONITOR));
    assert!(!covers_monitor((width, height - 1), MONITOR));
    // Beyond the slack the size is the game's own, not the window manager's.
    assert!(!covers_monitor((width + slack + 1, height), MONITOR));
    assert!(!covers_monitor((width, height + slack + 1), MONITOR));
}

#[test]
fn a_window_larger_than_the_slack_is_reasserted() {
    let mut g = ExternalResizeGuard::new();
    // A mode's outer rect that overshoots the monitor is a size the game
    // chose, so the monitor rect is re-applied over it.
    let oversize = (MONITOR.0 + 64, MONITOR.1 + 64);
    assert_eq!(g.decide(oversize, MONITOR), ExternalResizeAction::Reassert);
}

#[test]
fn a_shrunk_window_is_reasserted_once_per_size() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
    // The same clamp coming back means the re-assert did not stick.
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Suppressed
    );
    // A different size is a new event and earns its own re-assert.
    assert_eq!(
        g.decide((1608, 934), MONITOR),
        ExternalResizeAction::Reassert
    );
}

#[test]
fn covering_the_monitor_clears_the_guard_and_refills_the_budget() {
    let mut g = ExternalResizeGuard::new();
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
    assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
    // The previously re-asserted size counts as a fresh event again.
    assert_eq!(
        g.decide((1288, 834), MONITOR),
        ExternalResizeAction::Reassert
    );
}

#[test]
fn alternating_clamp_sizes_exhaust_the_budget() {
    let mut g = ExternalResizeGuard::new();
    for i in 0..REASSERT_BUDGET {
        let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
        assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
    }
    assert_eq!(
        g.decide((100, 100), MONITOR),
        ExternalResizeAction::Suppressed
    );
    assert_eq!(
        g.decide((300, 300), MONITOR),
        ExternalResizeAction::Suppressed
    );
}

#[test]
fn reset_refills_the_budget() {
    let mut g = ExternalResizeGuard::new();
    for i in 0..REASSERT_BUDGET {
        let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
        assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
    }
    g.reset();
    assert_eq!(
        g.decide((100, 100), MONITOR),
        ExternalResizeAction::Reassert
    );
}
