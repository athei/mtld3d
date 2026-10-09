//! Unit tests for the clauses of a fullscreen transition's log line.
//!
//! The lines are what a user's log says about an alt-tab or a mode change,
//! so each clause is pinned as rendered: the leave reasons, a rect with its
//! origin (on a monitor left of the primary too), every window placement and
//! the session mode with and without a refresh rate.

use super::{LeaveReason, RectLabel, SessionMode, WindowPlacement};
use crate::display_mode::ModeRequest;

#[test]
fn every_leave_reason_has_its_name() {
    assert_eq!(LeaveReason::WindowedReset.to_string(), "windowed Reset");
    assert_eq!(
        LeaveReason::ProvisionalWindowedReset.to_string(),
        "windowed Reset, provisional until its client area is checked"
    );
    assert_eq!(LeaveReason::Release.to_string(), "device release");
    assert_eq!(LeaveReason::FailedCreate.to_string(), "failed CreateDevice");
}

#[test]
fn a_rect_renders_its_size_and_origin() {
    assert_eq!(
        RectLabel::new(0, 0, 1728, 1117).to_string(),
        "1728x1117 at (0, 0)"
    );
    assert_eq!(
        RectLabel::new(-1728, 40, 0, 1157).to_string(),
        "1728x1117 at (-1728, 40)"
    );
}

#[test]
fn every_window_placement_says_where_the_window_went() {
    assert_eq!(
        WindowPlacement::Placed(RectLabel::new(0, 0, 1280, 720)).to_string(),
        "window 1280x720 at (0, 0)"
    );
    assert_eq!(
        WindowPlacement::NoMonitor.to_string(),
        "window not moved, the monitor rect is unreadable"
    );
    assert_eq!(
        WindowPlacement::AppOwned.to_string(),
        "window left to the app (D3DCREATE_NOWINDOWCHANGES)"
    );
    assert_eq!(
        WindowPlacement::Gone.to_string(),
        "window already destroyed"
    );
    assert_eq!(
        WindowPlacement::Unmoved.to_string(),
        "window not moved, its windowed rect was never read"
    );
}

#[test]
fn the_session_mode_names_the_mode_or_its_absence() {
    let mode = ModeRequest {
        width: 1280,
        height: 720,
        refresh_hz: 60,
    };
    assert_eq!(
        SessionMode::new(Some(mode)).to_string(),
        "session display mode 1280x720@60Hz"
    );
    assert_eq!(
        SessionMode::new(None).to_string(),
        "no session display mode"
    );
}
