//! A shown device window made the child of another shown window by `SetParent`.
//!
//! Wine's Mac driver gives a window that becomes a child no Cocoa window of
//! its own: it destroys the toplevel's Cocoa window and moves the client
//! view, with the device's metal view in it, into the new parent's. The
//! layer has to follow the view there: its presents reach the new window,
//! and once the new window is hidden its presents skip their drawables, as
//! they do for any window that shows nothing. A layer that still watches the
//! destroyed window misses the new window's occlusion and keeps asking a
//! hidden window for drawables.

use std::time::{Duration, Instant};

use mtld3d_tests::{
    GWL_STYLE, Harness, HarnessConfig, WS_CHILD, WS_POPUP, WindowStyle, create_styled_window,
    destroy_window, make_topmost, pump_until_quiet, set_parent, set_window_long, set_window_pos,
    show_window, window_long,
};
use mtld3d_types::{D3D_OK, D3DPRESENT_INTERVAL_IMMEDIATE};

use crate::device::{
    DRAWABLE_LOG_FILTER, logged_lines, per_process_window_origin, run_in_private_log_child,
    running_as,
};

/// The name the workload child runs under.
const CHILD_NAME: &str = "reparent-display.exe";

/// How long the toplevel presents before the move while it acquires no drawable.
///
/// A present into a window not yet on screen skips its drawable and returns
/// at once, so the toplevel is given time rather than a number of presents.
const WARMUP: Duration = Duration::from_secs(3);

/// Drawables the window has to acquire after the move.
///
/// Twice the presents between two of the layer's main-thread
/// reconciliations (`HEADROOM_REFRESH_PRESENTS`), so a reconciliation runs
/// while the new window is shown.
const AFTER_MOVE: usize = 64;

/// How long the window presents after the move to acquire [`AFTER_MOVE`] drawables.
const AFTER_MOVE_BOUND: Duration = Duration::from_secs(10);

/// How long the window presents once its new parent is hidden.
const HIDDEN: Duration = Duration::from_secs(3);

/// Drawables a present into the hidden parent may still acquire.
///
/// The presents queued before the occlusion notification lands.
const HIDDEN_DRAWABLES: usize = 5;

/// Presents the window has to make while its parent is hidden, for the check to mean anything.
const HIDDEN_PRESENTS: u32 = 30;

/// The width and height of both windows.
const SIZE: (i32, i32) = (640, 480);

/// Presents between two reads of the log while presenting towards a drawable count.
const PRESENTS_PER_READ: u32 = 8;

#[test]
fn a_device_window_made_a_child_follows_its_new_window() {
    if running_as(CHILD_NAME) {
        workload();
        return;
    }
    // The drawables are counted in the process log, which in the suite's
    // process carries every other test's presents; the workload runs in a
    // process of its own. Its shown windows take the activation, so the
    // display mode is held while it runs, as `kept_view_display` does.
    let mode = Harness::factory_only();
    mode.hold_display_mode();
    run_in_private_log_child(
        CHILD_NAME,
        "reparent_display::a_device_window_made_a_child_follows_its_new_window",
        DRAWABLE_LOG_FILTER,
    );
}

/// The drawables the presenter acquired for `hwnd` so far.
fn acquired_on(hwnd: usize) -> usize {
    logged_lines(&format!("drawable acquired on window {hwnd:#x} for")).len()
}

/// Present one frame into `h`'s window after pumping its messages.
fn present_one(h: &Harness) {
    assert!(h.pump(), "the window's thread got no WM_QUIT");
    assert_eq!(h.clear_target(0xFF20_4060), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
}

/// Present into `h`'s window until it acquired `target` drawables or `bound` passed; the count.
///
/// The presenter logs a drawable a moment after the present that queued it
/// returned, so the count is read once more after the last present.
fn present_until(h: &Harness, target: usize, bound: Duration) -> usize {
    let deadline = Instant::now() + bound;
    loop {
        for _ in 0..PRESENTS_PER_READ {
            present_one(h);
        }
        let acquired = acquired_on(h.hwnd());
        if acquired >= target {
            return acquired;
        }
        if Instant::now() >= deadline {
            std::thread::sleep(Duration::from_millis(500));
            return acquired_on(h.hwnd());
        }
    }
}

/// Present into a shown toplevel, make it a child of another shown window, then hide that one.
///
/// The toplevel is what says whether this session shows windows at all:
/// behind a locked screen every present into it is skipped as occluded, and
/// there is nothing to compare with, so the test notes that and ends. After
/// the move the window has to acquire [`AFTER_MOVE`] more drawables, and once
/// the parent is hidden at most [`HIDDEN_DRAWABLES`].
fn workload() {
    let (x, y) = per_process_window_origin();
    let h = Harness::create(&HarnessConfig {
        visible: true,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        ..HarnessConfig::default()
    });
    set_window_pos(h.hwnd(), x, y, SIZE.0, SIZE.1);
    h.make_topmost();
    let before = present_until(&h, 1, WARMUP);
    if before == 0 {
        eprintln!(
            "a_device_window_made_a_child_follows_its_new_window: a shown window acquired no \
             drawable (a locked screen, or a display that shows no window); skipped"
        );
        return;
    }

    let parent = create_styled_window(SIZE.0, SIZE.1, true, &WindowStyle::Borderless);
    set_window_pos(parent, x, y, SIZE.0, SIZE.1);
    make_topmost(parent);
    let style = window_long(h.hwnd(), GWL_STYLE);
    set_window_long(h.hwnd(), GWL_STYLE, (style & !WS_POPUP) | WS_CHILD);
    set_parent(h.hwnd(), parent);
    set_window_pos(h.hwnd(), 0, 0, SIZE.0, SIZE.1);
    assert!(
        pump_until_quiet(Duration::from_millis(200), Duration::from_secs(2)),
        "the window's thread got no WM_QUIT"
    );
    let moved = acquired_on(h.hwnd());
    let after = present_until(&h, moved + AFTER_MOVE, AFTER_MOVE_BOUND);
    eprintln!(
        "drawables: {before} before the move, {} after it",
        after - moved
    );
    assert!(
        after >= moved + AFTER_MOVE,
        "the window made a child acquired {} drawables in {AFTER_MOVE_BOUND:?} of presenting \
         after the move, expected {AFTER_MOVE}",
        after - moved
    );

    show_window(parent, false);
    assert!(
        pump_until_quiet(Duration::from_millis(200), Duration::from_secs(2)),
        "the window's thread got no WM_QUIT"
    );
    let shown = acquired_on(h.hwnd());
    let start = Instant::now();
    let mut presents = 0_u32;
    while start.elapsed() < HIDDEN {
        present_one(&h);
        presents += 1;
    }
    std::thread::sleep(Duration::from_millis(500));
    let while_hidden = acquired_on(h.hwnd()) - shown;
    eprintln!("hidden parent: {presents} presents, {while_hidden} drawables acquired");
    assert!(
        presents >= HIDDEN_PRESENTS,
        "{presents} presents in {HIDDEN:?} into the hidden parent, expected {HIDDEN_PRESENTS}"
    );
    assert!(
        while_hidden <= HIDDEN_DRAWABLES,
        "{while_hidden} of {presents} presents into the hidden parent acquired a drawable, \
         expected at most {HIDDEN_DRAWABLES}"
    );

    // The child goes with its device first; destroying the parent first
    // would destroy the child under the harness.
    drop(h);
    destroy_window(parent);
}
