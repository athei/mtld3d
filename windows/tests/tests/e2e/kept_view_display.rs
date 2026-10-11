//! A shown window that takes over the metal view of a window destroyed a moment before.
//!
//! An application that destroys its device window and creates the next one
//! at once, as a game does on a video-mode change, has the layer's kept view
//! moved into the new window. The new window has to display what it
//! presents, and its presents have to keep the frame cap, which the display
//! enforces only for drawables it shows. A layer moved while its old window
//! still fades out displays nothing, and its first presents run at the
//! panel's rate whatever the cap until the presenter's hold for a window
//! that displays nothing takes over, so attach takes the fade out of the
//! window's order-out.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, HarnessConfig, set_window_pos};
use mtld3d_types::{D3D_OK, D3DPRESENT_INTERVAL_IMMEDIATE};

use crate::device::{logged_lines, run_in_private_log_child, running_as};

/// The name the workload child runs under.
const CHILD_NAME: &str = "kept-view-display.exe";

/// The child's log filter: unix info records and the per-drawable `presented:` lines.
const LOG_FILTER: &str = "warn,mtld3d::unix=info,mtld3d::unix::present=trace";

/// The line the presented-cadence probe writes for each displayed drawable after the first.
const DISPLAYED: &str = "presented: seq=";

/// Presents before the timed ones, so the window is on screen and the pace has settled.
const WARMUP: u32 = 15;

/// Timed presents per window.
const PRESENTS: u32 = 45;

/// The configuration entry of the frame cap both devices present under.
const FRAME_CAP_ENTRY: &str = "present.maxFps=30";

#[test]
fn a_window_that_takes_a_kept_view_at_once_displays_it_at_the_frame_cap() {
    if running_as(CHILD_NAME) {
        workload();
        return;
    }
    // Which drawables were displayed is read out of the process log, which
    // in the suite's process carries every other test's presents; the
    // workload runs in a process of its own, whose only kept view is its
    // first device's.
    run_in_private_log_child(
        CHILD_NAME,
        "kept_view_display::a_window_that_takes_a_kept_view_at_once_displays_it_at_the_frame_cap",
        LOG_FILTER,
    );
}

/// A device on a shown window of its own, presenting `IMMEDIATE` under the frame cap.
///
/// The window is made topmost: the workload runs in a background process,
/// whose windows open behind the active application's, and a window they
/// cover displays nothing.
///
/// Each process puts its windows in a place of its own, so the windows of two
/// suites running at once do not cover each other; the second window of one
/// process takes the place of the first.
fn shown_window() -> Harness {
    let h = Harness::create(&HarnessConfig {
        visible: true,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: FRAME_CAP_ENTRY,
        ..HarnessConfig::default()
    });
    let slot = i32::try_from(std::process::id() % 32).expect("below 32");
    set_window_pos(
        h.hwnd(),
        64 + (slot % 8) * 96,
        64 + (slot / 8) * 96,
        640,
        480,
    );
    h.make_topmost();
    h
}

/// The mean time between `PRESENTS` presents into `h`'s window, after the warm-up.
fn mean_present_interval(h: &Harness) -> Duration {
    let present = || {
        assert!(h.pump(), "the window's thread got no WM_QUIT");
        assert_eq!(h.clear_target(0xFF20_4060), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    };
    for _ in 0..WARMUP {
        present();
    }
    let start = Instant::now();
    for _ in 0..PRESENTS {
        present();
    }
    start.elapsed() / PRESENTS
}

/// Present into a shown window, destroy it, and present into the window created right after.
///
/// The first window gets a view of its own and is what says whether this
/// session shows windows at all: behind a locked screen, or on a display
/// that shows none, it displays nothing and there is nothing to compare
/// with, so the test notes that and ends. The second window takes the
/// first one's view, which the log names exactly once, and has to display
/// at least half of its presents and keep the frame cap.
fn workload() {
    let first = shown_window();
    mean_present_interval(&first);
    let displayed_first = logged_lines(DISPLAYED).len();
    if displayed_first == 0 {
        eprintln!(
            "a_window_that_takes_a_kept_view_at_once_displays_it_at_the_frame_cap: a shown \
             window displayed none of its presents (a locked screen, or a display that shows no \
             window); skipped"
        );
        return;
    }
    let from = first.hwnd();
    drop(first);

    let second = shown_window();
    let mean = mean_present_interval(&second);
    let moved = logged_lines("is moved into window");
    assert_eq!(moved.len(), 1, "the kept view moved once: {moved:?}");
    assert!(
        moved[0].contains(&format!("kept from window {from:#x},"))
            && moved[0].contains(&format!("moved into window {:#x},", second.hwnd())),
        "the move names the destroyed window and the new one: {}",
        moved[0]
    );
    // The first device's last drawables may report after its release; a
    // pool's worth of them is all that can.
    let displayed_second = logged_lines(DISPLAYED)
        .len()
        .saturating_sub(displayed_first + 3);
    let presents = usize::try_from(WARMUP + PRESENTS).expect("a present count fits usize");
    assert!(
        displayed_second >= presents / 2,
        "the window that took the kept view displayed {displayed_second} of its {presents} presents"
    );
    let frame_cap: u32 = FRAME_CAP_ENTRY
        .rsplit_once('=')
        .and_then(|(_, hz)| hz.parse().ok())
        .expect("the entry names the cap in hertz");
    let cap = Duration::from_secs(1) / frame_cap;
    assert!(
        mean >= cap.mul_f64(0.9) && mean <= cap.mul_f64(1.5),
        "{FRAME_CAP_ENTRY}: presents {mean:?} apart, expected about {cap:?}"
    );
}
