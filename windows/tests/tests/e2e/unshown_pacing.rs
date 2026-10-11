//! The pace of a window the compositor does not show.
//!
//! A present carries a minimum duration when the game asks for vsync or a
//! divided interval, or `present.maxFps` caps it, and the display enforces
//! that duration only against a frame it shows. The suite's windows are
//! hidden, which is how a minimised, fully covered or screen-locked window
//! reads too, so the presenter holds their presents to the duration on its
//! own clock and the frame-latency bound paces the game with it. Before it
//! did, such a window presented as fast as the game could, whatever it asked
//! for.
//!
//! A runner can be slower than the pace: its unpaced frames alone may take
//! longer than a vsync interval. So each workload also times the same frames
//! unpaced, on a second device in the same process, before and after the
//! paced ones, and holds the paced frames to the pace or to that time,
//! whichever is longer. Each runs in a process of its own, alone in its log
//! directory, so the lines it reads back (the panel's interval, the drawables
//! acquired) are its own. A window that reads as shown on some machine is
//! paced by the display instead, and the test fails on that premise by name.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, HarnessConfig};
use mtld3d_types::{D3D_OK, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE};

use crate::device::{
    DRAWABLE_LOG_FILTER, await_logged_lines, logged_lines, run_in_private_log_child, running_as,
};

/// Presents before the timed ones, so the pace has settled.
const WARMUP: u32 = 10;

/// Unpaced presents timed before and after the paced ones.
const UNPACED_PRESENTS: u32 = 30;

/// The time between presents into `h`'s window, one entry per present after the warm-up.
fn present_intervals(h: &Harness, presents: u32) -> Vec<Duration> {
    let present = || {
        assert_eq!(h.clear_target(0xFF20_4060), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    };
    for _ in 0..WARMUP {
        present();
    }
    let mut last = Instant::now();
    (0..presents)
        .map(|_| {
            present();
            let now = Instant::now();
            let interval = now - last;
            last = now;
            interval
        })
        .collect()
}

fn mean(intervals: &[Duration]) -> Duration {
    intervals.iter().sum::<Duration>()
        / u32::try_from(intervals.len()).expect("a present count fits u32")
}

/// A hidden window that presents `IMMEDIATE` with no frame cap: nothing paces it.
fn unpaced_device() -> Harness {
    Harness::create(&HarnessConfig {
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: "present.maxFps=0",
        ..HarnessConfig::default()
    })
}

/// Time `paced`'s presents against `pace`, with `unpaced` timed around them as the floor.
///
/// The mean no faster than the pace, less 15 % for timer slack and for the
/// catch-up after a late wake that lands just inside the timed presents,
/// which is what a window nothing holds fails; a runner whose unpaced frames
/// are that slow already cannot tell, and says so instead. The median not slower than
/// the pace or the unpaced frames, whichever is longer, by more than a fifth
/// and a few milliseconds, which is what a hold that adds to the frame time
/// rather than overlapping it fails. The median rather than the mean, because
/// a CI runner launches its test processes in the background, and macOS
/// coalesces such a process's timers: a hold ends up to about a hundred
/// milliseconds late, and the frames ready meanwhile then go at once on the
/// schedule they missed (`taskpolicy -b` shows the same locally). The late
/// wakes and the bursts after them are the tails; the mean comes out slower
/// than the pace there, the median does not.
fn assert_paced(paced: &Harness, unpaced: &Harness, pace: Duration, presents: u32, what: &str) {
    let before = mean(&present_intervals(unpaced, UNPACED_PRESENTS));
    let mut intervals = present_intervals(paced, presents);
    let after = mean(&present_intervals(unpaced, UNPACED_PRESENTS));
    let floor = before.max(after);
    let measured = mean(&intervals);
    intervals.sort_unstable();
    let percentile = |p: usize| intervals[(intervals.len() - 1) * p / 100];
    let drawables = logged_lines(&format!(
        "drawable acquired on window {:#x} for",
        paced.hwnd()
    ))
    .len();
    let report = format!(
        "{what}: presents {measured:?} apart (p10 {:?}, p50 {:?}, p90 {:?}) against a pace of \
         {pace:?}; the same frames unpaced take {before:?} before and {after:?} after; at \
         least {drawables} drawables acquired for the paced window",
        percentile(10),
        percentile(50),
        percentile(90),
    );
    // The premise: the presenter skips this window's drawables, so its pace
    // is the presenter's. A window that reads as shown is paced by the
    // display instead, which this test does not measure.
    assert_eq!(drawables, 0, "{report}: the window reads as shown here");
    if floor < pace.mul_f64(0.85) {
        assert!(
            measured >= pace.mul_f64(0.85),
            "{report}: faster than the pace"
        );
    } else {
        eprintln!("{report}; the unpaced frames are as slow as the pace here, so it is not timed");
    }
    assert!(
        percentile(50) <= pace.max(floor).mul_f64(1.2) + Duration::from_millis(3),
        "{report}: the median is slower than the pace and the unpaced frames allow"
    );
}

/// The name the workload child of `a_frame_cap_paces_a_hidden_window` runs under.
const CAP_CHILD_NAME: &str = "unshown-cap.exe";

/// A frame cap paces a hidden window that presents `IMMEDIATE`.
#[test]
fn a_frame_cap_paces_a_hidden_window() {
    if running_as(CAP_CHILD_NAME) {
        cap_workload();
        return;
    }
    run_in_private_log_child(
        CAP_CHILD_NAME,
        "unshown_pacing::a_frame_cap_paces_a_hidden_window",
        DRAWABLE_LOG_FILTER,
    );
}

fn cap_workload() {
    let unpaced = unpaced_device();
    let paced = Harness::create(&HarnessConfig {
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries: "present.maxFps=30",
        ..HarnessConfig::default()
    });
    assert_paced(
        &paced,
        &unpaced,
        Duration::from_secs(1) / 30,
        45,
        "present.maxFps = 30",
    );
}

/// The name the workload child of `vsync_paces_a_hidden_window_at_the_panel_interval` runs under.
const VSYNC_CHILD_NAME: &str = "unshown-vsync.exe";

/// A vsync request paces a hidden window at the interval of the panel under it.
///
/// The interval is the minimum present duration attach derived from the
/// panel's refresh ceiling, read out of the process log.
#[test]
fn vsync_paces_a_hidden_window_at_the_panel_interval() {
    if running_as(VSYNC_CHILD_NAME) {
        vsync_workload();
        return;
    }
    run_in_private_log_child(
        VSYNC_CHILD_NAME,
        "unshown_pacing::vsync_paces_a_hidden_window_at_the_panel_interval",
        DRAWABLE_LOG_FILTER,
    );
}

fn vsync_workload() {
    let unpaced = unpaced_device();
    let paced = Harness::create(&HarnessConfig {
        presentation_interval: D3DPRESENT_INTERVAL_ONE,
        ..HarnessConfig::default()
    });
    let Some(interval) = panel_interval() else {
        return;
    };
    // About a second of presents, at any refresh rate.
    let presents = u32::try_from(Duration::from_secs(1).as_nanos() / interval.as_nanos())
        .expect("a second holds few intervals")
        .clamp(30, 240);
    assert_paced(&paced, &unpaced, interval, presents, "vsync");
}

/// The minimum present duration attach derived for the paced device.
///
/// The unpaced device has none, so the paced device's is the longest of the
/// two lines. `None`, with a note, when the panel reports no refresh ceiling
/// (a virtual display can): a vsync request then sets no pace, by design, and
/// there is nothing to measure.
fn panel_interval() -> Option<Duration> {
    const NEEDLE: &str = "present: minimum present duration ";
    let lines = await_logged_lines(NEEDLE, 2);
    let seconds = lines
        .iter()
        .map(|line| {
            line.split_once(NEEDLE)
                .and_then(|(_, rest)| rest.split_once('s'))
                .map(|(seconds, _)| seconds.parse::<f64>().expect("the duration is a number"))
                .expect("the line names the duration in seconds")
        })
        .fold(0.0_f64, f64::max);
    if seconds <= 0.0 {
        eprintln!(
            "vsync_paces_a_hidden_window_at_the_panel_interval: the panel reports no refresh \
             ceiling, so a vsync request sets no pace here; skipped ({lines:?})"
        );
        return None;
    }
    Some(Duration::from_secs_f64(seconds))
}
