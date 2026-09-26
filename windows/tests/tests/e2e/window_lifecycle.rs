//! Devices and windows created and destroyed on several threads at once.
//!
//! Every test of the suite creates a window, attaches a device to it and
//! tears both down again, on its own thread beside the others. The Mac
//! driver builds each window's Cocoa objects on the process's main thread
//! and releases the window itself on the thread that destroys it, so
//! several threads creating and destroying at once is the shape that meets
//! every ordering rule of that teardown. This test does nothing else, many
//! times over, so that a rule the layer's attach and detach break shows up
//! here rather than as a rare death of a whole-suite run.
//!
//! Every round runs on a thread of its own, as every test of the suite does.
//! One thread can also render through harness after harness: a window the
//! harness destroys leaves no `WM_QUIT` behind for the next one to pump.

use mtld3d_tests::{
    Harness, HarnessConfig, WindowStyle, assert_pixel_eq, post_quit_message, spawn_scoped,
};

/// Lanes creating and destroying at once.
const LANES: usize = 6;
/// Create-and-destroy rounds per lane.
const ROUNDS: usize = 25;

/// Each lane creates a device on a window of its own, presents once and releases both, repeatedly.
///
/// The device release keeps the layer's metal view for the window's next
/// device, so the window's destruction takes the driver's own views and the
/// window apart under a metal view that is still alive; the kept view is
/// then moved into the next window of any lane that attaches with no kept
/// view of its own, or released, on the main thread, when a later device's
/// view displaces it.
#[test]
fn devices_and_windows_come_and_go_on_several_threads_at_once() {
    std::thread::scope(|scope| {
        for _ in 0..LANES {
            spawn_scoped(scope, || {
                for _ in 0..ROUNDS {
                    std::thread::scope(|round| {
                        spawn_scoped(round, || {
                            let h = Harness::create(&HarnessConfig {
                                window_style: WindowStyle::Framed,
                                ..HarnessConfig::default()
                            });
                            h.render_once(0xFF00_FF00, |_| {});
                        });
                    });
                }
            });
        }
    });
}

/// A thread that dropped a harness renders on the next one it creates.
///
/// Dropping the first harness releases its device and destroys its window on
/// this thread, whose window procedure answers `WM_DESTROY` with a quit; the
/// second harness pumps the same queue before its first frame and must find
/// no quit there.
#[test]
fn a_thread_renders_on_a_second_harness_after_dropping_the_first() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;

    let first = Harness::new();
    first.render_once(RED, |_| {});
    assert_pixel_eq(first.read_pixel(1, 1), RED, "first harness");
    drop(first);

    let second = Harness::new();
    second.render_once(BLUE, |_| {});
    assert_pixel_eq(
        second.read_pixel(1, 1),
        BLUE,
        "second harness on the same thread",
    );
}

/// A quit posted before a harness is dropped still ends the next pump on that thread.
///
/// The window destruction takes back only the quit it posted itself, so a
/// quit that was already pending reaches the next harness's pump, which
/// removes it, and the frame after that renders.
#[test]
fn a_quit_pending_before_a_harness_drop_reaches_the_next_pump() {
    const GREEN: u32 = 0xFF00_FF00;

    let first = Harness::new();
    post_quit_message();
    drop(first);

    let second = Harness::new();
    assert!(!second.pump(), "the pending quit reaches the pump");
    second.render_once(GREEN, |_| {});
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second harness after the quit",
    );
}
