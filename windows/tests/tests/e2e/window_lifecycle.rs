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
//! Every round runs on a thread of its own, as every test of the suite does:
//! the harness window's `WM_DESTROY` posts `WM_QUIT` to its thread's queue,
//! so a thread that destroyed one window cannot render on the next.

use mtld3d_tests::{Harness, HarnessConfig, WindowStyle};

/// Lanes creating and destroying at once.
const LANES: usize = 6;
/// Create-and-destroy rounds per lane.
const ROUNDS: usize = 25;

/// Each lane creates a device on a window of its own, presents once and releases both, repeatedly.
///
/// The device release retires the layer's metal view on the main thread
/// before the window goes, and the window's destruction then takes the
/// driver's own views and the window apart.
#[test]
fn devices_and_windows_come_and_go_on_several_threads_at_once() {
    std::thread::scope(|scope| {
        for _ in 0..LANES {
            scope.spawn(|| {
                for _ in 0..ROUNDS {
                    std::thread::scope(|round| {
                        round.spawn(|| {
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
