//! Several devices alive at once, each on its own window.
//!
//! The unix side keeps one attachment record per device, keyed by the metal
//! view attach created, and every present and every teardown addresses its
//! own. These tests drive two and three devices side by side through the
//! paths that look a record up: interleaved presents past the interval at
//! which the presenting thread asks the main thread for a display
//! reconciliation, a teardown beside a device that keeps presenting, and an
//! attach beside a device that is already live.

use mtld3d_tests::{Harness, assert_pixel_eq};

/// Two devices attached at once present independently, and a teardown leaves the other alone.
///
/// Both devices present past the interval at which the presenting thread
/// queues a headroom refresh on the main thread, so that walk runs on each
/// device's own view while the other is live; each then reads back its own
/// colour. Releasing the first device retires its record while the second
/// keeps presenting, a third device attaches beside the live second, and the
/// two go away in the reverse order.
///
/// What this cannot detect: with a process-wide record in place of the
/// per-device one these devices still render their own frames, because the
/// harness windows are hidden, the headroom is 1.0 everywhere and a present
/// routes by its own layer. What it guards is the per-device lookup on every
/// present and detach path: a present that dereferences a retired record, a
/// detach that retires the wrong one, or a lock order that wedges between the
/// submit thread and the main thread's reconciliation ends this test rather
/// than passing it.
#[test]
fn two_live_devices_present_independently() {
    const PRESENTS: u32 = 40;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;

    let first = Harness::new();
    let second = Harness::new();
    for _ in 0..PRESENTS {
        first.render_once(RED, |_| {});
        second.render_once(GREEN, |_| {});
    }
    assert_pixel_eq(
        first.read_pixel(1, 1),
        RED,
        "first device beside the second",
    );
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second device beside the first",
    );

    // The window outlives the device it served: destroying it here would post
    // WM_QUIT into the thread queue the other devices then pump.
    assert_eq!(
        first.release_device(),
        0,
        "the first device is fully released"
    );
    for _ in 0..PRESENTS {
        second.render_once(GREEN, |_| {});
    }
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second device after the first was released",
    );

    let third = Harness::new();
    for _ in 0..PRESENTS {
        second.render_once(GREEN, |_| {});
        third.render_once(BLUE, |_| {});
    }
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second device beside the third",
    );
    assert_pixel_eq(
        third.read_pixel(1, 1),
        BLUE,
        "third device attached beside the live second",
    );
    drop(third);
    drop(second);
}
