//! Several devices alive at once, each on its own window.
//!
//! The unix side keeps one attachment record per device, keyed by the metal
//! view attach created, and every present and every teardown addresses its
//! own. These tests drive two and three devices side by side through the
//! paths that look a record up: interleaved presents past the interval at
//! which the presenting thread asks the main thread for a display
//! reconciliation, a teardown beside a device that keeps presenting, and an
//! attach beside a device that is already live. The last two tests move the
//! devices onto two threads: one where each device's wait for its own frame
//! meets the other's submissions in the unix-side registry of in-flight
//! command buffers, and one where two devices under `render.scale` read back
//! at once, so each device's readback resolve has to run in a scratch texture
//! of its own.

use std::sync::atomic::{AtomicU32, Ordering};

use mtld3d_tests::{Harness, HarnessConfig, SharedDevice, SharedQuery, assert_pixel_eq};
use mtld3d_types::{
    D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DGETDATA_FLUSH, D3DISSUE_BEGIN,
    D3DISSUE_END, D3DQUERYTYPE_OCCLUSION,
};

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

/// Drive one device through `FRAMES` frames, each flushed through its occlusion query.
///
/// `GetData(D3DGETDATA_FLUSH)` parks on the unix-side wait for the frame the
/// `Present` before it submitted, so every frame of this loop is one wait
/// against the registry of in-flight command buffers.
fn present_and_flush(
    device: &SharedDevice<'_>,
    query: &SharedQuery<'_>,
    frames: u32,
) -> Vec<(&'static str, i32)> {
    let mut results = Vec::new();
    for _ in 0..frames {
        results.push(("Issue(BEGIN)", query.issue(D3DISSUE_BEGIN)));
        results.push(("Issue(END)", query.issue(D3DISSUE_END)));
        results.push(("Present", device.present()));
        results.push(("GetData(FLUSH)", query.data_u32(D3DGETDATA_FLUSH).0));
    }
    results
}

/// Two devices on two threads each wait for their own frames.
///
/// Each worker drives one `D3DCREATE_MULTITHREADED` device: it brackets an
/// empty frame with an occlusion query, presents, and flushes the query,
/// which waits for the frame it just submitted. The two devices mint the
/// same sequence numbers, so the unix-side registry of in-flight command
/// buffers has to tell them apart: a wait that lands on the other device's
/// buffer ends the process under the Metal validation layer when that buffer
/// is not yet committed, and without the layer a wait whose entry the other
/// device's completion removed retires a frame early. Both windows are
/// created, pumped and destroyed on this thread, so the workers never touch
/// a window. The collision is timing-dependent, so one run guards the lookup
/// rather than proving it.
#[test]
fn two_devices_on_two_threads_wait_for_their_own_frames() {
    const FRAMES: u32 = 300;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let config = HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        ..HarnessConfig::default()
    };
    let first = Harness::create(&config);
    let second = Harness::create(&config);
    let first_query = first
        .create_query(D3DQUERYTYPE_OCCLUSION)
        .expect("OCCLUSION query is supported");
    let second_query = second
        .create_query(D3DQUERYTYPE_OCCLUSION)
        .expect("OCCLUSION query is supported");
    let first_shared = first.shared();
    let second_shared = second.shared();
    let first_shared_query = first_shared.share_query(&first_query);
    let second_shared_query = second_shared.share_query(&second_query);
    let finished = AtomicU32::new(0);

    std::thread::scope(|scope| {
        let first_worker = scope.spawn(|| {
            let results = present_and_flush(&first_shared, &first_shared_query, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            results
        });
        let second_worker = scope.spawn(|| {
            let results = present_and_flush(&second_shared, &second_shared_query, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            results
        });
        while finished.load(Ordering::Acquire) < 2 {
            assert!(first.pump(), "WM_QUIT on the first window");
            assert!(second.pump(), "WM_QUIT on the second window");
            std::thread::yield_now();
        }
        for worker in [first_worker, second_worker] {
            let results = worker.join().expect("a worker thread panicked");
            // `GetData` may answer `S_FALSE` (1) for a query the GPU has not
            // retired; every other call answers `D3D_OK`, and no call fails.
            for (call, hr) in &results {
                assert!(*hr >= 0, "{call} on a worker thread failed: 0x{hr:08X}");
            }
        }
    });

    first.render_once(RED, |_| {});
    second.render_once(GREEN, |_| {});
    assert_pixel_eq(
        first.read_pixel(1, 1),
        RED,
        "first device after its thread stopped",
    );
    assert_pixel_eq(
        second.read_pixel(1, 1),
        GREEN,
        "second device after its thread stopped",
    );
}

/// What one worker's readbacks came to.
struct ReadbackOutcome {
    /// Readbacks that returned a colour other than the one cleared to.
    mismatches: u32,
    /// The first mismatch, as `(frame, pixel)`.
    first_wrong: Option<(u32, u32)>,
    /// The first failing call, by name, with its hr.
    failed: Option<(&'static str, i32)>,
}

/// Clear, present and read back the centre pixel `frames` times.
///
/// Stops at the first failing call; a wrong colour is counted and the loop
/// goes on, so the count says how often the collision landed.
fn readback_own_colour(device: &SharedDevice<'_>, colour: u32, frames: u32) -> ReadbackOutcome {
    let mut outcome = ReadbackOutcome {
        mismatches: 0,
        first_wrong: None,
        failed: None,
    };
    for frame in 0..frames {
        let hr = device.clear_target(colour);
        if hr < 0 {
            outcome.failed = Some(("Clear", hr));
            return outcome;
        }
        let hr = device.present();
        if hr < 0 {
            outcome.failed = Some(("Present", hr));
            return outcome;
        }
        match device.read_pixel(320, 240) {
            Ok(pixel) if pixel == colour => {}
            Ok(pixel) => {
                outcome.mismatches += 1;
                outcome.first_wrong.get_or_insert((frame, pixel));
            }
            Err(failed) => {
                outcome.failed = Some(failed);
                return outcome;
            }
        }
    }
    outcome
}

/// Two scaled devices on two threads each read back their own pixels.
///
/// Under `render.scale` a readback resolves the render-resolution back buffer
/// up to its reported size in a scratch texture, then blits the scratch into
/// the caller's memory, both on the device's own queue. Two devices at one
/// size and format asked for the scratch at once, and Metal orders command
/// buffers within a queue only, so the other device's resolve could land
/// between this device's resolve and its blit: each read the other's frame.
/// The scratch is keyed by the queue now, and this drives two devices of the
/// same geometry through the readback at once. The collision is
/// timing-dependent, so one run guards the keying rather than proving it.
#[test]
fn two_scaled_devices_on_two_threads_read_back_their_own_pixels() {
    const FRAMES: u32 = 300;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;

    let config = HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        config_entries: "render.scale=0.5",
        ..HarnessConfig::default()
    };
    let first = Harness::create(&config);
    let second = Harness::create(&config);
    let first_shared = first.shared();
    let second_shared = second.shared();
    let finished = AtomicU32::new(0);

    std::thread::scope(|scope| {
        let first_worker = scope.spawn(|| {
            let results = readback_own_colour(&first_shared, RED, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            results
        });
        let second_worker = scope.spawn(|| {
            let results = readback_own_colour(&second_shared, GREEN, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            results
        });
        while finished.load(Ordering::Acquire) < 2 {
            assert!(first.pump(), "WM_QUIT on the first window");
            assert!(second.pump(), "WM_QUIT on the second window");
            std::thread::yield_now();
        }
        for (name, worker) in [("first", first_worker), ("second", second_worker)] {
            let outcome = worker.join().expect("a worker thread panicked");
            assert!(
                outcome.failed.is_none(),
                "{name} device: {} failed on its worker thread: 0x{:08X}",
                outcome.failed.map_or("", |(call, _)| call),
                outcome.failed.map_or(0, |(_, hr)| hr)
            );
            assert_eq!(
                outcome.mismatches,
                0,
                "{name} device read another device's pixels in {} of {FRAMES} readbacks, \
                 first at frame {} reading 0x{:08X}",
                outcome.mismatches,
                outcome.first_wrong.map_or(0, |(frame, _)| frame),
                outcome.first_wrong.map_or(0, |(_, pixel)| pixel)
            );
        }
    });
}
