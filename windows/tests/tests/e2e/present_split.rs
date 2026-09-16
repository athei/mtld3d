//! The submit thread commits the frame; a presenter per device presents it.
//!
//! A present queues a packet the unix-side presenter turns into a drawable
//! acquisition and a present command buffer of its own, after the frame's
//! render command buffer has committed. The render buffer never waits for a
//! drawable, so a read-back that flushes the frame in progress completes
//! while the presenter is parked, and one device's parked presenter leaves
//! another device presenting. `debug.presentGateFile` is the seam: while the
//! file exists the presenter blocks before acquiring a drawable, and a test
//! holds it exactly across the read-back it wants to see complete. With the
//! gate inside the one thunk that replays, acquires and commits, the flush
//! behind the read-back drains a submit thread parked there, the read-back
//! never returns, and the harness reports the test as a timeout rather than
//! as a wrong pixel. The same seam holds a read-back behind a full pipeline:
//! every stage ahead of the presenter filled with a present, which is the
//! most copies one read-back can need and what the slot array is sized for.

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use mtld3d_tests::{Harness, HarnessConfig, assert_pixel_eq, spawn_scoped};
use mtld3d_types::{
    D3D_OK, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DLOCK_READONLY,
};

/// A gate file of this test's own under the prefix's temp directory, and the entry naming it.
///
/// Created here, so the queue that resolves the path at creation finds it.
/// The entry is leaked: `HarnessConfig::config_entries` is `&'static str`,
/// and a test runs once per process. The path carries no `;`, the entry
/// separator.
fn gate_file(tag: &str) -> (PathBuf, &'static str) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock follows Unix epoch")
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("present-gate-{tag}-{}-{stamp}", std::process::id()));
    std::fs::File::create(&path).expect("create the presenter gate file");
    let entry = format!("debug.presentGateFile={}", path.display());
    (path, Box::leak(entry.into_boxed_str()))
}

/// A read-back of the frame in progress completes while the presenter is parked.
///
/// One present while the gate is held: the one copy the read-back's flush
/// takes is of the frame the parked presenter holds.
#[test]
fn a_readback_completes_while_the_presenter_is_parked() {
    const FIRST: u32 = 0xFFFF_0000;
    const SECOND: u32 = 0xFF00_00FF;
    let (gate, entry) = gate_file("readback");
    let h = Harness::with_config(entry);
    assert_eq!(
        h.clear_target(FIRST),
        D3D_OK,
        "clear the frame that is presented"
    );
    // The packet is queued; the presenter reaches the gate and parks on it.
    assert_eq!(h.present(), D3D_OK, "present while the gate is held");
    assert_eq!(
        h.clear_target(SECOND),
        D3D_OK,
        "clear the frame that is not presented"
    );
    {
        let backbuffer = h.back_buffer(0);
        // The read-only lock flushes the frame through the submit thread and
        // reads the back buffer; the presenter has not moved.
        let locked = backbuffer.lock_rect(D3DLOCK_READONLY);
        assert_eq!(
            locked.as_u32(1)[0],
            SECOND,
            "the read-back shows the clear the presenter has not reached"
        );
        assert!(
            gate.exists(),
            "nothing on the read-back path lifted the gate"
        );
    }
    // Lift the gate: the first frame goes to the screen from its copy, and
    // the presents below run through a presenter that is no longer parked.
    std::fs::remove_file(&gate).expect("remove the gate file");
    for _ in 0..2 {
        h.render_once(SECOND, |_| {});
    }
    assert_pixel_eq(h.read_pixel(1, 1), SECOND, "after the gate was lifted");
    // The drop waits for the presenter to go idle and its last present to
    // retire.
}

/// A read-back behind a full pipeline completes while the presenter is parked.
///
/// Five presents fill every stage ahead of the presenter without blocking
/// the API thread: the present the presenter holds at the gate, the submit
/// parked in its wait for it, the payload in the work channel, the frame the
/// encoder holds while it waits for a payload, and the frame in the channel
/// to the encoder. The read-back's flush hurries all of them, and each copies
/// the image of the present ahead of it into a slot before the partial frame
/// copies the last; the slot array is sized for exactly this. One slot short,
/// the last copy waits for the oldest present to commit, which the gate
/// holds, and the harness reports a timeout.
#[test]
fn a_readback_behind_a_full_pipeline_completes_while_the_presenter_is_parked() {
    const STAGES: u32 = 5;
    const LAST: u32 = 0xFFFF_00FF;
    let (gate, entry) = gate_file("pipeline");
    let h = Harness::with_config(entry);
    for stage in 1..=STAGES {
        assert_eq!(
            h.clear_target(0xFF00_0000 | (stage * 0x0020_2020)),
            D3D_OK,
            "clear a frame that is presented"
        );
        assert_eq!(h.present(), D3D_OK, "present into a filling pipeline");
    }
    assert_eq!(
        h.clear_target(LAST),
        D3D_OK,
        "clear the frame that is not presented"
    );
    {
        let backbuffer = h.back_buffer(0);
        let locked = backbuffer.lock_rect(D3DLOCK_READONLY);
        assert_eq!(
            locked.as_u32(1)[0],
            LAST,
            "the read-back shows the clear behind five unpresented frames"
        );
        assert!(
            gate.exists(),
            "nothing on the read-back path lifted the gate"
        );
    }
    std::fs::remove_file(&gate).expect("remove the gate file");
    for _ in 0..2 {
        h.render_once(LAST, |_| {});
    }
    assert_pixel_eq(h.read_pixel(1, 1), LAST, "after the gate was lifted");
}

/// One device's parked presenter leaves another device presenting and reading back.
///
/// The gate is on the first device's queue alone. The second device, created
/// `D3DCREATE_MULTITHREADED`, clears, presents and reads back on a worker
/// while the first sits behind its gate on this thread, which pumps both
/// windows; then the first device's own read-back completes while parked.
#[test]
fn a_parked_presenter_leaves_another_device_presenting() {
    const FRAMES: u32 = 60;
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    let (gate, entry) = gate_file("isolation");
    let gated = Harness::with_config(entry);
    let free = Harness::create(&HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        ..HarnessConfig::default()
    });
    assert_eq!(gated.clear_target(RED), D3D_OK, "clear the gated device");
    assert_eq!(gated.present(), D3D_OK, "park the gated device's presenter");

    let free_shared = free.shared();
    let finished = AtomicU32::new(0);
    let (mismatches, failed) = std::thread::scope(|scope| {
        let worker = spawn_scoped(scope, || {
            let mut mismatches = 0u32;
            let mut failed: Option<(&'static str, i32)> = None;
            for _ in 0..FRAMES {
                let hr = free_shared.clear_target(GREEN);
                if hr < 0 {
                    failed = Some(("Clear", hr));
                    break;
                }
                let hr = free_shared.present();
                if hr < 0 {
                    failed = Some(("Present", hr));
                    break;
                }
                match free_shared.read_pixel(320, 240) {
                    Ok(pixel) if pixel == GREEN => {}
                    Ok(_) => mismatches += 1,
                    Err(call) => {
                        failed = Some(call);
                        break;
                    }
                }
            }
            finished.store(1, Ordering::Release);
            (mismatches, failed)
        });
        while finished.load(Ordering::Acquire) == 0 {
            assert!(gated.pump(), "WM_QUIT on the gated window");
            assert!(free.pump(), "WM_QUIT on the free window");
            std::thread::yield_now();
        }
        worker.join().expect("the worker thread panicked")
    });
    assert_eq!(
        failed, None,
        "a call on the free device failed beside a parked presenter"
    );
    assert_eq!(
        mismatches, 0,
        "the free device read back its own colour every frame"
    );

    // The gated device's own read-back completes while parked: the flush
    // behind `GetRenderTargetData` commits its frame without the presenter.
    assert_eq!(
        gated.clear_target(BLUE),
        D3D_OK,
        "clear the gated device again"
    );
    assert_pixel_eq(
        gated.read_pixel(1, 1),
        BLUE,
        "the gated device reads back while parked",
    );
    assert!(
        gate.exists(),
        "the gate held across both devices' read-backs"
    );
    std::fs::remove_file(&gate).expect("remove the gate file");
    gated.render_once(RED, |_| {});
    assert_pixel_eq(
        gated.read_pixel(1, 1),
        RED,
        "the gated device after the gate was lifted",
    );
    drop(free);
    drop(gated);
}
