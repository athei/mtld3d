//! The frame-latency bound: `Present` returns once the frame its latency counts back to finished.
//!
//! A device keeps at most its maximum frame latency of presented frames
//! unfinished: 3 by default, what an extended device's
//! `SetMaximumFrameLatency` asks for, and `present.maxFrameLatency` in place
//! of either when it names a number. The observation is an EVENT query
//! issued at the end of a frame that keeps the GPU busy: its first poll
//! without `D3DGETDATA_FLUSH` reads the retirement counter and submits
//! nothing, so once the bound has had `Present` wait for that frame, the
//! poll reports it complete. Before the bound existed, nothing waited on
//! the GPU and the poll could still read `S_FALSE` there. The latency values
//! are pinned with `debug.presentGateFile`: a paced first present parked at
//! the gate keeps the first frame unfinished, so whether the present the
//! latency names returns while the gate stands says what the device
//! enforces, whatever the GPU's speed.

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use mtld3d_tests::{Factory, Harness, HarnessConfig, PosColorVertex, Query, spawn_scoped};
use mtld3d_types::{
    D3D_OK, D3DBLEND_ONE, D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED,
    D3DCULL_NONE, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DISSUE_END, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPT_TRIANGLELIST, D3DQUERYTYPE_EVENT, D3DRS_ALPHABLENDENABLE, D3DRS_CULLMODE,
    D3DRS_DESTBLEND, D3DRS_LIGHTING, D3DRS_SRCBLEND, D3DRS_ZENABLE,
};

/// The query answers from the GPU, never from the immediate stub.
const EVENT_FROM_GPU: &str = "query.eventImmediate=false";

/// The same, on a device that presents into its occluded window, paced by a frame cap.
///
/// Every present then waits for the screen, so a frame finishes only once
/// its present was shown: the presented handlers have to run for the bound
/// to let `Present` go on.
const PACED: &str = "query.eventImmediate=false;debug.presentOccluded=true;present.maxFps=60";

/// How long a present held by the bound is watched before the gate lifts.
///
/// Below the one-second cap on the paced half of the wait, so the cap
/// cannot be what holds it.
const HELD_PRESENT_WATCH: Duration = Duration::from_millis(500);

/// Full-frame quads the busy frame draws, blended so none of them can be skipped.
const BUSY_QUADS: usize = 600;
/// Quads per `DrawPrimitiveUP` call of the busy frame.
const QUADS_PER_DRAW: usize = 100;

fn device(factory: Factory, config_entries: &'static str) -> Harness {
    Harness::create(&HarnessConfig {
        width: 1920,
        height: 1080,
        presentation_interval: D3DPRESENT_INTERVAL_IMMEDIATE,
        config_entries,
        factory,
        ..HarnessConfig::default()
    })
}

/// Present one frame that keeps the GPU busy, with an EVENT query issued at its end.
fn present_busy_frame(h: &Harness) -> Query<'_> {
    const fn corner(x: f32, y: f32) -> PosColorVertex {
        PosColorVertex {
            x,
            y,
            z: 0.5,
            color: 0x0801_0101,
        }
    }
    let quad = [
        corner(-1.0, 1.0),
        corner(1.0, 1.0),
        corner(-1.0, -1.0),
        corner(1.0, 1.0),
        corner(1.0, -1.0),
        corner(-1.0, -1.0),
    ];
    let batch: Vec<PosColorVertex> = quad
        .iter()
        .copied()
        .cycle()
        .take(QUADS_PER_DRAW * 6)
        .collect();
    let query = h
        .create_query(D3DQUERYTYPE_EVENT)
        .expect("EVENT query is supported");
    h.render_once(0xFF00_0000, |h| {
        assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ONE), D3D_OK);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
        let triangles = u32::try_from(QUADS_PER_DRAW * 2).expect("a triangle count fits u32");
        for _ in 0..BUSY_QUADS / QUADS_PER_DRAW {
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, triangles, &batch),
                D3D_OK,
                "a busy draw"
            );
        }
        assert_eq!(query.issue(D3DISSUE_END), D3D_OK, "Issue(END)");
    });
    query
}

/// Present `count` frames that clear and draw nothing.
fn present_empty_frames(h: &Harness, count: u32) {
    for _ in 0..count {
        h.render_once(0xFF00_0000, |_| {});
    }
}

/// The busy frame's query, polled once without a flush, reports the frame finished.
fn assert_finished(query: &Query<'_>, what: &str) {
    assert_eq!(query.status(0), D3D_OK, "{what}");
}

/// A plain device waits at its third present after a frame for that frame to finish.
#[test]
fn a_plain_device_finishes_a_frame_three_presents_later() {
    let h = device(Factory::Plain, EVENT_FROM_GPU);
    let query = present_busy_frame(&h);
    present_empty_frames(&h, 3);
    assert_finished(&query, "three presents after it, the busy frame finished");
}

/// An extended device that asks for a latency of one waits for the frame at the next present.
#[test]
fn a_latency_of_one_finishes_a_frame_at_the_next_present() {
    let h = device(Factory::Extended, EVENT_FROM_GPU);
    assert_eq!(h.set_maximum_frame_latency(1), D3D_OK);
    let query = present_busy_frame(&h);
    present_empty_frames(&h, 1);
    assert_finished(&query, "one present after it, the busy frame finished");
}

/// Presents that wait for the screen finish their frames once shown, so a paced device goes on.
///
/// The busy frame's query also finishes three presents later, as on an
/// unpaced device. Twenty more paced presents at the 60 Hz cap take about a
/// third of a second when the presented handlers end their frames; if only
/// the one-second cap on the paced half ended them, they would take twenty.
#[test]
fn a_paced_device_finishes_its_frames_once_they_are_shown() {
    let h = device(Factory::Plain, PACED);
    let query = present_busy_frame(&h);
    present_empty_frames(&h, 3);
    assert_finished(
        &query,
        "three paced presents after it, the busy frame finished",
    );
    let started = Instant::now();
    present_empty_frames(&h, 20);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "twenty paced presents took {:?}: their frames ended by the cap, not by being shown",
        started.elapsed()
    );
}

/// A game latency of ten lets a fourth present return while the first frame's present is parked.
///
/// At the default of three the fourth present would wait for that frame.
#[test]
fn a_game_latency_of_ten_lets_a_fourth_present_past_a_parked_frame() {
    let (gate, h) = gated(Factory::Extended, "");
    assert_eq!(h.set_maximum_frame_latency(10), D3D_OK);
    present_while_gated(&h, &gate, 4);
    lift(&gate);
}

/// `present.maxFrameLatency=1` over a game asking for ten holds the second present.
#[test]
fn a_configured_latency_of_one_holds_the_second_present() {
    let (gate, h) = gated(Factory::Extended, "present.maxFrameLatency=1");
    assert_eq!(h.set_maximum_frame_latency(10), D3D_OK);
    assert_eq!(h.maximum_frame_latency(), (D3D_OK, 10), "the game's value");
    present_while_gated(&h, &gate, 1);
    next_present_waits_for_the_gate(&h, &gate);
}

/// `present.maxFrameLatency=10` over a game asking for one lets a fourth present return.
#[test]
fn a_configured_latency_of_ten_lets_a_fourth_present_past_a_game_asking_for_one() {
    let (gate, h) = gated(Factory::Extended, "present.maxFrameLatency=10");
    assert_eq!(h.set_maximum_frame_latency(1), D3D_OK);
    present_while_gated(&h, &gate, 4);
    lift(&gate);
}

/// At the default latency, the fourth present waits for the parked first frame to be shown.
///
/// The first frame's GPU work finished long before; only its present, which
/// the gate holds, keeps it unfinished. A bound that counted GPU completion
/// alone would let the fourth present return.
#[test]
fn a_paced_frame_holds_the_present_three_later_until_it_is_shown() {
    let (gate, h) = gated(Factory::Plain, "");
    present_while_gated(&h, &gate, 3);
    next_present_waits_for_the_gate(&h, &gate);
}

/// A `D3DCREATE_MULTITHREADED` device whose presents wait for the screen, its presenter parked.
///
/// `extra` adds entries after the gate, the occluded presents and the 60 Hz
/// cap that paces every present.
fn gated(factory: Factory, extra: &str) -> (PathBuf, Harness) {
    let (gate, entry) = super::present_split::gate_file("latency");
    let entries = format!("{entry};debug.presentOccluded=true;present.maxFps=60;{extra}");
    let h = Harness::create(&HarnessConfig {
        config_entries: Box::leak(entries.into_boxed_str()),
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        factory,
        ..HarnessConfig::default()
    });
    (gate, h)
}

/// Present `count` frames, each returning while the gate stands.
fn present_while_gated(h: &Harness, gate: &Path, count: u32) {
    for at in 1..=count {
        assert_eq!(h.clear_target(0xFF00_0000), D3D_OK, "clear frame {at}");
        assert_eq!(
            h.present(),
            D3D_OK,
            "present {at} returns while the gate stands"
        );
    }
    assert!(gate.exists(), "nothing lifted the gate");
}

/// Present once more from a worker: held while the gate stands, returned once it lifts.
fn next_present_waits_for_the_gate(h: &Harness, gate: &Path) {
    let shared = h.shared();
    let returned = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = spawn_scoped(scope, || {
            let hr = shared.present();
            returned.store(true, Ordering::Release);
            hr
        });
        let watch = Instant::now();
        while watch.elapsed() < HELD_PRESENT_WATCH {
            assert!(h.pump(), "WM_QUIT while the present waits");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            !returned.load(Ordering::Acquire),
            "the present waits for the parked frame"
        );
        lift(gate);
        assert_eq!(
            worker.join().expect("the presenting worker panicked"),
            D3D_OK,
            "the held present returns once the gate lifts"
        );
    });
}

fn lift(gate: &Path) {
    assert!(gate.exists(), "nothing lifted the gate");
    std::fs::remove_file(gate).expect("remove the gate file");
}
