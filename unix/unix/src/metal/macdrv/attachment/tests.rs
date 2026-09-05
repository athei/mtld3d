//! Unit tests for the attachment registry: one record per attached view.
//!
//! No `AppKit` runs here. The view, layer and window addresses are opaque
//! constants the registry only ever compares, and the PE-side sinks are
//! local `AtomicU32`s whose addresses stand in for the device-owned box, so
//! the publish helpers write into real words the test can read back. Every
//! test uses its own address range, since the registry is one map for the
//! process and the tests run in parallel.
//!
//! What they pin is that two records coexist and are torn down one at a
//! time, that each record seeds and carries its own derived state, that the
//! present-geometry streaks are independent, that nothing is written into a
//! sink once its record is unregistered, and that a view address handed out
//! again names a new record rather than the old one.

use core::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use mtld3d_shared::mtl::ColorSpacePolicy;

use super::{
    AttachFlags, AttachLatches, Attachment, HEADROOM_REFRESH_PRESENTS, find, find_by_layer, live,
    publish_backing_scale, register, request_cursor_kick_all, unregister,
};
use crate::metal::{
    command::PresentGeometry,
    macdrv::{PresentPacing, pack_pacing},
};

fn latches(
    flags: AttachFlags,
    backing_scale: u32,
    sinks: Option<&(AtomicU32, AtomicU32)>,
) -> AttachLatches {
    let (backing_scale_sink, cursor_kick_sink) = sinks.map_or((0, 0), |(scale, kick)| {
        (
            core::ptr::from_ref(scale) as usize,
            core::ptr::from_ref(kick) as usize,
        )
    });
    AttachLatches {
        flags,
        color_space: ColorSpacePolicy::Passthrough,
        pacing_bits: pack_pacing(&PresentPacing {
            vsync_requested: true,
            max_fps: 60,
        }),
        backing_scale,
        backing_scale_sink,
        cursor_kick_sink,
    }
}

fn is_same(a: &Arc<Attachment>, b: &Arc<Attachment>) -> bool {
    Arc::ptr_eq(a, b)
}

#[test]
fn two_attachments_coexist() {
    const VIEW_A: usize = 0x1_0000;
    const VIEW_B: usize = 0x1_1000;

    let a = register(VIEW_A, VIEW_A + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(VIEW_B, VIEW_B + 8, &latches(AttachFlags::empty(), 2, None));
    assert!(is_same(&find(VIEW_A).expect("A is live"), &a));
    assert!(is_same(&find(VIEW_B).expect("B is live"), &b));
    let live_views: Vec<usize> = live().iter().map(|att| att.view()).collect();
    assert!(live_views.contains(&VIEW_A) && live_views.contains(&VIEW_B));
    assert_eq!(a.backing_scale(), 1, "A keeps what its own attach latched");
    assert_eq!(b.backing_scale(), 2, "B keeps what its own attach latched");

    unregister(VIEW_A);
    unregister(VIEW_B);
}

#[test]
fn detaching_the_first_leaves_the_second() {
    const VIEW_A: usize = 0x2_0000;
    const VIEW_B: usize = 0x2_1000;

    register(VIEW_A, VIEW_A + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(VIEW_B, VIEW_B + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(
        unregister(VIEW_A).is_some(),
        "the first device's teardown finds its record"
    );
    assert!(find(VIEW_A).is_none(), "the first record is gone");
    assert!(
        is_same(&find(VIEW_B).expect("the second record is untouched"), &b),
        "the second device keeps its own record"
    );

    unregister(VIEW_B);
}

#[test]
fn detaching_the_second_leaves_the_first() {
    const VIEW_A: usize = 0x3_0000;
    const VIEW_B: usize = 0x3_1000;

    let a = register(
        VIEW_A,
        VIEW_A + 8,
        &latches(AttachFlags::HDR_ACTIVE, 2, None),
    );
    register(VIEW_B, VIEW_B + 8, &latches(AttachFlags::empty(), 1, None));
    a.set_window_occluded(true);
    a.set_headroom(4.0);
    assert!(
        unregister(VIEW_B).is_some(),
        "the second device's teardown finds its record"
    );
    let still_a = find(VIEW_A).expect("the first record is untouched");
    assert!(is_same(&still_a, &a));
    assert!(
        still_a.hdr_active(),
        "the first device's layer mode survives"
    );
    assert!(
        still_a.window_occluded(),
        "the first device's occlusion survives"
    );
    assert_eq!(still_a.headroom().to_bits(), 4.0_f32.to_bits());
    assert_eq!(still_a.backing_scale(), 2);

    unregister(VIEW_A);
}

#[test]
fn an_unknown_view_detaches_nothing() {
    const VIEW: usize = 0x4_0000;

    let att = register(VIEW, VIEW + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(
        unregister(VIEW + 16).is_none(),
        "another device's teardown finds no record"
    );
    assert!(
        unregister(0).is_none(),
        "a device that never attached finds no record"
    );
    assert!(is_same(&find(VIEW).expect("the record is untouched"), &att));

    unregister(VIEW);
}

#[test]
fn lookup_by_layer_finds_its_own_record() {
    const VIEW_A: usize = 0x5_0000;
    const LAYER_A: usize = 0x5_0100;
    const VIEW_B: usize = 0x5_1000;
    const LAYER_B: usize = 0x5_1100;

    let a = register(VIEW_A, LAYER_A, &latches(AttachFlags::empty(), 1, None));
    let b = register(VIEW_B, LAYER_B, &latches(AttachFlags::empty(), 1, None));
    assert!(is_same(&find_by_layer(LAYER_A).expect("A by layer"), &a));
    assert!(is_same(&find_by_layer(LAYER_B).expect("B by layer"), &b));
    assert!(
        find_by_layer(LAYER_B + 8).is_none(),
        "an unattached layer has no record"
    );
    unregister(VIEW_A);
    assert!(
        find_by_layer(LAYER_A).is_none(),
        "a detached layer has no record"
    );
    assert!(is_same(
        &find_by_layer(LAYER_B).expect("B is untouched"),
        &b
    ));

    unregister(VIEW_B);
}

#[test]
fn each_record_seeds_its_own_defaults() {
    const VIEW: usize = 0x6_0000;

    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 144,
    };
    let att = register(
        VIEW,
        VIEW + 8,
        &AttachLatches {
            flags: AttachFlags::HDR_ENABLE_REQUESTED | AttachFlags::HDR_ACTIVE,
            color_space: ColorSpacePolicy::Accurate,
            pacing_bits: pack_pacing(&pacing),
            backing_scale: 2,
            backing_scale_sink: 0,
            cursor_kick_sink: 0,
        },
    );
    assert_eq!(
        att.headroom().to_bits(),
        1.0_f32.to_bits(),
        "the headroom the present pass treats as the identity curve",
    );
    assert!(
        att.last_logged_headroom().is_none(),
        "the first refresh logs a baseline"
    );
    assert!(
        att.begin_headroom_refresh(),
        "the first present queues a refresh"
    );
    assert!(
        !att.begin_headroom_refresh(),
        "one refresh outstanding at a time"
    );
    att.end_headroom_refresh();
    let due_again = (0..=HEADROOM_REFRESH_PRESENTS).any(|_| att.begin_headroom_refresh());
    assert!(due_again, "a refresh is due again after the interval");
    assert!(
        !att.window_occluded(),
        "not occluded until the observer says so"
    );
    assert_eq!(
        att.min_present_duration_sec().to_bits(),
        0.0_f64.to_bits(),
        "no throttle until attach derives one"
    );
    assert_eq!(
        att.pacing_bits(),
        pack_pacing(&pacing),
        "the pacing the latches carried"
    );
    assert_eq!(att.backing_scale(), 2, "the scale the latches carried");
    assert!(att.hdr_active(), "the layer mode attach configured");
    assert!(att.hdr_enable_requested());
    assert_eq!(att.color_space(), ColorSpacePolicy::Accurate);
    assert_eq!(att.window(), 0, "no window until the main thread finds one");

    unregister(VIEW);
}

#[test]
fn derived_state_is_per_record() {
    const VIEW_A: usize = 0x7_0000;
    const VIEW_B: usize = 0x7_1000;

    let a = register(VIEW_A, VIEW_A + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(VIEW_B, VIEW_B + 8, &latches(AttachFlags::empty(), 1, None));
    a.set_hdr_active(true);
    a.set_window_occluded(true);
    a.set_headroom(2.5);
    a.set_last_logged_headroom(2.5);
    a.set_min_present_duration(1.0 / 120.0);
    a.set_backing_scale(2);
    a.set_window(0x7_0F00);
    a.set_pacing_bits(pack_pacing(&PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    }));

    assert!(!b.hdr_active());
    assert!(!b.window_occluded());
    assert_eq!(b.headroom().to_bits(), 1.0_f32.to_bits());
    assert!(b.last_logged_headroom().is_none());
    assert_eq!(b.min_present_duration_sec().to_bits(), 0.0_f64.to_bits());
    assert_eq!(b.backing_scale(), 1);
    assert_eq!(b.window(), 0);
    assert_eq!(
        b.pacing_bits(),
        pack_pacing(&PresentPacing {
            vsync_requested: true,
            max_fps: 60,
        }),
    );
    assert!(a.hdr_active() && a.window_occluded());
    assert_eq!(a.headroom().to_bits(), 2.5_f32.to_bits());
    assert_eq!(
        a.last_logged_headroom().map(f32::to_bits),
        Some(2.5_f32.to_bits())
    );
    assert_eq!(
        a.min_present_duration_sec().to_bits(),
        (1.0_f64 / 120.0).to_bits()
    );
    assert_eq!(a.backing_scale(), 2);
    assert_eq!(a.window(), 0x7_0F00);

    unregister(VIEW_A);
    unregister(VIEW_B);
}

#[test]
fn geometry_streaks_are_independent() {
    const VIEW_A: usize = 0x8_0000;
    const VIEW_B: usize = 0x8_1000;

    let a = register(VIEW_A, VIEW_A + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(VIEW_B, VIEW_B + 8, &latches(AttachFlags::empty(), 1, None));
    let enlarged = PresentGeometry {
        src: (640, 480),
        dst: (1280, 960),
    };
    let mut settled = false;
    for _ in 0..30 {
        settled = a.present_settled(enlarged);
    }
    assert!(settled, "thirty presents at one geometry settle A");
    assert!(
        !b.present_settled(enlarged),
        "B's first present is its own count, not A's"
    );
    let other = PresentGeometry {
        src: (640, 480),
        dst: (1290, 970),
    };
    assert!(!a.present_settled(other), "a change on A restarts A");
    assert!(
        !b.present_settled(enlarged),
        "B is two presents in, untouched by A"
    );

    unregister(VIEW_A);
    unregister(VIEW_B);
}

#[test]
fn a_detached_record_publishes_nowhere() {
    const VIEW: usize = 0x9_0000;

    let sinks = (AtomicU32::new(0), AtomicU32::new(0));
    let att = register(
        VIEW,
        VIEW + 8,
        &latches(AttachFlags::empty(), 1, Some(&sinks)),
    );
    publish_backing_scale(&att, 2);
    assert_eq!(
        sinks.0.load(Ordering::Relaxed),
        2,
        "a live record publishes into its sink"
    );
    request_cursor_kick_all();
    assert_eq!(
        sinks.1.load(Ordering::Acquire),
        1,
        "a live record is kicked"
    );

    assert!(unregister(VIEW).is_some());
    sinks.0.store(0, Ordering::Relaxed);
    sinks.1.store(0, Ordering::Relaxed);
    publish_backing_scale(&att, 3);
    assert_eq!(
        sinks.0.load(Ordering::Relaxed),
        0,
        "nothing is written after the teardown"
    );
    request_cursor_kick_all();
    assert_eq!(
        sinks.1.load(Ordering::Acquire),
        0,
        "nothing is kicked after the teardown"
    );
}

#[test]
fn a_reregistered_view_address_is_a_new_record() {
    const VIEW: usize = 0xA_0000;

    let sinks = (AtomicU32::new(0), AtomicU32::new(0));
    let first = register(
        VIEW,
        VIEW + 8,
        &latches(AttachFlags::empty(), 1, Some(&sinks)),
    );
    assert!(unregister(VIEW).is_some());
    let second = register(VIEW, VIEW + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(!is_same(&first, &second), "the address names a new record");
    assert!(is_same(
        &find(VIEW).expect("the new record is live"),
        &second
    ));
    // The old record's handle is stale even though its key is live again:
    // the guard compares records, not keys, so its sink is never written.
    publish_backing_scale(&first, 2);
    assert_eq!(
        sinks.0.load(Ordering::Relaxed),
        0,
        "the stale record publishes nowhere"
    );
    assert!(unregister(VIEW).is_some());
    assert!(find(VIEW).is_none());
}
