//! Unit tests for the attachment registry: one record per attached view.
//!
//! No `AppKit` runs here. The view, layer and window addresses are opaque
//! numbers the registry only ever compares, and the PE-side sinks are
//! local `AtomicU32`s whose addresses stand in for the device-owned box, so
//! the publish helpers write into real words the test can read back. Every
//! test takes the addresses it registers from a block `test_address_block`
//! hands it alone, and the test that looks records up by window takes that
//! window from its block too, since the registry is one map for the process
//! and the tests run in parallel.
//!
//! What they pin is that two records coexist and are torn down one at a
//! time, that each record seeds and carries its own derived state, that the
//! present-geometry streaks are independent, that nothing is written into a
//! sink once its record is unregistered, that a view address handed out
//! again names a new record rather than the old one, and that a Reset's
//! re-pacing makes one record's reconciliation due at once, defers to a
//! refresh already in flight, and leaves the other records' cadence alone.
//! They also pin which client surface a window's departing device hands the
//! window to: the newest live record's on that window that holds one, that
//! only that record's device is asked to show it again when Wine hides its
//! client view, and when a window that reads visible counts as displaying
//! nothing: after a run of undisplayed paced presents, until the next
//! displayed one.

use core::sync::atomic::{AtomicU32, Ordering};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use mtld3d_shared::mtl::{ClientSurfaceRequest, ColorSpacePolicy};

use super::{
    AttachFlags, AttachLatches, Attachment, HEADROOM_REFRESH_PRESENTS, UNDISPLAYED_PACED_LIMIT,
    find, find_by_layer, live, next_unshown_release, publish_backing_scale, register,
    request_client_surface, request_cursor_kick_all, retain_newest_surface_on, test_address_block,
    undisplayed_run_edge, unregister,
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
        hwnd: 0x40,
        flags,
        color_space: ColorSpacePolicy::Passthrough,
        pacing_bits: pack_pacing(&PresentPacing {
            vsync_requested: true,
            max_fps: 60,
        }),
        backing_scale,
        backing_scale_sink,
        cursor_kick_sink,
        client_surface_request_sink: 0,
        client_surface: 0,
    }
}

fn is_same(a: &Arc<Attachment>, b: &Arc<Attachment>) -> bool {
    Arc::ptr_eq(a, b)
}

#[test]
fn two_attachments_coexist() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    let a = register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 2, None));
    assert!(is_same(&find(view_a).expect("A is live"), &a));
    assert!(is_same(&find(view_b).expect("B is live"), &b));
    let live_views: Vec<usize> = live().iter().map(|att| att.view()).collect();
    assert!(live_views.contains(&view_a) && live_views.contains(&view_b));
    assert_eq!(a.backing_scale(), 1, "A keeps what its own attach latched");
    assert_eq!(b.backing_scale(), 2, "B keeps what its own attach latched");

    unregister(view_a);
    unregister(view_b);
}

#[test]
fn detaching_the_first_leaves_the_second() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(
        unregister(view_a).is_some(),
        "the first device's teardown finds its record"
    );
    assert!(find(view_a).is_none(), "the first record is gone");
    assert!(
        is_same(&find(view_b).expect("the second record is untouched"), &b),
        "the second device keeps its own record"
    );

    unregister(view_b);
}

#[test]
fn detaching_the_second_leaves_the_first() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    let a = register(
        view_a,
        view_a + 8,
        &latches(AttachFlags::HDR_ACTIVE, 2, None),
    );
    register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
    a.set_window_occluded(true);
    a.set_headroom(4.0);
    assert!(
        unregister(view_b).is_some(),
        "the second device's teardown finds its record"
    );
    let still_a = find(view_a).expect("the first record is untouched");
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

    unregister(view_a);
}

#[test]
fn an_unknown_view_detaches_nothing() {
    let view = test_address_block();

    let att = register(view, view + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(
        unregister(view + 16).is_none(),
        "another device's teardown finds no record"
    );
    assert!(
        unregister(0).is_none(),
        "a device that never attached finds no record"
    );
    assert!(is_same(&find(view).expect("the record is untouched"), &att));

    unregister(view);
}

#[test]
fn lookup_by_layer_finds_its_own_record() {
    let view_a = test_address_block();
    let layer_a = view_a + 0x100;
    let view_b = view_a + 0x1000;
    let layer_b = view_b + 0x100;

    let a = register(view_a, layer_a, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, layer_b, &latches(AttachFlags::empty(), 1, None));
    assert!(is_same(&find_by_layer(layer_a).expect("A by layer"), &a));
    assert!(is_same(&find_by_layer(layer_b).expect("B by layer"), &b));
    assert!(
        find_by_layer(layer_b + 8).is_none(),
        "an unattached layer has no record"
    );
    unregister(view_a);
    assert!(
        find_by_layer(layer_a).is_none(),
        "a detached layer has no record"
    );
    assert!(is_same(
        &find_by_layer(layer_b).expect("B is untouched"),
        &b
    ));

    unregister(view_b);
}

#[test]
fn each_record_seeds_its_own_defaults() {
    let view = test_address_block();

    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 144,
    };
    let att = register(
        view,
        view + 8,
        &AttachLatches {
            hwnd: 0x40,
            flags: AttachFlags::HDR_ENABLE_REQUESTED | AttachFlags::HDR_ACTIVE,
            color_space: ColorSpacePolicy::Accurate,
            pacing_bits: pack_pacing(&pacing),
            backing_scale: 2,
            backing_scale_sink: 0,
            cursor_kick_sink: 0,
            client_surface_request_sink: 0,
            client_surface: 0,
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

    unregister(view);
}

#[test]
fn derived_state_is_per_record() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    let a = register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
    a.set_hdr_active(true);
    a.set_window_occluded(true);
    a.set_headroom(2.5);
    a.set_last_logged_headroom(2.5);
    a.set_min_present_duration(1.0 / 120.0);
    a.set_backing_scale(2);
    a.set_window(view_a + 0xF00);
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
    assert_eq!(a.window(), view_a + 0xF00);

    unregister(view_a);
    unregister(view_b);
}

#[test]
fn geometry_streaks_are_independent() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    let a = register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
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

    unregister(view_a);
    unregister(view_b);
}

#[test]
fn a_detached_record_publishes_nowhere() {
    let view = test_address_block();

    let sinks = (AtomicU32::new(0), AtomicU32::new(0));
    let att = register(
        view,
        view + 8,
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

    assert!(unregister(view).is_some());
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
    let view = test_address_block();

    let sinks = (AtomicU32::new(0), AtomicU32::new(0));
    let first = register(
        view,
        view + 8,
        &latches(AttachFlags::empty(), 1, Some(&sinks)),
    );
    assert!(unregister(view).is_some());
    let second = register(view, view + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(!is_same(&first, &second), "the address names a new record");
    assert!(is_same(
        &find(view).expect("the new record is live"),
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
    assert!(unregister(view).is_some());
    assert!(find(view).is_none());
}

#[test]
fn a_reset_makes_the_refresh_due_at_once() {
    let view = test_address_block();

    let att = register(view, view + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(
        att.begin_headroom_refresh(),
        "the first present queues the attach-time refresh"
    );
    att.end_headroom_refresh();
    assert!(
        !att.begin_headroom_refresh(),
        "one present in, the next poll is an interval away"
    );
    assert!(
        att.request_refresh(),
        "a Reset with no refresh in flight queues one now"
    );
    att.end_headroom_refresh();
    assert!(
        !att.begin_headroom_refresh(),
        "the Reset's refresh was the one refresh; the interval restarted with it"
    );

    unregister(view);
}

#[test]
fn a_reset_during_a_refresh_in_flight_defers_to_the_next_present() {
    let view = test_address_block();

    let att = register(view, view + 8, &latches(AttachFlags::empty(), 1, None));
    assert!(att.begin_headroom_refresh(), "the poll queued a refresh");
    assert!(
        !att.request_refresh(),
        "the Reset finds it in flight and queues nothing"
    );
    att.end_headroom_refresh();
    assert!(
        att.begin_headroom_refresh(),
        "the next present queues another, which reads the pacing the Reset latched"
    );

    unregister(view);
}

#[test]
fn a_reset_repaces_one_record_only() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;

    let a = register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
    // Both records are past their attach-time refresh.
    assert!(a.begin_headroom_refresh());
    a.end_headroom_refresh();
    assert!(b.begin_headroom_refresh());
    b.end_headroom_refresh();

    let immediate = pack_pacing(&PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    });
    a.set_pacing_bits(immediate);
    assert!(a.request_refresh(), "A's Reset queues A's refresh");
    assert_eq!(a.pacing_bits(), immediate);
    assert_eq!(
        b.pacing_bits(),
        pack_pacing(&PresentPacing {
            vsync_requested: true,
            max_fps: 60,
        }),
        "B keeps the pacing its own attach latched"
    );
    assert!(
        !b.begin_headroom_refresh(),
        "B's next present is not due: A's Reset left B's interval alone"
    );
    a.end_headroom_refresh();

    unregister(view_a);
    unregister(view_b);
}

/// Latches for a record on `hwnd` holding a reference on `client_surface`.
fn surface_latches(hwnd: u64, client_surface: usize) -> AttachLatches {
    AttachLatches {
        hwnd,
        client_surface,
        ..latches(AttachFlags::empty(), 1, None)
    }
}

#[test]
fn the_newest_live_record_on_the_window_names_the_surface() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;
    let view_c = view_a + 0x2000;
    let view_other = view_a + 0x3000;
    let hwnd = u64::try_from(view_a + 0x40).expect("a test address fits a u64");

    let retained = |surface: usize| surface;
    assert_eq!(
        retain_newest_surface_on(hwnd, retained),
        None,
        "no record yet"
    );

    register(view_a, view_a + 8, &surface_latches(hwnd, 0xa000));
    std::thread::sleep(std::time::Duration::from_millis(2));
    register(view_b, view_b + 8, &surface_latches(hwnd, 0xb000));
    std::thread::sleep(std::time::Duration::from_millis(2));
    register(view_c, view_c + 8, &surface_latches(hwnd, 0));
    register(
        view_other,
        view_other + 8,
        &surface_latches(hwnd + 1, 0xd000),
    );
    assert_eq!(
        retain_newest_surface_on(hwnd, retained),
        Some(0xb000),
        "the newest record holding a surface, on this window only"
    );

    unregister(view_b);
    assert_eq!(
        retain_newest_surface_on(hwnd, retained),
        Some(0xa000),
        "an unregistered record names nothing"
    );
    assert_eq!(
        retain_newest_surface_on(hwnd, |_| 0),
        None,
        "a surface the retain refuses is not handed back"
    );

    unregister(view_a);
    assert_eq!(retain_newest_surface_on(hwnd, retained), None);
    unregister(view_c);
    unregister(view_other);
}

/// The unshown pace releases one present per duration, counted from the schedule.
///
/// A frame ready before its slot waits for it, even when the presenter gets
/// to the frame late, so a late wake does not slow the pace; a frame ready
/// after its slot goes when it is ready, so a slow game does not burst.
#[test]
fn the_unshown_pace_releases_one_present_per_duration() {
    let pace = Duration::from_millis(16);
    let start = Instant::now();
    assert_eq!(
        next_unshown_release(None, start, pace),
        start,
        "the first present goes when it is ready"
    );
    assert_eq!(
        next_unshown_release(Some(start), start, pace),
        start + pace,
        "the next one a duration after the previous release"
    );
    assert_eq!(
        next_unshown_release(Some(start + pace), start + pace / 2, pace),
        start + pace * 2,
        "counted from the previous release, not from when the frame was ready"
    );
    let late = start + pace * 10;
    assert_eq!(
        next_unshown_release(Some(start), late, pace),
        late,
        "a frame ready after its slot goes when it is ready"
    );
}

/// Frames queued during a late wake keep one slot each on the schedule.
///
/// A background process has its timers coalesced, so the presenter reaches
/// frames well past their slots; the slot depends on when each frame was
/// ready and on the previous slot, never on when the presenter asks. Three
/// frames ready early get the next three slots, which are already past by
/// then, so they go at once and the average pace holds.
#[test]
fn frames_queued_during_a_late_wake_keep_their_slots() {
    let pace = Duration::from_millis(33);
    let start = Instant::now();
    let ready = start + pace / 2;
    let mut slot = start;
    for n in 1..=3 {
        slot = next_unshown_release(Some(slot), ready, pace);
        assert_eq!(
            slot,
            start + pace * n,
            "frame {n} takes its slot on the schedule"
        );
    }
}

/// Each record keeps its own unshown pace, as the display paces each layer on its own.
#[test]
fn each_record_keeps_its_own_unshown_pace() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;
    let a = register(view_a, view_a + 8, &latches(AttachFlags::empty(), 1, None));
    let b = register(view_b, view_b + 8, &latches(AttachFlags::empty(), 1, None));
    let pace = Duration::from_millis(16);
    let now = Instant::now() + Duration::from_secs(1);
    assert_eq!(a.unshown_release_at(now, pace), now);
    assert_eq!(
        a.unshown_release_at(now, pace),
        now + pace,
        "the record kept its release"
    );
    assert_eq!(
        b.unshown_release_at(now, pace),
        now,
        "the other record starts its own"
    );
    assert_eq!(a.unshown_release_at(now, pace), now + pace * 2);
    unregister(view_a);
    unregister(view_b);
}

/// The run of undisplayed paced presents changes the window's state only at its limit and its end.
#[test]
fn an_undisplayed_run_changes_the_state_at_the_limit_and_at_its_end() {
    assert_eq!(
        undisplayed_run_edge(0, false),
        None,
        "one undisplayed present"
    );
    assert_eq!(
        undisplayed_run_edge(UNDISPLAYED_PACED_LIMIT - 1, false),
        Some(true),
        "the present that completes the run"
    );
    assert_eq!(
        undisplayed_run_edge(UNDISPLAYED_PACED_LIMIT, false),
        None,
        "a run past the limit"
    );
    assert_eq!(
        undisplayed_run_edge(u32::MAX, false),
        None,
        "a run saturated at the top"
    );
    assert_eq!(
        undisplayed_run_edge(UNDISPLAYED_PACED_LIMIT - 1, true),
        None,
        "a displayed present ends a run short of the limit"
    );
    assert_eq!(
        undisplayed_run_edge(UNDISPLAYED_PACED_LIMIT, true),
        Some(false),
        "a displayed present ends a run that reached it"
    );
    assert_eq!(
        undisplayed_run_edge(0, true),
        None,
        "displayed after displayed"
    );
}

/// A record displays nothing after the limit of undisplayed paced presents, until one is displayed.
#[test]
fn a_record_displays_nothing_from_the_limit_until_a_displayed_present() {
    let view = test_address_block();
    let att = register(view, view + 8, &latches(AttachFlags::empty(), 1, None));
    for _ in 0..4 {
        assert_eq!(att.note_paced_present(false), None);
    }
    assert_eq!(
        att.note_paced_present(true),
        None,
        "a displayed present resets a short run"
    );
    for _ in 1..UNDISPLAYED_PACED_LIMIT {
        assert_eq!(att.note_paced_present(false), None);
        assert!(!att.displays_nothing(), "short of the limit");
    }
    assert_eq!(
        att.note_paced_present(false),
        Some(true),
        "the run reached the limit"
    );
    assert!(att.displays_nothing());
    assert_eq!(
        att.note_paced_present(false),
        None,
        "no second edge past it"
    );
    assert!(att.displays_nothing());
    assert_eq!(
        att.note_paced_present(true),
        Some(false),
        "the first displayed present"
    );
    assert!(!att.displays_nothing());
    unregister(view);
}

/// Latches for a record on `hwnd` holding `client_surface`, asking through `sink`.
fn request_latches(hwnd: u64, client_surface: usize, sink: &AtomicU32) -> AttachLatches {
    AttachLatches {
        client_surface_request_sink: core::ptr::from_ref(sink) as usize,
        ..surface_latches(hwnd, client_surface)
    }
}

#[test]
fn only_the_newest_record_with_a_surface_asks_for_it_again() {
    let view_a = test_address_block();
    let view_b = view_a + 0x1000;
    let view_c = view_a + 0x2000;
    let hwnd = u64::try_from(view_a + 0x40).expect("a test address fits a u64");

    let sink_a = AtomicU32::new(0);
    let sink_b = AtomicU32::new(0);
    let sink_c = AtomicU32::new(0);
    register(view_a, view_a + 8, &request_latches(hwnd, 0xa000, &sink_a));
    assert!(
        request_client_surface(view_a, ClientSurfaceRequest::VIEW_HIDDEN),
        "the only device on the window asks"
    );
    assert!(
        request_client_surface(view_a, ClientSurfaceRequest::CLIENT_AREA_CHANGED),
        "and asks again for another reason"
    );
    assert_eq!(
        sink_a.swap(0, Ordering::Acquire),
        (ClientSurfaceRequest::VIEW_HIDDEN | ClientSurfaceRequest::CLIENT_AREA_CHANGED).bits(),
        "the reasons accumulate until the device takes them"
    );

    std::thread::sleep(std::time::Duration::from_millis(2));
    register(view_b, view_b + 8, &request_latches(hwnd, 0xb000, &sink_b));
    std::thread::sleep(std::time::Duration::from_millis(2));
    register(view_c, view_c + 8, &request_latches(hwnd, 0, &sink_c));
    assert!(
        !request_client_surface(view_a, ClientSurfaceRequest::VIEW_HIDDEN),
        "a device a newer one on its window outranks is hidden on purpose"
    );
    assert!(
        !request_client_surface(view_c, ClientSurfaceRequest::VIEW_HIDDEN),
        "a record without a surface has nothing to show"
    );
    assert!(request_client_surface(
        view_b,
        ClientSurfaceRequest::VIEW_HIDDEN
    ));
    assert_eq!(sink_a.load(Ordering::Acquire), 0);
    assert_eq!(sink_c.load(Ordering::Acquire), 0);
    assert_eq!(
        sink_b.swap(0, Ordering::Acquire),
        ClientSurfaceRequest::VIEW_HIDDEN.bits()
    );

    unregister(view_b);
    assert!(
        !request_client_surface(view_b, ClientSurfaceRequest::VIEW_HIDDEN),
        "an unregistered record asks nothing"
    );
    assert_eq!(sink_b.load(Ordering::Acquire), 0);
    assert!(
        request_client_surface(view_a, ClientSurfaceRequest::VIEW_HIDDEN),
        "the window's newest device again once the later one left"
    );

    unregister(view_a);
    unregister(view_c);
}

#[test]
fn a_record_without_a_request_word_asks_nothing() {
    let view = test_address_block();
    let hwnd = u64::try_from(view + 0x40).expect("a test address fits a u64");

    register(view, view + 8, &surface_latches(hwnd, 0xf000));
    assert!(!request_client_surface(
        view,
        ClientSurfaceRequest::VIEW_HIDDEN
    ));
    unregister(view);
}
