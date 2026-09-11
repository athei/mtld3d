//! Unit tests for the software cursor's pure decisions.
//!
//! `sprite_origin` turns a pointer position and a sprite's geometry into the
//! overlay window's frame origin, in Cocoa's bottom-left screen coordinates.
//! `overlay_visible` folds the five visibility inputs into one answer, and the
//! tests pin that every blocker hides the sprite on its own. `rect_contains`
//! is the pointer-inside-the-client-area test, with its half-open edges, and
//! `peak_changed` decides which headroom moves re-render the sprite: the 5%
//! rule the log uses, plus the `1.0` boundary where the frame switches
//! pipelines. `warp_since` is the run-loop observer's test for a pointer warp
//! the sprite has not followed yet.

use objc2_core_foundation::{CGPoint, CGRect, CGSize};

use super::{
    CAPTURE_SILENCE_MS, Sprite, SpriteGeometry, VisibilityInputs, overlay_visible, peak_changed,
    pointer_captured, rect_contains, sprite_origin, warp_since,
};

fn geometry() -> SpriteGeometry {
    SpriteGeometry {
        width: 32.0,
        height: 32.0,
        hotspot_x: 4.0,
        hotspot_y: 6.0,
        scale: 2.0,
    }
}

#[test]
fn sprite_origin_puts_the_hotspot_under_the_pointer() {
    // The hotspot is 6 pt below the sprite's top; the window's origin is its
    // bottom, 26 pt below the pointer, and 4 pt to the left.
    assert_eq!(sprite_origin((100.0, 200.0), &geometry()), (96.0, 174.0));
}

#[test]
fn sprite_origin_with_a_zero_hotspot_hangs_the_sprite_below_the_pointer() {
    let geometry = SpriteGeometry {
        hotspot_x: 0.0,
        hotspot_y: 0.0,
        ..geometry()
    };
    assert_eq!(sprite_origin((100.0, 200.0), &geometry), (100.0, 168.0));
}

#[test]
fn sprite_geometry_is_in_points_not_sprite_pixels() {
    let sprite = Sprite {
        width: 64,
        height: 48,
        x_hotspot: 8,
        y_hotspot: 12,
        scale: 2,
        pixels: Box::new([]),
    };
    assert_eq!(
        SpriteGeometry::of(&sprite, 2),
        SpriteGeometry {
            width: 32.0,
            height: 24.0,
            hotspot_x: 4.0,
            hotspot_y: 6.0,
            scale: 2.0,
        }
    );
}

#[test]
fn sprite_geometry_divides_by_the_retina_factor_not_the_sprite_scale() {
    // `cursor.scale = 2` on a non-retina prefix: winemac shows the 64 px
    // hardware cursor at 64 pt, and so must the overlay.
    let sprite = Sprite {
        width: 64,
        height: 64,
        x_hotspot: 8,
        y_hotspot: 8,
        scale: 2,
        pixels: Box::new([]),
    };
    assert_eq!(
        SpriteGeometry::of(&sprite, 1),
        SpriteGeometry {
            width: 64.0,
            height: 64.0,
            hotspot_x: 8.0,
            hotspot_y: 8.0,
            scale: 1.0,
        }
    );
    // An unpublished layer scale reads as 1.
    assert_eq!(
        SpriteGeometry::of(&sprite, 0),
        SpriteGeometry::of(&sprite, 1)
    );
}

#[test]
fn overlay_shows_only_when_wanted_active_and_inside() {
    let shown =
        VisibilityInputs::WANTED | VisibilityInputs::APP_ACTIVE | VisibilityInputs::POINTER_INSIDE;
    assert!(overlay_visible(shown));
    assert!(!overlay_visible(shown - VisibilityInputs::WANTED));
    assert!(!overlay_visible(shown - VisibilityInputs::APP_ACTIVE));
    assert!(!overlay_visible(shown - VisibilityInputs::POINTER_INSIDE));
}

#[test]
fn an_occluded_or_miniaturized_game_window_hides_the_overlay() {
    let shown =
        VisibilityInputs::WANTED | VisibilityInputs::APP_ACTIVE | VisibilityInputs::POINTER_INSIDE;
    assert!(!overlay_visible(shown | VisibilityInputs::OCCLUDED));
    assert!(!overlay_visible(shown | VisibilityInputs::MINIATURIZED));
    assert!(!overlay_visible(VisibilityInputs::empty()));
}

#[test]
fn a_captured_pointer_hides_the_overlay() {
    let shown =
        VisibilityInputs::WANTED | VisibilityInputs::APP_ACTIVE | VisibilityInputs::POINTER_INSIDE;
    assert!(!overlay_visible(shown | VisibilityInputs::CAPTURED));
}

#[test]
fn pointer_is_captured_only_when_it_keeps_moving_without_events() {
    assert!(pointer_captured(CAPTURE_SILENCE_MS, true));
    assert!(pointer_captured(CAPTURE_SILENCE_MS * 10, true));
    // Idle pointer: silence is just idleness.
    assert!(!pointer_captured(CAPTURE_SILENCE_MS * 10, false));
    // Moving with events flowing: the events are what we saw it with.
    assert!(!pointer_captured(CAPTURE_SILENCE_MS - 1, true));
}

#[test]
fn rect_contains_is_half_open() {
    let rect = CGRect {
        origin: CGPoint { x: 10.0, y: 20.0 },
        size: CGSize {
            width: 100.0,
            height: 50.0,
        },
    };
    assert!(rect_contains(rect, CGPoint { x: 10.0, y: 20.0 }));
    assert!(rect_contains(rect, CGPoint { x: 109.9, y: 69.9 }));
    assert!(!rect_contains(rect, CGPoint { x: 110.0, y: 30.0 }));
    assert!(!rect_contains(rect, CGPoint { x: 50.0, y: 70.0 }));
    assert!(!rect_contains(rect, CGPoint { x: 9.9, y: 30.0 }));
}

#[test]
fn peak_changes_follow_the_five_percent_rule() {
    assert!(!peak_changed(2.0, 2.0));
    assert!(!peak_changed(2.0, 2.08));
    assert!(peak_changed(2.0, 2.2));
    assert!(peak_changed(2.0, 1.8));
}

#[test]
fn crossing_the_passthrough_boundary_always_re_renders() {
    // 1.0 to 1.02 is well under 5%, but the frame switches from the
    // pass-through to the BT.2446 pipeline there and the sprite must follow.
    assert!(peak_changed(1.0, 1.02));
    assert!(peak_changed(1.02, 1.0));
    assert!(!peak_changed(1.01, 1.02));
}

#[test]
fn a_warp_is_followed_once_and_a_cleared_warp_time_is_not_a_warp() {
    // A fresh warp time is a move nothing else reported.
    assert!(warp_since(0.0, 12.5));
    assert!(warp_since(12.5, 13.0));
    // The same warp seen again is not followed twice.
    assert!(!warp_since(12.5, 12.5));
    // winemac clears the time once an event newer than the warp arrived; the
    // pointer watch handled that event, so there is nothing to follow.
    assert!(!warp_since(12.5, 0.0));
    assert!(!warp_since(0.0, 0.0));
}

fn observation(x: f64, millis: u64) -> super::PointerObservation {
    super::PointerObservation {
        position: CGPoint { x, y: 0.0 },
        now: millis * 1_000_000,
        warp: 0.0,
        capture_epoch: 0,
        flags: super::PointerFlags::SHOWN | super::PointerFlags::ACTIVE,
    }
}

#[test]
fn captured_wine_input_without_clipping_keeps_the_watchdog_fresh() {
    let mut input = super::InputState::default();
    for event in 1u32..=100 {
        let sample = observation(f64::from(event), u64::from(event) * 20);
        assert_eq!(
            input.observe(event as usize, sample.position, sample.now),
            Some(false)
        );
        input.reconcile(&sample);
        assert!(!input.captured);
    }
}

#[test]
fn duplicate_local_and_current_event_observations_do_not_extend_silence() {
    let mut input = super::InputState::default();
    assert_eq!(input.observe(1, CGPoint::default(), 0), Some(false));
    for millis in [10, 60, 100] {
        let sample = observation(10.0, millis);
        assert_eq!(input.observe(1, sample.position, sample.now), None);
        input.reconcile(&sample);
    }
    assert!(
        input.captured,
        "an idle currentEvent cannot conceal external capture"
    );
    assert_eq!(input.at_ns, Some(0));
}

#[test]
fn new_input_recovers_a_capture_before_a_delayed_present_check() {
    let mut input = super::InputState::default();
    input.observe(1, CGPoint::default(), 0);
    input.reconcile(&observation(20.0, 100));
    assert!(input.captured);
    assert_eq!(
        input.observe(2, CGPoint { x: 20.0, y: 0.0 }, 101_000_000),
        Some(true)
    );
    input.reconcile(&observation(20.0, 150));
    assert!(
        !input.captured,
        "queued present samples current input, not an old decision"
    );
}

#[test]
fn inactive_or_hidden_watch_resumes_with_a_fresh_silence_baseline() {
    let mut input = super::InputState::default();
    input.observe(1, CGPoint::default(), 0);
    input.reconcile(&observation(20.0, 100));
    assert!(input.suspend());
    assert!(!input.captured);
    assert!(input.at_ns.is_none());
    assert!(!input.suspend());
    // Old currentEvent stays old, even after a long pause and pointer motion.
    assert_eq!(input.observe(1, CGPoint::default(), 10_000_000_000), None);
    input.reconcile(&observation(100.0, 10_000));
    assert!(!input.captured);
    assert_eq!(input.at_ns, Some(10_000_000_000));
    input.reconcile(&observation(200.0, 10_100));
    assert!(
        input.captured,
        "the active watchdog still detects new capture"
    );
}

#[test]
fn a_warp_or_clipped_move_is_legitimate_and_hide_show_bursts_reset_capture() {
    let mut input = super::InputState::default();
    input.observe(1, CGPoint::default(), 0);
    let mut sample = observation(100.0, 100);
    sample.warp = 12.5;
    input.reconcile(&sample);
    assert!(!input.captured);
    assert_eq!(input.at_ns, Some(100_000_000));
    sample.now += 100_000_000;
    input.reconcile(&sample);
    assert_eq!(
        input.at_ns,
        Some(100_000_000),
        "idle warp is not fresh input"
    );
    sample.position.x += 50.0;
    sample.flags.insert(super::PointerFlags::CLIPPED);
    input.reconcile(&sample);
    assert!(!input.captured);
    sample.flags.remove(super::PointerFlags::CLIPPED);
    sample.position.x += 50.0;
    sample.now += 100_000_000;
    input.reconcile(&sample);
    assert!(input.captured);
    sample.capture_epoch += 1; // Hide and show coalesced; latest visibility is still shown.
    input.reconcile(&sample);
    assert!(!input.captured);
}

fn attachment(view: usize) -> std::sync::Arc<super::Attachment> {
    super::attachment::register(
        view,
        view + 8,
        &super::attachment::AttachLatches {
            flags: super::attachment::AttachFlags::empty(),
            color_space: mtld3d_shared::mtl::ColorSpacePolicy::Passthrough,
            pacing_bits: 0,
            backing_scale: 1,
            backing_scale_sink: 0,
            cursor_kick_sink: 0,
        },
    )
}

fn request(hash: u64) -> mtld3d_shared::SetCursorOverlayParams {
    mtld3d_shared::SetCursorOverlayParams {
        hash,
        pixels_ptr: 0,
        pixels_len: 4,
        width: 1,
        height: 1,
        x_hotspot: 0,
        y_hotspot: 0,
        scale: 1,
        flags: mtld3d_shared::mtl::CursorOverlayFlags::VISIBLE,
        pad0: 0,
        view_handle: mtld3d_shared::MetalHandle::NULL,
    }
}

#[test]
fn concurrent_updates_keep_owner_mode_and_sprite_together() {
    use std::sync::Mutex;
    const A: usize = 0xc0_0000;
    const B: usize = 0xc0_1000;
    let a = attachment(A);
    let b = attachment(B);
    let shared = Mutex::new(super::Shared::default());
    std::thread::scope(|scope| {
        for (view, hash) in [(A, 10), (B, 20)] {
            let shared = &shared;
            scope.spawn(move || {
                for _ in 0..500 {
                    assert!(
                        shared
                            .lock()
                            .unwrap()
                            .update(view, &request(hash), Some(&[255; 4]))
                    );
                }
            });
        }
        for _ in 0..500 {
            let snapshot = shared.lock().unwrap().snapshot();
            if let Some(owner) = snapshot.owner {
                assert_eq!(
                    snapshot.hash,
                    if std::sync::Arc::ptr_eq(&owner, &a) {
                        10
                    } else {
                        20
                    }
                );
                assert!(snapshot.sprite.is_some());
            }
        }
    });
    assert!(super::attachment::unregister(A).is_some());
    assert!(super::attachment::unregister(B).is_some());
    assert!(!std::sync::Arc::ptr_eq(&a, &b));
}

#[test]
fn detached_owner_cannot_clear_a_new_owner_or_a_reused_address() {
    const A: usize = 0xc1_0000;
    const B: usize = 0xc1_1000;
    let a = attachment(A);
    let b = attachment(B);
    let mut shared = super::Shared::default();
    assert!(shared.update(A, &request(1), Some(&[1; 4])));
    let retired = super::attachment::unregister(A).unwrap();
    assert!(shared.update(B, &request(2), Some(&[2; 4])));
    assert!(!shared.detach(&retired));
    assert!(std::sync::Arc::ptr_eq(shared.owner.as_ref().unwrap(), &b));
    let new_a = attachment(A);
    assert!(shared.update(A, &request(3), Some(&[3; 4])));
    assert!(!shared.detach(&a));
    assert!(std::sync::Arc::ptr_eq(
        shared.owner.as_ref().unwrap(),
        &new_a
    ));
    assert_eq!(shared.hash, 3);
    super::attachment::unregister(A);
    super::attachment::unregister(B);
}

#[test]
fn admission_after_unregister_fails_and_hardware_takeover_clears_the_sprite() {
    use mtld3d_shared::mtl::CursorOverlayFlags;
    const A: usize = 0xc2_0000;
    const B: usize = 0xc2_1000;
    let a = attachment(A);
    attachment(B);
    let mut shared = super::Shared::default();
    assert!(shared.update(A, &request(1), Some(&[1; 4])));
    let mut hardware = request(0);
    hardware.flags |= CursorOverlayFlags::HARDWARE;
    assert!(shared.update(B, &hardware, None));
    let snapshot = shared.snapshot();
    assert!(snapshot.flags.contains(CursorOverlayFlags::HARDWARE));
    assert!(snapshot.sprite.is_none());
    assert_eq!(snapshot.hash, 0);
    assert!(!shared.detach(&a));
    // An unchanged software hash must retake ownership after a hardware takeover.
    assert!(shared.update(A, &request(1), None));
    assert!(shared.snapshot().sprite.is_some());
    super::attachment::unregister(A);
    assert!(!shared.update(A, &request(1), None));
    assert!(shared.detach(&a));
    super::attachment::unregister(B);
}

#[test]
fn rejected_upload_preserves_the_previous_request_and_reentrant_apply_stays_pending() {
    const A: usize = 0xc3_0000;
    attachment(A);
    let mut shared = super::Shared::default();
    assert!(shared.update(A, &request(1), Some(&[1; 4])));
    let first = shared.snapshot();
    assert!(!shared.update(A, &request(2), None));
    assert_eq!(shared.hash, 1);
    assert!(shared.update(A, &request(2), Some(&[2; 4])));
    shared.applied(first.revision, true);
    assert!(
        shared.pending,
        "an old apply cannot settle a reentrant update"
    );
    shared.applied(shared.revision, false);
    assert!(
        shared.pending,
        "creation or drawing failures retain the request"
    );
    shared.applied(shared.revision, true);
    assert!(!shared.pending);
    super::attachment::unregister(A);
}

#[test]
fn identical_requests_preserve_retries_without_resubmitting_completed_state() {
    const A: usize = 0xc4_0000;
    const B: usize = 0xc4_1000;
    attachment(A);
    attachment(B);
    let mut shared = super::Shared::default();
    assert!(shared.update(A, &request(1), Some(&[1; 4])));
    let first = shared.revision;
    assert!(shared.update(A, &request(1), None));
    assert!(shared.pending, "the first apply still needs to complete");
    assert_eq!(shared.revision, first);
    shared.applied(first, false);
    assert!(shared.update(A, &request(1), None));
    assert!(shared.pending, "an unchanged request retries failed work");
    shared.applied(first, true);
    assert!(shared.update(A, &request(1), None));
    assert!(
        !shared.pending,
        "completed identical state needs no dispatch"
    );
    let mut hidden = request(1);
    hidden
        .flags
        .remove(mtld3d_shared::mtl::CursorOverlayFlags::VISIBLE);
    let epoch = shared.capture_epoch;
    assert!(shared.update(A, &hidden, None));
    assert!(shared.update(A, &request(1), None));
    assert!(
        shared.capture_epoch > epoch,
        "coalesced hides remain observable"
    );
    assert!(shared.pending);
    let revision = shared.revision;
    assert!(shared.update(B, &request(1), None));
    assert!(
        shared.revision > revision,
        "the same hash can have a new owner"
    );
    super::attachment::unregister(A);
    super::attachment::unregister(B);
}

#[test]
fn hardware_only_visibility_needs_no_overlay_but_software_handoff_does() {
    use mtld3d_shared::mtl::CursorOverlayFlags;
    const A: usize = 0xc5_0000;
    let owner = attachment(A);
    let mut shared = super::Shared::default();
    let mut hardware = request(0);
    hardware.flags |= CursorOverlayFlags::HARDWARE;
    assert!(shared.update(A, &hardware, None));
    assert!(!shared.pending);
    hardware.flags.remove(CursorOverlayFlags::VISIBLE);
    let epoch = shared.capture_epoch;
    assert!(shared.update(A, &hardware, None));
    assert!(shared.capture_epoch > epoch);
    assert!(!shared.pending);
    assert!(std::sync::Arc::ptr_eq(
        shared.owner.as_ref().unwrap(),
        &owner
    ));
    assert!(shared.update(A, &request(1), Some(&[1; 4])));
    shared.applied(shared.revision, false); // The old software draw may need clearing.
    assert!(shared.update(A, &hardware, None));
    assert!(
        shared.pending,
        "hardware takeover must clear the software overlay"
    );
    assert!(shared.update(A, &hardware, None));
    assert!(
        shared.pending,
        "unchanged hardware state retries a failed clear"
    );
    super::attachment::unregister(A);
}

#[test]
fn native_hide_without_a_sprite_wakes_main_and_retires_with_its_device() {
    use mtld3d_shared::mtl::CursorOverlayFlags;
    const A: usize = 0xc6_0000;
    let owner = attachment(A);
    let mut shared = super::Shared::default();
    let mut native = request(0);
    native.flags = CursorOverlayFlags::HARDWARE | CursorOverlayFlags::NATIVE_HIDDEN;
    assert!(shared.update(A, &native, None));
    assert!(
        shared.pending,
        "a native hide must wake main without an overlay"
    );
    let snapshot = shared.snapshot();
    assert!(snapshot.sprite.is_none());
    assert!(snapshot.flags.contains(CursorOverlayFlags::NATIVE_HIDDEN));
    shared.applied(snapshot.revision, true);
    assert!(!shared.pending);
    native.flags = CursorOverlayFlags::HARDWARE;
    assert!(shared.update(A, &native, None));
    assert!(shared.pending, "show must release the native blank on main");
    assert!(shared.detach(&owner));
    assert!(shared.snapshot().flags.is_empty());
    assert!(shared.pending, "detach must reconcile the retired owner");
    super::attachment::unregister(A);
}

fn content(hash: u64) -> super::Content {
    super::Content::Sprite {
        hash,
        mode: super::LayerMode::Sdr,
        peak: 1.0,
        geometry: geometry(),
    }
}

#[test]
fn allocation_and_encoding_failures_retry_only_on_the_next_opportunity() {
    let mut state = super::ContentState::default();
    for stage in [
        "texture",
        "drawable",
        "command buffer",
        "pipeline",
        "encoder",
    ] {
        let mut calls = 0;
        assert!(
            !state.ensure(content(1), |_, _, _| {
                calls += 1;
                false
            }),
            "{stage}"
        );
        assert_eq!(calls, 1, "no immediate retry loop at {stage}");
        assert!(state.current().is_none());
    }
    assert!(state.ensure(content(1), |_, _, result| {
        result.store(super::COMPLETED, std::sync::atomic::Ordering::Release);
        true
    }));
    assert!(state.completed());
}

#[test]
fn completion_failure_retries_and_stale_callbacks_cannot_invalidate_a_new_submission() {
    use std::sync::atomic::Ordering;
    let mut state = super::ContentState::default();
    let mut old_result = None;
    assert!(state.ensure(content(1), |_, _, result| {
        old_result = Some(result);
        true
    }));
    assert!(!state.completed());
    let old_result = old_result.unwrap();
    old_result.store(super::FAILED, Ordering::Release);
    assert!(state.current().is_none());
    assert!(state.ensure(content(1), |_, _, result| {
        result.store(super::COMPLETED, Ordering::Release);
        true
    }));
    old_result.store(super::FAILED, Ordering::Release);
    assert!(state.completed());
    state.invalidate(); // Owner or colorspace handoff, even with the same sprite hash.
    assert!(state.ensure(content(1), |_, _, result| {
        result.store(super::COMPLETED, Ordering::Release);
        true
    }));
    old_result.store(super::FAILED, Ordering::Release);
    assert_eq!(state.current(), Some(&content(1)));
}

#[test]
fn sprite_geometry_invalidates_content_even_if_hash_and_color_mode_match() {
    let mut state = super::ContentState::default();
    assert!(state.ensure(content(1), |_, _, _| true));
    let changed = || super::Content::Sprite {
        hash: 1,
        mode: super::LayerMode::Sdr,
        peak: 1.0,
        geometry: SpriteGeometry {
            width: 64.0,
            scale: 1.0,
            ..geometry()
        },
    };
    let mut draws = 0;
    assert!(state.ensure(changed(), |_, _, _| {
        draws += 1;
        true
    }));
    assert_eq!(draws, 1);
    assert!(state.ensure(changed(), |_, _, _| {
        draws += 1;
        true
    }));
    assert_eq!(draws, 1, "matching submitted content is reused");
    assert!(!state.ensure(super::Content::Transparent, |_, _, _| false));
    assert!(
        state.current().is_none(),
        "a refused clear never becomes cached transparency"
    );
}

#[test]
fn same_mode_color_handoffs_compare_profile_format_and_edr_independently() {
    use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3, kCGColorSpaceSRGB};
    use objc2_metal::MTLPixelFormat;

    // SAFETY: immutable CoreGraphics names supplied by the framework.
    let srgb_name = unsafe { kCGColorSpaceSRGB };
    // SAFETY: immutable CoreGraphics name supplied by the framework.
    let p3_name = unsafe { kCGColorSpaceDisplayP3 };
    let srgb = CGColorSpace::with_name(Some(srgb_name)).unwrap();
    let p3 = CGColorSpace::with_name(Some(p3_name)).unwrap();
    let current = super::LayerConfiguration {
        format: MTLPixelFormat::BGRA8Unorm,
        colorspace: Some(&srgb),
        edr: false,
    };
    let mut next = super::LayerConfiguration {
        format: MTLPixelFormat::BGRA8Unorm,
        colorspace: Some(&p3),
        edr: false,
    };
    assert_ne!(current, next, "two SDR profiles still require invalidation");
    next.colorspace = Some(&srgb);
    assert_eq!(current, next);
    next.format = MTLPixelFormat::RGBA16Float;
    assert_ne!(current, next);
    next.format = current.format;
    next.edr = true;
    assert_ne!(current, next);
    next.edr = false;
    next.colorspace = None;
    assert_ne!(current, next);
}
