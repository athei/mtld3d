//! Unit tests for the present-throttle duration and the layer-mode decision.
//!
//! `min_present_duration` folds the guest's vsync request and the user's
//! `present.maxFps` ceiling into the minimum seconds handed to
//! `presentDrawable:afterMinimumDuration:`. Bit-exact assertions pin every
//! combination of the two inputs: the lower of the two rates wins, an
//! unknown panel rate still honours the user cap, and only IMMEDIATE with
//! no cap resolves to `0.0` for an unthrottled free run.
//!
//! `layer_mode_for` and `layer_mode_change` decide which `CAMetalLayer`
//! configuration a screen asks for and whether the applied one has to be
//! replaced. `min_present_duration_change` and `backing_scale_change` are
//! their two neighbours in the same reconciliation. All four are the whole of
//! the display-follow decision, so the tests below pin both directions of a
//! display change, the user's off switch, the degenerate ceilings, and the
//! case that must *not* reconfigure.
//!
//! `detach_metal_layer` is the other half of that path: the reconciliation
//! runs from process-lifetime observers, so what stops it walking a released
//! view, or re-deriving against a display nothing is bound to, is that
//! teardown leaves nothing bound. One test covers it, because the record and
//! the derived state it clears are process-wide.

use core::sync::atomic::Ordering;

use mtld3d_shared::{MetalHandle, mtl_handle::NSViewKind};

use super::{
    BACKING_SCALE_SINK_PTR, CURRENT_BACKING_SCALE, CURRENT_HEADROOM_BITS, HDR_ACTIVE,
    LAST_LOGGED_HEADROOM_BITS, LayerMode, PRESENT_PACING_BITS, PresentPacing, WINDOW_OCCLUDED,
    backing_scale_change, backing_scale_from, detach_metal_layer, display_state_is_latched,
    is_bound_window, layer_mode_change, layer_mode_for, min_present_duration,
    min_present_duration_change, pack_pacing, unpack_pacing, with_bound_display,
};

#[test]
fn vsync_only_paces_at_panel_rate() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn cap_only_bounds_the_free_run() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 30,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 30.0).to_bits());
}

#[test]
fn lower_rate_wins_when_both_active() {
    let below_panel = PresentPacing {
        vsync_requested: true,
        max_fps: 60,
    };
    let d = min_present_duration(120.0, &below_panel);
    assert_eq!(d.to_bits(), (1.0_f64 / 60.0).to_bits());

    let above_panel = PresentPacing {
        vsync_requested: true,
        max_fps: 240,
    };
    let d = min_present_duration(120.0, &above_panel);
    assert_eq!(d.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn immediate_and_uncapped_free_runs() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let d = min_present_duration(120.0, &pacing);
    assert_eq!(d.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn unknown_panel_rate_still_honours_the_cap() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 60,
    };
    let d = min_present_duration(0.0, &pacing);
    assert_eq!(d.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn an_edr_panel_asks_for_the_hdr_layer() {
    assert_eq!(layer_mode_for(16.0, true), LayerMode::Hdr);
    assert_eq!(layer_mode_for(1.5, true), LayerMode::Hdr);
}

#[test]
fn an_sdr_panel_asks_for_the_sdr_layer() {
    assert_eq!(layer_mode_for(1.0, true), LayerMode::Sdr);
    assert_eq!(layer_mode_for(0.0, true), LayerMode::Sdr);
}

#[test]
fn the_user_switch_forces_sdr_on_any_panel() {
    assert_eq!(layer_mode_for(16.0, false), LayerMode::Sdr);
}

#[test]
fn degenerate_ceilings_are_sdr() {
    assert_eq!(layer_mode_for(f64::NAN, true), LayerMode::Sdr);
    assert_eq!(layer_mode_for(f64::INFINITY, true), LayerMode::Sdr);
}

#[test]
fn staying_on_one_display_never_reconfigures() {
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
    assert_eq!(layer_mode_change(LayerMode::Sdr, 1.0, true), None);
    // The user switch is off, so an EDR panel is already correctly on the
    // SDR layer and a poll must leave it there.
    assert_eq!(layer_mode_change(LayerMode::Sdr, 16.0, false), None);
}

#[test]
fn moving_onto_an_sdr_display_asks_for_the_sdr_layer() {
    assert_eq!(
        layer_mode_change(LayerMode::Hdr, 1.0, true),
        Some(LayerMode::Sdr)
    );
}

#[test]
fn moving_onto_an_edr_display_asks_for_the_hdr_layer() {
    assert_eq!(
        layer_mode_change(LayerMode::Sdr, 16.0, true),
        Some(LayerMode::Hdr)
    );
}

#[test]
fn a_brightness_change_alone_does_not_reconfigure() {
    // The decision reads the static panel ceiling, which brightness and
    // thermal state do not move; the live headroom they do move drives the
    // present shader's peak instead. Both readings of the same XDR panel
    // therefore keep the HDR layer.
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
    assert_eq!(layer_mode_change(LayerMode::Hdr, 16.0, true), None);
}

#[test]
fn teardown_unbinds_the_display_and_resets_what_it_derived() {
    const VIEW: usize = 0x1000;
    const LAYER: usize = 0x2000;
    const WINDOW: usize = 0x3000;

    with_bound_display(|bound| {
        bound.view = VIEW;
        bound.layer = LAYER;
        bound.window = WINDOW;
    });
    HDR_ACTIVE.store(true, Ordering::Relaxed);
    CURRENT_HEADROOM_BITS.store(4.0_f32.to_bits(), Ordering::Relaxed);
    LAST_LOGGED_HEADROOM_BITS.store(4.0_f32.to_bits(), Ordering::Relaxed);
    WINDOW_OCCLUDED.store(true, Ordering::Relaxed);
    PRESENT_PACING_BITS.store(
        pack_pacing(&PresentPacing {
            vsync_requested: true,
            max_fps: 60,
        }),
        Ordering::Relaxed,
    );
    CURRENT_BACKING_SCALE.store(2, Ordering::Relaxed);
    BACKING_SCALE_SINK_PTR.store(0x4000, Ordering::Relaxed);
    assert!(
        display_state_is_latched(),
        "the reconciliation has a display to re-derive against"
    );
    assert!(
        is_bound_window(WINDOW),
        "the bound window matches while attached"
    );
    assert!(!is_bound_window(WINDOW + 8), "another window never matches");

    // SAFETY: `MetalHandle::new` asks for a retained object's address or `0`;
    // these addresses are only ever compared here, never dereferenced.
    let other_view = unsafe { MetalHandle::<NSViewKind>::new(VIEW as u64 + 8) };
    detach_metal_layer(other_view);
    with_bound_display(|bound| {
        assert_eq!(
            bound.view, VIEW,
            "another device's teardown leaves the bound view alone"
        );
    });

    // SAFETY: as above.
    let view = unsafe { MetalHandle::<NSViewKind>::new(VIEW as u64) };
    detach_metal_layer(view);

    with_bound_display(|bound| {
        assert_eq!(bound.view, 0, "the view the headroom refresh walks");
        assert_eq!(
            bound.layer, 0,
            "the layer the display-follow path reconfigures"
        );
        assert_eq!(
            bound.window, 0,
            "the window the occlusion observer filters by"
        );
    });
    assert!(
        !is_bound_window(WINDOW),
        "the released window no longer matches"
    );
    assert!(!is_bound_window(0), "nothing bound matches nothing");
    assert!(
        !HDR_ACTIVE.load(Ordering::Relaxed),
        "no layer carries an HDR configuration"
    );
    assert_eq!(
        CURRENT_HEADROOM_BITS.load(Ordering::Relaxed),
        1.0_f32.to_bits(),
        "the headroom the present pass treats as the identity curve",
    );
    assert_eq!(
        LAST_LOGGED_HEADROOM_BITS.load(Ordering::Relaxed),
        0,
        "the next session logs its own headroom baseline",
    );
    assert!(
        !WINDOW_OCCLUDED.load(Ordering::Relaxed),
        "no window suppresses a present"
    );
    assert_eq!(
        PRESENT_PACING_BITS.load(Ordering::Relaxed),
        0,
        "the next attach latches the pacing its own guest asked for",
    );
    assert_eq!(
        CURRENT_BACKING_SCALE.load(Ordering::Relaxed),
        0,
        "no display's backing scale is published",
    );
    assert_eq!(
        BACKING_SCALE_SINK_PTR.load(Ordering::Relaxed),
        0,
        "nothing is written into a guest that may have unloaded d3d9.dll",
    );
    assert!(
        !display_state_is_latched(),
        "the reconciliation has nothing to re-derive against"
    );
}

#[test]
fn pacing_survives_the_round_trip_through_one_word() {
    for (vsync_requested, max_fps) in [(true, 0), (false, 0), (true, 60), (false, 240)] {
        let packed = pack_pacing(&PresentPacing {
            vsync_requested,
            max_fps,
        });
        let back = unpack_pacing(packed);
        assert_eq!(back.vsync_requested, vsync_requested);
        assert_eq!(back.max_fps, max_fps);
    }
}

#[test]
fn the_widest_cap_survives_the_round_trip() {
    let packed = pack_pacing(&PresentPacing {
        vsync_requested: true,
        max_fps: u32::MAX,
    });
    let back = unpack_pacing(packed);
    assert!(back.vsync_requested);
    assert_eq!(back.max_fps, u32::MAX);
}

#[test]
fn staying_on_one_panel_never_rederives_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(min_present_duration_change(applied, 120.0, &pacing), None);
}

#[test]
fn moving_onto_a_slower_panel_lengthens_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    let changed = min_present_duration_change(applied, 60.0, &pacing).expect("panel rate moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn moving_onto_a_faster_panel_shortens_the_throttle() {
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &pacing);
    let changed = min_present_duration_change(applied, 120.0, &pacing).expect("panel rate moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 120.0).to_bits());
}

#[test]
fn a_user_cap_below_both_panels_holds_the_throttle_still() {
    // The cap is the lower rate on either display, so the duration the
    // present site uses does not move and nothing is rewritten or logged.
    let pacing = PresentPacing {
        vsync_requested: true,
        max_fps: 30,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(min_present_duration_change(applied, 60.0, &pacing), None);
}

#[test]
fn a_free_running_session_stays_unthrottled_on_any_panel() {
    let pacing = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let applied = min_present_duration(120.0, &pacing);
    assert_eq!(applied.to_bits(), 0.0_f64.to_bits());
    assert_eq!(min_present_duration_change(applied, 60.0, &pacing), None);
}

#[test]
fn backing_scale_rounds_and_clamps_into_the_hcursor_range() {
    assert_eq!(backing_scale_from(1.0), 1);
    assert_eq!(backing_scale_from(2.0), 2);
    assert_eq!(backing_scale_from(1.4), 1);
    assert_eq!(backing_scale_from(1.5), 2);
    // No screen at all, and a pathological reading, both land on identity
    // rather than a factor the HCURSOR builder would reject.
    assert_eq!(backing_scale_from(0.0), 1);
    assert_eq!(backing_scale_from(-4.0), 1);
    assert_eq!(backing_scale_from(f64::NAN), 1);
    assert_eq!(backing_scale_from(99.0), 8);
}

#[test]
fn staying_on_one_display_never_republishes_the_scale() {
    assert_eq!(backing_scale_change(2, 2.0), None);
    assert_eq!(backing_scale_change(1, 1.0), None);
}

#[test]
fn moving_between_displays_of_different_scale_republishes() {
    assert_eq!(backing_scale_change(2, 1.0), Some(1));
    assert_eq!(backing_scale_change(1, 2.0), Some(2));
}
