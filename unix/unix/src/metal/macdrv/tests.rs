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
//! replaced. They are the whole of the display-follow decision, so the tests
//! below pin both directions of a display change, the user's off switch, the
//! degenerate ceilings, and the case that must *not* reconfigure.

use super::{LayerMode, PresentPacing, layer_mode_change, layer_mode_for, min_present_duration};

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
