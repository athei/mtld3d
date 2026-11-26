//! Unit tests for the layer-mode decision.
//!
//! `layer_mode_for` and `layer_mode_change` decide which `CAMetalLayer`
//! configuration a screen asks for and whether the applied one has to be
//! replaced. `min_present_duration_change` and `backing_scale_change` are
//! their two neighbours in the same reconciliation. All four are the whole of
//! the display-follow decision, so the tests below pin both directions of a
//! display change, the user's off switch, the degenerate ceilings, and the
//! case that must *not* reconfigure. A Reset that flips the guest's
//! `PresentationInterval` reaches the throttle through the same
//! reconciliation with the panel unchanged, so both directions of that flip
//! and the user cap that holds through it are pinned there too.
//!
//! `window_change` decides whether a reconciliation re-points a record at
//! the window its view sits in now, which Wine changes when it reparents the
//! device window. The tests pin a move between windows, a view that left its
//! window, a view found in its first one, and the case that keeps the
//! occlusion the observer last wrote.
//!
//! The other half of that path, what a device's teardown retires and what
//! it leaves alone for the devices still attached, is the attachment
//! registry's, and lives in `attachment/tests.rs`.

use super::{
    LayerMode, backing_scale_change, backing_scale_from, layer_mode_change, layer_mode_for,
    min_present_duration_change, window_change,
};
use crate::metal::layer::{PresentPacing, min_present_duration};

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
fn a_reset_to_immediate_lifts_the_throttle() {
    let vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &vsync);
    assert_eq!(applied.to_bits(), (1.0_f64 / 60.0).to_bits());
    let immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let changed = min_present_duration_change(applied, 60.0, &immediate).expect("the pacing moved");
    assert_eq!(changed.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn a_reset_back_to_vsync_restores_the_panel_throttle() {
    let immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 0,
    };
    let applied = min_present_duration(60.0, &immediate);
    assert_eq!(applied.to_bits(), 0.0_f64.to_bits());
    let vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 0,
    };
    let changed = min_present_duration_change(applied, 60.0, &vsync).expect("the pacing moved");
    assert_eq!(changed.to_bits(), (1.0_f64 / 60.0).to_bits());
}

#[test]
fn a_user_cap_holds_the_throttle_through_a_vsync_flip() {
    // The cap is the lower rate whichever way the guest's interval goes, so
    // the duration the present site uses does not move in either direction.
    let capped_vsync = PresentPacing {
        vsync_requested: true,
        max_fps: 30,
    };
    let capped_immediate = PresentPacing {
        vsync_requested: false,
        max_fps: 30,
    };
    let applied = min_present_duration(60.0, &capped_vsync);
    assert_eq!(applied.to_bits(), (1.0_f64 / 30.0).to_bits());
    assert_eq!(
        min_present_duration_change(applied, 60.0, &capped_immediate),
        None
    );
    let applied = min_present_duration(60.0, &capped_immediate);
    assert_eq!(
        min_present_duration_change(applied, 60.0, &capped_vsync),
        None
    );
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

#[test]
fn a_view_moved_into_another_window_repoints_the_record() {
    assert_eq!(window_change(0xA000, 0xB000), Some(0xB000));
}

#[test]
fn a_view_that_left_its_window_clears_the_record() {
    assert_eq!(window_change(0xA000, 0), Some(0));
}

#[test]
fn a_view_found_in_its_first_window_points_the_record_at_it() {
    assert_eq!(window_change(0, 0xA000), Some(0xA000));
}

#[test]
fn a_view_still_in_its_window_keeps_the_observers_state() {
    assert_eq!(window_change(0xA000, 0xA000), None);
    assert_eq!(window_change(0, 0), None);
}
