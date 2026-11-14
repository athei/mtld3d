//! Unit tests for the present pacing.
//!
//! `min_present_duration` folds the guest's vsync request and the user's
//! `present.maxFps` ceiling into the minimum seconds handed to
//! `presentDrawable:afterMinimumDuration:`. Bit-exact assertions pin every
//! combination of the two inputs: the lower of the two rates wins, an
//! unknown panel rate still honours the user cap, and only IMMEDIATE with
//! no cap resolves to `0.0` for an unthrottled free run.
//!
//! `pack_pacing` and `unpack_pacing` fold the same two inputs into the one
//! word an attachment record holds. The round trip is pinned for both vsync
//! states, with and without a cap, and for the widest cap.

use super::{PresentPacing, min_present_duration, pack_pacing, unpack_pacing};

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

/// `present.nativeDrawable`: an overhanging drawable shrinks to the panel, aspect kept.
#[test]
fn a_drawable_larger_than_the_panel_fits_it_and_nothing_else_moves() {
    use super::fit_drawable_to_panel as fit;
    // The built-in panel under a 1470x956 "looks like" mode, window fullscreen.
    assert_eq!(fit((2944, 1915), Some((2560, 1664))), (2558, 1664));
    assert_eq!(fit((2940, 1912), Some((2560, 1664))), (2559, 1664));
    // Fits already: a panel-native mode, a non-retina layer, a small window.
    assert_eq!(fit((2560, 1664), Some((2560, 1664))), (2560, 1664));
    assert_eq!(fit((1472, 957), Some((2560, 1664))), (1472, 957));
    // Overhanging one axis only still scales both.
    assert_eq!(fit((2600, 1000), Some((2560, 1664))), (2560, 985));
    // Unknown panel, or the key off, and degenerate sizes are left alone.
    assert_eq!(fit((2944, 1915), None), (2944, 1915));
    assert_eq!(fit((0, 1915), Some((2560, 1664))), (0, 1915));
}
