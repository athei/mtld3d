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
//!
//! `backing_pixels` takes the screen's frame to the layer's scale, and
//! `ScreenPixels` round-trips through its one word. The panel cap itself is
//! tested where it lives, in `mtld3d_shared::render_size`.

use super::{
    PresentPacing, ScreenPixels, backing_pixels, min_present_duration, pack_pacing, unpack_pacing,
};

/// A 1470x956 "looks like" mode on a 2560x1664 panel: its frame in points.
const AIR_POINTS: (u32, u32) = (1470, 956);
/// That frame in retina mode, at scale 2.
const AIR_BACKING: (u32, u32) = (2940, 1912);
const AIR_PANEL: (u32, u32) = (2560, 1664);

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

#[test]
fn the_backing_follows_the_layer_scale() {
    assert_eq!(backing_pixels(Some(AIR_POINTS), 2.0), Some(AIR_BACKING));
    // Retina off: the backing is in the layer's pixels, the points, so a
    // layer at scale 1 is never capped below its own size.
    assert_eq!(backing_pixels(Some(AIR_POINTS), 1.0), Some(AIR_POINTS));
    assert_eq!(backing_pixels(None, 2.0), None);
    assert_eq!(backing_pixels(Some(AIR_POINTS), 0.0), None);
}

#[test]
fn screen_pixels_survive_the_round_trip_through_one_word() {
    let back = ScreenPixels::unpack(
        ScreenPixels {
            panel: Some(AIR_PANEL),
            frame: Some(AIR_POINTS),
        }
        .pack(),
    );
    assert_eq!(
        (back.panel, back.frame),
        (Some(AIR_PANEL), Some(AIR_POINTS))
    );
    let back = ScreenPixels::unpack(ScreenPixels::UNKNOWN.pack());
    assert_eq!((back.panel, back.frame), (None, None));
    // A dimension past 16 bits is unknown rather than truncated.
    let back = ScreenPixels::unpack(
        ScreenPixels {
            panel: Some((70_000, 1664)),
            frame: Some(AIR_POINTS),
        }
        .pack(),
    );
    assert_eq!((back.panel, back.frame), (None, Some(AIR_POINTS)));
}
