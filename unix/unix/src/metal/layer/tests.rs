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
//! `panel_drawable_size` caps a drawable at the panel in a scaled display
//! mode. The cases pin a whole screen landing exactly on the panel, the
//! per-axis result of a window, the source-fit condition on both sides of
//! the cap, every input that leaves the layer's own size, and the extreme
//! inputs. `backing_pixels` takes the screen's frame to the layer's scale,
//! `scaled_down_to_panel` picks the magnification filter, and `ScreenPixels`
//! round-trips through its one word.

use super::{
    PresentPacing, ScreenPixels, backing_pixels, min_present_duration, pack_pacing,
    panel_drawable_size, scaled_down_to_panel, unpack_pacing,
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
fn a_fullscreen_layer_in_a_scaled_mode_lands_on_the_panel() {
    let size = panel_drawable_size(
        AIR_BACKING,
        Some(AIR_BACKING),
        Some(AIR_PANEL),
        Some((2048, 1332)),
    );
    assert_eq!(size, AIR_PANEL);
    // "More Space" on a 3456x2234 panel.
    let size = panel_drawable_size(
        (4112, 2658),
        Some((4112, 2658)),
        Some((3456, 2234)),
        Some((2048, 1324)),
    );
    assert_eq!(size, (3456, 2234));
}

#[test]
fn a_window_is_capped_per_axis() {
    // 1800 * 1664 / 1912 = 1566.53, rounded to nearest.
    let size = panel_drawable_size(
        (2940, 1800),
        Some(AIR_BACKING),
        Some(AIR_PANEL),
        Some((1470, 900)),
    );
    assert_eq!(size, (2560, 1567));
}

#[test]
fn the_source_has_to_fit_the_cap() {
    let cap = |source| {
        panel_drawable_size(
            AIR_BACKING,
            Some(AIR_BACKING),
            Some(AIR_PANEL),
            Some(source),
        )
    };
    assert_eq!(cap(AIR_PANEL), AIR_PANEL, "a source at the cap is capped");
    assert_eq!(
        cap((2561, 1000)),
        AIR_BACKING,
        "wider than the cap keeps the layer's size"
    );
    assert_eq!(
        cap((1000, 1665)),
        AIR_BACKING,
        "taller than the cap keeps the layer's size"
    );
    assert_eq!(
        cap(AIR_BACKING),
        AIR_BACKING,
        "a native-backing source keeps its 1:1 copy"
    );
}

#[test]
fn a_panel_at_or_above_the_backing_never_moves_the_drawable() {
    let source = Some((1024, 768));
    // An exact 2x mode: the backing store is the panel.
    let exact = (3456, 2234);
    assert_eq!(
        panel_drawable_size(exact, Some(exact), Some(exact), source),
        exact
    );
    // Retina off: the layer's pixels are points, below the panel.
    let points = (1470, 956);
    assert_eq!(
        panel_drawable_size(points, Some(points), Some(AIR_PANEL), source),
        points
    );
}

#[test]
fn an_unknown_input_keeps_the_layer_size() {
    let source = Some((1024, 768));
    assert_eq!(
        panel_drawable_size(AIR_BACKING, Some(AIR_BACKING), None, source),
        AIR_BACKING
    );
    assert_eq!(
        panel_drawable_size(AIR_BACKING, None, Some(AIR_PANEL), source),
        AIR_BACKING
    );
    assert_eq!(
        panel_drawable_size(AIR_BACKING, Some(AIR_BACKING), Some(AIR_PANEL), None),
        AIR_BACKING
    );
    assert_eq!(
        panel_drawable_size(AIR_BACKING, Some((0, 1912)), Some(AIR_PANEL), source),
        AIR_BACKING
    );
}

#[test]
fn a_layer_smaller_than_its_share_of_the_panel_is_left_alone() {
    // The cap of a 1x1 layer rounds to 1, never to 0, so nothing shrinks.
    assert_eq!(
        panel_drawable_size((1, 1), Some(AIR_BACKING), Some(AIR_PANEL), Some((1, 1))),
        (1, 1)
    );
    // A backing smaller than the panel would enlarge; it does not.
    let natural = (1200, 800);
    assert_eq!(
        panel_drawable_size(
            natural,
            Some((1470, 956)),
            Some(AIR_PANEL),
            Some((640, 480))
        ),
        natural
    );
}

#[test]
fn the_extreme_inputs_do_not_overflow() {
    let max = (u32::MAX, u32::MAX);
    let panel = (u32::MAX - 1, u32::MAX - 1);
    assert_eq!(
        panel_drawable_size(max, Some(max), Some(panel), Some((1, 1))),
        panel
    );
    assert_eq!(
        panel_drawable_size(max, Some((1, 1)), Some(max), Some((1, 1))),
        max
    );
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
fn the_screen_is_scaled_down_only_past_the_panel() {
    assert!(scaled_down_to_panel(Some(AIR_BACKING), Some(AIR_PANEL)));
    assert!(scaled_down_to_panel(Some((2561, 1000)), Some(AIR_PANEL)));
    assert!(
        !scaled_down_to_panel(Some(AIR_PANEL), Some(AIR_PANEL)),
        "an exact 2x mode"
    );
    assert!(
        !scaled_down_to_panel(Some(AIR_POINTS), Some(AIR_PANEL)),
        "a layer at scale 1"
    );
    assert!(!scaled_down_to_panel(None, Some(AIR_PANEL)));
    assert!(!scaled_down_to_panel(Some(AIR_BACKING), None));
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
