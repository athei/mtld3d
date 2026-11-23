//! Unit tests for the scaled display mode sizes.
//!
//! `scale_rounded` is pinned against the float quotient it replaces, on
//! both sides of a half, and at the extremes. `panel_drawable_size` caps a
//! drawable at the panel in a scaled display mode. The cases pin a whole
//! screen landing exactly on the panel, the per-axis result of a window, the
//! source-fit condition on both sides of the cap, every input that leaves the
//! layer's own size, and the extreme inputs. `scaled_down_to_panel` is the
//! condition both the magnification filter and `render.panelFit` key on.

use super::{panel_drawable_size, scale_rounded, scaled_down_to_panel};

/// A 1470x956 "looks like" mode on a 2560x1664 panel, in retina mode.
const AIR_BACKING: (u32, u32) = (2940, 1912);
/// The same frame at scale 1, in points.
const AIR_POINTS: (u32, u32) = (1470, 956);
const AIR_PANEL: (u32, u32) = (2560, 1664);

#[test]
fn scale_rounded_rounds_halves_up() {
    assert_eq!(scale_rounded(1800, 1664, 1912), 1567, "1566.53");
    assert_eq!(scale_rounded(3, 1, 2), 2, "1.5 rounds up");
    assert_eq!(scale_rounded(5, 1, 4), 1, "1.25 rounds down");
    assert_eq!(scale_rounded(7, 3, 4), 5, "5.25 rounds down");
    assert_eq!(scale_rounded(4112, 3456, 4112), 3456, "exact");
    assert_eq!(scale_rounded(0, 3, 4), 0);
}

#[test]
fn scale_rounded_matches_the_percentage_rule() {
    // `(size * percent + 50) / 100` is the rule a percentage scale used.
    for percent in 1..=100_u32 {
        for size in 0..=4096_u32 {
            let expected = (u64::from(size) * u64::from(percent) + 50) / 100;
            assert_eq!(
                scale_rounded(size, percent, 100),
                expected,
                "{size} at {percent}%"
            );
        }
    }
}

#[test]
fn scale_rounded_does_not_overflow() {
    assert_eq!(
        scale_rounded(u32::MAX, u32::MAX, u32::MAX),
        u64::from(u32::MAX)
    );
    assert_eq!(
        scale_rounded(u32::MAX, u32::MAX, 1),
        u64::from(u32::MAX) * u64::from(u32::MAX)
    );
    assert_eq!(scale_rounded(u32::MAX, 1, u32::MAX), 1);
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
    assert_eq!(
        panel_drawable_size(AIR_POINTS, Some(AIR_POINTS), Some(AIR_PANEL), source),
        AIR_POINTS
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
        panel_drawable_size(natural, Some(AIR_POINTS), Some(AIR_PANEL), Some((640, 480))),
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
