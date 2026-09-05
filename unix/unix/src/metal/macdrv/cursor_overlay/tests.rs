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
    assert_eq!(sprite_origin((100.0, 200.0), geometry()), (96.0, 174.0));
}

#[test]
fn sprite_origin_with_a_zero_hotspot_hangs_the_sprite_below_the_pointer() {
    let geometry = SpriteGeometry {
        hotspot_x: 0.0,
        hotspot_y: 0.0,
        ..geometry()
    };
    assert_eq!(sprite_origin((100.0, 200.0), geometry), (100.0, 168.0));
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
