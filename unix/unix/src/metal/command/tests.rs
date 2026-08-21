//! Unit tests for present routing and the geometry settle filter.
//!
//! `present_route` is checked across the geometries a present can produce: equal extents
//! copy, an enlargement in both axes reaches `MetalFX` only if it exists, and anything
//! else falls to the stretch shader. The settle tests pin the other half of the decision,
//! that a scaler is spent on a geometry that held still rather than on a ratio, which is
//! what keeps a window drag from building one scaler per frame.

use super::{PresentGeometry, PresentRoute, SETTLED_PRESENTS, geometry_settled, present_route};

/// Matching extents take the blit whether or not `MetalFX` exists.
#[test]
fn equal_extents_route_to_the_blit() {
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), true),
        PresentRoute::Copy
    );
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), false),
        PresentRoute::Copy
    );
}

/// A larger drawable is `MetalFX`'s job, and the shader's without it.
#[test]
fn enlargement_routes_to_metalfx_when_present() {
    assert_eq!(
        present_route((1280, 720), (1920, 1080), true),
        PresentRoute::Upscale
    );
    assert_eq!(
        present_route((1280, 720), (1920, 1080), false),
        PresentRoute::Stretch
    );
}

/// The scaler only enlarges, so a smaller drawable is always the shader's.
#[test]
fn minification_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 1080), (1280, 720), true),
        PresentRoute::Stretch
    );
}

/// A drawable larger in one axis and smaller in the other is a stretch.
///
/// `MTLFXSpatialScaler` rejects the pair, and the blit would leave the
/// axis where the drawable is larger unwritten.
#[test]
fn mixed_axis_change_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 720), (1280, 1080), true),
        PresentRoute::Stretch
    );
    assert_eq!(
        present_route((1280, 1080), (1920, 720), true),
        PresentRoute::Stretch
    );
}

/// One axis equal and the other larger still enlarges.
#[test]
fn single_axis_enlargement_routes_to_metalfx() {
    assert_eq!(
        present_route((1920, 1080), (1920, 1200), true),
        PresentRoute::Upscale
    );
}

/// Every `render.scale` the config accepts reaches the quality path.
///
/// The knob's range is `(0, 1.0]`, and the smallest enlargement a user can
/// ask for deliberately (`0.99`, a ratio of 1.0098) sits *inside* the band
/// a live window resize produces, which is why routing does not judge an
/// enlargement by its ratio. The back-buffer dimension mirrors
/// `RenderScale::dimension` in `mtld3d-core`, which lives in the other
/// workspace and is not a dependency here.
#[test]
fn every_render_scale_setting_routes_to_metalfx() {
    let dimension = |logical: usize, percent: usize| (logical * percent).div_ceil(100).max(1);
    for percent in 1..100 {
        let src = (dimension(2560, percent), dimension(1600, percent));
        assert_eq!(
            present_route(src, (2560, 1600), true),
            PresentRoute::Upscale,
            "render.scale = {percent}% must reach MetalFX"
        );
    }
}

/// A resize drag is filtered by never settling, not by its ratio.
///
/// These are geometries measured off a live drag. Each is a legitimate
/// `Upscale` on geometry alone; what keeps them off the scaler is that
/// the next present carries a different pair.
#[test]
fn a_resize_drag_is_filtered_by_settling_not_by_geometry() {
    let drag = [
        ((2452, 1532), (2454, 1534)),
        ((2474, 1546), (2484, 1552)),
        ((2400, 1498), (2408, 1504)),
    ];
    let mut seen = None;
    for (src, dst) in drag {
        assert_eq!(present_route(src, dst, true), PresentRoute::Upscale);
        assert!(
            !geometry_settled(&mut seen, PresentGeometry { src, dst }),
            "{src:?} → {dst:?} lasted one present and must not build a scaler"
        );
    }
}

/// A geometry settles only after holding still for consecutive presents.
#[test]
fn geometry_settles_after_holding_still() {
    let steady = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1080),
    };
    let mut seen = None;
    let settled: Vec<bool> = (0..=SETTLED_PRESENTS)
        .map(|_| geometry_settled(&mut seen, steady))
        .collect();
    let expected: Vec<bool> = (1..=SETTLED_PRESENTS + 1)
        .map(|n| n >= SETTLED_PRESENTS)
        .collect();
    assert_eq!(settled, expected);
}

/// A window being dragged larger never settles, so it never builds a scaler.
///
/// Each frame of the drag is a different enlargement, which is exactly the
/// case that would otherwise leak one `MTLFXSpatialScaler` per frame.
#[test]
fn a_geometry_that_changes_every_present_never_settles() {
    let mut seen = None;
    for height in 0..64 {
        let dragging = PresentGeometry {
            src: (960, 540),
            dst: (1920, 1080 + height),
        };
        assert!(
            !geometry_settled(&mut seen, dragging),
            "a geometry seen once must not count as settled"
        );
    }
}

/// Settling restarts from scratch after the geometry changes.
#[test]
fn a_changed_geometry_restarts_the_count() {
    let before = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1080),
    };
    let after = PresentGeometry {
        src: (960, 540),
        dst: (1920, 1200),
    };
    let mut seen = None;
    for _ in 0..SETTLED_PRESENTS * 2 {
        geometry_settled(&mut seen, before);
    }
    assert!(!geometry_settled(&mut seen, after));
    assert_eq!(
        geometry_settled(&mut seen, before),
        SETTLED_PRESENTS <= 2,
        "returning to a geometry starts its count over, it does not resume"
    );
}
