//! Unit tests for present routing, the geometry settle filter and the blit copy guard.
//!
//! `present_route` is checked across the geometries a present can produce: equal extents
//! copy, an enlargement in both axes reaches `MetalFX` only if it exists, and anything
//! else falls to the stretch shader. The settle tests pin the other half of the decision,
//! that a scaler is spent on a geometry that held still rather than on a ratio, which is
//! what keeps a window drag from building one scaler per frame.
//!
//! `copy_texture_reject` is checked against each condition Metal validates on
//! `copyFromTexture:`, plus the pairs it accepts: an identical pair, a sub-rect inside a
//! mip level, and a linear format against its sRGB twin.

use objc2_metal::MTLPixelFormat;

use super::{
    CopyEndpoint, CopyRejectReason, PresentGeometry, PresentRoute, SETTLED_PRESENTS,
    copy_texture_reject, geometry_settled, present_route,
};

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

/// A square `BGRA8` texture with one mip level, copied from its origin.
fn endpoint(size: usize) -> CopyEndpoint {
    CopyEndpoint {
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        sample_count: 1,
        width: size,
        height: size,
        level: 0,
        levels: 1,
        origin_x: 0,
        origin_y: 0,
    }
}

/// A pair that agrees on everything copies.
#[test]
fn a_matching_pair_is_accepted() {
    assert_eq!(
        copy_texture_reject(&endpoint(256), &endpoint(256), 256, 256),
        None
    );
}

/// Differing sample counts are the case the RESZ resolve exists for.
#[test]
fn a_sample_count_change_is_rejected() {
    let mut src = endpoint(256);
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256),
        Some(CopyRejectReason::SampleCountMismatch)
    );
}

/// Two unrelated formats of the same size are still a reject.
#[test]
fn a_format_change_is_rejected() {
    let mut dst = endpoint(256);
    dst.pixel_format = MTLPixelFormat::RGBA8Unorm;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 256, 256),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// A linear format and its sRGB twin are two views of one base format.
#[test]
fn an_srgb_twin_is_accepted_in_either_direction() {
    let mut srgb = endpoint(256);
    srgb.pixel_format = MTLPixelFormat::BGRA8Unorm_sRGB;
    assert_eq!(copy_texture_reject(&endpoint(256), &srgb, 256, 256), None);
    assert_eq!(copy_texture_reject(&srgb, &endpoint(256), 256, 256), None);
}

/// The format check runs before the sample-count check, so it reports first.
#[test]
fn a_pair_that_differs_in_both_reports_the_format() {
    let mut src = endpoint(256);
    src.pixel_format = MTLPixelFormat::RGBA8Unorm;
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// The region is bounds-checked against the source as well as the destination.
#[test]
fn a_region_leaving_either_end_is_rejected() {
    assert_eq!(
        copy_texture_reject(&endpoint(128), &endpoint(256), 256, 256),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    assert_eq!(
        copy_texture_reject(&endpoint(256), &endpoint(128), 256, 256),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// The origin counts towards the bound, so a sub-rect can still overrun.
#[test]
fn an_offset_sub_rect_is_bounded_by_the_origin() {
    let mut src = endpoint(256);
    src.origin_x = 128;
    src.origin_y = 128;
    assert_eq!(copy_texture_reject(&src, &endpoint(256), 128, 128), None);
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 129, 128),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// Bounds are the addressed mip level's extent, not the base level's.
#[test]
fn the_bound_is_the_addressed_mip_level() {
    let mut src = endpoint(256);
    src.levels = 9;
    src.level = 2;
    let mut dst = endpoint(64);
    dst.levels = 7;
    assert_eq!(copy_texture_reject(&src, &dst, 64, 64), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 65, 64),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// A level past the end of either chain has no extent to copy through.
#[test]
fn a_missing_mip_level_is_rejected() {
    let mut src = endpoint(256);
    src.level = 1;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 1, 1),
        Some(CopyRejectReason::SourceLevelMissing)
    );
    let mut dst = endpoint(256);
    dst.level = 1;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 1, 1),
        Some(CopyRejectReason::DestinationLevelMissing)
    );
}

/// Each reason keys its own one-shot warn.
#[test]
fn every_reject_reason_has_a_distinct_key() {
    let reasons = [
        CopyRejectReason::FormatMismatch,
        CopyRejectReason::SampleCountMismatch,
        CopyRejectReason::SourceLevelMissing,
        CopyRejectReason::DestinationLevelMissing,
        CopyRejectReason::SourceRegionOutOfBounds,
        CopyRejectReason::DestinationRegionOutOfBounds,
    ];
    let mut keys: Vec<u64> = reasons.iter().map(|r| r.key()).collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), reasons.len());
    assert!(reasons.iter().all(|r| !r.as_str().is_empty()));
}
