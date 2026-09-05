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
//!
//! The buffer/texture pair, `copy_buffer_to_texture_reject` and its readback mirror, is
//! checked on both of its bounds: the region against the addressed mip level, rounded up
//! to the block grid so a compressed level below one block still takes a whole block, and
//! the buffer against the rows and slices the strides walk.
//!
//! `first_pending` is the lookup behind a GPU-retire wait: it is checked to answer with the
//! smallest registered seq at or past the target on the waiting device, and never with
//! another device's entry, however the seqs of the two interleave.

use std::collections::BTreeMap;

use mtld3d_shared::mtl::{BlockLayout, PixelFormat};
use objc2_metal::MTLPixelFormat;

use super::{
    CopyBufferEndpoint, CopyEndpoint, CopyRegion, CopyRejectReason, PresentGeometry, PresentRoute,
    SETTLED_PRESENTS, copy_buffer_to_texture_reject, copy_texture_reject,
    copy_texture_to_buffer_reject, first_pending, geometry_settled, present_route,
};

/// Two device identities that sort either side of each other's seqs.
const DEVICE_A: u64 = 0x1000;
const DEVICE_B: u64 = 0x2000;

/// A wait answers with its own device's smallest seq at or past the target.
///
/// Device B holds the seq A's wait would find first in a registry keyed by
/// seq alone.
#[test]
fn a_wait_answers_with_its_own_devices_next_seq() {
    let map: BTreeMap<(u64, u64), u32> = [
        ((DEVICE_A, 5), 15),
        ((DEVICE_B, 5), 25),
        ((DEVICE_B, 6), 26),
        ((DEVICE_A, 7), 17),
    ]
    .into_iter()
    .collect();
    assert_eq!(first_pending(&map, DEVICE_A, 5), Some(&15));
    assert_eq!(first_pending(&map, DEVICE_A, 6), Some(&17));
    assert_eq!(first_pending(&map, DEVICE_A, 7), Some(&17));
    assert_eq!(first_pending(&map, DEVICE_A, 8), None);
    assert_eq!(first_pending(&map, DEVICE_B, 6), Some(&26));
}

/// Another device's entries never answer a wait, whatever seqs it holds.
#[test]
fn another_devices_entries_never_answer_a_wait() {
    let map: BTreeMap<(u64, u64), u32> = (1..=10).map(|seq| ((DEVICE_B, seq), 20)).collect();
    assert_eq!(first_pending(&map, DEVICE_A, 1), None);
    assert_eq!(first_pending(&map, DEVICE_A, 0), None);
    assert_eq!(first_pending(&map, DEVICE_B, 3), Some(&20));
    assert_eq!(first_pending(&map, DEVICE_B, 11), None);
}

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
        depth: 1,
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
        CopyRejectReason::SourceBufferTooShort,
        CopyRejectReason::DestinationBufferTooShort,
    ];
    let mut keys: Vec<u64> = reasons.iter().map(|r| r.key()).collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), reasons.len());
    assert!(reasons.iter().all(|r| !r.as_str().is_empty()));
}

/// A `BGRA8` staging buffer holding `rows` tightly packed rows of `size` pixels.
fn upload_buffer(size: usize, rows: usize) -> CopyBufferEndpoint {
    CopyBufferEndpoint {
        length: size * 4 * rows,
        offset: 0,
        bytes_per_row: size * 4,
        bytes_per_image: size * 4 * rows,
    }
}

/// A single-slice region of `w` by `h` pixels.
const fn region(w: usize, h: usize) -> CopyRegion {
    CopyRegion {
        width: w,
        height: h,
        depth: 1,
    }
}

/// The block layout every `BGRA8` upload in these tests is measured in.
fn bgra8_block() -> BlockLayout {
    PixelFormat::Bgra8Unorm.block_layout()
}

/// A whole-level upload out of a buffer that holds exactly the level.
#[test]
fn a_matching_upload_is_accepted() {
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &endpoint(256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
}

/// An overhanging destination is rejected before Metal sees it.
///
/// This is the shape Metal reports as a destination origin plus a source width
/// exceeding the level's width.
#[test]
fn an_upload_overhanging_the_level_is_rejected() {
    let mut dst = endpoint(256);
    dst.origin_x = 128;
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &dst,
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
    // Half the width still fits at that origin.
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &dst,
            &region(128, 256),
            bgra8_block(),
        ),
        None
    );
}

/// Bounds are the addressed level's extent, not the base level's.
#[test]
fn an_upload_is_bounded_by_the_addressed_level() {
    let mut dst = endpoint(256);
    dst.levels = 9;
    dst.level = 2;
    assert_eq!(
        copy_buffer_to_texture_reject(&upload_buffer(64, 64), &dst, &region(64, 64), bgra8_block(),),
        None
    );
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(128, 128),
            &dst,
            &region(65, 64),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// A level past the end of the chain has no extent to upload into.
#[test]
fn an_upload_to_a_missing_level_is_rejected() {
    let mut dst = endpoint(256);
    dst.level = 1;
    assert_eq!(
        copy_buffer_to_texture_reject(&upload_buffer(1, 1), &dst, &region(1, 1), bgra8_block(),),
        Some(CopyRejectReason::DestinationLevelMissing)
    );
}

/// One byte short of the rows the strides walk is a reject.
#[test]
fn a_short_source_buffer_is_rejected() {
    assert_eq!(
        copy_buffer_to_texture_reject(
            &upload_buffer(256, 256),
            &endpoint(256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
    let mut short = upload_buffer(256, 256);
    short.length -= 1;
    assert_eq!(
        copy_buffer_to_texture_reject(&short, &endpoint(256), &region(256, 256), bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// The copy stops at the last row's own pixels, not at the end of its stride.
///
/// A sub-rect at the right edge of the last row of a staging buffer starts that
/// row part-way in, so charging it a whole stride would reject an upload whose
/// bytes the buffer holds.
#[test]
fn the_last_row_is_bounded_by_its_pixels_not_its_stride() {
    let mut src = upload_buffer(256, 256);
    src.offset = 255 * 256 * 4 + 128 * 4;
    let mut dst = endpoint(512);
    dst.origin_x = 128;
    dst.origin_y = 255;
    assert_eq!(
        copy_buffer_to_texture_reject(&src, &dst, &region(128, 1), bgra8_block()),
        None
    );
    // One pixel more than the 512 bytes left in the buffer.
    assert_eq!(
        copy_buffer_to_texture_reject(&src, &dst, &region(129, 1), bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// Compressed rows are counted in blocks, so a `BC1` level needs an eighth of the bytes.
#[test]
fn a_compressed_upload_counts_block_rows() {
    let block = PixelFormat::Bc1Rgba.block_layout();
    let mut dst = endpoint(128);
    dst.pixel_format = MTLPixelFormat::BC1_RGBA;
    // 32 block rows of 32 blocks, 8 bytes each.
    let exact = CopyBufferEndpoint {
        length: 256 * 32,
        offset: 0,
        bytes_per_row: 256,
        bytes_per_image: 256 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &dst, &region(128, 128), block),
        None
    );
    let short = CopyBufferEndpoint {
        length: 256 * 32 - 1,
        offset: 0,
        bytes_per_row: 256,
        bytes_per_image: 256 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&short, &dst, &region(128, 128), block),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
}

/// A compressed level below one block still addresses a whole block.
#[test]
fn a_compressed_level_under_one_block_takes_a_whole_block() {
    let block = PixelFormat::Bc1Rgba.block_layout();
    let mut dst = endpoint(4);
    dst.pixel_format = MTLPixelFormat::BC1_RGBA;
    dst.levels = 3;
    dst.level = 2;
    let one_block = CopyBufferEndpoint {
        length: 8,
        offset: 0,
        bytes_per_row: 8,
        bytes_per_image: 8,
    };
    // The level is one pixel; the copy names the 4x4 block that holds it.
    assert_eq!(
        copy_buffer_to_texture_reject(&one_block, &dst, &region(4, 4), block),
        None
    );
    // Two blocks wide is past the level however the extent is rounded.
    assert_eq!(
        copy_buffer_to_texture_reject(&one_block, &dst, &region(8, 4), block),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// A volume upload reads one slice stride per slice past the first.
#[test]
fn a_volume_upload_counts_every_slice() {
    let mut dst = endpoint(32);
    dst.depth = 4;
    let box_region = CopyRegion {
        width: 32,
        height: 32,
        depth: 4,
    };
    let exact = CopyBufferEndpoint {
        length: 32 * 4 * 32 * 4,
        offset: 0,
        bytes_per_row: 32 * 4,
        bytes_per_image: 32 * 4 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &dst, &box_region, bgra8_block()),
        None
    );
    let three_slices = CopyBufferEndpoint {
        length: 32 * 4 * 32 * 3,
        offset: 0,
        bytes_per_row: 32 * 4,
        bytes_per_image: 32 * 4 * 32,
    };
    assert_eq!(
        copy_buffer_to_texture_reject(&three_slices, &dst, &box_region, bgra8_block()),
        Some(CopyRejectReason::SourceBufferTooShort)
    );
    // The same box against a texture that has one slice.
    assert_eq!(
        copy_buffer_to_texture_reject(&exact, &endpoint(32), &box_region, bgra8_block()),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// The readback mirror reports the texture as the source and the buffer as the destination.
#[test]
fn a_readback_reports_the_ends_the_other_way_round() {
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(256),
            &upload_buffer(256, 256),
            &region(256, 256),
            bgra8_block(),
        ),
        None
    );
    // The caller asks for more pixels than the source holds, which is what a
    // declined resolve of a render-resolution frame leaves behind.
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(128),
            &upload_buffer(256, 256),
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    assert_eq!(
        copy_texture_to_buffer_reject(
            &endpoint(256),
            &upload_buffer(256, 255),
            &region(256, 256),
            bgra8_block(),
        ),
        Some(CopyRejectReason::DestinationBufferTooShort)
    );
}
