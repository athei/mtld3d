//! Unit tests for present routing, the geometry settle filter and the blit copy guard.
//!
//! `present_route` is checked across the geometries a present can produce: an enlargement
//! in both axes reaches `MetalFX` only if it exists, and anything else, equal extents
//! included, falls to the present shader. The settle tests pin the other half of the
//! decision, that a scaler is spent on a geometry that held still rather than on a ratio,
//! which is what keeps a window drag from building one scaler per frame.
//!
//! The present passes run on the GPU into Private, render-target-only destinations shaped
//! like a framebuffer-only drawable. The SDR pass without a ramp, and the exact copy the
//! `MetalFX` route ends with, at matching odd extents must give back the source byte for
//! byte. A ramp pass that cannot be encoded must leave the plain copy on the destination,
//! and the HDR pass-through must write the linear values of the source into an
//! `RGBA16Float` destination. The HDR tone map is checked against a float64 port of
//! its stage, against the pass-through at a headroom of 1, and for never putting a grey
//! below SDR while rising with the code and the headroom. Where `MetalFX` exists, the
//! upscaled HDR present is checked for keeping white at the headroom, for adding no
//! steps that go down to a bright gradient over what SDR scaling gives, and for equalling
//! the 1:1 tone map of the `Perceptual` scale with and without a gamma ramp.
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

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use mtld3d_shared::{
    ExtraColorDesc, MetalHandle, PassDescriptor,
    mtl::{BlockLayout, LoadAction, PixelFormat, PresentDebugFlags, StoreAction},
};
use objc2::{
    Message as _,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::{NSDictionary, NSError, NSLocalizedDescriptionKey, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLOrigin, MTLPixelFormat,
    MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_metal_fx::MTLFXSpatialScalerColorProcessingMode;

use super::{
    CopyBufferEndpoint, CopyEndpoint, CopyRegion, CopyRejectReason, DeviceRecord, EncodeContext,
    PendingCmdBuf, PresentGeometry, PresentRoute, SETTLED_PRESENTS, command_buffer_error,
    commit_registered, copy_buffer_to_texture_reject, copy_texture_reject,
    copy_texture_to_buffer_reject, encode_upload_cmd_buf, first_pending, geometry_settled,
    install_frame_handler, present_route, publish_idle_upload, readback_completed, retire_finished,
    submit_frame, submit_frame_with, wait_for_gpu_retire,
};
use crate::metal::{
    depth_transfer::PlanePool,
    submission::{FrameSubmission, RetirementCounter, SubmissionOutcome, SubmitDescription},
    transient::{SubmitStamp, UploadRing},
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
    assert_eq!(
        first_pending(&map, DEVICE_A, 5).map(|(_, value)| value),
        Some(&15)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 6).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 7).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_A, 8).map(|(_, value)| value),
        Some(&17)
    );
    assert_eq!(
        first_pending(&map, DEVICE_B, 6).map(|(_, value)| value),
        Some(&26)
    );
}

/// Another device's entries never answer a wait, whatever seqs it holds.
#[test]
fn another_devices_entries_never_answer_a_wait() {
    let map: BTreeMap<(u64, u64), u32> = (1..=10).map(|seq| ((DEVICE_B, seq), 20)).collect();
    assert_eq!(first_pending(&map, DEVICE_A, 1), None);
    assert_eq!(first_pending(&map, DEVICE_A, 0), None);
    assert_eq!(
        first_pending(&map, DEVICE_B, 3).map(|(_, value)| value),
        Some(&20)
    );
    assert_eq!(
        first_pending(&map, DEVICE_B, 11).map(|(_, value)| value),
        Some(&20)
    );
}

/// Matching extents take the shader's exact copy whether or not `MetalFX` exists.
///
/// The drawable is framebuffer-only, so no blit can write it.
#[test]
fn equal_extents_route_to_the_shader() {
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), true),
        PresentRoute::Shader
    );
    assert_eq!(
        present_route((1920, 1080), (1920, 1080), false),
        PresentRoute::Shader
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
        PresentRoute::Shader
    );
}

/// The scaler only enlarges, so a smaller drawable is always the shader's.
#[test]
fn minification_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 1080), (1280, 720), true),
        PresentRoute::Shader
    );
}

/// A drawable larger in one axis and smaller in the other is a stretch.
///
/// `MTLFXSpatialScaler` rejects the pair; the shader covers every pixel of
/// the drawable whatever the ratio.
#[test]
fn mixed_axis_change_routes_to_the_shader() {
    assert_eq!(
        present_route((1920, 720), (1280, 1080), true),
        PresentRoute::Shader
    );
    assert_eq!(
        present_route((1280, 1080), (1920, 720), true),
        PresentRoute::Shader
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
    let dimension = |logical: usize, percent: usize| ((logical * percent + 50) / 100).max(1);
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
        copy_texture_reject(&endpoint(256), &endpoint(256), 256, 256, 1),
        None
    );
}

/// Differing sample counts are the case the RESZ resolve exists for.
#[test]
fn a_sample_count_change_is_rejected() {
    let mut src = endpoint(256);
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::SampleCountMismatch)
    );
}

/// Two unrelated formats of the same size are still a reject.
#[test]
fn a_format_change_is_rejected() {
    let mut dst = endpoint(256);
    dst.pixel_format = MTLPixelFormat::RGBA8Unorm;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 256, 256, 1),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// A linear format and its sRGB twin are two views of one base format.
#[test]
fn an_srgb_twin_is_accepted_in_either_direction() {
    let mut srgb = endpoint(256);
    srgb.pixel_format = MTLPixelFormat::BGRA8Unorm_sRGB;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &srgb, 256, 256, 1),
        None
    );
    assert_eq!(
        copy_texture_reject(&srgb, &endpoint(256), 256, 256, 1),
        None
    );
}

/// The format check runs before the sample-count check, so it reports first.
#[test]
fn a_pair_that_differs_in_both_reports_the_format() {
    let mut src = endpoint(256);
    src.pixel_format = MTLPixelFormat::RGBA8Unorm;
    src.sample_count = 4;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::FormatMismatch)
    );
}

/// The region is bounds-checked against the source as well as the destination.
#[test]
fn a_region_leaving_either_end_is_rejected() {
    assert_eq!(
        copy_texture_reject(&endpoint(128), &endpoint(256), 256, 256, 1),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    assert_eq!(
        copy_texture_reject(&endpoint(256), &endpoint(128), 256, 256, 1),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
}

/// The origin counts towards the bound, so a sub-rect can still overrun.
#[test]
fn an_offset_sub_rect_is_bounded_by_the_origin() {
    let mut src = endpoint(256);
    src.origin_x = 128;
    src.origin_y = 128;
    assert_eq!(copy_texture_reject(&src, &endpoint(256), 128, 128, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 129, 128, 1),
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
    assert_eq!(copy_texture_reject(&src, &dst, 64, 64, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 65, 64, 1),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// A level past the end of either chain has no extent to copy through.
#[test]
fn a_missing_mip_level_is_rejected() {
    let mut src = endpoint(256);
    src.level = 1;
    assert_eq!(
        copy_texture_reject(&src, &endpoint(256), 1, 1, 1),
        Some(CopyRejectReason::SourceLevelMissing)
    );
    let mut dst = endpoint(256);
    dst.level = 1;
    assert_eq!(
        copy_texture_reject(&endpoint(256), &dst, 1, 1, 1),
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

/// A missing final sequence still waits for earlier committed GPU work.
#[test]
fn missing_final_submit_waits_for_earlier_work() {
    let queue = test_queue();
    let event = queue.device().newSharedEvent().expect("shared event");
    let cb = queue.commandBuffer().expect("command buffer");
    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    let coherent = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let counter = atomic_address(&coherent);
    let record = test_record(&queue);
    record
        .pending()
        .lock()
        .insert((counter, 1), PendingCmdBuf(cb.clone()));
    cb.commit();
    let (done, watchdog) = release_event_after_wait(event);
    wait_for_gpu_retire(record.pending(), 2, counter, 0, atomic_address(&failed));
    let status_at_return = cb.status();
    let retired_at_return = coherent.load(Ordering::Acquire);
    let _ = done.send(());
    watchdog.join().unwrap();
    cb.waitUntilCompleted();
    record.pending().lock().remove(&(counter, 1));
    assert_eq!(status_at_return, MTLCommandBufferStatus::Completed);
    assert_eq!(
        retired_at_return, 1,
        "the missing sequence was not submitted"
    );
}

/// Both failure positions protect committed work and publish failure before retirement.
#[test]
fn cpu_submit_failure_drains_draw_and_upload_handlers() {
    cpu_submit_failure_drain(false);
}

#[test]
fn cpu_submit_failure_after_upload_commit_drains_handlers() {
    cpu_submit_failure_drain(true);
}

fn cpu_submit_failure_drain(upload_committed: bool) {
    let queue = test_queue();
    let event = queue.device().newSharedEvent().expect("shared event");
    let cb = queue.commandBuffer().expect("command buffer");
    if !upload_committed {
        cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    }
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let draw_counter = atomic_address(&coherent);
    let upload_counter = atomic_address(&upload);
    let handler_done = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&handler_done);
    let handler = block2::RcBlock::new(
        move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            handler_flag.store(true, Ordering::Release);
        },
    );
    // SAFETY: Metal retains the block, whose only capture owns its atomic.
    unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
    let record = test_record(&queue);
    record
        .pending()
        .lock()
        .insert((draw_counter, 1), PendingCmdBuf(cb.clone()));
    cb.commit();
    if upload_committed {
        cb.waitUntilCompleted();
    }
    // In the upload case only the upload is parked. Waiting for the older
    // draw alone cannot satisfy the lifetime assertion.
    let upload_gate = queue.commandBuffer().expect("upload gate");
    if upload_committed {
        upload_gate.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    }
    upload_gate.commit();
    let params = test_submit_params(&coherent, &upload, &failed);
    let texture = upload_test_texture(&queue);
    let upload_pass = upload_test_pass(&texture);
    let (done, watchdog) = release_event_after_wait(event);
    let mut committed_upload = None;
    let frame = FrameSubmission {
        description: &params,
        blits: &[],
        passes: &[],
    };
    let success = submit_frame_with(&record, &frame, |_| {
        if upload_committed {
            let upload_cb = encode_test_upload(
                &record,
                &queue,
                core::slice::from_ref(&upload_pass),
                &params,
            )
            .expect("an upload buffer");
            commit_registered(
                record.pending(),
                &upload_cb,
                params.upload_retirement.address(),
                params.submit_seq,
            );
            // The upload at seq 2 is parked behind its own gate. Register
            // a draw at the same seq to prove the counter identities do not collide.
            record
                .pending()
                .lock()
                .insert((draw_counter, 2), PendingCmdBuf(cb.clone()));
            let pending = record.pending().lock();
            assert!(pending.contains_key(&(draw_counter, 2)));
            committed_upload = pending.get(&(upload_counter, 2)).map(|cb| cb.0.clone());
            drop(pending);
            assert!(committed_upload.is_some());
        }
        false
    });
    let upload_status_at_return = committed_upload.as_ref().map(|cb| cb.status());
    let status_at_return = cb.status();
    let handler_at_return = handler_done.load(Ordering::Acquire);
    let counters_at_return = (
        coherent.load(Ordering::Acquire),
        upload.load(Ordering::Acquire),
        failed.load(Ordering::Acquire),
    );
    let _ = done.send(());
    watchdog.join().unwrap();
    cb.waitUntilCompleted();
    // Cleanup precedes assertions so the pre-fix reproduction frees no live sink.
    let pending_upload = record
        .pending()
        .lock()
        .get(&(upload_counter, 2))
        .map(|cb| cb.0.clone());
    if let Some(upload_cb) = pending_upload {
        upload_cb.waitUntilCompleted();
    }
    upload_gate.waitUntilCompleted();
    record
        .pending()
        .lock()
        .retain(|&(counter, _), _| counter != draw_counter && counter != upload_counter);
    assert!(!success.success);
    assert_eq!(status_at_return, MTLCommandBufferStatus::Completed);
    assert!(handler_at_return, "callback sinks must be unused on return");
    if upload_committed {
        assert_eq!(
            upload_status_at_return,
            Some(MTLCommandBufferStatus::Completed)
        );
    }
    assert_eq!(counters_at_return, (2, 2, 2));
}

/// Encode `passes` into an upload command buffer with a ring of the test's own.
fn encode_test_upload(
    record: &Arc<DeviceRecord>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    passes: &[PassDescriptor],
    params: &SubmitDescription,
) -> Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
    let device = queue.device();
    let mut ring = UploadRing::default();
    let mut planes = PlanePool::default();
    let mut ctx = EncodeContext {
        device: &device,
        stamp: SubmitStamp::new(params).upload(),
        ring: &mut ring,
        planes: &mut planes,
    };
    encode_upload_cmd_buf(
        record,
        queue,
        &[],
        passes,
        params,
        &mut SubmissionOutcome::new(),
        &mut ctx,
    )
}

fn test_queue() -> Retained<ProtocolObject<dyn MTLCommandQueue>> {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    device.newCommandQueue().expect("command queue")
}

fn atomic_address(value: &AtomicU64) -> u64 {
    core::ptr::from_ref(value) as u64
}

/// A record around the test's queue, as `create_command_queue` would build one.
///
/// Takes a retain of its own, which the record releases when it drops, so the
/// caller's `Retained` stays valid. No presenter thread: the submissions here
/// carry no present.
fn test_record(queue: &ProtocolObject<dyn MTLCommandQueue>) -> Arc<DeviceRecord> {
    let retained = queue.retain();
    // SAFETY: `Retained::into_raw` transfers this test's extra retain into the
    // handle, which the record's `Drop` releases.
    let handle = unsafe { MetalHandle::new(Retained::into_raw(retained) as u64) };
    DeviceRecord::new(handle, None, PresentDebugFlags::empty())
}

fn test_submit_params(
    coherent: &AtomicU64,
    upload: &AtomicU64,
    failed: &AtomicU64,
) -> SubmitDescription {
    SubmitDescription {
        upload_pass_count: 0,
        present_layer: MetalHandle::NULL,
        present_texture: MetalHandle::NULL,
        present_view: MetalHandle::NULL,
        submit_seq: 2,
        // SAFETY: each test drains its callbacks before these counters leave scope.
        draw_retirement: unsafe { RetirementCounter::from_address(atomic_address(coherent)) },
        // SAFETY: each test drains its callbacks before these counters leave scope.
        upload_retirement: unsafe { RetirementCounter::from_address(atomic_address(upload)) },
        // SAFETY: each test drains its callbacks before these counters leave scope.
        failed_submission: unsafe { RetirementCounter::from_address(atomic_address(failed)) },
    }
}

/// The watchdog also releases immediately when a broken wait returns early.
fn release_event_after_wait(
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> (mpsc::Sender<()>, thread::JoinHandle<()>) {
    let (done, waiting) = mpsc::channel();
    let watchdog = thread::spawn(move || {
        let _ = waiting.recv_timeout(Duration::from_millis(100));
        event.setSignaledValue(1);
    });
    (done, watchdog)
}

/// Volume depth shrinks per mip and is checked at both ends independently.
#[test]
fn texture_copy_depth_is_bounded_by_each_mip() {
    let mut src = endpoint(8);
    src.depth = 8;
    src.levels = 4;
    src.level = 1;
    let mut dst = endpoint(4);
    dst.depth = 4;
    dst.levels = 3;
    assert_eq!(copy_texture_reject(&src, &dst, 4, 4, 4), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 4, 4, 5),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
    dst.depth = 2;
    assert_eq!(
        copy_texture_reject(&src, &dst, 4, 4, 3),
        Some(CopyRejectReason::DestinationRegionOutOfBounds)
    );
    assert_eq!(copy_texture_reject(&src, &dst, 4, 4, 2), None);
    src.level = 3;
    dst.level = 2;
    assert_eq!(copy_texture_reject(&src, &dst, 1, 1, 1), None);
    assert_eq!(
        copy_texture_reject(&src, &dst, 1, 1, 2),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

/// Ordinary 2D and cube mip copies address one depth plane.
#[test]
fn flat_texture_copies_reject_multiple_depth_planes() {
    assert_eq!(
        copy_texture_reject(&endpoint(4), &endpoint(4), 4, 4, 1),
        None
    );
    assert_eq!(
        copy_texture_reject(&endpoint(4), &endpoint(4), 4, 4, 2),
        Some(CopyRejectReason::SourceRegionOutOfBounds)
    );
}

#[test]
fn upload_prefix_finishes_before_its_retirement_signal() {
    let queue = test_queue();
    let texture = upload_test_texture(&queue);
    let pass = upload_test_pass(&texture);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let params = test_submit_params(&coherent, &upload, &failed);
    let record = test_record(&queue);
    let upload_cb =
        encode_test_upload(&record, &queue, &[pass], &params).expect("an upload buffer");
    commit_registered(
        record.pending(),
        &upload_cb,
        params.upload_retirement.address(),
        params.submit_seq,
    );
    wait_for_gpu_retire(
        record.pending(),
        params.submit_seq,
        atomic_address(&upload),
        0,
        atomic_address(&failed),
    );
    assert_eq!(upload.load(Ordering::Acquire), params.submit_seq);
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    assert_eq!(failed.load(Ordering::Acquire), 0);
    assert_eq!(upload_test_pixel(&queue, &texture), [0, 0, 255, 255]);
}

#[test]
fn frame_submission_accepts_empty_no_upload_and_all_upload_prefixes() {
    for (pass_count, upload_count, separate_upload) in
        [(0, 0, true), (1, 0, true), (1, 1, true), (1, 1, false)]
    {
        let queue = test_queue();
        let texture = upload_test_texture(&queue);
        let pass = upload_test_pass(&texture);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let mut params = test_submit_params(&coherent, &upload, &failed);
        params.upload_pass_count = upload_count;
        if !separate_upload {
            params.upload_retirement = RetirementCounter::NONE;
        }
        let passes = if pass_count == 0 {
            &[][..]
        } else {
            core::slice::from_ref(&pass)
        };
        let frame = FrameSubmission {
            description: &params,
            blits: &[],
            passes,
        };
        let record = test_record(&queue);
        assert!(submit_frame(&record, &frame).success);
        wait_for_gpu_retire(
            record.pending(),
            params.submit_seq,
            atomic_address(&coherent),
            params.upload_retirement.address(),
            atomic_address(&failed),
        );
        assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(failed.load(Ordering::Acquire), 0);
        // A frame without an upload buffer is published on the upload counter
        // too, since none is in flight.
        assert_eq!(
            upload.load(Ordering::Acquire),
            if separate_upload {
                params.submit_seq
            } else {
                0
            }
        );
        if pass_count != 0 {
            assert_eq!(upload_test_pixel(&queue, &texture), [0, 0, 255, 255]);
        }
    }
}

#[test]
fn frame_submission_rejects_invalid_upload_prefix() {
    for (pass_count, upload_count) in [(0, 1), (1, 2)] {
        let queue = test_queue();
        let texture = upload_test_texture(&queue);
        let pass = upload_test_pass(&texture);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let mut params = test_submit_params(&coherent, &upload, &failed);
        params.upload_pass_count = upload_count;
        let frame = FrameSubmission {
            description: &params,
            blits: &[],
            passes: if pass_count == 0 {
                &[]
            } else {
                core::slice::from_ref(&pass)
            },
        };
        let record = test_record(&queue);
        let outcome = submit_frame(&record, &frame);
        assert!(!outcome.success);
        assert_eq!(outcome.drawable_wait_ns, 0);
        assert_eq!(outcome.present_wait_ns, 0);
        assert!(outcome.snapshot_flags.is_empty());
        assert_eq!(failed.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
        assert_eq!(upload.load(Ordering::Acquire), params.submit_seq);
    }
}

fn upload_test_texture(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: a 1x1 non-mipmapped BGRA8 texture is a valid 2D descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::BGRA8Unorm,
            1,
            1,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let texture = queue
        .device()
        .newTextureWithDescriptor(&desc)
        .expect("upload texture");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-upload-texture")));
    texture
}

fn upload_test_pass(texture: &ProtocolObject<dyn MTLTexture>) -> PassDescriptor {
    PassDescriptor {
        // SAFETY: the test owns this texture until the command buffer retires.
        color_texture: unsafe { MetalHandle::new(core::ptr::from_ref(texture) as u64) },
        color_resolve_texture: MetalHandle::NULL,
        depth_texture: MetalHandle::NULL,
        commands_ptr: 0,
        visibility_result_buffer: MetalHandle::NULL,
        leading_blits_ptr: 0,
        color_load_action: LoadAction::Clear,
        color_store_action: StoreAction::Store,
        clear_r: 1.0f32.to_bits(),
        clear_g: 0,
        clear_b: 0,
        clear_a: 1.0f32.to_bits(),
        depth_load_action: LoadAction::DontCare,
        depth_store_action: StoreAction::DontCare,
        depth_clear_value: 0,
        stencil_load_action: LoadAction::DontCare,
        stencil_store_action: StoreAction::DontCare,
        stencil_clear_value: 0,
        command_count: 0,
        leading_blits_count: 0,
        pass_flags: PassDescriptor::pack_flags(0, 0, 0),
        reserved: 0,
        extra_color: [ExtraColorDesc::NONE; 3],
    }
}

fn upload_test_pixel(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> [u8; 4] {
    let buffer = queue
        .device()
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("readback buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-upload-pixel")));
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-upload-readback")));
    let blit = cmd.blitCommandEncoder().expect("readback blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-upload-copy")));
    // SAFETY: the source is 1x1, the destination holds a 256-byte aligned
    // row, and both resources stay alive until the copy completes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: 1, height: 1, depth: 1 }, &buffer, 0, 256, 256,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the shared buffer holds the completed four-byte pixel copy.
    unsafe { buffer.contents().cast::<[u8; 4]>().read() }
}

/// Completion is success only for the successful terminal state.
#[test]
fn readback_completion_rejects_failed_and_unfinished_states() {
    for status in [
        MTLCommandBufferStatus::NotEnqueued,
        MTLCommandBufferStatus::Enqueued,
        MTLCommandBufferStatus::Committed,
        MTLCommandBufferStatus::Scheduled,
        MTLCommandBufferStatus::Error,
    ] {
        assert!(!readback_completed(status, || None), "{status:?}");
    }
    assert!(readback_completed(
        MTLCommandBufferStatus::Completed,
        || { panic!("a successful readback must not fetch error diagnostics") }
    ));
}

/// A real `NSError` crosses the same completion decision without any GPU submission.
#[test]
fn readback_completion_preserves_driver_error_details() {
    let description = "Caused GPU Hang Error (00000003:kIOAccelCommandBufferCallbackErrorHang)";
    let value = NSString::from_str(description);
    // SAFETY: Foundation exports this immutable NSString key.
    let key = unsafe { NSLocalizedDescriptionKey };
    let info = NSDictionary::from_slices(&[key], &[AsRef::<AnyObject>::as_ref(&*value)]);
    // SAFETY: the user-info dictionary contains an NSString under the description key.
    let error = unsafe {
        NSError::errorWithDomain_code_userInfo(
            &NSString::from_str("MTLCommandBufferErrorDomain"),
            2,
            Some(&info),
        )
    };
    assert_eq!(
        command_buffer_error(Some(&error)),
        (2, description.to_owned())
    );
    assert_eq!(command_buffer_error(None), (0, String::new()));
    let mut inspected = false;
    assert!(!readback_completed(MTLCommandBufferStatus::Error, || {
        inspected = true;
        Some(error)
    }));
    assert!(inspected, "a failed readback must inspect its driver error");
}

/// A tone map failing after the HDR upscale encoded still converts the original SDR frame.
#[test]
fn hdr_upscale_failure_reencodes_the_original_source() {
    use objc2_metal::{MTLClearColor, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction};
    let queue = test_queue();
    let device = queue.device();
    if !crate::metal::upscale::is_available(&device) {
        eprintln!("MetalFX unsupported, skipping HDR fallback regression");
        return;
    }
    let cache = crate::metal::upscale::UpscaleCache::new();
    let src =
        crate::metal::upscale::scratch_target(&device, &cache, 32, 32, PixelFormat::Bgra8Unorm)
            .expect("source");
    let dst =
        crate::metal::upscale::scratch_target(&device, &cache, 64, 64, PixelFormat::Rgba16Float)
            .expect("destination");
    let cmd = queue.commandBuffer().expect("reference command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-reference")));
    let pass = MTLRenderPassDescriptor::new();
    // SAFETY: attachment zero exists on every render pass descriptor.
    let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    color.setTexture(Some(&src));
    color.setLoadAction(MTLLoadAction::Clear);
    color.setStoreAction(MTLStoreAction::Store);
    color.setClearColor(MTLClearColor {
        red: 0.75,
        green: 0.5,
        blue: 0.25,
        alpha: 1.0,
    });
    let encoder = cmd
        .renderCommandEncoderWithDescriptor(&pass)
        .expect("clear encoder");
    encoder.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-source")));
    encoder.endEncoding();
    assert!(super::encode_hdr_present(&cmd, &src, &dst, 2.0, 0));
    cmd.commit();
    cmd.waitUntilCompleted();
    let expected = upload_test_pixel(&queue, &dst);
    assert_ne!(expected, [0; 4]);
    let cmd = queue.commandBuffer().expect("fallback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-fallback")));
    super::clear_drawable(&cmd, &dst);
    let invoked = std::cell::Cell::new(false);
    assert!(super::encode_hdr_present_upscaled_with(
        &cmd,
        &cache,
        &src,
        &dst,
        2.0,
        0,
        |_, _, _| {
            invoked.set(true);
            false
        }
    ));
    assert!(
        invoked.get(),
        "the failure must come after the scaler encoded"
    );
    crate::metal::upscale::retire_evicted(&cmd, &cache);
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(upload_test_pixel(&queue, &dst), expected);
    crate::metal::upscale::retire(&cache);
}

/// Invalid second-plane uploads must not publish the already encoded depth plane.
#[test]
fn depth_plane_failure_aborts_the_pair_and_retry_retains_sources() {
    use mtld3d_shared::{BlitCommand, BlitCommandType, CopyBufferToTextureInfo};
    use objc2_metal::{MTLBlitOption, MTLEvent};

    let queue = test_queue();
    let device = queue.device();
    // SAFETY: a 2x1 private combined depth texture has one valid mip.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::Depth32Float_Stencil8,
            2,
            1,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("depth destination");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-depth-pair")));
    let color = upload_test_texture(&queue);
    let readback = device
        .newBufferWithLength_options(512, MTLResourceOptions::StorageModeShared)
        .expect("plane readback");
    readback.setLabel(Some(&NSString::from_str("mtld3d-test-depth-readback")));
    let read = || {
        let cb = queue.commandBuffer().expect("read command buffer");
        cb.setLabel(Some(&NSString::from_str("mtld3d-test-depth-read")));
        let blit = cb.blitCommandEncoder().expect("read encoder");
        blit.setLabel(Some(&NSString::from_str("mtld3d-test-depth-read-planes")));
        for (offset, plane) in [
            (0, MTLBlitOption::DepthFromDepthStencil),
            (256, MTLBlitOption::StencilFromDepthStencil),
        ] {
            // SAFETY: both 2x1 planes fit their disjoint aligned rows, retained until completion.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                    &texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: 2, height: 1, depth: 1 },
                    &readback, offset, 256, 256, plane,
                );
            }
        }
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: completion made the shared depth row CPU-visible.
        let depths = unsafe { readback.contents().cast::<[f32; 2]>().read() };
        let masks = readback
            .contents()
            .cast::<u8>()
            .as_ptr()
            .wrapping_add(256)
            .cast::<[u8; 2]>();
        // SAFETY: the completed stencil copy occupies the second aligned row.
        let stencil = unsafe { masks.read() };
        (depths, stencil)
    };
    for (generation, depths, stencil) in
        [(0, [0.25f32, 0.75], [17u8, 239]), (1, [0.5, 1.0], [91, 75])]
    {
        let depth = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .expect("depth upload");
        depth.setLabel(Some(&NSString::from_str("mtld3d-test-depth-upload")));
        let masks = device
            .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
            .expect("stencil upload");
        masks.setLabel(Some(&NSString::from_str("mtld3d-test-stencil-upload")));
        // SAFETY: the fresh shared depth buffer holds two float values.
        unsafe { depth.contents().cast::<[f32; 2]>().write(depths) };
        // SAFETY: the fresh shared stencil buffer holds two stencil bytes.
        unsafe { masks.contents().cast::<[u8; 2]>().write(stencil) };
        let make = |buffer: &ProtocolObject<dyn MTLBuffer>, kind| {
            let mut cmd = BlitCommand::copy_buffer_to_texture(&CopyBufferToTextureInfo {
                buffer_handle: core::ptr::from_ref(buffer) as u64,
                buffer_offset: 0,
                bytes_per_row: 256,
                texture_handle: core::ptr::from_ref(&*texture) as u64,
                destination_slice: 0,
                mip_level: 0,
                origin_x: 0,
                origin_y: 0,
                region_w: 2,
                region_h: 1,
                depth: 1,
                bytes_per_image: 256,
            });
            cmd.cmd = kind as u32;
            cmd
        };
        let commands = [
            make(&depth, BlitCommandType::CopyBufferToDepth),
            make(&masks, BlitCommandType::CopyBufferToStencil),
        ];
        if generation != 0 {
            for bad in 0..6 {
                let mut rejected = commands;
                match bad {
                    0 => rejected[1].src_handle = 0,
                    1 => rejected[1].dst_handle = 0,
                    2 => rejected[1].mip_level = 1,
                    3 => rejected[1].src_offset = 256,
                    4 => rejected[1].region_w = 3,
                    _ => rejected[1].dst_handle = core::ptr::from_ref(&*color) as u64,
                }
                let coherent = AtomicU64::new(0);
                let upload = AtomicU64::new(0);
                let failed = AtomicU64::new(0);
                let params = test_submit_params(&coherent, &upload, &failed);
                let frame = FrameSubmission {
                    description: &params,
                    blits: &rejected,
                    passes: &[],
                };
                let record = test_record(&queue);
                assert!(
                    !submit_frame(&record, &frame).success,
                    "invalid plane case {bad}"
                );
                assert_eq!(failed.load(Ordering::Acquire), params.submit_seq);
                assert_eq!(coherent.load(Ordering::Acquire), params.submit_seq);
                assert_eq!(read(), ([0.25, 0.75], [17, 239]));
            }
        }
        let event = device.newSharedEvent().expect("completion gate");
        event.setLabel(Some(&NSString::from_str("mtld3d-test-depth-gate")));
        let cb = queue.commandBuffer().expect("upload command buffer");
        cb.setLabel(Some(&NSString::from_str("mtld3d-test-depth-pair-upload")));
        cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
        let coherent = AtomicU64::new(0);
        let upload = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let params = test_submit_params(&coherent, &upload, &failed);
        let mut ring = UploadRing::default();
        let mut planes = PlanePool::default();
        let mut ctx = EncodeContext {
            device: &device,
            stamp: SubmitStamp::new(&params),
            ring: &mut ring,
            planes: &mut planes,
        };
        assert!(super::encode_leading_blits(
            &cb,
            &commands,
            super::BlitSite::FrameLeading,
            &mut ctx,
        ));
        drop(depth);
        drop(masks);
        cb.commit();
        event.setSignaledValue(1);
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read(), (depths, stencil));
    }
}

/// A frame without uploads moves the upload counter only while no upload buffer is in flight.
#[test]
fn an_idle_upload_counter_follows_frames_that_upload_nothing() {
    let queue = test_queue();
    let record = test_record(&queue);
    let upload = AtomicU64::new(1);
    let counter = atomic_address(&upload);
    publish_idle_upload(record.pending(), counter, 4);
    assert_eq!(upload.load(Ordering::Acquire), 4);
    let cb = queue.commandBuffer().expect("buffer");
    record
        .pending()
        .lock()
        .insert((counter, 5), PendingCmdBuf(cb));
    publish_idle_upload(record.pending(), counter, 6);
    assert_eq!(upload.load(Ordering::Acquire), 4);
    record.pending().lock().remove(&(counter, 5));
    publish_idle_upload(record.pending(), counter, 7);
    assert_eq!(upload.load(Ordering::Acquire), 7);
}

/// A buffer released uncommitted runs its handlers and still moves neither counter.
///
/// An ended buffer sits registered at seq 1 on both counters, so a handler
/// that went on to retire would publish it: only the early return keeps the
/// counters at 0.
#[test]
fn a_buffer_released_uncommitted_never_advances_a_counter() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let params = test_submit_params(&coherent, &upload, &failed);
    for counter in [
        params.draw_retirement.address(),
        params.upload_retirement.address(),
    ] {
        let ended = queue.commandBuffer().expect("ended");
        ended.commit();
        ended.waitUntilCompleted();
        record
            .pending()
            .lock()
            .insert((counter, 1), PendingCmdBuf(ended));
    }
    let texture = upload_test_texture(&queue);
    let upload_pass = upload_test_pass(&texture);
    objc2::rc::autoreleasepool(|_| {
        let frame = queue.commandBuffer().expect("frame");
        install_frame_handler(&frame, &record, &params);
        let upload_cb = encode_test_upload(
            &record,
            &queue,
            core::slice::from_ref(&upload_pass),
            &params,
        )
        .expect("an upload buffer");
        drop((frame, upload_cb));
    });
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    assert_eq!(upload.load(Ordering::Acquire), 0);
    assert_eq!(failed.load(Ordering::Acquire), 0);
    for counter in [
        params.draw_retirement.address(),
        params.upload_retirement.address(),
    ] {
        retire_finished(record.pending(), counter, 0, "test");
    }
    assert_eq!(coherent.load(Ordering::Acquire), 1);
    assert_eq!(upload.load(Ordering::Acquire), 1);
}

/// A counter moves over ended buffers from the oldest and stops at the first still running.
///
/// The later buffer ends first: the counter must not name it while the
/// earlier one still runs, and names both once that one ends.
#[test]
fn a_counter_never_passes_a_buffer_still_running() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let counter = atomic_address(&coherent);
    let event = queue.device().newSharedEvent().expect("gate");
    let first = queue.commandBuffer().expect("first");
    first.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event), 1);
    first.commit();
    // A second queue, so the later buffer does not queue behind the parked one.
    let other = test_queue();
    let second = other.commandBuffer().expect("second");
    second.commit();
    second.waitUntilCompleted();
    {
        let mut map = record.pending().lock();
        map.insert((counter, 1), PendingCmdBuf(first.clone()));
        map.insert((counter, 2), PendingCmdBuf(second));
    }
    retire_finished(record.pending(), counter, 0, "test");
    assert_eq!(coherent.load(Ordering::Acquire), 0);
    event.setSignaledValue(1);
    first.waitUntilCompleted();
    retire_finished(record.pending(), counter, 0, "test");
    assert_eq!(coherent.load(Ordering::Acquire), 2);
    assert!(record.pending().lock().is_empty());
}

/// A wait publishes a frame without uploads that the submit-time publication had to skip.
///
/// The previous frame's upload buffer is still registered when a mid-frame
/// submission with no upload buffer commits, so that submission leaves the
/// upload counter behind. The retirement wait that follows proves every
/// upload buffer up to the draw it waited for, so the upload counter reaches
/// the same sequence and a gate on both counters does not stop at the older
/// upload.
#[test]
fn a_retirement_wait_publishes_uploads_through_the_draw_it_waited_for() {
    let queue = test_queue();
    let record = test_record(&queue);
    let coherent = AtomicU64::new(0);
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let (draw_counter, upload_counter) = (atomic_address(&coherent), atomic_address(&upload));
    for (counter, seq) in [(upload_counter, 1), (draw_counter, 1), (draw_counter, 2)] {
        let cb = queue.commandBuffer().expect("buffer");
        cb.commit();
        record
            .pending()
            .lock()
            .insert((counter, seq), PendingCmdBuf(cb));
    }
    publish_idle_upload(record.pending(), upload_counter, 2);
    assert_eq!(upload.load(Ordering::Acquire), 0);
    wait_for_gpu_retire(
        record.pending(),
        2,
        draw_counter,
        upload_counter,
        atomic_address(&failed),
    );
    assert_eq!(coherent.load(Ordering::Acquire), 2);
    assert_eq!(upload.load(Ordering::Acquire), 2);
    assert!(record.pending().lock().is_empty());
}

/// A submitted frame finishes for the latency bound when its draw buffer's handler retires it.
///
/// Drives the real path: the submission notes itself on the device's signal
/// before its draw buffer commits, and the completion handler wakes the
/// `Present` waiting for it.
#[test]
fn a_committed_frame_wakes_the_frame_latency_wait() {
    use std::{sync::mpsc, time::Duration};

    use crate::metal::FrameWait;

    let queue = test_queue();
    let record = test_record(&queue);
    let floor = AtomicU64::new(0);
    let coherent = Arc::new(AtomicU64::new(0));
    let upload = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    // SAFETY: `floor` outlives every use of the signal in this test.
    record
        .present()
        .frame_signal()
        .attach(unsafe { RetirementCounter::from_address(atomic_address(&floor)) });
    let (sender, receiver) = mpsc::channel();
    let waiter = {
        let signal = Arc::clone(record.present().frame_signal());
        let coherent = Arc::clone(&coherent);
        std::thread::spawn(move || {
            // SAFETY: the waiter owns a reference to the atomic it reads.
            let draw = unsafe { RetirementCounter::from_address(atomic_address(&coherent)) };
            let _ = sender.send(signal.wait(draw, 2, crate::metal::PACED_PRESENT_CAP));
        })
    };
    assert!(receiver.recv_timeout(Duration::from_millis(50)).is_err());
    let params = test_submit_params(&coherent, &upload, &failed);
    let frame = FrameSubmission {
        description: &params,
        blits: &[],
        passes: &[],
    };
    assert!(
        submit_frame(&record, &frame).success,
        "an empty frame submits"
    );
    assert_eq!(
        receiver.recv_timeout(Duration::from_secs(5)),
        Ok(FrameWait::Finished)
    );
    waiter.join().unwrap();
    assert_eq!(
        floor.load(Ordering::Acquire),
        2,
        "the floor follows the noted submission"
    );
    wait_for_gpu_retire(
        record.pending(),
        2,
        atomic_address(&coherent),
        atomic_address(&upload),
        atomic_address(&failed),
    );
}

/// Odd extents a present copies at, off any power-of-two texel grid.
const EXACT_EXTENTS: [(usize, usize); 2] = [(1366, 768), (1311, 847)];

/// The present shader's SDR pass without a ramp copies matching extents byte for byte.
///
/// At matching extents that pass is the whole present, into a drawable no
/// blit can write. A fragment that missed its texel centre would filter in a
/// neighbour, and random contents make every such blend visible.
#[test]
fn the_sdr_present_pass_copies_matching_extents_exactly() {
    let queue = test_queue();
    for (width, height) in EXACT_EXTENTS {
        let (src, expected) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
        let dst = drawable_shaped(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
        let cmd = queue.commandBuffer().expect("command buffer");
        cmd.setLabel(Some(&NSString::from_str("mtld3d-test-present-copy")));
        assert!(super::encode_present_copy(&cmd, &src, &dst, 0));
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
        assert_texels_equal(&texture_bytes(&queue, &dst), &expected, 4, width);
    }
}

/// The copy that carries a `MetalFX` output onto the drawable reproduces it in both formats.
///
/// The SDR drawable is `BGRA8Unorm` and the HDR one `RGBA16Float`, whose
/// random contents are finite normal halves, positive and negative, beyond
/// `1.0` included.
#[test]
fn the_exact_copy_reproduces_its_source_in_both_drawable_formats() {
    let queue = test_queue();
    for format in [MTLPixelFormat::BGRA8Unorm, MTLPixelFormat::RGBA16Float] {
        for (width, height) in EXACT_EXTENTS {
            let (src, expected) = random_source(&queue, format, width, height);
            let dst = drawable_shaped(&queue, format, width, height);
            let cmd = queue.commandBuffer().expect("command buffer");
            cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-copy")));
            assert!(super::encode_exact_copy(&cmd, &src, &dst), "{format:?}");
            cmd.commit();
            cmd.waitUntilCompleted();
            assert_eq!(
                cmd.status(),
                MTLCommandBufferStatus::Completed,
                "{:?}",
                cmd.error()
            );
            assert_texels_equal(
                &texture_bytes(&queue, &dst),
                &expected,
                bytes_per_texel(format),
                width,
            );
        }
    }
}

/// A ramp pass that cannot be encoded leaves the plain copy on the destination.
///
/// The ramp pipeline here is a null handle, so its pass is skipped the way a
/// pass whose gamma table went away is. The plain pass must then write the
/// render-target-only destination, which no blit could.
#[test]
fn a_ramp_pass_that_fails_falls_back_to_the_plain_copy() {
    let queue = test_queue();
    let (width, height) = EXACT_EXTENTS[1];
    let (src, expected) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
    let dst = drawable_shaped(&queue, MTLPixelFormat::BGRA8Unorm, width, height);
    let resources =
        crate::metal::present::ensure_resources(&queue.device()).expect("present pipelines");
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-ramp-fallback")));
    assert!(super::encode_present_pass(
        &cmd,
        &src,
        &dst,
        &super::PresentStage {
            plain: resources.copy,
            ramp: Some(0),
            uniforms: None,
            gamma_layer: 1,
        },
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    assert_texels_equal(&texture_bytes(&queue, &dst), &expected, 4, width);
}

/// The HDR pass-through writes the source's linear values into an `RGBA16Float` destination.
///
/// The destination is Private and render-target only, like the float
/// drawable of an EDR layer. One texel value covers the three branches of the
/// decode: zero, a mid-tone on the power curve, and full scale.
#[test]
fn the_hdr_passthrough_writes_linear_values_onto_a_float_destination() {
    let queue = test_queue();
    let (width, height) = EXACT_EXTENTS[1];
    // Blue, green, red, alpha of a `BGRA8Unorm` texel.
    let texel = [0_u8, 128, 255, 255];
    let src = uploaded(
        &queue,
        MTLPixelFormat::BGRA8Unorm,
        (width, height),
        &texel.repeat(width * height),
    );
    let dst = drawable_shaped(&queue, MTLPixelFormat::RGBA16Float, width, height);
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-passthrough")));
    assert!(super::encode_hdr_present(&cmd, &src, &dst, 1.0, 0));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    let mid_tone = ((128.0_f32 / 255.0 + 0.055) / 1.055).powf(2.4);
    let expected = [1.0, mid_tone, 0.0, 1.0];
    let bytes = texture_bytes(&queue, &dst);
    for (index, texel) in bytes.as_chunks::<8>().0.iter().enumerate() {
        let channels = texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| half_to_f32(u16::from_le_bytes(*half)));
        for (channel, (read, wanted)) in channels.zip(expected).enumerate() {
            assert!(
                (read - wanted).abs() <= 1.0e-3,
                "texel {index} channel {channel} reads {read}, the linear value is {wanted}"
            );
        }
    }
}

/// The tone map's output against a float64 port of the stage at reference points.
///
/// The port is the BT.2446 stage step for step in `f64`: the sRGB decode,
/// Method A at `L_HDR = min(max(headroom, 1) × 203, 10000)` with
/// `L_SDR = 100`, the smooth maximum in ln luminance of `Y × 203` and Method
/// A (`max(a, b) + k h² / 4`, `h = max(k − |a − b|, 0) / k`,
/// `k = min(0.12, L_HDR / 203 − 1)`), one PQ ratio on I, Ct and Cp, ST 2084,
/// and the output divided by 203. The sources are 16-bit sRGB codes: 30235 is
/// linear 0.18 (mid grey), 65535 white, (48192, 35199, 25465) linear
/// (0.5, 0.25, 0.125). White reaches the headroom up to PQ's 10000 nits,
/// 49.26 × SDR white, and a headroom of 60 or an infinite one stops there
/// with every channel finite, a saturated blue included.
#[test]
fn the_hdr_tone_map_matches_its_reference_values() {
    const GREY: [u16; 4] = [30235, 30235, 30235, 65535];
    const WHITE: [u16; 4] = [65535; 4];
    const COLOUR: [u16; 4] = [48192, 35199, 25465, 65535];
    const BLUE: [u16; 4] = [0, 0, 65535, 65535];
    let cases: [([u16; 4], f32, [f32; 3]); 14] = [
        (GREY, 2.0, [0.193_365; 3]),
        (GREY, 5.0, [0.378_151; 3]),
        (GREY, 15.0, [0.833_174; 3]),
        (GREY, 60.0, [1.916_529; 3]),
        (GREY, f32::INFINITY, [1.916_529; 3]),
        (WHITE, 1.5, [1.5; 3]),
        (WHITE, 5.0, [5.0; 3]),
        (WHITE, 15.0, [15.0; 3]),
        (WHITE, 60.0, [49.261_257; 3]),
        (COLOUR, 2.0, [0.564_657, 0.279_665, 0.138_375]),
        (COLOUR, 5.0, [1.216_523, 0.565_387, 0.260_549]),
        (COLOUR, 15.0, [3.011_8, 1.286_592, 0.538_12]),
        (BLUE, 60.0, [0.699_825, 0.0, 18.197_902]),
        (BLUE, f32::INFINITY, [0.699_825, 0.0, 18.197_902]),
    ];
    let queue = test_queue();
    for (texel, headroom, expected) in cases {
        let bytes: Vec<u8> = texel
            .iter()
            .flat_map(|channel| channel.to_le_bytes())
            .collect();
        let src = uploaded(
            &queue,
            MTLPixelFormat::RGBA16Unorm,
            (4, 4),
            &bytes.repeat(16),
        );
        let values = hdr_values(&queue, &src, |cmd, dst| {
            super::encode_hdr_present(cmd, &src, dst, headroom, 0)
        });
        for (channel, wanted) in expected.into_iter().enumerate() {
            let read = values[0][channel];
            assert!(
                (read - wanted).abs() <= wanted.mul_add(4.0e-3, 1.0e-4),
                "{texel:?} at a headroom of {headroom}: channel {channel} reads {read}, the \
                 reference is {wanted}"
            );
        }
    }
}

/// At a headroom of 1 the BT.2446 stage gives the pass-through's picture.
///
/// The present takes the pass-through stage up to a headroom of 1 and the
/// BT.2446 stage above it, so the two have to agree there for the switch
/// to make no step. Every 8-bit grey code and a row of colours run through
/// both; they agree within the rounding of a half.
#[test]
fn the_hdr_tone_map_at_a_headroom_of_one_is_the_sdr_picture() {
    let queue = test_queue();
    // Row 0 is every grey code, row 1 a colour per code: BGRA bytes.
    let greys = (0..=255_u8).flat_map(|code| [code, code, code, 255]);
    let colours = (0..=255_u8).flat_map(|code| [code / 2, 255 - code, code, 255]);
    let bytes: Vec<u8> = greys.chain(colours).collect();
    let src = uploaded(&queue, MTLPixelFormat::BGRA8Unorm, (256, 2), &bytes);
    let resources = crate::metal::present::ensure_resources(&queue.device()).expect("pipelines");
    let passthrough = hdr_values(&queue, &src, |cmd, dst| {
        super::encode_hdr_present(cmd, &src, dst, 1.0, 0)
    });
    let tone_mapped = hdr_values(&queue, &src, |cmd, dst| {
        super::encode_present_pass(
            cmd,
            &src,
            dst,
            &super::PresentStage {
                plain: resources.bt2446,
                ramp: None,
                uniforms: Some(crate::metal::present::hdr_uniforms(1.0)),
                gamma_layer: 0,
            },
        )
    });
    for (index, (sdr, hdr)) in passthrough.iter().zip(&tone_mapped).enumerate() {
        for (channel, (read, wanted)) in hdr.iter().zip(sdr).take(3).enumerate() {
            assert!(
                (read - wanted).abs() <= wanted.mul_add(2.0e-3, 1.0e-5),
                "texel {index} channel {channel}: BT.2446 at a headroom of 1 reads {read}, the \
                 pass-through {wanted}"
            );
        }
    }
}

/// The tone map never puts a grey below SDR and rises with both the code and the headroom.
///
/// Every 8-bit grey code, from a headroom of 1 (the pass-through) to 2 in
/// steps of 0.05, and at 3, 5, 15 and 60. No code reads below its SDR value
/// or below the code under it or below itself at the headroom before, and
/// over 1 to 2 no code rises by more than white does per step (0.05; the
/// float64 port's largest is 0.05005), so the switch from the pass-through
/// stage makes no step. The tolerances are the rounding of a half.
#[test]
fn the_hdr_tone_map_never_darkens_and_rises_with_code_and_headroom() {
    let queue = test_queue();
    let bytes: Vec<u8> = (0..=255_u8)
        .flat_map(|code| [code, code, code, 255])
        .collect();
    let src = uploaded(&queue, MTLPixelFormat::BGRA8Unorm, (256, 1), &bytes);
    let headrooms = (0..=20_u8)
        .map(|step| f32::from(step).mul_add(0.05, 1.0))
        .chain([3.0, 5.0, 15.0, 60.0]);
    let mut previous: Option<(f32, Vec<f32>)> = None;
    for headroom in headrooms {
        let values: Vec<f32> = hdr_values(&queue, &src, |cmd, dst| {
            super::encode_hdr_present(cmd, &src, dst, headroom, 0)
        })
        .iter()
        .map(|texel| texel[0])
        .collect();
        for (code, &value) in (0..=255_u8).zip(&values) {
            let encoded = f32::from(code) / 255.0;
            let sdr = if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            };
            assert!(
                value >= sdr.mul_add(-2.0e-3, sdr) - 1.0e-6,
                "code {code} at a headroom of {headroom} reads {value}, under its SDR {sdr}"
            );
        }
        for (code, pair) in values.windows(2).enumerate() {
            assert!(
                pair[1] >= pair[0].mul_add(-1.0e-3, pair[0]),
                "code {} at a headroom of {headroom} reads {}, under code {code}'s {}",
                code + 1,
                pair[1],
                pair[0]
            );
        }
        if let Some((last_headroom, last)) = &previous {
            for (code, (now, before)) in values.iter().zip(last).enumerate() {
                assert!(
                    *now >= before.mul_add(-1.0e-3, *before),
                    "code {code} reads {now} at a headroom of {headroom}, {before} at \
                     {last_headroom}"
                );
                if headroom <= 2.0 {
                    assert!(
                        now - before <= 0.053,
                        "code {code} rises from {before} to {now} between a headroom of \
                         {last_headroom} and {headroom}"
                    );
                }
            }
        }
        previous = Some((headroom, values));
    }
}

/// An upscaled HDR present keeps white at the headroom and a bright gradient's steps.
///
/// `MetalFX` enlarges an SDR frame 2x into an HDR drawable at headrooms of 4
/// and 15. Flat white must read within 0.5 % of the tone map at drawable
/// resolution without `MetalFX`, away from the outer [`WHITE_MARGIN`] texels,
/// where the scaler writes 254 for 255 in SDR as well. Codes 204 to 255 must
/// have no more steps that go down along a row than `Perceptual` scaling of
/// the same SDR frame has. The gradient's error against the tone map at
/// drawable resolution is printed beside the same figures for the order that
/// tone-maps at render size and scales the float result in `HDR` mode.
#[test]
fn an_upscaled_hdr_present_keeps_white_and_the_gradient_steps() {
    let queue = test_queue();
    let device = queue.device();
    if !crate::metal::upscale::is_available(&device) {
        eprintln!("MetalFX unsupported, skipping the upscaled HDR present measurement");
        return;
    }
    let cache = crate::metal::upscale::UpscaleCache::new();
    let source = (52 * GRADIENT_COLUMNS_PER_CODE, 8);
    let drawable = (source.0 * 2, source.1 * 2);
    let white = uploaded(
        &queue,
        MTLPixelFormat::BGRA8Unorm,
        source,
        &[255_u8; 4].repeat(source.0 * source.1),
    );
    let row: Vec<u8> = (204..=255_u8)
        .flat_map(|code| [code, code, code, 255].repeat(GRADIENT_COLUMNS_PER_CODE))
        .collect();
    let gradient = uploaded(
        &queue,
        MTLPixelFormat::BGRA8Unorm,
        source,
        &row.repeat(source.1),
    );
    let sdr_down = down_steps(&sdr_upscaled_row(&queue, &cache, &gradient, drawable));
    for headroom in [4.0_f32, 15.0] {
        let (white_wanted, gradient_wanted) =
            white_and_gradient(&queue, &white, &gradient, drawable, |cmd, src, dst| {
                super::encode_hdr_present(cmd, src, dst, headroom, 0)
            });
        let wanted = white_wanted[0][0];
        let (white_read, gradient_read) =
            white_and_gradient(&queue, &white, &gradient, drawable, |cmd, src, dst| {
                present_upscaled(cmd, &cache, src, dst, headroom)
            });
        let presented = UpscaleErrors::new(
            &white_read,
            wanted,
            drawable,
            &gradient_read,
            &gradient_wanted,
        );
        let (white_read, gradient_read) =
            white_and_gradient(&queue, &white, &gradient, drawable, |cmd, src, dst| {
                tone_mapped_then_scaled(cmd, &cache, src, dst, headroom)
            });
        let hdr_mode = UpscaleErrors::new(
            &white_read,
            wanted,
            drawable,
            &gradient_read,
            &gradient_wanted,
        );
        eprintln!(
            "headroom {headroom}, white {wanted}: present {presented}; tone map then HDR-mode \
             scale {hdr_mode}; SDR Perceptual {sdr_down} steps down"
        );
        assert!(
            presented.white_error <= 5.0e-3,
            "white at a headroom of {headroom}: {presented}"
        );
        assert!(
            presented.down <= sdr_down,
            "the gradient at a headroom of {headroom} has more steps down than SDR's \
             {sdr_down}: {presented}"
        );
    }
    crate::metal::upscale::retire(&cache);
}

/// The upscaled HDR present equals, texel for texel, the 1:1 tone map of the `Perceptual` scale.
///
/// A random frame, near-black, colours and alpha included, enlarged by 4:3
/// in each axis, at headrooms of 1, 4 and 15, without a gamma ramp and with
/// one: every texel of the present must equal `encode_hdr_present` at 1:1
/// from the same frame scaled by `upscale::encode` into a `BGRA8Unorm`
/// target. The ramp must change the picture, so its pass is the one that ran.
#[test]
fn the_upscaled_hdr_present_tone_maps_the_perceptual_scale_texel_for_texel() {
    use crate::metal::macdrv::attachment;
    let queue = test_queue();
    let device = queue.device();
    if !crate::metal::upscale::is_available(&device) {
        eprintln!("MetalFX unsupported, skipping the upscaled HDR present comparison");
        return;
    }
    let cache = crate::metal::upscale::UpscaleCache::new();
    let drawable = (400, 268);
    let (src, _) = random_source(&queue, MTLPixelFormat::BGRA8Unorm, 300, 201);
    let scaled = texture_2d(
        &device,
        MTLPixelFormat::BGRA8Unorm,
        drawable,
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
    );
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-perceptual-scale")));
    assert!(crate::metal::upscale::encode(
        &cmd,
        &device,
        &cache,
        &src,
        &scaled,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    // A ramp that moves every channel: red squared, green kept, blue inverted.
    let view = attachment::test_address_block();
    let layer = view + 8;
    attachment::register(
        view,
        layer,
        &attachment::AttachLatches {
            hwnd: 0x42,
            flags: attachment::AttachFlags::empty(),
            color_space: mtld3d_shared::mtl::ColorSpacePolicy::Passthrough,
            pacing_bits: 0,
            backing_scale: 1,
            backing_scale_sink: 0,
            cursor_kick_sink: 0,
            client_surface_request_sink: 0,
            screen_sizes_sink: 0,
            client_surface: 0,
        },
    );
    let ramp: Vec<u16> = (0..=255_u32)
        .flat_map(|code| {
            let level = |value: u32| u16::try_from(value).expect("a 16-bit level");
            [
                level(code * code * 65535 / (255 * 255)),
                level(code * 257),
                level((255 - code) * 257),
                u16::MAX,
            ]
        })
        .collect();
    let layer_key = u64::try_from(layer).expect("a 64-bit address");
    // SAFETY: `set_gamma_ramp` reads the handle only as the key of the
    // layer's table and never dereferences it; no object stands behind it.
    let layer_handle = unsafe { MetalHandle::new(layer_key) };
    assert!(crate::metal::gamma::set_gamma_ramp(
        layer_handle,
        Some(&ramp)
    ));
    for headroom in [1.0_f32, 4.0, 15.0] {
        let mut plain = None;
        for gamma_layer in [0, layer] {
            let presented = hdr_values_at(&queue, drawable, |cmd, dst| {
                super::encode_hdr_present_upscaled(cmd, &cache, &src, dst, headroom, gamma_layer)
            });
            let wanted = hdr_values_at(&queue, drawable, |cmd, dst| {
                super::encode_hdr_present(cmd, &scaled, dst, headroom, gamma_layer)
            });
            if let Some(index) = presented
                .iter()
                .zip(&wanted)
                .position(|(read, want)| read.map(f32::to_bits) != want.map(f32::to_bits))
            {
                panic!(
                    "texel ({}, {}) at a headroom of {headroom}, ramp {}: the present reads {:?}, \
                     the tone map of the Perceptual scale {:?}",
                    index % drawable.0,
                    index / drawable.0,
                    gamma_layer != 0,
                    presented[index],
                    wanted[index]
                );
            }
            match plain.take() {
                None => plain = Some(presented),
                Some(unramped) => assert!(
                    unramped
                        .iter()
                        .zip(&presented)
                        .any(|(before, after)| before.map(f32::to_bits) != after.map(f32::to_bits)),
                    "the ramp must change the picture at a headroom of {headroom}"
                ),
            }
        }
    }
    assert!(crate::metal::gamma::set_gamma_ramp(layer_handle, None));
    attachment::unregister(view);
    crate::metal::upscale::retire(&cache);
}

/// Source columns per code of the measured gradient.
///
/// Four, so that the 2x enlargement has a flat run between two codes. At one
/// column per code the scaler sharpens every step into a zigzag in SDR as
/// well, and a count of the steps that go down measures that instead.
const GRADIENT_COLUMNS_PER_CODE: usize = 4;

/// Texels at each edge of the drawable that the white check leaves out.
const WHITE_MARGIN: usize = 4;

/// What an upscaled HDR present reads against the tone map at drawable resolution.
struct UpscaleErrors {
    /// The largest relative error of a white texel inside [`WHITE_MARGIN`].
    white_error: f64,
    /// The gradient row's mean relative error.
    mean: f64,
    /// The gradient row's root-mean-square relative error.
    rms: f64,
    /// The gradient row's largest relative error.
    max: f64,
    /// Steps along the gradient row whose value goes down.
    down: usize,
}

impl UpscaleErrors {
    /// Compare a white frame against `wanted` and a gradient row against its reference.
    fn new(
        white: &[[f32; 4]],
        wanted: f32,
        (width, height): (usize, usize),
        gradient: &[f32],
        reference: &[f32],
    ) -> Self {
        let white_error = white
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                let (x, y) = (index % width, index / width);
                (WHITE_MARGIN..width - WHITE_MARGIN).contains(&x)
                    && (WHITE_MARGIN..height - WHITE_MARGIN).contains(&y)
            })
            .map(|(_, texel)| (f64::from(texel[0]) / f64::from(wanted) - 1.0).abs())
            .fold(0.0, f64::max);
        let errors: Vec<f64> = gradient
            .iter()
            .zip(reference)
            .map(|(read, wanted)| f64::from(*read) / f64::from(*wanted) - 1.0)
            .collect();
        let count = f64::from(u32::try_from(errors.len()).expect("one short row"));
        Self {
            white_error,
            mean: errors.iter().sum::<f64>() / count,
            rms: (errors.iter().map(|error| error * error).sum::<f64>() / count).sqrt(),
            max: errors.iter().fold(0.0, |max, error| error.abs().max(max)),
            down: down_steps(gradient),
        }
    }
}

impl core::fmt::Display for UpscaleErrors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "white {:.3} %, gradient mean {:+.3} % rms {:.3} % max {:.3} %, {} steps down",
            self.white_error * 100.0,
            self.mean * 100.0,
            self.rms * 100.0,
            self.max * 100.0,
            self.down
        )
    }
}

/// Run `encode` on the white and the gradient frame into HDR drawables of `extent`.
///
/// Returns every texel of the white frame and the gradient's middle row.
fn white_and_gradient(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    white: &ProtocolObject<dyn MTLTexture>,
    gradient: &ProtocolObject<dyn MTLTexture>,
    extent: (usize, usize),
    encode: impl Fn(
        &ProtocolObject<dyn MTLCommandBuffer>,
        &ProtocolObject<dyn MTLTexture>,
        &ProtocolObject<dyn MTLTexture>,
    ) -> bool,
) -> (Vec<[f32; 4]>, Vec<f32>) {
    let white = hdr_values_at(queue, extent, |cmd, dst| encode(cmd, white, dst));
    let gradient = hdr_values_at(queue, extent, |cmd, dst| encode(cmd, gradient, dst));
    (white, middle_row(&gradient, extent))
}

/// The present's upscaled HDR route, which must reach `MetalFX` rather than its fallback.
fn present_upscaled(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    cache: &crate::metal::upscale::UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    peak: f32,
) -> bool {
    let scaled = std::cell::Cell::new(false);
    let encoded = super::encode_hdr_present_upscaled_with(
        cmd,
        cache,
        src,
        dst,
        peak,
        0,
        |cmd, enlarged, drawable| {
            scaled.set(true);
            super::encode_hdr_present(cmd, enlarged, drawable, peak, 0)
        },
    );
    assert!(
        scaled.get(),
        "the upscaled HDR present must go through MetalFX"
    );
    encoded
}

/// Tone-map at render size, then scale the float result in `HDR` mode.
fn tone_mapped_then_scaled(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    cache: &crate::metal::upscale::UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    peak: f32,
) -> bool {
    let scratch = texture_2d(
        &cmd.device(),
        MTLPixelFormat::RGBA16Float,
        (src.width(), src.height()),
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
    );
    super::encode_hdr_present(cmd, src, &scratch, peak, 0)
        && crate::metal::upscale::encode(
            cmd,
            &cmd.device(),
            cache,
            &scratch,
            dst,
            MTLFXSpatialScalerColorProcessingMode::HDR,
        )
}

/// The middle row's red codes after `Perceptual` scaling of `src` into a `BGRA8Unorm` drawable.
fn sdr_upscaled_row(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    cache: &crate::metal::upscale::UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    (width, height): (usize, usize),
) -> Vec<f32> {
    let dst = drawable_shaped(queue, MTLPixelFormat::BGRA8Unorm, width, height);
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-sdr-upscale")));
    assert!(crate::metal::upscale::encode(
        &cmd,
        &queue.device(),
        cache,
        src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    let row = width * 4;
    texture_bytes(queue, &dst)[height / 2 * row..(height / 2 + 1) * row]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|texel| f32::from(texel[2]))
        .collect()
}

/// The red channel of the middle row of a frame of `width` by `height` texels.
fn middle_row(values: &[[f32; 4]], (width, height): (usize, usize)) -> Vec<f32> {
    values[height / 2 * width..(height / 2 + 1) * width]
        .iter()
        .map(|texel| texel[0])
        .collect()
}

/// How many steps along `row` go down.
fn down_steps(row: &[f32]) -> usize {
    row.windows(2).filter(|pair| pair[1] < pair[0]).count()
}

/// Run `encode` into an `RGBA16Float` destination of `src`'s size and read every texel.
fn hdr_values(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    src: &ProtocolObject<dyn MTLTexture>,
    encode: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>, &ProtocolObject<dyn MTLTexture>) -> bool,
) -> Vec<[f32; 4]> {
    hdr_values_at(queue, (src.width(), src.height()), encode)
}

/// Run `encode` into an `RGBA16Float` destination of `extent` and read every texel.
///
/// The destination is Private and render-target only, like the float
/// drawable of an EDR layer. Every channel must be a finite half.
fn hdr_values_at(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    (width, height): (usize, usize),
    encode: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>, &ProtocolObject<dyn MTLTexture>) -> bool,
) -> Vec<[f32; 4]> {
    let dst = drawable_shaped(queue, MTLPixelFormat::RGBA16Float, width, height);
    let cmd = queue.commandBuffer().expect("command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-hdr-tone-map")));
    assert!(encode(&cmd, &dst), "the HDR pass must encode");
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    texture_bytes(queue, &dst)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|texel| {
            let halves = texel.as_chunks::<2>().0;
            core::array::from_fn(|channel| {
                let bits = u16::from_le_bytes(halves[channel]);
                assert_ne!(
                    bits & 0x7c00,
                    0x7c00,
                    "channel {channel} is not a finite half"
                );
                half_to_f32(bits)
            })
        })
        .collect()
}

/// A finite IEEE half's value.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = bits & 0x3ff;
    if exponent == 0 {
        return sign * f32::from(mantissa) / 16_777_216.0;
    }
    let normal = f32::from_bits(((exponent + 112) << 23) | (u32::from(mantissa) << 13));
    sign * normal
}

/// Bytes per texel of the two drawable formats and the 16-bit test sources.
fn bytes_per_texel(format: MTLPixelFormat) -> usize {
    if format == MTLPixelFormat::RGBA16Float || format == MTLPixelFormat::RGBA16Unorm {
        8
    } else {
        4
    }
}

/// A Private, shader-readable texture filled with random texels, and those texels' bytes.
///
/// `RGBA16Float` texels are finite normal halves: a NaN payload or a denormal
/// is not something a drawable ever carries.
fn random_source(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    width: usize,
    height: usize,
) -> (Retained<ProtocolObject<dyn MTLTexture>>, Vec<u8>) {
    let bpp = bytes_per_texel(format);
    let mut state =
        0x9e37_79b9_7f4a_7c15_u64 ^ u64::try_from(width * 4096 + height).expect("a small extent");
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let bytes: Vec<u8> = if bpp == 8 {
        (0..width * height * 4)
            .flat_map(|_| {
                let random = next();
                let sign = u16::from(random & 1 == 1) << 15;
                let exponent = u16::try_from(1 + (random >> 1) % 30).expect("below 31") << 10;
                let mantissa = u16::try_from((random >> 8) & 0x3ff).expect("ten bits");
                (sign | exponent | mantissa).to_le_bytes()
            })
            .collect()
    } else {
        (0..width * height)
            .flat_map(|_| {
                u32::try_from(next() >> 32)
                    .expect("the high half")
                    .to_le_bytes()
            })
            .collect()
    };
    (uploaded(queue, format, (width, height), &bytes), bytes)
}

/// A Private, shader-readable texture holding `bytes`, rows packed.
///
/// Uploaded through a blit from a shared buffer, which works whether or not
/// the GPU has unified memory.
fn uploaded(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    (width, height): (usize, usize),
    bytes: &[u8],
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    let device = queue.device();
    let row = width * bytes_per_texel(format);
    assert_eq!(bytes.len(), row * height, "one packed image");
    let buffer = device
        .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
        .expect("upload buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload")));
    // SAFETY: the shared buffer was created with exactly `bytes.len()` bytes
    // and nothing else references it yet.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    let texture = texture_2d(
        &device,
        format,
        (width, height),
        MTLTextureUsage::ShaderRead,
    );
    let cmd = queue.commandBuffer().expect("upload command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload")));
    let blit = cmd.blitCommandEncoder().expect("upload blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-exact-upload-copy")));
    // SAFETY: the buffer holds `height` rows of `row` bytes, exactly the
    // texture's extent, and both stay alive until the copy completes.
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            &buffer, 0, row, row * height, MTLSize { width, height, depth: 1 },
            &texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    texture
}

/// A texture shaped like a framebuffer-only drawable: Private and render-target only.
fn drawable_shaped(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    format: MTLPixelFormat,
    width: usize,
    height: usize,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    texture_2d(
        &queue.device(),
        format,
        (width, height),
        MTLTextureUsage::RenderTarget,
    )
}

/// One Private, single-level 2D texture.
fn texture_2d(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    (width, height): (usize, usize),
    usage: MTLTextureUsage,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    // SAFETY: objc2 typed binding; a single-level 2D descriptor of a colour
    // format at an extent within Metal's limits.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, width, height, false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(usage);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("test texture");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-exact")));
    texture
}

/// Every texel of `texture`, rows packed, read back through a staging buffer.
fn texture_bytes(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * bytes_per_texel(texture.pixelFormat());
    let stride = row.next_multiple_of(256);
    let buffer = queue
        .device()
        .newBufferWithLength_options(stride * height, MTLResourceOptions::StorageModeShared)
        .expect("readback buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback")));
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback")));
    let blit = cmd.blitCommandEncoder().expect("readback blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-exact-readback-copy")));
    // SAFETY: the buffer holds `height` rows of `stride` bytes, enough for
    // the whole texture, and both stay alive until the copy completes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width, height, depth: 1 },
            &buffer, 0, stride, stride * height,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the completed copy initialized every row of the shared buffer,
    // which is `stride * height` bytes long.
    let all = unsafe {
        core::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), stride * height)
    };
    all.chunks_exact(stride)
        .flat_map(|line| &line[..row])
        .copied()
        .collect()
}

/// Assert two packed images are equal, naming the first texel that differs.
fn assert_texels_equal(actual: &[u8], expected: &[u8], bpp: usize, width: usize) {
    assert_eq!(actual.len(), expected.len());
    let pairs = || actual.chunks_exact(bpp).zip(expected.chunks_exact(bpp));
    if let Some(first) = pairs().position(|(read, wrote)| read != wrote) {
        let differing = pairs().filter(|(read, wrote)| read != wrote).count();
        panic!(
            "texel ({}, {}) reads {:?} where the source holds {:?}; {differing} texels differ",
            first % width,
            first / width,
            &actual[first * bpp..(first + 1) * bpp],
            &expected[first * bpp..(first + 1) * bpp],
        );
    }
}
