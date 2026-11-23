use std::sync::Arc;

use mtld3d_shared::command_header::COMMAND_HEADER_BYTES;

use super::*;
use crate::{
    encoder_data::{BeginVisibilityOp, SetViewportOp},
    visibility::VisibilityQueryCore,
};

fn first_payload<T: CommandRecord>(scratch: &mut ScratchArena) -> &T {
    let (address, length) = scratch.command_ranges().next().unwrap();
    assert!(length >= COMMAND_HEADER_BYTES as u64 + size_of::<T>() as u64);
    // SAFETY: the test retains the recorder's scratch arena and this exact initialized
    // command region. Its fixed header precedes a payload with the asserted size.
    let payload = unsafe {
        core::slice::from_raw_parts(
            (address + COMMAND_HEADER_BYTES as u64) as *const u8,
            size_of::<T>(),
        )
    };
    crate::encoder_records::borrow(payload).unwrap()
}

#[test]
fn typed_viewport_is_constructed_in_final_command_storage() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_typed(
            &mut scratch,
            SetViewportOp {
                x: 7,
                y: 9,
                width: 640,
                height: 480,
                min_z: 0.25,
                max_z: 0.75,
            },
        )
        .unwrap();
    let record = first_payload::<SetViewportRecord>(&mut scratch);
    assert_eq!(
        (record.x, record.y, record.width, record.height),
        (7, 9, 640, 480)
    );
    assert_eq!((record.min_z, record.max_z), (0.25, 0.75));
    assert_eq!(recorder.len(), 1);
    assert!(recorder.rejected_ops.is_empty());
}

#[test]
fn typed_query_retains_original_owner_and_captured_generation() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    recorder
        .record_typed(
            &mut scratch,
            BeginVisibilityOp {
                generation: 37,
                c: query,
            },
        )
        .unwrap();
    assert_eq!(first_payload::<QueryRecord>(&mut scratch).generation, 37);
    assert_eq!(recorder.queries.len(), 1);
    assert!(weak.upgrade().is_some());
    assert!(recorder.rejected_ops.is_empty());
    // SAFETY: this fixture was never published and no native reader can adopt the lease.
    unsafe { recorder.queries[0].cancel_unadopted() };
    drop(recorder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn a_depth_transfer_names_a_texture_end_by_its_texture_and_level() {
    use mtld3d_shared::{
        MetalHandle,
        mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage},
    };

    use crate::{
        encoder_data::{ResolveDepthSurfaceOp, TextureInfo},
        ids::TextureId,
    };

    let id = TextureId::new_unique();
    let info = TextureInfo {
        texture_id: id,
        d3d_format: mtld3d_types::D3DFMT_INTZ,
        width: 480,
        height: 360,
        depth: 1,
        levels: 2,
        pixel_format: PixelFormat::Depth32FloatStencil8,
        create_flags: TextureCreateFlags::empty(),
        swizzle: [Swizzle::Zero; 4],
        usage_flags: TextureUsage::DEPTH_STENCIL,
    };
    // SAFETY: an opaque test value nothing dereferences.
    let plane = unsafe { MetalHandle::new(0x20) };
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_typed(
            &mut scratch,
            ResolveDepthSurfaceOp {
                transfer: DepthTransfer {
                    source: StretchKind::DepthStencil(plane),
                    source_level: 0,
                    source_size: (320, 240),
                    source_format: PixelFormat::Depth32FloatStencil8,
                    source_samples: 1,
                    destination: StretchKind::Texture(info),
                    destination_level: 1,
                    destination_size: (240, 180),
                    destination_format: PixelFormat::Depth32FloatStencil8,
                },
            },
        )
        .unwrap();
    let record = first_payload::<DepthTransferRecord>(&mut scratch);
    assert_eq!(record.source.texture_id(), None, "the plane is a handle");
    assert_eq!((record.source.kind, record.source.handle), (2, 0x20));
    assert_eq!(record.destination.texture_id(), Some(id));
    assert_eq!(record.destination.kind, 0);
    assert_eq!(record.destination.handle, 0);
    assert_eq!(
        (
            record.destination.texture.width,
            record.destination.texture.levels
        ),
        (480, 2)
    );
    assert_eq!((record.source_level, record.destination_level), (0, 1));
    assert_eq!((record.source_width, record.source_height), (320, 240));
    assert_eq!(
        (record.destination_width, record.destination_height),
        (240, 180)
    );
    assert_eq!(record.reserved, 0);
    assert!(recorder.rejected_ops.is_empty());
}

#[test]
fn latched_failure_retains_typed_owner_without_publishing_command() {
    let mut scratch = ScratchArena::new();
    let mut recorder = FrameRecorder::new();
    recorder.error = Some(WireError::AllocationFailed);
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    assert!(
        recorder
            .record_typed(
                &mut scratch,
                BeginVisibilityOp {
                    generation: 9,
                    c: query
                }
            )
            .is_err()
    );
    assert!(scratch.command_ranges().next().is_none());
    assert!(recorder.queries.is_empty());
    assert_eq!(recorder.rejected_ops.len(), 1);
    assert!(weak.upgrade().is_some());
    drop(recorder);
    assert!(weak.upgrade().is_none());
}
