use std::sync::{Arc, Weak};

use mtld3d_shared::{MetalHandle, mtl::PixelFormat, record_handle::DeviceRecordHandle};

use super::*;
use crate::{
    encoder_data::{
        BeginVisibilityOp, FrameData, FrameInit, StageUploadOp, TextureInfo, TextureUploadJob,
        UploadTextureOp,
    },
    encoder_records::{QueryRecord, StageUploadRecord, borrow},
    guest_pages::GuestOwnedPage,
    guest_queries::QueryLeaseCache,
    ids::BufferId,
    page_box::PageBox,
    passes::BackbufferContents,
    render_scale::RenderScale,
    visibility::VisibilityQueryCore,
};

pub(super) fn empty_frame() -> FrameData {
    FrameData::new(&FrameInit {
        device_handle: MetalHandle::NULL,
        record_handle: DeviceRecordHandle::NULL,
        backbuffer_handle: MetalHandle::NULL,
        backbuffer_srgb_handle: MetalHandle::NULL,
        backbuffer_msaa_handle: MetalHandle::NULL,
        backbuffer_msaa_srgb_handle: MetalHandle::NULL,
        backbuffer_sample_count: 1,
        layer_handle: MetalHandle::NULL,
        view_handle: MetalHandle::NULL,
        backbuffer_width: 64,
        backbuffer_height: 64,
        backbuffer_format: PixelFormat::Bgra8Unorm,
        render_scale: RenderScale::IDENTITY,
        backbuffer_contents: BackbufferContents::Undefined,
        depth_texture: MetalHandle::NULL,
        depth_has_stencil: false,
    })
}

pub(super) fn seal(mut frame: FrameData, recorder: FrameRecorder) -> FramePacket {
    frame.recorder = Some(recorder);
    FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("valid fixture: {error:?}"))
}

fn packet_with_leases() -> (FramePacket, Weak<VisibilityQueryCore>) {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    for value in [17_u8, 29] {
        let mut page = PageBox::new_zeroed(4);
        page.as_mut_slice()[..4].fill(value);
        recorder
            .record_typed(
                &mut frame.scratch,
                StageUploadOp {
                    buffer_id: BufferId::new_unique(),
                    page_box: page,
                    dst_offset: 0,
                    size: 4,
                },
            )
            .unwrap();
    }
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    recorder
        .record_typed(
            &mut frame.scratch,
            BeginVisibilityOp {
                generation: 7,
                c: query,
            },
        )
        .unwrap();
    (seal(frame, recorder), weak)
}

// The caller keeps owner alive until native replay and submit have released every borrow.
pub(super) fn admit(owner: &mut FramePacket) -> ReplayPacket {
    // SAFETY: this fixture retains the sole owner through all native uses and quiescence.
    unsafe { owner.mark_admitted() };
    // SAFETY: all records were published by the real typed recorder and remain immutable.
    unsafe {
        prepare_packet(
            owner.metadata_bytes(),
            owner.operation_bytes(),
            owner.completion_address(),
        )
    }
    .unwrap()
}

pub(super) fn replay(
    packet: &mut ReplayPacket,
    consume: impl FnOnce(CommandView<'_>, &mut NativeFrame, &mut ReplayState) -> Result<(), WireError>,
) -> Result<bool, WireError> {
    // SAFETY: these fixture callbacks never replace the frame or clear its scratch. Every
    // borrowed command and adopted owner remains under the packet's retained lease.
    unsafe { packet.replay_one(consume) }
}

fn consume_leases(
    command: &CommandView<'_>,
    pages: &mut Vec<GuestOwnedPage>,
    queries: &mut Vec<Arc<VisibilityQueryCore>>,
    cache: &mut QueryLeaseCache,
) -> Result<(), WireError> {
    match command.opcode() {
        EncoderOpcode::StageUpload => {
            assert_eq!(command.payload().len(), 56);
            let record = borrow::<StageUploadRecord>(command.payload())?;
            assert_eq!((record.offset, record.size), (0, 4));
            // SAFETY: the real producer retained this unique owned page descriptor.
            let page = unsafe { record.page.adopt()? };
            // SAFETY: the fixture initialized the first four bytes and the guard retains them.
            let bytes = unsafe { core::slice::from_raw_parts(page.as_ptr(), 4) };
            assert_eq!(bytes, &[if pages.is_empty() { 17 } else { 29 }; 4]);
            pages.push(page);
        }
        EncoderOpcode::BeginVisibility => {
            let record = borrow::<QueryRecord>(command.payload())?;
            assert_eq!(record.generation, 7);
            // SAFETY: the real producer retained this unique query lease until native drop.
            queries.push(unsafe { cache.adopt(&record.descriptor)? });
        }
        _ => panic!("unexpected fixture opcode"),
    }
    Ok(())
}

/// Consume every queued notification, as the device's maintenance pass does before its walk.
fn drain_all(owner: &FramePacket) -> usize {
    owner.drain_test_completions(&mut crate::guest_completions::CompletionDrain::default())
}

#[test]
fn typed_dispatch_keeps_pages_queries_and_submit_storage_alive() {
    let (mut owner, query) = packet_with_leases();
    let mut packet = admit(&mut owner);
    let mut pages = Vec::new();
    let mut queries = Vec::new();
    let mut cache = QueryLeaseCache::default();
    let mut count = 0;
    while replay(&mut packet, |command, _, _| {
        consume_leases(&command, &mut pages, &mut queries, &mut cache)
    })
    .unwrap()
    {
        count += 1;
    }
    assert_eq!(count, 3);
    assert_eq!((pages.len(), queries.len()), (2, 1));
    let submitted = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}"));
    assert!(!owner.maintain());
    drop(submitted);
    let mut drain = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut drain);
    assert!(
        !owner.maintain(),
        "native page and query owners remain live after frame drop"
    );
    assert!(query.upgrade().is_some());
    drop(pages);
    drop(queries);
    cache.maintain();
    assert_eq!(owner.drain_test_completions(&mut drain), 3);
    assert!(owner.maintain());
    assert!(query.upgrade().is_none());
}

#[test]
fn final_command_requires_exhaustion_before_frame_transfer() {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_constant_bytes(
            &mut frame.scratch,
            EncoderOpcode::SetVsConstRange,
            3,
            1,
            &[0x5a; 16],
        )
        .unwrap();
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    assert!(
        replay(&mut packet, |command, _, _| {
            let constants = command.constants()?;
            assert_eq!((constants.start_row, constants.rows), (3, 1));
            assert_eq!(constants.data.as_slice(), &[0x5a; 16]);
            Ok(())
        })
        .unwrap()
    );
    let Err((error, mut packet)) = packet.into_frame() else {
        panic!("must observe exhaustion");
    };
    assert_eq!(error, WireError::InvalidValue);
    assert!(!owner.maintain());
    assert!(!replay(&mut packet, |_, _, _| panic!("no command remains")).unwrap());
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete: {error:?}"));
    assert!(!owner.maintain());
    drop(frame);
    assert!(
        !owner.maintain(),
        "the replay completion waits to be consumed"
    );
    assert_eq!(drain_all(&owner), 1);
    assert!(owner.maintain());
}

#[test]
fn failed_dispatch_keeps_the_packet_lease_until_quarantine_is_released() {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    recorder
        .record_constant_bytes(
            &mut frame.scratch,
            EncoderOpcode::SetVsConstRange,
            0,
            1,
            &[0; 16],
        )
        .unwrap();
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let error = replay(&mut packet, |_, _, _| Err(WireError::InvalidValue));
    assert_eq!(error, Err(WireError::InvalidValue));
    assert_eq!(
        replay(&mut packet, |_, _, _| panic!("failure must be sticky")),
        Err(WireError::InvalidValue)
    );
    let Err((_, packet)) = packet.into_frame() else {
        panic!("failed replay must be quarantined");
    };
    assert!(!owner.maintain());
    assert!(!owner.was_rejected());
    drop(packet);
    drain_all(&owner);
    assert!(owner.was_rejected());
    assert!(!owner.maintain());
    // SAFETY: all native users have been dropped, as after device shutdown.
    unsafe { owner.cancel_unadopted() };
    assert!(owner.maintain());
}

#[test]
fn inline_constants_cross_regions_and_keep_payload_until_submit_drop() {
    let mut frame = empty_frame();
    frame.scratch = ScratchArena::with_chunk_size(128);
    let mut recorder = FrameRecorder::new();
    for row in 0_u16..12 {
        recorder
            .record_constant_bytes(
                &mut frame.scratch,
                EncoderOpcode::SetVsConstRange,
                row,
                4,
                &[u8::try_from(row).unwrap(); 64],
            )
            .unwrap();
    }
    let mut owner = seal(frame, recorder);
    assert!(
        owner.operation_bytes().len() > 16,
        "fixture must exercise region rollover"
    );
    let mut packet = admit(&mut owner);
    let mut payloads = Vec::new();
    while replay(&mut packet, |command, _, _| {
        let constants = command.constants()?;
        assert_eq!(constants.start_row as usize, payloads.len());
        assert_eq!(constants.rows, 4);
        payloads.push(constants.data);
        Ok(())
    })
    .unwrap()
    {}
    assert_eq!(payloads.len(), 12);
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete: {error:?}"));
    assert!(!owner.maintain());
    for (row, payload) in payloads.iter().enumerate() {
        assert_eq!(payload.as_slice(), &[u8::try_from(row).unwrap(); 64]);
    }
    drop(payloads);
    drop(frame);
    drain_all(&owner);
    assert!(owner.maintain());
}

#[test]
fn invalid_constant_capture_latches_error_without_publishing_a_command() {
    let mut recorder = FrameRecorder::new();
    let mut scratch = ScratchArena::new();
    assert_eq!(
        recorder.record_constant_bytes(
            &mut scratch,
            EncoderOpcode::SetVsConstRange,
            255,
            2,
            &[0; 32]
        ),
        Err(WireError::InvalidValue)
    );
    assert_eq!(recorder.recording_error(), Some(WireError::InvalidValue));
    assert_eq!(
        recorder.record_constant_bytes(
            &mut scratch,
            EncoderOpcode::SetPsConstRange,
            0,
            1,
            &[0; 16]
        ),
        Err(WireError::InvalidValue)
    );
    assert_eq!(recorder.count, 0);
    assert!(scratch.command_descriptor_bytes().is_empty());
}

#[test]
fn single_stream_records_match_checked_writer_and_reuse_failed_rollover() {
    use mtld3d_shared::mtl::{IndexType, PrimitiveType};

    use crate::{
        draw_data::{ExtraStreams, StreamBinding},
        encoder_draw::draw_record::{
            BoundVertices, DrawPrefix, IndexBuffer, StreamRecord, bound_payload_size,
            write_bound_into,
        },
    };

    for index_kind in [None, Some(IndexType::UInt16), Some(IndexType::UInt32)] {
        for owned_empty in [false, true] {
            let vertices = BoundVertices {
                first: StreamRecord::from_binding(&StreamBinding {
                    stream: 7,
                    buffer_id: BufferId::new_unique(),
                    backing_ptr: 0x1234,
                    backing_len: 8192,
                    backing_generation: 91,
                    offset: 12,
                    stride: 24,
                    freq: 0x4000_0011,
                }),
                extra: if owned_empty {
                    ExtraStreams::Owned(Box::new([]))
                } else {
                    ExtraStreams::EMPTY
                },
                stream0_freq: 0x8000_0003,
            };
            let index = index_kind.map(|kind| IndexBuffer {
                buffer: BufferId::new_unique().raw(),
                address: 0x5678,
                length: 4096,
                generation: 123,
                offset: 16,
                kind: kind as u8,
                reserved: [0; 3],
            });
            let prefix = |indexed| {
                if indexed {
                    DrawPrefix::indexed(PrimitiveType::Triangle, i32::MIN, 9)
                } else {
                    DrawPrefix::nonindexed(PrimitiveType::Triangle, u32::MAX - 9, 9)
                }
            };
            let length = bound_payload_size(&vertices, index.as_ref()).unwrap();
            let mut expected = vec![0xcc; length];
            write_bound_into(
                prefix(index.is_some()),
                &vertices,
                index.as_ref(),
                &mut expected,
                length,
            )
            .unwrap();
            let command_bytes = length + mtld3d_shared::command_header::COMMAND_HEADER_BYTES;
            let mut scratch = ScratchArena::with_chunk_size(command_bytes);
            let mut recorder = FrameRecorder::new();
            recorder
                .record_bound_draw(
                    &mut scratch,
                    prefix(index.is_some()),
                    &vertices,
                    index.as_ref(),
                )
                .unwrap();
            let committed: Vec<_> = scratch.command_ranges().collect();
            assert_eq!(committed.len(), 1);
            let used = scratch.bytes_used();
            assert_eq!(
                recorder.record_bound_draw(
                    &mut scratch,
                    prefix(index.is_none()),
                    &vertices,
                    index.as_ref(),
                ),
                Err(WireError::InvalidValue)
            );
            assert_eq!(scratch.bytes_used(), used);
            // The rejected record's rollover named an empty region; the committed one is intact.
            let after_failure: Vec<_> = scratch.command_ranges().collect();
            assert_eq!(after_failure.len(), 2);
            assert_eq!(after_failure[0], committed[0]);
            assert_eq!(after_failure[1].1, 0);
            assert_eq!(recorder.len(), 1);
            scratch.publish_command_region().unwrap();
            // SAFETY: the live recorder and arena retain this unchanged committed region.
            let mut cursor =
                unsafe { replay::CommandCursor::new(scratch.command_descriptor_bytes()) }.unwrap();
            // SAFETY: the initialized region remains owned and immutable through this read.
            let record = unsafe { cursor.next_record() }.unwrap().unwrap();
            assert_eq!(record.payload, expected);
            assert_eq!(
                recorder.record_bound_draw(
                    &mut scratch,
                    prefix(index.is_some()),
                    &vertices,
                    index.as_ref(),
                ),
                Err(WireError::InvalidValue)
            );
            assert_eq!(scratch.bytes_used(), used);
            recorder.reset();
            scratch.clear();
            for _ in 0..2 {
                recorder
                    .record_bound_draw(
                        &mut scratch,
                        prefix(index.is_some()),
                        &vertices,
                        index.as_ref(),
                    )
                    .unwrap();
            }
            let reused: Vec<_> = scratch.command_ranges().collect();
            assert_eq!(reused.len(), 2);
            assert_eq!(reused[0], committed[0]);
            assert_eq!(scratch.chunk_count(), 2);
            scratch.publish_command_region().unwrap();
            // SAFETY: both completed records remain owned by the live recorder and arena.
            let mut cursor =
                unsafe { replay::CommandCursor::new(scratch.command_descriptor_bytes()) }.unwrap();
            for _ in 0..2 {
                // SAFETY: the same immutable command regions remain retained during iteration.
                let record = unsafe { cursor.next_record() }.unwrap().unwrap();
                assert_eq!(record.payload, expected);
            }
            assert!(cursor.is_complete());
        }
    }
}

#[test]
fn capture_failure_survives_later_snapshot_draw_and_owned_control() {
    use crate::draw_data::{DrawOp, ExtraStreams, IndexSource, StreamBinding, VertexSource};

    for first_error in [WireError::AllocationFailed, WireError::InvalidValue] {
        let mut recorder = FrameRecorder::new();
        let mut scratch = ScratchArena::new();
        recorder
            .record_constant_bytes(&mut scratch, EncoderOpcode::SetPsConstRange, 0, 1, &[0; 16])
            .unwrap();
        let committed_bytes = scratch.bytes_used();
        // Model an allocator or capture failure at the recorder's existing completion point.
        assert_eq!(recorder.finish_record(Err(first_error)), Err(first_error));
        assert_eq!(
            recorder.record_snapshot_delta(
                &mut scratch,
                &crate::encoder_draw::SnapshotDelta::default()
            ),
            Err(first_error)
        );
        let draw = DrawOp {
            metal_prim: mtld3d_shared::mtl::PrimitiveType::Triangle,
            vertex_source: VertexSource::Bound {
                first: StreamBinding {
                    stream: 0,
                    buffer_id: BufferId::new_unique(),
                    backing_ptr: 0,
                    backing_len: 4096,
                    backing_generation: 1,
                    offset: 0,
                    stride: 16,
                    freq: 1,
                },
                extra: ExtraStreams::EMPTY,
                stream0_freq: 1,
            },
            index_source: IndexSource::None {
                start_vertex: 0,
                vertex_count: 3,
            },
        };
        assert_eq!(recorder.record_draw(&mut scratch, &draw), Err(first_error));
        let VertexSource::Bound {
            first,
            extra,
            stream0_freq,
        } = draw.vertex_source
        else {
            unreachable!()
        };
        let vertices = crate::encoder_draw::draw_record::BoundVertices {
            first: crate::encoder_draw::draw_record::StreamRecord::from_binding(&first),
            extra,
            stream0_freq,
        };
        assert_eq!(
            recorder.record_bound_draw(
                &mut scratch,
                crate::encoder_draw::draw_record::DrawPrefix::nonindexed(
                    mtld3d_shared::mtl::PrimitiveType::Triangle,
                    0,
                    3,
                ),
                &vertices,
                None,
            ),
            Err(first_error)
        );
        // A later invalid input must not replace an earlier allocation failure.
        assert_eq!(
            recorder.record_constant_bytes(
                &mut scratch,
                EncoderOpcode::SetVsConstRange,
                255,
                2,
                &[0; 32]
            ),
            Err(first_error)
        );
        let query = VisibilityQueryCore::new();
        let weak = Arc::downgrade(&query);
        assert_eq!(
            recorder.record_typed(
                &mut scratch,
                BeginVisibilityOp {
                    generation: 1,
                    c: query
                }
            ),
            Err(first_error)
        );
        assert_eq!(recorder.recording_error(), Some(first_error));
        assert_eq!(recorder.count, 1);
        assert_eq!(scratch.bytes_used(), committed_bytes);
        assert!(weak.upgrade().is_some());
        drop(recorder);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn canceled_packet_returns_completion_slots_to_its_pool() {
    let (mut packet, query) = packet_with_leases();
    let pool = packet
        .recorder
        .as_ref()
        .expect("recorder")
        .completion_pool
        .clone();
    let mut original: Vec<_> = packet
        .pages
        .iter()
        .filter_map(GuestPageLease::token)
        .chain(packet.owned_pages.iter().map(GuestOwnedPageLease::token))
        .chain(packet.queries.iter().filter_map(GuestQueryLease::token))
        .collect();
    original.sort_unstable();
    assert_eq!(original.len(), 3);
    let mut cursor = crate::guest_completions::CompletionDrain::default();
    // SAFETY: this packet was never exposed to a native consumer.
    unsafe { packet.cancel_unadopted() };
    // Three lease cancellations and the packet's own replay completion.
    assert_eq!(packet.drain_test_completions(&mut cursor), 4);
    assert!(packet.maintain());
    drop(packet);
    assert!(query.upgrade().is_none());
    let replacements: Vec<_> = (0..3).map(|_| pool.allocate(false)).collect();
    let mut reused: Vec<_> = replacements
        .iter()
        .map(crate::guest_completions::CompletionSlot::token)
        .collect();
    reused.sort_unstable();
    assert_eq!(
        reused, original,
        "cancellation must recycle every published slot"
    );
    for slot in &replacements {
        slot.completion().publish();
    }
    let mut completed = 0;
    pool.drain(&mut cursor, 3, |_| completed += 1);
    assert_eq!(completed, 3);
    for slot in replacements {
        pool.recycle(slot);
    }
}

#[test]
fn early_native_lease_events_do_not_retire_a_pending_or_rejected_packet() {
    let frame = empty_frame();
    let mut packet =
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("empty fixture: {error:?}"));
    let pool = packet
        .recorder
        .as_ref()
        .expect("recorder")
        .completion_pool
        .clone();
    let original = Arc::new(PageBox::new_zeroed(4));
    let read = crate::page_box::PageBoxRead::new(Arc::clone(&original));
    let lease = GuestPageLease::for_read_pooled(read, &pool, None);
    let descriptor = lease.descriptor();
    packet.pages.push(lease);
    // SAFETY: the fixture retains one packet and models its only native consumer.
    unsafe { packet.mark_admitted() };
    // SAFETY: the retained lease permits this sole adoption until its native read drops.
    let native = unsafe { descriptor.adopt_read() }.expect("native read");
    drop(native);
    let mut cursor = crate::guest_completions::CompletionDrain::default();
    assert_eq!(packet.drain_test_completions(&mut cursor), 2);
    assert_eq!(packet.take_leases().count(), 0);
    assert!(!packet.maintain());
    assert_eq!(packet.pages.len(), 1);
    assert!(
        original.has_readers(),
        "pending replay keeps its original read guard"
    );

    // SAFETY: the packet retains its completion cell and this models the native
    // decoder rejecting only after dropping its partially reconstructed owners.
    let complete = unsafe { &*(packet.completion_address() as *const LeaseCompletion) };
    complete.publish_rejected();
    assert_eq!(packet.take_leases().count(), 0);
    assert!(!packet.maintain());
    assert!(
        original.has_readers(),
        "rejection remains quarantined until shutdown"
    );
    // SAFETY: the only native owner was dropped above; the runtime is quiescent.
    unsafe { packet.cancel_unadopted() };
    packet.drain_test_completions(&mut cursor);
    assert!(packet.maintain());
    assert!(!original.has_readers());
}

#[test]
fn canonical_upload_retains_source_and_feedback_through_native_use() {
    use mtld3d_shared::mtl::{Swizzle, TextureCreateFlags, TextureUsage};
    use mtld3d_types::D3DFMT_A8R8G8B8;

    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    let backing = Arc::new(PageBox::new_zeroed(64));
    let weak_page = Arc::downgrade(&backing);
    let redirty = Arc::new(crate::upload_redirty::RedirtyQueue::new());
    let weak_feedback = Arc::downgrade(&redirty);
    let job = TextureUploadJob {
        info: TextureInfo {
            texture_id: crate::ids::TextureId::new_unique(),
            d3d_format: D3DFMT_A8R8G8B8,
            width: 4,
            height: 4,
            depth: 1,
            levels: 1,
            pixel_format: PixelFormat::Bgra8Unorm,
            create_flags: TextureCreateFlags::empty(),
            swizzle: [Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::Alpha],
            usage_flags: TextureUsage::empty(),
        },
        staging: crate::page_box::PageBoxRead::new(backing),
        level: 0,
        destination_slice: 0,
        staging_index: 0,
        origin_x: 0,
        origin_y: 0,
        region_w: 4,
        region_h: 4,
        src_d3d_format: D3DFMT_A8R8G8B8,
        src_pitch: 16,
        bytes_per_pixel: 4,
        depth: 1,
        slice_pitch: 64,
        redirty,
        release_staging: true,
        upload_generation: 1,
        snapshot_offset: None,
    };
    recorder
        .record_typed(&mut frame.scratch, UploadTextureOp { job })
        .unwrap();
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let mut read = None;
    let mut feedback = None;
    assert!(
        replay(&mut packet, |command, _, _| {
            assert!(matches!(command.opcode(), EncoderOpcode::UploadTexture));
            let record = borrow::<crate::encoder_records::TextureUploadRecord>(command.payload())?;
            assert_eq!(record.upload_generation, 1);
            record.validate_source(record.page.wire_fields()[2])?;
            // SAFETY: the actual producer retained these unique read and feedback descriptors.
            read = Some(unsafe { record.page.adopt_read()? });
            // SAFETY: the feedback lease remains owned by this admitted packet through native use.
            feedback = Some(unsafe { record.redirty.adopt()? });
            Ok(())
        })
        .unwrap()
    );
    assert!(!replay(&mut packet, |_, _, _| panic!("one upload")).unwrap());
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete: {error:?}"));
    drop(frame);
    let mut drain = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut drain);
    assert!(!owner.maintain());
    assert!(weak_page.upgrade().is_some());
    assert!(weak_feedback.upgrade().is_some());
    drop(read);
    drop(feedback);
    owner.drain_test_completions(&mut drain);
    assert!(owner.maintain());
    drop(owner);
    assert!(weak_page.upgrade().is_none());
    assert!(weak_feedback.upgrade().is_none());
}

/// A 4x4 A8R8G8B8 upload at `origin` of a 16x16 level, from `source` at `pitch`.
fn upload_job(
    source: Arc<PageBox>,
    snapshot_offset: Option<u32>,
    pitch: u32,
    origin: u32,
) -> TextureUploadJob {
    use mtld3d_shared::mtl::{Swizzle, TextureCreateFlags, TextureUsage};
    use mtld3d_types::D3DFMT_A8R8G8B8;
    TextureUploadJob {
        info: TextureInfo {
            texture_id: crate::ids::TextureId::new_unique(),
            d3d_format: D3DFMT_A8R8G8B8,
            width: 16,
            height: 16,
            depth: 1,
            levels: 1,
            pixel_format: PixelFormat::Bgra8Unorm,
            create_flags: TextureCreateFlags::empty(),
            swizzle: [Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::Alpha],
            usage_flags: TextureUsage::empty(),
        },
        staging: crate::page_box::PageBoxRead::new(source),
        level: 0,
        destination_slice: 0,
        staging_index: 0,
        origin_x: origin,
        origin_y: 4,
        region_w: 4,
        region_h: 4,
        src_d3d_format: D3DFMT_A8R8G8B8,
        src_pitch: pitch,
        bytes_per_pixel: 4,
        depth: 1,
        slice_pitch: pitch * 16,
        redirty: Arc::new(crate::upload_redirty::RedirtyQueue::new()),
        release_staging: false,
        upload_generation: 0,
        snapshot_offset,
    }
}

/// A snapshot upload's record carries its chunk offset under the snapshot flag, span checked.
///
/// The staging form keeps a zero offset and its level-shaped span. A
/// snapshot's rows must start on the arena's alignment and end inside the
/// chunk its lease names, and its box must lie inside the level, overflow
/// included.
#[test]
fn snapshot_upload_records_carry_their_offset_and_check_their_span() {
    use crate::{encoder_data::UploadTextureOpFlags, encoder_records::TextureUploadRecord};
    let staging = Arc::new(PageBox::new_zeroed(16 * 64));
    let chunk = Arc::new(PageBox::new_zeroed(65_536));
    let cases = [
        (Arc::clone(&staging), None, 64, 4, true),
        (Arc::clone(&chunk), Some(0), 16, 4, true),
        (Arc::clone(&chunk), Some(4096), 16, 12, true),
        (Arc::clone(&chunk), Some(8), 16, 4, false),
        (Arc::clone(&chunk), Some(65_536 - 48), 16, 4, false),
        (Arc::clone(&chunk), Some(0), 12, 4, false),
        // Four texels from x = 14 reach past the 16-texel level.
        (Arc::clone(&chunk), Some(0), 16, 14, false),
        (Arc::clone(&chunk), Some(0), 16, u32::MAX - 1, false),
    ];
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    for (source, offset, pitch, origin, _) in &cases {
        recorder
            .record_typed(
                &mut frame.scratch,
                UploadTextureOp {
                    job: upload_job(Arc::clone(source), *offset, *pitch, *origin),
                },
            )
            .unwrap();
    }
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let mut seen = Vec::new();
    let mut native = Vec::new();
    while replay(&mut packet, |command, _, _| {
        let record = borrow::<TextureUploadRecord>(command.payload())?;
        let flagged = record.mip_flags & u32::from(UploadTextureOpFlags::SNAPSHOT.bits()) != 0;
        seen.push((
            flagged,
            record.snapshot_offset(),
            record.source_offset,
            record.validate_source(record.page.wire_fields()[2]).is_ok(),
        ));
        // SAFETY: the actual producer retained these unique read and feedback descriptors.
        let read = unsafe { record.page.adopt_read()? };
        // SAFETY: the feedback lease remains owned by this admitted packet through native use.
        let feedback = unsafe { record.redirty.adopt()? };
        native.push((read, feedback));
        Ok(())
    })
    .unwrap()
    {}
    let expected: Vec<_> = cases
        .iter()
        .map(|(_, offset, _, _, valid)| (offset.is_some(), *offset, offset.unwrap_or(0), *valid))
        .collect();
    assert_eq!(seen, expected);
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete: {error:?}"));
    drop(frame);
    drop(native);
    let mut drain = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut drain);
    assert!(owner.maintain());
    drop(owner);
    assert!(!chunk.has_readers() && !staging.has_readers());
}

#[test]
fn failed_recording_keeps_resource_owner_until_explicit_quiescence() {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::new();
    assert!(
        recorder
            .record_constant_bytes(
                &mut frame.scratch,
                EncoderOpcode::SetVsConstRange,
                256,
                1,
                &[0; 16]
            )
            .is_err()
    );
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    assert!(
        recorder
            .record_typed(
                &mut frame.scratch,
                BeginVisibilityOp {
                    generation: 4,
                    c: query
                }
            )
            .is_err()
    );
    frame.recorder = Some(recorder);
    let Err((_, mut owner)) = FramePacket::new(frame) else {
        panic!("recording error must remain sticky");
    };
    assert!(weak.upgrade().is_some());
    assert!(!owner.maintain());
    // SAFETY: no packet was admitted and no native consumer can access these resources.
    unsafe { owner.cancel_unadopted() };
    let mut drain = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut drain);
    assert!(owner.maintain());
    drop(owner);
    assert!(weak.upgrade().is_none());
}

#[test]
fn indexed_up_borrows_original_arena_payload_through_region_reuse_and_submit() {
    use mtld3d_shared::mtl::{IndexType, PrimitiveType};

    use crate::{
        draw_data::{DrawOp, IndexSource, VertexSource, arena_alloc_bytes},
        encoder_draw::draw_record::{DrawView, IndexView, VertexView},
    };

    let mut frame = empty_frame();
    frame.scratch = ScratchArena::with_chunk_size(128);
    let mut source = vec![0x5a; 4800];
    let mut indices = [0_u8, 0, 1, 0, 2, 0];
    // SAFETY: the owning frame retains both initialized captures through replay and submit.
    let vertices = unsafe { arena_alloc_bytes(&mut frame.scratch, &source) };
    // SAFETY: the same owning frame retains this initialized index capture.
    let index_data = unsafe { arena_alloc_bytes(&mut frame.scratch, &indices) };
    source.fill(0);
    indices.fill(0xff);
    let mut recorder = FrameRecorder::new();
    recorder
        .record_draw(
            &mut frame.scratch,
            &DrawOp {
                metal_prim: PrimitiveType::Triangle,
                vertex_source: VertexSource::Up {
                    bytes: vertices,
                    size: 4800,
                    stride: 16,
                },
                index_source: IndexSource::Up {
                    bytes: index_data,
                    index_count: 3,
                    index_type: IndexType::UInt16,
                },
            },
        )
        .unwrap();
    for row in 0..12 {
        recorder
            .record_constant_bytes(
                &mut frame.scratch,
                EncoderOpcode::SetPsConstRange,
                row,
                4,
                &[0; 64],
            )
            .unwrap();
    }
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let mut retained = None;
    assert!(
        replay(&mut packet, |command, _, _| {
            assert!(matches!(command.opcode(), EncoderOpcode::Draw));
            let draw = DrawView::new(command.payload())?;
            let &VertexView::Up { record, stride } = draw.vertices() else {
                panic!("UP vertex input");
            };
            assert_eq!(stride, 16);
            // SAFETY: this authentic record refers to the retained frame arena payload.
            let vertex_bytes = unsafe { record.bytes() };
            let &IndexView::Up {
                record,
                index_count,
            } = draw.indices()
            else {
                panic!("UP index input");
            };
            assert_eq!(index_count, 3);
            assert!(matches!(record.index_type()?, IndexType::UInt16));
            // SAFETY: this authentic record refers to the retained initialized index payload.
            let index_bytes = unsafe { record.bytes() };
            assert_eq!(vertex_bytes.as_raw(), vertices.as_raw());
            assert_eq!(index_bytes.as_raw(), index_data.as_raw());
            retained = Some((vertex_bytes, index_bytes));
            Ok(())
        })
        .unwrap()
    );
    let mut count = 0;
    while replay(&mut packet, |command, _, _| {
        command.constants()?;
        count += 1;
        Ok(())
    })
    .unwrap()
    {}
    assert_eq!(count, 12);
    let submitted = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete: {error:?}"));
    assert!(!owner.maintain());
    let (vertices, indices) = retained.unwrap();
    assert_eq!(vertices.as_slice(), &[0x5a; 4800]);
    assert_eq!(indices.as_slice(), &[0, 0, 1, 0, 2, 0]);
    drop(submitted);
    drain_all(&owner);
    assert!(owner.maintain());
}

#[test]
fn retired_buffer_outlives_replay_and_recording_storage_reuse() {
    use crate::{encoder_data::PendingVbibRetention, page_box_pool::PageBoxPool};

    let pool = Box::leak(Box::new(PageBoxPool::new(65536)));
    let cells = crate::guest_completions::CompletionPool::new();
    let mut recorder = FrameRecorder::with_completion_pool(cells.clone());
    recorder.pagebox_pool = Some(pool);
    let mut frame = empty_frame();
    let page = PageBox::new_zeroed(12);
    let address = page.as_ptr();
    recorder.capture_vbib_retention(
        &mut frame.scratch,
        PendingVbibRetention {
            buffer_id: BufferId::new_unique(),
            page_box: page,
            last_submit_seq: 42,
        },
    );
    let allocation = recorder.owned_pages.as_ptr();
    let capacity = recorder.owned_pages.capacity();
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let mut native = None;
    assert!(
        replay(&mut packet, |command, _, _| {
            assert!(matches!(command.opcode(), EncoderOpcode::RetainVbib));
            assert_eq!(command.payload().len(), 48);
            let record = borrow::<metadata::VbibRetentionRecord>(command.payload())?;
            assert_eq!(record.last_submit_seq, 42);
            // SAFETY: this packet retains the sole owner until the native guard retires.
            native = Some(unsafe { record.page.adopt()? });
            Ok(())
        })
        .unwrap()
    );
    assert!(!replay(&mut packet, |_, _, _| panic!("one retirement")).unwrap());
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}")),
    );
    assert_eq!(
        owner.take_leases().count(),
        0,
        "replay completion not consumed yet"
    );
    assert_eq!(drain_all(&owner), 1);
    let mut leases: Vec<_> = owner.take_leases().collect();
    assert_eq!(leases.len(), 1);
    let (_, recorder) = owner.take_recording_storage().expect("completed recording");
    assert!(recorder.owned_pages.is_empty());
    assert_eq!(recorder.owned_pages.capacity(), capacity);
    assert_eq!(recorder.owned_pages.as_ptr(), allocation);
    assert!(owner.maintain());
    drop(owner);
    assert!(pool.acquire(12).is_none());
    assert!(!leases[0].maintain());
    drop(native);
    assert!(
        !leases[0].maintain(),
        "queued acknowledgment is not consumed"
    );
    cells.drain(
        &mut crate::guest_completions::CompletionDrain::default(),
        16,
        |_| {},
    );
    let mut lease = leases.pop().unwrap();
    assert!(lease.maintain());
    for slot in lease.into_slots().into_iter().flatten() {
        cells.recycle(slot);
    }
    assert_eq!(
        pool.acquire(12).expect("retired original pages").as_ptr(),
        address
    );
}

#[test]
fn recording_storage_reuse_keeps_untransferred_owners_on_packet() {
    let (mut owner, weak) = packet_with_leases();
    let cells = owner.recorder.as_ref().unwrap().completion_pool.clone();
    let allocation = owner.owned_pages.as_ptr();
    // SAFETY: this fixture was never admitted and none of its owners has native users.
    unsafe { owner.cancel_unadopted() };
    assert!(
        owner.take_recording_storage().is_none(),
        "completion not consumed yet"
    );
    // The replay completion was published last, so a budget of one consumes it alone
    // and leaves the lease cancellations queued.
    let mut cursor = crate::guest_completions::CompletionDrain::default();
    let consumed = cells.drain(&mut cursor, 1, |event| {
        assert_eq!(event, crate::guest_completions::REPLAY_COMPLETION_TOKEN);
    });
    assert_eq!(consumed, 1);
    assert!(cells.has_ready(), "the budget's remainder stays queued");
    let (_, recorder) = owner.take_recording_storage().expect("cancelled recording");
    assert!(recorder.owned_pages.is_empty());
    assert_eq!(recorder.owned_pages.capacity(), 0);
    assert_eq!(owner.owned_pages.len(), 2);
    assert_eq!(owner.owned_pages.as_ptr(), allocation);
    assert!(!owner.maintain(), "notifications still require consumption");
    assert!(weak.upgrade().is_some());
    cells.drain(&mut cursor, 16, |_| {});
    assert!(!cells.has_ready());
    assert!(owner.maintain());
    assert!(weak.upgrade().is_none());
}

#[test]
fn failed_recording_retirement_waits_for_explicit_quiescence() {
    use crate::{encoder_data::PendingVbibRetention, page_box_pool::PageBoxPool};

    let pool = Box::leak(Box::new(PageBoxPool::new(65536)));
    let mut recorder = FrameRecorder::new();
    recorder.pagebox_pool = Some(pool);
    assert!(
        recorder
            .finish_record(Err(WireError::AllocationFailed))
            .is_err()
    );
    let mut frame = empty_frame();
    let page = PageBox::new_zeroed(12);
    let address = page.as_ptr();
    recorder.capture_vbib_retention(
        &mut frame.scratch,
        PendingVbibRetention {
            buffer_id: BufferId::new_unique(),
            page_box: page,
            last_submit_seq: 42,
        },
    );
    frame.recorder = Some(recorder);
    let Err((_, mut owner)) = FramePacket::new(frame) else {
        panic!("sticky recording failure")
    };
    assert!(owner.take_recording_storage().is_none());
    assert!(!owner.maintain());
    assert_eq!(owner.take_leases().count(), 0);
    assert!(pool.acquire(12).is_none());
    // SAFETY: the fixture has no earlier GPU work or native users of this retired allocation.
    unsafe { owner.cancel_unadopted() };
    assert!(!owner.maintain());
    owner.drain_test_completions(&mut crate::guest_completions::CompletionDrain::default());
    assert!(owner.maintain());
    assert_eq!(
        pool.acquire(12).expect("canceled original pages").as_ptr(),
        address
    );
}

#[test]
fn fixed_metadata_records_keep_alignment_across_chunk_rollover() {
    use crate::{encoder_packet::metadata::LayerPacingRecord, present::LayerPacing};

    let mut frame = empty_frame();
    frame.scratch = ScratchArena::with_chunk_size(64);
    let mut recorder = FrameRecorder::new();
    for layer in 1..=3 {
        recorder.capture_pacing(
            &mut frame.scratch,
            layer,
            LayerPacing {
                display_sync: true,
                max_fps: 60,
            },
        );
    }
    let mut owner = seal(frame, recorder);
    assert!(
        owner.operation_bytes().len() > 16,
        "fixture crosses command chunks"
    );
    let mut packet = admit(&mut owner);
    let mut seen = 0;
    let mut alignment_residues = 0;
    while replay(&mut packet, |command, _, _| {
        assert!(matches!(command.opcode(), EncoderOpcode::SetLayerPacing));
        let record = borrow::<LayerPacingRecord>(command.payload())?;
        seen += 1;
        assert_eq!(
            (record.layer, record.display_sync, record.max_fps),
            (seen, 1, 60)
        );
        alignment_residues |= 1 << (command.payload().as_ptr() as usize % 16);
        Ok(())
    })
    .unwrap()
    {}
    assert_eq!(seen, 3);
    assert_eq!(alignment_residues, 1 | (1 << 8));
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete: {error:?}")),
    );
    drain_all(&owner);
    assert!(owner.maintain());
}

#[test]
fn recovered_recording_storage_keeps_every_lease_vector_capacity() {
    let (mut owner, _query) = packet_with_leases();
    let owned_capacity = owner.owned_pages.capacity();
    let query_capacity = owner.queries.capacity();
    let mut packet = admit(&mut owner);
    let mut pages = Vec::new();
    let mut native_queries = Vec::new();
    let mut cache = QueryLeaseCache::default();
    while replay(&mut packet, |command, _, _| {
        consume_leases(&command, &mut pages, &mut native_queries, &mut cache)
    })
    .unwrap()
    {}
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}")),
    );
    drain_all(&owner);
    let handed_over: Vec<_> = owner.take_leases().collect();
    assert_eq!(handed_over.len(), 3);
    let (_, recorder) = owner.take_recording_storage().expect("completed recording");
    assert!(owned_capacity > 0 && query_capacity > 0);
    assert!(recorder.owned_pages.is_empty() && recorder.queries.is_empty());
    assert_eq!(recorder.owned_pages.capacity(), owned_capacity);
    assert_eq!(recorder.queries.capacity(), query_capacity);
    // The leases live on elsewhere; release them the way the fixture's native side would.
    drop(pages);
    drop(native_queries);
    cache.maintain();
    drain_all(&owner);
    let pool = &recorder.completion_pool;
    for mut lease in handed_over {
        assert!(lease.maintain());
        for slot in lease.into_slots().into_iter().flatten() {
            pool.recycle(slot);
        }
    }
}

fn replayed_draw_payloads(frame: FrameData, recorder: FrameRecorder) -> Vec<Vec<u8>> {
    let mut owner = seal(frame, recorder);
    let mut packet = admit(&mut owner);
    let mut payloads = Vec::new();
    while replay(&mut packet, |command, _, _| {
        assert!(matches!(command.opcode(), EncoderOpcode::Draw));
        payloads.push(command.payload().to_vec());
        Ok(())
    })
    .unwrap()
    {}
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete: {error:?}")),
    );
    drain_all(&owner);
    assert!(owner.maintain());
    payloads
}

fn single_stream_binding() -> crate::draw_data::StreamBinding {
    crate::draw_data::StreamBinding {
        stream: 5,
        buffer_id: BufferId::new_unique(),
        backing_ptr: 0x0001_2340,
        backing_len: 8192,
        backing_generation: 0x1122_3344_5566_7788,
        offset: 12,
        stride: 24,
        freq: 0x4000_0011,
    }
}

/// Record `draw` four times through the direct append, with a chunk that holds two records.
///
/// The first attempt finds no open region and the third a full one; both fall back to the
/// checked writer, which opens a region, so the frame spans two regions.
fn append_single_stream_draws<D: crate::encoder_draw::draw_record::SingleStreamDraw>(
    vertices: &crate::encoder_draw::draw_record::BoundVertices,
    draw: &D,
    record_bytes: usize,
) -> Vec<Vec<u8>> {
    let mut frame = empty_frame();
    frame.scratch = ScratchArena::with_chunk_size(2 * record_bytes + 16);
    let mut recorder = FrameRecorder::new();
    let mut appended = Vec::new();
    for _ in 0..4 {
        let used = frame.scratch.bytes_used();
        let count = recorder.len();
        let direct = recorder.try_append_single_stream(&mut frame.scratch, vertices, draw);
        if direct {
            assert_eq!(frame.scratch.bytes_used(), used + record_bytes as u64);
        } else {
            assert_eq!(frame.scratch.bytes_used(), used);
            assert_eq!(recorder.len(), count);
            recorder
                .record_bound_draw(&mut frame.scratch, draw.prefix(), vertices, draw.index())
                .unwrap();
        }
        assert_eq!(recorder.len(), count + 1);
        appended.push(direct);
    }
    assert_eq!(appended, [false, true, false, true]);
    assert_eq!(
        frame.scratch.chunk_count(),
        2,
        "fixture must cross a region boundary"
    );
    replayed_draw_payloads(frame, recorder)
}

#[test]
fn single_stream_append_matches_the_draw_op_encoding_across_regions() {
    use mtld3d_shared::{
        command_header::COMMAND_HEADER_BYTES,
        mtl::{IndexType, PrimitiveType},
    };

    use crate::{
        draw_data::{DrawOp, ExtraStreams, IndexSource, VertexSource},
        encoder_draw::draw_record::{
            BoundVertices, IndexBuffer, IndexedDraw, NonindexedDraw, SINGLE_BOUND_BYTES,
            SINGLE_INDEXED_BOUND_BYTES, StreamRecord,
        },
    };

    for index_type in [None, Some(IndexType::UInt16), Some(IndexType::UInt32)] {
        let binding = single_stream_binding();
        let stream0_freq = 0x8000_0003;
        let index_buffer = BufferId::new_unique();
        let draw_op = DrawOp {
            metal_prim: PrimitiveType::TriangleStrip,
            vertex_source: VertexSource::Bound {
                first: binding.copy_value(),
                extra: ExtraStreams::EMPTY,
                stream0_freq,
            },
            index_source: index_type.map_or(
                IndexSource::None {
                    start_vertex: u32::MAX - 9,
                    vertex_count: 9,
                },
                |index_type| IndexSource::Bound {
                    buffer_id: index_buffer,
                    backing_ptr: 0x0005_6780,
                    backing_len: 4096,
                    backing_generation: 123,
                    offset: 16,
                    index_count: 9,
                    index_type,
                    base_vertex: i32::MIN,
                },
            ),
        };
        let mut frame = empty_frame();
        let mut recorder = FrameRecorder::new();
        for _ in 0..4 {
            recorder.record_draw(&mut frame.scratch, &draw_op).unwrap();
        }
        let expected = replayed_draw_payloads(frame, recorder);

        let vertices = BoundVertices {
            first: StreamRecord::from_binding(&binding),
            extra: ExtraStreams::EMPTY,
            stream0_freq,
        };
        let actual = index_type.map_or_else(
            || {
                append_single_stream_draws(
                    &vertices,
                    &NonindexedDraw {
                        primitive: PrimitiveType::TriangleStrip,
                        start_vertex: u32::MAX - 9,
                        vertex_count: 9,
                    },
                    COMMAND_HEADER_BYTES + SINGLE_BOUND_BYTES,
                )
            },
            |index_type| {
                append_single_stream_draws(
                    &vertices,
                    &IndexedDraw {
                        primitive: PrimitiveType::TriangleStrip,
                        base_vertex: i32::MIN,
                        index_count: 9,
                        index: &IndexBuffer {
                            buffer: index_buffer.raw(),
                            address: 0x0005_6780,
                            length: 4096,
                            generation: 123,
                            offset: 16,
                            kind: index_type as u8,
                            reserved: [0; 3],
                        },
                    },
                    COMMAND_HEADER_BYTES + SINGLE_INDEXED_BOUND_BYTES,
                )
            },
        );
        assert_eq!(actual.len(), 4);
        assert_eq!(actual, expected);
    }
}

#[test]
fn single_stream_append_declines_latched_errors_and_extra_streams_without_writing() {
    use mtld3d_shared::mtl::PrimitiveType;

    use crate::{
        draw_data::ExtraStreams,
        encoder_draw::draw_record::{
            BoundVertices, NonindexedDraw, SingleStreamDraw, StreamRecord,
        },
    };

    let draw = NonindexedDraw {
        primitive: PrimitiveType::Triangle,
        start_vertex: 3,
        vertex_count: 6,
    };
    let single = BoundVertices {
        first: StreamRecord::from_binding(&single_stream_binding()),
        extra: ExtraStreams::EMPTY,
        stream0_freq: 1,
    };
    for extra in [
        ExtraStreams::Owned(Box::new([])),
        ExtraStreams::Owned(Box::new([single_stream_binding()])),
    ] {
        let mut scratch = ScratchArena::new();
        let mut recorder = FrameRecorder::new();
        recorder
            .record_bound_draw(&mut scratch, draw.prefix(), &single, None)
            .unwrap();
        let vertices = BoundVertices {
            first: StreamRecord::from_binding(&single_stream_binding()),
            extra,
            stream0_freq: 1,
        };
        let used = scratch.bytes_used();
        assert!(!recorder.try_append_single_stream(&mut scratch, &vertices, &draw));
        assert_eq!((scratch.bytes_used(), recorder.len()), (used, 1));
        recorder
            .record_bound_draw(&mut scratch, draw.prefix(), &vertices, None)
            .unwrap();
        assert_eq!(recorder.len(), 2);
    }

    for error in [WireError::AllocationFailed, WireError::InvalidValue] {
        let mut scratch = ScratchArena::new();
        let mut recorder = FrameRecorder::new();
        recorder
            .record_bound_draw(&mut scratch, draw.prefix(), &single, None)
            .unwrap();
        assert!(recorder.try_append_single_stream(&mut scratch, &single, &draw));
        assert_eq!(recorder.finish_record(Err(error)), Err(error));
        let used = scratch.bytes_used();
        assert!(!recorder.try_append_single_stream(&mut scratch, &single, &draw));
        assert_eq!((scratch.bytes_used(), recorder.len()), (used, 2));
        assert_eq!(
            recorder.record_bound_draw(&mut scratch, draw.prefix(), &single, None),
            Err(error)
        );
        assert_eq!((scratch.bytes_used(), recorder.len()), (used, 2));
        assert_eq!(recorder.recording_error(), Some(error));
    }
}

#[test]
fn cursor_borrows_typed_payloads_at_both_eight_byte_positions() {
    use super::replay::CommandCursor;
    use crate::encoder_records::{IdRecord, borrow, write};

    let mut arena = ScratchArena::with_chunk_size(64);
    arena.push_fixed_record(1, 0, 0, |_| Ok(())).unwrap();
    let ordinary = arena.alloc(&[9; 3]);
    assert_eq!(ordinary % 16, 0);
    arena
        .push_fixed_record(2, 0, size_of::<IdRecord>(), |bytes| {
            write(
                bytes,
                IdRecord {
                    id: 0x1234_5678_9abc_def0,
                },
            )
        })
        .unwrap();
    arena.push_fixed_record(3, 0, 0, |_| Ok(())).unwrap();
    arena
        .push_fixed_record(4, 0, size_of::<IdRecord>(), |bytes| {
            write(bytes, IdRecord { id: 17 })
        })
        .unwrap();
    arena.publish_command_region().unwrap();
    // SAFETY: the arena retains its table and every immutable region through iteration.
    let mut cursor = unsafe { CommandCursor::new(arena.command_descriptor_bytes()) }.unwrap();
    for (opcode, id, payload_modulo) in [
        (1, None, 8),
        (2, Some(0x1234_5678_9abc_def0), 0),
        (3, None, 0),
        (4, Some(17), 8),
    ] {
        // SAFETY: the authentic command table and initialized regions remain unchanged above.
        let record = unsafe { cursor.next_record() }.unwrap().unwrap();
        assert_eq!(record.opcode, opcode);
        assert_eq!(record.payload.as_ptr() as usize % 16, payload_modulo);
        if let Some(id) = id {
            assert_eq!(borrow::<IdRecord>(record.payload).unwrap().id, id);
        } else {
            assert!(record.payload.is_empty());
        }
    }
    assert!(cursor.is_complete());
    // SAFETY: the same retained table remains valid when checking its end.
    assert!(unsafe { cursor.next_record() }.unwrap().is_none());
}

#[test]
fn publish_counts_commands_appended_through_the_arena_slot() {
    use super::replay::CommandCursor;
    use crate::encoder_records::{IdRecord, borrow};

    let mut arena = ScratchArena::with_chunk_size(64);
    arena.push_fixed_record(1, 0, 0, |_| Ok(())).unwrap();
    arena
        .command_slot::<IdRecord>()
        .unwrap()
        .write(2, 0, IdRecord { id: 41 });
    arena.publish_command_region().unwrap();
    assert_eq!(
        arena
            .command_ranges()
            .map(|(_, used)| used)
            .collect::<Vec<_>>(),
        [24]
    );
    // SAFETY: the arena retains its table and every immutable region through iteration.
    let mut cursor = unsafe { CommandCursor::new(arena.command_descriptor_bytes()) }.unwrap();
    // SAFETY: the authentic command table and initialized regions remain unchanged above.
    let first = unsafe { cursor.next_record() }.unwrap().unwrap();
    assert_eq!((first.opcode, first.payload.len()), (1, 0));
    // SAFETY: the same retained table remains valid for the second record.
    let second = unsafe { cursor.next_record() }.unwrap().unwrap();
    assert_eq!(second.opcode, 2);
    assert_eq!(borrow::<IdRecord>(second.payload).unwrap().id, 41);
    assert!(cursor.is_complete());
}
