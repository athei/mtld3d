use std::sync::{Arc, Weak};

use mtld3d_shared::{MetalHandle, mtl::PixelFormat, record_handle::DeviceRecordHandle};

use super::*;
use crate::{
    encoder_data::{AdoptProgramOp, BeginVisibilityOp, FrameData, FrameInit, Op},
    ids::BufferId,
    page_box::PageBox,
    render_scale::RenderScale,
    visibility::VisibilityQueryCore,
};

pub(super) fn frame_with_leases() -> (FrameData, Weak<VisibilityQueryCore>) {
    let mut frame = FrameData::new(&FrameInit {
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
    });
    frame.ops.push(Op::StageUpload {
        buffer_id: BufferId::new_unique(),
        page_box: PageBox::new_zeroed(4),
        dst_offset: 0,
        size: 4,
    });
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    frame
        .ops
        .push(Op::BeginVisibility(Box::new(BeginVisibilityOp {
            generation: 1,
            c: query,
        })));
    frame.ops.push(Op::AdoptProgram(Box::new(AdoptProgramOp {
        registration: 41,
    })));
    frame.ops.push(Op::StageUpload {
        buffer_id: BufferId::new_unique(),
        page_box: PageBox::new_zeroed(4),
        dst_offset: 0,
        size: 4,
    });
    (frame, weak)
}

fn packet_with_leases() -> (FramePacket, Weak<VisibilityQueryCore>) {
    let (frame, weak) = frame_with_leases();
    (
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("valid fixture: {error:?}")),
        weak,
    )
}

fn record_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        offsets.push(at);
        let length = u32::from_le_bytes(bytes[at + 2..at + 6].try_into().expect("record length"));
        at += 6 + usize::try_from(length).expect("length fits usize");
    }
    assert_eq!(at, bytes.len());
    offsets
}

fn rejects_without_adopting(mutate: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>)) {
    rejects_wire(false, mutate);
}

fn rejects_wire(table: bool, mutate: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>)) {
    let (packet, weak_query) = packet_with_leases();
    rejects_packet(packet, &weak_query, table, mutate);
}

fn rejects_packet(
    mut packet: FramePacket,
    weak_query: &Weak<VisibilityQueryCore>,
    table: bool,
    mutate: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>),
) {
    let page_count = packet.pages.len();
    let mut completions = crate::guest_completions::CompletionDrain::default();
    let mut metadata = packet.metadata_bytes().to_vec();
    let mut operations = if table {
        packet.operation_bytes().to_vec()
    } else {
        packet
            .recorder
            .as_ref()
            .expect("recorder")
            .slab
            .ranges()
            .flat_map(|(address, length)| {
                let length = usize::try_from(length).expect("record span fits usize");
                // SAFETY: this fixture's retained frame arena owns every unmodified recorded
                // span, and the copy finishes before any decode or packet maintenance.
                unsafe { core::slice::from_raw_parts(address as *const u8, length) }
            })
            .copied()
            .collect()
    };
    mutate(&mut metadata, &mut operations);
    let mut queries = QueryLeaseCache::default();
    let mut resolutions = 0;
    let decoded = if table {
        // SAFETY: original metadata retains every owner; invalid chunk addresses are
        // rejected against its inventory before a slice can be constructed.
        unsafe {
            decode_packet(
                &metadata,
                &operations,
                packet.completion_address(),
                &mut queries,
                |_| {
                    resolutions += 1;
                    Err(WireError::InvalidValue)
                },
            )
        }
    } else {
        // SAFETY: the copied immutable record bytes retain the original packet's valid
        // owners. Invalid scalar/range fields must fail validation before adoption.
        unsafe {
            decode_chunks(
                &metadata,
                &[&operations],
                packet.completion_address(),
                &mut queries,
                |_| {
                    resolutions += 1;
                    Err(WireError::InvalidValue)
                },
            )
        }
    };
    assert!(
        decoded.is_err(),
        "the entire malformed packet must be rejected"
    );
    assert_eq!(resolutions, 0, "validation must precede program adoption");
    assert_eq!(
        packet.drain_test_completions(&mut completions),
        0,
        "validation must not publish any pooled ownership transition"
    );
    assert_eq!(packet.pages.len(), page_count);
    assert!(
        packet.pages.iter_mut().all(|lease| !lease.maintain()),
        "no page may be adopted and dropped during validation"
    );
    assert_eq!(packet.queries.len(), 1);
    assert!(
        packet.queries.iter().all(|lease| !lease.completed()),
        "no query may be adopted and dropped during validation"
    );
    assert!(
        weak_query.upgrade().is_some(),
        "packet still owns the guest query"
    );
    // SAFETY: rejection returned no native frame and assertions establish zero adoption.
    // No other decoder or replay consumer exists in this test.
    unsafe { packet.cancel_unadopted() };
    let notifications = packet.drain_test_completions(&mut completions);
    assert!(
        notifications > page_count,
        "cancellation must retire every page and query"
    );
    assert!(
        packet.maintain(),
        "rejection releases every lease without waiting"
    );
    drop(packet);
    queries.maintain();
    assert!(
        weak_query.upgrade().is_none(),
        "no canceled guest query leaked"
    );
}

#[test]
fn later_unknown_tag_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| {
        let at = *record_offsets(bytes).last().expect("last record");
        bytes[at..at + 2].copy_from_slice(&u16::MAX.to_le_bytes());
    });
}

#[test]
fn later_overflowing_length_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| {
        let at = *record_offsets(bytes).last().expect("last record");
        bytes[at + 2..at + 6].copy_from_slice(&u32::MAX.to_le_bytes());
    });
}

#[test]
fn later_truncated_payload_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| {
        bytes.pop().expect("nonempty packet");
    });
}

#[test]
fn trailing_partial_record_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| bytes.push(1));
}

#[test]
fn later_page_range_overflow_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| {
        // StageUpload is buffer ID, destination offset, size, then its page descriptor.
        let descriptor = *record_offsets(bytes).last().expect("last record") + 6 + 16;
        bytes[descriptor..descriptor + 8].copy_from_slice(&(u64::MAX - 4095).to_le_bytes());
    });
}

#[test]
fn later_page_logical_range_does_not_adopt_earlier_owners() {
    rejects_without_adopting(|_, bytes| {
        let descriptor = *record_offsets(bytes).last().expect("last record") + 6 + 16;
        bytes[descriptor + 16..descriptor + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    });
}

#[test]
fn later_changed_page_generation_is_not_a_published_lease() {
    rejects_without_adopting(|_, bytes| {
        let descriptor = *record_offsets(bytes).last().expect("last record") + 6 + 16;
        let generation = u64::from_le_bytes(
            bytes[descriptor + 24..descriptor + 32]
                .try_into()
                .expect("generation"),
        );
        bytes[descriptor + 24..descriptor + 32]
            .copy_from_slice(&generation.wrapping_add(1).to_le_bytes());
    });
}

#[test]
fn truncated_metadata_does_not_adopt_operation_owners() {
    rejects_without_adopting(|metadata, _| {
        metadata.pop().expect("metadata exists");
    });
}

#[test]
fn trailing_metadata_does_not_adopt_operation_owners() {
    rejects_without_adopting(|metadata, _| metadata.push(0));
}

#[test]
fn valid_packet_adopts_each_owner_and_releases_it_after_native_frame_drop() {
    let (mut packet, weak_query) = packet_with_leases();
    let mut queries = QueryLeaseCache::default();
    let mut resolutions = 0;
    let tokens = [
        0xFFFF_0200,
        0x0200_0001,
        0x800F_0800,
        0xA0E4_0000,
        0x0000_FFFF,
    ];
    // SAFETY: this test retains the sole packet while its one decoder and frame run.
    unsafe { packet.mark_admitted() };
    // SAFETY: the packet and its original immutable buffers remain retained until native drop.
    let decoded = unsafe {
        decode_packet(
            packet.metadata_bytes(),
            packet.operation_bytes(),
            packet.completion_address(),
            &mut queries,
            |registration| {
                assert_eq!(registration, 41);
                resolutions += 1;
                Ok((
                    ProgramId::from_tokens(&tokens),
                    crate::dxso::parse(&tokens).expect("valid shader"),
                ))
            },
        )
    }
    .expect("uncorrupted fixture must decode");
    assert_eq!(resolutions, 1);
    assert_eq!(decoded.ops.len(), 4);
    assert!(!packet.maintain(), "native frame still retains the packet");
    assert!(weak_query.upgrade().is_some());
    drop(decoded);
    let mut completions = crate::guest_completions::CompletionDrain::default();
    assert!(packet.drain_test_completions(&mut completions) > 0);
    assert!(
        packet.maintain(),
        "native frame drop acknowledges all decoded owners"
    );
    drop(packet);
    queries.maintain();
    assert!(weak_query.upgrade().is_none());
}

#[test]
fn duplicate_page_publication_cannot_adopt_one_completion_twice() {
    rejects_without_adopting(|_, bytes| {
        let offsets = record_offsets(bytes);
        let duplicate = bytes[offsets[0]..offsets[1]].to_vec();
        bytes.extend_from_slice(&duplicate);
    });
}

#[test]
fn duplicate_program_registration_is_rejected_before_registry_consumption() {
    rejects_without_adopting(|_, bytes| {
        let offsets = record_offsets(bytes);
        let duplicate = bytes[offsets[2]..offsets[3]].to_vec();
        bytes.extend_from_slice(&duplicate);
    });
}

#[test]
fn null_empty_chunk_is_rejected_before_constructing_a_slice() {
    rejects_wire(true, |_, table| {
        table[6..14].copy_from_slice(&0_u64.to_le_bytes());
        table[14..18].copy_from_slice(&0_u32.to_le_bytes());
    });
}

#[test]
fn chunk_outside_retained_inventory_is_rejected_before_dereference() {
    rejects_wire(true, |_, table| {
        table[6..14].copy_from_slice(&1_u64.to_le_bytes());
    });
}

#[test]
fn overflowing_chunk_range_is_rejected_before_dereference() {
    rejects_wire(true, |_, table| {
        table[6..14].copy_from_slice(&(u64::MAX - 7).to_le_bytes());
    });
}

#[test]
fn invalid_chunk_table_tag_rejects_the_packet() {
    rejects_wire(true, |_, table| {
        table[..2].copy_from_slice(&u16::MAX.to_le_bytes());
    });
}

#[test]
fn trailing_chunk_table_bytes_reject_the_packet() {
    rejects_wire(true, |_, table| table.push(0));
}

fn packet_with_upload(
    mutate: impl FnOnce(&mut TextureUploadJob),
) -> (
    FramePacket,
    Weak<VisibilityQueryCore>,
    Weak<PageBox>,
    Weak<crate::upload_redirty::RedirtyQueue>,
) {
    upload_packet_with_source(64, mutate)
}

fn upload_packet_with_source(
    source_bytes: usize,
    mutate: impl FnOnce(&mut TextureUploadJob),
) -> (
    FramePacket,
    Weak<VisibilityQueryCore>,
    Weak<PageBox>,
    Weak<crate::upload_redirty::RedirtyQueue>,
) {
    use mtld3d_shared::mtl::{Swizzle, TextureCreateFlags, TextureUsage};
    use mtld3d_types::D3DFMT_A8R8G8B8;

    let (mut frame, query) = frame_with_leases();
    let backing = Arc::new(PageBox::new_zeroed(source_bytes));
    let weak_page = Arc::downgrade(&backing);
    let redirty = Arc::new(crate::upload_redirty::RedirtyQueue::new());
    let weak_feedback = Arc::downgrade(&redirty);
    let mut job = TextureUploadJob {
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
    };
    mutate(&mut job);
    frame
        .ops
        .push(Op::UploadTexture(Box::new(UploadTextureOp { job })));
    (
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("upload fixture: {error:?}")),
        query,
        weak_page,
        weak_feedback,
    )
}

fn rejects_upload(mutate: impl FnOnce(&mut TextureUploadJob)) {
    let (packet, query, page, feedback) = packet_with_upload(mutate);
    assert_eq!(packet.pages.len(), 3);
    rejects_packet(packet, &query, true, |_, _| {});
    assert!(page.upgrade().is_none(), "canceled upload page leaked");
    assert!(
        feedback.upgrade().is_none(),
        "canceled feedback owner leaked"
    );
}

#[test]
fn later_upload_pitch_exceeding_logical_source_is_rejected_before_adoption() {
    // The page is padded to a larger mapping, but only its 64 logical bytes are valid.
    rejects_upload(|job| job.src_pitch = 32);
}

#[test]
fn later_upload_row_extent_exceeding_source_pitch_is_rejected_before_adoption() {
    rejects_upload(|job| job.region_w = 5);
}

#[test]
fn later_upload_depth_exceeding_texture_is_rejected_before_adoption() {
    rejects_upload(|job| job.depth = 2);
}

#[test]
fn later_volume_upload_slice_span_exceeding_source_is_rejected_before_adoption() {
    rejects_upload(|job| {
        job.info.depth = 2;
        job.depth = 2;
        job.slice_pitch = u32::MAX;
    });
}

#[test]
fn valid_upload_packet_releases_source_and_feedback_after_native_drop() {
    valid_upload_releases(upload_packet_with_source(64, |_| {}));
}

fn valid_upload_releases(
    fixture: (
        FramePacket,
        Weak<VisibilityQueryCore>,
        Weak<PageBox>,
        Weak<crate::upload_redirty::RedirtyQueue>,
    ),
) {
    let (mut packet, query, page, feedback) = fixture;
    let mut queries = QueryLeaseCache::default();
    let tokens = [
        0xFFFF_0200,
        0x0200_0001,
        0x800F_0800,
        0xA0E4_0000,
        0x0000_FFFF,
    ];
    // SAFETY: this test owns the sole retained packet and its only decoder.
    unsafe { packet.mark_admitted() };
    // SAFETY: all original immutable buffers and owners live through native frame drop.
    let decoded = unsafe {
        decode_packet(
            packet.metadata_bytes(),
            packet.operation_bytes(),
            packet.completion_address(),
            &mut queries,
            |registration| {
                assert_eq!(registration, 41);
                Ok((
                    ProgramId::from_tokens(&tokens),
                    crate::dxso::parse(&tokens).expect("valid shader"),
                ))
            },
        )
    }
    .expect("valid upload must decode");
    assert_eq!(decoded.ops.len(), 5);
    assert!(page.upgrade().is_some());
    assert!(feedback.upgrade().is_some());
    assert!(!packet.maintain());
    drop(decoded);
    let mut completions = crate::guest_completions::CompletionDrain::default();
    assert!(packet.drain_test_completions(&mut completions) > 0);
    assert!(packet.maintain());
    drop(packet);
    queries.maintain();
    assert!(query.upgrade().is_none());
    assert!(page.upgrade().is_none());
    assert!(feedback.upgrade().is_none());
}

fn planar_upload(job: &mut TextureUploadJob) {
    job.info.d3d_format = mtld3d_types::D3DFMT_NV12;
    job.src_d3d_format = mtld3d_types::D3DFMT_NV12;
    job.info.height = 6;
    job.region_h = 6;
    job.src_pitch = 4;
    job.bytes_per_pixel = 1;
    job.slice_pitch = 24;
}

#[test]
fn valid_planar_upload_includes_chroma_storage_rows() {
    valid_upload_releases(upload_packet_with_source(24, planar_upload));
}

#[test]
fn planar_upload_missing_chroma_rows_is_rejected_before_adoption() {
    let (packet, query, page, feedback) = upload_packet_with_source(16, planar_upload);
    rejects_packet(packet, &query, true, |_, _| {});
    assert!(page.upgrade().is_none());
    assert!(feedback.upgrade().is_none());
}

#[test]
fn large_color_upload_borrows_retained_arena_payload_without_inline_copy() {
    let (mut frame, _) = frame_with_leases();
    frame.ops.clear();
    let mut source = vec![0x5a; 8192];
    let address = frame.scratch.alloc(&source);
    let pointer = core::ptr::NonNull::new(address as *mut u8).expect("arena allocation");
    // SAFETY: the initialized allocation stays immutable in the retained frame arena.
    let bytes = unsafe { crate::draw_data::ScratchSlice::from_raw_parts(pointer, 8192) };
    source.fill(0);
    frame.ops.push(Op::UploadColor(Box::new(UploadColorOp {
        color_handle: 1,
        bytes,
        width: 32,
        height: 64,
        src_stride: 128,
    })));
    let mut packet = FramePacket::new(frame)
        .unwrap_or_else(|(error, _)| panic!("large upload fixture: {error:?}"));
    let encoded_bytes: u64 = packet
        .recorder
        .as_ref()
        .expect("recorder")
        .slab
        .ranges()
        .map(|(_, length)| length)
        .sum();
    assert!(
        encoded_bytes < 128,
        "payload must not be copied into operation records"
    );
    let mut queries = QueryLeaseCache::default();
    // SAFETY: this test retains the sole packet through its only native consumer.
    unsafe { packet.mark_admitted() };
    // SAFETY: original metadata, command spans and payload remain owned by the packet.
    let decoded = unsafe {
        decode_packet(
            packet.metadata_bytes(),
            packet.operation_bytes(),
            packet.completion_address(),
            &mut queries,
            |_| panic!("fixture contains no shader registration"),
        )
    }
    .expect("large upload decodes");
    let Op::UploadColor(upload) = &decoded.ops[0] else {
        panic!("color upload expected")
    };
    assert_eq!(upload.bytes.as_raw(), (address, 8192));
    assert!(upload.bytes.as_slice().iter().all(|byte| *byte == 0x5a));
    assert!(!packet.maintain());
    drop(decoded);
    assert!(packet.maintain());
}

#[test]
fn latched_recording_failure_retains_unpublished_operation_owner() {
    let mut recorder = FrameRecorder::new();
    let mut arena = ScratchArena::default();
    let query = VisibilityQueryCore::new();
    let weak = Arc::downgrade(&query);
    recorder.error = Some(WireError::TooLarge);
    let result = recorder.try_record(
        &mut arena,
        Op::BeginVisibility(Box::new(BeginVisibilityOp {
            generation: 1,
            c: query,
        })),
    );
    assert!(matches!(result, Err(WireError::TooLarge)));
    assert!(
        weak.upgrade().is_some(),
        "failed recording must retain the operation owner"
    );
    assert_eq!(recorder.rejected_ops.len(), 1);
    assert!(recorder.slab.ranges().next().is_none());
    // No operation was published, so this fixture is already quiescent.
    drop(recorder);
    assert!(weak.upgrade().is_none());
}

#[test]
fn command_chunk_cannot_alias_retained_query_mailbox() {
    let (packet, query) = packet_with_leases();
    let mailbox = packet.queries[0].descriptor().wire_fields()[0];
    rejects_packet(packet, &query, true, |_, table| {
        table[6..14].copy_from_slice(&mailbox.to_le_bytes());
        table[14..18].copy_from_slice(&4_u32.to_le_bytes());
    });
}

#[test]
fn command_chunk_cannot_alias_identical_bytes_in_payload_allocation() {
    let (mut frame, query) = frame_with_leases();
    let mut recorder = FrameRecorder::new();
    for operation in core::mem::take(&mut frame.ops) {
        recorder.record(&mut frame.scratch, operation);
    }
    let (address, length) = recorder.slab.ranges().next().expect("first command span");
    // SAFETY: recording retained the initialized span in this still-live frame arena.
    let original = unsafe {
        core::slice::from_raw_parts(address as *const u8, usize::try_from(length).unwrap())
    };
    let duplicate = original.to_vec();
    let payload_address = frame.scratch.alloc(&duplicate);
    assert_ne!(address, payload_address);
    frame.recorder = Some(recorder);
    let packet = FramePacket::new(frame)
        .unwrap_or_else(|(error, _)| panic!("duplicate payload fixture: {error:?}"));
    rejects_packet(packet, &query, true, |_, table| {
        // These bytes are valid commands in an authenticated immutable allocation,
        // but that allocation was published as payload, not this command span.
        table[6..14].copy_from_slice(&payload_address.to_le_bytes());
    });
}

#[test]
fn scratch_payload_cannot_alias_retained_query_mailbox() {
    let (mut frame, query) = frame_with_leases();
    let address = frame.scratch.alloc(&[0x55; 4]);
    let pointer = core::ptr::NonNull::new(address as *mut u8).expect("arena allocation");
    // SAFETY: the frame retains this initialized immutable four-byte payload.
    let bytes = unsafe { crate::draw_data::ScratchSlice::from_raw_parts(pointer, 4) };
    frame.ops.push(Op::UploadColor(Box::new(UploadColorOp {
        color_handle: 1,
        bytes,
        width: 1,
        height: 1,
        src_stride: 4,
    })));
    let packet =
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("scratch fixture: {error:?}"));
    let mailbox = packet.queries[0].descriptor().wire_fields()[0];
    rejects_packet(packet, &query, false, |_, operations| {
        // UploadColor begins with the texture handle then the scratch pointer and size.
        let pointer_at = record_offsets(operations).last().unwrap() + 6 + 8;
        operations[pointer_at..pointer_at + 8].copy_from_slice(&mailbox.to_le_bytes());
    });
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
        .chain(packet.queries.iter().filter_map(GuestQueryLease::token))
        .collect();
    original.sort_unstable();
    assert_eq!(original.len(), 3);
    let mut cursor = crate::guest_completions::CompletionDrain::default();
    // SAFETY: this packet was never exposed to a native consumer.
    unsafe { packet.cancel_unadopted() };
    assert_eq!(packet.drain_test_completions(&mut cursor), 3);
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
fn command_and_payload_cannot_alias_reply_or_readback_destinations() {
    use std::sync::atomic::{AtomicU32, Ordering};

    for command in [true, false] {
        for reply in [true, false] {
            let (mut frame, query) = frame_with_leases();
            let mut destination = [0xa5_u8; 64];
            let destination_address = destination.as_mut_ptr() as u64;
            let done = Arc::new(AtomicU32::new(0));
            let done_address = Arc::as_ptr(&done) as u64;
            frame
                .ops
                .push(Op::ReadDeviceBuffer(Box::new(ReadDeviceBufferOp {
                    done: Arc::clone(&done).into(),
                    buffer_id: BufferId::new_unique(),
                    dst_ptr: destination_address,
                    dst_len: 64,
                })));
            let address = frame.scratch.alloc(&[0x55; 4]);
            let pointer = core::ptr::NonNull::new(address as *mut u8).expect("arena allocation");
            // SAFETY: the owning frame retains this initialized immutable payload.
            let bytes = unsafe { crate::draw_data::ScratchSlice::from_raw_parts(pointer, 4) };
            frame.ops.push(Op::UploadColor(Box::new(UploadColorOp {
                color_handle: 1,
                bytes,
                width: 1,
                height: 1,
                src_stride: 4,
            })));
            let packet = FramePacket::new(frame)
                .unwrap_or_else(|(error, _)| panic!("mutable destination fixture: {error:?}"));
            let alias = if reply {
                done_address
            } else {
                destination_address
            };
            rejects_packet(packet, &query, command, |_, records| {
                if command {
                    records[6..14].copy_from_slice(&alias.to_le_bytes());
                    records[14..18].copy_from_slice(&4_u32.to_le_bytes());
                } else {
                    let pointer_at = record_offsets(records).last().unwrap() + 6 + 8;
                    records[pointer_at..pointer_at + 8].copy_from_slice(&alias.to_le_bytes());
                }
            });
            assert_eq!(done.load(Ordering::Relaxed), 0);
            assert_eq!(destination, [0xa5; 64]);
        }
    }
}

#[test]
fn early_native_lease_events_do_not_retire_a_pending_or_rejected_packet() {
    let (mut frame, _) = frame_with_leases();
    frame.ops.clear();
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
    let lease = GuestPageLease::for_read_pooled(read, &pool);
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
fn parsed_metadata_moves_into_frame_without_reading_metadata_again() {
    let (mut frame, _) = frame_with_leases();
    frame.ops.clear();
    frame.submit_seq = 91;
    let mut packet = FramePacket::new(frame)
        .unwrap_or_else(|(error, _)| panic!("valid metadata fixture: {error:?}"));
    let bytes = packet.metadata_bytes().to_vec();
    // SAFETY: the packet owns all described allocations throughout this decode.
    let parsed = unsafe { parse_metadata(&bytes) }.unwrap();
    assert_eq!(parsed.frame.submit_seq, 91);
    drop(bytes);
    let replay = prepare_chunks(
        parsed,
        Vec::new(),
        ReplayCompletion {
            address: packet.completion_address(),
            rejected: true,
        },
        |_| Err(WireError::InvalidValue),
    )
    .unwrap();
    let decoded = reconstruct_packet(replay, &mut QueryLeaseCache::default()).unwrap();
    assert_eq!(decoded.submit_seq, 91);
    // SAFETY: decoding succeeded and this test owns the sole native frame.
    unsafe { packet.mark_admitted() };
    drop(decoded);
    assert!(packet.maintain());
}

#[test]
fn borrowed_draw_and_constant_records_preserve_order_and_payload_leases() {
    use mtld3d_shared::mtl::PrimitiveType;

    use crate::draw_data::{DrawOp, IndexSource, VertexSource, arena_alloc_bytes};

    let (mut frame, _) = frame_with_leases();
    frame.ops.clear();
    // SAFETY: frame owns this immutable capture through packet and decoded-frame teardown.
    let data = unsafe { arena_alloc_bytes(frame.scratch_mut(), &[0x5a; 64]) };
    frame.ops.extend([
        Op::SetVsConstRange {
            start_row: 3,
            rows: 4,
            data,
        },
        Op::SetPsConstRange {
            start_row: 5,
            rows: 4,
            data,
        },
        Op::SetFfVsConstRange {
            start_row: 7,
            rows: 4,
            data,
        },
        Op::Draw(DrawOp {
            metal_prim: PrimitiveType::Triangle,
            vertex_source: VertexSource::Up {
                bytes: data,
                size: 64,
                stride: 16,
            },
            index_source: IndexSource::None {
                start_vertex: 0,
                vertex_count: 3,
            },
        }),
    ]);
    let mut packet = FramePacket::new(frame)
        .unwrap_or_else(|(error, _)| panic!("valid borrowed records: {error:?}"));
    let mut queries = QueryLeaseCache::default();
    // SAFETY: the fixture retains this sole packet through its native consumer.
    unsafe { packet.mark_admitted() };
    // SAFETY: packet retains all authentic metadata, command and payload allocations.
    let decoded = unsafe {
        decode_packet(
            packet.metadata_bytes(),
            packet.operation_bytes(),
            packet.completion_address(),
            &mut queries,
            |_| panic!("no program registration"),
        )
    }
    .unwrap();
    assert_eq!(decoded.ops.len(), 4);
    for (op, expected_row) in decoded.ops[..3].iter().zip([3, 5, 7]) {
        let (start_row, rows, captured) = match op {
            Op::SetVsConstRange {
                start_row,
                rows,
                data,
            }
            | Op::SetPsConstRange {
                start_row,
                rows,
                data,
            }
            | Op::SetFfVsConstRange {
                start_row,
                rows,
                data,
            } => (*start_row, *rows, data),
            _ => panic!("constant operation expected"),
        };
        assert_eq!((start_row, rows), (expected_row, 4));
        assert_eq!(captured.as_raw(), data.as_raw());
        assert_eq!(captured.as_slice(), &[0x5a; 64]);
    }
    assert!(matches!(decoded.ops[0], Op::SetVsConstRange { .. }));
    assert!(matches!(decoded.ops[1], Op::SetPsConstRange { .. }));
    assert!(matches!(decoded.ops[2], Op::SetFfVsConstRange { .. }));
    let Op::Draw(draw) = &decoded.ops[3] else {
        panic!("draw expected")
    };
    let VertexSource::Up {
        bytes,
        size,
        stride,
    } = &draw.vertex_source
    else {
        panic!("UP source expected")
    };
    assert_eq!((*size, *stride), (64, 16));
    assert_eq!(bytes.as_raw(), data.as_raw());
    assert!(!packet.maintain());
    drop(decoded);
    assert!(packet.maintain());
}

#[test]
fn streamed_packet_retains_recording_until_submit_owner_drops() {
    let (mut frame, weak) = frame_with_leases();
    frame.ops.retain(|op| !matches!(op, Op::AdoptProgram(_)));
    let mut owner =
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("fixture: {error:?}"));
    // SAFETY: the admitted fixture owner retains all immutable command bytes and leases.
    unsafe {
        owner.mark_admitted();
    }
    // SAFETY: owner retains the immutable matched-producer packet until completion.
    let mut packet = unsafe {
        prepare_packet(
            owner.metadata_bytes(),
            owner.operation_bytes(),
            owner.completion_address(),
            |_| unreachable!(),
        )
        .unwrap()
    };
    assert!(
        packet.frame().ops.is_empty(),
        "native stream must not materialize an op vector"
    );
    let mut queries = QueryLeaseCache::default();
    let mut count = 0;
    while let Some(op) = packet.next_op(&mut queries).unwrap() {
        count += 1;
        drop(op);
    }
    assert_eq!(count, 3);
    assert!(
        !owner.maintain(),
        "encoder completion does not retire submit's borrowed bytes"
    );
    let submit_frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("replay finished: {error:?}"));
    assert!(
        !owner.maintain(),
        "submit owner still retains the recording lease"
    );
    drop(submit_frame);
    let mut completions = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut completions);
    assert!(owner.maintain());
    assert!(weak.upgrade().is_none());
}

#[test]
fn typed_constant_capture_rejects_invalid_extent_and_latches_failure() {
    let mut recorder = FrameRecorder::new();
    let mut scratch = ScratchArena::new();
    // SAFETY: this local arena remains alive through every use of the capture.
    let data = unsafe { crate::draw_data::arena_alloc_bytes(&mut scratch, &[0; 16]) };
    assert_eq!(
        recorder.record_vs_constants(&mut scratch, 255, 2, data),
        Err(WireError::InvalidValue)
    );
    assert_eq!(recorder.recording_error(), Some(WireError::InvalidValue));
    assert_eq!(
        recorder.record_ps_constants(&mut scratch, 0, 1, data),
        Err(WireError::InvalidValue)
    );
    assert_eq!(recorder.count, 0);
    let mut recorder = FrameRecorder::new();
    assert_eq!(
        recorder.record_ff_vs_constants(&mut scratch, 0, 2, data),
        Err(WireError::InvalidValue)
    );
    assert_eq!(recorder.count, 0);
}

#[test]
fn typed_draw_and_constant_capture_roundtrips_without_owned_operations() {
    use mtld3d_shared::mtl::PrimitiveType;

    use crate::draw_data::{DrawOp, IndexSource, VertexSource, arena_alloc_bytes};

    let (mut frame, _) = frame_with_leases();
    frame.ops.clear();
    // SAFETY: the packet retains this arena through decoded frame teardown.
    let data = unsafe { arena_alloc_bytes(frame.scratch_mut(), &[0x5a; 64]) };
    let mut recorder = FrameRecorder::new();
    recorder
        .record_vs_constants(frame.scratch_mut(), 1, 4, data)
        .unwrap();
    recorder
        .record_ps_constants(frame.scratch_mut(), 2, 4, data)
        .unwrap();
    recorder
        .record_ff_vs_constants(frame.scratch_mut(), 3, 4, data)
        .unwrap();
    recorder
        .record_draw(
            frame.scratch_mut(),
            &DrawOp {
                metal_prim: PrimitiveType::Triangle,
                vertex_source: VertexSource::Up {
                    bytes: data,
                    size: 64,
                    stride: 16,
                },
                index_source: IndexSource::None {
                    start_vertex: 0,
                    vertex_count: 3,
                },
            },
        )
        .unwrap();
    assert_eq!(recorder.count, 4);
    frame.recorder = Some(recorder);
    let mut packet = FramePacket::new(frame)
        .unwrap_or_else(|(error, _)| panic!("valid typed records: {error:?}"));
    // SAFETY: this test owns the sole packet and retains it through replay.
    unsafe { packet.mark_admitted() };
    // SAFETY: packet retains authentic command, metadata and payload allocations.
    let decoded = unsafe {
        decode_packet(
            packet.metadata_bytes(),
            packet.operation_bytes(),
            packet.completion_address(),
            &mut QueryLeaseCache::default(),
            |_| panic!("no registration"),
        )
    }
    .unwrap();
    assert_eq!(decoded.ops.len(), 4);
    for (op, expected) in decoded.ops[..3].iter().zip([1, 2, 3]) {
        let (start_row, rows, bytes) = match op {
            Op::SetVsConstRange {
                start_row,
                rows,
                data,
            }
            | Op::SetPsConstRange {
                start_row,
                rows,
                data,
            }
            | Op::SetFfVsConstRange {
                start_row,
                rows,
                data,
            } => (*start_row, *rows, data),
            _ => panic!("constant expected"),
        };
        assert_eq!((start_row, rows), (expected, 4));
        assert_eq!(bytes.as_slice(), &[0x5a; 64]);
    }
    assert!(matches!(decoded.ops[3], Op::Draw(_)));
    drop(decoded);
    assert!(packet.maintain());
}

#[test]
fn unfinished_replay_returns_storage_for_native_quarantine() {
    let (mut frame, _) = frame_with_leases();
    frame.ops.retain(|op| !matches!(op, Op::AdoptProgram(_)));
    let mut owner =
        FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("fixture: {error:?}"));
    // SAFETY: this test keeps the immutable packet alive through native quarantine.
    unsafe { owner.mark_admitted() };
    // SAFETY: the fixture is an authentic retained matched-producer packet.
    let mut packet = unsafe {
        prepare_packet(
            owner.metadata_bytes(),
            owner.operation_bytes(),
            owner.completion_address(),
            |_| unreachable!(),
        )
    }
    .unwrap();
    let mut queries = QueryLeaseCache::default();
    drop(packet.next_op(&mut queries).unwrap().unwrap());
    // SAFETY: allocating a snapshot does not invalidate existing arena allocations.
    let snapshot = unsafe { packet.frame_mut() }
        .scratch
        .alloc_value(0x1234_u32);
    let Err((error, quarantined)) = packet.into_frame() else {
        panic!("an unfinished stream must retain its owner");
    };
    assert_eq!(error, WireError::InvalidValue);
    assert!(!owner.maintain());
    // SAFETY: quarantine retains the native snapshot until its last cached user is gone.
    assert_eq!(unsafe { *snapshot }, 0x1234);
    drop(quarantined);
    assert!(owner.was_rejected());
    // SAFETY: all native readers and owners are now gone, as after encoder shutdown.
    unsafe { owner.cancel_unadopted() };
    let mut completions = crate::guest_completions::CompletionDrain::default();
    owner.drain_test_completions(&mut completions);
    assert!(owner.maintain());
}
