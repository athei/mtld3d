use std::sync::Arc;

use mtld3d_shared::encoder_protocol::EncoderOpcode;

use super::*;
use crate::{
    encoder_data::{StageUploadOp, TextureInfo, TextureUploadJob, UploadTextureOp},
    encoder_packet::tests::{admit, empty_frame, replay, seal},
    encoder_records::{StageUploadRecord, TextureUploadRecord, borrow},
    guest_pages::GuestOwnedPage,
    ids::BufferId,
    page_box::{PAGE_SIZE, PageBox, PageBoxRead},
    page_box_pool::PageBoxPool,
    upload_redirty::RedirtyQueue,
};

#[derive(Default, Debug, PartialEq, Eq)]
struct Calls {
    rejected: usize,
    cancelled: Vec<u64>,
}

impl RetirementHooks for Calls {
    fn packet_rejected(&mut self) {
        self.rejected += 1;
    }

    fn cancel_registration(&mut self, registration: u64) {
        self.cancelled.push(registration);
    }
}

fn constant_packet(pool: &CompletionPool) -> FramePacket {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::with_completion_pool(pool.clone());
    recorder
        .record_constant_bytes(
            &mut frame.scratch,
            EncoderOpcode::SetVsConstRange,
            0,
            1,
            &[0x33; 16],
        )
        .unwrap();
    seal(frame, recorder)
}

fn upload_packet(pool: &CompletionPool) -> FramePacket {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::with_completion_pool(pool.clone());
    for value in [3_u8, 5] {
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
    seal(frame, recorder)
}

fn registered(retirement: &PacketRetirement) -> Vec<usize> {
    let entries = retirement.leases.entries.iter().enumerate();
    entries
        .filter_map(|(token, entry)| entry.as_ref().map(|_| token))
        .collect()
}

#[test]
fn an_empty_queue_leaves_nothing_to_do_and_a_replay_completion_reclaims_the_packet_at_once() {
    let pool = CompletionPool::new();
    let mut owner = constant_packet(&pool);
    let mut packet = admit(&mut owner);
    let mut retirement = PacketRetirement::default();
    retirement.push(owner);
    let mut calls = Calls::default();

    assert!(!pool.has_ready(), "admission publishes nothing");
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        retirement.pending.len(),
        1,
        "the packet waits for its replay"
    );
    assert!(retirement.take_storage().is_none());

    while replay(&mut packet, |_, _, _| Ok(())).unwrap() {}
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}"));
    assert!(!pool.has_ready(), "the submit reader still holds the frame");
    retirement.maintain(&pool, &mut calls);
    assert_eq!(retirement.pending.len(), 1);

    drop(frame);
    assert!(pool.has_ready(), "the replay completion is queued");
    retirement.maintain(&pool, &mut calls);
    assert!(!pool.has_ready(), "one pass consumed the whole queue");
    assert!(
        retirement.pending.is_empty(),
        "the finished packet is released"
    );
    assert!(
        retirement.take_storage().is_some(),
        "the same pass recovers the packet's recording storage"
    );
    assert_eq!(calls, Calls::default());
}

#[test]
fn handed_over_leases_retire_only_after_replay_completion_and_their_final_notification() {
    let pool = CompletionPool::new();
    let mut owner = upload_packet(&pool);
    let mut packet = admit(&mut owner);
    let mut retirement = PacketRetirement::default();
    retirement.push(owner);
    let mut calls = Calls::default();
    let mut native = Vec::<GuestOwnedPage>::new();
    while replay(&mut packet, |command, _, _| {
        assert!(matches!(command.opcode(), EncoderOpcode::StageUpload));
        let record = borrow::<StageUploadRecord>(command.payload())?;
        // SAFETY: the real producer retained this unique owned page descriptor.
        native.push(unsafe { record.page.adopt()? });
        Ok(())
    })
    .unwrap()
    {}
    let frame = packet
        .into_frame()
        .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}"));

    retirement.maintain(&pool, &mut calls);
    assert!(
        registered(&retirement).is_empty(),
        "no lease leaves its packet before the packet's replay completion"
    );
    assert_eq!(retirement.pending.len(), 1);

    drop(frame);
    retirement.maintain(&pool, &mut calls);
    let tokens = registered(&retirement);
    assert_eq!(tokens.len(), 2, "native owners still hold both pages");
    assert!(retirement.pending.is_empty());
    assert!(retirement.leases.retired.is_empty());
    assert!(!pool.has_ready());
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        registered(&retirement),
        tokens,
        "no notification, no retirement"
    );

    drop(native);
    assert!(pool.has_ready());
    retirement.maintain(&pool, &mut calls);
    assert!(registered(&retirement).is_empty());
    assert!(retirement.leases.aliases.iter().all(Option::is_none));
    assert!(
        retirement.leases.retired.is_empty(),
        "the pass returned every retired slot"
    );
    let mut reused = [pool.allocate(false).token(), pool.allocate(false).token()]
        .map(|token| usize::try_from(token).unwrap());
    reused.sort_unstable();
    assert_eq!(reused.as_slice(), tokens, "both slots are free again");
    assert_eq!(calls, Calls::default());
}

#[test]
fn rejection_and_unadmitted_registrations_report_through_the_hooks() {
    let pool = CompletionPool::new();
    let mut retirement = PacketRetirement::default();
    let mut calls = Calls::default();

    let mut rejected = constant_packet(&pool);
    drop(admit(&mut rejected));
    retirement.push(rejected);
    assert!(pool.has_ready(), "the rejection is queued");
    retirement.maintain(&pool, &mut calls);
    assert_eq!(calls.rejected, 1);
    assert_eq!(
        retirement.pending.len(),
        1,
        "a rejected packet waits for quiescence"
    );

    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::with_completion_pool(pool.clone());
    recorder.registrations.push(41);
    frame.recorder = Some(recorder);
    let unadmitted = FramePacket::new(frame).unwrap_or_else(|(error, _)| panic!("{error:?}"));
    retirement.push(unadmitted);
    assert!(
        !pool.has_ready(),
        "nothing announces a packet native never admitted"
    );
    retirement.maintain(&pool, &mut calls);
    assert_eq!(calls.cancelled, [41]);
    retirement.maintain(&pool, &mut calls);
    assert_eq!(calls.cancelled, [41], "each registration is cancelled once");

    // SAFETY: nothing native ever saw these packets beyond the dropped decoder.
    unsafe { retirement.cancel_after_quiescence(&pool) };
    assert!(retirement.pending.is_empty());
    assert!(!pool.has_ready());
}

/// A packet retiring one pooled VBIB page, as the retention tier submits, and that page.
fn retention_packet(
    pool: &CompletionPool,
    pages: &'static crate::page_box_pool::PageBoxPool,
) -> (FramePacket, *const u8) {
    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::with_completion_pool(pool.clone());
    recorder.pagebox_pool = Some(pages);
    let page = PageBox::new_zeroed(12);
    let address = page.as_ptr();
    recorder.capture_vbib_retention(
        &mut frame.scratch,
        crate::encoder_data::PendingVbibRetention {
            buffer_id: BufferId::new_unique(),
            page_box: page,
            last_submit_seq: 9,
        },
    );
    (seal(frame, recorder), address)
}

/// Replay the packet and release every native owner, as a submission that waited does.
fn replay_and_release(owner: &mut FramePacket) {
    let mut packet = admit(owner);
    let mut native = Vec::<GuestOwnedPage>::new();
    while replay(&mut packet, |command, _, _| {
        assert!(matches!(command.opcode(), EncoderOpcode::RetainVbib));
        let record =
            borrow::<crate::encoder_packet::metadata::VbibRetentionRecord>(command.payload())?;
        // SAFETY: the real producer retained this unique owned page descriptor.
        native.push(unsafe { record.page.adopt()? });
        Ok(())
    })
    .unwrap()
    {}
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}")),
    );
    drop(native);
}

#[test]
fn a_synchronous_submission_frees_the_pages_native_released_before_it_returns() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(crate::page_box_pool::PageBoxPool::new(65536)));
    let (mut owner, address) = retention_packet(&pool, pages);
    replay_and_release(&mut owner);
    let mut retirement = PacketRetirement::default();
    retirement.push_submitted(owner, true, &pool, &mut Calls::default());
    assert_eq!(
        pages
            .acquire(12)
            .expect("freed before the submission returned")
            .as_ptr(),
        address
    );
    assert!(retirement.pending.is_empty());
    assert!(registered(&retirement).is_empty());
    assert!(!pool.has_ready());
}

#[test]
fn a_present_leaves_retirement_to_the_next_frame_pass() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(crate::page_box_pool::PageBoxPool::new(65536)));
    let (mut owner, address) = retention_packet(&pool, pages);
    replay_and_release(&mut owner);
    let mut retirement = PacketRetirement::default();
    let mut calls = Calls::default();
    retirement.push_submitted(owner, false, &pool, &mut calls);
    assert!(pages.acquire(12).is_none(), "no pass runs inside a Present");
    assert_eq!(retirement.pending.len(), 1);
    assert!(pool.has_ready(), "the next frame's pass will see it");
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        pages.acquire(12).expect("freed by that pass").as_ptr(),
        address
    );
}

#[test]
fn a_replay_completion_another_drain_consumed_is_maintained_at_the_push() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(crate::page_box_pool::PageBoxPool::new(65536)));
    let (mut owner, address) = retention_packet(&pool, pages);
    replay_and_release(&mut owner);
    // A second thread's pass drained the queue before this packet was retained.
    pool.drain(&mut CompletionDrain::default(), 4096, |_| {});
    assert!(!pool.has_ready(), "nothing is left to announce the packet");
    assert!(owner.replay_consumed());
    let mut retirement = PacketRetirement::default();
    retirement.push_submitted(owner, false, &pool, &mut Calls::default());
    assert!(
        retirement.pending.is_empty(),
        "the push itself reclaimed the packet"
    );
    assert_eq!(
        pages.acquire(12).expect("freed at the push").as_ptr(),
        address
    );
}

/// A packet uploading all of `staging`'s level through a pooled lease offered to `pages`.
fn staging_upload_packet(
    pool: &CompletionPool,
    pages: &'static PageBoxPool,
    staging: &Arc<PageBox>,
) -> FramePacket {
    use mtld3d_shared::mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage};
    use mtld3d_types::D3DFMT_A8R8G8B8;

    let mut frame = empty_frame();
    let mut recorder = FrameRecorder::with_completion_pool(pool.clone());
    recorder.pagebox_pool = Some(pages);
    let job = TextureUploadJob {
        info: TextureInfo {
            texture_id: crate::ids::TextureId::new_unique(),
            d3d_format: D3DFMT_A8R8G8B8,
            width: 64,
            height: 64,
            depth: 1,
            levels: 1,
            pixel_format: PixelFormat::Bgra8Unorm,
            create_flags: TextureCreateFlags::empty(),
            swizzle: [Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::Alpha],
            usage_flags: TextureUsage::empty(),
        },
        staging: PageBoxRead::new(Arc::clone(staging)),
        level: 0,
        destination_slice: 0,
        staging_index: 0,
        origin_x: 0,
        origin_y: 0,
        region_w: 64,
        region_h: 64,
        src_d3d_format: D3DFMT_A8R8G8B8,
        src_pitch: 256,
        bytes_per_pixel: 4,
        depth: 1,
        slice_pitch: 256 * 64,
        redirty: Arc::new(RedirtyQueue::new()),
        release_staging: false,
        upload_generation: 1,
    };
    recorder
        .record_typed(&mut frame.scratch, UploadTextureOp { job })
        .unwrap();
    seal(frame, recorder)
}

/// Replay the upload, returning the native read and feedback owners it adopted.
fn replay_upload(owner: &mut FramePacket) -> (PageBoxRead, Arc<RedirtyQueue>) {
    let mut packet = admit(owner);
    let mut adopted = None;
    while replay(&mut packet, |command, _, _| {
        assert!(matches!(command.opcode(), EncoderOpcode::UploadTexture));
        let record = borrow::<TextureUploadRecord>(command.payload())?;
        // SAFETY: the real producer retained these unique read and feedback descriptors.
        let read = unsafe { record.page.adopt_read()? };
        // SAFETY: the feedback lease stays with this admitted packet through native use.
        let feedback = unsafe { record.redirty.adopt()? };
        adopted = Some((read, feedback));
        Ok(())
    })
    .unwrap()
    {}
    drop(
        packet
            .into_frame()
            .unwrap_or_else(|(error, _)| panic!("complete replay: {error:?}")),
    );
    adopted.expect("one upload")
}

#[test]
fn staging_released_before_its_upload_retires_is_parked_by_the_retiring_pass() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let staging = Arc::new(PageBox::new_uninit(64 * 256));
    let address = staging.as_ptr();
    let mut owner = staging_upload_packet(&pool, pages, &staging);
    let native = replay_upload(&mut owner);
    let mut retirement = PacketRetirement::default();
    let mut calls = Calls::default();
    retirement.push_submitted(owner, false, &pool, &mut calls);
    assert!(
        !pages.recycle_staging(staging),
        "the texture's release leaves the pages with the upload lease"
    );
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        pages.staging_bytes(),
        0,
        "native code still holds the page and its wrapper keepalive"
    );
    assert!(pages.acquire_staging(64 * 256).is_none());
    drop(native);
    retirement.maintain(&pool, &mut calls);
    assert!(registered(&retirement).is_empty());
    assert_eq!(pages.staging_bytes(), PAGE_SIZE);
    assert_eq!(
        pages
            .acquire_staging(64 * 256)
            .expect("parked by the pass that retired the lease")
            .as_ptr(),
        address
    );
    assert_eq!(calls, Calls::default());
}

#[test]
fn device_teardown_retires_staging_leases_and_the_drain_empties_the_lane() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let mut retirement = PacketRetirement::default();
    let mut calls = Calls::default();
    let mut native = Vec::new();
    for _ in 0..3 {
        let staging = Arc::new(PageBox::new_uninit(64 * 256));
        let mut owner = staging_upload_packet(&pool, pages, &staging);
        native.push(replay_upload(&mut owner));
        retirement.push_submitted(owner, false, &pool, &mut calls);
        assert!(!pages.recycle_staging(staging));
    }
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        registered(&retirement).len(),
        6,
        "a page lease and a feedback lease per upload"
    );
    // Native destruction joins every worker and drops every native owner first.
    drop(native);
    // SAFETY: the native owners above are gone and nothing else can reach the packets.
    unsafe { retirement.cancel_after_quiescence(&pool) };
    assert!(retirement.pending.is_empty());
    assert!(registered(&retirement).is_empty());
    assert!(!pool.has_ready());
    assert_eq!(pages.staging_bytes(), 3 * PAGE_SIZE);
    assert_eq!(pages.drain_staging(), 3 * PAGE_SIZE);
    assert_eq!(pages.staging_bytes(), 0);
    assert_eq!(pages.pooled_bytes(), 0);
    assert!(pages.acquire_staging(64 * 256).is_none());
}

#[test]
fn a_native_wrapper_keepalive_defers_parking_until_it_drops() {
    let pool = CompletionPool::new();
    let pages = Box::leak(Box::new(PageBoxPool::new(usize::MAX)));
    let staging = Arc::new(PageBox::new_uninit(64 * 256));
    let address = staging.as_ptr();
    let mut owner = staging_upload_packet(&pool, pages, &staging);
    let (read, feedback) = replay_upload(&mut owner);
    // The cached staging wrapper keeps its own clone of the adopted native box, as
    // `MipStagingBuffer.keepalive` does, and outlives the upload's read.
    let keepalive = Arc::clone(read.backing());
    let mut retirement = PacketRetirement::default();
    let mut calls = Calls::default();
    retirement.push_submitted(owner, false, &pool, &mut calls);
    assert!(
        !pages.recycle_staging(staging),
        "the texture is released first"
    );
    drop(read);
    drop(feedback);
    retirement.maintain(&pool, &mut calls);
    assert_eq!(
        pages.staging_bytes(),
        0,
        "the wrapper keepalive still pins the pages"
    );
    assert!(pages.acquire_staging(64 * 256).is_none());
    assert_eq!(
        registered(&retirement).len(),
        1,
        "the page lease stays registered"
    );
    // Retention destroys the wrapper after GPU retirement, then drops the keepalive.
    drop(keepalive);
    retirement.maintain(&pool, &mut calls);
    assert!(registered(&retirement).is_empty());
    assert_eq!(pages.staging_bytes(), PAGE_SIZE);
    assert_eq!(
        pages
            .acquire_staging(64 * 256)
            .expect("parked once the keepalive dropped")
            .as_ptr(),
        address
    );
    assert_eq!(calls, Calls::default());
}
