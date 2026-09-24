use std::sync::atomic::{AtomicU64, Ordering};

use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice};

use super::*;

fn address(counter: &AtomicU64) -> u64 {
    core::ptr::from_ref(counter) as u64
}

fn stamp(seq: u64, draw: &AtomicU64, upload: &AtomicU64) -> SubmitStamp {
    SubmitStamp {
        seq,
        upload: false,
        draw_counter: address(draw),
        upload_counter: address(upload),
    }
}

/// Write `len` bytes of `fill` and return the chunk's address and the offset.
fn put(
    ring: &mut UploadRing,
    device: &ProtocolObject<dyn MTLDevice>,
    stamp: &SubmitStamp,
    len: usize,
    fill: u8,
) -> (usize, usize) {
    let bytes = vec![fill; len];
    let slice = ring.write(device, stamp, &bytes, 16).expect("ring write");
    (core::ptr::from_ref(slice.buffer).addr(), slice.offset)
}

/// A payload lands at the offset the ring returns, in the buffer it returns.
#[test]
fn a_payload_is_readable_at_its_offset() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let s = stamp(2, &draw, &upload);
    let mut ring = UploadRing::default();
    let _ = put(&mut ring, &device, &s, 3, 0x11);
    let slice = ring
        .write(&device, &s, &[1, 2, 3, 4, 5], 16)
        .expect("ring write");
    assert_eq!(slice.offset, 16, "the second payload starts at the next aligned offset");
    let contents = slice.buffer.contents().as_ptr().cast::<u8>();
    // SAFETY: the shared buffer holds at least `offset + 5` bytes, just written.
    let read = unsafe { core::slice::from_raw_parts(contents.add(slice.offset), 5) };
    assert_eq!(read, &[1, 2, 3, 4, 5]);
}

/// A full chunk is started again only once every submission that read it retired.
#[test]
fn a_chunk_is_not_started_again_before_its_readers_retire() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut ring = UploadRing::default();
    // One payload per frame, too large for two to share a chunk.
    let big = FIRST_CHUNK_BYTES * 5 / 8;
    let frame = |seq: u64, ring: &mut UploadRing| {
        let s = stamp(seq, &draw, &upload);
        let placed = put(ring, &device, &s, big, 1);
        ring.end_submission(&s);
        placed
    };
    let (a, offset) = frame(2, &mut ring);
    assert_eq!(offset, 0);
    let (b, _) = frame(3, &mut ring);
    assert_ne!(b, a, "frame 2 is still reading its chunk");
    draw.store(1, Ordering::Release);
    let (c, _) = frame(4, &mut ring);
    assert_ne!(c, a, "frame 2 is still reading its chunk");
    assert_ne!(c, b, "frame 3 is still reading its chunk");
    draw.store(2, Ordering::Release);
    assert_eq!(frame(5, &mut ring), (a, 0), "frame 2 retired, so its chunk starts again");
}

/// A chunk the upload command buffer read waits for the upload counter too.
#[test]
fn an_upload_read_holds_the_chunk_until_the_upload_counter_retires() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut ring = UploadRing::default();
    let big = FIRST_CHUNK_BYTES * 5 / 8;
    let frame = |s: &SubmitStamp, ring: &mut UploadRing| {
        let placed = put(ring, &device, s, big, 1);
        ring.end_submission(s);
        placed
    };
    let (a, _) = frame(&stamp(2, &draw, &upload).upload(), &mut ring);
    let _ = frame(&stamp(3, &draw, &upload), &mut ring);
    // The render counter reaching past frame 2 does not free an upload read.
    draw.store(3, Ordering::Release);
    let (c, _) = frame(&stamp(4, &draw, &upload), &mut ring);
    assert_ne!(c, a, "frame 2's upload buffer has not retired");
    upload.store(2, Ordering::Release);
    assert_eq!(
        frame(&stamp(5, &draw, &upload), &mut ring),
        (a, 0),
        "both counters retired frame 2's chunk"
    );
}

/// A submission without counters never retires what it used.
#[test]
fn an_unstamped_use_never_retires() {
    let (draw, upload) = (AtomicU64::new(u64::MAX - 1), AtomicU64::new(u64::MAX - 1));
    let params = SubmitFrameParams {
        record_handle: mtld3d_shared::record_handle::DeviceRecordHandle::NULL,
        blit_commands_ptr: 0,
        blit_command_count: 0,
        blit_commands_need_encoder: 0,
        passes_ptr: 0,
        pass_count: 0,
        upload_pass_count: 0,
        present_layer: mtld3d_shared::MetalHandle::NULL,
        present_texture: mtld3d_shared::MetalHandle::NULL,
        submit_seq: 0,
        coherent_seq_ptr: address(&draw),
        upload_coherent_seq_ptr: address(&upload),
        failed_submit_seq_ptr: 0,
        drawable_wait_ns: 0,
        present_view: mtld3d_shared::MetalHandle::NULL,
        present_wait_ns: 0,
        snapshot_flags: mtld3d_shared::mtl::SnapshotFlags::empty(),
        pad0: 0,
    };
    let unstamped = SubmitStamp::new(&params);
    assert!(!unstamped.persistent());
    let mut last = LastUse::default();
    last.record(&unstamped);
    assert!(!last.retired(&unstamped));
    assert!(LastUse::default().retired(&unstamped), "an unused resource is free");
}

/// Spare chunks beyond the few the next frames need go once they retired.
#[test]
fn retired_spares_are_trimmed() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut ring = UploadRing::default();
    let big = FIRST_CHUNK_BYTES * 5 / 8;
    // Eight frames in flight, one chunk each: nothing can be trimmed.
    for seq in 2..10 {
        let s = stamp(seq, &draw, &upload);
        let _ = put(&mut ring, &device, &s, big, 1);
        ring.end_submission(&s);
    }
    assert_eq!(ring.chunk_count(), 8);
    draw.store(9, Ordering::Release);
    let s = stamp(10, &draw, &upload);
    ring.end_submission(&s);
    assert_eq!(ring.chunk_count(), SPARE_CHUNKS + 1);
}
