use std::sync::atomic::{AtomicU64, Ordering};

use objc2_metal::MTLCreateSystemDefaultDevice;

use super::*;

fn stamp(seq: u64, draw: &AtomicU64, upload: &AtomicU64) -> SubmitStamp {
    SubmitStamp::for_counters(seq, draw, upload)
}

/// A set in flight on an earlier submission is never handed out again until it retires.
#[test]
fn a_plane_set_waits_for_its_submission_to_retire() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut pool = PlanePool::default();
    let first = stamp(2, &draw, &upload);
    let a = pool
        .acquire(&device, 64, 64, true, &first, None)
        .expect("planes");
    pool.end_submission(&first);
    let second = stamp(3, &draw, &upload);
    let b = pool
        .acquire(&device, 64, 64, true, &second, None)
        .expect("planes");
    assert_ne!(a, b, "submission 2 has not retired");
    pool.end_submission(&second);
    draw.store(2, Ordering::Release);
    let third = stamp(4, &draw, &upload);
    assert_eq!(
        pool.acquire(&device, 64, 64, true, &third, None),
        Some(a),
        "submission 2 retired, so its set is reused"
    );
}

/// Two transfers in one command buffer share a set; one transfer's two sets differ.
#[test]
fn one_command_buffer_reuses_a_set_but_input_and_output_differ() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut pool = PlanePool::default();
    let s = stamp(2, &draw, &upload);
    let input = pool
        .acquire(&device, 64, 64, false, &s, None)
        .expect("input");
    let output = pool
        .acquire(&device, 32, 32, false, &s, Some(input))
        .expect("output");
    assert_ne!(input, output, "a transfer reads one set and writes another");
    assert_eq!(
        pool.acquire(&device, 64, 64, false, &s, None),
        Some(input),
        "a later transfer in the same command buffer reuses the set"
    );
    // The upload command buffer of the same submission is a different buffer.
    let upload_stamp = s.upload();
    let other = pool
        .acquire(&device, 64, 64, false, &upload_stamp, None)
        .expect("planes");
    assert!(
        other != input && other != output,
        "the render buffer has not retired"
    );
}

/// A set too small for the transfer, or lacking a stencil plane, is not reused.
#[test]
fn a_set_is_reused_only_when_its_planes_fit() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut pool = PlanePool::default();
    let s = stamp(2, &draw, &upload);
    let small = pool
        .acquire(&device, 16, 16, false, &s, None)
        .expect("planes");
    let larger = pool
        .acquire(&device, 128, 16, false, &s, None)
        .expect("planes");
    assert_ne!(small, larger);
    let with_stencil = pool
        .acquire(&device, 16, 16, true, &s, None)
        .expect("planes");
    assert!(with_stencil != small && with_stencil != larger);
    assert_eq!(
        pool.acquire(&device, 16, 16, true, &s, None),
        Some(with_stencil)
    );
}

/// Retired sets beyond the spares go at the end of a submission.
#[test]
fn retired_plane_sets_are_trimmed() {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let (draw, upload) = (AtomicU64::new(0), AtomicU64::new(0));
    let mut pool = PlanePool::default();
    for seq in 2..8 {
        let s = stamp(seq, &draw, &upload);
        let _ = pool.acquire(&device, 16, 16, false, &s, None);
        pool.end_submission(&s);
    }
    assert_eq!(pool.len(), 6);
    draw.store(7, Ordering::Release);
    pool.end_submission(&stamp(8, &draw, &upload));
    assert_eq!(pool.len(), SPARE_PLANE_SETS);
}
