//! Unit tests for the occlusion-query logic behind `IDirect3DQuery9`.
//!
//! Covers slot allocation (bump, reset, exhaustion), `sum_slots` over half-open and
//! out-of-range spans, and the BEGIN/END/publish state machine with its `u32` clamp. The
//! pool and state cases pin the lifetime rules: intake publishes only queries whose END frame
//! has retired, reuse waits for `coherent_seq` to reach a buffer's `release_seq`, and an
//! over-cap retire hands the evicted entry back. The segment cases pin what a span that
//! outlives its submit does: it is cut at the boundary, reopened in the continuation, and
//! the two sums add up, while a span that lost its slots reads `u32::MAX` unless it held
//! no draw, where zero is exact. Reissuing before intake discards the pending segments of
//! the abandoned bracket while keeping the replacement bracket's result.

use mtld3d_shared::MetalHandle;

use super::{
    MAX_SLOTS, QueryStatus, RetiredVisibilityBuffer, VisibilityBufferPool,
    VisibilityOffsetAllocator, VisibilityQueryCore, logical_samples, sum_slots,
};
use crate::page_box::PageBox;

fn dummy_buf(seq: u64) -> RetiredVisibilityBuffer {
    // SAFETY: tests; opaque value never dereferenced.
    let handle = unsafe { MetalHandle::new(0xDEAD_BEEF) };
    RetiredVisibilityBuffer::new(PageBox::new_zeroed(8192), handle, seq)
}

/// Write a sample count into one slot of a test buffer, as the GPU would.
fn write_slot(buf: &mut RetiredVisibilityBuffer, slot: usize, value: u64) {
    let bytes = value.to_le_bytes();
    buf.backing_mut().as_mut_slice()[slot * 8..slot * 8 + 8].copy_from_slice(&bytes);
}

#[test]
fn sum_slots_single_span() {
    let slots = [0u64, 42, 0, 0];
    assert_eq!(sum_slots(&slots, 1, 2), 42);
}

#[test]
fn sum_slots_multi_span() {
    let slots = [0u64, 10, 20, 30, 0];
    assert_eq!(sum_slots(&slots, 1, 4), 60);
}

#[test]
fn sum_slots_empty_range() {
    let slots = [0u64; 8];
    assert_eq!(sum_slots(&slots, 5, 5), 0);
}

#[test]
fn sum_slots_out_of_range_saturates() {
    let slots = [1u64, 2, 3];
    // end past buffer length saturates at slots.len().
    assert_eq!(sum_slots(&slots, 0, 10), 6);
}

#[test]
fn sum_slots_begin_past_end_returns_zero() {
    let slots = [1u64, 2, 3];
    assert_eq!(sum_slots(&slots, 2, 1), 0);
}

#[test]
fn allocator_bump_monotonic() {
    let mut a = VisibilityOffsetAllocator::new();
    assert_eq!(a.bump(), Some(0));
    assert_eq!(a.bump(), Some(1));
    assert_eq!(a.bump(), Some(2));
    assert_eq!(a.next, 3);
}

#[test]
fn allocator_reset_returns_used_and_restarts() {
    let mut a = VisibilityOffsetAllocator::new();
    a.bump();
    a.bump();
    a.bump();
    assert_eq!(a.next, 3);
    a.reset();
    assert_eq!(a.next, 0);
    assert_eq!(a.bump(), Some(0));
}

#[test]
fn allocator_exhaust_then_reset() {
    let mut a = VisibilityOffsetAllocator::new();
    for _ in 0..MAX_SLOTS {
        assert!(a.bump().is_some());
    }
    assert!(!a.exhausted);
    assert!(a.bump().is_none());
    assert!(a.exhausted);
    a.reset();
    assert!(!a.exhausted);
    assert_eq!(a.bump(), Some(0));
}

#[test]
fn query_core_status_transitions() {
    let core = VisibilityQueryCore::new();
    assert_eq!(core.status(), QueryStatus::NeverIssued);
    assert_eq!(core.seq_end_loaded(), 0);
    core.begin(10, 3, (640, 480), (640, 480), 0);
    assert_eq!(core.status(), QueryStatus::Pending);
    assert_eq!(core.offset_begin(), 3);
    // BEGIN does not record seq_end — END drives the GetData(FLUSH)
    // gate.
    assert_eq!(core.seq_end_loaded(), 0);
    core.end(10, 1);
    assert_eq!(core.status(), QueryStatus::Pending);
    assert_eq!(core.seq_end_loaded(), 10);
    core.accumulate_segment(42);
    core.publish_span();
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 42);
}

#[test]
fn query_core_adds_up_the_segments_of_a_split_span() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    // The submit boundary folds the first segment in and reopens the span
    // against the continuation frame's allocator.
    core.accumulate_segment(1_000);
    core.resume(2, 0);
    assert_eq!(core.offset_begin(), 0);
    assert_eq!(
        core.status(),
        QueryStatus::Pending,
        "a resumed span is not done"
    );
    core.end(2, 1);
    core.accumulate_segment(234);
    core.publish_span();
    assert_eq!(core.get_u32(), 1_234);
    assert_eq!(core.seq_end_loaded(), 2);
}

#[test]
fn query_core_uncounted_span_publishes_fully_visible() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 4);
    core.accumulate_segment(1_000);
    core.mark_uncounted();
    core.end(1, 5);
    core.publish_span();
    assert_eq!(
        core.get_u32(),
        u32::MAX,
        "a span that lost part of its count reads fully visible, never a partial sum"
    );
    // A re-issue clears the flag, so the next span reports its real count.
    core.begin(2, 0, (640, 480), (640, 480), 5);
    core.end(2, 6);
    core.accumulate_segment(7);
    core.publish_span();
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn query_core_uncounted_span_with_no_draw_publishes_zero() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 9);
    core.mark_uncounted();
    // The same draw total at END: nothing was issued inside the span, so
    // zero is the exact answer and the permissive one would be invented.
    core.end(1, 9);
    core.publish_span();
    assert_eq!(core.get_u32(), 0);
}

#[test]
fn query_core_reissue_resets_accumulator() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    core.end(1, 1);
    core.accumulate_segment(100);
    core.publish_span();
    assert_eq!(core.get_u32(), 100);
    // Re-issue with a different span: accumulator must zero out.
    core.begin(2, 2, (640, 480), (640, 480), 0);
    assert_eq!(core.status(), QueryStatus::Pending);
    core.end(2, 1);
    core.accumulate_segment(7);
    core.publish_span();
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn query_core_u32_clamp() {
    let core = VisibilityQueryCore::new();
    core.begin(0, 0, (640, 480), (640, 480), 0);
    core.end(0, 1);
    core.accumulate_segment(u64::MAX);
    core.publish_span();
    assert_eq!(core.get_u32(), u32::MAX);
}

#[test]
fn pool_retire_and_reuse() {
    let mut pool = VisibilityBufferPool::new(4);
    assert!(pool.acquire().is_none());
    assert!(pool.retire(dummy_buf(5)).is_none());
    // Not yet released.
    assert!(pool.acquire().is_none());
    pool.release_up_to(5);
    let reused = pool.acquire().expect("buffer should be free after release");
    assert_eq!(reused.metal_handle().raw(), 0xDEAD_BEEF);
    assert_eq!(reused.release_seq, 5);
}

#[test]
fn pool_holds_in_flight_until_seq_catches_up() {
    let mut pool = VisibilityBufferPool::new(4);
    pool.retire(dummy_buf(10));
    pool.release_up_to(7);
    assert_eq!(pool.retired.len(), 1);
    assert_eq!(pool.free.len(), 0);
    pool.release_up_to(10);
    assert_eq!(pool.retired.len(), 0);
    assert_eq!(pool.free.len(), 1);
}

#[test]
fn pool_free_cap_evicts_on_overfill() {
    let mut pool = VisibilityBufferPool::new(2);
    assert!(pool.retire(dummy_buf(1)).is_none());
    assert!(pool.retire(dummy_buf(2)).is_none());
    // Third retiree pushes total to 3 > cap 2 → one eviction.
    let evicted = pool.retire(dummy_buf(3));
    assert!(evicted.is_some());
}

#[test]
fn state_intake_completed_respects_seq() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let c1 = VisibilityQueryCore::new();
    let c2 = VisibilityQueryCore::new();
    c1.begin(5, 0, (640, 480), (640, 480), 0);
    c1.end(5, 1);
    c2.begin(10, 2, (640, 480), (640, 480), 0);
    c2.end(10, 1);
    state.push_pending(5, c1.clone(), (0, 1), true);
    state.push_pending(10, c2.clone(), (2, 3), true);
    // Retire one buffer at each seq. No GPU counters in the test
    // buffers → sum is 0, so `intake_completed` publishes 0.
    state.pool.retire(dummy_buf(5));
    state.pool.retire(dummy_buf(10));

    // coherent_seq = 7: only c1 (seq 5) should publish.
    state.intake_completed(7);
    assert_eq!(c1.status(), QueryStatus::Issued);
    assert_eq!(c2.status(), QueryStatus::Pending);
    assert_eq!(state.pending.len(), 1);

    // coherent_seq = 10: c2 publishes.
    state.intake_completed(10);
    assert_eq!(c2.status(), QueryStatus::Issued);
    assert_eq!(state.pending.len(), 0);
}

#[test]
fn state_intake_adds_a_split_span_up_across_its_two_frames() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    // Frame 1 counted slot 0, frame 2 slot 0 of its own buffer.
    let mut first = dummy_buf(1);
    write_slot(&mut first, 0, 900);
    let mut second = dummy_buf(2);
    write_slot(&mut second, 0, 100);
    state.pool.retire(first);
    state.pool.retire(second);
    state.push_pending(1, core.clone(), (0, 1), false);
    core.resume(2, 0);
    core.end(2, 1);
    state.push_pending(2, core.clone(), (0, 1), true);

    state.intake_completed(1);
    assert_eq!(
        core.status(),
        QueryStatus::Pending,
        "a span is not done while a later segment is outstanding"
    );
    state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 1_000);
}

#[test]
fn state_reissue_before_intake_discards_the_abandoned_span() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    let mut buffer = dummy_buf(1);
    write_slot(&mut buffer, 0, 100);
    write_slot(&mut buffer, 1, 7);
    state.pool.retire(buffer);

    core.begin(1, 0, (640, 480), (640, 480), 0);
    core.end(1, 1);
    state.push_pending(1, core.clone(), (0, 1), true);

    // Reissuing before the first segment retires abandons that bracket. Both
    // segments still share the frame's visibility buffer, but only the second
    // bracket belongs to the result the application asked for most recently.
    core.begin(1, 1, (640, 480), (640, 480), 1);
    core.end(1, 2);
    state.push_pending(1, core.clone(), (1, 2), true);

    state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn state_reissue_waits_for_the_replacement_segments_sequence() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();

    let mut abandoned = dummy_buf(1);
    write_slot(&mut abandoned, 0, 100);
    state.pool.retire(abandoned);
    core.begin(1, 0, (640, 480), (640, 480), 0);
    core.end(1, 1);
    state.push_pending(1, core.clone(), (0, 1), true);

    let mut replacement = dummy_buf(2);
    write_slot(&mut replacement, 0, 7);
    state.pool.retire(replacement);
    core.begin(2, 0, (640, 480), (640, 480), 1);
    core.end(2, 2);
    state.push_pending(2, core.clone(), (0, 1), true);

    state.intake_completed(1);
    assert_eq!(
        core.status(),
        QueryStatus::Pending,
        "the abandoned result does not complete the replacement bracket"
    );
    assert_eq!(state.pending.len(), 1);

    state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn state_intake_reads_an_empty_segment_without_a_buffer() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    // A continuation frame that drew nothing reserved no slot and so no
    // buffer either; its empty segment must not make the span permissive.
    state.push_pending(1, core.clone(), (0, 0), false);
    let mut second = dummy_buf(2);
    write_slot(&mut second, 0, 55);
    state.pool.retire(second);
    core.resume(2, 0);
    core.end(2, 1);
    state.push_pending(2, core.clone(), (0, 1), true);

    state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 55);
}

#[test]
fn state_intake_answers_permissively_when_a_counted_segments_buffer_is_gone() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    core.end(1, 1);
    // Nothing retired at seq 1: the slots the span counted into cannot be
    // read back.
    state.push_pending(1, core.clone(), (0, 2), true);
    state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), u32::MAX);
}

#[test]
fn state_split_and_resume_close_and_reopen_every_open_span() {
    use super::VisibilityQueryState;
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    state.push_active(&core);
    // A second BEGIN on the same query leaves one entry.
    state.push_active(&core);
    assert_eq!(state.active_count(), 1);
    state.bump_slot();
    state.bump_slot();
    core.begin(1, 0, (640, 480), (640, 480), 0);

    state.split_open_spans(1);
    assert_eq!(state.pending.len(), 1);
    assert_eq!(state.pending[0].span, (0, 2), "cut at the high-water mark");
    assert!(!state.pending[0].closes_span);

    state.reset_frame();
    assert_eq!(
        state.active_count(),
        1,
        "reset_frame leaves the open span for the continuation"
    );
    state.resume_open_spans(2);
    assert_eq!(
        core.offset_begin(),
        0,
        "reopened against the fresh allocator"
    );

    state.remove_active(&core);
    assert_eq!(state.active_count(), 0);
}

#[test]
fn state_drain_leaves_an_open_span_to_the_frame_that_continues_it() {
    use super::{QueryStatus, VisibilityQueryState};
    // A `Reset` finalizes what it can and then takes every buffer, which is
    // the same cut a submit makes for a span the application left open. The
    // segment already counted has been summed by then, and the span itself
    // continues in the frame that follows, so the two halves still add up.
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    state.bump_slot();
    core.begin(1, 0, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&core);
    state.note_draw();
    let mut first = dummy_buf(1);
    write_slot(&mut first, 0, 900);
    state.pool.retire(first);

    // The flush ahead of the drain, then the drain's own finalize.
    state.split_open_spans(1);
    state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Pending, "END has not run yet");
    let drained = state.drain_all_buffers();
    assert_eq!(drained.len(), 1, "the frame's buffer leaves with the drain");
    assert_eq!(
        state.active_count(),
        1,
        "the open span survives the drain that takes the buffers"
    );

    // The frame that continues the span.
    state.reset_frame();
    state.resume_open_spans(2);
    assert_eq!(
        core.offset_begin(),
        0,
        "reopened against the fresh allocator"
    );
    state.bump_slot();
    state.note_draw();
    let mut second = dummy_buf(2);
    write_slot(&mut second, 0, 100);
    state.pool.retire(second);
    core.end(2, state.draws_seen());
    state.remove_active(&core);
    state.push_pending(2, core.clone(), (core.offset_begin(), 1), true);

    state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(
        core.get_u32(),
        1_000,
        "both halves of the span the drain cut"
    );
}

#[test]
fn state_mark_exhausted_makes_every_open_span_uncounted() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new();
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&core);
    state.note_draw();
    state.mark_exhausted();
    assert!(state.exhausted_this_frame());
    core.end(1, state.draws_seen());
    state.push_pending(1, core.clone(), (0, 0), true);
    state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(
        core.get_u32(),
        u32::MAX,
        "an exhausted frame reports fully visible, not the zero an empty span sums to"
    );
}

#[test]
fn state_draw_counter_carries_across_a_frame_boundary() {
    use super::VisibilityQueryState;
    let mut state = VisibilityQueryState::new();
    state.note_draw();
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&core);
    state.note_draw();
    // The submit boundary resets the slot allocator, never the draw count.
    state.reset_frame();
    state.note_draw();
    state.mark_exhausted();
    core.end(2, state.draws_seen());
    core.publish_span();
    assert_eq!(
        core.get_u32(),
        u32::MAX,
        "two draws inside the span, and the count of them was lost"
    );
}

#[test]
fn state_reset_frame_clears_per_frame_fields() {
    use super::VisibilityQueryState;
    let mut state = VisibilityQueryState::new();
    state.bump_slot();
    state.install_current_buffer(dummy_buf(0));
    state.mark_exhausted();
    assert_eq!(state.current_buffer_handle().raw(), 0xDEAD_BEEF);
    assert!(state.exhausted_this_frame());
    assert_eq!(state.allocator.next, 1);

    // `retire_current_buffer` must run first — reset_frame does
    // not touch the current buffer slot.
    state.retire_current_buffer(42);
    state.reset_frame();
    assert!(state.current_buffer_handle().is_null());
    assert!(!state.exhausted_this_frame());
    assert_eq!(state.allocator.next, 0);
}

#[test]
fn retire_current_buffer_returns_evicted_when_over_cap() {
    use super::VisibilityQueryState;
    // Force a small cap by poking a fresh state's pool. The public
    // `new()` uses 16; we want the over-cap path in-test.
    let mut state = VisibilityQueryState::new();
    state.pool = super::VisibilityBufferPool::new(2);

    // Three frames with a visibility buffer each. Two fit, the
    // third exceeds cap → the oldest is evicted and must be
    // returned to the caller, not dropped in place.
    state.install_current_buffer(dummy_buf(0));
    assert!(state.retire_current_buffer(1).is_none());
    state.install_current_buffer(dummy_buf(0));
    assert!(state.retire_current_buffer(2).is_none());
    state.install_current_buffer(dummy_buf(0));
    let evicted = state
        .retire_current_buffer(3)
        .expect("over-cap retire must hand the evicted entry back");
    // Evicted entry carries the oldest release_seq (the caller
    // gates MTLBuffer destruction on coherent_seq >= this).
    assert_eq!(evicted.release_seq(), 1);
    let (_backing, handle, release_seq) = evicted.into_parts();
    assert_eq!(handle.raw(), 0xDEAD_BEEF);
    assert_eq!(release_seq, 1);
}

const QUERIES: u32 = 100;
const PASS_BOUNDARIES: u32 = 5;
const TOTAL: u32 = QUERIES * (2 + PASS_BOUNDARIES);

#[test]
fn scaling_smoke_hundred_queries() {
    // 100 queries × (2 base slots + 5 pass-boundary bumps) = 700
    // slots, still under MAX_SLOTS budget.
    let mut a = VisibilityOffsetAllocator::new();
    for _ in 0..TOTAL {
        assert!(a.bump().is_some());
    }
    assert_eq!(a.next, TOTAL);
    assert!(!a.exhausted);

    // Forge a slot array where each query contributes 10 visible
    // pixels split evenly across its slots.
    let total_usize = TOTAL as usize;
    let slots: Vec<u64> = (0..total_usize).map(|_| 10).collect();
    let sum = sum_slots(&slots, 0, TOTAL);
    assert_eq!(
        u32::try_from(sum).expect("700 * 10 = 7000 fits u32"),
        TOTAL * 10,
    );
}

#[test]
fn logical_samples_passes_equal_and_unknown_areas_through() {
    assert_eq!(logical_samples(0, 307_200, 307_200), 0);
    assert_eq!(logical_samples(307_200, 307_200, 307_200), 307_200);
    assert_eq!(logical_samples(u64::MAX, 307_200, 307_200), u64::MAX);
    assert_eq!(
        logical_samples(42, 0, 307_200),
        42,
        "no target bound at BEGIN"
    );
    assert_eq!(
        logical_samples(42, 76_800, 0),
        42,
        "no target bound at BEGIN"
    );
}

#[test]
fn logical_samples_scales_by_the_area_ratio_rounding_to_nearest() {
    // A 640x480 frame at 50% rasterizes 320x240 samples.
    assert_eq!(logical_samples(76_800, 320 * 240, 640 * 480), 307_200);
    // At 75% a 480x360 grid; 172_800 * 16 / 9.
    assert_eq!(logical_samples(172_800, 480 * 360, 640 * 480), 307_200);
    // A dimension the scale does not divide rounds up on the render grid:
    // 3456x2234 at 75% is 2592x1676, and a full-frame count is exact only
    // through the actual ratio, where the nominal 16/9 would give 7_723_008.
    assert_eq!(
        logical_samples(2592 * 1676, 2592 * 1676, 3456 * 2234),
        3456 * 2234
    );
    // One sample at 75% is 1.78 reported pixels, rounded to 2; at 50% it is 4.
    assert_eq!(logical_samples(1, 480 * 360, 640 * 480), 2);
    assert_eq!(logical_samples(1, 320 * 240, 640 * 480), 4);
    // 5 samples at 75% are 8.89, rounded to 9.
    assert_eq!(logical_samples(5, 480 * 360, 640 * 480), 9);
}

#[test]
fn logical_samples_saturates_instead_of_wrapping() {
    assert_eq!(logical_samples(u64::MAX, 320 * 240, 640 * 480), u64::MAX);
}

#[test]
fn finalize_reports_the_count_in_reported_pixels_of_the_target_begun_against() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (320, 240), 0);
    core.end(1, 1);
    core.accumulate_segment(76_800);
    core.publish_span();
    assert_eq!(core.get_u32(), 307_200);
    assert_eq!(core.get_u64(), 307_200);
    assert_eq!(core.status(), QueryStatus::Issued);
}
