//! Unit tests for the occlusion-query logic behind `IDirect3DQuery9`.
//!
//! Covers slot allocation over the current buffer (bump, reset, a full buffer), `sum_slots`
//! over half-open and out-of-range spans of a buffer's bytes, a span summed near the last slot
//! of a buffer of the device's size, and the BEGIN/END/publish state machine with its `u32`
//! clamp. The sizing cases pin the policy: a frame's first buffer is one page until recent
//! submissions reserved more, then a quarter above their decayed peak rounded up to a power of
//! two, never more than the device's buffer; a full buffer below that size grows in place and
//! one at it spills into another, a span open across the spill summing the slots of both. The
//! pool and state cases pin the lifetime rules: intake publishes only queries whose END frame
//! has retired, reuse waits for `coherent_seq` to reach a buffer's `release_seq`, hands out the
//! smallest buffer that fits with its last frame's slots zeroed, retiring never evicts (a frame
//! of more buffers than the pool keeps sums every one), and the intake hands back the free
//! buffers past the cap, the smallest first. The segment cases pin what a span
//! that outlives its submit does: it is cut at the boundary, reopened in the continuation, and
//! the two sums add up, while a span that lost its slots reads `u32::MAX` unless it held no
//! draw, where zero is exact. Reissuing before intake discards the pending segments of the
//! abandoned bracket while keeping the replacement bracket's result.

use mtld3d_shared::MetalHandle;

use super::{
    BUFFER_SLOTS_256K, BufferNeed, MIN_BUFFER_SLOTS, QueryStatus, RetiredVisibilityBuffer,
    SLOT_BYTES, VisibilityBufferPool, VisibilityOffsetAllocator, VisibilityQueryCore,
    VisibilityQueryState, logical_samples, sum_slots,
};
use crate::page_box::PageBox;

fn dummy_buf(seq: u64) -> RetiredVisibilityBuffer {
    // SAFETY: tests; opaque value never dereferenced.
    let handle = unsafe { MetalHandle::new(0xDEAD_BEEF) };
    RetiredVisibilityBuffer::new(PageBox::new_zeroed(8192), handle, seq)
}

/// A zeroed test buffer of `slots` slots.
fn buf_of(slots: u32) -> RetiredVisibilityBuffer {
    // SAFETY: tests; opaque value never dereferenced.
    let handle = unsafe { MetalHandle::new(0xDEAD_BEEF) };
    RetiredVisibilityBuffer::new(
        PageBox::new_zeroed((slots * SLOT_BYTES) as usize),
        handle,
        0,
    )
}

/// Write a sample count into one slot of a test buffer, as the GPU would.
fn write_slot(buf: &mut RetiredVisibilityBuffer, slot: usize, value: u64) {
    let bytes = value.to_ne_bytes();
    buf.backing.as_mut_slice()[slot * 8..slot * 8 + 8].copy_from_slice(&bytes);
}

/// Read one slot of a test buffer back.
fn read_slot(buf: &RetiredVisibilityBuffer, slot: usize) -> u64 {
    let bytes = &buf.backing.as_slice()[slot * 8..slot * 8 + 8];
    u64::from_ne_bytes(bytes.try_into().expect("an 8-byte slot"))
}

/// Lay counters out as the bytes of a visibility buffer.
fn slot_bytes(slots: &[u64]) -> Vec<u8> {
    slots.iter().flat_map(|slot| slot.to_ne_bytes()).collect()
}

/// Run one submission that reserves `slots` slots in a first buffer of their size.
fn submit_reserving(state: &mut VisibilityQueryState, seq: u64, slots: u32) {
    let slots_wanted = slots.max(MIN_BUFFER_SLOTS);
    let buf = state
        .try_acquire_reusable(slots_wanted)
        .unwrap_or_else(|| buf_of(slots_wanted));
    state.install_first_buffer(buf);
    for _ in 0..slots {
        state.bump_slot().expect("the buffer holds the slots");
    }
    state.retire_frame_buffers(seq);
    state.pool.release_up_to(seq);
    state.reset_frame();
}

/// The retired buffer `index` of the frame `seq`.
fn retired_mut(
    state: &mut VisibilityQueryState,
    seq: u64,
    index: u32,
) -> &mut RetiredVisibilityBuffer {
    state
        .pool
        .retired
        .iter_mut()
        .find(|buf| buf.release_seq == seq && buf.index == index)
        .expect("the frame's buffer")
}

#[test]
fn sum_slots_single_span() {
    let slots = slot_bytes(&[0, 42, 0, 0]);
    assert_eq!(sum_slots(&slots, 1, 2), 42);
}

#[test]
fn sum_slots_multi_span() {
    let slots = slot_bytes(&[0, 10, 20, 30, 0]);
    assert_eq!(sum_slots(&slots, 1, 4), 60);
}

#[test]
fn sum_slots_empty_range() {
    let slots = slot_bytes(&[0; 8]);
    assert_eq!(sum_slots(&slots, 5, 5), 0);
}

#[test]
fn sum_slots_out_of_range_saturates() {
    let slots = slot_bytes(&[1, 2, 3]);
    // end past buffer length saturates at the backing's last slot.
    assert_eq!(sum_slots(&slots, 0, 10), 6);
}

#[test]
fn sum_slots_begin_past_end_returns_zero() {
    let slots = slot_bytes(&[1, 2, 3]);
    assert_eq!(sum_slots(&slots, 2, 1), 0);
}

#[test]
fn intake_sums_a_span_counted_up_to_the_last_slot_of_a_device_size_buffer() {
    // An engine that tests each visible primitive's bounds runs well past a
    // thousand slots in a frame; a query there sums its own slots like the
    // first one does.
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let early = VisibilityQueryCore::new();
    let late = VisibilityQueryCore::new();
    let mut buffer = buf_of(BUFFER_SLOTS_256K);
    buffer.release_seq = 1;
    write_slot(&mut buffer, 0, 11);
    write_slot(&mut buffer, 5_000, 300);
    write_slot(&mut buffer, 5_001, 20);
    write_slot(&mut buffer, BUFFER_SLOTS_256K as usize - 1, 4);
    state.pool.retire(buffer);
    early.begin(1, 0, (640, 480), (640, 480), 0);
    early.end(1, 1);
    state.push_pending(1, early.clone(), (0, 1), true);
    late.begin(1, 5_000, (640, 480), (640, 480), 1);
    late.end(1, 2);
    state.push_pending(1, late.clone(), (5_000, BUFFER_SLOTS_256K), true);

    let _ = state.intake_completed(1);
    assert_eq!(early.get_u32(), 11);
    assert_eq!(late.get_u32(), 324, "every slot up to the last one");
}

#[test]
fn reuse_zeroes_the_slots_the_last_frame_reserved() {
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    state.install_first_buffer(buf_of(MIN_BUFFER_SLOTS));
    for _ in 0..3 {
        state.bump_slot();
    }
    state.retire_frame_buffers(1);
    // The GPU counted into the frame's slots before it retired.
    let retired = retired_mut(&mut state, 1, 0);
    assert_eq!(retired.reserved, 3);
    write_slot(retired, 0, 7);
    write_slot(retired, 2, 9);
    state.pool.release_up_to(1);

    let reused = state
        .try_acquire_reusable(MIN_BUFFER_SLOTS)
        .expect("released for reuse");
    assert_eq!(read_slot(&reused, 0), 0);
    assert_eq!(read_slot(&reused, 2), 0, "the last reserved slot");
    assert_eq!(reused.reserved, 0);
}

#[test]
fn a_first_buffer_is_one_page_until_submissions_reserve_more() {
    let state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    assert_eq!(state.first_buffer_slots(), MIN_BUFFER_SLOTS);
    assert_eq!(state.buffer_need(), BufferNeed::First(MIN_BUFFER_SLOTS));
}

#[test]
fn a_first_buffer_follows_the_decayed_peak_with_headroom() {
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    submit_reserving(&mut state, 1, 3_000);
    assert_eq!(
        state.first_buffer_slots(),
        4_096,
        "a quarter above 3,000, rounded up to a power of two"
    );
    // A light frame does not shrink the next one at once.
    submit_reserving(&mut state, 2, 10);
    assert_eq!(state.first_buffer_slots(), 4_096);
    // The peak decays by a thirty-second a submission, back to one page.
    for seq in 3..200 {
        submit_reserving(&mut state, seq, 10);
    }
    assert_eq!(state.first_buffer_slots(), MIN_BUFFER_SLOTS);
}

#[test]
fn a_first_buffer_never_exceeds_the_device_size() {
    let mut state = VisibilityQueryState::new(4_096);
    state.peak_slots = 1_000_000;
    assert_eq!(state.first_buffer_slots(), 4_096);
    state.peak_slots = u32::MAX;
    assert_eq!(state.first_buffer_slots(), 4_096);
}

#[test]
fn a_full_buffer_below_the_device_size_grows_and_keeps_its_slots() {
    let mut state = VisibilityQueryState::new(8_192);
    state.install_first_buffer(buf_of(MIN_BUFFER_SLOTS));
    for _ in 0..MIN_BUFFER_SLOTS {
        state.bump_slot().expect("a slot");
    }
    assert_eq!(state.bump_slot(), None, "the first buffer is full");
    assert_eq!(state.buffer_need(), BufferNeed::Grow(2 * MIN_BUFFER_SLOTS));
    assert!(
        state
            .install_grown_buffer(buf_of(2 * MIN_BUFFER_SLOTS))
            .is_empty()
    );
    assert_eq!(
        state.bump_slot(),
        Some(MIN_BUFFER_SLOTS),
        "the larger buffer takes over the slots handed out"
    );
    assert_eq!(
        state.current_buffer_index(),
        0,
        "still the frame's one buffer"
    );
    assert_eq!(
        state.pool.free.len(),
        1,
        "the replaced buffer, never submitted, is free at once"
    );
}

#[test]
fn a_span_open_across_a_spill_sums_the_slots_of_both_buffers() {
    let mut state = VisibilityQueryState::new(MIN_BUFFER_SLOTS);
    state.install_first_buffer(buf_of(MIN_BUFFER_SLOTS));
    let closed = VisibilityQueryCore::new();
    let spanning = VisibilityQueryCore::new();
    // A query that closes before the buffer fills.
    let begin = state.bump_slot().expect("a slot");
    closed.begin(1, begin, (640, 480), (640, 480), state.draws_seen());
    state.note_draw();
    let end = state.bump_slot().expect("a slot");
    closed.end(1, state.draws_seen());
    state.push_pending(1, closed.clone(), (begin, end), true);
    // A query open while the buffer fills.
    let begin = state.bump_slot().expect("a slot");
    spanning.begin(1, begin, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&spanning);
    state.note_draw();
    while state.bump_slot().is_some() {}
    assert_eq!(state.buffer_need(), BufferNeed::Spill(MIN_BUFFER_SLOTS));

    state.spill_into(1, buf_of(MIN_BUFFER_SLOTS));
    assert_eq!(state.current_buffer_index(), 1);
    assert_eq!(
        spanning.offset_begin(),
        0,
        "reopened at the next buffer's start"
    );
    state.note_draw();
    let after = state.bump_slot().expect("a slot of the next buffer");
    assert_eq!(after, 0);
    let end = state.bump_slot().expect("a slot");
    spanning.end(1, state.draws_seen());
    state.remove_active(&spanning);
    state.push_pending(1, spanning.clone(), (spanning.offset_begin(), end), true);

    state.retire_frame_buffers(1);
    write_slot(retired_mut(&mut state, 1, 0), 0, 5);
    write_slot(retired_mut(&mut state, 1, 0), 7, 100);
    write_slot(retired_mut(&mut state, 1, 1), 0, 23);
    assert_eq!(retired_mut(&mut state, 1, 0).reserved, MIN_BUFFER_SLOTS);
    assert_eq!(retired_mut(&mut state, 1, 1).reserved, 2);

    let _ = state.intake_completed(1);
    assert_eq!(closed.get_u32(), 5, "the span inside the first buffer");
    assert_eq!(spanning.status(), QueryStatus::Issued);
    assert_eq!(
        spanning.get_u32(),
        123,
        "the span's slots in the full buffer and in the next one"
    );
}

#[test]
fn pool_hands_out_the_smallest_free_buffer_that_fits() {
    let mut pool = VisibilityBufferPool::new(8);
    for slots in [8_192, 2_048, 4_096] {
        assert!(pool.give_back(buf_of(slots)).is_empty());
    }
    assert_eq!(pool.acquire(3_000).map(|buf| buf.slots()), Some(4_096));
    assert_eq!(pool.acquire(10_000).map(|buf| buf.slots()), None);
    assert_eq!(pool.acquire(1).map(|buf| buf.slots()), Some(2_048));
}

#[test]
fn pool_over_cap_evicts_the_smallest_free_buffer_first() {
    let mut pool = VisibilityBufferPool::new(2);
    assert!(pool.give_back(buf_of(8_192)).is_empty());
    assert!(pool.give_back(buf_of(2_048)).is_empty());
    let evicted = pool.give_back(buf_of(4_096));
    assert_eq!(evicted.len(), 1, "over the cap by one");
    assert_eq!(evicted[0].slots(), 2_048);
}

#[test]
fn allocator_bump_monotonic() {
    let mut a = VisibilityOffsetAllocator::new();
    a.reset(8);
    assert_eq!(a.bump(), Some(0));
    assert_eq!(a.bump(), Some(1));
    assert_eq!(a.bump(), Some(2));
    assert_eq!(a.next, 3);
}

#[test]
fn allocator_reset_returns_used_and_restarts() {
    let mut a = VisibilityOffsetAllocator::new();
    a.reset(8);
    a.bump();
    a.bump();
    a.bump();
    assert_eq!(a.next, 3);
    a.reset(8);
    assert_eq!(a.next, 0);
    assert_eq!(a.bump(), Some(0));
}

#[test]
fn allocator_stops_at_its_buffer_and_has_none_before_one() {
    let mut a = VisibilityOffsetAllocator::new();
    assert_eq!(a.bump(), None, "no buffer yet");
    a.reset(MIN_BUFFER_SLOTS);
    for _ in 0..MIN_BUFFER_SLOTS {
        assert!(a.bump().is_some());
    }
    assert!(a.bump().is_none());
    a.reset(MIN_BUFFER_SLOTS);
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
    assert!(pool.acquire(1).is_none());
    pool.retire(dummy_buf(5));
    // Not yet released.
    assert!(pool.acquire(1).is_none());
    pool.release_up_to(5);
    let reused = pool
        .acquire(1)
        .expect("buffer should be free after release");
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
fn pool_caps_the_free_list_and_never_the_retired_one() {
    let mut pool = VisibilityBufferPool::new(2);
    for seq in 1..=3 {
        pool.retire(dummy_buf(seq));
    }
    assert_eq!(
        pool.retired.len(),
        3,
        "a retired buffer still holds counts to read"
    );
    let evicted = pool.release_up_to(3);
    assert_eq!(evicted.len(), 1, "three come free, two are kept");
    assert_eq!(pool.free.len(), 2);
    assert!(pool.retired.is_empty());
}

#[test]
fn state_intake_completed_respects_seq() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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
    let _ = state.intake_completed(7);
    assert_eq!(c1.status(), QueryStatus::Issued);
    assert_eq!(c2.status(), QueryStatus::Pending);
    assert_eq!(state.pending.len(), 1);

    // coherent_seq = 10: c2 publishes.
    let _ = state.intake_completed(10);
    assert_eq!(c2.status(), QueryStatus::Issued);
    assert_eq!(state.pending.len(), 0);
}

#[test]
fn state_intake_adds_a_split_span_up_across_its_two_frames() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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

    let _ = state.intake_completed(1);
    assert_eq!(
        core.status(),
        QueryStatus::Pending,
        "a span is not done while a later segment is outstanding"
    );
    let _ = state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 1_000);
}

#[test]
fn state_reissue_before_intake_discards_the_abandoned_span() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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

    let _ = state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn state_reissue_waits_for_the_replacement_segments_sequence() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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

    let _ = state.intake_completed(1);
    assert_eq!(
        core.status(),
        QueryStatus::Pending,
        "the abandoned result does not complete the replacement bracket"
    );
    assert_eq!(state.pending.len(), 1);

    let _ = state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn state_intake_reads_an_empty_segment_without_a_buffer() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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

    let _ = state.intake_completed(2);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 55);
}

#[test]
fn state_intake_answers_permissively_when_a_counted_segments_buffer_is_gone() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), 0);
    core.end(1, 1);
    // Nothing retired at seq 1: the slots the span counted into cannot be
    // read back.
    state.push_pending(1, core.clone(), (0, 2), true);
    let _ = state.intake_completed(1);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), u32::MAX);
}

#[test]
fn state_split_and_resume_close_and_reopen_every_open_span() {
    use super::VisibilityQueryState;
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let core = VisibilityQueryCore::new();
    state.push_active(&core);
    // A second BEGIN on the same query leaves one entry.
    state.push_active(&core);
    assert_eq!(state.active_count(), 1);
    state.allocator.reset(MIN_BUFFER_SLOTS);
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
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let core = VisibilityQueryCore::new();
    state.allocator.reset(MIN_BUFFER_SLOTS);
    state.bump_slot();
    core.begin(1, 0, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&core);
    state.note_draw();
    let mut first = dummy_buf(1);
    write_slot(&mut first, 0, 900);
    state.pool.retire(first);

    // The flush ahead of the drain, then the drain's own finalize.
    state.split_open_spans(1);
    let _ = state.intake_completed(1);
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
    state.allocator.reset(MIN_BUFFER_SLOTS);
    state.bump_slot();
    state.note_draw();
    let mut second = dummy_buf(2);
    write_slot(&mut second, 0, 100);
    state.pool.retire(second);
    core.end(2, state.draws_seen());
    state.remove_active(&core);
    state.push_pending(2, core.clone(), (core.offset_begin(), 1), true);

    let _ = state.intake_completed(2);
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
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&core);
    state.note_draw();
    state.mark_exhausted();
    assert!(state.exhausted_this_frame());
    core.end(1, state.draws_seen());
    state.push_pending(1, core.clone(), (0, 0), true);
    let _ = state.intake_completed(1);
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
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
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
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    state.install_first_buffer(dummy_buf(0));
    state.bump_slot();
    state.mark_exhausted();
    assert_eq!(state.buffer_handle(0).raw(), 0xDEAD_BEEF);
    assert!(state.exhausted_this_frame());
    assert_eq!(state.allocator.next, 1);

    // `retire_frame_buffers` must run first: reset_frame does
    // not touch the frame's buffers.
    state.retire_frame_buffers(42);
    state.reset_frame();
    assert!(state.buffer_handle(0).is_null());
    assert!(!state.exhausted_this_frame());
    assert_eq!(state.allocator.next, 0);
    assert_eq!(state.buffer_need(), BufferNeed::First(MIN_BUFFER_SLOTS));
}

#[test]
fn intake_hands_back_the_free_buffers_past_the_cap() {
    // Force a small cap by poking a fresh state's pool. The public
    // `new()` uses 16; we want the over-cap path in-test.
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    state.pool = VisibilityBufferPool::new(2);

    // Three frames with a visibility buffer each, all retired before the
    // first intake: none is let go of while its frame is in flight.
    for seq in 1..=3 {
        state.install_first_buffer(dummy_buf(0));
        state.retire_frame_buffers(seq);
        state.reset_frame();
    }
    assert_eq!(state.pool.retired.len(), 3);
    // Once all three are free, two stay and one must be handed back to
    // the caller, not dropped in place.
    let mut evicted = state.intake_completed(3);
    assert_eq!(evicted.len(), 1, "the free list past its cap");
    let evicted = evicted.pop().expect("one entry");
    // The caller gates MTLBuffer destruction on coherent_seq >= this.
    assert!(evicted.release_seq() <= 3);
    let (_backing, handle, _release_seq) = evicted.into_parts();
    assert_eq!(handle.raw(), 0xDEAD_BEEF);
    assert_eq!(state.pool.free.len(), 2);
}

#[test]
fn a_frame_of_more_buffers_than_the_pool_keeps_sums_every_one() {
    // 21 buffers in one frame, more than the 16 free buffers the pool
    // keeps: every one is still there when intake reads the span open
    // across all of them.
    const BUFFERS: u32 = 21;
    let mut state = VisibilityQueryState::new(MIN_BUFFER_SLOTS);
    state.install_first_buffer(buf_of(MIN_BUFFER_SLOTS));
    let spanning = VisibilityQueryCore::new();
    let begin = state.bump_slot().expect("a slot");
    spanning.begin(1, begin, (640, 480), (640, 480), state.draws_seen());
    state.push_active(&spanning);
    state.note_draw();
    for _ in 1..BUFFERS {
        while state.bump_slot().is_some() {}
        assert_eq!(state.buffer_need(), BufferNeed::Spill(MIN_BUFFER_SLOTS));
        state.spill_into(1, buf_of(MIN_BUFFER_SLOTS));
    }
    // The command whose allocation spilled counts at slot 0 of the last
    // buffer, and END reserves the slot after it.
    assert_eq!(state.bump_slot(), Some(0));
    let end = state.bump_slot().expect("a slot of the last buffer");
    spanning.end(1, state.draws_seen());
    state.remove_active(&spanning);
    state.push_pending(1, spanning.clone(), (spanning.offset_begin(), end), true);
    state.retire_frame_buffers(1);
    assert_eq!(
        state.pool.retired.len(),
        BUFFERS as usize,
        "nothing evicted"
    );

    write_slot(retired_mut(&mut state, 1, 0), 0, 1);
    write_slot(retired_mut(&mut state, 1, 10), 5, 4);
    write_slot(retired_mut(&mut state, 1, BUFFERS - 1), 0, 2);
    let evicted = state.intake_completed(1);
    assert_eq!(spanning.status(), QueryStatus::Issued);
    assert_eq!(
        spanning.get_u32(),
        7,
        "the first, a middle and the last buffer"
    );
    assert_eq!(
        evicted.len(),
        BUFFERS as usize - 16,
        "the free list keeps 16 once the frame retires"
    );
}

const QUERIES: u32 = 100;
const PASS_BOUNDARIES: u32 = 5;
const TOTAL: u32 = QUERIES * (2 + PASS_BOUNDARIES);

#[test]
fn scaling_smoke_hundred_queries() {
    // 100 queries × (2 base slots + 5 pass-boundary bumps) = 700
    // slots, inside the first buffer.
    let mut a = VisibilityOffsetAllocator::new();
    a.reset(MIN_BUFFER_SLOTS);
    for _ in 0..TOTAL {
        assert!(a.bump().is_some());
    }
    assert_eq!(a.next, TOTAL);

    // Forge a slot array where each query contributes 10 visible
    // pixels split evenly across its slots.
    let total_usize = TOTAL as usize;
    let slots = slot_bytes(&vec![10; total_usize]);
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
    // A dimension the scale does not divide rounds to a whole texel on the
    // render grid: 3456x2234 at 75% is 2592x1676 (1675.5 rounds to nearest,
    // half up), and a full-frame count is exact only
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

#[test]
fn api_reissue_hides_old_completion_before_recorded_begin_executes() {
    let core = VisibilityQueryCore::new();
    let first = core.mark_armed();
    core.begin_recorded(first, 1, 0, (1, 1), (1, 1), 0);
    assert_eq!(core.mark_end_requested(), first);
    core.end_recorded(first, 1, 1);
    let mut state = super::VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let mut buffer = dummy_buf(1);
    write_slot(&mut buffer, 0, 123);
    state.pool.retire(buffer);
    state.push_pending(1, core.clone(), (0, 1), true);
    let second = core.mark_armed();
    assert_eq!(core.mark_end_requested(), second);
    assert_eq!(core.seq_end_loaded(), 0);
    let _ = state.intake_completed(1);
    assert_eq!(core.status(), super::QueryStatus::Pending);
    assert_eq!(core.get_u64(), 0);
    assert_eq!(core.seq_end_loaded(), 0);
    core.begin_recorded(second, 2, 0, (1, 1), (1, 1), 0);
    core.end_recorded(second, 2, 1);
    let mut replacement = dummy_buf(2);
    write_slot(&mut replacement, 0, 7);
    state.pool.retire(replacement);
    state.push_pending(2, core.clone(), (0, 1), true);
    let _ = state.intake_completed(2);
    assert_eq!(core.status(), super::QueryStatus::Issued);
    assert_eq!(core.get_u64(), 7);
    assert_eq!(core.seq_end_loaded(), 2);
}

#[test]
fn queued_brackets_execute_their_recorded_generation() {
    let core = VisibilityQueryCore::new();
    let first = core.mark_armed();
    assert_eq!(core.mark_end_requested(), first);
    let second = core.mark_armed();
    assert_eq!(core.mark_end_requested(), second);
    core.begin_recorded(first, 1, 0, (1, 1), (1, 1), 0);
    core.end_recorded(first, 1, 1);
    assert_eq!(core.issue_generation(), first);
    assert_eq!(core.seq_end_loaded(), 0);
    core.accumulate_segment(123);
    core.publish_span();
    assert_eq!(core.status(), super::QueryStatus::Pending);
    core.begin_recorded(second, 2, 0, (1, 1), (1, 1), 1);
    core.end_recorded(second, 2, 2);
    assert_eq!(core.issue_generation(), second);
    core.accumulate_segment(7);
    core.publish_span();
    assert_eq!(core.status(), super::QueryStatus::Issued);
    assert_eq!(core.get_u64(), 7);
    assert_eq!(core.seq_end_loaded(), 2);
}

#[test]
fn an_end_opens_an_empty_span_only_on_a_query_never_begun() {
    use super::EndIssue;
    let core = VisibilityQueryCore::new();
    assert!(!core.span_open(), "a query never begun has no span");
    assert_eq!(core.end_issue(), EndIssue::OpenAndClose);
    core.mark_armed();
    assert!(core.span_open());
    assert_eq!(core.end_issue(), EndIssue::Close, "a begun span closes");
    core.mark_end_requested();
    assert!(!core.span_open());
    assert_eq!(
        core.end_issue(),
        EndIssue::Keep,
        "a closed span keeps the result its first END made"
    );
    core.mark_armed();
    assert_eq!(core.end_issue(), EndIssue::Close, "a restarted span closes");
}

#[test]
fn state_a_span_begun_without_a_slot_splits_into_an_empty_segment() {
    use super::{QueryStatus, VisibilityQueryState};
    let mut state = VisibilityQueryState::new(BUFFER_SLOTS_256K);
    let mut buffer = dummy_buf(1);
    write_slot(&mut buffer, 0, 900);
    // Another span counted into slot 0, the buffer filled, and the larger
    // one that would follow it could not be allocated.
    state.allocator.reset(MIN_BUFFER_SLOTS);
    while state.bump_slot().is_some() {}
    state.mark_exhausted();
    let starved = VisibilityQueryCore::new();
    starved.begin(
        1,
        state.next_slot(),
        (640, 480),
        (640, 480),
        state.draws_seen(),
    );
    starved.mark_uncounted();
    state.push_active(&starved);
    state.pool.retire(buffer);
    state.split_open_spans(1);
    state.reset_frame();
    state.resume_open_spans(2);
    starved.end(2, state.draws_seen());
    state.remove_active(&starved);
    state.push_pending(2, starved.clone(), (starved.offset_begin(), 0), true);
    let _ = state.intake_completed(2);
    assert_eq!(starved.status(), QueryStatus::Issued);
    assert_eq!(
        starved.get_u32(),
        0,
        "a span with no draw in it counts none of the slots reserved before it began"
    );
}
