//! Unit tests for the occlusion-query logic behind `IDirect3DQuery9`.
//!
//! Covers slot allocation (bump, reset, exhaustion), `sum_slots` over half-open and
//! out-of-range spans, and the BEGIN/END/finalize state machine with its `u32` clamp. The
//! pool and state cases pin the lifetime rules: intake finalizes only queries whose END frame
//! has retired, reuse waits for `coherent_seq` to reach a buffer's `release_seq`, and an
//! over-cap retire hands the evicted entry back.

use mtld3d_shared::MetalHandle;

use super::{
    MAX_SLOTS, QueryStatus, RetiredVisibilityBuffer, VisibilityBufferPool,
    VisibilityOffsetAllocator, VisibilityQueryCore, logical_samples, sum_slots,
};
use crate::{page_box::PageBox, render_scale::RenderScale};

fn dummy_buf(seq: u64) -> RetiredVisibilityBuffer {
    // SAFETY: tests; opaque value never dereferenced.
    let handle = unsafe { MetalHandle::new(0xDEAD_BEEF) };
    RetiredVisibilityBuffer::new(PageBox::new_zeroed(8192), handle, seq)
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
    core.begin(10, 3, RenderScale::IDENTITY);
    assert_eq!(core.status(), QueryStatus::Pending);
    assert_eq!(core.offset_begin(), 3);
    // BEGIN does not record seq_end — END drives the GetData(FLUSH)
    // gate.
    assert_eq!(core.seq_end_loaded(), 0);
    core.end(10, 7);
    assert_eq!(core.status(), QueryStatus::Pending);
    assert_eq!(core.offset_end_internal(), 7);
    assert_eq!(core.seq_end_loaded(), 10);
    core.finalize(42);
    assert_eq!(core.status(), QueryStatus::Issued);
    assert_eq!(core.get_u32(), 42);
}

#[test]
fn query_core_reissue_resets_accumulator() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, RenderScale::IDENTITY);
    core.end(1, 1);
    core.finalize(100);
    assert_eq!(core.get_u32(), 100);
    // Re-issue with a different span: accumulator must zero out.
    core.begin(2, 2, RenderScale::IDENTITY);
    assert_eq!(core.status(), QueryStatus::Pending);
    core.end(2, 3);
    core.finalize(7);
    assert_eq!(core.get_u32(), 7);
}

#[test]
fn query_core_u32_clamp() {
    let core = VisibilityQueryCore::new();
    core.begin(0, 0, RenderScale::IDENTITY);
    core.end(0, 1);
    core.finalize(u64::MAX);
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
    c1.begin(5, 0, RenderScale::IDENTITY);
    c1.end(5, 1);
    c2.begin(10, 2, RenderScale::IDENTITY);
    c2.end(10, 3);
    state.push_pending(5, c1.clone());
    state.push_pending(10, c2.clone());
    // Retire one buffer at each seq. No GPU counters in the test
    // buffers → sum is 0, so `intake_completed` finalizes with 0.
    state.pool.retire(dummy_buf(5));
    state.pool.retire(dummy_buf(10));

    // coherent_seq = 7: only c1 (seq 5) should finalize.
    state.intake_completed(7);
    assert_eq!(c1.status(), QueryStatus::Issued);
    assert_eq!(c2.status(), QueryStatus::Pending);
    assert_eq!(state.pending.len(), 1);

    // coherent_seq = 10: c2 finalizes.
    state.intake_completed(10);
    assert_eq!(c2.status(), QueryStatus::Issued);
    assert_eq!(state.pending.len(), 0);
}

#[test]
fn state_reset_frame_clears_per_frame_fields() {
    use super::VisibilityQueryState;
    let mut state = VisibilityQueryState::new();
    state.bump_slot();
    state.inc_active();
    state.install_current_buffer(dummy_buf(0));
    state.mark_exhausted();
    assert_eq!(state.active_count(), 1);
    assert_eq!(state.current_buffer_handle().raw(), 0xDEAD_BEEF);
    assert!(state.exhausted_this_frame());
    assert_eq!(state.allocator.next, 1);

    // `retire_current_buffer` must run first — reset_frame does
    // not touch the current buffer slot.
    state.retire_current_buffer(42);
    state.reset_frame();
    assert_eq!(state.active_count(), 0);
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
fn logical_samples_passes_the_identity_through() {
    assert_eq!(logical_samples(0, 100), 0);
    assert_eq!(logical_samples(307_200, 100), 307_200);
    assert_eq!(logical_samples(u64::MAX, 100), u64::MAX);
    assert_eq!(
        logical_samples(42, 0),
        42,
        "an unparsed percentage is the identity"
    );
}

#[test]
fn logical_samples_divides_by_the_squared_scale_rounding_to_nearest() {
    // A 640x480 frame at 50% rasterizes 320x240 samples.
    assert_eq!(logical_samples(76_800, 50), 307_200);
    // At 75% a 480x360 frame; 172_800 * 16 / 9.
    assert_eq!(logical_samples(172_800, 75), 307_200);
    // One sample at 75% is 1.78 reported pixels, rounded to 2; at 50% it is 4.
    assert_eq!(logical_samples(1, 75), 2);
    assert_eq!(logical_samples(1, 50), 4);
    // 5 samples at 75% are 8.89, rounded to 9.
    assert_eq!(logical_samples(5, 75), 9);
}

#[test]
fn logical_samples_saturates_instead_of_wrapping() {
    assert_eq!(logical_samples(u64::MAX, 50), u64::MAX);
}

#[test]
fn finalize_reports_the_count_in_reported_pixels_of_the_scale_begun_at() {
    let core = VisibilityQueryCore::new();
    core.begin(1, 0, RenderScale::from_percent(50));
    core.end(1, 1);
    core.finalize(76_800);
    assert_eq!(core.get_u32(), 307_200);
    assert_eq!(core.get_u64(), 307_200);
    assert_eq!(core.status(), QueryStatus::Issued);
}
