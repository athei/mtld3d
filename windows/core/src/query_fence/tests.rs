use super::{
    MAX_EARLY_OCCLUSION_SUBMITS, OCCLUSION_VISIBLE, early_occlusion_answer,
    early_occlusion_submit_due, end_in_recording_frame, event_completed, event_needs_submit,
};

/// An EVENT query completes only when its frame retires.
///
/// Applications use this query as their own fence for recycling dynamic
/// buffer storage, so an implementation that answers "complete" while the
/// frame is outstanding hands back memory a queued draw is still reading.
#[test]
fn an_event_query_completes_only_once_its_frame_retires() {
    assert!(
        !event_completed(9, 8),
        "a frame the GPU has not reached is not complete",
    );
    assert!(event_completed(9, 9), "the frame's own seq completes it");
    assert!(event_completed(9, 40), "a later seq completes it too");
}

/// A query that was never issued has nothing outstanding.
#[test]
fn a_never_issued_event_query_reads_as_complete() {
    assert!(event_completed(0, 0));
    assert!(event_completed(0, 7));
}

/// A query in the open frame forces that frame to be submitted.
///
/// D3D9 lets a poll without `D3DGETDATA_FLUSH` never complete, and nothing
/// else will push the frame out on the caller's behalf, so the layer submits
/// rather than leave a title that polls without the flag waiting forever.
#[test]
fn an_event_query_in_the_open_frame_forces_a_submit() {
    assert!(
        event_needs_submit(7, 7),
        "issued into the frame being recorded"
    );
    assert!(
        !event_needs_submit(6, 7),
        "an earlier frame was already submitted",
    );
    assert!(
        !event_needs_submit(0, 7),
        "a query never issued waits on nothing"
    );
}

/// An END stamped with the recording frame's seq is unsent; an earlier one is not.
///
/// The occlusion `D3DGETDATA_FLUSH` wait submits the recording frame only in
/// the first case, so a query ended in a frame already handed over costs no
/// mid-frame submission.
#[test]
fn an_end_is_in_the_recording_frame_only_under_its_seq() {
    assert!(
        end_in_recording_frame(7, 7),
        "ended in the frame being recorded"
    );
    assert!(
        !end_in_recording_frame(6, 7),
        "ended in a frame already handed over"
    );
    assert!(!end_in_recording_frame(0, 7), "never ended");
}

/// An early-submit occlusion read never reports a count it did not retire.
///
/// A zeroed or stale count reads as "occluded" and made large on-screen
/// objects vanish; only the real result of the exact issue may be returned,
/// anything else answers "visible".
#[test]
fn an_early_occlusion_read_answers_zero_only_when_it_is_real() {
    assert_eq!(early_occlusion_answer(false, 0), OCCLUSION_VISIBLE);
    assert_eq!(early_occlusion_answer(false, 1234), OCCLUSION_VISIBLE);
    assert_eq!(early_occlusion_answer(true, 0), 0);
    assert_eq!(early_occlusion_answer(true, 1234), 1234);
    assert_eq!(OCCLUSION_VISIBLE, u64::from(u32::MAX));
}

#[test]
fn an_early_occlusion_submit_needs_an_end_and_stays_bounded() {
    assert!(!early_occlusion_submit_due(false, 0));
    assert!(early_occlusion_submit_due(true, 0));
    assert!(early_occlusion_submit_due(
        true,
        MAX_EARLY_OCCLUSION_SUBMITS - 1
    ));
    assert!(!early_occlusion_submit_due(
        true,
        MAX_EARLY_OCCLUSION_SUBMITS
    ));
}
