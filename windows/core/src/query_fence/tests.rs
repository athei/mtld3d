use super::{event_completed, event_needs_submit};

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
