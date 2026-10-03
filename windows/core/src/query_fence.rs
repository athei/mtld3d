//! When a `D3DQUERYTYPE_EVENT` query reports completion.
//!
//! The rule lives here beside the other lock and submission decisions so it
//! can be tested without a device: the PE side owns only the seq bookkeeping
//! and the flush.

/// Whether an EVENT query issued at `end_seq` has completed.
///
/// `end_seq` of zero is a query that was never issued, which has nothing
/// outstanding and so reads as complete. Otherwise the GPU has to have
/// retired that frame: an application recycles storage behind this answer,
/// so reporting completion early hands it memory a queued draw still reads.
#[must_use]
pub const fn event_completed(end_seq: u64, coherent_seq: u64) -> bool {
    end_seq == 0 || coherent_seq >= end_seq
}

/// Whether reporting on an EVENT query has to submit the open frame first.
///
/// A query issued into the frame still being recorded can only retire once
/// that frame reaches the GPU. D3D9 leaves a poll without `D3DGETDATA_FLUSH`
/// free never to complete, and the references only submit on the flag; this
/// layer submits either way, because the alternative is a title that polls
/// without it hanging rather than rendering.
#[must_use]
pub const fn event_needs_submit(end_seq: u64, current_seq: u64) -> bool {
    end_in_recording_frame(end_seq, current_seq)
}

/// Whether a query's `Issue(D3DISSUE_END)` stamped with `end_seq` is still in the recording frame.
///
/// `current_seq` is the seq the frame being recorded will be submitted under,
/// so an END stamped with it has not been handed to the encoder, and a wait
/// for its result has to submit that frame first. An END stamped with an
/// earlier seq rode a frame already handed over, and the encoder takes every
/// later request after that frame, so waiting needs no submission of its own.
/// Zero is a query never ended, which is in no frame.
#[must_use]
pub const fn end_in_recording_frame(end_seq: u64, current_seq: u64) -> bool {
    end_seq != 0 && end_seq >= current_seq
}

#[cfg(test)]
mod tests;
