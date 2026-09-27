use super::{EncoderControlParams, SubmitEncoderFrameParams};
use crate::{Thunk, encoder_protocol::EncoderSubmitMode};

#[test]
fn dispatch_trace_reads_typed_submission_modes_and_rejects_invalid_raw_values() {
    let mut params = SubmitEncoderFrameParams {
        runtime: 0,
        metadata_ptr: 0,
        operations_ptr: 0,
        completion: 0,
        metadata_len: 0,
        operations_len: 0,
        mode: u32::from(EncoderSubmitMode::Queue),
        admitted: 0,
    };
    for mode in [
        EncoderSubmitMode::Queue,
        EncoderSubmitMode::WaitForSubmit,
        EncoderSubmitMode::WaitForGpu,
    ] {
        params.mode = u32::from(mode);
        assert_eq!(
            params.perf_submit_mode().map(u32::from),
            Some(u32::from(mode))
        );
    }
    for invalid in [3, u32::MAX] {
        params.mode = invalid;
        assert!(params.perf_submit_mode().is_none());
    }
}

#[test]
fn other_thunks_have_no_submission_mode_even_with_matching_raw_fields() {
    let params = EncoderControlParams {
        runtime: 0,
        argument: 0,
        textures_ptr: 0,
        textures_len: 0,
        command: 0,
    };
    assert!(params.perf_submit_mode().is_none());
}
