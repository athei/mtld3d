//! Device-local native encoder lifecycle requests.
//!
//! Only the PE device owning the opaque runtime handle may destroy it. Its API lock
//! excludes concurrent calls while native shutdown drains every queued operation.

use crate::{MetalHandle, Thunk, Thunks, mtl_handle::MTLDeviceKind};

#[cfg(all(test, perf_tracking))]
mod tests;

/// Configuration record: resolved config, GPU capabilities, then optional Unix cache path.
pub const CONFIG_RECORD: u16 = 1;

#[repr(C, align(8))]
pub struct CreateEncoderParams {
    pub device: MetalHandle<MTLDeviceKind>,
    pub config_ptr: u64,
    /// PE-owned aligned `AtomicU32` retained through native destruction.
    pub failure_ptr: u64,
    pub config_len: u32,
    pub result: i32,
    pub runtime: u64,
}

#[repr(C, align(8))]
pub struct DestroyEncoderParams {
    pub runtime: u64,
}

impl Thunk for CreateEncoderParams {
    const CODE: u32 = Thunks::CreateEncoder as u32;
}

impl Thunk for DestroyEncoderParams {
    const CODE: u32 = Thunks::DestroyEncoder as u32;
}

const _: () = {
    assert!(size_of::<CreateEncoderParams>() == 40);
    assert!(align_of::<CreateEncoderParams>() == 8);
    assert!(core::mem::offset_of!(CreateEncoderParams, runtime) == 32);
    assert!(size_of::<DestroyEncoderParams>() == 8);
};

/// One immutable packet borrowed until its replay completion mailbox is published.
#[repr(C, align(8))]
pub struct SubmitEncoderFrameParams {
    pub runtime: u64,
    pub metadata_ptr: u64,
    pub operations_ptr: u64,
    pub completion: u64,
    pub metadata_len: u32,
    pub operations_len: u32,
    /// Zero queues normally, one waits for submit, two also drains GPU retention.
    pub mode: u32,
    /// Set before waiting for acknowledgment, once native admission succeeds.
    pub admitted: u32,
}

/// Ordered native controls that do not carry a frame.
#[repr(C, align(8))]
pub struct EncoderControlParams {
    pub runtime: u64,
    pub argument: u64,
    pub textures_ptr: u64,
    pub textures_len: u32,
    /// Zero drains retention, one intakes visibility, two resets resources.
    pub command: u32,
}

impl Thunk for SubmitEncoderFrameParams {
    const CODE: u32 = Thunks::SubmitEncoderFrame as u32;

    #[cfg(perf_tracking)]
    fn perf_submit_mode(&self) -> Option<crate::encoder_protocol::EncoderSubmitMode> {
        crate::encoder_protocol::EncoderSubmitMode::from_repr(self.mode)
    }
}

impl Thunk for EncoderControlParams {
    const CODE: u32 = Thunks::EncoderControl as u32;
}

const _: () = {
    assert!(size_of::<SubmitEncoderFrameParams>() == 48);
    assert!(align_of::<SubmitEncoderFrameParams>() == 8);
    assert!(size_of::<EncoderControlParams>() == 32);
};
