//! Device-local native encoder lifecycle requests.
//!
//! Only the PE device owning the opaque runtime handle may destroy it. Its API lock
//! excludes concurrent calls while native shutdown drains every queued operation.

use crate::{
    MetalHandle, Thunk, Thunks, mtl_handle::MTLDeviceKind, record_handle::DeviceRecordHandle,
};

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
    /// PERF source calibration retained through native destruction, or zero.
    pub source_clock_ptr: u64,
    pub record_handle: DeviceRecordHandle,
    /// PE-owned aligned `AtomicU64` counters retained through native destruction.
    pub coherent_seq_ptr: u64,
    pub upload_coherent_seq_ptr: u64,
    pub failed_submit_seq_ptr: u64,
    pub retained_bytes_ptr: u64,
    /// The frame-latency floor: no display-paced present at or below it is outstanding.
    pub paced_present_seq_ptr: u64,
    /// The largest free block at the address-space watch's last walk, in MiB plus one; zero before.
    ///
    /// An `AtomicU64` the device owns, stored by the PE watch thread of a
    /// 32-bit build and read by the encoder for the Metal HUD rows
    /// (`mtld3d_core::hud_stats::largest_free_word`).
    pub largest_free_word_ptr: u64,
    /// `d3d9.dll`'s live page-box bytes, an `AtomicU64` static that lives as long as the image.
    pub guest_page_bytes_ptr: u64,
    /// `d3d9.dll`'s recycle pool's parked bytes, an `AtomicU64` in a process-lifetime static.
    pub guest_pool_bytes_ptr: u64,
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
    assert!(size_of::<CreateEncoderParams>() == 120);
    assert!(align_of::<CreateEncoderParams>() == 8);
    assert!(core::mem::offset_of!(CreateEncoderParams, runtime) == 32);
    assert!(core::mem::offset_of!(CreateEncoderParams, source_clock_ptr) == 40);
    assert!(core::mem::offset_of!(CreateEncoderParams, record_handle) == 48);
    assert!(core::mem::offset_of!(CreateEncoderParams, coherent_seq_ptr) == 56);
    assert!(core::mem::offset_of!(CreateEncoderParams, upload_coherent_seq_ptr) == 64);
    assert!(core::mem::offset_of!(CreateEncoderParams, failed_submit_seq_ptr) == 72);
    assert!(core::mem::offset_of!(CreateEncoderParams, retained_bytes_ptr) == 80);
    assert!(core::mem::offset_of!(CreateEncoderParams, paced_present_seq_ptr) == 88);
    assert!(core::mem::offset_of!(CreateEncoderParams, largest_free_word_ptr) == 96);
    assert!(core::mem::offset_of!(CreateEncoderParams, guest_page_bytes_ptr) == 104);
    assert!(core::mem::offset_of!(CreateEncoderParams, guest_pool_bytes_ptr) == 112);
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
    /// The [`crate::encoder_protocol::EncoderControl`] to run, as its wire value.
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
