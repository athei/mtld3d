//! PE device proxy for native encoding and retained immutable frame packets.
//!
//! The API lock serializes calls. Packet storage remains guest-owned until native
//! replay and every borrowed resource lease acknowledge completion.

use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering},
};

pub use mtld3d_core::encoder_data::{
    ColorFillTarget, DepthTransfer, FrameData, FrameDataFlags, FrameInit, ResampledUpload,
    RetiredColorTarget, SubmitFence, TextureInfo, TextureUploadJob, VbibWarmupEntry,
};
use mtld3d_core::{
    config::Mtld3dConfig,
    encoder_packet::{FramePacket, FrameRecorder, PacketRetirement, RetirementHooks},
    encoder_value::WireValue,
    gpu_caps::GpuCaps,
    guest_completions::CompletionPool,
};
use mtld3d_shared::{
    MetalHandle,
    encoder_protocol::{EncoderControl, EncoderSubmitMode},
    encoder_runtime::{
        CONFIG_RECORD, CreateEncoderParams, DestroyEncoderParams, EncoderControlParams,
        SubmitEncoderFrameParams,
    },
    encoder_wire::FrameSlab,
    mtl_handle::MTLDeviceKind,
    record_handle::DeviceRecordHandle,
    shader_create::CancelShaderProgramParams,
};
use mtld3d_types::{D3D_OK, D3DERR_DEVICELOST, E_OUTOFMEMORY};

use crate::{LOG_TARGET, unix_call::unix_call};

#[cfg(perf_tracking)]
mod calibration;

/// PE owners of counters borrowed by native workers for the runtime's lifetime.
pub struct EncoderCounters {
    coherent_seq: Arc<AtomicU64>,
    upload_coherent_seq: Arc<AtomicU64>,
    failed_submit_seq: Arc<AtomicU64>,
    retained_bytes: Arc<AtomicU64>,
}

impl EncoderCounters {
    pub fn coherent_seq(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.coherent_seq)
    }

    pub fn upload_coherent_seq(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.upload_coherent_seq)
    }

    pub fn failed_submit_seq(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.failed_submit_seq)
    }

    pub fn retained_bytes(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.retained_bytes)
    }

    fn new() -> Self {
        Self {
            coherent_seq: Arc::new(AtomicU64::new(0)),
            upload_coherent_seq: Arc::new(AtomicU64::new(0)),
            failed_submit_seq: Arc::new(AtomicU64::new(0)),
            retained_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    fn retain_after_failed_destroy(&self) {
        // An unacknowledged destruction cannot prove native counter access stopped.
        let _coherent = Arc::into_raw(Arc::clone(&self.coherent_seq));
        let _upload = Arc::into_raw(Arc::clone(&self.upload_coherent_seq));
        let _failed = Arc::into_raw(Arc::clone(&self.failed_submit_seq));
        let _retained = Arc::into_raw(Arc::clone(&self.retained_bytes));
    }
}

pub struct EncoderThread {
    runtime: u64,
    counters: EncoderCounters,
    #[cfg(perf_tracking)]
    source_clock: calibration::SourceClock,
    gpu_caps: GpuCaps,
    failure: AtomicI32,
    native_failure: Box<AtomicU32>,
    completions: CompletionPool,
    /// Submitted packets, their handed-over leases and recovered recording storage.
    retirement: Mutex<PacketRetirement>,
    /// `debug.failNextSubmit`, armed until the first submission it refuses.
    fail_next_submit: AtomicBool,
}

impl EncoderThread {
    /// This device's calibrated TSC frequency, `None` until its background worker published it.
    ///
    /// For API-thread perf code that needs a time in cycles without paying
    /// the calibration sleep itself.
    #[cfg(perf_tracking)]
    pub fn source_clock_hz(&self) -> Option<u64> {
        self.source_clock.hz()
    }

    pub fn spawn(
        device: MetalHandle<MTLDeviceKind>,
        record_handle: DeviceRecordHandle,
        gpu_caps: GpuCaps,
        config: &Mtld3dConfig,
    ) -> Result<Self, i32> {
        let cache_path = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|parent| parent.join("mtld3d_shaders.bin")))
            .and_then(|path| crate::wine_path::unix_path(&path));
        if config.shader_cache_enable && cache_path.is_none() {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "encoder: cannot translate game shader cache path");
        }
        let mut settings = FrameSlab::new();
        settings
            .push_record(CONFIG_RECORD, |writer| {
                config.write_wire(writer)?;
                gpu_caps.write_wire(writer)?;
                cache_path.write_wire(writer)
            })
            .map_err(|_| E_OUTOFMEMORY)?;
        let counters = EncoderCounters::new();
        let native_failure = Box::new(AtomicU32::new(0));
        #[cfg(perf_tracking)]
        let source_clock = calibration::SourceClock::new();
        let mut params = CreateEncoderParams {
            failure_ptr: core::ptr::from_ref(native_failure.as_ref()) as u64,
            device,
            record_handle,
            coherent_seq_ptr: Arc::as_ptr(&counters.coherent_seq) as u64,
            upload_coherent_seq_ptr: Arc::as_ptr(&counters.upload_coherent_seq) as u64,
            failed_submit_seq_ptr: Arc::as_ptr(&counters.failed_submit_seq) as u64,
            retained_bytes_ptr: Arc::as_ptr(&counters.retained_bytes) as u64,
            config_ptr: settings.as_bytes().as_ptr() as u64,
            config_len: u32::try_from(settings.as_bytes().len()).map_err(|_| E_OUTOFMEMORY)?,
            result: E_OUTOFMEMORY,
            runtime: 0,
            #[cfg(perf_tracking)]
            source_clock_ptr: source_clock.address(),
            #[cfg(not(perf_tracking))]
            source_clock_ptr: 0,
        };
        let status = unix_call(&mut params);
        if status != D3D_OK || params.result != D3D_OK || params.runtime == 0 {
            return Err(if status != D3D_OK {
                status
            } else if params.result != D3D_OK {
                params.result
            } else {
                E_OUTOFMEMORY
            });
        }
        Ok(Self {
            runtime: params.runtime,
            counters,
            #[cfg(perf_tracking)]
            source_clock,
            gpu_caps,
            failure: AtomicI32::new(D3D_OK),
            native_failure,
            completions: CompletionPool::new(),
            retirement: Mutex::default(),
            fail_next_submit: AtomicBool::new(config.fail_next_submit),
        })
    }

    pub const fn counters(&self) -> &EncoderCounters {
        &self.counters
    }

    #[must_use]
    pub const fn runtime(&self) -> u64 {
        self.runtime
    }

    #[must_use]
    pub const fn gpu_caps(&self) -> GpuCaps {
        self.gpu_caps
    }

    /// Report the first failed encode, admission or synchronous native control.
    ///
    /// # Errors
    ///
    /// Returns the latched HRESULT until the device is destroyed.
    pub fn status(&self) -> Result<(), i32> {
        mtld3d_core::encoder_failure::status(&self.failure, &self.native_failure)
    }

    /// Report a failure already observed by PE without polling native work.
    ///
    /// # Errors
    /// Returns the first latched HRESULT, if any.
    pub fn known_status(&self) -> Result<(), i32> {
        mtld3d_core::encoder_failure::known_status(&self.failure)
    }

    /// Latch `status` as the device failure unless one is latched; `cause` names the step.
    pub fn record_failure(&self, status: i32, cause: &str) -> i32 {
        mtld3d_core::encoder_failure::record_failure(&self.failure, status, cause)
    }

    /// Padded bytes of the texture staging only this device's upload leases still keep.
    ///
    /// Walks every retained page lease under the retirement lock, so it is
    /// for the address-space watch's samples, not for a frame.
    pub fn upload_lease_bytes(&self) -> u64 {
        let mut tally = mtld3d_core::guest_pages::LeaseOnlyPages::default();
        self.lock_retirement().tally_page_leases(&mut tally);
        tally.bytes()
    }

    fn lock_retirement(&self) -> MutexGuard<'_, PacketRetirement> {
        self.retirement
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Retire what native replay and the GPU have finished with.
    ///
    /// Every lease notification and every packet's replay completion lands on the
    /// device's completion queue, so an empty queue means the last pass left nothing
    /// to do, and this returns without taking a lock.
    fn maintain_pending(&self) {
        if !self.completions.has_ready() {
            return;
        }
        self.maintain_retirement(&mut self.lock_retirement());
    }

    fn maintain_retirement(&self, retirement: &mut PacketRetirement) {
        retirement.maintain(&self.completions, &mut RetirementCalls { encoder: self });
    }

    /// Retain a packet whose submission failed and settle its registrations now.
    ///
    /// A packet native code never admitted publishes no notification, so it is
    /// maintained here rather than behind the queue check.
    fn retain_failed(&self, packet: FramePacket) {
        self.retain_submitted(packet, true);
    }

    fn retain_submitted(&self, packet: FramePacket, synchronous: bool) {
        self.lock_retirement().push_submitted(
            packet,
            synchronous,
            &self.completions,
            &mut RetirementCalls { encoder: self },
        );
    }

    /// Hand the next frame the storage a finished packet gave up, or fresh storage.
    ///
    /// This is the one maintenance pass per `Present`: it runs just before the next
    /// frame records, when the most packets can have finished. A waiting submission
    /// maintains once more on its own return.
    pub fn reuse_recording_storage(&self, frame: &mut FrameData) {
        let storage = {
            let mut retirement = self.lock_retirement();
            if self.completions.has_ready() {
                self.maintain_retirement(&mut retirement);
            }
            retirement.take_storage()
        };
        if let Some((scratch, recorder)) = storage {
            frame.scratch = scratch;
            frame.recorder = Some(recorder);
        } else {
            let mut recorder = FrameRecorder::with_completion_pool(self.completions.clone());
            recorder.set_pagebox_pool(&crate::page_box_pool::PAGEBOX_POOL);
            frame.recorder = Some(recorder);
        }
    }

    fn submit(&self, frame: FrameData, mode: EncoderSubmitMode) -> Result<(), i32> {
        let prior_status = self.status();
        let mut packet = match FramePacket::new(frame) {
            Ok(packet) => packet,
            Err((error, packet)) => {
                log::error!(target: LOG_TARGET, "encoder: frame encoding failed: {error:?}");
                let failure = self.record_failure(
                    match error {
                        mtld3d_shared::encoder_wire::WireError::AllocationFailed => E_OUTOFMEMORY,
                        _ => D3DERR_DEVICELOST,
                    },
                    "frame encoding failed",
                );
                self.retain_failed(*packet);
                return Err(failure);
            }
        };
        if let Err(failure) = prior_status {
            self.retain_failed(packet);
            return Err(failure);
        }
        if self.fail_next_submit.load(Ordering::Relaxed) {
            return Err(self.refuse_for_test(packet));
        }
        let mut params = SubmitEncoderFrameParams {
            runtime: self.runtime,
            metadata_ptr: packet.metadata_bytes().as_ptr() as u64,
            operations_ptr: packet.operation_bytes().as_ptr() as u64,
            metadata_len: u32::try_from(packet.metadata_bytes().len())
                .expect("metadata wire length bounded"),
            operations_len: u32::try_from(packet.operation_bytes().len())
                .expect("operation wire length bounded"),
            completion: packet.completion_address(),
            mode: u32::from(mode),
            admitted: 0,
        };
        let status = unix_call(&mut params);
        // Even rejected metadata can retire allocations used by earlier GPU work.
        // Keep every failed packet until native destruction proves quiescence.
        if status != D3D_OK || params.admitted == 0 {
            if params.admitted != 0 {
                // SAFETY: the native queue now owns its borrowing contract until completion.
                unsafe {
                    packet.mark_admitted();
                }
            }
            self.retain_failed(packet);
            log::error!(target: LOG_TARGET, "encoder: native frame submission failed {status:#x}, admitted={}", params.admitted);
            return Err(self.record_failure(status, "native frame submission failed"));
        }
        // SAFETY: the native queue now owns its borrowing contract until completion.
        unsafe {
            packet.mark_admitted();
        }
        // A waiting submission (a mid-frame flush, the retention tier) frees what native
        // code released before returning, as its callers promise; `Present` leaves that to
        // the next frame's pass.
        self.retain_submitted(packet, mode != EncoderSubmitMode::Queue);
        self.status()
    }

    /// Refuse `packet` the way a native rejection would, for `debug.failNextSubmit`.
    ///
    /// Out of line and cold: the key is a test seam, armed only in the
    /// suite, and the submission path pays one relaxed load for it.
    #[cold]
    #[inline(never)]
    fn refuse_for_test(&self, packet: FramePacket) -> i32 {
        self.fail_next_submit.store(false, Ordering::Relaxed);
        log::error!(target: LOG_TARGET, "encoder: debug.failNextSubmit refused the frame submission");
        self.retain_failed(packet);
        self.record_failure(
            D3DERR_DEVICELOST,
            "debug.failNextSubmit refused the submission",
        )
    }

    pub fn send_frame(&self, frame: FrameData) -> Result<(), i32> {
        self.submit(frame, EncoderSubmitMode::Queue)
    }
    pub fn mid_frame_submit(&self, frame: FrameData) -> Result<(), i32> {
        self.submit(frame, EncoderSubmitMode::WaitForSubmit)
    }
    pub fn mid_frame_submit_for_retention(&self, frame: FrameData) -> Result<(), i32> {
        self.submit(frame, EncoderSubmitMode::WaitForGpu)
    }

    fn control(&self, command: EncoderControl, argument: u64, textures: &[u64]) -> Result<(), i32> {
        self.status()?;
        let command = u32::from(command);
        let mut params = EncoderControlParams {
            runtime: self.runtime,
            command,
            argument,
            textures_ptr: textures.as_ptr() as u64,
            textures_len: u32::try_from(textures.len()).expect("texture handle count fits u32"),
        };
        let status = unix_call(&mut params);
        if status != D3D_OK {
            log::error!(target: LOG_TARGET, "encoder: native control {command} failed {status:#x}");
            return Err(self.record_failure(status, "native encoder control failed"));
        }
        self.maintain_pending();
        self.status()
    }

    pub fn drain_retired_now(&self) -> Result<(), i32> {
        self.control(EncoderControl::DrainRetention, 0, &[])
    }
    pub fn intake_visibility_for(&self, seq: u64) -> Result<(), i32> {
        self.control(EncoderControl::IntakeVisibility, seq, &[])
    }
    pub fn reset(&self, textures: &[u64]) -> Result<(), i32> {
        self.control(EncoderControl::Reset, 0, textures)
    }

    pub fn shutdown(&mut self) -> Result<(), i32> {
        if self.runtime == 0 {
            return Ok(());
        }
        #[cfg(perf_tracking)]
        self.source_clock.wait();
        let mut params = DestroyEncoderParams {
            runtime: self.runtime,
        };
        let status = unix_call(&mut params);
        if status != D3D_OK {
            log::error!(target: LOG_TARGET, "encoder: native destruction failed {status:#x}");
            return Err(self.record_failure(status, "native encoder destruction failed"));
        }
        self.runtime = 0;
        let retirement = self
            .retirement
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: native destruction joined every worker and retired GPU references, so
        // no native user or publisher remains.
        unsafe { retirement.cancel_after_quiescence(&self.completions) };
        // Every staging lease this device handed over has retired above, so no native
        // reference to a parked staging box remains; free them rather than keep committed
        // pages for a texture set that has gone with its device. The lane is process-wide,
        // so boxes other live devices parked go too; they only lose warm pages.
        crate::page_box_pool::PAGEBOX_POOL.drain_staging();
        Ok(())
    }
}

impl Drop for EncoderThread {
    fn drop(&mut self) {
        let _shutdown = self.shutdown();
        if self.runtime != 0 {
            self.counters.retain_after_failed_destroy();
            #[cfg(perf_tracking)]
            self.source_clock.retain_after_failed_destroy();
            std::mem::forget(std::mem::replace(
                &mut self.native_failure,
                Box::new(AtomicU32::new(0)),
            ));
            // A failed destruction cannot prove native readers have stopped.
            // Keep guest backing alive rather than free memory still borrowed by them.
            self.retirement
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .forget_native_owners();
        }
    }
}

/// The device calls one maintenance pass makes: failure latching and shader cancellation.
struct RetirementCalls<'a> {
    encoder: &'a EncoderThread,
}

impl RetirementHooks for RetirementCalls<'_> {
    fn packet_rejected(&mut self) {
        self.encoder.record_failure(
            D3DERR_DEVICELOST,
            "the native runtime rejected a frame packet",
        );
    }

    fn cancel_registration(&mut self, registration: u64) {
        let mut params = CancelShaderProgramParams {
            runtime: self.encoder.runtime,
            registration,
        };
        let status = unix_call(&mut params);
        if status != D3D_OK {
            log::error!(target: LOG_TARGET, "encoder: shader cancellation failed {status:#x}");
            self.encoder
                .record_failure(status, "native shader cancellation failed");
        }
    }
}
