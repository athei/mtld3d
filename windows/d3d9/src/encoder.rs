//! PE device proxy for native encoding and retained immutable frame packets.
//!
//! The API lock serializes calls. Packet storage remains guest-owned until native
//! replay and every borrowed resource lease acknowledge completion.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI32, AtomicU32, AtomicU64},
};

pub use mtld3d_core::encoder_data::{
    ColorFillTarget, DepthTransfer, FrameData, FrameDataFlags, FrameInit, ResampledUpload,
    RetiredColorTarget, SubmitFence, TextureInfo, TextureUploadJob, VbibWarmupEntry,
};
use mtld3d_core::{
    config::Mtld3dConfig,
    encoder_packet::{FramePacket, FrameRecorder, PacketLease},
    encoder_value::WireValue,
    gpu_caps::GpuCaps,
    guest_completions::{CompletionDrain, CompletionPool},
    scratch::ScratchArena,
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

#[derive(Default)]
struct LeaseRegistry {
    entries: Vec<Option<PacketLease>>,
    aliases: Vec<Option<usize>>,
    drain: CompletionDrain,
}

impl LeaseRegistry {
    fn insert(&mut self, mut lease: PacketLease, pool: &CompletionPool) {
        if lease.maintain() {
            for slot in lease.into_slots().into_iter().flatten() {
                pool.recycle(slot);
            }
            return;
        }
        let tokens = lease.tokens();
        let primary = usize::try_from(tokens[0].expect("pooled packet lease"))
            .expect("local slot token fits usize");
        if self.entries.len() <= primary {
            self.entries.resize_with(primary + 1, || None);
        }
        assert!(
            self.entries[primary].is_none(),
            "completion owner is unique"
        );
        for token in tokens.into_iter().flatten() {
            let token = usize::try_from(token).expect("local slot token fits usize");
            if self.aliases.len() <= token {
                self.aliases.resize(token + 1, None);
            }
            assert!(self.aliases[token].is_none(), "completion alias is unique");
            self.aliases[token] = Some(primary);
        }
        self.entries[primary] = Some(lease);
    }

    fn drain(&mut self, pool: &CompletionPool) -> usize {
        let mut consumed = 0;
        let Self {
            entries,
            aliases,
            drain,
        } = self;
        pool.drain(drain, 4096, |event| {
            consumed += 1;
            let Ok(token) = usize::try_from(event / 2) else {
                return;
            };
            let Some(primary) = aliases.get(token).copied().flatten() else {
                return;
            };
            let Some(entry) = entries.get_mut(primary) else {
                return;
            };
            if entry.as_mut().is_some_and(PacketLease::maintain)
                && let Some(lease) = entry.take()
            {
                for token in lease.tokens().into_iter().flatten() {
                    let token = usize::try_from(token).expect("registered local token");
                    aliases[token] = None;
                }
                for slot in lease.into_slots().into_iter().flatten() {
                    pool.recycle(slot);
                }
            }
            // Events preceding replay completion leave their consumed state in the
            // retained cells. Inserting the lease checks that state once.
        });
        consumed
    }
}

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
    pending: Mutex<Vec<FramePacket>>,
    failure: AtomicI32,
    native_failure: Box<AtomicU32>,
    recycled: Mutex<Vec<(ScratchArena, FrameRecorder)>>,
    completions: CompletionPool,
    leases: Mutex<LeaseRegistry>,
}

impl EncoderThread {
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
            pending: Mutex::new(Vec::new()),
            failure: AtomicI32::new(D3D_OK),
            native_failure,
            recycled: Mutex::new(Vec::with_capacity(2)),
            completions: CompletionPool::new(),
            leases: Mutex::default(),
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

    pub fn record_failure(&self, status: i32) -> i32 {
        mtld3d_core::encoder_failure::record_failure(&self.failure, status)
    }

    fn maintain_pending(&self) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut leases = self
            .leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending.retain_mut(|packet| {
            for lease in packet.take_leases() { leases.insert(lease, &self.completions); }
            if packet.was_rejected() {
                self.record_failure(D3DERR_DEVICELOST);
            }
            for registration in packet.take_rejected_registrations() {
                let mut params = CancelShaderProgramParams { runtime: self.runtime, registration };
                let status = unix_call(&mut params);
                if status != D3D_OK {
                    log::error!(target: LOG_TARGET, "encoder: shader cancellation failed {status:#x}");
                    self.record_failure(status);
                }
            }
            if let Some(storage) = packet.take_recording_storage() {
                let mut recycled = self.recycled.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if recycled.len() < 2 { recycled.push(storage); }
            }
            !packet.maintain()
        });
        drop(pending);
        leases.drain(&self.completions);
    }

    pub fn reuse_recording_storage(&self, frame: &mut FrameData) {
        self.maintain_pending();
        let storage = self
            .recycled
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop();
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
                self.pending
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(*packet);
                log::error!(target: LOG_TARGET, "encoder: frame encoding failed: {error:?}");
                return Err(self.record_failure(match error {
                    mtld3d_shared::encoder_wire::WireError::AllocationFailed => E_OUTOFMEMORY,
                    _ => D3DERR_DEVICELOST,
                }));
            }
        };
        if let Err(failure) = prior_status {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(packet);
            return Err(failure);
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
        if params.admitted != 0 {
            // SAFETY: the native queue now owns its borrowing contract until completion.
            unsafe {
                packet.mark_admitted();
            }
        }
        // Even rejected metadata can retire allocations used by earlier GPU work.
        // Keep every failed packet until native destruction proves quiescence.
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(packet);
        self.maintain_pending();
        if status != D3D_OK || params.admitted == 0 {
            log::error!(target: LOG_TARGET, "encoder: native frame submission failed {status:#x}, admitted={}", params.admitted);
            return Err(self.record_failure(status));
        }
        self.status()
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
            return Err(self.record_failure(status));
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
        self.source_clock.join();
        let mut params = DestroyEncoderParams {
            runtime: self.runtime,
        };
        let status = unix_call(&mut params);
        if status != D3D_OK {
            log::error!(target: LOG_TARGET, "encoder: native destruction failed {status:#x}");
            return Err(self.record_failure(status));
        }
        self.runtime = 0;
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for packet in pending.iter_mut() {
            // SAFETY: native destruction joined every worker and retired GPU references.
            unsafe {
                packet.cancel_unadopted();
            }
        }
        let leases = self
            .leases
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Native publishers have stopped. Consume every cancellation node before
        // dropping its retained packet, including a detached drain tail.
        while leases.drain(&self.completions) == 4096 {}
        pending.clear();
        drop(pending);
        leases.entries.clear();
        leases.aliases.clear();
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
            let pending = self
                .pending
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::forget(std::mem::take(pending));
            std::mem::forget(std::mem::take(
                self.leases
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ));
        }
    }
}
