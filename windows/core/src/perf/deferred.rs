//! Unit-tagged samples waiting for independently calibrated source and native clocks.

use std::{collections::VecDeque, sync::Arc};

use mtld3d_shared::clock_calibration::ClockCalibration;
use strum::EnumCount;

use super::{CommandBufferRole, FrameSample, clock_scale::scale_ticks};

const MAX_PENDING_SAMPLES: usize = 4096;

#[derive(Default)]
pub(super) struct SubmitNanos {
    pub drawable: u64,
    pub present: u64,
    pub blits: u64,
    pub passes: u64,
    pub commit: u64,
    pub gpu: [u64; CommandBufferRole::COUNT],
}

impl SubmitNanos {
    pub fn apply(self, sample: &mut FrameSample, native_hz: u64) {
        sample.enc.drawable_wait_cycles = scale_ticks(self.drawable, 1_000_000_000, native_hz);
        sample.enc.present_wait_cycles = scale_ticks(self.present, 1_000_000_000, native_hz);
        sample.enc.submit_blits_cycles = scale_ticks(self.blits, 1_000_000_000, native_hz);
        sample.enc.submit_passes_cycles = scale_ticks(self.passes, 1_000_000_000, native_hz);
        sample.enc.submit_commit_cycles = scale_ticks(self.commit, 1_000_000_000, native_hz);
        for (target, ns) in sample.enc.gpu_cycles.iter_mut().zip(self.gpu) {
            *target = scale_ticks(ns, 1_000_000_000, native_hz);
        }
    }
}

pub(super) struct PendingSample {
    pub sample: FrameSample,
    pub compilation: super::compilation::DeferredFrame,
    pub nanos: SubmitNanos,
    pub captured_at: u64,
}

pub(super) struct ClockedSamples {
    source: u64,
    native: Arc<ClockCalibration>,
    pending: VecDeque<PendingSample>,
    failure: Option<&'static str>,
    rejected: u64,
}

impl ClockedSamples {
    /// The device retains source's immutable published mailbox until final drain.
    pub const unsafe fn new(source: u64, native: Arc<ClockCalibration>) -> Self {
        Self {
            source,
            native,
            pending: VecDeque::new(),
            failure: None,
            rejected: 0,
        }
    }

    pub fn frequencies(&self) -> Result<Option<(u64, u64)>, &'static str> {
        if let Some(reason) = self.failure {
            return Err(reason);
        }
        // SAFETY: creation retains this aligned source mailbox through final drain.
        let source = unsafe { &*(self.source as *const ClockCalibration) };
        let source = source.get().map_err(|_| "source calibration failed")?;
        let native = self.native.get().map_err(|_| "native calibration failed")?;
        Ok(source.zip(native))
    }

    pub fn push(&mut self, sample: PendingSample) {
        if self.failure.is_some() {
            self.rejected = self.rejected.saturating_add(1);
            return;
        }
        if self.pending.len() == MAX_PENDING_SAMPLES {
            self.invalidate("calibration pending sample limit exceeded");
            self.rejected = self.rejected.saturating_add(1);
            return;
        }
        self.pending.push_back(sample);
    }

    pub fn take_ready(
        &mut self,
    ) -> Option<(FrameSample, super::compilation::DeferredFrame, u64, u64)> {
        let (source_hz, native_hz) = match self.frequencies() {
            Ok(Some(frequencies)) => frequencies,
            Ok(None) => return None,
            Err(reason) => {
                self.invalidate(reason);
                return None;
            }
        };
        let PendingSample {
            mut sample,
            compilation,
            nanos,
            captured_at,
        } = self.pending.pop_front()?;
        sample.counters.rescale_durations(source_hz, native_hz);
        sample.timing.rescale_durations(source_hz, native_hz);
        nanos.apply(&mut sample, native_hz);
        sample.api_cyc = sample.counters.api_cycles_by_category.iter().sum();
        sample.api_work = sample
            .api_cyc
            .saturating_sub(sample.timing.present_block_cycles);
        sample.outside_d3d9 = sample
            .timing
            .frame_total_cycles
            .saturating_sub(sample.api_cyc);
        Some((sample, compilation, native_hz, captured_at))
    }

    pub fn invalidate(&mut self, reason: &'static str) {
        if self.failure.is_none() {
            log::error!(target: super::LOG_TARGET,
                "perf-invalid: {reason}; retained_samples={}, duration_unit=elapsed_source_ticks; rendering continues",
                self.pending.len());
            self.failure = Some(reason);
        }
    }

    pub fn finish(&mut self) {
        if !self.pending.is_empty() && self.failure.is_none() {
            self.invalidate("calibration incomplete at shutdown");
        }
        if let Some(reason) = self.failure {
            log::error!(target: super::LOG_TARGET,
                "perf-invalid: final reason={reason}, retained_samples={}, rejected_samples={}",
                self.pending.len(), self.rejected);
        }
    }
}

#[cfg(test)]
mod tests;
