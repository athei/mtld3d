//! PERF-only source-clock publication retained through native shutdown.

use std::{sync::Arc, thread::JoinHandle};

use mtld3d_shared::clock_calibration::ClockCalibration;

use crate::LOG_TARGET;

struct FailureOnDrop(Arc<ClockCalibration>);

impl Drop for FailureOnDrop {
    fn drop(&mut self) {
        if matches!(self.0.get(), Ok(None)) {
            // SAFETY: this guard owns the only publication on the worker. An
            // early return or unwind must finish pending telemetry before exit.
            unsafe { self.0.publish_failed() };
        }
    }
}

pub(super) struct SourceClock {
    clock: Arc<ClockCalibration>,
    worker: Option<JoinHandle<()>>,
}

impl SourceClock {
    pub(super) fn new() -> Self {
        let clock = Arc::new(ClockCalibration::new());
        let publication = Arc::clone(&clock);
        let worker = match std::thread::Builder::new()
            .name("mtld3d-source-clock".into())
            .spawn(move || {
                let publication = FailureOnDrop(publication);
                let hz = mtld3d_shared::tsc::tsc_hz();
                // SAFETY: this worker owns the sole publication; SourceClock and
                // this Arc retain the mailbox through publication and native use.
                unsafe { publication.0.publish_ready(hz) };
            }) {
            Ok(worker) => Some(worker),
            Err(error) => {
                log::warn!(target: LOG_TARGET, "source clock calibration could not start: {error}; timing invalid");
                // SAFETY: spawn failed, so no worker can publish this mailbox.
                unsafe { clock.publish_failed() };
                None
            }
        };
        Self { clock, worker }
    }

    pub(super) fn address(&self) -> u64 {
        Arc::as_ptr(&self.clock) as u64
    }

    pub(super) fn join(&mut self) {
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::warn!(target: LOG_TARGET, "source clock calibration failed; timing invalid");
            if matches!(self.clock.get(), Ok(None)) {
                // SAFETY: the failed worker is joined and never published; this
                // is now the only publisher and SourceClock retains the mailbox.
                unsafe { self.clock.publish_failed() };
            }
        }
    }

    pub(super) fn retain_after_failed_destroy(&self) {
        // Native readers may still use the mailbox. Only the failed-destruction
        // fallback retains an extra owner, matching the device's other mailboxes.
        let _retained = Arc::into_raw(Arc::clone(&self.clock));
    }
}

impl Drop for SourceClock {
    fn drop(&mut self) {
        self.join();
    }
}
