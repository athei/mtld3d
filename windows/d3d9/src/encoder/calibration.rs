//! PERF-only source-clock publication retained through native shutdown.

use std::{
    sync::{
        Arc,
        atomic::{Ordering, fence},
    },
    thread::JoinHandle,
    time::Duration,
};

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

    /// The calibrated frequency once the worker has published it; `None` while pending or failed.
    pub(super) fn hz(&self) -> Option<u64> {
        self.clock.get().ok().flatten()
    }

    /// Wait for the calibration worker to end, then let its handle go.
    ///
    /// Polls `is_finished` rather than calling `JoinHandle::join`: Wine can
    /// invalidate a thread handle held for a long session, and `join` panics
    /// on the failed wait, which ends the process at device release.
    /// `is_finished` reads the count std keeps on the thread's result, so it
    /// never waits on the OS handle.
    pub(super) fn wait(&mut self) {
        let Some(worker) = self.worker.take() else {
            return;
        };
        while !worker.is_finished() {
            std::thread::sleep(Duration::from_millis(1));
        }
        // `is_finished` may read the count without ordering; the worker's
        // last writes, its publication included, happen before its release
        // of that count, and this fence makes them visible to the check below.
        fence(Ordering::Acquire);
        drop(worker);
        if matches!(self.clock.get(), Ok(None)) {
            log::warn!(target: LOG_TARGET, "source clock calibration ended without a result; timing invalid");
            // SAFETY: the worker has ended without publishing; this is now
            // the only publisher and SourceClock retains the mailbox.
            unsafe { self.clock.publish_failed() };
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
        self.wait();
    }
}
