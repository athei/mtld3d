//! Fixed-layout upload events ordered by one device queue's shared counter.

use core::sync::atomic::{AtomicU64, Ordering};

/// An upload either produced commands or requires its original dirty rectangle to be restored.
#[repr(u32)]
pub enum UploadOutcome {
    Declined = 1,
    Emitted = 2,
}

/// An upload emits once and may later decline if an aborted-submit replay fails.
///
/// Each nonzero slot is an event sequence. PE merges events from every retained frame in this
/// sequence, so GPU retention cannot delay an earlier success behind a later failure.
#[repr(C, align(8))]
pub struct UploadFeedback {
    sequence: AtomicU64,
    emitted_notification: AtomicU64,
    declined_notification: AtomicU64,
    emitted: AtomicU64,
    declined: AtomicU64,
}

impl UploadFeedback {
    #[must_use]
    pub const fn new(sequence: u64, emitted_notification: u64, declined_notification: u64) -> Self {
        Self {
            sequence: AtomicU64::new(sequence),
            emitted_notification: AtomicU64::new(emitted_notification),
            declined_notification: AtomicU64::new(declined_notification),
            emitted: AtomicU64::new(0),
            declined: AtomicU64::new(0),
        }
    }

    /// Initialize a recycled record with a new queue counter and notification cells.
    ///
    /// # Safety
    ///
    /// No native publisher or PE event consumer may still access the previous record. Both old
    /// event notifications and its final ownership acknowledgment have been consumed.
    pub unsafe fn reset(&self, sequence: u64, emitted: u64, declined: u64) {
        self.sequence.store(sequence, Ordering::Relaxed);
        self.emitted_notification.store(emitted, Ordering::Relaxed);
        self.declined_notification
            .store(declined, Ordering::Relaxed);
        self.emitted.store(0, Ordering::Relaxed);
        self.declined.store(0, Ordering::Relaxed);
    }

    /// Publish one event in the original queue's order.
    ///
    /// # Safety
    ///
    /// The sequence address names a live aligned `AtomicU64` pinned by the PE queue owner.
    /// Both notification addresses name pinned, initialized and aligned completion cells.
    /// One native encoder serializes publication for this upload.
    ///
    /// # Panics
    ///
    /// Panics on sequence exhaustion rather than reusing a pending event identity.
    pub unsafe fn publish(&self, outcome: &UploadOutcome) {
        let (slot, notification) = match outcome {
            UploadOutcome::Emitted => (
                &self.emitted,
                self.emitted_notification.load(Ordering::Relaxed),
            ),
            UploadOutcome::Declined => (
                &self.declined,
                self.declined_notification.load(Ordering::Relaxed),
            ),
        };
        if slot.load(Ordering::Acquire) != 0 {
            crate::log_once_warn!(target: "mtld3d::unix", "duplicate upload feedback event");
            return;
        }
        // SAFETY: the lease retains the original queue's initialized aligned counter.
        let sequence = unsafe {
            crate::InPtr::<AtomicU64>::new(self.sequence.load(Ordering::Relaxed) as *const _)
        };
        let value = sequence
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                old.checked_add(1)
            })
            .expect("upload feedback sequence exhausted")
            + 1;
        slot.store(value, Ordering::Release);
        // SAFETY: publication's contract retains this aligned initialized notification cell.
        let cell = unsafe {
            crate::InPtr::<crate::encoder_wire::LeaseCompletion>::new(notification as *const _)
        };
        cell.publish();
    }

    /// Notify terminal absence for event slots that were never published.
    ///
    /// # Safety
    ///
    /// Both notification cells are aligned, initialized and pinned until all queued notifications
    /// have been consumed. No more event publication occurs after this call.
    pub unsafe fn finish_notifications(&self) {
        for address in [
            self.emitted_notification.load(Ordering::Relaxed),
            self.declined_notification.load(Ordering::Relaxed),
        ] {
            // SAFETY: the caller retains each initialized notification through consumption.
            let cell = unsafe {
                crate::InPtr::<crate::encoder_wire::LeaseCompletion>::new(address as *const _)
            };
            cell.publish();
        }
    }

    #[must_use]
    pub fn events(&self) -> (u64, u64) {
        (
            self.emitted.load(Ordering::Acquire),
            self.declined.load(Ordering::Acquire),
        )
    }
}

const _: () = {
    assert!(size_of::<UploadFeedback>() == 40);
    assert!(align_of::<UploadFeedback>() == 8);
};
