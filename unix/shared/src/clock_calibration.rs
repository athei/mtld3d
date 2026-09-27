//! Nonblocking publication of one runtime's calibrated timestamp frequency.

use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicU32, Ordering},
};

#[cfg(test)]
mod tests;

/// Calibration finished without a usable frequency.
#[derive(Debug, PartialEq, Eq)]
pub struct CalibrationFailed;

#[derive(strum::FromRepr)]
#[repr(u32)]
enum CalibrationState {
    Pending = 0,
    Ready = 1,
    Failed = 2,
}

/// Fixed-width mailbox read by either runtime while its owner retains it.
#[repr(C, align(8))]
pub struct ClockCalibration {
    state: AtomicU32,
    reserved: u32,
    hz: UnsafeCell<u64>,
}

// SAFETY: one publisher writes hz before publishing Ready with Release. Readers
// acquire Ready before reading hz, which never changes after publication.
unsafe impl Sync for ClockCalibration {}

impl Default for ClockCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockCalibration {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: AtomicU32::new(CalibrationState::Pending as u32),
            reserved: 0,
            hz: UnsafeCell::new(0),
        }
    }

    /// Read the frequency without waiting for calibration.
    ///
    /// # Errors
    /// Returns `CalibrationFailed` after the publisher reports failure.
    pub fn get(&self) -> Result<Option<u64>, CalibrationFailed> {
        match CalibrationState::from_repr(self.state.load(Ordering::Acquire)) {
            Some(CalibrationState::Pending) => Ok(None),
            // SAFETY: acquiring Ready observes the publisher's completed write;
            // its one-publication contract forbids any subsequent mutation.
            Some(CalibrationState::Ready) => Ok(Some(unsafe { *self.hz.get() })),
            _ => Err(CalibrationFailed),
        }
    }

    /// Publish a completed calibration. Zero reports failure.
    ///
    /// # Safety
    /// This is the mailbox's only publication. No other publisher can access it,
    /// and its owner retains the mailbox until every reader has stopped.
    pub unsafe fn publish_ready(&self, hz: u64) {
        if hz == 0 {
            self.state
                .store(CalibrationState::Failed as u32, Ordering::Release);
        } else {
            // SAFETY: the caller owns the sole publication; readers cannot read
            // this field before the following Release publication.
            unsafe {
                *self.hz.get() = hz;
            }
            self.state
                .store(CalibrationState::Ready as u32, Ordering::Release);
        }
    }

    /// Publish failure without changing rendering state.
    ///
    /// # Safety
    /// This is the mailbox's only publication. Its owner retains it until every
    /// reader has stopped, and no other publisher can access it.
    pub unsafe fn publish_failed(&self) {
        self.state
            .store(CalibrationState::Failed as u32, Ordering::Release);
    }
}

const _: () = {
    assert!(size_of::<ClockCalibration>() == 16);
    assert!(align_of::<ClockCalibration>() == 8);
    assert!(core::mem::offset_of!(ClockCalibration, state) == 0);
    assert!(core::mem::offset_of!(ClockCalibration, reserved) == 4);
    assert!(core::mem::offset_of!(ClockCalibration, hz) == 8);
};
