//! Known PE failures and explicit observation of the native failure mailbox.

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use mtld3d_types::{D3D_OK, D3DERR_DEVICELOST, E_OUTOFMEMORY};

#[cfg(test)]
mod tests;

/// Read the failure already observed by PE without polling native work.
///
/// # Errors
/// Returns the first latched failure, if any.
pub fn known_status(failure: &AtomicI32) -> Result<(), i32> {
    match failure.load(Ordering::Acquire) {
        D3D_OK => Ok(()),
        failure => Err(failure),
    }
}

/// Observe native failure before reporting the first latched PE failure.
///
/// # Errors
/// Returns the first latched failure, including a newly observed native failure.
pub fn status(failure: &AtomicI32, native_failure: &AtomicU32) -> Result<(), i32> {
    if native_failure.load(Ordering::Acquire) != 0 {
        return Err(record_failure(
            failure,
            D3DERR_DEVICELOST,
            "the native runtime failed a frame it had admitted",
        ));
    }
    known_status(failure)
}

/// Latch the first failure, preserving allocation failures as out-of-memory.
///
/// `status` is what the failing step returned and `cause` names that step.
/// The latch that wins logs both once, since every later call that reports
/// device state answers with the latched `HRESULT` and names no cause of its
/// own; a failure after the latch only returns the first one.
pub fn record_failure(failure: &AtomicI32, status: i32, cause: &str) -> i32 {
    let latched = if status == E_OUTOFMEMORY {
        status
    } else {
        D3DERR_DEVICELOST
    };
    match failure.compare_exchange(D3D_OK, latched, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            log::error!(
                target: crate::LOG_TARGET,
                "device failure latched as {latched:#010x}: {cause} (status {status:#x}); \
                 every later call that reports device state returns it until the device is \
                 released"
            );
            latched
        }
        Err(previous) => previous,
    }
}
