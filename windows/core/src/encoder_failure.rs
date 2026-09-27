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
        return Err(record_failure(failure, D3DERR_DEVICELOST));
    }
    known_status(failure)
}

/// Latch the first failure, preserving allocation failures as out-of-memory.
pub fn record_failure(failure: &AtomicI32, status: i32) -> i32 {
    let status = if status == E_OUTOFMEMORY {
        status
    } else {
        D3DERR_DEVICELOST
    };
    match failure.compare_exchange(D3D_OK, status, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => status,
        Err(previous) => previous,
    }
}
