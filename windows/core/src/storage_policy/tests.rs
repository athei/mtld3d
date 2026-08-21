//! Unit tests for the CPU-visible buffer storage-mode policy.
//!
//! Two cases pin `buffer_storage_mode`: unified memory maps to `Shared`,
//! everything else to `Managed`. The mapping is one line, but the `Managed`
//! arm is what obliges callers to issue `didModifyRange:`, and nothing on an
//! Apple Silicon machine would notice it being wrong.

use super::*;

#[test]
fn uma_picks_shared() {
    assert_eq!(buffer_storage_mode(true), StorageMode::Shared);
}

#[test]
fn non_uma_picks_managed() {
    assert_eq!(buffer_storage_mode(false), StorageMode::Managed);
}
