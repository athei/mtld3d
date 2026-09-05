//! Unit tests for the CPU-visible buffer storage-mode policy.
//!
//! Two cases pin `buffer_storage_mode`: unified memory maps to `Shared`,
//! everything else to `Managed`. The mapping is one line, but the `Managed`
//! arm is what obliges callers to issue `didModifyRange:`, and nothing on an
//! Apple Silicon machine would notice it being wrong. The third pins the
//! GPU-written buffer to `Shared` whatever the device, since a `Managed`
//! one would read back as zeros there and nothing on Apple Silicon would
//! notice that either.

use super::*;

#[test]
fn uma_picks_shared() {
    assert_eq!(buffer_storage_mode(true), StorageMode::Shared);
}

#[test]
fn non_uma_picks_managed() {
    assert_eq!(buffer_storage_mode(false), StorageMode::Managed);
}

#[test]
fn gpu_written_buffers_are_shared_on_every_device() {
    assert_eq!(gpu_written_buffer_storage_mode(), StorageMode::Shared);
}
