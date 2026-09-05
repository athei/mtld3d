//! Picks `MTLStorageMode` for CPU-visible buffers based on the device's unified-memory bit.
//!
//! On Apple Silicon (UMA) every CPU-visible buffer is `Shared` — the GPU
//! reads directly from the same physical pages the CPU wrote, no copy or
//! notification needed. On Intel / AMD discrete GPUs the same allocation has
//! to be `Managed`, and every CPU-side mutation needs a `didModifyRange:` so
//! Metal knows to copy the dirty range to VRAM before the next GPU read.
//!
//! Textures are always `Private` — nothing CPU-writes a texture directly
//! (all uploads are `copyFromBuffer:toTexture:` blits), so only their staging
//! buffers come through here.
//!
//! A buffer the GPU writes and the CPU only reads is the other way round: on
//! a discrete GPU a `Managed` one would hold the GPU's writes in VRAM until a
//! `synchronizeResource:` blit brought them across, so it is `Shared` on every
//! device, where the GPU's writes land in system memory as the command buffer
//! completes.

use mtld3d_shared::mtl::StorageMode;

/// Storage mode for the VB / IB / texture-staging buffers the CPU writes.
///
/// That covers anything wrapped via `newBufferWithBytesNoCopy:` over a PE-side
/// `PageBox` that the CPU fills for the GPU to read.
///
/// Render-target / depth buffers stay `Private` regardless — they're never
/// CPU-visible.
#[must_use]
pub const fn buffer_storage_mode(unified_memory: bool) -> StorageMode {
    if unified_memory {
        StorageMode::Shared
    } else {
        StorageMode::Managed
    }
}

/// Storage mode for a buffer the GPU writes and the CPU reads back.
///
/// The visibility result buffer: the GPU writes the counters, the CPU reads
/// them once the command buffer has completed, and the CPU's only write is
/// the zeroing before reuse. `Shared` on every device, so the counters are
/// readable without a `synchronizeResource:` blit and the zeroing needs no
/// `didModifyRange:`; `Managed` would need both and nothing enqueues either.
#[must_use]
pub const fn gpu_written_buffer_storage_mode() -> StorageMode {
    StorageMode::Shared
}

#[cfg(test)]
mod tests;
