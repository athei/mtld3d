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

use mtld3d_shared::mtl::StorageMode;

/// Storage mode for VB / IB / visibility / texture-staging buffers.
///
/// That covers anything wrapped via `newBufferWithBytesNoCopy:` over a PE-side
/// `PageBox`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uma_picks_shared() {
        assert_eq!(buffer_storage_mode(true), StorageMode::Shared);
    }

    #[test]
    fn non_uma_picks_managed() {
        assert_eq!(buffer_storage_mode(false), StorageMode::Managed);
    }
}
