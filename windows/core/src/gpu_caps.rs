//! Snapshot of `MTLDevice` capabilities the PE side must consult.
//!
//! Every storage-mode / alignment / `didModifyRange:` decision reads it.
//!
//! Captured once at `CreateCommandQueue` time and stashed on
//! `DeviceInner`; passed by value into the encoder thread so it never
//! roundtrips through the unix side at draw time.
//!
//! See `storage_policy` for the consumer fns.

#[derive(Clone, Copy, Debug)]
pub struct GpuCaps {
    /// `MTLDevice.hasUnifiedMemory`.
    ///
    /// True on Apple Silicon (and any future UMA Mac), false on Intel
    /// iGPU + AMD dGPU. Drives the Shared-vs-Managed buffer storage
    /// choice and gates whether the encoder enqueues `didModifyRange:`
    /// after CPU writes.
    pub unified_memory: bool,
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`, in bytes.
    ///
    /// 16 on Apple Silicon, 256 on Mac2 (AMD/Intel). Used as the floor
    /// for blit-staging `bytes_per_row`.
    pub min_linear_texture_align: u32,
}

impl GpuCaps {
    /// Default for any host-side test that doesn't care about the platform branch.
    ///
    /// UMA + 16-byte floor matches Apple Silicon.
    #[must_use]
    pub const fn apple_silicon_default() -> Self {
        Self {
            unified_memory: true,
            min_linear_texture_align: 16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_apple_silicon() {
        let caps = GpuCaps::apple_silicon_default();
        assert!(caps.unified_memory);
        assert_eq!(caps.min_linear_texture_align, 16);
    }
}
