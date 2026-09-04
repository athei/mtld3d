//! Snapshot of `MTLDevice` capabilities the PE side must consult.
//!
//! Every storage-mode / alignment / `didModifyRange:` decision reads it.
//!
//! Captured once at `CreateCommandQueue` time and stashed on
//! `DeviceInner`; passed by value into the encoder thread so it never
//! roundtrips through the unix side at draw time.
//!
//! The `intel.*` config keys fold in through [`GpuCaps::with_intel_overrides`]
//! before the snapshot is stored, so every consumer sees the Intel answer
//! when one is forced and no consumer consults the config itself.
//!
//! See `storage_policy` for the consumer fns.

/// The linear-texture row alignment a Mac2 (Intel/AMD) device reports for `BGRA8Unorm`.
///
/// Apple-family devices report 16. `intel.linearAlign256` raises a smaller
/// device value to this floor so the padded-staging and upload-pass paths
/// those devices take run on Apple Silicon too.
pub const MAC2_LINEAR_TEXTURE_ALIGN: u32 = 256;

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

    /// Apply the `intel.managedMemory` and `intel.linearAlign256` answers over the device's.
    ///
    /// Each override only ever moves a field towards the Mac2 answer:
    /// `managed_memory` clears `unified_memory`, `linear_align256` raises the
    /// alignment floor to [`MAC2_LINEAR_TEXTURE_ALIGN`] and keeps a larger
    /// device value. A device that already gives the Mac2 answer is left as
    /// it is, so the result describes an Intel/AMD Mac whether it was
    /// measured or forced.
    #[must_use]
    pub const fn with_intel_overrides(self, managed_memory: bool, linear_align256: bool) -> Self {
        let min_linear_texture_align =
            if linear_align256 && self.min_linear_texture_align < MAC2_LINEAR_TEXTURE_ALIGN {
                MAC2_LINEAR_TEXTURE_ALIGN
            } else {
                self.min_linear_texture_align
            };
        Self {
            unified_memory: self.unified_memory && !managed_memory,
            min_linear_texture_align,
        }
    }
}

#[cfg(test)]
mod tests;
