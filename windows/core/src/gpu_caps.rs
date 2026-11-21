//! Snapshot of `MTLDevice` capabilities the PE side must consult.
//!
//! Every staging-alignment decision reads it.
//!
//! Captured once at `CreateCommandQueue` time and stashed on
//! `DeviceInner`; passed by value into the encoder thread so it never
//! roundtrips through the unix side at draw time.

use mtld3d_shared::mtl::DeviceCapsFlags;

use crate::visibility::{BUFFER_SLOTS_64K, BUFFER_SLOTS_256K};

#[derive(Clone, Copy, Debug)]
pub struct GpuCaps {
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`, in bytes.
    ///
    /// Used as the floor for blit-staging `bytes_per_row`.
    pub min_linear_texture_align: u32,
    /// The device's boolean answers from the `GetDeviceInfo` thunk.
    ///
    /// The encoder consults one of them, `RESOLVE_NEEDS_RETIRE`, through
    /// [`Self::resolve_needs_retire`]; the caps advertisement reads the
    /// rest from its own process-wide copy of the same answer.
    pub device_caps: DeviceCapsFlags,
}

impl GpuCaps {
    /// Default for any host-side test that doesn't care about the platform branch.
    ///
    /// The 16-byte floor matches Apple Silicon.
    #[must_use]
    pub const fn apple_silicon_default() -> Self {
        Self {
            min_linear_texture_align: 16,
            device_caps: DeviceCapsFlags::empty(),
        }
    }

    /// Whether a copy out of a resolve target waits for the resolving command buffer.
    ///
    /// Metal orders the command buffers of one queue, so the answer is no on
    /// every real GPU; the paravirtualized device answers yes because it can
    /// hand a later command buffer's copy the content the target held before
    /// the resolve.
    #[must_use]
    pub const fn resolve_needs_retire(self) -> bool {
        self.device_caps
            .contains(DeviceCapsFlags::RESOLVE_NEEDS_RETIRE)
    }

    /// How many slots one occlusion-query visibility buffer holds on this device.
    ///
    /// The most the device's largest visibility query offset addresses:
    /// [`BUFFER_SLOTS_256K`] where it is 256 KB, [`BUFFER_SLOTS_64K`]
    /// elsewhere. It bounds one buffer, not a frame, which continues in
    /// another buffer once one is full.
    #[must_use]
    pub const fn visibility_buffer_slots(self) -> u32 {
        if self
            .device_caps
            .contains(DeviceCapsFlags::VISIBILITY_OFFSET_256K)
        {
            BUFFER_SLOTS_256K
        } else {
            BUFFER_SLOTS_64K
        }
    }
}

/// Why `CreateDevice` refuses this GPU, or `None` when mtld3d runs on it.
///
/// The layer relies on what every Apple GPU guarantees and no Intel or AMD
/// GPU offers: unified memory, the packed 16-bit pixel formats, and linear
/// filtering of 32-bit float textures. `caps` and `device_name` are the
/// `GetDeviceInfo` answer; an empty name is a process with no Metal device,
/// whose caps are empty too.
#[must_use]
pub fn unsupported_gpu(caps: DeviceCapsFlags, device_name: &str) -> Option<String> {
    if caps.contains(DeviceCapsFlags::APPLE_GPU) {
        return None;
    }
    Some(if device_name.is_empty() {
        "mtld3d supports only Apple Silicon GPUs; this process has no Metal device".to_owned()
    } else {
        format!("mtld3d supports only Apple Silicon GPUs; {device_name} is not one")
    })
}

#[cfg(test)]
mod tests;
