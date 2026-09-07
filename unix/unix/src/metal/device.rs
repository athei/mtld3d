use std::sync::LazyLock;

use mtld3d_shared::{
    MetalHandle,
    mtl::DeviceCapsFlags,
    mtl_handle::{
        MTLCommandQueueKind, MTLDeviceKind, MTLRenderPipelineStateKind, MTLTextureKind, NSViewKind,
    },
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
    MTLPixelFormat, MTLStorageMode,
};

use super::{
    handle::{IntoRetained, ReleaseRetain},
    macdrv::{detach_metal_layer, release_metal_view},
};
use crate::LOG_TARGET;

// MTLCreateSystemDefaultDevice requires CoreGraphics to be linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// The one `MTLDevice` the process uses, as a retained handle.
///
/// The process-wide Metal caches (the present and cursor pipelines, the
/// readback resolve library and its pipelines, the null textures, and the
/// upload, clear and blit quads) hold objects built on whichever device first
/// reached them, and Metal rejects an encode that mixes objects of two
/// devices. Resolving the system default device once and handing that same
/// object to every D3D device is what makes their grain correct rather than
/// assumed.
///
/// The retain this handle carries is never released: the device is a
/// process-lifetime object, like the caches built on it. `MetalHandle` is a
/// transparent `u64` newtype, so the static is trivially `Send + Sync`.
static PINNED_DEVICE: LazyLock<MetalHandle<MTLDeviceKind>> = LazyLock::new(|| {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return MetalHandle::<MTLDeviceKind>::NULL;
    };
    // SAFETY: `Retained::into_raw` transfers the retain into the `u64` and
    // `MetalHandle::new` adopts it. Nothing releases this handle, so the
    // retain stays live for the rest of the process.
    unsafe { MetalHandle::<MTLDeviceKind>::new(Retained::into_raw(device) as u64) }
});

/// The process's Metal device, with one retain for the caller.
///
/// Warns once when the system default device is no longer the pinned one,
/// which automatic graphics switching on a dual-GPU Mac can arrange between
/// two D3D device creations. The pinned device is kept in that case, so every
/// object in the process-wide caches keeps belonging to the device the next
/// command buffer encodes against.
fn pinned_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = PINNED_DEVICE.into_retained()?;
    if let Some(current) = MTLCreateSystemDefaultDevice()
        && current.registryID() != device.registryID()
    {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "the system default Metal device is now {} (registry id {}), not the pinned {} \
             (registry id {}); the pinned device is kept, since the process-wide pipeline, \
             texture and buffer caches were built on it",
            current.name(),
            current.registryID(),
            device.name(),
            device.registryID(),
        );
    }
    Some(device)
}

/// Returns (`device_name`, `registry_id`, capability bits) from the process's Metal device.
pub fn default_device_info() -> Option<(String, u64, DeviceCapsFlags)> {
    let device = pinned_device()?;
    let name = device.name().to_string();
    let registry_id = device.registryID();
    let mut caps = DeviceCapsFlags::empty();
    caps.set(
        DeviceCapsFlags::SAMPLER_BORDER,
        supports_sampler_border(&device),
    );
    caps.set(
        DeviceCapsFlags::NATIVE_PACKED16,
        supports_native_packed16(&device),
    );
    caps.set(
        DeviceCapsFlags::FLOAT32_FILTERING,
        supports_float32_filtering(&device),
    );
    caps.set(
        DeviceCapsFlags::SAMPLE_COUNT_2,
        device.supportsTextureSampleCount(2),
    );
    caps.set(
        DeviceCapsFlags::SAMPLE_COUNT_4,
        device.supportsTextureSampleCount(4),
    );
    caps.set(
        DeviceCapsFlags::SAMPLE_COUNT_8,
        device.supportsTextureSampleCount(8),
    );
    caps.set(
        DeviceCapsFlags::RESOLVE_NEEDS_RETIRE,
        resolve_needs_retire(&device),
    );
    Some((name, registry_id, caps))
}

/// True when the device can create border-colour samplers.
///
/// On paper a Mac2-family feature, but the paravirtualized device on CI
/// runners claims Mac2 and still aborts sampler creation on a border
/// colour, so the device is checked for that as well.
pub fn supports_sampler_border(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.supportsFamily(MTLGPUFamily::Mac2) && !is_paravirtual(device)
}

/// True when the device implements the `MirrorClampToEdge` address mode.
///
/// Every GPU family macOS runs on implements it and Metal offers no query for
/// it, so the paravirtualized device is the whole predicate: it rejects the
/// descriptor with `MTLSamplerAddressModeMirrorClampToEdge is not supported on
/// this device`. The sampler path substitutes `MirrorRepeat` when this is
/// false, up front, because the rejected creation is API misuse the validation
/// layer logs whether or not a retry succeeds.
pub fn supports_sampler_mirror_clamp(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    !is_paravirtual(device)
}

/// True when the Metal device is the paravirtualized one a CI runner exposes.
///
/// It answers `supportsFamily:` like the Mac2 device it stands in for while
/// implementing less than one, so a feature it is known to reject is gated on
/// its name instead.
fn is_paravirtual(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.name().to_string().contains("Paravirtual")
}

/// True when a copy out of a resolve target has to wait for the resolve's command buffer.
///
/// Every real GPU orders the command buffers of one queue, so a copy in a
/// later command buffer reads the resolved content. The paravirtualized
/// device can hand that copy the content the target held before the
/// resolve, so the encoder waits for the resolving command buffer to
/// complete first on it. Metal offers no query for the fault, so the
/// device's name is the whole predicate.
fn resolve_needs_retire(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    is_paravirtual(device)
}

/// True when the device belongs to Metal's Apple GPU family.
///
/// The Apple family is what an Apple Silicon Mac's GPU claims; an Intel or
/// AMD GPU claims Mac2 only, and the paravirtualized device a CI runner
/// exposes claims nothing. Two texture rules branch on it: which pixel
/// formats exist natively, and whether a view that changes only transfer
/// function or swizzle needs the `PixelFormatView` usage on its base texture.
pub fn is_apple_family(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.supportsFamily(MTLGPUFamily::Apple2)
}

/// True when the packed 16-bit pixel formats exist natively on this device.
///
/// `B5G6R5Unorm` / `Bgr5A1Unorm` / `Abgr4Unorm` are listed Apple-family-only
/// in Metal's pixel-format capability table; the Mac2 Bronze driver
/// (Intel/AMD) raises a validation abort when a texture descriptor names one.
/// When false, the PE side backs the corresponding D3D formats with
/// `Bgra8Unorm` and widens their texels in the GPU upload pass.
pub fn supports_native_packed16(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    is_apple_family(device)
}

/// Storage mode for a texture the CPU writes with `replaceRegion:`.
///
/// `Shared` where the CPU and GPU share memory; a device without unified
/// memory rejects `Shared` textures outright, and `Managed` is the mode that
/// keeps a CPU-writable copy there. `replaceRegion:` carries the write across
/// on its own for a `Managed` texture, so the caller needs nothing further.
pub fn cpu_written_texture_storage(device: &ProtocolObject<dyn MTLDevice>) -> MTLStorageMode {
    if device.hasUnifiedMemory() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Managed
    }
}

/// True when single-precision float textures sample with linear filtering.
///
/// The query covers exactly `R32Float` / `RG32Float` / `RGBA32Float`; the
/// half-float formats are filterable on every family Metal's pixel-format
/// capability table lists for macOS, so they need no query. The PE side
/// answers `CheckDeviceFormat(D3DUSAGE_QUERY_FILTER)` for R32F / G32R32F /
/// A32B32G32R32F with this bit.
pub fn supports_float32_filtering(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.supports32BitFloatFiltering()
}

/// Snapshot of the Metal device values the PE side needs.
///
/// Used to pick storage modes, alignment, and `didModifyRange:` policy.
/// Captured once at device creation, never re-queried.
pub struct DeviceCaps {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    pub queue_handle: MetalHandle<MTLCommandQueueKind>,
    /// `MTLDevice.hasUnifiedMemory`. False on Intel/AMD discrete GPUs.
    pub unified_memory: bool,
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`, in bytes.
    ///
    /// 16 on Apple Silicon, 256 on Mac2 (AMD/Intel).
    pub min_linear_texture_align: u32,
}

/// Creates an `MTLCommandQueue` on the process's Metal device.
///
/// Snapshots the device caps the PE side needs at creation time. Every D3D
/// device is handed the same `MTLDevice`, so the process-wide Metal caches and
/// the command buffers that bind from them always name one device.
pub fn create_command_queue() -> Option<DeviceCaps> {
    let device = pinned_device()?;
    let queue = device.newCommandQueue()?;
    let queue_label = objc2_foundation::NSString::from_str("mtld3d");
    queue.setLabel(Some(&queue_label));
    let unified_memory = device.hasUnifiedMemory();
    let min_linear_texture_align = u32::try_from(
        device.minimumLinearTextureAlignmentForPixelFormat(MTLPixelFormat::BGRA8Unorm),
    )
    .expect("Metal min linear texture alignment fits u32");

    // SAFETY: `Retained::into_raw` transfers each retain into the
    // returned `u64`; `MetalHandle::new` adopts that retain into a
    // typed handle. The PE side keeps the handle alive until the
    // matching destroy thunk fires. The device retain is the one
    // `pinned_device` took for this caller, not the pin's own, so
    // `destroy_command_queue` releasing it leaves the pinned device live
    // for the next D3D device and for the process-wide caches.
    let device_handle =
        unsafe { MetalHandle::<MTLDeviceKind>::new(Retained::into_raw(device) as u64) };
    // SAFETY: as above.
    let queue_handle =
        unsafe { MetalHandle::<MTLCommandQueueKind>::new(Retained::into_raw(queue) as u64) };
    Some(DeviceCaps {
        device_handle,
        queue_handle,
        unified_memory,
        min_linear_texture_align,
    })
}

/// Releases `MTLDevice` + `MTLCommandQueue` + backbuffer + pipeline.
///
/// If `view_handle` is non-null, releases the macdrv metal view.
pub fn destroy_command_queue(
    device_handle: MetalHandle<MTLDeviceKind>,
    queue_handle: MetalHandle<MTLCommandQueueKind>,
    view_handle: MetalHandle<NSViewKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
    pipeline_handle: MetalHandle<MTLRenderPipelineStateKind>,
    depth_texture_handle: MetalHandle<MTLTextureKind>,
) {
    // Drop the latched view, layer and window first: the main thread
    // reconciles them against the display it is told about, and this call is
    // about to release the view all three belong to.
    detach_metal_layer(view_handle);

    // Force-drain any in-flight or recently-completed command buffers
    // before we drop the queue + device. The PE-side encoder already
    // waited for `coherent_seq` to catch up before destroying its
    // resource handles, but Apple's `addCompletedHandler` can fire
    // *before* Metal's queue-internal release of the command buffer
    // and its referenced resources. Without this fence, MTLBuffers
    // wrapping `bytesNoCopy` PE pages can stay alive past our
    // `objc_release` because Metal's queue-internal retain hasn't
    // dropped yet — a subsequent `newBufferWithBytesNoCopy:` over the
    // same page on a fresh device returns nil ("page already wired").
    // Committing an empty command buffer and `waitUntilCompleted` on
    // the soon-to-be-destroyed queue gives Metal a synchronisation
    // point to flush that internal cleanup.
    if let Some(queue) = queue_handle.into_retained()
        && let Some(fence) = queue.commandBuffer()
    {
        let label = objc2_foundation::NSString::from_str("mtld3d-shutdown-fence");
        fence.setLabel(Some(&label));
        fence.commit();
        fence.waitUntilCompleted();
    }
    // The queue's scratch targets (readback resolve, HDR present) and its
    // MetalFX scalers go after the fence, which is what proves no command
    // buffer of this queue still touches them, and before the queue itself.
    super::upscale::retire_scratch(queue_handle);
    super::upscale::retire_scalers(queue_handle);

    // SAFETY: PE side has flushed the GPU and is dropping its only
    // copy of each handle, so the canonical retain transferred at creation
    // time is now ours to release. (Same rationale for each of the five
    // `release_retain` calls below.)
    unsafe { pipeline_handle.release_retain() };
    // SAFETY: as above.
    crate::metal::destroy_texture(depth_texture_handle.raw());
    crate::metal::destroy_texture(backbuffer_handle.raw());
    // SAFETY: as above.
    unsafe { queue_handle.release_retain() };
    // SAFETY: as above.
    unsafe { device_handle.release_retain() };
    if !view_handle.is_null() {
        release_metal_view(view_handle);
    }
}

#[cfg(test)]
mod tests;
