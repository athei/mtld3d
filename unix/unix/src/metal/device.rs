use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use mtld3d_shared::{
    MetalHandle,
    mtl::{DeviceCapsFlags, PresentDebugFlags},
    mtl_handle::{MTLCommandQueueKind, MTLDeviceKind, MTLTextureKind, NSViewKind},
    record_handle::DeviceRecordHandle,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
    MTLPixelFormat,
};

use super::{
    handle::{IntoRetained, ReleaseRetain},
    macdrv::{detach_metal_layer, retire_metal_view},
    record::DeviceRecord,
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
        log::error!(
            target: LOG_TARGET,
            "MTLCreateSystemDefaultDevice returned no device; the process has no Metal device"
        );
        return MetalHandle::<MTLDeviceKind>::NULL;
    };
    // SAFETY: `Retained::into_raw` transfers the retain into the `u64` and
    // `MetalHandle::new` adopts it. Nothing releases this handle, so the
    // retain stays live for the rest of the process.
    unsafe { MetalHandle::<MTLDeviceKind>::new(Retained::into_raw(device) as u64) }
});

/// The process's Metal device, with one retain for the caller.
fn pinned_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    PINNED_DEVICE.into_retained()
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
    caps.set(DeviceCapsFlags::APPLE_GPU, is_apple_family(&device));
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
    caps.set(
        DeviceCapsFlags::VISIBILITY_OFFSET_256K,
        device.supportsFamily(MTLGPUFamily::Apple7),
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
/// It claims the Mac2 family while implementing less than it, so a feature it
/// is known to reject is gated on its name instead.
fn is_paravirtual(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    is_paravirtual_name(&device.name().to_string())
}

/// True when a Metal device name is the paravirtualized device's.
fn is_paravirtual_name(name: &str) -> bool {
    name.contains("Paravirtual")
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

/// True when a refused texture or view create on the device is worth asking for again.
///
/// Every real GPU refuses a valid descriptor only when it is out of memory,
/// and asking again at once gets the same answer. The paravirtualized device
/// refuses one now and then while several threads create and release
/// textures, with its kernel logging the new object's id as one that already
/// exists; requests on other threads succeed within the same second, so the
/// refusal is not a lasting state of the device. Metal offers no query for
/// the fault, so the device's name is the whole predicate. A device of that
/// name that never shows the fault gains only the retry's bounded wait on a
/// create it refuses.
pub fn refuses_creates_transiently(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    is_paravirtual(device)
}

/// True when the device belongs to Metal's Apple GPU family.
///
/// The Apple family is what an Apple Silicon Mac's GPU claims, and the
/// virtual GPU of a macOS VM on Apple Silicon claims it too; an Intel or AMD
/// GPU claims Mac2 only. The PE side refuses to create a device without it.
pub fn is_apple_family(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    device.supportsFamily(MTLGPUFamily::Apple2)
}

/// Snapshot of the Metal device values the PE side needs.
///
/// Used to pick the staging alignment. Captured once at device creation,
/// never re-queried.
pub struct DeviceCaps {
    pub device_handle: MetalHandle<MTLDeviceKind>,
    /// The device's record, which owns the queue and its presentation state.
    pub record_handle: DeviceRecordHandle,
    /// `device.minimumLinearTextureAlignmentForPixelFormat(BGRA8Unorm)`, in bytes.
    pub min_linear_texture_align: u32,
}

/// Creates an `MTLCommandQueue` on the process's Metal device.
///
/// Snapshots the device caps the PE side needs at creation time. Every D3D
/// device is handed the same `MTLDevice`, so the process-wide Metal caches and
/// the command buffers that bind from them always name one device.
pub fn create_command_queue(
    gate: Option<PathBuf>,
    present_debug: PresentDebugFlags,
) -> Option<DeviceCaps> {
    let device = pinned_device()?;
    let queue = device.newCommandQueue()?;
    let queue_label = objc2_foundation::NSString::from_str("mtld3d");
    queue.setLabel(Some(&queue_label));
    // The record takes the queue's retain before the presenter thread
    // exists, so the thread's own retain at start lands on a live object.
    // SAFETY: `Retained::into_raw` transfers the retain into the raw
    // address, which the record's `Drop` releases.
    let queue_handle =
        unsafe { MetalHandle::<MTLCommandQueueKind>::new(Retained::into_raw(queue) as u64) };
    let record = DeviceRecord::new(queue_handle, gate, present_debug);
    if !super::presenter::spawn(&record) {
        return None;
    }
    let min_linear_texture_align = u32::try_from(
        device.minimumLinearTextureAlignmentForPixelFormat(MTLPixelFormat::BGRA8Unorm),
    )
    .expect("Metal min linear texture alignment fits u32");

    // SAFETY: `Retained::into_raw` transfers the retain into the returned
    // `u64`; `MetalHandle::new` adopts that retain into a typed handle. The
    // PE side keeps the handle alive until the matching destroy thunk fires.
    // The device retain is the one `pinned_device` took for this caller, not
    // the pin's own, so `destroy_command_queue` releasing it leaves the
    // pinned device live for the next D3D device and for the process-wide
    // caches.
    let device_handle =
        unsafe { MetalHandle::<MTLDeviceKind>::new(Retained::into_raw(device) as u64) };
    Some(DeviceCaps {
        device_handle,
        record_handle: record.into_handle(),
        min_linear_texture_align,
    })
}

/// Releases `MTLDevice` + `MTLCommandQueue` + backbuffer + depth texture.
///
/// If `view_handle` is non-null, retires the macdrv metal view: kept for the
/// window's next device, see `macdrv::retire_metal_view`.
pub fn destroy_command_queue(
    device_handle: MetalHandle<MTLDeviceKind>,
    device_record: &Arc<DeviceRecord>,
    view_handle: MetalHandle<NSViewKind>,
    backbuffer_handle: MetalHandle<MTLTextureKind>,
    depth_texture_handle: MetalHandle<MTLTextureKind>,
) {
    let queue_handle = device_record.queue();
    // The presenter first: it may still be inside `nextDrawable` on the
    // layer this call retires, and its last present must be committed before
    // the fence below can order it. The PE side has already drained its
    // submit thread and waited for presentation to go idle.
    super::presenter::stop_and_join(device_record.present());
    // Drop the latched view, layer and window first: the main thread
    // reconciles them against the display it is told about, and this call is
    // about to release the view all three belong to.
    let record = detach_metal_layer(view_handle);

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
    // The device's scratch targets (readback resolve, HDR present) and its
    // MetalFX scalers go after the fence, which is what proves no command
    // buffer of this queue still touches them, and before the queue itself.
    super::upscale::retire(device_record.upscale());

    // The PE side has flushed the GPU and is dropping its only copy of each
    // handle, so the canonical retain transferred at creation time is now
    // ours to release.
    crate::metal::destroy_texture(depth_texture_handle.raw());
    crate::metal::destroy_texture(backbuffer_handle.raw());
    // The queue's own retain rides on the record, which the caller drops
    // once this returns.
    // SAFETY: as above: the device's canonical retain is ours to release.
    unsafe { device_handle.release_retain() };
    if !view_handle.is_null() {
        retire_metal_view(view_handle, record.as_deref());
    }
}

#[cfg(test)]
mod tests;
