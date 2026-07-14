use mtld3d_shared::{
    MetalHandle,
    mtl_handle::{
        MTLCommandQueueKind, MTLDeviceKind, MTLRenderPipelineStateKind, MTLTextureKind, NSViewKind,
    },
};
use objc2::rc::Retained;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
};

use super::{
    handle::{IntoRetained, ReleaseRetain},
    macdrv::release_metal_view,
};

// MTLCreateSystemDefaultDevice requires CoreGraphics to be linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Returns (`device_name`, `registry_id`) from the system default Metal device.
pub fn default_device_info() -> Option<(String, u64)> {
    let device = MTLCreateSystemDefaultDevice()?;
    let name = device.name().to_string();
    let registry_id = device.registryID();
    Some((name, registry_id))
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

/// Creates an `MTLDevice` + `MTLCommandQueue` pair.
///
/// Snapshots the device caps the PE side needs at creation time.
pub fn create_command_queue() -> Option<DeviceCaps> {
    let device = MTLCreateSystemDefaultDevice()?;
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
    // matching destroy thunk fires.
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

    // SAFETY: PE side has flushed the GPU and is dropping its only
    // copy of each handle, so the canonical retain transferred at creation
    // time is now ours to release. (Same rationale for each of the five
    // `release_retain` calls below.)
    unsafe { pipeline_handle.release_retain() };
    // SAFETY: as above.
    unsafe { depth_texture_handle.release_retain() };
    // SAFETY: as above.
    unsafe { backbuffer_handle.release_retain() };
    // SAFETY: as above.
    unsafe { queue_handle.release_retain() };
    // SAFETY: as above.
    unsafe { device_handle.release_retain() };
    if !view_handle.is_null() {
        release_metal_view(view_handle);
    }
}
