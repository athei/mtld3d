//! Unit test for the bound on the `MetalFX` scaler cache.
//!
//! Scalers are the one Metal object here that is not leaked for the
//! process: each holds tens of MiB of intermediates, and a window drag
//! walks through a fresh geometry per size the user rests at. The test
//! encodes through four times as many geometries as the cap holds and
//! pins both halves of the bound, that the live cache never exceeds
//! `MAX_CACHED_SCALERS`, and that every eviction is released once the
//! following command buffer retires. It skips when the GPU has no
//! `MetalFX`.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
    MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
};

/// Live scaler count, or `None` when the GPU has no `MetalFX` at all.
fn cached_scaler_count() -> Option<usize> {
    let cache = super::CACHE.get().and_then(Option::as_ref)?;
    let cache = cache.lock().ok()?;
    Some(cache.scalers.len())
}

/// Scalers evicted but not yet released.
fn pending_release_count() -> usize {
    super::CACHE
        .get()
        .and_then(Option::as_ref)
        .and_then(|cache| cache.lock().ok().map(|cache| cache.evicted.len()))
        .unwrap_or(0)
}

/// Walking through more geometries than the cap holds evicts, and releases.
///
/// This is the window-resize case: every size the user rests at is a new
/// scaler, and one at `1920x1200 → 2560x1600` costs ~16 MiB of
/// intermediates, so an unbounded cache turns a drag into hundreds of MiB
/// that never come back. Device memory is the wrong thing to assert on
/// (Metal defers deallocation, and the debug layer holds resources for
/// validation), so this asserts the invariant that bounds it instead.
#[test]
fn walking_through_geometries_bounds_the_cache_and_releases_evictions() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    let Some(queue) = device.newCommandQueue() else {
        return;
    };
    let texture = |w: usize, h: usize, usage: MTLTextureUsage| {
        // SAFETY: objc2 typed binding; a class method building a descriptor.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                w,
                h,
                false,
            )
        };
        desc.setUsage(usage);
        desc.setStorageMode(MTLStorageMode::Private);
        device.newTextureWithDescriptor(&desc)
    };

    let geometries = super::MAX_CACHED_SCALERS * 4;
    for step in 0..geometries {
        let (w, h) = (640 + step * 2, 400 + step * 2);
        let (Some(src), Some(dst)) = (
            texture(w / 2, h / 2, MTLTextureUsage::ShaderRead),
            texture(
                w,
                h,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderWrite,
            ),
        ) else {
            return;
        };
        let Some(cmd_buf) = queue.commandBuffer() else {
            return;
        };
        if !super::encode(
            &cmd_buf,
            &device,
            &src,
            &dst,
            super::MTLFXSpatialScalerColorProcessingMode::Perceptual,
        ) {
            eprintln!("MetalFX unavailable on this GPU — skipping");
            return;
        }
        cmd_buf.commit();
        // Waiting is what lets the eviction handler run before we look.
        cmd_buf.waitUntilCompleted();

        assert!(
            cached_scaler_count().is_none_or(|live| live <= super::MAX_CACHED_SCALERS),
            "cache grew past {} at geometry {step}",
            super::MAX_CACHED_SCALERS,
        );
    }

    assert_eq!(
        pending_release_count(),
        0,
        "every eviction must be released by the command buffer that followed it"
    );
    assert_eq!(
        cached_scaler_count(),
        Some(super::MAX_CACHED_SCALERS),
        "{geometries} distinct geometries must leave the cache exactly full"
    );
}
