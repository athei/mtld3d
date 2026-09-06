//! Unit tests for the `MetalFX` scaler cache bound and the per-queue scratch.
//!
//! Scalers are the one Metal object here that is not leaked for the
//! process: each holds tens of MiB of intermediates, and a window drag
//! walks through a fresh geometry per size the user rests at. The first
//! test encodes through four times as many geometries as the cap holds and
//! pins both halves of the bound, that the live cache never exceeds
//! `MAX_CACHED_SCALERS`, and that every eviction is released once the
//! following command buffer retires. It skips when the GPU has no
//! `MetalFX`.
//!
//! The scratch tests pin the other cache in this module: one scratch per
//! queue and geometry, and a retire that takes one queue's entries only.

use mtld3d_shared::{MetalHandle, mtl::PixelFormat, mtl_handle::MTLCommandQueueKind};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
    MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
};
use rustc_hash::FxHashMap;

use super::{ScratchKey, take_queue_entries};

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

/// Live scratch entries keyed by `queue`.
fn scratch_entries_for(queue: MetalHandle<MTLCommandQueueKind>) -> usize {
    super::SCRATCH
        .get()
        .and_then(|cache| cache.lock().ok())
        .map_or(0, |scratch| {
            scratch
                .keys()
                .filter(|key| key.queue == queue.raw())
                .count()
        })
}

/// A key for `queue` at one fixed geometry.
fn key(queue: u64) -> ScratchKey {
    ScratchKey {
        queue,
        width: 64,
        height: 64,
        format: PixelFormat::Bgra8Unorm,
    }
}

/// A retire takes every entry of its queue and nothing of another's.
#[test]
fn a_retire_takes_one_queues_entries_only() {
    let mut scratch = FxHashMap::default();
    scratch.insert(key(1), 0x10);
    scratch.insert(
        ScratchKey {
            format: PixelFormat::Rgba16Float,
            ..key(1)
        },
        0x11,
    );
    scratch.insert(key(2), 0x20);

    let mut retired = take_queue_entries(&mut scratch, 1);
    retired.sort_unstable();
    assert_eq!(retired, [0x10, 0x11], "both of queue 1's entries go");
    assert_eq!(scratch.get(&key(2)), Some(&0x20), "queue 2's entry stays");
    assert!(
        take_queue_entries(&mut scratch, 3).is_empty(),
        "a queue with no entries retires nothing"
    );
}

/// Two queues at one geometry get two textures, and a retire frees one queue's.
///
/// The readback resolve of two devices at one back-buffer size used to share
/// a scratch across their queues (#445); a scratch is per queue now, and the
/// device that goes away takes only its own with it.
#[test]
fn two_queues_get_their_own_scratch_and_a_retire_takes_only_its_own() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    let (Some(first), Some(second)) = (device.newCommandQueue(), device.newCommandQueue()) else {
        return;
    };
    // SAFETY: the handles wrap live queues this test owns for its duration;
    // the scratch functions only read the raw address as a key.
    let first_handle = unsafe {
        MetalHandle::<MTLCommandQueueKind>::new(objc2::rc::Retained::as_ptr(&first) as u64)
    };
    // SAFETY: as above.
    let second_handle = unsafe {
        MetalHandle::<MTLCommandQueueKind>::new(objc2::rc::Retained::as_ptr(&second) as u64)
    };
    let scratch = |queue| {
        super::scratch_target(&device, queue, 64, 64, PixelFormat::Bgra8Unorm)
            .map(|texture| objc2::rc::Retained::as_ptr(&texture) as usize)
    };

    let Some(first_scratch) = scratch(first_handle) else {
        eprintln!("no Private scratch on this device — skipping");
        return;
    };
    let second_scratch = scratch(second_handle).expect("the second queue gets a scratch");
    assert_ne!(
        first_scratch, second_scratch,
        "two queues at one geometry must not share a scratch"
    );
    assert_eq!(
        scratch(first_handle),
        Some(first_scratch),
        "the same queue at the same geometry gets its scratch back"
    );

    assert_eq!(scratch_entries_for(first_handle), 1);
    assert_eq!(scratch_entries_for(second_handle), 1);

    super::retire_scratch(first_handle);
    assert_eq!(
        scratch_entries_for(first_handle),
        0,
        "the retire takes the first queue's entry"
    );
    assert_eq!(
        scratch(second_handle),
        Some(second_scratch),
        "retiring the first queue leaves the second's scratch in place"
    );
    // The retired texture's address may be handed out again, so what the
    // next request proves is that the entry was minted afresh, not its value.
    assert!(
        scratch(first_handle).is_some(),
        "the retired queue's next request mints a fresh scratch"
    );
    assert_eq!(scratch_entries_for(first_handle), 1);
    super::retire_scratch(first_handle);
    super::retire_scratch(second_handle);
    assert_eq!(scratch_entries_for(second_handle), 0);
}
