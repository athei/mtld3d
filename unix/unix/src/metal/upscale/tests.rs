//! Unit tests for the `MetalFX` scaler cache bound and the per-queue scratch.
//!
//! Scalers are the one Metal object here that is not leaked for the
//! process: each holds tens of MiB of intermediates, and a window drag
//! walks through a fresh geometry per size the user rests at. The first
//! test encodes through four times as many geometries as the cap holds and
//! pins both halves of the bound, that one queue's live entries never exceed
//! `MAX_CACHED_SCALERS`, and that every eviction is released once the
//! following command buffer retires. It skips when the GPU has no
//! `MetalFX`.
//!
//! Both caches in this module are keyed by the queue, which is the device.
//! The map-only tests pin that bookkeeping without a Metal object behind the
//! entries: a bound and an eviction list per queue, and a retire that takes
//! one queue's entries only. The GPU-backed tests pin that two queues at one
//! geometry are served two objects rather than one stateful one.

use mtld3d_shared::{MetalHandle, mtl::PixelFormat, mtl_handle::MTLCommandQueueKind};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
    MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_metal_fx::{MTLFXSpatialScaler, MTLFXSpatialScalerColorProcessingMode};
use rustc_hash::FxHashMap;

use super::{
    MAX_CACHED_SCALERS, ScalerCache, ScalerEntry, ScalerKey, ScalerSlot, ScratchKey,
    evict_least_recently_used, live_for_queue, take_evicted, take_queue_entries,
    take_queue_scalers,
};

/// A queue's live scaler count, or `None` when the GPU has no `MetalFX` at all.
fn cached_scaler_count(queue: MetalHandle<MTLCommandQueueKind>) -> Option<usize> {
    let cache = super::CACHE.get().and_then(Option::as_ref)?;
    let cache = cache.lock().ok()?;
    Some(live_for_queue(&cache, queue.raw()))
}

/// Scalers `queue` has evicted but not yet released.
fn pending_release_count(queue: MetalHandle<MTLCommandQueueKind>) -> usize {
    super::CACHE
        .get()
        .and_then(Option::as_ref)
        .and_then(|cache| cache.lock().ok())
        .and_then(|cache| cache.evicted.get(&queue.raw()).map(Vec::len))
        .unwrap_or(0)
}

/// A `MetalHandle` naming a queue this process owns for the test's duration.
fn queue_handle(
    queue: &objc2::rc::Retained<ProtocolObject<dyn MTLCommandQueue>>,
) -> MetalHandle<MTLCommandQueueKind> {
    // SAFETY: the handle wraps a live queue the caller holds; the cache only
    // reads the raw address as a key.
    unsafe { MetalHandle::<MTLCommandQueueKind>::new(objc2::rc::Retained::as_ptr(queue) as u64) }
}

/// A slot standing for a scaler, never dereferenced and never released.
fn slot(addr: usize) -> ScalerSlot {
    ScalerSlot(core::ptr::without_provenance_mut::<
        ProtocolObject<dyn MTLFXSpatialScaler>,
    >(addr))
}

/// A key for `queue` at the geometry `size` names.
fn scaler_key(queue: u64, size: u32) -> ScalerKey {
    ScalerKey {
        queue,
        input_width: size,
        input_height: size,
        output_width: size * 2,
        output_height: size * 2,
        color_format: MTLPixelFormat::BGRA8Unorm.0,
        output_format: MTLPixelFormat::BGRA8Unorm.0,
        mode: MTLFXSpatialScalerColorProcessingMode::Perceptual,
    }
}

/// The per-queue bound, in the `u32` the geometry helpers count in.
fn bound() -> u32 {
    u32::try_from(MAX_CACHED_SCALERS).expect("the bound is a small constant")
}

/// A cache holding `count` entries for `queue`, oldest first.
fn cache_with(queue: u64, count: u32, first_addr: usize) -> ScalerCache {
    let mut cache = ScalerCache {
        scalers: FxHashMap::default(),
        tick: 0,
        evicted: FxHashMap::default(),
    };
    for step in 0..count {
        cache.tick += 1;
        cache.scalers.insert(
            scaler_key(queue, 64 + step),
            ScalerEntry {
                slot: slot(first_addr + step as usize),
                last_used: cache.tick,
            },
        );
    }
    cache
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
    let queue_handle = queue_handle(&queue);
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
            queue_handle,
            &src,
            &dst,
            MTLFXSpatialScalerColorProcessingMode::Perceptual,
        ) {
            eprintln!("MetalFX unavailable on this GPU — skipping");
            return;
        }
        cmd_buf.commit();
        // Waiting is what lets the eviction handler run before we look.
        cmd_buf.waitUntilCompleted();

        assert!(
            cached_scaler_count(queue_handle).is_none_or(|live| live <= MAX_CACHED_SCALERS),
            "cache grew past {MAX_CACHED_SCALERS} at geometry {step}",
        );
    }

    assert_eq!(
        pending_release_count(queue_handle),
        0,
        "every eviction must be released by the command buffer that followed it"
    );
    assert_eq!(
        cached_scaler_count(queue_handle),
        Some(MAX_CACHED_SCALERS),
        "{geometries} distinct geometries must leave the cache exactly full"
    );
    super::retire_scalers(queue_handle);
    assert_eq!(
        cached_scaler_count(queue_handle),
        Some(0),
        "retiring the queue releases every scaler it held"
    );
}

/// The bound, the eviction list and a retire are all per queue.
///
/// One queue at its cap must evict its own least recently used entry and
/// nothing of the other queue's, park it under its own key, and hand a retire
/// its live and its evicted slots together. Map bookkeeping only: the slots
/// stand for scalers and are never dereferenced or released.
#[test]
fn the_bound_the_eviction_list_and_a_retire_are_per_queue() {
    let mut cache = cache_with(1, bound(), 0x1000);
    for step in 0..bound() {
        cache.tick += 1;
        cache.scalers.insert(
            scaler_key(2, 64 + step),
            ScalerEntry {
                slot: slot(0x2000 + step as usize),
                last_used: cache.tick,
            },
        );
    }
    assert_eq!(live_for_queue(&cache, 1), MAX_CACHED_SCALERS);
    assert_eq!(live_for_queue(&cache, 2), MAX_CACHED_SCALERS);

    evict_least_recently_used(&mut cache, 1);
    assert_eq!(
        live_for_queue(&cache, 1),
        MAX_CACHED_SCALERS - 1,
        "the evicting queue loses one entry"
    );
    assert_eq!(
        live_for_queue(&cache, 2),
        MAX_CACHED_SCALERS,
        "a queue at its own cap keeps every entry when another evicts"
    );
    assert!(
        !cache.scalers.contains_key(&scaler_key(1, 64)),
        "the least recently used of the evicting queue is the victim"
    );

    let mut evicted = take_evicted(&mut cache, 2);
    assert!(evicted.is_empty(), "queue 2 evicted nothing");
    evicted = take_evicted(&mut cache, 1);
    assert_eq!(
        evicted.iter().map(|slot| slot.0).collect::<Vec<_>>(),
        vec![slot(0x1000).0],
        "the eviction waits under the queue whose command buffers order it"
    );

    cache.evicted.entry(1).or_default().push(slot(0x1001));
    let retired = take_queue_scalers(&mut cache, 1);
    assert_eq!(
        retired.len(),
        MAX_CACHED_SCALERS,
        "a retire takes the queue's live entries and its pending eviction"
    );
    assert_eq!(live_for_queue(&cache, 1), 0);
    assert_eq!(
        live_for_queue(&cache, 2),
        MAX_CACHED_SCALERS,
        "the other queue keeps its scalers"
    );
    assert!(
        take_queue_scalers(&mut cache, 3).is_empty(),
        "a queue with no entries retires nothing"
    );
}

/// Two queues at one geometry are served two scalers, not one shared object.
///
/// `MTLFXSpatialScaler` carries its input and output textures as properties
/// across an encode, so one object shared by two presenting devices lets each
/// encode with the other's textures set.
#[test]
fn two_queues_at_one_geometry_get_their_own_scaler() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let (Some(first), Some(second)) = (device.newCommandQueue(), device.newCommandQueue()) else {
        return;
    };
    let (first_handle, second_handle) = (queue_handle(&first), queue_handle(&second));
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
    let (Some(src), Some(dst)) = (
        texture(320, 200, MTLTextureUsage::ShaderRead),
        texture(
            640,
            400,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderWrite,
        ),
    ) else {
        return;
    };

    let scaler = |queue: MetalHandle<MTLCommandQueueKind>| {
        let key = super::scaler_key(
            queue.raw(),
            &src,
            &dst,
            MTLFXSpatialScalerColorProcessingMode::Perceptual,
        )?;
        let cache = super::CACHE.get().and_then(Option::as_ref)?;
        let cache = cache.lock().ok()?;
        cache.scalers.get(&key).map(|entry| entry.slot.0)
    };
    if !super::can_scale(
        &device,
        first_handle,
        &src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ) {
        eprintln!("MetalFX unavailable on this GPU, skipping");
        return;
    }
    assert!(super::can_scale(
        &device,
        second_handle,
        &src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ));

    let (Some(first_scaler), Some(second_scaler)) = (scaler(first_handle), scaler(second_handle))
    else {
        panic!("both queues must have a cached scaler at this geometry");
    };
    assert_ne!(
        first_scaler, second_scaler,
        "two queues at one geometry must not share a stateful scaler"
    );
    assert_eq!(
        scaler(first_handle),
        Some(first_scaler),
        "the same queue at the same geometry gets its scaler back"
    );

    super::retire_scalers(first_handle);
    assert_eq!(
        scaler(first_handle),
        None,
        "the retire takes the first queue's scaler"
    );
    assert_eq!(
        scaler(second_handle),
        Some(second_scaler),
        "retiring the first queue leaves the second's scaler in place"
    );
    super::retire_scalers(second_handle);
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
    let (first_handle, second_handle) = (queue_handle(&first), queue_handle(&second));
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
