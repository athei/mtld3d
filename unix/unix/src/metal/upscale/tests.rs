//! Unit tests for the `MetalFX` scaler cache bound and the per-device scratch.
//!
//! Scalers are the one Metal object here that is not leaked for the
//! process: each holds tens of MiB of intermediates, and a window drag
//! walks through a fresh geometry per size the user rests at. The first
//! test encodes through four times as many geometries as the cap holds and
//! pins both halves of the bound, that a device's live entries never exceed
//! `MAX_CACHED_SCALERS`, and that every eviction is released once the
//! following command buffer retires. It skips when the GPU has no
//! `MetalFX`.
//!
//! Both caches in this module belong to one device's record. The map-only
//! tests pin that bookkeeping without a Metal object behind the entries: a
//! bound and an eviction list per cache, and a retire that empties one cache
//! and leaves another alone. The GPU-backed tests pin that two devices at one
//! geometry are served two objects rather than one stateful one.

use mtld3d_shared::mtl::PixelFormat;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_metal_fx::{MTLFXSpatialScaler, MTLFXSpatialScalerColorProcessingMode};
use rustc_hash::FxHashMap;

use super::{
    MAX_CACHED_SCALERS, ScalerCache, ScalerEntry, ScalerKey, ScalerSlot, ScratchKey, UpscaleCache,
    evict_least_recently_used, take_evicted, take_scalers,
};

/// A cache's live scaler count.
fn cached_scaler_count(cache: &UpscaleCache) -> Option<usize> {
    Some(cache.scalers.lock().ok()?.scalers.len())
}

/// Scalers a cache has evicted but not yet released.
fn pending_release_count(cache: &UpscaleCache) -> usize {
    cache
        .scalers
        .lock()
        .map_or(0, |scalers| scalers.evicted.len())
}

/// A slot standing for a scaler, never dereferenced and never released.
fn slot(addr: usize) -> ScalerSlot {
    ScalerSlot {
        scaler: core::ptr::without_provenance_mut::<ProtocolObject<dyn MTLFXSpatialScaler>>(addr),
        output: core::ptr::null_mut(),
    }
}

/// A key at the geometry `size` names.
fn scaler_key(size: u32) -> ScalerKey {
    ScalerKey {
        input_width: size,
        input_height: size,
        output_width: size * 2,
        output_height: size * 2,
        color_format: MTLPixelFormat::BGRA8Unorm.0,
        output_format: MTLPixelFormat::BGRA8Unorm.0,
        mode: MTLFXSpatialScalerColorProcessingMode::Perceptual,
    }
}

/// The per-device bound, in the `u32` the geometry helpers count in.
fn bound() -> u32 {
    u32::try_from(MAX_CACHED_SCALERS).expect("the bound is a small constant")
}

/// A cache holding `count` entries, oldest first.
fn cache_with(count: u32, first_addr: usize) -> ScalerCache {
    let mut cache = ScalerCache {
        scalers: FxHashMap::default(),
        tick: 0,
        evicted: Vec::new(),
    };
    for step in 0..count {
        cache.tick += 1;
        cache.scalers.insert(
            scaler_key(64 + step),
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
    let cache = UpscaleCache::new();
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
            &cache,
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
            cached_scaler_count(&cache).is_none_or(|live| live <= MAX_CACHED_SCALERS),
            "cache grew past {MAX_CACHED_SCALERS} at geometry {step}",
        );
    }

    assert_eq!(
        pending_release_count(&cache),
        0,
        "every eviction must be released by the command buffer that followed it"
    );
    assert_eq!(
        cached_scaler_count(&cache),
        Some(MAX_CACHED_SCALERS),
        "{geometries} distinct geometries must leave the cache exactly full"
    );
    super::retire(&cache);
    assert_eq!(
        cached_scaler_count(&cache),
        Some(0),
        "retiring the device releases every scaler it held"
    );
}

/// The bound, the eviction list and a retire all belong to one device.
///
/// A device at its cap must evict its own least recently used entry and park
/// it in its own list, and a retire must hand back its live and its evicted
/// slots together while another device's cache keeps everything. Map
/// bookkeeping only: the slots stand for scalers and are never dereferenced
/// or released.
#[test]
fn the_bound_the_eviction_list_and_a_retire_are_per_device() {
    let mut first = cache_with(bound(), 0x1000);
    let mut second = cache_with(bound(), 0x2000);
    assert_eq!(first.scalers.len(), MAX_CACHED_SCALERS);
    assert_eq!(second.scalers.len(), MAX_CACHED_SCALERS);

    evict_least_recently_used(&mut first);
    assert_eq!(
        first.scalers.len(),
        MAX_CACHED_SCALERS - 1,
        "the evicting device loses one entry"
    );
    assert_eq!(
        second.scalers.len(),
        MAX_CACHED_SCALERS,
        "a device at its own cap keeps every entry when another evicts"
    );
    assert!(
        !first.scalers.contains_key(&scaler_key(64)),
        "the least recently used entry is the victim"
    );

    assert!(
        take_evicted(&mut second).is_empty(),
        "the second device evicted nothing"
    );
    let evicted = take_evicted(&mut first);
    assert_eq!(
        evicted.iter().map(|slot| slot.scaler).collect::<Vec<_>>(),
        vec![slot(0x1000).scaler],
        "the eviction waits for a command buffer of the device that made it"
    );

    first.evicted.push(slot(0x1001));
    let retired = take_scalers(&mut first);
    assert_eq!(
        retired.len(),
        MAX_CACHED_SCALERS,
        "a retire takes the live entries and the pending eviction"
    );
    assert_eq!(first.scalers.len(), 0);
    assert_eq!(
        second.scalers.len(),
        MAX_CACHED_SCALERS,
        "the other device keeps its scalers"
    );
    assert!(
        take_scalers(&mut first).is_empty(),
        "a cache with no entries retires nothing"
    );
}

/// Two devices at one geometry are served two scalers, not one shared object.
///
/// `MTLFXSpatialScaler` carries its input and output textures as properties
/// across an encode, so one object shared by two presenting devices lets each
/// encode with the other's textures set.
#[test]
fn two_devices_at_one_geometry_get_their_own_scaler() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil, skipping");
        return;
    };
    let (first_cache, second_cache) = (UpscaleCache::new(), UpscaleCache::new());
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

    let scaler = |cache: &UpscaleCache| {
        let key = super::scaler_key(
            &src,
            &dst,
            MTLFXSpatialScalerColorProcessingMode::Perceptual,
        )?;
        let scalers = cache.scalers.lock().ok()?;
        scalers.scalers.get(&key).map(|entry| entry.slot.scaler)
    };
    if !super::can_scale(
        &device,
        &first_cache,
        &src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ) {
        eprintln!("MetalFX unavailable on this GPU, skipping");
        return;
    }
    assert!(super::can_scale(
        &device,
        &second_cache,
        &src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ));

    let (Some(first_scaler), Some(second_scaler)) = (scaler(&first_cache), scaler(&second_cache))
    else {
        panic!("both devices must have a cached scaler at this geometry");
    };
    assert_ne!(
        first_scaler, second_scaler,
        "two devices at one geometry must not share a stateful scaler"
    );
    assert_eq!(
        scaler(&first_cache),
        Some(first_scaler),
        "the same device at the same geometry gets its scaler back"
    );

    super::retire(&first_cache);
    assert_eq!(
        scaler(&first_cache),
        None,
        "the retire takes the first device's scaler"
    );
    assert_eq!(
        scaler(&second_cache),
        Some(second_scaler),
        "retiring the first device leaves the second's scaler in place"
    );
    super::retire(&second_cache);
}

/// Live scratch entries in one device's cache.
fn scratch_entries(cache: &UpscaleCache) -> usize {
    cache.scratch.lock().map_or(0, |scratch| scratch.len())
}

/// A key at one fixed geometry.
fn key() -> ScratchKey {
    ScratchKey {
        width: 64,
        height: 64,
        format: PixelFormat::Bgra8Unorm,
    }
}

/// A retire empties one device's scratch map and leaves another's alone.
#[test]
fn a_retire_takes_one_devices_entries_only() {
    let first = UpscaleCache::new();
    let second = UpscaleCache::new();
    {
        let mut scratch = first.scratch.lock().expect("a fresh cache lock");
        scratch.insert(key(), 0x10);
        scratch.insert(
            ScratchKey {
                format: PixelFormat::Rgba16Float,
                ..key()
            },
            0x11,
        );
    }
    second
        .scratch
        .lock()
        .expect("a fresh cache lock")
        .insert(key(), 0x20);

    let mut retired: Vec<u64> = first
        .scratch
        .lock()
        .expect("a fresh cache lock")
        .drain()
        .map(|(_, handle)| handle)
        .collect();
    retired.sort_unstable();
    assert_eq!(retired, [0x10, 0x11], "both of the device's entries go");
    assert_eq!(
        second
            .scratch
            .lock()
            .expect("a fresh cache lock")
            .get(&key()),
        Some(&0x20),
        "the other device's entry stays"
    );
}

/// Two devices at one geometry get two textures, and a retire frees one device's.
///
/// The readback resolve of two devices at one back-buffer size used to share
/// a scratch across their queues (#445); a scratch belongs to a device now,
/// and the device that goes away takes only its own with it.
#[test]
fn two_devices_get_their_own_scratch_and_a_retire_takes_only_its_own() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        eprintln!("MTLCreateSystemDefaultDevice returned nil — skipping");
        return;
    };
    let (first_cache, second_cache) = (UpscaleCache::new(), UpscaleCache::new());
    let scratch = |cache: &UpscaleCache| {
        super::scratch_target(&device, cache, 64, 64, PixelFormat::Bgra8Unorm)
            .map(|texture| objc2::rc::Retained::as_ptr(&texture) as usize)
    };

    let Some(first_scratch) = scratch(&first_cache) else {
        eprintln!("no Private scratch on this device — skipping");
        return;
    };
    let second_scratch = scratch(&second_cache).expect("the second device gets a scratch");
    assert_ne!(
        first_scratch, second_scratch,
        "two devices at one geometry must not share a scratch"
    );
    assert_eq!(
        scratch(&first_cache),
        Some(first_scratch),
        "the same device at the same geometry gets its scratch back"
    );

    assert_eq!(scratch_entries(&first_cache), 1);
    assert_eq!(scratch_entries(&second_cache), 1);

    super::retire(&first_cache);
    assert_eq!(
        scratch_entries(&first_cache),
        0,
        "the retire takes the first device's entry"
    );
    assert_eq!(
        scratch(&second_cache),
        Some(second_scratch),
        "retiring the first device leaves the second's scratch in place"
    );
    // The retired texture's address may be handed out again, so what the
    // next request proves is that the entry was minted afresh, not its value.
    assert!(
        scratch(&first_cache).is_some(),
        "the retired device's next request mints a fresh scratch"
    );
    assert_eq!(scratch_entries(&first_cache), 1);
    super::retire(&first_cache);
    super::retire(&second_cache);
    assert_eq!(scratch_entries(&second_cache), 0);
}

/// GPU fixtures fail on unexpected allocation errors once `MetalFX` is supported.
fn gpu() -> Option<objc2::rc::Retained<ProtocolObject<dyn MTLDevice>>> {
    let device = MTLCreateSystemDefaultDevice()?;
    if !super::is_available(&device) {
        eprintln!("MetalFX unsupported, skipping GPU regression");
        return None;
    }
    Some(device)
}

fn target(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
    format: MTLPixelFormat,
    storage: MTLStorageMode,
    usage: MTLTextureUsage,
) -> objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLTexture>> {
    // SAFETY: a small, single-level, square texture with a supported color format.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, size, size, false,
        )
    };
    desc.setStorageMode(storage);
    desc.setUsage(usage);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("test texture");
    objc2_metal::MTLResource::setLabel(
        &*texture,
        Some(&objc2_foundation::NSString::from_str("mtld3d-test-upscale")),
    );
    texture
}

fn fill(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn objc2_metal::MTLTexture>,
    red: f64,
) {
    use objc2_metal::{MTLCommandEncoder, MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction};
    let pass = MTLRenderPassDescriptor::new();
    // SAFETY: attachment zero exists on every render pass descriptor.
    let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    color.setTexture(Some(texture));
    color.setLoadAction(MTLLoadAction::Clear);
    color.setStoreAction(MTLStoreAction::Store);
    color.setClearColor(objc2_metal::MTLClearColor {
        red,
        green: 0.25,
        blue: 0.5,
        alpha: 1.0,
    });
    let encoder = cmd
        .renderCommandEncoderWithDescriptor(&pass)
        .expect("clear encoder");
    encoder.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-test-upscale-clear",
    )));
    encoder.endEncoding();
}

fn pixels(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn objc2_metal::MTLTexture>,
) -> Vec<u8> {
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
        MTLResourceOptions,
    };
    let bytes_per_pixel = if texture.pixelFormat() == MTLPixelFormat::RGBA16Float {
        8
    } else {
        4
    };
    let stride = (texture.width() * bytes_per_pixel).next_multiple_of(256);
    let length = stride * texture.height();
    let buffer = queue
        .device()
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .expect("readback");
    objc2_metal::MTLResource::setLabel(
        &*buffer,
        Some(&objc2_foundation::NSString::from_str(
            "mtld3d-test-upscale-pixels",
        )),
    );
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-test-upscale-readback",
    )));
    let blit = cmd.blitCommandEncoder().expect("readback encoder");
    blit.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-test-upscale-readback-copy",
    )));
    // SAFETY: the full texture fits the aligned buffer; both resources stay
    // alive until the command buffer completes below.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture, 0, 0, objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            objc2_metal::MTLSize { width: texture.width(), height: texture.height(), depth: 1 },
            &buffer, 0, stride, length,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    let mut pixels = Vec::with_capacity(texture.width() * texture.height() * bytes_per_pixel);
    for row in 0..texture.height() {
        // SAFETY: each row's pixels were initialized by the completed copy;
        // padding is excluded and the buffer owns every addressed row.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .wrapping_add(row * stride),
                texture.width() * bytes_per_pixel,
            )
        };
        pixels.extend_from_slice(bytes);
    }
    pixels
}

fn output_count(cache: &UpscaleCache) -> usize {
    cache
        .scalers
        .lock()
        .expect("cache lock")
        .scalers
        .values()
        .filter(|entry| !entry.slot.output.is_null())
        .count()
}

/// The indirect route must produce exactly the same pixels as direct Private output.
#[test]
fn managed_outputs_match_private_outputs_in_sdr_and_hdr() {
    use objc2_metal::{MTLCommandBufferStatus, MTLResource};
    let Some(device) = gpu() else { return };
    let queue = device.newCommandQueue().expect("queue");
    let cache = UpscaleCache::new();
    for (format, mode, red) in [
        (
            MTLPixelFormat::BGRA8Unorm,
            MTLFXSpatialScalerColorProcessingMode::Perceptual,
            0.75,
        ),
        (
            MTLPixelFormat::RGBA16Float,
            MTLFXSpatialScalerColorProcessingMode::HDR,
            2.0,
        ),
    ] {
        let src = target(
            &device,
            32,
            format,
            MTLStorageMode::Private,
            MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
        );
        let direct = target(
            &device,
            64,
            format,
            MTLStorageMode::Private,
            MTLTextureUsage::Unknown,
        );
        let copied = target(
            &device,
            64,
            format,
            MTLStorageMode::Managed,
            MTLTextureUsage::Unknown,
        );
        assert_eq!(copied.storageMode(), MTLStorageMode::Managed);
        let cmd = queue.commandBuffer().expect("command buffer");
        cmd.setLabel(Some(&objc2_foundation::NSString::from_str(
            "mtld3d-test-upscale-pair",
        )));
        fill(&cmd, &src, red);
        assert!(super::encode(&cmd, &device, &cache, &src, &direct, mode));
        assert_eq!(
            output_count(&cache),
            0,
            "direct output allocates no intermediate"
        );
        assert!(super::can_scale(&device, &cache, &src, &copied, mode));
        assert_eq!(output_count(&cache), 1);
        assert!(super::encode(&cmd, &device, &cache, &src, &copied, mode));
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
        let actual = pixels(&queue, &copied);
        assert_eq!(actual, pixels(&queue, &direct));
        if format == MTLPixelFormat::RGBA16Float {
            for pixel in actual.as_chunks::<8>().0 {
                assert!(
                    u16::from_le_bytes([pixel[0], pixel[1]]) > 0x3c00,
                    "HDR red must exceed 1.0"
                );
            }
        } else {
            assert!(
                actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|pixel| pixel[2] > 128 && pixel[3] == 255)
            );
        }
        super::retire(&cache);
    }
}

#[test]
fn unknown_usage_is_permissive_and_explicit_usage_must_cover_requirements() {
    let required = MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    assert!(super::usage_supports(MTLTextureUsage::Unknown, required));
    assert!(super::usage_supports(
        required | MTLTextureUsage::RenderTarget,
        required
    ));
    assert!(!super::usage_supports(
        MTLTextureUsage::ShaderRead,
        required
    ));
}

/// Failed preparation leaves no output, and a failed copy never reports presentation success.
#[test]
fn allocation_scaler_and_copy_failures_remain_recoverable() {
    let Some(device) = gpu() else { return };
    let queue = device.newCommandQueue().expect("queue");
    let src = target(
        &device,
        32,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Private,
        MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
    );
    let dst = target(
        &device,
        64,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Managed,
        MTLTextureUsage::Unknown,
    );
    let mode = MTLFXSpatialScalerColorProcessingMode::Perceptual;
    let key = super::scaler_key(&src, &dst, mode).expect("key");
    let mut scalers = ScalerCache {
        scalers: FxHashMap::default(),
        tick: 0,
        evicted: Vec::new(),
    };
    assert!(super::scaler_in_with(&mut scalers, &device, key, |_, _| None).is_none());
    assert!(scalers.scalers.is_empty());
    let cache = UpscaleCache::new();
    let mut slot = super::build_scaler(&device, &key).expect("scaler");
    assert!(
        slot.prepare_with(&device, &src, &dst, |_, _, _| None)
            .is_none()
    );
    assert!(slot.output.is_null());
    assert!(slot.prepare(&device, &src, &dst).is_some());
    slot.release();
    let cmd = queue.commandBuffer().expect("command buffer");
    fill(&cmd, &src, 0.75);
    assert!(!super::encode_with(
        &cmd,
        &device,
        &cache,
        &src,
        &dst,
        mode,
        |_, _, _| false
    ));
    // A later attempt can still encode and cover the full destination.
    assert!(super::encode(&cmd, &device, &cache, &src, &dst, mode));
    cmd.commit();
    cmd.waitUntilCompleted();
    assert!(
        pixels(&queue, &dst)
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[2] > 128 && p[3] == 255)
    );
    super::retire(&cache);
}

/// Resizing two live devices keeps each output private to its device and bounded.
#[test]
fn queued_resizes_bound_outputs_and_preserve_two_devices_pixels() {
    use objc2_metal::MTLCommandBufferStatus;
    let Some(device) = gpu() else { return };
    let queues = [
        device.newCommandQueue().expect("first queue"),
        device.newCommandQueue().expect("second queue"),
    ];
    let caches = [UpscaleCache::new(), UpscaleCache::new()];
    let mut submitted = Vec::new();
    for step in 0..(MAX_CACHED_SCALERS + 4) {
        for (index, queue) in queues.iter().enumerate() {
            let cache = &caches[index];
            let src = target(
                &device,
                16 + step,
                MTLPixelFormat::BGRA8Unorm,
                MTLStorageMode::Private,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
            );
            let dst = target(
                &device,
                32 + step * 2,
                MTLPixelFormat::BGRA8Unorm,
                MTLStorageMode::Managed,
                MTLTextureUsage::Unknown,
            );
            let cmd = queue.commandBuffer().expect("command buffer");
            cmd.setLabel(Some(&objc2_foundation::NSString::from_str(
                "mtld3d-test-upscale-resize",
            )));
            fill(&cmd, &src, if index == 0 { 0.25 } else { 0.75 });
            assert!(super::encode(
                &cmd,
                &device,
                cache,
                &src,
                &dst,
                MTLFXSpatialScalerColorProcessingMode::Perceptual
            ));
            cmd.commit();
            submitted.push((index, cmd, dst));
            assert!(output_count(cache) <= MAX_CACHED_SCALERS);
        }
    }
    // No per-frame wait: older encoded resources must survive cache eviction.
    for (index, cmd, dst) in submitted {
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
        let actual = pixels(&queues[index], &dst);
        assert!(
            actual
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| (p[2] > 128) == (index == 1) && p[3] == 255)
        );
    }
    let [first, second] = &caches;
    assert_eq!(output_count(first), MAX_CACHED_SCALERS);
    assert_eq!(output_count(second), MAX_CACHED_SCALERS);
    super::retire(first);
    assert_eq!(output_count(first), 0);
    assert_eq!(output_count(second), MAX_CACHED_SCALERS);
    super::retire(second);
    // A retired cache must build a fresh entry for its next request.
    let src = target(
        &device,
        32,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Private,
        MTLTextureUsage::ShaderRead,
    );
    let dst = target(
        &device,
        64,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Managed,
        MTLTextureUsage::Unknown,
    );
    assert!(super::can_scale(
        &device,
        first,
        &src,
        &dst,
        MTLFXSpatialScalerColorProcessingMode::Perceptual
    ));
    assert_eq!(output_count(first), 1);
    super::retire(first);
}

/// A preflight eviction must retire even when no subsequent upscale is encoded.
#[test]
fn fallback_submission_drains_preflight_evictions() {
    let Some(device) = gpu() else { return };
    let queue = device.newCommandQueue().expect("queue");
    let cache = UpscaleCache::new();
    for step in 0..=MAX_CACHED_SCALERS {
        let src = target(
            &device,
            16 + step,
            MTLPixelFormat::BGRA8Unorm,
            MTLStorageMode::Private,
            MTLTextureUsage::ShaderRead,
        );
        let dst = target(
            &device,
            32 + step * 2,
            MTLPixelFormat::BGRA8Unorm,
            MTLStorageMode::Managed,
            MTLTextureUsage::Unknown,
        );
        assert!(super::can_scale(
            &device,
            &cache,
            &src,
            &dst,
            MTLFXSpatialScalerColorProcessingMode::Perceptual
        ));
    }
    assert_eq!(pending_release_count(&cache), 1);
    let cmd = queue.commandBuffer().expect("fallback command buffer");
    cmd.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-test-upscale-fallback",
    )));
    super::retire_evicted(&cmd, &cache);
    assert_eq!(pending_release_count(&cache), 0);
    cmd.commit();
    cmd.waitUntilCompleted();
    super::retire(&cache);
    assert_eq!(output_count(&cache), 0);
}

/// Explicitly incompatible output usage requires an intermediate even on Private storage.
#[test]
fn usage_requirements_select_an_intermediate_and_reject_invalid_input() {
    let Some(device) = gpu() else { return };
    let cache = UpscaleCache::new();
    let src = target(
        &device,
        32,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Private,
        MTLTextureUsage::ShaderRead,
    );
    let dst = target(
        &device,
        64,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Private,
        MTLTextureUsage::ShaderRead,
    );
    let mode = MTLFXSpatialScalerColorProcessingMode::Perceptual;
    assert!(super::can_scale(&device, &cache, &src, &dst, mode));
    assert_eq!(
        output_count(&cache),
        1,
        "a shader-read-only output cannot serve the scaler's writes"
    );
    let invalid = target(
        &device,
        32,
        MTLPixelFormat::BGRA8Unorm,
        MTLStorageMode::Private,
        MTLTextureUsage::RenderTarget,
    );
    assert!(!super::can_scale(&device, &cache, &invalid, &dst, mode));
    super::retire(&cache);
}
