//! `MetalFX` spatial upscale of the back buffer on the way to the drawable.
//!
//! The back buffer and the drawable are no longer the same size: the drawable
//! covers the window, while the back buffer is the grid we rasterize on. When
//! the two differ, present has to resample, and `MTLBlitCommandEncoder` only
//! does 1:1 copies.
//!
//! `MTLFXSpatialScaler` is the good path for that resample. It is an
//! edge-aware upscaler rather than a stretched bilinear sample, so a
//! half-resolution frame comes back materially sharper than the compositor's
//! own scaling would give. Compatible drawables are written directly; other
//! destinations receive a copy from a Private output.
//!
//! In SDR the scaler runs on the game's `BGRA8Unorm` back buffer in
//! `Perceptual` colour-processing mode, which is what an sRGB-encoded 8-bit
//! surface wants. In HDR the present shader tone-maps first, at render
//! resolution, and the scaler runs on that `RGBA16Float` result in `HDR` mode —
//! the mode built for values beyond `[0, 1]`.
//!
//! Two cases it does not serve. **A drawable smaller than the back buffer**:
//! the spatial scaler only enlarges, which is why `render.scale` is capped at
//! `1.0`. And **a GPU without `MetalFX`**: [`is_supported`] answers that once
//! at layer attach, and the PE side then holds `render.scale` at `1.0` so the
//! scaler is not the only thing standing between a scaled frame and the
//! screen. Both fall to the present shader's filtered stretch
//! (`PresentPipelines::copy`), which covers any ratio; this module is the
//! quality path, not the correctness one.
//!
//! Scalers and scratch targets live in an [`UpscaleCache`] on the device's
//! own record, keyed by (input size, output size, format, colour mode). They
//! are per device because the input and output textures are properties set on
//! the scaler and read by the encode that follows them, so one object shared
//! by two devices presenting at one window size would let each write the
//! other's frame through it.
//!
//! Unlike the pipelines in `blit.rs` / `clear_quad.rs` / `present.rs`, they are
//! **not** leaked for the process: a resize walks through a new key per size
//! the window rests at, and each one holds ~16 MiB of intermediates. The cache
//! is bounded ([`MAX_CACHED_SCALERS`]) and evicts its least-recently-used
//! entry, with the release deferred to a command buffer of the device's queue;
//! everything a device holds, live and evicted, is released in
//! `DestroyCommandQueue`.

use std::{
    collections::hash_map::Entry,
    sync::{Mutex, OnceLock},
};

use block2::RcBlock;
use mtld3d_shared::{
    MetalHandle,
    mtl::PixelFormat,
    mtl_handle::{MTLDeviceKind, MTLTextureKind},
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLOrigin,
    MTLPixelFormat, MTLResource, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use objc2_metal_fx::{
    MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode,
    MTLFXSpatialScalerDescriptor,
};
use rustc_hash::FxHashMap;

use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// Cache key: a scaler is bound to its geometry, formats and colour mode.
///
/// `MTLFXSpatialScaler` fixes all three at creation, so a change in any of
/// them needs a new instance rather than a mutation. The device is not in the
/// key because the cache is the device's: a scaler is stateful across an
/// encode, its colour and output textures being properties, so two devices
/// sharing one object could each encode with the other's textures set.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ScalerKey {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    color_format: usize,
    output_format: usize,
    mode: MTLFXSpatialScalerColorProcessingMode,
}

/// Scaler geometries one device keeps alive at once.
///
/// A scaler is not a cheap thing to hold: measured on an M-series GPU, one at
/// `1920x1200 → 2560x1600` costs **~16 MiB** of device memory for its
/// intermediates. Steady-state play needs two at most (the present pair and
/// the HDR float scratch), so this never evicts during a game. What it bounds
/// is a *window being resized*, which walks through a new geometry for every
/// size the user rests at: twelve of them in forty seconds of dragging,
/// measured, which unbounded is ~190 MiB that never comes back. The bound is
/// per device because the thing it sizes is: a second device drags a window of
/// its own, and its geometries must not evict the game's.
const MAX_CACHED_SCALERS: usize = 8;

/// Whether the pinned `MTLDevice` supports `MetalFX` at all.
///
/// A machine fact, latched once so an unsupported GPU pays one query instead
/// of one per frame: every D3D device is handed the same `MTLDevice`.
static SUPPORTED: OnceLock<bool> = OnceLock::new();

/// One device's `MetalFX` caches, owned by its record.
///
/// Both halves are per device because what they hold is: a scaler is stateful
/// across an encode, and a scratch target is only ordered against the command
/// buffers of the queue that resolves into it.
pub struct UpscaleCache {
    scalers: Mutex<ScalerCache>,
    /// Scratch targets by geometry and format, as raw texture handles.
    ///
    /// Stores the wire handle rather than a `Retained` so the map is trivially
    /// `Send`; each use re-borrows through `IntoRetained`, which bumps the
    /// refcount and leaves the cache's own retain live.
    scratch: Mutex<FxHashMap<ScratchKey, u64>>,
}

impl Default for UpscaleCache {
    fn default() -> Self {
        Self::new()
    }
}

impl UpscaleCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scalers: Mutex::new(ScalerCache {
                scalers: FxHashMap::default(),
                tick: 0,
                evicted: Vec::new(),
            }),
            scratch: Mutex::new(FxHashMap::default()),
        }
    }
}

/// The live scalers, plus what it takes to bound them.
struct ScalerCache {
    /// One entry per geometry currently served.
    scalers: FxHashMap<ScalerKey, ScalerEntry>,
    /// Monotonic lookup counter that orders [`ScalerEntry::last_used`].
    tick: u64,
    /// Evicted scalers awaiting a command buffer to outlive them.
    ///
    /// Eviction cannot release: a scaler may still be referenced by a command
    /// buffer the GPU has not finished. They wait here until [`encode`] has a
    /// command buffer *of this device's queue* to hang the release off, which
    /// is the only ordering Metal offers.
    evicted: Vec<ScalerSlot>,
}

/// One cached scaler and its recency.
struct ScalerEntry {
    slot: ScalerSlot,
    /// [`ScalerCache::tick`] at the most recent lookup.
    last_used: u64,
}

/// A scaler and its optional Private output, owned until queue retirement.
///
/// Raw pointers keep the cache Send. Neither retain is copied: eviction moves
/// the slot to a completion handler, and queue shutdown takes remaining slots.
struct ScalerSlot {
    scaler: *mut ProtocolObject<dyn MTLFXSpatialScaler>,
    output: *mut ProtocolObject<dyn MTLTexture>,
}

// SAFETY: the pointers own canonical retains, accessed only under the cache
// mutex. Metal encoding is serialized per queue; retirement moves the slot
// to that queue's completion handler, which releases each retain exactly once.
unsafe impl Send for ScalerSlot {}

impl ScalerSlot {
    /// Release both canonical retains after the owning queue's GPU work ends.
    fn release(self) {
        // SAFETY: this slot was removed from the cache and is consumed once,
        // after GPU completion. The pointer owns the scaler's original retain.
        drop(unsafe { Retained::from_raw(self.scaler) });
        // SAFETY: output is null or owns the intermediate's original retain;
        // no queued work can use it after the caller's retirement boundary.
        drop(unsafe { Retained::from_raw(self.output) });
    }

    /// Prepare a compatible output without encoding any work.
    fn prepare(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        src: &ProtocolObject<dyn MTLTexture>,
        dst: &ProtocolObject<dyn MTLTexture>,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        self.prepare_with(device, src, dst, create_output)
    }

    fn prepare_with(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        src: &ProtocolObject<dyn MTLTexture>,
        dst: &ProtocolObject<dyn MTLTexture>,
        create: impl FnOnce(
            &ProtocolObject<dyn MTLDevice>,
            &ProtocolObject<dyn MTLTexture>,
            MTLTextureUsage,
        ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        // SAFETY: the cache mutex holds this slot's canonical scaler retain.
        let scaler = unsafe { &*self.scaler };
        // SAFETY: typed property read on the live cached scaler.
        let input_usage = unsafe { scaler.colorTextureUsage() };
        // SAFETY: typed property read on the live cached scaler.
        let output_usage = unsafe { scaler.outputTextureUsage() };
        if !usage_supports(src.usage(), input_usage) || src.isFramebufferOnly() {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "upscale: input usage {:?} cannot serve {:?}; present shader stretches instead",
                src.usage(), input_usage);
            return None;
        }
        if dst.isFramebufferOnly() {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "upscale: framebuffer-only destination cannot receive the upscale; present shader stretches instead");
            return None;
        }
        if dst.storageMode() == MTLStorageMode::Private && usage_supports(dst.usage(), output_usage)
        {
            return Some(dst.retain());
        }
        if self.output.is_null() {
            let output = create(device, dst, output_usage)?;
            self.output = Retained::into_raw(output);
        }
        // SAFETY: the cache holds the intermediate's canonical retain, and
        // the returned retain keeps this borrow independent of the cache lock.
        unsafe { Retained::retain(self.output) }
    }
}

/// Unknown usage permits all operations; explicit usage must contain every required bit.
fn usage_supports(actual: MTLTextureUsage, required: MTLTextureUsage) -> bool {
    actual == MTLTextureUsage::Unknown || actual.contains(required)
}

/// Create the scaler's Private output at the destination's exact extent and format.
fn create_output(
    device: &ProtocolObject<dyn MTLDevice>,
    dst: &ProtocolObject<dyn MTLTexture>,
    usage: MTLTextureUsage,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    // SAFETY: typed constructor; dimensions and format come from a live destination.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            dst.pixelFormat(),
            dst.width(),
            dst.height(),
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(usage);
    let Some(texture) = device.newTextureWithDescriptor(&desc) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "upscale: Private output allocation failed for {}x{} {:?} usage {:?}; present shader stretches instead",
            dst.width(), dst.height(), dst.pixelFormat(), usage);
        return None;
    };
    texture.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-upscale-output",
    )));
    log::info!(target: LOG_TARGET,
        "present: MetalFX Private output {}x{} {:?}, destination storage {:?} usage {:?}",
        dst.width(), dst.height(), dst.pixelFormat(), dst.storageMode(), dst.usage());
    Some(texture)
}

/// Copy a completed upscale without another resample or color conversion.
fn copy_output(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
) -> bool {
    let Some(blit) = cmd_buf.blitCommandEncoder() else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "upscale: output copy encoder unavailable; present shader stretches instead");
        return false;
    };
    blit.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-upscale-copy",
    )));
    // SAFETY: prepare created src with dst's exact format and extent; both
    // are non-framebuffer-only textures, retained by this command buffer.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            src, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: dst.width(), height: dst.height(), depth: 1 },
            dst, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.endEncoding();
    true
}

/// Encode a `MetalFX` spatial upscale of `src` into `dst`.
///
/// `cache` is the device's own, and `cmd_buf` a buffer of its queue: that is
/// what orders the release of what this call evicts.
///
/// Returns `false` when `MetalFX` cannot serve this pair: an unsupported GPU
/// or a scaler Metal declined to build. The caller then falls to the present
/// shader's stretch, which serves any pair; never to the 1:1 blit, which
/// would leave part of the drawable unwritten.
pub fn encode(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> bool {
    encode_with(cmd_buf, device, cache, src, dst, mode, copy_output)
}

fn encode_with(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
    copy: impl FnOnce(
        &ProtocolObject<dyn MTLCommandBuffer>,
        &ProtocolObject<dyn MTLTexture>,
        &ProtocolObject<dyn MTLTexture>,
    ) -> bool,
) -> bool {
    let Some(key) = scaler_key(src, dst, mode) else {
        return false;
    };
    if !supported(device) {
        return false;
    }
    // Keep texture binding and encode atomic with respect to other lookups.
    let Ok(mut cache) = cache.scalers.lock() else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "upscale: scaler cache lock poisoned; present shader stretches instead");
        return false;
    };
    let Some(slot) = scaler_in(&mut cache, device, key) else {
        return false;
    };
    let Some(output) = slot.prepare(device, src, dst) else {
        return false;
    };
    // SAFETY: the cache mutex holds the scaler's canonical retain through encode.
    let scaler = unsafe { &*slot.scaler };
    // SAFETY: prepare checked src against the scaler's required usage.
    unsafe { scaler.setColorTexture(Some(src)) };
    // SAFETY: prepare selected Private storage with the required output usage.
    unsafe { scaler.setOutputTexture(Some(&output)) };
    // SAFETY: no encoder is open; both textures match the scaler descriptor.
    unsafe { scaler.encodeToCommandBuffer(cmd_buf) };
    // The command buffer retains encoded resources. Clear the scaler's bindings
    // so an idle cached scaler cannot keep a drawable out of the layer's pool.
    // SAFETY: nullable property write on the live scaler after encoding.
    unsafe { scaler.setColorTexture(None) };
    // SAFETY: nullable property write on the live scaler after encoding.
    unsafe { scaler.setOutputTexture(None) };
    let direct = core::ptr::eq(Retained::as_ptr(&output), core::ptr::from_ref(dst));
    let encoded = direct || copy(cmd_buf, &output, dst);
    log::debug!(target: "mtld3d::unix::present",
        "MetalFX encoded={encoded} direct={direct} {}x{} -> {}x{} {:?}, destination storage {:?} usage {:?}",
        src.width(), src.height(), dst.width(), dst.height(), mode, dst.storageMode(), dst.usage());
    let evicted = take_evicted(&mut cache);
    drop(cache);
    release_when_retired(cmd_buf, evicted);
    encoded
}

/// Schedule pending evictions even when presentation falls back after preflight.
pub fn retire_evicted(cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>, cache: &UpscaleCache) {
    let evicted = {
        let Ok(mut cache) = cache.scalers.lock() else {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "upscale: poisoned cache leaves evictions pending until the device goes away");
            return;
        };
        take_evicted(&mut cache)
    };
    release_when_retired(cmd_buf, evicted);
}

/// Release `evicted` once `cmd_buf` retires.
///
/// A scaler cannot be released at eviction: the GPU may still be running work
/// encoded from it. Metal executes a queue's command buffers in commit order,
/// and every slot here was evicted from the key of the queue `cmd_buf` belongs
/// to, so this buffer completing means every buffer that could reference one
/// of them has completed too. Another queue's evictions wait for a buffer of
/// their own, or for [`retire_scalers`] when that queue goes away.
///
/// The block owns the slots through a mutex so that a handler Metal somehow
/// ran twice would find the list empty rather than over-release.
fn release_when_retired(cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>, evicted: Vec<ScalerSlot>) {
    if evicted.is_empty() {
        return;
    }
    let evicted = Mutex::new(evicted);

    let handler = RcBlock::new(
        move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let Ok(mut evicted) = evicted.lock() else {
                mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                    "upscale: retirement lock poisoned; evicted resources remain allocated");
                return;
            };
            for slot in evicted.drain(..) {
                slot.release();
            }
        },
    );
    // SAFETY: objc2 typed binding; Metal retains the block on
    // `addCompletedHandler`, so the local may drop when this returns.
    unsafe { cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler)) };
}

/// Take the scalers `queue` has evicted and not yet released.
fn take_evicted(cache: &mut ScalerCache) -> Vec<ScalerSlot> {
    core::mem::take(&mut cache.evicted)
}

/// Whether [`encode`] would serve this pair, without encoding anything.
///
/// The HDR present path has to tone-map into a scratch texture *before* the
/// upscale can run, and a scaler that declines after that point would strand
/// the tone-mapped frame: the fallback tone-maps the back buffer again,
/// straight to the drawable. Asking first keeps that fallback free.
///
/// Builds and caches this device's scaler on the way, so the [`encode`] that
/// follows a `true` answer is a hash lookup.
pub fn can_scale(
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &UpscaleCache,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> bool {
    let Some(key) = scaler_key(src, dst, mode) else {
        return false;
    };
    if !supported(device) {
        return false;
    }
    let Ok(mut cache) = cache.scalers.lock() else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "upscale: scaler cache lock poisoned during preflight; present shader stretches instead");
        return false;
    };
    scaler_in(&mut cache, device, key)
        .and_then(|slot| slot.prepare(device, src, dst))
        .is_some()
}

/// The key this pair scales under, or `None` when `MetalFX` cannot serve it.
fn scaler_key(
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> Option<ScalerKey> {
    let (in_w, in_h) = (src.width(), src.height());
    let (out_w, out_h) = (dst.width(), dst.height());
    // Defensive: the scaler only enlarges, and `render.scale` is capped at
    // 1.0 precisely so this cannot happen. Declining beats asking Metal to
    // build a scaler it will refuse.
    if in_w > out_w || in_h > out_h {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "upscale: {in_w}x{in_h} cannot enlarge into {out_w}x{out_h}; present shader stretches instead");
        return None;
    }
    Some(ScalerKey {
        input_width: truncate(in_w),
        input_height: truncate(in_h),
        output_width: truncate(out_w),
        output_height: truncate(out_h),
        color_format: src.pixelFormat().0,
        output_format: dst.pixelFormat().0,
        mode,
    })
}

/// The scaler cache, or `None` once this GPU is known to have no `MetalFX`.
fn supported(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    *SUPPORTED.get_or_init(|| {
        // SAFETY: objc2 typed binding; a class method taking a live device.
        let supported = unsafe { MTLFXSpatialScalerDescriptor::supportsDevice(device) };
        if !supported {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "MetalFX spatial upscaling is unavailable on this GPU, render.scale \
                 will be held at 1.0 so present stays a 1:1 copy"
            );
        }
        supported
    })
}

/// Look up, or build and cache, the scaler for `key`.
fn scaler_in<'a>(
    cache: &'a mut ScalerCache,
    device: &ProtocolObject<dyn MTLDevice>,
    key: ScalerKey,
) -> Option<&'a mut ScalerSlot> {
    scaler_in_with(cache, device, key, build_scaler)
}

fn scaler_in_with<'a>(
    cache: &'a mut ScalerCache,
    device: &ProtocolObject<dyn MTLDevice>,
    key: ScalerKey,
    build: impl FnOnce(&ProtocolObject<dyn MTLDevice>, &ScalerKey) -> Option<ScalerSlot>,
) -> Option<&'a mut ScalerSlot> {
    cache.tick += 1;
    let tick = cache.tick;
    if !cache.scalers.contains_key(&key) {
        // Build before eviction so a declined scaler never displaces a usable one.
        let slot = build(device, &key)?;
        if cache.scalers.len() >= MAX_CACHED_SCALERS {
            evict_least_recently_used(cache);
        }
        cache.scalers.insert(
            key,
            ScalerEntry {
                slot,
                last_used: tick,
            },
        );
    }
    let entry = cache
        .scalers
        .get_mut(&key)
        .expect("scaler was found or inserted above");
    entry.last_used = tick;
    Some(&mut entry.slot)
}

/// Move this device's least recently used scaler to its eviction list.
///
/// Least-recently-used rather than oldest-built: the geometry present is
/// currently running is refreshed on every lookup, so it is never the victim.
/// The cache is the device's own, so another device's scalers are never in
/// reach here.
fn evict_least_recently_used(cache: &mut ScalerCache) {
    let Some(victim) = cache
        .scalers
        .iter()
        .min_by_key(|(_, entry)| entry.last_used)
        .map(|(key, _)| *key)
    else {
        return;
    };
    if let Some(entry) = cache.scalers.remove(&victim) {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "present: more than {MAX_CACHED_SCALERS} MetalFX geometries in use on one device, \
             retiring the least recently used ({}x{} → {}x{})",
            victim.input_width, victim.input_height,
            victim.output_width, victim.output_height,
        );
        cache.evicted.push(entry.slot);
    }
}

/// Whether this GPU can run a `MetalFX` spatial upscale.
///
/// Answered once at layer attach so the PE side knows whether `render.scale`
/// can be honoured. Without `MetalFX` a scaled frame would reach the screen
/// through the present shader's plain bilinear magnify, which is a worse
/// picture than not scaling at all; the scale is held at `1.0` instead.
pub fn is_supported(device_handle: MetalHandle<MTLDeviceKind>) -> bool {
    device_handle
        .into_retained()
        .is_some_and(|device| is_available(&device))
}

/// [`is_supported`] for a device already in hand.
///
/// Present-time callers reach the device through `cmd_buf.device()` and hold no
/// handle. They ask this before allocating anything a declined scaler would
/// orphan.
pub fn is_available(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    supported(device)
}

/// Build one scaler for `key`, or `None` if Metal declines.
fn build_scaler(device: &ProtocolObject<dyn MTLDevice>, key: &ScalerKey) -> Option<ScalerSlot> {
    // Every `objc2-metal-fx` binding is generated `unsafe`, so each property
    // write needs its own block. They are all plain scalar setters on an
    // owned descriptor; the sizes came from live textures and so are within
    // Metal's dimension limits by construction.
    // SAFETY: objc2 typed binding; `new` on a plain NSObject subclass.
    let desc = unsafe { MTLFXSpatialScalerDescriptor::new() };
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setInputWidth(key.input_width as usize) };
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setInputHeight(key.input_height as usize) };
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setOutputWidth(key.output_width as usize) };
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setOutputHeight(key.output_height as usize) };
    // SAFETY: scalar property write; the format came from a live texture.
    unsafe { desc.setColorTextureFormat(MTLPixelFormat(key.color_format)) };
    // SAFETY: scalar property write; the format came from a live texture.
    unsafe { desc.setOutputTextureFormat(MTLPixelFormat(key.output_format)) };
    // `Perceptual` for the sRGB-encoded BGRA8 back buffer, `HDR` for the
    // tone-mapped float scratch the HDR present path feeds in. The caller
    // picks; there is no format-sniffing here.
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setColorProcessingMode(key.mode) };
    // `inputContentWidth`/`inputContentHeight` are left at their defaults:
    // they mark the used sub-rect of the colour texture, and we always feed
    // the whole back buffer.

    // SAFETY: objc2 typed binding; the descriptor is fully populated and the
    // device is live. Documented to return nil rather than throw on failure.
    let scaler = unsafe { desc.newSpatialScalerWithDevice(device) };
    let Some(scaler) = scaler else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "MetalFX declined a spatial scaler for {}x{} → {}x{}, this frame presents unscaled",
            key.input_width, key.input_height, key.output_width, key.output_height,
        );
        return None;
    };
    log::info!(
        target: LOG_TARGET,
        "present: configured MetalFX spatial scaler {}x{} -> {}x{}",
        key.input_width, key.input_height, key.output_width, key.output_height,
    );
    // The cache owns this retain from here: an eviction hands it to a
    // completion handler on a command buffer of the same queue, and
    // `retire_scalers` releases whatever the queue still holds.
    Some(ScalerSlot {
        scaler: Retained::into_raw(scaler),
        output: core::ptr::null_mut(),
    })
}

/// Release everything this device's caches hold, scalers and scratch targets.
///
/// Called from `DestroyCommandQueue` after the device's shutdown fence, so no
/// command buffer of its queue can still encode from a scaler or read a
/// scratch target, and so an eviction of its own is not left waiting for a
/// command buffer that will never be committed.
pub fn retire(cache: &UpscaleCache) {
    let retired = cache.scalers.lock().map_or_else(
        |_| {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "retire: the scaler cache lock is poisoned; the device's scalers stay allocated"
            );
            Vec::new()
        },
        |mut scalers| take_scalers(&mut scalers),
    );
    for slot in retired {
        slot.release();
    }
    let retired: Vec<u64> = cache.scratch.lock().map_or_else(
        |_| {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "retire: the scratch cache lock is poisoned; the device's scratch targets stay \
                 allocated"
            );
            Vec::new()
        },
        |mut scratch| scratch.drain().map(|(_, handle)| handle).collect(),
    );
    for handle in retired {
        super::texture::destroy_texture(handle);
    }
}

/// Take every scaler, live and evicted, out of one device's cache.
///
/// The map half of [`retire`], kept apart so a unit test can pin what a
/// retire takes without a Metal object behind it.
fn take_scalers(cache: &mut ScalerCache) -> Vec<ScalerSlot> {
    let mut retired = take_evicted(cache);
    retired.extend(cache.scalers.drain().map(|(_, entry)| entry.slot));
    retired
}

/// Identity of one scratch target within a device: its geometry and format.
///
/// The device is the cache, not part of the key: Metal orders command buffers
/// within one queue only, so two devices at one size and format sharing a
/// scratch let one device's resolve land between the other's resolve and its
/// blit, and each read the other's frame (#445). A cache per device gives
/// each one a texture of its own.
///
/// Two callers share the cache: the readback resolve wants a `BGRA8Unorm`
/// target at the reported back-buffer size, the HDR present path wants an
/// `Rgba16Float` one at render size, so the format is what tells their
/// entries apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ScratchKey {
    width: u32,
    height: u32,
    format: PixelFormat,
}

/// Get, or create and cache, a `Private` scratch texture of this size and format.
///
/// `Private` is not a preference: `MTLFXSpatialScaler` rejects an output
/// texture in any other storage mode, and only the Metal debug layer reports
/// it. [`super::texture::create_upscale_target`] pins it.
///
/// Returns `None` if Metal declines the texture; the caller decides what a
/// missing scratch means for its path.
pub fn scratch_target(
    device: &ProtocolObject<dyn MTLDevice>,
    cache: &UpscaleCache,
    width: u32,
    height: u32,
    format: PixelFormat,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let key = ScratchKey {
        width,
        height,
        format,
    };
    let handle = {
        let mut scratch = cache.scratch.lock().ok()?;
        match scratch.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let built = super::texture::create_upscale_target(device, width, height, format)?;
                *entry.insert(built.raw())
            }
        }
    };
    // SAFETY: the handle came from `create_upscale_target`, which adopted the
    // texture's canonical retain; the cache holds it until the device retires.
    unsafe { MetalHandle::<MTLTextureKind>::new(handle) }.into_retained()
}

/// Narrow a Metal texture dimension to the `u32` the cache key stores.
///
/// Metal caps texture dimensions well below `u32::MAX` (16384 on every device
/// we target), so the value always fits; the saturating form keeps the
/// conversion total without an `expect` on a path that runs per present.
fn truncate(dim: usize) -> u32 {
    u32::try_from(dim).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests;
