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
//! own scaling would give. It writes the drawable directly.
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
//! Scalers are cached per (input size, output size, format, colour mode).
//! Unlike the pipelines in `blit.rs` / `clear_quad.rs` / `present.rs`, they are
//! **not** leaked for the process: a resize walks through a new key per size
//! the window rests at, and each one holds ~16 MiB of intermediates. The cache
//! is bounded ([`MAX_CACHED_SCALERS`]) and evicts least-recently-used, with the
//! release deferred to a command buffer that outlives the evicted scaler.

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
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode,
    MTLFXSpatialScalerDescriptor,
};
use rustc_hash::FxHashMap;

use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// Cache key: a scaler is bound to its exact geometry, formats and colour mode.
///
/// `MTLFXSpatialScaler` fixes all of these at creation, so a change in any of
/// them needs a new instance rather than a mutation.
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

/// Scaler geometries kept alive at once.
///
/// A scaler is not a cheap thing to hold: measured on an M-series GPU, one at
/// `1920x1200 → 2560x1600` costs **~16 MiB** of device memory for its
/// intermediates. Steady-state play needs three at most (the present pair, the
/// readback resolve, and the HDR float scratch), so this never evicts during a
/// game. What it bounds is a *window being resized*, which walks through a new
/// geometry for every size the user rests at: twelve of them in forty seconds
/// of dragging, measured, which unbounded is ~190 MiB that never comes back.
const MAX_CACHED_SCALERS: usize = 8;

/// Scaler cache, `None` once the device is known unsupported.
///
/// The outer `Option` latches the `supportsDevice` answer so an unsupported
/// GPU pays one query instead of one per frame.
static CACHE: OnceLock<Option<Mutex<ScalerCache>>> = OnceLock::new();

/// The live scalers, plus what it takes to bound them.
struct ScalerCache {
    /// One entry per geometry currently served.
    scalers: FxHashMap<ScalerKey, ScalerEntry>,
    /// Monotonic lookup counter that orders [`ScalerEntry::last_used`].
    tick: u64,
    /// Evicted scalers, awaiting a command buffer to outlive them.
    ///
    /// Eviction cannot release: a scaler may still be referenced by a command
    /// buffer the GPU has not finished. They wait here until [`encode`] has a
    /// command buffer to hang the release off.
    evicted: Vec<ScalerSlot>,
}

/// One cached scaler and its recency.
struct ScalerEntry {
    slot: ScalerSlot,
    /// [`ScalerCache::tick`] at the most recent lookup.
    last_used: u64,
}

/// An owned `MTLFXSpatialScaler`, kept as a raw pointer so the map is `Send`.
#[derive(Clone, Copy)]
struct ScalerSlot(*mut ProtocolObject<dyn MTLFXSpatialScaler>);

// SAFETY: the slot is only ever dereferenced back into a borrowed
// `&ProtocolObject` under the cache mutex, and `MTLFXSpatialScaler` is a Metal
// object whose methods Apple documents as callable from any thread. The
// pointer itself came from `Retained::into_raw` and is only turned back into a
// `Retained` once, in the completion handler that releases it.
unsafe impl Send for ScalerSlot {}

/// Encode a `MetalFX` spatial upscale of `src` into `dst`.
///
/// Returns `false` when `MetalFX` cannot serve this pair: an unsupported GPU
/// or a scaler Metal declined to build. The caller then falls to the present
/// shader's stretch, which serves any pair; never to the 1:1 blit, which
/// would leave part of the drawable unwritten.
pub fn encode(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> bool {
    let Some(slot) = scaler_for(device, src, dst, mode) else {
        return false;
    };
    // SAFETY: `slot.0` came from `build_scaler` and is released only from a
    // completion handler on a command buffer committed after this one, so the
    // pointee outlives this borrow and the work encoded from it.
    let scaler = unsafe { &*slot.0 };

    // SAFETY: objc2 typed binding; `src` is a live texture carrying the
    // `ShaderRead` usage the scaler requires.
    unsafe { scaler.setColorTexture(Some(src)) };
    // SAFETY: objc2 typed binding; `dst` is the drawable's texture (or an
    // owned private target), both valid scaler outputs.
    unsafe { scaler.setOutputTexture(Some(dst)) };
    // SAFETY: objc2 typed binding; encoding opens no render pass of its own,
    // and the caller guarantees no encoder is currently open on `cmd_buf`.
    unsafe { scaler.encodeToCommandBuffer(cmd_buf) };
    release_evicted_when_retired(cmd_buf);
    true
}

/// Release evicted scalers once `cmd_buf` retires.
///
/// A scaler cannot be released at eviction: the GPU may still be running work
/// encoded from it. Metal executes a queue's command buffers in commit order
/// and mtld3d has one queue, so this buffer completing means every buffer that
/// could reference an already-evicted scaler has completed too.
///
/// The block owns the slots through a mutex so that a handler Metal somehow
/// ran twice would find the list empty rather than over-release.
fn release_evicted_when_retired(cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>) {
    let Some(cache) = CACHE.get().and_then(Option::as_ref) else {
        return;
    };
    let Ok(mut cache) = cache.lock() else {
        return;
    };
    if cache.evicted.is_empty() {
        return;
    }
    let evicted = Mutex::new(core::mem::take(&mut cache.evicted));
    drop(cache);

    let handler = RcBlock::new(
        move |_cb: core::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let Ok(mut evicted) = evicted.lock() else {
                return;
            };
            for slot in evicted.drain(..) {
                // SAFETY: the pointer came from `Retained::into_raw` in
                // `build_scaler`, was removed from the cache before landing
                // here, and is turned back into a `Retained` exactly once —
                // `drain` cannot yield it twice.
                drop(unsafe { Retained::from_raw(slot.0) });
            }
        },
    );
    // SAFETY: objc2 typed binding; Metal retains the block on
    // `addCompletedHandler`, so the local may drop when this returns.
    unsafe { cmd_buf.addCompletedHandler(RcBlock::as_ptr(&handler)) };
}

/// Whether [`encode`] would serve this pair, without encoding anything.
///
/// The HDR present path has to tone-map into a scratch texture *before* the
/// upscale can run, and a scaler that declines after that point would strand
/// the tone-mapped frame: the fallback tone-maps the back buffer again,
/// straight to the drawable. Asking first keeps that fallback free.
///
/// Builds and caches the scaler on the way, so the [`encode`] that follows a
/// `true` answer is a hash lookup.
pub fn can_scale(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> bool {
    scaler_for(device, src, dst, mode).is_some()
}

/// Look up, or build and cache, the scaler for this pair.
fn scaler_for(
    device: &ProtocolObject<dyn MTLDevice>,
    src: &ProtocolObject<dyn MTLTexture>,
    dst: &ProtocolObject<dyn MTLTexture>,
    mode: MTLFXSpatialScalerColorProcessingMode,
) -> Option<ScalerSlot> {
    let (in_w, in_h) = (src.width(), src.height());
    let (out_w, out_h) = (dst.width(), dst.height());
    // Defensive: the scaler only enlarges, and `render.scale` is capped at
    // 1.0 precisely so this cannot happen. Declining beats asking Metal to
    // build a scaler it will refuse.
    if in_w > out_w || in_h > out_h {
        return None;
    }

    let cache = CACHE.get_or_init(|| init_cache(device)).as_ref()?;
    let key = ScalerKey {
        input_width: truncate(in_w),
        input_height: truncate(in_h),
        output_width: truncate(out_w),
        output_height: truncate(out_h),
        color_format: src.pixelFormat().0,
        output_format: dst.pixelFormat().0,
        mode,
    };

    let mut cache = cache.lock().ok()?;
    cache.tick += 1;
    let tick = cache.tick;
    if let Some(entry) = cache.scalers.get_mut(&key) {
        entry.last_used = tick;
        return Some(entry.slot);
    }

    // A miss builds, which is expensive enough that the settle gate in
    // `command.rs` keeps transient geometry from ever reaching here.
    let slot = build_scaler(device, &key)?;
    if cache.scalers.len() >= MAX_CACHED_SCALERS {
        evict_least_recently_used(&mut cache);
    }
    cache.scalers.insert(
        key,
        ScalerEntry {
            slot,
            last_used: tick,
        },
    );
    drop(cache);
    Some(slot)
}

/// Move the least recently used scaler to the eviction list.
///
/// Least-recently-used rather than oldest-built: the geometry present is
/// currently running is refreshed on every lookup, so it is never the victim.
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
            "present: more than {MAX_CACHED_SCALERS} MetalFX geometries in use, \
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
    CACHE.get_or_init(|| init_cache(device)).is_some()
}

/// One-shot `supportsDevice` probe, latched for the process.
fn init_cache(device: &ProtocolObject<dyn MTLDevice>) -> Option<Mutex<ScalerCache>> {
    // SAFETY: objc2 typed binding; a class method taking a live device.
    let supported = unsafe { MTLFXSpatialScalerDescriptor::supportsDevice(device) };
    if !supported {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "MetalFX spatial upscaling is unavailable on this GPU — render.scale \
             will be held at 1.0 so present stays a 1:1 copy"
        );
        return None;
    }
    Some(Mutex::new(ScalerCache {
        scalers: FxHashMap::default(),
        tick: 0,
        evicted: Vec::new(),
    }))
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
            "MetalFX declined a spatial scaler for {}x{} → {}x{} — this frame presents unscaled",
            key.input_width, key.input_height, key.output_width, key.output_height,
        );
        return None;
    };
    log::info!(
        target: LOG_TARGET,
        "present: MetalFX spatial upscale {}x{} → {}x{}",
        key.input_width, key.input_height, key.output_width, key.output_height,
    );
    // Leaked for process lifetime, same as the present/blit pipelines: the
    // Metal device and queue outlive every frame and are themselves leaked.
    Some(ScalerSlot(Retained::into_raw(scaler)))
}

/// Cache key for a scratch target: one texture per size and format.
///
/// Two callers share the cache — [`resolve_for_readback`] wants a `BGRA8Unorm`
/// target at the reported back-buffer size, the HDR present path wants an
/// `Rgba16Float` one at render size — so the format is what tells their
/// entries apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ScratchKey {
    width: u32,
    height: u32,
    format: PixelFormat,
}

/// Process-lifetime cache of scratch targets, by raw texture handle.
///
/// Stores the wire handle rather than a `Retained` so the map is trivially
/// `Send`; each use re-borrows through `IntoRetained`, which bumps the refcount
/// and leaves the cache's own retain live.
static SCRATCH: OnceLock<Mutex<FxHashMap<ScratchKey, u64>>> = OnceLock::new();

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
    width: u32,
    height: u32,
    format: PixelFormat,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let key = ScratchKey {
        width,
        height,
        format,
    };
    let cache = SCRATCH.get_or_init(|| Mutex::new(FxHashMap::default()));
    let handle = {
        let mut scratch = cache.lock().ok()?;
        match scratch.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let built = super::texture::create_upscale_target(device, width, height, format)?;
                *entry.insert(built.raw())
            }
        }
    };
    // SAFETY: the handle came from `create_upscale_target`, which adopted the
    // texture's canonical retain; the cache holds it for process lifetime.
    unsafe { MetalHandle::<MTLTextureKind>::new(handle) }.into_retained()
}

/// Resolve `src` to `out_w` x `out_h` for a CPU readback, returning the resolved texture.
///
/// `GetRenderTargetData`, a back-buffer `LockRect` and `GetDC` all owe the game
/// pixels at the resolution D3D9 reports, but under `render.scale` the back
/// buffer is rasterized smaller. Running the *same* upscale the display path
/// runs, into a scratch texture of the reported size, is what makes a readback
/// agree with what is on screen; resampling on the CPU instead would be both
/// slower and visibly worse.
///
/// Encodes onto `cmd_buf` ahead of the caller's blit encoder, so the resolve
/// and the readback are one command buffer and one wait. Returns `None` when
/// `MetalFX` cannot serve the pair, leaving the caller to read `src` directly.
///
/// In SDR the scaler cache key is the same one present uses, so a readback at
/// the frame's own scale reuses an already-built scaler. In HDR present scales
/// a float texture instead, so the two keys diverge and this path builds its
/// own — which is right, because the back buffer it reads is still `BGRA8Unorm`
/// and still wants `Perceptual`.
pub fn resolve_for_readback(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    device: &ProtocolObject<dyn MTLDevice>,
    src: &ProtocolObject<dyn MTLTexture>,
    out_w: u32,
    out_h: u32,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    // The resolve target has to match the source's format, and the only source
    // that ever needs resolving is the back buffer, which is pinned to
    // `BGRA8Unorm`. Gating on that declines rather than guessing if it ever
    // does vary.
    if src.pixelFormat() != MTLPixelFormat::BGRA8Unorm {
        return None;
    }
    let target = scratch_target(device, out_w, out_h, PixelFormat::Bgra8Unorm);
    let Some(target) = target else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "readback resolve target {out_w}x{out_h} could not be created — readback reads \
             the render-resolution frame instead and will be the wrong size"
        );
        return None;
    };
    // The back buffer is sRGB-encoded, the same input the display path feeds
    // the scaler, so the readback resolve shares its colour mode and its
    // cached scaler.
    if !encode(
        cmd_buf,
        device,
        src,
        &target,
        MTLFXSpatialScalerColorProcessingMode::Perceptual,
    ) {
        return None;
    }
    Some(target)
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
