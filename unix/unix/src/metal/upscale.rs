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
//! One case it does not serve: **a GPU without `MetalFX`**. [`is_supported`]
//! answers that once at layer attach and the PE side then keeps the drawable
//! the same size as the back buffer, so present stays a 1:1 copy.
//!
//! Scalers are cached per (input size, output size, format, colour mode) and
//! leaked for process lifetime, the same posture as `blit.rs` / `clear_quad.rs`
//! / `hdr_present.rs`. A resize changes the key, so a `Reset` builds a new one
//! and the old entry is simply never looked up again.

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Mutex, OnceLock},
};

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

/// Process-lifetime scaler cache, `None` once the device is known unsupported.
///
/// The outer `Option` latches the `supportsDevice` answer so an unsupported
/// GPU pays one query instead of one per frame.
static CACHE: OnceLock<Option<Mutex<HashMap<ScalerKey, ScalerSlot>>>> = OnceLock::new();

/// A leaked `MTLFXSpatialScaler`, kept as a raw pointer so the map is `Send`.
#[derive(Clone, Copy)]
struct ScalerSlot(*mut ProtocolObject<dyn MTLFXSpatialScaler>);

// SAFETY: the slot is only ever dereferenced back into a borrowed
// `&ProtocolObject` under the cache mutex, and `MTLFXSpatialScaler` is a Metal
// object whose methods Apple documents as callable from any thread. The
// pointer itself is a leaked `Retained`, so it stays valid for the process.
unsafe impl Send for ScalerSlot {}

/// Encode a `MetalFX` spatial upscale of `src` into `dst`.
///
/// Returns `false` when `MetalFX` cannot serve this pair: an unsupported GPU
/// or a scaler Metal declined to build. The caller then leaves the frame to
/// the 1:1 present blit, which is only correct because the PE side keeps the
/// two sizes equal whenever [`is_supported`] said no.
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
    // SAFETY: `slot.0` is a leaked `Retained` produced by `build_scaler` and
    // never released, so the pointee outlives this borrow.
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
    true
}

/// Whether [`encode`] would serve this pair, without encoding anything.
///
/// The HDR present path has to tone-map into a scratch texture *before* the
/// upscale can run, and a scaler that declines after that point would strand
/// the tone-mapped frame: the 1:1 fallback blit copies the source extent, so it
/// cannot stand in for a resample. Asking first keeps the fallback free.
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

    let mut scalers = cache.lock().ok()?;
    match scalers.entry(key) {
        Entry::Occupied(entry) => Some(*entry.get()),
        Entry::Vacant(entry) => Some(*entry.insert(build_scaler(device, &key)?)),
    }
}

/// Whether this GPU can run a `MetalFX` spatial upscale.
///
/// Answered once at layer attach so the PE side knows whether it may size the
/// drawable independently of the back buffer. With no `MetalFX` there is no way
/// to resize a frame at present time, so the two sizes have to stay equal and
/// Core Animation scales the layer instead.
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
fn init_cache(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Mutex<HashMap<ScalerKey, ScalerSlot>>> {
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
    Some(Mutex::new(HashMap::new()))
}

/// Build and leak one scaler for `key`, or `None` if Metal declines.
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
static SCRATCH: OnceLock<Mutex<HashMap<ScratchKey, u64>>> = OnceLock::new();

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
    let cache = SCRATCH.get_or_init(|| Mutex::new(HashMap::new()));
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
