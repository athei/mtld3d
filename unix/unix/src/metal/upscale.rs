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
//! The scaler runs on the game's `BGRA8Unorm` back buffer in `Perceptual`
//! colour-processing mode, which is what an sRGB-encoded 8-bit surface wants.
//!
//! Two cases it does not serve, both handled by the caller:
//!
//! - **A GPU without `MetalFX`.** [`is_supported`] answers this once at layer
//!   attach and the PE side then keeps the drawable the same size as the back
//!   buffer, so present stays a 1:1 copy.
//! - **HDR.** The drawable is float and the inverse tone map has to run after
//!   the upscale, so the scaler cannot write it. The present shader's own
//!   bilinear sample covers the resample instead.
//!
//! Scalers are cached per (input size, output size, format) and leaked for
//! process lifetime, the same posture as `blit.rs` / `clear_quad.rs` /
//! `hdr_present.rs`. A resize changes the key, so a `Reset` builds a new one
//! and the old entry is simply never looked up again.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use mtld3d_shared::{MetalHandle, mtl_handle::MTLDeviceKind};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode,
    MTLFXSpatialScalerDescriptor,
};

use crate::{LOG_TARGET, metal::handle::IntoRetained};

/// Cache key: a scaler is bound to its exact geometry and formats.
///
/// `MTLFXSpatialScaler` fixes input/output size and pixel format at creation,
/// so a change in any of them needs a new instance rather than a mutation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ScalerKey {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    color_format: usize,
    output_format: usize,
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
) -> bool {
    let (in_w, in_h) = (src.width(), src.height());
    let (out_w, out_h) = (dst.width(), dst.height());
    // Defensive: the scaler only enlarges, and `render.scale` is capped at
    // 1.0 precisely so this cannot happen. Declining beats asking Metal to
    // build a scaler it will refuse.
    if in_w > out_w || in_h > out_h {
        return false;
    }

    let Some(cache) = CACHE.get_or_init(|| init_cache(device)).as_ref() else {
        return false;
    };
    let key = ScalerKey {
        input_width: truncate(in_w),
        input_height: truncate(in_h),
        output_width: truncate(out_w),
        output_height: truncate(out_h),
        color_format: src.pixelFormat().0,
        output_format: dst.pixelFormat().0,
    };

    let Ok(mut scalers) = cache.lock() else {
        return false;
    };
    let slot = if let Some(&slot) = scalers.get(&key) {
        slot
    } else {
        let Some(built) = build_scaler(device, &key) else {
            return false;
        };
        scalers.insert(key, built);
        built
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

/// Whether this GPU can run a `MetalFX` spatial upscale.
///
/// Answered once at layer attach so the PE side knows whether it may size the
/// drawable independently of the back buffer. With no `MetalFX` there is no way
/// to resize a frame at present time, so the two sizes have to stay equal and
/// Core Animation scales the layer instead.
pub fn is_supported(device_handle: MetalHandle<MTLDeviceKind>) -> bool {
    device_handle
        .into_retained()
        .is_some_and(|device| CACHE.get_or_init(|| init_cache(&device)).is_some())
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
    // `Perceptual` is the default and the right one for our sRGB-encoded
    // BGRA8 back buffer; set it explicitly so the choice is visible.
    // SAFETY: scalar property write on an owned descriptor.
    unsafe { desc.setColorProcessingMode(MTLFXSpatialScalerColorProcessingMode::Perceptual) };
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

/// Narrow a Metal texture dimension to the `u32` the cache key stores.
///
/// Metal caps texture dimensions well below `u32::MAX` (16384 on every device
/// we target), so the value always fits; the saturating form keeps the
/// conversion total without an `expect` on a path that runs per present.
fn truncate(dim: usize) -> u32 {
    u32::try_from(dim).unwrap_or(u32::MAX)
}
