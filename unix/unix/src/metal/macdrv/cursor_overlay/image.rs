//! Immutable cursor images made from completed offscreen Metal output.

use core::ptr::NonNull;

use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_foundation::{CFData, CFRetained};
use objc2_core_graphics::{
    CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGImageComponentInfo,
};
use objc2_metal::{MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLTexture};
use objc2_quartz_core::CALayer;

use super::LOG_TARGET;

#[cfg(test)]
mod tests;

/// A retained transparent surface, used without another GPU submission.
pub fn transparent() -> Option<CFRetained<CGImage>> {
    let Some(space) = CGColorSpace::new_device_rgb() else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: RGB color space allocation failed");
        return None;
    };
    from_pixels(1, 1, MTLPixelFormat::BGRA8Unorm, Some(&space), &[0; 4])
}

/// Copy a completed, CPU-visible texture into an independently owned image.
///
/// The caller observes its command buffer's completion first. Managed textures
/// must have a synchronizeResource blit in that command buffer.
pub fn readback(
    texture: &ProtocolObject<dyn MTLTexture>,
    space: Option<&CGColorSpace>,
) -> Option<CFRetained<CGImage>> {
    let (bytes_per_pixel, _, _) = layout(texture.pixelFormat())?;
    let width = texture.width();
    let height = texture.height();
    let row = width.checked_mul(bytes_per_pixel)?;
    let mut pixels = vec![0; row.checked_mul(height)?];
    // SAFETY: the completed texture is CPU-visible; the slice spans every row
    // of this exact region and remains exclusively borrowed for the copy.
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            NonNull::new(pixels.as_mut_ptr())
                .expect("nonempty cursor pixels")
                .cast(),
            row,
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            },
            0,
        );
    }
    from_pixels(width, height, texture.pixelFormat(), space, &pixels)
}

/// Set a `CGImage`, which `CALayer` retains as its documented contents type.
pub fn set_contents(layer: &CALayer, image: &CGImage) {
    // SAFETY: CALayer.contents accepts a CGImageRef. This borrowed object view
    // is used only by that setter; no NSObject method is sent to the image.
    let object = unsafe { NonNull::from(image).cast::<AnyObject>().as_ref() };
    // SAFETY: the argument is the live CGImage above; CALayer retains it.
    unsafe { layer.setContents(Some(object)) };
}

fn layout(format: MTLPixelFormat) -> Option<(usize, usize, CGBitmapInfo)> {
    match format {
        MTLPixelFormat::BGRA8Unorm => Some((
            4,
            8,
            CGBitmapInfo(
                CGImageAlphaInfo::PremultipliedFirst.0 | CGImageByteOrderInfo::Order32Little.0,
            ),
        )),
        MTLPixelFormat::RGBA16Float => Some((
            8,
            16,
            CGBitmapInfo(
                CGImageAlphaInfo::PremultipliedLast.0
                    | CGImageByteOrderInfo::Order16Little.0
                    | CGImageComponentInfo::Float.0,
            ),
        )),
        _ => {
            mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: unsupported image pixel format {format:?}");
            None
        }
    }
}

fn from_pixels(
    width: usize,
    height: usize,
    format: MTLPixelFormat,
    space: Option<&CGColorSpace>,
    pixels: &[u8],
) -> Option<CFRetained<CGImage>> {
    let (bytes_per_pixel, bits, info) = layout(format)?;
    let row = width.checked_mul(bytes_per_pixel)?;
    if width == 0 || height == 0 || row.checked_mul(height) != Some(pixels.len()) {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: invalid image byte extent");
        return None;
    }
    let data = CFData::from_bytes(pixels);
    let Some(provider) = CGDataProvider::with_cf_data(Some(&data)) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: image data provider allocation failed");
        return None;
    };
    // SAFETY: validated packed rows; the provider owns a copy of all bytes,
    // the format describes their layout, and a null decode uses normal values.
    let image = unsafe {
        CGImage::new(
            width,
            height,
            bits,
            bytes_per_pixel * 8,
            row,
            space,
            info,
            Some(&provider),
            core::ptr::null(),
            false,
            CGColorRenderingIntent::RenderingIntentDefault,
        )
    };
    if image.is_none() {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET, "cursor: CGImage creation failed for {format:?}");
    }
    image
}
