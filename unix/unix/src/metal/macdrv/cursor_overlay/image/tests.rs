use objc2_core_graphics::{CGDataProvider, CGImageAlphaInfo, kCGColorSpaceExtendedLinearSRGB};
use objc2_metal::MTLPixelFormat;

use super::{CGColorSpace, CGImage, from_pixels, transparent};

#[test]
fn cursor_images_preserve_premultiplied_layout_and_own_their_bytes() {
    // SAFETY: immutable CoreGraphics color-space name.
    let name = unsafe { kCGColorSpaceExtendedLinearSRGB };
    let space = CGColorSpace::with_name(Some(name)).unwrap();
    let mut bytes = vec![0, 0x40, 0, 0x3c, 0, 0, 0, 0x38];
    let image = from_pixels(1, 1, MTLPixelFormat::RGBA16Float, Some(&space), &bytes).unwrap();
    bytes.fill(0);
    assert_eq!(CGImage::bits_per_component(Some(&image)), 16);
    assert_eq!(
        CGImage::alpha_info(Some(&image)),
        CGImageAlphaInfo::PremultipliedLast
    );
    let provider = CGImage::data_provider(Some(&image)).unwrap();
    let retained = CGDataProvider::data(Some(&provider)).unwrap();
    assert_eq!(retained.to_vec(), &[0, 0x40, 0, 0x3c, 0, 0, 0, 0x38]);
    let clear = transparent().unwrap();
    assert_eq!(
        CGImage::alpha_info(Some(&clear)),
        CGImageAlphaInfo::PremultipliedFirst
    );
    assert_eq!(CGImage::bits_per_pixel(Some(&clear)), 32);
}

#[test]
fn cursor_image_rejects_incomplete_rows_and_unsupported_formats() {
    let space = CGColorSpace::new_device_rgb().unwrap();
    assert!(from_pixels(2, 1, MTLPixelFormat::BGRA8Unorm, Some(&space), &[0; 4]).is_none());
    assert!(from_pixels(1, 1, MTLPixelFormat::R8Unorm, Some(&space), &[0]).is_none());
}
