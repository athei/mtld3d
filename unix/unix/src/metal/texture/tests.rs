use mtld3d_shared::mtl::PixelFormat;
use objc2_metal::MTLPixelFormat;

use super::{is_resolvable_color_format, mtl_pixel_format, wire_pixel_format};

/// Every wire pixel format, in declaration order.
const ALL: [PixelFormat; 29] = [
    PixelFormat::A8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::R16Unorm,
    PixelFormat::R16Float,
    PixelFormat::Rg8Unorm,
    PixelFormat::Rg8Snorm,
    PixelFormat::B5G6R5Unorm,
    PixelFormat::Abgr4Unorm,
    PixelFormat::Bgr5A1Unorm,
    PixelFormat::Rg16Unorm,
    PixelFormat::R32Float,
    PixelFormat::Rg16Float,
    PixelFormat::Rgba8Unorm,
    PixelFormat::Rgba8UnormSrgb,
    PixelFormat::Bgra8Unorm,
    PixelFormat::Bgra8UnormSrgb,
    PixelFormat::Rg32Float,
    PixelFormat::Rgba16Unorm,
    PixelFormat::Rgba16Float,
    PixelFormat::Rgba32Float,
    PixelFormat::Bc1Rgba,
    PixelFormat::Bc1RgbaSrgb,
    PixelFormat::Bc2Rgba,
    PixelFormat::Bc2RgbaSrgb,
    PixelFormat::Bc3Rgba,
    PixelFormat::Bc3RgbaSrgb,
    PixelFormat::Bc4RUnorm,
    PixelFormat::Depth32Float,
    PixelFormat::Depth32FloatStencil8,
];

#[test]
fn wire_pixel_format_inverts_mtl_pixel_format_for_every_format() {
    for format in ALL {
        assert_eq!(
            wire_pixel_format(mtl_pixel_format(format)),
            Some(format),
            "{format:?} round-trips through its Metal format"
        );
    }
}

#[test]
fn wire_pixel_format_declines_a_format_mtld3d_never_creates() {
    assert_eq!(wire_pixel_format(MTLPixelFormat::RGB10A2Unorm), None);
    assert_eq!(wire_pixel_format(MTLPixelFormat::Invalid), None);
}

#[test]
fn only_uncompressed_colour_formats_are_resolvable() {
    let resolvable: Vec<PixelFormat> = ALL
        .into_iter()
        .filter(|format| is_resolvable_color_format(*format))
        .collect();
    assert_eq!(
        resolvable.len(),
        20,
        "20 uncompressed colour formats: {resolvable:?}"
    );
    assert!(!is_resolvable_color_format(PixelFormat::Bc1Rgba));
    assert!(!is_resolvable_color_format(PixelFormat::Bc4RUnorm));
    assert!(!is_resolvable_color_format(PixelFormat::Depth32Float));
    assert!(is_resolvable_color_format(PixelFormat::Rgba32Float));
    assert!(is_resolvable_color_format(PixelFormat::A8Unorm));
}
