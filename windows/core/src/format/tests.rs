//! Unit tests for the D3D9 format mappings.
//!
//! Apple Silicon has no 24-bit depth, so the whole D24 family and the FOURCC
//! sampleable-depth formats collapse onto `Depth32Float`, and the
//! stencil-bearing ones onto `Depth32FloatStencil8`. These pin that collapse,
//! keep `is_depth_format` in step with the mapping it wraps, and check that
//! color and unknown formats stay unmapped. The colour side pins the
//! wide-channel family (16-bit unorm and the floats): Metal format, pitch,
//! and the missing-channel swizzle, plus `is_mapped_color_format` tracking
//! the lookup table.

use super::{
    D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16, D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_D15S1,
    D3DFMT_D16, D3DFMT_D16_LOCKABLE, D3DFMT_D24FS8, D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24X8,
    D3DFMT_D32, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_G16R16, D3DFMT_G16R16F,
    D3DFMT_G32R32F, D3DFMT_INTZ, D3DFMT_R16F, D3DFMT_R32F, PixelFormat, Swizzle, is_depth_format,
    is_mapped_color_format, map_d3d_depth_format, map_d3d_format,
};

#[test]
fn depth_only_formats_promote_to_depth32float() {
    // Apple Silicon has no Depth24Unorm — D24X8, D32, D16, and the
    // lockable variants all share Depth32Float.
    for fmt in [
        D3DFMT_D16_LOCKABLE,
        D3DFMT_D32,
        D3DFMT_D24X8,
        D3DFMT_D16,
        D3DFMT_D32F_LOCKABLE,
        // FOURCC sampleable-depth — engines (incl. WoW CSM) gate the
        // shadow-map path on at least one of these being available.
        D3DFMT_INTZ,
        D3DFMT_DF24,
        D3DFMT_DF16,
    ] {
        assert_eq!(
            map_d3d_depth_format(fmt),
            Some(PixelFormat::Depth32Float),
            "format {fmt} should map to Depth32Float"
        );
    }
}

#[test]
fn stencil_bearing_formats_promote_to_depth32float_stencil8() {
    for fmt in [D3DFMT_D15S1, D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24FS8] {
        assert_eq!(
            map_d3d_depth_format(fmt),
            Some(PixelFormat::Depth32FloatStencil8),
            "format {fmt} should map to Depth32FloatStencil8"
        );
    }
}

#[test]
fn non_depth_formats_return_none() {
    assert_eq!(map_d3d_depth_format(D3DFMT_A8R8G8B8), None);
    assert_eq!(map_d3d_depth_format(0), None);
    assert_eq!(map_d3d_depth_format(0xFFFF_FFFF), None);
}

#[test]
fn is_depth_format_matches_map() {
    assert!(is_depth_format(D3DFMT_D24X8));
    assert!(is_depth_format(D3DFMT_D24S8));
    assert!(is_depth_format(D3DFMT_INTZ));
    assert!(is_depth_format(D3DFMT_DF24));
    assert!(is_depth_format(D3DFMT_DF16));
    assert!(!is_depth_format(D3DFMT_A8R8G8B8));
}

#[test]
fn the_wide_channel_family_maps_to_its_metal_counterpart() {
    // D3D9 names these formats most-significant channel first, so the
    // stored order is the reverse of the name and matches Metal's
    // R-then-G-then-B-then-A layout byte for byte. Channels a format does
    // not store sample as 1.0, so only the four-channel members go through
    // unswizzled. The 16-bit unorm pair follows the same rule as the
    // floats.
    let one = Swizzle::One;
    let red_only = Some([Swizzle::Red, one, one, one]);
    let red_green = Some([Swizzle::Red, Swizzle::Green, one, one]);
    for (fmt, expected, bytes, swizzle) in [
        (D3DFMT_G16R16, PixelFormat::Rg16Unorm, 4, red_green),
        (D3DFMT_A16B16G16R16, PixelFormat::Rgba16Unorm, 8, None),
        (D3DFMT_R16F, PixelFormat::R16Float, 2, red_only),
        (D3DFMT_G16R16F, PixelFormat::Rg16Float, 4, red_green),
        (D3DFMT_A16B16G16R16F, PixelFormat::Rgba16Float, 8, None),
        (D3DFMT_R32F, PixelFormat::R32Float, 4, red_only),
        (D3DFMT_G32R32F, PixelFormat::Rg32Float, 8, red_green),
        (D3DFMT_A32B32G32R32F, PixelFormat::Rgba32Float, 16, None),
    ] {
        let mapping = map_d3d_format(fmt).expect("float format is mapped");
        assert_eq!(mapping.metal_pixel_format(), expected, "format {fmt}");
        assert_eq!(mapping.bytes_per_pixel(), bytes, "format {fmt}");
        assert_eq!(mapping.swizzle(), swizzle, "format {fmt}");
        assert!(!mapping.is_compressed(), "format {fmt}");
    }
    // Only the four-channel members carry alpha; the others read A = 1.
    assert!(
        map_d3d_format(D3DFMT_A16B16G16R16F)
            .expect("mapped")
            .has_alpha()
    );
    assert!(!map_d3d_format(D3DFMT_G16R16F).expect("mapped").has_alpha());
    assert!(
        map_d3d_format(D3DFMT_A16B16G16R16)
            .expect("mapped")
            .has_alpha()
    );
    assert!(!map_d3d_format(D3DFMT_G16R16).expect("mapped").has_alpha());
}

#[test]
fn is_mapped_color_format_tracks_the_lookup() {
    // The `CheckDeviceFormat` texture answer is derived from this, so it
    // must stay exactly the set the create paths accept.
    for fmt in [D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16F, D3DFMT_G32R32F] {
        assert!(is_mapped_color_format(fmt), "format {fmt}");
        assert!(map_d3d_format(fmt).is_some(), "format {fmt}");
    }
    for fmt in [0, 0xFFFF_FFFF, D3DFMT_D24S8] {
        assert!(!is_mapped_color_format(fmt), "format {fmt}");
        assert!(map_d3d_format(fmt).is_none(), "format {fmt}");
    }
}
