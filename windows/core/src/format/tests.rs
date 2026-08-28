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
    D3DFMT_D32, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1, D3DFMT_G16R16,
    D3DFMT_G16R16F, D3DFMT_G32R32F, D3DFMT_INTZ, D3DFMT_R5G6B5, D3DFMT_R16F, D3DFMT_R32F,
    D3DFMT_X8R8G8B8, PixelFormat, Swizzle, depth_format_bytes_per_pixel, is_depth_format,
    is_mapped_color_format, map_d3d_depth_format, map_d3d_format, surface_bytes,
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
        // FOURCC sampleable-depth, minus INTZ (it carries a stencil
        // plane, tested with the stencil-bearing family below).
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
    // INTZ belongs here: it is the sampleable twin of D24S8 and carries
    // its stencil plane.
    for fmt in [
        D3DFMT_D15S1,
        D3DFMT_D24S8,
        D3DFMT_D24X4S4,
        D3DFMT_D24FS8,
        D3DFMT_INTZ,
    ] {
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

#[test]
fn device_mapping_expands_the_packed_16_bit_family_only_without_native_support() {
    use mtld3d_types::{D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_R5G6B5};

    use super::map_d3d_format_device;

    // native_packed16 = true: identical to the plain lookup for every format.
    for fmt in [
        D3DFMT_R5G6B5,
        D3DFMT_A1R5G5B5,
        D3DFMT_A4R4G4B4,
        D3DFMT_A8R8G8B8,
        D3DFMT_A16B16G16R16F,
    ] {
        let native = map_d3d_format_device(fmt, true).expect("mapped");
        let plain = map_d3d_format(fmt).expect("mapped");
        assert_eq!(
            native.metal_pixel_format(),
            plain.metal_pixel_format(),
            "format {fmt}"
        );
        assert_eq!(native.swizzle(), plain.swizzle(), "format {fmt}");
        assert_eq!(
            native.bytes_per_pixel(),
            plain.bytes_per_pixel(),
            "format {fmt}"
        );
    }

    // native_packed16 = false: the three packed members back Bgra8Unorm while
    // keeping their 2-byte SOURCE layout (Lock pitch and staging sizing).
    let r5g6b5 = map_d3d_format_device(D3DFMT_R5G6B5, false).expect("mapped");
    assert_eq!(r5g6b5.metal_pixel_format(), PixelFormat::Bgra8Unorm);
    assert_eq!(r5g6b5.bytes_per_pixel(), 2);
    assert_eq!(r5g6b5.block_bytes(), 2);
    // No swizzle on any of the three: the upload pass writes D3D channel
    // order and an opaque alpha, and a swizzled view cannot be an attachment.
    assert_eq!(r5g6b5.swizzle(), None, "upload pass forces alpha opaque");
    assert!(!r5g6b5.has_alpha());

    let a1r5g5b5 = map_d3d_format_device(D3DFMT_A1R5G5B5, false).expect("mapped");
    assert_eq!(a1r5g5b5.metal_pixel_format(), PixelFormat::Bgra8Unorm);
    assert_eq!(a1r5g5b5.bytes_per_pixel(), 2);
    assert_eq!(a1r5g5b5.swizzle(), None);
    assert!(a1r5g5b5.has_alpha());

    let a4r4g4b4 = map_d3d_format_device(D3DFMT_A4R4G4B4, false).expect("mapped");
    assert_eq!(a4r4g4b4.metal_pixel_format(), PixelFormat::Bgra8Unorm);
    assert_eq!(a4r4g4b4.bytes_per_pixel(), 2);
    assert_eq!(
        a4r4g4b4.swizzle(),
        None,
        "upload pass writes D3D channel order"
    );
    assert!(a4r4g4b4.has_alpha());

    // Non-packed formats are untouched by the flag.
    let bgra = map_d3d_format_device(D3DFMT_A8R8G8B8, false).expect("mapped");
    assert_eq!(bgra.metal_pixel_format(), PixelFormat::Bgra8Unorm);
    assert_eq!(bgra.bytes_per_pixel(), 4);
}

#[test]
fn only_the_single_precision_floats_depend_on_device_filtering() {
    use mtld3d_types::{
        D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
        D3DUSAGE_RENDERTARGET,
    };

    use super::supports_usage_query;

    // Every usage shape that reaches the classifier, with and without the
    // D3DUSAGE_QUERY_FILTER bit a title adds to it.
    const PLAIN: [u32; 4] = [
        0,
        D3DUSAGE_RENDERTARGET,
        D3DUSAGE_DEPTHSTENCIL,
        D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
    ];
    const HALF_FLOATS: [u32; 3] = [D3DFMT_R16F, D3DFMT_G16R16F, D3DFMT_A16B16G16R16F];
    const SINGLE_FLOATS: [u32; 3] = [D3DFMT_R32F, D3DFMT_G32R32F, D3DFMT_A32B32G32R32F];

    for float32_filtering in [true, false] {
        for fmt in HALF_FLOATS.into_iter().chain(SINGLE_FLOATS) {
            for usage in PLAIN {
                // Without the filter bit the answer is the same on either
                // device: renderability and blending are device-independent.
                assert!(
                    supports_usage_query(fmt, usage, float32_filtering),
                    "format {fmt} usage {usage:#x} filtering {float32_filtering}",
                );
            }
        }
        // Half floats filter on every GPU family.
        for fmt in HALF_FLOATS {
            for usage in PLAIN {
                assert!(
                    supports_usage_query(fmt, usage | D3DUSAGE_QUERY_FILTER, float32_filtering),
                    "format {fmt} usage {usage:#x} filtering {float32_filtering}",
                );
            }
        }
        // The single-precision three follow the device answer, in every
        // usage shape the filter bit can be combined with.
        for fmt in SINGLE_FLOATS {
            for usage in PLAIN {
                assert_eq!(
                    supports_usage_query(fmt, usage | D3DUSAGE_QUERY_FILTER, float32_filtering),
                    float32_filtering,
                    "format {fmt} usage {usage:#x} filtering {float32_filtering}",
                );
            }
        }
        // A format outside the float family is never gated.
        assert!(supports_usage_query(
            D3DFMT_A8R8G8B8,
            D3DUSAGE_QUERY_FILTER,
            float32_filtering
        ));
        assert!(supports_usage_query(
            D3DFMT_A16B16G16R16,
            D3DUSAGE_QUERY_FILTER,
            float32_filtering
        ));
    }
}

/// Every depth format `map_d3d_depth_format` maps has a byte size here.
///
/// The two tables are consulted in sequence by `surface_bytes`, so a depth
/// format present in one and missing from the other is charged zero bytes
/// against the `GetAvailableTextureMem` budget.
#[test]
fn depth_size_table_covers_the_depth_mapping() {
    for fmt in [
        D3DFMT_D16,
        D3DFMT_D16_LOCKABLE,
        D3DFMT_D15S1,
        D3DFMT_D24X8,
        D3DFMT_D24S8,
        D3DFMT_D24X4S4,
        D3DFMT_D24FS8,
        D3DFMT_D32,
        D3DFMT_D32F_LOCKABLE,
        D3DFMT_DF16,
        D3DFMT_DF24,
        D3DFMT_INTZ,
    ] {
        assert!(
            map_d3d_depth_format(fmt).is_some(),
            "{fmt:#x} is a depth format"
        );
        assert!(
            depth_format_bytes_per_pixel(fmt).is_some(),
            "{fmt:#x} has no depth byte size"
        );
    }
    assert_eq!(depth_format_bytes_per_pixel(D3DFMT_A8R8G8B8), None);
}

#[test]
fn surface_bytes_charges_colour_and_depth_surfaces() {
    // A standalone 2048x2048 A8R8G8B8 render target is 16 MiB.
    assert_eq!(surface_bytes(2048, 2048, D3DFMT_A8R8G8B8), 16 * 1024 * 1024);
    assert_eq!(surface_bytes(2048, 2048, D3DFMT_X8R8G8B8), 16 * 1024 * 1024);
    // Source bytes per pixel, not the Metal backing: R5G6B5 is 2 bytes even
    // where it is expanded to BGRA8, and D24S8 is 4 even though the Metal
    // texture behind it is `Depth32Float_Stencil8`.
    assert_eq!(surface_bytes(256, 128, D3DFMT_R5G6B5), 256 * 128 * 2);
    assert_eq!(surface_bytes(1024, 1024, D3DFMT_D24S8), 4 * 1024 * 1024);
    assert_eq!(surface_bytes(1024, 1024, D3DFMT_D16), 2 * 1024 * 1024);
    // Block-compressed formats are charged by block: DXT1 is 8 bytes per
    // 4x4 block, so half a byte per texel.
    assert_eq!(surface_bytes(64, 64, D3DFMT_DXT1), 64 * 64 / 2);
    // A format with neither mapping is charged nothing rather than panicking.
    assert_eq!(surface_bytes(64, 64, 0), 0);
}
