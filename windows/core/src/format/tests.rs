//! Unit tests for the D3D9 depth-format mapping.
//!
//! Apple Silicon has no 24-bit depth, so the whole D24 family and the FOURCC
//! sampleable-depth formats collapse onto `Depth32Float`, and the
//! stencil-bearing ones onto `Depth32FloatStencil8`. These pin that collapse,
//! keep `is_depth_format` in step with the mapping it wraps, and check that
//! color and unknown formats stay unmapped.

use super::{
    D3DFMT_A8R8G8B8, D3DFMT_D15S1, D3DFMT_D16, D3DFMT_D16_LOCKABLE, D3DFMT_D24FS8, D3DFMT_D24S8,
    D3DFMT_D24X4S4, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16, D3DFMT_DF24,
    D3DFMT_INTZ, PixelFormat, is_depth_format, map_d3d_depth_format,
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
