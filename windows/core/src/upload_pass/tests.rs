//! Decode selection and the create-time render-target predicate.
//!
//! The predicate has to stay a superset of the per-upload decode: every
//! upload that resolves a decode must land on a texture the create path
//! already gave `RenderTarget` usage, or the render pass would fail
//! validation.

use mtld3d_shared::mtl::PixelFormat;
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_DXT1, D3DFMT_R5G6B5, D3DFMT_X8R8G8B8,
};

use super::{UploadDecode, is_expanded_upload, is_expansion, needs_render_target, upload_decode};

#[test]
fn packed16_formats_decode_only_against_a_bgra8_texture() {
    for (format, expected) in [
        (D3DFMT_R5G6B5, UploadDecode::R5G6B5),
        (D3DFMT_A1R5G5B5, UploadDecode::A1R5G5B5),
        (D3DFMT_A4R4G4B4, UploadDecode::A4R4G4B4),
    ] {
        assert_eq!(
            upload_decode(format, PixelFormat::Bgra8Unorm),
            Some(expected),
            "{format:#x} expands on a Bgra8Unorm texture"
        );
        assert!(is_expansion(expected), "{format:#x} is an expansion");
        assert!(is_expanded_upload(format, PixelFormat::Bgra8Unorm));
        assert_eq!(
            upload_decode(format, PixelFormat::B5G6R5Unorm),
            None,
            "{format:#x} on a native packed texture stays a blit"
        );
    }
}

#[test]
fn only_the_swizzle_free_bgra8_source_takes_the_copy_decode() {
    assert_eq!(
        upload_decode(D3DFMT_A8R8G8B8, PixelFormat::Bgra8Unorm),
        Some(UploadDecode::CopyBgra8)
    );
    assert!(!is_expansion(UploadDecode::CopyBgra8));
    assert!(!is_expanded_upload(
        D3DFMT_A8R8G8B8,
        PixelFormat::Bgra8Unorm
    ));
    // X8R8G8B8 shares the Metal format but carries the alpha-forcing sampler
    // swizzle, which a render-target handle cannot keep.
    assert_eq!(
        upload_decode(D3DFMT_X8R8G8B8, PixelFormat::Bgra8Unorm),
        None
    );
    assert_eq!(upload_decode(D3DFMT_DXT1, PixelFormat::Bc1Rgba), None);
}

#[test]
fn expansion_needs_the_attachment_at_every_size() {
    // No mip is small enough to need padding, and the expansion still has to
    // render: there is no blit that widens 2 bpp to 4 bpp.
    assert!(needs_render_target(
        D3DFMT_R5G6B5,
        PixelFormat::Bgra8Unorm,
        1024,
        1,
        16
    ));
}

#[test]
fn copy_decode_needs_the_attachment_only_below_the_alignment() {
    // 256 wide, 9 levels: the 1x1 mip is 4 bytes, under the 16-byte floor.
    assert!(needs_render_target(
        D3DFMT_A8R8G8B8,
        PixelFormat::Bgra8Unorm,
        256,
        9,
        16
    ));
    // Same texture, single level: 1024 bytes a row, always blittable.
    assert!(!needs_render_target(
        D3DFMT_A8R8G8B8,
        PixelFormat::Bgra8Unorm,
        256,
        1,
        16
    ));
    // Mac2's 256-byte floor pulls in every mip below 64 texels wide: four
    // levels reach the 32-wide mip (128 bytes a row), three stop at 64.
    assert!(needs_render_target(
        D3DFMT_A8R8G8B8,
        PixelFormat::Bgra8Unorm,
        256,
        4,
        256
    ));
    assert!(!needs_render_target(
        D3DFMT_A8R8G8B8,
        PixelFormat::Bgra8Unorm,
        256,
        3,
        256
    ));
    // A format with no decode never gets the usage bit.
    assert!(!needs_render_target(
        D3DFMT_X8R8G8B8,
        PixelFormat::Bgra8Unorm,
        256,
        9,
        16
    ));
}

#[test]
fn wire_values_match_the_shader_cases() {
    assert_eq!(UploadDecode::R5G6B5.wire(), 0);
    assert_eq!(UploadDecode::A1R5G5B5.wire(), 1);
    assert_eq!(UploadDecode::A4R4G4B4.wire(), 2);
    assert_eq!(UploadDecode::CopyBgra8.wire(), 3);
    assert_eq!(UploadDecode::R5G6B5.bytes_per_texel(), 2);
    assert_eq!(UploadDecode::CopyBgra8.bytes_per_texel(), 4);
}
