//! Decode selection and the create-time render-target predicate.
//!
//! The predicate has to stay a superset of the per-upload decode: every
//! upload that resolves a decode must land on a texture the create path
//! already gave `RenderTarget` usage, or the render pass would fail
//! validation.

use mtld3d_shared::mtl::PixelFormat;
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_DXT1, D3DFMT_R5G6B5,
    D3DFMT_R8G8B8, D3DFMT_X1R5G5B5, D3DFMT_X8R8G8B8,
};

use super::{
    UploadDecode, is_expanded_upload, is_expansion, needs_render_target, snapshot_pass_base,
    upload_decode, upload_pass_decode,
};

#[test]
fn packed16_formats_stay_on_the_blit_path() {
    for (format, native) in [
        (D3DFMT_R5G6B5, PixelFormat::B5G6R5Unorm),
        (D3DFMT_A1R5G5B5, PixelFormat::Bgr5A1Unorm),
        (D3DFMT_X1R5G5B5, PixelFormat::Bgr5A1Unorm),
        (D3DFMT_A4R4G4B4, PixelFormat::Abgr4Unorm),
    ] {
        assert_eq!(
            upload_decode(format, native),
            None,
            "{format:#x} on its native packed texture stays a blit"
        );
        assert!(!is_expanded_upload(format, native));
    }
}

/// 24-bit `R8G8B8` expands, and only against a BGRA8 texture.
///
/// No device has a Metal counterpart for a three-byte colour format, so the
/// decode is not conditional on anything. `A8B8G8R8` shares the byte count
/// but has its own native Metal format, so it keeps the blit.
#[test]
fn the_24_bit_format_always_expands() {
    assert_eq!(
        upload_decode(D3DFMT_R8G8B8, PixelFormat::Bgra8Unorm),
        Some(UploadDecode::R8G8B8)
    );
    assert!(is_expansion(UploadDecode::R8G8B8));
    assert!(is_expanded_upload(D3DFMT_R8G8B8, PixelFormat::Bgra8Unorm));
    // The expansion has no blit form, so the attachment is needed whatever
    // the mip pitch is.
    assert!(needs_render_target(
        D3DFMT_R8G8B8,
        PixelFormat::Bgra8Unorm,
        1024,
        1,
        16
    ));
    assert_eq!(
        upload_decode(D3DFMT_A8B8G8R8, PixelFormat::Rgba8Unorm),
        None,
        "the reversed-channel 32-bit format has a native Metal counterpart"
    );
    assert!(!is_expanded_upload(
        D3DFMT_A8B8G8R8,
        PixelFormat::Rgba8Unorm
    ));
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
    // A 256-byte floor pulls in every mip below 64 texels wide: four levels
    // reach the 32-wide mip (128 bytes a row), three stop at 64.
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
    assert_eq!(UploadDecode::CopyBgra8.wire(), 3);
    assert_eq!(UploadDecode::R8G8B8.wire(), 5);
    assert_eq!(UploadDecode::CopyBgra8.bytes_per_texel(), 4);
    assert_eq!(UploadDecode::R8G8B8.bytes_per_texel(), 3);
}

#[test]
fn the_pass_takes_every_expansion_and_a_copy_only_under_the_alignment() {
    for pitch in [4, 16, 256, 4096] {
        assert_eq!(
            upload_pass_decode(D3DFMT_R8G8B8, PixelFormat::Bgra8Unorm, pitch, 16),
            Some(UploadDecode::R8G8B8),
            "an expansion at pitch {pitch}"
        );
    }
    assert_eq!(
        upload_pass_decode(D3DFMT_A8R8G8B8, PixelFormat::Bgra8Unorm, 8, 16),
        Some(UploadDecode::CopyBgra8)
    );
    assert_eq!(
        upload_pass_decode(D3DFMT_A8R8G8B8, PixelFormat::Bgra8Unorm, 16, 16),
        None
    );
    assert_eq!(
        upload_pass_decode(D3DFMT_A8R8G8B8, PixelFormat::Bgra8Unorm, 128, 256),
        Some(UploadDecode::CopyBgra8)
    );
    assert_eq!(
        upload_pass_decode(D3DFMT_DXT1, PixelFormat::Bc1Rgba, 8, 256),
        None
    );
}

/// The shader's address of every texel of a snapshotted box, in its wrapping `uint` math.
fn shader_address(base: u32, texel: (u32, u32), pitch: u32, bpp: u32) -> u32 {
    base.wrapping_add(texel.1.wrapping_mul(pitch))
        .wrapping_add(texel.0.wrapping_mul(bpp))
}

#[test]
fn the_snapshot_base_wraps_back_to_the_box_for_every_texel() {
    // A box at (5, 7) of a 24-bit level, its snapshot 48 bytes a row at 16.
    let (origin, pitch, bpp, offset) = ((5, 7), 48, 3, 16);
    let base = snapshot_pass_base(offset, origin, pitch, bpp);
    assert!(base > offset, "the base wraps below zero");
    for y in 0..4 {
        for x in 0..10 {
            assert_eq!(
                shader_address(base, (origin.0 + x, origin.1 + y), pitch, bpp),
                offset + y * pitch + x * bpp,
                "texel ({x}, {y}) of the box"
            );
        }
    }
    // A box far into the level lands below its offset without wrapping.
    let base = snapshot_pass_base(196_608, (1, 1), 64, 2);
    assert_eq!(base, 196_608 - 66);
    assert_eq!(shader_address(base, (1, 1), 64, 2), 196_608);
}
