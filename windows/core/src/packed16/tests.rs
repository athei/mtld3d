use mtld3d_shared::mtl::PixelFormat;
use mtld3d_types::{D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_R5G6B5};

use super::{Packed16Kind, expand_rows, expansion_kind};

fn expand_one(kind: Packed16Kind, texel: u16) -> [u8; 4] {
    let src = texel.to_le_bytes();
    let mut dst = [0u8; 4];
    expand_rows(kind, &src, 2, &mut dst, 4, 1, 1);
    dst
}

#[test]
fn classifier_expands_only_packed_sources_backed_by_bgra8() {
    assert!(expansion_kind(D3DFMT_R5G6B5, PixelFormat::Bgra8Unorm).is_some());
    assert!(expansion_kind(D3DFMT_A1R5G5B5, PixelFormat::Bgra8Unorm).is_some());
    assert!(expansion_kind(D3DFMT_A4R4G4B4, PixelFormat::Bgra8Unorm).is_some());
    // Native path: the GPU format is the packed one, never Bgra8Unorm.
    assert!(expansion_kind(D3DFMT_R5G6B5, PixelFormat::B5G6R5Unorm).is_none());
    assert!(expansion_kind(D3DFMT_A1R5G5B5, PixelFormat::Bgr5A1Unorm).is_none());
    assert!(expansion_kind(D3DFMT_A4R4G4B4, PixelFormat::Abgr4Unorm).is_none());
    // A 32-bit source over Bgra8Unorm is the ordinary non-expansion upload.
    assert!(expansion_kind(D3DFMT_A8R8G8B8, PixelFormat::Bgra8Unorm).is_none());
}

#[test]
fn r5g6b5_endpoints_and_primaries_are_exact() {
    // BGRA byte order in the expanded output.
    assert_eq!(expand_one(Packed16Kind::R5G6B5, 0x0000), [0, 0, 0, 0xFF]);
    assert_eq!(
        expand_one(Packed16Kind::R5G6B5, 0xFFFF),
        [0xFF, 0xFF, 0xFF, 0xFF]
    );
    assert_eq!(expand_one(Packed16Kind::R5G6B5, 0xF800), [0, 0, 0xFF, 0xFF]); // red
    assert_eq!(expand_one(Packed16Kind::R5G6B5, 0x07E0), [0, 0xFF, 0, 0xFF]); // green
    assert_eq!(expand_one(Packed16Kind::R5G6B5, 0x001F), [0xFF, 0, 0, 0xFF]); // blue
}

#[test]
fn a1r5g5b5_alpha_bit_maps_to_0_or_255() {
    assert_eq!(
        expand_one(Packed16Kind::A1R5G5B5, 0x7FFF),
        [0xFF, 0xFF, 0xFF, 0]
    );
    assert_eq!(
        expand_one(Packed16Kind::A1R5G5B5, 0xFFFF),
        [0xFF, 0xFF, 0xFF, 0xFF]
    );
    assert_eq!(
        expand_one(Packed16Kind::A1R5G5B5, 0xFC00),
        [0, 0, 0xFF, 0xFF]
    ); // opaque red
    assert_eq!(
        expand_one(Packed16Kind::A1R5G5B5, 0x83E0),
        [0, 0xFF, 0, 0xFF]
    ); // opaque green
    assert_eq!(
        expand_one(Packed16Kind::A1R5G5B5, 0x801F),
        [0xFF, 0, 0, 0xFF]
    ); // opaque blue
}

#[test]
fn a4r4g4b4_nibbles_widen_by_replication() {
    assert_eq!(expand_one(Packed16Kind::A4R4G4B4, 0x0000), [0, 0, 0, 0]);
    assert_eq!(
        expand_one(Packed16Kind::A4R4G4B4, 0xFFFF),
        [0xFF, 0xFF, 0xFF, 0xFF]
    );
    assert_eq!(
        expand_one(Packed16Kind::A4R4G4B4, 0xFF00),
        [0, 0, 0xFF, 0xFF]
    ); // opaque red
    assert_eq!(
        expand_one(Packed16Kind::A4R4G4B4, 0xF0F0),
        [0, 0xFF, 0, 0xFF]
    ); // opaque green
    assert_eq!(
        expand_one(Packed16Kind::A4R4G4B4, 0xF00F),
        [0xFF, 0, 0, 0xFF]
    ); // opaque blue
    // 4-bit x → x * 17: mid-nibble 0x8 widens to 0x88.
    assert_eq!(
        expand_one(Packed16Kind::A4R4G4B4, 0x8888),
        [0x88, 0x88, 0x88, 0x88]
    );
}

#[test]
fn five_and_six_bit_widening_replicates_high_bits() {
    // 5-bit 0x10 → (0x10 << 3) | (0x10 >> 2) = 0x84.
    assert_eq!(expand_one(Packed16Kind::A1R5G5B5, 0x0010), [0x84, 0, 0, 0]);
    // 6-bit 0x20 → (0x20 << 2) | (0x20 >> 4) = 0x82 in the green lane.
    assert_eq!(
        expand_one(Packed16Kind::R5G6B5, 0x20 << 5),
        [0, 0x82, 0, 0xFF]
    );
}

#[test]
fn pitches_larger_than_tight_rows_are_honoured() {
    // 2x2 sub-rect out of a 4-texel-wide source (pitch 8), expanded into a
    // destination padded to 32 bytes per row. Padding bytes stay untouched.
    let mut src = [0u8; 16];
    src[0..2].copy_from_slice(&0xF800u16.to_le_bytes()); // (0,0) red
    src[2..4].copy_from_slice(&0x07E0u16.to_le_bytes()); // (1,0) green
    src[8..10].copy_from_slice(&0x001Fu16.to_le_bytes()); // (0,1) blue
    src[10..12].copy_from_slice(&0xFFFFu16.to_le_bytes()); // (1,1) white

    let mut dst = [0xABu8; 64];
    expand_rows(Packed16Kind::R5G6B5, &src, 8, &mut dst, 32, 2, 2);

    assert_eq!(&dst[0..4], &[0, 0, 0xFF, 0xFF]);
    assert_eq!(&dst[4..8], &[0, 0xFF, 0, 0xFF]);
    assert_eq!(&dst[32..36], &[0xFF, 0, 0, 0xFF]);
    assert_eq!(&dst[36..40], &[0xFF, 0xFF, 0xFF, 0xFF]);
    // Row padding and the second row's tail keep the fill pattern.
    assert!(dst[8..32].iter().all(|&b| b == 0xAB));
    assert!(dst[40..].iter().all(|&b| b == 0xAB));
}
