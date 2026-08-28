use mtld3d_types::{D3DFMT_A4R4G4B4, D3DFMT_DXT1, D3DFMT_YUY2};

use super::*;

/// A `ConvertRegion` covering a whole `width` x `height` level pair.
fn whole(width: u32, height: u32, src_pitch: usize, dst_pitch: usize) -> ConvertRegion {
    ConvertRegion {
        src_x: 0,
        src_y: 0,
        dst_x: 0,
        dst_y: 0,
        width,
        height,
        src_pitch,
        dst_pitch,
        src_slice_pitch: src_pitch * height as usize,
        dst_slice_pitch: dst_pitch * height as usize,
        depth: 1,
    }
}

#[test]
fn the_rgb_formats_convert_in_both_directions() {
    for src in [D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFMT_R5G6B5] {
        for dst in [D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFMT_R5G6B5] {
            assert!(can_convert(src, dst), "{src:#x} into {dst:#x}");
        }
    }
}

#[test]
fn packed_yuv_converts_as_a_source_only() {
    assert!(can_convert(D3DFMT_YUY2, D3DFMT_X8R8G8B8));
    assert!(!can_convert(D3DFMT_X8R8G8B8, D3DFMT_YUY2));
}

#[test]
fn compressed_and_unlisted_formats_do_not_convert() {
    assert!(!can_convert(D3DFMT_DXT1, D3DFMT_A8R8G8B8));
    assert!(!can_convert(D3DFMT_A8R8G8B8, D3DFMT_DXT1));
    assert!(!can_convert(D3DFMT_A8R8G8B8, D3DFMT_A4R4G4B4));
    assert!(!can_convert(D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8));
}

#[test]
fn a8r8g8b8_into_x8r8g8b8_keeps_the_colour_and_forces_alpha_opaque() {
    let src = 0x0011_2233_u32.to_le_bytes();
    let mut dst = [0u8; 4];
    assert!(convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(1, 1, 4, 4)
    ));
    assert_eq!(u32::from_le_bytes(dst), 0xFF11_2233);
}

#[test]
fn x8r8g8b8_into_a8r8g8b8_reads_the_source_alpha_as_opaque() {
    let src = 0x0000_FF00_u32.to_le_bytes();
    let mut dst = [0u8; 4];
    assert!(convert_region(
        &mut dst,
        D3DFMT_A8R8G8B8,
        &src,
        D3DFMT_X8R8G8B8,
        &whole(1, 1, 4, 4)
    ));
    assert_eq!(u32::from_le_bytes(dst), 0xFF00_FF00);
}

#[test]
fn r5g6b5_widens_a_saturated_channel_to_255() {
    // 0xF800 is R=31, G=0, B=0.
    let src = 0xF800_u16.to_le_bytes();
    let mut dst = [0u8; 4];
    assert!(convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_R5G6B5,
        &whole(1, 1, 2, 4)
    ));
    assert_eq!(u32::from_le_bytes(dst), 0xFFFF_0000);
}

#[test]
fn a8r8g8b8_into_r5g6b5_truncates_each_channel() {
    let src = 0xFFFF_0000_u32.to_le_bytes();
    let mut dst = [0u8; 2];
    assert!(convert_region(
        &mut dst,
        D3DFMT_R5G6B5,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(1, 1, 4, 2)
    ));
    assert_eq!(u16::from_le_bytes(dst), 0xF800);
}

#[test]
fn a_region_lands_at_the_destination_origin() {
    // 2x2 source, 4x4 destination, region (1,1)..(2,2) landing at (2,3).
    let mut src = [0u8; 2 * 2 * 4];
    src[4 * 3..4 * 4].copy_from_slice(&0x00AA_BBCC_u32.to_le_bytes());
    let mut dst = [0u8; 4 * 4 * 4];
    let region = ConvertRegion {
        src_x: 1,
        src_y: 1,
        dst_x: 2,
        dst_y: 3,
        width: 1,
        height: 1,
        src_pitch: 8,
        dst_pitch: 16,
        src_slice_pitch: 16,
        dst_slice_pitch: 64,
        depth: 1,
    };
    assert!(convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_A8R8G8B8,
        &region
    ));
    let texel = 3 * 16 + 2 * 4;
    assert_eq!(
        u32::from_le_bytes(dst[texel..texel + 4].try_into().unwrap()),
        0xFFAA_BBCC
    );
    // Nothing else was written.
    assert_eq!(dst.iter().filter(|b| **b != 0).count(), 4);
}

#[test]
fn every_depth_slice_converts() {
    // 1x1x2 volume: each slice holds one texel.
    let mut src = [0u8; 8];
    src[..4].copy_from_slice(&0xFFFF_0000_u32.to_le_bytes());
    src[4..].copy_from_slice(&0xFF00_FF00_u32.to_le_bytes());
    let mut dst = [0u8; 4];
    let region = ConvertRegion {
        src_x: 0,
        src_y: 0,
        dst_x: 0,
        dst_y: 0,
        width: 1,
        height: 1,
        src_pitch: 4,
        dst_pitch: 2,
        src_slice_pitch: 4,
        dst_slice_pitch: 2,
        depth: 2,
    };
    assert!(convert_region(
        &mut dst,
        D3DFMT_R5G6B5,
        &src,
        D3DFMT_A8R8G8B8,
        &region
    ));
    assert_eq!(u16::from_le_bytes([dst[0], dst[1]]), 0xF800);
    assert_eq!(u16::from_le_bytes([dst[2], dst[3]]), 0x07E0);
}

#[test]
fn a_region_running_past_the_destination_is_reported() {
    let src = [0u8; 4 * 4];
    let mut dst = [0u8; 4];
    assert!(!convert_region(
        &mut dst,
        D3DFMT_A8R8G8B8,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(4, 1, 16, 16)
    ));
}

#[test]
fn a_region_running_past_the_source_is_reported() {
    let src = [0u8; 4];
    let mut dst = [0u8; 4 * 4];
    assert!(!convert_region(
        &mut dst,
        D3DFMT_A8R8G8B8,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(4, 1, 16, 16)
    ));
}

#[test]
fn an_unconvertible_pair_writes_nothing() {
    let src = [0xFFu8; 8];
    let mut dst = [0u8; 8];
    assert!(!convert_region(
        &mut dst,
        D3DFMT_A4R4G4B4,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(2, 1, 8, 4)
    ));
    assert!(dst.iter().all(|b| *b == 0));
}
