use mtld3d_types::{D3DFMT_DXT1, D3DFMT_L16, D3DFMT_V8U8, D3DFMT_YUY2};

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

/// Every colour format the codec covers, one per `is_convertible_rgb` arm.
const CONVERTIBLE: [u32; 12] = [
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    D3DFMT_A8B8G8R8,
    D3DFMT_X8B8G8R8,
    D3DFMT_R8G8B8,
    D3DFMT_R5G6B5,
    D3DFMT_A1R5G5B5,
    D3DFMT_X1R5G5B5,
    D3DFMT_A4R4G4B4,
    D3DFMT_L8,
    D3DFMT_A8,
    D3DFMT_A8L8,
];

/// Convert one texel of `src` and hand back the `dst_bpp` destination bytes.
fn one_texel(dst_format: u32, dst_bpp: usize, src_format: u32, src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0u8; dst_bpp];
    assert!(
        convert_region(
            &mut dst,
            dst_format,
            src,
            src_format,
            &whole(1, 1, src.len(), dst_bpp)
        ),
        "{src_format:#x} into {dst_format:#x}"
    );
    dst
}

/// Convert one texel into a 32-bit `A8R8G8B8` word.
fn into_argb(src_format: u32, src: &[u8]) -> u32 {
    let dst = one_texel(D3DFMT_A8R8G8B8, 4, src_format, src);
    u32::from_le_bytes([dst[0], dst[1], dst[2], dst[3]])
}

/// Convert an `A8R8G8B8` word into one texel of a 16-bit format.
fn from_argb_16(dst_format: u32, argb: u32) -> u16 {
    let dst = one_texel(dst_format, 2, D3DFMT_A8R8G8B8, &argb.to_le_bytes());
    u16::from_le_bytes([dst[0], dst[1]])
}

#[test]
fn the_colour_formats_convert_in_both_directions() {
    for src in CONVERTIBLE {
        for dst in CONVERTIBLE {
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
    assert!(!can_convert(D3DFMT_A8R8G8B8, D3DFMT_V8U8));
    assert!(!can_convert(D3DFMT_V8U8, D3DFMT_A8R8G8B8));
    assert!(!can_convert(D3DFMT_A8R8G8B8, D3DFMT_L16));
    assert!(!can_convert(D3DFMT_L16, D3DFMT_A8R8G8B8));
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
fn the_reversed_channel_formats_swap_red_and_blue() {
    // A8B8G8R8 stores [R, G, B, A] in ascending addresses, so the same word
    // read as A8R8G8B8 is the colour with red and blue exchanged.
    assert_eq!(
        into_argb(D3DFMT_A8B8G8R8, &0xFF00_00FF_u32.to_le_bytes()),
        0xFFFF_0000
    );
    let dst = one_texel(
        D3DFMT_A8B8G8R8,
        4,
        D3DFMT_A8R8G8B8,
        &0xFFFF_0000_u32.to_le_bytes(),
    );
    assert_eq!(
        u32::from_le_bytes([dst[0], dst[1], dst[2], dst[3]]),
        0xFF00_00FF
    );
}

#[test]
fn x8b8g8r8_reads_and_writes_its_padding_byte_as_opaque() {
    assert_eq!(
        into_argb(D3DFMT_X8B8G8R8, &0x0000_00FF_u32.to_le_bytes()),
        0xFFFF_0000
    );
    // A transparent source still encodes with the ignored byte opaque.
    let dst = one_texel(
        D3DFMT_X8B8G8R8,
        4,
        D3DFMT_A8R8G8B8,
        &0x00FF_0000_u32.to_le_bytes(),
    );
    assert_eq!(
        u32::from_le_bytes([dst[0], dst[1], dst[2], dst[3]]),
        0xFF00_00FF
    );
}

#[test]
fn r8g8b8_carries_three_bytes_per_texel() {
    // 24-bit [B, G, R], with no channel left for alpha.
    assert_eq!(into_argb(D3DFMT_R8G8B8, &[0x00, 0x00, 0xFF]), 0xFFFF_0000);
    assert_eq!(
        one_texel(
            D3DFMT_R8G8B8,
            3,
            D3DFMT_A8R8G8B8,
            &0x00FF_0000_u32.to_le_bytes()
        ),
        vec![0x00, 0x00, 0xFF]
    );
}

#[test]
fn a1r5g5b5_round_trips_through_a8r8g8b8() {
    // A=1, R=0, G=31, B=0.
    const GREEN_1555: u16 = 0x83E0;
    const GREEN: u32 = 0xFF00_FF00;
    assert_eq!(into_argb(D3DFMT_A1R5G5B5, &GREEN_1555.to_le_bytes()), GREEN);
    assert_eq!(from_argb_16(D3DFMT_A1R5G5B5, GREEN), GREEN_1555);
    // Every channel saturated widens to white, and back.
    assert_eq!(
        into_argb(D3DFMT_A1R5G5B5, &0xFFFF_u16.to_le_bytes()),
        0xFFFF_FFFF
    );
    assert_eq!(from_argb_16(D3DFMT_A1R5G5B5, 0xFFFF_FFFF), 0xFFFF);
}

#[test]
fn a1r5g5b5_carries_its_one_bit_alpha_both_ways() {
    // The same colour with the alpha bit clear decodes transparent.
    assert_eq!(
        into_argb(D3DFMT_A1R5G5B5, &0x03E0_u16.to_le_bytes()),
        0x0000_FF00
    );
    // Encoding rounds to the nearer of the two alpha values.
    assert_eq!(from_argb_16(D3DFMT_A1R5G5B5, 0x7F00_FF00), 0x03E0);
    assert_eq!(from_argb_16(D3DFMT_A1R5G5B5, 0x8000_FF00), 0x83E0);
}

#[test]
fn x1r5g5b5_reads_and_writes_its_top_bit_as_opaque() {
    // Top bit clear, yet the decode reports opaque.
    assert_eq!(
        into_argb(D3DFMT_X1R5G5B5, &0x03E0_u16.to_le_bytes()),
        0xFF00_FF00
    );
    // A transparent source still encodes with the ignored bit set.
    assert_eq!(from_argb_16(D3DFMT_X1R5G5B5, 0x0000_FF00), 0x83E0);
}

#[test]
fn a4r4g4b4_round_trips_through_a8r8g8b8() {
    // A=F, R=F, G=0, B=0.
    const RED_4444: u16 = 0xFF00;
    const RED: u32 = 0xFFFF_0000;
    assert_eq!(into_argb(D3DFMT_A4R4G4B4, &RED_4444.to_le_bytes()), RED);
    assert_eq!(from_argb_16(D3DFMT_A4R4G4B4, RED), RED_4444);
    // A half-scale nibble replicates into both halves of the byte.
    assert_eq!(
        into_argb(D3DFMT_A4R4G4B4, &0x8888_u16.to_le_bytes()),
        0x8888_8888
    );
}

#[test]
fn l8_replicates_its_luminance_and_takes_rec709_luma_back() {
    assert_eq!(into_argb(D3DFMT_L8, &[0x80]), 0xFF80_8080);
    // Rec. 709 luma of pure green is 0.7154 * 255, rounded.
    assert_eq!(
        one_texel(
            D3DFMT_L8,
            1,
            D3DFMT_A8R8G8B8,
            &0xFF00_FF00_u32.to_le_bytes()
        ),
        vec![182]
    );
    // A grey encodes back to itself.
    assert_eq!(
        one_texel(
            D3DFMT_L8,
            1,
            D3DFMT_A8R8G8B8,
            &0xFF80_8080_u32.to_le_bytes()
        ),
        vec![0x80]
    );
}

#[test]
fn a8_carries_alpha_only() {
    // A8 has no colour: its RGB decodes black.
    assert_eq!(into_argb(D3DFMT_A8, &[0x7F]), 0x7F00_0000);
    assert_eq!(
        one_texel(
            D3DFMT_A8,
            1,
            D3DFMT_A8R8G8B8,
            &0x7FFF_FFFF_u32.to_le_bytes()
        ),
        vec![0x7F]
    );
}

#[test]
fn a8l8_pairs_luminance_with_alpha() {
    // Luminance in the low byte, alpha in the high one.
    assert_eq!(into_argb(D3DFMT_A8L8, &[0x40, 0x80]), 0x8040_4040);
    assert_eq!(
        one_texel(
            D3DFMT_A8L8,
            2,
            D3DFMT_A8R8G8B8,
            &0x8000_FF00_u32.to_le_bytes()
        ),
        vec![182, 0x80]
    );
}

#[test]
fn the_packed_16_bit_formats_convert_between_each_other() {
    // 5-6-5 green, into 5-5-5 green: the 6-bit lane loses its low bit.
    assert_eq!(
        from_argb_16(
            D3DFMT_A1R5G5B5,
            into_argb(D3DFMT_R5G6B5, &0x07E0_u16.to_le_bytes())
        ),
        0x83E0
    );
    // 4-4-4-4 red into 5-6-5: the widened nibble truncates back to 5 bits.
    let red = into_argb(D3DFMT_A4R4G4B4, &0xFF00_u16.to_le_bytes());
    assert_eq!(from_argb_16(D3DFMT_R5G6B5, red), 0xF800);
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
        D3DFMT_V8U8,
        &src,
        D3DFMT_A8R8G8B8,
        &whole(2, 1, 8, 4)
    ));
    assert!(dst.iter().all(|b| *b == 0));
}
