use mtld3d_types::{D3DFMT_DXT1, D3DFMT_L16, D3DFMT_NV12, D3DFMT_V8U8, D3DFMT_YUY2, D3DFMT_YV12};

use super::*;
use crate::stretch_rect::yuv_to_rgb8;

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

#[test]
fn wide_sources_quantize_only_into_bgra8() {
    for source in [
        mtld3d_types::D3DFMT_A16B16G16R16,
        mtld3d_types::D3DFMT_A32B32G32R32F,
    ] {
        assert!(can_convert(source, D3DFMT_A8R8G8B8));
        assert!(!can_convert_update(source, D3DFMT_A8R8G8B8));
        for destination in [source, D3DFMT_R5G6B5, D3DFMT_L8, D3DFMT_X8R8G8B8] {
            let mut destination_bytes = [0x5a; 16];
            assert!(!convert_region(
                &mut destination_bytes,
                destination,
                &[0xff; 16],
                source,
                &whole(1, 1, 16, 16)
            ));
            assert_eq!(destination_bytes, [0x5a; 16]);
        }
    }
}

#[test]
fn unorm16_quantization_covers_every_channel_value() {
    for value in 0..=u16::MAX {
        let source: Vec<_> = [value, value, value, value]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let actual = into_argb(mtld3d_types::D3DFMT_A16B16G16R16, &source);
        // Nearest member of the destination's exact normalized grid: consecutive
        // 8-bit values are 257 source units apart, with no integer halfway tie.
        let expected = u32::from((value / 257) + u16::from(value % 257 > 128));
        assert_eq!(actual, expected * 0x0101_0101, "source {value}");
    }
}

#[test]
fn float32_quantization_preserves_source_precision_and_clamps() {
    for (channels, expected) in [
        (
            [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0],
            0x0000_ff00,
        ),
        (
            [0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0, 127.5 / 255.0],
            0x8001_0203,
        ),
        (
            [126.5 / 255.0, 128.5 / 255.0, 253.5 / 255.0, 254.5 / 255.0],
            0xfe7f_80fd,
        ),
        (
            [0.499 / 255.0, 0.501 / 255.0, 2.499 / 255.0, 2.501 / 255.0],
            0x0300_0102,
        ),
        ([-0.25, 1.25, 0.5, 1.0], 0xff00_ff80),
    ] {
        let source: Vec<_> = channels.into_iter().flat_map(f32::to_le_bytes).collect();
        assert_eq!(
            into_argb(mtld3d_types::D3DFMT_A32B32G32R32F, &source),
            expected
        );
    }
}

#[test]
fn wide_regions_preserve_padding_origins_and_other_slices() {
    for (format, pixel) in [
        (
            mtld3d_types::D3DFMT_A16B16G16R16,
            [0x3333_u16, 0x6666, 0x9999, 0xcccc]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            mtld3d_types::D3DFMT_A32B32G32R32F,
            [0.2_f32, 0.4, 0.6, 0.8]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        ),
    ] {
        let bpp = pixel.len();
        let src_pitch = 3 * bpp + 7;
        let src_slice_pitch = src_pitch * 3 + 9;
        let mut src = vec![0; src_slice_pitch * 2];
        let mut dst = vec![0x5a; 136];
        let mut expected = dst.clone();
        for z in 0..2 {
            for y in 1..3 {
                let off = z * src_slice_pitch + y * src_pitch + bpp;
                src[off..off + bpp].copy_from_slice(&pixel);
                let out = z * 68 + (y - 1) * 20 + 8;
                expected[out..out + 4].copy_from_slice(&0xcc33_6699_u32.to_le_bytes());
            }
        }
        let region = ConvertRegion {
            src_x: 1,
            src_y: 1,
            dst_x: 2,
            dst_y: 0,
            width: 1,
            height: 2,
            src_pitch,
            dst_pitch: 20,
            src_slice_pitch,
            dst_slice_pitch: 68,
            depth: 2,
        };
        assert!(convert_region(
            &mut dst,
            D3DFMT_A8R8G8B8,
            &src,
            format,
            &region
        ));
        assert_eq!(dst, expected);
    }
}

/// Four chroma pairs with U and V distinct in each, one per 2x2 luma block in turn.
const PLANAR_CHROMA: [(u8, u8); 4] = [(0x5a, 0xf0), (0x36, 0x22), (0xf0, 0x6e), (0x80, 0x80)];

/// The chroma pair block `(cx, cy)` of the test pattern carries.
fn planar_chroma(cx: usize, cy: usize) -> (u8, u8) {
    PLANAR_CHROMA[(cx + 2 * cy) % PLANAR_CHROMA.len()]
}

/// The luma the test pattern carries at `(x, y)`, distinct per texel.
fn planar_luma(x: usize, y: usize) -> u8 {
    u8::try_from(0x30 + 0x10 * y + 3 * x).unwrap()
}

/// A planar surface holding the test pattern, and the pitch it is laid out at.
///
/// The pitch is the width rounded up to four bytes, so a width that is not a
/// multiple of four leaves padding the chroma addressing has to step over.
/// Padding bytes hold `0xEE`, which no plane of the pattern uses.
fn planar_pattern(d3d_format: u32, width: usize, height: usize) -> (Vec<u8>, usize) {
    let pitch = width.next_multiple_of(4);
    let chroma_rows = height.div_ceil(2);
    let mut bytes = vec![0xEEu8; pitch * (height + chroma_rows)];
    for y in 0..height {
        for x in 0..width {
            bytes[y * pitch + x] = planar_luma(x, y);
        }
    }
    let chroma = pitch * height;
    for cy in 0..chroma_rows {
        for cx in 0..width.div_ceil(2) {
            let (u, v) = planar_chroma(cx, cy);
            if d3d_format == D3DFMT_YV12 {
                let half = pitch / 2;
                bytes[chroma + cy * half + cx] = v;
                bytes[chroma + chroma_rows * half + cy * half + cx] = u;
            } else {
                bytes[chroma + cy * pitch + 2 * cx] = u;
                bytes[chroma + cy * pitch + 2 * cx + 1] = v;
            }
        }
    }
    (bytes, pitch)
}

/// The `X8R8G8B8` word the test pattern decodes to at `(x, y)`.
fn planar_expected(x: usize, y: usize) -> u32 {
    let (cb, cr) = planar_chroma(x / 2, y / 2);
    let (red, green, blue) = yuv_to_rgb8(planar_luma(x, y), cb, cr);
    0xFF00_0000 | (u32::from(red) << 16) | (u32::from(green) << 8) | u32::from(blue)
}

#[test]
fn planar_yuv_converts_as_a_stretch_source_only() {
    for src in [D3DFMT_YV12, D3DFMT_NV12] {
        for dst in CONVERTIBLE {
            assert!(can_convert(src, dst), "{src:#x} into {dst:#x}");
            assert!(!can_convert(dst, src), "{dst:#x} into {src:#x}");
            // The Update APIs never see a planar endpoint.
            assert!(!can_convert_update(src, dst), "{src:#x} into {dst:#x}");
            assert!(!can_convert_update(dst, src), "{dst:#x} into {src:#x}");
        }
        assert!(!can_convert(src, D3DFMT_YV12));
        assert!(!can_convert(src, D3DFMT_NV12));
        assert!(!can_convert(src, D3DFMT_YUY2));
        assert!(!can_convert(src, mtld3d_types::D3DFMT_A16B16G16R16));
    }
}

#[test]
fn a_planar_chroma_sample_covers_exactly_its_two_by_two_luma_block() {
    // Width 6 lays out at a pitch of 8, so every plane carries padding.
    let (width, height) = (6usize, 4usize);
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let (src, pitch) = planar_pattern(format, width, height);
        let mut dst = vec![0u8; width * 4 * height];
        assert!(convert_region(
            &mut dst,
            D3DFMT_X8R8G8B8,
            &src,
            format,
            &whole(6, 4, pitch, width * 4)
        ));
        for y in 0..height {
            for x in 0..width {
                let off = (y * width + x) * 4;
                let got = u32::from_le_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
                assert_eq!(got, planar_expected(x, y), "{format:#x} at ({x}, {y})");
            }
        }
    }
}

#[test]
fn a_planar_source_rect_with_an_odd_origin_keeps_its_chroma_blocks() {
    let (width, height) = (6usize, 4usize);
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let (src, pitch) = planar_pattern(format, width, height);
        let mut dst = vec![0x77u8; width * 4 * height];
        let region = ConvertRegion {
            src_x: 1,
            src_y: 1,
            dst_x: 2,
            dst_y: 1,
            width: 3,
            height: 2,
            src_pitch: pitch,
            dst_pitch: width * 4,
            src_slice_pitch: pitch * height,
            dst_slice_pitch: width * 4 * height,
            depth: 1,
        };
        assert!(convert_region(
            &mut dst,
            D3DFMT_X8R8G8B8,
            &src,
            format,
            &region
        ));
        for y in 0..height {
            for x in 0..width {
                let off = (y * width + x) * 4;
                let got = u32::from_le_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
                let inside = (2..5).contains(&x) && (1..3).contains(&y);
                let expected = if inside {
                    planar_expected(x - 1, y)
                } else {
                    0x7777_7777
                };
                assert_eq!(got, expected, "{format:#x} at ({x}, {y})");
            }
        }
    }
}

#[test]
fn an_odd_nv12_height_converts_its_last_row_from_the_last_chroma_row() {
    let (width, height) = (5usize, 3usize);
    let (src, pitch) = planar_pattern(D3DFMT_NV12, width, height);
    let mut dst = vec![0u8; width * 4 * height];
    assert!(convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_NV12,
        &whole(5, 3, pitch, width * 4)
    ));
    for x in 0..width {
        let off = (2 * width + x) * 4;
        let got = u32::from_le_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
        assert_eq!(got, planar_expected(x, 2), "at ({x}, 2)");
    }
}

#[test]
fn a_planar_source_into_l8_stores_the_luma_of_the_decoded_colour() {
    // Pure red: the Rec. 709 luma of (255, 0, 0) is 54, not the 0x51 the Y
    // plane holds.
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let mut src = vec![0x51u8; 4 * 2];
        if format == D3DFMT_YV12 {
            src.extend_from_slice(&[0xf0, 0xf0, 0x5a, 0x5a]);
        } else {
            src.extend_from_slice(&[0x5a, 0xf0, 0x5a, 0xf0]);
        }
        let mut dst = [0u8; 8];
        assert!(convert_region(
            &mut dst,
            D3DFMT_L8,
            &src,
            format,
            &whole(4, 2, 4, 4)
        ));
        assert_eq!(dst, [54; 8], "{format:#x}");
    }
}

#[test]
fn a_planar_region_outside_the_luma_plane_is_refused() {
    let (src, pitch) = planar_pattern(D3DFMT_NV12, 6, 4);
    let mut dst = vec![0u8; 6 * 4 * 8];
    // Five rows of a four-row surface: the fifth would read the chroma plane
    // as luma.
    let mut region = whole(6, 4, pitch, 24);
    region.height = 5;
    region.dst_slice_pitch = 24 * 8;
    assert!(!convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_NV12,
        &region
    ));
    // A slice that is not a whole number of rows names no luma row count.
    let mut region = whole(6, 4, pitch, 24);
    region.src_slice_pitch += 1;
    assert!(!convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_NV12,
        &region
    ));
    // A planar level is a single slice.
    let mut region = whole(6, 4, pitch, 24);
    region.depth = 2;
    assert!(!convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src,
        D3DFMT_NV12,
        &region
    ));
    // A source allocation short of its chroma plane.
    let region = whole(6, 4, pitch, 24);
    assert!(!convert_region(
        &mut dst,
        D3DFMT_X8R8G8B8,
        &src[..pitch * 4],
        D3DFMT_NV12,
        &region
    ));
}
