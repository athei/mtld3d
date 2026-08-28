//! CPU re-encoding of one texel region between two uncompressed colour formats.
//!
//! Two D3D9 paths copy between surfaces whose formats differ and are expected
//! to convert rather than to reinterpret the bytes: the cross-format
//! `StretchRect` into an offscreen-plain destination, and `UpdateSurface` /
//! `UpdateTexture`, which accept a mismatched pair. Both run here, on the CPU
//! staging that backs the destination level.
//!
//! The codec covers every uncompressed colour format whose channels are
//! unsigned normalised and at most 8 bits wide, in both directions, plus the
//! packed 4:2:2 YUV formats as a source. RGBA8 is the intermediate, so it
//! carries those formats without loss. [`can_convert`] answers for a pair up
//! front, so a caller can reject the pairs no codec covers instead of writing
//! reinterpreted bytes.

use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8L8, D3DFMT_A8R8G8B8,
    D3DFMT_L8, D3DFMT_R5G6B5, D3DFMT_R8G8B8, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8,
};

use crate::stretch_rect::{decode_packed_yuv, is_packed_yuv};

/// One [`convert_region`] copy: the shared extent, both origins, both layouts.
///
/// Origins and extent are in texels; the pitches are in bytes. `depth` is 1
/// for a 2D level or a cube face, and the slice count the two levels share for
/// a volume level, whose slices sit back to back in one allocation.
pub struct ConvertRegion {
    pub src_x: u32,
    pub src_y: u32,
    pub dst_x: u32,
    pub dst_y: u32,
    pub width: u32,
    pub height: u32,
    pub src_pitch: usize,
    pub dst_pitch: usize,
    pub src_slice_pitch: usize,
    pub dst_slice_pitch: usize,
    pub depth: usize,
}

/// Whether [`convert_region`] re-encodes this source format into this destination format.
///
/// Destinations are the uncompressed 8-bit-or-narrower unsigned normalised
/// colour formats; sources are those plus the packed 4:2:2 YUV pair. Anything
/// else (block-compressed, depth, palettised, signed, and the 16-bit-per-
/// channel and floating-point colour formats) has no codec here.
#[must_use]
pub const fn can_convert(src_format: u32, dst_format: u32) -> bool {
    (is_convertible_rgb(src_format) || is_packed_yuv(src_format)) && is_convertible_rgb(dst_format)
}

/// Re-encode `region` of `src` into `dst`, one texel at a time.
///
/// `src` and `dst` are whole level allocations, addressed through the region's
/// pitches. Returns false for a format pair [`can_convert`] rejects and for a
/// region that runs past either allocation, in which case the destination
/// holds however much of the region was converted before the overrun.
#[must_use]
pub fn convert_region(
    dst: &mut [u8],
    dst_format: u32,
    src: &[u8],
    src_format: u32,
    region: &ConvertRegion,
) -> bool {
    if !can_convert(src_format, dst_format) {
        return false;
    }
    let yuv_src = is_packed_yuv(src_format);
    let src_bpp = rgb_bpp(src_format);
    let dst_bpp = rgb_bpp(dst_format);
    for z in 0..region.depth {
        for row in 0..region.height {
            let src_row =
                z * region.src_slice_pitch + (region.src_y + row) as usize * region.src_pitch;
            let dst_row =
                z * region.dst_slice_pitch + (region.dst_y + row) as usize * region.dst_pitch;
            for col in 0..region.width {
                let sx = (region.src_x + col) as usize;
                let rgba = if yuv_src {
                    // Packed 4:2:2: the texel's macropixel is the four bytes at
                    // the even column; its parity picks the luma sample.
                    let off = src_row + (sx & !1) * 2;
                    let Some(mp) = src.get(off..off + 4) else {
                        return false;
                    };
                    let Some((r, g, b)) =
                        decode_packed_yuv(src_format, [mp[0], mp[1], mp[2], mp[3]], sx & 1 == 1)
                    else {
                        return false;
                    };
                    (r, g, b, 0xff)
                } else {
                    let off = src_row + sx * src_bpp;
                    let Some(px) = src.get(off..off + src_bpp) else {
                        return false;
                    };
                    decode_rgb_pixel(src_format, px)
                };
                let off = dst_row + (region.dst_x + col) as usize * dst_bpp;
                let Some(out) = dst.get_mut(off..off + dst_bpp) else {
                    return false;
                };
                encode_rgb_pixel(dst_format, rgba, out);
            }
        }
    }
    true
}

/// The uncompressed colour formats the converter decodes and encodes.
///
/// Every member's channels are unsigned normalised and at most 8 bits wide, so
/// the RGBA8 intermediate holds a decoded texel exactly.
const fn is_convertible_rgb(d3d_format: u32) -> bool {
    matches!(
        d3d_format,
        D3DFMT_A8R8G8B8
            | D3DFMT_X8R8G8B8
            | D3DFMT_A8B8G8R8
            | D3DFMT_X8B8G8R8
            | D3DFMT_R8G8B8
            | D3DFMT_R5G6B5
            | D3DFMT_A1R5G5B5
            | D3DFMT_X1R5G5B5
            | D3DFMT_A4R4G4B4
            | D3DFMT_L8
            | D3DFMT_A8
            | D3DFMT_A8L8
    )
}

/// Bytes per texel of an `is_convertible_rgb` format.
const fn rgb_bpp(d3d_format: u32) -> usize {
    match d3d_format {
        D3DFMT_L8 | D3DFMT_A8 => 1,
        D3DFMT_R5G6B5 | D3DFMT_A1R5G5B5 | D3DFMT_X1R5G5B5 | D3DFMT_A4R4G4B4 | D3DFMT_A8L8 => 2,
        D3DFMT_R8G8B8 => 3,
        _ => 4, // A8R8G8B8 / X8R8G8B8 / A8B8G8R8 / X8B8G8R8
    }
}

/// Widen a 5-bit channel to 8 bits, replicating its top bits into the low ones.
const fn expand5(v: u8) -> u8 {
    (v << 3) | (v >> 2)
}

/// Widen a 6-bit channel to 8 bits, replicating its top bits into the low ones.
const fn expand6(v: u8) -> u8 {
    (v << 2) | (v >> 4)
}

/// Widen a 4-bit channel to 8 bits, replicating it into the low nibble.
const fn expand4(v: u8) -> u8 {
    (v << 4) | v
}

/// The luminance an RGB triple carries, as D3D's colour-space conversions weight it.
///
/// Rec. 709 luma: the weights sum to 1, so a 0..=255 input gives a 0..=255
/// result, and the added half is the round-to-nearest the fixed-point form
/// needs.
fn luminance(r: u8, g: u8, b: u8) -> u8 {
    let weighted =
        2125 * u32::from(r) + 7154 * u32::from(g) + 721 * u32::from(b) + LUMA_ROUNDING_HALF;
    ((weighted / LUMA_WEIGHT_TOTAL) & 0xff) as u8
}

/// The denominator the fixed-point [`luminance`] weights share.
const LUMA_WEIGHT_TOTAL: u32 = 10000;

/// Half of [`LUMA_WEIGHT_TOTAL`], added before the divide to round to nearest.
const LUMA_ROUNDING_HALF: u32 = LUMA_WEIGHT_TOTAL / 2;

/// Decode one `is_convertible_rgb` texel from its little-endian bytes into `(r, g, b, a)`.
///
/// The `*R8G8B8` formats store `[B, G, R]` and then the alpha or padding byte
/// the 32bpp ones carry; the `*B8G8R8` pair reverses that to `[R, G, B]`. The
/// packed 16-bit formats hold their channels in one little-endian `u16`,
/// widened to 8 bits by bit replication. An X format's alpha reads as opaque,
/// `L8` replicates its luminance across RGB, and `A8` carries no colour, so
/// its RGB reads as black. `px` is at least `rgb_bpp` long.
fn decode_rgb_pixel(d3d_format: u32, px: &[u8]) -> (u8, u8, u8, u8) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let v = u16::from_le_bytes([px[0], px[1]]);
            let r5 = ((v >> 11) & 0x1f) as u8;
            let g6 = ((v >> 5) & 0x3f) as u8;
            let b5 = (v & 0x1f) as u8;
            (expand5(r5), expand6(g6), expand5(b5), 0xff)
        }
        // A1R5G5B5 / X1R5G5B5: A[15] R[14:10] G[9:5] B[4:0]. X1's top bit is
        // undefined; report opaque.
        D3DFMT_A1R5G5B5 | D3DFMT_X1R5G5B5 => {
            let v = u16::from_le_bytes([px[0], px[1]]);
            let r5 = ((v >> 10) & 0x1f) as u8;
            let g5 = ((v >> 5) & 0x1f) as u8;
            let b5 = (v & 0x1f) as u8;
            let a = if d3d_format == D3DFMT_X1R5G5B5 || v & 0x8000 != 0 {
                0xff
            } else {
                0x00
            };
            (expand5(r5), expand5(g5), expand5(b5), a)
        }
        // A4R4G4B4: A[15:12] R[11:8] G[7:4] B[3:0].
        D3DFMT_A4R4G4B4 => {
            let v = u16::from_le_bytes([px[0], px[1]]);
            let a4 = ((v >> 12) & 0xf) as u8;
            let r4 = ((v >> 8) & 0xf) as u8;
            let g4 = ((v >> 4) & 0xf) as u8;
            let b4 = (v & 0xf) as u8;
            (expand4(r4), expand4(g4), expand4(b4), expand4(a4))
        }
        // A8L8: luminance in the low byte, alpha in the high one.
        D3DFMT_A8L8 => (px[0], px[0], px[0], px[1]),
        D3DFMT_L8 => (px[0], px[0], px[0], 0xff),
        D3DFMT_A8 => (0, 0, 0, px[0]),
        // The reversed-channel twins store [R, G, B, A/X].
        D3DFMT_A8B8G8R8 => (px[0], px[1], px[2], px[3]),
        D3DFMT_X8B8G8R8 => (px[0], px[1], px[2], 0xff),
        // R8G8B8 is 24-bit [B, G, R], with no channel left to carry alpha;
        // X8R8G8B8 adds a fourth byte that is undefined. Both report opaque.
        D3DFMT_R8G8B8 | D3DFMT_X8R8G8B8 => (px[2], px[1], px[0], 0xff),
        // A8R8G8B8 (only remaining is_convertible_rgb case).
        _ => (px[2], px[1], px[0], px[3]),
    }
}

/// Encode `(r, g, b, a)` into one `is_convertible_rgb` texel's little-endian bytes.
///
/// Inverse of `decode_rgb_pixel`. A channel narrower than 8 bits takes the top
/// bits of its input, which is what the widening decode reverses; a 1-bit
/// alpha rounds instead, so a half-opaque source lands opaque. An X format's
/// bits are written opaque, and a luminance destination takes the Rec. 709
/// luma of the colour. `out` is at least `rgb_bpp` long.
fn encode_rgb_pixel(d3d_format: u32, (r, g, b, a): (u8, u8, u8, u8), out: &mut [u8]) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let packed = (u16::from(r >> 3) << 11) | (u16::from(g >> 2) << 5) | u16::from(b >> 3);
            out[..2].copy_from_slice(&packed.to_le_bytes());
        }
        D3DFMT_A1R5G5B5 | D3DFMT_X1R5G5B5 => {
            let opaque = d3d_format == D3DFMT_X1R5G5B5 || a >= 0x80;
            let packed = (u16::from(opaque) << 15)
                | (u16::from(r >> 3) << 10)
                | (u16::from(g >> 3) << 5)
                | u16::from(b >> 3);
            out[..2].copy_from_slice(&packed.to_le_bytes());
        }
        D3DFMT_A4R4G4B4 => {
            let packed = (u16::from(a >> 4) << 12)
                | (u16::from(r >> 4) << 8)
                | (u16::from(g >> 4) << 4)
                | u16::from(b >> 4);
            out[..2].copy_from_slice(&packed.to_le_bytes());
        }
        D3DFMT_A8L8 => {
            out[0] = luminance(r, g, b);
            out[1] = a;
        }
        D3DFMT_L8 => out[0] = luminance(r, g, b),
        D3DFMT_A8 => out[0] = a,
        D3DFMT_A8B8G8R8 | D3DFMT_X8B8G8R8 => {
            out[0] = r;
            out[1] = g;
            out[2] = b;
            out[3] = if d3d_format == D3DFMT_X8B8G8R8 {
                0xff
            } else {
                a
            };
        }
        D3DFMT_R8G8B8 => {
            out[0] = b;
            out[1] = g;
            out[2] = r;
        }
        D3DFMT_X8R8G8B8 => {
            out[0] = b;
            out[1] = g;
            out[2] = r;
            out[3] = 0xff;
        }
        // A8R8G8B8 (only remaining is_convertible_rgb case).
        _ => {
            out[0] = b;
            out[1] = g;
            out[2] = r;
            out[3] = a;
        }
    }
}

#[cfg(test)]
mod tests;
