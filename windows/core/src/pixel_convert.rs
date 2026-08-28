//! CPU re-encoding of one texel region between two uncompressed colour formats.
//!
//! Two D3D9 paths copy between surfaces whose formats differ and are expected
//! to convert rather than to reinterpret the bytes: the cross-format
//! `StretchRect` into an offscreen-plain destination, and `UpdateSurface` /
//! `UpdateTexture`, which accept a mismatched pair. Both run here, on the CPU
//! staging that backs the destination level.
//!
//! The codec covers the simple RGB formats in both directions plus the packed
//! 4:2:2 YUV formats as a source. [`can_convert`] answers for a pair up front,
//! so a caller can reject the pairs no codec covers instead of writing
//! reinterpreted bytes.

use mtld3d_types::{D3DFMT_A8R8G8B8, D3DFMT_R5G6B5, D3DFMT_X8R8G8B8};

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
/// Destinations are the three simple RGB formats; sources are those plus the
/// packed 4:2:2 YUV pair. Anything else (block-compressed, depth, the
/// floating-point and packed-16 colour formats) has no codec here.
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

/// The simple uncompressed RGB formats the converter decodes and encodes.
const fn is_convertible_rgb(d3d_format: u32) -> bool {
    matches!(
        d3d_format,
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 | D3DFMT_R5G6B5
    )
}

/// Bytes per texel of an `is_convertible_rgb` format.
const fn rgb_bpp(d3d_format: u32) -> usize {
    match d3d_format {
        D3DFMT_R5G6B5 => 2,
        _ => 4, // A8R8G8B8 / X8R8G8B8
    }
}

/// Decode one `is_convertible_rgb` texel from its little-endian bytes into `(r, g, b, a)`.
///
/// The 32bpp formats store `[B, G, R, A/X]`; R5G6B5 packs `RRRRR GGGGGG BBBBB`
/// into a little-endian `u16` (channels bit-replicated up to 8 bits). X8's
/// alpha reads as opaque. `px` is at least `rgb_bpp` long.
fn decode_rgb_pixel(d3d_format: u32, px: &[u8]) -> (u8, u8, u8, u8) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let v = u16::from_le_bytes([px[0], px[1]]);
            let r5 = ((v >> 11) & 0x1f) as u8;
            let g6 = ((v >> 5) & 0x3f) as u8;
            let b5 = (v & 0x1f) as u8;
            (
                (r5 << 3) | (r5 >> 2),
                (g6 << 2) | (g6 >> 4),
                (b5 << 3) | (b5 >> 2),
                0xff,
            )
        }
        // X8R8G8B8: alpha byte is undefined; report opaque.
        D3DFMT_X8R8G8B8 => (px[2], px[1], px[0], 0xff),
        // A8R8G8B8 (only remaining is_convertible_rgb case).
        _ => (px[2], px[1], px[0], px[3]),
    }
}

/// Encode `(r, g, b, a)` into one `is_convertible_rgb` texel's little-endian bytes.
///
/// Inverse of `decode_rgb_pixel`. X8's byte is written opaque. `out` is at
/// least `rgb_bpp` long.
fn encode_rgb_pixel(d3d_format: u32, (r, g, b, a): (u8, u8, u8, u8), out: &mut [u8]) {
    match d3d_format {
        D3DFMT_R5G6B5 => {
            let packed = (u16::from(r >> 3) << 11) | (u16::from(g >> 2) << 5) | u16::from(b >> 3);
            out[..2].copy_from_slice(&packed.to_le_bytes());
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
