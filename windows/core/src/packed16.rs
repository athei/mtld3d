//! CPU expansion of the packed 16-bit D3D formats to BGRA8.
//!
//! Devices without Metal's packed 16-bit pixel formats (Intel/AMD Mac2 —
//! Apple-family-only formats) back A4R4G4B4 / R5G6B5 / A1R5G5B5 textures
//! with `Bgra8Unorm` (`format::map_d3d_format_device`) and widen the texels
//! here at upload time. The PE-side staging keeps the original 16-bit D3D
//! layout (Lock semantics untouched); only the blit source handed to the GPU
//! is expanded.

use mtld3d_shared::mtl::PixelFormat;
use mtld3d_types::{D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_R5G6B5};

/// Which packed 16-bit source layout an expansion upload decodes.
#[derive(Clone, Copy)]
pub enum Packed16Kind {
    R5G6B5,
    A1R5G5B5,
    A4R4G4B4,
}

/// Expansion-upload classifier: `Some` iff this (source, GPU) pair expands.
///
/// Derives the decision from data every upload job already carries: the
/// source D3D format and the Metal format frozen at create time. On a device
/// with native packed 16-bit formats the GPU format is the packed one, never
/// `Bgra8Unorm`, so the native upload path cannot take the expansion branch
/// by construction.
#[must_use]
pub const fn expansion_kind(src_d3d_format: u32, gpu_format: PixelFormat) -> Option<Packed16Kind> {
    if !matches!(gpu_format, PixelFormat::Bgra8Unorm) {
        return None;
    }
    match src_d3d_format {
        D3DFMT_R5G6B5 => Some(Packed16Kind::R5G6B5),
        D3DFMT_A1R5G5B5 => Some(Packed16Kind::A1R5G5B5),
        D3DFMT_A4R4G4B4 => Some(Packed16Kind::A4R4G4B4),
        _ => None,
    }
}

/// Expand `rows` rows of packed 16-bit texels into BGRA8 bytes.
///
/// `src` holds little-endian `u16` texels at `src_pitch` bytes per row;
/// `dst` receives 4 bytes per texel in `Bgra8Unorm` memory order ([B, G, R,
/// A]) at `dst_pitch` bytes per row. Pitches may exceed the tight row size
/// (sub-rect sources, alignment-padded destinations); trailing destination
/// padding is left untouched. Channel widening replicates the high bits into
/// the low ones so 0 stays 0 and the max value maps to 255 exactly.
pub fn expand_rows(
    kind: Packed16Kind,
    src: &[u8],
    src_pitch: usize,
    dst: &mut [u8],
    dst_pitch: usize,
    width_px: usize,
    rows: usize,
) {
    for row in 0..rows {
        let src_row = &src[row * src_pitch..row * src_pitch + width_px * 2];
        let dst_row = &mut dst[row * dst_pitch..row * dst_pitch + width_px * 4];
        for (texel, out) in src_row.chunks_exact(2).zip(dst_row.chunks_exact_mut(4)) {
            let bits = u16::from_le_bytes([texel[0], texel[1]]);
            // Each arm yields Bgra8Unorm memory order: [B, G, R, A].
            let bgra = match kind {
                Packed16Kind::R5G6B5 => {
                    // R[11-15] G[5-10] B[0-4], no alpha (sampled as 1.0).
                    [
                        widen5((bits & 0x1F) as u8),
                        widen6(((bits >> 5) & 0x3F) as u8),
                        widen5(((bits >> 11) & 0x1F) as u8),
                        0xFF,
                    ]
                }
                Packed16Kind::A1R5G5B5 => {
                    // A[15] R[10-14] G[5-9] B[0-4].
                    [
                        widen5((bits & 0x1F) as u8),
                        widen5(((bits >> 5) & 0x1F) as u8),
                        widen5(((bits >> 10) & 0x1F) as u8),
                        if bits & 0x8000 == 0 { 0x00 } else { 0xFF },
                    ]
                }
                Packed16Kind::A4R4G4B4 => {
                    // A[12-15] R[8-11] G[4-7] B[0-3].
                    [
                        widen4((bits & 0xF) as u8),
                        widen4(((bits >> 4) & 0xF) as u8),
                        widen4(((bits >> 8) & 0xF) as u8),
                        widen4(((bits >> 12) & 0xF) as u8),
                    ]
                }
            };
            out.copy_from_slice(&bgra);
        }
    }
}

/// Widen a 4-bit channel to 8 bits (`0xF` → `0xFF` exactly).
#[inline]
const fn widen4(v: u8) -> u8 {
    v * 17
}

/// Widen a 5-bit channel to 8 bits (`0x1F` → `0xFF` exactly).
#[inline]
const fn widen5(v: u8) -> u8 {
    (v << 3) | (v >> 2)
}

/// Widen a 6-bit channel to 8 bits (`0x3F` → `0xFF` exactly).
#[inline]
const fn widen6(v: u8) -> u8 {
    (v << 2) | (v >> 4)
}

#[cfg(test)]
mod tests;
