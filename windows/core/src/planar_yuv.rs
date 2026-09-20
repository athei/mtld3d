//! Byte layout of the planar 4:2:0 YUV surface formats, `YV12` and `NV12`.
//!
//! A planar surface locks as one allocation: `height` luma rows of `pitch`
//! bytes, then the chroma at half resolution in both directions. `YV12` stores
//! a V plane and then a U plane, each of `ceil(height / 2)` rows striding half
//! the pitch. `NV12` stores one plane of the same row count striding the whole
//! pitch, U and V interleaved. Every plane is addressed from the pitch and
//! never from the width, so a width that is not a multiple of four, whose
//! pitch is wider than it, moves the chroma planes with the pitch.
//!
//! The allocation is `pitch * storage_rows` bytes, where `storage_rows` is the
//! luma rows plus the chroma rows. Both `YV12` half-pitch planes together fill
//! exactly the rows one `NV12` plane does, so the two formats share the size.
//! The same bytes back a one-byte-per-texel texture of `pitch` by
//! `storage_rows`, in which the byte at offset `o` is texel
//! `(o % pitch, o / pitch)` for every plane.

use mtld3d_types::{D3DFMT_NV12, D3DFMT_YV12};

use crate::{caps::MAX_TEXTURE_DIM, format::linear_row_pitch};

/// Where the planes of one planar YUV surface sit in its allocation.
///
/// Offsets take luma coordinates; the chroma accessors halve them, so the four
/// texels of a 2x2 luma block name one chroma sample.
pub struct PlanarYuvLayout {
    pitch: u32,
    luma_rows: u32,
    chroma_rows: u32,
}

impl PlanarYuvLayout {
    /// Bytes per luma row, the pitch a lock reports.
    #[must_use]
    pub const fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Rows of `pitch` bytes the allocation holds: the luma rows, then the chroma rows.
    #[must_use]
    pub const fn storage_rows(&self) -> u32 {
        self.luma_rows + self.chroma_rows
    }

    /// Size of the whole allocation, every plane included.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.pitch as usize * self.storage_rows() as usize
    }

    /// Offset of the luma byte of texel `(x, y)`.
    #[must_use]
    pub const fn luma_offset(&self, x: usize, y: usize) -> usize {
        y * self.pitch as usize + x
    }

    /// Offset of the `NV12` U byte shared by texel `(x, y)`; its V byte is the next one.
    #[must_use]
    pub const fn nv12_uv_offset(&self, x: usize, y: usize) -> usize {
        self.chroma_base() + (y / 2) * self.pitch as usize + 2 * (x / 2)
    }

    /// Offset of the `YV12` V byte shared by texel `(x, y)`.
    #[must_use]
    pub const fn yv12_v_offset(&self, x: usize, y: usize) -> usize {
        self.chroma_base() + (y / 2) * self.half_pitch() + x / 2
    }

    /// Offset of the `YV12` U byte shared by texel `(x, y)`.
    ///
    /// The U plane starts where the V plane's `chroma_rows` half-pitch rows end.
    #[must_use]
    pub const fn yv12_u_offset(&self, x: usize, y: usize) -> usize {
        self.yv12_v_offset(x, y) + self.chroma_rows as usize * self.half_pitch()
    }

    /// Whether `(x, y)` names a texel of the luma plane.
    ///
    /// The bound on `x` is the pitch, not the width: the layout does not know
    /// the width, and a column in the row padding still addresses bytes of its
    /// own row and of its own chroma sample.
    #[must_use]
    pub const fn contains(&self, x: usize, y: usize) -> bool {
        x < self.pitch as usize && y < self.luma_rows as usize
    }

    /// Offset of the first chroma byte, right after the last luma row.
    const fn chroma_base(&self) -> usize {
        self.pitch as usize * self.luma_rows as usize
    }

    /// Bytes per `YV12` chroma row.
    const fn half_pitch(&self) -> usize {
        self.pitch as usize / 2
    }
}

/// The layout of a `width` x `height` surface of `d3d_format`.
///
/// `None` for any format but `YV12` and `NV12`, for an empty extent, for a
/// `YV12` surface of odd height, and for an extent whose pitch or storage row
/// count is past the largest texture edge, since the allocation is also the
/// extent of the texture that backs the surface. An odd `YV12` height is left
/// out because its U plane has no agreed origin: it follows either
/// `floor(height / 2)` or `ceil(height / 2)` V rows depending on who wrote the
/// surface, and a guess would exchange chroma rows silently. `NV12` has one
/// chroma plane, so an odd height only rounds its row count up.
#[must_use]
pub const fn planar_yuv_layout(
    d3d_format: u32,
    width: u32,
    height: u32,
) -> Option<PlanarYuvLayout> {
    if width == 0 || width > MAX_TEXTURE_DIM {
        return None;
    }
    layout_from_pitch(d3d_format, linear_row_pitch(width, 1), height)
}

/// The layout of a surface of `d3d_format` known by its lock pitch and luma row count.
///
/// The form a reader of locked bytes uses, which has the pitch and not the
/// width. Rejects what [`planar_yuv_layout`] rejects, and an odd pitch, which
/// would split a chroma pair or a half-pitch row across two rows.
#[must_use]
pub fn planar_yuv_layout_from_pitch(
    d3d_format: u32,
    pitch: usize,
    luma_rows: usize,
) -> Option<PlanarYuvLayout> {
    layout_from_pitch(
        d3d_format,
        u32::try_from(pitch).ok()?,
        u32::try_from(luma_rows).ok()?,
    )
}

/// Shared tail of the two constructors: every rule that does not need the width.
const fn layout_from_pitch(d3d_format: u32, pitch: u32, luma_rows: u32) -> Option<PlanarYuvLayout> {
    let even_rows_only = match d3d_format {
        D3DFMT_YV12 => true,
        D3DFMT_NV12 => false,
        _ => return None,
    };
    if pitch == 0 || luma_rows == 0 || pitch > MAX_TEXTURE_DIM || luma_rows > MAX_TEXTURE_DIM {
        return None;
    }
    // An even pitch keeps a chroma pair, and half a pitch, inside its own row.
    if !pitch.is_multiple_of(2) || (even_rows_only && !luma_rows.is_multiple_of(2)) {
        return None;
    }
    let chroma_rows = luma_rows.div_ceil(2);
    if luma_rows + chroma_rows > MAX_TEXTURE_DIM {
        return None;
    }
    Some(PlanarYuvLayout {
        pitch,
        luma_rows,
        chroma_rows,
    })
}

#[cfg(test)]
mod tests;
