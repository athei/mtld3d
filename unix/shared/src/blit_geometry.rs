//! Row geometry shared by both directions of a Metal texture blit.
//!
//! Metal measures a texture copy in *block* rows, not pixel rows. For a
//! block-compressed level one row of `bytes_per_row` covers `block_height`
//! pixel rows, so a slice is `ceil(height / block_height)` rows long, not
//! `height`. An uncompressed format has a block height of 1 and the same
//! formula collapses to the familiar `bytes_per_row * height`.
//!
//! Both directions derive the value here: the PE-side upload jobs building a
//! `CopyBufferToTexture` blit command, and the unix-side `copyFromTexture:
//! toBuffer:` readback. The unix side has no format table, so the block height
//! travels to it on the readback thunk's parameters.

/// Number of block rows a `height`-pixel region covers.
///
/// `block_height` is the source format's block height: 1 for an uncompressed
/// format, 4 for the BC family. Zero is read as 1, so a caller that leaves the
/// field at its default gets the uncompressed answer rather than a division
/// trap.
#[must_use]
pub const fn block_rows(height: u32, block_height: u32) -> u32 {
    if block_height <= 1 {
        height
    } else {
        height.div_ceil(block_height)
    }
}

/// Byte size of one image slice, for Metal's `bytesPerImage` argument.
///
/// `bytes_per_row` is the stride of a single block row, already in the layout
/// the blit reads (padded strides included), and `height` is the region height
/// in pixels.
#[must_use]
pub const fn bytes_per_image(bytes_per_row: u32, height: u32, block_height: u32) -> u32 {
    bytes_per_row.saturating_mul(block_rows(height, block_height))
}

#[cfg(test)]
mod tests;
