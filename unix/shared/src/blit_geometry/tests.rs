//! Unit tests for the block-row geometry a Metal texture blit is measured in.
//!
//! The cases that matter are the ones where a pixel-row count and a block-row count
//! disagree: a 4x4-block format at a height that is a multiple of the block height, at
//! a height that is not, and at a height below one block. An uncompressed format
//! (block height 1, and the degenerate 0) has to keep answering `bytes_per_row *
//! height`, since every existing caller relies on that.

use super::{block_rows, bytes_per_image};

/// An uncompressed format counts pixel rows, and an unset block height reads as one.
#[test]
fn uncompressed_counts_pixel_rows() {
    assert_eq!(block_rows(0, 1), 0);
    assert_eq!(block_rows(1, 1), 1);
    assert_eq!(block_rows(17, 1), 17);
    assert_eq!(block_rows(256, 0), 256);

    assert_eq!(bytes_per_image(1024, 256, 1), 1024 * 256);
    assert_eq!(bytes_per_image(4, 1, 1), 4);
    assert_eq!(bytes_per_image(1024, 256, 0), 1024 * 256);
}

/// A 4x4-block format counts block rows, rounding a partial row up.
#[test]
fn compressed_counts_block_rows() {
    assert_eq!(block_rows(4, 4), 1);
    assert_eq!(block_rows(8, 4), 2);
    assert_eq!(block_rows(256, 4), 64);

    // A 256x256 BC1 level: 64 blocks across at 8 bytes each is a 512-byte
    // block row, and 64 block rows make the level 32 KiB, not 128 KiB.
    assert_eq!(bytes_per_image(512, 256, 4), 512 * 64);
    // A single 4x4 BC1 block is the whole slice.
    assert_eq!(bytes_per_image(8, 4, 4), 8);
}

/// A height that is not a multiple of the block height rounds up to a whole block row.
#[test]
fn partial_block_row_rounds_up() {
    assert_eq!(block_rows(1, 4), 1);
    assert_eq!(block_rows(2, 4), 1);
    assert_eq!(block_rows(5, 4), 2);
    assert_eq!(block_rows(7, 4), 2);
    assert_eq!(block_rows(9, 4), 3);

    // The bottom of a BC1 mip chain: 2x2 and 1x1 levels each still occupy one
    // 8-byte block.
    assert_eq!(bytes_per_image(8, 2, 4), 8);
    assert_eq!(bytes_per_image(8, 1, 4), 8);
    // A 16x6 BC1 sub-rect spans two block rows of 4 blocks.
    assert_eq!(bytes_per_image(32, 6, 4), 64);
}

/// An overflowing product saturates rather than wrapping to a short slice.
#[test]
fn overflow_saturates() {
    assert_eq!(bytes_per_image(u32::MAX, 8, 4), u32::MAX);
}
