use mtld3d_types::{D3DFMT_NV12, D3DFMT_UYVY, D3DFMT_X8R8G8B8, D3DFMT_YUY2, D3DFMT_YV12};

use super::*;
use crate::caps::MAX_TEXTURE_DIM;

#[test]
fn an_even_surface_locks_at_its_width_with_the_chroma_after_the_luma_rows() {
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let layout = planar_yuv_layout(format, 20, 16).expect("20x16 is a valid planar extent");
        assert_eq!(layout.pitch(), 20);
        assert_eq!(layout.storage_rows(), 24);
        assert_eq!(layout.total_bytes(), 480);
        assert_eq!(layout.luma_offset(0, 0), 0);
        assert_eq!(layout.luma_offset(19, 15), 319);
    }
    let yv12 = planar_yuv_layout(D3DFMT_YV12, 20, 16).unwrap();
    assert_eq!(yv12.yv12_v_offset(0, 0), 320);
    assert_eq!(yv12.yv12_u_offset(0, 0), 400);
    assert_eq!(yv12.yv12_u_offset(19, 15), 479);
    let nv12 = planar_yuv_layout(D3DFMT_NV12, 20, 16).unwrap();
    assert_eq!(nv12.nv12_uv_offset(0, 0), 320);
    // U of the last block; its V is the allocation's last byte.
    assert_eq!(nv12.nv12_uv_offset(19, 15), 478);
}

#[test]
fn chroma_rows_stride_half_the_pitch_not_half_the_width() {
    // Width 22 rounds to a pitch of 24, so a YV12 chroma row is 12 bytes while
    // half the width is 11: the second chroma row separates the two readings.
    let layout = planar_yuv_layout(D3DFMT_YV12, 22, 16).unwrap();
    assert_eq!(layout.pitch(), 24);
    assert_eq!(layout.total_bytes(), 24 * 24);
    assert_eq!(layout.yv12_v_offset(0, 2), 24 * 16 + 12);
    assert_ne!(layout.yv12_v_offset(0, 2), 24 * 16 + 11);
    assert_eq!(layout.yv12_u_offset(0, 0), 24 * 16 + 8 * 12);
    assert_eq!(layout.yv12_u_offset(21, 15), 24 * 16 + 8 * 12 + 7 * 12 + 10);
    // NV12 chroma rows stride the full pitch.
    let nv12 = planar_yuv_layout(D3DFMT_NV12, 22, 16).unwrap();
    assert_eq!(nv12.nv12_uv_offset(0, 2), 24 * 16 + 24);
    assert_eq!(nv12.nv12_uv_offset(21, 15), 24 * 16 + 7 * 24 + 20);
}

#[test]
fn an_odd_width_keeps_every_plane_pitch_relative() {
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let layout = planar_yuv_layout(format, 21, 16).unwrap();
        assert_eq!(layout.pitch(), 24);
        assert_eq!(layout.storage_rows(), 24);
    }
    // The odd last column shares the chroma sample of column 20.
    let yv12 = planar_yuv_layout(D3DFMT_YV12, 21, 16).unwrap();
    assert_eq!(yv12.yv12_v_offset(20, 0), 24 * 16 + 10);
    assert_eq!(yv12.yv12_u_offset(20, 0), 24 * 16 + 8 * 12 + 10);
    let nv12 = planar_yuv_layout(D3DFMT_NV12, 21, 16).unwrap();
    assert_eq!(nv12.nv12_uv_offset(20, 0), 24 * 16 + 20);
}

#[test]
fn an_odd_nv12_height_rounds_its_chroma_rows_up() {
    let layout = planar_yuv_layout(D3DFMT_NV12, 20, 15).unwrap();
    assert_eq!(layout.storage_rows(), 15 + 8);
    assert_eq!(layout.total_bytes(), 20 * 23);
    // The last luma row has a chroma row of its own.
    assert_eq!(layout.nv12_uv_offset(18, 14), 20 * 15 + 7 * 20 + 18);
}

#[test]
fn an_odd_yv12_height_has_no_layout() {
    assert!(planar_yuv_layout(D3DFMT_YV12, 20, 15).is_none());
    assert!(planar_yuv_layout_from_pitch(D3DFMT_YV12, 20, 15).is_none());
    assert!(planar_yuv_layout(D3DFMT_YV12, 20, 14).is_some());
}

#[test]
fn other_formats_and_empty_extents_have_no_layout() {
    for format in [D3DFMT_YUY2, D3DFMT_UYVY, D3DFMT_X8R8G8B8, 0] {
        assert!(planar_yuv_layout(format, 20, 16).is_none(), "{format:#x}");
        assert!(
            planar_yuv_layout_from_pitch(format, 20, 16).is_none(),
            "{format:#x}"
        );
    }
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        assert!(planar_yuv_layout(format, 0, 16).is_none());
        assert!(planar_yuv_layout(format, 20, 0).is_none());
        assert!(planar_yuv_layout_from_pitch(format, 0, 16).is_none());
        assert!(planar_yuv_layout_from_pitch(format, 20, 0).is_none());
    }
    // Half a YV12 pitch has to be a whole number of bytes.
    assert!(planar_yuv_layout_from_pitch(D3DFMT_YV12, 21, 16).is_none());
}

#[test]
fn an_extent_past_the_texture_limit_has_no_layout() {
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        assert!(planar_yuv_layout(format, MAX_TEXTURE_DIM, 16).is_some());
        assert!(planar_yuv_layout(format, MAX_TEXTURE_DIM + 1, 16).is_none());
        // 10922 luma rows plus 5461 chroma rows is 16383 storage rows; two
        // more luma rows pass the limit.
        assert!(planar_yuv_layout(format, 16, 10922).is_some());
        assert!(planar_yuv_layout(format, 16, 10924).is_none());
        assert!(planar_yuv_layout(format, 16, u32::MAX - 1).is_none());
    }
}

#[test]
fn the_highest_offset_of_every_plane_is_inside_the_allocation() {
    for (width, height) in [(20, 16), (22, 16), (21, 16), (1, 2), (5, 4), (4096, 2048)] {
        let yv12 = planar_yuv_layout(D3DFMT_YV12, width, height).unwrap();
        let (x, y) = (width as usize - 1, height as usize - 1);
        assert!(yv12.luma_offset(x, y) < yv12.total_bytes());
        assert!(yv12.yv12_v_offset(x, y) < yv12.yv12_u_offset(0, 0));
        assert!(yv12.yv12_u_offset(x, y) < yv12.total_bytes());
    }
    for (width, height) in [(20, 16), (22, 16), (21, 15), (1, 1), (5, 3), (4096, 2047)] {
        let nv12 = planar_yuv_layout(D3DFMT_NV12, width, height).unwrap();
        let (x, y) = (width as usize - 1, height as usize - 1);
        assert!(nv12.luma_offset(x, y) < nv12.nv12_uv_offset(0, 0));
        // The V byte follows the U byte the accessor names.
        assert!(nv12.nv12_uv_offset(x, y) + 1 < nv12.total_bytes());
    }
}

#[test]
fn a_layout_rebuilt_from_its_pitch_addresses_the_same_bytes() {
    let from_extent = planar_yuv_layout(D3DFMT_YV12, 22, 16).unwrap();
    let from_pitch = planar_yuv_layout_from_pitch(D3DFMT_YV12, 24, 16).unwrap();
    assert_eq!(from_pitch.total_bytes(), from_extent.total_bytes());
    assert_eq!(
        from_pitch.yv12_u_offset(21, 15),
        from_extent.yv12_u_offset(21, 15)
    );
}
