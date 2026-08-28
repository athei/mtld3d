//! Unit tests for the host-testable half of `StretchRect`.
//!
//! `parse_rect` is checked against its clamping contract: a null rect covers the whole
//! surface, out-of-bounds corners clamp to the surface instead of wrapping, and a rect
//! that is empty after clamping reports `None` so the caller can fail the call. A
//! separate check keeps every `RejectReason` key distinct, which is what makes the
//! once-per-reason warn fire once per reason rather than collapsing to a single line.
//!
//! `same_surface_route` is pinned against the four shapes a within-one-surface copy
//! takes: disjoint rects that the blit encoder can copy in place, overlapping rects and
//! scaled rects that have to stage through a scratch texture, and an identical pair that
//! writes each texel its own value. Two cube faces of one texture are disjoint whatever
//! their rects say, so the face pair is pinned alongside the mip pair.
//!
//! The packed-YUV cases pin the source decode: which `BlitDecode` a format selects and
//! the discriminants the fragment shader matches on, the fixed-point `yuv_to_rgb8`
//! against reference samples in both the full-range and reduced-range conventions, and
//! the macropixel byte order that separates `YUY2` from `UYVY`. The conversion has a
//! float twin in the blit shader, so a change here that is not mirrored there shows up
//! as a colour shift no other test would catch.

use super::*;

#[test]
fn null_rect_is_full_surface() {
    assert_eq!(
        parse_rect(None, 100, 200),
        Some(StretchRegion {
            x: 0,
            y: 0,
            w: 100,
            h: 200
        })
    );
}

#[test]
fn rect_clamped_against_surface() {
    assert_eq!(
        parse_rect(Some((-10, -20, 50, 60)), 100, 100),
        Some(StretchRegion {
            x: 0,
            y: 0,
            w: 50,
            h: 60
        })
    );
    assert_eq!(
        parse_rect(Some((10, 20, 200, 300)), 100, 100),
        Some(StretchRegion {
            x: 10,
            y: 20,
            w: 90,
            h: 80
        })
    );
}

#[test]
fn empty_rect_returns_none() {
    assert_eq!(parse_rect(Some((50, 50, 50, 50)), 100, 100), None);
    assert_eq!(parse_rect(Some((100, 0, 200, 100)), 100, 100), None);
}

#[test]
fn reject_keys_are_distinct() {
    let keys: Vec<u64> = [
        RejectReason::FormatMismatch,
        RejectReason::Scaling,
        RejectReason::UnsupportedSource,
        RejectReason::UnsupportedDestination,
    ]
    .iter()
    .map(|r| r.key())
    .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(keys.len(), sorted.len());
}

const fn region(x: u32, y: u32, w: u32, h: u32) -> StretchRegion {
    StretchRegion { x, y, w, h }
}

#[test]
fn disjoint_same_surface_rects_copy_in_place() {
    // Side by side, corner to corner, and touching edges: none of these share
    // a texel, so the blit encoder can copy inside the one texture.
    for (src, dst) in [
        (region(0, 0, 16, 16), region(16, 0, 16, 16)),
        (region(0, 0, 16, 16), region(0, 16, 16, 16)),
        (region(0, 0, 16, 16), region(64, 64, 16, 16)),
        (region(32, 32, 8, 8), region(24, 24, 8, 8)),
    ] {
        assert_eq!(
            same_surface_route(src, dst, 0, 0, 0, 0),
            SameSurfaceRoute::Direct,
            "{src:?} -> {dst:?}"
        );
    }
    // Two mip levels are different texels whatever the rects say.
    assert_eq!(
        same_surface_route(region(0, 0, 16, 16), region(0, 0, 16, 16), 0, 1, 0, 0),
        SameSurfaceRoute::Direct
    );
}

#[test]
fn same_surface_rects_on_two_cube_faces_copy_in_place() {
    // Two faces are two slices of one texture, so the same rect on each names
    // different texels: the copy is real and the blit encoder can do it in
    // place, whether or not the rects would have overlapped on one face.
    for (src, dst) in [
        (region(0, 0, 16, 16), region(0, 0, 16, 16)),
        (region(0, 0, 16, 16), region(8, 8, 16, 16)),
        (region(0, 0, 16, 16), region(64, 64, 16, 16)),
    ] {
        assert_eq!(
            same_surface_route(src, dst, 0, 0, 1, 3),
            SameSurfaceRoute::Direct,
            "{src:?} -> {dst:?}"
        );
    }
    // A size change still stages through the scratch: the render quad cannot
    // sample the texture it draws into, faces or no faces.
    assert_eq!(
        same_surface_route(region(0, 0, 32, 32), region(0, 0, 16, 16), 0, 1, 1, 3),
        SameSurfaceRoute::Scratch
    );
    // One face copied onto itself is still the no-op.
    assert_eq!(
        same_surface_route(region(4, 8, 16, 16), region(4, 8, 16, 16), 0, 0, 3, 3),
        SameSurfaceRoute::Skip
    );
}

#[test]
fn overlapping_same_surface_rects_need_a_scratch() {
    for (src, dst) in [
        (region(0, 0, 16, 16), region(8, 0, 16, 16)),
        (region(0, 0, 16, 16), region(0, 8, 16, 16)),
        (region(8, 8, 16, 16), region(0, 0, 16, 16)),
        (region(0, 0, 32, 32), region(8, 8, 32, 32)),
    ] {
        assert_eq!(
            same_surface_route(src, dst, 0, 0, 0, 0),
            SameSurfaceRoute::Scratch,
            "{src:?} -> {dst:?}"
        );
    }
}

#[test]
fn scaled_same_surface_rects_need_a_scratch() {
    // The render quad cannot sample the texture it draws into, so a size
    // change stages through a scratch even when the rects are disjoint and
    // even across mip levels.
    assert_eq!(
        same_surface_route(region(0, 0, 32, 32), region(64, 64, 16, 16), 0, 0, 0, 0),
        SameSurfaceRoute::Scratch
    );
    assert_eq!(
        same_surface_route(region(0, 0, 16, 16), region(0, 0, 32, 16), 0, 1, 0, 0),
        SameSurfaceRoute::Scratch
    );
}

#[test]
fn identical_same_surface_rects_are_a_no_op() {
    assert_eq!(
        same_surface_route(region(4, 8, 16, 16), region(4, 8, 16, 16), 2, 2, 0, 0),
        SameSurfaceRoute::Skip
    );
    // The same rect at two levels is a real copy, not the no-op.
    assert_eq!(
        same_surface_route(region(4, 8, 16, 16), region(4, 8, 16, 16), 0, 2, 0, 0),
        SameSurfaceRoute::Direct
    );
}

#[test]
fn blit_decode_follows_the_source_format() {
    assert!(matches!(blit_decode(D3DFMT_YUY2), BlitDecode::Yuy2));
    assert!(matches!(blit_decode(D3DFMT_UYVY), BlitDecode::Uyvy));
    assert!(matches!(
        blit_decode(mtld3d_types::D3DFMT_X8R8G8B8),
        BlitDecode::None
    ));
    // The uniform values are the discriminants the MSL matches on.
    assert_eq!(BlitDecode::None.uniform().to_bits(), 0.0f32.to_bits());
    assert_eq!(BlitDecode::Yuy2.uniform().to_bits(), 1.0f32.to_bits());
    assert_eq!(BlitDecode::Uyvy.uniform().to_bits(), 2.0f32.to_bits());
    assert!(is_packed_yuv(D3DFMT_YUY2) && is_packed_yuv(D3DFMT_UYVY));
    assert!(!is_packed_yuv(mtld3d_types::D3DFMT_R5G6B5));
}

#[test]
fn yuv_to_rgb8_matches_wine_yuv_layout_table() {
    // (y, u, v) -> (rgb_full, rgb_reduced): the reference samples
    // desktop drivers are held to, each accepted within 1 in either
    // convention (full-range or reduced-range luma).
    let rows: [(u8, u8, u8, u32, u32); 16] = [
        (0x10, 0x80, 0x80, 0x00_0000, 0x10_1010),
        (0xeb, 0x80, 0x80, 0xff_ffff, 0xeb_ebeb),
        (0x51, 0x5a, 0xf0, 0xff_0000, 0xee_0e0e),
        (0x91, 0x36, 0x22, 0x00_ff01, 0x0d_ee0e),
        (0x29, 0xf0, 0x6e, 0x00_00ff, 0x10_0fef),
        (0x7e, 0x80, 0x80, 0x80_8080, 0x7e_7e7e),
        (0x00, 0x80, 0x80, 0x00_0000, 0x00_0000),
        (0xff, 0x80, 0x80, 0xff_ffff, 0xff_ffff),
        (0x00, 0x00, 0x00, 0x00_8800, 0x00_8800),
        (0xff, 0x00, 0x00, 0x4a_ff14, 0x4c_ff1c),
        (0x00, 0xff, 0x00, 0x00_24ee, 0x00_30e1),
        (0x00, 0x00, 0xff, 0xb8_0000, 0xb2_0000),
        (0xff, 0xff, 0x00, 0x4a_ffff, 0x4c_ffff),
        (0xff, 0x00, 0xff, 0xff_e114, 0xff_d01c),
        (0x00, 0xff, 0xff, 0xb8_00ee, 0xb2_00e1),
        (0xff, 0xff, 0xff, 0xff_7dff, 0xff_78ff),
    ];
    let close = |got: u32, expected: u32| {
        [16, 8, 0].iter().all(|&shift| {
            let a = i32::try_from((got >> shift) & 0xff).unwrap_or(0);
            let e = i32::try_from((expected >> shift) & 0xff).unwrap_or(0);
            (a - e).abs() <= 1
        })
    };
    for (y, u, v, full, reduced) in rows {
        let (r, g, b) = yuv_to_rgb8(y, u, v);
        let got = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
        assert!(
            close(got, full) || close(got, reduced),
            "yuv ({y:#x}, {u:#x}, {v:#x}) -> {got:#08x}, expected {full:#08x} or {reduced:#08x}"
        );
    }
}

#[test]
fn packed_yuv_macropixel_byte_order() {
    // The DWORD 0x4cff4c54 read as UYVY is (U 0x54, Y0 0x4c, V 0xff,
    // Y1 0x4c): pure red for both pixels. Read as YUY2 it is (Y0 0x54,
    // U 0x4c, Y1 0xff, V 0x4c): the two pixels differ (a dark green and
    // a light green, reference 0x0b8b00 / 0xb6ffa3 within 18).
    let mp = 0x4cff_4c54u32.to_le_bytes();
    assert_eq!(
        decode_packed_yuv(D3DFMT_UYVY, mp, false),
        Some((0xff, 0x00, 0x00))
    );
    assert_eq!(
        decode_packed_yuv(D3DFMT_UYVY, mp, true),
        Some((0xff, 0x00, 0x00))
    );
    let (r, g, b) = decode_packed_yuv(D3DFMT_YUY2, mp, false).unwrap();
    assert!(r <= 0x0b + 18 && (0x8b - 18..=0x8b + 18).contains(&g) && b <= 18);
    let (r, g, b) = decode_packed_yuv(D3DFMT_YUY2, mp, true).unwrap();
    assert!(
        (0xb6 - 18..=0xb6 + 18).contains(&r)
            && g >= 0xff - 18
            && (0xa3 - 18..=0xa3 + 18).contains(&b)
    );
    assert_eq!(
        decode_packed_yuv(mtld3d_types::D3DFMT_X8R8G8B8, mp, false),
        None
    );
}
