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
//!
//! The planar cases pin the two 4:2:0 selectors, the plane order that separates `YV12`
//! from `NV12` with a sample whose U and V differ, the pitch-relative chroma addressing
//! on a surface whose pitch is wider than its width, and the route a planar endpoint
//! takes through `StretchRect`.

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
        RejectReason::PlanarDestination,
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

#[test]
fn blit_decode_selects_the_planar_formats() {
    use mtld3d_types::{D3DFMT_NV12, D3DFMT_YV12};

    assert!(matches!(blit_decode(D3DFMT_YV12), BlitDecode::Yv12));
    assert!(matches!(blit_decode(D3DFMT_NV12), BlitDecode::Nv12));
    // The uniform values are the discriminants the MSL matches on.
    assert_eq!(BlitDecode::Yv12.uniform().to_bits(), 3.0f32.to_bits());
    assert_eq!(BlitDecode::Nv12.uniform().to_bits(), 4.0f32.to_bits());
    assert!(is_planar_yuv(D3DFMT_YV12) && is_planar_yuv(D3DFMT_NV12));
    assert!(!is_planar_yuv(D3DFMT_YUY2) && !is_planar_yuv(D3DFMT_UYVY));
    assert!(!is_packed_yuv(D3DFMT_YV12) && !is_packed_yuv(D3DFMT_NV12));
}

/// A planar surface of `pitch` x `luma_rows` holding one colour.
///
/// The planes are written through the layout the lock exposes: `YV12` stores V
/// ahead of U, `NV12` interleaves U then V.
fn planar_fill(d3d_format: u32, pitch: usize, luma_rows: usize, yuv: (u8, u8, u8)) -> Vec<u8> {
    use mtld3d_types::D3DFMT_YV12;

    let chroma_rows = luma_rows.div_ceil(2);
    let mut bytes = vec![0u8; pitch * (luma_rows + chroma_rows)];
    bytes[..pitch * luma_rows].fill(yuv.0);
    let chroma = &mut bytes[pitch * luma_rows..];
    if d3d_format == D3DFMT_YV12 {
        let plane = pitch / 2 * chroma_rows;
        chroma[..plane].fill(yuv.2);
        chroma[plane..2 * plane].fill(yuv.1);
    } else {
        for pair in chroma.as_chunks_mut::<2>().0 {
            *pair = [yuv.1, yuv.2];
        }
    }
    bytes
}

#[test]
fn planar_decode_reads_v_and_u_from_their_own_planes() {
    use mtld3d_types::{D3DFMT_NV12, D3DFMT_YV12};

    // (0x51, 0x5a, 0xf0) is pure red; with U and V exchanged it is blue-ish,
    // so a plane or interleave exchange cannot pass.
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let red = planar_fill(format, 20, 16, (0x51, 0x5a, 0xf0));
        for (x, y) in [(0, 0), (19, 0), (0, 15), (19, 15), (7, 9)] {
            assert_eq!(
                decode_planar_yuv(format, &red, 20, 16, x, y),
                Some((0xff, 0x00, 0x00)),
                "{format:#x} at ({x}, {y})"
            );
        }
        let exchanged = planar_fill(format, 20, 16, (0x51, 0xf0, 0x5a));
        let (r, _, b) = decode_planar_yuv(format, &exchanged, 20, 16, 3, 3).unwrap();
        assert!(r < 0x40 && b > 0xc0, "{format:#x}: ({r:#x}, {b:#x})");
    }
}

#[test]
fn planar_decode_matches_the_reference_value_table() {
    use mtld3d_types::{D3DFMT_NV12, D3DFMT_YV12};

    let rows: [(u8, u8, u8, u32, u32); 17] = [
        (0x40, 0x40, 0x40, 0x00_8400, 0x00_8400),
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
    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        for (y, u, v, full, reduced) in rows {
            let bytes = planar_fill(format, 20, 16, (y, u, v));
            let (r, g, b) = decode_planar_yuv(format, &bytes, 20, 16, 12, 10).unwrap();
            let got = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
            assert!(
                close(got, full) || close(got, reduced),
                "{format:#x} ({y:#x}, {u:#x}, {v:#x}) -> {got:#08x}, expected {full:#08x} or \
                 {reduced:#08x}"
            );
        }
    }
}

#[test]
fn planar_chroma_is_addressed_by_the_pitch() {
    use mtld3d_types::D3DFMT_YV12;

    // A 22-wide surface locks at a pitch of 24. Its second V row starts 12
    // bytes into the plane; a reader striding half the width would start at
    // 11 and take the grey byte planted there, which is padding of the first
    // V row.
    let (pitch, luma_rows) = (24usize, 16usize);
    let mut bytes = planar_fill(D3DFMT_YV12, pitch, luma_rows, (0x51, 0x5a, 0xf0));
    let v_plane = pitch * luma_rows;
    bytes[v_plane + 11] = 0x80;
    assert_eq!(
        decode_planar_yuv(D3DFMT_YV12, &bytes, pitch, luma_rows, 0, 2),
        Some((0xff, 0x00, 0x00))
    );
    // The same grey at the real start of the second V row is what rows 2
    // and 3 decode with.
    bytes[v_plane + 12] = 0x80;
    let (r, _, _) = decode_planar_yuv(D3DFMT_YV12, &bytes, pitch, luma_rows, 1, 3).unwrap();
    assert!(r < 0xa0, "the second V row was not read at half the pitch");
}

#[test]
fn planar_decode_rejects_what_it_cannot_address() {
    use mtld3d_types::{D3DFMT_NV12, D3DFMT_YV12};

    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        let bytes = planar_fill(format, 20, 16, (0x51, 0x5a, 0xf0));
        // One byte short of the last chroma sample.
        let truncated = &bytes[..bytes.len() - 1];
        assert_eq!(decode_planar_yuv(format, truncated, 20, 16, 19, 15), None);
        assert!(decode_planar_yuv(format, truncated, 20, 16, 0, 0).is_some());
        // Outside the luma plane.
        assert_eq!(decode_planar_yuv(format, &bytes, 20, 16, 20, 0), None);
        assert_eq!(decode_planar_yuv(format, &bytes, 20, 16, 0, 16), None);
        assert_eq!(decode_planar_yuv(format, &bytes, 0, 16, 0, 0), None);
    }
    let bytes = planar_fill(D3DFMT_YUY2, 20, 16, (0x51, 0x5a, 0xf0));
    assert_eq!(decode_planar_yuv(D3DFMT_YUY2, &bytes, 20, 16, 0, 0), None);
    // An odd YV12 height has no defined U-plane origin.
    let bytes = planar_fill(D3DFMT_YV12, 20, 15, (0x51, 0x5a, 0xf0));
    assert_eq!(decode_planar_yuv(D3DFMT_YV12, &bytes, 20, 15, 0, 0), None);
}

#[test]
fn a_planar_endpoint_routes_by_destination_class() {
    use mtld3d_types::{D3DFMT_A8R8G8B8, D3DFMT_NV12, D3DFMT_V8U8, D3DFMT_X8R8G8B8, D3DFMT_YV12};

    for format in [D3DFMT_YV12, D3DFMT_NV12] {
        // Into a render target the quad decodes, scaled or not.
        assert_eq!(
            planar_stretch_route(format, D3DFMT_X8R8G8B8, true, false),
            PlanarStretch::RenderQuad
        );
        assert_eq!(
            planar_stretch_route(format, D3DFMT_A8R8G8B8, true, true),
            PlanarStretch::RenderQuad
        );
        // Into an offscreen plain only the 1:1 CPU conversion exists.
        assert_eq!(
            planar_stretch_route(format, D3DFMT_X8R8G8B8, false, false),
            PlanarStretch::CpuConvert
        );
        assert_eq!(
            planar_stretch_route(format, D3DFMT_X8R8G8B8, false, true),
            PlanarStretch::Reject(RejectReason::Scaling)
        );
        assert_eq!(
            planar_stretch_route(format, D3DFMT_V8U8, false, false),
            PlanarStretch::Reject(RejectReason::FormatMismatch)
        );
        // A planar destination is never written, whatever the source is.
        for dst_is_render_target in [false, true] {
            assert_eq!(
                planar_stretch_route(D3DFMT_X8R8G8B8, format, dst_is_render_target, false),
                PlanarStretch::Reject(RejectReason::PlanarDestination)
            );
            assert_eq!(
                planar_stretch_route(format, format, dst_is_render_target, false),
                PlanarStretch::Reject(RejectReason::PlanarDestination)
            );
        }
    }
    assert_eq!(
        planar_stretch_route(D3DFMT_YUY2, D3DFMT_X8R8G8B8, true, false),
        PlanarStretch::NotPlanar
    );
    assert_eq!(
        planar_stretch_route(D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, false, true),
        PlanarStretch::NotPlanar
    );
}
