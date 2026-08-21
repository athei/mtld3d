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
        RejectReason::SameSurface,
    ]
    .iter()
    .map(|r| r.key())
    .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(keys.len(), sorted.len());
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
