//! Rect parsing + validation for `IDirect3DDevice9::StretchRect`.
//!
//! The actual blit dispatch lives in `windows/d3d9` (it needs a Metal
//! handle); only the pure host-testable parts live here.

use mtld3d_types::{D3DFMT_UYVY, D3DFMT_YUY2};

/// Parsed source / destination region for a `StretchRect`.
///
/// Coordinates are clamped against the surface dimensions; an empty
/// region (after clamping) is reported as `None` by `parse`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StretchRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Parse a D3D9 `RECT*` (4 × i32, `left/top/right/bottom`) clamped against `(full_w, full_h)`.
///
/// `NULL` means "full surface". Returns `None` for a degenerate / empty rect
/// — caller treats as `D3DERR_INVALIDCALL`.
///
/// `rect_ptr` is opaque: the wrapper crate owns the unsafe deref since
/// `D3DRECT` lives in `mtld3d-types` (not depended on here). Caller must
/// either pass `None` or a `Some((x1, y1, x2, y2))` already extracted.
#[must_use]
pub fn parse_rect(
    extracted: Option<(i32, i32, i32, i32)>,
    full_w: u32,
    full_h: u32,
) -> Option<StretchRegion> {
    let Some((x1, y1, x2, y2)) = extracted else {
        return Some(StretchRegion {
            x: 0,
            y: 0,
            w: full_w,
            h: full_h,
        });
    };
    let x1 = x1.max(0).cast_unsigned();
    let y1 = y1.max(0).cast_unsigned();
    let x2 = x2.max(0).cast_unsigned();
    let y2 = y2.max(0).cast_unsigned();
    let x = x1.min(full_w);
    let y = y1.min(full_h);
    let right = x2.min(full_w);
    let bottom = y2.min(full_h);
    if right <= x || bottom <= y {
        return None;
    }
    Some(StretchRegion {
        x,
        y,
        w: right - x,
        h: bottom - y,
    })
}

/// Why a `StretchRect` was rejected.
///
/// Carried in `log_once_warn_by!` keys so each distinct mismatch fires
/// exactly once instead of flooding the warn surface with the same line
/// per draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Source / destination differ in pixel format.
    FormatMismatch,
    /// Source / destination region differ in size — scaling is not supported (1:1 only).
    Scaling,
    /// Source surface has no Metal backing.
    ///
    /// E.g. a depth-stencil standalone surface, or a surface type we
    /// don't recognise.
    UnsupportedSource,
    /// Destination surface has no Metal backing.
    UnsupportedDestination,
    /// Source and destination resolve to the same Metal texture handle.
    ///
    /// Metal disallows self-overlap blits.
    SameSurface,
}

impl RejectReason {
    /// Stable u64 key used by `log_once_warn_by!` so each reason fires once.
    ///
    /// Keying on the discriminant keeps the reasons distinct: they are
    /// neither collapsed into a single `log_once_warn!` nor repeated per draw.
    #[must_use]
    pub const fn key(self) -> u64 {
        self as u64
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FormatMismatch => "format mismatch (no conversion path)",
            Self::Scaling => "src and dst dimensions differ (no scaling)",
            Self::UnsupportedSource => "source surface has no Metal backing",
            Self::UnsupportedDestination => "destination surface has no Metal backing",
            Self::SameSurface => "src and dst are the same Metal texture",
        }
    }
}

/// Source-side decode the `StretchRect` render quad applies while sampling.
///
/// Reaches the blit fragment function as a uniform (`src_level.y`), so the one
/// pipeline per destination format serves every source format: mode 0 samples
/// the source as-is, the YUV modes fetch the 4:2:2 macropixel and convert it
/// to RGB. The discriminants are the uniform's values; the MSL in
/// `unix/unix/src/metal/blit.rs` matches on them.
#[repr(u32)]
pub enum BlitDecode {
    /// Sample the source texture as-is (any RGB format).
    None = 0,
    /// `D3DFMT_YUY2`: macropixel bytes `Y0 U Y1 V`, backed by an RG8 texture.
    Yuy2 = 1,
    /// `D3DFMT_UYVY`: macropixel bytes `U Y0 V Y1`, backed by an RG8 texture.
    Uyvy = 2,
}

impl BlitDecode {
    /// The value the blit fragment function reads from its uniform.
    #[must_use]
    pub const fn uniform(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Yuy2 => 1.0,
            Self::Uyvy => 2.0,
        }
    }
}

/// The decode a `StretchRect` source of `d3d_format` needs.
#[must_use]
pub const fn blit_decode(d3d_format: u32) -> BlitDecode {
    match d3d_format {
        D3DFMT_YUY2 => BlitDecode::Yuy2,
        D3DFMT_UYVY => BlitDecode::Uyvy,
        _ => BlitDecode::None,
    }
}

/// Whether `d3d_format` is one of the two packed 4:2:2 YUV formats.
#[must_use]
pub const fn is_packed_yuv(d3d_format: u32) -> bool {
    matches!(d3d_format, D3DFMT_YUY2 | D3DFMT_UYVY)
}

/// Convert one reduced-range `Y'CbCr` sample to 8-bit RGB.
///
/// BT.601 coefficients with the luma scaled from `[16, 235]` and the chroma
/// centred on 128, the convention every desktop driver applies to packed
/// YUV surfaces. Computed in 16.16 fixed point, rounded to nearest and
/// clamped; it agrees with the float version in the blit fragment function
/// (`unix/unix/src/metal/blit.rs`) on every reference sample, keep the two
/// in step.
#[must_use]
pub fn yuv_to_rgb8(luma: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    // 16.16 fixed-point forms of 1.164, 0.063 * 255, 1.596, 0.392, 0.813, 2.017.
    const LUMA_GAIN: i64 = 76_284;
    const LUMA_OFFSET: i64 = 1_052_836;
    const R_FROM_CR: i64 = 104_595;
    const G_FROM_CB: i64 = 25_690;
    const G_FROM_CR: i64 = 53_281;
    const B_FROM_CB: i64 = 132_186;
    let scaled_luma = (((i64::from(luma) << 16) - LUMA_OFFSET) * LUMA_GAIN) >> 16;
    // `cb - 127.5` and `cr - 127.5`, in 16.16.
    let chroma_b = (i64::from(cb) * 2 - 255) << 15;
    let chroma_r = (i64::from(cr) * 2 - 255) << 15;
    let red = scaled_luma + ((chroma_r * R_FROM_CR) >> 16);
    let green = scaled_luma - ((chroma_b * G_FROM_CB) >> 16) - ((chroma_r * G_FROM_CR) >> 16);
    let blue = scaled_luma + ((chroma_b * B_FROM_CB) >> 16);
    let to8 =
        |channel: i64| u8::try_from(((channel + (1 << 15)) >> 16).clamp(0, 255)).unwrap_or(u8::MAX);
    (to8(red), to8(green), to8(blue))
}

/// Decode one pixel of a packed 4:2:2 macropixel (two pixels in four bytes).
///
/// `odd` selects the second pixel's luma sample; both pixels share the
/// chroma pair. `None` for a format that is not packed YUV.
#[must_use]
pub fn decode_packed_yuv(d3d_format: u32, macropixel: [u8; 4], odd: bool) -> Option<(u8, u8, u8)> {
    let [b0, b1, b2, b3] = macropixel;
    let (y, u, v) = match d3d_format {
        D3DFMT_YUY2 => (if odd { b2 } else { b0 }, b1, b3),
        D3DFMT_UYVY => (if odd { b3 } else { b1 }, b0, b2),
        _ => return None,
    };
    Some(yuv_to_rgb8(y, u, v))
}

#[cfg(test)]
mod tests {
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
}
