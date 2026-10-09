//! Rect parsing + validation for `IDirect3DDevice9::StretchRect`.
//!
//! The actual blit dispatch lives in `windows/d3d9` (it needs a Metal
//! handle); only the pure host-testable parts live here: the rect check, the
//! routes a same-texture pair and a planar YUV endpoint take, the source
//! decode selector, and the CPU twins of the YUV decodes the blit fragment
//! function runs.

use mtld3d_types::{
    D3DFMT_NV12, D3DFMT_UYVY, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DFMT_YV12,
};

use crate::{
    encoder_data::StretchSurfaceFlags, format::FormatMapping, pixel_convert::can_convert,
    planar_yuv::planar_yuv_layout_from_pitch,
};

/// Parsed source / destination region for a `StretchRect`.
///
/// Always inside its surface and non-empty: `parse_rect` refuses anything
/// else rather than clamping it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StretchRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Parse a D3D9 `RECT*` (4 × i32, `left/top/right/bottom`) against a `(full_w, full_h)` surface.
///
/// `NULL` means "full surface". D3D9 refuses a rect that is empty or
/// inverted, and one that leaves the surface: a negative edge, or a far edge
/// past the surface's extent. Nothing is clamped, since a clamped source rect
/// silently becomes a scale and a clamped destination rect a shifted copy.
///
/// `rect_ptr` is opaque: the wrapper crate owns the unsafe deref since
/// `D3DRECT` lives in `mtld3d-types` (not depended on here). Caller must
/// either pass `None` or a `Some((x1, y1, x2, y2))` already extracted.
///
/// # Errors
///
/// [`RejectReason::EmptyRect`] for an empty or inverted rect,
/// [`RejectReason::RectOutsideSurface`] for one that leaves the surface.
pub const fn parse_rect(
    extracted: Option<(i32, i32, i32, i32)>,
    full_w: u32,
    full_h: u32,
) -> Result<StretchRegion, RejectReason> {
    let Some((x1, y1, x2, y2)) = extracted else {
        return Ok(StretchRegion {
            x: 0,
            y: 0,
            w: full_w,
            h: full_h,
        });
    };
    if x2 <= x1 || y2 <= y1 {
        return Err(RejectReason::EmptyRect);
    }
    if x1 < 0 || y1 < 0 || x2.cast_unsigned() > full_w || y2.cast_unsigned() > full_h {
        return Err(RejectReason::RectOutsideSurface);
    }
    Ok(StretchRegion {
        x: x1.cast_unsigned(),
        y: y1.cast_unsigned(),
        w: x2.abs_diff(x1),
        h: y2.abs_diff(y1),
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
    /// Destination is a planar YUV surface, which nothing encodes into.
    PlanarDestination,
    /// A source or destination rect is empty or inverted.
    EmptyRect,
    /// A source or destination rect has an edge outside its surface.
    RectOutsideSurface,
    /// The destination is neither a render target nor a default-pool offscreen plain.
    DestinationClass,
    /// An offscreen-plain destination with a source that is no offscreen plain.
    SourceIntoOffscreenPlain,
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
            Self::PlanarDestination => "destination is a planar YUV surface (decode only)",
            Self::EmptyRect => "a source or destination rect is empty or inverted",
            Self::RectOutsideSurface => "a source or destination rect leaves its surface",
            Self::DestinationClass => {
                "the destination is neither a render target nor an offscreen-plain surface"
            }
            Self::SourceIntoOffscreenPlain => {
                "an offscreen-plain destination takes only an offscreen-plain source"
            }
        }
    }
}

/// The surface-class rule a colour `StretchRect` pair breaks on a plain device, if any.
///
/// D3D9 copies, with or without a stretch, into a render target (a standalone
/// one, the back buffer, or a level or face of a render-target texture) from
/// any default-pool colour surface, and into a default-pool offscreen plain
/// only from another offscreen plain. An ordinary texture level or cube face
/// is never a destination. `None` means the classes allow the pair; pools,
/// formats, rects and depth-stencil pairs are judged elsewhere.
#[must_use]
pub const fn class_reject(
    src: StretchSurfaceFlags,
    dst: StretchSurfaceFlags,
) -> Option<RejectReason> {
    if dst.contains(StretchSurfaceFlags::IS_RENDER_TARGET) {
        None
    } else if !dst.contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT) {
        Some(RejectReason::DestinationClass)
    } else if src.contains(StretchSurfaceFlags::IS_OFFSCREEN_PLAIN_DEFAULT) {
        None
    } else {
        Some(RejectReason::SourceIntoOffscreenPlain)
    }
}

/// Whether an extended device accepts a `StretchRect` between surfaces of no eligible class.
///
/// A plain device copies only into a render target, or between two default-pool
/// offscreen plains. An extended device also copies between any two
/// default-pool surfaces, a texture level that is no render target included,
/// as long as the call is a plain copy: no rects at all (a rect covering the
/// whole surface does not count), one format, one size. `src` and `dst` are
/// each surface's `(format, width, height)`.
#[must_use]
pub fn extended_whole_surface_copy(
    extended: bool,
    rects_given: bool,
    src: (u32, u32, u32),
    dst: (u32, u32, u32),
) -> bool {
    extended && !rects_given && src == dst
}

/// How a `StretchRect` whose two endpoints are one Metal texture is carried out.
///
/// D3D9 performs a copy between two rectangles of the same surface; only an
/// overlapping pair is undefined. Metal's blit encoder copies within a single
/// texture as long as the two regions do not overlap, and the render quad
/// cannot sample the texture it draws into at all, so the cases split three
/// ways.
#[derive(Debug, PartialEq, Eq)]
pub enum SameSurfaceRoute {
    /// The two regions name the same texels, so the copy writes what is there.
    Skip,
    /// Disjoint regions of equal size: one blit inside the texture.
    Direct,
    /// Overlapping regions, or a size change: stage through a scratch texture.
    Scratch,
}

/// Route a `StretchRect` whose source and destination resolve to one texture.
///
/// Regions are in the texture's own coordinates and carry the mip level and the
/// array slice each addresses. Two different levels, and two different slices,
/// are different texels, so neither pair can overlap. A slice is a cube map's
/// `D3DCUBEMAP_FACES` index, and 0 for every texture that holds a single one.
#[must_use]
pub const fn same_surface_route(
    src_region: StretchRegion,
    dst_region: StretchRegion,
    src_mip: u32,
    dst_mip: u32,
    src_slice: u32,
    dst_slice: u32,
) -> SameSurfaceRoute {
    if src_region.w != dst_region.w || src_region.h != dst_region.h {
        return SameSurfaceRoute::Scratch;
    }
    if src_mip != dst_mip || src_slice != dst_slice {
        return SameSurfaceRoute::Direct;
    }
    if src_region.x == dst_region.x && src_region.y == dst_region.y {
        return SameSurfaceRoute::Skip;
    }
    if regions_overlap(src_region, dst_region) {
        SameSurfaceRoute::Scratch
    } else {
        SameSurfaceRoute::Direct
    }
}

/// How a `StretchRect` with a planar YUV endpoint is carried out.
///
/// A planar surface is a decode source only. Nothing encodes into one, the
/// render quad decodes it into a render target at any size, and the CPU
/// converter decodes it 1:1 into an offscreen plain, which cannot be rendered
/// into.
#[derive(Debug, PartialEq, Eq)]
pub enum PlanarStretch {
    /// Neither endpoint is planar: the ordinary format rules decide.
    NotPlanar,
    /// Planar source into a render target: the render quad decodes while sampling.
    RenderQuad,
    /// Planar source copied 1:1 into an offscreen plain: the CPU converter decodes.
    CpuConvert,
    /// No path carries the pair out.
    Reject(RejectReason),
}

/// Route a `StretchRect` by its planar YUV endpoints, if it has any.
///
/// `dst_is_render_target` separates the two destination classes `StretchRect`
/// accepts, the other being a default-pool offscreen plain; `scaling` is
/// whether the two rects differ in size.
#[must_use]
pub const fn planar_stretch_route(
    src_format: u32,
    dst_format: u32,
    dst_is_render_target: bool,
    scaling: bool,
) -> PlanarStretch {
    if is_planar_yuv(dst_format) {
        return PlanarStretch::Reject(RejectReason::PlanarDestination);
    }
    if !is_planar_yuv(src_format) {
        return PlanarStretch::NotPlanar;
    }
    if dst_is_render_target {
        return PlanarStretch::RenderQuad;
    }
    if scaling {
        return PlanarStretch::Reject(RejectReason::Scaling);
    }
    if can_convert(src_format, dst_format) {
        PlanarStretch::CpuConvert
    } else {
        PlanarStretch::Reject(RejectReason::FormatMismatch)
    }
}

/// Source-side decode the `StretchRect` render quad applies while sampling.
///
/// Reaches the blit fragment function as a uniform (`src_level.y`), so the one
/// pipeline per destination format serves every source format: mode 0 samples
/// the source as-is, the packed modes fetch the 4:2:2 macropixel and the
/// planar modes the luma texel and its 4:2:0 chroma sample, and all four YUV
/// modes convert to RGB. The opaque-alpha mode samples as-is and replaces the
/// alpha with one. The discriminants are the uniform's values; the MSL in
/// `unix/unix/src/metal/blit.rs` matches on them.
#[repr(u32)]
pub enum BlitDecode {
    /// Sample the source texture as-is (any RGB format).
    None = 0,
    /// `D3DFMT_YUY2`: macropixel bytes `Y0 U Y1 V`, backed by an RG8 texture.
    Yuy2 = 1,
    /// `D3DFMT_UYVY`: macropixel bytes `U Y0 V Y1`, backed by an RG8 texture.
    Uyvy = 2,
    /// `D3DFMT_YV12`: luma rows, a V plane, a U plane, backed by one R8 texture.
    Yv12 = 3,
    /// `D3DFMT_NV12`: luma rows, one interleaved U, V plane, backed by one R8 texture.
    Nv12 = 4,
    /// An X format: sample as-is, alpha one, since its padding bits carry no alpha.
    OpaqueAlpha = 5,
}

impl BlitDecode {
    /// The value the blit fragment function reads from its uniform.
    #[must_use]
    pub const fn uniform(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Yuy2 => 1.0,
            Self::Uyvy => 2.0,
            Self::Yv12 => 3.0,
            Self::Nv12 => 4.0,
            Self::OpaqueAlpha => 5.0,
        }
    }
}

/// The decode a `StretchRect` source of `d3d_format` needs.
#[must_use]
pub const fn blit_decode(d3d_format: u32) -> BlitDecode {
    match d3d_format {
        D3DFMT_YUY2 => BlitDecode::Yuy2,
        D3DFMT_UYVY => BlitDecode::Uyvy,
        D3DFMT_YV12 => BlitDecode::Yv12,
        D3DFMT_NV12 => BlitDecode::Nv12,
        D3DFMT_X8R8G8B8 | D3DFMT_X8B8G8R8 | D3DFMT_X1R5G5B5 => BlitDecode::OpaqueAlpha,
        _ => BlitDecode::None,
    }
}

/// Whether a copy of the bytes would hand a source's padding to the destination as alpha.
///
/// Takes the mappings the two textures were created with on this device. A
/// source without alpha whose storage still has an alpha lane (an X format,
/// or a format widened to BGRA8) leaves undefined padding or a forced value
/// in it, and D3D9 reads such a surface's alpha as one, so a destination that
/// does carry alpha has to receive alpha one rather than those bits. The other
/// direction is a byte copy, since every reader of a destination without alpha
/// already ignores the alpha an A source leaves in it.
#[must_use]
pub const fn exposes_padding_as_alpha(src: &FormatMapping, dst: &FormatMapping) -> bool {
    !src.has_alpha() && dst.has_alpha()
}

/// Whether the render quad into `rect` writes every pixel of a destination level of `size`.
///
/// Both are in the destination texture's own space. The quad is drawn under
/// a viewport and a scissor equal to `rect`, with no blending and every
/// colour channel enabled, and its one triangle covers the whole viewport,
/// so it writes every sample of every pixel inside `rect`. A rect that starts
/// at the origin and spans the level therefore leaves nothing of the previous
/// contents for its pass to load.
#[must_use]
pub const fn quad_covers_destination(rect: StretchRegion, size: (u32, u32)) -> bool {
    rect.x == 0 && rect.y == 0 && rect.w == size.0 && rect.h == size.1
}

/// Whether `d3d_format` is one of the two packed 4:2:2 YUV formats.
#[must_use]
pub const fn is_packed_yuv(d3d_format: u32) -> bool {
    matches!(d3d_format, D3DFMT_YUY2 | D3DFMT_UYVY)
}

/// Whether a copy of the bytes between two formats that share their storage would reinterpret them.
///
/// `YUY2` and `UYVY` are both stored as two-channel bytes, as `A8L8` is, but
/// each packed YUV format orders its luma and chroma bytes its own way, so a
/// verbatim copy between one of them and any other format hands the
/// destination bytes in a layout it does not have. No conversion between the
/// two exists either: `CheckDeviceFormatConversion` answers no for every such
/// pair, since packed YUV is never a destination.
#[must_use]
pub const fn reinterprets_packed_yuv(src_format: u32, dst_format: u32) -> bool {
    src_format != dst_format && (is_packed_yuv(src_format) || is_packed_yuv(dst_format))
}

/// Whether `d3d_format` is one of the two planar 4:2:0 YUV formats.
#[must_use]
pub const fn is_planar_yuv(d3d_format: u32) -> bool {
    matches!(d3d_format, D3DFMT_YV12 | D3DFMT_NV12)
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

/// Decode texel `(x, y)` of a locked planar 4:2:0 surface (`YV12` / `NV12`).
///
/// `src` is the whole allocation, `pitch` its lock pitch and `luma_rows` its
/// height. The texel's luma is its own byte; its chroma is the sample its 2x2
/// block shares, taken unfiltered. `None` for a format that is not planar
/// YUV, for a layout `planar_yuv` rejects, for a texel outside the luma plane,
/// and for an allocation too short to hold the bytes the texel names.
#[must_use]
pub fn decode_planar_yuv(
    d3d_format: u32,
    src: &[u8],
    pitch: usize,
    luma_rows: usize,
    x: usize,
    y: usize,
) -> Option<(u8, u8, u8)> {
    let layout = planar_yuv_layout_from_pitch(d3d_format, pitch, luma_rows)?;
    if !layout.contains(x, y) {
        return None;
    }
    let luma = *src.get(layout.luma_offset(x, y))?;
    let (cb, cr) = if d3d_format == D3DFMT_YV12 {
        (
            *src.get(layout.yv12_u_offset(x, y))?,
            *src.get(layout.yv12_v_offset(x, y))?,
        )
    } else {
        let pair = layout.nv12_uv_offset(x, y);
        (*src.get(pair)?, *src.get(pair + 1)?)
    };
    Some(yuv_to_rgb8(luma, cb, cr))
}

/// Whether two half-open regions of one mip level share a texel.
///
/// D3D9 clamps every rect to the surface, so `x + w` and `y + h` are bounded
/// by the 16384 maximum surface dimension and cannot overflow.
const fn regions_overlap(a: StretchRegion, b: StretchRegion) -> bool {
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
}

#[cfg(test)]
mod tests;
