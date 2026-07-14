use log::warn;
use mtld3d_shared::mtl::{PixelFormat, Swizzle};
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8L8, D3DFMT_A8R8G8B8,
    D3DFMT_A32B32G32R32F, D3DFMT_ATI1, D3DFMT_D15S1, D3DFMT_D16, D3DFMT_D16_LOCKABLE,
    D3DFMT_D24FS8, D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_D32F_LOCKABLE,
    D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1, D3DFMT_DXT2, D3DFMT_DXT3, D3DFMT_DXT4, D3DFMT_DXT5,
    D3DFMT_INTZ, D3DFMT_L8, D3DFMT_L16, D3DFMT_R5G6B5, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_UYVY,
    D3DFMT_V8U8, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
};

use super::LOG_TARGET;

pub struct FormatMapping {
    metal_pixel_format: PixelFormat,
    bytes_per_pixel: u32,
    block_width: u32,
    block_height: u32,
    block_bytes: u32,
    swizzle: Option<[Swizzle; 4]>,
    /// Whether the D3D9 format carries a real alpha channel.
    ///
    /// Distinguishes formats that share one Metal pixel format but differ in
    /// alpha semantics — notably X8R8G8B8 (false) vs A8R8G8B8 (true), which
    /// both back `Bgra8Unorm`. Consumed by the blend-factor translation: on a
    /// no-alpha render target D3D9 treats destination alpha as the constant
    /// 1.0, so `D3DBLEND_DESTALPHA`/`INVDESTALPHA` must resolve to One/Zero
    /// rather than sampling the physically-stored (undefined) X byte.
    has_alpha: bool,
}

impl FormatMapping {
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        self.block_width > 1
    }

    #[must_use]
    pub const fn metal_pixel_format(&self) -> PixelFormat {
        self.metal_pixel_format
    }

    #[must_use]
    pub const fn swizzle(&self) -> Option<[Swizzle; 4]> {
        self.swizzle
    }

    /// True when the source D3D9 format has a real alpha channel.
    ///
    /// See the `has_alpha` field doc — the blend-factor translation reads
    /// this to clamp destination-alpha blend factors on alpha-less render
    /// targets.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Source-format bytes per pixel.
    ///
    /// Zero for compressed formats (BC1/2/3), where uploads go by block size
    /// and sub-rect upload is gated on 4×4 alignment — a full-mip fallback
    /// today.
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> u32 {
        self.bytes_per_pixel
    }

    #[must_use]
    pub const fn block_width(&self) -> u32 {
        self.block_width
    }

    #[must_use]
    pub const fn block_height(&self) -> u32 {
        self.block_height
    }

    #[must_use]
    pub const fn block_bytes(&self) -> u32 {
        self.block_bytes
    }
}

/// Friendly name for a D3DFMT_* code.
///
/// Returns `"D3DFMT_<code>"` for formats not in the canonical mtld3d mapping
/// table — keeps log lines readable for both supported and unmapped formats
/// without callers needing to special-case "unknown".
#[must_use]
pub const fn format_name(d3d_format: u32) -> &'static str {
    match d3d_format {
        D3DFMT_A8R8G8B8 => "A8R8G8B8",
        D3DFMT_X8R8G8B8 => "X8R8G8B8",
        D3DFMT_R5G6B5 => "R5G6B5",
        D3DFMT_A1R5G5B5 => "A1R5G5B5",
        D3DFMT_A4R4G4B4 => "A4R4G4B4",
        D3DFMT_A8 => "A8",
        D3DFMT_A8L8 => "A8L8",
        D3DFMT_L8 => "L8",
        D3DFMT_L16 => "L16",
        D3DFMT_R16F => "R16F",
        D3DFMT_R32F => "R32F",
        D3DFMT_A32B32G32R32F => "A32B32G32R32F",
        D3DFMT_ATI1 => "ATI1",
        D3DFMT_V8U8 => "V8U8",
        D3DFMT_DXT1 => "DXT1",
        D3DFMT_DXT2 => "DXT2",
        D3DFMT_DXT3 => "DXT3",
        D3DFMT_DXT4 => "DXT4",
        D3DFMT_DXT5 => "DXT5",
        D3DFMT_YUY2 => "YUY2",
        D3DFMT_UYVY => "UYVY",
        D3DFMT_D16_LOCKABLE => "D16_LOCKABLE",
        D3DFMT_D32 => "D32",
        D3DFMT_D15S1 => "D15S1",
        D3DFMT_D24S8 => "D24S8",
        D3DFMT_D24X8 => "D24X8",
        D3DFMT_D24X4S4 => "D24X4S4",
        D3DFMT_D16 => "D16",
        D3DFMT_D32F_LOCKABLE => "D32F_LOCKABLE",
        D3DFMT_D24FS8 => "D24FS8",
        D3DFMT_INTZ => "INTZ",
        D3DFMT_DF24 => "DF24",
        D3DFMT_DF16 => "DF16",
        _ => "D3DFMT_unknown",
    }
}

/// Map a D3D9 depth-stencil format to its Metal pixel format.
///
/// Apple Silicon has no native 24-bit depth format, so the entire D24
/// family promotes to `Depth32Float`. Stencil-bearing variants share
/// `Depth32FloatStencil8`. The FOURCC sampleable-depth formats
/// (`INTZ`/`DF24`/`DF16`) are 24- or 16-bit depth-only on real hardware;
/// they likewise promote to `Depth32Float` — the receiver shader uses
/// `depth2d<float>` + `sample_compare` regardless of the source
/// precision, so collapsing them keeps the create/sample path uniform.
/// Returns `None` for non-depth or unknown formats.
///
/// Used by both `CreateDepthStencilSurface` (standalone depth surface) and
/// `CreateTexture` with `D3DUSAGE_DEPTHSTENCIL` (sampleable shadow map),
/// so the depth-format mapping has a single source of truth.
#[must_use]
pub const fn map_d3d_depth_format(d3d_format: u32) -> Option<PixelFormat> {
    match d3d_format {
        D3DFMT_D16_LOCKABLE | D3DFMT_D32 | D3DFMT_D24X8 | D3DFMT_D16 | D3DFMT_D32F_LOCKABLE
        | D3DFMT_INTZ | D3DFMT_DF24 | D3DFMT_DF16 => Some(PixelFormat::Depth32Float),
        D3DFMT_D15S1 | D3DFMT_D24S8 | D3DFMT_D24X4S4 | D3DFMT_D24FS8 => {
            Some(PixelFormat::Depth32FloatStencil8)
        }
        _ => None,
    }
}

/// True for any D3D9 depth/stencil format `map_d3d_depth_format` recognises.
#[must_use]
pub const fn is_depth_format(d3d_format: u32) -> bool {
    map_d3d_depth_format(d3d_format).is_some()
}

/// True for the FOURCC "readable raw depth" formats (`INTZ`/`DF24`/`DF16`).
///
/// Unlike the implicit depth-stencil formats (D24X8/D24S8/D16/…) — which are
/// sampled through a hardware depth COMPARISON (`sample_compare`, the
/// shadow-map path) — these return the RAW stored normalized depth from a
/// plain `.sample()` broadcast to all channels. These three FOURCC formats are
/// excluded from the hardware shadow-comparison path, so a pixel shader's
/// `texld` against an INTZ/DF24/DF16 texture must fetch the raw depth, not a
/// 0/1 PCF result.
#[must_use]
pub const fn is_raw_depth_fetch_format(d3d_format: u32) -> bool {
    matches!(d3d_format, D3DFMT_INTZ | D3DFMT_DF24 | D3DFMT_DF16)
}

/// Whether `d3d_format` is one of the S3TC/DXT block-compressed formats.
///
/// D3D9 size-checks `DXT1`..`DXT5` at texture/surface creation: the top mip's
/// width and height must be multiples of the 4×4 block, else `INVALIDCALL`. The
/// ATI1/ATI2 block formats are deliberately NOT included — real drivers do not
/// size-check those, so only the DXT family is rejected.
#[must_use]
pub const fn is_dxt_format(d3d_format: u32) -> bool {
    matches!(
        d3d_format,
        D3DFMT_DXT1 | D3DFMT_DXT2 | D3DFMT_DXT3 | D3DFMT_DXT4 | D3DFMT_DXT5
    )
}

#[must_use]
pub fn map_d3d_format(d3d_format: u32) -> Option<FormatMapping> {
    match d3d_format {
        D3DFMT_A8R8G8B8 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Bgra8Unorm,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_X8R8G8B8 => Some(FormatMapping {
            // Same memory layout as A8R8G8B8 but D3D9 semantics say the X
            // byte is "don't care" and sampling returns alpha = 1. Metal
            // reads the X byte as the alpha channel; without a swizzle the
            // shader sees alpha = 0 (or whatever garbage was in that byte),
            // which makes every SRC_ALPHA-blended draw invisible. Force the
            // alpha output to 1 via the texture swizzle.
            metal_pixel_format: PixelFormat::Bgra8Unorm,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::One]),
            // The X byte is "don't care"; D3D9 samples alpha as 1.0 and, as a
            // render target, destination alpha is the constant 1.0 (no alpha
            // channel to blend against).
            has_alpha: false,
        }),
        D3DFMT_A8 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::A8Unorm,
            bytes_per_pixel: 1,
            block_width: 1,
            block_height: 1,
            block_bytes: 1,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_L8 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R8Unorm,
            bytes_per_pixel: 1,
            block_width: 1,
            block_height: 1,
            block_bytes: 1, // L8: replicate R to RGB, alpha = 1.0
            swizzle: Some([Swizzle::Red, Swizzle::Red, Swizzle::Red, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_A8L8 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rg8Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2, // A8L8: luminance=R to RGB, alpha=G
            swizzle: Some([Swizzle::Red, Swizzle::Red, Swizzle::Red, Swizzle::Green]),
            has_alpha: true,
        }),
        D3DFMT_R5G6B5 => Some(FormatMapping {
            // 16-bit packed 5/6/5. Bit-identical to Metal's B5G6R5Unorm
            // (both: B[0-4] G[5-10] R[11-15]); the 2-byte source uploads
            // straight in with no CPU expansion and no swizzle.
            metal_pixel_format: PixelFormat::B5G6R5Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            swizzle: None,
            has_alpha: false,
        }),
        D3DFMT_A1R5G5B5 => Some(FormatMapping {
            // 16-bit packed 5/5/5/1. Bit-identical to Metal's BGR5A1Unorm
            // (both: B[0-4] G[5-9] R[10-14] A[15]); no expansion, no swizzle.
            metal_pixel_format: PixelFormat::Bgr5A1Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_A4R4G4B4 => Some(FormatMapping {
            // 16-bit packed 4/4/4/4. Metal has only ABGR4Unorm (A[0-3] B[4-7]
            // G[8-11] R[12-15]), whose bit order differs from D3D's A4R4G4B4
            // (B[0-3] G[4-7] R[8-11] A[12-15]). Upload the raw bytes and recover
            // D3D channel order with a sampler swizzle: the GPU reads
            // (R,G,B,A)=(D_A,D_R,D_G,D_B), so map out.R←G, out.G←B, out.B←A,
            // out.A←R. The swizzle is sampling-only, so A4R4G4B4 is not a
            // render-target format (see `is_render_target_format`).
            metal_pixel_format: PixelFormat::Abgr4Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            swizzle: Some([Swizzle::Green, Swizzle::Blue, Swizzle::Alpha, Swizzle::Red]),
            has_alpha: true,
        }),
        D3DFMT_V8U8 => Some(FormatMapping {
            // Signed two-channel (tangent-space normals etc.) — exact Metal match.
            metal_pixel_format: PixelFormat::Rg8Snorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2, // D3D9 samples the absent B/A of a 2-channel format as 1.0
            // ({R,G,1,1}); Metal's Rg8Snorm default gives B=0.
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::One, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_DXT1 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Bc1Rgba,
            bytes_per_pixel: 0,
            block_width: 4,
            block_height: 4,
            block_bytes: 8,
            swizzle: None,
            has_alpha: false,
        }),
        // DXT2 and DXT3 are the same BC2 block layout; DXT2's premultiplied
        // alpha is a sampling convention Metal does not distinguish, so both map
        // to BC2.
        D3DFMT_DXT2 | D3DFMT_DXT3 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Bc2Rgba,
            bytes_per_pixel: 0,
            block_width: 4,
            block_height: 4,
            block_bytes: 16,
            swizzle: None,
            has_alpha: true,
        }),
        // DXT4 and DXT5 are the same BC3 block layout (DXT4 = premultiplied).
        D3DFMT_DXT4 | D3DFMT_DXT5 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Bc3Rgba,
            bytes_per_pixel: 0,
            block_width: 4,
            block_height: 4,
            block_bytes: 16,
            swizzle: None,
            has_alpha: true,
        }),
        // YUY2/UYVY are 4:2:2 packed YUV (2 bytes per pixel, 4 per 2-pixel
        // macropixel). We don't do YUV→RGB sampling, so they back a creatable,
        // lockable 2-byte surface/volume (RG8) for the conformance lock/offset
        // checks; sampling such a texture would be wrong, but nothing in the
        // target workload uses YUV. Treated as 2 bytes/pixel (1x1 block) so the
        // lock pitch is `width * 2`, matching D3D9.
        D3DFMT_YUY2 | D3DFMT_UYVY => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rg8Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            swizzle: None,
            has_alpha: false,
        }),
        D3DFMT_L16 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R16Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2, // L16: 16-bit luminance — replicate R to RGB, alpha = 1.0.
            swizzle: Some([Swizzle::Red, Swizzle::Red, Swizzle::Red, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_R16F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R16Float,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2, // R16F: single red channel; G=B=0, A=1 (Metal's native single-channel
            // sample), matching D3D9.
            swizzle: None,
            has_alpha: false,
        }),
        D3DFMT_R32F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R32Float,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            swizzle: None,
            has_alpha: false,
        }),
        D3DFMT_A32B32G32R32F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rgba32Float,
            bytes_per_pixel: 16,
            block_width: 1,
            block_height: 1,
            block_bytes: 16,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_ATI1 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Bc4RUnorm,
            bytes_per_pixel: 0,
            block_width: 4,
            block_height: 4,
            block_bytes: 8, // ATI1N (BC4): single red channel — replicate to RGB, alpha = 1.0.
            swizzle: Some([Swizzle::Red, Swizzle::Red, Swizzle::Red, Swizzle::One]),
            has_alpha: false,
        }),
        _ => {
            warn!(target: LOG_TARGET, "reject map_d3d_format(format={d3d_format}) → unsupported");
            None
        }
    }
}

/// Compute mip dimensions and byte size for a given mip level.
#[must_use]
pub fn compute_mip_size(
    base_width: u32,
    base_height: u32,
    level: u32,
    fmt: &FormatMapping,
) -> (u32, u32, u32, u32) {
    let w = (base_width >> level).max(1);
    let h = (base_height >> level).max(1);

    if fmt.is_compressed() {
        let blocks_x = w.div_ceil(fmt.block_width);
        let blocks_y = h.div_ceil(fmt.block_height);
        let bytes_per_row = blocks_x * fmt.block_bytes;
        let byte_size = bytes_per_row * blocks_y;
        (w, h, byte_size, bytes_per_row)
    } else {
        let bytes_per_row = w * fmt.bytes_per_pixel;
        let byte_size = bytes_per_row * h;
        (w, h, byte_size, bytes_per_row)
    }
}

/// Compute the number of mip levels for a texture.
#[must_use]
pub fn compute_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 {
        return 1;
    }
    32 - max_dim.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::{
        D3DFMT_A8R8G8B8, D3DFMT_D15S1, D3DFMT_D16, D3DFMT_D16_LOCKABLE, D3DFMT_D24FS8,
        D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16,
        D3DFMT_DF24, D3DFMT_INTZ, PixelFormat, is_depth_format, map_d3d_depth_format,
    };

    #[test]
    fn depth_only_formats_promote_to_depth32float() {
        // Apple Silicon has no Depth24Unorm — D24X8, D32, D16, and the
        // lockable variants all share Depth32Float.
        for fmt in [
            D3DFMT_D16_LOCKABLE,
            D3DFMT_D32,
            D3DFMT_D24X8,
            D3DFMT_D16,
            D3DFMT_D32F_LOCKABLE,
            // FOURCC sampleable-depth — engines (incl. WoW CSM) gate the
            // shadow-map path on at least one of these being available.
            D3DFMT_INTZ,
            D3DFMT_DF24,
            D3DFMT_DF16,
        ] {
            assert_eq!(
                map_d3d_depth_format(fmt),
                Some(PixelFormat::Depth32Float),
                "format {fmt} should map to Depth32Float"
            );
        }
    }

    #[test]
    fn stencil_bearing_formats_promote_to_depth32float_stencil8() {
        for fmt in [D3DFMT_D15S1, D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24FS8] {
            assert_eq!(
                map_d3d_depth_format(fmt),
                Some(PixelFormat::Depth32FloatStencil8),
                "format {fmt} should map to Depth32FloatStencil8"
            );
        }
    }

    #[test]
    fn non_depth_formats_return_none() {
        assert_eq!(map_d3d_depth_format(D3DFMT_A8R8G8B8), None);
        assert_eq!(map_d3d_depth_format(0), None);
        assert_eq!(map_d3d_depth_format(0xFFFF_FFFF), None);
    }

    #[test]
    fn is_depth_format_matches_map() {
        assert!(is_depth_format(D3DFMT_D24X8));
        assert!(is_depth_format(D3DFMT_D24S8));
        assert!(is_depth_format(D3DFMT_INTZ));
        assert!(is_depth_format(D3DFMT_DF24));
        assert!(is_depth_format(D3DFMT_DF16));
        assert!(!is_depth_format(D3DFMT_A8R8G8B8));
    }
}
