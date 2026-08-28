use log::warn;
use mtld3d_shared::mtl::{PixelFormat, Swizzle};
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8L8, D3DFMT_A8R8G8B8,
    D3DFMT_A16B16G16R16, D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_ATI1, D3DFMT_D15S1,
    D3DFMT_D16, D3DFMT_D16_LOCKABLE, D3DFMT_D24FS8, D3DFMT_D24S8, D3DFMT_D24X4S4, D3DFMT_D24X8,
    D3DFMT_D32, D3DFMT_D32F_LOCKABLE, D3DFMT_DF16, D3DFMT_DF24, D3DFMT_DXT1, D3DFMT_DXT2,
    D3DFMT_DXT3, D3DFMT_DXT4, D3DFMT_DXT5, D3DFMT_G16R16, D3DFMT_G16R16F, D3DFMT_G32R32F,
    D3DFMT_INTZ, D3DFMT_L8, D3DFMT_L16, D3DFMT_R5G6B5, D3DFMT_R8G8B8, D3DFMT_R16F, D3DFMT_R32F,
    D3DFMT_UYVY, D3DFMT_V8U8, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DRTYPE_CUBETEXTURE, D3DRTYPE_INDEXBUFFER, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE,
    D3DRTYPE_VERTEXBUFFER, D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE, D3DUSAGE_AUTOGENMIPMAP,
    D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DMAP, D3DUSAGE_DONOTCLIP, D3DUSAGE_DYNAMIC, D3DUSAGE_NPATCHES,
    D3DUSAGE_POINTS, D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_LEGACYBUMPMAP,
    D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE,
    D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_QUERY_WRAPANDMIP, D3DUSAGE_RENDERTARGET,
    D3DUSAGE_RTPATCHES, D3DUSAGE_SOFTWAREPROCESSING,
};

use super::LOG_TARGET;
use crate::render_scale::RenderScale;

// Usage bits `usage_allowed_for_rtype` weighs. The named D3D9 usage flags
// outside this set are the ones the runtime strips before it validates:
// `D3DUSAGE_WRITEONLY` is a lock hint rather than a capability question, and
// `D3DUSAGE_NONSECURE` belongs to the protected-content path.
const VALIDATED_USAGE: u32 = D3DUSAGE_RENDERTARGET
    | D3DUSAGE_DEPTHSTENCIL
    | D3DUSAGE_SOFTWAREPROCESSING
    | D3DUSAGE_DONOTCLIP
    | D3DUSAGE_POINTS
    | D3DUSAGE_RTPATCHES
    | D3DUSAGE_NPATCHES
    | D3DUSAGE_DYNAMIC
    | D3DUSAGE_AUTOGENMIPMAP
    | D3DUSAGE_DMAP
    | D3DUSAGE_QUERY_LEGACYBUMPMAP
    | D3DUSAGE_QUERY_SRGBREAD
    | D3DUSAGE_QUERY_FILTER
    | D3DUSAGE_QUERY_SRGBWRITE
    | D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING
    | D3DUSAGE_QUERY_VERTEXTEXTURE
    | D3DUSAGE_QUERY_WRAPANDMIP;

// The usage a resource type expresses only while it can be bound as a shader
// resource: every sampling question, plus the two create hints that only a
// texture pool honours.
const SAMPLED_USAGE: u32 = D3DUSAGE_DYNAMIC
    | D3DUSAGE_SOFTWAREPROCESSING
    | D3DUSAGE_QUERY_FILTER
    | D3DUSAGE_QUERY_SRGBREAD
    | D3DUSAGE_QUERY_SRGBWRITE
    | D3DUSAGE_QUERY_VERTEXTEXTURE
    | D3DUSAGE_QUERY_WRAPANDMIP;

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
/// A mapped format renders as its bare name (`"A8R8G8B8"`); anything outside
/// the canonical mtld3d mapping table renders as the fixed string
/// `"D3DFMT_unknown"`. The function is `const` and returns a `&'static str`,
/// so it cannot render the code itself: a log line that has to identify an
/// unmapped format prints the raw code beside the name.
#[must_use]
pub const fn format_name(d3d_format: u32) -> &'static str {
    match d3d_format {
        D3DFMT_A8R8G8B8 => "A8R8G8B8",
        D3DFMT_X8R8G8B8 => "X8R8G8B8",
        D3DFMT_A8B8G8R8 => "A8B8G8R8",
        D3DFMT_X8B8G8R8 => "X8B8G8R8",
        D3DFMT_R8G8B8 => "R8G8B8",
        D3DFMT_R5G6B5 => "R5G6B5",
        D3DFMT_A1R5G5B5 => "A1R5G5B5",
        D3DFMT_X1R5G5B5 => "X1R5G5B5",
        D3DFMT_A4R4G4B4 => "A4R4G4B4",
        D3DFMT_A8 => "A8",
        D3DFMT_A8L8 => "A8L8",
        D3DFMT_L8 => "L8",
        D3DFMT_L16 => "L16",
        D3DFMT_G16R16 => "G16R16",
        D3DFMT_A16B16G16R16 => "A16B16G16R16",
        D3DFMT_R16F => "R16F",
        D3DFMT_G16R16F => "G16R16F",
        D3DFMT_A16B16G16R16F => "A16B16G16R16F",
        D3DFMT_R32F => "R32F",
        D3DFMT_G32R32F => "G32R32F",
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
/// `Depth32FloatStencil8` — INTZ included: it is the sampleable twin of
/// D24S8 and carries its stencil plane. DF24/DF16 are depth-only fetch
/// formats and promote to plain `Depth32Float`.
/// Returns `None` for non-depth or unknown formats.
///
/// Used by both `CreateDepthStencilSurface` (standalone depth surface) and
/// `CreateTexture` with `D3DUSAGE_DEPTHSTENCIL` (sampleable shadow map),
/// so the depth-format mapping has a single source of truth.
#[must_use]
pub const fn map_d3d_depth_format(d3d_format: u32) -> Option<PixelFormat> {
    match d3d_format {
        D3DFMT_D16_LOCKABLE | D3DFMT_D32 | D3DFMT_D24X8 | D3DFMT_D16 | D3DFMT_D32F_LOCKABLE
        | D3DFMT_DF24 | D3DFMT_DF16 => Some(PixelFormat::Depth32Float),
        // INTZ is the sampleable twin of D24S8 and CARRIES ITS STENCIL
        // PLANE: a deferred engine marks material/sky ids in the stencil of
        // the same buffer it later samples raw depth from, and a
        // stencil-less mapping silently no-ops every one of those writes
        // and gates. DF24/DF16 are depth-only fetch formats and stay so.
        D3DFMT_D15S1 | D3DFMT_D24S8 | D3DFMT_D24X4S4 | D3DFMT_D24FS8 | D3DFMT_INTZ => {
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

/// Map a D3D9 colour format to its Metal counterpart, warning when there is none.
///
/// The lookup itself lives in `lookup_d3d_format`; this wrapper adds the
/// rejection warn, so callers that only ask "is this format mapped?" (the
/// `CheckDeviceFormat` predicates, which are hit by routine capability probes
/// for formats no one intends to create) can use the silent form instead.
#[must_use]
pub fn map_d3d_format(d3d_format: u32) -> Option<FormatMapping> {
    let mapping = lookup_d3d_format(d3d_format);
    if mapping.is_none() {
        warn!(target: LOG_TARGET, "reject map_d3d_format(format={d3d_format}) → unsupported");
    }
    mapping
}

/// True when `lookup_d3d_format` has a Metal counterpart for `d3d_format`.
///
/// This is the set the colour create paths accept, so it is also the set
/// `CheckDeviceFormat` must advertise for `D3DRTYPE_TEXTURE`. Silent: a
/// capability probe for an unmapped format is not a fault. Device-independent:
/// the packed 16-bit members stay in the set on every device because
/// `map_d3d_format_device` backs them with `Bgra8Unorm` where the native
/// formats are missing.
#[must_use]
pub const fn is_mapped_color_format(d3d_format: u32) -> bool {
    lookup_d3d_format(d3d_format).is_some()
}

/// `map_d3d_format`, honouring whether the device has the packed 16-bit formats.
///
/// On a device with native packed 16-bit support the answer is identical to
/// `map_d3d_format`. Without it (Intel/AMD Mac2, or `debug.expandPacked16`),
/// the packed 16-bit D3D formats are backed by `Bgra8Unorm` instead;
/// `bytes_per_pixel`/`block_bytes` stay 2 because they describe the SOURCE
/// layout (Lock pitch, staging sizing — D3D9 Lock semantics are unchanged),
/// and the upload widens texels to 32-bit in the GPU upload pass
/// (`upload_pass`). Create paths that freeze a Metal format into a texture
/// must use this form; layout-only callers may keep `map_d3d_format`.
#[must_use]
pub fn map_d3d_format_device(d3d_format: u32, native_packed16: bool) -> Option<FormatMapping> {
    if !native_packed16 {
        let expanded = match d3d_format {
            // The upload pass writes D3D channel order directly into BGRA8
            // and forces alpha opaque for the alpha-less R5G6B5 and
            // X1R5G5B5, so none of the four needs a sampler swizzle, in
            // particular A4R4G4B4 drops the native path's ABGR4
            // channel-order workaround and X1R5G5B5 the native path's forced
            // alpha. A swizzle would also be fatal here rather than
            // cosmetic: Metal refuses `RenderTarget` usage on a swizzled
            // texture view, and the expansion writes these textures through
            // a render pass.
            D3DFMT_R5G6B5 | D3DFMT_X1R5G5B5 => Some(FormatMapping {
                metal_pixel_format: PixelFormat::Bgra8Unorm,
                bytes_per_pixel: 2,
                block_width: 1,
                block_height: 1,
                block_bytes: 2,
                swizzle: None,
                has_alpha: false,
            }),
            D3DFMT_A1R5G5B5 | D3DFMT_A4R4G4B4 => Some(FormatMapping {
                metal_pixel_format: PixelFormat::Bgra8Unorm,
                bytes_per_pixel: 2,
                block_width: 1,
                block_height: 1,
                block_bytes: 2,
                swizzle: None,
                has_alpha: true,
            }),
            _ => None,
        };
        if expanded.is_some() {
            return expanded;
        }
    }
    map_d3d_format(d3d_format)
}

/// The device-dependent half of a `CheckDeviceFormat` usage query.
///
/// `D3DUSAGE_QUERY_FILTER` asks whether a format samples with linear
/// filtering. Metal's pixel-format capability table makes every colour format
/// in the mapping table filterable on both the Apple and the Mac2 families
/// with one exception: the single-precision floats R32F / G32R32F /
/// A32B32G32R32F filter only where `MTLDevice.supports32BitFloatFiltering`
/// holds, which `float32_filtering` carries from the device. The half-float
/// members (R16F / G16R16F / A16B16G16R16F) filter on every family and stay
/// advertised either way.
///
/// Every other usage bit is device-independent and passes through: the
/// render-target, blending and sRGB arms are the caller's to answer, and
/// renderability needs no device query at all, since the same table lists all
/// six float members as colour-renderable on both families.
#[must_use]
pub const fn supports_usage_query(d3d_format: u32, usage: u32, float32_filtering: bool) -> bool {
    if usage & D3DUSAGE_QUERY_FILTER == 0 {
        return true;
    }
    float32_filtering
        || !matches!(
            d3d_format,
            D3DFMT_R32F | D3DFMT_G32R32F | D3DFMT_A32B32G32R32F
        )
}

/// Whether a `CheckDeviceFormat` query on `rtype` may carry `usage`.
///
/// D3D9 weighs the usage bits against the resource type before it looks at the
/// format at all: a type expresses only the bits its resources can be created
/// or bound with, and a query carrying any other bit answers
/// `D3DERR_NOTAVAILABLE` whatever the format is. The group that separates the
/// types is the sampling one (`QUERY_FILTER`, `QUERY_SRGBREAD`,
/// `QUERY_VERTEXTEXTURE`, `QUERY_WRAPANDMIP`, plus the `DYNAMIC` and
/// `SOFTWAREPROCESSING` create hints): it presumes a shader-resource binding,
/// so a plain `D3DRTYPE_SURFACE` cannot answer yes to any of it. A surface
/// keeps `RENDERTARGET`, `DEPTHSTENCIL` and `QUERY_POSTPIXELSHADER_BLENDING`,
/// and gains `QUERY_SRGBWRITE` only next to `RENDERTARGET`, the encode being a
/// property of the render pass rather than of the surface.
///
/// Bits outside `VALIDATED_USAGE` ride through a query without affecting the
/// answer, and an unrecognised `rtype` allows nothing.
#[must_use]
pub const fn usage_allowed_for_rtype(usage: u32, rtype: u32) -> bool {
    let allowed = match rtype {
        D3DRTYPE_SURFACE => {
            let base = D3DUSAGE_RENDERTARGET
                | D3DUSAGE_DEPTHSTENCIL
                | D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING;
            if usage & D3DUSAGE_RENDERTARGET == 0 {
                base
            } else {
                base | D3DUSAGE_QUERY_SRGBWRITE
            }
        }
        D3DRTYPE_TEXTURE => {
            D3DUSAGE_RENDERTARGET
                | D3DUSAGE_DEPTHSTENCIL
                | D3DUSAGE_AUTOGENMIPMAP
                | D3DUSAGE_QUERY_LEGACYBUMPMAP
                | D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING
                | SAMPLED_USAGE
        }
        // A cube map has no depth-stencil binding, and the legacy bump-map
        // question is asked of 2D textures only.
        D3DRTYPE_CUBETEXTURE => {
            D3DUSAGE_RENDERTARGET
                | D3DUSAGE_AUTOGENMIPMAP
                | D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING
                | SAMPLED_USAGE
        }
        // A volume is sampled only: no render-target or depth binding, and no
        // mip generation.
        D3DRTYPE_VOLUME | D3DRTYPE_VOLUMETEXTURE => {
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING | SAMPLED_USAGE
        }
        // A buffer expresses one capability bit; the vertex-processing hints
        // it also accepts at creation (POINTS, NPATCHES, DONOTCLIP) are not
        // questions about the format.
        D3DRTYPE_VERTEXBUFFER | D3DRTYPE_INDEXBUFFER => D3DUSAGE_DYNAMIC,
        _ => 0,
    };
    (usage & VALIDATED_USAGE & !allowed) == 0
}

#[must_use]
const fn lookup_d3d_format(d3d_format: u32) -> Option<FormatMapping> {
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
        D3DFMT_A8B8G8R8 => Some(FormatMapping {
            // The reversed-channel twin of A8R8G8B8: stored R, G, B, A in
            // ascending addresses, which is Metal's RGBA8Unorm byte for byte.
            metal_pixel_format: PixelFormat::Rgba8Unorm,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_X8B8G8R8 => Some(FormatMapping {
            // X8B8G8R8 is to A8B8G8R8 what X8R8G8B8 is to A8R8G8B8: the same
            // memory layout with a "don't care" fourth byte that D3D9 samples
            // as alpha = 1. Force it with the texture swizzle for the same
            // reason, or every SRC_ALPHA-blended draw reads the stored byte.
            metal_pixel_format: PixelFormat::Rgba8Unorm,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_R8G8B8 => Some(FormatMapping {
            // 24-bit B, G, R in ascending addresses, with no Metal
            // counterpart at all: no GPU family has a three-byte colour
            // format. The texels are widened to BGRA8 by the GPU upload pass
            // (`upload_pass`), which also forces alpha opaque, the way the
            // packed 16-bit formats are widened on a device without them.
            // `bytes_per_pixel` stays 3 because it describes the SOURCE
            // layout that D3D9 Lock semantics expose. Unconditional, unlike
            // the packed 16-bit expansion: there is no device where a native
            // format could serve this one.
            metal_pixel_format: PixelFormat::Bgra8Unorm,
            bytes_per_pixel: 3,
            block_width: 1,
            block_height: 1,
            block_bytes: 3,
            // No sampler swizzle: the upload pass writes D3D channel order
            // into the BGRA8 backing directly, and a swizzled view cannot
            // carry the `RenderTarget` usage that pass writes through.
            swizzle: None,
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
        D3DFMT_X1R5G5B5 => Some(FormatMapping {
            // The A1R5G5B5 bit layout with the top bit "don't care": same
            // Metal BGR5A1Unorm, with the alpha output forced to 1 the way
            // X8R8G8B8 forces its X byte. The swizzle is sampling-only, so
            // X1R5G5B5 is not a render-target format (see
            // `is_render_target_format`).
            metal_pixel_format: PixelFormat::Bgr5A1Unorm,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::One]),
            has_alpha: false,
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
        D3DFMT_G16R16 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rg16Unorm,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            // Two-channel unorm samples as (r, g, 1, 1) in D3D9, the same
            // missing-channel rule the float family follows below.
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::One, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_A16B16G16R16 => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rgba16Unorm,
            bytes_per_pixel: 8,
            block_width: 1,
            block_height: 1,
            block_bytes: 8,
            // Named most-significant first, so the stored order is R, G, B, A:
            // byte-identical to Metal's RGBA16Unorm.
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_R16F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R16Float,
            bytes_per_pixel: 2,
            block_width: 1,
            block_height: 1,
            block_bytes: 2,
            // D3D9 reads the channels a float format does not store as 1.0,
            // not 0.0 (an R16F texel of 0.0 samples as (0, 1, 1, 1)). Metal's
            // native single-channel sample is (r, 0, 0, 1), so the missing
            // lanes are forced.
            swizzle: Some([Swizzle::Red, Swizzle::One, Swizzle::One, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_G16R16F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rg16Float,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            // Blue and alpha are not stored, so they sample as 1.0 (see R16F).
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::One, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_A16B16G16R16F => Some(FormatMapping {
            // The D3D9 name lists channels most-significant first, so the
            // stored order is R, G, B, A in ascending addresses — byte-for-byte
            // Metal's RGBA16Float, no swizzle needed. Same reasoning as
            // A32B32G32R32F below.
            metal_pixel_format: PixelFormat::Rgba16Float,
            bytes_per_pixel: 8,
            block_width: 1,
            block_height: 1,
            block_bytes: 8,
            swizzle: None,
            has_alpha: true,
        }),
        D3DFMT_R32F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::R32Float,
            bytes_per_pixel: 4,
            block_width: 1,
            block_height: 1,
            block_bytes: 4,
            // Green, blue and alpha sample as 1.0 (see R16F).
            swizzle: Some([Swizzle::Red, Swizzle::One, Swizzle::One, Swizzle::One]),
            has_alpha: false,
        }),
        D3DFMT_G32R32F => Some(FormatMapping {
            metal_pixel_format: PixelFormat::Rg32Float,
            bytes_per_pixel: 8,
            block_width: 1,
            block_height: 1,
            block_bytes: 8,
            // Blue and alpha are not stored, so they sample as 1.0 (see R16F).
            swizzle: Some([Swizzle::Red, Swizzle::Green, Swizzle::One, Swizzle::One]),
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
        _ => None,
    }
}

/// Row pitch of a CPU-visible linear surface, in bytes.
///
/// `width * bytes_per_pixel` rounded up to a 4-byte boundary. D3D9 leaves the
/// row pitch to the driver, so a host-visible pixel store picks one and
/// keeps it everywhere: the size the backing buffer is allocated at, the pitch
/// `LockRect` reports, the `bytes_per_row` its GPU read-back and upload run
/// at, and the pitch of the DIB a `GetDC` wraps around it. This rounding is
/// the one GDI computes for a DIB of the same width and bit count, and it
/// rejects anything narrower, so a store on a tighter stride cannot be handed
/// to GDI without the DIB reading past the end of every row.
#[must_use]
pub const fn linear_row_pitch(width: u32, bytes_per_pixel: u32) -> u32 {
    width.saturating_mul(bytes_per_pixel).next_multiple_of(4)
}

/// Mip dimensions, staging byte size and row stride for one mip level.
///
/// An uncompressed level strides by [`linear_row_pitch`], the one row pitch
/// every host-visible store of the format uses, so a texture level and an
/// offscreen surface of the same format and width report the same pitch from
/// `LockRect` and step their rows the same way. A compressed level strides by
/// whole block rows, which is a multiple of four bytes at every width already.
/// The byte size is that stride times the level's row count: block rows for a
/// compressed format, texel rows for a linear one.
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
        linear_mip_size(base_width, base_height, level, fmt.bytes_per_pixel)
    }
}

/// Row pitch of one level, from the block geometry a texture carries.
///
/// [`compute_mip_size`] answers this from a [`FormatMapping`]; a caller that
/// has already unpacked its format into block parameters asks here instead. A
/// compressed row is a whole number of blocks wide, and an uncompressed one
/// strides by [`linear_row_pitch`], so the two agree with the sizes their
/// chains were built at.
#[must_use]
pub const fn block_row_pitch(
    width: u32,
    block_width: u32,
    block_bytes: u32,
    bytes_per_pixel: u32,
) -> u32 {
    if block_width > 1 {
        width.div_ceil(block_width).saturating_mul(block_bytes)
    } else {
        linear_row_pitch(width, bytes_per_pixel)
    }
}

/// Mip dimensions, byte size and row stride for one level of a linear format.
///
/// The uncompressed half of [`compute_mip_size`], for a format carried as a
/// bare pixel size rather than a [`FormatMapping`]: a depth format has no
/// colour mapping to hand over, and its chain is measured on the same formula
/// a colour chain is so that the two are charged alike against the
/// `GetAvailableTextureMem` budget. The level strides by [`linear_row_pitch`]
/// and occupies that stride times its texel row count.
#[must_use]
pub fn linear_mip_size(
    base_width: u32,
    base_height: u32,
    level: u32,
    bytes_per_pixel: u32,
) -> (u32, u32, u32, u32) {
    let w = (base_width >> level).max(1);
    let h = (base_height >> level).max(1);
    let bytes_per_row = linear_row_pitch(w, bytes_per_pixel);
    (w, h, bytes_per_row.saturating_mul(h), bytes_per_row)
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

/// Number of mip levels for a volume texture.
///
/// The 3D form of [`compute_mip_count`]: a volume level halves depth along
/// with width and height and every extent floors at one texel, so the chain
/// ends at 1x1x1 and its length is measured on the largest of the three.
#[must_use]
pub fn compute_volume_mip_count(width: u32, height: u32, depth: u32) -> u32 {
    compute_mip_count(width.max(height), depth)
}

/// The level count a create resolves its `Levels` argument to.
///
/// `0` asks for the whole chain. Any other value is the count the caller
/// wants, capped at `natural`, the chain [`compute_mip_count`] (or
/// [`compute_volume_mip_count`]) measures for the dimensions: past that every
/// further level would repeat the last one, and Metal refuses a
/// `mipmapLevelCount` above it outright rather than allocating the repeats.
#[must_use]
pub fn resolve_mip_levels(requested: u32, natural: u32) -> u32 {
    if requested == 0 {
        natural
    } else {
        requested.min(natural)
    }
}

/// Which allocation shape a standalone surface's VRAM charge covers.
///
/// The two standalone-surface entry points spend memory differently once
/// their surface is multisampled, so the charge has to know which one it is
/// sizing.
#[derive(Clone, Copy)]
pub enum StandaloneSurfaceKind {
    /// `CreateRenderTarget`: a single-sample texture, plus a companion above one sample.
    ColorTarget,
    /// `CreateDepthStencilSurface`: the attachment texture alone, multisampled in place.
    DepthStencil,
}

/// Bytes a standalone surface occupies, every texture the create allocated.
///
/// `sample_count` is the surface's Metal sample count, 1 for a single-sampled
/// surface. A multisampled colour target is two allocations, the single-sample
/// texture an application resolves into plus a companion `sample_count` times
/// its size, so it is charged `1 + sample_count` times the single-sample
/// figure. A multisampled depth-stencil surface is one allocation, the
/// multisampled attachment itself (D3D9 offers no way to read one back, so
/// there is nothing to resolve into), so it is charged `sample_count` times.
/// The sRGB twin of either is a view of a texture already charged and costs
/// no memory of its own.
///
/// `width`/`height` are the logical dimensions the surface reports, and
/// `render_scale` the factor its Metal textures were created at: a surface
/// rasterized below the size it reports occupies the smaller extent, so that
/// is what it is charged, on the pitch its own width gives.
#[must_use]
pub fn standalone_surface_bytes(
    width: u32,
    height: u32,
    d3d_format: u32,
    sample_count: u32,
    kind: StandaloneSurfaceKind,
    render_scale: RenderScale,
) -> u64 {
    let copies = match kind {
        StandaloneSurfaceKind::ColorTarget if sample_count > 1 => {
            u64::from(sample_count).saturating_add(1)
        }
        StandaloneSurfaceKind::ColorTarget => 1,
        StandaloneSurfaceKind::DepthStencil => u64::from(sample_count.max(1)),
    };
    let bytes = surface_bytes(
        render_scale.dimension(width),
        render_scale.dimension(height),
        d3d_format,
    );
    bytes.saturating_mul(copies)
}

/// Bytes one single-level `width` x `height` surface of `d3d_format` occupies.
///
/// The single size formula behind the `GetAvailableTextureMem` accounting for
/// the two standalone-surface entry points, `CreateRenderTarget` and
/// `CreateDepthStencilSurface`, so a surface is charged and refunded from
/// identical inputs. Multisampling multiplies it in
/// [`standalone_surface_bytes`], which is also where the extent the Metal
/// textures were created at enters; this measures whatever extent it is
/// handed. The format is the application's own D3D9 one rather than the Metal
/// format the surface is really backed by: a 24-bit depth format promoted to
/// `Depth32Float` is a substitution of ours, and an application sizing its
/// resource budget from this call reasons in the units it passed in. Both
/// branches measure level 0 of the format's own mip chain, so a standalone
/// surface and the top level of a texture of the same size and format are
/// charged the same bytes: a colour format through [`compute_mip_size`], a
/// depth format, which has no [`FormatMapping`] to hand over, through
/// [`linear_mip_size`] on its bare pixel size. Returns 0 for a format with
/// neither a colour nor a depth mapping.
#[must_use]
pub fn surface_bytes(width: u32, height: u32, d3d_format: u32) -> u64 {
    if let Some(fmt) = lookup_d3d_format(d3d_format) {
        let (_, _, byte_size, _) = compute_mip_size(width, height, 0, &fmt);
        return u64::from(byte_size);
    }
    if let Some(bpp) = depth_format_bytes_per_pixel(d3d_format) {
        let (_, _, byte_size, _) = linear_mip_size(width, height, 0, bpp);
        return u64::from(byte_size);
    }
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "surface_bytes({d3d_format:#x}): no colour or depth mapping → 0 bytes charged"
    );
    0
}

/// Bytes per pixel a D3D9 depth/stencil format occupies on the wire.
///
/// The D3D9 figure, kept in step with `map_d3d_depth_format`: every format
/// that has a Metal depth mapping has a size here, so a depth surface is
/// never charged zero bytes. The Metal texture behind it is wider for the
/// 24-bit family, which is a substitution of ours and not part of the budget
/// an application sizes from. Both the standalone-surface charge in
/// `surface_bytes` and the per-level row pitch a depth texture is sized with
/// read this one table.
#[must_use]
pub const fn depth_format_bytes_per_pixel(d3d_format: u32) -> Option<u32> {
    match d3d_format {
        D3DFMT_D16 | D3DFMT_D16_LOCKABLE | D3DFMT_D15S1 | D3DFMT_DF16 => Some(2),
        D3DFMT_D32 | D3DFMT_D32F_LOCKABLE | D3DFMT_D24X8 | D3DFMT_D24S8 | D3DFMT_D24X4S4
        | D3DFMT_D24FS8 | D3DFMT_DF24 | D3DFMT_INTZ => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
