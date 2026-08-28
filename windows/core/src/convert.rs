use std::hash::{Hash, Hasher};

use mtld3d_shared::{
    VertexAttrDesc,
    mtl::{
        AddressMode, BlendFactor, BlendOperation, ColorWriteMask, CompareFunc, CullMode, IndexType,
        MinMagFilter, MipFilter, PrimitiveType, StencilOp, VertexFormat,
    },
};
use mtld3d_types::{
    D3DBLEND_BLENDFACTOR, D3DBLEND_DESTALPHA, D3DBLEND_DESTCOLOR, D3DBLEND_INVBLENDFACTOR,
    D3DBLEND_INVDESTALPHA, D3DBLEND_INVDESTCOLOR, D3DBLEND_INVSRCALPHA, D3DBLEND_INVSRCCOLOR,
    D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_SRCALPHASAT, D3DBLEND_SRCCOLOR, D3DBLEND_ZERO,
    D3DBLENDOP_ADD, D3DBLENDOP_MAX, D3DBLENDOP_MIN, D3DBLENDOP_REVSUBTRACT, D3DBLENDOP_SUBTRACT,
    D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_GREATER, D3DCMP_GREATEREQUAL, D3DCMP_LESS,
    D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCMP_NOTEQUAL, D3DCULL_CCW, D3DCULL_CW, D3DCULL_NONE,
    D3DDECL_END_STREAM, D3DDECLMETHOD_DEFAULT, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_DEC3N,
    D3DDECLTYPE_FLOAT1, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4,
    D3DDECLTYPE_FLOAT16_2, D3DDECLTYPE_FLOAT16_4, D3DDECLTYPE_SHORT2, D3DDECLTYPE_SHORT2N,
    D3DDECLTYPE_SHORT4, D3DDECLTYPE_SHORT4N, D3DDECLTYPE_UBYTE4, D3DDECLTYPE_UBYTE4N,
    D3DDECLTYPE_UDEC3, D3DDECLTYPE_USHORT2N, D3DDECLTYPE_USHORT4N, D3DDECLUSAGE_BLENDINDICES,
    D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION,
    D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_PSIZE, D3DDECLUSAGE_TEXCOORD, D3DFMT_A1R5G5B5,
    D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16,
    D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_G16R16, D3DFMT_G16R16F, D3DFMT_G32R32F,
    D3DFMT_L8, D3DFMT_R5G6B5, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_LASTBETA_D3DCOLOR, D3DFVF_LASTBETA_UBYTE4,
    D3DFVF_NORMAL, D3DFVF_POSITION_MASK, D3DFVF_PSIZE, D3DFVF_SPECULAR, D3DFVF_TEXCOUNT_MASK,
    D3DFVF_TEXCOUNT_SHIFT, D3DFVF_TEXTUREFORMAT1, D3DFVF_TEXTUREFORMAT3, D3DFVF_TEXTUREFORMAT4,
    D3DFVF_XYZ, D3DFVF_XYZB1, D3DFVF_XYZB2, D3DFVF_XYZB3, D3DFVF_XYZB4, D3DFVF_XYZB5,
    D3DFVF_XYZRHW, D3DFVF_XYZW, D3DPT_LINELIST, D3DPT_LINESTRIP, D3DPT_POINTLIST,
    D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP, D3DSTENCILOP_DECR,
    D3DSTENCILOP_DECRSAT, D3DSTENCILOP_INCR, D3DSTENCILOP_INCRSAT, D3DSTENCILOP_INVERT,
    D3DSTENCILOP_KEEP, D3DSTENCILOP_REPLACE, D3DSTENCILOP_ZERO, D3DTADDRESS_BORDER,
    D3DTADDRESS_CLAMP, D3DTADDRESS_MIRROR, D3DTADDRESS_MIRRORONCE, D3DTADDRESS_WRAP,
    D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT, D3DVERTEXELEMENT9,
    MAX_STREAMS,
};
use xxhash_rust::xxh3::Xxh3;

use crate::dxso::{DeclUsage, ff_attr_index_for_semantic};

/// `(usage, usage_index) → input register index` pulled from a parsed VS's `dcl_*` declarations.
///
/// Used at draw time to resolve a bound vertex declaration's elements
/// against the VS's expected inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSemantic {
    pub usage: DeclUsage,
    pub usage_index: u8,
    pub register_index: u16,
}

// ── D3D9→Metal translation helpers ──

/// Decode a D3DCOLOR (ARGB byte order, A in MSB) into normalised RGBA floats.
///
/// Suitable for Metal's `setBlendColorRed:green:blue:alpha:` and other
/// render-pass color slots.
///
/// Mirrors the byte unpack done inside `Clear()` (`device.rs`
/// `D3DCLEAR_TARGET` branch); kept as a unit-tested helper so
/// `D3DRS_BLENDFACTOR` and any future D3DCOLOR consumer share one source of
/// truth.
#[must_use]
pub fn d3dcolor_to_rgba_f32(color: u32) -> [f32; 4] {
    // D3DCOLOR = 0xAARRGGBB → little-endian bytes are [B, G, R, A].
    let [b, g, r, a] = color.to_le_bytes();
    [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        f32::from(a) / 255.0,
    ]
}

/// Apply the sRGB transfer function to the colour lanes of a normalised RGBA value.
///
/// `Clear` under `D3DRS_SRGBWRITEENABLE` stores the clear colour encoded
/// exactly as a draw would store its pixel-shader output: the same OETF the
/// pixel shader applies (`c <= 0.0031308 ? 12.92 c : 1.055 c^(1/2.4) - 0.055`),
/// colour lanes only, alpha untouched. Lanes outside `[0, 1]` are clamped
/// first, as the render target would clamp them.
#[must_use]
pub fn linear_to_srgb_rgba(rgba: [f32; 4]) -> [f32; 4] {
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.003_130_8 {
            12.92 * c
        } else {
            c.powf(1.0 / 2.4).mul_add(1.055, -0.055)
        }
    };
    [encode(rgba[0]), encode(rgba[1]), encode(rgba[2]), rgba[3]]
}

/// Repeat `pixel` across `dst`, filling every byte.
///
/// The `ColorFill` splat. Writes the pattern once and then doubles it with
/// `copy_within`, so the fill runs at memcpy rate rather than one pixel at a
/// time. A trailing partial pixel (a row that is not a whole number of
/// pixels wide) is filled with the pattern's leading bytes, and an empty
/// pattern leaves `dst` untouched.
pub fn splat_pixel_pattern(dst: &mut [u8], pixel: &[u8]) {
    let bpp = pixel.len();
    if bpp == 0 || dst.is_empty() {
        return;
    }
    let head = bpp.min(dst.len());
    dst[..head].copy_from_slice(&pixel[..head]);
    let mut filled = head;
    while filled < dst.len() {
        let take = filled.min(dst.len() - filled);
        dst.copy_within(..take, filled);
        filled += take;
    }
}

/// Encode a D3DCOLOR into one pixel's destination-format bytes for `ColorFill`.
///
/// Returns `None` for formats whose fill encoding isn't implemented yet (the
/// caller still succeeds but leaves the surface unfilled). Byte layouts
/// follow the D3D9 `ColorFill` promotion rules for each destination format.
#[must_use]
pub fn d3dcolor_fill_pixel_bytes(color: u32, d3d_format: u32) -> Option<Vec<u8>> {
    // D3DCOLOR = 0xAARRGGBB → little-endian bytes are [B, G, R, A].
    let [b, g, r, a] = color.to_le_bytes();
    match d3d_format {
        // BGRA8 store order: the fill reads back identically as the D3DCOLOR
        // (X8 surfaces ignore the alpha byte at read time).
        D3DFMT_A8R8G8B8 | D3DFMT_X8R8G8B8 => Some(vec![b, g, r, a]),
        // RGBA8 store order: the reversed-channel twins of the pair above,
        // whose surfaces are also colour attachments on this device, so a
        // `D3DPOOL_DEFAULT` offscreen plain in one of them has to fill.
        D3DFMT_A8B8G8R8 | D3DFMT_X8B8G8R8 => Some(vec![r, g, b, a]),
        // 16-bit packed R5G6B5: top 5 bits of red, top 6 of green, top 5 of
        // blue. Little-endian 2-byte value (e.g. 0xdeadbeef → 0xadfd).
        D3DFMT_R5G6B5 => {
            let packed =
                ((u16::from(r) >> 3) << 11) | ((u16::from(g) >> 2) << 5) | (u16::from(b) >> 3);
            Some(packed.to_le_bytes().to_vec())
        }
        // 16-bit packed 5/5/5/1 and its X twin: the top 5 bits of each colour
        // channel and the top bit of alpha. X1R5G5B5 stores that alpha bit
        // the way X8R8G8B8 stores its alpha byte, and reads force it to 1
        // either way (e.g. 0xdeadbeef -> 0xd6fd).
        D3DFMT_A1R5G5B5 | D3DFMT_X1R5G5B5 => {
            let packed = ((u16::from(a) >> 7) << 15)
                | ((u16::from(r) >> 3) << 10)
                | ((u16::from(g) >> 3) << 5)
                | (u16::from(b) >> 3);
            Some(packed.to_le_bytes().to_vec())
        }
        // 16-bit packed 4/4/4/4: the top nibble of every channel, alpha in
        // the high nibble (e.g. 0xdeadbeef -> 0xdabe).
        D3DFMT_A4R4G4B4 => {
            let packed = ((u16::from(a) >> 4) << 12)
                | ((u16::from(r) >> 4) << 8)
                | ((u16::from(g) >> 4) << 4)
                | (u16::from(b) >> 4);
            Some(packed.to_le_bytes().to_vec())
        }
        // The single-channel 8-bit pair: a luminance destination takes the
        // colour's luminance, an alpha destination its alpha byte.
        D3DFMT_L8 => Some(vec![d3dcolor_luminance(r, g, b)]),
        D3DFMT_A8 => Some(vec![a]),
        // Float formats carry the D3DCOLOR channels normalised to [0, 1], in
        // channel order R, G, B, A — the D3D9 names list them most-significant
        // first, so the stored order is the reverse of the name.
        D3DFMT_R32F => Some((f32::from(r) / 255.0).to_le_bytes().to_vec()),
        D3DFMT_G32R32F => {
            let mut bytes = (f32::from(r) / 255.0).to_le_bytes().to_vec();
            bytes.extend_from_slice(&(f32::from(g) / 255.0).to_le_bytes());
            Some(bytes)
        }
        D3DFMT_A32B32G32R32F => {
            let mut bytes = Vec::with_capacity(16);
            for channel in d3dcolor_to_rgba_f32(color) {
                bytes.extend_from_slice(&channel.to_le_bytes());
            }
            Some(bytes)
        }
        D3DFMT_R16F => Some(f32_to_f16_bits(f32::from(r) / 255.0).to_le_bytes().to_vec()),
        // 16-bit unorm widens each 8-bit channel by replication (0xab -> 0xabab),
        // which is exact for the 0 and 255 endpoints and within half an LSB
        // elsewhere.
        D3DFMT_G16R16 => {
            let mut bytes = unorm8_to_unorm16(r).to_le_bytes().to_vec();
            bytes.extend_from_slice(&unorm8_to_unorm16(g).to_le_bytes());
            Some(bytes)
        }
        D3DFMT_A16B16G16R16 => {
            let mut bytes = Vec::with_capacity(8);
            for channel in [r, g, b, a] {
                bytes.extend_from_slice(&unorm8_to_unorm16(channel).to_le_bytes());
            }
            Some(bytes)
        }
        D3DFMT_G16R16F => {
            let mut bytes = f32_to_f16_bits(f32::from(r) / 255.0).to_le_bytes().to_vec();
            bytes.extend_from_slice(&f32_to_f16_bits(f32::from(g) / 255.0).to_le_bytes());
            Some(bytes)
        }
        D3DFMT_A16B16G16R16F => {
            let mut bytes = Vec::with_capacity(8);
            for channel in d3dcolor_to_rgba_f32(color) {
                bytes.extend_from_slice(&f32_to_f16_bits(channel).to_le_bytes());
            }
            Some(bytes)
        }
        _ => None,
    }
}

/// The luminance of a D3DCOLOR's colour channels, as an 8-bit unorm.
///
/// D3D9's rule for writing a colour into a luminance destination weights the
/// channels by Rec. 709 (0.2125 R, 0.7154 G, 0.0721 B), the same weights D3DX
/// applies when it converts an RGB surface into an L8 or L16 one. The
/// arithmetic runs in ten-thousandths so it stays integral, and the weights
/// sum to one, so an all-0xff colour lands exactly on 0xff.
fn d3dcolor_luminance(r: u8, g: u8, b: u8) -> u8 {
    let weighted = 2125 * u32::from(r) + 7154 * u32::from(g) + 721 * u32::from(b);
    u8::try_from((weighted + 5_000) / 10_000).expect("Rec. 709 weights sum to one")
}

/// Widen an 8-bit unorm channel to 16 bits by replication.
const fn unorm8_to_unorm16(channel: u8) -> u16 {
    u16::from_le_bytes([channel, channel])
}

/// Encode an `f32` as IEEE-754 binary16 bits, rounding to nearest even.
///
/// The D3D9 half-float formats (`R16F`, `G16R16F`, `A16B16G16R16F`) store this
/// encoding, and `ColorFill` has to produce it on the CPU because the fill
/// writes destination bytes directly. Magnitudes above the binary16 range
/// saturate to an infinity, magnitudes below the smallest subnormal flush to a
/// zero, and a NaN stays a NaN — the same edges the GPU's conversion has.
#[must_use]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    // Exponent thresholds, in binary32 biased form, so the whole encode stays
    // in unsigned arithmetic: binary16's bias is 112 lower, its largest finite
    // exponent is 142, its smallest normal is 113, and the ten subnormal steps
    // reach down to 103.
    const HALF_BIAS_SHIFT: u32 = 112;
    const FIRST_OVERFLOW: u32 = 143;
    const SMALLEST_NORMAL: u32 = 113;
    const SMALLEST_SUBNORMAL: u32 = 103;

    let bits = value.to_bits();
    let sign: u32 = if bits >> 31 == 0 { 0 } else { 0x8000 };
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;

    // Infinity keeps an empty payload; NaN keeps its quiet bit, so a NaN in
    // never becomes an infinity out.
    if exponent == 0xff {
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return narrow_f16(sign | 0x7c00 | payload);
    }
    if exponent >= FIRST_OVERFLOW {
        return narrow_f16(sign | 0x7c00);
    }

    // Normal and subnormal differ only in where the 24-bit significand lands:
    // a normal keeps a rebiased exponent field and drops 13 mantissa bits, a
    // subnormal zeroes the exponent field and shifts the significand
    // (implicit bit restored) further down. Rounding is the same on both, and
    // a carry out of the mantissa propagates into the exponent field on its
    // own — including the carry that turns the largest finite into infinity.
    let (unrounded, dropped, shift) = if exponent >= SMALLEST_NORMAL {
        let field = exponent - HALF_BIAS_SHIFT;
        ((field << 10) | (mantissa >> 13), mantissa & 0x1fff, 13)
    } else if exponent >= SMALLEST_SUBNORMAL {
        let shift = HALF_BIAS_SHIFT + 14 - exponent;
        let significand = mantissa | 0x0080_0000;
        (
            significand >> shift,
            significand & ((1 << shift) - 1),
            shift,
        )
    } else {
        // Smaller than the smallest subnormal, zero included.
        return narrow_f16(sign);
    };

    let halfway = 1 << (shift - 1);
    let round_up = dropped > halfway || (dropped == halfway && unrounded & 1 == 1);
    narrow_f16(sign | (unrounded + u32::from(round_up)))
}

/// Narrow an assembled binary16 bit pattern, which is 16 bits wide by construction.
fn narrow_f16(bits: u32) -> u16 {
    u16::try_from(bits).expect("binary16 pattern is 16 bits wide")
}

/// D3DCMP_* → Metal compare function.
pub fn d3d_to_metal_cmp(d3d_func: u32) -> CompareFunc {
    match d3d_func {
        D3DCMP_NEVER => CompareFunc::Never,
        D3DCMP_LESS => CompareFunc::Less,
        D3DCMP_EQUAL => CompareFunc::Equal,
        D3DCMP_LESSEQUAL => CompareFunc::LessEqual,
        D3DCMP_GREATER => CompareFunc::Greater,
        D3DCMP_NOTEQUAL => CompareFunc::NotEqual,
        D3DCMP_GREATEREQUAL => CompareFunc::GreaterEqual,
        D3DCMP_ALWAYS => CompareFunc::Always,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "d3d_to_metal_cmp: D3DCMP {other} unmapped → Always");
            CompareFunc::Always
        }
    }
}

/// D3DSTENCILOP_* → Metal stencil operation.
pub fn d3d_to_metal_stencil_op(d3d_op: u32) -> StencilOp {
    match d3d_op {
        D3DSTENCILOP_KEEP => StencilOp::Keep,
        D3DSTENCILOP_ZERO => StencilOp::Zero,
        D3DSTENCILOP_REPLACE => StencilOp::Replace,
        D3DSTENCILOP_INCRSAT => StencilOp::IncrementClamp,
        D3DSTENCILOP_DECRSAT => StencilOp::DecrementClamp,
        D3DSTENCILOP_INVERT => StencilOp::Invert,
        D3DSTENCILOP_INCR => StencilOp::IncrementWrap,
        D3DSTENCILOP_DECR => StencilOp::DecrementWrap,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "d3d_to_metal_stencil_op: D3DSTENCILOP {other} unmapped → Keep");
            StencilOp::Keep
        }
    }
}

/// D3DBLEND_* → Metal blend factor.
pub fn d3d_to_metal_blend(d3d_blend: u32) -> BlendFactor {
    match d3d_blend {
        D3DBLEND_ZERO => BlendFactor::Zero,
        D3DBLEND_ONE => BlendFactor::One,
        D3DBLEND_SRCCOLOR => BlendFactor::SourceColor,
        D3DBLEND_INVSRCCOLOR => BlendFactor::OneMinusSourceColor,
        D3DBLEND_SRCALPHA => BlendFactor::SourceAlpha,
        D3DBLEND_INVSRCALPHA => BlendFactor::OneMinusSourceAlpha,
        D3DBLEND_DESTALPHA => BlendFactor::DestinationAlpha,
        D3DBLEND_INVDESTALPHA => BlendFactor::OneMinusDestinationAlpha,
        D3DBLEND_DESTCOLOR => BlendFactor::DestinationColor,
        D3DBLEND_INVDESTCOLOR => BlendFactor::OneMinusDestinationColor,
        D3DBLEND_SRCALPHASAT => BlendFactor::SourceAlphaSaturated,
        D3DBLEND_BLENDFACTOR => BlendFactor::BlendColor,
        D3DBLEND_INVBLENDFACTOR => BlendFactor::OneMinusBlendColor,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "d3d_to_metal_blend: D3DBLEND {other} unmapped → Zero");
            BlendFactor::Zero
        }
    }
}

/// D3DBLEND_* → Metal blend factor, honouring the render target's alpha channel.
///
/// D3D9 spec: on a render target whose format has no alpha channel (e.g.
/// X8R8G8B8), destination alpha reads as the constant 1.0. So
/// `D3DBLEND_DESTALPHA` resolves to `One` and `D3DBLEND_INVDESTALPHA` to
/// `Zero` — the physically-stored alpha byte (undefined on an X8 target,
/// whatever a prior clear left behind) must never be sampled. On an
/// alpha-bearing target this is identical to [`d3d_to_metal_blend`].
///
/// `rt_has_alpha` comes from `map_d3d_format(fmt).has_alpha()` for the bound
/// colour RT; it is threaded through the pipeline snapshot so X8 and A8
/// pipelines hash to distinct cache keys (the remapped factors flow into
/// both the key and the wire params).
#[must_use]
pub fn d3d_to_metal_blend_rt(d3d_blend: u32, rt_has_alpha: bool) -> BlendFactor {
    if !rt_has_alpha {
        match d3d_blend {
            D3DBLEND_DESTALPHA => return BlendFactor::One, // dest alpha = 1.0
            D3DBLEND_INVDESTALPHA => return BlendFactor::Zero, // 1 - 1.0 = 0.0
            _ => {}
        }
    }
    d3d_to_metal_blend(d3d_blend)
}

/// D3DBLENDOP_* → Metal blend operation.
///
/// D3D9 values: ADD=1, SUBTRACT=2, REVSUBTRACT=3, MIN=4, MAX=5.
pub fn d3d_to_metal_blend_op(d3d_op: u32) -> BlendOperation {
    match d3d_op {
        D3DBLENDOP_ADD => BlendOperation::Add,
        D3DBLENDOP_SUBTRACT => BlendOperation::Subtract,
        D3DBLENDOP_REVSUBTRACT => BlendOperation::ReverseSubtract,
        D3DBLENDOP_MIN => BlendOperation::Min,
        D3DBLENDOP_MAX => BlendOperation::Max,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "d3d_to_metal_blend_op: D3DBLENDOP {other} unmapped → Add");
            BlendOperation::Add
        }
    }
}

/// Scale D3D9's raw `D3DRS_DEPTHBIAS` value into a Metal `setDepthBias` value.
///
/// The raw value is a float stored in the state DWORD; the scaled result
/// is sized for the active depth-buffer format.
///
/// D3D9's contract is "1 ULP at the depth-buffer's resolution", so the
/// scale factor is `1 / depth_min_unit`. mtld3d maps every D3D9 depth
/// format (D16 / D24X8 / D24S8 / D32 / D32F) to `MTLPixelFormat::Depth32Float`
/// (or `Depth32Float_Stencil8`) — see
/// `unix/unix/src/metal/texture.rs`. For `Depth32Float` the minimum
/// representable depth step in the `[0, 1]` projected range is 2^-23
/// (the float mantissa width), so the scale is `1 << 23`.
///
/// `D3DRS_SLOPESCALEDEPTHBIAS` is a unit-less multiplier, so it does
/// not need scaling — pass it straight through to `setDepthBias`.
#[must_use]
pub fn d3d_depth_bias_to_metal(raw_d3d: u32) -> f32 {
    // 2^23 = 8_388_608 — exactly representable in f32 (literal is exact).
    const D32_FLOAT_BIAS_SCALE: f32 = 8_388_608.0;
    f32::from_bits(raw_d3d) * D32_FLOAT_BIAS_SCALE
}

/// `-1e-4` as `f32` bits — magnitude of the implicit decal-bias.
///
/// Applied by `emit_draw` when `looks_like_decal` matches. Negative
/// pushes toward camera (D3D9 depth: 0 = near, 1 = far). After the
/// `d3d_depth_bias_to_metal` scaling (`1 << 23`) this lands around
/// `-838.9` Metal units — large enough to swamp ULP-level noise from
/// divergent FP rounding between pipelines on Apple Silicon, small
/// enough that genuine geometry an order of magnitude further from the
/// surface still composites correctly. At grazing angles a typical
/// structural eye-space delta between two decal/surface VSes lands in
/// `(3e-5, 1e-4]` on observed pixels at `z ≈ 0.92`, which sets the
/// lower bound. Stored as the precomputed IEEE-754 bit pattern via
/// `f32::to_bits` so the magnitude can be tuned in float-literal form
/// while the call site keeps consuming `u32`.
pub const IMPLICIT_DECAL_BIAS_RAW: u32 = (-1.0e-4_f32).to_bits();

/// Slope-scale component applied alongside `IMPLICIT_DECAL_BIAS_RAW`.
///
/// Applied when `looks_like_decal` fires AND the game has not supplied
/// its own `D3DRS_SLOPESCALEDEPTHBIAS`.
///
/// Metal's `setDepthBias(bias, slopeScale, clamp)` adds
/// `m × slopeScale + r × bias` to the fragment's depth, where
/// `m = max(|dz/dx|, |dz/dy|)` is the screen-space depth slope and `r`
/// is the depth-format's minimum representable step. For a flat
/// surface viewed straight on, `m ≈ 0` so the slope term contributes
/// nothing and the absolute `IMPLICIT_DECAL_BIAS_RAW` handles ULP
/// noise on its own (flat-decal case). For surfaces viewed at grazing
/// angles — wet-ground-style wakes stretching across many screen
/// pixels at `z ≈ 0.92`, where the per-pixel depth derivative is
/// `~0.001` — `m` is large and the slope term contributes `~0.001` of
/// pull-toward-camera, comfortably swamping the structural eye-space
/// delta between the wake's VS and the surface VS even when that
/// delta exceeds the absolute budget. Standard "polygon offset"
/// shape — same combination GL drivers and Vulkan's `vkCmdSetDepthBias`
/// use.
pub const IMPLICIT_DECAL_SLOPE_SCALE: f32 = -1.5;

/// Render-state inputs to `looks_like_decal`.
///
/// Narrow over `RenderStateSnapshot` — only the prongs the predicate
/// actually reads — so the heuristic stays testable in `mtld3d-core`
/// without dragging the COM-wrapper layer in.
#[derive(Clone, Copy, Debug)]
pub struct DecalHeuristicInputs {
    pub depth_enable: u32,
    pub depth_write: u32,
    pub blend_enable: u32,
    pub raw_depth_bias: u32,
    pub raw_slope_scale: u32,
}

/// Returns `true` when the draw matches a typical alpha-blended decal.
///
/// Pattern: depth-test on, depth-write off, alpha-blend on, and the
/// game has not already supplied a `D3DRS_DEPTHBIAS` /
/// `SLOPESCALEDEPTHBIAS`. On Apple Silicon (no `Depth24Unorm`; D3D9
/// D24S8 maps to `Depth32Float`) the finer depth precision exposes
/// ULP noise that the depth-buffer quantization absorbed on Windows.
/// `emit_draw` replaces the game's zero bias with
/// `IMPLICIT_DECAL_BIAS_RAW` when this fires.
#[must_use]
pub const fn looks_like_decal(i: DecalHeuristicInputs) -> bool {
    i.depth_enable != 0
        && i.depth_write == 0
        && i.blend_enable != 0
        && i.raw_depth_bias == 0
        && i.raw_slope_scale == 0
}

/// D3DCULL_* → Metal cull mode.
pub fn d3d_to_metal_cull(d3d_cull: u32) -> CullMode {
    match d3d_cull {
        D3DCULL_NONE => CullMode::None,
        D3DCULL_CW => CullMode::Front,
        D3DCULL_CCW => CullMode::Back,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "d3d_to_metal_cull: D3DCULL {other} unmapped → None");
            CullMode::None
        }
    }
}

/// D3DTEXF_* → Metal sampler min/mag filter.
pub fn d3d_to_metal_min_mag_filter(d3d_filter: u32) -> MinMagFilter {
    match d3d_filter {
        D3DTEXF_POINT => MinMagFilter::Nearest,
        D3DTEXF_LINEAR | D3DTEXF_ANISOTROPIC => MinMagFilter::Linear,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "d3d_to_metal_min_mag_filter: D3DTEXF {other} unmapped → Nearest"
            );
            MinMagFilter::Nearest
        }
    }
}

/// D3DTEXF_* → Metal sampler mip filter.
pub fn d3d_to_metal_mip_filter(d3d_filter: u32) -> MipFilter {
    match d3d_filter {
        D3DTEXF_NONE => MipFilter::NotMipmapped,
        D3DTEXF_POINT => MipFilter::Nearest,
        D3DTEXF_LINEAR | D3DTEXF_ANISOTROPIC => MipFilter::Linear,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "d3d_to_metal_mip_filter: D3DTEXF {other} unmapped → NotMipmapped"
            );
            MipFilter::NotMipmapped
        }
    }
}

/// `D3DSAMP_BORDERCOLOR` (a D3DCOLOR) → the nearest Metal border preset.
///
/// Metal samplers offer three border colours. Transparent black, opaque
/// black and opaque white map exactly; anything else takes opaque black and
/// is logged once per colour, since the border then reads differently from
/// what the game asked for.
pub fn d3d_border_color_to_metal(color: u32) -> mtld3d_shared::mtl::BorderColor {
    if let Some(preset) = border_color_preset(color) {
        return preset;
    }
    mtld3d_shared::log_once_warn_by!(
        target: crate::LOG_TARGET,
        key: u64::from(color),
        "D3DSAMP_BORDERCOLOR {color:#010x} has no Metal preset → opaque black"
    );
    mtld3d_shared::mtl::BorderColor::OpaqueBlack
}

/// The Metal border preset a D3DCOLOR maps to exactly, if any.
///
/// Const so the sampler cache key can fold it in; the logging fallback for
/// other colours lives in [`d3d_border_color_to_metal`].
#[must_use]
pub const fn border_color_preset(color: u32) -> Option<mtld3d_shared::mtl::BorderColor> {
    use mtld3d_shared::mtl::BorderColor;
    match color {
        0x0000_0000 => Some(BorderColor::TransparentBlack),
        0xFF00_0000 => Some(BorderColor::OpaqueBlack),
        0xFFFF_FFFF => Some(BorderColor::OpaqueWhite),
        _ => None,
    }
}

/// D3DTADDRESS_* → Metal sampler address mode.
pub fn d3d_to_metal_address_mode(d3d_mode: u32) -> AddressMode {
    match d3d_mode {
        D3DTADDRESS_WRAP => AddressMode::Repeat,
        D3DTADDRESS_MIRROR => AddressMode::MirrorRepeat,
        D3DTADDRESS_CLAMP => AddressMode::ClampToEdge,
        D3DTADDRESS_BORDER => AddressMode::ClampToBorderColor,
        D3DTADDRESS_MIRRORONCE => AddressMode::MirrorClampToEdge,
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "d3d_to_metal_address_mode: D3DTADDRESS {other} unmapped → Repeat"
            );
            AddressMode::Repeat
        }
    }
}

/// D3D9 color write enable bits → Metal color write mask.
///
/// D3D9 packs bits low-to-high (bit 0 = R); Metal packs high-to-low (bit 3 = R).
#[must_use]
pub fn d3d_to_metal_write_mask(d3d_mask: u32) -> ColorWriteMask {
    let mut metal = ColorWriteMask::empty();
    if d3d_mask & 1 != 0 {
        metal |= ColorWriteMask::RED;
    }
    if d3d_mask & 2 != 0 {
        metal |= ColorWriteMask::GREEN;
    }
    if d3d_mask & 4 != 0 {
        metal |= ColorWriteMask::BLUE;
    }
    if d3d_mask & 8 != 0 {
        metal |= ColorWriteMask::ALPHA;
    }
    metal
}

/// D3DPT_* → Metal primitive type.
pub fn d3d_to_metal_primitive(d3d_type: u32) -> Option<PrimitiveType> {
    match d3d_type {
        D3DPT_POINTLIST => Some(PrimitiveType::Point),
        D3DPT_LINELIST => Some(PrimitiveType::Line),
        D3DPT_LINESTRIP => Some(PrimitiveType::LineStrip),
        D3DPT_TRIANGLELIST => Some(PrimitiveType::Triangle),
        D3DPT_TRIANGLESTRIP => Some(PrimitiveType::TriangleStrip),
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "D3DPRIMITIVETYPE {other} unhandled → draw dropped");
            None
        }
    }
}

/// Where a triangle fan's vertex indices come from.
enum FanSource<'a> {
    /// `DrawPrimitive` / `DrawPrimitiveUP`: vertices back-to-back from `start_vertex`.
    Sequential { start_vertex: u32 },
    /// `DrawIndexedPrimitive` / `DrawIndexedPrimitiveUP`: application indices.
    ///
    /// `src` holds the fan's indices at `index_size` (2 or 4) bytes each
    /// starting at the draw's first index, and `base_vertex` is folded into
    /// every one so the rewritten list is absolute.
    Indexed {
        src: &'a [u8],
        index_size: usize,
        base_vertex: i32,
    },
}

impl FanSource<'_> {
    /// Absolute vertex index of fan vertex `k`.
    ///
    /// `None` when `k` reads past the source stream, the index size is
    /// neither 2 nor 4 bytes, or the index leaves `u32` once the base vertex
    /// is folded in.
    fn vertex(&self, k: usize) -> Option<u32> {
        match *self {
            Self::Sequential { start_vertex } => start_vertex.checked_add(u32::try_from(k).ok()?),
            Self::Indexed {
                src,
                index_size,
                base_vertex,
            } => {
                let first = k.checked_mul(index_size)?;
                let raw = src.get(first..first.checked_add(index_size)?)?;
                let index = match index_size {
                    2 => u32::from(u16::from_le_bytes([raw[0], raw[1]])),
                    4 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    other => {
                        mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "triangle fan: {other}-byte indices unhandled → draw dropped");
                        return None;
                    }
                };
                u32::try_from(i64::from(index) + i64::from(base_vertex)).ok()
            }
        }
    }
}

/// A triangle fan rewritten as a triangle-list index stream.
///
/// Metal has no triangle-fan primitive, and triangle `i` of a fan is fan
/// vertices `0, i + 1, i + 2`, so a fan draws as an indexed triangle list
/// over the caller's untouched vertices. Construction resolves the vertex
/// span and the index width the list needs; [`FanRewrite::write`] then fills
/// a caller-owned buffer of [`FanRewrite::byte_len`] bytes, so the draw path
/// writes the list straight into the frame's scratch arena instead of
/// allocating a stream per call.
pub struct FanRewrite<'a> {
    source: FanSource<'a>,
    primitive_count: u32,
    index_count: u32,
    index_type: IndexType,
    min_vertex: u32,
    max_vertex: u32,
}

impl<'a> FanRewrite<'a> {
    /// Rewrite a non-indexed fan whose vertices run from `start_vertex`.
    ///
    /// `None` when the fan's last vertex leaves `u32`.
    #[must_use]
    pub fn sequential(start_vertex: u32, primitive_count: u32) -> Option<Self> {
        Self::new(FanSource::Sequential { start_vertex }, primitive_count)
    }

    /// Rewrite an indexed fan over `primitive_count + 2` application indices.
    ///
    /// `src` holds them at `index_size` (2 or 4) bytes each from the draw's
    /// first index; `base_vertex` is folded into every one. `None` when
    /// `src` is short, the index size is unknown, or an index leaves `u32`
    /// after the base offset.
    #[must_use]
    pub fn indexed(
        src: &'a [u8],
        index_size: usize,
        base_vertex: i32,
        primitive_count: u32,
    ) -> Option<Self> {
        Self::new(
            FanSource::Indexed {
                src,
                index_size,
                base_vertex,
            },
            primitive_count,
        )
    }

    fn new(source: FanSource<'a>, primitive_count: u32) -> Option<Self> {
        let count = usize::try_from(primitive_count.checked_add(2)?).ok()?;
        let index_count = primitive_count.checked_mul(3)?;
        let mut min_vertex = u32::MAX;
        let mut max_vertex = 0;
        for k in 0..count {
            let vertex = source.vertex(k)?;
            min_vertex = min_vertex.min(vertex);
            max_vertex = max_vertex.max(vertex);
        }
        let index_type = if max_vertex > u32::from(u16::MAX) {
            IndexType::UInt32
        } else {
            IndexType::UInt16
        };
        Some(Self {
            source,
            primitive_count,
            index_count,
            index_type,
            min_vertex,
            max_vertex,
        })
    }

    /// Indices the rewritten triangle list holds.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }

    /// `UInt16` when every index fits, `UInt32` otherwise.
    #[must_use]
    pub const fn index_type(&self) -> IndexType {
        self.index_type
    }

    /// Lowest vertex-buffer index the list references.
    #[must_use]
    pub const fn min_vertex(&self) -> u32 {
        self.min_vertex
    }

    /// Highest vertex-buffer index the list references.
    #[must_use]
    pub const fn max_vertex(&self) -> u32 {
        self.max_vertex
    }

    /// Bytes [`FanRewrite::write`] needs for the whole list.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.index_count as usize * self.index_stride()
    }

    /// Bytes one index of the rewritten list occupies.
    const fn index_stride(&self) -> usize {
        match self.index_type {
            IndexType::UInt16 => 2,
            IndexType::UInt32 => 4,
        }
    }

    /// Write the triangle-list indices into `out`, little-endian.
    ///
    /// Writes the triangles that fit; the caller sizes `out` with
    /// [`FanRewrite::byte_len`]. Fan vertex `i + 2` of one triangle is fan
    /// vertex `i + 1` of the next, so the source is read once per fan vertex
    /// rather than once per index.
    ///
    /// # Panics
    ///
    /// Panics if a fan vertex stops resolving, or stops fitting the index
    /// width, between construction and this call. Both are settled by
    /// construction over the same immutable source.
    pub fn write(&self, out: &mut [u8]) {
        const RESOLVED: &str = "every fan vertex resolved at construction";
        let stride = self.index_stride();
        let first = self.source.vertex(0).expect(RESOLVED);
        let mut prev = self.source.vertex(1).expect(RESOLVED);
        let triangles = self.primitive_count as usize;
        for (i, triangle) in out.chunks_exact_mut(stride * 3).take(triangles).enumerate() {
            let next = self.source.vertex(i + 2).expect(RESOLVED);
            for (slot, vertex) in [first, prev, next].into_iter().enumerate() {
                put_fan_index(
                    &mut triangle[slot * stride..(slot + 1) * stride],
                    vertex,
                    self.index_type,
                );
            }
            prev = next;
        }
    }
}

/// Write one triangle-list index into `dst` at the list's width.
fn put_fan_index(dst: &mut [u8], vertex: u32, index_type: IndexType) {
    match index_type {
        IndexType::UInt16 => {
            let narrow =
                u16::try_from(vertex).expect("narrow index stream only when every index fits");
            dst.copy_from_slice(&narrow.to_le_bytes());
        }
        IndexType::UInt32 => dst.copy_from_slice(&vertex.to_le_bytes()),
    }
}

/// Most triangles the shared 16-bit fan pattern can address.
///
/// Triangle `i` of a fan reads fan vertex `i + 2`, and a 16-bit index stops
/// at `u16::MAX`.
pub const FAN_PATTERN_MAX_TRIANGLES: u32 = u16::MAX as u32 - 1;

/// Bytes the shared 16-bit fan pattern needs for `primitive_count` triangles.
#[must_use]
pub const fn fan_pattern_bytes(primitive_count: u32) -> usize {
    primitive_count as usize * 3 * 2
}

/// Write the first `primitive_count` triangles of the shared fan pattern.
///
/// Triangle `i` is `(0, i + 1, i + 2)` as little-endian `u16`, relative to
/// the fan's first vertex, which the draw supplies as its base vertex. Every
/// non-indexed fan is a prefix of this one pattern, so the encoder keeps a
/// single buffer of it instead of generating indices per draw. Writes the
/// triangles that fit in `out`; the caller sizes it with
/// [`fan_pattern_bytes`].
pub fn fill_fan_pattern_u16(out: &mut [u8], primitive_count: u32) {
    let triangles = usize::try_from(primitive_count).unwrap_or(usize::MAX);
    for (i, tri) in out.chunks_exact_mut(6).take(triangles).enumerate() {
        // `i + 2 <= FAN_PATTERN_MAX_TRIANGLES + 1 == u16::MAX` whenever the
        // caller respects the pattern limit; clamp rather than wrap past it.
        let second = u16::try_from(i + 1).unwrap_or(u16::MAX);
        let third = u16::try_from(i + 2).unwrap_or(u16::MAX);
        tri[0..2].copy_from_slice(&0u16.to_le_bytes());
        tri[2..4].copy_from_slice(&second.to_le_bytes());
        tri[4..6].copy_from_slice(&third.to_le_bytes());
    }
}

/// Compute vertex count from D3D9 primitive type and primitive count.
pub fn vertex_count(d3d_type: u32, primitive_count: u32) -> u32 {
    match d3d_type {
        D3DPT_POINTLIST => primitive_count,        // point list
        D3DPT_LINELIST => primitive_count * 2,     // line list
        D3DPT_LINESTRIP => primitive_count + 1,    // line strip
        D3DPT_TRIANGLELIST => primitive_count * 3, // triangle list
        D3DPT_TRIANGLESTRIP | D3DPT_TRIANGLEFAN => primitive_count + 2, // strip / fan
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "vertex_count: D3DPRIMITIVETYPE {other} unhandled → 0 verts");
            0
        }
    }
}

/// `D3DDECLTYPE_*` → `(MTLVertexFormat, size_bytes)`.
///
/// **D3DCOLOR footnote:** `D3DCOLOR` is ARGB packed bytes in memory
/// (`[B, G, R, A]` little-endian). Metal's
/// `MTLVertexFormat::UChar4Normalized_BGRA` performs the BGRA→RGBA swizzle
/// at vertex-fetch time, so the shader receives the float4 in `(R, G, B, A)`
/// lane order just as the D3D9 programmable-shader ABI promises. Without
/// this, shaders that read BLENDWEIGHT/BLENDINDICES declared as D3DCOLOR
/// pick up the wrong lanes (visible as mis-skinned hair / character
/// blends), and FF color inputs would need a compensating `.zyxw`
/// swizzle.
pub fn decl_type_to_metal_format(ty: u8) -> (VertexFormat, u32) {
    match ty {
        D3DDECLTYPE_FLOAT1 => (VertexFormat::Float, 4),
        D3DDECLTYPE_FLOAT2 => (VertexFormat::Float2, 8),
        D3DDECLTYPE_FLOAT3 => (VertexFormat::Float3, 12),
        D3DDECLTYPE_FLOAT4 => (VertexFormat::Float4, 16),
        D3DDECLTYPE_D3DCOLOR => (VertexFormat::UChar4NormalizedBgra, 4),
        D3DDECLTYPE_UBYTE4 => (VertexFormat::UChar4, 4),
        D3DDECLTYPE_SHORT2 => (VertexFormat::Short2, 4),
        D3DDECLTYPE_SHORT4 => (VertexFormat::Short4, 8),
        D3DDECLTYPE_UBYTE4N => (VertexFormat::UChar4Normalized, 4),
        D3DDECLTYPE_SHORT2N => (VertexFormat::Short2Normalized, 4),
        D3DDECLTYPE_SHORT4N => (VertexFormat::Short4Normalized, 8),
        D3DDECLTYPE_USHORT2N => (VertexFormat::UShort2Normalized, 4),
        D3DDECLTYPE_USHORT4N => (VertexFormat::UShort4Normalized, 8),
        D3DDECLTYPE_FLOAT16_2 => (VertexFormat::Half2, 4),
        D3DDECLTYPE_FLOAT16_4 => (VertexFormat::Half4, 8),
        // Packed 10-10-10 formats have no direct Metal equivalent — mark
        // invalid and log at the caller. Uncommon in SM2 content.
        D3DDECLTYPE_UDEC3 | D3DDECLTYPE_DEC3N => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "D3DDECLTYPE UDEC3/DEC3N has no Metal format — element dropped");
            (VertexFormat::Invalid, 0)
        }
        other => {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "D3DDECLTYPE {other} unhandled — element dropped (no Metal format)"
            );
            (VertexFormat::Invalid, 0)
        }
    }
}

/// Convert an FVF bitmask into an equivalent `D3DVERTEXELEMENT9[]` sequence.
///
/// The terminator is excluded; the total vertex stride is returned
/// alongside the elements.
///
/// # Panics
///
/// Panics if the FVF encodes counts outside D3D9 spec bounds (`XYZBn` betas > 5,
/// tex-coord count > 8, Metal vertex format size > `u16::MAX`). All three are
/// unreachable for any FVF produced by a real D3D9 caller.
#[must_use]
pub fn fvf_to_elements(fvf: u32) -> (Vec<D3DVERTEXELEMENT9>, u32) {
    let mut elements: Vec<D3DVERTEXELEMENT9> = Vec::new();
    let mut push = |ty: u8, usage: u8, usage_index: u8| {
        elements.push(D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0, // filled in at the end
            type_: ty,
            method: D3DDECLMETHOD_DEFAULT,
            usage,
            usage_index,
        });
    };

    match fvf & D3DFVF_POSITION_MASK {
        // XYZ (0x002) and XYZW (0x4002) both mask to 0x002 under
        // D3DFVF_POSITION_MASK; the W bit (0x4000) distinguishes them, so XYZW
        // must be detected against the unmasked fvf (POSITION FLOAT4, not FLOAT3).
        D3DFVF_XYZ if (fvf & D3DFVF_XYZW) == D3DFVF_XYZW => {
            push(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITION, 0);
        }
        D3DFVF_XYZ => push(D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0),
        pos @ (D3DFVF_XYZB1 | D3DFVF_XYZB2 | D3DFVF_XYZB3 | D3DFVF_XYZB4 | D3DFVF_XYZB5) => {
            push(D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0);
            // Each XYZBn: first 3 floats are position, then n blend-weight
            // lanes. The last lane may be a packed index (UBYTE4 / D3DCOLOR).
            let mut betas =
                u8::try_from(((pos - D3DFVF_XYZB1) >> 1) + 1).expect("D3D9 XYZBn betas ≤ 5");
            // D3D9 quirk: `D3DFVF_XYZB2 | LASTBETA_D3DCOLOR` packs the blend
            // *weight* into a D3DCOLOR and the blend *index* into UBYTE4. Every
            // other `XYZBn` keeps the weight as a float vector and the index as
            // the LASTBETA type. Follows the D3D9 FVF→vertex-declaration
            // conversion rule.
            let xyzb2_d3dcolor = pos == D3DFVF_XYZB2 && fvf & D3DFVF_LASTBETA_D3DCOLOR != 0;
            let last_beta_ty = if xyzb2_d3dcolor {
                Some(D3DDECLTYPE_UBYTE4)
            } else if fvf & D3DFVF_LASTBETA_D3DCOLOR != 0 {
                Some(D3DDECLTYPE_D3DCOLOR)
            } else if fvf & D3DFVF_LASTBETA_UBYTE4 != 0 {
                Some(D3DDECLTYPE_UBYTE4)
            } else if pos == D3DFVF_XYZB5 {
                Some(D3DDECLTYPE_FLOAT1)
            } else {
                None
            };
            if last_beta_ty.is_some() && betas > 0 {
                betas -= 1;
            }
            if betas > 0 {
                let ty = if xyzb2_d3dcolor {
                    D3DDECLTYPE_D3DCOLOR
                } else {
                    match betas {
                        1 => D3DDECLTYPE_FLOAT1,
                        2 => D3DDECLTYPE_FLOAT2,
                        3 => D3DDECLTYPE_FLOAT3,
                        _ => D3DDECLTYPE_FLOAT4,
                    }
                };
                push(ty, D3DDECLUSAGE_BLENDWEIGHT, 0);
            }
            if let Some(ty) = last_beta_ty {
                push(ty, D3DDECLUSAGE_BLENDINDICES, 0);
            }
        }
        D3DFVF_XYZRHW => push(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITIONT, 0),
        _ => {}
    }

    if fvf & D3DFVF_NORMAL != 0 {
        push(D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_NORMAL, 0);
    }
    if fvf & D3DFVF_PSIZE != 0 {
        push(D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_PSIZE, 0);
    }
    if fvf & D3DFVF_DIFFUSE != 0 {
        push(D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR, 0);
    }
    if fvf & D3DFVF_SPECULAR != 0 {
        push(D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR, 1);
    }

    let tex_count = u8::try_from(((fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT).min(8))
        .expect("clamped above to 8");
    for i in 0..tex_count {
        let size = (fvf >> (16 + u32::from(i) * 2)) & 0x3;
        let ty = match size {
            D3DFVF_TEXTUREFORMAT1 => D3DDECLTYPE_FLOAT1,
            D3DFVF_TEXTUREFORMAT3 => D3DDECLTYPE_FLOAT3,
            D3DFVF_TEXTUREFORMAT4 => D3DDECLTYPE_FLOAT4,
            // D3DFVF_TEXTUREFORMAT2 is value 0; falls through with the spec
            // default "2D coords" interpretation.
            _ => D3DDECLTYPE_FLOAT2,
        };
        push(ty, D3DDECLUSAGE_TEXCOORD, i);
    }

    // Fill offsets by laying elements out contiguously on stream 0.
    let mut offset: u16 = 0;
    for e in &mut elements {
        e.offset = offset;
        offset += u16::try_from(decl_type_to_metal_format(e.type_).1)
            .expect("Metal vertex format size ≤ 16 bytes");
    }
    (elements, u32::from(offset))
}

/// The Metal vertex attributes a declaration resolves to, plus what each stream it names needs.
///
/// Produced by [`resolve_attrs_for_vs`] / [`resolve_attrs_for_ff`] and
/// consumed by the draw path, which lays out one vertex buffer per used
/// stream and binds that stream's buffer at the Metal slot of the same index.
pub struct ResolvedAttrs {
    /// One entry per consumed element; `buffer_index` is the element's stream.
    pub attrs: Vec<VertexAttrDesc>,
    /// Per stream, `max(offset + size)` over the elements kept in `attrs`.
    ///
    /// Only kept elements count: Metal validates a layout's stride against
    /// the attributes in the descriptor, and an element the shader never
    /// consumes is not in it. Counting unconsumed tail fields would widen
    /// the fetch step past the stream's true stride and mis-fetch every
    /// vertex after the first — applications legitimately bind a packed
    /// buffer under a shared declaration whose trailing elements only exist
    /// for other shaders. Zero for a stream with no kept element.
    pub extents: [u32; MAX_STREAMS as usize],
    /// Bit `s` set: stream `s` feeds at least one consumed attribute.
    ///
    /// Only these streams get a layout and a binding; a used stream with no
    /// vertex buffer bound reads zeros through a constant layout.
    pub used_streams: u16,
}

/// Resolve a vertex declaration's elements against a programmable VS's input semantics.
///
/// The returned `attr_index` for each kept element is the VS `vN` register
/// bound to the matching `(usage, usage_index)`. Elements whose semantic the
/// VS does not consume are skipped silently — Metal accepts a descriptor that
/// declares more data than the shader reads.
#[must_use]
pub fn resolve_attrs_for_vs(
    elements: &[D3DVERTEXELEMENT9],
    semantics: &[InputSemantic],
) -> ResolvedAttrs {
    // VS semantic not declared by the shader: Metal accepts extra data, so
    // there is intentionally no warning for an unmatched element.
    resolve_attrs(elements, "programmable", |e| {
        lookup_semantic(semantics, e.usage, e.usage_index)
    })
}

/// Walk a declaration once, mapping each element to a Metal attribute through `attr_for`.
///
/// `attr_for` returns the `[[attribute(N)]]` slot for an element or `None`
/// to leave it out of the descriptor. Elements on a stream past the slot
/// table are dropped with a once-per-path warning; a declaration validator
/// only checks structure, so such a stream can reach a draw.
fn resolve_attrs(
    elements: &[D3DVERTEXELEMENT9],
    path: &'static str,
    mut attr_for: impl FnMut(&D3DVERTEXELEMENT9) -> Option<u16>,
) -> ResolvedAttrs {
    let mut attrs = Vec::with_capacity(elements.len());
    let mut extents = [0u32; MAX_STREAMS as usize];
    let mut used_streams: u16 = 0;
    for e in elements {
        let stream = u32::from(e.stream);
        if stream >= MAX_STREAMS {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "{path} vertex decl: element on stream={stream} past MaxStreams dropped"
            );
            continue;
        }
        let (format, size) = decl_type_to_metal_format(e.type_);
        if format == VertexFormat::Invalid {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "{path} vertex decl: type={} has no Metal format → element dropped",
                e.type_
            );
            continue;
        }
        if let Some(reg) = attr_for(e) {
            attrs.push(VertexAttrDesc {
                attr_index: u32::from(reg),
                buffer_index: stream,
                offset: u32::from(e.offset),
                format,
            });
            used_streams |= 1 << stream;
            let extent = &mut extents[stream as usize];
            *extent = (*extent).max(u32::from(e.offset) + size);
        }
    }
    ResolvedAttrs {
        attrs,
        extents,
        used_streams,
    }
}

/// Vertex-layout flags derived from a declaration element list, suitable for building an `FfVsKey`.
///
/// Derived from the element list rather than the FVF mask, so the
/// `SetVertexDeclaration` path (FVF = 0) and the FVF path agree on
/// `tex_coord_count`. A key built from the mask alone reports zero texcoord
/// sets for a declaration-driven draw, and the FF VS then emits
/// `out.texcoordN = float4(0.0)` for every varying — paired FF or
/// programmable pixel shaders would sample every texture at UV (0,0).
#[derive(Clone, Copy, Default)]
pub struct FfVsLayout {
    /// Boolean predicates derived from the vertex declaration.
    ///
    /// See `FfVsLayoutFlags` for bit semantics.
    pub flags: FfVsLayoutFlags,
    pub tex_coord_count: u8,
    /// Declared component count (1..=4) of each TEXCOORD set.
    ///
    /// Indexed by the element's `usage_index` (coord set). `0` means the set
    /// is not declared in the vertex stream. Drives the D3D9 fixed-function
    /// texture-coordinate transform expansion rule (a `FLOATn` texcoord pads
    /// component `n` to 1.0 before a `D3DTTFF_COUNT2..4` matrix multiply, and
    /// the projective-divide component defaults to `n - 1`).
    pub tex_coord_dims: [u8; 8],
    /// Number of float weights inferred from a BLENDWEIGHT element's type.
    ///
    /// FLOAT1 → 1, FLOAT2 → 2, FLOAT3 → 3, FLOAT4 / UBYTE4N → 4. Zero when no
    /// BLENDWEIGHT element is declared. Drives whether the FF VS emit needs
    /// the blending input attribute (slot 12).
    pub declared_weights_count: u8,
}

bitflags::bitflags! {
    /// Boolean predicates for `FfVsLayout`.
    ///
    /// Each bit mirrors the presence of one vertex declaration element kind.
    /// Transient builder state — not part of the shader-cache key.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FfVsLayoutFlags: u8 {
        /// Vertex declaration has a NORMAL element.
        const HAS_NORMAL = 1 << 0;
        /// Vertex declaration has a COLOR0 element.
        const HAS_COLOR0 = 1 << 1;
        /// Vertex declaration has a COLOR1 element.
        const HAS_COLOR1 = 1 << 2;
        /// Vertex declaration has a POSITIONT (XYZRHW) element.
        const HAS_RHW = 1 << 3;
        /// Vertex declaration has a BLENDINDICES element.
        ///
        /// Drives whether the FF VS emit needs the indexed-palette input
        /// attribute (slot 13).
        const DECLARED_INDICES = 1 << 4;
        /// The vertex format came from `SetVertexDeclaration`, not `SetFVF`.
        ///
        /// A COLORVERTEX material source pointing at a vertex colour the
        /// declaration omits reads 0 (FVF instead falls back to the material).
        const USES_VERTEX_DECL = 1 << 5;
        /// Vertex declaration has a PSIZE element (per-vertex point size).
        const HAS_PSIZE = 1 << 6;
    }
}

impl FfVsLayout {
    #[inline]
    #[must_use]
    pub const fn has_normal(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::HAS_NORMAL)
    }
    #[inline]
    #[must_use]
    pub const fn uses_vertex_decl(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::USES_VERTEX_DECL)
    }
    #[inline]
    #[must_use]
    pub const fn has_psize(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::HAS_PSIZE)
    }
    #[inline]
    #[must_use]
    pub const fn has_color0(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::HAS_COLOR0)
    }
    #[inline]
    #[must_use]
    pub const fn has_color1(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::HAS_COLOR1)
    }
    #[inline]
    #[must_use]
    pub const fn has_rhw(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::HAS_RHW)
    }
    #[inline]
    #[must_use]
    pub const fn declared_indices(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::DECLARED_INDICES)
    }
}

/// Derive the [`FfVsLayout`] flags from a vertex declaration's elements.
///
/// # Panics
///
/// Panics if `tex_coord_count` exceeds the `u8` range (clamped to ≤8 by the
/// loop, so unreachable).
pub fn ff_vs_layout_from_elements(elements: &[D3DVERTEXELEMENT9], uses_decl: bool) -> FfVsLayout {
    let mut flags = FfVsLayoutFlags::empty();
    flags.set(FfVsLayoutFlags::USES_VERTEX_DECL, uses_decl);
    let mut max_texcoord_index: Option<u8> = None;
    let mut tex_coord_dims = [0u8; 8];
    let mut declared_weights_count = 0u8;
    for e in elements {
        // Every stream the declaration names reaches the descriptor, so the
        // layout flags follow the elements regardless of stream: an element
        // on a stream nothing feeds reads zeros through that stream's
        // constant layout, which is what a D3D9 vertex reads there too. The
        // one exception is a stream past the slot table, which
        // `resolve_attrs_*` drops; its flags must drop with it or the FF VS
        // would declare an attribute the descriptor lacks.
        if u32::from(e.stream) >= MAX_STREAMS {
            continue;
        }
        match e.usage {
            u if u == D3DDECLUSAGE_NORMAL => flags.insert(FfVsLayoutFlags::HAS_NORMAL),
            u if u == D3DDECLUSAGE_COLOR && e.usage_index == 0 => {
                flags.insert(FfVsLayoutFlags::HAS_COLOR0);
            }
            u if u == D3DDECLUSAGE_COLOR && e.usage_index == 1 => {
                flags.insert(FfVsLayoutFlags::HAS_COLOR1);
            }
            u if u == D3DDECLUSAGE_TEXCOORD => {
                max_texcoord_index =
                    Some(max_texcoord_index.map_or(e.usage_index, |prev| prev.max(e.usage_index)));
                if (e.usage_index as usize) < tex_coord_dims.len() {
                    tex_coord_dims[e.usage_index as usize] = decl_type_dim(e.type_);
                }
            }
            u if u == D3DDECLUSAGE_POSITIONT => flags.insert(FfVsLayoutFlags::HAS_RHW),
            u if u == D3DDECLUSAGE_BLENDWEIGHT => {
                // FLOAT1 → 1, FLOAT2 → 2, FLOAT3 → 3, FLOAT4 / UBYTE4N → 4.
                // D3DDECLTYPE_FLOAT1 = 0, FLOAT2 = 1, FLOAT3 = 2, FLOAT4 = 3,
                // UBYTE4N = 8. Any other type is rare; default to 4 lanes.
                declared_weights_count = match e.type_ {
                    0 => 1,
                    1 => 2,
                    2 => 3,
                    _ => 4,
                };
            }
            u if u == D3DDECLUSAGE_BLENDINDICES => flags.insert(FfVsLayoutFlags::DECLARED_INDICES),
            u if u == D3DDECLUSAGE_PSIZE => flags.insert(FfVsLayoutFlags::HAS_PSIZE),
            _ => {}
        }
    }
    // D3D9 spec caps TEXCOORD usage_index at 7 (D3DDP_MAXTEXCOORD = 8).
    // FfVsKey's per-stage arrays (tci_modes, tci_coord_indices, tt_flags)
    // are sized [u8; 8]; a larger usage_index would index out of bounds on
    // the encoder thread. Clamp at the source and surface the offending raw
    // value once per distinct usage_index.
    let tex_coord_count = match max_texcoord_index {
        Some(m) if m >= 8 => {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(m),
                "ff_vs_layout: TEXCOORD usage_index {m} exceeds D3DDP_MAXTEXCOORD (8) — clamping"
            );
            8
        }
        Some(m) => m + 1,
        None => 0,
    };
    assert!(
        tex_coord_count <= 8,
        "ff_vs_layout_from_elements clamp violated: tex_coord_count={tex_coord_count}"
    );
    FfVsLayout {
        flags,
        tex_coord_count,
        tex_coord_dims,
        declared_weights_count,
    }
}

/// Same as [`resolve_attrs_for_vs`] but uses the FF VS's attribute convention.
///
/// See `crate::dxso::ff_attr_index_for_semantic`. The FF VS has no `dcl_*`
/// declarations — its input layout is fixed.
#[must_use]
pub fn resolve_attrs_for_ff(elements: &[D3DVERTEXELEMENT9]) -> ResolvedAttrs {
    resolve_attrs(elements, "FF", |e| {
        let reg = ff_attr_index_for_semantic(e.usage, e.usage_index);
        if reg.is_none() {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: (u64::from(e.usage) << 8) | u64::from(e.usage_index),
                "FF vertex decl: usage={} usage_index={} has no attribute register → element dropped",
                e.usage,
                e.usage_index
            );
        }
        reg
    })
}

/// Convenience: hash a contiguous `&[D3DVERTEXELEMENT9]` for use as a pipeline-cache key.
///
/// The element array uniquely identifies a vertex layout; two decls with the
/// same elements produce the same hash.
#[must_use]
pub fn hash_elements(elements: &[D3DVERTEXELEMENT9]) -> u64 {
    let mut h = Xxh3::new();
    for e in elements {
        e.hash(&mut h);
    }
    h.finish()
}

/// Element terminator check matching `D3DDECL_END()`: `stream == 0xFF`.
#[must_use]
pub const fn is_decl_end(e: &D3DVERTEXELEMENT9) -> bool {
    e.stream == D3DDECL_END_STREAM
}

/// A vertex declaration as stored by `CreateVertexDeclaration`.
pub struct PackedVertexDecl {
    /// The elements the game passed, `D3DDECL_END` terminator included.
    pub elements_with_end: Vec<D3DVERTEXELEMENT9>,
    /// `hash_elements` over the real elements; the pipeline-cache identity.
    pub hash: u64,
    /// Bit `s` set: some element lives on stream `s`.
    ///
    /// Lets the draw path pick the streams to snapshot without walking the
    /// elements per draw. Streams past the slot table contribute no bit.
    pub stream_mask: u16,
}

/// Validate + pack the raw element slice a game passes to `CreateVertexDeclaration`.
///
/// Returns `None` only if the slice has no terminator: D3D9 validates
/// structure, not which streams the layout spans, and callers rely on a
/// valid object back so their own `Release(decl)` doesn't fault. The `stream`
/// field is part of each element's hash, so layouts that differ only by
/// stream stay distinct in the pipeline cache.
pub fn pack_vertex_decl(elements: &[D3DVERTEXELEMENT9]) -> Option<PackedVertexDecl> {
    let end_pos = elements.iter().position(is_decl_end)?;
    let mut packed = Vec::with_capacity(end_pos + 1);
    packed.extend_from_slice(&elements[..=end_pos]);
    let hash = hash_elements(&packed[..end_pos]);
    let stream_mask = packed[..end_pos]
        .iter()
        .filter(|e| u32::from(e.stream) < MAX_STREAMS)
        .fold(0u16, |m, e| m | (1 << e.stream));
    Some(PackedVertexDecl {
        elements_with_end: packed,
        hash,
        stream_mask,
    })
}

/// Map a `D3DDECLTYPE` byte to the float component count of a fixed-function texcoord set.
///
/// `FLOAT1..4` are the overwhelmingly common texcoord types; the
/// packed/normalized integer types are mapped to their natural lane width so
/// the transform expansion rule sees a sensible dimension.
/// `D3DDECLTYPE_UNUSED` (and anything unrecognised) maps to 0.
const fn decl_type_dim(type_: u8) -> u8 {
    match type_ {
        0 => 1,                                // FLOAT1
        1 | 6 | 9 | 11 | 15 => 2,              // FLOAT2, SHORT2(N), USHORT2N, FLOAT16_2
        2 | 13 | 14 => 3,                      // FLOAT3, UDEC3, DEC3N
        3 | 4 | 5 | 7 | 8 | 10 | 12 | 16 => 4, // FLOAT4, D3DCOLOR, UBYTE4(N), SHORT4(N), USHORT4N, FLOAT16_4
        _ => 0,                                // UNUSED / unknown
    }
}

fn lookup_semantic(semantics: &[InputSemantic], usage: u8, usage_index: u8) -> Option<u16> {
    semantics
        .iter()
        .find(|s| decl_usage_to_byte(s.usage) == usage && s.usage_index == usage_index)
        .map(|s| s.register_index)
}

const fn decl_usage_to_byte(u: crate::dxso::DeclUsage) -> u8 {
    match u {
        crate::dxso::DeclUsage::Position => D3DDECLUSAGE_POSITION,
        crate::dxso::DeclUsage::BlendWeight => D3DDECLUSAGE_BLENDWEIGHT,
        crate::dxso::DeclUsage::BlendIndices => D3DDECLUSAGE_BLENDINDICES,
        crate::dxso::DeclUsage::Normal => D3DDECLUSAGE_NORMAL,
        crate::dxso::DeclUsage::PSize => D3DDECLUSAGE_PSIZE,
        crate::dxso::DeclUsage::Texcoord => D3DDECLUSAGE_TEXCOORD,
        crate::dxso::DeclUsage::Tangent => 6,
        crate::dxso::DeclUsage::Binormal => 7,
        crate::dxso::DeclUsage::TessFactor => 8,
        crate::dxso::DeclUsage::PositionT => D3DDECLUSAGE_POSITIONT,
        crate::dxso::DeclUsage::Color => D3DDECLUSAGE_COLOR,
        crate::dxso::DeclUsage::Fog => 11,
        crate::dxso::DeclUsage::Depth => 12,
        crate::dxso::DeclUsage::Sample => 13,
    }
}

#[cfg(test)]
mod tests;
