use std::hash::{Hash, Hasher};

use mtld3d_shared::{
    VertexAttrDesc,
    mtl::{
        AddressMode, BlendFactor, BlendOperation, ColorWriteMask, CompareFunc, CullMode,
        MinMagFilter, MipFilter, PrimitiveType, VertexFormat,
    },
};
use mtld3d_types::{
    D3DBLEND_BLENDFACTOR, D3DBLEND_DESTALPHA, D3DBLEND_DESTCOLOR, D3DBLEND_INVBLENDFACTOR,
    D3DBLEND_INVDESTALPHA, D3DBLEND_INVDESTCOLOR, D3DBLEND_INVSRCALPHA, D3DBLEND_INVSRCCOLOR,
    D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_SRCALPHASAT, D3DBLEND_SRCCOLOR, D3DBLEND_ZERO,
    D3DBLENDOP_ADD, D3DBLENDOP_MAX, D3DBLENDOP_MIN, D3DBLENDOP_REVSUBTRACT, D3DBLENDOP_SUBTRACT,
    D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_GREATER, D3DCMP_GREATEREQUAL, D3DCMP_LESS,
    D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCMP_NOTEQUAL, D3DCULL_CCW, D3DCULL_CW, D3DCULL_NONE,
    D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_DEC3N, D3DDECLTYPE_FLOAT1,
    D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4, D3DDECLTYPE_FLOAT16_2,
    D3DDECLTYPE_FLOAT16_4, D3DDECLTYPE_SHORT2, D3DDECLTYPE_SHORT2N, D3DDECLTYPE_SHORT4,
    D3DDECLTYPE_SHORT4N, D3DDECLTYPE_UBYTE4, D3DDECLTYPE_UBYTE4N, D3DDECLTYPE_UDEC3,
    D3DDECLTYPE_USHORT2N, D3DDECLTYPE_USHORT4N, D3DDECLUSAGE_BLENDINDICES,
    D3DDECLUSAGE_BLENDWEIGHT, D3DDECLUSAGE_COLOR, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION,
    D3DDECLUSAGE_POSITIONT, D3DDECLUSAGE_PSIZE, D3DDECLUSAGE_TEXCOORD, D3DFMT_A8R8G8B8,
    D3DFMT_R5G6B5, D3DFMT_R32F, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_LASTBETA_D3DCOLOR,
    D3DFVF_LASTBETA_UBYTE4, D3DFVF_NORMAL, D3DFVF_POSITION_MASK, D3DFVF_PSIZE, D3DFVF_SPECULAR,
    D3DFVF_TEXCOUNT_MASK, D3DFVF_TEXCOUNT_SHIFT, D3DFVF_TEXTUREFORMAT1, D3DFVF_TEXTUREFORMAT3,
    D3DFVF_TEXTUREFORMAT4, D3DFVF_XYZ, D3DFVF_XYZB1, D3DFVF_XYZB2, D3DFVF_XYZB3, D3DFVF_XYZB4,
    D3DFVF_XYZB5, D3DFVF_XYZRHW, D3DFVF_XYZW, D3DPT_LINELIST, D3DPT_LINESTRIP, D3DPT_POINTLIST,
    D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP, D3DTADDRESS_BORDER, D3DTADDRESS_CLAMP,
    D3DTADDRESS_MIRROR, D3DTADDRESS_MIRRORONCE, D3DTADDRESS_WRAP, D3DTEXF_ANISOTROPIC,
    D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT, D3DVERTEXELEMENT9,
};

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
        // 16-bit packed R5G6B5: top 5 bits of red, top 6 of green, top 5 of
        // blue. Little-endian 2-byte value (e.g. 0xdeadbeef → 0xadfd).
        D3DFMT_R5G6B5 => {
            let packed =
                ((u16::from(r) >> 3) << 11) | ((u16::from(g) >> 2) << 5) | (u16::from(b) >> 3);
            Some(packed.to_le_bytes().to_vec())
        }
        // Single 32-bit float carrying the red channel normalised to [0, 1].
        D3DFMT_R32F => Some((f32::from(r) / 255.0).to_le_bytes().to_vec()),
        _ => None,
    }
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

/// D3DTADDRESS_* → Metal sampler address mode.
pub fn d3d_to_metal_address_mode(d3d_mode: u32) -> AddressMode {
    match d3d_mode {
        D3DTADDRESS_WRAP => AddressMode::Repeat,
        D3DTADDRESS_MIRROR => AddressMode::MirrorRepeat,
        D3DTADDRESS_CLAMP => AddressMode::ClampToEdge,
        D3DTADDRESS_BORDER => AddressMode::ClampToZero,
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
        D3DPT_POINTLIST => {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "DrawPrimitive(D3DPT_POINTLIST) — Metal renders as 1-pixel points; D3DRS_POINTSIZE / D3DRS_POINTSPRITEENABLE not honored"
            );
            Some(PrimitiveType::Point)
        }
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

/// Expand a triangle-**fan** vertex stream into a triangle **list**.
///
/// Metal has no triangle-fan primitive. A fan of `primitive_count + 2`
/// vertices makes `primitive_count` triangles, where triangle `i` is fan
/// vertices `0, i+1, i+2`. `src` holds the fan vertices back-to-back at
/// `stride` bytes each (at least `(primitive_count + 2) * stride` bytes);
/// the returned buffer holds `primitive_count * 3` vertices ready for a
/// `PrimitiveType::Triangle` draw.
#[must_use]
pub fn expand_triangle_fan(src: &[u8], stride: usize, primitive_count: u32) -> Vec<u8> {
    let pc = primitive_count as usize;
    let mut out = Vec::with_capacity(pc.saturating_mul(3).saturating_mul(stride));
    let vertex = |i: usize| &src[i * stride..(i + 1) * stride];
    for i in 0..pc {
        out.extend_from_slice(vertex(0));
        out.extend_from_slice(vertex(i + 1));
        out.extend_from_slice(vertex(i + 2));
    }
    out
}

/// Compute vertex count from D3D9 primitive type and primitive count.
pub fn vertex_count(d3d_type: u32, primitive_count: u32) -> u32 {
    match d3d_type {
        D3DPT_POINTLIST => primitive_count,         // point list
        D3DPT_LINELIST => primitive_count * 2,      // line list
        D3DPT_LINESTRIP => primitive_count + 1,     // line strip
        D3DPT_TRIANGLELIST => primitive_count * 3,  // triangle list
        D3DPT_TRIANGLESTRIP => primitive_count + 2, // triangle strip
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
            method: 0, // D3DDECLMETHOD_DEFAULT
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

/// Resolve a vertex declaration's elements against a programmable VS's input semantics.
///
/// The returned `attr_index` for each kept element is the VS `vN` register
/// bound to the matching `(usage, usage_index)`. Elements whose semantic the
/// VS does not consume are skipped silently — Metal accepts a descriptor that
/// declares more data than the shader reads.
///
/// `stride` is the maximum `offset + element_size` across *all* elements
/// on stream 0, including skipped ones: the stride is a property of the
/// vertex buffer layout, not the VS.
pub fn resolve_attrs_for_vs(
    elements: &[D3DVERTEXELEMENT9],
    semantics: &[InputSemantic],
) -> (Vec<VertexAttrDesc>, u32) {
    let mut attrs = Vec::with_capacity(semantics.len());
    let mut stride: u32 = 0;
    for e in elements {
        if e.stream != 0 {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "programmable vertex decl: element on stream={} dropped (multi-stream unsupported)",
                e.stream
            );
            continue;
        }
        let (format, size) = decl_type_to_metal_format(e.type_);
        stride = stride.max(u32::from(e.offset) + size);
        if format == VertexFormat::Invalid {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "programmable vertex decl: type={} has no Metal format → element dropped",
                e.type_
            );
            continue;
        }
        if let Some(reg) = lookup_semantic(semantics, e.usage, e.usage_index) {
            attrs.push(VertexAttrDesc {
                attr_index: u32::from(reg),
                buffer_index: 0,
                offset: u32::from(e.offset),
                format,
            });
        }
        // VS semantic not declared by the shader: Metal accepts extra data,
        // so we intentionally do NOT warn — this is expected.
    }
    (attrs, stride)
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
        /// A COLOR0 (diffuse) element is declared on a non-zero, unbound stream.
        ///
        /// The element is dropped from the descriptor, and no stream-0 COLOR0
        /// exists — so the unlit FF VS must output black, not the white
        /// default.
        const DIFFUSE_DECLARED_UNBOUND = 1 << 5;
        /// The vertex format came from `SetVertexDeclaration`, not `SetFVF`.
        ///
        /// A COLORVERTEX material source pointing at a vertex colour the
        /// declaration omits reads 0 (FVF instead falls back to the material).
        const USES_VERTEX_DECL = 1 << 6;
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
    pub const fn has_diffuse_declared_unbound(&self) -> bool {
        self.flags
            .contains(FfVsLayoutFlags::DIFFUSE_DECLARED_UNBOUND)
    }
    #[inline]
    #[must_use]
    pub const fn uses_vertex_decl(&self) -> bool {
        self.flags.contains(FfVsLayoutFlags::USES_VERTEX_DECL)
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
        // Only stream 0 is rendered (single-stream architecture), and
        // `resolve_attrs_for_ff` drops non-zero streams from the vertex
        // descriptor. The layout flags must agree with the attributes the
        // descriptor actually carries — otherwise the FF VS declares an
        // `[[attribute(N)]]` (e.g. a stream-1 COLOR) that the descriptor lacks
        // and Metal drops the draw. A material colour source bound to a dropped
        // stream then correctly falls back to the material constant.
        if e.stream != 0 {
            // A diffuse (COLOR0) declared on an unbound stream is dropped from
            // the descriptor, but the vertex reads 0 from it — so the unlit FF
            // VS must output black, not the white "absent diffuse" default.
            // A stream-0 COLOR0 (`HAS_COLOR0`) still takes precedence in the
            // emit.
            if e.usage == D3DDECLUSAGE_COLOR && e.usage_index == 0 {
                flags.insert(FfVsLayoutFlags::DIFFUSE_DECLARED_UNBOUND);
            }
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
pub fn resolve_attrs_for_ff(elements: &[D3DVERTEXELEMENT9]) -> (Vec<VertexAttrDesc>, u32) {
    let mut attrs = Vec::with_capacity(elements.len());
    let mut stride: u32 = 0;
    for e in elements {
        if e.stream != 0 {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "FF vertex decl: element on stream={} dropped (multi-stream unsupported)",
                e.stream
            );
            continue;
        }
        let (format, size) = decl_type_to_metal_format(e.type_);
        stride = stride.max(u32::from(e.offset) + size);
        if format == VertexFormat::Invalid {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "FF vertex decl: type={} has no Metal format → element dropped",
                e.type_
            );
            continue;
        }
        if let Some(reg) = ff_attr_index_for_semantic(e.usage, e.usage_index) {
            attrs.push(VertexAttrDesc {
                attr_index: u32::from(reg),
                buffer_index: 0,
                offset: u32::from(e.offset),
                format,
            });
        } else {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: (u64::from(e.usage) << 8) | u64::from(e.usage_index),
                "FF vertex decl: usage={} usage_index={} has no attribute register → element dropped",
                e.usage,
                e.usage_index
            );
        }
    }
    (attrs, stride)
}

/// Convenience: hash a contiguous `&[D3DVERTEXELEMENT9]` for use as a pipeline-cache key.
///
/// The element array uniquely identifies a vertex layout; two decls with the
/// same elements produce the same hash.
#[must_use]
pub fn hash_elements(elements: &[D3DVERTEXELEMENT9]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
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

/// Validate + pack the raw element slice a game passes to `CreateVertexDeclaration`.
///
/// Returns the full element array *including* the `D3DDECL_END` terminator
/// plus the precomputed hash (see `hash_elements`) on success. Returns `None`
/// if the slice has no terminator or if any real element uses a non-zero
/// stream (multi-stream is not supported).
pub fn pack_vertex_decl(elements: &[D3DVERTEXELEMENT9]) -> Option<(Vec<D3DVERTEXELEMENT9>, u64)> {
    let end_pos = elements.iter().position(is_decl_end)?;
    // Multi-stream declarations are *accepted*: D3D9 `CreateVertexDeclaration`
    // validates only structure, not how many streams the layout spans, and
    // callers rely on a valid object back so their own `Release(decl)`
    // doesn't fault. We still only render stream 0 —
    // `resolve_attrs_for_vs` / the FF path drop elements on other streams — so
    // multi-stream draws are wrong, not crashes. The `stream` field is part of
    // each element's hash, so layouts that differ only by stream stay distinct
    // in the attrs cache.
    let mut packed = Vec::with_capacity(end_pos + 1);
    packed.extend_from_slice(&elements[..=end_pos]);
    let hash = hash_elements(&packed[..end_pos]);
    Some((packed, hash))
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
mod tests {
    use super::*;
    use crate::dxso::DeclUsage;

    #[test]
    fn triangle_fan_expands_to_list() {
        // 5 fan vertices (1 byte each) → 3 triangles: (0,1,2),(0,2,3),(0,3,4).
        let src = [10u8, 11, 12, 13, 14];
        let out = expand_triangle_fan(&src, 1, 3);
        assert_eq!(out, vec![10, 11, 12, 10, 12, 13, 10, 13, 14]);
    }

    #[test]
    fn triangle_fan_respects_stride() {
        // 4 vertices of 2 bytes → 2 triangles: (0,1,2),(0,2,3).
        let src = [0u8, 0, 1, 1, 2, 2, 3, 3];
        let out = expand_triangle_fan(&src, 2, 2);
        assert_eq!(out, vec![0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 3, 3]);
    }

    fn pos3() -> D3DVERTEXELEMENT9 {
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        }
    }

    fn tex0(offset: u16) -> D3DVERTEXELEMENT9 {
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset,
            type_: D3DDECLTYPE_FLOAT2,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 0,
        }
    }

    fn to_bits4(arr: [f32; 4]) -> [u32; 4] {
        [
            arr[0].to_bits(),
            arr[1].to_bits(),
            arr[2].to_bits(),
            arr[3].to_bits(),
        ]
    }

    #[test]
    fn d3dcolor_to_rgba_default_is_white() {
        // D3DRS_BLENDFACTOR's default is 0xFFFFFFFF (opaque white).
        let rgba = d3dcolor_to_rgba_f32(0xFFFF_FFFF);
        assert_eq!(to_bits4(rgba), to_bits4([1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn d3dcolor_to_rgba_zero_is_transparent_black() {
        let rgba = d3dcolor_to_rgba_f32(0x0000_0000);
        assert_eq!(to_bits4(rgba), to_bits4([0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn d3dcolor_to_rgba_argb_byte_order() {
        // 0xAARRGGBB. A=0x80, R=0x40, G=0x20, B=0x10. The u8→f32 path
        // is exact (each byte fits f32 mantissa), so bit-equality holds.
        let rgba = d3dcolor_to_rgba_f32(0x8040_2010);
        assert_eq!(rgba[0].to_bits(), (f32::from(0x40u8) / 255.0).to_bits());
        assert_eq!(rgba[1].to_bits(), (f32::from(0x20u8) / 255.0).to_bits());
        assert_eq!(rgba[2].to_bits(), (f32::from(0x10u8) / 255.0).to_bits());
        assert_eq!(rgba[3].to_bits(), (f32::from(0x80u8) / 255.0).to_bits());
    }

    #[test]
    fn color_fill_a8r8g8b8_roundtrips_the_d3dcolor() {
        // BGRA8 bytes read back as the same D3DCOLOR: filling 0xdeadbeef must
        // read back 0xdeadbeef.
        let bytes = d3dcolor_fill_pixel_bytes(0xdead_beef, D3DFMT_A8R8G8B8).unwrap();
        assert_eq!(bytes, vec![0xef, 0xbe, 0xad, 0xde]);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0xdead_beef
        );
    }

    #[test]
    fn color_fill_r32f_is_red_channel_normalized() {
        // R=0xad → 0xad/255.0: ColorFill promotes the red byte to a
        // normalized float.
        let bytes = d3dcolor_fill_pixel_bytes(0x00ad_0000, D3DFMT_R32F).unwrap();
        let f = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(f.to_bits(), (f32::from(0xadu8) / 255.0).to_bits());
    }

    #[test]
    fn color_fill_r5g6b5_packs_top_bits() {
        // Filling 0xdeadbeef into an R5G6B5 surface packs to the 16-bit value
        // 0xadfd (R=0xad>>3, G=0xbe>>2, B=0xef>>3).
        let bytes = d3dcolor_fill_pixel_bytes(0xdead_beef, D3DFMT_R5G6B5).unwrap();
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0xadfd);
    }

    #[test]
    fn color_fill_unsupported_format_is_none() {
        // Block / expanded / unmapped formats aren't encoded yet.
        assert!(d3dcolor_fill_pixel_bytes(0xffff_ffff, D3DFMT_X8R8G8B8).is_some());
        assert!(d3dcolor_fill_pixel_bytes(0xffff_ffff, 0x0000_0000).is_none());
    }

    #[test]
    fn decl_type_to_metal_format_table() {
        // Each D3DDECLTYPE we support maps to a typed VertexFormat and a
        // size. If anyone flips a mapping here without updating both sides,
        // this catches it.
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT1),
            (VertexFormat::Float, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT2),
            (VertexFormat::Float2, 8)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT3),
            (VertexFormat::Float3, 12)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT4),
            (VertexFormat::Float4, 16)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_D3DCOLOR),
            (VertexFormat::UChar4NormalizedBgra, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_UBYTE4),
            (VertexFormat::UChar4, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_UBYTE4N),
            (VertexFormat::UChar4Normalized, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_SHORT2),
            (VertexFormat::Short2, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_SHORT4),
            (VertexFormat::Short4, 8)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_SHORT2N),
            (VertexFormat::Short2Normalized, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_SHORT4N),
            (VertexFormat::Short4Normalized, 8)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_USHORT2N),
            (VertexFormat::UShort2Normalized, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_USHORT4N),
            (VertexFormat::UShort4Normalized, 8)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT16_2),
            (VertexFormat::Half2, 4)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_FLOAT16_4),
            (VertexFormat::Half4, 8)
        );
        // Unsupported types report INVALID so the caller can skip.
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_UDEC3),
            (VertexFormat::Invalid, 0)
        );
        assert_eq!(
            decl_type_to_metal_format(D3DDECLTYPE_DEC3N),
            (VertexFormat::Invalid, 0)
        );
    }

    #[test]
    fn fvf_synthesize_elements_position_normal_tex1() {
        let (elems, stride) = fvf_to_elements(D3DFVF_XYZ | D3DFVF_NORMAL | (1 << 8));
        assert_eq!(elems.len(), 3);
        assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITION);
        assert_eq!(elems[0].type_, D3DDECLTYPE_FLOAT3);
        assert_eq!(elems[0].offset, 0);
        assert_eq!(elems[1].usage, D3DDECLUSAGE_NORMAL);
        assert_eq!(elems[1].offset, 12);
        assert_eq!(elems[2].usage, D3DDECLUSAGE_TEXCOORD);
        assert_eq!(elems[2].usage_index, 0);
        assert_eq!(elems[2].offset, 24);
        assert_eq!(stride, 32);
    }

    #[test]
    fn fvf_synthesize_elements_xyzrhw_diffuse_tex1() {
        let (elems, stride) = fvf_to_elements(D3DFVF_XYZRHW | D3DFVF_DIFFUSE | (1 << 8));
        assert_eq!(elems.len(), 3);
        assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITIONT);
        assert_eq!(elems[0].type_, D3DDECLTYPE_FLOAT4);
        assert_eq!(elems[1].usage, D3DDECLUSAGE_COLOR);
        assert_eq!(elems[1].usage_index, 0);
        assert_eq!(elems[1].type_, D3DDECLTYPE_D3DCOLOR);
        assert_eq!(elems[1].offset, 16);
        assert_eq!(elems[2].usage, D3DDECLUSAGE_TEXCOORD);
        assert_eq!(elems[2].offset, 20);
        assert_eq!(stride, 28);
    }

    #[test]
    fn fvf_to_elements_matches_d3d9_blend_matrix() {
        // Each row is (type_, usage, usage_index, offset); the table maps an
        // fvf to its expected element rows via the canonical D3D9 FVF ->
        // declaration conversion. Covers every XYZBn / LASTBETA combination,
        // including the XYZB2|D3DCOLOR quirk (weight = D3DCOLOR, index =
        // UBYTE4).
        type Row = (u8, u8, u8, u16);
        let cases: &[(u32, &[Row])] = &[
            (
                D3DFVF_XYZ,
                &[(D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0)],
            ),
            (
                D3DFVF_XYZW,
                &[(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITION, 0, 0)],
            ),
            (
                D3DFVF_XYZRHW,
                &[(D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_POSITIONT, 0, 0)],
            ),
            (
                D3DFVF_XYZB1,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB1 | D3DFVF_LASTBETA_UBYTE4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB1 | D3DFVF_LASTBETA_D3DCOLOR,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB2,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB2 | D3DFVF_LASTBETA_UBYTE4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 16),
                ],
            ),
            (
                D3DFVF_XYZB2 | D3DFVF_LASTBETA_D3DCOLOR,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 16),
                ],
            ),
            (
                D3DFVF_XYZB3,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB3 | D3DFVF_LASTBETA_UBYTE4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 20),
                ],
            ),
            (
                D3DFVF_XYZB3 | D3DFVF_LASTBETA_D3DCOLOR,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT2, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 20),
                ],
            ),
            (
                D3DFVF_XYZB4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                ],
            ),
            (
                D3DFVF_XYZB4 | D3DFVF_LASTBETA_UBYTE4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 24),
                ],
            ),
            (
                D3DFVF_XYZB4 | D3DFVF_LASTBETA_D3DCOLOR,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 24),
                ],
            ),
            (
                D3DFVF_XYZB5,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_FLOAT1, D3DDECLUSAGE_BLENDINDICES, 0, 28),
                ],
            ),
            (
                D3DFVF_XYZB5 | D3DFVF_LASTBETA_UBYTE4,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_UBYTE4, D3DDECLUSAGE_BLENDINDICES, 0, 28),
                ],
            ),
            (
                D3DFVF_XYZB5 | D3DFVF_LASTBETA_D3DCOLOR,
                &[
                    (D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION, 0, 0),
                    (D3DDECLTYPE_FLOAT4, D3DDECLUSAGE_BLENDWEIGHT, 0, 12),
                    (D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_BLENDINDICES, 0, 28),
                ],
            ),
        ];
        for (fvf, expected) in cases {
            let (elems, _stride) = fvf_to_elements(*fvf);
            assert_eq!(
                elems.len(),
                expected.len(),
                "element count for fvf {fvf:#x}"
            );
            for (i, (ty, usage, usage_index, offset)) in expected.iter().enumerate() {
                assert_eq!(elems[i].type_, *ty, "type fvf {fvf:#x} elem {i}");
                assert_eq!(elems[i].usage, *usage, "usage fvf {fvf:#x} elem {i}");
                assert_eq!(
                    elems[i].usage_index, *usage_index,
                    "usage_index fvf {fvf:#x} elem {i}"
                );
                assert_eq!(elems[i].offset, *offset, "offset fvf {fvf:#x} elem {i}");
                assert_eq!(elems[i].stream, 0, "stream fvf {fvf:#x} elem {i}");
                assert_eq!(elems[i].method, 0, "method fvf {fvf:#x} elem {i}");
            }
        }
    }

    #[test]
    fn fvf_synthesize_elements_xyzb3() {
        let (elems, stride) = fvf_to_elements(D3DFVF_XYZB3);
        // XYZB3 with no LASTBETA flag: 3 floats position + 3 blend weights.
        assert_eq!(elems.len(), 2);
        assert_eq!(elems[0].usage, D3DDECLUSAGE_POSITION);
        assert_eq!(elems[1].usage, D3DDECLUSAGE_BLENDWEIGHT);
        assert_eq!(elems[1].type_, D3DDECLTYPE_FLOAT3);
        assert_eq!(stride, 24);
    }

    #[test]
    fn resolve_attrs_for_vs_swaps_register_indices() {
        // VS declares position on v2 and texcoord0 on v7 — the resolved
        // attr_index must match the register, not the FVF convention.
        let semantics = vec![
            InputSemantic {
                usage: DeclUsage::Position,
                usage_index: 0,
                register_index: 2,
            },
            InputSemantic {
                usage: DeclUsage::Texcoord,
                usage_index: 0,
                register_index: 7,
            },
        ];
        let elems = [pos3(), tex0(12)];
        let (attrs, stride) = resolve_attrs_for_vs(&elems, &semantics);
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].attr_index, 2);
        assert_eq!(attrs[1].attr_index, 7);
        assert_eq!(stride, 20);
    }

    #[test]
    fn resolve_attrs_skips_unused_semantics() {
        // VS declares only POSITION; NORMAL in the decl is silently dropped.
        let semantics = vec![InputSemantic {
            usage: DeclUsage::Position,
            usage_index: 0,
            register_index: 0,
        }];
        let elems = [
            pos3(),
            D3DVERTEXELEMENT9 {
                stream: 0,
                offset: 12,
                type_: D3DDECLTYPE_FLOAT3,
                method: 0,
                usage: D3DDECLUSAGE_NORMAL,
                usage_index: 0,
            },
        ];
        let (attrs, stride) = resolve_attrs_for_vs(&elems, &semantics);
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].attr_index, 0);
        // Stride still covers the normal element so the vertex buffer
        // layout is correct even with an unused attribute.
        assert_eq!(stride, 24);
    }

    #[test]
    fn resolve_attrs_for_ff_matches_ff_convention() {
        // POSITION → attr(0), TEXCOORD0 → attr(4). Must agree with
        // `crate::dxso::ff_attr_index_for_semantic`.
        let elems = [pos3(), tex0(12)];
        let (attrs, stride) = resolve_attrs_for_ff(&elems);
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs[0].attr_index, 0);
        assert_eq!(attrs[1].attr_index, 4);
        assert_eq!(stride, 20);
    }

    fn end() -> D3DVERTEXELEMENT9 {
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: mtld3d_types::D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        }
    }

    #[test]
    fn pack_vertex_decl_hash_stable_across_calls() {
        let elems = [pos3(), tex0(12), end()];
        let (_, h_a) = pack_vertex_decl(&elems).expect("pack a");
        let (_, h_b) = pack_vertex_decl(&elems).expect("pack b");
        assert_eq!(h_a, h_b);
        let swapped = [pos3(), tex0(16), end()];
        let (_, h_c) = pack_vertex_decl(&swapped).expect("pack c");
        assert_ne!(h_a, h_c);
    }

    #[test]
    fn pack_vertex_decl_accepts_multi_stream_distinct_hash() {
        // A non-zero stream is accepted (D3D9 creation succeeds; we render
        // only stream 0). Two layouts that differ *only* by stream must hash
        // differently so the attrs cache keeps them apart.
        let on_stream = |stream| D3DVERTEXELEMENT9 {
            stream,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        };
        let a = pack_vertex_decl(&[on_stream(0), end()]).expect("stream 0 accepted");
        let b = pack_vertex_decl(&[on_stream(1), end()]).expect("stream 1 accepted");
        assert_ne!(a.1, b.1, "stream must participate in the decl hash");
    }

    #[test]
    fn pack_vertex_decl_requires_terminator() {
        assert!(pack_vertex_decl(&[pos3()]).is_none());
    }

    #[test]
    fn pack_vertex_decl_preserves_terminator_in_output() {
        let elems = [pos3(), tex0(12), end()];
        let (packed, _) = pack_vertex_decl(&elems).expect("pack");
        assert_eq!(packed.len(), 3);
        assert_eq!(packed.last().unwrap().stream, D3DDECL_END_STREAM);
    }

    #[test]
    fn ff_vs_layout_clamps_tex_coord_count_to_8() {
        // A vertex declaration that claims TEXCOORD at usage_index = 12
        // must not produce tex_coord_count > 8 — FfVsKey's per-stage
        // arrays are [u8; 8] and OOB-crashed the encoder thread.
        let elements = [
            pos3(),
            D3DVERTEXELEMENT9 {
                stream: 0,
                offset: 12,
                type_: D3DDECLTYPE_FLOAT2,
                method: 0,
                usage: D3DDECLUSAGE_TEXCOORD,
                usage_index: 12,
            },
        ];
        let layout = ff_vs_layout_from_elements(&elements, false);
        assert_eq!(layout.tex_coord_count, 8);
    }

    #[test]
    fn ff_vs_layout_in_spec_usage_index_7_yields_8() {
        let elements = [
            pos3(),
            D3DVERTEXELEMENT9 {
                stream: 0,
                offset: 12,
                type_: D3DDECLTYPE_FLOAT2,
                method: 0,
                usage: D3DDECLUSAGE_TEXCOORD,
                usage_index: 7,
            },
        ];
        let layout = ff_vs_layout_from_elements(&elements, false);
        assert_eq!(layout.tex_coord_count, 8);
    }

    #[test]
    fn ff_vs_layout_single_tex0_yields_1() {
        let layout = ff_vs_layout_from_elements(&[pos3(), tex0(12)], false);
        assert_eq!(layout.tex_coord_count, 1);
    }

    #[test]
    fn d3d_depth_bias_zero_passes_through() {
        // D3DRS_DEPTHBIAS default is 0.0 (u32 0). Most draws don't touch
        // it — the scaled output must stay exactly zero so games that
        // never write the state see no rasterizer offset.
        assert_eq!(d3d_depth_bias_to_metal(0).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn d3d_depth_bias_scales_by_two_pow_23() {
        // D3D9 spec: 1 ULP at the depth resolution. Metal's setDepthBias
        // takes the value in absolute float units of the depth format.
        // mtld3d's depth always resolves to Depth32Float (mantissa = 23
        // bits), so the scale is 2^23.
        let raw = 1.0f32.to_bits();
        let scaled = d3d_depth_bias_to_metal(raw);
        // 2^23 = 8_388_608.0 is exactly representable in f32; bit-equality holds.
        assert_eq!(scaled.to_bits(), 8_388_608.0_f32.to_bits());
    }

    #[test]
    fn d3d_depth_bias_negative_pushes_toward_camera() {
        // Negative bias is the canonical decal-pull-forward direction.
        // Sign must be preserved through the scale.
        // raw = -1.0 / 2^23 → scale × raw = -1.0
        let raw = (-(1.0_f32 / 8_388_608.0_f32)).to_bits();
        let scaled = d3d_depth_bias_to_metal(raw);
        assert!((scaled - -1.0).abs() < 1e-6);
    }

    #[test]
    fn looks_like_decal_fires_on_alpha_blended_no_bias() {
        // Canonical decal pattern: depth-test on, depth-write off,
        // alpha-blend on, game's DEPTHBIAS + SLOPESCALEDEPTHBIAS both
        // zero. Predicate fires → caller substitutes
        // IMPLICIT_DECAL_BIAS_RAW for the zero game bias.
        let inputs = DecalHeuristicInputs {
            depth_enable: 1,
            depth_write: 0,
            blend_enable: 1,
            raw_depth_bias: 0,
            raw_slope_scale: 0,
        };
        assert!(looks_like_decal(inputs));
    }

    #[test]
    fn looks_like_decal_skips_alpha_blended_depth_writer() {
        // An alpha-blended draw that ALSO writes depth is not a decal:
        // the depth-write prong excludes it, so it keeps the game's own
        // bias. Widening the predicate to such draws would need a
        // different signal (e.g. D3DRS_ALPHATESTENABLE).
        let inputs = DecalHeuristicInputs {
            depth_enable: 1,
            depth_write: 1,
            blend_enable: 1,
            raw_depth_bias: 0,
            raw_slope_scale: 0,
        };
        assert!(!looks_like_decal(inputs));
    }

    #[test]
    fn looks_like_decal_skips_game_supplied_bias() {
        // Alpha-blended decal-shaped draw whose game-side
        // D3DRS_DEPTHBIAS is already non-zero. The predicate declines,
        // so the game's own bias is left alone rather than clobbered.
        let inputs = DecalHeuristicInputs {
            depth_enable: 1,
            depth_write: 0,
            blend_enable: 1,
            raw_depth_bias: 0x3a83_126f, // ~ +1e-3 as f32 bits
            raw_slope_scale: 0,
        };
        assert!(!looks_like_decal(inputs));
    }

    #[test]
    fn looks_like_decal_skips_opaque_draw() {
        // No alpha blend → not a decal pattern. Solid geometry that
        // happens to disable depth-write (e.g. a deferred normals
        // prepass) shouldn't be pulled toward camera.
        let inputs = DecalHeuristicInputs {
            depth_enable: 1,
            depth_write: 0,
            blend_enable: 0,
            raw_depth_bias: 0,
            raw_slope_scale: 0,
        };
        assert!(!looks_like_decal(inputs));
    }

    #[test]
    fn implicit_decal_bias_scales_to_safe_metal_band() {
        // Magnitude band rationale:
        // (a) > ~500 Metal units swamps the depth-buffer's 2^-23
        //     step plus the structural eye-space delta observed
        //     between two SM3 pipelines on Apple Silicon at grazing
        //     angles;
        // (b) < ~5000 keeps flat decals from punching through
        //     adjacent geometry on steep terrain.
        // Tune the constant if a future workload forces it out of
        // this band; the test catches accidental order-of-magnitude
        // changes.
        let metal = d3d_depth_bias_to_metal(IMPLICIT_DECAL_BIAS_RAW);
        assert!(
            metal < 0.0,
            "implicit bias must pull toward camera, got {metal}"
        );
        let mag = -metal;
        assert!(mag > 500.0, "magnitude {mag} too small to swamp ULP noise");
        assert!(
            mag < 5000.0,
            "magnitude {mag} risks punching through terrain"
        );
    }

    #[test]
    fn d3d_to_metal_blend_op_table() {
        assert_eq!(d3d_to_metal_blend_op(1), BlendOperation::Add);
        assert_eq!(d3d_to_metal_blend_op(2), BlendOperation::Subtract);
        assert_eq!(d3d_to_metal_blend_op(3), BlendOperation::ReverseSubtract);
        assert_eq!(d3d_to_metal_blend_op(4), BlendOperation::Min);
        assert_eq!(d3d_to_metal_blend_op(5), BlendOperation::Max);
        // Unknown → Add (with warn).
        assert_eq!(d3d_to_metal_blend_op(0), BlendOperation::Add);
        assert_eq!(d3d_to_metal_blend_op(99), BlendOperation::Add);
    }
}
