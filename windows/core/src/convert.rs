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

/// A triangle fan rewritten as a triangle-list index stream.
///
/// Built by [`triangle_fan_indices`] / [`triangle_fan_indices_from`] for the
/// bound-buffer draw paths: the vertices stay in the application's vertex
/// buffers and only this generated index list is staged per draw.
pub struct FanIndices {
    /// `primitive_count * 3` little-endian indices, each `index_type` wide.
    pub bytes: Vec<u8>,
    /// `UInt16` when every index fits, `UInt32` otherwise.
    pub index_type: IndexType,
    /// Lowest vertex-buffer index the list references.
    pub min_vertex: u32,
    /// Highest vertex-buffer index the list references.
    pub max_vertex: u32,
}

/// Pack `fan` (the fan's vertices, in order) into triangles `0, i+1, i+2`.
fn build_fan_indices(fan: &[u32], primitive_count: u32) -> FanIndices {
    let (min_vertex, max_vertex) = fan
        .iter()
        .fold((u32::MAX, 0), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let wide = max_vertex > u32::from(u16::MAX);
    let index_type = if wide {
        IndexType::UInt32
    } else {
        IndexType::UInt16
    };
    let pc = primitive_count as usize;
    let mut bytes = Vec::with_capacity(pc * 3 * if wide { 4 } else { 2 });
    let mut push = |v: u32| {
        if wide {
            bytes.extend_from_slice(&v.to_le_bytes());
        } else {
            let narrow = u16::try_from(v).expect("narrow index stream only when every index fits");
            bytes.extend_from_slice(&narrow.to_le_bytes());
        }
    };
    for i in 0..pc {
        push(fan[0]);
        push(fan[i + 1]);
        push(fan[i + 2]);
    }
    FanIndices {
        bytes,
        index_type,
        min_vertex,
        max_vertex,
    }
}

/// Index stream for a non-indexed triangle fan (`DrawPrimitive`).
///
/// Metal has no triangle-fan primitive. The fan's `primitive_count + 2`
/// vertices sit back-to-back from `start_vertex`, and triangle `i` is fan
/// vertices `0, i+1, i+2`; the result references them by absolute index.
/// `None` when the vertex range overflows `u32`.
#[must_use]
pub fn triangle_fan_indices(start_vertex: u32, primitive_count: u32) -> Option<FanIndices> {
    let count = primitive_count.checked_add(2)?;
    start_vertex.checked_add(count - 1)?;
    let fan: Vec<u32> = (0..count).map(|k| start_vertex + k).collect();
    Some(build_fan_indices(&fan, primitive_count))
}

/// Index stream for an indexed triangle fan (`DrawIndexedPrimitive`).
///
/// `src` holds the fan's `primitive_count + 2` application indices, each
/// `index_size` (2 or 4) bytes, starting at the draw's `StartIndex`.
/// `base_vertex` is folded into every index so the result is absolute, which
/// is what the inline-index draw form takes. `None` when `src` is short, the
/// index size is unknown, or an index leaves `u32` after the base offset.
#[must_use]
pub fn triangle_fan_indices_from(
    src: &[u8],
    index_size: usize,
    base_vertex: i32,
    primitive_count: u32,
) -> Option<FanIndices> {
    let count = usize::try_from(primitive_count.checked_add(2)?).ok()?;
    if src.len() < count.checked_mul(index_size)? {
        return None;
    }
    let mut fan = Vec::with_capacity(count);
    for k in 0..count {
        let raw = &src[k * index_size..(k + 1) * index_size];
        let index = match index_size {
            2 => u32::from(u16::from_le_bytes([raw[0], raw[1]])),
            4 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            _ => return None,
        };
        fan.push(u32::try_from(i64::from(index) + i64::from(base_vertex)).ok()?);
    }
    Some(build_fan_indices(&fan, primitive_count))
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

/// The Metal vertex attributes a declaration resolves to, plus what each stream it names needs.
///
/// Produced by [`resolve_attrs_for_vs`] / [`resolve_attrs_for_ff`] and
/// consumed by the draw path, which lays out one vertex buffer per used
/// stream and binds that stream's buffer at the Metal slot of the same index.
pub struct ResolvedAttrs {
    /// One entry per consumed element; `buffer_index` is the element's stream.
    pub attrs: Vec<VertexAttrDesc>,
    /// Per stream, `max(offset + size)` over every element on it.
    ///
    /// Unconsumed elements count too: the stream's vertex buffer layout must
    /// cover this extent, since Metal rejects a pipeline whose attribute
    /// reaches past its layout's stride. Zero for a stream the declaration
    /// never names.
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
        let extent = &mut extents[stream as usize];
        *extent = (*extent).max(u32::from(e.offset) + size);
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

    fn u16_indices(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    fn u32_indices(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn nonindexed_fan_becomes_absolute_u16_triangles() {
        // DrawPrimitive(FAN, start 10, 3 prims): fan vertices 10..=14.
        let fan = triangle_fan_indices(10, 3).expect("fits");
        assert_eq!(fan.index_type, IndexType::UInt16);
        assert_eq!(
            u16_indices(&fan.bytes),
            vec![10, 11, 12, 10, 12, 13, 10, 13, 14]
        );
        assert_eq!((fan.min_vertex, fan.max_vertex), (10, 14));
    }

    #[test]
    fn fan_widens_to_u32_past_u16_range() {
        let fan = triangle_fan_indices(0xFFFE, 1).expect("fits");
        assert_eq!(fan.index_type, IndexType::UInt32);
        assert_eq!(u32_indices(&fan.bytes), vec![0xFFFE, 0xFFFF, 0x1_0000]);
        assert!(triangle_fan_indices(u32::MAX - 1, 1).is_none());
    }

    #[test]
    fn indexed_fan_folds_the_base_vertex_in() {
        // 16-bit app indices 5,6,7,8 with base vertex 100: triangles over
        // 105..=108.
        let src: Vec<u8> = [5u16, 6, 7, 8]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let fan = triangle_fan_indices_from(&src, 2, 100, 2).expect("fits");
        assert_eq!(fan.index_type, IndexType::UInt16);
        assert_eq!(u16_indices(&fan.bytes), vec![105, 106, 107, 105, 107, 108]);
        assert_eq!((fan.min_vertex, fan.max_vertex), (105, 108));
        // A negative base is legal as long as no index goes below zero.
        let fan = triangle_fan_indices_from(&src, 2, -5, 2).expect("fits");
        assert_eq!(u16_indices(&fan.bytes), vec![0, 1, 2, 0, 2, 3]);
        assert!(triangle_fan_indices_from(&src, 2, -6, 2).is_none());
    }

    #[test]
    fn indexed_fan_reads_32_bit_indices_and_rejects_short_streams() {
        let src: Vec<u8> = [1u32, 2, 0x2_0000]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let fan = triangle_fan_indices_from(&src, 4, 0, 1).expect("fits");
        assert_eq!(fan.index_type, IndexType::UInt32);
        assert_eq!(u32_indices(&fan.bytes), vec![1, 2, 0x2_0000]);
        assert!(triangle_fan_indices_from(&src, 4, 0, 2).is_none());
        assert!(triangle_fan_indices_from(&src, 3, 0, 1).is_none());
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
    fn linear_to_srgb_encodes_colour_lanes_only() {
        // 0x7f linear stores as 0xbb once sRGB-encoded; alpha passes through.
        let rgba = linear_to_srgb_rgba(d3dcolor_to_rgba_f32(0x407f_7f7f));
        // The stored byte, as the exactly representable float it rounds to.
        let to_byte = |v: f32| (v * 255.0).round().to_bits();
        let byte = |b: u8| f32::from(b).to_bits();
        assert_eq!(to_byte(rgba[0]), byte(0xbb));
        assert_eq!(to_byte(rgba[1]), byte(0xbb));
        assert_eq!(to_byte(rgba[2]), byte(0xbb));
        assert_eq!(rgba[3].to_bits(), (f32::from(0x40u8) / 255.0).to_bits());
        // The end points map onto themselves (to within a ulp at white), and
        // an over-range lane clamps before encoding.
        let ends = linear_to_srgb_rgba([0.0, 1.0, 2.0, 1.0]);
        assert_eq!(ends[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(to_byte(ends[1]), byte(0xff));
        assert_eq!(to_byte(ends[2]), byte(0xff));
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
        let resolved = resolve_attrs_for_vs(&elems, &semantics);
        assert_eq!(resolved.attrs.len(), 2);
        assert_eq!(resolved.attrs[0].attr_index, 2);
        assert_eq!(resolved.attrs[1].attr_index, 7);
        assert_eq!(resolved.extents[0], 20);
        assert_eq!(resolved.used_streams, 0b1);
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
        let resolved = resolve_attrs_for_vs(&elems, &semantics);
        assert_eq!(resolved.attrs.len(), 1);
        assert_eq!(resolved.attrs[0].attr_index, 0);
        // The extent still covers the normal element so the vertex buffer
        // layout is correct even with an unused attribute.
        assert_eq!(resolved.extents[0], 24);
    }

    #[test]
    fn resolve_attrs_for_ff_matches_ff_convention() {
        // POSITION → attr(0), TEXCOORD0 → attr(4). Must agree with
        // `crate::dxso::ff_attr_index_for_semantic`.
        let elems = [pos3(), tex0(12)];
        let resolved = resolve_attrs_for_ff(&elems);
        assert_eq!(resolved.attrs.len(), 2);
        assert_eq!(resolved.attrs[0].attr_index, 0);
        assert_eq!(resolved.attrs[1].attr_index, 4);
        assert_eq!(resolved.extents[0], 20);
    }

    #[test]
    fn resolve_attrs_keeps_each_stream_separate() {
        // POSITION on stream 0, COLOR0 on stream 1 at offset 0, an unconsumed
        // NORMAL on stream 1 past it: stream 1's extent covers the normal,
        // the colour attribute points at buffer 1, and both streams are used.
        let elems = [
            pos3(),
            D3DVERTEXELEMENT9 {
                stream: 1,
                offset: 0,
                type_: D3DDECLTYPE_D3DCOLOR,
                method: 0,
                usage: D3DDECLUSAGE_COLOR,
                usage_index: 0,
            },
            D3DVERTEXELEMENT9 {
                stream: 1,
                offset: 4,
                type_: D3DDECLTYPE_FLOAT3,
                method: 0,
                usage: D3DDECLUSAGE_NORMAL,
                usage_index: 0,
            },
        ];
        let semantics = vec![
            InputSemantic {
                usage: DeclUsage::Position,
                usage_index: 0,
                register_index: 0,
            },
            InputSemantic {
                usage: DeclUsage::Color,
                usage_index: 0,
                register_index: 1,
            },
        ];
        let resolved = resolve_attrs_for_vs(&elems, &semantics);
        assert_eq!(resolved.attrs.len(), 2);
        assert_eq!(resolved.attrs[0].buffer_index, 0);
        assert_eq!(resolved.attrs[1].buffer_index, 1);
        assert_eq!(resolved.attrs[1].attr_index, 1);
        assert_eq!(resolved.extents[0], 12);
        assert_eq!(resolved.extents[1], 16);
        assert_eq!(resolved.used_streams, 0b11);

        // A stream that only carries unconsumed elements is not used, but its
        // extent is still reported.
        let resolved = resolve_attrs_for_vs(&elems, &semantics[..1]);
        assert_eq!(resolved.used_streams, 0b1);
        assert_eq!(resolved.extents[1], 16);

        // The FF path maps streams the same way.
        let resolved = resolve_attrs_for_ff(&elems);
        assert_eq!(resolved.attrs.len(), 3);
        assert_eq!(resolved.attrs[1].buffer_index, 1);
        assert_eq!(resolved.used_streams, 0b11);
    }

    #[test]
    fn resolve_attrs_drops_streams_past_the_slot_table() {
        let elems = [
            pos3(),
            D3DVERTEXELEMENT9 {
                stream: 16,
                offset: 0,
                type_: D3DDECLTYPE_D3DCOLOR,
                method: 0,
                usage: D3DDECLUSAGE_COLOR,
                usage_index: 0,
            },
        ];
        let resolved = resolve_attrs_for_ff(&elems);
        assert_eq!(resolved.attrs.len(), 1);
        assert_eq!(resolved.used_streams, 0b1);
        let layout = ff_vs_layout_from_elements(&elems, true);
        assert!(
            !layout.has_color0(),
            "dropped element leaves no flag behind"
        );
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
        let h_a = pack_vertex_decl(&elems).expect("pack a").hash;
        let h_b = pack_vertex_decl(&elems).expect("pack b").hash;
        assert_eq!(h_a, h_b);
        let swapped = [pos3(), tex0(16), end()];
        let h_c = pack_vertex_decl(&swapped).expect("pack c").hash;
        assert_ne!(h_a, h_c);
    }

    #[test]
    fn pack_vertex_decl_multi_stream_distinct_hash_and_mask() {
        // Two layouts that differ *only* by stream must hash differently so
        // the pipeline cache keeps them apart, and the stream mask names the
        // streams the draw path has to snapshot.
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
        assert_ne!(a.hash, b.hash, "stream must participate in the decl hash");
        assert_eq!(a.stream_mask, 0b01);
        assert_eq!(b.stream_mask, 0b10);
        let both = pack_vertex_decl(&[on_stream(0), tex0(0), on_stream(3), end()]).expect("pack");
        assert_eq!(both.stream_mask, 0b1001);
        // A stream past the slot table is accepted (D3D9 validates structure
        // only) but contributes no bit.
        let wide = pack_vertex_decl(&[on_stream(0), on_stream(16), end()]).expect("pack");
        assert_eq!(wide.stream_mask, 0b1);
    }

    #[test]
    fn pack_vertex_decl_requires_terminator() {
        assert!(pack_vertex_decl(&[pos3()]).is_none());
    }

    #[test]
    fn pack_vertex_decl_preserves_terminator_in_output() {
        let elems = [pos3(), tex0(12), end()];
        let packed = pack_vertex_decl(&elems).expect("pack").elements_with_end;
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
