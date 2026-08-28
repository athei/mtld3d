use mtld3d_types::{
    AddressCaps, BlendCaps, Caps2, Caps3, CmpCaps, CursorCaps, D3D_MAX_SIMULTANEOUS_RENDERTARGETS,
    D3DCAPS9, D3DDEVTYPE_HAL, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE,
    D3DPS20_MAX_DYNAMICFLOWCONTROLDEPTH, D3DPS20_MAX_NUMINSTRUCTIONSLOTS, D3DPS20_MAX_NUMTEMPS,
    D3DPS20_MAX_STATICFLOWCONTROLDEPTH, D3DPS30_INSTRUCTIONSLOTS_MAX, D3DVBF_3WEIGHTS,
    D3DVS20_MAX_DYNAMICFLOWCONTROLDEPTH, D3DVS20_MAX_NUMTEMPS, D3DVS20_MAX_STATICFLOWCONTROLDEPTH,
    D3DVS30_INSTRUCTIONSLOTS_MAX, DeclTypeCaps, DevCaps, DevCaps2, FilterCaps, FvfCaps, LineCaps,
    MAX_POINT_SIZE, MAX_STREAMS, MAX_VERTEX_SHADER_CONST, MAX_VOLUME_EXTENT, PrimitiveMiscCaps,
    Ps20Caps, RasterCaps, ShadeCaps, StencilCaps, TexOpCaps, TextureCaps, Vs20Caps, VtxpCaps,
    d3dps_version, d3dvs_version,
};

use crate::ff_state::MAX_ACTIVE_LIGHTS;

// Caps are a *truthful floor* under current capability: they advertise only
// what the renderer actually implements. Every default below therefore names
// its bits one by one rather than reaching for `Flags::all()`: the diagnostic
// in `apply_advertise_all` is the only place a whole spec field goes out at
// once. Re-add a bit in the same commit that lands the feature.

/// Largest texture edge, and the widest aspect ratio, we accept.
///
/// Also the D3D9 maximum render-target dimension, so scissor / viewport
/// coordinates derived from it fit in `u16`.
const MAX_TEXTURE_DIM: u32 = 16384;

/// Address-space wrap count for a single texture coordinate.
const MAX_TEXTURE_REPEAT: u32 = 8192;

/// Anisotropic-filter tap count reachable through `D3DSAMP_MAXANISOTROPY`.
///
/// Reaches `setMaxAnisotropy:` on `MTLSamplerDescriptor` unchanged, whose own
/// ceiling is 16.
const MAX_ANISOTROPY: u32 = 16;

/// Largest homogeneous W the clipper accepts.
const MAX_VERTEX_W: f32 = 1e10;

/// Fixed-function texture blend stages, which is also the simultaneous-texture count.
///
/// D3D9 defines 8 FF blend stages and the FF pipeline can sample one texture
/// per stage, so `MaxTextureBlendStages` and `MaxSimultaneousTextures` are the
/// same number. It is likewise the texture-coordinate-set count reported
/// through `FVFCaps`. Distinct from the 16 sampler slots an SM3 pixel shader
/// addresses, which no `D3DCAPS9` field reports.
const FF_TEXTURE_STAGES: u32 = 8;

/// Matrices blended into one vertex by the FF vertex-blending path.
///
/// Hardware vertex blending is wired end to end: `FfState` carries
/// `world_palette[256]` routed from `D3DTS_WORLDMATRIX(i)`,
/// `build_vs_constants` packs the active bones × 4 rows after the existing
/// layout, and `emit_vs` blends position + normal via the explicit-weight loop
/// with an implicit last weight. `D3DVBF_3WEIGHTS` is the spec maximum per
/// vertex, so the cap is that mode's explicit weight count plus the implicit
/// one. The palette size is the separate `D3DTS_WORLDMATRIX(0..255)` range,
/// which `D3DCAPS9` has no field for (games discover it by trial).
const MAX_VERTEX_BLEND_MATRICES: u32 = D3DVBF_3WEIGHTS + 1;

/// Primitives a single `DrawPrimitive` call may emit.
///
/// One triangle per three of the addressable vertices below.
const MAX_PRIMITIVE_COUNT: u32 = 0x0055_5555;

/// Largest index value an index buffer may contain (24-bit index range).
const MAX_VERTEX_INDEX: u32 = 0x00FF_FFFF;

/// Largest vertex stride a stream may declare, in bytes.
const MAX_STREAM_STRIDE: u32 = 508;

/// `PixelShader1xMaxValue`: the clamp applied to PS 1.x intermediate results.
///
/// The largest finite half-float, which is what D3D9-era hardware clamped to.
/// SM2 and SM3 shaders ignore the field.
const PIXEL_SHADER_1X_MAX_VALUE: f32 = 65504.0;

/// Render targets bound at once on the default path: the D3D9 maximum of four.
///
/// Each one is a colour attachment of the same Metal render pass with its own
/// format and write mask (`D3DRS_COLORWRITEENABLE1..3`), and the pixel shader
/// exports one `[[color(i)]]` per target it writes. D3D9 advertises
/// render-target format support through `CheckDeviceFormat` rather than a
/// `D3DCAPS9` flag.
const DEFAULT_SIMULTANEOUS_RTS: u32 = D3D_MAX_SIMULTANEOUS_RENDERTARGETS;

/// Driver-level caps: resource management, dynamic textures, gamma, mip generation.
const CAPS2_DEFAULT: Caps2 = Caps2::CANMANAGERESOURCE
    .union(Caps2::DYNAMICTEXTURES)
    .union(Caps2::FULLSCREENGAMMA)
    .union(Caps2::CANAUTOGENMIPMAP);

/// Present-path caps: the swap chain preserves alpha across a flip or discard.
const CAPS3_DEFAULT: Caps3 = Caps3::ALPHA_FULLSCREEN_FLIP_OR_DISCARD;

/// Cursor caps: the full-colour Win32 `HCURSOR` path is live.
const CURSOR_CAPS_DEFAULT: CursorCaps = CursorCaps::COLOR;

/// Device caps: memory classes we accept geometry from, plus HW T&L and rasterization.
const DEV_CAPS_DEFAULT: DevCaps = DevCaps::EXECUTESYSTEMMEMORY
    .union(DevCaps::EXECUTEVIDEOMEMORY)
    .union(DevCaps::TLVERTEXSYSTEMMEMORY)
    .union(DevCaps::TLVERTEXVIDEOMEMORY)
    .union(DevCaps::TEXTURESYSTEMMEMORY)
    .union(DevCaps::DRAWPRIMTLVERTEX)
    .union(DevCaps::HWTRANSFORMANDLIGHT)
    .union(DevCaps::HWRASTERIZATION);

/// Rasterizer-adjacent caps that are not covered by a more specific field.
///
/// `CLIPTLVERTS` is factual: Metal clips every vertex — including the
/// pre-transformed XYZRHW (TL) verts handed through as clip-space — to the NDC
/// volume, so post-transform clipping always happens. The three MRT bits are
/// what the Metal render pass gives every colour attachment: its own write
/// mask (`INDEPENDENTWRITEMASKS`), its own pixel format
/// (`MRTINDEPENDENTBITDEPTHS`) and blending (`MRTPOSTPIXELSHADERBLENDING`).
const PRIMITIVE_MISC_DEFAULT: PrimitiveMiscCaps = PrimitiveMiscCaps::MASKZ
    .union(PrimitiveMiscCaps::CULLNONE)
    .union(PrimitiveMiscCaps::CULLCW)
    .union(PrimitiveMiscCaps::CULLCCW)
    .union(PrimitiveMiscCaps::COLORWRITEENABLE)
    .union(PrimitiveMiscCaps::CLIPTLVERTS)
    .union(PrimitiveMiscCaps::BLENDOP)
    .union(PrimitiveMiscCaps::INDEPENDENTWRITEMASKS)
    .union(PrimitiveMiscCaps::SEPARATEALPHABLEND)
    .union(PrimitiveMiscCaps::MRTINDEPENDENTBITDEPTHS)
    .union(PrimitiveMiscCaps::MRTPOSTPIXELSHADERBLENDING)
    .union(PrimitiveMiscCaps::POSTBLENDSRGBCONVERT);

/// Rasterizer caps.
///
/// `DEPTHBIAS` + `SLOPESCALEDEPTHBIAS` reflect the explicit-RS bias path that
/// reaches Metal's `setDepthBias:slopeScale:clamp:` per draw. `ANISOTROPY`
/// advertises the wired path `D3DSAMP_MAXANISOTROPY` →
/// `SamplerSnapshot.max_anisotropy` → `CreateSamplerStateParams.max_anisotropy`
/// → `setMaxAnisotropy:` on `MTLSamplerDescriptor`; without the cap bit,
/// well-behaved games clamp to `MAXANISOTROPY=1` and never ask for it.
/// `MIPMAPLODBIAS` advertises `D3DSAMP_MIPMAPLODBIAS`, which Metal expresses
/// at the sample site rather than on the sampler: the bias reaches the pixel
/// shader as a per-slot uniform and shifts the mip every implicit-LOD sample
/// selects.
const RASTER_DEFAULT: RasterCaps = RasterCaps::ZTEST
    .union(RasterCaps::FOGVERTEX)
    .union(RasterCaps::FOGRANGE)
    .union(RasterCaps::MIPMAPLODBIAS)
    .union(RasterCaps::ANISOTROPY)
    .union(RasterCaps::ZFOG)
    .union(RasterCaps::SCISSORTEST)
    .union(RasterCaps::SLOPESCALEDEPTHBIAS)
    .union(RasterCaps::DEPTHBIAS);

/// Blend factors, for both the colour and the alpha blend equation.
///
/// Every `D3DBLEND_*` value `convert::d3d_to_metal_blend` maps, which is the
/// contiguous `ZERO..SRCALPHASAT` range plus `BLENDFACTOR`. The `BOTH*` factors
/// are DX7-era shorthand D3D9 no longer honours, and the dual-source `*2`
/// factors need a second pixel-shader output.
const BLEND_DEFAULT: BlendCaps = BlendCaps::ZERO
    .union(BlendCaps::ONE)
    .union(BlendCaps::SRCCOLOR)
    .union(BlendCaps::INVSRCCOLOR)
    .union(BlendCaps::SRCALPHA)
    .union(BlendCaps::INVSRCALPHA)
    .union(BlendCaps::DESTALPHA)
    .union(BlendCaps::INVDESTALPHA)
    .union(BlendCaps::DESTCOLOR)
    .union(BlendCaps::INVDESTCOLOR)
    .union(BlendCaps::SRCALPHASAT)
    .union(BlendCaps::BLENDFACTOR);

/// Shading caps: the FF vertex shader emits `out.color0` + `out.color1`.
const SHADE_DEFAULT: ShadeCaps = ShadeCaps::COLORGOURAUDRGB
    .union(ShadeCaps::SPECULARGOURAUDRGB)
    .union(ShadeCaps::ALPHAGOURAUDBLEND);

/// Texture caps.
///
/// Every [`TextureCaps::RESTRICTIONS`] bit stays clear: Metal supports
/// non-power-of-two textures unconditionally (mipmaps + wrap addressing), which
/// D3D9 signals by leaving them off, and texture creation accepts arbitrary
/// sizes. `TTFF_PROJECTED` is honored in FF pixel-shader sample emission;
/// `PERSPECTIVE` is a factual statement — Metal interpolates
/// perspective-correctly by default and `dxso::emit` never emits `[[flat]]` /
/// `[[no_perspective]]` qualifiers.
const TEXTURE_DEFAULT: TextureCaps = TextureCaps::PERSPECTIVE
    .union(TextureCaps::ALPHA)
    .union(TextureCaps::PROJECTED)
    .union(TextureCaps::MIPMAP)
    .union(TextureCaps::CUBEMAP)
    .union(TextureCaps::MIPCUBEMAP)
    .union(TextureCaps::VOLUMEMAP)
    .union(TextureCaps::MIPVOLUMEMAP);

/// Filters `StretchRect` accepts: point and linear, which is also all it validates.
///
/// The copy itself is a blit, so the two are indistinguishable in the result;
/// the cap describes what the call takes without `INVALIDCALL`.
const STRETCH_RECT_FILTER: FilterCaps = FilterCaps::MINFPOINT
    .union(FilterCaps::MINFLINEAR)
    .union(FilterCaps::MAGFPOINT)
    .union(FilterCaps::MAGFLINEAR);

/// Half-width of the guard band, in pixels, on every side.
///
/// Metal clips in homogeneous space, so there is no rasterizer guard band to
/// run out of; the D3D9 convention for such hardware is the +-32768 band that
/// desktop drivers report.
const GUARD_BAND: f32 = 32768.0;

/// Filter caps for the 2D texture path: point, linear, and anisotropic minification.
///
/// Device-wide, not per format: whether one format samples with linear
/// filtering is `CheckDeviceFormat(D3DUSAGE_QUERY_FILTER)`, which answers the
/// single-precision float family from the device (`format::supports_usage_query`).
const FILTER_DEFAULT: FilterCaps = FilterCaps::MINFPOINT
    .union(FilterCaps::MINFLINEAR)
    .union(FilterCaps::MINFANISOTROPIC)
    .union(FilterCaps::MIPFPOINT)
    .union(FilterCaps::MIPFLINEAR)
    .union(FilterCaps::MAGFPOINT)
    .union(FilterCaps::MAGFLINEAR);

/// Addressing modes.
///
/// All six, including `MIRRORONCE` (`D3DTADDRESS_MIRRORONCE`), which
/// `convert::d3d_to_metal_address_mode` maps to
/// `AddressMode::MirrorClampToEdge` through the same code path as the other
/// four modes. `MIRRORONCE` stays advertised on a device that does not
/// implement that Metal address mode (the paravirtualized one a CI runner
/// exposes): the sampler path substitutes `MirrorRepeat`, which agrees with
/// MIRRORONCE wherever content samples. `BORDER` is the one mode that comes
/// back out below, because clamping to edge instead is visible in the image a
/// title that asked for a border colour gets.
const ADDRESS_DEFAULT: AddressCaps = AddressCaps::WRAP
    .union(AddressCaps::MIRROR)
    .union(AddressCaps::CLAMP)
    .union(AddressCaps::BORDER)
    .union(AddressCaps::INDEPENDENTUV)
    .union(AddressCaps::MIRRORONCE);

/// Stencil operations the depth-stencil state builder implements.
///
/// Every `D3DSTENCILOP_*` has an exact `MTLStencilOperation` counterpart, and
/// `TWOSIDED` is covered by writing both `MTLDepthStencilDescriptor` faces
/// from the `D3DRS_CCW_STENCIL*` states. Metal's default front-facing winding
/// is clockwise, which is also D3D9's front, so the D3D9 front-face states
/// land on Metal's front face.
const STENCIL_DEFAULT: StencilCaps = StencilCaps::all();

/// Fixed-function texture blend operations the FF pixel-shader emitter implements.
///
/// `BLENDTEXTUREALPHAPM` is intentionally absent: `dxso::ff` does not emit the
/// premultiplied form.
const TEXOP_DEFAULT: TexOpCaps = TexOpCaps::DISABLE
    .union(TexOpCaps::SELECTARG1)
    .union(TexOpCaps::SELECTARG2)
    .union(TexOpCaps::MODULATE)
    .union(TexOpCaps::MODULATE2X)
    .union(TexOpCaps::MODULATE4X)
    .union(TexOpCaps::ADD)
    .union(TexOpCaps::ADDSIGNED)
    .union(TexOpCaps::ADDSIGNED2X)
    .union(TexOpCaps::SUBTRACT)
    .union(TexOpCaps::ADDSMOOTH)
    .union(TexOpCaps::BLENDDIFFUSEALPHA)
    .union(TexOpCaps::BLENDTEXTUREALPHA)
    .union(TexOpCaps::BLENDFACTORALPHA)
    .union(TexOpCaps::BLENDCURRENTALPHA);

/// Line-drawing caps: textured, depth-tested, blended, alpha-tested lines.
const LINE_DEFAULT: LineCaps = LineCaps::TEXTURE
    .union(LineCaps::ZTEST)
    .union(LineCaps::BLEND)
    .union(LineCaps::ALPHACMP);

/// FVF caps: the texture-coordinate-set count plus `PSIZE`.
///
/// The fixed-function vertex shader reads a `D3DFVF_PSIZE` element as the
/// per-vertex point size (`dxso::ff::emit_point_size`).
const FVF_DEFAULT: FvfCaps = FvfCaps::texcoord_sets(FF_TEXTURE_STAGES).union(FvfCaps::PSIZE);

/// Vertex-processing caps.
///
/// The FF vertex shader honors TCI texgen modes, all three light types
/// (directional / point / spot cone), and both specular view-vector models via
/// `D3DRS_LOCALVIEWER`.
const VTXP_DEFAULT: VtxpCaps = VtxpCaps::TEXGEN
    .union(VtxpCaps::MATERIALSOURCE7)
    .union(VtxpCaps::DIRECTIONALLIGHTS)
    .union(VtxpCaps::POSITIONALLIGHTS)
    .union(VtxpCaps::LOCALVIEWER);

/// Optional vertex-declaration element types `decl_type_to_metal_format` accepts.
///
/// `UDEC3` / `DEC3N` are rejected (no Metal equivalent), so their bits stay
/// off. The `FLOAT1`..`FLOAT4` and `D3DCOLOR` types are baseline and carry no
/// bit at all.
const DECL_TYPES_DEFAULT: DeclTypeCaps = DeclTypeCaps::UBYTE4
    .union(DeclTypeCaps::UBYTE4N)
    .union(DeclTypeCaps::SHORT2N)
    .union(DeclTypeCaps::SHORT4N)
    .union(DeclTypeCaps::USHORT2N)
    .union(DeclTypeCaps::USHORT4N)
    .union(DeclTypeCaps::FLOAT16_2)
    .union(DeclTypeCaps::FLOAT16_4);

/// The `DEVCAPS2` bits that are truthful on the default path.
///
/// `StretchRect` from a texture-level surface into a render target is
/// supported by the blit path; a `SetStreamSource` byte offset is honoured on
/// every stream, and two declaration elements may share one offset (Metal
/// places attributes independently). `apply_advertise_all` widens the field
/// to every spec bit.
const DEV_CAPS2_DEFAULT: DevCaps2 = DevCaps2::CAN_STRETCHRECT_FROM_TEXTURES
    .union(DevCaps2::STREAMOFFSET)
    .union(DevCaps2::VERTEXELEMENTSCANSHARESTREAMOFFSET);

/// Texture caps the diagnostic ORs in.
///
/// Every spec bit except the restriction group: over-advertising a *limitation*
/// would make a game do less, not more, which is the opposite of what the
/// diagnostic is for.
const ADVERTISE_ALL_TEXTURE: TextureCaps = TextureCaps::all().difference(TextureCaps::RESTRICTIONS);

/// Filter caps the diagnostic ORs in.
///
/// `CONVOLUTIONMONO` stays out: it is a separate filter kind rather than a
/// MIN/MIP/MAG mode, with no sampler-side path or warn to surface an attempt.
const ADVERTISE_ALL_FILTER: FilterCaps = FilterCaps::all().difference(FilterCaps::CONVOLUTIONMONO);

/// Single entry point for both `IDirect3D9::GetDeviceCaps` and `IDirect3DDevice9::GetDeviceCaps`.
///
/// Runs `fill_default`, then ORs in every spec bit per field when `caps_all` is
/// `true` (the resolved `debug.capsAll` from `mtld3d.conf`). The override is
/// process-wide — no per-call-site opt-in — so games can't accidentally see a
/// half-advertised cap set.
pub fn fill(caps: &mut D3DCAPS9, caps_all: bool, sampler_border: bool) {
    fill_default(caps);
    if !sampler_border {
        // The device cannot create border-colour samplers (virtualized CI
        // devices); a title that checks the cap then avoids the address mode
        // instead of hitting the clamp-to-edge substitution.
        let strip = !AddressCaps::BORDER.bits();
        caps.texture_address_caps &= strip;
        caps.volume_texture_address_caps &= strip;
    }
    if caps_all {
        apply_advertise_all(caps);
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "debug.capsAll=true: advertising spec-max caps for bring-up diagnostic — visual rendering may degrade"
        );
    }
}

const fn fill_default(caps: &mut D3DCAPS9) {
    // SAFETY: `caps` is a valid `&mut D3DCAPS9`; zeroing all bytes is sound
    // because every field is a primitive integer (no Drop, no padding invariants).
    unsafe { core::ptr::write_bytes(std::ptr::from_mut::<D3DCAPS9>(caps), 0, 1) };

    caps.device_type = D3DDEVTYPE_HAL;
    caps.caps2 = CAPS2_DEFAULT.bits();
    caps.caps3 = CAPS3_DEFAULT.bits();
    // The two intervals the present path implements: display-rate vsync
    // (`ONE`, also what `DEFAULT` resolves to) and `IMMEDIATE`. The divided
    // rates are accepted by `CreateDevice` but fall through to `ONE` with a
    // warn, so they are not advertised. 3DMark05 refuses to start without
    // `IMMEDIATE` here.
    caps.presentation_intervals = D3DPRESENT_INTERVAL_ONE | D3DPRESENT_INTERVAL_IMMEDIATE;
    caps.cursor_caps = CURSOR_CAPS_DEFAULT.bits();
    caps.dev_caps = DEV_CAPS_DEFAULT.bits();
    caps.primitive_misc_caps = PRIMITIVE_MISC_DEFAULT.bits();
    caps.raster_caps = RASTER_DEFAULT.bits();
    // Every `D3DCMP_*` function reaches Metal through
    // `convert::d3d_to_metal_compare`, for both the depth and the alpha test.
    caps.z_cmp_caps = CmpCaps::all().bits();
    caps.alpha_cmp_caps = CmpCaps::all().bits();
    caps.src_blend_caps = BLEND_DEFAULT.bits();
    caps.dest_blend_caps = BLEND_DEFAULT.bits();
    caps.shade_caps = SHADE_DEFAULT.bits();
    caps.texture_caps = TEXTURE_DEFAULT.bits();
    caps.texture_filter_caps = FILTER_DEFAULT.bits();
    // Cube and volume textures sample through the same Metal sampler state as
    // 2D textures (`MTLTextureTypeCube` / `Type3D` on the unix texture path),
    // so their filter and address caps are the 2D ones. The `VOLUMEMAP`
    // TextureCaps bit is deliberately left off: it un-gates Wine's
    // unbound-sampler visual test, which needs an unbound sampler to read back
    // opaque black — a separate defect to fix before the bit is honest.
    caps.cube_texture_filter_caps = FILTER_DEFAULT.bits();
    caps.volume_texture_filter_caps = FILTER_DEFAULT.bits();
    caps.texture_address_caps = ADDRESS_DEFAULT.bits();
    caps.volume_texture_address_caps = ADDRESS_DEFAULT.bits();
    caps.stretch_rect_filter_caps = STRETCH_RECT_FILTER.bits();
    // Vertex texture fetch is not implemented: no sampler binds on the vertex
    // stage and `SetTexture` rejects the `D3DVERTEXTEXTURESAMPLER` range, so
    // Vertex texture fetch: point and linear min/mag filtering, no mip
    // filter bit — `texldl` supplies its LOD explicitly, and Metal samples
    // any level from a vertex function. Titles gate whole effect paths
    // (per-sprite occlusion, displacement) on this being non-zero next to
    // the matching `CheckDeviceFormat(QUERY_VERTEXTEXTURE)` answer.
    caps.vertex_texture_filter_caps = FILTER_DEFAULT.bits();
    caps.stencil_caps = STENCIL_DEFAULT.bits();
    caps.texture_op_caps = TEXOP_DEFAULT.bits();
    caps.max_texture_blend_stages = FF_TEXTURE_STAGES;
    caps.max_simultaneous_textures = FF_TEXTURE_STAGES;
    caps.max_texture_width = MAX_TEXTURE_DIM;
    caps.max_texture_height = MAX_TEXTURE_DIM;
    caps.max_volume_extent = MAX_VOLUME_EXTENT;
    caps.max_texture_repeat = MAX_TEXTURE_REPEAT;
    caps.max_texture_aspect_ratio = MAX_TEXTURE_DIM;
    caps.max_anisotropy = MAX_ANISOTROPY;
    caps.max_vertex_w = MAX_VERTEX_W;
    caps.guard_band_left = -GUARD_BAND;
    caps.guard_band_top = -GUARD_BAND;
    caps.guard_band_right = GUARD_BAND;
    caps.guard_band_bottom = GUARD_BAND;
    caps.line_caps = LINE_DEFAULT.bits();
    caps.fvf_caps = FVF_DEFAULT.bits();
    caps.vertex_processing_caps = VTXP_DEFAULT.bits();
    caps.max_active_lights = MAX_ACTIVE_LIGHTS;
    caps.max_vertex_blend_matrices = MAX_VERTEX_BLEND_MATRICES;
    caps.max_point_size = MAX_POINT_SIZE;
    caps.max_primitive_count = MAX_PRIMITIVE_COUNT;
    caps.vertex_shader_version = d3dvs_version(3, 0);
    caps.max_vertex_shader_const = MAX_VERTEX_SHADER_CONST;
    caps.pixel_shader_version = d3dps_version(3, 0);
    caps.pixel_shader_1x_max_value = PIXEL_SHADER_1X_MAX_VALUE;
    // The SM2.x sub-structs at their spec maxima, which is what a 3.0 device
    // reports. Each bit names something the DXSO emitter translates: `setp`
    // and predicated writes (PREDICATION), `dsx`/`dsy` (GRADIENTINSTRUCTIONS),
    // `if`/`ifc`/`breakc`/`loop`/`rep` (the flow-control depths), and MSL has
    // neither a dependent-read nor a texture-instruction limit nor a swizzle
    // restriction. Engines of the ps_2_x era pick their shader profile from
    // these fields; all-zero reads as "no ps_2_x at all" to them.
    caps.vs20_caps.caps = Vs20Caps::PREDICATION.bits();
    caps.vs20_caps.dynamic_flow_control_depth = D3DVS20_MAX_DYNAMICFLOWCONTROLDEPTH;
    caps.vs20_caps.num_temps = D3DVS20_MAX_NUMTEMPS;
    caps.vs20_caps.static_flow_control_depth = D3DVS20_MAX_STATICFLOWCONTROLDEPTH;
    caps.ps20_caps.caps = Ps20Caps::all().bits();
    caps.ps20_caps.dynamic_flow_control_depth = D3DPS20_MAX_DYNAMICFLOWCONTROLDEPTH;
    caps.ps20_caps.num_temps = D3DPS20_MAX_NUMTEMPS;
    caps.ps20_caps.static_flow_control_depth = D3DPS20_MAX_STATICFLOWCONTROLDEPTH;
    caps.ps20_caps.num_instruction_slots = D3DPS20_MAX_NUMINSTRUCTIONSLOTS;
    // SM3 capability claims sized to Metal's reality, not the D3D9 spec floor.
    // The `*_INSTRUCTIONSLOTS_MAX` values are the spec's stated upper bound and
    // what top-tier 2007 SM3 cards advertised; Metal has no per-shader
    // instruction limit a game would practically hit, so claiming the spec max
    // is honest. Some games key effect-detail variants off these values —
    // advertising the floor (512) could pin them to a low-quality path.
    // `*_executed` is the theoretical "instructions the GPU can execute per
    // dispatch": `u32::MAX` says "no enforced cap", matching what Metal-backed
    // drivers actually deliver.
    caps.max_vertex_shader_30_instruction_slots = D3DVS30_INSTRUCTIONSLOTS_MAX;
    caps.max_pixel_shader_30_instruction_slots = D3DPS30_INSTRUCTIONSLOTS_MAX;
    caps.max_v_shader_instructions_executed = u32::MAX;
    caps.max_p_shader_instructions_executed = u32::MAX;
    caps.max_vertex_index = MAX_VERTEX_INDEX;
    caps.max_streams = MAX_STREAMS;
    caps.max_stream_stride = MAX_STREAM_STRIDE;
    caps.decl_types = DECL_TYPES_DEFAULT.bits();
    caps.num_simultaneous_rts = DEFAULT_SIMULTANEOUS_RTS;
    caps.dev_caps2 = DEV_CAPS2_DEFAULT.bits();
    // One adapter, which is its own group of one.
    caps.number_of_adapters_in_group = 1;
    // User clip planes: the vertex shaders emit one `[[clip_distance]]` lane
    // per enabled plane from the `VsDraw` uniform (`crate::vs_draw`), whose
    // `MAX_CLIP_PLANES` this must match (pinned by a unit test; a const fn
    // cannot convert the usize).
    caps.max_user_clip_planes = 6;
}

/// Bring-up diagnostic: over-advertise caps only where the fallout would show up in the log.
///
/// ORs in every spec bit for each bitmask field whose consumer warn coverage is
/// solid, and raises the handful of numeric fields whose
/// attempted-but-unimplemented paths have detection hooks upstream
/// (`SetRenderTarget`-index / vertex-decl `BLENDWEIGHT` /
/// `D3DPT_POINTLIST` draw). Skips `pixel_shader_version` /
/// `vertex_shader_version` / `vs20_caps` / `ps20_caps` — the default path
/// already reports the spec maxima there, and the DXSO parser has zero warn
/// coverage, so a shader-version bump would risk silent miscompile with no
/// log signal.
fn apply_advertise_all(caps: &mut D3DCAPS9) {
    caps.raster_caps |= RasterCaps::all().bits();
    caps.texture_caps |= ADVERTISE_ALL_TEXTURE.bits();
    caps.texture_filter_caps |= ADVERTISE_ALL_FILTER.bits();
    caps.cube_texture_filter_caps |= ADVERTISE_ALL_FILTER.bits();
    caps.volume_texture_filter_caps |= ADVERTISE_ALL_FILTER.bits();
    caps.vertex_texture_filter_caps |= ADVERTISE_ALL_FILTER.bits();
    caps.stretch_rect_filter_caps |= ADVERTISE_ALL_FILTER.bits();
    caps.texture_address_caps |= AddressCaps::all().bits();
    caps.volume_texture_address_caps |= AddressCaps::all().bits();
    caps.src_blend_caps |= BlendCaps::all().bits();
    caps.dest_blend_caps |= BlendCaps::all().bits();
    caps.primitive_misc_caps |= PrimitiveMiscCaps::all().bits();
    caps.shade_caps |= ShadeCaps::all().bits();
    caps.vertex_processing_caps |= VtxpCaps::all().bits();
    caps.dev_caps |= DevCaps::all().bits();
    caps.dev_caps2 |= DevCaps2::all().bits();
    caps.line_caps |= LineCaps::all().bits();
    caps.texture_op_caps |= TexOpCaps::all().bits();
    // Field-shape (non-bitmask) raise: num_simultaneous_rts is already the
    // spec maximum on the default path, so this is a no-op kept for symmetry.
    // `max_vertex_blend_matrices` and `max_point_size` need no raise: the
    // truthful values in `fill_default` are already what the path supports.
    caps.num_simultaneous_rts = caps
        .num_simultaneous_rts
        .max(D3D_MAX_SIMULTANEOUS_RENDERTARGETS);
}

#[cfg(test)]
mod tests;
