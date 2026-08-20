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

/// Point size the caps diagnostic raises `MaxPointSize` to.
///
/// Large enough that a game which gates its point-sprite path on the cap
/// takes it, so `D3DPT_POINTLIST` draws reach the `d3d_to_metal_primitive`
/// warn instead of being skipped upstream.
const ADVERTISE_ALL_MAX_POINT_SIZE: f32 = 64.0;

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
const RASTER_DEFAULT: RasterCaps = RasterCaps::ZTEST
    .union(RasterCaps::FOGVERTEX)
    .union(RasterCaps::FOGRANGE)
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
/// four modes.
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

/// FVF caps: the texture-coordinate-set count, with no flag bits.
///
/// `PSIZE` stays off until the FVF decoder handles a point-size element.
const FVF_DEFAULT: FvfCaps = FvfCaps::texcoord_sets(FF_TEXTURE_STAGES);

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
pub fn fill(caps: &mut D3DCAPS9, caps_all: bool) {
    fill_default(caps);
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
    // `vertex_texture_filter_caps` stays zero. That is a legal SM3 shape (ATI's
    // R5xx shipped it), and `CheckDeviceFormat` denies
    // `D3DUSAGE_QUERY_VERTEXTEXTURE` to match.
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
    // Field-shape (non-bitmask) raises. Each has detection wired upstream
    // so the game's attempts at the path land as warns:
    //  - num_simultaneous_rts: already the spec maximum on the default path,
    //    so the raise is a no-op kept for symmetry.
    //  - max_point_size: D3DPT_POINTLIST draws fire a log_once_warn in
    //    d3d_to_metal_primitive (Metal still renders 1-pixel points).
    // `max_vertex_blend_matrices` needs no raise: the truthful floor in
    // `fill_default` is already the spec maximum.
    caps.num_simultaneous_rts = caps
        .num_simultaneous_rts
        .max(D3D_MAX_SIMULTANEOUS_RENDERTARGETS);
    caps.max_point_size = caps.max_point_size.max(ADVERTISE_ALL_MAX_POINT_SIZE);
}

#[cfg(test)]
mod tests {
    use mtld3d_types::{
        AddressCaps, BlendCaps, CmpCaps, D3DCAPS9, DeclTypeCaps, DevCaps, DevCaps2, FilterCaps,
        FvfCaps, LineCaps, PrimitiveMiscCaps, RasterCaps, ShadeCaps, StencilCaps, TexOpCaps,
        TextureCaps, VtxpCaps, d3dps_version, d3dvs_version,
    };

    use super::{FF_TEXTURE_STAGES, apply_advertise_all, fill_default};

    fn filled() -> D3DCAPS9 {
        // Calls `fill_default` directly so the assertions describe the
        // baseline cap set independent of any `caps_all` override.
        // SAFETY: D3DCAPS9 is POD (all integer fields, no Drop); zero is
        // a valid initial state that `fill_default` then overwrites.
        let mut caps: D3DCAPS9 = unsafe { core::mem::zeroed() };
        fill_default(&mut caps);
        caps
    }

    #[test]
    fn default_caps_advertise_hardware_rasterization() {
        assert_ne!(
            filled().dev_caps & DevCaps::HWRASTERIZATION.bits(),
            0,
            "Metal-backed HAL must advertise hardware rasterization"
        );
    }

    // Each field below is spelled out bit by bit, independently of the
    // constant `fill_default` reads, because every bit is backed by a
    // Consumed classifier arm: the test fails if a future edit silently
    // drops one from caps while its consumer is still live.
    #[test]
    fn primitive_misc_caps_matches_implementation() {
        let expected = PrimitiveMiscCaps::MASKZ
            | PrimitiveMiscCaps::CULLNONE
            | PrimitiveMiscCaps::CULLCW
            | PrimitiveMiscCaps::CULLCCW
            | PrimitiveMiscCaps::COLORWRITEENABLE
            | PrimitiveMiscCaps::CLIPTLVERTS
            | PrimitiveMiscCaps::BLENDOP
            | PrimitiveMiscCaps::INDEPENDENTWRITEMASKS
            | PrimitiveMiscCaps::SEPARATEALPHABLEND
            | PrimitiveMiscCaps::MRTINDEPENDENTBITDEPTHS
            | PrimitiveMiscCaps::MRTPOSTPIXELSHADERBLENDING
            | PrimitiveMiscCaps::POSTBLENDSRGBCONVERT;
        assert_eq!(filled().primitive_misc_caps, expected.bits());
        // CLIPPLANESCALEDPOINTS is meaningless given max_point_size = 1.0.
        assert_eq!(
            filled().primitive_misc_caps & PrimitiveMiscCaps::CLIPPLANESCALEDPOINTS.bits(),
            0
        );
    }

    #[test]
    fn raster_caps_matches_implementation() {
        let expected = RasterCaps::ZTEST
            | RasterCaps::FOGVERTEX
            | RasterCaps::FOGRANGE
            | RasterCaps::ANISOTROPY
            | RasterCaps::ZFOG
            | RasterCaps::SCISSORTEST
            | RasterCaps::SLOPESCALEDEPTHBIAS
            | RasterCaps::DEPTHBIAS;
        assert_eq!(filled().raster_caps, expected.bits());
    }

    #[test]
    fn texture_caps_matches_implementation() {
        let expected = TextureCaps::ALPHA
            | TextureCaps::PERSPECTIVE
            | TextureCaps::PROJECTED
            | TextureCaps::MIPMAP
            | TextureCaps::CUBEMAP
            | TextureCaps::MIPCUBEMAP
            | TextureCaps::VOLUMEMAP
            | TextureCaps::MIPVOLUMEMAP;
        assert_eq!(filled().texture_caps, expected.bits());
    }

    #[test]
    fn texture_restriction_bits_never_advertised() {
        // POW2 / SQUAREONLY / CUBEMAP_POW2 / VOLUMEMAP_POW2 each claim a
        // creation-time limitation the renderer does not have, and
        // NONPOW2CONDITIONAL is only meaningful alongside POW2. Non-power-of-two
        // support is unconditional, so the whole group stays clear, including
        // under the diagnostic, which otherwise widens the field to every spec bit.
        assert_eq!(
            filled().texture_caps & TextureCaps::RESTRICTIONS.bits(),
            0,
            "default caps advertise a texture restriction"
        );
        assert_eq!(
            advertised().texture_caps & TextureCaps::RESTRICTIONS.bits(),
            0,
            "capsAll advertises a texture restriction"
        );
    }

    #[test]
    fn texture_address_caps_matches_implementation() {
        let expected = AddressCaps::WRAP
            | AddressCaps::MIRROR
            | AddressCaps::CLAMP
            | AddressCaps::BORDER
            | AddressCaps::INDEPENDENTUV
            | AddressCaps::MIRRORONCE;
        assert_eq!(filled().texture_address_caps, expected.bits());
    }

    #[test]
    fn filter_caps_matches_implementation() {
        let expected = FilterCaps::MINFPOINT
            | FilterCaps::MINFLINEAR
            | FilterCaps::MINFANISOTROPIC
            | FilterCaps::MIPFPOINT
            | FilterCaps::MIPFLINEAR
            | FilterCaps::MAGFPOINT
            | FilterCaps::MAGFLINEAR;
        assert_eq!(filled().texture_filter_caps, expected.bits());
    }

    #[test]
    fn blend_caps_matches_implementation() {
        // The contiguous D3DBLEND_ZERO..SRCALPHASAT range plus BLENDFACTOR,
        // which is exactly what `convert::d3d_to_metal_blend` maps.
        let expected = BlendCaps::ZERO
            | BlendCaps::ONE
            | BlendCaps::SRCCOLOR
            | BlendCaps::INVSRCCOLOR
            | BlendCaps::SRCALPHA
            | BlendCaps::INVSRCALPHA
            | BlendCaps::DESTALPHA
            | BlendCaps::INVDESTALPHA
            | BlendCaps::DESTCOLOR
            | BlendCaps::INVDESTCOLOR
            | BlendCaps::SRCALPHASAT
            | BlendCaps::BLENDFACTOR;
        assert_eq!(filled().src_blend_caps, expected.bits());
        assert_eq!(filled().dest_blend_caps, expected.bits());
    }

    #[test]
    fn compare_caps_advertise_every_function() {
        // Both the depth test and the alpha test route every D3DCMP_* value
        // through `convert::d3d_to_metal_compare`.
        assert_eq!(filled().z_cmp_caps, CmpCaps::all().bits());
        assert_eq!(filled().alpha_cmp_caps, CmpCaps::all().bits());
        assert!(CmpCaps::all().contains(CmpCaps::NEVER | CmpCaps::ALWAYS));
    }

    #[test]
    fn shade_caps_matches_implementation() {
        let expected = ShadeCaps::COLORGOURAUDRGB
            | ShadeCaps::SPECULARGOURAUDRGB
            | ShadeCaps::ALPHAGOURAUDBLEND;
        assert_eq!(filled().shade_caps, expected.bits());
    }

    #[test]
    fn texture_op_caps_exclude_premultiplied_blend() {
        // `dxso::ff` emits every listed op; BLENDTEXTUREALPHAPM has no emitter.
        let expected = TexOpCaps::DISABLE
            | TexOpCaps::SELECTARG1
            | TexOpCaps::SELECTARG2
            | TexOpCaps::MODULATE
            | TexOpCaps::MODULATE2X
            | TexOpCaps::MODULATE4X
            | TexOpCaps::ADD
            | TexOpCaps::ADDSIGNED
            | TexOpCaps::ADDSIGNED2X
            | TexOpCaps::SUBTRACT
            | TexOpCaps::ADDSMOOTH
            | TexOpCaps::BLENDDIFFUSEALPHA
            | TexOpCaps::BLENDTEXTUREALPHA
            | TexOpCaps::BLENDFACTORALPHA
            | TexOpCaps::BLENDCURRENTALPHA;
        assert_eq!(filled().texture_op_caps, expected.bits());
        assert_eq!(
            filled().texture_op_caps & TexOpCaps::BLENDTEXTUREALPHAPM.bits(),
            0
        );
    }

    #[test]
    fn line_caps_matches_implementation() {
        let expected = LineCaps::TEXTURE | LineCaps::ZTEST | LineCaps::BLEND | LineCaps::ALPHACMP;
        assert_eq!(filled().line_caps, expected.bits());
    }

    #[test]
    fn decl_types_exclude_unmappable_formats() {
        // `decl_type_to_metal_format` rejects UDEC3 / DEC3N: no Metal equivalent.
        let expected = DeclTypeCaps::UBYTE4
            | DeclTypeCaps::UBYTE4N
            | DeclTypeCaps::SHORT2N
            | DeclTypeCaps::SHORT4N
            | DeclTypeCaps::USHORT2N
            | DeclTypeCaps::USHORT4N
            | DeclTypeCaps::FLOAT16_2
            | DeclTypeCaps::FLOAT16_4;
        assert_eq!(filled().decl_types, expected.bits());
        assert_eq!(
            filled().decl_types & (DeclTypeCaps::UDEC3 | DeclTypeCaps::DEC3N).bits(),
            0
        );
    }

    #[test]
    fn fvf_caps_carry_a_texcoord_count_not_flags() {
        // The low 16 bits of FVFCaps are a count, not a bitmask. PSIZE stays
        // off until the FVF decoder handles a point-size element.
        let caps = filled();
        assert_eq!(
            caps.fvf_caps & FvfCaps::TEXCOORDCOUNTMASK.bits(),
            FF_TEXTURE_STAGES
        );
        assert_eq!(caps.fvf_caps & FvfCaps::PSIZE.bits(), 0);
    }

    #[test]
    fn vertex_processing_caps_matches_implementation() {
        let expected = VtxpCaps::TEXGEN
            | VtxpCaps::MATERIALSOURCE7
            | VtxpCaps::DIRECTIONALLIGHTS
            | VtxpCaps::POSITIONALLIGHTS
            | VtxpCaps::LOCALVIEWER;
        assert_eq!(filled().vertex_processing_caps, expected.bits());
    }

    #[test]
    fn dev_caps2_advertises_stretchrect_and_stream_offsets() {
        assert_eq!(
            filled().dev_caps2,
            (DevCaps2::CAN_STRETCHRECT_FROM_TEXTURES
                | DevCaps2::STREAMOFFSET
                | DevCaps2::VERTEXELEMENTSCANSHARESTREAMOFFSET)
                .bits()
        );
    }

    #[test]
    fn stencil_advertises_every_operation() {
        // Each D3DSTENCILOP_* maps 1:1 onto an MTLStencilOperation, so the
        // truthful floor is the whole field.
        assert_eq!(filled().stencil_caps, StencilCaps::all().bits());
    }

    #[test]
    fn vertex_blending_advertises_four_matrices_per_vertex() {
        // FF VS hardware vertex blending is wired end-to-end: D3DTS_WORLDMATRIX(i)
        // → FfState::world_palette, build_vs_constants packs active bones,
        // emit_vs blends position + normal. D3DVBF_3WEIGHTS (3 weights + 1
        // implicit) is the spec maximum per vertex.
        assert_eq!(
            filled().max_vertex_blend_matrices,
            mtld3d_types::D3DVBF_3WEIGHTS + 1
        );
    }

    #[test]
    fn active_light_cap_matches_the_ff_slot_count() {
        // The advertised cap and the FF fast-path slot array are the same
        // constant, so a game cannot enable a light the FF VS has no slot for.
        assert_eq!(
            filled().max_active_lights,
            crate::ff_state::MAX_ACTIVE_LIGHTS
        );
    }

    #[test]
    fn shader_versions_advertise_sm3() {
        // D3D9 packs the version as 0xFFFE_<major><minor> for VS,
        // 0xFFFF_<major><minor> for PS. Bumping the major component
        // changes the wire value the runtime inspects to gate which
        // shader bytecode versions the game compiles against.
        assert_eq!(filled().vertex_shader_version, d3dvs_version(3, 0));
        assert_eq!(filled().pixel_shader_version, d3dps_version(3, 0));
        assert_eq!(d3dvs_version(3, 0), 0xFFFE_0300);
        assert_eq!(d3dps_version(3, 0), 0xFFFF_0300);
    }

    #[test]
    fn sm3_instruction_slots_advertise_spec_max() {
        // The `*_INSTRUCTIONSLOTS_MAX` values are the SM3 spec ceiling and
        // what 2007-era top-tier SM3 cards advertised. Metal has no
        // per-shader instruction limit a game would practically hit;
        // advertising the floor (512) could pin some games to low-quality
        // effect variants.
        assert_eq!(
            filled().max_vertex_shader_30_instruction_slots,
            mtld3d_types::D3DVS30_INSTRUCTIONSLOTS_MAX
        );
        assert_eq!(
            filled().max_pixel_shader_30_instruction_slots,
            mtld3d_types::D3DPS30_INSTRUCTIONSLOTS_MAX
        );
    }

    #[test]
    fn sm3_executed_instruction_caps_advertise_no_practical_limit() {
        // `*_executed` is "instructions the GPU can execute per shader
        // dispatch". On a Metal backend there's no enforced cap;
        // `u32::MAX` is the conventional way to advertise "no limit".
        assert_eq!(filled().max_v_shader_instructions_executed, u32::MAX);
        assert_eq!(filled().max_p_shader_instructions_executed, u32::MAX);
    }

    #[test]
    fn vs20_ps20_caps_report_the_spec_maxima() {
        // Every bit is backed by an emitter translation (setp + predicated
        // writes, dsx/dsy, if/ifc/breakc/loop/rep) and MSL imposes none of
        // the ps_2_0 limits the remaining bits lift. A 3.0 device reports the
        // maxima; zeros here made 3DMark05 refuse to start.
        let caps = filled();
        assert_eq!(
            caps.vs20_caps.caps,
            mtld3d_types::Vs20Caps::PREDICATION.bits()
        );
        assert_eq!(caps.vs20_caps.dynamic_flow_control_depth, 24);
        assert_eq!(caps.vs20_caps.num_temps, 32);
        assert_eq!(caps.vs20_caps.static_flow_control_depth, 4);
        assert_eq!(caps.ps20_caps.caps, mtld3d_types::Ps20Caps::all().bits());
        assert_eq!(caps.ps20_caps.dynamic_flow_control_depth, 24);
        assert_eq!(caps.ps20_caps.num_temps, 32);
        assert_eq!(caps.ps20_caps.static_flow_control_depth, 4);
        assert_eq!(caps.ps20_caps.num_instruction_slots, 512);
    }

    #[test]
    fn cube_and_volume_textures_share_the_2d_sampler_caps() {
        let caps = filled();
        assert_eq!(caps.cube_texture_filter_caps, caps.texture_filter_caps);
        assert_eq!(caps.volume_texture_filter_caps, caps.texture_filter_caps);
        assert_eq!(caps.volume_texture_address_caps, caps.texture_address_caps);
        assert_ne!(caps.texture_caps & TextureCaps::VOLUMEMAP.bits(), 0);
        assert_ne!(caps.texture_caps & TextureCaps::MIPVOLUMEMAP.bits(), 0);
        assert_eq!(caps.max_volume_extent, mtld3d_types::MAX_VOLUME_EXTENT);
    }

    #[test]
    fn stretch_rect_filter_caps_match_what_the_call_accepts() {
        let expected = FilterCaps::MINFPOINT
            | FilterCaps::MINFLINEAR
            | FilterCaps::MAGFPOINT
            | FilterCaps::MAGFLINEAR;
        assert_eq!(filled().stretch_rect_filter_caps, expected.bits());
    }

    #[test]
    fn vertex_texture_fetch_is_not_advertised() {
        // No sampler binds on the vertex stage; `CheckDeviceFormat` denies
        // `D3DUSAGE_QUERY_VERTEXTEXTURE` to match.
        assert_eq!(filled().vertex_texture_filter_caps, 0);
    }

    #[test]
    fn presentation_intervals_are_the_two_the_present_path_honours() {
        assert_eq!(
            filled().presentation_intervals,
            mtld3d_types::D3DPRESENT_INTERVAL_ONE | mtld3d_types::D3DPRESENT_INTERVAL_IMMEDIATE
        );
    }

    #[test]
    fn guard_band_and_adapter_group_are_filled() {
        let caps = filled();
        assert_eq!(caps.guard_band_left.to_bits(), (-32768.0f32).to_bits());
        assert_eq!(caps.guard_band_top.to_bits(), (-32768.0f32).to_bits());
        assert_eq!(caps.guard_band_right.to_bits(), 32768.0f32.to_bits());
        assert_eq!(caps.guard_band_bottom.to_bits(), 32768.0f32.to_bits());
        assert_eq!(caps.number_of_adapters_in_group, 1);
    }

    fn advertised() -> D3DCAPS9 {
        // SAFETY: D3DCAPS9 is POD (all integer fields, no Drop); zero is
        // a valid initial state that `fill_default` then overwrites.
        let mut caps: D3DCAPS9 = unsafe { core::mem::zeroed() };
        fill_default(&mut caps);
        apply_advertise_all(&mut caps);
        caps
    }

    #[test]
    fn advertise_all_does_not_touch_silent_miscompile_fields() {
        // DXSO parser has no warn coverage — bumping shader_version invites
        // silent shader miscompiles, with no log signal to catch the bad
        // path, and the SM2.x sub-structs already sit at their maxima.
        // Everything else now has upstream detection (RS warns, vertex-decl
        // warn, d3d_to_metal_primitive POINTLIST warn) so it moved into the
        // diagnostic OR-in.
        let caps = advertised();
        let base = filled();
        assert_eq!(caps.vertex_shader_version, d3dvs_version(3, 0));
        assert_eq!(caps.pixel_shader_version, d3dps_version(3, 0));
        assert_eq!(caps.vs20_caps.caps, base.vs20_caps.caps);
        assert_eq!(caps.ps20_caps.caps, base.ps20_caps.caps);
        assert_eq!(caps.ps20_caps.num_temps, base.ps20_caps.num_temps);
    }

    #[test]
    fn default_path_advertises_four_render_targets() {
        assert_eq!(
            filled().num_simultaneous_rts,
            mtld3d_types::D3D_MAX_SIMULTANEOUS_RENDERTARGETS
        );
    }

    #[test]
    fn advertise_all_raises_mrt_stencil_point() {
        let caps = advertised();
        assert!(
            caps.num_simultaneous_rts >= mtld3d_types::D3D_MAX_SIMULTANEOUS_RENDERTARGETS,
            "MRT raise"
        );
        assert_eq!(caps.stencil_caps, StencilCaps::all().bits(), "stencil ops");
        assert!(
            StencilCaps::all().contains(StencilCaps::KEEP | StencilCaps::TWOSIDED),
            "stencil mask must span the single-sided ops and the two-sided bit"
        );
        assert!(
            caps.max_point_size >= super::ADVERTISE_ALL_MAX_POINT_SIZE,
            "point-size raise"
        );
        // vertex_blend_matrices stays at the truthful floor from fill_default.
        assert_eq!(
            caps.max_vertex_blend_matrices,
            mtld3d_types::D3DVBF_3WEIGHTS + 1
        );
    }

    #[test]
    fn advertise_all_is_superset_of_default() {
        let default_caps = filled();
        let advertised_caps = advertised();
        // Every bit set in the default fill must still be set after the
        // OR-in (catches accidental mask narrowing in apply_advertise_all).
        for (default_bits, advertised_bits, name) in [
            (
                default_caps.raster_caps,
                advertised_caps.raster_caps,
                "raster_caps",
            ),
            (
                default_caps.texture_caps,
                advertised_caps.texture_caps,
                "texture_caps",
            ),
            (
                default_caps.texture_filter_caps,
                advertised_caps.texture_filter_caps,
                "texture_filter_caps",
            ),
            (
                default_caps.texture_address_caps,
                advertised_caps.texture_address_caps,
                "texture_address_caps",
            ),
            (
                default_caps.src_blend_caps,
                advertised_caps.src_blend_caps,
                "src_blend_caps",
            ),
            (
                default_caps.dest_blend_caps,
                advertised_caps.dest_blend_caps,
                "dest_blend_caps",
            ),
            (
                default_caps.primitive_misc_caps,
                advertised_caps.primitive_misc_caps,
                "primitive_misc_caps",
            ),
            (
                default_caps.shade_caps,
                advertised_caps.shade_caps,
                "shade_caps",
            ),
            (
                default_caps.vertex_processing_caps,
                advertised_caps.vertex_processing_caps,
                "vertex_processing_caps",
            ),
            (default_caps.dev_caps, advertised_caps.dev_caps, "dev_caps"),
            (
                default_caps.dev_caps2,
                advertised_caps.dev_caps2,
                "dev_caps2",
            ),
            (
                default_caps.line_caps,
                advertised_caps.line_caps,
                "line_caps",
            ),
            (
                default_caps.texture_op_caps,
                advertised_caps.texture_op_caps,
                "texture_op_caps",
            ),
            (
                default_caps.decl_types,
                advertised_caps.decl_types,
                "decl_types",
            ),
        ] {
            assert_eq!(
                default_bits & advertised_bits,
                default_bits,
                "{name}: advertised mask dropped a default bit"
            );
        }
    }
}
