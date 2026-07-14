use mtld3d_types::{D3DCAPS9, D3DDEVCAPS2_CAN_STRETCHRECT_FROM_TEXTURES};

const fn fill_default(caps: &mut D3DCAPS9) {
    // SAFETY: `caps` is a valid `&mut D3DCAPS9`; zeroing all bytes is sound
    // because every field is a primitive integer (no Drop, no padding invariants).
    unsafe { core::ptr::write_bytes(std::ptr::from_mut::<D3DCAPS9>(caps), 0, 1) };

    // Caps are a *truthful floor* under current capability: advertising only
    // what the renderer actually implements. Re-add bits in the same commit
    // that lands the feature.
    caps.device_type = 1; // D3DDEVTYPE_HAL
    caps.caps2 = 0x7002_0000; // CANMANAGERESOURCE | DYNAMICTEXTURES | FULLSCREENGAMMA | CANAUTOGENMIPMAP
    caps.caps3 = 0x0000_0020; // ALPHA_FULLSCREEN_FLIP_OR_DISCARD
    caps.cursor_caps = 0x0000_0001; // D3DCURSORCAPS_COLOR — Win32 HCURSOR path is live
    caps.dev_caps = 0x0001_05F0; // EXECUTESYSTEMMEMORY | EXECUTEVIDEOMEMORY | TLVERTEXSYSTEMMEMORY
    // | TLVERTEXVIDEOMEMORY | PUREDEVICE | DRAWPRIMTLVERTEX | HWTRANSFORMANDLIGHT
    caps.primitive_misc_caps = 0x0022_0AF2; // MASKZ (0x02) | CULLNONE (0x10) | CULLCW (0x20)
    // | CULLCCW (0x40) | COLORWRITEENABLE (0x80) | CLIPTLVERTS (0x200) | BLENDOP (0x800)
    // | SEPARATEALPHABLEND (0x2_0000) | POSTBLENDSRGBCONVERT (0x20_0000)
    // CLIPTLVERTS is factual: Metal clips every vertex — including the
    // pre-transformed XYZRHW (TL) verts we hand through as clip-space — to the
    // NDC volume, so post-transform clipping always happens.
    // ZTEST | FOGVERTEX | FOGRANGE | ZFOG | ANISOTROPY | SLOPESCALEDEPTHBIAS
    // | DEPTHBIAS | SCISSORTEST. DEPTHBIAS + SLOPESCALEDEPTHBIAS reflect the
    // explicit-RS bias path that's been Consumed and reaches Metal's
    // setDepthBias:slopeScale:clamp: per draw. ANISOTROPY advertises the
    // wired path D3DSAMP_MAXANISOTROPY → SamplerSnapshot.max_anisotropy →
    // CreateSamplerStateParams.max_anisotropy → setMaxAnisotropy: on
    // MTLSamplerDescriptor; without the cap bit, well-behaved games
    // clamp to MAXANISOTROPY=1 and never ask for it.
    caps.raster_caps = 0x0723_0090;
    caps.z_cmp_caps = 0xFF; // all 8 comparison functions
    caps.alpha_cmp_caps = 0xFF;
    caps.src_blend_caps = 0x27FF; // ZERO..SRCALPHASAT | BLENDFACTOR
    caps.dest_blend_caps = 0x27FF;
    caps.shade_caps = 0x0000_4208; // COLORGOURAUDRGB (0x08) | SPECULARGOURAUDRGB (0x200)
    // | ALPHAGOURAUDBLEND (0x4000) — FF VS emits out.color0 + out.color1
    caps.texture_caps = 0x0000_4405; // ALPHA (0x04) | PERSPECTIVE (0x01)
    // | PROJECTED (0x400) | MIPMAP (0x4000). POW2 (0x2) and NONPOW2CONDITIONAL
    // (0x100) are BOTH clear: Metal supports non-power-of-2 textures
    // unconditionally (mipmaps + wrap addressing), which D3D9 signals by leaving
    // both flags off. NONPOW2CONDITIONAL is only valid alongside POW2. Texture
    // creation accepts arbitrary sizes; TTFF_PROJECTED honored in FF PS sample
    // emission; PERSPECTIVE is a factual statement — Metal interpolates
    // perspective-correctly by default and we never emit `[[flat]]` /
    // `[[no_perspective]]` qualifiers in dxso::emit.
    caps.texture_filter_caps = 0x0303_0700; // MINFPOINT | MINFLINEAR | MINFANISOTROPIC
    // | MIPFPOINT | MIPFLINEAR | MAGFPOINT | MAGFLINEAR
    // WRAP | MIRROR | CLAMP | BORDER | INDEPENDENTUV | MIRRORONCE. MIRRORONCE
    // (D3DTADDRESS_MIRRORONCE = 5) maps to AddressMode::MirrorClampToEdge in
    // `convert::d3d_to_metal_address_mode` — same code path as the other four modes.
    caps.texture_address_caps = 0x3F;
    caps.stencil_caps = 0; // no stencil wiring in the depth-stencil state builder
    caps.texture_op_caps = 0x0000_BFFF; // DISABLE | SELECTARG1 | SELECTARG2
    // | MODULATE | MODULATE2X | MODULATE4X | ADD
    // | ADDSIGNED | ADDSIGNED2X | SUBTRACT | ADDSMOOTH
    // | BLENDDIFFUSEALPHA | BLENDTEXTUREALPHA | BLENDFACTORALPHA | BLENDCURRENTALPHA
    // (BLENDTEXTUREALPHAPM = 0x4000 intentionally off — ff.rs doesn't emit it)
    caps.max_texture_blend_stages = 8;
    caps.max_simultaneous_textures = 8;
    caps.max_texture_width = 16384;
    caps.max_texture_height = 16384;
    caps.max_texture_repeat = 8192;
    caps.max_texture_aspect_ratio = 16384;
    caps.max_anisotropy = 16;
    caps.max_vertex_w = 1e10;
    caps.line_caps = 0x0F; // TEXTURE | ZTEST | BLEND | ALPHACMP
    caps.fvf_caps = 0x0000_0008; // 8 texcoords (no PSIZE until FVF decoder handles it)
    // TEXGEN (0x01) | MATERIALSOURCE7 (0x02) | DIRECTIONALLIGHTS (0x08)
    // | POSITIONALLIGHTS (0x10) | LOCALVIEWER (0x20) — FF VS honors TCI
    // texgen modes, all three light types (directional / point / spot
    // cone), and both specular view-vector models via D3DRS_LOCALVIEWER.
    caps.vertex_processing_caps = 0x0000_003B;
    caps.max_active_lights = 8;
    // Hardware vertex blending is wired: FfState carries world_palette[256]
    // routed from D3DTS_WORLDMATRIX(i), build_vs_constants packs the active
    // bones × 4 rows after the existing layout, emit_vs blends position +
    // normal via the explicit-weight loop with implicit last weight. The
    // spec-max-per-vertex cap is 4 matrices (D3DVBF_3WEIGHTS = 3 weights
    // + 1 implicit). Palette size is the separate D3DTS_WORLDMATRIX(0..255)
    // range; D3DCAPS9 has no field for it (games discover by trial).
    caps.max_vertex_blend_matrices = 4;
    caps.max_point_size = mtld3d_types::MAX_POINT_SIZE;
    caps.max_primitive_count = 0x0055_5555;
    caps.vertex_shader_version = 0xFFFE_0300; // VS 3.0
    caps.max_vertex_shader_const = 256;
    caps.pixel_shader_version = 0xFFFF_0300; // PS 3.0
    caps.pixel_shader_1x_max_value = 65504.0; // PS 1.x clamp; SM2/SM3 ignore it
    // SM3 capability claims sized to Metal's reality, not the D3D9
    // spec floor. `D3DPS30_INSTRUCTIONSLOTS_MAX` / `D3DVS30_INSTRUCTIONSLOTS_MAX`
    // (32768) is the spec's stated upper bound and what top-tier 2007
    // SM3 cards advertised; Metal has no per-shader instruction limit
    // we'd practically hit, so claiming the spec MAX is honest. Some
    // games key effect-detail variants off these values — advertising
    // the floor (512) could pin them to a low-quality path.
    // `*_executed` is theoretical "instructions GPU can execute per
    // dispatch" — `u32::MAX` says "no enforced cap", matching what
    // Metal-backed drivers actually deliver.
    caps.max_vertex_shader_30_instruction_slots = 32768;
    caps.max_pixel_shader_30_instruction_slots = 32768;
    caps.max_v_shader_instructions_executed = u32::MAX;
    caps.max_p_shader_instructions_executed = u32::MAX;
    caps.max_vertex_index = 0x00FF_FFFF;
    caps.max_streams = 16;
    caps.max_stream_stride = 508;
    // D3DDTCAPS_* bits for the D3DDECLTYPE variants `decl_type_to_metal_format`
    // accepts. FLOAT1/2/3/4 and D3DCOLOR are baseline (no bit). UDEC3/DEC3N
    // are rejected (no Metal equivalent), so those bits stay off.
    //   UBYTE4   (0x001) | UBYTE4N  (0x002)
    //   SHORT2N  (0x004) | SHORT4N  (0x008)
    //   USHORT2N (0x010) | USHORT4N (0x020)
    //   FLOAT16_2(0x100) | FLOAT16_4(0x200)
    caps.decl_types = 0x0000_033F;
    // Single render target only — `SetRenderTarget(index > 0)` is explicitly
    // rejected on the device-side. Multi-pass support for RT0 (water
    // reflections, portrait models) does not need a bit in D3DCAPS9; D3D9
    // advertises render-target format support through `CheckDeviceFormat`
    // rather than a `D3DCAPS9` flag.
    caps.num_simultaneous_rts = 1;
    // StretchRect from a texture-level surface into a render target is supported
    // by the blit path, so advertise it (the only DEVCAPS2 bit truthful on the
    // default path; `apply_advertise_all` later widens to `ALL_DEV_CAPS2`).
    caps.dev_caps2 = D3DDEVCAPS2_CAN_STRETCHRECT_FROM_TEXTURES;
}

/// Single entry point for both `IDirect3D9::GetDeviceCaps` and `IDirect3DDevice9::GetDeviceCaps`.
///
/// Runs `fill_default`, then OR-in every `ALL_*_CAPS` mask when
/// `caps_all` is `true` (the resolved `debug.capsAll` from
/// `mtld3d.conf`). The override is process-wide — no per-call-site
/// opt-in — so games can't accidentally see a half-advertised cap
/// set.
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

/// Bring-up diagnostic: over-advertise caps only where the fallout would show up in the log.
///
/// OR-in spec-max bits for every bitmask field whose consumer warn coverage
/// is solid, and raise the handful of numeric fields whose
/// attempted-but-unimplemented paths have detection hooks upstream (stencil
/// RS / `SetRenderTarget`-index / vertex-decl `BLENDWEIGHT` /
/// `D3DPT_POINTLIST` draw). Skips `pixel_shader_version` /
/// `vertex_shader_version` / `vs20_caps` / `ps20_caps` — DXSO parser has
/// zero warn coverage, so a shader-version bump risks silent miscompile
/// with no log signal.
fn apply_advertise_all(caps: &mut D3DCAPS9) {
    caps.raster_caps |= ALL_RASTER_CAPS;
    caps.texture_caps |= ALL_TEXTURE_CAPS_EXCEPT_POW2;
    caps.texture_filter_caps |= ALL_FILTER_CAPS;
    caps.cube_texture_filter_caps |= ALL_FILTER_CAPS;
    caps.volume_texture_filter_caps |= ALL_FILTER_CAPS;
    caps.vertex_texture_filter_caps |= ALL_FILTER_CAPS;
    caps.stretch_rect_filter_caps |= ALL_FILTER_CAPS;
    caps.texture_address_caps |= ALL_ADDRESS_CAPS;
    caps.volume_texture_address_caps |= ALL_ADDRESS_CAPS;
    caps.src_blend_caps |= ALL_BLEND_CAPS;
    caps.dest_blend_caps |= ALL_BLEND_CAPS;
    caps.primitive_misc_caps |= ALL_PRIMITIVE_MISC_CAPS;
    caps.shade_caps |= ALL_SHADE_CAPS;
    caps.vertex_processing_caps |= ALL_VTXP_CAPS;
    caps.dev_caps |= ALL_DEV_CAPS;
    caps.dev_caps2 |= ALL_DEV_CAPS2;
    caps.line_caps |= ALL_LINE_CAPS;
    caps.texture_op_caps |= ALL_TEXOP_CAPS;
    // Field-shape (non-bitmask) raises. Each has detection wired upstream
    // so the game's attempts at the path land as warns:
    //  - stencil_caps: D3DRS_STENCIL* are PortCandidate; warn_rs_non_default_once fires.
    //  - num_simultaneous_rts: SetRenderTarget(index>0) already warn!s with the index.
    //  - max_vertex_blend_matrices: D3DRS_VERTEXBLEND falls into NotImplemented
    //    (warn fires); BLENDWEIGHT/BLENDINDICES vertex-decl elements fire at
    //    CreateVertexDeclaration time.
    //  - max_point_size: D3DPT_POINTLIST draws fire a log_once_warn in
    //    d3d_to_metal_primitive (Metal still renders 1-pixel points).
    caps.stencil_caps |= ALL_STENCIL_CAPS;
    caps.num_simultaneous_rts = caps.num_simultaneous_rts.max(4);
    // `max_vertex_blend_matrices` is now the truthful floor (4 in
    // `fill_default`) — no longer env-mode-gated.
    caps.max_point_size = caps.max_point_size.max(64.0);
}

// Spec-defined bit masks per D3DCAPS9 field. Composed from the D3DPxxx_*
// enums in d3d9caps.h. Used only by `apply_advertise_all`; the per-field
// constants here are NOT consumer-verified — that's the whole point of
// the diagnostic, surface what the game tries to use that we don't
// implement.

const ALL_RASTER_CAPS: u32 = 0x0F77_A191; // DITHER, ZTEST, FOGVERTEX, FOGTABLE,
// MIPMAPLODBIAS, ZBUFFERLESSHSR, FOGRANGE, ANISOTROPY, WBUFFER, WFOG, ZFOG,
// COLORPERSPECTIVE, SCISSORTEST, SLOPESCALEDEPTHBIAS, DEPTHBIAS,
// MULTISAMPLE_TOGGLE

// Everything from D3DPTEXTURECAPS_* except POW2 (0x01) and NONPOW2CONDITIONAL
// (0x100). POW2 set means "textures must be pow2" — a restriction we don't
// have. NONPOW2CONDITIONAL is only valid alongside POW2 (it means "non-pow2
// conditionally, with restrictions"); since we support non-pow2 unconditionally,
// both stay clear so games see full NPOT support even in capsAll mode.
const ALL_TEXTURE_CAPS_EXCEPT_POW2: u32 = 0x0027_EEE2;

const ALL_FILTER_CAPS: u32 = 0x1F03_1F00; // MIN/MAG/MIP × POINT/LINEAR/ANISOTROPIC
// + PYRAMIDALQUAD / GAUSSIANQUAD variants

const ALL_ADDRESS_CAPS: u32 = 0x3F; // WRAP | MIRROR | CLAMP | BORDER |
// INDEPENDENTUV | MIRRORONCE

const ALL_BLEND_CAPS: u32 = 0xFFFF; // every D3DPBLENDCAPS_* including
// SRCCOLOR2/INVSRCCOLOR2 dual-source SM3 bits; `convert::d3d_to_metal_blend`
// warns on unmapped D3DBLEND values

const ALL_PRIMITIVE_MISC_CAPS: u32 = 0x003F_DFF6; // MASKZ through
// POSTBLENDSRGBCONVERT; skips obsolete LINEPATTERNREP and undefined 0x2000

const ALL_SHADE_CAPS: u32 = 0x0008_4208; // COLORGOURAUDRGB,
// SPECULARGOURAUDRGB, ALPHAGOURAUDBLEND, FOGGOURAUD — the FLAT variants
// were DX7-era and aren't part of the D3D9 spec

const ALL_VTXP_CAPS: u32 = 0x0000_037B; // TEXGEN, MATERIALSOURCE7,
// DIRECTIONALLIGHTS, POSITIONALLIGHTS, LOCALVIEWER, TWEENING,
// TEXGEN_SPHEREMAP, NO_TEXGEN_NONLOCALVIEWER

const ALL_DEV_CAPS: u32 = 0x01FF_FFF0; // every D3DDEVCAPS_* from
// EXECUTESYSTEMMEMORY through NPATCHES

const ALL_DEV_CAPS2: u32 = 0x0000_005F; // STREAMOFFSET, DMAPNPATCH,
// ADAPTIVETESSRTPATCH, ADAPTIVETESSNPATCH, CAN_STRETCHRECT_FROM_TEXTURES,
// PRESAMPLEDDMAPNPATCH, VERTEXELEMENTSCANSHARESTREAMOFFSET (subset of
// known D3DDEVCAPS2_* bits)

const ALL_LINE_CAPS: u32 = 0x3F; // TEXTURE | ZTEST | BLEND | ALPHACMP | FOG
// | ANTIALIAS

const ALL_TEXOP_CAPS: u32 = 0x07FF_FFFF; // every D3DTEXOPCAPS_* up through
// DOTPRODUCT3, MULTIPLYADD, LERP — ff.rs falls back via the TSS warn

const ALL_STENCIL_CAPS: u32 = 0x0000_01FF; // KEEP | ZERO | REPLACE | INCRSAT
// | DECRSAT | INVERT | INCR | DECR | TWOSIDED. D3DRS_STENCIL* are PortCandidate
// (device.rs rs_classify), so writes fire warn_rs_non_default_once per slot.

#[cfg(test)]
mod tests {
    use mtld3d_types::D3DCAPS9;

    use super::{apply_advertise_all, fill_default};

    const D3DPMISCCAPS_MASKZ: u32 = 0x0000_0002;
    const D3DPMISCCAPS_CULLNONE: u32 = 0x0000_0010;
    const D3DPMISCCAPS_CULLCW: u32 = 0x0000_0020;
    const D3DPMISCCAPS_CULLCCW: u32 = 0x0000_0040;
    const D3DPMISCCAPS_COLORWRITEENABLE: u32 = 0x0000_0080;
    const D3DPMISCCAPS_CLIPPLANESCALEDPOINTS: u32 = 0x0000_0100;
    const D3DPMISCCAPS_CLIPTLVERTS: u32 = 0x0000_0200;
    const D3DPMISCCAPS_BLENDOP: u32 = 0x0000_0800;
    const D3DPMISCCAPS_SEPARATEALPHABLEND: u32 = 0x0002_0000;
    const D3DPMISCCAPS_POSTBLENDSRGBCONVERT: u32 = 0x0020_0000;

    const D3DPTEXTURECAPS_PERSPECTIVE: u32 = 0x0000_0001;
    const D3DPTEXTURECAPS_POW2: u32 = 0x0000_0002;
    const D3DPTEXTURECAPS_ALPHA: u32 = 0x0000_0004;
    const D3DPTEXTURECAPS_NONPOW2CONDITIONAL: u32 = 0x0000_0100;
    const D3DPTEXTURECAPS_PROJECTED: u32 = 0x0000_0400;
    const D3DPTEXTURECAPS_MIPMAP: u32 = 0x0000_4000;

    const D3DPTADDRESSCAPS_WRAP: u32 = 0x0000_0001;
    const D3DPTADDRESSCAPS_MIRROR: u32 = 0x0000_0002;
    const D3DPTADDRESSCAPS_CLAMP: u32 = 0x0000_0004;
    const D3DPTADDRESSCAPS_BORDER: u32 = 0x0000_0008;
    const D3DPTADDRESSCAPS_INDEPENDENTUV: u32 = 0x0000_0010;
    const D3DPTADDRESSCAPS_MIRRORONCE: u32 = 0x0000_0020;

    const D3DPSHADECAPS_COLORGOURAUDRGB: u32 = 0x0000_0008;
    const D3DPSHADECAPS_SPECULARGOURAUDRGB: u32 = 0x0000_0200;
    const D3DPSHADECAPS_ALPHAGOURAUDBLEND: u32 = 0x0000_4000;

    const D3DVTXPCAPS_TEXGEN: u32 = 0x0000_0001;
    const D3DVTXPCAPS_MATERIALSOURCE7: u32 = 0x0000_0002;
    const D3DVTXPCAPS_DIRECTIONALLIGHTS: u32 = 0x0000_0008;
    const D3DVTXPCAPS_POSITIONALLIGHTS: u32 = 0x0000_0010;
    const D3DVTXPCAPS_LOCALVIEWER: u32 = 0x0000_0020;

    const D3DPRASTERCAPS_ZTEST: u32 = 0x0000_0010;
    const D3DPRASTERCAPS_FOGVERTEX: u32 = 0x0000_0080;
    const D3DPRASTERCAPS_FOGRANGE: u32 = 0x0001_0000;
    const D3DPRASTERCAPS_ANISOTROPY: u32 = 0x0002_0000;
    const D3DPRASTERCAPS_ZFOG: u32 = 0x0020_0000;
    const D3DPRASTERCAPS_SCISSORTEST: u32 = 0x0100_0000;
    const D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS: u32 = 0x0200_0000;
    const D3DPRASTERCAPS_DEPTHBIAS: u32 = 0x0400_0000;

    fn filled() -> D3DCAPS9 {
        // Calls `fill_default` directly so the assertions describe the
        // baseline cap set independent of any `caps_all` override.
        // SAFETY: D3DCAPS9 is POD (all integer fields, no Drop); zero is
        // a valid initial state that `fill_default` then overwrites.
        let mut caps: D3DCAPS9 = unsafe { core::mem::zeroed() };
        fill_default(&mut caps);
        caps
    }

    // Each bit below is backed by a Consumed classifier arm — the test
    // fails if a future edit silently drops one from caps while its
    // consumer is still live.
    #[test]
    fn primitive_misc_caps_matches_implementation() {
        let expected = D3DPMISCCAPS_MASKZ
            | D3DPMISCCAPS_CULLNONE
            | D3DPMISCCAPS_CULLCW
            | D3DPMISCCAPS_CULLCCW
            | D3DPMISCCAPS_COLORWRITEENABLE
            | D3DPMISCCAPS_CLIPTLVERTS
            | D3DPMISCCAPS_BLENDOP
            | D3DPMISCCAPS_SEPARATEALPHABLEND
            | D3DPMISCCAPS_POSTBLENDSRGBCONVERT;
        assert_eq!(filled().primitive_misc_caps, expected);
        // CLIPPLANESCALEDPOINTS is meaningless given max_point_size = 1.0.
        assert_eq!(
            filled().primitive_misc_caps & D3DPMISCCAPS_CLIPPLANESCALEDPOINTS,
            0
        );
    }

    #[test]
    fn raster_caps_matches_implementation() {
        let expected = D3DPRASTERCAPS_ZTEST
            | D3DPRASTERCAPS_FOGVERTEX
            | D3DPRASTERCAPS_FOGRANGE
            | D3DPRASTERCAPS_ANISOTROPY
            | D3DPRASTERCAPS_ZFOG
            | D3DPRASTERCAPS_SCISSORTEST
            | D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS
            | D3DPRASTERCAPS_DEPTHBIAS;
        assert_eq!(filled().raster_caps, expected);
    }

    #[test]
    fn texture_caps_matches_implementation() {
        let expected = D3DPTEXTURECAPS_ALPHA
            | D3DPTEXTURECAPS_PERSPECTIVE
            | D3DPTEXTURECAPS_PROJECTED
            | D3DPTEXTURECAPS_MIPMAP;
        assert_eq!(filled().texture_caps, expected);
        // POW2 ("textures must be pow2") and NONPOW2CONDITIONAL (valid only with
        // POW2) are both clear — we support non-pow2 unconditionally.
        assert_eq!(filled().texture_caps & D3DPTEXTURECAPS_POW2, 0);
        assert_eq!(
            filled().texture_caps & D3DPTEXTURECAPS_NONPOW2CONDITIONAL,
            0
        );
    }

    #[test]
    fn texture_address_caps_matches_implementation() {
        let expected = D3DPTADDRESSCAPS_WRAP
            | D3DPTADDRESSCAPS_MIRROR
            | D3DPTADDRESSCAPS_CLAMP
            | D3DPTADDRESSCAPS_BORDER
            | D3DPTADDRESSCAPS_INDEPENDENTUV
            | D3DPTADDRESSCAPS_MIRRORONCE;
        assert_eq!(filled().texture_address_caps, expected);
    }

    #[test]
    fn shade_caps_matches_implementation() {
        let expected = D3DPSHADECAPS_COLORGOURAUDRGB
            | D3DPSHADECAPS_SPECULARGOURAUDRGB
            | D3DPSHADECAPS_ALPHAGOURAUDBLEND;
        assert_eq!(filled().shade_caps, expected);
    }

    #[test]
    fn vertex_processing_caps_matches_implementation() {
        let expected = D3DVTXPCAPS_TEXGEN
            | D3DVTXPCAPS_MATERIALSOURCE7
            | D3DVTXPCAPS_DIRECTIONALLIGHTS
            | D3DVTXPCAPS_POSITIONALLIGHTS
            | D3DVTXPCAPS_LOCALVIEWER;
        assert_eq!(filled().vertex_processing_caps, expected);
    }

    #[test]
    fn stencil_remains_unimplemented() {
        assert_eq!(filled().stencil_caps, 0);
    }

    #[test]
    fn vertex_blending_advertises_four_matrices_per_vertex() {
        // FF VS hardware vertex blending is wired end-to-end: D3DTS_WORLDMATRIX(i)
        // → FfState::world_palette, build_vs_constants packs active bones,
        // emit_vs blends position + normal. D3DVBF_3WEIGHTS (3 weights + 1
        // implicit) is the spec maximum per vertex.
        assert_eq!(filled().max_vertex_blend_matrices, 4);
    }

    #[test]
    fn shader_versions_advertise_sm3() {
        // D3D9 packs the version as 0xFFFE_<major><minor> for VS,
        // 0xFFFF_<major><minor> for PS. Bumping the major component
        // changes the wire value the runtime inspects to gate which
        // shader bytecode versions the game compiles against.
        assert_eq!(filled().vertex_shader_version, 0xFFFE_0300);
        assert_eq!(filled().pixel_shader_version, 0xFFFF_0300);
    }

    #[test]
    fn sm3_instruction_slots_advertise_spec_max() {
        // `D3DVS30_INSTRUCTIONSLOTS_MAX` / `D3DPS30_INSTRUCTIONSLOTS_MAX`
        // = 32768 is the SM3 spec ceiling and what 2007-era top-tier
        // SM3 cards advertised. Metal has no per-shader instruction
        // limit we'd practically hit; advertising the floor (512)
        // could pin some games to low-quality effect variants.
        assert_eq!(filled().max_vertex_shader_30_instruction_slots, 32768);
        assert_eq!(filled().max_pixel_shader_30_instruction_slots, 32768);
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
    fn vs20_ps20_caps_remain_zero_until_sm2_extensions_implemented() {
        // We don't advertise predication, dynamic flow control, or other
        // SM2.x extensions yet. Leave these structs zero — flipping any
        // bit must land in the same commit as the implementation.
        let caps = filled();
        assert_eq!(caps.vs20_caps.caps, 0);
        assert_eq!(caps.vs20_caps.dynamic_flow_control_depth, 0);
        assert_eq!(caps.vs20_caps.static_flow_control_depth, 0);
        assert_eq!(caps.ps20_caps.caps, 0);
        assert_eq!(caps.ps20_caps.dynamic_flow_control_depth, 0);
        assert_eq!(caps.ps20_caps.static_flow_control_depth, 0);
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
        // DXSO parser has no warn coverage — bumping shader_version or
        // SM2.x sub-struct fields invites silent shader miscompiles, with
        // no log signal to catch the bad path. Everything else now has
        // upstream detection (RS warns, vertex-decl warn,
        // d3d_to_metal_primitive POINTLIST warn) so it moved into the
        // diagnostic OR-in.
        let caps = advertised();
        assert_eq!(caps.vertex_shader_version, 0xFFFE_0300);
        assert_eq!(caps.pixel_shader_version, 0xFFFF_0300);
        assert_eq!(caps.vs20_caps.caps, 0);
        assert_eq!(caps.ps20_caps.caps, 0);
    }

    #[test]
    fn advertise_all_raises_mrt_stencil_point() {
        let caps = advertised();
        assert!(caps.num_simultaneous_rts >= 4, "MRT raise");
        assert_eq!(caps.stencil_caps, 0x0000_01FF, "stencil full mask");
        assert!(caps.max_point_size >= 64.0, "point-size raise");
        // vertex_blend_matrices moved to fill_default (truthful floor 4).
        assert_eq!(caps.max_vertex_blend_matrices, 4);
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
                default_caps.line_caps,
                advertised_caps.line_caps,
                "line_caps",
            ),
            (
                default_caps.texture_op_caps,
                advertised_caps.texture_op_caps,
                "texture_op_caps",
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
