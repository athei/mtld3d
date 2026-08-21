//! Unit tests for the `D3DCAPS9` defaults and the `debug.capsAll` override.
//!
//! Most bitmask fields covered here are respelled bit by bit, so a bit dropped while its consumer
//! is live fails; `z_cmp_caps`, `alpha_cmp_caps` and `stencil_caps` pin the whole spec field, and
//! many fields `fill_default` writes carry no assertion. Other cases tie a cap to its backing
//! code: clip planes to the vertex-shader uniform, active lights to the FF light slots. The
//! `debug.capsAll` fill is pinned as a superset that leaves shader versions and SM2.x caps alone.

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
fn user_clip_planes_match_the_uniform_capacity() {
    // 3DMark05 requires at least one; the uniform carries six, the
    // D3D9-era hardware figure, one `[[clip_distance]]` lane each.
    let caps = filled();
    assert_eq!(
        usize::try_from(caps.max_user_clip_planes).expect("fits"),
        crate::vs_draw::MAX_CLIP_PLANES
    );
    assert!(caps.max_user_clip_planes >= 1);
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
    // CLIPPLANESCALEDPOINTS stays off: scaled points are clipped as
    // points, not as the quads they rasterize to.
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
    let expected =
        ShadeCaps::COLORGOURAUDRGB | ShadeCaps::SPECULARGOURAUDRGB | ShadeCaps::ALPHAGOURAUDBLEND;
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
fn fvf_caps_carry_a_texcoord_count_and_psize() {
    // The low 16 bits of FVFCaps are a count, not a bitmask; PSIZE is the
    // one flag bit, backed by the FF VS reading the per-vertex size.
    let caps = filled();
    assert_eq!(
        caps.fvf_caps & FvfCaps::TEXCOORDCOUNTMASK.bits(),
        FF_TEXTURE_STAGES
    );
    assert_ne!(caps.fvf_caps & FvfCaps::PSIZE.bits(), 0);
}

#[test]
fn point_size_cap_is_the_render_state_default() {
    // D3D9 defines the D3DRS_POINTSIZE_MAX default as MaxPointSize; both
    // read mtld3d_types::MAX_POINT_SIZE, and 3DMark05 requires >= 64.
    let caps = filled();
    assert_eq!(
        caps.max_point_size.to_bits(),
        mtld3d_types::MAX_POINT_SIZE.to_bits()
    );
    assert!(caps.max_point_size >= 64.0);
    let defaults = mtld3d_types::render_state_defaults();
    assert_eq!(
        defaults[mtld3d_types::D3DRS_POINTSIZE_MAX as usize],
        caps.max_point_size.to_bits()
    );
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
fn advertise_all_raises_mrt_and_stencil() {
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
    // max_point_size is already truthful in fill_default; no raise.
    assert_eq!(
        caps.max_point_size.to_bits(),
        filled().max_point_size.to_bits()
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
