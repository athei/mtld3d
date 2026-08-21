//! Unit tests for the Fixed-Function pipeline state stored on `DeviceInner`.
//!
//! Pins the light and texture-transform masks the setters maintain (and that a state-block
//! restore must rebuild), the FF shader keys derived from render state (fog source, local
//! viewer, texcoord routing past a disabled color op, vertex blend), sparse lights compacting
//! into dense eye-space shader slots, the const-row extent checked against the `vs_c` rows the
//! emitter reads, and `inverse` round-tripping affine matrices while rejecting singular ones.

use mtld3d_types::{
    D3DMATRIX, D3DRS_DEPTHBIAS, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY, D3DRS_FOGENABLE,
    D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE, D3DRS_TEXTUREFACTOR,
    D3DTOP_MODULATE, D3DTSS_BUMPENVMAT00, D3DTSS_COLOROP, D3DTSS_TEXTURETRANSFORMFLAGS,
    RENDER_STATE_COUNT, render_state_defaults,
};

use super::{FfState, FfVsLayout, VariantFlags, VariantKey, build_fog_color_bytes};
use crate::convert::FfVsLayoutFlags;

fn rs() -> [u32; RENDER_STATE_COUNT] {
    render_state_defaults()
}

#[test]
fn build_ps_constants_is_tfactor_only() {
    let mut states = rs();
    states[D3DRS_TEXTUREFACTOR as usize] = 0xFF80_4020;
    let bytes = FfState::new().build_ps_constants(&states);
    assert_eq!(bytes.len(), 16, "ps_c now only carries texture factor");
    // First float4 decodes back to texture factor (RGBA float).
    let r = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let g = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let b = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let a = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    // 0xFF80_4020 = ARGB(255,128,64,32).
    assert!((r - 128.0 / 255.0).abs() < 1e-4);
    assert!((g - 64.0 / 255.0).abs() < 1e-4);
    assert!((b - 32.0 / 255.0).abs() < 1e-4);
    assert!((a - 1.0).abs() < 1e-4);
}

#[test]
fn fog_color_bytes_empty_when_fog_off() {
    let mut states = rs();
    states[D3DRS_FOGCOLOR as usize] = 0xFFFF_00FF;
    let variant = VariantKey::default();
    assert_eq!(variant.fog_mode, 0);
    assert_eq!(build_fog_color_bytes(&states, variant).1, 0);
}

#[test]
fn projection_is_ortho_treats_negative_zero_as_zero() {
    use mtld3d_types::{D3DMATRIX, D3DTS_PROJECTION};
    let mut ff = FfState::new();
    // Identity's 4th column is (0,0,0,1) → orthographic.
    assert!(ff.projection_is_ortho());
    // A negative-zero in the column must still count as zero.
    let mut proj = D3DMATRIX::IDENTITY;
    proj.m[3] = -0.0;
    proj.m[7] = -0.0;
    proj.m[11] = -0.0;
    ff.set_transform(D3DTS_PROJECTION, &proj);
    assert!(
        ff.projection_is_ortho(),
        "-0.0 in the projection's 4th column must count as 0"
    );
    // A genuine perspective column is not orthographic.
    proj.m[11] = 0.5;
    ff.set_transform(D3DTS_PROJECTION, &proj);
    assert!(!ff.projection_is_ortho());
}

#[test]
fn table_fog_wins_over_vertex_mode_and_keys_source_on_projection() {
    let mut states = rs();
    states[D3DRS_FOGENABLE as usize] = 1;
    states[D3DRS_FOGVERTEXMODE as usize] = 1; // D3DFOG_EXP
    states[D3DRS_FOGTABLEMODE as usize] = 3; // D3DFOG_LINEAR

    // Identity projection (4th column (0,0,0,1)) = orthographic → Z source.
    let mut ff = FfState::new();
    let variant = ff.variant_key(&states, false);
    assert_eq!(variant.fog_mode, 0, "table fog must zero the vertex mode");
    assert_eq!(variant.fog_table_mode, 3);
    assert!(
        !variant.flags.contains(VariantFlags::FOG_SOURCE_W),
        "ortho projection → Z source"
    );

    // Perspective-marked projection (_44 != 1) → W source.
    let mut proj = D3DMATRIX::IDENTITY;
    proj.m[15] = 1.01;
    ff.set_transform(mtld3d_types::D3DTS_PROJECTION, &proj);
    let variant = ff.variant_key(&states, false);
    assert!(
        variant.flags.contains(VariantFlags::FOG_SOURCE_W),
        "non-ortho projection → W source"
    );

    // Table fog applies on the RHW path too.
    let variant = ff.variant_key(&states, true);
    assert_eq!(variant.fog_table_mode, 3);
    assert_eq!(variant.fog_mode, 0);

    // Vertex fog only: no table mode, no source bit churn from the
    // (still perspective) projection.
    states[D3DRS_FOGTABLEMODE as usize] = 0;
    let variant = ff.variant_key(&states, false);
    assert_eq!(variant.fog_mode, 1);
    assert_eq!(variant.fog_table_mode, 0);
    assert!(!variant.flags.contains(VariantFlags::FOG_SOURCE_W));
}

/// `restore_filtered` writes back only the FF state owned by the block type.
///
/// Transforms + material are `All`-only, lights are vertex-pipeline, and
/// texture-stage states split per index. `All` must match `restore_into`.
#[test]
fn restore_filtered_respects_block_type() {
    use mtld3d_types::{
        D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DTOP_DISABLE,
        D3DTOP_MODULATE, D3DTS_VIEW, D3DTSS_TEXCOORDINDEX, StateBlockType,
    };

    use super::FfStateSnapshot;

    const COLOROP: usize = D3DTSS_COLOROP as usize;
    const TCI: usize = D3DTSS_TEXCOORDINDEX as usize;

    // One distinctive value per category.
    let mut src = FfState::new();
    src.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            range: 42.0,
            ..Default::default()
        },
    );
    src.set_light_enabled(0, true);
    let mut view = D3DMATRIX::IDENTITY;
    view.m[0] = 7.0;
    src.set_transform(D3DTS_VIEW, &view);
    src.set_material(&D3DMATERIAL9 {
        power: 13.0,
        ..Default::default()
    });
    src.set_texture_stage_state(0, COLOROP, D3DTOP_DISABLE); // pixel-only TSS
    src.set_texture_stage_state(0, TCI, 5); // vertex + pixel TSS
    let snap = FfStateSnapshot::from(&src);

    // VERTEXSTATE: lights + vertex TSS restored; transforms/material/pixel TSS untouched.
    let mut v = FfState::new();
    snap.restore_filtered(&mut v, StateBlockType::Vertex);
    assert_eq!(
        v.light(0).range.to_bits(),
        42.0_f32.to_bits(),
        "vertex restores lights"
    );
    assert!(v.light_enabled(0), "vertex restores light-enable");
    assert_eq!(
        v.transform(D3DTS_VIEW).unwrap().m[0].to_bits(),
        1.0_f32.to_bits(),
        "vertex leaves transforms at default"
    );
    assert_eq!(
        v.material().power.to_bits(),
        0.0_f32.to_bits(),
        "vertex leaves material"
    );
    assert_eq!(
        v.texture_stage_state(0, COLOROP),
        D3DTOP_MODULATE,
        "vertex leaves pixel-only TSS at stage-0 default"
    );
    assert_eq!(
        v.texture_stage_state(0, TCI),
        5,
        "vertex restores texcoord index"
    );

    // PIXELSTATE: pixel TSS restored; lights/transforms untouched.
    let mut p = FfState::new();
    snap.restore_filtered(&mut p, StateBlockType::Pixel);
    assert_eq!(
        p.light(0).range.to_bits(),
        0.0_f32.to_bits(),
        "pixel leaves lights"
    );
    assert!(!p.light_enabled(0), "pixel leaves light-enable");
    assert_eq!(
        p.texture_stage_state(0, COLOROP),
        D3DTOP_DISABLE,
        "pixel restores color op"
    );
    assert_eq!(
        p.texture_stage_state(0, TCI),
        5,
        "pixel restores texcoord index"
    );

    // ALL: everything restored, identical to restore_into.
    let mut a = FfState::new();
    snap.restore_filtered(&mut a, StateBlockType::All);
    let mut a_ref = FfState::new();
    snap.restore_into(&mut a_ref);
    assert_eq!(a.light(0).range.to_bits(), 42.0_f32.to_bits());
    assert_eq!(
        a.transform(D3DTS_VIEW).unwrap().m[0].to_bits(),
        7.0_f32.to_bits()
    );
    assert_eq!(a.material().power.to_bits(), 13.0_f32.to_bits());
    assert_eq!(a.texture_stage_state(0, COLOROP), D3DTOP_DISABLE);
    assert_eq!(
        a.texture_stage_state(0, TCI),
        a_ref.texture_stage_state(0, TCI),
        "restore_filtered(All) matches restore_into"
    );
}

/// `build_vs_key` must populate `tci_coord_indices` for every stage the VB layout declares.
///
/// This holds for every stage the layout declares an attribute for, even
/// when the FF PS color-blend chain terminates earlier via
/// `D3DTSS_COLOROP == D3DTOP_DISABLE`.
///
/// A programmable PS bound over FF VS commonly samples several textures
/// while the captured FF state leaves stage 1+'s `COLOROP` at its default
/// `DISABLE` (the game doesn't enable FF blending when a programmable PS
/// is bound). Stopping TCI decode at the first `COLOROP_DISABLE` would
/// leave `tci_coord_indices[1..]` at their `[0; 8]` init, routing every
/// VS texcoord output onto `v4`; the PS would then sample every texture
/// at `v4`'s coord set instead of the distinct sets each stage expects,
/// collapsing the intended multi-texture result.
#[test]
fn tci_indices_preserved_past_colorop_disable_terminator() {
    let mut ff = FfState::new();
    // Make stage 0 default-enabled (COLOROP=MODULATE) but leave stages
    // 1+ at their default `COLOROP_DISABLE` — exactly the shape a
    // programmable-PS draw with FF VS produces. Without the fix the
    // loop breaks at stage 1 before reading its TEXCOORDINDEX.
    assert_eq!(
        ff.texture_stage_state(0, D3DTSS_COLOROP as usize),
        D3DTOP_MODULATE,
        "stage 0 default COLOROP must be MODULATE"
    );
    ff.set_texture_stage_state(1, D3DTSS_COLOROP as usize, 1 /* D3DTOP_DISABLE */);

    let layout = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_COLOR0,
        tex_coord_count: 3,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };
    // bound_texture_mask = stages 0/1/2 all have textures bound.
    let key = ff.build_vs_key(&rs(), layout, 0b0000_0111);

    // D3D9 spec default for `D3DTSS_TEXCOORDINDEX` is the stage index.
    // The fix preserves that for stages past the FF PS chain
    // terminator; the broken behaviour collapsed them all to 0.
    assert_eq!(
        &key.tci_coord_indices[..3],
        &[0u8, 1, 2],
        "tci_coord_indices[1..3] must stay populated; collapsing them to 0 \
         would route every FF VS texcoord output onto v4",
    );
    assert_eq!(
        key.tex_coord_count, 3,
        "VS still emits 3 texcoord outputs driven by VB layout"
    );
}

#[test]
fn local_viewer_flag_canonicalizes_on_lighting_and_specular() {
    use mtld3d_types::{D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_SPECULARENABLE};
    let ff = FfState::new();
    let layout = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_NORMAL,
        tex_coord_count: 0,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };

    // RS defaults: LIGHTING=1, LOCALVIEWER=1, SPECULARENABLE=0 — the
    // bit stays clear while no specular term reads V.
    let mut states = rs();
    let key = ff.build_vs_key(&states, layout, 0);
    assert!(!key.local_viewer(), "no specular → no LOCAL_VIEWER bit");

    // Specular on + default LOCALVIEWER=1 → set.
    states[D3DRS_SPECULARENABLE as usize] = 1;
    let key = ff.build_vs_key(&states, layout, 0);
    assert!(key.local_viewer(), "specular + RS default → set");

    // Explicit LOCALVIEWER=0 → infinite viewer.
    states[D3DRS_LOCALVIEWER as usize] = 0;
    let key = ff.build_vs_key(&states, layout, 0);
    assert!(!key.local_viewer(), "RS off → infinite viewer");

    // Lighting off clears it even with specular + localviewer on.
    states[D3DRS_LOCALVIEWER as usize] = 1;
    states[D3DRS_LIGHTING as usize] = 0;
    let key = ff.build_vs_key(&states, layout, 0);
    assert!(!key.local_viewer(), "unlit → no LOCAL_VIEWER bit");
}

#[test]
fn fog_color_bytes_two_rows_when_fog_on() {
    let mut states = rs();
    // D3DCOLOR 0xFF80_40C0 = ARGB(255, 128, 64, 192) → R=128/255, G=64/255, B=192/255, A=1.0
    states[D3DRS_FOGCOLOR as usize] = 0xFF80_40C0;
    states[D3DRS_FOGSTART as usize] = 0.5f32.to_bits();
    states[D3DRS_FOGEND as usize] = 10.0f32.to_bits();
    states[D3DRS_FOGDENSITY as usize] = 2.0f32.to_bits();
    states[D3DRS_DEPTHBIAS as usize] = 0.1f32.to_bits();
    let variant = VariantKey {
        fog_mode: 3,
        ..Default::default()
    };
    let (bytes, len) = build_fog_color_bytes(&states, variant);
    assert_eq!(len, 32);
    let comp = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let (r, g, b, a) = (comp(0), comp(1), comp(2), comp(3));
    assert!((r - 128.0 / 255.0).abs() < 1e-4, "r = {r}");
    assert!((g - 64.0 / 255.0).abs() < 1e-4, "g = {g}");
    assert!((b - 192.0 / 255.0).abs() < 1e-4, "b = {b}");
    assert!((a - 1.0).abs() < 1e-4, "a = {a}");
    // Row 1: (start, end, density, depth-bias), raw f32 bit copies.
    assert_eq!((comp(4), comp(5), comp(6), comp(7)), (0.5, 10.0, 2.0, 0.1));
}

#[test]
fn tss_warn_latch_is_per_stage() {
    // BUMPENVMAT00 is NotImplemented (not in the Consumed list), so a
    // non-default write fires warn_tss_non_default_once. Default for
    // BUMPENVMAT00 is 0; write 1 to stages 0 and 1. A latch keyed only
    // on `ty` would set tss_warn_fired[ty] on stage 0 and silently
    // swallow stage 1 — the per-stage latch must fire for both.
    let mut state = FfState::new();
    state.set_texture_stage_state(0, D3DTSS_BUMPENVMAT00 as usize, 1);
    state.set_texture_stage_state(1, D3DTSS_BUMPENVMAT00 as usize, 1);
    assert!(state.tss_warn_fired(0, D3DTSS_BUMPENVMAT00 as usize));
    assert!(state.tss_warn_fired(1, D3DTSS_BUMPENVMAT00 as usize));
}

#[test]
fn set_transform_world_matrix_index_routes_to_palette() {
    use mtld3d_types::{D3DMATRIX, D3DTS_WORLD};
    let mut state = FfState::new();
    // D3DTS_WORLD is palette[0] — must not bump high-water above 0.
    let m = D3DMATRIX::IDENTITY;
    assert!(state.set_transform(D3DTS_WORLD, &m));
    assert_eq!(state.world_palette_used(), 1);
    // D3DTS_WORLDMATRIX(5) = state 261 → palette[5].
    let mut m5 = D3DMATRIX::IDENTITY;
    m5.m[3] = 7.0; // distinguishable value in row 0 col 3
    assert!(state.set_transform(256 + 5, &m5));
    assert_eq!(state.world_palette_used(), 6, "high water 5 → used = 6");
    assert!((state.world_palette()[5].m[3] - 7.0).abs() < f32::EPSILON);
    // Slot 0 unchanged by the slot-5 write.
    assert!(state.world_palette()[0].m[3].abs() < f32::EPSILON);
}

#[test]
fn resolve_vertex_blend_count_normal_mode() {
    let layout_with_weights = FfVsLayout {
        flags: FfVsLayoutFlags::empty(),
        tex_coord_count: 0,
        tex_coord_dims: [0; 8],
        declared_weights_count: 3,
    };
    // D3DVBF_1WEIGHTS → 2 matrices; sequential mode.
    assert_eq!(
        super::resolve_vertex_blend_count(1, layout_with_weights, false),
        2
    );
    // D3DVBF_3WEIGHTS → 4 matrices.
    assert_eq!(
        super::resolve_vertex_blend_count(3, layout_with_weights, false),
        4
    );
    // D3DVBF_DISABLE → 0.
    assert_eq!(
        super::resolve_vertex_blend_count(0, layout_with_weights, false),
        0
    );
    // Tweening unsupported → 0.
    assert_eq!(
        super::resolve_vertex_blend_count(255, layout_with_weights, false),
        0
    );
}

#[test]
fn resolve_vertex_blend_count_indexed_only() {
    let layout_with_indices = FfVsLayout {
        flags: FfVsLayoutFlags::DECLARED_INDICES,
        tex_coord_count: 0,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };
    // D3DVBF_0WEIGHTS + INDEXED → 1 matrix (single-bone indexed).
    assert_eq!(
        super::resolve_vertex_blend_count(256, layout_with_indices, true),
        1
    );
    // D3DVBF_0WEIGHTS without INDEXED → 0 (mode requires indices).
    assert_eq!(
        super::resolve_vertex_blend_count(256, layout_with_indices, false),
        0
    );
}

#[test]
fn set_light_sets_active_and_directional_masks() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT9};
    let mut state = FfState::new();
    assert_eq!(state.light_active_mask(), 0);
    assert_eq!(state.light_directional_mask(), 0);

    let dir = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        ..D3DLIGHT9::default()
    };
    state.set_light(3, &dir);
    state.set_light_enabled(3, true);
    assert_eq!(state.light_active_mask(), 1 << 3);
    assert_eq!(state.light_directional_mask(), 1 << 3);

    let pt = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        ..D3DLIGHT9::default()
    };
    state.set_light(5, &pt);
    state.set_light_enabled(5, true);
    assert_eq!(state.light_active_mask(), (1 << 3) | (1 << 5));
    assert_eq!(
        state.light_directional_mask(),
        1 << 3,
        "POINT must not set dir bit"
    );
}

#[test]
fn set_light_with_type_zero_clears_set_bit() {
    use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9};
    let mut state = FfState::new();
    let pt = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        ..D3DLIGHT9::default()
    };
    state.set_light(2, &pt);
    state.set_light_enabled(2, true);
    assert_eq!(state.light_active_mask(), 1 << 2);

    // SetLight with Type=0 should drop the slot from the set mask, so
    // light_active_mask clears even though LightEnable(2, TRUE) remains.
    // (D3DLIGHT9::default() is DIRECTIONAL — construct Type=0 directly.)
    let zero = D3DLIGHT9 {
        type_: 0,
        ..D3DLIGHT9::default()
    };
    state.set_light(2, &zero);
    assert_eq!(state.light_active_mask(), 0);
}

#[test]
fn light_enable_toggles_active_mask_when_set_bit_present() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT9};
    let mut state = FfState::new();
    let dir = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        ..D3DLIGHT9::default()
    };
    state.set_light(0, &dir);
    // Set but not enabled → not active.
    assert_eq!(state.light_active_mask(), 0);
    state.set_light_enabled(0, true);
    assert_eq!(state.light_active_mask(), 1);
    state.set_light_enabled(0, false);
    assert_eq!(state.light_active_mask(), 0);
    assert_eq!(
        state.light_directional_mask(),
        1,
        "dir bit persists across enable toggles"
    );
}

#[test]
fn set_light_maintains_spot_mask() {
    use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};
    let mut state = FfState::new();
    let spot = D3DLIGHT9 {
        type_: D3DLIGHT_SPOT,
        ..D3DLIGHT9::default()
    };
    state.set_light(1, &spot);
    state.set_light_enabled(1, true);
    assert_eq!(state.light_spot_mask(), 1 << 1);
    assert_eq!(
        state.light_directional_mask(),
        0,
        "SPOT must not set the dir bit"
    );
    assert_eq!(state.light_active_mask(), 1 << 1);

    // Re-typing the slot clears the spot bit.
    let pt = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        ..D3DLIGHT9::default()
    };
    state.set_light(1, &pt);
    assert_eq!(state.light_spot_mask(), 0);
}

#[test]
fn restore_recomputes_derived_masks() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_SPOT, D3DLIGHT9};

    use super::FfStateSnapshot;
    // State-block Apply restores the light/TSS arrays wholesale,
    // bypassing the setters that maintain the masks — the captured
    // light masks must land with the lights array, and tt_active_mask
    // must re-derive from the restored stage states.
    let mut src = FfState::new();
    src.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_SPOT,
            ..D3DLIGHT9::default()
        },
    );
    src.set_light_enabled(0, true);
    src.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
    let snap = FfStateSnapshot::from(&src);

    let mut dst = FfState::new();
    dst.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            ..D3DLIGHT9::default()
        },
    );
    snap.restore_into(&mut dst);
    assert_eq!(dst.light_spot_mask(), 1, "spot bit from restored light");
    assert_eq!(
        dst.light_directional_mask(),
        0,
        "stale dir bit must clear on restore"
    );
    assert_eq!(
        dst.light_active_mask(),
        1,
        "set mask re-derived from restored lights"
    );
    assert_eq!(
        dst.tt_active_mask(),
        1 << 2,
        "tt mask re-derived from restored stage states"
    );
}

#[test]
fn light_defined_tracks_set_and_enable() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT9};
    let mut state = FfState::new();
    assert!(!state.light_defined(0));
    assert!(!state.light_defined(4));

    // LightEnable defines a previously-undefined slot with the D3D9 default
    // directional light (white diffuse), so GetLight can report it.
    state.set_light_enabled(4, true);
    assert!(state.light_defined(4));
    assert_eq!(state.light(4).type_, D3DLIGHT_DIRECTIONAL);
    assert_eq!(state.light(4).diffuse.r.to_bits(), 1.0f32.to_bits());
    // The materialized default light contributes like an explicit
    // SetLight would: an enable-only light lights the scene.
    assert_eq!(state.light_active_mask(), 1 << 4);
    assert_eq!(state.light_directional_mask(), 1 << 4);
    state.set_light_enabled(4, false);
    assert_eq!(state.light_active_mask(), 0);

    // SetLight defines a slot regardless of light type.
    let zero = D3DLIGHT9 {
        type_: 0,
        ..D3DLIGHT9::default()
    };
    state.set_light(0, &zero);
    assert!(state.light_defined(0));
}

#[test]
fn set_tt_flags_toggles_tt_active_mask() {
    let mut state = FfState::new();
    assert_eq!(state.tt_active_mask(), 0);

    state.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
    assert_eq!(state.tt_active_mask(), 1 << 2);

    state.set_texture_stage_state(5, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 0x101);
    assert_eq!(state.tt_active_mask(), (1 << 2) | (1 << 5));

    state.set_texture_stage_state(2, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 0);
    assert_eq!(state.tt_active_mask(), 1 << 5);
}

#[test]
fn ff_state_new_clears_all_masks() {
    let state = FfState::new();
    assert_eq!(state.light_active_mask(), 0);
    assert_eq!(state.light_directional_mask(), 0);
    assert_eq!(state.tt_active_mask(), 0);
}

#[test]
fn set_texture_stage_state_reports_value_change() {
    // The `changed` return gates snapshot dirty-marking: a same-value
    // write must report `false` so the redundant FF-key rebuild is
    // skipped; a real change must report `true`.
    let mut state = FfState::new();
    let ty = D3DTSS_COLOROP as usize;
    let initial = state.texture_stage_state(0, ty);

    assert!(
        !state.set_texture_stage_state(0, ty, initial),
        "re-writing the existing value reports unchanged"
    );
    assert!(
        state.set_texture_stage_state(0, ty, initial + 1),
        "writing a new value reports changed"
    );
    assert!(
        !state.set_texture_stage_state(0, ty, initial + 1),
        "re-writing the now-current value reports unchanged"
    );
}

// ─────────────────────────────────────────────────────────────────────
// FF VS const-row extent tests
//
// `ff_vs_row_count` derives the per-draw upload extent from `FfVsKey`
// gating + `FfState.tt_active_mask` + `world_palette_used`. Row
// indices (fog 8, material 10..14, lights 15..62, TTFF 63..94,
// palette 95+) are load-bearing — these tests pin the cascade.
// ─────────────────────────────────────────────────────────────────────

fn make_vs_key(flags: super::FfVsFlags, fog_mode: u8) -> super::FfVsKey {
    super::FfVsKey {
        flags,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 0,
        ambient_source: 0,
        specular_source: 0,
        emissive_source: 0,
        fog_mode,
        tci_modes: [0; 8],
        tci_coord_indices: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
    }
}

#[test]
fn ff_vs_row_count_xyzrhw_is_one_row() {
    let mut key = make_vs_key(super::FfVsFlags::HAS_RHW, 0);
    // Add some noise to make sure has_rhw short-circuits past it.
    key.light_active_mask = 0xFF;
    key.tt_flags = [0xFF; 8];
    assert_eq!(FfState::new().ff_vs_row_count(&key), 1);
}

#[test]
fn ff_vs_row_count_unlit_no_fog_no_tt() {
    // Unlit reads only WV/Proj (rows 0..7) + diffuse fallback (row 10).
    let key = make_vs_key(super::FfVsFlags::empty(), 0);
    assert_eq!(FfState::new().ff_vs_row_count(&key), 11);
}

#[test]
fn ff_vs_row_count_unlit_only_fog() {
    // Fog at row 8 < diffuse fallback row 10, so row 10 still wins.
    let key = make_vs_key(super::FfVsFlags::empty(), 3);
    assert_eq!(FfState::new().ff_vs_row_count(&key), 11);
}

#[test]
fn ff_vs_row_count_lit_no_lights() {
    let key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    // Lit reads through material.emissive (row 13).
    assert_eq!(FfState::new().ff_vs_row_count(&key), 14);
}

#[test]
fn ff_vs_row_count_lit_specular_no_lights() {
    let flags = super::FfVsFlags::LIGHTING_ENABLED | super::FfVsFlags::SPECULAR_ENABLE;
    let key = make_vs_key(flags, 0);
    // Material power lives at row 14.
    assert_eq!(FfState::new().ff_vs_row_count(&key), 15);
}

#[test]
fn ff_vs_row_count_one_light() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    key.light_active_mask = 1;
    key.light_directional_mask = 1;
    // Light 0 tail = row 15 + 0*6 + 5 = 20.
    assert_eq!(FfState::new().ff_vs_row_count(&key), 21);
}

#[test]
fn ff_vs_row_count_one_light_with_fog() {
    // Fog folds into the lit block, adding no extra rows beyond what
    // light 0 already forced.
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 3);
    key.light_active_mask = 1;
    key.light_directional_mask = 1;
    assert_eq!(FfState::new().ff_vs_row_count(&key), 21);
}

#[test]
fn ff_vs_row_count_light_7() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    key.light_active_mask = 1 << 7;
    // Light 7 tail = 15 + 7*6 + 5 = 62.
    assert_eq!(FfState::new().ff_vs_row_count(&key), 63);
}

#[test]
fn ff_vs_row_count_tt_stage_4() {
    let mut state = FfState::new();
    state.set_texture_stage_state(4, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
    let key = make_vs_key(super::FfVsFlags::empty(), 0);
    // Stage 4 tail = 63 + 4*4 + 3 = 82.
    assert_eq!(state.ff_vs_row_count(&key), 83);
}

#[test]
fn ff_vs_row_count_full_no_blend() {
    let mut state = FfState::new();
    for s in 0..8usize {
        state.set_texture_stage_state(s, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
    }
    let mut key = make_vs_key(
        super::FfVsFlags::LIGHTING_ENABLED | super::FfVsFlags::SPECULAR_ENABLE,
        3,
    );
    key.light_active_mask = 0xFF;
    key.light_directional_mask = 0xFF;
    // Worst case without blend: TTFF stage 7 tail = 63+7*4+3 = 94.
    assert_eq!(state.ff_vs_row_count(&key), 95);
}

#[test]
fn ff_vs_row_count_blend_extends_palette() {
    use mtld3d_types::{D3DMATRIX, D3DTS_WORLD};
    let mut state = FfState::new();
    // Touch palette[4] via D3DTS_WORLDMATRIX(4) = 260 to push the
    // high-water mark to 4 → world_palette_used() = 5.
    state.set_transform(D3DTS_WORLD, &D3DMATRIX::IDENTITY);
    state.set_transform(256 + 4, &D3DMATRIX::IDENTITY);
    let mut key = make_vs_key(super::FfVsFlags::empty(), 0);
    key.vertex_blend_count = 2;
    // 95 + 5*4 = 115 rows.
    assert_eq!(state.ff_vs_row_count(&key), 115);
}

// ─────────────────────────────────────────────────────────────────────
// Drift guard: regex over the emitted MSL must agree with our
// inline `max_const_row` derivation. Catches any future edit that
// reorders rows in the emitter without updating the derivation.
// ─────────────────────────────────────────────────────────────────────

fn emitter_high_water(key: &super::FfVsKey) -> u16 {
    use crate::dxso::emit_vs_ff;
    let msl = emit_vs_ff(key);
    let mut max: u16 = 0;
    let mut scanner = msl.as_str();
    while let Some(pos) = scanner.find("vs_c[") {
        let rest = &scanner[pos + "vs_c[".len()..];
        let end = rest.find(']').expect("vs_c[ without closing ]");
        let n: u16 = rest[..end].parse().expect("vs_c index must be u16");
        if n > max {
            max = n;
        }
        scanner = &rest[end + 1..];
    }
    max
}

fn derive_max_const_row(state: &FfState, key: &super::FfVsKey) -> u16 {
    state.ff_vs_row_count(key) - 1
}

#[test]
fn max_const_row_matches_emitter_high_water_unlit() {
    let key = make_vs_key(super::FfVsFlags::empty(), 0);
    let state = FfState::new();
    // Emitter reads diffuse fallback row 10. Our derive matches.
    let emit = emitter_high_water(&key);
    let derive = derive_max_const_row(&state, &key);
    assert!(
        emit <= derive,
        "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
    );
}

#[test]
fn max_const_row_matches_emitter_high_water_lit_no_lights() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
    let state = FfState::new();
    let emit = emitter_high_water(&key);
    let derive = derive_max_const_row(&state, &key);
    assert!(
        emit <= derive,
        "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
    );
}

#[test]
fn max_const_row_matches_emitter_high_water_lit_light0() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
    key.light_active_mask = 1;
    key.light_directional_mask = 1;
    let state = FfState::new();
    let emit = emitter_high_water(&key);
    let derive = derive_max_const_row(&state, &key);
    assert!(
        emit <= derive,
        "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
    );
}

#[test]
fn max_const_row_matches_emitter_high_water_lit_light7() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 0);
    key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
    key.light_active_mask = 1 << 7;
    let state = FfState::new();
    let emit = emitter_high_water(&key);
    let derive = derive_max_const_row(&state, &key);
    assert!(
        emit <= derive,
        "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
    );
}

#[test]
fn max_const_row_matches_emitter_high_water_lit_fog_tt_stage_4() {
    let mut key = make_vs_key(super::FfVsFlags::LIGHTING_ENABLED, 3);
    key.flags.set(super::FfVsFlags::HAS_NORMAL, true);
    key.light_active_mask = 1;
    key.light_directional_mask = 1;
    // Emitter reads tt_flags[s] to gate the TTFF rows; mirror in key.
    key.tt_flags[4] = 2;
    key.tex_coord_count = 5;
    let mut state = FfState::new();
    state.set_texture_stage_state(4, D3DTSS_TEXTURETRANSFORMFLAGS as usize, 2);
    let emit = emitter_high_water(&key);
    let derive = derive_max_const_row(&state, &key);
    assert!(
        emit <= derive,
        "emitter reads vs_c[{emit}] but we only wrote rows 0..={derive}"
    );
}

#[test]
fn resolve_vertex_blend_count_decl_mismatch_falls_back() {
    let layout_no_blend = FfVsLayout {
        flags: FfVsLayoutFlags::empty(),
        tex_coord_count: 0,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };
    // Game asks for blending but decl has no BLENDWEIGHT → 0.
    assert_eq!(
        super::resolve_vertex_blend_count(1, layout_no_blend, false),
        0
    );
    // Game enables INDEXED but decl has no BLENDINDICES → 0.
    let layout_weights_only = FfVsLayout {
        declared_weights_count: 2,
        ..layout_no_blend
    };
    assert_eq!(
        super::resolve_vertex_blend_count(2, layout_weights_only, true),
        0
    );
}

// ─────────────────────────────────────────────────────────────────────
// Sparse-light compaction + eye-space packing.
// ─────────────────────────────────────────────────────────────────────

/// Read `rows` packed `[f32; 4]` rows back out of a section pointer.
///
/// Decodes the raw bytes, sidestepping any pointer-alignment cast.
///
/// # Safety
///
/// `ptr` must point at `rows` consecutive `[f32; 4]` values (16 bytes
/// each), as returned by a `build_*_section` helper.
unsafe fn read_section_rows(ptr: *mut u8, rows: usize) -> Vec<[f32; 4]> {
    let byte_len = rows * 16;
    // SAFETY: caller guarantees `rows * 16` initialized bytes at `ptr`.
    let bytes = unsafe { core::slice::from_raw_parts(ptr.cast_const(), byte_len) };
    bytes
        .chunks_exact(16)
        .map(|row| {
            let lane = |k: usize| {
                f32::from_le_bytes(
                    row[k * 4..k * 4 + 4]
                        .try_into()
                        .expect("4-byte f32 lane slice"),
                )
            };
            [lane(0), lane(1), lane(2), lane(3)]
        })
        .collect()
}

/// Assert two `[f32; 4]` rows match within a tight tolerance.
///
/// The section data is exact, but float-array `assert_eq!` trips clippy's
/// `float_cmp`.
fn assert_row_eq(got: [f32; 4], want: [f32; 4], what: &str) {
    for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-6, "{what} lane {k}: got {g}, want {w}");
    }
}

/// Build the [`super::FfVsKey`] a lit, normal-carrying draw would produce.
///
/// Reads the real `D3DRS_LIGHTING` default-on state so `build_vs_key`
/// derives the compacted light masks.
fn lit_vs_key(state: &FfState) -> super::FfVsKey {
    use mtld3d_types::{D3DRS_LIGHTING, RENDER_STATE_COUNT};
    let mut rs = [0u32; RENDER_STATE_COUNT];
    rs[D3DRS_LIGHTING as usize] = 1;
    let layout = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_NORMAL,
        tex_coord_count: 0,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };
    state.build_vs_key(&rs, layout, 0)
}

#[test]
fn sparse_light_index_compacts_to_slot_zero() {
    use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9};
    // Sparse light addressing: a single light at index 123.
    let mut state = FfState::new();
    let light = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        ..D3DLIGHT9::default()
    };
    state.set_light_at(123, &light);
    state.set_light_enabled_at(123, true);

    // The physical fast-path mask is empty — the light lives in overflow.
    assert_eq!(
        state.light_active_mask(),
        0,
        "overflow light not in fast mask"
    );

    let active = state.resolve_active_lights();
    assert_eq!(active.len, 1, "one enabled overflow light compacts");
    assert_eq!(active.as_slice()[0].ty, D3DLIGHT_POINT);

    // And the derived key mask is the single low bit.
    let key = lit_vs_key(&state);
    assert_eq!(
        key.light_active_mask, 0b1,
        "compacted active mask must be slot 0 set"
    );
}

#[test]
fn build_vs_key_compacts_sparse_lights_in_index_order() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};
    // Lights at fast-path 5 (POINT) and overflow 100 (SPOT), 200 (DIR).
    let mut state = FfState::new();
    state.set_light(
        5,
        &D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            ..D3DLIGHT9::default()
        },
    );
    state.set_light_enabled(5, true);
    state.set_light_at(
        100,
        &D3DLIGHT9 {
            type_: D3DLIGHT_SPOT,
            ..D3DLIGHT9::default()
        },
    );
    state.set_light_enabled_at(100, true);
    state.set_light_at(
        200,
        &D3DLIGHT9 {
            type_: D3DLIGHT_DIRECTIONAL,
            ..D3DLIGHT9::default()
        },
    );
    state.set_light_enabled_at(200, true);

    let key = lit_vs_key(&state);
    // Three compacted slots → low three bits.
    assert_eq!(key.light_active_mask, 0b111);
    // Slot 0 = index 5 = POINT (neither type bit), slot 1 = index 100 =
    // SPOT, slot 2 = index 200 = DIRECTIONAL.
    assert_eq!(key.light_spot_mask, 0b010, "spot at compacted slot 1");
    assert_eq!(
        key.light_directional_mask, 0b100,
        "directional at compacted slot 2"
    );
}

#[test]
fn eye_space_point_light_position_matches_hand_calc() {
    use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT9, D3DMATRIX, D3DTS_VIEW, D3DVECTOR};

    use crate::scratch::ScratchArena;
    // VIEW = pure translation {0.5, 0.5, 0}.
    let view = D3DMATRIX {
        m: [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.5, 0.5, 0.0, 1.0, //
        ],
    };
    let mut state = FfState::new();
    state.set_transform(D3DTS_VIEW, &view);
    state.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            position: D3DVECTOR {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            ..D3DLIGHT9::default()
        },
    );
    state.set_light_enabled(0, true);

    let key = lit_vs_key(&state);
    let mut scratch = ScratchArena::new();
    let (start, rows, ptr) = state
        .build_lights_section(&key, &mut scratch)
        .expect("one active light → section present");
    assert_eq!(start, 15);
    assert_eq!(rows, 6, "one light = 6 rows");
    // SAFETY: build_lights_section wrote `rows` [f32;4] rows at `ptr`.
    let data = unsafe { read_section_rows(ptr, rows as usize) };
    // Row 0 = eye-space position + type-w. Hand calc for `v * view`:
    //   x' = 1*1 + 2*0 + 3*0 + 0.5 = 1.5
    //   y' = 1*0 + 2*1 + 3*0 + 0.5 = 2.5
    //   z' = 1*0 + 2*0 + 3*1 + 0   = 3.0
    // type-w = 1.0 (POINT).
    assert_row_eq(data[0], [1.5, 2.5, 3.0, 1.0], "eye-space POINT position");
}

#[test]
fn contiguous_index0_light_packing_is_byte_identical() {
    // Guards the common WoW / e2e path: a single light at index 0 with
    // identity VIEW (eye == world) must pack to the exact rows hard-coded
    // below, so a future refactor of the compaction / eye-space path
    // can't silently drift them.
    use mtld3d_types::{D3DCOLORVALUE, D3DLIGHT_POINT, D3DLIGHT9, D3DVECTOR};

    use crate::scratch::ScratchArena;
    let mut state = FfState::new();
    state.set_light(
        0,
        &D3DLIGHT9 {
            type_: D3DLIGHT_POINT,
            diffuse: D3DCOLORVALUE {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 1.0,
            },
            position: D3DVECTOR {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            },
            attenuation0: 1.0,
            attenuation1: 0.1,
            attenuation2: 0.01,
            range: 100.0,
            ..D3DLIGHT9::default()
        },
    );
    state.set_light_enabled(0, true);

    let key = lit_vs_key(&state);
    // Identity VIEW ⇒ compacted slot 0 == physical slot 0, eye == world.
    assert_eq!(key.light_active_mask, 0b1);
    let mut scratch = ScratchArena::new();
    let (start, rows, ptr) = state.build_lights_section(&key, &mut scratch).unwrap();
    assert_eq!((start, rows), (15, 6));
    // SAFETY: 6 [f32;4] rows written at ptr.
    let data = unsafe { read_section_rows(ptr, 6) };
    // Position row: world == eye under identity view; POINT type-w = 1.
    assert_row_eq(data[0], [4.0, 5.0, 6.0, 1.0], "position row");
    // Diffuse color row (row base+2).
    assert_row_eq(data[2], [0.25, 0.5, 0.75, 1.0], "diffuse row");
    // Attenuation row (row base+4): a0, a1, a2, range.
    assert_row_eq(data[4], [1.0, 0.1, 0.01, 100.0], "attenuation row");
}

#[test]
fn view_change_marks_lights_dirty() {
    use mtld3d_types::{D3DMATRIX, D3DTS_VIEW};
    let mut state = FfState::new();
    // Clear the cold-start all-dirty so we observe the SetTransform mark.
    let _ = state.take_ff_vs_dirty();
    state.set_transform(D3DTS_VIEW, &D3DMATRIX::IDENTITY);
    let dirty = state.take_ff_vs_dirty();
    assert!(
        dirty.contains(super::FfVsDirty::LIGHTS),
        "a VIEW change must invalidate the eye-space LIGHTS section"
    );
}

mod inverse {
    use mtld3d_types::D3DMATRIX;

    use crate::ff_state::FfState;

    fn assert_close(a: &D3DMATRIX, b: &D3DMATRIX) {
        for (x, y) in a.m.iter().zip(b.m) {
            assert!((x - y).abs() < 1e-5, "{:?} != {:?}", a.m, b.m);
        }
    }

    #[test]
    fn inverse_undoes_a_translation_and_a_scaled_rotation() {
        let mut t = D3DMATRIX::IDENTITY;
        t.m[12] = 3.0;
        t.m[13] = -2.0;
        t.m[14] = 7.5;
        let inv = FfState::inverse(&t).expect("translation is invertible");
        assert_close(&FfState::mat_mul(&t, &inv), &D3DMATRIX::IDENTITY);
        // 90-degree rotation about Z scaled by 2, translated.
        let r = D3DMATRIX {
            m: [
                0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0,
            ],
        };
        let inv = FfState::inverse(&r).expect("rigid transform is invertible");
        assert_close(&FfState::mat_mul(&r, &inv), &D3DMATRIX::IDENTITY);
        assert_close(&FfState::mat_mul(&inv, &r), &D3DMATRIX::IDENTITY);
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let mut z = D3DMATRIX::IDENTITY;
        z.m[5] = 0.0;
        assert!(FfState::inverse(&z).is_none());
        assert!(FfState::inverse(&D3DMATRIX { m: [0.0; 16] }).is_none());
    }
}
