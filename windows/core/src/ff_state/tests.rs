//! Unit tests for the Fixed-Function pipeline state stored on `DeviceInner`.
//!
//! Pins the light and texture-transform masks the setters maintain (and that a state-block
//! restore must rebuild), the FF shader keys derived from render state (fog source, local
//! viewer, texcoord routing past a disabled color op, vertex blend), sparse lights compacting
//! into dense eye-space shader slots, the const-row extent checked against the `vs_c` rows the
//! emitter reads, `inverse` round-tripping affine matrices while rejecting singular ones, and
//! the texture-stage-state warn latch firing per stage for unconsumed slots and never for the
//! bump-environment slots, which route to the texbem uniform alone, and the unimplemented
//! texture-operation warning firing at the write, once per slot, while a value outside the
//! `D3DTOP_*` space reads as the stage default instead, the draw-time narrowing of stage
//! arguments and result registers reading the stage default with one warning per stage and
//! state, and the FF VS source fingerprint moving on every palette growth, light enable and
//! light type change while holding across matrix, material and light-parameter writes that
//! leave the FF VS key and row count alone.

use std::sync::Mutex;

use mtld3d_types::{
    D3DFOG_EXP, D3DFOG_LINEAR, D3DMATRIX, D3DRS_DEPTHBIAS, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_TEXTUREFACTOR, D3DTA_TEXTURE, D3DTOP_BUMPENVMAP, D3DTOP_LERP, D3DTOP_MODULATE,
    D3DTOP_MULTIPLYADD, D3DTSS_ALPHAOP, D3DTSS_BUMPENVLOFFSET, D3DTSS_BUMPENVLSCALE,
    D3DTSS_BUMPENVMAT00, D3DTSS_BUMPENVMAT01, D3DTSS_BUMPENVMAT10, D3DTSS_BUMPENVMAT11,
    D3DTSS_COLORARG0, D3DTSS_COLOROP, D3DTSS_CONSTANT, D3DTSS_TEXCOORDINDEX,
    D3DTSS_TEXTURETRANSFORMFLAGS, RENDER_STATE_COUNT, render_state_defaults,
};

use super::{
    FfState, FfVsLayout, TssWriteFeeds, VariantFlags, VariantKey, build_fog_color_bytes,
    stage_enum_value, tss_write_feeds,
};
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
    states[D3DRS_FOGVERTEXMODE as usize] = D3DFOG_EXP;
    states[D3DRS_FOGTABLEMODE as usize] = D3DFOG_LINEAR;

    // Identity projection (4th column (0,0,0,1)) = orthographic → Z source.
    let mut ff = FfState::new();
    let variant = ff.variant_key(&states, false, true);
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
    let variant = ff.variant_key(&states, false, true);
    assert!(
        variant.flags.contains(VariantFlags::FOG_SOURCE_W),
        "non-ortho projection → W source"
    );

    // Table fog applies on the RHW path too.
    let variant = ff.variant_key(&states, true, true);
    assert_eq!(variant.fog_table_mode, 3);
    assert_eq!(variant.fog_mode, 0);

    // Vertex fog only: no table mode, no source bit churn from the
    // (still perspective) projection.
    states[D3DRS_FOGTABLEMODE as usize] = 0;
    let variant = ff.variant_key(&states, false, true);
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
        D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATERIAL9, D3DMATRIX, D3DTOP_DISABLE, D3DTOP_MODULATE,
        D3DTS_VIEW, D3DTSS_TEXCOORDINDEX, StateBlockType,
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

/// `build_vs_key` carries the whole texgen byte of `D3DTSS_TEXCOORDINDEX` into the key.
///
/// The sphere-map mode is bit 18 of the state, above the two bits the three
/// camera-space modes use, so a mask narrower than the byte would turn it
/// into passthru.
#[test]
fn spheremap_texgen_mode_reaches_the_vs_key() {
    let mut ff = FfState::new();
    ff.set_texture_stage_state(0, D3DTSS_TEXCOORDINDEX as usize, 0x0004_0001);
    let layout = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_NORMAL,
        tex_coord_count: 1,
        tex_coord_dims: [0; 8],
        declared_weights_count: 0,
    };
    let key = ff.build_vs_key(&rs(), layout, 0b0000_0001);
    assert_eq!(key.tci_modes[0], 4);
    assert_eq!(key.tci_coord_indices[0], 1);
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
    // COLORARG0 is NotImplemented (not in the Consumed list), so a
    // non-default write fires warn_tss_non_default_once. Default for
    // COLORARG0 is D3DTA_CURRENT; write D3DTA_TEXTURE to stages 0 and 1.
    // A latch keyed only on `ty` would set tss_warn_fired[ty] on stage 0
    // and silently swallow stage 1, so the per-stage latch must fire for
    // both.
    let mut state = FfState::new();
    state.set_texture_stage_state(0, D3DTSS_COLORARG0 as usize, D3DTA_TEXTURE);
    state.set_texture_stage_state(1, D3DTSS_COLORARG0 as usize, D3DTA_TEXTURE);
    assert!(state.tss_warn_fired(0, D3DTSS_COLORARG0 as usize));
    assert!(state.tss_warn_fired(1, D3DTSS_COLORARG0 as usize));
}

#[test]
fn bump_env_tss_writes_do_not_warn() {
    // The bump-environment matrix and luminance slots feed the texbem PS
    // uniform, so a non-default write is consumed and must not fire the
    // "written but not consumed" latch on any stage.
    let slots = [
        D3DTSS_BUMPENVMAT00,
        D3DTSS_BUMPENVMAT01,
        D3DTSS_BUMPENVMAT10,
        D3DTSS_BUMPENVMAT11,
        D3DTSS_BUMPENVLSCALE,
        D3DTSS_BUMPENVLOFFSET,
    ];
    let mut state = FfState::new();
    for stage in 0..8 {
        for ty in slots {
            state.set_texture_stage_state(stage, ty as usize, 1.0f32.to_bits());
            assert!(
                !state.tss_warn_fired(stage, ty as usize),
                "D3DTSS_{ty} (stage {stage}) fired the not-consumed warn"
            );
        }
    }
}

#[test]
fn unimplemented_texture_op_write_warns_once_per_slot() {
    // The warning fires at the write, so a warm shader cache that never runs
    // the emitter still reports the operation. An implemented operation never
    // warns, and neither does the not-consumed latch, since both op slots are
    // consumed.
    let color = D3DTSS_COLOROP as usize;
    let alpha = D3DTSS_ALPHAOP as usize;
    let mut state = FfState::new();
    state.set_texture_stage_state(0, color, D3DTOP_MODULATE);
    assert!(!state.texture_op_warn_fired(color, D3DTOP_MODULATE));
    state.set_texture_stage_state(1, color, D3DTOP_BUMPENVMAP);
    assert!(state.texture_op_warn_fired(color, D3DTOP_BUMPENVMAP));
    assert!(!state.texture_op_warn_fired(alpha, D3DTOP_BUMPENVMAP));
    assert!(!state.texture_op_warn_fired(color, D3DTOP_MULTIPLYADD));
    state.set_texture_stage_state(2, alpha, D3DTOP_MULTIPLYADD);
    assert!(state.texture_op_warn_fired(alpha, D3DTOP_MULTIPLYADD));
    assert!(!state.texture_op_warn_fired(color, D3DTOP_MULTIPLYADD));
    assert!(!state.tss_warn_fired(1, color));
    assert!(!state.tss_warn_fired(2, alpha));
}

#[test]
fn texture_op_outside_the_space_reads_the_stage_default() {
    // The emitter renders a value outside the `D3DTOP_*` space as it renders
    // an unimplemented operation, so the key builder reads the stage default
    // instead, and that read carries the warning. The write does not also
    // warn as an unimplemented operation.
    for op in [0, D3DTOP_LERP + 1] {
        let mut state = FfState::new();
        state.set_texture_stage_state(0, D3DTSS_COLOROP as usize, op);
        assert!(!state.texture_op_warn_fired(D3DTSS_COLOROP as usize, op));
        let key = state.build_ps_key(&rs(), 0);
        assert_eq!(u32::from(key.stages[0].color_op), D3DTOP_MODULATE);
    }
}

#[test]
fn bump_env_tss_writes_feed_only_the_bump_uniform() {
    // No FF key, variant or FF constant reads a bump-environment slot, so a
    // write to one routes to the texbem uniform alone. The neighbouring
    // slots keep their routing.
    for ty in [
        D3DTSS_BUMPENVMAT00,
        D3DTSS_BUMPENVMAT01,
        D3DTSS_BUMPENVMAT10,
        D3DTSS_BUMPENVMAT11,
        D3DTSS_BUMPENVLSCALE,
        D3DTSS_BUMPENVLOFFSET,
    ] {
        assert!(
            matches!(tss_write_feeds(ty), TssWriteFeeds::BumpEnv),
            "D3DTSS_{ty} does not route to the bump uniform alone"
        );
    }
    assert!(matches!(
        tss_write_feeds(D3DTSS_CONSTANT),
        TssWriteFeeds::StageConstant
    ));
    for ty in [
        D3DTSS_COLOROP,
        D3DTSS_TEXCOORDINDEX,
        D3DTSS_TEXTURETRANSFORMFLAGS,
        D3DTSS_COLORARG0,
    ] {
        assert!(
            matches!(tss_write_feeds(ty), TssWriteFeeds::FfPipeline),
            "D3DTSS_{ty} does not route to the FF pipeline"
        );
    }
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
        reserved: 0,
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
        .as_chunks::<16>()
        .0
        .iter()
        .map(|row| {
            let (lanes, _) = row.as_chunks::<4>();
            core::array::from_fn(|k| f32::from_le_bytes(lanes[k]))
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

#[test]
fn overflow_light_writes_mark_lights_dirty() {
    use mtld3d_types::{D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9};
    // A light past the eight fast-path slots still packs into the LIGHTS
    // section when it is enabled, so its SetLight and LightEnable must
    // re-upload the section like a fast-path write does.
    assert_overflow_write_marks_lights("SetLight defines an overflow light", |s| {
        s.set_light_at(
            9,
            &D3DLIGHT9 {
                type_: D3DLIGHT_POINT,
                ..D3DLIGHT9::default()
            },
        );
    });
    assert_overflow_write_marks_lights("SetLight rewrites an overflow light", |s| {
        s.set_light_at(
            100,
            &D3DLIGHT9 {
                type_: D3DLIGHT_SPOT,
                range: 3.0,
                ..D3DLIGHT9::default()
            },
        );
    });
    assert_overflow_write_marks_lights("LightEnable turns an overflow light on", |s| {
        s.set_light_enabled_at(9, true);
    });
    assert_overflow_write_marks_lights("LightEnable turns an overflow light off", |s| {
        s.set_light_enabled_at(100, false);
    });
}

/// Run `write` on a state with enabled overflow light 100 and assert it marked LIGHTS.
fn assert_overflow_write_marks_lights(name: &str, write: fn(&mut FfState)) {
    let mut state = FfState::new();
    state.set_light_at(100, &mtld3d_types::D3DLIGHT9::default());
    state.set_light_enabled_at(100, true);
    // Clear the cold-start all-dirty so the write's own mark shows.
    let _ = state.take_ff_vs_dirty();
    write(&mut state);
    assert!(
        state.take_ff_vs_dirty().contains(super::FfVsDirty::LIGHTS),
        "{name}: the LIGHTS section was not marked"
    );
}

#[test]
fn state_block_restores_mark_the_sections_they_write() {
    use mtld3d_types::{D3DMATERIAL9, D3DTS_PROJECTION, StateBlockType};

    use super::{FfStateSnapshot, FfVsDirty};
    // A restore writes the arrays directly, past the setters and their marks,
    // so the encoder mirror only learns of the restored values if the restore
    // marks the sections itself.
    let snapshot = FfStateSnapshot::from(&FfState::new());
    let cases = [
        (StateBlockType::All, FfVsDirty::all()),
        (StateBlockType::Vertex, FfVsDirty::LIGHTS | FfVsDirty::TT),
        (StateBlockType::Pixel, FfVsDirty::TT),
    ];
    for (block_type, expected) in cases {
        let mut state = FfState::new();
        state.set_transform(D3DTS_PROJECTION, &D3DMATRIX::IDENTITY);
        state.set_material(&D3DMATERIAL9::default());
        let _ = state.take_ff_vs_dirty();
        snapshot.restore_filtered(&mut state, block_type);
        assert_eq!(
            state.take_ff_vs_dirty(),
            expected,
            "{block_type:?} restore marked the wrong sections"
        );
    }
    let mut state = FfState::new();
    let _ = state.take_ff_vs_dirty();
    snapshot.restore_into(&mut state);
    assert_eq!(
        state.take_ff_vs_dirty(),
        FfVsDirty::all(),
        "restore_into marked the wrong sections"
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

    /// The cofactor inverse with `det2` and `sum3` supplied, so one body serves both precisions.
    fn cofactor_inverse<T>(
        m: &[T; 16],
        det2: impl Fn(T, T, T, T) -> T,
        sum3: impl Fn([T; 6]) -> T,
    ) -> [T; 16]
    where
        T: Copy
            + core::ops::Add<Output = T>
            + core::ops::Neg<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::Div<Output = T>
            + From<f32>,
    {
        let s = [
            det2(m[0], m[4], m[1], m[5]),
            det2(m[0], m[4], m[2], m[6]),
            det2(m[0], m[4], m[3], m[7]),
            det2(m[1], m[5], m[2], m[6]),
            det2(m[1], m[5], m[3], m[7]),
            det2(m[2], m[6], m[3], m[7]),
        ];
        let c = [
            det2(m[8], m[12], m[9], m[13]),
            det2(m[8], m[12], m[10], m[14]),
            det2(m[8], m[12], m[11], m[15]),
            det2(m[9], m[13], m[10], m[14]),
            det2(m[9], m[13], m[11], m[15]),
            det2(m[10], m[14], m[11], m[15]),
        ];
        let det = sum3([s[0], c[5], -s[1], c[4], s[2], c[3]])
            + sum3([s[3], c[2], -s[4], c[1], s[5], c[0]]);
        let inv = T::from(1.0) / det;
        let row = |v: [T; 6]| sum3(v) * inv;
        [
            row([m[5], c[5], -m[6], c[4], m[7], c[3]]),
            row([-m[1], c[5], m[2], c[4], -m[3], c[3]]),
            row([m[13], s[5], -m[14], s[4], m[15], s[3]]),
            row([-m[9], s[5], m[10], s[4], -m[11], s[3]]),
            row([-m[4], c[5], m[6], c[2], -m[7], c[1]]),
            row([m[0], c[5], -m[2], c[2], m[3], c[1]]),
            row([-m[12], s[5], m[14], s[2], -m[15], s[1]]),
            row([m[8], s[5], -m[10], s[2], m[11], s[1]]),
            row([m[4], c[4], -m[5], c[2], m[7], c[0]]),
            row([-m[0], c[4], m[1], c[2], -m[3], c[0]]),
            row([m[12], s[4], -m[13], s[2], m[15], s[0]]),
            row([-m[8], s[4], m[9], s[2], -m[11], s[0]]),
            row([-m[4], c[3], m[5], c[1], -m[6], c[0]]),
            row([m[0], c[3], -m[1], c[1], m[2], c[0]]),
            row([-m[12], s[3], m[13], s[1], -m[14], s[0]]),
            row([m[8], s[3], -m[9], s[1], m[10], s[0]]),
        ]
    }

    /// A left-handed look-at view matrix, row-vector convention, as a game builds one.
    fn look_at(eye: [f32; 3], at: [f32; 3]) -> D3DMATRIX {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot =
            |a: [f32; 3], b: [f32; 3]| [a[0] * b[0], a[1] * b[1], a[2] * b[2]].iter().sum::<f32>();
        let cross = |a: [f32; 3], b: [f32; 3]| {
            let (x0, x1) = (a[1] * b[2], a[2] * b[1]);
            let (y0, y1) = (a[2] * b[0], a[0] * b[2]);
            let (z0, z1) = (a[0] * b[1], a[1] * b[0]);
            [x0 - x1, y0 - y1, z0 - z1]
        };
        let norm = |a: [f32; 3]| {
            let len = dot(a, a).sqrt();
            [a[0] / len, a[1] / len, a[2] / len]
        };
        let z = norm(sub(at, eye));
        let x = norm(cross([0.0, 1.0, 0.0], z));
        let y = cross(z, x);
        D3DMATRIX {
            m: [
                x[0],
                y[0],
                z[0],
                0.0,
                x[1],
                y[1],
                z[1],
                0.0,
                x[2],
                y[2],
                z[2],
                0.0,
                -dot(x, eye),
                -dot(y, eye),
                -dot(z, eye),
                1.0,
            ],
        }
    }

    #[test]
    fn unfused_inverse_is_as_close_to_the_exact_one_as_the_fused_form_was() {
        let mut checked = 0;
        for ex in [-900.0f32, -35.5, -1.0, 0.25, 3.0, 128.0, 5000.0] {
            for ey in [-12.0f32, 0.5, 40.0, 700.0] {
                for (tx, tz) in [
                    (0.0f32, 1.0f32),
                    (10.0, -3.0),
                    (-250.0, 90.0),
                    (1.0e3, 1.0e3),
                ] {
                    let ez = ex * 0.5;
                    let view = look_at([ex, ey, ez], [ex + tx, ey - 1.0, ez + tz]);
                    for scale in [1.0f32, 0.01, 40.0] {
                        let mut m = view;
                        for v in &mut m.m[..12] {
                            *v *= scale;
                        }
                        let new = FfState::inverse(&m).expect("a view matrix is invertible").m;
                        let fused = cofactor_inverse(
                            &m.m,
                            |a: f32, b, c, d| a.mul_add(d, -(b * c)),
                            |v| v[0].mul_add(v[1], v[2].mul_add(v[3], v[4] * v[5])),
                        );
                        let exact = cofactor_inverse(
                            &m.m.map(f64::from),
                            |a: f64, b, c, d| {
                                let (ad, bc) = (a * d, b * c);
                                ad - bc
                            },
                            |v| {
                                let products = [v[0] * v[1], v[2] * v[3], v[4] * v[5]];
                                products[0] + products[1] + products[2]
                            },
                        );
                        // An entry is a sum of products as large as the largest
                        // entry of its row, so that bounds its rounding error.
                        for (row, at) in (0..16).step_by(4).zip(0..) {
                            let scale = exact[row..row + 4]
                                .iter()
                                .fold(1.0f64, |acc, v| acc.max(v.abs()));
                            for i in row..row + 4 {
                                let (new_error, fused_error) = (
                                    (f64::from(new[i]) - exact[i]).abs() / scale,
                                    (f64::from(fused[i]) - exact[i]).abs() / scale,
                                );
                                assert!(
                                    new_error <= 1e-6 && fused_error <= 1e-6,
                                    "row {at} entry {i}: {new_error:e} unfused, {fused_error:e} fused, for {:?}",
                                    m.m
                                );
                            }
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 7 * 4 * 4 * 3);
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let mut z = D3DMATRIX::IDENTITY;
        z.m[5] = 0.0;
        assert!(FfState::inverse(&z).is_none());
        assert!(FfState::inverse(&D3DMATRIX { m: [0.0; 16] }).is_none());
    }
}

#[test]
fn range_fog_keys_only_computed_vertex_fog() {
    use mtld3d_types::{D3DFOG_EXP2, D3DRS_RANGEFOGENABLE};

    use crate::dxso::FfVsFlags;

    let ff = FfState::new();
    for mode in [0, D3DFOG_EXP, D3DFOG_EXP2, D3DFOG_LINEAR] {
        for (enabled, table, rhw) in [(1, 0, false), (0, 0, false), (1, 3, false), (1, 0, true)] {
            let mut states = rs();
            states[D3DRS_FOGENABLE as usize] = enabled;
            states[D3DRS_FOGVERTEXMODE as usize] = mode;
            states[D3DRS_FOGTABLEMODE as usize] = table;
            let mut layout = FfVsLayout::default();
            layout.flags.set(FfVsLayoutFlags::HAS_RHW, rhw);
            let ordinary = ff.build_vs_key(&states, layout, 0);
            states[D3DRS_RANGEFOGENABLE as usize] = 1;
            let mut range = ff.build_vs_key(&states, layout, 0);
            let active = mode != 0 && enabled != 0 && table == 0 && !rhw;
            assert_eq!(range.flags.contains(FfVsFlags::RANGE_FOG), active);
            range.flags.remove(FfVsFlags::RANGE_FOG);
            assert_eq!(ordinary, range, "range fog changes only its active key bit");
        }
    }
}

#[test]
fn shader_owned_fog_reuses_the_existing_disabled_key_and_source() {
    use crate::{
        dxso::{emit_ps_programmable, parse},
        shader_key::ff_key_hash,
    };

    let shader =
        parse(&[0xffff_0300, 0x0200_0001, 0x800f_0800, 0xa0e4_0000, 0xffff]).expect("constant PS3");
    let mut ff = FfState::new();
    let mut states = rs();
    let disabled = ff.variant_key(&states, false, true);
    let source = emit_ps_programmable(&shader, disabled).expect("disabled shader");
    assert!(!source.contains("fog_data"));
    states[D3DRS_FOGENABLE as usize] = 1;
    for projection_w in [1.0, 2.0] {
        let mut projection = D3DMATRIX::IDENTITY;
        projection.m[15] = projection_w;
        ff.set_transform(mtld3d_types::D3DTS_PROJECTION, &projection);
        for table_mode in 0..=3 {
            states[D3DRS_FOGTABLEMODE as usize] = table_mode;
            for vertex_mode in 0..=3 {
                states[D3DRS_FOGVERTEXMODE as usize] = vertex_mode;
                let canonical = ff.variant_key(&states, false, false);
                assert_eq!(canonical, disabled);
                assert_eq!(ff_key_hash(&canonical), ff_key_hash(&disabled));
                assert_eq!(build_fog_color_bytes(&states, canonical).1, 0);
                assert_eq!(emit_ps_programmable(&shader, canonical).unwrap(), source);
                assert_ne!(ff.variant_key(&states, false, true), canonical);
            }
        }
    }
}

#[test]
fn resultarg_narrows_to_the_typed_destination_and_preserves_raw_state() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG, render_state_defaults};

    use crate::dxso::FfStageResult;
    let mut state = FfState::new();
    let rs = render_state_defaults();
    for value in [D3DTA_TEMP, D3DTA_CURRENT, 0, 0x105, u32::MAX] {
        state.set_texture_stage_state(0, D3DTSS_RESULTARG as usize, value);
        let key = state.build_ps_key(&rs, 1);
        assert_eq!(
            state.texture_stage_states[0][D3DTSS_RESULTARG as usize],
            value
        );
        assert_eq!(
            key.stages[0].result(),
            if value == D3DTA_TEMP {
                FfStageResult::Temp
            } else {
                FfStageResult::Current
            }
        );
        assert!(key.stages[0].has_texture());
    }
    assert_eq!(
        FfState::new().build_ps_key(&rs, 0).stages[0].result(),
        FfStageResult::Current
    );
}

#[test]
fn per_stage_constants_pack_fresh_prefix_without_changing_keys() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTOP_SELECTARG1, D3DTSS_COLORARG1, D3DTSS_CONSTANT};

    use crate::scratch::ScratchArena;

    let mut ff = FfState::new();
    let states = rs();
    for stage in 0..8 {
        ff.set_texture_stage_state(stage, D3DTSS_COLOROP as usize, D3DTOP_SELECTARG1);
        ff.set_texture_stage_state(stage, D3DTSS_COLORARG1 as usize, D3DTA_CONSTANT);
        ff.set_texture_stage_state(stage, D3DTSS_CONSTANT as usize, 0x8040_2010);
    }
    let key = ff.build_ps_key(&states, 0);
    let mut scratch = ScratchArena::new();
    let ptr = ff.build_ps_stage_constants(&states, key.constant_rows(), &mut scratch);
    // SAFETY: the builder initialized nine 16-byte rows retained in scratch.
    let first = unsafe { core::slice::from_raw_parts(ptr, 144) };
    assert_eq!(&first[..16], &ff.build_ps_constants(&states));
    let expected = [64.0_f32 / 255.0, 32.0 / 255.0, 16.0 / 255.0, 128.0 / 255.0];
    for row in first[16..].as_chunks::<16>().0 {
        for (bytes, expected) in row.as_chunks::<4>().0.iter().zip(expected) {
            assert_eq!(*bytes, expected.to_le_bytes());
        }
    }
    ff.set_texture_stage_state(7, D3DTSS_CONSTANT as usize, 0xFF12_3456);
    assert_eq!(
        key,
        ff.build_ps_key(&states, 0),
        "values are not shader identity"
    );
    let next = ff.build_ps_stage_constants(&states, 9, &mut scratch);
    assert_ne!(ptr, next, "queued draws keep immutable constant rows");
    // SAFETY: this second allocation also contains nine initialized rows.
    let second = unsafe { core::slice::from_raw_parts(next, 144) };
    assert_ne!(&first[128..], &second[128..]);
    let short = ff.build_ps_stage_constants(&states, 2, &mut scratch);
    // SAFETY: the requested two initialized rows occupy 32 bytes.
    let short = unsafe { core::slice::from_raw_parts(short, 32) };
    assert_eq!(short, &first[..32]);
}

// ─────────────────────────────────────────────────────────────────────
// World-matrix palette bound
//
// `D3DTS_WORLDMATRIX(i)` accepts i up to 255, but the FF VS constant
// block holds the palette only to `MAX_VERTEX_BLEND_MATRIX_INDEX`, the
// index advertised as `D3DCAPS9::MaxVertexBlendMatrixIndex`. Both the
// row count and the packed section stop there, whatever the title set.
// ─────────────────────────────────────────────────────────────────────

/// Rows of the FF VS constant block a draw can bind.
///
/// The encoder's `ff_vs_constants_mirror` is this many `float4` rows, and its
/// own compile-time assert pins the same number against the advertised index.
const FF_VS_CONST_ROWS: u16 = 256;

fn blend_key() -> super::FfVsKey {
    let mut key = make_vs_key(super::FfVsFlags::empty(), 0);
    key.vertex_blend_count = 2;
    key
}

fn state_with_palette_high_water(index: u32) -> FfState {
    let mut state = FfState::new();
    state.set_transform(mtld3d_types::D3DTS_WORLD, &D3DMATRIX::IDENTITY);
    state.set_transform(256 + index, &D3DMATRIX::IDENTITY);
    state
}

#[test]
fn ff_vs_row_count_stops_at_the_advertised_palette_index() {
    let key = blend_key();
    let cap = u16::try_from(super::MAX_VERTEX_BLEND_MATRIX_INDEX).expect("cap fits u16");
    let full = super::FF_VS_PALETTE_BASE_ROW + (cap + 1) * 4;
    // The last index the block holds is packed whole.
    assert_eq!(
        state_with_palette_high_water(u32::from(cap)).ff_vs_row_count(&key),
        full
    );
    // One past it, and the highest index D3D9 accepts, add no rows.
    for index in [u32::from(cap) + 1, 255] {
        let rows = state_with_palette_high_water(index).ff_vs_row_count(&key);
        assert_eq!(rows, full, "D3DTS_WORLDMATRIX({index}) extended the count");
        assert!(
            rows <= FF_VS_CONST_ROWS,
            "D3DTS_WORLDMATRIX({index}) counts past the FF VS constant block"
        );
    }
}

#[test]
fn build_palette_section_packs_no_matrix_past_the_advertised_index() {
    use crate::scratch::ScratchArena;
    let cap = usize::try_from(super::MAX_VERTEX_BLEND_MATRIX_INDEX).expect("cap fits usize");
    let key = blend_key();
    for index in [cap, cap + 1, 255] {
        let mut state = state_with_palette_high_water(u32::try_from(index).expect("index ≤ 255"));
        // A distinct diagonal in the last matrix the block holds and in the
        // one after it, so the packed bytes name which was taken. The view is
        // identity, so the transpose leaves `m[0]` where it was.
        let mut marked = D3DMATRIX::IDENTITY;
        marked.m[0] = 9.0;
        state.set_transform(256 + u32::try_from(cap).expect("cap fits u32"), &marked);
        let mut past = D3DMATRIX::IDENTITY;
        past.m[0] = 5.0;
        state.set_transform(256 + u32::try_from(cap + 1).expect("cap fits u32"), &past);

        let mut scratch = ScratchArena::new();
        let (start, rows, ptr) = state
            .build_palette_section(&key, &mut scratch)
            .expect("vertex blending is on");
        assert_eq!(start, super::FF_VS_PALETTE_BASE_ROW);
        let expected = u16::try_from((cap + 1) * 4).expect("palette rows fit u16");
        assert_eq!(
            rows, expected,
            "D3DTS_WORLDMATRIX({index}) packed extra rows"
        );
        assert!(
            start + rows <= FF_VS_CONST_ROWS,
            "D3DTS_WORLDMATRIX({index}) packs past the FF VS constant block"
        );
        // SAFETY: the builder initialized `rows` 16-byte rows at `ptr`.
        let packed = unsafe { read_section_rows(ptr, usize::from(rows)) };
        assert!(
            (packed[cap * 4][0] - 9.0).abs() < f32::EPSILON,
            "the last matrix the block holds was not packed"
        );
    }
}

#[test]
fn inline_variable_sections_fill_exact_queried_rows() {
    use core::mem::MaybeUninit;

    fn destination(rows: u16) -> Vec<MaybeUninit<[f32; 4]>> {
        vec![MaybeUninit::new([f32::from_bits(0x7fc1_2345); 4]); usize::from(rows)]
    }
    fn assert_filled(rows: &[MaybeUninit<[f32; 4]>]) {
        for row in rows {
            // SAFETY: every test destination was initialized before the fill.
            let row = unsafe { row.assume_init_ref() };
            assert!(row.iter().all(|value| value.to_bits() != 0x7fc1_2345));
        }
    }
    for count in 0..=8 {
        let mut state = FfState::new();
        for slot in 0..count {
            state.set_light(
                slot,
                &mtld3d_types::D3DLIGHT9 {
                    type_: mtld3d_types::D3DLIGHT_DIRECTIONAL,
                    ..mtld3d_types::D3DLIGHT9::default()
                },
            );
            state.set_light_enabled(slot, true);
        }
        let key = lit_vs_key(&state);
        let rows = FfState::lights_section_rows(&key);
        assert_eq!(usize::from(rows), count * 6);
        let mut dst = destination(rows);
        state.fill_lights_section(&key, &mut dst);
        assert_filled(&dst);
    }
    for high in 0..8 {
        let mut state = FfState::new();
        state.tt_active_mask = 1 << high;
        let rows = state.tt_section_rows();
        assert_eq!(usize::from(rows), (high + 1) * 4);
        let mut dst = destination(rows);
        state.fill_tt_section(&mut dst);
        assert_filled(&dst);
    }
    for index in [0, 1, 7, 39] {
        let state = state_with_palette_high_water(index);
        let key = blend_key();
        let rows = state.palette_section_rows(&key);
        assert_eq!(u32::from(rows), (index + 1) * 4);
        let mut dst = destination(rows);
        state.fill_palette_section(&key, &mut dst);
        assert_filled(&dst);
    }
}

/// The warnings the `log` facade delivered in this test process.
///
/// A `static` because the resource is process-wide: the `log` crate takes one
/// logger per process, set once, and the warn-once latches under test write
/// only through it.
static WARNINGS: WarningSink = WarningSink(Mutex::new(Vec::new()));

/// A `log` sink that keeps every warning's text.
struct WarningSink(Mutex<Vec<String>>);

impl log::Log for WarningSink {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            self.0
                .lock()
                .expect("warning sink poisoned")
                .push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

/// Route this process's warnings into [`WARNINGS`]; later calls keep the first logger.
fn capture_warnings() {
    let _ = log::set_logger(&WARNINGS);
    log::set_max_level(log::LevelFilter::Warn);
}

/// How many captured warnings contain `needle`.
fn warnings_containing(needle: &str) -> usize {
    WARNINGS
        .0
        .lock()
        .expect("warning sink poisoned")
        .iter()
        .filter(|line| line.contains(needle))
        .count()
}

/// A colour argument wider than a byte reads the stage default, not its low byte, and warns once.
#[test]
fn a_colour_argument_wider_than_a_byte_reads_the_stage_default_and_warns_once() {
    use mtld3d_types::{D3DTSS_COLORARG2, texture_stage_state_defaults};
    capture_warnings();
    let mut states = texture_stage_state_defaults(6);
    // The low byte is D3DTA_TEXTURE, which a truncation would read.
    states[D3DTSS_COLORARG2 as usize] = 0x0102;
    let default = texture_stage_state_defaults(6)[D3DTSS_COLORARG2 as usize].to_le_bytes()[0];
    for _ in 0..3 {
        assert_eq!(stage_enum_value(&states, 6, D3DTSS_COLORARG2), default);
    }
    assert_eq!(
        warnings_containing("FF: stage 6 D3DTSS_3 = 0x102 outside its value space"),
        1
    );
}

/// A result register other than CURRENT or TEMP reads the stage default, warning once per stage.
#[test]
fn a_result_register_outside_current_and_temp_reads_the_stage_default() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TEMP, D3DTSS_RESULTARG, texture_stage_state_defaults};
    capture_warnings();
    let mut states = texture_stage_state_defaults(5);
    let current = u8::try_from(D3DTA_CURRENT).expect("D3DTA_CURRENT fits a byte");
    states[D3DTSS_RESULTARG as usize] = D3DTA_TEMP;
    assert_eq!(
        u32::from(stage_enum_value(&states, 5, D3DTSS_RESULTARG)),
        D3DTA_TEMP
    );
    assert_eq!(warnings_containing("FF: stage 5 D3DTSS_28 ="), 0);
    states[D3DTSS_RESULTARG as usize] = D3DTA_TEXTURE;
    assert_eq!(stage_enum_value(&states, 5, D3DTSS_RESULTARG), current);
    // TEMP in the low byte of a wider value is not TEMP.
    states[D3DTSS_RESULTARG as usize] = 0x100 | D3DTA_TEMP;
    assert_eq!(stage_enum_value(&states, 5, D3DTSS_RESULTARG), current);
    // The latch is per stage and state, so the second value stays quiet.
    assert_eq!(warnings_containing("FF: stage 5 D3DTSS_28 ="), 1);
    assert_eq!(
        warnings_containing("FF: stage 5 D3DTSS_28 = 0x2 outside its value space"),
        1
    );
}

// ── Which FF writes can move the FF VS source ──
//
// The SetTransform and MultiplyTransform thunks compare
// `FfState::vs_source_transform_inputs` across the write, SetLight and
// LightEnable compare `FfState::vs_source_light_inputs`, and SetMaterial
// never marks `VS_SOURCE`. These tests hold each comparison to the key and
// row count it stands in for.

/// The fingerprint a transform thunk compares, widened for the tests.
fn transform_inputs(state: &FfState) -> u64 {
    u64::from(state.vs_source_transform_inputs())
}

/// The fingerprint a light thunk compares, widened for the tests.
fn light_inputs(state: &FfState) -> u64 {
    u64::from(state.vs_source_light_inputs())
}

/// `SetMaterial` compares nothing: it never marks `VS_SOURCE`.
const fn material_inputs(_: &FfState) -> u64 {
    0
}

/// The FF VS sources a draw could build from `state`: key and row count.
///
/// One unlit, one lit with specular, and one lit with indexed three-weight
/// blending over a declaration with weights and indices, so every
/// setter-owned input (the light masks, the palette extent) is read by at
/// least one of them.
fn vs_sources(state: &FfState) -> Vec<(super::FfVsKey, u16)> {
    use mtld3d_types::{
        D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_SPECULARENABLE, D3DRS_VERTEXBLEND,
        D3DVBF_3WEIGHTS,
    };
    let plain = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_NORMAL,
        tex_coord_count: 1,
        tex_coord_dims: [2, 0, 0, 0, 0, 0, 0, 0],
        declared_weights_count: 0,
    };
    let blended = FfVsLayout {
        flags: FfVsLayoutFlags::HAS_NORMAL | FfVsLayoutFlags::DECLARED_INDICES,
        declared_weights_count: 3,
        ..plain
    };
    let unlit = rs();
    let mut lit = rs();
    lit[D3DRS_LIGHTING as usize] = 1;
    lit[D3DRS_SPECULARENABLE as usize] = 1;
    let mut blend = lit;
    blend[D3DRS_VERTEXBLEND as usize] = D3DVBF_3WEIGHTS;
    blend[D3DRS_INDEXEDVERTEXBLENDENABLE as usize] = 1;
    [(unlit, plain), (lit, plain), (blend, blended)]
        .iter()
        .map(|(states, layout)| {
            let key = state.build_vs_key(states, *layout, 0b1);
            let rows = state.ff_vs_row_count(&key);
            (key, rows)
        })
        .collect()
}

/// A matrix whose every element depends on `seed`, including the 4th column.
fn seeded_matrix(seed: f32) -> D3DMATRIX {
    let mut m = D3DMATRIX::IDENTITY;
    for (i, value) in m.m.iter_mut().enumerate() {
        let i = f32::from(u8::try_from(i).expect("16 elements fit u8"));
        *value = (seed * (i + 1.0)).mul_add(0.125, *value);
    }
    m
}

/// A light of type `ty` whose every parameter depends on `seed`.
fn seeded_light(ty: u32, seed: f32) -> mtld3d_types::D3DLIGHT9 {
    use mtld3d_types::{D3DCOLORVALUE, D3DLIGHT9, D3DVECTOR};
    let color = D3DCOLORVALUE {
        r: seed,
        g: seed * 0.5,
        b: seed * 0.25,
        a: seed * 0.125,
    };
    let vector = D3DVECTOR {
        x: seed,
        y: -seed,
        z: seed + 1.0,
    };
    D3DLIGHT9 {
        type_: ty,
        diffuse: color,
        specular: color,
        ambient: color,
        position: vector,
        direction: vector,
        range: seed * 10.0,
        falloff: seed,
        attenuation0: seed,
        attenuation1: seed * 0.5,
        attenuation2: seed * 0.25,
        theta: seed * 0.1,
        phi: seed * 0.2,
    }
}

/// A material whose every parameter depends on `seed`.
fn seeded_material(seed: f32) -> mtld3d_types::D3DMATERIAL9 {
    use mtld3d_types::{D3DCOLORVALUE, D3DMATERIAL9};
    let color = D3DCOLORVALUE {
        r: seed,
        g: seed * 0.5,
        b: seed * 0.25,
        a: seed * 0.125,
    };
    D3DMATERIAL9 {
        diffuse: color,
        ambient: color,
        specular: color,
        emissive: color,
        power: seed * 8.0,
    }
}

/// Lights 0 (POINT) and 1 (SPOT) set and enabled; slot 2 set DIRECTIONAL but disabled.
fn lit_state() -> FfState {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT};
    let mut state = FfState::new();
    state.set_light_at(0, &seeded_light(D3DLIGHT_POINT, 1.0));
    state.set_light_enabled_at(0, true);
    state.set_light_at(1, &seeded_light(D3DLIGHT_SPOT, 1.0));
    state.set_light_enabled_at(1, true);
    state.set_light_at(2, &seeded_light(D3DLIGHT_DIRECTIONAL, 1.0));
    state
}

/// [`lit_state`] plus an enabled SPOT light at overflow index 100.
fn lit_state_with_overflow() -> FfState {
    let mut state = lit_state();
    state.set_light_at(100, &seeded_light(mtld3d_types::D3DLIGHT_SPOT, 1.0));
    state.set_light_enabled_at(100, true);
    state
}

/// One FF write and the fingerprint its thunk compares across it.
struct FfWrite {
    name: &'static str,
    write: fn(&mut FfState),
    fingerprint: fn(&FfState) -> u64,
}

#[test]
fn vs_source_inputs_move_on_every_key_input_the_ff_setters_write() {
    use mtld3d_types::{D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DTS_WORLD};
    let cases: [(fn() -> FfState, FfWrite); 17] = [
        (
            FfState::new,
            FfWrite {
                name: "SetTransform grows the palette",
                write: |s| {
                    s.set_transform(D3DTS_WORLD + 3, &seeded_matrix(1.0));
                },
                fingerprint: transform_inputs,
            },
        ),
        (
            FfState::new,
            FfWrite {
                name: "MultiplyTransform grows the palette",
                write: |s| {
                    s.multiply_transform(D3DTS_WORLD + 5, &seeded_matrix(1.0));
                },
                fingerprint: transform_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "LightEnable turns a set light on",
                write: |s| s.set_light_enabled_at(2, true),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "LightEnable turns a light off",
                write: |s| s.set_light_enabled_at(0, false),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "LightEnable materializes a light on",
                write: |s| s.set_light_enabled_at(5, true),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "SetLight changes POINT to SPOT",
                write: |s| s.set_light_at(0, &seeded_light(D3DLIGHT_SPOT, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "SetLight changes SPOT to DIRECTIONAL",
                write: |s| s.set_light_at(1, &seeded_light(D3DLIGHT_DIRECTIONAL, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "SetLight changes POINT to DIRECTIONAL",
                write: |s| s.set_light_at(0, &seeded_light(D3DLIGHT_DIRECTIONAL, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "SetLight changes SPOT to POINT",
                write: |s| s.set_light_at(1, &seeded_light(D3DLIGHT_POINT, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "SetLight changes an enabled light to type 0",
                write: |s| s.set_light_at(0, &seeded_light(0, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            || {
                let mut state = lit_state();
                state.set_light_at(0, &seeded_light(0, 1.0));
                state
            },
            FfWrite {
                name: "SetLight gives an enabled type-0 light a type",
                write: |s| s.set_light_at(0, &seeded_light(D3DLIGHT_POINT, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state_with_overflow,
            FfWrite {
                name: "SetLight changes an overflow light's type",
                write: |s| s.set_light_at(100, &seeded_light(D3DLIGHT_DIRECTIONAL, 1.0)),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state_with_overflow,
            FfWrite {
                name: "LightEnable turns an overflow light off",
                write: |s| s.set_light_enabled_at(100, false),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state,
            FfWrite {
                name: "LightEnable materializes an overflow light on",
                write: |s| s.set_light_enabled_at(200, true),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state_with_overflow,
            FfWrite {
                name: "LightEnable ahead of an overflow light shifts its compacted slot",
                write: |s| s.set_light_enabled_at(2, true),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state_with_overflow,
            FfWrite {
                name: "LightEnable turns a fast-path light off beside an overflow light",
                write: |s| s.set_light_enabled_at(0, false),
                fingerprint: light_inputs,
            },
        ),
        (
            lit_state_with_overflow,
            FfWrite {
                name: "SetLight retypes a fast-path light beside an overflow light",
                write: |s| s.set_light_at(0, &seeded_light(D3DLIGHT_DIRECTIONAL, 1.0)),
                fingerprint: light_inputs,
            },
        ),
    ];
    for (
        setup,
        FfWrite {
            name,
            write,
            fingerprint,
        },
    ) in cases
    {
        let mut state = setup();
        let inputs = fingerprint(&state);
        let sources = vs_sources(&state);
        write(&mut state);
        assert_ne!(
            vs_sources(&state),
            sources,
            "{name}: the case does not change the FF VS source, so it pins nothing"
        );
        assert_ne!(
            fingerprint(&state),
            inputs,
            "{name}: the FF VS source changed but the setter's fingerprint did not, so the \
             write would not mark VS_SOURCE"
        );
    }
}

#[test]
fn vs_source_inputs_hold_across_value_only_ff_writes() {
    use mtld3d_types::{
        D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DTS_PROJECTION, D3DTS_TEXTURE0,
        D3DTS_VIEW, D3DTS_WORLD,
    };
    let writes = [
        FfWrite {
            name: "SetTransform WORLD",
            write: |s| {
                s.set_transform(D3DTS_WORLD, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "SetTransform WORLDMATRIX under the high water",
            write: |s| {
                s.set_transform(D3DTS_WORLD + 2, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "SetTransform VIEW",
            write: |s| {
                s.set_transform(D3DTS_VIEW, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "SetTransform PROJECTION",
            write: |s| {
                s.set_transform(D3DTS_PROJECTION, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "SetTransform TEXTURE0",
            write: |s| {
                s.set_transform(D3DTS_TEXTURE0, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "MultiplyTransform VIEW",
            write: |s| {
                s.multiply_transform(D3DTS_VIEW, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "MultiplyTransform WORLD",
            write: |s| {
                s.multiply_transform(D3DTS_WORLD, &seeded_matrix(2.0));
            },
            fingerprint: transform_inputs,
        },
        FfWrite {
            name: "SetMaterial",
            write: |s| s.set_material(&seeded_material(2.0)),
            fingerprint: material_inputs,
        },
        FfWrite {
            name: "SetLight keeps an active light's type",
            write: |s| s.set_light_at(0, &seeded_light(D3DLIGHT_POINT, 2.0)),
            fingerprint: light_inputs,
        },
        FfWrite {
            name: "SetLight retypes a disabled light",
            write: |s| s.set_light_at(2, &seeded_light(D3DLIGHT_SPOT, 2.0)),
            fingerprint: light_inputs,
        },
        FfWrite {
            name: "LightEnable repeats an enable",
            write: |s| s.set_light_enabled_at(1, true),
            fingerprint: light_inputs,
        },
        FfWrite {
            name: "LightEnable materializes a light off",
            write: |s| s.set_light_enabled_at(6, false),
            fingerprint: light_inputs,
        },
        FfWrite {
            name: "SetLight keeps an overflow light's type",
            write: |s| s.set_light_at(100, &seeded_light(D3DLIGHT_SPOT, 2.0)),
            fingerprint: light_inputs,
        },
        FfWrite {
            name: "SetLight retypes a disabled overflow light",
            write: |s| s.set_light_at(101, &seeded_light(D3DLIGHT_DIRECTIONAL, 2.0)),
            fingerprint: light_inputs,
        },
    ];
    for (setup_name, setup) in [
        ("fast path", lit_state as fn() -> FfState),
        ("overflow", lit_state_with_overflow),
    ] {
        for FfWrite {
            name,
            write,
            fingerprint,
        } in &writes
        {
            let mut state = setup();
            // Palette high water 3, and a disabled overflow light at 101, so
            // the writes under the high water and the disabled retype have
            // something to leave alone.
            state.set_transform(D3DTS_WORLD + 3, &D3DMATRIX::IDENTITY);
            state.set_light_at(101, &seeded_light(D3DLIGHT_POINT, 1.0));
            let inputs = fingerprint(&state);
            let sources = vs_sources(&state);
            write(&mut state);
            assert_eq!(
                vs_sources(&state),
                sources,
                "{setup_name}, {name}: the write changed the FF VS source"
            );
            assert_eq!(
                fingerprint(&state),
                inputs,
                "{setup_name}, {name}: the write leaves the FF VS source alone but would mark \
                 VS_SOURCE"
            );
        }
    }
}

/// Drift guard: a write that leaves its setter's fingerprint alone leaves the FF VS source alone.
///
/// Holds for every transform, material and light write, over the key and the
/// row count, each write judged by the fingerprint its thunk compares. A
/// deterministic walk over writes that vary every matrix slot, every material
/// and light parameter, every light type and enable. It runs once on the
/// fast-path slots alone and once with overflow indices mixed in, since the
/// light fingerprint encodes the two cases differently and a state never
/// returns from the second to the first; the second walk can enable more
/// than eight lights, so the compaction's truncation is exercised too. A new
/// key input that one of these setters writes, and that its fingerprint does
/// not cover, fails here.
#[test]
fn unchanged_vs_source_inputs_imply_an_unchanged_vs_source() {
    walk_ff_writes(&[0, 1, 3, 7], 0x2545_f491_4f6c_dd1d);
    walk_ff_writes(&[0, 1, 2, 3, 4, 5, 6, 7, 9, 150], 0x9e37_79b9_7f4a_7c15);
}

/// One walk of [`unchanged_vs_source_inputs_imply_an_unchanged_vs_source`] over `light_indices`.
fn walk_ff_writes(light_indices: &[u32], mut rng: u64) {
    use mtld3d_types::{D3DTS_PROJECTION, D3DTS_TEXTURE0, D3DTS_VIEW, D3DTS_WORLD};
    const TRANSFORMS: [u32; 7] = [
        D3DTS_WORLD,
        D3DTS_WORLD + 1,
        D3DTS_WORLD + 7,
        D3DTS_WORLD + 60,
        D3DTS_VIEW,
        D3DTS_PROJECTION,
        D3DTS_TEXTURE0 + 1,
    ];
    const LIGHT_TYPES: [u32; 5] = [0, 1, 2, 3, 4];
    let mut next = |bound: usize| {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(rng >> 33).expect("31 bits fit usize") % bound
    };
    let mut state = FfState::new();
    let (mut held, mut moved, mut truncated) = (0u32, 0u32, 0u32);
    for step in 0..4000 {
        let seed = f32::from(u8::try_from(next(8)).expect("under 8")) * 0.5;
        let op = next(5);
        let fingerprint: fn(&FfState) -> u64 = match op {
            0 | 1 => transform_inputs,
            2 => material_inputs,
            _ => light_inputs,
        };
        let inputs = fingerprint(&state);
        let sources = vs_sources(&state);
        let what = match op {
            0 => {
                let slot = TRANSFORMS[next(TRANSFORMS.len())];
                state.set_transform(slot, &seeded_matrix(seed));
                format!("SetTransform({slot})")
            }
            1 => {
                let slot = TRANSFORMS[next(TRANSFORMS.len())];
                state.multiply_transform(slot, &seeded_matrix(seed));
                format!("MultiplyTransform({slot})")
            }
            2 => {
                state.set_material(&seeded_material(seed));
                "SetMaterial".to_owned()
            }
            3 => {
                let index = light_indices[next(light_indices.len())];
                let ty = LIGHT_TYPES[next(LIGHT_TYPES.len())];
                state.set_light_at(index, &seeded_light(ty, seed));
                format!("SetLight({index}, type {ty})")
            }
            _ => {
                let index = light_indices[next(light_indices.len())];
                // Bias toward enabling so the walk spends time with many
                // lights on, past the eight the compaction keeps.
                let on = next(6) != 0;
                state.set_light_enabled_at(index, on);
                format!("LightEnable({index}, {on})")
            }
        };
        let enabled = light_indices
            .iter()
            .filter(|&&index| {
                state.is_light_enabled_at(index)
                    && state.get_light_at(index).is_some_and(|l| l.type_ != 0)
            })
            .count();
        if enabled > super::MAX_ACTIVE_LIGHTS as usize {
            truncated += 1;
        }
        if fingerprint(&state) == inputs {
            held += 1;
            assert_eq!(
                vs_sources(&state),
                sources,
                "step {step}, {what}: the FF VS source changed but the setter's fingerprint did \
                 not"
            );
        } else {
            moved += 1;
        }
    }
    assert!(
        held > 1000 && moved > 100,
        "the walk over {light_indices:?} must exercise both outcomes: {held} held, {moved} moved"
    );
    if light_indices.len() > super::MAX_ACTIVE_LIGHTS as usize {
        assert!(
            truncated > 300,
            "the walk over {light_indices:?} must run past {} active lights: {truncated} steps did",
            super::MAX_ACTIVE_LIGHTS
        );
    }
}

/// An unhonoured `D3DTS_*` index is dropped, warned once per index, and marks nothing.
///
/// `SetTransform` and `MultiplyTransform` keep separate latches, so the second
/// setter still warns for an index the first one already reported.
#[test]
fn an_unhonoured_transform_index_is_dropped_and_warned_once_per_setter_and_index() {
    capture_warnings();
    let mut state = FfState::new();
    let _ = state.take_ff_vs_dirty();
    for _ in 0..3 {
        assert!(!state.set_transform(1000, &D3DMATRIX::IDENTITY));
        assert!(!state.multiply_transform(1000, &D3DMATRIX::IDENTITY));
    }
    assert!(!state.set_transform(1001, &D3DMATRIX::IDENTITY));
    assert!(
        state.take_ff_vs_dirty().is_empty(),
        "a dropped value marks no section"
    );
    assert_eq!(
        warnings_containing("SetTransform: D3DTS_1000 not honoured"),
        1
    );
    assert_eq!(
        warnings_containing("MultiplyTransform: D3DTS_1000 not honoured"),
        1
    );
    assert_eq!(
        warnings_containing("SetTransform: D3DTS_1001 not honoured"),
        1
    );
}
