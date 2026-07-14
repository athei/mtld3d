//! Unit tests for the FF MSL emitter.
//!
//! These assert the presence of key fragments in the generated source —
//! they do NOT invoke a Metal compiler.

use super::{
    emit::{VariantFlags, VariantKey},
    ff::{FfPsKey, FfStage, FfVsFlags, FfVsKey, emit_ps_ff, emit_vs_ff},
};

fn emit_pair_for_tests(vs_key: &FfVsKey, ps_key: &FfPsKey, variant: VariantKey) -> String {
    let vs = emit_vs_ff(vs_key);
    let ps = emit_ps_ff(ps_key, variant);
    format!("{vs}\n{ps}")
}

fn stage_disable() -> FfStage {
    FfStage {
        color_op: 1, // D3DTOP_DISABLE
        ..FfStage::default()
    }
}

fn default_vs_key() -> FfVsKey {
    FfVsKey {
        flags: FfVsFlags::HAS_COLOR0 | FfVsFlags::COLOR_VERTEX,
        input_tex_coord_count: 0,
        tex_coord_count: 0,
        light_active_mask: 0,
        light_directional_mask: 0,
        light_spot_mask: 0,
        diffuse_source: 1,  // D3DMCS_COLOR1 (spec default)
        ambient_source: 0,  // D3DMCS_MATERIAL (spec default)
        specular_source: 2, // D3DMCS_COLOR2 (spec default)
        emissive_source: 0, // D3DMCS_MATERIAL (spec default)
        fog_mode: 0,
        tci_modes: [0; 8],
        tci_coord_indices: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
    }
}

fn default_ps_key() -> FfPsKey {
    FfPsKey {
        stages: [stage_disable(); 8],
        specular_add: false,
        tt_projected_mask: 0,
    }
}

#[test]
fn emits_two_step_wv_then_proj() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // Two-step `Proj · (WV · pos)` decomposition. Step 1: WV · pos at
    // vs_c[0..3] → `pos_view`. Step 2: Proj · pos_view at vs_c[4..7]
    // → out.position. Plain MSL `dot()` — hardware dot-product on
    // Apple Silicon. The two-step shape is what's load-bearing here;
    // pre-multiplying WVP CPU-side would give a different FP-rounding
    // shape than any programmable shader.
    assert!(msl.contains("float4 pos_view = float4("), "{msl}");
    assert!(msl.contains("dot(pos, vs_c[0])"), "{msl}");
    assert!(msl.contains("dot(pos, vs_c[3])"), "{msl}");
    assert!(msl.contains("dot(pos_view, vs_c[4])"), "{msl}");
    assert!(msl.contains("dot(pos_view, vs_c[7])"), "{msl}");
    assert!(msl.contains("out.position"), "{msl}");
}

#[test]
fn emits_diffuse_color_identity_when_no_lighting() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // D3DCOLOR is bound as MTLVertexFormat::UChar4Normalized_BGRA, which does
    // the BGRA→RGBA swizzle at vertex-fetch time — the shader reads `.xyzw`
    // directly without a compensating `.zyxw`.
    assert!(msl.contains("out.color0 = in.v2;"), "{msl}");
    assert!(!msl.contains("in.v2.zyxw"), "{msl}");
}

#[test]
fn emits_one_directional_light() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1; // D3DLIGHT_DIRECTIONAL slot 0

    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // The eye normal uses the D3D9 normal matrix via the cofactor (cross-product)
    // form, and is NOT renormalized when D3DRS_NORMALIZENORMALS is clear (the
    // default) — a non-unit model normal scales the lighting.
    assert!(msl.contains("cross(wvr1, wvr2)"), "{msl}");
    // The cofactor inputs must be the WV columns (vs_c[i].xyz), NOT the
    // transposed components — feeding the transpose computes the inverse
    // rotation and makes lighting swim as the camera turns.
    assert!(msl.contains("float3 wvr0 = vs_c[0].xyz;"), "{msl}");
    assert!(!msl.contains("float3(vs_c[0].x, vs_c[1].x"), "{msl}");
    assert!(!msl.contains("n = normalize(n)"), "{msl}");
    assert!(msl.contains("ndotl"), "{msl}");
    assert!(msl.contains("saturate(diffuseAccum)"), "{msl}");
    // Per-light ambient contribution must be modulated by material ambient —
    // without it, saturate clips highlight detail into flat softness.
    assert!(msl.contains("atten * (vs_c["), "{msl}");
    // Alpha preserved from material diffuse after saturate.
    assert!(msl.contains("lit.a = "), "{msl}");
}

#[test]
fn specular_disabled_emits_zero_color1_and_no_pow() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    // specular_enable is false in the default key.
    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(
        msl.contains("out.color1 = float4(saturate(specAccum), 0.0);"),
        "{msl}"
    );
    assert!(!msl.contains("pow(ndoth"), "{msl}");
}

#[test]
fn specular_enabled_emits_blinn_phong_pow() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    vs.flags.set(FfVsFlags::SPECULAR_ENABLE, true);
    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(msl.contains("pow(ndoth, mat_power)"), "{msl}");
    assert!(
        msl.contains("out.color1 = float4(saturate(specAccum), 0.0);"),
        "{msl}"
    );
}

#[test]
fn specular_term_reads_light_specular_row() {
    // The specular weight is lightSpecular × matSpecular — the light's
    // dedicated specular row at base+5 (slot 0 → vs_c[20]), not its
    // diffuse row at base+2.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.flags.set(FfVsFlags::SPECULAR_ENABLE, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(
        msl.contains("specAccum += atten * specFactor * (vs_c[20].rgb"),
        "{msl}"
    );
}

#[test]
fn local_viewer_flag_selects_view_vector_model() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.flags.set(FfVsFlags::SPECULAR_ENABLE, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;

    // LOCAL_VIEWER clear → constant infinite-viewer direction.
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("float3 V = float3(0.0, 0.0, -1.0);"), "{msl}");
    assert!(!msl.contains("normalize(-posEye)"), "{msl}");

    // LOCAL_VIEWER set → per-vertex direction to the eye.
    vs.flags.set(FfVsFlags::LOCAL_VIEWER, true);
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("float3 V = normalize(-posEye);"), "{msl}");
}

#[test]
fn spot_light_emits_cone_factor() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_spot_mask = 1;
    let msl = emit_vs_ff(&vs);
    // Slot 0 rows: direction 16, ambient 18, attenuation 19, specular 20.
    assert!(msl.contains("float rho = dot(-L, vs_c[16].xyz);"), "{msl}");
    assert!(
        msl.contains("pow(saturate(rho * vs_c[20].w + vs_c[18].w), vs_c[16].w)"),
        "{msl}"
    );
    // Spot keeps the POINT distance attenuation + range cutoff.
    assert!(msl.contains("atten_k"), "{msl}");
}

#[test]
fn directional_and_point_lights_emit_no_cone_factor() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 0b11;
    vs.light_directional_mask = 0b01; // slot 0 directional, slot 1 point
    let msl = emit_vs_ff(&vs);
    assert!(!msl.contains("rho"), "{msl}");
}

#[test]
fn fog_mode_4_sources_factor_from_specular_alpha() {
    // fog_mode 4 (vertex+table fog both D3DFOG_NONE) reads the COLOR1/specular
    // alpha as the per-vertex fog factor.
    let mut vs = default_vs_key();
    vs.flags.insert(FfVsFlags::HAS_COLOR1);
    vs.fog_mode = 4;
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("out.fog = float4(in.v3.w, 0.0, 0.0, 0.0);"),
        "fog_mode 4 must read the specular alpha:\n{msl}"
    );
    // No declared specular ⇒ default oFog = 1.0 (unfogged).
    let mut vs_no_spec = default_vs_key();
    vs_no_spec.fog_mode = 4;
    let msl_no_spec = emit_vs_ff(&vs_no_spec);
    assert!(
        msl_no_spec.contains("out.fog = float4(1.0);"),
        "fog_mode 4 without specular must default to unfogged:\n{msl_no_spec}"
    );
}

#[test]
fn ps_specular_add_emitted_before_fog_when_enabled() {
    let mut ps = default_ps_key();
    ps.specular_add = true;
    let variant = VariantKey {
        fog_mode: 3,
        ..VariantKey::default()
    };
    let msl = emit_ps_ff(&ps, variant);
    let add = msl
        .find("current = float4(saturate(current.rgb + in.color1.rgb), current.a);")
        .expect("specular add missing");
    let fog = msl.find("mix(fog_data[0].rgb").expect("fog blend missing");
    assert!(add < fog, "specular add must precede the fog blend:\n{msl}");
}

#[test]
fn ps_specular_add_absent_when_disabled() {
    let msl = emit_ps_ff(&default_ps_key(), VariantKey::default());
    assert!(!msl.contains("in.color1"), "{msl}");
}

#[test]
fn d3dta_specular_resolves_to_color1() {
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,   // D3DTOP_SELECTARG1
        color_arg1: 4, // D3DTA_SPECULAR
        alpha_op: 2,   // D3DTOP_SELECTARG1
        alpha_arg1: 0, // D3DTA_DIFFUSE
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&ps, VariantKey::default());
    assert!(msl.contains("in.color1"), "{msl}");
}

#[test]
fn d3dta_specular_alpha_replicate_broadcasts() {
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,      // D3DTOP_SELECTARG1
        color_arg1: 0x24, // D3DTA_SPECULAR | D3DTA_ALPHAREPLICATE
        alpha_op: 2,      // D3DTOP_SELECTARG1
        alpha_arg1: 0,    // D3DTA_DIFFUSE
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&ps, VariantKey::default());
    assert!(msl.contains("in.color1.aaaa"), "{msl}");
}

#[test]
fn unlit_missing_color0_defaults_to_white() {
    // A missing DIFFUSE stream reads opaque white (the D3D9 default for
    // an absent COLOR0), not the material diffuse constant.
    let mut vs = default_vs_key();
    vs.flags.remove(FfVsFlags::HAS_COLOR0);
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("out.color0 = float4(1.0);"), "{msl}");
    assert!(!msl.contains("out.color0 = vs_c[10];"), "{msl}");
}

#[test]
fn unlit_color1_passes_through_vertex_specular() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_COLOR1, true);
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("out.color1 = in.v3;"), "{msl}");
}

#[test]
fn rhw_color1_passes_through_vertex_specular() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    vs.flags.set(FfVsFlags::HAS_COLOR1, true);
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("out.color1 = in.v3;"), "{msl}");
}

#[test]
fn diffuse_material_source_color1_reads_vertex_color() {
    // WoW writes DIFFUSEMATERIALSOURCE = MCS_COLOR1 (1) with COLORVERTEX = TRUE.
    // The lit modulator must become `in.v2` instead of `vs_c[10]` (the BGRA
    // swizzle happens at vertex-fetch via UChar4Normalized_BGRA). The diffuse
    // material constant lives at row 10 of the VS constant buffer (see the
    // layout in `ff.rs`).
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1; // D3DLIGHT_DIRECTIONAL slot 0
    vs.diffuse_source = 1; // MCS_COLOR1
    vs.flags.set(FfVsFlags::COLOR_VERTEX, true);

    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(msl.contains("* in.v2"), "{msl}");
    assert!(!msl.contains("in.v2.zyxw"), "{msl}");
    assert!(!msl.contains("* vs_c[10]"), "{msl}");
}

#[test]
fn material_source_ignored_when_color_vertex_false() {
    // COLORVERTEX = FALSE: material-source selectors are ignored; always read
    // from the material constant. VS constant-buffer rows: vs_c[9] = global
    // ambient, vs_c[10] = material.diffuse, vs_c[11] = material.ambient.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    vs.diffuse_source = 1;
    vs.ambient_source = 1;
    vs.flags.set(FfVsFlags::COLOR_VERTEX, false);

    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(msl.contains("vs_c[9] * vs_c[11]"), "{msl}");
    assert!(msl.contains("* vs_c[10]"), "{msl}");
}

#[test]
fn emits_texture_sample_and_modulate() {
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 4,   // D3DTOP_MODULATE
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT
        alpha_op: 4,   // D3DTOP_MODULATE
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(msl.contains("texture2d<float> s0 [[texture(0)]]"), "{msl}");
    assert!(msl.contains("sampler samp0 [[sampler(0)]]"), "{msl}");
    assert!(
        msl.contains("float4 t0 = s0.sample(samp0, in.texcoord0.xy);"),
        "{msl}"
    );
    assert!(msl.contains("(t0 * current)"), "{msl}");
}

#[test]
fn depth_sampler_mask_emits_depth2d_and_sample_compare() {
    // A depth-format texture bound to an FF stage (sampleable shadow map) must
    // emit `depth2d<float>` + `sample_compare`, not `texture2d` + plain sample:
    // binding a `Depth32Float` texture to a `texture2d` slot, or using a
    // comparison sampler without `sample_compare`, both fail Metal validation.
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,   // SELECTARG1
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT
        alpha_op: 2,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };

    // Colour variant (mask 0): plain texture2d + sample.
    let plain = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(
        plain.contains("texture2d<float> s0 [[texture(0)]]"),
        "{plain}"
    );
    assert!(
        plain.contains("float4 t0 = s0.sample(samp0, in.texcoord0.xy);"),
        "{plain}"
    );

    // Depth-bound variant (slot 0 set): depth2d + sample_compare.
    let depth = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            depth_sampler_mask: 0b1,
            depth_fetch_mask: 0,
            ..VariantKey::default()
        },
    );
    assert!(
        depth.contains("depth2d<float> s0 [[texture(0)]]"),
        "{depth}"
    );
    assert!(
        depth.contains(
            "float4 t0 = float4(s0.sample_compare(samp0, in.texcoord0.xy, saturate(in.texcoord0.z), level(0)));"
        ),
        "{depth}"
    );
    assert!(
        !depth.contains("texture2d<float> s0"),
        "depth slot must not also emit texture2d: {depth}"
    );
}

#[test]
fn volume_sampler_mask_emits_texture3d_and_xyz_sample() {
    // A volume (3D) texture bound to an FF stage must emit `texture3d<float>`
    // and sample with the texcoord's `.xyz` — the backing MTLTexture is
    // `MTLTextureType3D`, and binding it to a `texture2d` slot fails Metal's
    // type-check and samples black.
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,   // SELECTARG1
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT
        alpha_op: 2,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };

    let volume = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            volume_sampler_mask: 0b1,
            ..VariantKey::default()
        },
    );
    assert!(
        volume.contains("texture3d<float> s0 [[texture(0)]]"),
        "{volume}"
    );
    assert!(
        volume.contains("float4 t0 = s0.sample(samp0, in.texcoord0.xyz);"),
        "{volume}"
    );
    assert!(
        !volume.contains("texture2d<float> s0"),
        "volume slot must not also emit texture2d: {volume}"
    );

    // Projected volume stage divides the full .xyz by .w.
    let mut ps_proj = ps;
    ps_proj.tt_projected_mask = 0b1;
    let projected = emit_pair_for_tests(
        &vs,
        &ps_proj,
        VariantKey {
            volume_sampler_mask: 0b1,
            ..VariantKey::default()
        },
    );
    assert!(
        projected.contains(
            "float4 t0 = s0.sample(samp0, (in.texcoord0.w != 0.0 ? in.texcoord0.xyz / in.texcoord0.w : float3(0.0)));"
        ),
        "{projected}"
    );
}

#[test]
fn emits_alpha_test_discard() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let variant = VariantKey {
        alpha_func: 5, // D3DCMP_GREATER
        fog_mode: 0,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        volume_sampler_mask: 0,
        tt_projected_mask: 0,
        flags: VariantFlags::empty(),
    };
    let msl = emit_pair_for_tests(&vs, &ps, variant);
    assert!(
        msl.contains("if (!(oC0.a > alpha_ref)) discard_fragment();"),
        "{msl}"
    );
    assert!(
        msl.contains("constant float &alpha_ref [[buffer(14)]]"),
        "{msl}"
    );
}

#[test]
fn emits_fog_blend_on_buffer_13_when_enabled() {
    let mut vs = default_vs_key();
    vs.fog_mode = 3; // D3DFOG_LINEAR
    let ps = default_ps_key();
    let variant = VariantKey {
        alpha_func: 0,
        fog_mode: 3,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        volume_sampler_mask: 0,
        tt_projected_mask: 0,
        flags: VariantFlags::empty(),
    };
    let msl = emit_pair_for_tests(&vs, &ps, variant);
    assert!(
        msl.contains("constant float4 *fog_data [[buffer(13)]]"),
        "fog data must bind on slot 13: {msl}"
    );
    assert!(
        msl.contains("mix(fog_data[0].rgb, oC0.rgb, saturate(in.fog.x))"),
        "PS must blend fog with fog_data[0]: {msl}"
    );
    // Fog color binds on its own buffer, and the PS constant buffer holds only
    // row 0 (texture factor) — nothing may index `ps_c[1]`.
    assert!(
        !msl.contains("ps_c[1]"),
        "fog color moved off ps_c — no reference should remain: {msl}"
    );
}

#[test]
fn omits_fog_blend_when_disabled() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(!msl.contains("fog_data"), "{msl}");
    assert!(!msl.contains("in.fog.x"), "{msl}");
}

#[test]
fn table_fog_computes_per_pixel_factor() {
    let vs = default_vs_key();
    let ps = default_ps_key();

    // LINEAR, orthographic projection: source = pixel depth + DEPTHBIAS.
    let linear_z = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            fog_table_mode: 3,
            ..VariantKey::default()
        },
    );
    assert!(
        linear_z.contains("float fog_c = in.fog_z + fog_data[1].w;"),
        "{linear_z}"
    );
    assert!(
        linear_z.contains(
            "float fog_f = fog_range == 0.0 ? 0.0 : saturate((fog_data[1].y - fog_c) / fog_range);"
        ),
        "{linear_z}"
    );
    assert!(
        linear_z.contains("mix(fog_data[0].rgb, oC0.rgb, fog_f)"),
        "{linear_z}"
    );
    assert!(
        !linear_z.contains("in.fog.x"),
        "table fog must ignore the vertex factor: {linear_z}"
    );

    // EXP, perspective projection: source = eye W.
    let exp_w = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            fog_table_mode: 1,
            flags: VariantFlags::FOG_SOURCE_W,
            ..VariantKey::default()
        },
    );
    assert!(
        exp_w.contains("float fog_c = 1.0 / in.position.w;"),
        "{exp_w}"
    );
    assert!(
        exp_w.contains("float fog_f = saturate(precise::exp(-fog_data[1].z * fog_c));"),
        "{exp_w}"
    );

    // EXP2 squares the density-scaled distance.
    let exp2 = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            fog_table_mode: 2,
            flags: VariantFlags::FOG_SOURCE_W,
            ..VariantKey::default()
        },
    );
    assert!(
        exp2.contains("float fog_f = saturate(precise::exp(-fog_dz * fog_dz));"),
        "{exp2}"
    );
}

#[test]
fn omits_alpha_test_when_always() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let variant = VariantKey {
        alpha_func: 8, // D3DCMP_ALWAYS
        fog_mode: 0,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        volume_sampler_mask: 0,
        tt_projected_mask: 0,
        flags: VariantFlags::empty(),
    };
    let msl = emit_pair_for_tests(&vs, &ps, variant);
    assert!(!msl.contains("discard_fragment()"), "{msl}");
}

#[test]
fn rhw_skips_wvp_transform_and_lighting() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // XYZRHW path uses viewport slot at vs_c[0].xy, not WVP transform.
    assert!(
        !msl.contains("dot(pos, vs_c[0])"),
        "RHW path must not emit WVP transform: {msl}"
    );
    assert!(msl.contains("vs_c[0].xy"), "{msl}");
    assert!(msl.contains("ndc_x"), "{msl}");
    assert!(msl.contains("ndc_y"), "{msl}");
    // Texcoord still passed through.
    assert!(msl.contains("out.texcoord0"), "{msl}");
    // Lighting code must not be emitted.
    assert!(!msl.contains("ndotl"), "{msl}");
    assert!(!msl.contains("normalize(float3(dot("), "{msl}");
}

#[test]
fn multi_stage_modulate() {
    let mut vs = default_vs_key();
    vs.tex_coord_count = 2;
    vs.input_tex_coord_count = 2;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,   // SELECTARG1
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT
        alpha_op: 2,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    ps.stages[1] = FfStage {
        color_op: 4,   // MODULATE
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT (stage 0's output)
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(msl.contains("s0.sample(samp0, in.texcoord0.xy)"), "{msl}");
    assert!(msl.contains("s1.sample(samp1, in.texcoord1.xy)"), "{msl}");
    assert!(msl.contains("(t1 * current)"), "{msl}");
}

#[test]
fn stops_at_first_disabled_stage() {
    let vs = default_vs_key();
    let mut ps = default_ps_key();
    // Stage 0 ADD; stage 1 disabled; stage 2 would be MODULATE but must be ignored.
    ps.stages[0] = FfStage {
        color_op: 7,   // ADD
        color_arg1: 0, // DIFFUSE
        color_arg2: 1, // CURRENT
        alpha_op: 2,   // SELECTARG1
        alpha_arg1: 0,
        alpha_arg2: 1,
        has_texture: false,
    };
    ps.stages[1] = stage_disable();
    ps.stages[2] = FfStage {
        color_op: 4, // MODULATE
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // Stage 2 must not emit a texture sample because iteration stopped at stage 1.
    assert!(!msl.contains("s2.sample"), "{msl}");
    assert!(!msl.contains("t2"), "{msl}");
}

#[test]
fn tci_passthru_honours_coord_index_override() {
    // D3DTSS_TEXCOORDINDEX[stage=0] = 2 means "stage 0 reads input coord set 2"
    // with passthru mode. The VS output slot 0 must wire to in.v6 (= input
    // texcoord 2), not in.v4. Requires at least 3 input texcoord attributes
    // declared (v4..v6) since we read coord-set 2.
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 3;
    vs.tci_coord_indices[0] = 2;
    vs.tex_coord_dims[2] = 2; // coord-set 2 is FLOAT2
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 4,
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(
        msl.contains("float4 raw0 = float4(in.v6.xy, 0.0, 0.0);"),
        "{msl}"
    );
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
    // PS still samples slot 0 (stage-aligned varying).
    assert!(msl.contains("s0.sample(samp0, in.texcoord0.xy)"), "{msl}");
}

#[test]
fn tci_cameraspacereflection_emits_reflection_vector() {
    // D3DTSS_TEXCOORDINDEX[stage=0] = 0x30000 (TCI_CAMERASPACEREFLECTION).
    // Requires `has_normal` so eye-space normal is available. VS must emit
    // the reflection calculation into Varyings.texcoord[0].
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 3;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 4,
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // Eye-space normal + position must be declared (pre-scan hoist, since
    // lighting is disabled in the default key).
    assert!(msl.contains("float3 n = normalize("), "{msl}");
    assert!(msl.contains("float3 posEye ="), "{msl}");
    // Reflection vector: R = 2 * N * dot(N, E) - E.
    assert!(msl.contains("2.0 * n * dot(n, E_tci)"), "{msl}");
    assert!(msl.contains("raw0 = float4(R_tci, 0.0);"), "{msl}");
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
    // Passthru from a bogus input v4 must NOT be emitted at slot 0.
    assert!(!msl.contains("in.v4"), "{msl}");
}

#[test]
fn tci_cameraspacenormal_without_normal_falls_back() {
    // TCI_CAMERASPACENORMAL=1 needs a vertex normal. Without one we must
    // silently fall back to passthru (and warn).
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tci_modes[0] = 1;
    vs.tci_coord_indices[0] = 0;
    vs.tex_coord_dims[0] = 2; // coord-set 0 is FLOAT2
    // has_normal = false in default key.
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(
        msl.contains("float4 raw0 = float4(in.v4.xy, 0.0, 0.0);"),
        "{msl}"
    );
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
}

#[test]
fn point_light_applies_range_cutoff() {
    // Point light contribution must be zeroed beyond the light's range —
    // `atten *= step(dist, atten_k.w)` per D3D9 spec.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 0; // D3DLIGHT_POINT slot 0
    let msl = emit_pair_for_tests(&vs, &default_ps_key(), VariantKey::default());
    assert!(msl.contains("atten *= step(dist, atten_k.w);"), "{msl}");
    // atten_k must be declared as float4 so .w is available.
    assert!(msl.contains("float4 atten_k = vs_c["), "{msl}");
}

#[test]
fn ttff_count2_emits_texture_matrix_mul() {
    // D3DTSS_TEXTURETRANSFORMFLAGS = D3DTTFF_COUNT2 (2) on stage 0 with a
    // FLOAT2 coordinate (n=2). The VS pads the first unbacked component
    // (index 2) to 1.0, multiplies by the transposed texture matrix at
    // vs_c[63..66], and keeps 2 components (COUNT2).
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    vs.tt_flags[0] = 2;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 4,
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // First unbacked component (index n=2) padded to 1.0 before the matmul.
    assert!(msl.contains("raw0[2] = 1.0;"), "{msl}");
    // 4×dot against consecutive matrix rows at vs_c[63..66].
    assert!(msl.contains("dot(raw0, vs_c[63])"), "{msl}");
    assert!(msl.contains("dot(raw0, vs_c[66])"), "{msl}");
    // COUNT2 keeps two components, zeroing the rest (non-projected).
    assert!(
        msl.contains("out.texcoord0 = float4(r0.x, r0.y, 0.0, 0.0);"),
        "{msl}"
    );
}

#[test]
fn ttff_disable_skips_matrix_mul() {
    // Without TTFF set the VS passes the raw coordinate through — no matrix
    // multiplication and no component padding.
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    // tt_flags[0] = 0 by default (D3DTTFF_DISABLE).
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(!msl.contains("vs_c[63]"), "{msl}");
    assert!(!msl.contains("raw0[2] = 1.0;"), "{msl}");
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
}

#[test]
fn active_stage_without_input_texcoords_does_not_reference_v4() {
    // FVF=XYZ|DIFFUSE with texture stage 0 active produces
    // `input_tex_coord_count = 0, tex_coord_count = 1`. The VS must emit an
    // output varying so the PS can sample, but must NOT declare `v4` as an
    // attribute or read `in.v4` — the MTLVertexDescriptor built from the
    // FVF has no slot 4, and Metal rejects the pipeline with
    // "Vertex attribute v4(4) is missing from the vertex descriptor".
    let mut vs = default_vs_key();
    vs.input_tex_coord_count = 0;
    vs.tex_coord_count = 1;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 4, // MODULATE
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        has_texture: true,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    assert!(
        !msl.contains("v4 [[attribute(4)]]"),
        "VS must not declare v4 when input_tex_coord_count = 0: {msl}"
    );
    assert!(
        !msl.contains("in.v4"),
        "VS must not read in.v4 when it's not declared: {msl}"
    );
    // Stage 0's varying still exists — it must carry the zero fallback.
    assert!(msl.contains("float4 raw0 = float4(0.0);"), "{msl}");
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
}

#[test]
fn tci_cameraspaceposition_reuses_lighting_poseye() {
    // When lighting is enabled the lighting branch already declares `posEye`;
    // the TCI pre-scan must NOT redeclare it (would be a duplicate-variable
    // compile error in Metal).
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 2; // CAMERASPACEPOSITION
    let ps = default_ps_key();
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    let pos_eye_decls = msl.matches("float3 posEye =").count();
    assert_eq!(pos_eye_decls, 1, "posEye declared exactly once: {msl}");
    assert!(msl.contains("float4 raw0 = float4(posEye, 0.0);"), "{msl}");
    assert!(msl.contains("out.texcoord0 = raw0;"), "{msl}");
}

#[test]
fn emit_vs_ff_tex_coord_count_8_rhw_does_not_panic() {
    // Guards the `for i in 0..vs.tex_coord_count` loop that indexes the
    // per-stage `[u8; 8]` arrays (tci_modes etc.): the construction-side
    // clamp caps tex_coord_count at 8, so every slot must be walked fully
    // without an out-of-bounds panic.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    vs.input_tex_coord_count = 8;
    vs.tex_coord_count = 8;
    let ps = default_ps_key();
    // Exercise every slot so any future OOB shows up here.
    let _ = emit_pair_for_tests(&vs, &ps, VariantKey::default());
}

#[test]
fn emit_vs_ff_tex_coord_count_8_non_rhw_does_not_panic() {
    // Same invariant for the non-XYZRHW branch (the per-stage texcoord loop and
    // the `vs.tci_modes[..active]` TCI pre-scan in `emit_vs`).
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, false);
    vs.input_tex_coord_count = 8;
    vs.tex_coord_count = 8;
    let ps = default_ps_key();
    let _ = emit_pair_for_tests(&vs, &ps, VariantKey::default());
}

#[test]
fn vertex_blend_sequential_2_weight_emits_implicit_last_weight_and_palette_reads() {
    // D3DVBF_2WEIGHTS → vertex_blend_count = 3 (2 explicit + 1 implicit).
    // Sequential mode reads palette[0..2] directly from vs_c[95 + i*4].
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.vertex_blend_count = 3;
    vs.flags.set(FfVsFlags::VERTEX_BLEND_INDEXED, false);
    vs.declared_weights_count = 2;
    let msl = emit_vs_ff(&vs);
    // VertexIn slot for blend_weight (slot 12) — no blend_indices slot in
    // sequential mode (declared_indices is false).
    assert!(
        msl.contains("float4 blend_weight [[attribute(12)]]"),
        "{msl}"
    );
    assert!(
        !msl.contains("blend_indices"),
        "no indices in sequential: {msl}"
    );
    // Two explicit weight iterations + implicit last weight.
    assert!(msl.contains("in.blend_weight[0]"), "{msl}");
    assert!(msl.contains("in.blend_weight[1]"), "{msl}");
    assert!(msl.contains("weight_sum = 0.0"), "{msl}");
    assert!(
        msl.contains("1.0 - weight_sum"),
        "implicit last weight: {msl}"
    );
    // Palette reads at vs_c + 95 + idx * 4.
    assert!(msl.contains("vs_c + 95 + idx * 4u"), "{msl}");
    // Sequential idx assignment for explicit slots and implicit last.
    assert!(msl.contains("uint idx = 0u"), "{msl}");
    assert!(msl.contains("uint idx = 1u"), "{msl}");
    assert!(msl.contains("uint idx = 2u"), "implicit last idx: {msl}");
    // Normal blended via top-3x3 (m[i].xyz) into n_blend.
    assert!(msl.contains("n_blend"), "{msl}");
    assert!(msl.contains("dot(in.v1.xyz, m[0].xyz)"), "{msl}");
}

#[test]
fn vertex_blend_indexed_3_weight_reads_blend_indices() {
    // D3DVBF_3WEIGHTS → vertex_blend_count = 4 (3 explicit + 1 implicit).
    // Indexed mode reads per-vertex BLENDINDICES instead of sequential
    // matrix indices.
    let mut vs = default_vs_key();
    vs.vertex_blend_count = 4;
    vs.flags.set(FfVsFlags::VERTEX_BLEND_INDEXED, true);
    vs.declared_weights_count = 3;
    vs.flags.set(FfVsFlags::DECLARED_INDICES, true);
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("float4 blend_weight [[attribute(12)]]"),
        "{msl}"
    );
    assert!(
        msl.contains("uint4 blend_indices [[attribute(13)]]"),
        "{msl}"
    );
    // Three explicit + one implicit indexed reads.
    for i in 0..4 {
        assert!(
            msl.contains(&format!("in.blend_indices[{i}]")),
            "indexed mode reads blend_indices[{i}]: {msl}"
        );
    }
}

#[test]
fn vertex_blend_indexed_only_0_weights_single_matrix() {
    // D3DVBF_0WEIGHTS + INDEXED → vertex_blend_count = 1.
    // Single-bone path: no weight loop, single matrix at blend_indices[0],
    // implicit weight = 1.0 (no `weight_sum`).
    let mut vs = default_vs_key();
    vs.vertex_blend_count = 1;
    vs.flags.set(FfVsFlags::VERTEX_BLEND_INDEXED, true);
    vs.declared_weights_count = 0;
    vs.flags.set(FfVsFlags::DECLARED_INDICES, true);
    let msl = emit_vs_ff(&vs);
    // No BLENDWEIGHT slot needed.
    assert!(!msl.contains("blend_weight"), "{msl}");
    assert!(
        msl.contains("uint4 blend_indices [[attribute(13)]]"),
        "{msl}"
    );
    // No accumulator weight sum — single-matrix shortcut path.
    assert!(
        !msl.contains("weight_sum"),
        "single matrix has no weight_sum: {msl}"
    );
    assert!(msl.contains("in.blend_indices[0]"), "{msl}");
}

#[test]
fn vertex_blend_off_emits_unchanged_position_math() {
    // When vertex_blend_count = 0 the non-blending math must be exactly the
    // plain position decomposition with no blend paths, so meshes without
    // blending render identically.
    let vs = default_vs_key();
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("float4 pos_view = float4(dot(pos, vs_c[0]), dot(pos, vs_c[1]), dot(pos, vs_c[2]), dot(pos, vs_c[3]));"),
        "non-blending path unchanged: {msl}"
    );
    assert!(!msl.contains("blend_weight"), "{msl}");
    assert!(!msl.contains("blend_indices"), "{msl}");
    assert!(!msl.contains("n_blend"), "{msl}");
}

#[test]
fn xyzrhw_emits_half_pixel_offset_window_to_ndc() {
    // Pre-transformed (XYZRHW) verts map window coords → NDC with the Y-flip
    // plus a half-pixel rasterization fixup. D3D9's window→NDC mapping is
    // shifted half a pixel from Metal's, so without the offset on-boundary
    // geometry lands one pixel up-left of the D3D9 reference. Half a pixel is
    // `1/vp` in NDC (NDC spans 2.0 across `vp` pixels); the fixup moves +right
    // (`+ 1.0 / vp.x`) and +down (`- 1.0 / vp.y`, since Metal NDC is +y-up),
    // folded into ndc_x/ndc_y before the `* w` divide.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    let msl = emit_vs_ff(&vs);
    // Base window→NDC mapping still present.
    assert!(msl.contains("* 2.0 - 1.0"), "{msl}");
    assert!(
        msl.contains("1.0 - ((in.v0.y - vp_origin.y) / vp.y) * 2.0"),
        "{msl}"
    );
    // Half-pixel offset folded into the projected position.
    assert!(
        msl.contains("* 2.0 - 1.0 + 1.0 / vp.x"),
        "ndc_x must carry the +half-pixel-right offset:\n{msl}"
    );
    assert!(
        msl.contains("* 2.0 - 1.0 / vp.y"),
        "ndc_y must carry the +half-pixel-down offset:\n{msl}"
    );
}

#[test]
fn ff_transform_emits_half_pixel_pos_fixup() {
    // The FF transformed-position path (non-XYZRHW) declares the buffer-13
    // `pos_fixup` uniform and shifts clip-space position half a pixel
    // right/down so on-boundary geometry matches the D3D9 window→NDC
    // reference.
    let vs = default_vs_key();
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("constant float4 &pos_fixup [[buffer(13)]]"),
        "FF VS must declare the pos_fixup uniform at slot 13:\n{msl}"
    );
    assert!(
        msl.contains("out.position.x += pos_fixup.x * out.position.w;")
            && msl.contains("out.position.y += pos_fixup.y * out.position.w;"),
        "FF transform must apply the half-pixel pos_fixup epilogue:\n{msl}"
    );
}
