//! Unit tests for the FF MSL emitter.
//!
//! These assert the presence of key fragments in the generated source —
//! they do NOT invoke a Metal compiler.

use mtld3d_shared::mtl::{PS_LOD_BIAS_SLOT, VS_POS_FIXUP_SLOT};
use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_GREATER, D3DFOG_LINEAR, D3DMCS_COLOR1, D3DMCS_COLOR2, D3DMCS_MATERIAL,
    D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTA_DIFFUSE, D3DTA_SPECULAR, D3DTA_TEXTURE,
    D3DTOP_DISABLE, D3DTOP_MODULATE, D3DTOP_SELECTARG1,
};

use super::{FfPsKey, FfStage, FfStageFlags, FfVsFlags, FfVsKey, emit_ps_ff, emit_vs_ff};
use crate::dxso::emit::{VariantFlags, VariantKey};

/// D3D enum constant at the key's narrow width.
fn narrow(v: u32) -> u8 {
    u8::try_from(v).expect("D3D9 fixed-function enum value ≤ u8::MAX")
}

fn emit_pair_for_tests(vs_key: &FfVsKey, ps_key: &FfPsKey, variant: VariantKey) -> String {
    let vs = emit_vs_ff(vs_key);
    let ps = emit_ps_ff(ps_key, variant);
    format!("{vs}\n{ps}")
}

fn stage_disable() -> FfStage {
    FfStage {
        color_op: narrow(D3DTOP_DISABLE),
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
        // D3D9 spec defaults for the four material-source render states.
        diffuse_source: narrow(D3DMCS_COLOR1),
        ambient_source: narrow(D3DMCS_MATERIAL),
        specular_source: narrow(D3DMCS_COLOR2),
        emissive_source: narrow(D3DMCS_MATERIAL),
        fog_mode: 0,
        tci_modes: [0; 8],
        tci_coord_indices: [0; 8],
        tex_coord_dims: [0; 8],
        tt_flags: [0; 8],
        vertex_blend_count: 0,
        declared_weights_count: 0,
        clip_plane_count: 0,
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
fn premultiplied_blend_reads_unmodified_texture_alpha_for_both_channels() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_TFACTOR, D3DTOP_BLENDTEXTUREALPHAPM};

    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_BLENDTEXTUREALPHAPM),
        color_arg1: narrow(D3DTA_DIFFUSE | D3DTA_COMPLEMENT | D3DTA_ALPHAREPLICATE),
        color_arg2: narrow(D3DTA_TFACTOR),
        alpha_op: narrow(D3DTOP_BLENDTEXTUREALPHAPM),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        alpha_arg2: narrow(D3DTA_CURRENT),
        flags: FfStageFlags::HAS_TEXTURE,
    };
    assert_eq!(ps.sampled_stage_mask(), 1);
    assert!(ps.reads_texture_factor());
    let msl = emit_ps_ff(&ps, VariantKey::default());
    assert!(
        msl.contains("float4 t0 = s0.sample(samp0, in.texcoord0.xy);"),
        "{msl}"
    );
    assert!(
        msl.contains("saturate((1.0 - in.color0.aaaa) + ps_c[0] * (1.0 - t0.a))"),
        "{msl}"
    );
    assert!(
        msl.contains("saturate(in.color0 + current * (1.0 - t0.a))"),
        "{msl}"
    );

    ps.stages[0].flags.remove(FfStageFlags::HAS_TEXTURE);
    assert_eq!(ps.sampled_stage_mask(), 0);
    let missing = emit_ps_ff(&ps, VariantKey::default());
    assert!(!missing.contains("[[texture(0)]]"), "{missing}");
    assert!(
        missing.contains("saturate((1.0 - in.color0.aaaa) + ps_c[0])"),
        "{missing}"
    );
    assert!(
        missing.contains("saturate(in.color0 + current)"),
        "{missing}"
    );

    ps.stages[0].color_arg1 = narrow(D3DTA_TEXTURE | D3DTA_COMPLEMENT);
    let explicit_missing = emit_ps_ff(&ps, VariantKey::default());
    assert!(
        explicit_missing.contains("current = float4((current).rgb,"),
        "{explicit_missing}"
    );
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
    // The eye normal uses the D3D9 normal matrix, the upper-left block of the
    // full 4x4 inverse of WV, built from generalised cross products of the
    // WV columns (`vs_c[0..3]`), and is NOT renormalized when
    // D3DRS_NORMALIZENORMALS is clear (the default) — a non-unit model normal
    // scales the lighting.
    assert!(msl.contains("static inline float4 mtld3d_cross4("), "{msl}");
    assert!(
        msl.contains("float4 nadj0 = -mtld3d_cross4(vs_c[1], vs_c[2], vs_c[3]);"),
        "{msl}"
    );
    assert!(
        msl.contains("float4 nadj1 = mtld3d_cross4(vs_c[2], vs_c[3], vs_c[0]);"),
        "{msl}"
    );
    assert!(
        msl.contains("float4 nadj2 = -mtld3d_cross4(vs_c[3], vs_c[0], vs_c[1]);"),
        "{msl}"
    );
    // The inverse rows are applied to the model normal as-is; only the
    // singular fallback reads the transposed components.
    assert!(
        msl.contains("dot(nadj0.xyz, in.v1.xyz), dot(nadj1.xyz, in.v1.xyz), dot(nadj2.xyz, in.v1.xyz)) / nwvdet"),
        "{msl}"
    );
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
fn the_light_vector_is_declared_only_where_it_is_read() {
    // `L` has three readers: the diffuse N.L term and the specular
    // half-angle, both of which need a vertex normal, and the spot cone
    // factor. A light with none of them would declare a local nothing reads.
    for normal in [false, true] {
        for specular in [false, true] {
            for (directional, spot) in [(true, false), (false, false), (false, true)] {
                let mut vs = default_vs_key();
                vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
                vs.flags.set(FfVsFlags::HAS_NORMAL, normal);
                vs.flags.set(FfVsFlags::SPECULAR_ENABLE, specular);
                vs.light_active_mask = 1;
                vs.light_directional_mask = u8::from(directional);
                vs.light_spot_mask = u8::from(spot);
                let msl = emit_vs_ff(&vs);
                let case = format!(
                    "normal={normal} specular={specular} directional={directional} spot={spot}\n{msl}"
                );

                assert_eq!(
                    msl.matches("float3 L = ").count(),
                    usize::from(normal || spot),
                    "{case}"
                );
                // Each reader appears exactly under the condition that emits
                // it, so a declaration is present wherever one is read.
                assert_eq!(msl.contains("dot(n, L)"), normal, "{case}");
                assert_eq!(
                    msl.contains("normalize(L + V)"),
                    normal && specular,
                    "{case}"
                );
                assert_eq!(msl.contains("dot(-L,"), spot, "{case}");
                // Slot 0's ambient row is 18; its term reads no light vector
                // and applies to every light, and the POINT / SPOT distance
                // attenuation keeps its own use of `toL`.
                assert!(msl.contains("diffuseAccum += atten * (vs_c[18]"), "{case}");
                assert_eq!(
                    msl.contains("float dist = length(toL);"),
                    !directional,
                    "{case}"
                );
            }
        }
    }
}

#[test]
fn the_eye_space_position_is_declared_only_where_it_is_read() {
    // `posEye` has three kinds of reader: the vertex-to-light vector of a
    // POINT or SPOT light, the local-viewer `V`, which also needs a normal
    // and specular, and the texgen modes that read camera-space position or
    // a reflection vector. A key with none of them would declare a local
    // nothing reads; a key with one that lost the declaration would not
    // compile, so every combination is swept.
    for bits in 0u8..16 {
        let lighting = bits & 1 != 0;
        let normal = bits & 2 != 0;
        let specular = bits & 4 != 0;
        let local_viewer = bits & 8 != 0;
        // (active, directional, spot) masks: no light, directional, point, spot.
        for (light_active, directional, spot) in [(0, 0, 0), (1, 1, 0), (1, 0, 0), (1, 0, 1)] {
            // TCI modes: passthru, CAMERASPACENORMAL, CAMERASPACEPOSITION,
            // CAMERASPACEREFLECTIONVECTOR, SPHEREMAP.
            for tci in 0u8..=4 {
                for blend in [0u8, 1] {
                    let mut vs = default_vs_key();
                    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, lighting);
                    vs.flags.set(FfVsFlags::HAS_NORMAL, normal);
                    vs.flags.set(FfVsFlags::SPECULAR_ENABLE, specular);
                    vs.flags.set(FfVsFlags::LOCAL_VIEWER, local_viewer);
                    vs.light_active_mask = light_active;
                    vs.light_directional_mask = directional;
                    vs.light_spot_mask = spot;
                    vs.tex_coord_count = 1;
                    vs.input_tex_coord_count = 1;
                    vs.tci_modes[0] = tci;
                    vs.vertex_blend_count = blend;
                    vs.declared_weights_count = blend;
                    let msl = emit_vs_ff(&vs);
                    let case = format!(
                        "lighting={lighting} normal={normal} specular={specular} local_viewer={local_viewer} light={light_active}/{directional}/{spot} tci={tci} blend={blend}\n{msl}"
                    );

                    // Mode 3 without a vertex normal falls back to passthru,
                    // which reads no eye-space position; mode 4 reads one
                    // either way.
                    let texgen_reads = tci == 2 || tci == 4 || (tci == 3 && normal);
                    let light_vector_reads = lighting && light_active != 0 && directional == 0;
                    let local_viewer_reads = lighting && normal && specular && local_viewer;
                    let reads = texgen_reads || light_vector_reads || local_viewer_reads;

                    assert_eq!(
                        msl.matches("float3 posEye = ").count(),
                        usize::from(reads),
                        "{case}"
                    );
                    // Each reader appears exactly under the condition that
                    // emits it, so a declaration is present wherever one is
                    // read and the name never occurs undeclared.
                    assert_eq!(msl.contains(" - posEye;"), light_vector_reads, "{case}");
                    assert_eq!(
                        msl.contains("normalize(-posEye)"),
                        local_viewer_reads,
                        "{case}"
                    );
                    assert_eq!(msl.contains("float4(posEye, 0.0)"), tci == 2, "{case}");
                    assert_eq!(
                        msl.contains("normalize(posEye)"),
                        tci == 4 || (tci == 3 && normal),
                        "{case}"
                    );
                    assert_eq!(msl.contains("posEye"), reads, "{case}");
                    // The blended path has its own declaration site, and the
                    // infinite-viewer `V` is the term that must not move with
                    // the declaration.
                    assert_eq!(
                        msl.contains("float3 posEye = pos_view.xyz;"),
                        reads && blend != 0,
                        "{case}"
                    );
                    assert_eq!(
                        msl.contains("float3 V = float3(0.0, 0.0, -1.0);"),
                        lighting && normal && specular && !local_viewer,
                        "{case}"
                    );
                }
            }
        }
    }
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
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_SPECULAR),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&ps, VariantKey::default());
    assert!(msl.contains("in.color1"), "{msl}");
}

#[test]
fn d3dta_specular_alpha_replicate_broadcasts() {
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_SPECULAR | D3DTA_ALPHAREPLICATE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
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
        color_op: narrow(D3DTOP_MODULATE),
        color_arg1: narrow(D3DTA_TEXTURE),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_MODULATE),
        alpha_arg1: narrow(D3DTA_TEXTURE),
        alpha_arg2: narrow(D3DTA_CURRENT),
        flags: FfStageFlags::HAS_TEXTURE,
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
        flags: FfStageFlags::HAS_TEXTURE,
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
            fetch4_mask: 0,
            fetch4_alpha_mask: 0,
            raw_depth_red_mask: 0,
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
        flags: FfStageFlags::HAS_TEXTURE,
    };

    let volume = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            volume_sampler_mask: 0b1,
            cube_sampler_mask: 0,
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
            cube_sampler_mask: 0,
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
fn cube_sampler_mask_emits_texturecube_and_xyz_sample() {
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: 2,
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 2,
        alpha_arg1: 2,
        alpha_arg2: 1,
        flags: FfStageFlags::HAS_TEXTURE,
    };

    let cube = emit_pair_for_tests(
        &vs,
        &ps,
        VariantKey {
            cube_sampler_mask: 1,
            ..VariantKey::default()
        },
    );
    assert!(
        cube.contains("texturecube<float> s0 [[texture(0)]]"),
        "{cube}"
    );
    assert!(
        cube.contains("float4 t0 = s0.sample(samp0, in.texcoord0.xyz);"),
        "{cube}"
    );
    assert!(!cube.contains("texture2d<float> s0"), "{cube}");
}

#[test]
fn emits_alpha_test_discard() {
    let vs = default_vs_key();
    let ps = default_ps_key();
    let variant = VariantKey {
        alpha_func: narrow(D3DCMP_GREATER),
        fog_mode: 0,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        sample_mask: 0,
        volume_sampler_mask: 0,
        cube_sampler_mask: 0,
        tt_projected_mask: 0,
        color_out_mask: 0,
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
    vs.fog_mode = narrow(D3DFOG_LINEAR);
    let ps = default_ps_key();
    let variant = VariantKey {
        alpha_func: 0,
        fog_mode: 3,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        sample_mask: 0,
        volume_sampler_mask: 0,
        cube_sampler_mask: 0,
        tt_projected_mask: 0,
        color_out_mask: 0,
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
        alpha_func: narrow(D3DCMP_ALWAYS),
        fog_mode: 0,
        fog_table_mode: 0,
        depth_sampler_mask: 0,
        depth_fetch_mask: 0,
        fetch4_mask: 0,
        fetch4_alpha_mask: 0,
        raw_depth_red_mask: 0,
        sample_mask: 0,
        volume_sampler_mask: 0,
        cube_sampler_mask: 0,
        tt_projected_mask: 0,
        color_out_mask: 0,
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
        flags: FfStageFlags::HAS_TEXTURE,
    };
    ps.stages[1] = FfStage {
        color_op: 4,   // MODULATE
        color_arg1: 2, // TEXTURE
        color_arg2: 1, // CURRENT (stage 0's output)
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        flags: FfStageFlags::HAS_TEXTURE,
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
        flags: FfStageFlags::empty(),
    };
    ps.stages[1] = stage_disable();
    ps.stages[2] = FfStage {
        color_op: 4, // MODULATE
        color_arg1: 2,
        color_arg2: 1,
        alpha_op: 4,
        alpha_arg1: 2,
        alpha_arg2: 1,
        flags: FfStageFlags::HAS_TEXTURE,
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
        flags: FfStageFlags::HAS_TEXTURE,
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
    // D3D9 defines R = 2 (E.N) N - E with E the unit vector from the vertex
    // to the eye. `posEye` is the vertex in camera space, so
    // I = normalize(posEye) = -E and R = I - 2 (I.N) N = reflect(I, N).
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
        flags: FfStageFlags::HAS_TEXTURE,
    };
    let msl = emit_pair_for_tests(&vs, &ps, VariantKey::default());
    // Eye-space normal + position must be declared (pre-scan hoist, since
    // lighting is disabled in the default key).
    assert_eq!(msl.matches("float3 n = normalize(").count(), 1, "{msl}");
    assert_eq!(msl.matches("float3 posEye =").count(), 1, "{msl}");
    assert!(
        msl.contains(concat!(
            "    float4 raw0;\n",
            "    {\n",
            "        float3 E_tci = normalize(posEye);\n",
            "        float3 R_tci = reflect(E_tci, n);\n",
            "        raw0 = float4(R_tci, 0.0);\n",
            "    }\n",
            "    out.texcoord0 = raw0;\n",
        )),
        "{msl}"
    );
    // The mirrored form, which feeds the eye-to-vertex direction into the
    // vertex-to-eye formula, is the negated vector.
    assert!(!msl.contains("dot(n, E_tci)"), "{msl}");
    // Passthru from a bogus input v4 must NOT be emitted at slot 0.
    assert!(!msl.contains("in.v4"), "{msl}");
}

#[test]
fn tci_cameraspacereflection_vertex_blended_reads_the_blended_locals() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.vertex_blend_count = 1;
    vs.declared_weights_count = 1;
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 3;
    let msl = emit_vs_ff(&vs);
    assert_eq!(
        msl.matches("    float3 posEye = pos_view.xyz;\n").count(),
        1,
        "{msl}"
    );
    assert_eq!(
        msl.matches("    float3 n = normalize(n_blend);\n").count(),
        1,
        "{msl}"
    );
    assert!(
        msl.contains(concat!(
            "        float3 E_tci = normalize(posEye);\n",
            "        float3 R_tci = reflect(E_tci, n);\n",
            "        raw0 = float4(R_tci, 0.0);\n",
        )),
        "{msl}"
    );
    assert!(!msl.contains("dot(n, E_tci)"), "{msl}");
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
        flags: FfStageFlags::HAS_TEXTURE,
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
        flags: FfStageFlags::HAS_TEXTURE,
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
fn eye_space_locals_are_declared_once_for_every_lighting_normal_and_tci_mix() {
    // `posEye` and `n` each have two possible declaration sites, the texgen
    // pre-scan and the lighting branch, and the two sites own them under
    // different conditions: lighting declares `posEye` whenever one of its
    // own terms reads it, which the point light below always does, and `n`
    // only with a vertex normal. A second declaration of either is a Metal
    // compile error, a missing one an undeclared identifier, so every
    // combination pins the count of both.
    for blended in [false, true] {
        for lighting in [false, true] {
            for normal in [false, true] {
                for mode in 0..=4u8 {
                    let mut vs = default_vs_key();
                    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, lighting);
                    vs.flags.set(FfVsFlags::HAS_NORMAL, normal);
                    // One directional and one point light: the point light
                    // consumes `posEye`, both consume `n`.
                    vs.light_active_mask = 0b11;
                    vs.light_directional_mask = 0b01;
                    if blended {
                        vs.vertex_blend_count = 1;
                        vs.declared_weights_count = 1;
                    }
                    vs.tex_coord_count = 1;
                    vs.tci_modes[0] = mode;
                    let msl = emit_vs_ff(&vs);
                    let case = format!(
                        "blended={blended} lighting={lighting} normal={normal} tci={mode}\n{msl}"
                    );

                    let pos_eye_decls = msl.matches("float3 posEye =").count();
                    let normal_decls = msl.matches("float3 n =").count();
                    // A normal-less CAMERASPACEREFLECTIONVECTOR stage falls
                    // back to passthru and reads no eye-space position, so
                    // the pre-scan does not hoist one for it.
                    let wants_pos_eye = lighting || mode == 2 || mode == 4 || (mode == 3 && normal);
                    let wants_normal = normal && (lighting || matches!(mode, 1 | 3 | 4));
                    assert_eq!(pos_eye_decls, usize::from(wants_pos_eye), "{case}");
                    assert_eq!(normal_decls, usize::from(wants_normal), "{case}");

                    // Every consumer has its declaration.
                    let pos_eye_uses = msl.matches("posEye").count() - pos_eye_decls;
                    let normal_uses =
                        msl.matches("dot(n, ").count() + msl.matches("reflect(E_tci, n)").count();
                    assert!(pos_eye_uses == 0 || pos_eye_decls == 1, "{case}");
                    assert!(normal_uses == 0 || normal_decls == 1, "{case}");

                    // The texgen source each mode resolves to.
                    let raw = match mode {
                        1 if normal => "float4 raw0 = float4(n_texgen, 0.0);",
                        2 => "float4 raw0 = float4(posEye, 0.0);",
                        3 if normal => "raw0 = float4(R_tci, 0.0);",
                        4 => "raw0 = float4(R_tci.xy / m_tci + 0.5, 0.0, 0.0);",
                        _ => "float4 raw0 = float4(0.0);",
                    };
                    assert!(msl.contains(raw), "{case}");
                    // The reflection vector reflects the view direction about
                    // the vertex normal in both modes that read it.
                    if mode == 3 && normal {
                        assert!(msl.contains("float3 R_tci = reflect(E_tci, n);"), "{case}");
                    }
                    // The sphere map reflects about the vertex normal, and
                    // about a zero normal when the vertex has none.
                    if mode == 4 {
                        let reflection = if normal {
                            "float3 R_tci = reflect(E_tci, n);"
                        } else {
                            "float3 R_tci = E_tci;"
                        };
                        assert!(msl.contains(reflection), "{case}");
                    }
                    // Lighting without a normal keeps its position-dependent
                    // ambient term and drops the N.L term.
                    if lighting {
                        assert!(
                            msl.contains("float3 toL = vs_c[21].xyz - posEye;"),
                            "{case}"
                        );
                        assert_eq!(msl.contains("float ndotl ="), normal, "{case}");
                    }
                }
            }
        }
    }
}

#[test]
fn tci_spheremap_emits_sphere_map_of_the_reflection_vector() {
    // D3DTSS_TEXCOORDINDEX = 0x40000 (TCI_SPHEREMAP). With lighting off the
    // pre-scan hoists both eye-space locals, and the stage emits
    // R = reflect(normalize(posEye), n), m = 2 * |R + (0, 0, 1)| and the
    // coordinate (R.x / m + 0.5, R.y / m + 0.5, 0, 0).
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    vs.tci_modes[0] = 4;
    let msl = emit_vs_ff(&vs);
    assert_eq!(msl.matches("float3 n = normalize(").count(), 1, "{msl}");
    assert_eq!(msl.matches("float3 posEye =").count(), 1, "{msl}");
    for line in [
        "        float3 E_tci = normalize(posEye);\n",
        "        float3 R_tci = reflect(E_tci, n);\n",
        "        float m_tci = 2.0 * length(R_tci + float3(0.0, 0.0, 1.0));\n",
        "        raw0 = float4(R_tci.xy / m_tci + 0.5, 0.0, 0.0);\n",
        "    out.texcoord0 = raw0;\n",
    ] {
        assert!(msl.contains(line), "missing {line:?}: {msl}");
    }
    // The declared input coordinate is not what the stage reads.
    assert!(!msl.contains("in.v4"), "{msl}");
}

#[test]
fn tci_spheremap_without_normal_reflects_about_a_zero_normal() {
    // A vertex without a normal reads a zero normal, so the reflection vector
    // is the view direction itself: the stage still generates coordinates,
    // needs no `n`, and does not fall back to the input coordinate.
    let mut vs = default_vs_key();
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    vs.tci_modes[0] = 4;
    let msl = emit_vs_ff(&vs);
    assert_eq!(msl.matches("float3 posEye =").count(), 1, "{msl}");
    assert!(!msl.contains("float3 n ="), "{msl}");
    assert!(msl.contains("        float3 R_tci = E_tci;\n"), "{msl}");
    assert!(
        msl.contains("raw0 = float4(R_tci.xy / m_tci + 0.5, 0.0, 0.0);"),
        "{msl}"
    );
    assert!(!msl.contains("in.v4"), "{msl}");
}

#[test]
fn tci_spheremap_lit_reuses_the_lighting_locals() {
    // With lighting on and a normal the lighting branch owns both locals; the
    // sphere map reads them and the pre-scan hoists neither.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.flags.set(FfVsFlags::LIGHTING_ENABLED, true);
    vs.light_active_mask = 1;
    vs.light_directional_mask = 1;
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 4;
    let msl = emit_vs_ff(&vs);
    assert_eq!(msl.matches("float3 posEye =").count(), 1, "{msl}");
    assert_eq!(msl.matches("float3 n =").count(), 1, "{msl}");
    assert!(!msl.contains("float3 n = normalize("), "{msl}");
    assert!(msl.contains("float3 R_tci = reflect(E_tci, n);"), "{msl}");
}

#[test]
fn tci_spheremap_vertex_blended_reads_the_blended_locals() {
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.vertex_blend_count = 1;
    vs.declared_weights_count = 1;
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 4;
    let msl = emit_vs_ff(&vs);
    assert_eq!(
        msl.matches("    float3 posEye = pos_view.xyz;\n").count(),
        1,
        "{msl}"
    );
    assert_eq!(
        msl.matches("    float3 n = normalize(n_blend);\n").count(),
        1,
        "{msl}"
    );
    assert!(msl.contains("float3 R_tci = reflect(E_tci, n);"), "{msl}");
}

#[test]
fn tci_spheremap_texture_transform_applies_to_the_generated_coordinate() {
    // The generated coordinate has dimension 3, so D3DTTFF_COUNT2 pads
    // component 3 to 1.0 and multiplies (u, v, 0, 1) by the stage matrix,
    // after the sphere map has been computed.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.tex_coord_count = 1;
    vs.tci_modes[0] = 4;
    vs.tt_flags[0] = 2;
    let msl = emit_vs_ff(&vs);
    let generated = msl
        .find("raw0 = float4(R_tci.xy / m_tci + 0.5, 0.0, 0.0);")
        .unwrap_or_else(|| panic!("no generated coordinate: {msl}"));
    let pad = msl
        .find("    raw0[3] = 1.0;\n")
        .unwrap_or_else(|| panic!("no pad: {msl}"));
    let mul = msl
        .find("    float4 r0 = float4(dot(raw0, vs_c[63]), dot(raw0, vs_c[64]), dot(raw0, vs_c[65]), dot(raw0, vs_c[66]));\n")
        .unwrap_or_else(|| panic!("no matrix multiply: {msl}"));
    assert!(generated < pad && pad < mul, "{msl}");
    assert!(
        msl.contains("    out.texcoord0 = float4(r0.x, r0.y, 0.0, 0.0);\n"),
        "{msl}"
    );
}

#[test]
fn tci_modes_above_spheremap_pass_the_input_coordinate_through() {
    // Modes 5 and up are undefined; the stage reads its input coordinate and
    // hoists nothing.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_NORMAL, true);
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    vs.tci_modes[0] = 5;
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("float4 raw0 = float4(in.v4.xy, 0.0, 0.0);"),
        "{msl}"
    );
    assert!(!msl.contains("posEye"), "{msl}");
    assert!(!msl.contains("R_tci"), "{msl}");
}

#[test]
fn tci_spheremap_on_xyzrhw_passes_the_input_coordinate_through() {
    // Pre-transformed vertices have no eye space, so every texgen mode reads
    // the declared coordinate.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    vs.tex_coord_count = 1;
    vs.input_tex_coord_count = 1;
    vs.tex_coord_dims[0] = 2;
    vs.tci_modes[0] = 4;
    let msl = emit_vs_ff(&vs);
    assert!(msl.contains("out.texcoord0 = float4(("), "{msl}");
    assert!(msl.contains("in.v4"), "{msl}");
    assert!(!msl.contains("R_tci"), "{msl}");
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
fn rhw_emits_pos_fixup_selected_depth_clamp() {
    // The XYZRHW epilogue realizes the D3D9 depth-clamp rule in the shader:
    // `pos_fixup.z != 0` (set per-draw when the depth test is inactive)
    // clamps clip-space z to [0, w]. Shader-side because encoder-level
    // `MTLDepthClipMode::Clamp` is not honoured by every Metal device, and
    // AFTER the `fog_z` write so table fog keeps reading unclamped depth.
    let mut vs = default_vs_key();
    vs.flags.set(FfVsFlags::HAS_RHW, true);
    let msl = emit_vs_ff(&vs);
    let clamp =
        "if (pos_fixup.z != 0.0) { out.position.z = clamp(out.position.z, 0.0, out.position.w); }";
    assert!(
        msl.contains(clamp),
        "RHW epilogue must carry the pos_fixup.z-selected depth clamp:\n{msl}"
    );
    let fog_z = msl.find("out.fog_z =").expect("RHW epilogue writes fog_z");
    let clamp_at = msl.find(clamp).expect("checked above");
    assert!(
        fog_z < clamp_at,
        "fog_z must be written from the unclamped position:\n{msl}"
    );
    // The regular (non-RHW) transform path clips like every other draw and
    // must not grow the clamp.
    let msl_plain = emit_vs_ff(&default_vs_key());
    assert!(
        !msl_plain.contains("pos_fixup.z != 0.0"),
        "non-RHW FF VS must not emit the depth clamp:\n{msl_plain}"
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
        msl.contains(&format!(
            "constant float4 &pos_fixup [[buffer({VS_POS_FIXUP_SLOT})]]"
        )),
        "FF VS must declare the pos_fixup uniform at its slot:\n{msl}"
    );
    assert!(
        msl.contains("out.position.x += pos_fixup.x * out.position.w;")
            && msl.contains("out.position.y += pos_fixup.y * out.position.w;"),
        "FF transform must apply the half-pixel pos_fixup epilogue:\n{msl}"
    );
}

#[test]
fn ff_point_size_converts_the_clamped_size_to_render_pixels() {
    // D3D9 states every point size in the resolution it reports, so the
    // POINTSIZE_MIN/MAX clamp runs there and the result converts once, by
    // `pos_fixup.w`, into the render pixels `[[point_size]]` is measured in.
    let mut vs = default_vs_key();
    for flags in [
        vs.flags,
        vs.flags | FfVsFlags::HAS_PSIZE,
        vs.flags | FfVsFlags::HAS_RHW,
    ] {
        vs.flags = flags;
        let msl = emit_vs_ff(&vs);
        assert!(
            msl.contains(
                "out.point_size = clamp(psize, vs_draw.point.y, vs_draw.point.z) * pos_fixup.w;"
            ),
            "FF VS must convert the clamped point size to render pixels:\n{msl}"
        );
    }
}

#[test]
fn ff_point_scale_measures_the_viewport_height_in_reported_pixels() {
    // The attenuation factor is `Vh * Si / sqrt(A + B*De + C*De^2)` with `Vh`
    // the viewport height D3D9 reports. `pos_fixup.y` carries the reciprocal
    // of the height in the bound target's own space, so dividing by
    // `pos_fixup.w` takes it back to the reported one and the epilogue's
    // single conversion stays the only one.
    let mut vs = default_vs_key();
    vs.flags |= FfVsFlags::POINT_SCALE;
    let msl = emit_vs_ff(&vs);
    assert!(
        msl.contains("psize *= ((-1.0 / pos_fixup.y) / pos_fixup.w) / sqrt(max("),
        "FF point scale must read the viewport height in reported pixels:\n{msl}"
    );
}

#[test]
fn lod_bias_variant_biases_the_fixed_function_sample() {
    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_TEXTURE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TEXTURE),
        flags: FfStageFlags::HAS_TEXTURE,
        ..FfStage::default()
    };

    let plain = emit_ps_ff(&ps, VariantKey::default());
    assert!(
        !plain.contains("lod_bias"),
        "an unbiased scene keeps the shader it had:\n{plain}"
    );

    let variant = VariantKey {
        flags: VariantFlags::LOD_BIAS,
        ..VariantKey::default()
    };
    let biased = emit_ps_ff(&ps, variant);
    assert!(
        biased.contains(&format!(
            "constant float4 *lod_bias [[buffer({PS_LOD_BIAS_SLOT})]]"
        )),
        "the biased variant takes the per-slot bias table:\n{biased}"
    );
    assert!(
        biased.contains("s0.sample(samp0, in.texcoord0.xy, bias(lod_bias[0].x))"),
        "stage 0 samples with its own slot's bias:\n{biased}"
    );
}

#[test]
fn ff_ps_writes_the_sample_mask_when_the_variant_carries_one() {
    // The fixed-function pixel pipeline returns one colour, so the struct
    // exists only to carry the `[[sample_mask]]` output Metal needs for
    // `D3DRS_MULTISAMPLEMASK`.
    let vs = default_vs_key();
    let ps = default_ps_key();
    let variant = VariantKey {
        sample_mask: 0b0001,
        flags: VariantFlags::SAMPLE_MASK,
        ..VariantKey::default()
    };
    let msl = emit_pair_for_tests(&vs, &ps, variant);
    assert!(
        msl.contains("uint oMask [[sample_mask]];"),
        "the FF PS output struct must declare the mask:\n{msl}"
    );
    assert!(
        msl.contains("return FfPsOut { oC0, 1u };"),
        "and write the variant's value:\n{msl}"
    );
}

#[test]
fn ff_ps_returns_a_bare_colour_without_a_sample_mask() {
    let msl = emit_pair_for_tests(&default_vs_key(), &default_ps_key(), VariantKey::default());
    assert!(!msl.contains("sample_mask"), "{msl}");
    assert!(msl.contains("    return oC0;"), "{msl}");
}

#[test]
fn range_fog_changes_only_the_vertex_distance_expression() {
    for mode in 1..=3 {
        for blends in [0, 2] {
            let mut key = default_vs_key();
            key.fog_mode = mode;
            key.vertex_blend_count = blends;
            key.declared_weights_count = u8::from(blends != 0);
            let ordinary = emit_vs_ff(&key);
            assert!(ordinary.contains("float eyeZ = abs(dot(pos, vs_c[2]));"));
            assert!(!ordinary.contains("length(pos_view.xyz)"));
            key.flags.insert(FfVsFlags::RANGE_FOG);
            let range = emit_vs_ff(&key);
            assert!(range.contains("float eyeZ = length(pos_view.xyz);"));
            assert_eq!(
                range.replace("length(pos_view.xyz)", "abs(dot(pos, vs_c[2]))"),
                ordinary,
                "ordinary fog keeps the same source and range changes only distance"
            );
        }
    }
}

#[test]
fn dotproduct3_supplies_the_whole_stage_result() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_TFACTOR, D3DTOP_DOTPRODUCT3};

    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_DOTPRODUCT3),
        color_arg1: narrow(D3DTA_DIFFUSE | D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT),
        color_arg2: narrow(D3DTA_TFACTOR),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let expected = "    current = float4(saturate(4.0 * dot(((1.0 - in.color0.aaaa)).rgb - 0.5, (ps_c[0]).rgb - 0.5)));";
    let msl = emit_ps_ff(&ps, VariantKey::default());
    assert!(msl.lines().any(|line| line == expected), "{msl}");
    // The separate alpha operation cannot affect a DOTPRODUCT3 color stage.
    ps.stages[0].alpha_op = narrow(D3DTOP_MODULATE);
    ps.stages[0].alpha_arg1 = narrow(D3DTA_TFACTOR);
    assert_eq!(msl, emit_ps_ff(&ps, VariantKey::default()));
}

#[test]
fn modulate_keeps_separate_color_and_alpha_arguments() {
    use mtld3d_types::D3DTA_TFACTOR;

    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_MODULATE),
        color_arg1: narrow(D3DTA_TEXTURE),
        color_arg2: narrow(D3DTA_DIFFUSE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TFACTOR),
        flags: FfStageFlags::HAS_TEXTURE,
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&ps, VariantKey::default());
    let expected = "    current = float4(((t0 * in.color0)).rgb, (ps_c[0]).a);";
    assert!(msl.lines().any(|line| line == expected), "{msl}");
}

#[test]
fn dotproduct3_unbound_color_keeps_independent_alpha() {
    use mtld3d_types::{D3DTA_TFACTOR, D3DTOP_DOTPRODUCT3};

    let mut ps = default_ps_key();
    ps.stages[0] = FfStage {
        color_op: narrow(D3DTOP_DOTPRODUCT3),
        color_arg1: narrow(D3DTA_TEXTURE),
        color_arg2: narrow(D3DTA_DIFFUSE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TFACTOR),
        ..FfStage::default()
    };
    let fallback = emit_ps_ff(&ps, VariantKey::default());
    ps.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
    ps.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
    assert_eq!(fallback, emit_ps_ff(&ps, VariantKey::default()));
    assert!(fallback.contains("current = float4((current).rgb, (ps_c[0]).a);"));
}

#[test]
fn premultiplied_alpha_follows_effective_dotproduct3_color() {
    use mtld3d_types::{D3DTA_TFACTOR, D3DTOP_BLENDTEXTUREALPHAPM, D3DTOP_DOTPRODUCT3};

    for has_texture in [false, true] {
        let mut ps = default_ps_key();
        ps.stages[0] = FfStage {
            color_op: narrow(D3DTOP_DOTPRODUCT3),
            color_arg1: narrow(D3DTA_DIFFUSE),
            color_arg2: narrow(D3DTA_TFACTOR),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_DIFFUSE),
            alpha_arg2: narrow(D3DTA_TFACTOR),
            flags: if has_texture {
                FfStageFlags::HAS_TEXTURE
            } else {
                FfStageFlags::empty()
            },
        };
        let dot = emit_ps_ff(&ps, VariantKey::default());
        ps.stages[0].alpha_op = narrow(D3DTOP_BLENDTEXTUREALPHAPM);
        assert_eq!(dot, emit_ps_ff(&ps, VariantKey::default()));

        // A missing explicit color argument falls back before DOT3 can
        // override alpha, leaving the independent PM operation effective.
        ps.stages[0].flags.remove(FfStageFlags::HAS_TEXTURE);
        ps.stages[0].color_arg1 = narrow(D3DTA_TEXTURE);
        let fallback = emit_ps_ff(&ps, VariantKey::default());
        assert!(
            fallback
                .contains("current = float4((current).rgb, (saturate(in.color0 + ps_c[0])).a);")
        );
        ps.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
        ps.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
        assert_eq!(fallback, emit_ps_ff(&ps, VariantKey::default()));
    }
}

#[test]
fn temp_register_keeps_both_old_channels_until_whole_result_assignment() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_TEMP, D3DTA_TFACTOR};

    use super::FfStageResult;
    let mut stages = [stage_disable(); 8];
    for stage in &mut stages[..3] {
        stage.color_op = narrow(D3DTOP_SELECTARG1);
        stage.alpha_op = narrow(D3DTOP_SELECTARG1);
    }
    stages[0].color_arg1 = narrow(D3DTA_TFACTOR);
    stages[0].alpha_arg1 = narrow(D3DTA_TFACTOR);
    stages[0].set_result(FfStageResult::Temp);
    stages[1].color_arg1 = narrow(D3DTA_TEMP | D3DTA_ALPHAREPLICATE);
    stages[1].alpha_arg1 = narrow(D3DTA_TEMP | D3DTA_COMPLEMENT);
    stages[1].set_result(FfStageResult::Temp);
    stages[2].color_arg1 = narrow(D3DTA_TEMP);
    stages[2].alpha_arg1 = narrow(D3DTA_CURRENT);
    let key = FfPsKey {
        stages,
        specular_add: false,
        tt_projected_mask: 0,
    };
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert_eq!(msl.matches("float4 temp = float4(0.0);").count(), 1);
    assert!(msl.contains("temp = float4((temp.aaaa).rgb, ((1.0 - temp)).a);"));
    assert!(msl.contains("current = float4((temp).rgb, (current).a);"));
    assert!(msl.contains("float4 oC0 = current;"));
    assert!(!msl.contains("temp.rgb ="));
    assert!(!msl.contains("temp.a ="));
}

#[test]
fn unused_temp_operands_and_disabled_stages_emit_no_register() {
    use mtld3d_types::D3DTA_TEMP;

    use super::FfStageResult;
    let mut stages = [stage_disable(); 8];
    stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_CURRENT),
        color_arg2: narrow(D3DTA_TEMP),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        alpha_arg2: narrow(D3DTA_TEMP),
        flags: FfStageFlags::empty(),
    };
    stages[1].set_result(FfStageResult::Temp);
    stages[2] = stages[0];
    stages[2].color_arg1 = narrow(D3DTA_TEMP);
    let mut key = FfPsKey {
        stages,
        specular_add: false,
        tt_projected_mask: 0,
    };
    assert!(!emit_ps_ff(&key, VariantKey::default()).contains("float4 temp"));
    // The unbound-texture fallback replaces this whole color expression.
    key.stages[0].color_op = narrow(D3DTOP_MODULATE);
    key.stages[0].color_arg1 = narrow(D3DTA_TEXTURE);
    assert!(!emit_ps_ff(&key, VariantKey::default()).contains("float4 temp"));
    key.stages[0].flags.insert(FfStageFlags::HAS_TEXTURE);
    assert!(emit_ps_ff(&key, VariantKey::default()).contains("float4 temp"));
}

#[test]
fn packed_stage_preserves_legacy_current_hash_stream_and_key_sizes() {
    use std::hash::{Hash, Hasher};

    use super::FfStageResult;
    #[derive(Hash)]
    struct LegacyStage {
        color_op: u8,
        color_arg1: u8,
        color_arg2: u8,
        alpha_op: u8,
        alpha_arg1: u8,
        alpha_arg2: u8,
        has_texture: bool,
    }
    #[derive(Hash)]
    struct LegacyKey {
        stages: [LegacyStage; 8],
        specular_add: bool,
        tt_projected_mask: u8,
    }
    #[derive(Default)]
    struct HashWrites(Vec<u8>);
    impl Hasher for HashWrites {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }
    fn writes(value: &impl Hash) -> Vec<u8> {
        let mut h = HashWrites::default();
        value.hash(&mut h);
        h.0
    }
    assert_eq!(size_of::<FfStage>(), 7);
    assert_eq!(size_of::<FfPsKey>(), 58);
    assert_eq!(align_of::<FfPsKey>(), 1);
    for mask in 0u8..=u8::MAX {
        let stages = std::array::from_fn(|i| FfStage {
            color_op: narrow(D3DTOP_MODULATE),
            color_arg1: narrow(D3DTA_TEXTURE),
            color_arg2: narrow(D3DTA_DIFFUSE),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_CURRENT),
            alpha_arg2: narrow(D3DTA_TEXTURE),
            flags: if mask & (1 << i) != 0 {
                FfStageFlags::HAS_TEXTURE
            } else {
                FfStageFlags::empty()
            },
        });
        let mut key = FfPsKey {
            stages,
            specular_add: mask & 1 != 0,
            tt_projected_mask: mask,
        };
        let legacy = LegacyKey {
            stages: key.stages.each_ref().map(|s| LegacyStage {
                color_op: s.color_op,
                color_arg1: s.color_arg1,
                color_arg2: s.color_arg2,
                alpha_op: s.alpha_op,
                alpha_arg1: s.alpha_arg1,
                alpha_arg2: s.alpha_arg2,
                has_texture: s.has_texture(),
            }),
            specular_add: key.specular_add,
            tt_projected_mask: key.tt_projected_mask,
        };
        assert_eq!(writes(&key), writes(&legacy), "CURRENT stream, mask {mask}");
        assert_eq!(
            crate::shader_cache::ff_key_hash(&key),
            crate::shader_cache::ff_key_hash(&legacy)
        );
        let current = key.clone();
        key.stages[0].set_result(FfStageResult::Temp);
        assert_ne!(key, current);
        assert_ne!(writes(&key), writes(&current));
        assert_ne!(
            crate::shader_cache::ff_key_hash(&key),
            crate::shader_cache::ff_key_hash(&current)
        );
        assert_eq!(key.stages[0].has_texture(), mask & 1 != 0);
    }
}

#[test]
fn temp_detection_tracks_effective_dotproduct3_alpha_consumption() {
    use mtld3d_types::{D3DTA_TEMP, D3DTA_TFACTOR, D3DTOP_DOTPRODUCT3};

    use super::FfStageResult;
    let mut key = default_ps_key();
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_DOTPRODUCT3),
        color_arg1: narrow(D3DTA_DIFFUSE),
        color_arg2: narrow(D3DTA_TFACTOR),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_TEMP),
        ..FfStage::default()
    };
    let current = emit_ps_ff(&key, VariantKey::default());
    assert!(
        !current.contains("float4 temp"),
        "effective DOT3 ignores TEMP alpha input"
    );
    key.stages[0].set_result(FfStageResult::Temp);
    let temp = emit_ps_ff(&key, VariantKey::default());
    assert!(temp.contains("float4 temp = float4(0.0);"));
    assert!(temp.contains("temp = float4(saturate(4.0 * dot("));
    assert!(
        !temp.contains("(temp).a"),
        "ignored alpha op must not split the DOT3 write"
    );
    key.stages[0].set_result(FfStageResult::Current);
    key.stages[0].color_arg1 = narrow(D3DTA_TEXTURE);
    let unbound = emit_ps_ff(&key, VariantKey::default());
    assert!(unbound.contains("float4 temp = float4(0.0);"));
    assert!(unbound.contains("current = float4((current).rgb, (temp).a);"));
}

#[test]
fn per_stage_constant_extent_follows_effective_operands() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTOP_DOTPRODUCT3, D3DTOP_SELECTARG2};

    let mut key = default_ps_key();
    let active = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_DIFFUSE),
        color_arg2: narrow(D3DTA_CONSTANT),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_CURRENT),
        alpha_arg2: narrow(D3DTA_CONSTANT),
        flags: FfStageFlags::empty(),
    };
    key.stages[0] = active;
    key.stages[7] = FfStage {
        color_arg1: narrow(D3DTA_CONSTANT),
        ..active
    };
    assert_eq!(
        key.constant_rows(),
        0,
        "unused operands and disabled suffix"
    );
    key.stages[0].color_op = narrow(D3DTOP_SELECTARG2);
    assert_eq!(key.constant_rows(), 2);
    key.stages[0].color_op = narrow(D3DTOP_MODULATE);
    key.stages[0].color_arg1 = narrow(D3DTA_TEXTURE);
    assert_eq!(key.constant_rows(), 0, "unbound color fallback");
    key.stages[0].flags.insert(FfStageFlags::HAS_TEXTURE);
    assert_eq!(
        key.constant_rows(),
        2,
        "texture occupancy makes constant live"
    );
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_DOTPRODUCT3),
        color_arg2: narrow(D3DTA_DIFFUSE),
        alpha_arg1: narrow(D3DTA_CONSTANT),
        ..active
    };
    assert_eq!(key.constant_rows(), 0, "effective DOT3 ignores alpha");
    key.stages[0].color_arg1 = narrow(D3DTA_TEXTURE);
    assert_eq!(
        key.constant_rows(),
        2,
        "DOT3 fallback retains independent alpha"
    );
    key.stages[..7].fill(active);
    assert_eq!(
        key.constant_rows(),
        9,
        "highest active stage has its own row"
    );
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert!(msl.contains("current = float4((ps_c[8]).rgb"));
    // DISABLE and unknown alpha operations retain the emitter's arg1 fallback.
    key.stages = [stage_disable(); 8];
    for alpha_op in [narrow(D3DTOP_DISABLE), 255] {
        key.stages[0] = FfStage {
            alpha_op,
            alpha_arg1: narrow(D3DTA_CONSTANT),
            ..active
        };
        assert_eq!(key.constant_rows(), 2);
    }
}

#[test]
fn modulate_alpha_add_color_unbound_operands_keep_ordinary_fallback_source() {
    use mtld3d_types::{D3DTA_CURRENT, D3DTA_TFACTOR, D3DTOP_MODULATEALPHA_ADDCOLOR};
    for (arg1, arg2) in [
        (D3DTA_TEXTURE, D3DTA_DIFFUSE),
        (D3DTA_DIFFUSE, D3DTA_TEXTURE),
    ] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_MODULATEALPHA_ADDCOLOR),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_TFACTOR),
            ..FfStage::default()
        };
        let fallback = emit_ps_ff(&key, VariantKey::default());
        assert!(!fallback.contains("[[texture("));
        assert!(!fallback.contains("[[sampler("));
        key.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
        key.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
        assert_eq!(fallback, emit_ps_ff(&key, VariantKey::default()));
    }
}

#[test]
fn modulate_alpha_add_color_nontexture_inputs_add_no_sampler_or_uniform() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTOP_MODULATEALPHA_ADDCOLOR};
    let mut key = default_ps_key();
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_MODULATEALPHA_ADDCOLOR),
        color_arg1: narrow(D3DTA_DIFFUSE | D3DTA_COMPLEMENT),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert!(!msl.contains("[[texture("));
    assert!(!msl.contains("[[sampler("));
    assert!(!key.reads_texture_factor());
    assert_eq!(key.sampled_stage_mask(), 0);
    assert!(msl.contains("saturate((1.0 - in.color0) + ((1.0 - in.color0)).a * current)"));
}

#[test]
fn modulate_alpha_add_color_constant_rows_follow_effective_binary_arguments() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_CURRENT, D3DTOP_MODULATEALPHA_ADDCOLOR};
    for (arg1, arg2) in [
        (D3DTA_CONSTANT, D3DTA_CURRENT),
        (D3DTA_CURRENT, D3DTA_CONSTANT),
    ] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_SELECTARG1),
            color_arg1: narrow(D3DTA_DIFFUSE),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_DIFFUSE),
            ..FfStage::default()
        };
        key.stages[1] = FfStage {
            color_op: narrow(D3DTOP_MODULATEALPHA_ADDCOLOR),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_CURRENT),
            ..FfStage::default()
        };
        assert_eq!(key.constant_rows(), 3);
        assert!(key.stages[1].reads_argument(D3DTA_CONSTANT));
        // Explicit missing textures short-circuit the complete color operation.
        if arg1 == D3DTA_CONSTANT {
            key.stages[1].color_arg2 = narrow(D3DTA_TEXTURE);
        } else {
            key.stages[1].color_arg1 = narrow(D3DTA_TEXTURE);
        }
        assert_eq!(key.constant_rows(), 0);
        assert!(!key.stages[1].reads_argument(D3DTA_CONSTANT));
    }
}

#[test]
fn modulate_color_add_alpha_unbound_operands_keep_ordinary_fallback_source() {
    use mtld3d_types::{
        D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTA_TFACTOR, D3DTOP_MODULATECOLOR_ADDALPHA,
    };
    for (arg1, arg2) in [
        (D3DTA_TEXTURE, D3DTA_DIFFUSE),
        (D3DTA_DIFFUSE, D3DTA_TEXTURE),
        (D3DTA_TEXTURE | D3DTA_COMPLEMENT, D3DTA_DIFFUSE),
        (D3DTA_DIFFUSE, D3DTA_TEXTURE | D3DTA_ALPHAREPLICATE),
        (
            D3DTA_TEXTURE | D3DTA_COMPLEMENT | D3DTA_ALPHAREPLICATE,
            D3DTA_DIFFUSE,
        ),
    ] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_MODULATECOLOR_ADDALPHA),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_TFACTOR),
            ..FfStage::default()
        };
        let fallback = emit_ps_ff(&key, VariantKey::default());
        assert!(!fallback.contains("[[texture("));
        assert!(!fallback.contains("[[sampler("));
        key.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
        key.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
        assert_eq!(fallback, emit_ps_ff(&key, VariantKey::default()));
    }
}

#[test]
fn modulate_color_add_alpha_nontexture_inputs_add_no_sampler_or_uniform() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_CURRENT, D3DTOP_MODULATECOLOR_ADDALPHA};
    let mut key = default_ps_key();
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_MODULATECOLOR_ADDALPHA),
        color_arg1: narrow(D3DTA_DIFFUSE | D3DTA_COMPLEMENT),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert!(!msl.contains("[[texture("));
    assert!(!msl.contains("[[sampler("));
    assert!(!key.reads_texture_factor());
    assert_eq!(key.sampled_stage_mask(), 0);
    assert!(msl.contains("saturate((1.0 - in.color0) * current + ((1.0 - in.color0)).a)"));
}

#[test]
fn modulate_color_add_alpha_constant_rows_follow_effective_binary_arguments() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_CURRENT, D3DTOP_MODULATECOLOR_ADDALPHA};
    for (arg1, arg2) in [
        (D3DTA_CONSTANT, D3DTA_CURRENT),
        (D3DTA_CURRENT, D3DTA_CONSTANT),
    ] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_SELECTARG1),
            color_arg1: narrow(D3DTA_DIFFUSE),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_DIFFUSE),
            ..FfStage::default()
        };
        key.stages[1] = FfStage {
            color_op: narrow(D3DTOP_MODULATECOLOR_ADDALPHA),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_CURRENT),
            ..FfStage::default()
        };
        assert_eq!(key.constant_rows(), 3);
        assert!(key.stages[1].reads_argument(D3DTA_CONSTANT));
        // Explicit missing textures short-circuit the complete color operation.
        if arg1 == D3DTA_CONSTANT {
            key.stages[1].color_arg2 = narrow(D3DTA_TEXTURE);
        } else {
            key.stages[1].color_arg1 = narrow(D3DTA_TEXTURE);
        }
        assert_eq!(key.constant_rows(), 0);
        assert!(!key.stages[1].reads_argument(D3DTA_CONSTANT));
    }
}

#[test]
fn modulate_color_add_alpha_dependency_extent_keeps_alpha_and_factor_policy() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_TFACTOR, D3DTOP_MODULATECOLOR_ADDALPHA};
    let mut key = default_ps_key();
    let stage = FfStage {
        color_op: narrow(D3DTOP_MODULATECOLOR_ADDALPHA),
        color_arg1: narrow(D3DTA_DIFFUSE),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_CURRENT),
        ..FfStage::default()
    };
    key.stages[0] = stage;
    key.stages[1] = stage;
    assert_eq!(key.constant_rows(), 0);
    key.stages[1].color_arg1 = narrow(D3DTA_CONSTANT);
    assert_eq!(key.constant_rows(), 3);
    key.stages[1].color_arg2 = narrow(D3DTA_TEXTURE);
    assert_eq!(key.constant_rows(), 0);
    key.stages[1].alpha_arg1 = narrow(D3DTA_CONSTANT);
    assert_eq!(key.constant_rows(), 3, "independent alpha remains live");
    key.stages[1].alpha_arg1 = narrow(D3DTA_TFACTOR);
    assert_eq!(key.constant_rows(), 1);
    key.stages[1].alpha_arg1 = narrow(D3DTA_CURRENT);
    key.stages[1].color_arg1 = narrow(D3DTA_TFACTOR);
    assert_eq!(
        key.constant_rows(),
        1,
        "texture-factor capture stays conservative"
    );
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert!(
        !msl.contains("ps_c[0]"),
        "suppressed factor has no expression"
    );
    key.stages[1].color_arg2 = narrow(D3DTA_CURRENT);
    assert_eq!(key.constant_rows(), 1, "live factor uses only row zero");
    key.stages[1].color_arg1 = narrow(D3DTA_CONSTANT);
    key.stages[0] = stage_disable();
    assert_eq!(key.constant_rows(), 0, "disabled cascade ends dependencies");
}

#[test]
fn modulate_inverse_alpha_add_color_fallback_preserves_source_identity() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_TFACTOR, D3DTOP_MODULATEINVALPHA_ADDCOLOR};
    for (arg1, arg2) in [
        (D3DTA_TEXTURE | D3DTA_COMPLEMENT, D3DTA_DIFFUSE),
        (D3DTA_DIFFUSE, D3DTA_TEXTURE | D3DTA_ALPHAREPLICATE),
    ] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_MODULATEINVALPHA_ADDCOLOR),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_TFACTOR),
            ..FfStage::default()
        };
        let fallback = emit_ps_ff(&key, VariantKey::default());
        assert!(!fallback.contains("[[texture("));
        assert!(!fallback.contains("[[sampler("));
        key.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
        key.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
        assert_eq!(fallback, emit_ps_ff(&key, VariantKey::default()));
    }
}

#[test]
fn modulate_inverse_alpha_add_color_uses_existing_binary_constant_extent() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_TFACTOR, D3DTOP_MODULATEINVALPHA_ADDCOLOR};
    for constant_first in [true, false] {
        let mut key = default_ps_key();
        key.stages[0] = FfStage {
            color_op: narrow(D3DTOP_SELECTARG1),
            color_arg1: narrow(D3DTA_DIFFUSE),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_DIFFUSE),
            ..FfStage::default()
        };
        key.stages[1] = FfStage {
            color_op: narrow(D3DTOP_MODULATEINVALPHA_ADDCOLOR),
            color_arg1: narrow(if constant_first {
                D3DTA_CONSTANT
            } else {
                D3DTA_CURRENT
            }),
            color_arg2: narrow(if constant_first {
                D3DTA_CURRENT
            } else {
                D3DTA_CONSTANT
            }),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_CURRENT),
            ..FfStage::default()
        };
        assert_eq!(key.constant_rows(), 3);
        if constant_first {
            key.stages[1].color_arg2 = narrow(D3DTA_TEXTURE);
        } else {
            key.stages[1].color_arg1 = narrow(D3DTA_TEXTURE);
        }
        assert_eq!(key.constant_rows(), 0, "unbound color suppresses CONSTANT");
        key.stages[1].alpha_arg1 = narrow(D3DTA_CONSTANT);
        assert_eq!(key.constant_rows(), 3, "independent alpha retains CONSTANT");
        key.stages[1].alpha_arg1 = narrow(D3DTA_TFACTOR);
        assert_eq!(key.constant_rows(), 1);
        key.stages[1].alpha_arg1 = narrow(D3DTA_CURRENT);
        key.stages[1].color_arg1 = narrow(D3DTA_TFACTOR);
        key.stages[1].color_arg2 = narrow(D3DTA_TEXTURE);
        assert_eq!(
            key.constant_rows(),
            1,
            "inherited conservative TFACTOR extent"
        );
        assert!(
            emit_ps_ff(&key, VariantKey::default())
                .contains("current = float4((current).rgb, (current).a);")
        );
        key.stages[0].color_op = narrow(D3DTOP_DISABLE);
        assert_eq!(key.constant_rows(), 0, "disabled tail does not contribute");
    }
}

#[test]
fn modulate_inv_color_add_alpha_unbound_operands_keep_ordinary_fallback_source() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTA_TFACTOR, D3DTOP_MODULATEINVCOLOR_ADDALPHA};
    for modifier in [
        0,
        D3DTA_COMPLEMENT,
        D3DTA_ALPHAREPLICATE,
        D3DTA_COMPLEMENT | D3DTA_ALPHAREPLICATE,
    ] {
        for (arg1, arg2) in [
            (D3DTA_TEXTURE | modifier, D3DTA_DIFFUSE),
            (D3DTA_DIFFUSE, D3DTA_TEXTURE | modifier),
        ] {
            let mut key = default_ps_key();
            key.stages[0] = FfStage {
                color_op: narrow(D3DTOP_MODULATEINVCOLOR_ADDALPHA),
                color_arg1: narrow(arg1),
                color_arg2: narrow(arg2),
                alpha_op: narrow(D3DTOP_SELECTARG1),
                alpha_arg1: narrow(D3DTA_TFACTOR),
                ..FfStage::default()
            };
            let fallback = emit_ps_ff(&key, VariantKey::default());
            assert!(!fallback.contains("[[texture("));
            assert!(!fallback.contains("[[sampler("));
            key.stages[0].color_op = narrow(D3DTOP_SELECTARG1);
            key.stages[0].color_arg1 = narrow(D3DTA_CURRENT);
            assert_eq!(fallback, emit_ps_ff(&key, VariantKey::default()));
        }
    }
}

#[test]
fn modulate_inv_color_add_alpha_nontexture_inputs_add_no_resource() {
    use mtld3d_types::{D3DTA_COMPLEMENT, D3DTOP_MODULATEINVCOLOR_ADDALPHA};
    let mut key = default_ps_key();
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_MODULATEINVCOLOR_ADDALPHA),
        color_arg1: narrow(D3DTA_DIFFUSE | D3DTA_COMPLEMENT),
        color_arg2: narrow(D3DTA_CURRENT),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    let msl = emit_ps_ff(&key, VariantKey::default());
    assert!(!msl.contains("[[texture("));
    assert!(!msl.contains("[[sampler("));
    assert_eq!(key.constant_rows(), 0);
    assert_eq!(key.sampled_stage_mask(), 0);
    assert!(msl.contains("saturate((1.0 - (1.0 - in.color0)) * current + ((1.0 - in.color0)).a)"));
}

#[test]
fn modulate_inv_color_add_alpha_constant_rows_follow_binary_arguments() {
    use mtld3d_types::{D3DTA_CONSTANT, D3DTA_TFACTOR, D3DTOP_MODULATEINVCOLOR_ADDALPHA};
    let mut key = default_ps_key();
    key.stages[0] = FfStage {
        color_op: narrow(D3DTOP_SELECTARG1),
        color_arg1: narrow(D3DTA_DIFFUSE),
        alpha_op: narrow(D3DTOP_SELECTARG1),
        alpha_arg1: narrow(D3DTA_DIFFUSE),
        ..FfStage::default()
    };
    for (arg1, arg2, rows) in [
        (D3DTA_CONSTANT, D3DTA_CURRENT, 3),
        (D3DTA_CURRENT, D3DTA_CONSTANT, 3),
        (D3DTA_TFACTOR, D3DTA_CURRENT, 1),
        (D3DTA_CURRENT, D3DTA_DIFFUSE, 0),
        (D3DTA_CONSTANT, D3DTA_TEXTURE, 0),
        (D3DTA_TEXTURE, D3DTA_CONSTANT, 0),
        // TFACTOR retains the existing conservative uniform extent.
        (D3DTA_TFACTOR, D3DTA_TEXTURE, 1),
    ] {
        key.stages[1] = FfStage {
            color_op: narrow(D3DTOP_MODULATEINVCOLOR_ADDALPHA),
            color_arg1: narrow(arg1),
            color_arg2: narrow(arg2),
            alpha_op: narrow(D3DTOP_SELECTARG1),
            alpha_arg1: narrow(D3DTA_CURRENT),
            ..FfStage::default()
        };
        assert_eq!(key.constant_rows(), rows, "args {arg1}/{arg2}");
        assert_eq!(key.stages[1].reads_argument(D3DTA_CONSTANT), rows == 3);
        if arg2 == D3DTA_TEXTURE {
            let fallback = emit_ps_ff(&key, VariantKey::default());
            assert!(fallback.contains("current = float4((current).rgb, (current).a)"));
            key.stages[1].alpha_arg1 = narrow(D3DTA_CONSTANT);
            assert_eq!(key.constant_rows(), 3);
            key.stages[1].alpha_arg1 = narrow(D3DTA_TFACTOR);
            assert_eq!(key.constant_rows(), 1);
        }
    }
    key.stages[0] = stage_disable();
    key.stages[1].color_arg1 = narrow(D3DTA_CONSTANT);
    assert_eq!(key.constant_rows(), 0, "stages after DISABLE are inactive");
}

#[test]
fn modulate_inv_color_add_alpha_constant_values_leave_shader_identity_unchanged() {
    use mtld3d_types::{
        D3DTA_CONSTANT, D3DTA_TFACTOR, D3DTOP_MODULATEINVCOLOR_ADDALPHA, D3DTSS_ALPHAARG1,
        D3DTSS_ALPHAOP, D3DTSS_COLORARG0, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
        D3DTSS_CONSTANT, render_state_defaults,
    };

    use crate::ff_state::FfState;
    let mut state = FfState::new();
    for (slot, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_DIFFUSE),
    ] {
        state.set_texture_stage_state(0, usize::try_from(slot).unwrap(), value);
    }
    for (slot, value) in [
        (D3DTSS_COLOROP, D3DTOP_MODULATEINVCOLOR_ADDALPHA),
        (D3DTSS_COLORARG0, D3DTA_TEXTURE),
        (D3DTSS_COLORARG1, D3DTA_CONSTANT),
        (D3DTSS_COLORARG2, D3DTA_TFACTOR),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_CURRENT),
    ] {
        state.set_texture_stage_state(1, usize::try_from(slot).unwrap(), value);
    }
    let render_states = render_state_defaults();
    let key = state.build_ps_key(&render_states, 0);
    assert_eq!(
        key.constant_rows(),
        3,
        "unused unbound ARG0 does not suppress the binary operation"
    );
    let source = emit_ps_ff(&key, VariantKey::default());
    for alpha in [0x40, 0x80, 0x40] {
        state.set_texture_stage_state(
            1,
            usize::try_from(D3DTSS_CONSTANT).unwrap(),
            (alpha << 24) | 0x0033_99cc,
        );
        let changed = state.build_ps_key(&render_states, 0);
        assert_eq!(key, changed);
        assert_eq!(source, emit_ps_ff(&changed, VariantKey::default()));
    }
}
