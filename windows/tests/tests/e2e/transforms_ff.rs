//! Fixed-function transform + texture-stage routing + alpha test.

use mtld3d_tests::{
    Harness, LitVertex, PosVertex, SpecularVertex, Texture, Vertex, assert_pixel_approx,
};
use mtld3d_types::{
    D3DCMP_GREATER, D3DCOLORVALUE, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_NORMAL, D3DFVF_SPECULAR,
    D3DFVF_XYZ, D3DLIGHT_DIRECTIONAL, D3DLIGHT_POINT, D3DLIGHT_SPOT, D3DLIGHT9, D3DMATERIAL9,
    D3DMCS_MATERIAL, D3DPT_TRIANGLELIST, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE,
    D3DRS_AMBIENT, D3DRS_AMBIENTMATERIALSOURCE, D3DRS_DIFFUSEMATERIALSOURCE,
    D3DRS_EMISSIVEMATERIALSOURCE, D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_SPECULARENABLE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTA_ALPHAREPLICATE,
    D3DTA_DIFFUSE, D3DTA_SPECULAR, D3DTA_TEXTURE, D3DTADDRESS_WRAP, D3DTEXF_POINT, D3DTOP_MODULATE,
    D3DTOP_SELECTARG1, D3DTS_PROJECTION, D3DTS_TEXTURE0, D3DTS_VIEW, D3DTS_WORLD, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DTSS_TEXCOORDINDEX,
    D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_COUNT2, D3DVECTOR,
};

#[rustfmt::skip]
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

const BLUE: u32 = 0xFF00_00FF;

const fn solid_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

const fn specular_triangle(diffuse: u32, specular: u32) -> [SpecularVertex; 3] {
    [
        SpecularVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            diffuse,
            specular,
        },
        SpecularVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            diffuse,
            specular,
        },
        SpecularVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            diffuse,
            specular,
        },
    ]
}

#[test]
fn transform_round_trips() {
    let h = Harness::new();
    assert_eq!(h.set_transform(D3DTS_VIEW, &IDENTITY), 0, "SetTransform");
    assert_eq!(
        h.transform(D3DTS_VIEW).map(f32::to_bits),
        IDENTITY.map(f32::to_bits),
        "GetTransform must return the matrix we set",
    );
}

#[test]
fn ff_passes_vertex_diffuse() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    // Identity WVP flips the device onto the emit_ff path; result equals the
    // hard-coded passthrough.
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    h.select_diffuse_stage(0);

    let tri = solid_triangle(0xFF00_FF00);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(h.read_pixel(10, 10), BLUE, "corner stays background");
    assert_eq!(
        h.read_pixel(320, 280),
        0xFF00_FF00,
        "center is FF vertex green"
    );
}

#[test]
fn alpha_test_discards_transparent() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    h.select_diffuse_stage(0);

    assert_eq!(h.set_render_state(D3DRS_ALPHATESTENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHAFUNC, D3DCMP_GREATER), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHAREF, 0x80), 0);

    // alpha = 0 (A byte of D3DCOLOR) fails GREATER 0x80 → every fragment killed.
    let tri = solid_triangle(0x0000_FF00);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        BLUE,
        "alpha test discarded all fragments"
    );
}

#[rustfmt::skip]
const SCALE_2X: [f32; 16] = [
    2.0, 0.0, 0.0, 0.0,
    0.0, 2.0, 0.0, 0.0,
    0.0, 0.0, 2.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

#[test]
fn multiply_transform_composes() {
    let h = Harness::new();
    assert_eq!(
        h.set_transform(D3DTS_VIEW, &IDENTITY),
        0,
        "SetTransform identity"
    );
    // identity * scale = scale.
    assert_eq!(
        h.multiply_transform(D3DTS_VIEW, &SCALE_2X),
        0,
        "MultiplyTransform"
    );
    assert_eq!(
        h.transform(D3DTS_VIEW).map(f32::to_bits),
        SCALE_2X.map(f32::to_bits),
        "VIEW = identity * scale",
    );
}

#[test]
fn transform_states_round_trip() {
    let h = Harness::new();
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION, D3DTS_TEXTURE0] {
        assert_eq!(h.set_transform(state, &SCALE_2X), 0, "SetTransform {state}");
        assert_eq!(
            h.transform(state).map(f32::to_bits),
            SCALE_2X.map(f32::to_bits),
            "GetTransform {state} round-trip",
        );
    }
}

#[test]
fn material_round_trips() {
    let h = Harness::new();
    let diffuse = D3DCOLORVALUE {
        r: 0.25,
        g: 0.5,
        b: 0.75,
        a: 1.0,
    };
    let material = D3DMATERIAL9 {
        diffuse,
        ambient: D3DCOLORVALUE::default(),
        specular: D3DCOLORVALUE::default(),
        emissive: D3DCOLORVALUE::default(),
        power: 16.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");
    let got = h.material();
    assert_eq!(
        got.power.to_bits(),
        16.0_f32.to_bits(),
        "material power round-trip"
    );
    assert_eq!(
        got.diffuse.g.to_bits(),
        0.5_f32.to_bits(),
        "material diffuse round-trip"
    );
}

#[test]
fn light_round_trips_and_enables() {
    let h = Harness::new();
    let light = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        range: 50.0,
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight");
    let got = h.light(0);
    assert_eq!(got.type_, D3DLIGHT_POINT, "light type round-trip");
    assert_eq!(
        got.range.to_bits(),
        50.0_f32.to_bits(),
        "light range round-trip"
    );

    assert!(!h.light_enabled(0), "lights default disabled");
    assert_eq!(h.light_enable(0, true), 0, "LightEnable(0, true)");
    assert!(h.light_enabled(0), "GetLightEnable reflects enable");
}

#[test]
fn texture_arg_alpha_replicate() {
    // D3DTA_ALPHAREPLICATE broadcasts the diffuse alpha channel across RGB:
    // diffuse 0x80ff00ff (alpha 0x80) renders 0x808080.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_DIFFUSE | D3DTA_ALPHAREPLICATE),
        0,
        "COLORARG1 = DIFFUSE | ALPHAREPLICATE",
    );

    let tri = solid_triangle(0x80ff_00ff);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });

    let px = h.read_pixel(320, 280);
    let (r, g, b) = ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
    assert!(
        r.abs_diff(0x80) <= 2 && g.abs_diff(0x80) <= 2 && b.abs_diff(0x80) <= 2,
        "alpha (0x80) replicated to RGB, got 0x{px:08x}",
    );
}

#[test]
fn unlit_missing_diffuse_renders_white() {
    // An FVF without DIFFUSE reads opaque white for the FF diffuse input
    // — not the material diffuse constant.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF");
    h.select_diffuse_stage(0);

    let tri = [
        PosVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
        },
        PosVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
        },
        PosVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
        },
    ];
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFFFF_FFFF,
        "missing COLOR0 defaults to opaque white"
    );
}

#[test]
fn specular_add_joins_cascade_when_enabled() {
    // D3D9 end-of-cascade specular add: with lighting off, oD1 is the vertex
    // COLOR1 attribute; SPECULARENABLE adds its rgb to the cascade result
    // after the last texture stage. Diffuse green 0x80 + specular red 0x80 →
    // (0x80, 0x80, 0x00); with SPECULARENABLE off the red never lands.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_SPECULAR),
        0,
        "SetFVF"
    );
    h.select_diffuse_stage(0);
    let tri = specular_triangle(0xFF00_8000, 0xFF80_0000);

    assert_eq!(
        h.set_render_state(D3DRS_SPECULARENABLE, 1),
        0,
        "specular on"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    let px = h.read_pixel(320, 280);
    let (r, g, b) = ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
    assert!(
        r.abs_diff(0x80) <= 2 && g.abs_diff(0x80) <= 2 && b <= 2,
        "specular red added to diffuse green, got 0x{px:08x}",
    );

    assert_eq!(
        h.set_render_state(D3DRS_SPECULARENABLE, 0),
        0,
        "specular off"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    let px = h.read_pixel(320, 280);
    let (r, g) = ((px >> 16) & 0xff, (px >> 8) & 0xff);
    assert!(
        r <= 2 && g.abs_diff(0x80) <= 2,
        "specular add disabled leaves diffuse only, got 0x{px:08x}",
    );
}

#[test]
fn lit_specular_uses_light_specular_color() {
    // The Blinn-Phong specular term weights by lightSpecular × matSpecular.
    // A green-diffuse / red-specular directional light over a black-diffuse,
    // white-specular material must produce a red highlight — green would
    // mean the term still reads the light's diffuse row.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(
        h.set_render_state(D3DRS_SPECULARENABLE, 1),
        0,
        "specular on"
    );
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    h.select_diffuse_stage(0);

    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        ambient: D3DCOLORVALUE::default(),
        specular: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        emissive: D3DCOLORVALUE::default(),
        power: 1.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");

    let light = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        diffuse: D3DCOLORVALUE {
            r: 0.0,
            g: 1.0,
            b: 0.0,
            a: 0.0,
        },
        specular: D3DCOLORVALUE {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
        direction: D3DVECTOR {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight");
    assert_eq!(h.light_enable(0, true), 0, "LightEnable");

    // Camera-facing triangle: normals point back at the viewer, so
    // ndotl = 1 and the half-vector term is large across the surface.
    let tri = [
        LitVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            nx: 0.0,
            ny: 0.0,
            nz: -1.0,
        },
        LitVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            nx: 0.0,
            ny: 0.0,
            nz: -1.0,
        },
        LitVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            nx: 0.0,
            ny: 0.0,
            nz: -1.0,
        },
    ];
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    let px = h.read_pixel(320, 280);
    let (r, g, b) = ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
    assert!(
        r >= 0xC0 && g <= 2 && b <= 2,
        "red highlight from light specular (diffuse green must not leak), got 0x{px:08x}",
    );
}

#[test]
fn lit_without_normal_still_emits_ambient_and_emissive() {
    // FF lighting with no vertex normal: the per-light N·L diffuse/specular
    // terms drop to zero, but the emissive and (global) ambient contributions
    // are normal-independent and must still light the surface. With global
    // ambient = white and a material whose ambient.b = 0.5, emissive.b = 0.25,
    // the result is emissive + ambient*global = 0.25 + 0.5 = 0.75 blue (0xC0).
    // A regression that gates all of lighting on the normal renders the raw
    // (white default) vertex colour instead.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(
        h.set_render_state(D3DRS_AMBIENT, 0xFFFF_FFFF),
        0,
        "global ambient white"
    );
    assert_eq!(
        h.set_render_state(D3DRS_AMBIENTMATERIALSOURCE, D3DMCS_MATERIAL),
        0,
        "ambient from material"
    );
    assert_eq!(
        h.set_render_state(D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_MATERIAL),
        0,
        "emissive from material"
    );
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF (no normal)");
    h.select_diffuse_stage(0);

    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE::default(),
        ambient: D3DCOLORVALUE {
            r: 0.0,
            g: 0.0,
            b: 0.5,
            a: 0.0,
        },
        specular: D3DCOLORVALUE::default(),
        emissive: D3DCOLORVALUE {
            r: 0.0,
            g: 0.0,
            b: 0.25,
            a: 0.0,
        },
        power: 0.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");

    let quad = [
        PosVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
        },
        PosVertex {
            x: -1.0,
            y: 1.0,
            z: 0.5,
        },
        PosVertex {
            x: 1.0,
            y: -1.0,
            z: 0.5,
        },
        PosVertex {
            x: 1.0,
            y: -1.0,
            z: 0.5,
        },
        PosVertex {
            x: -1.0,
            y: 1.0,
            z: 0.5,
        },
        PosVertex {
            x: 1.0,
            y: 1.0,
            z: 0.5,
        },
    ];
    h.render_once(0xFF00_0000, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    let px = h.read_pixel(320, 240);
    let (r, g, b) = ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
    assert!(
        r <= 2 && g <= 2 && b.abs_diff(0xC0) <= 2,
        "no-normal lit draw must emit ambient+emissive (0x0000_00c0), got 0x{px:08x}",
    );
}

#[test]
fn spot_light_cone_limits_lighting() {
    // Spot at the eye aimed down +z. FF lighting is Gouraud (evaluated at
    // vertices), so each probe triangle sits entirely inside or entirely
    // outside the cone: the inner one within theta (full umbra factor),
    // the outer one beyond phi (zero).
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    h.select_diffuse_stage(0);

    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        },
        ambient: D3DCOLORVALUE::default(),
        specular: D3DCOLORVALUE::default(),
        emissive: D3DCOLORVALUE::default(),
        power: 0.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");

    let light = D3DLIGHT9 {
        type_: D3DLIGHT_SPOT,
        diffuse: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        direction: D3DVECTOR {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        range: 10.0,
        attenuation0: 1.0,
        falloff: 1.0,
        theta: 0.9,                        // ~52° full inner cone → half-angle ~26°
        phi: core::f32::consts::FRAC_PI_3, // 60° full outer cone → half-angle 30°
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight");
    assert_eq!(h.light_enable(0, true), 0, "LightEnable");

    let vert = |vx: f32, vy: f32| LitVertex {
        x: vx,
        y: vy,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: -1.0,
    };
    // Inner triangle: ±0.1 around the view axis at z = 0.5 → ~16° off
    // axis, inside the umbra. Outer triangle: 0.5..0.9 off axis → 45°+,
    // outside phi.
    let tris = [
        vert(0.0, 0.1),
        vert(0.1, -0.1),
        vert(-0.1, -0.1),
        vert(0.7, 0.3),
        vert(0.9, -0.3),
        vert(0.5, -0.3),
    ];
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &tris), 0, "draw");
    });

    let inside = h.read_pixel(320, 240);
    let (r, g, b) = ((inside >> 16) & 0xff, (inside >> 8) & 0xff, inside & 0xff);
    assert!(
        r >= 0xE0 && g >= 0xE0 && b >= 0xE0,
        "umbra vertexes take full diffuse, got 0x{inside:08x}",
    );

    let outside = h.read_pixel(544, 264);
    let (r, g, b) = (
        (outside >> 16) & 0xff,
        (outside >> 8) & 0xff,
        outside & 0xff,
    );
    assert!(
        r <= 2 && g <= 2 && b <= 2,
        "beyond-phi vertexes get zero light, got 0x{outside:08x}",
    );
}

#[test]
fn local_viewer_models_diverge_off_axis() {
    // D3DRS_LOCALVIEWER selects the specular view-vector model. For an
    // off-axis surface lit head-on by a directional light, the infinite
    // viewer's constant V is parallel to L (half-vector dot = 1 → full
    // highlight) while the local viewer's per-vertex V tilts away from
    // the axis (dimmer highlight under a power of 8).
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(
        h.set_render_state(D3DRS_SPECULARENABLE, 1),
        0,
        "specular on"
    );
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    h.select_diffuse_stage(0);

    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        ambient: D3DCOLORVALUE::default(),
        specular: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        emissive: D3DCOLORVALUE::default(),
        power: 8.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");

    let light = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        specular: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        direction: D3DVECTOR {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight");
    assert_eq!(h.light_enable(0, true), 0, "LightEnable");

    let vert = |vx: f32, vy: f32| LitVertex {
        x: vx,
        y: vy,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: -1.0,
    };
    // Off-axis triangle; probe at its centroid (NDC 0.6, 0.167 → pixel
    // 512, 200).
    let tri = [vert(0.4, 0.3), vert(0.8, 0.3), vert(0.6, -0.1)];

    assert_eq!(h.set_render_state(D3DRS_LOCALVIEWER, 1), 0, "local viewer");
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    let local = (h.read_pixel(512, 200) >> 16) & 0xff;

    assert_eq!(
        h.set_render_state(D3DRS_LOCALVIEWER, 0),
        0,
        "infinite viewer"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    let infinite = (h.read_pixel(512, 200) >> 16) & 0xff;

    assert!(
        infinite >= 0xF0,
        "infinite viewer: V ∥ L → full highlight, got 0x{infinite:02x}",
    );
    assert!(
        (0x40..=0xC8).contains(&local),
        "local viewer: tilted V dims the off-axis highlight, got 0x{local:02x}",
    );
}

#[test]
fn texture_arg_specular_selects_vertex_color1() {
    // D3DTA_SPECULAR routes the interpolated specular color (oD1) into the
    // stage cascade: SELECTARG1 on it must render the vertex COLOR1.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_SPECULAR),
        0,
        "SetFVF"
    );
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_SPECULAR),
        0,
        "COLORARG1 = SPECULAR",
    );

    let tri = specular_triangle(0xFF00_FF00, 0xFFFF_0000);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFFFF_0000,
        "stage selects vertex specular red"
    );
}

#[test]
fn sparse_light_indices_round_trip() {
    // D3D9 lets SetLight / LightEnable address light indices beyond
    // MaxActiveLights — that cap bounds only how many lights contribute to a
    // single draw, not the addressable range. Every slot up to MaxActiveLights —
    // and one past it — must round-trip. Indices at or above the 8 fast-path
    // slots take the sparse overflow store.
    let h = Harness::new();
    let max = h.device_caps().max_active_lights;

    // Enable each light up to the advertised maximum, then one beyond it.
    for i in 1..=(max + 1) {
        assert_eq!(h.light_enable(i, true), 0, "LightEnable({i}, true)");
        assert!(h.light_enabled(i), "light {i} reads back enabled");
    }

    // A SetLight at a high sparse index round-trips through GetLight.
    let high = max + 5;
    let light = D3DLIGHT9 {
        type_: D3DLIGHT_POINT,
        range: 25.0,
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(high, &light), 0, "SetLight(high)");
    let got = h.light(high);
    assert_eq!(
        got.type_, D3DLIGHT_POINT,
        "high-index light type round-trip"
    );
    assert_eq!(
        got.range.to_bits(),
        25.0_f32.to_bits(),
        "high-index light range round-trip"
    );

    // Disabling a high sparse light sticks.
    assert_eq!(h.light_enable(high, false), 0, "LightEnable(high, false)");
    assert!(
        !h.light_enabled(high),
        "high-index light reads back disabled"
    );
}

/// `D3DTSS_TCI_CAMERASPACEPOSITION`.
///
/// The texgen mode occupies bits 16..23 of `D3DTSS_TEXCOORDINDEX`.
const TCI_CAMERASPACEPOSITION: u32 = 2 << 16;

/// Grey level of the lit rows' ambient plus emissive sum, as a channel value.
const TEXGEN_AMBIENT_LEVEL: u32 = 0x80;
/// The same with the directional light's full N.L diffuse term added.
const TEXGEN_DIFFUSE_LEVEL: u32 = 0xBF;
/// Unlit geometry without a vertex colour reads opaque white.
const TEXGEN_UNLIT_LEVEL: u32 = 0xFF;

/// A full-viewport quad at z = 0.5 as two triangles, positions only.
const TEXGEN_CORNERS: [(f32, f32); 6] = [
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (1.0, 1.0),
];

/// Arm stage 0 to modulate a 2x2 texture, addressed by eye-space position, by the diffuse colour.
///
/// Every transform is the identity, so the generated coordinate is the vertex
/// position: `u = x`, `v = y`, both spanning -1..1 over the viewport. WRAP
/// addressing with POINT filtering turns that into quarter-viewport bands
/// alternating between the two texel columns and the two texel rows. The
/// texture is (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
///
/// The lighting inputs are chosen so each term is a distinct grey: global
/// ambient white over material ambient 0.25, emissive 0.25, and a directional
/// light along +z whose white diffuse meets material diffuse 0.25. A vertex
/// without a normal takes ambient plus emissive (0.5) and no N.L term; a
/// vertex facing the light adds the full diffuse (0.75).
fn arm_eye_position_texgen(h: &Harness) -> Texture<'_> {
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    for (state, value) in [
        (D3DRS_AMBIENT, 0xFFFF_FFFF),
        (D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_MATERIAL),
        (D3DRS_AMBIENTMATERIALSOURCE, D3DMCS_MATERIAL),
        (D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_MATERIAL),
    ] {
        assert_eq!(h.set_render_state(state, value), 0, "SetRenderState");
    }
    let grey = D3DCOLORVALUE {
        r: 0.25,
        g: 0.25,
        b: 0.25,
        a: 1.0,
    };
    let material = D3DMATERIAL9 {
        diffuse: grey,
        ambient: grey,
        specular: D3DCOLORVALUE::default(),
        emissive: grey,
        power: 0.0,
    };
    assert_eq!(h.set_material(&material), 0, "SetMaterial");
    let light = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        diffuse: D3DCOLORVALUE {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        direction: D3DVECTOR {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight");
    assert_eq!(h.light_enable(0, true), 0, "LightEnable");

    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0)
        .write_u32(&[0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF]);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_MODULATE),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_COLORARG2, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        (D3DTSS_TEXCOORDINDEX, TCI_CAMERASPACEPOSITION),
    ] {
        assert_eq!(
            h.set_texture_stage_state(0, state, value),
            0,
            "SetTextureStageState"
        );
    }
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "SetSamplerState");
    }
    tex
}

/// Probe the centre of one band per texel, plus one band right of the origin.
///
/// `level` is the grey the diffuse colour contributes, so each probe expects
/// its texel scaled by it. A draw Metal rejected leaves the clear colour, and
/// a stage that fell back to the absent vertex coordinate reads the (0,0)
/// texel everywhere.
fn assert_eye_position_texgen(h: &Harness, level: u32, context: &str) {
    let grey = |mask: u32| 0xFF00_0000 | ((level * 0x0001_0101) & mask);
    for (x, y, mask, texel) in [
        (80, 180, 0x00FF_0000, "red (0,0) at x=-0.75 y=0.25"),
        (240, 180, 0x0000_FF00, "green (1,0) at x=-0.25 y=0.25"),
        (80, 60, 0x0000_00FF, "blue (0,1) at x=-0.75 y=0.75"),
        (240, 60, 0x00FF_FFFF, "white (1,1) at x=-0.25 y=0.75"),
        (560, 420, 0x0000_FF00, "green (1,0) at x=0.75 y=-0.75"),
    ] {
        assert_pixel_approx(
            h.read_pixel(x, y),
            grey(mask),
            2,
            &format!("{context}: {texel}"),
        );
    }
}

#[test]
fn texgen_cameraspaceposition_lit_without_normal() {
    // Lighting declares the eye-space position for its own use whether or not
    // the vertex has a normal, and the texgen stage reads that same value.
    // Without a normal the light's N.L term is dropped, so the texel is
    // modulated by ambient plus emissive alone.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF (no normal)");
    let _tex = arm_eye_position_texgen(&h);
    let quad = TEXGEN_CORNERS.map(|(x, y)| PosVertex { x, y, z: 0.5 });
    h.render_once(0xFFFF_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    assert_eye_position_texgen(&h, TEXGEN_AMBIENT_LEVEL, "lit, no normal");
}

#[test]
fn texgen_cameraspaceposition_lit_with_normal() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    let _tex = arm_eye_position_texgen(&h);
    let quad = TEXGEN_CORNERS.map(|(x, y)| LitVertex {
        x,
        y,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: -1.0,
    });
    h.render_once(0xFFFF_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    assert_eye_position_texgen(&h, TEXGEN_DIFFUSE_LEVEL, "lit, normal");
}

#[test]
fn texgen_cameraspaceposition_unlit() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF (no normal)");
    let _tex = arm_eye_position_texgen(&h);
    let quad = TEXGEN_CORNERS.map(|(x, y)| PosVertex { x, y, z: 0.5 });
    h.render_once(0xFFFF_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    assert_eye_position_texgen(&h, TEXGEN_UNLIT_LEVEL, "unlit, no normal");
}

/// `D3DTSS_TCI_SPHEREMAP`, in the same byte of `D3DTSS_TEXCOORDINDEX`.
const TCI_SPHEREMAP: u32 = 4 << 16;

/// The colour of texel (`col`, `row`) of the 4x4 sphere-map texture.
///
/// Red encodes the column and green the row in steps of 0x55, so a probe
/// names the texel it read and a swapped or negated axis lands on a different
/// colour. Blue is a constant 0x40, which keeps every texel apart from the
/// magenta clear colour.
const fn sphere_texel(col: u32, row: u32) -> u32 {
    0xFF00_0040 | ((col * 0x55) << 16) | ((row * 0x55) << 8)
}

/// Arm stage 0 to show a 4x4 texture addressed by the sphere map, unmodulated.
///
/// `view` and `projection` place the quad: the sphere map depends on the
/// direction from the eye to the vertex, so the tests move the geometry far
/// from the eye in view space, which makes that direction the same for every
/// vertex to within a hundredth, and undo the move in the projection.
fn arm_sphere_map<'h>(h: &'h Harness, view: &[f32; 16], projection: &[f32; 16]) -> Texture<'h> {
    assert_eq!(h.set_transform(D3DTS_WORLD, &IDENTITY), 0, "world");
    assert_eq!(h.set_transform(D3DTS_VIEW, view), 0, "view");
    assert_eq!(h.set_transform(D3DTS_PROJECTION, projection), 0, "proj");
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, 0);
    let mut texels = [0u32; 16];
    for row in 0..4u32 {
        for col in 0..4u32 {
            texels[(row * 4 + col) as usize] = sphere_texel(col, row);
        }
    }
    tex.lock_rect(0, 0).write_u32(&texels);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        (D3DTSS_TEXCOORDINDEX, TCI_SPHEREMAP),
    ] {
        assert_eq!(
            h.set_texture_stage_state(0, state, value),
            0,
            "SetTextureStageState"
        );
    }
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "SetSamplerState");
    }
    tex
}

/// View matrix that moves the geometry 100 units down the view axis.
#[rustfmt::skip]
const SPHERE_VIEW_AXIAL: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 100.0, 1.0,
];

/// Projection that brings eye-space z = 100 back to depth 0.5.
#[rustfmt::skip]
const SPHERE_PROJ_AXIAL: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.005, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Draw one quad per viewport quadrant, each with its own normal, under the sphere map.
///
/// The quadrant with signs (`sx`, `sy`) carries the unit normal
/// (0.576 `sx`, 0.168 `sy`, -0.8). With the eye-to-vertex direction
/// E = (0, 0, 1): N.E = -0.8, R = E - 2 (N.E) N = E + 1.6 N
/// = (0.9216 `sx`, 0.2688 `sy`, -0.28), R + (0, 0, 1) has length
/// sqrt(0.84935 + 0.07225 + 0.5184) = 1.2, so m = 2.4 and
/// (u, v) = (0.5 + 0.384 `sx`, 0.5 + 0.112 `sy`): 0.884 or 0.116 across,
/// 0.612 or 0.388 down. The x and y components differ so that a swapped axis
/// moves the coordinate to another texel. E is off the axis by at most 0.01
/// at the outer corners, which moves a coordinate by less than 0.006, and
/// every pixel interpolates between vertices that all lie in one texel.
fn draw_sphere_map_quadrants(h: &Harness) {
    let mut vertices = Vec::with_capacity(24);
    for (sx, sy) in [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        for (cx, cy) in TEXGEN_CORNERS {
            // Map the corner from -1..1 onto the quadrant without mirroring
            // it, so every quad keeps the clockwise winding.
            vertices.push(LitVertex {
                x: f32::midpoint(sx, cx),
                y: f32::midpoint(sy, cy),
                z: 0.0,
                nx: 0.576 * sx,
                ny: 0.168 * sy,
                nz: -0.8,
            });
        }
    }
    h.render_once(0xFFFF_00FF, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 8, &vertices),
            0,
            "draw"
        );
    });
}

/// Probe the centre of each quadrant for the texel (`col`, `row`) it must show.
///
/// The array is ordered right-top, right-bottom, left-top, left-bottom, the
/// order [`draw_sphere_map_quadrants`] draws in.
fn assert_sphere_map_quadrants(h: &Harness, expected: [(u32, u32); 4], context: &str) {
    for ((x, y), (col, row)) in [(480, 120), (480, 360), (160, 120), (160, 360)]
        .into_iter()
        .zip(expected)
    {
        assert_pixel_approx(
            h.read_pixel(x, y),
            sphere_texel(col, row),
            2,
            &format!("{context}: texel ({col}, {row}) at ({x}, {y})"),
        );
    }
}

/// Texels the untransformed sphere map selects: column 3 or 0 from x, row 2 or 1 from y.
const SPHERE_QUADRANT_TEXELS: [(u32, u32); 4] = [(3, 2), (3, 1), (0, 2), (0, 1)];

#[test]
fn texgen_spheremap_selects_the_texel_by_normal() {
    // The FVF carries no texture coordinate, so a stage that passed its input
    // through reads texel (0, 0) everywhere.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    let _tex = arm_sphere_map(&h, &SPHERE_VIEW_AXIAL, &SPHERE_PROJ_AXIAL);
    draw_sphere_map_quadrants(&h);
    assert_sphere_map_quadrants(&h, SPHERE_QUADRANT_TEXELS, "unlit");
}

#[test]
fn texgen_spheremap_lit_reads_the_lighting_normal() {
    // With lighting on the sphere map reads the eye-space normal and position
    // the lighting computation declares. The view is a pure translation, so
    // that normal is the vertex normal and the texels are the unlit ones.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "lighting on");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    let _tex = arm_sphere_map(&h, &SPHERE_VIEW_AXIAL, &SPHERE_PROJ_AXIAL);
    draw_sphere_map_quadrants(&h);
    assert_sphere_map_quadrants(&h, SPHERE_QUADRANT_TEXELS, "lit");
}

#[test]
fn texgen_spheremap_texture_transform_applies_after_generation() {
    // COUNT2 multiplies the generated (u, v, 0, 1) by the stage matrix, here
    // u' = v + 0.25 and v' = u. The quarter offset rides on the fourth
    // component, which only a generated coordinate of dimension 3 pads to 1.
    // Right-top: (0.612 + 0.25, 0.884) = (0.862, 0.884), texel (3, 3);
    // right-bottom: (0.638, 0.884), texel (2, 3); left-top: (0.862, 0.116),
    // texel (3, 0); left-bottom: (0.638, 0.116), texel (2, 0).
    #[rustfmt::skip]
    const SWAP_AND_SHIFT: [f32; 16] = [
        0.0,  1.0, 0.0, 0.0,
        1.0,  0.0, 0.0, 0.0,
        0.0,  0.0, 1.0, 0.0,
        0.25, 0.0, 0.0, 1.0,
    ];
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0, "SetFVF");
    let _tex = arm_sphere_map(&h, &SPHERE_VIEW_AXIAL, &SPHERE_PROJ_AXIAL);
    assert_eq!(h.set_transform(D3DTS_TEXTURE0, &SWAP_AND_SHIFT), 0, "tex");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_COUNT2),
        0,
        "COUNT2"
    );
    draw_sphere_map_quadrants(&h);
    assert_sphere_map_quadrants(&h, [(3, 3), (2, 3), (3, 0), (2, 0)], "transformed");
}

#[test]
fn texgen_spheremap_without_normal_maps_the_view_direction() {
    // A vertex without a normal reads a zero normal, so R is the eye-to-vertex
    // direction E. The view moves the quad to (50, -50, 80) and the projection
    // moves it back: E = (5, -5, 8) / sqrt(114) = (0.4683, -0.4683, 0.7493),
    // m = 2 sqrt(2 + 2 * 0.7493) = 3.7409, (u, v) = (0.6252, 0.3748), texel
    // (2, 1) over the whole quad. The FVF has no texture coordinate, so a
    // stage that fell back to its input reads texel (0, 0).
    #[rustfmt::skip]
    const VIEW: [f32; 16] = [
        1.0,   0.0,  0.0,  0.0,
        0.0,   1.0,  0.0,  0.0,
        0.0,   0.0,  1.0,  0.0,
        50.0, -50.0, 80.0, 1.0,
    ];
    #[rustfmt::skip]
    const PROJECTION: [f32; 16] = [
        1.0,   0.0,  0.0,     0.0,
        0.0,   1.0,  0.0,     0.0,
        0.0,   0.0,  0.00625, 0.0,
        -50.0, 50.0, 0.0,     1.0,
    ];
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF (no normal)");
    let _tex = arm_sphere_map(&h, &VIEW, &PROJECTION);
    let quad = TEXGEN_CORNERS.map(|(x, y)| PosVertex { x, y, z: 0.0 });
    h.render_once(0xFFFF_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    for (x, y) in [(320, 240), (80, 60), (560, 420)] {
        assert_pixel_approx(
            h.read_pixel(x, y),
            sphere_texel(2, 1),
            2,
            &format!("no normal: texel (2, 1) at ({x}, {y})"),
        );
    }
}
