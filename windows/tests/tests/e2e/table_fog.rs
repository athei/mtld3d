//! Table fog formulas, shader eligibility and transitions between fog sources.

use mtld3d_tests::{Harness, HarnessConfig, PosColorVertex, assert_pixel_approx};
use mtld3d_types::{
    D3DCMP_GREATER, D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFOG_EXP, D3DFOG_EXP2, D3DFOG_LINEAR,
    D3DFOG_NONE, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DFVF_XYZW, D3DMATRIX,
    D3DPT_TRIANGLESTRIP, D3DRECT, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE,
    D3DRS_CULLMODE, D3DRS_DEPTHBIAS, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY, D3DRS_FOGENABLE,
    D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE, D3DRS_LIGHTING,
    D3DRS_RANGEFOGENABLE, D3DRS_SCISSORTESTENABLE, D3DRS_ZENABLE, D3DSBT_ALL, D3DSBT_PIXELSTATE,
    D3DTS_PROJECTION, D3DVIEWPORT9,
};

const RED: u32 = 0x80ff_0000;
const GREEN: u32 = 0x8000_ff00;
const BLACK: u32 = 0xff00_0000;
// Position is supplied as FLOAT4; oFog=.25 deliberately differs from table Z/W.
const VS1: &[u32] = &[
    0xfffe_0101,
    0x0000_001f,
    0x8000_0000,
    0x900f_0000,
    1,
    0xc00f_0000,
    0x90e4_0000,
    1,
    0xd00f_0000,
    0xa0e4_0000,
    1,
    0xc001_0001,
    0xa000_0001,
    0xffff,
];
const VS2: &[u32] = &[
    0xfffe_0200,
    0x0200_001f,
    0x8000_0000,
    0x900f_0000,
    0x0200_0001,
    0xc00f_0000,
    0x90e4_0000,
    0x0200_0001,
    0xd00f_0000,
    0xa0e4_0000,
    0x0200_0001,
    0xc001_0001,
    0xa000_0001,
    0xffff,
];
const VS3: &[u32] = &[
    0xfffe_0300,
    0x0200_001f,
    0x8000_0000,
    0x900f_0000,
    0x0200_001f,
    0x8000_0000,
    0xe00f_0000,
    0x0200_001f,
    0x8000_000a,
    0xe00f_0001,
    0x0200_0001,
    0xe00f_0000,
    0x90e4_0000,
    0x0200_0001,
    0xe00f_0001,
    0xa0e4_0000,
    0xffff,
];

const fn pixel_shader_code(version: u32) -> [u32; 5] {
    if version == 1 {
        [0xffff_0101, 1, 0x800f_0000, 0xa0e4_0000, 0xffff]
    } else {
        [
            0xffff_0000 | (version << 8),
            0x0200_0001,
            0x800f_0800,
            0xa0e4_0000,
            0xffff,
        ]
    }
}

#[repr(C)]
struct Position4Vertex {
    position: [f32; 4],
    color: u32,
}

fn vertices() -> [PosColorVertex; 4] {
    [(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)].map(|(x, y)| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: RED,
    })
}

fn clip_vertices() -> [Position4Vertex; 4] {
    [(-2.0, 2.0), (-2.0, -2.0), (2.0, 2.0), (2.0, -2.0)].map(|(x, y)| Position4Vertex {
        position: [x, y, 1.0, 2.0],
        color: RED,
    })
}

fn set(h: &Harness, state: u32, value: u32) {
    assert_eq!(h.set_render_state(state, value), 0);
}

fn harness() -> Harness {
    let h = Harness::create(&HarnessConfig {
        width: 128,
        height: 128,
        ..HarnessConfig::default()
    });
    h.select_diffuse_stage(0);
    for (state, value) in [
        (D3DRS_LIGHTING, 0),
        (D3DRS_CULLMODE, D3DCULL_NONE),
        (D3DRS_ZENABLE, 0),
        (D3DRS_FOGENABLE, 1),
        (D3DRS_FOGCOLOR, 0x0000_ff00),
        (D3DRS_FOGTABLEMODE, D3DFOG_LINEAR),
        (D3DRS_FOGVERTEXMODE, D3DFOG_LINEAR),
        (D3DRS_FOGSTART, 0.0_f32.to_bits()),
        (D3DRS_FOGEND, 4.0_f32.to_bits()),
        (D3DRS_FOGDENSITY, 0.5_f32.to_bits()),
    ] {
        set(&h, state, value);
    }
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[1.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0]),
        0
    );
    assert_eq!(h.set_pixel_shader_constant_f(0, &[1.0, 0.0, 0.0, 0.5]), 0);
    h
}

fn draw<V>(h: &Harness, vertices: &[V]) {
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, vertices), 0);
}

fn render<V>(h: &Harness, vertices: &[V]) -> u32 {
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    draw(h, vertices);
    assert_eq!(h.end_scene(), 0);
    h.read_pixel(64, 64)
}

#[test]
fn table_fog_formulas_use_z_or_w_and_preserve_alpha() {
    let h = harness();
    let target = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &target), 0);
    for code in [None, Some(VS1), Some(VS2)] {
        let shader = code.map(|code| h.create_vertex_shader(code));
        let pixel_code = pixel_shader_code(if code == Some(VS1) { 1 } else { 2 });
        let pixel = shader.as_ref().map(|_| h.create_pixel_shader(&pixel_code));
        if let (Some(shader), Some(pixel)) = (&shader, &pixel) {
            assert_eq!(h.set_vertex_shader(shader), 0);
            assert_eq!(h.set_pixel_shader(pixel), 0);
        } else {
            assert_eq!(h.clear_vertex_shader(), 0);
            assert_eq!(h.clear_pixel_shader(), 0);
        }
        for perspective in [false, true] {
            let mut projection = D3DMATRIX::IDENTITY.m;
            if perspective {
                for diagonal in [0, 5, 10, 15] {
                    projection[diagonal] = 2.0;
                }
            }
            assert_eq!(h.set_transform(D3DTS_PROJECTION, &projection), 0);
            for (mode, z_color, w_color) in [
                (D3DFOG_LINEAR, 0x80df_2000, 0x8080_8000),
                (D3DFOG_EXP, 0x80c7_3800, 0x805e_a100),
                (D3DFOG_EXP2, 0x80f0_0f00, 0x805e_a100),
            ] {
                set(&h, D3DRS_FOGTABLEMODE, mode);
                for range in [0, 1] {
                    set(&h, D3DRS_RANGEFOGENABLE, range);
                    for enabled in [1, 0, 1] {
                        set(&h, D3DRS_FOGENABLE, enabled);
                        let expected = if enabled == 0 {
                            RED
                        } else if perspective {
                            w_color
                        } else {
                            z_color
                        };
                        let actual = if shader.is_some() {
                            assert_eq!(h.set_fvf(D3DFVF_XYZW | D3DFVF_DIFFUSE), 0);
                            render(&h, &clip_vertices())
                        } else {
                            assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
                            render(&h, &vertices())
                        };
                        assert_pixel_approx(actual, expected, 1, "table fog formula");
                        if shader.is_none() {
                            assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0);
                            let pretransformed =
                                [(0.0, 0.0), (0.0, 128.0), (128.0, 0.0), (128.0, 128.0)].map(
                                    |(x, y)| Position4Vertex {
                                        position: [x, y, 0.5, 0.5],
                                        color: RED,
                                    },
                                );
                            assert_pixel_approx(
                                render(&h, &pretransformed),
                                expected,
                                1,
                                "RHW table fog",
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn table_fog_is_shader_owned_in_sm3_and_legacy_binding_restores_it() {
    let h = harness();
    let target = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZW | D3DFVF_DIFFUSE), 0);
    let older = h.create_vertex_shader(VS2);
    let newer = h.create_vertex_shader(VS3);
    let older_pixel = h.create_pixel_shader(&pixel_shader_code(2));
    let newer_pixel = h.create_pixel_shader(&pixel_shader_code(3));
    assert_eq!(h.set_vertex_shader(&newer), 0);
    assert_eq!(h.set_pixel_shader(&newer_pixel), 0);
    for table in 0..=3 {
        set(&h, D3DRS_FOGTABLEMODE, table);
        for vertex in 0..=3 {
            set(&h, D3DRS_FOGVERTEXMODE, vertex);
            assert_pixel_approx(render(&h, &clip_vertices()), RED, 1, "SM3 owns fog");
        }
    }
    set(&h, D3DRS_FOGTABLEMODE, D3DFOG_LINEAR);
    set(&h, D3DRS_SCISSORTESTENABLE, 1);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    for (left, right, vertex, pixel) in [
        (0, 40, &older, &older_pixel),
        (40, 80, &newer, &newer_pixel),
        (80, 128, &older, &older_pixel),
    ] {
        assert_eq!(h.set_vertex_shader(vertex), 0);
        assert_eq!(h.set_pixel_shader(pixel), 0);
        assert_eq!(
            h.set_scissor_rect(&D3DRECT {
                x1: left,
                y1: 0,
                x2: right,
                y2: 128
            }),
            0
        );
        draw(&h, &clip_vertices());
    }
    assert_eq!(h.end_scene(), 0);
    for (x, expected) in [(20, 0x80df_2000), (60, RED), (100, 0x80df_2000)] {
        assert_pixel_approx(
            h.read_pixel(x, 64),
            expected,
            1,
            "same-scene shader transition",
        );
    }
    set(&h, D3DRS_SCISSORTESTENABLE, 0);
    for kind in [D3DSBT_ALL, D3DSBT_PIXELSTATE] {
        let block = h.create_state_block(kind);
        assert_eq!(h.set_vertex_shader(&newer), 0);
        assert_eq!(h.set_pixel_shader(&newer_pixel), 0);
        assert_pixel_approx(render(&h, &clip_vertices()), RED, 1, "SM3 before restore");
        // A pixel-only block does not restore VS; keep the drawn pair valid.
        assert_eq!(h.set_vertex_shader(&older), 0);
        assert_eq!(block.apply(), 0);
        assert_pixel_approx(
            render(&h, &clip_vertices()),
            0x80df_2000,
            1,
            "snapshot restores legacy fog",
        );
    }
    assert_eq!(h.begin_state_block(), 0);
    assert_eq!(h.set_vertex_shader(&newer), 0);
    assert_eq!(h.set_pixel_shader(&newer_pixel), 0);
    let recorded = h.end_state_block();
    assert_eq!(recorded.apply(), 0);
    assert_pixel_approx(
        render(&h, &clip_vertices()),
        RED,
        1,
        "recorded block selects SM3",
    );
    assert_eq!(h.set_vertex_shader(&older), 0);
    assert_eq!(h.set_pixel_shader(&older_pixel), 0);
    assert_pixel_approx(
        render(&h, &clip_vertices()),
        0x80df_2000,
        1,
        "legacy fog after recorded block",
    );
}

#[test]
fn table_fog_projection_range_and_alpha_state_changes_reach_the_draw() {
    let h = harness();
    let target = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    set(&h, D3DRS_FOGSTART, 1.0_f32.to_bits());
    set(&h, D3DRS_FOGEND, 1.2_f32.to_bits());
    set(&h, D3DRS_RANGEFOGENABLE, 1);
    assert_pixel_approx(render(&h, &vertices()), RED, 1, "table overrides range");
    set(&h, D3DRS_FOGTABLEMODE, D3DFOG_NONE);
    assert_pixel_approx(render(&h, &vertices()), GREEN, 1, "range restored");
    set(&h, D3DRS_RANGEFOGENABLE, 0);
    assert_pixel_approx(
        render(&h, &vertices()),
        RED,
        1,
        "ordinary vertex fog restored",
    );
    set(&h, D3DRS_FOGTABLEMODE, D3DFOG_LINEAR);
    set(&h, D3DRS_FOGSTART, 0.0_f32.to_bits());
    set(&h, D3DRS_FOGEND, 4.0_f32.to_bits());
    for alpha in [0, 0xff00_0000] {
        set(&h, D3DRS_FOGCOLOR, alpha | 0x0000_ff00);
        assert_pixel_approx(
            render(&h, &vertices()),
            0x80df_2000,
            1,
            "fog color alpha ignored",
        );
    }
    set(&h, D3DRS_ALPHATESTENABLE, 1);
    set(&h, D3DRS_ALPHAFUNC, D3DCMP_GREATER);
    set(&h, D3DRS_ALPHAREF, 127);
    assert_pixel_approx(
        render(&h, &vertices()),
        0x80df_2000,
        1,
        "fog preserves alpha-test input",
    );
    set(&h, D3DRS_ALPHAREF, 129);
    assert_eq!(render(&h, &vertices()), BLACK);
    set(&h, D3DRS_ALPHATESTENABLE, 0);
    assert_eq!(
        h.set_viewport(&D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: 128,
            height: 128,
            min_z: 0.2,
            max_z: 0.6
        }),
        0
    );
    assert_pixel_approx(
        render(&h, &vertices()),
        0x80df_2000,
        1,
        "Z before viewport depth mapping",
    );
    set(&h, D3DRS_DEPTHBIAS, 0.1_f32.to_bits());
    assert_pixel_approx(
        render(&h, &vertices()),
        0x80d9_2600,
        1,
        "raw bias contributes to Z fog",
    );
}

#[test]
fn table_fog_exponential_factor_is_computed_per_pixel() {
    let h = harness();
    let target = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    set(&h, D3DRS_FOGTABLEMODE, D3DFOG_EXP);
    set(&h, D3DRS_FOGDENSITY, 2.0_f32.to_bits());
    let gradient = [
        (-1.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
    ]
    .map(|(x, y, z)| PosColorVertex {
        x,
        y,
        z,
        color: RED,
    });
    // exp(-2*.5) is about .368. Interpolating endpoint factors instead
    // would give .568, a distinct red component near 145 instead of 94.
    assert_pixel_approx(
        render(&h, &gradient),
        0x805e_a100,
        2,
        "per-pixel exponential fog",
    );
}
