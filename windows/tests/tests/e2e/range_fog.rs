//! Range-based vertex fog, its eye-space source and inactive controls.

use mtld3d_tests::{Harness, HarnessConfig, PosColorVertex, assert_pixel_approx};
use mtld3d_types::{
    D3DCULL_NONE, D3DFOG_EXP, D3DFOG_EXP2, D3DFOG_LINEAR, D3DFOG_NONE, D3DFVF_DIFFUSE,
    D3DFVF_LASTBETA_UBYTE4, D3DFVF_SPECULAR, D3DFVF_XYZ, D3DFVF_XYZB1, D3DFVF_XYZB2, D3DFVF_XYZRHW,
    D3DMATRIX, D3DPT_TRIANGLESTRIP, D3DRECT, D3DRS_CULLMODE, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_RANGEFOGENABLE, D3DRS_SCISSORTESTENABLE,
    D3DRS_VERTEXBLEND, D3DRS_ZENABLE, D3DSBT_ALL, D3DSBT_VERTEXSTATE, D3DTS_VIEW, D3DTS_WORLD,
    D3DVBF_1WEIGHTS,
};

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLACK: u32 = 0xFF00_0000;

fn harness() -> Harness {
    let h = Harness::create(&HarnessConfig {
        width: 128,
        height: 128,
        ..HarnessConfig::default()
    });
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    h.select_diffuse_stage(0);
    for (state, value) in [
        (D3DRS_LIGHTING, 0),
        (D3DRS_CULLMODE, D3DCULL_NONE),
        (D3DRS_ZENABLE, 0),
        (D3DRS_FOGENABLE, 1),
        (D3DRS_FOGCOLOR, GREEN),
        (D3DRS_FOGTABLEMODE, D3DFOG_NONE),
        (D3DRS_FOGVERTEXMODE, D3DFOG_LINEAR),
        (D3DRS_FOGSTART, 0.75_f32.to_bits()),
        (D3DRS_FOGEND, 0.8_f32.to_bits()),
        (D3DRS_FOGDENSITY, 1.0_f32.to_bits()),
    ] {
        assert_eq!(h.set_render_state(state, value), 0);
    }
    h
}

fn vertices() -> [PosColorVertex; 4] {
    [(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)].map(|(x, y)| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: RED,
    })
}

fn draw<V>(h: &Harness, vertices: &[V]) {
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, vertices), 0);
}

#[test]
fn range_fog_uses_distance_in_all_three_formulas() {
    let h = harness();
    // Every vertex is 1.5 from the eye, but only 0.5 along Z. Identical
    // factors at all four corners also distinguish per-vertex from pixel fog.
    for (mode, ordinary, ranged) in [
        (D3DFOG_LINEAR, RED, GREEN),
        (D3DFOG_EXP, 0xFF9B_6400, 0xFF39_C600),
        (D3DFOG_EXP2, 0xFFC7_3800, 0xFF1B_E400),
    ] {
        assert_eq!(h.set_render_state(D3DRS_FOGVERTEXMODE, mode), 0);
        for (range, expected) in [(0, ordinary), (1, ranged), (0, ordinary)] {
            assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, range), 0);
            h.render_once(BLACK, |d| draw(d, &vertices()));
            assert_pixel_approx(h.read_pixel(64, 64), expected, 1, "vertex fog factor");
        }
    }
}

#[test]
fn range_fog_toggles_within_a_scene_and_stateblocks_restore_it() {
    let h = harness();
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(d.set_render_state(D3DRS_RANGEFOGENABLE, 0), 0);
        assert_eq!(
            d.set_scissor_rect(&D3DRECT {
                x1: 0,
                y1: 0,
                x2: 64,
                y2: 128
            }),
            0
        );
        draw(d, &vertices());
        assert_eq!(d.set_render_state(D3DRS_RANGEFOGENABLE, 1), 0);
        assert_eq!(
            d.set_scissor_rect(&D3DRECT {
                x1: 64,
                y1: 0,
                x2: 128,
                y2: 128
            }),
            0
        );
        draw(d, &vertices());
    });
    assert_eq!(h.read_pixel(32, 64), RED);
    assert_eq!(h.read_pixel(96, 64), GREEN);
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 0), 0);
    for kind in [D3DSBT_ALL, D3DSBT_VERTEXSTATE] {
        assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, 1), 0);
        let block = h.create_state_block(kind);
        assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, 0), 0);
        assert_eq!(block.apply(), 0);
        h.render_once(BLACK, |d| draw(d, &vertices()));
        assert_eq!(h.read_pixel(64, 64), GREEN);
    }
}

#[repr(C)]
struct BlendedVertex {
    position: [f32; 3],
    weight: f32,
    color: u32,
}
#[repr(C)]
struct IndexedVertex {
    position: [f32; 3],
    weight: f32,
    indices: [u8; 4],
    color: u32,
}

#[test]
fn range_fog_uses_transformed_and_blended_eye_position() {
    let h = harness();
    assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_FOGSTART, 1.55_f32.to_bits()), 0);
    assert_eq!(h.set_render_state(D3DRS_FOGEND, 1.58_f32.to_bits()), 0);
    let mut view = D3DMATRIX::IDENTITY.m;
    assert_eq!(h.set_transform(D3DTS_VIEW, &view), 0);
    let scaled = |scale: f32| {
        let mut matrix = D3DMATRIX::IDENTITY.m;
        matrix[0] = scale;
        matrix[5] = scale;
        matrix[10] = scale;
        matrix
    };
    let small = vertices().map(|v| PosColorVertex {
        x: v.x * 0.5,
        y: v.y * 0.5,
        z: v.z * 0.5,
        color: RED,
    });
    assert_eq!(h.set_transform(D3DTS_WORLD, &scaled(2.0)), 0);
    // World-space length is 1.5. The view translation raises the eye-space
    // length to sqrt(2.5625), about 1.60, crossing both fog thresholds.
    h.render_once(BLACK, |d| draw(d, &small));
    assert_eq!(
        h.read_pixel(64, 64),
        RED,
        "world position before view translation"
    );
    view[14] = 0.25;
    assert_eq!(h.set_transform(D3DTS_VIEW, &view), 0);
    h.render_once(BLACK, |d| draw(d, &small));
    assert_eq!(h.read_pixel(64, 64), GREEN, "world/view before fog");

    // Weighted world1/world2 (or world0/world1) produces the same scale2
    // position. Object position or an unblended world0 would remain red.
    assert_eq!(h.set_render_state(D3DRS_VERTEXBLEND, D3DVBF_1WEIGHTS), 0);
    assert_eq!(h.set_transform(D3DTS_WORLD, &scaled(1.0)), 0);
    assert_eq!(h.set_transform(D3DTS_WORLD + 1, &scaled(3.0)), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZB1 | D3DFVF_DIFFUSE), 0);
    let blended = small.map(|v| BlendedVertex {
        position: [v.x, v.y, v.z],
        weight: 0.5,
        color: RED,
    });
    h.render_once(BLACK, |d| draw(d, &blended));
    assert_eq!(
        h.read_pixel(64, 64),
        GREEN,
        "sequential blending before fog"
    );

    assert_eq!(h.set_transform(D3DTS_WORLD + 1, &scaled(1.0)), 0);
    assert_eq!(h.set_transform(D3DTS_WORLD + 2, &scaled(3.0)), 0);
    assert_eq!(h.set_render_state(D3DRS_INDEXEDVERTEXBLENDENABLE, 1), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZB2 | D3DFVF_LASTBETA_UBYTE4 | D3DFVF_DIFFUSE),
        0
    );
    let indexed = small.map(|v| IndexedVertex {
        position: [v.x, v.y, v.z],
        weight: 0.5,
        indices: [1, 2, 0, 0],
        color: RED,
    });
    h.render_once(BLACK, |d| draw(d, &indexed));
    assert_eq!(h.read_pixel(64, 64), GREEN, "indexed blending before fog");
}

#[repr(C)]
struct SuppliedFogVertex {
    position: [f32; 4],
    diffuse: u32,
    specular: u32,
}

#[test]
fn range_fog_leaves_disabled_table_and_supplied_fog_unchanged() {
    let h = harness();
    for range in [0, 1] {
        assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, range), 0);
        assert_eq!(h.set_render_state(D3DRS_FOGENABLE, 0), 0);
        h.render_once(BLACK, |d| draw(d, &vertices()));
        assert_eq!(h.read_pixel(64, 64), RED);
        assert_eq!(h.set_render_state(D3DRS_FOGENABLE, 1), 0);
        assert_eq!(h.set_render_state(D3DRS_FOGTABLEMODE, D3DFOG_LINEAR), 0);
        h.render_once(BLACK, |d| draw(d, &vertices()));
        assert_eq!(h.read_pixel(64, 64), RED, "table fog ignores range flag");
        assert_eq!(h.set_render_state(D3DRS_FOGTABLEMODE, D3DFOG_NONE), 0);
    }
    let rhw =
        [(0.0, 0.0), (0.0, 128.0), (128.0, 0.0), (128.0, 128.0)].map(|(x, y)| SuppliedFogVertex {
            position: [x, y, 0.5, 1.0],
            diffuse: RED,
            specular: 0x4000_0000,
        });
    assert_eq!(
        h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_SPECULAR),
        0
    );
    for range in [0, 1] {
        assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, range), 0);
        h.render_once(BLACK, |d| draw(d, &rhw));
        assert_pixel_approx(
            h.read_pixel(64, 64),
            0xFF40_BF00,
            1,
            "RHW supplies specular alpha",
        );
    }
}

#[test]
fn programmable_vertex_fog_ignores_range_state() {
    let h = harness();
    // vs_2_0: dcl_position v0; mov oPos,v0; mov oD0,c0; mov oFog,c1.x.
    let vs = h.create_vertex_shader(&[
        0xFFFE_0200,
        0x0200_001F,
        0,
        0x900F_0000,
        0x0200_0001,
        0xC00F_0000,
        0x90E4_0000,
        0x0200_0001,
        0xD00F_0000,
        0xA0E4_0000,
        0x0200_0001,
        0xC001_0001,
        0xA000_0001,
        0xFFFF,
    ]);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[1.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0]),
        0
    );
    for range in [0, 1] {
        assert_eq!(h.set_render_state(D3DRS_RANGEFOGENABLE, range), 0);
        h.render_once(BLACK, |d| draw(d, &vertices()));
        assert_pixel_approx(h.read_pixel(64, 64), 0xFF40_BF00, 1, "programmable oFog");
    }
}
