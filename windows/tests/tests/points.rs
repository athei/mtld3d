//! Point size and point sprites.
//!
//! A `D3DPT_POINTLIST` draw rasterizes each vertex as a square of
//! `D3DRS_POINTSIZE` pixels (or the vertex's own PSIZE, or the vertex
//! shader's `oPts`), clamped to `POINTSIZE_MIN..=POINTSIZE_MAX` and optionally
//! attenuated by eye distance; `D3DRS_POINTSPRITEENABLE` textures the square
//! as a whole quad. Every test draws one point at the screen centre and reads
//! pixels at known offsets from it.

use mtld3d_tests::{Harness, PosColorVertex, PosVertex, TexturedVertex};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_PSIZE, D3DFVF_TEX1,
    D3DFVF_XYZ, D3DPT_POINTLIST, D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DRS_POINTSCALE_A,
    D3DRS_POINTSCALE_B, D3DRS_POINTSCALE_C, D3DRS_POINTSCALEENABLE, D3DRS_POINTSIZE,
    D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN, D3DRS_POINTSPRITEENABLE, D3DSAMP_MAGFILTER,
    D3DSAMP_MINFILTER, D3DTA_TEXTURE, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTS_PROJECTION,
    D3DTSS_COLORARG1, D3DTSS_COLOROP,
};

const BLACK: u32 = 0xFF00_0000;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;
const WHITE: u32 = 0xFFFF_FFFF;

/// Screen centre of the 640x480 harness back buffer.
const CX: u32 = 320;
const CY: u32 = 240;

fn set_float_rs(h: &Harness, rs: u32, value: f32) {
    assert_eq!(
        h.set_render_state(rs, value.to_bits()),
        0,
        "SetRenderState({rs})"
    );
}

/// Fixed-function diffuse passthrough for clip-space `PosColorVertex` draws.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
}

const fn centre_point() -> [PosColorVertex; 1] {
    [PosColorVertex {
        x: 0.0,
        y: 0.0,
        z: 0.5,
        color: GREEN,
    }]
}

/// Draw one green point at the centre and assert the square it covers.
///
/// `inside` is an offset (in pixels, along both axes) that must be green and
/// `outside` one that must still be background. Offsets stay a few pixels
/// away from the square's edge so the centre's half-pixel placement does not
/// matter.
fn assert_point_extent(h: &Harness, inside: u32, outside: u32, what: &str) {
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &centre_point()),
            0,
            "POINTLIST draws"
        );
    });
    assert_eq!(h.read_pixel(CX, CY), GREEN, "{what}: centre");
    assert_eq!(
        h.read_pixel(CX + inside, CY + inside),
        GREEN,
        "{what}: {inside} px inside the square"
    );
    assert_eq!(
        h.read_pixel(CX - inside, CY - inside),
        GREEN,
        "{what}: {inside} px inside the square (other corner)"
    );
    assert_eq!(
        h.read_pixel(CX + outside, CY),
        BLACK,
        "{what}: {outside} px right of the square"
    );
    assert_eq!(
        h.read_pixel(CX, CY - outside),
        BLACK,
        "{what}: {outside} px above the square"
    );
}

#[test]
fn render_state_point_size_sets_the_square() {
    let h = Harness::new();
    arm_diffuse(&h);
    // Default POINTSIZE is 1.0: the centre pixel and nothing around it.
    assert_point_extent(&h, 0, 3, "default size 1");
    set_float_rs(&h, D3DRS_POINTSIZE, 32.0);
    assert_point_extent(&h, 12, 20, "size 32");
    set_float_rs(&h, D3DRS_POINTSIZE, 8.0);
    assert_point_extent(&h, 2, 7, "size 8");
}

#[test]
fn untextured_xyz_point_under_an_ortho_projection_covers_its_pixels() {
    // The classic point-size probe: an XYZ-only FVF (no diffuse, the
    // texture stages at their defaults with nothing bound), a projection
    // that maps pixel coordinates to clip space, and a 15 px point at
    // (64, 64). The point must be white out to 7 px from its centre and the
    // clear colour from 8 px on, in both axes.
    let h = Harness::with_depth();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    // [-1, 1] x [1, -1] -> [0, 640] x [0, 480], z untouched (D3D row-vector layout).
    let ortho: [f32; 16] = [
        2.0 / 640.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / 480.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        1.0,
    ];
    assert_eq!(h.set_transform(D3DTS_PROJECTION, &ortho), 0, "projection");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF(XYZ)");
    set_float_rs(&h, D3DRS_POINTSIZE, 15.0);
    let point = [PosVertex {
        x: 64.0,
        y: 64.0,
        z: 0.1,
    }];
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_POINTLIST, 1, &point), 0);
    });
    for (x, y) in [
        (64 + 7, 64),
        (64 - 7, 64),
        (64, 64 + 7),
        (64, 64 - 7),
        (64, 64),
    ] {
        assert_eq!(
            h.read_pixel(x, y),
            WHITE,
            "({x}, {y}) inside the 15 px point"
        );
    }
    for (x, y) in [(64 + 8, 64), (64 - 8, 64), (64, 64 + 8), (64, 64 - 8)] {
        assert_eq!(
            h.read_pixel(x, y),
            BLUE,
            "({x}, {y}) outside the 15 px point"
        );
    }
}

#[test]
fn several_point_sizes_in_one_scene_each_keep_their_own() {
    // The classic point-size probe's sequence: a Clear outside the scene, then one
    // scene with several points whose D3DRS_POINTSIZE changes between draws,
    // and a readback with no Present in between. Every point must come out
    // at the size that was current when it was drawn.
    let h = Harness::with_depth();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    let ortho: [f32; 16] = [
        2.0 / 640.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / 480.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        1.0,
    ];
    let at = |x: f32| [PosVertex { x, y: 64.0, z: 0.1 }];
    assert!(h.pump(), "WM_QUIT before render");
    // Wine first draws one textured point in a scene of its own, before the
    // projection is set and without a Present, then clears.
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_TEX1), 0, "SetFVF(XYZ|TEX1)");
    let warm = [TexturedVertex {
        x: 64.0,
        y: 64.0,
        z: 0.1,
        color: 0,
        u: 0.0,
        v: 0.0,
    }];
    assert_eq!(h.begin_scene(), 0, "BeginScene (warm-up)");
    assert_eq!(h.draw_primitive_up(D3DPT_POINTLIST, 1, &warm), 0);
    assert_eq!(h.end_scene(), 0, "EndScene (warm-up)");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLUE, 1.0, 0),
        0,
        "Clear outside the scene"
    );
    assert_eq!(h.set_transform(D3DTS_PROJECTION, &ortho), 0, "projection");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF(XYZ)");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    for (x, size) in [(64.0, 15.0), (128.0, 31.0), (192.0, 30.75), (256.0, 63.0)] {
        set_float_rs(&h, D3DRS_POINTSIZE, size);
        assert_eq!(h.draw_primitive_up(D3DPT_POINTLIST, 1, &at(x)), 0);
    }
    set_float_rs(&h, D3DRS_POINTSIZE, 1.0);
    assert_eq!(h.draw_primitive_up(D3DPT_POINTLIST, 1, &at(320.0)), 0);
    assert_eq!(h.end_scene(), 0, "EndScene");
    // (x, half-extent r): white out to r, blue from r + 1.
    for (x, r) in [(64u32, 7u32), (128, 15), (192, 15), (256, 31), (320, 0)] {
        for (px, py) in [(x + r, 64), (x - r, 64), (x, 64 + r), (x, 64 - r)] {
            assert_eq!(
                h.read_pixel(px, py),
                WHITE,
                "({px}, {py}) inside point at {x}"
            );
        }
        let r = r + 1;
        for (px, py) in [(x + r, 64), (x - r, 64), (x, 64 + r), (x, 64 - r)] {
            assert_eq!(
                h.read_pixel(px, py),
                BLUE,
                "({px}, {py}) outside point at {x}"
            );
        }
    }
}

#[test]
fn point_size_clamps_to_min_and_max() {
    let h = Harness::new();
    arm_diffuse(&h);
    set_float_rs(&h, D3DRS_POINTSIZE, 32.0);
    set_float_rs(&h, D3DRS_POINTSIZE_MAX, 8.0);
    assert_point_extent(&h, 2, 7, "32 clamped down to max 8");
    set_float_rs(&h, D3DRS_POINTSIZE_MAX, 64.0);
    set_float_rs(&h, D3DRS_POINTSIZE, 1.0);
    set_float_rs(&h, D3DRS_POINTSIZE_MIN, 24.0);
    assert_point_extent(&h, 8, 16, "1 clamped up to min 24");
}

#[test]
fn psize_vertex_element_overrides_the_render_state() {
    // D3DFVF_PSIZE follows the position in the FVF layout; the per-vertex
    // size wins over D3DRS_POINTSIZE.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PsizeVertex {
        x: f32,
        y: f32,
        z: f32,
        psize: f32,
        color: u32,
    }
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_PSIZE | D3DFVF_DIFFUSE),
        0,
        "SetFVF with PSIZE"
    );
    set_float_rs(&h, D3DRS_POINTSIZE, 2.0);
    let point = [PsizeVertex {
        x: 0.0,
        y: 0.0,
        z: 0.5,
        psize: 24.0,
        color: GREEN,
    }];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &point),
            0,
            "POINTLIST with PSIZE draws"
        );
    });
    assert_eq!(h.read_pixel(CX, CY), GREEN, "centre");
    assert_eq!(
        h.read_pixel(CX + 8, CY + 8),
        GREEN,
        "inside the 24 px square"
    );
    assert_eq!(h.read_pixel(CX + 16, CY), BLACK, "outside the 24 px square");
}

#[test]
fn point_scale_multiplies_by_viewport_height_over_the_attenuation() {
    // D3D9: S = Vh * Si / sqrt(A + B * De + C * De^2). With A = 1 and B = C = 0
    // the size is Si * 480 on the 640x480 back buffer, so 0.05 becomes 24 px.
    // With A = 0, C = 1 and the point at eye distance 0.5 (identity
    // transforms, z = 0.5) it is 0.05 * 480 / 0.5 = 48 px.
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(h.set_render_state(D3DRS_POINTSCALEENABLE, 1), 0);
    set_float_rs(&h, D3DRS_POINTSIZE, 0.05);
    set_float_rs(&h, D3DRS_POINTSCALE_A, 1.0);
    set_float_rs(&h, D3DRS_POINTSCALE_B, 0.0);
    set_float_rs(&h, D3DRS_POINTSCALE_C, 0.0);
    assert_point_extent(&h, 8, 16, "constant attenuation: 24 px");
    set_float_rs(&h, D3DRS_POINTSCALE_A, 0.0);
    set_float_rs(&h, D3DRS_POINTSCALE_C, 1.0);
    assert_point_extent(&h, 20, 28, "quadratic attenuation at distance 0.5: 48 px");
    // Off again: the raw 0.05 clamps up to POINTSIZE_MIN = 1.
    assert_eq!(h.set_render_state(D3DRS_POINTSCALEENABLE, 0), 0);
    assert_point_extent(&h, 0, 3, "scale off");
}

/// `vs_2_0 { dcl_position v0; mov oPos, v0; mov oD0, v1; mov oPts, c0.x }`.
const VS_OPTS: [u32; 16] = [
    0xFFFE_0200,
    0x0200_001F,
    0x8000_0000,
    0x900F_0000, // dcl_position v0
    0x0200_001F,
    0x8000_000A,
    0x900F_0001, // dcl_color0 v1
    0x0200_0001,
    0xC00F_0000,
    0x90E4_0000, // mov oPos, v0
    0x0200_0001,
    0xD00F_0000,
    0x90E4_0001, // mov oD0, v1
    0x0200_0001,
    0xC00F_0002,
    0xA000_0000, // mov oPts, c0.x
];

/// Same shader without the `oPts` write.
const VS_NO_OPTS: [u32; 13] = [
    0xFFFE_0200,
    0x0200_001F,
    0x8000_0000,
    0x900F_0000,
    0x0200_001F,
    0x8000_000A,
    0x900F_0001,
    0x0200_0001,
    0xC00F_0000,
    0x90E4_0000,
    0x0200_0001,
    0xD00F_0000,
    0x90E4_0001,
];

fn with_end(code: &[u32]) -> Vec<u32> {
    let mut v = code.to_vec();
    v.push(0x0000_FFFF);
    v
}

#[test]
fn programmable_vs_point_size_comes_from_opts_or_the_render_state() {
    let h = Harness::new();
    arm_diffuse(&h);
    set_float_rs(&h, D3DRS_POINTSIZE, 16.0);
    // No oPts write: the render state sizes the point.
    let vs_plain = h.create_vertex_shader(&with_end(&VS_NO_OPTS));
    assert_eq!(h.set_vertex_shader(&vs_plain), 0, "SetVertexShader");
    assert_point_extent(&h, 5, 12, "VS without oPts: render state 16 px");
    // oPts wins over the render state, and is still clamped to POINTSIZE_MAX.
    let vs_opts = h.create_vertex_shader(&with_end(&VS_OPTS));
    assert_eq!(h.set_vertex_shader(&vs_opts), 0, "SetVertexShader");
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[32.0, 0.0, 0.0, 0.0]),
        0,
        "c0 = point size"
    );
    assert_point_extent(&h, 12, 20, "VS oPts: 32 px");
    set_float_rs(&h, D3DRS_POINTSIZE_MAX, 8.0);
    assert_point_extent(&h, 2, 7, "VS oPts clamped to max 8");
}

/// A 2x2 texture with one colour per quadrant: red, green / blue, white.
fn quadrant_texture(h: &Harness) -> mtld3d_tests::Texture<'_> {
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[RED, GREEN, BLUE, WHITE]);
    tex
}

/// Stage 0 samples the texture with point filtering and outputs it as-is.
fn arm_texture_stage(h: &Harness, tex: &mtld3d_tests::Texture<'_>) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
}

/// One textured point at the centre whose own texcoord would sample white.
const fn textured_centre_point() -> [TexturedVertex; 1] {
    [TexturedVertex {
        x: 0.0,
        y: 0.0,
        z: 0.5,
        color: WHITE,
        u: 0.75,
        v: 0.75,
    }]
}

fn assert_sprite_quadrants(h: &Harness, what: &str) {
    assert_eq!(
        h.read_pixel(CX - 8, CY - 8),
        RED,
        "{what}: top-left quadrant"
    );
    assert_eq!(
        h.read_pixel(CX + 8, CY - 8),
        GREEN,
        "{what}: top-right quadrant"
    );
    assert_eq!(
        h.read_pixel(CX - 8, CY + 8),
        BLUE,
        "{what}: bottom-left quadrant"
    );
    assert_eq!(
        h.read_pixel(CX + 8, CY + 8),
        WHITE,
        "{what}: bottom-right quadrant"
    );
    assert_eq!(
        h.read_pixel(CX + 24, CY),
        BLACK,
        "{what}: outside the sprite"
    );
}

#[test]
fn point_sprite_textures_the_square_with_the_point_coordinate() {
    let h = Harness::new();
    let tex = quadrant_texture(&h);
    arm_texture_stage(&h, &tex);
    set_float_rs(&h, D3DRS_POINTSIZE, 32.0);
    // Sprites off: the vertex's own (0.75, 0.75) samples the white texel
    // across the whole square.
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &textured_centre_point()),
            0
        );
    });
    assert_eq!(
        h.read_pixel(CX - 8, CY - 8),
        WHITE,
        "no sprite: vertex texcoord"
    );
    assert_eq!(
        h.read_pixel(CX + 8, CY + 8),
        WHITE,
        "no sprite: vertex texcoord"
    );
    // Sprites on: each quadrant of the square samples its own texel, with
    // (0,0) at the top left.
    assert_eq!(h.set_render_state(D3DRS_POINTSPRITEENABLE, 1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &textured_centre_point()),
            0
        );
    });
    assert_sprite_quadrants(&h, "fixed-function sprite");
}

#[test]
fn point_sprite_state_leaves_triangles_alone() {
    // POINTSPRITEENABLE only affects point primitives: a triangle drawn with
    // it on keeps its vertex texcoords.
    let h = Harness::new();
    let tex = quadrant_texture(&h);
    arm_texture_stage(&h, &tex);
    assert_eq!(h.set_render_state(D3DRS_POINTSPRITEENABLE, 1), 0);
    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u: 0.25,
        v: 0.75,
    };
    // A triangle covering the centre, every vertex at the blue texel.
    let tri = [v(-1.0, 1.0), v(1.0, 1.0), v(0.0, -1.0)];
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_eq!(h.read_pixel(CX, CY), BLUE, "triangle keeps its texcoords");
}

/// `ps_2_0 { dcl t0; dcl_2d s0; texld r0, t0, s0; mov oC0, r0 }`.
const PS_TEXLD: [u32; 15] = [
    0xFFFF_0200,
    0x0200_001F,
    0x8000_0000,
    0xB00F_0000, // dcl t0
    0x0200_001F,
    0x9000_0000,
    0xA00F_0800, // dcl_2d s0
    0x0300_0042,
    0x800F_0000,
    0xB0E4_0000,
    0xA0E4_0800, // texld r0, t0, s0
    0x0200_0001,
    0x800F_0800,
    0x80E4_0000, // mov oC0, r0
    0x0000_FFFF,
];

#[test]
fn point_sprite_replaces_programmable_ps_texcoords() {
    let h = Harness::new();
    let tex = quadrant_texture(&h);
    arm_texture_stage(&h, &tex);
    let ps = h.create_pixel_shader(&PS_TEXLD);
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");
    set_float_rs(&h, D3DRS_POINTSIZE, 32.0);
    assert_eq!(h.set_render_state(D3DRS_POINTSPRITEENABLE, 1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &textured_centre_point()),
            0
        );
    });
    assert_sprite_quadrants(&h, "ps_2_0 sprite");
}
