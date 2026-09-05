//! User clip planes.
//!
//! `SetClipPlane` + `D3DRS_CLIPPLANEENABLE` discard every fragment on the
//! negative side of `a*x + b*y + c*z + d*w`. The fixed-function pipeline
//! reads the planes in world space, a programmable vertex shader in clip
//! space, `D3DRS_CLIPPING` gates them, pre-transformed vertices ignore them,
//! and a `D3DSBT_ALL` state block carries them. Every test draws a
//! full-screen green quad and reads pixels above and below the plane.

use mtld3d_tests::{Harness, PosColorVertex, RhwVertex};
use mtld3d_types::{
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DPT_TRIANGLELIST, D3DRS_CLIPPING,
    D3DRS_CLIPPLANEENABLE, D3DRS_LIGHTING, D3DSBT_ALL, D3DTS_VIEW,
};

const BLACK: u32 = 0xFF00_0000;
const GREEN: u32 = 0xFF00_FF00;
const YELLOW: u32 = 0xFFFF_FF00;

/// Fixed-function diffuse passthrough for clip-space `PosColorVertex` draws.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
}

fn full_quad() -> [PosColorVertex; 6] {
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: GREEN,
    };
    [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ]
}

fn draw_quad(h: &Harness) {
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_quad()),
            0,
            "quad draws"
        );
    });
}

/// Plane `y >= d_ndc` in a space where y runs -1..1 bottom to top.
const fn plane_y_above(d: f32) -> [f32; 4] {
    [0.0, 1.0, 0.0, -d]
}

#[test]
fn fixed_function_plane_keeps_the_positive_side() {
    // Plane y >= 0 with identity transforms: the top half of the screen stays,
    // the bottom half is clipped. Rows 100 and 380 sit well inside each half.
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.0)), 0, "SetClipPlane");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
    draw_quad(&h);
    assert_eq!(h.read_pixel(320, 100), GREEN, "above the plane");
    assert_eq!(h.read_pixel(320, 380), BLACK, "below the plane is clipped");
    // Disabling the plane brings the bottom half back.
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 0), 0);
    draw_quad(&h);
    assert_eq!(h.read_pixel(320, 380), GREEN, "plane disabled");
}

#[test]
fn fixed_function_planes_are_world_space() {
    // The view matrix translates y by +0.5 (D3D row-vector layout puts it in
    // row 3). A world-space plane y >= 0 becomes y_view >= 0.5 on screen, so
    // only the top quarter survives; an eye-space reading would keep the
    // whole top half.
    let h = Harness::new();
    arm_diffuse(&h);
    let mut view = [0.0f32; 16];
    view[0] = 1.0;
    view[5] = 1.0;
    view[10] = 1.0;
    view[15] = 1.0;
    view[13] = 0.5;
    assert_eq!(h.set_transform(D3DTS_VIEW, &view), 0, "view");
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.0)), 0, "SetClipPlane");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
    draw_quad(&h);
    // y_ndc = 0.75 -> row 60: kept. y_ndc = 0.25 -> row 180: clipped
    // (y_world = -0.25). Row 380 is clipped either way.
    assert_eq!(h.read_pixel(320, 60), GREEN, "top quarter (world y > 0)");
    assert_eq!(
        h.read_pixel(320, 180),
        BLACK,
        "second quarter (world y < 0)"
    );
    assert_eq!(h.read_pixel(320, 380), BLACK, "bottom half");
}

#[test]
fn clipping_render_state_gates_the_planes() {
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.0)), 0, "SetClipPlane");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_CLIPPING, 0), 0);
    draw_quad(&h);
    assert_eq!(
        h.read_pixel(320, 380),
        GREEN,
        "CLIPPING off: no user clipping"
    );
    assert_eq!(h.set_render_state(D3DRS_CLIPPING, 1), 0);
    draw_quad(&h);
    assert_eq!(h.read_pixel(320, 380), BLACK, "CLIPPING on: clipped again");
}

#[test]
fn several_planes_intersect_and_sparse_indices_pack() {
    // Planes 1 (y >= 0) and 5 (x >= 0) with plane 0 left disabled: only the
    // top-right quadrant survives, which proves the enabled planes are packed
    // from their sparse indices.
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(
        h.set_clip_plane(0, [0.0, -1.0, 0.0, 0.0]),
        0,
        "unused plane 0"
    );
    assert_eq!(h.set_clip_plane(1, plane_y_above(0.0)), 0, "plane 1");
    assert_eq!(h.set_clip_plane(5, [1.0, 0.0, 0.0, 0.0]), 0, "plane 5");
    assert_eq!(
        h.set_render_state(D3DRS_CLIPPLANEENABLE, (1 << 1) | (1 << 5)),
        0
    );
    draw_quad(&h);
    assert_eq!(h.read_pixel(480, 100), GREEN, "top-right kept");
    assert_eq!(h.read_pixel(160, 100), BLACK, "top-left clipped by plane 5");
    assert_eq!(
        h.read_pixel(480, 380),
        BLACK,
        "bottom-right clipped by plane 1"
    );
}

#[test]
fn pre_transformed_vertices_ignore_the_planes() {
    // D3D9 never applies user clip planes to XYZRHW geometry.
    let h = Harness::new();
    assert_eq!(h.clear_texture(0), 0, "no texture");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0, "SetFVF");
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.0)), 0, "SetClipPlane");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
    let v = |x: f32, y: f32| RhwVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        color: GREEN,
    };
    let quad = [
        v(0.0, 0.0),
        v(640.0, 0.0),
        v(0.0, 480.0),
        v(640.0, 0.0),
        v(640.0, 480.0),
        v(0.0, 480.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_eq!(h.read_pixel(320, 100), GREEN);
    assert_eq!(h.read_pixel(320, 380), GREEN, "RHW geometry is not clipped");
}

/// `vs_2_0 { dcl_position v0; dcl_color0 v1; mov oPos, v0; mov oD0, v1 }`.
const VS_PASSTHROUGH: [u32; 14] = [
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
    0x0000_FFFF,
];

#[test]
fn programmable_vs_planes_are_clip_space() {
    // With a pass-through shader the clip-space plane y >= 0.5 keeps the top
    // quarter only; the view matrix set below must not move it (a programmable
    // VS owns its own transforms).
    let h = Harness::new();
    arm_diffuse(&h);
    let vs = h.create_vertex_shader(&VS_PASSTHROUGH);
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    let mut view = [0.0f32; 16];
    view[0] = 1.0;
    view[5] = 1.0;
    view[10] = 1.0;
    view[15] = 1.0;
    view[13] = -0.9;
    assert_eq!(h.set_transform(D3DTS_VIEW, &view), 0, "view (ignored)");
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.5)), 0, "SetClipPlane");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
    draw_quad(&h);
    assert_eq!(h.read_pixel(320, 60), GREEN, "y_ndc 0.75 kept");
    assert_eq!(h.read_pixel(320, 180), BLACK, "y_ndc 0.25 clipped");
    assert_eq!(h.read_pixel(320, 380), BLACK, "bottom clipped");
}

#[test]
fn enabling_a_plane_between_two_draws_of_the_same_vertex_shader_takes_effect() {
    // Wine's clip_planes_test shape: one quad with the planes off, then the
    // enable bit flips and a second quad draws with the same shader. The
    // second draw must pick up the clipped shader variant.
    let h = Harness::new();
    arm_diffuse(&h);
    let vs = h.create_vertex_shader(&VS_PASSTHROUGH);
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    assert_eq!(h.set_clip_plane(0, plane_y_above(0.0)), 0, "SetClipPlane");
    let v = |x: f32, y: f32, color: u32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color,
    };
    let quad = |color: u32| {
        [
            v(-1.0, 1.0, color),
            v(1.0, 1.0, color),
            v(-1.0, -1.0, color),
            v(1.0, 1.0, color),
            v(1.0, -1.0, color),
            v(-1.0, -1.0, color),
        ]
    };
    h.render_once(BLACK, |d| {
        assert_eq!(d.set_render_state(D3DRS_CLIPPLANEENABLE, 0), 0);
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(YELLOW)), 0);
        assert_eq!(d.set_render_state(D3DRS_CLIPPLANEENABLE, 1), 0);
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(GREEN)), 0);
    });
    assert_eq!(h.read_pixel(320, 100), GREEN, "second quad above the plane");
    assert_eq!(
        h.read_pixel(320, 380),
        YELLOW,
        "second quad clipped below it"
    );
}

#[test]
fn indices_past_the_cap_alias_the_last_plane() {
    // D3D9: with MaxUserClipPlanes = 6, SetClipPlane and GetClipPlane on any
    // index >= 5 address slot 5; planes 0..5 keep their own values.
    let h = Harness::new();
    assert_eq!(h.device_caps().max_user_clip_planes, 6, "MaxUserClipPlanes");
    for i in 0..12u8 {
        let d = f32::from(i);
        assert_eq!(
            h.set_clip_plane(u32::from(i), [2.0, 8.0, 5.0, d]),
            0,
            "SetClipPlane({i})"
        );
    }
    for i in 0..12u8 {
        let expected_d = if i >= 5 { 11.0 } else { f32::from(i) };
        assert_eq!(
            h.get_clip_plane(u32::from(i)),
            (0, [2.0, 8.0, 5.0, expected_d]),
            "GetClipPlane({i})"
        );
    }
}

#[test]
fn state_block_all_captures_and_restores_the_planes() {
    let h = Harness::new();
    arm_diffuse(&h);
    let captured = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(h.set_clip_plane(2, captured), 0, "SetClipPlane");
    let block = h.create_state_block(D3DSBT_ALL);
    assert_eq!(h.set_clip_plane(2, [0.0; 4]), 0, "overwrite");
    assert_eq!(h.get_clip_plane(2), (0, [0.0; 4]));
    assert_eq!(block.apply(), 0, "Apply");
    assert_eq!(
        h.get_clip_plane(2),
        (0, captured),
        "D3DSBT_ALL restores SetClipPlane"
    );
    // And the restored plane is live for rendering: plane 2 = y >= 0.
    assert_eq!(h.set_clip_plane(2, plane_y_above(0.0)), 0);
    let block = h.create_state_block(D3DSBT_ALL);
    assert_eq!(h.set_clip_plane(2, [0.0, -1.0, 0.0, 0.0]), 0, "flip it");
    assert_eq!(block.apply(), 0, "Apply");
    assert_eq!(h.set_render_state(D3DRS_CLIPPLANEENABLE, 1 << 2), 0);
    draw_quad(&h);
    assert_eq!(
        h.read_pixel(320, 100),
        GREEN,
        "restored plane keeps the top"
    );
    assert_eq!(
        h.read_pixel(320, 380),
        BLACK,
        "restored plane clips the bottom"
    );
}
