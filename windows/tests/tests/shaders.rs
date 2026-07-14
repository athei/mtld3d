//! Programmable pipeline: hand-assembled `vs_2_0` / `ps_2_0`.
//!
//! Bound and driven with a pixel-shader constant, verified by the rendered
//! colour.

use mtld3d_tests::{Harness, PosVertex};
use mtld3d_types::{D3DERR_INVALIDCALL, D3DFVF_XYZ, D3DPT_TRIANGLELIST};

/// `vs_2_0`: `dcl_position v0; mov oPos, v0;`
const VS_BC: [u32; 8] = [
    0xFFFE_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (1) | (2 << 24),
    (4 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    0x0000_FFFF,
];

/// `ps_2_0`: `mov oC0, c0;` (c0 supplied via the constant buffer).
const PS_BC: [u32; 5] = [
    0xFFFF_0200,
    (1) | (2 << 24),
    (1 << 11) | (0xF << 16),
    (2 << 28) | (0xE4 << 16),
    0x0000_FFFF,
];

const fn centered_triangle() -> [PosVertex; 3] {
    [
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
    ]
}

#[test]
fn user_shader_constant_drives_color() {
    let h = Harness::new();
    let vs = h.create_vertex_shader(&VS_BC);
    let ps = h.create_pixel_shader(&PS_BC);
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "SetFVF");

    let tri = centered_triangle();

    // c0 = red → triangle renders red.
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[1.0, 0.0, 0.0, 1.0]),
        0,
        "SetPSConstF(red)"
    );
    h.render_once(0xFF00_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFFFF_0000,
        "constant red via user shader"
    );

    // c0 = green → same geometry now renders green (constant rebind path).
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[0.0, 1.0, 0.0, 1.0]),
        0,
        "SetPSConstF(green)"
    );
    h.render_once(0xFF00_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFF00_FF00,
        "constant green via user shader"
    );

    assert_eq!(h.clear_vertex_shader(), 0, "unbind VS");
    assert_eq!(h.clear_pixel_shader(), 0, "unbind PS");
}

/// `vs_2_0`: `dcl_position v0; add oPos, v0, c0;` — translate by VS constant c0.
const VS_TRANSLATE: [u32; 9] = [
    0xFFFE_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (2) | (3 << 24), // add: dst + 2 src tokens
    (4 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    (2 << 28) | (0xE4 << 16),
    0x0000_FFFF,
];

#[test]
fn vertex_shader_constant_translates_geometry() {
    let h = Harness::new();
    let vs = h.create_vertex_shader(&VS_TRANSLATE);
    let ps = h.create_pixel_shader(&PS_BC);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0);
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[0.0, 1.0, 0.0, 1.0]),
        0,
        "PS green"
    );
    let tri = centered_triangle();

    // c0 = 0 → triangle at the origin covers the centre.
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[0.0, 0.0, 0.0, 0.0]),
        0,
        "VS c0 = 0"
    );
    h.render_once(0xFF00_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_eq!(
        h.read_pixel(320, 240),
        0xFF00_FF00,
        "centred triangle is green"
    );

    // c0 = (+2, 0) → translated off-screen, centre reverts to background.
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[2.0, 0.0, 0.0, 0.0]),
        0,
        "VS c0 = +2x"
    );
    h.render_once(0xFF00_00FF, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_ne!(
        h.read_pixel(320, 240),
        0xFF00_FF00,
        "translated triangle left the centre"
    );
}

#[test]
fn float_shader_constant_setters_accept() {
    let h = Harness::new();
    let vs = h.create_vertex_shader(&VS_BC);
    let ps = h.create_pixel_shader(&PS_BC);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[1.0, 2.0, 3.0, 4.0]),
        0,
        "VS const F"
    );
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[1.0, 0.0, 0.0, 1.0]),
        0,
        "PS const F"
    );
}

#[test]
fn float_shader_constant_setters_reject_out_of_range() {
    // D3D9 validates the `[start, start + count)` register window: the last
    // in-range write succeeds, anything past the file (or a `-1` start whose
    // unsigned value would wrap) is D3DERR_INVALIDCALL. A clamping write that
    // silently returned S_OK lets the canonical `while (SUCCEEDED(Set(i++)))`
    // probe loop spin forever, so this contract is load-bearing. vs_3_0
    // exposes 256 float constants, ps_3_0 exposes 224.
    let h = Harness::new();
    let one = [0.0_f32; 4];
    let four = [0.0_f32; 16];

    // Vertex: c255 is the last valid register; c256 and a 4-wide window from
    // c253 both overflow the 256-row file.
    assert_eq!(
        h.set_vertex_shader_constant_f(255, &one),
        0,
        "VS c255 in range"
    );
    assert_eq!(
        h.set_vertex_shader_constant_f(256, &one),
        D3DERR_INVALIDCALL,
        "VS c256 past file"
    );
    assert_eq!(
        h.set_vertex_shader_constant_f(253, &four),
        D3DERR_INVALIDCALL,
        "VS 4-wide window past c255"
    );
    assert_eq!(
        h.set_vertex_shader_constant_f(u32::MAX, &one),
        D3DERR_INVALIDCALL,
        "VS -1 start must not wrap into range"
    );

    // Pixel: the file stops at c223 (ps_3_0), 32 registers below the vertex one.
    assert_eq!(
        h.set_pixel_shader_constant_f(223, &one),
        0,
        "PS c223 in range"
    );
    assert_eq!(
        h.set_pixel_shader_constant_f(224, &one),
        D3DERR_INVALIDCALL,
        "PS c224 past file"
    );
    assert_eq!(
        h.set_pixel_shader_constant_f(u32::MAX, &one),
        D3DERR_INVALIDCALL,
        "PS -1 start must not wrap into range"
    );
}

#[test]
fn integer_and_bool_shader_constants_round_trip() {
    // SM2/SM3 integer/bool constant registers are stored even though the MSL
    // emit does not consume them yet: Set succeeds and Get reads them back.
    let h = Harness::new();
    assert_eq!(
        h.set_vertex_shader_constant_i(0, &[1, 2, 3, 4]),
        0,
        "VS const I set"
    );
    assert_eq!(
        h.set_vertex_shader_constant_b(0, &[1, 0]),
        0,
        "VS const B set"
    );
    assert_eq!(
        h.set_pixel_shader_constant_i(0, &[5, 6, 7, 8]),
        0,
        "PS const I set"
    );
    assert_eq!(
        h.set_pixel_shader_constant_b(0, &[0, 1]),
        0,
        "PS const B set"
    );

    let (hr, vs_i) = h.get_vertex_shader_constant_i(0, 1);
    assert_eq!(hr, 0, "VS const I get");
    assert_eq!(vs_i, [1, 2, 3, 4], "VS const I round-trip");

    let (hr, vs_b) = h.get_vertex_shader_constant_b(0, 2);
    assert_eq!(hr, 0, "VS const B get");
    assert_eq!(vs_b, [1, 0], "VS const B round-trip");

    let (hr, ps_i) = h.get_pixel_shader_constant_i(0, 1);
    assert_eq!(hr, 0, "PS const I get");
    assert_eq!(ps_i, [5, 6, 7, 8], "PS const I round-trip");

    let (hr, ps_b) = h.get_pixel_shader_constant_b(0, 2);
    assert_eq!(hr, 0, "PS const B get");
    assert_eq!(ps_b, [0, 1], "PS const B round-trip");
}

#[test]
fn float_shader_constants_round_trip() {
    // GetVertexShaderConstantF / GetPixelShaderConstantF read back the values
    // written by the matching Set. Values are exactly representable, so a copy
    // round-trip compares bit-exact.
    let h = Harness::new();
    assert_eq!(
        h.set_vertex_shader_constant_f(0, &[1.0, 2.0, 3.0, 4.0]),
        0,
        "VS const F set"
    );
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[0.5, 0.25, 0.0, 1.0]),
        0,
        "PS const F set"
    );

    let (hr, vs_f) = h.get_vertex_shader_constant_f(0, 1);
    assert_eq!(hr, 0, "VS const F get");
    assert_eq!(vs_f, [1.0, 2.0, 3.0, 4.0], "VS const F round-trip");

    let (hr, ps_f) = h.get_pixel_shader_constant_f(0, 1);
    assert_eq!(hr, 0, "PS const F get");
    assert_eq!(ps_f, [0.5, 0.25, 0.0, 1.0], "PS const F round-trip");
}
