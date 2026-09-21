//! Render-state *execution*: blend, colour-write masking, scissor, culling.
//!
//! Plus get/set round-trips and spec-default verification.

use mtld3d_tests::{
    Harness, HarnessConfig, PosColorVertex, Rgba8, Surface, Texture, TexturedVertex,
    assert_pixel_approx,
};
use mtld3d_types::{
    D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLENDOP_ADD, D3DCLEAR_STENCIL,
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_LESS, D3DCMP_LESSEQUAL,
    D3DCULL_CCW, D3DCULL_CW, D3DCULL_NONE, D3DFILL_SOLID, D3DFILL_WIREFRAME, D3DFMT_A8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_ALPHABLENDENABLE, D3DRS_BLENDOP, D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_DEPTHBIAS,
    D3DRS_DESTBLEND, D3DRS_FILLMODE, D3DRS_LIGHTING, D3DRS_SCISSORTESTENABLE, D3DRS_SRCBLEND,
    D3DRS_SRGBWRITEENABLE, D3DRS_STENCILENABLE, D3DRS_STENCILFUNC, D3DRS_STENCILMASK,
    D3DRS_STENCILPASS, D3DRS_STENCILREF, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DSTENCILOP_KEEP,
    D3DSTENCILOP_REPLACE, D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_RENDERTARGET,
    render_state_defaults,
};

const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;
const WHITE: u32 = 0xFFFF_FFFF;

fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
}

/// A full clip-space quad (two triangles) of one colour.
fn fill_quad(color: u32) -> [PosColorVertex; 6] {
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color,
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

/// A full clip-space quad at a chosen depth.
fn quad_at_depth(color: u32, z: f32) -> [PosColorVertex; 6] {
    let mut q = fill_quad(color);
    for v in &mut q {
        v.z = z;
    }
    q
}

const fn centered_triangle(color: u32) -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

#[test]
fn defaults_match_spec() {
    // A depth device so ZENABLE's "TRUE when depth present" default applies.
    let h = Harness::with_depth();
    let spec = render_state_defaults();
    for state in [
        mtld3d_types::D3DRS_ZENABLE,
        mtld3d_types::D3DRS_ZWRITEENABLE,
        mtld3d_types::D3DRS_ZFUNC,
        mtld3d_types::D3DRS_FILLMODE,
        mtld3d_types::D3DRS_CULLMODE,
        mtld3d_types::D3DRS_SHADEMODE,
        mtld3d_types::D3DRS_LIGHTING,
        mtld3d_types::D3DRS_ALPHABLENDENABLE,
        D3DRS_SRCBLEND,
        D3DRS_DESTBLEND,
        D3DRS_BLENDOP,
        mtld3d_types::D3DRS_ALPHATESTENABLE,
        mtld3d_types::D3DRS_ALPHAFUNC,
        mtld3d_types::D3DRS_STENCILENABLE,
        D3DRS_SCISSORTESTENABLE,
        D3DRS_COLORWRITEENABLE,
        mtld3d_types::D3DRS_TEXTUREFACTOR,
        mtld3d_types::D3DRS_FOGENABLE,
    ] {
        assert_eq!(
            h.render_state(state),
            spec[state as usize],
            "RenderState {state} default mismatch",
        );
    }
}

#[test]
fn set_get_round_trip() {
    let h = Harness::new();
    for (state, value) in [
        (D3DRS_CULLMODE, D3DCULL_CW),
        (D3DRS_SRCBLEND, D3DBLEND_SRCALPHA),
        (D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA),
        (D3DRS_COLORWRITEENABLE, 0x0000_0007),
        (mtld3d_types::D3DRS_STENCILREF, 0x42),
        (mtld3d_types::D3DRS_TEXTUREFACTOR, 0x1234_5678),
    ] {
        assert_eq!(
            h.set_render_state(state, value),
            0,
            "SetRenderState {state}"
        );
        assert_eq!(
            h.render_state(state),
            value,
            "GetRenderState {state} round-trip"
        );
    }
}

/// The render states whose value has to land inside a D3D9 enum space.
///
/// `SetRenderState` takes a DWORD, so each of these can be handed a value the
/// space does not contain; the layer reads the state's spec default instead.
const ENUM_STATES: [u32; 22] = [
    mtld3d_types::D3DRS_ZFUNC,
    D3DRS_FILLMODE,
    mtld3d_types::D3DRS_ALPHAFUNC,
    D3DRS_STENCILFUNC,
    mtld3d_types::D3DRS_CCW_STENCILFUNC,
    D3DRS_SRCBLEND,
    mtld3d_types::D3DRS_SRCBLENDALPHA,
    D3DRS_DESTBLEND,
    mtld3d_types::D3DRS_DESTBLENDALPHA,
    D3DRS_BLENDOP,
    mtld3d_types::D3DRS_BLENDOPALPHA,
    D3DRS_CULLMODE,
    mtld3d_types::D3DRS_STENCILFAIL,
    mtld3d_types::D3DRS_STENCILZFAIL,
    D3DRS_STENCILPASS,
    mtld3d_types::D3DRS_CCW_STENCILFAIL,
    mtld3d_types::D3DRS_CCW_STENCILZFAIL,
    mtld3d_types::D3DRS_CCW_STENCILPASS,
    D3DRS_COLORWRITEENABLE,
    mtld3d_types::D3DRS_COLORWRITEENABLE1,
    mtld3d_types::D3DRS_COLORWRITEENABLE2,
    mtld3d_types::D3DRS_COLORWRITEENABLE3,
];

#[test]
fn out_of_range_enum_states_draw_what_the_defaults_draw() {
    let h = Harness::new();
    arm_diffuse(&h);
    // Alpha test on so the fixed-function variant key narrows ALPHAFUNC too;
    // the default compare passes every fragment either way.
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_ALPHATESTENABLE, 1),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_ALPHAREF, 0), 0);

    let spec = render_state_defaults();
    for state in ENUM_STATES {
        assert_eq!(
            h.set_render_state(state, spec[state as usize]),
            0,
            "SetRenderState {state} default"
        );
    }
    let quad = fill_quad(WHITE);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "in-range draw"
        );
    });
    let expected = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        expected.r > 200 && expected.g > 200 && expected.b > 200,
        "white under the defaults, got {expected:?}"
    );

    // A DWORD no D3D9 enum space contains. The call succeeds (native stores
    // any DWORD) and reads back verbatim, and the draw that follows is the
    // one the spec default produces.
    for state in ENUM_STATES {
        let poison = 0x0001_0000 | spec[state as usize];
        assert_eq!(
            h.set_render_state(state, poison),
            0,
            "SetRenderState {state} out of range"
        );
        assert_eq!(
            h.render_state(state),
            poison,
            "GetRenderState {state} round-trip"
        );
    }
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "out-of-range draw"
        );
    });
    assert_eq!(
        Rgba8::from_pixel(h.read_pixel(320, 240)),
        expected,
        "out-of-range enum states drew a different frame"
    );
}

#[test]
fn alpha_blend_src_over_dest() {
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_BLENDOP, D3DBLENDOP_ADD), 0);

    // Green at alpha 0.5 over an opaque blue background → ~(0, 128, 128).
    let quad = fill_quad(0x8000_FF00);
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "blend draw"
        );
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(px.r < 20, "red stays 0, got {px:?}");
    assert!((110..=145).contains(&px.g), "green ~half, got {px:?}");
    assert!((110..=145).contains(&px.b), "blue ~half, got {px:?}");
}

#[test]
fn additive_blend_accumulates() {
    let h = Harness::new();
    arm_diffuse(&h);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ONE), 0);

    // Opaque green added onto a dark-red background → (64, 255, 0).
    let quad = fill_quad(0xFF00_FF00);
    h.render_once(0xFF40_0000, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "additive draw"
        );
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (48..=80).contains(&px.r),
        "red retained from dest, got {px:?}"
    );
    assert!(px.g > 240, "green saturated, got {px:?}");
    assert!(px.b < 20, "blue stays 0, got {px:?}");
}

#[test]
fn colorwrite_mask_drops_red() {
    let h = Harness::new();
    arm_diffuse(&h);
    // Enable GREEN|BLUE|ALPHA, mask out RED.
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0x0000_000E), 0);

    let quad = fill_quad(0xFFFF_FFFF); // white
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "masked draw"
        );
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(px.r < 20, "red masked off (stays cleared 0), got {px:?}");
    assert!(px.g > 200 && px.b > 200, "green+blue written, got {px:?}");
}

#[test]
fn scissor_clips_draw() {
    let h = Harness::new();
    arm_diffuse(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0, "clear before enabling scissor");
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
    assert_eq!(
        h.set_scissor_rect(&D3DRECT {
            x1: 0,
            y1: 0,
            x2: 320,
            y2: 240
        }),
        0
    );
    let quad = fill_quad(0xFFFF_0000); // red, full screen
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "scissored draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(h.read_pixel(160, 120), 0xFFFF_0000, "inside scissor is red");
    assert_eq!(h.read_pixel(480, 360), BLACK, "outside scissor stays black");
}

#[test]
fn cull_mode_discriminates_winding() {
    let h = Harness::new();
    arm_diffuse(&h);
    let tri = centered_triangle(GREEN);

    // NONE never culls.
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "CULL_NONE draws the triangle"
    );

    // CW and CCW must disagree — exactly one culls this winding.
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_CW), 0);
    h.render_once(BLACK, |d| {
        let _ = d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri);
    });
    let cw = h.read_pixel(320, 280);

    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_CCW), 0);
    h.render_once(BLACK, |d| {
        let _ = d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri);
    });
    let ccw = h.read_pixel(320, 280);

    assert_ne!(cw, ccw, "CW vs CCW must cull opposite windings");
}

#[test]
fn stencil_render_states_round_trip() {
    let h = Harness::with_depth();
    for (state, value) in [
        (D3DRS_STENCILENABLE, 1),
        (D3DRS_STENCILFUNC, D3DCMP_EQUAL),
        (D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE),
        (D3DRS_STENCILREF, 0x7F),
        (D3DRS_STENCILMASK, 0x00FF),
    ] {
        assert_eq!(
            h.set_render_state(state, value),
            0,
            "SetRenderState {state}"
        );
        assert_eq!(
            h.render_state(state),
            value,
            "GetRenderState {state} round-trip"
        );
    }
}

#[test]
fn wireframe_fill_mode_leaves_triangle_interior_clear() {
    let h = Harness::new();
    arm_diffuse(&h);
    let tri = centered_triangle(GREEN);

    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "solid fill covers the interior"
    );

    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0);
    });
    assert_eq!(
        h.read_pixel(320, 280),
        BLACK,
        "wireframe leaves the triangle interior clear"
    );
}

#[test]
fn stencil_test_gates_rendering() {
    // Stamp the reference under a centred triangle, then draw a fullscreen quad
    // that only passes where the stamp landed. The quad covers the corner too,
    // so a corner still holding the clear colour is the proof that the stencil
    // test rejected those fragments rather than the quad missing them.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        // The depth plane starts undefined and the depth test is on, so it is
        // cleared along with the stencil the test is about.
        assert_eq!(
            d.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 0),
            0,
            "depth and stencil clear"
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_ALWAYS), 0);
        assert_eq!(
            d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE),
            0
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, 1), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &centered_triangle(BLUE)),
            0,
            "stamp draw"
        );

        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(GREEN)),
            0,
            "gated draw"
        );
    });
    assert_eq!(h.read_pixel(320, 240), GREEN, "inside the stamp");
    assert_eq!(h.read_pixel(10, 10), BLACK, "outside the stamp");
}

#[test]
fn additive_pass_selected_by_depth_equal_reaches_the_target() {
    // A renderer that writes depth in one pass and adds light in a second
    // selects the same fragments with `EQUAL`, which only holds while the
    // second pass rasterizes the depth the first one stored. Any nudge
    // applied to the blended pass alone makes it match nothing and vanish.
    // Prime depth with an opaque quad, then draw the same geometry additively
    // under `EQUAL`: the additive white saturates the primed green, so a
    // still-green centre is the proof that the second pass was rejected.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    let quad = quad_at_depth(GREEN, 0.5);
    h.render_once(BLACK, |d| {
        assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0, "depth clear");
        assert_eq!(d.set_render_state(mtld3d_types::D3DRS_ZENABLE, 1), 0);
        assert_eq!(d.set_render_state(mtld3d_types::D3DRS_ZWRITEENABLE, 1), 0);
        assert_eq!(
            d.set_render_state(mtld3d_types::D3DRS_ZFUNC, D3DCMP_LESSEQUAL),
            0
        );
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "depth-priming pass"
        );

        assert_eq!(d.set_render_state(mtld3d_types::D3DRS_ZWRITEENABLE, 0), 0);
        assert_eq!(
            d.set_render_state(mtld3d_types::D3DRS_ZFUNC, D3DCMP_EQUAL),
            0
        );
        assert_eq!(d.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
        assert_eq!(d.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE), 0);
        assert_eq!(d.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ONE), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(WHITE, 0.5)),
            0,
            "additive pass"
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        WHITE,
        "the EQUAL-selected additive pass reached the target"
    );
}

#[test]
fn depth_bias_is_an_absolute_offset_at_every_depth() {
    // `D3DRS_DEPTHBIAS` is added to the fragment's depth as it stands, whatever
    // that depth is. A quad one `gap` behind a stored depth under `LESS` passes
    // once the bias exceeds the gap and not before, and the amount it takes must
    // not depend on where in the depth range the pair sits: a float depth buffer
    // that scales the bias by the depth's exponent delivers half of it at 0.75
    // and a thirty-second of it at 0.047.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    let gap = 1.0_f32 / 4096.0;
    let wins = |z0: f32, bias: f32| {
        h.render_once(BLACK, |d| {
            assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0, "depth clear");
            assert_eq!(d.set_render_state(D3DRS_ZENABLE, 1), 0);
            assert_eq!(d.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
            assert_eq!(d.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), 0);
            assert_eq!(d.set_render_state(D3DRS_DEPTHBIAS, 0), 0);
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(GREEN, z0)),
                0,
                "stored depth"
            );
            assert_eq!(d.set_render_state(D3DRS_DEPTHBIAS, bias.to_bits()), 0);
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(RED, z0 + gap)),
                0,
                "biased quad"
            );
        });
        h.read_pixel(320, 240) == RED
    };
    for z0 in [0.75_f32, 0.375, 0.1875, 0.046_875] {
        assert!(
            !wins(z0, -0.75 * gap),
            "z0={z0}: a bias under the gap leaves the quad behind"
        );
        assert!(
            wins(z0, -1.5 * gap),
            "z0={z0}: a bias over the gap brings the quad in front"
        );
    }
}

#[test]
fn depth_bias_never_clips_geometry_on_a_depth_plane() {
    // D3D9 biases a fragment after clipping and clamps the result to the depth
    // range, so a quad lying on the far plane under a positive bias, or on the
    // near plane under a negative one, is still drawn. A bias added to the
    // clip-space position alone would push either one out of the clip volume.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    for (z, bias) in [(1.0_f32, 0.1_f32), (0.0, -0.1)] {
        h.render_once(BLACK, |d| {
            assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0, "depth clear");
            assert_eq!(d.set_render_state(D3DRS_ZENABLE, 1), 0);
            assert_eq!(d.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL), 0);
            assert_eq!(d.set_render_state(D3DRS_DEPTHBIAS, bias.to_bits()), 0);
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(GREEN, z)),
                0,
                "biased quad on the plane"
            );
        });
        assert_eq!(
            h.read_pixel(320, 240),
            GREEN,
            "z={z} bias={bias}: the quad survives the bias"
        );
    }
}

#[test]
fn stencil_clear_leaves_depth_intact() {
    // D3DCLEAR_STENCIL alone must not disturb the depth plane it shares on
    // D24S8. Prime depth with a near quad, clear only stencil, then draw a
    // farther quad: it stays rejected, so the near colour survives.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0, "depth clear");
        let near = quad_at_depth(GREEN, 0.25);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &near),
            0,
            "near draw"
        );
        assert_eq!(d.clear(D3DCLEAR_STENCIL, 0, 1.0, 0), 0, "stencil clear");
        let far = quad_at_depth(BLUE, 0.75);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &far),
            0,
            "far draw"
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "depth survived the stencil clear"
    );
}

#[test]
fn combined_depth_stencil_clear_resets_both_planes_mid_frame() {
    // Clear(ZBUFFER | STENCIL) after a draw takes the clear-quad path, and
    // one quad has to reset both planes. Prime depth with a near quad, clear
    // both planes with stencil 1, then draw a farther quad gated on stencil
    // EQUAL 1: it paints only if depth went back to 1.0 and stencil holds 1.
    // A nearer quad gated on stencil 2 then fails the stencil test, which
    // proves the plane holds exactly 1 rather than passing by accident.
    let h = Harness::with_depth();
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(GREEN, 0.25)),
            0,
            "near draw"
        );
        assert_eq!(
            d.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 1),
            0,
            "combined clear"
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, 1), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(BLUE, 0.75)),
            0,
            "far draw behind the primed depth"
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, 2), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad_at_depth(RED, 0.5)),
            0,
            "draw with the wrong reference"
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        BLUE,
        "one quad reset depth to 1.0 and stencil to 1"
    );
}

#[test]
fn stencil_reference_is_compared_through_the_mask() {
    // D3D9 compares (ref & mask) against (stencil & mask), and apps do set a
    // reference wider than the 8-bit attachment, sweeping it across a pass to
    // read stencil values back one at a time. Only the quad whose low byte
    // matches the stored value may paint.
    const STORED: u32 = 0x3;
    let h = Harness::with_depth();
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, STORED),
            0,
            "depth and stencil clear"
        );
        assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILMASK, 0xFF), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP), 0);
        for i in 0..16u32 {
            assert_eq!(d.set_render_state(D3DRS_STENCILREF, 0x0000_FF00 | i), 0);
            let shade = 0xFF00_0000 | (i * 16);
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(shade)),
                0,
                "painter draw {i}"
            );
        }
    });
    assert_eq!(
        h.read_pixel(320, 240),
        0xFF00_0000 | (STORED * 16),
        "only the reference matching the stored value may paint"
    );
}

#[test]
fn stencil_enable_without_a_stencil_attachment_is_dropped() {
    // A depth-only surface with D3DRS_STENCILENABLE left set from an earlier
    // pass must not build a stencil-enabled state: Metal rejects one against a
    // pass with no stencil attachment.
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(mtld3d_types::D3DFMT_D16),
        ..HarnessConfig::default()
    });
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0, "depth clear");
        assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), 0);
        assert_eq!(d.set_render_state(D3DRS_STENCILREF, 0x7F), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(GREEN)),
            0,
            "draw with a stale stencil enable"
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the draw was not stencil-gated"
    );
}

#[test]
fn stencil_clear_on_a_depth_only_surface_succeeds() {
    // D3D9 masks D3DCLEAR_STENCIL against a format with no stencil plane
    // rather than failing the call, so a game that always passes the flag
    // keeps working after binding a depth-only surface.
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(mtld3d_types::D3DFMT_D16),
        ..HarnessConfig::default()
    });
    assert_eq!(h.clear(D3DCLEAR_STENCIL, 0, 1.0, 0x7F), 0, "stencil clear");
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 0x7F),
        0,
        "combined depth+stencil clear"
    );
}

/// A Clear issued after a draw, before any Present, must paint the whole target.
///
/// Inside a pass with draws the clear becomes a full-screen quad, and that
/// quad is drawn under whatever cull mode the previous draw left; D3D's
/// default `CULL_CCW` used to cull it whole, so the second scene rendered over
/// the first one's leftovers instead of the clear colour (the reason a
/// conformance point-size probe's background came out black).
#[test]
fn clear_after_a_draw_in_the_same_pass_is_not_culled() {
    let h = Harness::with_depth();
    arm_diffuse(&h);
    let small = centered_triangle(GREEN);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene (first scene)");
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &small), 0);
    assert_eq!(h.end_scene(), 0, "EndScene (first scene)");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLUE, 1.0, 0),
        0,
        "Clear between the scenes"
    );
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &small), 0);
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(h.read_pixel(320, 240), GREEN, "triangle at the centre");
    assert_eq!(h.read_pixel(72, 64), BLUE, "background is the clear colour");
    assert_eq!(
        h.read_pixel(600, 400),
        BLUE,
        "background is the clear colour"
    );
}

/// A full clip-space quad (two triangles) sampling a texture over 0..1 UVs.
fn textured_fill_quad(color: u32) -> [TexturedVertex; 6] {
    let v = |x: f32, y: f32, u: f32, vt: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color,
        u,
        v: vt,
    };
    [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ]
}

/// Copy `rt` onto the back buffer unchanged, so `read_pixel` sees its stored bytes.
///
/// Point-sampled with `D3DSAMP_SRGBTEXTURE` left at 0, so the texel arrives
/// raw: whatever the sRGB-write test wrote into the target is what lands.
fn blit_rt_to_backbuffer(h: &Harness, rt: &Texture<'_>, backbuffer: &Surface<'_>) {
    assert_eq!(h.set_render_target(0, backbuffer), 0, "restore backbuffer");
    assert_eq!(h.clear_target(BLACK), 0, "clear backbuffer");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(
        h.set_render_state(D3DRS_ALPHABLENDENABLE, 0),
        0,
        "blend off"
    );
    assert_eq!(h.set_texture(0, rt), 0, "bind RT as texture");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF TEX1"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &textured_fill_quad(WHITE)),
        0,
        "RT copy draw"
    );
}

/// An sRGB-capable render target for the `D3DRS_SRGBWRITEENABLE` tests.
fn srgb_render_target(h: &Harness) -> Texture<'_> {
    h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    )
}

/// `D3DRS_SRGBWRITEENABLE` encodes an opaque draw's colour on write.
///
/// Linear 0x80 (0.502) leaves the target holding 0xBC, the value Windows
/// produces for the same draw. Pins the encode itself, independent of where
/// in the pipeline it happens.
#[test]
fn srgb_write_encodes_an_opaque_draw() {
    let h = Harness::new();
    let rt = srgb_render_target(&h);
    let rt_surface = rt.surface_level(0);
    let backbuffer = h.render_target(0);
    arm_diffuse(&h);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.clear_target(BLACK), 0, "clear RT");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 1), 0, "sRGB on");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(0xFF80_8080)),
        0,
        "sRGB-write draw"
    );
    blit_rt_to_backbuffer(&h, &rt, &backbuffer);
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xFFBC_BCBC,
        3,
        "linear 0x80 must be stored sRGB-encoded as ~0xBC",
    );
    assert_eq!(h.clear_texture(0), 0, "unbind RT texture");
}

/// `D3DRS_SRGBWRITEENABLE` blends in linear space and encodes afterwards.
///
/// The D3D9 order the `D3DPMISCCAPS_POSTBLENDSRGBCONVERT` cap promises:
/// the destination is decoded to linear, the blend runs there, and the
/// result is encoded on write. Over a stored 0x80 (linear 0.216), white at
/// half alpha gives linear 0.608, which encodes to ~0xCD. Encoding in the
/// pixel shader instead blends 1.0 against the stored 0.502 in gamma space
/// and lands on ~0xC0.
#[test]
fn srgb_write_blends_in_linear_then_encodes() {
    let h = Harness::new();
    let rt = srgb_render_target(&h);
    let rt_surface = rt.surface_level(0);
    let backbuffer = h.render_target(0);
    arm_diffuse(&h);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    // The destination is written with sRGB writes off, so the target holds
    // 0x80 verbatim: the gamma-space encoding of linear 0.216.
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(h.clear_target(0xFF80_8080), 0, "clear RT mid-grey");

    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 1), 0, "sRGB on");
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0, "blend on");
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(0x80FF_FFFF)),
        0,
        "half-alpha white over the mid-grey"
    );
    blit_rt_to_backbuffer(&h, &rt, &backbuffer);
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    // Alpha carries no transfer function, so it blends to 0.75 either way.
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xBFCD_CDCD,
        3,
        "the blend must run on linear values, not on the encoded ones",
    );
    assert_eq!(h.clear_texture(0), 0, "unbind RT texture");
}

/// The back buffer takes the same linear blend as an offscreen target.
///
/// Source-engine titles blend decals, glass and particles straight onto the
/// swap chain, so the back buffer needs the sRGB view as much as a
/// render-target texture does. Same arithmetic as
/// `srgb_write_blends_in_linear_then_encodes`, without the copy.
#[test]
fn srgb_write_blends_in_linear_on_the_backbuffer() {
    let h = Harness::new();
    arm_diffuse(&h);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(h.clear_target(0xFF80_8080), 0, "clear mid-grey");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 1), 0, "sRGB on");
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0, "blend on");
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(0x80FF_FFFF)),
        0,
        "half-alpha white over the mid-grey"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xBFCD_CDCD,
        3,
        "the back buffer must blend on linear values too",
    );
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(
        h.set_render_state(D3DRS_ALPHABLENDENABLE, 0),
        0,
        "blend off"
    );
}

/// A `CreateRenderTarget` surface takes the same linear blend.
///
/// It carries its own Metal colour texture rather than a texture's, so it
/// needs its own sRGB view; the pixels come back through
/// `GetRenderTargetData` while it is still the bound target.
#[test]
fn srgb_write_blends_in_linear_on_a_standalone_render_target() {
    let h = Harness::new();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let backbuffer = h.render_target(0);
    arm_diffuse(&h);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind RT");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(h.clear_target(0xFF80_8080), 0, "clear RT mid-grey");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 1), 0, "sRGB on");
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0, "blend on");
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(0x80FF_FFFF)),
        0,
        "half-alpha white over the mid-grey"
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_approx(
        h.read_pixel(32, 32),
        0xBFCD_CDCD,
        3,
        "a standalone render target must blend on linear values too",
    );
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), 0, "sRGB off");
    assert_eq!(
        h.set_render_state(D3DRS_ALPHABLENDENABLE, 0),
        0,
        "blend off"
    );
}

/// A device released straight after an sRGB-write present tears down in order.
///
/// With `D3DRS_SRGBWRITEENABLE` set the frame names the back buffer's sRGB
/// twin view as its colour attachment, and `Release` follows the `Present`
/// with nothing in between, so the frame is still with the encoder or the
/// submit thread when the teardown begins. The twin's destroy therefore waits
/// behind the flush and the encoder's GPU-idle wait like every other implicit
/// surface, and only then goes out ahead of the base texture the queue-destroy
/// thunk releases. Each round takes a device of its own, and the second
/// configuration repeats the teardown under the forced Intel storage answers,
/// the hardware where a premature release is visible.
///
/// What this pins: the rounds run to completion, every call answers `D3D_OK`,
/// the release reaches a zero refcount, and the process the suite shares is
/// still alive afterwards, with the unix side's live-texture ledger seeing one
/// destroy per texture (a destroy of a handle it does not hold fails a build
/// with debug assertions). What it cannot pin: a retain taken on a released
/// Metal object is usually silent on an Apple GPU, so a green run is not
/// evidence that the order is right, only that this order is not visibly
/// wrong.
#[test]
fn a_device_released_after_an_srgb_write_present_tears_down_cleanly() {
    const ROUNDS: u32 = 6;
    const FRAMES: u32 = 2;
    const MID_GREY: u32 = 0xFF80_8080;

    for entries in ["", "intel.managedMemory=true;intel.linearAlign256=true"] {
        for round in 0..ROUNDS {
            let h = Harness::create(&HarnessConfig {
                config_entries: entries,
                ..HarnessConfig::default()
            });
            arm_diffuse(&h);
            assert_eq!(
                h.set_render_state(D3DRS_SRGBWRITEENABLE, 1),
                0,
                "sRGB on ({entries:?} round {round})"
            );
            for frame in 0..FRAMES {
                assert_eq!(
                    h.begin_scene(),
                    0,
                    "BeginScene (round {round} frame {frame})"
                );
                assert_eq!(h.clear_target(MID_GREY), 0, "clear mid-grey");
                assert_eq!(
                    h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &fill_quad(WHITE)),
                    0,
                    "sRGB-write draw"
                );
                assert_eq!(h.end_scene(), 0, "EndScene");
                assert_eq!(h.present(), 0, "Present");
            }
            // No readback and no wait: either would retire the frames that the
            // release has to carry through the teardown.
            assert_eq!(
                h.release_device(),
                0,
                "the device is fully released ({entries:?} round {round})"
            );
        }
    }
}
