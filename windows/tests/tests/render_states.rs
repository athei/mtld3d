//! Render-state *execution*: blend, colour-write masking, scissor, culling.
//!
//! Plus get/set round-trips and spec-default verification.

use mtld3d_tests::{Harness, PosColorVertex, Rgba8};
use mtld3d_types::{
    D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLENDOP_ADD, D3DCMP_EQUAL,
    D3DCULL_CCW, D3DCULL_CW, D3DCULL_NONE, D3DFILL_SOLID, D3DFILL_WIREFRAME, D3DFVF_DIFFUSE,
    D3DFVF_XYZ, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_ALPHABLENDENABLE, D3DRS_BLENDOP,
    D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_DESTBLEND, D3DRS_FILLMODE, D3DRS_LIGHTING,
    D3DRS_SCISSORTESTENABLE, D3DRS_SRCBLEND, D3DRS_STENCILENABLE, D3DRS_STENCILFUNC,
    D3DRS_STENCILMASK, D3DRS_STENCILPASS, D3DRS_STENCILREF, D3DSTENCILOP_REPLACE,
    render_state_defaults,
};

const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;

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
    // The stencil render states store and read back, but mtld3d does not yet
    // translate them into a Metal stencil descriptor (create_depth_stencil_state
    // sets only depth compare + write), so stencil testing does not gate
    // rendering — a documented limitation, not exercised by the target workload.
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
fn wireframe_fill_mode_is_a_noop() {
    // Metal has no native wireframe fill; mtld3d classifies D3DFILL_WIREFRAME as
    // an unimplemented port-candidate and renders solid. Pin that: the interior
    // stays filled in both modes (no target workload uses wireframe).
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
        GREEN,
        "wireframe is a no-op — interior still filled"
    );
}
