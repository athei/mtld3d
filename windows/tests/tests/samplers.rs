//! Sampler states: addressing modes, filtering, and get/set round-trips.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex};
use mtld3d_types::{
    D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPT_TRIANGLELIST, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_BORDERCOLOR, D3DSAMP_MAGFILTER, D3DSAMP_MAXANISOTROPY,
    D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTADDRESS_BORDER, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP,
    D3DTEXF_LINEAR, D3DTEXF_POINT,
};

const BLACK: u32 = 0xFF00_0000;
const YELLOW: u32 = 0xFFFF_FF00;

// Pixel (400,60) on a 640×480 target with UVs spanning 0..2 samples u≈1.25,
// v≈0.25 — where CLAMP (→ column 1) and WRAP (→ column 0) hit different texels,
// and u>1 selects the border under BORDER addressing.
const PROBE_X: u32 = 400;
const PROBE_Y: u32 = 60;

/// A 2×2 texture: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
fn rgbw_2x2(h: &Harness) -> Texture<'_> {
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0)
        .write_u32(&[0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF]);
    tex
}

/// A full-backbuffer quad whose UVs span `0..uv_max` in both axes.
const fn uv_quad(uv_max: f32) -> [TexturedVertex; 6] {
    const W: u32 = 0xFFFF_FFFF;
    let m = uv_max;
    [
        TexturedVertex {
            x: -1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: m,
            v: 0.0,
        },
        TexturedVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: m,
        },
        TexturedVertex {
            x: 1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: m,
            v: 0.0,
        },
        TexturedVertex {
            x: 1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: m,
            v: m,
        },
        TexturedVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: m,
        },
    ]
}

fn arm_texture(h: &Harness, tex: &Texture<'_>, address: u32, filter: u32) {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MINFILTER, filter), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAGFILTER, filter), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_ADDRESSU, address), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_ADDRESSV, address), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
}

#[test]
fn sampler_state_round_trips() {
    let h = Harness::new();
    for (state, value) in [
        (D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
        (D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_LINEAR),
        (D3DSAMP_MAXANISOTROPY, 8),
        (D3DSAMP_BORDERCOLOR, 0xFFFF_FF00),
    ] {
        assert_eq!(
            h.set_sampler_state(0, state, value),
            0,
            "SetSamplerState {state}"
        );
        assert_eq!(
            h.sampler_state(0, state),
            value,
            "GetSamplerState {state} round-trip"
        );
    }
}

#[test]
fn clamp_and_wrap_addressing_differ() {
    // Sampling beyond u,v = 1 must depend on the addressing mode.
    let h = Harness::new();
    let tex = rgbw_2x2(&h);
    let quad = uv_quad(2.0);

    arm_texture(&h, &tex, D3DTADDRESS_CLAMP, D3DTEXF_POINT);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let clamp = h.read_pixel(PROBE_X, PROBE_Y);

    arm_texture(&h, &tex, D3DTADDRESS_WRAP, D3DTEXF_POINT);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let wrap = h.read_pixel(PROBE_X, PROBE_Y);

    assert_ne!(
        clamp, wrap,
        "CLAMP and WRAP must sample differently past the unit square"
    );
}

#[test]
fn border_addressing_uses_metal_black_preset() {
    // Metal samplers support only preset border colours (transparent / opaque
    // black / white), not an arbitrary D3DSAMP_BORDERCOLOR. BORDER addressing is
    // applied (out-of-range texels read as the border, distinct from CLAMP's
    // edge texel) but the requested colour is ignored — the border reads black.
    // Pinned as a Metal limitation; D3DSAMP_BORDERCOLOR still round-trips above.
    let h = Harness::new();
    let tex = rgbw_2x2(&h);
    let quad = uv_quad(2.0);

    arm_texture(&h, &tex, D3DTADDRESS_BORDER, D3DTEXF_POINT);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_BORDERCOLOR, YELLOW),
        0,
        "border colour stored"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });

    let px = Rgba8::from_pixel(h.read_pixel(PROBE_X, PROBE_Y));
    assert!(
        px.r < 40 && px.g < 40 && px.b < 40,
        "border is Metal's black preset (arbitrary BORDERCOLOR unsupported), got {px:?}",
    );
}

#[test]
fn point_and_linear_filtering_differ() {
    // At a texel boundary, point picks one texel; linear blends neighbours.
    let h = Harness::new();
    let tex = rgbw_2x2(&h);
    let quad = uv_quad(1.0);

    arm_texture(&h, &tex, D3DTADDRESS_CLAMP, D3DTEXF_POINT);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let point = h.read_pixel(320, 240); // dead centre — texel boundary

    arm_texture(&h, &tex, D3DTADDRESS_CLAMP, D3DTEXF_LINEAR);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let linear = h.read_pixel(320, 240);

    assert_ne!(
        point, linear,
        "LINEAR must blend where POINT snaps to a texel"
    );
}
