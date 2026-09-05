//! Sampler states: addressing modes, filtering, and get/set round-trips.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex, assert_pixel_approx, assert_pixel_eq};
use mtld3d_types::{
    D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE,
    D3DFVF_TEX1, D3DFVF_XYZ, D3DPT_TRIANGLELIST, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND,
    D3DRS_SRCBLEND, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_BORDERCOLOR, D3DSAMP_MAGFILTER,
    D3DSAMP_MAXANISOTROPY, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DSAMP_MIPMAPLODBIAS,
    D3DSAMP_SRGBTEXTURE, D3DTA_TEXTURE, D3DTADDRESS_BORDER, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP,
    D3DTEXF_LINEAR, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP,
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
fn white_border_colour_reads_as_white() {
    // Opaque white is one of Metal's three border presets, so BORDER
    // addressing with D3DSAMP_BORDERCOLOR = 0xFFFFFFFF reads white past the
    // unit square (the classic shadow-map border: outside the light frustum
    // counts as lit).
    let h = Harness::new();
    if h.device_caps().texture_address_caps & mtld3d_types::AddressCaps::BORDER.bits() == 0 {
        // The device cannot create border-colour samplers (virtualized CI
        // devices); the cap is stripped and a title would not use BORDER.
        return;
    }
    let tex = rgbw_2x2(&h);
    let quad = uv_quad(2.0);

    arm_texture(&h, &tex, D3DTADDRESS_BORDER, D3DTEXF_POINT);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_BORDERCOLOR, 0xFFFF_FFFF),
        0,
        "border colour stored"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });

    let px = Rgba8::from_pixel(h.read_pixel(PROBE_X, PROBE_Y));
    assert!(
        px.r > 215 && px.g > 215 && px.b > 215,
        "border is Metal's opaque-white preset, got {px:?}",
    );
}

#[test]
fn non_preset_border_colour_falls_back_to_black() {
    // Metal samplers support only preset border colours (transparent / opaque
    // black / white), not an arbitrary D3DSAMP_BORDERCOLOR. BORDER addressing is
    // applied (out-of-range texels read as the border, distinct from CLAMP's
    // edge texel) but a colour outside the presets falls back to opaque black.
    // Pinned as a Metal limitation; D3DSAMP_BORDERCOLOR still round-trips above.
    let h = Harness::new();
    if h.device_caps().texture_address_caps & mtld3d_types::AddressCaps::BORDER.bits() == 0 {
        // The device cannot create border-colour samplers (virtualized CI
        // devices); the cap is stripped and a title would not use BORDER.
        return;
    }
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
        "non-preset border colour falls back to Metal's opaque-black preset, got {px:?}",
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

/// `D3DSAMP_SRGBTEXTURE=1` decodes the sampled texel from sRGB to linear.
///
/// A mid-gray 0x80 texel (0.502 sRGB-encoded) decodes to linear ~0.216
/// (0x37). Source-engine games gate their whole gamma-correct pipeline on
/// this decode — without it Half-Life 2 drops to an untested shader-gamma
/// fallback that renders its lightmaps black. The state must also take
/// effect on a mid-scene flip over an unchanged texture bind: the decode
/// lives in which texture view is bound, so the flip has to re-emit the
/// bind, not just the sampler.
#[test]
fn srgbtexture_decodes_on_sample() {
    let h = Harness::new();
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[0xFF80_8080]);
    let quad = uv_quad(1.0);

    arm_texture(&h, &tex, D3DTADDRESS_CLAMP, D3DTEXF_POINT);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_eq(
        h.read_pixel(320, 240),
        0xFF80_8080,
        "SRGBTEXTURE=0 must return the raw texel",
    );

    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        assert_eq!(d.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 1), 0);
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xFF37_3737,
        2,
        "mid-scene SRGBTEXTURE=1 flip must decode 0x80 to linear ~0x37",
    );
    assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 0), 0);
}

/// The sRGB twin view of an X8R8G8B8 texture keeps the alpha=1 swizzle.
///
/// The texel's X byte is 0, so a twin view that dropped the swizzle samples
/// alpha 0 and the SRCALPHA/ZERO blend turns the quad black; a missing
/// decode returns the raw 0xBB instead of linear ~0x7F.
#[test]
fn srgbtexture_x8_twin_keeps_alpha_swizzle() {
    let h = Harness::new();
    if h.device_is_paravirtual() {
        // The paravirtual device samples a swizzle view through the base
        // texture's lanes, so the lane this format fills by swizzle reads the
        // stored byte there.
        return;
    }
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_X8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[0x00BB_BBBB]);
    let quad = uv_quad(1.0);

    arm_texture(&h, &tex, D3DTADDRESS_CLAMP, D3DTEXF_POINT);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 1), 0);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ZERO), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xFF7F_7F7F,
        2,
        "X8R8G8B8 with SRGBTEXTURE=1 must decode 0xBB to ~0x7F at full alpha",
    );
}

/// `ps_3_0 { dcl_2d s0; dcl_texcoord0 v0; texld r0, v0, s0; mov oC0, r0; }`
///
/// Token layout as in `render_target.rs`: bit 31 set, register type split
/// across bits `[30:28]` and `[12:11]`, `0xE4` the `.xyzw` swizzle and `0xF`
/// the write mask. `texld` computes its own LOD, so the sampler bias applies.
#[rustfmt::skip]
const PS_SAMPLE_TEXTURE: [u32; 15] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
    0x0200_001F, 0x8000_0005, 0x900F_0000,              // dcl_texcoord0 v0
    0x0300_0042, 0x800F_0000, 0x90E4_0000, 0xA0E4_0800, // texld r0, v0, s0
    0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
    0x0000_FFFF,                                        // end
];

/// Base dimension of the mip-tinted texture, and the pixel span it is drawn at.
const MIP_TEX_DIM: u32 = 64;

/// One flat colour per mip level of a `MIP_TEX_DIM` chain (64 → 1, 7 levels).
const MIP_TINTS: [u32; 7] = [
    0xFFFF_0000, // 0: red
    0xFF00_FF00, // 1: green
    0xFF00_00FF, // 2: blue
    0xFFFF_FF00, // 3: yellow
    0xFFFF_00FF, // 4: magenta
    0xFF00_FFFF, // 5: cyan
    0xFFFF_FFFF, // 6: white
];

/// A full mip chain whose every level is a different solid colour.
///
/// Reading back the drawn pixel therefore names the level the sampler picked.
fn mip_tinted_texture(h: &Harness) -> Texture<'_> {
    let tex = h.create_texture(MIP_TEX_DIM, MIP_TEX_DIM, 0, 0, D3DFMT_A8R8G8B8, 0);
    assert_eq!(
        usize::try_from(tex.level_count()).expect("level count fits usize"),
        MIP_TINTS.len(),
        "64x64 full mip chain"
    );
    for (level, &tint) in MIP_TINTS.iter().enumerate() {
        let level = u32::try_from(level).expect("level fits u32");
        let dim = usize::try_from(MIP_TEX_DIM >> level).expect("mip dim fits usize");
        tex.lock_rect(level, 0).write_u32(&vec![tint; dim * dim]);
    }
    tex
}

/// A quad covering exactly `MIP_TEX_DIM` backbuffer pixels in both axes.
///
/// One texel per pixel puts the implicit LOD at 0, so the level the sampler
/// picks is the bias, and nothing else.
fn texel_to_pixel_quad() -> [TexturedVertex; 6] {
    const W: u32 = 0xFFFF_FFFF;
    let dim = f32::from(u16::try_from(MIP_TEX_DIM).expect("mip texture dim fits u16"));
    let x1 = 2.0f32.mul_add(dim / 640.0, -1.0);
    let y1 = 2.0f32.mul_add(-dim / 480.0, 1.0);
    let corner = |x: f32, y: f32, u: f32, v: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: W,
        u,
        v,
    };
    [
        corner(-1.0, 1.0, 0.0, 0.0),
        corner(x1, 1.0, 1.0, 0.0),
        corner(-1.0, y1, 0.0, 1.0),
        corner(x1, 1.0, 1.0, 0.0),
        corner(x1, y1, 1.0, 1.0),
        corner(-1.0, y1, 0.0, 1.0),
    ]
}

/// Draw the mip-tinted quad at `bias` and read the colour back.
fn sample_at_bias(h: &Harness, bias: f32) -> u32 {
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MIPMAPLODBIAS, bias.to_bits()),
        0,
        "SetSamplerState(MIPMAPLODBIAS)"
    );
    assert_eq!(
        h.sampler_state(0, D3DSAMP_MIPMAPLODBIAS),
        bias.to_bits(),
        "MIPMAPLODBIAS round-trip"
    );
    let quad = texel_to_pixel_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    h.read_pixel(MIP_TEX_DIM / 2, MIP_TEX_DIM / 2)
}

fn arm_mip_tinted(h: &Harness, tex: &Texture<'_>) {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler {state}");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
}

#[test]
fn mipmap_lod_bias_shifts_the_sampled_mip() {
    // The fixed-function cascade samples through the bias: at one texel per
    // pixel the implicit LOD is 0, so a bias of 2 must move the sample two
    // levels coarser and read that level's tint instead.
    let h = Harness::new();
    let tex = mip_tinted_texture(&h);
    arm_mip_tinted(&h, &tex);

    let unbiased = sample_at_bias(&h, 0.0);
    let biased = sample_at_bias(&h, 2.0);

    assert_eq!(unbiased, MIP_TINTS[0], "no bias samples the base level");
    assert_eq!(biased, MIP_TINTS[2], "a +2 bias samples two levels coarser");
    assert_ne!(unbiased, biased, "the bias must change the sampled mip");
}

#[test]
fn render_lod_bias_keeps_the_base_level_under_the_scale() {
    // Under `render.scale` the sampler derives its LOD from the render grid:
    // at a half scale a quad drawn at one texel per presented pixel covers
    // half a texel per render pixel, so the implicit LOD is 1 and the sample
    // lands one level coarser than the presented size warrants. The default
    // `render.lodBias` adds `log2(scale)` to every sampled stage, so the
    // unbiased sample reads the base level again and the game's own bias
    // still lands where it says.
    //
    // Pins its own scale (a clean half, so the LOD is exactly 1) rather than
    // inheriting the run's: at the identity there is nothing to compensate,
    // and this has to fail in the ordinary `make test` if it regresses.
    let h = Harness::with_config("render.scale=0.5");
    let tex = mip_tinted_texture(&h);
    arm_mip_tinted(&h, &tex);

    let unbiased = sample_at_bias(&h, 0.0);
    let biased = sample_at_bias(&h, 2.0);

    assert_eq!(
        unbiased, MIP_TINTS[0],
        "the compensation cancels the render grid's LOD"
    );
    assert_eq!(
        biased, MIP_TINTS[2],
        "the game's +2 bias still lands two levels coarser"
    );
}

#[test]
fn render_lod_bias_off_leaves_the_mip_to_the_render_grid() {
    // With the key off the sampler follows the render grid: at a half scale
    // the unbiased sample reads level 1, which is what the game would get at
    // that resolution natively.
    let h = Harness::with_config("render.scale=0.5;render.lodBias=false");
    let tex = mip_tinted_texture(&h);
    arm_mip_tinted(&h, &tex);

    let unbiased = sample_at_bias(&h, 0.0);
    assert_eq!(
        unbiased, MIP_TINTS[1],
        "the render grid's LOD picks level 1"
    );
}

#[test]
fn mipmap_lod_bias_shifts_a_programmable_shader_sample() {
    // Same contract through a `ps_3_0` `texld`: the bias is sampler state, not
    // a fixed-function feature, so it reaches the programmable emitter too.
    let h = Harness::new();
    let tex = mip_tinted_texture(&h);
    arm_mip_tinted(&h, &tex);
    let ps = h.create_pixel_shader(&PS_SAMPLE_TEXTURE);
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");

    let unbiased = sample_at_bias(&h, 0.0);
    let biased = sample_at_bias(&h, 3.0);

    assert_eq!(unbiased, MIP_TINTS[0], "no bias samples the base level");
    assert_eq!(
        biased, MIP_TINTS[3],
        "a +3 bias samples three levels coarser"
    );
    assert_eq!(h.clear_pixel_shader(), 0, "SetPixelShader(null)");
}
