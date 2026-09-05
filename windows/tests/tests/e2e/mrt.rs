//! Multiple simultaneous render targets.
//!
//! A `ps_3_0` shader writes `oC0` and `oC1` into two render-target textures
//! bound at slots 0 and 1; the pixels are read back through
//! `GetRenderTargetData`. The contract tests pin the slot rules: four slots,
//! slot 0 never null, unbound slots report `D3DERR_NOTFOUND`, `Reset` unbinds
//! everything above slot 0.

use mtld3d_tests::{Harness, PosColorVertex, Rgba8, Surface, TexturedVertex};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DERR_INVALIDCALL, D3DERR_NOTFOUND, D3DFMT_A8R8G8B8, D3DFMT_R32F,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRECT, D3DRS_COLORWRITEENABLE1, D3DRS_LIGHTING, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTADDRESS_CLAMP, D3DTEXF_POINT,
    D3DUSAGE_RENDERTARGET, PrimitiveMiscCaps,
};

const BLACK: u32 = 0xFF00_0000;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;

/// `ps_3_0 { def c0, 0,1,0,1; def c1, 0,0,1,1; mov oC0, c0; mov oC1, c1; }`
///
/// Green to render target 0, blue to render target 1. Tokens follow the
/// `D3DSHADER_PARAM` layout (bit 31 set; register type split across bits
/// `[30:28]` and `[12:11]`; `0xE4` = `.xyzw` swizzle; `0xF` write mask), so
/// the `oC1` destination is `0x800F_0801` and the `c1` source `0xA0E4_0001`.
#[rustfmt::skip]
const PS_TWO_TARGETS: [u32; 20] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x3F80_0000, //   0, 1, 0, 1
    0x0500_0051, 0xA00F_0001,                           // def c1,
    0x0000_0000, 0x0000_0000, 0x3F80_0000, 0x3F80_0000, //   0, 0, 1, 1
    0x0200_0001, 0x800F_0800, 0xA0E4_0000,              // mov oC0, c0
    0x0200_0001, 0x800F_0801, 0xA0E4_0001,              // mov oC1, c1
    0x0000_FFFF,                                        // end
];

/// `ps_3_0 { def c0, 0,1,0,1; mov oC0, c0; }`: green to render target 0 only.
#[rustfmt::skip]
const PS_ONE_TARGET: [u32; 11] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x3F80_0000, //   0, 1, 0, 1
    0x0200_0001, 0x800F_0800, 0xA0E4_0000,              // mov oC0, c0
    0x0000_FFFF,                                        // end
];

/// `ps_3_0 { def c0, 1,1,1,1; mov oC0, c0; mov oC1, c0; }`: white to both.
#[rustfmt::skip]
const PS_WHITE_BOTH: [u32; 14] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x3F80_0000, 0x3F80_0000, 0x3F80_0000, 0x3F80_0000, //   1, 1, 1, 1
    0x0200_0001, 0x800F_0800, 0xA0E4_0000,              // mov oC0, c0
    0x0200_0001, 0x800F_0801, 0xA0E4_0000,              // mov oC1, c0
    0x0000_FFFF,                                        // end
];

/// A clip-space triangle covering the whole target.
const fn full_cover() -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
    ]
}

/// Read one pixel of a DEFAULT-pool render target through `GetRenderTargetData`.
fn read_rt_pixel(h: &Harness, rt: &Surface<'_>, x: u32, y: u32) -> u32 {
    let (hr, desc) = rt.desc();
    assert_eq!(hr, D3D_OK, "GetDesc");
    let sysmem = h.create_offscreen_plain_surface(
        desc.width,
        desc.height,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(rt, &sysmem),
        D3D_OK,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let idx = (y * pitch_px + x) as usize;
    locked.as_u32(idx + 1)[idx]
}

fn assert_color(actual: u32, expected: u32, what: &str) {
    let a = Rgba8::from_pixel(actual);
    let e = Rgba8::from_pixel(expected);
    assert!(
        a.approx_eq(e, 8),
        "{what}: expected {e:?} ({expected:#010x}), got {a:?} ({actual:#010x})"
    );
}

/// Two 64x64 render-target textures, bound at slots 0 and 1, with FF lighting off.
fn two_targets(h: &Harness) -> (mtld3d_tests::Texture<'_>, mtld3d_tests::Texture<'_>) {
    let make = || {
        h.create_texture(
            64,
            64,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A8R8G8B8,
            D3DPOOL_DEFAULT,
        )
    };
    let (rt0, rt1) = (make(), make());
    assert_eq!(
        h.set_render_target(0, &rt0.surface_level(0)),
        D3D_OK,
        "bind slot 0"
    );
    assert_eq!(
        h.set_render_target(1, &rt1.surface_level(0)),
        D3D_OK,
        "bind slot 1"
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    (rt0, rt1)
}

#[test]
fn pixel_shader_writes_each_target() {
    let h = Harness::new();
    let (rt0, rt1) = two_targets(&h);
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear both");
    let ps = h.create_pixel_shader(&PS_TWO_TARGETS);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        GREEN,
        "oC0 lands in slot 0",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        BLUE,
        "oC1 lands in slot 1",
    );
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}

#[test]
fn clear_reaches_every_bound_target_and_an_unwritten_target_keeps_it() {
    let h = Harness::new();
    let (rt0, rt1) = two_targets(&h);
    assert_eq!(h.clear_target(RED), D3D_OK, "clear both red");
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 5, 5),
        RED,
        "slot 1 cleared",
    );
    // The shader writes oC0 only: slot 1 must keep the clear colour.
    let ps = h.create_pixel_shader(&PS_ONE_TARGET);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        GREEN,
        "slot 0 drawn",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        RED,
        "slot 1 untouched",
    );
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}

#[test]
fn color_write_enable1_masks_slot_1_only() {
    let h = Harness::new();
    let (rt0, rt1) = two_targets(&h);
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear both");
    // Red channel only on slot 1; slot 0 keeps the default full mask.
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE1, 0x1), D3D_OK);
    let ps = h.create_pixel_shader(&PS_WHITE_BOTH);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        0xFFFF_FFFF,
        "slot 0 full",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        RED,
        "slot 1 red only",
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE1, 0xF), D3D_OK);
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}

#[test]
fn slot_contract() {
    let h = Harness::new();
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let surface = rt.surface_level(0);
    assert_eq!(
        h.set_render_target(4, &surface),
        D3DERR_INVALIDCALL,
        "slot 4"
    );
    assert_eq!(
        h.clear_render_target(0),
        D3DERR_INVALIDCALL,
        "slot 0 to NULL"
    );
    let (hr, out) = h.render_target_hr(4);
    assert_eq!(hr, D3DERR_INVALIDCALL, "GetRenderTarget(4)");
    assert!(out.is_none());

    let (hr, out) = h.render_target_hr(1);
    assert_eq!(hr, D3DERR_NOTFOUND, "unbound slot 1");
    assert!(out.is_none(), "NOTFOUND leaves the out-pointer null");

    assert_eq!(h.set_render_target(1, &surface), D3D_OK, "bind slot 1");
    let (hr, out) = h.render_target_hr(1);
    assert_eq!(hr, D3D_OK, "bound slot 1");
    assert_eq!(
        out.expect("bound slot returns the surface").as_ptr(),
        surface.as_ptr(),
        "GetRenderTarget(1) returns the bound surface"
    );
    assert_eq!(h.clear_render_target(1), D3D_OK, "slot 1 to NULL");
    let (hr, _) = h.render_target_hr(1);
    assert_eq!(hr, D3DERR_NOTFOUND, "unbound again");
}

#[test]
fn reset_unbinds_the_extra_targets() {
    let h = Harness::new();
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(
        h.set_render_target(1, &rt.surface_level(0)),
        D3D_OK,
        "bind slot 1"
    );
    // A DEFAULT-pool texture blocks Reset while it lives.
    drop(rt);
    assert_eq!(h.reset(640, 480), D3D_OK, "Reset");
    let backbuffer = h.back_buffer(0);
    let rt0 = h.render_target(0);
    assert_eq!(
        rt0.as_ptr(),
        backbuffer.as_ptr(),
        "slot 0 is the new backbuffer"
    );
    let (hr, out) = h.render_target_hr(1);
    assert_eq!(hr, D3DERR_NOTFOUND, "slot 1 unbound by Reset");
    assert!(out.is_none());
}

#[test]
fn mismatched_target_is_cleared_but_not_drawn() {
    let h = Harness::new();
    let rt0 = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt1 = h.create_texture(
        32,
        32,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(
        h.set_render_target(0, &rt0.surface_level(0)),
        D3D_OK,
        "bind slot 0"
    );
    assert_eq!(
        h.set_render_target(1, &rt1.surface_level(0)),
        D3D_OK,
        "bind slot 1"
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    assert_eq!(h.clear(D3DCLEAR_TARGET, RED, 1.0, 0), D3D_OK, "clear both");
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 16, 16),
        RED,
        "smaller slot 1 cleared",
    );
    let ps = h.create_pixel_shader(&PS_TWO_TARGETS);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        GREEN,
        "slot 0 drawn",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 16, 16),
        RED,
        "a target sized unlike slot 0 is left out of the draw",
    );
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}

#[test]
fn caps_advertise_four_targets() {
    let h = Harness::new();
    let caps = h.device_caps();
    assert_eq!(caps.num_simultaneous_rts, 4);
    let misc = PrimitiveMiscCaps::from_bits_truncate(caps.primitive_misc_caps);
    assert!(misc.contains(
        PrimitiveMiscCaps::INDEPENDENTWRITEMASKS
            | PrimitiveMiscCaps::MRTINDEPENDENTBITDEPTHS
            | PrimitiveMiscCaps::MRTPOSTPIXELSHADERBLENDING
    ));
}

#[test]
fn mid_pass_clear_reaches_every_target_without_ending_the_draw_sequence() {
    // Draw, then Clear with the pass still open, then draw again: the clear
    // runs as an in-pass quad writing both targets, and the second draw lands
    // on top of it.
    let h = Harness::new();
    let (rt0, rt1) = two_targets(&h);
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear both");
    let ps = h.create_pixel_shader(&PS_TWO_TARGETS);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "first draw"
    );
    assert_eq!(h.clear_target(RED), D3D_OK, "mid-pass clear");
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        RED,
        "slot 0 cleared mid-pass",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        RED,
        "slot 1 cleared mid-pass",
    );
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "second draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_color(
        read_rt_pixel(&h, &rt0.surface_level(0), 32, 32),
        GREEN,
        "slot 0 drawn after the clear",
    );
    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        BLUE,
        "slot 1 drawn after the clear",
    );
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}

#[test]
fn rect_clear_reaches_every_target_inside_the_rect_only() {
    let h = Harness::new();
    let (rt0, rt1) = two_targets(&h);
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear both");
    let rect = D3DRECT {
        x1: 0,
        y1: 0,
        x2: 32,
        y2: 32,
    };
    assert_eq!(h.clear_target_rects(RED, &[rect]), D3D_OK, "rect clear");
    for (rt, name) in [(&rt0, "slot 0"), (&rt1, "slot 1")] {
        let surface = rt.surface_level(0);
        assert_color(read_rt_pixel(&h, &surface, 8, 8), RED, name);
        assert_color(read_rt_pixel(&h, &surface, 48, 48), BLACK, name);
    }
}

/// `ps_3_0` writing a distinct constant to each of the four color outputs.
///
/// `oC3` carries `36/255` in `.x`, the shape a deferred G-buffer uses for a
/// material-id plane in a single-channel float target: an 8-bit id scaled to
/// `[0, 1]`, decoded later by multiplying with 255.
#[rustfmt::skip]
const PS_FOUR_TARGETS: [u32; 38] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x3F80_0000, //   1, 0, 0, 1
    0x0500_0051, 0xA00F_0001,                           // def c1,
    0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x3F80_0000, //   0, 1, 0, 1
    0x0500_0051, 0xA00F_0002,                           // def c2,
    0x0000_0000, 0x0000_0000, 0x3F80_0000, 0x3F80_0000, //   0, 0, 1, 1
    0x0500_0051, 0xA00F_0003,                           // def c3,
    0x3E10_9091, 0x0000_0000, 0x0000_0000, 0x3F80_0000, //   36/255, 0, 0, 1
    0x0200_0001, 0x800F_0800, 0xA0E4_0000,              // mov oC0, c0
    0x0200_0001, 0x800F_0801, 0xA0E4_0001,              // mov oC1, c1
    0x0200_0001, 0x800F_0802, 0xA0E4_0002,              // mov oC2, c2
    0x0200_0001, 0x800F_0803, 0xA0E4_0003,              // mov oC3, c3
    0x0000_FFFF,                                        // end
];

/// `ps_3_0 { dcl_2d s0; dcl_texcoord0 v0; texld r0, v0, s0; mov oC0, r0.x; }`
///
/// Broadcasts the sampled `.x` to every channel, so the readback below can
/// assert the single-channel float value through an 8-bit target.
#[rustfmt::skip]
const PS_SAMPLE_X: [u32; 15] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
    0x0200_001F, 0x8000_0005, 0x900F_0000,              // dcl_texcoord0 v0
    0x0300_0042, 0x800F_0000, 0x90E4_0000, 0xA0E4_0800, // texld r0, v0, s0
    0x0200_0001, 0x800F_0800, 0x8000_0000,              // mov oC0, r0.x
    0x0000_FFFF,                                        // end
];

#[test]
fn four_targets_with_an_r32f_extra_written_and_sampled() {
    // The deferred G-buffer shape: three 8-bit color planes plus one R32F
    // plane bound as slot 3, all written by one ps_3_0 draw, and the float
    // plane sampled by a later pass (a lighting shader decoding the material
    // id it carries). The extras sidecar must route `oC3` into the R32F
    // target across the format mix, and the follow-up pass must read back
    // the exact stored value.
    let h = Harness::new();
    let make8 = || {
        h.create_texture(
            64,
            64,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A8R8G8B8,
            D3DPOOL_DEFAULT,
        )
    };
    let (rt0, rt1, rt2) = (make8(), make8(), make8());
    let rt3 = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_R32F,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &rt0.surface_level(0)), D3D_OK);
    assert_eq!(h.set_render_target(1, &rt1.surface_level(0)), D3D_OK);
    assert_eq!(h.set_render_target(2, &rt2.surface_level(0)), D3D_OK);
    assert_eq!(h.set_render_target(3, &rt3.surface_level(0)), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear all four");

    let ps = h.create_pixel_shader(&PS_FOUR_TARGETS);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "G-buffer draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);

    assert_color(
        read_rt_pixel(&h, &rt1.surface_level(0), 32, 32),
        GREEN,
        "oC1",
    );
    assert_color(
        read_rt_pixel(&h, &rt2.surface_level(0), 32, 32),
        BLUE,
        "oC2",
    );

    // Second pass: extras unbound, the float plane becomes a sampler input.
    assert_eq!(h.set_render_target(0, &backbuffer), D3D_OK);
    for slot in 1..=3 {
        assert_eq!(h.clear_render_target(slot), D3D_OK, "unbind extra");
    }
    let ps_sample = h.create_pixel_shader(&PS_SAMPLE_X);
    assert_eq!(h.set_pixel_shader(&ps_sample), D3D_OK);
    assert_eq!(h.set_texture(0, &rt3), D3D_OK, "bind the R32F plane");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), D3D_OK, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.clear_target(BLACK), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        D3D_OK,
        "sample draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (33..=39).contains(&center.r) && (33..=39).contains(&center.g),
        "the R32F extra should hold 36/255 and sample back as gray 36, got {center:?}"
    );
    let corner = Rgba8::from_pixel(h.read_pixel(10, 10));
    assert!(
        corner.r < 20,
        "outside the quad stays the cleared black, got {corner:?}"
    );

    assert_eq!(h.clear_pixel_shader(), D3D_OK);
    assert_eq!(h.clear_texture(0), D3D_OK);
}

/// A clip-space triangle covering the bottom-left corner only.
const fn corner_cover() -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: -0.5,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: -0.5,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
    ]
}

#[test]
fn a_clear_stays_ahead_of_a_pass_that_draws_into_its_target_as_an_extra() {
    // Clear(rt) with nothing drawn, then an MRT pass that carries rt at slot 1
    // and writes `oC1` into it, then a pass that rebinds rt at slot 0 and draws
    // into a corner. The clear must land before the MRT writes, not on top of
    // them: the centre pixel belongs to the MRT pass.
    let h = Harness::new();
    let make = || {
        h.create_texture(
            64,
            64,
            1,
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A8R8G8B8,
            D3DPOOL_DEFAULT,
        )
    };
    let (rt, other) = (make(), make());
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    assert_eq!(
        h.set_render_target(0, &rt.surface_level(0)),
        D3D_OK,
        "bind rt at slot 0"
    );
    assert_eq!(h.clear_target(RED), D3D_OK, "clear rt");

    assert_eq!(
        h.set_render_target(0, &other.surface_level(0)),
        D3D_OK,
        "bind the other target at slot 0"
    );
    assert_eq!(
        h.set_render_target(1, &rt.surface_level(0)),
        D3D_OK,
        "bind rt at slot 1"
    );
    let two = h.create_pixel_shader(&PS_TWO_TARGETS);
    assert_eq!(h.set_pixel_shader(&two), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &full_cover()),
        D3D_OK,
        "MRT draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);

    assert_eq!(h.clear_render_target(1), D3D_OK, "unbind slot 1");
    assert_eq!(
        h.set_render_target(0, &rt.surface_level(0)),
        D3D_OK,
        "rebind rt at slot 0"
    );
    let one = h.create_pixel_shader(&PS_ONE_TARGET);
    assert_eq!(h.set_pixel_shader(&one), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &corner_cover()),
        D3D_OK,
        "corner draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);

    let surface = rt.surface_level(0);
    assert_color(
        read_rt_pixel(&h, &surface, 32, 32),
        BLUE,
        "the MRT pass's oC1 survives",
    );
    assert_color(
        read_rt_pixel(&h, &surface, 4, 60),
        GREEN,
        "the corner draw lands on top",
    );
    assert_eq!(h.clear_pixel_shader(), D3D_OK);
}
