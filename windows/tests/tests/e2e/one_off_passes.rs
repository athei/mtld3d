//! One-off passes against a target other than the bound one.
//!
//! A `ColorFill`, a scaling `StretchRect` and a `Clear` that reaches a render
//! target sized unlike target 0 each bind their own target for one pass and
//! then put the device's bindings back. These tests pin both halves: what the
//! pass writes, and that the depth attachment the device had bound (its mip
//! level and its sample count) is the one the next draw tests against.

use mtld3d_tests::{
    Harness, HarnessConfig, PosColorVertex, Reading, Rgba8, RhwVertex, Surface, assert_or_reread,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_LESSEQUAL, D3DFMT_A8R8G8B8, D3DFMT_D24S8,
    D3DFMT_INTZ, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DLOCK_READONLY,
    D3DMULTISAMPLE_4_SAMPLES, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DRS_LIGHTING, D3DRS_POINTSIZE, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DTEXF_LINEAR, D3DTEXF_NONE, D3DUSAGE_DEPTHSTENCIL,
};

use super::msaa::{RT_SIZE, sample_intz};

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const WHITE: u32 = 0xFFFF_FFFF;
const BLACK: u32 = 0xFF00_0000;

/// Edge of the multisampled render targets the `ColorFill` tests paint.
const MS_SIZE: u32 = 64;

/// A single triangle covering the whole viewport at depth `z`, in `color`.
const fn fullscreen(z: f32, color: u32) -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z,
            color,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z,
            color,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z,
            color,
        },
    ]
}

/// A pre-transformed triangle with its right angle at the origin and legs `size` long.
const fn corner(size: f32, color: u32) -> [RhwVertex; 3] {
    [
        RhwVertex {
            x: 0.0,
            y: 0.0,
            z: 0.5,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: size,
            y: 0.0,
            z: 0.5,
            rhw: 1.0,
            color,
        },
        RhwVertex {
            x: 0.0,
            y: size,
            z: 0.5,
            rhw: 1.0,
            color,
        },
    ]
}

/// Arm an unlit vertex-colour draw with a LESSEQUAL depth test that writes depth.
fn arm_depth_test(h: &Harness) {
    assert_eq!(
        h.set_render_state(D3DRS_ZENABLE, 1),
        D3D_OK,
        "depth test on"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZWRITEENABLE, 1),
        D3D_OK,
        "depth write on"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL),
        D3D_OK,
        "ZFUNC"
    );
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    // Lighting defaults on and the vertices carry no normal, which would
    // light every draw black; the tests read the vertex colour.
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );
}

/// Pixel `(x, y)` of a single-sampled render target, read through `GetRenderTargetData`.
fn surface_pixel(h: &Harness, surface: &Surface<'_>, x: u32, y: u32) -> u32 {
    let (hr, desc) = surface.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the read-back source");
    let sysmem = h.create_offscreen_plain_surface(
        desc.width,
        desc.height,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(surface, &sysmem),
        D3D_OK,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let index = (y * pitch_px + x) as usize;
    locked.as_u32(index + 1)[index]
}

/// Pixel `(x, y)` of a multisampled surface after a `StretchRect` resolve.
///
/// D3D9 rejects `GetRenderTargetData` on a multisampled source; the resolve
/// into a single-sampled target of the same size is the step it asks for.
fn resolved_pixel(h: &Harness, source: &Surface<'_>, x: u32, y: u32) -> u32 {
    let (hr, desc) = source.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the multisampled source");
    let plain = h.create_render_target(desc.width, desc.height, D3DFMT_X8R8G8B8);
    assert_eq!(
        h.stretch_rect(source, &plain, D3DTEXF_NONE),
        D3D_OK,
        "StretchRect resolve"
    );
    surface_pixel(h, &plain, x, y)
}

/// The depth the probe's INTZ is cleared to before the RESZ, so a RESZ that never ran reads it.
const PROBE_PRIMER: f32 = 0.5;

/// Which depth sample zero of the bound 4x surface holds at the centre, as a RESZ reads it.
///
/// For the failure report of the multisampled `ColorFill` test, after its
/// colour readings. A fresh INTZ is bound as depth and cleared to
/// [`PROBE_PRIMER`], the surface goes back, a RESZ copies its sample zero
/// into the INTZ, and the INTZ is sampled back at its centre. The text gives
/// the read and names the value of the test's scene, or the primer, that its
/// eight-bit depth matches. A step that fails panics with its own message.
fn depth_probe(h: &Harness) -> String {
    let depth = h.depth_stencil_surface().expect("a depth surface is bound");
    let (hr, desc) = depth.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the depth surface");
    let intz = h.create_texture(
        desc.width,
        desc.height,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    // Primed through a target of the INTZ's own size: binding a smaller one
    // would cut the clear's viewport to that size and miss the centre.
    let prime_rt = h.create_render_target(desc.width, desc.height, D3DFMT_A8R8G8B8);
    let colour = h.render_target(0);
    assert_eq!(
        h.set_render_target(0, &prime_rt),
        D3D_OK,
        "bind the priming target"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&intz.surface_level(0)),
        D3D_OK,
        "bind the INTZ as depth"
    );
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, 0, PROBE_PRIMER, 0),
        D3D_OK,
        "clear the INTZ to the primer"
    );
    assert_eq!(
        h.set_render_target(0, &colour),
        D3D_OK,
        "rebind the render target"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&depth),
        D3D_OK,
        "rebind the depth surface"
    );
    assert_eq!(h.set_texture(0, &intz), D3D_OK, "bind the RESZ destination");
    assert_eq!(
        h.set_render_state(D3DRS_POINTSIZE, 0x7fa0_5000),
        D3D_OK,
        "the RESZ magic value"
    );
    let sample_rt = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8);
    let read = Rgba8::from_pixel(sample_intz(h, &sample_rt, &intz));
    let value = f32::from(read.r) / 255.0;
    let matched = [
        (0.2, "the near draw's 0.2"),
        (1.0, "the clear's 1.0"),
        (0.8, "the far draw's 0.8"),
        (PROBE_PRIMER, "the primer: the RESZ did not run"),
    ]
    .into_iter()
    .find(|(depth, _)| (value - depth).abs() <= 2.0 / 255.0)
    .map_or("none of 1.0, 0.2 and 0.8", |(_, name)| name);
    format!("sample zero at the centre reads {read:?} (depth {value:.3}), {matched}")
}

/// A far draw must lose against level 1 of a depth texture across `interrupt`.
///
/// Level 1 of a 1280x960 INTZ chain is the back buffer's size and is bound
/// beside it; level 0 is given a depth behind the far draw first, so a draw
/// that tests against level 0 instead passes. A near red draw, then
/// `interrupt` (a one-off pass against another target), then a far green
/// draw: the green must fail the depth test the red one wrote into level 1.
///
/// Level 0 gets its depth from a draw, not from a `Clear` alone. A clear with
/// no draw after it is a clear-only pass, and the pass optimiser moves that
/// clear onto the next pass attaching the same depth texture, whatever its
/// level: the level-1 pass would take it and level 0 would keep the zeros it
/// was created with, which the far draw fails against on either level.
fn far_draw_loses_in_depth_level_1_across(interrupt: impl Fn(&Harness), context: &str) {
    let h = Harness::new();
    let backbuffer = h.back_buffer(0);
    let depth_tex = h.create_texture(
        1280,
        960,
        2,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let level0 = depth_tex.surface_level(0);
    let level1 = depth_tex.surface_level(1);

    let base_sized = h.create_render_target(1280, 960, D3DFMT_X8R8G8B8);
    assert_eq!(
        h.set_render_target(0, &base_sized),
        D3D_OK,
        "level-0-sized target"
    );
    assert_eq!(h.set_depth_stencil_surface(&level0), D3D_OK, "bind level 0");
    arm_depth_test(&h);
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3D_OK,
        "level 0 at the far plane"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen(0.9, WHITE)),
        D3D_OK,
        "level 0 behind the far draw"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");

    assert_eq!(h.set_render_target(0, &backbuffer), D3D_OK, "back buffer");
    assert_eq!(h.set_depth_stencil_surface(&level1), D3D_OK, "bind level 1");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3D_OK,
        "clear colour and level 1"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen(0.3, RED)),
        D3D_OK,
        "near draw"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    interrupt(&h);
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen(0.7, GREEN)),
        D3D_OK,
        "far draw"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        center.g < 40,
        "{context}: the far draw loses the depth test in level 1, got {center:?}"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK);
}

#[test]
fn color_fill_leaves_the_depth_mip_level_bound() {
    far_draw_loses_in_depth_level_1_across(
        |h| {
            let other = h.create_render_target(64, 64, D3DFMT_X8R8G8B8);
            assert_eq!(h.color_fill_hr(&other, BLUE), D3D_OK, "ColorFill");
        },
        "after a ColorFill of another target",
    );
}

#[test]
fn scaling_stretch_rect_leaves_the_depth_mip_level_bound() {
    far_draw_loses_in_depth_level_1_across(
        |h| {
            let src = h.create_render_target(32, 32, D3DFMT_X8R8G8B8);
            let dst = h.create_render_target(64, 64, D3DFMT_X8R8G8B8);
            assert_eq!(
                h.stretch_rect(&src, &dst, D3DTEXF_LINEAR),
                D3D_OK,
                "scaling StretchRect"
            );
        },
        "after a scaling StretchRect between two other targets",
    );
}

#[test]
fn clear_of_a_target_outside_the_pass_leaves_the_depth_mip_level_bound() {
    far_draw_loses_in_depth_level_1_across(
        |h| {
            // Sized unlike target 0, so the pass leaves it out and the
            // `Clear` reaches it through a scoped pass of its own.
            let small = h.create_render_target(32, 32, D3DFMT_X8R8G8B8);
            assert_eq!(h.set_render_target(1, &small), D3D_OK, "bind slot 1");
            assert_eq!(
                h.clear(D3DCLEAR_TARGET, BLACK, 1.0, 0),
                D3D_OK,
                "Clear both targets"
            );
            assert_eq!(h.clear_render_target(1), D3D_OK, "unbind slot 1");
        },
        "after a Clear that reached a target outside the pass",
    );
}

#[test]
fn color_fill_leaves_a_multisampled_depth_attachment_bound() {
    // The depth surface of a 4x swap chain is 4x too. A `ColorFill` binds its
    // own single-sampled target without depth for one pass; the depth
    // attachment it puts back has to be declared 4x again, or every later
    // pass drops it as disagreeing with the colour target.
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        multi_sample_type: D3DMULTISAMPLE_4_SAMPLES,
        ..HarnessConfig::default()
    });
    let other = h.create_render_target(64, 64, D3DFMT_X8R8G8B8);
    arm_depth_test(&h);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3D_OK,
        "clear colour and depth"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen(0.2, WHITE)),
        D3D_OK,
        "near draw"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.color_fill_hr(&other, RED), D3D_OK, "ColorFill");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen(0.8, BLUE)),
        D3D_OK,
        "far draw"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");

    // A wrong centre is read twice more and the depth attachment probed
    // before the test fails, so the report names the stage that lost the
    // draws. A right one returns before any of that runs.
    let (width, height) = h.dims();
    let at = (width / 2, height / 2);
    let backbuffer = h.back_buffer(0);
    let (hr, desc) = backbuffer.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the back buffer");
    let plain = h.create_render_target(desc.width, desc.height, D3DFMT_X8R8G8B8);
    let read = || surface_pixel(&h, &plain, at.0, at.1);
    let resolve_and_read = || {
        assert_eq!(
            h.stretch_rect(&backbuffer, &plain, D3DTEXF_NONE),
            D3D_OK,
            "StretchRect resolve"
        );
        read()
    };
    let reading = |pixel: u32| {
        let center = Rgba8::from_pixel(pixel);
        Reading::described(format!("{center:?}"), center.r > 200 && center.g > 200)
    };
    assert_or_reread(
        &h,
        "the far blue draw fails the depth test the near white one wrote",
        "white at the centre",
        &reading(resolve_and_read()),
        || reading(read()),
        || {
            let center = Rgba8::from_pixel(resolve_and_read());
            Reading::described(
                format!("{center:?}\n  depth probe: {}", depth_probe(&h)),
                center.r > 200 && center.g > 200,
            )
        },
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK);
}

#[test]
fn color_fill_of_a_multisampled_target_survives_a_later_draw() {
    // The fill has to reach the multisampled samples, not only the
    // single-sample texture a resolve writes: the draw after it loads the
    // samples and resolves them over the whole target.
    let h = Harness::new();
    let rt = h.create_render_target_ms(
        (MS_SIZE, MS_SIZE),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    let backbuffer = h.back_buffer(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );

    assert_eq!(h.set_render_target(0, &rt), D3D_OK, "bind the 4x target");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.clear_target(BLACK), D3D_OK, "Clear");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &corner(128.0, WHITE)),
        D3D_OK,
        "cover the target in white"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.set_render_target(0, &backbuffer), D3D_OK, "unbind it");

    assert_eq!(
        h.color_fill_hr(&rt, BLUE),
        D3D_OK,
        "ColorFill the 4x target"
    );

    assert_eq!(h.set_render_target(0, &rt), D3D_OK, "bind it again");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &corner(8.0, RED)),
        D3D_OK,
        "a small draw in one corner"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");

    let center = Rgba8::from_pixel(resolved_pixel(&h, &rt, MS_SIZE / 2, MS_SIZE / 2));
    assert!(
        center.b > 200 && center.r < 40 && center.g < 40,
        "the fill survives the corner draw's resolve, got {center:?}"
    );
    let drawn = Rgba8::from_pixel(resolved_pixel(&h, &rt, 1, 1));
    assert!(
        drawn.r > 200 && drawn.b < 40,
        "the corner draw lands over the fill, got {drawn:?}"
    );
}

#[test]
fn color_fill_of_the_bound_multisampled_target_paints_its_rect() {
    // A fill of the target that is already bound reuses the open 4x pass, so
    // the clear quad it draws there has to be built for four samples.
    let h = Harness::new();
    let rt = h.create_render_target_ms(
        (MS_SIZE, MS_SIZE),
        D3DFMT_X8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );
    assert_eq!(h.set_render_target(0, &rt), D3D_OK, "bind the 4x target");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.clear_target(BLACK), D3D_OK, "Clear");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &corner(128.0, WHITE)),
        D3D_OK,
        "cover the target in white"
    );
    assert_eq!(
        h.color_fill_rect_hr(&rt, (16, 16, 48, 48), BLUE),
        D3D_OK,
        "ColorFill a rect of the bound 4x target"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");

    let inside = Rgba8::from_pixel(resolved_pixel(&h, &rt, MS_SIZE / 2, MS_SIZE / 2));
    assert!(
        inside.b > 200 && inside.r < 40 && inside.g < 40,
        "the rect carries the fill on every sample, got {inside:?}"
    );
    let outside = Rgba8::from_pixel(resolved_pixel(&h, &rt, 4, 4));
    assert!(
        outside.r > 200 && outside.g > 200 && outside.b > 200,
        "outside the rect the draw stays, got {outside:?}"
    );
}
