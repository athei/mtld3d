//! `Reset` and `ResetEx` on an extended device.
//!
//! An extended device keeps everything across `Reset` but its targets and the
//! viewport extent: render states, bindings, the viewport's depth range, an
//! open scene and the contents of its default-pool resources all survive,
//! and references to default-pool resources, implicit surfaces and state
//! blocks do not block it. Render target 0 returns to the back buffer, 1 to 3
//! are unbound, the depth stencil returns to the implicit surface or to none,
//! and the viewport and scissor cover the new back buffer. A back buffer or
//! depth surface the application holds across it keeps the old surface, its
//! size and contents, with the device as its container. A rejected `Reset`
//! or `ResetEx` changes nothing and leaves no `Reset` owed, and `ResetEx`
//! names a display mode of the back buffer's size exactly when fullscreen.
//! Leaving fullscreen leaves the window where fullscreen put it and gives it
//! back the style, visibility included, it had before, also when the driver
//! or another thread moved the window while the display mode was restored,
//! shown or hidden; a move the window's own procedure made then stays.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use mtld3d_tests::{
    Factory, Harness, HarnessConfig, Rect, Texture, TexturedVertex, WS_VISIBLE, WindowStyle,
    assert_pixel_eq, create_window, destroy_window, enumerate_display_sizes,
    move_window_on_next_display_change, set_window_pos, spawn_scoped,
};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODEEX, D3DDISPLAYMODEEX_SIZE, D3DERR_INVALIDCALL, D3DERR_NOTFOUND,
    D3DFMT_A8R8G8B8, D3DFMT_D16, D3DFMT_D24S8, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_XYZ, D3DLOCK_READONLY, D3DMULTISAMPLE_4_SAMPLES, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM,
    D3DPRESENT_PARAMETERS, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_LIGHTING, D3DSAMP_MAGFILTER,
    D3DSAMP_MINFILTER, D3DSBT_ALL, D3DTEXF_POINT, D3DUSAGE_RENDERTARGET, D3DVIEWPORT9,
    E_NOINTERFACE, IID_IDIRECT3DDEVICE9, IID_IDIRECT3DSWAPCHAIN9,
};

const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;

fn extended(depth: bool) -> Harness {
    Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        depth_format: depth.then_some(D3DFMT_D24S8),
        ..HarnessConfig::default()
    })
}

/// A default-pool A8R8G8B8 texture of one level filled with `color`, through `UpdateTexture`.
fn filled_default_texture(h: &Harness, color: u32) -> Texture<'_> {
    let staging = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    staging.lock_rect(0, 0).write_u32_rect(8, 8, &[color; 64]);
    let texture = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&staging, &texture),
        D3D_OK,
        "UpdateTexture"
    );
    texture
}

/// Draw `texture` over the whole back buffer and read its centre.
fn sample_center(h: &Harness, texture: &Texture<'_>) -> u32 {
    const W: u32 = 0xFFFF_FFFF;
    let quad = [
        (-1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 0.0),
        (-1.0, -1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
        (1.0, -1.0, 1.0, 1.0),
        (-1.0, -1.0, 0.0, 1.0),
    ]
    .map(|(x, y, u, v)| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: W,
        u,
        v,
    });
    assert_eq!(h.set_texture(0, texture), D3D_OK, "SetTexture");
    h.select_texture_stage(0);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    h.render_once(0xFF00_0000, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            D3D_OK,
            "draw"
        );
    });
    let (width, height) = h.dims();
    h.read_pixel(width / 2, height / 2)
}

#[test]
fn an_extended_reset_keeps_state_and_rebinds_only_the_targets_and_the_viewport_extent() {
    let h = extended(true);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    let viewport = D3DVIEWPORT9 {
        x: 10,
        y: 20,
        width: 100,
        height: 100,
        min_z: 0.25,
        max_z: 0.75,
    };
    assert_eq!(h.set_viewport(&viewport), D3D_OK);
    let scissor = D3DRECT {
        x1: 10,
        y1: 20,
        x2: 30,
        y2: 40,
    };
    assert_eq!(h.set_scissor_rect(&scissor), D3D_OK);
    let texture = h.create_texture(16, 16, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(h.set_texture(3, &texture), D3D_OK);
    let vb = h.create_vertex_buffer(64, 0, 0, D3DPOOL_DEFAULT);
    assert_eq!(h.set_stream_source(0, &vb, 0, 16), D3D_OK);

    assert_eq!(h.reset(400, 300), D3D_OK, "Reset with live bindings");
    assert_eq!(h.present_ex(0), D3D_OK, "no Reset is owed");

    assert_eq!(h.render_state(D3DRS_LIGHTING), 0, "render states survive");
    let after = h.viewport();
    assert_eq!(
        (after.x, after.y, after.width, after.height),
        (0, 0, 400, 300),
        "the viewport covers the new back buffer"
    );
    assert_eq!(
        (after.min_z.to_bits(), after.max_z.to_bits()),
        (0.25f32.to_bits(), 0.75f32.to_bits()),
        "and keeps its depth range"
    );
    let rect = h.scissor_rect();
    assert_eq!(
        (rect.x1, rect.y1, rect.x2, rect.y2),
        (0, 0, 400, 300),
        "the scissor too"
    );
    assert!(
        h.texture_matches_raw(3, texture.as_ptr()),
        "the texture stays bound"
    );
    let (hr, bound, offset, stride) = h.get_stream_source(0);
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (bound.map(|vb| vb.as_ptr()), offset, stride),
        (Some(vb.as_ptr()), 0, 16),
        "the stream source stays bound"
    );
}

#[test]
fn a_plain_reset_still_returns_the_state_an_extended_one_keeps_to_its_defaults() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.reset(400, 300), D3D_OK);
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        1,
        "D3DRS_LIGHTING back to TRUE"
    );
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.reset(320, 240), D3D_OK);
    assert_eq!(
        h.end_scene(),
        D3DERR_INVALIDCALL,
        "a plain Reset ends the scene"
    );
}

#[test]
fn default_pool_resources_keep_their_contents_across_an_extended_reset() {
    let h = extended(false);
    let target = h.create_texture(
        32,
        32,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let target_surface = target.surface_level(0);
    assert_eq!(h.color_fill_hr(&target_surface, RED), D3D_OK, "ColorFill");
    let sampled = filled_default_texture(&h, BLUE);

    assert_eq!(
        h.reset(320, 240),
        D3D_OK,
        "Reset with live default-pool resources"
    );

    let readback = h.create_offscreen_plain_surface(32, 32, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&target_surface, &readback),
        D3D_OK
    );
    let lock = readback.lock_rect(D3DLOCK_READONLY);
    let pixel = lock.as_u32(1)[0];
    drop(lock);
    assert_pixel_eq(pixel, RED, "the render-target texture keeps its fill");
    assert_pixel_eq(
        sample_center(&h, &sampled),
        BLUE,
        "the sampled texture keeps its texels",
    );
}

#[test]
fn references_to_default_resources_implicit_surfaces_and_state_blocks_do_not_block_it() {
    let h = extended(true);
    let back_buffer = h.back_buffer(0);
    let (hr, depth) = h.depth_stencil_surface_hr();
    assert_eq!(hr, D3D_OK);
    let plain = h.create_offscreen_plain_surface(16, 16, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let texture = filled_default_texture(&h, RED);
    assert_eq!(h.set_texture(0, &texture), D3D_OK);
    let block = h.create_state_block(D3DSBT_ALL);

    assert_eq!(
        h.reset(400, 300),
        D3D_OK,
        "nothing the application holds blocks it"
    );
    assert_eq!(h.present_ex(0), D3D_OK, "no Reset is owed");
    drop((block, plain, depth, back_buffer));
    h.render_once(BLUE, |_| {});
    assert_pixel_eq(h.read_pixel(200, 150), BLUE, "the device draws after it");
}

#[test]
fn a_held_back_buffer_and_depth_surface_keep_the_old_surfaces_across_an_extended_reset() {
    let h = extended(true);
    assert_eq!(h.clear_target(RED), D3D_OK);
    let old_back_buffer = h.back_buffer(0);
    let (hr, old_depth) = h.depth_stencil_surface_hr();
    assert_eq!(hr, D3D_OK);
    let old_depth = old_depth.expect("the implicit depth surface");
    let (hr, chain, _) = old_back_buffer.get_container(&IID_IDIRECT3DSWAPCHAIN9);
    assert_eq!(
        (hr, chain.is_null()),
        (D3D_OK, false),
        "the swap chain's back buffer"
    );

    assert_eq!(h.reset(400, 300), D3D_OK);

    let (hr, desc) = old_back_buffer.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (desc.width, desc.height),
        (640, 480),
        "the held back buffer keeps its size"
    );
    let (hr, desc) = old_depth.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (desc.width, desc.height),
        (640, 480),
        "so does the held depth surface"
    );
    assert_eq!(
        old_back_buffer.get_container(&IID_IDIRECT3DSWAPCHAIN9).0,
        E_NOINTERFACE,
        "it is no longer the swap chain's"
    );
    let (hr, container, _) = old_back_buffer.get_container(&IID_IDIRECT3DDEVICE9);
    assert_eq!(
        (hr, container),
        (D3D_OK, h.device()),
        "its container is the device"
    );

    let readback = h.create_offscreen_plain_surface(640, 480, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&old_back_buffer, &readback),
        D3D_OK
    );
    let lock = readback.lock_rect(D3DLOCK_READONLY);
    let pixel = lock.as_u32(1)[0];
    drop(lock);
    assert_pixel_eq(
        pixel,
        RED,
        "the held back buffer keeps what was drawn into it",
    );

    let new_back_buffer = h.back_buffer(0);
    assert_ne!(
        new_back_buffer.as_ptr(),
        old_back_buffer.as_ptr(),
        "a new back buffer object"
    );
    let (hr, desc) = new_back_buffer.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!((desc.width, desc.height), (400, 300), "at the new size");
    drop((new_back_buffer, old_back_buffer, old_depth));
    h.render_once(BLUE, |_| {});
    assert_pixel_eq(
        h.read_pixel(200, 150),
        BLUE,
        "the device draws into the new one",
    );
}

#[test]
fn a_window_resize_keeps_the_viewport_depth_range_on_an_extended_device_alone() {
    for (factory, depth_range) in [
        (Factory::Extended, (0.25f32, 0.75f32)),
        (Factory::Plain, (0.0, 1.0)),
    ] {
        let h = Harness::create(&HarnessConfig {
            factory,
            ..HarnessConfig::default()
        });
        let viewport = D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width: 640,
            height: 480,
            min_z: 0.25,
            max_z: 0.75,
        };
        assert_eq!(h.set_viewport(&viewport), D3D_OK);
        mtld3d_tests::set_window_pos(h.hwnd(), 0, 0, 400, 300);
        assert!(h.pump(), "WM_QUIT after the resize");
        let after = h.viewport();
        assert_eq!(
            (after.width, after.height),
            (400, 300),
            "the viewport follows the window"
        );
        assert_eq!(
            (after.min_z.to_bits(), after.max_z.to_bits()),
            (depth_range.0.to_bits(), depth_range.1.to_bits()),
            "the depth range after the resize"
        );
    }
}

#[test]
fn an_open_scene_survives_an_extended_reset() {
    let h = extended(false);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.reset(320, 240), D3D_OK);
    assert_eq!(h.end_scene(), D3D_OK, "the scene is still open");
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.end_scene(), D3D_OK);
}

#[test]
fn an_extended_reset_unbinds_targets_one_to_three_and_restores_the_depth_surface() {
    let h = extended(true);
    let extras = [1, 2, 3].map(|_| h.create_render_target(64, 64, D3DFMT_A8R8G8B8));
    for (slot, extra) in (1..).zip(&extras) {
        assert_eq!(
            h.set_render_target(slot, extra),
            D3D_OK,
            "bind render target {slot}"
        );
    }
    let own = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &own), D3D_OK);
    let own_depth = h.create_depth_stencil_surface(64, 64, D3DFMT_D24S8);
    assert_eq!(h.set_depth_stencil_surface(&own_depth), D3D_OK);

    assert_eq!(h.reset(320, 240), D3D_OK);
    for slot in 1..4 {
        let (hr, bound) = h.render_target_hr(slot);
        assert_eq!(
            (hr, bound.is_none()),
            (D3DERR_NOTFOUND, true),
            "render target {slot} is unbound"
        );
    }
    assert_eq!(
        h.render_target(0).as_ptr(),
        h.back_buffer(0).as_ptr(),
        "render target 0 is the back buffer"
    );
    let (hr, depth) = h.depth_stencil_surface_hr();
    assert_eq!(hr, D3D_OK);
    let depth = depth.expect("the implicit depth surface is bound");
    assert_ne!(
        depth.as_ptr(),
        own_depth.as_ptr(),
        "not the application's own"
    );
    let (hr, desc) = depth.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (desc.format, desc.width, desc.height),
        (D3DFMT_D24S8, 320, 240),
        "the auto depth-stencil at the new size"
    );
    drop(depth);

    let mut pp = h.windowed_present_params(320, 240);
    pp.enable_auto_depth_stencil = 0;
    pp.auto_depth_stencil_format = 0;
    assert_eq!(h.reset_ex(&mut pp, None), D3D_OK);
    let (hr, depth) = h.depth_stencil_surface_hr();
    assert_eq!(
        (hr, depth.is_none()),
        (D3DERR_NOTFOUND, true),
        "without one, none"
    );
}

/// Hold the implicit depth surface across a `ResetEx` that `change` makes; check what it reports.
fn assert_held_depth_keeps_its_own_desc(
    what: &str,
    change: impl FnOnce(&mut D3DPRESENT_PARAMETERS),
) {
    let h = extended(true);
    let (hr, held) = h.depth_stencil_surface_hr();
    assert_eq!(hr, D3D_OK);
    let held = held.expect("the implicit depth surface");
    let mut pp = h.windowed_present_params(320, 240);
    change(&mut pp);
    assert_eq!(h.reset_ex(&mut pp, None), D3D_OK, "ResetEx to {what}");
    let (hr, desc) = held.desc();
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        (desc.format, desc.multi_sample_type, desc.width, desc.height),
        (D3DFMT_D24S8, 0, 640, 480),
        "the held depth surface after a Reset to {what}"
    );
    let (hr, container, _) = held.get_container(&IID_IDIRECT3DDEVICE9);
    assert_eq!(
        (hr, container),
        (D3D_OK, h.device()),
        "its container is the device"
    );
}

#[test]
fn a_held_depth_surface_keeps_its_own_format_across_an_extended_reset_to_another() {
    assert_held_depth_keeps_its_own_desc("another auto depth-stencil format", |pp| {
        pp.auto_depth_stencil_format = D3DFMT_D16;
    });
}

#[test]
fn a_held_depth_surface_keeps_its_own_samples_across_an_extended_reset_to_multisampling() {
    assert_held_depth_keeps_its_own_desc("multisampling", |pp| {
        pp.multi_sample_type = D3DMULTISAMPLE_4_SAMPLES;
    });
}

#[test]
fn a_held_depth_surface_keeps_its_own_desc_across_an_extended_reset_without_auto_depth() {
    assert_held_depth_keeps_its_own_desc("no auto depth-stencil", |pp| {
        pp.enable_auto_depth_stencil = 0;
        pp.auto_depth_stencil_format = 0;
    });
}

#[test]
fn a_recording_state_block_survives_an_extended_reset() {
    let h = extended(false);
    assert_eq!(h.begin_state_block(), D3D_OK);
    assert_eq!(h.reset(320, 240), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    let block = h.end_state_block();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), D3D_OK);
    assert_eq!(block.apply(), D3D_OK);
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "the block recorded across the Reset"
    );
}

#[test]
fn reset_ex_names_a_mode_exactly_when_fullscreen_and_a_refusal_changes_nothing() {
    let h = extended(false);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    let (width, height) = h.dims();
    let mode = D3DDISPLAYMODEEX {
        size: D3DDISPLAYMODEEX_SIZE,
        width,
        height,
        refresh_rate: 0,
        format: D3DFMT_X8R8G8B8,
        scan_line_ordering: 0,
    };

    let mut windowed = h.windowed_present_params(width, height);
    assert_eq!(
        h.reset_ex(&mut windowed, Some(&mode)),
        D3DERR_INVALIDCALL,
        "windowed with a mode"
    );
    let mut fullscreen = h.windowed_present_params(width, height);
    fullscreen.windowed = 0;
    assert_eq!(
        h.reset_ex(&mut fullscreen, None),
        D3DERR_INVALIDCALL,
        "fullscreen without one"
    );
    let mismatched = D3DDISPLAYMODEEX {
        width: width - 1,
        ..mode
    };
    let mut fullscreen = h.windowed_present_params(width, height);
    fullscreen.windowed = 0;
    assert_eq!(
        h.reset_ex(&mut fullscreen, Some(&mismatched)),
        D3DERR_INVALIDCALL,
        "a mode of another size"
    );

    assert_eq!(h.render_state(D3DRS_LIGHTING), 0, "the state is untouched");
    assert_eq!(h.end_scene(), D3D_OK, "the scene is still open");
    assert_eq!(h.present_ex(0), D3D_OK, "no Reset is owed");
    assert_eq!(
        h.reset_ex(&mut windowed, None),
        D3D_OK,
        "a windowed ResetEx without one"
    );
}

#[test]
fn a_rejected_extended_reset_leaves_the_device_working_and_its_state_untouched() {
    let h = extended(false);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    let texture = filled_default_texture(&h, RED);
    assert_eq!(h.set_texture(0, &texture), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);

    let (width, height) = h.dims();
    let mut no_swap_effect = h.windowed_present_params(width, height);
    no_swap_effect.swap_effect = 0;
    assert_eq!(
        h.reset_params(&mut no_swap_effect),
        D3DERR_INVALIDCALL,
        "swap effect 0"
    );
    let mut too_many = h.windowed_present_params(width, height);
    too_many.back_buffer_count = 31;
    assert_eq!(
        h.reset_params(&mut too_many),
        D3DERR_INVALIDCALL,
        "31 back buffers"
    );

    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "render states are untouched"
    );
    assert!(
        h.texture_matches_raw(0, texture.as_ptr()),
        "so are the bindings"
    );
    assert_eq!(h.end_scene(), D3D_OK, "and the open scene");
    assert_eq!(h.present_ex(0), D3D_OK, "no Reset is owed");
    assert_pixel_eq(sample_center(&h, &texture), RED, "the device draws on");
}

/// The window's rect and style around a fullscreen `Reset` and back on a hidden framed window.
struct FullscreenRoundTrip {
    windowed_rect: Rect,
    windowed_style: u32,
    fullscreen_rect: Rect,
    left_rect: Rect,
    left_style: u32,
}

/// Take a hidden framed window fullscreen through `Reset` and back on a device from `factory`.
///
/// `None` when the display lists no 640x480 mode to take.
fn fullscreen_round_trip(factory: Factory) -> Option<FullscreenRoundTrip> {
    if !enumerate_display_sizes().contains(&(640, 480)) {
        return None;
    }
    let h = Harness::create(&HarnessConfig {
        factory,
        window_style: WindowStyle::Framed,
        ..HarnessConfig::default()
    });
    // Held before the first read of the window's geometry, so no other test's
    // mode-set falls between the reads this test compares.
    h.hold_display_mode();
    let windowed_rect = h.window_rect();
    let windowed_style = h.window_style();
    assert_eq!(windowed_style & WS_VISIBLE, 0, "starts hidden");

    let mut pp = h.windowed_present_params(640, 480);
    pp.windowed = 0;
    pp.device_window = h.hwnd();
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");
    let fullscreen_rect = h.window_rect();
    assert_ne!(
        h.window_style() & WS_VISIBLE,
        0,
        "fullscreen shows the window"
    );

    assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
    Some(FullscreenRoundTrip {
        windowed_rect,
        windowed_style,
        fullscreen_rect,
        left_rect: h.window_rect(),
        left_style: h.window_style(),
    })
}

#[test]
fn leaving_fullscreen_keeps_the_window_where_it_is_and_gives_back_its_visibility() {
    let Some(trip) = fullscreen_round_trip(Factory::Extended) else {
        return;
    };
    assert_eq!(
        (trip.left_rect, trip.left_style),
        (trip.fullscreen_rect, trip.windowed_style),
        "the window keeps the fullscreen rect and gets its own style back, hidden as it was"
    );
}

#[test]
fn leaving_fullscreen_on_a_plain_device_restores_the_rect_and_keeps_the_window_shown() {
    let Some(trip) = fullscreen_round_trip(Factory::Plain) else {
        return;
    };
    assert_eq!(
        (trip.left_rect, trip.left_style),
        (trip.windowed_rect, trip.windowed_style | WS_VISIBLE),
        "the window gets its old rect and style back, still shown"
    );
}

/// How far another thread moves the window while a leave restores the display mode.
///
/// Small enough that the window still overlaps the menu bar, where `AppKit`
/// moves it further, to the top of the work area, as it does in the field.
const MOVED_DURING_THE_RESTORE: i32 = 40;

/// Where the window's own procedure puts the window's top edge while a leave restores the mode.
///
/// Well inside the work area, below the menu bar, so `AppKit` leaves a shown
/// window there; and a place rather than an offset, so a driver move that
/// lands before the procedure runs does not change where the window ends.
const APP_TOP_DURING_THE_RESTORE: i32 = 200;

/// The longest a helper thread waits for the display mode or for the test thread.
const HELPER_WAIT: Duration = Duration::from_secs(10);

/// An extended device on a framed 640x480 window, taken fullscreen at 640x480 through `Reset`.
///
/// Returns the harness, holding the display mode, and the fullscreen rect;
/// `None` when the display lists no 640x480 mode to take.
fn extended_fullscreen_at_640x480(shown: bool) -> Option<(Harness, Rect)> {
    if !enumerate_display_sizes().contains(&(640, 480)) {
        return None;
    }
    let h = Harness::create(&HarnessConfig {
        factory: Factory::Extended,
        window_style: WindowStyle::Framed,
        visible: shown,
        ..HarnessConfig::default()
    });
    h.hold_display_mode();
    let mut pp = h.windowed_present_params(640, 480);
    pp.windowed = 0;
    pp.device_window = h.hwnd();
    assert_eq!(h.reset_params(&mut pp), D3D_OK, "fullscreen Reset");
    assert_eq!(
        Harness::current_display_mode(),
        (640, 480),
        "the fullscreen Reset set the 640x480 mode"
    );
    let fullscreen_rect = h.window_rect();
    Some((h, fullscreen_rect))
}

/// `rect` moved `down` pixels.
const fn moved_down(rect: Rect, down: i32) -> Rect {
    Rect {
        left: rect.left,
        top: rect.top + down,
        right: rect.right,
        bottom: rect.bottom + down,
    }
}

/// Leave fullscreen while another thread moves the window during the mode restore.
///
/// One thread owns a window it never pumps, created after the fullscreen
/// mode-set, so the leave's `WM_DISPLAYCHANGE` broadcast waits on it for the
/// broadcast's timeout. Another moves the device window as soon as the mode
/// is back, and the leaving thread applies that move inside the wait, as it
/// applies the move `AppKit` makes when it keeps a window it still shows
/// below the menu bar. Returns the rect after the windowed `Reset`.
fn leave_while_another_thread_moves_the_window(h: &Harness, fullscreen_rect: Rect) -> Rect {
    let device_window = h.hwnd();
    let done = AtomicBool::new(false);
    let moved = AtomicBool::new(false);
    let (ready, ready_receiver) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        let (done, moved) = (&done, &moved);
        spawn_scoped(scope, move || {
            let silent = create_window(160, 120, false);
            ready
                .send(())
                .expect("the test waits for the silent window");
            let deadline = Instant::now() + HELPER_WAIT;
            while !done.load(Ordering::Acquire) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            destroy_window(silent);
        });
        spawn_scoped(scope, move || {
            let deadline = Instant::now() + HELPER_WAIT;
            while Harness::current_display_mode() == (640, 480) {
                if Instant::now() >= deadline {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let target = moved_down(fullscreen_rect, MOVED_DURING_THE_RESTORE);
            set_window_pos(
                device_window,
                target.left,
                target.top,
                target.right - target.left,
                target.bottom - target.top,
            );
            moved.store(true, Ordering::Release);
        });
        ready_receiver
            .recv()
            .expect("the silent window's thread made its window");
        assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
        // The move is a message to this thread, which it answers only while
        // it waits on another thread: before `Reset` returned, or in the
        // pump below.
        let landed_in_the_leave = moved.load(Ordering::Acquire);
        let left_rect = h.window_rect();
        done.store(true, Ordering::Release);
        let deadline = Instant::now() + HELPER_WAIT;
        while !moved.load(Ordering::Acquire) && Instant::now() < deadline {
            let _ = h.pump();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            landed_in_the_leave,
            "the other thread's move landed inside the leave"
        );
        left_rect
    })
}

/// A window another thread moves during the mode restore still ends at the fullscreen rect.
///
/// The move stands in for the one `AppKit` makes under Wine on macOS, which
/// the leaving thread applies while it waits on the mode change's broadcast.
/// On Windows a move another thread of the application made in that span
/// would stay, where the layer puts back the rect the window had before the
/// restore: the two cannot be told apart there, and `AppKit`'s move is the
/// one that happens.
#[test]
fn leaving_fullscreen_puts_back_the_fullscreen_rect_after_a_move_during_the_mode_restore() {
    let Some((h, fullscreen_rect)) = extended_fullscreen_at_640x480(false) else {
        return;
    };
    let left_rect = leave_while_another_thread_moves_the_window(&h, fullscreen_rect);
    assert_eq!(
        left_rect, fullscreen_rect,
        "the window ends at the fullscreen rect it had before the mode restore"
    );
}

/// A shown window another thread moves during the mode restore ends there too, still shown.
#[test]
fn leaving_fullscreen_puts_back_a_shown_window_moved_during_the_mode_restore() {
    let Some((h, fullscreen_rect)) = extended_fullscreen_at_640x480(true) else {
        return;
    };
    let left_rect = leave_while_another_thread_moves_the_window(&h, fullscreen_rect);
    assert_eq!(
        (left_rect, h.window_style() & WS_VISIBLE),
        (fullscreen_rect, WS_VISIBLE),
        "the shown window ends at the fullscreen rect it had before the mode restore, still shown"
    );
}

/// A move the window's own procedure makes answering the mode restore stays.
///
/// The procedure puts the window at a place of its own, as a game does: the
/// driver may already have moved the still-shown window below the menu bar
/// while the broadcast waited on another thread's window, before the message
/// reached this one, and the window ends where the procedure put it either
/// way. That place is inside the work area, where `AppKit` leaves it.
#[test]
fn leaving_fullscreen_keeps_a_move_the_window_procedure_makes_during_the_mode_restore() {
    let Some((h, fullscreen_rect)) = extended_fullscreen_at_640x480(false) else {
        return;
    };
    move_window_on_next_display_change(h.hwnd(), APP_TOP_DURING_THE_RESTORE);
    assert_eq!(h.reset(640, 480), D3D_OK, "windowed Reset");
    assert_eq!(
        h.window_rect(),
        moved_down(
            fullscreen_rect,
            APP_TOP_DURING_THE_RESTORE - fullscreen_rect.top
        ),
        "the window stays where its procedure moved it"
    );
}
