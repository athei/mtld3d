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

use mtld3d_tests::{Factory, Harness, HarnessConfig, Texture, TexturedVertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DDISPLAYMODEEX, D3DERR_INVALIDCALL, D3DERR_NOTFOUND, D3DFMT_A8R8G8B8, D3DFMT_D24S8,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_DEFAULT,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_LIGHTING, D3DSAMP_MAGFILTER,
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
    assert_eq!(h.test_cooperative_level(), D3D_OK);

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
    let first = lock.read_bytes(4);
    drop(lock);
    let pixel = u32::from_le_bytes(first.try_into().expect("four bytes"));
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
    assert_eq!(h.test_cooperative_level(), D3D_OK);
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
    let first = lock.read_bytes(4);
    drop(lock);
    let pixel = u32::from_le_bytes(first.try_into().expect("four bytes"));
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
    let extra = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(1, &extra), D3D_OK);
    let own = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &own), D3D_OK);
    assert_eq!(h.clear_depth_stencil_surface(), D3D_OK);

    assert_eq!(h.reset(320, 240), D3D_OK);
    let (hr, slot1) = h.render_target_hr(1);
    assert_eq!(
        (hr, slot1.is_none()),
        (D3DERR_NOTFOUND, true),
        "render target 1 is unbound"
    );
    assert_eq!(
        h.render_target(0).as_ptr(),
        h.back_buffer(0).as_ptr(),
        "render target 0 is the back buffer"
    );
    let (hr, depth) = h.depth_stencil_surface_hr();
    assert_eq!(
        (hr, depth.is_some()),
        (D3D_OK, true),
        "the implicit depth surface is bound"
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

#[test]
fn reset_ex_names_a_mode_exactly_when_fullscreen_and_a_refusal_changes_nothing() {
    let h = extended(false);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    let (width, height) = h.dims();
    let mode = D3DDISPLAYMODEEX {
        size: 24,
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

    assert_eq!(h.test_cooperative_level(), D3D_OK, "no Reset is owed");
    assert_eq!(h.render_state(D3DRS_LIGHTING), 0, "the state is untouched");
    assert_eq!(h.end_scene(), D3D_OK, "the scene is still open");
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

    assert_eq!(h.test_cooperative_level(), D3D_OK, "no Reset is owed");
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
    assert_pixel_eq(sample_center(&h, &texture), RED, "the device draws on");
}
