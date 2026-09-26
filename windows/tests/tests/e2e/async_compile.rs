//! First-use shader and pipeline builds on the encoder's worker threads.
//!
//! The tests run with the shader cache off, so each draw's libraries and
//! pipeline are new to the device and build on a worker. Under
//! `shader.asyncCompile = true` a draw whose build is still in flight is
//! left out of the frame only when its target is rebuilt every frame (the
//! back buffer under the discard swap effect, or a target cleared in this
//! frame and the one before, and read by nothing kept) and no occlusion
//! query is counting. Any other such draw is kept: it is encoded with a
//! placeholder pipeline, and the frame's submission waits for its builds
//! and binds the real one, so it is built before its frame is submitted.
//! With the option off, as in the rest of the suite, every such draw is
//! kept that way.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, Surface, TexturedVertex, Vertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DFMT_A8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN, D3DISSUE_END,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_OCCLUSION, D3DRS_ALPHABLENDENABLE, D3DRS_COLORWRITEENABLE, D3DRS_DESTBLEND,
    D3DRS_LIGHTING, D3DRS_SRCBLEND, D3DTEXF_NONE, D3DUSAGE_RENDERTARGET,
};

const ASYNC: &str = "shader.asyncCompile=true;shaderCache.enable=false";
const SYNC: &str = "shader.asyncCompile=false;shaderCache.enable=false";

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;

/// How long a build may take to land before the test gives up on it.
const BUILD_DEADLINE: Duration = Duration::from_secs(20);

/// One triangle covering the whole viewport, in the vertex colour `color`.
const fn covering_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color,
        },
        Vertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color,
        },
        Vertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color,
        },
    ]
}

/// A device under `shader.asyncCompile` set up for unlit vertex-colour draws.
fn async_device() -> Harness {
    device_with(ASYNC)
}

/// A device under `entries` set up for unlit vertex-colour draws.
fn device_with(entries: &'static str) -> Harness {
    let h = Harness::with_config(entries);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );
    h
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

/// Draw `tri` (when given) into `rt`, cleared to `clear` first when it is given, in one frame.
fn frame_into_target(h: &Harness, rt: &Surface<'_>, clear: Option<u32>, tri: Option<&[Vertex; 3]>) {
    let backbuffer = h.render_target(0);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the back buffer");
    assert_eq!(
        h.set_render_target(0, rt),
        D3D_OK,
        "bind the offscreen target"
    );
    if let Some(color) = clear {
        assert_eq!(h.clear_target(color), D3D_OK, "clear the offscreen target");
    }
    if let Some(tri) = tri {
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, tri),
            D3D_OK,
            "DrawPrimitiveUP"
        );
    }
    assert_eq!(
        h.set_render_target(0, &backbuffer),
        D3D_OK,
        "restore the back buffer"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");
}

/// A first-seen shader drawn to the back buffer is left out until its build lands, then drawn.
#[test]
fn a_back_buffer_draw_is_left_out_until_its_build_lands() {
    let h = async_device();
    let tri = covering_triangle(RED);
    let draw = |dev: &Harness| {
        assert_eq!(
            dev.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
            D3D_OK,
            "DrawPrimitiveUP"
        );
    };
    h.render_once(BLUE, draw);
    assert_pixel_eq(
        h.read_pixel(320, 240),
        BLUE,
        "the first frame leaves the draw out: its libraries were only just queued",
    );
    let deadline = Instant::now() + BUILD_DEADLINE;
    let mut frames = 1u32;
    loop {
        h.render_once(BLUE, draw);
        frames += 1;
        let pixel = h.read_pixel(320, 240);
        if pixel == RED {
            break;
        }
        assert_pixel_eq(
            pixel,
            BLUE,
            "a frame before the build lands shows the clear alone",
        );
        assert!(
            Instant::now() < deadline,
            "the draw was still left out after {frames} frames"
        );
    }
}

/// A draw into an offscreen target no clear reached this frame is built before its submission.
#[test]
fn a_draw_into_an_uncleared_target_is_built_before_its_frame_is_submitted() {
    let h = async_device();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    frame_into_target(&h, &rt, None, Some(&covering_triangle(RED)));
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        RED,
        "a target the frame did not clear may be read for the rest of its life, \
         so its first draw is never left out",
    );
}

/// A target cleared only in this frame may be a one-off render, so its draw is kept.
#[test]
fn a_draw_into_a_target_cleared_only_this_frame_is_built_before_its_frame_is_submitted() {
    let h = async_device();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    frame_into_target(&h, &rt, Some(GREEN), Some(&covering_triangle(RED)));
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        RED,
        "a clear and a draw once, as a baked texture is made, never loses the draw",
    );
}

/// A target cleared every frame is rebuilt every frame, so its draw is left out until it builds.
#[test]
fn a_draw_into_a_target_cleared_every_frame_is_left_out_until_its_build_lands() {
    let h = async_device();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let tri = covering_triangle(RED);
    frame_into_target(&h, &rt, Some(GREEN), None);
    frame_into_target(&h, &rt, Some(GREEN), Some(&tri));
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        GREEN,
        "the second cleared frame leaves the draw out and keeps the clear",
    );
    let deadline = Instant::now() + BUILD_DEADLINE;
    let mut frames = 2u32;
    loop {
        frame_into_target(&h, &rt, Some(GREEN), Some(&tri));
        frames += 1;
        let pixel = read_rt_pixel(&h, &rt, 32, 32);
        if pixel == RED {
            break;
        }
        assert_pixel_eq(
            pixel,
            GREEN,
            "a frame before the build lands shows the clear alone",
        );
        assert!(
            Instant::now() < deadline,
            "the draw was still left out after {frames} frames"
        );
    }
}

/// A first-seen draw inside a counting occlusion query is kept, so the query counts its samples.
#[test]
fn a_draw_counted_by_an_occlusion_query_is_built_before_its_frame_is_submitted() {
    let h =
        device_with("shader.asyncCompile=true;shaderCache.enable=false;query.flushImmediate=false");
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    let tri = covering_triangle(RED);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.clear_target(BLUE), D3D_OK, "Clear");
    assert_eq!(q.issue(D3DISSUE_BEGIN), D3D_OK, "Issue(BEGIN)");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
        D3D_OK,
        "DrawPrimitiveUP"
    );
    assert_eq!(q.issue(D3DISSUE_END), D3D_OK, "Issue(END)");
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");
    let (hr, samples) = q.data_u32(D3DGETDATA_FLUSH);
    assert_eq!(hr, D3D_OK, "GetData(FLUSH)");
    assert_ne!(
        samples, 0,
        "a draw the application counts is never left out of the count"
    );
}

/// A scratch target copied into part of a kept one every frame is kept too, and so is its draw.
///
/// The copy covers a corner of the kept target only: a copy over a whole
/// target rebuilds it as a clear does, and the kept target would then be
/// rebuilt every frame itself.
#[test]
fn a_draw_into_a_scratch_target_copied_into_a_kept_one_is_built_before_its_frame_is_submitted() {
    let h = async_device();
    let scratch = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let kept = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    let tri = covering_triangle(RED);
    let frame = |draw: bool| {
        let backbuffer = h.render_target(0);
        assert!(h.pump(), "WM_QUIT before render");
        assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
        assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the back buffer");
        assert_eq!(h.set_render_target(0, &scratch), D3D_OK, "bind scratch");
        assert_eq!(h.clear_target(GREEN), D3D_OK, "clear scratch");
        if draw {
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
                D3D_OK,
                "DrawPrimitiveUP"
            );
        }
        assert_eq!(
            h.set_render_target(0, &backbuffer),
            D3D_OK,
            "restore the back buffer"
        );
        assert_eq!(
            h.stretch_rect_rects(
                &scratch,
                (0, 0, 64, 64),
                &kept,
                (0, 0, 64, 64),
                D3DTEXF_NONE
            ),
            D3D_OK,
            "copy scratch into a corner of the kept target"
        );
        assert_eq!(h.end_scene(), D3D_OK, "EndScene");
        assert_eq!(h.present(), D3D_OK, "Present");
    };
    // Two frames make the scratch target one that is cleared every frame,
    // and their copies make it one whose content is kept.
    frame(false);
    frame(false);
    frame(true);
    assert_pixel_eq(
        read_rt_pixel(&h, &kept, 32, 32),
        RED,
        "a draw whose target is copied into kept content is never left out",
    );
}

/// A scratch target sampled into a kept one stays kept when a dropped clear-only pass precedes it.
///
/// Each frame clears a target nothing reads, a clear-only pass the pass
/// rules drop, then clears the scratch texture and samples it into a kept
/// target. Which pass sampled what is judged before those rules remove a
/// pass, so the dropped pass cannot shift the read off the kept pass, and
/// the draw into the scratch texture is built before its frame is submitted.
#[test]
fn a_scratch_texture_sampled_into_a_kept_target_after_a_dropped_clear_is_kept() {
    let h = async_device();
    let unread = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let scratch = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let scratch_surface = scratch.surface_level(0);
    let kept = h.create_render_target(128, 128, D3DFMT_A8R8G8B8);
    let tri = covering_triangle(RED);
    let textured = covering_triangle(0xFFFF_FFFF).map(|v| TexturedVertex {
        x: v.x,
        y: v.y,
        z: v.z,
        color: v.color,
        u: v.x.mul_add(0.5, 0.5),
        v: v.y.mul_add(-0.5, 0.5),
    });
    let frame = |draw: bool| {
        let backbuffer = h.render_target(0);
        assert!(h.pump(), "WM_QUIT before render");
        assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
        assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the back buffer");
        assert_eq!(
            h.set_render_target(0, &unread),
            D3D_OK,
            "bind the unread target"
        );
        assert_eq!(h.clear_target(GREEN), D3D_OK, "a clear nothing reads");
        assert_eq!(
            h.set_render_target(0, &scratch_surface),
            D3D_OK,
            "bind scratch"
        );
        assert_eq!(h.clear_target(GREEN), D3D_OK, "clear scratch");
        if draw {
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
                D3D_OK,
                "draw into scratch"
            );
        }
        assert_eq!(
            h.set_render_target(0, &kept),
            D3D_OK,
            "bind the kept target"
        );
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
            D3D_OK,
            "textured FVF"
        );
        assert_eq!(h.set_texture(0, &scratch), D3D_OK, "sample scratch");
        h.select_texture_stage(0);
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &textured),
            D3D_OK,
            "sample scratch into the kept target"
        );
        assert_eq!(h.clear_texture(0), D3D_OK, "unbind scratch");
        h.select_diffuse_stage(0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
        assert_eq!(
            h.set_render_target(0, &backbuffer),
            D3D_OK,
            "restore the back buffer"
        );
        assert_eq!(h.end_scene(), D3D_OK, "EndScene");
        assert_eq!(h.present(), D3D_OK, "Present");
    };
    frame(false);
    frame(false);
    frame(true);
    assert_pixel_eq(
        read_rt_pixel(&h, &kept, 64, 64),
        RED,
        "a draw into a texture a kept target samples is never left out",
    );
}

const YELLOW: u32 = 0xFFFF_FF00;

/// `ps_2_0 { def c0, r, g, b, 1; mov oC0, c0; }`, a first-seen shader writing `color`.
fn solid_ps(color: u32) -> [u32; 11] {
    let [b, g, r, _] = color.to_le_bytes();
    let unit = |c: u8| (f32::from(c) / 255.0).to_bits();
    [
        0xFFFF_0200, // ps_2_0
        0x0500_0051, // def
        0xA00F_0000, //   c0,
        unit(r),
        unit(g),
        unit(b),
        1.0f32.to_bits(),
        0x0200_0001, // mov
        0x800F_0800, //   oC0,
        0xA0E4_0000, //   c0
        0x0000_FFFF, // end
    ]
}

/// `ps_2_0 { mov oC0, oDepth; }`: `CreatePixelShader` accepts it, its library build rejects it.
///
/// A depth output is not a readable source, so no library comes of it and
/// every draw that binds it is dropped.
const PS_THAT_FAILS_TO_BUILD: [u32; 5] = [
    0xFFFF_0200, // ps_2_0
    0x0200_0001, // mov
    0x800F_0800, //   oC0,
    0x90E4_0800, //   oDepth
    0x0000_FFFF, // end
];

/// Two triangles covering the vertical band from clip-space `left` to `right`.
const fn band(left: f32, right: f32) -> [Vertex; 6] {
    const fn corner(x: f32, y: f32) -> Vertex {
        Vertex {
            x,
            y,
            z: 0.5,
            color: 0xFFFF_FFFF,
        }
    }
    [
        corner(left, -1.0),
        corner(left, 1.0),
        corner(right, -1.0),
        corner(right, -1.0),
        corner(left, 1.0),
        corner(right, 1.0),
    ]
}

/// The left, middle and right thirds of the viewport, in clip space.
const THIRDS: [(f32, f32); 3] = [
    (-1.0, -1.0 / 3.0),
    (-1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 1.0),
];

/// Draw one band per `(third, shader)`, each with its own pixel shader.
fn draw_bands(h: &Harness, bands: &[(usize, &mtld3d_tests::PixelShader<'_>)]) {
    for &(third, ps) in bands {
        let (left, right) = THIRDS[third];
        assert_eq!(h.set_pixel_shader(ps), D3D_OK, "SetPixelShader");
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &band(left, right)),
            D3D_OK,
            "DrawPrimitiveUP"
        );
    }
}

/// Three first-seen pixel shaders in one frame all show in that frame with the option off.
///
/// None of them may be left out, so each binds a placeholder, and the
/// submission builds all three before it goes to the GPU.
#[test]
fn three_first_seen_shaders_in_one_frame_all_show_in_it() {
    let h = device_with(SYNC);
    let shaders = [RED, GREEN, YELLOW].map(|color| h.create_pixel_shader(&solid_ps(color)));
    h.render_once(BLUE, |dev| {
        draw_bands(dev, &[(0, &shaders[0]), (1, &shaders[1]), (2, &shaders[2])]);
    });
    for (x, color) in [(106, RED), (320, GREEN), (533, YELLOW)] {
        assert_pixel_eq(
            h.read_pixel(x, 240),
            color,
            "every first-seen shader shows in the first frame that draws it",
        );
    }
}

/// Three first-seen pixel shaders into an uncleared target all show in their frame.
#[test]
fn three_first_seen_shaders_into_an_uncleared_target_all_show_in_their_frame() {
    let h = async_device();
    let rt = h.create_render_target(96, 32, D3DFMT_A8R8G8B8);
    let shaders = [RED, GREEN, YELLOW].map(|color| h.create_pixel_shader(&solid_ps(color)));
    let backbuffer = h.render_target(0);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the back buffer");
    assert_eq!(
        h.set_render_target(0, &rt),
        D3D_OK,
        "bind the offscreen target"
    );
    draw_bands(&h, &[(0, &shaders[0]), (1, &shaders[1]), (2, &shaders[2])]);
    assert_eq!(
        h.set_render_target(0, &backbuffer),
        D3D_OK,
        "restore the back buffer"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");
    for (x, color) in [(16, RED), (48, GREEN), (80, YELLOW)] {
        assert_pixel_eq(
            read_rt_pixel(&h, &rt, x, 16),
            color,
            "a target no clear rebuilds keeps every draw of the frame",
        );
    }
}

/// A draw whose library fails to build is removed from its pass, and the draws around it render.
///
/// The failed shader is drawn between two good ones in one pass, all three
/// first seen in that frame. Its placeholder and its draw go, the region it
/// covers keeps the clear, and the frame and the next one submit cleanly.
#[test]
fn a_draw_whose_library_fails_is_removed_and_the_draws_around_it_render() {
    let h = device_with(SYNC);
    let red = h.create_pixel_shader(&solid_ps(RED));
    let failing = h.create_pixel_shader(&PS_THAT_FAILS_TO_BUILD);
    let yellow = h.create_pixel_shader(&solid_ps(YELLOW));
    for frame in ["the frame the builds were queued in", "the next frame"] {
        h.render_once(BLUE, |dev| {
            draw_bands(dev, &[(0, &red), (1, &failing), (2, &yellow)]);
        });
        for (x, color) in [(106, RED), (320, BLUE), (533, YELLOW)] {
            assert_pixel_eq(
                h.read_pixel(x, 240),
                color,
                &format!("{frame}: the good draws render and the failed one leaves the clear"),
            );
        }
    }
}

/// A read-back in the frame that drew a first-seen shader sees the draw.
///
/// `GetRenderTargetData` submits the frame so far, so the placeholder is
/// resolved by that submission rather than at `Present`.
#[test]
fn a_read_back_in_the_frame_of_a_first_seen_draw_sees_it() {
    let h = device_with(SYNC);
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let ps = h.create_pixel_shader(&solid_ps(GREEN));
    let backbuffer = h.render_target(0);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.set_render_target(0, &rt),
        D3D_OK,
        "bind the offscreen target"
    );
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the offscreen target");
    draw_bands(&h, &[(0, &ps), (1, &ps), (2, &ps)]);
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        GREEN,
        "the read-back's submission builds the draw's shader first",
    );
    assert_eq!(
        h.set_render_target(0, &backbuffer),
        D3D_OK,
        "restore the back buffer"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");
}

/// Draw the three bands of `shaders` into `rt`, uncleared, in one frame, applying `state` first.
fn frame_of_bands(
    h: &Harness,
    rt: &Surface<'_>,
    shaders: &[mtld3d_tests::PixelShader<'_>; 3],
    state: impl Fn(&Harness, usize),
) {
    let backbuffer = h.render_target(0);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(h.set_render_target(0, rt), D3D_OK, "bind the target");
    for (third, ps) in shaders.iter().enumerate() {
        state(h, third);
        draw_bands(h, &[(third, ps)]);
    }
    assert_eq!(
        h.set_render_target(0, &backbuffer),
        D3D_OK,
        "restore the back buffer"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");
}

/// Built shaders drawn under render states they never met show in the first frame that draws them.
///
/// A first frame builds the three pixel shaders and their pipelines under
/// the default state. The next frame draws each into an uncleared target
/// under a blend or write-mask state of its own, each of which writes the
/// shader's colour: the libraries are built and only the pipeline is new,
/// so each draw binds a placeholder for the pipeline alone.
fn built_shaders_under_new_pipeline_states(entries: &'static str) {
    let h = device_with(entries);
    let warm = h.create_render_target(96, 32, D3DFMT_A8R8G8B8);
    let rt = h.create_render_target(96, 32, D3DFMT_A8R8G8B8);
    let shaders = [RED, GREEN, YELLOW].map(|color| h.create_pixel_shader(&solid_ps(color)));
    frame_of_bands(&h, &warm, &shaders, |_, _| {});
    frame_of_bands(&h, &rt, &shaders, |dev, third| {
        let set = |state, value| {
            assert_eq!(dev.set_render_state(state, value), D3D_OK, "SetRenderState");
        };
        match third {
            0 => {
                set(D3DRS_ALPHABLENDENABLE, 1);
                set(D3DRS_SRCBLEND, D3DBLEND_ONE);
                set(D3DRS_DESTBLEND, D3DBLEND_ZERO);
            }
            1 => {
                set(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
                set(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
            }
            _ => {
                set(D3DRS_ALPHABLENDENABLE, 0);
                set(D3DRS_COLORWRITEENABLE, 0x7);
            }
        }
    });
    for (x, color) in [(16, RED), (48, GREEN), (80, YELLOW)] {
        assert_pixel_eq(
            read_rt_pixel(&h, &rt, x, 16) | 0xFF00_0000,
            color,
            "a draw whose pipeline alone is new shows in its frame",
        );
    }
}

/// With the option off, built shaders under new pipeline states show in their frame.
#[test]
fn built_shaders_under_new_pipeline_states_show_in_their_frame() {
    built_shaders_under_new_pipeline_states(SYNC);
}

/// With the option on, built shaders under new pipeline states into an uncleared target show too.
#[test]
fn built_shaders_under_new_pipeline_states_into_an_uncleared_target_show_in_their_frame() {
    built_shaders_under_new_pipeline_states(ASYNC);
}

/// Sampling scratch content into retained stencil must protect the scratch producer.
fn stencil_sample_dependency(entries: &'static str) {
    use mtld3d_types::{
        D3DCLEAR_STENCIL, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_GREATER,
        D3DFMT_INTZ, D3DRS_ALPHAFUNC, D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_STENCILENABLE,
        D3DRS_STENCILFUNC, D3DRS_STENCILPASS, D3DRS_STENCILREF, D3DRS_ZENABLE, D3DSTENCILOP_KEEP,
        D3DSTENCILOP_REPLACE, D3DUSAGE_DEPTHSTENCIL,
    };
    let h = device_with(entries);
    let scratch = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let scratch_rt = scratch.surface_level(0);
    let ds = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let ds_surface = ds.surface_level(0);
    let backbuffer = h.render_target(0);
    let cold = h.create_pixel_shader(&solid_ps(RED));
    let textured = covering_triangle(0xFFFF_FFFF).map(|v| TexturedVertex {
        x: v.x,
        y: v.y,
        z: v.z,
        color: v.color,
        u: 0.5,
        v: 0.5,
    });
    // Clear both planes twice: depth, cleared again every frame below, is
    // regenerated from the first consumer frame on, while stencil is never
    // cleared again and so is retained.
    for _ in 0..2 {
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.set_depth_stencil_surface(&ds_surface), D3D_OK);
        assert_eq!(
            h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 0),
            D3D_OK
        );
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
        let _ = h.read_pixel(320, 240);
    }
    // The first read marks scratch when its frame is submitted; the second is
    // margin before the cold producer writes its mask.
    for frame in 0..3 {
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.clear_depth_stencil_surface(), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ALPHATESTENABLE, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 15), D3D_OK);
        assert_eq!(h.set_render_target(0, &scratch_rt), D3D_OK);
        assert_eq!(h.clear_target(0), D3D_OK);
        if frame == 2 {
            assert_eq!(h.set_pixel_shader(&cold), D3D_OK);
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(RED)),
                D3D_OK
            );
        }
        assert_eq!(h.clear_pixel_shader(), D3D_OK);
        assert_eq!(h.set_render_target(0, &backbuffer), D3D_OK);
        assert_eq!(h.set_depth_stencil_surface(&ds_surface), D3D_OK);
        assert_eq!(h.clear_target(BLUE), D3D_OK);
        assert_eq!(h.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_STENCILFUNC, D3DCMP_ALWAYS), D3D_OK);
        assert_eq!(
            h.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE),
            D3D_OK
        );
        assert_eq!(h.set_render_state(D3DRS_STENCILREF, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ALPHATESTENABLE, 1), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ALPHAFUNC, D3DCMP_GREATER), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ALPHAREF, 127), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0), D3D_OK);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
        assert_eq!(h.set_texture(0, &scratch), D3D_OK);
        h.select_texture_stage(0);
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &textured),
            D3D_OK
        );
        assert_eq!(h.clear_texture(0), D3D_OK);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
        let _ = h.read_pixel(320, 240);
    }
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.clear_target(BLUE), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ALPHATESTENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), D3D_OK);
    assert_eq!(
        h.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP),
        D3D_OK
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 15), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_triangle(GREEN)),
        D3D_OK
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
    assert_pixel_eq(
        h.read_pixel(320, 240),
        GREEN,
        "the retained stencil includes the scratch producer",
    );
}

/// A scratch draw contributing to retained stencil is kept with async compilation enabled.
#[test]
fn retained_stencil_keeps_its_sampled_scratch_producer() {
    stencil_sample_dependency(ASYNC);
}

/// The same retained-stencil workload renders with draw skipping disabled.
#[test]
fn retained_stencil_sample_dependency_with_skipping_disabled() {
    stencil_sample_dependency(SYNC);
}
