//! First-use shader and pipeline builds on the encoder's worker threads.
//!
//! Every test here runs under `shader.asyncCompile = true` with the shader
//! cache off, so each draw's libraries and pipeline are new to the device and
//! build on a worker. A draw whose build is still in flight is left out of
//! the frame only when its target is rebuilt every frame (the back buffer
//! under the discard swap effect, or a target cleared in this frame and the
//! one before, and read by nothing kept) and no occlusion query is
//! counting; a draw into any other target waits for its build. The rest of
//! the suite runs with the option off, where every such draw waits.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, Surface, Vertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN,
    D3DISSUE_END, D3DLOCK_READONLY, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DQUERYTYPE_OCCLUSION,
    D3DRS_LIGHTING, D3DTEXF_NONE,
};

const ASYNC: &str = "shader.asyncCompile=true;shaderCache.enable=false";

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

/// A draw into an offscreen target no clear reached this frame waits for its build.
#[test]
fn a_draw_into_an_uncleared_target_waits_for_its_build() {
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

/// A target cleared only in this frame may be a one-off render, so its draw waits.
#[test]
fn a_draw_into_a_target_cleared_only_this_frame_waits_for_its_build() {
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

/// A first-seen draw inside a counting occlusion query waits, so the query counts its samples.
#[test]
fn a_draw_counted_by_an_occlusion_query_waits_for_its_build() {
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

/// A scratch target copied into a kept one every frame is kept too, so its draw waits.
#[test]
fn a_draw_into_a_scratch_target_copied_into_a_kept_one_waits_for_its_build() {
    let h = async_device();
    let scratch = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let kept = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
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
            h.stretch_rect(&scratch, &kept, D3DTEXF_NONE),
            D3D_OK,
            "copy scratch into the kept target"
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
