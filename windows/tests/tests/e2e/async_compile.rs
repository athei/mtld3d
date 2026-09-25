//! First-use shader and pipeline builds on the encoder's worker threads.
//!
//! Every test here runs under `shader.asyncCompile = true` with the shader
//! cache off, so each draw's libraries and pipeline are new to the device and
//! build on a worker. A draw whose build is still in flight is left out of
//! the frame only when its target is redrawn every frame (the back buffer, or
//! a target cleared earlier in the frame); a draw into any other target waits
//! for its build. The rest of the suite runs with the option off, where every
//! such draw waits.

use std::time::{Duration, Instant};

use mtld3d_tests::{Harness, Surface, Vertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRS_LIGHTING,
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
    let h = Harness::with_config(ASYNC);
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

/// Draw `tri` into `rt`, cleared to `clear` first when it is given, inside one frame.
fn frame_into_target(h: &Harness, rt: &Surface<'_>, clear: Option<u32>, tri: &[Vertex; 3]) {
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
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, tri),
        D3D_OK,
        "DrawPrimitiveUP"
    );
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
    frame_into_target(&h, &rt, None, &covering_triangle(RED));
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        RED,
        "a target the frame did not clear may be read for the rest of its life, \
         so its first draw is never left out",
    );
}

/// A draw into an offscreen target cleared earlier in the frame is left out like a back-buffer one.
#[test]
fn a_draw_into_a_target_cleared_this_frame_is_left_out_until_its_build_lands() {
    let h = async_device();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let tri = covering_triangle(RED);
    frame_into_target(&h, &rt, Some(GREEN), &tri);
    assert_pixel_eq(
        read_rt_pixel(&h, &rt, 32, 32),
        GREEN,
        "the first frame leaves the draw out and keeps the clear",
    );
    let deadline = Instant::now() + BUILD_DEADLINE;
    let mut frames = 1u32;
    loop {
        frame_into_target(&h, &rt, Some(GREEN), &tri);
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
