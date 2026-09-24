//! Passes whose every draw has `D3DRS_COLORWRITEENABLE` at 0.
//!
//! Such a pass, with a depth attachment and no colour `Clear` of its own,
//! goes to the GPU with no colour attachment at all and its pipelines swapped
//! for no-colour variants. The colour target then has to come out of it
//! holding exactly what it held going in, whichever view of it the pass
//! would have bound: the sRGB view that `D3DRS_SRGBWRITEENABLE` selects, or
//! the multisampled surface behind a multisampled target. A view left
//! attached disagrees with the no-colour pipeline (the validation layer the
//! suite runs under rejects the draw) and discards its contents on store.

use mtld3d_tests::{Harness, RhwVertex, Surface, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESS, D3DFMT_A8R8G8B8,
    D3DFMT_D24S8, D3DFVF_DIFFUSE, D3DFVF_XYZRHW, D3DLOCK_READONLY, D3DMULTISAMPLE_4_SAMPLES,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRS_COLORWRITEENABLE, D3DRS_LIGHTING,
    D3DRS_SRGBWRITEENABLE, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DTEXF_NONE,
};

const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;
const WHITE: u32 = 0xFFFF_FFFF;

/// Edge of the multisampled render target.
const RT_SIZE: u32 = 64;
/// [`RT_SIZE`] as the vertex positions state it.
const RT_SIZE_F: f32 = 64.0;

/// A pre-transformed quad over `left..right` by `top..bottom` at depth `z`, in `color`.
fn quad(left: f32, right: f32, top: f32, bottom: f32, z: f32, color: u32) -> [RhwVertex; 6] {
    let v = |x: f32, y: f32| RhwVertex {
        x,
        y,
        z,
        rhw: 1.0,
        color,
    };
    [
        v(left, top),
        v(right, top),
        v(left, bottom),
        v(right, top),
        v(right, bottom),
        v(left, bottom),
    ]
}

/// Arm the fixed-function pipeline for unlit pre-transformed draws under `ZFUNC` less.
fn arm(h: &Harness) {
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), D3D_OK, "SetFVF");
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );
    h.select_diffuse_stage(0);
    assert_eq!(
        h.set_render_state(D3DRS_ZENABLE, 1),
        D3D_OK,
        "depth test on"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZWRITEENABLE, 1),
        D3D_OK,
        "depth writes on"
    );
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
}

/// The back buffer's extent as the pre-transformed vertex positions state it.
fn back_buffer_extent(h: &Harness) -> (f32, f32) {
    let (width, height) = h.dims();
    (
        f32::from(u16::try_from(width).expect("back-buffer width fits u16")),
        f32::from(u16::try_from(height).expect("back-buffer height fits u16")),
    )
}

/// Read the pixels at `points` of a single-sampled `RT_SIZE`-square render target.
fn render_target_pixels(h: &Harness, rt: &Surface<'_>, points: &[(u32, u32)]) -> Vec<u32> {
    let sysmem =
        h.create_offscreen_plain_surface(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(rt, &sysmem),
        D3D_OK,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let row = locked.as_u32((RT_SIZE * pitch_px) as usize);
    points
        .iter()
        .map(|&(x, y)| row[(y * pitch_px + x) as usize])
        .collect()
}

/// A colour-masked pass encoding through the back buffer's sRGB view keeps the colour.
///
/// The frame is three passes on the back buffer and its depth buffer: a
/// blue fill with sRGB writes off, a nearer full-screen draw with sRGB
/// writes on and colour writes off, which only lays down depth, and then
/// sRGB writes off again with two draws that test against that depth. Each
/// change of `D3DRS_SRGBWRITEENABLE` ends the pass, since the view is chosen
/// when a pass opens. The middle pass is the colour-masked one, and it
/// would bind the sRGB view. Afterwards the left half, where the last green
/// draw sits behind the middle pass's depth, still holds the first pass's
/// blue, and the right half holds the last red draw, which is nearer still.
#[test]
fn a_colour_masked_pass_on_the_srgb_view_keeps_the_back_buffer() {
    let h = Harness::with_depth();
    let (width, height) = back_buffer_extent(&h);
    let half = width / 2.0;
    arm(&h);

    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3D_OK,
        "clear colour + depth"
    );
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, width, 0.0, height, 0.5, BLUE)
        ),
        D3D_OK,
        "blue fill"
    );

    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 1), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, width, 0.0, height, 0.25, WHITE)
        ),
        D3D_OK,
        "colour-masked depth draw"
    );

    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, half, 0.0, height, 0.375, GREEN)
        ),
        D3D_OK,
        "green behind the masked pass's depth"
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(half, width, 0.0, height, 0.125, RED)
        ),
        D3D_OK,
        "red in front of it"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(h.present(), D3D_OK, "Present");

    let (width_px, height_px) = h.dims();
    assert_pixel_eq(
        h.read_pixel(width_px / 4, height_px / 2),
        BLUE,
        "the colour-masked pass leaves the blue fill, and its depth hides the green",
    );
    assert_pixel_eq(
        h.read_pixel(width_px * 3 / 4, height_px / 2),
        RED,
        "the pass after the colour-masked one draws",
    );
}

/// A colour-masked pass between two multisampled colour passes keeps the samples.
///
/// A 4x render target drawn blue under one 4x depth surface, then a
/// colour-masked draw under a second 4x depth surface, then a red strip over
/// the left quarter under the first depth surface again: each depth change
/// ends the pass. The colour-masked pass is not the target's last in the
/// submission, so it does not take the resolve, and the third pass loads the
/// multisampled surface it left. The resolve through `StretchRect` then
/// shows the red strip and, right of it, the blue the first pass stored.
#[test]
fn a_colour_masked_pass_between_multisampled_passes_keeps_the_samples() {
    let h = Harness::new();
    arm(&h);
    let target = h.create_render_target_ms(
        (RT_SIZE, RT_SIZE),
        D3DFMT_A8R8G8B8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    let resolve = h.create_render_target(RT_SIZE, RT_SIZE, D3DFMT_A8R8G8B8);
    let depth = |what: &str| {
        let (hr, surface) = h.create_depth_stencil_surface_ms_hr(
            (RT_SIZE, RT_SIZE),
            D3DFMT_D24S8,
            (D3DMULTISAMPLE_4_SAMPLES, 0),
        );
        assert_eq!(hr, D3D_OK, "CreateDepthStencilSurface(4x) for the {what}");
        surface.expect("multisampled depth surface")
    };
    let scene_depth = depth("scene");
    let masked_depth = depth("colour-masked pass");
    assert_eq!(
        h.set_render_target(0, &target),
        D3D_OK,
        "SetRenderTarget(4x)"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&scene_depth),
        D3D_OK,
        "scene depth"
    );

    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3D_OK,
        "clear colour + depth"
    );
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, RT_SIZE_F, 0.0, RT_SIZE_F, 0.5, BLUE)
        ),
        D3D_OK,
        "blue fill"
    );

    assert_eq!(
        h.set_depth_stencil_surface(&masked_depth),
        D3D_OK,
        "colour-masked pass depth"
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, RT_SIZE_F, 0.0, RT_SIZE_F, 0.25, WHITE)
        ),
        D3D_OK,
        "colour-masked depth draw"
    );

    assert_eq!(
        h.set_depth_stencil_surface(&scene_depth),
        D3D_OK,
        "scene depth again"
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, RT_SIZE_F / 4.0, 0.0, RT_SIZE_F, 0.25, RED)
        ),
        D3D_OK,
        "red strip"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");

    assert_eq!(
        h.stretch_rect(&target, &resolve, D3DTEXF_NONE),
        D3D_OK,
        "StretchRect resolve"
    );
    let middle = RT_SIZE / 2;
    let pixels = render_target_pixels(&h, &resolve, &[(4, middle), (RT_SIZE - 4, middle)]);
    assert_pixel_eq(pixels[0], RED, "the pass after the colour-masked one draws");
    assert_pixel_eq(
        pixels[1],
        BLUE,
        "the multisampled surface keeps the blue the first pass stored",
    );
}
