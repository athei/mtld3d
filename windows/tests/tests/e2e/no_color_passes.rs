//! Passes whose draws leave the colour target unwritten.
//!
//! Such a pass, with a depth attachment and no colour `Clear` of its own,
//! goes to the GPU with no colour attachment at all and its pipelines swapped
//! for no-colour variants. The colour target then has to come out of it
//! holding exactly what it held going in, whichever view of it the pass
//! would have bound: the sRGB view that `D3DRS_SRGBWRITEENABLE` selects, or
//! the multisampled surface behind a multisampled target. A view left
//! attached disagrees with the no-colour pipeline (the validation layer the
//! suite runs under rejects the draw) and discards its contents on store.
//!
//! Leaving the colour target out must not change the area the pass
//! rasterizes, with one exception: a 1x1 render target 0, alone and
//! unwritten, over a larger depth surface. There the draws reach the whole
//! depth surface, whether or not the target was cleared, whether the pass
//! also has draws that write it, and whether the pixel shader or the write
//! mask leaves it unwritten; a colour `Clear` still reaches the 1x1 target,
//! and a depth `Clear` bounded by a scissor reaches the part of the depth
//! surface it names. A 2x2 target keeps its own 2x2 area, and a depth surface
//! that `render.scale` reduces keeps the 1x1 area.

use mtld3d_tests::{Harness, RhwVertex, Surface, assert_pixel_eq, render_scale_is_identity};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESS, D3DFMT_A8R8G8B8,
    D3DFMT_D24S8, D3DFVF_DIFFUSE, D3DFVF_XYZRHW, D3DLOCK_READONLY, D3DMULTISAMPLE_4_SAMPLES,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_COLORWRITEENABLE, D3DRS_LIGHTING,
    D3DRS_SCISSORTESTENABLE, D3DRS_SRGBWRITEENABLE, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DTEXF_NONE, D3DVIEWPORT9,
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

/// Read the pixels at `points` of a single-sampled `width`x`height` render target.
fn render_target_pixels(
    h: &Harness,
    rt: &Surface<'_>,
    (width, height): (u32, u32),
    points: &[(u32, u32)],
) -> Vec<u32> {
    let sysmem =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(rt, &sysmem),
        D3D_OK,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let row = locked.as_u32((height * pitch_px) as usize);
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
    let pixels = render_target_pixels(
        &h,
        &resolve,
        (RT_SIZE, RT_SIZE),
        &[(4, middle), (RT_SIZE - 4, middle)],
    );
    assert_pixel_eq(pixels[0], RED, "the pass after the colour-masked one draws");
    assert_pixel_eq(
        pixels[1],
        BLUE,
        "the multisampled surface keeps the blue the first pass stored",
    );
}

/// Edge of the large render target and of the depth surface under the small target.
const BIG: u32 = 256;
/// [`BIG`] as the vertex positions state it.
const BIG_F: f32 = 256.0;
/// Where the depth surface is probed: its first texel, its middle, its last texel.
const PROBES: [(u32, u32); 3] = [(0, 0), (BIG / 2, BIG / 2), (BIG - 1, BIG - 1)];

/// A small render target 0 over a [`BIG`]-square depth surface, and a target to read it through.
struct SmallOverDepth<'h> {
    big: Surface<'h>,
    small: Surface<'h>,
    small_edge: u32,
    depth: Surface<'h>,
}

/// Bind `small_edge`-square render target 0 over a depth surface cleared to `depth_value`.
///
/// The large target is bound first with the depth surface and cleared red
/// with the depth at `depth_value`, in a pass of its own. Then the small
/// target replaces it, which resets the viewport to the small target, and
/// the viewport is set back to the whole depth surface. The scene stays
/// open for the caller's draws.
fn small_over_depth(h: &Harness, small_edge: u32, depth_value: f32) -> SmallOverDepth<'_> {
    let big = h.create_render_target(BIG, BIG, D3DFMT_A8R8G8B8);
    let small = h.create_render_target(small_edge, small_edge, D3DFMT_A8R8G8B8);
    let depth = h.create_depth_stencil_surface(BIG, BIG, D3DFMT_D24S8);
    arm(h);
    assert_eq!(
        h.set_render_target(0, &big),
        D3D_OK,
        "bind the large target"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&depth),
        D3D_OK,
        "bind the depth surface"
    );
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, depth_value, 0),
        D3D_OK,
        "clear the large target and the depth surface"
    );
    assert_eq!(
        h.set_render_target(0, &small),
        D3D_OK,
        "bind the small target"
    );
    let viewport = D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: BIG,
        height: BIG,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(
        h.set_viewport(&viewport),
        D3D_OK,
        "viewport over the depth surface"
    );
    SmallOverDepth {
        big,
        small,
        small_edge,
        depth,
    }
}

/// A full-depth-surface quad at `z`.
fn full_quad(z: f32, color: u32) -> [RhwVertex; 6] {
    quad(0.0, BIG_F, 0.0, BIG_F, z, color)
}

/// Draw a full quad at `z` with colour writes off.
fn masked_draw(h: &Harness, z: f32) {
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_quad(z, WHITE)),
        D3D_OK,
        "colour-masked draw over the depth surface"
    );
}

impl SmallOverDepth<'_> {
    /// Read the depth surface through the large target, then the small target's first pixel.
    ///
    /// The large target comes back with the same depth surface, and a green
    /// quad at z = 0.75 is drawn over it with depth writes off under `LESS`:
    /// a texel the earlier draws left at 0.5 keeps the red clear, a texel
    /// still at 1.0 turns green. Returns the three [`PROBES`] and the small
    /// target's pixel (0, 0).
    fn probe(&self, h: &Harness) -> ([u32; 3], u32) {
        assert_eq!(h.clear_pixel_shader(), D3D_OK, "fixed-function pixel stage");
        assert_eq!(
            h.set_render_target(0, &self.big),
            D3D_OK,
            "rebind the large target"
        );
        assert_eq!(
            h.set_depth_stencil_surface(&self.depth),
            D3D_OK,
            "same depth surface"
        );
        assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), D3D_OK);
        assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), D3D_OK);
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_quad(0.75, GREEN)),
            D3D_OK,
            "green depth probe"
        );
        assert_eq!(h.end_scene(), D3D_OK, "EndScene");
        let big = render_target_pixels(h, &self.big, (BIG, BIG), &PROBES);
        let small = render_target_pixels(
            h,
            &self.small,
            (self.small_edge, self.small_edge),
            &[(0, 0)],
        );
        ([big[0], big[1], big[2]], small[0])
    }
}

/// Assert each probe of the depth surface reads `expected`, in [`PROBES`] order.
fn assert_probes(probes: [u32; 3], expected: [u32; 3], what: &str) {
    for ((pixel, want), (x, y)) in probes.iter().zip(expected).zip(PROBES) {
        assert_pixel_eq(*pixel, want, &format!("{what}: depth probe at ({x}, {y})"));
    }
}

/// A cleared, unwritten 1x1 render target 0 over a larger depth surface draws at the depth extent.
///
/// The 1x1 target is cleared blue and then drawn with colour writes off. Its
/// draws reach every texel of the depth surface, and the clear still reaches
/// the target.
#[test]
fn a_cleared_unwritten_1x1_target_draws_over_the_whole_depth_surface() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET, BLUE, 1.0, 0),
        D3D_OK,
        "clear the 1x1 target"
    );
    masked_draw(&h, 0.5);
    let (probes, small) = s.probe(&h);
    assert_probes(
        probes,
        [RED; 3],
        "the masked draw wrote the whole depth surface",
    );
    assert_pixel_eq(small, BLUE, "the 1x1 target keeps its clear");
}

/// A draw writing the 1x1 target after a masked one leaves the masked draw at the depth extent.
///
/// The masked draw covers the whole depth surface at 0.5. The draw after it
/// writes the target and covers only the target's texel, at 0.25.
#[test]
fn a_1x1_target_written_after_a_masked_draw_keeps_the_masked_depth() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET, BLUE, 1.0, 0),
        D3D_OK,
        "clear the 1x1 target"
    );
    masked_draw(&h, 0.5);
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, 1.0, 0.0, 1.0, 0.25, WHITE)
        ),
        D3D_OK,
        "a draw writing the 1x1 target"
    );
    let (probes, small) = s.probe(&h);
    assert_probes(
        probes,
        [RED; 3],
        "the masked draw wrote the whole depth surface",
    );
    assert_pixel_eq(small, WHITE, "the writing draw reached the 1x1 target");
}

/// An uncleared, unwritten 1x1 render target 0 draws over the whole depth surface.
#[test]
fn an_uncleared_unwritten_1x1_target_draws_over_the_whole_depth_surface() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    masked_draw(&h, 0.5);
    let (probes, _) = s.probe(&h);
    assert_probes(
        probes,
        [RED; 3],
        "the masked draw wrote the whole depth surface",
    );
}

/// A 1x1 render target 0 that the draw writes keeps its one-texel area.
#[test]
fn a_written_1x1_target_keeps_its_one_texel_area() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_quad(0.5, BLUE)),
        D3D_OK,
        "a draw writing the 1x1 target"
    );
    let (probes, small) = s.probe(&h);
    assert_probes(
        probes,
        [RED, GREEN, GREEN],
        "the draw reached the target's texel alone",
    );
    assert_pixel_eq(small, BLUE, "the draw wrote the 1x1 target");
}

/// A cleared, unwritten 2x2 render target 0 keeps its own 2x2 area.
#[test]
fn a_cleared_unwritten_2x2_target_keeps_its_area() {
    let h = Harness::new();
    let s = small_over_depth(&h, 2, 1.0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET, BLUE, 1.0, 0),
        D3D_OK,
        "clear the 2x2 target"
    );
    masked_draw(&h, 0.5);
    let (probes, small) = s.probe(&h);
    assert_probes(
        probes,
        [RED, GREEN, GREEN],
        "the masked draw stayed inside the 2x2 area",
    );
    assert_pixel_eq(small, BLUE, "the 2x2 target keeps its clear");
}

/// An uncleared, unwritten 2x2 render target 0 keeps its own 2x2 area.
///
/// Leaving out a target nothing writes must not widen what the pass
/// rasterizes to the larger depth surface.
#[test]
fn an_uncleared_unwritten_2x2_target_keeps_its_area() {
    let h = Harness::new();
    let s = small_over_depth(&h, 2, 1.0);
    masked_draw(&h, 0.5);
    let (probes, _) = s.probe(&h);
    assert_probes(
        probes,
        [RED, GREEN, GREEN],
        "the masked draw stayed inside the 2x2 area",
    );
}

/// `ps_3_0 { def c0, 1,1,1,1; mov oC1, c0; }`: writes render target 1 alone, never `oC0`.
#[rustfmt::skip]
const PS_OC1_ONLY: [u32; 11] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x3F80_0000, 0x3F80_0000, 0x3F80_0000, 0x3F80_0000, //   1, 1, 1, 1
    0x0200_0001, 0x800F_0801, 0xA0E4_0000,              // mov oC1, c0
    0x0000_FFFF,                                        // end
];

/// A 1x1 render target 0 the pixel shader never writes draws over the whole depth surface.
///
/// The write mask stays at 0xF; the shader writes only `oC1`, and no render
/// target 1 is bound.
#[test]
fn a_1x1_target_the_shader_leaves_unwritten_draws_over_the_whole_depth_surface() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    let ps = h.create_pixel_shader(&PS_OC1_ONLY);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK, "oC1-only pixel shader");
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_quad(0.5, WHITE)),
        D3D_OK,
        "a draw whose shader writes no oC0"
    );
    let (probes, _) = s.probe(&h);
    assert_probes(probes, [RED; 3], "the draw wrote the whole depth surface");
}

/// A scissored depth `Clear` over an unwritten 1x1 target reaches the half of the depth it names.
///
/// The masked draw lays 0.5 over the whole depth surface, then a depth clear
/// under a scissor over the right half puts that half back to 1.0.
#[test]
fn a_scissored_depth_clear_over_an_unwritten_1x1_target_reaches_its_half() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 1.0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET, BLUE, 1.0, 0),
        D3D_OK,
        "clear the 1x1 target"
    );
    masked_draw(&h, 0.5);
    let right_half = D3DRECT {
        x1: 128,
        y1: 0,
        x2: 256,
        y2: 256,
    };
    assert_eq!(
        h.set_scissor_rect(&right_half),
        D3D_OK,
        "scissor over the right half"
    );
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), D3D_OK);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0),
        D3D_OK,
        "depth clear of the right half"
    );
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 0), D3D_OK);
    let (probes, small) = s.probe(&h);
    assert_probes(
        probes,
        [RED, GREEN, GREEN],
        "the right half was cleared back, the left half kept the draw",
    );
    let left = render_target_pixels(&h, &s.big, (BIG, BIG), &[(64, 128)]);
    assert_pixel_eq(left[0], RED, "the left half keeps the masked draw's depth");
    assert_pixel_eq(small, BLUE, "the 1x1 target keeps its clear");
}

/// A depth `Clear` issued through a 1x1 render target 0 reaches the whole depth surface.
///
/// The depth surface starts at 0.0. With the 1x1 target bound, one `Clear`
/// of the target and the depth follows, and nothing is drawn before the
/// large target comes back.
#[test]
fn a_depth_clear_through_a_1x1_target_reaches_the_whole_depth_surface() {
    let h = Harness::new();
    let s = small_over_depth(&h, 1, 0.0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLUE, 1.0, 0),
        D3D_OK,
        "clear the 1x1 target and the depth surface"
    );
    let (probes, small) = s.probe(&h);
    assert_probes(probes, [GREEN; 3], "the depth clear reached every texel");
    assert_pixel_eq(small, BLUE, "the 1x1 target keeps its clear");
}

/// An unwritten 1x1 render target 0 over the back buffer's depth surface, at either scale.
///
/// A masked quad over the left half at z = 0.5, then a green quad over the
/// whole back buffer at z = 0.75. At the identity scale the masked draw
/// reaches the left half of the depth surface. Under a reduced
/// `render.scale` that depth surface is rasterized smaller, the pass keeps
/// the 1x1 target's area, and only the first texel of the depth surface
/// changes.
#[test]
fn an_unwritten_1x1_target_over_the_back_buffer_depth_follows_the_render_scale() {
    let h = Harness::with_depth();
    let (width, height) = back_buffer_extent(&h);
    let (width_px, height_px) = h.dims();
    let back_buffer = h.back_buffer(0);
    let small = h.create_render_target(1, 1, D3DFMT_A8R8G8B8);
    arm(&h);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, 1.0, 0),
        D3D_OK,
        "clear colour + depth"
    );
    assert_eq!(
        h.set_render_target(0, &small),
        D3D_OK,
        "bind the 1x1 target"
    );
    let viewport = D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: width_px,
        height: height_px,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(
        h.set_viewport(&viewport),
        D3D_OK,
        "viewport over the back buffer"
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, width / 2.0, 0.0, height, 0.5, WHITE)
        ),
        D3D_OK,
        "masked draw over the left half"
    );
    assert_eq!(
        h.set_render_target(0, &back_buffer),
        D3D_OK,
        "rebind the back buffer"
    );
    assert_eq!(h.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &quad(0.0, width, 0.0, height, 0.75, GREEN)
        ),
        D3D_OK,
        "green depth probe"
    );
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    let left = h.read_pixel(width_px / 4, height_px / 2);
    let right = h.read_pixel(width_px * 5 / 8, height_px / 2);
    if render_scale_is_identity() {
        assert_pixel_eq(left, RED, "the masked draw reached the left half");
    } else {
        assert_pixel_eq(left, GREEN, "a scaled depth surface keeps the 1x1 area");
    }
    assert_pixel_eq(right, GREEN, "the right half was never drawn");
}
