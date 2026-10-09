//! Textures and the viewport across a same-size `Reset` that changes only the present interval.
//!
//! A `Reset` that keeps the back-buffer size and multisampling skips the
//! device's surface recreate: it delivers the open frame, restores the state
//! defaults and queues the pacing change. `device.rs` pins an upload queued
//! before such a `Reset` and the stage binding masks it clears. These tests
//! add a texture released by the application while stage 0 still holds it, a
//! `Reset` that changes only the presentation interval, replacements created
//! after the `Reset` in the managed and dynamic default pools and at another
//! shape, a state block recorded before the `Reset` and applied after it, and
//! the fullscreen same-size `Reset`. The replacement must show its own texels,
//! and the draws after the `Reset` must keep the extent of the full-target
//! viewport the `Reset` restores.

use mtld3d_tests::{Harness, Texture, TexturedVertex, assert_pixel_eq, enumerate_display_sizes};
use mtld3d_types::{
    D3D_OK, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DFVF_XYZRHW,
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPRESENT_INTERVAL_IMMEDIATE, D3DPRESENT_INTERVAL_ONE,
    D3DPT_TRIANGLELIST, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_DYNAMIC, D3DVIEWPORT9,
};

const BLACK: u32 = 0xFF00_0000;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const WHITE: u32 = 0xFFFF_FFFF;

/// The released texture's size.
const RELEASED_SIZE: (u32, u32) = (64, 64);

/// The pretransformed quad's pixel rect, `[left, right) x [top, bottom)`.
const RHW_RECT: (u16, u16, u16, u16) = (100, 300, 100, 200);

/// Pixels the probes keep away from an edge, so the scaled leg reads the same answer.
const MARGIN: u32 = 4;

/// Pretransformed position, diffuse colour and one texture coordinate.
#[repr(C)]
struct RhwTexturedVertex {
    x: f32,
    y: f32,
    z: f32,
    rhw: f32,
    color: u32,
    u: f32,
    v: f32,
}

/// When the replacement texture is created and filled, relative to the `Reset`.
enum Made {
    /// Before it, while the released texture is still alive through its binding.
    BeforeReset,
    /// After it, once the `Reset`'s unbind has let the released texture go.
    AfterReset,
}

/// One run of the release, replace, `Reset`, rebind sequence.
struct Scenario {
    made: Made,
    pool: u32,
    usage: u32,
    size: (u32, u32),
    /// A viewport set just before the `Reset`, which the `Reset` must replace.
    viewport_before: Option<D3DVIEWPORT9>,
    /// Bind the replacement through a state block recorded before the `Reset`.
    through_state_block: bool,
}

impl Scenario {
    /// A managed replacement the size and format of the released texture.
    const fn managed(made: Made) -> Self {
        Self {
            made,
            pool: D3DPOOL_MANAGED,
            usage: 0,
            size: RELEASED_SIZE,
            viewport_before: None,
            through_state_block: false,
        }
    }
}

/// A texture whose left half is `left` and right half is `right`.
fn two_colour_texture(
    h: &Harness,
    (width, height): (u32, u32),
    pool: u32,
    usage: u32,
    left: u32,
    right: u32,
) -> Texture<'_> {
    let tex = h.create_texture(width, height, 1, usage, D3DFMT_A8R8G8B8, pool);
    let (w, rows) = (width as usize, height as usize);
    let texels: Vec<u32> = (0..w * rows)
        .map(|i| if i % w < w / 2 { left } else { right })
        .collect();
    tex.lock_rect(0, 0).write_u32_rect(w, rows, &texels);
    tex
}

/// Sample stage 0's texture as is, point-filtered and clamped.
fn arm_stage0(h: &Harness) {
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(
            h.set_sampler_state(0, state, value),
            D3D_OK,
            "SetSamplerState"
        );
    }
}

/// The pretransformed quad over [`RHW_RECT`], its texture coordinates spanning the texture.
fn rhw_quad() -> [RhwTexturedVertex; 6] {
    let (left, right, top, bottom) = RHW_RECT;
    let corner = |x: u16, y: u16, u, v| RhwTexturedVertex {
        x: f32::from(x) - 0.5,
        y: f32::from(y) - 0.5,
        z: 0.5,
        rhw: 1.0,
        color: WHITE,
        u,
        v,
    };
    [
        corner(left, top, 0.0, 0.0),
        corner(right, top, 1.0, 0.0),
        corner(left, bottom, 0.0, 1.0),
        corner(right, top, 1.0, 0.0),
        corner(right, bottom, 1.0, 1.0),
        corner(left, bottom, 0.0, 1.0),
    ]
}

/// The bottom-left quarter of clip space, sampling the texture's right half.
///
/// Its pixel extent is the viewport's, so a viewport left over from before the
/// `Reset` shows as a smaller or displaced quad.
fn clip_quad() -> [TexturedVertex; 6] {
    let corner = |x, y| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u: 0.75,
        v: 0.5,
    };
    [
        corner(-1.0, -0.5),
        corner(-0.5, -0.5),
        corner(-1.0, -1.0),
        corner(-0.5, -0.5),
        corner(-0.5, -1.0),
        corner(-1.0, -1.0),
    ]
}

/// One frame that draws both quads with whatever stage 0 holds.
fn draw_frame(h: &Harness) {
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_TEX1),
            D3D_OK,
            "SetFVF pretransformed"
        );
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &rhw_quad()),
            D3D_OK,
            "pretransformed draw"
        );
        assert_eq!(
            d.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
            D3D_OK,
            "SetFVF untransformed"
        );
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &clip_quad()),
            D3D_OK,
            "untransformed draw"
        );
    });
}

/// Check both quads show `left`/`right` at the extent a full-target viewport gives them.
fn assert_frame(h: &Harness, left: u32, right: u32, what: &str) {
    let (width, height) = h.dims();
    let (quad_left, quad_right, quad_top, quad_bottom) = (
        u32::from(RHW_RECT.0),
        u32::from(RHW_RECT.1),
        u32::from(RHW_RECT.2),
        u32::from(RHW_RECT.3),
    );
    let mid_x = u32::midpoint(quad_left, quad_right);
    let mid_y = u32::midpoint(quad_top, quad_bottom);
    let probes = [
        (
            quad_left + MARGIN,
            mid_y,
            left,
            "pretransformed quad, left half",
        ),
        (
            quad_right - MARGIN,
            mid_y,
            right,
            "pretransformed quad, right half",
        ),
        (
            quad_left - MARGIN,
            mid_y,
            BLACK,
            "left of the pretransformed quad",
        ),
        (
            quad_right + MARGIN,
            mid_y,
            BLACK,
            "right of the pretransformed quad",
        ),
        (
            mid_x,
            quad_top - MARGIN,
            BLACK,
            "above the pretransformed quad",
        ),
        (
            mid_x,
            quad_bottom + MARGIN,
            BLACK,
            "below the pretransformed quad",
        ),
        // The clip quad covers x in [0, w/4) and y in [3h/4, h) of a full viewport.
        (
            width / 4 - MARGIN,
            height * 3 / 4 + MARGIN,
            right,
            "clip quad, inner corner",
        ),
        (MARGIN, height - MARGIN, right, "clip quad, outer corner"),
        (
            width / 4 + MARGIN,
            height - MARGIN,
            BLACK,
            "right of the clip quad",
        ),
        (
            MARGIN,
            height * 3 / 4 - MARGIN,
            BLACK,
            "above the clip quad",
        ),
    ];
    for (x, y, expected, place) in probes {
        assert_pixel_eq(
            h.read_pixel(x, y),
            expected,
            &format!("{what}: {place} at ({x}, {y})"),
        );
    }
}

/// Run `s`: draw with texture A, release it while bound, sample replacement B after the `Reset`.
fn run(h: &Harness, s: &Scenario) {
    let full = {
        let (width, height) = h.dims();
        D3DVIEWPORT9 {
            x: 0,
            y: 0,
            width,
            height,
            min_z: 0.0,
            max_z: 1.0,
        }
    };
    let make_b = || two_colour_texture(h, s.size, s.pool, s.usage, GREEN, BLUE);
    let a = two_colour_texture(h, RELEASED_SIZE, D3DPOOL_MANAGED, 0, RED, RED);
    assert_eq!(h.set_texture(0, &a), D3D_OK, "SetTexture(A)");
    arm_stage0(h);
    draw_frame(h);
    assert_frame(h, RED, RED, "texture A before the Reset");
    // The last frame before the `Reset` samples A too, and A stays bound.
    draw_frame(h);
    let released = a.as_ptr();
    drop(a);

    let early = matches!(s.made, Made::BeforeReset).then(make_b);
    let block = s.through_state_block.then(|| {
        let b = early
            .as_ref()
            .expect("a state block binds a texture made before the Reset");
        assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
        assert_eq!(h.set_texture(0, b), D3D_OK, "SetTexture(B) recorded");
        arm_stage0(h);
        h.end_state_block()
    });
    if let Some(vp) = s.viewport_before {
        assert_eq!(h.set_viewport(&vp), D3D_OK, "SetViewport before the Reset");
    }

    let (hr, mut pp) = h.implicit_swapchain().present_parameters();
    assert_eq!(hr, D3D_OK, "GetPresentParameters");
    let before = (pp.back_buffer_width, pp.back_buffer_height);
    pp.presentation_interval = if pp.presentation_interval == D3DPRESENT_INTERVAL_IMMEDIATE {
        D3DPRESENT_INTERVAL_ONE
    } else {
        D3DPRESENT_INTERVAL_IMMEDIATE
    };
    assert_eq!(
        h.reset_params(&mut pp),
        D3D_OK,
        "same-size Reset changing the interval"
    );
    assert_eq!(
        (pp.back_buffer_width, pp.back_buffer_height),
        before,
        "the Reset keeps the back buffer size"
    );
    let vp = h.viewport();
    assert_eq!(
        (vp.x, vp.y, vp.width, vp.height),
        (0, 0, full.width, full.height),
        "the Reset restores the full-target viewport"
    );

    let late = matches!(s.made, Made::AfterReset).then(make_b);
    // The D3D9 object pointer is one identity the replacement may reuse, and no
    // test can arrange that. Run directly, the test binary says whether it did.
    if let Some(b) = &late {
        eprintln!(
            "{}: the replacement {} the released texture's object pointer",
            std::thread::current().name().unwrap_or("same_size_reset"),
            if b.as_ptr() == released {
                "reuses"
            } else {
                "does not reuse"
            },
        );
    }
    let b = early.as_ref().or(late.as_ref()).expect("one texture B");
    if let Some(block) = &block {
        assert_eq!(block.apply(), D3D_OK, "Apply the state block");
    } else {
        assert_eq!(h.set_texture(0, b), D3D_OK, "SetTexture(B)");
        arm_stage0(h);
    }
    for _ in 0..3 {
        draw_frame(h);
    }
    assert_frame(h, GREEN, BLUE, "texture B after the Reset");
    drop(block);
}

/// A replacement made before the `Reset`, and a smaller viewport the `Reset` replaces.
#[test]
fn a_texture_made_before_a_same_size_reset_samples_at_the_restored_viewport() {
    let h = Harness::new();
    run(
        &h,
        &Scenario {
            viewport_before: Some(D3DVIEWPORT9 {
                x: 0,
                y: 0,
                width: 320,
                height: 240,
                min_z: 0.0,
                max_z: 1.0,
            }),
            ..Scenario::managed(Made::BeforeReset)
        },
    );
}

/// A managed replacement made after the `Reset` let the released texture go.
#[test]
fn a_managed_texture_made_after_a_same_size_reset_samples_its_own_texels() {
    let h = Harness::new();
    run(&h, &Scenario::managed(Made::AfterReset));
}

/// A dynamic default-pool replacement, which only a texture made after the `Reset` can be.
#[test]
fn a_dynamic_default_texture_made_after_a_same_size_reset_samples_its_own_texels() {
    let h = Harness::new();
    run(
        &h,
        &Scenario {
            pool: D3DPOOL_DEFAULT,
            usage: D3DUSAGE_DYNAMIC,
            ..Scenario::managed(Made::AfterReset)
        },
    );
}

/// A replacement of the released texture's byte size but another shape, made after the `Reset`.
#[test]
fn a_reshaped_texture_made_after_a_same_size_reset_samples_its_own_texels() {
    let h = Harness::new();
    run(
        &h,
        &Scenario {
            size: (128, 32),
            ..Scenario::managed(Made::AfterReset)
        },
    );
}

/// A state block recorded before the `Reset` binds the replacement after it.
#[test]
fn a_state_block_from_before_a_same_size_reset_binds_the_replacement() {
    let h = Harness::new();
    run(
        &h,
        &Scenario {
            through_state_block: true,
            ..Scenario::managed(Made::BeforeReset)
        },
    );
}

/// The fullscreen form, whose same-size `Reset` also re-applies the fullscreen window.
#[test]
fn a_fullscreen_same_size_reset_keeps_the_replacement_at_full_size() {
    // A fullscreen `Reset` sets the mode through user32, which accepts only a
    // mode the display lists; on a single-mode display the request takes the
    // non-mode path, which `device.rs` tests, and there is nothing to measure.
    if !enumerate_display_sizes().contains(&(640, 480)) {
        return;
    }
    let h = Harness::fullscreen(640, 480);
    run(&h, &Scenario::managed(Made::BeforeReset));
}
