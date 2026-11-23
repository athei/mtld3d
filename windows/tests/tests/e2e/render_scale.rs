//! Coordinate-space coverage for `render.scale`.
//!
//! `render.scale` rasterizes the back buffer smaller than the resolution D3D9
//! reports and lets `MetalFX` resolve it on the way out. Every coordinate a game
//! supplies stays in the reported space, so each test here asserts on reported
//! coordinates and must pass **unchanged** at any scale.
//!
//! That is the point: run the suite normally for the default path, and
//! `make test SCALE=0.75` (or `0.5`, or a non-dividing `0.67`) to prove the two
//! spaces never leaked into each other. A test that only held at 1.0 would be
//! asserting on the render space by accident.
//!
//! Exact-colour comparisons survive resampling because every region asserted on
//! is a flat block several pixels from any colour boundary, where a filtered
//! resample reproduces the source colour exactly. Sampling *on* a boundary would
//! be scale-dependent by construction, so the probes stay away from them.
//!
//! A few tests set `render.scale` themselves instead of inheriting the run's,
//! because the quantities they cover (a point's rasterized diameter, and the
//! memory a resource created at the reported back-buffer size occupies) convert
//! between the two spaces rather than staying in one, so a run at the default
//! scale would not exercise the conversion at all. The two-device test pins
//! both of its scales for a different reason: what it covers is a pair of
//! devices at *different* scales, which no single run-wide value gives it.

use std::sync::atomic::{AtomicU32, Ordering};

use mtld3d_tests::{
    Harness, HarnessConfig, PosColorVertex, RhwVertex, SharedDevice, Surface, Texture,
    TexturedVertex, assert_pixel_approx, assert_pixel_eq, spawn_scoped,
};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESS, D3DCMP_LESSEQUAL,
    D3DCREATE_HARDWARE_VERTEXPROCESSING, D3DCREATE_MULTITHREADED, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8,
    D3DFMT_A16B16G16R16F, D3DFMT_A32B32G32R32F, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_INTZ,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DLOCK_READONLY,
    D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPT_POINTLIST, D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP,
    D3DRECT, D3DRS_LIGHTING, D3DRS_POINTSIZE, D3DRS_SCISSORTESTENABLE, D3DRS_STENCILENABLE,
    D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV,
    D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_RENDERTARGET, D3DVIEWPORT9,
};

use super::{
    device::{await_logged_lines, logged_lines, run_in_private_log_child, running_as},
    render_target::{draw_fill, read_back, sample_mip_level_4},
    samplers::PS_SAMPLE_TEXTURE,
};

const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const BLACK: u32 = 0xFF00_0000;
const WHITE: u32 = 0xFFFF_FFFF;
/// A colour whose alpha is neither zero nor opaque, so either mistake shows.
const TRANSLUCENT: u32 = 0x4000_FF00;
/// Red as GDI's `COLORREF` (`0x00BBGGRR`) reports it.
const RED_COLORREF: u32 = 0x0000_00FF;

/// The sub-rect every viewport-bound test narrows to, in reported coordinates.
///
/// Deliberately not a clean fraction of the 640x480 frame, and every probe
/// below sits at least 50 reported pixels from one of its edges so a rounding
/// difference at a non-dividing scale cannot move a probe across a boundary.
const NARROW: D3DVIEWPORT9 = D3DVIEWPORT9 {
    x: 128,
    y: 96,
    width: 384,
    height: 288,
    min_z: 0.0,
    max_z: 1.0,
};

/// Two triangles covering the whole viewport, sampling all of stage 0's texture.
///
/// Under the default identity transforms clip `[-1, 1]` is the viewport, so the
/// texture's `[0, 1]` maps onto the whole frame and a region of the texture
/// lands on the same fraction of the frame whatever either one is rasterized at.
const FULLSCREEN_TEXTURED_QUAD: [TexturedVertex; 6] = [
    TexturedVertex {
        x: -1.0,
        y: 1.0,
        z: 0.5,
        color: WHITE,
        u: 0.0,
        v: 0.0,
    },
    TexturedVertex {
        x: 1.0,
        y: 1.0,
        z: 0.5,
        color: WHITE,
        u: 1.0,
        v: 0.0,
    },
    TexturedVertex {
        x: -1.0,
        y: -1.0,
        z: 0.5,
        color: WHITE,
        u: 0.0,
        v: 1.0,
    },
    TexturedVertex {
        x: 1.0,
        y: 1.0,
        z: 0.5,
        color: WHITE,
        u: 1.0,
        v: 0.0,
    },
    TexturedVertex {
        x: 1.0,
        y: -1.0,
        z: 0.5,
        color: WHITE,
        u: 1.0,
        v: 1.0,
    },
    TexturedVertex {
        x: -1.0,
        y: -1.0,
        z: 0.5,
        color: WHITE,
        u: 0.0,
        v: 1.0,
    },
];

/// One triangle covering the whole viewport at a constant `z`, coloured `GREEN`.
///
/// `D3DFVF_XYZ` under the default identity transforms, so `[-1, 1]` is the
/// viewport: the same shape the depth-occlusion tests use to read a depth
/// buffer back through the depth test, there being no direct depth readback.
const fn covering_quad(z: f32) -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z,
            color: GREEN,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z,
            color: GREEN,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z,
            color: GREEN,
        },
    ]
}

/// Set up a depth-tested, depth-write-disabled diffuse draw under `compare`.
fn depth_probe_setup(h: &Harness, compare: u32) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, compare), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
}

#[test]
fn scissor_confines_a_clear_to_its_rect() {
    let h = Harness::new();
    // The scissor goes on *inside* the frame: `render_once` opens with its own
    // full-target clear, and enabling it earlier would clip that too, leaving
    // the outside never written rather than blue.
    h.render_once(BLUE, |h| {
        assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
        assert_eq!(
            h.set_scissor_rect(&D3DRECT {
                x1: 160,
                y1: 120,
                x2: 480,
                y2: 360,
            }),
            0,
        );
        assert_eq!(h.clear_target(RED), 0, "scissored full-target clear");
    });

    // Well inside the scissor: the second clear reached it.
    assert_pixel_eq(h.read_pixel(320, 240), RED, "scissor interior");
    // Well outside on every side: the first clear's colour survived.
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside, top-left");
    assert_pixel_eq(h.read_pixel(600, 40), BLUE, "outside, top-right");
    assert_pixel_eq(h.read_pixel(40, 440), BLUE, "outside, bottom-left");
    assert_pixel_eq(h.read_pixel(600, 440), BLUE, "outside, bottom-right");
}

#[test]
fn clear_rects_fill_only_the_rects_given() {
    let h = Harness::new();
    // Two disjoint blocks, deliberately not on a clean fraction of the frame so
    // a rounding error in either direction moves a probe off its colour.
    let rects = [
        D3DRECT {
            x1: 48,
            y1: 48,
            x2: 208,
            y2: 176,
        },
        D3DRECT {
            x1: 400,
            y1: 272,
            x2: 592,
            y2: 432,
        },
    ];

    h.render_once(BLUE, |h| {
        assert_eq!(h.clear_target_rects(RED, &rects), 0, "Clear with pRects");
    });

    assert_pixel_eq(h.read_pixel(128, 112), RED, "inside first rect");
    assert_pixel_eq(h.read_pixel(496, 352), RED, "inside second rect");
    assert_pixel_eq(h.read_pixel(300, 240), BLUE, "between the two rects");
    assert_pixel_eq(h.read_pixel(20, 20), BLUE, "outside both");
}

#[test]
fn clear_rects_intersect_the_scissor() {
    // Both the rects and the scissor are game-supplied, so both convert; the
    // intersection has to be taken in one space or the overlap comes out wrong.
    let h = Harness::new();
    // Straddles the scissor's left edge: only the right part may be cleared.
    let rects = [D3DRECT {
        x1: 32,
        y1: 160,
        x2: 320,
        y2: 320,
    }];

    h.render_once(BLUE, |h| {
        assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
        assert_eq!(
            h.set_scissor_rect(&D3DRECT {
                x1: 160,
                y1: 120,
                x2: 480,
                y2: 360,
            }),
            0,
        );
        assert_eq!(h.clear_target_rects(RED, &rects), 0);
    });

    assert_pixel_eq(h.read_pixel(260, 240), RED, "inside rect and scissor");
    assert_pixel_eq(h.read_pixel(80, 240), BLUE, "in rect, outside scissor");
    assert_pixel_eq(h.read_pixel(400, 240), BLUE, "in scissor, outside rect");
}

#[test]
fn a_scissor_covering_the_target_clears_it_between_two_draws() {
    // A scissored clear whose scissor is the whole target is a whole-target
    // clear, landing after the draw before it and before the draws after it,
    // on both planes. Every draw is a covering quad, confined by a smaller
    // scissor for the later two.
    let h = Harness::with_depth();
    depth_probe_setup(&h, D3DCMP_LESSEQUAL);
    let (width, height) = h.dims();
    let whole = D3DRECT {
        x1: 0,
        y1: 0,
        x2: width.cast_signed(),
        y2: height.cast_signed(),
    };
    let left = D3DRECT {
        x1: 64,
        y1: 160,
        x2: 256,
        y2: 320,
    };
    let right = D3DRECT {
        x1: 384,
        y1: 160,
        x2: 576,
        y2: 320,
    };

    assert_eq!(h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLUE, 1.0, 0), 0);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
        0,
        "the target holds a draw before the clear",
    );
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
    assert_eq!(h.set_scissor_rect(&whole), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, 0.25, 0),
        0,
        "scissored clear over the whole target",
    );
    // Depth 0.25 passes a quad at 0.0 and rejects one at 0.5.
    assert_eq!(h.set_scissor_rect(&left), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.0)),
        0
    );
    assert_eq!(h.set_scissor_rect(&right), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(h.read_pixel(160, 240), GREEN, "a draw after the clear");
    assert_pixel_eq(
        h.read_pixel(480, 240),
        RED,
        "the cleared depth rejects the far quad",
    );
    assert_pixel_eq(
        h.read_pixel(40, 40),
        RED,
        "the clear replaced the first draw",
    );
    assert_pixel_eq(h.read_pixel(600, 440), RED, "far corner");
}

#[test]
fn a_rect_covering_render_target_zero_leaves_a_larger_depth_surface_outside_it() {
    // D3D9 lets the depth surface exceed render target 0. A rect past every
    // edge of a 256x256 target covers that target once clipped to the
    // viewport, but only the corner of a 512x512 depth surface, whose rest
    // keeps its depth. Neither size is the back buffer's, so no scale applies.
    let h = Harness::with_depth();
    depth_probe_setup(&h, D3DCMP_LESSEQUAL);
    let large = h.create_render_target(512, 512, D3DFMT_X8R8G8B8);
    let small = h.create_render_target(256, 256, D3DFMT_X8R8G8B8);
    let depth = h.create_depth_stencil_surface(512, 512, D3DFMT_D24S8);
    assert_eq!(h.set_depth_stencil_surface(&depth), 0);
    assert_eq!(h.set_render_target(0, &large), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );

    assert_eq!(h.set_render_target(0, &small), 0);
    assert_eq!(
        h.clear_rects(
            D3DCLEAR_ZBUFFER,
            BLACK,
            0.0,
            0,
            &[D3DRECT {
                x1: -64,
                y1: -64,
                x2: 1024,
                y2: 1024,
            }],
        ),
        0,
        "a rect larger than render target 0",
    );

    assert_eq!(h.set_render_target(0, &large), 0);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(64, 64),
        BLACK,
        "inside render target 0's extent the depth is 0.0",
    );
    assert_pixel_eq(
        h.read_pixel(400, 400),
        GREEN,
        "the depth surface outside it keeps 1.0",
    );
}

#[test]
fn viewport_bounds_a_clear_in_reported_coordinates() {
    let h = Harness::new();
    // Narrowed inside the frame, after `render_once`'s own full-target clear.
    h.render_once(BLUE, |h| {
        assert_eq!(
            h.set_viewport(&D3DVIEWPORT9 {
                x: 128,
                y: 96,
                width: 384,
                height: 288,
                min_z: 0.0,
                max_z: 1.0,
            }),
            0,
        );
        assert_eq!(h.clear_target(RED), 0, "viewport-bounded clear");
    });

    assert_pixel_eq(h.read_pixel(320, 240), RED, "viewport interior");
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside the viewport");
    assert_pixel_eq(
        h.read_pixel(600, 440),
        BLUE,
        "outside the viewport, far corner",
    );
}

#[test]
fn a_combined_clear_bounds_its_colour_plane_to_the_viewport() {
    // D3D9 bounds `Clear` by the viewport whichever planes it names. A
    // `TARGET | ZBUFFER` clear is what most titles issue at the top of a
    // frame, so the colour plane has to honour the viewport there exactly as
    // it does on its own.
    let h = Harness::with_depth();
    h.render_once(BLUE, |h| {
        assert_eq!(h.set_viewport(&NARROW), 0);
        assert_eq!(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, 1.0, 0),
            0,
            "combined colour + depth clear under a narrowed viewport",
        );
    });

    assert_pixel_eq(h.read_pixel(320, 240), RED, "viewport interior");
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside the viewport");
    assert_pixel_eq(
        h.read_pixel(600, 440),
        BLUE,
        "outside the viewport, far corner",
    );
}

#[test]
fn a_viewport_bounded_depth_clear_leaves_the_depth_outside_it() {
    // Read the depth buffer back the only way this suite can: clear it to 1.0
    // everywhere, clear it to 0.0 inside the viewport alone, then draw a
    // covering quad at 0.5 under `LESSEQUAL` with depth writes off. It survives
    // exactly where depth is still 1.0.
    let h = Harness::with_depth();
    depth_probe_setup(&h, D3DCMP_LESSEQUAL);
    let full = h.viewport();

    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_viewport(&NARROW), 0);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 0.0, 0),
        0,
        "whole-target depth clear under a narrowed viewport",
    );
    // Back to the whole frame: the covering quad has to reach the pixels
    // outside the narrowed viewport for them to be worth probing.
    assert_eq!(h.set_viewport(&full), 0);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(320, 240),
        BLACK,
        "inside the viewport depth is 0.0 and rejects the quad",
    );
    assert_pixel_eq(
        h.read_pixel(40, 40),
        GREEN,
        "outside the viewport depth keeps 1.0 and accepts the quad",
    );
    assert_pixel_eq(
        h.read_pixel(600, 440),
        GREEN,
        "outside the viewport, far corner",
    );
}

#[test]
fn a_bounded_depth_clear_writes_the_raw_value_under_a_partitioned_depth_range() {
    // `Clear`'s Z is a raw depth value: `MinZ`/`MaxZ` scale a transformed
    // vertex, never a clear. A bounded depth clear is painted by a quad whose
    // depth is the vertex's clip-space z, so without care the viewport's depth
    // range remaps it. Under `MinZ = 0.5` a clear to 0.0 would land at 0.5,
    // which the 0.25 quad below then passes instead of failing.
    let h = Harness::with_depth();
    depth_probe_setup(&h, D3DCMP_LESS);
    let full = h.viewport();

    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(
        h.set_viewport(&D3DVIEWPORT9 {
            min_z: 0.5,
            ..NARROW
        }),
        0,
    );
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, BLACK, 0.0, 0), 0);
    assert_eq!(h.set_viewport(&full), 0);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.25)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(320, 240),
        BLACK,
        "the clear wrote a raw 0.0, not 0.5 remapped through MinZ",
    );
    assert_pixel_eq(
        h.read_pixel(40, 40),
        GREEN,
        "outside the viewport depth keeps 1.0 and accepts the quad",
    );
}

#[test]
fn rebinding_the_backbuffer_keeps_its_coordinate_space() {
    // D3D9 forbids a null RT0, so restoring the back buffer means binding its
    // surface. Anything that decided "is the back buffer bound" by looking for
    // a null render-target pointer would stop converting here and render at the
    // wrong size for the rest of the frame.
    let h = Harness::new();
    let rt = h.create_render_target(256, 256, D3DFMT_X8R8G8B8);
    let backbuffer = h.back_buffer(0);

    assert_eq!(h.set_render_target(0, &rt), 0, "bind a game render target");
    assert_eq!(
        h.set_render_target(0, &backbuffer),
        0,
        "restore the back buffer by surface",
    );

    // The viewport reset by that last bind must describe the *reported* size.
    let vp = h.viewport();
    assert_eq!(
        (vp.x, vp.y, vp.width, vp.height),
        (0, 0, h.dims().0, h.dims().1),
        "rebinding the back buffer resets the viewport to its reported size",
    );

    h.render_once(BLUE, |h| {
        assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
        assert_eq!(
            h.set_scissor_rect(&D3DRECT {
                x1: 160,
                y1: 120,
                x2: 480,
                y2: 360,
            }),
            0,
        );
        assert_eq!(h.clear_target(GREEN), 0);
    });

    assert_pixel_eq(h.read_pixel(320, 240), GREEN, "scissor still converts");
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside the scissor");
}

#[test]
fn a_readback_keeps_the_alpha_the_frame_carries() {
    // Under a reduced scale the back buffer is rasterized smaller than the size
    // D3D9 reports, so `GetRenderTargetData` has to resample it up on the way
    // to the caller. That resample owes the game all four channels: a title
    // reading the back buffer back gets the alpha it wrote, not an opaque one.
    let h = Harness::new();
    h.render_once(TRANSLUCENT, |_| {});

    assert_pixel_eq(
        h.read_pixel(320, 240),
        TRANSLUCENT,
        "the readback carries the alpha the frame was cleared to",
    );
    assert_pixel_eq(
        h.read_pixel(40, 40),
        TRANSLUCENT,
        "including at the edge of the frame",
    );
}

#[test]
fn a_game_render_target_is_unaffected_by_the_scale() {
    // A game-created target is always exactly the size it asked for, so its
    // coordinates must pass through untouched no matter what the back buffer
    // is doing.
    let h = Harness::new();
    let rt = h.create_render_target(256, 256, D3DFMT_X8R8G8B8);
    assert_eq!(h.set_render_target(0, &rt), 0);

    let vp = h.viewport();
    assert_eq!(
        (vp.x, vp.y, vp.width, vp.height),
        (0, 0, 256, 256),
        "SetRenderTarget resets the viewport to the RT's own size",
    );

    let sc = h.scissor_rect();
    assert_eq!(
        (sc.x1, sc.y1, sc.x2, sc.y2),
        (0, 0, 256, 256),
        "and the scissor likewise",
    );
}

#[test]
fn reset_resize_keeps_reported_coordinates() {
    let h = Harness::new();
    assert_eq!(h.reset(800, 600), 0, "resize Reset must succeed");
    assert_eq!(h.dims(), (800, 600), "harness tracks the new reported dims");

    let vp = h.viewport();
    assert_eq!(
        (vp.width, vp.height),
        (800, 600),
        "viewport follows the reported size, not the rasterized one",
    );

    h.render_once(BLUE, |h| {
        assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
        assert_eq!(
            h.set_scissor_rect(&D3DRECT {
                x1: 200,
                y1: 150,
                x2: 600,
                y2: 450,
            }),
            0,
        );
        assert_eq!(h.clear_target(RED), 0);
    });

    // (700, 500) only exists in the grown back buffer.
    assert_pixel_eq(
        h.read_pixel(400, 300),
        RED,
        "inside the post-resize scissor",
    );
    assert_pixel_eq(h.read_pixel(700, 500), BLUE, "grown area, outside scissor");
}

/// A point's diameter is a length D3D9 states in the reported space.
///
/// Every other primitive gets its extent from vertex positions, which the
/// projection already carries into whatever space the frame rasterizes in. A
/// point takes a diameter in pixels instead, and `[[point_size]]` is measured
/// in the pixels actually rasterized, so the size has to convert on the way
/// down or a point keeps its reported diameter in render pixels and comes back
/// `1 / scale` too wide.
///
/// Pins its own scale (a clean half, so the conversion is exact) rather than
/// inheriting the run's: at the identity the conversion is unobservable, and
/// this must fail in the ordinary `make test` if it regresses.
#[test]
fn a_point_keeps_its_reported_diameter_under_the_scale() {
    // The scale under test, resolved by this harness alone. The parser keeps
    // the last entry, so this wins over a `make test SCALE=<n>` run too.
    let h = Harness::with_config("render.scale=0.5");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    // 64 is `POINTSIZE_MAX`'s default, so nothing clamps it: the square spans
    // 32 reported pixels either side of the centre, and every probe below sits
    // 8 pixels clear of that edge.
    assert_eq!(
        h.set_render_state(D3DRS_POINTSIZE, 64.0_f32.to_bits()),
        0,
        "SetRenderState(POINTSIZE)"
    );
    let point = [PosColorVertex {
        x: 0.0,
        y: 0.0,
        z: 0.5,
        color: GREEN,
    }];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_POINTLIST, 1, &point),
            0,
            "POINTLIST draws"
        );
    });

    assert_pixel_eq(h.read_pixel(320, 240), GREEN, "the point's centre");
    assert_pixel_eq(h.read_pixel(344, 264), GREEN, "24 px inside the square");
    assert_pixel_eq(h.read_pixel(296, 216), GREEN, "the opposite corner");
    // 40 px out is background for a 64 px square and inside a 128 px one, so
    // an unconverted size fails here.
    assert_pixel_eq(h.read_pixel(360, 240), BLACK, "40 px right of the square");
    assert_pixel_eq(h.read_pixel(320, 200), BLACK, "40 px above the square");
}

/// A reported back-buffer size whose 75% edges carry a fraction below one half.
///
/// 803 and 603 scale to 602.25 and 452.25, where a texture extent rounded one
/// way and a viewport edge rounded the other disagree by a texel, leaving the
/// last column and row of the rasterized frame outside a full-target viewport.
const ODD_FRAME: (u32, u32) = (803, 603);

/// A device at [`ODD_FRAME`] rasterizing at 75%, whatever the run's own scale.
///
/// Its viewport is the whole target, set through `SetViewport`. Pins the scale
/// rather than inheriting it: at the identity the two roundings cannot
/// disagree, and this must fail in the ordinary `make test` if it regresses.
/// The parser keeps the last entry, so this wins over a `make test SCALE=<n>`
/// run too.
fn odd_frame_at_three_quarters() -> Harness {
    let h = Harness::create(&HarnessConfig {
        width: ODD_FRAME.0,
        height: ODD_FRAME.1,
        config_entries: "render.scale=0.75",
        ..HarnessConfig::default()
    });
    assert_eq!(h.dims(), ODD_FRAME, "the device reports the requested size");
    // Set explicitly, as a game does and as `SetRenderTarget` does for it: a
    // device that never saw `SetViewport` reads the bound texture's own extent
    // instead, which is not the conversion under test.
    let full = D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: ODD_FRAME.0,
        height: ODD_FRAME.1,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(h.set_viewport(&full), 0, "SetViewport(full target)");
    h
}

/// Probe the last reported column and row, and the middle, for `expected`.
fn assert_edges_read(h: &Harness, expected: u32, what: &str) {
    let (width, height) = ODD_FRAME;
    let (last_x, last_y) = (width - 1, height - 1);
    assert_pixel_eq(h.read_pixel(width / 2, height / 2), expected, what);
    assert_pixel_eq(
        h.read_pixel(last_x, height / 2),
        expected,
        &format!("{what}: last column"),
    );
    assert_pixel_eq(
        h.read_pixel(width / 2, last_y),
        expected,
        &format!("{what}: last row"),
    );
    assert_pixel_eq(
        h.read_pixel(last_x, last_y),
        expected,
        &format!("{what}: last corner"),
    );
}

/// A draw under a full-target viewport covers the target's last column and row.
///
/// The readback resolves the rasterized frame up to the reported size, so a
/// render column the draw never reached comes back as the last reported one.
#[test]
fn a_full_target_draw_reaches_the_last_column_and_row_of_an_odd_sized_frame() {
    let h = odd_frame_at_three_quarters();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    h.render_once(RED, |h| {
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
            0,
            "covering draw",
        );
    });
    assert_edges_read(&h, GREEN, "full-viewport draw");
}

/// A whole-target `Clear` under a full-target viewport reaches the last column and row.
#[test]
fn a_whole_target_clear_reaches_the_last_column_and_row_of_an_odd_sized_frame() {
    let h = odd_frame_at_three_quarters();
    h.render_once(BLUE, |_| {});
    assert_edges_read(&h, BLUE, "whole-target clear");
}

/// A draw into mip level 1 of a scaled target reaches that level's last column and row.
///
/// Metal sizes the level from the scaled base: 67% of 853x659 is 572x442, so
/// level 1 is 286x221 texels, a texel wider and taller than the 285x220 the
/// scale makes of the level's reported 426x329. Bound with a full-level
/// viewport and scissor, a pass measured against the scaled reported extent
/// leaves the level's last column and row undrawn. Pins its own scale,
/// because at the suite's 0.75 these sizes do not mismatch.
#[test]
fn a_draw_into_a_scaled_mip_level_reaches_its_last_column_and_row() {
    let config = "render.scale=0.67";
    let (width, height) = (853, 659);
    let h = Harness::create(&HarnessConfig {
        width,
        height,
        config_entries: config,
        ..HarnessConfig::default()
    });
    let rt = h.create_texture(
        width,
        height,
        2,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let level = rt.surface_level(1);
    let (level_w, level_h) = (width / 2, height / 2);
    assert_eq!(h.set_render_target(0, &level), 0, "{config}: bind level 1");
    let full = D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: level_w,
        height: level_h,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(h.set_viewport(&full), 0, "{config}: full-level viewport");
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
    assert_eq!(
        h.set_scissor_rect(&D3DRECT {
            x1: 0,
            y1: 0,
            x2: level_w.cast_signed(),
            y2: level_h.cast_signed(),
        }),
        0,
        "{config}: full-level scissor",
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    h.render_once(RED, |h| {
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &covering_quad(0.5)),
            0,
            "{config}: covering draw",
        );
    });
    let (last_x, last_y) = (level_w - 1, level_h - 1);
    for (x, y, what) in [
        (level_w / 2, level_h / 2, "middle"),
        (last_x, level_h / 2, "last column"),
        (level_w / 2, last_y, "last row"),
        (last_x, last_y, "last corner"),
    ] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("{config}: level 1 {what}"),
        );
    }
}

#[test]
fn color_fill_of_a_target_at_the_backbuffer_size_uses_reported_coordinates() {
    // A render target created at the reported back-buffer size belongs to the
    // same image and rasterizes at the same scale, so a `ColorFill` sub-rect
    // has to be converted the way a viewport or a scissor is. Both the fill
    // rect and the probes are reported coordinates, so the result is the same
    // at every scale.
    let h = Harness::new();
    let (width, height) = h.dims();
    let rt = h.create_texture(
        width,
        height,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let surface = rt.surface_level(0);
    assert_eq!(h.set_render_target(0, &surface), 0, "bind the target");
    assert_eq!(h.clear_target(BLUE), 0, "seed the whole target");
    assert_eq!(
        h.color_fill_rect_hr(&surface, (128, 96, 512, 384), RED),
        0,
        "ColorFill a sub-rect of the target",
    );

    assert_pixel_eq(h.read_pixel(320, 240), RED, "the middle of the fill rect");
    assert_pixel_eq(h.read_pixel(180, 140), RED, "inside, near the top-left");
    assert_pixel_eq(h.read_pixel(460, 340), RED, "inside, near the bottom-right");
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside, above and left");
    assert_pixel_eq(h.read_pixel(600, 440), BLUE, "outside, below and right");
}

#[test]
fn color_fill_of_a_scaled_targets_mip_level_addresses_that_level() {
    // A mip level of such a target is rasterized at the scale as well, so its
    // extent is not the halved reported extent its descriptor carries. The
    // fill rect is given in the level's reported coordinates and read back by
    // sampling the level over the whole frame, where a fill that covered the
    // wrong fraction of the level lands at the wrong fraction of the frame.
    let h = Harness::new();
    let (width, height) = h.dims();
    let rt = h.create_texture(
        width,
        height,
        2,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let level1 = rt.surface_level(1);
    let (hr, desc) = level1.desc();
    assert_eq!(hr, 0, "GetDesc on level 1");
    assert_eq!(
        (desc.width, desc.height),
        (width / 2, height / 2),
        "level 1 reports half the reported size whatever the scale",
    );

    assert_eq!(h.color_fill_hr(&level1, BLUE), 0, "seed the whole level");
    assert_eq!(
        h.color_fill_rect_hr(&level1, (0, 0, 160, 120), RED),
        0,
        "fill the level's top-left quarter",
    );

    // Sample level 1 across the whole frame: `MAXMIPLEVEL` clamps the level of
    // detail up so the magnified draw reads level 1 rather than level 0.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_texture(0, &rt), 0, "bind the target as a texture");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAXMIPLEVEL, 1),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler state");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |h| {
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &FULLSCREEN_TEXTURED_QUAD),
            0,
            "sample level 1 over the whole frame",
        );
    });

    // The filled quarter is the top-left quarter of the level, so it covers the
    // top-left quarter of the frame: (320, 240) is the corner it stops at.
    assert_pixel_eq(h.read_pixel(160, 120), RED, "inside the filled quarter");
    assert_pixel_eq(
        h.read_pixel(370, 280),
        BLUE,
        "just past the filled quarter on both axes",
    );
    assert_pixel_eq(h.read_pixel(560, 400), BLUE, "the opposite corner");
}

/// An A8R8G8B8 render-target texture of `levels` levels at the reported back-buffer size.
fn backbuffer_sized_target(h: &Harness, levels: u32) -> Texture<'_> {
    let (width, height) = h.dims();
    h.create_texture(
        width,
        height,
        levels,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    )
}

/// A whole-level `UpdateSurface` into a render-target texture at the back-buffer size reaches it.
///
/// Such a texture is rasterized at the scale, so the rows the update carries
/// describe a larger extent than the level's Metal texture has, and the write
/// resamples on its way in.
#[test]
fn a_write_into_a_back_buffer_sized_render_target_texture_reaches_it() {
    let h = Harness::new();
    let (width, height) = h.dims();
    let target = backbuffer_sized_target(&h, 1);
    let level = target.surface_level(0);
    let source =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u32(&vec![GREEN; (width * height) as usize]);
    let whole = D3DRECT {
        x1: 0,
        y1: 0,
        x2: width.cast_signed(),
        y2: height.cast_signed(),
    };
    assert_eq!(
        h.update_surface_region_hr(&source, &whole, &level, (0, 0)),
        D3D_OK,
        "UpdateSurface of the whole level"
    );
    let pixels = read_back(&h, &level, (width, height), D3DFMT_A8R8G8B8);
    for (x, y) in [(width / 2, height / 2), (8, 8), (width - 8, height - 8)] {
        assert_eq!(
            pixels[(y * width + x) as usize],
            GREEN,
            "texel ({x}, {y}) holds the write"
        );
    }
}

/// A `LockRect` of a sub-rect of such a texture writes that rect and leaves the rest.
///
/// A whole-level lock seeds the level blue first, so the resample of the
/// smaller red rect after it reads rows staged where the blue ones were. The
/// rect's edges are multiples of 100, which every scale converts exactly, and
/// the probes in its second and second-to-last rows and columns read back
/// through its outermost rasterized texels and nothing beyond them, at 0.5,
/// 0.67 and 0.75 as at the identity, so they hold red only if the resample
/// took nothing from beyond the rect's own rows.
#[test]
fn a_partial_lock_of_a_back_buffer_sized_render_target_texture_writes_only_its_rect() {
    let h = Harness::new();
    let (width, height) = h.dims();
    let target = backbuffer_sized_target(&h, 1);
    let level = target.surface_level(0);
    {
        let mut locked = target.lock_rect(0, 0);
        locked.write_u32_rect(
            width as usize,
            height as usize,
            &vec![BLUE; (width * height) as usize],
        );
    }
    let (x1, y1, x2, y2) = (100u32, 100u32, 500u32, 400u32);
    {
        let rect = [
            x1.cast_signed(),
            y1.cast_signed(),
            x2.cast_signed(),
            y2.cast_signed(),
        ];
        let mut locked = target.lock_rect_partial(0, &rect, 0);
        let (w, ht) = ((x2 - x1) as usize, (y2 - y1) as usize);
        locked.write_u32_rect(w, ht, &vec![RED; w * ht]);
    }
    let pixels = read_back(&h, &level, (width, height), D3DFMT_A8R8G8B8);
    for (x, y, expected, what) in [
        (320, 240, RED, "the middle of the locked rect"),
        (x1 + 1, 240, RED, "the rect's second column"),
        (320, y1 + 1, RED, "the rect's second row"),
        (x2 - 2, 240, RED, "the rect's second-to-last column"),
        (320, y2 - 2, RED, "the rect's second-to-last row"),
        (40, 40, BLUE, "outside, above and left"),
        (600, 440, BLUE, "outside, below and right"),
    ] {
        assert_eq!(
            pixels[(y * width + x) as usize],
            expected,
            "texel ({x}, {y}): {what}"
        );
    }
}

/// `UpdateTexture` into such a texture writes its mip level 1 as well as level 0.
///
/// Level 1 is rasterized at the scale too, at Metal's halving of the scaled
/// base. The source's level 1 is red in its top-left quarter and blue
/// elsewhere, read back at the level's reported extent.
#[test]
fn update_texture_into_a_back_buffer_sized_render_target_texture_writes_mip_level_1() {
    let h = Harness::new();
    let (width, height) = h.dims();
    let target = backbuffer_sized_target(&h, 2);
    let source = h.create_texture(width, height, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let (w1, h1) = (width / 2, height / 2);
    {
        let mut locked = source.lock_rect(0, 0);
        locked.write_u32_rect(
            width as usize,
            height as usize,
            &vec![GREEN; (width * height) as usize],
        );
    }
    {
        let texels: Vec<u32> = (0..h1)
            .flat_map(|y| (0..w1).map(move |x| if x < w1 / 2 && y < h1 / 2 { RED } else { BLUE }))
            .collect();
        let mut locked = source.lock_rect(1, 0);
        locked.write_u32_rect(w1 as usize, h1 as usize, &texels);
    }
    assert_eq!(
        h.update_texture_hr(&source, &target),
        D3D_OK,
        "UpdateTexture of both levels"
    );
    let level0 = read_back(
        &h,
        &target.surface_level(0),
        (width, height),
        D3DFMT_A8R8G8B8,
    );
    assert_eq!(
        level0[(height / 2 * width + width / 2) as usize],
        GREEN,
        "level 0 holds its write"
    );
    let level1 = read_back(&h, &target.surface_level(1), (w1, h1), D3DFMT_A8R8G8B8);
    for (x, y, expected, what) in [
        (w1 / 4, h1 / 4, RED, "inside the red quarter"),
        (w1 * 3 / 4, h1 / 4, BLUE, "right of the red quarter"),
        (w1 / 4, h1 * 3 / 4, BLUE, "below the red quarter"),
        (w1 * 3 / 4, h1 * 3 / 4, BLUE, "the opposite quarter"),
    ] {
        assert_eq!(
            level1[(y * w1 + x) as usize],
            expected,
            "level 1 texel ({x}, {y}): {what}"
        );
    }
}

/// What GDI draws into such a texture through `GetDC` reaches it at `ReleaseDC`.
///
/// The level is seeded blue by a `ColorFill`; GDI paints a red block at its
/// top-left corner. GDI writes a zero alpha byte into the A8R8G8B8 texels it
/// paints, so the colour channels are what the probes compare.
#[test]
fn get_dc_drawing_into_a_back_buffer_sized_render_target_texture_reaches_it() {
    const BLOCK: u32 = 64;
    let h = Harness::new();
    let (width, height) = h.dims();
    let target = backbuffer_sized_target(&h, 1);
    let level = target.surface_level(0);
    assert_eq!(
        h.color_fill_hr(&level, BLUE),
        D3D_OK,
        "seed the whole level"
    );
    let dc = level.dc();
    dc.fill_block(BLOCK.cast_signed(), RED_COLORREF);
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");
    let pixels = read_back(&h, &level, (width, height), D3DFMT_A8R8G8B8);
    for (x, y, expected, what) in [
        (BLOCK / 2, BLOCK / 2, RED, "inside GDI's block"),
        (BLOCK * 2, BLOCK / 2, BLUE, "right of the block"),
        (width / 2, height / 2, BLUE, "far from the block"),
    ] {
        assert_eq!(
            pixels[(y * width + x) as usize] & 0x00FF_FFFF,
            expected & 0x00FF_FFFF,
            "texel ({x}, {y}): {what}"
        );
    }
}

/// A scaled render-target texture written on one device keeps the write on the next to sample it.
///
/// Both devices pin `render.scale=0.75`. The first writes the texture green
/// through `UpdateSurface`; the second binds it, which moves the texture to
/// that device and uploads every level written so far there, and samples it
/// across its frame.
#[test]
fn a_scaled_target_written_on_one_device_reaches_the_device_that_samples_it() {
    let config = HarnessConfig {
        config_entries: "render.scale=0.75",
        ..HarnessConfig::default()
    };
    let first = Harness::create(&config);
    let second = Harness::create(&config);
    let (width, height) = first.dims();
    let target = backbuffer_sized_target(&first, 1);
    let source =
        first.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u32(&vec![GREEN; (width * height) as usize]);
    assert_eq!(
        first.update_surface_hr(&source, &target.surface_level(0)),
        D3D_OK,
        "UpdateSurface on the first device"
    );
    assert_eq!(
        second.set_render_state(D3DRS_LIGHTING, 0),
        0,
        "lighting off"
    );
    assert_eq!(
        second.set_texture(0, &target),
        0,
        "bind on the second device"
    );
    second.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(
            second.set_sampler_state(0, state, value),
            0,
            "sampler state"
        );
    }
    assert_eq!(second.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    second.render_once(BLACK, |h| {
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &FULLSCREEN_TEXTURED_QUAD),
            0,
            "sample the texture over the whole frame",
        );
    });
    assert_pixel_eq(
        second.read_pixel(width / 2, height / 2),
        GREEN,
        "the second device samples the first device's write",
    );
    assert_eq!(second.clear_texture(0), 0, "unbind");
}

/// A full-chain render-target texture at the back-buffer size draws into and reads its levels.
///
/// The texture reports the chain of the reported size, ten levels for
/// 640x480, while the texture that rasterizes at the scale may hold one level
/// fewer (480x360 holds nine). Level 0 is cleared red through a bind and the
/// last level, one texel, filled green; each reads back what it was given.
#[test]
fn a_full_chain_target_at_the_backbuffer_size_draws_and_reads_its_levels() {
    let h = Harness::new();
    let (width, height) = h.dims();
    let back = h.render_target(0);
    let target = h.create_texture(
        width,
        height,
        0,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let levels = target.level_count();
    let full_chain = 32 - width.max(height).leading_zeros();
    assert_eq!(levels, full_chain, "the reported size's whole chain");
    let base = target.surface_level(0);
    assert_eq!(h.set_render_target(0, &base), D3D_OK, "bind level 0");
    assert_eq!(h.clear_target(RED), D3D_OK, "clear level 0 red");
    assert_eq!(
        h.set_render_target(0, &back),
        D3D_OK,
        "restore the back buffer"
    );
    let last = target.surface_level(levels - 1);
    let (hr, desc) = last.desc();
    assert_eq!(hr, D3D_OK, "GetDesc on the last level");
    assert_eq!(h.color_fill_hr(&last, GREEN), D3D_OK, "fill the last level");
    let pixels = read_back(&h, &base, (width, height), D3DFMT_A8R8G8B8);
    assert_eq!(
        pixels[(height / 2 * width + width / 2) as usize],
        RED,
        "level 0 holds the clear"
    );
    let pixels = read_back(&h, &last, (desc.width, desc.height), D3DFMT_A8R8G8B8);
    assert_eq!(pixels[0], GREEN, "the last level holds the fill");
}

/// An autogen render-target texture at the back-buffer size regenerates its chain from a write.
///
/// The chain is seeded red by a draw into level 0, an `UpdateSurface` writes
/// level 0 green, and a sample pinned to level 4 reads green: the chain follows
/// the resampled write rather than the level 0 from before it. Pins
/// `render.scale=0.75` so the scaled write runs on every leg; at the default
/// 640x480 the chain it reports is a level longer than the 480x360 texture
/// that rasterizes can hold.
#[test]
fn an_autogen_back_buffer_sized_render_target_texture_regenerates_from_a_write() {
    let h = Harness::create(&HarnessConfig {
        config_entries: "render.scale=0.75",
        ..HarnessConfig::default()
    });
    let (width, height) = h.dims();
    let back = h.render_target(0);
    let target = h.create_texture(
        width,
        height,
        1,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let level = target.surface_level(0);
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.set_render_target(0, &level), D3D_OK, "bind the target");
    assert_eq!(h.begin_scene(), D3D_OK, "BeginScene");
    draw_fill(&h, RED);
    assert_eq!(h.end_scene(), D3D_OK, "EndScene");
    assert_eq!(
        h.set_render_target(0, &back),
        D3D_OK,
        "restore the back buffer"
    );
    let source =
        h.create_offscreen_plain_surface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u32(&vec![GREEN; (width * height) as usize]);
    assert_eq!(
        h.update_surface_hr(&source, &level),
        D3D_OK,
        "UpdateSurface of the whole level"
    );
    sample_mip_level_4(&h, &target);
    assert_eq!(h.present(), D3D_OK, "Present");
    assert_pixel_eq(
        h.read_pixel(width / 2, height / 2),
        GREEN,
        "the small level carries the write",
    );
}

#[test]
fn a_standalone_target_at_the_backbuffer_size_fills_and_copies_in_reported_coordinates() {
    // A `CreateRenderTarget` surface created at the reported back-buffer size
    // belongs to the same image and is rasterized at the same scale, so both
    // operations on it convert: the `ColorFill` sub-rect against the surface's
    // own texture, and the whole-surface `StretchRect` that copies it to the
    // back buffer. The surface carries the scale it was created at, so neither
    // endpoint depends on what the back buffer measures at the time. Fill rect
    // and probes are reported coordinates, so the result is the same at every
    // scale.
    let h = Harness::new();
    let (width, height) = h.dims();
    let rt = h.create_render_target(width, height, D3DFMT_X8R8G8B8);
    assert_eq!(h.color_fill_hr(&rt, BLUE), 0, "seed the whole target");
    assert_eq!(
        h.color_fill_rect_hr(&rt, (128, 96, 512, 384), RED),
        0,
        "ColorFill a sub-rect of the target",
    );

    let backbuffer = h.render_target(0);
    assert_eq!(
        h.stretch_rect(&rt, &backbuffer, D3DTEXF_NONE),
        0,
        "copy the whole target onto the back buffer",
    );

    assert_pixel_eq(h.read_pixel(320, 240), RED, "the middle of the fill rect");
    assert_pixel_eq(h.read_pixel(180, 140), RED, "inside, near the top-left");
    assert_pixel_eq(h.read_pixel(460, 340), RED, "inside, near the bottom-right");
    assert_pixel_eq(h.read_pixel(40, 40), BLUE, "outside, above and left");
    assert_pixel_eq(h.read_pixel(600, 440), BLUE, "outside, below and right");
}

/// The `ps_3_0` quadrant shader, which colours a fragment by its `vPos`.
///
/// `dcl vPos.xy; sub r0.xy, vPos.xy, c0.zw; mov r1, c0; mov r2.a, c0.y;
/// mov r2.r, c0.y; cmp r2.g, r0.x, r1.x, r1.y; cmp r2.b, r0.y, r1.x, r1.y;
/// mov oC0, r2.` Green once `vPos.x` has reached the centre in `c0.zw`, blue
/// once `vPos.y` has, cyan past both and black short of either, with alpha
/// zero everywhere. Every probe on it sits well inside a quadrant, so the
/// colour is the same at every scale as long as the device applied its own.
const QUADRANT_PS: [u32; 31] = [
    0xFFFF_0300,
    0x0200_001F,
    0x8000_0000,
    0x9003_1000,
    0x0300_0002,
    0x8003_0000,
    0x9054_1000,
    0xA1FE_0000,
    0x0200_0001,
    0x800F_0001,
    0xA0E4_0000,
    0x0200_0001,
    0x8008_0002,
    0xA055_0000,
    0x0200_0001,
    0x8001_0002,
    0xA055_0000,
    0x0400_0058,
    0x8002_0002,
    0x8000_0000,
    0x8000_0001,
    0x8055_0001,
    0x0400_0058,
    0x8004_0002,
    0x8055_0000,
    0x8000_0001,
    0x8055_0001,
    0x0200_0001,
    0x800F_0800,
    0x80E4_0002,
    0x0000_FFFF,
];

/// The quadrant shader's `c0`: `(1, 0, centre x, centre y)`.
///
/// `.x` and `.y` are the true and false colours each comparison selects,
/// `.zw` the centre of the 640x480 the harness reports.
const QUADRANT_CENTRE: [f32; 4] = [1.0, 0.0, 320.0, 240.0];

#[test]
fn a_ps_3_0_vpos_reads_reported_pixel_coordinates_under_the_scale() {
    // `vPos` is the pixel a fragment lands on, and a shader compares it
    // against the back-buffer size it was told, or divides by that size for a
    // screen-space UV. Under `render.scale` the fragment lands on a render
    // pixel, so the register is scaled into the reported space and floored,
    // which keeps its integer contract (`frc(vPos)` stays zero). The quadrant
    // shader colours by `vPos` against the reported centre (320, 240): green
    // once `vPos.x` reaches it, blue once `vPos.y` does. The fraction shader
    // returns `frc(vPos)`. Every probe is well inside a quadrant, so the
    // result is the same at every scale.
    //
    // Pins its own scale (a clean half) rather than inheriting the run's: at
    // the identity the register needs no conversion, and this has to fail in
    // the ordinary `make test` if it regresses.
    //
    // ps_3_0: def c0, 0, 0, 0, 0; dcl vPos.xy; mov r0, c0; frc r0.xy, vPos.xy;
    // mov oC0, r0.
    const FRACTION_PS: [u32; 20] = [
        0xFFFF_0300,
        0x0500_0051,
        0xA00F_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0200_001F,
        0x8000_0000,
        0x9003_1000,
        0x0200_0001,
        0x800F_0000,
        0xA0E4_0000,
        0x0200_0013,
        0x8003_0000,
        0x9054_1000,
        0x0200_0001,
        0x800F_0800,
        0x80E4_0000,
        0x0000_FFFF,
    ];
    let h = Harness::with_config("render.scale=0.5");
    let (width, height) = h.dims();
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0, "SetFVF");
    let v = |x: f32, y: f32| RhwVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        color: WHITE,
    };
    // The reported dims fit u16; convert without a precision-loss cast.
    let to_f = |v: u32| f32::from(u16::try_from(v).expect("reported dims fit u16"));
    let (w, hgt) = (to_f(width), to_f(height));
    let quad = [
        v(0.0, 0.0),
        v(w, 0.0),
        v(0.0, hgt),
        v(w, 0.0),
        v(w, hgt),
        v(0.0, hgt),
    ];
    let rgb = |x: u32, y: u32| h.read_pixel(x, y) & 0x00FF_FFFF;

    let quadrant = h.create_pixel_shader(&QUADRANT_PS);
    assert_eq!(h.set_pixel_shader(&quadrant), 0, "SetPixelShader");
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &QUADRANT_CENTRE),
        0,
        "the reported centre in c0.zw"
    );
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_eq!(
        rgb(160, 120),
        0x0000_0000,
        "top-left: vPos short of the centre on both axes"
    );
    assert_eq!(
        rgb(480, 120),
        0x0000_FF00,
        "top-right: vPos.x past the centre"
    );
    assert_eq!(
        rgb(160, 360),
        0x0000_00FF,
        "bottom-left: vPos.y past the centre"
    );
    assert_eq!(
        rgb(480, 360),
        0x0000_FFFF,
        "bottom-right: both past the centre"
    );

    let fraction = h.create_pixel_shader(&FRACTION_PS);
    assert_eq!(h.set_pixel_shader(&fraction), 0, "SetPixelShader");
    h.render_once(WHITE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_eq!(
        rgb(160, 120),
        0,
        "frc(vPos) is zero: the register stays an integer"
    );
    assert_eq!(rgb(480, 360), 0, "and so in the far quadrant");
    assert_eq!(h.clear_pixel_shader(), 0, "SetPixelShader(null)");
}

#[test]
fn a_scaled_target_reads_back_at_the_reported_extent_in_every_colour_format() {
    // `GetRenderTargetData` owes the game the target at the extent D3D9
    // reports. A `CreateRenderTarget` surface at the back-buffer size is
    // rasterized at `render.scale`, so the readback resolves it up first, and
    // that resolve has to work in the target's own format: a game reads its
    // HDR pass back in a float format, its UI in an 8-bit one. Each format is
    // cleared to opaque red through the target and read back through a
    // SYSTEMMEM plain of the same format at the first, the middle and the
    // last pixel; the last one is outside the scaled texture altogether.
    //
    // Pins its own scale (a clean half, so both extents are exact) rather
    // than inheriting the run's: at the identity there is no resolve, and
    // this has to fail in the ordinary `make test` if it regresses.
    const RED: u32 = 0xFFFF_0000;
    // Each format's own encoding of opaque red, as the u32 words of one pixel.
    const FORMATS: [(u32, &str, &[u32]); 4] = [
        (D3DFMT_X8R8G8B8, "X8R8G8B8", &[0x00FF_0000]),
        (D3DFMT_A8B8G8R8, "A8B8G8R8", &[0xFF00_00FF]),
        (
            D3DFMT_A16B16G16R16F,
            "A16B16G16R16F",
            &[0x0000_3C00, 0x3C00_0000],
        ),
        (
            D3DFMT_A32B32G32R32F,
            "A32B32G32R32F",
            &[0x3F80_0000, 0, 0, 0x3F80_0000],
        ),
    ];
    let h = Harness::with_config("render.scale=0.5");
    let (width, height) = h.dims();
    let backbuffer = h.render_target(0);
    for (format, name, red) in FORMATS {
        let rt = h.create_render_target(width, height, format);
        assert_eq!(h.set_render_target(0, &rt), 0, "{name}: bind the target");
        assert_eq!(h.clear_target(RED), 0, "{name}: clear it red");
        assert_eq!(
            h.set_render_target(0, &backbuffer),
            0,
            "{name}: restore the back buffer"
        );

        let dst = h.create_offscreen_plain_surface(width, height, format, D3DPOOL_SYSTEMMEM);
        assert_eq!(
            h.get_render_target_data_hr(&rt, &dst),
            D3D_OK,
            "{name}: GetRenderTargetData at the reported extent"
        );
        let locked = dst.lock_rect(D3DLOCK_READONLY);
        let words_per_pixel = red.len();
        let pitch_words = locked.pitch().cast_unsigned() as usize / 4;
        let words = locked.as_u32(pitch_words * height as usize);
        let pixel =
            |x: usize, y: usize| &words[y * pitch_words + x * words_per_pixel..][..words_per_pixel];
        for (x, y, where_) in [
            (0, 0, "the first pixel"),
            (width as usize / 2, height as usize / 2, "the middle"),
            (width as usize - 1, height as usize - 1, "the last pixel"),
        ] {
            let got = pixel(x, y);
            let got = match format {
                // The X channel of X8R8G8B8 carries no defined value.
                D3DFMT_X8R8G8B8 => vec![got[0] & 0x00FF_FFFF],
                _ => got.to_vec(),
            };
            assert_eq!(got, red, "{name}: {where_} reads the clear colour");
        }
    }
}

/// A depth texture at the back-buffer size is charged the chain it occupies.
///
/// Such a texture is the depth buffer of the main view, so its Metal levels are
/// created at `render.scale` of the reported size while `GetLevelDesc` keeps
/// answering with the size the application asked for. The texture-memory budget
/// follows the memory, not the descriptor. A texture of any other size is an
/// intermediate the game picked a resolution for and costs what it asked for at
/// every scale.
///
/// Pins its own scale (a clean half, so both extents are exact) rather than
/// inheriting the run's: at the identity there is nothing to convert, and this
/// has to fail in the ordinary `make test` if it regresses.
#[test]
fn a_depth_texture_at_the_backbuffer_size_is_charged_its_scaled_chain() {
    // A size that is not the back buffer's, for the texture that keeps its own.
    const OWN_SIZE: u32 = 256;

    // The scale under test, resolved by this harness alone. The parser keeps
    // the last entry, so this wins over a `make test SCALE=<n>` run too.
    let h = Harness::with_config("render.scale=0.5");
    let (width, height) = h.dims();
    // One level of a four-byte depth format, at half of each reported edge.
    let scaled_bytes = (width / 2) * 4 * (height / 2);
    let base = h.available_texture_mem();
    assert!(base > 4 * scaled_bytes, "budget {base} leaves room");

    let cost = {
        let depth = h.create_texture(
            width,
            height,
            1,
            D3DUSAGE_DEPTHSTENCIL,
            D3DFMT_D24S8,
            D3DPOOL_DEFAULT,
        );
        let (hr, desc) = depth.surface_level(0).desc();
        assert_eq!(hr, 0, "GetDesc on the depth level");
        assert_eq!(
            (desc.width, desc.height),
            (width, height),
            "the level reports the requested size whatever the scale",
        );
        base - h.available_texture_mem()
    };
    assert_eq!(
        cost, scaled_bytes,
        "a back-buffer-sized depth texture costs the chain its Metal levels hold"
    );
    assert_eq!(
        h.available_texture_mem(),
        base,
        "releasing it gives those bytes back"
    );

    // A size of its own is not the main view, so such a texture keeps its
    // resolution and is charged all of it.
    let own_cost = {
        let _shadow = h.create_texture(
            OWN_SIZE,
            OWN_SIZE,
            1,
            D3DUSAGE_DEPTHSTENCIL,
            D3DFMT_D24S8,
            D3DPOOL_DEFAULT,
        );
        base - h.available_texture_mem()
    };
    assert_eq!(
        own_cost,
        OWN_SIZE * OWN_SIZE * 4,
        "a shadow map costs the size it asked for"
    );
}

/// A standalone surface at the back-buffer size is charged what it occupies.
///
/// `CreateRenderTarget` and `CreateDepthStencilSurface` at the reported
/// back-buffer size hand out the main view's own attachments, so their Metal
/// textures are created at `render.scale` while `GetDesc` keeps answering with
/// the size the application asked for. The texture-memory budget follows the
/// memory, not the descriptor, and the refund on release follows the charge. A
/// surface of any other size is an intermediate the game picked a resolution
/// for and costs what it asked for at every scale.
///
/// Pins its own scale (a clean half, so both extents are exact) rather than
/// inheriting the run's: at the identity there is nothing to convert, and this
/// has to fail in the ordinary `make test` if it regresses.
#[test]
fn a_standalone_surface_at_the_backbuffer_size_is_charged_its_scaled_extent() {
    // A size that is not the back buffer's, for the surface that keeps its own.
    const OWN_SIZE: u32 = 256;

    // The scale under test, resolved by this harness alone. The parser keeps
    // the last entry, so this wins over a `make test SCALE=<n>` run too.
    let h = Harness::with_config("render.scale=0.5");
    let (width, height) = h.dims();
    // Both formats are four bytes a texel, at half of each reported edge.
    let scaled_bytes = (width / 2) * 4 * (height / 2);
    let base = h.available_texture_mem();
    assert!(base > 8 * scaled_bytes, "budget {base} leaves room");

    let color_cost = {
        let rt = h.create_render_target(width, height, D3DFMT_X8R8G8B8);
        let (hr, desc) = rt.desc();
        assert_eq!(hr, 0, "GetDesc on the render target");
        assert_eq!(
            (desc.width, desc.height),
            (width, height),
            "the surface reports the requested size whatever the scale",
        );
        base - h.available_texture_mem()
    };
    assert_eq!(
        color_cost, scaled_bytes,
        "a back-buffer-sized render target costs the texture it holds"
    );
    assert_eq!(
        h.available_texture_mem(),
        base,
        "releasing it gives those bytes back"
    );

    let depth_cost = {
        let ds = h.create_depth_stencil_surface(width, height, D3DFMT_D24S8);
        let (hr, desc) = ds.desc();
        assert_eq!(hr, 0, "GetDesc on the depth-stencil surface");
        assert_eq!(
            (desc.width, desc.height),
            (width, height),
            "the surface reports the requested size whatever the scale",
        );
        base - h.available_texture_mem()
    };
    assert_eq!(
        depth_cost, scaled_bytes,
        "a back-buffer-sized depth-stencil surface costs the texture it holds"
    );
    assert_eq!(
        h.available_texture_mem(),
        base,
        "releasing it gives those bytes back"
    );

    // A size of its own is not the main view, so such a surface keeps its
    // resolution and is charged all of it.
    let own_cost = {
        let _offscreen = h.create_render_target(OWN_SIZE, OWN_SIZE, D3DFMT_X8R8G8B8);
        base - h.available_texture_mem()
    };
    assert_eq!(
        own_cost,
        OWN_SIZE * OWN_SIZE * 4,
        "an intermediate target costs the size it asked for"
    );
}

/// What one worker's quadrant probes came to.
struct QuadrantOutcome {
    /// Probes that read a colour other than the one their quadrant owes.
    mismatches: u32,
    /// The first mismatch, as `(frame, x, y, colour)`.
    first_wrong: Option<(u32, u32, u32, u32)>,
    /// The first failing call, by name, with its hr.
    failed: Option<(&'static str, i32)>,
}

/// Draw the quadrant shader over the whole target and probe two pixels, `frames` times.
///
/// Both probes sit well inside their quadrant, so their colours hold at any
/// scale as long as the device applied its own. A wrong colour is counted and
/// the loop goes on, so the count says how often the crossing landed; the
/// first failing call stops it.
fn probe_own_quadrants(
    device: &SharedDevice<'_>,
    quad: &[RhwVertex; 6],
    frames: u32,
) -> QuadrantOutcome {
    /// `(x, y, colour)` in reported coordinates, alpha masked off.
    const PROBES: [(u32, u32, u32); 2] = [(480, 120, 0x0000_FF00), (160, 360, 0x0000_00FF)];

    let mut outcome = QuadrantOutcome {
        mismatches: 0,
        first_wrong: None,
        failed: None,
    };
    for frame in 0..frames {
        let calls = [
            ("Clear", device.clear_target(BLACK)),
            ("BeginScene", device.begin_scene()),
            (
                "DrawPrimitiveUP",
                device.draw_primitive_up(D3DPT_TRIANGLELIST, 2, quad),
            ),
            ("EndScene", device.end_scene()),
            ("Present", device.present()),
        ];
        if let Some(&(call, hr)) = calls.iter().find(|(_, hr)| *hr < 0) {
            outcome.failed = Some((call, hr));
            return outcome;
        }
        for (x, y, owed) in PROBES {
            match device.read_pixel(x, y) {
                Ok(pixel) if pixel & 0x00FF_FFFF == owed => {}
                Ok(pixel) => {
                    outcome.mismatches += 1;
                    outcome.first_wrong.get_or_insert((frame, x, y, pixel));
                }
                Err(failed) => {
                    outcome.failed = Some(failed);
                    return outcome;
                }
            }
        }
    }
    outcome
}

#[test]
fn two_devices_at_different_scales_read_their_own_vpos() {
    // A `ps_3_0` shader that declares `vPos` into a target rasterized below
    // the resolution D3D9 reports takes a shader variant whose `PsDraw`
    // uniform carries that target's scale, and both the variant flag and the
    // uniform come from the encoder of the device the target belongs to. Two
    // devices drawing that shader at once want two different compensations
    // and are owed the same picture in the reported space, so a compensation
    // that reached one device from the other reads as that device answering
    // in the wrong quadrants: at half scale an uncompensated `vPos` is short
    // of the centre everywhere the probes look, and at the identity a doubled
    // one is past it. Both devices pin their own scale so the pair is the
    // same on a scaled run of the suite.
    //
    // The crossing is timing-dependent, so one run guards the derivation
    // rather than proving it.
    const FRAMES: u32 = 120;

    let with_scale = |entries: &'static str| HarnessConfig {
        behavior_flags: D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
        config_entries: entries,
        ..HarnessConfig::default()
    };
    let half = Harness::create(&with_scale("render.scale=0.5"));
    let identity = Harness::create(&with_scale("render.scale=1.0"));

    let half_shader = half.create_pixel_shader(&QUADRANT_PS);
    assert_eq!(half.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0, "SetFVF");
    assert_eq!(half.set_pixel_shader(&half_shader), 0, "SetPixelShader");
    assert_eq!(
        half.set_pixel_shader_constant_f(0, &QUADRANT_CENTRE),
        0,
        "the reported centre in c0.zw"
    );
    let identity_shader = identity.create_pixel_shader(&QUADRANT_PS);
    assert_eq!(
        identity.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE),
        0,
        "SetFVF"
    );
    assert_eq!(
        identity.set_pixel_shader(&identity_shader),
        0,
        "SetPixelShader"
    );
    assert_eq!(
        identity.set_pixel_shader_constant_f(0, &QUADRANT_CENTRE),
        0,
        "the reported centre in c0.zw"
    );

    // Both devices report 640x480, so one screen-covering quad serves both.
    let (width, height) = half.dims();
    // The reported dims fit u16; convert without a precision-loss cast.
    let to_f = |v: u32| f32::from(u16::try_from(v).expect("reported dims fit u16"));
    let (w, hgt) = (to_f(width), to_f(height));
    let v = |x: f32, y: f32| RhwVertex {
        x,
        y,
        z: 0.5,
        rhw: 1.0,
        color: WHITE,
    };
    let quad = [
        v(0.0, 0.0),
        v(w, 0.0),
        v(0.0, hgt),
        v(w, 0.0),
        v(w, hgt),
        v(0.0, hgt),
    ];

    let half_shared = half.shared();
    let identity_shared = identity.shared();
    let finished = AtomicU32::new(0);
    std::thread::scope(|scope| {
        let half_worker = spawn_scoped(scope, || {
            let outcome = probe_own_quadrants(&half_shared, &quad, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            outcome
        });
        let identity_worker = spawn_scoped(scope, || {
            let outcome = probe_own_quadrants(&identity_shared, &quad, FRAMES);
            finished.fetch_add(1, Ordering::AcqRel);
            outcome
        });
        while finished.load(Ordering::Acquire) < 2 {
            assert!(half.pump(), "WM_QUIT on the half-scale window");
            assert!(identity.pump(), "WM_QUIT on the identity window");
            std::thread::yield_now();
        }
        for (name, worker) in [("half-scale", half_worker), ("identity", identity_worker)] {
            let outcome = worker.join().expect("a worker thread panicked");
            assert!(
                outcome.failed.is_none(),
                "{name} device: {} failed on its worker thread: 0x{:08X}",
                outcome.failed.map_or("", |(call, _)| call),
                outcome.failed.map_or(0, |(_, hr)| hr)
            );
            let (frame, x, y, pixel) = outcome.first_wrong.unwrap_or_default();
            assert_eq!(
                outcome.mismatches,
                0,
                "{name} device answered vPos in another device's space in {} of {} probes, \
                 first at frame {frame}, reported ({x}, {y}), reading 0x{pixel:08X}",
                outcome.mismatches,
                FRAMES * 2
            );
        }
    });
}

/// A quad over the clip-space box `[x0, x1] x [y0, y1]` at depth `z` in `color`, as a strip.
const fn clip_box(x0: f32, x1: f32, y0: f32, y1: f32, z: f32, color: u32) -> [PosColorVertex; 4] {
    [
        PosColorVertex {
            x: x0,
            y: y1,
            z,
            color,
        },
        PosColorVertex {
            x: x1,
            y: y1,
            z,
            color,
        },
        PosColorVertex {
            x: x0,
            y: y0,
            z,
            color,
        },
        PosColorVertex {
            x: x1,
            y: y0,
            z,
            color,
        },
    ]
}

/// A quad over the whole viewport at depth `z` in `color`.
const fn viewport_quad(z: f32, color: u32) -> [PosColorVertex; 4] {
    clip_box(-1.0, 1.0, -1.0, 1.0, z, color)
}

/// A device with the auto D24S8 depth-stencil, pinned at `render.scale=0.75`.
///
/// The tests of a render target of its own size over the back buffer's depth
/// surface pin the scale, since at the identity the two share one grid and
/// nothing they cover happens.
fn depth_harness_at_three_quarters() -> Harness {
    Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        config_entries: "render.scale=0.75",
        ..HarnessConfig::default()
    })
}

/// Set up a diffuse draw with the depth test under `compare` and depth writes as `write` says.
fn depth_draw_setup(h: &Harness, compare: u32, write: bool) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, u32::from(write)), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, compare), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
}

#[test]
fn a_small_target_over_the_back_buffer_depth_keeps_its_depth_writes_inside_its_extent() {
    // A 400x300 target is not the back buffer's size and rasterizes unscaled,
    // while the implicit depth surface scales with the back buffer. D3D9
    // pairs render-target pixel (x, y) with depth sample (x, y), so the depth
    // the draw writes stays inside x < 400 and y < 300 and the back buffer
    // reads it there. Holds at any scale.
    let h = Harness::with_depth();
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    let lower = D3DVIEWPORT9 {
        x: 0,
        y: 60,
        width: 640,
        height: 420,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(h.set_viewport(&lower), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, RED)),
        0
    );
    assert_eq!(h.set_render_target(0, &back_buffer), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(1.0, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(200, 90),
        BLACK,
        "inside the small target the drawn depth rejects the quad",
    );
    for (x, y) in [(440, 90), (520, 90), (520, 150)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("({x}, {y}) is past the small target and keeps the cleared depth"),
        );
    }
}

#[test]
fn depth_outside_a_small_targets_extent_survives_a_pass_through_it() {
    // Depth written through the scaled back buffer, then a pass through an
    // unscaled 400x300 target that writes depth inside its own extent only.
    // The back buffer then reads the small target's depth inside x < 400 and
    // y < 300, and its own everywhere else, unchanged by the pass between.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    depth_draw_setup(&h, D3DCMP_ALWAYS, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, RED)),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.75, BLUE)),
        0
    );
    assert_eq!(h.set_render_target(0, &back_buffer), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, false);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(200, 150),
        GREEN,
        "inside the small target its depth 0.75 passes the quad",
    );
    for (x, y) in [(520, 150), (200, 400), (600, 440)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            RED,
            &format!("({x}, {y}) keeps the back buffer's depth 0.25"),
        );
    }
}

#[test]
fn a_submission_ending_on_a_small_target_hands_its_depth_to_the_next_one() {
    // The pass through the unscaled 400x300 target is the last one of its
    // submission: the read-back of that target sends it. The draw after it,
    // in the next submission, reads the depth that pass wrote, in place.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    assert_pixel_eq(
        h.read_pixel(200, 150),
        BLUE,
        "the small target's draw, sent by its read-back",
    );
    assert_eq!(h.set_render_target(0, &back_buffer), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, false);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(200, 150),
        BLACK,
        "the small target's depth 0.25 rejects the quad",
    );
    for (x, y) in [(520, 150), (200, 400)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("({x}, {y}) is past the small target and keeps the cleared depth"),
        );
    }
}

#[test]
fn a_scaled_target_over_an_unscaled_depth_surface_keeps_its_depth_on_the_scaled_grid() {
    // The reverse pairing, pinned at what it does: the back buffer, scaled to
    // 480x360, over an 800x600 depth surface, which is not the back buffer's
    // size and stays unscaled. Depth the back buffer writes over x < 320
    // lands on the depth surface's texels x < 240, the scaled grid's, so a
    // pass through an unscaled 800x600 target reads it there. D3D9 reads it
    // at x < 320, y < 480, where (280, 100) and (280, 400) would be black.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let depth = h.create_depth_stencil_surface(800, 600, D3DFMT_D24S8);
    let large = h.create_render_target(800, 600, D3DFMT_X8R8G8B8);
    assert_eq!(h.set_depth_stencil_surface(&depth), 0);
    assert_eq!(h.set_render_target(0, &large), 0);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "the whole depth surface, through the target of its own size"
    );
    assert_eq!(h.set_render_target(0, &back_buffer), 0);
    depth_draw_setup(&h, D3DCMP_ALWAYS, true);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLESTRIP,
            2,
            &clip_box(-1.0, 0.0, -1.0, 1.0, 0.25, RED)
        ),
        0
    );
    assert_eq!(h.set_render_target(0, &large), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, false);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(100, 100),
        BLACK,
        "inside both grids' written area",
    );
    for (x, y) in [(280, 100), (280, 400), (500, 100)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("({x}, {y}) lies past the scaled grid's written texels"),
        );
    }
}

/// The whole reported frame as a viewport, for a target smaller than it.
const FULL_FRAME: D3DVIEWPORT9 = D3DVIEWPORT9 {
    x: 0,
    y: 0,
    width: 640,
    height: 480,
    min_z: 0.0,
    max_z: 1.0,
};

/// Draw a quad at 0.5 over the back buffer, depth-tested `LESSEQUAL` without depth writes.
///
/// It survives exactly where the depth surface holds 0.5 or more.
fn probe_quad_over_the_back_buffer(h: &Harness, back_buffer: &Surface<'_>) {
    assert_eq!(h.set_render_target(0, back_buffer), 0);
    depth_draw_setup(h, D3DCMP_LESSEQUAL, false);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);
}

/// Assert that depth 0.25 lies inside the 400x300 extent and 1.0 outside it.
fn assert_small_extent_holds_the_near_depth(h: &Harness) {
    assert_pixel_eq(
        h.read_pixel(200, 150),
        BLACK,
        "inside the small target the near depth rejects the quad",
    );
    for (x, y) in [(520, 150), (200, 400)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("({x}, {y}) is past the small target and keeps the cleared depth"),
        );
    }
}

#[test]
fn a_pass_through_a_small_target_continued_after_a_read_back_keeps_its_depth_in_place() {
    // The read-back of the 400x300 target ends a submission in the middle of
    // its pass; the draw after it continues through the same target. Both
    // draws land in place, so the nearer second one wins inside the target.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.75, BLUE)),
        0
    );
    assert_pixel_eq(
        h.read_pixel(200, 150),
        BLUE,
        "the first draw, sent by the read-back",
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, RED)),
        0
    );
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);
}

#[test]
fn a_rect_depth_clear_past_a_small_target_reaches_the_scaled_depth_surface() {
    // A depth rect reaching past an unscaled 400x300 target, under a viewport
    // over the whole frame, clears the implicit depth surface over the whole
    // rect, as at the identity scale.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    assert_eq!(h.set_viewport(&FULL_FRAME), 0);
    assert_eq!(
        h.clear_rects(
            D3DCLEAR_ZBUFFER,
            BLACK,
            0.0,
            0,
            &[D3DRECT {
                x1: 0,
                y1: 0,
                x2: 600,
                y2: 450,
            }],
        ),
        0,
        "a depth rect past the small target"
    );
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    for (x, y) in [(200, 150), (560, 420)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            BLACK,
            &format!("({x}, {y}) lies inside the rect and holds 0.0"),
        );
    }
    assert_pixel_eq(
        h.read_pixel(620, 465),
        GREEN,
        "outside the rect the depth keeps 1.0",
    );
}

#[test]
fn a_depth_stretch_rect_after_a_small_target_copies_the_depth_it_drew() {
    // A whole-surface depth copy of the implicit surface while a pass through
    // an unscaled 400x300 target is still bound over it: the copy holds the
    // depth that pass drew, in place.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let implicit = h.depth_stencil_surface().expect("the auto depth surface");
    let copy = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    assert_eq!(
        h.stretch_rect(&implicit, &copy, D3DTEXF_NONE),
        0,
        "copy the implicit depth surface"
    );
    assert_eq!(h.set_depth_stencil_surface(&copy), 0);
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);
}

#[test]
fn a_depth_surface_bound_over_a_small_target_takes_its_draws_in_place() {
    // A pass through an unscaled 400x300 target over the implicit surface,
    // then another depth surface at the back buffer's size bound under the
    // same target, cleared and drawn: the second surface holds that draw in
    // place, and the back buffer reads it there.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let other = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.75, BLUE)),
        0
    );
    assert_eq!(h.set_depth_stencil_surface(&other), 0);
    assert_eq!(h.set_viewport(&FULL_FRAME), 0);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, RED)),
        0
    );
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);
}

#[test]
fn a_resz_resolve_after_a_small_target_copies_the_depth_it_drew() {
    // A RESZ resolve into an INTZ texture at the back buffer's size while a
    // pass through an unscaled 400x300 target is still bound over the
    // implicit surface: the INTZ holds the depth that pass drew, in place,
    // read back by binding it as the back buffer's depth surface.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let implicit = h.depth_stencil_surface().expect("the auto depth surface");
    let intz = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_depth_stencil_surface(&intz.surface_level(0)), 0);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 0.0, 0),
        0,
        "prime the INTZ to 0.0, which a RESZ that never ran leaves"
    );
    assert_eq!(h.set_depth_stencil_surface(&implicit), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    assert_eq!(h.set_texture(0, &intz), 0, "bind the RESZ destination");
    assert_eq!(
        h.set_render_state(D3DRS_POINTSIZE, 0x7fa0_5000),
        0,
        "the RESZ magic value"
    );
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(h.set_depth_stencil_surface(&intz.surface_level(0)), 0);
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);
}

/// An INTZ texture at the back buffer's size, which scales with the back buffer.
fn back_buffer_sized_intz(h: &Harness) -> Texture<'_> {
    h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    )
}

#[test]
fn a_small_target_over_a_back_buffer_sized_depth_texture_keeps_its_depth_writes_inside_its_extent()
{
    // As over the implicit depth surface: a 400x300 target rasterizes
    // unscaled over a level of an INTZ texture that scales with the back
    // buffer, and the depth the draw writes stays inside x < 400 and y < 300.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let intz = back_buffer_sized_intz(&h);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_depth_stencil_surface(&intz.surface_level(0)), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    let lower = D3DVIEWPORT9 {
        x: 0,
        y: 60,
        width: 640,
        height: 420,
        min_z: 0.0,
        max_z: 1.0,
    };
    assert_eq!(h.set_viewport(&lower), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, RED)),
        0
    );
    assert_eq!(h.set_render_target(0, &back_buffer), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(1.0, GREEN)),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_eq(
        h.read_pixel(200, 90),
        BLACK,
        "inside the small target the drawn depth rejects the quad",
    );
    for (x, y) in [(440, 90), (520, 90), (520, 150)] {
        assert_pixel_eq(
            h.read_pixel(x, y),
            GREEN,
            &format!("({x}, {y}) is past the small target and keeps the cleared depth"),
        );
    }
}

#[test]
fn a_draw_through_a_small_target_samples_the_depth_its_pass_wrote_into_a_depth_texture() {
    // An INTZ texture at the back buffer's size is the depth surface of a
    // pass through an unscaled 400x300 target and is bound at stage 0 before
    // the pass draws. A draw writes depth 0.25 over the target, then a draw
    // without a depth test samples the INTZ across the target: D3D9 hands it
    // the depth the pass wrote, 0.25 inside reported x < 400 and y < 300 and
    // the cleared 1.0 past it. The quad maps the whole texture onto the
    // target, so target pixel (x, y) reads texture pixel (1.6 x, 1.6 y).
    let h = depth_harness_at_three_quarters();
    let intz = back_buffer_sized_intz(&h);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_depth_stencil_surface(&intz.surface_level(0)), 0);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    assert_eq!(h.set_texture(0, &intz), 0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_NONE),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0);
    }
    // The diffuse stage reads no texture, so this draw samples nothing.
    depth_draw_setup(&h, D3DCMP_ALWAYS, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );

    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    let shader = h.create_pixel_shader(&PS_SAMPLE_TEXTURE);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &FULLSCREEN_TEXTURED_QUAD),
        0
    );
    assert_eq!(h.end_scene(), 0);

    assert_pixel_approx(
        h.read_pixel(100, 100),
        0x4040_4040,
        1,
        "texture pixel (160, 160) holds the depth the pass wrote",
    );
    for (x, y) in [(300, 100), (100, 250)] {
        assert_pixel_approx(
            h.read_pixel(x, y),
            WHITE,
            1,
            &format!("target pixel ({x}, {y}) reads past the pass's extent, the cleared depth"),
        );
    }
}

#[test]
fn a_depth_surface_of_another_format_under_a_small_target_takes_its_draws_in_place() {
    // A pass through an unscaled 400x300 target over the implicit D24S8
    // surface, then one over a D24X8 surface at the back buffer's size: the
    // second surface has no stencil plane, so the pass draws into a plane of
    // another format and the first one's is given up. Each surface holds the
    // depth its pass drew, in place.
    let h = depth_harness_at_three_quarters();
    let back_buffer = h.render_target(0);
    let implicit = h.depth_stencil_surface().expect("the auto depth surface");
    let other = h.create_depth_stencil_surface(640, 480, D3DFMT_D24X8);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    assert_eq!(h.set_depth_stencil_surface(&other), 0);
    assert_eq!(h.set_viewport(&FULL_FRAME), 0);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, RED)),
        0
    );
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);

    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.set_depth_stencil_surface(&implicit), 0);
    assert_eq!(h.clear(D3DCLEAR_TARGET, BLACK, 1.0, 0), 0);
    probe_quad_over_the_back_buffer(&h, &back_buffer);
    assert_small_extent_holds_the_near_depth(&h);
}

/// Draw depth 0.25 through a new 400x300 target and read it in place through the back buffer.
///
/// Every reference it takes is released on return, so a `Reset` may follow.
fn small_target_pass_holds_its_depth(h: &Harness) {
    let back_buffer = h.render_target(0);
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);
    depth_draw_setup(h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    probe_quad_over_the_back_buffer(h, &back_buffer);
    assert_small_extent_holds_the_near_depth(h);
}

#[test]
fn a_reset_between_two_passes_through_a_small_target_leaves_the_second_its_depth() {
    // `Reset` gives up the plane the first pass drew into, with the implicit
    // depth surface it stood in for. The pass after it draws into a new one
    // over the new surface, at the same size and then at another.
    let h = depth_harness_at_three_quarters();
    small_target_pass_holds_its_depth(&h);
    assert_eq!(h.reset(640, 480), 0, "Reset between the two passes");
    small_target_pass_holds_its_depth(&h);
    assert_eq!(h.reset(800, 600), 0, "Reset to another size");
    small_target_pass_holds_its_depth(&h);
}

/// The name the workload child of the span-counting test runs under.
const SPAN_COUNT_CHILD_NAME: &str = "depth-plane-spans.exe";

/// That child's log filter: the unix side's command records, one per depth transfer.
const SPAN_COUNT_LOG_FILTER: &str = "warn,mtld3d::unix::command=debug";

/// What the record of every depth transfer carries.
const DEPTH_TRANSFER_RECORD: &str = "site=depth-transfer/0";

#[test]
fn a_draw_that_cannot_read_or_write_depth_opens_no_span() {
    if running_as(SPAN_COUNT_CHILD_NAME) {
        span_count_workload();
        return;
    }
    // The transfers are counted out of the process log, where every other
    // test's would land beside them, so the workload runs in a process of
    // its own, alone in its log directory.
    run_in_private_log_child(
        SPAN_COUNT_CHILD_NAME,
        "render_scale::a_draw_that_cannot_read_or_write_depth_opens_no_span",
        SPAN_COUNT_LOG_FILTER,
    );
}

/// Draws through a 400x300 target that leave the depth surface alone, then one that writes it.
///
/// A draw with the depth test on, a compare that always passes and depth
/// writes off (the state of a glow pass), and one with the stencil test on
/// in its default always-pass, keep-everything state, read and change no
/// depth or stencil value, so neither carries the depth surface into a plane.
/// Each read-back sends its submission, and the unix side writes a transfer's
/// record before the read-back returns. The depth-writing draw after them
/// records the two transfers of one span, which also shows the count sees
/// transfers at all.
fn span_count_workload() {
    let h = depth_harness_at_three_quarters();
    let small = h.create_render_target(400, 300, D3DFMT_A8R8G8B8);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(h.set_render_target(0, &small), 0);

    depth_draw_setup(&h, D3DCMP_ALWAYS, false);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, BLUE)),
        0
    );
    assert_pixel_eq(h.read_pixel(200, 150), BLUE, "the always-passing draw");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 1), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.25, RED)),
        0
    );
    assert_pixel_eq(h.read_pixel(200, 150), RED, "the idle-stencil draw");
    assert!(
        !logged_lines("readback-copy").is_empty(),
        "the log carries the command records of the submissions so far"
    );
    assert_eq!(
        logged_lines(DEPTH_TRANSFER_RECORD),
        Vec::<String>::new(),
        "a draw that reads and writes no depth or stencil carried the depth surface"
    );

    assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 0), 0);
    depth_draw_setup(&h, D3DCMP_LESSEQUAL, true);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &viewport_quad(0.5, GREEN)),
        0
    );
    assert_pixel_eq(h.read_pixel(200, 150), GREEN, "the depth-writing draw");
    assert_eq!(
        await_logged_lines(DEPTH_TRANSFER_RECORD, 2).len(),
        2,
        "one span: the transfer into the plane and the one back"
    );
    assert_eq!(h.end_scene(), 0);
}
