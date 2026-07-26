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
//! is a flat block several pixels from any colour boundary, where an edge-aware
//! upscale reproduces the source colour exactly. Sampling *on* a boundary would
//! be scale-dependent by construction, so the probes stay away from them.

use mtld3d_tests::{Harness, assert_pixel_eq};
use mtld3d_types::{D3DFMT_X8R8G8B8, D3DRECT, D3DRS_SCISSORTESTENABLE, D3DVIEWPORT9};

const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;

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
