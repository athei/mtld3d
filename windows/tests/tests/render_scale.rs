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
//! scale would not exercise the conversion at all.

use mtld3d_tests::{Harness, PosColorVertex, TexturedVertex, assert_pixel_eq};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_LESS, D3DCMP_LESSEQUAL, D3DFMT_A8R8G8B8,
    D3DFMT_D24S8, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT,
    D3DPT_POINTLIST, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_LIGHTING, D3DRS_POINTSIZE,
    D3DRS_SCISSORTESTENABLE, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_RENDERTARGET,
    D3DVIEWPORT9,
};

const RED: u32 = 0xFFFF_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const BLACK: u32 = 0xFF00_0000;
const WHITE: u32 = 0xFFFF_FFFF;
/// A colour whose alpha is neither zero nor opaque, so either mistake shows.
const TRANSLUCENT: u32 = 0x4000_FF00;

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
    // The harness process owns its environment and no other thread runs yet;
    // extend the suite-wide config with the scale under test. The parser keeps
    // the last segment, so this wins over a `make test SCALE=<n>` run too.
    let merged = format!(
        "{};render.scale=0.5",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the harness
    // and with it the config read are only constructed below).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };

    let h = Harness::new();
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

    // The harness process owns its environment and no other thread runs yet;
    // extend the suite-wide config with the scale under test. The parser keeps
    // the last segment, so this wins over a `make test SCALE=<n>` run too.
    let merged = format!(
        "{};render.scale=0.5",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the harness
    // and with it the config read are only constructed below).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };

    let h = Harness::new();
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

    // The harness process owns its environment and no other thread runs yet;
    // extend the suite-wide config with the scale under test. The parser keeps
    // the last segment, so this wins over a `make test SCALE=<n>` run too.
    let merged = format!(
        "{};render.scale=0.5",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the harness
    // and with it the config read are only constructed below).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };

    let h = Harness::new();
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
