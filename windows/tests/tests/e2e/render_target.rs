//! Offscreen render target round-trip and depth-buffered occlusion.
//!
//! The round-trip renders to a texture, then samples it.

use mtld3d_tests::{
    CubeTexture, Harness, HarnessConfig, PosColorVertex, Rgba8, Surface, TexturedVertex, Vertex,
    VolumeVertex,
};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_SRCALPHA, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER,
    D3DCMP_ALWAYS, D3DCMP_LESS, D3DCMP_LESSEQUAL, D3DERR_INVALIDCALL, D3DERR_NOTFOUND,
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8R8G8B8, D3DFMT_A16B16G16R16F,
    D3DFMT_A32B32G32R32F, D3DFMT_D24S8, D3DFMT_INTZ, D3DFMT_L8, D3DFMT_R5G6B5, D3DFMT_UYVY,
    D3DFMT_X1R5G5B5, D3DFMT_X8R8G8B8, D3DFMT_YUY2, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_DISCARD, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND,
    D3DRS_LIGHTING, D3DRS_SRCBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER,
    D3DSAMP_MIPFILTER, D3DTA_DIFFUSE, D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTEXF_LINEAR,
    D3DTEXF_NONE, D3DTEXF_POINT, D3DTOP_MODULATE, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP, D3DUSAGE_AUTOGENMIPMAP,
    D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_RENDERTARGET, D3DVIEWPORT9,
};

const RED: u32 = 0xFFFF_0000;
const BLACK: u32 = 0xFF00_0000;
const WHITE: u32 = 0xFFFF_FFFF;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;

/// `left`/`top`/`right`/`bottom` in surface coordinates.
const fn rect(x1: i32, y1: i32, x2: i32, y2: i32) -> D3DRECT {
    D3DRECT { x1, y1, x2, y2 }
}

/// `ps_3_0 { dcl_2d s0; dcl_texcoord0 v0; texld r0, v0, s0; mov oC0, r0; }`
///
/// Tokens follow the `D3DSHADER_PARAM` layout (bit 31 set; register type split
/// across bits `[30:28]` and `[12:11]`; `0xE4` = `.xyzw` swizzle; `0xF` write
/// mask). Sampling an INTZ (`Depth32Float`) texture on s0 drives the emitter's
/// raw-depth-fetch variant: `depth2d` bound + a plain `.sample()` returning the
/// stored normalized depth (INTZ/DF24/DF16 are NOT shadow-compare formats).
#[rustfmt::skip]
const PS_SAMPLE_DEPTH: [u32; 15] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
    0x0200_001F, 0x8000_0005, 0x900F_0000,              // dcl_texcoord0 v0
    0x0300_0042, 0x800F_0000, 0x90E4_0000, 0xA0E4_0800, // texld r0, v0, s0
    0x0200_0001, 0x800F_0800, 0x80E4_0000,              // mov oC0, r0
    0x0000_FFFF,                                        // end
];

/// A single triangle covering the whole viewport, in `color`.
const fn fullscreen_triangle(color: u32) -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color,
        },
    ]
}

#[test]
fn render_to_texture_then_sample() {
    let h = Harness::new();

    let rt = h.create_texture(
        256,
        256,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);
    let backbuffer = h.render_target(0);

    // MODULATE(texture, diffuse) so pass 2 shows the texel; set once.
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_MODULATE),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_COLORARG2, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0, "TSS");
    }
    // Lighting defaults ON; the draws below carry a diffuse colour but no normal,
    // so the lit path emits only the (zero) material ambient + emissive — black.
    // Disable lighting to exercise the unlit vertex-colour path this test checks.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");

    // ── Pass 1: fill the RT red (clear + an explicit draw so TBDR can't drop it).
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    assert_eq!(h.clear_texture(0), 0, "no texture for the fill draw");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    let fill = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill),
        0,
        "RT fill draw"
    );

    // ── Pass 2: back to the backbuffer, sample the RT onto a centred quad.
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");
    assert_eq!(h.clear_target(BLACK), 0, "clear backbuffer black");
    assert_eq!(h.set_texture(0, &rt), 0, "bind RT as texture");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF TEX1"
    );

    let quad = [
        TexturedVertex {
            x: -0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 1.0,
        },
        TexturedVertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 1.0,
        },
        TexturedVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 1.0,
        },
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample-RT draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    // Quad covers clip (-0.5,-0.5)..(0.5,0.5) → pixels (160,120)..(480,360).
    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        center.r > 200 && center.g < 40 && center.b < 40,
        "center samples red RT, got {center:?}"
    );
    let corner = Rgba8::from_pixel(h.read_pixel(10, 10));
    assert!(
        corner.r < 20 && corner.g < 20 && corner.b < 20,
        "corner stays black, got {corner:?}"
    );

    assert_eq!(h.clear_texture(0), 0, "unbind RT texture");
}

#[test]
fn depth_test_near_occludes_far() {
    let h = Harness::with_depth();
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "ZENABLE");
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL),
        0,
        "ZFUNC"
    );
    // Lighting defaults ON; these depth-marker quads carry a diffuse colour but
    // no normal, so the lit path would render them black. Disable lighting to
    // exercise the unlit vertex-colour path (the colours are depth markers).
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");

    let far = [
        Vertex {
            x: 0.0,
            y: 0.8,
            z: 0.7,
            color: RED,
        },
        Vertex {
            x: 0.8,
            y: -0.8,
            z: 0.7,
            color: RED,
        },
        Vertex {
            x: -0.8,
            y: -0.8,
            z: 0.7,
            color: RED,
        },
    ];
    let near = [
        Vertex {
            x: -0.2,
            y: 0.8,
            z: 0.3,
            color: 0xFF00_00FF,
        },
        Vertex {
            x: 0.6,
            y: -0.8,
            z: 0.3,
            color: 0xFF00_00FF,
        },
        Vertex {
            x: -1.0,
            y: -0.8,
            z: 0.3,
            color: 0xFF00_00FF,
        },
    ];
    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, GREEN, 1.0, 0),
        0,
        "clear color+depth"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &far),
        0,
        "far draw"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &near),
        0,
        "near draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(h.read_pixel(10, 10), GREEN, "background cleared green");
    let overlap = Rgba8::from_pixel(h.read_pixel(280, 300));
    assert!(
        overlap.b > overlap.r,
        "overlap: near blue wins, got {overlap:?}"
    );
    let far_only = Rgba8::from_pixel(h.read_pixel(500, 350));
    assert!(
        far_only.r > far_only.b,
        "far-only region is red, got {far_only:?}"
    );
}

#[test]
fn auto_depth_stencil_get_set_round_trip() {
    // A depth device exposes its auto depth-stencil; the save/restore pattern
    // (Get → … → Set) round-trips.
    let h = Harness::with_depth();
    let ds = h
        .depth_stencil_surface()
        .expect("auto depth-stencil present");
    let (hr, _desc) = ds.desc();
    assert_eq!(hr, 0, "depth-stencil surface describes");
    assert_eq!(
        h.set_depth_stencil_surface(&ds),
        0,
        "SetDepthStencilSurface(saved)"
    );
}

#[test]
fn create_depth_stencil_surface_succeeds() {
    let h = Harness::new();
    let ds = h.create_depth_stencil_surface(256, 256, D3DFMT_D24S8);
    let (hr, _desc) = ds.desc();
    assert_eq!(hr, 0, "created depth-stencil surface describes");
}

#[test]
fn get_depth_stencil_surface_reports_the_bound_surface() {
    // `GetDepthStencilSurface` answers with the object `SetDepthStencilSurface`
    // bound, not with the device's auto depth-stencil, so pointer identity
    // holds and the save/restore pattern round-trips through an app-created
    // surface. With nothing bound it reports `D3DERR_NOTFOUND` and nulls the
    // caller's out-pointer.
    let h = Harness::with_depth();
    let implicit = h
        .depth_stencil_surface()
        .expect("auto depth-stencil present");
    let custom = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
    assert_eq!(
        h.set_depth_stencil_surface(&custom),
        0,
        "bind the app-created depth-stencil"
    );

    let saved = h.depth_stencil_surface().expect("a depth-stencil is bound");
    assert_eq!(
        saved.as_ptr(),
        custom.as_ptr(),
        "GetDepthStencilSurface must hand back the bound surface"
    );

    // Save / bind another / restore: the restore has to put the app surface
    // back, not the auto depth the saved handle would name if `Get` reported
    // the implicit shell.
    assert_eq!(
        h.set_depth_stencil_surface(&implicit),
        0,
        "temporarily bind the auto depth-stencil"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&saved),
        0,
        "restore the saved depth-stencil"
    );
    let restored = h
        .depth_stencil_surface()
        .expect("a depth-stencil is bound after the restore");
    assert_eq!(
        restored.as_ptr(),
        custom.as_ptr(),
        "the restore must leave the app-created surface bound"
    );

    assert_eq!(h.clear_depth_stencil_surface(), 0, "unbind depth-stencil");
    let (hr, none) = h.depth_stencil_surface_hr();
    assert_eq!(hr, D3DERR_NOTFOUND, "no depth-stencil bound");
    assert!(none.is_none(), "a rejected Get nulls the out-pointer");
}

#[test]
fn depth_clear_with_rects_touches_only_the_rects() {
    // `Clear(D3DCLEAR_ZBUFFER, pRects)` clears depth inside the rects and
    // leaves the rest of the attachment alone, like a colour clear does.
    let h = Harness::with_depth();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);

    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let left_half = [D3DRECT {
        x1: 0,
        y1: 0,
        x2: 320,
        y2: 480,
    }];
    assert_eq!(
        h.clear_rects(D3DCLEAR_ZBUFFER, BLACK, 0.0, 0, &left_half),
        0,
        "rect-bounded depth clear"
    );

    // A full-screen quad at 0.5 passes only where depth stayed 1.0.
    let cover = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: GREEN,
        },
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &cover), 0);
    assert_eq!(h.end_scene(), 0);
    assert_eq!(
        h.read_pixel(160, 240),
        BLACK,
        "inside the rect depth is 0.0 and rejects the quad"
    );
    assert_eq!(
        h.read_pixel(480, 240),
        GREEN,
        "outside the rect depth keeps 1.0 and accepts the quad"
    );
}

#[test]
fn depth_to_depth_stretch_rect_copies_depth() {
    // A full-surface depth→depth StretchRect copies the source depth, so a
    // later depth test against the destination sees the copied values rather
    // than the destination's own clear.
    let h = Harness::with_depth();
    let src = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
    let dst = h.depth_stencil_surface().expect("implicit depth-stencil");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);

    // Source: depth 0.0 everywhere, 0.5 over the top-left 480x360 and 1.0
    // over the top-left 320x240 (rect-bounded clears, so the source pass
    // holds only clear quads).
    assert_eq!(h.set_depth_stencil_surface(&src), 0, "bind source depth");
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, BLACK, 0.0, 0), 0);
    let rect = |x2, y2| {
        [D3DRECT {
            x1: 0,
            y1: 0,
            x2,
            y2,
        }]
    };
    assert_eq!(
        h.clear_rects(D3DCLEAR_ZBUFFER, BLACK, 0.5, 0, &rect(480, 360)),
        0
    );
    assert_eq!(
        h.clear_rects(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0, &rect(320, 240)),
        0
    );
    // Copies into the still-unbound destination are valid and are then
    // overwritten by its clear below; the filtered form is accepted too.
    assert_eq!(h.stretch_rect(&src, &dst, D3DTEXF_POINT), 0, "early copy");
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_LINEAR),
        0,
        "filtered copy"
    );

    // Destination: cleared to red / 1.0, then its depth is overwritten by
    // the copy while the red clear must survive underneath.
    assert_eq!(
        h.set_depth_stencil_surface(&dst),
        0,
        "bind destination depth"
    );
    assert_eq!(h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, 1.0, 0), 0);
    assert_eq!(h.stretch_rect(&src, &dst, D3DTEXF_POINT), 0, "depth copy");

    // Two full-screen quads, green at 0.33 and blue at 0.66, depth-tested
    // against the copy without writing depth.
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL), 0);
    let cover = |z: f32, color: u32| {
        [
            PosColorVertex {
                x: -1.0,
                y: 3.0,
                z,
                color,
            },
            PosColorVertex {
                x: 3.0,
                y: -1.0,
                z,
                color,
            },
            PosColorVertex {
                x: -1.0,
                y: -1.0,
                z,
                color,
            },
        ]
    };
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &cover(0.33, GREEN)),
        0
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &cover(0.66, BLUE)),
        0
    );
    assert_eq!(h.end_scene(), 0);
    // Copied 1.0: both quads pass, blue on top. Copied 0.5: only green.
    // Copied 0.0: neither, the red clear shows.
    let expected = [
        [BLUE, BLUE, GREEN, RED],
        [BLUE, BLUE, GREEN, RED],
        [GREEN, GREEN, GREEN, RED],
        [RED, RED, RED, RED],
    ];
    for (i, row) in (0u32..).zip(&expected) {
        for (j, &want) in (0u32..).zip(row) {
            let x = 80 * (2 * j + 1);
            let y = 60 * (2 * i + 1);
            assert_eq!(
                h.read_pixel(x, y),
                want,
                "depth copied into the destination at ({x}, {y})"
            );
        }
    }
}

#[test]
fn stretch_rect_addresses_source_and_destination_mip_levels() {
    // A StretchRect between surfaces that are upper mip levels reads and
    // writes those levels, on both the scaling and the 1:1 path.
    let h = Harness::new();
    let tex = h.create_texture(
        128,
        128,
        2,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let level0 = tex.surface_level(0);
    let level1 = tex.surface_level(1);
    let backbuffer = h.render_target(0);
    let small = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);

    assert_eq!(h.set_render_target(0, &level0), 0);
    assert_eq!(h.clear_target(RED), 0);
    assert_eq!(h.set_render_target(0, &level1), 0);
    assert_eq!(h.clear_target(GREEN), 0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0);

    // Scaling path: level 1 (64x64) onto the 640x480 back buffer.
    assert_eq!(
        h.stretch_rect(&level1, &backbuffer, D3DTEXF_NONE),
        0,
        "scaled copy from level 1"
    );
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "scaled copy samples the source's own level"
    );

    // 1:1 path: level 1 (64x64) into a 64x64 target, then that target onto
    // the back buffer so it can be read.
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.stretch_rect(&level1, &small, D3DTEXF_NONE),
        0,
        "1:1 copy from level 1"
    );
    assert_eq!(
        h.stretch_rect(&small, &backbuffer, D3DTEXF_NONE),
        0,
        "scaled copy of the 1:1 result"
    );
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "1:1 copy reads the source's own level"
    );

    // 1:1 into level 1: paint the small target, copy it into level 1, then
    // read level 1 back through the scaling path.
    assert_eq!(h.set_render_target(0, &small), 0);
    assert_eq!(h.clear_target(WHITE), 0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(
        h.stretch_rect(&small, &level1, D3DTEXF_NONE),
        0,
        "1:1 copy into level 1"
    );
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.stretch_rect(&level1, &backbuffer, D3DTEXF_NONE),
        0,
        "scaled copy from the written level 1"
    );
    assert_eq!(
        h.read_pixel(320, 240),
        WHITE,
        "1:1 copy writes the destination's own level"
    );
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.stretch_rect(&level0, &backbuffer, D3DTEXF_NONE),
        0,
        "scaled copy from level 0"
    );
    assert_eq!(
        h.read_pixel(320, 240),
        RED,
        "level 0 is untouched by the level-1 writes"
    );
}

#[test]
fn clear_zbuffer_without_depth_stencil_is_invalid() {
    // `Clear(D3DCLEAR_ZBUFFER)` with no depth-stencil attachment bound is
    // invalid per the D3D9 spec. The guard must key on
    // whether a depth-stencil is *actually* bound, not on whether an auto
    // depth-stencil exists: a custom depth surface bound for an offscreen
    // render target still satisfies the clear.

    // (a) Explicit `SetDepthStencilSurface(NULL)` leaves no attachment: a
    // depth clear must fail.
    let h = Harness::with_depth();
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "auto depth-stencil present: depth clear succeeds",
    );
    assert_eq!(h.clear_depth_stencil_surface(), 0, "unbind depth-stencil");
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        D3DERR_INVALIDCALL,
        "depth clear with no depth-stencil bound is invalid",
    );

    // (b) An offscreen render target with a custom depth surface bound: the
    // auto depth handle does not reflect that surface, so the guard must not
    // regress this combined color+depth clear to INVALIDCALL.
    let rt = h.create_render_target(256, 256, D3DFMT_A8R8G8B8);
    let depth = h.create_depth_stencil_surface(256, 256, D3DFMT_D24S8);
    assert_eq!(
        h.set_render_target(0, &rt),
        0,
        "bind offscreen color target"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&depth),
        0,
        "bind custom depth surface",
    );
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "color+depth clear with a bound custom depth surface succeeds",
    );
}

#[test]
fn create_render_target_rejects_unrenderable_format() {
    // CreateRenderTarget is implemented for renderable color formats (see
    // create_render_target_default_pool_reports_desc); an unmappable / non-
    // renderable format is still rejected with INVALIDCALL.
    let h = Harness::new();
    assert_eq!(
        h.create_render_target_hr(640, 480, 0 /* D3DFMT_UNKNOWN */),
        D3DERR_INVALIDCALL,
        "CreateRenderTarget rejects an unmappable format",
    );
}

#[test]
fn back_buffer_desc_matches_device() {
    let h = Harness::new();
    let bb = h.back_buffer(0);
    let (hr, desc) = bb.desc();
    assert_eq!(hr, 0, "GetDesc");
    assert_eq!(
        (desc.width, desc.height),
        (640, 480),
        "backbuffer dimensions"
    );
}

#[test]
fn stretch_rect_accepts_one_to_one_same_format() {
    // StretchRect is accepted for a 1:1 same-format blit between a render-target
    // texture surface and the backbuffer (both BGRA8), and the copy carries the
    // source's content: a clear-only pass whose target is then copied out must
    // keep its store.
    let h = Harness::new();
    let rt = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);
    let backbuffer = h.render_target(0);

    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    assert_eq!(
        h.stretch_rect(&rt_surface, &backbuffer, D3DTEXF_NONE),
        0,
        "1:1 same-format StretchRect is accepted",
    );
    assert_eq!(
        h.read_pixel(320, 240),
        RED,
        "the copy carries the cleared colour into the backbuffer"
    );
}

#[test]
fn stretch_rect_copies_between_disjoint_rects_of_one_surface() {
    // D3D9 copies between two rectangles of one surface; titles use it to
    // scroll or duplicate a UI region. The rects here do not overlap, so the
    // copy rides the blit encoder inside the single texture.
    let h = Harness::new();
    let bb = h.render_target(0);
    assert_eq!(h.clear_target(BLACK), 0, "clear the back buffer");
    assert_eq!(
        h.clear_target_rects(RED, &[rect(0, 0, 64, 64)]),
        0,
        "paint the source block"
    );

    assert_eq!(
        h.stretch_rect_regions(
            &bb,
            &rect(0, 0, 64, 64),
            &bb,
            &rect(256, 128, 320, 192),
            D3DTEXF_NONE,
        ),
        D3D_OK,
        "a disjoint copy inside one surface is accepted"
    );
    assert_eq!(
        h.read_pixel(288, 160),
        RED,
        "the block reached the destination rect"
    );
    assert_eq!(h.read_pixel(32, 32), RED, "the source block is untouched");
    assert_eq!(
        h.read_pixel(400, 300),
        BLACK,
        "nothing outside the rects moved"
    );
}

#[test]
fn stretch_rect_shifts_an_overlapping_rect_of_one_surface() {
    // An overlapping copy reads the whole source region before it writes any
    // of the destination, so both halves of the source land shifted rather
    // than being smeared by the copy's own writes.
    let h = Harness::new();
    let bb = h.render_target(0);
    assert_eq!(h.clear_target(BLACK), 0, "clear the back buffer");
    assert_eq!(
        h.clear_target_rects(RED, &[rect(0, 0, 32, 64)]),
        0,
        "paint the source's left half"
    );
    assert_eq!(
        h.clear_target_rects(GREEN, &[rect(32, 0, 64, 64)]),
        0,
        "paint the source's right half"
    );

    assert_eq!(
        h.stretch_rect_regions(
            &bb,
            &rect(0, 0, 64, 64),
            &bb,
            &rect(32, 0, 96, 64),
            D3DTEXF_NONE,
        ),
        D3D_OK,
        "an overlapping copy inside one surface is accepted"
    );
    assert_eq!(
        h.read_pixel(16, 32),
        RED,
        "the part of the source the destination does not cover keeps its colour"
    );
    assert_eq!(
        h.read_pixel(48, 32),
        RED,
        "the source's left half arrives 32 pixels to the right"
    );
    assert_eq!(
        h.read_pixel(80, 32),
        GREEN,
        "the source's right half arrives 32 pixels to the right"
    );
    assert_eq!(
        h.read_pixel(100, 32),
        BLACK,
        "nothing past the destination rect moved"
    );
}

/// Sampling the INTZ texture that is still the bound depth attachment.
///
/// A deferred renderer keeps its scene depth bound for the depth test while
/// its light-volume draws sample it to reconstruct positions. Metal forbids
/// reading an attachment of the running pass, so the encoder copies the
/// attachment before such a draw and binds the copy: the sampled value is
/// the depth written earlier, not garbage. The depth is 0.25 (a value a
/// `1 - z` mix-up cannot fake) and the stage's filters are LINEAR, which
/// the fetch sampler must override: Apple GPUs cannot filter `Depth32Float`.
#[test]
fn intz_depth_sampled_while_bound_as_depth_attachment() {
    let h = Harness::new();
    let depth_tex = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth_surf = depth_tex.surface_level(0);
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "color target");
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "bind INTZ as depth"
    );
    assert_eq!(h.clear_texture(0), 0, "no sampler while writing depth");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let occluder = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &occluder),
        0,
        "depth write draw"
    );
    // The INTZ texture stays bound as the depth attachment: depth test on,
    // depth write off, and the same texture sampled through stage 0.
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_texture(0, &depth_tex), 0, "bind INTZ as a sampler");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        (D3DSAMP_MAGFILTER, D3DTEXF_LINEAR),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample-depth draw with the attachment still bound"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (48..=90).contains(&center.r)
            && (48..=90).contains(&center.g)
            && (48..=90).contains(&center.b),
        "the depth written by the occluder (0.25) samples back as dark gray, got {center:?}"
    );
    assert_eq!(h.clear_texture(0), 0, "unbind INTZ");
}

#[test]
fn intz_depth_sample_via_fixed_function() {
    // The cascade-shadow plumbing under the fixed-function pixel pipeline: an
    // INTZ texture is bound as a depth target, has depth rendered into it, then
    // is rebound as an FF texture stage and sampled in a later pass. Because the
    // texture is `Depth32Float`, the FF emitter must declare it `depth2d<float>`
    // and read it with `sample_compare` (the slot is a LessEqual comparison
    // sampler) — a plain `texture2d` + `sample()` trips Metal validation, which
    // is on under `make test`. `make test` is the regression guard for that.
    let h = Harness::new();
    let depth_tex = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth_surf = depth_tex.surface_level(0);
    let backbuffer = h.render_target(0);

    // ── Pass 1: write a known depth (0.5) into the INTZ surface.
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "color target");
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "bind INTZ as depth"
    );
    assert_eq!(
        h.clear_texture(0),
        0,
        "no sampler bound while writing depth"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 0.5, 0),
        0
    );
    let occluder = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &occluder),
        0,
        "depth write draw"
    );

    // ── Pass 2: swap to a scratch depth target (so INTZ is no longer the live
    // depth attachment), then sample the INTZ texture through stage 0.
    // The sample pass needs no depth: unbind it so INTZ stops being the live
    // depth attachment (otherwise it would be both attachment and sampler in a
    // single Metal encoder) — no separate depth surface, no format to match.
    assert_eq!(
        h.clear_depth_stencil_surface(),
        0,
        "unbind depth for the sample pass"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZENABLE, 0),
        0,
        "depth off for sample"
    );
    assert_eq!(h.set_texture(0, &depth_tex), 0, "bind INTZ as a sampler");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    // INTZ is a "readable raw depth" format (not a shadow-compare format):
    // `.sample()`
    // returns the stored normalized depth (0.5) broadcast to all channels — NOT
    // a 0/1 shadow comparison. Stage 0 MODULATE(texture, white diffuse) →
    // mid-gray ~0.5. Centre quad clips (-0.5,-0.5)..(0.5,0.5) → pixels
    // (160,120)..(480,360).
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample-depth draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (96..=160).contains(&center.r)
            && (96..=160).contains(&center.g)
            && (96..=160).contains(&center.b),
        "raw INTZ depth fetch (0.5) modulated by white diffuse should be ~mid-gray, got {center:?}"
    );
    let corner = Rgba8::from_pixel(h.read_pixel(10, 10));
    assert!(
        corner.r < 40 && corner.g < 40 && corner.b < 40,
        "corner stays cleared black, got {corner:?}"
    );

    assert_eq!(h.clear_texture(0), 0, "unbind INTZ");
}

/// A depth texture keeps its content across a standalone depth-surface bind.
///
/// A shadow-map pass renders into a `CreateTexture(DEPTHSTENCIL)` texture, and
/// the engine then binds a `CreateDepthStencilSurface` surface for the rest of
/// the frame, which ends the texture's pass. Nothing samples the texture that
/// frame, so the only thing keeping the attachment's store action at `Store` is
/// the `is_sampleable` flag `SetDepthStencilSurface` reports for a
/// texture-backed depth surface. The next frame restores that surface and
/// samples the texture: the depth written a frame earlier must still be there.
#[test]
fn sampleable_depth_survives_a_standalone_depth_bind() {
    let h = Harness::new();
    let depth_tex = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth_surf = depth_tex.surface_level(0);
    let scratch_depth = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
    let backbuffer = h.render_target(0);

    // ── Frame 1: write depth 0.5 into the texture, then bind the standalone
    // depth surface, which ends the texture's pass with no sample behind it.
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "color target");
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "bind the depth texture"
    );
    assert_eq!(
        h.clear_texture(0),
        0,
        "no sampler bound while writing depth"
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let occluder = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &occluder),
        0,
        "depth write draw"
    );
    assert_eq!(
        h.set_depth_stencil_surface(&scratch_depth),
        0,
        "bind the standalone depth surface"
    );
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear the standalone depth"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    // ── Frame 2: restore the texture's surface, then unbind depth (the
    // texture cannot be attachment and sampler in one Metal encoder) and
    // sample the depth written last frame.
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "restore the depth texture"
    );
    assert_eq!(
        h.clear_depth_stencil_surface(),
        0,
        "unbind depth for the sample pass"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZENABLE, 0),
        0,
        "depth off for sample"
    );
    assert_eq!(
        h.set_texture(0, &depth_tex),
        0,
        "bind the depth texture as a sampler"
    );
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample-depth draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    // Raw INTZ fetch of the stored 0.5 modulated by white diffuse: mid-gray.
    // A discarded attachment reads back as the 1.0 clear (white) or garbage.
    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (96..=160).contains(&center.r)
            && (96..=160).contains(&center.g)
            && (96..=160).contains(&center.b),
        "the depth written before the standalone bind samples back as mid-gray, got {center:?}"
    );
    let corner = Rgba8::from_pixel(h.read_pixel(10, 10));
    assert!(
        corner.r < 40 && corner.g < 40 && corner.b < 40,
        "corner stays cleared black, got {corner:?}"
    );
    assert_eq!(h.clear_texture(0), 0, "unbind the depth texture");
}

#[test]
fn intz_depth_sample_via_programmable_ps() {
    // Same INTZ create → render-depth → sample plumbing as the FF variant, but
    // the sampling pass runs a hand-assembled `ps_3_0` that does `texld` on s0.
    // Because slot 0 holds a `Depth32Float` texture, `depth_sampler_mask`
    // selects the `depth2d` + `sample_compare` variant — the path the real
    // cascade-shadow shaders take. The FF vertex pipeline feeds the
    // programmable PS (VS/PS source resolve independently). `make test` runs
    // with Metal validation on, so a depth/`texture2d` mismatch would fail here.
    let h = Harness::new();
    let depth_tex = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth_surf = depth_tex.surface_level(0);
    let backbuffer = h.render_target(0);

    // ── Pass 1: write a known depth (0.5) into the INTZ surface (FF pipeline).
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "color target");
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "bind INTZ as depth"
    );
    assert_eq!(
        h.clear_texture(0),
        0,
        "no sampler bound while writing depth"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 0.5, 0),
        0
    );
    let occluder = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: WHITE,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &occluder),
        0,
        "depth write draw"
    );

    // ── Pass 2: swap depth target, bind a programmable PS, sample the INTZ.
    let ps = h.create_pixel_shader(&PS_SAMPLE_DEPTH);
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");
    // The sample pass needs no depth: unbind it so INTZ stops being the live
    // depth attachment (otherwise it would be both attachment and sampler in a
    // single Metal encoder) — no separate depth surface, no format to match.
    assert_eq!(
        h.clear_depth_stencil_surface(),
        0,
        "unbind depth for the sample pass"
    );
    assert_eq!(
        h.set_render_state(D3DRS_ZENABLE, 0),
        0,
        "depth off for sample"
    );
    assert_eq!(h.set_texture(0, &depth_tex), 0, "bind INTZ as a sampler");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    // INTZ raw depth fetch: `texld` returns the stored normalized depth (0.5),
    // NOT a shadow comparison. The PS moves it to the output → mid-gray quad.
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample-depth draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (96..=160).contains(&center.r)
            && (96..=160).contains(&center.g)
            && (96..=160).contains(&center.b),
        "programmable raw INTZ depth fetch (texld→mov oC0) should output the stored 0.5 (mid-gray), got {center:?}"
    );
    let corner = Rgba8::from_pixel(h.read_pixel(10, 10));
    assert!(
        corner.r < 40 && corner.g < 40 && corner.b < 40,
        "corner stays cleared black, got {corner:?}"
    );

    assert_eq!(h.clear_pixel_shader(), 0, "unbind PS");
    assert_eq!(h.clear_texture(0), 0, "unbind INTZ");
}

#[test]
fn color_fill_render_target_texture_succeeds() {
    // ColorFill on a DEFAULT-pool render-target texture surface succeeds and
    // fills it. A non-RT texture is rejected.
    let h = Harness::new();
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(
        h.color_fill_hr(&rt.surface_level(0), 0xFF80_4020),
        0,
        "ColorFill on a DEFAULT render-target texture → S_OK",
    );

    // A plain managed texture (no RENDERTARGET usage) is not fillable.
    let plain = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(
        h.color_fill_hr(&plain.surface_level(0), 0xFF80_4020),
        D3DERR_INVALIDCALL,
        "ColorFill on a non-RT texture → INVALIDCALL",
    );
}

#[test]
fn fresh_default_offscreen_plain_round_trips_through_lock_rect() {
    // A DEFAULT offscreen plain is lockable, so it owns its level-0 staging
    // from creation on. Three locks a fresh surface has to serve out of that
    // staging: a read-only one on a surface nothing has written, a write, and
    // the read after it, with no draw, ColorFill or StretchRect in between. A
    // ColorFill of a plain no lock has ever touched reads back the same way.
    let h = Harness::new();
    let surface = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let (hr, desc) = surface.desc();
    assert_eq!(hr, D3D_OK, "a DEFAULT offscreen plain describes");
    assert_eq!(
        (desc.width, desc.height),
        (64, 64),
        "at the requested extent"
    );
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "in the requested pool");

    let pitch_px = {
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        assert!(
            pitch_px >= 64,
            "a read-only lock of a never-written plain maps a full row, got {pitch_px}",
        );
        pitch_px
    };

    let pattern: Vec<u32> = (0..pitch_px * 64)
        .map(|i| 0xFF00_0000 | (i.wrapping_mul(7) & 0x00FF_FFFF))
        .collect();
    {
        let mut locked = surface.lock_rect(0);
        assert_eq!(
            locked.pitch().cast_unsigned() / 4,
            pitch_px,
            "the pitch is stable across locks",
        );
        locked.write_u32(&pattern);
    }
    {
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        assert_eq!(
            locked.as_u32(pattern.len()),
            pattern.as_slice(),
            "LockRect reads back what the lock before it wrote",
        );
    }

    let fresh = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.color_fill_hr(&fresh, GREEN),
        D3D_OK,
        "ColorFill of a plain that has never been locked",
    );
    let locked = fresh.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let pixels = locked.as_u32((pitch_px * 64) as usize);
    assert_eq!(
        pixels[(32 * pitch_px + 32) as usize],
        GREEN,
        "the fill is visible to the lock that follows it",
    );
}

#[test]
fn color_fill_sub_rect_of_offscreen_plain_reads_back() {
    // A lockable DEFAULT offscreen-plain surface reads its fill back through
    // LockRect, so the fill has to be visible to the very next lock: inside the
    // rect it is the fill colour, outside it the seed the lock before wrote.
    // A rect hanging over the edge fills the part that lands on the surface.
    let h = Harness::new();
    let surface = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);

    {
        let mut locked = surface.lock_rect(0);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        let seed = vec![GREEN; (pitch_px * 64) as usize];
        locked.write_u32(&seed);
    }

    assert_eq!(
        h.color_fill_rect_hr(&surface, (16, 16, 48, 48), BLUE),
        D3D_OK,
        "ColorFill of a sub-rect on a DEFAULT offscreen-plain surface",
    );
    assert_eq!(
        h.color_fill_rect_hr(&surface, (56, 56, 96, 96), RED),
        D3D_OK,
        "ColorFill of a rect hanging over the surface edge is clipped, not rejected",
    );

    let locked = surface.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let pixels = locked.as_u32((pitch_px * 64) as usize);
    let at = |x: u32, y: u32| pixels[(y * pitch_px + x) as usize];
    assert_eq!(at(32, 32), BLUE, "inside the filled sub-rect");
    assert_eq!(at(16, 16), BLUE, "the sub-rect's top-left corner");
    assert_eq!(at(47, 47), BLUE, "the sub-rect's bottom-right corner");
    assert_eq!(at(8, 8), GREEN, "outside the sub-rect keeps the seed");
    assert_eq!(
        at(48, 48),
        GREEN,
        "one pixel past the sub-rect keeps the seed"
    );
    assert_eq!(at(60, 60), RED, "inside the clipped overhanging rect");
    assert_eq!(at(55, 55), GREEN, "outside the clipped overhanging rect");
}

#[test]
fn color_fill_of_offscreen_plain_packs_the_destination_format() {
    // A DEFAULT offscreen plain reads its fill back out of the staging the
    // fill wrote, so LockRect sees the encoded pixel itself: the packed
    // 16-bit formats hold the top bits of each D3DCOLOR channel, L8 the
    // colour's luminance and A8 its alpha byte. Every surface is seeded
    // first, so a fill that never lands reads back as the seed.
    const FILL: u32 = 0xDEAD_BEEF;
    let h = Harness::new();

    for (format, name, expected) in [
        (D3DFMT_A1R5G5B5, "A1R5G5B5", 0xD6FDu16),
        (D3DFMT_X1R5G5B5, "X1R5G5B5", 0xD6FD),
        (D3DFMT_A4R4G4B4, "A4R4G4B4", 0xDABE),
    ] {
        let surface = h.create_offscreen_plain_surface(64, 64, format, D3DPOOL_DEFAULT);
        {
            let mut locked = surface.lock_rect(0);
            let lanes = locked.pitch().cast_unsigned() / 2 * 64;
            locked.write(&vec![0x5555u16; lanes as usize]);
        }
        assert_eq!(
            h.color_fill_hr(&surface, FILL),
            D3D_OK,
            "ColorFill of a DEFAULT {name} offscreen plain",
        );
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        let pitch_lanes = locked.pitch().cast_unsigned() / 2;
        let lanes = locked.as_u16((pitch_lanes * 64) as usize);
        assert_eq!(
            lanes[(32 * pitch_lanes + 32) as usize],
            expected,
            "{name} packs the fill colour into its own layout",
        );
    }

    for (format, name, expected) in [(D3DFMT_L8, "L8", 0xBEu8), (D3DFMT_A8, "A8", 0xDE)] {
        let surface = h.create_offscreen_plain_surface(64, 64, format, D3DPOOL_DEFAULT);
        {
            let mut locked = surface.lock_rect(0);
            let bytes = locked.pitch().cast_unsigned() * 64;
            locked.write(&vec![0x55u8; bytes as usize]);
        }
        assert_eq!(
            h.color_fill_hr(&surface, FILL),
            D3D_OK,
            "ColorFill of a DEFAULT {name} offscreen plain",
        );
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        let pitch = locked.pitch().cast_unsigned();
        let bytes = locked.as_u8((pitch * 64) as usize);
        assert_eq!(
            bytes[(32 * pitch + 32) as usize],
            expected,
            "{name} takes the channel it stores",
        );
    }
}

#[test]
fn stretch_rect_into_offscreen_plain_is_visible_to_lock_rect() {
    // A StretchRect into a lockable DEFAULT offscreen plain writes the plain's
    // Metal texture; its LockRect reads CPU staging, so the lock has to
    // materialise the level from the texture. Whole-surface first, then a
    // sub-rect, which must land in the destination without disturbing the
    // pixels the copy did not cover.
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let dst = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let fill = |surface: &Surface<'_>, color: u32| {
        let mut locked = surface.lock_rect(0);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.write_u32(&vec![color; (pitch_px * 64) as usize]);
    };
    let read = |surface: &Surface<'_>, x: u32, y: u32| {
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.as_u32((pitch_px * 64) as usize)[(y * pitch_px + x) as usize]
    };

    fill(&src, RED);
    fill(&dst, GREEN);
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect between two DEFAULT offscreen plains",
    );
    assert_eq!(
        read(&dst, 32, 32),
        RED,
        "LockRect reads the copied pixels, not the seed the destination held",
    );

    fill(&src, BLUE);
    fill(&dst, GREEN);
    assert_eq!(
        h.stretch_rect_rects(&src, (0, 0, 32, 32), &dst, (32, 32, 64, 64), D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect of a sub-rect between two offscreen plains",
    );
    assert_eq!(read(&dst, 48, 48), BLUE, "inside the copied sub-rect");
    assert_eq!(read(&dst, 32, 32), BLUE, "the sub-rect's top-left corner");
    assert_eq!(
        read(&dst, 63, 63),
        BLUE,
        "the sub-rect's bottom-right corner"
    );
    assert_eq!(read(&dst, 16, 48), GREEN, "outside the copied sub-rect");
    assert_eq!(read(&dst, 31, 31), GREEN, "one pixel before the sub-rect");
}

#[test]
fn discard_lock_after_a_stretch_rect_keeps_what_the_lock_wrote() {
    // A StretchRect into a lockable DEFAULT offscreen plain leaves the level's
    // pixels on the plain's Metal texture alone. D3DLOCK_DISCARD declares them
    // dead, so that lock hands out staging without reading them back, and the
    // level belongs to the staging from then on: a claim left standing would
    // send the next lock to the GPU for the pixels the application had just
    // overwritten, and hand them back as the level's contents.
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let dst = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let fill = |surface: &Surface<'_>, color: u32, flags: u32| {
        let mut locked = surface.lock_rect(flags);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.write_u32(&vec![color; (pitch_px * 64) as usize]);
    };
    let read = |surface: &Surface<'_>, x: u32, y: u32| {
        let locked = surface.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.as_u32((pitch_px * 64) as usize)[(y * pitch_px + x) as usize]
    };

    fill(&src, RED, 0);
    fill(&dst, GREEN, 0);
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect between two DEFAULT offscreen plains",
    );
    fill(&dst, BLUE, D3DLOCK_DISCARD);
    assert_eq!(
        read(&dst, 32, 32),
        BLUE,
        "the lock after a discard reads what the discard lock wrote, not the blitted pixels",
    );
}

#[test]
fn stretch_rect_rejects_a_render_target_into_an_offscreen_plain() {
    // D3D9 allows an offscreen-plain destination only from an offscreen-plain
    // source: a render-target source, standalone or texture-backed, and the back
    // buffer are all INVALIDCALL. The reverse direction is allowed.
    let h = Harness::new();
    let plain = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let standalone = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);

    for (source, what) in [
        (&standalone, "a standalone CreateRenderTarget surface"),
        (&rt_surface, "a render-target texture surface"),
    ] {
        assert_eq!(
            h.stretch_rect(source, &plain, D3DTEXF_NONE),
            D3DERR_INVALIDCALL,
            "StretchRect from {what} into an offscreen plain",
        );
    }
    assert_eq!(
        h.stretch_rect(&plain, &standalone, D3DTEXF_NONE),
        D3D_OK,
        "StretchRect from an offscreen plain into a render target",
    );
}

#[test]
fn stretch_rect_into_a_texture_level_is_visible_to_get_dc() {
    // A StretchRect into a render-target texture's level writes that level's
    // Metal texture alone, while a GetDC on the level's surface builds its DIB
    // over the texture's CPU staging, so the DC has to take the read back the
    // claim on the level makes a LockRect take. The lock seeds the staging with
    // a colour the blit never writes, so a DC that skips the read back reads
    // green where the blit left red.
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let src = h.create_render_target(SIZE, SIZE, D3DFMT_A8R8G8B8);
    assert_eq!(h.color_fill_hr(&src, RED), D3D_OK, "fill the source red");
    let dst = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    {
        let mut locked = dst.lock_rect(0, 0);
        locked.write_u32(&[GREEN; TEXELS]);
    }
    let level = dst.surface_level(0);
    assert_eq!(
        h.stretch_rect(&src, &level, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect from a render target into a texture level",
    );

    let dc = level.dc();
    let last = (SIZE - 1).cast_signed();
    for (x, y, name) in [(0, 0, "first texel"), (last, last, "last texel")] {
        assert_eq!(
            dc.get_pixel(x, y),
            RED_COLORREF,
            "the DC reads the blit's {name}"
        );
    }
    assert_eq!(dc.release(), 0, "ReleaseDC");
}

#[test]
fn get_dc_on_a_default_offscreen_plain_rejects_while_it_is_locked() {
    // The plain's LockRect is recorded on the level-0 staging of the texture it
    // owns, not on the surface shell, so GetDC has to consult the texture to
    // see it. Rejected the same way a texture level's own surface is.
    const SIZE: u32 = 16;
    let sentinel = 0xdead_beef_usize as *mut core::ffi::c_void;
    let h = Harness::new();
    let plain = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    {
        let _locked = plain.lock_rect(0);
        let (hr, out) = plain.get_dc(sentinel);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "GetDC while the plain's LockRect is outstanding must return INVALIDCALL"
        );
        assert_eq!(
            out, sentinel,
            "a rejected GetDC must not write through the out HDC"
        );
    }
    let dc = plain.dc();
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");
}

#[test]
fn get_dc_on_a_default_offscreen_plain_round_trips_through_gdi() {
    // The classic GDI-on-a-surface case: a game draws text or an overlay into
    // a DEFAULT offscreen plain and copies the result somewhere. The plain's
    // pixels live in the level-0 staging of the texture it owns, so the DC
    // covers that store and reads the ColorFill that came before it, and what
    // GDI drew through the DC survives ReleaseDC in both directions the
    // surface can be read: a LockRect of the same staging, and a StretchRect
    // into the back buffer, which reads the level's Metal texture instead.
    //
    // GDI paints a block and every probe stays well inside its colour: under
    // a `render.scale` the "1:1" StretchRect lands in a back buffer rasterized
    // smaller and the read-back resolves it up again, and a lone pixel does
    // not survive that pair (it comes back as a blend of itself and its
    // neighbours). An interior pixel of a block does, at any scale.
    const SIZE: u32 = 64;
    const BLOCK: i32 = 32;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    // GDI knows no alpha: SetPixel stores the three colour bytes and leaves
    // the fourth at zero, so the pixel comes back as red with no alpha.
    const GDI_RED: u32 = 0x00FF_0000;
    let h = Harness::new();
    let plain = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.color_fill_hr(&plain, GREEN),
        D3D_OK,
        "ColorFill the plain green"
    );

    let dc = plain.dc();
    assert_eq!(
        dc.get_pixel(32, 32),
        GREEN_COLORREF,
        "the DC reads the fill the plain holds",
    );
    dc.fill_block(BLOCK, RED_COLORREF);
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");

    {
        let locked = plain.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        let px = locked.as_u32((pitch_px * SIZE) as usize);
        assert_eq!(
            px[(10 * pitch_px + 10) as usize],
            GDI_RED,
            "LockRect reads what GDI drew through the DC",
        );
        assert_eq!(
            px[(48 * pitch_px + 48) as usize],
            GREEN,
            "and the fill everywhere GDI left alone",
        );
    }

    let bb = h.render_target(0);
    let rect = (0, 0, SIZE.cast_signed(), SIZE.cast_signed());
    assert_eq!(
        h.stretch_rect_rects(&plain, rect, &bb, rect, D3DTEXF_NONE),
        D3D_OK,
        "1:1 StretchRect from the plain into the back buffer",
    );
    // The back buffer is X8R8G8B8, so only its three colour channels carry a
    // defined value; compare on those alone.
    assert_rgb_close(
        h.read_pixel(10, 10),
        GDI_RED,
        0,
        "GDI's block reaches the back buffer",
    );
    assert_rgb_close(
        h.read_pixel(48, 48),
        GREEN,
        0,
        "and so does the fill around it",
    );
}

#[test]
fn stretch_rect_into_a_cube_face_is_visible_to_that_face_get_dc() {
    // A StretchRect into a cube face's surface writes that face's slice of the
    // cube's Metal texture and never its CPU staging, so a GetDC on the face has
    // to read that face back. Face 0 is filled red and face 3 is blitted green:
    // a claim without a face dimension leaves face 3's staging stale, and a
    // destination slice without a face lands the blit on face 0.
    const EDGE: u32 = 64;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let src = h.create_render_target(EDGE, EDGE, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.color_fill_hr(&src, GREEN),
        D3D_OK,
        "fill the source green"
    );
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face0 = cube.surface(0, 0);
    let face3 = cube.surface(3, 0);
    assert_eq!(h.color_fill_hr(&face0, RED), D3D_OK, "fill face 0 red");
    assert_eq!(
        h.stretch_rect(&src, &face3, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect from a render target into a cube face",
    );

    let last = (EDGE - 1).cast_signed();
    let dc = face3.dc();
    for (x, y, name) in [(0, 0, "first texel"), (last, last, "last texel")] {
        assert_eq!(
            dc.get_pixel(x, y),
            GREEN_COLORREF,
            "face 3's DC reads the blit's {name}"
        );
    }
    assert_eq!(dc.release(), 0, "ReleaseDC on face 3");
    // A cube map's faces share one DC lock, so face 0 is readable only once
    // face 3's device context is gone.
    let dc = face0.dc();
    assert_eq!(
        dc.get_pixel(0, 0),
        RED_COLORREF,
        "the blit into face 3 left face 0 alone"
    );
    assert_eq!(dc.release(), 0, "ReleaseDC on face 0");
}

#[test]
fn stretch_rect_out_of_a_cube_face_reads_that_face() {
    // A 1:1 same-format StretchRect replays as a blit copy, which names a
    // source slice as well as a destination one, so a cube source has to read
    // the face the call named. Faces 0 and 3 carry different colours and the
    // destination is a plain 2D render target, so a copy pinned to slice 0
    // answers with face 0's fill instead of face 3's.
    const EDGE: u32 = 64;
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face0 = cube.surface(0, 0);
    let face3 = cube.surface(3, 0);
    assert_eq!(h.color_fill_hr(&face0, RED), D3D_OK, "fill face 0 red");
    assert_eq!(h.color_fill_hr(&face3, GREEN), D3D_OK, "fill face 3 green");

    let dst = h.create_render_target(EDGE, EDGE, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.color_fill_hr(&dst, BLUE),
        D3D_OK,
        "fill the destination blue"
    );
    assert_eq!(
        h.stretch_rect(&face3, &dst, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect out of a cube face",
    );
    assert_eq!(
        read_surface_pixel(&h, &dst, 1, 1),
        GREEN,
        "the copy read face 3 rather than face 0",
    );
}

/// Bind `cube`, sample it across the back buffer along `direction`, return the centre pixel.
///
/// The direction is constant across the quad, so every pixel reads the one cube
/// face that direction names and the face's own orientation drops out. The
/// three-component texcoord `VolumeVertex` carries is the direction the
/// fixed-function cube lookup takes.
fn sample_cube_face(h: &Harness, cube: &CubeTexture<'_>, direction: (f32, f32, f32)) -> u32 {
    assert_eq!(h.set_cube_texture(0, cube), 0, "bind the cube");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler state");
    }
    // D3DFVF_TEXCOORDSIZE3(0) is bit 16.
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | 0x0001_0000),
        0,
        "SetFVF with a three-component texcoord"
    );
    let vertex = |x: f32, y: f32| VolumeVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u: direction.0,
        v: direction.1,
        w: direction.2,
    };
    let quad = [
        vertex(-1.0, 1.0),
        vertex(1.0, 1.0),
        vertex(-1.0, -1.0),
        vertex(1.0, 1.0),
        vertex(1.0, -1.0),
        vertex(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "cube sample draw"
        );
    });
    let pixel = h.read_pixel(320, 240);
    assert_eq!(h.clear_texture(0), 0, "unbind the cube");
    pixel
}

#[test]
fn scaling_stretch_rect_into_a_cube_face_writes_that_face() {
    // A scaling StretchRect runs the render quad, which attaches the
    // destination as a colour target: a cube destination has to attach the face
    // the call named as the attachment's slice, or the quad lands on face 0.
    // Face 0 is filled red and a 32x32 green source is scaled onto face 3, so a
    // quad without a face reads green where face 0's own fill belongs.
    const EDGE: u32 = 64;
    let h = Harness::new();
    let src = h.create_render_target(EDGE / 2, EDGE / 2, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.color_fill_hr(&src, GREEN),
        D3D_OK,
        "fill the source green"
    );
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face0 = cube.surface(0, 0);
    let face3 = cube.surface(3, 0);
    assert_eq!(h.color_fill_hr(&face0, RED), D3D_OK, "fill face 0 red");
    assert_eq!(
        h.stretch_rect(&src, &face3, D3DTEXF_POINT),
        D3D_OK,
        "scaling StretchRect from a render target into a cube face",
    );

    assert_eq!(
        sample_cube_face(&h, &cube, (0.0, -1.0, 0.0)),
        GREEN,
        "face 3 carries the scaled blit",
    );
    assert_eq!(
        sample_cube_face(&h, &cube, (1.0, 0.0, 0.0)),
        RED,
        "the blit into face 3 left face 0 alone",
    );
}

#[test]
fn scaling_stretch_rect_out_of_a_cube_face_reads_that_face() {
    // The render quad's fragment function samples a 2D texture, so a cube
    // source reaches it as a view of the face the call named. Bound as a cube
    // it samples face 0 instead, so face 0 and face 2 are filled differently
    // and the blit out of face 2 must carry face 2's colour.
    const EDGE: u32 = 64;
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face0 = cube.surface(0, 0);
    let face2 = cube.surface(2, 0);
    assert_eq!(h.color_fill_hr(&face0, RED), D3D_OK, "fill face 0 red");
    assert_eq!(h.color_fill_hr(&face2, BLUE), D3D_OK, "fill face 2 blue");

    let backbuffer = h.render_target(0);
    assert_eq!(h.clear_target(BLACK), 0, "clear the back buffer");
    assert_eq!(
        h.stretch_rect(&face2, &backbuffer, D3DTEXF_POINT),
        D3D_OK,
        "scaling StretchRect from a cube face onto the back buffer",
    );
    assert_eq!(
        h.read_pixel(320, 240),
        BLUE,
        "the blit sampled face 2, not face 0",
    );
}

#[test]
fn stretch_rect_between_two_cube_faces_copies_face_to_face() {
    // Both surfaces of a copy inside one cube map resolve to the same Metal
    // texture, so the route is picked inside that texture: two faces are two
    // slices and the same rect on each names different texels, which is a real
    // copy the blit encoder runs in place. A route blind to the faces reads the
    // pair as one rect copied onto itself and skips the call, and a copy pinned
    // to slice 0 lands on face 0. Face 1 is green, face 3 is red, and the full
    // face is copied 1 to 1.
    const EDGE: u32 = 64;
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face1 = cube.surface(1, 0);
    let face3 = cube.surface(3, 0);
    assert_eq!(h.color_fill_hr(&face1, GREEN), D3D_OK, "fill face 1 green");
    assert_eq!(h.color_fill_hr(&face3, RED), D3D_OK, "fill face 3 red");
    assert_eq!(
        h.stretch_rect(&face1, &face3, D3DTEXF_NONE),
        D3D_OK,
        "1:1 same-format StretchRect from one cube face onto another",
    );

    assert_eq!(
        read_surface_pixel(&h, &face3, 1, 1),
        GREEN,
        "face 3 carries face 1's colour",
    );
    assert_eq!(
        read_surface_pixel(&h, &face1, 1, 1),
        GREEN,
        "the source face is left as it was",
    );
}

#[test]
fn scaling_stretch_rect_between_two_cube_faces_copies_face_to_face() {
    // The scaling form of the same pair stages through a scratch texture, since
    // the render quad cannot sample the texture it draws into. Both halves of
    // that detour carry a face: the copy out reads the source face and the quad
    // attaches the destination face. Face 1 is green and face 3 is red, and the
    // whole of face 1 lands in face 3's top-left quarter.
    const EDGE: u32 = 64;
    const HALF: i32 = (EDGE / 2).cast_signed();
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face1 = cube.surface(1, 0);
    let face3 = cube.surface(3, 0);
    assert_eq!(h.color_fill_hr(&face1, GREEN), D3D_OK, "fill face 1 green");
    assert_eq!(h.color_fill_hr(&face3, RED), D3D_OK, "fill face 3 red");
    assert_eq!(
        h.stretch_rect_rects(
            &face1,
            (0, 0, EDGE.cast_signed(), EDGE.cast_signed()),
            &face3,
            (0, 0, HALF, HALF),
            D3DTEXF_POINT,
        ),
        D3D_OK,
        "scaling StretchRect from one cube face onto another",
    );

    assert_eq!(
        read_surface_pixel(&h, &face3, 1, 1),
        GREEN,
        "the scaled copy landed in face 3, carrying face 1's colour",
    );
    assert_eq!(
        read_surface_pixel(&h, &face3, EDGE - 2, EDGE - 2),
        RED,
        "and left the rest of face 3 alone",
    );
}

#[test]
fn color_fill_of_a_cube_face_reaches_that_face_alone() {
    // ColorFill on a cube face's surface runs a render pass over that face's
    // slice and leaves the face's CPU staging untouched, so the GetDC that reads
    // it back has to name the same face. Two faces are filled different colours:
    // a fill without a face writes both into face 0, and a DC without the claim
    // reads the staging neither fill reached.
    const EDGE: u32 = 32;
    const BLUE_COLORREF: u32 = 0x00FF_0000;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face2 = cube.surface(2, 0);
    let face5 = cube.surface(5, 0);
    assert_eq!(h.color_fill_hr(&face2, BLUE), D3D_OK, "fill face 2 blue");
    assert_eq!(h.color_fill_hr(&face5, RED), D3D_OK, "fill face 5 red");

    let dc = face2.dc();
    assert_eq!(
        dc.get_pixel(1, 1),
        BLUE_COLORREF,
        "face 2 keeps its own fill"
    );
    assert_eq!(dc.release(), 0, "ReleaseDC on face 2");
    let dc = face5.dc();
    assert_eq!(
        dc.get_pixel(1, 1),
        RED_COLORREF,
        "face 5 keeps its own fill"
    );
    assert_eq!(dc.release(), 0, "ReleaseDC on face 5");
}

#[test]
fn color_fill_render_target_overwrites_earlier_draws() {
    // ColorFill on a render target is ordered against the draws around it: it
    // wipes what the frame already drew, and the draw after it blends against
    // the fill colour rather than against what the fill replaced.
    let h = Harness::new();
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture bound");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    assert_eq!(h.clear_target(BLACK), 0, "clear RT black");

    // An opaque red triangle over the whole target, which the fill must erase.
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen_triangle(RED)),
        0,
        "pre-fill draw",
    );
    assert_eq!(h.end_scene(), 0);

    assert_eq!(h.color_fill_hr(&rt_surface, BLUE), D3D_OK, "ColorFill blue");

    // A half-transparent red triangle blended over the fill.
    for (state, value) in [
        (D3DRS_ALPHABLENDENABLE, 1),
        (D3DRS_SRCBLEND, D3DBLEND_SRCALPHA),
        (D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA),
    ] {
        assert_eq!(h.set_render_state(state, value), 0, "blend state");
    }
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fullscreen_triangle(0x80FF_0000)),
        0,
        "blended draw",
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(
        h.set_render_state(D3DRS_ALPHABLENDENABLE, 0),
        0,
        "blend off"
    );

    let sysmem = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&rt_surface, &sysmem),
        0,
        "read the render target back",
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let idx = (32 * pitch_px + 32) as usize;
    let center = Rgba8::from_pixel(locked.as_u32(idx + 1)[idx]);
    assert!(
        (96..=160).contains(&center.r) && center.g < 40 && (96..=160).contains(&center.b),
        "half-alpha red over the blue fill, got {center:?}",
    );
}

#[test]
fn scaled_stretch_rect_into_a_texture_level_is_visible_to_lock_rect() {
    // A scaling StretchRect cannot go through Metal's blit encoder, so it
    // renders the source onto a quad covering the destination's render-target
    // texture level, writing that level's Metal texture alone. A LockRect on
    // the level reads CPU staging, so it has to take the read back the claim on
    // the level makes. The lock seeds the staging with a colour the quad never
    // writes, so a lock that skips the read back reads green where the quad
    // left red.
    const SIZE: u32 = 32;
    let h = Harness::new();
    let src = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.color_fill_hr(&src, RED), D3D_OK, "fill the source red");
    let dst = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    {
        let mut locked = dst.lock_rect(0, 0);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.write_u32(&vec![GREEN; (pitch_px * SIZE) as usize]);
    }
    let level = dst.surface_level(0);
    assert_eq!(
        h.stretch_rect(&src, &level, D3DTEXF_POINT),
        D3D_OK,
        "64x64 render target into a 32x32 render-target texture level",
    );

    let locked = dst.lock_rect(0, D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let texels = locked.as_u32((pitch_px * SIZE) as usize);
    for (x, y, name) in [(0, 0, "first texel"), (SIZE - 1, SIZE - 1, "last texel")] {
        assert_eq!(
            texels[(y * pitch_px + x) as usize],
            RED,
            "the lock reads the scaled copy's {name}"
        );
    }
}

#[test]
fn cross_format_stretch_rect_into_a_texture_level_is_visible_to_lock_rect() {
    // Same-size but cross-format, which the blit encoder cannot convert, so it
    // takes the same render quad a scale does and the same claim has to follow.
    // The R5G6B5 source is a DEFAULT offscreen plain, whose Metal format
    // differs from the A8R8G8B8 destination's.
    const SIZE: u32 = 32;
    const RED_565: u16 = 0xF800;
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    {
        let mut locked = src.lock_rect(0);
        let pitch_px = locked.pitch().cast_unsigned() / 2;
        locked.write(&vec![RED_565; (pitch_px * SIZE) as usize]);
    }
    let dst = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    {
        let mut locked = dst.lock_rect(0, 0);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.write_u32(&vec![GREEN; (pitch_px * SIZE) as usize]);
    }
    let level = dst.surface_level(0);
    assert_eq!(
        h.stretch_rect(&src, &level, D3DTEXF_POINT),
        D3D_OK,
        "R5G6B5 offscreen plain into an A8R8G8B8 render-target texture level",
    );

    let locked = dst.lock_rect(0, D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let texels = locked.as_u32((pitch_px * SIZE) as usize);
    for (x, y, name) in [(0, 0, "first texel"), (SIZE - 1, SIZE - 1, "last texel")] {
        assert_eq!(
            texels[(y * pitch_px + x) as usize],
            RED,
            "the lock reads the converted copy's {name}"
        );
    }
}

#[test]
fn color_fill_lockable_render_target_is_visible_to_lock_rect() {
    // A lockable CreateRenderTarget surface serves LockRect out of CPU staging
    // while ColorFill paints its Metal texture, so the lock has to read the
    // texture back: the whole-surface fill, then a sub-rect that leaves the
    // rest of the surface on the colour the fill before it left.
    let h = Harness::new();
    let rt = h.create_lockable_render_target(64, 64, D3DFMT_A8R8G8B8);
    let read = |x: u32, y: u32| {
        let locked = rt.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.as_u32((pitch_px * 64) as usize)[(y * pitch_px + x) as usize]
    };

    assert_eq!(
        h.color_fill_hr(&rt, GREEN),
        D3D_OK,
        "whole-surface ColorFill"
    );
    assert_eq!(read(32, 32), GREEN, "LockRect reads the fill colour");

    assert_eq!(
        h.color_fill_rect_hr(&rt, (16, 16, 48, 48), BLUE),
        D3D_OK,
        "sub-rect ColorFill",
    );
    assert_eq!(read(32, 32), BLUE, "inside the filled sub-rect");
    assert_eq!(read(16, 16), BLUE, "the sub-rect's top-left corner");
    assert_eq!(read(47, 47), BLUE, "the sub-rect's bottom-right corner");
    assert_eq!(
        read(8, 8),
        GREEN,
        "outside the sub-rect keeps the first fill"
    );
    assert_eq!(read(48, 48), GREEN, "one pixel past the sub-rect");
}

#[test]
fn draw_into_a_lockable_render_target_is_visible_to_lock_rect() {
    // Same contract for the other GPU writer of a lockable CreateRenderTarget
    // surface: a Clear and a draw land in its Metal texture, and the LockRect
    // that follows reports them rather than the staging they never touched.
    let h = Harness::new();
    let bb = h.render_target(0);
    let rt = h.create_lockable_render_target(64, 64, D3DFMT_A8R8G8B8);
    let read = |x: u32, y: u32| {
        let locked = rt.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.as_u32((pitch_px * 64) as usize)[(y * pitch_px + x) as usize]
    };

    assert_eq!(h.set_render_target(0, &rt), 0, "bind the lockable RT");
    assert_eq!(h.clear_target(GREEN), 0, "clear it green");
    assert_eq!(h.set_render_target(0, &bb), 0, "restore the backbuffer");
    assert_eq!(read(32, 32), GREEN, "LockRect reads the clear colour");

    assert_eq!(h.set_render_target(0, &rt), 0, "bind the lockable RT again");
    draw_fill(&h, RED);
    assert_eq!(h.set_render_target(0, &bb), 0, "restore the backbuffer");
    assert_eq!(read(32, 32), RED, "LockRect reads the drawn colour");
}

#[test]
fn get_dc_on_a_lockable_render_target_round_trips_through_the_gpu() {
    // GetDC hands out a DIB over the same CPU staging LockRect serves, so it
    // owes the surface the same coherence in both directions: the DC shows
    // what the GPU painted before it, and what GDI draws into the DC reaches
    // the colour texture at ReleaseDC.
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let rt = h.create_lockable_render_target(64, 64, D3DFMT_A8R8G8B8);

    assert_eq!(h.color_fill_hr(&rt, GREEN), D3D_OK, "ColorFill green");
    let dc = rt.dc();
    assert_eq!(
        dc.get_pixel(32, 32),
        GREEN_COLORREF,
        "the DC reads the fill the GPU painted, not the staging under it",
    );
    assert_eq!(
        dc.set_pixel(10, 10, RED_COLORREF),
        RED_COLORREF,
        "SetPixel into the DC stores the colour it was handed",
    );
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");

    // GDI knows no alpha: SetPixel stores the three colour bytes and leaves
    // the fourth at zero, so the pixel comes back as red with no alpha.
    assert_eq!(
        read_surface_pixel(&h, &rt, 10, 10),
        0x00FF_0000,
        "what GDI drew into the DC reaches the colour texture",
    );
    assert_eq!(
        read_surface_pixel(&h, &rt, 32, 32),
        GREEN,
        "the pixels GDI left alone still hold the fill",
    );
}

#[test]
fn get_dc_on_an_odd_width_16_bit_lockable_render_target_reaches_the_last_row() {
    // A row of an odd number of 2-byte pixels is not a whole number of
    // dwords, and GDI steps a DIB by the row length rounded up to four bytes.
    // The staging the DC wraps has to carry that same stride, or every row the
    // DC reads starts two bytes late and the last one falls out of the buffer
    // entirely: its pixels read as black and GDI's own drawing into it never
    // reaches the colour texture.
    const W: u32 = 33;
    const H: u32 = 4;
    const GREEN_565: u16 = 0x07E0;
    const RED_565: u16 = 0xF800;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    // A device without Metal's packed 16-bit pixel formats does not advertise
    // them as render targets and rejects the create to match, which leaves
    // nothing to hold a DC over. That contract is pinned in `expand16`.
    if h.check_device_format(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_RENDERTARGET,
        mtld3d_types::D3DRTYPE_SURFACE,
        D3DFMT_R5G6B5,
    ) != D3D_OK
    {
        assert_ne!(
            h.create_render_target_hr(W, H, D3DFMT_R5G6B5),
            D3D_OK,
            "a 16-bit render target is rejected where the caps deny it"
        );
        return;
    }
    let rt = h.create_lockable_render_target(W, H, D3DFMT_R5G6B5);
    assert_eq!(h.color_fill_hr(&rt, GREEN), D3D_OK, "ColorFill green");

    let last_x = (W - 1).cast_signed();
    let last_y = (H - 1).cast_signed();
    let dc = rt.dc();
    assert_eq!(
        dc.get_pixel(last_x, last_y),
        GREEN_COLORREF,
        "the last pixel of the last row is inside the DIB the DC wraps",
    );
    assert_eq!(
        dc.set_pixel(last_x, last_y, RED_COLORREF),
        RED_COLORREF,
        "SetPixel stores full-scale channels exactly in a 5-6-5 DIB",
    );
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");

    let locked = rt.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() as usize / 2;
    let px = locked.as_u16(pitch_px * H as usize);
    let last_row = pitch_px * (H as usize - 1);
    assert_eq!(
        px[last_row + W as usize - 1],
        RED_565,
        "what GDI drew into the last pixel reached the colour texture",
    );
    assert_eq!(
        px[last_row], GREEN_565,
        "the rest of the last row still holds the fill",
    );
    assert_eq!(px[0], GREEN_565, "so does the first row");
}

#[test]
fn a_back_buffer_sized_lockable_render_target_keeps_its_reported_extent() {
    // A lockable render target is CPU-addressable at the extent D3D9 reports,
    // so it declines `render.scale` even at the back buffer's own size, which
    // is what earns an ordinary render target the scale. Its staging, the
    // read-back that fills it and the upload that pushes it back all address
    // that one extent, and so does every path that binds or fills it, so the
    // probes below sit on the far corner a scaled texture would not reach.
    // Runs at any `render.scale`: the coordinates are the reported ones.
    let h = Harness::new();
    let (w, height) = h.dims();
    let bb = h.render_target(0);
    let rt = h.create_lockable_render_target(w, height, D3DFMT_A8R8G8B8);
    let read = |x: u32, y: u32| {
        let locked = rt.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        locked.as_u32((pitch_px * height) as usize)[(y * pitch_px + x) as usize]
    };

    // Bound as a render target: the clear covers the whole texture, so the far
    // corner carries the clear colour rather than whatever the create left.
    assert_eq!(h.set_render_target(0, &rt), 0, "bind the lockable RT");
    assert_eq!(h.clear_target(GREEN), 0, "clear it green");
    assert_eq!(h.set_render_target(0, &bb), 0, "restore the backbuffer");
    assert_eq!(read(0, 0), GREEN, "the clear reaches the first pixel");
    assert_eq!(
        read(w - 1, height - 1),
        GREEN,
        "the clear reaches the last pixel"
    );

    // Filled on the GPU: the read-back that serves LockRect covers the same
    // extent, pixel for pixel and with no resample in between.
    assert_eq!(h.color_fill_hr(&rt, RED), D3D_OK, "whole-surface ColorFill");
    assert_eq!(read(0, 0), RED, "the fill reaches the first pixel");
    assert_eq!(
        read(w - 1, height - 1),
        RED,
        "the fill reaches the last pixel"
    );

    // Written on the CPU: the unlock upload covers the same extent again, and
    // `GetRenderTargetData` reads the colour texture back to prove it landed.
    {
        let mut locked = rt.lock_rect(0);
        locked.write_u32(&vec![BLUE; (w * height) as usize]);
    }
    assert_eq!(
        read_surface_pixel(&h, &rt, 0, 0),
        BLUE,
        "the upload reaches the first pixel of the colour texture"
    );
    assert_eq!(
        read_surface_pixel(&h, &rt, w - 1, height - 1),
        BLUE,
        "the upload reaches the last pixel of the colour texture"
    );
}

/// A `ColorFill` into an autogen-mipmap render target regenerates its chain.
///
/// The fill lands on level 0 through the GPU path, so the lower levels have
/// to be rebuilt from it before the next sample reads them.
#[test]
fn color_fill_autogen_render_target_regenerates_the_mip_chain() {
    // The runtime owns a D3DUSAGE_AUTOGENMIPMAP texture's mip chain, so a
    // ColorFill into level 0 regenerates it. Seed the chain red through a
    // render into the texture, fill level 0 green, then sample a level the
    // fill never touched: it reads green once the fill regenerates, red while
    // the chain is stale.
    let h = Harness::new();
    let rt = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);
    let backbuffer = h.render_target(0);

    // Seed: render red into the texture. Unbinding it regenerates the chain,
    // so every level holds red before the fill.
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    draw_fill(&h, RED);
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");
    assert_eq!(h.end_scene(), 0, "EndScene");

    assert_eq!(
        h.color_fill_hr(&rt_surface, GREEN),
        D3D_OK,
        "ColorFill green"
    );

    // Sample level 4 (4x4) of the 64x64 chain. MAXMIPLEVEL is the most
    // detailed level the sampler may use, so the draw cannot read the filled
    // level 0 instead.
    assert_eq!(h.clear_target(BLACK), 0, "clear backbuffer black");
    assert_eq!(h.set_texture(0, &rt), 0, "bind the filled texture");
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0, "TSS");
    }
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAXMIPLEVEL, 4),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF TEX1"
    );
    let quad = [
        TexturedVertex {
            x: -0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 1.0,
        },
        TexturedVertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 1.0,
            v: 1.0,
        },
        TexturedVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: WHITE,
            u: 0.0,
            v: 1.0,
        },
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample the regenerated mip"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        center.g > 200 && center.r < 40 && center.b < 40,
        "the small mip carries the fill colour, got {center:?}"
    );
}

#[test]
fn surface_ops_contracts() {
    let h = Harness::new();
    let bb = h.back_buffer(0);
    // ColorFill on a standalone colour surface (the implicit backbuffer) fills
    // its live colour texture and succeeds.
    assert_eq!(
        h.color_fill_hr(&bb, RED),
        D3D_OK,
        "ColorFill on the standalone backbuffer succeeds"
    );
    // GetRenderTargetData / GetFrontBufferData require a D3DPOOL_SYSTEMMEM
    // destination; a DEFAULT-pool backbuffer dst is rejected.
    assert_eq!(
        h.get_render_target_data_hr(&bb, &bb),
        D3DERR_INVALIDCALL,
        "GetRenderTargetData rejects a non-SYSTEMMEM dst",
    );
    assert_eq!(
        h.get_front_buffer_data_hr(&bb),
        D3DERR_INVALIDCALL,
        "GetFrontBufferData rejects a non-SYSTEMMEM dst",
    );
    // D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH and D3DPOOL_DEFAULT offscreen plain
    // surfaces are supported; MANAGED is not.
    assert_eq!(
        h.create_offscreen_plain_surface_hr(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT),
        0,
        "CreateOffscreenPlainSurface(D3DPOOL_DEFAULT) succeeds",
    );
    assert_eq!(
        h.create_offscreen_plain_surface_hr(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED),
        D3DERR_INVALIDCALL,
        "CreateOffscreenPlainSurface(D3DPOOL_MANAGED) is rejected",
    );
}

#[test]
fn create_render_target_default_pool_reports_desc() {
    // CreateRenderTarget yields a D3DPOOL_DEFAULT surface that reports
    // D3DUSAGE_RENDERTARGET; a D3DPOOL_DEFAULT offscreen-plain surface reports
    // no usage. Both are GPU-resident (pool DEFAULT = 0).
    let h = Harness::new();

    let rt = h.create_render_target(64, 48, D3DFMT_A8R8G8B8);
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0, "render-target GetDesc");
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "render target is DEFAULT pool");
    assert_eq!(
        desc.usage, D3DUSAGE_RENDERTARGET,
        "render target reports D3DUSAGE_RENDERTARGET"
    );
    assert_eq!((desc.width, desc.height), (64, 48), "render-target dims");
    assert_eq!(desc.format, D3DFMT_A8R8G8B8, "render-target format");

    let off = h.create_offscreen_plain_surface(64, 48, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let (hr, desc) = off.desc();
    assert_eq!(hr, 0, "offscreen-plain GetDesc");
    assert_eq!(
        desc.pool, D3DPOOL_DEFAULT,
        "offscreen-plain is DEFAULT pool"
    );
    assert_eq!(desc.usage, 0, "offscreen-plain reports no usage flags");
}

#[test]
fn create_render_target_rgba32f_succeeds() {
    // D3DFMT_A32B32G32R32F (128-bit float) is a renderable Metal format
    // (MTLPixelFormatRGBA32Float); CreateRenderTarget must accept it. A NULL
    // return would fault a subsequent SetRenderTarget.
    let h = Harness::new();
    let rt = h.create_render_target(64, 48, D3DFMT_A32B32G32R32F);
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0, "RGBA32F render-target GetDesc");
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "render target is DEFAULT pool");
    assert_eq!(
        desc.usage, D3DUSAGE_RENDERTARGET,
        "reports RENDERTARGET usage"
    );
    assert_eq!(
        desc.format, D3DFMT_A32B32G32R32F,
        "RGBA32F format round-trips"
    );
}

#[test]
fn create_render_target_rgba16f_succeeds() {
    // D3DFMT_A16B16G16R16F is the half-float HDR scene target D3D9 engines
    // ask for (MTLPixelFormatRGBA16Float). CreateRenderTarget must accept it,
    // and GetDesc must report the format back unsubstituted.
    let h = Harness::new();
    let rt = h.create_render_target(64, 48, D3DFMT_A16B16G16R16F);
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0, "RGBA16F render-target GetDesc");
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "render target is DEFAULT pool");
    assert_eq!(
        desc.usage, D3DUSAGE_RENDERTARGET,
        "reports RENDERTARGET usage"
    );
    assert_eq!(
        desc.format, D3DFMT_A16B16G16R16F,
        "RGBA16F format round-trips"
    );
}

/// Decode IEEE-754 binary16 bits into an `f32`.
///
/// Covers zero, subnormals and normals — everything a `[0, 1]` colour
/// read-back can produce.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x03ff) / 1024.0;
    if exponent == 0 {
        sign * mantissa * 2.0_f32.powi(-14)
    } else {
        sign * (1.0 + mantissa) * 2.0_f32.powi(exponent - 15)
    }
}

#[test]
fn render_into_rgba16f_target_round_trips() {
    // Draw a known diffuse colour into a half-float render target and read it
    // back: the create, the colour attachment, and the read-back blit all have
    // to agree on RGBA16Float. This is the shape an engine's HDR scene pass
    // uses before it tone-maps down to the 8-bit backbuffer.
    let h = Harness::new();
    let backbuffer = h.render_target(0);
    let rt = h.create_render_target(64, 64, D3DFMT_A16B16G16R16F);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind half-float RT");
    assert_eq!(h.clear_target(0), 0, "clear RT to 0");
    // 0xFF804020 → R = 0x80/255, G = 0x40/255, B = 0x20/255.
    draw_fill_at_z(&h, 0xFF80_4020, 0.5);
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    let sysmem = h.create_offscreen_plain_surface(64, 64, D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&rt, &sysmem),
        0,
        "GetRenderTargetData half-float RT → SYSTEMMEM"
    );
    let lanes = {
        let locked = sysmem.lock_rect(D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).expect("non-negative pitch");
        // One texel is four halves; sample the middle of the surface.
        let texel = 32 * pitch / 2 + 32 * 4;
        let halves = locked.as_u16(texel + 4);
        [
            f16_to_f32(halves[texel]),
            f16_to_f32(halves[texel + 1]),
            f16_to_f32(halves[texel + 2]),
            f16_to_f32(halves[texel + 3]),
        ]
    };
    let expected = [
        f32::from(0x80u8) / 255.0,
        f32::from(0x40u8) / 255.0,
        f32::from(0x20u8) / 255.0,
        1.0,
    ];
    for (lane, (got, want)) in lanes.into_iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "half-float RT lane {lane} should hold {want}; got {got}"
        );
    }
}

#[test]
fn lock_rect_on_a_non_lockable_render_target_is_rejected() {
    // `CreateRenderTarget` with `Lockable == FALSE` is a GPU-only colour
    // surface with no CPU bytes behind it, so D3D9 answers every `LockRect` of
    // it with `INVALIDCALL`, read-only locks included, and `UnlockRect` with no
    // lock held answers the same. Reading such a target back is what
    // `GetRenderTargetData` is for. The same surface created lockable locks.
    let h = Harness::new();
    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    for flags in [0, D3DLOCK_READONLY] {
        let (hr, bits_null) = rt.lock_rect_probe(flags);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "LockRect(flags={flags:#x}) on a non-lockable render target must be rejected"
        );
        assert!(
            !bits_null,
            "a rejected LockRect must leave the caller's D3DLOCKED_RECT untouched"
        );
        assert_eq!(
            rt.unlock_rect(),
            D3DERR_INVALIDCALL,
            "UnlockRect with no lock held must be rejected"
        );
    }

    let lockable = h.create_lockable_render_target(64, 64, D3DFMT_A8R8G8B8);
    let (hr, bits_null) = lockable.lock_rect_probe(D3DLOCK_READONLY);
    assert_eq!(hr, D3D_OK, "a lockable render target still locks");
    assert!(!bits_null, "an accepted LockRect hands back a pointer");
    assert_eq!(
        lockable.unlock_rect(),
        D3D_OK,
        "UnlockRect closes the lockable render target's lock"
    );
}

#[test]
fn lock_rect_on_a_half_float_render_target_reads_at_its_own_pitch() {
    // A lockable render target's staging is a host-visible store like any
    // other: it is sized, filled from the GPU and reported at the row pitch its
    // own format asks for. A half-float target is eight bytes per texel, so a
    // store laid out at four bytes per texel reports half the stride its rows
    // are really at and asks the read-back for a copy narrower than one row of
    // the source.
    const W: u32 = 64;
    const H: u32 = 64;
    const LANES_PER_TEXEL: usize = 4;
    // 0xFF804020: R = 0x80/255, G = 0x40/255, B = 0x20/255, A = 1.
    const FILL: u32 = 0xFF80_4020;
    let h = Harness::new();
    let backbuffer = h.render_target(0);
    let rt = h.create_lockable_render_target(W, H, D3DFMT_A16B16G16R16F);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind half-float RT");
    assert_eq!(h.clear_target(FILL), 0, "clear the half-float RT");
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    let locked = rt.lock_rect(D3DLOCK_READONLY);
    let pitch = usize::try_from(locked.pitch()).expect("non-negative pitch");
    assert_eq!(
        pitch,
        W as usize * 8,
        "a half-float row is eight bytes per texel wide",
    );
    let lanes_per_row = pitch / 2;
    let lanes = locked.as_u16(lanes_per_row * H as usize);
    let expected = [
        f32::from(0x80u8) / 255.0,
        f32::from(0x40u8) / 255.0,
        f32::from(0x20u8) / 255.0,
        1.0,
    ];
    for (label, texel) in [
        ("first texel of the first row", 0),
        (
            "last texel of the last row",
            lanes_per_row * (H as usize - 1) + (W as usize - 1) * LANES_PER_TEXEL,
        ),
    ] {
        for (lane, want) in expected.into_iter().enumerate() {
            let got = f16_to_f32(lanes[texel + lane]);
            assert!(
                (got - want).abs() < 0.01,
                "{label} lane {lane} should hold {want}; got {got}",
            );
        }
    }
}

#[test]
fn render_to_default_pool_target_round_trips() {
    // A DEFAULT-pool render target can be bound, drawn into, and is then a valid
    // GetRenderTargetData source into a SYSTEMMEM surface — i.e. create_color_target
    // produces a real, renderable, readable Metal texture that SetRenderTarget and
    // the readback blit both resolve via metal_color_handle. Metal validation is on
    // under `make test`, so a malformed RT attachment would abort the draw.
    //
    // Nothing inside the frame samples the offscreen RT, so only the read-back
    // note keeps its colour store; the pixel assert at the end pins that.
    const TEAL: u32 = 0xFF00_8080;
    let h = Harness::new();
    // Capture the implicit backbuffer so we can restore RT0 before `rt` drops.
    let bb = h.render_target(0);

    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind DEFAULT RT");
    assert_eq!(h.clear_target(TEAL), 0, "clear RT teal");
    assert_eq!(h.clear_texture(0), 0, "no texture for the fill draw");
    // Lighting defaults on; a lit vertex with no lights would come out
    // black instead of its diffuse colour.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    // Emit the diffuse colour directly so the fill does not depend on a bound
    // texture.
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0, "TSS");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    let fill = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: TEAL,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: TEAL,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: TEAL,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill),
        0,
        "RT fill draw"
    );
    // Restore the backbuffer: finalises the RT pass and avoids the device
    // retaining a dangling pointer to `rt` after it drops.
    assert_eq!(h.set_render_target(0, &bb), 0, "restore backbuffer RT");

    assert_eq!(
        read_surface_pixel(&h, &rt, 32, 32),
        TEAL,
        "the drawn-into DEFAULT RT reads back its fill colour"
    );
}

#[test]
fn stretch_rect_between_default_pool_targets() {
    // 1:1 same-format StretchRect between two DEFAULT render targets: a
    // standalone colour surface works as both src and dst, and a Clear issued
    // on the source right before the copy lands first, as D3D9 ordered it.
    let h = Harness::new();
    let bb = h.render_target(0);

    let src = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let dst = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &src), 0, "bind src");
    assert_eq!(h.clear_target(GREEN), 0, "clear src green");
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_NONE),
        0,
        "1:1 same-format StretchRect between DEFAULT RTs",
    );
    assert_eq!(h.set_render_target(0, &bb), 0, "restore backbuffer");
    assert_eq!(
        read_surface_pixel(&h, &dst, 32, 32),
        GREEN,
        "the copy reads the source after its pending clear"
    );
}

/// Read one pixel of `surface` as `0xAARRGGBB` through `GetRenderTargetData`.
fn read_surface_pixel(h: &Harness, surface: &mtld3d_tests::Surface<'_>, x: u32, y: u32) -> u32 {
    let (hr, desc) = surface.desc();
    assert_eq!(hr, 0, "GetDesc for read_surface_pixel");
    let sysmem = h.create_offscreen_plain_surface(
        desc.width,
        desc.height,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(surface, &sysmem),
        0,
        "GetRenderTargetData for read_surface_pixel"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let idx = (y * pitch_px + x) as usize;
    locked.as_u32(idx + 1)[idx]
}

/// Draw a full-target triangle in `color` through the diffuse channel.
fn draw_fill(h: &Harness, color: u32) {
    // Lighting defaults on and would replace the diffuse with black.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture for the fill draw");
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0, "TSS");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    let fill = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill),
        0,
        "fill draw"
    );
}

#[test]
fn stretch_rect_from_rendered_target_survives_present() {
    // Render into an offscreen RT, copy it to the backbuffer, Present. Nothing
    // samples the RT, so its last-use store is the optimiser's to elide; the
    // copy reads it from device memory after the pass, so the store must
    // stay. Observed on the next frame, which only reads the persistent
    // backbuffer back.
    let h = Harness::new();
    let bb = h.render_target(0);
    let rt = h.create_render_target(640, 480, D3DFMT_A8R8G8B8);

    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.set_render_target(0, &rt), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    draw_fill(&h, GREEN);
    assert_eq!(h.set_render_target(0, &bb), 0, "restore backbuffer");
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(
        h.stretch_rect(&rt, &bb, D3DTEXF_NONE),
        0,
        "StretchRect RT -> backbuffer"
    );
    assert_eq!(h.present(), 0, "Present");

    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the backbuffer holds the RT's rendered content after Present"
    );
}

#[test]
fn stretch_rect_into_a_target_with_a_pending_clear_keeps_the_copy() {
    // Clear(backbuffer) with no pass open, then copy a rendered RT into the
    // backbuffer: D3D9 ordered the clear first, so the copy wins.
    let h = Harness::new();
    let bb = h.render_target(0);
    let rt = h.create_render_target(640, 480, D3DFMT_A8R8G8B8);

    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.set_render_target(0, &rt), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    draw_fill(&h, GREEN);
    assert_eq!(h.set_render_target(0, &bb), 0, "restore backbuffer");
    assert_eq!(h.clear_target(BLACK), 0, "clear backbuffer black");
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(
        h.stretch_rect(&rt, &bb, D3DTEXF_NONE),
        0,
        "StretchRect RT -> backbuffer"
    );

    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the copy lands after the backbuffer clear"
    );
}

#[test]
fn get_render_target_data_reads_backbuffer() {
    // The conformance read-back chain: render a known colour, then
    // GetRenderTarget(0) → CreateOffscreenPlainSurface(SYSTEMMEM) →
    // GetRenderTargetData → LockRect, and confirm the locked pixel decodes to
    // the rendered colour. Distinct R/G/B in the fill colour catches any channel
    // swizzle in the blit/lock path. (This is the chain `Harness::read_pixel`
    // itself runs; here we drive it explicitly to assert the lock layout.)
    const ORANGE: u32 = 0xFFFF_8000;
    let h = Harness::new();
    assert_eq!(h.clear_target(ORANGE), 0, "clear backbuffer orange");
    assert_eq!(h.present(), 0, "present");

    let bb = h.render_target(0);
    let (hr, desc) = bb.desc();
    assert_eq!(hr, 0, "backbuffer GetDesc");
    let sysmem = h.create_offscreen_plain_surface(
        desc.width,
        desc.height,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(&bb, &sysmem),
        0,
        "GetRenderTargetData backbuffer → SYSTEMMEM",
    );

    let (x, y) = (320u32, 240u32);
    let pixel = {
        let locked = sysmem.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        let idx = (y * pitch_px + x) as usize;
        locked.as_u32(idx + 1)[idx]
    };

    // The locked pixel decodes to the rendered orange (R≈255, G≈128, B≈0).
    let c = Rgba8::from_pixel(pixel);
    assert!(
        c.r > 200 && c.g > 100 && c.g < 160 && c.b < 40,
        "read-back decodes to orange, got {c:?}",
    );
}

#[test]
fn get_render_target_data_writes_rows_at_the_reported_pitch() {
    // An odd width in a 16-bit format is where the tight row stride and the
    // one `LockRect` reports part company: 33 R5G6B5 texels are 66 bytes, and
    // a linear system-memory surface reports 68, the next four-byte boundary.
    // The read-back writes its rows at the reported stride, so row n starts
    // where the lock reads it and the last row's tail is written too. At the
    // tight stride row n would land 2n bytes early (row 1 ending in row 2's
    // first texel) and the last row's last texels would keep whatever the
    // backing already held.
    const W: u32 = 33;
    const H: u32 = 4;
    // `W * 2` rounded up to the next four-byte boundary, in bytes and in lanes.
    const PITCH: usize = 68;
    const LANES: usize = PITCH / 2 * H as usize;
    // BLUE and GREEN in B5G6R5 (`B[0..5] G[5..11] R[11..16]`).
    const BLUE_565: u16 = 0x001F;
    const GREEN_565: u16 = 0x07E0;
    const SENTINEL: u16 = 0xAAAA;

    let h = Harness::new();
    // A device without Metal's packed 16-bit pixel formats does not advertise
    // them as render targets and rejects the create to match, which leaves
    // nothing to read back. That contract is pinned in `expand16`.
    if h.check_device_format(
        D3DFMT_X8R8G8B8,
        D3DUSAGE_RENDERTARGET,
        mtld3d_types::D3DRTYPE_SURFACE,
        D3DFMT_R5G6B5,
    ) != D3D_OK
    {
        assert_ne!(
            h.create_render_target_hr(W, H, D3DFMT_R5G6B5),
            D3D_OK,
            "a 16-bit render target is rejected where the caps deny it"
        );
        return;
    }
    let backbuffer = h.render_target(0);
    let rt = h.create_render_target(W, H, D3DFMT_R5G6B5);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind the R5G6B5 target");
    assert_eq!(h.clear_target(BLUE), 0, "clear the R5G6B5 target blue");
    let stripe = D3DRECT {
        x1: 0,
        y1: 1,
        x2: 33,
        y2: 2,
    };
    assert_eq!(h.clear_target_rects(GREEN, &[stripe]), 0, "row 1 green");
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    let sysmem = h.create_offscreen_plain_surface(W, H, D3DFMT_R5G6B5, D3DPOOL_SYSTEMMEM);
    // Seed the backing so a byte the read-back never writes reads back as
    // something no clear colour produces; the create leaves it uninitialised.
    {
        let mut seed = sysmem.lock_rect(0);
        assert_eq!(
            usize::try_from(seed.pitch()).expect("non-negative pitch"),
            PITCH,
            "LockRect reports the four-byte-rounded pitch"
        );
        seed.write(&[SENTINEL; LANES]);
    }
    assert_eq!(
        h.get_render_target_data_hr(&rt, &sysmem),
        0,
        "GetRenderTargetData R5G6B5 RT → SYSTEMMEM",
    );

    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let lanes = locked.as_u16(LANES);
    let texel = |x: usize, y: usize| lanes[y * (PITCH / 2) + x];
    assert_eq!(texel(0, 1), GREEN_565, "row 1 starts green");
    assert_eq!(
        texel(32, 1),
        GREEN_565,
        "row 1 ends green, not row 2's blue"
    );
    assert_eq!(
        texel(32, 3),
        BLUE_565,
        "the last row's last texel is written"
    );
}

#[test]
fn get_render_target_data_fills_a_system_memory_texture_level() {
    // D3D9 takes any system-memory surface as the destination, and a title
    // that screenshots or feeds a reflection reads the back buffer into a
    // level of a D3DPOOL_SYSTEMMEM texture rather than into an offscreen
    // plain surface. Both CPU-only pools qualify.
    const TEAL: u32 = 0xFF00_8080;
    let h = Harness::new();
    assert_eq!(h.clear_target(TEAL), 0, "clear backbuffer teal");
    assert_eq!(h.present(), 0, "present");

    let bb = h.render_target(0);
    let (hr, desc) = bb.desc();
    assert_eq!(hr, 0, "backbuffer GetDesc");

    for (pool, name) in [
        (D3DPOOL_SYSTEMMEM, "D3DPOOL_SYSTEMMEM"),
        (D3DPOOL_SCRATCH, "D3DPOOL_SCRATCH"),
    ] {
        let texture = h.create_texture(desc.width, desc.height, 1, 0, D3DFMT_A8R8G8B8, pool);
        let level = texture.surface_level(0);
        assert_eq!(
            h.get_render_target_data_hr(&bb, &level),
            0,
            "GetRenderTargetData backbuffer → {name} texture level",
        );
        let pixel = {
            let locked = level.lock_rect(D3DLOCK_READONLY);
            let pitch_px = locked.pitch().cast_unsigned() / 4;
            let idx = (240 * pitch_px + 320) as usize;
            locked.as_u32(idx + 1)[idx]
        };
        // The locked pixel decodes to the cleared teal (R=0, G=B≈128).
        let c = Rgba8::from_pixel(pixel);
        assert!(
            c.r < 40 && c.g > 100 && c.g < 160 && c.b > 100 && c.b < 160,
            "{name} level decodes to teal, got {c:?}",
        );
    }
}

#[test]
fn get_front_buffer_data_fills_a_system_memory_texture_level() {
    // Same destination rule on the front-buffer read; the source is the
    // presented image rather than a caller-named render target.
    const PURPLE: u32 = 0xFF80_0080;
    let h = Harness::new();
    assert_eq!(h.clear_target(PURPLE), 0, "clear backbuffer purple");
    assert_eq!(h.present(), 0, "present");

    let (hr, desc) = h.render_target(0).desc();
    assert_eq!(hr, 0, "backbuffer GetDesc");
    let texture = h.create_texture(
        desc.width,
        desc.height,
        1,
        0,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    let level = texture.surface_level(0);
    assert_eq!(
        h.get_front_buffer_data_hr(&level),
        0,
        "GetFrontBufferData → SYSTEMMEM texture level",
    );
    let pixel = {
        let locked = level.lock_rect(D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() / 4;
        let idx = (240 * pitch_px + 320) as usize;
        locked.as_u32(idx + 1)[idx]
    };
    // The locked pixel decodes to the cleared purple (R=B≈128, G=0).
    let c = Rgba8::from_pixel(pixel);
    assert!(
        c.r > 100 && c.r < 160 && c.g < 40 && c.b > 100 && c.b < 160,
        "front-buffer level decodes to purple, got {c:?}",
    );
}

#[test]
fn readback_rejects_a_destination_that_is_not_the_source_in_system_memory() {
    // The destination rules are D3D9's: a system-memory surface with the
    // source's extent and format. A level of a GPU-resident texture, and a
    // system-memory destination of another size, are both INVALIDCALL rather
    // than a copy of whatever fits.
    let h = Harness::new();
    let bb = h.render_target(0);
    let (hr, desc) = bb.desc();
    assert_eq!(hr, 0, "backbuffer GetDesc");

    let managed = h.create_texture(
        desc.width,
        desc.height,
        1,
        0,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    assert_eq!(
        h.get_render_target_data_hr(&bb, &managed.surface_level(0)),
        D3DERR_INVALIDCALL,
        "a D3DPOOL_MANAGED level is not a system-memory destination",
    );

    let small = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&bb, &small.surface_level(0)),
        D3DERR_INVALIDCALL,
        "a smaller destination is rejected, not filled with a scaled copy",
    );
    assert_eq!(
        h.get_front_buffer_data_hr(&small.surface_level(0)),
        D3DERR_INVALIDCALL,
        "GetFrontBufferData applies the same extent rule",
    );

    let wrong_format = h.create_texture(
        desc.width,
        desc.height,
        1,
        0,
        D3DFMT_R5G6B5,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(&bb, &wrong_format.surface_level(0)),
        D3DERR_INVALIDCALL,
        "a destination with another byte layout is rejected, not converted",
    );
}

#[test]
fn set_render_target_resets_viewport_and_scissor() {
    // D3D9: SetRenderTarget(0, rt) snaps the viewport and scissor rect to the
    // new target's full dimensions, overriding any rect set beforehand. The
    // harness device is 640x480.
    let h = Harness::new();

    let default_scissor = h.scissor_rect();
    assert_eq!(
        (default_scissor.x2, default_scissor.y2),
        (640, 480),
        "default scissor covers the full backbuffer",
    );

    let rt = h.create_texture(
        128,
        128,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let rt_surface = rt.surface_level(0);

    // Bind the 128x128 RT: viewport + scissor follow it.
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    let vp = h.viewport();
    assert_eq!((vp.width, vp.height), (128, 128), "viewport follows RT");
    let sc = h.scissor_rect();
    assert_eq!(
        (sc.x1, sc.y1, sc.x2, sc.y2),
        (0, 0, 128, 128),
        "scissor follows RT",
    );

    // A custom viewport + scissor, then a re-bind of the same RT, resets both.
    assert_eq!(
        h.set_viewport(&D3DVIEWPORT9 {
            x: 10,
            y: 20,
            width: 30,
            height: 40,
            min_z: 0.25,
            max_z: 0.75,
        }),
        0,
        "custom viewport",
    );
    assert_eq!(
        h.set_scissor_rect(&D3DRECT {
            x1: 50,
            y1: 60,
            x2: 70,
            y2: 80,
        }),
        0,
        "custom scissor",
    );
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "re-bind RT");
    let vp = h.viewport();
    assert_eq!(
        (vp.x, vp.y, vp.width, vp.height),
        (0, 0, 128, 128),
        "re-bind resets the custom viewport",
    );
    let sc = h.scissor_rect();
    assert_eq!(
        (sc.x1, sc.y1, sc.x2, sc.y2),
        (0, 0, 128, 128),
        "re-bind resets the custom scissor",
    );
}

#[test]
fn sample_float_texture_into_float_rt_round_trips() {
    // Render a float-texture sample into a custom A32B32G32R32F render target,
    // then read it back. The sample, the float texture, and the float-RT
    // readback each work in isolation; this test guards their combination.
    const W: u32 = 200;

    let h = Harness::new();
    let tex = h.create_texture(W, W, 1, 0, D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED);
    {
        let lr = tex.lock_rect(0, 0);
        let pitch = usize::try_from(lr.pitch()).expect("non-negative pitch");
        let base = lr.bits_ptr();
        let dim = f32::from(u16::try_from(W).expect("W < 65536"));
        for y in 0..W {
            let fy = f32::from(u16::try_from(y).expect("y < 65536")) / dim;
            for x in 0..W {
                let fx = f32::from(u16::try_from(x).expect("x < 65536")) / dim;
                let px = [fx, fy, 0.0_f32, 1.0_f32];
                let off = y as usize * pitch + x as usize * 16;
                // SAFETY: `off` is in-bounds of the locked region (y<W, x<W, pitch>=W*16).
                let dst = unsafe { base.add(off) };
                // SAFETY: `dst` is valid for the 16 bytes of one float4 texel.
                unsafe { core::ptr::copy_nonoverlapping(px.as_ptr().cast::<u8>(), dst, 16) };
            }
        }
    }

    let backbuffer = h.render_target(0);
    let rt = h.create_render_target(256, 256, D3DFMT_A32B32G32R32F);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind float RT");
    assert_eq!(h.clear_target(0), 0, "clear RT to 0");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.set_texture(0, &tex), 0, "bind float texture");
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_COLORARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);

    let v = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0,
        u: 0.5,
        v: 0.25,
    };
    let quad = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "RT sample draw"
    );
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    let sysmem =
        h.create_offscreen_plain_surface(256, 256, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&rt, &sysmem),
        0,
        "GetRenderTargetData float RT → SYSTEMMEM"
    );
    let (cx, cy) = (128usize, 128usize);
    let (r, g) = {
        let locked = sysmem.lock_rect(D3DLOCK_READONLY);
        let pitch_u32 = locked.pitch().cast_unsigned() as usize / 4;
        let idx = cy * pitch_u32 + cx * 4;
        let px = locked.as_u32(idx + 4);
        (f32::from_bits(px[idx]), f32::from_bits(px[idx + 1]))
    };
    assert!(
        (r - 0.5).abs() < 0.05 && (g - 0.25).abs() < 0.05,
        "float RT sample of (0.5,0.25) should be ~(0.5,0.25); got ({r},{g})"
    );
}

/// Draw a full-cover triangle in `color` at clip-space depth `z`.
fn draw_fill_at_z(h: &Harness, color: u32, z: f32) {
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_LIGHTING, 0),
        0,
        "lighting off"
    );
    assert_eq!(h.clear_texture(0), 0, "no texture for the fill draw");
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_DIFFUSE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0, "TSS");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    let fill = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z,
            color,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z,
            color,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z,
            color,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill),
        0,
        "fill draw"
    );
}

#[test]
fn depth_survives_a_mid_frame_readback_flush() {
    // A readback taken between two depth-tested draw groups forces a mid-frame
    // flush (NO_PRESENT). The depth surface the first group wrote must survive
    // it: the far second group is gated LESS against the primed near depth and
    // must fail. Before the fix the flush elided the depth store (Rule B) and
    // reset first-use (Rule A), so the far group tested against discarded
    // depth. The colour side is deterministic here: the near group's green is
    // the surviving pixel iff depth held.
    let h = Harness::with_depth();
    assert!(h.pump(), "WM_QUIT before render");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0,
        "clear colour + depth",
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "z on");

    // Near group: green, writes depth 0.25.
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0, "zwrite on");
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL),
        0,
        "zfunc lessequal"
    );
    draw_fill_at_z(&h, GREEN, 0.25);

    // Force a mid-frame flush between the groups (readback of the backbuffer).
    let _ = h.read_pixel(0, 0);

    // Far group: red at 0.75, gated LESS. 0.75 < 0.25 is false, so it must
    // fail against the primed depth and leave green in place.
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0, "zwrite off");
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS),
        0,
        "zfunc less"
    );
    draw_fill_at_z(&h, RED, 0.75);

    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(h.present(), 0, "Present");
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the far group failed depth against the primed near depth that survived the flush",
    );
}

/// Assert each RGB channel of `got` is within `tolerance` of `expected` (alpha ignored).
fn assert_rgb_close(got: u32, expected: u32, tolerance: i32, context: &str) {
    let channel = |c: u32, shift: u32| i32::try_from((c >> shift) & 0xff).unwrap_or(0);
    let close = [16, 8, 0]
        .iter()
        .all(|&shift| (channel(got, shift) - channel(expected, shift)).abs() <= tolerance);
    assert!(
        close,
        "{context}: got {got:#010x}, expected {expected:#08x} (tolerance {tolerance})"
    );
}

#[test]
fn stretch_rect_decodes_packed_yuv_into_the_backbuffer() {
    // A 4x1 DEFAULT offscreen-plain YUV surface holding the macropixel under
    // test plus a neutral filler, StretchRect'd with POINT onto the 640x480
    // backbuffer (a scaling blit, so the render quad does the decode). The
    // left half of the target shows pixel 0, the right half pixel 1. Expected
    // colours are the reference values desktop drivers produce for these
    // macropixels; drivers disagree on the exact Y'CbCr convention, hence the
    // tolerance of 18.
    let h = Harness::new();
    let bb = h.render_target(0);
    let cases: [(u32, &str, u32, u32, u32); 8] = [
        (D3DFMT_UYVY, "UYVY", 0x4cff_4c54, 0x00ff_0000, 0x00ff_0000),
        (D3DFMT_UYVY, "UYVY", 0x0080_0080, 0x0000_0000, 0x0000_0000),
        (D3DFMT_UYVY, "UYVY", 0xff80_ff80, 0x00ff_ffff, 0x00ff_ffff),
        (D3DFMT_UYVY, "UYVY", 0xff00_0000, 0x0000_8700, 0x004b_ff1c),
        (D3DFMT_YUY2, "YUY2", 0x4cff_4c54, 0x000b_8b00, 0x00b6_ffa3),
        (D3DFMT_YUY2, "YUY2", 0x0080_0080, 0x0000_ff00, 0x0000_ff00),
        (D3DFMT_YUY2, "YUY2", 0xff80_ff80, 0x00ff_00ff, 0x00ff_00ff),
        (D3DFMT_YUY2, "YUY2", 0x1c6b_1cff, 0x006d_ff45, 0x0000_d500),
    ];
    for (format, name, input, left, right) in cases {
        let surface = h.create_offscreen_plain_surface(4, 1, format, D3DPOOL_DEFAULT);
        {
            let mut locked = surface.lock_rect(0);
            locked.write(&[input, 0x0080_0080u32]);
        }
        assert_eq!(
            h.clear_target(BLACK),
            0,
            "clear before {name} {input:#010x}"
        );
        assert_eq!(
            h.stretch_rect(&surface, &bb, D3DTEXF_POINT),
            D3D_OK,
            "StretchRect {name} {input:#010x} onto the backbuffer"
        );
        // Each source pixel covers a 160 px band of the target; probe the
        // middle of the first two so a reading stays clear of the colour
        // boundary between them, where a resolved frame drifts further than
        // the convention tolerance allows.
        assert_rgb_close(
            h.read_pixel(80, 240),
            left,
            18,
            &format!("{name} {input:#010x} pixel 0"),
        );
        assert_rgb_close(
            h.read_pixel(240, 240),
            right,
            18,
            &format!("{name} {input:#010x} pixel 1"),
        );
    }
}

#[test]
fn stretch_rect_decodes_packed_yuv_into_an_offscreen_plain_surface() {
    // A 1:1 YUV -> X8R8G8B8 copy into an offscreen-plain destination has no
    // GPU path (the 1:1 blit cannot convert and the quad needs a render
    // target), so the CPU converter decodes the macropixels; a later lock of
    // the destination reads the converted pixels.
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 1, D3DFMT_UYVY, D3DPOOL_DEFAULT);
    {
        let mut locked = src.lock_rect(0);
        // Pixels 0/1: red; pixels 2/3: white.
        locked.write(&[0x4cff_4c54u32, 0xff80_ff80]);
    }
    let dst = h.create_offscreen_plain_surface(4, 1, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_NONE),
        D3D_OK,
        "1:1 UYVY -> X8R8G8B8 into an offscreen-plain surface"
    );
    let locked = dst.lock_rect(D3DLOCK_READONLY);
    let px = locked.as_u32(4);
    assert_rgb_close(px[0], 0x00ff_0000, 18, "pixel 0");
    assert_rgb_close(px[1], 0x00ff_0000, 18, "pixel 1");
    assert_rgb_close(px[2], 0x00ff_ffff, 18, "pixel 2");
    assert_rgb_close(px[3], 0x00ff_ffff, 18, "pixel 3");
}

/// A converting `StretchRect` out of a source level larger than the destination.
///
/// The CPU converter takes a source rectangle and a destination origin, so the
/// extent it writes has to come from what both levels hold rather than from the
/// source alone: a 4x1 source feeding a 2x1 destination is the pair whose
/// unclipped extent would run the row loop past the destination staging.
#[test]
fn stretch_rect_converts_a_sub_rect_into_a_smaller_offscreen_surface() {
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 1, D3DFMT_UYVY, D3DPOOL_DEFAULT);
    {
        let mut locked = src.lock_rect(0);
        // Pixels 0/1: red; pixels 2/3: white.
        locked.write(&[0x4cff_4c54u32, 0xff80_ff80]);
    }
    let dst = h.create_offscreen_plain_surface(2, 1, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT);
    let rect = D3DRECT {
        x1: 2,
        y1: 0,
        x2: 4,
        y2: 1,
    };
    assert_eq!(
        h.stretch_rect_region_hr(&src, &rect, &dst, D3DTEXF_NONE),
        D3D_OK,
        "UYVY sub-rect -> X8R8G8B8 into a smaller offscreen-plain surface"
    );
    let locked = dst.lock_rect(D3DLOCK_READONLY);
    let px = locked.as_u32(2);
    assert_rgb_close(px[0], 0x00ff_ffff, 18, "pixel 0");
    assert_rgb_close(px[1], 0x00ff_ffff, 18, "pixel 1");
}

#[test]
fn stretch_rect_converts_r5g6b5_into_x8r8g8b8() {
    // CheckDeviceFormatConversion(R5G6B5, X8R8G8B8) answers S_OK; the scaling
    // render quad is the path that makes it true. Each 16-bit source pixel
    // covers a 160-pixel band of the backbuffer.
    let h = Harness::new();
    let bb = h.render_target(0);
    let src = h.create_offscreen_plain_surface(4, 1, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    {
        let mut locked = src.lock_rect(0);
        locked.write(&[0xf800u16, 0x07e0, 0x001f, 0xffff]);
    }
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.stretch_rect(&src, &bb, D3DTEXF_POINT),
        D3D_OK,
        "R5G6B5 -> X8R8G8B8 scaling StretchRect"
    );
    assert_eq!(h.read_pixel(80, 240), RED, "pixel 0 is red");
    assert_eq!(h.read_pixel(240, 240), GREEN, "pixel 1 is green");
    assert_eq!(h.read_pixel(400, 240), BLUE, "pixel 2 is blue");
    assert_eq!(h.read_pixel(560, 240), WHITE, "pixel 3 is white");
}

/// A depth texture with a mip chain: a level binds as the depth attachment.
///
/// An engine's depth pyramid renders depth into successive levels through
/// `GetSurfaceLevel(n)`. Level 1 of a 1280x960 chain is the back buffer's size,
/// so it is bound beside the back buffer and a farther draw after a nearer one
/// must lose the depth test in that level.
#[test]
fn depth_texture_mip_level_binds_as_depth_attachment() {
    let h = Harness::new();
    let depth_tex = h.create_texture(
        1280,
        960,
        2,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(depth_tex.level_count(), 2, "both levels created");
    let (hr, desc) = depth_tex.level_desc(1);
    assert_eq!(hr, 0, "GetLevelDesc(1)");
    assert_eq!(
        (desc.width, desc.height),
        (640, 480),
        "level 1 is half the base level"
    );

    let depth_surf = depth_tex.surface_level(1);
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "color target");
    assert_eq!(
        h.set_depth_stencil_surface(&depth_surf),
        0,
        "bind level 1 as the depth attachment"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, mtld3d_types::D3DCMP_LESSEQUAL),
        0
    );
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    // Lighting defaults on and the vertices carry no normal, which would
    // light every draw black; the test reads the vertex colour.
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_LIGHTING, 0), 0);

    let tri = |z: f32, color: u32| {
        [
            PosColorVertex {
                x: -1.0,
                y: 3.0,
                z,
                color,
            },
            PosColorVertex {
                x: 3.0,
                y: -1.0,
                z,
                color,
            },
            PosColorVertex {
                x: -1.0,
                y: -1.0,
                z,
                color,
            },
        ]
    };
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri(0.3, RED)),
        0,
        "near draw"
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri(0.7, GREEN)),
        0,
        "far draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        center.r > 200 && center.g < 40,
        "the far draw loses the depth test in level 1, got {center:?}"
    );
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
}

#[test]
fn resz_resolve_copies_bound_depth_into_the_stage0_texture() {
    // The RESZ hack: SetRenderState(POINTSIZE, 0x7fa05000) resolves the
    // bound depth-stencil into the depth texture at stage 0. Engines on the
    // matching vendor path use it as their only depth hand-off, so support
    // is probed via the RESZ pseudo-format and must answer D3D_OK.
    let h = Harness::new();
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            mtld3d_types::D3DUSAGE_RENDERTARGET,
            mtld3d_types::D3DRTYPE_SURFACE,
            mtld3d_types::D3DFMT_RESZ,
        ),
        0,
        "RESZ pseudo-format probe answers available"
    );

    let depth_src = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth_dst = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);

    // Pass 1: write depth 0.25 into the source through the FF pipeline.
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(h.set_depth_stencil_surface(&depth_src.surface_level(0)), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let occluder = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &occluder),
        0,
        "depth write draw"
    );

    // The resolve: destination at stage 0, then the magic POINTSIZE write.
    assert_eq!(h.set_texture(0, &depth_dst), 0, "bind resolve destination");
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_POINTSIZE, 0x7fa0_5000),
        0
    );

    // Pass 2: sample the DESTINATION; only the resolve can have filled it.
    let ps = h.create_pixel_shader(&PS_SAMPLE_DEPTH);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    assert_eq!(h.clear_depth_stencil_surface(), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let quad = [
        v(-0.5, 0.5, 0.0, 0.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(-0.5, -0.5, 0.0, 1.0),
        v(0.5, 0.5, 1.0, 0.0),
        v(0.5, -0.5, 1.0, 1.0),
        v(-0.5, -0.5, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(BLACK), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "sample the resolved copy"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let center = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        (48..=90).contains(&center.r) && (48..=90).contains(&center.g),
        "the resolved depth (0.25) samples back as dark gray, got {center:?}"
    );

    assert_eq!(h.clear_pixel_shader(), 0);
    assert_eq!(h.clear_texture(0), 0);
}

/// `ps_3_0`: sample s0 at the interpolated texcoord, write it to oDepth.
///
/// `dcl_2d s0; dcl_texcoord0 v0; texld r0, v0, s0; mov oC0, c0; mov
/// oDepth, r0.x;` — the depth-restore shape a deferred engine uses to
/// copy scene depth into the persistent depth buffer for its late
/// (sprite/alpha) pass. `oDepth` is register type `DEPTHOUT` (9).
#[rustfmt::skip]
const PS_RESTORE_DEPTH: [u32; 21] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
    0x0200_001F, 0x8000_0005, 0x900F_0000,              // dcl_texcoord0 v0
    0x0500_0051, 0xA00F_0000,                           // def c0,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, //   0, 0, 0, 0
    0x0300_0042, 0x800F_0000, 0x90E4_0000, 0xA0E4_0800, // texld r0, v0, s0
    0x0200_0001, 0x9001_0800, 0x8000_0000,              // mov oDepth, r0.x
    0x0000_FFFF,                                        // end
];

#[test]
fn odepth_restore_feeds_a_later_sprite_z_test() {
    // The deferred late-pass depth hand-off: scene depth lives in an INTZ
    // texture; a full-screen draw with color writes OFF, stencil REPLACE,
    // ZFUNC=ALWAYS and z-write ON copies it into the bound depth buffer by
    // writing oDepth per pixel; sprites drawn afterwards z-test LESSEQUAL
    // against the restored values. A sprite behind restored near geometry
    // must be occluded; over restored far background it must show.
    let h = Harness::with_depth();
    let scene_depth = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);
    let implicit = h.depth_stencil_surface().expect("implicit depth");

    // Pass 1: scene depth into the INTZ texture — cleared 1.0, an occluder
    // quad at 0.25 over the LEFT half.
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(
        h.set_depth_stencil_surface(&scene_depth.surface_level(0)),
        0
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let left_occluder = [
        PosColorVertex {
            x: -1.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &left_occluder),
        0,
        "scene occluder"
    );

    // Pass 2: back to the implicit depth buffer, cleared to 0.1 (a value
    // that hides the sprite everywhere if the restore does not land), then
    // the restore draw: full-screen, color writes off, stencil REPLACE,
    // ZFUNC=ALWAYS + z-write, PS samples the INTZ and writes oDepth.
    assert_eq!(h.set_depth_stencil_surface(&implicit), 0);
    assert_eq!(h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, RED, 0.1, 0), 0);
    let ps = h.create_pixel_shader(&PS_RESTORE_DEPTH);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    assert_eq!(h.set_texture(0, &scene_depth), 0, "bind INTZ");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILENABLE, 1), 0);
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_STENCILFUNC, D3DCMP_ALWAYS),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILREF, 7), 0);
    assert_eq!(
        h.set_render_state(
            mtld3d_types::D3DRS_STENCILPASS,
            mtld3d_types::D3DSTENCILOP_REPLACE
        ),
        0
    );
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let full = [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full),
        0,
        "depth-restore draw"
    );

    // Pass 3: color writes back on, stencil off, a green sprite quad at
    // z=0.6 z-tested LESSEQUAL against the restored depth.
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0xF),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILENABLE, 0), 0);
    assert_eq!(h.clear_pixel_shader(), 0);
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    let sprite = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.6,
            color: GREEN,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.6,
            color: GREEN,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.6,
            color: GREEN,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &sprite),
        0,
        "late sprite draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    // Left half: restored 0.25 occludes the 0.6 sprite → red clear shows.
    // Right half: restored 1.0 lets it through → green. A stale 0.1 buffer
    // (restore never landed) reads red on BOTH sides.
    assert_eq!(
        h.read_pixel(160, 240),
        RED,
        "sprite occluded where the restore copied near depth"
    );
    assert_eq!(
        h.read_pixel(480, 240),
        GREEN,
        "sprite visible where the restore copied far depth"
    );
}

/// `ps_3_0`: sample s0 at texcoord, write it to BOTH oC0 and oDepth.
#[rustfmt::skip]
const PS_RESTORE_DEPTH_VIS: [u32; 18] = [
    0xFFFF_0300,                                        // ps_3_0
    0x0200_001F, 0x9000_0000, 0xA00F_0800,              // dcl_2d s0
    0x0200_001F, 0x8000_0005, 0x900F_0000,              // dcl_texcoord0 v0
    0x0300_0042, 0x800F_0000, 0x90E4_0000, 0xA0E4_0800, // texld r0, v0, s0
    0x0200_0001, 0x800F_0800, 0x8000_0000,              // mov oC0, r0.x
    0x0200_0001, 0x9001_0800, 0x8000_0000,              // mov oDepth, r0.x
    0x0000_FFFF,                                        // end
];

#[test]
fn odepth_restore_probe_shows_the_sampled_depth() {
    // Diagnosis twin of `odepth_restore_feeds_a_later_sprite_z_test`: same
    // sample, color writes ON, painting the sampled INTZ value as
    // grayscale. Left half must be dark (0.25), right half white (1.0).
    let h = Harness::with_depth();
    let scene_depth = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);
    let implicit = h.depth_stencil_surface().expect("implicit depth");
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(
        h.set_depth_stencil_surface(&scene_depth.surface_level(0)),
        0
    );
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    let left_occluder = [
        PosColorVertex {
            x: -1.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: 1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: 0.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.25,
            color: WHITE,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &left_occluder),
        0
    );

    let _ = &implicit;
    assert_eq!(h.clear_depth_stencil_surface(), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    assert_eq!(h.clear_target(RED), 0);
    let ps = h.create_pixel_shader(&PS_RESTORE_DEPTH_VIS);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    assert_eq!(h.set_texture(0, &scene_depth), 0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0);
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let full = [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ];
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full), 0);
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let left = Rgba8::from_pixel(h.read_pixel(160, 240));
    let right = Rgba8::from_pixel(h.read_pixel(480, 240));
    assert!(
        (48..=90).contains(&left.r),
        "left half painted with sampled 0.25, got {left:?}"
    );
    assert!(
        right.r > 220,
        "right half painted with sampled 1.0, got {right:?}"
    );
}

#[test]
fn intz_carries_a_working_stencil_plane() {
    // INTZ is the sampleable twin of D24S8 and carries its stencil: a
    // deferred engine REPLACE-writes material/sky ids into the stencil of
    // the same buffer it later samples raw depth from, then gates late
    // draws on stencil EQUAL/NOTEQUAL. With a stencil-less mapping those
    // gates silently pass everywhere.
    let h = Harness::with_depth();
    let intz = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(h.set_depth_stencil_surface(&intz.surface_level(0)), 0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_ALWAYS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    assert_eq!(
        h.clear(
            D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER | mtld3d_types::D3DCLEAR_STENCIL,
            BLACK,
            1.0,
            0
        ),
        0
    );

    // Mark stencil = 7 over the left half (color writes off).
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILENABLE, 1), 0);
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_STENCILFUNC, D3DCMP_ALWAYS),
        0
    );
    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILREF, 7), 0);
    assert_eq!(
        h.set_render_state(
            mtld3d_types::D3DRS_STENCILPASS,
            mtld3d_types::D3DSTENCILOP_REPLACE
        ),
        0
    );
    let left = |z: f32, color: u32| {
        [
            PosColorVertex {
                x: -1.0,
                y: 1.0,
                z,
                color,
            },
            PosColorVertex {
                x: 0.0,
                y: 1.0,
                z,
                color,
            },
            PosColorVertex {
                x: -1.0,
                y: -1.0,
                z,
                color,
            },
            PosColorVertex {
                x: 0.0,
                y: 1.0,
                z,
                color,
            },
            PosColorVertex {
                x: 0.0,
                y: -1.0,
                z,
                color,
            },
            PosColorVertex {
                x: -1.0,
                y: -1.0,
                z,
                color,
            },
        ]
    };
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &left(0.5, WHITE)),
        0,
        "stencil mark draw"
    );

    // Full-screen green quad gated on stencil EQUAL 7: left half only.
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0xF),
        0
    );
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_STENCILFUNC, mtld3d_types::D3DCMP_EQUAL),
        0
    );
    assert_eq!(
        h.set_render_state(
            mtld3d_types::D3DRS_STENCILPASS,
            mtld3d_types::D3DSTENCILOP_KEEP
        ),
        0
    );
    let cover = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: GREEN,
        },
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &cover),
        0,
        "stencil-gated draw"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(
        h.read_pixel(160, 240),
        GREEN,
        "stencil EQUAL 7 passes where the mark landed"
    );
    assert_eq!(
        h.read_pixel(480, 240),
        BLACK,
        "stencil EQUAL 7 rejects the unmarked half"
    );

    assert_eq!(h.set_render_state(mtld3d_types::D3DRS_STENCILENABLE, 0), 0);
}

/// `depth.aliasSameSize`: a same-size depth-stencil bind inherits contents.
///
/// Engines of the D3D9 era rely on equal-size depth-stencil surfaces
/// sharing one physical driver allocation: they render scene depth with
/// one depth texture bound, then bind a *different* same-size depth
/// texture and z-test against the scene depth through it, with no copy
/// anywhere in the API stream. With the option on, rebinding texture A
/// after rendering into texture B must make A's z-test see B's contents.
#[test]
fn same_size_depth_bind_inherits_contents_when_aliased() {
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        config_entries: "depth.aliasSameSize=true",
        ..HarnessConfig::default()
    });
    let a = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let b = h.create_texture(
        640,
        480,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESSEQUAL), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);

    let half = |x0: f32, x1: f32, z: f32| {
        let v = |x: f32, y: f32| PosColorVertex {
            x,
            y,
            z,
            color: WHITE,
        };
        [
            v(x0, 1.0),
            v(x1, 1.0),
            v(x0, -1.0),
            v(x1, 1.0),
            v(x1, -1.0),
            v(x0, -1.0),
        ]
    };

    assert_eq!(h.begin_scene(), 0);
    // Depth A: left half occluded at 0.25, right half stays at the clear.
    assert_eq!(h.set_depth_stencil_surface(&a.surface_level(0)), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
        0
    );
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0),
        0
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &half(-1.0, 0.0, 0.25)),
        0,
        "occluder into depth A"
    );
    // Depth B: cleared (killing the A→B carry), right half occluded at 0.25.
    assert_eq!(h.set_depth_stencil_surface(&b.surface_level(0)), 0);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, BLACK, 1.0, 0), 0);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &half(0.0, 1.0, 0.25)),
        0,
        "occluder into depth B"
    );
    // Rebind A with no clear: the carry must hand it B's contents (right
    // blocked at 0.25, left back at 1.0), not leave its own.
    assert_eq!(h.set_depth_stencil_surface(&a.surface_level(0)), 0);
    assert_eq!(
        h.set_render_state(mtld3d_types::D3DRS_COLORWRITEENABLE, 0xF),
        0
    );
    let full = |z: f32, color: u32| {
        let v = |x: f32, y: f32| PosColorVertex { x, y, z, color };
        [
            v(-1.0, 1.0),
            v(1.0, 1.0),
            v(-1.0, -1.0),
            v(1.0, 1.0),
            v(1.0, -1.0),
            v(-1.0, -1.0),
        ]
    };
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full(0.5, GREEN)),
        0,
        "z-tested full-screen quad"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(
        h.read_pixel(160, 240),
        GREEN,
        "left half passes: the inherited depth is clear there"
    );
    assert_eq!(
        h.read_pixel(480, 240),
        BLACK,
        "right half fails: the inherited depth carries B's occluder"
    );
}

/// A depth-stencil surface created and released once per frame, 64 times over.
///
/// Each `CreateDepthStencilSurface` surface owns a Metal depth texture, and
/// the device holds the surface alive while it is bound, so the texture is
/// released one frame after the application drops its own reference: the
/// binding of the next surface is what finalizes the previous one, while the
/// frame that drew against it is still in flight. `make test` runs with
/// `MTL_DEBUG_LAYER` on, so a destroy that lands before that frame retires
/// aborts the process instead of reading freed storage. Every iteration also
/// depth-tests through its own fresh surface, which fails if a retire took a
/// texture the next binding still needed.
#[test]
fn depth_stencil_surfaces_released_across_frames_stay_sound() {
    const ROUNDS: u32 = 64;

    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZWRITEENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);

    let quad = |z: f32, color: u32| {
        [
            PosColorVertex {
                x: -1.0,
                y: 3.0,
                z,
                color,
            },
            PosColorVertex {
                x: 3.0,
                y: -1.0,
                z,
                color,
            },
            PosColorVertex {
                x: -1.0,
                y: -1.0,
                z,
                color,
            },
        ]
    };
    let near = quad(0.25, GREEN);
    let far = quad(0.75, RED);

    for round in 0..ROUNDS {
        let ds = h.create_depth_stencil_surface(640, 480, D3DFMT_D24S8);
        let (hr, desc) = ds.desc();
        assert_eq!(hr, 0, "round {round}: created surface describes");
        assert_eq!(
            (desc.width, desc.height),
            (640, 480),
            "round {round}: extent"
        );
        assert_eq!(
            h.set_depth_stencil_surface(&ds),
            0,
            "round {round}: bind the fresh depth surface"
        );
        assert_eq!(
            h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLACK, 1.0, 0),
            0,
            "round {round}: clear colour and depth"
        );
        assert_eq!(h.begin_scene(), 0);
        assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &near), 0);
        assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &far), 0);
        assert_eq!(h.end_scene(), 0);
        // Read back on the first, a middle and the last round: the depth test
        // has to keep working on every fresh texture, and the readback is a
        // GPU sync, which is what lets the retention queue drain.
        if round == 0 || round == ROUNDS / 2 || round == ROUNDS - 1 {
            assert_eq!(
                h.read_pixel(320, 240),
                GREEN,
                "round {round}: the near quad owns the pixel, so the fresh \
                 depth surface cleared and tested"
            );
        }
        assert_eq!(h.present(), 0, "round {round}: present");
        // Dropping the surface here leaves the device's binding as its last
        // reference; the next round's bind releases it and queues the retire.
    }

    // Unbind so the final surface retires while the device is still live,
    // then prove the device still renders with no depth attachment at all.
    assert_eq!(h.clear_depth_stencil_surface(), 0, "unbind depth-stencil");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    assert_eq!(h.clear(D3DCLEAR_TARGET, BLACK, 1.0, 0), 0);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &far), 0);
    assert_eq!(h.end_scene(), 0);
    assert_eq!(
        h.read_pixel(320, 240),
        RED,
        "the device renders after every depth surface has retired"
    );
}

#[test]
fn get_render_target_data_from_a_cube_face_reads_that_face() {
    // The source surface names one subresource of the cube's Metal texture, so
    // the read-back has to blit that face and that mip rather than the
    // texture's first slice. Faces 0 and 3 carry different colours and the mip
    // chain puts a third on face 3's level 1, so a read pinned to slice 0
    // answers the face-3 reads with face 0's content.
    const EDGE: u32 = 64;
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        EDGE,
        2,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let face0 = cube.surface(0, 0);
    let face3 = cube.surface(3, 0);
    let face3_mip1 = cube.surface(3, 1);
    assert_eq!(h.color_fill_hr(&face0, RED), D3D_OK, "fill face 0 red");
    assert_eq!(h.color_fill_hr(&face3, GREEN), D3D_OK, "fill face 3 green");
    assert_eq!(
        h.color_fill_hr(&face3_mip1, BLUE),
        D3D_OK,
        "fill face 3 level 1 blue"
    );

    assert_eq!(
        read_surface_pixel(&h, &face3, 1, 1),
        GREEN,
        "GetRenderTargetData reads face 3 rather than face 0"
    );
    assert_eq!(
        read_surface_pixel(&h, &face3_mip1, 1, 1),
        BLUE,
        "GetRenderTargetData reads face 3's level 1 rather than face 0's"
    );
    assert_eq!(
        read_surface_pixel(&h, &face0, 1, 1),
        RED,
        "face 0 still reads its own fill"
    );
}
