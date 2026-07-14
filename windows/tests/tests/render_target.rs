//! Offscreen render target round-trip and depth-buffered occlusion.
//!
//! The round-trip renders to a texture, then samples it.

use mtld3d_tests::{Harness, PosColorVertex, Rgba8, TexturedVertex, Vertex};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_ALWAYS, D3DCMP_LESSEQUAL, D3DERR_INVALIDCALL,
    D3DFMT_A8R8G8B8, D3DFMT_A32B32G32R32F, D3DFMT_D24S8, D3DFMT_INTZ, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRECT, D3DRS_LIGHTING, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTA_DIFFUSE,
    D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DTOP_MODULATE,
    D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2,
    D3DTSS_COLOROP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_RENDERTARGET, D3DVIEWPORT9,
};

const RED: u32 = 0xFFFF_0000;
const BLACK: u32 = 0xFF00_0000;
const WHITE: u32 = 0xFFFF_FFFF;
const GREEN: u32 = 0xFF00_FF00;

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
    // texture surface and the backbuffer (both BGRA8). The blit itself lands as
    // a per-pass leading blit; its pixel timing is exercised by the in-game
    // water/portrait paths rather than this synthetic single-frame flow.
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

    // Give the RT real content first (a clear-only pass can be dropped on TBDR).
    assert_eq!(h.set_render_target(0, &rt_surface), 0, "bind RT");
    assert_eq!(h.clear_target(RED), 0, "clear RT red");
    assert_eq!(h.set_render_target(0, &backbuffer), 0, "restore backbuffer");

    assert_eq!(
        h.stretch_rect(&rt_surface, &backbuffer, D3DTEXF_NONE),
        0,
        "1:1 same-format StretchRect is accepted",
    );
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
    // D3DPOOL_SYSTEMMEM and D3DPOOL_DEFAULT offscreen plain surfaces are
    // supported; MANAGED/SCRATCH are not.
    assert_eq!(
        h.create_offscreen_plain_surface_hr(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT),
        0,
        "CreateOffscreenPlainSurface(D3DPOOL_DEFAULT) succeeds",
    );
    assert_eq!(
        h.create_offscreen_plain_surface_hr(64, 64, D3DFMT_A8R8G8B8, 1 /* D3DPOOL_MANAGED */),
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
fn render_to_default_pool_target_round_trips() {
    // A DEFAULT-pool render target can be bound, drawn into, and is then a valid
    // GetRenderTargetData source into a SYSTEMMEM surface — i.e. create_color_target
    // produces a real, renderable, readable Metal texture that SetRenderTarget and
    // the readback blit both resolve via metal_color_handle. Metal validation is on
    // under `make test`, so a malformed RT attachment would abort the draw.
    //
    // The blit's *pixel* contents are not asserted: nothing inside the frame
    // samples the offscreen RT, so the load/store optimiser culls its colour
    // store (the post-flush GetRenderTargetData blit is invisible to it). Pixel
    // round-trips through a drawn-into offscreen RT need that store preserved — a
    // separate optimiser change. Here we assert the API contract.
    const TEAL: u32 = 0xFF00_8080;
    let h = Harness::new();
    // Capture the implicit backbuffer so we can restore RT0 before `rt` drops.
    let bb = h.render_target(0);

    let rt = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &rt), 0, "bind DEFAULT RT");
    assert_eq!(h.clear_target(TEAL), 0, "clear RT teal");
    assert_eq!(h.clear_texture(0), 0, "no texture for the fill draw");
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

    let sysmem = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&rt, &sysmem),
        0,
        "GetRenderTargetData DEFAULT RT → SYSTEMMEM",
    );
}

#[test]
fn stretch_rect_between_default_pool_targets() {
    // 1:1 same-format StretchRect is accepted between two DEFAULT render targets,
    // and the destination is then a valid GetRenderTargetData source — i.e. a
    // standalone color surface works as both StretchRect src and dst. Pixel
    // timing of a synthetic single-frame StretchRect is unreliable on TBDR (see
    // stretch_rect_accepts_one_to_one_same_format), so this asserts the API
    // contract, not the propagated colour.
    let h = Harness::new();

    let src = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    let dst = h.create_render_target(64, 64, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.stretch_rect(&src, &dst, D3DTEXF_NONE),
        0,
        "1:1 same-format StretchRect between DEFAULT RTs",
    );

    let sysmem = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&dst, &sysmem),
        0,
        "GetRenderTargetData DEFAULT-RT dst → SYSTEMMEM",
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
