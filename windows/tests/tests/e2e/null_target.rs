//! The `NULL` FOURCC render-target format: a render target with no colour storage.
//!
//! The device offers it as a 2D render target, a surface or a texture with
//! `D3DUSAGE_RENDERTARGET` alone, at `D3DMULTISAMPLE_NONE`, over any depth
//! format. A `NULL` surface reports its own extent and format, a lockable one
//! hands `LockRect` a writable buffer with a non-zero pitch, a `ColorFill` of
//! one succeeds and changes nothing, and neither end of a `StretchRect` may be
//! one. Bound as render target 0 it takes no colour write: a colour `Clear`
//! through it succeeds without effect, and its draws and rect depth clears
//! write the depth surface alone, over the depth surface's extent rather than
//! its own, scaled or not and at the depth surface's sample count, whether the
//! target is a surface or a texture level.

use mtld3d_tests::{Harness, Rgba8, RhwVertex, Surface, TexturedVertex};
use mtld3d_types::{
    D3D_OK, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_LESS, D3DERR_INVALIDCALL,
    D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_INTZ, D3DFMT_NULL,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DLOCK_READONLY,
    D3DMULTISAMPLE_4_SAMPLES, D3DMULTISAMPLE_NONE, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DRS_LIGHTING, D3DRS_ZENABLE, D3DRS_ZFUNC,
    D3DRS_ZWRITEENABLE, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE,
    D3DRTYPE_VOLUMETEXTURE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER,
    D3DSAMP_MINFILTER, D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DYNAMIC, D3DUSAGE_QUERY_SRGBWRITE, D3DUSAGE_RENDERTARGET, D3DVIEWPORT9,
};

const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const GREEN: u32 = 0xFF00_FF00;
const WHITE: u32 = 0xFFFF_FFFF;

/// The display format the capability queries are asked against.
const DISPLAY: u32 = D3DFMT_X8R8G8B8;

/// Edge of the `NULL` target the depth-only passes draw through.
///
/// Smaller than every depth surface it is drawn over, so a draw that reached
/// only the `NULL` target's own extent would leave most of the depth surface
/// untouched.
const NULL_EDGE: u32 = 64;

/// A pre-transformed quad over `left..right` by `top..bottom` at depth `z`, in `color`.
fn rhw_quad(left: f32, right: f32, top: f32, bottom: f32, z: f32, color: u32) -> [RhwVertex; 6] {
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

/// Arm the fixed-function pipeline for unlit pre-transformed draws that test and write depth.
fn arm_depth_draws(h: &Harness) {
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
    assert_eq!(
        h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS),
        D3D_OK,
        "ZFUNC"
    );
}

/// A whole-target viewport over `width`x`height`.
const fn full_viewport(width: u32, height: u32) -> D3DVIEWPORT9 {
    D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width,
        height,
        min_z: 0.0,
        max_z: 1.0,
    }
}

/// The back buffer's extent as the pre-transformed vertex positions state it.
fn back_buffer_extent(h: &Harness) -> (f32, f32) {
    let (width, height) = h.dims();
    (
        f32::from(u16::try_from(width).expect("back-buffer width fits u16")),
        f32::from(u16::try_from(height).expect("back-buffer height fits u16")),
    )
}

/// `CheckDeviceFormat` offers `NULL` as a 2D render target and nothing else.
///
/// `CheckDepthStencilMatch` pairs it with a depth format, and
/// `CheckDeviceMultiSampleType` answers it at `D3DMULTISAMPLE_NONE` only.
#[test]
fn null_is_offered_as_a_single_sampled_2d_render_target() {
    let h = Harness::new();
    for rtype in [D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE] {
        assert_eq!(
            h.check_device_format(DISPLAY, D3DUSAGE_RENDERTARGET, rtype, D3DFMT_NULL),
            D3D_OK,
            "RENDERTARGET rtype {rtype}"
        );
        for usage in [
            0,
            D3DUSAGE_DEPTHSTENCIL,
            D3DUSAGE_DYNAMIC,
            D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_SRGBWRITE,
        ] {
            assert_eq!(
                h.check_device_format(DISPLAY, usage, rtype, D3DFMT_NULL),
                D3DERR_NOTAVAILABLE,
                "usage {usage:#x} rtype {rtype}"
            );
        }
    }
    for rtype in [D3DRTYPE_CUBETEXTURE, D3DRTYPE_VOLUMETEXTURE] {
        assert_eq!(
            h.check_device_format(DISPLAY, D3DUSAGE_RENDERTARGET, rtype, D3DFMT_NULL),
            D3DERR_NOTAVAILABLE,
            "rtype {rtype}"
        );
    }
    for depth in [D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_INTZ] {
        assert_eq!(
            h.check_depth_stencil_match(D3DFMT_X8R8G8B8, D3DFMT_NULL, depth),
            D3D_OK,
            "CheckDepthStencilMatch NULL over {depth:#x}"
        );
    }
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_NULL, 1, D3DMULTISAMPLE_NONE)
            .0,
        D3D_OK,
        "CheckDeviceMultiSampleType(NULL, NONE)"
    );
    assert_eq!(
        h.check_device_multi_sample_type(D3DFMT_NULL, 1, D3DMULTISAMPLE_4_SAMPLES)
            .0,
        D3DERR_NOTAVAILABLE,
        "CheckDeviceMultiSampleType(NULL, 4 samples)"
    );
}

/// `NULL` surfaces and textures create in the shapes the answers offer, and lock and fill.
///
/// A lockable 128x128 `NULL` render target reports its extent and format,
/// locks to a writable buffer with a non-zero pitch and unlocks; a
/// non-lockable one refuses the lock. `ColorFill` succeeds on either and on a
/// level of a full-chain `NULL` render-target texture, `StretchRect` from or
/// to one is refused, `UpdateSurface` and `UpdateTexture` with a `NULL` source
/// or destination are refused, and the shapes the answers refuse do not
/// create.
#[test]
fn null_surfaces_and_textures_create_lock_and_fill() {
    let h = Harness::new();
    let lockable = h.create_lockable_render_target(128, 128, D3DFMT_NULL);
    let (hr, desc) = lockable.desc();
    assert_eq!(hr, D3D_OK, "GetDesc");
    assert_eq!(
        (desc.format, desc.width, desc.height),
        (D3DFMT_NULL, 128, 128),
        "GetDesc"
    );
    assert_eq!(desc.usage, D3DUSAGE_RENDERTARGET, "GetDesc usage");
    {
        let locked = lockable.lock_rect(0);
        assert!(locked.pitch() >= 128 * 4, "pitch {}", locked.pitch());
        let bits = locked.bits_ptr();
        assert!(!bits.is_null(), "pBits");
        // SAFETY: the lock maps at least one row of `pitch` writable bytes.
        unsafe { bits.write(0x5a) };
    }
    let plain = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);
    let (hr, _) = plain.lock_rect_probe(0);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "LockRect on a non-lockable NULL target"
    );

    assert_eq!(
        h.color_fill_hr(&lockable, GREEN),
        D3D_OK,
        "ColorFill lockable"
    );
    assert_eq!(h.color_fill_hr(&plain, GREEN), D3D_OK, "ColorFill");
    let colour = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_A8R8G8B8);
    assert_eq!(
        h.stretch_rect(&plain, &colour, D3DTEXF_NONE),
        D3DERR_INVALIDCALL,
        "StretchRect from NULL"
    );
    assert_eq!(
        h.stretch_rect(&colour, &plain, D3DTEXF_NONE),
        D3DERR_INVALIDCALL,
        "StretchRect to NULL"
    );

    let texture = h.create_texture(
        128,
        128,
        0,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_NULL,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(texture.level_count(), 8, "full chain");
    let level = texture.surface_level(1);
    let (hr, desc) = level.desc();
    assert_eq!(hr, D3D_OK, "level GetDesc");
    assert_eq!(
        (desc.format, desc.width, desc.height),
        (D3DFMT_NULL, 64, 64),
        "level GetDesc"
    );
    assert_eq!(h.color_fill_hr(&level, BLUE), D3D_OK, "ColorFill level");

    // No system-memory `NULL` resource exists, so neither update has a source
    // of the destination's format, and a `NULL` source is in the wrong pool.
    let sysmem =
        h.create_offscreen_plain_surface(NULL_EDGE, NULL_EDGE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.update_surface_hr(&sysmem, &level),
        D3DERR_INVALIDCALL,
        "UpdateSurface into a NULL level"
    );
    assert_eq!(
        h.update_surface_hr(&sysmem, &plain),
        D3DERR_INVALIDCALL,
        "UpdateSurface into a NULL surface"
    );
    assert_eq!(
        h.update_surface_hr(&lockable, &colour),
        D3DERR_INVALIDCALL,
        "UpdateSurface from a NULL surface"
    );
    let sysmem_texture = h.create_texture(128, 128, 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.update_texture_hr(&sysmem_texture, &texture),
        D3DERR_INVALIDCALL,
        "UpdateTexture into a NULL texture"
    );
    let default_texture = h.create_texture(128, 128, 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&texture, &default_texture),
        D3DERR_INVALIDCALL,
        "UpdateTexture from a NULL texture"
    );

    let (hr, multisampled) = h.create_render_target_ms_hr(
        (NULL_EDGE, NULL_EDGE),
        D3DFMT_NULL,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
        0,
    );
    drop(multisampled);
    assert_ne!(hr, D3D_OK, "multisampled NULL render target");
    for (usage, pool) in [
        (0, D3DPOOL_DEFAULT),
        (0, D3DPOOL_MANAGED),
        (D3DUSAGE_DYNAMIC, D3DPOOL_DEFAULT),
    ] {
        let (hr, out) = h.try_create_texture(16, 16, 1, usage, D3DFMT_NULL, pool);
        assert!(out.is_null(), "usage {usage:#x} pool {pool}");
        assert_ne!(hr, D3D_OK, "usage {usage:#x} pool {pool}");
    }
}

/// A colour `Clear` through a `NULL` render target 0 with no depth surface succeeds without effect.
///
/// The back buffer keeps the colour it was cleared to before, and a draw
/// through the `NULL` target with nothing to write is dropped without effect.
#[test]
fn a_colour_clear_through_a_null_target_changes_nothing() {
    let h = Harness::new();
    let back_buffer = h.render_target(0);
    let depth = h.depth_stencil_surface();
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.clear_target(BLUE), D3D_OK, "clear the back buffer");
    let null = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);
    assert_eq!(h.set_render_target(0, &null), D3D_OK, "bind NULL");
    assert_eq!(h.clear_depth_stencil_surface(), D3D_OK, "unbind depth");
    assert_eq!(h.clear_target(GREEN), D3D_OK, "colour Clear through NULL");
    arm_depth_draws(&h);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK, "depth off");
    let edge = f32::from(u16::try_from(NULL_EDGE).expect("edge fits u16"));
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, edge, 0.0, edge, 0.5, GREEN)
        ),
        D3D_OK,
        "draw through NULL with no depth"
    );
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    if let Some(depth) = depth.as_ref() {
        assert_eq!(h.set_depth_stencil_surface(depth), D3D_OK);
    }
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
    let (width, height) = h.dims();
    for (x, y) in [(8, 8), (width / 2, height / 2)] {
        assert_eq!(h.read_pixel(x, y), BLUE, "back buffer at ({x}, {y})");
    }
}

/// A depth-only pass through a `NULL` render target 0 writes a sampleable depth surface.
///
/// A 64x64 `NULL` target over a 256x256 `INTZ` texture level, under a
/// viewport over the whole depth surface: a `Clear` of the target and the
/// depth reaches the whole depth surface, and a draw at depth 0.25 over the
/// left half reaches past the `NULL` target's own extent. The `INTZ` texture,
/// sampled back onto the back buffer, reads 0.25 on the left half, inside
/// and outside the `NULL` target's area, and the cleared 1.0 on the right.
#[test]
fn a_depth_only_pass_through_a_null_target_writes_an_intz_surface() {
    const DEPTH_EDGE: u32 = 256;
    const DEPTH_EDGE_F: f32 = 256.0;
    let h = Harness::new();
    let back_buffer = h.render_target(0);
    let depth_texture = h.create_texture(
        DEPTH_EDGE,
        DEPTH_EDGE,
        1,
        D3DUSAGE_DEPTHSTENCIL,
        D3DFMT_INTZ,
        D3DPOOL_DEFAULT,
    );
    let depth = depth_texture.surface_level(0);
    let null = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);

    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.set_render_target(0, &null), D3D_OK, "bind NULL");
    assert_eq!(h.set_depth_stencil_surface(&depth), D3D_OK, "bind INTZ");
    assert_eq!(
        h.set_viewport(&full_viewport(DEPTH_EDGE, DEPTH_EDGE)),
        D3D_OK,
        "viewport over the depth surface"
    );
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, GREEN, 1.0, 0),
        D3D_OK,
        "clear through NULL"
    );
    arm_depth_draws(&h);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, DEPTH_EDGE_F / 2.0, 0.0, DEPTH_EDGE_F, 0.25, GREEN)
        ),
        D3D_OK,
        "depth-only draw"
    );

    // Sample the INTZ texture onto the back buffer, with depth unbound so it
    // is no attachment of the sampling pass.
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    assert_eq!(h.clear_depth_stencil_surface(), D3D_OK, "unbind depth");
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK, "depth off");
    assert_eq!(h.clear_target(BLACK), D3D_OK);
    assert_eq!(h.set_texture(0, &depth_texture), D3D_OK, "bind INTZ");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), D3D_OK, "sampler");
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    let v = |x: f32, y: f32, u: f32, vv: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u,
        v: vv,
    };
    let screen = [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ];
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &screen),
        D3D_OK,
        "sampling draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
    assert_eq!(h.clear_texture(0), D3D_OK, "unbind INTZ");

    // Back-buffer probes and the depth texel each one samples.
    let (width, height) = h.dims();
    let at = |u: u32, v: u32| (u * width / DEPTH_EDGE, v * height / DEPTH_EDGE);
    let near = |name: &str, (x, y): (u32, u32), expected: u8| {
        let pixel = Rgba8::from_pixel(h.read_pixel(x, y));
        assert!(
            pixel.r.abs_diff(expected) <= 12,
            "{name}: back buffer at ({x}, {y}) reads {pixel:?}, expected about {expected}"
        );
    };
    // 0.25 inside the NULL target's 64x64 area and past it on the left half.
    near("inside the NULL area", at(32, 32), 64);
    near("left half below the NULL area", at(32, 200), 64);
    near("left half right of the NULL area", at(100, 128), 64);
    // The cleared 1.0 on the right half.
    near("right half", at(200, 128), 255);
}

/// A `NULL` render target 0 over the back buffer's depth surface draws in that surface's space.
///
/// The implicit depth surface is reduced with the back buffer under
/// `render.scale`, and the `NULL` target follows it: a draw at depth 0.25
/// over the left half of the back buffer under a viewport over the whole back
/// buffer, through a 64x64 `NULL` target, lays its depth over the whole left
/// half. A green quad at depth 0.5 drawn through the back buffer afterwards
/// then shows on the right half only, at any scale.
#[test]
fn a_null_target_over_the_back_buffer_depth_draws_in_its_space() {
    let h = Harness::with_depth();
    let back_buffer = h.render_target(0);
    let null = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);
    let (width, height) = h.dims();
    let (width_f, height_f) = back_buffer_extent(&h);

    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, BLUE, 1.0, 0),
        D3D_OK,
        "clear the back buffer and its depth"
    );
    assert_eq!(h.set_render_target(0, &null), D3D_OK, "bind NULL");
    assert_eq!(
        h.set_viewport(&full_viewport(width, height)),
        D3D_OK,
        "viewport over the back buffer"
    );
    arm_depth_draws(&h);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, width_f / 2.0, 0.0, height_f, 0.25, GREEN)
        ),
        D3D_OK,
        "depth-only draw"
    );
    assert_eq!(h.set_render_target(0, &back_buffer), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, width_f, 0.0, height_f, 0.5, GREEN)
        ),
        D3D_OK,
        "depth-tested draw"
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);

    for (x, y) in [(width / 8, height / 8), (width / 4, height * 7 / 8)] {
        assert_eq!(h.read_pixel(x, y), BLUE, "left half at ({x}, {y})");
    }
    // Just right of the middle: a draw converted in the `NULL` target's own
    // unscaled space would reach past it under a reduced `render.scale`.
    for (x, y) in [
        (width * 9 / 16, height / 2),
        (width * 3 / 4, height / 8),
        (width * 7 / 8, height * 7 / 8),
    ] {
        assert_eq!(h.read_pixel(x, y), GREEN, "right half at ({x}, {y})");
    }
}

/// Edge of the depth surfaces the read-back cases draw a `NULL` target over.
const DEPTH_EDGE: u32 = 128;
/// [`DEPTH_EDGE`] as the vertex positions state it.
const DEPTH_EDGE_F: f32 = 128.0;

/// Probes on the left and right halves of a [`DEPTH_EDGE`] square, inside and past the `NULL` area.
const HALF_PROBES: [(u32, u32); 4] = [(16, 16), (40, 100), (80, 16), (112, 100)];

/// Bind `null` as render target 0 over `depth`, under a viewport over the whole depth surface.
fn bind_null_over(h: &Harness, null: &Surface<'_>, depth: &Surface<'_>) {
    assert_eq!(h.set_render_target(0, null), D3D_OK, "bind NULL");
    assert_eq!(h.set_depth_stencil_surface(depth), D3D_OK, "bind depth");
    assert_eq!(
        h.set_viewport(&full_viewport(DEPTH_EDGE, DEPTH_EDGE)),
        D3D_OK,
        "viewport over the depth surface"
    );
}

/// Draw a depth-0.25 quad over the left half of the depth surface through the bound target.
fn draw_left_half(h: &Harness) {
    arm_depth_draws(h);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, DEPTH_EDGE_F / 2.0, 0.0, DEPTH_EDGE_F, 0.25, GREEN)
        ),
        D3D_OK,
        "depth-only draw"
    );
}

/// Read `depth` back through a colour target of its size and sample count, at [`HALF_PROBES`].
///
/// A green depth-0.5 quad drawn under `ZFUNC` less over a black target shows
/// where the depth surface holds more than 0.5 and stays black where it holds
/// less. A multisampled target is resolved with `StretchRect` first.
fn depth_read_back(h: &Harness, depth: &Surface<'_>, samples: u32) -> Vec<u32> {
    let colour = if samples > 1 {
        h.create_render_target_ms((DEPTH_EDGE, DEPTH_EDGE), D3DFMT_A8R8G8B8, (samples, 0))
    } else {
        h.create_render_target(DEPTH_EDGE, DEPTH_EDGE, D3DFMT_A8R8G8B8)
    };
    assert_eq!(
        h.set_render_target(0, &colour),
        D3D_OK,
        "bind the colour target"
    );
    assert_eq!(h.set_depth_stencil_surface(depth), D3D_OK, "bind depth");
    assert_eq!(h.clear_target(BLACK), D3D_OK, "clear the colour target");
    arm_depth_draws(h);
    assert_eq!(
        h.draw_primitive_up(
            D3DPT_TRIANGLELIST,
            2,
            &rhw_quad(0.0, DEPTH_EDGE_F, 0.0, DEPTH_EDGE_F, 0.5, GREEN)
        ),
        D3D_OK,
        "depth-tested draw"
    );
    let single = if samples > 1 {
        let single = h.create_render_target(DEPTH_EDGE, DEPTH_EDGE, D3DFMT_A8R8G8B8);
        assert_eq!(
            h.stretch_rect(&colour, &single, D3DTEXF_NONE),
            D3D_OK,
            "resolve"
        );
        single
    } else {
        colour
    };
    let sysmem = h.create_offscreen_plain_surface(
        DEPTH_EDGE,
        DEPTH_EDGE,
        D3DFMT_A8R8G8B8,
        D3DPOOL_SYSTEMMEM,
    );
    assert_eq!(
        h.get_render_target_data_hr(&single, &sysmem),
        D3D_OK,
        "GetRenderTargetData"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let pixels = locked.as_u32((DEPTH_EDGE * pitch_px) as usize);
    HALF_PROBES
        .iter()
        .map(|&(x, y)| pixels[(y * pitch_px + x) as usize])
        .collect()
}

/// The left half holds the drawn 0.25 and the right half the cleared 1.0.
fn assert_left_half_drawn(probes: &[u32], what: &str) {
    assert_eq!(
        probes,
        [BLACK, BLACK, GREEN, GREEN],
        "{what}: probes at {HALF_PROBES:?}"
    );
}

/// A level of a `NULL` render-target texture draws depth only, over its depth surface's extent.
///
/// Level 1 of a 64x64 `NULL` texture, 32x32, is render target 0 over a
/// 128x128 D24S8 surface: a whole depth `Clear` and a draw over the left half
/// reach the depth surface past the level's own extent.
#[test]
fn a_level_of_a_null_texture_draws_depth_only_over_its_depth_surface() {
    let h = Harness::new();
    let depth = h.create_depth_stencil_surface(DEPTH_EDGE, DEPTH_EDGE, D3DFMT_D24S8);
    let texture = h.create_texture(
        NULL_EDGE,
        NULL_EDGE,
        0,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_NULL,
        D3DPOOL_DEFAULT,
    );
    let level = texture.surface_level(1);
    assert_eq!(h.begin_scene(), D3D_OK);
    bind_null_over(&h, &level, &depth);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, GREEN, 1.0, 0),
        D3D_OK,
        "clear through the NULL level"
    );
    draw_left_half(&h);
    let probes = depth_read_back(&h, &depth, 1);
    assert_eq!(h.end_scene(), D3D_OK);
    assert_left_half_drawn(&probes, "NULL texture level");
}

/// A rect depth `Clear` through a `NULL` render target 0 reaches its rect past the target's extent.
///
/// A 64x64 `NULL` target over a 128x128 D24S8 surface: the whole surface is
/// cleared to 0.25, then a rect `Clear` covering the right half, which lies
/// entirely past the `NULL` target's area, clears it to 1.0.
#[test]
fn a_rect_depth_clear_through_a_null_target_reaches_past_its_extent() {
    let h = Harness::new();
    let depth = h.create_depth_stencil_surface(DEPTH_EDGE, DEPTH_EDGE, D3DFMT_D24S8);
    let null = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);
    let edge = i32::try_from(DEPTH_EDGE).expect("edge fits i32");
    assert_eq!(h.begin_scene(), D3D_OK);
    bind_null_over(&h, &null, &depth);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER, 0, 0.25, 0),
        D3D_OK,
        "whole depth clear"
    );
    assert_eq!(
        h.clear_rects(
            D3DCLEAR_ZBUFFER,
            0,
            1.0,
            0,
            &[D3DRECT {
                x1: edge / 2,
                y1: 0,
                x2: edge,
                y2: edge,
            }]
        ),
        D3D_OK,
        "rect depth clear"
    );
    let probes = depth_read_back(&h, &depth, 1);
    assert_eq!(h.end_scene(), D3D_OK);
    assert_left_half_drawn(&probes, "rect clear");
}

/// A `NULL` render target 0 over a multisampled depth surface draws at its sample count.
///
/// The `NULL` target is single-sampled, the 128x128 D24S8 surface four times
/// multisampled: the depth-only pass runs at four samples, and a four-sample
/// colour target over the same surface, resolved, reads the draw back.
#[test]
fn a_null_target_over_a_multisampled_depth_surface_draws_at_its_sample_count() {
    let h = Harness::new();
    if h.check_device_multi_sample_type(D3DFMT_D24S8, 1, D3DMULTISAMPLE_4_SAMPLES)
        .0
        != D3D_OK
    {
        return;
    }
    let (hr, depth) = h.create_depth_stencil_surface_ms_hr(
        (DEPTH_EDGE, DEPTH_EDGE),
        D3DFMT_D24S8,
        (D3DMULTISAMPLE_4_SAMPLES, 0),
    );
    assert_eq!(hr, D3D_OK, "four-sample depth surface");
    let depth = depth.expect("created");
    let null = h.create_render_target(NULL_EDGE, NULL_EDGE, D3DFMT_NULL);
    assert_eq!(h.begin_scene(), D3D_OK);
    bind_null_over(&h, &null, &depth);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), D3D_OK, "depth clear");
    draw_left_half(&h);
    let probes = depth_read_back(&h, &depth, D3DMULTISAMPLE_4_SAMPLES);
    assert_eq!(h.end_scene(), D3D_OK);
    assert_left_half_drawn(&probes, "four-sample depth");
}
