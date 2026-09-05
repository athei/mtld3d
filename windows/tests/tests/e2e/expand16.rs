//! The packed 16-bit expansion path, forced on via `intel.expandPacked16`.
//!
//! On devices without Metal's packed 16-bit pixel formats (Intel/AMD Mac2),
//! A4R4G4B4 / R5G6B5 / A1R5G5B5 / X1R5G5B5 textures are backed by BGRA8 and
//! widened by
//! the GPU upload pass, and the 16-bit render-target formats stop being
//! advertised. These tests run that whole path on Apple Silicon by forcing
//! the config key, so it cannot rot between rare Intel-hardware runs.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex, VolumeVertex, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_SRCALPHA, D3DERR_NOTAVAILABLE, D3DFMT_A1R5G5B5,
    D3DFMT_A4R4G4B4, D3DFMT_R5G6B5, D3DFMT_X1R5G5B5, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_TEXTUREFORMAT3, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPT_TRIANGLELIST,
    D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND, D3DRS_SRCBLEND, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER,
    D3DSAMP_MIPFILTER, D3DTADDRESS_CLAMP, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_RENDERTARGET,
};

const BLACK: u32 = 0xFF00_0000;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const WHITE: u32 = 0xFFFF_FFFF;

/// A device whose interface takes the expansion path.
///
/// The key is resolved by this harness's `Direct3DCreate9` alone, so the
/// rest of the suite, sharing the process, keeps the device's own answer.
fn expanding_harness() -> Harness {
    Harness::with_config("intel.expandPacked16=true")
}

/// A full-backbuffer quad (two triangles) with UVs spanning the unit square.
const fn fullscreen_quad() -> [TexturedVertex; 6] {
    const W: u32 = 0xFFFF_FFFF;
    const fn v(x: f32, y: f32, u: f32, tv: f32) -> TexturedVertex {
        TexturedVertex {
            x,
            y,
            z: 0.5,
            color: W,
            u,
            v: tv,
        }
    }
    [
        v(-1.0, 1.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(-1.0, -1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0, 0.0),
        v(1.0, -1.0, 1.0, 1.0),
        v(-1.0, -1.0, 0.0, 1.0),
    ]
}

fn point_clamp(h: &Harness) {
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_NONE),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
}

/// Fill the whole of mip `level` with one packed 16-bit texel.
///
/// Writes row by row against the pitch the lock reports: a low mip's row
/// stride is not the tight `width * 2`.
fn fill_level(tex: &Texture<'_>, level: u32, side: u32, texel: u16) {
    let locked = tex.lock_rect(level, 0);
    let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
    let row: Vec<u16> = vec![texel; side as usize];
    for y in 0..side as usize {
        // SAFETY: the lock maps `side` rows at `pitch` stride, and each row
        // holds `side` texels.
        let dst = unsafe { locked.bits_ptr().add(y * pitch) };
        // SAFETY: `side` texels fit in the locked row per above.
        unsafe {
            core::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), dst, side as usize * 2);
        }
    }
}

/// Bind `tex`, sample mip `level` across the backbuffer, return the centre pixel.
///
/// `D3DSAMP_MAXMIPLEVEL` pins the most detailed level the sampler may use;
/// the quad magnifies, so that is the level every fragment reads.
fn sample_level(h: &Harness, tex: &Texture<'_>, level: u32) -> u32 {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "sample draw"
        );
    });
    h.read_pixel(320, 240)
}

/// Bind `tex`, sample it across the backbuffer, return the pixel at `(x, y)`.
fn sample_at(h: &Harness, tex: &Texture<'_>, x: u32, y: u32) -> Rgba8 {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "sample draw"
        );
    });
    Rgba8::from_pixel(h.read_pixel(x, y))
}

#[test]
fn expanded_color_formats_sample_red() {
    let h = expanding_harness();
    // 1×1 opaque-red texel encoded for each format (little-endian bytes).
    let cases: [(u32, &[u8]); 4] = [
        (D3DFMT_R5G6B5, &[0x00, 0xF8]),   // R=31
        (D3DFMT_A1R5G5B5, &[0x00, 0xFC]), // A=1 R=31
        (D3DFMT_X1R5G5B5, &[0x00, 0x7C]), // X=0 R=31
        (D3DFMT_A4R4G4B4, &[0x00, 0xFF]), // A=F R=F
    ];
    for (format, bytes) in cases {
        let tex = h.create_texture(1, 1, 1, 0, format, 0);
        tex.lock_rect(0, 0).write(bytes);
        let px = sample_at(&h, &tex, 320, 240);
        assert!(
            px.r > 200 && px.g < 60 && px.b < 60,
            "format {format:#x} red, got {px:?}"
        );
    }
}

/// The expansion forces `X1R5G5B5`'s alpha opaque in the widened texel.
///
/// Its BGRA8 backing carries no sampler swizzle (a swizzled view cannot be a
/// render-pass attachment), so the opaque alpha has to come out of the upload
/// pass itself. A texel with the padding bit clear must still blend as fully
/// opaque.
#[test]
fn expanded_x1r5g5b5_blends_opaque_with_its_top_bit_clear() {
    const BLUE_CLEAR: u32 = 0xFF00_00FF;
    // X=0 R=31: red with the padding bit clear.
    const RED555: u16 = 0x7C00;
    let h = expanding_harness();
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_X1R5G5B5, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write(&[RED555]);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
    let quad = fullscreen_quad();
    h.render_once(BLUE_CLEAR, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "blend draw"
        );
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        px.r > 200 && px.b < 60,
        "the widened texel blends over blue at alpha 1, got {px:?}"
    );
}

#[test]
fn expanded_partial_lock_updates_only_the_dirty_rect() {
    // 4×4 A4R4G4B4, MANAGED so partial locks track a dirty sub-rect. Fill
    // green everywhere, sample once (uploads the whole mip), then rewrite the
    // centre 2×2 red through a partial lock. The re-upload scopes the pass to
    // that sub-rect, so the origin offset must reach the right staging texels
    // and the untouched border texels must survive the pass's load action.
    const GREEN16: u16 = 0xF0F0; // A=F R=0 G=F B=0
    const RED16: u16 = 0xFF00; // A=F R=F G=0 B=0
    let h = expanding_harness();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A4R4G4B4, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write(&[GREEN16; 16]);
    let px = sample_at(&h, &tex, 320, 240);
    assert!(px.g > 200 && px.r < 60, "initial fill green, got {px:?}");

    {
        let locked = tex.lock_rect_partial(0, &[1, 1, 3, 3], 0);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
        let base = locked.bits_ptr();
        let [lo, hi] = RED16.to_le_bytes();
        let red_row = [lo, hi, lo, hi];
        for row in 0..2usize {
            // SAFETY: the lock maps the 2×2 sub-rect at `base` with `pitch`
            // row stride; both rows × 2 texels stay inside the mapping.
            let dst = unsafe { base.add(row * pitch) };
            // SAFETY: 4 bytes (2 texels) fit in the locked row per above.
            unsafe { core::ptr::copy_nonoverlapping(red_row.as_ptr(), dst, 4) };
        }
    }
    // Texel (2,2) is inside the rect: red. Texel (0,0) is outside: still the
    // original green. Point sampling maps each texel of the 4×4 onto a
    // 160×120 backbuffer cell.
    let inside = sample_at(&h, &tex, 320 + 80, 240 + 60);
    assert!(
        inside.r > 200 && inside.g < 60,
        "texel inside the dirty rect red, got {inside:?}"
    );
    let outside = sample_at(&h, &tex, 80, 60);
    assert!(
        outside.g > 200 && outside.r < 60,
        "texel outside the dirty rect keeps green, got {outside:?}"
    );
}

#[test]
fn expanded_mip_chain_samples_every_level() {
    // 8x8 R5G6B5, full chain: 8, 4, 2 and 1 texels wide. Their tight row
    // pitches are 16, 8, 4 and 2 bytes, so every mip but the top is below
    // the 16-byte linear texture alignment a blit source has to meet on
    // Apple Silicon. The upload reads the staging by texel instead, so the
    // pitch is unconstrained and each level lands on its own mip.
    let h = expanding_harness();
    let tex = h.create_texture(8, 8, 0, 0, D3DFMT_R5G6B5, D3DPOOL_MANAGED);
    assert_eq!(tex.level_count(), 4, "8x8 full mip chain");
    let levels: [(u16, u32); 4] = [
        (0xF800, RED),
        (0x07E0, GREEN),
        (0x001F, BLUE),
        (0xFFFF, WHITE),
    ];
    for (level, (texel, _)) in levels.iter().enumerate() {
        let level = u32::try_from(level).expect("mip index fits u32");
        fill_level(&tex, level, 8 >> level, *texel);
    }
    for (level, (_, expected)) in levels.iter().enumerate() {
        let level = u32::try_from(level).expect("mip index fits u32");
        assert_pixel_eq(
            sample_level(&h, &tex, level),
            *expected,
            &format!("expanded mip {level}"),
        );
    }
}

#[test]
fn expanded_render_target_caps_are_denied() {
    let h = expanding_harness();
    for format in [D3DFMT_R5G6B5, D3DFMT_A1R5G5B5] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_RENDERTARGET,
                D3DRTYPE_SURFACE,
                format
            ),
            D3DERR_NOTAVAILABLE,
            "RT usage denied for {format:#x}"
        );
        assert_ne!(
            h.create_render_target_hr(64, 64, format),
            D3D_OK,
            "CreateRenderTarget({format:#x}) rejected"
        );
        let (hr, _tex) =
            h.try_create_texture(64, 64, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT);
        assert_ne!(hr, D3D_OK, "CreateTexture(RT, {format:#x}) rejected");
    }
    // The sampled answers stay advertised: the expansion makes them true.
    for format in [
        D3DFMT_R5G6B5,
        D3DFMT_A1R5G5B5,
        D3DFMT_X1R5G5B5,
        D3DFMT_A4R4G4B4,
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_TEXTURE, format),
            D3D_OK,
            "sampled texture answer for {format:#x}"
        );
    }
    // X1R5G5B5 is sampling-only on every device (its native mapping carries
    // the alpha-forcing swizzle), so its render-target answer is denied here
    // for the same reason it is denied natively.
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            D3DUSAGE_RENDERTARGET,
            D3DRTYPE_SURFACE,
            D3DFMT_X1R5G5B5
        ),
        D3DERR_NOTAVAILABLE,
        "RT usage denied for X1R5G5B5"
    );
    // Conversion SOURCE side and the backbuffer question are
    // device-independent: a 16-bit source is sampled (expansion covers it)
    // and a 16-bit backbuffer substitutes to BGRA8 at CreateDevice.
    assert_eq!(
        h.check_device_format_conversion(D3DFMT_R5G6B5, D3DFMT_X8R8G8B8),
        D3D_OK,
        "R5G6B5 stays a conversion source"
    );
    assert_eq!(
        h.check_device_type(D3DFMT_X8R8G8B8, D3DFMT_R5G6B5, true),
        D3D_OK,
        "16-bit windowed backbuffer stays advertised"
    );
}

#[test]
fn expanded_cube_face_samples_red() {
    let h = expanding_harness();
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A4R4G4B4, D3DPOOL_MANAGED);
    cube.lock_rect(0, 0, 0).write(&[0xFF00u16; 16]); // +X face opaque red
    assert_eq!(h.set_cube_texture(0, &cube), 0);
    h.select_texture_stage(0);
    point_clamp(&h);
    // D3DFVF_TEXCOORDSIZE3(0) is bit 16.
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    let v = |x: f32, y: f32| VolumeVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u: 1.0, // +X direction vector
        v: 0.0,
        w: 0.0,
    };
    let quad = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_eq(h.read_pixel(320, 240), RED, "expanded +X cube face");
}

#[test]
fn expanded_volume_slices_sample_their_colors() {
    let h = expanding_harness();
    let (hr, tex) = h.try_create_volume_texture([2, 2, 2], 1, 0, D3DFMT_R5G6B5, D3DPOOL_MANAGED);
    assert_eq!(hr, 0, "MANAGED R5G6B5 volume");
    let tex = tex.expect("volume texture");
    // Slice 0 red, slice 1 blue.
    tex.write_u16(
        0,
        &[
            0xF800, 0xF800, 0xF800, 0xF800, 0x001F, 0x001F, 0x001F, 0x001F,
        ],
    );
    assert_eq!(h.set_volume_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    for (w, expected, name) in [(0.25f32, RED, "slice 0"), (0.75, BLUE, "slice 1")] {
        let v = |x: f32, y: f32| VolumeVertex {
            x,
            y,
            z: 0.5,
            color: WHITE,
            u: 0.5,
            v: 0.5,
            w,
        };
        let quad = [
            v(-1.0, 1.0),
            v(1.0, 1.0),
            v(-1.0, -1.0),
            v(1.0, 1.0),
            v(1.0, -1.0),
            v(-1.0, -1.0),
        ];
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
                0,
                "volume sample draw"
            );
        });
        assert_pixel_eq(h.read_pixel(320, 240), expected, name);
    }
}

#[test]
fn expanded_offscreen_plain_is_a_stretch_rect_source() {
    let h = expanding_harness();
    // Clone of render_target.rs's R5G6B5 → X8R8G8B8 conversion test: the
    // DEFAULT offscreen-plain source is BGRA8-backed here, and the scaling
    // render quad samples the expanded texels.
    let bb = h.render_target(0);
    let src = h.create_offscreen_plain_surface(4, 1, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    {
        let mut locked = src.lock_rect(0);
        locked.write(&[0xF800u16, 0x07E0, 0x001F, 0xFFFF]);
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
