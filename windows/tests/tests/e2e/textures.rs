//! Texture create → lock/write → bind → sample across formats and mip levels.
//!
//! Plus cube and volume texture contracts.

use mtld3d_tests::{
    Harness, LockedRect, Rgba8, Texture, TexturedVertex, VolumeVertex, assert_pixel_approx,
    assert_pixel_eq,
};
use mtld3d_types::{
    D3DBLEND_INVSRCALPHA, D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DBOX, D3DERR_INVALIDCALL,
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8, D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_ATI1,
    D3DFMT_DXT1, D3DFMT_DXT5, D3DFMT_INTZ, D3DFMT_L8, D3DFMT_NV12, D3DFMT_Q8W8V8U8, D3DFMT_R5G6B5,
    D3DFMT_R8G8B8, D3DFMT_UYVY, D3DFMT_V8U8, D3DFMT_V16U16, D3DFMT_X1R5G5B5, D3DFMT_X8B8G8R8,
    D3DFMT_X8R8G8B8, D3DFMT_YUY2, D3DFMT_YV12, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_TEXTUREFORMAT3,
    D3DFVF_XYZ, D3DLOCK_DISCARD, D3DLOCK_NO_DIRTY_UPDATE, D3DLOCK_READONLY, D3DPOOL_DEFAULT,
    D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND, D3DRS_SRCBLEND, D3DRTYPE_SURFACE, D3DRTYPE_VOLUME,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER,
    D3DSAMP_MIPFILTER, D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR,
    D3DTEXF_NONE, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP,
    D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET,
};

const BLACK: u32 = 0xFF00_0000;

/// A full-backbuffer quad (two triangles) with UVs spanning the unit square.
///
/// White vertex colour so MODULATE passes the texel through.
const fn fullscreen_quad() -> [TexturedVertex; 6] {
    const W: u32 = 0xFFFF_FFFF;
    [
        TexturedVertex {
            x: -1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: 1.0,
        },
        TexturedVertex {
            x: 1.0,
            y: 1.0,
            z: 0.5,
            color: W,
            u: 1.0,
            v: 0.0,
        },
        TexturedVertex {
            x: 1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: 1.0,
            v: 1.0,
        },
        TexturedVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: W,
            u: 0.0,
            v: 1.0,
        },
    ]
}

fn point_clamp(h: &Harness) {
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
}

/// Bind `tex`, sample it across the backbuffer, return the centre pixel.
fn sample_center(h: &Harness, tex: &Texture<'_>) -> Rgba8 {
    sample_at(h, tex, 320, 240)
}

/// Bind `tex`, sample it across the backbuffer, return the pixel at `(x, y)`.
///
/// The quad spans the unit square, so the read point picks which texel the
/// returned pixel came from: a test that asserts on one texel of a texture
/// coarser than the target reads a point well inside that texel's band.
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

/// Bind `tex`, sample it across the backbuffer, return the pixels at `points`.
///
/// One draw feeds every point, so the caller reads several texels of the same
/// sampled image.
fn sample_points<const N: usize>(
    h: &Harness,
    tex: &Texture<'_>,
    points: [(u32, u32); N],
) -> [u32; N] {
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
    points.map(|(x, y)| h.read_pixel(x, y))
}

#[test]
fn create_lock_sample_2x2() {
    let h = Harness::new();
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, 0);
    {
        let mut locked = tex.lock_rect(0, 0);
        assert_eq!(locked.pitch(), 8, "2px * 4 bytes/px row pitch");
        locked.write_u32(&[0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF]);
    }
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    point_clamp(&h);
    h.select_texture_stage(0);
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
            "DrawPrimitiveUP"
        );
    });
    let tl = Rgba8::from_pixel(h.read_pixel(160, 120));
    let tr = Rgba8::from_pixel(h.read_pixel(480, 120));
    let bl = Rgba8::from_pixel(h.read_pixel(160, 360));
    let br = Rgba8::from_pixel(h.read_pixel(480, 360));
    assert!(
        tl.r > 200 && tl.g < 50 && tl.b < 50,
        "top-left red, got {tl:?}"
    );
    assert!(
        tr.r < 50 && tr.g > 200 && tr.b < 50,
        "top-right green, got {tr:?}"
    );
    assert!(
        bl.r < 50 && bl.g < 50 && bl.b > 200,
        "bottom-left blue, got {bl:?}"
    );
    assert!(
        br.r > 200 && br.g > 200 && br.b > 200,
        "bottom-right white, got {br:?}"
    );
}

#[test]
fn sysmem_lock_rect_pitch_is_dword_aligned() {
    let h = Harness::new();
    // A 5×5 `D3DFMT_R5G6B5` (2 bytes/pixel) system-memory surface: the raw row
    // stride is `5 * 2 = 10`, which D3D9 rounds up to the next 4-byte boundary,
    // so `LockRect` must report a pitch of `12` (and a 4-aligned pitch). Some
    // applications depend on the exact value, not just the alignment.
    let surf = h.create_offscreen_plain_surface(5, 5, D3DFMT_R5G6B5, D3DPOOL_SYSTEMMEM);
    let locked = surf.lock_rect(0);
    let pitch = locked.pitch();
    assert_eq!(pitch & 3, 0, "pitch {pitch} must be 4-byte aligned");
    assert_eq!(
        pitch, 12,
        "5×5 R5G6B5 sysmem pitch is 12 (10 rounded up to 4)"
    );
}

/// A system-memory texture level and an offscreen surface answer one pitch.
///
/// D3D9 leaves the row pitch to the driver, but an application that reads it
/// from one system-memory store and steps another by it needs the two to
/// agree, and a 16-bit format at an odd width is where a tight stride and a
/// dword-rounded one part company. The rows written at the reported stride
/// then have to survive `UpdateTexture` and sampling: a level whose upload
/// steps by a different stride shears the image.
#[test]
fn a_sysmem_texture_level_and_surface_share_one_pitch_at_an_odd_16_bit_width() {
    const WIDTH: u32 = 33;
    const HEIGHT: u32 = 4;
    // R5G6B5 opaque red and blue.
    const RED: u16 = 0xF800;
    const BLUE: u16 = 0x001F;
    let h = Harness::new();
    let surf = h.create_offscreen_plain_surface(WIDTH, HEIGHT, D3DFMT_R5G6B5, D3DPOOL_SYSTEMMEM);
    let surface_pitch = surf.lock_rect(0).pitch();
    let src = h.create_texture(WIDTH, HEIGHT, 1, 0, D3DFMT_R5G6B5, D3DPOOL_SYSTEMMEM);
    let level_pitch = src.lock_rect(0, D3DLOCK_READONLY).pitch();
    assert_eq!(
        surface_pitch, 68,
        "{WIDTH} texels of two bytes is 66, reported as 68"
    );
    assert_eq!(
        level_pitch, surface_pitch,
        "a {WIDTH}x{HEIGHT} R5G6B5 level and surface report one pitch"
    );

    // Fill the level row by row at the pitch it reports, the last texel of
    // the last row red and every other one blue. A level whose upload steps
    // by another stride puts that texel somewhere else.
    {
        let locked = src.lock_rect(0, 0);
        let pitch = usize::try_from(locked.pitch()).expect("a positive row pitch");
        let base = locked.bits_ptr();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let texel = if row == HEIGHT as usize - 1 && col == WIDTH as usize - 1 {
                    RED
                } else {
                    BLUE
                };
                let bytes = texel.to_le_bytes();
                // SAFETY: the lock maps `HEIGHT` rows of `pitch` bytes and a
                // row holds `WIDTH` two-byte texels, so `row * pitch + col * 2`
                // is inside the mapped level.
                let texel_ptr = unsafe { base.add(row * pitch + col * 2) };
                // SAFETY: `texel_ptr` addresses two writable bytes of the
                // level (above), disjoint from the stack-local `bytes`.
                unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), texel_ptr, bytes.len()) };
            }
        }
    }
    let dst = h.create_texture(WIDTH, HEIGHT, 1, 0, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    assert_eq!(h.update_texture_hr(&src, &dst), 0, "UpdateTexture");
    // The last texel of the last row covers the back buffer's bottom-right
    // corner under the unit-square quad; the centre is one of the blue ones.
    let [corner, centre] = sample_points(&h, &dst, [(630, 470), (320, 240)]);
    let corner = Rgba8::from_pixel(corner);
    assert!(
        corner.r > 200 && corner.b < 60,
        "the last texel of the last row is red, got {corner:?}"
    );
    let centre = Rgba8::from_pixel(centre);
    assert!(
        centre.b > 200 && centre.r < 60,
        "the level's other texels are blue, got {centre:?}"
    );
}

#[test]
fn color_formats_sample_red() {
    let h = Harness::new();
    // 1×1 opaque-red texel encoded for each format (little-endian bytes).
    let cases: [(u32, &[u8]); 5] = [
        (D3DFMT_X8R8G8B8, &[0x00, 0x00, 0xFF, 0x00]), // BGRX
        (D3DFMT_R5G6B5, &[0x00, 0xF8]),               // R=31
        (D3DFMT_A1R5G5B5, &[0x00, 0xFC]),             // A=1 R=31
        (D3DFMT_X1R5G5B5, &[0x00, 0x7C]),             // X=0 R=31
        (D3DFMT_A4R4G4B4, &[0x00, 0xFF]),             // A=F R=F
    ];
    for (format, bytes) in cases {
        let tex = h.create_texture(1, 1, 1, 0, format, 0);
        tex.lock_rect(0, 0).write(bytes);
        let px = sample_center(&h, &tex);
        assert!(
            px.r > 200 && px.g < 60 && px.b < 60,
            "format {format:#x} red, got {px:?}"
        );
    }
}

/// `X1R5G5B5` samples alpha as 1.0 whatever its top bit holds.
///
/// The bit is padding in D3D9, so a texel that leaves it clear still blends
/// as fully opaque. Reading it as an alpha channel, which is what the
/// `A1R5G5B5` mapping would do, blends the draw away entirely.
#[test]
fn x1r5g5b5_blends_opaque_with_its_top_bit_clear() {
    const BLUE: u32 = 0xFF00_00FF;
    // X=0 R=31: red with the padding bit clear.
    const RED555: u16 = 0x7C00;
    let h = Harness::new();
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
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "blend draw"
        );
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        px.r > 200 && px.b < 60,
        "the texel blends over blue at alpha 1, got {px:?}"
    );
}

/// A `GetDC` on an `X1R5G5B5` level reads and writes it through the 5-5-5 DIB.
///
/// The DIB's `BI_BITFIELDS` masks cover the three colour channels only, so
/// the padding bit is outside anything GDI reads or writes and the level's
/// texels round-trip through it unchanged.
#[test]
fn get_dc_on_an_x1r5g5b5_level_round_trips_a_texel() {
    const GREEN555: u16 = 0x03E0;
    const RED555: u16 = 0x7C00;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_X1R5G5B5, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write(&[GREEN555; 16]);

    let surface = tex.surface_level(0);
    let dc = surface.dc();
    assert_eq!(
        dc.get_pixel(3, 3),
        GREEN_COLORREF,
        "the DC reads the texels the lock wrote"
    );
    assert_eq!(
        dc.set_pixel(3, 3, RED_COLORREF),
        RED_COLORREF,
        "SetPixel stores full-scale channels exactly in a 5-5-5 DIB"
    );
    assert_eq!(dc.release(), 0, "ReleaseDC");

    {
        let locked = tex.lock_rect(0, D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() as usize / 2;
        let texels = locked.as_u16(pitch_px * 4);
        assert_eq!(
            texels[pitch_px * 3 + 3] & 0x7FFF,
            RED555,
            "what GDI drew reached the level's staging"
        );
        assert_eq!(
            texels[0] & 0x7FFF,
            GREEN555,
            "the texels GDI left alone kept the lock's own pixels"
        );
    }

    // The quad spans the unit square over a 640x480 target, so texel (3, 3)
    // covers x 480..640, y 360..480 and texel (0, 0) covers x 0..160,
    // y 0..120.
    let drawn = sample_at(&h, &tex, 560, 420);
    assert!(
        drawn.r > 200 && drawn.g < 60,
        "a draw samples the texel GDI drew, got {drawn:?}"
    );
    let untouched = sample_at(&h, &tex, 80, 60);
    assert!(
        untouched.g > 200 && untouched.r < 60,
        "the texels GDI left alone still sample as the lock wrote them, got {untouched:?}"
    );
}

#[test]
fn luminance_format_samples_gray() {
    let h = Harness::new();
    if h.device_is_paravirtual() {
        // The paravirtual device samples a swizzle view through the base
        // texture's lanes, so the lane this format fills by swizzle reads the
        // stored byte there.
        return;
    }
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_L8, 0);
    tex.lock_rect(0, 0).write::<u8>(&[0x80]);
    let px = sample_center(&h, &tex);
    // L8 replicates luminance across RGB → mid-gray.
    assert!(
        (100..=150).contains(&px.r) && px.r == px.g && px.g == px.b,
        "L8 0x80 → gray, got {px:?}",
    );
}

#[test]
fn dxt1_block_samples_solid_color() {
    let h = Harness::new();
    // One DXT1 block (4×4): both endpoints = red565 (0xF800), all indices 0.
    let tex = h.create_texture(4, 4, 1, 0, D3DFMT_DXT1, 0);
    tex.lock_rect(0, 0)
        .write::<u8>(&[0x00, 0xF8, 0x00, 0xF8, 0x00, 0x00, 0x00, 0x00]);
    let px = sample_center(&h, &tex);
    assert!(
        px.r > 200 && px.g < 60 && px.b < 60,
        "DXT1 solid red, got {px:?}"
    );
}

#[test]
fn mip_chain_levels_and_dimensions() {
    let h = Harness::new();
    // levels = 0 → full chain: 4×4, 2×2, 1×1.
    let tex = h.create_texture(4, 4, 0, 0, D3DFMT_A8R8G8B8, 0);
    assert_eq!(tex.level_count(), 3, "4x4 full mip chain has 3 levels");
    for (level, dim) in [(0u32, 4u32), (1, 2), (2, 1)] {
        let (hr, desc) = tex.level_desc(level);
        assert_eq!(hr, 0, "GetLevelDesc({level})");
        assert_eq!((desc.width, desc.height), (dim, dim), "level {level} dims");
    }
    // A non-zero mip surface is reachable.
    let _surf = tex.surface_level(1);
    // SetLOD is a managed-pool-only control (D3D9 spec); on a DEFAULT-pool
    // texture it is a no-op — it returns the previous LOD (0) and GetLOD stays 0.
    assert_eq!(tex.set_lod(2), 0, "SetLOD returns previous LOD");
    assert_eq!(tex.lod(), 0, "GetLOD stays 0 — LOD clamp is managed-only");
}

#[test]
fn low_mips_below_the_linear_alignment_sample_their_own_texels() {
    // 8x8 A8R8G8B8, full chain: 8, 4, 2 and 1 texels wide, so row pitches of
    // 32, 16, 8 and 4 bytes. The bottom two are below the 16-byte linear
    // texture alignment a blit source must meet on Apple Silicon (every mip
    // below 64 texels wide is, on the 256-byte Mac2 floor), which routes
    // their uploads through the GPU upload pass instead. Each level must
    // still carry exactly the texels written into it.
    let h = Harness::new();
    let tex = h.create_texture(8, 8, 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(tex.level_count(), 4, "8x8 full mip chain");
    let colors = [0xFFFF_0000u32, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF];
    for (level, color) in colors.iter().enumerate() {
        let level = u32::try_from(level).expect("mip index fits u32");
        let side = (8 >> level) as usize;
        let locked = tex.lock_rect(level, 0);
        let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
        let row = vec![*color; side];
        for y in 0..side {
            // SAFETY: the lock maps `side` rows at `pitch` stride.
            let dst = unsafe { locked.bits_ptr().add(y * pitch) };
            // SAFETY: `side` texels fit in the locked row per above.
            unsafe {
                core::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), dst, side * 4);
            }
        }
    }
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for (level, color) in colors.iter().enumerate() {
        let level = u32::try_from(level).expect("mip index fits u32");
        // The quad magnifies, so the most-detailed-level clamp is the level
        // every fragment reads.
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_pixel_eq(
            sample_center(&h, &tex).to_pixel(),
            *color,
            &format!("mip {level}"),
        );
    }
}

#[test]
fn level_desc_reports_surface_type() {
    let h = Harness::new();
    let tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, 0);
    // A texture level is itself a surface: `GetLevelDesc` must report
    // `D3DRTYPE_SURFACE`, not the container's `D3DRTYPE_TEXTURE`.
    let (hr, desc) = tex.level_desc(0);
    assert_eq!(hr, 0, "GetLevelDesc(0)");
    assert_eq!(
        desc.resource_type, D3DRTYPE_SURFACE,
        "level desc Type is D3DRTYPE_SURFACE"
    );
}

#[test]
fn autogen_mipmap_texture_creates() {
    let h = Harness::new();
    let tex = h.create_texture(
        64,
        64,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    // An AUTOGENMIPMAP texture exposes a single app-visible level (the runtime
    // owns the generated chain), and GetLevelDesc reports the texture's usage.
    assert_eq!(tex.level_count(), 1, "autogen texture exposes 1 level");
    let (hr, desc) = tex.level_desc(0);
    assert_eq!(hr, 0, "GetLevelDesc(0)");
    assert_eq!(
        desc.usage, D3DUSAGE_AUTOGENMIPMAP,
        "GetLevelDesc reports AUTOGENMIPMAP usage"
    );
    // AutoGen filter type defaults to LINEAR, rejects D3DTEXF_NONE, and
    // round-trips any other value (Metal's generateMipmaps is fixed-linear, so
    // this is app-visible state only).
    assert_eq!(
        tex.auto_gen_filter_type(),
        D3DTEXF_LINEAR,
        "default autogen filter is LINEAR"
    );
    assert_eq!(
        tex.set_auto_gen_filter_type(D3DTEXF_NONE),
        D3DERR_INVALIDCALL,
        "D3DTEXF_NONE is not a valid autogen filter"
    );
    assert_eq!(
        tex.set_auto_gen_filter_type(D3DTEXF_ANISOTROPIC),
        0,
        "ANISOTROPIC accepted"
    );
    assert_eq!(
        tex.auto_gen_filter_type(),
        D3DTEXF_ANISOTROPIC,
        "autogen filter round-trips"
    );
}

#[test]
fn autogen_mipmap_texture_rejects_sub_level_unlock() {
    let h = Harness::new();
    let tex = h.create_texture(
        64,
        64,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    assert_eq!(tex.level_count(), 1, "autogen texture exposes 1 level");
    // The sub-levels belong to the runtime, so every per-level entry point
    // rejects them, `UnlockRect` included: it must answer the same
    // INVALIDCALL `LockRect` does rather than treat the level as unlocked.
    assert_eq!(
        tex.unlock_rect(1),
        D3DERR_INVALIDCALL,
        "UnlockRect past the app-visible chain"
    );
    // Level zero stays reachable, where an Unlock without a matching Lock is
    // the S_OK case for a texture level.
    assert_eq!(tex.unlock_rect(0), 0, "UnlockRect on the exposed level");
}

#[test]
fn default_pool_texture_lock_splits_by_entry_point() {
    let h = Harness::new();
    // D3D9 makes a `D3DPOOL_DEFAULT` texture without `D3DUSAGE_DYNAMIC`
    // unlockable through either entry point. The level surface answers that
    // contract; `IDirect3DTexture9::LockRect` serves the lock out of the
    // level's CPU staging instead, so a title that streams into a DEFAULT
    // texture it never marked DYNAMIC keeps working.
    let tex = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let (hr, bits_null) = tex.surface_level(0).lock_rect_probe(0);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "surface LockRect on a static DEFAULT level"
    );
    // A rejected lock leaves the struct untouched, so the garbage seed the
    // probe plants is still there and reads as a non-null pointer.
    assert!(
        !bits_null,
        "the rejected surface lock leaves pBits untouched"
    );
    let (hr, bits_null) = tex.lock_rect_probe(0, 0);
    assert_eq!(hr, 0, "texture LockRect serves the same level");
    assert!(!bits_null, "the served lock hands out a pointer");
    assert_eq!(tex.unlock_rect(0), 0, "UnlockRect after the served lock");

    // `D3DUSAGE_DYNAMIC` is what makes the level lockable in D3D9, and both
    // entry points then agree.
    let dynamic = h.create_texture(
        16,
        16,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    let surface = dynamic.surface_level(0);
    let (hr, bits_null) = surface.lock_rect_probe(0);
    assert_eq!(hr, 0, "surface LockRect on a DYNAMIC DEFAULT level");
    assert!(!bits_null, "the DYNAMIC surface lock hands out a pointer");
    assert_eq!(surface.unlock_rect(), 0, "UnlockRect through the surface");
}

#[test]
fn cube_textures_create_in_all_pools() {
    let h = Harness::new();
    for (pool, name) in [
        (D3DPOOL_DEFAULT, "DEFAULT"),
        (D3DPOOL_SCRATCH, "SCRATCH"),
        (D3DPOOL_MANAGED, "MANAGED"),
        (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
    ] {
        assert_eq!(
            h.create_cube_texture(64, 1, 0, D3DFMT_A8R8G8B8, pool),
            0,
            "{name}-pool cube texture creates",
        );
    }
    // Volume (3D) textures are created as `MTLTextureType3D`; the call
    // succeeds (LockBox / binding work; box→texture upload is a follow-up).
    assert_eq!(
        h.create_volume_texture([32, 32, 32], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT),
        0,
        "CreateVolumeTexture succeeds",
    );
}

#[test]
fn volume_texture_pool_usage_and_lock_rules() {
    // DYNAMIC is a DEFAULT/SYSTEMMEM-pool property; a DEFAULT-pool volume is
    // lockable only when DYNAMIC; the other pools always lock.
    let h = Harness::new();
    for (pool, usage, create_hr, lock_hr) in [
        (D3DPOOL_DEFAULT, 0, 0, D3DERR_INVALIDCALL),
        (D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC, 0, 0),
        (D3DPOOL_SYSTEMMEM, 0, 0, 0),
        (D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC, 0, 0),
        (D3DPOOL_MANAGED, 0, 0, 0),
        (D3DPOOL_MANAGED, D3DUSAGE_DYNAMIC, D3DERR_INVALIDCALL, 0),
        (D3DPOOL_SCRATCH, 0, 0, 0),
        (D3DPOOL_SCRATCH, D3DUSAGE_DYNAMIC, D3DERR_INVALIDCALL, 0),
    ] {
        let (hr, texture) = h.try_create_volume_texture([4, 4, 4], 1, usage, D3DFMT_A8R8G8B8, pool);
        assert_eq!(hr, create_hr, "create pool={pool} usage={usage:#x}");
        let Some(texture) = texture else {
            continue;
        };
        let (hr, bits_null) = texture.lock_box_probe(0, 0);
        assert_eq!(hr, lock_hr, "lock pool={pool} usage={usage:#x}");
        if lock_hr == 0 {
            assert!(!bits_null, "a successful lock hands out a pointer");
            assert_eq!(texture.unlock_box(0), 0, "unlock pool={pool}");
        } else {
            assert!(bits_null, "a rejected lock leaves pBits null");
        }
    }
}

#[test]
fn volume_texture_level_desc_walks_the_chain() {
    let h = Harness::new();
    let (hr, texture) =
        h.try_create_volume_texture([2, 4, 8], 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0);
    let texture = texture.expect("volume texture");
    let (hr, desc) = texture.level_desc(0);
    assert_eq!(hr, 0, "GetLevelDesc(0)");
    assert_eq!(desc.resource_type, D3DRTYPE_VOLUME);
    assert_eq!((desc.width, desc.height, desc.depth), (2, 4, 8));
    assert_eq!(desc.format, D3DFMT_A8R8G8B8);
    assert_eq!(desc.pool, D3DPOOL_SYSTEMMEM);
    assert_eq!(desc.usage, 0);
    let (hr, desc) = texture.level_desc(2);
    assert_eq!(hr, 0, "GetLevelDesc(2)");
    assert_eq!((desc.width, desc.height, desc.depth), (1, 1, 2));
    let (hr, _) = texture.level_desc(4);
    assert_eq!(hr, D3DERR_INVALIDCALL, "level past the chain");
}

#[test]
fn scratch_extension_cubes_are_cpu_only_resources() {
    let h = Harness::new();
    for format in [D3DFMT_ATI1, D3DFMT_YUY2, D3DFMT_UYVY] {
        assert_eq!(
            h.create_cube_texture(4, 1, 0, format, D3DPOOL_SCRATCH),
            0,
            "SCRATCH extension cube creates",
        );
        assert_eq!(
            h.create_cube_texture(4, 1, 0, format, D3DPOOL_DEFAULT),
            D3DERR_INVALIDCALL,
            "GPU extension cube remains unsupported",
        );
    }
}

/// `YV12` and `NV12` create as `D3DPOOL_DEFAULT` offscreen plains and as nothing else.
///
/// A planar level holds its chroma rows after its luma rows, so it is larger
/// than pitch times height. Only the default-pool plain is sized for that:
/// every CPU pool, every texture type in every pool, and the render-target
/// and depth-stencil creates are refused with the out pointer nulled. A
/// `YV12` plain of odd height is refused too, its U plane having no agreed
/// origin, as is an extent whose luma and chroma rows pass the texture limit.
#[test]
fn planar_yuv_creates_only_as_a_default_pool_offscreen_plain() {
    let h = Harness::new();
    let pools = [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ];
    for (format, name) in [(D3DFMT_YV12, "YV12"), (D3DFMT_NV12, "NV12")] {
        for (width, height) in [(20, 16), (21, 16), (22, 2)] {
            let (hr, out) =
                h.create_offscreen_plain_surface_seeded(width, height, format, D3DPOOL_DEFAULT);
            assert_eq!(hr, 0, "{name} {width}x{height} DEFAULT offscreen plain");
            assert!(
                !out.is_null(),
                "{name} {width}x{height} hands back a surface"
            );
        }
        let odd = h.create_offscreen_plain_surface_seeded(20, 15, format, D3DPOOL_DEFAULT);
        if format == D3DFMT_NV12 {
            assert_eq!(odd.0, 0, "NV12 rounds an odd height's chroma rows up");
        } else {
            assert_eq!(
                odd,
                (D3DERR_INVALIDCALL, core::ptr::null_mut()),
                "YV12 of odd height is refused with the out pointer nulled"
            );
        }
        // 10924 luma rows and 5462 chroma rows are 16386 rows of storage.
        assert_eq!(
            h.create_offscreen_plain_surface_seeded(16, 10924, format, D3DPOOL_DEFAULT),
            (D3DERR_INVALIDCALL, core::ptr::null_mut()),
            "{name}: storage rows past the texture limit"
        );
        for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
            assert_eq!(
                h.create_offscreen_plain_surface_seeded(20, 16, format, pool),
                (D3DERR_INVALIDCALL, core::ptr::null_mut()),
                "{name} offscreen plain in pool {pool}"
            );
        }
        for pool in pools {
            assert_eq!(
                h.try_create_texture(16, 16, 1, 0, format, pool),
                (D3DERR_INVALIDCALL, core::ptr::null_mut()),
                "{name} texture in pool {pool}"
            );
            assert_eq!(
                h.try_create_cube_texture(16, 1, 0, format, pool),
                (D3DERR_INVALIDCALL, core::ptr::null_mut()),
                "{name} cube texture in pool {pool}"
            );
            let (hr, volume) = h.try_create_volume_texture([16, 16, 2], 1, 0, format, pool);
            assert_eq!(
                hr, D3DERR_INVALIDCALL,
                "{name} volume texture in pool {pool}"
            );
            assert!(volume.is_none(), "{name} volume texture in pool {pool}");
        }
        for usage in [D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET] {
            assert_eq!(
                h.try_create_texture(16, 16, 1, usage, format, D3DPOOL_DEFAULT),
                (D3DERR_INVALIDCALL, core::ptr::null_mut()),
                "{name} texture with usage {usage:#x}"
            );
        }
        assert_eq!(
            h.create_render_target_hr(16, 16, format),
            D3DERR_INVALIDCALL,
            "{name} render target"
        );
        let (hr, depth) = h.create_depth_stencil_surface_ms_hr((16, 16), format, (0, 0));
        assert_eq!(hr, D3DERR_INVALIDCALL, "{name} depth-stencil surface");
        assert!(depth.is_none(), "{name} depth-stencil surface");
    }
}

/// A `D3DPOOL_SYSTEMMEM` texture is created CPU-side and samples once bound.
///
/// The pool allocates no Metal texture, so a texture an application only locks
/// or copies from never reaches the GPU. D3D9 does sample one that is bound at
/// a texture stage, so the bind is what gives it a Metal texture, carrying
/// everything written by then; a later write reaches it like any other.
#[test]
fn systemmem_texture_samples_once_bound_at_a_stage() {
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    tex.lock_rect(0, 0).write_u32(&[GREEN; 4]);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        GREEN,
        "the first bind uploads what was already written",
    );
    tex.lock_rect(0, 0).write_u32(&[BLUE; 4]);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        BLUE,
        "a later write reaches the same texture",
    );
}

/// A `D3DPOOL_SYSTEMMEM` texture reads back through `LockRect` what it wrote.
///
/// The pool's whole contract: the texels live in system memory, so a second
/// lock sees the first lock's write with no device involved.
#[test]
fn systemmem_texture_lock_roundtrips_its_texels() {
    const TEXELS: [u32; 4] = [0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF];
    let h = Harness::new();
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    tex.lock_rect(0, 0).write_u32(&TEXELS);
    let locked = tex.lock_rect(0, D3DLOCK_READONLY);
    assert_eq!(locked.pitch(), 8, "2 px * 4 bytes/px row pitch");
    assert_eq!(
        locked.as_u32(TEXELS.len()),
        &TEXELS[..],
        "the second lock reads the first lock's write",
    );
}

/// A `D3DPOOL_SYSTEMMEM` texture keeps its texels across a `Reset`.
///
/// It neither blocks the reset nor is lost by it, so it is still a usable
/// `UpdateTexture` source afterwards.
#[test]
fn systemmem_texture_survives_reset() {
    const RED: u32 = 0xFFFF_0000;
    let h = Harness::new();
    let src = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write_u32(&[RED; 4]);
    assert_eq!(h.reset(512, 384), 0, "resize Reset with the texture alive");
    {
        let locked = src.lock_rect(0, D3DLOCK_READONLY);
        assert_eq!(locked.as_u32(4), &[RED; 4][..], "texels survive the Reset");
    }
    assert_eq!(h.reset(640, 480), 0, "Reset back to the sampling size");
    let dst = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        0,
        "UpdateTexture after the Reset"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        RED,
        "the post-Reset copy reaches the GPU",
    );
}

/// A block-compressed volume texture exists only as a `D3DPOOL_SCRATCH` resource.
///
/// Metal has no 3D block-compressed texture, so a DXT volume is representable
/// at all only because the scratch pool allocates none: it is created, locked
/// and written entirely CPU-side. Every GPU-resident pool rejects the format.
#[test]
fn scratch_block_compressed_volume_is_cpu_only() {
    let h = Harness::new();
    let (hr, volume) = h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_DXT1, D3DPOOL_SCRATCH);
    assert_eq!(hr, 0, "SCRATCH DXT1 volume creates");
    let volume = volume.expect("SCRATCH DXT1 volume");
    let (hr, bits_null) = volume.lock_box_probe(0, 0);
    assert_eq!(hr, 0, "LockBox on a scratch volume");
    assert!(!bits_null, "a successful lock hands out a pointer");
    assert_eq!(volume.unlock_box(0), 0, "UnlockBox");
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
        let (hr, _) = h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_DXT1, pool);
        assert_eq!(hr, D3DERR_INVALIDCALL, "DXT1 volume in pool {pool}");
    }
    // The device still renders after the scratch volume has been created.
    h.render_once(BLACK, |_| {});
    assert_eq!(h.read_pixel(320, 240), BLACK, "the frame after the create");
}

/// No system-memory surface can be a render target.
///
/// `SetRenderTarget` requires a destination carrying `D3DUSAGE_RENDERTARGET`,
/// which neither CPU-only pool can, whether the surface is a standalone
/// offscreen-plain one or a texture level.
#[test]
fn system_memory_surfaces_are_rejected_as_render_targets() {
    let h = Harness::new();
    for pool in [D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        let surface = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, pool);
        assert_eq!(
            h.set_render_target(0, &surface),
            D3DERR_INVALIDCALL,
            "offscreen-plain surface in pool {pool}",
        );
    }
    let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.set_render_target(0, &tex.surface_level(0)),
        D3DERR_INVALIDCALL,
        "SYSTEMMEM texture level",
    );
}

/// A format that is not colour-renderable cannot be created as a render target.
///
/// `CheckDeviceFormat` answers `NOTAVAILABLE` for every one of these next to
/// `D3DUSAGE_RENDERTARGET`, so both creates that can carry the usage have to
/// refuse the caller that skipped the probe rather than hand back a surface
/// nothing can draw into. The cost of leniency is not a wrong pixel: Metal
/// refuses a colour attachment it cannot render into with an abort, so a
/// block-compressed render target takes the process down at the first draw, and
/// `A8`, whose Metal twin is sampleable only, takes it down inside texture
/// creation before any draw is reached.
#[test]
fn creates_reject_a_render_target_in_a_non_renderable_format() {
    let h = Harness::new();
    for format in [
        D3DFMT_DXT1,
        D3DFMT_DXT5,
        D3DFMT_ATI1,
        D3DFMT_A8,
        D3DFMT_L8,
        D3DFMT_V8U8,
        D3DFMT_A4R4G4B4,
    ] {
        let (hr, ptr) =
            h.try_create_texture(64, 64, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "CreateTexture(format={format:#x}, RENDERTARGET)"
        );
        assert!(ptr.is_null(), "format {format:#x} returned a texture");
        assert_eq!(
            h.create_render_target_hr(64, 64, format),
            D3DERR_INVALIDCALL,
            "CreateRenderTarget(format={format:#x})"
        );
    }
}

#[test]
fn managed_cube_autogen_creates() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        64,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    assert_eq!(cube.level_count(), 1, "autogen exposes only level zero");
}

#[test]
fn autogen_mipmap_cube_rejects_sub_level_surface() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(
        64,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    assert_eq!(cube.level_count(), 1, "autogen exposes only level zero");
    // Level zero is the one face surface an application can hold; `surface`
    // fails the test if the call it wraps is rejected.
    let level_zero = cube.surface(0, 0);
    let (hr, _) = level_zero.desc();
    assert_eq!(hr, 0, "the exposed face surface describes");
    // A sub-level face is driver-owned: handing one out would let an
    // application bind a level `GetLevelCount` says is not there as a render
    // target, under the generated chain AUTOGENMIPMAP owns.
    let (hr, surface) = cube.try_surface(0, 1);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "GetCubeMapSurface past the app-visible chain"
    );
    assert!(surface.is_null(), "a rejected call leaves the slot null");
}

#[test]
fn cube_render_target_faces_generate_mips_independently() {
    let h = Harness::new();
    let backbuffer = h.render_target(0);
    let cube = h.create_cube_texture_owned(
        64,
        0,
        D3DUSAGE_RENDERTARGET | D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(cube.level_count(), 1, "autogen exposes only level zero");

    let positive_x = cube.surface(0, 0);
    assert_eq!(h.set_render_target(0, &positive_x), 0);
    assert_eq!(h.clear_target(0xFFFF_0000), 0);

    let negative_x = cube.surface(1, 0);
    assert_eq!(h.set_render_target(0, &negative_x), 0);
    assert_eq!(h.clear_target(0xFF00_FF00), 0);
    assert_eq!(h.set_render_target(0, &backbuffer), 0);

    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 2), 0);
    assert_pixel_eq(
        sample_cube_x(&h, &cube, 1.0),
        0xFFFF_0000,
        "generated positive-X mip",
    );
    assert_pixel_eq(
        sample_cube_x(&h, &cube, -1.0),
        0xFF00_FF00,
        "generated negative-X mip",
    );
}

#[test]
fn managed_dxt_cube_keeps_faces_independent() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_DXT1, D3DPOOL_MANAGED);
    assert_eq!(cube.level_count(), 1);

    let face0 = [0x11u8; 8];
    let face1 = [0x77u8; 8];
    {
        let mut lock = cube.lock_rect(0, 0, 0);
        lock.write(&face0);
    }
    {
        let mut lock = cube.lock_rect(1, 0, 0);
        lock.write(&face1);
    }
    {
        let lock = cube.lock_rect(0, 0, mtld3d_types::D3DLOCK_READONLY);
        // SAFETY-free byte view through the existing typed lock helper.
        assert_eq!(lock.as_u32(2), &[0x1111_1111; 2]);
    }
    {
        let lock = cube.lock_rect(1, 0, mtld3d_types::D3DLOCK_READONLY);
        assert_eq!(lock.as_u32(2), &[0x7777_7777; 2]);
    }

    let face2 = cube.surface(2, 0);
    {
        let mut lock = face2.lock_rect(0);
        lock.write(&[0xA5u8; 8]);
    }
    {
        let lock = cube.lock_rect(2, 0, mtld3d_types::D3DLOCK_READONLY);
        assert_eq!(lock.as_u32(2), &[0xA5A5_A5A5; 2]);
    }
}

/// A DXT cube uploads one block row per one-block face level and samples each level.
///
/// A 4x4 face is a single block whose row pitch sits under the linear texture
/// alignment on either GPU family for DXT1, so every level takes the padded
/// copy. That copy reads whole rows and is declined when the rows asked for run
/// past the level's staging, which four texel rows of a one-block level would.
#[test]
fn managed_dxt_cube_samples_every_level() {
    const RED: u16 = 0xF800;
    const GREEN: u16 = 0x07E0;
    const BLUE: u16 = 0x001F;
    const WHITE: u16 = 0xFFFF;
    const fn argb(color565: u16) -> u32 {
        match color565 {
            RED => 0xFFFF_0000,
            GREEN => 0xFF00_FF00,
            BLUE => 0xFF00_00FF,
            _ => 0xFFFF_FFFF,
        }
    }
    let h = Harness::new();
    // Positive-X and negative-X colours per level; no two neighbours in face
    // or level share one, so an exchanged face or level reads differently.
    let faces = [[RED, GREEN], [BLUE, WHITE], [GREEN, RED]];
    for format in [D3DFMT_DXT1, D3DFMT_DXT5] {
        let cube = h.create_cube_texture_owned(4, 0, 0, format, D3DPOOL_MANAGED);
        assert_eq!(cube.level_count(), 3, "format {format:#x}");
        for (level, colors) in (0u32..).zip(faces) {
            for (face, color) in (0u32..).zip(colors) {
                let mut block = if format == D3DFMT_DXT5 {
                    // Two equal opaque alpha endpoints, every alpha index 0.
                    vec![0xFF, 0xFF, 0, 0, 0, 0, 0, 0]
                } else {
                    Vec::new()
                };
                block.extend_from_slice(&dxt1_solid_block(color));
                cube.lock_rect(face, level, 0).write(block.as_slice());
            }
        }
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        for (level, colors) in (0u32..).zip(faces) {
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
            for (direction_x, color) in [1.0, -1.0].into_iter().zip(colors) {
                assert_pixel_approx(
                    sample_cube_x(&h, &cube, direction_x),
                    argb(color),
                    1,
                    &format!("format {format:#x} level {level} direction {direction_x}"),
                );
            }
        }
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE), 0);
    }
}

#[test]
fn state_block_restores_cube_binding() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(h.set_cube_texture(0, &cube), 0);

    // Texture bindings are D3DSBT_ALL state. A filtered PIXELSTATE block
    // captures texture *stage* states and sampler states, never the
    // SetTexture bindings themselves, so applying one must leave the stage
    // as it is.
    let pixel = h.create_state_block(mtld3d_types::D3DSBT_PIXELSTATE);
    let all = h.create_state_block(mtld3d_types::D3DSBT_ALL);
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(pixel.apply(), 0);
    assert!(
        h.texture_matches_raw(0, core::ptr::null_mut()),
        "a PIXELSTATE apply must not restore texture bindings",
    );
    assert_eq!(all.apply(), 0);
    assert!(
        h.texture_matches_raw(0, cube.as_ptr()),
        "a D3DSBT_ALL apply restores the cube binding",
    );
    assert_eq!(h.clear_texture(0), 0);
}

#[repr(C)]
struct CubeVertex {
    x: f32,
    y: f32,
    z: f32,
    color: u32,
    u: f32,
    v: f32,
    w: f32,
}

const fn cube_vertex(x: f32, y: f32, direction_x: f32) -> CubeVertex {
    CubeVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: direction_x,
        v: 0.0,
        w: 0.0,
    }
}

fn sample_cube_x(h: &Harness, cube: &mtld3d_tests::CubeTexture<'_>, direction_x: f32) -> u32 {
    assert_eq!(h.set_cube_texture(0, cube), 0);
    h.select_texture_stage(0);
    point_clamp(h);
    // D3DFVF_TEXCOORDSIZE3(0) is bit 16.
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | 0x0001_0000),
        0
    );
    let quad = [
        cube_vertex(-1.0, 1.0, direction_x),
        cube_vertex(1.0, 1.0, direction_x),
        cube_vertex(-1.0, -1.0, direction_x),
        cube_vertex(1.0, 1.0, direction_x),
        cube_vertex(1.0, -1.0, direction_x),
        cube_vertex(-1.0, -1.0, direction_x),
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let pixel = h.read_pixel(320, 240);
    assert_eq!(h.clear_texture(0), 0);
    pixel
}

#[test]
fn fixed_function_cube_sampling_uses_direction_coordinates() {
    let h = Harness::new();
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    {
        let mut face = cube.lock_rect(0, 0, 0);
        face.write_u32(&[0xFFFF_0000; 16]);
    }
    assert_pixel_eq(
        sample_cube_x(&h, &cube, 1.0),
        0xFFFF_0000,
        "fixed-function cube sample",
    );
}

#[test]
fn update_surface_uploads_the_selected_cube_face() {
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[0xFF00_FF00; 16]);
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let negative_x = cube.surface(1, 0);
    assert_eq!(h.update_surface_hr(&src, &negative_x), 0);
    assert_pixel_eq(
        sample_cube_x(&h, &cube, -1.0),
        0xFF00_FF00,
        "UpdateSurface destination cube face",
    );
}

/// `UpdateSurface` copies a system-memory 2D level into one cube face level.
///
/// The source's level zero and the destination's other face carry different
/// colours, so the rendered result pins both endpoint selections independently.
#[test]
fn update_surface_copies_a_2d_level_into_a_cube_face() {
    const BLUE: u32 = 0xFF00_00FF;
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    let h = Harness::new();
    let src = h.create_texture(8, 8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u32>(&[BLUE; 64]);
    src.lock_rect(1, 0).write::<u32>(&[GREEN; 16]);

    let dst = h.create_cube_texture_owned(8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let control = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    control.lock_rect(0).write_u32(&[RED; 16]);
    assert_eq!(
        h.update_surface_hr(&control, &dst.surface(0, 1)),
        0,
        "standalone source into positive-X mip 1"
    );
    assert_eq!(
        h.update_surface_hr(&src.surface_level(1), &dst.surface(1, 1)),
        0,
        "2D mip 1 into negative-X mip 1"
    );

    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    assert_pixel_eq(sample_cube_x(&h, &dst, 1.0), RED, "positive-X control face");
    assert_pixel_eq(
        sample_cube_x(&h, &dst, -1.0),
        GREEN,
        "negative-X updated face",
    );
}

/// `UpdateSurface` copies one system-memory cube face level into a 2D level.
///
/// Other source subresources and destination level zero carry different
/// colours, so the rendered result pins the chosen face and both mip levels.
#[test]
fn update_surface_copies_a_cube_face_into_a_2d_level() {
    const BLUE: u32 = 0xFF00_00FF;
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    const WHITE: u32 = 0xFFFF_FFFF;
    let h = Harness::new();
    let src = h.create_cube_texture_owned(8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 1, 0).write_u32(&[BLUE; 16]);
    src.lock_rect(1, 0, 0).write_u32(&[WHITE; 64]);
    src.lock_rect(1, 1, 0).write_u32(&[GREEN; 16]);

    let dst = h.create_texture(8, 8, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let control = h.create_texture(8, 8, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    control.lock_rect(0, 0).write::<u32>(&[RED; 64]);
    assert_eq!(
        h.update_surface_hr(&control.surface_level(0), &dst.surface_level(0)),
        0,
        "2D control into destination mip 0"
    );
    assert_eq!(
        h.update_surface_hr(&src.surface(1, 1), &dst.surface_level(1)),
        0,
        "negative-X mip 1 into 2D mip 1"
    );

    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        RED,
        "destination mip 0 control",
    );
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        GREEN,
        "destination mip 1 update",
    );
}

/// A whole cube-face update preserves the content captured by earlier draws.
///
/// The updated face is not face zero, and the other face and mip stay readable
/// after the rename. Both linear and sRGB bindings must retain their old storage.
#[test]
fn intra_frame_cube_update_keeps_per_draw_content() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    const YELLOW: u32 = 0xFFFF_FF00;
    let h = Harness::new();
    for (size, srgb) in [(2, 0), (2, 1), (64, 0), (64, 1)] {
        let texels = usize::try_from(size * size).expect("small cube face");
        let mip_texels = texels / 4;
        let cube = h.create_cube_texture_owned(size, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        let source =
            h.create_offscreen_plain_surface(size, size, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
        let mip_source = h.create_offscreen_plain_surface(
            size / 2,
            size / 2,
            D3DFMT_A8R8G8B8,
            D3DPOOL_SYSTEMMEM,
        );
        mip_source.lock_rect(0).write_u32(&vec![YELLOW; mip_texels]);
        for face in 0..6 {
            let color = if face == 1 { RED } else { GREEN };
            source.lock_rect(0).write_u32(&vec![color; texels]);
            assert_eq!(h.update_surface_hr(&source, &cube.surface(face, 0)), 0);
            assert_eq!(h.update_surface_hr(&mip_source, &cube.surface(face, 1)), 0);
        }
        assert_eq!(
            h.set_sampler_state(0, mtld3d_types::D3DSAMP_SRGBTEXTURE, srgb),
            0
        );
        assert_pixel_eq(sample_cube_x(&h, &cube, -1.0), RED, "primed face");
        assert_eq!(h.set_cube_texture(0, &cube), 0);
        source.lock_rect(0).write_u32(&vec![BLUE; texels]);
        let face = cube.surface(1, 0);
        let quad = |left, right| {
            [
                cube_vertex(left, 1.0, -1.0),
                cube_vertex(right, 1.0, -1.0),
                cube_vertex(left, -1.0, -1.0),
                cube_vertex(right, 1.0, -1.0),
                cube_vertex(right, -1.0, -1.0),
                cube_vertex(left, -1.0, -1.0),
            ]
        };
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(-1.0, 0.0)),
                0
            );
            assert_eq!(h.update_surface_hr(&source, &face), 0);
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(0.0, 1.0)),
                0
            );
        });
        let pixels = [h.read_pixel(160, 240), h.read_pixel(480, 240)];
        assert_eq!(
            pixels,
            [RED, BLUE],
            "cube draws bracketing UpdateSurface, size={size}, sRGB={srgb}"
        );
        assert_pixel_eq(sample_cube_x(&h, &cube, 1.0), GREEN, "untouched face");
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
        assert_pixel_eq(sample_cube_x(&h, &cube, -1.0), YELLOW, "untouched mip");
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE), 0);
    }
}

/// Run the frame that releases the staging of the levels uploaded so far.
///
/// A level of the class that releases its staging after an upload keeps it
/// until the encoder answers that the upload reached the command stream, and
/// the answer is acted on at the next `Present`. A test that wants a released
/// level takes one more frame after the draw that uploaded it.
fn release_uploaded_staging(h: &Harness) {
    assert_eq!(h.present(), 0, "the Present that releases uploaded staging");
}

/// Upload answers ignore released textures and preserve writes made after the upload.
///
/// Readback retires every upload without a Present, leaving their answers
/// queued together. Half the textures are freed before that queue is drained;
/// the survivors and newly allocated textures hold bytes the GPU has not seen.
#[test]
fn upload_answers_skip_released_textures_and_keep_new_writes() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let mut uploaded = Vec::new();
    for _ in 0..8 {
        let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        tex.lock_rect(0, 0).write_u32(&[RED; 16]);
        let quad = bind_for_quadrant_draws(&h, &tex);
        assert_eq!(h.begin_scene(), 0);
        assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        assert_eq!(h.end_scene(), 0);
        assert_pixel_eq(h.read_pixel(320, 240), RED, "the upload reached the GPU");
        uploaded.push(tex);
    }
    assert_eq!(h.clear_texture(0), 0);
    uploaded.truncate(4);
    for tex in &uploaded {
        tex.lock_rect(0, 0).write_u32(&[GREEN; 16]);
    }
    let fresh: Vec<_> = (0..4)
        .map(|_| {
            let tex = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
            tex.lock_rect(0, 0).write_u32(&[BLUE; 16]);
            tex
        })
        .collect();

    release_uploaded_staging(&h);
    for tex in &uploaded {
        assert_pixel_eq(
            sample_center(&h, tex).to_pixel(),
            GREEN,
            "an earlier upload answer keeps a later write",
        );
    }
    for tex in &fresh {
        assert_pixel_eq(
            sample_center(&h, tex).to_pixel(),
            BLUE,
            "a released texture's answer cannot consume a new texture's staging",
        );
    }
}

/// A default-pool texture the game cannot lock takes a second `UpdateTexture`.
///
/// Its staging goes away once the first upload has been emitted (the GPU holds
/// the only copy, as on real D3D9); the second update re-creates it, and what
/// samples back is the second fill.
#[test]
fn default_pool_texture_takes_a_second_update_after_its_upload() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let src = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write::<u32>(&[RED; 4]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0, "first UpdateTexture");
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), RED, "first fill");
    release_uploaded_staging(&h);

    src.lock_rect(0, 0).write::<u32>(&[GREEN; 4]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0, "second UpdateTexture");
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "second fill");
}

/// Fill `w` x `h` texels of a locked sub-rect with `color`, honouring its row pitch.
fn fill_locked_rect(locked: &LockedRect<'_>, w: usize, h: usize, color: u32) {
    let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
    let base = locked.bits_ptr();
    let row = vec![color; w];
    for y in 0..h {
        // SAFETY: the lock maps `h` rows of at least `w` texels at `base` with
        // `pitch` row stride, so row `y` starts inside the mapping.
        let dst = unsafe { base.add(y * pitch) };
        // SAFETY: `w` texels are 4 bytes each and fit in the locked row above.
        unsafe { core::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), dst, w * 4) };
    }
}

/// Bind `tex` for point-sampled full-screen draws and hand back the quad.
///
/// Split out of [`sample_quadrants`] for the tests that lock between two
/// draws of one scene.
fn bind_for_quadrant_draws(h: &Harness, tex: &Texture<'_>) -> [TexturedVertex; 6] {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    fullscreen_quad()
}

/// The four quadrant centres of the back buffer, in the order [`QUADRANTS`] lists them.
fn read_quadrants(h: &Harness) -> [u32; 4] {
    [
        h.read_pixel(160, 120),
        h.read_pixel(480, 120),
        h.read_pixel(480, 360),
        h.read_pixel(160, 360),
    ]
}

/// Bind `tex`, sample it across the backbuffer, return the four quadrant centres.
///
/// Clockwise from the top left, the order [`QUADRANTS`] lists them in.
fn sample_quadrants(h: &Harness, tex: &Texture<'_>) -> [u32; 4] {
    let quad = bind_for_quadrant_draws(h, tex);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "sample draw"
        );
    });
    read_quadrants(h)
}

/// Draw `tex`, run `lock_and_write` while that draw's upload is in flight, draw again.
///
/// The lock lands between two draws of one scene, so the level's previous
/// upload has been submitted and not retired: the shape in which a Lock
/// renames its staging instead of writing in place. What the second draw
/// samples is what the back buffer holds afterwards.
fn lock_under_in_flight_upload(
    h: &Harness,
    tex: &Texture<'_>,
    lock_and_write: impl FnOnce(),
) -> [u32; 4] {
    let quad = bind_for_quadrant_draws(h, tex);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "draw before the lock"
        );
        lock_and_write();
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "draw after the lock"
        );
    });
    read_quadrants(h)
}

/// The four quadrants of a 64x64 level: colour, source rect, name.
const QUADRANTS: [(u32, [i32; 4], &str); 4] = [
    (0xFFFF_0000, [0, 0, 32, 32], "top left"),
    (0xFF00_FF00, [32, 0, 64, 32], "top right"),
    (0xFF00_00FF, [32, 32, 64, 64], "bottom right"),
    (0xFFFF_FF00, [0, 32, 32, 64], "bottom left"),
];

/// Partial updates that together cover a default-pool level leave it whole.
///
/// Each `UpdateTexture` carries one quadrant, so no single upload covers the
/// level and the staging is only released once the four of them do. What
/// samples back afterwards must still be the four quadrants, and a fifth
/// partial update landing after the release must reach the GPU without
/// disturbing the three quadrants it does not touch.
#[test]
fn partial_updates_covering_a_default_pool_level_keep_its_pixels() {
    const SIZE: u32 = 64;
    const BASE: u32 = 0xFF80_8080;
    const REPAINT: u32 = 0xFF00_FFFF;
    let h = Harness::new();
    let src = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0)
        .write_u32(&[BASE; (SIZE * SIZE) as usize]);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        0,
        "whole-level UpdateTexture"
    );
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), BASE, "base fill");

    for (color, rect, name) in QUADRANTS {
        {
            let locked = src.lock_rect_partial(0, &rect, D3DLOCK_NO_DIRTY_UPDATE);
            fill_locked_rect(&locked, 32, 32, color);
        }
        assert_eq!(src.add_dirty_rect_partial(&rect), 0, "AddDirtyRect {name}");
        assert_eq!(h.update_texture_hr(&src, &dst), 0, "UpdateTexture {name}");
    }
    let sampled = sample_quadrants(&h, &dst);
    for (i, (color, _, name)) in QUADRANTS.into_iter().enumerate() {
        assert_pixel_eq(sampled[i], color, name);
    }

    release_uploaded_staging(&h);

    // The four writes covered the level, so its staging is gone. A fifth
    // partial update re-creates it and must upload only its own rect: the
    // pixels the GPU already holds are the only copy of the other three.
    let (_, rect, name) = QUADRANTS[0];
    {
        let locked = src.lock_rect_partial(0, &rect, D3DLOCK_NO_DIRTY_UPDATE);
        fill_locked_rect(&locked, 32, 32, REPAINT);
    }
    assert_eq!(src.add_dirty_rect_partial(&rect), 0, "AddDirtyRect repaint");
    assert_eq!(h.update_texture_hr(&src, &dst), 0, "UpdateTexture repaint");
    let repainted = sample_quadrants(&h, &dst);
    assert_pixel_eq(repainted[0], REPAINT, name);
    for (i, (color, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(repainted[i], color, name);
    }
}

/// A sub-rectangle `UpdateSurface` leaves the rest of the destination level alone.
///
/// The destination is a default-pool texture whose staging is released once
/// its first whole-level upload has been emitted, so the partial update
/// re-creates a staging buffer holding only the copied rectangle. Uploading
/// the whole mip from it would push uninitialised pages over the GPU content
/// the copy never touched.
#[test]
fn update_surface_sub_rect_keeps_the_rest_of_the_level() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let level = dst.surface_level(0);

    let whole = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    whole.lock_rect(0).write_u32(&[GREEN; 16]);
    assert_eq!(
        h.update_surface_hr(&whole, &level),
        0,
        "whole-level UpdateSurface"
    );
    // The sampling draw submits the whole-level upload; the frame after it
    // releases the destination's staging.
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        GREEN,
        "whole-level fill",
    );
    release_uploaded_staging(&h);

    let patch = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    patch.lock_rect(0).write_u32(&[RED; 16]);
    let rect = D3DRECT {
        x1: 0,
        y1: 0,
        x2: 2,
        y2: 2,
    };
    assert_eq!(
        h.update_surface_region_hr(&patch, &rect, &level, (0, 0)),
        0,
        "sub-rect UpdateSurface"
    );

    // The 4x4 level spans the 640x480 backbuffer, so each of these reads one
    // texel: (0,0) inside the copied rectangle, (3,0) and (0,3) outside it.
    let [inside, right_of_rect, below_rect] =
        sample_points(&h, &dst, [(80, 60), (560, 60), (80, 420)]);
    assert_pixel_eq(inside, RED, "texel inside the updated rectangle");
    assert_pixel_eq(right_of_rect, GREEN, "texel right of the updated rectangle");
    assert_pixel_eq(below_rect, GREEN, "texel below the updated rectangle");
}

/// A plain lock of a released default-pool level hands back the level's texels.
///
/// The whole-level write reaches the GPU at the next draw and the staging goes
/// once that upload is answered, so a second lock has nothing left in system
/// memory. D3D9 promises
/// that lock the level's current contents, which leaves reading them back from
/// the GPU as the only honest answer; the pages alone read as garbage.
#[test]
fn lock_of_a_released_default_pool_level_reads_the_level_back() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    // One distinct texel per position, so an uninitialised page cannot pass.
    let written: Vec<u32> = (0..SIZE * SIZE).map(|i| 0xFF00_0000 | i).collect();
    {
        let mut locked = tex.lock_rect(0, 0);
        assert_eq!(locked.pitch(), 256, "64 texels * 4 bytes/texel row pitch");
        locked.write_u32(&written);
    }
    // The draw is what uploads the level; which texel the centre sample lands
    // on is not what this pins.
    let _sampled = sample_center(&h, &tex);
    release_uploaded_staging(&h);

    let locked = tex.lock_rect(0, 0);
    assert_eq!(
        locked.as_u32(TEXELS),
        written.as_slice(),
        "the lock reads the texels the level holds"
    );
}

/// `GetDC` on a released default-pool level maps the level's own texels.
///
/// The draw uploads the level and the staging goes once that upload is
/// answered, so the pixels live on the GPU alone and the slot points at the one
/// page every released level shares. A DIB over that page reads whatever it
/// holds and its writes reach every other released level, so the DC reads the
/// level back first, the way a `LockRect` of it does.
#[test]
fn get_dc_on_a_released_default_pool_level_reads_the_level_back() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const GREEN: u32 = 0xFF00_FF00;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    {
        let mut locked = tex.lock_rect(0, 0);
        locked.write_u32(&[GREEN; TEXELS]);
    }
    // The draw is what uploads the level.
    assert_pixel_eq(sample_center(&h, &tex).to_pixel(), GREEN, "upload");
    release_uploaded_staging(&h);

    let surface = tex.surface_level(0);
    let dc = surface.dc();
    let last = (SIZE - 1).cast_signed();
    for (x, y, name) in [(0, 0, "first texel"), (last, last, "last texel")] {
        assert_eq!(
            dc.get_pixel(x, y),
            GREEN_COLORREF,
            "the DC reads the {name}"
        );
    }
    assert_eq!(
        dc.set_pixel(0, 0, RED_COLORREF),
        RED_COLORREF,
        "SetPixel stores full-scale channels exactly in an 8-8-8-8 DIB",
    );
    assert_eq!(dc.release(), 0, "ReleaseDC");

    // The quad spans the unit square over a 640x480 target, so texel (0, 0)
    // covers x 0..10, y 0..7: read a point inside that band.
    let painted = sample_at(&h, &tex, 5, 3);
    assert!(
        painted.r > 200 && painted.g < 50 && painted.b < 50,
        "a draw samples the texel GDI painted, so it reached the GPU, got {painted:?}"
    );
    let untouched = sample_center(&h, &tex);
    assert!(
        untouched.r < 50 && untouched.g > 200 && untouched.b < 50,
        "the texels GDI left alone still sample as the lock wrote them, got {untouched:?}"
    );
}

/// A draw under a held `GetDC` leaves the level's staging where the DIB is.
///
/// The `UnlockRect` marks the level dirty, so the first draw after it uploads
/// the level and would release the staging the DC's DIB aliases: the DIB would
/// then read and write a page nothing owns, and GDI's drawing would reach the
/// texture nowhere. A held device context pins the level for its lifetime, so
/// the upload `ReleaseDC` schedules carries what GDI drew.
#[test]
fn a_draw_under_a_held_device_context_keeps_the_level_staging() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const GREEN: u32 = 0xFF00_FF00;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    {
        let mut locked = tex.lock_rect(0, 0);
        locked.write_u32(&[GREEN; TEXELS]);
    }

    // Nothing has uploaded the level yet, so the DC maps the pages the lock
    // wrote rather than a re-materialised copy of them.
    let surface = tex.surface_level(0);
    let dc = surface.dc();
    assert_eq!(
        dc.get_pixel(0, 0),
        GREEN_COLORREF,
        "the DC reads the texels the lock wrote"
    );
    // Two draws while the DC is held: the first uploads the level and is the
    // one that would release its staging, the second retires the upload job
    // holding the only other reference to those pages.
    for pass in ["first draw under the DC", "second draw under the DC"] {
        assert_pixel_eq(sample_center(&h, &tex).to_pixel(), GREEN, pass);
    }
    assert_eq!(
        dc.set_pixel(0, 0, RED_COLORREF),
        RED_COLORREF,
        "SetPixel stores full-scale channels exactly in an 8-8-8-8 DIB",
    );
    assert_eq!(dc.release(), 0, "ReleaseDC");

    // The quad spans the unit square over a 640x480 target, so texel (0, 0)
    // covers x 0..10, y 0..7: read a point inside that band.
    let painted = sample_at(&h, &tex, 5, 3);
    assert!(
        painted.r > 200 && painted.g < 50 && painted.b < 50,
        "a draw samples the texel GDI painted through the held DC, got {painted:?}"
    );
    let untouched = sample_center(&h, &tex);
    assert!(
        untouched.r < 50 && untouched.g > 200 && untouched.b < 50,
        "the texels GDI left alone still sample as the lock wrote them, got {untouched:?}"
    );
}

/// A texture level is either mapped or holds a DC, never both.
///
/// A level's `LockRect` is recorded on the parent texture, so nothing about the
/// lock is visible on the surface shell itself; `GetDC` has to consult the
/// parent to see it. Each call succeeds once the other side has been given up.
#[test]
fn get_dc_and_lock_rect_on_a_texture_level_exclude_each_other() {
    const SIZE: u32 = 8;
    let sentinel = 0xdead_beef_usize as *mut core::ffi::c_void;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let surface = tex.surface_level(0);

    {
        let _locked = surface.lock_rect(0);
        let (hr, out) = surface.get_dc(sentinel);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "GetDC while the level's LockRect is outstanding must return INVALIDCALL"
        );
        assert_eq!(
            out, sentinel,
            "a rejected GetDC must not write through the out HDC"
        );
    }
    let dc = surface.dc();

    let (hr, bits_null) = surface.lock_rect_probe(0);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "LockRect while the level's DC is open must return INVALIDCALL"
    );
    assert!(
        !bits_null,
        "a rejected LockRect must not write through the out D3DLOCKED_RECT"
    );
    assert_eq!(dc.release(), 0, "ReleaseDC");

    let (hr, bits_null) = surface.lock_rect_probe(0);
    assert_eq!(hr, 0, "the released DC leaves the level lockable again");
    assert!(!bits_null, "an accepted LockRect maps the level");
    assert_eq!(surface.unlock_rect(), 0, "UnlockRect");
}

/// `GetDC` on one level is rejected while another level of the texture is locked.
///
/// D3D9 gates `GetDC` on the whole resource: any outstanding map, on any
/// sub-resource, rejects it. Each level surface is its own shell, so the state
/// the call reads has to be the parent texture's rather than the shell's.
#[test]
fn get_dc_on_a_texture_level_is_rejected_while_another_level_is_locked() {
    const SIZE: u32 = 8;
    let sentinel = 0xdead_beef_usize as *mut core::ffi::c_void;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let level0 = tex.surface_level(0);

    {
        let _locked = tex.lock_rect(1, 0);
        let (hr, out) = level0.get_dc(sentinel);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "GetDC on level 0 while level 1 is locked must return INVALIDCALL"
        );
        assert_eq!(
            out, sentinel,
            "a rejected GetDC must not write through the out HDC"
        );
    }

    let dc = level0.dc();
    assert_eq!(
        dc.release(),
        0,
        "the released lock leaves the texture DC-able again"
    );
}

/// `IDirect3DTexture9::LockRect` is rejected while a level surface holds a DC.
///
/// The DC is taken through a level shell and blocks the whole resource, so both
/// the texture entry point and another level's surface have to see it. Every
/// level is lockable again once the DC is released.
#[test]
fn texture_lock_rect_is_rejected_while_a_level_holds_a_dc() {
    const SIZE: u32 = 8;
    const LEVELS: u32 = 2;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, LEVELS, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let level0 = tex.surface_level(0);
    let level1 = tex.surface_level(1);
    let dc = level0.dc();

    for level in 0..LEVELS {
        let (hr, bits_null) = tex.lock_rect_probe(level, 0);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "LockRect({level}) while the texture holds a DC must return INVALIDCALL"
        );
        assert!(
            !bits_null,
            "a rejected LockRect must not write through the out D3DLOCKED_RECT"
        );
    }
    let (hr, _) = level1.lock_rect_probe(0);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "another level's surface LockRect must see the DC too"
    );

    assert_eq!(dc.release(), 0, "ReleaseDC");

    for level in 0..LEVELS {
        let (hr, bits_null) = tex.lock_rect_probe(level, 0);
        assert_eq!(hr, 0, "the released DC leaves level {level} lockable again");
        assert!(!bits_null, "an accepted LockRect maps the level");
        assert_eq!(tex.unlock_rect(level), 0, "UnlockRect({level})");
    }
}

/// `D3DLOCK_DISCARD` on a released default-pool level rewrites it whole.
///
/// DISCARD declares the level's contents dead, so the lock takes the fresh
/// pages as they are and skips the read back. What the application writes
/// through them is what the level holds afterwards.
#[test]
fn discard_lock_of_a_released_default_pool_level_rewrites_it() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const FIRST: u32 = 0xFFFF_0000;
    const SECOND: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    {
        let mut locked = tex.lock_rect(0, 0);
        locked.write_u32(&[FIRST; TEXELS]);
    }
    assert_pixel_eq(sample_center(&h, &tex).to_pixel(), FIRST, "first fill");
    release_uploaded_staging(&h);

    {
        let mut locked = tex.lock_rect(0, D3DLOCK_DISCARD);
        locked.write_u32(&[SECOND; TEXELS]);
    }
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        SECOND,
        "discard rewrite",
    );
}

/// A whole-level lock without `D3DLOCK_DISCARD` keeps the texels the game does not rewrite.
///
/// The level is DEFAULT-pool DYNAMIC and its previous upload is still in
/// flight, which is where the lock renames its staging. D3D9 hands a plain
/// lock the level's current contents whatever the usage says, and a game
/// that locks a whole page to rewrite a few blocks of it relies on that: the
/// three quadrants the lock does not touch must sample back as they were.
#[test]
fn dynamic_whole_level_lock_without_discard_keeps_the_texels_it_does_not_rewrite() {
    const SIZE: u32 = 64;
    const BASE: u32 = 0xFFFF_0000;
    const REPAINT: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    tex.lock_rect(0, 0)
        .write_u32(&[BASE; (SIZE * SIZE) as usize]);
    let sampled = lock_under_in_flight_upload(&h, &tex, || {
        tex.lock_rect(0, 0)
            .write_u32_rect(32, 32, &[REPAINT; 32 * 32]);
    });
    assert_pixel_eq(sampled[0], REPAINT, "the rewritten quadrant");
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(sampled[i], BASE, name);
    }
}

/// A `D3DLOCK_DISCARD` sub-rect lock leaves the rest of the level alone.
///
/// A discard of part of a level cannot be honoured, so the lock is served as
/// a plain partial one: the flag is dropped, the quadrant is written in place
/// and the other three keep their texels. The previous upload is in flight so
/// the lock is the contended kind, where a fresh staging would otherwise be
/// handed out. The quadrant's own upload carries only its rect, so the GPU
/// copy hides a staging with holes in it until something publishes the whole
/// level again; a later whole-level lock does exactly that, and the three
/// quadrants have to come through it as well.
#[test]
fn partial_discard_lock_leaves_the_rest_of_the_level_alone() {
    const SIZE: u32 = 64;
    const BASE: u32 = 0xFFFF_0000;
    const REPAINT: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    tex.lock_rect(0, 0)
        .write_u32(&[BASE; (SIZE * SIZE) as usize]);
    let (_, rect, _) = QUADRANTS[0];
    let sampled = lock_under_in_flight_upload(&h, &tex, || {
        tex.lock_rect_partial(0, &rect, D3DLOCK_DISCARD)
            .write_u32_rect(32, 32, &[REPAINT; 32 * 32]);
    });
    assert_pixel_eq(sampled[0], REPAINT, "the rewritten quadrant");
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(sampled[i], BASE, name);
    }

    // A whole-level lock that rewrites the same quadrant publishes the
    // staging in full: what it holds outside that quadrant is what the level
    // shows afterwards.
    let republished = lock_under_in_flight_upload(&h, &tex, || {
        tex.lock_rect(0, 0)
            .write_u32_rect(32, 32, &[REPAINT; 32 * 32]);
    });
    assert_pixel_eq(
        republished[0],
        REPAINT,
        "the rewritten quadrant, republished",
    );
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(republished[i], BASE, &format!("{name}, republished"));
    }
}

/// `D3DLOCK_DISCARD` on a managed level is ignored.
///
/// A managed level is served from its staging for its whole life and every
/// later re-upload publishes all of it, so its contents are never dead. The
/// lock preserves them like a plain one: a whole-level DISCARD lock that
/// rewrites one quadrant leaves the other three as they were.
#[test]
fn discard_lock_of_a_managed_level_is_ignored() {
    const SIZE: u32 = 64;
    const BASE: u32 = 0xFFFF_0000;
    const REPAINT: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0)
        .write_u32(&[BASE; (SIZE * SIZE) as usize]);
    let sampled = lock_under_in_flight_upload(&h, &tex, || {
        tex.lock_rect(0, D3DLOCK_DISCARD)
            .write_u32_rect(32, 32, &[REPAINT; 32 * 32]);
    });
    assert_pixel_eq(sampled[0], REPAINT, "the rewritten quadrant");
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(sampled[i], BASE, name);
    }
}

/// One solid DXT1 block: both endpoints `color565`, every index 0.
const fn dxt1_solid_block(color565: u16) -> [u8; 8] {
    let [lo, hi] = color565.to_le_bytes();
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// A full-screen quad whose direction coordinates sweep the `+X` cube face.
///
/// D3D9 maps `+X` with `s = -z` and `t = -y`, so the top-left of the screen
/// looks along `(1, 1, 1)` and reads the face's top-left texel.
const fn positive_x_face_quad() -> [CubeVertex; 6] {
    const fn v(x: f32, y: f32, dir_y: f32, dir_z: f32) -> CubeVertex {
        CubeVertex {
            x,
            y,
            z: 0.5,
            color: 0xFFFF_FFFF,
            u: 1.0,
            v: dir_y,
            w: dir_z,
        }
    }
    [
        v(-1.0, 1.0, 1.0, 1.0),
        v(1.0, 1.0, 1.0, -1.0),
        v(-1.0, -1.0, -1.0, 1.0),
        v(1.0, 1.0, 1.0, -1.0),
        v(1.0, -1.0, -1.0, -1.0),
        v(-1.0, -1.0, -1.0, 1.0),
    ]
}

/// Bind `cube` for point-sampled `+X` face draws and hand back the quad.
///
/// The cube form of [`bind_for_quadrant_draws`], for the tests that draw more
/// than once in a scene.
fn bind_cube_for_face_draws(h: &Harness, cube: &mtld3d_tests::CubeTexture<'_>) -> [CubeVertex; 6] {
    assert_eq!(h.set_cube_texture(0, cube), 0, "SetTexture cube");
    h.select_texture_stage(0);
    point_clamp(h);
    // D3DFVF_TEXCOORDSIZE3(0) is bit 16.
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | 0x0001_0000),
        0,
        "SetFVF cube"
    );
    positive_x_face_quad()
}

/// Bind `cube`, sample its `+X` face across the back buffer, read the quadrant centres.
///
/// Clockwise from the top left, the order [`QUADRANTS`] lists them in. The
/// cube is unbound again, so the caller can release the device it was sampled
/// on without the stage holding the texture.
fn sample_cube_face_quadrants(h: &Harness, cube: &mtld3d_tests::CubeTexture<'_>) -> [u32; 4] {
    let quad = bind_cube_for_face_draws(h, cube);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "cube sample draw"
        );
    });
    let sampled = read_quadrants(h);
    assert_eq!(h.clear_texture(0), 0, "unbind the cube");
    sampled
}

enum CubeUpdateSource {
    StandaloneSurface,
    CubeSurface,
    CubeTexture,
}

fn check_partial_cube_update_preserves_gpu_pixels(format: u32, source: &CubeUpdateSource) {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let dst = h.create_cube_texture_owned(
        4,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(h.color_fill_hr(&dst.surface(0, 0), RED), 0);
    // Sampling and reading the primer completes the GPU write before the copy.
    for pixel in sample_cube_face_quadrants(&h, &dst) {
        assert_pixel_eq(pixel, RED, "completed ColorFill primer");
    }
    let blue = if format == D3DFMT_X8R8G8B8 {
        BLUE & 0x00FF_FFFF
    } else {
        BLUE
    };
    match source {
        CubeUpdateSource::StandaloneSurface => {
            let src = h.create_offscreen_plain_surface(2, 2, format, D3DPOOL_SYSTEMMEM);
            src.lock_rect(0).write_u32(&[blue; 4]);
            assert_eq!(h.update_surface_hr(&src, &dst.surface(0, 0)), 0);
        }
        CubeUpdateSource::CubeSurface => {
            let src = h.create_cube_texture_owned(4, 1, 0, format, D3DPOOL_SYSTEMMEM);
            src.lock_rect(1, 0, 0).write_u32(&[blue; 16]);
            assert_eq!(
                h.update_surface_region_hr(
                    &src.surface(1, 0),
                    &D3DRECT {
                        x1: 0,
                        y1: 0,
                        x2: 2,
                        y2: 2
                    },
                    &dst.surface(0, 0),
                    (0, 0),
                ),
                0,
            );
        }
        CubeUpdateSource::CubeTexture => {
            let src = h.create_cube_texture_owned(4, 1, 0, format, D3DPOOL_SYSTEMMEM);
            let consumed = h.create_cube_texture_owned(4, 1, 0, format, D3DPOOL_DEFAULT);
            // Consume the creation-time whole-face dirty regions before the partial lock.
            assert_eq!(h.update_cube_texture_hr(&src, &consumed), 0);
            src.lock_rect_partial(0, 0, &[0, 0, 2, 2], 0)
                .write_u32_rect(2, 2, &[blue; 4]);
            assert_eq!(h.update_cube_texture_hr(&src, &dst), 0);
        }
    }
    let sampled = sample_cube_face_quadrants(&h, &dst);
    assert_pixel_eq(sampled[0], BLUE, "copied quadrant");
    for (index, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(sampled[index], RED, name);
    }
}

#[test]
fn partial_cube_update_from_standalone_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(
        D3DFMT_A8R8G8B8,
        &CubeUpdateSource::StandaloneSurface,
    );
}

#[test]
fn partial_cube_update_from_converting_standalone_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(
        D3DFMT_X8R8G8B8,
        &CubeUpdateSource::StandaloneSurface,
    );
}

#[test]
fn partial_cube_update_from_cube_surface_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(D3DFMT_A8R8G8B8, &CubeUpdateSource::CubeSurface);
}

#[test]
fn partial_cube_update_from_converting_cube_surface_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(D3DFMT_X8R8G8B8, &CubeUpdateSource::CubeSurface);
}

#[test]
fn partial_cube_update_texture_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(D3DFMT_A8R8G8B8, &CubeUpdateSource::CubeTexture);
}

#[test]
fn partial_cube_update_texture_converting_preserves_gpu_pixels() {
    check_partial_cube_update_preserves_gpu_pixels(D3DFMT_X8R8G8B8, &CubeUpdateSource::CubeTexture);
}

/// A partial lock's `UnlockRect` publishes the rect it named, on every upload path.
///
/// Each quadrant is locked while the previous draw's upload is in flight and
/// sampled after the next one, so every lock is a contended partial lock and
/// every upload carries one quadrant: through the uncompressed blit with its
/// origin offset, through the block-compressed blit on a block-aligned rect,
/// and through the cube form of the same bookkeeping. A wrong origin or pitch
/// in any of them lands a quadrant somewhere else.
#[test]
fn partial_locks_publish_their_own_rect_on_every_upload_path() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    // DXT1: a 32x32 quadrant is 8 rows of 8 blocks, 64 bytes per block row.
    const DXT1_COLORS: [u16; 4] = [0xF800, 0x07E0, 0x001F, 0xFFE0];
    let h = Harness::new();

    let tex = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    tex.lock_rect(0, 0).write_u32(&[BLACK; TEXELS]);
    let quad = bind_for_quadrant_draws(&h, &tex);
    h.render_once(BLACK, |d| {
        for (color, rect, name) in QUADRANTS {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
                0,
                "draw before {name}"
            );
            tex.lock_rect_partial(0, &rect, 0)
                .write_u32_rect(32, 32, &[color; 32 * 32]);
        }
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "final uncompressed draw"
        );
    });
    let sampled = read_quadrants(&h);
    for (i, (color, _, name)) in QUADRANTS.into_iter().enumerate() {
        assert_pixel_eq(sampled[i], color, &format!("A8R8G8B8 {name}"));
    }

    let dxt = h.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_DXT1,
        D3DPOOL_DEFAULT,
    );
    dxt.lock_rect(0, 0).write::<u8>(&[0; TEXELS / 2]);
    let quad = bind_for_quadrant_draws(&h, &dxt);
    h.render_once(BLACK, |d| {
        for (i, (_, rect, name)) in QUADRANTS.into_iter().enumerate() {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
                0,
                "draw before {name}"
            );
            let block = dxt1_solid_block(DXT1_COLORS[i]);
            let bytes: Vec<u8> = block.iter().copied().cycle().take(64 * 8).collect();
            dxt.lock_rect_partial(0, &rect, 0)
                .write_u8_rect(64, 8, &bytes);
        }
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "final DXT1 draw"
        );
    });
    let sampled = read_quadrants(&h);
    for (i, (color, _, name)) in QUADRANTS.into_iter().enumerate() {
        assert_pixel_approx(sampled[i], color, 8, &format!("DXT1 {name}"));
    }

    let cube = h.create_cube_texture_owned(SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    cube.lock_rect(0, 0, 0).write_u32(&[BLACK; TEXELS]);
    let quad = bind_cube_for_face_draws(&h, &cube);
    h.render_once(BLACK, |d| {
        for (color, rect, name) in QUADRANTS {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
                0,
                "draw before {name}"
            );
            cube.lock_rect_partial(0, 0, &rect, 0)
                .write_u32_rect(32, 32, &[color; 32 * 32]);
        }
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "final cube draw"
        );
    });
    let sampled = read_quadrants(&h);
    for (i, (color, _, name)) in QUADRANTS.into_iter().enumerate() {
        assert_pixel_eq(sampled[i], color, &format!("cube +X {name}"));
    }
    assert_eq!(h.clear_texture(0), 0, "unbind the cube");
}

/// A cube face uploads whole onto the device it migrates to.
///
/// A `D3DPOOL_MANAGED` cube outlives the device that created it: the
/// application keeps the texture, releases the device, creates another and
/// binds the texture there. That device's Metal texture is empty, so every
/// level the application has written is uploaded again whole. A partial lock
/// the old device never flushed leaves an upload rect behind, and honouring it
/// on the new device would upload that rect alone and leave the rest of the
/// face at the empty texture's zeros.
#[test]
fn a_cube_face_uploads_whole_onto_the_device_it_migrates_to() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const BASE: u32 = 0xFF00_FF00;
    const PATCH: u32 = 0xFFFF_0000;
    let first = Harness::new();
    let cube = first.create_cube_texture_owned(SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    cube.lock_rect(0, 0, 0).write_u32(&[BASE; TEXELS]);
    let sampled = sample_cube_face_quadrants(&first, &cube);
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate() {
        assert_pixel_eq(sampled[i], BASE, name);
    }

    // No draw follows this lock, so the top-left quadrant is still the face's
    // pending upload rect when the device goes away under it.
    let (_, rect, _) = QUADRANTS[0];
    cube.lock_rect_partial(0, 0, &rect, 0)
        .write_u32_rect(32, 32, &[PATCH; 32 * 32]);
    assert_eq!(
        first.release_device(),
        0,
        "a managed cube holds no reference on the device"
    );

    let second = Harness::new();
    let migrated = sample_cube_face_quadrants(&second, &cube);
    assert_pixel_eq(migrated[0], PATCH, "the quadrant the partial lock wrote");
    for (i, (_, _, name)) in QUADRANTS.into_iter().enumerate().skip(1) {
        assert_pixel_eq(migrated[i], BASE, name);
    }
}

/// A texture that migrates between two live devices leaves the first one's registry.
///
/// Each device keeps a registry of raw `TextureInner` pointers, and a texture
/// drops out of it when it is freed, from the device it belongs to at that
/// moment. Binding it on a second device that is alive beside the first moves
/// it between them, so an entry the first device keeps names a texture it no
/// longer owns. Releasing that device walks the registry and writes through
/// every entry, and `EvictManagedResources` walks the same list.
///
/// The two textures cover the two shapes that walk reaches. One is freed
/// first, so a stale entry would be dereferenced after its allocation is
/// gone. The other is still live, where the walk detaches a texture that
/// belongs to the second device, and `GetDevice` is what says so: a detached
/// texture answers `D3DERR_INVALIDCALL`, so the migrated texture naming the
/// device it migrated to is the assertion that the first device let go of it.
#[test]
fn a_texture_migrating_between_live_devices_leaves_the_first_devices_registry() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;

    let first = Harness::new();
    let second = Harness::new();

    let kept = first.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    kept.lock_rect(0, 0).write_u32(&[GREEN; TEXELS]);
    let freed = first.create_texture(SIZE, SIZE, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    freed.lock_rect(0, 0).write_u32(&[RED; TEXELS]);
    assert_pixel_eq(
        sample_center(&first, &kept).to_pixel(),
        GREEN,
        "the kept texture on the device that created it",
    );
    assert_pixel_eq(
        sample_center(&first, &freed).to_pixel(),
        RED,
        "the freed texture on the device that created it",
    );

    // Both migrate on their first bind under the second device, which draws
    // while the first device is still live.
    assert_pixel_eq(
        sample_center(&second, &kept).to_pixel(),
        GREEN,
        "the kept texture on the device it migrated to",
    );
    assert_pixel_eq(
        sample_center(&second, &freed).to_pixel(),
        RED,
        "the freed texture on the device it migrated to",
    );

    // A bound stage holds a reference on the texture, so both devices give
    // theirs up before either texture is freed.
    assert_eq!(first.clear_texture(0), 0, "unbind on the first device");
    assert_eq!(second.clear_texture(0), 0, "unbind on the second device");
    drop(freed);
    assert_eq!(
        first.release_device(),
        0,
        "the first device is fully released"
    );

    let before = second.device_refcount();
    let (hr, dev) = kept.get_device();
    assert_eq!(hr, 0, "GetDevice on the texture that migrated");
    assert_eq!(dev, second.device(), "the device the texture migrated to");
    // SAFETY: `dev` is the reference `GetDevice` just handed out.
    let back = unsafe { second.release_device_ref(dev) };
    assert_eq!(
        back, before,
        "the reference handed out is the one given back"
    );
    assert_pixel_eq(
        sample_center(&second, &kept).to_pixel(),
        GREEN,
        "the kept texture once the first device is gone",
    );
    assert_eq!(
        second.clear_texture(0),
        0,
        "unbind before the texture is freed"
    );
}

/// A `D3DPOOL_DEFAULT` texture that migrates between two live devices takes its pin with it.
///
/// Every pool but `D3DPOOL_MANAGED` holds one reference on the device that
/// created it for the texture's public lifetime, and the runtime derives that
/// device from the texture's current one. A bind under a second device that is
/// alive beside the first repoints the texture, so the reference has to move
/// at the same moment: left where it was, the last `Release` hands a reference
/// back to the device the texture migrated to, which never took one, and the
/// creating device stays pinned for the life of the process.
///
/// Both devices' counts are read around the migration and around the texture's
/// `Release`. The `Reset` a referenced `D3DPOOL_DEFAULT` resource blocks reads
/// the same state from the other side: after the migration it is the adopting
/// device whose `Reset` the texture has to reject.
///
/// `D3DUSAGE_DYNAMIC` keeps the level's staging, so the sample on the second
/// device reads what the migration re-uploads instead of what a released
/// staging leaves behind. The reference follows from the pool, so the usage
/// does not change what is under test.
#[test]
fn a_default_texture_migrating_between_live_devices_moves_its_device_pin() {
    const SIZE: u32 = 64;
    const TEXELS: usize = (SIZE * SIZE) as usize;
    const GREEN: u32 = 0xFF00_FF00;

    let first = Harness::new();
    let second = Harness::new();
    let first_base = first.device_refcount();
    let second_base = second.device_refcount();

    let tex = first.create_texture(
        SIZE,
        SIZE,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    tex.lock_rect(0, 0).write_u32(&[GREEN; TEXELS]);
    assert_eq!(
        first.device_refcount(),
        first_base + 1,
        "the texture pins the device that created it"
    );
    assert_eq!(
        second.device_refcount(),
        second_base,
        "the other device is untouched by the create"
    );
    assert_pixel_eq(
        sample_center(&first, &tex).to_pixel(),
        GREEN,
        "the texture on the device that created it",
    );

    // The first bind under the second device migrates it, with the first
    // device still live.
    assert_pixel_eq(
        sample_center(&second, &tex).to_pixel(),
        GREEN,
        "the texture on the device it migrated to",
    );
    // A bound stage holds a reference on the texture, so both devices give
    // theirs up before the counts are read.
    assert_eq!(first.clear_texture(0), 0, "unbind on the first device");
    assert_eq!(second.clear_texture(0), 0, "unbind on the second device");
    assert_eq!(
        first.device_refcount(),
        first_base,
        "the pin left the device the texture migrated off"
    );
    assert_eq!(
        second.device_refcount(),
        second_base + 1,
        "the pin arrived on the device the texture migrated to"
    );

    // The `Reset` blocker the engine counts for a DEFAULT resource moved with
    // the reference it rides on.
    assert_eq!(
        first.reset(320, 240),
        0,
        "the device the texture left has nothing blocking Reset"
    );
    assert_eq!(
        second.reset(320, 240),
        D3DERR_INVALIDCALL,
        "the device that adopted the texture is the one it blocks"
    );

    drop(tex);
    assert_eq!(
        first.device_refcount(),
        first_base,
        "the freed texture takes nothing from the device it migrated off"
    );
    assert_eq!(
        second.device_refcount(),
        second_base,
        "the freed texture gives its reference back to the device it named"
    );
    assert_eq!(
        second.reset(320, 240),
        0,
        "Reset succeeds once the texture is freed"
    );
}

/// An `UpdateSurface` from system memory reaches the very next draw.
///
/// The staging write only reaches the GPU through the bind-time
/// `flush_dirty_mips`, which the API thread runs while it rebuilds a dirty
/// snapshot. Two draws inside one scene with nothing but the update between
/// them leave the snapshot clean, so the update has to dirty it itself or the
/// second draw samples the texels the first one saw.
#[test]
fn update_surface_from_system_memory_reaches_the_next_draw() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dst = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let level = dst.surface_level(0);

    let first = h.create_offscreen_plain_surface(2, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    first.lock_rect(0).write_u32(&[GREEN; 4]);
    assert_eq!(
        h.update_surface_hr(&first, &level),
        0,
        "first UpdateSurface"
    );
    // Binds the texture, arms the sampler and submits the first upload, so the
    // frame below starts from the state this leaves behind.
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "first fill");

    let second = h.create_offscreen_plain_surface(2, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    second.lock_rect(0).write_u32(&[RED; 4]);

    // The first draw consumes the frame-start snapshot dirtiness; the update is
    // then the only call before the second draw, which covers the backbuffer
    // again.
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "draw before the update"
        );
        assert_eq!(
            d.update_surface_hr(&second, &level),
            0,
            "second UpdateSurface"
        );
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "draw after the update"
        );
    });
    assert_pixel_eq(
        h.read_pixel(320, 240),
        RED,
        "the draw after the update must sample the updated texels",
    );
}

/// `UpdateSurface` copies a compressed standalone system-memory source.
///
/// A DXT1 surface reports one row of eight-byte blocks rather than a linear
/// bytes-per-pixel pitch. The copy must carry that block-row stride into the
/// destination texture without weakening the existing format and pool checks.
#[test]
fn update_surface_copies_a_compressed_standalone_source() {
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    {
        let mut locked = src.lock_rect(0);
        assert_eq!(locked.pitch(), 8, "one DXT1 block row");
        locked.write::<u8>(&dxt1_solid_block(0xF800));
    }

    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src, &dst.surface_level(0)),
        0,
        "UpdateSurface from a DXT1 standalone source"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFFFF_0000,
        "copied DXT1 block",
    );
}

/// `UpdateSurface` rejects a destination outside `D3DPOOL_DEFAULT`.
///
/// The source is a standalone system-memory offscreen surface, which reaches
/// the destination level's staging directly instead of going through a source
/// texture, so the destination pool contract has to hold on that path too. The
/// matching default-pool destination in the same test keeps the check narrow.
#[test]
fn update_surface_rejects_a_standalone_source_into_a_managed_destination() {
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[GREEN; 16]);

    let managed = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(
        h.update_surface_hr(&src, &managed.surface_level(0)),
        D3DERR_INVALIDCALL,
        "UpdateSurface into a managed destination"
    );

    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src, &dst.surface_level(0)),
        0,
        "UpdateSurface into a matching default-pool destination"
    );
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "accepted fill");
}

/// `UpdateSurface` rejects a standalone-source pair the CPU codec cannot convert.
///
/// `V8U8` is signed, so it is outside the codec's unsigned normalised set and
/// there is nothing to re-encode the source with. Both formats are 16 bits per
/// pixel and the source is the destination's size, so without the check the
/// call would land signed two-channel bits the destination reads as 5-6-5.
#[test]
fn update_surface_rejects_a_standalone_source_the_codec_cannot_convert() {
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_V8U8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write::<u16>(&[0x83E0; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src, &dst.surface_level(0)),
        D3DERR_INVALIDCALL,
        "UpdateSurface into a destination of another format"
    );
}

/// `UpdateSurface` rejects a standalone source outside `D3DPOOL_SYSTEMMEM`.
///
/// A `D3DPOOL_SCRATCH` offscreen-plain surface carries the same CPU backing as
/// a system-memory one, so the source pool is the only thing separating the
/// two on the standalone-source path. Scratch is a CPU-only staging pool rather
/// than a device resource, and D3D9 answers `D3DERR_INVALIDCALL` for it. The
/// system-memory source accepted into the same destination keeps the check
/// narrow.
#[test]
fn update_surface_rejects_a_scratch_standalone_source() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);

    let scratch = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH);
    scratch.lock_rect(0).write_u32(&[RED; 16]);
    assert_eq!(
        h.update_surface_hr(&scratch, &dst.surface_level(0)),
        D3DERR_INVALIDCALL,
        "UpdateSurface from a scratch source"
    );

    let sysmem = h.create_offscreen_plain_surface(4, 4, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    sysmem.lock_rect(0).write_u32(&[GREEN; 16]);
    assert_eq!(
        h.update_surface_hr(&sysmem, &dst.surface_level(0)),
        0,
        "UpdateSurface from a system-memory source"
    );
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "accepted fill");
}

#[test]
fn update_texture_keeps_cube_faces_independent() {
    let h = Harness::new();
    let src = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0, 0).write_u32(&[0xFFFF_0000; 16]);
    src.lock_rect(1, 0, 0).write_u32(&[0xFF00_FF00; 16]);
    let dst = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(h.update_cube_texture_hr(&src, &dst), 0);
    assert_pixel_eq(
        sample_cube_x(&h, &dst, 1.0),
        0xFFFF_0000,
        "UpdateTexture positive-X face",
    );
    assert_pixel_eq(
        sample_cube_x(&h, &dst, -1.0),
        0xFF00_FF00,
        "UpdateTexture negative-X face",
    );
}

/// `UpdateTexture` accepts a source whose levels are still mapped.
///
/// The per-endpoint rejection `UpdateSurface` applies does not carry over to
/// the container entry point: D3D9 validates no lock state on `UpdateTexture`,
/// and an application that leaves one level of a mip chain mapped while
/// updating the chain gets the copy and `D3D_OK`. Rejecting the call would be
/// a divergence, so the accepting answer is pinned here.
#[test]
fn update_texture_accepts_a_source_with_a_level_locked() {
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(4, 4, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write::<u32>(&[GREEN; 16]);

    let held = src.lock_rect(1, 0);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        0,
        "UpdateTexture with the second source level mapped"
    );
    drop(held);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        GREEN,
        "top level after the update",
    );
}

/// Sample the volume bound on stage 0 at texcoord `(0.5, 0.5, w)` with point filtering.
fn sample_volume_depth(h: &Harness, w: f32) -> u32 {
    let quad = volume_sample_quad(-1.0, 1.0, [0.5, 0.5, w]);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            0,
            "volume sample draw"
        );
    });
    h.read_pixel(320, 240)
}

/// A multi-slice update after a completed upload preserves the earlier draw.
#[test]
fn volume_update_keeps_per_draw_content() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;
    let harnesses = [
        Harness::new(),
        Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true"),
    ];
    for h in &harnesses {
        let (hr, source) =
            h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
        assert_eq!(hr, 0);
        let source = source.expect("source");
        let (hr, destination) =
            h.try_create_volume_texture([4, 4, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(hr, 0);
        let destination = destination.expect("destination");
        source.write_u32(0, &[RED; 64]);
        assert_eq!(h.update_volume_texture_hr(&source, &destination), 0);
        assert_eq!(h.set_volume_texture(0, &destination), 0);
        h.select_texture_stage(0);
        point_clamp(h);
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
            0
        );
        assert_pixel_eq(sample_volume_depth(h, 0.875), RED, "primed volume");
        source.write_u32(0, &[BLUE; 64]);
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.draw_primitive_up(
                    D3DPT_TRIANGLELIST,
                    2,
                    &volume_sample_quad(-1.0, 0.0, [0.5, 0.5, 0.875])
                ),
                0
            );
            assert_eq!(h.update_volume_texture_hr(&source, &destination), 0);
            assert_eq!(
                d.draw_primitive_up(
                    D3DPT_TRIANGLELIST,
                    2,
                    &volume_sample_quad(0.0, 1.0, [0.5, 0.5, 0.875])
                ),
                0
            );
        });
        let pixels = [h.read_pixel(160, 240), h.read_pixel(480, 240)];
        assert_eq!(pixels, [RED, BLUE], "volume draws bracketing UpdateTexture");
    }
}

/// A partial volume update preserves earlier draws and every unwritten subresource.
#[test]
fn volume_partial_update_keeps_versions_and_untouched_mips() {
    check_volume_partial_update();
}

fn check_volume_partial_update() {
    const BLUE: u32 = 0xFF00_00FF;
    const COLORS: [[u32; 4]; 3] = [
        [0xFFFF_0000, 0xFF00_FF00, 0xFFFF_FF00, 0xFFFF_00FF],
        [0xFFFF_FFFF, 0xFF80_8080, 0, 0],
        [0xFFFF_8000, 0, 0, 0],
    ];
    let harnesses = [
        Harness::new(),
        Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true"),
    ];
    for h in &harnesses {
        let (hr, texture) =
            h.try_create_volume_texture([4, 4, 4], 3, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        assert_eq!(hr, 0);
        let texture = texture.expect("managed volume");
        for (level, colors) in COLORS.iter().enumerate() {
            let width = 4 >> level;
            let texels: Vec<_> = colors[..width]
                .iter()
                .flat_map(|color| core::iter::repeat_n(*color, width * width))
                .collect();
            texture.write_u32(u32::try_from(level).expect("three levels"), &texels);
        }
        assert_eq!(h.set_volume_texture(0, &texture), 0);
        h.select_texture_stage(0);
        point_clamp(h);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
            0
        );
        assert_pixel_eq(sample_volume_depth(h, 0.875), COLORS[0][3], "primed volume");
        h.render_once(BLACK, |d| {
            for (index, (left, right)) in [(-1.0, -0.333), (-0.333, 0.333), (0.333, 1.0)]
                .into_iter()
                .enumerate()
            {
                if index == 1 {
                    texture.write_box_u32(
                        0,
                        &D3DBOX {
                            left: 2,
                            top: 2,
                            right: 4,
                            bottom: 4,
                            front: 2,
                            back: 4,
                        },
                        &[BLUE; 8],
                    );
                }
                assert_eq!(
                    d.draw_primitive_up(
                        D3DPT_TRIANGLELIST,
                        2,
                        &volume_sample_quad(left, right, [0.875, 0.875, 0.875]),
                    ),
                    0
                );
            }
        });
        assert_eq!(
            [
                h.read_pixel(100, 240),
                h.read_pixel(320, 240),
                h.read_pixel(540, 240)
            ],
            [COLORS[0][3], BLUE, BLUE],
            "partial volume versions"
        );
        for (level, colors) in COLORS.iter().enumerate() {
            assert_eq!(
                h.set_sampler_state(
                    0,
                    D3DSAMP_MAXMIPLEVEL,
                    u32::try_from(level).expect("three levels")
                ),
                0
            );
            let depth = 4 >> level;
            for (slice, color) in colors[..depth].iter().enumerate() {
                let w = (f32::from(u8::try_from(slice).expect("four slices")) + 0.5)
                    / f32::from(u8::try_from(depth).expect("four slices"));
                for u in [0.125, 0.875] {
                    for v in [0.125, 0.875] {
                        h.render_once(BLACK, |d| {
                            assert_eq!(
                                d.draw_primitive_up(
                                    D3DPT_TRIANGLELIST,
                                    2,
                                    &volume_sample_quad(-1.0, 1.0, [u, v, w])
                                ),
                                0
                            );
                        });
                        let expected = if level == 0 && slice >= 2 && u > 0.5 && v > 0.5 {
                            BLUE
                        } else {
                            *color
                        };
                        assert_pixel_eq(
                            h.read_pixel(320, 240),
                            expected,
                            &format!("mip {level} slice {slice} uv {u},{v}"),
                        );
                    }
                }
            }
        }
    }
}

/// A full upload of a one-slice upper mip preserves every other mip and depth slice.
#[test]
fn volume_upper_mip_rename_preserves_every_slice() {
    check_volume_upper_mip_rename(false);
}

/// A partial upper-mip lock preserves its complement and every other mip and slice.
#[test]
fn volume_upper_mip_partial_rename_preserves_every_slice() {
    check_volume_upper_mip_rename(true);
}

fn check_volume_upper_mip_rename(partial: bool) {
    const BLUE: u32 = 0xFF00_00FF;
    const COLORS: [[u32; 4]; 4] = [
        [0xFFFF_0000, 0xFF00_FF00, 0xFFFF_FF00, 0xFFFF_00FF],
        [0xFF00_FFFF, 0xFFFF_FFFF, 0, 0],
        [0xFFFF_0000, 0, 0, 0],
        [0xFF00_FF00, 0, 0, 0],
    ];
    let h = Harness::new();
    let (hr, texture) =
        h.try_create_volume_texture([8, 8, 4], 4, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, 0);
    let texture = texture.expect("volume texture");
    for (level, colors) in COLORS.iter().enumerate() {
        let width = 8 >> level;
        let depth = (4 >> level).max(1);
        let texels: Vec<u32> = colors[..depth]
            .iter()
            .flat_map(|color| core::iter::repeat_n(*color, width * width))
            .collect();
        texture.write_u32(u32::try_from(level).expect("four levels"), &texels);
    }
    assert_eq!(h.set_volume_texture(0, &texture), 0);
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    assert_pixel_eq(
        sample_volume_depth(&h, 0.875),
        COLORS[0][3],
        "primed deep slice",
    );
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                2,
                &volume_sample_quad(-1.0, 0.0, [0.5, 0.5, 0.875])
            ),
            0
        );
        if partial {
            texture.write_box_u32(
                2,
                &D3DBOX {
                    left: 1,
                    top: 1,
                    right: 2,
                    bottom: 2,
                    front: 0,
                    back: 1,
                },
                &[BLUE],
            );
        } else {
            texture.write_u32(2, &[BLUE; 4]);
        }
        assert_eq!(
            d.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                2,
                &volume_sample_quad(0.0, 1.0, [0.5, 0.5, 0.875])
            ),
            0
        );
    });
    assert_eq!(
        [h.read_pixel(160, 240), h.read_pixel(480, 240)],
        [COLORS[0][3]; 2],
        "upper mip update preserves the deep base slice across the rename"
    );
    for (level, colors) in COLORS.iter().enumerate() {
        assert_eq!(
            h.set_sampler_state(
                0,
                D3DSAMP_MAXMIPLEVEL,
                u32::try_from(level).expect("four levels")
            ),
            0
        );
        let depth = (4 >> level).max(1);
        for (slice, color) in colors[..depth].iter().enumerate() {
            let w = (f32::from(u8::try_from(slice).expect("four slices")) + 0.5)
                / f32::from(u8::try_from(depth).expect("four slices"));
            for uv in [0.25, 0.75] {
                h.render_once(BLACK, |d| {
                    assert_eq!(
                        d.draw_primitive_up(
                            D3DPT_TRIANGLELIST,
                            2,
                            &volume_sample_quad(-1.0, 1.0, [uv, uv, w])
                        ),
                        0
                    );
                });
                let expected = if level == 2 && (!partial || uv > 0.5) {
                    BLUE
                } else {
                    *color
                };
                assert_pixel_eq(
                    h.read_pixel(320, 240),
                    expected,
                    &format!("mip {level} slice {slice} uv {uv}"),
                );
            }
        }
    }
}

/// Queued volume uploads retain the bytes each draw sampled.
#[test]
fn volume_staging_regular_updates() {
    let h = Harness::new();
    check_volume_staging_updates(&h, 4, D3DFMT_A8R8G8B8);
}

/// Converted volume uploads retain each queued version.
#[test]
fn volume_staging_regular_converted_updates() {
    let h = Harness::new();
    check_volume_staging_updates(&h, 4, D3DFMT_X8R8G8B8);
}

/// A single-slice volume retains each queued staging version.
#[test]
fn volume_staging_regular_single_slice_updates() {
    let h = Harness::new();
    check_volume_staging_updates(&h, 1, D3DFMT_A8R8G8B8);
}

/// A cold partial volume write preserves the first queued upload.
#[test]
fn volume_staging_regular_cold_partial_update() {
    let h = Harness::new();
    check_volume_staging_partial_updates(&h, false, false);
}

/// Repeated partial volume writes preserve draw versions and untouched mips.
#[test]
fn volume_staging_regular_partial_updates() {
    let h = Harness::new();
    check_volume_staging_partial_updates(&h, true, true);
}

/// Repeated cold volume writes retain every queued snapshot.
#[test]
fn volume_staging_regular_cold_partial_updates() {
    let h = Harness::new();
    check_volume_staging_partial_updates(&h, false, true);
}

/// Queued volume uploads retain the bytes each draw sampled.
#[test]
fn volume_staging_forced_updates() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_updates(&h, 4, D3DFMT_A8R8G8B8);
}

/// Converted volume uploads retain each queued version.
#[test]
fn volume_staging_forced_converted_updates() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_updates(&h, 4, D3DFMT_X8R8G8B8);
}

/// A single-slice volume retains each queued staging version.
#[test]
fn volume_staging_forced_single_slice_updates() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_updates(&h, 1, D3DFMT_A8R8G8B8);
}

/// A cold partial volume write preserves the first queued upload.
#[test]
fn volume_staging_forced_cold_partial_update() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_partial_updates(&h, false, false);
}

/// Repeated partial volume writes preserve draw versions and untouched mips.
#[test]
fn volume_staging_forced_partial_updates() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_partial_updates(&h, true, true);
}

/// Repeated cold volume writes retain every queued snapshot.
#[test]
fn volume_staging_forced_cold_partial_updates() {
    let h = Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true");
    check_volume_staging_partial_updates(&h, false, true);
}

fn check_volume_staging_updates(h: &Harness, depth: u32, destination_format: u32) {
    const COLORS: [u32; 3] = [0xFFFF_0000, 0xFF00_00FF, 0xFF00_FF00];
    let (hr, source) =
        h.try_create_volume_texture([4, 4, depth], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0);
    let source = source.expect("source volume");
    let (hr, destination) =
        h.try_create_volume_texture([4, 4, depth], 1, 0, destination_format, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0);
    let destination = destination.expect("destination volume");
    let texel_count = 16 * usize::try_from(depth).expect("four slices");
    source.write_u32(0, &vec![COLORS[0]; texel_count]);
    assert_eq!(h.update_volume_texture_hr(&source, &destination), 0);
    assert_eq!(h.set_volume_texture(0, &destination), 0);
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    assert_pixel_eq(sample_volume_depth(h, 0.875), COLORS[0], "primed volume");
    h.render_once(BLACK, |d| {
        for (index, (left, right)) in [(-1.0, -0.333), (-0.333, 0.333), (0.333, 1.0)]
            .into_iter()
            .enumerate()
        {
            if index != 0 {
                source.write_u32(0, &vec![COLORS[index]; texel_count]);
                assert_eq!(h.update_volume_texture_hr(&source, &destination), 0);
            }
            assert_eq!(
                d.draw_primitive_up(
                    D3DPT_TRIANGLELIST,
                    2,
                    &volume_sample_quad(left, right, [0.5, 0.5, 0.875]),
                ),
                0
            );
        }
    });
    assert_eq!(
        [
            h.read_pixel(100, 240),
            h.read_pixel(320, 240),
            h.read_pixel(540, 240)
        ],
        COLORS,
        "volume versions across two UpdateTexture calls"
    );
    for w in [0.125, 0.375, 0.625, 0.875] {
        assert_pixel_eq(sample_volume_depth(h, w), COLORS[2], "final volume slice");
    }
}

fn check_volume_staging_partial_updates(h: &Harness, prime: bool, repeated: bool) {
    const BLUE: u32 = 0xFF00_00FF;
    const CYAN: u32 = 0xFF00_FFFF;
    const COLORS: [[u32; 4]; 3] = [
        [0xFFFF_0000, 0xFF00_FF00, 0xFFFF_FF00, 0xFFFF_00FF],
        [0xFFFF_FFFF, 0xFF80_8080, 0, 0],
        [0xFFFF_8000, 0, 0, 0],
    ];
    let final_color = if repeated { CYAN } else { BLUE };
    let (hr, texture) =
        h.try_create_volume_texture([4, 4, 4], 3, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, 0);
    let texture = texture.expect("managed volume");
    for (level, colors) in COLORS.iter().enumerate() {
        let width = 4 >> level;
        let texels: Vec<_> = colors[..width]
            .iter()
            .flat_map(|color| core::iter::repeat_n(*color, width * width))
            .collect();
        texture.write_u32(u32::try_from(level).expect("three levels"), &texels);
    }
    assert_eq!(h.set_volume_texture(0, &texture), 0);
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    if prime {
        assert_pixel_eq(sample_volume_depth(h, 0.875), COLORS[0][3], "primed volume");
    }
    h.render_once(BLACK, |d| {
        for (index, (left, right)) in [(-1.0, -0.333), (-0.333, 0.333), (0.333, 1.0)]
            .into_iter()
            .enumerate()
        {
            if index == 1 || (index == 2 && repeated) {
                texture.write_box_u32(
                    0,
                    &D3DBOX {
                        left: 2,
                        top: 2,
                        right: 4,
                        bottom: 4,
                        front: 2,
                        back: 4,
                    },
                    &[if index == 1 { BLUE } else { CYAN }; 8],
                );
            }
            assert_eq!(
                d.draw_primitive_up(
                    D3DPT_TRIANGLELIST,
                    2,
                    &volume_sample_quad(left, right, [0.875, 0.875, 0.875]),
                ),
                0
            );
        }
    });
    assert_eq!(
        [
            h.read_pixel(100, 240),
            h.read_pixel(320, 240),
            h.read_pixel(540, 240)
        ],
        [COLORS[0][3], BLUE, final_color],
        "partial volume versions (prime={prime})"
    );
    for (level, colors) in COLORS.iter().enumerate() {
        assert_eq!(
            h.set_sampler_state(
                0,
                D3DSAMP_MAXMIPLEVEL,
                u32::try_from(level).expect("three levels")
            ),
            0
        );
        let depth = 4 >> level;
        for (slice, color) in colors[..depth].iter().enumerate() {
            let w = (f32::from(u8::try_from(slice).expect("four slices")) + 0.5)
                / f32::from(u8::try_from(depth).expect("four slices"));
            for u in [0.125, 0.875] {
                for v in [0.125, 0.875] {
                    h.render_once(BLACK, |d| {
                        assert_eq!(
                            d.draw_primitive_up(
                                D3DPT_TRIANGLELIST,
                                2,
                                &volume_sample_quad(-1.0, 1.0, [u, v, w])
                            ),
                            0
                        );
                    });
                    let expected = if level == 0 && slice >= 2 && u > 0.5 && v > 0.5 {
                        final_color
                    } else {
                        *color
                    };
                    assert_pixel_eq(
                        h.read_pixel(320, 240),
                        expected,
                        &format!("mip {level} slice {slice} uv {u},{v} (prime={prime})"),
                    );
                }
            }
        }
    }
}

fn volume_sample_quad(left: f32, right: f32, coord: [f32; 3]) -> [VolumeVertex; 6] {
    let vertex = |x, y| VolumeVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u: coord[0],
        v: coord[1],
        w: coord[2],
    };
    [
        vertex(left, 1.0),
        vertex(right, 1.0),
        vertex(left, -1.0),
        vertex(right, 1.0),
        vertex(right, -1.0),
        vertex(left, -1.0),
    ]
}

/// `UpdateTexture` carries every slice of a SYSTEMMEM volume into its DEFAULT twin.
///
/// The pattern an engine uses to upload a colour-grading LUT: fill a
/// system-memory volume through `LockBox`, then `UpdateTexture` it into the
/// default-pool volume the shader samples. Each slice carries its own colour
/// so a copy that forgets the dirty mark (nothing arrives) or stops after the
/// first slice (every deeper lookup reads slice 0, or whatever the GPU
/// allocation held) is told apart from a correct one.
#[test]
fn update_texture_copies_every_volume_slice() {
    const SLICE_COLORS: [u32; 4] = [0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF];
    let h = Harness::new();
    let (hr, src) =
        h.try_create_volume_texture([2, 2, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0, "SYSTEMMEM volume");
    let src = src.expect("source volume");
    let texels: Vec<u32> = SLICE_COLORS.iter().flat_map(|&color| [color; 4]).collect();
    src.write_u32(0, &texels);
    let (hr, dst) = h.try_create_volume_texture([2, 2, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0, "DEFAULT volume");
    let dst = dst.expect("destination volume");
    assert_eq!(h.update_volume_texture_hr(&src, &dst), 0, "UpdateTexture");

    assert_eq!(h.set_volume_texture(0, &dst), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    for (z, expected) in (0u8..).zip(SLICE_COLORS) {
        // Slice centres of a four-deep volume.
        let w = (f32::from(z) + 0.5) / 4.0;
        assert_pixel_eq(
            sample_volume_depth(&h, w),
            expected,
            &format!("volume slice {z} after UpdateTexture"),
        );
    }
}

/// `UpdateTexture` aligns depth-dominant volume mip chains at their lowest levels.
///
/// A 1x1x8 source has one more mip than a 1x1x4 destination. The source's red
/// top level must be skipped, then every green lower level must line up with
/// the destination level of the same extent.
#[test]
fn update_texture_matches_depth_dominant_volume_mips() {
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    let h = Harness::new();
    let (hr, src) =
        h.try_create_volume_texture([1, 1, 8], 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0, "SYSTEMMEM source volume");
    let src = src.expect("source volume");
    assert_eq!(src.level_count(), 4, "1x1x8 full mip chain");
    src.write_u32(0, &[RED; 8]);
    src.write_u32(1, &[GREEN; 4]);
    src.write_u32(2, &[GREEN; 2]);
    src.write_u32(3, &[GREEN; 1]);

    let (hr, dst) = h.try_create_volume_texture([1, 1, 4], 0, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0, "DEFAULT destination volume");
    let dst = dst.expect("destination volume");
    assert_eq!(dst.level_count(), 3, "1x1x4 full mip chain");
    assert_eq!(h.update_volume_texture_hr(&src, &dst), 0, "UpdateTexture");

    assert_eq!(h.set_volume_texture(0, &dst), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    for (level, depth) in [(0, 4u8), (1, 2), (2, 1)] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        for z in 0..depth {
            let w = (f32::from(z) + 0.5) / f32::from(depth);
            assert_pixel_eq(
                sample_volume_depth(&h, w),
                GREEN,
                &format!("destination mip {level} slice {z}"),
            );
        }
    }
}

/// `UpdateTexture` rejects a source and destination of different resource types.
///
/// D3D9 pairs the two resources by type, and the vtable slot takes an
/// `IDirect3DBaseTexture9` on both sides, so a 2D texture and a volume texture
/// reach the same entry point. Separating only cube from non-cube lets a 2D
/// source write the first slice of a volume destination and report success,
/// leaving the deeper slices holding whatever was there.
#[test]
fn update_texture_rejects_a_resource_type_mismatch() {
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    let h = Harness::new();
    let (hr, primer) =
        h.try_create_volume_texture([2, 2, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0, "SYSTEMMEM volume");
    let primer = primer.expect("source volume");
    primer.write_u32(0, &[GREEN; 16]);
    let (hr, dst) = h.try_create_volume_texture([2, 2, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0, "DEFAULT volume");
    let dst = dst.expect("destination volume");
    assert_eq!(
        h.update_volume_texture_hr(&primer, &dst),
        0,
        "priming UpdateTexture"
    );

    let flat_src = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    flat_src.lock_rect(0, 0).write::<u32>(&[RED; 4]);
    assert_eq!(
        h.update_texture_into_volume_hr(&flat_src, &dst),
        D3DERR_INVALIDCALL,
        "2D source into a volume destination"
    );
    let flat_dst = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_volume_into_texture_hr(&primer, &flat_dst),
        D3DERR_INVALIDCALL,
        "volume source into a 2D destination"
    );

    assert_eq!(h.set_volume_texture(0, &dst), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    assert_pixel_eq(
        sample_volume_depth(&h, 0.125),
        GREEN,
        "first slice after the rejected update",
    );
}

/// `UpdateTexture` accepts a volume pair whose levels hold a single depth slice.
///
/// A `CreateVolumeTexture` resource one slice deep is backed by a 2D Metal
/// texture, so pairing the two resources on the backing kind rejects a pair
/// D3D9 accepts. The type is the one the create call asked for.
#[test]
fn update_texture_accepts_a_single_slice_volume_pair() {
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let (hr, src) =
        h.try_create_volume_texture([2, 2, 1], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0, "SYSTEMMEM volume");
    let src = src.expect("source volume");
    src.write_u32(0, &[GREEN; 4]);
    let (hr, dst) = h.try_create_volume_texture([2, 2, 1], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0, "DEFAULT volume");
    let dst = dst.expect("destination volume");
    assert_eq!(h.update_volume_texture_hr(&src, &dst), 0, "UpdateTexture");
}

/// A level from `GetVolumeLevel` locked and written reaches the texture.
///
/// The level shell forwards to the parent's per-level lock, so the write
/// lands in the staging the texture uploads from; every slice samples back.
#[test]
fn volume_level_lock_box_writes_reach_the_texture() {
    const SLICE_COLORS: [u32; 4] = [0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FFFF];
    let h = Harness::new();
    let (hr, tex) = h.try_create_volume_texture([2, 2, 4], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(hr, 0, "MANAGED volume");
    let tex = tex.expect("volume texture");
    let texels: Vec<u32> = SLICE_COLORS.iter().flat_map(|&color| [color; 4]).collect();
    {
        let (hr, level) = tex.get_volume_level(0);
        assert_eq!(hr, 0, "GetVolumeLevel");
        let level = level.expect("volume level");
        let (hr, desc) = level.desc();
        assert_eq!(hr, 0, "IDirect3DVolume9::GetDesc");
        assert_eq!((desc.width, desc.height, desc.depth), (2, 2, 4));
        assert_eq!(desc.pool, D3DPOOL_MANAGED);
        level.write_u32(&texels);
    }

    assert_eq!(h.set_volume_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    for (z, expected) in (0u8..).zip(SLICE_COLORS) {
        let w = (f32::from(z) + 0.5) / 4.0;
        assert_pixel_eq(
            sample_volume_depth(&h, w),
            expected,
            &format!("volume slice {z} written through GetVolumeLevel"),
        );
    }
}

/// `D3DFMT_V8U8` must sample its content, not black.
///
/// Signed two-channel, → `Rg8Snorm` with {R,G,1,1} swizzle. A 1x1 texel of
/// signed (+1,+1) reads as (1,1,1,1) → white. Confirms `V8U8`
/// create/upload/sample works in isolation (a full FF-alpha +
/// per-texel-bias `V8U8` setup is not covered here).
#[test]
fn v8u8_signed_texture_samples_nonzero() {
    let h = Harness::new();
    if h.device_is_paravirtual() {
        // The paravirtual device samples a swizzle view through the base
        // texture's lanes, so the lane this format fills by swizzle reads the
        // stored byte there.
        return;
    }
    // Signed bytes: 0x7F = +127 ≈ +1.0 in each channel.
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_V8U8, 0);
    tex.lock_rect(0, 0).write::<u8>(&[0x7F, 0x7F]);
    let px = sample_center(&h, &tex);
    assert!(
        px.r > 200 && px.g > 200 && px.b > 200,
        "V8U8 (+1,+1) must sample ~white via {{R,G,1,1}}; got {px:?}"
    );
}

/// A quad spanning `[x0, x1]` horizontally (full height).
///
/// UVs over the unit square, white vertex colour.
const fn horizontal_quad(x0: f32, x1: f32) -> [TexturedVertex; 6] {
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
        v(x0, 1.0, 0.0, 0.0),
        v(x1, 1.0, 1.0, 0.0),
        v(x0, -1.0, 0.0, 1.0),
        v(x1, 1.0, 1.0, 0.0),
        v(x1, -1.0, 1.0, 1.0),
        v(x0, -1.0, 0.0, 1.0),
    ]
}

/// Per-draw texture versioning: the first draw must NOT sample the later write.
///
/// A texture re-locked and rewritten BETWEEN two draws of ONE presented
/// frame must show each draw the content it had at that draw's point in
/// the command stream. Native D3D9 uploads managed textures at draw
/// validation (each draw sees the content current at that point in the
/// command stream); our upload blits all execute frame-head (before every
/// pass), so the encoder renames the `MTLTexture` at overlap instead
/// (fresh handle for later draws, earlier draws keep the old content).
/// Without the rename both halves collapse to the frame-final bytes and
/// the left half reads blue.
#[test]
fn intra_frame_relock_keeps_per_draw_content() {
    let h = Harness::new();
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, 0);
    tex.lock_rect(0, 0).write_u32(&[0xFFFF_0000]); // version 1: red
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    let left = horizontal_quad(-1.0, 0.0);
    let right = horizontal_quad(0.0, 1.0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &left),
            0,
            "left draw (version 1)"
        );
        // Rewrite the texel mid-frame, between the two draws.
        tex.lock_rect(0, 0).write_u32(&[0xFF00_00FF]); // version 2: blue
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &right),
            0,
            "right draw (version 2)"
        );
    });
    let l = Rgba8::from_pixel(h.read_pixel(160, 240));
    let r = Rgba8::from_pixel(h.read_pixel(480, 240));
    assert!(
        l.r > 200 && l.g < 50 && l.b < 50,
        "left half must keep the pre-relock red (per-draw versioning), got {l:?}"
    );
    assert!(
        r.r < 50 && r.g < 50 && r.b > 200,
        "right half must sample the post-relock blue, got {r:?}"
    );
}

/// Bind `tex` and read back the single texel at `(u, v)`.
///
/// Every vertex of the quad carries the same texture coordinate, so with point
/// filtering the whole backbuffer is that one texel and the centre pixel reads
/// it. Lets a test address one texel of a texture whose two dimensions differ.
fn sample_texel(h: &Harness, tex: &Texture<'_>, u: f32, v: f32) -> u32 {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    let vertex = |x: f32, y: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u,
        v,
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
            "texel sample draw"
        );
    });
    h.read_pixel(320, 240)
}

/// `UpdateTexture` from a transposed source copies the region both levels share.
///
/// D3D9 pairs source and destination mips on the larger of width and height, so
/// a 2x4 source pairs with a 4x2 destination and the call succeeds. Only the 2x2
/// overlap is defined. A copy driven by the source extent alone runs four rows
/// into a two-row destination, walks off the end of its staging part-way
/// through, and abandons the whole update, leaving the destination untouched.
#[test]
fn update_texture_from_a_transposed_source_copies_the_shared_region() {
    const GREEN: u32 = 0xFF00_FF00;
    const RED: u32 = 0xFFFF_0000;
    let h = Harness::new();
    let dst = h.create_texture(4, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let primer = h.create_texture(4, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    primer.lock_rect(0, 0).write::<u32>(&[GREEN; 8]);
    assert_eq!(
        h.update_texture_hr(&primer, &dst),
        0,
        "priming UpdateTexture"
    );
    assert_pixel_eq(sample_texel(&h, &dst, 0.125, 0.25), GREEN, "primed texel");

    let transposed = h.create_texture(2, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    transposed.lock_rect(0, 0).write::<u32>(&[RED; 8]);
    assert_eq!(
        h.update_texture_hr(&transposed, &dst),
        0,
        "transposed UpdateTexture"
    );
    assert_pixel_eq(
        sample_texel(&h, &dst, 0.125, 0.25),
        RED,
        "shared texel (0,0)",
    );
    assert_pixel_eq(
        sample_texel(&h, &dst, 0.375, 0.75),
        RED,
        "shared texel (1,1)",
    );
    assert_pixel_eq(
        sample_texel(&h, &dst, 0.625, 0.25),
        GREEN,
        "texel (2,0) kept",
    );
    assert_pixel_eq(
        sample_texel(&h, &dst, 0.875, 0.75),
        GREEN,
        "texel (3,1) kept",
    );
}

#[test]
fn get_dc_on_an_odd_width_16_bit_texture_level_round_trips_a_texel() {
    // A row of an odd number of 2-byte texels is not a whole number of dwords,
    // and GDI steps a DIB by the row length rounded up to four bytes, rejecting
    // any pitch below that. A texture level's staging carries that same stride,
    // so the DIB aliases it directly; a level two bytes short of it would start
    // every row late and run the last one off the end of the allocation.
    const W: u32 = 33;
    const H: u32 = 4;
    const GREEN_565: u16 = 0x07E0;
    const RED_565: u16 = 0xF800;
    const GREEN_COLORREF: u32 = 0x0000_FF00;
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let tex = h.create_texture(W, H, 1, 0, D3DFMT_R5G6B5, D3DPOOL_MANAGED);
    {
        let mut locked = tex.lock_rect(0, 0);
        let pitch = locked.pitch();
        assert_eq!(
            pitch,
            (W * 2).next_multiple_of(4).cast_signed(),
            "the level locks at the dword-rounded stride GDI derives for its DIB"
        );
        let pitch_px = pitch.cast_unsigned() / 2;
        let seed = vec![GREEN_565; (pitch_px * H) as usize];
        locked.write(&seed);
    }

    // The last texel of the last row is the one a DIB over the tighter staging
    // never reaches: its row starts two bytes late and runs off the end.
    let (last_x, last_y) = ((W - 1).cast_signed(), (H - 1).cast_signed());
    let surface = tex.surface_level(0);
    let dc = surface.dc();
    assert_eq!(
        dc.get_pixel(last_x, last_y),
        GREEN_COLORREF,
        "the DC reads the texels the lock wrote, last row included",
    );
    assert_eq!(
        dc.set_pixel(last_x, last_y, RED_COLORREF),
        RED_COLORREF,
        "SetPixel stores full-scale channels exactly in a 5-6-5 DIB",
    );
    assert_eq!(dc.release(), 0, "ReleaseDC");

    {
        let locked = tex.lock_rect(0, mtld3d_types::D3DLOCK_READONLY);
        let pitch_px = locked.pitch().cast_unsigned() as usize / 2;
        let texels = locked.as_u16(pitch_px * H as usize);
        assert_eq!(
            texels[pitch_px * (H as usize - 1) + W as usize - 1],
            RED_565,
            "what GDI drew into the last texel reached the level's staging",
        );
        assert_eq!(
            texels[0], GREEN_565,
            "the texels GDI left alone kept the lock's own pixels",
        );
    }

    // The quad spans the unit square over a 640x480 target, so texel (32, 3)
    // covers roughly x 621..640, y 360..480: read the middle of that band, well
    // clear of its edges.
    let last = sample_at(&h, &tex, 630, 420);
    assert!(
        last.r > 200 && last.g < 50 && last.b < 50,
        "a draw samples the last texel GDI drew, so the level reached the GPU, got {last:?}"
    );
    let untouched = sample_at(&h, &tex, 10, 60);
    assert!(
        untouched.r < 50 && untouched.g > 200 && untouched.b < 50,
        "the texels GDI left alone still sample as the lock wrote them, got {untouched:?}"
    );
}

/// A single-slice default-pool volume takes a second `UpdateTexture`.
///
/// A `CreateVolumeTexture` of depth 1 is backed by a plain 2D Metal texture, so
/// nothing but the creation call tells it apart from an ordinary 2D texture,
/// and the class that releases its staging after an upload must still exclude
/// it: the volume paths write and upload a level whole and re-create it as a
/// single 2D slice. The first update is drawn and its frame answered (the
/// point a released level would go), then a second update has to reach the GPU.
#[test]
fn single_slice_default_volume_takes_a_second_update_after_its_upload() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let (hr, src) =
        h.try_create_volume_texture([2, 2, 1], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(hr, 0, "SYSTEMMEM volume");
    let src = src.expect("source volume");
    let (hr, dst) = h.try_create_volume_texture([2, 2, 1], 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(hr, 0, "DEFAULT volume");
    let dst = dst.expect("destination volume");

    src.write_u32(0, &[RED; 4]);
    assert_eq!(
        h.update_volume_texture_hr(&src, &dst),
        0,
        "first UpdateTexture"
    );
    assert_eq!(h.set_volume_texture(0, &dst), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0,
        "SetFVF"
    );
    assert_pixel_eq(sample_volume_depth(&h, 0.5), RED, "first fill");
    release_uploaded_staging(&h);

    src.write_u32(0, &[GREEN; 4]);
    assert_eq!(
        h.update_volume_texture_hr(&src, &dst),
        0,
        "second UpdateTexture"
    );
    assert_pixel_eq(sample_volume_depth(&h, 0.5), GREEN, "second fill");
}

/// `UpdateSurface` converts a mismatched format pair instead of rejecting it.
///
/// D3D9 accepts a source and a destination whose formats differ and converts
/// the texels; only a pair no codec covers fails. The source's undefined alpha
/// byte reads as opaque, so what lands in the destination is the source colour.
#[test]
fn update_surface_converts_x8r8g8b8_into_a8r8g8b8() {
    const RED: u32 = 0x00FF_0000;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u32>(&[RED; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface across formats"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFFFF_0000,
        "converted texel",
    );
}

/// `UpdateSurface` re-encodes a 16-bit source into a 32-bit destination.
///
/// The two formats disagree on bytes per texel, so a raw copy of the source
/// rows would land two texels of source in one texel of destination and read
/// half the level. Each channel widens by bit replication, so a saturated
/// source channel arrives at 255.
#[test]
fn update_surface_converts_r5g6b5_into_x8r8g8b8() {
    // R=0, G=63, B=0.
    const GREEN_565: u16 = 0x07E0;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_R5G6B5, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u16>(&[GREEN_565; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface across formats"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFF00_FF00,
        "converted texel",
    );
}

/// `UpdateTexture` converts a mismatched format pair over the whole mip.
///
/// The destination is half the bytes per texel of the source, so the copy has
/// to re-encode rather than move rows: a saturated 8-bit channel packs into
/// the 5- and 6-bit lanes and widens back to 255 when it is sampled.
#[test]
fn update_texture_converts_a8r8g8b8_into_r5g6b5() {
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u32>(&[GREEN; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_R5G6B5, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        0,
        "UpdateTexture across formats"
    );
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "converted texel");
}

/// `UpdateSurface` re-encodes a packed 16-bit source with an alpha bit.
///
/// `A1R5G5B5` splits its 16 bits differently from the `R5G6B5` the codec
/// already carried: five bits of green instead of six, and the top bit an
/// alpha channel rather than part of red. A source texel that is opaque green
/// arrives as opaque green rather than as the colour a 5-6-5 reader would see.
#[test]
fn update_surface_converts_a1r5g5b5_into_a8r8g8b8() {
    // A=1, R=0, G=31, B=0.
    const GREEN_1555: u16 = 0x83E0;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_A1R5G5B5, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u16>(&[GREEN_1555; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface from a 1-5-5-5 source"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFF00_FF00,
        "converted texel",
    );
}

/// `UpdateTexture` encodes into a packed 16-bit destination.
///
/// The destination is half the bytes per texel of the source and packs its
/// channels four bits each, so the copy has to re-encode. A saturated 8-bit
/// channel packs into a saturated nibble and widens back to 255 when it is
/// sampled.
#[test]
fn update_texture_converts_a8r8g8b8_into_a4r4g4b4() {
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u32>(&[GREEN; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A4R4G4B4, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        0,
        "UpdateTexture into a 4-4-4-4 destination"
    );
    assert_pixel_eq(sample_center(&h, &dst).to_pixel(), GREEN, "converted texel");
}

/// `UpdateSurface` re-encodes a single-channel luminance source.
///
/// `L8` is one byte per texel against the destination's four, and its one
/// channel stands for all three colour channels. The destination texel is the
/// grey the source luminance names, with alpha opaque.
#[test]
fn update_surface_converts_l8_into_a8r8g8b8() {
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_L8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u8>(&[0x80; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface from a luminance source"
    );
    assert_pixel_approx(
        sample_center(&h, &dst).to_pixel(),
        0xFF80_8080,
        2,
        "converted texel",
    );
}

/// `UpdateSurface` re-encodes a 24-bit source into a 32-bit destination.
///
/// `R8G8B8` is three bytes a texel against the destination's four, so the row
/// the source hands over is neither the destination's length nor a whole
/// number of destination texels. Each texel gains the opaque alpha the source
/// has no channel for.
#[test]
fn update_surface_converts_r8g8b8_into_a8r8g8b8() {
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_R8G8B8, D3DPOOL_SYSTEMMEM);
    {
        let mut locked = src.lock_rect(0, 0);
        assert_eq!(locked.pitch(), 12, "4 texels of three bytes, dword aligned");
        // B, G, R per texel: opaque red.
        locked.write::<u8>(&[0x00, 0x00, 0xFF].repeat(16));
    }
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface from a 24-bit source"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFFFF_0000,
        "converted texel",
    );
}

/// `UpdateSurface` re-encodes a reversed-channel source.
///
/// `A8B8G8R8` and `A8R8G8B8` are the same four bytes a texel with red and blue
/// exchanged, so a raw copy would land the source colour with those two
/// channels swapped. The conversion keeps the colour.
#[test]
fn update_surface_converts_a8b8g8r8_into_a8r8g8b8() {
    // R=255, G=0, B=0, A=255 in ascending addresses.
    const RED_ABGR: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_A8B8G8R8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write::<u32>(&[RED_ABGR; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src.surface_level(0), &dst.surface_level(0)),
        0,
        "UpdateSurface from a reversed-channel source"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFFFF_0000,
        "converted texel",
    );
}

/// `UpdateTexture` still rejects a format pair the CPU codec cannot convert.
///
/// The codec covers the simple uncompressed colour formats. A
/// block-compressed source has no decoder here, so rather than writing
/// reinterpreted bytes the call answers `D3DERR_INVALIDCALL`.
#[test]
fn update_texture_rejects_a_format_pair_with_no_converter() {
    let h = Harness::new();
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_texture_hr(&src, &dst),
        D3DERR_INVALIDCALL,
        "UpdateTexture from a compressed source"
    );
}

/// `UpdateSurface` converts a mismatched pair from a standalone source too.
///
/// The source is a system-memory offscreen-plain surface, which is not
/// texture-backed and so reaches the destination level's staging directly
/// rather than through the texture-to-texture path. D3D9 converts a mismatched
/// format pair on both. The source's undefined alpha byte reads as opaque, so
/// what lands in the destination is the source colour.
#[test]
fn update_surface_converts_a_standalone_source_into_another_format() {
    // X8R8G8B8 red, with the ignored byte left at zero.
    const RED_X8: u32 = 0x00FF_0000;
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[RED_X8; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src, &dst.surface_level(0)),
        0,
        "UpdateSurface from a standalone source across formats"
    );
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        0xFFFF_0000,
        "converted texel",
    );
}

/// A partial converting standalone update preserves GPU-written pixels outside its rectangle.
///
/// `ColorFill` writes only the destination's Metal texture. The later
/// `UpdateSurface` converts into CPU staging, so it must read the GPU-owned
/// level back before changing the top-left quadrant.
#[test]
fn a_partial_converting_standalone_update_preserves_gpu_pixels() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN_X8: u32 = 0x0000_FF00;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dst = h.create_texture(
        4,
        4,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(h.color_fill_hr(&dst.surface_level(0), RED), 0, "GPU fill");
    for pixel in sample_quadrants(&h, &dst) {
        assert_pixel_eq(pixel, RED, "completed GPU fill");
    }

    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[GREEN_X8; 16]);
    let rect = D3DRECT {
        x1: 0,
        y1: 0,
        x2: 2,
        y2: 2,
    };
    assert_eq!(
        h.update_surface_region_hr(&src, &rect, &dst.surface_level(0), (0, 0)),
        0,
        "partial converting UpdateSurface"
    );

    let sampled = sample_quadrants(&h, &dst);
    assert_pixel_eq(sampled[0], GREEN, "converted quadrant");
    for pixel in &sampled[1..] {
        assert_pixel_eq(*pixel, RED, "GPU pixel outside the converted rectangle");
    }
}

/// A whole converting standalone update remains authoritative before a later raw update.
///
/// The conversion replaces the whole GPU-owned level with blue in staging.
/// A following raw update of the top-left quadrant must preserve that blue,
/// rather than reading the older red GPU contents back over it.
#[test]
fn a_converting_standalone_update_survives_a_later_partial_raw_update() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE_X8: u32 = 0x0000_00FF;
    const BLUE: u32 = 0xFF00_00FF;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dst = h.create_texture(
        4,
        4,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    assert_eq!(h.color_fill_hr(&dst.surface_level(0), RED), 0, "GPU fill");
    for pixel in sample_quadrants(&h, &dst) {
        assert_pixel_eq(pixel, RED, "completed GPU fill");
    }

    let whole = h.create_offscreen_plain_surface(4, 4, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    whole.lock_rect(0).write_u32(&[BLUE_X8; 16]);
    assert_eq!(
        h.update_surface_hr(&whole, &dst.surface_level(0)),
        0,
        "whole converting UpdateSurface"
    );
    let partial = h.create_offscreen_plain_surface(2, 2, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    partial.lock_rect(0).write_u32(&[GREEN; 4]);
    assert_eq!(
        h.update_surface_hr(&partial, &dst.surface_level(0)),
        0,
        "later raw UpdateSurface"
    );

    let sampled = sample_quadrants(&h, &dst);
    assert_pixel_eq(sampled[0], GREEN, "raw quadrant");
    for pixel in &sampled[1..] {
        assert_pixel_eq(*pixel, BLUE, "converted pixel outside the later raw update");
    }
}

/// The converting standalone-source `UpdateSurface` reaches a cube face.
///
/// A cube destination takes its own staging path, keyed by face, so it needs
/// the conversion as much as the 2D one. Only the addressed face is written.
#[test]
fn update_surface_converts_a_standalone_source_into_a_cube_face() {
    // X8R8G8B8 green, with the ignored byte left at zero.
    const GREEN_X8: u32 = 0x0000_FF00;
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[GREEN_X8; 16]);
    let cube = h.create_cube_texture_owned(4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    let negative_x = cube.surface(1, 0);
    assert_eq!(
        h.update_surface_hr(&src, &negative_x),
        0,
        "UpdateSurface from a standalone source into a cube face"
    );
    assert_pixel_eq(
        sample_cube_x(&h, &cube, -1.0),
        0xFF00_FF00,
        "converted cube-face texel",
    );
}

/// A standalone-source `UpdateSurface` into a compressed destination is rejected.
///
/// The codec encodes the simple uncompressed colour formats only, so a `DXT1`
/// destination has no encoder here and the call answers `D3DERR_INVALIDCALL`
/// rather than writing 32bpp rows over compressed blocks.
#[test]
fn update_surface_rejects_a_standalone_source_into_a_compressed_destination() {
    let h = Harness::new();
    let src = h.create_offscreen_plain_surface(4, 4, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0).write_u32(&[0x00FF_0000; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
    assert_eq!(
        h.update_surface_hr(&src, &dst.surface_level(0)),
        D3DERR_INVALIDCALL,
        "UpdateSurface into a block-compressed destination"
    );
}

/// A `Levels` past the chain the extent allows resolves to that chain.
///
/// D3D9 measures a mip chain at `floor(log2(max_dim)) + 1` levels, counting
/// depth for a volume, and every level past it would only repeat the last
/// one. A create that asks for more gets the chain its dimensions do have,
/// through the same resolution the colour, depth, cube and volume paths share;
/// the request at exactly the natural count is unchanged. All four are
/// `D3DPOOL_DEFAULT`, the pool that backs the create with a real Metal
/// texture, whose level count is the one an over-long request would exceed.
#[test]
fn a_level_count_past_the_natural_chain_resolves_to_the_chain() {
    let h = Harness::new();

    // 64x64: seven levels, 64 down to 1.
    for (levels, expected) in [(8u32, 7u32), (7, 7)] {
        let tex = h.create_texture(64, 64, levels, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(tex.level_count(), expected, "CreateTexture levels={levels}");
        let (hr, _) = tex.level_desc(expected - 1);
        assert_eq!(hr, 0, "last level of the chain, levels={levels}");
        let (hr, _) = tex.level_desc(expected);
        assert_eq!(hr, D3DERR_INVALIDCALL, "past the chain, levels={levels}");
    }

    // The chain the resolution settles on has to be one Metal will allocate,
    // which is what an over-long request handed straight through would not be:
    // fill level 0 of a managed texture created that way and sample it, so the
    // bind creates the backing texture the request would have failed.
    let tex = h.create_texture(64, 64, 8, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(tex.level_count(), 7, "managed CreateTexture levels=8");
    {
        let mut locked = tex.lock_rect(0, 0);
        assert_eq!(locked.pitch(), 64 * 4, "64px * 4 bytes/px row pitch");
        locked.write_u32(&[0xFFFF_0000u32; 64 * 64]);
    }
    let center = sample_center(&h, &tex);
    assert!(
        center.r > 200 && center.g < 50 && center.b < 50,
        "the clamped chain samples level 0, got {center:?}"
    );

    // A 64x64 depth texture measures the same chain as its colour twin.
    for (levels, expected) in [(8u32, 7u32), (7, 7)] {
        let tex = h.create_texture(
            64,
            64,
            levels,
            D3DUSAGE_DEPTHSTENCIL,
            D3DFMT_INTZ,
            D3DPOOL_DEFAULT,
        );
        assert_eq!(
            tex.level_count(),
            expected,
            "depth CreateTexture levels={levels}"
        );
    }

    // A 32-texel cube face: six levels.
    for (levels, expected) in [(7u32, 6u32), (6, 6)] {
        let cube = h.create_cube_texture_owned(32, levels, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(
            cube.level_count(),
            expected,
            "CreateCubeTexture levels={levels}"
        );
    }

    // 8x4x2: the chain runs on the 8-texel width, four levels, and the depth
    // halves with it.
    for (levels, expected) in [(5u32, 4u32), (4, 4)] {
        let (hr, volume) =
            h.try_create_volume_texture([8, 4, 2], levels, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
        assert_eq!(hr, 0, "CreateVolumeTexture levels={levels}");
        let volume = volume.expect("volume texture");
        assert_eq!(
            volume.level_count(),
            expected,
            "CreateVolumeTexture levels={levels}"
        );
        let (hr, desc) = volume.level_desc(expected - 1);
        assert_eq!(hr, 0, "last level of the chain, levels={levels}");
        assert_eq!(
            (desc.width, desc.height, desc.depth),
            (1, 1, 1),
            "the chain ends at one texel in every extent"
        );
        let (hr, _) = volume.level_desc(expected);
        assert_eq!(hr, D3DERR_INVALIDCALL, "past the chain, levels={levels}");
    }
}

/// Fill a 2x2 level with four texels of `bytes_per_texel` bytes each.
///
/// Row by row at the pitch the lock reports: a 2x2 level of a three-byte
/// format holds six bytes in an eight-byte row, so a writer that packs the
/// four texels contiguously puts the second row in the first row's padding.
fn fill_2x2(tex: &Texture<'_>, texels: [&[u8]; 4]) {
    let locked = tex.lock_rect(0, 0);
    let pitch = usize::try_from(locked.pitch()).expect("a positive row pitch");
    let bytes_per_texel = texels[0].len();
    for (index, texel) in texels.into_iter().enumerate() {
        assert_eq!(texel.len(), bytes_per_texel, "one texel size for the level");
        let offset = (index / 2) * pitch + (index % 2) * bytes_per_texel;
        // SAFETY: the lock maps two rows of `pitch` bytes and a row holds two
        // texels, so `offset + bytes_per_texel` stays inside the mapping.
        let dst = unsafe { locked.bits_ptr().add(offset) };
        // SAFETY: `dst` addresses `bytes_per_texel` writable bytes of the
        // level (above), disjoint from the caller's `texel`.
        unsafe { core::ptr::copy_nonoverlapping(texel.as_ptr(), dst, bytes_per_texel) };
    }
}

/// Assert the four quadrants of the back buffer read red, green, blue, white.
fn assert_2x2_quadrants(h: &Harness, tex: &Texture<'_>, what: &str) {
    let [tl, tr, bl, br] = sample_points(h, tex, [(160, 120), (480, 120), (160, 360), (480, 360)])
        .map(Rgba8::from_pixel);
    assert!(
        tl.r > 200 && tl.g < 50 && tl.b < 50,
        "{what} top-left red, got {tl:?}"
    );
    assert!(
        tr.r < 50 && tr.g > 200 && tr.b < 50,
        "{what} top-right green, got {tr:?}"
    );
    assert!(
        bl.r < 50 && bl.g < 50 && bl.b > 200,
        "{what} bottom-left blue, got {bl:?}"
    );
    assert!(
        br.r > 200 && br.g > 200 && br.b > 200,
        "{what} bottom-right white, got {br:?}"
    );
}

/// The reversed-channel 32-bit pair round-trips through a lock and a sample.
///
/// `A8B8G8R8` and `X8B8G8R8` store R, G, B, then the alpha or padding byte in
/// ascending addresses, the reverse of the `A8R8G8B8` family. A mapping that
/// reused the BGRA backing would swap red and blue in every quadrant.
#[test]
fn reversed_channel_formats_sample_their_texels() {
    let h = Harness::new();
    for (format, texels) in [
        (
            D3DFMT_A8B8G8R8,
            [0xFF00_00FFu32, 0xFF00_FF00, 0xFFFF_0000, 0xFFFF_FFFF],
        ),
        (
            D3DFMT_X8B8G8R8,
            [0x0000_00FFu32, 0x0000_FF00, 0x00FF_0000, 0x00FF_FFFF],
        ),
    ] {
        let tex = h.create_texture(2, 2, 1, 0, format, D3DPOOL_MANAGED);
        {
            let mut locked = tex.lock_rect(0, 0);
            assert_eq!(locked.pitch(), 8, "2 texels of four bytes");
            locked.write_u32(&texels);
        }
        assert_2x2_quadrants(&h, &tex, &format!("{format:#x}"));
    }
}

/// `X8B8G8R8` samples alpha as 1 whatever the padding byte holds.
///
/// The same rule `X8R8G8B8` follows: the fourth byte is "don't care", so a
/// SRC_ALPHA-blended draw must come through at full strength rather than
/// multiplying by the stored zero.
#[test]
fn x8b8g8r8_samples_alpha_as_one() {
    let h = Harness::new();
    if h.device_is_paravirtual() {
        // The paravirtual device samples a swizzle view through the base
        // texture's lanes, so the lane this format fills by swizzle reads the
        // stored byte there.
        return;
    }
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_X8B8G8R8, D3DPOOL_MANAGED);
    // R = 0xFF, G = 0x40, B = 0x20, padding byte 0x00.
    tex.lock_rect(0, 0).write_u32(&[0x0020_40FF]);
    assert_eq!(h.set_texture(0, &tex), 0, "SetTexture");
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        0
    );
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA), 0);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ZERO), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        0,
        "SetFVF"
    );
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    let px = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert!(
        px.r > 200 && (0x30..=0x50).contains(&px.g) && (0x10..=0x30).contains(&px.b),
        "the padding byte must not scale the draw, got {px:?}"
    );
}

/// 24-bit `R8G8B8` round-trips through a lock, the GPU widening, and a sample.
///
/// No GPU family has a three-byte colour format, so the level is backed by
/// BGRA8 and widened by the upload pass. The lock keeps the D3D9 layout: three
/// bytes a texel in B, G, R order, at a row pitch rounded up to a dword, so a
/// 2x2 level strides eight bytes over six bytes of texels.
#[test]
fn r8g8b8_expands_and_samples_its_texels() {
    let h = Harness::new();
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(
        tex.lock_rect(0, 0).pitch(),
        8,
        "2 texels of three bytes, rounded up to a dword"
    );
    fill_2x2(
        &tex,
        [
            &[0x00, 0x00, 0xFF],
            &[0x00, 0xFF, 0x00],
            &[0xFF, 0x00, 0x00],
            &[0xFF, 0xFF, 0xFF],
        ],
    );
    assert_2x2_quadrants(&h, &tex, "R8G8B8");
}

/// An untouched mip uploaded in the same frame survives a texture rename.
#[test]
fn intra_frame_rename_preserves_earlier_mip_upload() {
    const RED: u32 = 0xFFFF_0000;
    const YELLOW: u32 = 0xFFFF_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let texture = h.create_texture(4, 4, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    texture.lock_rect(0, 0).write_u32(&[RED; 16]);
    texture.lock_rect(1, 0).write_u32(&[YELLOW; 4]);
    assert_eq!(h.set_texture(0, &texture), 0);
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &horizontal_quad(-1.0, 0.0)),
            0
        );
        texture.lock_rect(0, 0).write_u32(&[BLUE; 16]);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &horizontal_quad(0.0, 1.0)),
            0
        );
    });
    assert_eq!(
        [h.read_pixel(160, 240), h.read_pixel(480, 240)],
        [RED, BLUE],
        "updated mip keeps per-draw contents"
    );
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    assert_pixel_eq(
        sample_texel(&h, &texture, 0.5, 0.5),
        YELLOW,
        "earlier upload of untouched mip survives rename",
    );
}

/// A partial update preserves texels an earlier upload pass wrote outside its rectangle.
#[test]
fn intra_frame_rename_preserves_earlier_upload_outside_patch() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    let texture = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    texture.lock_rect(0, 0).write_u32(&[RED; 4]);
    let patch = h.create_offscreen_plain_surface(1, 1, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    patch.lock_rect(0).write_u32(&[BLUE]);
    let level = texture.surface_level(0);
    assert_eq!(h.set_texture(0, &texture), 0);
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &horizontal_quad(-1.0, 0.0)),
            0
        );
        assert_eq!(d.update_surface_hr(&patch, &level), 0);
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &horizontal_quad(0.0, 1.0)),
            0
        );
    });
    // The earlier sample stays outside the patch, so an in-place staging
    // update cannot affect this control.
    assert_pixel_eq(h.read_pixel(240, 360), RED, "earlier untouched texel");
    assert_pixel_eq(h.read_pixel(400, 120), BLUE, "updated texel");
    assert_pixel_eq(h.read_pixel(560, 360), RED, "preserved untouched texel");
}

/// Each renamed autogen texture generates its mip chain after its own upload.
#[test]
fn intra_frame_rename_keeps_generated_mips_in_upload_order() {
    const COLORS: [u32; 3] = [0xFFFF_0000, 0xFF00_00FF, 0xFF00_FF00];
    const EDGES: [f32; 4] = [-1.0, -0.333_333_34, 0.333_333_34, 1.0];
    let h = Harness::new();
    let texture = h.create_texture(
        2,
        2,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    assert_eq!(h.set_texture(0, &texture), 0);
    h.select_texture_stage(0);
    point_clamp(&h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    h.render_once(BLACK, |d| {
        for (index, color) in COLORS.into_iter().enumerate() {
            texture.lock_rect(0, 0).write_u32(&[color; 4]);
            assert_eq!(
                d.draw_primitive_up(
                    D3DPT_TRIANGLELIST,
                    2,
                    &horizontal_quad(EDGES[index], EDGES[index + 1]),
                ),
                0
            );
        }
    });
    for (x, color) in [106, 320, 533].into_iter().zip(COLORS) {
        assert_pixel_eq(h.read_pixel(x, 240), color, "generated mip at its draw");
    }
}

#[test]
fn plain_depth_textures_preserve_the_gpu_only_resource_contract() {
    let h = Harness::new();
    for format in [
        mtld3d_types::D3DFMT_D16_LOCKABLE,
        mtld3d_types::D3DFMT_D32F_LOCKABLE,
        mtld3d_types::D3DFMT_D24FS8,
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, mtld3d_types::D3DRTYPE_TEXTURE, format),
            mtld3d_types::D3DERR_NOTAVAILABLE
        );
        let (hr, ptr) = h.try_create_texture(17, 9, 1, 0, format, D3DPOOL_DEFAULT);
        assert_eq!(hr, D3DERR_INVALIDCALL);
        assert!(ptr.is_null());
    }
    for format in [
        mtld3d_types::D3DFMT_D16,
        mtld3d_types::D3DFMT_D24X8,
        mtld3d_types::D3DFMT_D24S8,
        mtld3d_types::D3DFMT_D32,
        D3DFMT_INTZ,
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, mtld3d_types::D3DRTYPE_TEXTURE, format),
            0,
            "plain depth format {format}"
        );
        let memory_before = h.available_texture_mem();
        let tex = h.create_texture(17, 9, 0, 0, format, D3DPOOL_DEFAULT);
        let memory_with_plain = h.available_texture_mem();
        let attachment = h.create_texture(17, 9, 0, D3DUSAGE_DEPTHSTENCIL, format, D3DPOOL_DEFAULT);
        assert_eq!(
            memory_before - memory_with_plain,
            memory_with_plain - h.available_texture_mem()
        );
        drop(attachment);
        assert_eq!(tex.level_count(), 5);
        for level in 0..5 {
            let (hr, desc) = tex.level_desc(level);
            assert_eq!(hr, 0);
            assert_eq!(
                (desc.width, desc.height),
                ((17 >> level).max(1), (9 >> level).max(1))
            );
            assert_eq!(
                (desc.format, desc.usage, desc.pool),
                (format, 0, D3DPOOL_DEFAULT)
            );
            assert_eq!(tex.lock_rect_probe(level, 0), (D3DERR_INVALIDCALL, false));
            let surface = tex.surface_level(level);
            assert_eq!(surface.lock_rect_probe(0), (D3DERR_INVALIDCALL, false));
            assert_eq!(h.set_depth_stencil_surface(&surface), D3DERR_INVALIDCALL);
            assert_eq!(h.set_render_target(0, &surface), D3DERR_INVALIDCALL);
        }
        assert_eq!(h.set_texture(0, &tex), 0);
        assert_eq!(h.clear_texture(0), 0);
        for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
            let (hr, ptr) = h.try_create_texture(17, 9, 1, 0, format, pool);
            assert_eq!(hr, D3DERR_INVALIDCALL);
            assert!(ptr.is_null());
        }
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_AUTOGENMIPMAP,
                mtld3d_types::D3DRTYPE_TEXTURE,
                format
            ),
            mtld3d_types::D3DOK_NOAUTOGEN
        );
        for levels in [0, 1] {
            let fallback = h.create_texture(
                17,
                9,
                levels,
                D3DUSAGE_AUTOGENMIPMAP,
                format,
                D3DPOOL_DEFAULT,
            );
            assert_eq!(fallback.level_count(), 1);
            let (hr, desc) = fallback.level_desc(0);
            assert_eq!(hr, 0);
            assert_eq!(
                (desc.width, desc.height, desc.usage),
                (17, 9, D3DUSAGE_AUTOGENMIPMAP)
            );
            assert_eq!(fallback.level_desc(1).0, D3DERR_INVALIDCALL);
            fallback.generate_mip_sub_levels();
        }
        let (hr, ptr) =
            h.try_create_texture(17, 9, 2, D3DUSAGE_AUTOGENMIPMAP, format, D3DPOOL_DEFAULT);
        assert_eq!(hr, D3DERR_INVALIDCALL);
        assert!(ptr.is_null());
        {
            let usage = D3DUSAGE_RENDERTARGET;
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    usage,
                    mtld3d_types::D3DRTYPE_TEXTURE,
                    format
                ),
                mtld3d_types::D3DERR_NOTAVAILABLE
            );
            let (hr, ptr) = h.try_create_texture(17, 9, 1, usage, format, D3DPOOL_DEFAULT);
            assert_eq!(hr, D3DERR_INVALIDCALL);
            assert!(ptr.is_null());
        }
        for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
            assert_eq!(
                h.create_offscreen_plain_surface_hr(17, 9, format, pool),
                D3DERR_INVALIDCALL
            );
        }
        for usage in [
            mtld3d_types::D3DUSAGE_QUERY_SRGBREAD,
            mtld3d_types::D3DUSAGE_QUERY_SRGBWRITE,
            mtld3d_types::D3DUSAGE_QUERY_VERTEXTEXTURE,
        ] {
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    usage,
                    mtld3d_types::D3DRTYPE_TEXTURE,
                    format
                ),
                mtld3d_types::D3DERR_NOTAVAILABLE
            );
        }
        for resource in [
            D3DRTYPE_SURFACE,
            mtld3d_types::D3DRTYPE_CUBETEXTURE,
            mtld3d_types::D3DRTYPE_VOLUMETEXTURE,
        ] {
            assert_eq!(
                h.check_device_format(D3DFMT_X8R8G8B8, 0, resource, format),
                mtld3d_types::D3DERR_NOTAVAILABLE
            );
        }
    }
}

/// Managed CPU edits without publication preserve the previously sampled image.
#[test]
fn managed_dirty_no_dirty_and_readonly_visibility() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let mut observed = Vec::new();
    for flags in [D3DLOCK_READONLY, D3DLOCK_NO_DIRTY_UPDATE] {
        let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
        assert_pixel_eq(sample_center(&h, &tex).to_pixel(), RED, "initialized");
        // The previous pixel read has retired the upload. This intentionally
        // writes through READONLY to mirror Wine's managed visibility test.
        tex.lock_rect(0, flags).write_u32(&[GREEN; 64 * 64]);
        let cpu = tex.lock_rect(0, D3DLOCK_READONLY).as_u32(1)[0];
        let gpu = sample_center(&h, &tex).to_pixel();
        observed.push((cpu, gpu));
    }
    assert_eq!(observed, vec![(GREEN, RED), (GREEN, RED)]);
}

/// Explicit publication after an unannounced CPU edit must narrow the upload.
#[test]
fn managed_dirty_add_dirty_rect_publication() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    assert_pixel_eq(sample_center(&h, &tex).to_pixel(), RED, "initialized");
    tex.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32(&[GREEN; 64 * 64]);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        RED,
        "unannounced CPU edit",
    );
    assert_eq!(tex.add_dirty_rect_partial(&[16, 16, 48, 48]), 0);
    let partial = (
        sample_center(&h, &tex).to_pixel(),
        sample_at(&h, &tex, 40, 30).to_pixel(),
    );
    assert_eq!(tex.add_dirty_rect(), 0);
    let full = (
        sample_center(&h, &tex).to_pixel(),
        sample_at(&h, &tex, 40, 30).to_pixel(),
    );
    assert_eq!((partial, full), ((GREEN, RED), (GREEN, GREEN)));
}

/// Initialization and eviction use the CPU image even without dirty updates.
#[test]
fn managed_dirty_initial_and_evicted_image() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    for flags in [D3DLOCK_READONLY, D3DLOCK_NO_DIRTY_UPDATE] {
        let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        tex.lock_rect(0, flags).write_u32(&[RED; 64 * 64]);
        assert_pixel_eq(
            sample_center(&h, &tex).to_pixel(),
            RED,
            "initial dirty image",
        );
        tex.lock_rect(0, flags).write_u32(&[GREEN; 64 * 64]);
        assert_eq!(h.evict_managed_resources(), 0);
        let gpu = sample_center(&h, &tex).to_pixel();
        assert_pixel_eq(gpu, GREEN, "eviction republishes CPU image");
    }
}

/// Independent mip pages are enough to reproduce explicit publication failure.
#[test]
fn managed_dirty_independent_mip_publication() {
    const RED: u32 = 0xFFFF_0000;
    const BLUE: u32 = 0xFF00_00FF;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    let tex = h.create_texture(64, 64, 2, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    tex.lock_rect(1, 0).write_u32(&[BLUE; 32 * 32]);
    for (level, expected) in [(0, RED), (1, BLUE)] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_pixel_eq(
            sample_center(&h, &tex).to_pixel(),
            expected,
            "initialized mip",
        );
    }
    tex.lock_rect(0, D3DLOCK_READONLY)
        .write_u32(&[GREEN; 64 * 64]);
    tex.lock_rect(1, D3DLOCK_READONLY)
        .write_u32(&[GREEN; 32 * 32]);
    assert_eq!(tex.add_dirty_rect(), 0);
    let mut observed = Vec::new();
    for level in 0..2 {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        observed.push(sample_center(&h, &tex).to_pixel());
    }
    assert_eq!(observed, vec![GREEN, GREEN]);
}

/// Draw on either side of an unannounced edit while the first upload is queued.
fn managed_dirty_queued_edit(partial: bool, publish: bool) -> [u32; 2] {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    let mut left = bind_for_quadrant_draws(&h, &tex);
    for vertex in &mut left {
        vertex.x = vertex.x.midpoint(-1.0);
        vertex.u = 0.25;
        vertex.v = 0.5;
    }
    let mut right = fullscreen_quad();
    for vertex in &mut right {
        vertex.x = vertex.x.midpoint(1.0);
        vertex.u = 0.25;
        vertex.v = 0.5;
    }
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &left), 0);
        if partial {
            let locked = tex.lock_rect_partial(0, &[0, 0, 32, 64], D3DLOCK_NO_DIRTY_UPDATE);
            fill_locked_rect(&locked, 32, 64, GREEN);
        } else {
            tex.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
                .write_u32(&[GREEN; 64 * 64]);
        }
        if publish {
            assert_eq!(tex.add_dirty_rect(), 0);
        }
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &right), 0);
        assert_eq!(d.clear_texture(0), 0);
        // Queued uploads and draw snapshots must survive the last API owner.
        drop(tex);
    });
    [h.read_pixel(160, 240), h.read_pixel(480, 240)]
}

#[test]
fn managed_dirty_queued_full_no_dirty_visibility() {
    assert_eq!(managed_dirty_queued_edit(false, false), [0xFFFF_0000; 2]);
}

#[test]
fn managed_dirty_queued_full_explicit_publication() {
    assert_eq!(
        managed_dirty_queued_edit(false, true),
        [0xFFFF_0000, 0xFF00_FF00]
    );
}

#[test]
fn managed_dirty_queued_partial_kept_exception() {
    let observed = managed_dirty_queued_edit(true, false);
    // Current partial-lock policy intentionally aliases the earlier upload's
    // backing. Old pixels are not a guaranteed contract for this overlap.
    assert!(matches!(observed[0], 0xFFFF_0000 | 0xFF00_FF00));
    assert_eq!(observed[1], 0xFF00_FF00);
}

/// A no-dirty write neither clears an older dirty region nor expands it.
#[test]
fn managed_dirty_preserves_pending_publication_regions() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;
    let h = Harness::new();
    for whole in [false, true] {
        let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
        tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
        assert_pixel_eq(sample_center(&h, &tex).to_pixel(), RED, "initialized");
        if whole {
            tex.lock_rect(0, 0).write_u32(&[GREEN; 64 * 64]);
        } else {
            fill_locked_rect(&tex.lock_rect_partial(0, &[0, 0, 32, 64], 0), 32, 64, GREEN);
        }
        fill_locked_rect(
            &tex.lock_rect_partial(0, &[32, 0, 64, 64], D3DLOCK_NO_DIRTY_UPDATE),
            32,
            64,
            BLUE,
        );
        assert_pixel_eq(
            sample_at(&h, &tex, 160, 240).to_pixel(),
            GREEN,
            "older dirty region",
        );
        assert_pixel_eq(
            sample_at(&h, &tex, 480, 240).to_pixel(),
            if whole { BLUE } else { RED },
            "no-dirty keeps prior coverage",
        );
    }
}

/// Initial dirtiness and surface-level publication use the same managed image.
#[test]
fn managed_dirty_initial_partial_lock_and_surface_publication() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    // Initialize the full CPU image before narrowing the first no-dirty lock.
    tex.lock_rect(0, D3DLOCK_READONLY)
        .write_u32(&[RED; 64 * 64]);
    fill_locked_rect(
        &tex.lock_rect_partial(0, &[0, 0, 32, 64], D3DLOCK_NO_DIRTY_UPDATE),
        32,
        64,
        GREEN,
    );
    assert_pixel_eq(
        sample_at(&h, &tex, 160, 240).to_pixel(),
        GREEN,
        "initial edited pixels",
    );
    assert_pixel_eq(
        sample_at(&h, &tex, 480, 240).to_pixel(),
        RED,
        "initial unedited pixels",
    );
    tex.surface_level(0)
        .lock_rect(D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32(&[RED; 64 * 64]);
    assert_pixel_eq(
        sample_at(&h, &tex, 160, 240).to_pixel(),
        GREEN,
        "surface no-dirty stays CPU-side",
    );
    assert_eq!(tex.add_dirty_rect(), 0);
    assert_pixel_eq(
        sample_at(&h, &tex, 160, 240).to_pixel(),
        RED,
        "surface edit explicitly published",
    );
    assert_eq!(h.reset(640, 480), 0);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        RED,
        "managed pixels survive reset",
    );
}

/// Odd mip rectangles round outward without publishing unrelated pixels.
#[test]
fn managed_dirty_partial_publication_scales_across_independent_mips() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(65, 33, 3, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for level in 0..3 {
        let (w, height) = (65 >> level, 33 >> level);
        fill_locked_rect(&tex.lock_rect(level, 0), w, height, RED);
    }
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        RED,
        "initial chain upload",
    );
    for level in 0..3 {
        fill_locked_rect(
            &tex.lock_rect(level, D3DLOCK_NO_DIRTY_UPDATE),
            65 >> level,
            33 >> level,
            GREEN,
        );
    }
    assert_eq!(tex.add_dirty_rect_partial(&[17, 9, 47, 25]), 0);
    for level in 0..3 {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_pixel_eq(
            sample_texel(&h, &tex, 0.5, 0.5),
            GREEN,
            "scaled published rectangle",
        );
        assert_pixel_eq(
            sample_texel(&h, &tex, 0.05, 0.05),
            RED,
            "outside scaled rectangle",
        );
    }
}

/// Managed explicit publication regenerates AUTOGEN's hidden GPU levels.
#[test]
fn managed_dirty_autogen_rebuilds_only_after_publication() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let tex = h.create_texture(
        64,
        64,
        0,
        D3DUSAGE_AUTOGENMIPMAP,
        D3DFMT_A8R8G8B8,
        D3DPOOL_MANAGED,
    );
    tex.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 4), 0);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        RED,
        "initial generated mip",
    );
    tex.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32(&[GREEN; 64 * 64]);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        RED,
        "hidden mip stays unchanged",
    );
    assert_eq!(tex.add_dirty_rect(), 0);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        GREEN,
        "published base regenerates hidden mip",
    );
}

/// Other pools retain their established no-dirty behavior.
#[test]
fn managed_dirty_change_preserves_other_pool_contracts() {
    const RED: u32 = 0xFFFF_0000;
    const GREEN: u32 = 0xFF00_FF00;
    let h = Harness::new();
    let dynamic = h.create_texture(
        64,
        64,
        1,
        D3DUSAGE_DYNAMIC,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    );
    dynamic.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    assert_pixel_eq(
        sample_center(&h, &dynamic).to_pixel(),
        RED,
        "dynamic initial",
    );
    dynamic
        .lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32(&[GREEN; 64 * 64]);
    assert_pixel_eq(
        sample_center(&h, &dynamic).to_pixel(),
        GREEN,
        "dynamic no-dirty still uploads",
    );
    let src = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write_u32(&[RED; 64 * 64]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        RED,
        "initial source copy",
    );
    src.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32(&[GREEN; 64 * 64]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        RED,
        "clean source is not recopied",
    );
    assert_eq!(src.add_dirty_rect(), 0);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        GREEN,
        "explicit source metadata still works",
    );
}

/// Premultiplied texture alpha applies to both channels and implicit samples.
#[test]
fn blend_texture_alpha_premultiplied() {
    use mtld3d_types::{
        D3DRS_LIGHTING, D3DRS_TEXTUREFACTOR, D3DTA_ALPHAREPLICATE, D3DTA_COMPLEMENT, D3DTA_CURRENT,
        D3DTA_DIFFUSE, D3DTA_TFACTOR, D3DTOP_BLENDTEXTUREALPHA, D3DTOP_BLENDTEXTUREALPHAPM,
        D3DTOP_DISABLE, D3DTOP_MODULATE, D3DTSS_ALPHAARG2, D3DTSS_COLORARG1, D3DTSS_COLORARG2,
        D3DTSS_COLOROP,
    };
    struct Case {
        name: &'static str,
        texel: Option<u32>,
        diffuse: u32,
        factor: u32,
        arg1: u32,
        arg2: u32,
        alpha: bool,
        post_modulate: bool,
        ordinary: bool,
        expected: u32,
    }
    let cases = [
        Case {
            name: "premultiplied",
            texel: Some(0x8040_2010),
            diffuse: 0xff80_8080,
            factor: 0x4020_4080,
            arg1: D3DTA_TEXTURE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x0050_4050,
        },
        Case {
            name: "ordinary control",
            texel: Some(0x8040_2010),
            diffuse: 0xff80_8080,
            factor: 0x4020_4080,
            arg1: D3DTA_TEXTURE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: true,
            expected: 0x0030_3048,
        },
        Case {
            name: "Wine texop row",
            texel: Some(0x9900_ff00),
            diffuse: 0x55ff_0000,
            factor: 0xdd33_3333,
            arg1: D3DTA_TEXTURE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x0014_ff14,
        },
        Case {
            name: "implicit alpha low",
            texel: Some(0x4020_1008),
            diffuse: 0x4040_8020,
            factor: 0x8080_2040,
            arg1: D3DTA_DIFFUSE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x00a0_9850,
        },
        Case {
            name: "implicit alpha high",
            texel: Some(0xc020_1008),
            diffuse: 0x4040_8020,
            factor: 0x8080_2040,
            arg1: D3DTA_DIFFUSE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x0060_8830,
        },
        Case {
            name: "saturate before next stage",
            texel: Some(0x40c0_a0e0),
            diffuse: 0xff80_8080,
            factor: 0xff80_8080,
            arg1: D3DTA_TEXTURE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: true,
            ordinary: false,
            expected: 0x0080_8080,
        },
        Case {
            name: "color argument modifiers",
            texel: Some(0x8040_2010),
            diffuse: 0x4040_8020,
            factor: 0x8020_4080,
            arg1: D3DTA_TEXTURE | D3DTA_ALPHAREPLICATE | D3DTA_COMPLEMENT,
            arg2: D3DTA_DIFFUSE | D3DTA_ALPHAREPLICATE,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x009f_9f9f,
        },
        Case {
            name: "alpha operation",
            texel: Some(0x8040_2010),
            diffuse: 0x4040_8020,
            factor: 0x8020_4080,
            arg1: D3DTA_DIFFUSE,
            arg2: D3DTA_TFACTOR,
            alpha: true,
            post_modulate: false,
            ordinary: false,
            expected: 0x0080_8080,
        },
        Case {
            name: "alpha argument modifier",
            texel: Some(0x8040_2010),
            diffuse: 0x4040_8020,
            factor: 0x8020_4080,
            arg1: D3DTA_DIFFUSE | D3DTA_COMPLEMENT,
            arg2: D3DTA_TFACTOR,
            alpha: true,
            post_modulate: false,
            ordinary: false,
            expected: 0x00ff_ffff,
        },
        Case {
            name: "missing implicit texture reference choice",
            texel: None,
            diffuse: 0x4040_8020,
            factor: 0x8080_2040,
            arg1: D3DTA_DIFFUSE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x00c0_a060,
        },
        Case {
            name: "missing explicit texture existing policy",
            texel: None,
            diffuse: 0x4040_8020,
            factor: 0x8080_2040,
            arg1: D3DTA_TEXTURE,
            arg2: D3DTA_TFACTOR,
            alpha: false,
            post_modulate: false,
            ordinary: false,
            expected: 0x0040_8020,
        },
    ];
    let h = Harness::new();
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    point_clamp(&h);
    let mut failures = Vec::new();
    for case in cases {
        let tex = case.texel.map(|value| {
            let tex = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
            tex.lock_rect(0, 0).write_u32(&[value]);
            tex
        });
        if let Some(tex) = &tex {
            assert_eq!(h.set_texture(0, tex), 0);
        } else {
            assert_eq!(h.clear_texture(0), 0);
        }
        assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, case.factor), 0);
        let op = if case.ordinary {
            D3DTOP_BLENDTEXTUREALPHA
        } else {
            D3DTOP_BLENDTEXTUREALPHAPM
        };
        for (state, value) in [
            (
                D3DTSS_COLOROP,
                if case.alpha { D3DTOP_SELECTARG1 } else { op },
            ),
            (
                D3DTSS_COLORARG1,
                if case.alpha { D3DTA_DIFFUSE } else { case.arg1 },
            ),
            (D3DTSS_COLORARG2, case.arg2),
            (
                D3DTSS_ALPHAOP,
                if case.alpha { op } else { D3DTOP_SELECTARG1 },
            ),
            (
                D3DTSS_ALPHAARG1,
                if case.alpha { case.arg1 } else { D3DTA_DIFFUSE },
            ),
            (D3DTSS_ALPHAARG2, case.arg2),
        ] {
            assert_eq!(h.set_texture_stage_state(0, state, value), 0);
        }
        for (state, value) in [
            (
                D3DTSS_COLOROP,
                if case.alpha {
                    D3DTOP_SELECTARG1
                } else if case.post_modulate {
                    D3DTOP_MODULATE
                } else {
                    D3DTOP_DISABLE
                },
            ),
            (
                D3DTSS_COLORARG1,
                D3DTA_CURRENT | if case.alpha { D3DTA_ALPHAREPLICATE } else { 0 },
            ),
            (D3DTSS_COLORARG2, D3DTA_DIFFUSE),
            (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
            (D3DTSS_ALPHAARG1, D3DTA_CURRENT),
        ] {
            assert_eq!(h.set_texture_stage_state(1, state, value), 0);
        }
        assert_eq!(
            h.set_texture_stage_state(2, D3DTSS_COLOROP, D3DTOP_DISABLE),
            0
        );
        let mut quad = fullscreen_quad();
        for vertex in &mut quad {
            vertex.color = case.diffuse;
        }
        h.render_once(BLACK, |d| {
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        });
        let observed = h.read_pixel(320, 240) & 0x00ff_ffff;
        eprintln!(
            "PM_CASE {} observed={observed:06x} expected={:06x}",
            case.name, case.expected
        );
        let matches = [0, 8, 16]
            .into_iter()
            .all(|shift| ((observed >> shift) & 255).abs_diff((case.expected >> shift) & 255) <= 2);
        if !matches {
            failures.push((case.name, observed, case.expected));
        }
        assert_eq!(h.clear_texture(0), 0);
    }
    assert!(failures.is_empty(), "PM failures: {failures:x?}");
}

/// Asserts a sampled V16U16 pixel, leaving the swizzle-filled blue lane to physical GPUs.
///
/// Red and green are the stored signed lanes and are compared on every device,
/// and so is alpha, which a two-channel format samples as one with or without
/// a swizzle. Blue is one only through the view swizzle. The paravirtual
/// device samples a swizzle view through the base texture's lanes, so blue is
/// left out of the comparison there and required on every physical GPU.
#[track_caller]
fn assert_v16u16_pixel(h: &Harness, actual: u32, expected: u32, tol: u8, context: &str) {
    const BLUE: u32 = 0x0000_00ff;
    let mask = if h.device_is_paravirtual() {
        !BLUE
    } else {
        u32::MAX
    };
    assert_pixel_approx(actual & mask, expected & mask, tol, context);
}

/// Asserts an exact sampled pixel for a signed format in the shared contract test.
///
/// V16U16 takes its blue lane from the view swizzle and goes through
/// [`assert_v16u16_pixel`]. Q8W8V8U8 stores all four lanes, so every lane is
/// compared on every device.
#[track_caller]
fn assert_signed_texture_pixel(
    h: &Harness,
    format: u32,
    actual: u32,
    expected: u32,
    context: &str,
) {
    if format == D3DFMT_V16U16 {
        assert_v16u16_pixel(h, actual, expected, 0, context);
    } else {
        assert_pixel_eq(actual, expected, context);
    }
}

/// V16U16 preserves signed endpoints and fills missing blue with one.
#[test]
fn v16u16_signed_extrema_and_missing_channels() {
    // ps_1_1: sample t0, then map signed channels from [-1,1] into [0,1].
    const PS: &[u32] = &[
        0xffff_0101,
        0x0000_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0004,
        0x800f_0000,
        0xb0e4_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let shader = h.create_pixel_shader(PS);
    assert_eq!(h.set_pixel_shader(&shader), 0, "SetPixelShader");
    for (u, v, expected) in [
        (i16::MIN, i16::MAX, 0xff00_ffff),
        (-32767, 0, 0xff00_80ff),
        (0, i16::MIN, 0xff80_00ff),
        (i16::MAX, i16::MAX, 0xffff_ffff),
    ] {
        let tex = h.create_texture(1, 1, 1, 0, D3DFMT_V16U16, D3DPOOL_MANAGED);
        tex.lock_rect(0, 0).write::<i16>(&[u, v]);
        assert_v16u16_pixel(
            &h,
            sample_center(&h, &tex).to_pixel(),
            expected,
            1,
            "signed V16U16 sample with blue one",
        );
    }
}

/// Missing alpha samples as one even when both stored signed channels are zero.
#[test]
fn v16u16_missing_alpha_is_one() {
    const PS: &[u32] = &[
        0xffff_0101,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0001,
        0x800f_0000,
        0xb0ff_0000,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let ps = h.create_pixel_shader(PS);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_V16U16, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write::<i16>(&[0, 0]);
    assert_pixel_eq(
        sample_center(&h, &tex).to_pixel(),
        0xffff_ffff,
        "missing alpha",
    );
}

/// Native byte layout is preserved through explicit mips and partial updates.
#[test]
fn v16u16_native_bytes_mips_and_partial_updates() {
    let h = Harness::new();
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        let usage = if pool == D3DPOOL_DEFAULT {
            D3DUSAGE_DYNAMIC
        } else {
            0
        };
        let tex = h.create_texture(4, 4, 0, usage, D3DFMT_V16U16, pool);
        assert_eq!(tex.level_count(), 3);
        for level in 0..3 {
            let (hr, desc) = tex.level_desc(level);
            assert_eq!(hr, 0);
            assert_eq!(
                (desc.format, desc.pool, desc.usage),
                (D3DFMT_V16U16, pool, usage)
            );
            let mut lock = tex.lock_rect(level, 0);
            assert_eq!(lock.pitch(), i32::try_from(desc.width * 4).unwrap());
            let values = vec![0x8000_7fff; (desc.width * desc.height) as usize];
            lock.write_u32(&values);
        }
        {
            let lock = tex.lock_rect(2, D3DLOCK_READONLY);
            // SAFETY: the held 1x1 V16U16 lock contains four initialized bytes.
            let raw = unsafe { lock.bits_ptr().cast::<u32>().read_unaligned() };
            assert_eq!(raw, 0x8000_7fff, "native signed bytes pool={pool}");
        }
        let cube = h.create_cube_texture_owned(4, 0, usage, D3DFMT_V16U16, pool);
        assert_eq!(cube.level_count(), 3);
        cube.lock_rect(0, 2, 0).write_u32(&[0x8000_7fff]);
        let lock = cube.lock_rect(0, 2, D3DLOCK_READONLY);
        assert_eq!(lock.pitch(), 4);
        // SAFETY: the held 1x1 V16U16 cube lock contains four initialized bytes.
        let raw = unsafe { lock.bits_ptr().cast::<u32>().read_unaligned() };
        assert_eq!(raw, 0x8000_7fff, "native cube bytes pool={pool}");
        let (hr, volume) = h.try_create_volume_texture([2, 2, 2], 0, usage, D3DFMT_V16U16, pool);
        assert_eq!(hr, 0);
        let volume = volume.expect("volume native storage");
        assert_eq!(volume.level_count(), 2);
        assert_eq!(volume.level_desc(0).1.usage, usage);
        volume.write_u32(1, &[0x8000_7fff]);
    }
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_V16U16, D3DPOOL_SYSTEMMEM);
    src.lock_rect(0, 0).write_u32(&[0x0000_7fff; 16]);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_V16U16, D3DPOOL_DEFAULT);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    assert_v16u16_pixel(
        &h,
        sample_center(&h, &dst).to_pixel(),
        0xffff_00ff,
        0,
        "UpdateTexture signed pair",
    );
    src.lock_rect_partial(0, &[0, 0, 1, 1], 0)
        .write_u32(&[0x7fff_0000]);
    assert_eq!(
        h.update_surface_region_hr(
            &src.surface_level(0),
            &D3DRECT {
                x1: 0,
                y1: 0,
                x2: 1,
                y2: 1
            },
            &dst.surface_level(0),
            (2, 2)
        ),
        0
    );
    assert_v16u16_pixel(
        &h,
        sample_at(&h, &dst, 400, 300).to_pixel(),
        0xff00_ffff,
        0,
        "partial UpdateSurface",
    );
    assert_v16u16_pixel(
        &h,
        sample_at(&h, &dst, 80, 60).to_pixel(),
        0xffff_00ff,
        0,
        "untouched pixel",
    );
    let mip = h.create_texture(4, 4, 0, 0, D3DFMT_V16U16, D3DPOOL_MANAGED);
    for (level, value) in [(0, 0x0000_7fff), (1, 0x7fff_0000), (2, 0x7fff_7fff)] {
        let size = (4usize >> level).pow(2);
        mip.lock_rect(level, 0).write_u32(&vec![value; size]);
    }
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for (level, expected) in [(0, 0xffff_00ff), (1, 0xff00_ffff), (2, 0xffff_ffff)] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_v16u16_pixel(
            &h,
            sample_center(&h, &mip).to_pixel(),
            expected,
            0,
            "explicit mip",
        );
    }
}

/// Cube faces and volume slices retain their native format through `UpdateTexture`.
#[test]
fn v16u16_cube_and_volume_updates() {
    let h = Harness::new();
    for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
        let src = h.create_cube_texture_owned(4, 0, 0, D3DFMT_V16U16, pool);
        assert_eq!(src.level_count(), 3);
        for face in 0..6 {
            for level in 0..3 {
                let count = (4usize >> level).pow(2);
                src.lock_rect(face, level, 0).write_u32(&vec![
                    if face == 0 {
                        0x0000_7fff
                    } else {
                        0x7fff_0000
                    };
                    count
                ]);
            }
        }
        let dst = h.create_cube_texture_owned(4, 0, 0, D3DFMT_V16U16, D3DPOOL_DEFAULT);
        let sampled = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_cube_texture_hr(&src, &dst), 0);
            &dst
        } else {
            &src
        };
        assert_v16u16_pixel(
            &h,
            sample_cube_x(&h, sampled, 1.0),
            0xffff_00ff,
            0,
            "positive X cube",
        );
        assert_v16u16_pixel(
            &h,
            sample_cube_x(&h, sampled, -1.0),
            0xff00_ffff,
            0,
            "negative X cube",
        );
        let (hr, src) = h.try_create_volume_texture([2, 2, 2], 0, 0, D3DFMT_V16U16, pool);
        assert_eq!(hr, 0);
        let src = src.expect("volume");
        assert_eq!(src.level_count(), 2);
        src.write_u32(
            0,
            &[
                0x0000_7fff,
                0x0000_7fff,
                0x0000_7fff,
                0x0000_7fff,
                0x7fff_0000,
                0x7fff_0000,
                0x7fff_0000,
                0x7fff_0000,
            ],
        );
        src.write_u32(1, &[0x7fff_7fff]);
        let (hr, dst) =
            h.try_create_volume_texture([2, 2, 2], 0, 0, D3DFMT_V16U16, D3DPOOL_DEFAULT);
        assert_eq!(hr, 0);
        let dst = dst.expect("default volume");
        let sampled = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_volume_texture_hr(&src, &dst), 0);
            &dst
        } else {
            &src
        };
        assert_eq!(h.set_volume_texture(0, sampled), 0);
        h.select_texture_stage(0);
        point_clamp(&h);
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
            0
        );
        assert_v16u16_pixel(
            &h,
            sample_volume_depth(&h, 0.25),
            0xffff_00ff,
            0,
            "first slice",
        );
        assert_v16u16_pixel(
            &h,
            sample_volume_depth(&h, 0.75),
            0xff00_ffff,
            0,
            "second slice",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
        assert_v16u16_pixel(
            &h,
            sample_volume_depth(&h, 0.5),
            0xffff_ffff,
            0,
            "volume mip",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        assert_eq!(h.clear_texture(0), 0);
    }
}

/// Query capabilities match creation, including a one-level AUTOGEN fallback.
#[test]
fn v16u16_queries_and_noautogen_contract() {
    let h = Harness::new();
    signed_texture_queries_and_noautogen(
        &h,
        D3DFMT_V16U16,
        0x0000_7fff_u32,
        0x7fff_0000,
        0xffff_00ff,
        0xff00_ffff,
    );
}

/// The query, creation and one-level AUTOGEN contract the signed native formats share.
fn signed_texture_queries_and_noautogen<T: Copy>(
    h: &Harness,
    format: u32,
    first: T,
    second: T,
    first_pixel: u32,
    second_pixel: u32,
) {
    use mtld3d_types::{
        D3DERR_NOTAVAILABLE, D3DOK_NOAUTOGEN, D3DRTYPE_CUBETEXTURE, D3DRTYPE_TEXTURE,
        D3DRTYPE_VOLUMETEXTURE, D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
        D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_VERTEXTEXTURE,
    };
    for rtype in [
        D3DRTYPE_TEXTURE,
        D3DRTYPE_CUBETEXTURE,
        D3DRTYPE_VOLUMETEXTURE,
        D3DRTYPE_VOLUME,
    ] {
        for usage in [
            0,
            D3DUSAGE_DYNAMIC,
            D3DUSAGE_QUERY_FILTER,
            D3DUSAGE_QUERY_VERTEXTEXTURE,
        ] {
            assert_eq!(
                h.check_device_format(D3DFMT_X8R8G8B8, usage, rtype, format),
                0,
                "rtype={rtype} usage={usage:#x}"
            );
        }
        for usage in [
            D3DUSAGE_RENDERTARGET,
            D3DUSAGE_DEPTHSTENCIL,
            D3DUSAGE_QUERY_SRGBREAD,
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
        ] {
            assert_eq!(
                h.check_device_format(D3DFMT_X8R8G8B8, usage, rtype, format),
                D3DERR_NOTAVAILABLE
            );
        }
        let expected = if matches!(rtype, D3DRTYPE_VOLUMETEXTURE | D3DRTYPE_VOLUME) {
            D3DERR_NOTAVAILABLE
        } else {
            D3DOK_NOAUTOGEN
        };
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, D3DUSAGE_AUTOGENMIPMAP, rtype, format),
            expected
        );
    }
    for rtype in [
        D3DRTYPE_TEXTURE,
        D3DRTYPE_CUBETEXTURE,
        D3DRTYPE_VOLUMETEXTURE,
        D3DRTYPE_VOLUME,
    ] {
        for extra in [
            0,
            D3DUSAGE_QUERY_FILTER,
            D3DUSAGE_AUTOGENMIPMAP,
            D3DUSAGE_RENDERTARGET,
            D3DUSAGE_QUERY_SRGBREAD,
        ] {
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    mtld3d_types::D3DUSAGE_QUERY_SRGBWRITE | extra,
                    rtype,
                    format
                ),
                D3DERR_NOTAVAILABLE
            );
        }
    }
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED] {
        for levels in [0, 1] {
            let tex = h.create_texture(4, 4, levels, D3DUSAGE_AUTOGENMIPMAP, format, pool);
            assert_eq!(tex.level_count(), 1);
            let (hr, desc) = tex.level_desc(0);
            assert_eq!(hr, 0);
            assert_eq!(desc.usage, D3DUSAGE_AUTOGENMIPMAP);
            assert_eq!(tex.level_desc(1).0, D3DERR_INVALIDCALL);
            assert_eq!(tex.auto_gen_filter_type(), D3DTEXF_LINEAR);
            assert_eq!(tex.set_auto_gen_filter_type(D3DTEXF_POINT), 0);
            assert_eq!(tex.auto_gen_filter_type(), D3DTEXF_POINT);
            assert_eq!(
                tex.set_auto_gen_filter_type(D3DTEXF_NONE),
                D3DERR_INVALIDCALL
            );
            assert_eq!(tex.auto_gen_filter_type(), D3DTEXF_POINT);
            tex.lock_rect(0, 0).write(&[first; 16]);
            tex.generate_mip_sub_levels();
            assert_signed_texture_pixel(
                h,
                format,
                sample_center(h, &tex).to_pixel(),
                first_pixel,
                "NOAUTOGEN top-level upload",
            );
            tex.lock_rect(0, 0).write(&[second; 16]);
            tex.generate_mip_sub_levels();
            assert_signed_texture_pixel(
                h,
                format,
                sample_center(h, &tex).to_pixel(),
                second_pixel,
                "NOAUTOGEN second publication",
            );
            let cube = h.create_cube_texture_owned(4, levels, D3DUSAGE_AUTOGENMIPMAP, format, pool);
            assert_eq!(cube.level_count(), 1);
            let (hr, desc) = cube.surface(0, 0).desc();
            assert_eq!(hr, 0);
            assert_eq!(desc.usage, D3DUSAGE_AUTOGENMIPMAP);
            let (hr, out) = cube.try_surface(0, 1);
            assert_eq!(hr, D3DERR_INVALIDCALL);
            assert!(out.is_null());
            assert_eq!(cube.auto_gen_filter_type(), D3DTEXF_LINEAR);
            assert_eq!(cube.set_auto_gen_filter_type(D3DTEXF_POINT), 0);
            assert_eq!(cube.auto_gen_filter_type(), D3DTEXF_POINT);
            assert_eq!(
                cube.set_auto_gen_filter_type(D3DTEXF_NONE),
                D3DERR_INVALIDCALL
            );
            assert_eq!(cube.auto_gen_filter_type(), D3DTEXF_POINT);
            if pool == D3DPOOL_DEFAULT {
                let source = h.create_cube_texture_owned(4, 1, 0, format, D3DPOOL_SYSTEMMEM);
                for face in 0..6 {
                    source.lock_rect(face, 0, 0).write(&[first; 16]);
                }
                assert_eq!(h.update_cube_texture_hr(&source, &cube), 0);
            } else {
                cube.lock_rect(0, 0, 0).write(&[first; 16]);
            }
            cube.generate_mip_sub_levels();
            assert_signed_texture_pixel(
                h,
                format,
                sample_cube_x(h, &cube, 1.0),
                first_pixel,
                "NOAUTOGEN cube publication",
            );
        }
    }
    for (pool, levels, usage) in [
        (D3DPOOL_DEFAULT, 2, D3DUSAGE_AUTOGENMIPMAP),
        (D3DPOOL_MANAGED, 2, D3DUSAGE_AUTOGENMIPMAP),
        (D3DPOOL_SYSTEMMEM, 1, D3DUSAGE_AUTOGENMIPMAP),
        (D3DPOOL_SCRATCH, 1, D3DUSAGE_AUTOGENMIPMAP),
        (D3DPOOL_DEFAULT, 1, D3DUSAGE_RENDERTARGET),
        (D3DPOOL_DEFAULT, 1, D3DUSAGE_DEPTHSTENCIL),
        (D3DPOOL_MANAGED, 1, D3DUSAGE_DYNAMIC),
        (
            D3DPOOL_MANAGED,
            1,
            D3DUSAGE_DYNAMIC | D3DUSAGE_AUTOGENMIPMAP,
        ),
    ] {
        let (hr, out) = h.try_create_texture(4, 4, levels, usage, format, pool);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "texture pool={pool} levels={levels} usage={usage:#x}"
        );
        assert!(out.is_null());
        let (hr, out) = h.try_create_cube_texture(4, levels, usage, format, pool);
        assert_eq!(
            hr, D3DERR_INVALIDCALL,
            "cube pool={pool} levels={levels} usage={usage:#x}"
        );
        assert!(out.is_null());
    }
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        assert_eq!(
            h.create_volume_texture([2, 2, 2], 1, D3DUSAGE_AUTOGENMIPMAP, format, pool),
            D3DERR_INVALIDCALL
        );
    }
}

/// Signed Q remains negative through SM1 sampling and is observable in RGB.
#[test]
fn q8w8v8u8_signed_alpha_and_channels() {
    // ps_1_1: tex t0; mad r0, t0.aaaa, c0, c0, with c0=.5.
    const PS: &[u32] = &[
        0xffff_0101,
        0x0000_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0004,
        0x800f_0000,
        0xb0ff_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let shader = h.create_pixel_shader(PS);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    for (bits, expected) in [
        (0x8000_7f81, 0x0000_0000),
        (0x817f_0080, 0x0000_0000),
        (0x0080_817f, 0x8080_8080),
        (0x7f81_8000, 0xffff_ffff),
    ] {
        let tex = h.create_texture(1, 1, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED);
        tex.lock_rect(0, 0).write_u32(&[bits]);
        assert_pixel_approx(sample_center(&h, &tex).to_pixel(), expected, 1, "signed Q");
    }
}

#[test]
fn q8w8v8u8_queries_and_noautogen_contract() {
    use mtld3d_types::{
        D3DERR_NOTAVAILABLE, D3DRTYPE_CUBETEXTURE, D3DRTYPE_TEXTURE, D3DRTYPE_VOLUMETEXTURE,
        D3DUSAGE_QUERY_LEGACYBUMPMAP, D3DUSAGE_QUERY_WRAPANDMIP,
    };
    let h = Harness::new();
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_SURFACE, D3DFMT_Q8W8V8U8),
        0
    );
    for rtype in [
        D3DRTYPE_TEXTURE,
        D3DRTYPE_CUBETEXTURE,
        D3DRTYPE_VOLUMETEXTURE,
        D3DRTYPE_VOLUME,
    ] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_WRAPANDMIP,
                rtype,
                D3DFMT_Q8W8V8U8
            ),
            0
        );
        if rtype != D3DRTYPE_TEXTURE {
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    D3DUSAGE_QUERY_LEGACYBUMPMAP,
                    rtype,
                    D3DFMT_Q8W8V8U8
                ),
                D3DERR_NOTAVAILABLE
            );
        }
    }
    signed_texture_queries_and_noautogen(
        &h,
        D3DFMT_Q8W8V8U8,
        0x7f00_007f_u32,
        0x7f00_7f00,
        0xffff_0000,
        0xff00_ff00,
    );
}

/// Read all signed channels without clamping before the range conversion.
fn signed_rgba_shader() -> Vec<u32> {
    vec![
        0xffff_0101,
        0x0000_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0004,
        0x800f_0000,
        0xb0e4_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_ffff,
    ]
}

#[test]
fn q8w8v8u8_sm1_negative_alpha_floor() {
    // RG maps Q; B is one only for Q < -1.
    let ps = [
        0xffff_0101,
        0x0000_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0000_0051,
        0xa00f_0001,
        0x3f80_0000,
        0x3f80_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0051,
        0xa00f_0002,
        0x0000_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0000,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0004,
        0x8007_0000,
        0xb0ff_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_0003,
        0x8008_0000,
        0xb1ff_0000,
        0xa0e4_0000,
        0x0000_0050,
        0x8008_0000,
        0x80ff_0000,
        0xa0ff_0001,
        0xa0ff_0002,
        0x0000_0005,
        0x8007_0001,
        0xa0e4_0001,
        0x80e4_0000,
        0x0000_0004,
        0x8007_0000,
        0x80ff_0000,
        0xa0e4_0002,
        0x80e4_0001,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let shader = h.create_pixel_shader(&ps);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    for (q, expected) in [
        (0x80, 0x0000_0000),
        (0x81, 0x0000_0000),
        (0, 0x0080_8000),
        (0x7f, 0x00ff_ff00),
        (0xc0, 0x003f_3f00),
    ] {
        let tex = h.create_texture(1, 1, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED);
        tex.lock_rect(0, 0).write_u32(&[q << 24 | 0x007f_0080]);
        let actual = sample_center(&h, &tex).to_pixel();
        assert_eq!(actual & 255, 0, "Q must not sample below -1");
        assert_pixel_approx(actual, expected, 1, "signed alpha floor");
    }
}

#[test]
fn q8w8v8u8_fixed_function_signed_color_and_alpha() {
    use mtld3d_types::{
        D3DRS_TEXTUREFACTOR, D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTA_TFACTOR, D3DTOP_ADD,
        D3DTSS_ALPHAARG2, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    };
    let h = Harness::new();
    let tex = h.create_texture(1, 1, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[0xf030_10e0]);
    assert_eq!(h.set_texture(0, &tex), 0);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x8080_8080), 0);
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_ADD),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_COLORARG2, D3DTA_TFACTOR),
        (D3DTSS_ALPHAOP, D3DTOP_ADD),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAARG2, D3DTA_TFACTOR),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0);
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    point_clamp(&h);
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(h.read_pixel(320, 240), 0x6040_a0e0, 1, "FF signed RGB");
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLORARG1, D3DTA_CURRENT | D3DTA_ALPHAREPLICATE),
        0
    );
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(h.read_pixel(320, 240), 0x6060_6060, 1, "FF signed Q");
}

#[test]
fn q8w8v8u8_native_bytes_mips_updates_and_plain_surfaces() {
    let h = Harness::new();
    let shader = h.create_pixel_shader(&signed_rgba_shader());
    assert_eq!(h.set_pixel_shader(&shader), 0);
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        let usage = if pool == D3DPOOL_DEFAULT {
            D3DUSAGE_DYNAMIC
        } else {
            0
        };
        let tex = h.create_texture(3, 3, 0, usage, D3DFMT_Q8W8V8U8, pool);
        assert_eq!(tex.level_count(), 2);
        for level in 0..2 {
            let desc = tex.level_desc(level).1;
            let values = vec![0x817f_0080; (desc.width * desc.height) as usize];
            let mut lock = tex.lock_rect(
                level,
                if pool == D3DPOOL_DEFAULT {
                    D3DLOCK_DISCARD
                } else {
                    0
                },
            );
            assert_eq!(lock.pitch(), i32::try_from(desc.width * 4).unwrap());
            lock.write_u32(&values);
        }
        let lock = tex.lock_rect(1, D3DLOCK_READONLY);
        // SAFETY: this held 1x1 lock contains one initialized four-byte texel.
        let raw = unsafe { lock.bits_ptr().cast::<u32>().read_unaligned() };
        assert_eq!(raw, 0x817f_0080);
        drop(lock);
        if matches!(pool, D3DPOOL_DEFAULT | D3DPOOL_MANAGED) {
            assert_pixel_approx(
                sample_center(&h, &tex).to_pixel(),
                0x0000_80ff,
                1,
                "native signed bytes",
            );
        }
    }
    let src = h.create_texture(4, 4, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(4, 4, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write_u32(&[0x817f_0080; 16]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    src.lock_rect_partial(0, &[0, 0, 1, 1], 0)
        .write_u32(&[0x8000_817f]);
    assert_eq!(
        h.update_surface_region_hr(
            &src.surface_level(0),
            &D3DRECT {
                x1: 0,
                y1: 0,
                x2: 1,
                y2: 1
            },
            &dst.surface_level(0),
            (2, 2)
        ),
        0
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 400, 300).to_pixel(),
        0x00ff_0080,
        1,
        "partial signed update",
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 80, 60).to_pixel(),
        0x0000_80ff,
        1,
        "untouched signed texel",
    );
    let unsigned = h.create_texture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    unsigned.lock_rect(0, 0).write_u32(&[0xffff_ffff; 16]);
    assert_eq!(h.update_texture_hr(&unsigned, &dst), D3DERR_INVALIDCALL);
    assert_eq!(
        h.update_surface_region_hr(
            &unsigned.surface_level(0),
            &D3DRECT {
                x1: 0,
                y1: 0,
                x2: 1,
                y2: 1
            },
            &dst.surface_level(0),
            (2, 2)
        ),
        D3DERR_INVALIDCALL
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 400, 300).to_pixel(),
        0x00ff_0080,
        1,
        "rejected mixed conversion preserves destination",
    );
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        let surf = h.create_offscreen_plain_surface(3, 2, D3DFMT_Q8W8V8U8, pool);
        assert_eq!(surf.desc().1.format, D3DFMT_Q8W8V8U8);
        surf.lock_rect(0).write_u32(&[0x8081_007f; 6]);
        let lock = surf.lock_rect(D3DLOCK_READONLY);
        assert_eq!(lock.pitch(), 12);
        // SAFETY: the held surface lock contains six initialized texels.
        let raw = unsafe { lock.bits_ptr().cast::<u32>().read_unaligned() };
        assert_eq!(raw, 0x8081_007f);
        drop(lock);
        assert_eq!(surf.get_dc(core::ptr::null_mut()).0, D3DERR_INVALIDCALL);
    }
    assert_eq!(
        h.create_offscreen_plain_surface_hr(3, 2, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED),
        D3DERR_INVALIDCALL
    );
}

fn signed_q8_pixel(raw: u32) -> u32 {
    let converted = raw.to_le_bytes().map(|byte| {
        let signed = i32::from(i8::from_ne_bytes([byte])).max(-127);
        u32::try_from(((signed + 127) * 255 + 127) / 254).unwrap()
    });
    converted[3] << 24 | converted[0] << 16 | converted[1] << 8 | converted[2]
}

#[test]
fn q8w8v8u8_cube_faces_volume_slices_and_mips() {
    let h = Harness::new();
    let shader = h.create_pixel_shader(&signed_rgba_shader());
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let values = [
        0x807f_0080,
        0x817f_00c0,
        0xc07f_00e0,
        0x007f_0000,
        0x407f_0020,
        0x7f7f_007f,
    ];
    let directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
        let src = h.create_cube_texture_owned(2, 0, 0, D3DFMT_Q8W8V8U8, pool);
        for (face, value) in values.into_iter().enumerate() {
            src.lock_rect(u32::try_from(face).unwrap(), 0, 0)
                .write_u32(&[value; 4]);
            src.lock_rect(u32::try_from(face).unwrap(), 1, 0)
                .write_u32(&[0x007f_0081]);
        }
        let dst = h.create_cube_texture_owned(2, 0, 0, D3DFMT_Q8W8V8U8, D3DPOOL_DEFAULT);
        let sample = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_cube_texture_hr(&src, &dst), 0);
            &dst
        } else {
            &src
        };
        assert_eq!(h.set_cube_texture(0, sample), 0);
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
            0
        );
        point_clamp(&h);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        for level in [0, 1] {
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
            for (face, direction) in directions.into_iter().enumerate() {
                let quad = fullscreen_quad().map(|v| CubeVertex {
                    x: v.x,
                    y: v.y,
                    z: v.z,
                    color: v.color,
                    u: direction[0],
                    v: direction[1],
                    w: direction[2],
                });
                h.render_once(BLACK, |d| {
                    assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
                });
                let raw = if level == 0 {
                    values[face]
                } else {
                    0x007f_0081
                };
                assert_pixel_approx(
                    h.read_pixel(320, 240),
                    signed_q8_pixel(raw),
                    1,
                    "signed cube RGBA",
                );
            }
        }
        assert_eq!(h.clear_texture(0), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        let src = h
            .try_create_volume_texture([3, 2, 2], 0, 0, D3DFMT_Q8W8V8U8, pool)
            .1
            .expect("Q8 volume");
        let mut data = vec![0x807f_0081; 6];
        data.extend_from_slice(&[0xc000_7f80; 6]);
        src.write_u32(0, &data);
        src.write_u32(1, &[0x7f81_007f]);
        let dst = h
            .try_create_volume_texture([3, 2, 2], 0, 0, D3DFMT_Q8W8V8U8, D3DPOOL_DEFAULT)
            .1
            .expect("Q8 destination volume");
        let sample = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_volume_texture_hr(&src, &dst), 0);
            &dst
        } else {
            &src
        };
        assert_eq!(h.set_volume_texture(0, sample), 0);
        for (w, value) in [(0.25, 0x807f_0081), (0.75, 0xc000_7f80)] {
            assert_pixel_approx(
                sample_volume_depth(&h, w),
                signed_q8_pixel(value),
                1,
                "signed volume RGBA",
            );
        }
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
        assert_pixel_approx(
            sample_volume_depth(&h, 0.5),
            signed_q8_pixel(0x7f81_007f),
            1,
            "signed volume mip",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        assert_eq!(h.clear_texture(0), 0);
    }
}

#[test]
fn q8w8v8u8_filter_wrap_and_modern_shader() {
    // ps_2_0: dcl t0; dcl_2d s0; texld r0,t0,s0; mad oC0,r0,c0,c0.
    let ps = [
        0xffff_0200,
        0x0500_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0200_001f,
        0x8000_0000,
        0xb00f_0000,
        0x0200_001f,
        0x9000_0000,
        0xa00f_0800,
        0x0300_0042,
        0x800f_0000,
        0xb0e4_0000,
        0xa0e4_0800,
        0x0400_0004,
        0x800f_0800,
        0x80e4_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let shader = h.create_pixel_shader(&ps);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let tex = h.create_texture(2, 1, 1, 0, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[0x8080_8080, 0x7f7f_7f7f]);
    assert_eq!(h.set_texture(0, &tex), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    for (u, filter, address, expected) in [
        (0.5, D3DTEXF_LINEAR, D3DTADDRESS_CLAMP, 0x8080_8080),
        (1.25, D3DTEXF_POINT, mtld3d_types::D3DTADDRESS_WRAP, 0),
    ] {
        for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
            assert_eq!(h.set_sampler_state(0, state, filter), 0);
        }
        assert_eq!(h.set_sampler_state(0, D3DSAMP_ADDRESSU, address), 0);
        let mut quad = fullscreen_quad();
        for vertex in &mut quad {
            vertex.u = u;
            vertex.v = 0.5;
        }
        h.render_once(BLACK, |d| {
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        });
        assert_pixel_approx(
            h.read_pixel(320, 240),
            expected,
            1,
            "signed filtering/wrapping",
        );
    }
    let mip = h.create_texture(4, 4, 0, 0, D3DFMT_Q8W8V8U8, D3DPOOL_MANAGED);
    for (level, raw) in [(0, 0x8080_8080), (1, 0), (2, 0x7f7f_7f7f)] {
        mip.lock_rect(level, 0)
            .write_u32(&vec![raw; (4usize >> level).pow(2)]);
    }
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for (level, expected) in [(0, 0), (1, 0x8080_8080), (2, 0xffff_ffff)] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_pixel_approx(
            sample_center(&h, &mip).to_pixel(),
            expected,
            1,
            "modern explicit mip",
        );
    }
}

/// All four signed16 lanes survive native texture storage and sampling.
#[test]
fn q16w16v16u16_signed_endpoints_and_bytes() {
    let h = Harness::new();
    let shader = h.create_pixel_shader(&signed_rgba_shader());
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let tex = h.create_texture(
        1,
        1,
        1,
        0,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        D3DPOOL_MANAGED,
    );
    let lanes = [i16::MIN, -32767, i16::MAX, 0];
    tex.lock_rect(0, 0).write(&lanes);
    assert_eq!(
        tex.lock_rect(0, D3DLOCK_READONLY).as_u16(4),
        &[0x8000, 0x8001, 0x7fff, 0]
    );
    assert_pixel_approx(
        sample_center(&h, &tex).to_pixel(),
        0x8000_00ff,
        1,
        "signed16 lanes",
    );
}

/// Adjacent signed16 samples remain distinct through float32 shader output.
#[test]
fn q16w16v16u16_float_precision_all_lanes() {
    use mtld3d_types::{D3DFMT_A32B32G32R32F, D3DFMT_Q16W16V16U16};
    let h = Harness::new();
    for modern in [false, true] {
        // PS2 reads t0, PS3 reads the TEXCOORD0 varying v0 from a matching VS3.
        let input = if modern { 0x90e4_0000 } else { 0xb0e4_0000 };
        let ps = [
            if modern { 0xffff_0300 } else { 0xffff_0200 },
            0x0200_001f,
            if modern { 0x8000_0005 } else { 0x8000_0000 },
            if modern { 0x900f_0000 } else { 0xb00f_0000 },
            0x0200_001f,
            0x9000_0000,
            0xa00f_0800,
            0x0300_0042,
            0x800f_0000,
            input,
            0xa0e4_0800,
            0x0200_0001,
            0x800f_0800,
            0x80e4_0000,
            0x0000_ffff,
        ];
        let shader = h.create_pixel_shader(&ps);
        assert_eq!(h.set_pixel_shader(&shader), 0);
        let vertex = h.create_vertex_shader(&[
            0xfffe_0300,
            0x0200_001f,
            0x8000_0000,
            0x900f_0000,
            0x0200_001f,
            0x8000_0005,
            0x900f_0001,
            0x0200_001f,
            0x8000_0000,
            0xe00f_0000,
            0x0200_001f,
            0x8000_0005,
            0xe00f_0001,
            0x0200_0001,
            0xe00f_0000,
            0x90e4_0000,
            0x0200_0001,
            0xe00f_0001,
            0x90e4_0001,
            0x0000_ffff,
        ]);
        if modern {
            assert_eq!(h.set_vertex_shader(&vertex), 0);
        }
        for state in [
            mtld3d_types::D3DRS_ALPHABLENDENABLE,
            mtld3d_types::D3DRS_FOGENABLE,
            mtld3d_types::D3DRS_SRGBWRITEENABLE,
        ] {
            assert_eq!(h.set_render_state(state, 0), 0);
        }
        assert_eq!(
            h.set_sampler_state(0, mtld3d_types::D3DSAMP_SRGBTEXTURE, 0),
            0
        );
        point_clamp(&h);
        let tex = h.create_texture(1, 1, 1, 0, D3DFMT_Q16W16V16U16, D3DPOOL_MANAGED);
        let backbuffer = h.render_target(0);
        let target = h.create_render_target(64, 48, D3DFMT_A32B32G32R32F);
        let readback =
            h.create_offscreen_plain_surface(64, 48, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
        let quad = fullscreen_quad().map(|mut v| {
            v.u = 0.5;
            v.v = 0.5;
            v
        });
        for values in [
            [i16::MIN, -32767, 0, i16::MAX],
            [i16::MAX, 0, -32767, i16::MIN],
            [16384, 16385, -16384, -16385],
            [16386, -16387, -16386, 16387],
            [1, -1, 32, -32],
            [64, -64, 1, -1],
        ] {
            tex.lock_rect(0, 0).write(&values);
            assert_eq!(h.set_texture(0, &tex), 0);
            assert_eq!(h.set_render_target(0, &target), 0);
            assert_eq!(h.begin_scene(), 0);
            assert_eq!(h.clear_target(0), 0);
            assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
            assert_eq!(h.end_scene(), 0);
            assert_eq!(h.set_render_target(0, &backbuffer), 0);
            assert_eq!(h.get_render_target_data_hr(&target, &readback), 0);
            let lock = readback.lock_rect(D3DLOCK_READONLY);
            let offset = 24 * usize::try_from(lock.pitch()).unwrap() / 4 + 32 * 4;
            let raw = lock.as_u32(offset + 4);
            for (lane, value) in values.into_iter().enumerate() {
                let actual = f32::from_bits(raw[offset + lane]);
                let expected = (f32::from(value) / 32767.0).max(-1.0);
                assert!(
                    (actual - expected).abs() < 0.25 / 32767.0,
                    "PS{} lane{lane} raw{value}: got{actual}, expected{expected}",
                    if modern { 3 } else { 2 }
                );
            }
        }
    }
}

/// Queries agree with creation, with the one-level AUTOGEN contract of the signed formats.
#[test]
fn q16w16v16u16_queries_and_noautogen_contract() {
    use mtld3d_types::{
        D3DERR_NOTAVAILABLE, D3DRTYPE_CUBETEXTURE, D3DRTYPE_TEXTURE, D3DRTYPE_VOLUMETEXTURE,
        D3DUSAGE_QUERY_LEGACYBUMPMAP, D3DUSAGE_QUERY_WRAPANDMIP,
    };
    let h = Harness::new();
    assert_eq!(
        h.check_device_format(
            D3DFMT_X8R8G8B8,
            0,
            D3DRTYPE_SURFACE,
            mtld3d_types::D3DFMT_Q16W16V16U16
        ),
        0
    );
    for rtype in [
        D3DRTYPE_TEXTURE,
        D3DRTYPE_CUBETEXTURE,
        D3DRTYPE_VOLUMETEXTURE,
        D3DRTYPE_VOLUME,
    ] {
        assert_eq!(
            h.check_device_format(
                D3DFMT_X8R8G8B8,
                D3DUSAGE_QUERY_WRAPANDMIP,
                rtype,
                mtld3d_types::D3DFMT_Q16W16V16U16
            ),
            0
        );
        if rtype != D3DRTYPE_TEXTURE {
            assert_eq!(
                h.check_device_format(
                    D3DFMT_X8R8G8B8,
                    D3DUSAGE_QUERY_LEGACYBUMPMAP,
                    rtype,
                    mtld3d_types::D3DFMT_Q16W16V16U16
                ),
                D3DERR_NOTAVAILABLE
            );
        }
    }
    signed_texture_queries_and_noautogen(
        &h,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        [32767_i16, 0, 0, 32767],
        [0_i16, 32767, 0, 32767],
        0xffff_0000,
        0xff00_ff00,
    );
}

/// The A8R8G8B8 pixel the half-scale, half-bias remap shader writes for four signed lanes.
fn signed_q16_pixel(lanes: [i16; 4]) -> u32 {
    let rgba = lanes.map(|lane| {
        let signed = i32::from(lane).max(-32767);
        u32::try_from(((signed + 32767) * 255 + 32767) / 65534).unwrap()
    });
    rgba[3] << 24 | rgba[0] << 16 | rgba[1] << 8 | rgba[2]
}

/// Locks expose eight bytes per texel, and updates copy them unchanged in every pool.
#[test]
fn q16w16v16u16_native_bytes_updates_and_plain_surfaces() {
    use mtld3d_types::D3DFMT_Q16W16V16U16 as FORMAT;
    let h = Harness::new();
    let shader = h.create_pixel_shader(&signed_rgba_shader());
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let first = [i16::MIN, -1, 16385, -32767];
    let second = [32767_i16, 16384, -16385, 8193];
    for pool in [
        D3DPOOL_DEFAULT,
        D3DPOOL_MANAGED,
        D3DPOOL_SYSTEMMEM,
        D3DPOOL_SCRATCH,
    ] {
        let dynamic = pool == D3DPOOL_DEFAULT;
        let tex = h.create_texture(
            3,
            3,
            0,
            if dynamic { D3DUSAGE_DYNAMIC } else { 0 },
            FORMAT,
            pool,
        );
        assert_eq!(tex.level_count(), 2);
        for level in [0, 1] {
            let desc = tex.level_desc(level).1;
            let mut lock = tex.lock_rect(level, if dynamic { D3DLOCK_DISCARD } else { 0 });
            assert_eq!(lock.pitch(), i32::try_from(desc.width * 8).unwrap());
            lock.write(&vec![first; (desc.width * desc.height) as usize]);
        }
        for level in [0, 1] {
            let desc = tex.level_desc(level).1;
            let lock = tex.lock_rect(level, D3DLOCK_READONLY);
            let bytes = lock.as_u16((desc.width * desc.height * 4) as usize);
            for lanes in bytes.as_chunks::<4>().0 {
                assert_eq!(*lanes, first.map(|v| u16::from_ne_bytes(v.to_ne_bytes())));
            }
        }
        if matches!(pool, D3DPOOL_DEFAULT | D3DPOOL_MANAGED) {
            assert_pixel_approx(
                sample_center(&h, &tex).to_pixel(),
                signed_q16_pixel(first),
                1,
                "signed16 upload",
            );
        }
    }
    let src = h.create_texture(3, 3, 1, 0, FORMAT, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(3, 3, 1, 0, FORMAT, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write(&[first; 9]);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    src.lock_rect_partial(0, &[1, 1, 2, 2], 0).write(&[second]);
    let region = D3DRECT {
        x1: 1,
        y1: 1,
        x2: 2,
        y2: 2,
    };
    assert_eq!(
        h.update_surface_region_hr(
            &src.surface_level(0),
            &region,
            &dst.surface_level(0),
            (2, 2)
        ),
        0
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 550, 400).to_pixel(),
        signed_q16_pixel(second),
        1,
        "offset eight-byte update",
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 80, 60).to_pixel(),
        signed_q16_pixel(first),
        1,
        "untouched eight-byte texel",
    );
    let wrong = h.create_texture(3, 3, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    wrong.lock_rect(0, 0).write_u32(&[0xffff_ffff; 9]);
    assert_eq!(h.update_texture_hr(&wrong, &dst), D3DERR_INVALIDCALL);
    assert_eq!(
        h.update_surface_region_hr(
            &wrong.surface_level(0),
            &region,
            &dst.surface_level(0),
            (2, 2)
        ),
        D3DERR_INVALIDCALL
    );
    assert_pixel_approx(
        sample_at(&h, &dst, 550, 400).to_pixel(),
        signed_q16_pixel(second),
        1,
        "invalid update preserves destination",
    );
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        let surface = h.create_offscreen_plain_surface(3, 2, FORMAT, pool);
        assert_eq!(surface.desc().1.format, FORMAT);
        surface.lock_rect(0).write(&[second; 6]);
        let lock = surface.lock_rect(D3DLOCK_READONLY);
        assert_eq!(lock.pitch(), 24);
        for raw in lock.as_u16(24).as_chunks::<4>().0 {
            assert_eq!(*raw, second.map(|v| u16::from_ne_bytes(v.to_ne_bytes())));
        }
        drop(lock);
        assert_eq!(surface.get_dc(core::ptr::null_mut()).0, D3DERR_INVALIDCALL);
    }
    assert_eq!(
        h.create_offscreen_plain_surface_hr(3, 2, FORMAT, D3DPOOL_MANAGED),
        D3DERR_INVALIDCALL
    );
    let (hr, out) = h.try_create_texture(0, 2, 1, 0, FORMAT, D3DPOOL_MANAGED);
    assert_eq!(hr, D3DERR_INVALIDCALL);
    assert!(out.is_null());
}

/// Cube faces, volume slices, their mips and a partial box keep eight-byte texels apart.
#[test]
fn q16w16v16u16_cube_volume_mips_and_partial_box() {
    use mtld3d_types::{D3DBOX, D3DFMT_Q16W16V16U16 as FORMAT};
    let h = Harness::new();
    let shader = h.create_pixel_shader(&signed_rgba_shader());
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let first = [i16::MIN, 32767, -16384, -32767];
    let second = [32767_i16, -16385, 16384, 8193];
    let directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
        let cube = h.create_cube_texture_owned(2, 0, 0, FORMAT, pool);
        for face in 0..6 {
            let value = [
                i16::try_from(face * 8191).unwrap_or(32767),
                -16385,
                16384,
                if face % 2 == 0 { i16::MIN } else { -8193 },
            ];
            cube.lock_rect(face, 0, 0).write(&[value; 4]);
            cube.lock_rect(face, 1, 0).write(&[second]);
            assert_eq!(
                cube.lock_rect(face, 0, D3DLOCK_READONLY).as_u16(16),
                [value; 4]
                    .as_flattened()
                    .iter()
                    .map(|v| v.cast_unsigned())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                cube.lock_rect(face, 1, D3DLOCK_READONLY).as_u16(4),
                second.map(i16::cast_unsigned)
            );
        }
        let dst = h.create_cube_texture_owned(2, 0, 0, FORMAT, D3DPOOL_DEFAULT);
        let sampled = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_cube_texture_hr(&cube, &dst), 0);
            &dst
        } else {
            &cube
        };
        assert_eq!(h.set_cube_texture(0, sampled), 0);
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
            0
        );
        point_clamp(&h);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
        for level in [0, 1] {
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
            for (face, direction) in directions.into_iter().enumerate() {
                let quad = fullscreen_quad().map(|v| CubeVertex {
                    x: v.x,
                    y: v.y,
                    z: v.z,
                    color: v.color,
                    u: direction[0],
                    v: direction[1],
                    w: direction[2],
                });
                h.render_once(0, |d| {
                    assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
                });
                let value = if level == 0 {
                    [
                        i16::try_from(face * 8191).unwrap_or(32767),
                        -16385,
                        16384,
                        if face % 2 == 0 { i16::MIN } else { -8193 },
                    ]
                } else {
                    second
                };
                assert_pixel_approx(
                    h.read_pixel(320, 240),
                    signed_q16_pixel(value),
                    1,
                    "signed16 cube RGBA",
                );
            }
        }
        assert_eq!(h.clear_texture(0), 0);
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        let volume = h
            .try_create_volume_texture([3, 2, 2], 0, 0, FORMAT, pool)
            .1
            .expect("signed16 volume");
        volume.write_i16x4(0, &[first; 12]);
        volume.write_i16x4(1, &[second]);
        volume.write_box_i16x4(
            0,
            &D3DBOX {
                left: 1,
                top: 0,
                right: 2,
                bottom: 2,
                front: 1,
                back: 2,
            },
            &[second; 2],
        );
        let mut expected = vec![first; 12];
        expected[7] = second;
        expected[10] = second;
        assert_eq!(volume.read_i16x4(0), (24, 48, expected));
        assert_eq!(volume.read_i16x4(1), (8, 8, vec![second]));
        let dst = h
            .try_create_volume_texture([3, 2, 2], 0, 0, FORMAT, D3DPOOL_DEFAULT)
            .1
            .expect("default signed16 volume");
        let sampled = if pool == D3DPOOL_SYSTEMMEM {
            assert_eq!(h.update_volume_texture_hr(&volume, &dst), 0);
            &dst
        } else {
            &volume
        };
        assert_eq!(h.set_volume_texture(0, sampled), 0);
        assert_pixel_approx(
            sample_volume_depth(&h, 0.25),
            signed_q16_pixel(first),
            1,
            "untouched z slice",
        );
        assert_pixel_approx(
            sample_volume_depth(&h, 0.75),
            signed_q16_pixel(second),
            1,
            "nonzero x/z box update",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
        assert_pixel_approx(
            sample_volume_depth(&h, 0.5),
            signed_q16_pixel(second),
            1,
            "eight-byte volume mip",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        assert_eq!(h.clear_texture(0), 0);
    }
    let scratch = h
        .try_create_volume_texture([3, 2, 2], 1, 0, FORMAT, D3DPOOL_SCRATCH)
        .1
        .expect("scratch volume");
    scratch.write_i16x4(0, &[first; 12]);
    assert_eq!(scratch.read_i16x4(0), (24, 48, vec![first; 12]));
    let dynamic = h
        .try_create_volume_texture(
            [3, 2, 2],
            1,
            mtld3d_types::D3DUSAGE_DYNAMIC,
            FORMAT,
            D3DPOOL_DEFAULT,
        )
        .1
        .expect("default dynamic signed16 volume");
    dynamic.write_i16x4(0, &[second; 12]);
    assert_eq!(dynamic.read_i16x4(0), (24, 48, vec![second; 12]));
    assert_eq!(h.set_volume_texture(0, &dynamic), 0);
    assert_pixel_approx(
        sample_volume_depth(&h, 0.5),
        signed_q16_pixel(second),
        1,
        "default dynamic volume upload",
    );
}

/// Both minimum Q encodings sample as minus one, never below it.
#[test]
fn q16w16v16u16_sm1_negative_alpha_floor() {
    // RG maps Q; B is one only for Q < -1.
    let mut ps = vec![
        0xffff_0101,
        0x0000_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0000_0051,
        0xa00f_0001,
        0x3f80_0000,
        0x3f80_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0051,
        0xa00f_0002,
        0x0000_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0000,
        0x0000_0042,
        0xb00f_0000,
        0x0000_0004,
        0x8007_0000,
        0xb0ff_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_0003,
        0x8008_0000,
        0xb1ff_0000,
        0xa0e4_0000,
        0x0000_0050,
        0x8008_0000,
        0x80ff_0000,
        0xa0ff_0001,
        0xa0ff_0002,
        0x0000_0005,
        0x8007_0001,
        0xa0e4_0001,
        0x80e4_0000,
        0x0000_0004,
        0x8007_0000,
        0x80ff_0000,
        0xa0e4_0002,
        0x80e4_0001,
        0x0000_ffff,
    ];
    let h = Harness::new();
    for modern in [false, true] {
        if modern {
            ps[0] = 0xffff_0104;
            let tex = ps
                .windows(2)
                .position(|v| v == [0x0000_0042, 0xb00f_0000])
                .unwrap();
            ps.splice(tex..tex + 2, [0x0000_0042, 0x800f_0000, 0xb0e4_0000]);
            for token in &mut ps[tex + 3..] {
                if *token == 0xb0ff_0000 {
                    *token = 0x80ff_0000;
                }
                if *token == 0xb1ff_0000 {
                    *token = 0x81ff_0000;
                }
            }
        }
        let shader = h.create_pixel_shader(&ps);
        assert_eq!(h.set_pixel_shader(&shader), 0);
        for (q, expected) in [
            (i16::MIN, 0x0000_0000),
            (-32767, 0x0000_0000),
            (0, 0x0080_8000),
            (32767, 0x00ff_ff00),
            (-16384, 0x0040_4000),
        ] {
            let tex = h.create_texture(
                1,
                1,
                1,
                0,
                mtld3d_types::D3DFMT_Q16W16V16U16,
                D3DPOOL_MANAGED,
            );
            tex.lock_rect(0, 0).write(&[[i16::MIN, 32767, 0, q]]);
            let actual = sample_center(&h, &tex).to_pixel();
            assert_eq!(actual & 255, 0, "Q must not sample below -1");
            assert_pixel_approx(actual, expected, 1, "signed alpha floor");
        }
    }
}

/// Fixed-function stages see signed colour lanes and a signed Q lane.
#[test]
fn q16w16v16u16_fixed_function_signed_color_and_alpha() {
    use mtld3d_types::{
        D3DRS_TEXTUREFACTOR, D3DTA_ALPHAREPLICATE, D3DTA_CURRENT, D3DTA_TFACTOR, D3DTOP_ADD,
        D3DTSS_ALPHAARG2, D3DTSS_COLORARG1, D3DTSS_COLORARG2, D3DTSS_COLOROP,
    };
    let h = Harness::new();
    let tex = h.create_texture(
        1,
        1,
        1,
        0,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        D3DPOOL_MANAGED,
    );
    tex.lock_rect(0, 0)
        .write(&[[-8192_i16, 4096, 12288, -4096]]);
    assert_eq!(h.set_texture(0, &tex), 0);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x8080_8080), 0);
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_ADD),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_COLORARG2, D3DTA_TFACTOR),
        (D3DTSS_ALPHAOP, D3DTOP_ADD),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAARG2, D3DTA_TFACTOR),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0);
    }
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    point_clamp(&h);
    let quad = fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(h.read_pixel(320, 240), 0x6040_a0e0, 1, "FF signed RGB");
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(1, D3DTSS_COLORARG1, D3DTA_CURRENT | D3DTA_ALPHAREPLICATE),
        0
    );
    h.render_once(BLACK, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    assert_pixel_approx(h.read_pixel(320, 240), 0x6060_6060, 1, "FF signed Q");
    for (q, expected) in [(i16::MIN, 0), (0, 0x8080_8080), (32767, 0xffff_ffff)] {
        tex.lock_rect(0, 0).write(&[[-8192_i16, 4096, 12288, q]]);
        h.render_once(BLACK, |d| {
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        });
        assert_pixel_approx(h.read_pixel(320, 240), expected, 1, "FF Q endpoint");
    }
}

/// Near-zero codes, linear filtering, wrapping and explicit mips through `ps_2_0`.
#[test]
fn q16w16v16u16_filter_wrap_and_modern_shader() {
    // ps_2_0: dcl t0; dcl_2d s0; texld r0,t0,s0; mad oC0,r0,c0,c0.
    let ps = [
        0xffff_0200,
        0x0500_0051,
        0xa00f_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x3f00_0000,
        0x0200_001f,
        0x8000_0000,
        0xb00f_0000,
        0x0200_001f,
        0x9000_0000,
        0xa00f_0800,
        0x0300_0042,
        0x800f_0000,
        0xb0e4_0000,
        0xa0e4_0800,
        0x0400_0004,
        0x800f_0800,
        0x80e4_0000,
        0xa0e4_0000,
        0xa0e4_0000,
        0x0000_ffff,
    ];
    let h = Harness::new();
    let mut amplified = ps.to_vec();
    amplified[3..7].fill(128.0_f32.to_bits());
    amplified.splice(
        7..7,
        [
            0x0500_0051,
            0xa00f_0001,
            0x3f00_0000,
            0x3f00_0000,
            0x3f00_0000,
            0x3f00_0000,
        ],
    );
    let addend = amplified.len() - 2;
    amplified[addend] = 0xa0e4_0001;
    let shader = h.create_pixel_shader(&amplified);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let small = h.create_texture(
        1,
        1,
        1,
        0,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        D3DPOOL_MANAGED,
    );
    for (values, expected) in [
        ([32_i16, -32, 64, -64], 0x40a0_60bf),
        ([-64_i16, 64, -32, 32], 0xa040_bf60),
    ] {
        small.lock_rect(0, 0).write(&[values]);
        assert_pixel_approx(
            sample_center(&h, &small).to_pixel(),
            expected,
            1,
            "SNORM16 near-zero samples amplified by 128",
        );
    }
    let shader = h.create_pixel_shader(&ps);
    assert_eq!(h.set_pixel_shader(&shader), 0);
    let tex = h.create_texture(
        2,
        1,
        1,
        0,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        D3DPOOL_MANAGED,
    );
    tex.lock_rect(0, 0).write(&[[i16::MIN; 4], [32767_i16; 4]]);
    assert_eq!(h.set_texture(0, &tex), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    for (u, filter, address, expected) in [
        (0.5, D3DTEXF_LINEAR, D3DTADDRESS_CLAMP, 0x8080_8080),
        (1.25, D3DTEXF_POINT, mtld3d_types::D3DTADDRESS_WRAP, 0),
    ] {
        for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
            assert_eq!(h.set_sampler_state(0, state, filter), 0);
        }
        assert_eq!(h.set_sampler_state(0, D3DSAMP_ADDRESSU, address), 0);
        let mut quad = fullscreen_quad();
        for vertex in &mut quad {
            vertex.u = u;
            vertex.v = 0.5;
        }
        h.render_once(BLACK, |d| {
            assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        });
        assert_pixel_approx(
            h.read_pixel(320, 240),
            expected,
            1,
            "signed filtering/wrapping",
        );
    }
    let mip = h.create_texture(
        4,
        4,
        0,
        0,
        mtld3d_types::D3DFMT_Q16W16V16U16,
        D3DPOOL_MANAGED,
    );
    for (level, raw) in [(0, [i16::MIN; 4]), (1, [0_i16; 4]), (2, [32767_i16; 4])] {
        mip.lock_rect(level, 0)
            .write(&vec![raw; (4usize >> level).pow(2)]);
    }
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for (level, expected) in [(0, 0), (1, 0x8080_8080), (2, 0xffff_ffff)] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        assert_pixel_approx(
            sample_center(&h, &mip).to_pixel(),
            expected,
            1,
            "modern explicit mip",
        );
    }
}
