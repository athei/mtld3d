//! Texture create → lock/write → bind → sample across formats and mip levels.
//!
//! Plus cube and volume texture contracts.

use mtld3d_tests::{
    Harness, LockedRect, Rgba8, Texture, TexturedVertex, VolumeVertex, assert_pixel_eq,
};
use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_ATI1,
    D3DFMT_DXT1, D3DFMT_L8, D3DFMT_R5G6B5, D3DFMT_UYVY, D3DFMT_V8U8, D3DFMT_X8R8G8B8, D3DFMT_YUY2,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_TEXTUREFORMAT3, D3DFVF_XYZ, D3DLOCK_DISCARD,
    D3DLOCK_NO_DIRTY_UPDATE, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT, D3DRTYPE_SURFACE, D3DRTYPE_VOLUME,
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER,
    D3DSAMP_MIPFILTER, D3DTADDRESS_CLAMP, D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR, D3DTEXF_NONE,
    D3DTEXF_POINT, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DYNAMIC, D3DUSAGE_RENDERTARGET,
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

#[test]
fn color_formats_sample_red() {
    let h = Harness::new();
    // 1×1 opaque-red texel encoded for each format (little-endian bytes).
    let cases: [(u32, &[u8]); 4] = [
        (D3DFMT_X8R8G8B8, &[0x00, 0x00, 0xFF, 0x00]), // BGRX
        (D3DFMT_R5G6B5, &[0x00, 0xF8]),               // R=31
        (D3DFMT_A1R5G5B5, &[0x00, 0xFC]),             // A=1 R=31
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

#[test]
fn luminance_format_samples_gray() {
    let h = Harness::new();
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

/// A default-pool texture the game cannot lock takes a second `UpdateTexture`.
///
/// Its staging goes away once the first upload has been submitted (the GPU
/// holds the only copy, as on real D3D9); the second update re-creates it,
/// and what samples back is the second fill.
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

/// Bind `tex`, sample it across the backbuffer, return the four quadrant centres.
///
/// Clockwise from the top left, the order [`QUADRANTS`] lists them in.
fn sample_quadrants(h: &Harness, tex: &Texture<'_>) -> [u32; 4] {
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
    [
        h.read_pixel(160, 120),
        h.read_pixel(480, 120),
        h.read_pixel(480, 360),
        h.read_pixel(160, 360),
    ]
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
/// its first whole-level upload has been submitted, so the partial update
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
    // The sampling draw submits the whole-level upload, which releases the
    // destination's staging.
    assert_pixel_eq(
        sample_center(&h, &dst).to_pixel(),
        GREEN,
        "whole-level fill",
    );

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
/// with it, so a second lock has nothing left in system memory. D3D9 promises
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
    // The draw is what uploads the level and releases its staging; which texel
    // the centre sample lands on is not what this pins.
    let _sampled = sample_center(&h, &tex);

    let locked = tex.lock_rect(0, 0);
    assert_eq!(
        locked.as_u32(TEXELS),
        written.as_slice(),
        "the lock reads the texels the level holds"
    );
}

/// `GetDC` on a released default-pool level maps the level's own texels.
///
/// The draw uploads the level and the staging goes with the upload, so the
/// pixels live on the GPU alone and the level's staging slot points at the one
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
    // The draw is what uploads the level and releases its staging.
    assert_pixel_eq(sample_center(&h, &tex).to_pixel(), GREEN, "upload");

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

/// Sample the volume bound on stage 0 at texcoord `(0.5, 0.5, w)` with point filtering.
fn sample_volume_depth(h: &Harness, w: f32) -> u32 {
    const WHITE: u32 = 0xFFFF_FFFF;
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
    h.read_pixel(320, 240)
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

/// `D3DFMT_V8U8` must sample its content, not black.
///
/// Signed two-channel, → `Rg8Snorm` with {R,G,1,1} swizzle. A 1x1 texel of
/// signed (+1,+1) reads as (1,1,1,1) → white. Confirms `V8U8`
/// create/upload/sample works in isolation (a full FF-alpha +
/// per-texel-bias `V8U8` setup is not covered here).
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

#[test]
fn v8u8_signed_texture_samples_nonzero() {
    let h = Harness::new();
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
    // any pitch below that. A texture level's staging is allocated at the tight
    // `width * bpp` stride its GPU upload steps by, which for this level is two
    // bytes short, so `GetDC` has to hand GDI a DIB of its own at the rounded
    // stride and copy back what GDI drew.
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
        assert_eq!(
            locked.pitch(),
            (W * 2).cast_signed(),
            "the level locks at its own tight row stride"
        );
        locked.write(&[GREEN_565; (W * H) as usize]);
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
