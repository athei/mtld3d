//! Texture create → lock/write → bind → sample across formats and mip levels.
//!
//! Plus the cube/volume stub contracts.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex};
use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_DXT1, D3DFMT_L8,
    D3DFMT_R5G6B5, D3DFMT_V8U8, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DRTYPE_SURFACE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_ANISOTROPIC, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT,
    D3DUSAGE_AUTOGENMIPMAP,
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
    Rgba8::from_pixel(h.read_pixel(320, 240))
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
fn cube_texture_cpu_pools_create_default_rejects() {
    let h = Harness::new();
    // A sampleable (GPU) cube needs D3DPTEXTURECAPS_CUBEMAP, which is off, so a
    // DEFAULT-pool cube — which would require an MTLTextureTypeCube — is rejected.
    assert_eq!(
        h.create_cube_texture(64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT),
        D3DERR_INVALIDCALL,
        "DEFAULT-pool cube texture is rejected without GPU cube-map support",
    );
    // The CPU pools get a creatable, lockable shell (six faces share one CPU
    // store; never sampled). Each must succeed and release cleanly.
    for (pool, name) in [
        (D3DPOOL_SCRATCH, "SCRATCH"),
        (D3DPOOL_MANAGED, "MANAGED"),
        (D3DPOOL_SYSTEMMEM, "SYSTEMMEM"),
    ] {
        assert_eq!(
            h.create_cube_texture(64, 1, 0, D3DFMT_A8R8G8B8, pool),
            0,
            "{name}-pool cube texture is a creatable CPU shell",
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

/// `D3DFMT_V8U8` must sample its content, not black.
///
/// Signed two-channel, → `Rg8Snorm` with {R,G,1,1} swizzle. A 1x1 texel of
/// signed (+1,+1) reads as (1,1,1,1) → white. Confirms `V8U8`
/// create/upload/sample works in isolation (a full FF-alpha +
/// per-texel-bias `V8U8` setup is not covered here).
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
