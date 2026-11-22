//! `ATI2` (3Dc) textures: BC5 storage, a byte-per-texel lock and the D3D9 channel order.
//!
//! The format is a 2D and cube texture format in every pool, a SCRATCH-only
//! volume and offscreen plain, and never a render target. Its lock reports a
//! byte per texel over the compressed blocks: the pitch is the level's width
//! rounded up to the 4-texel block, and a sub-rect starts `y * pitch + x`
//! bytes in. Each block samples green from its first eight bytes and red from
//! its second, with blue and alpha one.

use mtld3d_tests::{CubeTexture, Harness, Texture, TexturedVertex, assert_pixel_approx};
use mtld3d_types::{
    D3D_OK, D3DBOX, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8, D3DFMT_ATI2,
    D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_READONLY,
    D3DMULTISAMPLE_NONE, D3DOK_NOAUTOGEN, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE,
    D3DRTYPE_TEXTURE, D3DRTYPE_VOLUMETEXTURE, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER, D3DTEXF_POINT,
    D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC, D3DUSAGE_QUERY_FILTER,
    D3DUSAGE_QUERY_LEGACYBUMPMAP, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD,
    D3DUSAGE_QUERY_SRGBWRITE, D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_QUERY_WRAPANDMIP,
    D3DUSAGE_RENDERTARGET,
};

/// The display format every query here is asked against.
const DISPLAY: u32 = D3DFMT_X8R8G8B8;

/// Half of a block whose sixteen texels hold 50 %: two equal endpoints, every index zero.
const HALF: [u8; 8] = [0x7f, 0x7f, 0, 0, 0, 0, 0, 0];

/// Half of a block whose sixteen texels hold zero.
const ZERO: [u8; 8] = [0; 8];

/// One block from its first and its second eight bytes.
fn block(first: [u8; 8], second: [u8; 8]) -> Vec<u8> {
    let mut bytes = first.to_vec();
    bytes.extend_from_slice(&second);
    bytes
}

/// What D3D9 samples from a block whose first half is 50 % and second half zero.
const GREEN_HALF: u32 = 0xff00_7fff;

/// What D3D9 samples from a block whose first half is zero and second half 50 %.
const RED_HALF: u32 = 0xff7f_00ff;

/// What D3D9 samples from a block of zeros: only blue and alpha, which are one.
const NEITHER: u32 = 0xff00_00ff;

/// Point-sampled fixed-function stage 0 that passes the texel through.
fn setup(h: &Harness) {
    h.select_texture_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
        assert_eq!(h.set_sampler_state(0, state, D3DTEXF_POINT), D3D_OK);
    }
}

/// Draw the bound texture over the whole target at the constant coordinate `(u, 0.5)`.
fn sample(h: &Harness, u: f32) -> u32 {
    let v = |x, y| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xffff_ffff,
        u,
        v: 0.5,
    };
    let quad = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];
    h.render_once(0, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), D3D_OK);
    });
    h.read_pixel(320, 240)
}

/// The capability answers of `ATI2` follow its BC5 storage.
///
/// A 2D or cube texture answers yes to plain sampling, filtering, dynamic
/// usage, wrap-and-mip and vertex texture fetch, and `D3DOK_NOAUTOGEN` to mip
/// generation, since it is no render target. Render-target, blending, depth,
/// sRGB and bump-map questions answer no, as do the volume and the plain
/// surface, which exist in `D3DPOOL_SCRATCH` alone.
#[test]
fn ati2_capability_answers_follow_the_bc5_storage() {
    let h = Harness::factory_only();
    let answer = |usage: u32, rtype: u32| h.check_device_format(DISPLAY, usage, rtype, D3DFMT_ATI2);
    for rtype in [D3DRTYPE_TEXTURE, D3DRTYPE_CUBETEXTURE] {
        for usage in [
            0,
            D3DUSAGE_QUERY_FILTER,
            D3DUSAGE_DYNAMIC,
            D3DUSAGE_QUERY_WRAPANDMIP,
            D3DUSAGE_QUERY_VERTEXTEXTURE,
        ] {
            assert_eq!(
                answer(usage, rtype),
                D3D_OK,
                "rtype {rtype} usage {usage:#x}"
            );
        }
        assert_eq!(
            answer(D3DUSAGE_AUTOGENMIPMAP, rtype),
            D3DOK_NOAUTOGEN,
            "rtype {rtype} mip generation"
        );
        for usage in [
            D3DUSAGE_RENDERTARGET,
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
            D3DUSAGE_DEPTHSTENCIL,
            D3DUSAGE_QUERY_SRGBREAD,
            D3DUSAGE_QUERY_SRGBWRITE,
        ] {
            assert_eq!(
                answer(usage, rtype),
                D3DERR_NOTAVAILABLE,
                "rtype {rtype} usage {usage:#x}"
            );
        }
    }
    assert_eq!(
        answer(D3DUSAGE_QUERY_LEGACYBUMPMAP, D3DRTYPE_TEXTURE),
        D3DERR_NOTAVAILABLE
    );
    for usage in [0, D3DUSAGE_QUERY_FILTER] {
        assert_eq!(
            answer(usage, D3DRTYPE_VOLUMETEXTURE),
            D3DERR_NOTAVAILABLE,
            "volume usage {usage:#x}"
        );
    }
    for usage in [0, D3DUSAGE_RENDERTARGET, D3DUSAGE_DEPTHSTENCIL] {
        assert_eq!(
            answer(usage, D3DRTYPE_SURFACE),
            D3DERR_NOTAVAILABLE,
            "surface usage {usage:#x}"
        );
    }
}

/// `ATI2` creates as a 2D and a cube texture in every pool, and only there.
///
/// Dynamic usage in the default pool and an extent off the block grid create
/// too. A render-target or depth usage, a render target, and a volume or an
/// offscreen plain outside `D3DPOOL_SCRATCH` are refused.
#[test]
fn ati2_creates_as_2d_and_cube_textures_in_every_pool() {
    let h = Harness::new();
    for (pool, usage) in [
        (D3DPOOL_DEFAULT, 0),
        (D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC),
        (D3DPOOL_MANAGED, 0),
        (D3DPOOL_SYSTEMMEM, 0),
        (D3DPOOL_SCRATCH, 0),
    ] {
        let (hr, out) = h.try_create_texture(16, 16, 0, usage, D3DFMT_ATI2, pool);
        assert_eq!(hr, D3D_OK, "texture pool {pool} usage {usage:#x}");
        let texture = Texture::from_raw(out);
        assert_eq!(texture.level_count(), 5, "pool {pool}");
        let (hr, desc) = texture.level_desc(0);
        assert_eq!(hr, D3D_OK);
        assert_eq!(
            (desc.format, desc.pool, desc.usage),
            (D3DFMT_ATI2, pool, usage),
            "pool {pool}"
        );
        let (hr, out) = h.try_create_cube_texture(16, 0, usage, D3DFMT_ATI2, pool);
        assert_eq!(hr, D3D_OK, "cube pool {pool} usage {usage:#x}");
        drop(CubeTexture::from_raw(out));
    }
    let (hr, out) = h.try_create_texture(6, 10, 1, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
    assert_eq!(hr, D3D_OK, "an extent off the block grid");
    drop(Texture::from_raw(out));

    for usage in [D3DUSAGE_RENDERTARGET, D3DUSAGE_DEPTHSTENCIL] {
        let (hr, out) = h.try_create_texture(16, 16, 1, usage, D3DFMT_ATI2, D3DPOOL_DEFAULT);
        assert_eq!(
            (hr, out.is_null()),
            (D3DERR_INVALIDCALL, true),
            "usage {usage:#x}"
        );
    }
    let (hr, target) =
        h.create_render_target_ms_hr((16, 16), D3DFMT_ATI2, (D3DMULTISAMPLE_NONE, 0), 0);
    assert_eq!((hr, target.is_none()), (D3DERR_INVALIDCALL, true));

    let (hr, volume) = h.try_create_volume_texture([16, 16, 2], 1, 0, D3DFMT_ATI2, D3DPOOL_SCRATCH);
    assert_eq!(hr, D3D_OK, "SCRATCH volume");
    drop(volume);
    assert_eq!(
        h.create_offscreen_plain_surface_seeded(16, 16, D3DFMT_ATI2, D3DPOOL_SCRATCH)
            .0,
        D3D_OK,
        "SCRATCH plain"
    );
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
        let (hr, volume) = h.try_create_volume_texture([16, 16, 2], 1, 0, D3DFMT_ATI2, pool);
        assert_eq!(
            (hr, volume.is_none()),
            (D3DERR_INVALIDCALL, true),
            "volume pool {pool}"
        );
        let (hr, out) = h.create_offscreen_plain_surface_seeded(16, 16, D3DFMT_ATI2, pool);
        assert_eq!(
            (hr, out.is_null()),
            (D3DERR_INVALIDCALL, true),
            "plain pool {pool}"
        );
    }
}

/// A lock of an `ATI2` level reports a byte per texel over the compressed blocks.
///
/// The pitch is the level's width rounded up to the block, tiny mips
/// included, and a sub-rect lock points `y * pitch + x` bytes past the
/// level's start, on a texture in every pool, a cube face and a SCRATCH
/// plain. The lower-right 4x4 of a 128x128 level, traversed row by row at
/// that pitch, ends on the level's last byte.
#[test]
fn ati2_locks_a_byte_per_texel() {
    let h = Harness::new();
    for (width, height, pitch) in [(128, 128, 128), (6, 10, 8), (257, 3, 260)] {
        let texture = h.create_texture(width, height, 1, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
        assert_eq!(
            texture.lock_rect(0, D3DLOCK_READONLY).pitch(),
            pitch,
            "{width}x{height}"
        );
    }
    let chain = h.create_texture(8, 8, 0, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
    for (level, pitch) in [(0, 8), (1, 4), (2, 4), (3, 4)] {
        assert_eq!(
            chain.lock_rect(level, D3DLOCK_READONLY).pitch(),
            pitch,
            "level {level}"
        );
    }

    let rect = [60, 60, 68, 68];
    let corner = [124, 124, 128, 128];
    for (pool, usage) in [
        (D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC),
        (D3DPOOL_MANAGED, 0),
        (D3DPOOL_SYSTEMMEM, 0),
        (D3DPOOL_SCRATCH, 0),
    ] {
        let texture = h.create_texture(128, 128, 1, usage, D3DFMT_ATI2, pool);
        let base = texture.lock_rect(0, D3DLOCK_READONLY).bits_ptr().addr();
        // The corner's last row ends at 124 * 128 + 124 + 3 * 128 + 4 = 16384
        // bytes, the 32 by 32 blocks of 16 bytes the level holds.
        for (region, offset) in [(rect, 60 * 128 + 60), (corner, 124 * 128 + 124)] {
            let lock = texture.lock_rect_partial(0, &region, D3DLOCK_READONLY);
            assert_eq!(lock.pitch(), 128, "pool {pool}");
            assert_eq!(
                lock.bits_ptr().addr() - base,
                offset,
                "pool {pool} {region:?}"
            );
        }
        let cube = h.create_cube_texture_owned(128, 1, usage, D3DFMT_ATI2, pool);
        let base = cube.lock_rect(3, 0, D3DLOCK_READONLY).bits_ptr().addr();
        let lock = cube.lock_rect_partial(3, 0, &rect, D3DLOCK_READONLY);
        assert_eq!(lock.pitch(), 128, "cube pool {pool}");
        assert_eq!(lock.bits_ptr().addr() - base, 7740, "cube pool {pool}");
    }
    let plain = h.create_offscreen_plain_surface(128, 128, D3DFMT_ATI2, D3DPOOL_SCRATCH);
    let base = plain.lock_rect(D3DLOCK_READONLY).bits_ptr().addr();
    let lock = plain.lock_rect_partial(&rect, D3DLOCK_READONLY);
    assert_eq!(lock.pitch(), 128, "SCRATCH plain");
    assert_eq!(lock.bits_ptr().addr() - base, 7740, "SCRATCH plain");
}

/// A SCRATCH `ATI2` volume locks a byte per texel too.
///
/// The row pitch is the width rounded up to the block, a slice spans the
/// level's block rows, and a box starts `front * slice + top * row + left`
/// bytes in.
#[test]
fn an_ati2_scratch_volume_locks_a_byte_per_texel() {
    let h = Harness::new();
    let (hr, volume) = h.try_create_volume_texture([24, 8, 8], 1, 0, D3DFMT_ATI2, D3DPOOL_SCRATCH);
    assert_eq!(hr, D3D_OK);
    let volume = volume.expect("SCRATCH volume");
    let (hr, row, slice, base) = volume.lock_box_layout(0, None);
    assert_eq!((hr, row, slice), (D3D_OK, 24, 192));
    assert_eq!(volume.unlock_box(0), D3D_OK);
    let region = D3DBOX {
        left: 4,
        top: 4,
        right: 8,
        bottom: 8,
        front: 3,
        back: 8,
    };
    let (hr, _, _, bits) = volume.lock_box_layout(0, Some(&region));
    assert_eq!(hr, D3D_OK);
    assert_eq!(bits - base, 3 * 192 + 4 * 24 + 4);
    assert_eq!(volume.unlock_box(0), D3D_OK);
}

/// A level of a default-pool `ATI2` texture locks without dynamic usage.
///
/// D3D9 refuses that lock for every other format; the A8R8G8B8 control keeps
/// the refusal. A default-pool lock keeps the block grid: a rect that splits
/// a block is refused there and accepted in the managed pool.
#[test]
fn a_default_pool_ati2_level_locks_without_dynamic_usage() {
    let h = Harness::new();
    let texture = h.create_texture(16, 16, 1, 0, D3DFMT_ATI2, D3DPOOL_DEFAULT);
    let level = texture.surface_level(0);
    assert_eq!(level.lock_rect_probe(0), (D3D_OK, false), "2D level");
    assert_eq!(level.unlock_rect(), D3D_OK);
    let cube = h.create_cube_texture_owned(16, 1, 0, D3DFMT_ATI2, D3DPOOL_DEFAULT);
    let face = cube.surface(2, 0);
    assert_eq!(face.lock_rect_probe(0), (D3D_OK, false), "cube face");
    assert_eq!(face.unlock_rect(), D3D_OK);
    let control = h.create_texture(16, 16, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT);
    assert_eq!(
        control.surface_level(0).lock_rect_probe(0).0,
        D3DERR_INVALIDCALL,
        "A8R8G8B8 control"
    );

    let split = [2, 0, 4, 4];
    let dynamic = h.create_texture(16, 16, 1, D3DUSAGE_DYNAMIC, D3DFMT_ATI2, D3DPOOL_DEFAULT);
    assert_eq!(
        dynamic
            .surface_level(0)
            .lock_rect_partial_probe(&split, 0)
            .0,
        D3DERR_INVALIDCALL,
        "default pool"
    );
    let managed = h.create_texture(16, 16, 1, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
    let level = managed.surface_level(0);
    assert_eq!(
        level.lock_rect_partial_probe(&split, 0).0,
        D3D_OK,
        "managed pool"
    );
    assert_eq!(level.unlock_rect(), D3D_OK);
}

/// Each block samples green from its first eight bytes and red from its second.
///
/// An 8x4 level of two blocks, one with each half at 50 %, reads green in
/// its left half and red in its right, with blue and alpha one, written
/// through a managed texture's lock and through `UpdateTexture` into a
/// non-dynamic default-pool one alike. That default-pool level then locks
/// through its surface and reads back the blocks.
#[test]
fn ati2_samples_green_from_the_first_half_and_red_from_the_second() {
    let h = Harness::new();
    setup(&h);
    let mut blocks = block(HALF, ZERO);
    blocks.extend(block(ZERO, HALF));

    let managed = h.create_texture(8, 4, 1, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
    {
        let mut lock = managed.lock_rect(0, 0);
        assert_eq!(lock.pitch(), 8);
        lock.write(&blocks);
    }
    assert_eq!(h.set_texture(0, &managed), D3D_OK);
    assert_pixel_approx(sample(&h, 0.25), GREEN_HALF, 1, "managed first block");
    assert_pixel_approx(sample(&h, 0.75), RED_HALF, 1, "managed second block");

    let source = h.create_texture(8, 4, 1, 0, D3DFMT_ATI2, D3DPOOL_SYSTEMMEM);
    source.lock_rect(0, 0).write(&blocks);
    let target = h.create_texture(8, 4, 1, 0, D3DFMT_ATI2, D3DPOOL_DEFAULT);
    assert_eq!(h.update_texture_hr(&source, &target), D3D_OK);
    assert_eq!(h.set_texture(0, &target), D3D_OK);
    assert_pixel_approx(sample(&h, 0.25), GREEN_HALF, 1, "default first block");
    assert_pixel_approx(sample(&h, 0.75), RED_HALF, 1, "default second block");
    // The default-pool level, sampled and so uploaded, locks through its
    // surface and hands back the blocks it holds.
    let level = target.surface_level(0);
    let lock = level.lock_rect(D3DLOCK_READONLY);
    assert_eq!(lock.pitch(), 8);
    assert_eq!(lock.as_u8(blocks.len()), blocks.as_slice());
}

/// A sub-rect lock publishes the whole level.
///
/// The lock of the left 8x4 of a 16x4 level points at the level's start
/// with a 16-byte pitch, so its third row is the first half of the third
/// block, which lies outside the rect's blocks. Zeroing it there turns that
/// block from green to neither, and the draw sees it.
#[test]
fn an_ati2_sub_rect_lock_publishes_the_whole_level() {
    let h = Harness::new();
    setup(&h);
    let texture = h.create_texture(16, 4, 1, 0, D3DFMT_ATI2, D3DPOOL_MANAGED);
    texture.lock_rect(0, 0).write(&block(HALF, ZERO).repeat(4));
    assert_eq!(h.set_texture(0, &texture), D3D_OK);
    assert_pixel_approx(sample(&h, 0.625), GREEN_HALF, 1, "third block before");
    {
        let mut lock = texture.lock_rect_partial(0, &[0, 0, 8, 4], 0);
        assert_eq!(lock.pitch(), 16);
        let mut rows = HALF.to_vec();
        rows.extend_from_slice(&HALF);
        rows.extend_from_slice(&ZERO);
        lock.write_u8_rect(8, 3, &rows);
    }
    assert_pixel_approx(sample(&h, 0.625), NEITHER, 1, "third block after");
    assert_pixel_approx(sample(&h, 0.125), GREEN_HALF, 1, "first block unchanged");
    assert_pixel_approx(sample(&h, 0.375), GREEN_HALF, 1, "second block unchanged");
}
