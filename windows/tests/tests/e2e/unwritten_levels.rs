//! A texture level nothing writes samples black, whatever device memory its texture lands on.
//!
//! D3D9 backs a new texture with zeroed memory, so a DXT1 level that is never
//! written reads as all-zero blocks, opaque black, and so do the texels a
//! partial write of a new level leaves alone. Each probe follows the release
//! of textures of its shape whose levels held a bright block, which is the
//! memory a new texture of that shape can be given.

use mtld3d_tests::{
    CubeTexture, Harness, Rgba8, Surface, Texture, TexturedVertex, VolumeTexture, VolumeVertex,
};
use mtld3d_types::{
    D3DBOX, D3DFMT_DXT1, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_TEXTUREFORMAT3,
    D3DFVF_XYZ, D3DLOCK_DISCARD, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRECT, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_ADDRESSW,
    D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_DYNAMIC,
};

const BLACK: u32 = 0xFF00_0000;

/// Side of the square DXT1 textures the probes create.
const PROBE_SIDE: u32 = 16;

/// Release-then-probe rounds a test runs; any round may land on released memory.
const ROUNDS: usize = 4;

/// Textures of the probe's shape released before each probe.
const RELEASED_PER_ROUND: usize = 8;

/// A clear that is not black, so a probe draw that never landed shows.
const CLEAR: u32 = 0xFFFF_00FF;

/// The colour lanes of a sampled pixel, which every device stores as written.
const RGB: u32 = 0x00FF_FFFF;

/// The colour lanes of a DXT1 block whose endpoints are `0xF800`.
const RED_RGB: u32 = 0x00FF_0000;

/// The centres of the four DXT1 block columns of a [`PROBE_SIDE`] level drawn across the target.
const BLOCK_COLUMNS: [(u32, u32); 4] = [(80, 240), (240, 240), (400, 240), (560, 240)];

/// What the block columns of a level written in its outer two columns alone sample, colour lanes.
const OUTER_COLUMNS_ONLY: [u32; 4] = [RED_RGB, 0, 0, RED_RGB];

const YELLOW_565: u16 = 0xFFE0;
const RED_565: u16 = 0xF800;

/// The 32-bit texel the released volumes and 32-bit levels are filled with.
const YELLOW_X8: u32 = 0xFFFF_FF00;

/// The 32-bit texel a probe's partial write stores.
const RED_X8: u32 = 0xFFFF_0000;

/// A full-backbuffer quad (two triangles) with UVs spanning the unit square.
///
/// White vertex colour so MODULATE passes the texel through.
fn fullscreen_quad() -> [TexturedVertex; 6] {
    let corner = |x: f32, y: f32, u: f32, v: f32| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xFFFF_FFFF,
        u,
        v,
    };
    [
        corner(-1.0, 1.0, 0.0, 0.0),
        corner(1.0, 1.0, 1.0, 0.0),
        corner(-1.0, -1.0, 0.0, 1.0),
        corner(1.0, 1.0, 1.0, 0.0),
        corner(1.0, -1.0, 1.0, 1.0),
        corner(-1.0, -1.0, 0.0, 1.0),
    ]
}

/// A full-backbuffer quad whose direction coordinates sweep the `+X` cube face.
///
/// D3D9 maps `+X` with `s = -z` and `t = -y`, so the screen's top left looks
/// along `(1, 1, 1)` and reads the face's top-left texel, as a 2D quad would.
fn positive_x_face_quad() -> [VolumeVertex; 6] {
    volume_quad(|x, y| [1.0, y, -x])
}

/// A full-backbuffer quad sampling a volume at `(u, 0.5, w)`, `u` spanning the screen's width.
fn volume_row_quad(w: f32) -> [VolumeVertex; 6] {
    volume_quad(|x, _| [f32::midpoint(x, 1.0), 0.5, w])
}

/// A full-backbuffer quad whose three texture coordinates `coord` gives at each corner.
fn volume_quad(coord: impl Fn(f32, f32) -> [f32; 3]) -> [VolumeVertex; 6] {
    let corner = |x: f32, y: f32| {
        let [u, v, w] = coord(x, y);
        VolumeVertex {
            x,
            y,
            z: 0.5,
            color: 0xFFFF_FFFF,
            u,
            v,
            w,
        }
    };
    [
        corner(-1.0, 1.0),
        corner(1.0, 1.0),
        corner(-1.0, -1.0),
        corner(1.0, 1.0),
        corner(1.0, -1.0),
        corner(-1.0, -1.0),
    ]
}

/// Point-sample stage 0 with every address mode clamped.
fn point_clamp(h: &Harness) {
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSW, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
}

/// Draw `quad` with `fvf` over `clear`, read `points`, and unbind stage 0.
fn draw_and_read<V, const N: usize>(
    h: &Harness,
    fvf: u32,
    quad: &[V; 6],
    clear: u32,
    points: [(u32, u32); N],
) -> [u32; N] {
    point_clamp(h);
    assert_eq!(h.set_fvf(fvf), 0, "SetFVF");
    h.render_once(clear, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, quad), 0, "draw");
    });
    let read = points.map(|(x, y)| h.read_pixel(x, y));
    assert_eq!(h.clear_texture(0), 0, "unbind stage 0");
    read
}

/// Draw `tex` point-sampled across the back buffer over `clear`, read `points`.
fn draw_2d<const N: usize>(
    h: &Harness,
    tex: &Texture<'_>,
    clear: u32,
    points: [(u32, u32); N],
) -> [u32; N] {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    draw_and_read(
        h,
        D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1,
        &fullscreen_quad(),
        clear,
        points,
    )
}

/// Draw the `+X` face of `cube` point-sampled across the back buffer, read `points`.
fn draw_cube_face<const N: usize>(
    h: &Harness,
    cube: &CubeTexture<'_>,
    clear: u32,
    points: [(u32, u32); N],
) -> [u32; N] {
    assert_eq!(h.set_cube_texture(0, cube), 0, "SetTexture cube");
    draw_and_read(
        h,
        D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16),
        &positive_x_face_quad(),
        clear,
        points,
    )
}

/// Draw the row `v = 0.5` of slice `slice` of a two-slice `volume`, read `points`.
fn draw_volume_row<const N: usize>(
    h: &Harness,
    (volume, slice): (&VolumeTexture<'_>, u8),
    clear: u32,
    points: [(u32, u32); N],
) -> [u32; N] {
    assert_eq!(h.set_volume_texture(0, volume), 0, "SetTexture volume");
    draw_and_read(
        h,
        D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16),
        &volume_row_quad(f32::midpoint(f32::from(slice), 0.5)),
        clear,
        points,
    )
}

/// Draw `tex` point-sampled across the back buffer over `clear`, return the quadrant centres.
fn draw_quadrants(h: &Harness, tex: &Texture<'_>, clear: u32) -> [u32; 4] {
    draw_2d(
        h,
        tex,
        clear,
        [(160, 120), (480, 120), (480, 360), (160, 360)],
    )
}

/// Fail unless a released texture's level reached the GPU as the bright yellow it was filled with.
#[track_caller]
fn assert_released_yellow(pixel: u32) {
    let px = Rgba8::from_pixel(pixel);
    assert!(
        px.r > 200 && px.g > 200 && px.b < 60,
        "the released texture's level reached the GPU, got {px:?}"
    );
}

/// Run `release_one` [`RELEASED_PER_ROUND`] times, each followed by the Present that retires it.
///
/// `release_one` creates a texture of the probe's shape, fills it, draws it
/// and drops it, so its staging goes back to the page-box pool once the
/// upload that read it retires.
fn release_round(h: &Harness, release_one: impl Fn()) {
    for _ in 0..RELEASED_PER_ROUND {
        release_one();
        assert_eq!(
            h.present(),
            0,
            "the Present that retires the released texture"
        );
    }
}

/// Run [`ROUNDS`] rounds of releases followed by a probe, collect each probe's colour lanes.
fn probe_after_releases<const N: usize>(
    h: &Harness,
    release_one: impl Fn(),
    probe: impl Fn() -> [u32; N],
) -> Vec<[u32; N]> {
    (0..ROUNDS)
        .map(|_| {
            release_round(h, &release_one);
            probe().map(|pixel| pixel & RGB)
        })
        .collect()
}

/// One solid DXT1 block: both endpoints `color565`, every index 0.
const fn dxt1_solid_block(color565: u16) -> [u8; 8] {
    let [lo, hi] = color565.to_le_bytes();
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// One DXT1 level `side >> level` texels wide filled with `block`, as block rows.
///
/// Returns the bytes of one block row and the block rows, tightly packed.
fn dxt1_level_bytes(side: u32, level: u32, block: [u8; 8]) -> (usize, Vec<u8>) {
    let blocks_across = (side >> level).div_ceil(4) as usize;
    let row_bytes = blocks_across * block.len();
    let bytes = block
        .iter()
        .copied()
        .cycle()
        .take(row_bytes * blocks_across)
        .collect();
    (row_bytes, bytes)
}

/// Fill `level` of a [`PROBE_SIDE`] DXT1 texture with `block` through `LockRect`.
fn lock_dxt1_level(_h: &Harness, tex: &Texture<'_>, level: u32, block: [u8; 8]) {
    let (row_bytes, bytes) = dxt1_level_bytes(PROBE_SIDE, level, block);
    tex.lock_rect(level, 0)
        .write_u8_rect(row_bytes, bytes.len() / row_bytes, &bytes);
}

/// Fill `level` of a [`PROBE_SIDE`] DXT1 texture with `block` by `UpdateSurface`.
///
/// The source is a system-memory surface of the level's size, so nothing
/// but the named level of the destination is written.
fn update_dxt1_level(h: &Harness, tex: &Texture<'_>, level: u32, block: [u8; 8]) {
    update_dxt1_surface(h, &tex.surface_level(level), PROBE_SIDE >> level, block);
}

/// Fill the `side`-square DXT1 surface `dst` with `block` by `UpdateSurface`.
fn update_dxt1_surface(h: &Harness, dst: &Surface<'_>, side: u32, block: [u8; 8]) {
    let (row_bytes, bytes) = dxt1_level_bytes(side, 0, block);
    let source = h.create_offscreen_plain_surface(side, side, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u8_rect(row_bytes, bytes.len() / row_bytes, &bytes);
    assert_eq!(h.update_surface_hr(&source, dst), 0, "UpdateSurface");
}

/// Write red into the first and last block columns of a [`PROBE_SIDE`] DXT1 surface.
///
/// Two disjoint `UpdateSurface` writes from one system-memory block column,
/// so the upload they schedule spans the two columns between them.
fn update_outer_block_columns(h: &Harness, dst: &Surface<'_>) {
    let rows = (PROBE_SIDE / 4) as usize;
    let column = dxt1_solid_block(RED_565).repeat(rows);
    let source = h.create_offscreen_plain_surface(4, PROBE_SIDE, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    source.lock_rect(0).write_u8_rect(8, rows, &column);
    let rect = D3DRECT {
        x1: 0,
        y1: 0,
        x2: 4,
        y2: PROBE_SIDE.cast_signed(),
    };
    for x in [0, PROBE_SIDE - 4] {
        assert_eq!(
            h.update_surface_region_hr(&source, &rect, dst, (x.cast_signed(), 0)),
            0,
            "UpdateSurface of the block column at x {x}"
        );
    }
}

/// Fail unless every probe sampled red in its outer block columns and black between them.
#[track_caller]
fn assert_outer_columns_only(sampled: &[[u32; 4]], what: &str) {
    assert!(
        sampled.iter().all(|columns| *columns == OUTER_COLUMNS_ONLY),
        "{what}: the block columns between the two writes, never written, sample black in every \
         round, got {sampled:06x?}"
    );
}

/// Sample level 1 of a new two-level DXT1 texture whose level 1 nothing writes.
///
/// Before each of four probes, eight textures of the same shape in `pool`
/// are created, both levels filled yellow by `fill`, drawn and released, so
/// the device memory a new texture of that shape can land on held a bright
/// block. Each probe has only level 0 filled, red, by `fill`, and its four
/// quadrant centres, drawn over a non-black clear, must read opaque black.
fn check_unwritten_level_samples_black(
    pool: u32,
    fill: impl Fn(&Harness, &Texture<'_>, u32, [u8; 8]),
) {
    const SIDE: u32 = PROBE_SIDE;
    let yellow = dxt1_solid_block(YELLOW_565);
    let red = dxt1_solid_block(RED_565);
    let h = Harness::new();
    // The quad magnifies, so the most detailed level the clamp allows is
    // the level every fragment reads: level 1.
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    let mut sampled = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        release_round(&h, || {
            let tex = h.create_texture(SIDE, SIDE, 2, 0, D3DFMT_DXT1, pool);
            fill(&h, &tex, 0, yellow);
            fill(&h, &tex, 1, yellow);
            assert_released_yellow(draw_quadrants(&h, &tex, BLACK)[0]);
        });
        let probe = h.create_texture(SIDE, SIDE, 2, 0, D3DFMT_DXT1, pool);
        fill(&h, &probe, 0, red);
        sampled.push(draw_quadrants(&h, &probe, CLEAR));
    }
    assert!(
        sampled.iter().all(|quadrants| *quadrants == [BLACK; 4]),
        "pool {pool}: the probe's level 1, never written, samples opaque black in every \
         round, got {sampled:08x?}"
    );
}

/// A managed level no upload reaches samples opaque black after same-shape textures go.
///
/// The levels are written by `LockRect`, so the probe's level 1 has CPU
/// staging that no upload reads.
#[test]
fn a_level_no_upload_reaches_samples_black_after_released_textures_of_its_shape() {
    check_unwritten_level_samples_black(D3DPOOL_MANAGED, lock_dxt1_level);
}

/// A default-pool level no update reaches samples opaque black after same-shape textures go.
///
/// The levels are written only by `UpdateSurface` from system-memory
/// surfaces, so the probe's level 1 receives nothing from any CPU copy and
/// samples whatever device memory its texture was given.
#[test]
fn a_default_pool_level_no_update_reaches_samples_black_after_released_textures() {
    check_unwritten_level_samples_black(D3DPOOL_DEFAULT, update_dxt1_level);
}

/// The texels between two `UpdateSurface` writes of a new default-pool level sample black.
///
/// The level's staging is re-created at the first write, from the page-box
/// pool the released textures' levels went back to, and the upload of the
/// two writes is their bounding box, which covers the columns between them.
#[test]
fn texels_between_two_updates_of_a_new_default_pool_level_sample_black() {
    let yellow = dxt1_solid_block(YELLOW_565);
    let h = Harness::new();
    let sampled = probe_after_releases(
        &h,
        || {
            let tex = h.create_texture(PROBE_SIDE, PROBE_SIDE, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
            update_dxt1_level(&h, &tex, 0, yellow);
            assert_released_yellow(draw_quadrants(&h, &tex, BLACK)[0]);
        },
        || {
            let probe =
                h.create_texture(PROBE_SIDE, PROBE_SIDE, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
            update_outer_block_columns(&h, &probe.surface_level(0));
            draw_2d(&h, &probe, CLEAR, BLOCK_COLUMNS)
        },
    );
    assert_outer_columns_only(&sampled, "default-pool 2D level");
}

/// The texels between two `UpdateSurface` writes of a new default-pool cube face sample black.
///
/// A cube keeps the staging its create took from the page-box pool, which
/// the released cubes' faces went back to.
#[test]
fn texels_between_two_updates_of_a_new_default_pool_cube_face_sample_black() {
    let yellow = dxt1_solid_block(YELLOW_565);
    let h = Harness::new();
    let sampled = probe_after_releases(
        &h,
        || {
            let cube = h.create_cube_texture_owned(PROBE_SIDE, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
            for face in 0..6 {
                update_dxt1_surface(&h, &cube.surface(face, 0), PROBE_SIDE, yellow);
            }
            assert_released_yellow(draw_cube_face(&h, &cube, BLACK, [(320, 240)])[0]);
        },
        || {
            let probe = h.create_cube_texture_owned(PROBE_SIDE, 1, 0, D3DFMT_DXT1, D3DPOOL_DEFAULT);
            update_outer_block_columns(&h, &probe.surface(0, 0));
            draw_cube_face(&h, &probe, CLEAR, BLOCK_COLUMNS)
        },
    );
    assert_outer_columns_only(&sampled, "default-pool cube face");
}

/// The texels between two partial locks of a new dynamic default-pool level sample black.
///
/// A dynamic texture keeps the staging its create took from the page-box
/// pool, and the upload of the two locks is their bounding box.
#[test]
fn texels_between_two_locks_of_a_new_dynamic_default_pool_level_sample_black() {
    let yellow = dxt1_solid_block(YELLOW_565);
    let rows = (PROBE_SIDE / 4) as usize;
    let column = dxt1_solid_block(RED_565).repeat(rows);
    let h = Harness::new();
    let create = || {
        h.create_texture(
            PROBE_SIDE,
            PROBE_SIDE,
            1,
            D3DUSAGE_DYNAMIC,
            D3DFMT_DXT1,
            D3DPOOL_DEFAULT,
        )
    };
    let sampled = probe_after_releases(
        &h,
        || {
            let tex = create();
            lock_dxt1_level(&h, &tex, 0, yellow);
            assert_released_yellow(draw_quadrants(&h, &tex, BLACK)[0]);
        },
        || {
            let probe = create();
            let side = PROBE_SIDE.cast_signed();
            for left in [0, side - 4] {
                probe
                    .lock_rect_partial(0, &[left, 0, left + 4, side], 0)
                    .write_u8_rect(8, rows, &column);
            }
            draw_2d(&h, &probe, CLEAR, BLOCK_COLUMNS)
        },
    );
    assert_outer_columns_only(&sampled, "dynamic default-pool 2D level");
}

/// The texels a partial `LockBox` leaves alone in a new dynamic default-pool volume sample black.
///
/// The volume keeps the staging its create took from the page-box pool, and
/// a volume level uploads whole, so one partial lock is enough to carry the
/// rest of the staging to the GPU.
#[test]
fn texels_a_partial_lock_box_leaves_alone_in_a_new_dynamic_default_pool_volume_sample_black() {
    const EXTENT: [u32; 3] = [8, 8, 2];
    const TEXELS: usize = 8 * 8 * 2;
    // The left half of the volume, every row and slice.
    const LEFT_HALF: D3DBOX = D3DBOX {
        left: 0,
        top: 0,
        right: 4,
        bottom: 8,
        front: 0,
        back: 2,
    };
    // The centres of the left and right halves of the row the draw samples.
    const HALVES: [(u32, u32); 2] = [(160, 240), (480, 240)];
    let h = Harness::new();
    let create = || {
        let (hr, volume) = h.try_create_volume_texture(
            EXTENT,
            1,
            D3DUSAGE_DYNAMIC,
            D3DFMT_X8R8G8B8,
            D3DPOOL_DEFAULT,
        );
        assert_eq!(hr, 0, "CreateVolumeTexture");
        volume.expect("a volume texture")
    };
    let sampled = probe_after_releases(
        &h,
        || {
            let released = create();
            released.write_u32(0, &[YELLOW_X8; TEXELS]);
            assert_released_yellow(draw_volume_row(&h, (&released, 0), BLACK, [(320, 240)])[0]);
        },
        || {
            let probe = create();
            probe.write_box_u32(0, &LEFT_HALF, &[RED_X8; TEXELS / 2]);
            draw_volume_row(&h, (&probe, 0), CLEAR, HALVES)
        },
    );
    assert!(
        sampled.iter().all(|halves| *halves == [RED_X8 & RGB, 0]),
        "dynamic default-pool volume: the half the lock left alone samples black in every \
         round, got {sampled:06x?}"
    );
}

/// The slice an `UpdateTexture` from a shallower source leaves alone in a new volume samples black.
///
/// D3D9 pairs volume levels on their largest extent, so an 8x8x1 source
/// updates an 8x8x2 level: the copy covers every row of the level but only
/// its first slice, and the level uploads whole.
#[test]
fn the_slice_an_update_texture_from_a_shallower_volume_leaves_alone_samples_black() {
    const EXTENT: [u32; 3] = [8, 8, 2];
    const TEXELS: usize = 8 * 8 * 2;
    let h = Harness::new();
    let volume = |extent, pool| {
        let (hr, volume) = h.try_create_volume_texture(extent, 1, 0, D3DFMT_X8R8G8B8, pool);
        assert_eq!(hr, 0, "CreateVolumeTexture {extent:?} in pool {pool}");
        volume.expect("a volume texture")
    };
    let sampled = probe_after_releases(
        &h,
        || {
            let source = volume(EXTENT, D3DPOOL_SYSTEMMEM);
            source.write_u32(0, &[YELLOW_X8; TEXELS]);
            let released = volume(EXTENT, D3DPOOL_DEFAULT);
            assert_eq!(h.update_volume_texture_hr(&source, &released), 0);
            assert_released_yellow(draw_volume_row(&h, (&released, 1), BLACK, [(320, 240)])[0]);
        },
        || {
            // Created first, so its staging is a box a released volume parked.
            let probe = volume(EXTENT, D3DPOOL_DEFAULT);
            let source = volume([8, 8, 1], D3DPOOL_SYSTEMMEM);
            source.write_u32(0, &[RED_X8; TEXELS / 2]);
            assert_eq!(
                h.update_volume_texture_hr(&source, &probe),
                0,
                "UpdateTexture from the 8x8x1 source"
            );
            [0, 1].map(|slice| draw_volume_row(&h, (&probe, slice), CLEAR, [(320, 240)])[0])
        },
    );
    assert!(
        sampled.iter().all(|slices| *slices == [RED_X8 & RGB, 0]),
        "default-pool volume: the slice the copy left alone samples black in every round, got \
         {sampled:06x?}"
    );
}

/// A first `D3DLOCK_DISCARD` lock of a new dynamic default-pool level hands out zero.
///
/// The lock covers the whole level and its upload does too, so the block
/// columns the write after it leaves alone sample black.
#[test]
fn a_first_discard_lock_of_a_new_dynamic_default_pool_level_hands_out_zero() {
    let yellow = dxt1_solid_block(YELLOW_565);
    let rows = (PROBE_SIDE / 4) as usize;
    let column = dxt1_solid_block(RED_565).repeat(rows);
    let h = Harness::new();
    let create = || {
        h.create_texture(
            PROBE_SIDE,
            PROBE_SIDE,
            1,
            D3DUSAGE_DYNAMIC,
            D3DFMT_DXT1,
            D3DPOOL_DEFAULT,
        )
    };
    let sampled = probe_after_releases(
        &h,
        || {
            let tex = create();
            lock_dxt1_level(&h, &tex, 0, yellow);
            assert_released_yellow(draw_quadrants(&h, &tex, BLACK)[0]);
        },
        || {
            let probe = create();
            // The first block of every block row: the first block column.
            probe
                .lock_rect(0, D3DLOCK_DISCARD)
                .write_u8_rect(8, rows, &column);
            draw_2d(&h, &probe, CLEAR, BLOCK_COLUMNS)
        },
    );
    assert!(
        sampled.iter().all(|columns| *columns == [RED_RGB, 0, 0, 0]),
        "dynamic default-pool level under D3DLOCK_DISCARD: the block columns the write left \
         alone sample black in every round, got {sampled:06x?}"
    );
}

/// A device context of a new default-pool level maps zero where GDI draws nothing.
///
/// `GetDC` re-creates the level's staging from the page-box pool the
/// released textures' levels went back to, and `ReleaseDC` uploads the whole
/// level. GDI paints the top-left quadrant red; the other three sample black.
#[test]
fn a_device_context_of_a_new_default_pool_level_maps_zero_where_gdi_draws_nothing() {
    const SIDE: u32 = PROBE_SIDE;
    // `COLORREF` red, `0x00BBGGRR`.
    const RED_COLORREF: u32 = 0x0000_00FF;
    let h = Harness::new();
    let create = || h.create_texture(SIDE, SIDE, 1, 0, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT);
    let sampled = probe_after_releases(
        &h,
        || {
            let tex = create();
            let source =
                h.create_offscreen_plain_surface(SIDE, SIDE, D3DFMT_X8R8G8B8, D3DPOOL_SYSTEMMEM);
            source
                .lock_rect(0)
                .write_u32(&vec![YELLOW_X8; (SIDE * SIDE) as usize]);
            assert_eq!(
                h.update_surface_hr(&source, &tex.surface_level(0)),
                0,
                "UpdateSurface"
            );
            assert_released_yellow(draw_quadrants(&h, &tex, BLACK)[0]);
        },
        || {
            let probe = create();
            let surface = probe.surface_level(0);
            let dc = surface.dc();
            dc.fill_block((SIDE / 2).cast_signed(), RED_COLORREF);
            assert_eq!(dc.release(), 0, "ReleaseDC");
            draw_quadrants(&h, &probe, CLEAR)
        },
    );
    assert!(
        sampled
            .iter()
            .all(|quadrants| *quadrants == [RED_X8 & RGB, 0, 0, 0]),
        "default-pool level under GetDC: the quadrants GDI left alone sample black in every \
         round, got {sampled:06x?}"
    );
}
