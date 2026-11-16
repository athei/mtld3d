//! A texture level nothing writes samples black, whatever device memory its texture lands on.
//!
//! D3D9 backs a new texture with zeroed memory, so a DXT1 level that is never
//! written reads as all-zero blocks, opaque black. Each probe follows the
//! release of textures of its shape whose levels held a bright block, which
//! is the memory a new texture of that shape can be given.

use mtld3d_tests::{Harness, Rgba8, Texture, TexturedVertex};
use mtld3d_types::{
    D3DFMT_DXT1, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER,
    D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTADDRESS_CLAMP, D3DTEXF_POINT,
};

const BLACK: u32 = 0xFF00_0000;

/// Side of the square DXT1 textures the probes create.
const PROBE_SIDE: u32 = 16;

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

/// Draw `tex` point-sampled across the back buffer over `clear`, return the quadrant centres.
fn draw_quadrants(h: &Harness, tex: &Texture<'_>, clear: u32) -> [u32; 4] {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
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
        "SetFVF"
    );
    let quad = fullscreen_quad();
    h.render_once(clear, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0, "draw");
    });
    [
        h.read_pixel(160, 120),
        h.read_pixel(480, 120),
        h.read_pixel(480, 360),
        h.read_pixel(160, 360),
    ]
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
    let side = PROBE_SIDE >> level;
    let (row_bytes, bytes) = dxt1_level_bytes(PROBE_SIDE, level, block);
    let source = h.create_offscreen_plain_surface(side, side, D3DFMT_DXT1, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u8_rect(row_bytes, bytes.len() / row_bytes, &bytes);
    assert_eq!(
        h.update_surface_hr(&source, &tex.surface_level(level)),
        0,
        "UpdateSurface into level {level}"
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
    const ROUNDS: usize = 4;
    const RELEASED_PER_ROUND: usize = 8;
    // A clear that is not black, so a probe draw that never landed shows.
    const CLEAR: u32 = 0xFFFF_00FF;
    const SIDE: u32 = PROBE_SIDE;
    let yellow = dxt1_solid_block(0xFFE0);
    let red = dxt1_solid_block(0xF800);
    let h = Harness::new();
    // The quad magnifies, so the most detailed level the clamp allows is
    // the level every fragment reads: level 1.
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    let mut sampled = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        for _ in 0..RELEASED_PER_ROUND {
            let tex = h.create_texture(SIDE, SIDE, 2, 0, D3DFMT_DXT1, pool);
            fill(&h, &tex, 0, yellow);
            fill(&h, &tex, 1, yellow);
            let px = Rgba8::from_pixel(draw_quadrants(&h, &tex, BLACK)[0]);
            assert!(
                px.r > 200 && px.g > 200 && px.b < 60,
                "the released texture's level 1 reached the GPU, got {px:?}"
            );
            assert_eq!(h.clear_texture(0), 0);
            drop(tex);
            assert_eq!(
                h.present(),
                0,
                "the Present that retires the released texture"
            );
        }
        let probe = h.create_texture(SIDE, SIDE, 2, 0, D3DFMT_DXT1, pool);
        fill(&h, &probe, 0, red);
        sampled.push(draw_quadrants(&h, &probe, CLEAR));
        assert_eq!(h.clear_texture(0), 0);
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
