//! The non-UMA storage policy, forced via `intel.managedMemory` and `intel.linearAlign256`.
//!
//! An Intel/AMD Mac keeps CPU-visible buffers in Metal's Managed storage
//! mode, notifies every CPU write with `didModifyRange:`, and needs a
//! 256-byte row alignment for a linear texture upload, so mips narrower than
//! that floor go through padded staging or the GPU upload pass. These tests
//! force both answers so the paths run on Apple Silicon in every `make test`,
//! where they would otherwise wait for the rare Intel-hardware run.
//!
//! What they can prove there: the Managed buffers are created, written, drawn
//! from, renamed and read back with the right pixels, and the padded and
//! upload-pass texture paths carry the texels written into every level. What
//! they cannot: on unified memory the GPU sees a CPU write with or without its
//! notify, so a `didModifyRange:` that was left out does not fail here. The
//! `GpuCaps` unit tests pin that the keys reach the snapshot; nothing D3D9
//! answers changes under them, so no assertion below can tell the forced run
//! from the native one.

use mtld3d_tests::{Harness, Texture, TexturedVertex, Vertex};
use mtld3d_types::{
    D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFMT_INDEX16, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_XYZ, D3DLOCK_DISCARD, D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED,
    D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DSAMP_ADDRESSU,
    D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY,
};

const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
const BLACK: u32 = 0xFF00_0000;
const BLUE: u32 = 0xFF00_00FF;
const MAGENTA: u32 = 0xFFFF_00FF;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;
const WHITE: u32 = 0xFFFF_FFFF;

/// A device under the non-UMA storage policy and the Mac2 alignment floor.
///
/// Both keys are resolved by this harness's `Direct3DCreate9` alone, so the
/// rest of the suite, sharing the process, keeps the device's own answers.
fn non_uma_harness() -> Harness {
    Harness::with_config("intel.managedMemory=true;intel.linearAlign256=true")
}

fn stride() -> u32 {
    u32::try_from(size_of::<Vertex>()).expect("vertex stride fits u32")
}

const fn solid_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

/// Drive the fixed-function pipeline so a draw shows the vertex diffuse colour.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(
        h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE),
        0,
        "cull off"
    );
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(FVF), 0, "SetFVF");
}

/// A full-backbuffer quad (two triangles) with UVs spanning the unit square.
const fn fullscreen_quad() -> [TexturedVertex; 6] {
    const fn v(x: f32, y: f32, u: f32, tv: f32) -> TexturedVertex {
        TexturedVertex {
            x,
            y,
            z: 0.5,
            color: WHITE,
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

/// Bind `tex`, sample mip `level` across the backbuffer, return the centre pixel.
///
/// `D3DSAMP_MAXMIPLEVEL` pins the most detailed level the sampler may use;
/// the quad magnifies, so that is the level every fragment reads.
fn sample_level(h: &Harness, tex: &Texture<'_>, level: u32) -> u32 {
    assert_eq!(h.set_texture(0, tex), 0, "SetTexture");
    h.select_texture_stage(0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
        (D3DSAMP_MAXMIPLEVEL, level),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0, "sampler");
    }
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

/// Fill the whole of mip `level` with one 32-bit texel, honouring the lock's pitch.
fn fill_level(tex: &Texture<'_>, level: u32, side: usize, texel: u32) {
    let locked = tex.lock_rect(level, 0);
    let pitch = usize::try_from(locked.pitch()).expect("positive pitch");
    let row = vec![texel; side];
    for y in 0..side {
        // SAFETY: the lock maps `side` rows at `pitch` stride.
        let dst = unsafe { locked.bits_ptr().add(y * pitch) };
        // SAFETY: `side` texels fit in the locked row per above.
        unsafe {
            core::ptr::copy_nonoverlapping(row.as_ptr().cast::<u8>(), dst, side * 4);
        }
    }
}

/// A Managed vertex buffer and a Managed index buffer draw what was written into them.
#[test]
fn managed_vertex_and_index_buffers_draw() {
    let h = non_uma_harness();
    let verts = [
        Vertex {
            x: -0.5,
            y: 0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: MAGENTA,
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 1, 3, 2];
    let vb = h.create_vertex_buffer(stride() * 4, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&verts);
    let ib = h.create_index_buffer(12, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    ib.lock(0, 0, 0).write(&indices);

    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    assert_eq!(h.set_indices(&ib), 0, "SetIndices");
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            0,
            "DrawIndexedPrimitive"
        );
    });
    assert_eq!(h.read_pixel(320, 240), MAGENTA, "the indexed quad renders");
    assert_eq!(
        h.read_pixel(10, 10),
        BLUE,
        "outside the quad stays background"
    );
}

/// A dynamic Managed buffer refilled with `D3DLOCK_DISCARD` draws each fill, mid-frame too.
///
/// The second frame discards between two draws of the same triangle, which
/// renames the backing while the first draw still references the old one.
#[test]
fn managed_dynamic_discard_refill_renames() {
    let h = non_uma_harness();
    let vb = h.create_vertex_buffer(
        stride() * 3,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        FVF,
        D3DPOOL_DEFAULT,
    );
    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );

    vb.lock(0, 0, D3DLOCK_DISCARD).write(&solid_triangle(GREEN));
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(h.read_pixel(320, 280), GREEN, "the first fill is green");

    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "first draw");
        vb.lock(0, 0, D3DLOCK_DISCARD).write(&solid_triangle(RED));
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "second draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        RED,
        "the mid-frame DISCARD refill is what the second draw reads"
    );
}

/// Every level of a chain whose pitches all fall under the 256-byte floor samples its texels.
///
/// 32x32 in a 32-bit format has pitches of 128, 64, 32, 16, 8 and 4 bytes:
/// on Apple Silicon only the last two are under the 16-byte alignment, on
/// the forced floor all six are. A8R8G8B8 takes the GPU upload pass;
/// X8R8G8B8, whose backing is swizzled, takes the padded-staging repack.
#[test]
fn mip_chain_under_the_256_byte_floor_samples_every_level() {
    let h = non_uma_harness();
    let colors = [RED, GREEN, BLUE, WHITE, MAGENTA, 0xFFFF_FF00];
    for (format, name) in [(D3DFMT_A8R8G8B8, "A8R8G8B8"), (D3DFMT_X8R8G8B8, "X8R8G8B8")] {
        let tex = h.create_texture(32, 32, 0, 0, format, D3DPOOL_MANAGED);
        assert_eq!(tex.level_count(), 6, "{name} 32x32 full mip chain");
        for (level, color) in colors.iter().enumerate() {
            let side = 32usize >> level;
            let level = u32::try_from(level).expect("mip index fits u32");
            fill_level(&tex, level, side, *color);
        }
        for (level, color) in colors.iter().enumerate() {
            let level = u32::try_from(level).expect("mip index fits u32");
            assert_eq!(
                sample_level(&h, &tex, level),
                *color,
                "{name} mip {level} carries its own texels"
            );
        }
    }
}

/// A lockable render target whose rows are under the floor takes a CPU write and reads it back.
///
/// 16x16 A8R8G8B8 rows are 64 bytes, so the unlock upload pads them to the
/// 256-byte stride before the blit into the colour texture, and the
/// `GetRenderTargetData` readback proves the texels landed.
#[test]
fn lockable_render_target_rows_pad_to_the_floor() {
    let h = non_uma_harness();
    let rt = h.create_lockable_render_target(16, 16, D3DFMT_A8R8G8B8);
    {
        let mut locked = rt.lock_rect(0);
        let mut texels = vec![GREEN; 16 * 16];
        texels[16 * 15 + 15] = RED;
        locked.write_u32_rect(16, 16, &texels);
    }
    let sysmem = h.create_offscreen_plain_surface(16, 16, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(&rt, &sysmem),
        0,
        "GetRenderTargetData from the lockable target"
    );
    let locked = sysmem.lock_rect(D3DLOCK_READONLY);
    let pitch_px = locked.pitch().cast_unsigned() / 4;
    let px = locked.as_u32((pitch_px * 16) as usize);
    let at = |x: u32, y: u32| px[(y * pitch_px + x) as usize];
    assert_eq!(at(0, 0), GREEN, "the first texel of the upload landed");
    assert_eq!(at(8, 8), GREEN, "the middle of the upload landed");
    assert_eq!(at(15, 15), RED, "the last texel of the last row landed");
}
