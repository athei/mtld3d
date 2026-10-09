//! The CPU staging of a default-pool render-target texture.
//!
//! Such a texture keeps no staging until a CPU path needs a level, and that
//! path reads the level back from the GPU first, so what the CPU sees is what
//! the passes drew. The residency check reads the address-space watch's debug
//! line in a process of its own; the rest read pixels through `LockRect`,
//! `GetDC`, `GetRenderTargetData` and a sampling draw. A level a pass draws
//! into again is read back again at the next CPU read, unless it holds a CPU
//! write no upload has carried yet.

use mtld3d_tests::{Harness, Surface, Texture};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_LIGHTING, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER,
    D3DTA_TEXTURE, D3DTADDRESS_CLAMP, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLOROP, D3DUSAGE_RENDERTARGET,
};

use super::{
    device::{await_logged_lines, run_in_private_log_child, running_as},
    render_target::{draw_fill, read_back, textured_fullscreen_quad},
};

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLACK: u32 = 0xFF00_0000;
/// Green as GDI's `COLORREF` (`0x00BBGGRR`) reports it.
const GREEN_COLORREF: u32 = 0x0000_FF00;

/// The side of the small targets the pixel checks use.
const SIZE: u32 = 64;

/// The executable name the residency test runs its workload under.
const RESIDENCY_CHILD_NAME: &str = "render-target-staging.exe";

/// The child's log filter: the address-space watch's periodic breakdown.
const RESIDENCY_LOG_FILTER: &str = "warn,mtld3d::d3d9::mem_watch=debug";

/// Render-target textures drawn into hold no staging while no CPU path has used them.
///
/// Four 1024x1024 A8R8G8B8 targets are 16 MiB of levels. Each is drawn into,
/// and the first `Present` logs the address-space watch's breakdown, whose
/// texture staging holder then counts none of them. The workload runs in a
/// process of its own, so the line it reads holds its device's textures
/// alone.
#[test]
fn render_target_textures_hold_no_staging_until_a_cpu_path_uses_them() {
    if running_as(RESIDENCY_CHILD_NAME) {
        residency_workload();
        return;
    }
    run_in_private_log_child(
        RESIDENCY_CHILD_NAME,
        "render_target_staging::render_target_textures_hold_no_staging_until_a_cpu_path_uses_them",
        RESIDENCY_LOG_FILTER,
    );
}

/// Draw into four large render-target textures, present once and read the watch's line.
fn residency_workload() {
    const TARGETS: usize = 4;
    const SIDE: u32 = 1024;
    let h = Harness::new();
    let back = h.render_target(0);
    let targets: Vec<Texture<'_>> = (0..TARGETS)
        .map(|_| create_target(&h, SIDE, SIDE))
        .collect();
    for target in &targets {
        assert_eq!(
            h.set_render_target(0, &target.surface_level(0)),
            D3D_OK,
            "bind a large target"
        );
        draw_fill(&h, GREEN);
    }
    assert_eq!(
        h.set_render_target(0, &back),
        D3D_OK,
        "restore the back buffer"
    );
    assert_eq!(h.present(), D3D_OK, "the first Present samples the watch");
    let lines = await_logged_lines("address space:", 1);
    let line = &lines[0];
    let staging = texture_staging_mib(line);
    assert!(
        staging < 8,
        "the texture staging holder counts {staging} MiB with 16 MiB of drawn render \
         targets alive: {line}"
    );
    assert!(
        line.contains("render target 0 /"),
        "the staging split names no render-target staging: {line}"
    );
}

/// The texture staging holder of an address-space watch line, in MiB.
fn texture_staging_mib(line: &str) -> u32 {
    let (_, holders) = line
        .split_once("page boxes ")
        .expect("the line names the page boxes");
    let (_, staging) = holders
        .split_once("texture staging ")
        .expect("the page boxes name texture staging");
    staging
        .split(',')
        .next()
        .and_then(|value| value.trim().parse().ok())
        .expect("texture staging is a number of MiB")
}

/// The first `LockRect` of a level the GPU drew returns the drawn pixels.
///
/// Checked on a small target and on one at the back buffer's size, which a
/// `render.scale` leg rasterizes smaller and resolves back up for the read.
#[test]
fn the_first_lock_of_a_drawn_render_target_texture_reads_the_drawn_pixels() {
    let h = Harness::new();
    let back = h.render_target(0);
    let (width, height) = h.dims();
    for (w, ht) in [(SIZE, SIZE), (width, height)] {
        let target = create_target(&h, w, ht);
        assert_eq!(
            h.set_render_target(0, &target.surface_level(0)),
            D3D_OK,
            "bind the {w}x{ht} target"
        );
        draw_fill(&h, GREEN);
        assert_eq!(
            h.set_render_target(0, &back),
            D3D_OK,
            "restore the back buffer"
        );
        let texels = locked_texels(&target, w, ht);
        for (x, y, what) in [
            (0, 0, "first texel"),
            (w / 2, ht / 2, "centre texel"),
            (w - 1, ht - 1, "last texel"),
        ] {
            assert_eq!(
                texels[(y * w + x) as usize],
                GREEN,
                "{w}x{ht} target, {what}: the lock reads the draw"
            );
        }
    }
}

/// A render-target texture nothing drew into locks as zeros.
///
/// D3D9 leaves such a target's contents undefined; the layer answers zero,
/// what a target reads before anything draws into it.
#[test]
fn a_never_drawn_render_target_texture_locks_as_zeros() {
    let h = Harness::new();
    let target = create_target(&h, SIZE, SIZE);
    let texels = locked_texels(&target, SIZE, SIZE);
    assert!(
        texels.iter().all(|&texel| texel == 0),
        "every texel of a never-drawn target reads zero"
    );
}

/// A partial `UpdateSurface` into a drawn target keeps the pixels it leaves alone.
///
/// The update writes the top-left 16x16 texels; the rest keep the draw, in
/// the staging a lock reads and in the texture a draw samples.
#[test]
fn a_partial_update_surface_into_a_drawn_render_target_texture_keeps_the_rest() {
    const REGION: u32 = 16;
    let h = Harness::new();
    let back = h.render_target(0);
    let target = create_target(&h, SIZE, SIZE);
    let level = target.surface_level(0);
    assert_eq!(h.set_render_target(0, &level), D3D_OK, "bind the target");
    draw_fill(&h, RED);
    assert_eq!(
        h.set_render_target(0, &back),
        D3D_OK,
        "restore the back buffer"
    );

    let source = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u32(&[GREEN; (SIZE * SIZE) as usize]);
    let region = D3DRECT {
        x1: 0,
        y1: 0,
        x2: REGION.cast_signed(),
        y2: REGION.cast_signed(),
    };
    assert_eq!(
        h.update_surface_region_hr(&source, &region, &level, (0, 0)),
        D3D_OK,
        "UpdateSurface of the top-left corner"
    );

    let texels = locked_texels(&target, SIZE, SIZE);
    for (x, y, expected, what) in [
        (4, 4, GREEN, "inside the update"),
        (REGION - 1, REGION - 1, GREEN, "the update's last texel"),
        (REGION, 4, RED, "right of the update"),
        (40, 40, RED, "far from the update"),
    ] {
        assert_eq!(
            texels[(y * SIZE + x) as usize],
            expected,
            "lock, texel ({x}, {y}) {what}"
        );
    }

    sample_to_back_buffer(&h, &target);
    let (width, height) = h.dims();
    // The centre of texel (x, y) on the stretched quad.
    let pixel = |x: u32, y: u32| {
        h.read_pixel(
            (2 * x + 1) * width / (2 * SIZE),
            (2 * y + 1) * height / (2 * SIZE),
        )
    };
    assert_eq!(pixel(4, 4), GREEN, "sampled inside the update");
    assert_eq!(pixel(40, 40), RED, "sampled far from the update");
}

/// `GetDC` on a level the GPU drew reads the drawn pixels.
///
/// What the DC hands back at its release keeps them on the GPU too.
#[test]
fn get_dc_on_a_drawn_render_target_texture_reads_the_drawn_pixels() {
    let h = Harness::new();
    let back = h.render_target(0);
    let target = create_target(&h, SIZE, SIZE);
    let level = target.surface_level(0);
    assert_eq!(h.set_render_target(0, &level), D3D_OK, "bind the target");
    draw_fill(&h, GREEN);
    assert_eq!(
        h.set_render_target(0, &back),
        D3D_OK,
        "restore the back buffer"
    );
    let dc = level.dc();
    let last = (SIZE - 1).cast_signed();
    for (x, y, what) in [(0, 0, "first texel"), (last, last, "last texel")] {
        assert_eq!(
            dc.get_pixel(x, y),
            GREEN_COLORREF,
            "the DC reads the draw's {what}"
        );
    }
    assert_eq!(dc.release(), D3D_OK, "ReleaseDC");
    let pixels = read_back(&h, &level, (SIZE, SIZE), D3DFMT_A8R8G8B8);
    assert_eq!(
        pixels[0], GREEN,
        "the target still holds the draw after ReleaseDC"
    );
}

/// A second `LockRect` after more draws returns the new pixels.
///
/// The first lock leaves the level with staging of its own; the draws after
/// it write the Metal texture alone, so the second lock reads the level back
/// again rather than handing out what the first one read.
#[test]
fn a_lock_after_more_draws_reads_the_new_pixels() {
    let h = Harness::new();
    let back = h.render_target(0);
    let target = create_target(&h, SIZE, SIZE);
    let level = target.surface_level(0);
    for color in [GREEN, RED, GREEN] {
        draw_into(&h, &level, &back, color);
        let texels = locked_texels(&target, SIZE, SIZE);
        assert_eq!(
            texels[(SIZE * SIZE / 2 + SIZE / 2) as usize],
            color,
            "the lock after the {color:#010x} draw reads it"
        );
    }
}

/// A lock after an `UpdateSurface` with no draw between them reads the update.
///
/// The update lands in the staging and uploads at the next bind, so the
/// level's Metal texture still holds the draw from before it. The lock reads
/// the staging, which holds the newer pixels, rather than reading the older
/// ones back over them. The bind after it uploads the update as it would have.
#[test]
fn a_lock_after_update_surface_before_any_draw_reads_the_update() {
    let h = Harness::new();
    let back = h.render_target(0);
    let target = create_target(&h, SIZE, SIZE);
    let level = target.surface_level(0);
    draw_into(&h, &level, &back, RED);
    assert_eq!(
        locked_texels(&target, SIZE, SIZE)[0],
        RED,
        "the lock before the update reads the draw"
    );

    let source = h.create_offscreen_plain_surface(SIZE, SIZE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    source
        .lock_rect(0)
        .write_u32(&[GREEN; (SIZE * SIZE) as usize]);
    let whole = D3DRECT {
        x1: 0,
        y1: 0,
        x2: SIZE.cast_signed(),
        y2: SIZE.cast_signed(),
    };
    assert_eq!(
        h.update_surface_region_hr(&source, &whole, &level, (0, 0)),
        D3D_OK,
        "UpdateSurface of the whole level"
    );
    let texels = locked_texels(&target, SIZE, SIZE);
    assert!(
        texels.iter().all(|&texel| texel == GREEN),
        "the lock after the update reads the update"
    );

    sample_to_back_buffer(&h, &target);
    let (width, height) = h.dims();
    assert_eq!(
        h.read_pixel(width / 2, height / 2),
        GREEN,
        "the bind after the lock uploads the update"
    );
}

/// `LockRect` on a render-target texture level's surface is refused, as D3D9 refuses it.
///
/// Only the texture's own `LockRect` maps such a level, a kept divergence;
/// the surface's entry point keeps D3D9's answer.
#[test]
fn a_render_target_texture_level_surface_refuses_lock_rect() {
    let h = Harness::new();
    let target = create_target(&h, SIZE, SIZE);
    let (hr, _) = target.surface_level(0).lock_rect_probe(D3DLOCK_READONLY);
    assert_eq!(
        hr, D3DERR_INVALIDCALL,
        "LockRect on the level's surface is refused"
    );
}

/// Bind `target` as render target 0, fill it with `color` by a draw, and restore `back`.
fn draw_into(h: &Harness, target: &Surface<'_>, back: &Surface<'_>, color: u32) {
    assert_eq!(h.set_render_target(0, target), D3D_OK, "bind the target");
    draw_fill(h, color);
    assert_eq!(
        h.set_render_target(0, back),
        D3D_OK,
        "restore the back buffer"
    );
}

/// A single-level A8R8G8B8 default-pool render-target texture.
fn create_target(h: &Harness, width: u32, height: u32) -> Texture<'_> {
    h.create_texture(
        width,
        height,
        1,
        D3DUSAGE_RENDERTARGET,
        D3DFMT_A8R8G8B8,
        D3DPOOL_DEFAULT,
    )
}

/// Every texel of level 0, row after row, through a read-only `LockRect`.
fn locked_texels(target: &Texture<'_>, width: u32, height: u32) -> Vec<u32> {
    let locked = target.lock_rect(0, D3DLOCK_READONLY);
    let pitch = locked.pitch().cast_unsigned() / 4;
    let words = locked.as_u32((pitch * height) as usize);
    (0..height as usize)
        .flat_map(|y| {
            words[y * pitch as usize..][..width as usize]
                .iter()
                .copied()
        })
        .collect()
}

/// Draw `texture` over the whole back buffer, point-sampled, and present.
fn sample_to_back_buffer(h: &Harness, texture: &Texture<'_>) {
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), D3D_OK, "TSS");
    }
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        D3D_OK,
        "lighting off"
    );
    assert_eq!(h.set_texture(0, texture), D3D_OK, "bind the target");
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), D3D_OK, "sampler");
    }
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1),
        D3D_OK,
        "SetFVF TEX1"
    );
    let quad = textured_fullscreen_quad();
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
            D3D_OK,
            "sample the target"
        );
    });
    assert_eq!(h.clear_texture(0), D3D_OK, "unbind the target");
}
