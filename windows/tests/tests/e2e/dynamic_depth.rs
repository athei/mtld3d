//! Packed dynamic depth uploads and deferred GPU-authoritative locks.

use mtld3d_tests::{
    Harness, HarnessConfig, LockedRect, PosColorVertex, Reading, Texture, VolumeVertex,
    assert_or_reread, assert_pixel_eq,
};
use mtld3d_types::{
    D3DCLEAR_STENCIL, D3DCLEAR_ZBUFFER, D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A8R8G8B8,
    D3DFMT_D16, D3DFMT_D24S8, D3DFMT_D24X8, D3DFMT_D32, D3DFMT_INTZ, D3DFMT_X8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DLOCK_DISCARD, D3DLOCK_NO_DIRTY_UPDATE,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DRS_POINTSIZE, D3DRS_ZENABLE, D3DRTYPE_TEXTURE,
    D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTEXF_POINT,
    D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC,
};

#[test]
fn accepted_formats_have_packed_lockable_mips() {
    let h = Harness::new();
    for format in [D3DFMT_D16, D3DFMT_D24X8, D3DFMT_D24S8] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, D3DUSAGE_DYNAMIC, D3DRTYPE_TEXTURE, format),
            0
        );
        let t = h.create_texture(17, 9, 0, D3DUSAGE_DYNAMIC, format, D3DPOOL_DEFAULT);
        assert_eq!(t.level_count(), 5);
        for level in 0..5 {
            let (hr, desc) = t.level_desc(level);
            assert_eq!(hr, 0);
            let (w, height) = ((17 >> level).max(1), (9 >> level).max(1));
            assert_eq!(
                (desc.width, desc.height, desc.format, desc.usage),
                (w, height, format, D3DUSAGE_DYNAMIC)
            );
            write_constant(&t, level, format, w, height, 0);
            let locked = t.lock_rect(level, D3DLOCK_READONLY);
            let bpp = if format == D3DFMT_D16 { 2 } else { 4 };
            assert!(locked.pitch() >= i32::try_from(w * bpp).unwrap());
            drop(locked);
            let surface = t.surface_level(level);
            assert_eq!(surface.lock_rect_probe(D3DLOCK_READONLY).0, 0);
            assert_eq!(surface.unlock_rect(), 0);
            assert_eq!(h.set_depth_stencil_surface(&surface), D3DERR_INVALIDCALL);
        }
        for pool in [D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
            let (hr, ptr) = h.try_create_texture(17, 9, 1, D3DUSAGE_DYNAMIC, format, pool);
            assert_eq!(hr, D3DERR_INVALIDCALL);
            assert!(ptr.is_null());
        }
        let (hr, ptr) = h.try_create_texture(
            17,
            9,
            1,
            D3DUSAGE_DYNAMIC | D3DUSAGE_DEPTHSTENCIL,
            format,
            D3DPOOL_DEFAULT,
        );
        assert_eq!(hr, D3DERR_INVALIDCALL);
        assert!(ptr.is_null());
    }
    for format in [D3DFMT_D32, D3DFMT_INTZ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, D3DUSAGE_DYNAMIC, D3DRTYPE_TEXTURE, format),
            D3DERR_NOTAVAILABLE
        );
    }
}

#[test]
fn uploaded_depth_samples_each_packed_format_and_tail_mip() {
    let h = Harness::new();
    setup_sample(&h);
    for format in [D3DFMT_D16, D3DFMT_D24X8, D3DFMT_D24S8] {
        let t = h.create_texture(17, 9, 0, D3DUSAGE_DYNAMIC, format, D3DPOOL_DEFAULT);
        for level in 0..5 {
            let (w, height) = ((17 >> level).max(1), (9 >> level).max(1));
            write_constant(&t, level, format, w, height, 0);
            assert_eq!(h.set_texture(0, &t), 0);
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
            draw_sample(&h, -1.0, 0.0);
            write_constant(&t, level, format, w, height, D3DLOCK_DISCARD);
            assert_eq!(h.set_texture(0, &t), 0);
            draw_sample(&h, 0.0, 1.0);
            assert_eq!(h.present(), 0);
            assert_pixel_eq(
                h.read_pixel(160, 240),
                0x0000_0000,
                "old depth upload stays near",
            );
            assert_pixel_eq(
                h.read_pixel(480, 240),
                0xffff_ffff,
                "discard depth upload becomes far",
            );
        }
    }
}

#[test]
fn depth_only_resz_reads_back_uploaded_stencil_and_preserves_no_dirty_update() {
    let h = Harness::new();
    let target = h.create_render_target(16, 16, D3DFMT_A8R8G8B8);
    let source = h.create_depth_stencil_surface(16, 16, D3DFMT_D16);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    let t = h.create_texture(16, 16, 1, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    let packed: Vec<_> = (0..256u32).map(|n| 0x2000_0000 | n).collect();
    t.lock_rect(0, 0).write_u32_rect(16, 16, &packed);
    assert_eq!(h.set_texture(0, &t), 0);
    setup_sample(&h);
    draw_sample(&h, -1.0, 1.0);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    // Staging changes alone must not replace the already queued GPU stencil.
    t.lock_rect(0, D3DLOCK_NO_DIRTY_UPDATE)
        .write_u32_rect(16, 16, &[0; 256]);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, 0, 0.75, 0), 0);
    resolve(&h, &t);
    let locked = t.lock_rect(0, D3DLOCK_READONLY);
    let pitch = usize::try_from(locked.pitch()).unwrap() / 4;
    let words = locked.as_u32(pitch * 15 + 16);
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(
                words[y * pitch + x],
                0xbfff_ff00 | u32::try_from(y * 16 + x).unwrap(),
                "GPU depth and uploaded stencil at {x},{y}"
            );
        }
    }
}

#[test]
fn resz_partial_lock_and_discard_preserve_other_mips() {
    let h = Harness::new();
    let target = h.create_render_target(16, 16, D3DFMT_A8R8G8B8);
    let source = h.create_depth_stencil_surface(16, 16, D3DFMT_D24S8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    let t = h.create_texture(16, 16, 5, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    t.lock_rect(4, 0).write_u32(&[0x1234_5678]);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 0.25, 0xa7),
        0
    );
    resolve(&h, &t);
    t.lock_rect_partial(0, &[3, 2, 5, 4], 0)
        .write_u32_rect(2, 2, &[0x8000_0033; 4]);
    {
        let locked = t.lock_rect(0, D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).unwrap() / 4;
        let words = locked.as_u32(pitch * 15 + 16);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    words[y * pitch + x],
                    if (3..5).contains(&x) && (2..4).contains(&y) {
                        0x8000_0033
                    } else {
                        0x4000_00a7
                    }
                );
            }
        }
    }
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 0.75, 0x5c),
        0
    );
    resolve(&h, &t);
    t.lock_rect(0, D3DLOCK_DISCARD)
        .write_u32_rect(16, 16, &[0xc000_0049; 256]);
    assert_eq!(h.set_texture(0, &t), 0);
    assert_eq!(h.present(), 0);
    assert_eq!(t.lock_rect(4, D3DLOCK_READONLY).as_u32(1)[0], 0x1234_5678);
    assert_eq!(t.lock_rect(0, D3DLOCK_READONLY).as_u32(1)[0], 0xc000_0049);
}

#[test]
fn multisample_resz_reads_sample_zero_depth_and_stencil() {
    use mtld3d_types::{
        D3DCMP_ALWAYS, D3DMULTISAMPLE_4_SAMPLES, D3DRS_MULTISAMPLEMASK, D3DRS_STENCILENABLE,
        D3DRS_STENCILFUNC, D3DRS_STENCILPASS, D3DRS_STENCILREF, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
        D3DSTENCILOP_REPLACE,
    };
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        multi_sample_type: D3DMULTISAMPLE_4_SAMPLES,
        ..HarnessConfig::default()
    });
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    for (state, value) in [
        (D3DRS_LIGHTING, 0),
        (D3DRS_ZENABLE, 1),
        (D3DRS_ZWRITEENABLE, 1),
        (D3DRS_ZFUNC, D3DCMP_ALWAYS),
        (D3DRS_STENCILENABLE, 1),
        (D3DRS_STENCILFUNC, D3DCMP_ALWAYS),
        (D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE),
    ] {
        assert_eq!(h.set_render_state(state, value), 0);
    }
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 0), 0);
    for (sample, (z, stencil)) in [(0.5, 75), (0.25, 11), (0.75, 239), (0.125, 91)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(h.set_render_state(D3DRS_MULTISAMPLEMASK, 1 << sample), 0);
        assert_eq!(h.set_render_state(D3DRS_STENCILREF, stencil), 0);
        let v = |x, y| PosColorVertex {
            x,
            y,
            z,
            color: 0xffff_ffff,
        };
        assert_eq!(
            h.draw_primitive_up(
                D3DPT_TRIANGLELIST,
                1,
                &[v(-1.0, -1.0), v(-1.0, 3.0), v(3.0, -1.0)]
            ),
            0
        );
    }
    let t = h.create_texture(640, 480, 1, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    resolve(&h, &t);
    let locked = t.lock_rect(0, D3DLOCK_READONLY);
    let (h, t) = (&h, &t);
    assert_or_reread(
        h,
        "RESZ out of a 4x depth surface reads sample zero",
        "0x8000004b (depth 0x800000, stencil 0x4b) at every probe",
        &sample_zero_words(&locked),
        || sampled_gpu_depth(h, t),
        move || {
            // The level cannot be locked twice, and the write below claims it
            // for the GPU again, so this lock reads the texture back as well.
            drop(locked);
            resolve(h, t);
            sample_zero_words(&t.lock_rect(0, D3DLOCK_READONLY))
        },
    );
}

/// The four probes of a locked 640x480 D24S8 level against sample zero's depth and stencil.
fn sample_zero_words(locked: &LockedRect<'_>) -> Reading {
    const PROBES: [(usize, usize); 4] = [(160, 120), (480, 120), (160, 360), (480, 360)];
    let pitch = usize::try_from(locked.pitch()).unwrap() / 4;
    let words = locked.as_u32(pitch * 479 + 640);
    let probes = PROBES.map(|(x, y)| words[y * pitch + x]);
    let zero = (0..480)
        .flat_map(|y| &words[y * pitch..y * pitch + 640])
        .filter(|&&word| word == 0)
        .count();
    let shown = PROBES
        .iter()
        .zip(probes)
        .map(|(&(x, y), word)| {
            format!(
                "{x},{y}: {word:#010x} (depth {:#08x}, stencil {:#04x})",
                word >> 8,
                word & 0xff
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    Reading::described(
        format!("{shown}; {zero} of 307200 words are zero"),
        probes == [0x8000_004b; 4],
    )
}

/// Observe the GPU copy of `t`'s level 0 through a draw, while its first lock is still held.
///
/// A second lock cannot do it: the first `READONLY` lock of a level a RESZ
/// write claimed reads the texture back and hands authority to the staging,
/// so the next lock returns that staging and never reaches the GPU. A
/// `READONLY` lock publishes nothing, so a draw binds the Metal texture as the
/// transfer left it and uploads nothing over it.
///
/// Two draws compare a reference depth against the texture's centre on a
/// single-sampled target cleared to blue, with no depth surface bound, so no
/// multisampled pass is involved. The comparison passes, and draws white,
/// where the reference is at most the stored depth. The left band uses 0.375
/// and the middle band 0.625, so white then transparent black brackets the
/// stored depth between the two, which sample zero's 0.5 satisfies and a zero
/// or a cleared 1.0 does not. It says nothing about stencil or about any other
/// texel. Blue in the undrawn right band says the probe's own pass ran.
fn sampled_gpu_depth(h: &Harness, t: &Texture<'_>) -> Reading {
    const BLUE: u32 = 0xff00_00ff;
    const WHITE: u32 = 0xffff_ffff;
    let bands = compared_bands(h, t, &[0.375, 0.625]);
    Reading::described(
        format!(
            "the GPU copy sampled against 0.375, against 0.625, and the undrawn band: {bands:08x?} \
             (0xffffffff = holds at least the reference, 0 = holds less, 0xff0000ff = the \
             probe's clear)"
        ),
        bands == [WHITE, 0, BLUE],
    )
}

/// Comparison-sample the centre of `t`'s level 0 against each reference, one band each.
///
/// The bands split the left three quarters of a single-sampled 16x16 target
/// cleared to blue, drawn with no depth surface bound. A band is white where
/// its reference is at most the stored depth and transparent black where it is
/// more. Returns one pixel per band in order, then one from the undrawn right
/// quarter, which stays blue when the probe's own pass ran. The render target
/// and depth surface bound on entry are bound again on return.
fn compared_bands(h: &Harness, t: &Texture<'_>, references: &[f32]) -> Vec<u32> {
    use mtld3d_types::{D3DRS_MULTISAMPLEMASK, D3DRS_STENCILENABLE};
    let back = h.render_target(0);
    let depth = h.depth_stencil_surface();
    let probe = h.create_render_target(16, 16, D3DFMT_A8R8G8B8);
    assert_eq!(h.set_render_target(0, &probe), 0);
    setup_sample(h);
    assert_eq!(h.set_render_state(D3DRS_STENCILENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_MULTISAMPLEMASK, u32::MAX), 0);
    assert_eq!(h.set_texture(0, t), 0);
    assert_eq!(h.clear_target(0xff00_00ff), 0);
    let count = u8::try_from(references.len()).expect("a handful of bands");
    let edge = |band: u8| -1.0 + 1.5 * f32::from(band) / f32::from(count);
    let mut columns = Vec::new();
    for (band, &reference) in (0..count).zip(references) {
        let (left, right) = (edge(band), edge(band + 1));
        let v = |x, y| VolumeVertex {
            x,
            y,
            z: 0.5,
            color: 0xffff_ffff,
            u: 0.5,
            v: 0.5,
            w: reference,
        };
        let quad = [
            v(left, 1.0),
            v(right, 1.0),
            v(left, -1.0),
            v(right, 1.0),
            v(right, -1.0),
            v(left, -1.0),
        ];
        assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
        columns.push((24 * u32::from(band) + 12) / (2 * u32::from(count)));
    }
    columns.push(14);
    let bands = columns.into_iter().map(|x| h.read_pixel(x, 8)).collect();
    assert_eq!(h.set_render_target(0, &back), 0);
    if let Some(depth) = depth {
        assert_eq!(h.set_depth_stencil_surface(&depth), 0);
    }
    bands
}

#[test]
fn scaled_resz_resamples_common_planes_in_logical_space() {
    use mtld3d_types::D3DRECT;
    for config_entries in ["render.scale=0.75", "render.scale=0.67"] {
        let h = Harness::create(&HarnessConfig {
            depth_format: Some(D3DFMT_D24S8),
            config_entries,
            ..HarnessConfig::default()
        });
        let t = h.create_texture(640, 480, 1, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
        for (x, y, z, stencil) in [
            (0, 0, 0.25, 17),
            (320, 0, 0.5, 83),
            (0, 240, 0.75, 151),
            (320, 240, 1.0, 239),
        ] {
            assert_eq!(
                h.clear_rects(
                    D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,
                    0,
                    z,
                    stencil,
                    &[D3DRECT {
                        x1: x,
                        y1: y,
                        x2: x + 320,
                        y2: y + 240
                    }]
                ),
                0
            );
        }
        resolve(&h, &t);
        let locked = t.lock_rect(0, D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).unwrap() / 4;
        let words = locked.as_u32(pitch * 479 + 640);
        for (x, y, expected) in [
            (160, 120, 0x4000_0011),
            (480, 120, 0x8000_0053),
            (160, 360, 0xbfff_ff97),
            (480, 360, 0xffff_ffef),
        ] {
            assert_eq!(
                words[y * pitch + x],
                expected,
                "scaled common planes {config_entries} at {x},{y}"
            );
        }
    }
}

#[test]
fn discard_after_pending_resz_versions_the_gpu_destination() {
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        ..HarnessConfig::default()
    });
    let t = h.create_texture(640, 480, 1, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 0.25, 0x33),
        0
    );
    resolve(&h, &t);
    // No draw has sampled this destination. The pending GPU write alone must
    // protect its version from this frame-prefix DISCARD upload.
    t.lock_rect(0, D3DLOCK_DISCARD)
        .write_u32_rect(640, 480, &vec![0x8000_0099; 640 * 480]);
    let source = h.create_depth_stencil_surface(640, 480, D3DFMT_D16);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, 0, 0.75, 0), 0);
    resolve(&h, &t);
    assert_eq!(
        t.lock_rect(0, D3DLOCK_READONLY).as_u32(1)[0],
        0xbfff_ff99,
        "second RESZ preserves the new version's GPU stencil"
    );
}

#[test]
fn overlapping_partial_upload_keeps_old_draw_and_retires_after_release() {
    let h = Harness::new();
    setup_sample(&h);
    let t = h.create_texture(16, 16, 1, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    write_constant(&t, 0, D3DFMT_D24S8, 16, 16, 0);
    assert_eq!(h.set_texture(0, &t), 0);
    draw_sample(&h, -1.0, 0.0);
    t.lock_rect_partial(0, &[7, 7, 9, 9], 0)
        .write_u32_rect(2, 2, &[0xc000_0061; 4]);
    draw_sample(&h, 0.0, 1.0);
    assert_eq!(h.clear_texture(0), 0);
    drop(t);
    assert_eq!(h.present(), 0);
    assert_pixel_eq(
        h.read_pixel(160, 240),
        0,
        "old packed upload retained through draw",
    );
    assert_pixel_eq(
        h.read_pixel(480, 240),
        0xffff_ffff,
        "partial update survives early resource release",
    );
}

fn resolve(h: &Harness, texture: &Texture<'_>) {
    assert_eq!(h.set_texture(0, texture), 0);
    assert_eq!(h.set_render_state(D3DRS_POINTSIZE, 0x7fa0_5000), 0);
}

fn write_constant(t: &Texture<'_>, level: u32, format: u32, width: u32, height: u32, flags: u32) {
    let far = flags & D3DLOCK_DISCARD != 0;
    let bytes = match format {
        D3DFMT_D16 => if far { 0xc000u16 } else { 0x4000u16 }
            .to_le_bytes()
            .to_vec(),
        D3DFMT_D24S8 => if far { 0xc000_00a7u32 } else { 0x4000_005cu32 }
            .to_le_bytes()
            .to_vec(),
        _ => if far { 0xa7c0_0000u32 } else { 0x5c40_0000u32 }
            .to_le_bytes()
            .to_vec(),
    };
    t.lock_rect(level, flags).write_u8_rect(
        width as usize * bytes.len(),
        height as usize,
        &bytes.repeat((width * height) as usize),
    );
}

fn setup_sample(h: &Harness) {
    assert_eq!(h.clear_depth_stencil_surface(), 0);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (1 << 16)),
        0
    );
    h.select_texture_stage(0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
}

fn draw_sample(h: &Harness, left: f32, right: f32) {
    let v = |x, y| VolumeVertex {
        x,
        y,
        z: 0.5,
        color: 0xffff_ffff,
        u: 0.5,
        v: 0.5,
        w: 0.5,
    };
    let quad = [
        v(left, 1.0),
        v(right, 1.0),
        v(left, -1.0),
        v(right, 1.0),
        v(right, -1.0),
        v(left, -1.0),
    ];
    assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
}

#[test]
fn pending_resz_survives_discard_upload_to_another_mip() {
    let h = Harness::new();
    let source = h.create_depth_stencil_surface(16, 16, D3DFMT_D24S8);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    let t = h.create_texture(16, 16, 3, D3DUSAGE_DYNAMIC, D3DFMT_D24S8, D3DPOOL_DEFAULT);
    t.lock_rect(0, D3DLOCK_DISCARD)
        .write_u32_rect(16, 16, &[0x2000_0022; 256]);
    t.lock_rect(2, D3DLOCK_DISCARD)
        .write_u32_rect(4, 4, &[0x6000_0033; 16]);
    assert_eq!(
        h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 0.75, 0xa7),
        0
    );
    resolve(&h, &t);
    t.lock_rect(1, D3DLOCK_DISCARD)
        .write_u32_rect(8, 8, &[0x4000_005c; 64]);
    setup_sample(&h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    draw_sample(&h, -1.0, 0.0);
    t.lock_rect_partial(1, &[3, 3, 5, 5], 0)
        .write_u32_rect(2, 2, &[0xc000_0075; 4]);
    draw_sample(&h, 0.0, 1.0);
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(h.present(), 0);
    assert_pixel_eq(
        h.read_pixel(160, 240),
        0,
        "first ordered mip upload stays near",
    );
    assert_pixel_eq(
        h.read_pixel(480, 240),
        0xffff_ffff,
        "second ordered mip upload becomes far",
    );
    assert_eq!(t.lock_rect(0, D3DLOCK_READONLY).as_u32(1)[0], 0xbfff_ffa7);
    assert_eq!(t.lock_rect(1, D3DLOCK_READONLY).as_u32(1)[0], 0x4000_005c);
    assert_eq!(t.lock_rect(2, D3DLOCK_READONLY).as_u32(1)[0], 0x6000_0033);
}

#[test]
fn resz_readback_packs_every_admitted_format() {
    let h = Harness::new();
    let target = h.create_render_target(17, 9, D3DFMT_A8R8G8B8);
    let source = h.create_depth_stencil_surface(17, 9, D3DFMT_D24S8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    for (format, expected) in [
        (D3DFMT_D16, 0x4000u16.to_le_bytes().to_vec()),
        (D3DFMT_D24X8, 0x0040_0000u32.to_le_bytes().to_vec()),
        (D3DFMT_D24S8, 0x4000_00a7u32.to_le_bytes().to_vec()),
    ] {
        let texture = h.create_texture(17, 9, 1, D3DUSAGE_DYNAMIC, format, D3DPOOL_DEFAULT);
        assert_eq!(
            h.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 0.25, 0xa7),
            0
        );
        resolve(&h, &texture);
        let locked = texture.lock_rect(0, D3DLOCK_READONLY);
        let pitch = usize::try_from(locked.pitch()).unwrap();
        let packed = locked.as_u8(pitch * 8 + 17 * expected.len());
        for y in 0..9 {
            for x in 0..17 {
                let offset = y * pitch + x * expected.len();
                assert_eq!(
                    &packed[offset..offset + expected.len()],
                    &expected,
                    "format {format}, pixel {x},{y}"
                );
            }
        }
    }
}

/// A READONLY lock of a level only the GPU has written publishes nothing.
///
/// The Metal texture keeps depth as a float and a D16 lock hands out 16-bit
/// codes, so an upload of the read-back staging would replace the stored depth
/// with its nearest code. The transfer writes a depth that lies between two
/// codes, and the probe compares against a reference between that depth and
/// the code below it: the band is white while the texture holds what the
/// transfer wrote, and black once the code has been uploaded over it.
#[test]
fn readonly_lock_of_a_resz_written_level_uploads_nothing() {
    const WHITE: u32 = 0xffff_ffff;
    const BLUE: u32 = 0xff00_00ff;
    const STORED: f32 = 32_768.45 / 65_535.0;
    const BETWEEN: f32 = 32_768.225 / 65_535.0;
    let h = Harness::new();
    let target = h.create_render_target(16, 16, D3DFMT_A8R8G8B8);
    let source = h.create_depth_stencil_surface(16, 16, D3DFMT_D24S8);
    assert_eq!(h.set_render_target(0, &target), 0);
    assert_eq!(h.set_depth_stencil_surface(&source), 0);
    assert_eq!(h.clear(D3DCLEAR_ZBUFFER, 0, STORED, 0), 0);
    for lock in [false, true] {
        let t = h.create_texture(16, 16, 1, D3DUSAGE_DYNAMIC, D3DFMT_D16, D3DPOOL_DEFAULT);
        resolve(&h, &t);
        if lock {
            let locked = t.lock_rect(0, D3DLOCK_READONLY);
            assert_eq!(locked.as_u8(2), [0x00, 0x80], "the read-back's D16 code");
        }
        assert_eq!(
            compared_bands(&h, &t, &[0.375, BETWEEN, 0.625]),
            [WHITE, WHITE, 0, BLUE],
            "the transferred depth against 0.375, a value just under it, and 0.625, then the \
             undrawn band (READONLY lock first: {lock})"
        );
    }
}
