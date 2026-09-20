//! Native packed ten-bit textures and their two-bit alpha lane.

use mtld3d_tests::{
    Harness, LockedRect, Texture, TexturedVertex, VolumeVertex, assert_pixel_approx,
};
use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_A2B10G10R10, D3DFMT_A2R10G10B10,
    D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8, D3DFMT_G16R16, D3DFMT_R32F, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE,
    D3DFVF_TEX1, D3DFVF_TEXTUREFORMAT3, D3DFVF_XYZ, D3DLOCK_DISCARD, D3DLOCK_READONLY,
    D3DOK_NOAUTOGEN, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM,
    D3DPT_TRIANGLELIST, D3DRECT, D3DRTYPE_CUBETEXTURE, D3DRTYPE_SURFACE, D3DRTYPE_TEXTURE,
    D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER,
    D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP,
    D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_POINT, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_DEPTHSTENCIL,
    D3DUSAGE_DYNAMIC, D3DUSAGE_QUERY_FILTER, D3DUSAGE_QUERY_LEGACYBUMPMAP,
    D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE,
    D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_QUERY_WRAPANDMIP, D3DUSAGE_RENDERTARGET,
};

/// Each colour lane and the alpha bits sit where the format names them, all ten bits kept.
pub fn channels_alpha_and_lower_rgb_bits(format: u32) {
    let h = Harness::new();
    let tex = h.create_texture(1, 1, 1, 0, format, D3DPOOL_MANAGED);
    for alpha in 0..4 {
        for (r, g, b) in [(5, 341, 1022), (1022, 341, 5)] {
            let raw = packed(format, r, g, b, alpha);
            tex.lock_rect(0, 0).write_u32(&[raw]);
            assert_pixel_approx(
                sample(&h, &tex, [0.5; 2]),
                pixel(format, raw),
                1,
                "RGBA lanes",
            );
            let ps = h.create_pixel_shader(&sample_shader(0xff));
            assert_eq!(h.set_pixel_shader(&ps), 0);
            assert_eq!(h.set_pixel_shader_constant_f(0, &[1.0; 4]), 0);
            assert_eq!(h.set_pixel_shader_constant_f(1, &[0.0; 4]), 0);
            assert_pixel_approx(
                sample(&h, &tex, [0.5; 2]),
                (alpha * 85) * 0x0101_0101,
                1,
                "alpha replication",
            );
            assert_eq!(h.clear_pixel_shader(), 0);
        }
    }
    for (swizzle, shift) in [
        (0x00, shifts(format)[0]),
        (0x55, 10),
        (0xaa, shifts(format)[2]),
    ] {
        let ps = h.create_pixel_shader(&sample_shader(swizzle));
        assert_eq!(h.set_pixel_shader(&ps), 0);
        assert_eq!(h.set_pixel_shader_constant_f(0, &[255.75; 4]), 0);
        for (base, offset) in [(0, 0.0), (512, -128.0), (1020, -255.0)] {
            assert_eq!(h.set_pixel_shader_constant_f(1, &[offset; 4]), 0);
            for low in 0..4 {
                tex.lock_rect(0, 0).write_u32(&[(base + low) << shift]);
                let code = [0, 64, 128, 191][low as usize];
                assert_pixel_approx(
                    sample(&h, &tex, [0.5; 2]),
                    code * 0x0101_0101,
                    1,
                    "amplified low ten-bit precision",
                );
            }
        }
    }
}

/// Locks expose the native words in every pool, and copies and updates move them unchanged.
pub fn native_words_pools_mips_and_updates(format: u32) {
    let h = Harness::new();
    let values = [
        packed(format, 5, 341, 1022, 0),
        packed(format, 1022, 341, 5, 1),
        packed(format, 513, 514, 515, 2),
        packed(format, 1, 2, 3, 3),
        packed(format, 1020, 1021, 1023, 1),
        packed(format, 17, 42, 65, 2),
    ];
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
        let tex = h.create_texture(3, 2, 0, usage, format, pool);
        assert_eq!(tex.level_count(), 2);
        let flags = if pool == D3DPOOL_DEFAULT {
            D3DLOCK_DISCARD
        } else {
            0
        };
        tex.lock_rect(0, flags).write_u32_rect(3, 2, &values);
        assert_words(&tex.lock_rect(0, D3DLOCK_READONLY), 3, &values);
        tex.lock_rect_partial(0, &[1, 1, 2, 2], 0)
            .write_u32(&[values[0]]);
        let mut expected = values;
        expected[4] = values[0];
        assert_words(&tex.lock_rect(0, D3DLOCK_READONLY), 3, &expected);
        tex.lock_rect(1, 0).write_u32(&[values[2]]);
        assert_words(&tex.lock_rect(1, D3DLOCK_READONLY), 1, &[values[2]]);
        if matches!(pool, D3DPOOL_DEFAULT | D3DPOOL_MANAGED) {
            assert_pixel_approx(
                sample(&h, &tex, [5.0 / 6.0, 0.25]),
                pixel(format, values[2]),
                1,
                "native upload",
            );
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
            assert_pixel_approx(
                sample(&h, &tex, [0.5; 2]),
                pixel(format, values[2]),
                1,
                "explicit mip",
            );
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        }
    }
    let raw_source = h.create_offscreen_plain_surface(3, 2, format, D3DPOOL_DEFAULT);
    let raw_destination = h.create_offscreen_plain_surface(3, 2, format, D3DPOOL_DEFAULT);
    raw_source.lock_rect(0).write_u32_rect(3, 2, &values);
    assert_eq!(
        h.stretch_rect(&raw_source, &raw_destination, D3DTEXF_NONE),
        0
    );
    assert_words(&raw_destination.lock_rect(D3DLOCK_READONLY), 3, &values);
    let src = h.create_texture(3, 2, 1, 0, format, D3DPOOL_SYSTEMMEM);
    let dst = h.create_texture(3, 2, 1, 0, format, D3DPOOL_DEFAULT);
    src.lock_rect(0, 0).write_u32_rect(3, 2, &values);
    assert_eq!(h.update_texture_hr(&src, &dst), 0);
    assert_pixel_approx(
        sample(&h, &dst, [0.5, 0.25]),
        pixel(format, values[1]),
        1,
        "whole native update",
    );
    src.lock_rect_partial(0, &[0, 0, 1, 1], 0)
        .write_u32(&[values[2]]);
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
            (1, 1)
        ),
        0
    );
    assert_pixel_approx(
        sample(&h, &dst, [0.5, 0.75]),
        pixel(format, values[2]),
        1,
        "partial native update",
    );
    assert_pixel_approx(
        sample(&h, &dst, [1.0 / 6.0, 0.25]),
        pixel(format, values[0]),
        1,
        "untouched native update",
    );
}

/// Queries agree with creation, with one level and the usage kept for AUTOGEN requests.
pub fn queries_and_noautogen(format: u32) {
    let h = Harness::new();
    for kind in [
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
            D3DUSAGE_QUERY_WRAPANDMIP,
        ] {
            assert_eq!(
                h.check_device_format(D3DFMT_X8R8G8B8, usage, kind, format),
                0,
                "kind={kind} usage={usage:#x}"
            );
        }
        for unsupported in [
            D3DUSAGE_RENDERTARGET,
            D3DUSAGE_DEPTHSTENCIL,
            D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
            D3DUSAGE_QUERY_SRGBREAD,
            D3DUSAGE_QUERY_SRGBWRITE,
            D3DUSAGE_QUERY_LEGACYBUMPMAP,
        ] {
            for extra in [0, D3DUSAGE_AUTOGENMIPMAP, D3DUSAGE_QUERY_FILTER] {
                assert_eq!(
                    h.check_device_format(D3DFMT_X8R8G8B8, unsupported | extra, kind, format),
                    D3DERR_NOTAVAILABLE
                );
            }
        }
        let expected = if matches!(kind, D3DRTYPE_TEXTURE | D3DRTYPE_CUBETEXTURE) {
            D3DOK_NOAUTOGEN
        } else {
            D3DERR_NOTAVAILABLE
        };
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, D3DUSAGE_AUTOGENMIPMAP, kind, format),
            expected
        );
    }
    assert_eq!(
        h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_SURFACE, format),
        0
    );
    assert_eq!(
        h.check_device_type(format, D3DFMT_X8R8G8B8, true),
        D3DERR_NOTAVAILABLE
    );
    assert_eq!(h.create_render_target_hr(4, 4, format), D3DERR_INVALIDCALL);
    for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED] {
        for levels in [0, 1] {
            let tex = h.create_texture(4, 4, levels, D3DUSAGE_AUTOGENMIPMAP, format, pool);
            assert_eq!(tex.level_count(), 1);
            assert_eq!(tex.level_desc(0).1.usage, D3DUSAGE_AUTOGENMIPMAP);
            assert_eq!(tex.level_desc(1).0, D3DERR_INVALIDCALL);
            assert_eq!(tex.set_auto_gen_filter_type(D3DTEXF_POINT), 0);
            assert_eq!(tex.auto_gen_filter_type(), D3DTEXF_POINT);
            for raw in [packed(format, 1023, 0, 5, 1), packed(format, 1, 1023, 0, 2)] {
                tex.lock_rect(0, 0).write_u32(&[raw; 16]);
                tex.generate_mip_sub_levels();
                assert_pixel_approx(
                    sample(&h, &tex, [0.5; 2]),
                    pixel(format, raw),
                    1,
                    "NOAUTOGEN publication",
                );
            }
            let cube = h.create_cube_texture_owned(4, levels, D3DUSAGE_AUTOGENMIPMAP, format, pool);
            assert_eq!(cube.level_count(), 1);
            assert_eq!(cube.surface(0, 0).desc().1.usage, D3DUSAGE_AUTOGENMIPMAP);
            assert_eq!(cube.try_surface(0, 1).0, D3DERR_INVALIDCALL);
            let source = h.create_cube_texture_owned(4, 1, 0, format, D3DPOOL_SYSTEMMEM);
            for face in 0..6 {
                source
                    .lock_rect(face, 0, 0)
                    .write_u32(&[packed(format, 1023, 0, 0, 1); 16]);
            }
            if pool == D3DPOOL_DEFAULT {
                assert_eq!(h.update_cube_texture_hr(&source, &cube), 0);
            } else {
                cube.lock_rect(0, 0, 0)
                    .write_u32(&[packed(format, 1023, 0, 0, 1); 16]);
            }
            cube.generate_mip_sub_levels();
            assert_eq!(h.set_cube_texture(0, &cube), 0);
            assert_pixel_approx(
                sample_3d(&h, [1.0, 0.0, 0.0]),
                0x55ff_0000,
                1,
                "NOAUTOGEN cube",
            );
            assert_eq!(h.clear_texture(0), 0);
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
        assert_eq!(
            h.try_create_texture(4, 4, levels, usage, format, pool).0,
            D3DERR_INVALIDCALL
        );
        assert_eq!(
            h.try_create_cube_texture(4, levels, usage, format, pool).0,
            D3DERR_INVALIDCALL
        );
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

/// Cube faces, volume slices and their mips stay apart, and filtering and wrapping are native.
pub fn cube_volume_filter_and_mips(format: u32) {
    let h = Harness::new();
    let directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let values = [
        packed(format, 5, 341, 1022, 0),
        packed(format, 1022, 341, 5, 1),
        packed(format, 512, 513, 514, 2),
        packed(format, 1, 2, 3, 3),
        packed(format, 1023, 0, 0, 1),
        packed(format, 0, 1023, 0, 2),
    ];
    let src = h.create_cube_texture_owned(2, 0, 0, format, D3DPOOL_SYSTEMMEM);
    let dst = h.create_cube_texture_owned(2, 0, 0, format, D3DPOOL_DEFAULT);
    for face in 0..6 {
        src.lock_rect(face, 0, 0)
            .write_u32(&[values[face as usize]; 4]);
        src.lock_rect(face, 1, 0)
            .write_u32(&[values[(face as usize + 1) % 6]]);
    }
    assert_eq!(h.update_cube_texture_hr(&src, &dst), 0);
    assert_eq!(h.set_cube_texture(0, &dst), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    for level in [0, 1] {
        assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
        for (face, direction) in directions.into_iter().enumerate() {
            assert_pixel_approx(
                sample_3d(&h, direction),
                pixel(format, values[(face + level as usize) % 6]),
                1,
                "cube face/mip",
            );
        }
    }
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
    let src = h
        .try_create_volume_texture([3, 2, 2], 0, 0, format, D3DPOOL_SYSTEMMEM)
        .1
        .unwrap();
    let dst = h
        .try_create_volume_texture([3, 2, 2], 0, 0, format, D3DPOOL_DEFAULT)
        .1
        .unwrap();
    let mut words = vec![values[1]; 6];
    words.extend_from_slice(&[values[2]; 6]);
    src.write_u32(0, &words);
    src.write_u32(1, &[values[3]]);
    assert_eq!(h.update_volume_texture_hr(&src, &dst), 0);
    assert_eq!(h.set_volume_texture(0, &dst), 0);
    for (depth, value) in [(0.25, values[1]), (0.75, values[2])] {
        assert_pixel_approx(
            sample_3d(&h, [0.5, 0.5, depth]),
            pixel(format, value),
            1,
            "volume slice",
        );
    }
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 1), 0);
    assert_pixel_approx(
        sample_3d(&h, [0.5; 3]),
        pixel(format, values[3]),
        1,
        "volume mip",
    );
    assert_eq!(h.clear_texture(0), 0);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
    let tex = h.create_texture(2, 1, 1, 0, format, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&[
        packed(format, 0, 1023, 0, 0),
        packed(format, 1023, 0, 1023, 3),
    ]);
    assert_eq!(h.set_texture(0, &tex), 0);
    setup_2d(&h);
    for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
        assert_eq!(h.set_sampler_state(0, state, D3DTEXF_LINEAR), 0);
    }
    draw_sample(&h, [0.5; 2]);
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0x8080_8080,
        1,
        "native linear filter",
    );
    point_clamp(&h);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP),
        0
    );
    draw_sample(&h, [1.25, 0.5]);
    assert_pixel_approx(h.read_pixel(320, 240), 0x0000_ff00, 1, "native wrap");
}

/// `ColorFill` writes the nearest codes, and a lock after a GPU copy reads the copied words.
pub fn colorfill_and_gpu_authority(format: u32) {
    let h = Harness::new();
    let surface = h.create_offscreen_plain_surface(3, 2, format, D3DPOOL_DEFAULT);
    let raw = packed(format, 513, 17, 1022, 2);
    for (channel, ten, alpha) in [
        (0, 0, 0),
        (42, 168, 0),
        (43, 173, 1),
        (63, 253, 1),
        (127, 509, 1),
        (128, 514, 2),
        (212, 850, 2),
        (213, 855, 3),
        (255, 1023, 3),
    ] {
        let color = u32::from_le_bytes([channel; 4]);
        let expected = packed(format, ten, ten, ten, alpha);
        assert_eq!(h.color_fill_hr(&surface, color), 0);
        assert_words(&surface.lock_rect(D3DLOCK_READONLY), 3, &[expected; 6]);
        assert_eq!(
            h.stretch_rect(&surface, &h.back_buffer(0), D3DTEXF_POINT),
            0
        );
        assert_eq!(h.present(), 0);
        assert_pixel_approx(
            h.read_pixel(320, 240),
            pixel(format, expected),
            1,
            "ColorFill native sampled lanes",
        );
    }
    let src = h.create_offscreen_plain_surface(3, 2, format, D3DPOOL_DEFAULT);
    src.lock_rect(0).write_u32(&[raw; 6]);
    assert_eq!(h.stretch_rect(&src, &surface, D3DTEXF_NONE), 0);
    assert_words(&surface.lock_rect(D3DLOCK_READONLY), 3, &[raw; 6]);
    // Restore GPU authority before the partial fill must preserve other texels.
    assert_eq!(h.stretch_rect(&src, &surface, D3DTEXF_NONE), 0);
    assert_eq!(
        h.stretch_rect(&surface, &h.back_buffer(0), D3DTEXF_POINT),
        0
    );
    assert_eq!(h.color_fill_rect_hr(&surface, (1, 1, 2, 2), 0x7f2b_00ff), 0);
    let mut expected = [raw; 6];
    expected[4] = packed(format, 173, 0, 1023, 1);
    assert_eq!(h.present(), 0);
    assert_pixel_approx(
        h.read_pixel(320, 240),
        pixel(format, raw),
        1,
        "queued source before partial fill",
    );
    assert_words(&surface.lock_rect(D3DLOCK_READONLY), 3, &expected);
    let sentinel = core::ptr::dangling_mut::<core::ffi::c_void>();
    assert_eq!(surface.get_dc(sentinel), (D3DERR_INVALIDCALL, sentinel));
    for pool in [D3DPOOL_SYSTEMMEM, D3DPOOL_SCRATCH] {
        let cpu = h.create_offscreen_plain_surface(3, 2, format, pool);
        assert_eq!(h.color_fill_hr(&cpu, 0), D3DERR_INVALIDCALL);
        cpu.lock_rect(0).write_u32(&[raw; 6]);
        assert_words(&cpu.lock_rect(D3DLOCK_READONLY), 3, &[raw; 6]);
        assert_eq!(cpu.get_dc(sentinel), (D3DERR_INVALIDCALL, sentinel));
    }
    let texture = h.create_texture(3, 2, 1, 0, format, D3DPOOL_DEFAULT);
    assert_eq!(
        h.color_fill_hr(&texture.surface_level(0), 0),
        D3DERR_INVALIDCALL
    );
}

/// Copies to and from the other ten-bit order and other formats are rejected with no write.
pub fn rejects_mixed_raw_copies(format: u32) {
    let h = Harness::new();
    let twin = if format == D3DFMT_A2B10G10R10 {
        D3DFMT_A2R10G10B10
    } else {
        D3DFMT_A2B10G10R10
    };
    for other in [
        twin,
        D3DFMT_A8R8G8B8,
        D3DFMT_A8B8G8R8,
        D3DFMT_G16R16,
        D3DFMT_R32F,
    ] {
        for (source, target) in [(format, other), (other, format)] {
            let src = h.create_texture(2, 2, 1, 0, source, D3DPOOL_SYSTEMMEM);
            let dst = h.create_texture(2, 2, 1, D3DUSAGE_DYNAMIC, target, D3DPOOL_DEFAULT);
            src.lock_rect(0, 0).write_u32(&[0x1234_5678; 4]);
            dst.lock_rect(0, 0).write_u32(&[0x8765_4321; 4]);
            assert_eq!(h.update_texture_hr(&src, &dst), D3DERR_INVALIDCALL);
            assert_eq!(
                h.update_surface_region_hr(
                    &src.surface_level(0),
                    &D3DRECT {
                        x1: 0,
                        y1: 0,
                        x2: 2,
                        y2: 2
                    },
                    &dst.surface_level(0),
                    (0, 0)
                ),
                D3DERR_INVALIDCALL
            );
            assert_words(&dst.lock_rect(0, D3DLOCK_READONLY), 2, &[0x8765_4321; 4]);
            let src = h.create_offscreen_plain_surface(2, 2, source, D3DPOOL_DEFAULT);
            let dst = h.create_offscreen_plain_surface(2, 2, target, D3DPOOL_DEFAULT);
            src.lock_rect(0).write_u32(&[0x1234_5678; 4]);
            dst.lock_rect(0).write_u32(&[0x8765_4321; 4]);
            assert_eq!(h.stretch_rect(&src, &dst, D3DTEXF_NONE), D3DERR_INVALIDCALL);
            assert_words(&dst.lock_rect(D3DLOCK_READONLY), 2, &[0x8765_4321; 4]);
        }
    }
}

/// The word of `format` holding ten-bit red, green and blue codes and a two-bit alpha code.
pub const fn packed(format: u32, r: u32, g: u32, b: u32, alpha: u32) -> u32 {
    let shifts = shifts(format);
    (alpha << 30) | (r << shifts[0]) | (g << 10) | (b << shifts[2])
}

/// The A8R8G8B8 pixel a word of `format` samples as, each lane rounded to nearest.
pub fn pixel(format: u32, raw: u32) -> u32 {
    let rgb = shifts(format).map(|shift| (((raw >> shift) & 1023) * 255 + 511) / 1023);
    (((raw >> 30) * 85) << 24) | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
}

/// Assert a whole-subresource lock's pitch and its raw little-endian words.
fn assert_words(lock: &LockedRect<'_>, width: usize, expected: &[u32]) {
    assert_eq!(lock.pitch(), i32::try_from(width * 4).unwrap());
    for (i, word) in expected.iter().enumerate() {
        // SAFETY: each caller holds the initialized whole-subresource lock for
        // exactly expected.len() pixels; its verified pitch is width * 4.
        let actual = unsafe {
            lock.bits_ptr()
                .wrapping_add(i * 4)
                .cast::<u32>()
                .read_unaligned()
        };
        assert_eq!(actual, *word, "raw word {i}");
    }
}

/// `ps_2_0` tokens sampling stage 0 and writing `sample.<swizzle> * c0 + c1`.
fn sample_shader(swizzle: u32) -> Vec<u32> {
    // ps_2_0: sample s0, then apply the caller's scale and offset to chosen lanes.
    vec![
        0xffff_0200,
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
        0x8000_0000 | (swizzle << 16),
        0xa0e4_0000,
        0xa0e4_0001,
        0x0000_ffff,
    ]
}

/// A full-target quad whose every vertex carries `uv`.
fn quad(uv: [f32; 2]) -> [TexturedVertex; 6] {
    [
        (-1.0, 1.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    ]
    .map(|(x, y)| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: 0xffff_ffff,
        u: uv[0],
        v: uv[1],
    })
}

/// Point filtering and CLAMP addressing on stage 0.
fn point_clamp(h: &Harness) {
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0);
    }
}

/// Fixed-function stage 0 selecting the texture, for the textured FVF.
fn setup_2d(h: &Harness) {
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
}

/// Draw the full-target quad sampling at `uv`.
fn draw_sample(h: &Harness, uv: [f32; 2]) {
    h.render_once(0, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(uv)), 0);
    });
}

/// The centre pixel after drawing `tex` sampled at `uv`.
pub fn sample(h: &Harness, tex: &Texture<'_>, uv: [f32; 2]) -> u32 {
    assert_eq!(h.set_texture(0, tex), 0);
    setup_2d(h);
    draw_sample(h, uv);
    h.read_pixel(320, 240)
}

/// The centre pixel after drawing the bound cube or volume texture sampled at `coords`.
fn sample_3d(h: &Harness, coords: [f32; 3]) -> u32 {
    h.select_texture_stage(0);
    point_clamp(h);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    let vertices = quad([0.5; 2]).map(|v| VolumeVertex {
        x: v.x,
        y: v.y,
        z: v.z,
        color: v.color,
        u: coords[0],
        v: coords[1],
        w: coords[2],
    });
    h.render_once(0, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &vertices), 0);
    });
    h.read_pixel(320, 240)
}

/// Bit positions of the red, green and blue lanes in a word of `format`.
const fn shifts(format: u32) -> [u32; 3] {
    if format == D3DFMT_A2B10G10R10 {
        [0, 10, 20]
    } else {
        [20, 10, 0]
    }
}
