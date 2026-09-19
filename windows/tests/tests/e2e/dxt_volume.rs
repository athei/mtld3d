//! Native block-compressed volume storage, publication and sampling.

use mtld3d_tests::{Harness, VolumeVertex, assert_pixel_approx};
use mtld3d_types::{
    D3DFMT_DXT1, D3DFMT_DXT2, D3DFMT_DXT3, D3DFMT_DXT4, D3DFMT_DXT5, D3DFVF_DIFFUSE, D3DFVF_TEX1,
    D3DFVF_TEXTUREFORMAT3, D3DFVF_XYZ, D3DPOOL_MANAGED, D3DPT_TRIANGLELIST, D3DSAMP_MAGFILTER,
    D3DSAMP_MINFILTER, D3DTA_TEXTURE, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAOP,
};

const FORMATS: [u32; 5] = [
    D3DFMT_DXT1,
    D3DFMT_DXT2,
    D3DFMT_DXT3,
    D3DFMT_DXT4,
    D3DFMT_DXT5,
];

fn solid_block(format: u32, color: u16, alpha: u8) -> Vec<u8> {
    let [lo, hi] = color.to_le_bytes();
    let mut block = match format {
        D3DFMT_DXT1 => Vec::new(),
        D3DFMT_DXT2 | D3DFMT_DXT3 => vec![(alpha / 17) * 17; 8],
        D3DFMT_DXT4 | D3DFMT_DXT5 => vec![alpha, alpha, 0, 0, 0, 0, 0, 0],
        other => panic!("unsupported DXT format {other:#x}"),
    };
    block.extend_from_slice(&[lo, hi, lo, hi, 0, 0, 0, 0]);
    block
}

fn setup(h: &Harness) {
    h.select_texture_stage(0);
    assert_eq!(
        h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | (D3DFVF_TEXTUREFORMAT3 << 16)),
        0
    );
    for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
        assert_eq!(h.set_sampler_state(0, state, D3DTEXF_POINT), 0);
    }
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        0
    );
    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
        0
    );
}

fn quad(left: f32, right: f32, coord: [f32; 3]) -> [VolumeVertex; 6] {
    let v = |x, y| VolumeVertex {
        x,
        y,
        z: 0.5,
        color: 0xffff_ffff,
        u: coord[0],
        v: coord[1],
        w: coord[2],
    };
    [
        v(left, 1.0),
        v(right, 1.0),
        v(left, -1.0),
        v(right, 1.0),
        v(right, -1.0),
        v(left, -1.0),
    ]
}

fn sample(h: &Harness, coord: [f32; 3]) -> u32 {
    let quad = quad(-1.0, 1.0, coord);
    h.render_once(0, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad), 0);
    });
    h.read_pixel(320, 240)
}

#[test]
fn dxt_volume_ungated_create_upload_and_sample_slices() {
    let h = Harness::new();
    setup(&h);
    for format in FORMATS {
        let (hr, texture) = h.try_create_volume_texture([8, 4, 2], 1, 0, format, D3DPOOL_MANAGED);
        assert_eq!(hr, 0, "ungated BC volume creation, format={format:#x}");
        let texture = texture.expect("successful BC volume");
        let mut blocks = solid_block(format, 0xf800, 255).repeat(2);
        blocks.extend(solid_block(format, 0x001f, 255).repeat(2));
        texture.write_blocks(0, None, &blocks);
        let (row, slice, readback) = texture.read_blocks(0);
        assert_eq!(row, if format == D3DFMT_DXT1 { 16 } else { 32 });
        assert_eq!(slice, row);
        assert_eq!(readback, blocks);
        assert_eq!(h.set_volume_texture(0, &texture), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.25]),
            0xffff_0000,
            1,
            "first BC slice",
        );
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.75]),
            0xff00_00ff,
            1,
            "second BC slice",
        );
    }
}

#[test]
fn dxt_volume_scratch_block_rows_and_short_mips() {
    use mtld3d_types::{D3DPOOL_SCRATCH, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MIPFILTER};
    let h = Harness::new();
    setup(&h);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT), 0);
    let control = h.create_texture(4, 4, 1, 0, D3DFMT_DXT1, D3DPOOL_MANAGED);
    control
        .lock_rect(0, 0)
        .write_u8_rect(8, 1, &solid_block(D3DFMT_DXT1, 0x07e0, 255));
    assert_eq!(h.set_texture(0, &control), 0);
    assert_pixel_approx(
        sample(&h, [0.5, 0.5, 0.0]),
        0xff00_ff00,
        1,
        "ordinary BC1 control",
    );
    for format in FORMATS {
        for extent in [[4u32, 2048, 1], [8, 2048, 2]] {
            let (hr, texture) = h.try_create_volume_texture(extent, 0, 0, format, D3DPOOL_SCRATCH);
            assert_eq!(hr, 0);
            let texture = texture.expect("scratch compressed volume");
            assert_eq!(texture.level_count(), 12);
            for level in 0..12 {
                let [w, h, d] = extent.map(|n| (n >> level).max(1));
                let color = if level % 2 == 0 { 0xf800 } else { 0x001f };
                let blocks = solid_block(format, color, 255)
                    .repeat((w.div_ceil(4) * h.div_ceil(4) * d) as usize);
                texture.write_blocks(level, None, &blocks);
                let (row, slice, actual) = texture.read_blocks(level);
                let block_bytes = if format == D3DFMT_DXT1 { 8 } else { 16 };
                assert_eq!(
                    u32::try_from(row).expect("positive row pitch"),
                    w.div_ceil(4) * block_bytes
                );
                assert_eq!(
                    u32::try_from(slice).expect("positive slice pitch"),
                    u32::try_from(row).expect("positive row pitch") * h.div_ceil(4)
                );
                assert_eq!(actual, blocks);
            }
            assert_eq!(h.set_volume_texture(0, &texture), 0);
            for level in 0..12 {
                assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, level), 0);
                assert_pixel_approx(
                    sample(&h, [0.5, 0.5, 0.5]),
                    if level % 2 == 0 {
                        0xffff_0000
                    } else {
                        0xff00_00ff
                    },
                    1,
                    "compressed volume explicit mip",
                );
            }
            assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), 0);
        }
    }
}

#[test]
fn dxt_volume_queries_pools_and_exclusions_agree() {
    use mtld3d_types::{
        D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, D3DPOOL_SCRATCH,
        D3DPOOL_SYSTEMMEM, D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE, D3DUSAGE_AUTOGENMIPMAP,
        D3DUSAGE_DEPTHSTENCIL, D3DUSAGE_DYNAMIC, D3DUSAGE_QUERY_FILTER,
        D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, D3DUSAGE_QUERY_SRGBREAD, D3DUSAGE_QUERY_SRGBWRITE,
        D3DUSAGE_QUERY_VERTEXTEXTURE, D3DUSAGE_QUERY_WRAPANDMIP, D3DUSAGE_RENDERTARGET,
    };
    let h = Harness::new();
    for format in FORMATS {
        for kind in [D3DRTYPE_VOLUME, D3DRTYPE_VOLUMETEXTURE] {
            for usage in [
                0,
                D3DUSAGE_DYNAMIC,
                D3DUSAGE_QUERY_FILTER,
                D3DUSAGE_QUERY_SRGBREAD,
                D3DUSAGE_QUERY_VERTEXTEXTURE,
                D3DUSAGE_QUERY_WRAPANDMIP,
                D3DUSAGE_QUERY_FILTER | D3DUSAGE_QUERY_SRGBREAD | D3DUSAGE_QUERY_VERTEXTEXTURE,
            ] {
                assert_eq!(
                    h.check_device_format(D3DFMT_X8R8G8B8, usage, kind, format),
                    0,
                    "format={format:#x} kind={kind} usage={usage:#x}"
                );
            }
            for usage in [
                D3DUSAGE_AUTOGENMIPMAP,
                D3DUSAGE_DEPTHSTENCIL,
                D3DUSAGE_RENDERTARGET,
                D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING,
                D3DUSAGE_QUERY_SRGBWRITE,
                D3DUSAGE_QUERY_SRGBWRITE | D3DUSAGE_QUERY_FILTER,
                D3DUSAGE_QUERY_SRGBWRITE | D3DUSAGE_QUERY_SRGBREAD,
                D3DUSAGE_QUERY_SRGBWRITE | D3DUSAGE_AUTOGENMIPMAP,
            ] {
                assert_eq!(
                    h.check_device_format(D3DFMT_X8R8G8B8, usage, kind, format),
                    D3DERR_NOTAVAILABLE
                );
            }
        }
        for pool in [
            D3DPOOL_DEFAULT,
            D3DPOOL_MANAGED,
            D3DPOOL_SYSTEMMEM,
            D3DPOOL_SCRATCH,
        ] {
            assert_eq!(
                h.create_volume_texture_null_output(format, pool),
                D3DERR_INVALIDCALL
            );
            for usage in [0, D3DUSAGE_DYNAMIC] {
                let (hr, tex) = h.try_create_volume_texture([12, 8, 3], 0, usage, format, pool);
                if usage != 0 && matches!(pool, D3DPOOL_MANAGED | D3DPOOL_SCRATCH) {
                    assert_eq!(hr, D3DERR_INVALIDCALL);
                    assert!(tex.is_none());
                    continue;
                }
                assert_eq!(hr, 0);
                let tex = tex.expect("valid volume");
                assert_eq!(tex.level_count(), 4);
                for level in 0..4 {
                    let (hr, desc) = tex.level_desc(level);
                    assert_eq!(hr, 0);
                    assert_eq!((desc.format, desc.usage, desc.pool), (format, usage, pool));
                    assert_eq!(
                        [desc.width, desc.height, desc.depth],
                        [12u32, 8, 3].map(|v| (v >> level).max(1))
                    );
                }
                let locked = tex.lock_box_probe(0, 0);
                if pool == D3DPOOL_DEFAULT && usage == 0 {
                    assert_eq!(locked, (D3DERR_INVALIDCALL, true));
                } else {
                    assert_eq!(locked, (0, false));
                    assert_eq!(tex.lock_box_probe(0, 0), (D3DERR_INVALIDCALL, true));
                    assert_eq!(tex.unlock_box(0), 0);
                }
                assert_eq!(tex.lock_box_probe(4, 0), (D3DERR_INVALIDCALL, true));
            }
            for usage in [
                D3DUSAGE_AUTOGENMIPMAP,
                D3DUSAGE_DEPTHSTENCIL,
                D3DUSAGE_RENDERTARGET,
            ] {
                let (hr, tex) = h.try_create_volume_texture([4, 4, 2], 1, usage, format, pool);
                assert_eq!(hr, D3DERR_INVALIDCALL);
                assert!(tex.is_none());
            }
            for extent in [[0, 4, 2], [4, 0, 2], [4, 4, 0], [5, 4, 2], [4, 5, 2]] {
                let (hr, tex) = h.try_create_volume_texture(extent, 1, 0, format, pool);
                assert_eq!(hr, D3DERR_INVALIDCALL);
                assert!(tex.is_none());
            }
        }
    }
    for format in [
        mtld3d_types::D3DFMT_ATI1,
        u32::from_le_bytes(*b"ATI2"),
        mtld3d_types::D3DFMT_YUY2,
        mtld3d_types::D3DFMT_UYVY,
    ] {
        assert_eq!(
            h.check_device_format(D3DFMT_X8R8G8B8, 0, D3DRTYPE_VOLUMETEXTURE, format),
            D3DERR_NOTAVAILABLE
        );
        for pool in [D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM] {
            let (hr, tex) = h.try_create_volume_texture([4, 4, 2], 1, 0, format, pool);
            assert_eq!(hr, D3DERR_INVALIDCALL);
            assert!(tex.is_none());
        }
    }
}

#[test]
fn dxt_volume_partial_boxes_preserve_other_blocks_and_publish_update_texture() {
    use mtld3d_types::{D3DBOX, D3DERR_INVALIDCALL, D3DPOOL_DEFAULT, D3DPOOL_SYSTEMMEM};
    let h = Harness::new();
    setup(&h);
    for format in FORMATS {
        let (_, src) = h.try_create_volume_texture([12, 8, 3], 2, 0, format, D3DPOOL_SYSTEMMEM);
        let src = src.expect("system memory volume");
        let red = solid_block(format, 0xf800, 255);
        let blue = solid_block(format, 0x001f, 255);
        let mut expected = red.repeat(18);
        src.write_blocks(0, None, &expected);
        src.write_blocks(1, None, &red.repeat(2));
        let region = D3DBOX {
            left: 4,
            top: 4,
            right: 8,
            bottom: 8,
            front: 1,
            back: 2,
        };
        src.write_blocks(0, Some(&region), &blue);
        expected[10 * red.len()..11 * red.len()].copy_from_slice(&blue);
        assert_eq!(src.read_blocks(0).2, expected);
        assert_eq!(src.read_blocks(1).2, red.repeat(2));
        for bad in [
            D3DBOX { left: 1, ..region },
            D3DBOX { top: 1, ..region },
            D3DBOX { right: 7, ..region },
            D3DBOX {
                bottom: 7,
                ..region
            },
            D3DBOX { back: 4, ..region },
            D3DBOX { right: 4, ..region },
        ] {
            assert_eq!(
                src.lock_box_region_probe(0, Some(&bad), 0),
                (D3DERR_INVALIDCALL, true)
            );
            assert_eq!(src.read_blocks(0).2, expected, "rejection preserves blocks");
        }
        // A short-mip edge need not be a multiple of the four-texel block size.
        let edge = D3DBOX {
            left: 4,
            top: 0,
            right: 6,
            bottom: 4,
            front: 0,
            back: 1,
        };
        src.write_blocks(1, Some(&edge), &blue);
        let mut tail = red.clone();
        tail.extend_from_slice(&blue);
        assert_eq!(src.read_blocks(1).2, tail);
        let (_, dst) = h.try_create_volume_texture([12, 8, 3], 2, 0, format, D3DPOOL_DEFAULT);
        let dst = dst.expect("default volume");
        assert_eq!(h.update_volume_texture_hr(&src, &dst), 0);
        assert_eq!(h.set_volume_texture(0, &dst), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.75, 0.5]),
            0xff00_00ff,
            1,
            "partial x/y/z block published",
        );
        for coord in [
            [1.0 / 6.0, 0.75, 0.5],
            [0.5, 0.25, 0.5],
            [0.5, 0.75, 1.0 / 6.0],
            [0.5, 0.75, 5.0 / 6.0],
        ] {
            assert_pixel_approx(
                sample(&h, coord),
                0xffff_0000,
                1,
                "untouched block/slice preserved",
            );
        }
        // Stage binding owns the destination after the application releases its reference.
        drop(dst);
        drop(src);
        assert_pixel_approx(
            sample(&h, [0.5, 0.75, 0.5]),
            0xff00_00ff,
            1,
            "bound compressed volume retained",
        );
    }
}

#[test]
fn dxt_volume_srgb_decode_and_linear_slice_filtering() {
    use mtld3d_types::{D3DSAMP_SRGBTEXTURE, D3DTEXF_LINEAR};
    let h = Harness::new();
    setup(&h);
    for format in FORMATS {
        let (_, tex) = h.try_create_volume_texture([4, 4, 2], 1, 0, format, D3DPOOL_MANAGED);
        let tex = tex.expect("managed volume");
        let mut bytes = solid_block(format, 0xf800, 255);
        bytes.extend(solid_block(format, 0x001f, 255));
        tex.write_blocks(0, None, &bytes);
        assert_eq!(h.set_volume_texture(0, &tex), 0);
        for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
            assert_eq!(h.set_sampler_state(0, state, D3DTEXF_LINEAR), 0);
        }
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.5]),
            0xff80_0080,
            2,
            "linear interpolation between BC slices",
        );
        for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
            assert_eq!(h.set_sampler_state(0, state, D3DTEXF_POINT), 0);
        }
        tex.write_blocks(0, None, &solid_block(format, 0x8410, 255).repeat(2));
        assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 0), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.75]),
            0xff84_8284,
            2,
            "linear view retains encoded BC channels",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 1), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.75]),
            0xff3b_393b,
            2,
            "BC volume sRGB view decodes channels",
        );
        assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 0), 0);
    }
}

#[test]
fn dxt_volume_alpha_modes_and_managed_reset() {
    let h = Harness::new();
    setup(&h);
    for format in FORMATS {
        let (_, tex) = h.try_create_volume_texture([4, 4, 2], 1, 0, format, D3DPOOL_MANAGED);
        let tex = tex.expect("managed volume");
        let transparent = if format == D3DFMT_DXT1 {
            vec![0, 0, 255, 255, 255, 255, 255, 255]
        } else {
            solid_block(format, 0xf800, 0)
        };
        let mut bytes = transparent;
        bytes.extend(solid_block(format, 0xf800, 255));
        tex.write_blocks(0, None, &bytes);
        assert_eq!(h.set_volume_texture(0, &tex), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.25]),
            if format == D3DFMT_DXT1 {
                0
            } else {
                0x00ff_0000
            },
            1,
            "transparent compressed block",
        );
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.75]),
            0xffff_0000,
            1,
            "opaque compressed block",
        );
        assert_eq!(h.clear_texture(0), 0);
        assert_eq!(h.reset(640, 480), 0);
        setup(&h);
        assert_eq!(tex.read_blocks(0).2, bytes);
        assert_eq!(h.set_volume_texture(0, &tex), 0);
        assert_pixel_approx(
            sample(&h, [0.5, 0.5, 0.75]),
            0xffff_0000,
            1,
            "managed compressed volume after Reset",
        );
    }
}

/// VS3 reads an explicit LOD from a volume declaration, then exports the texel as COLOR0.
#[rustfmt::skip]
const VS_VOLUME_FETCH: [u32;30]=[
    0xfffe_0300,0x0200_001f,0x8000_0000,0x900f_0000,
    0x0200_001f,0xa000_0000,0xa00f_0800,
    0x0200_001f,0x8000_0000,0xe00f_0000,
    0x0200_001f,0x8000_000a,0xe00f_0001,
    0x0500_0051,0xa00f_0004,0x3f00_0000,0x3f00_0000,0x3f40_0000,0,
    0x0200_0001,0xe00f_0000,0x90e4_0000,
    0x0300_005f,0x800f_0000,0xa0e4_0004,0xa0e4_0800,
    0x0200_0001,0xe00f_0001,0x80e4_0000,0x0000_ffff,
];

#[test]
fn dxt_volume_vertex_fetch_follows_native_resource_dimension() {
    let h = Harness::new();
    setup(&h);
    let vs = h.create_vertex_shader(&VS_VOLUME_FETCH);
    let ps = h.create_pixel_shader(&[
        0xffff_0300,
        0x0200_001f,
        0x8000_000a,
        0x900f_0000,
        0x0200_0001,
        0x800f_0800,
        0x90e4_0000,
        0xffff,
    ]);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    for format in FORMATS {
        for depth in [1, 2] {
            let (_, tex) =
                h.try_create_volume_texture([4, 4, depth], 1, 0, format, D3DPOOL_MANAGED);
            let tex = tex.expect("vertex sample volume");
            let bytes = if depth == 1 {
                solid_block(format, 0x001f, 255)
            } else {
                let mut b = solid_block(format, 0xf800, 255);
                b.extend(solid_block(format, 0x001f, 255));
                b
            };
            tex.write_blocks(0, None, &bytes);
            assert_eq!(h.set_volume_texture(257, &tex), 0);
            for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER] {
                assert_eq!(h.set_sampler_state(257, state, D3DTEXF_POINT), 0);
            }
            assert_pixel_approx(
                sample(&h, [0.5, 0.5, 0.75]),
                0xff00_00ff,
                1,
                "VS3 fetch reads the native bound shape",
            );
        }
    }
}

#[test]
fn dxt_volume_pixel_shaders_sample_all_compressed_formats() {
    let h = Harness::new();
    setup(&h);
    let vs3 = h.create_vertex_shader(&[
        0xfffe_0300,
        0x0200_001f,
        0x8000_0000,
        0x900f_0000,
        0x0200_001f,
        0x8000_0000,
        0xe00f_0000,
        0x0200_0001,
        0xe00f_0000,
        0x90e4_0000,
        0xffff,
    ]);
    for version in [2, 3] {
        if version == 3 {
            assert_eq!(h.set_vertex_shader(&vs3), 0);
        } else {
            assert_eq!(h.clear_vertex_shader(), 0);
        }
        // Constant coordinates avoid interpolation differences; sampler is explicitly volume.
        let tokens = if version == 2 {
            vec![
                0xffff_0200,
                0x0200_001f,
                0xa000_0000,
                0xa00f_0800,
                0x0500_0051,
                0xa00f_0000,
                0x3f00_0000,
                0x3f00_0000,
                0x3f40_0000,
                0,
                0x0300_0042,
                0x800f_0000,
                0xa0e4_0000,
                0xa0e4_0800,
                0x0200_0001,
                0x800f_0800,
                0x80e4_0000,
                0xffff,
            ]
        } else {
            vec![
                0xffff_0300,
                0x0200_001f,
                0xa000_0000,
                0xa00f_0800,
                0x0500_0051,
                0xa00f_0000,
                0x3f00_0000,
                0x3f00_0000,
                0x3f40_0000,
                0,
                0x0300_0042,
                0x800f_0000,
                0xa0e4_0000,
                0xa0e4_0800,
                0x0200_0001,
                0x800f_0800,
                0x80e4_0000,
                0xffff,
            ]
        };
        let ps = h.create_pixel_shader(&tokens);
        assert_eq!(h.set_pixel_shader(&ps), 0);
        for format in FORMATS {
            for depth in [1, 2] {
                let (_, tex) =
                    h.try_create_volume_texture([4, 4, depth], 1, 0, format, D3DPOOL_MANAGED);
                let tex = tex.expect("pixel sample volume");
                let mut b = if depth == 2 {
                    solid_block(format, 0xf800, 255)
                } else {
                    Vec::new()
                };
                b.extend(solid_block(format, 0x001f, 255));
                tex.write_blocks(0, None, &b);
                assert_eq!(h.set_volume_texture(0, &tex), 0);
                assert_pixel_approx(
                    sample(&h, [0.5, 0.5, 0.75]),
                    0xff00_00ff,
                    1,
                    "programmable BC volume sample",
                );
            }
        }
    }
}

#[test]
fn dxt_volume_queued_partial_write_preserves_earlier_draw_and_release() {
    use mtld3d_types::{D3DBOX, D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC};
    let h = Harness::new();
    setup(&h);
    for format in FORMATS {
        for (pool, usage) in [(D3DPOOL_MANAGED, 0), (D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC)] {
            let (_, texture) = h.try_create_volume_texture([8, 8, 2], 1, usage, format, pool);
            let texture = texture.expect("writable compressed volume");
            texture.write_blocks(0, None, &solid_block(format, 0xf800, 255).repeat(8));
            assert_eq!(h.set_volume_texture(0, &texture), 0);
            h.render_once(0, |d| {
                assert_eq!(
                    d.draw_primitive_up(
                        D3DPT_TRIANGLELIST,
                        2,
                        &quad(-1.0, 0.0, [0.75, 0.75, 0.75])
                    ),
                    0
                );
                texture.write_blocks(
                    0,
                    Some(&D3DBOX {
                        left: 4,
                        top: 4,
                        right: 8,
                        bottom: 8,
                        front: 1,
                        back: 2,
                    }),
                    &solid_block(format, 0x001f, 255),
                );
                assert_eq!(
                    d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(0.0, 1.0, [0.75, 0.75, 0.75])),
                    0
                );
                drop(texture);
            });
            assert_pixel_approx(
                h.read_pixel(160, 240),
                0xffff_0000,
                1,
                "earlier BC upload owns its source version",
            );
            assert_pixel_approx(
                h.read_pixel(480, 240),
                0xff00_00ff,
                1,
                "later BC upload owns replacement source",
            );
            assert_pixel_approx(
                sample(&h, [0.25, 0.75, 0.75]),
                0xffff_0000,
                1,
                "partial rename preserves other blocks",
            );
        }
    }
}

#[test]
fn dxt_volume_premultiplied_aliases_keep_raw_samples_and_application_blending() {
    use mtld3d_types::{
        D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND, D3DRS_SRCBLEND,
    };
    let h = Harness::new();
    setup(&h);
    for (alias, ordinary, alpha, blue) in [
        (D3DFMT_DXT2, D3DFMT_DXT3, 136u8, 60u32),
        (D3DFMT_DXT4, D3DFMT_DXT5, 128u8, 64u32),
    ] {
        let block: [u8; 16] = if alias == D3DFMT_DXT2 {
            [
                0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0, 0x40, 0, 0x40, 0, 0, 0, 0,
            ]
        } else {
            [0x80, 0x80, 0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0x40, 0, 0, 0, 0]
        };
        let raw = block.repeat(2);
        let mut observed = Vec::new();
        for format in [alias, ordinary] {
            let (hr, tex) = h.try_create_volume_texture([4, 4, 2], 1, 0, format, D3DPOOL_MANAGED);
            assert_eq!(hr, 0);
            let tex = tex.expect("premultiplied alias");
            assert_eq!(tex.level_desc(0).1.format, format);
            tex.write_blocks(0, None, &raw);
            assert_eq!(tex.read_blocks(0).2, raw);
            assert_eq!(h.set_volume_texture(0, &tex), 0);
            assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 0), 0);
            let pixel = sample(&h, [0.5, 0.5, 0.75]);
            assert_pixel_approx(
                pixel,
                (u32::from(alpha) << 24) | 0x0042_0000,
                1,
                "raw premultiplied content is neither multiplied nor divided",
            );
            observed.push(pixel);
            assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE), 0);
            assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA), 0);
            assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
            h.render_once(0xff00_0080, |d| {
                assert_eq!(
                    d.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad(-1.0, 1.0, [0.5, 0.5, 0.75])),
                    0
                );
            });
            assert_pixel_approx(
                h.read_pixel(320, 240),
                0xff42_0000 | blue,
                1,
                "application selects premultiplied blend factors",
            );
        }
        assert_eq!(observed[0], observed[1]);
    }
}
