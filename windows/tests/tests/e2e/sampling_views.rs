//! Render-target storage and sampling use the format's distinct channel contracts.

use mtld3d_tests::{Harness, PosColorVertex, Rgba8, TexturedVertex, VolumeVertex};
use mtld3d_types::{
    D3D_OK, D3DBLEND_INVSRCALPHA, D3DBLEND_ONE, D3DBLEND_SRCALPHA, D3DBLEND_ZERO, D3DFMT_G16R16,
    D3DFMT_G16R16F, D3DFMT_G32R32F, D3DFMT_R16F, D3DFMT_R32F, D3DFMT_X8B8G8R8, D3DFMT_X8R8G8B8,
    D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ, D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST,
    D3DRS_ALPHABLENDENABLE, D3DRS_DESTBLEND, D3DRS_LIGHTING, D3DRS_SRCBLEND, D3DRS_SRGBWRITEENABLE,
    D3DRS_ZENABLE, D3DSAMP_MAGFILTER, D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER,
    D3DSAMP_SRGBTEXTURE, D3DTA_TEXTURE, D3DTEXF_POINT, D3DTOP_SELECTARG1, D3DTSS_ALPHAARG1,
    D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLOROP, D3DUSAGE_RENDERTARGET,
};

#[test]
fn g16r16_blended_target_samples_missing_channels() {
    blended_target(D3DFMT_G16R16, 2);
}

#[test]
fn r16f_blended_target_samples_missing_channels() {
    blended_target(D3DFMT_R16F, 1);
}

#[test]
fn g16r16f_blended_target_samples_missing_channels() {
    blended_target(D3DFMT_G16R16F, 2);
}

#[test]
fn r32f_blended_target_samples_missing_channels() {
    blended_target(D3DFMT_R32F, 1);
}

#[test]
fn g32r32f_blended_target_samples_missing_channels() {
    blended_target(D3DFMT_G32R32F, 2);
}

#[test]
fn x8r8g8b8_blended_target_samples_opaque_alpha() {
    blended_target(D3DFMT_X8R8G8B8, 3);
}

#[test]
fn x8b8g8r8_blended_target_samples_opaque_alpha() {
    blended_target(D3DFMT_X8B8G8R8, 3);
}

struct SamplingCase {
    face: Option<u32>,
    level: u32,
    srgb_write: bool,
    srgb_read: bool,
    vertex_fetch: bool,
}

fn blended_target(format: u32, stored_channels: u8) {
    for level in [0, 2] {
        verify_target(
            format,
            stored_channels,
            &SamplingCase {
                face: None,
                level,
                srgb_write: false,
                srgb_read: stored_channels < 3,
                vertex_fetch: false,
            },
        );
    }
}

#[test]
fn cube_faces_and_mips_keep_sampling_channels() {
    for (format, channels) in formats() {
        for face in 0..6 {
            verify_target(
                format,
                channels,
                &SamplingCase {
                    face: Some(face),
                    level: 2,
                    srgb_write: false,
                    srgb_read: channels < 3,
                    vertex_fetch: false,
                },
            );
        }
    }
}

#[test]
fn vertex_stage_reads_render_target_sampling_views() {
    for (format, channels) in formats() {
        verify_target(
            format,
            channels,
            &SamplingCase {
                face: None,
                level: 2,
                srgb_write: false,
                srgb_read: channels < 3,
                vertex_fetch: true,
            },
        );
    }
}

#[test]
fn x8_srgb_attachment_and_sampling_roles_are_independent() {
    for format in [D3DFMT_X8R8G8B8, D3DFMT_X8B8G8R8] {
        for srgb_write in [false, true] {
            for srgb_read in [false, true] {
                verify_target(
                    format,
                    3,
                    &SamplingCase {
                        face: Some(3),
                        level: 2,
                        srgb_write,
                        srgb_read,
                        vertex_fetch: false,
                    },
                );
            }
        }
    }
}

const fn formats() -> [(u32, u8); 7] {
    [
        (D3DFMT_G16R16, 2),
        (D3DFMT_R16F, 1),
        (D3DFMT_G16R16F, 2),
        (D3DFMT_R32F, 1),
        (D3DFMT_G32R32F, 2),
        (D3DFMT_X8R8G8B8, 3),
        (D3DFMT_X8B8G8R8, 3),
    ]
}

fn verify_target(format: u32, stored_channels: u8, case: &SamplingCase) {
    let h = Harness::new();
    let target = case
        .face
        .is_none()
        .then(|| h.create_texture(64, 64, 3, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT));
    let cube = case.face.map(|_| {
        h.create_cube_texture_owned(64, 3, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT)
    });
    let backbuffer = h.render_target(0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 0), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    let surface = target.as_ref().map_or_else(
        || {
            cube.as_ref()
                .expect("cube")
                .surface(case.face.expect("face"), case.level)
        },
        |target| target.surface_level(case.level),
    );
    assert_eq!(h.set_render_target(0, &surface), D3D_OK);
    drop(surface);
    assert_eq!(
        h.set_render_state(D3DRS_SRGBWRITEENABLE, u32::from(case.srgb_write)),
        D3D_OK
    );
    assert_eq!(h.clear_target(0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_ONE), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_ZERO), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill(0x8010_3000)),
        D3D_OK
    );
    assert_eq!(
        h.set_render_state(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA),
        D3D_OK
    );
    assert_eq!(
        h.set_render_state(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA),
        D3D_OK
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill(0x8020_1000)),
        D3D_OK
    );
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 0), D3D_OK);
    assert_eq!(h.set_render_target(0, &backbuffer), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_SRGBWRITEENABLE, 0), D3D_OK);
    let stage = if case.vertex_fetch { 257 } else { 0 };
    if let Some(target) = target.as_ref() {
        assert_eq!(h.set_texture(stage, target), D3D_OK);
    } else {
        assert_eq!(
            h.set_cube_texture(stage, cube.as_ref().expect("cube")),
            D3D_OK
        );
    }
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, D3DTA_TEXTURE),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_TEXTURE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), D3D_OK);
    }
    assert_eq!(
        h.set_sampler_state(stage, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(
        h.set_sampler_state(stage, D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    // Formats without an sRGB encoding still need their linear sample view.
    assert_eq!(
        h.set_sampler_state(stage, D3DSAMP_SRGBTEXTURE, u32::from(case.srgb_read)),
        D3D_OK
    );
    assert_eq!(
        h.set_sampler_state(stage, D3DSAMP_MIPFILTER, D3DTEXF_POINT),
        D3D_OK
    );
    assert_eq!(
        h.set_sampler_state(stage, D3DSAMP_MAXMIPLEVEL, case.level),
        D3D_OK
    );
    if case.vertex_fetch {
        // Position passes through; an explicit-LOD vertex fetch becomes COLOR0.
        let vs = h.create_vertex_shader(&[
            0xfffe_0300,
            0x0200_001f,
            0x8000_0000,
            0x900f_0000,
            0x0200_001f,
            0x9000_0000,
            0xa00f_0800,
            0x0200_001f,
            0x8000_0000,
            0xe00f_0000,
            0x0200_001f,
            0x8000_000a,
            0xe00f_0001,
            0x0300_005f,
            0x800f_0000,
            0xa0e4_0000,
            0xa0e4_0800,
            0x0200_0001,
            0xe00f_0000,
            0x90e4_0000,
            0x0200_0001,
            0xe00f_0001,
            0x80e4_0000,
            0x0000_ffff,
        ]);
        let ps = h.create_pixel_shader(&[
            0xffff_0300,
            0x0200_001f,
            0x8000_000a,
            0x900f_0000,
            0x0200_0001,
            0x800f_0800,
            0x90e4_0000,
            0x0000_ffff,
        ]);
        assert_eq!(
            h.set_vertex_shader_constant_f(
                0,
                &[
                    0.5,
                    0.5,
                    0.0,
                    f32::from(u16::try_from(case.level).expect("test mip fits u16"))
                ]
            ),
            D3D_OK
        );
        assert_eq!(h.set_vertex_shader(&vs), D3D_OK);
        assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
        assert_eq!(h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &fill(0)), D3D_OK);
        assert_eq!(h.clear_vertex_shader(), D3D_OK);
        assert_eq!(h.clear_pixel_shader(), D3D_OK);
    } else if let Some(face) = case.face {
        let direction = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ][face as usize];
        let vertices = fill(0xffff_ffff).map(|v| VolumeVertex {
            x: v.x,
            y: v.y,
            z: v.z,
            color: v.color,
            u: direction[0],
            v: direction[1],
            w: direction[2],
        });
        assert_eq!(
            h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1 | 0x0001_0000),
            D3D_OK
        );
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &vertices),
            D3D_OK
        );
    } else {
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
        assert_eq!(
            h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &sample()),
            D3D_OK
        );
    }
    // Release before final pass analysis and submission, keeping only queued GPU uses.
    assert_eq!(h.clear_texture(stage), D3D_OK);
    drop(target);
    drop(cube);
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
    let color = Rgba8::from_pixel(h.read_pixel(320, 240));
    let control = ordinary_sample_control(&h, format);
    if !h.device_is_paravirtual() {
        if stored_channels == 1 {
            assert_eq!(control.g, 255);
        }
        if stored_channels < 3 {
            assert_eq!(control.b, 255);
        }
        assert_eq!(control.a, 255);
    }
    // Standard sRGB transfer applied to the blended 24/255 and 32/255 lanes.
    let (red, green) = match (case.srgb_write, case.srgb_read && stored_channels == 3) {
        (true, false) => (86, 99),
        (false, true) => (2, 4),
        _ => (24, 32),
    };
    assert!(color.r.abs_diff(red) <= 2, "stored red: {color:?}");
    if stored_channels >= 2 {
        assert!(color.g.abs_diff(green) <= 2, "stored green: {color:?}");
    } else {
        assert_eq!(color.g, control.g, "missing green: {color:?}");
    }
    assert_eq!(
        color.b,
        if stored_channels >= 3 { 0 } else { control.b },
        "blue: {color:?}"
    );
    assert_eq!(color.a, control.a, "missing alpha: {color:?}");
}

// A same-format ordinary sample isolates the hosted device's ignored-swizzle
// limitation. Stored target lanes remain exact on every device; physical GPUs
// also assert the specified all-ones absent lanes above.
fn ordinary_sample_control(h: &Harness, format: u32) -> Rgba8 {
    let texture = h.create_texture(2, 2, 1, 0, format, D3DPOOL_DEFAULT);
    let bytes_per_pixel = match format {
        D3DFMT_R16F => 2,
        D3DFMT_G32R32F => 8,
        _ => 4,
    };
    let mut bytes = vec![0; bytes_per_pixel * 4];
    if matches!(format, D3DFMT_X8R8G8B8 | D3DFMT_X8B8G8R8) {
        for pixel in bytes.as_chunks_mut::<4>().0 {
            pixel[3] = 128;
        }
    }
    texture
        .lock_rect(0, 0)
        .write_u8_rect(bytes_per_pixel * 2, 2, &bytes);
    assert_eq!(h.set_texture(0, &texture), D3D_OK);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MAXMIPLEVEL, 0), D3D_OK);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_SRGBTEXTURE, 0), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    assert_eq!(h.begin_scene(), D3D_OK);
    assert_eq!(h.clear_target(0), D3D_OK);
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &sample()),
        D3D_OK
    );
    assert_eq!(h.end_scene(), D3D_OK);
    assert_eq!(h.present(), D3D_OK);
    let color = Rgba8::from_pixel(h.read_pixel(320, 240));
    assert_eq!(color.r, 0, "ordinary sample control: {color:?}");
    assert_eq!(h.clear_texture(0), D3D_OK);
    color
}

const fn fill(color: u32) -> [PosColorVertex; 3] {
    [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color,
        },
    ]
}

fn sample() -> [TexturedVertex; 3] {
    fill(0xffff_ffff).map(|v| TexturedVertex {
        x: v.x,
        y: v.y,
        z: v.z,
        color: v.color,
        u: 0.5,
        v: 0.5,
    })
}
