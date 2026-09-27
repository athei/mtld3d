//! Recorded inputs and resource lifetimes across queued encoder work.
//!
//! Pixel checks retain several frame outputs before reading any back, so
//! caller buffers and released resources cannot be borrowed until a later
//! frame drains the queue. Device release also follows queued cold draws.

use mtld3d_tests::{
    DrawIndexedUpParams, Harness, HarnessConfig, Surface, TexturedVertex, Vertex, assert_pixel_eq,
};
use mtld3d_types::{
    D3D_OK, D3DCULL_NONE, D3DFMT_A8R8G8B8, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_TEX1, D3DFVF_XYZ,
    D3DLOCK_READONLY, D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DPT_TRIANGLELIST,
    D3DRS_CULLMODE, D3DRS_LIGHTING, D3DUSAGE_RENDERTARGET, D3DUSAGE_WRITEONLY,
};

const BLACK: u32 = 0xFF00_0000;
const COLORS: [u32; 4] = [0xFFFF_0000, 0xFF00_FF00, 0xFF00_00FF, 0xFFFF_FF00];
const CONSTANT_PS: [u32; 5] = [
    0xFFFF_0200,
    0x0200_0001,
    0x800F_0800,
    0xA0E4_0000,
    0x0000_FFFF,
];

fn device(entries: &'static str) -> Harness {
    let h = Harness::create(&HarnessConfig {
        width: 64,
        height: 64,
        config_entries: entries,
        ..HarnessConfig::default()
    });
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), D3D_OK);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), D3D_OK);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), D3D_OK);
    h
}

const fn triangle() -> [Vertex; 3] {
    [
        Vertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: 0xFFFF_FFFF,
        },
        Vertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: 0xFFFF_FFFF,
        },
        Vertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: 0xFFFF_FFFF,
        },
    ]
}

fn constants(color: u32) -> [f32; 4] {
    let [b, g, r, _] = color.to_le_bytes();
    [
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
        1.0,
    ]
}

fn pixel(h: &Harness, target: &Surface<'_>, x: u32, y: u32) -> u32 {
    let copy = h.create_offscreen_plain_surface(64, 64, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(h.get_render_target_data_hr(target, &copy), D3D_OK);
    let lock = copy.lock_rect(D3DLOCK_READONLY);
    let at = usize::try_from(y * lock.pitch().cast_unsigned() / 4 + x).expect("pixel index");
    lock.as_u32(at + 1)[at]
}

/// UP input and constants are snapshots even when their caller overwrites them before Present.
#[test]
fn queued_frames_keep_overwritten_vertices_and_constants() {
    let h = device("shader.asyncCompile=false;shaderCache.enable=false");
    let shader = h.create_pixel_shader(&CONSTANT_PS);
    assert_eq!(h.set_pixel_shader(&shader), D3D_OK);
    let back = h.render_target(0);
    let targets: Vec<_> = (0..12)
        .map(|_| {
            h.create_texture(
                64,
                64,
                1,
                D3DUSAGE_RENDERTARGET,
                D3DFMT_A8R8G8B8,
                D3DPOOL_DEFAULT,
            )
        })
        .collect();
    let mut vertices = vec![triangle()[0]; 300];
    assert!(size_of_val(vertices.as_slice()) > 4096);
    let mut indices = [0_u16, 1, 2];
    let indexed = DrawIndexedUpParams {
        prim: D3DPT_TRIANGLELIST,
        min_vertex_index: 0,
        num_vertices: 300,
        prim_count: 1,
        index_format: D3DFMT_INDEX16,
    };
    let mut tint = [0.0; 4];
    for (frame, target) in targets.iter().enumerate() {
        let surface = target.surface_level(0);
        assert_eq!(h.set_render_target(0, &surface), D3D_OK);
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.clear_target(BLACK), D3D_OK);
        vertices[..3].copy_from_slice(&triangle());
        indices.copy_from_slice(&[0, 1, 2]);
        // A narrower triangle leaves the top-right probe uncovered, catching
        // a stale full-screen vertex snapshot independently of the colour.
        for vertex in &mut vertices {
            vertex.x = (vertex.x - 1.0) / 2.0;
        }
        tint.copy_from_slice(&constants(COLORS[frame % COLORS.len()]));
        assert_eq!(h.set_pixel_shader_constant_f(0, &tint), D3D_OK);
        if frame % 2 == 0 {
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &vertices[..3]),
                D3D_OK
            );
        } else {
            assert_eq!(
                h.draw_indexed_primitive_up(&indexed, &indices, &vertices),
                D3D_OK
            );
        }
        indices.fill(0);
        for vertex in &mut vertices {
            vertex.x = 10.0;
        }
        tint.fill(0.0);
        assert_eq!(h.set_pixel_shader_constant_f(0, &tint), D3D_OK);
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.set_render_target(0, &back), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    }
    for (frame, target) in targets.iter().enumerate() {
        let surface = target.surface_level(0);
        assert_pixel_eq(
            pixel(&h, &surface, 8, 48),
            COLORS[frame % COLORS.len()],
            "recorded tint and UP vertices",
        );
        assert_pixel_eq(
            pixel(&h, &surface, 56, 16),
            BLACK,
            "outside recorded triangle",
        );
    }
}

/// Upload sources and bound buffers survive their public references being released before Present.
#[test]
fn queued_draws_keep_released_texture_vertex_and_index_buffers() {
    let h = device("shader.asyncCompile=false;shaderCache.enable=false");
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), D3D_OK);
    h.select_texture_stage(0);
    let back = h.render_target(0);
    let replacement = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    replacement.lock(0, 0, 0).write(&[0_u16, 0, 0]);
    let targets: Vec<_> = (0..12)
        .map(|_| {
            h.create_texture(
                64,
                64,
                1,
                D3DUSAGE_RENDERTARGET,
                D3DFMT_A8R8G8B8,
                D3DPOOL_DEFAULT,
            )
        })
        .collect();
    for (frame, target) in targets.iter().enumerate() {
        let surface = target.surface_level(0);
        assert_eq!(h.set_render_target(0, &surface), D3D_OK);
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.clear_target(BLACK), D3D_OK);
        {
            let texture = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
            texture
                .lock_rect(0, 0)
                .write_u32_rect(1, 1, &[COLORS[frame % COLORS.len()]]);
            let vertices = triangle().map(|v| TexturedVertex {
                x: v.x,
                y: v.y,
                z: v.z,
                color: v.color,
                u: 0.5,
                v: 0.5,
            });
            let stride = u32::try_from(size_of::<TexturedVertex>()).expect("vertex stride");
            let vb = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
            vb.lock(0, 0, 0).write(&vertices);
            let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
            ib.lock(0, 0, 0).write(&[0_u16, 1, 2]);
            assert_eq!(h.set_stream_source(0, &vb, 0, stride), D3D_OK);
            assert_eq!(h.set_indices(&ib), D3D_OK);
            assert_eq!(h.set_texture(0, &texture), D3D_OK);
            assert_eq!(
                h.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
                D3D_OK
            );
            assert_eq!(h.set_stream_source_null(0, 0, 0), D3D_OK);
            assert_eq!(h.set_indices(&replacement), D3D_OK);
            assert_eq!(h.clear_texture(0), D3D_OK);
        }
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.set_render_target(0, &back), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
    }
    for (frame, target) in targets.iter().enumerate() {
        assert_pixel_eq(
            pixel(&h, &target.surface_level(0), 32, 32),
            COLORS[frame % COLORS.len()],
            "released draw resources",
        );
    }
}

/// Actual device release follows queued cold shader draws, then a new device renders correctly.
///
/// No readback drains the cold draws before release. Their compilation may
/// already have finished, so this checks the lifetime path without claiming
/// a deterministic overlap with a running compiler.
#[test]
fn devices_release_after_queued_cold_draws_and_new_devices_render() {
    for round in 0..8_u8 {
        let h = device("shader.asyncCompile=true;shaderCache.enable=false");
        assert_eq!(h.begin_scene(), D3D_OK);
        assert_eq!(h.clear_target(BLACK), D3D_OK);
        for index in 0..16_u8 {
            let tint =
                constants(0xFF00_0000 | (u32::from(round + 1) << 16) | (u32::from(index + 1) << 8));
            let shader = h.create_pixel_shader(&[
                0xFFFF_0200,
                0x0500_0051,
                0xA00F_0000,
                tint[0].to_bits(),
                tint[1].to_bits(),
                tint[2].to_bits(),
                tint[3].to_bits(),
                0x0200_0001,
                0x800F_0800,
                0xA0E4_0000,
                0x0000_FFFF,
            ]);
            assert_eq!(h.set_pixel_shader(&shader), D3D_OK);
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &triangle()),
                D3D_OK
            );
            assert_eq!(h.clear_pixel_shader(), D3D_OK);
        }
        assert_eq!(h.end_scene(), D3D_OK);
        assert_eq!(h.present(), D3D_OK);
        assert_eq!(h.release_device(), 0, "the actual device must be destroyed");
        drop(h);
        let fresh = device("shader.asyncCompile=false;shaderCache.enable=false");
        let color = COLORS[usize::from(round) % COLORS.len()];
        let vertices = triangle().map(|mut v| {
            v.color = color;
            v
        });
        fresh.select_diffuse_stage(0);
        fresh.render_once(BLACK, |d| {
            assert_eq!(
                d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &vertices),
                D3D_OK
            );
        });
        assert_pixel_eq(
            fresh.read_pixel(32, 32),
            color,
            "new device after cold-draw teardown",
        );
        assert_eq!(
            fresh.release_device(),
            0,
            "fresh device has no leaked public references"
        );
    }
}

/// A single-test process has exactly twelve draw-bearing frames and no readback flushes.
#[test]
fn twelve_draw_frames_have_no_readback_flushes() {
    let h = device("shader.asyncCompile=false;shaderCache.enable=false");
    {
        let shader = h.create_pixel_shader(&CONSTANT_PS);
        assert_eq!(h.set_pixel_shader(&shader), D3D_OK);
        for frame in 0..12 {
            assert_eq!(h.begin_scene(), D3D_OK);
            assert_eq!(h.clear_target(BLACK), D3D_OK);
            assert_eq!(
                h.set_pixel_shader_constant_f(0, &constants(COLORS[frame % COLORS.len()])),
                D3D_OK
            );
            assert_eq!(
                h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &triangle()),
                D3D_OK
            );
            assert_eq!(h.end_scene(), D3D_OK);
            assert_eq!(h.present(), D3D_OK);
        }
        assert_eq!(h.clear_pixel_shader(), D3D_OK);
    }
    assert_eq!(h.release_device(), 0);
}
