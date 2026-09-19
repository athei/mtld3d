//! Triangle wireframe rasterization and dynamic fill-state lifetime.

use mtld3d_tests::{DrawIndexedUpParams, Harness, HarnessConfig, PosColorVertex};
use mtld3d_types::{
    D3DCLEAR_STENCIL, D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_EQUAL, D3DCULL_NONE, D3DFILL_POINT,
    D3DFILL_SOLID, D3DFILL_WIREFRAME, D3DFMT_A8R8G8B8, D3DFMT_D24S8, D3DFMT_INDEX16,
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DLOCK_READONLY, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM,
    D3DPT_LINELIST, D3DPT_POINTLIST, D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP,
    D3DRECT, D3DRS_COLORWRITEENABLE, D3DRS_CULLMODE, D3DRS_FILLMODE, D3DRS_LIGHTING,
    D3DRS_POINTSIZE, D3DRS_SCISSORTESTENABLE, D3DRS_STENCILENABLE, D3DRS_STENCILFUNC,
    D3DRS_STENCILPASS, D3DRS_STENCILREF, D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DSTENCILOP_KEEP,
    D3DSTENCILOP_REPLACE,
};

const BLACK: u32 = 0xFF00_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;
const RED: u32 = 0xFFFF_0000;
const SIDE: u32 = 128;
const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;

fn harness(depth: bool) -> Harness {
    let h = Harness::create(&HarnessConfig {
        width: SIDE,
        height: SIDE,
        depth_format: depth.then_some(D3DFMT_D24S8),
        ..HarnessConfig::default()
    });
    arm(&h);
    h
}

fn arm(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    assert_eq!(h.set_fvf(FVF), 0);
    h.select_diffuse_stage(0);
}

fn vertices(color: u32, z: f32) -> [PosColorVertex; 4] {
    [(-0.75, 0.75), (-0.75, -0.75), (0.75, 0.75), (0.75, -0.75)].map(|(x, y)| PosColorVertex {
        x,
        y,
        z,
        color,
    })
}

fn square(h: &Harness, color: u32, z: f32) {
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &vertices(color, z)),
        0
    );
}

fn pixels(h: &Harness) -> Vec<u32> {
    let rt = h.render_target(0);
    let (hr, desc) = rt.desc();
    assert_eq!(hr, 0);
    let resolved = (desc.multi_sample_type != mtld3d_types::D3DMULTISAMPLE_NONE)
        .then(|| h.create_render_target(SIDE, SIDE, desc.format));
    if let Some(target) = &resolved {
        assert_eq!(h.stretch_rect(&rt, target, mtld3d_types::D3DTEXF_NONE), 0);
    }
    let sys = h.create_offscreen_plain_surface(SIDE, SIDE, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM);
    assert_eq!(
        h.get_render_target_data_hr(resolved.as_ref().unwrap_or(&rt), &sys),
        0
    );
    let lock = sys.lock_rect(D3DLOCK_READONLY);
    let pitch = usize::try_from(lock.pitch()).expect("positive pitch") / 4;
    let data = lock.as_u32(pitch * SIDE as usize);
    (0..SIDE as usize)
        .flat_map(|row| {
            data[row * pitch..row * pitch + SIDE as usize]
                .iter()
                .copied()
        })
        .collect()
}

const fn pixel(data: &[u32], x: usize, y: usize) -> u32 {
    data[y * SIDE as usize + x]
}

fn assert_edge(data: &[u32], color: u32) {
    // A small neighborhood tolerates the resolve footprint of a scaled target.
    assert!(
        (13..19).any(|x| (58..70).any(|y| pixel(data, x, y) & color & 0x00FF_FFFF != 0)),
        "left edge must rasterize"
    );
}

#[test]
fn all_triangle_topologies_and_draw_entry_points_use_wireframe() {
    let h = harness(false);
    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
    let verts = vertices(GREEN, 0.5);
    for (prim, order) in [
        (D3DPT_TRIANGLELIST, vec![0u16, 1, 2, 2, 1, 3]),
        (D3DPT_TRIANGLESTRIP, vec![0, 1, 2, 3]),
        (D3DPT_TRIANGLEFAN, vec![0, 1, 3, 2]),
    ] {
        let ordered: Vec<_> = order.iter().map(|&i| verts[usize::from(i)]).collect();
        let stride = u32::try_from(core::mem::size_of::<PosColorVertex>()).unwrap();
        let vb = h.create_vertex_buffer(
            stride * u32::try_from(ordered.len()).unwrap(),
            0,
            FVF,
            D3DPOOL_MANAGED,
        );
        vb.lock(0, 0, 0).write(&ordered);
        let ib = h.create_index_buffer(
            u32::try_from(order.len() * 2).unwrap(),
            0,
            D3DFMT_INDEX16,
            D3DPOOL_MANAGED,
        );
        ib.lock(0, 0, 0).write(&order);
        for route in 0..4 {
            h.render_once(BLACK, |d| match route {
                0 => assert_eq!(d.draw_primitive_up(prim, 2, &ordered), 0),
                1 => {
                    assert_eq!(d.set_stream_source(0, &vb, 0, stride), 0);
                    assert_eq!(d.draw_primitive(prim, 0, 2), 0);
                }
                2 => assert_eq!(
                    d.draw_indexed_primitive_up(
                        &DrawIndexedUpParams {
                            prim,
                            min_vertex_index: 0,
                            num_vertices: 4,
                            prim_count: 2,
                            index_format: D3DFMT_INDEX16,
                        },
                        &order,
                        &verts
                    ),
                    0
                ),
                _ => {
                    vb.lock(0, 0, 0).write(&verts);
                    assert_eq!(d.set_stream_source(0, &vb, 0, stride), 0);
                    assert_eq!(d.set_indices(&ib), 0);
                    assert_eq!(d.draw_indexed_primitive(prim, 0, 0, 4, 0, 2), 0);
                }
            });
            let data = pixels(&h);
            assert_eq!(
                pixel(&data, 64, 32),
                BLACK,
                "interior: topology {prim}, route {route}"
            );
            assert_edge(&data, GREEN);
        }
    }
}

#[test]
fn fill_changes_leave_line_and_point_inputs_unchanged() {
    let h = harness(false);
    assert_eq!(h.set_render_state(D3DRS_POINTSIZE, 9f32.to_bits()), 0);
    let verts = vertices(GREEN, 0.5);
    for prim in [D3DPT_LINELIST, D3DPT_POINTLIST] {
        let mut reference = Vec::new();
        for mode in [D3DFILL_SOLID, D3DFILL_WIREFRAME, D3DFILL_SOLID] {
            assert_eq!(h.set_render_state(D3DRS_FILLMODE, mode), 0);
            h.render_once(BLACK, |d| {
                assert_eq!(d.draw_primitive_up(prim, 2, &verts), 0);
            });
            let data = pixels(&h);
            assert!(
                data.iter()
                    .any(|&p| p & 0x00FF_0000 == 0 && p & 0x0000_FF00 != 0)
            );
            if reference.is_empty() {
                reference = data;
            } else {
                assert_eq!(data, reference);
            }
        }
    }
}

#[test]
fn color_clear_fills_interior_and_next_draw_restores_wireframe() {
    let h = harness(false);
    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
    h.render_once(BLACK, |d| {
        square(d, GREEN, 0.5);
        assert_eq!(d.clear_target(BLUE), 0);
        square(d, RED, 0.5);
    });
    let data = pixels(&h);
    assert_eq!(pixel(&data, 64, 32), BLUE, "clear fills wireframe interior");
    assert_edge(&data, RED);
    assert_eq!(pixel(&data, 4, 4), BLUE, "clear covers the target");
}

#[test]
fn scaled_stretch_into_current_target_fills_and_restores_wireframe() {
    let h = harness(false);
    let src = h.create_render_target(32, 32, D3DFMT_A8R8G8B8);
    let dst = h.render_target(0);
    assert_eq!(h.color_fill_hr(&src, BLUE), 0);
    for cull in [
        D3DCULL_NONE,
        mtld3d_types::D3DCULL_CW,
        mtld3d_types::D3DCULL_CCW,
    ] {
        assert_eq!(h.clear_target(BLACK), 0);
        assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, cull), 0);
        assert_eq!(h.begin_scene(), 0);
        square(&h, GREEN, 0.5);
        assert_eq!(h.end_scene(), 0);
        // EndScene does not close the Metal encoder. The same target, no depth
        // attachment and a distinct source let the blit reuse its wireframe state.
        assert_eq!(h.stretch_rect(&src, &dst, mtld3d_types::D3DTEXF_POINT), 0);
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
        assert_eq!(h.begin_scene(), 0);
        square(&h, RED, 0.5);
        assert_eq!(h.end_scene(), 0);
        let data = pixels(&h);
        assert_eq!(pixel(&data, 64, 32), BLUE, "blit fills the interior");
        assert_eq!(pixel(&data, 4, 4), BLUE, "blit covers the target");
        assert_edge(&data, RED);
    }
}

#[test]
fn depth_stencil_clears_fill_interior_and_restore_wireframe() {
    let h = harness(true);
    for flags in [
        D3DCLEAR_ZBUFFER,
        D3DCLEAR_STENCIL,
        D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL,
    ] {
        h.render_once(BLACK, |d| {
            assert_eq!(d.clear(D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL, 0, 1.0, 0), 0);
            assert_eq!(d.set_render_state(D3DRS_SCISSORTESTENABLE, 0), 0);
            assert_eq!(d.set_render_state(D3DRS_STENCILENABLE, 1), 0);
            assert_eq!(
                d.set_render_state(D3DRS_STENCILFUNC, mtld3d_types::D3DCMP_ALWAYS),
                0
            );
            assert_eq!(
                d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_REPLACE),
                0
            );
            assert_eq!(d.set_render_state(D3DRS_STENCILREF, 2), 0);
            assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
            square(d, BLACK, 0.25);
            assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
            square(d, BLACK, 0.1);
            assert_eq!(d.clear(flags, 0, 1.0, 1), 0);
            let reference = if flags & D3DCLEAR_STENCIL != 0 { 1 } else { 2 };
            let z = if flags & D3DCLEAR_ZBUFFER != 0 {
                0.75
            } else {
                0.05
            };
            assert_eq!(d.set_render_state(D3DRS_STENCILFUNC, D3DCMP_EQUAL), 0);
            assert_eq!(d.set_render_state(D3DRS_STENCILPASS, D3DSTENCILOP_KEEP), 0);
            assert_eq!(d.set_render_state(D3DRS_STENCILREF, reference), 0);
            square(d, GREEN, z);
            assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
            assert_eq!(d.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
            assert_eq!(
                d.set_scissor_rect(&D3DRECT {
                    x1: 40,
                    y1: 40,
                    x2: 88,
                    y2: 88
                }),
                0
            );
            square(d, BLUE, z);
        });
        let data = pixels(&h);
        assert_eq!(
            pixel(&data, 64, 48),
            BLUE,
            "clear fills both requested planes: {flags}"
        );
        assert_eq!(
            pixel(&data, 64, 32),
            BLACK,
            "draw after clear stays wireframe: {flags}"
        );
        assert_edge(&data, GREEN);
    }
}

#[test]
fn removed_color_clear_preserves_solid_depth_draw_and_wireframe_restore() {
    let h = harness(true);
    let back = h.render_target(0);
    let unused = h.create_render_target(SIDE, SIDE, D3DFMT_A8R8G8B8);
    h.render_once(BLACK, |d| {
        assert_eq!(d.set_render_target(0, &unused), 0);
        assert_eq!(d.clear(D3DCLEAR_ZBUFFER, 0, 1.0, 0), 0);
        assert_eq!(d.set_render_state(D3DRS_COLORWRITEENABLE, 0), 0);
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
        square(d, BLACK, 0.25);
        assert_eq!(
            d.clear_rects(
                D3DCLEAR_TARGET,
                RED,
                1.0,
                0,
                &[D3DRECT {
                    x1: 32,
                    y1: 32,
                    x2: 96,
                    y2: 96
                }]
            ),
            0
        );
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
        square(d, BLACK, 0.5);
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
        square(d, BLACK, 0.1);
        assert_eq!(d.set_render_target(0, &back), 0);
        assert_eq!(d.set_render_state(D3DRS_COLORWRITEENABLE, 0xF), 0);
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
        square(d, BLUE, 0.75);
        assert_eq!(d.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
        assert_eq!(
            d.set_scissor_rect(&D3DRECT {
                x1: 32,
                y1: 48,
                x2: 96,
                y2: 96
            }),
            0
        );
        square(d, GREEN, 0.375);
    });
    assert_eq!(
        h.read_pixel(64, 32),
        BLACK,
        "solid draw occludes the farther probe"
    );
    assert_eq!(
        h.read_pixel(64, 80),
        GREEN,
        "restored wireframe leaves the solid interior depth intact"
    );
}

#[test]
fn state_blocks_restore_fill_and_reset_restores_solid() {
    let h = harness(false);
    for kind in [D3DSBT_ALL, D3DSBT_PIXELSTATE] {
        assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
        let block = h.create_state_block(kind);
        assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_SOLID), 0);
        h.render_once(BLACK, |d| {
            square(d, BLUE, 0.5);
            assert_eq!(block.apply(), 0);
            square(d, RED, 0.5);
        });
        assert_eq!(h.render_state(D3DRS_FILLMODE), D3DFILL_WIREFRAME);
        let data = pixels(&h);
        assert_eq!(pixel(&data, 64, 32), BLUE);
        assert_edge(&data, RED);
    }
    assert_eq!(h.reset(SIDE, SIDE), 0);
    arm(&h);
    assert_eq!(h.render_state(D3DRS_FILLMODE), D3DFILL_SOLID);
    h.render_once(BLACK, |d| square(d, GREEN, 0.5));
    assert_eq!(h.read_pixel(64, 32), GREEN);
}

#[test]
fn point_fill_remains_a_stored_solid_fallback_after_wireframe() {
    let h = harness(false);
    h.render_once(BLACK, |d| {
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
        square(d, RED, 0.5);
        assert_eq!(d.set_render_state(D3DRS_FILLMODE, D3DFILL_POINT), 0);
        square(d, GREEN, 0.5);
    });
    assert_eq!(h.render_state(D3DRS_FILLMODE), D3DFILL_POINT);
    assert_eq!(h.read_pixel(64, 32), GREEN);
}

#[test]
fn wireframe_keeps_triangle_culling() {
    let h = harness(false);
    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
    let mut counts = Vec::new();
    for cull in [
        mtld3d_types::D3DCULL_CW,
        mtld3d_types::D3DCULL_CCW,
        D3DCULL_NONE,
    ] {
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, cull), 0);
        h.render_once(BLACK, |d| square(d, GREEN, 0.5));
        counts.push(pixels(&h).iter().filter(|&&p| p != BLACK).count());
    }
    assert_ne!(
        counts[0] == 0,
        counts[1] == 0,
        "exactly one winding is culled"
    );
    assert_eq!(
        counts[2],
        counts[0].max(counts[1]),
        "no-cull keeps the same edges"
    );
}

#[test]
fn wireframe_on_multisampled_target_leaves_interior_clear() {
    let h = Harness::create(&HarnessConfig {
        width: SIDE,
        height: SIDE,
        multi_sample_type: mtld3d_types::D3DMULTISAMPLE_4_SAMPLES,
        ..HarnessConfig::default()
    });
    arm(&h);
    assert_eq!(h.set_render_state(D3DRS_FILLMODE, D3DFILL_WIREFRAME), 0);
    h.render_once(BLACK, |d| square(d, GREEN, 0.5));
    let data = pixels(&h);
    assert_eq!(pixel(&data, 64, 32), BLACK);
    assert_edge(&data, GREEN);
}
