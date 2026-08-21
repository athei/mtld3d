use mtld3d_types::render_state_defaults;

use super::*;

/// Lane `i` as its f32 bit pattern, so the asserts compare exactly.
fn lane(bytes: &[u8; VS_DRAW_BYTES], i: usize) -> u32 {
    u32::from_le_bytes([
        bytes[i * 4],
        bytes[i * 4 + 1],
        bytes[i * 4 + 2],
        bytes[i * 4 + 3],
    ])
}

fn row(bytes: &[u8; VS_DRAW_BYTES], r: usize) -> [u32; 4] {
    [
        lane(bytes, r * 4),
        lane(bytes, r * 4 + 1),
        lane(bytes, r * 4 + 2),
        lane(bytes, r * 4 + 3),
    ]
}

fn bits(v: [f32; 4]) -> [u32; 4] {
    v.map(f32::to_bits)
}

const NO_PLANES: [[f32; 4]; MAX_CLIP_PLANES] = [[0.0; 4]; MAX_CLIP_PLANES];

#[test]
fn defaults_pack_size_one_clamped_to_the_cap_and_identity_scale() {
    let bytes = build_vs_draw_bytes(&render_state_defaults(), &D3DMATRIX::IDENTITY, &NO_PLANES);
    assert_eq!(lane(&bytes, 0), 1.0f32.to_bits(), "POINTSIZE");
    assert_eq!(lane(&bytes, 1), 1.0f32.to_bits(), "POINTSIZE_MIN");
    assert_eq!(
        lane(&bytes, 2),
        mtld3d_types::MAX_POINT_SIZE.to_bits(),
        "POINTSIZE_MAX"
    );
    assert_eq!(lane(&bytes, 4), 1.0f32.to_bits(), "POINTSCALE_A");
    assert_eq!(lane(&bytes, 5), 0.0f32.to_bits(), "POINTSCALE_B");
    assert_eq!(lane(&bytes, 6), 0.0f32.to_bits(), "POINTSCALE_C");
}

#[test]
fn every_point_state_lands_in_its_lane() {
    let mut rs = render_state_defaults();
    rs[D3DRS_POINTSIZE as usize] = 32.0f32.to_bits();
    rs[D3DRS_POINTSIZE_MIN as usize] = 2.0f32.to_bits();
    rs[D3DRS_POINTSIZE_MAX as usize] = 48.0f32.to_bits();
    rs[D3DRS_POINTSCALE_A as usize] = 0.5f32.to_bits();
    rs[D3DRS_POINTSCALE_B as usize] = 0.25f32.to_bits();
    rs[D3DRS_POINTSCALE_C as usize] = 0.125f32.to_bits();
    let bytes = build_vs_draw_bytes(&rs, &D3DMATRIX::IDENTITY, &NO_PLANES);
    assert_eq!(
        [0, 1, 2, 4, 5, 6].map(|i| lane(&bytes, i)),
        [32.0f32, 2.0, 48.0, 0.5, 0.25, 0.125].map(f32::to_bits)
    );
    assert_eq!(lane(&bytes, 3), 0);
    assert_eq!(lane(&bytes, 7), 0);
}

#[test]
fn msl_struct_matches_the_byte_layout() {
    // Twelve float4 rows, named as the emitters read them.
    assert!(VS_DRAW_MSL.contains("float4 point;"));
    assert!(VS_DRAW_MSL.contains("float4 point_scale;"));
    assert!(VS_DRAW_MSL.contains("float4 inv_view[4];"));
    assert!(VS_DRAW_MSL.contains("float4 clip[6];"));
    assert_eq!((2 + 4 + MAX_CLIP_PLANES) * 16, VS_DRAW_BYTES);
}

#[test]
fn identity_view_packs_an_identity_inverse() {
    let bytes = build_vs_draw_bytes(&render_state_defaults(), &D3DMATRIX::IDENTITY, &NO_PLANES);
    for r in 0..4 {
        let mut expect = [0.0f32; 4];
        expect[r] = 1.0;
        assert_eq!(row(&bytes, 2 + r), bits(expect), "inv_view row {r}");
    }
}

#[test]
fn inverse_view_rows_map_eye_space_back_to_world() {
    // A D3D view translating by (1, 2, 3): pos_view = pos_world * V puts
    // the translation in row 3. The packed rows are the columns of V^-1,
    // so dot(pos_view, row_i) recovers world lane i.
    let mut view = D3DMATRIX::IDENTITY;
    view.m[12] = 1.0;
    view.m[13] = 2.0;
    view.m[14] = 3.0;
    let bytes = build_vs_draw_bytes(&render_state_defaults(), &view, &NO_PLANES);
    let pos_view = [1.0 + 10.0, 2.0 + 20.0, 3.0 + 30.0, 1.0];
    let world: Vec<u32> = (0..4)
        .map(|i| {
            let r = row(&bytes, 2 + i).map(f32::from_bits);
            (0..4).map(|k| pos_view[k] * r[k]).sum::<f32>().to_bits()
        })
        .collect();
    assert_eq!(world, [10.0f32, 20.0, 30.0, 1.0].map(f32::to_bits));
}

#[test]
fn enabled_planes_pack_from_index_zero_and_clipping_off_drops_them() {
    let mut rs = render_state_defaults();
    let mut planes = NO_PLANES;
    planes[1] = [0.0, 1.0, 0.0, 0.5];
    planes[4] = [1.0, 0.0, 0.0, -0.25];
    planes[5] = [9.0; 4];
    rs[D3DRS_CLIPPLANEENABLE as usize] = (1 << 1) | (1 << 4) | (1 << 7);
    assert_eq!(clip_plane_count(&rs), 2, "bit 7 is past MaxUserClipPlanes");
    let bytes = build_vs_draw_bytes(&rs, &D3DMATRIX::IDENTITY, &planes);
    assert_eq!(
        row(&bytes, 6),
        bits(planes[1]),
        "first enabled plane at clip[0]"
    );
    assert_eq!(
        row(&bytes, 7),
        bits(planes[4]),
        "second enabled plane at clip[1]"
    );
    assert_eq!(row(&bytes, 8), [0; 4], "disabled plane 5 is not packed");
    rs[D3DRS_CLIPPING as usize] = 0;
    assert_eq!(clip_plane_count(&rs), 0, "CLIPPING off disables the planes");
    let bytes = build_vs_draw_bytes(&rs, &D3DMATRIX::IDENTITY, &planes);
    assert_eq!(row(&bytes, 6), [0; 4]);
}
