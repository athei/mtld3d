//! Unit tests for the per-draw vertex-stage uniform packing.
//!
//! Pins the exact byte layout the vertex shaders read: point size with its
//! clamp range and scale factors in lane order, the inverse view rows that map
//! an eye-space position back to world space, and the enabled clip planes
//! packed from index zero (`D3DRS_CLIPPING` off drops them all). One check ties
//! `VS_DRAW_MSL` to `VS_DRAW_BYTES` so a shifted lane cannot pass silently.

use mtld3d_types::{D3DRS_POINTSIZE, render_state_defaults};

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

fn build_vs_draw_bytes(
    rs: &[u32; RENDER_STATE_COUNT],
    point_size: u32,
    view: &D3DMATRIX,
    planes: &[[f32; 4]],
) -> [u8; VS_DRAW_BYTES] {
    VsDrawState::new().build_bytes(rs, point_size, view, planes)
}

#[test]
fn defaults_pack_size_one_clamped_to_the_cap_and_identity_scale() {
    let bytes = build_vs_draw_bytes(
        &render_state_defaults(),
        1.0f32.to_bits(),
        &D3DMATRIX::IDENTITY,
        &NO_PLANES,
    );
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
    let bytes = build_vs_draw_bytes(
        &rs,
        rs[D3DRS_POINTSIZE as usize],
        &D3DMATRIX::IDENTITY,
        &NO_PLANES,
    );
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
    let bytes = build_vs_draw_bytes(
        &render_state_defaults(),
        1.0f32.to_bits(),
        &D3DMATRIX::IDENTITY,
        &NO_PLANES,
    );
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
    let mut rs = render_state_defaults();
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let bytes = build_vs_draw_bytes(&rs, 1.0f32.to_bits(), &view, &NO_PLANES);
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
    let bytes = build_vs_draw_bytes(
        &rs,
        rs[D3DRS_POINTSIZE as usize],
        &D3DMATRIX::IDENTITY,
        &planes,
    );
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
    let bytes = build_vs_draw_bytes(
        &rs,
        rs[D3DRS_POINTSIZE as usize],
        &D3DMATRIX::IDENTITY,
        &planes,
    );
    assert_eq!(row(&bytes, 6), [0; 4]);
}

#[test]
fn disabled_planes_skip_inversion_and_reenable_reads_the_current_view() {
    let mut state = VsDrawState::new();
    let mut rs = render_state_defaults();
    let mut view = D3DMATRIX::IDENTITY;
    view.m[13] = 0.5;
    for mask in [0, 1 << MAX_CLIP_PLANES] {
        rs[D3DRS_CLIPPLANEENABLE as usize] = mask;
        let bytes = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
        assert_eq!(row(&bytes, 3), bits([0.0, 1.0, 0.0, 0.0]));
        assert_eq!(state.inversions, 0);
    }
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let first = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert_eq!(row(&first, 3), bits([0.0, 1.0, 0.0, -0.5]));
    assert_eq!(state.inversions, 1);
    rs[D3DRS_CLIPPING as usize] = 0;
    view.m[13] = -0.5;
    let disabled = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert_eq!(row(&disabled, 3), bits([0.0, 1.0, 0.0, 0.0]));
    assert_eq!(state.inversions, 1);
    rs[D3DRS_CLIPPING as usize] = 1;
    let enabled = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert_eq!(row(&enabled, 3), bits([0.0, 1.0, 0.0, 0.5]));
    assert_eq!(state.inversions, 2);
}

#[test]
fn point_and_plane_updates_reuse_the_inverse() {
    let mut state = VsDrawState::new();
    let mut rs = render_state_defaults();
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let mut planes = NO_PLANES;
    let view = D3DMATRIX::IDENTITY;
    let first = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &planes);
    for size in 1..8u16 {
        rs[D3DRS_POINTSIZE as usize] = f32::from(size).to_bits();
        planes[1] = [1.0, 2.0, 3.0, f32::from(size)];
        rs[D3DRS_CLIPPLANEENABLE as usize] = 1 << 1;
        let next = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &planes);
        assert_eq!(next[32..96], first[32..96]);
        assert_eq!(lane(&next, 0), f32::from(size).to_bits());
        assert_eq!(row(&next, 6), bits(planes[1]));
    }
    assert_eq!(state.inversions, 1);
}

#[test]
fn every_view_lane_is_keyed_by_its_exact_bits() {
    let mut state = VsDrawState::new();
    let mut rs = render_state_defaults();
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let mut view = D3DMATRIX::IDENTITY;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    for i in 0..16 {
        view.m[i] = f32::from_bits(view.m[i].to_bits() ^ 0x8000_0000);
        let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
        assert_eq!(
            state.inversions,
            i + 2,
            "view lane {i}, including signed zero"
        );
        let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
        assert_eq!(state.inversions, i + 2);
    }
    view.m[0] = f32::from_bits(0x7fc0_0001);
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert_eq!(state.inversions, 18, "identical NaN payload is reusable");
    view.m[0] = f32::from_bits(0x7fc0_0002);
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert_eq!(state.inversions, 19, "different NaN payload is a new key");
}

#[test]
fn enabled_planes_match_uncached_packing_for_general_and_singular_views() {
    let mut state = VsDrawState::new();
    let mut rs = render_state_defaults();
    rs[D3DRS_CLIPPLANEENABLE as usize] = 0b10_0101;
    rs[D3DRS_POINTSIZE as usize] = 3.5f32.to_bits();
    let planes = [[1.0, -2.0, 0.25, 0.5]; MAX_CLIP_PLANES];
    let matrices = [
        D3DMATRIX::IDENTITY,
        D3DMATRIX {
            m: [
                2.0, 0.25, 0.0, 0.125, 0.5, 3.0, 0.75, 0.0, 0.0, 0.5, 4.0, 0.25, 1.0, 2.0, 3.0, 1.0,
            ],
        },
        D3DMATRIX { m: [0.0; 16] },
        D3DMATRIX {
            m: [f32::INFINITY; 16],
        },
    ];
    for view in &matrices {
        let inverse = FfState::inverse(view).unwrap_or(D3DMATRIX::IDENTITY);
        let expected = pack_bytes(
            &rs,
            rs[D3DRS_POINTSIZE as usize],
            &FfState::transpose(&inverse),
            &planes,
        );
        assert_eq!(
            state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], view, &planes),
            expected
        );
        assert_eq!(
            state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], view, &planes),
            expected
        );
    }
    assert_eq!(state.inversions, matrices.len());
}

#[test]
fn independent_devices_and_restored_transforms_keep_their_own_inverse() {
    use mtld3d_types::D3DTS_VIEW;

    use crate::ff_state::FfStateSnapshot;

    let mut first = VsDrawState::new();
    let mut second = VsDrawState::new();
    let mut ff = FfState::new();
    let saved = FfStateSnapshot::from(&ff);
    let mut rs = render_state_defaults();
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let identity = first.build_bytes(
        &rs,
        rs[D3DRS_POINTSIZE as usize],
        &D3DMATRIX::IDENTITY,
        &NO_PLANES,
    );
    let mut translation = D3DMATRIX::IDENTITY;
    translation.m[13] = 0.5;
    assert!(ff.multiply_transform(D3DTS_VIEW, &translation));
    let shifted = first.build_bytes(
        &rs,
        rs[D3DRS_POINTSIZE as usize],
        ff.transform(D3DTS_VIEW).unwrap(),
        &NO_PLANES,
    );
    assert_ne!(shifted, identity);
    assert_eq!(
        second.build_bytes(
            &rs,
            rs[D3DRS_POINTSIZE as usize],
            &D3DMATRIX::IDENTITY,
            &NO_PLANES
        ),
        identity
    );
    saved.restore_into(&mut ff);
    assert_eq!(
        first.build_bytes(
            &rs,
            rs[D3DRS_POINTSIZE as usize],
            ff.transform(D3DTS_VIEW).unwrap(),
            &NO_PLANES
        ),
        identity
    );
    assert_eq!(first.inversions, 3);
    assert_eq!(second.inversions, 1);
}

#[cfg(perf_tracking)]
#[test]
fn perf_outcomes_follow_consumption_and_cached_singular_fallback() {
    let mut state = VsDrawState::new();
    let mut rs = render_state_defaults();
    let mut view = D3DMATRIX::IDENTITY;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Bypass));
    rs[D3DRS_CLIPPLANEENABLE as usize] = 1;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Recompute));
    rs[D3DRS_POINTSIZE as usize] = 2.0f32.to_bits();
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Hit));
    view.m[1] = -0.0;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Recompute));
    view.m = [0.0; 16];
    let first = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Recompute));
    assert_eq!(
        first,
        state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES)
    );
    assert!(matches!(state.last_use(), InverseViewUse::Hit));
    rs[D3DRS_CLIPPING as usize] = 0;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Bypass));
    rs[D3DRS_CLIPPING as usize] = 1;
    let _ = state.build_bytes(&rs, rs[D3DRS_POINTSIZE as usize], &view, &NO_PLANES);
    assert!(matches!(state.last_use(), InverseViewUse::Hit));
}
