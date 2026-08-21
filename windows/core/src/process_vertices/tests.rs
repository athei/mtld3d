use mtld3d_types::{D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DFVF_XYZRHW};

use super::*;

fn viewport() -> D3DVIEWPORT9 {
    D3DVIEWPORT9 {
        x: 0,
        y: 0,
        width: 640,
        height: 480,
        min_z: 0.0,
        max_z: 1.0,
    }
}

#[test]
fn identity_transform_maps_to_screen_coordinates() {
    // A quad at z=0 through an identity WVP maps through a 640x480
    // viewport to (x*320+320, -y*240+240, 0, 1).
    let quad: [[f32; 3]; 4] = [
        [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
    ];
    let mut src = Vec::new();
    for p in &quad {
        for c in p {
            src.extend_from_slice(&c.to_le_bytes());
        }
        src.extend_from_slice(&0xffff_0000u32.to_le_bytes()); // DIFFUSE
    }
    let out = process_vertices(&ProcessVerticesRequest {
        src: &src,
        src_stride: 16,
        src_fvf: D3DFVF_XYZ | D3DFVF_DIFFUSE,
        dst_fvf: D3DFVF_XYZRHW,
        count: 4,
        wvp: D3DMATRIX::IDENTITY,
        viewport: viewport(),
    })
    .expect("transform");
    assert_eq!(out.len(), 4 * 16);
    for (i, p) in quad.iter().enumerate() {
        let base = i * 16;
        let read =
            |o: usize| f32::from_le_bytes(out[base + o..base + o + 4].try_into().unwrap());
        assert!((read(0) - p[0].mul_add(320.0, 320.0)).abs() < 1e-3, "x {i}");
        assert!(
            (read(4) - (-p[1]).mul_add(240.0, 240.0)).abs() < 1e-3,
            "y {i}"
        );
        assert!(read(8).abs() < 1e-3, "z {i}");
        assert!((read(12) - 1.0).abs() < 1e-6, "rhw {i}");
    }
}

#[test]
fn perspective_divide_produces_rhw() {
    // A projection with w = 2 halves the NDC and yields rhw = 0.5.
    let mut wvp = D3DMATRIX::IDENTITY;
    wvp.m[15] = 2.0; // w = x*0 + ... + 1*2 = 2
    let mut src = Vec::new();
    for c in [0.0f32, 0.0, 0.0] {
        src.extend_from_slice(&c.to_le_bytes());
    }
    let out = process_vertices(&ProcessVerticesRequest {
        src: &src,
        src_stride: 12,
        src_fvf: D3DFVF_XYZ,
        dst_fvf: D3DFVF_XYZRHW,
        count: 1,
        wvp,
        viewport: viewport(),
    })
    .expect("transform");
    let rhw = f32::from_le_bytes(out[12..16].try_into().unwrap());
    assert!((rhw - 0.5).abs() < 1e-6, "rhw = {rhw}");
}

#[test]
fn destination_without_position_is_rejected() {
    let src = vec![0u8; 12];
    assert!(
        process_vertices(&ProcessVerticesRequest {
            src: &src,
            src_stride: 12,
            src_fvf: D3DFVF_XYZ,
            dst_fvf: D3DFVF_DIFFUSE,
            count: 1,
            wvp: D3DMATRIX::IDENTITY,
            viewport: viewport(),
        })
        .is_none()
    );
}
