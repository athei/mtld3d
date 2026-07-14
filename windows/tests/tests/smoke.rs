//! Spike coverage: the two simplest end-to-end paths, ported to the shared `Harness`.
//!
//! Validates the nextest-under-wine runner before the full suite.

use mtld3d_tests::{Harness, Rgba8, Vertex, assert_pixel_eq};
use mtld3d_types::{D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DPT_TRIANGLELIST, D3DRS_LIGHTING};

const BLUE: u32 = 0xFF00_00FF;

#[test]
fn clear_fills_backbuffer() {
    let h = Harness::new();
    h.render_once(BLUE, |_| {});

    assert_pixel_eq(h.read_pixel(10, 10), BLUE, "corner after clear");
    assert_pixel_eq(h.read_pixel(320, 240), BLUE, "center after clear");
}

#[test]
fn draw_primitive_up_triangle() {
    let h = Harness::new();
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
    // Lighting defaults ON in D3D9. These vertices carry a diffuse colour but no
    // normal, so the lit path emits only the normal-independent material ambient
    // + emissive — for the default (zero) material that is black, discarding the
    // vertex colour. Disable lighting to exercise the unlit vertex-colour
    // (Gouraud) path this test is checking.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");

    let verts = [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color: 0xFFFF_0000,
        }, // top, red
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: 0xFF00_FF00,
        }, // right, green
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: 0xFF00_00FF,
        }, // left, blue
    ];

    h.render_once(BLUE, |dev| {
        assert_eq!(
            dev.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &verts),
            0,
            "DrawPrimitiveUP",
        );
    });

    // Corner stays background blue; the triangle covers the lower-centre.
    assert_pixel_eq(h.read_pixel(10, 10), BLUE, "corner outside triangle");

    let center = Rgba8::from_pixel(h.read_pixel(320, 280));
    assert_ne!(
        center.to_pixel(),
        BLUE,
        "center should be inside the triangle"
    );
    assert!(
        center.r > 0 && center.g > 0 && center.b > 0,
        "center should interpolate the three vertex colours, got {center:?}",
    );
}
