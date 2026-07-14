//! Draw paths beyond the smoke triangle.
//!
//! Phase 1 covers the pre-transformed (XYZRHW) screen-space path used by
//! 2D/UI geometry.

use mtld3d_tests::{DrawIndexedUpParams, Harness, PosColorVertex, RhwVertex};
use mtld3d_types::{
    D3DERR_INVALIDCALL, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DFVF_XYZRHW, D3DPOOL_DEFAULT,
    D3DPT_LINELIST, D3DPT_LINESTRIP, D3DPT_POINTLIST, D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST,
    D3DPT_TRIANGLESTRIP, D3DRS_LIGHTING, D3DUSAGE_WRITEONLY,
};

const MAGENTA: u32 = 0xFFFF_00FF;
const BLACK: u32 = 0xFF00_0000;
const GREEN: u32 = 0xFF00_FF00;

#[test]
fn xyzrhw_quad_maps_to_screen_rect() {
    let h = Harness::new();
    // Stage 0 may carry a binding from an earlier device in this process; this
    // harness is fresh, but make the routing explicit.
    assert_eq!(h.clear_texture(0), 0, "no texture bound");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZRHW | D3DFVF_DIFFUSE), 0, "SetFVF");

    // RHW=1 screen-space quad covering pixels (100,100)..(200,200).
    let quad = [
        RhwVertex {
            x: 100.0,
            y: 100.0,
            z: 0.5,
            rhw: 1.0,
            color: MAGENTA,
        },
        RhwVertex {
            x: 200.0,
            y: 100.0,
            z: 0.5,
            rhw: 1.0,
            color: MAGENTA,
        },
        RhwVertex {
            x: 100.0,
            y: 200.0,
            z: 0.5,
            rhw: 1.0,
            color: MAGENTA,
        },
        RhwVertex {
            x: 200.0,
            y: 200.0,
            z: 0.5,
            rhw: 1.0,
            color: MAGENTA,
        },
    ];

    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &quad),
            0,
            "DrawPrimitiveUP strip"
        );
    });

    assert_eq!(
        h.read_pixel(150, 150),
        MAGENTA,
        "inside the screen-space rect"
    );
    assert_eq!(
        h.read_pixel(50, 50),
        BLACK,
        "outside the rect stays background"
    );
}

#[test]
fn draw_and_present_balance_device_refcount() {
    // A draw + Present must not leak a device reference (e.g. via an implicit
    // render-target / depth-stencil surface bound through the public refcount).
    // Per the D3D9 refcount model the device count after a draw + Present is
    // exactly what it was before, so a surviving reference here is a leak.
    let h = Harness::new();
    arm_diffuse(&h);
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: GREEN,
    };
    let tri = [v(0.0, 0.5), v(0.5, -0.5), v(-0.5, -0.5)];
    let base = h.device_refcount();
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
            0,
            "DrawPrimitiveUP",
        );
    });
    assert_eq!(
        h.device_refcount(),
        base,
        "a draw + Present leaves the device refcount balanced",
    );
}

/// Arm fixed-function diffuse passthrough for clip-space `PosColorVertex` draws.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "SetFVF");
}

#[test]
fn every_primitive_type_draws() {
    let h = Harness::new();
    arm_diffuse(&h);
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: GREEN,
    };
    let points = [v(0.0, 0.0)];
    let line = [v(-0.5, 0.0), v(0.5, 0.0)];
    let strip3 = [v(-0.5, 0.0), v(0.0, 0.5), v(0.5, 0.0)];
    // Each list is sized for one primitive of its kind; the path must accept it.
    for (prim, count, verts) in [
        (D3DPT_POINTLIST, 1u32, &points[..]),
        (D3DPT_LINELIST, 1, &line[..]),
        (D3DPT_LINESTRIP, 1, &line[..]),
        (D3DPT_TRIANGLESTRIP, 1, &strip3[..]),
    ] {
        h.render_once(BLACK, |d| {
            assert_eq!(
                d.draw_primitive_up(prim, count, verts),
                0,
                "primitive {prim} draws"
            );
        });
    }
}

#[test]
fn triangle_fan_draws_as_triangle_list() {
    // Metal has no triangle-fan primitive, so mtld3d expands a fan into a
    // triangle list. A 4-vertex fan (2 triangles) is a diamond covering the
    // screen centre; the corners stay background.
    let h = Harness::new();
    arm_diffuse(&h);
    let fan = [
        PosColorVertex {
            x: 0.0,
            y: 0.6,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 0.6,
            y: 0.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 0.0,
            y: -0.6,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: -0.6,
            y: 0.0,
            z: 0.5,
            color: GREEN,
        },
    ];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLEFAN, 2, &fan),
            0,
            "TRIANGLEFAN draws (expanded to a triangle list)",
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "centre is inside the fan diamond",
    );
    assert_eq!(
        h.read_pixel(10, 10),
        BLACK,
        "corner is outside the fan diamond",
    );
}

#[test]
fn indexed_primitive_up_draws() {
    // DrawIndexedPrimitiveUP feeds inline vertices + an inline index stream; the
    // index data is copied into a transient Metal buffer per draw. The triangle
    // covers the screen centre; the corners stay background.
    let h = Harness::new();
    arm_diffuse(&h);
    let verts = [
        PosColorVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: GREEN,
        },
    ];
    let indices: [u16; 3] = [0, 1, 2];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_indexed_primitive_up(
                &DrawIndexedUpParams {
                    prim: D3DPT_TRIANGLELIST,
                    min_vertex_index: 0,
                    num_vertices: 3,
                    prim_count: 1,
                    index_format: D3DFMT_INDEX16,
                },
                &indices,
                &verts,
            ),
            0,
            "DrawIndexedPrimitiveUP draws",
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "centre is inside the triangle"
    );
    assert_eq!(
        h.read_pixel(10, 10),
        BLACK,
        "corner is outside the triangle"
    );
}

#[test]
fn indexed_primitive_up_triangle_fan_draws() {
    // A fan via DrawIndexedPrimitiveUP expands the index stream into a triangle
    // list (Metal has no fan primitive). A 4-index fan (2 triangles) is a
    // diamond covering the centre.
    let h = Harness::new();
    arm_diffuse(&h);
    let verts = [
        PosColorVertex {
            x: 0.0,
            y: 0.6,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 0.6,
            y: 0.0,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: 0.0,
            y: -0.6,
            z: 0.5,
            color: GREEN,
        },
        PosColorVertex {
            x: -0.6,
            y: 0.0,
            z: 0.5,
            color: GREEN,
        },
    ];
    let indices: [u16; 4] = [0, 1, 2, 3];
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_indexed_primitive_up(
                &DrawIndexedUpParams {
                    prim: D3DPT_TRIANGLEFAN,
                    min_vertex_index: 0,
                    num_vertices: 4,
                    prim_count: 2,
                    index_format: D3DFMT_INDEX16,
                },
                &indices,
                &verts,
            ),
            0,
            "indexed TRIANGLEFAN draws (expanded to a triangle list)",
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "centre is inside the fan diamond"
    );
    assert_eq!(
        h.read_pixel(10, 10),
        BLACK,
        "corner is outside the fan diamond"
    );
}

#[test]
fn process_vertices_is_a_stub() {
    let h = Harness::new();
    assert_eq!(
        h.process_vertices_hr(),
        D3DERR_INVALIDCALL,
        "ProcessVertices is a documented stub",
    );
}

#[test]
fn draw_without_decl_or_fvf_is_invalid_but_a_bound_draw_still_renders() {
    // With neither a vertex declaration nor an FVF bound, the runtime has no
    // way to interpret the vertex stream and a `Draw*` must reject with
    // `D3DERR_INVALIDCALL` per the D3D9 spec. The same draw
    // must still succeed and render once an FVF (and thus an implicit
    // declaration) is bound, proving the guard fires only when both are absent.
    let h = Harness::new();
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: GREEN,
    };
    let tri = [v(0.0, 0.5), v(0.5, -0.5), v(-0.5, -0.5)];
    let stride = u32::try_from(core::mem::size_of::<PosColorVertex>()).expect("stride fits u32");

    // A live stream source is bound throughout, so the only thing missing in
    // the reject case is the vertex layout source.
    let vb = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&tri);
    assert_eq!(h.set_stream_source(0, &vb, 0, stride), 0, "SetStreamSource");

    // (a) No declaration, no FVF -> the draw is invalid.
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(
        h.set_vertex_declaration_null(),
        0,
        "SetVertexDeclaration(NULL)"
    );
    assert_eq!(h.fvf(), 0, "no FVF after SetVertexDeclaration(NULL)");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(
        h.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
        D3DERR_INVALIDCALL,
        "DrawPrimitive with neither decl nor FVF rejects",
    );
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
        D3DERR_INVALIDCALL,
        "DrawPrimitiveUP with neither decl nor FVF rejects",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");

    // (b) An FVF binds an implicit declaration -> a normal draw still renders.
    arm_diffuse(&h);
    h.render_once(BLACK, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive with an FVF bound succeeds",
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "the FVF-bound triangle renders at the screen centre",
    );
}

#[test]
fn bound_buffer_uses_stream_source_stride_not_decl_extent() {
    // A bound `DrawPrimitive` steps the vertex stream by the `SetStreamSource`
    // stride, NOT the vertex declaration's min-extent. When the application's
    // vertex struct is larger than the declared elements (trailing padding),
    // using the min-extent fetches every vertex past the first at the wrong
    // offset and the primitive degenerates — the screen centre would miss.
    use mtld3d_types::{
        D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
        D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DVERTEXELEMENT9,
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PaddedVertex {
        x: f32,
        y: f32,
        z: f32,
        color: u32,
        _pad: [f32; 4],
    }

    let h = Harness::new();
    let elem = |offset: u16, type_: u8, usage: u8| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    };
    // Declaration min-extent is 16 (FLOAT3@0 + D3DCOLOR@12); struct is 32.
    let decl_elems = [
        elem(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        elem(12, D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];
    let decl = h.create_vertex_declaration(&decl_elems);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);

    let v = |x: f32, y: f32| PaddedVertex {
        x,
        y,
        z: 0.5,
        color: GREEN,
        _pad: [0.0; 4],
    };
    // Full-screen quad as a triangle strip: with the wrong stride, verts 1..3
    // decode from the wrong offsets and the quad no longer covers the centre.
    let quad = [v(-1.0, 1.0), v(1.0, 1.0), v(-1.0, -1.0), v(1.0, -1.0)];
    let stride = u32::try_from(core::mem::size_of::<PaddedVertex>()).expect("stride fits u32");
    let vb = h.create_vertex_buffer(stride * 4, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&quad);
    assert_eq!(h.set_stream_source(0, &vb, 0, stride), 0, "SetStreamSource");

    h.render_once(MAGENTA, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLESTRIP, 0, 2),
            0,
            "bound padded-stride DrawPrimitive",
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        GREEN,
        "padded bound vertices must step by the SetStreamSource stride",
    );
}
