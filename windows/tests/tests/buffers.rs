//! Vertex/index buffers and buffer-backed draws (`DrawPrimitive` / `DrawIndexedPrimitive`).
//!
//! Plus the stream/index getter round-trip and the `SetStreamSourceFreq` stub
//! contract.

use mtld3d_tests::{Harness, Vertex};
use mtld3d_types::{
    D3D_OK, D3DERR_INVALIDCALL, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DLOCK_DISCARD,
    D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DRTYPE_INDEXBUFFER,
    D3DRTYPE_VERTEXBUFFER, D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY,
};

const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
const BLUE: u32 = 0xFF00_00FF;
const MAGENTA: u32 = 0xFFFF_00FF;

fn stride() -> u32 {
    u32::try_from(size_of::<Vertex>()).expect("vertex stride fits u32")
}

const fn solid_triangle(color: u32) -> [Vertex; 3] {
    [
        Vertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color,
        },
    ]
}

/// Drive the fixed-function pipeline so a draw shows the vertex diffuse colour.
fn arm_diffuse(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(FVF), 0, "SetFVF");
}

#[test]
fn draw_primitive_from_vertex_buffer() {
    let h = Harness::new();
    let tri = solid_triangle(0xFF00_FF00);
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&tri);

    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive"
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFF00_FF00,
        "VB triangle renders green"
    );
}

#[test]
fn draw_indexed_primitive_from_buffers() {
    let h = Harness::new();
    // A quad as four corners + two triangles of indices.
    let verts = [
        Vertex {
            x: -0.5,
            y: 0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: 0.5,
            y: 0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
            color: MAGENTA,
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 1, 3, 2];

    let vb = h.create_vertex_buffer(stride() * 4, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&verts);
    let ib = h.create_index_buffer(12, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    ib.lock(0, 0, 0).write(&indices);

    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    assert_eq!(h.set_indices(&ib), 0, "SetIndices");
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
            0,
            "DIP"
        );
    });
    assert_eq!(
        h.read_pixel(320, 240),
        MAGENTA,
        "indexed quad renders magenta"
    );
    assert_eq!(h.read_pixel(10, 10), BLUE, "outside quad stays background");
}

#[test]
fn dynamic_vertex_buffer_discard_refill() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(
        stride() * 3,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        FVF,
        D3DPOOL_DEFAULT,
    );
    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );

    vb.lock(0, 0, D3DLOCK_DISCARD)
        .write(&solid_triangle(0xFF00_FF00));
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(h.read_pixel(320, 280), 0xFF00_FF00, "first fill is green");

    vb.lock(0, 0, D3DLOCK_DISCARD)
        .write(&solid_triangle(0xFFFF_0000));
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(
        h.read_pixel(320, 280),
        0xFFFF_0000,
        "DISCARD refill shows red"
    );
}

#[test]
fn vertex_buffer_desc_round_trips() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    let (hr, desc) = vb.desc();
    assert_eq!(hr, 0, "GetDesc");
    assert_eq!(desc.size, stride() * 3, "size");
    assert_eq!(desc.fvf, FVF, "fvf");
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "pool");
    assert_eq!(desc.resource_type, D3DRTYPE_VERTEXBUFFER, "resource type");
    assert_ne!(
        desc.usage & D3DUSAGE_WRITEONLY,
        0,
        "WRITEONLY usage retained"
    );
}

#[test]
fn index_buffer_desc_round_trips() {
    let h = Harness::new();
    let ib = h.create_index_buffer(12, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    let (hr, desc) = ib.desc();
    assert_eq!(hr, 0, "GetDesc");
    assert_eq!(desc.size, 12, "size");
    assert_eq!(desc.format, D3DFMT_INDEX16, "format");
    assert_eq!(desc.pool, D3DPOOL_DEFAULT, "pool");
    assert_eq!(desc.resource_type, D3DRTYPE_INDEXBUFFER, "resource type");
}

#[test]
fn stream_source_higher_index_roundtrips() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);

    // Higher streams (1..max_streams) are accepted and round-trip their binding,
    // even though only stream 0 is ever rendered. A caller that binds a higher
    // stream and reads it back — relying on the binding to outlive its own
    // Release — must see the buffer.
    assert_eq!(
        h.set_stream_source(1, &vb, 8, stride()),
        D3D_OK,
        "stream 1 accepted",
    );
    let (hr, got, offset, stride_out) = h.get_stream_source(1);
    assert_eq!(hr, D3D_OK, "GetStreamSource(1) bound");
    assert_eq!(
        got.expect("stream 1 bound").as_ptr(),
        vb.as_ptr(),
        "GetStreamSource(1) returns the bound VB",
    );
    assert_eq!(
        (offset, stride_out),
        (8, stride()),
        "stream 1 offset/stride round-trip",
    );

    // A NULL bind clears the buffer but retains offset/stride (same quirk as
    // stream 0).
    assert_eq!(h.set_stream_source_null(1, 0, 0), D3D_OK);
    let (hr, got, offset, stride_out) = h.get_stream_source(1);
    assert_eq!(hr, D3D_OK, "GetStreamSource(1) after NULL bind");
    assert!(got.is_none(), "stream 1 cleared");
    assert_eq!(
        (offset, stride_out),
        (8, stride()),
        "stream 1 offset/stride retained across a NULL bind",
    );

    // A stream index at or beyond max_streams (16) is out of range → INVALIDCALL.
    assert_eq!(
        h.set_stream_source(16, &vb, 0, stride()),
        D3DERR_INVALIDCALL,
        "stream index >= max_streams rejected",
    );
}

#[test]
fn buffer_getters_roundtrip() {
    let h = Harness::new();

    // Unbound: both succeed and report "nothing bound" (NULL out-pointer).
    let (hr, vb, offset, stride_out) = h.get_stream_source(0);
    assert_eq!(hr, D3D_OK, "GetStreamSource(0) unbound");
    assert!(vb.is_none(), "no stream bound");
    assert_eq!((offset, stride_out), (0, 0), "unbound offset/stride zeroed");
    let (hr, ib) = h.get_indices();
    assert_eq!(hr, D3D_OK, "GetIndices unbound");
    assert!(ib.is_none(), "no index buffer bound");

    // Bound: the getter hands back the same object plus the offset/stride.
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    assert_eq!(h.set_stream_source(0, &vb, 4, stride()), D3D_OK);
    let (hr, got, offset, stride_out) = h.get_stream_source(0);
    assert_eq!(hr, D3D_OK, "GetStreamSource(0) bound");
    assert_eq!(
        got.expect("stream 0 bound").as_ptr(),
        vb.as_ptr(),
        "GetStreamSource returns the bound VB",
    );
    assert_eq!(
        (offset, stride_out),
        (4, stride()),
        "offset/stride round-trip"
    );

    let ib = h.create_index_buffer(64, D3DUSAGE_DYNAMIC, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    assert_eq!(h.set_indices(&ib), D3D_OK);
    let (hr, got) = h.get_indices();
    assert_eq!(hr, D3D_OK, "GetIndices bound");
    assert_eq!(
        got.expect("index buffer bound").as_ptr(),
        ib.as_ptr(),
        "GetIndices returns the bound IB",
    );

    // Clearing the stream source with a NULL buffer retains the previous
    // offset/stride (a D3D9 quirk): GetStreamSource reports a NULL buffer but
    // the last non-null stride.
    assert_eq!(h.set_stream_source_null(0, 0, 0), D3D_OK);
    let (hr, got, offset, stride_out) = h.get_stream_source(0);
    assert_eq!(hr, D3D_OK, "GetStreamSource(0) after NULL bind");
    assert!(got.is_none(), "stream 0 cleared");
    assert_eq!(
        (offset, stride_out),
        (4, stride()),
        "offset/stride retained across a NULL stream-source bind",
    );

    // SetStreamSourceFreq remains unimplemented.
    assert_eq!(
        h.set_stream_source_freq(0, 1),
        D3DERR_INVALIDCALL,
        "SetStreamSourceFreq stub",
    );
}
