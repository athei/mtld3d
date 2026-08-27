//! Vertex/index buffers and buffer-backed draws (`DrawPrimitive` / `DrawIndexedPrimitive`).
//!
//! Plus the stream/index getter round-trip. Streams beyond stream 0 and
//! `SetStreamSourceFreq` are exercised in `streams.rs`.

use mtld3d_tests::{Harness, Vertex, VertexBuffer};
use mtld3d_types::{
    D3D_OK, D3DCULL_NONE, D3DERR_INVALIDCALL, D3DFMT_INDEX16, D3DFVF_DIFFUSE, D3DFVF_XYZ,
    D3DLOCK_DISCARD, D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DRS_CULLMODE, D3DRS_LIGHTING,
    D3DRTYPE_INDEXBUFFER, D3DRTYPE_VERTEXBUFFER, D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY,
};

const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
const BLUE: u32 = 0xFF00_00FF;
const MAGENTA: u32 = 0xFFFF_00FF;
const GREEN: u32 = 0xFF00_FF00;
const RED: u32 = 0xFFFF_0000;

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
    let tri = solid_triangle(GREEN);
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
    assert_eq!(h.read_pixel(320, 280), GREEN, "VB triangle renders green");
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

    vb.lock(0, 0, D3DLOCK_DISCARD).write(&solid_triangle(GREEN));
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(h.read_pixel(320, 280), GREEN, "first fill is green");

    vb.lock(0, 0, D3DLOCK_DISCARD).write(&solid_triangle(RED));
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(h.read_pixel(320, 280), RED, "DISCARD refill shows red");
}

/// `buffer.ignoreLockBounds`: a write past the announced `Lock` range reaches the GPU (VB).
///
/// A `D3DPOOL_DEFAULT` non-`DYNAMIC` buffer is `Staged`: its CPU staging and
/// the device buffer the GPU reads are separate allocations, and only what
/// `Unlock` uploads ever crosses. A few D3D9-era titles write outside the
/// window they named at `Lock` and a real driver never noticed, because the
/// pointer it handed back was into the one allocation the GPU read. With the
/// option on, uploading only the announcement would leave the previous
/// triangle on screen here: the four announced bytes are vertex 0's `x`,
/// which both triangles share.
#[test]
fn staged_vertex_buffer_upload_ignores_the_announced_lock_range() {
    // `buffer.ignoreLockBounds` is off by default, so this test asks for
    // it. The harness process owns its environment and no other thread
    // runs yet; extend the suite-wide config with the option under test.
    let merged = format!(
        "{};buffer.ignoreLockBounds=true",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the
    // harness and with it the config read are only constructed below).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };

    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 3, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    arm_diffuse(&h);
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );

    // Whole-buffer fill, so the device buffer holds a known triangle.
    vb.lock(0, 0, 0).write(&solid_triangle(GREEN));
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive after the whole-buffer fill"
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "whole-buffer fill draws green"
    );

    // Announce four bytes at offset 0, write all three vertices.
    vb.lock(0, 4, 0).write(&solid_triangle(RED));
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive after the narrow-announcement refill"
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        RED,
        "the vertices written past the announced Lock range reached the GPU"
    );
}

/// `buffer.ignoreLockBounds`: a write past the announced `Lock` range reaches the GPU (IB).
///
/// Same shape as the vertex-buffer case, and it needs the same option on.
/// Both index triples start at vertex 0, so the two announced bytes carry no
/// change and an announcement-only upload leaves the first triangle
/// selected.
#[test]
fn staged_index_buffer_upload_ignores_the_announced_lock_range() {
    // `buffer.ignoreLockBounds` is off by default, so this test asks for
    // it. The harness process owns its environment and no other thread
    // runs yet; extend the suite-wide config with the option under test.
    let merged = format!(
        "{};buffer.ignoreLockBounds=true",
        std::env::var("MTLD3D_CONFIG").unwrap_or_default()
    );
    // SAFETY: single-threaded at this point in the test process (the
    // harness and with it the config read are only constructed below).
    unsafe { std::env::set_var("MTLD3D_CONFIG", merged) };

    let h = Harness::new();
    // Two disjoint triangles sharing vertex 0 (top centre): 0-1-2 fills the
    // left of the screen, 0-3-2 the right, with the shared v0-v2 edge as the
    // boundary. Culling off so neither winding matters.
    let verts = [
        Vertex {
            x: 0.0,
            y: 0.9,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: -0.9,
            y: -0.9,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: -0.05,
            y: -0.9,
            z: 0.5,
            color: MAGENTA,
        },
        Vertex {
            x: 0.9,
            y: -0.9,
            z: 0.5,
            color: MAGENTA,
        },
    ];
    let left: [u16; 3] = [0, 1, 2];
    let right: [u16; 3] = [0, 3, 2];
    // Well inside each triangle at y = 400 of the 640x480 backbuffer, where
    // the left one spans x in [69, 306] and the right one x in [306, 571].
    let (in_left, in_right) = ((150, 400), (450, 400));

    let vb = h.create_vertex_buffer(stride() * 4, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(&verts);
    let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    ib.lock(0, 0, 0).write(&left);

    arm_diffuse(&h);
    assert_eq!(
        h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE),
        0,
        "culling off"
    );
    assert_eq!(
        h.set_stream_source(0, &vb, 0, stride()),
        0,
        "SetStreamSource"
    );
    assert_eq!(h.set_indices(&ib), 0, "SetIndices");

    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 1),
            0,
            "DIP after the whole-buffer fill"
        );
    });
    assert_eq!(
        h.read_pixel(in_left.0, in_left.1),
        MAGENTA,
        "indices 0-1-2 fill the left triangle"
    );
    assert_eq!(
        h.read_pixel(in_right.0, in_right.1),
        BLUE,
        "the right triangle is background before the reindex"
    );

    // Announce two bytes at offset 0, write all three indices.
    ib.lock(0, 2, 0).write(&right);
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 1),
            0,
            "DIP after the narrow-announcement reindex"
        );
    });
    assert_eq!(
        h.read_pixel(in_right.0, in_right.1),
        MAGENTA,
        "the indices written past the announced Lock range reached the GPU"
    );
    assert_eq!(
        h.read_pixel(in_left.0, in_left.1),
        BLUE,
        "the left triangle is gone once indices 0-3-2 are the ones drawn"
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

    // Higher streams (1..max_streams) round-trip their binding. A caller that
    // binds a higher stream and reads it back — relying on the binding to
    // outlive its own Release — must see the buffer.
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
}

/// The two triangles `released_backing_*` draws, side by side and disjoint.
///
/// The left one covers pixel (160, 280), the right one (480, 280), so a
/// readback at each says which half of the buffer the GPU is reading.
fn side_by_side_triangles(left: u32, right: u32) -> [Vertex; 6] {
    let tri = |centre: f32, color: u32| {
        [
            Vertex {
                x: centre - 0.4,
                y: -0.5,
                z: 0.5,
                color,
            },
            Vertex {
                x: centre + 0.4,
                y: -0.5,
                z: 0.5,
                color,
            },
            Vertex {
                x: centre,
                y: 0.5,
                z: 0.5,
                color,
            },
        ]
    };
    let [l0, l1, l2] = tri(-0.5, left);
    let [r0, r1, r2] = tri(0.5, right);
    [l0, l1, l2, r0, r1, r2]
}

/// Arm the fixed-function pipeline and bind `vb` as the only stream, culling off.
fn arm_two_triangles(h: &Harness, vb: &VertexBuffer<'_>) {
    arm_diffuse(h);
    assert_eq!(
        h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE),
        0,
        "cull off"
    );
    assert_eq!(
        h.set_stream_source(0, vb, 0, stride()),
        0,
        "SetStreamSource"
    );
}

/// Draw both triangles and read the pixel at the centre of each.
fn draw_and_sample(h: &Harness) -> (u32, u32) {
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 2),
            0,
            "DrawPrimitive of both triangles"
        );
    });
    (h.read_pixel(160, 280), h.read_pixel(480, 280))
}

/// A sub-range refill of a released backing leaves the rest of the buffer alone.
///
/// A `D3DPOOL_DEFAULT` `D3DUSAGE_WRITEONLY` non-`DYNAMIC` buffer releases its
/// CPU staging once an upload has carried every byte: D3D9 promises no
/// readback, and inside a 32-bit title the copy competes with the title's own
/// address space. The next `Lock` re-creates the staging with nothing in it, so
/// the upload has to stay inside the window that `Lock` announced. The right
/// triangle's vertices are rewritten and the left triangle's, which no `Lock`
/// has touched since the release, must still be on the GPU.
#[test]
fn writeonly_default_vertex_buffer_keeps_the_half_a_sub_range_relock_leaves_alone() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 6, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    arm_two_triangles(&h, &vb);

    vb.lock(0, 0, 0)
        .write(&side_by_side_triangles(GREEN, MAGENTA));
    assert_eq!(
        draw_and_sample(&h),
        (GREEN, MAGENTA),
        "the whole-buffer fill draws both triangles"
    );

    // Announce the second triangle's three vertices and rewrite only those.
    let half = stride() * 3;
    vb.lock(half, half, 0)
        .write(&side_by_side_triangles(RED, RED)[3..]);
    assert_eq!(
        draw_and_sample(&h),
        (GREEN, RED),
        "the refilled half changes and the untouched half keeps its device bytes"
    );
}

/// A `SizeToLock` of zero past offset zero refills the tail without erasing the head.
///
/// `SizeToLock == 0` names no narrower window than "to the end of the buffer",
/// so an ordinary upload widens to the whole buffer rather than trusting it.
/// A backing re-created after a release holds zeros outside what this very
/// `Lock` writes, so widening there would push those zeros over the head of
/// the device buffer: the announced tail is the most the upload may carry.
#[test]
fn writeonly_default_vertex_buffer_keeps_its_head_across_a_zero_size_tail_relock() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(stride() * 6, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    arm_two_triangles(&h, &vb);

    vb.lock(0, 0, 0)
        .write(&side_by_side_triangles(GREEN, MAGENTA));
    assert_eq!(
        draw_and_sample(&h),
        (GREEN, MAGENTA),
        "the whole-buffer fill draws both triangles"
    );

    // `SizeToLock == 0` from the halfway offset: the tail, announced the way
    // D3D9 documents it.
    vb.lock(stride() * 3, 0, 0)
        .write(&side_by_side_triangles(RED, RED)[3..]);
    assert_eq!(
        draw_and_sample(&h),
        (GREEN, RED),
        "the head of the device buffer survives a tail-only refill"
    );
}

/// `Reset` after a buffer released its backing, and buffers still work after it.
///
/// A released backing leaves the bytes on the GPU alone, so nothing can
/// re-upload them if the Metal device is recreated under them; that trade is
/// what the release path warns about once. `Reset` itself keeps the device and
/// its buffers, and D3D9 makes the application release its default-pool
/// resources first, so the outcome a caller sees is an ordinary `Reset`
/// followed by ordinary buffer draws.
#[test]
fn reset_after_a_released_backing_leaves_buffer_draws_working() {
    let h = Harness::new();
    let (width, height) = h.dims();
    let vb = h.create_vertex_buffer(stride() * 6, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    arm_two_triangles(&h, &vb);
    vb.lock(0, 0, 0)
        .write(&side_by_side_triangles(GREEN, MAGENTA));
    assert_eq!(
        draw_and_sample(&h),
        (GREEN, MAGENTA),
        "the fill draws before the Reset"
    );

    // `Reset` rejects any outstanding application reference to a
    // `D3DPOOL_DEFAULT` resource, so the buffer goes first.
    drop(vb);
    assert_eq!(h.reset(width, height), D3D_OK, "Reset at the same size");

    let vb = h.create_vertex_buffer(stride() * 6, D3DUSAGE_WRITEONLY, FVF, D3DPOOL_DEFAULT);
    arm_two_triangles(&h, &vb);
    vb.lock(0, 0, 0).write(&side_by_side_triangles(RED, GREEN));
    assert_eq!(
        draw_and_sample(&h),
        (RED, GREEN),
        "a buffer created after the Reset fills and draws"
    );
}
