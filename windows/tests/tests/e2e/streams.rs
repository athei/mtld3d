//! Vertex streams beyond stream 0 and hardware instancing.
//!
//! A declaration split across two streams drawn through the programmable
//! pipeline, the `SetStreamSourceFreq` contract, instanced indexed draws
//! (count, step rate, and the non-indexed exemption), and state-block capture
//! of stream bindings and frequencies.

use mtld3d_tests::{DrawIndexedUpParams, GuardedSlice, Harness, PosVertex, assert_pixel_approx};
use mtld3d_types::{
    D3D_OK, D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DERR_INVALIDCALL,
    D3DFMT_INDEX16, D3DPOOL_DEFAULT, D3DPT_TRIANGLEFAN, D3DPT_TRIANGLELIST, D3DSBT_ALL,
    D3DSTREAMSOURCE_INDEXEDDATA, D3DSTREAMSOURCE_INSTANCEDATA, D3DUSAGE_DYNAMIC,
    D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9, D3DVIEWPORT9,
};

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;

/// `vs_2_0`: `dcl_position v0; dcl_texcoord v2; mov oPos, v0; mov oD0, v2;`
const VS_POS_COLOR_TEXCOORD: [u32; 14] = [
    0xFFFE_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (31) | (2 << 24),
    u32::from_ne_bytes([D3DDECLUSAGE_TEXCOORD, 0, 0, 0]),
    (1 << 28) | (0xF << 16) | 2,
    (1) | (2 << 24),
    (4 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    (1) | (2 << 24),
    (5 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16) | 2,
    0x0000_FFFF,
];

/// `vs_2_0`: `dcl_position v0; dcl_color v1; mov oPos, v0; mov oD0, v1;`
const VS_POS_COLOR: [u32; 14] = [
    0xFFFE_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (31) | (2 << 24),
    u32::from_ne_bytes([D3DDECLUSAGE_COLOR, 0, 0, 0]),
    (1 << 28) | (0xF << 16) | 1,
    (1) | (2 << 24),
    (4 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    (1) | (2 << 24),
    (5 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16) | 1,
    0x0000_FFFF,
];

/// `vs_2_0`: `dcl_position v0; dcl_texcoord v1; mov oPos, v0; add oPos.xy, v0, v1;`
///
/// The per-instance offset rides TEXCOORD0; only `xy` are added so the
/// padded `w = 1` of the two `FLOAT3` inputs does not double.
const VS_INSTANCED: [u32; 15] = [
    0xFFFE_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (31) | (2 << 24),
    u32::from_ne_bytes([D3DDECLUSAGE_TEXCOORD, 0, 0, 0]),
    (1 << 28) | (0xF << 16) | 1,
    (1) | (2 << 24),
    (4 << 28) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    (2) | (3 << 24),
    (4 << 28) | (0x3 << 16),
    (1 << 28) | (0xE4 << 16),
    (1 << 28) | (0xE4 << 16) | 1,
    0x0000_FFFF,
];

/// `ps_2_0`: `dcl v0; mov oC0, v0;`
const PS_DIFFUSE: [u32; 8] = [
    0xFFFF_0200,
    (31) | (2 << 24),
    0x0000_0000,
    (1 << 28) | (0xF << 16),
    (1) | (2 << 24),
    (1 << 11) | (0xF << 16),
    (1 << 28) | (0xE4 << 16),
    0x0000_FFFF,
];

/// `ps_2_0`: `mov oC0, c0;` (c0 supplied via the constant buffer).
const PS_CONST: [u32; 5] = [
    0xFFFF_0200,
    (1) | (2 << 24),
    (1 << 11) | (0xF << 16),
    (2 << 28) | (0xE4 << 16),
    0x0000_FFFF,
];

const fn end() -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream: D3DDECL_END_STREAM,
        offset: 0,
        type_: D3DDECLTYPE_UNUSED,
        method: 0,
        usage: 0,
        usage_index: 0,
    }
}

const fn element(stream: u16, type_: u8, usage: u8) -> D3DVERTEXELEMENT9 {
    D3DVERTEXELEMENT9 {
        stream,
        offset: 0,
        type_,
        method: 0,
        usage,
        usage_index: 0,
    }
}

/// POSITION float3 on stream 0, COLOR d3dcolor on stream 1.
const fn pos_stream0_color_stream1() -> [D3DVERTEXELEMENT9; 3] {
    [
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        element(1, D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR),
        end(),
    ]
}

/// POSITION float3 on stream 0, TEXCOORD0 float3 on stream 1.
const fn pos_stream0_offset_stream1() -> [D3DVERTEXELEMENT9; 3] {
    [
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        element(1, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_TEXCOORD),
        end(),
    ]
}

fn stride_of<T>() -> u32 {
    u32::try_from(size_of::<T>()).expect("stride fits u32")
}

const fn centered_triangle() -> [PosVertex; 3] {
    [
        PosVertex {
            x: 0.0,
            y: 0.5,
            z: 0.5,
        },
        PosVertex {
            x: 0.5,
            y: -0.5,
            z: 0.5,
        },
        PosVertex {
            x: -0.5,
            y: -0.5,
            z: 0.5,
        },
    ]
}

/// A position and a colour, 16 bytes, the stride the short-stride tests bind.
#[repr(C)]
#[derive(Clone, Copy)]
struct PackedVertex {
    x: f32,
    y: f32,
    z: f32,
    color: u32,
}

/// A declaration split across two streams reaches a programmable VS intact.
///
/// Position comes from stream 0, the diffuse colour from stream 1 (a
/// separate vertex buffer with its own stride); the VS routes the colour to
/// `oD0` and the PS returns it.
#[test]
fn two_stream_declaration_drives_programmable_draw() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&pos_stream0_color_stream1());
    let vs = h.create_vertex_shader(&VS_POS_COLOR);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");

    let tri = centered_triangle();
    let colors = [GREEN; 3];
    let positions = h.create_vertex_buffer(
        stride_of::<PosVertex>() * 3,
        D3DUSAGE_WRITEONLY,
        0,
        D3DPOOL_DEFAULT,
    );
    positions.lock(0, 0, 0).write(&tri);
    let diffuse = h.create_vertex_buffer(4 * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    diffuse.lock(0, 0, 0).write(&colors);
    assert_eq!(
        h.set_stream_source(0, &positions, 0, stride_of::<PosVertex>()),
        D3D_OK
    );
    assert_eq!(h.set_stream_source(1, &diffuse, 0, 4), D3D_OK);

    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "colour from stream 1 reaches the pixel"
    );
}

/// Sparse declaration streams skip missing bindings and ignore undeclared ones.
#[test]
fn sparse_declared_streams_preserve_missing_binding_behavior() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&[
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_TEXCOORD),
        element(3, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        element(7, D3DDECLTYPE_D3DCOLOR, D3DDECLUSAGE_COLOR),
        end(),
    ]);
    let vs = h.create_vertex_shader(&VS_POS_COLOR);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), D3D_OK);
    assert_eq!(h.set_vertex_shader(&vs), D3D_OK);
    assert_eq!(h.set_pixel_shader(&ps), D3D_OK);
    let positions = h.create_vertex_buffer(
        stride_of::<PosVertex>() * 3,
        D3DUSAGE_WRITEONLY,
        0,
        D3DPOOL_DEFAULT,
    );
    positions.lock(0, 0, 0).write(&centered_triangle());
    let diffuse = h.create_vertex_buffer(4 * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    diffuse.lock(0, 0, 0).write(&[GREEN; 3]);
    assert_eq!(
        h.set_stream_source(3, &positions, 0, stride_of::<PosVertex>()),
        D3D_OK
    );
    assert_eq!(h.set_stream_source(7, &diffuse, 0, 4), D3D_OK);
    // Stream 0 is declared but unbound; stream 15 is bound but undeclared.
    assert_eq!(h.set_stream_source_null(0, 0, 0), D3D_OK);
    assert_eq!(
        h.set_stream_source(15, &positions, 0, stride_of::<PosVertex>()),
        D3D_OK
    );
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), D3D_OK);
    });
    assert_eq!(h.read_pixel(320, 280), GREEN);

    assert_eq!(h.set_stream_source_null(7, 0, 0), D3D_OK);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), D3D_OK);
    });
    assert_eq!(h.read_pixel(320, 280), 0, "missing colour reads zeros");

    assert_eq!(h.set_stream_source_null(3, 0, 0), D3D_OK);
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            D3DERR_INVALIDCALL
        );
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
            D3D_OK
        );
    });
    assert_eq!(h.read_pixel(320, 280), BLUE, "no declared stream is bound");
}

/// A stride below an unconsumed declaration tail still fetches every vertex.
///
/// A shared declaration can carry trailing elements only other shaders read;
/// a mesh drawn with a shader that ignores them legitimately binds a buffer
/// packed at the span of the consumed elements alone. The vertex fetch must
/// step by that bound stride: covering the unconsumed tail instead would
/// read every vertex past the first from the wrong offset.
#[test]
fn stride_below_an_unconsumed_decl_tail_still_fetches_vertices() {
    // POSITION @0 and COLOR @12 are consumed by VS_POS_COLOR; the TEXCOORD0
    // FLOAT3 tail at 16..28 is not, and the bound buffer does not carry it.
    let elements = [
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 16,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 0,
        },
        end(),
    ];

    let h = Harness::new();
    let decl = h.create_vertex_declaration(&elements);
    let vs = h.create_vertex_shader(&VS_POS_COLOR);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");

    let tri = centered_triangle();
    let packed: Vec<PackedVertex> = tri
        .iter()
        .map(|p| PackedVertex {
            x: p.x,
            y: p.y,
            z: p.z,
            color: GREEN,
        })
        .collect();
    let stride = stride_of::<PackedVertex>();
    let vb = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(packed.as_slice());
    assert_eq!(h.set_stream_source(0, &vb, 0, stride), D3D_OK);

    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "vertices step by the bound stride, not the unconsumed tail's extent"
    );
}

/// Crossing attributes render from dynamic and released static backing, offsets and UP data.
///
/// The shader reads color at byte 28 while position steps every 16 bytes, so
/// each vertex shows the next one's colour field. Prefix vertices, a nonzero
/// stream offset and start/base indices keep the addressing honest: the
/// vertex before the drawn ones carries red, which a crossing binding that
/// dropped the stream offset would read. A UP draw supplies only its own
/// vertices, so the last one's crossing colour reads zero.
#[test]
fn stride_below_a_consumed_attribute_still_places_the_triangle() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&crossing_color_elements());
    let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let mut packed = packed_triangle(GREEN).to_vec();
    packed.push(PackedVertex {
        x: 5.0,
        y: 5.0,
        z: 0.0,
        color: GREEN,
    });
    let stride = stride_of::<PackedVertex>();
    for usage in [D3DUSAGE_WRITEONLY | D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY] {
        let mut with_prefix = vec![
            PackedVertex {
                x: 5.0,
                y: 5.0,
                z: 0.0,
                color: BLUE
            };
            2
        ];
        with_prefix.extend_from_slice(&packed);
        with_prefix[2].color = RED;
        let vb = h.create_vertex_buffer(stride * 6, usage, 0, D3DPOOL_DEFAULT);
        vb.lock(0, 0, 0).write(with_prefix.as_slice());
        assert_eq!(h.set_stream_source(0, &vb, stride, stride), D3D_OK);
        // The readback retires the full upload; the static WRITEONLY backing can then be released.
        for _ in 0..2 {
            h.render_once(BLUE, |d| {
                assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 1, 1), 0);
            });
            assert_eq!(
                h.read_pixel(320, 280),
                GREEN,
                "bound positions and crossing color, usage {usage}"
            );
        }
        let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
        ib.lock(0, 0, 0).write(&[2u16, 3, 4]);
        assert_eq!(h.set_indices(&ib), 0);
        h.render_once(BLUE, |d| {
            assert_eq!(
                d.draw_indexed_primitive(D3DPT_TRIANGLELIST, -1, 2, 3, 0, 1),
                0
            );
        });
        assert_eq!(
            h.read_pixel(320, 280),
            GREEN,
            "indexed negative base plus stream offset"
        );
    }
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &packed[..3]), 0);
    });
    assert_last_crossing_color_reads_zero(&h, "UP");
    let mut indexed = vec![
        PackedVertex {
            x: 5.0,
            y: 5.0,
            z: 0.0,
            color: BLUE
        };
        2
    ];
    indexed.extend_from_slice(&packed[..3]);
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive_up(
                &DrawIndexedUpParams {
                    prim: D3DPT_TRIANGLELIST,
                    min_vertex_index: 2,
                    num_vertices: 3,
                    prim_count: 1,
                    index_format: D3DFMT_INDEX16,
                },
                &[2u16, 3, 4],
                indexed.as_slice()
            ),
            0
        );
    });
    assert_last_crossing_color_reads_zero(&h, "indexed UP with a nonzero minimum");
    let mut large = Vec::new();
    for _ in 0..90 {
        large.extend_from_slice(&packed[..3]);
    }
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 90, large.as_slice()),
            0
        );
    });
    assert_last_crossing_color_reads_zero(&h, "oversized UP through the upload ring");
}

/// Read a centred triangle near its first and its last vertex.
///
/// The first vertex's crossing colour is green; the last one's lies past the
/// vertices a UP draw supplies and reads zero, which shows as black.
fn assert_last_crossing_color_reads_zero(h: &Harness, context: &str) {
    assert_pixel_approx(
        h.read_pixel(320, 130) | 0xFF00_0000,
        GREEN,
        0x30,
        &format!("{context}: near the first vertex"),
    );
    assert_pixel_approx(
        h.read_pixel(170, 355) | 0xFF00_0000,
        0xFF00_0000,
        0x30,
        &format!("{context}: near the last vertex"),
    );
}

/// A crossing stream shares the draw with a normal stream at another slot.
///
/// Color comes from advanced bytes of stream 0 and position from stream 1; remapping
/// must not overwrite the normal position binding or its stride.
#[test]
fn crossing_color_stream_keeps_a_regular_second_stream() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&[
        D3DVERTEXELEMENT9 {
            stream: 1,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 16,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 0,
        },
        end(),
    ]);
    let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let pos = h.create_vertex_buffer(36, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    pos.lock(0, 0, 0).write(&centered_triangle());
    let color = h.create_vertex_buffer(28, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    color
        .lock(0, 0, 0)
        .write(&[BLUE, BLUE, BLUE, BLUE, GREEN, GREEN, GREEN]);
    assert_eq!(h.set_stream_source(0, &color, 0, 4), 0);
    assert_eq!(h.set_stream_source(1, &pos, 0, 12), 0);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "crossing color keeps stream 1's position"
    );
}

/// An instanced crossing color coexists with a normal position stream.
///
/// Both a divisor of two and a zero-rate Constant stream retain their stepping
/// after moving the color's byte offset into a Metal buffer binding.
#[test]
fn crossing_instance_stream_keeps_its_divisor_and_constant_step() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&[
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        D3DVERTEXELEMENT9 {
            stream: 1,
            offset: 16,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 0,
        },
        end(),
    ]);
    let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let pos = h.create_vertex_buffer(36, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    pos.lock(0, 0, 0).write(&centered_triangle());
    let colors = h.create_vertex_buffer(24, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    colors
        .lock(0, 0, 0)
        .write(&[BLUE, BLUE, BLUE, BLUE, RED, GREEN]);
    assert_eq!(h.set_stream_source(0, &pos, 0, 12), 0);
    assert_eq!(h.set_stream_source(1, &colors, 0, 4), 0);
    assert_eq!(
        h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 3),
        0
    );
    let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    ib.lock(0, 0, 0).write(&[0u16, 1, 2]);
    assert_eq!(h.set_indices(&ib), 0);
    // Three instances at one place, the last on top: at divisor 2 it reads
    // element 1, a Constant stream element 0.
    for (rate, expected) in [(2, GREEN), (0, RED)] {
        assert_eq!(
            h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | rate),
            0
        );
        for _ in 0..2 {
            h.render_once(BLUE, |d| {
                assert_eq!(
                    d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
                    0
                );
            });
            assert_eq!(
                h.read_pixel(320, 280),
                expected,
                "crossing instance color with rate {rate}"
            );
        }
    }
}

/// The position at byte 0 and a colour read at byte 28 of a 16-byte vertex.
const fn crossing_color_elements() -> [D3DVERTEXELEMENT9; 3] {
    [
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 28,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_TEXCOORD,
            usage_index: 0,
        },
        end(),
    ]
}

/// The centred triangle with every colour field set to `color`.
fn packed_triangle(color: u32) -> [PackedVertex; 3] {
    centered_triangle().map(|p| PackedVertex {
        x: p.x,
        y: p.y,
        z: p.z,
        color,
    })
}

/// Each UP draw reads no byte past the vertices it supplies.
///
/// D3D9 promises `count * stride` readable bytes behind the pointer, and the
/// vertices here end where a no-access page begins. The last vertex's
/// crossing colour lies past them, so it reads zero instead of faulting, for
/// `DrawPrimitiveUP` as a list and as a fan and for `DrawIndexedPrimitiveUP`.
#[test]
fn up_draws_read_no_vertex_past_the_ones_they_supply() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&crossing_color_elements());
    let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let guarded = GuardedSlice::new(&packed_triangle(GREEN));
    for prim in [D3DPT_TRIANGLELIST, D3DPT_TRIANGLEFAN] {
        h.render_once(BLUE, |d| {
            assert_eq!(d.draw_primitive_up(prim, 1, guarded.as_slice()), 0);
        });
        assert_last_crossing_color_reads_zero(&h, &format!("DrawPrimitiveUP type {prim}"));
    }
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_indexed_primitive_up(
                &DrawIndexedUpParams {
                    prim: D3DPT_TRIANGLELIST,
                    min_vertex_index: 0,
                    num_vertices: 3,
                    prim_count: 1,
                    index_format: D3DFMT_INDEX16,
                },
                &[0u16, 1, 2],
                guarded.as_slice()
            ),
            0
        );
    });
    assert_last_crossing_color_reads_zero(&h, "DrawIndexedPrimitiveUP");
}

/// A later `Lock` in the frame leaves an indexed crossing draw its vertices.
///
/// The draw reads a `Staged` buffer through an index buffer, so it records
/// its read range as reaching the end of the buffer. The crossing colour
/// widens a finite range but must leave that one alone: a range cut to the
/// crossing tail (the first 16 bytes) would miss the `Lock` of bytes 16 to 64
/// that follows, which then reaches the earlier draw.
#[test]
fn a_staged_indexed_crossing_draw_keeps_its_vertices_from_a_later_lock() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&crossing_color_elements());
    let vs = h.create_vertex_shader(&VS_POS_COLOR_TEXCOORD);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let stride = stride_of::<PackedVertex>();
    let mut vertices = packed_triangle(GREEN).to_vec();
    vertices.push(PackedVertex {
        x: 5.0,
        y: 5.0,
        z: 0.0,
        color: GREEN,
    });
    let vb = h.create_vertex_buffer(stride * 4, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(vertices.as_slice());
    let ib = h.create_index_buffer(6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
    ib.lock(0, 0, 0).write(&[0u16, 1, 2]);
    assert_eq!(h.set_stream_source(0, &vb, 0, stride), D3D_OK);
    assert_eq!(h.set_indices(&ib), 0);
    h.render_once(BLUE, |d| {
        let viewport = |x| D3DVIEWPORT9 {
            x,
            y: 0,
            width: 200,
            height: 480,
            min_z: 0.0,
            max_z: 1.0,
        };
        assert_eq!(d.set_viewport(&viewport(0)), D3D_OK);
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
            0
        );
        for vertex in &mut vertices[1..] {
            vertex.color = RED;
        }
        vb.lock(stride, 3 * stride, 0).write(&vertices[1..]);
        assert_eq!(d.set_viewport(&viewport(220)), D3D_OK);
        assert_eq!(
            d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 3, 0, 1),
            0
        );
    });
    assert_eq!(
        h.read_pixel(100, 280),
        GREEN,
        "the earlier draw keeps the colours it was drawn with"
    );
    assert_eq!(
        h.read_pixel(320, 280),
        RED,
        "the refill reaches the later draw"
    );
}

/// A stream offset off a four-byte boundary still draws when nothing crosses.
///
/// Neither `SetStreamSource` nor the draw rejects such an offset, and only
/// the extra binding of an attribute past its stride is held to Metal's
/// alignment.
#[test]
fn a_stream_offset_off_a_four_byte_boundary_still_draws() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&[
        element(0, D3DDECLTYPE_FLOAT3, D3DDECLUSAGE_POSITION),
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        end(),
    ]);
    let vs = h.create_vertex_shader(&VS_POS_COLOR);
    let ps = h.create_pixel_shader(&PS_DIFFUSE);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    assert_eq!(h.set_vertex_shader(&vs), 0);
    assert_eq!(h.set_pixel_shader(&ps), 0);
    let stride = stride_of::<PackedVertex>();
    let mut bytes = vec![0u8; 2];
    for vertex in packed_triangle(GREEN) {
        bytes.extend_from_slice(&vertex.x.to_le_bytes());
        bytes.extend_from_slice(&vertex.y.to_le_bytes());
        bytes.extend_from_slice(&vertex.z.to_le_bytes());
        bytes.extend_from_slice(&vertex.color.to_le_bytes());
    }
    let length = u32::try_from(bytes.len()).expect("a small buffer");
    let vb = h.create_vertex_buffer(length, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    vb.lock(0, 0, 0).write(bytes.as_slice());
    assert_eq!(h.set_stream_source(0, &vb, 2, stride), D3D_OK);
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0);
    });
    assert_eq!(h.read_pixel(320, 280), GREEN, "drawn from offset 2");
}

/// A zero stride feeds every vertex the element at the stream offset.
///
/// D3D9 defines a `SetStreamSource` stride of 0 as one element for the whole
/// draw, the way an engine binds a small zero-filled buffer to a stream its
/// shader declares but the mesh does not carry. Stream 1 holds four offsets,
/// only the second of which keeps the triangle on screen, and is bound at
/// that element with stride 0: every vertex must read it. Stepping the
/// stream per vertex instead would move two of the three vertices off
/// screen (and, past the buffer's end, fetch out of bounds).
#[test]
fn zero_stride_stream_feeds_every_vertex_the_element_at_its_offset() {
    let h = Harness::new();
    let decl = h.create_vertex_declaration(&pos_stream0_offset_stream1());
    let vs = h.create_vertex_shader(&VS_INSTANCED);
    let ps = h.create_pixel_shader(&PS_CONST);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
    assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");
    assert_eq!(
        h.set_pixel_shader_constant_f(0, &[1.0, 0.0, 0.0, 1.0]),
        0,
        "red constant"
    );

    let far = PosVertex {
        x: 5.0,
        y: 5.0,
        z: 0.0,
    };
    let none = PosVertex {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let offsets = [far, none, far, far];
    let stride = stride_of::<PosVertex>();
    let tri = centered_triangle();
    let positions = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    positions.lock(0, 0, 0).write(&tri);
    let constant = h.create_vertex_buffer(stride * 4, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    constant.lock(0, 0, 0).write(&offsets);
    assert_eq!(h.set_stream_source(0, &positions, 0, stride), D3D_OK);
    assert_eq!(h.set_stream_source(1, &constant, stride, 0), D3D_OK);

    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        RED,
        "every vertex reads the zero offset at the bound element"
    );
}

/// The `SetStreamSourceFreq` / `GetStreamSourceFreq` contract.
///
/// Defaults to 1 on every stream, rejects the combinations the runtime
/// rejects while leaving the stored word untouched, and round-trips the flag
/// bits through the getter.
#[test]
fn stream_source_freq_contract() {
    let h = Harness::new();
    assert_eq!(h.get_stream_source_freq(0), (D3D_OK, 1), "default stream 0");
    assert_eq!(h.get_stream_source_freq(1), (D3D_OK, 1), "default stream 1");

    assert_eq!(h.set_stream_source_freq(1, 1), D3D_OK, "plain 1");
    assert_eq!(
        h.set_stream_source_freq(0, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3DERR_INVALIDCALL,
        "INSTANCEDATA on stream 0"
    );
    assert_eq!(
        h.set_stream_source_freq(1, 0),
        D3DERR_INVALIDCALL,
        "literal zero"
    );
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, 1),
        "a rejected set leaves the word untouched"
    );

    assert_eq!(h.set_stream_source_freq(1, 2), D3D_OK, "count 2");
    assert_eq!(h.get_stream_source_freq(1), (D3D_OK, 2));
    assert_eq!(
        h.set_stream_source_freq(1, D3DSTREAMSOURCE_INDEXEDDATA),
        D3D_OK,
        "INDEXEDDATA with a zero count is a non-zero word"
    );
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, D3DSTREAMSOURCE_INDEXEDDATA),
        "flag round-trips"
    );
    assert_eq!(
        h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA),
        D3D_OK
    );
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, D3DSTREAMSOURCE_INSTANCEDATA)
    );
    assert_eq!(
        h.set_stream_source_freq(
            1,
            D3DSTREAMSOURCE_INSTANCEDATA | D3DSTREAMSOURCE_INDEXEDDATA
        ),
        D3DERR_INVALIDCALL,
        "both flags"
    );
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, D3DSTREAMSOURCE_INSTANCEDATA),
        "still the last accepted word"
    );

    assert_eq!(
        h.set_stream_source_freq(16, 1),
        D3DERR_INVALIDCALL,
        "stream past MaxStreams"
    );
    assert_eq!(
        h.get_stream_source_freq(16).0,
        D3DERR_INVALIDCALL,
        "getter past MaxStreams"
    );
}

/// Instanced geometry: a small quad on stream 0, four per-instance offsets on stream 1.
///
/// Returns the harness with shaders, declaration, buffers and a red pixel
/// constant bound; `draw_instances` issues the indexed draw. The four
/// instance centres land on the probe points `(160,360)`, `(480,360)`,
/// `(480,120)`, `(160,120)` in draw order.
struct InstancedScene {
    h: Harness,
}

const INSTANCE_PROBES: [(u32, u32); 4] = [(160, 360), (480, 360), (480, 120), (160, 120)];

impl InstancedScene {
    fn new() -> Self {
        let h = Harness::new();
        let decl = h.create_vertex_declaration(&pos_stream0_offset_stream1());
        let vs = h.create_vertex_shader(&VS_INSTANCED);
        let ps = h.create_pixel_shader(&PS_CONST);
        assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
        assert_eq!(h.set_vertex_shader(&vs), 0, "SetVertexShader");
        assert_eq!(h.set_pixel_shader(&ps), 0, "SetPixelShader");
        assert_eq!(
            h.set_pixel_shader_constant_f(0, &[1.0, 0.0, 0.0, 1.0]),
            0,
            "red constant"
        );

        // Bottom-left, top-left, bottom-right, top-right: vertices 0, 1, 2
        // and 2, 1, 3 both wind clockwise on screen, which the default
        // `D3DCULL_CCW` keeps.
        let quad = [
            PosVertex {
                x: -0.1,
                y: -0.1,
                z: 0.5,
            },
            PosVertex {
                x: -0.1,
                y: 0.1,
                z: 0.5,
            },
            PosVertex {
                x: 0.1,
                y: -0.1,
                z: 0.5,
            },
            PosVertex {
                x: 0.1,
                y: 0.1,
                z: 0.5,
            },
        ];
        let offsets = [
            PosVertex {
                x: -0.5,
                y: -0.5,
                z: 0.0,
            },
            PosVertex {
                x: 0.5,
                y: -0.5,
                z: 0.0,
            },
            PosVertex {
                x: 0.5,
                y: 0.5,
                z: 0.0,
            },
            PosVertex {
                x: -0.5,
                y: 0.5,
                z: 0.0,
            },
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 1, 3];
        let stride = stride_of::<PosVertex>();
        let vertices = h.create_vertex_buffer(stride * 4, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
        vertices.lock(0, 0, 0).write(&quad);
        let instances = h.create_vertex_buffer(stride * 4, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
        instances.lock(0, 0, 0).write(&offsets);
        let ib = h.create_index_buffer(12, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT);
        ib.lock(0, 0, 0).write(&indices);
        assert_eq!(h.set_stream_source(0, &vertices, 0, stride), D3D_OK);
        assert_eq!(h.set_stream_source(1, &instances, 0, stride), D3D_OK);
        assert_eq!(h.set_indices(&ib), D3D_OK);
        // The device holds its own references; the wrappers may drop here.
        drop((decl, vs, ps, vertices, instances, ib));
        Self { h }
    }

    fn draw_indexed(&self) {
        self.h.render_once(BLUE, |d| {
            assert_eq!(
                d.draw_indexed_primitive(D3DPT_TRIANGLELIST, 0, 0, 4, 0, 2),
                0,
                "DrawIndexedPrimitive"
            );
        });
    }

    fn assert_instances(&self, drawn: [bool; 4]) {
        for (i, ((x, y), expect_drawn)) in INSTANCE_PROBES.iter().zip(drawn).enumerate() {
            let want = if expect_drawn { RED } else { BLUE };
            assert_eq!(self.h.read_pixel(*x, *y), want, "instance {i}");
        }
    }
}

/// `INDEXEDDATA | 4` on stream 0 with a per-instance stream 1 draws four instances.
#[test]
fn indexed_draw_renders_every_instance() {
    let s = InstancedScene::new();
    assert_eq!(
        s.h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 4),
        D3D_OK
    );
    assert_eq!(
        s.h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3D_OK
    );
    s.draw_indexed();
    s.assert_instances([true, true, true, true]);
}

/// The instance count follows stream 0's frequency: two instances draw two quads.
#[test]
fn instance_count_follows_stream_zero() {
    let s = InstancedScene::new();
    assert_eq!(
        s.h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 2),
        D3D_OK
    );
    assert_eq!(
        s.h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3D_OK
    );
    s.draw_indexed();
    s.assert_instances([true, true, false, false]);
}

/// `INSTANCEDATA | 2` advances the per-instance stream every second instance.
///
/// Four instances read offsets 0, 0, 1, 1: the first two quads are drawn
/// (twice each), the last two positions stay clear.
#[test]
fn instance_step_rate_advances_every_n_instances() {
    let s = InstancedScene::new();
    assert_eq!(
        s.h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 4),
        D3D_OK
    );
    assert_eq!(
        s.h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 2),
        D3D_OK
    );
    s.draw_indexed();
    s.assert_instances([true, true, false, false]);
}

/// A non-indexed draw never instances: one quad at the first offset.
#[test]
fn non_indexed_draw_ignores_instancing() {
    let s = InstancedScene::new();
    assert_eq!(
        s.h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 4),
        D3D_OK
    );
    assert_eq!(
        s.h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3D_OK
    );
    s.h.render_once(BLUE, |d| {
        // Vertices 0..3 as a list make one triangle (0, 1, 2): the lower-left
        // half of the quad, which still covers the quad centre's row below
        // the diagonal. Probe slightly below-left of the centre.
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            0,
            "DrawPrimitive"
        );
    });
    assert_eq!(s.h.read_pixel(150, 370), RED, "first instance drawn");
    assert_eq!(s.h.read_pixel(470, 370), BLUE, "second instance not drawn");
    assert_eq!(s.h.read_pixel(470, 130), BLUE, "third instance not drawn");
    assert_eq!(s.h.read_pixel(150, 130), BLUE, "fourth instance not drawn");
}

/// Without a per-instance stream, stream 0's count is ignored: one instance.
#[test]
fn instance_count_needs_a_per_instance_stream() {
    let s = InstancedScene::new();
    assert_eq!(
        s.h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 4),
        D3D_OK
    );
    // Stream 1 stays per-vertex, so every vertex reads its own offset and
    // the four quad corners scatter into the four quadrants. None of the
    // probe centres receives a full quad; the draw is simply not instanced.
    s.draw_indexed();
    s.assert_instances([false, false, false, false]);
}

/// A recorded state block captures bindings and frequencies of streams beyond 0.
#[test]
fn recorded_state_block_captures_higher_streams() {
    let h = Harness::new();
    let vb = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);

    assert_eq!(h.begin_state_block(), 0, "BeginStateBlock");
    assert_eq!(h.set_stream_source(1, &vb, 4, 16), D3D_OK);
    assert_eq!(
        h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3D_OK
    );
    let sb = h.end_state_block();

    // Recording diverted both calls; the live device is untouched.
    let (hr, bound, _, _) = h.get_stream_source(1);
    assert_eq!(hr, D3D_OK);
    assert!(bound.is_none(), "recording does not bind");
    assert_eq!(h.get_stream_source_freq(1), (D3D_OK, 1));

    assert_eq!(sb.apply(), 0, "Apply");
    let (hr, bound, offset, stride) = h.get_stream_source(1);
    assert_eq!(hr, D3D_OK);
    assert_eq!(
        bound.expect("stream 1 bound by Apply").as_ptr(),
        vb.as_ptr()
    );
    assert_eq!((offset, stride), (4, 16));
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, D3DSTREAMSOURCE_INSTANCEDATA | 1)
    );
}

/// A `D3DSBT_ALL` snapshot restores every stream's binding and frequency.
#[test]
fn all_state_block_restores_streams() {
    let h = Harness::new();
    let vb0 = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    let vb1 = h.create_vertex_buffer(64, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    assert_eq!(h.set_stream_source(0, &vb0, 0, 12), D3D_OK);
    assert_eq!(h.set_stream_source(1, &vb1, 8, 16), D3D_OK);
    assert_eq!(
        h.set_stream_source_freq(0, D3DSTREAMSOURCE_INDEXEDDATA | 3),
        D3D_OK
    );
    assert_eq!(
        h.set_stream_source_freq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1),
        D3D_OK
    );
    let sb = h.create_state_block(D3DSBT_ALL);

    assert_eq!(h.set_stream_source_null(0, 0, 0), D3D_OK);
    assert_eq!(h.set_stream_source(1, &vb0, 0, 4), D3D_OK);
    assert_eq!(h.set_stream_source_freq(0, 1), D3D_OK);
    assert_eq!(h.set_stream_source_freq(1, 1), D3D_OK);

    assert_eq!(sb.apply(), 0, "Apply ALL");
    let (hr, bound, offset, stride) = h.get_stream_source(0);
    assert_eq!(hr, D3D_OK);
    assert_eq!(bound.expect("stream 0 restored").as_ptr(), vb0.as_ptr());
    assert_eq!((offset, stride), (0, 12));
    let (hr, bound, offset, stride) = h.get_stream_source(1);
    assert_eq!(hr, D3D_OK);
    assert_eq!(bound.expect("stream 1 restored").as_ptr(), vb1.as_ptr());
    assert_eq!((offset, stride), (8, 16));
    assert_eq!(
        h.get_stream_source_freq(0),
        (D3D_OK, D3DSTREAMSOURCE_INDEXEDDATA | 3)
    );
    assert_eq!(
        h.get_stream_source_freq(1),
        (D3D_OK, D3DSTREAMSOURCE_INSTANCEDATA | 1)
    );
}
