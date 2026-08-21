//! Vertex streams beyond stream 0 and hardware instancing.
//!
//! A declaration split across two streams drawn through the programmable
//! pipeline, the `SetStreamSourceFreq` contract, instanced indexed draws
//! (count, step rate, and the non-indexed exemption), and state-block capture
//! of stream bindings and frequencies.

use mtld3d_tests::{Harness, PosVertex};
use mtld3d_types::{
    D3D_OK, D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_COLOR, D3DDECLUSAGE_POSITION, D3DDECLUSAGE_TEXCOORD, D3DERR_INVALIDCALL,
    D3DFMT_INDEX16, D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DSBT_ALL, D3DSTREAMSOURCE_INDEXEDDATA,
    D3DSTREAMSOURCE_INSTANCEDATA, D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9,
};

const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const BLUE: u32 = 0xFF00_00FF;

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
