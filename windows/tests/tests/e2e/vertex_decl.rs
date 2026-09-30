//! Vertex declarations.
//!
//! Create, bind, drive a fixed-function draw, and read the bound
//! declaration back; a declaration split across two streams, including a
//! stream nothing is bound to; `SetFVF` of the FVF already set binding its
//! declaration again after another one replaced it.

use mtld3d_tests::{Harness, PosColorVertex, PosVertex, VertexDeclaration};
use mtld3d_types::{
    D3D_OK, D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_COLOR, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DFVF_DIFFUSE, D3DFVF_XYZ,
    D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DRS_LIGHTING, D3DUSAGE_WRITEONLY, D3DVERTEXELEMENT9,
};

/// POSITION float3 on stream 0, COLOR d3dcolor on stream 1.
const fn two_stream_decl() -> [D3DVERTEXELEMENT9; 3] {
    [
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 1,
            offset: 0,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ]
}

const fn centered_positions() -> [PosVertex; 3] {
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

/// Lighting off, stage 0 selecting the vertex diffuse, the two-stream declaration bound.
fn arm_two_stream_ff(h: &Harness) {
    let decl = h.create_vertex_declaration(&two_stream_decl());
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
}

/// POSITION float3 @ 0, COLOR d3dcolor @ 12, terminated by `D3DDECL_END`.
const fn pos_color_decl() -> [D3DVERTEXELEMENT9; 3] {
    [
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ]
}

#[test]
fn vertex_declaration_drives_ff_draw() {
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;

    let h = Harness::new();
    let decl = h.create_vertex_declaration(&pos_color_decl());
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(
        h.vertex_declaration_raw(),
        decl.as_ptr(),
        "GetVertexDeclaration returns the bound declaration",
    );

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);

    let tri = [
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
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
            0,
            "decl-driven draw"
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "declaration-driven FF draw renders vertex colour"
    );
}

/// A declaration with an *unused* NORMAL between POSITION and COLOR still delivers COLOR.
///
/// Lighting is off, so the normal is dead. The layout has the shape of a
/// POSITION+NORMAL+TEXCOORD declaration, in which a live attribute lands at a
/// non-contiguous attribute index: the COLOR element past the unused NORMAL
/// must still reach the fixed-function stage through a `DrawPrimitiveUP` draw.
#[test]
fn decl_with_unused_normal_still_delivers_color() {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PosNormalColorVertex {
        x: f32,
        y: f32,
        z: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        color: u32,
    }

    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;

    let decl = [
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 12,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_NORMAL,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 24,
            type_: D3DDECLTYPE_D3DCOLOR,
            method: 0,
            usage: D3DDECLUSAGE_COLOR,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ];

    let h = Harness::new();
    let decl = h.create_vertex_declaration(&decl);
    assert_eq!(h.set_vertex_declaration(&decl), 0, "SetVertexDeclaration");
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);

    let v = |x: f32, y: f32| PosNormalColorVertex {
        x,
        y,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: 1.0,
        color: GREEN,
    };
    let tri = [v(0.0, 0.5), v(0.5, -0.5), v(-0.5, -0.5)];
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "COLOR after an unused NORMAL must still be delivered"
    );
}

/// The fixed-function pipeline reads a declaration split across two streams.
///
/// Positions come from the stream-0 buffer, the diffuse colour from a
/// second buffer bound at stream 1 with its own stride.
#[test]
fn two_stream_declaration_drives_ff_draw() {
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;

    let h = Harness::new();
    arm_two_stream_ff(&h);
    let stride = u32::try_from(size_of::<PosVertex>()).expect("stride fits u32");
    let positions = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    positions.lock(0, 0, 0).write(&centered_positions());
    let diffuse = h.create_vertex_buffer(4 * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    diffuse.lock(0, 0, 0).write(&[GREEN; 3]);
    assert_eq!(h.set_stream_source(0, &positions, 0, stride), D3D_OK);
    assert_eq!(h.set_stream_source(1, &diffuse, 0, 4), D3D_OK);

    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1), 0, "draw");
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "diffuse from stream 1 reaches the fixed-function stage"
    );
}

/// A declared stream with nothing bound reads zeros and the draw succeeds.
///
/// The colour element lives on stream 1, which stays unbound: all four
/// components of the diffuse read as zero, alpha included, so the
/// unlit fixed-function draw writes transparent black.
#[test]
fn unbound_declared_stream_reads_zeros() {
    const BLACK: u32 = 0x0000_0000;
    const BLUE: u32 = 0xFF00_00FF;

    let h = Harness::new();
    arm_two_stream_ff(&h);
    let stride = u32::try_from(size_of::<PosVertex>()).expect("stride fits u32");
    let positions = h.create_vertex_buffer(stride * 3, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT);
    positions.lock(0, 0, 0).write(&centered_positions());
    assert_eq!(h.set_stream_source(0, &positions, 0, stride), D3D_OK);
    assert_eq!(h.set_stream_source_null(1, 0, 0), D3D_OK);

    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive(D3DPT_TRIANGLELIST, 0, 1),
            D3D_OK,
            "a draw reading an unbound stream succeeds"
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        BLACK,
        "the unbound colour stream reads zeros"
    );

    // The same declaration over inline vertices: stream 0 is the UP data,
    // stream 1 still reads zeros.
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &centered_positions()),
            D3D_OK,
            "UP draw with a two-stream declaration"
        );
    });
    assert_eq!(h.read_pixel(320, 280), BLACK, "UP draw reads zeros too");
}

/// POSITION float3 on stream 0 and nothing else, terminated by `D3DDECL_END`.
const fn position_only_decl() -> [D3DVERTEXELEMENT9; 2] {
    [
        D3DVERTEXELEMENT9 {
            stream: 0,
            offset: 0,
            type_: D3DDECLTYPE_FLOAT3,
            method: 0,
            usage: D3DDECLUSAGE_POSITION,
            usage_index: 0,
        },
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ]
}

/// Draw a green triangle through the bound vertex format and read its centre.
///
/// The vertices carry a diffuse colour, so only a format that declares it
/// draws green; the position-only declaration leaves the colour at its
/// default.
fn draw_green_through_bound_format(h: &Harness) -> u32 {
    const GREEN: u32 = 0xFF00_FF00;
    const BLUE: u32 = 0xFF00_00FF;

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0, "lighting off");
    assert_eq!(h.clear_texture(0), 0, "no texture");
    h.select_diffuse_stage(0);
    let tri = centered_positions().map(|p| PosColorVertex {
        x: p.x,
        y: p.y,
        z: p.z,
        color: GREEN,
    });
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &tri),
            D3D_OK,
            "draw through the bound format"
        );
    });
    h.read_pixel(320, 280)
}

/// `SetFVF` of the FVF set before a `SetVertexDeclaration` binds the FVF's declaration again.
///
/// The FVF-to-declaration step can return early for an FVF that is already
/// bound; the declaration bound in between has to defeat that.
#[test]
fn set_fvf_rebinds_its_declaration_after_set_vertex_declaration() {
    const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;

    let h = Harness::new();
    assert_eq!(h.set_fvf(FVF), D3D_OK, "first SetFVF");
    let implicit = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    let position_only = h.create_vertex_declaration(&position_only_decl());
    assert_eq!(
        h.set_vertex_declaration(&position_only),
        D3D_OK,
        "SetVertexDeclaration"
    );
    assert_eq!(h.fvf(), 0, "a bound declaration clears the FVF");

    assert_eq!(h.set_fvf(FVF), D3D_OK, "second SetFVF");
    assert_eq!(h.fvf(), FVF, "GetFVF after the second SetFVF");
    let rebound = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_eq!(
        rebound.as_ptr(),
        implicit.as_ptr(),
        "the second SetFVF binds the FVF's declaration again"
    );
    assert_eq!(
        draw_green_through_bound_format(&h),
        0xFF00_FF00,
        "the draw reads the diffuse the FVF declares"
    );
}

/// `SetFVF` of the FVF the device reports binds its declaration again after a block replaced it.
///
/// A recorded `SetVertexDeclaration` applied over an FVF binds its
/// declaration without clearing the FVF the device reports, so the FVF on
/// its own does not say whether its declaration is still the bound one.
#[test]
fn set_fvf_rebinds_its_declaration_after_a_recorded_block_bound_another() {
    const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;

    let h = Harness::new();
    let position_only = h.create_vertex_declaration(&position_only_decl());
    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(
        h.set_vertex_declaration(&position_only),
        D3D_OK,
        "record SetVertexDeclaration"
    );
    let block = h.end_state_block();

    assert_eq!(h.set_fvf(FVF), D3D_OK, "first SetFVF");
    let implicit = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_eq!(block.apply(), D3D_OK, "Apply the recorded declaration");
    let applied = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_eq!(
        applied.as_ptr(),
        position_only.as_ptr(),
        "Apply binds the recorded declaration"
    );

    assert_eq!(h.set_fvf(FVF), D3D_OK, "second SetFVF");
    let rebound = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_eq!(
        rebound.as_ptr(),
        implicit.as_ptr(),
        "the second SetFVF binds the FVF's declaration again"
    );
    assert_eq!(
        draw_green_through_bound_format(&h),
        0xFF00_FF00,
        "the draw reads the diffuse the FVF declares"
    );
}
