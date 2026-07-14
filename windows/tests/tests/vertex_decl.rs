//! Vertex declarations.
//!
//! Create, bind, drive a fixed-function draw, and read the bound
//! declaration back.

use mtld3d_tests::{Harness, PosColorVertex};
use mtld3d_types::{
    D3DDECL_END_STREAM, D3DDECLTYPE_D3DCOLOR, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_COLOR, D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DPT_TRIANGLELIST,
    D3DRS_LIGHTING, D3DVERTEXELEMENT9,
};

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
