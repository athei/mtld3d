//! Fixed-function vertex pipeline: lighting outputs, material sources, light types and texgen.
//!
//! Also indexed vertex blending over a vertex format that carries no indices.

use mtld3d_tests::{Harness, LitVertex, TexturedVertex, assert_pixel_approx};
use mtld3d_types::{
    D3DCOLORVALUE, D3DCULL_NONE, D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_NORMAL, D3DDECLUSAGE_POSITION, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_NORMAL,
    D3DFVF_TEX1, D3DFVF_XYZ, D3DFVF_XYZB1, D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATERIAL9,
    D3DPOOL_MANAGED, D3DPT_TRIANGLESTRIP, D3DRS_CULLMODE, D3DRS_INDEXEDVERTEXBLENDENABLE,
    D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_SPECULARENABLE, D3DRS_TEXTUREFACTOR,
    D3DRS_VERTEXBLEND, D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_MAGFILTER, D3DSAMP_MINFILTER,
    D3DTA_ALPHAREPLICATE, D3DTA_DIFFUSE, D3DTA_SPECULAR, D3DTA_TEXTURE, D3DTA_TFACTOR,
    D3DTADDRESS_WRAP, D3DTEXF_POINT, D3DTOP_MODULATE, D3DTOP_SELECTARG1, D3DTS_PROJECTION,
    D3DTS_VIEW, D3DTS_WORLD, D3DTSS_ALPHAARG1, D3DTSS_ALPHAOP, D3DTSS_COLORARG1, D3DTSS_COLORARG2,
    D3DTSS_COLOROP, D3DTSS_TCI_CAMERASPACENORMAL, D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR,
    D3DTSS_TEXCOORDINDEX, D3DVBF_1WEIGHTS, D3DVECTOR, D3DVERTEXELEMENT9,
};

#[rustfmt::skip]
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

const BLUE: u32 = 0xFF00_00FF;
const RED: u32 = 0xFFFF_0000;
const GREEN: u32 = 0xFF00_FF00;
const WHITE: u32 = 0xFFFF_FFFF;

const BLACK_VALUE: D3DCOLORVALUE = D3DCOLORVALUE {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

const WHITE_VALUE: D3DCOLORVALUE = D3DCOLORVALUE {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// A device with identity transforms and culling off, lit or not.
fn harness(lighting: bool) -> Harness {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, u32::from(lighting)), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    for state in [D3DTS_WORLD, D3DTS_VIEW, D3DTS_PROJECTION] {
        assert_eq!(h.set_transform(state, &IDENTITY), 0, "SetTransform");
    }
    h
}

/// A centred quad of half the viewport, every vertex carrying the normal `(nx, ny, nz)`.
fn lit_quad(normal: [f32; 3]) -> [LitVertex; 4] {
    [(-0.5, 0.5), (-0.5, -0.5), (0.5, 0.5), (0.5, -0.5)].map(|(x, y)| LitVertex {
        x,
        y,
        z: 0.5,
        nx: normal[0],
        ny: normal[1],
        nz: normal[2],
    })
}

/// A directional light shining along `direction` with the given diffuse and specular colours.
fn directional(direction: [f32; 3], diffuse: D3DCOLORVALUE, specular: D3DCOLORVALUE) -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        diffuse,
        specular,
        direction: D3DVECTOR {
            x: direction[0],
            y: direction[1],
            z: direction[2],
        },
        ..D3DLIGHT9::default()
    }
}

/// Route stage 0's colour from `color_arg` and its alpha from the diffuse colour.
fn select_color_from(h: &Harness, color_arg: u32) {
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_SELECTARG1),
        (D3DTSS_COLORARG1, color_arg),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0);
    }
}

/// Draw `quad` in a frame cleared to blue.
fn draw<V>(h: &Harness, quad: &[V; 4]) {
    h.render_once(BLUE, |d| {
        assert_eq!(d.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, quad), 0, "draw");
    });
}

/// A lit specular colour keeps the alpha of the light and the material specular.
///
/// The light and the material carry a specular alpha of one and black
/// specular RGB, the material diffuse is opaque black, and the stage shows
/// the specular alpha replicated: the
/// highlight's alpha is (N.H)^P = 1 across the quad, so it reads white
/// where an alpha of zero reads black.
#[test]
fn lit_specular_alpha_reaches_a_stage() {
    let h = harness(true);
    assert_eq!(h.set_render_state(D3DRS_SPECULARENABLE, 1), 0);
    // The infinite viewer makes H = N at every vertex.
    assert_eq!(h.set_render_state(D3DRS_LOCALVIEWER, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    let alpha_only = D3DCOLORVALUE {
        a: 1.0,
        ..BLACK_VALUE
    };
    let material = D3DMATERIAL9 {
        diffuse: alpha_only,
        specular: alpha_only,
        power: 1.0,
        ..D3DMATERIAL9::default()
    };
    assert_eq!(h.set_material(&material), 0);
    let light = directional([0.0, 0.0, 1.0], BLACK_VALUE, alpha_only);
    assert_eq!(h.set_light(0, &light), 0);
    assert_eq!(h.light_enable(0, true), 0);
    select_color_from(&h, D3DTA_SPECULAR | D3DTA_ALPHAREPLICATE);
    draw(&h, &lit_quad([0.0, 0.0, -1.0]));
    assert_pixel_approx(
        h.read_pixel(320, 240),
        WHITE,
        2,
        "specular alpha (N.H)^P = 1, replicated",
    );
}

/// With a material power of zero, a surface turned away from the viewer gets no specular.
///
/// The normal faces away from the viewer and half towards the light, so
/// N.L = 0.5 and N.H = -0.5. The specular term needs both positive; a power
/// of zero raised the clamped N.H of zero to one and added the light's whole
/// specular colour. Diffuse and ambient are black, so the specular add is
/// all the quad shows.
#[test]
fn zero_power_specular_skips_a_surface_turned_from_the_viewer() {
    let h = harness(true);
    assert_eq!(h.set_render_state(D3DRS_SPECULARENABLE, 1), 0);
    // The infinite viewer, V = (0, 0, -1), at every vertex.
    assert_eq!(h.set_render_state(D3DRS_LOCALVIEWER, 0), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            a: 1.0,
            ..BLACK_VALUE
        },
        specular: WHITE_VALUE,
        power: 0.0,
        ..D3DMATERIAL9::default()
    };
    assert_eq!(h.set_material(&material), 0);
    // L, the direction towards the light, is (0.866, 0, 0.5).
    let light = directional([-0.866, 0.0, -0.5], BLACK_VALUE, WHITE_VALUE);
    assert_eq!(h.set_light(0, &light), 0);
    assert_eq!(h.light_enable(0, true), 0);
    select_color_from(&h, D3DTA_DIFFUSE);
    draw(&h, &lit_quad([0.0, 0.0, 1.0]));
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xFF00_0000,
        2,
        "no specular where N.H is not positive",
    );
}

/// The lit diffuse alpha is clamped to [0, 1] before the pixel stage reads it.
///
/// The material diffuse alpha is 2, and the stage modulates the replicated
/// diffuse alpha by a texture factor of 0x80: a clamped alpha gives 0x80,
/// an unclamped one saturated the product to 0xff.
#[test]
fn lit_diffuse_alpha_is_clamped() {
    let h = harness(true);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            a: 2.0,
            ..BLACK_VALUE
        },
        ..D3DMATERIAL9::default()
    };
    assert_eq!(h.set_material(&material), 0);
    assert_eq!(h.set_render_state(D3DRS_TEXTUREFACTOR, 0x8080_8080), 0);
    for (state, value) in [
        (D3DTSS_COLOROP, D3DTOP_MODULATE),
        (D3DTSS_COLORARG1, D3DTA_DIFFUSE | D3DTA_ALPHAREPLICATE),
        (D3DTSS_COLORARG2, D3DTA_TFACTOR),
        (D3DTSS_ALPHAOP, D3DTOP_SELECTARG1),
        (D3DTSS_ALPHAARG1, D3DTA_DIFFUSE),
    ] {
        assert_eq!(h.set_texture_stage_state(0, state, value), 0);
    }
    draw(&h, &lit_quad([0.0, 0.0, -1.0]));
    assert_pixel_approx(
        h.read_pixel(320, 240),
        0xFF80_8080,
        2,
        "clamped diffuse alpha 1 times the factor 0x80",
    );
}

/// A lit declaration draw without a diffuse colour takes the material diffuse.
///
/// Under the D3D9 defaults (`D3DRS_COLORVERTEX` on, the diffuse source
/// `D3DMCS_COLOR1`) a vertex format that carries no diffuse colour falls
/// back to the material, through `SetVertexDeclaration` as through
/// `SetFVF`; the declaration path read it as zero and drew the quad black.
#[test]
fn lit_declaration_without_a_diffuse_colour_takes_the_material_diffuse() {
    let h = harness(true);
    let element = |offset, usage| D3DVERTEXELEMENT9 {
        stream: 0,
        offset,
        type_: D3DDECLTYPE_FLOAT3,
        method: 0,
        usage,
        usage_index: 0,
    };
    let decl = h.create_vertex_declaration(&[
        element(0, D3DDECLUSAGE_POSITION),
        element(12, D3DDECLUSAGE_NORMAL),
        D3DVERTEXELEMENT9 {
            stream: D3DDECL_END_STREAM,
            offset: 0,
            type_: D3DDECLTYPE_UNUSED,
            method: 0,
            usage: 0,
            usage_index: 0,
        },
    ]);
    assert_eq!(h.set_vertex_declaration(&decl), 0);
    let material = D3DMATERIAL9 {
        diffuse: D3DCOLORVALUE {
            r: 1.0,
            a: 1.0,
            ..BLACK_VALUE
        },
        ..D3DMATERIAL9::default()
    };
    assert_eq!(h.set_material(&material), 0);
    let light = directional([0.0, 0.0, 1.0], WHITE_VALUE, BLACK_VALUE);
    assert_eq!(h.set_light(0, &light), 0);
    assert_eq!(h.light_enable(0, true), 0);
    select_color_from(&h, D3DTA_DIFFUSE);
    draw(&h, &lit_quad([0.0, 0.0, -1.0]));
    assert_pixel_approx(
        h.read_pixel(320, 240),
        RED,
        2,
        "material diffuse lit by a white light",
    );
}

/// A light whose type is none of POINT, SPOT and DIRECTIONAL lights nothing.
///
/// `SetLight` accepts it and `GetLight` reports it back, and it can be
/// enabled, but a type of 4 at the eye with a long range adds no light
/// where it was read as a point light and lit the white material.
#[test]
fn a_light_of_no_valid_type_lights_nothing() {
    let h = harness(true);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    let material = D3DMATERIAL9 {
        diffuse: WHITE_VALUE,
        ..D3DMATERIAL9::default()
    };
    assert_eq!(h.set_material(&material), 0);
    let light = D3DLIGHT9 {
        type_: 4,
        diffuse: WHITE_VALUE,
        range: 100.0,
        attenuation0: 1.0,
        ..D3DLIGHT9::default()
    };
    assert_eq!(h.set_light(0, &light), 0, "SetLight of type 4");
    assert_eq!(h.light(0).type_, 4, "GetLight reports the type back");
    assert_eq!(h.light_enable(0, true), 0);
    select_color_from(&h, D3DTA_DIFFUSE);
    draw(&h, &lit_quad([0.0, 0.0, -1.0]));
    assert_pixel_approx(h.read_pixel(320, 240), 0xFF00_0000, 2, "no light");
}

/// A 2x2 texture, red, green, blue and white from the top left, sampled by point at stage 0.
const TEXGEN_TEXELS: [u32; 4] = [RED, GREEN, BLUE, WHITE];

/// A quad of `TexturedVertex`es over `x` and `y` in clip space, every vertex at `(0.25, 0.75)`.
fn textured_quad(x: (f32, f32), y: (f32, f32)) -> [TexturedVertex; 4] {
    [(x.0, y.1), (x.0, y.0), (x.1, y.1), (x.1, y.0)].map(|(x, y)| TexturedVertex {
        x,
        y,
        z: 0.5,
        color: WHITE,
        u: 0.25,
        v: 0.75,
    })
}

/// CAMERASPACENORMAL and CAMERASPACEREFLECTIONVECTOR without a vertex normal read a zero normal.
///
/// The vertices carry no normal and the coordinate (0.25, 0.75), which
/// names the blue texel. A zero normal generates (0, 0, 0) for the camera
/// space normal, the red texel, and for the reflection vector the unit
/// eye-to-vertex direction, which over a small quad at clip (0.5, 0.25)
/// lies near (0.66, 0.33), the green texel. Both modes read the blue texel
/// where they passed the vertex's own coordinate through.
#[test]
fn normal_less_texgen_generates_from_a_zero_normal() {
    let h = harness(false);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1), 0);
    let tex = h.create_texture(2, 2, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    tex.lock_rect(0, 0).write_u32(&TEXGEN_TEXELS);
    assert_eq!(h.set_texture(0, &tex), 0);
    for (state, value) in [
        (D3DSAMP_MINFILTER, D3DTEXF_POINT),
        (D3DSAMP_MAGFILTER, D3DTEXF_POINT),
        (D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP),
        (D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP),
    ] {
        assert_eq!(h.set_sampler_state(0, state, value), 0);
    }
    select_color_from(&h, D3DTA_TEXTURE);

    assert_eq!(
        h.set_texture_stage_state(0, D3DTSS_TEXCOORDINDEX, D3DTSS_TCI_CAMERASPACENORMAL),
        0
    );
    draw(&h, &textured_quad((-0.25, 0.25), (-0.25, 0.25)));
    assert_pixel_approx(
        h.read_pixel(320, 240),
        RED,
        2,
        "camera-space normal (0, 0, 0)",
    );

    assert_eq!(
        h.set_texture_stage_state(
            0,
            D3DTSS_TEXCOORDINDEX,
            D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR
        ),
        0
    );
    draw(&h, &textured_quad((0.4, 0.6), (0.2, 0.3)));
    assert_pixel_approx(
        h.read_pixel(480, 180),
        GREEN,
        2,
        "reflection about a zero normal: the eye-to-vertex direction",
    );
}

/// Position, one blend weight and a diffuse colour, for `D3DFVF_XYZB1 | D3DFVF_DIFFUSE`.
#[repr(C)]
struct SequentialBlendVertex {
    position: [f32; 3],
    weight: f32,
    color: u32,
}

/// Indexed vertex blending without a BLENDINDICES element blends the sequential matrices.
///
/// `D3DRS_INDEXEDVERTEXBLENDENABLE` is on but the FVF carries one weight and
/// no indices, so `D3DVBF_1WEIGHTS` reads matrices 0 and 1 in order. The
/// whole weight sits on the implicit second matrix, which moves the quad
/// right by 0.6; the draw that dropped blending kept it at the centre.
#[test]
fn indexed_blending_without_indices_blends_sequentially() {
    let h = harness(false);
    assert_eq!(h.set_transform(D3DTS_WORLD + 1, &translate_x(0.6)), 0);
    assert_eq!(h.set_render_state(D3DRS_VERTEXBLEND, D3DVBF_1WEIGHTS), 0);
    assert_eq!(h.set_render_state(D3DRS_INDEXEDVERTEXBLENDENABLE, 1), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZB1 | D3DFVF_DIFFUSE), 0);
    h.select_diffuse_stage(0);
    let quad = [(-0.25, 0.25), (-0.25, -0.25), (0.25, 0.25), (0.25, -0.25)].map(|(x, y)| {
        SequentialBlendVertex {
            position: [x, y, 0.5],
            weight: 0.0,
            color: RED,
        }
    });
    draw(&h, &quad);
    assert_eq!(h.read_pixel(512, 240), RED, "the quad moved by matrix 1");
    assert_eq!(h.read_pixel(320, 240), BLUE, "nothing left at the centre");
}

/// Row-major translation along x.
const fn translate_x(x: f32) -> [f32; 16] {
    let mut m = IDENTITY;
    m[12] = x;
    m
}
