//! State-block capture/apply round-trip.

use mtld3d_tests::{Harness, LitVertex, PosColorVertex, VertexDeclaration, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DCOLORVALUE, D3DCULL_NONE, D3DDECL_END_STREAM, D3DDECLTYPE_FLOAT3,
    D3DDECLTYPE_UNUSED, D3DDECLUSAGE_POSITION, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE,
    D3DFVF_NORMAL, D3DFVF_XYZ, D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DMATERIAL9, D3DPOOL_MANAGED,
    D3DPT_TRIANGLELIST, D3DPT_TRIANGLESTRIP, D3DRECT, D3DRS_ALPHABLENDENABLE, D3DRS_CULLMODE,
    D3DRS_LIGHTING, D3DRS_SCISSORTESTENABLE, D3DSAMP_DMAPOFFSET, D3DSAMP_MINFILTER, D3DSBT_ALL,
    D3DSBT_PIXELSTATE, D3DSBT_VERTEXSTATE, D3DTEXF_LINEAR, D3DTEXF_POINT, D3DVECTOR,
    D3DVERTEXELEMENT9, D3DVIEWPORT9,
};

const BLUE: u32 = 0xFF00_00FF;
const RED: u32 = 0xFFFF_0000;
const VERTEX_SAMPLER_0: u32 = 257;

const INITIAL_VIEWPORT: D3DVIEWPORT9 = D3DVIEWPORT9 {
    x: 32,
    y: 24,
    width: 576,
    height: 432,
    min_z: 0.1,
    max_z: 0.9,
};

const REFRESHED_VIEWPORT: D3DVIEWPORT9 = D3DVIEWPORT9 {
    x: 128,
    y: 96,
    width: 384,
    height: 288,
    min_z: 0.0,
    max_z: 1.0,
};

const INITIAL_SCISSOR: D3DRECT = D3DRECT {
    x1: 48,
    y1: 40,
    x2: 592,
    y2: 440,
};

const REFRESHED_SCISSOR: D3DRECT = D3DRECT {
    x1: 64,
    y1: 48,
    x2: 384,
    y2: 336,
};

fn assert_viewport(actual: D3DVIEWPORT9, expected: D3DVIEWPORT9, message: &str) {
    assert_eq!(actual.x, expected.x, "{message}: x");
    assert_eq!(actual.y, expected.y, "{message}: y");
    assert_eq!(actual.width, expected.width, "{message}: width");
    assert_eq!(actual.height, expected.height, "{message}: height");
    assert_eq!(
        actual.min_z.to_bits(),
        expected.min_z.to_bits(),
        "{message}: min_z"
    );
    assert_eq!(
        actual.max_z.to_bits(),
        expected.max_z.to_bits(),
        "{message}: max_z"
    );
}

fn assert_scissor(actual: D3DRECT, expected: D3DRECT, message: &str) {
    assert_eq!(actual.x1, expected.x1, "{message}: x1");
    assert_eq!(actual.y1, expected.y1, "{message}: y1");
    assert_eq!(actual.x2, expected.x2, "{message}: x2");
    assert_eq!(actual.y2, expected.y2, "{message}: y2");
}

#[test]
fn capture_apply_restores_render_state() {
    let h = Harness::new();
    // CreateStateBlock(D3DSBT_ALL) snapshots the device's current state.
    let sb = h.create_state_block(D3DSBT_ALL);

    let before = h.render_state(D3DRS_LIGHTING);
    let flipped = u32::from(before == 0);
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, flipped),
        0,
        "mutate LIGHTING"
    );
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        flipped,
        "mutation took effect"
    );

    assert_eq!(sb.apply(), 0, "StateBlock::Apply");
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        before,
        "Apply restores captured LIGHTING"
    );
}

#[test]
fn all_block_restores_captured_viewport_and_scissor() {
    let h = Harness::new();
    let full_viewport = h.viewport();
    let full_scissor = h.scissor_rect();
    assert_eq!(h.set_viewport(&INITIAL_VIEWPORT), 0, "initial viewport");
    assert_eq!(h.set_scissor_rect(&INITIAL_SCISSOR), 0, "initial scissor");
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);

    let sb = h.create_state_block(D3DSBT_ALL);
    assert_eq!(h.set_viewport(&full_viewport), 0, "mutate viewport");
    assert_eq!(h.set_scissor_rect(&full_scissor), 0, "mutate scissor");
    assert_viewport(h.viewport(), full_viewport, "mutated viewport");
    assert_scissor(h.scissor_rect(), full_scissor, "mutated scissor");
    assert_eq!(sb.apply(), 0, "Apply initial capture");
    assert_viewport(h.viewport(), INITIAL_VIEWPORT, "restored initial viewport");
    assert_scissor(
        h.scissor_rect(),
        INITIAL_SCISSOR,
        "restored initial scissor",
    );

    assert_eq!(h.set_viewport(&REFRESHED_VIEWPORT), 0, "refreshed viewport");
    assert_eq!(
        h.set_scissor_rect(&REFRESHED_SCISSOR),
        0,
        "refreshed scissor"
    );
    assert_eq!(sb.capture(), 0, "Capture refreshed state");

    assert_eq!(
        h.set_viewport(&full_viewport),
        0,
        "second viewport mutation"
    );
    assert_eq!(
        h.set_scissor_rect(&full_scissor),
        0,
        "second scissor mutation"
    );
    assert_eq!(h.set_render_state(D3DRS_SCISSORTESTENABLE, 0), 0);
    let triangle = [
        PosColorVertex {
            x: -1.0,
            y: 3.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: 3.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
        PosColorVertex {
            x: -1.0,
            y: -1.0,
            z: 0.5,
            color: RED,
        },
    ];
    h.render_once(BLUE, |d| {
        assert_eq!(sb.apply(), 0, "Apply refreshed capture");
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLELIST, 1, &triangle),
            0,
            "draw with restored viewport and scissor"
        );
    });

    assert_viewport(
        h.viewport(),
        REFRESHED_VIEWPORT,
        "restored refreshed viewport",
    );
    assert_scissor(
        h.scissor_rect(),
        REFRESHED_SCISSOR,
        "restored refreshed scissor",
    );
    assert_pixel_eq(h.read_pixel(256, 200), RED, "inside both restored bounds");
    assert_pixel_eq(h.read_pixel(96, 200), BLUE, "outside restored viewport");
    assert_pixel_eq(h.read_pixel(448, 200), BLUE, "outside restored scissor");
}

#[test]
fn all_block_restores_vertex_texture_and_sampler() {
    let h = Harness::new();
    let first = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let second = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    assert_eq!(
        h.set_texture(VERTEX_SAMPLER_0, &first),
        0,
        "initial texture"
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        0,
        "initial sampler"
    );
    let sb = h.create_state_block(D3DSBT_ALL);

    assert_eq!(
        h.set_texture(VERTEX_SAMPLER_0, &second),
        0,
        "mutate texture"
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        0,
        "mutate sampler"
    );
    assert!(
        h.texture_matches_raw(VERTEX_SAMPLER_0, second.as_ptr()),
        "vertex texture mutation took effect"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER),
        D3DTEXF_POINT,
        "vertex sampler mutation took effect"
    );
    assert_eq!(sb.apply(), 0, "Apply initial capture");
    assert!(
        h.texture_matches_raw(VERTEX_SAMPLER_0, first.as_ptr()),
        "Apply restores captured vertex texture"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER),
        D3DTEXF_LINEAR,
        "Apply restores captured vertex sampler"
    );

    assert_eq!(
        h.set_texture(VERTEX_SAMPLER_0, &second),
        0,
        "refreshed texture"
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        0,
        "refreshed sampler"
    );
    assert_eq!(sb.capture(), 0, "Capture refreshed state");
    assert_eq!(
        h.set_texture(VERTEX_SAMPLER_0, &first),
        0,
        "second texture mutation"
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        0,
        "second sampler mutation"
    );
    assert_eq!(sb.apply(), 0, "Apply refreshed capture");
    assert!(
        h.texture_matches_raw(VERTEX_SAMPLER_0, second.as_ptr()),
        "Apply restores refreshed vertex texture"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER),
        D3DTEXF_POINT,
        "Apply restores refreshed vertex sampler"
    );
}

#[test]
fn vertex_state_block_leaves_all_only_state() {
    let h = Harness::new();
    let full_viewport = h.viewport();
    let full_scissor = h.scissor_rect();
    let first = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);
    let second = h.create_texture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED);

    assert_eq!(h.set_viewport(&INITIAL_VIEWPORT), 0);
    assert_eq!(h.set_scissor_rect(&INITIAL_SCISSOR), 0);
    assert_eq!(h.set_texture(VERTEX_SAMPLER_0, &first), 0);
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        0
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET, 3),
        0
    );
    let sb = h.create_state_block(D3DSBT_VERTEXSTATE);

    assert_eq!(h.set_viewport(&full_viewport), 0);
    assert_eq!(h.set_scissor_rect(&full_scissor), 0);
    assert_eq!(h.set_texture(VERTEX_SAMPLER_0, &second), 0);
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        0
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET, 9),
        0
    );
    assert_eq!(sb.apply(), 0, "Apply VERTEXSTATE");

    assert_viewport(h.viewport(), full_viewport, "VERTEXSTATE leaves viewport");
    assert_scissor(h.scissor_rect(), full_scissor, "VERTEXSTATE leaves scissor");
    assert!(
        h.texture_matches_raw(VERTEX_SAMPLER_0, second.as_ptr()),
        "VERTEXSTATE leaves vertex texture"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER),
        D3DTEXF_POINT,
        "VERTEXSTATE leaves ordinary vertex sampler state"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET),
        3,
        "VERTEXSTATE restores its vertex sampler member"
    );
}

#[test]
fn pixel_state_block_filters_vertex_sampler_members() {
    let h = Harness::new();
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        0
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET, 3),
        0
    );
    let sb = h.create_state_block(D3DSBT_PIXELSTATE);

    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        0
    );
    assert_eq!(
        h.set_sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET, 9),
        0
    );
    assert_eq!(sb.apply(), 0, "Apply PIXELSTATE");
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_MINFILTER),
        D3DTEXF_LINEAR,
        "PIXELSTATE restores an ordinary vertex sampler member"
    );
    assert_eq!(
        h.sampler_state(VERTEX_SAMPLER_0, D3DSAMP_DMAPOFFSET),
        9,
        "PIXELSTATE leaves the VERTEXSTATE-only sampler member"
    );
}

#[test]
fn vertex_state_block_restores_fvf() {
    let h = Harness::new();
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0, "initial FVF");
    let sb = h.create_state_block(D3DSBT_VERTEXSTATE);
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "mutate FVF");
    assert_eq!(sb.apply(), 0, "Apply VERTEXSTATE");
    assert_eq!(
        h.fvf(),
        D3DFVF_XYZ | D3DFVF_DIFFUSE,
        "VERTEXSTATE restores FVF"
    );
}

#[test]
fn recorded_set_fvf_restores_implicit_declaration_and_draw_layout() {
    const BLUE: u32 = 0xFF00_00FF;
    const GREEN: u32 = 0xFF00_FF00;
    const TARGET_FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;

    let h = Harness::new();
    assert_eq!(h.set_fvf(D3DFVF_XYZ), 0, "initial FVF");
    let initial_decl = VertexDeclaration::from_raw(h.vertex_declaration_raw());

    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.set_fvf(TARGET_FVF), D3D_OK, "record SetFVF");
    let sb = h.end_state_block();
    assert_eq!(sb.apply(), D3D_OK, "Apply recorded SetFVF");

    assert_eq!(h.fvf(), TARGET_FVF, "Apply restores the recorded FVF");
    let applied_decl = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_ne!(
        applied_decl.as_ptr(),
        initial_decl.as_ptr(),
        "Apply binds the recorded FVF's implicit declaration",
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
            D3D_OK,
            "draw with the recorded FVF layout",
        );
    });
    assert_eq!(
        h.read_pixel(320, 280),
        GREEN,
        "the restored declaration supplies vertex diffuse",
    );
}

#[test]
fn recorded_zero_fvf_keeps_the_last_nonzero_fvf_binding() {
    const TARGET_FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;

    let h = Harness::new();
    assert_eq!(h.set_fvf(D3DFVF_XYZ), D3D_OK, "initial FVF");
    let initial_decl = VertexDeclaration::from_raw(h.vertex_declaration_raw());

    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.set_fvf(TARGET_FVF), D3D_OK, "record nonzero SetFVF");
    assert_eq!(h.set_fvf(0), D3D_OK, "record SetFVF(0)");
    let sb = h.end_state_block();
    assert_eq!(sb.apply(), D3D_OK, "Apply recorded FVF sequence");

    assert_eq!(
        h.fvf(),
        TARGET_FVF,
        "SetFVF(0) leaves the last nonzero FVF intact",
    );
    let applied_decl = VertexDeclaration::from_raw(h.vertex_declaration_raw());
    assert_ne!(
        applied_decl.as_ptr(),
        initial_decl.as_ptr(),
        "SetFVF(0) leaves the last nonzero implicit declaration bound",
    );
}

#[test]
fn pixel_state_block_restores_sampler() {
    let h = Harness::new();
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR),
        0,
        "initial filter"
    );
    let sb = h.create_state_block(D3DSBT_PIXELSTATE);
    assert_eq!(
        h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT),
        0,
        "mutate filter"
    );
    assert_eq!(sb.apply(), 0, "Apply PIXELSTATE");
    assert_eq!(
        h.sampler_state(0, D3DSAMP_MINFILTER),
        D3DTEXF_LINEAR,
        "PIXELSTATE restores sampler"
    );
}

/// A `D3DSBT_VERTEXSTATE` block must restore vertex render states.
///
/// It must leave pixel render states at their live value. `D3DRS_LIGHTING` is
/// vertex state; `D3DRS_ALPHABLENDENABLE` is pixel state.
#[test]
fn vertex_state_block_filters_render_state() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    let sb = h.create_state_block(D3DSBT_VERTEXSTATE);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 0), 0);
    assert_eq!(sb.apply(), 0, "Apply VERTEXSTATE");
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        1,
        "VERTEXSTATE restores vertex render state"
    );
    assert_eq!(
        h.render_state(D3DRS_ALPHABLENDENABLE),
        0,
        "VERTEXSTATE leaves pixel render state untouched"
    );
}

/// The pixel-block mirror of [`vertex_state_block_filters_render_state`].
#[test]
fn pixel_state_block_filters_render_state() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 1), 0);
    let sb = h.create_state_block(D3DSBT_PIXELSTATE);
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    assert_eq!(h.set_render_state(D3DRS_ALPHABLENDENABLE, 0), 0);
    assert_eq!(sb.apply(), 0, "Apply PIXELSTATE");
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "PIXELSTATE leaves vertex render state untouched"
    );
    assert_eq!(
        h.render_state(D3DRS_ALPHABLENDENABLE),
        1,
        "PIXELSTATE restores pixel render state"
    );
}

/// Sampler state is pixel-pipeline, so a `D3DSBT_VERTEXSTATE` block must not touch it.
///
/// The positive case is [`pixel_state_block_restores_sampler`].
#[test]
fn vertex_state_block_leaves_sampler() {
    let h = Harness::new();
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR), 0);
    let sb = h.create_state_block(D3DSBT_VERTEXSTATE);
    assert_eq!(h.set_sampler_state(0, D3DSAMP_MINFILTER, D3DTEXF_POINT), 0);
    assert_eq!(sb.apply(), 0, "Apply VERTEXSTATE");
    assert_eq!(
        h.sampler_state(0, D3DSAMP_MINFILTER),
        D3DTEXF_POINT,
        "VERTEXSTATE leaves sampler state untouched"
    );
}

/// Lights are vertex-pipeline state.
///
/// A `D3DSBT_VERTEXSTATE` block restores the light-enable flag, a
/// `D3DSBT_PIXELSTATE` block leaves it.
#[test]
fn light_enable_follows_vertex_pipeline_filter() {
    let lit = D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        range: 5.0,
        ..Default::default()
    };

    let hv = Harness::new();
    assert_eq!(hv.set_light(0, &lit), 0);
    assert_eq!(hv.light_enable(0, true), 0);
    let vsb = hv.create_state_block(D3DSBT_VERTEXSTATE);
    assert_eq!(hv.light_enable(0, false), 0);
    assert_eq!(vsb.apply(), 0, "Apply VERTEXSTATE");
    assert!(hv.light_enabled(0), "VERTEXSTATE restores light-enable");

    let hp = Harness::new();
    assert_eq!(hp.set_light(0, &lit), 0);
    assert_eq!(hp.light_enable(0, true), 0);
    let psb = hp.create_state_block(D3DSBT_PIXELSTATE);
    assert_eq!(hp.light_enable(0, false), 0);
    assert_eq!(psb.apply(), 0, "Apply PIXELSTATE");
    assert!(
        !hp.light_enabled(0),
        "PIXELSTATE leaves light-enable untouched"
    );
}

/// A second `BeginStateBlock` is rejected and the open recording survives it.
///
/// Both halves matter. The rejection is what D3D9 specifies, and leaving the
/// recording alone is what makes it sticky: an application that begins a block
/// and never ends it has every later `BeginStateBlock` rejected, while the
/// `EndStateBlock` that follows still closes the recording that was already
/// open. A stray reset of the recording here would look like a fix and would
/// silently hand the application a block it never recorded.
#[test]
fn begin_state_block_while_recording_is_rejected() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "start lit");
    assert_eq!(h.begin_state_block(), D3D_OK, "first BeginStateBlock");
    // Recorded before the rejection, so a recording that the rejection
    // restarted would lose it and hand back an empty block.
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        0,
        "record LIGHTING=0"
    );
    assert_eq!(
        h.begin_state_block(),
        D3DERR_INVALIDCALL,
        "BeginStateBlock while recording"
    );

    let sb = h.end_state_block();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "back to lit");
    assert_eq!(
        sb.apply(),
        0,
        "Apply the block the rejection did not disturb"
    );
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "the state recorded before the rejected Begin is still in the block"
    );
}

#[test]
fn begin_end_state_block_records_changes() {
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "start lit");
    assert_eq!(h.begin_state_block(), 0, "BeginStateBlock");
    assert_eq!(
        h.set_render_state(D3DRS_LIGHTING, 0),
        0,
        "record LIGHTING=0"
    );
    let sb = h.end_state_block();

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0, "back to lit");
    assert_eq!(sb.apply(), 0, "Apply recorded block");
    assert_eq!(
        h.render_state(D3DRS_LIGHTING),
        0,
        "recorded LIGHTING=0 replayed"
    );
}

/// A full-screen lit quad facing the viewer.
fn facing_quad() -> [LitVertex; 4] {
    [(-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)].map(|(x, y)| LitVertex {
        x,
        y,
        z: 0.5,
        nx: 0.0,
        ny: 0.0,
        nz: -1.0,
    })
}

const fn colour(r: f32, g: f32, b: f32) -> D3DCOLORVALUE {
    D3DCOLORVALUE { r, g, b, a: 1.0 }
}

/// A material of `diffuse` alone.
fn diffuse_material(diffuse: D3DCOLORVALUE) -> D3DMATERIAL9 {
    D3DMATERIAL9 {
        diffuse,
        ..D3DMATERIAL9::default()
    }
}

/// A directional light of `diffuse` shining down +z, onto the quad's face.
fn frontal_light(diffuse: D3DCOLORVALUE) -> D3DLIGHT9 {
    D3DLIGHT9 {
        type_: D3DLIGHT_DIRECTIONAL,
        diffuse,
        direction: D3DVECTOR {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        ..D3DLIGHT9::default()
    }
}

/// Draw the lit quad in a frame of its own and return the centre pixel's `(r, g, b)`.
fn draw_lit(h: &Harness) -> (u32, u32, u32) {
    h.render_once(BLUE, |d| {
        assert_eq!(
            d.draw_primitive_up(D3DPT_TRIANGLESTRIP, 2, &facing_quad()),
            0
        );
    });
    let px = h.read_pixel(320, 240);
    ((px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff)
}

#[test]
fn applied_blocks_restore_the_fixed_function_lights_and_material_a_draw_reads() {
    // Apply writes the captured lights and material straight into the device
    // state. A draw after it must light with the restored values, not with the
    // ones the previous draw uploaded.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_material(&diffuse_material(colour(1.0, 1.0, 1.0))), 0);
    assert_eq!(h.set_light(0, &frontal_light(colour(1.0, 0.0, 0.0))), 0);
    assert_eq!(h.light_enable(0, true), 0);

    // VERTEXSTATE carries the lights.
    let vertex = h.create_state_block(D3DSBT_VERTEXSTATE);
    assert_eq!(h.set_light(0, &frontal_light(colour(0.0, 1.0, 0.0))), 0);
    let (r, g, b) = draw_lit(&h);
    assert!(
        r <= 2 && g >= 0xF0 && b <= 2,
        "live green light, got ({r}, {g}, {b})"
    );
    assert_eq!(vertex.apply(), 0, "Apply VERTEXSTATE");
    let (r, g, b) = draw_lit(&h);
    assert!(
        r >= 0xF0 && g <= 2 && b <= 2,
        "VERTEXSTATE restored the red light, got ({r}, {g}, {b})"
    );

    // ALL carries the material too.
    let all = h.create_state_block(D3DSBT_ALL);
    assert_eq!(h.set_material(&diffuse_material(colour(0.5, 0.0, 0.0))), 0);
    let (r, g, b) = draw_lit(&h);
    assert!(
        r.abs_diff(0x80) <= 2 && g <= 2 && b <= 2,
        "live half-red material, got ({r}, {g}, {b})"
    );
    assert_eq!(all.apply(), 0, "Apply ALL");
    let (r, g, b) = draw_lit(&h);
    assert!(
        r >= 0xF0 && g <= 2 && b <= 2,
        "ALL restored the white material, got ({r}, {g}, {b})"
    );
}

#[test]
fn created_blocks_restore_a_light_past_the_eight_fast_path_slots() {
    // Light 9 lives past the eight slots the fixed-function constants keep
    // per index, but D3D9 addresses lights without a bound, and both block
    // types that carry lights carry it: its parameters, its enable, and the
    // lighting a draw after Apply reads.
    for (sbt, name) in [(D3DSBT_ALL, "ALL"), (D3DSBT_VERTEXSTATE, "VERTEXSTATE")] {
        let h = Harness::new();
        assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_material(&diffuse_material(colour(1.0, 1.0, 1.0))), 0);
        assert_eq!(h.set_light(9, &frontal_light(colour(1.0, 0.0, 0.0))), 0);
        assert_eq!(h.light_enable(9, true), 0);

        let block = h.create_state_block(sbt);
        assert_eq!(h.set_light(9, &frontal_light(colour(0.0, 1.0, 0.0))), 0);
        assert_eq!(h.light_enable(9, false), 0);
        let (r, g, b) = draw_lit(&h);
        assert!(
            r <= 2 && g <= 2 && b <= 2,
            "{name}: light 9 disabled before Apply, got ({r}, {g}, {b})"
        );

        assert_eq!(block.apply(), 0, "Apply {name}");
        let restored = h.light(9).diffuse;
        assert_eq!(
            (restored.r.to_bits(), restored.g.to_bits()),
            (1.0f32.to_bits(), 0.0f32.to_bits()),
            "{name}: GetLight(9) reports the captured red light"
        );
        assert!(h.light_enabled(9), "{name}: light 9 enabled again");
        let (r, g, b) = draw_lit(&h);
        assert!(
            r >= 0xF0 && g <= 2 && b <= 2,
            "{name}: the restored red light 9 lights the quad, got ({r}, {g}, {b})"
        );
    }
}

#[test]
fn created_blocks_leave_a_light_defined_after_the_capture() {
    // A block applies the lights it captured. Light 3, defined and enabled
    // after the block was created, stays defined and enabled through Apply
    // and lights the draw after it.
    for (sbt, name) in [(D3DSBT_ALL, "ALL"), (D3DSBT_VERTEXSTATE, "VERTEXSTATE")] {
        let h = Harness::new();
        assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
        assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
        assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
        h.select_diffuse_stage(0);
        assert_eq!(h.set_material(&diffuse_material(colour(1.0, 1.0, 1.0))), 0);

        let block = h.create_state_block(sbt);
        assert_eq!(h.set_light(3, &frontal_light(colour(0.0, 1.0, 0.0))), 0);
        assert_eq!(h.light_enable(3, true), 0);
        assert_eq!(block.apply(), 0, "Apply {name}");
        assert!(h.light_enabled(3), "{name}: light 3 stays enabled");
        assert_eq!(
            h.light(3).diffuse.g.to_bits(),
            1.0f32.to_bits(),
            "{name}: light 3 keeps its parameters"
        );
        let (r, g, b) = draw_lit(&h);
        assert!(
            r <= 2 && g >= 0xF0 && b <= 2,
            "{name}: light 3 lights the quad, got ({r}, {g}, {b})"
        );
    }
}

/// POSITION float3 on stream 0, nothing else: a declaration the game created.
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

/// What `GetFVF` reports with `decl` bound through `SetVertexDeclaration`, the direct bind.
fn fvf_of_direct_bind(h: &Harness, decl: &VertexDeclaration<'_>) -> u32 {
    assert_eq!(h.set_vertex_declaration(decl), D3D_OK, "direct bind");
    h.fvf()
}

#[test]
fn applied_all_block_keeps_the_fvf_bound_after_a_capture_without_one() {
    // A block created with no declaration bound restores none, so the FVF
    // set after it stays bound through Apply: GetFVF keeps reporting it and
    // a lit draw without a diffuse colour falls back to the material, as the
    // FVF does.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_material(&diffuse_material(colour(1.0, 0.0, 0.0))), 0);
    assert_eq!(h.set_light(0, &frontal_light(colour(1.0, 1.0, 1.0))), 0);
    assert_eq!(h.light_enable(0, true), 0);
    let block = h.create_state_block(D3DSBT_ALL);

    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    assert_eq!(block.apply(), D3D_OK, "Apply ALL");
    assert_eq!(h.fvf(), D3DFVF_XYZ | D3DFVF_NORMAL, "GetFVF after Apply");
    let (r, g, b) = draw_lit(&h);
    assert!(
        r >= 0xF0 && g <= 2 && b <= 2,
        "the FVF's material fallback lights the quad red, got ({r}, {g}, {b})"
    );
}

#[test]
fn applied_recorded_declaration_reports_the_fvf_of_the_direct_bind() {
    // A recorded SetVertexDeclaration applied over an FVF leaves the device
    // where SetVertexDeclaration of the same declaration would.
    const FVF: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
    let h = Harness::new();
    let position_only = h.create_vertex_declaration(&position_only_decl());
    let direct = fvf_of_direct_bind(&h, &position_only);

    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.set_vertex_declaration(&position_only), D3D_OK, "record");
    let block = h.end_state_block();
    assert_eq!(h.set_fvf(FVF), D3D_OK, "SetFVF");
    assert_eq!(block.apply(), D3D_OK, "Apply");
    assert_eq!(
        VertexDeclaration::from_raw(h.vertex_declaration_raw()).as_ptr(),
        position_only.as_ptr(),
        "Apply binds the recorded declaration"
    );
    assert_eq!(h.fvf(), direct, "GetFVF after Apply");
}

#[test]
fn captured_recorded_set_fvf_restores_the_declaration_bound_at_capture() {
    // A recorded SetFVF is a declaration change, and a Capture refreshes it
    // from the bound declaration, even one the game created.
    const RECORDED: u32 = D3DFVF_XYZ | D3DFVF_DIFFUSE;
    const LATER: u32 = D3DFVF_XYZ | D3DFVF_NORMAL;
    let h = Harness::new();
    let position_only = h.create_vertex_declaration(&position_only_decl());
    let direct = fvf_of_direct_bind(&h, &position_only);

    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.set_fvf(RECORDED), D3D_OK, "record SetFVF");
    let block = h.end_state_block();
    assert_eq!(h.set_vertex_declaration(&position_only), D3D_OK);
    assert_eq!(block.capture(), D3D_OK, "Capture");
    assert_eq!(h.set_fvf(LATER), D3D_OK, "SetFVF after the capture");
    assert_eq!(block.apply(), D3D_OK, "Apply");
    assert_eq!(
        VertexDeclaration::from_raw(h.vertex_declaration_raw()).as_ptr(),
        position_only.as_ptr(),
        "Apply binds the declaration bound at the capture"
    );
    assert_eq!(h.fvf(), direct, "GetFVF after Apply");
}

/// Assert `light` is the default light `LightEnable` creates: white, directional, down +z.
fn assert_default_light(light: &D3DLIGHT9, message: &str) {
    assert_eq!(light.type_, D3DLIGHT_DIRECTIONAL, "{message}: type");
    let diffuse = [
        light.diffuse.r,
        light.diffuse.g,
        light.diffuse.b,
        light.diffuse.a,
    ];
    assert_eq!(
        diffuse.map(f32::to_bits),
        [1.0f32, 1.0, 1.0, 0.0].map(f32::to_bits),
        "{message}: diffuse"
    );
    let direction = [light.direction.x, light.direction.y, light.direction.z];
    assert_eq!(
        direction.map(f32::to_bits),
        [0.0f32, 0.0, 1.0].map(f32::to_bits),
        "{message}: direction"
    );
}

#[test]
fn recorded_light_enable_of_an_unset_light_applies_the_default_light() {
    // A recorded block holds a light's parameters with its enable, so enabling
    // a light the recording never set records the default light `LightEnable`
    // creates: Apply writes it over the device's light, and a Capture
    // refreshes it from the device. A recorded disable defines nothing.
    let h = Harness::new();
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 1), 0);
    assert_eq!(h.set_render_state(D3DRS_CULLMODE, D3DCULL_NONE), 0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_NORMAL), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_material(&diffuse_material(colour(1.0, 1.0, 1.0))), 0);
    let red = frontal_light(colour(1.0, 0.0, 0.0));
    let green = frontal_light(colour(0.0, 1.0, 0.0));
    assert_eq!(h.set_light(0, &red), 0);
    assert_eq!(h.light_enable(0, false), 0);
    assert_eq!(h.set_light(1, &green), 0);
    assert_eq!(h.light_enable(1, true), 0);

    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.light_enable(0, true), D3D_OK, "record enable of light 0");
    assert_eq!(
        h.light_enable(1, false),
        D3D_OK,
        "record disable of light 1"
    );
    let block = h.end_state_block();
    assert!(!h.light_enabled(0), "the recording left light 0 disabled");

    assert_eq!(block.apply(), D3D_OK, "Apply");
    assert_default_light(&h.light(0), "light 0 after Apply");
    assert!(h.light_enabled(0), "light 0 enabled by Apply");
    assert_eq!(
        h.light(1).diffuse.g.to_bits(),
        1.0f32.to_bits(),
        "a recorded disable keeps light 1's parameters"
    );
    assert!(!h.light_enabled(1), "light 1 disabled by Apply");
    let (r, g, b) = draw_lit(&h);
    assert!(
        r >= 0xF0 && g >= 0xF0 && b >= 0xF0,
        "the default white light lights the quad, got ({r}, {g}, {b})"
    );

    // Capture refreshes the recorded light 0 from the device, parameters too.
    assert_eq!(h.set_light(0, &green), 0);
    assert_eq!(h.light_enable(1, false), 0);
    assert_eq!(block.capture(), D3D_OK, "Capture");
    assert_eq!(h.set_light(0, &red), 0);
    assert_eq!(h.light_enable(0, false), 0);
    assert_eq!(block.apply(), D3D_OK, "Apply after Capture");
    let light = h.light(0);
    assert_eq!(
        [light.diffuse.r, light.diffuse.g].map(f32::to_bits),
        [0.0f32, 1.0].map(f32::to_bits),
        "Apply writes light 0 as captured"
    );
    assert!(h.light_enabled(0), "light 0 enabled as captured");
}

#[test]
fn recorded_light_enable_after_a_recorded_set_light_keeps_its_parameters() {
    // Enabling a light the recording has set records no default over it.
    let h = Harness::new();
    let blue = frontal_light(colour(0.0, 0.0, 1.0));
    assert_eq!(h.begin_state_block(), D3D_OK, "BeginStateBlock");
    assert_eq!(h.set_light(3, &blue), D3D_OK, "record SetLight");
    assert_eq!(h.light_enable(3, true), D3D_OK, "record enable");
    let block = h.end_state_block();
    assert_eq!(block.apply(), D3D_OK, "Apply");
    let light = h.light(3);
    assert_eq!(
        [light.diffuse.r, light.diffuse.b].map(f32::to_bits),
        [0.0f32, 1.0].map(f32::to_bits),
        "light 3 keeps the recorded parameters"
    );
    assert!(h.light_enabled(3), "light 3 enabled");
}
