//! State-block capture/apply round-trip.

use mtld3d_tests::{Harness, PosColorVertex, VertexDeclaration, assert_pixel_eq};
use mtld3d_types::{
    D3D_OK, D3DCULL_NONE, D3DERR_INVALIDCALL, D3DFMT_A8R8G8B8, D3DFVF_DIFFUSE, D3DFVF_XYZ,
    D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DPOOL_MANAGED, D3DPT_TRIANGLELIST, D3DRECT,
    D3DRS_ALPHABLENDENABLE, D3DRS_CULLMODE, D3DRS_LIGHTING, D3DRS_SCISSORTESTENABLE,
    D3DSAMP_DMAPOFFSET, D3DSAMP_MINFILTER, D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DSBT_VERTEXSTATE,
    D3DTEXF_LINEAR, D3DTEXF_POINT, D3DVIEWPORT9,
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
