//! State-block capture/apply round-trip.

use mtld3d_tests::Harness;
use mtld3d_types::{
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DLIGHT_DIRECTIONAL, D3DLIGHT9, D3DRS_ALPHABLENDENABLE,
    D3DRS_LIGHTING, D3DSAMP_MINFILTER, D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DSBT_VERTEXSTATE,
    D3DTEXF_LINEAR, D3DTEXF_POINT,
};

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
