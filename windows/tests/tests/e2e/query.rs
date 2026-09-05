//! Query objects: the EVENT fence path (issue → get-data signalled).

use mtld3d_tests::{Harness, PosColorVertex, Query};
use mtld3d_types::{
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN, D3DISSUE_END, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_EVENT, D3DQUERYTYPE_OCCLUSION, D3DQUERYTYPE_TIMESTAMP, D3DRS_LIGHTING,
};

/// A full-frame quad in clip space, one solid colour.
const FULL_FRAME_QUAD: [PosColorVertex; 6] = [
    quad_vertex(-1.0, 1.0),
    quad_vertex(1.0, 1.0),
    quad_vertex(-1.0, -1.0),
    quad_vertex(1.0, 1.0),
    quad_vertex(1.0, -1.0),
    quad_vertex(-1.0, -1.0),
];

const fn quad_vertex(x: f32, y: f32) -> PosColorVertex {
    PosColorVertex {
        x,
        y,
        z: 0.5,
        color: 0xFF00_FF00,
    }
}

/// Put the device in the fixed-function state the counting draws below need.
fn arm_for_counting_draws(h: &Harness) {
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
}

/// Draw the full-frame quad, asserting the call succeeded.
fn draw_full_frame(h: &Harness, what: &str) {
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &FULL_FRAME_QUAD),
        0,
        "{what}"
    );
}

/// Read a finished occlusion count, asserting `GetData` reported a result.
fn occlusion_count(q: &Query<'_>, what: &str) -> u32 {
    let (hr, count) = q.data_u32(D3DGETDATA_FLUSH);
    assert_eq!(hr, 0, "GetData(FLUSH) for {what}");
    count
}

/// Assert a count is two full frames' worth, within the rounding a scale costs.
fn assert_two_full_frames(count: u32, dims: (u32, u32), what: &str) {
    let expected = 2 * dims.0 * dims.1;
    assert!(
        count.abs_diff(expected) <= expected / 100,
        "{what}: expected both draws counted (~{expected} samples), got {count}"
    );
}

#[test]
fn event_query_signals() {
    let h = Harness::new();
    // Null-out probe: a supported type returns S_OK.
    assert_eq!(
        h.query_supported(D3DQUERYTYPE_EVENT),
        0,
        "EVENT CreateQuery probe"
    );

    let q = h
        .create_query(D3DQUERYTYPE_EVENT)
        .expect("EVENT query is supported");
    assert_eq!(q.data_size(), 4, "EVENT result is a 4-byte BOOL");

    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    let (hr, signalled) = q.data_u32(0);
    assert_eq!(hr, 0, "GetData");
    assert_eq!(signalled, 1, "EVENT query reports signalled");
}

#[test]
fn occlusion_query_counts_visible_pixels() {
    let h = Harness::new();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };

    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: 0xFF00_FF00,
    };
    let quad = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "visible draw"
    );
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let (hr, count) = q.data_u32(D3DGETDATA_FLUSH);
    assert_eq!(hr, 0, "GetData(FLUSH)");
    assert!(
        count > 1000,
        "fullscreen quad covers many samples, got {count}"
    );
}

#[test]
fn occlusion_query_counts_in_reported_pixels_under_the_scale() {
    // A game reads an occlusion count against the pixels it was told the
    // back buffer has: a lens flare fades by a disc's area in those pixels, a
    // threshold is stated in them. Under `render.scale` the rasterizer
    // produces fewer samples, so the count is scaled back up into the
    // reported space before the game reads it. A quad covering the whole
    // frame counts exactly the reported pixel count.
    //
    // Pins its own scale (a clean half, so the render extent is exact) rather
    // than inheriting the run's: at the identity there is nothing to convert,
    // and this has to fail in the ordinary `make test` if it regresses.
    let h = Harness::with_config("render.scale=0.5");
    let (width, height) = h.dims();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    assert_eq!(h.set_render_state(D3DRS_LIGHTING, 0), 0);
    h.select_diffuse_stage(0);
    assert_eq!(h.set_fvf(D3DFVF_XYZ | D3DFVF_DIFFUSE), 0);
    let v = |x: f32, y: f32| PosColorVertex {
        x,
        y,
        z: 0.5,
        color: 0xFF00_FF00,
    };
    let quad = [
        v(-1.0, 1.0),
        v(1.0, 1.0),
        v(-1.0, -1.0),
        v(1.0, 1.0),
        v(1.0, -1.0),
        v(-1.0, -1.0),
    ];

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &quad),
        0,
        "fullscreen draw"
    );
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let (hr, count) = q.data_u32(D3DGETDATA_FLUSH);
    assert_eq!(hr, 0, "GetData(FLUSH)");
    assert_eq!(
        count,
        width * height,
        "a fullscreen quad counts the reported pixels, not the rasterized samples"
    );
}

#[test]
fn timestamp_query_contract() {
    let h = Harness::new();
    // TIMESTAMP is not backed by a Metal counter here; pin whatever the device
    // reports (supported → a usable object, or unsupported → no object).
    let supported = h.query_supported(D3DQUERYTYPE_TIMESTAMP) == 0;
    assert_eq!(
        h.create_query(D3DQUERYTYPE_TIMESTAMP).is_some(),
        supported,
        "CreateQuery(TIMESTAMP) agrees with the null-out probe",
    );
}

#[test]
fn occlusion_count_survives_a_pass_split_between_begin_and_end() {
    // A `Clear` reaching a pass that an occlusion query is counting on ends
    // that pass, so every draw after it lands on a Metal encoder of its own.
    // A render encoder starts with visibility counting off, so the count has
    // to be re-armed on the new pass or the rest of the span reads as
    // occluded. `query.flushImmediate=false` so `GetData` reports the counted
    // result rather than the permissive stub the fence-only reading gets.
    let h = Harness::with_config("query.flushImmediate=false");
    let dims = h.dims();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "draw before the split");
    assert_eq!(h.clear_target(0xFF00_0000), 0, "clear splits the pass");
    draw_full_frame(&h, "draw after the split");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_two_full_frames(
        occlusion_count(&q, "the split span"),
        dims,
        "a pass split inside the span",
    );
}

#[test]
fn occlusion_count_survives_a_flush_between_begin_and_end() {
    // Reading a *closed* query with `D3DGETDATA_FLUSH` submits the frame
    // being recorded, which lands in the middle of the still-open query's
    // span: its two halves count into two different frames' slot arrays. The
    // span has to be cut at the boundary and reopened in the continuation, or
    // the sum reads against the wrong buffer.
    let h = Harness::with_config("query.flushImmediate=false");
    let dims = h.dims();
    let Some(closed) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    let Some(open) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(closed.issue(D3DISSUE_BEGIN), 0);
    assert_eq!(closed.issue(D3DISSUE_END), 0);
    assert_eq!(open.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "draw before the flush");
    // The blocking read of the closed query is what submits mid-span.
    assert_eq!(
        occlusion_count(&closed, "the closed query"),
        0,
        "a span with no draw in it counts nothing"
    );
    draw_full_frame(&h, "draw after the flush");
    assert_eq!(open.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_two_full_frames(
        occlusion_count(&open, "the flushed span"),
        dims,
        "a submit inside the span",
    );
}

#[test]
fn occlusion_query_past_the_slot_budget_reads_fully_visible() {
    // The per-frame slot budget is finite (two slots per BEGIN/END pair plus
    // one per pass the span crosses). A query that gets no slot and did draw
    // cannot be counted, and the answer for "unknown" is the permissive
    // `u32::MAX`: reporting the zero its empty span sums to would read as
    // full occlusion and make a title cull geometry it should draw. A query
    // that drew nothing is not unknown at all, budget or no budget, and
    // reports the zero it counted.
    //
    // The filler pair count is comfortably past the budget rather than
    // exactly at it, so the span that follows is starved even if the budget
    // grows; a budget that grew past this fails the test rather than quietly
    // stopping to test the fallback.
    const FILLERS: usize = 700;

    let h = Harness::with_config("query.flushImmediate=false");
    let fillers: Vec<Query<'_>> = (0..FILLERS)
        .map(|_| {
            h.create_query(D3DQUERYTYPE_OCCLUSION)
                .expect("OCCLUSION query should be supported")
        })
        .collect();
    let Some(starved) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    for q in &fillers {
        assert_eq!(q.issue(D3DISSUE_BEGIN), 0);
        assert_eq!(q.issue(D3DISSUE_END), 0);
    }
    assert_eq!(starved.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "draw the starved span cannot count");
    assert_eq!(starved.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(
        occlusion_count(&fillers[0], "the first filler"),
        0,
        "a query that got its slots and saw no draw counts no samples"
    );
    assert_eq!(
        occlusion_count(&fillers[FILLERS - 1], "the last filler"),
        0,
        "a query past the budget that saw no draw still counts no samples"
    );
    assert_eq!(
        occlusion_count(&starved, "the starved span"),
        u32::MAX,
        "a query the frame had no slot left for, with a draw in it, reads fully visible"
    );
}
