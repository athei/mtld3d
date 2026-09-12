//! Query objects: the EVENT fence path (issue → get-data signalled).

use mtld3d_tests::{Harness, HarnessConfig, PosColorVertex, Query};
use mtld3d_types::{
    D3DCLEAR_TARGET, D3DCLEAR_ZBUFFER, D3DCMP_LESS, D3DFMT_D24S8, D3DFMT_X8R8G8B8, D3DFVF_DIFFUSE,
    D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN, D3DISSUE_END, D3DLOCK_DISCARD,
    D3DLOCK_NOOVERWRITE, D3DPOOL_DEFAULT, D3DPT_TRIANGLELIST, D3DQUERYTYPE_EVENT,
    D3DQUERYTYPE_OCCLUSION, D3DQUERYTYPE_TIMESTAMP, D3DRS_LIGHTING, D3DRS_ZENABLE, D3DRS_ZFUNC,
    D3DUSAGE_DYNAMIC, D3DUSAGE_WRITEONLY, S_FALSE,
};

/// Frames a pending EVENT query is given to retire before the test fails.
const EVENT_POLL_LIMIT: u32 = 16;

/// A full-frame quad in clip space at depth `z`, one solid colour.
const fn full_frame_quad(z: f32) -> [PosColorVertex; 6] {
    [
        quad_vertex(-1.0, 1.0, z),
        quad_vertex(1.0, 1.0, z),
        quad_vertex(-1.0, -1.0, z),
        quad_vertex(1.0, 1.0, z),
        quad_vertex(1.0, -1.0, z),
        quad_vertex(-1.0, -1.0, z),
    ]
}

/// The quad every counting draw that has no depth buffer under it uses.
const FULL_FRAME_QUAD: [PosColorVertex; 6] = full_frame_quad(0.5);

const fn quad_vertex(x: f32, y: f32, z: f32) -> PosColorVertex {
    PosColorVertex {
        x,
        y,
        z,
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

/// Draw the full-frame quad at depth `z`, asserting the call succeeded.
fn draw_full_frame_at(h: &Harness, z: f32, what: &str) {
    assert_eq!(
        h.draw_primitive_up(D3DPT_TRIANGLELIST, 2, &full_frame_quad(z)),
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

/// Assert a count is `frames` full frames' worth, within the rounding a scale costs.
fn assert_full_frames(count: u32, frames: u32, dims: (u32, u32), what: &str) {
    let expected = frames * dims.0 * dims.1;
    assert!(
        count.abs_diff(expected) <= expected / 100,
        "{what}: expected {frames} full frame(s) counted (~{expected} samples), got {count}"
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

    // An EVENT query reports completion once the GPU has retired the work it
    // was issued after, so the answer is polled rather than assumed: an
    // implementation that reports completion on the first call is one that
    // never waits, and an application recycling storage behind this fence
    // would overwrite what a queued draw still reads.
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    let mut signalled = 0;
    let mut hr = S_FALSE;
    for _ in 0..EVENT_POLL_LIMIT {
        let (poll_hr, value) = q.data_u32(D3DGETDATA_FLUSH);
        hr = poll_hr;
        signalled = value;
        assert!(hr == 0 || hr == S_FALSE, "GetData reported {hr:#x}");
        // The BOOL tracks the status: pending is FALSE, completed is TRUE.
        assert_eq!(
            u32::from(hr == 0),
            signalled,
            "the result disagrees with the status it was returned with",
        );
        if hr == 0 {
            break;
        }
        h.render_once(0xFF00_0000, |_| {});
    }
    assert_eq!(hr, 0, "EVENT query never reported completion");
    assert_eq!(signalled, 1, "a completed EVENT query reports signalled");
}

#[test]
fn occlusion_query_counts_visible_pixels() {
    // The result is the samples the draws inside the span produced, so a quad
    // covering the frame counts the frame's pixels. `query.flushImmediate` is
    // pinned false rather than inherited: the immediate answer is a stub that
    // reports every span fully visible, and a run that turned it on would
    // satisfy a loose assertion without a single slot being summed.
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
    draw_full_frame(&h, "visible draw");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_full_frames(
        occlusion_count(&q, "the visible span"),
        1,
        dims,
        "a quad covering the frame",
    );
}

#[test]
fn status_only_occlusion_poll_reports_readiness_and_flushes() {
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
    draw_full_frame(&h, "the counted draw");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);

    let status_only = q.status(0);
    let (buffered, _) = q.data_u32(0);
    assert_eq!(status_only, 1, "status-only poll before submission");
    assert_eq!(status_only, buffered, "both poll forms report readiness");

    assert_eq!(
        q.status(D3DGETDATA_FLUSH),
        0,
        "status-only FLUSH submits the pending query"
    );
    let (ready, count) = q.data_u32(0);
    assert_eq!(ready, 0, "the submitted query is ready");
    assert_full_frames(
        count,
        1,
        dims,
        "the query made ready by the status-only FLUSH",
    );
}

#[test]
fn occlusion_query_counts_nothing_for_a_depth_occluded_draw() {
    // What a title acts on is the *visible* sample count: a draw whose every
    // sample fails the depth test contributes nothing, which is the whole
    // reason to issue the query. Two spans in one frame, the near one counting
    // a full frame, so the far one's zero is the depth test's answer rather
    // than a counter that never ran or a slot that was never summed.
    let h = Harness::create(&HarnessConfig {
        depth_format: Some(D3DFMT_D24S8),
        config_entries: "query.flushImmediate=false",
        ..HarnessConfig::default()
    });
    let dims = h.dims();
    let Some(near) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    let Some(far) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);
    assert_eq!(h.set_render_state(D3DRS_ZENABLE, 1), 0, "ZENABLE");
    assert_eq!(h.set_render_state(D3DRS_ZFUNC, D3DCMP_LESS), 0, "ZFUNC");

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(
        h.clear(D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xFF00_0000, 1.0, 0),
        0,
        "clear colour and depth"
    );
    assert_eq!(
        near.issue(D3DISSUE_BEGIN),
        0,
        "Issue(BEGIN) for the near span"
    );
    draw_full_frame_at(&h, 0.5, "the near draw");
    assert_eq!(near.issue(D3DISSUE_END), 0, "Issue(END) for the near span");
    assert_eq!(
        far.issue(D3DISSUE_BEGIN),
        0,
        "Issue(BEGIN) for the far span"
    );
    draw_full_frame_at(&h, 0.9, "the far draw");
    assert_eq!(far.issue(D3DISSUE_END), 0, "Issue(END) for the far span");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_full_frames(
        occlusion_count(&near, "the near span"),
        1,
        dims,
        "a draw in front of the cleared depth",
    );
    assert_eq!(
        occlusion_count(&far, "the far span"),
        0,
        "a draw every sample of which fails the depth test counts no samples"
    );
}

#[test]
fn occlusion_query_flush_poll_stubs_the_count_under_flush_immediate() {
    // `query.flushImmediate=true` gives up the count to save the API-thread
    // time the spec-correct wait costs, and answers a `D3DGETDATA_FLUSH` poll
    // of a pending query with the permissive `u32::MAX` instead. The poll sits
    // inside the recording frame, before anything is submitted, so the query
    // is pending for certain and the stub is the only answer the arm can give.
    let h = Harness::with_config("query.flushImmediate=true");
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "the draw the poll gives up counting");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");

    assert_eq!(
        occlusion_count(&q, "the stubbed poll"),
        u32::MAX,
        "the immediate answer reports fully visible instead of the count"
    );
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);
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
    let h = Harness::with_config("render.scale=0.5;query.flushImmediate=false");
    let (width, height) = h.dims();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "fullscreen draw");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_eq!(
        occlusion_count(&q, "the scaled span"),
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

    assert_full_frames(
        occlusion_count(&q, "the split span"),
        2,
        dims,
        "a pass split inside the span",
    );
}

#[test]
fn occlusion_count_survives_a_render_target_round_trip_between_begin_and_end() {
    // A `Clear` is one way into a fresh pass inside a span; a render-target
    // change is the other, and the one a title takes when it renders a shadow
    // map or a reflection in the middle of the span it is measuring. Binding
    // another target ends the pass, and binding the first one back leaves the
    // next draw to open a pass of its own, which starts with visibility
    // counting off and has to be armed again.
    let h = Harness::with_config("query.flushImmediate=false");
    let dims = h.dims();
    // At the back buffer's own size, so the round trip changes the attachment
    // and nothing else: `SetRenderTarget` snaps the viewport to the target it
    // binds, and a target of another size would put a viewport restore in the
    // way of what this test is about.
    let offscreen = h.create_render_target(dims.0, dims.1, D3DFMT_X8R8G8B8);
    let back = h.back_buffer(0);
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "draw before the round trip");
    assert_eq!(
        h.set_render_target(0, &offscreen),
        0,
        "bind the offscreen target"
    );
    assert_eq!(
        h.set_render_target(0, &back),
        0,
        "bind the back buffer back"
    );
    draw_full_frame(&h, "draw after the round trip");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    assert_full_frames(
        occlusion_count(&q, "the round-trip span"),
        2,
        dims,
        "a render-target round trip inside the span",
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

    assert_full_frames(
        occlusion_count(&open, "the flushed span"),
        2,
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

#[test]
fn an_ended_span_is_finalized_by_the_reset_that_flushes_its_frame() {
    // A resizing `Reset` flushes the frame the application is recording,
    // which is the frame carrying the last `Issue(END)` before it, waits for
    // the GPU, and then takes the visibility pool down. The count has to be
    // summed out of that pool while it is still there: a query left `Pending`
    // is one the application still holds, and every later `GetData` for it
    // answers `S_FALSE`. Under `query.flushImmediate=false` that makes the
    // blocking arm a poll loop with no end, which is why the key is pinned
    // here rather than left at the permissive stub the default gives.
    let h = Harness::with_config("query.flushImmediate=false");
    let (width, height) = h.dims();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "the counted draw");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    // No Present: the Reset's own flush is what submits the counting frame,
    // so the span is queued after that frame's intake has already run.
    assert_eq!(
        h.reset(width / 2, height / 2),
        0,
        "resize Reset must succeed"
    );

    let count = occlusion_count(&q, "the span the Reset flushed");
    let expected = width * height;
    assert!(
        count.abs_diff(expected) <= expected / 100,
        "the span counted its draw against the pre-Reset target \
         (~{expected} samples), got {count}"
    );
}

#[test]
fn occlusion_count_survives_a_reset_between_begin_and_end() {
    // A resizing `Reset` waits for the GPU and then takes the visibility pool
    // down, in the middle of a span the application left open across it. That
    // is the cut a submit boundary makes, so the span has to continue in the
    // frame after the `Reset`: a query the `Reset` forgot arms no pass there,
    // and its `Issue(END)` builds a slot range out of the frame that is gone,
    // answering with one frame's count or with a zero that reads as full
    // occlusion. A same-size `Reset` keeps the pool and its span is the
    // submit-boundary case above.
    let h = Harness::with_config("query.flushImmediate=false");
    let (width, height) = h.dims();
    let Some(q) = h.create_query(D3DQUERYTYPE_OCCLUSION) else {
        panic!("OCCLUSION query should be supported");
    };
    arm_for_counting_draws(&h);

    assert!(h.pump(), "WM_QUIT");
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    assert_eq!(q.issue(D3DISSUE_BEGIN), 0, "Issue(BEGIN)");
    draw_full_frame(&h, "the draw before the Reset");
    assert_eq!(h.end_scene(), 0);
    // No Present: the flush the `Reset` performs is what submits the frame
    // carrying the first half of the span.
    assert_eq!(
        h.reset(width / 2, height / 2),
        0,
        "resize Reset must succeed"
    );
    let (reset_width, reset_height) = h.dims();

    // `Reset` restores the device to its state defaults, the fixed-function
    // setup the counted draw needs included.
    arm_for_counting_draws(&h);
    assert_eq!(h.begin_scene(), 0);
    assert_eq!(h.clear_target(0xFF00_0000), 0);
    draw_full_frame(&h, "the draw after the Reset");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");
    assert_eq!(h.end_scene(), 0);
    assert_eq!(h.present(), 0);

    let expected = width * height + reset_width * reset_height;
    let count = occlusion_count(&q, "the span the Reset cut");
    assert!(
        count.abs_diff(expected) <= expected / 100,
        "both halves of the span counted, the second against the target the \
         Reset made (~{expected} samples), got {count}"
    );
}

/// A short EVENT read fills what was asked for and nothing past it.
///
/// D3D9 copies the result into the caller's buffer at the caller's size, so a
/// two-byte read takes the low half of the BOOL and leaves the rest of the
/// buffer as the caller left it. Wine pins the same shape for occlusion.
#[test]
fn a_short_event_read_leaves_the_bytes_past_it_alone() {
    let h = Harness::new();
    let q = h
        .create_query(D3DQUERYTYPE_EVENT)
        .expect("EVENT query is supported");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");

    // Drive it to completion first, so the value under test is the TRUE the
    // caller is owed rather than a pending FALSE.
    let mut hr = S_FALSE;
    for _ in 0..EVENT_POLL_LIMIT {
        hr = q.data_bytes(&mut [0u8; 4], D3DGETDATA_FLUSH);
        if hr == 0 {
            break;
        }
        h.render_once(0xFF00_0000, |_| {});
    }
    assert_eq!(hr, 0, "EVENT query never reported completion");

    let mut buf = [0xFFu8; 4];
    assert_eq!(
        q.data_bytes(&mut buf[..2], D3DGETDATA_FLUSH),
        0,
        "2-byte read"
    );
    assert_eq!(
        u16::from_le_bytes([buf[0], buf[1]]),
        1,
        "the low half of the BOOL is the signalled value",
    );
    assert_eq!(
        [buf[2], buf[3]],
        [0xFF, 0xFF],
        "bytes past the requested size were modified",
    );
}

/// An EVENT query gates reuse of a buffer a queued draw still reads.
///
/// This is the fence's whole purpose: a title recycles dynamic vertex storage
/// behind it, so reporting completion early hands back a range a queued draw
/// is still reading. The buffer is `D3DPOOL_DEFAULT | D3DUSAGE_DYNAMIC`, whose
/// pages the GPU reads directly, and the refill takes `D3DLOCK_NOOVERWRITE`,
/// which writes in place. Answering the poll before the draw retires puts the
/// second colour under the first draw.
#[test]
fn an_event_query_gates_reuse_of_a_buffer_a_draw_is_reading() {
    const DRAWN: u32 = 0xFF00_FF00;
    const REFILL: u32 = 0xFFFF_0000;
    const BACKGROUND: u32 = 0xFF00_00FF;

    let h = Harness::new();
    let stride = u32::try_from(size_of::<PosColorVertex>()).expect("stride fits u32");
    let vb = h.create_vertex_buffer(
        stride * 6,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        D3DFVF_XYZ | D3DFVF_DIFFUSE,
        D3DPOOL_DEFAULT,
    );
    arm_for_counting_draws(&h);
    assert_eq!(h.set_stream_source(0, &vb, 0, stride), 0, "SetStreamSource");

    let quad = |color: u32| {
        let mut q = FULL_FRAME_QUAD;
        for v in &mut q {
            v.color = color;
        }
        q
    };
    vb.lock(0, 0, D3DLOCK_DISCARD).write(&quad(DRAWN));

    let q = h
        .create_query(D3DQUERYTYPE_EVENT)
        .expect("EVENT query is supported");
    assert_eq!(h.begin_scene(), 0, "BeginScene");
    assert_eq!(h.clear_target(BACKGROUND), 0, "Clear");
    assert_eq!(
        h.draw_primitive(D3DPT_TRIANGLELIST, 0, 2),
        0,
        "the draw whose vertices are about to be overwritten",
    );
    assert_eq!(h.end_scene(), 0, "EndScene");
    assert_eq!(q.issue(D3DISSUE_END), 0, "Issue(END)");

    // Poll without the flush flag, the way a title fencing its own reuse does.
    // Bounded by wall clock rather than iterations, so a slow GPU cannot fail
    // it for being slow.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if q.status(0) == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the EVENT query never reported completion",
        );
        std::thread::yield_now();
    }

    // The fence said the GPU is done, so this range is the application's again.
    vb.lock(0, 0, D3DLOCK_NOOVERWRITE).write(&quad(REFILL));

    assert_eq!(
        h.read_pixel(320, 240),
        DRAWN,
        "the refill landed under a draw the fence said had finished",
    );
}
