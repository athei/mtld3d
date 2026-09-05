//! Query objects: the EVENT fence path (issue → get-data signalled).

use mtld3d_tests::{Harness, PosColorVertex};
use mtld3d_types::{
    D3DFVF_DIFFUSE, D3DFVF_XYZ, D3DGETDATA_FLUSH, D3DISSUE_BEGIN, D3DISSUE_END, D3DPT_TRIANGLELIST,
    D3DQUERYTYPE_EVENT, D3DQUERYTYPE_OCCLUSION, D3DQUERYTYPE_TIMESTAMP, D3DRS_LIGHTING,
};

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
