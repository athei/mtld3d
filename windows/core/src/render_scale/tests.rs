use super::*;

#[test]
fn identity_changes_nothing() {
    let s = RenderScale::IDENTITY;
    assert!(s.is_identity());
    for d in [0, 1, 7, 640, 1920, 16384] {
        assert_eq!(s.dimension(d), d);
    }
    assert_eq!(s.rect(13, 27, 640, 480), (13, 27, 640, 480));
}

#[test]
fn dimension_halves() {
    let s = RenderScale::from_percent(50);
    assert_eq!(s.dimension(1920), 960);
    assert_eq!(s.dimension(1080), 540);
}

#[test]
fn dimension_never_collapses_to_zero() {
    let s = RenderScale::from_percent(25);
    assert_eq!(s.dimension(1), 1);
    assert_eq!(s.dimension(2), 1);
    // Zero in, zero out: the caller rejects those separately.
    assert_eq!(s.dimension(0), 0);
}

#[test]
fn from_percent_clamps_out_of_range() {
    assert_eq!(RenderScale::from_percent(0), RenderScale::from_percent(1));
    assert_eq!(RenderScale::from_percent(10_000), RenderScale::IDENTITY);
}

#[test]
fn abutting_rects_stay_abutting() {
    // Three tiles sharing edges at 100 and 300 must still share them
    // after scaling, at a ratio that does not divide evenly.
    let s = RenderScale::from_percent(75);
    let (ax, _, aw, _) = s.rect(0, 0, 100, 10);
    let (bx, _, bw, _) = s.rect(100, 0, 200, 10);
    let (cx, _, cw, _) = s.rect(300, 0, 50, 10);
    assert_eq!(ax + aw, bx, "tile A must end exactly where B starts");
    assert_eq!(bx + bw, cx, "tile B must end exactly where C starts");
    assert_eq!(cx + cw, s.rect(0, 0, 350, 10).2, "total width preserved");
}

#[test]
fn rect_scales_origin_and_extent_together() {
    let s = RenderScale::from_percent(50);
    assert_eq!(s.rect(100, 200, 400, 300), (50, 100, 200, 150));
}

#[test]
fn signed_rect_matches_the_unsigned_one() {
    // `Clear` carries D3DRECTs and the scissor carries (x, y, w, h); both
    // must land on the same pixels or a clipped clear seams against the
    // scissor that clipped it.
    let scale = RenderScale::from_percent(75);
    let (x, y, width, height) = scale.rect(40, 24, 120, 80);
    assert_eq!(
        scale.rect_edges_i32((40, 24, 160, 104)),
        (
            x.cast_signed(),
            y.cast_signed(),
            (x + width).cast_signed(),
            (y + height).cast_signed()
        )
    );
}

#[test]
fn signed_rect_identity_is_exact() {
    let s = RenderScale::IDENTITY;
    assert_eq!(s.rect_edges_i32((-5, 0, 640, 480)), (-5, 0, 640, 480));
}

#[test]
fn signed_rect_clamps_negative_edges() {
    // Off-attachment edges are the caller's to clip; scaling must not
    // wrap or panic on them.
    let s = RenderScale::from_percent(50);
    assert_eq!(s.rect_edges_i32((-100, -20, 200, 100)), (0, 0, 100, 50));
}

#[test]
fn never_enlarges() {
    // Supersampling is not offered: the present-side scaler only
    // upscales, so a render bigger than the drawable has no path home.
    let s = RenderScale::from_percent(200);
    assert!(s.is_identity());
    assert_eq!(s.dimension(960), 960);
}
