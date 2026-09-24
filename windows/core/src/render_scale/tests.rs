//! Unit tests for logical-to-render resolution conversion.
//!
//! Pins the properties the call sites depend on: 100% is an exact identity,
//! `dimension` never collapses a small target to zero, and an out-of-range
//! percentage clamps (rendering above the presented size is not offered). The
//! rect cases cover edge-based scaling, so abutting rects still share an edge,
//! a rect spanning the logical extent spans the texture that extent creates,
//! and `rect` and `rect_edges_i32` land on the same pixels.
//! `factor` carries the same ratio as a float, for lengths the shaders scale.

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

/// A full-target rect ends exactly where the texture it addresses ends.
///
/// A target's Metal texture is `dimension` of its logical extent and a
/// full-target viewport or `Clear` rect goes through `rect`, so the two must
/// agree at the far edge or the last texel column or row is never drawn.
#[test]
fn a_full_target_rect_spans_the_whole_texture() {
    for percent in 1..=100 {
        let s = RenderScale::from_percent(percent);
        for n in 0..=4096 {
            let d = s.dimension(n);
            assert_eq!(s.rect(0, 0, n, n), (0, 0, d, d), "{percent}% of {n}");
            let (d, n) = (d.cast_signed(), n.cast_signed());
            assert_eq!(
                s.rect_edges_i32((0, 0, n, n)),
                (0, 0, d, d),
                "{percent}% of {n}, signed"
            );
        }
    }
}

/// Two rects split at any column tile the whole target with no gap or overlap.
#[test]
fn a_split_at_any_edge_tiles_the_whole_texture() {
    const EXTENT: u32 = 803;
    for percent in [1, 33, 50, 67, 75, 99] {
        let s = RenderScale::from_percent(percent);
        for split in 0..=EXTENT {
            let (ax, _, aw, _) = s.rect(0, 0, split, 1);
            let (bx, _, bw, _) = s.rect(split, 0, EXTENT - split, 1);
            assert_eq!(ax + aw, bx, "{percent}%: split at {split} seams");
            assert_eq!(bx + bw, s.dimension(EXTENT), "{percent}%: split at {split}");
        }
    }
}

/// A full-level rect lands on the texture Metal allocated for the level.
#[test]
fn a_full_level_rect_spans_the_level_metal_allocated() {
    for percent in 1..=100 {
        let s = RenderScale::from_percent(percent);
        for base in 1..=1024u32 {
            for level in 0..=4u32 {
                let logical = (base >> level).max(1);
                let extent =
                    TargetExtent::mip_level(s, (logical, logical), (s.dimension(base), 1), level);
                let texture = (s.dimension(base) >> level).max(1);
                assert_eq!(extent.texture().0, texture);
                assert_eq!(
                    extent.rect(0, 0, logical, 1).2,
                    texture,
                    "{percent}% of {base} at level {level}"
                );
                let signed = logical.cast_signed();
                assert_eq!(
                    extent.rect_edges_i32((0, 0, signed, 1)).2,
                    texture.cast_signed(),
                    "{percent}% of {base} at level {level}, signed"
                );
            }
        }
    }
}

/// Pinning the far edge keeps every edge in order and adjacent rects abutting.
#[test]
fn a_pinned_level_rect_still_tiles() {
    // 67% of a 1920 base is 1286 texels, so level 1 is 643 wide while the
    // scale of its reported 960 is 643 too, and level 2 is 321 against a
    // reported 480 that scales to 322: the far edge is pulled in by one.
    let s = RenderScale::from_percent(67);
    let extent = TargetExtent::mip_level(s, (480, 1), (s.dimension(1920), 1), 2);
    assert_eq!(extent.texture().0, 321);
    let mut previous = 0;
    for v in 0..=600 {
        let (x, _, w, _) = extent.rect(0, 0, v, 1);
        assert_eq!(x, 0);
        assert!(w >= previous, "edge {v} moved backwards");
        previous = w;
        let (bx, _, bw, _) = extent.rect(v, 0, 480u32.saturating_sub(v), 1);
        if v <= 480 {
            assert_eq!(bx, w, "split at {v} seams");
            assert_eq!(bx + bw, 321, "split at {v} ends on the texture");
        }
    }
}

/// A surface or level 0 converts exactly as the scale's own rect does.
#[test]
fn a_whole_target_extent_converts_like_the_scale() {
    for percent in [33, 50, 67, 75] {
        let s = RenderScale::from_percent(percent);
        let extent = TargetExtent::whole(s, (803, 603));
        for (x, y, w, h) in [(0, 0, 803, 603), (13, 7, 400, 300), (100, 100, 900, 900)] {
            assert_eq!(extent.rect(x, y, w, h), s.rect(x, y, w, h), "{percent}%");
        }
        let r = (-5, 3, 803, 700);
        assert_eq!(extent.rect_edges_i32(r), s.rect_edges_i32(r), "{percent}%");
    }
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

#[test]
fn factor_is_the_render_over_logical_ratio() {
    assert!((RenderScale::IDENTITY.factor() - 1.0).abs() < f32::EPSILON);
    assert!((RenderScale::from_percent(50).factor() - 0.5).abs() < f32::EPSILON);
    assert!((RenderScale::from_percent(75).factor() - 0.75).abs() < f32::EPSILON);
    // A length converted with the factor lands where `dimension` puts the
    // matching extent, give or take the rounding rule each side uses.
    let s = RenderScale::from_percent(67);
    let scaled = 480.0_f32 * s.factor();
    assert!((scaled - 321.6).abs() < 0.05);
}

#[test]
fn the_identity_has_no_lod_bias() {
    assert_eq!(RenderScale::IDENTITY.lod_bias().to_bits(), 0.0f32.to_bits());
}

#[test]
fn a_reduced_scale_biases_by_its_log2() {
    assert_eq!(
        RenderScale::from_percent(50).lod_bias().to_bits(),
        (-1.0f32).to_bits()
    );
    assert_eq!(
        RenderScale::from_percent(25).lod_bias().to_bits(),
        (-2.0f32).to_bits()
    );
    let bias = RenderScale::from_percent(75).lod_bias();
    assert!(
        (bias - 0.75f32.log2()).abs() < 1e-6,
        "75% biases by log2(0.75), got {bias}"
    );
}
