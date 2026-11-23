//! Unit tests for logical-to-render resolution conversion.
//!
//! Pins the properties the call sites depend on: the identity is exact,
//! `size` never collapses a small target to zero, and an out-of-range
//! percentage clamps (rendering above the presented size is not offered). The
//! rect cases cover edge-based scaling, so abutting rects still share an edge,
//! a rect spanning the logical extent spans the texture that extent creates,
//! and `rect` and `rect_edges_i32` land on the same pixels, at a percentage
//! and at a scale that differs between the axes.
//! `factor` carries the same ratio as a float, for lengths the shaders scale.
//!
//! `panel_fit` is the `render.panelFit` trigger: the panel's exact pixels from
//! a back buffer above the panel of a screen scaled down to it, the identity
//! otherwise, and its composition with a percentage rounds once.

use mtld3d_shared::render_size::RenderScaleWire;

use super::*;

/// "More Space" on a 16-inch panel: the backing store in retina mode.
const MORE_SPACE: (u32, u32) = (4112, 2658);
const PANEL_16: (u32, u32) = (3456, 2234);
/// A 1470x956 "looks like" mode on a 13-inch panel, in retina mode.
const AIR_BACKING: (u32, u32) = (2940, 1912);
const AIR_PANEL: (u32, u32) = (2560, 1664);

/// One logical dimension at `s`, in its width axis.
fn dim(s: RenderScale, n: u32) -> u32 {
    s.size((n, n)).0
}

/// The panel fit of a full-screen back buffer on the 16-inch "More Space" screen.
fn more_space() -> RenderScale {
    panel_fit(true, MORE_SPACE, Some(PANEL_16), Some(MORE_SPACE))
}

#[test]
fn identity_changes_nothing() {
    let s = RenderScale::IDENTITY;
    assert!(s.is_identity());
    for d in [0, 1, 7, 640, 1920, 16384] {
        assert_eq!(s.size((d, d + 1)), (d, d + 1));
    }
    assert_eq!(s.rect(13, 27, 640, 480), (13, 27, 640, 480));
    assert_eq!(s.factor().to_bits(), 1.0_f32.to_bits());
}

#[test]
fn size_halves() {
    let s = RenderScale::from_percent(50);
    assert_eq!(s.size((1920, 1080)), (960, 540));
}

#[test]
fn size_never_collapses_to_zero() {
    let s = RenderScale::from_percent(25);
    assert_eq!(s.size((1, 2)), (1, 1));
    // Zero in, zero out: the caller rejects those separately.
    assert_eq!(s.size((0, 0)), (0, 0));
}

#[test]
fn from_percent_clamps_out_of_range() {
    assert_eq!(RenderScale::from_percent(0), RenderScale::from_percent(1));
    assert_eq!(RenderScale::from_percent(10_000), RenderScale::IDENTITY);
    assert_eq!(RenderScale::from_percent(100), RenderScale::IDENTITY);
}

/// A percentage keeps the sizes and rects a single percentage gave, bit for bit.
#[test]
fn every_percentage_keeps_the_percentage_rule() {
    let rule = |n: u32, percent: u32| {
        if n == 0 || percent == 100 {
            return n;
        }
        u32::try_from((u64::from(n) * u64::from(percent) + 50) / 100)
            .expect("at most n")
            .max(1)
    };
    for percent in 1..=100 {
        let s = RenderScale::from_percent(percent);
        for n in 0..=4096 {
            let m = 4096 - n;
            assert_eq!(
                s.size((n, m)),
                (rule(n, percent), rule(m, percent)),
                "{percent}%"
            );
            let (x, y) = (n / 3, m / 5);
            let expected = (
                rule(x, percent),
                rule(y, percent),
                rule(x + n, percent) - rule(x, percent),
                rule(y + m, percent) - rule(y, percent),
            );
            assert_eq!(s.rect(x, y, n, m), expected, "{percent}% rect");
        }
    }
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
/// A target's Metal texture is `size` of its logical extent and a
/// full-target viewport or `Clear` rect goes through `rect`, so the two must
/// agree at the far edge or the last texel column or row is never drawn.
#[test]
fn a_full_target_rect_spans_the_whole_texture() {
    let scales = (1..=100).map(RenderScale::from_percent).chain([
        more_space(),
        panel_fit(true, AIR_BACKING, Some(AIR_PANEL), Some(AIR_BACKING)),
    ]);
    for s in scales {
        for n in 0..=4096 {
            let m = 4096 - n;
            let (w, h) = s.size((n, m));
            assert_eq!(s.rect(0, 0, n, m), (0, 0, w, h), "{s:?} of {n}x{m}");
            assert_eq!(
                s.rect_edges_i32((0, 0, n.cast_signed(), m.cast_signed())),
                (0, 0, w.cast_signed(), h.cast_signed()),
                "{s:?} of {n}x{m}, signed"
            );
        }
    }
}

/// Two rects split at any column or row tile the whole target with no gap or overlap.
#[test]
fn a_split_at_any_edge_tiles_the_whole_texture() {
    const EXTENT: u32 = 803;
    let scales = [1, 33, 50, 67, 75, 99]
        .map(RenderScale::from_percent)
        .into_iter()
        .chain([
            more_space(),
            RenderScale::from_percent(75).compose(more_space()),
        ]);
    for s in scales {
        let (w, h) = s.size((EXTENT, EXTENT));
        for split in 0..=EXTENT {
            let (ax, _, aw, _) = s.rect(0, 0, split, 1);
            let (bx, _, bw, _) = s.rect(split, 0, EXTENT - split, 1);
            assert_eq!(ax + aw, bx, "{s:?}: column split at {split} seams");
            assert_eq!(bx + bw, w, "{s:?}: column split at {split}");
            let (_, ay, _, ah) = s.rect(0, 0, 1, split);
            let (_, by, _, bh) = s.rect(0, split, 1, EXTENT - split);
            assert_eq!(ay + ah, by, "{s:?}: row split at {split} seams");
            assert_eq!(by + bh, h, "{s:?}: row split at {split}");
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
                    TargetExtent::mip_level(s, (logical, logical), (dim(s, base), 1), level);
                let texture = (dim(s, base) >> level).max(1);
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

/// Each axis of a level pins against its own extent when the axes scale differently.
#[test]
fn a_non_uniform_level_pins_each_axis_on_its_own() {
    let s = RenderScale::from_percent(67).compose(more_space());
    for (width, height) in [(1920, 1080), (4112, 2658), (803, 601), (17, 3)] {
        let base = s.size((width, height));
        for level in 0..=4u32 {
            let logical = ((width >> level).max(1), (height >> level).max(1));
            let extent = TargetExtent::mip_level(s, logical, base, level);
            let texture = extent.texture();
            assert_eq!(
                texture,
                ((base.0 >> level).max(1), (base.1 >> level).max(1))
            );
            let full = extent.rect(0, 0, logical.0, logical.1);
            assert_eq!(
                (full.2, full.3),
                texture,
                "{width}x{height} at level {level}"
            );
            for split in 0..=logical.0.max(logical.1) {
                let (sx, sy) = (split.min(logical.0), split.min(logical.1));
                let (_, _, aw, ah) = extent.rect(0, 0, sx, sy);
                let (bx, by, bw, bh) = extent.rect(sx, sy, logical.0 - sx, logical.1 - sy);
                assert_eq!((aw, ah), (bx, by), "split at {split} seams");
                assert_eq!(
                    (bx + bw, by + bh),
                    texture,
                    "split at {split} ends on the texture"
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
    let extent = TargetExtent::mip_level(s, (480, 1), (dim(s, 1920), 1), 2);
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
    let scales = [33, 50, 67, 75]
        .map(RenderScale::from_percent)
        .into_iter()
        .chain([more_space()]);
    for s in scales {
        let extent = TargetExtent::whole(s, (803, 603));
        for (x, y, w, h) in [(0, 0, 803, 603), (13, 7, 400, 300), (100, 100, 900, 900)] {
            assert_eq!(extent.rect(x, y, w, h), s.rect(x, y, w, h), "{s:?}");
        }
        let r = (-5, 3, 803, 700);
        assert_eq!(extent.rect_edges_i32(r), s.rect_edges_i32(r), "{s:?}");
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
    for scale in [RenderScale::from_percent(75), more_space()] {
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
    assert_eq!(dim(s, 960), 960);
}

#[test]
fn factor_is_the_render_over_logical_ratio() {
    assert_eq!(
        RenderScale::from_percent(50).factor().to_bits(),
        0.5_f32.to_bits()
    );
    // A percentage gives exactly the quotient a percentage gave.
    for percent in 1..=100_u8 {
        assert_eq!(
            RenderScale::from_percent(u32::from(percent))
                .factor()
                .to_bits(),
            (f32::from(percent) / 100.0).to_bits(),
            "{percent}%"
        );
    }
    // A length converted with the factor lands where `size` puts the
    // matching extent, give or take the rounding rule each side uses.
    let s = RenderScale::from_percent(67);
    let scaled = 480.0_f32 * s.factor();
    assert!((scaled - 321.6).abs() < 0.05);
}

/// Axes that differ give their geometric mean, and the bias follows it.
#[test]
fn a_non_uniform_factor_is_the_geometric_mean() {
    let s = panel_fit(true, AIR_BACKING, Some(AIR_PANEL), Some(AIR_BACKING));
    let width = 2560.0_f64 / 2940.0;
    let height = 1664.0_f64 / 1912.0;
    let mean = (width * height).sqrt();
    let factor = f64::from(s.factor());
    assert!((factor - mean).abs() < 1e-6, "factor {factor}, mean {mean}");
    assert!(factor > width.min(height) && factor < width.max(height));
    let bias = f64::from(s.lod_bias());
    assert!((bias - mean.log2()).abs() < 1e-6, "bias {bias}");
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

#[test]
fn the_panel_fit_lands_on_the_panel_exactly() {
    assert_eq!(more_space().size(MORE_SPACE), PANEL_16);
    let air = panel_fit(true, AIR_BACKING, Some(AIR_PANEL), Some(AIR_BACKING));
    assert_eq!(air.size(AIR_BACKING), AIR_PANEL);
}

/// The fit takes the window's share of the panel, which is the drawable cap of the same window.
#[test]
fn the_panel_fit_matches_the_drawable_cap() {
    use mtld3d_shared::render_size::panel_drawable_size;

    for back_buffer in [(4112, 2560), (4112, 2500), (3500, 2658), (3457, 1000)] {
        let s = panel_fit(true, back_buffer, Some(PANEL_16), Some(MORE_SPACE));
        let render = s.size(back_buffer);
        let drawable =
            panel_drawable_size(back_buffer, Some(MORE_SPACE), Some(PANEL_16), Some(render));
        assert_eq!(render, drawable, "{back_buffer:?}");
    }
}

#[test]
fn the_panel_fit_triggers_above_the_panel_of_a_scaled_screen() {
    // A maximized window and a fullscreen back buffer at the desktop size.
    assert!(!panel_fit(true, (4112, 2560), Some(PANEL_16), Some(MORE_SPACE)).is_identity());
    assert!(!more_space().is_identity());
    // One axis above the panel is enough.
    assert!(!panel_fit(true, (3000, 2300), Some(PANEL_16), Some(MORE_SPACE)).is_identity());
}

#[test]
fn the_panel_fit_is_the_identity_otherwise() {
    let fit = |enabled, back_buffer, panel, backing| {
        let s = panel_fit(enabled, back_buffer, panel, backing);
        assert!(s.is_identity());
        assert_eq!(s, RenderScale::IDENTITY);
        assert_eq!(s.factor().to_bits(), 1.0_f32.to_bits());
        assert_eq!(s.size(back_buffer), back_buffer);
    };
    // A back buffer at or below the panel.
    fit(true, PANEL_16, Some(PANEL_16), Some(MORE_SPACE));
    fit(true, (2624, 1696), Some(PANEL_16), Some(MORE_SPACE));
    // A screen that is not scaled down: its backing store is the panel.
    fit(true, MORE_SPACE, Some(PANEL_16), Some(PANEL_16));
    // Retina off: the backing in the layer's pixels is below the panel.
    fit(true, MORE_SPACE, Some(PANEL_16), Some((2056, 1329)));
    // An unknown or degenerate size.
    fit(true, MORE_SPACE, None, Some(MORE_SPACE));
    fit(true, MORE_SPACE, Some(PANEL_16), None);
    fit(true, MORE_SPACE, Some((0, 2234)), Some(MORE_SPACE));
    fit(true, MORE_SPACE, Some(PANEL_16), Some((70_000, 2658)));
    // `render.panelFit = false`.
    fit(false, MORE_SPACE, Some(PANEL_16), Some(MORE_SPACE));
}

#[test]
fn an_axis_not_scaled_down_keeps_its_size() {
    let s = panel_fit(true, (4112, 2234), Some(PANEL_16), Some((4112, 2234)));
    assert_eq!(s.size((4112, 2234)), (3456, 2234));
}

/// The user factor and the panel fit multiply into one ratio, rounded once.
#[test]
fn a_composed_scale_rounds_once() {
    let s = RenderScale::from_percent(75).compose(more_space());
    let once = |n: u32, panel: u32, backing: u32| {
        u32::try_from(scale_rounded(n, 75 * panel, 100 * backing)).expect("at most n")
    };
    let twice = |n: u32, panel: u32, backing: u32| {
        let fitted = u32::try_from(scale_rounded(n, panel, backing)).expect("at most n");
        u32::try_from(scale_rounded(fitted, 75, 100)).expect("at most n")
    };
    let mut differs = false;
    for n in 1..=4112 {
        let (w, h) = s.size((n, n));
        assert_eq!(w, once(n, 3456, 4112), "width {n}");
        assert_eq!(h, once(n, 2234, 2658), "height {n}");
        differs |= w != twice(n, 3456, 4112) || h != twice(n, 2234, 2658);
    }
    assert!(differs, "rounding twice lands elsewhere for some size");
    assert_eq!(s.size(MORE_SPACE), (2592, 1676));
    assert_eq!(more_space().compose(RenderScale::from_percent(75)), s);
    assert_eq!(RenderScale::IDENTITY.compose(more_space()), more_space());
}

/// A device's scale follows the back buffer's size and the screen it was sized against.
#[test]
fn a_back_buffer_scale_follows_the_size_and_the_screen() {
    let mut scale = BackBufferScale::new(RenderScale::from_percent(75), true);
    assert_eq!(
        scale.for_back_buffer(MORE_SPACE),
        RenderScale::from_percent(75),
        "no screen known yet: the user's scale alone"
    );
    assert!(scale.set_screen(Some(PANEL_16), Some(MORE_SPACE)));
    assert!(
        !scale.set_screen(Some(PANEL_16), Some(MORE_SPACE)),
        "the same screen again"
    );
    assert_eq!(
        scale.for_back_buffer(MORE_SPACE).size(MORE_SPACE),
        (2592, 1676)
    );
    assert_eq!(
        scale.for_back_buffer((1920, 1080)),
        RenderScale::from_percent(75),
        "a back buffer within the panel"
    );
    assert!(scale.set_screen(Some(AIR_PANEL), Some(AIR_BACKING)));
    assert_eq!(
        scale.for_back_buffer(AIR_BACKING).size(AIR_BACKING),
        (1920, 1248)
    );
    let mut off = BackBufferScale::new(RenderScale::IDENTITY, false);
    off.set_screen(Some(PANEL_16), Some(MORE_SPACE));
    assert!(
        off.for_back_buffer(MORE_SPACE).is_identity(),
        "render.panelFit = false"
    );
}

#[test]
fn a_scale_survives_the_wire() {
    for s in [
        RenderScale::IDENTITY,
        RenderScale::from_percent(75),
        more_space(),
        RenderScale::from_percent(33).compose(more_space()),
    ] {
        assert!(RenderScale::is_valid_wire(&s.to_wire()));
        assert_eq!(RenderScale::from_wire(&s.to_wire()), s);
    }
    // An unreduced ratio reads as the same scale.
    let wire = RenderScaleWire {
        width_num: 75,
        width_den: 100,
        height_num: 3,
        height_den: 4,
    };
    assert_eq!(RenderScale::from_wire(&wire), RenderScale::from_percent(75));
}

#[test]
fn an_invalid_wire_axis_reads_as_the_identity() {
    for (num, den) in [(0, 100), (101, 100), (1, 0), (0, 0)] {
        let wire = RenderScaleWire {
            width_num: num,
            width_den: den,
            height_num: 3,
            height_den: 4,
        };
        assert!(!RenderScale::is_valid_wire(&wire), "{num}/{den}");
        assert_eq!(RenderScale::from_wire(&wire).size((400, 400)), (400, 300));
    }
}

#[test]
fn a_scaled_chain_holds_only_the_levels_its_base_can_have() {
    let three_quarters = RenderScale::from_percent(75);
    let half = RenderScale::from_percent(50);
    let two_thirds = RenderScale::from_percent(67);
    // 640x480 has ten levels, what `levels = 0` asks for; 480x360 nine, 320x240 nine.
    assert_eq!(rasterized_level_count(three_quarters, (640, 480), 10), 9);
    assert_eq!(rasterized_level_count(half, (640, 480), 10), 9);
    // 1920x1080 has eleven; 1286x724 at 0.67 has eleven too.
    assert_eq!(rasterized_level_count(two_thirds, (1920, 1080), 11), 11);
    // A single level, and a short explicit chain, are never lengthened.
    assert_eq!(rasterized_level_count(three_quarters, (640, 480), 1), 1);
    assert_eq!(rasterized_level_count(three_quarters, (640, 480), 4), 4);
    // The identity keeps whatever the texture reports.
    assert_eq!(
        rasterized_level_count(RenderScale::IDENTITY, (640, 480), 10),
        10
    );
}

#[test]
fn a_reported_level_past_the_chain_lands_on_its_last_level() {
    for level in 0..9 {
        assert_eq!(rasterized_level(level, 9), level, "a level the chain holds");
    }
    assert_eq!(
        rasterized_level(9, 9),
        8,
        "the reported tail shares the last level"
    );
    assert_eq!(rasterized_level(12, 9), 8);
    assert_eq!(rasterized_level(0, 1), 0, "a single-level texture");
    assert_eq!(rasterized_level(3, 1), 0);
}
