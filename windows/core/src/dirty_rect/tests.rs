//! Unit tests for the `DirtyRect` clamp, clip and full helpers.
//!
//! Four cases drive `clamp` against a 128x128 mip: the interior identity, trimming a rect that
//! overhangs the mip, rejecting one whose origin sits past the edge, and the saturating add
//! that keeps an absurd width from wrapping into a bogus in-bounds rect. A fifth checks that
//! `full` builds a rect at the origin spanning the mip extent it is given.
//!
//! Four more drive `clip_to_level`, the copy-region form: a rect overhanging the right edge, one
//! overhanging the bottom edge, a block-compressed rect whose rounding out to the 4x4 block grid
//! would reach past a level that stops mid-block, and one that misses the level entirely.
//!
//! Four drive `clip_copy_region`: a region that fits both levels comes back with only its
//! origins; a transposed pair (a 2x4 source level into a 4x2 destination level, which
//! `UpdateTexture` accepts because the two agree on their larger dimension) comes back as the
//! 2x2 they share; a source level twice the destination's in both dimensions comes back halved;
//! and a region whose destination origin sits mid-level comes back as the part that still fits,
//! which is the shape the format-converting copy takes.

use super::{DirtyRect, clip_copy_region};

#[test]
fn clamp_inside_is_identity() {
    let r = DirtyRect {
        x: 10,
        y: 10,
        w: 50,
        h: 50,
    };
    assert_eq!(r.clamp(128, 128), Some(r));
}

#[test]
fn clamp_overflow_trims_to_mip() {
    let r = DirtyRect {
        x: 100,
        y: 100,
        w: 100,
        h: 100,
    };
    assert_eq!(
        r.clamp(128, 128),
        Some(DirtyRect {
            x: 100,
            y: 100,
            w: 28,
            h: 28,
        })
    );
}

#[test]
fn clamp_origin_past_mip_is_none() {
    let r = DirtyRect {
        x: 200,
        y: 0,
        w: 10,
        h: 10,
    };
    assert_eq!(r.clamp(128, 128), None);
}

#[test]
fn clamp_saturating_add_overflow_is_safe() {
    let r = DirtyRect {
        x: u32::MAX - 1,
        y: 0,
        w: u32::MAX,
        h: 1,
    };
    // right saturates to u32::MAX, clamped to mip width.
    assert_eq!(
        r.clamp(128, 128),
        None,
        "x past mip width clamps to zero area"
    );
}

#[test]
fn full_spans_mip() {
    let r = DirtyRect::full(64, 32);
    assert_eq!(
        r,
        DirtyRect {
            x: 0,
            y: 0,
            w: 64,
            h: 32,
        }
    );
}

#[test]
fn clip_to_level_trims_a_rect_overhanging_the_right_edge() {
    let r = DirtyRect {
        x: 0,
        y: 0,
        w: 4,
        h: 2,
    };
    assert_eq!(
        r.clip_to_level(2, 4, 1, 1),
        Some(DirtyRect {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        })
    );
}

#[test]
fn clip_to_level_trims_a_rect_overhanging_the_bottom_edge() {
    let r = DirtyRect {
        x: 0,
        y: 0,
        w: 2,
        h: 4,
    };
    assert_eq!(
        r.clip_to_level(4, 2, 1, 1),
        Some(DirtyRect {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        })
    );
}

#[test]
fn clip_to_level_keeps_block_rounding_inside_the_level() {
    // A 6x6 block-compressed level: the 4x4 block grid runs to 8, the level
    // stops at 6, so the rounded-out region has to stop there too.
    let r = DirtyRect {
        x: 5,
        y: 5,
        w: 1,
        h: 1,
    };
    assert_eq!(
        r.clip_to_level(6, 6, 4, 4),
        Some(DirtyRect {
            x: 4,
            y: 4,
            w: 2,
            h: 2,
        })
    );
}

#[test]
fn clip_to_level_rejects_a_rect_outside_the_level() {
    let r = DirtyRect {
        x: 8,
        y: 0,
        w: 4,
        h: 4,
    };
    assert_eq!(r.clip_to_level(8, 8, 1, 1), None);
}

#[test]
fn clip_copy_region_keeps_a_region_both_levels_hold() {
    let region = DirtyRect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    assert_eq!(
        clip_copy_region(region, (0, 0), (8, 8), (8, 8), (1, 1)),
        Some((
            region,
            DirtyRect {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }
        ))
    );
}

#[test]
fn clip_copy_region_trims_a_transposed_pair_to_what_they_share() {
    let region = DirtyRect {
        x: 0,
        y: 0,
        w: 2,
        h: 4,
    };
    let shared = DirtyRect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };
    assert_eq!(
        clip_copy_region(region, (0, 0), (2, 4), (4, 2), (1, 1)),
        Some((shared, shared))
    );
}

#[test]
fn clip_copy_region_trims_a_source_larger_than_the_destination_level() {
    let region = DirtyRect {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };
    let shared = DirtyRect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };
    assert_eq!(
        clip_copy_region(region, (0, 0), (4, 4), (2, 2), (1, 1)),
        Some((shared, shared))
    );
}

#[test]
fn clip_copy_region_trims_a_region_landing_mid_destination_level() {
    let region = DirtyRect {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };
    assert_eq!(
        clip_copy_region(region, (2, 2), (4, 4), (4, 4), (1, 1)),
        Some((
            DirtyRect {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
            },
            DirtyRect {
                x: 2,
                y: 2,
                w: 2,
                h: 2,
            }
        ))
    );
}
