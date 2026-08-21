//! Unit tests for the `DirtyRect` clamp and full helpers.
//!
//! Four cases drive `clamp` against a 128x128 mip: the interior identity, trimming a rect that
//! overhangs the mip, rejecting one whose origin sits past the edge, and the saturating add
//! that keeps an absurd width from wrapping into a bogus in-bounds rect. A fifth checks that
//! `full` builds a rect at the origin spanning the mip extent it is given.

use super::DirtyRect;

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
