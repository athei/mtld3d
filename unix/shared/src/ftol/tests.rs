//! Unit tests for the x87-free `ftol` truncation kernel.
//!
//! Every in-range case is checked against Rust's own `as i64` cast, so the bit
//! manipulation stays equivalent to the reference lowering: signs, the boundaries at
//! `2^52` and `2^63`, subnormals collapsing to zero, and a sweep across the whole
//! exponent range. NaN, infinities and overflowing magnitudes are pinned separately,
//! since those must return the x87 indefinite value rather than saturate.

use super::ftol;

/// In-range values match the hardware truncating cast exactly.
#[test]
fn matches_truncating_cast_in_range() {
    for &x in &[
        0.0_f64,
        -0.0,
        0.5,
        -0.5,
        0.999_999_9,
        1.0,
        1.5,
        -1.5,
        2.75,
        -2.75,
        12345.678,
        -12345.678,
        4_294_967_295.9,
        (1_u64 << 52) as f64 + 0.5,
        -((1_u64 << 52) as f64) - 0.5,
        9.223_372_036_854_774e18,  // largest f64 below 2^63
        -9.223_372_036_854_776e18, // exactly -2^63
        f64::MIN_POSITIVE,         // subnormal-adjacent -> 0
        5e-324,                    // smallest subnormal -> 0
    ] {
        assert_eq!(ftol(x), x as i64, "x={x:e}");
    }
}

/// NaN, infinities, and out-of-range magnitudes store the indefinite value.
#[test]
fn specials_store_indefinite() {
    for &x in &[
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9.223_372_036_854_776e18, // exactly 2^63 (not representable)
        1e19,
        -1e19,
        f64::MAX,
        f64::MIN,
    ] {
        assert_eq!(ftol(x), i64::MIN, "x={x:e}");
    }
}

/// Sweep the exponent range against the reference cast.
#[test]
fn exponent_sweep() {
    let mut x = 1.0_f64;
    while x < 9.2e18 {
        for &v in &[x, -x, x * 1.5, -x * 1.5, x + 0.25] {
            assert_eq!(ftol(v), v as i64, "v={v:e}");
        }
        x *= 2.0;
    }
}
