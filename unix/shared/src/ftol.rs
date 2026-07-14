//! x87-free float→integer truncation.
//!
//! On 32-bit x86 a compiler-lowered `f64 as i64` emits an x87 `fld`/`fisttp`
//! (or the fnstcw/fistp control-word dance) plus a saturation clamp — under
//! Rosetta each x87 op expands ~20×, so every such cast in a hot path shows up
//! as a profiler row. This kernel truncates from the raw IEEE bits in pure
//! integer code instead — used by the i686 trig kernel and any other code
//! needing a float→int cast without the x87 tax.
#![allow(
    // Bit-manipulation on IEEE fields: the shifts/masks/casts are the point.
    clippy::pedantic,
    clippy::nursery
)]

/// CRT float-to-i64 truncation (`__ftol`): round toward zero.
///
/// The x87 indefinite value (`i64::MIN`) stands in for NaN, infinities, and
/// out-of-range inputs.
///
/// Pure integer bit manipulation — the original swaps the FPU control word
/// twice around an `fistp` for every float-to-int cast in the binary; a
/// float-to-i64 cast lowered by the compiler on 32-bit x86 would re-emit that
/// same control-word dance, so the truncation is built from the raw bits
/// instead. `-2^63` itself shares the indefinite bit pattern, so every
/// `|x| >= 2^63` case collapses to one return.
pub fn ftol(x: f64) -> i64 {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp < 1023 {
        // |x| < 1 (zero and subnormals included) truncates to 0.
        return 0;
    }
    let e = exp - 1075;
    if e >= 11 {
        // NaN/Inf (exp = 0x7ff) and every |x| >= 2^63: the fistp indefinite
        // value. -2^63 lands on the same pattern, so it needs no carve-out.
        return i64::MIN;
    }
    let m = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let mag = if e >= 0 { m << e } else { m >> -e };
    if bits >> 63 == 0 {
        mag as i64
    } else {
        -(mag as i64)
    }
}

#[cfg(test)]
mod tests {
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
}
