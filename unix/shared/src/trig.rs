//! Branchless single-precision sine/cosine, tuned to the game's accuracy.
//!
//! Lives in `mtld3d-shared` because the i686 d3d9 fixed-function state builder
//! needs libcall-free trig (spotlight cone cosines). `f32::sin`/`f32::cos` lower
//! to a scalar *C*-libm call whose result lands in the x87 `ST0` register —
//! opaque to the optimizer, so it blocks vectorization of the surrounding
//! kernel. A pure-Rust `libm` crate would instead inline under LTO, so
//! "it is a call" is *not* the reason to hand-roll these. The real reason is
//! cost vs. accuracy: `libm`'s `sinf`/`cosf` are
//! correctly-rounded — full-range Payne–Hanek reduction plus extra polynomial
//! terms — which is heavier than the game needs; even fully inlined it would
//! be slower than this tuned few-ULP `|x| < 8192` fast path, which
//! auto-vectorizes to packed SSE. (A standalone transcendental *leaf* with no
//! surrounding kernel to vectorize — e.g. `Math::Pow` — has no such cost
//! argument, so there the `libm` crate is used for its correctness and free
//! IEEE special cases.)
//!
//! These are clean-room polynomial approximations (the classic Cody–Waite
//! range reduction onto `[-pi/4, pi/4]` with minimax polynomials). The common
//! `|x| < 8192` path is branchless `f32` that inlines and auto-vectorizes;
//! accuracy is a few ULP versus a double-precision reference. Larger `|x|` (where
//! the `f32` index would overflow and the reduction would lose all precision,
//! returning huge/Inf/NaN) falls back to an out-of-line `f64` reduction so the
//! result stays bounded and correct — callers do occasionally pass such a large
//! angle, and a garbage result there must not propagate downstream.
#![allow(
    clippy::pedantic,
    clippy::nursery,
    // `suboptimal_flops` would suggest `mul_add`, but the i686/x86-64-v2 baseline
    // has no FMA, so `mul_add` lowers to a slow libm `fma` call — the polynomials
    // are written as explicit mul/add on purpose.
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    // The f64 Cody–Waite reduction constants are kept bit-exact (Cephes values).
    clippy::excessive_precision
)]

/// `4 / pi` — scales the argument so its integer part is the octant index.
const FOPI: f32 = 1.273_239_5;
/// `pi / 4` split into three parts for extended-precision (Cody–Waite) argument reduction.
///
/// Subtracting `j * (DP1 + DP2 + DP3)` cancels far more significant bits than
/// one `f32`-rounded `pi/4` could.
const DP1: f32 = 0.785_156_25;
const DP2: f32 = 2.418_756_5e-4;
const DP3: f32 = 3.774_895e-8;
/// Largest `|x|` the fast `f32` reduction stays exact for.
///
/// Beyond this the `as i32` octant index overflows and the `f32` subtraction
/// loses all precision (a multiple of `pi/4` near `1e9` has no fractional
/// bits left in `f32`), which would return a huge/Inf/NaN instead of a value
/// in `[-1, 1]`. Larger arguments reduce in `f64` (below).
const FAST_F32_LIMIT: f32 = 8192.0;
/// `f64` reduction constants for `|x| >= FAST_F32_LIMIT`.
///
/// `4/pi` and the three-part `pi/4` (Cody–Waite) are carried at double
/// precision so the octant index and reduced argument stay accurate out to the
/// largest angles the game can pass.
const FOPI_F64: f64 = 1.273_239_544_735_162_7;
const DP1_F64: f64 = 0.785_398_125_648_498_54;
const DP2_F64: f64 = 3.774_894_707_930_798_2e-8;
const DP3_F64: f64 = 2.695_151_429_079_059_6e-15;
/// Minimax coefficients for `sin` on the reduced interval (Horner, descending).
const SINCOF0: f32 = -1.951_529_6e-4;
const SINCOF1: f32 = 8.332_161e-3;
const SINCOF2: f32 = -1.666_665_5e-1;
/// Minimax coefficients for `cos` on the reduced interval (Horner, descending).
const COSCOF0: f32 = 2.443_315_7e-5;
const COSCOF1: f32 = -1.388_731_6e-3;
const COSCOF2: f32 = 4.166_664_6e-2;

/// Sign bit of an `f32`.
const SIGN: u32 = 0x8000_0000;

/// Branchless float select: returns `if_set` where `mask` is all-ones, else `if_clear`.
///
/// The mask is always one of `0` / `0xFFFF_FFFF`. LLVM lowers this to a blend,
/// so the two polynomials are both evaluated and merged without a branch.
fn select(mask: u32, if_set: f32, if_clear: f32) -> f32 {
    f32::from_bits((if_set.to_bits() & mask) | (if_clear.to_bits() & !mask))
}

/// Sine and cosine of `x` (radians), computed together.
///
/// The range reduction and both polynomials are shared, so a caller needing
/// both pays for one reduction.
pub fn sin_cos(x: f32) -> (f32, f32) {
    let sign_in = x.to_bits() & SIGN;
    let xa = x.abs();

    // Inf/NaN reduce to NaN (as libm does) and would otherwise saturate the octant
    // index; handle them up front so neither reduction path sees a non-finite value.
    if !xa.is_finite() {
        return (f32::NAN, f32::NAN);
    }

    // Octant index `j` (rounded up to even so the reduced argument stays in
    // `[-pi/4, pi/4]`) and the reduced argument `xx`. Periodicity is handled by
    // the bit tests below, so `j` is never masked to its low 3 bits. For small
    // `|x|` the fast `f32` Cody–Waite is exact and inlines; large `|x|` reduces in
    // `f64` via an out-of-line cold helper so the hot path stays lean.
    let (ju, xx) = if xa < FAST_F32_LIMIT {
        let mut j = (xa * FOPI) as i32;
        j = (j + 1) & !1;
        let y = j as f32;
        let mut xx = xa;
        xx -= y * DP1;
        xx -= y * DP2;
        xx -= y * DP3;
        (j as u32, xx)
    } else {
        reduce_large(xa)
    };

    // Octant bookkeeping (all mod-8 via single-bit tests):
    //   bit 2 (`& 4`) flips the sine sign; the cosine sign is the same test on
    //   `j - 2`. bit 1 (`& 2`) selects which polynomial approximates which.
    let swap_sign_sin = (ju & 4) << 29;
    let sign_cos = (!ju.wrapping_sub(2) & 4) << 29;
    let poly_mask = if ju & 2 == 0 { 0xFFFF_FFFF } else { 0 };

    let z = xx * xx;

    // cos polynomial: 1 - z/2 + z²·P(z).
    let mut yc = COSCOF0;
    yc = yc * z + COSCOF1;
    yc = yc * z + COSCOF2;
    yc = yc * z * z;
    yc = yc - 0.5 * z + 1.0;

    // sin polynomial: xx + xx·z·Q(z).
    let mut ys = SINCOF0;
    ys = ys * z + SINCOF1;
    ys = ys * z + SINCOF2;
    ys = ys * z * xx + xx;

    // In octants where `j & 2 == 0` the sine is the sin polynomial and the cosine
    // the cos polynomial; otherwise they swap.
    let sin_u = select(poly_mask, ys, yc);
    let cos_u = select(poly_mask, yc, ys);

    let s = f32::from_bits(sin_u.to_bits() ^ (swap_sign_sin ^ sign_in));
    let c = f32::from_bits(cos_u.to_bits() ^ sign_cos);
    (s, c)
}

/// Cold-path range reduction for `|x| >= FAST_F32_LIMIT`.
///
/// Computes the octant index and reduced argument in `f64`, where the `f32`
/// Cody–Waite would overflow the index and lose all precision (returning
/// huge/Inf/NaN). Kept out of line so the common small-angle path inlines
/// lean; `xa` is finite (the caller guards Inf/NaN) and non-negative.
#[cold]
#[inline(never)]
fn reduce_large(xa: f32) -> (u32, f32) {
    let xad = f64::from(xa);
    // `wrapping_add` guards the (game-unreachable) `|x| > ~7e18` case where the
    // index saturates `i64`.
    let mut j = crate::ftol::ftol(xad * FOPI_F64);
    j = j.wrapping_add(1) & !1;
    let y = j as f64;
    let mut xx = xad;
    xx -= y * DP1_F64;
    xx -= y * DP2_F64;
    xx -= y * DP3_F64;
    (j as u32, xx as f32)
}

/// Sine of `x` (radians).
///
/// Shares [`sin_cos`]; the unused cosine is cheap and folds away when the
/// caller only needs the sine.
pub fn sin(x: f32) -> f32 {
    sin_cos(x).0
}

/// Cosine of `x` (radians).
///
/// Shares [`sin_cos`]; the unused sine folds away when the caller only needs
/// the cosine.
pub fn cos(x: f32) -> f32 {
    sin_cos(x).1
}

/// `pi / 2` and `pi / 4`, the arctangent reduction anchors.
const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
const FRAC_PI_4: f32 = core::f32::consts::FRAC_PI_4;
/// `tan(3*pi/8)` and `tan(pi/8)` — the two arctangent range-reduction thresholds.
const TAN_3PI_8: f32 = 2.414_213_5;
const TAN_PI_8: f32 = 0.414_213_57;

/// Arctangent of `x` in radians, result in `[-pi/2, pi/2]`.
///
/// Minimax polynomial on `[-tan(pi/8), tan(pi/8)]` with two reciprocal/sum
/// reductions folding the rest of the line in (the classic Cephes `atanf`),
/// accurate to ~1 ULP.
pub fn atan(x: f32) -> f32 {
    let sign = x.is_sign_negative();
    let a = x.abs();

    // Fold |x| into [0, tan(pi/8)] and remember the constant the reduction adds.
    let (mut y, a) = if a > TAN_3PI_8 {
        (FRAC_PI_2, -1.0 / a)
    } else if a > TAN_PI_8 {
        (FRAC_PI_4, (a - 1.0) / (a + 1.0))
    } else {
        (0.0, a)
    };

    let z = a * a;
    y += ((((0.080_537_44 * z - 0.138_776_85) * z + 0.199_777_11) * z - 0.333_329_5) * z) * a + a;
    if sign { -y } else { y }
}

/// Full-plane arctangent of `y / x` in radians, result in `(-pi, pi]`.
///
/// The quadrant is resolved from the signs of both arguments (`x == 0` and
/// `y == 0` handled). Built on [`atan`].
pub fn atan2(y: f32, x: f32) -> f32 {
    if x > 0.0 {
        atan(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            atan(y / x) + PI
        } else {
            atan(y / x) - PI
        }
    } else if y > 0.0 {
        FRAC_PI_2
    } else if y < 0.0 {
        -FRAC_PI_2
    } else {
        0.0
    }
}

/// Arccosine of `x` (clamped domain `[-1, 1]`), result in `[0, pi]`.
///
/// Formulated as `atan2(sqrt((1-x)(1+x)), x)` — the standard identity;
/// `(1-x)(1+x)` keeps precision near `|x| = 1` better than `1 - x*x`. `sqrt` is
/// the hardware `sqrtss`, not a libm call.
pub fn acos(x: f32) -> f32 {
    atan2(((1.0 - x) * (1.0 + x)).max(0.0).sqrt(), x)
}

const PI: f32 = core::f32::consts::PI;

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::{acos, atan, atan2, sin_cos};

    /// Arc-function tolerance: the reduction's division adds error over the raw polynomial.
    ///
    /// So a slightly looser bound than the sin/cos `TOL`.
    const ATOL: f32 = 6e-6;

    /// Max absolute error against an `f64` reference over a dense sweep.
    ///
    /// The polynomial is `f32`, so a few ULP near `±1` is expected; `2e-6` is
    /// a tight bound that still passes (and would catch a coefficient or sign
    /// error).
    const TOL: f32 = 2e-6;

    fn ref_sin(x: f32) -> f32 {
        f64::from(x).sin() as f32
    }
    fn ref_cos(x: f32) -> f32 {
        f64::from(x).cos() as f32
    }

    #[test]
    fn exact_at_cardinal_angles() {
        for (x, s, c) in [
            (0.0, 0.0, 1.0),
            (PI / 2.0, 1.0, 0.0),
            (PI, 0.0, -1.0),
            (3.0 * PI / 2.0, -1.0, 0.0),
        ] {
            let (gs, gc) = sin_cos(x);
            assert!((gs - s).abs() < TOL, "sin({x}) = {gs}, want {s}");
            assert!((gc - c).abs() < TOL, "cos({x}) = {gc}, want {c}");
        }
    }

    #[test]
    fn matches_reference_over_dense_sweep() {
        // 0..N across several periods, both signs.
        let n = 20_000;
        let mut max_s = 0.0f32;
        let mut max_c = 0.0f32;
        for i in 0..n {
            let x = (i as f32 / n as f32) * 64.0 - 32.0; // [-32, 32)
            let (gs, gc) = sin_cos(x);
            max_s = max_s.max((gs - ref_sin(x)).abs());
            max_c = max_c.max((gc - ref_cos(x)).abs());
        }
        assert!(max_s < TOL, "max sin error {max_s} exceeds {TOL}");
        assert!(max_c < TOL, "max cos error {max_c} exceeds {TOL}");
    }

    #[test]
    fn atan_matches_reference() {
        let n = 20_000;
        let mut max = 0.0f32;
        for i in 0..n {
            let x = (i as f32 / n as f32) * 200.0 - 100.0; // [-100, 100)
            let got = atan(x);
            max = max.max((got - f64::from(x).atan() as f32).abs());
        }
        assert!(max < ATOL, "max atan error {max} exceeds {ATOL}");
    }

    #[test]
    fn atan2_resolves_all_quadrants() {
        // Cardinal directions land on the exact axis angles.
        assert!((atan2(0.0, 1.0) - 0.0).abs() < ATOL);
        assert!((atan2(1.0, 0.0) - PI / 2.0).abs() < ATOL);
        assert!((atan2(0.0, -1.0) - PI).abs() < ATOL);
        assert!((atan2(-1.0, 0.0) + PI / 2.0).abs() < ATOL);
        assert_eq!(atan2(0.0, 0.0), 0.0);

        // Dense sweep over the plane vs the f64 reference (skip the origin).
        let n = 200;
        let mut max = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                let y = (i as f32 / n as f32) * 8.0 - 4.0;
                let x = (j as f32 / n as f32) * 8.0 - 4.0;
                if x == 0.0 && y == 0.0 {
                    continue;
                }
                let got = atan2(y, x);
                let want = f64::from(y).atan2(f64::from(x)) as f32;
                max = max.max((got - want).abs());
            }
        }
        assert!(max < ATOL, "max atan2 error {max} exceeds {ATOL}");
    }

    #[test]
    fn acos_matches_reference_incl_endpoints() {
        assert!((acos(1.0) - 0.0).abs() < ATOL);
        assert!((acos(-1.0) - PI).abs() < ATOL);
        assert!((acos(0.0) - PI / 2.0).abs() < ATOL);

        let n = 20_000;
        let mut max_ac = 0.0f32;
        for i in 0..=n {
            let x = (i as f32 / n as f32) * 2.0 - 1.0; // [-1, 1]
            max_ac = max_ac.max((acos(x) - f64::from(x).acos() as f32).abs());
        }
        assert!(max_ac < ATOL, "max acos error {max_ac} exceeds {ATOL}");
    }

    #[test]
    fn large_angles_stay_bounded_and_track_reference() {
        // The f32-only reduction returned huge/Inf/NaN past ~1e6 (overflowed
        // octant index + cancellation). Large |x| now reduces in f64 and must
        // stay in [-1,1] and close to the f64 reference. (Beyond ~1e15 even f64
        // reduction degrades; callers never approach that.)
        for &x in &[
            8192.5f32, 1.0e5, 1.0e6, 1.0e7, 1.0e9, 1.0e12, -1.0e9, -3.0e6,
        ] {
            let (s, c) = sin_cos(x);
            assert!(
                s.is_finite() && c.is_finite(),
                "non-finite at x={x}: ({s},{c})"
            );
            assert!(
                s.abs() <= 1.0001 && c.abs() <= 1.0001,
                "out of range at x={x}: ({s},{c})"
            );
            let (rs, rc) = (f64::from(x).sin() as f32, f64::from(x).cos() as f32);
            assert!((s - rs).abs() < 1e-3, "sin off at x={x}: {s} vs {rs}");
            assert!((c - rc).abs() < 1e-3, "cos off at x={x}: {c} vs {rc}");
        }
    }

    #[test]
    fn non_finite_inputs_do_not_panic() {
        // Inf/NaN angles return NaN (as libm does) — must not panic or hang.
        for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let (s, c) = sin_cos(x);
            assert!(s.is_nan() && c.is_nan(), "x={x} -> ({s},{c})");
        }
    }

    #[test]
    fn odd_even_symmetry() {
        for i in 0..1000 {
            let x = i as f32 * 0.013;
            let (sp, cp) = sin_cos(x);
            let (sn, cn) = sin_cos(-x);
            assert!((sn + sp).abs() < TOL, "sin not odd at {x}");
            assert!((cn - cp).abs() < TOL, "cos not even at {x}");
        }
    }

    #[test]
    fn pythagorean_identity_holds() {
        for i in 0..2000 {
            let x = i as f32 * 0.031 - 31.0;
            let (s, c) = sin_cos(x);
            assert!((s * s + c * c - 1.0).abs() < 1e-5, "sin²+cos² off at {x}");
        }
    }
}
