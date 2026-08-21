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
mod tests;
