//! Accuracy tests for the hand-rolled single-precision trig approximations.
//!
//! A dense sweep compares `sin_cos` against an `f64` reference under a fixed
//! error bound, so a mistyped minimax coefficient or a flipped sign fails
//! loudly instead of skewing a spotlight cone. The rest pins the octant
//! bookkeeping (cardinal angles, odd/even symmetry, the Pythagorean identity),
//! the `f64` fallback that keeps huge angles inside `[-1, 1]`, and non-finite
//! inputs returning NaN rather than panicking or hanging.

use core::f32::consts::PI;

use super::sin_cos;

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
