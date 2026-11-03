//! Numeric casts where the cast lints fire but the bounds are established by the caller.
//!
//! Grouping them under one mod-level allow collapses what would otherwise be
//! four per-site allows into one. Each fn is the raw cast; callers document
//! the bound that justifies it.

/// Saturating `f64 → u32`.
///
/// NaN/negative → 0, ≥ `u32::MAX` → `u32::MAX`; all other inputs land in
/// `(0.0, u32::MAX)` where the cast is exact.
pub fn f64_to_u32_saturating(v: f64) -> u32 {
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    if v >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    v as u32
}

/// Saturating `f64 → u64`.
///
/// NaN/negative → 0, ≥ `u64::MAX` → `u64::MAX`; all other inputs land in
/// `(0.0, u64::MAX)` where the cast is exact to f64 precision.
pub fn f64_to_u64_saturating(v: f64) -> u64 {
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    if v >= u64::MAX as f64 {
        return u64::MAX;
    }
    v as u64
}

/// `f64 → f32` narrowing.
///
/// Caller establishes the bound where mantissa loss is acceptable.
pub const fn f64_to_f32(v: f64) -> f32 {
    v as f32
}

/// `i32 → f32` narrowing.
///
/// Caller establishes `|v|` is well inside the f32 mantissa (< 2^24) so
/// the cast is exact.
pub const fn i32_to_f32(v: i32) -> f32 {
    v as f32
}
