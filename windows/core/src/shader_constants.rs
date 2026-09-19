//! Bitwise comparison of floating-point shader constant rows.

/// Whether two constant windows differ, including NaN payloads and signed zero.
///
/// Compare whole four-word rows so each row can use integer vector equality.
/// Array mapping preserves every bit without imposing SIMD alignment on callers.
#[must_use]
#[inline]
pub fn rows_differ(cur: &[[f32; 4]], new: &[[f32; 4]]) -> bool {
    cur.len() != new.len()
        || cur
            .iter()
            .zip(new)
            .any(|(a, b)| a.map(f32::to_bits) != b.map(f32::to_bits))
}

#[cfg(test)]
mod tests;
