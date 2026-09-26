//! The robust statistics the A/B verdicts rest on: the median and the median absolute deviation.
//!
//! A benchmark round now and then lands on a machine hiccup, and a mean or
//! a standard deviation lets that one round move the verdict. The median
//! and the MAD ignore up to half the rounds being wild.

/// The scale that makes the MAD estimate the standard deviation of normally distributed data.
pub const MAD_SIGMA: f64 = 1.4826;

/// The median of `values`: the middle one, or the mean of the middle two; NaN for none.
#[must_use]
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    if count == 0 {
        return f64::NAN;
    }
    let mid = count / 2;
    if count % 2 == 1 {
        sorted[mid]
    } else {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    }
}

/// The median absolute deviation of `values` from their median; NaN for none.
#[must_use]
pub fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    median(&deviations)
}

#[cfg(test)]
mod tests;
