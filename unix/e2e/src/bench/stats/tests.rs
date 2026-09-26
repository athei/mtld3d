//! Unit tests for the median and the MAD.

use super::*;

fn bits(value: f64) -> u64 {
    value.to_bits()
}

#[test]
fn the_median_of_an_odd_count_is_the_middle_value() {
    assert_eq!(bits(median(&[3.0, 1.0, 2.0])), bits(2.0));
    assert_eq!(bits(median(&[7.0])), bits(7.0));
}

#[test]
fn the_median_of_an_even_count_is_the_mean_of_the_middle_two() {
    assert_eq!(bits(median(&[4.0, 1.0, 3.0, 2.0])), bits(2.5));
    assert_eq!(bits(median(&[1.0, 2.0])), bits(1.5));
}

#[test]
fn the_median_of_nothing_is_nan() {
    assert!(median(&[]).is_nan());
    assert!(mad(&[]).is_nan());
}

#[test]
fn the_mad_of_an_odd_count() {
    // Median 3; deviations 2, 1, 0, 1, 97; their median 1.
    assert_eq!(bits(mad(&[1.0, 2.0, 3.0, 4.0, 100.0])), bits(1.0));
}

#[test]
fn the_mad_of_an_even_count() {
    // Median 2.5; deviations 1.5, 0.5, 0.5, 1.5; their median 1.
    assert_eq!(bits(mad(&[1.0, 2.0, 3.0, 4.0])), bits(1.0));
}

#[test]
fn equal_values_have_no_spread() {
    assert_eq!(bits(mad(&[5.0, 5.0, 5.0])), bits(0.0));
}
