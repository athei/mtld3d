//! Backbuffer pixel helpers shared by every readback assertion.
//!
//! [`Harness::read_pixel`](crate::Harness::read_pixel) returns the BGRA8
//! backbuffer word read as a little-endian `u32`, i.e. `0xAARRGGBB`.
//! Decomposing through `to_le_bytes` yields the byte channels directly
//! (`[B, G, R, A]`) with no lossy casts.

/// A backbuffer pixel split into 8-bit channels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    /// Split a `0xAARRGGBB` backbuffer word into channels.
    #[must_use]
    pub const fn from_pixel(pixel: u32) -> Self {
        let [b, g, r, a] = pixel.to_le_bytes();
        Self { r, g, b, a }
    }

    /// Re-pack into the `0xAARRGGBB` word the readback returns.
    #[must_use]
    pub const fn to_pixel(self) -> u32 {
        u32::from_le_bytes([self.b, self.g, self.r, self.a])
    }

    /// True when every channel is within `tol` of `other`.
    ///
    /// For filtered or blended results where exact equality is unreasonable.
    #[must_use]
    pub const fn approx_eq(self, other: Self, tol: u8) -> bool {
        self.r.abs_diff(other.r) <= tol
            && self.g.abs_diff(other.g) <= tol
            && self.b.abs_diff(other.b) <= tol
            && self.a.abs_diff(other.a) <= tol
    }
}

/// Assert a backbuffer pixel exactly equals an expected `0xAARRGGBB` word.
///
/// # Panics
/// Panics with a channel-decoded diagnostic when the words differ.
#[track_caller]
pub fn assert_pixel_eq(actual: u32, expected: u32, context: &str) {
    assert!(
        actual == expected,
        "{context}: expected 0x{expected:08X} {:?}, got 0x{actual:08X} {:?}",
        Rgba8::from_pixel(expected),
        Rgba8::from_pixel(actual),
    );
}

/// Assert a backbuffer pixel is within `tol` per channel of an expected word.
///
/// # Panics
/// Panics with a channel-decoded diagnostic when any channel is out of range.
#[track_caller]
pub fn assert_pixel_approx(actual: u32, expected: u32, tol: u8, context: &str) {
    let a = Rgba8::from_pixel(actual);
    let e = Rgba8::from_pixel(expected);
    assert!(
        a.approx_eq(e, tol),
        "{context}: expected ~0x{expected:08X} {e:?} (tol {tol}), got 0x{actual:08X} {a:?}",
    );
}
