//! Cursor bitmap validation and acknowledged upload decisions.
//!
//! Shared by both cursor modes before the COM wrapper reads a locked surface.

use mtld3d_types::D3DFMT_A8R8G8B8;

/// The bitmap dimensions and hotspot before cursor scaling.
pub struct BitmapLayout {
    pub width: u32,
    pub height: u32,
    pub x_hotspot: u32,
    pub y_hotspot: u32,
}

impl BitmapLayout {
    /// Check the D3D9 format, extent, and the arithmetic used by either cursor mode.
    #[must_use]
    pub fn valid(&self, format: u32, display: (u32, u32), scale: u32) -> bool {
        format == D3DFMT_A8R8G8B8
            && self.width.is_power_of_two()
            && self.height.is_power_of_two()
            && self.width <= display.0
            && self.height <= display.1
            && self.scaled(scale).is_some()
    }

    /// Scaled dimensions and hotspot, bounded for GDI, Rust slices, and the wire length.
    #[must_use]
    pub fn scaled(&self, scale: u32) -> Option<Self> {
        if !(1..=8).contains(&scale) {
            return None;
        }
        let result = Self {
            width: self.width.checked_mul(scale)?,
            height: self.height.checked_mul(scale)?,
            x_hotspot: self.x_hotspot.checked_mul(scale)?,
            y_hotspot: self.y_hotspot.checked_mul(scale)?,
        };
        i32::try_from(result.width).ok()?;
        i32::try_from(result.height).ok()?;
        let bytes = result.width.checked_mul(result.height)?.checked_mul(4)?;
        isize::try_from(bytes).ok()?;
        Some(result)
    }

    /// Validate a locked row layout without dereferencing the supplied address.
    ///
    /// The surface owns the allocation contract. This rejects null pointers,
    /// short or negative pitches, and offsets that cannot be used by Rust slices.
    #[must_use]
    pub fn row_pitch(&self, address: usize, pitch: i32) -> Option<usize> {
        if address == 0 || self.height == 0 || self.width == 0 {
            return None;
        }
        let row = usize::try_from(self.width).ok()?.checked_mul(4)?;
        let pitch = usize::try_from(pitch).ok()?;
        if pitch < row {
            return None;
        }
        let last = usize::try_from(self.height - 1).ok()?.checked_mul(pitch)?;
        let length = last.checked_add(row)?;
        isize::try_from(length).ok()?;
        address.checked_add(length)?;
        Some(pitch)
    }
}

/// Reconcile an upload, retrying a rejected hash-only request once with pixels.
///
/// The result is the upload acknowledgment to retain for the next call. A failed
/// full upload remains unknown, so visibility changes and retargets retry it.
pub fn reconcile_upload(known: bool, mut send: impl FnMut(bool) -> bool) -> bool {
    (known && send(false)) || send(true)
}

#[cfg(test)]
mod tests;
