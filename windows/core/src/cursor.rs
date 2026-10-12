//! Cursor bitmap validation, acknowledged upload decisions and the resize hide rule.
//!
//! Shared by both cursor modes before the COM wrapper reads a locked surface.

use mtld3d_types::D3DFMT_A8R8G8B8;

/// `SIZE_RESTORED`: the `WM_SIZE` `wParam` of a window resized, neither minimised nor maximised.
pub const SIZE_RESTORED: usize = 0;
/// `SIZE_MINIMIZED`: the `WM_SIZE` `wParam` of a window that was minimised.
pub const SIZE_MINIMIZED: usize = 1;
/// `SIZE_MAXIMIZED`: the `WM_SIZE` `wParam` of a window that was maximised.
pub const SIZE_MAXIMIZED: usize = 2;
/// `SIZE_MAXSHOW`: the `WM_SIZE` `wParam` telling a window another one was restored from maximised.
pub const SIZE_MAXSHOW: usize = 3;
/// `SIZE_MAXHIDE`: the `WM_SIZE` `wParam` telling a window another one was maximised.
pub const SIZE_MAXHIDE: usize = 4;

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

/// Whether a `WM_SIZE` of `size_type` drops the hides the game issues while handling it.
///
/// Some games hide the cursor while they answer a resize and show it again
/// only seconds later, which leaves the pointer gone for that long. The
/// device keeps a shown cursor shown across the game's handling of the
/// message dispatch the resize arrived in. Wine sends a restore as one
/// `WM_SYSCOMMAND` inside which the `WM_SIZE` and the activation messages
/// that follow it are sent, so that dispatch is the whole restore. A hide
/// the game issues once that dispatch has returned is honoured, since it is
/// the game hiding the cursor for its own reasons (a mouselook, say). A
/// minimise sizes nothing the game reloads for, and `SIZE_MAXSHOW` and
/// `SIZE_MAXHIDE` tell a window about another window, so they keep nothing.
/// A cursor the game had already hidden has nothing to keep.
#[must_use]
pub const fn resize_suppresses_hide(size_type: usize, cursor_shown: bool) -> bool {
    cursor_shown && matches!(size_type, SIZE_RESTORED | SIZE_MAXIMIZED)
}

#[cfg(test)]
mod tests;
