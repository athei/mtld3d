//! Axis-aligned rectangle describing the sub-region of a texture mip.
//!
//! Delimits what a `Lock` / `AddDirtyRect` call touched. Pure geometry — no
//! platform APIs — so the clamp / full helpers can be host-tested.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl DirtyRect {
    #[must_use]
    pub const fn full(w: u32, h: u32) -> Self {
        Self { x: 0, y: 0, w, h }
    }

    /// Clamp to `(mip_w, mip_h)`.
    ///
    /// Returns `None` when the rect falls entirely outside the mip, so the
    /// caller can treat it as a no-op.
    #[must_use]
    pub fn clamp(self, mip_w: u32, mip_h: u32) -> Option<Self> {
        let x = self.x.min(mip_w);
        let y = self.y.min(mip_h);
        let right = self.x.saturating_add(self.w).min(mip_w);
        let bottom = self.y.saturating_add(self.h).min(mip_h);
        if right <= x || bottom <= y {
            return None;
        }
        Some(Self {
            x,
            y,
            w: right - x,
            h: bottom - y,
        })
    }
}

#[cfg(test)]
mod tests;
