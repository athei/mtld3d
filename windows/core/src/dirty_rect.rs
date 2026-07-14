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
mod tests {
    use super::DirtyRect;

    #[test]
    fn clamp_inside_is_identity() {
        let r = DirtyRect {
            x: 10,
            y: 10,
            w: 50,
            h: 50,
        };
        assert_eq!(r.clamp(128, 128), Some(r));
    }

    #[test]
    fn clamp_overflow_trims_to_mip() {
        let r = DirtyRect {
            x: 100,
            y: 100,
            w: 100,
            h: 100,
        };
        assert_eq!(
            r.clamp(128, 128),
            Some(DirtyRect {
                x: 100,
                y: 100,
                w: 28,
                h: 28,
            })
        );
    }

    #[test]
    fn clamp_origin_past_mip_is_none() {
        let r = DirtyRect {
            x: 200,
            y: 0,
            w: 10,
            h: 10,
        };
        assert_eq!(r.clamp(128, 128), None);
    }

    #[test]
    fn clamp_saturating_add_overflow_is_safe() {
        let r = DirtyRect {
            x: u32::MAX - 1,
            y: 0,
            w: u32::MAX,
            h: 1,
        };
        // right saturates to u32::MAX, clamped to mip width.
        assert_eq!(
            r.clamp(128, 128),
            None,
            "x past mip width clamps to zero area"
        );
    }

    #[test]
    fn full_spans_mip() {
        let r = DirtyRect::full(64, 32);
        assert_eq!(
            r,
            DirtyRect {
                x: 0,
                y: 0,
                w: 64,
                h: 32,
            }
        );
    }
}
