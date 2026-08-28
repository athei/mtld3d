//! Axis-aligned rectangle describing the sub-region of a texture mip.
//!
//! Delimits what a `Lock` / `AddDirtyRect` call touched, and the region a copy
//! between two mips may write. Pure geometry, no platform APIs, so the clamp,
//! clip and full helpers can be host-tested.

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

    /// The bounding box of `self` and `other`.
    ///
    /// The upload and source-dirty trackers keep one rect per level, so two
    /// writes merge into the box enclosing both: the texels between them are
    /// copied once more than they had to be, never skipped.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.w).max(other.x + other.w);
        let bottom = (self.y + self.h).max(other.y + other.h);
        Self {
            x,
            y,
            w: right - x,
            h: bottom - y,
        }
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

    /// Clip a copy region to `(level_w, level_h)` on the format's block grid.
    ///
    /// The region first grows to the blocks enclosing it, because a
    /// block-compressed copy addresses whole blocks, and is then trimmed to the
    /// level. A trailing partial block at the level's own edge survives, which
    /// is what the upload blit accepts. `block_w` / `block_h` are `1` for
    /// uncompressed formats, where this reduces to [`Self::clamp`].
    ///
    /// Returns `None` when nothing of the region lies inside the level.
    #[must_use]
    pub fn clip_to_level(
        self,
        level_w: u32,
        level_h: u32,
        block_w: u32,
        block_h: u32,
    ) -> Option<Self> {
        let bw = block_w.max(1);
        let bh = block_h.max(1);
        let x = self.x - self.x % bw;
        let y = self.y - self.y % bh;
        let right = self
            .x
            .saturating_add(self.w)
            .div_ceil(bw)
            .saturating_mul(bw);
        let bottom = self
            .y
            .saturating_add(self.h)
            .div_ceil(bh)
            .saturating_mul(bh);
        Self {
            x,
            y,
            w: right - x,
            h: bottom - y,
        }
        .clamp(level_w, level_h)
    }
}

/// Clip a copy region to the two mip levels it spans.
///
/// `region` is the rectangle within the source level and `dst_origin` where its
/// top-left corner lands in the destination level. Each half is rounded out to
/// the format's block grid and trimmed to its own level, then the smaller of the
/// two extents is applied to both, so neither half reaches past its level and
/// the two stay the same size. `block` is `(1, 1)` for uncompressed formats.
///
/// Returns `None` when the two levels share no part of the region.
#[must_use]
pub fn clip_copy_region(
    region: DirtyRect,
    dst_origin: (u32, u32),
    src_level: (u32, u32),
    dst_level: (u32, u32),
    block: (u32, u32),
) -> Option<(DirtyRect, DirtyRect)> {
    let src = region.clip_to_level(src_level.0, src_level.1, block.0, block.1)?;
    let dst = DirtyRect {
        x: dst_origin.0,
        y: dst_origin.1,
        w: region.w,
        h: region.h,
    }
    .clip_to_level(dst_level.0, dst_level.1, block.0, block.1)?;
    let w = src.w.min(dst.w);
    let h = src.h.min(dst.h);
    Some((DirtyRect { w, h, ..src }, DirtyRect { w, h, ..dst }))
}

#[cfg(test)]
mod tests;
