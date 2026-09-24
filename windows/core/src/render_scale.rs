//! Conversion between the resolution D3D9 reports and the one we rasterize on.
//!
//! `render.scale` splits the back buffer in two. **Logical** is what D3D9
//! reports and the space every game-supplied coordinate lives in: viewports,
//! scissor rects, `Clear` rects, `StretchRect` regions, surface descriptors.
//! **Render** is the Metal texture we actually draw into, `logical × scale`.
//! Present bridges the two.
//!
//! The rule is that a value only converts where it becomes a Metal command,
//! and only when the bound render target is the back buffer. Anything the game
//! can read back must stay logical, or D3D9's coordinate space stops agreeing
//! with `GetClientRect` and mouse input.
//!
//! The back buffer is not the only surface that shrinks. A render target or
//! depth-stencil the game creates at the reported back-buffer size is part of
//! the same image and is rasterized at the same scale, the auto depth-stencil
//! and a `D3DUSAGE_DEPTHSTENCIL` texture (INTZ, DF24, DF16, a plain depth
//! format) alike, so a colour/depth pair stays the same size and a depth
//! resolve from the attachment into such a texture stays a same-size copy.
//! Descriptors keep reporting the logical size; whatever addresses the Metal
//! texture itself, an attachment extent or a full-surface blit, measures it in
//! render space.
//!
//! A scale of 100% is an exact identity on every conversion here, so the
//! default configuration cannot perturb a single pixel.

/// Fraction of the logical resolution that gets rasterized, as a percentage.
///
/// Constructed from `Mtld3dConfig::render_scale_percent`, which the parser has
/// already bounded, and clamped again here so a caller cannot smuggle a zero
/// in and collapse a texture dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderScale(u32);

impl RenderScale {
    /// The identity scale: render and logical resolutions are the same.
    pub const IDENTITY: Self = Self(100);

    /// Build from a percentage, clamping to a range that cannot degenerate.
    #[must_use]
    pub const fn from_percent(percent: u32) -> Self {
        // Clamped to `[1, 100]` to match the parser. The ceiling is real:
        // rendering above the presented size would need a downscale on
        // present and `MTLFXSpatialScaler` only enlarges. The floor only
        // keeps the percentage itself non-zero; `dimension` separately
        // guarantees a non-zero result.
        Self(if percent == 0 {
            1
        } else if percent > 100 {
            100
        } else {
            percent
        })
    }

    /// `true` when no conversion changes anything.
    ///
    /// Call sites use this to skip the scaling work entirely, so the default
    /// configuration runs the same code path it did before the knob existed.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        self.0 == 100
    }

    /// The scale as a percentage, for logging.
    #[must_use]
    pub const fn percent(self) -> u32 {
        self.0
    }

    /// Render pixels per logical pixel, for the shader-side length conversions.
    ///
    /// The rect conversions above move coordinates; a length that D3D9 states
    /// in logical pixels and Metal consumes in render pixels (the point size)
    /// needs the ratio itself. Exactly `1.0` at the identity, so multiplying
    /// by it is a no-op on the default path.
    #[must_use]
    pub fn factor(self) -> f32 {
        if self.is_identity() {
            return 1.0;
        }
        // The percentage is bounded to `[1, 100]`, so both operands are
        // exactly representable and the quotient carries no surprise.
        f32::from(u8::try_from(self.0).unwrap_or(100)) / 100.0
    }

    /// The mip LOD bias that keeps texture detail at the logical resolution.
    ///
    /// A sampler derives its LOD from the render grid, which is `factor()`
    /// times finer in each axis than the logical one, so it lands
    /// `log2(1 / factor)` levels coarser than the logical size warrants.
    /// This is the compensating bias, `log2(factor)`: exactly `0.0` at the
    /// identity, `-1.0` at 50%.
    #[must_use]
    pub fn lod_bias(self) -> f32 {
        if self.is_identity() {
            return 0.0;
        }
        self.factor().log2()
    }

    /// Convert one logical dimension, or one rect edge, to render resolution.
    ///
    /// Rounds to nearest. This is the one rule every conversion here goes
    /// through: a texture is created at `dimension` of its logical extent and
    /// [`Self::rect`] scales each rect edge with it, so a rect that ends at the
    /// logical extent ends exactly at the texture's, and a shared edge maps to
    /// one value from both rects that meet on it.
    ///
    /// Never returns zero for a non-zero input: a back buffer dimension of `0`
    /// is rejected long before this, but a small render target scaled down
    /// hard could otherwise round to nothing and fail texture creation. With the
    /// floor the mapping is still non-decreasing, so a converted rect never
    /// inverts.
    #[must_use]
    pub fn dimension(self, logical: u32) -> u32 {
        if self.is_identity() || logical == 0 {
            return logical;
        }
        let scaled = (u64::from(logical) * u64::from(self.0) + 50) / 100;
        // `logical` is a texture dimension or a rect edge and the scale is at
        // most 100%, so the product cannot approach `u32::MAX`; saturate
        // rather than cast so the conversion stays total either way.
        u32::try_from(scaled).unwrap_or(u32::MAX).max(1)
    }

    /// Convert a logical `(x, y, width, height)` rect to render resolution.
    ///
    /// Scales the rect's **edges**, not its origin and size independently.
    /// Doing it per-component lets abutting rects gap or overlap by a pixel at
    /// scales that are not a clean fraction, which shows up as seams between
    /// tiled scissor regions or between a viewport and a clear rect.
    #[must_use]
    pub fn rect(self, x: u32, y: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
        if self.is_identity() {
            return (x, y, width, height);
        }
        let (x1, x2) = (self.dimension(x), self.dimension(x.saturating_add(width)));
        let (y1, y2) = (self.dimension(y), self.dimension(y.saturating_add(height)));
        (x1, y1, x2 - x1, y2 - y1)
    }

    /// Convert a logical half-open `(x1, y1, x2, y2)` rect to render resolution.
    ///
    /// The signed counterpart of [`Self::rect`], for the `D3DRECT`-shaped
    /// coordinates `Clear` and `StretchRect` carry. Scales the same edges the
    /// same way, so a rect converted through either entry point lands on the
    /// same pixels.
    ///
    /// A negative edge lies outside the attachment and is clipped by the caller
    /// either way, so it clamps to zero before the unsigned scale rather than
    /// inventing a rounding rule for the half-plane D3D9 cannot address.
    #[must_use]
    pub fn rect_edges_i32(self, r: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
        if self.is_identity() {
            return r;
        }
        let e = |v: i32| {
            let scaled = self.dimension(v.max(0).cast_unsigned());
            i32::try_from(scaled).unwrap_or(i32::MAX)
        };
        (e(r.0), e(r.1), e(r.2), e(r.3))
    }
}

/// A bound render target's extent in both spaces, and the conversion into it.
///
/// `logical` is the size D3D9 reports for the bound surface or mip level, the
/// space every rect the game supplies lives in. `texture` is the extent Metal
/// allocated for it. For a surface and for level 0 that is `dimension` of
/// `logical`, but a deeper level of a scaled texture is Metal's own halving of
/// the scaled base, which can differ from `dimension` of the level's logical
/// size by a texel. The rect conversions here scale through `dimension` and
/// then pin the far edge: an edge at the logical extent lands exactly on the
/// texture's, an edge inside it never passes it, and an edge beyond it never
/// falls short. So a full-target rect covers the attachment at every level,
/// and adjacent rects still share an edge.
pub struct TargetExtent {
    scale: RenderScale,
    logical: (u32, u32),
    texture: (u32, u32),
}

impl TargetExtent {
    /// An extent whose texture Metal allocated at `texture` for the reported `logical`.
    ///
    /// At the identity scale `texture` is `logical`: nothing converts.
    #[must_use]
    pub const fn new(scale: RenderScale, logical: (u32, u32), texture: (u32, u32)) -> Self {
        Self {
            scale,
            logical,
            texture,
        }
    }

    /// A surface, or level 0 of a texture: the texture is `dimension` of `logical`.
    #[must_use]
    pub fn whole(scale: RenderScale, logical: (u32, u32)) -> Self {
        let texture = (scale.dimension(logical.0), scale.dimension(logical.1));
        Self::new(scale, logical, texture)
    }

    /// Mip `level` of a texture Metal created at `base_texture`, reported as `logical`.
    ///
    /// Metal sizes every level from the base, `max(1, base >> level)` per axis,
    /// so that is the extent a pass attaching the level measures against.
    #[must_use]
    pub fn mip_level(
        scale: RenderScale,
        logical: (u32, u32),
        base_texture: (u32, u32),
        level: u32,
    ) -> Self {
        let halve = |base: u32| base.checked_shr(level).unwrap_or(0).max(1);
        Self::new(
            scale,
            logical,
            (halve(base_texture.0), halve(base_texture.1)),
        )
    }

    #[must_use]
    pub const fn scale(&self) -> RenderScale {
        self.scale
    }

    #[must_use]
    pub const fn logical(&self) -> (u32, u32) {
        self.logical
    }

    #[must_use]
    pub const fn texture(&self) -> (u32, u32) {
        self.texture
    }

    /// Convert a logical `(x, y, width, height)` rect into the texture's space.
    ///
    /// [`RenderScale::rect`] with the far edge pinned to the texture's extent.
    #[must_use]
    pub fn rect(&self, x: u32, y: u32, width: u32, height: u32) -> (u32, u32, u32, u32) {
        if self.scale.is_identity() {
            return (x, y, width, height);
        }
        let (x1, x2) = (self.edge_x(x), self.edge_x(x.saturating_add(width)));
        let (y1, y2) = (self.edge_y(y), self.edge_y(y.saturating_add(height)));
        (x1, y1, x2 - x1, y2 - y1)
    }

    /// Convert a logical half-open `(x1, y1, x2, y2)` rect into the texture's space.
    ///
    /// [`RenderScale::rect_edges_i32`] with the far edge pinned the way
    /// [`Self::rect`] pins it; a negative edge clamps to zero first.
    #[must_use]
    pub fn rect_edges_i32(&self, r: (i32, i32, i32, i32)) -> (i32, i32, i32, i32) {
        if self.scale.is_identity() {
            return r;
        }
        let signed = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        let unsigned = |v: i32| v.max(0).cast_unsigned();
        (
            signed(self.edge_x(unsigned(r.0))),
            signed(self.edge_y(unsigned(r.1))),
            signed(self.edge_x(unsigned(r.2))),
            signed(self.edge_y(unsigned(r.3))),
        )
    }

    fn edge_x(&self, v: u32) -> u32 {
        pinned_edge(self.scale.dimension(v), v, self.logical.0, self.texture.0)
    }

    fn edge_y(&self, v: u32) -> u32 {
        pinned_edge(self.scale.dimension(v), v, self.logical.1, self.texture.1)
    }
}

impl Default for RenderScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Pin a scaled edge `scaled` of logical edge `v` against an axis's two extents.
///
/// The edge at `logical` is `texture`; an edge short of it stays within the
/// texture and one past it stays beyond, which keeps the mapping monotonic.
fn pinned_edge(scaled: u32, v: u32, logical: u32, texture: u32) -> u32 {
    match v.cmp(&logical) {
        core::cmp::Ordering::Less => scaled.min(texture),
        core::cmp::Ordering::Equal => texture,
        core::cmp::Ordering::Greater => scaled.max(texture),
    }
}

#[cfg(test)]
mod tests;
