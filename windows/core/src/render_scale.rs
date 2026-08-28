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

    /// Convert one logical dimension to its render-resolution counterpart.
    ///
    /// Never returns zero for a non-zero input: a back buffer dimension of `0`
    /// is rejected long before this, but a small render target scaled down
    /// hard could otherwise round to nothing and fail texture creation.
    #[must_use]
    pub fn dimension(self, logical: u32) -> u32 {
        if self.is_identity() || logical == 0 {
            return logical;
        }
        let scaled = (u64::from(logical) * u64::from(self.0)).div_ceil(100);
        // `logical` is a texture dimension and the scale is at most 100%, so
        // the product cannot approach `u32::MAX`; saturate rather than cast so
        // the conversion stays total either way.
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
        let (x1, x2) = (self.edge(x), self.edge(x.saturating_add(width)));
        let (y1, y2) = (self.edge(y), self.edge(y.saturating_add(height)));
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
            let scaled = self.edge(v.max(0).cast_unsigned());
            i32::try_from(scaled).unwrap_or(i32::MAX)
        };
        (e(r.0), e(r.1), e(r.2), e(r.3))
    }

    /// Scale a single rect edge, rounding to nearest.
    ///
    /// Rounding to nearest (rather than the `dimension` ceiling) is what makes
    /// adjacent rects tile: a shared edge maps to one value from both sides.
    fn edge(self, v: u32) -> u32 {
        let scaled = (u64::from(v) * u64::from(self.0) + 50) / 100;
        u32::try_from(scaled).unwrap_or(u32::MAX)
    }
}

impl Default for RenderScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests;
