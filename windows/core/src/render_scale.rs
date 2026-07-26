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
mod tests {
    use super::*;

    #[test]
    fn identity_changes_nothing() {
        let s = RenderScale::IDENTITY;
        assert!(s.is_identity());
        for d in [0, 1, 7, 640, 1920, 16384] {
            assert_eq!(s.dimension(d), d);
        }
        assert_eq!(s.rect(13, 27, 640, 480), (13, 27, 640, 480));
    }

    #[test]
    fn dimension_halves() {
        let s = RenderScale::from_percent(50);
        assert_eq!(s.dimension(1920), 960);
        assert_eq!(s.dimension(1080), 540);
    }

    #[test]
    fn dimension_never_collapses_to_zero() {
        let s = RenderScale::from_percent(25);
        assert_eq!(s.dimension(1), 1);
        assert_eq!(s.dimension(2), 1);
        // Zero in, zero out: the caller rejects those separately.
        assert_eq!(s.dimension(0), 0);
    }

    #[test]
    fn from_percent_clamps_out_of_range() {
        assert_eq!(RenderScale::from_percent(0), RenderScale::from_percent(1));
        assert_eq!(RenderScale::from_percent(10_000), RenderScale::IDENTITY);
    }

    #[test]
    fn abutting_rects_stay_abutting() {
        // Three tiles sharing edges at 100 and 300 must still share them
        // after scaling, at a ratio that does not divide evenly.
        let s = RenderScale::from_percent(75);
        let (ax, _, aw, _) = s.rect(0, 0, 100, 10);
        let (bx, _, bw, _) = s.rect(100, 0, 200, 10);
        let (cx, _, cw, _) = s.rect(300, 0, 50, 10);
        assert_eq!(ax + aw, bx, "tile A must end exactly where B starts");
        assert_eq!(bx + bw, cx, "tile B must end exactly where C starts");
        assert_eq!(cx + cw, s.rect(0, 0, 350, 10).2, "total width preserved");
    }

    #[test]
    fn rect_scales_origin_and_extent_together() {
        let s = RenderScale::from_percent(50);
        assert_eq!(s.rect(100, 200, 400, 300), (50, 100, 200, 150));
    }

    #[test]
    fn never_enlarges() {
        // Supersampling is not offered: the present-side scaler only
        // upscales, so a render bigger than the drawable has no path home.
        let s = RenderScale::from_percent(200);
        assert!(s.is_identity());
        assert_eq!(s.dimension(960), 960);
    }
}
