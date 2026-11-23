//! Conversion between the resolution D3D9 reports and the one we rasterize on.
//!
//! A render scale splits the back buffer in two. **Logical** is what D3D9
//! reports and the space every game-supplied coordinate lives in: viewports,
//! scissor rects, `Clear` rects, `StretchRect` regions, surface descriptors.
//! **Render** is the Metal texture we actually draw into, `logical × scale`
//! per axis. Present bridges the two.
//!
//! Two settings make the scale. `render.scale` is one fraction for both
//! axes. `render.panelFit` rasterizes a back buffer larger than the panel of
//! a screen scaled down to it at the panel's share, panel over backing per
//! axis ([`panel_fit`]). The two compose into one exact ratio per axis
//! ([`RenderScale::compose`]), and every size and edge rounds once from it.
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
//! The identity is an exact identity on every conversion here, so the
//! default configuration on an unscaled screen cannot perturb a single pixel.

use mtld3d_shared::render_size::{RenderScaleWire, scale_rounded, scaled_down_to_panel};

/// The largest screen dimension [`panel_fit`] takes a ratio from.
///
/// Screens are far below it. The bound keeps a composed denominator, a
/// `render.scale` denominator of at most 100 times a backing dimension,
/// inside a `u32`.
const MAX_SCREEN_DIMENSION: u32 = 1 << 16;

/// Render pixels per logical pixel, an exact ratio per axis.
///
/// Each axis holds a reduced fraction in `(0, 1]`: rendering above the
/// presented size would need a downscale on present, and `MTLFXSpatialScaler`
/// only enlarges. A percentage builds the same ratio in both axes; the panel
/// fit builds panel over backing per axis. Every constructor keeps the
/// fraction reduced, so two scales that convert alike compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderScale(RenderScaleWire);

impl RenderScale {
    /// The identity scale: render and logical resolutions are the same.
    pub const IDENTITY: Self = Self(RenderScaleWire::IDENTITY);

    /// Build from a percentage for both axes, clamping to a range that cannot degenerate.
    ///
    /// Constructed from `Mtld3dConfig::render_scale_percent`, which the parser
    /// has already bounded, and clamped again here so a caller cannot smuggle
    /// a zero in. The floor only keeps the percentage itself non-zero;
    /// [`Self::size`] separately guarantees a non-zero result.
    #[must_use]
    pub const fn from_percent(percent: u32) -> Self {
        let percent = if percent == 0 {
            1
        } else if percent > 100 {
            100
        } else {
            percent
        };
        let (num, den) = reduce(percent, 100);
        Self(RenderScaleWire {
            width_num: num,
            width_den: den,
            height_num: num,
            height_den: den,
        })
    }

    /// Read a scale back from the wire.
    ///
    /// An axis outside `(0, 1]`, which [`Self::is_valid_wire`] rejects, reads
    /// as the identity, so a dimension never collapses or grows.
    #[must_use]
    pub const fn from_wire(wire: &RenderScaleWire) -> Self {
        let (width_num, width_den) = checked_axis(wire.width_num, wire.width_den);
        let (height_num, height_den) = checked_axis(wire.height_num, wire.height_den);
        Self(RenderScaleWire {
            width_num,
            width_den,
            height_num,
            height_den,
        })
    }

    /// Whether every axis of `wire` is a ratio in `(0, 1]`.
    #[must_use]
    pub const fn is_valid_wire(wire: &RenderScaleWire) -> bool {
        valid_axis(wire.width_num, wire.width_den) && valid_axis(wire.height_num, wire.height_den)
    }

    /// The scale as it crosses the boundary.
    #[must_use]
    pub const fn to_wire(self) -> RenderScaleWire {
        self.0
    }

    /// `true` when no conversion changes anything.
    ///
    /// Call sites use this to skip the scaling work entirely, so the default
    /// configuration runs the same code path it did before the knob existed.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        self.0.width_num == self.0.width_den && self.0.height_num == self.0.height_den
    }

    /// `self` applied after `other`: the product of the two ratios per axis.
    ///
    /// The product is exact, so a size scaled by the composition rounds once
    /// rather than once per factor. A product whose reduced denominator would
    /// not fit a `u32`, which no percentage and [`panel_fit`] pair reaches,
    /// is halved until it does.
    #[must_use]
    pub fn compose(self, other: Self) -> Self {
        let (width_num, width_den) = compose_axis(
            (self.0.width_num, self.0.width_den),
            (other.0.width_num, other.0.width_den),
        );
        let (height_num, height_den) = compose_axis(
            (self.0.height_num, self.0.height_den),
            (other.0.height_num, other.0.height_den),
        );
        Self(RenderScaleWire {
            width_num,
            width_den,
            height_num,
            height_den,
        })
    }

    /// Render pixels per logical pixel, for the shader-side length conversions.
    ///
    /// The rect conversions move coordinates; a length that D3D9 states in
    /// logical pixels and Metal consumes in render pixels (the point size)
    /// needs the ratio itself. A scale that differs between the axes gives the
    /// geometric mean of the two. A built-in Mac panel's scaled modes keep the
    /// panel's aspect, so there the axes differ by under 0.1% and the mean is
    /// either axis; a screen whose panel has another aspect than its backing
    /// (the CI runner's 1920x1080 under 2048x1536, 0.9375 by 0.703) puts a
    /// length off by the spread between them. Exactly `1.0` at the identity, so multiplying by
    /// it is a no-op on the default path, and exactly the percentage's
    /// quotient for a percentage.
    #[must_use]
    pub fn factor(self) -> f32 {
        if self.is_identity() {
            return 1.0;
        }
        let (width, height) = self.axis_factors();
        if self.0.width_num == self.0.height_num && self.0.width_den == self.0.height_den {
            return width;
        }
        (width * height).sqrt()
    }

    /// Render pixels per logical pixel along each axis, for a position the shader converts.
    ///
    /// `vPos` maps a rasterized position back to the reported one, which needs
    /// each axis's own ratio rather than [`Self::factor`]'s mean. Each is
    /// exactly `1.0` at the identity and exactly the percentage's quotient for
    /// a percentage.
    #[must_use]
    pub fn axis_factors(self) -> (f32, f32) {
        let ratio = |num: u32, den: u32| {
            if num == den {
                1.0
            } else {
                u32_to_f32(num) / u32_to_f32(den)
            }
        };
        (
            ratio(self.0.width_num, self.0.width_den),
            ratio(self.0.height_num, self.0.height_den),
        )
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

    /// Convert a logical `(width, height)` size, or a point's two edges, to render resolution.
    ///
    /// Rounds each axis to nearest by its own ratio, halves up. This is the
    /// one rule every conversion here goes through: a texture is created at
    /// `size` of its logical extent and [`Self::rect`] scales each rect edge
    /// with it, so a rect that ends at the logical extent ends exactly at the
    /// texture's, and a shared edge maps to one value from both rects that
    /// meet on it. A percentage gives `(logical × percent + 50) / 100`.
    ///
    /// Never returns zero for a non-zero input: a back buffer dimension of `0`
    /// is rejected long before this, but a small render target scaled down
    /// hard could otherwise round to nothing and fail texture creation. With the
    /// floor the mapping is still non-decreasing, so a converted rect never
    /// inverts.
    #[must_use]
    pub fn size(self, logical: (u32, u32)) -> (u32, u32) {
        (self.width_edge(logical.0), self.height_edge(logical.1))
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
        let (x1, x2) = (self.width_edge(x), self.width_edge(x.saturating_add(width)));
        let (y1, y2) = (
            self.height_edge(y),
            self.height_edge(y.saturating_add(height)),
        );
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
        let signed = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        let unsigned = |v: i32| v.max(0).cast_unsigned();
        (
            signed(self.width_edge(unsigned(r.0))),
            signed(self.height_edge(unsigned(r.1))),
            signed(self.width_edge(unsigned(r.2))),
            signed(self.height_edge(unsigned(r.3))),
        )
    }

    fn width_edge(self, v: u32) -> u32 {
        scale_axis(v, self.0.width_num, self.0.width_den)
    }

    fn height_edge(self, v: u32) -> u32 {
        scale_axis(v, self.0.height_num, self.0.height_den)
    }
}

/// A bound render target's extent in both spaces, and the conversion into it.
///
/// `logical` is the size D3D9 reports for the bound surface or mip level, the
/// space every rect the game supplies lives in. `texture` is the extent Metal
/// allocated for it. For a surface and for level 0 that is `size` of
/// `logical`, but a deeper level of a scaled texture is Metal's own halving of
/// the scaled base, which can differ from `size` of the level's logical
/// size by a texel. The rect conversions here scale each axis through its own
/// ratio and then pin the far edge: an edge at the logical extent lands
/// exactly on the texture's, an edge inside it never passes it, and an edge
/// beyond it never falls short. So a full-target rect covers the attachment at
/// every level, and adjacent rects still share an edge.
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

    /// A surface, or level 0 of a texture: the texture is `size` of `logical`.
    #[must_use]
    pub fn whole(scale: RenderScale, logical: (u32, u32)) -> Self {
        Self::new(scale, logical, scale.size(logical))
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
        pinned_edge(self.scale.width_edge(v), v, self.logical.0, self.texture.0)
    }

    fn edge_y(&self, v: u32) -> u32 {
        pinned_edge(self.scale.height_edge(v), v, self.logical.1, self.texture.1)
    }
}

/// What decides a back buffer's render scale besides its size.
///
/// The user's `render.scale` (already resolved against what the GPU can
/// resample), `render.panelFit`, and the panel and backing sizes of the
/// window's screen. A device keeps one and asks it again whenever its back
/// buffer is sized: at creation, `Reset` and an auto-resize.
pub struct BackBufferScale {
    user: RenderScale,
    panel_fit: bool,
    panel: Option<(u32, u32)>,
    backing: Option<(u32, u32)>,
}

impl BackBufferScale {
    /// The user's scale and `render.panelFit`, on a screen whose sizes are not known yet.
    #[must_use]
    pub const fn new(user: RenderScale, panel_fit: bool) -> Self {
        Self {
            user,
            panel_fit,
            panel: None,
            backing: None,
        }
    }

    /// Adopt the panel and backing sizes of the window's screen; `true` when either changed.
    pub fn set_screen(&mut self, panel: Option<(u32, u32)>, backing: Option<(u32, u32)>) -> bool {
        let changed = self.panel != panel || self.backing != backing;
        self.panel = panel;
        self.backing = backing;
        changed
    }

    /// The panel size the scale was taken against, `None` when unknown.
    #[must_use]
    pub const fn panel(&self) -> Option<(u32, u32)> {
        self.panel
    }

    /// The backing size the scale was taken against, `None` when unknown.
    #[must_use]
    pub const fn backing(&self) -> Option<(u32, u32)> {
        self.backing
    }

    /// The scale a back buffer of `size` rasterizes at: the user's after the panel fit.
    #[must_use]
    pub fn for_back_buffer(&self, size: (u32, u32)) -> RenderScale {
        self.user
            .compose(panel_fit(self.panel_fit, size, self.panel, self.backing))
    }
}

/// The scale that rasterizes `back_buffer` at the panel's pixels, or the identity.
///
/// A screen scaled down to its panel (its backing store larger than the
/// panel in either axis) composites every frame at the backing size and the
/// display scales it down. A back buffer larger than the panel in either axis
/// then rasterizes at back buffer × panel / backing per axis, the size
/// present caps the drawable at
/// ([`panel_drawable_size`](mtld3d_shared::render_size::panel_drawable_size)),
/// so present is a 1:1 copy. `enabled` is `render.panelFit`. An unknown or
/// out-of-range size, an unscaled screen, a back buffer within the panel and
/// a disabled fit give the identity; an axis whose panel is not smaller than
/// its backing keeps its own size.
#[must_use]
pub fn panel_fit(
    enabled: bool,
    back_buffer: (u32, u32),
    panel: Option<(u32, u32)>,
    backing: Option<(u32, u32)>,
) -> RenderScale {
    let in_range = |size: Option<(u32, u32)>| {
        size.filter(|&(width, height)| {
            (1..=MAX_SCREEN_DIMENSION).contains(&width)
                && (1..=MAX_SCREEN_DIMENSION).contains(&height)
        })
    };
    let (panel, backing) = (in_range(panel), in_range(backing));
    let (Some(panel_size), Some(backing_size)) = (panel, backing) else {
        return RenderScale::IDENTITY;
    };
    let above_panel = back_buffer.0 > panel_size.0 || back_buffer.1 > panel_size.1;
    if !enabled || !above_panel || !scaled_down_to_panel(backing, panel) {
        return RenderScale::IDENTITY;
    }
    let axis = |panel: u32, backing: u32| {
        if panel < backing {
            reduce(panel, backing)
        } else {
            (1, 1)
        }
    };
    let (width_num, width_den) = axis(panel_size.0, backing_size.0);
    let (height_num, height_den) = axis(panel_size.1, backing_size.1);
    RenderScale(RenderScaleWire {
        width_num,
        width_den,
        height_num,
        height_den,
    })
}

/// How many mip levels the Metal texture of a texture rasterized at `scale` holds.
///
/// D3D9 reports `levels`, counted on the reported base extent `logical`; the
/// Metal texture is created at `scale` of that extent, and a smaller base can
/// hold a shorter chain (640x480 has ten levels, its 480x360 at 0.75 nine).
/// The Metal chain is the shorter of the two, so it never asks Metal for a
/// level its base cannot have. At the identity it is `levels` itself.
#[must_use]
pub fn rasterized_level_count(scale: RenderScale, logical: (u32, u32), levels: u32) -> u32 {
    if scale.is_identity() {
        return levels;
    }
    let (width, height) = scale.size(logical);
    levels.min(crate::format::compute_mip_count(width, height))
}

/// The Metal level a reported mip `level` lands on in a chain of `rasterized_levels`.
///
/// Every level the Metal chain holds is itself. A reported level past its end,
/// which only a texture [`rasterized_level_count`] shortened has, lands on the
/// chain's last level: those tail levels are a texel or two on a side and all
/// share that one Metal level, so a write into one is seen by the others.
#[must_use]
pub const fn rasterized_level(level: u32, rasterized_levels: u32) -> u32 {
    if level < rasterized_levels {
        level
    } else {
        rasterized_levels.saturating_sub(1)
    }
}

impl Default for RenderScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Scale one logical edge by `num / den`, never to zero from a non-zero edge.
fn scale_axis(v: u32, num: u32, den: u32) -> u32 {
    if num == den || v == 0 {
        return v;
    }
    // `num < den`, so the result is at most `v`; saturate rather than cast
    // so the conversion stays total either way.
    u32::try_from(scale_rounded(v, num, den))
        .unwrap_or(u32::MAX)
        .max(1)
}

/// `num / den` in lowest terms; `den` is non-zero.
const fn reduce(num: u32, den: u32) -> (u32, u32) {
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    (num / a, den / a)
}

/// Whether `num / den` is a ratio in `(0, 1]`.
const fn valid_axis(num: u32, den: u32) -> bool {
    num != 0 && num <= den
}

/// `num / den` in lowest terms when it lies in `(0, 1]`, else the identity.
const fn checked_axis(num: u32, den: u32) -> (u32, u32) {
    if valid_axis(num, den) {
        reduce(num, den)
    } else {
        (1, 1)
    }
}

/// The product of two ratios in lowest terms, held to a `u32` denominator.
fn compose_axis(a: (u32, u32), b: (u32, u32)) -> (u32, u32) {
    let mut num = u64::from(a.0) * u64::from(b.0);
    let mut den = u64::from(a.1) * u64::from(b.1);
    let (mut x, mut y) = (num, den);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (num, den) = (num / x, den / x);
    while den > u64::from(u32::MAX) {
        (num, den) = ((num >> 1).max(1), den >> 1);
    }
    // Both fit: the denominator was halved into a `u32` above and the
    // numerator is at most the denominator.
    (
        u32::try_from(num).unwrap_or(u32::MAX),
        u32::try_from(den).unwrap_or(u32::MAX),
    )
}

/// The nearest `f32` to `v`, with no lossy cast.
///
/// The high half scaled by 2^16 is exact, so the one rounding is the add's.
fn u32_to_f32(v: u32) -> f32 {
    let [b0, b1, b2, b3] = v.to_le_bytes();
    let high = f32::from(u16::from_le_bytes([b2, b3])) * 65_536.0;
    high + f32::from(u16::from_le_bytes([b0, b1]))
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
