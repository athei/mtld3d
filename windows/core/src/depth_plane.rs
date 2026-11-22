//! The unscaled depth plane a pass draws into when only its depth surface is scaled.
//!
//! Under `render.scale` a render target 0 that is not back-buffer sized
//! rasterizes unscaled, while a depth surface created at the back-buffer size
//! scales with the back buffer. Metal rasterizes a pass on one grid, so a pass
//! pairing the two would test and write depth texel (x, y) of the scaled
//! surface for colour pixel (x, y), a texel that stands for another reported
//! pixel. Such a pass binds a depth texture of the surface's reported size and
//! format instead, the plane, which shares the colour target's grid. A
//! nearest-texel transfer carries the surface into the plane when a pass first
//! opens with the pairing, and back when render target 0 or the depth binding
//! changes, before anything else reads the surface, and before the frame is
//! submitted; the passes in between share the one span. Nearest up then
//! nearest down is the identity at any scale up to one, so depth the passes do
//! not touch survives the round trip unchanged.

use mtld3d_shared::{MetalHandle, mtl::PixelFormat, mtl_handle::MTLTextureKind};

use crate::{encoder_data::DepthTransfer, render_scale::RenderScale};

/// How render target 0 and the depth surface share a raster grid.
#[derive(Debug, PartialEq, Eq)]
pub enum Pairing {
    /// Both are rasterized at one scale, or the pass binds no depth surface beside render target 0.
    Shared,
    /// Render target 0 unscaled over a scaled, single-sampled depth surface: the plane serves it.
    Mixed,
    /// The mixed shape over a multisampled depth surface, which the transfer cannot write.
    MixedMultisampled,
    /// Render target 0 scaled over an unscaled depth surface, which keeps the scaled grid.
    Reverse,
}

/// The attachments a pass would open with, as far as their grids go.
pub struct PassTargets {
    /// What render target 0 is rasterized at.
    pub rt0_scale: RenderScale,
    /// Whether render target 0 is a `NULL` surface, shaped after the depth surface.
    pub rt0_is_null: bool,
    /// Render target 0's sample count.
    pub rt0_samples: u8,
    /// What the depth surface is rasterized at.
    pub depth_scale: RenderScale,
    /// The depth surface's sample count.
    pub depth_samples: u8,
}

/// Classify the pass a draw or clear opens from what its attachments are rasterized at.
///
/// A `NULL` render target 0 is shaped after the depth surface, scale
/// included, so it never pairs with one of another grid. A pass whose
/// render target 0 and depth surface disagree on samples leaves the depth
/// surface out, so it pairs with nothing either.
#[must_use]
pub const fn pairing(targets: &PassTargets) -> Pairing {
    let rt0_scaled = !targets.rt0_scale.is_identity();
    let depth_scaled = !targets.depth_scale.is_identity();
    if targets.rt0_is_null
        || targets.rt0_samples != targets.depth_samples
        || rt0_scaled == depth_scaled
    {
        Pairing::Shared
    } else if rt0_scaled {
        Pairing::Reverse
    } else if targets.depth_samples > 1 {
        Pairing::MixedMultisampled
    } else {
        Pairing::Mixed
    }
}

/// The scaled depth surface a span carries into the plane and back.
pub struct DepthSurface {
    /// The surface's Metal texture.
    pub handle: MetalHandle<MTLTextureKind>,
    /// The texture's extent, the reported one under `scale`.
    pub texture_size: (u32, u32),
    /// The extent D3D9 reports for the surface, which the plane takes.
    pub logical_size: (u32, u32),
    /// The texture's format, which the plane takes.
    pub format: PixelFormat,
    /// What the surface is rasterized at, for binding it again when the span closes.
    pub scale: RenderScale,
}

/// The plane a device owns and the span that has it standing in for a depth surface.
///
/// The device allocates the plane, binds it while the span is open and
/// retires it; this keeps which surface the plane holds and builds the
/// transfers in and out.
pub struct DepthPlane {
    plane: Option<Plane>,
    /// The surface whose depth the plane holds while a span is open.
    span: Option<DepthSurface>,
    /// The reported size and format of a plane the device was refused, not asked for again.
    refused: Option<PlaneKey>,
}

impl DepthPlane {
    /// No plane allocated and no span open.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plane: None,
            span: None,
            refused: None,
        }
    }

    /// Whether a plane for `surface` was refused, so allocating one again is not worth a try.
    #[must_use]
    pub fn refused_for(&self, surface: &DepthSurface) -> bool {
        self.refused == Some(PlaneKey::of(surface))
    }

    /// Remember that a plane for `surface` was refused, until a plane is installed or taken.
    pub const fn note_refused(&mut self, surface: &DepthSurface) {
        self.refused = Some(PlaneKey::of(surface));
    }

    /// Whether a span is open, so the plane stands in for a depth surface.
    #[must_use]
    pub const fn span_open(&self) -> bool {
        self.span.is_some()
    }

    /// The plane, when one of `surface`'s reported size and format is allocated.
    #[must_use]
    pub fn plane_for(&self, surface: &DepthSurface) -> Option<MetalHandle<MTLTextureKind>> {
        let key = PlaneKey::of(surface);
        self.plane
            .as_ref()
            .filter(|plane| plane.key == key)
            .map(|plane| plane.handle)
    }

    /// Install a newly allocated plane for `surface` and hand back the one it replaces.
    ///
    /// The replaced plane is the caller's to retire. A refusal remembered
    /// for an earlier key is forgotten.
    ///
    /// # Panics
    /// Panics while a span is open: the plane in use cannot be replaced.
    pub fn replace_plane(
        &mut self,
        handle: MetalHandle<MTLTextureKind>,
        surface: &DepthSurface,
    ) -> Option<MetalHandle<MTLTextureKind>> {
        assert!(self.span.is_none(), "the plane of an open span is replaced");
        self.refused = None;
        self.plane
            .replace(Plane {
                handle,
                key: PlaneKey::of(surface),
            })
            .map(|plane| plane.handle)
    }

    /// Open a span over `surface` and return the transfer that carries it into the plane.
    ///
    /// `None` when no plane of the surface's size and format is allocated or a
    /// span is already open; nothing changes then.
    pub fn open(&mut self, surface: DepthSurface) -> Option<DepthTransfer> {
        if self.span.is_some() {
            return None;
        }
        let plane = self.plane_for(&surface)?;
        let transfer = DepthTransfer {
            source: surface.handle,
            source_level: 0,
            source_size: surface.texture_size,
            source_format: surface.format,
            source_samples: 1,
            destination: plane,
            destination_size: surface.logical_size,
            destination_format: surface.format,
        };
        self.span = Some(surface);
        Some(transfer)
    }

    /// Close the open span: the transfer that carries the plane back, and the surface to rebind.
    ///
    /// # Panics
    /// Panics if the open span has no plane, which [`Self::open`] and
    /// [`Self::replace_plane`] rule out.
    pub fn close(&mut self) -> Option<(DepthTransfer, DepthSurface)> {
        let surface = self.span.take()?;
        let plane = self
            .plane
            .as_ref()
            .expect("an open span has its plane")
            .handle;
        let transfer = DepthTransfer {
            source: plane,
            source_level: 0,
            source_size: surface.logical_size,
            source_format: surface.format,
            source_samples: 1,
            destination: surface.handle,
            destination_size: surface.texture_size,
            destination_format: surface.format,
        };
        Some((transfer, surface))
    }

    /// Forget an open span without carrying the plane back, for a frame that is dropped whole.
    pub const fn abandon_span(&mut self) {
        self.span = None;
    }

    /// Take the plane for retirement, and forget a remembered refusal.
    ///
    /// # Panics
    /// Panics while a span is open: close it first.
    pub fn take_plane(&mut self) -> Option<MetalHandle<MTLTextureKind>> {
        assert!(self.span.is_none(), "the plane of an open span is taken");
        self.refused = None;
        self.plane.take().map(|plane| plane.handle)
    }
}

impl Default for DepthPlane {
    fn default() -> Self {
        Self::new()
    }
}

/// The device's plane: a Private depth texture of one key.
struct Plane {
    handle: MetalHandle<MTLTextureKind>,
    key: PlaneKey,
}

/// What a plane serves: a depth surface's reported size and format.
#[derive(PartialEq, Eq)]
struct PlaneKey {
    logical_size: (u32, u32),
    format: PixelFormat,
}

impl PlaneKey {
    const fn of(surface: &DepthSurface) -> Self {
        Self {
            logical_size: surface.logical_size,
            format: surface.format,
        }
    }
}

#[cfg(test)]
mod tests;
