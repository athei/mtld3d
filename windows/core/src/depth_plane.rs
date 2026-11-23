//! The unscaled depth plane a pass draws into when only its depth surface is scaled.
//!
//! Under `render.scale` a render target 0 that is not back-buffer sized
//! rasterizes unscaled, while a depth surface created at the back-buffer size
//! scales with the back buffer. Metal rasterizes a pass on one grid, so a pass
//! pairing the two would test and write depth texel (x, y) of the scaled
//! surface for colour pixel (x, y), a texel that stands for another reported
//! pixel. Such a pass binds a depth texture of the surface's reported size and
//! format instead, the plane, which shares the colour target's grid. A
//! nearest-texel transfer carries the surface into the plane at the first draw
//! of the pairing that reads or writes depth or stencil, or at a depth or
//! stencil clear, and back when render target 0 or the depth binding changes,
//! before anything else reads the surface, and before the frame is submitted;
//! the passes in between share the one span. A draw that can neither fail a
//! test nor change a value (`depth_stencil_state::draw_reads_or_writes`), the
//! depth test on with a compare that always passes and depth writes off, as a
//! glow or interface pass draws, needs no plane and opens none. Nearest up
//! then nearest down is the identity at any scale up to one, so depth the
//! passes do not touch survives the round trip unchanged.
//!
//! The plane serves only that shape. A scaled render target 0 over a depth
//! surface at another scale, unscaled or scaled otherwise, keeps the target's
//! grid (`Pairing::Unserved`): the back buffer's scale follows its size under
//! `render.panelFit`, so a target and a surface created at two sizes can keep
//! two scales. The device warns once when a pass pairs them.

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
    /// Render target 0 scaled at another scale than the depth surface: no plane serves it.
    ///
    /// A scaled target over an unscaled surface, or the two at two different
    /// scales, which a target and a surface kept across a change of the back
    /// buffer's scale (`render.panelFit` follows its size) can be. The pass
    /// keeps the target's grid, and its depth lands off the reported pixels.
    Unserved,
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
/// surface out, so it pairs with nothing either. Otherwise the two share a
/// grid only at equal scales, which compare as reduced ratios per axis.
#[must_use]
pub fn pairing(targets: &PassTargets) -> Pairing {
    if targets.rt0_is_null
        || targets.rt0_samples != targets.depth_samples
        || targets.rt0_scale == targets.depth_scale
    {
        Pairing::Shared
    } else if !targets.rt0_scale.is_identity() {
        Pairing::Unserved
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

impl DepthSurface {
    /// Whether the surface has a stencil plane.
    #[must_use]
    pub fn has_stencil(&self) -> bool {
        self.format == PixelFormat::Depth32FloatStencil8
    }
}

/// The plane a device owns and the span that has it standing in for a depth surface.
///
/// The device allocates the plane, binds it while the span is open and
/// retires it; this keeps which surface the plane holds and builds the
/// transfers in and out. It also keeps what the device classified the bound
/// targets as, so a draw whose render states changed asks no surface again:
/// the device forgets the answer wherever render target 0, the depth binding
/// or an implicit surface changes.
pub struct DepthPlane {
    plane: Option<Plane>,
    /// The surface whose depth the plane holds while a span is open.
    span: Option<DepthSurface>,
    /// The reported size and format of a plane the device was refused, not asked for again.
    refused: Option<PlaneKey>,
    targets: Targets,
}

impl DepthPlane {
    /// No plane allocated, no span open, and the targets not classified.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plane: None,
            span: None,
            refused: None,
            targets: Targets::Unknown,
        }
    }

    /// Whether the bound targets are classified since they last changed.
    #[must_use]
    pub const fn targets_known(&self) -> bool {
        !matches!(self.targets, Targets::Unknown)
    }

    /// Keep the classification of the bound targets: the mixed depth surface, or none.
    ///
    /// Only whether the surface has a stencil plane is kept of it; the device
    /// reads the surface again when it opens a span.
    pub fn note_targets(&mut self, mixed: Option<&DepthSurface>) {
        self.targets = mixed.map_or(Targets::Unserved, |surface| Targets::Mixed {
            has_stencil: surface.has_stencil(),
        });
    }

    /// Forget the classification: a target, the depth binding or an implicit surface changed.
    pub const fn forget_targets(&mut self) {
        self.targets = Targets::Unknown;
    }

    /// For targets classified as mixed, whether the depth surface has a stencil plane.
    #[must_use]
    pub const fn mixed_has_stencil(&self) -> Option<bool> {
        match self.targets {
            Targets::Mixed { has_stencil } => Some(has_stencil),
            Targets::Unknown | Targets::Unserved => None,
        }
    }

    /// Whether no draw or clear can open a span now, from what is kept alone.
    ///
    /// True while a span is open and while the targets are classified as
    /// pairing on one grid: the one test a draw over such targets pays.
    #[must_use]
    pub const fn rules_out_span(&self) -> bool {
        self.span.is_some() || matches!(self.targets, Targets::Unserved)
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
    ///
    /// The frame's bindings are reasserted after it, so the targets'
    /// classification goes with the span.
    pub const fn abandon_span(&mut self) {
        self.span = None;
        self.targets = Targets::Unknown;
    }

    /// Take the plane for retirement, and forget a remembered refusal and the targets.
    ///
    /// `Reset` and the device's release call this; both replace what the
    /// targets were classified from.
    ///
    /// # Panics
    /// Panics while a span is open: close it first.
    pub fn take_plane(&mut self) -> Option<MetalHandle<MTLTextureKind>> {
        assert!(self.span.is_none(), "the plane of an open span is taken");
        self.refused = None;
        self.targets = Targets::Unknown;
        self.plane.take().map(|plane| plane.handle)
    }
}

impl Default for DepthPlane {
    fn default() -> Self {
        Self::new()
    }
}

/// What the device classified the bound render target 0 and depth surface as.
enum Targets {
    /// Not classified since a target, the depth binding or an implicit surface changed.
    Unknown,
    /// No plane serves the pair: both on one grid, or a pairing left on the surface's grid.
    Unserved,
    /// An unscaled render target 0 over a scaled depth surface the plane can serve.
    Mixed {
        /// Whether that surface has a stencil plane.
        has_stencil: bool,
    },
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
