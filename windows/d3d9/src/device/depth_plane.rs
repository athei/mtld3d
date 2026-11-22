//! When a pass draws into the depth plane instead of the depth surface the application bound.
//!
//! `mtld3d_core::depth_plane` says what the plane is and builds its transfers.
//! The device decides here when a span opens (a draw that tests or writes
//! depth or stencil, or a depth or stencil clear, about to open a pass through
//! an unscaled render target 0 over a scaled depth surface) and when it closes
//! (render target 0 or the depth binding changes, a `StretchRect` or a RESZ
//! resolve reads or writes a depth surface, the frame is submitted, `Reset`,
//! release), allocates and retires the plane, and records the transfers and
//! the depth bindings into the frame. The saved bindings keep naming the
//! surface the application bound, so a fresh frame starts on it and opens a
//! span of its own when its first pass needs one.

use mtld3d_core::{
    depth_plane::{DepthSurface, Pairing, PassTargets, pairing},
    format::{is_null_render_target, map_d3d_depth_format},
    render_scale::RenderScale,
};
use mtld3d_shared::{
    CreateDepthTextureParams, MetalHandle, mtl::PixelFormat, mtl_handle::MTLTextureKind,
};
use mtld3d_types::{D3DRS_STENCILENABLE, D3DRS_ZENABLE};

use super::{
    DepthBinding, DeviceFlags, DeviceInner, LOG_TARGET, ResolveDepthSurfaceOp, SnapshotDirty,
};
use crate::unix_call::unix_call;

impl DeviceInner {
    /// Open a span ahead of a draw whose targets or render states changed, if the draw needs one.
    ///
    /// A draw that neither tests nor writes depth or stencil leaves the
    /// depth surface as it is, so it opens nothing; the draw after it that
    /// turns either test on does.
    pub fn open_depth_span_for_draw(&mut self) {
        if self.render_scale.is_identity() {
            return;
        }
        if self.render_state(D3DRS_ZENABLE as usize) == 0
            && self.render_state(D3DRS_STENCILENABLE as usize) == 0
        {
            return;
        }
        self.open_depth_span();
    }

    /// Open a span if the pass about to open pairs an unscaled render target 0 with a scaled depth.
    ///
    /// Called ahead of a draw that tests or writes depth or stencil and ahead
    /// of a depth or stencil clear. The transfer into the plane and the
    /// plane's bind go into the frame before the draw or clear, so its pass
    /// opens on the plane. A plane that cannot be allocated leaves the pass on
    /// the surface, as without one, warned once, and is not asked for again
    /// until its size or format changes or the device resets.
    pub fn open_depth_span(&mut self) {
        if self.render_scale.is_identity() || self.depth_plane.span_open() {
            return;
        }
        let Some(surface) = self.mixed_depth_surface() else {
            return;
        };
        let Some(plane) = self
            .depth_plane
            .plane_for(&surface)
            .or_else(|| self.allocate_depth_plane(&surface))
        else {
            return;
        };
        let logical_size = surface.logical_size;
        let has_stencil = surface.format == PixelFormat::Depth32FloatStencil8;
        let Some(transfer) = self.depth_plane.open(surface) else {
            return;
        };
        if self.frame_dump.active {
            self.frame_dump_event("depth plane: span opened");
        }
        self.push_control(ResolveDepthSurfaceOp { transfer });
        self.push_depth_binding_op(
            DepthBinding::Eager(plane, logical_size, RenderScale::IDENTITY),
            false,
            has_stencil,
            1,
        );
    }

    /// Close the open span: carry the plane back into its surface and bind the surface again.
    ///
    /// Called before anything replaces render target 0 or the depth binding,
    /// before a `StretchRect` that copies a depth surface or a RESZ resolve,
    /// and before every submission. The next draw re-reads its targets, so it
    /// opens a span again when its pass needs one.
    pub fn close_depth_span(&mut self) {
        let Some((transfer, surface)) = self.depth_plane.close() else {
            return;
        };
        if self.frame_dump.active {
            self.frame_dump_event("depth plane: span closed");
        }
        self.push_control(ResolveDepthSurfaceOp { transfer });
        let has_stencil = surface.format == PixelFormat::Depth32FloatStencil8;
        // A span only opens over a surface no sampler can reach, which is
        // what the binding `SetDepthStencilSurface` made for it says too.
        self.push_depth_binding_op(
            DepthBinding::Eager(surface.handle, surface.texture_size, surface.scale),
            false,
            has_stencil,
            1,
        );
        self.mark_snapshot_dirty(SnapshotDirty::RT_DS);
    }

    /// Close the open span and retire the plane, for `Reset` and the device's release.
    pub fn release_depth_plane(&mut self) {
        self.close_depth_span();
        if let Some(plane) = self.depth_plane.take_plane() {
            self.retire_depth_target(plane);
        }
    }

    /// Forget the open span of a frame that is dropped whole, which carries nothing back.
    pub const fn abandon_depth_span(&mut self) {
        self.depth_plane.abandon_span();
    }

    /// The depth surface a pass would pair with an unscaled render target 0 on another grid.
    ///
    /// `None` for every pairing on one grid. A texture-backed depth surface
    /// has no Metal handle on this side, the transfer cannot write a
    /// multisampled one, and a scaled render target 0 over an unscaled depth
    /// surface has no plane at all, so each keeps the pass on the surface's
    /// grid, warned once.
    fn mixed_depth_surface(&self) -> Option<DepthSurface> {
        let rt0 = self.bound_rt.render_target(0);
        let (rt0_scale, rt0_is_null, rt0_samples) = if rt0.is_null() {
            (self.render_scale, false, self.backbuffer_sample_count)
        } else {
            // SAFETY: a bound render target is kept live by its binding.
            let rt0 = unsafe { &*rt0 };
            (
                rt0.reported_extent().1,
                is_null_render_target(rt0.reported_format()),
                rt0.multi_sample().sample_count,
            )
        };
        let depth = self.bound_rt.depth_stencil();
        let (surface, sampleable, depth_samples) = if depth.is_null() {
            if self.flags.contains(DeviceFlags::DEPTH_EXPLICITLY_UNBOUND)
                || self.depth_stencil_handle.is_null()
            {
                return None;
            }
            let logical_size = (self.backbuffer_width, self.backbuffer_height);
            let scale = self.render_scale;
            (
                DepthSurface {
                    handle: self.depth_stencil_handle,
                    texture_size: (
                        scale.dimension(logical_size.0),
                        scale.dimension(logical_size.1),
                    ),
                    logical_size,
                    format: map_d3d_depth_format(self.depth_stencil_format)?,
                    scale,
                },
                false,
                self.backbuffer_sample_count,
            )
        } else {
            // SAFETY: a bound depth surface is kept live by its binding.
            let depth = unsafe { &*depth };
            let (logical_size, scale) = depth.reported_extent();
            (
                DepthSurface {
                    handle: depth.metal_depth_handle(),
                    texture_size: (
                        scale.dimension(logical_size.0),
                        scale.dimension(logical_size.1),
                    ),
                    logical_size,
                    format: map_d3d_depth_format(depth.depth_attachment_format())?,
                    scale,
                },
                depth.depth_is_sampleable(),
                depth.multi_sample().sample_count,
            )
        };
        let targets = PassTargets {
            rt0_scale,
            rt0_is_null,
            rt0_samples,
            depth_scale: surface.scale,
            depth_samples,
        };
        match pairing(&targets) {
            Pairing::Shared => None,
            Pairing::Mixed if sampleable => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "a pass through an unscaled render target over a scaled depth texture draws \
                     into the texture's scaled grid: its depth lands off the reported pixels"
                );
                None
            }
            Pairing::Mixed => (!surface.handle.is_null()).then_some(surface),
            Pairing::MixedMultisampled => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "a pass through an unscaled render target over a scaled multisampled depth \
                     surface draws into the surface's scaled grid: its depth lands off the \
                     reported pixels"
                );
                None
            }
            Pairing::Reverse => {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "a pass through a scaled render target over an unscaled depth surface of \
                     another size draws into the surface on the target's scaled grid: its depth \
                     lands off the reported pixels"
                );
                None
            }
        }
    }

    /// Allocate a plane for `surface`, retiring the one of another size or format it replaces.
    ///
    /// A refusal is remembered, so the next draws and clears over the same
    /// size and format do not ask again.
    fn allocate_depth_plane(
        &mut self,
        surface: &DepthSurface,
    ) -> Option<MetalHandle<MTLTextureKind>> {
        if self.depth_plane.refused_for(surface) {
            return None;
        }
        let (width, height) = surface.logical_size;
        let mut params = CreateDepthTextureParams {
            device_handle: self.device_handle,
            width,
            height,
            pixel_format: surface.format,
            sample_count: 1,
            texture_handle: MetalHandle::NULL,
        };
        let status = unix_call(&mut params);
        if status != 0 || params.texture_handle.is_null() {
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "depth plane: CreateDepthTexture({width}x{height}) failed (status={status:#x}) → \
                 the pass draws into the scaled depth surface"
            );
            self.depth_plane.note_refused(surface);
            return None;
        }
        if let Some(replaced) = self
            .depth_plane
            .replace_plane(params.texture_handle, surface)
        {
            self.retire_depth_target(replaced);
        }
        Some(params.texture_handle)
    }
}
