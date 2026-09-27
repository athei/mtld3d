use std::sync::atomic::Ordering;

use log::trace;
use mtld3d_core::{
    encoder_data::{
        BeginVisibilityOp, BindColorOp, BindDepthOp, BindDepthOpFlags, BlitSide, CarryDepthOp,
        ClearColorOp, ClearColorRectsOp, ClearDepthStencilOp, ClearDepthStencilRectsOp,
        ColorFillOp, ColorFillTarget, DepthBinding, DestroyTextureOp, EndVisibilityOp,
        GenerateMipmapsOp, GenerateMipmapsOrderedOp, NoteColorReadOp, ReadColorHandleOp,
        ReadDeviceBufferOp, ReadTextureColorHandleOp, ReadTextureHandleOp, RegisterProgramOp,
        ResolveDepthSurfaceOp, ResolveDepthTextureOp, ResolveDynamicDepthOp, RetireColorOp,
        RetireDepthOp, RtBinding, SetDumpDrawOp, SetVertexSamplerOp, SetVertexTextureOp,
        SetViewportOp, StretchBlitOp, StretchKind, StretchSurfaceFlags, StretchSurfaceInfo,
        UnbindExtraColorOp, UploadColorOp, UploadResampledOp, UploadTextureAndMipsOp,
        UploadTextureOp, UploadTextureOpFlags,
    },
    passes::ExtraColorSlot,
    render_scale::TargetExtent,
};
use mtld3d_shared::{MetalHandle, mtl::DeviceCapsFlags, mtl_handle::MTLTextureKind};

use super::{BLIT_TRACE_TARGET, FrameEncoder};

pub trait ExecuteOp {
    fn execute(self, enc: &mut FrameEncoder);
}

impl ExecuteOp for SetViewportOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            x,
            y,
            width,
            height,
            min_z,
            max_z,
        } = self;
        enc.set_viewport(x, y, width, height, min_z, max_z);
    }
}

impl ExecuteOp for SetVertexSamplerOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { slot, state } = self;
        let slot = usize::from(slot);
        enc.set_vertex_sampler_binding(slot, state);
    }
}

impl ExecuteOp for SetVertexTextureOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { slot, id } = self;
        let slot = usize::from(slot);
        enc.set_vertex_texture_binding(slot, id);
    }
}

impl ExecuteOp for BindDepthOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            binding,
            sample_count,
            flags,
        } = self;
        let is_sampleable = flags.contains(BindDepthOpFlags::SAMPLEABLE);
        let depth_has_stencil = flags.contains(BindDepthOpFlags::HAS_STENCIL);
        let (depth_texture, level, desc, unscaled) = match binding {
            DepthBinding::None => (
                MetalHandle::NULL,
                0,
                (0, 0, mtld3d_shared::mtl::PixelFormat::Depth32Float),
                false,
            ),
            DepthBinding::Eager(h, (w, hgt), scale) => {
                let format = if depth_has_stencil {
                    mtld3d_shared::mtl::PixelFormat::Depth32FloatStencil8
                } else {
                    mtld3d_shared::mtl::PixelFormat::Depth32Float
                };
                (h, 0, (w, hgt, format), scale.is_identity())
            }
            DepthBinding::Lazy(info, level, scale) => {
                // SAFETY: `get_or_create_texture` returns a Metal texture
                // handle from the typed `texture_cache` via `.raw()`.
                let handle =
                    unsafe { MetalHandle::<MTLTextureKind>::new(enc.get_or_create_texture(&info)) };
                let desc = (
                    (info.width >> level).max(1),
                    (info.height >> level).max(1),
                    info.pixel_format,
                );
                (handle, level, desc, scale.is_identity())
            }
        };
        enc.set_depth_attachment_desc(desc.0, desc.1, desc.2);
        enc.set_depth_stencil_attachment_level(
            depth_texture,
            level,
            (desc.0, desc.1),
            is_sampleable,
            depth_has_stencil,
        );
        // In lockstep with the bind, which resets the count: a depth
        // surface that disagrees with render target 0 is dropped at pass
        // open rather than handed to Metal.
        enc.set_depth_sample_count(sample_count);
        // In lockstep too: the bind clears it, and only an unscaled depth
        // surface may set a pass's extent in place of render target 0.
        enc.set_depth_unscaled(unscaled);
    }
}

impl ExecuteOp for BindColorOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { slot, info, scale } = self;
        let slot = usize::from(slot);
        let (handle, msaa, msaa_srgb, sample_count, extent, fmt, has_alpha, slice, level) =
            match info {
                RtBinding::Backbuffer {
                    handle,
                    msaa,
                    msaa_srgb,
                    sample_count,
                    width,
                    height,
                } => (
                    handle,
                    msaa,
                    msaa_srgb,
                    sample_count,
                    TargetExtent::whole(scale, (width, height)),
                    mtld3d_shared::mtl::PixelFormat::Bgra8Unorm,
                    // The backbuffer is an alpha-bearing A8R8G8B8 target
                    // (see `PassState::reset_frame`), so its destination-alpha
                    // blend factors resolve unclamped.
                    true,
                    0,
                    0,
                ),
                RtBinding::StandaloneColor {
                    handle,
                    srgb,
                    msaa,
                    msaa_srgb,
                    sample_count,
                    format,
                    has_alpha,
                    width,
                    height,
                } => {
                    enc.register_srgb_twin(srgb, handle);
                    (
                        handle,
                        msaa,
                        msaa_srgb,
                        sample_count,
                        TargetExtent::whole(scale, (width, height)),
                        format,
                        has_alpha,
                        0,
                        0,
                    )
                }
                RtBinding::Texture {
                    info,
                    has_alpha,
                    width,
                    height,
                    slice,
                    level,
                } => {
                    let fmt = info.pixel_format;
                    // `info` measures the base level in render texels, the
                    // extent the Metal texture was created at; the level
                    // bound is Metal's own halving of that.
                    let extent = TargetExtent::mip_level(
                        scale,
                        (width, height),
                        (info.width, info.height),
                        level,
                    );
                    let h = enc.get_or_create_texture(&info);
                    // SAFETY: `get_or_create_texture` returns a Metal texture
                    // handle from the encoder's typed `texture_cache` via `.raw()`.
                    (
                        unsafe { MetalHandle::<MTLTextureKind>::new(h) },
                        // D3D9 has no multisampled texture: only a surface
                        // from `CreateRenderTarget` or the swap chain can
                        // carry samples, so a texture-backed bind is always
                        // single-sampled.
                        MetalHandle::NULL,
                        MetalHandle::NULL,
                        1,
                        extent,
                        fmt,
                        has_alpha,
                        slice,
                        level,
                    )
                }
            };
        if slot != 0 {
            enc.set_extra_color_render_target(
                slot,
                Some(ExtraColorSlot {
                    texture: handle,
                    msaa_texture: msaa,
                    msaa_srgb_texture: msaa_srgb,
                    sample_count,
                    subresource: slice | (level << 16),
                    size: extent.texture(),
                    logical_size: extent.logical(),
                    format: fmt,
                    scale,
                    has_alpha,
                }),
            );
        } else {
            enc.set_color_render_target(&crate::encoder::ColorRtBinding {
                texture: handle,
                msaa_texture: msaa,
                msaa_srgb_texture: msaa_srgb,
                sample_count,
                logical_size: extent.logical(),
                size: extent.texture(),
                format: fmt,
                has_alpha,
                scale,
                subresource: (slice, level),
            });
        }
    }
}

impl ExecuteOp for GenerateMipmapsOrderedOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { old_id } = self;
        enc.run_generate_mipmaps_ordered(old_id);
    }
}

impl ExecuteOp for UnbindExtraColorOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { slot } = self;
        let slot = usize::from(slot);
        enc.set_extra_color_render_target(slot, None);
    }
}

impl ExecuteOp for DestroyTextureOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { tex_id } = self;
        enc.destroy_cached_texture(tex_id);
    }
}

impl ExecuteOp for ReadColorHandleOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            texture_id,
            slot_op,
        } = self;
        let h = enc.get_texture_handle_by_id(texture_id);
        if h != 0 {
            // SAFETY: `h` is a live retained MTLTexture handle from the
            // encoder texture cache.
            enc.note_color_read_back(unsafe { MetalHandle::new(h) });
        }
        slot_op.store(h, std::sync::atomic::Ordering::Release);
    }
}

impl ExecuteOp for NoteColorReadOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { src } = self;
        enc.note_color_read_back(src);
    }
}

impl ExecuteOp for ResolveDepthSurfaceOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { transfer } = self;
        enc.resolve_depth_surface(&transfer);
    }
}

impl ExecuteOp for StretchBlitOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            src_info,
            dst_info,
            src_region,
            dst_region,
            mip_level,
            render_quad,
            filter,
        } = self;
        emit_stretch_rect_blit(
            enc,
            &src_info,
            &dst_info,
            &StretchBlitParams {
                src_region,
                dst_region,
                mip_level,
                render_quad,
                filter,
            },
        );
    }
}

impl ExecuteOp for ColorFillOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { kind, fill } = self;
        let texture = match kind {
            // SAFETY: `get_or_create_texture` returns a Metal texture handle
            // from the encoder's typed `texture_cache` via `.raw()`.
            StretchKind::Texture(ti) => unsafe {
                MetalHandle::<MTLTextureKind>::new(enc.get_or_create_texture(&ti))
            },
            // A depth-stencil surface never reaches here (`device_color_fill`
            // rejects it), so both arms carry the colour handle.
            StretchKind::Backbuffer(handle) | StretchKind::DepthStencil(handle) => handle,
        };
        enc.color_fill_target(&ColorFillTarget { texture, ..fill });
    }
}

impl ExecuteOp for CarryDepthOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            prev_id,
            cur_id,
            mip_w,
            mip_h,
        } = self;
        enc.carry_depth_contents(prev_id, cur_id, mip_w, mip_h);
    }
}

impl ExecuteOp for ClearColorOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            r_bits,
            g_bits,
            b_bits,
            a_bits,
            srgb_write,
        } = self;
        enc.clear_color_bounded_to_viewport(r_bits, g_bits, b_bits, a_bits, srgb_write);
    }
}

impl ExecuteOp for ClearColorRectsOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            r_bits,
            g_bits,
            b_bits,
            a_bits,
            srgb_write,
            rects,
        } = self;
        enc.clear_color_rects(r_bits, g_bits, b_bits, a_bits, srgb_write, &rects);
    }
}

impl ExecuteOp for ClearDepthStencilRectsOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            depth,
            stencil,
            list,
        } = self;
        enc.clear_depth_stencil_rects(depth, stencil, &list);
    }
}

impl ExecuteOp for ClearDepthStencilOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { depth, stencil } = self;
        enc.clear_depth_stencil_bounded_to_viewport(depth, stencil);
    }
}

impl ExecuteOp for ResolveDynamicDepthOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { id, info } = self;
        let dst = enc.get_texture_handle_by_id(id);
        enc.resolve_dynamic_depth(dst, &info);
    }
}

impl ExecuteOp for ResolveDepthTextureOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { id, w, h, format } = self;
        let dst = enc.get_texture_handle_by_id(id);
        enc.resolve_depth_to_texture(dst, w, h, format);
    }
}

impl ExecuteOp for ReadDeviceBufferOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            done,
            buffer_id,
            dst_ptr,
            dst_len,
        } = self;
        done.store(
            enc.readback_device_buffer(buffer_id, dst_ptr, dst_len),
            Ordering::Release,
        );
    }
}

impl ExecuteOp for RegisterProgramOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { shader_id, program } = self;
        enc.register_program(shader_id, program);
    }
}

impl ExecuteOp for BeginVisibilityOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { c, generation } = self;
        enc.begin_visibility_query(&c, generation);
    }
}

impl ExecuteOp for EndVisibilityOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { core, generation } = self;
        enc.end_visibility_query(core, generation);
    }
}

impl ExecuteOp for RetireColorOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { retired } = self;
        enc.retire_color_target(&retired);
    }
}

impl ExecuteOp for RetireDepthOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { depth } = self;
        enc.retire_depth_target(depth);
    }
}

impl ExecuteOp for UploadColorOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            color_handle,
            bytes,
            width,
            height,
            src_stride,
        } = self;
        enc.upload_bytes_to_color_handle(color_handle, bytes.as_slice(), width, height, src_stride);
    }
}

impl ExecuteOp for UploadResampledOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { target, bytes } = self;
        enc.upload_bytes_resampled(&target, bytes.as_slice());
    }
}

impl ExecuteOp for ReadTextureHandleOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            texture_id,
            slot_op,
        } = self;
        slot_op.store(enc.get_texture_handle_by_id(texture_id), Ordering::Release);
    }
}

impl ExecuteOp for GenerateMipmapsOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { texture_id } = self;
        enc.run_generate_mipmaps(texture_id);
    }
}

impl ExecuteOp for ReadTextureColorHandleOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            texture_id,
            slot_op,
        } = self;
        let handle = enc.get_texture_handle_by_id(texture_id);
        slot_op.store(handle, Ordering::Release);
        // The store-action optimiser would drop the colour store of a pass
        // nothing samples in-frame, and the claim this read resolves is exactly
        // what such a pass wrote, so note the read before the flush decides.
        // SAFETY: `handle` is a live retained `MTLTexture` handle from the
        // encoder texture cache, or zero, which the note ignores.
        enc.note_color_read_back(unsafe { MetalHandle::<MTLTextureKind>::new(handle) });
    }
}

impl ExecuteOp for UploadTextureAndMipsOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self {
            job,
            texture_id,
            flags,
        } = self;
        let ordered = flags.contains(UploadTextureOpFlags::ORDERED);
        let regen_mipmaps = flags.contains(UploadTextureOpFlags::REGENERATE_MIPMAPS);
        if ordered {
            enc.run_ordered_texture_upload(job);
        } else {
            enc.run_texture_upload(job);
        }
        if regen_mipmaps {
            enc.run_generate_mipmaps(texture_id);
        }
    }
}

impl ExecuteOp for UploadTextureOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { job } = self;
        enc.run_texture_upload(job);
    }
}

impl ExecuteOp for SetDumpDrawOp {
    fn execute(self, enc: &mut FrameEncoder) {
        let Self { seq } = self;
        enc.set_dump_draw(seq);
    }
}

/// Blit geometry + mode for [`emit_stretch_rect_blit`].
struct StretchBlitParams {
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
    mip_level: u32,
    render_quad: bool,
    filter: u32,
}

/// Geometry for a `StretchRect` whose source and destination are one texture.
///
/// Regions and dimensions are already in the texture's own space; `src_mip` and
/// `src_slice` address the source subresource, while the destination level,
/// slice, format and surface class come from the accompanying
/// [`StretchSurfaceInfo`].
struct SameTextureBlitParams {
    handle: u64,
    src_region: mtld3d_core::stretch_rect::StretchRegion,
    dst_region: mtld3d_core::stretch_rect::StretchRegion,
    src_mip: u32,
    /// Array slice the source surface addresses, `None` for a single-slice texture.
    src_slice: Option<u32>,
    dst_dims: (u32, u32),
    render_quad: bool,
    filter: u32,
}

/// Convert a `StretchRect` region into the space of the texture it addresses.
///
/// A no-op for anything but a surface rasterized at a non-default
/// `render.scale`, and an exact identity at the default. A region spanning
/// the surface spans the subresource Metal allocated for it.
fn scale_stretch_region(
    info: &StretchSurfaceInfo,
    region: mtld3d_core::stretch_rect::StretchRegion,
) -> mtld3d_core::stretch_rect::StretchRegion {
    if info.scale.is_identity() {
        return region;
    }
    let extent = TargetExtent::new(info.scale, (info.width, info.height), info.texture_size);
    let (x, y, w, h) = extent.rect(region.x, region.y, region.w, region.h);
    mtld3d_core::stretch_rect::StretchRegion { x, y, w, h }
}

/// Encoder-thread body of `StretchRect`.
///
/// Resolves both endpoint handles via the texture cache, then either queues a
/// 1:1 sub-rect copy (same-size, same-format blit) or runs the render-quad path
/// (`render_quad` , sizes differ and/or formats differ; the destination is
/// guaranteed a render target by `device_stretch_rect`).
fn emit_stretch_rect_blit(
    enc: &mut FrameEncoder,
    src_info: &StretchSurfaceInfo,
    dst_info: &StretchSurfaceInfo,
    params: &StretchBlitParams,
) {
    use mtld3d_shared::{BlitCommand, CopyTextureSubRectInfo};

    let &StretchBlitParams {
        src_region,
        dst_region,
        mip_level,
        render_quad,
        filter,
    } = params;
    let src_handle = match &src_info.kind {
        StretchKind::Texture(info) => enc.get_or_create_texture(info),
        StretchKind::Backbuffer(h) | StretchKind::DepthStencil(h) => h.raw(),
    };
    // D3D9 resolves implicitly when a `StretchRect` reads a multisampled
    // surface. The blit runs after the passes recorded so far, so the last of
    // them that rendered into the multisampled companion takes the resolve;
    // for a single-sampled source this finds nothing and does nothing.
    // A `Clear` still waiting for a pass is one of those passes: D3D9 ordered
    // it before the copy, so it becomes a pass first and takes the resolve,
    // rather than an older pass handing the copy pre-clear content.
    if !src_info.msaa.is_null() {
        enc.flush_pending_clears();
    }
    // SAFETY: `src_handle` came from the encoder's texture cache or from a
    // surface's retained handle, both of which are `MTLTexture` handles.
    let src_texture = unsafe { MetalHandle::<MTLTextureKind>::new(src_handle) };
    enc.note_msaa_read(src_texture);
    // A source with a multisampled companion is a resolve target, and a
    // resolve the last submission stored into it must have completed before
    // this copy reads it on a device that does not order that itself.
    if !src_info.msaa.is_null() {
        enc.wait_for_resolve_retire();
    }
    let dst_handle = match &dst_info.kind {
        StretchKind::Texture(info) => enc.get_or_create_texture(info),
        StretchKind::Backbuffer(h) | StretchKind::DepthStencil(h) => h.raw(),
    };
    // What the copy writes may be kept for good, and a draw left out of the
    // source would then be baked into it. A copy over a whole colour target
    // rebuilds that target as a clear does.
    // SAFETY: `dst_handle` came from the encoder's texture cache or from a
    // surface's retained handle, both of which are `MTLTexture` handles.
    let dst_texture = unsafe { MetalHandle::<MTLTextureKind>::new(dst_handle) };
    enc.note_stretch_copy(&crate::encoder::StretchCopyTargets {
        src: src_texture,
        dst: dst_texture,
        dst_subresource: dst_info.slice.unwrap_or(0) | (dst_info.mip_level << 16),
        whole_color_dst: !matches!(dst_info.kind, StretchKind::DepthStencil(_))
            && dst_region.x == 0
            && dst_region.y == 0
            && dst_region.w == dst_info.width
            && dst_region.h == dst_info.height,
    });
    if src_handle == 0 || dst_handle == 0 {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: failed to resolve Metal texture (src={src_handle:#x}, dst={dst_handle:#x})"
        );
        return;
    }
    // `render.scale` shrinks the back buffer, so an endpoint that *is* the back
    // buffer has both its region and its extent converted; the ratio the blit
    // VS builds from the two is preserved, while the destination rect (which
    // drives an absolute viewport and scissor) lands on real pixels. An
    // endpoint the game created keeps its own coordinates.
    let src_region = scale_stretch_region(src_info, src_region);
    let dst_region = scale_stretch_region(dst_info, dst_region);
    let (src_dims, dst_dims) = (src_info.texture_size, dst_info.texture_size);

    // The API thread decided this from the game's own rects. Scaling only one
    // endpoint can turn a logically 1:1 copy into a physical resize, which the
    // blit encoder cannot do, so the transport choice is re-made here on the
    // sizes that actually reach Metal.
    // A multisampled destination has to go through the render quad whatever
    // the sizes: `MTLBlitCommandEncoder` cannot write a multisampled texture,
    // and the quad writes every sample of each pixel it covers, which is the
    // spread D3D9 defines for a copy into a multisampled surface.
    let render_quad = render_quad
        || dst_info.sample_count > 1
        || src_region.w != dst_region.w
        || src_region.h != dst_region.h;
    if src_handle == dst_handle {
        emit_same_texture_stretch(
            enc,
            dst_info,
            &SameTextureBlitParams {
                handle: src_handle,
                src_region,
                dst_region,
                src_mip: mip_level,
                src_slice: src_info.slice,
                dst_dims,
                render_quad,
                filter,
            },
        );
        return;
    }
    if render_quad
        && !dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_RENDER_TARGET)
    {
        // Only reachable with a non-default `render.scale`: the pair was 1:1
        // in the game's coordinates (so D3D9 accepted it against a
        // non-render-target destination) and only the back-buffer side shrank.
        // The render-quad path would have to bind a surface that cannot be a
        // colour attachment, so copy the overlapping region instead and say so
        // rather than silently corrupting the destination.
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: render.scale made a 1:1 copy into a non-render-target destination a \
             {}x{} → {}x{} resize, which a Metal blit cannot do; copying the overlap instead. \
             Set render.scale = 1.0 if this surface's contents matter",
            src_region.w, src_region.h, dst_region.w, dst_region.h,
        );
        enc.flush_pending_clears();
        enc.end_current_pass("stretch_rect");
        let region_w = src_region.w.min(dst_region.w);
        let region_h = src_region.h.min(dst_region.h);
        enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
            &CopyTextureSubRectInfo {
                src_texture: src_handle,
                dst_texture: dst_handle,
                mip_level,
                dst_mip_level: dst_info.mip_level,
                src_origin_x: src_region.x,
                src_origin_y: src_region.y,
                dst_origin_x: dst_region.x,
                dst_origin_y: dst_region.y,
                src_slice: src_info.slice.unwrap_or(0),
                dst_slice: dst_info.slice.unwrap_or(0),
                region_w,
                region_h,
            },
        ));
        if dst_info.autogen_texture_id.is_some() {
            enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(dst_handle));
        }
        return;
    }
    if render_quad {
        // Render-quad path (a size change and/or a format conversion): render
        // the source onto a quad covering the destination rect. The
        // destination's Metal colour format keys the blit pipeline and the pass
        // colour attachment; the source is sampled in its own format (a packed
        // YUV source is decoded to RGB by the fragment function), so this path
        // also converts a cross-format pair. `device_stretch_rect` guarantees
        // the destination is a render target here.
        // Device-aware: the pipeline's colour format must match the attachment
        // texture as created on this device (BGRA8 for an expanded 16-bit dst).
        let Some(dst_format) =
            map_for_encoder(dst_info.format, enc).map(|m| m.metal_pixel_format())
        else {
            mtld3d_shared::log_once_warn!(
                target: crate::LOG_TARGET,
                "StretchRect: scaling dst format 0x{:x} unmapped → drop",
                dst_info.format
            );
            return;
        };
        enc.stretch_blit_scaled(
            &BlitSide {
                handle: src_handle,
                rect: src_region,
                dims: src_dims,
                mip: src_info.mip_level,
                slice: src_info.slice,
                msaa: MetalHandle::NULL,
                msaa_srgb: MetalHandle::NULL,
                sample_count: 1,
            },
            &BlitSide {
                handle: dst_handle,
                rect: dst_region,
                dims: dst_dims,
                mip: dst_info.mip_level,
                slice: dst_info.slice,
                msaa: dst_info.msaa,
                msaa_srgb: dst_info.msaa_srgb,
                sample_count: dst_info.sample_count,
            },
            dst_format,
            mtld3d_core::stretch_rect::blit_decode(src_info.format),
            filter,
        );
        if dst_info.autogen_texture_id.is_some() {
            enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(dst_handle));
        }
        return;
    }
    // A `Clear` on either endpoint that is still waiting for a pass must land
    // before the copy: D3D9 ordered it first.
    enc.flush_pending_clears();
    enc.end_current_pass("stretch_rect");
    enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
        &CopyTextureSubRectInfo {
            src_texture: src_handle,
            dst_texture: dst_handle,
            mip_level,
            dst_mip_level: dst_info.mip_level,
            src_origin_x: src_region.x,
            src_origin_y: src_region.y,
            dst_origin_x: dst_region.x,
            dst_origin_y: dst_region.y,
            src_slice: src_info.slice.unwrap_or(0),
            dst_slice: dst_info.slice.unwrap_or(0),
            region_w: src_region.w,
            region_h: src_region.h,
        },
    ));
    // A StretchRect into an autogen texture's level 0 regenerates the mip chain.
    // It MUST run after the copy and in the SAME blit stream , the encoder's
    // leading `frame_blit_commands` (used by `run_generate_mipmaps`) would
    // execute before this copy and regenerate from an empty level 0 → black.
    if dst_info.autogen_texture_id.is_some() {
        enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(dst_handle));
    }
    trace!(
        target: BLIT_TRACE_TARGET,
        "StretchRect src={src_handle:#x} {sw}x{sh} src_rect={sx},{sy}+{rw}x{rh} \
         dst={dst_handle:#x} {dw}x{dh} dst_rect={dx},{dy}+{rw}x{rh} mip={mip_level}",
        sw = src_dims.0, sh = src_dims.1,
        sx = src_region.x, sy = src_region.y,
        dw = dst_dims.0, dh = dst_dims.1,
        dx = dst_region.x, dy = dst_region.y,
        rw = src_region.w, rh = src_region.h,
    );
}

/// Land a `Clear` still waiting for a pass, then close the pass, before a blit.
///
/// D3D9 ordered the clear first, so a copy queued ahead of it would either
/// read the pre-clear source or be wiped by the clear.
fn flush_clears_before_stretch(enc: &mut FrameEncoder) {
    enc.flush_pending_clears();
    enc.end_current_pass("stretch_rect");
}

/// Encoder-thread body of a `StretchRect` between two rects of one texture.
///
/// D3D9 performs the copy and reads the whole source region before writing any
/// of the destination, so an overlapping or scaled pair stages through a
/// scratch texture. Disjoint 1:1 rects, two mip levels and two cube faces
/// included, go straight through the blit encoder: Metal allows a copy inside a
/// single texture as long as the two subresource regions do not overlap.
fn emit_same_texture_stretch(
    enc: &mut FrameEncoder,
    dst_info: &StretchSurfaceInfo,
    params: &SameTextureBlitParams,
) {
    use mtld3d_core::stretch_rect::{SameSurfaceRoute, StretchRegion, same_surface_route};
    use mtld3d_shared::{BlitCommand, CopyTextureSubRectInfo};

    let &SameTextureBlitParams {
        handle,
        src_region,
        dst_region,
        src_mip,
        src_slice,
        dst_dims,
        render_quad,
        filter,
    } = params;
    let dst_mip = dst_info.mip_level;
    // A cube's faces are slices of the one texture, so the two endpoints can
    // name different faces of it; every other texture kind holds a single
    // slice and both sides read 0.
    let src_face = src_slice.unwrap_or(0);
    let dst_face = dst_info.slice.unwrap_or(0);
    let route = same_surface_route(src_region, dst_region, src_mip, dst_mip, src_face, dst_face);
    if route == SameSurfaceRoute::Skip {
        mtld3d_shared::log_once_info!(
            target: crate::LOG_TARGET,
            "StretchRect: source and destination name the same texels of one surface, \
             so the copy leaves it as it is"
        );
        return;
    }
    if route == SameSurfaceRoute::Direct {
        flush_clears_before_stretch(enc);
        enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
            &CopyTextureSubRectInfo {
                src_texture: handle,
                dst_texture: handle,
                mip_level: src_mip,
                dst_mip_level: dst_mip,
                src_origin_x: src_region.x,
                src_origin_y: src_region.y,
                dst_origin_x: dst_region.x,
                dst_origin_y: dst_region.y,
                src_slice: src_face,
                dst_slice: dst_face,
                region_w: src_region.w,
                region_h: src_region.h,
            },
        ));
        if dst_info.autogen_texture_id.is_some() {
            enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(handle));
        }
        return;
    }
    // Device-aware mapping: the scratch has to carry the Metal format the one
    // texture was actually created with, and the render quad keys its pipeline
    // and colour attachment off the same value.
    let Some(format) = map_for_encoder(dst_info.format, enc).map(|m| m.metal_pixel_format()) else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: format 0x{:x} unmapped → a copy inside that surface is dropped",
            dst_info.format
        );
        return;
    };
    if render_quad
        && !dst_info
            .flags
            .contains(StretchSurfaceFlags::IS_RENDER_TARGET)
    {
        // Only reachable under a non-default `render.scale` that rounds a
        // logically 1:1 pair to two different extents; the render quad would
        // have to bind a surface that cannot be a colour attachment.
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "StretchRect: a resizing copy inside one non-render-target surface has no Metal \
             path; the copy is dropped. Set render.scale = 1.0 if this surface's contents matter"
        );
        return;
    }
    let Some((scratch, scratch_w, scratch_h)) =
        enc.stretch_scratch_texture(handle, (src_region.w, src_region.h), format)
    else {
        return;
    };
    flush_clears_before_stretch(enc);
    enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
        &CopyTextureSubRectInfo {
            src_texture: handle,
            dst_texture: scratch,
            mip_level: src_mip,
            dst_mip_level: 0,
            src_origin_x: src_region.x,
            src_origin_y: src_region.y,
            dst_origin_x: 0,
            dst_origin_y: 0,
            src_slice: src_face,
            dst_slice: 0,
            region_w: src_region.w,
            region_h: src_region.h,
        },
    ));
    if render_quad {
        enc.stretch_blit_scaled(
            &BlitSide {
                handle: scratch,
                rect: StretchRegion {
                    x: 0,
                    y: 0,
                    w: src_region.w,
                    h: src_region.h,
                },
                dims: (scratch_w, scratch_h),
                mip: 0,
                slice: None,
                msaa: MetalHandle::NULL,
                msaa_srgb: MetalHandle::NULL,
                sample_count: 1,
            },
            &BlitSide {
                handle,
                rect: dst_region,
                dims: dst_dims,
                mip: dst_mip,
                slice: dst_info.slice,
                msaa: dst_info.msaa,
                msaa_srgb: dst_info.msaa_srgb,
                sample_count: dst_info.sample_count,
            },
            format,
            mtld3d_core::stretch_rect::blit_decode(dst_info.format),
            filter,
        );
    } else {
        enc.push_stretch_rect_blit(BlitCommand::copy_texture_to_texture_sub_rect(
            &CopyTextureSubRectInfo {
                src_texture: scratch,
                dst_texture: handle,
                mip_level: 0,
                dst_mip_level: dst_mip,
                src_origin_x: 0,
                src_origin_y: 0,
                dst_origin_x: dst_region.x,
                dst_origin_y: dst_region.y,
                src_slice: 0,
                dst_slice: dst_face,
                region_w: dst_region.w,
                region_h: dst_region.h,
            },
        ));
    }
    if dst_info.autogen_texture_id.is_some() {
        enc.push_stretch_rect_blit(BlitCommand::generate_mipmaps(handle));
    }
}
fn map_for_encoder(format: u32, enc: &FrameEncoder) -> Option<mtld3d_core::format::FormatMapping> {
    let native = enc
        .gpu_caps
        .device_caps
        .contains(DeviceCapsFlags::NATIVE_PACKED16)
        && !enc.config().expand_packed16;
    if !native {
        mtld3d_shared::log_once_info!(target: crate::LOG_TARGET,
            "packed 16-bit formats unavailable natively (forced={}): A4R4G4B4/R5G6B5/A1R5G5B5/X1R5G5B5 widen to BGRA8 in the GPU upload pass, 16-bit render targets are not advertised", enc.config().expand_packed16);
    }
    mtld3d_core::format::map_d3d_format_device(format, native)
}
