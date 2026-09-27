use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use log::trace;
use mtld3d_core::{
    encoder_data::{
        BindDepthOpFlags, BlitSide, ColorFillTarget, DepthTransfer, ResampledUpload,
        RetiredColorTarget, StretchSurfaceFlags,
    },
    encoder_packet::metadata::{
        BufferWarmupRecord, GammaRecord, LayerPacingRecord, StagingWarmupRecord,
    },
    encoder_records::{self as records, borrow, borrow_array, borrow_prefix},
    guest_queries::QueryLeaseCache,
    ids::{BufferId, TextureId},
    passes::ExtraColorSlot,
    render_scale::{RenderScale, TargetExtent},
};
use mtld3d_shared::{
    InPtr, MetalHandle, SetDisplaySyncEnabledParams, SetGammaRampParams,
    encoder_protocol::EncoderOpcode,
    encoder_wire::WireError,
    mtl::{DeviceCapsFlags, PixelFormat},
    mtl_handle::{CAMetalLayerKind, MTLTextureKind},
};

use super::{BLIT_TRACE_TARGET, FrameEncoder};
use crate::native_call;

/// Execute a borrowed control record while its packet retains every referenced owner.
///
/// Returns false for hot commands and shader adoption, which the frame replay state owns.
///
/// # Safety
/// The payload is an authentic immutable record from the matched producer. Every referenced
/// allocation and native handle remains valid through its last replay or submission use. Each
/// page, query and upload publication is adopted exactly once, and its PE owner remains retained
/// until native acknowledgment. Reply cells remain live and aligned through execution.
///
/// # Errors
/// Returns an error if the matched producer violates the command representation contract.
pub unsafe fn execute_command(
    enc: &mut FrameEncoder,
    opcode: &EncoderOpcode,
    operand: u16,
    payload: &[u8],
    queries: &mut QueryLeaseCache,
) -> Result<bool, WireError> {
    match opcode {
        EncoderOpcode::SetVsConstRange
        | EncoderOpcode::SetPsConstRange
        | EncoderOpcode::SetFfVsConstRange
        | EncoderOpcode::Draw
        | EncoderOpcode::SetSnapshot
        | EncoderOpcode::AdoptProgram
        | EncoderOpcode::RetainVbib => return Ok(false),
        EncoderOpcode::BindDepth => bind_depth(enc, operand, payload)?,
        EncoderOpcode::BindColor => bind_color(enc, operand, payload)?,
        _ => {
            if operand != 0 {
                return Err(WireError::InvalidValue);
            }
            execute_control(enc, opcode, payload, queries)?;
        }
    }
    Ok(true)
}

fn execute_control(
    enc: &mut FrameEncoder,
    opcode: &EncoderOpcode,
    payload: &[u8],
    queries: &mut QueryLeaseCache,
) -> Result<(), WireError> {
    match opcode {
        EncoderOpcode::WarmupTexture => {
            enc.get_or_create_texture_record(borrow::<records::TextureRecord>(payload)?)?;
        }
        EncoderOpcode::WarmupBuffer => {
            enc.warmup_buffer_record(borrow::<BufferWarmupRecord>(payload)?)?;
        }
        EncoderOpcode::WarmupStaging => {
            enc.warmup_staging_record(borrow::<StagingWarmupRecord>(payload)?)?;
        }
        EncoderOpcode::SetLayerPacing => {
            let r = borrow::<LayerPacingRecord>(payload)?;
            if r.display_sync > 1 {
                return Err(WireError::InvalidValue);
            }
            if r.layer != 0 {
                // SAFETY: the frame retains the layer named by this command through replay.
                let layer_handle = unsafe { MetalHandle::<CAMetalLayerKind>::new(r.layer) };
                let mut params = SetDisplaySyncEnabledParams {
                    layer_handle,
                    display_sync_enabled: r.display_sync,
                    max_fps: r.max_fps,
                };
                native_call(&mut params);
            }
        }
        EncoderOpcode::SetGamma => {
            let r = borrow::<GammaRecord>(payload)?;
            match r.mode {
                1 if r.entries_ptr == 0 && r.entries_len == 0 => {}
                2 if r.entries_len as usize == mtld3d_core::gamma::LUT_LANES => {
                    checked_address(
                        r.entries_ptr,
                        mtld3d_core::gamma::LUT_LANES * size_of::<u16>(),
                        align_of::<u16>(),
                    )?;
                }
                _ => return Err(WireError::InvalidValue),
            }
            if r.layer != 0 {
                // SAFETY: the frame retains the layer and gamma entries through this native call.
                let layer_handle = unsafe { MetalHandle::<CAMetalLayerKind>::new(r.layer) };
                let mut params = SetGammaRampParams {
                    layer_handle,
                    entries_ptr: r.entries_ptr,
                    entries_len: r.entries_len,
                    pad0: 0,
                };
                native_call(&mut params);
            }
        }
        EncoderOpcode::SetViewport => {
            let r = borrow::<records::SetViewportRecord>(payload)?;
            enc.set_viewport(r.x, r.y, r.width, r.height, r.min_z, r.max_z);
        }
        EncoderOpcode::SetVertexSampler => {
            let r = borrow::<records::SetVertexSamplerRecord>(payload)?;
            enc.set_vertex_sampler_binding(r.slot as usize, r.state);
        }
        EncoderOpcode::SetVertexTexture => {
            let r = borrow::<records::SetVertexTextureRecord>(payload)?;
            enc.set_vertex_texture_binding(
                r.slot as usize,
                (r.present != 0).then(|| TextureId::from_raw(r.id)),
            );
        }
        EncoderOpcode::GenerateMipmapsOrdered => {
            enc.run_generate_mipmaps_ordered(texture_id(payload)?);
        }
        EncoderOpcode::GenerateMipmaps => enc.run_generate_mipmaps(texture_id(payload)?),
        EncoderOpcode::UnbindExtraColor => enc.set_extra_color_render_target(
            borrow::<records::SlotRecord>(payload)?.value as usize,
            None,
        ),
        EncoderOpcode::DestroyTexture => enc.destroy_cached_texture(texture_id(payload)?),
        EncoderOpcode::ReadColorHandle
        | EncoderOpcode::ReadTextureColorHandle
        | EncoderOpcode::ReadTextureHandle => {
            let r = borrow::<records::ReadHandleRecord>(payload)?;
            let handle = enc.get_texture_handle_by_id(TextureId::from_raw(r.id));
            if !matches!(opcode, EncoderOpcode::ReadTextureHandle) {
                enc.note_color_read_back(texture_handle(handle));
            }
            store_reply_u64(r.reply, handle)?;
        }
        EncoderOpcode::NoteColorRead => {
            enc.note_color_read_back(texture_handle(borrow::<records::IdRecord>(payload)?.id));
        }
        EncoderOpcode::ResolveDepthSurface => {
            let r = borrow::<records::DepthTransferRecord>(payload)?;
            enc.resolve_depth_surface(&DepthTransfer {
                source: texture_handle(r.source),
                destination: texture_handle(r.destination),
                source_level: r.source_level,
                source_size: (r.source_width, r.source_height),
                source_format: pixel_format(r.source_format)?,
                source_samples: byte(r.source_samples)?,
                destination_size: (r.destination_width, r.destination_height),
                destination_format: pixel_format(r.destination_format)?,
            });
        }
        EncoderOpcode::StretchBlit => {
            let r = borrow::<records::StretchBlitRecord>(payload)?;
            let source = surface_handle(enc, &r.source.identity)?;
            let destination = surface_handle(enc, &r.destination.identity)?;
            emit_stretch_rect_blit(
                enc,
                &r.source,
                &r.destination,
                &StretchBlitParams {
                    source,
                    destination,
                    src_region: region(r.source_region),
                    dst_region: region(r.destination_region),
                    mip_level: r.mip_level,
                    render_quad: r.render_quad != 0,
                    filter: r.filter,
                },
            );
        }
        EncoderOpcode::ColorFill => {
            let r = borrow::<records::ColorFillRecord>(payload)?;
            let texture = texture_handle(surface_handle(enc, &r.identity)?);
            enc.color_fill_target(&ColorFillTarget {
                texture,
                logical_size: (r.logical_width, r.logical_height),
                texture_size: (r.texture_width, r.texture_height),
                format: pixel_format(r.format)?,
                scale: RenderScale::from_percent(r.scale),
                subresource: (r.slice, r.level),
                rect: (r.rect[0], r.rect[1], r.rect[2], r.rect[3]),
                rgba: (r.rgba[0], r.rgba[1], r.rgba[2], r.rgba[3]),
                msaa: texture_handle(r.msaa),
                msaa_srgb: texture_handle(r.msaa_srgb),
                sample_count: byte(r.sample_count)?,
                regenerate_mipmaps: r.regenerate_mipmaps != 0,
            });
        }
        EncoderOpcode::CarryDepth => {
            let r = borrow::<records::CarryDepthRecord>(payload)?;
            enc.carry_depth_contents(
                TextureId::from_raw(r.previous),
                TextureId::from_raw(r.current),
                r.width,
                r.height,
            );
        }
        EncoderOpcode::ClearColor => {
            let r = borrow::<records::ClearColorRecord>(payload)?;
            enc.clear_color_bounded_to_viewport(
                r.rgba[0],
                r.rgba[1],
                r.rgba[2],
                r.rgba[3],
                r.srgb_write != 0,
            );
        }
        EncoderOpcode::ClearColorRects => {
            let (r, tail) = borrow_prefix::<records::ClearColorRecord>(payload)?;
            enc.clear_color_rects(
                r.rgba[0],
                r.rgba[1],
                r.rgba[2],
                r.rgba[3],
                r.srgb_write != 0,
                &borrow_array::<records::RectRecord>(tail)?.iter().map(rect),
            );
        }
        EncoderOpcode::ClearDepthStencilRects => {
            let (r, tail) = borrow_prefix::<records::ClearDepthStencilRecord>(payload)?;
            enc.clear_depth_stencil_rects(
                (r.present & 1 != 0).then_some(r.depth),
                (r.present & 2 != 0).then_some(r.stencil),
                borrow_array::<records::RectRecord>(tail)?.iter().map(rect),
            );
        }
        EncoderOpcode::ClearDepthStencil => {
            let r = borrow::<records::ClearDepthStencilRecord>(payload)?;
            enc.clear_depth_stencil_bounded_to_viewport(
                (r.present & 1 != 0).then_some(r.depth),
                (r.present & 2 != 0).then_some(r.stencil),
            );
        }
        EncoderOpcode::ResolveDynamicDepth => {
            let r = borrow::<records::ResolveDynamicDepthRecord>(payload)?;
            enc.resolve_dynamic_depth_record(
                enc.get_texture_handle_by_id(TextureId::from_raw(r.id)),
                &r.texture,
            )?;
        }
        EncoderOpcode::ResolveDepthTexture => {
            let r = borrow::<records::ResolveDepthTextureRecord>(payload)?;
            enc.resolve_depth_to_texture(
                enc.get_texture_handle_by_id(TextureId::from_raw(r.id)),
                r.width,
                r.height,
                pixel_format(r.format)?,
            );
        }
        EncoderOpcode::ReadDeviceBuffer => {
            let r = borrow::<records::ReadDeviceBufferRecord>(payload)?;
            store_reply_u32(
                r.reply,
                u32::from(enc.readback_device_buffer(
                    BufferId::from_raw(r.id),
                    r.destination,
                    r.length,
                )),
            )?;
        }
        EncoderOpcode::BeginVisibility | EncoderOpcode::EndVisibility => {
            let r = borrow::<records::QueryRecord>(payload)?;
            // SAFETY: the frame publishes this descriptor exactly once and retains its mailbox.
            let core = unsafe { queries.adopt(&r.descriptor)? };
            if matches!(opcode, EncoderOpcode::BeginVisibility) {
                enc.begin_visibility_query(&core, r.generation);
            } else {
                enc.end_visibility_query(core, r.generation);
            }
        }
        EncoderOpcode::RetireColor => {
            let r = borrow::<records::RetireColorRecord>(payload)?;
            enc.retire_color_target(&RetiredColorTarget {
                base: texture_handle(r.base),
                srgb: texture_handle(r.srgb),
                msaa: texture_handle(r.msaa),
                msaa_srgb: texture_handle(r.msaa_srgb),
            });
        }
        EncoderOpcode::RetireDepth => {
            enc.retire_depth_target(texture_handle(borrow::<records::IdRecord>(payload)?.id));
        }
        EncoderOpcode::UploadColor => {
            let r = borrow::<records::UploadColorRecord>(payload)?;
            enc.upload_bytes_to_color_handle(
                r.handle,
                retained_bytes(&r.bytes)?,
                r.width,
                r.height,
                r.stride,
            );
        }
        EncoderOpcode::UploadResampled => {
            let r = borrow::<records::UploadResampledRecord>(payload)?;
            let t = &r.target;
            enc.upload_bytes_resampled(
                &ResampledUpload {
                    color_handle: t.handle,
                    format: pixel_format(t.format)?,
                    logical: (t.logical_width, t.logical_height),
                    texture: (t.texture_width, t.texture_height),
                    bytes_per_row: t.stride,
                    msaa: texture_handle(t.msaa),
                    msaa_srgb: texture_handle(t.msaa_srgb),
                    sample_count: byte(t.sample_count)?,
                },
                retained_bytes(&r.bytes)?,
            );
        }
        EncoderOpcode::UploadTexture | EncoderOpcode::UploadTextureAndMips => {
            let record = borrow::<records::TextureUploadRecord>(payload)?;
            // SAFETY: execute_command requires an authentic uniquely adopted upload record,
            // with both PE owners retained until the native upload and recovery readers retire.
            unsafe { enc.run_texture_record_upload(record)? };
        }
        EncoderOpcode::SetDumpDraw => {
            enc.set_dump_draw(borrow::<records::SlotRecord>(payload)?.value);
        }
        EncoderOpcode::StageUpload => {
            let r = borrow::<records::StageUploadRecord>(payload)?;
            // SAFETY: the PE lease retains this unique snapshot through upload completion and retries.
            let page = unsafe { r.page.adopt()? };
            enc.apply_stage_upload(
                BufferId::from_raw(r.id),
                page,
                u32::try_from(r.offset).map_err(|_| WireError::InvalidValue)?,
                u32::try_from(r.size).map_err(|_| WireError::InvalidValue)?,
            );
        }
        _ => return Err(WireError::InvalidValue),
    }
    Ok(())
}

fn texture_id(payload: &[u8]) -> Result<TextureId, WireError> {
    Ok(TextureId::from_raw(
        borrow::<records::IdRecord>(payload)?.id,
    ))
}
fn byte(value: u32) -> Result<u8, WireError> {
    u8::try_from(value).map_err(|_| WireError::InvalidValue)
}
fn pixel_format(value: u32) -> Result<PixelFormat, WireError> {
    PixelFormat::from_repr(value).ok_or(WireError::InvalidValue)
}
const fn texture_handle(value: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: the matched producer publishes retained texture handles or the null handle.
    unsafe { MetalHandle::new(value) }
}
const fn rect(r: &records::RectRecord) -> (i32, i32, i32, i32) {
    (r.x, r.y, r.right, r.bottom)
}
const fn region(r: [u32; 4]) -> mtld3d_core::stretch_rect::StretchRegion {
    mtld3d_core::stretch_rect::StretchRegion {
        x: r[0],
        y: r[1],
        w: r[2],
        h: r[3],
    }
}
fn surface_handle(
    enc: &mut FrameEncoder,
    r: &records::SurfaceIdentityRecord,
) -> Result<u64, WireError> {
    match r.kind {
        0 => enc.get_or_create_texture_record(&r.texture),
        1 | 2 => Ok(r.handle),
        _ => Err(WireError::InvalidValue),
    }
}
fn retained_bytes(span: &records::ByteSpan) -> Result<&[u8], WireError> {
    let length = usize::try_from(span.length).map_err(|_| WireError::InvalidValue)?;
    if length == 0 {
        return Ok(&[]);
    }
    let address = checked_address(span.address, length, 1)?;
    // SAFETY: frame admission retains the immutable payload allocation through submit replay.
    Ok(unsafe { std::slice::from_raw_parts(address as *const u8, length) })
}
fn checked_address(address: u64, length: usize, alignment: usize) -> Result<usize, WireError> {
    let address = usize::try_from(address).map_err(|_| WireError::InvalidValue)?;
    if address == 0
        || !address.is_multiple_of(alignment)
        || address.checked_add(length).is_none()
        || length > isize::MAX as usize
    {
        return Err(WireError::InvalidValue);
    }
    Ok(address)
}
fn store_reply_u64(address: u64, value: u64) -> Result<(), WireError> {
    let address = checked_address(address, size_of::<AtomicU64>(), align_of::<AtomicU64>())?;
    // SAFETY: the command's retained reply lease pins this initialized atomic through execution.
    let cell = unsafe { InPtr::<AtomicU64>::new(address as *const _) };
    cell.store(value, Ordering::Release);
    Ok(())
}
fn store_reply_u32(address: u64, value: u32) -> Result<(), WireError> {
    let address = checked_address(address, size_of::<AtomicU32>(), align_of::<AtomicU32>())?;
    // SAFETY: the command's retained reply lease pins this initialized atomic through execution.
    let cell = unsafe { InPtr::<AtomicU32>::new(address as *const _) };
    cell.store(value, Ordering::Release);
    Ok(())
}

/// Blit geometry + mode for [`emit_stretch_rect_blit`].
struct StretchBlitParams {
    source: u64,
    destination: u64,
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
/// [`records::SurfaceRecord`].
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
    info: &records::SurfaceRecord,
    region: mtld3d_core::stretch_rect::StretchRegion,
) -> mtld3d_core::stretch_rect::StretchRegion {
    if RenderScale::from_percent(info.scale).is_identity() {
        return region;
    }
    let extent = TargetExtent::new(
        RenderScale::from_percent(info.scale),
        (info.width, info.height),
        (info.texture_width, info.texture_height),
    );
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
    src_info: &records::SurfaceRecord,
    dst_info: &records::SurfaceRecord,
    params: &StretchBlitParams,
) {
    use mtld3d_shared::{BlitCommand, CopyTextureSubRectInfo};

    let &StretchBlitParams {
        source: src_handle,
        destination: dst_handle,
        src_region,
        dst_region,
        mip_level,
        render_quad,
        filter,
    } = params;
    // D3D9 resolves implicitly when a `StretchRect` reads a multisampled
    // surface. The blit runs after the passes recorded so far, so the last of
    // them that rendered into the multisampled companion takes the resolve;
    // for a single-sampled source this finds nothing and does nothing.
    // A `Clear` still waiting for a pass is one of those passes: D3D9 ordered
    // it before the copy, so it becomes a pass first and takes the resolve,
    // rather than an older pass handing the copy pre-clear content.
    if src_info.msaa != 0 {
        enc.flush_pending_clears();
    }
    // SAFETY: `src_handle` came from the encoder's texture cache or from a
    // surface's retained handle, both of which are `MTLTexture` handles.
    let src_texture = unsafe { MetalHandle::<MTLTextureKind>::new(src_handle) };
    enc.note_msaa_read(src_texture);
    // A source with a multisampled companion is a resolve target, and a
    // resolve the last submission stored into it must have completed before
    // this copy reads it on a device that does not order that itself.
    if src_info.msaa != 0 {
        enc.wait_for_resolve_retire();
    }
    // What the copy writes may be kept for good, and a draw left out of the
    // source would then be baked into it. A copy over a whole colour target
    // rebuilds that target as a clear does.
    // SAFETY: `dst_handle` came from the encoder's texture cache or from a
    // surface's retained handle, both of which are `MTLTexture` handles.
    let dst_texture = unsafe { MetalHandle::<MTLTextureKind>::new(dst_handle) };
    enc.note_stretch_copy(&crate::encoder::StretchCopyTargets {
        src: src_texture,
        dst: dst_texture,
        dst_subresource: surface_slice(dst_info).unwrap_or(0) | (dst_info.mip_level << 16),
        whole_color_dst: dst_info.identity.kind != 2
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
    let (src_dims, dst_dims) = (
        (src_info.texture_width, src_info.texture_height),
        (dst_info.texture_width, dst_info.texture_height),
    );

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
                src_slice: surface_slice(src_info),
                dst_dims,
                render_quad,
                filter,
            },
        );
        return;
    }
    if render_quad
        && !StretchSurfaceFlags::from_bits_retain(
            u8::try_from(dst_info.flags).expect("surface flags fit u8"),
        )
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
                src_slice: surface_slice(src_info).unwrap_or(0),
                dst_slice: surface_slice(dst_info).unwrap_or(0),
                region_w,
                region_h,
            },
        ));
        if dst_info.present & 2 != 0 {
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
                slice: surface_slice(src_info),
                msaa: MetalHandle::NULL,
                msaa_srgb: MetalHandle::NULL,
                sample_count: 1,
            },
            &BlitSide {
                handle: dst_handle,
                rect: dst_region,
                dims: dst_dims,
                mip: dst_info.mip_level,
                slice: surface_slice(dst_info),
                msaa: texture_handle(dst_info.msaa),
                msaa_srgb: texture_handle(dst_info.msaa_srgb),
                sample_count: u8::try_from(dst_info.sample_count)
                    .expect("typed producer sample count fits u8"),
            },
            dst_format,
            mtld3d_core::stretch_rect::blit_decode(src_info.format),
            filter,
        );
        if dst_info.present & 2 != 0 {
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
            src_slice: surface_slice(src_info).unwrap_or(0),
            dst_slice: surface_slice(dst_info).unwrap_or(0),
            region_w: src_region.w,
            region_h: src_region.h,
        },
    ));
    // A StretchRect into an autogen texture's level 0 regenerates the mip chain.
    // It MUST run after the copy and in the SAME blit stream , the encoder's
    // leading `frame_blit_commands` (used by `run_generate_mipmaps`) would
    // execute before this copy and regenerate from an empty level 0 → black.
    if dst_info.present & 2 != 0 {
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
    dst_info: &records::SurfaceRecord,
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
    let dst_face = surface_slice(dst_info).unwrap_or(0);
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
        if dst_info.present & 2 != 0 {
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
        && !StretchSurfaceFlags::from_bits_retain(
            u8::try_from(dst_info.flags).expect("surface flags fit u8"),
        )
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
                slice: surface_slice(dst_info),
                msaa: texture_handle(dst_info.msaa),
                msaa_srgb: texture_handle(dst_info.msaa_srgb),
                sample_count: u8::try_from(dst_info.sample_count)
                    .expect("typed producer sample count fits u8"),
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
    if dst_info.present & 2 != 0 {
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

fn surface_slice(info: &records::SurfaceRecord) -> Option<u32> {
    (info.present & 1 != 0).then_some(info.slice)
}

fn bind_depth(enc: &mut FrameEncoder, operand: u16, payload: &[u8]) -> Result<(), WireError> {
    let (handle, level, width, height, format, scale, samples, flags) = match operand {
        0 => {
            let r = borrow::<records::BindDepthNoneRecord>(payload)?;
            (
                0,
                0,
                0,
                0,
                PixelFormat::Depth32Float,
                0,
                r.sample_count,
                r.flags,
            )
        }
        1 => {
            let r = borrow::<records::BindDepthEagerRecord>(payload)?;
            let format = if BindDepthOpFlags::from_bits_retain(byte(r.flags)?)
                .contains(BindDepthOpFlags::HAS_STENCIL)
            {
                PixelFormat::Depth32FloatStencil8
            } else {
                PixelFormat::Depth32Float
            };
            (
                r.handle,
                0,
                r.width,
                r.height,
                format,
                r.scale,
                r.sample_count,
                r.flags,
            )
        }
        2 => {
            let r = borrow::<records::BindDepthLazyRecord>(payload)?;
            (
                enc.get_or_create_texture_record(&r.texture)?,
                r.level,
                (r.texture.width >> r.level).max(1),
                (r.texture.height >> r.level).max(1),
                r.texture.format()?,
                r.scale,
                r.sample_count,
                r.flags,
            )
        }
        _ => return Err(WireError::InvalidValue),
    };
    let flags = BindDepthOpFlags::from_bits(byte(flags)?).ok_or(WireError::InvalidValue)?;
    enc.set_depth_attachment_desc(width, height, format);
    enc.set_depth_stencil_attachment_level(
        texture_handle(handle),
        level,
        (width, height),
        flags.contains(BindDepthOpFlags::SAMPLEABLE),
        flags.contains(BindDepthOpFlags::HAS_STENCIL),
    );
    enc.set_depth_sample_count(byte(samples)?);
    enc.set_depth_unscaled(scale == 100);
    Ok(())
}

fn bind_color(enc: &mut FrameEncoder, operand: u16, payload: &[u8]) -> Result<(), WireError> {
    let (
        handle,
        msaa,
        msaa_srgb,
        sample_count,
        extent,
        format,
        has_alpha,
        slice,
        level,
        slot,
        scale,
    ) = match operand {
        0 => {
            let r = borrow::<records::BindColorBackbufferRecord>(payload)?;
            let scale = RenderScale::from_percent(r.scale);
            (
                r.handle,
                r.msaa,
                r.msaa_srgb,
                byte(r.sample_count)?,
                TargetExtent::whole(scale, (r.width, r.height)),
                PixelFormat::Bgra8Unorm,
                true,
                0,
                0,
                r.slot,
                scale,
            )
        }
        1 => {
            let r = borrow::<records::BindColorStandaloneRecord>(payload)?;
            enc.register_srgb_twin(texture_handle(r.srgb), texture_handle(r.handle));
            let scale = RenderScale::from_percent(r.scale);
            (
                r.handle,
                r.msaa,
                r.msaa_srgb,
                byte(r.sample_count)?,
                TargetExtent::whole(scale, (r.width, r.height)),
                pixel_format(r.format)?,
                r.has_alpha != 0,
                0,
                0,
                r.slot,
                scale,
            )
        }
        2 => {
            let r = borrow::<records::BindColorTextureRecord>(payload)?;
            let scale = RenderScale::from_percent(r.scale);
            (
                enc.get_or_create_texture_record(&r.texture)?,
                0,
                0,
                1,
                TargetExtent::mip_level(
                    scale,
                    (r.width, r.height),
                    (r.texture.width, r.texture.height),
                    r.level,
                ),
                r.texture.format()?,
                r.has_alpha != 0,
                r.slice,
                r.level,
                r.slot,
                scale,
            )
        }
        _ => return Err(WireError::InvalidValue),
    };
    if slot != 0 {
        enc.set_extra_color_render_target(
            slot as usize,
            Some(ExtraColorSlot {
                texture: texture_handle(handle),
                msaa_texture: texture_handle(msaa),
                msaa_srgb_texture: texture_handle(msaa_srgb),
                sample_count,
                subresource: slice | (level << 16),
                size: extent.texture(),
                logical_size: extent.logical(),
                format,
                scale,
                has_alpha,
            }),
        );
    } else {
        enc.set_color_render_target(&super::ColorRtBinding {
            texture: texture_handle(handle),
            msaa_texture: texture_handle(msaa),
            msaa_srgb_texture: texture_handle(msaa_srgb),
            sample_count,
            logical_size: extent.logical(),
            size: extent.texture(),
            format,
            has_alpha,
            scale,
            subresource: (slice, level),
        });
    }
    Ok(())
}
