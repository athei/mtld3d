//! Direct construction of fixed control records in the retained command stream.

use mtld3d_shared::{encoder_protocol::EncoderOpcode, encoder_wire::WireError};

use super::FrameRecorder;
use crate::{
    encoder_data::{
        AdoptProgramOp, DepthBinding, DepthTransfer, Op, ResampledUpload, RtBinding, StretchKind,
        StretchSurfaceInfo, TextureUploadJob,
    },
    encoder_records::{
        BindColorBackbufferRecord, BindColorStandaloneRecord, BindColorTextureRecord,
        BindDepthEagerRecord, BindDepthLazyRecord, BindDepthNoneRecord, ByteSpan, CarryDepthRecord,
        ClearColorRecord, ClearDepthStencilRecord, ColorFillRecord, CommandRecord,
        DepthTransferRecord, IdRecord, QueryRecord, ReadDeviceBufferRecord, ReadHandleRecord,
        RectRecord, ResampledTargetRecord, ResolveDepthTextureRecord, ResolveDynamicDepthRecord,
        RetireColorRecord, SetVertexSamplerRecord, SetVertexTextureRecord, SetViewportRecord,
        SlotRecord, StageUploadRecord, StretchBlitRecord, SurfaceIdentityRecord, SurfaceRecord,
        TextureRecord, TextureUploadRecord, UploadColorRecord, UploadResampledRecord,
    },
    encoder_reply::{ReplyBool, ReplyU64},
    guest_pages::{GuestOwnedPageLease, GuestPageLease},
    guest_queries::GuestQueryLease,
    scratch::ScratchArena,
    upload_redirty::GuestRedirtyLease,
};

fn span(value: crate::draw_data::ScratchSlice) -> ByteSpan {
    let (address, length) = value.as_raw();
    ByteSpan {
        address,
        length: u64::from(length),
    }
}

const fn identity(value: &StretchKind) -> SurfaceIdentityRecord {
    match value {
        StretchKind::Texture(value) => SurfaceIdentityRecord {
            texture: TextureRecord::capture(value),
            handle: 0,
            kind: 0,
            reserved: 0,
        },
        StretchKind::Backbuffer(handle) => SurfaceIdentityRecord {
            texture: TextureRecord::ZERO,
            handle: handle.raw(),
            kind: 1,
            reserved: 0,
        },
        StretchKind::DepthStencil(handle) => SurfaceIdentityRecord {
            texture: TextureRecord::ZERO,
            handle: handle.raw(),
            kind: 2,
            reserved: 0,
        },
    }
}

fn surface(value: &StretchSurfaceInfo) -> SurfaceRecord {
    SurfaceRecord {
        identity: identity(&value.kind),
        autogen_id: value
            .autogen_texture_id
            .map_or(0, crate::ids::TextureId::raw),
        msaa: value.msaa.raw(),
        msaa_srgb: value.msaa_srgb.raw(),
        width: value.width,
        height: value.height,
        texture_width: value.texture_size.0,
        texture_height: value.texture_size.1,
        scale: value.scale.percent(),
        format: value.format,
        mip_level: value.mip_level,
        slice: value.slice.unwrap_or(0),
        pool: value.pool,
        flags: u32::from(value.flags.bits()),
        sample_count: u32::from(value.sample_count),
        present: u32::from(value.slice.is_some())
            | (u32::from(value.autogen_texture_id.is_some()) << 1),
    }
}

const fn region(value: &crate::stretch_rect::StretchRegion) -> [u32; 4] {
    [value.x, value.y, value.w, value.h]
}

fn depth(value: &DepthTransfer) -> DepthTransferRecord {
    DepthTransferRecord {
        source: value.source.raw(),
        destination: value.destination.raw(),
        source_level: value.source_level,
        source_width: value.source_size.0,
        source_height: value.source_size.1,
        source_format: value.source_format as u32,
        source_samples: u32::from(value.source_samples),
        destination_width: value.destination_size.0,
        destination_height: value.destination_size.1,
        destination_format: value.destination_format as u32,
    }
}

fn resampled(value: &ResampledUpload) -> ResampledTargetRecord {
    ResampledTargetRecord {
        handle: value.color_handle,
        msaa: value.msaa.raw(),
        msaa_srgb: value.msaa_srgb.raw(),
        format: value.format as u32,
        logical_width: value.logical.0,
        logical_height: value.logical.1,
        texture_width: value.texture.0,
        texture_height: value.texture.1,
        stride: value.bytes_per_row,
        sample_count: u32::from(value.sample_count),
        reserved: 0,
    }
}

fn clear_depth(depth: Option<u32>, stencil: Option<u32>) -> ClearDepthStencilRecord {
    ClearDepthStencilRecord {
        depth: depth.unwrap_or(0),
        stencil: stencil.unwrap_or(0),
        present: u32::from(depth.is_some()) | (u32::from(stencil.is_some()) << 1),
        reserved: 0,
    }
}

impl FrameRecorder {
    fn fixed<T: CommandRecord>(
        &mut self,
        scratch: &mut ScratchArena,
        tag: EncoderOpcode,
        operand: u16,
        build: impl FnOnce() -> T,
    ) -> Result<(), WireError> {
        self.slab.push_fixed_record(
            scratch,
            tag.into(),
            operand,
            size_of::<T>(),
            |destination| crate::encoder_records::write(destination, build()),
        )
    }

    fn rects<T: CommandRecord>(
        &mut self,
        scratch: &mut ScratchArena,
        tag: EncoderOpcode,
        rects: &[(i32, i32, i32, i32)],
        build: impl FnOnce() -> T,
    ) -> Result<(), WireError> {
        let size = rects
            .len()
            .checked_mul(size_of::<RectRecord>())
            .and_then(|v| v.checked_add(size_of::<T>()))
            .ok_or(WireError::TooLarge)?;
        self.slab
            .push_fixed_record(scratch, tag.into(), 0, size, |destination| {
                let (head, tail) = destination.split_at_mut(size_of::<T>());
                crate::encoder_records::write(head, build())?;
                for (bytes, &(x, y, right, bottom)) in tail
                    .as_chunks_mut::<{ size_of::<RectRecord>() }>()
                    .0
                    .iter_mut()
                    .zip(rects)
                {
                    crate::encoder_records::write(
                        bytes,
                        RectRecord {
                            x,
                            y,
                            right,
                            bottom,
                        },
                    )?;
                }
                Ok(())
            })
    }
}

fn capture_upload(
    job: TextureUploadJob,
    destination: &mut [u8],
    pages: &mut Vec<GuestPageLease>,
    redirties: &mut Vec<GuestRedirtyLease>,
    pool: &crate::guest_completions::CompletionPool,
    mip_texture: u64,
    mip_flags: u32,
) -> Result<(), WireError> {
    let declined = crate::upload_redirty::RedirtyEntry {
        subresource: job.redirty_subresource(),
        face: job.destination_slice,
        level: job.level,
        rect: job.redirty_rect(),
    };
    let emitted = job.emitted_answer();
    let lease = GuestRedirtyLease::new_pooled(job.redirty, declined, emitted, pool);
    let redirty = lease.descriptor();
    redirties.push(lease);
    let lease = GuestPageLease::for_read_pooled(job.staging, pool);
    let page = lease.descriptor();
    pages.push(lease);
    let staging_index = u32::try_from(job.staging_index).map_err(|_| WireError::TooLarge)?;
    crate::encoder_records::write(
        destination,
        TextureUploadRecord {
            texture: TextureRecord::capture(&job.info),
            page,
            redirty,
            mip_texture,
            level: job.level,
            destination_slice: job.destination_slice,
            staging_index,
            origin_x: job.origin_x,
            origin_y: job.origin_y,
            width: job.region_w,
            height: job.region_h,
            source_format: job.src_d3d_format,
            pitch: job.src_pitch,
            bytes_per_pixel: job.bytes_per_pixel,
            depth: job.depth,
            slice_pitch: job.slice_pitch,
            release_staging: u32::from(job.release_staging),
            upload_generation: job.upload_generation,
            mip_flags,
            reserved: 0,
        },
    )
}

/// A typed API capture that constructs its final command without an intermediate operation enum.
pub trait CaptureControl: Sized {
    /// A registration transferred only after successful command publication.
    fn registration(&self) -> Option<u64> {
        None
    }
    /// Capture the command and retain any transferred owners.
    ///
    /// # Errors
    /// Returns an allocation or invalid input error; retained owners remain on the rejected frame.
    fn capture(
        self,
        recorder: &mut FrameRecorder,
        scratch: &mut ScratchArena,
    ) -> Result<(), WireError>;
    /// Preserve an uncaptured owner on the failure-only operation ledger.
    fn into_rejected(self) -> Op;
}

macro_rules! capture_fixed {
    ($recorder:ident,$scratch:ident,$tag:ident,$value:expr) => {
        $recorder.fixed($scratch, $tag, 0, || $value)?
    };
    ($recorder:ident,$scratch:ident,$tag:ident,$operand:expr,$value:expr) => {
        $recorder.fixed($scratch, $tag, $operand, || $value)?
    };
}

macro_rules! capture_control {
    ($ty:ident, $variant:ident, $v:ident, $recorder:ident, $scratch:ident, $tag:ident, $body:block) => {
        impl CaptureControl for crate::encoder_data::$ty {
            fn capture(self, $recorder: &mut FrameRecorder, $scratch: &mut ScratchArena) -> Result<(), WireError> {
                let $v = &self;
                let $tag = EncoderOpcode::$variant;
                let result = (|| {
                    $body
                    Ok(())
                })();
                if result.is_err() { $recorder.rejected_ops.push(self.into_rejected()); }
                result
            }
            fn into_rejected(self) -> Op { Op::$variant(crate::encoder_data::capture_op(self)) }
        }
    };
}
capture_control!(SetViewportOp, SetViewport, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        SetViewportRecord {
            x: v.x,
            y: v.y,
            width: v.width,
            height: v.height,
            min_z: v.min_z,
            max_z: v.max_z
        }
    );
});
capture_control!(
    SetVertexSamplerOp,
    SetVertexSampler,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            SetVertexSamplerRecord {
                slot: u32::from(v.slot),
                reserved: 0,
                state: v.state
            }
        );
    }
);
capture_control!(
    SetVertexTextureOp,
    SetVertexTexture,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            SetVertexTextureRecord {
                id: v.id.map_or(0, crate::ids::TextureId::raw),
                slot: u32::from(v.slot),
                present: u32::from(v.id.is_some())
            }
        );
    }
);
capture_control!(BindDepthOp, BindDepth, v, recorder, scratch, tag, {
    match &v.binding {
        DepthBinding::None => capture_fixed!(
            recorder,
            scratch,
            tag,
            0,
            BindDepthNoneRecord {
                sample_count: u32::from(v.sample_count),
                flags: u32::from(v.flags.bits())
            }
        ),
        DepthBinding::Eager(handle, size, scale) => capture_fixed!(
            recorder,
            scratch,
            tag,
            1,
            BindDepthEagerRecord {
                handle: handle.raw(),
                width: size.0,
                height: size.1,
                scale: scale.percent(),
                sample_count: u32::from(v.sample_count),
                flags: u32::from(v.flags.bits()),
                reserved: 0
            }
        ),
        DepthBinding::Lazy(info, level, scale) => capture_fixed!(
            recorder,
            scratch,
            tag,
            2,
            BindDepthLazyRecord {
                texture: TextureRecord::capture(info),
                level: *level,
                scale: scale.percent(),
                sample_count: u32::from(v.sample_count),
                flags: u32::from(v.flags.bits())
            }
        ),
    }
});
capture_control!(BindColorOp, BindColor, v, recorder, scratch, tag, {
    match &v.info {
        RtBinding::Backbuffer {
            handle,
            msaa,
            msaa_srgb,
            sample_count,
            width,
            height,
        } => capture_fixed!(
            recorder,
            scratch,
            tag,
            0,
            BindColorBackbufferRecord {
                handle: handle.raw(),
                msaa: msaa.raw(),
                msaa_srgb: msaa_srgb.raw(),
                width: *width,
                height: *height,
                slot: u32::from(v.slot),
                scale: v.scale.percent(),
                sample_count: u32::from(*sample_count),
                reserved: 0
            }
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
        } => capture_fixed!(
            recorder,
            scratch,
            tag,
            1,
            BindColorStandaloneRecord {
                handle: handle.raw(),
                srgb: srgb.raw(),
                msaa: msaa.raw(),
                msaa_srgb: msaa_srgb.raw(),
                width: *width,
                height: *height,
                slot: u32::from(v.slot),
                scale: v.scale.percent(),
                sample_count: u32::from(*sample_count),
                format: *format as u32,
                has_alpha: u32::from(*has_alpha),
                reserved: 0
            }
        ),
        RtBinding::Texture {
            info,
            has_alpha,
            width,
            height,
            slice,
            level,
        } => capture_fixed!(
            recorder,
            scratch,
            tag,
            2,
            BindColorTextureRecord {
                texture: TextureRecord::capture(info),
                width: *width,
                height: *height,
                slot: u32::from(v.slot),
                scale: v.scale.percent(),
                slice: *slice,
                level: *level,
                has_alpha: u32::from(*has_alpha),
                reserved: 0
            }
        ),
    }
});
capture_control!(
    GenerateMipmapsOrderedOp,
    GenerateMipmapsOrdered,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(recorder, scratch, tag, IdRecord { id: v.old_id.raw() });
    }
);
capture_control!(
    UnbindExtraColorOp,
    UnbindExtraColor,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            SlotRecord {
                value: u32::from(v.slot),
                reserved: 0
            }
        );
    }
);
capture_control!(
    DestroyTextureOp,
    DestroyTexture,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(recorder, scratch, tag, IdRecord { id: v.tex_id.raw() });
    }
);
capture_control!(NoteColorReadOp, NoteColorRead, v, recorder, scratch, tag, {
    capture_fixed!(recorder, scratch, tag, IdRecord { id: v.src.raw() });
});
capture_control!(
    ResolveDepthSurfaceOp,
    ResolveDepthSurface,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(recorder, scratch, tag, depth(&v.transfer));
    }
);
capture_control!(StretchBlitOp, StretchBlit, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        StretchBlitRecord {
            source: surface(&v.src_info),
            destination: surface(&v.dst_info),
            source_region: region(&v.src_region),
            destination_region: region(&v.dst_region),
            mip_level: v.mip_level,
            render_quad: u32::from(v.render_quad),
            filter: v.filter,
            reserved: 0
        }
    );
});
capture_control!(ColorFillOp, ColorFill, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        ColorFillRecord {
            identity: identity(&v.kind),
            handle: v.fill.texture.raw(),
            msaa: v.fill.msaa.raw(),
            msaa_srgb: v.fill.msaa_srgb.raw(),
            logical_width: v.fill.logical_size.0,
            logical_height: v.fill.logical_size.1,
            texture_width: v.fill.texture_size.0,
            texture_height: v.fill.texture_size.1,
            format: v.fill.format as u32,
            scale: v.fill.scale.percent(),
            slice: v.fill.subresource.0,
            level: v.fill.subresource.1,
            rect: [v.fill.rect.0, v.fill.rect.1, v.fill.rect.2, v.fill.rect.3],
            rgba: [v.fill.rgba.0, v.fill.rgba.1, v.fill.rgba.2, v.fill.rgba.3],
            sample_count: u32::from(v.fill.sample_count),
            regenerate_mipmaps: u32::from(v.fill.regenerate_mipmaps)
        }
    );
});
capture_control!(CarryDepthOp, CarryDepth, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        CarryDepthRecord {
            previous: v.prev_id.raw(),
            current: v.cur_id.raw(),
            width: v.mip_w,
            height: v.mip_h
        }
    );
});
capture_control!(ClearColorOp, ClearColor, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        ClearColorRecord {
            rgba: [v.r_bits, v.g_bits, v.b_bits, v.a_bits],
            srgb_write: u32::from(v.srgb_write),
            reserved: 0
        }
    );
});
capture_control!(
    ClearColorRectsOp,
    ClearColorRects,
    v,
    recorder,
    scratch,
    tag,
    {
        recorder.rects(scratch, tag, &v.rects, || ClearColorRecord {
            rgba: [v.r_bits, v.g_bits, v.b_bits, v.a_bits],
            srgb_write: u32::from(v.srgb_write),
            reserved: 0,
        })?;
    }
);
capture_control!(
    ClearDepthStencilRectsOp,
    ClearDepthStencilRects,
    v,
    recorder,
    scratch,
    tag,
    {
        recorder.rects(scratch, tag, &v.list, || clear_depth(v.depth, v.stencil))?;
    }
);
capture_control!(
    ClearDepthStencilOp,
    ClearDepthStencil,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(recorder, scratch, tag, clear_depth(v.depth, v.stencil));
    }
);
capture_control!(
    ResolveDynamicDepthOp,
    ResolveDynamicDepth,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            ResolveDynamicDepthRecord {
                id: v.id.raw(),
                texture: TextureRecord::capture(&v.info)
            }
        );
    }
);
capture_control!(
    ResolveDepthTextureOp,
    ResolveDepthTexture,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            ResolveDepthTextureRecord {
                id: v.id.raw(),
                width: v.w,
                height: v.h,
                format: v.format as u32,
                reserved: 0
            }
        );
    }
);
impl CaptureControl for AdoptProgramOp {
    fn registration(&self) -> Option<u64> {
        Some(self.registration)
    }
    fn capture(
        self,
        recorder: &mut FrameRecorder,
        scratch: &mut ScratchArena,
    ) -> Result<(), WireError> {
        let result = recorder.fixed(scratch, EncoderOpcode::AdoptProgram, 0, || IdRecord {
            id: self.registration,
        });
        if result.is_err() {
            recorder.rejected_ops.push(self.into_rejected());
        }
        result
    }
    fn into_rejected(self) -> Op {
        Op::AdoptProgram(crate::encoder_data::capture_op(self))
    }
}

capture_control!(RetireColorOp, RetireColor, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        RetireColorRecord {
            base: v.retired.base.raw(),
            srgb: v.retired.srgb.raw(),
            msaa: v.retired.msaa.raw(),
            msaa_srgb: v.retired.msaa_srgb.raw()
        }
    );
});
capture_control!(RetireDepthOp, RetireDepth, v, recorder, scratch, tag, {
    capture_fixed!(recorder, scratch, tag, IdRecord { id: v.depth.raw() });
});
capture_control!(UploadColorOp, UploadColor, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        UploadColorRecord {
            handle: v.color_handle,
            bytes: span(v.bytes),
            width: v.width,
            height: v.height,
            stride: v.src_stride,
            reserved: 0
        }
    );
});
capture_control!(
    UploadResampledOp,
    UploadResampled,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            UploadResampledRecord {
                target: resampled(&v.target),
                bytes: span(v.bytes)
            }
        );
    }
);
capture_control!(
    GenerateMipmapsOp,
    GenerateMipmaps,
    v,
    recorder,
    scratch,
    tag,
    {
        capture_fixed!(
            recorder,
            scratch,
            tag,
            IdRecord {
                id: v.texture_id.raw()
            }
        );
    }
);
capture_control!(SetDumpDrawOp, SetDumpDraw, v, recorder, scratch, tag, {
    capture_fixed!(
        recorder,
        scratch,
        tag,
        SlotRecord {
            value: v.seq,
            reserved: 0
        }
    );
});

struct CaptureOwners<'a> {
    pages: &'a mut Vec<GuestPageLease>,
    owned_pages: &'a mut Vec<GuestOwnedPageLease>,
    queries: &'a mut Vec<GuestQueryLease>,
    redirties: &'a mut Vec<GuestRedirtyLease>,
    replies_u64: &'a mut Vec<ReplyU64>,
    replies_bool: &'a mut Vec<ReplyBool>,
    completion_pool: &'a crate::guest_completions::CompletionPool,
}

impl FrameRecorder {
    fn fixed_owned<T: CaptureControl, R: CommandRecord>(
        &mut self,
        scratch: &mut ScratchArena,
        tag: EncoderOpcode,
        value: T,
        build: impl FnOnce(T, &mut [u8], &mut CaptureOwners<'_>) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        let mut pending = Some(value);
        let mut owners = CaptureOwners {
            pages: &mut self.pages,
            owned_pages: &mut self.owned_pages,
            queries: &mut self.queries,
            redirties: &mut self.redirties,
            replies_u64: &mut self.replies_u64,
            replies_bool: &mut self.replies_bool,
            completion_pool: &self.completion_pool,
        };
        let result =
            self.slab
                .push_fixed_record(scratch, tag.into(), 0, size_of::<R>(), |destination| {
                    build(
                        pending.take().ok_or(WireError::InvalidValue)?,
                        destination,
                        &mut owners,
                    )
                });
        if let Some(value) = pending {
            self.rejected_ops.push(value.into_rejected());
        }
        result
    }
}

macro_rules! capture_resource {
    ($ty:ident,$variant:ident,$record:ident,$v:ident,$destination:ident,$owners:ident,$body:expr) => {
        impl CaptureControl for crate::encoder_data::$ty {
            fn capture(
                self,
                recorder: &mut FrameRecorder,
                scratch: &mut ScratchArena,
            ) -> Result<(), WireError> {
                recorder.fixed_owned::<Self, $record>(
                    scratch,
                    EncoderOpcode::$variant,
                    self,
                    |$v, $destination, $owners| $body,
                )
            }
            fn into_rejected(self) -> Op {
                Op::$variant(crate::encoder_data::capture_op(self))
            }
        }
    };
}

impl CaptureControl for crate::encoder_data::StageUploadOp {
    fn capture(
        self,
        recorder: &mut FrameRecorder,
        scratch: &mut ScratchArena,
    ) -> Result<(), WireError> {
        let recycle_pool = recorder.pagebox_pool;
        recorder.fixed_owned::<Self, StageUploadRecord>(
            scratch,
            EncoderOpcode::StageUpload,
            self,
            |v, destination, owners| {
                let lease =
                    GuestOwnedPageLease::new(v.page_box, owners.completion_pool, recycle_pool);
                let page = lease.descriptor();
                owners.owned_pages.push(lease);
                crate::encoder_records::write(
                    destination,
                    StageUploadRecord {
                        id: v.buffer_id.raw(),
                        offset: u64::from(v.dst_offset),
                        size: u64::from(v.size),
                        page,
                    },
                )
            },
        )
    }
    fn into_rejected(self) -> Op {
        Op::StageUpload {
            buffer_id: self.buffer_id,
            page_box: self.page_box,
            dst_offset: self.dst_offset,
            size: self.size,
        }
    }
}

capture_resource!(
    ReadColorHandleOp,
    ReadColorHandle,
    ReadHandleRecord,
    v,
    destination,
    owners,
    {
        let reply = v.slot_op.address();
        owners.replies_u64.push(v.slot_op);
        crate::encoder_records::write(
            destination,
            ReadHandleRecord {
                id: v.texture_id.raw(),
                reply,
            },
        )
    }
);
capture_resource!(
    ReadTextureHandleOp,
    ReadTextureHandle,
    ReadHandleRecord,
    v,
    destination,
    owners,
    {
        let reply = v.slot_op.address();
        owners.replies_u64.push(v.slot_op);
        crate::encoder_records::write(
            destination,
            ReadHandleRecord {
                id: v.texture_id.raw(),
                reply,
            },
        )
    }
);
capture_resource!(
    ReadTextureColorHandleOp,
    ReadTextureColorHandle,
    ReadHandleRecord,
    v,
    destination,
    owners,
    {
        let reply = v.slot_op.address();
        owners.replies_u64.push(v.slot_op);
        crate::encoder_records::write(
            destination,
            ReadHandleRecord {
                id: v.texture_id.raw(),
                reply,
            },
        )
    }
);
capture_resource!(
    ReadDeviceBufferOp,
    ReadDeviceBuffer,
    ReadDeviceBufferRecord,
    v,
    destination,
    owners,
    {
        let reply = v.done.address();
        owners.replies_bool.push(v.done);
        crate::encoder_records::write(
            destination,
            ReadDeviceBufferRecord {
                id: v.buffer_id.raw(),
                destination: v.dst_ptr,
                length: v.dst_len,
                reply,
            },
        )
    }
);
capture_resource!(
    BeginVisibilityOp,
    BeginVisibility,
    QueryRecord,
    v,
    destination,
    owners,
    {
        let lease = GuestQueryLease::new_pooled(v.c, owners.completion_pool);
        let descriptor = lease.descriptor();
        owners.queries.push(lease);
        crate::encoder_records::write(
            destination,
            QueryRecord {
                generation: v.generation,
                descriptor,
            },
        )
    }
);
capture_resource!(
    EndVisibilityOp,
    EndVisibility,
    QueryRecord,
    v,
    destination,
    owners,
    {
        let lease = GuestQueryLease::new_pooled(v.core, owners.completion_pool);
        let descriptor = lease.descriptor();
        owners.queries.push(lease);
        crate::encoder_records::write(
            destination,
            QueryRecord {
                generation: v.generation,
                descriptor,
            },
        )
    }
);
capture_resource!(
    UploadTextureOp,
    UploadTexture,
    TextureUploadRecord,
    v,
    destination,
    owners,
    capture_upload(
        v.job,
        destination,
        owners.pages,
        owners.redirties,
        owners.completion_pool,
        0,
        0
    )
);
capture_resource!(
    UploadTextureAndMipsOp,
    UploadTextureAndMips,
    TextureUploadRecord,
    v,
    destination,
    owners,
    capture_upload(
        v.job,
        destination,
        owners.pages,
        owners.redirties,
        owners.completion_pool,
        v.texture_id.raw(),
        u32::from(v.flags.bits())
    )
);

#[cfg(test)]
mod tests;
