//! Frame recording and native reconstruction across the PE/Unix boundary.
//!
//! Operation bytes are captured as API calls record them. Packet admission writes only
//! frame metadata; guest allocations and replies remain owned by their allocating runtime.

use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64},
};

use mtld3d_shared::{
    encoder_protocol::EncoderOpcode,
    encoder_wire::{FrameSlab, LeaseCompletion, WireError, WireReader, WireWriter},
    record_handle::DeviceRecordHandle,
};

use crate::{
    buffer_rename::BufferMapMode,
    dxso::DxsoProgram,
    encoder_data::{
        BeginVisibilityOp, BindColorOp, BindDepthOp, CarryDepthOp, ClearColorOp, ClearColorRectsOp,
        ClearDepthStencilOp, ClearDepthStencilRectsOp, ColorFillOp, DestroyTextureOp,
        EndVisibilityOp, FrameData, FrameDataFlags, GenerateMipmapsOp, GenerateMipmapsOrderedOp,
        NoteColorReadOp, Op, ReadColorHandleOp, ReadDeviceBufferOp, ReadTextureColorHandleOp,
        ReadTextureHandleOp, RegisterProgramOp, ResolveDepthSurfaceOp, ResolveDepthTextureOp,
        ResolveDynamicDepthOp, RetireColorOp, RetireDepthOp, SetDumpDrawOp, SetVertexSamplerOp,
        SetVertexTextureOp, SetViewportOp, StretchBlitOp, TextureInfo, TextureUploadJob,
        UnbindExtraColorOp, UploadColorOp, UploadResampledOp, UploadTextureAndMipsOp,
        UploadTextureOp,
    },
    encoder_draw::{DrawReader, DrawWriter, read_scratch_slice, write_scratch_slice},
    encoder_reply::{ReplyBool, ReplyU64},
    encoder_value::WireValue,
    gamma::Change,
    guest_pages::{GuestPageDescriptor, GuestPageLease},
    guest_queries::{GuestQueryDescriptor, GuestQueryLease, QueryLeaseCache},
    ids::ProgramId,
    passes::BackbufferContents,
    present::LayerPacing,
    scratch::ScratchArena,
    upload_redirty::{GuestRedirtyDescriptor, GuestRedirtyLease},
};

mod metadata;
const FRAME_METADATA_TAG: u16 = 1;
const FRAME_CHUNK_TAG: u16 = 1;

mod owner;
mod recording;
use recording::RecordedSpans;
mod replay;
#[cfg(test)]
mod tests;
#[cfg(any(test, debug_assertions))]
mod validation;
use metadata::write_metadata;
pub use owner::{FramePacket, PacketLease};
pub use replay::ReplayPacket;

/// API-side operation bytes and owners awaiting native acknowledgment.
pub struct FrameRecorder {
    pagebox_pool: Option<&'static crate::page_box_pool::PageBoxPool>,
    completion_pool: crate::guest_completions::CompletionPool,
    slab: RecordedSpans,
    metadata: FrameSlab,
    draws: DrawWriter,
    pages: Vec<GuestPageLease>,
    queries: Vec<GuestQueryLease>,
    redirties: Vec<GuestRedirtyLease>,
    replies_u64: Vec<ReplyU64>,
    replies_bool: Vec<ReplyBool>,
    error: Option<WireError>,
    count: usize,
    ranges: Vec<(u64, u64)>,
    readbacks: Vec<(u64, u64)>,
    registrations: Vec<u64>,
    rejected_ops: Vec<Op>,
}

impl Default for FrameRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self::with_completion_pool(crate::guest_completions::CompletionPool::new())
    }

    /// Set the original runtime's pool for retired VB/IB allocations.
    pub const fn set_pagebox_pool(&mut self, pool: &'static crate::page_box_pool::PageBoxPool) {
        self.pagebox_pool = Some(pool);
    }

    #[must_use]
    pub fn with_completion_pool(completion_pool: crate::guest_completions::CompletionPool) -> Self {
        Self {
            pagebox_pool: None,
            completion_pool,
            slab: RecordedSpans::new(),
            metadata: FrameSlab::new(),
            draws: DrawWriter::new(),
            pages: Vec::new(),
            queries: Vec::new(),
            redirties: Vec::new(),
            replies_u64: Vec::new(),
            replies_bool: Vec::new(),
            error: None,
            count: 0,
            ranges: Vec::new(),
            readbacks: Vec::new(),
            registrations: Vec::new(),
            rejected_ops: Vec::new(),
        }
    }

    /// Record one owned operation without a later whole-frame serialization pass.
    pub fn record(&mut self, scratch: &mut ScratchArena, op: Op) {
        let _ = self.try_record(scratch, op);
    }

    /// Record an operation, returning a latched serialization error.
    ///
    /// # Errors
    /// Returns an allocation, size or invalid payload error and rejects the frame.
    pub fn try_record(&mut self, scratch: &mut ScratchArena, op: Op) -> Result<(), WireError> {
        if let Some(error) = self.error {
            self.rejected_ops.push(op);
            return Err(error);
        }
        #[cfg(any(test, debug_assertions))]
        capture_ranges(&op, &mut self.ranges);
        if let Op::ReadDeviceBuffer(value) = &op {
            self.readbacks.push((value.dst_ptr, value.dst_len));
        }
        let registration = match &op {
            Op::AdoptProgram(value) => Some(value.registration),
            _ => None,
        };
        let tag = op_tag(&op);
        let bound = op_record_bound(&op);
        // Draws and constant updates lend their fields to the writer. Moving the
        // entire operation through the ownership callback copies the large enum
        // twice even though these records transfer no Rust-owned values.
        let result = if matches!(
            op,
            Op::Draw(_)
                | Op::SetVsConstRange { .. }
                | Op::SetPsConstRange { .. }
                | Op::SetFfVsConstRange { .. }
        ) {
            let result = tag.and_then(|tag| {
                self.slab
                    .push_record(scratch, u16::from(tag), bound?, |writer| {
                        write_draw_or_constants(&op, writer, &mut self.draws)
                    })
            });
            if result.is_err() {
                self.rejected_ops.push(op);
            }
            result
        } else {
            let mut op = Some(op);
            let result = tag.and_then(|tag| {
                self.slab
                    .push_record(scratch, u16::from(tag), bound?, |writer| {
                        let op = op.take().ok_or(WireError::InvalidValue)?;
                        write_operation(
                            op,
                            writer,
                            &mut WriteContext {
                                draws: &mut self.draws,
                                pages: &mut self.pages,
                                queries: &mut self.queries,
                                replies_u64: &mut self.replies_u64,
                                replies_bool: &mut self.replies_bool,
                                redirties: &mut self.redirties,
                                pool: &self.completion_pool,
                                pagebox_pool: self.pagebox_pool,
                            },
                        )
                    })
            });
            if let Some(op) = op {
                self.rejected_ops.push(op);
            }
            result
        };
        if let Err(reason) = result {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "encoder operation recording failed: {reason:?}; rejecting frame");
            self.error = Some(reason);
        } else {
            self.count += 1;
            if let Some(registration) = registration {
                self.registrations.push(registration);
            }
        }
        result
    }

    /// Capture a draw without constructing the larger operation enum.
    ///
    /// # Errors
    /// Returns the latched frame error or a wire capture failure.
    pub fn record_draw(
        &mut self,
        scratch: &mut ScratchArena,
        draw: &crate::draw_data::DrawOp,
    ) -> Result<(), WireError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        #[cfg(any(test, debug_assertions))]
        capture_draw_ranges(draw, &mut self.ranges);
        let result =
            self.slab
                .push_record(scratch, u16::from(EncoderOpcode::Draw), 4096, |writer| {
                    self.draws.encode_draw(draw, writer)
                });
        self.finish_record(result)
    }

    /// Capture a VS constant delta without constructing an operation enum.
    ///
    /// # Errors
    /// Returns the latched frame error or a wire capture failure.
    pub fn record_vs_constants(
        &mut self,
        scratch: &mut ScratchArena,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) -> Result<(), WireError> {
        self.record_constants(
            scratch,
            EncoderOpcode::SetVsConstRange,
            start_row,
            rows,
            data,
        )
    }

    /// Capture a PS constant delta without constructing an operation enum.
    ///
    /// # Errors
    /// Returns the latched frame error or a wire capture failure.
    pub fn record_ps_constants(
        &mut self,
        scratch: &mut ScratchArena,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) -> Result<(), WireError> {
        self.record_constants(
            scratch,
            EncoderOpcode::SetPsConstRange,
            start_row,
            rows,
            data,
        )
    }

    /// Capture a fixed-function VS constant delta without an operation enum.
    ///
    /// # Errors
    /// Returns the latched frame error or a wire capture failure.
    pub fn record_ff_vs_constants(
        &mut self,
        scratch: &mut ScratchArena,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) -> Result<(), WireError> {
        self.record_constants(
            scratch,
            EncoderOpcode::SetFfVsConstRange,
            start_row,
            rows,
            data,
        )
    }

    fn record_constants(
        &mut self,
        scratch: &mut ScratchArena,
        opcode: EncoderOpcode,
        start_row: u16,
        rows: u16,
        data: crate::draw_data::ScratchSlice,
    ) -> Result<(), WireError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let result = self
            .slab
            .push_record(scratch, u16::from(opcode), 22, |writer| {
                write_const_range(start_row, rows, data, writer)
            });
        self.finish_record(result)
    }

    const fn finish_record(&mut self, result: Result<(), WireError>) -> Result<(), WireError> {
        match result {
            Ok(()) => self.count += 1,
            Err(error) => self.error = Some(error),
        }
        result
    }

    /// Record changed snapshot fields directly from API state.
    ///
    /// # Errors
    /// Returns a latched capture error or a failure writing this delta.
    pub fn record_snapshot_delta(
        &mut self,
        scratch: &mut ScratchArena,
        delta: &crate::encoder_draw::SnapshotDelta<'_>,
    ) -> Result<(), WireError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let result = self.slab.push_record(
            scratch,
            u16::from(EncoderOpcode::SetSnapshot),
            crate::encoder_draw::SNAPSHOT_DELTA_MAX_BYTES,
            |writer| self.draws.encode_snapshot_delta(delta, writer),
        );
        match result {
            Ok(()) => self.count += 1,
            Err(error) => self.error = Some(error),
        }
        result
    }

    #[must_use]
    pub const fn recording_error(&self) -> Option<WireError> {
        self.error
    }

    /// Reuse capture allocations only after replay and all local references have ended.
    pub fn reset(&mut self) {
        self.slab.clear();
        self.metadata.clear();
        self.draws.clear();
        self.ranges.clear();
        self.readbacks.clear();
        self.registrations.clear();
        self.rejected_ops.clear();
        self.replies_u64.clear();
        self.replies_bool.clear();
        self.error = None;
        self.count = 0;
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

const fn op_tag(op: &Op) -> Result<EncoderOpcode, WireError> {
    match op {
        Op::SetVsConstRange { .. } => Ok(EncoderOpcode::SetVsConstRange),
        Op::SetPsConstRange { .. } => Ok(EncoderOpcode::SetPsConstRange),
        Op::SetFfVsConstRange { .. } => Ok(EncoderOpcode::SetFfVsConstRange),
        Op::Draw(_) => Ok(EncoderOpcode::Draw),
        Op::SetViewport(_) => Ok(EncoderOpcode::SetViewport),
        Op::SetVertexSampler(_) => Ok(EncoderOpcode::SetVertexSampler),
        Op::SetVertexTexture(_) => Ok(EncoderOpcode::SetVertexTexture),
        Op::BindDepth(_) => Ok(EncoderOpcode::BindDepth),
        Op::BindColor(_) => Ok(EncoderOpcode::BindColor),
        Op::GenerateMipmapsOrdered(_) => Ok(EncoderOpcode::GenerateMipmapsOrdered),
        Op::UnbindExtraColor(_) => Ok(EncoderOpcode::UnbindExtraColor),
        Op::DestroyTexture(_) => Ok(EncoderOpcode::DestroyTexture),
        Op::ReadColorHandle(_) => Ok(EncoderOpcode::ReadColorHandle),
        Op::NoteColorRead(_) => Ok(EncoderOpcode::NoteColorRead),
        Op::ResolveDepthSurface(_) => Ok(EncoderOpcode::ResolveDepthSurface),
        Op::StretchBlit(_) => Ok(EncoderOpcode::StretchBlit),
        Op::ColorFill(_) => Ok(EncoderOpcode::ColorFill),
        Op::CarryDepth(_) => Ok(EncoderOpcode::CarryDepth),
        Op::ClearColor(_) => Ok(EncoderOpcode::ClearColor),
        Op::ClearColorRects(_) => Ok(EncoderOpcode::ClearColorRects),
        Op::ClearDepthStencilRects(_) => Ok(EncoderOpcode::ClearDepthStencilRects),
        Op::ClearDepthStencil(_) => Ok(EncoderOpcode::ClearDepthStencil),
        Op::ResolveDynamicDepth(_) => Ok(EncoderOpcode::ResolveDynamicDepth),
        Op::ResolveDepthTexture(_) => Ok(EncoderOpcode::ResolveDepthTexture),
        Op::ReadDeviceBuffer(_) => Ok(EncoderOpcode::ReadDeviceBuffer),
        Op::AdoptProgram(_) => Ok(EncoderOpcode::AdoptProgram),
        Op::BeginVisibility(_) => Ok(EncoderOpcode::BeginVisibility),
        Op::EndVisibility(_) => Ok(EncoderOpcode::EndVisibility),
        Op::RetireColor(_) => Ok(EncoderOpcode::RetireColor),
        Op::RetireDepth(_) => Ok(EncoderOpcode::RetireDepth),
        Op::UploadColor(_) => Ok(EncoderOpcode::UploadColor),
        Op::UploadResampled(_) => Ok(EncoderOpcode::UploadResampled),
        Op::ReadTextureHandle(_) => Ok(EncoderOpcode::ReadTextureHandle),
        Op::GenerateMipmaps(_) => Ok(EncoderOpcode::GenerateMipmaps),
        Op::ReadTextureColorHandle(_) => Ok(EncoderOpcode::ReadTextureColorHandle),
        Op::UploadTextureAndMips(_) => Ok(EncoderOpcode::UploadTextureAndMips),
        Op::UploadTexture(_) => Ok(EncoderOpcode::UploadTexture),
        Op::SetDumpDraw(_) => Ok(EncoderOpcode::SetDumpDraw),
        Op::StageUpload { .. } => Ok(EncoderOpcode::StageUpload),
        Op::RegisterProgram(_) | Op::SetSnapshot(_) => Err(WireError::InvalidValue),
    }
}

struct WriteContext<'a> {
    pagebox_pool: Option<&'static crate::page_box_pool::PageBoxPool>,
    draws: &'a mut DrawWriter,
    pages: &'a mut Vec<GuestPageLease>,
    queries: &'a mut Vec<GuestQueryLease>,
    replies_u64: &'a mut Vec<ReplyU64>,
    replies_bool: &'a mut Vec<ReplyBool>,
    redirties: &'a mut Vec<GuestRedirtyLease>,
    pool: &'a crate::guest_completions::CompletionPool,
}

fn write_operation(
    op: Op,
    writer: &mut WireWriter<'_>,
    context: &mut WriteContext<'_>,
) -> Result<(), WireError> {
    let WriteContext {
        pagebox_pool,
        draws,
        pages,
        queries,
        replies_u64,
        replies_bool,
        redirties,
        pool,
    } = context;
    match op {
        op @ (Op::SetVsConstRange { .. }
        | Op::SetPsConstRange { .. }
        | Op::SetFfVsConstRange { .. }
        | Op::Draw(_)) => write_draw_or_constants(&op, writer, draws),
        Op::SetViewport(value) => value.write_wire(writer),
        Op::SetVertexSampler(value) => value.write_wire(writer),
        Op::SetVertexTexture(value) => value.write_wire(writer),
        Op::BindDepth(value) => value.write_wire(writer),
        Op::BindColor(value) => value.write_wire(writer),
        Op::GenerateMipmapsOrdered(value) => value.write_wire(writer),
        Op::UnbindExtraColor(value) => value.write_wire(writer),
        Op::DestroyTexture(value) => value.write_wire(writer),
        Op::NoteColorRead(value) => value.write_wire(writer),
        Op::ResolveDepthSurface(value) => value.write_wire(writer),
        Op::StretchBlit(value) => value.write_wire(writer),
        Op::ColorFill(value) => value.write_wire(writer),
        Op::CarryDepth(value) => value.write_wire(writer),
        Op::ClearColor(value) => value.write_wire(writer),
        Op::ClearColorRects(value) => value.write_wire(writer),
        Op::ClearDepthStencilRects(value) => value.write_wire(writer),
        Op::ClearDepthStencil(value) => value.write_wire(writer),
        Op::ResolveDynamicDepth(value) => value.write_wire(writer),
        Op::ResolveDepthTexture(value) => value.write_wire(writer),
        Op::RetireColor(value) => value.write_wire(writer),
        Op::RetireDepth(value) => value.write_wire(writer),
        Op::UploadColor(value) => {
            value.color_handle.write_wire(writer)?;
            write_scratch_slice(value.bytes, writer)?;
            value.width.write_wire(writer)?;
            value.height.write_wire(writer)?;
            value.src_stride.write_wire(writer)
        }
        Op::UploadResampled(value) => {
            value.target.write_wire(writer)?;
            write_scratch_slice(value.bytes, writer)
        }
        Op::GenerateMipmaps(value) => value.write_wire(writer),
        Op::SetDumpDraw(value) => value.write_wire(writer),
        Op::ReadColorHandle(value) => {
            let address = value.slot_op.address();
            replies_u64.push(value.slot_op);
            value.texture_id.write_wire(writer)?;
            address.write_wire(writer)
        }
        Op::ReadTextureHandle(value) => {
            let address = value.slot_op.address();
            replies_u64.push(value.slot_op);
            value.texture_id.write_wire(writer)?;
            address.write_wire(writer)
        }
        Op::ReadTextureColorHandle(value) => {
            let address = value.slot_op.address();
            replies_u64.push(value.slot_op);
            value.texture_id.write_wire(writer)?;
            address.write_wire(writer)
        }
        Op::ReadDeviceBuffer(value) => {
            let address = value.done.address();
            replies_bool.push(value.done);
            value.buffer_id.write_wire(writer)?;
            value.dst_ptr.write_wire(writer)?;
            value.dst_len.write_wire(writer)?;
            address.write_wire(writer)
        }
        Op::BeginVisibility(value) => write_query(value.c, value.generation, writer, queries, pool),
        Op::EndVisibility(value) => {
            write_query(value.core, value.generation, writer, queries, pool)
        }
        Op::AdoptProgram(value) => value.registration.write_wire(writer),
        Op::RegisterProgram(_) | Op::SetSnapshot(_) => Err(WireError::InvalidValue),
        Op::UploadTexture(value) => write_upload(value.job, writer, pages, redirties, pool, None),
        Op::UploadTextureAndMips(value) => write_upload(
            value.job,
            writer,
            pages,
            redirties,
            pool,
            Some((value.texture_id, value.flags)),
        ),
        Op::StageUpload {
            buffer_id,
            page_box,
            dst_offset,
            size,
        } => {
            let lease = GuestPageLease::for_recyclable_pooled(page_box, pool, *pagebox_pool);
            let descriptor = lease.descriptor();
            pages.push(lease);
            buffer_id.write_wire(writer)?;
            dst_offset.write_wire(writer)?;
            size.write_wire(writer)?;
            descriptor.write_wire(writer)
        }
    }
}

/// Serialize borrowed hot-operation fields without moving the owning enum.
fn write_draw_or_constants(
    op: &Op,
    writer: &mut WireWriter<'_>,
    draws: &mut DrawWriter,
) -> Result<(), WireError> {
    match op {
        Op::SetVsConstRange {
            start_row,
            rows,
            data,
        }
        | Op::SetPsConstRange {
            start_row,
            rows,
            data,
        }
        | Op::SetFfVsConstRange {
            start_row,
            rows,
            data,
        } => write_const_range(*start_row, *rows, *data, writer),
        Op::Draw(draw) => draws.encode_draw(draw, writer),
        _ => Err(WireError::InvalidValue),
    }
}

fn write_const_range(
    start_row: u16,
    rows: u16,
    data: crate::draw_data::ScratchSlice,
    writer: &mut WireWriter<'_>,
) -> Result<(), WireError> {
    if usize::from(start_row) + usize::from(rows) > crate::draw_data::CONSTANT_ROWS
        || data.as_raw().1 < u32::from(rows) * 16
    {
        return Err(WireError::InvalidValue);
    }
    start_row.write_wire(writer)?;
    rows.write_wire(writer)?;
    write_scratch_slice(data, writer)
}

fn write_query(
    core: Arc<crate::visibility::VisibilityQueryCore>,
    generation: u64,
    writer: &mut WireWriter<'_>,
    queries: &mut Vec<GuestQueryLease>,
    pool: &crate::guest_completions::CompletionPool,
) -> Result<(), WireError> {
    let lease = GuestQueryLease::new_pooled(core, pool);
    let descriptor = lease.descriptor();
    queries.push(lease);
    generation.write_wire(writer)?;
    descriptor.write_wire(writer)
}

fn write_upload(
    job: TextureUploadJob,
    writer: &mut WireWriter<'_>,
    pages: &mut Vec<GuestPageLease>,
    redirties: &mut Vec<GuestRedirtyLease>,
    pool: &crate::guest_completions::CompletionPool,
    prefix: Option<(
        crate::ids::TextureId,
        crate::encoder_data::UploadTextureOpFlags,
    )>,
) -> Result<(), WireError> {
    let declined = crate::upload_redirty::RedirtyEntry {
        subresource: job.redirty_subresource(),
        face: job.destination_slice,
        level: job.level,
        rect: job.redirty_rect(),
    };
    let emitted = job.emitted_answer();
    let redirty = GuestRedirtyLease::new_pooled(job.redirty, declined, emitted, pool);
    let feedback = redirty.descriptor();
    redirties.push(redirty);
    let lease = GuestPageLease::for_read_pooled(job.staging, pool);
    let page = lease.descriptor();
    pages.push(lease);
    if let Some((texture_id, flags)) = prefix {
        texture_id.write_wire(writer)?;
        flags.write_wire(writer)?;
    }
    job.info.write_wire(writer)?;
    job.level.write_wire(writer)?;
    job.destination_slice.write_wire(writer)?;
    u32::try_from(job.staging_index)
        .map_err(|_| WireError::TooLarge)?
        .write_wire(writer)?;
    job.origin_x.write_wire(writer)?;
    job.origin_y.write_wire(writer)?;
    job.region_w.write_wire(writer)?;
    job.region_h.write_wire(writer)?;
    job.src_d3d_format.write_wire(writer)?;
    job.src_pitch.write_wire(writer)?;
    job.bytes_per_pixel.write_wire(writer)?;
    job.depth.write_wire(writer)?;
    job.slice_pitch.write_wire(writer)?;
    job.release_staging.write_wire(writer)?;
    job.upload_generation.write_wire(writer)?;
    feedback.write_wire(writer)?;
    page.write_wire(writer)
}

fn read_operation(
    tag: &EncoderOpcode,
    reader: &mut WireReader<'_>,
    draws: &mut DrawReader,
    scratch: &mut ScratchArena,
    queries: &mut QueryLeaseCache,
    resolve_program: &mut impl FnMut(u64) -> Result<(ProgramId, DxsoProgram), WireError>,
) -> Result<Op, WireError> {
    match tag {
        EncoderOpcode::SetVsConstRange => Ok(Op::SetVsConstRange {
            start_row: WireValue::read_wire(reader)?,
            rows: WireValue::read_wire(reader)?,
            data: read_scratch_slice(reader)?,
        }),
        EncoderOpcode::SetPsConstRange => Ok(Op::SetPsConstRange {
            start_row: WireValue::read_wire(reader)?,
            rows: WireValue::read_wire(reader)?,
            data: read_scratch_slice(reader)?,
        }),
        EncoderOpcode::SetFfVsConstRange => Ok(Op::SetFfVsConstRange {
            start_row: WireValue::read_wire(reader)?,
            rows: WireValue::read_wire(reader)?,
            data: read_scratch_slice(reader)?,
        }),
        EncoderOpcode::Draw => draws.decode_draw(reader).map(Op::Draw),
        EncoderOpcode::SetSnapshot => draws
            .decode_snapshot_delta(reader, scratch)
            .map(Op::SetSnapshot),
        EncoderOpcode::SetViewport => Ok(Op::SetViewport(crate::encoder_data::capture_op(
            SetViewportOp::read_wire(reader)?,
        ))),
        EncoderOpcode::SetVertexSampler => Ok(Op::SetVertexSampler(
            crate::encoder_data::capture_op(SetVertexSamplerOp::read_wire(reader)?),
        )),
        EncoderOpcode::SetVertexTexture => Ok(Op::SetVertexTexture(
            crate::encoder_data::capture_op(SetVertexTextureOp::read_wire(reader)?),
        )),
        EncoderOpcode::BindDepth => Ok(Op::BindDepth(crate::encoder_data::capture_op(
            BindDepthOp::read_wire(reader)?,
        ))),
        EncoderOpcode::BindColor => Ok(Op::BindColor(crate::encoder_data::capture_op(
            BindColorOp::read_wire(reader)?,
        ))),
        EncoderOpcode::GenerateMipmapsOrdered => Ok(Op::GenerateMipmapsOrdered(
            crate::encoder_data::capture_op(GenerateMipmapsOrderedOp::read_wire(reader)?),
        )),
        EncoderOpcode::UnbindExtraColor => Ok(Op::UnbindExtraColor(
            crate::encoder_data::capture_op(UnbindExtraColorOp::read_wire(reader)?),
        )),
        EncoderOpcode::DestroyTexture => Ok(Op::DestroyTexture(crate::encoder_data::capture_op(
            DestroyTextureOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ReadColorHandle => {
            let texture_id = WireValue::read_wire(reader)?;
            let slot_op = read_reply_u64(reader)?;
            Ok(Op::ReadColorHandle(crate::encoder_data::capture_op(
                ReadColorHandleOp {
                    texture_id,
                    slot_op,
                },
            )))
        }
        EncoderOpcode::NoteColorRead => Ok(Op::NoteColorRead(crate::encoder_data::capture_op(
            NoteColorReadOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ResolveDepthSurface => Ok(Op::ResolveDepthSurface(
            crate::encoder_data::capture_op(ResolveDepthSurfaceOp::read_wire(reader)?),
        )),
        EncoderOpcode::StretchBlit => Ok(Op::StretchBlit(crate::encoder_data::capture_op(
            StretchBlitOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ColorFill => Ok(Op::ColorFill(crate::encoder_data::capture_op(
            ColorFillOp::read_wire(reader)?,
        ))),
        EncoderOpcode::CarryDepth => Ok(Op::CarryDepth(crate::encoder_data::capture_op(
            CarryDepthOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ClearColor => Ok(Op::ClearColor(crate::encoder_data::capture_op(
            ClearColorOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ClearColorRects => Ok(Op::ClearColorRects(crate::encoder_data::capture_op(
            ClearColorRectsOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ClearDepthStencilRects => Ok(Op::ClearDepthStencilRects(
            crate::encoder_data::capture_op(ClearDepthStencilRectsOp::read_wire(reader)?),
        )),
        EncoderOpcode::ClearDepthStencil => Ok(Op::ClearDepthStencil(
            crate::encoder_data::capture_op(ClearDepthStencilOp::read_wire(reader)?),
        )),
        EncoderOpcode::ResolveDynamicDepth => Ok(Op::ResolveDynamicDepth(
            crate::encoder_data::capture_op(ResolveDynamicDepthOp::read_wire(reader)?),
        )),
        EncoderOpcode::ResolveDepthTexture => Ok(Op::ResolveDepthTexture(
            crate::encoder_data::capture_op(ResolveDepthTextureOp::read_wire(reader)?),
        )),
        EncoderOpcode::ReadDeviceBuffer => {
            let buffer_id = WireValue::read_wire(reader)?;
            let dst_ptr = reader.u64()?;
            let dst_len = reader.u64()?;
            validate_range(dst_ptr, dst_len, 1)?;
            let address = reader.u64()?;
            validate_range(
                address,
                size_of::<AtomicU32>() as u64,
                align_of::<AtomicU32>(),
            )?;
            // SAFETY: the trusted packet retains the reply cell until replay completion.
            let done = unsafe { ReplyBool::from_guest(address) };
            Ok(Op::ReadDeviceBuffer(crate::encoder_data::capture_op(
                ReadDeviceBufferOp {
                    done,
                    buffer_id,
                    dst_ptr,
                    dst_len,
                },
            )))
        }
        EncoderOpcode::AdoptProgram => {
            let (shader_id, program) = resolve_program(reader.u64()?)?;
            Ok(Op::RegisterProgram(crate::encoder_data::capture_op(
                RegisterProgramOp { shader_id, program },
            )))
        }
        EncoderOpcode::BeginVisibility => {
            let generation = reader.u64()?;
            let descriptor = GuestQueryDescriptor::read_wire(reader)?;
            // SAFETY: the packet retains this unique query publication until acknowledgment.
            let c = unsafe { queries.adopt(descriptor)? };
            Ok(Op::BeginVisibility(crate::encoder_data::capture_op(
                BeginVisibilityOp { generation, c },
            )))
        }
        EncoderOpcode::EndVisibility => {
            let generation = reader.u64()?;
            let descriptor = GuestQueryDescriptor::read_wire(reader)?;
            // SAFETY: the packet retains this unique query publication until acknowledgment.
            let core = unsafe { queries.adopt(descriptor)? };
            Ok(Op::EndVisibility(crate::encoder_data::capture_op(
                EndVisibilityOp { generation, core },
            )))
        }
        EncoderOpcode::RetireColor => Ok(Op::RetireColor(crate::encoder_data::capture_op(
            RetireColorOp::read_wire(reader)?,
        ))),
        EncoderOpcode::RetireDepth => Ok(Op::RetireDepth(crate::encoder_data::capture_op(
            RetireDepthOp::read_wire(reader)?,
        ))),
        EncoderOpcode::UploadColor => Ok(Op::UploadColor(crate::encoder_data::capture_op(
            UploadColorOp {
                color_handle: reader.u64()?,
                bytes: read_scratch_slice(reader)?,
                width: reader.u32()?,
                height: reader.u32()?,
                src_stride: reader.u32()?,
            },
        ))),
        EncoderOpcode::UploadResampled => Ok(Op::UploadResampled(crate::encoder_data::capture_op(
            UploadResampledOp {
                target: WireValue::read_wire(reader)?,
                bytes: read_scratch_slice(reader)?,
            },
        ))),
        EncoderOpcode::ReadTextureHandle => {
            let texture_id = WireValue::read_wire(reader)?;
            let slot_op = read_reply_u64(reader)?;
            Ok(Op::ReadTextureHandle(crate::encoder_data::capture_op(
                ReadTextureHandleOp {
                    texture_id,
                    slot_op,
                },
            )))
        }
        EncoderOpcode::GenerateMipmaps => Ok(Op::GenerateMipmaps(crate::encoder_data::capture_op(
            GenerateMipmapsOp::read_wire(reader)?,
        ))),
        EncoderOpcode::ReadTextureColorHandle => {
            let texture_id = WireValue::read_wire(reader)?;
            let slot_op = read_reply_u64(reader)?;
            Ok(Op::ReadTextureColorHandle(crate::encoder_data::capture_op(
                ReadTextureColorHandleOp {
                    texture_id,
                    slot_op,
                },
            )))
        }
        EncoderOpcode::UploadTextureAndMips => {
            let texture_id = WireValue::read_wire(reader)?;
            let flags = WireValue::read_wire(reader)?;
            let job = read_upload(reader)?;
            Ok(Op::UploadTextureAndMips(crate::encoder_data::capture_op(
                UploadTextureAndMipsOp {
                    job,
                    texture_id,
                    flags,
                },
            )))
        }
        EncoderOpcode::UploadTexture => read_upload(reader)
            .map(|job| Op::UploadTexture(crate::encoder_data::capture_op(UploadTextureOp { job }))),
        EncoderOpcode::SetDumpDraw => Ok(Op::SetDumpDraw(crate::encoder_data::capture_op(
            SetDumpDrawOp::read_wire(reader)?,
        ))),
        EncoderOpcode::StageUpload => {
            let buffer_id = WireValue::read_wire(reader)?;
            let dst_offset = reader.u32()?;
            let size = reader.u32()?;
            let descriptor = GuestPageDescriptor::read_wire(reader)?;
            // SAFETY: the packet retains this uniquely published page through acknowledgment.
            let page_box = unsafe { descriptor.adopt_owned()? };
            Ok(Op::StageUpload {
                buffer_id,
                page_box,
                dst_offset,
                size,
            })
        }
    }
}

fn read_reply_u64(reader: &mut WireReader<'_>) -> Result<ReplyU64, WireError> {
    let address = reader.u64()?;
    validate_range(
        address,
        size_of::<AtomicU64>() as u64,
        align_of::<AtomicU64>(),
    )?;
    // SAFETY: the trusted packet retains this aligned reply cell until replay completes.
    Ok(unsafe { ReplyU64::from_guest(address) })
}

fn validate_range(address: u64, length: u64, alignment: usize) -> Result<(), WireError> {
    let address = usize::try_from(address).map_err(|_| WireError::InvalidValue)?;
    let length = usize::try_from(length).map_err(|_| WireError::InvalidValue)?;
    if address == 0 || !address.is_multiple_of(alignment) || address.checked_add(length).is_none() {
        return Err(WireError::InvalidValue);
    }
    Ok(())
}

impl WireValue for DeviceRecordHandle {
    const MIN_WIRE_BYTES: usize = 8;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.raw().write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let raw = reader.u64()?;
        if raw != 0 && !reader.has_trusted_addresses() {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: the trusted reader establishes the live device-record identity and lifetime.
        Ok(unsafe { Self::new(raw) })
    }
}

impl WireValue for LayerPacing {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.display_sync.write_wire(writer)?;
        self.max_fps.write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            display_sync: WireValue::read_wire(reader)?,
            max_fps: reader.u32()?,
        })
    }
}

impl WireValue for Change {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        match self {
            Self::Remove => writer.u8(0),
            Self::Apply(lut) => {
                writer.u8(1)?;
                lut.as_ref().write_wire(writer)
            }
        }
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Remove),
            1 => Ok(Self::Apply(Box::new(WireValue::read_wire(reader)?))),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for BackbufferContents {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Undefined => 0,
            Self::Preserved => 1,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Undefined),
            1 => Ok(Self::Preserved),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for BufferMapMode {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u8(match self {
            Self::Direct => 0,
            Self::Staged => 1,
        })
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        match reader.u8()? {
            0 => Ok(Self::Direct),
            1 => Ok(Self::Staged),
            _ => Err(WireError::InvalidValue),
        }
    }
}

impl WireValue for FrameDataFlags {
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        self.bits().write_wire(writer)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Self::from_bits(reader.u8()?).ok_or(WireError::InvalidValue)
    }
}

/// Replay completion published after the native frame's other fields are dropped.
pub struct ReplayCompletion {
    address: u64,
    rejected: bool,
}

impl Drop for ReplayCompletion {
    fn drop(&mut self) {
        // SAFETY: decode_packet validates the address and its caller retains the cell until
        // acknowledgment. This guard is last in the frame, after every borrowed data consumer.
        let cell = unsafe { &*(self.address as *const LeaseCompletion) };
        if self.rejected {
            cell.publish_rejected();
        } else {
            cell.publish();
        }
    }
}

#[cfg(any(test, debug_assertions))]
fn capture_ranges(op: &Op, ranges: &mut Vec<(u64, u64)>) {
    if let Op::Draw(draw) = op {
        capture_draw_ranges(draw, ranges);
    }
}

#[cfg(any(test, debug_assertions))]
fn capture_draw_ranges(draw: &crate::draw_data::DrawOp, ranges: &mut Vec<(u64, u64)>) {
    use crate::draw_data::{IndexSource, VertexSource};
    if let VertexSource::Bound { first, extra, .. } = &draw.vertex_source {
        for stream in core::iter::once(first).chain(extra.iter()) {
            ranges.push((stream.backing_ptr as u64, stream.backing_len as u64));
        }
    }
    if let IndexSource::Bound {
        backing_ptr,
        backing_len,
        ..
    } = &draw.index_source
    {
        ranges.push((*backing_ptr as u64, *backing_len as u64));
    }
}

fn op_record_bound(op: &Op) -> Result<usize, WireError> {
    let variable = match op {
        Op::ClearColorRects(value) => value
            .rects
            .len()
            .checked_mul(16)
            .ok_or(WireError::TooLarge)?,
        Op::ClearDepthStencilRects(value) => value
            .list
            .len()
            .checked_mul(16)
            .ok_or(WireError::TooLarge)?,
        _ => 0,
    };
    variable.checked_add(4096).ok_or(WireError::TooLarge)
}

/// Decode one immutable packet on the native encoder thread.
///
/// # Safety
///
/// The metadata inventory must describe authentic retained allocations from the paired PE
/// recorder. Metadata, chunk table, chunks, all inventoried owners and the completion cell must
/// remain immutable and live until completion. This is the packet's only native decoder.
/// Every record must be a complete, semantically valid typed record produced by the matching
/// recorder, and each ownership descriptor must occur exactly once. Production replay
/// relies on this internal producer contract rather than an independent preflight walk.
///
/// # Errors
///
/// Rejects invalid metadata, command spans or missing shader registrations. Diagnostic
/// builds also audit every operation against the matched producer contract.
pub unsafe fn prepare_packet(
    metadata: &[u8],
    operations: &[u8],
    completion: u64,
    resolve_program: impl FnMut(u64) -> Result<(ProgramId, DxsoProgram), WireError>,
) -> Result<ReplayPacket, WireError> {
    validate_range(
        completion,
        size_of::<LeaseCompletion>() as u64,
        align_of::<LeaseCompletion>(),
    )?;
    let guard = ReplayCompletion {
        address: completion,
        rejected: true,
    };
    // SAFETY: the caller supplies authentic retained metadata identities and inventory.
    let parsed = unsafe { parse_metadata(metadata)? };
    let mut table = WireReader::new(operations);
    let mut chunks = Vec::new();
    #[cfg(any(test, debug_assertions))]
    let diagnostic_inventory = parsed.has_inventory;
    #[cfg(any(test, debug_assertions))]
    let mut command_spans = parsed.inventory.command_spans.iter();
    while let Some(mut record) = table.next_record()? {
        if record.tag != FRAME_CHUNK_TAG {
            return Err(WireError::InvalidValue);
        }
        let address = record.payload.u64()?;
        let length = record.payload.u32()?;
        if address == 0 || length == 0 {
            return Err(WireError::InvalidValue);
        }
        if !record.payload.is_empty() {
            return Err(WireError::InvalidValue);
        }
        validate_range(address, u64::from(length), 1)?;
        #[cfg(any(test, debug_assertions))]
        if diagnostic_inventory && command_spans.next() != Some(&(address, u64::from(length))) {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: the caller retains inventoried immutable chunks; the full extent was checked.
        chunks.push((address, length as usize));
    }
    #[cfg(any(test, debug_assertions))]
    if diagnostic_inventory && command_spans.next().is_some() {
        return Err(WireError::InvalidValue);
    }
    prepare_chunks(parsed, chunks, guard, resolve_program)
}

#[cfg(test)]
unsafe fn decode_chunks(
    metadata: &[u8],
    chunks: &[&[u8]],
    completion: u64,
    queries: &mut QueryLeaseCache,
    resolve_program: impl FnMut(u64) -> Result<(ProgramId, DxsoProgram), WireError>,
) -> Result<FrameData, WireError> {
    validate_range(
        completion,
        size_of::<LeaseCompletion>() as u64,
        align_of::<LeaseCompletion>(),
    )?;
    let guard = ReplayCompletion {
        address: completion,
        rejected: true,
    };
    // SAFETY: the test caller supplies the same retained metadata contract.
    let parsed = unsafe { parse_metadata(metadata)? };
    if !parsed.has_inventory {
        return Err(WireError::InvalidValue);
    }
    let chunks = chunks
        .iter()
        .map(|chunk| (chunk.as_ptr() as u64, chunk.len()))
        .collect();
    let packet = prepare_chunks(parsed, chunks, guard, resolve_program)?;
    reconstruct_packet(packet, queries)
}

fn prepare_chunks(
    mut parsed: metadata::ParsedMetadata,
    chunks: Vec<(u64, usize)>,
    guard: ReplayCompletion,
    mut resolve_program: impl FnMut(u64) -> Result<(ProgramId, DxsoProgram), WireError>,
) -> Result<ReplayPacket, WireError> {
    let inventory = &mut parsed.inventory;
    let registrations = std::mem::take(&mut inventory.registrations);
    #[cfg(any(test, debug_assertions))]
    if parsed.has_inventory {
        // Diagnostic builds audit the matched producer contract before any adoption.
        inventory.registrations.clone_from(&registrations);
        let ranges = std::mem::take(&mut inventory.ranges);
        let mut validator = validation::Validation::new(inventory)?;
        for &(address, length) in &chunks {
            // SAFETY: the retained command table contains authentic immutable producer spans.
            let bytes = unsafe { core::slice::from_raw_parts(address as *const u8, length) };
            // SAFETY: producer metadata describes the typed allocation inventory.
            let mut reader = unsafe { WireReader::new_trusted_with_ranges(bytes, &ranges) };
            while let Some(mut record) = reader.next_record()? {
                validator.operation(&EncoderOpcode::try_from(record.tag)?, &mut record.payload)?;
            }
        }
        validator.finish()?;
    }
    let mut programs = Vec::with_capacity(registrations.len());
    for registration in registrations {
        programs.push((registration, resolve_program(registration)?));
    }
    // SAFETY: the paired producer constructs well-formed records with unique descriptors;
    // diagnostic builds additionally audit that contract before adoption.
    let frame = unsafe { parsed.adopt()? };
    // SAFETY: the frame's final completion guard retains the matched producer's immutable ranges.
    Ok(unsafe { ReplayPacket::new(frame, chunks, programs, guard) })
}

/// Reconstruct a validated packet for core clients that need an owned operation list.
///
/// # Safety
/// The complete semantically valid matched-producer packet and all inventoried allocations
/// stay immutable and live through final completion, with unique ownership descriptors.
/// # Errors
/// Reports metadata, shader-registration or checked record-decoding failures.
pub unsafe fn decode_packet(
    metadata: &[u8],
    operations: &[u8],
    completion: u64,
    queries: &mut QueryLeaseCache,
    resolve_program: impl FnMut(u64) -> Result<(ProgramId, DxsoProgram), WireError>,
) -> Result<FrameData, WireError> {
    // SAFETY: the caller supplies the same retained immutable packet contract.
    let packet = unsafe { prepare_packet(metadata, operations, completion, resolve_program)? };
    reconstruct_packet(packet, queries)
}

fn reconstruct_packet(
    mut packet: ReplayPacket,
    queries: &mut QueryLeaseCache,
) -> Result<FrameData, WireError> {
    while let Some(op) = packet.next_op(queries)? {
        // SAFETY: only the operation list changes; all snapshot scratch remains retained.
        unsafe { packet.frame_mut() }.ops.push(op);
    }
    packet
        .into_frame()
        .map(|frame| *frame)
        .map_err(|(error, _packet)| error)
}

unsafe fn parse_metadata(bytes: &[u8]) -> Result<metadata::ParsedMetadata, WireError> {
    // SAFETY: the caller retains authentic typed metadata; operation ranges are checked separately.
    let mut reader = unsafe { WireReader::new_trusted(bytes) };
    let mut record = reader.next_record()?.ok_or(WireError::Truncated)?;
    if record.tag != FRAME_METADATA_TAG || !reader.is_empty() {
        return Err(WireError::InvalidValue);
    }
    let result = metadata::read_metadata(&mut record.payload)?;
    if !record.payload.is_empty() {
        return Err(WireError::InvalidValue);
    }
    Ok(result)
}

struct UploadFields {
    info: TextureInfo,
    level: u32,
    destination_slice: u32,
    staging_index: u32,
    origin_x: u32,
    origin_y: u32,
    region_w: u32,
    region_h: u32,
    src_d3d_format: u32,
    src_pitch: u32,
    bytes_per_pixel: u32,
    depth: u32,
    slice_pitch: u32,
    release_staging: bool,
    upload_generation: u32,
    redirty: GuestRedirtyDescriptor,
}

impl UploadFields {
    fn read(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        Ok(Self {
            info: WireValue::read_wire(reader)?,
            level: reader.u32()?,
            destination_slice: reader.u32()?,
            staging_index: reader.u32()?,
            origin_x: reader.u32()?,
            origin_y: reader.u32()?,
            region_w: reader.u32()?,
            region_h: reader.u32()?,
            src_d3d_format: reader.u32()?,
            src_pitch: reader.u32()?,
            bytes_per_pixel: reader.u32()?,
            depth: reader.u32()?,
            slice_pitch: reader.u32()?,
            release_staging: WireValue::read_wire(reader)?,
            upload_generation: reader.u32()?,
            redirty: WireValue::read_wire(reader)?,
        })
    }

    fn validate(&self, logical_len: u64) -> Result<(), WireError> {
        use mtld3d_shared::{blit_geometry::source_rows_end, mtl::TextureCreateFlags};
        if self.level >= 32 || self.level >= self.info.levels || self.depth == 0 {
            return Err(WireError::InvalidValue);
        }
        let cube = self
            .info
            .create_flags
            .contains(TextureCreateFlags::TYPE_CUBE);
        if self.destination_slice >= if cube { 6 } else { 1 } {
            return Err(WireError::InvalidValue);
        }
        let expected_index = self
            .destination_slice
            .checked_mul(self.info.levels)
            .and_then(|base| base.checked_add(self.level))
            .ok_or(WireError::InvalidValue)?;
        if self.staging_index != expected_index {
            return Err(WireError::InvalidValue);
        }
        let source_bpp = if matches!(
            self.src_d3d_format,
            mtld3d_types::D3DFMT_YV12 | mtld3d_types::D3DFMT_NV12
        ) {
            1
        } else if let Some(bytes) = crate::format::depth_format_bytes_per_pixel(self.src_d3d_format)
        {
            bytes
        } else {
            crate::format::map_d3d_format(self.src_d3d_format)
                .ok_or(WireError::InvalidValue)?
                .bytes_per_pixel()
        };
        if self.bytes_per_pixel != source_bpp {
            return Err(WireError::InvalidValue);
        }
        let mip_depth = (self.info.depth >> self.level).max(1);
        if self.depth > mip_depth {
            return Err(WireError::InvalidValue);
        }
        // Planar API capture describes the full storage texture, including chroma rows.
        // It is never a partial luma upload, even when the application dirtied a subrectangle.
        if matches!(
            self.src_d3d_format,
            mtld3d_types::D3DFMT_YV12 | mtld3d_types::D3DFMT_NV12
        ) && (self.level != 0
            || self.origin_x != 0
            || self.origin_y != 0
            || self.region_w != self.info.width
            || self.region_h != self.info.height
            || u64::from(self.src_pitch) * u64::from(self.info.height) > logical_len)
        {
            return Err(WireError::InvalidValue);
        }
        let end_x = self
            .origin_x
            .checked_add(self.region_w)
            .ok_or(WireError::InvalidValue)?;
        let end_y = self
            .origin_y
            .checked_add(self.region_h)
            .ok_or(WireError::InvalidValue)?;
        let (row_bytes, row_end) = if self.bytes_per_pixel == 0 {
            let format = crate::format::map_d3d_format(self.src_d3d_format)
                .ok_or(WireError::InvalidValue)?;
            if !format.is_compressed() {
                return Err(WireError::InvalidValue);
            }
            let mip_width = (self.info.width >> self.level).max(1);
            let mip_height = (self.info.height >> self.level).max(1);
            if end_x > mip_width || end_y > mip_height {
                return Err(WireError::InvalidValue);
            }
            let row_bytes = u64::from(mip_width.div_ceil(format.block_width()))
                * u64::from(format.block_bytes());
            (row_bytes, mip_height.div_ceil(format.block_height()))
        } else {
            (u64::from(end_x) * u64::from(self.bytes_per_pixel), end_y)
        };
        if row_bytes > u64::from(self.src_pitch) {
            return Err(WireError::InvalidValue);
        }
        let first_slice_end =
            source_rows_end(0, self.src_pitch, row_end).ok_or(WireError::InvalidValue)?;
        let required = if self.depth > 1 {
            if u64::from(self.slice_pitch) < first_slice_end {
                return Err(WireError::InvalidValue);
            }
            u64::from(self.slice_pitch)
                .checked_mul(u64::from(self.depth - 1))
                .and_then(|last| last.checked_add(first_slice_end))
                .ok_or(WireError::InvalidValue)?
        } else {
            first_slice_end
        };
        if required > logical_len {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }
}

fn read_upload(reader: &mut WireReader<'_>) -> Result<TextureUploadJob, WireError> {
    let fields = UploadFields::read(reader)?;
    let descriptor = GuestPageDescriptor::read_wire(reader)?;
    fields.validate(descriptor.wire_fields()[2])?;
    // SAFETY: full packet validation proved this retained lease is adopted exactly once.
    let staging = unsafe { descriptor.adopt_read()? };
    // SAFETY: full packet validation proved this feedback mailbox is retained and unique.
    let redirty = unsafe { fields.redirty.adopt()? };
    Ok(TextureUploadJob {
        info: fields.info,
        staging,
        level: fields.level,
        destination_slice: fields.destination_slice,
        staging_index: fields.staging_index as usize,
        origin_x: fields.origin_x,
        origin_y: fields.origin_y,
        region_w: fields.region_w,
        region_h: fields.region_h,
        src_d3d_format: fields.src_d3d_format,
        src_pitch: fields.src_pitch,
        bytes_per_pixel: fields.bytes_per_pixel,
        depth: fields.depth,
        slice_pitch: fields.slice_pitch,
        release_staging: fields.release_staging,
        upload_generation: fields.upload_generation,
        redirty,
    })
}
