//! Frame recording and borrowed native replay across the PE/Unix boundary.
//!
//! Operation bytes are captured as API calls record them. Packet admission writes only
//! frame metadata; guest allocations and replies remain owned by their allocating runtime.

use mtld3d_shared::{
    encoder_protocol::EncoderOpcode,
    encoder_wire::{LeaseCompletion, WireError},
};

use crate::{
    encoder_data::Op,
    encoder_draw::{DrawReader, DrawWriter},
    encoder_reply::{ReplyBool, ReplyU64},
    guest_pages::{GuestPageLease, GuestRetirementLease},
    guest_queries::GuestQueryLease,
    scratch::ScratchArena,
    upload_redirty::GuestRedirtyLease,
};

mod frame;
pub mod metadata;
pub use frame::{NativeFrame, NativeVbibRetention};

mod owner;
mod record_controls;
pub use record_controls::CaptureControl;

mod recording;
use recording::RecordedCommands;
mod replay;
#[cfg(test)]
mod tests;

use metadata::MetadataStorage;
pub use owner::{FramePacket, PacketLease};
pub use replay::{CommandView, ConstantRange, ReplayPacket, ReplayState};

/// Keeps a published gamma table at its original stable heap address while the ledger grows.
struct GammaTableOwner {
    table: Box<[u16; crate::gamma::LUT_LANES]>,
}

/// API-side operation bytes and owners awaiting native acknowledgment.
pub struct FrameRecorder {
    gamma_tables: Vec<GammaTableOwner>,
    pagebox_pool: Option<&'static crate::page_box_pool::PageBoxPool>,
    completion_pool: crate::guest_completions::CompletionPool,
    slab: RecordedCommands,
    metadata: MetadataStorage,
    draws: DrawWriter,
    pages: Vec<GuestPageLease>,
    retirements: Vec<GuestRetirementLease>,
    queries: Vec<GuestQueryLease>,
    redirties: Vec<GuestRedirtyLease>,
    replies_u64: Vec<ReplyU64>,
    replies_bool: Vec<ReplyBool>,
    error: Option<WireError>,
    count: usize,
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
            gamma_tables: Vec::new(),
            pagebox_pool: None,
            completion_pool,
            slab: RecordedCommands::new(),
            metadata: MetadataStorage::new(),
            draws: DrawWriter::new(),
            pages: Vec::new(),
            retirements: Vec::new(),
            queries: Vec::new(),
            redirties: Vec::new(),
            replies_u64: Vec::new(),
            replies_bool: Vec::new(),
            error: None,
            count: 0,
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
        match op {
            Op::Draw(draw) => self.record_draw(scratch, &draw),
            Op::SetVsConstRange {
                start_row,
                rows,
                data,
            } => self.record_vs_constants(scratch, start_row, rows, data),
            Op::SetPsConstRange {
                start_row,
                rows,
                data,
            } => self.record_ps_constants(scratch, start_row, rows, data),
            Op::SetFfVsConstRange {
                start_row,
                rows,
                data,
            } => self.record_ff_vs_constants(scratch, start_row, rows, data),
            Op::SetViewport(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::SetVertexSampler(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::SetVertexTexture(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::BindDepth(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::BindColor(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::GenerateMipmapsOrdered(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::UnbindExtraColor(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::DestroyTexture(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::NoteColorRead(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ResolveDepthSurface(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::StretchBlit(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ColorFill(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::CarryDepth(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ClearColor(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ClearColorRects(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ClearDepthStencilRects(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ClearDepthStencil(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ResolveDynamicDepth(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ResolveDepthTexture(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::AdoptProgram(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::RetireColor(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::RetireDepth(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::UploadColor(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::UploadResampled(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::GenerateMipmaps(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::SetDumpDraw(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ReadColorHandle(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ReadTextureHandle(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ReadTextureColorHandle(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::ReadDeviceBuffer(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::BeginVisibility(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::EndVisibility(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::UploadTexture(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::UploadTextureAndMips(value) => {
                #[cfg(not(windows))]
                let value = *value;
                self.record_typed(scratch, value)
            }
            Op::StageUpload {
                buffer_id,
                page_box,
                dst_offset,
                size,
            } => self.record_typed(
                scratch,
                crate::encoder_data::StageUploadOp {
                    buffer_id,
                    page_box,
                    dst_offset,
                    size,
                },
            ),
            op @ (Op::RegisterProgram(_) | Op::SetSnapshot(_)) => {
                self.rejected_ops.push(op);
                self.finish_record(Err(WireError::InvalidValue))
            }
        }
    }

    /// Construct a command directly from its typed API capture.
    ///
    /// # Errors
    /// Returns the latched error or a capture failure, retaining all consumed owners.
    pub fn record_typed<T: CaptureControl>(
        &mut self,
        scratch: &mut ScratchArena,
        value: T,
    ) -> Result<(), WireError> {
        if let Some(error) = self.error {
            self.rejected_ops.push(value.into_rejected());
            return Err(error);
        }
        let registration = value.registration();
        let result = value.capture(self, scratch);
        if result.is_ok()
            && let Some(id) = registration
        {
            self.registrations.push(id);
        }
        self.finish_record(result)
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
        let result = crate::encoder_draw::draw_payload_size(draw).and_then(|length| {
            self.slab.push_fixed_record(
                scratch,
                u16::from(EncoderOpcode::Draw),
                0,
                length,
                |destination| crate::encoder_draw::write_draw_into(draw, destination, length),
            )
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
        self.record_constant_bytes(scratch, opcode, start_row, rows, data.as_slice())
    }

    /// Capture constant source bytes directly into their final command payload.
    ///
    /// # Errors
    /// Rejects an invalid row extent, source length, opcode or a latched error.
    pub fn record_constant_bytes(
        &mut self,
        scratch: &mut ScratchArena,
        opcode: EncoderOpcode,
        start_row: u16,
        rows: u16,
        bytes: &[u8],
    ) -> Result<(), WireError> {
        self.record_constant_destination(scratch, opcode, start_row, rows, |destination| {
            let source = bytes
                .get(..destination.len())
                .ok_or(WireError::InvalidValue)?;
            destination.copy_from_slice(source);
            Ok(())
        })
    }

    /// Build fixed-function rows directly in their final retained destination.
    ///
    /// # Errors
    /// Returns the builder's failure, invalid row extent or a latched error.
    pub fn record_ff_vs_destination(
        &mut self,
        scratch: &mut ScratchArena,
        start_row: u16,
        rows: u16,
        fill: impl FnOnce(&mut [core::mem::MaybeUninit<[f32; 4]>]) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        self.record_constant_destination(
            scratch,
            EncoderOpcode::SetFfVsConstRange,
            start_row,
            rows,
            |destination| {
                // SAFETY: MaybeUninit rows accept every existing byte pattern.
                let (prefix, destination, suffix) =
                    unsafe { destination.align_to_mut::<core::mem::MaybeUninit<[f32; 4]>>() };
                if !prefix.is_empty()
                    || !suffix.is_empty()
                    || destination.len() != usize::from(rows)
                {
                    return Err(WireError::InvalidValue);
                }
                fill(destination)
            },
        )
    }

    fn record_constant_destination(
        &mut self,
        scratch: &mut ScratchArena,
        opcode: EncoderOpcode,
        start_row: u16,
        rows: u16,
        fill: impl FnOnce(&mut [u8]) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if usize::from(start_row) + usize::from(rows) > crate::draw_data::CONSTANT_ROWS
            || !matches!(
                opcode,
                EncoderOpcode::SetVsConstRange
                    | EncoderOpcode::SetPsConstRange
                    | EncoderOpcode::SetFfVsConstRange
            )
        {
            return self.finish_record(Err(WireError::InvalidValue));
        }
        let result = self.slab.push_fixed_record(
            scratch,
            u16::from(opcode),
            start_row,
            usize::from(rows) * 16,
            fill,
        );
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
        let result = self.slab.push_initialized_record(
            scratch,
            u16::from(EncoderOpcode::SetSnapshot),
            0,
            crate::encoder_draw::SNAPSHOT_DELTA_MAX_BYTES,
            |destination| self.draws.capture_snapshot(delta, destination),
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
        self.gamma_tables.clear();
        self.slab.clear();
        self.metadata.clear();
        self.draws.clear();
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

fn validate_range(address: u64, length: u64, alignment: usize) -> Result<(), WireError> {
    let address = usize::try_from(address).map_err(|_| WireError::InvalidValue)?;
    let length = usize::try_from(length).map_err(|_| WireError::InvalidValue)?;
    if address == 0 || !address.is_multiple_of(alignment) || address.checked_add(length).is_none() {
        return Err(WireError::InvalidValue);
    }
    Ok(())
}

/// Replay completion published after the native frame's other fields are dropped.
pub struct ReplayCompletion {
    address: u64,
    rejected: bool,
}

impl Drop for ReplayCompletion {
    fn drop(&mut self) {
        // SAFETY: prepare_packet validates the address and its caller retains the cell until
        // acknowledgment. This guard is last in the frame, after every borrowed data consumer.
        let cell = unsafe { &*(self.address as *const LeaseCompletion) };
        if self.rejected {
            cell.publish_rejected();
        } else {
            cell.publish();
        }
    }
}

/// Borrow one immutable packet on the native encoder thread.
///
/// # Safety
///
/// The metadata must describe authentic retained allocations from the paired PE
/// recorder. Metadata, command regions and payloads remain immutable until completion.
/// Their owners and the completion mailbox remain live throughout native replay.
/// This is the packet's only native consumer.
/// Every record must be a complete, semantically valid typed record produced by the matching
/// recorder, and each ownership descriptor must occur exactly once. Production replay
/// relies on this internal producer contract rather than an independent preflight walk.
///
/// # Errors
///
/// Rejects invalid metadata or command region descriptors.
pub unsafe fn prepare_packet(
    metadata: &[u8],
    operations: &[u8],
    completion: u64,
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
    // SAFETY: the publisher retains the aligned canonical header and every typed array.
    let view = unsafe { metadata::FrameView::from_bytes(metadata)? };
    // SAFETY: the same packet retains its immutable region descriptor table.
    let commands = unsafe { replay::CommandCursor::new(operations)? };
    // SAFETY: the sole admitted consumer retains this validated immutable metadata lease.
    let frame = unsafe { NativeFrame::new(&view) };
    // SAFETY: the paired producer supplies semantically valid typed commands with unique
    // ownership descriptors. Their storage remains retained through final submit completion.
    Ok(unsafe { ReplayPacket::new(frame, commands, guard) })
}
