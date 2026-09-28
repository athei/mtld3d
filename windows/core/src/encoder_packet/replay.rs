//! Replay of matched-producer command records while their frame lease remains live.

use mtld3d_shared::command_header::{
    COMMAND_ALIGNMENT, COMMAND_HEADER_BYTES, CommandHeader, CommandRegion,
};

use super::{DrawReader, EncoderOpcode, NativeFrame, ReplayCompletion, WireError};

/// Native snapshot state for this frame.
pub struct ReplayState {
    draws: DrawReader,
}

impl ReplayState {
    pub const fn draw_reader(&mut self) -> &mut DrawReader {
        &mut self.draws
    }
}

/// An immutable command stream and the native state retaining its frame lease.
pub struct ReplayPacket {
    frame: NativeFrame,
    commands: CommandCursor,
    state: ReplayState,
    exhausted: bool,
    failure: Option<WireError>,
    // Last: reject only after the native owners have been released.
    completion: Option<ReplayCompletion>,
}

impl ReplayPacket {
    /// The caller retains semantically valid matched-producer commands and all referenced storage.
    pub(super) const unsafe fn new(
        frame: NativeFrame,
        commands: CommandCursor,
        completion: ReplayCompletion,
    ) -> Self {
        Self {
            frame,
            commands,
            state: ReplayState {
                // SAFETY: the matched producer retains all typed records under this packet lease.
                draws: unsafe { DrawReader::new() },
            },
            exhausted: false,
            failure: None,
            completion: Some(completion),
        }
    }

    #[must_use]
    pub const fn frame(&self) -> &NativeFrame {
        &self.frame
    }

    /// Execute one command while its borrowed record and genuine runtime owners remain live.
    ///
    /// # Safety
    /// The consumer must not release the frame's lease, or clear, replace or release the
    /// storage it decodes snapshots into, while any snapshot, encoder state or submitted
    /// command can still reference them.
    /// Retained tokens may escape the callback only while that same frame lease remains live.
    ///
    /// # Errors
    /// Reports malformed command fields or a consumer error, poisoning further replay.
    pub unsafe fn replay_one(
        &mut self,
        consume: impl FnOnce(
            CommandView<'_>,
            &mut NativeFrame,
            &mut ReplayState,
        ) -> Result<(), WireError>,
    ) -> Result<bool, WireError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = (|| {
            // SAFETY: this packet retains the immutable region table and all command regions.
            let Some(record) = (unsafe { self.commands.next_record()? }) else {
                self.exhausted = true;
                return Ok(false);
            };
            let view = CommandView {
                opcode: EncoderOpcode::try_from(record.opcode)?,
                operand: record.operand,
                payload: record.payload,
            };
            consume(view, &mut self.frame, &mut self.state)?;
            Ok(true)
        })();
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    /// Transfer an exhausted frame to submit without releasing its immutable storage lease.
    ///
    /// # Errors
    /// Returns the retained packet if replay failed or has not reached the end of its regions.
    pub fn into_frame(mut self) -> Result<NativeFrame, (WireError, Box<Self>)> {
        if self.failure.is_some() || !self.exhausted || !self.commands.is_complete() {
            return Err((WireError::InvalidValue, Box::new(self)));
        }
        let Some(mut completion) = self.completion.take() else {
            return Err((WireError::InvalidValue, Box::new(self)));
        };
        completion.rejected = false;
        self.frame.replay_completion = Some(completion);
        Ok(self.frame)
    }
}

/// A command borrowed only for its immediate native dispatch.
pub struct CommandView<'a> {
    opcode: EncoderOpcode,
    operand: u16,
    payload: &'a [u8],
}

impl CommandView<'_> {
    #[must_use]
    pub const fn opcode(&self) -> &EncoderOpcode {
        &self.opcode
    }
    #[must_use]
    pub const fn operand(&self) -> u16 {
        self.operand
    }
    #[must_use]
    pub const fn payload(&self) -> &[u8] {
        self.payload
    }

    /// Borrow inline floating-point constant rows from this retained command.
    ///
    /// # Errors
    /// Rejects a non-constant opcode, invalid row extent, or oversized range.
    pub fn constants(&self) -> Result<ConstantRange, WireError> {
        constant_range(&self.opcode, self.operand, self.payload)
    }
}

pub struct ConstantRange {
    pub start_row: u16,
    pub rows: u16,
    pub data: crate::draw_data::ScratchSlice,
}

/// Sequential cursor over command regions retained by the frame lease.
pub(super) struct CommandCursor {
    table_address: u64,
    table_bytes: usize,
    table_offset: usize,
    region_address: u64,
    region_bytes: u32,
    region_offset: u32,
}

impl CommandCursor {
    /// The caller retains this descriptor table and every region it describes unchanged.
    pub(super) unsafe fn new(table: &[u8]) -> Result<Self, WireError> {
        if (!table.is_empty()
            && !(table.as_ptr() as usize).is_multiple_of(align_of::<CommandRegion>()))
            || !table.len().is_multiple_of(size_of::<CommandRegion>())
        {
            return Err(WireError::InvalidValue);
        }
        Ok(Self {
            table_address: table.as_ptr() as u64,
            table_bytes: table.len(),
            table_offset: 0,
            region_address: 0,
            region_bytes: 0,
            region_offset: 0,
        })
    }

    pub(super) const fn is_complete(&self) -> bool {
        self.table_offset == self.table_bytes && self.region_offset == self.region_bytes
    }

    /// The caller retains each authentic region and its payload unchanged.
    pub(super) unsafe fn next_record(&mut self) -> Result<Option<CommandRecord<'_>>, WireError> {
        if self.region_offset == self.region_bytes {
            if self.table_offset == self.table_bytes {
                return Ok(None);
            }
            let descriptor = (self.table_address as *const u8).wrapping_add(self.table_offset);
            // SAFETY: new checked the table alignment and complete fixed-size extent;
            // the frame retains these initialized integer-only descriptors.
            let descriptor =
                std::ptr::with_exposed_provenance::<CommandRegion>(descriptor.expose_provenance());
            // SAFETY: new checked the descriptor alignment and extent before this typed borrow.
            let descriptor = unsafe { &*descriptor };
            let address = descriptor.address;
            let length = descriptor.used_bytes;
            if descriptor.reserved != 0
                || length == 0
                || !(length as usize).is_multiple_of(COMMAND_ALIGNMENT)
            {
                return Err(WireError::InvalidValue);
            }
            super::validate_range(address, u64::from(length), COMMAND_ALIGNMENT)?;
            self.table_offset += mtld3d_shared::command_header::COMMAND_REGION_BYTES;
            self.region_address = address;
            self.region_bytes = length;
            self.region_offset = 0;
        }
        let remaining = self.region_bytes - self.region_offset;
        if (remaining as usize) < COMMAND_HEADER_BYTES {
            return Err(WireError::Truncated);
        }
        let address = self.region_address + u64::from(self.region_offset);
        // SAFETY: the retained region contains this aligned complete integer-only header;
        // the matched producer retains its initialized typed allocation through final submit.
        let header = unsafe { &*(address as *const CommandHeader) };
        let length = header.record_bytes;
        if (length as usize) < COMMAND_HEADER_BYTES {
            return Err(WireError::InvalidValue);
        }
        let alignment_mask =
            u32::try_from(COMMAND_ALIGNMENT - 1).map_err(|_| WireError::TooLarge)?;
        let stride = length
            .checked_add(alignment_mask)
            .ok_or(WireError::TooLarge)?
            & !alignment_mask;
        if stride > remaining {
            return Err(WireError::Truncated);
        }
        let payload_start = (address as *const u8).wrapping_add(COMMAND_HEADER_BYTES);
        // SAFETY: the exact logical payload fits within the retained command region.
        let payload = unsafe {
            core::slice::from_raw_parts(payload_start, length as usize - COMMAND_HEADER_BYTES)
        };
        self.region_offset += stride;
        Ok(Some(CommandRecord {
            opcode: header.opcode,
            operand: header.operand,
            payload,
        }))
    }
}

pub(super) struct CommandRecord<'a> {
    pub(super) opcode: u16,
    pub(super) operand: u16,
    pub(super) payload: &'a [u8],
}

fn constant_range(
    opcode: &EncoderOpcode,
    start_row: u16,
    payload: &[u8],
) -> Result<ConstantRange, WireError> {
    if !matches!(
        opcode,
        EncoderOpcode::SetVsConstRange
            | EncoderOpcode::SetPsConstRange
            | EncoderOpcode::SetFfVsConstRange
    ) {
        return Err(WireError::InvalidValue);
    }
    let row_bytes = size_of::<[f32; 4]>();
    let rows = payload.len() / row_bytes;
    if !payload.len().is_multiple_of(row_bytes)
        || rows > crate::draw_data::CONSTANT_ROWS
        || usize::from(start_row) + rows > crate::draw_data::CONSTANT_ROWS
    {
        return Err(WireError::InvalidValue);
    }
    let rows = u16::try_from(rows).map_err(|_| WireError::TooLarge)?;
    let data = if rows == 0 {
        crate::draw_data::ScratchSlice::EMPTY
    } else {
        let ptr =
            core::ptr::NonNull::new(payload.as_ptr().cast_mut()).ok_or(WireError::InvalidValue)?;
        // SAFETY: only retained command views invoke this helper; the immutable inline bytes
        // outlive all native tokens through the frame's final submit completion owner.
        unsafe {
            crate::draw_data::ScratchSlice::from_raw_parts(
                ptr,
                u32::try_from(payload.len()).map_err(|_| WireError::TooLarge)?,
            )
        }
    };
    Ok(ConstantRange {
        start_row,
        rows,
        data,
    })
}
