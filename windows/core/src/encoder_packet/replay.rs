//! Replay of matched-producer command records while their frame lease remains live.

use super::{
    DrawReader, DxsoProgram, EncoderOpcode, FrameData, Op, ProgramId, QueryLeaseCache,
    ReplayCompletion, WireError, WireReader, read_operation,
};

/// An immutable typed command stream and its sole native storage-lease owner.
///
/// The frame keeps decoded snapshots through submit's final CPU read. Dropping an unfinished
/// stream rejects its lease after releasing native owners, so PE quarantines it until teardown.
pub struct ReplayPacket {
    frame: Box<FrameData>,
    chunks: Vec<(u64, usize)>,
    programs: Vec<(u64, (ProgramId, DxsoProgram))>,
    draws: DrawReader,
    chunk: usize,
    offset: usize,
    failure: Option<WireError>,
    completion: Option<ReplayCompletion>,
}

impl ReplayPacket {
    /// The caller retains complete well-formed matched-producer records and their storage.
    pub(super) unsafe fn new(
        frame: FrameData,
        chunks: Vec<(u64, usize)>,
        programs: Vec<(u64, (ProgramId, DxsoProgram))>,
        completion: ReplayCompletion,
    ) -> Self {
        Self {
            frame: Box::new(frame),
            chunks,
            programs,
            // SAFETY: the matched producer guarantees typed immutable retained records.
            draws: unsafe { DrawReader::new() },
            chunk: 0,
            offset: 0,
            failure: None,
            completion: Some(completion),
        }
    }

    #[must_use]
    pub fn frame(&self) -> &FrameData {
        &self.frame
    }

    /// Borrow frame metadata and pending resource queues during replay.
    ///
    /// # Safety
    /// Do not clear, replace or move scratch storage while decoded snapshots refer to it.
    pub unsafe fn frame_mut(&mut self) -> &mut FrameData {
        &mut self.frame
    }

    /// Decode one typed operation directly into its native replay call.
    ///
    /// # Errors
    /// Reports a violation of the matched producer record contract.
    pub fn next_op(&mut self, queries: &mut QueryLeaseCache) -> Result<Option<Op>, WireError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self.next_record(queries);
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    fn next_record(&mut self, queries: &mut QueryLeaseCache) -> Result<Option<Op>, WireError> {
        while let Some(&(address, length)) = self.chunks.get(self.chunk) {
            if self.offset == length {
                self.chunk += 1;
                self.offset = 0;
                continue;
            }
            // SAFETY: this packet's final guard retains the immutable producer chunk.
            let start = (address as *const u8).wrapping_add(self.offset);
            // SAFETY: offset is advanced only within complete retained records.
            let bytes = unsafe { core::slice::from_raw_parts(start, length - self.offset) };
            // SAFETY: the matched producer records authentic typed ranges in immutable records;
            // the final guard retains every referenced allocation through replay.
            let mut reader = unsafe { WireReader::new_trusted(bytes) };
            let mut record = reader.next_record()?.ok_or(WireError::Truncated)?;
            self.offset += bytes.len() - reader.remaining_len();
            let programs = &mut self.programs;
            let op = read_operation(
                &EncoderOpcode::try_from(record.tag)?,
                &mut record.payload,
                &mut self.draws,
                &mut self.frame.scratch,
                queries,
                &mut |registration| {
                    let index = programs
                        .iter()
                        .position(|(id, _)| *id == registration)
                        .ok_or(WireError::InvalidValue)?;
                    Ok(programs.swap_remove(index).1)
                },
            )?;
            if !record.payload.is_empty() {
                return Err(WireError::InvalidValue);
            }
            return Ok(Some(op));
        }
        Ok(None)
    }

    /// Transfer the fully consumed frame to the submit owner without releasing its lease.
    ///
    /// # Errors
    /// Returns the retained packet on failure so snapshot users can quarantine its storage.
    /// Rejects an attempt to finish before every record has been consumed.
    pub fn into_frame(mut self) -> Result<Box<FrameData>, (WireError, Box<Self>)> {
        if self.failure.is_some() || self.chunk != self.chunks.len() || !self.programs.is_empty() {
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
