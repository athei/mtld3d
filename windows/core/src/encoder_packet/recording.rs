//! Command spans allocated directly in the frame's payload arena.

use mtld3d_shared::encoder_wire::{FrameSlab, WireError, WireWriter};

use crate::scratch::ScratchArena;

/// Descriptor table for command runs within a single frame arena.
///
/// This object owns no command or payload allocations. The associated frame
/// arena retains all bytes until replay ends, and is recycled with this table.
pub struct RecordedSpans {
    descriptors: FrameSlab,
    spans: Vec<(u64, usize, u64)>,
}

impl RecordedSpans {
    pub const fn new() -> Self {
        Self {
            descriptors: FrameSlab::new(),
            spans: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.descriptors.clear();
        self.spans.clear();
    }

    pub fn descriptor_bytes(&self) -> &[u8] {
        self.descriptors.as_bytes()
    }

    pub fn ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.spans
            .iter()
            .map(|&(address, length, _)| (address, length as u64))
    }

    pub fn push_record(
        &mut self,
        arena: &mut ScratchArena,
        tag: u16,
        bound: usize,
        write: impl FnOnce(&mut WireWriter<'_>) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        let (address, length, allocation) = arena.write_record(tag, bound, write)?;
        if let Some((previous, size, previous_allocation)) = self.spans.last_mut()
            && *previous_allocation == allocation
            && previous.checked_add(*size as u64) == Some(address)
        {
            *size = size.checked_add(length).ok_or(WireError::TooLarge)?;
            let length = u32::try_from(*size).map_err(|_| WireError::TooLarge)?;
            self.descriptors.replace_last_u32(length);
        } else {
            self.spans
                .try_reserve(1)
                .map_err(|_| WireError::AllocationFailed)?;
            self.descriptors
                .push_record(super::FRAME_CHUNK_TAG, |writer| {
                    writer.u64(address)?;
                    writer.u32(u32::try_from(length).map_err(|_| WireError::TooLarge)?)
                })?;
            self.spans.push((address, length, allocation));
        }
        Ok(())
    }
}
