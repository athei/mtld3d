//! Flat command regions retained by the frame's single arena owner.

use mtld3d_shared::{command_header::CommandRegion, encoder_wire::WireError};

use crate::scratch::ScratchArena;

#[cfg(test)]
mod tests;

/// Reusable fixed-width descriptors for contiguous command regions.
pub struct RecordedCommands {
    regions: Vec<CommandRegion>,
    last_region: u64,
}

impl RecordedCommands {
    pub const fn new() -> Self {
        Self {
            regions: Vec::new(),
            last_region: 0,
        }
    }

    pub fn clear(&mut self) {
        self.regions.clear();
        self.last_region = 0;
    }

    pub const fn descriptor_bytes(&self) -> &[u8] {
        // SAFETY: CommandRegion has no padding, every scalar field is initialized,
        // and this borrow prevents changes to the descriptor vector.
        unsafe {
            core::slice::from_raw_parts(
                self.regions.as_ptr().cast::<u8>(),
                core::mem::size_of_val(self.regions.as_slice()),
            )
        }
    }

    #[cfg(test)]
    pub fn ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.regions
            .iter()
            .map(|region| (region.address, u64::from(region.used_bytes)))
    }

    pub fn push_fixed_record(
        &mut self,
        arena: &mut ScratchArena,
        tag: u16,
        operand: u16,
        payload_bytes: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        self.push_initialized_record(arena, tag, operand, payload_bytes, |destination| {
            fill(destination)?;
            Ok(payload_bytes)
        })
    }

    pub fn push_initialized_record(
        &mut self,
        arena: &mut ScratchArena,
        tag: u16,
        operand: u16,
        bound: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<usize, WireError>,
    ) -> Result<(), WireError> {
        let command = arena.write_command(tag, operand, bound, fill)?;
        let length = u32::try_from(command.region_bytes).map_err(|_| WireError::TooLarge)?;
        if self.last_region == command.region_address {
            if let Some(region) = self.regions.last_mut() {
                region.used_bytes = length;
            }
        } else {
            self.regions
                .try_reserve(1)
                .map_err(|_| WireError::AllocationFailed)?;
            self.regions.push(CommandRegion {
                address: command.region_address,
                used_bytes: length,
                reserved: 0,
            });
            self.last_region = command.region_address;
        }
        Ok(())
    }
}
