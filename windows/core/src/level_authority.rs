//! Which copy of a texture level holds the pixels the level is defined by.
//!
//! A `StretchRect` into a texture-backed surface writes the destination's Metal
//! texture and never its CPU staging, so from that point the level's pixels live
//! on the GPU alone. Every path that writes a level's staging asks
//! [`LevelAuthorityMask::plan_write`] what to do about that first, and the answer
//! records that the staging defines the level once the write lands.
//!
//! Two writes exist. One that covers the whole level defines every byte of it,
//! so the GPU's copy is dead and no read back is worth paying for: a
//! `D3DLOCK_DISCARD` map promises exactly that, and a whole-level copy performs
//! it. One that leaves pixels untouched needs the level read back first, or the
//! staging the next upload pushes holds the GPU's pixels nowhere.

/// What a CPU write of a texture level has to do about the GPU's copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePlan {
    /// Write into the staging: it already holds every byte of the level.
    WriteStaging,
    /// Read the level back from the GPU, then write into the staging.
    ///
    /// The GPU holds pixels the staging does not and the write leaves some of
    /// them untouched, so only the read back makes the staging whole.
    ReadBackFirst,
    /// Write into the staging with no read back: the write defines every byte.
    Overwrite,
}

/// Which copy holds the pixels of each mip of one texture.
///
/// One bit per level, set while the Metal texture holds pixels the level's
/// staging does not. A D3D9 mip chain tops out at 15 levels, so the mask covers
/// every level a texture can carry; the bound is a guard, not a limit anything
/// reaches.
#[derive(Debug, Default)]
pub struct LevelAuthorityMask {
    gpu: u32,
}

impl LevelAuthorityMask {
    /// A texture whose staging defines every level.
    #[must_use]
    pub const fn new() -> Self {
        Self { gpu: 0 }
    }

    /// Record that the GPU wrote `level` and its staging did not receive the write.
    pub const fn gpu_wrote(&mut self, level: usize) {
        if level < u32::BITS as usize {
            self.gpu |= 1u32 << level;
        }
    }

    /// Whether the Metal texture holds pixels `level`'s staging does not.
    #[must_use]
    pub const fn gpu_holds(&self, level: usize) -> bool {
        level < u32::BITS as usize && self.gpu & (1u32 << level) != 0
    }

    /// Plan a CPU write of `level` and record that its staging defines it after.
    ///
    /// `whole_level` says the write covers every byte of the level. The claim is
    /// released whichever branch the caller takes, so a read back that cannot
    /// run costs one diagnostic rather than one per write, and a level the
    /// caller overwrites costs no read back at all.
    pub const fn plan_write(&mut self, level: usize, whole_level: bool) -> WritePlan {
        if !self.gpu_holds(level) {
            return WritePlan::WriteStaging;
        }
        if level < u32::BITS as usize {
            self.gpu &= !(1u32 << level);
        }
        if whole_level {
            WritePlan::Overwrite
        } else {
            WritePlan::ReadBackFirst
        }
    }
}

#[cfg(test)]
mod tests;
