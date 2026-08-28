//! Which copy of a texture subresource holds the pixels the subresource is defined by.
//!
//! A `StretchRect` or a `ColorFill` into a texture-backed surface writes the
//! destination's Metal texture and never its CPU staging, so from that point
//! the subresource's pixels live on the GPU alone. Every path that writes a
//! subresource's staging asks [`LevelAuthorityMask::plan_write`] what to do
//! about that first, and the answer records that the staging defines the
//! subresource once the write lands.
//!
//! A subresource is a (face, level) pair: a 2D or volume texture uses face 0
//! alone, a cube map all six. Face and level are tracked separately because a
//! write of one cube face says nothing about the other five.
//!
//! Two writes exist. One that covers the whole level defines every byte of it,
//! so the GPU's copy is dead and no read back is worth paying for: a
//! `D3DLOCK_DISCARD` map promises exactly that, and a whole-level copy performs
//! it. One that leaves pixels untouched needs the level read back first, or the
//! staging the next upload pushes holds the GPU's pixels nowhere.

/// What a CPU write of a texture subresource has to do about the GPU's copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePlan {
    /// Write into the staging: it already holds every byte of the subresource.
    WriteStaging,
    /// Read the subresource back from the GPU, then write into the staging.
    ///
    /// The GPU holds pixels the staging does not and the write leaves some of
    /// them untouched, so only the read back makes the staging whole.
    ReadBackFirst,
    /// Write into the staging with no read back: the write defines every byte.
    Overwrite,
}

/// A cube map's six faces. Every other texture kind occupies face 0 alone.
const FACE_COUNT: u32 = 6;

/// Which copy holds the pixels of each subresource of one texture.
///
/// One bit per level per face, set while the Metal texture holds pixels that
/// subresource's staging does not. A D3D9 mip chain tops out at 15 levels, so a
/// face's mask covers every level a texture can carry; the bound is a guard,
/// not a limit anything reaches.
#[derive(Debug, Default)]
pub struct LevelAuthorityMask {
    gpu: [u32; FACE_COUNT as usize],
}

impl LevelAuthorityMask {
    /// A texture whose staging defines every subresource.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            gpu: [0; FACE_COUNT as usize],
        }
    }

    /// Record that the GPU wrote `(face, level)` and its staging did not receive the write.
    pub const fn gpu_wrote(&mut self, face: u32, level: usize) {
        if face < FACE_COUNT && level < u32::BITS as usize {
            self.gpu[face as usize] |= 1u32 << level;
        }
    }

    /// Whether the Metal texture holds pixels `(face, level)`'s staging does not.
    #[must_use]
    pub const fn gpu_holds(&self, face: u32, level: usize) -> bool {
        face < FACE_COUNT
            && level < u32::BITS as usize
            && self.gpu[face as usize] & (1u32 << level) != 0
    }

    /// Plan a CPU write of `(face, level)` and record that its staging defines it after.
    ///
    /// `whole_level` says the write covers every byte of the subresource. The
    /// claim is released whichever branch the caller takes, so a read back that
    /// cannot run costs one diagnostic rather than one per write, and a
    /// subresource the caller overwrites costs no read back at all.
    pub const fn plan_write(&mut self, face: u32, level: usize, whole_level: bool) -> WritePlan {
        if !self.gpu_holds(face, level) {
            return WritePlan::WriteStaging;
        }
        self.gpu[face as usize] &= !(1u32 << level);
        if whole_level {
            WritePlan::Overwrite
        } else {
            WritePlan::ReadBackFirst
        }
    }
}

#[cfg(test)]
mod tests;
