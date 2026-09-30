//! The sections of the per-draw state snapshot.
//!
//! The device rebuilds a section of the snapshot only when a state write has
//! marked it dirty; each section here is one bit of that dirty mask.

/// One section of the per-draw snapshot, its discriminant the section's bit in the dirty mask.
///
/// The device's `SnapshotDirty` flags are built from these discriminants, so
/// the bit a draw rebuilds and the counter the perf summary names for it
/// have one home. Bit 2 is unassigned. Arrays indexed by section (the perf
/// summary's rebuild counters and section timers) are [`SnapshotSection::SLOTS`]
/// long, and the unassigned slot stays zero.
#[repr(u32)]
pub enum SnapshotSection {
    /// `RS`: the render-state snapshot.
    Rs = 0,
    /// `STAGES`: bound textures and per-stage sampler state.
    Stages = 1,
    /// `RT_DS`: depth and stencil presence on the bound target.
    RtDs = 3,
    /// `VDECL`: the vertex attribute layout.
    Vdecl = 4,
    /// `VARIANT`: the pipeline variant key.
    Variant = 5,
    /// `VS_SOURCE`: the FF VS key or the programmable VS source.
    VsSource = 6,
    /// `PS_SOURCE`: the FF PS key or the programmable PS source.
    PsSource = 7,
    /// `VS_CONST`: the FF VS constant sections.
    VsConst = 8,
    /// `PS_CONST`: the FF PS constants.
    PsConst = 9,
    /// `ALPHA_REF`: the alpha-reference bytes.
    AlphaRef = 10,
    /// `FOG_COLOR`: the fog-colour bytes.
    FogColor = 11,
    /// `BUMP_ENV`: the bump-environment matrix bytes.
    BumpEnv = 12,
    /// `VS_CONST_I`: the VS integer-constant file.
    VsConstI = 13,
    /// `VS_DRAW`: the per-draw `VsDraw` uniform.
    VsDraw = 14,
    /// `VS_CONST_B`: the VS boolean-constant bitmask.
    VsConstB = 15,
    /// `PS_CONST_I`: the PS integer-constant file.
    PsConstI = 16,
    /// `PS_CONST_B`: the PS boolean-constant bitmask.
    PsConstB = 17,
}

impl SnapshotSection {
    /// One past the highest section bit: the length of the per-section arrays.
    pub const SLOTS: usize = Self::PsConstB as usize + 1;

    /// This section's bit in the snapshot dirty mask.
    #[must_use]
    pub const fn bit(self) -> u32 {
        1 << self as u32
    }
}
