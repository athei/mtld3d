//! Mapping between D3D9 `D3DPRESENT_INTERVAL_*` and `CAMetalLayer`'s `displaySyncEnabled`.
//!
//! That property is Apple's recommended vsync knob.

pub mod present_interval {
    pub const DEFAULT: u32 = 0x0000_0000;
    pub const ONE: u32 = 0x0000_0001;
    pub const TWO: u32 = 0x0000_0002;
    pub const THREE: u32 = 0x0000_0004;
    pub const FOUR: u32 = 0x0000_0008;
    pub const IMMEDIATE: u32 = 0x8000_0000;
}

/// Result of mapping a `D3DPRESENT_INTERVAL_*` to `displaySyncEnabled`.
///
/// `Fallthrough` carries the same boolean as a supported choice but
/// signals the caller to fire a `log_once_warn_by!` keyed on the raw
/// input — non-1:1 ratios (TWO/THREE/FOUR) and unknown bit patterns
/// take this path. Display-rate is the only ratio honoured directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplaySync {
    On,
    Off,
    Fallthrough,
}

impl DisplaySync {
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::On | Self::Fallthrough)
    }
}

#[must_use]
pub const fn display_sync_for(interval: u32) -> DisplaySync {
    match interval {
        present_interval::DEFAULT | present_interval::ONE => DisplaySync::On,
        present_interval::IMMEDIATE => DisplaySync::Off,
        _ => DisplaySync::Fallthrough,
    }
}

/// Which capture marks frame `index` of a `total`-frame diagnostic run carries.
///
/// Returns `(start, stop)`: the first frame of the run starts the GPU
/// capture, the last one stops it, and a one-frame run does both. Frames
/// are numbered from 1; an index outside the run carries nothing.
#[must_use]
pub const fn capture_marks(index: u32, total: u32) -> (bool, bool) {
    if index == 0 || index > total {
        return (false, false);
    }
    (index == 1, index == total)
}

/// The capture marks a swapped-out frame hands to the frame replacing it.
///
/// A run ends with the frame the closing `Present` submits, so a swap that
/// does not present passes the run's stop to the continuation; left behind,
/// the encoder never sees it and the capture ends only with the process.
/// `submitted` says whether the outgoing frame still reaches the encoder,
/// which is what decides the start: a mid-frame flush keeps it with the piece
/// it sends, while a swap that drops its frame hands it on too, or the
/// capture never opens. Marks are `(start, stop)` as `capture_marks` returns
/// them, and the outgoing frame keeps whatever is not handed on.
#[must_use]
pub const fn carried_capture_marks(marks: (bool, bool), submitted: bool) -> (bool, bool) {
    let (start, stop) = marks;
    (start && !submitted, stop)
}

#[cfg(test)]
mod tests;
