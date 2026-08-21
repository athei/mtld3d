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

#[cfg(test)]
mod tests;
