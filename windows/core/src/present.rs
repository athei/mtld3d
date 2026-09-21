//! Mapping from a D3D9 `D3DPRESENT_INTERVAL_*` to the pacing a device asks of its layer.
//!
//! The layer paces presents by a minimum duration alone. An interval
//! contributes the vsync request, and a divided interval a frame-rate ceiling
//! on top of it; the user's `present.maxFps` rides the same ceiling.

use mtld3d_types::{
    D3DPRESENT_INTERVAL_DEFAULT, D3DPRESENT_INTERVAL_FOUR, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPRESENT_INTERVAL_ONE, D3DPRESENT_INTERVAL_THREE, D3DPRESENT_INTERVAL_TWO,
};

/// Result of mapping a `D3DPRESENT_INTERVAL_*` to the vsync request.
///
/// `Fallthrough` carries the same boolean as a supported choice but
/// signals the caller to fire a `log_once_warn_by!` keyed on the raw
/// input: a bit pattern that names no interval takes this path and runs at
/// display rate.
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

/// The pacing a device hands its layer: the vsync request and the frame-rate ceiling.
///
/// `max_fps` is in Hz and `0` means no ceiling. It is the effective ceiling
/// ([`effective_max_fps`]), so a divided interval and `present.maxFps` are
/// already folded into it and the unix side sees one number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPacing {
    pub display_sync: bool,
    pub max_fps: u32,
}

#[must_use]
pub const fn display_sync_for(interval: u32) -> DisplaySync {
    match interval {
        D3DPRESENT_INTERVAL_DEFAULT
        | D3DPRESENT_INTERVAL_ONE
        | D3DPRESENT_INTERVAL_TWO
        | D3DPRESENT_INTERVAL_THREE
        | D3DPRESENT_INTERVAL_FOUR => DisplaySync::On,
        D3DPRESENT_INTERVAL_IMMEDIATE => DisplaySync::Off,
        _ => DisplaySync::Fallthrough,
    }
}

/// How many refresh periods one present of `interval` spans.
///
/// `TWO`, `THREE` and `FOUR` answer 2, 3 and 4. Every other value answers 1:
/// display rate for `DEFAULT` and `ONE`, and nothing to divide for
/// `IMMEDIATE` or a pattern that names no interval.
#[must_use]
pub const fn interval_divisor(interval: u32) -> u32 {
    match interval {
        D3DPRESENT_INTERVAL_TWO => 2,
        D3DPRESENT_INTERVAL_THREE => 3,
        D3DPRESENT_INTERVAL_FOUR => 4,
        _ => 1,
    }
}

/// The frame-rate ceiling in Hz for `interval` on a `refresh_hz` mode, `0` for none.
///
/// A divided interval is a ceiling of `refresh_hz / N`, and the result is the
/// lower of that and `configured` (`present.maxFps`), where `0` on either
/// side means that side sets no ceiling. The quotient rounds up: the ceiling
/// becomes a minimum present duration, and one a little shorter than `N`
/// refresh periods still lands on the `N`th, while one a little longer would
/// slip to the period after it. An undivided interval, or a mode whose rate
/// is unknown (`refresh_hz == 0`), leaves `configured` as it is.
#[must_use]
pub const fn effective_max_fps(interval: u32, refresh_hz: u32, configured: u32) -> u32 {
    let divisor = interval_divisor(interval);
    if divisor == 1 || refresh_hz == 0 {
        return configured;
    }
    let divided = refresh_hz.div_ceil(divisor);
    if configured == 0 || divided < configured {
        divided
    } else {
        configured
    }
}

/// The pacing `interval` asks for on a `refresh_hz` mode under a `configured` ceiling.
#[must_use]
pub const fn layer_pacing_for(interval: u32, refresh_hz: u32, configured: u32) -> LayerPacing {
    LayerPacing {
        display_sync: display_sync_for(interval).enabled(),
        max_fps: effective_max_fps(interval, refresh_hz, configured),
    }
}

/// The pacing change a `Reset` leaves queued, given the pacing the layer holds.
///
/// `held` is the value the layer was last handed and `want` the value the
/// `Reset` resolved to. The answer is the whole queue rather than a delta:
/// `None` when the two agree, because the layer already paces the way the
/// guest asked for, so nothing has to go out and anything still queued is
/// dropped rather than written back as the value the layer never left.
#[must_use]
pub const fn queued_pacing(held: LayerPacing, want: LayerPacing) -> Option<LayerPacing> {
    if want.display_sync == held.display_sync && want.max_fps == held.max_fps {
        None
    } else {
        Some(want)
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
