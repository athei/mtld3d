//! The log line one `Reset` writes.
//!
//! Each `Reset` that reaches validation logs one line naming the back buffer
//! it leaves and everything the present parameters asked for, so a log at the
//! default level explains a resolution change, a windowed/fullscreen switch
//! or a vsync toggle without `RUST_LOG=debug`. Not every `Reset` is such a
//! change: some games call it on every step of a window drag, after the
//! device already followed the client area, and others retry a failing one
//! every frame. A success therefore logs at info only when
//! [`changes_presentation`] says it changed something, or when it recovers
//! the device, and the device picks the level of a failure. The device
//! decides the outcome; this module renders it and makes that comparison.

use core::fmt;

use mtld3d_types::{
    D3DPRESENT_INTERVAL_DEFAULT, D3DPRESENT_INTERVAL_FOUR, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPRESENT_INTERVAL_ONE, D3DPRESENT_INTERVAL_THREE, D3DPRESENT_INTERVAL_TWO,
    D3DPRESENT_PARAMETERS, D3DSWAPEFFECT_COPY, D3DSWAPEFFECT_DISCARD, D3DSWAPEFFECT_FLIP,
};

use crate::format::format_name;

/// `true` when a `Reset` adopting `next` changes what `previous` presented with.
///
/// `previous` is what the last `Reset` or `CreateDevice` stored, kept current
/// by the device's resize on `WM_SIZE`; `next` is what this `Reset` stores,
/// resolved the same way. A windowed back-buffer size is left out: the
/// device follows the client area on its own and stores the size it took,
/// so a game that calls `Reset` on every step of a window drag repeats the
/// size the device holds already, and the device reports a real resize
/// itself. Every other field counts, a presentation interval alone included,
/// since a vsync toggle reaches the device only through a `Reset`.
#[must_use]
pub const fn changes_presentation(
    previous: &D3DPRESENT_PARAMETERS,
    next: &D3DPRESENT_PARAMETERS,
) -> bool {
    let fullscreen_mode_changed = next.windowed == 0
        && (previous.back_buffer_width != next.back_buffer_width
            || previous.back_buffer_height != next.back_buffer_height
            || previous.full_screen_refresh_rate_in_hz != next.full_screen_refresh_rate_in_hz);
    fullscreen_mode_changed
        || previous.windowed != next.windowed
        || previous.back_buffer_format != next.back_buffer_format
        || previous.back_buffer_count != next.back_buffer_count
        || previous.multi_sample_type != next.multi_sample_type
        || previous.multi_sample_quality != next.multi_sample_quality
        || previous.swap_effect != next.swap_effect
        || previous.device_window != next.device_window
        || previous.enable_auto_depth_stencil != next.enable_auto_depth_stencil
        || previous.auto_depth_stencil_format != next.auto_depth_stencil_format
        || previous.flags != next.flags
        || previous.presentation_interval != next.presentation_interval
}

/// One `Reset` call, rendered as its log line.
///
/// `params` are the present parameters as far as the device resolved them
/// when the outcome was known: a windowed request's zero size and
/// `D3DFMT_UNKNOWN` read back as the client size and the display format
/// once the call got that far.
pub struct ResetSummary<'a, O> {
    /// The interface the call came through, `IDirect3DDevice9` or `IDirect3DDevice9Ex`.
    interface: &'static str,
    /// The back-buffer size before the call.
    old_size: (u32, u32),
    params: &'a D3DPRESENT_PARAMETERS,
    outcome: O,
}

impl<'a, O: fmt::Display> ResetSummary<'a, O> {
    /// Describe one `Reset` through `interface` that ended in `outcome`.
    pub const fn new(
        interface: &'static str,
        old_size: (u32, u32),
        params: &'a D3DPRESENT_PARAMETERS,
        outcome: O,
    ) -> Self {
        Self {
            interface,
            old_size,
            params,
            outcome,
        }
    }
}

impl<O: fmt::Display> fmt::Display for ResetSummary<'_, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pp = self.params;
        write!(
            f,
            "{}::Reset {}x{} -> {}x{} {}, {} back buffer{}",
            self.interface,
            self.old_size.0,
            self.old_size.1,
            pp.back_buffer_width,
            pp.back_buffer_height,
            FormatLabel(pp.back_buffer_format),
            pp.back_buffer_count,
            if pp.back_buffer_count == 1 { "" } else { "s" },
        )?;
        if pp.multi_sample_type != 0 {
            write!(
                f,
                ", multisample type {} quality {}",
                pp.multi_sample_type, pp.multi_sample_quality
            )?;
        }
        if pp.windowed != 0 {
            f.write_str(", windowed")?;
        } else if pp.full_screen_refresh_rate_in_hz == 0 {
            f.write_str(", fullscreen at the default refresh rate")?;
        } else {
            write!(
                f,
                ", fullscreen at {} Hz",
                pp.full_screen_refresh_rate_in_hz
            )?;
        }
        f.write_str(", swap effect ")?;
        match pp.swap_effect {
            D3DSWAPEFFECT_DISCARD => f.write_str("DISCARD")?,
            D3DSWAPEFFECT_FLIP => f.write_str("FLIP")?,
            D3DSWAPEFFECT_COPY => f.write_str("COPY")?,
            other => write!(f, "{other}")?,
        }
        f.write_str(", interval ")?;
        match pp.presentation_interval {
            D3DPRESENT_INTERVAL_DEFAULT => f.write_str("DEFAULT")?,
            D3DPRESENT_INTERVAL_ONE => f.write_str("ONE")?,
            D3DPRESENT_INTERVAL_TWO => f.write_str("TWO")?,
            D3DPRESENT_INTERVAL_THREE => f.write_str("THREE")?,
            D3DPRESENT_INTERVAL_FOUR => f.write_str("FOUR")?,
            D3DPRESENT_INTERVAL_IMMEDIATE => f.write_str("IMMEDIATE")?,
            other => write!(f, "{other:#x}")?,
        }
        if pp.enable_auto_depth_stencil == 0 {
            f.write_str(", no auto depth-stencil")?;
        } else {
            write!(
                f,
                ", auto depth-stencil {}",
                FormatLabel(pp.auto_depth_stencil_format)
            )?;
        }
        write!(f, ": {}", self.outcome)
    }
}

/// A `D3DFMT_*` code as the line names it: its name, or the raw code when it has none.
struct FormatLabel(u32);

impl fmt::Display for FormatLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `D3DFMT_UNKNOWN` is zero; it survives only on a rejection that
        // came before the windowed default was resolved.
        if self.0 == 0 {
            return f.write_str("UNKNOWN");
        }
        match format_name(self.0) {
            "D3DFMT_unknown" => write!(f, "format {}", self.0),
            name => f.write_str(name),
        }
    }
}

#[cfg(test)]
mod tests;
