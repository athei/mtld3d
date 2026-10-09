//! The log line one `Reset` writes.
//!
//! A `Reset` is rare: a resolution change, a windowed/fullscreen switch, a
//! device recovered after a loss. Each call therefore logs one line, at info
//! when it succeeds and at warn when it does not, naming the back buffer it
//! leaves and everything the present parameters asked for, so a log at the
//! default level explains a mode change without `RUST_LOG=debug`. The device
//! decides the outcome; this module only renders it.

use core::fmt;

use mtld3d_types::{
    D3DPRESENT_INTERVAL_DEFAULT, D3DPRESENT_INTERVAL_FOUR, D3DPRESENT_INTERVAL_IMMEDIATE,
    D3DPRESENT_INTERVAL_ONE, D3DPRESENT_INTERVAL_THREE, D3DPRESENT_INTERVAL_TWO,
    D3DPRESENT_PARAMETERS, D3DSWAPEFFECT_COPY, D3DSWAPEFFECT_DISCARD, D3DSWAPEFFECT_FLIP,
};

use crate::format::format_name;

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
