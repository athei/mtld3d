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
    D3DSWAPEFFECT_FLIPEX, D3DSWAPEFFECT_OVERLAY,
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
    /// The method called, `Reset` unless [`ResetSummary::via`] names `ResetEx`.
    method: ResetMethod,
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
            method: ResetMethod::Reset,
            old_size,
            params,
            outcome,
        }
    }

    /// The same line for a call through `method`.
    #[must_use]
    pub const fn via(mut self, method: ResetMethod) -> Self {
        self.method = method;
        self
    }
}

impl<O: fmt::Display> fmt::Display for ResetSummary<'_, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pp = self.params;
        write!(
            f,
            "{}::{} {}x{} -> {}x{} {}, {} back buffer{}",
            self.interface,
            self.method,
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
            D3DSWAPEFFECT_OVERLAY => f.write_str("OVERLAY")?,
            D3DSWAPEFFECT_FLIPEX => f.write_str("FLIPEX")?,
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

/// The method a `Reset` line names.
#[derive(Clone, Copy)]
pub enum ResetMethod {
    /// `IDirect3DDevice9::Reset`, the base slot, on either kind of device.
    Reset,
    /// `IDirect3DDevice9Ex::ResetEx`, which also takes the fullscreen display mode.
    ResetEx,
}

impl fmt::Display for ResetMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Reset => "Reset",
            Self::ResetEx => "ResetEx",
        })
    }
}

bitflags::bitflags! {
    /// What a successful `Reset` did, as the outcome its line ends with.
    ///
    /// A plain device's `Reset` returns every state to its default, so its
    /// outcome names only the back buffer and a recovery. An extended
    /// device's keeps its state and default-pool resources, detaches a back
    /// buffer or depth surface the application holds, and rebinds its targets
    /// to the swap chain, and the outcome says so.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ResetDone: u8 {
        /// The back buffer was recreated, at a new size or configuration.
        const RESIZED = 1 << 0;
        /// The device was waiting for a successful `Reset` and no longer is.
        const RECOVERED = 1 << 1;
        /// An extended device: state kept and targets rebound.
        const EXTENDED = 1 << 2;
        /// The application's back buffer kept the old surface, detached from the swap chain.
        const BACK_BUFFER_DETACHED = 1 << 3;
        /// The application's auto depth-stencil kept the old surface, detached from the device.
        const DEPTH_DETACHED = 1 << 4;
        /// An extended device whose earlier `Reset` failed, logged once, and which now succeeded.
        const FAILURES_ENDED = 1 << 5;
    }
}

impl fmt::Display for ResetDone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ok, ")?;
        if self.contains(Self::EXTENDED) {
            f.write_str("state and default-pool resources kept, ")?;
        }
        f.write_str(if self.contains(Self::RESIZED) {
            "back buffer recreated"
        } else {
            "same size, back buffer kept"
        })?;
        let detached = *self & (Self::BACK_BUFFER_DETACHED | Self::DEPTH_DETACHED);
        if detached == Self::BACK_BUFFER_DETACHED | Self::DEPTH_DETACHED {
            f.write_str(", the held back buffer and depth surface detached")?;
        } else if detached == Self::BACK_BUFFER_DETACHED {
            f.write_str(", the held back buffer detached")?;
        } else if detached == Self::DEPTH_DETACHED {
            f.write_str(", the held depth surface detached")?;
        }
        if self.contains(Self::EXTENDED) {
            f.write_str(", targets rebound to the swap chain")?;
        }
        if self.contains(Self::RECOVERED) {
            f.write_str(", device recovered")?;
        }
        if self.contains(Self::FAILURES_ENDED) {
            f.write_str(", ending a run of failed Resets")?;
        }
        Ok(())
    }
}

/// What a `Reset` that did not complete left behind, as its line says after the reason.
pub enum ResetFailureEffect {
    /// Nothing is said: a plain device's line, as it always was.
    Unstated,
    /// An extended device refused the request and changed nothing.
    Unchanged,
    /// An extended device kept its state but could not rebuild the swap chain.
    StateKeptResetOwed,
}

/// The outcome of a `Reset` that did not complete: why, and what it left behind.
pub struct ResetFailed<'a> {
    reason: fmt::Arguments<'a>,
    effect: ResetFailureEffect,
}

impl<'a> ResetFailed<'a> {
    /// A `Reset` that stopped for `reason`, leaving what `effect` says.
    #[must_use]
    pub const fn new(reason: fmt::Arguments<'a>, effect: ResetFailureEffect) -> Self {
        Self { reason, effect }
    }
}

impl fmt::Display for ResetFailed<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.reason.fmt(f)?;
        f.write_str(match self.effect {
            ResetFailureEffect::Unstated => "",
            ResetFailureEffect::Unchanged => "; the extended device is unchanged",
            ResetFailureEffect::StateKeptResetOwed => {
                "; the extended device keeps its state and needs another Reset"
            }
        })
    }
}

/// `true` when a successful `Reset` logs at info, `false` for debug.
///
/// `changed` says the `Reset` changed what the device presents with: a new
/// size, multisample configuration or any present parameter
/// [`changes_presentation`] counts, or a back buffer rebuilt because there was
/// none. A recreate an extended device makes only because the application
/// holds its back buffer or depth surface is no such change, so
/// [`ResetDone::RESIZED`] alone stays at debug: a game that holds a surface
/// and calls `Reset` on every step of a window drag would otherwise log each
/// step. A recovery and the end of a run of failures log at info.
#[must_use]
pub const fn success_logs_at_info(changed: bool, done: ResetDone) -> bool {
    changed || done.intersects(ResetDone::RECOVERED.union(ResetDone::FAILURES_ENDED))
}

/// `true` when a `Reset` that did not complete warns, `false` for debug.
///
/// A game retries a failing `Reset` every frame, so only the first failure of
/// a run warns. `recovering` is a device already waiting for a successful
/// `Reset`, whose first failure warned. `failure_logged` is an extended device
/// whose earlier rejection warned and changed nothing; its later rejections
/// stay at debug, but a failure that leaves it needing a `Reset` warns, since
/// an extended device can run on without one and nothing else would say so.
#[must_use]
pub const fn failure_warns(
    recovering: bool,
    failure_logged: bool,
    effect: &ResetFailureEffect,
) -> bool {
    !recovering && (matches!(effect, ResetFailureEffect::StateKeptResetOwed) || !failure_logged)
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
