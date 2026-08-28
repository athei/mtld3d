//! Policy for external resizes of a fullscreen device's window.
//!
//! A fullscreen device's window is supposed to cover the monitor, but the
//! game may resize it after a Reset: engines that manage their own window
//! compute an `AdjustWindowRect` outer size for the new mode and apply it,
//! which under a real mode-set would still cover the (now smaller) screen.
//! We never set a mode, so following that resize leaves a small borderless
//! window. The answer is to re-assert the monitor rect — but never in an
//! unbounded fight with a window manager that keeps clamping the window
//! back, so the decision lives here as pure, host-testable state.
//!
//! A rect the window manager answers with a pixel larger than the monitor is
//! not one of those resizes. The window still covers the display, and a
//! re-assert only repeats the round trip that produced the extra pixel, so a
//! size that covers the monitor within that rounding counts as covered.

/// What to do about one external resize of a fullscreen device's window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalResizeAction {
    /// The window covers the monitor already; nothing to do.
    ///
    /// Covering allows for the window manager's own rounding, so a rect it
    /// answers a pixel larger than the monitor lands here rather than in a
    /// re-assert that cannot win.
    Covered,
    /// Re-apply the monitor rect.
    Reassert,
    /// Stop fighting the window manager.
    ///
    /// This size was already re-asserted against, or the per-session budget
    /// ran out. Caller logs once and leaves the window as it is.
    Suppressed,
}

/// How many re-asserts one fullscreen session may spend before giving up.
///
/// A game that reacts to a mode change with a single `SetWindowPos` spends
/// one. Alternating clamp sizes from a window manager would spend them all;
/// the budget turns that pathology into a one-shot warning instead of a
/// re-assert per frame. Refilled whenever the window is seen covered and on
/// every fullscreen Reset.
const REASSERT_BUDGET: u8 = 8;

/// Pixels a window may exceed the monitor by and still count as covering it.
///
/// A window rect makes a round trip through the window manager's own
/// coordinate space, and a monitor dimension that does not land on that
/// space's grid can come back a pixel or two larger than the rect that was
/// asked for. Such a window covers the display, which is what the fullscreen
/// rect is for, and re-asserting the monitor rect only repeats the round
/// trip: without the slack the guard spends its budget on a window that was
/// never wrong and ends the session with the give-up warning.
const COVER_SLACK: u32 = 2;

/// `true` when `size` covers `monitor`, the window manager's rounding allowed for.
///
/// Covering is not equality: a size at least as large as the monitor on both
/// axes, by no more than the window manager's rounding, is what a fullscreen
/// window has to hold. Anything smaller on either axis leaves the desktop
/// showing, and anything larger than that is a size the game chose.
#[must_use]
pub const fn covers_monitor(size: (u32, u32), monitor: (u32, u32)) -> bool {
    covers_axis(size.0, monitor.0) && covers_axis(size.1, monitor.1)
}

/// One axis of [`covers_monitor`].
const fn covers_axis(size: u32, monitor: u32) -> bool {
    size >= monitor && size - monitor <= COVER_SLACK
}

/// Ping-pong guard for [`ExternalResizeAction`] decisions.
///
/// One per fullscreen session, carried beside the saved window state.
pub struct ExternalResizeGuard {
    /// The last client size a re-assert was issued against.
    ///
    /// The same size arriving again means our `SetWindowPos` did not stick
    /// (the window manager clamped it back); re-asserting again would loop.
    last_reasserted: Option<(u32, u32)>,
    budget: u8,
}

impl Default for ExternalResizeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalResizeGuard {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_reasserted: None,
            budget: REASSERT_BUDGET,
        }
    }

    /// Refill the budget for a fresh game-driven fullscreen transition.
    pub const fn reset(&mut self) {
        self.last_reasserted = None;
        self.budget = REASSERT_BUDGET;
    }

    /// Decide what to do about one external resize.
    ///
    /// `incoming` is the window's new client size, `monitor` the monitor
    /// rect it is supposed to cover, both in pixels.
    pub fn decide(&mut self, incoming: (u32, u32), monitor: (u32, u32)) -> ExternalResizeAction {
        if covers_monitor(incoming, monitor) {
            self.reset();
            return ExternalResizeAction::Covered;
        }
        if self.last_reasserted == Some(incoming) || self.budget == 0 {
            return ExternalResizeAction::Suppressed;
        }
        self.last_reasserted = Some(incoming);
        self.budget -= 1;
        ExternalResizeAction::Reassert
    }
}

#[cfg(test)]
mod tests;
