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

/// What to do about one external resize of a fullscreen device's window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalResizeAction {
    /// The window already covers the monitor; nothing to do.
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
        if incoming == monitor {
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
mod tests {
    use super::{ExternalResizeAction, ExternalResizeGuard, REASSERT_BUDGET};

    const MONITOR: (u32, u32) = (3456, 2234);

    #[test]
    fn a_covered_window_is_left_alone() {
        let mut g = ExternalResizeGuard::new();
        assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
    }

    #[test]
    fn a_shrunk_window_is_reasserted_once_per_size() {
        let mut g = ExternalResizeGuard::new();
        assert_eq!(
            g.decide((1288, 834), MONITOR),
            ExternalResizeAction::Reassert
        );
        // The same clamp coming back means the re-assert did not stick.
        assert_eq!(
            g.decide((1288, 834), MONITOR),
            ExternalResizeAction::Suppressed
        );
        // A different size is a new event and earns its own re-assert.
        assert_eq!(
            g.decide((1608, 934), MONITOR),
            ExternalResizeAction::Reassert
        );
    }

    #[test]
    fn covering_the_monitor_clears_the_guard_and_refills_the_budget() {
        let mut g = ExternalResizeGuard::new();
        assert_eq!(
            g.decide((1288, 834), MONITOR),
            ExternalResizeAction::Reassert
        );
        assert_eq!(g.decide(MONITOR, MONITOR), ExternalResizeAction::Covered);
        // The previously re-asserted size counts as a fresh event again.
        assert_eq!(
            g.decide((1288, 834), MONITOR),
            ExternalResizeAction::Reassert
        );
    }

    #[test]
    fn alternating_clamp_sizes_exhaust_the_budget() {
        let mut g = ExternalResizeGuard::new();
        for i in 0..REASSERT_BUDGET {
            let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
            assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
        }
        assert_eq!(
            g.decide((100, 100), MONITOR),
            ExternalResizeAction::Suppressed
        );
        assert_eq!(
            g.decide((300, 300), MONITOR),
            ExternalResizeAction::Suppressed
        );
    }

    #[test]
    fn reset_refills_the_budget() {
        let mut g = ExternalResizeGuard::new();
        for i in 0..REASSERT_BUDGET {
            let size = if i % 2 == 0 { (100, 100) } else { (200, 200) };
            assert_eq!(g.decide(size, MONITOR), ExternalResizeAction::Reassert);
        }
        g.reset();
        assert_eq!(
            g.decide((100, 100), MONITOR),
            ExternalResizeAction::Reassert
        );
    }
}
