//! Pacing `Present` on the API thread under `present.renderAhead = 0`.
//!
//! With render-ahead off the unix side presents free-running, so the cadence a
//! vsync request asks for has to come from the API thread. The pacer keeps the
//! next deadline and the last pacing decision it reported; the caller does the
//! waiting and the logging, which keeps this host-testable arithmetic.

use std::time::{Duration, Instant};

/// The cadence state of one device's `Present`.
pub struct PresentPacer {
    next_deadline: Option<Instant>,
    /// Whether a pacing decision has been reported yet.
    reported: bool,
    /// The period last reported; `None` for free-running.
    reported_period: Option<Duration>,
}

impl Default for PresentPacer {
    fn default() -> Self {
        Self::new()
    }
}

impl PresentPacer {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_deadline: None,
            reported: false,
            reported_period: None,
        }
    }

    /// The instant this `Present` should return at, or `None` when the cadence restarts.
    ///
    /// A deadline still ahead of `now` is kept and the next one advances by
    /// `period`, so the cadence does not drift with the caller's jitter. A
    /// deadline already passed restarts the cadence from `now`: one long frame
    /// costs one long frame, never a burst that catches up.
    pub fn deadline(&mut self, now: Instant, period: Duration) -> Option<Instant> {
        match self.next_deadline {
            Some(deadline) if deadline > now => {
                self.next_deadline = Some(deadline + period);
                Some(deadline)
            }
            _ => {
                self.next_deadline = Some(now + period);
                None
            }
        }
    }

    /// Whether `period` differs from the decision last reported; the first call always does.
    pub fn report_changed(&mut self, period: Option<Duration>) -> bool {
        if self.reported && self.reported_period == period {
            return false;
        }
        self.reported = true;
        self.reported_period = period;
        true
    }
}

#[cfg(test)]
mod tests;
