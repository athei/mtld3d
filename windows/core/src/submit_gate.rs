//! Gate the API thread on the submission of its own frame.
//!
//! With `present.renderAhead = 0` the API thread waits at `Present` until the
//! submit thread has issued the frame's command buffers, drawable wait
//! included. Nothing is then queued behind the display when the game asks for
//! a synchronous read-back, so the read-back waits for pending GPU work rather
//! than for drawables. Host-testable: a `Mutex<u64>` and a `Condvar`, no Metal.

use std::{
    sync::{Condvar, Mutex, PoisonError},
    time::{Duration, Instant},
};

/// The highest submit sequence number whose `SubmitFrame` thunk has returned.
pub struct SubmitGate {
    submitted: Mutex<u64>,
    changed: Condvar,
}

impl Default for SubmitGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SubmitGate {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            submitted: Mutex::new(0),
            changed: Condvar::new(),
        }
    }

    /// Record that `seq` has been submitted; a lower or repeated value changes nothing.
    pub fn publish(&self, seq: u64) {
        let mut submitted = self
            .submitted
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if seq <= *submitted {
            return;
        }
        *submitted = seq;
        drop(submitted);
        self.changed.notify_all();
    }

    /// The highest sequence published so far; zero before the first submit.
    #[must_use]
    pub fn submitted(&self) -> u64 {
        *self
            .submitted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Block until `seq` or a later sequence is published; `false` if `timeout` passes first.
    #[must_use]
    pub fn wait_for(&self, seq: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut submitted = self
            .submitted
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while *submitted < seq {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, _) = self
                .changed
                .wait_timeout(submitted, deadline - now)
                .unwrap_or_else(PoisonError::into_inner);
            submitted = guard;
        }
        drop(submitted);
        true
    }
}

#[cfg(test)]
mod tests;
