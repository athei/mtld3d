//! Waiting for a PE-side thread to end without joining it.
//!
//! Under Wine the Win32 handle behind a `JoinHandle` held for a long session
//! can come back invalid, and `JoinHandle::join` panics on the failed wait,
//! which under `panic = "abort"` ends the game at device release.
//! `JoinHandle::is_finished` reads the count std keeps on the thread's result
//! and never waits on the OS handle, so a thread the PE side starts is waited
//! for by polling it ("The PE side waits for its threads, it never joins
//! them" in `docs/CONVENTIONS.md`).

use std::{
    sync::atomic::{Ordering, fence},
    thread::{self, JoinHandle},
    time::Duration,
};

/// How long [`wait_until_finished`] sleeps between two polls.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Wait until `handle`'s thread has finished, then let the handle go.
///
/// Polls `is_finished` with a millisecond's sleep between polls, which only
/// teardown pays. Every write the thread made before it ended is visible to
/// the caller once this returns. The thread's result is dropped with the
/// handle.
pub fn wait_until_finished<T>(handle: JoinHandle<T>) {
    while !handle.is_finished() {
        thread::sleep(POLL_INTERVAL);
    }
    // `is_finished` may read the count without ordering; the thread's last
    // writes happen before its release of that count, and this fence makes
    // them visible to what the caller reads next.
    fence(Ordering::Acquire);
    drop(handle);
}

#[cfg(test)]
mod tests;
