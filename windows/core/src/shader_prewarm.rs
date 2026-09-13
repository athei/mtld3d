//! One-shot shader prewarm startup and cancellation.
//!
//! Only the worker owns the startup sender. A failed spawn therefore releases
//! the receiver, which disables persistence because no cache was validated.

use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::LOG_TARGET;

/// Lifetime handle for one device's prewarm worker.
///
/// Device release cancels and waits before encoder shutdown, so no prewarm
/// Metal call can race device cleanup.
pub struct PrewarmHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl PrewarmHandle {
    /// Start prewarm with the only sender for its completion channel.
    ///
    /// `Some` delivers a validated warm cache, including an empty cold start.
    /// `None` disables persistent writes. A rejected spawn drops the sender
    /// without running the body, so the receiver can recover without waiting.
    pub fn spawn<F, T>(run: F) -> (Self, Receiver<Option<T>>)
    where
        F: FnOnce(&AtomicBool) -> Option<T> + Send + 'static,
        T: Send + 'static,
    {
        Self::spawn_with(run, |work| {
            thread::Builder::new()
                .name("mtld3d-shader-prewarm".into())
                .spawn(work)
        })
    }

    /// Cancel prewarm and wait for all its work to finish. Idempotent.
    ///
    /// Work checks the stop flag between compiles. In-flight Metal calls and
    /// cache operations must finish before device cleanup proceeds.
    pub fn cancel_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            // Wine can invalidate the Win32 thread handle during long sessions.
            // JoinHandle::join would panic on WAIT_FAILED. is_finished reads
            // the std Packet's Arc count without waiting on that OS handle.
            while !join.is_finished() {
                thread::sleep(Duration::from_millis(1));
            }
            drop(join);
        }
    }

    fn spawn_with<F, T, S>(run: F, spawn: S) -> (Self, Receiver<Option<T>>)
    where
        F: FnOnce(&AtomicBool) -> Option<T> + Send + 'static,
        T: Send + 'static,
        S: FnOnce(Box<dyn FnOnce() + Send>) -> io::Result<JoinHandle<()>>,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop);
        let work = Box::new(move || {
            let result = run(&stop_for_thread);
            // Device release keeps the receiver alive until prewarm finishes.
            let _ = sender.send(result);
        });
        let join = match spawn(work) {
            Ok(join) => Some(join),
            Err(error) => {
                log::error!(
                    target: LOG_TARGET,
                    "shader_cache: failed to spawn prewarm thread, cache disabled: {error}"
                );
                None
            }
        };
        (Self { stop, join }, receiver)
    }
}

/// Wait for prewarm before accepting frames, disabling writes on disconnection.
///
/// A missing payload cannot authorize appends to an unvalidated cache file.
#[must_use]
pub fn receive<T>(receiver: &Receiver<Option<T>>) -> Option<T> {
    receiver.recv().unwrap_or_else(|_| {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "shader_cache: prewarm channel closed without payload, starting cold with cache disabled"
        );
        None
    })
}

#[cfg(test)]
mod tests;
