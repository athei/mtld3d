//! Reentrant per-device lock behind `D3DCREATE_MULTITHREADED`.
//!
//! An application that creates its device with the flag may call the device
//! and every object it created from any thread. Each COM entry point then
//! holds this lock for its duration: one owner thread at a time, re-entered
//! freely by that thread (`Reset` applies state through the setters an
//! application calls, a child `Release` reaches the device's own release),
//! released when the outermost entry returns. The threads the device owns
//! (encoder, submit, prewarm, log) never take it: none of them calls back
//! into the device, so a thread holding the lock while it waits on them
//! cannot form a cycle.
//!
//! Pure std. The `Mutex` guards the owner and depth for the few instructions
//! of a state update, never across an entry point's body, so no panic can
//! poison it while it matters; waiters block on a `Condvar` and are woken by
//! the release that empties the lock.

use std::{
    sync::{Condvar, Mutex, PoisonError},
    thread::ThreadId,
};

/// Reentrant lock: an owner thread plus a depth.
pub struct ApiLock {
    holder: Mutex<Holder>,
    released: Condvar,
}

/// Who holds the lock and how many times they entered it.
struct Holder {
    owner: Option<ThreadId>,
    depth: u32,
}

impl ApiLock {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            holder: Mutex::new(Holder {
                owner: None,
                depth: 0,
            }),
            released: Condvar::new(),
        }
    }

    /// Take the lock for the calling thread, or deepen it if the thread holds it already.
    ///
    /// Blocks while another thread holds it. The guard releases one level on
    /// drop; a thread that entered three times deep releases on its third.
    ///
    /// # Safety
    /// `self` must outlive the returned guard: the guard releases through a
    /// raw pointer when dropped.
    pub unsafe fn enter(&self) -> ApiGuard {
        let me = std::thread::current().id();
        let mut holder = self.holder.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            match holder.owner {
                Some(owner) if owner == me => {
                    holder.depth += 1;
                    break;
                }
                None => {
                    holder.owner = Some(me);
                    holder.depth = 1;
                    break;
                }
                Some(_) => {
                    holder = self
                        .released
                        .wait(holder)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        drop(holder);
        ApiGuard { lock: self }
    }

    /// Release one level; at depth zero the lock empties and one waiter wakes.
    fn leave(&self) {
        let mut holder = self.holder.lock().unwrap_or_else(PoisonError::into_inner);
        holder.depth -= 1;
        if holder.depth == 0 {
            holder.owner = None;
            drop(holder);
            self.released.notify_one();
        }
    }
}

impl Default for ApiLock {
    fn default() -> Self {
        Self::new()
    }
}

/// Releases one level of its lock on drop; `NOOP` for a device without the flag.
///
/// Holds a raw pointer rather than a reference so a thunk can go on to take
/// exclusive borrows of the objects the lock protects.
#[must_use = "bind to `_api`; `let _ = ...` releases the lock at once"]
pub struct ApiGuard {
    lock: *const ApiLock,
}

impl ApiGuard {
    /// The guard of a device that has no lock: dropping it does nothing.
    pub const NOOP: Self = Self {
        lock: core::ptr::null(),
    };
}

impl Drop for ApiGuard {
    fn drop(&mut self) {
        if self.lock.is_null() {
            return;
        }
        // SAFETY: `enter` requires its lock to outlive the guard, so a non-null
        // pointer names a lock that is still live.
        unsafe { (*self.lock).leave() };
    }
}

#[cfg(test)]
mod tests;
