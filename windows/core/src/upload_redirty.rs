//! Uploads the encoder never emitted, on their way back into the dirty state.
//!
//! A texture upload is scheduled on the API thread: the flush takes the
//! level's dirty bit and its pending rect, builds a job and hands it to the
//! encoder thread. Every step past that hand-off can still decline to emit
//! anything: the destination texture may fail to create, the staging wrapper
//! may fail, a padded repack may fail, and a texel-widening expansion has no
//! blit that could stand in for the pass it needs. The dirty state is already
//! gone by then, and `UnlockRect` publishes only the rectangle the game
//! locked, so nothing re-announces the region: the mip keeps whatever it held
//! until the game happens to write those texels again.
//!
//! This queue closes that gap. The encoder records the subresource and the
//! rectangle of every upload it did not emit; the API thread drains the queue
//! once a frame and marks each one dirty again, so the next bind retries.
//! Retries are bounded per subresource, because a decline with a permanent
//! cause repeats on every attempt and an unbounded retry would schedule a
//! failing upload on every draw that binds the texture.
//!
//! Pure bookkeeping: no Metal handles, no D3D9 objects, so the whole contract
//! is host-testable.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use rustc_hash::FxHashMap;

use crate::{dirty_rect::DirtyRect, ids::TextureId};

/// Times one subresource is re-marked dirty before the layer stops retrying.
///
/// A transient decline (an allocation that failed under memory pressure)
/// clears well inside this; a permanent one (a pipeline the device will
/// never compile) does not, and retrying it forever would cost a scheduled
/// upload on every draw that binds the texture for the rest of the run.
pub const MAX_REDIRTY_ATTEMPTS: u32 = 4;

/// The subresource an upload writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RedirtySubresource {
    /// The texture the upload belongs to.
    pub texture_id: TextureId,
    /// Index into the texture's per-subresource storage.
    ///
    /// The mip level for a 2D or volume texture, the cube subresource index
    /// for a cube face level. It is the index the scheduler already carries
    /// on the job, so no side table has to agree with it.
    pub index: u32,
}

/// One upload that reached no command buffer.
#[derive(Debug)]
pub struct RedirtyEntry {
    /// What the upload was writing.
    pub subresource: RedirtySubresource,
    /// Cube face the upload targeted; zero for every other texture kind.
    pub face: u32,
    /// Mip level the upload targeted.
    pub level: u32,
    /// Region of the level the upload was carrying.
    pub rect: DirtyRect,
}

/// Declined uploads, filled on the encoder thread and drained on the API thread.
///
/// Both counters exist so the two hot callers stay off the lock. The API
/// thread's drain reads `pending` once a frame and returns on a clear flag;
/// the encoder's acknowledgement of an emitted upload reads `tracked` and
/// returns while no subresource carries a decline record, which is the whole
/// of a run that never declines an upload.
pub struct RedirtyQueue {
    /// Set while `pending` holds an entry.
    pending: AtomicBool,
    /// Subresources currently carrying an attempt count.
    tracked: AtomicUsize,
    inner: Mutex<RedirtyInner>,
}

struct RedirtyInner {
    pending: Vec<RedirtyEntry>,
    attempts: FxHashMap<RedirtySubresource, u32>,
}

impl RedirtyQueue {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: AtomicBool::new(false),
            tracked: AtomicUsize::new(0),
            inner: Mutex::new(RedirtyInner {
                pending: Vec::new(),
                attempts: FxHashMap::default(),
            }),
        }
    }

    /// Record an upload the encoder did not emit; report whether it will be retried.
    ///
    /// `false` means the subresource has spent its budget: the caller warns
    /// and the region stays as the texture holds it. The entry is not queued
    /// in that case, so a spent subresource costs nothing per attempt beyond
    /// the lookup.
    ///
    /// # Panics
    ///
    /// If a previous caller panicked while holding the queue's lock.
    pub fn decline(&self, entry: RedirtyEntry) -> bool {
        let (retry, tracked) = {
            let mut inner = self.inner.lock().expect("redirty queue mutex poisoned");
            let attempts = inner.attempts.entry(entry.subresource).or_insert(0);
            *attempts += 1;
            let retry = *attempts <= MAX_REDIRTY_ATTEMPTS;
            if retry {
                inner.pending.push(entry);
            }
            (retry, inner.attempts.len())
        };
        self.tracked.store(tracked, Ordering::Relaxed);
        if retry {
            self.pending.store(true, Ordering::Release);
        }
        retry
    }

    /// Forget a subresource's attempt count after an upload of it was emitted.
    ///
    /// The budget counts consecutive declines, so an upload that reached the
    /// command stream gives the subresource its full budget back: a decline
    /// under transient memory pressure must not spend a texture's retries for
    /// the rest of the run.
    ///
    /// # Panics
    ///
    /// If a previous caller panicked while holding the queue's lock.
    pub fn note_emitted(&self, subresource: RedirtySubresource) {
        if self.tracked.load(Ordering::Relaxed) == 0 {
            return;
        }
        let tracked = {
            let mut inner = self.inner.lock().expect("redirty queue mutex poisoned");
            if inner.attempts.remove(&subresource).is_none() {
                return;
            }
            inner.attempts.len()
        };
        self.tracked.store(tracked, Ordering::Relaxed);
    }

    /// Take every upload waiting to be marked dirty again.
    ///
    /// # Panics
    ///
    /// If a previous caller panicked while holding the queue's lock.
    #[must_use]
    pub fn take_pending(&self) -> Vec<RedirtyEntry> {
        if !self.pending.swap(false, Ordering::Acquire) {
            return Vec::new();
        }
        let mut inner = self.inner.lock().expect("redirty queue mutex poisoned");
        core::mem::take(&mut inner.pending)
    }

    /// Whether a drain would find anything.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }
}

impl Default for RedirtyQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
