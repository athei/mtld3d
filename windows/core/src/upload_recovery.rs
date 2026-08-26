//! Seq-gated replay of uploads whose command buffer the GPU aborted.
//!
//! Every upload (a `Staged` VB/IB dirty-range copy, a texture mip blit) is
//! encoded into a command buffer, and its source memory is freed once that
//! buffer retires. Retirement is not success: a command buffer the driver
//! killed reaches `MTLCommandBufferStatus::Error`, discards every encode it
//! carried, and still runs its completion handler. Freeing on retirement
//! alone therefore loses the upload for the rest of the run, because the
//! layer never re-announces a region on its own and a title that fills
//! static geometry once at load time never announces it again.
//!
//! This queue separates the two facts. An entry is released only when the
//! GPU both retired its seq (`coherent_seq`) and did not fail at or after
//! it (`failed_seq`); otherwise the caller re-issues the copy and hands the
//! entry back through [`UploadRecoveryQueue::requeue`] under the new
//! frame's seq. Pure bookkeeping: no Metal handles, no blit shapes, so the
//! whole ordering contract is host-testable.

use std::collections::VecDeque;

use rustc_hash::FxHashSet;

/// Aborted submits one upload is replayed across before it is abandoned.
///
/// Each replay keeps the upload's source memory alive for another frame,
/// which counts against `memory.vbibRetentionCapMB`; retrying without a
/// bound would pin that cap under a GPU that keeps failing and force a
/// synchronous mid-frame GPU wait on every allocation. Three covers the
/// case this queue exists for, a single transient abort, and turns a
/// permanent GPU failure back into bounded memory.
pub const MAX_REISSUE_ATTEMPTS: u8 = 3;

/// What [`UploadRecoveryQueue::settle`] decided about one drained upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadFate {
    /// Its submit retired and did not fail: free the payload.
    Released,
    /// Its submit was aborted: re-issue the copy, then [`UploadRecoveryQueue::requeue`] it.
    Reissue,
    /// Aborted more than [`MAX_REISSUE_ATTEMPTS`] times: free the payload and warn.
    Abandoned,
}

/// One upload the GPU has not acknowledged yet.
///
/// `key` names the destination the upload writes (a `BufferId` for a
/// `Staged` VB/IB, a `TextureId` for a mip blit). It only has to be
/// consistent within one queue: its job is to group the entries whose
/// replay order relative to each other matters. `payload` is whatever the
/// caller needs to re-issue the copy.
pub struct PendingUpload<T> {
    key: u64,
    seq: u64,
    attempts: u8,
    payload: T,
}

impl<T> PendingUpload<T> {
    /// The destination this upload writes.
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.key
    }

    /// Replays this entry has already been through.
    #[must_use]
    pub const fn attempts(&self) -> u8 {
        self.attempts
    }

    /// Borrow what the caller needs to re-issue the copy.
    #[must_use]
    pub const fn payload(&self) -> &T {
        &self.payload
    }

    /// Take the payload, consuming the entry.
    #[must_use]
    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// Uploads awaiting a GPU acknowledgement, ordered by submit seq.
///
/// Entries go in at the seq of the frame whose command buffer carries
/// their copy and come out of the front once that seq has retired.
/// [`Self::requeue`] pushes a replayed entry to the back under the new
/// frame's strictly larger seq, so the queue stays non-decreasing in seq
/// and the front-gated drain stays correct.
///
/// Replay order: a replayed copy writes bytes that a later, successful
/// copy to the same key may have superseded, so re-issuing one entry in
/// isolation can leave a stale range sitting on top of a fresh one.
/// [`Self::settle`] therefore pulls in every queued entry sharing a key
/// with an aborted one, including entries whose own submit has not
/// retired yet, and hands them back in seq order. Replaying a copy whose
/// original is still in flight writes the same bytes twice, which costs a
/// blit and changes nothing.
pub struct UploadRecoveryQueue<T> {
    pending: VecDeque<PendingUpload<T>>,
}

impl<T> UploadRecoveryQueue<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }

    /// Uploads currently awaiting acknowledgement.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Record an upload just encoded into the frame submitting as `seq`.
    ///
    /// `seq` must not go backwards across calls: the steady-state drain
    /// gates on the front entry alone, so an out-of-order push would
    /// strand later entries behind it. Every producer stamps the encoder's
    /// current submit seq, which only advances.
    pub fn push(&mut self, key: u64, seq: u64, payload: T) {
        debug_assert!(
            self.pending.back().is_none_or(|back| back.seq <= seq),
            "upload recovery queue must stay non-decreasing in seq",
        );
        self.pending.push_back(PendingUpload {
            key,
            seq,
            attempts: 0,
            payload,
        });
    }

    /// Re-queue an entry the caller has just re-issued, under `reissue_seq`.
    ///
    /// Lands at the back, which is what keeps the queue monotonic:
    /// `reissue_seq` is the frame being built, so it exceeds every seq
    /// already queued. The attempt count advances here and nowhere else,
    /// so [`MAX_REISSUE_ATTEMPTS`] bounds the replay chain even when the
    /// same entry keeps riding failing submits.
    pub fn requeue(&mut self, entry: PendingUpload<T>, reissue_seq: u64) {
        debug_assert!(
            self.pending
                .back()
                .is_none_or(|back| back.seq <= reissue_seq),
            "re-issued upload must land after every queued entry",
        );
        self.pending.push_back(PendingUpload {
            key: entry.key,
            seq: reissue_seq,
            attempts: entry.attempts.saturating_add(1),
            payload: entry.payload,
        });
    }

    /// Drain every upload the GPU has finished with, classifying each.
    ///
    /// `coherent_seq` is the highest submit seq the GPU retired and
    /// `failed_seq` the highest it aborted. The caller decides what counts
    /// as retired: the encoder passes the lower of its two retirement
    /// counters, because only the upload command buffer's own handler ever
    /// records that buffer's abort. Entries come back front-first,
    /// so a caller that re-issues in iteration order reproduces the
    /// original write order. Entries whose seq has not retired stay
    /// queued unless an aborted entry shares their key, in which case
    /// they are replayed too so the key's final bytes are the newest ones.
    pub fn settle(
        &mut self,
        coherent_seq: u64,
        failed_seq: u64,
    ) -> Vec<(UploadFate, PendingUpload<T>)> {
        let poisoned = self.poisoned_keys(failed_seq);
        if poisoned.is_empty() {
            // Steady state: nothing failed, so the front-gated walk touches
            // only the entries it releases.
            let retired = self.front_run(|entry| entry.seq <= coherent_seq);
            return self
                .pending
                .drain(..retired)
                .map(|entry| (UploadFate::Released, entry))
                .collect();
        }
        let mut out = Vec::new();
        // Recovery: a poisoned key drags its whole tail out of the queue,
        // so the walk cannot stop at the first unretired entry.
        let mut kept: VecDeque<PendingUpload<T>> = VecDeque::new();
        for entry in self.pending.drain(..) {
            let owed = poisoned.contains(&entry.key);
            if !owed && entry.seq > coherent_seq {
                kept.push_back(entry);
                continue;
            }
            let fate = if !owed {
                UploadFate::Released
            } else if entry.attempts >= MAX_REISSUE_ATTEMPTS {
                UploadFate::Abandoned
            } else {
                UploadFate::Reissue
            };
            out.push((fate, entry));
        }
        self.pending = kept;
        out
    }

    /// Pop only the front entries the GPU acknowledged, leaving replays queued.
    ///
    /// For the callers that free memory without building a frame (the
    /// retention-cap relief drains): they have nowhere to put a replayed
    /// copy, so they must not take an entry that owes one. The walk stops at
    /// the first entry whose submit aborted, which also preserves the replay
    /// ordering [`Self::settle`] relies on: nothing behind an aborted entry
    /// is freed before that entry is replayed.
    pub fn release_acknowledged(
        &mut self,
        coherent_seq: u64,
        failed_seq: u64,
    ) -> Vec<PendingUpload<T>> {
        let clean = self.front_run(|entry| entry.seq <= coherent_seq && entry.seq > failed_seq);
        self.pending.drain(..clean).collect()
    }

    /// Length of the queue's leading run of entries satisfying `keep`.
    ///
    /// The front-gated drains take a prefix, and counting it first lets
    /// them use `drain`, which needs no fallible pop inside the loop.
    fn front_run(&self, keep: impl Fn(&PendingUpload<T>) -> bool) -> usize {
        self.pending.iter().take_while(|entry| keep(entry)).count()
    }

    /// Keys with at least one aborted entry, whose whole tail must replay.
    ///
    /// Empty on every frame the GPU did not fail, which is what keeps
    /// [`Self::settle`] on its cheap path.
    fn poisoned_keys(&self, failed_seq: u64) -> FxHashSet<u64> {
        let mut keys = FxHashSet::default();
        if failed_seq == 0 {
            return keys;
        }
        for entry in &self.pending {
            if entry.seq <= failed_seq {
                keys.insert(entry.key);
            }
        }
        keys
    }

    /// Take every queued entry, leaving the queue empty.
    ///
    /// Reset and shutdown call this after the GPU has gone idle: the
    /// payloads own Metal wrappers and PE-heap backings the caller must
    /// tear down in its own destroy-then-drop order.
    pub fn drain_all(&mut self) -> Vec<PendingUpload<T>> {
        self.pending.drain(..).collect()
    }
}

impl<T> Default for UploadRecoveryQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
