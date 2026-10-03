//! The encoder's small memo of recent pipeline snapshots and the pipelines they resolved to.
//!
//! A draw's [`PipelineSnapshot`] names its pipeline, and the draws of a frame
//! reuse a small working set of them: a run of draws shares one, and runs
//! alternate among a few (blend modes and shader variants of one kind of
//! object, a UI element and its masked quads). An equal snapshot means an
//! identical pipeline key, so the memo answers without building the key or
//! probing the pipeline cache.
//!
//! The memo holds only pipelines that were built, and it is never
//! invalidated: the pipeline cache it fronts never evicts, so a
//! snapshot-to-handle mapping stays valid for the device's lifetime. A
//! pending or failed snapshot is not recorded and goes to the cache on
//! every draw.

use crate::pipeline_state::PipelineSnapshot;

/// Snapshots the memo holds.
///
/// 24 entries do not hold every pipeline of a busy frame, only the ones its
/// draws alternate among: the `wow112` benchmark builds 88 pipelines, and
/// the memo answers 98.6 % of its draws, against 97.8 to 97.9 % in the World
/// of Warcraft 1.12 window that scene is calibrated on (WoW-14623). The
/// 3.3.5a frame needs 15.
pub const PIPELINE_MEMO_ENTRIES: usize = 24;

// The tag scan collects its candidates in a `u32` bit mask, one bit per entry.
const _: () = assert!(PIPELINE_MEMO_ENTRIES <= 32);

/// The last [`PIPELINE_MEMO_ENTRIES`] built snapshots, with least-recently-used replacement.
///
/// `front` is a copy of the entry the previous hit or insert used, compared
/// first at a fixed place exactly as the one-entry memo did. The entries are
/// found by an exact compare of a tag built from the shader functions and
/// the declaration, then the full snapshot compare; a hit there becomes the
/// new front.
pub struct PipelineMemo {
    front: Option<(PipelineSnapshot, u64)>,
    /// The entry `front` copies.
    front_at: usize,
    entries: [Option<PipelineSnapshot>; PIPELINE_MEMO_ENTRIES],
    handles: [u64; PIPELINE_MEMO_ENTRIES],
    tags: [u64; PIPELINE_MEMO_ENTRIES],
    /// When each entry was last used; zero for an entry never filled.
    stamps: [u64; PIPELINE_MEMO_ENTRIES],
    clock: u64,
}

impl Default for PipelineMemo {
    fn default() -> Self {
        Self {
            front: None,
            front_at: 0,
            entries: core::array::from_fn(|_| None),
            handles: [0; PIPELINE_MEMO_ENTRIES],
            tags: [0; PIPELINE_MEMO_ENTRIES],
            stamps: [0; PIPELINE_MEMO_ENTRIES],
            clock: 0,
        }
    }
}

impl PipelineMemo {
    /// The pipeline an equal snapshot resolved to, if the memo holds one.
    #[inline]
    pub fn lookup(&mut self, snapshot: &PipelineSnapshot) -> Option<u64> {
        if let Some((front, handle)) = &self.front
            && front == snapshot
        {
            return Some(*handle);
        }
        self.lookup_entries(snapshot)
    }

    /// Remember that `snapshot` resolved to the built pipeline `handle`.
    ///
    /// Replaces the least recently used entry, or an empty one. The caller
    /// records only a snapshot [`Self::lookup`] did not answer, so the memo
    /// holds each snapshot once.
    pub fn record(&mut self, snapshot: &PipelineSnapshot, handle: u64) {
        let victim = self.least_recent();
        self.entries[victim] = Some(snapshot.clone());
        self.handles[victim] = handle;
        self.tags[victim] = tag(snapshot);
        self.touch(victim);
    }

    /// How many entries hold a snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stamps.iter().filter(|&&stamp| stamp != 0).count()
    }

    /// Whether no entry holds a snapshot.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The entries other than the front one, filtered by tag and compared in full.
    fn lookup_entries(&mut self, snapshot: &PipelineSnapshot) -> Option<u64> {
        let wanted = tag(snapshot);
        let mut candidates = 0_u32;
        for (at, (&tag, &stamp)) in self.tags.iter().zip(&self.stamps).enumerate() {
            candidates |= u32::from(tag == wanted && stamp != 0) << at;
        }
        // `lookup` already compared the front entry's copy.
        candidates &= !(1 << self.front_at);
        while candidates != 0 {
            let at = candidates.trailing_zeros() as usize;
            candidates &= candidates - 1;
            if self.entries[at].as_ref() == Some(snapshot) {
                self.touch(at);
                return Some(self.handles[at]);
            }
        }
        None
    }

    fn least_recent(&self) -> usize {
        let mut victim = 0;
        for (at, &stamp) in self.stamps.iter().enumerate() {
            if stamp < self.stamps[victim] {
                victim = at;
            }
        }
        victim
    }

    /// Stamp entry `at` as the most recently used and make it the front.
    fn touch(&mut self, at: usize) {
        self.clock += 1;
        self.stamps[at] = self.clock;
        self.front_at = at;
        self.front = self.entries[at]
            .as_ref()
            .map(|snapshot| (snapshot.clone(), self.handles[at]));
    }
}

/// A pre-filter over the fields that most often tell two snapshots apart.
///
/// Equal snapshots have equal tags; the full compare decides.
const fn tag(snapshot: &PipelineSnapshot) -> u64 {
    snapshot.vs_fn.raw() ^ snapshot.ps_fn.raw().rotate_left(32) ^ snapshot.vdecl_hash
}

#[cfg(test)]
mod tests;
