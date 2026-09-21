//! Source-keyed index of resolved shader libraries, failures included.
//!
//! A draw names its shader by a source key and the encoder resolves that key
//! to the handles of a compiled library. The resolve can fail (the emitter
//! rejects the program, Metal rejects the MSL), and a failure is as much an
//! answer as a success: the same key yields the same source, so resolving it
//! again repeats the whole compile for the same result. The index therefore
//! remembers both outcomes, and a draw whose key has failed is dropped on one
//! probe.

use std::{borrow::Borrow, hash::Hash};

use rustc_hash::FxHashMap;

/// What a [`LibraryIndex`] knows about one source key.
#[derive(Debug, PartialEq, Eq)]
pub enum LibraryLookup<H> {
    /// The key resolved to these handles.
    Ready(H),
    /// The key was resolved before and failed; resolving it again is wasted work.
    Failed,
    /// The key has not been resolved yet.
    Unknown,
}

/// The outcome of every resolve, by source key.
///
/// `H` is the handle bundle of a resolved library. It is a few words wide and
/// handed out by value, which is why the index asks for `Copy`.
pub struct LibraryIndex<K, H> {
    entries: FxHashMap<K, Option<H>>,
}

impl<K, H> Default for LibraryIndex<K, H> {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }
}

impl<K: Eq + Hash, H: Copy> LibraryIndex<K, H> {
    /// What is known about `key`, in one probe and without cloning it.
    pub fn lookup<Q>(&self, key: &Q) -> LibraryLookup<H>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        match self.entries.get(key) {
            Some(Some(handles)) => LibraryLookup::Ready(*handles),
            Some(None) => LibraryLookup::Failed,
            None => LibraryLookup::Unknown,
        }
    }

    /// Remember how resolving `key` ended, `None` being a failure.
    pub fn record(&mut self, key: K, outcome: Option<H>) {
        self.entries.insert(key, outcome);
    }

    /// Forget the failed keys so each is resolved once more; successes stay.
    ///
    /// For a boundary after which a failure may no longer hold, such as a
    /// device reset following a compiler service that went away mid-build.
    pub fn forget_failures(&mut self) {
        self.entries.retain(|_, outcome| outcome.is_some());
    }

    /// Forget every key.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests;
