//! Keyed index of built Metal objects, failures included.
//!
//! A draw names a shader library by its source key and a render pipeline by
//! its pipeline key, and the encoder resolves that key to the handles of the
//! built object. The build can fail (the emitter rejects the program, Metal
//! rejects the MSL or the pipeline descriptor), and a failure is as much an
//! answer as a success: the same key yields the same inputs, so building it
//! again repeats the whole build for the same result. The index therefore
//! remembers both outcomes, and a draw whose key has failed is dropped on one
//! probe.

use std::{borrow::Borrow, hash::Hash};

use rustc_hash::FxHashMap;

/// What a [`BuildIndex`] knows about one key.
#[derive(Debug, PartialEq, Eq)]
pub enum BuildLookup<H> {
    /// The key was built and resolved to these handles.
    Ready(H),
    /// The key was built before and failed; building it again is wasted work.
    Failed,
    /// The key has not been built yet.
    Unknown,
}

/// The outcome of every build, by key.
///
/// `H` is the handle bundle of a built object. It is a few words wide and
/// handed out by value, which is why the index asks for `Copy`.
pub struct BuildIndex<K, H> {
    entries: FxHashMap<K, Option<H>>,
}

impl<K, H> Default for BuildIndex<K, H> {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }
}

impl<K: Eq + Hash, H: Copy> BuildIndex<K, H> {
    /// What is known about `key`, in one probe and without cloning it.
    pub fn lookup<Q>(&self, key: &Q) -> BuildLookup<H>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        match self.entries.get(key) {
            Some(Some(handles)) => BuildLookup::Ready(*handles),
            Some(None) => BuildLookup::Failed,
            None => BuildLookup::Unknown,
        }
    }

    /// Remember how building `key` ended, `None` being a failure.
    pub fn record(&mut self, key: K, outcome: Option<H>) {
        self.entries.insert(key, outcome);
    }

    /// Forget the failed keys so each is built once more; successes stay.
    ///
    /// For a boundary after which a failure may no longer hold, such as a
    /// device reset following a compiler service that went away mid-build.
    pub fn forget_failures(&mut self) {
        self.entries.retain(|_, outcome| outcome.is_some());
    }

    /// The handles of every key that built, in no particular order.
    pub fn ready(&self) -> impl Iterator<Item = H> + '_ {
        self.entries.values().filter_map(|outcome| *outcome)
    }

    /// How many keys the index has an answer for, failures included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index has no answer for any key.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Forget every key.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests;
