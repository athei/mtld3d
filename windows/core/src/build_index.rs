//! Keyed index of built Metal objects, failures included.
//!
//! A draw names a shader library by its source key and a render pipeline by
//! its pipeline key, and the encoder resolves that key to the handles of the
//! built object. The build can fail (the emitter rejects the program, Metal
//! rejects the MSL or the pipeline descriptor), and a failure is as much an
//! answer as a success: the same key yields the same inputs, so building it
//! again repeats the whole build for the same result. The index therefore
//! remembers both outcomes, and a draw whose key has failed is dropped on one
//! probe. A build handed to a worker thread is a third state, pending under
//! the ticket of its job, until the outcome is recorded over it.

use std::{borrow::Borrow, hash::Hash};

use rustc_hash::FxHashMap;

use crate::async_compile::JobTicket;

/// What a [`BuildIndex`] knows about one key.
#[derive(Debug, PartialEq, Eq)]
pub enum BuildLookup<H> {
    /// The key was built and resolved to these handles.
    Ready(H),
    /// The key was built before and failed; building it again is wasted work.
    Failed,
    /// The key is being built by the job holding this ticket.
    Pending(JobTicket),
    /// The key has not been built yet.
    Unknown,
}

/// One key's entry: built, failed, or waiting for its job.
enum Entry<H> {
    Ready(H),
    Failed,
    Pending(JobTicket),
}

/// The outcome of every build, by key.
///
/// `H` is the handle bundle of a built object. It is a few words wide and
/// handed out by value, which is why the index asks for `Copy`.
pub struct BuildIndex<K, H> {
    entries: FxHashMap<K, Entry<H>>,
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
            Some(Entry::Ready(handles)) => BuildLookup::Ready(*handles),
            Some(Entry::Failed) => BuildLookup::Failed,
            Some(Entry::Pending(ticket)) => BuildLookup::Pending(*ticket),
            None => BuildLookup::Unknown,
        }
    }

    /// Remember how building `key` ended, `None` being a failure.
    pub fn record(&mut self, key: K, outcome: Option<H>) {
        self.entries.insert(key, Self::entry_for(outcome));
    }

    /// Remember that the job holding `ticket` is building `key`.
    pub fn mark_pending(&mut self, key: K, ticket: JobTicket) {
        self.entries.insert(key, Entry::Pending(ticket));
    }

    /// Record the outcome of the job that was building `key`, by borrow.
    ///
    /// Answers `false` when the index holds nothing for `key`, which leaves
    /// it holding nothing: the key was never marked pending, or the index
    /// was cleared while the job ran.
    pub fn complete<Q>(&mut self, key: &Q, outcome: Option<H>) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let Some(entry) = self.entries.get_mut(key) else {
            return false;
        };
        *entry = Self::entry_for(outcome);
        true
    }

    /// Forget the failed keys so each is built once more; successes and pending builds stay.
    ///
    /// For a boundary after which a failure may no longer hold, such as a
    /// device reset that recreates its surfaces, following a compiler service
    /// that went away mid-build.
    pub fn forget_failures(&mut self) {
        self.entries
            .retain(|_, entry| !matches!(entry, Entry::Failed));
    }

    /// The handles of every key that built, in no particular order.
    pub fn ready(&self) -> impl Iterator<Item = H> + '_ {
        self.entries.values().filter_map(|entry| match entry {
            Entry::Ready(handles) => Some(*handles),
            Entry::Failed | Entry::Pending(_) => None,
        })
    }

    /// How many keys the index has an entry for, failures and pending builds included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index has no entry for any key.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Forget every key.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    const fn entry_for(outcome: Option<H>) -> Entry<H> {
        match outcome {
            Some(handles) => Entry::Ready(handles),
            None => Entry::Failed,
        }
    }
}

#[cfg(test)]
mod tests;
