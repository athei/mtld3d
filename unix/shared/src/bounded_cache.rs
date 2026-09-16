//! A least-recently-used cache with a fixed capacity.
//!
//! Sized for the cursor caches: a few dozen entries keyed by content hash,
//! where a miss costs one re-upload and unbounded growth would cost a GDI
//! handle or a Metal texture per new key for the life of the process. The
//! entries live in a `Vec` and every lookup is a linear scan, which at these
//! sizes is cheaper than hashing and keeps this crate free of a map dependency.

pub struct BoundedCache<K, V> {
    capacity: usize,
    /// Advances on every insert and hit; the entry with the smallest stamp goes first.
    tick: u64,
    entries: Vec<Entry<K, V>>,
}

struct Entry<K, V> {
    key: K,
    used: u64,
    value: V,
}

impl<K: PartialEq, V> BoundedCache<K, V> {
    /// A cache holding at most `capacity` entries; zero is taken as one.
    #[must_use]
    pub const fn new(capacity: usize) -> Self {
        Self {
            capacity: if capacity == 0 { 1 } else { capacity },
            tick: 0,
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `key` is cached, without counting as a use.
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.entries.iter().any(|entry| entry.key == *key)
    }

    /// The value for `key`, which becomes the most recently used entry.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let tick = self.next_tick();
        let entry = self.entries.iter_mut().find(|entry| entry.key == *key)?;
        entry.used = tick;
        Some(&entry.value)
    }

    /// Insert or replace `key`; a new key that does not fit evicts and returns the oldest entry.
    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        let tick = self.next_tick();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.key == key) {
            entry.used = tick;
            entry.value = value;
            return None;
        }
        let oldest = if self.entries.len() >= self.capacity {
            self.entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(index, _)| index)
        } else {
            None
        };
        let evicted = oldest.map(|index| {
            let entry = self.entries.swap_remove(index);
            (entry.key, entry.value)
        });
        self.entries.push(Entry {
            key,
            used: tick,
            value,
        });
        evicted
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.entries.iter().position(|entry| entry.key == *key)?;
        Some(self.entries.swap_remove(index).value)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Take every entry out, in no particular order.
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.entries.drain(..).map(|entry| (entry.key, entry.value))
    }

    const fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }
}

#[cfg(test)]
mod tests;
