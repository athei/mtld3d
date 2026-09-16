//! Unit tests for the bounded cache's eviction order.

use super::BoundedCache;

#[test]
fn the_least_recently_used_entry_is_evicted() {
    let mut cache = BoundedCache::new(3);
    assert!(cache.insert(1, "a").is_none());
    assert!(cache.insert(2, "b").is_none());
    assert!(cache.insert(3, "c").is_none());
    assert_eq!(cache.get(&1), Some(&"a"));
    assert_eq!(cache.insert(4, "d"), Some((2, "b")));
    assert!(cache.contains(&1) && cache.contains(&3) && cache.contains(&4));
    assert_eq!(cache.len(), 3);
}

#[test]
fn alternating_keys_within_capacity_never_evict() {
    let mut cache = BoundedCache::new(2);
    cache.insert(10, ());
    cache.insert(20, ());
    for _ in 0..100 {
        assert!(cache.get(&10).is_some());
        assert!(cache.get(&20).is_some());
    }
    assert_eq!(cache.len(), 2);
}

#[test]
fn contains_is_not_a_use_and_replacing_a_key_does_not_evict() {
    let mut cache = BoundedCache::new(2);
    cache.insert(1, 'a');
    cache.insert(2, 'b');
    assert!(cache.contains(&1));
    assert!(
        cache.insert(2, 'B').is_none(),
        "a replaced key keeps its slot"
    );
    assert_eq!(
        cache.insert(3, 'c'),
        Some((1, 'a')),
        "the untouched key goes"
    );
    assert_eq!(cache.get(&2), Some(&'B'));
}

#[test]
fn capacity_one_and_zero_hold_a_single_entry() {
    for capacity in [0, 1] {
        let mut cache = BoundedCache::new(capacity);
        assert!(cache.is_empty());
        assert!(cache.insert(1, 1).is_none());
        assert_eq!(cache.insert(2, 2), Some((1, 1)));
        assert_eq!(cache.len(), 1);
    }
}

#[test]
fn remove_clear_and_drain_give_the_entries_back() {
    let mut cache = BoundedCache::new(4);
    cache.insert(1, "a");
    cache.insert(2, "b");
    cache.insert(3, "c");
    assert_eq!(cache.remove(&2), Some("b"));
    assert_eq!(cache.remove(&2), None);
    let mut drained: Vec<_> = cache.drain().collect();
    drained.sort_unstable();
    assert_eq!(drained, vec![(1, "a"), (3, "c")]);
    assert!(cache.is_empty());
    cache.insert(5, "e");
    cache.clear();
    assert!(!cache.contains(&5));
}
