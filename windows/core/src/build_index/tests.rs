use super::{BuildIndex, BuildLookup};
use crate::async_compile::TicketSource;

/// A resolver whose compile is counted and fails for the keys it is told to.
struct CountingResolver {
    index: BuildIndex<String, u64>,
    failing: Vec<&'static str>,
    compiles: u32,
}

impl CountingResolver {
    fn new(failing: &[&'static str]) -> Self {
        Self {
            index: BuildIndex::default(),
            failing: failing.to_vec(),
            compiles: 0,
        }
    }

    /// One draw's resolve: probe the index, compile on an unknown key, record.
    fn draw(&mut self, key: &str) -> Option<u64> {
        match self.index.lookup(key) {
            BuildLookup::Ready(handles) => return Some(handles),
            BuildLookup::Failed | BuildLookup::Pending(_) => return None,
            BuildLookup::Unknown => {}
        }
        self.compiles += 1;
        let outcome = (!self.failing.contains(&key)).then_some(u64::from(self.compiles));
        self.index.record(key.to_owned(), outcome);
        outcome
    }
}

/// A key that failed to compile is compiled once, however often it is drawn.
#[test]
fn a_failed_key_is_compiled_once_across_repeated_draws() {
    let mut resolver = CountingResolver::new(&["broken"]);
    for _ in 0..250 {
        assert_eq!(resolver.draw("broken"), None, "the draw is dropped");
    }
    assert_eq!(resolver.compiles, 1, "one compile for 250 draws");
    assert_eq!(resolver.index.lookup("broken"), BuildLookup::Failed);
}

/// A failure is remembered for its own key and leaves every other key alone.
#[test]
fn a_failure_does_not_reach_other_keys() {
    let mut resolver = CountingResolver::new(&["broken"]);
    assert_eq!(resolver.draw("broken"), None);
    let good = resolver.draw("good");
    assert!(good.is_some(), "an unrelated key still compiles");
    assert_eq!(resolver.draw("good"), good, "and is served from the index");
    assert_eq!(resolver.draw("broken"), None);
    assert_eq!(resolver.compiles, 2, "one compile per key");
    assert_eq!(resolver.index.lookup("never drawn"), BuildLookup::Unknown);
}

/// Forgetting failures buys each failed key one more compile and keeps the successes.
#[test]
fn forgetting_failures_retries_each_failed_key_once() {
    let mut resolver = CountingResolver::new(&["broken"]);
    let good = resolver.draw("good");
    assert_eq!(resolver.draw("broken"), None);
    resolver.index.forget_failures();
    assert_eq!(resolver.index.lookup("broken"), BuildLookup::Unknown);
    assert_eq!(
        resolver.index.lookup("good"),
        BuildLookup::Ready(good.unwrap())
    );
    for _ in 0..10 {
        assert_eq!(resolver.draw("broken"), None);
        assert_eq!(resolver.draw("good"), good);
    }
    assert_eq!(
        resolver.compiles, 3,
        "good once, broken once before and once after"
    );
}

/// A failure that stopped holding resolves after the failures are forgotten.
#[test]
fn a_forgotten_failure_can_succeed() {
    let mut resolver = CountingResolver::new(&["flaky"]);
    assert_eq!(resolver.draw("flaky"), None);
    resolver.failing.clear();
    assert_eq!(resolver.draw("flaky"), None, "still remembered as failed");
    resolver.index.forget_failures();
    assert!(resolver.draw("flaky").is_some());
    assert_eq!(resolver.compiles, 2);
}

/// Clearing forgets successes and failures alike.
#[test]
fn clear_forgets_every_key() {
    let mut resolver = CountingResolver::new(&["broken"]);
    assert!(resolver.draw("good").is_some());
    assert_eq!(resolver.draw("broken"), None);
    resolver.index.clear();
    assert_eq!(resolver.index.lookup("good"), BuildLookup::Unknown);
    assert_eq!(resolver.index.lookup("broken"), BuildLookup::Unknown);
}

/// Teardown walks the built handles only, while the size counts every answer.
#[test]
fn ready_skips_failures_and_len_counts_them() {
    let mut resolver = CountingResolver::new(&["broken", "also broken"]);
    assert!(resolver.index.is_empty());
    let first = resolver.draw("first").unwrap();
    assert_eq!(resolver.draw("broken"), None);
    let second = resolver.draw("second").unwrap();
    assert_eq!(resolver.draw("also broken"), None);
    let mut ready: Vec<u64> = resolver.index.ready().collect();
    ready.sort_unstable();
    assert_eq!(
        ready,
        [first, second],
        "a failed key has no handle to release"
    );
    assert_eq!(resolver.index.len(), 4);
    resolver.index.forget_failures();
    assert_eq!(resolver.index.len(), 2);
    assert_eq!(resolver.index.ready().count(), 2);
}

/// A pending key answers with its ticket until its outcome is recorded over it.
#[test]
fn a_pending_key_answers_its_ticket_until_completed() {
    let mut tickets = TicketSource::new();
    let mut index: BuildIndex<String, u64> = BuildIndex::default();
    let ticket = tickets.issue();
    index.mark_pending("vs".to_owned(), ticket);
    assert_eq!(index.lookup("vs"), BuildLookup::Pending(ticket));
    assert_eq!(index.ready().count(), 0, "a pending key has no handle yet");
    assert_eq!(index.len(), 1, "but it has an entry");
    assert!(index.complete("vs", Some(7)));
    assert_eq!(index.lookup("vs"), BuildLookup::Ready(7));
    let failing = tickets.issue();
    assert_ne!(failing, ticket, "every ticket is new");
    index.mark_pending("ps".to_owned(), failing);
    assert!(index.complete("ps", None));
    assert_eq!(index.lookup("ps"), BuildLookup::Failed);
}

/// Completing a key the index never marked pending leaves it unknown.
#[test]
fn completing_an_unmarked_key_records_nothing() {
    let mut index: BuildIndex<String, u64> = BuildIndex::default();
    assert!(!index.complete("never queued", Some(1)));
    assert_eq!(index.lookup("never queued"), BuildLookup::Unknown);
    assert!(index.is_empty());
}

/// Forgetting failures keeps the builds still in flight, so their jobs still land.
#[test]
fn forgetting_failures_keeps_pending_builds() {
    let mut tickets = TicketSource::new();
    let mut index: BuildIndex<String, u64> = BuildIndex::default();
    let ticket = tickets.issue();
    index.mark_pending("in flight".to_owned(), ticket);
    index.record("broken".to_owned(), None);
    index.forget_failures();
    assert_eq!(index.lookup("in flight"), BuildLookup::Pending(ticket));
    assert_eq!(index.lookup("broken"), BuildLookup::Unknown);
    assert!(index.complete("in flight", Some(3)));
    assert_eq!(index.lookup("in flight"), BuildLookup::Ready(3));
}
