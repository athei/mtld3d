//! The reentrant lock's contract: the owner re-enters, everyone else waits until depth zero.
//!
//! Every lock here lives on the test's stack and every guard is dropped before
//! the lock goes out of scope, which is the lifetime `enter` asks for; the
//! scoped threads join before that as well.

use core::cell::UnsafeCell;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use super::{ApiGuard, ApiLock};

fn depth(lock: &ApiLock) -> u32 {
    lock.holder.lock().expect("holder mutex").depth
}

fn is_held(lock: &ApiLock) -> bool {
    lock.holder.lock().expect("holder mutex").owner.is_some()
}

#[test]
fn reenters_on_the_owner_thread() {
    let lock = ApiLock::new();
    // SAFETY: every guard drops before `lock` does.
    let outer = unsafe { lock.enter() };
    // SAFETY: as above.
    let middle = unsafe { lock.enter() };
    // SAFETY: as above.
    let inner = unsafe { lock.enter() };
    assert_eq!(depth(&lock), 3);
    drop(inner);
    assert_eq!(depth(&lock), 2);
    assert!(is_held(&lock));
    drop(middle);
    drop(outer);
    assert_eq!(depth(&lock), 0);
    assert!(!is_held(&lock));
}

#[test]
fn excludes_a_second_thread() {
    let lock = ApiLock::new();
    let hold = Duration::from_millis(50);
    thread::scope(|scope| {
        // SAFETY: dropped below, inside the scope, before `lock` goes away.
        let guard = unsafe { lock.enter() };
        let started = Instant::now();
        let other = scope.spawn(|| {
            // SAFETY: the guard drops inside this scope, before `lock` does.
            let _guard = unsafe { lock.enter() };
            Instant::now()
        });
        thread::sleep(hold);
        let released = Instant::now();
        drop(guard);
        let acquired = other.join().expect("the other thread panicked");
        assert!(
            acquired >= released,
            "the second thread got in {:?} after the owner took the lock, before it released",
            acquired - started
        );
    });
}

#[test]
fn releases_at_depth_zero_not_before() {
    let lock = ApiLock::new();
    let acquired = AtomicBool::new(false);
    thread::scope(|scope| {
        // SAFETY: both guards drop inside the scope, before `lock` does.
        let outer = unsafe { lock.enter() };
        // SAFETY: as above.
        let inner = unsafe { lock.enter() };
        let other = scope.spawn(|| {
            // SAFETY: the guard drops inside this scope, before `lock` does.
            let _guard = unsafe { lock.enter() };
            acquired.store(true, Ordering::Release);
        });
        drop(inner);
        thread::sleep(Duration::from_millis(50));
        assert!(
            !acquired.load(Ordering::Acquire),
            "the other thread got in while the owner still held one level"
        );
        drop(outer);
        other.join().expect("the other thread panicked");
        assert!(acquired.load(Ordering::Acquire));
    });
}

#[test]
fn noop_guard_touches_nothing() {
    let lock = ApiLock::new();
    // SAFETY: dropped below, before `lock`.
    let guard = unsafe { lock.enter() };
    drop(ApiGuard::NOOP);
    assert_eq!(
        depth(&lock),
        1,
        "a no-op guard does not release a real level"
    );
    drop(guard);
    drop(ApiGuard::NOOP);
    assert!(!is_held(&lock));
}

/// A counter with no synchronisation of its own; the lock is what keeps its sum exact.
struct Counter(UnsafeCell<u64>);

// SAFETY: every access happens under an `ApiLock` guard, which admits one
// thread at a time.
unsafe impl Sync for Counter {}

impl Counter {
    /// Add one, under the lock the caller holds.
    fn bump(&self) {
        // SAFETY: the caller holds the lock, so this is the only access.
        unsafe { *self.0.get() += 1 };
    }

    /// The sum, once every thread that bumped it has joined.
    fn total(&self) -> u64 {
        // SAFETY: the caller joined every thread; nothing else touches it.
        unsafe { *self.0.get() }
    }
}

#[test]
fn many_threads_count_to_n() {
    const THREADS: u64 = 8;
    const INCREMENTS: u64 = 1000;
    let lock = ApiLock::new();
    let counter = Counter(UnsafeCell::new(0));
    thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                for _ in 0..INCREMENTS {
                    // SAFETY: the guard drops at the end of the iteration,
                    // inside the scope, before `lock` does.
                    let _guard = unsafe { lock.enter() };
                    counter.bump();
                }
            });
        }
    });
    assert_eq!(counter.total(), THREADS * INCREMENTS);
}
