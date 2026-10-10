use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64},
    mpsc,
};

use super::*;

#[test]
fn returns_once_the_thread_has_ended_and_sees_its_last_write() {
    let written = Arc::new(AtomicU64::new(0));
    let written_by_thread = Arc::clone(&written);
    let (go, wait_for_go) = mpsc::channel::<()>();
    let handle = thread::spawn(move || {
        wait_for_go.recv().expect("the test lets the thread run");
        thread::sleep(Duration::from_millis(20));
        written_by_thread.store(42, Ordering::Relaxed);
    });
    assert!(!handle.is_finished(), "the thread waits for the go");
    go.send(()).expect("the thread is waiting");
    wait_until_finished(handle);
    assert_eq!(
        written.load(Ordering::Relaxed),
        42,
        "the thread's last write is visible after the wait"
    );
}

#[test]
fn returns_at_once_for_a_thread_that_already_ended() {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_in_thread = Arc::clone(&ran);
    let handle = thread::spawn(move || ran_in_thread.store(true, Ordering::Relaxed));
    while !handle.is_finished() {
        thread::yield_now();
    }
    wait_until_finished(handle);
    assert!(
        ran.load(Ordering::Relaxed),
        "the thread ran before the wait"
    );
}

#[test]
fn drops_a_result_the_thread_returned() {
    let result = Arc::new(());
    let result_in_thread = Arc::clone(&result);
    let handle = thread::spawn(move || result_in_thread);
    wait_until_finished(handle);
    assert_eq!(
        Arc::strong_count(&result),
        1,
        "the wait drops the result with the handle"
    );
}
