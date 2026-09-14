use std::{sync::mpsc::TryRecvError, time::Duration};

use super::*;

const DEADLINE: Duration = Duration::from_secs(5);

#[test]
fn rejected_spawn_disables_cache_and_releases_shutdown() {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_for_thread = Arc::clone(&ran);
    let (mut prewarm, receiver) = PrewarmHandle::spawn_with(
        move |_| {
            ran_for_thread.store(true, Ordering::Release);
            Some(42)
        },
        |work| {
            drop(work);
            Err(io::Error::other("injected thread spawn failure"))
        },
    );
    assert_eq!(receiver.try_recv(), Err(TryRecvError::Disconnected));
    assert!(!ran.load(Ordering::Acquire));
    assert!(prewarm.join.is_none());

    let (shutdown_tx, shutdown_rx) = mpsc::sync_channel(1);
    let (done_tx, done_rx) = mpsc::channel();
    let encoder = thread::spawn(move || {
        let warm = receive(&receiver);
        shutdown_rx.recv().expect("shutdown intake");
        done_tx.send(warm).expect("report shutdown");
    });
    prewarm.cancel_and_join();
    prewarm.cancel_and_join();
    shutdown_tx.send(()).expect("queue shutdown");
    assert_eq!(done_rx.recv_timeout(DEADLINE), Ok(None));
    encoder.join().expect("encoder finished");
}

#[test]
fn frames_wait_for_successful_prewarm_payload() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (mut prewarm, receiver) = PrewarmHandle::spawn(move |_| {
        started_tx.send(()).expect("worker started");
        release_rx.recv_timeout(DEADLINE).expect("release prewarm");
        Some(vec![42])
    });
    started_rx.recv_timeout(DEADLINE).expect("prewarm running");
    assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));

    let (frame_tx, frame_rx) = mpsc::sync_channel(1);
    let (done_tx, done_rx) = mpsc::channel();
    frame_tx.send(1).expect("queue frame before prewarm");
    let encoder = thread::spawn(move || {
        let warm = receive(&receiver).expect("validated prewarm result");
        let frame = frame_rx.recv().expect("frame intake");
        done_tx.send((warm, frame)).expect("report frame");
        assert_eq!(frame_rx.recv(), Ok(2));
    });
    assert_eq!(done_rx.try_recv(), Err(TryRecvError::Empty));
    release_tx.send(()).expect("complete prewarm");
    assert_eq!(done_rx.recv_timeout(DEADLINE), Ok((vec![42], 1)));
    prewarm.cancel_and_join();
    frame_tx.send(2).expect("queue shutdown");
    encoder.join().expect("encoder finished");
}

#[test]
fn cancellation_finishes_worker_with_unread_payload() {
    let (started_tx, started_rx) = mpsc::channel();
    let (mut prewarm, receiver) = PrewarmHandle::spawn(move |stop| {
        started_tx.send(()).expect("worker started");
        let deadline = std::time::Instant::now() + DEADLINE;
        while !stop.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "cancellation timed out"
            );
            thread::yield_now();
        }
        Some(42)
    });
    started_rx.recv_timeout(DEADLINE).expect("prewarm running");
    prewarm.cancel_and_join();
    assert!(prewarm.join.is_none());
    assert_eq!(receiver.recv_timeout(DEADLINE), Ok(Some(42)));
    prewarm.cancel_and_join();
}

#[test]
fn unusable_cache_differs_from_validated_empty_cache() {
    let (mut prewarm, receiver) = PrewarmHandle::spawn(|_| None::<Vec<u8>>);
    assert_eq!(receive(&receiver), None);
    prewarm.cancel_and_join();

    let (mut prewarm, receiver) = PrewarmHandle::spawn(|_| Some(Vec::<u8>::new()));
    assert_eq!(receive(&receiver), Some(Vec::new()));
    prewarm.cancel_and_join();
}
