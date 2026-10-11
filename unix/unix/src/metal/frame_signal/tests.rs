//! The frame-latency signal: what finishes a frame, and every event that wakes a waiter.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use super::*;

/// The minimum present duration the paced presents of these tests carry, far below any cap.
const PACING: Duration = Duration::from_millis(1);
/// How long a test lets a waiter run before it reads it as parked.
const PARKED: Duration = Duration::from_millis(50);
/// How long a test waits for a woken waiter to report.
const WOKEN: Duration = Duration::from_secs(5);

fn counter(atomic: &AtomicU64) -> RetirementCounter {
    // SAFETY: every test keeps its atomic alive past the signal's and the waiter's last use.
    unsafe { RetirementCounter::from_address(core::ptr::from_ref(atomic) as u64) }
}

fn attached(floor: &AtomicU64) -> Arc<FrameSignal> {
    let signal = Arc::new(FrameSignal::new());
    signal.attach(counter(floor));
    signal
}

/// Start a waiter for `target` on its own thread; its outcome arrives on the receiver.
fn spawn_waiter(
    signal: &Arc<FrameSignal>,
    draw: &Arc<AtomicU64>,
    target: u64,
) -> mpsc::Receiver<FrameWait> {
    let (sender, receiver) = mpsc::channel();
    let signal = Arc::clone(signal);
    let draw = Arc::clone(draw);
    thread::spawn(move || {
        let _ = sender.send(signal.wait(counter(&draw), target, PACED_PRESENT_CAP));
    });
    assert!(
        receiver.recv_timeout(PARKED).is_err(),
        "the waiter parks while its frame is unfinished"
    );
    receiver
}

fn shared_counter() -> Arc<AtomicU64> {
    Arc::new(AtomicU64::new(0))
}

/// An unpaced frame finishes with its draw buffer, and the floor follows the submissions.
#[test]
fn an_unpaced_frame_finishes_with_its_draw_buffer() {
    let draw = AtomicU64::new(0);
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    assert!(signal.note_submission(1, None).is_none());
    assert!(signal.note_submission(2, None).is_none());
    assert_eq!(
        floor.load(Ordering::Acquire),
        2,
        "nothing paced holds the floor"
    );
    draw.store(2, Ordering::Release);
    assert_eq!(
        signal.wait(counter(&draw), 2, PACED_PRESENT_CAP),
        FrameWait::Finished
    );
}

/// A paced present holds the floor below it until it resolves, whatever came after it.
#[test]
fn a_paced_present_holds_the_floor_until_it_resolves() {
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal
        .note_submission(3, Some(PACING))
        .expect("a paced present gets a mark");
    signal.note_submission(4, None);
    let later = signal.note_submission(5, Some(PACING)).expect("a mark");
    assert_eq!(floor.load(Ordering::Acquire), 2);
    later.resolve();
    assert_eq!(
        floor.load(Ordering::Acquire),
        2,
        "an older paced present still holds it"
    );
    mark.resolve();
    assert_eq!(floor.load(Ordering::Acquire), 5);
    drop(mark);
    drop(later);
    assert_eq!(
        floor.load(Ordering::Acquire),
        5,
        "resolving twice changes nothing"
    );
}

/// A draw buffer retiring wakes the waiter: the completion handler's and the CPU-failure path.
#[test]
fn a_retired_draw_buffer_wakes_the_waiter() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    signal.note_submission(7, None);
    let waiter = spawn_waiter(&signal, &draw, 7);
    draw.store(7, Ordering::Release);
    signal.retired();
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Finished));
}

/// The presented handler resolving a paced present wakes the waiter.
#[test]
fn a_presented_paced_present_wakes_the_waiter() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(4, Some(PACING)).expect("a mark");
    draw.store(4, Ordering::Release);
    let waiter = spawn_waiter(&signal, &draw, 4);
    mark.resolve();
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Finished));
}

/// A packet dropped unpresented, or a handler Metal releases uncalled, resolves through its drop.
#[test]
fn a_dropped_mark_wakes_the_waiter() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(4, Some(PACING)).expect("a mark");
    draw.store(4, Ordering::Release);
    let waiter = spawn_waiter(&signal, &draw, 4);
    drop(mark);
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Finished));
}

/// Releasing the paced presents, as a `Reset` or a dropped present does, wakes the waiter.
#[test]
fn releasing_the_paced_presents_wakes_the_waiter() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(6, Some(PACING)).expect("a mark");
    draw.store(6, Ordering::Release);
    let waiter = spawn_waiter(&signal, &draw, 6);
    signal.release_paced();
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Finished));
    assert_eq!(floor.load(Ordering::Acquire), 6);
    drop(mark);
}

/// Releasing through a sequence resolves the paced presents at or below it and wakes the waiter.
///
/// What a present dropped because its window is covered does: the presents
/// behind it stay owed until the presenter holds each to the window's pace.
#[test]
fn releasing_through_a_present_keeps_the_later_ones_owed() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let early = signal.note_submission(3, Some(PACING)).expect("a mark");
    let dropped = signal.note_submission(5, Some(PACING)).expect("a mark");
    let behind = signal.note_submission(7, Some(PACING)).expect("a mark");
    draw.store(7, Ordering::Release);
    let waiter = spawn_waiter(&signal, &draw, 5);
    signal.release_paced_through(5);
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Finished));
    assert_eq!(
        floor.load(Ordering::Acquire),
        6,
        "the present behind the release still holds the floor"
    );
    drop(behind);
    assert_eq!(floor.load(Ordering::Acquire), 7);
    drop(early);
    drop(dropped);
}

/// A failure that ends the encoder wakes the waiter of a frame that never committed.
#[test]
fn an_ended_encoder_wakes_the_waiter() {
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let waiter = spawn_waiter(&signal, &draw, 9);
    signal.end();
    assert_eq!(waiter.recv_timeout(WOKEN), Ok(FrameWait::Ended));
}

/// Once ended, the signal notes nothing and writes no more into the PE counter.
#[test]
fn an_ended_signal_publishes_nothing() {
    let draw = AtomicU64::new(0);
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(2, Some(PACING)).expect("a mark");
    signal.end();
    assert!(signal.note_submission(3, None).is_none());
    drop(mark);
    assert_eq!(
        floor.load(Ordering::Acquire),
        1,
        "the floor stays where the end left it"
    );
    assert_eq!(
        signal.wait(counter(&draw), 1, PACED_PRESENT_CAP),
        FrameWait::Ended
    );
}

/// A signal no encoder attached answers every wait as ended, so nothing can park on it.
#[test]
fn an_unattached_signal_never_parks() {
    let draw = AtomicU64::new(0);
    let signal = Arc::new(FrameSignal::new());
    assert!(signal.note_submission(1, Some(PACING)).is_none());
    assert_eq!(
        signal.wait(counter(&draw), 1, PACED_PRESENT_CAP),
        FrameWait::Ended
    );
}

/// A paced present nothing shows holds the wait only for the cap, once the GPU work is done.
#[test]
fn an_unshown_paced_present_holds_the_wait_only_for_the_cap() {
    const CAP: Duration = Duration::from_millis(100);
    let draw = AtomicU64::new(5);
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(5, Some(PACING)).expect("a mark");
    let started = std::time::Instant::now();
    assert_eq!(signal.wait(counter(&draw), 5, CAP), FrameWait::Finished);
    assert!(started.elapsed() >= CAP, "the wait lasted the cap");
    drop(mark);
}

/// The cap does not shorten a wait for GPU work still running.
#[test]
fn the_cap_starts_when_the_gpu_work_finishes() {
    const CAP: Duration = Duration::from_millis(20);
    let draw = shared_counter();
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(3, Some(PACING)).expect("a mark");
    let (sender, receiver) = mpsc::channel();
    {
        let signal = Arc::clone(&signal);
        let draw = Arc::clone(&draw);
        thread::spawn(move || {
            let _ = sender.send(signal.wait(counter(&draw), 3, CAP));
        });
    }
    assert!(
        receiver.recv_timeout(CAP * 5).is_err(),
        "the draw buffer still runs, so the cap has not started"
    );
    draw.store(3, Ordering::Release);
    signal.retired();
    assert_eq!(receiver.recv_timeout(WOKEN), Ok(FrameWait::Finished));
    drop(mark);
}

/// A fired cap releases the paced presents at or below its frame, so the next wait takes none.
#[test]
fn a_fired_cap_releases_the_presents_it_gave_up_on() {
    const CAP: Duration = Duration::from_millis(20);
    let draw = AtomicU64::new(6);
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let older = signal.note_submission(5, Some(PACING)).expect("a mark");
    let target = signal.note_submission(6, Some(PACING)).expect("a mark");
    let newer = signal.note_submission(7, Some(PACING)).expect("a mark");
    assert_eq!(signal.wait(counter(&draw), 6, CAP), FrameWait::Finished);
    assert_eq!(
        floor.load(Ordering::Acquire),
        6,
        "5 and 6 released, 7 still owed"
    );
    let again = std::time::Instant::now();
    assert_eq!(signal.wait(counter(&draw), 6, CAP), FrameWait::Finished);
    assert!(
        again.elapsed() < CAP,
        "the next wait for it returns at once"
    );
    drop((older, target, newer));
}

/// A display paced slower than the cap stretches it to three of its present durations.
#[test]
fn a_slow_display_pace_stretches_the_cap() {
    const CAP: Duration = Duration::from_millis(20);
    const SLOW: Duration = Duration::from_millis(100);
    let draw = AtomicU64::new(5);
    let floor = AtomicU64::new(0);
    let signal = attached(&floor);
    let mark = signal.note_submission(5, Some(SLOW)).expect("a mark");
    let started = std::time::Instant::now();
    assert_eq!(signal.wait(counter(&draw), 5, CAP), FrameWait::Finished);
    assert!(
        started.elapsed() >= SLOW * 3,
        "the wait lasted three present durations, not the cap"
    );
    drop(mark);
}
