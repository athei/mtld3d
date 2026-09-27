use std::sync::{Arc, mpsc};

use super::{CalibrationFailed, ClockCalibration};

#[test]
fn delayed_frequency_remains_pending_then_publishes_full_width_value() {
    let clock = Arc::new(ClockCalibration::new());
    let writer = Arc::clone(&clock);
    let (start, wait) = mpsc::channel();
    let task = std::thread::spawn(move || {
        wait.recv().expect("release calibration");
        // SAFETY: this task is the sole publisher and owns its retained mailbox.
        unsafe { writer.publish_ready(0x1234_5678_9abc_def0) };
    });
    for _ in 0..100 {
        assert_eq!(clock.get(), Ok(None));
    }
    start.send(()).expect("worker alive");
    task.join().expect("calibration worker");
    assert_eq!(clock.get(), Ok(Some(0x1234_5678_9abc_def0)));
}

#[test]
fn failure_and_zero_frequency_are_terminal_errors() {
    let failed = ClockCalibration::new();
    // SAFETY: this thread is the sole publisher of each local mailbox.
    unsafe { failed.publish_failed() };
    assert_eq!(failed.get(), Err(CalibrationFailed));
    let zero = ClockCalibration::new();
    // SAFETY: this thread is the sole publisher of the local mailbox.
    unsafe { zero.publish_ready(0) };
    assert_eq!(zero.get(), Err(CalibrationFailed));
}

#[test]
fn worker_retains_mailbox_until_publication_and_join() {
    let clock = Arc::new(ClockCalibration::new());
    let weak = Arc::downgrade(&clock);
    let (start, wait) = mpsc::channel();
    let task = std::thread::spawn(move || {
        wait.recv().expect("release worker");
        // SAFETY: the worker owns the sole publication and the final strong reference.
        unsafe { clock.publish_failed() };
    });
    assert!(weak.upgrade().is_some());
    start.send(()).expect("worker alive");
    task.join().expect("worker joined");
    assert!(weak.upgrade().is_none());
}
