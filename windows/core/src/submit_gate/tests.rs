use std::{sync::Arc, thread, time::Duration};

use super::SubmitGate;

#[test]
fn a_published_sequence_is_ready_at_once() {
    let gate = SubmitGate::new();
    gate.publish(3);
    assert!(gate.wait_for(3, Duration::from_millis(10)));
    assert!(gate.wait_for(2, Duration::from_millis(10)));
    assert_eq!(gate.submitted(), 3);
}

#[test]
fn publishing_a_lower_sequence_keeps_the_highest() {
    let gate = SubmitGate::new();
    gate.publish(5);
    gate.publish(2);
    assert_eq!(gate.submitted(), 5);
}

#[test]
fn waiting_times_out_without_a_publish() {
    let gate = SubmitGate::new();
    assert!(!gate.wait_for(1, Duration::from_millis(20)));
    assert_eq!(gate.submitted(), 0);
}

#[test]
fn a_publish_from_another_thread_releases_the_wait() {
    let gate = Arc::new(SubmitGate::new());
    let publisher = Arc::clone(&gate);
    let handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        publisher.publish(7);
    });
    assert!(gate.wait_for(7, Duration::from_secs(5)));
    handle.join().expect("publisher thread finishes");
}
