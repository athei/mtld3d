use std::{
    sync::mpsc::RecvTimeoutError,
    thread::{self, JoinHandle},
};

use super::*;

/// Long enough that no test waits it out unless it means to.
const LONG_WAIT: Duration = Duration::from_secs(60);

/// How long a test waits for an event before it fails.
const DEADLINE: Duration = Duration::from_secs(10);

/// What the mock watch reports, in order.
#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// `take` began on this sample.
    Taking(u32),
    /// `log` ran on this sample with these figures.
    Logged(u32, Option<&'static str>),
}

/// A watch whose samples are their indices, and which reports every step it takes.
struct MockWatch {
    events: Sender<Event>,
    /// When set, every `take` waits for one message on it before it returns.
    gate: Option<Receiver<()>>,
    lines_due: bool,
}

impl Watch for MockWatch {
    type Sample = u32;
    type Figures = &'static str;

    fn take(&mut self, index: u32) -> u32 {
        self.events
            .send(Event::Taking(index))
            .expect("the test listens");
        if let Some(gate) = &self.gate {
            gate.recv().expect("the test opens the gate");
        }
        index
    }

    fn lines_due(&self, _sample: &u32) -> bool {
        self.lines_due
    }

    fn log(&mut self, sample: &u32, figures: Option<&&'static str>) {
        self.events
            .send(Event::Logged(*sample, figures.copied()))
            .expect("the test listens");
    }
}

/// Run the sampler of `worker` with `watch` on a thread of its own.
fn start(worker: WorkerLink<&'static str>, mut watch: MockWatch) -> JoinHandle<()> {
    thread::spawn(move || worker.run(&mut watch))
}

/// The next event, failing the test after [`DEADLINE`].
fn next(events: &Receiver<Event>) -> Event {
    events
        .recv_timeout(DEADLINE)
        .expect("an event within the deadline")
}

/// Wait until the sampler has asked for the figures of `index`, leaving the request in place.
fn await_request<F>(api: &ApiLink<F>, index: u32) {
    let deadline = Instant::now() + DEADLINE;
    while api.request.load(Ordering::Relaxed) != index {
        assert!(Instant::now() < deadline, "no request for sample {index}");
        thread::yield_now();
    }
}

#[test]
fn dropping_the_api_side_ends_a_sampler_waiting_for_figures() {
    let (api, worker) = link(LONG_WAIT);
    let (events, event_rx) = mpsc::channel();
    let sampler = start(
        worker,
        MockWatch {
            events,
            gate: None,
            lines_due: true,
        },
    );
    assert_eq!(api.offer(0), Offer::Queued);
    await_request(&api, 0);
    let dropped = Instant::now();
    drop(api);
    sampler.join().expect("the sampler ends cleanly");
    assert!(
        dropped.elapsed() < LONG_WAIT / 2,
        "the sampler ended at the drop, not at the figure wait's end"
    );
    assert_eq!(next(&event_rx), Event::Taking(0));
    assert_eq!(
        event_rx.recv_timeout(Duration::ZERO),
        Err(RecvTimeoutError::Disconnected),
        "a sampler whose presenting side went logs nothing for the sample"
    );
}

#[test]
fn a_sample_queued_before_the_drop_is_taken_before_the_sampler_ends() {
    let (api, worker) = link(LONG_WAIT);
    let (events, event_rx) = mpsc::channel();
    let (gate, gate_rx) = mpsc::channel();
    let sampler = start(
        worker,
        MockWatch {
            events,
            gate: Some(gate_rx),
            lines_due: false,
        },
    );
    assert_eq!(api.offer(0), Offer::Queued);
    assert_eq!(next(&event_rx), Event::Taking(0));
    assert_eq!(
        api.offer(1),
        Offer::Queued,
        "one sample queues behind the one in progress"
    );
    drop(api);
    gate.send(()).expect("the sampler waits in take");
    gate.send(()).expect("the sampler takes the queued sample");
    sampler.join().expect("the sampler ends cleanly");
    assert_eq!(next(&event_rx), Event::Taking(1));
}

#[test]
fn an_offer_while_the_sampler_is_behind_is_skipped_without_waiting() {
    let (api, worker) = link(LONG_WAIT);
    let (events, event_rx) = mpsc::channel();
    let (gate, gate_rx) = mpsc::channel();
    let sampler = start(
        worker,
        MockWatch {
            events,
            gate: Some(gate_rx),
            lines_due: false,
        },
    );
    assert_eq!(api.offer(0), Offer::Queued);
    assert_eq!(next(&event_rx), Event::Taking(0));
    assert_eq!(api.offer(1), Offer::Queued);
    let offered = Instant::now();
    assert_eq!(
        api.offer(2),
        Offer::Skipped,
        "a second sample behind is skipped"
    );
    assert!(
        offered.elapsed() < DEADLINE,
        "the skipped offer returned without waiting for the sampler"
    );
    drop(api);
    gate.send(()).expect("release sample 0");
    gate.send(()).expect("release sample 1");
    sampler.join().expect("the sampler ends cleanly");
    assert_eq!(next(&event_rx), Event::Taking(1));
    assert_eq!(
        event_rx.recv_timeout(Duration::ZERO),
        Err(RecvTimeoutError::Disconnected),
        "the skipped sample is never taken"
    );
}

#[test]
fn an_offer_after_the_sampler_ended_says_so() {
    let (api, worker) = link::<&'static str>(LONG_WAIT);
    drop(worker);
    assert_eq!(api.offer(0), Offer::Ended);
    assert!(!api.answer(0, "late"), "nothing takes an answer either");
}

#[test]
fn each_answer_pairs_with_its_own_request_and_a_late_one_is_thrown_away() {
    let (api, worker) = link(Duration::from_millis(50));
    let (events, event_rx) = mpsc::channel();
    let sampler = start(
        worker,
        MockWatch {
            events,
            gate: None,
            lines_due: true,
        },
    );
    assert_eq!(api.offer(0), Offer::Queued);
    assert_eq!(next(&event_rx), Event::Taking(0));
    assert_eq!(
        next(&event_rx),
        Event::Logged(0, None),
        "an unanswered request logs without the figures"
    );
    assert_eq!(
        api.take_request(),
        None,
        "the sampler withdrew the request it gave up on"
    );
    // A present that took the request just before the withdrawal answers late.
    assert!(api.answer(0, "late"));

    assert_eq!(api.offer(1), Offer::Queued);
    assert_eq!(next(&event_rx), Event::Taking(1));
    let deadline = Instant::now() + DEADLINE;
    let index = loop {
        if let Some(index) = api.take_request() {
            break index;
        }
        assert!(Instant::now() < deadline, "no request for sample 1");
        thread::yield_now();
    };
    assert_eq!(index, 1, "the request names its sample");
    assert!(api.answer(index, "fresh"));
    assert_eq!(
        next(&event_rx),
        Event::Logged(1, Some("fresh")),
        "sample 1 logs its own answer, not the late one to sample 0"
    );
    assert_eq!(api.take_request(), None, "an answered request is gone");
    drop(api);
    sampler.join().expect("the sampler ends cleanly");
}

#[test]
fn a_newer_request_replaces_one_never_taken() {
    let (api, worker) = link::<&'static str>(LONG_WAIT);
    worker.request.store(3, Ordering::Relaxed);
    worker.request.store(4, Ordering::Relaxed);
    assert_eq!(api.take_request(), Some(4));
    assert_eq!(api.take_request(), None);
}
