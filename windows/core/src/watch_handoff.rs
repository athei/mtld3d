//! The hand-off between a presenting thread and the thread that samples for it.
//!
//! The address-space watch samples on a thread of its own so the presenting
//! thread never pays for a walk, but some figures of a line can only be read
//! on the presenting thread. The presenting side ([`ApiLink`]) offers a sample
//! index every so many presents and, at every present, answers a request for
//! its figures if one waits. The sampling side ([`WorkerLink::run`]) takes the
//! sample, decides whether it logs anything, asks for the figures only then,
//! and logs. What a sample is, which lines are due and how they read is the
//! caller's ([`Watch`]); the protocol is here.
//!
//! Nothing on the presenting side waits. The offer is a `try_send` into a
//! one-deep queue, so an offer made while the sampler is still on an earlier
//! sample with one queued behind it is skipped. A request is one index in an
//! atomic slot that the sampler sets and the presenting side swaps out, and
//! the sampler withdraws a request it gives up on, so the slot holds at most
//! the one request the sampler waits for, and checking for it costs the
//! presenting side one load per present. Answers go into an unbounded
//! queue, so an answer never blocks and is never refused; there is at most one
//! per request, and the sampler requests at most once per sample.
//!
//! The sampler waits for an answer at most `figures_wait`: a presenting side
//! that stops presenting must not keep a crossed threshold's warning from the
//! log. Each answer echoes its request's index, and the sampler throws away an
//! answer whose index is not the one it waits for, which is how the late
//! answer to a request it gave up on is discarded. Dropping the [`ApiLink`]
//! ends the sampler: a wait for a sample or for figures returns at once, and a
//! sample queued before the drop is still taken first.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError},
    },
    time::{Duration, Instant},
};

/// The request slot's value while no request waits; no sample index takes it.
const NO_REQUEST: u32 = u32::MAX;

/// What the sampling thread does with a sample: the caller's half of the hand-off.
pub trait Watch {
    /// One sample as the sampling thread took it.
    type Sample;
    /// The figures only the presenting thread can read.
    type Figures;

    /// Take sample `index`: the walk, and whatever it advances.
    fn take(&mut self, index: u32) -> Self::Sample;

    /// Whether `sample` logs anything, and so needs the presenting thread's figures.
    fn lines_due(&self, sample: &Self::Sample) -> bool;

    /// Log what `sample` makes due, with the figures, or `None` when no answer came in time.
    fn log(&mut self, sample: &Self::Sample, figures: Option<&Self::Figures>);
}

/// What became of an offered sample.
#[derive(Debug, PartialEq, Eq)]
pub enum Offer {
    /// The sample is queued for the sampling thread.
    Queued,
    /// The sampling thread has a sample queued already; this one is skipped.
    Skipped,
    /// The sampling thread has ended; nothing takes samples any more.
    Ended,
}

/// The presenting thread's ends of the hand-off.
pub struct ApiLink<F> {
    samples: SyncSender<u32>,
    request: Arc<AtomicU32>,
    answers: Sender<Answer<F>>,
}

/// The sampling thread's ends of the hand-off.
pub struct WorkerLink<F> {
    samples: Receiver<u32>,
    request: Arc<AtomicU32>,
    answers: Receiver<Answer<F>>,
    figures_wait: Duration,
}

/// Both ends of a new hand-off whose sampler waits `figures_wait` at most for an answer.
#[must_use]
pub fn link<F>(figures_wait: Duration) -> (ApiLink<F>, WorkerLink<F>) {
    let (samples, sample_rx) = mpsc::sync_channel(1);
    let (answers, answer_rx) = mpsc::channel();
    let request = Arc::new(AtomicU32::new(NO_REQUEST));
    (
        ApiLink {
            samples,
            request: Arc::clone(&request),
            answers,
        },
        WorkerLink {
            samples: sample_rx,
            request,
            answers: answer_rx,
            figures_wait,
        },
    )
}

impl<F> ApiLink<F> {
    /// Offer sample `index` to the sampling thread, never waiting for it.
    ///
    /// # Panics
    ///
    /// If `index` is `u32::MAX`, the value the request slot keeps for none.
    #[must_use]
    pub fn offer(&self, index: u32) -> Offer {
        assert_ne!(index, NO_REQUEST, "u32::MAX is no sample index");
        match self.samples.try_send(index) {
            Ok(()) => Offer::Queued,
            Err(TrySendError::Full(_)) => Offer::Skipped,
            Err(TrySendError::Disconnected(_)) => Offer::Ended,
        }
    }

    /// The index of the sample whose figures the sampling thread waits for, taking the request.
    #[must_use]
    pub fn take_request(&self) -> Option<u32> {
        // A load first, so a present with no request pending writes nothing.
        if self.request.load(Ordering::Relaxed) == NO_REQUEST {
            return None;
        }
        let index = self.request.swap(NO_REQUEST, Ordering::Relaxed);
        (index != NO_REQUEST).then_some(index)
    }

    /// Answer the request for sample `index`; `false` when the sampling thread has ended.
    #[must_use]
    pub fn answer(&self, index: u32, figures: F) -> bool {
        self.answers.send(Answer { index, figures }).is_ok()
    }
}

impl<F> WorkerLink<F> {
    /// Take every sample offered, and log what each makes due, until the presenting side goes.
    ///
    /// Returns when the [`ApiLink`] has been dropped: at the next wait for a
    /// sample once the queue is empty, or at once from a wait for figures.
    pub fn run<W: Watch<Figures = F>>(self, watch: &mut W) {
        while let Ok(index) = self.samples.recv() {
            let sample = watch.take(index);
            if !watch.lines_due(&sample) {
                continue;
            }
            // The slot is empty here: every wait before this one ended with
            // its request taken by a present or withdrawn on timeout.
            self.request.store(index, Ordering::Relaxed);
            match self.await_answer(index) {
                Waited::Answered(figures) => watch.log(&sample, Some(&figures)),
                Waited::TimedOut => watch.log(&sample, None),
                Waited::Ended => return,
            }
        }
    }

    /// Wait at most `figures_wait` for the answer to the request for `index`.
    fn await_answer(&self, index: u32) -> Waited<F> {
        let deadline = Instant::now() + self.figures_wait;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.answers.recv_timeout(left) {
                Ok(answer) if answer.index == index => return Waited::Answered(answer.figures),
                // The late answer to a request this thread gave up on.
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    // Withdraw the request so no later present computes
                    // figures for it; if a present took it already, its
                    // answer is thrown away at the next wait.
                    let _ = self.request.compare_exchange(
                        index,
                        NO_REQUEST,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    return Waited::TimedOut;
                }
                Err(RecvTimeoutError::Disconnected) => return Waited::Ended,
            }
        }
    }
}

/// How a wait for an answer ended.
enum Waited<F> {
    /// The answer to the request waited for.
    Answered(F),
    /// No answer to it within `figures_wait`.
    TimedOut,
    /// The presenting side is gone.
    Ended,
}

/// An answer, with the index of the request it answers.
struct Answer<F> {
    index: u32,
    figures: F,
}

#[cfg(test)]
mod tests;
