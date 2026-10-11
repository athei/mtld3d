//! When a presented frame counts as finished, and the `Present` wait for one.
//!
//! A device keeps at most its maximum frame latency of presented frames
//! unfinished: at the end of `Present(k)` the API thread waits until frame
//! `k - L` has finished. A frame finishes when its draw command buffer has
//! ended on the GPU, which the device's draw counter says, and, when its
//! present is paced (it carries a minimum duration: vsync, a divided
//! interval or the frame cap), once that present has reached the screen or
//! been discarded. Into a window the compositor does not show, the presenter
//! holds each paced present to that duration on its own clock before it
//! skips the drawable, or acquires one under `debug.presentOccluded`; into a
//! visible window whose recent paced presents all came back undisplayed it
//! holds them the same way and still presents them. An `IMMEDIATE` present
//! without a frame cap counts GPU completion alone.
//!
//! The paced half is the device's [`FrameSignal`]: the submit thread notes
//! each submission before its draw buffer commits, the presenter resolves a
//! paced present when the drawable is presented or the packet is dropped,
//! and the signal publishes into a PE-owned counter the sequence below which
//! nothing paced is outstanding, so a `Present` whose target has finished
//! never crosses into native code. The wait itself runs on the API thread
//! and wakes on every event that can end it: a draw buffer retiring, a paced
//! present resolving, a `Reset`, a present the presenter drops because the
//! window is covered or no drawable came, a failure that ends the encoder,
//! and teardown. The paced half of a wait is also capped at
//! [`PACED_PRESENT_CAP`], or three of the present's minimum durations when
//! that is longer, logged once when it fires, since what shows a present is
//! the compositor's and not ours; the presents it gave up on are released,
//! so the `Present` after it does not wait for them again.

use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

use super::submission::RetirementCounter;
use crate::LOG_TARGET;

/// How long a wait gives a paced present to be shown once its frame's GPU work has finished.
///
/// The layer's drawable timeout (`allowsNextDrawableTimeout`, documented
/// as one second): the presenter gives up on a drawable after as long, so
/// a present still unshown that much later is held by the compositor, not
/// by the GPU or the presenter.
pub const PACED_PRESENT_CAP: Duration = Duration::from_secs(1);

/// How a frame wait ended.
#[derive(Debug, PartialEq, Eq)]
pub enum FrameWait {
    /// The frame finished.
    Finished,
    /// The encoder ended (a failure, or teardown) before it did; nothing more will finish.
    Ended,
}

/// One device's frame-latency signal, shared by its submit, presenter and API threads.
pub struct FrameSignal {
    state: Mutex<SignalState>,
    /// Woken when anything a [`FrameSignal::wait`] reads may have moved.
    changed: Condvar,
}

/// A paced present still owed to the frame-latency signal.
///
/// Rides the present packet and, once the present commits, the drawable's
/// presented handler. Resolving is idempotent, and dropping resolves, so a
/// packet the presenter drops, a handler Metal releases without calling, and
/// a `Reset` that already cleared the entry all end it.
pub struct PacedMark {
    signal: Arc<FrameSignal>,
    seq: u64,
}

struct SignalState {
    /// Sequences of paced presents not yet presented or discarded, ascending.
    paced: VecDeque<u64>,
    /// Newest submission the submit thread noted.
    latest: u64,
    /// The minimum present duration of the newest paced present: the display's pace.
    pacing: Duration,
    /// The PE-owned counter the paced floor is published into; none once ended.
    floor: RetirementCounter,
    /// API threads inside [`FrameSignal::wait`].
    waiters: u32,
    /// The encoder ended; no frame outstanding now will finish.
    ended: bool,
}

impl Default for FrameSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameSignal {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(SignalState {
                paced: VecDeque::new(),
                latest: 0,
                pacing: Duration::ZERO,
                floor: RetirementCounter::NONE,
                waiters: 0,
                ended: true,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SignalState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start serving an encoder whose paced floor is `floor`.
    pub fn attach(&self, floor: RetirementCounter) {
        let mut state = self.lock();
        state.paced.clear();
        state.latest = 0;
        state.pacing = Duration::ZERO;
        state.floor = floor;
        state.ended = false;
    }

    /// Note a submission before its draw buffer commits; a paced present gets its mark.
    ///
    /// `pacing` is the present's minimum duration when it waits for the
    /// display, `None` for any other submission. Every submission is noted,
    /// presenting or not, so the floor reaches a sequence once nothing paced
    /// at or below it is outstanding. Noting before the commit is what lets a
    /// reader of the draw counter trust the floor: a draw counter at a
    /// sequence means that sequence was noted.
    pub fn note_submission(
        self: &Arc<Self>,
        seq: u64,
        pacing: Option<Duration>,
    ) -> Option<PacedMark> {
        let mut state = self.lock();
        if state.ended || seq == 0 {
            return None;
        }
        state.latest = state.latest.max(seq);
        let mark = pacing.map(|pacing| {
            state.paced.push_back(seq);
            state.pacing = pacing;
            PacedMark {
                signal: Arc::clone(self),
                seq,
            }
        });
        self.publish(&state);
        drop(state);
        mark
    }

    /// Wake the waiters: a draw buffer of this device retired.
    pub fn retired(&self) {
        let state = self.lock();
        if state.waiters > 0 {
            self.changed.notify_all();
        }
    }

    /// Resolve every paced present still owed.
    ///
    /// A `Reset` calls it once it has drained presentation, since the layer
    /// the presents were waiting on may be replaced. The presenter calls it
    /// when it drops a present because no drawable came: the compositor has
    /// stopped handing this layer drawables, so the ones already committed
    /// may not be shown for as long as that lasts, and a `Present` waiting on
    /// one of them has to go on.
    pub fn release_paced(&self) {
        let mut state = self.lock();
        state.paced.clear();
        self.publish(&state);
        drop(state);
    }

    /// Resolve the paced presents at or below `seq`, and keep the ones after it owed.
    ///
    /// The presenter calls it when it drops the present of `seq` because the
    /// window is covered: the drawables committed before it may not be shown
    /// while that lasts, and the presents behind it are still to be held to
    /// the window's pace, one by one, when the presenter reaches them.
    pub fn release_paced_through(&self, seq: u64) {
        let mut state = self.lock();
        self.release_through(&mut state, seq);
        drop(state);
    }

    /// End the signal: the encoder failed or is going away, and nothing outstanding will finish.
    ///
    /// Also stops every later publication into the PE counter, which the PE
    /// side frees once the encoder is destroyed, so a presented handler that
    /// runs after that writes nothing.
    pub fn end(&self) {
        let mut state = self.lock();
        state.ended = true;
        state.paced.clear();
        state.floor = RetirementCounter::NONE;
        drop(state);
        self.changed.notify_all();
    }

    /// Wait until the frame submitted as `target` finished, or the signal ended.
    ///
    /// `draw` is the device's draw counter. Returns at once when the frame
    /// has finished already. Once the frame's GPU work has finished, the wait
    /// for its paced present lasts at most `paced_cap`, or three of the
    /// display's minimum present durations when that is longer, so a frame
    /// cap of a few hertz is not mistaken for a stalled compositor. When the
    /// cap fires, the frame counts as finished, the paced presents at or
    /// below it are released and a warning is logged once.
    pub fn wait(&self, draw: RetirementCounter, target: u64, paced_cap: Duration) -> FrameWait {
        let mut state = self.lock();
        state.waiters += 1;
        let mut gpu_finished_at: Option<Instant> = None;
        let outcome = loop {
            if state.ended {
                break FrameWait::Ended;
            }
            if draw.load() < target {
                state = self
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            if floor_of(&state) >= target {
                break FrameWait::Finished;
            }
            let cap = paced_cap.max(state.pacing.saturating_mul(3));
            let since = *gpu_finished_at.get_or_insert_with(Instant::now);
            let left = cap.saturating_sub(since.elapsed());
            if left.is_zero() {
                mtld3d_shared::log_once_warn!(
                    target: LOG_TARGET,
                    "frame latency: a paced present was not shown {cap:?} after its frame's GPU \
                     work finished; Present goes on without it",
                );
                self.release_through(&mut state, target);
                break FrameWait::Finished;
            }
            state = self
                .changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        };
        state.waiters -= 1;
        outcome
    }

    /// Resolve the paced present of `seq`; a sequence no longer listed changes nothing.
    fn resolve(&self, seq: u64) {
        let mut state = self.lock();
        let Some(index) = state.paced.iter().position(|&paced| paced == seq) else {
            return;
        };
        state.paced.remove(index);
        self.publish(&state);
        drop(state);
    }

    /// Resolve the paced presents at or below `seq` and publish, under the state's lock.
    fn release_through(&self, state: &mut SignalState, seq: u64) {
        state.paced.retain(|&paced| paced > seq);
        self.publish(state);
    }

    /// Publish the floor and wake the waiters, under the state's lock.
    fn publish(&self, state: &SignalState) {
        state.floor.publish(floor_of(state));
        if state.waiters > 0 {
            self.changed.notify_all();
        }
    }
}

impl PacedMark {
    /// The present this mark stands for reached the screen or was discarded.
    pub fn resolve(&self) {
        self.signal.resolve(self.seq);
    }
}

impl Drop for PacedMark {
    fn drop(&mut self) {
        self.signal.resolve(self.seq);
    }
}

/// The highest sequence at or below which no paced present is outstanding.
fn floor_of(state: &SignalState) -> u64 {
    state
        .paced
        .front()
        .map_or(state.latest, |&oldest| oldest - 1)
}

#[cfg(test)]
mod tests;
