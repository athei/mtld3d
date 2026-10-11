//! The frame-latency bound: how many presented frames a device lets the GPU fall behind.
//!
//! When `Present(k)` returns, frame `k - L` has finished, so at most `L`
//! presented frames are outstanding: in the native pipeline, queued on the
//! GPU, or (for a present that waits for the display) not yet on screen. `L`
//! is the device's maximum frame latency: what an extended device's
//! `SetMaximumFrameLatency` asked for, 1 to 30, or 3 when it asked for
//! nothing and on a plain device. `present.maxFrameLatency` in `mtld3d.conf`
//! replaces the game's value in either direction when it names a number.
//!
//! A frame is named by the submit sequence of its present-bearing
//! submission. A mid-frame flush takes a sequence too but is no frame; the
//! retirement counters are exact, so the frame's own sequence covers it.

use crate::{
    extended::{DEFAULT_FRAME_LATENCY, MAX_FRAME_LATENCY},
    query_fence::event_completed,
};

/// Presents the ring keeps: the newest and the `MAX_FRAME_LATENCY` before it.
const RING: usize = MAX_FRAME_LATENCY as usize + 1;

/// The frame latency a device enforces.
///
/// `requested` is what the game's last `SetMaximumFrameLatency` stored (1 to
/// 30), `None` when it never called it; `configured` is
/// `present.maxFrameLatency`, `0` for `auto`. A configured number wins over
/// the game's value whichever is larger; `auto` takes the game's value, and
/// the D3D9 default of 3 when there is none.
#[must_use]
pub const fn enforced_frame_latency(requested: Option<u32>, configured: u32) -> u32 {
    if configured != 0 {
        return configured;
    }
    match requested {
        Some(requested) => requested,
        None => DEFAULT_FRAME_LATENCY,
    }
}

/// Whether the frame submitted at `target` finished.
///
/// `draw_retired` is the draw counter, every draw buffer up to it ended;
/// `paced_retired` is the floor below which no present that waits for the
/// display is outstanding.
#[must_use]
pub const fn frame_finished(target: u64, draw_retired: u64, paced_retired: u64) -> bool {
    event_completed(target, draw_retired) && event_completed(target, paced_retired)
}

/// One device's frame latency and the sequences of its recent presents.
pub struct FrameLatency {
    /// What the game's `SetMaximumFrameLatency` stored, `None` until it calls it.
    requested: Option<u32>,
    /// `present.maxFrameLatency`, `0` for `auto`.
    configured: u32,
    /// The sequences of the last [`RING`] presents, newest at `newest`; zero where none was.
    presents: [u64; RING],
    newest: u8,
    /// Whether this device has logged a `SetMaximumFrameLatency` request yet.
    announced: bool,
}

impl FrameLatency {
    /// The bound of a device whose configuration sets `present.maxFrameLatency` to `configured`.
    #[must_use]
    pub const fn new(configured: u32) -> Self {
        Self {
            requested: None,
            configured,
            presents: [0; RING],
            newest: 0,
            announced: false,
        }
    }

    /// The value `GetMaximumFrameLatency` reports: the game's own, or the default.
    #[must_use]
    pub const fn reported(&self) -> u32 {
        match self.requested {
            Some(requested) => requested,
            None => DEFAULT_FRAME_LATENCY,
        }
    }

    /// `present.maxFrameLatency`, `0` for `auto`.
    #[must_use]
    pub const fn configured(&self) -> u32 {
        self.configured
    }

    /// The bound `Present` enforces.
    #[must_use]
    pub const fn enforced(&self) -> u32 {
        enforced_frame_latency(self.requested, self.configured)
    }

    /// Store a validated `SetMaximumFrameLatency` value; `true` the first time on this device.
    ///
    /// The first answer is what the device logs once: the request, the
    /// configured value and what it enforces.
    pub const fn set(&mut self, requested: u32) -> bool {
        self.requested = Some(requested);
        !core::mem::replace(&mut self.announced, true)
    }

    /// Record the present submitted at `seq`; the frame `Present` waits for before returning.
    ///
    /// That is the present the enforced latency counts back from this one,
    /// `None` while the device has not presented that many frames yet.
    ///
    /// # Panics
    /// Never: the ring index stays below the ring's length.
    pub fn record_present(&mut self, seq: u64) -> Option<u64> {
        let newest = (usize::from(self.newest) + 1) % RING;
        self.newest = u8::try_from(newest).expect("the ring index stays below its length");
        self.presents[newest] = seq;
        let back = usize::try_from(self.enforced()).expect("the latency fits usize");
        let target = self.presents[(newest + RING - back) % RING];
        (target != 0).then_some(target)
    }
}

#[cfg(test)]
mod tests;
