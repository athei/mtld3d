//! Read guards on the PE-heap staging the frame's blits and upload passes read.
//!
//! A guard is held from encode time until the GPU retires the command
//! buffer that reads the staging. The queued guards' bytes feed the
//! summary's `blit source staging` gauge, which this type keeps in step:
//! the queues are private, so a guard leaves only through a path that takes
//! its bytes off the gauge.

use std::{collections::VecDeque, sync::Arc};

use mtld3d_core::{page_box::PageBoxRead, perf::EncoderPerfState};

/// One staging read keyed by the `submit_seq` of the frame that reads it.
struct PendingBlitRead {
    submit_seq: u64,
    read: PageBoxRead,
}

impl PendingBlitRead {
    /// Byte length of the retained staging, what the queue added to the gauge for it.
    fn byte_len(&self) -> usize {
        self.read.backing().len()
    }
}

/// The frame's staging reads and the queued reads of submitted frames.
#[derive(Default)]
pub struct BlitRetention {
    /// Reads of the frame being encoded; not on the gauge until queued.
    current: Vec<PageBoxRead>,
    /// Submitted frames' reads in `submit_seq` order, each one counted on the gauge.
    pending: VecDeque<PendingBlitRead>,
}

impl BlitRetention {
    /// Hold a read of staging a blit or upload pass of this frame reads.
    pub fn hold(&mut self, read: PageBoxRead) {
        self.current.push(read);
    }

    /// Queue this frame's reads under `submit_seq`, adding each one's bytes to the gauge.
    ///
    /// Called before submission, so the reads outlive the blit encode and
    /// commit whichever thread runs them.
    pub fn queue(&mut self, perf: &mut EncoderPerfState, submit_seq: u64) {
        for read in self.current.drain(..) {
            perf.bump_tex_staging_retained_add(read.backing().len());
            self.pending.push_back(PendingBlitRead { submit_seq, read });
        }
    }

    /// Release the queued reads whose sequence `coherent` has reached, taking their bytes off.
    ///
    /// A recovery job may still read these pages again after the emitted read
    /// retires, so its guard is independent of this queue.
    pub fn reclaim(&mut self, perf: &mut EncoderPerfState, coherent: u64) {
        while let Some(front) = self.pending.front() {
            if front.submit_seq > coherent {
                break;
            }
            let entry = self.pending.pop_front().expect("checked front");
            perf.bump_tex_staging_retained_sub(entry.byte_len());
            debug_assert!(
                Arc::strong_count(entry.read.backing()) >= 1,
                "pending blit Arc already orphaned"
            );
        }
    }

    /// Release every read, queued or not, taking the queued bytes off the gauge.
    ///
    /// For the reset and shutdown cleanups, once the GPU is idle and every
    /// wrapper around the staging is destroyed. The current frame's reads
    /// were never added, so they leave the gauge alone.
    pub fn release_all(&mut self, perf: &mut EncoderPerfState) {
        for entry in self.pending.drain(..) {
            perf.bump_tex_staging_retained_sub(entry.byte_len());
        }
        self.current.clear();
    }

    /// Queued reads, the summary's retention depth.
    pub fn queued(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests;
