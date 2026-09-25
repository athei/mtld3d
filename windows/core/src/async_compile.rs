//! Pure parts of building shader libraries and render pipelines off the encoder thread.
//!
//! The encoder hands a library or pipeline it has never built to a pool of
//! worker threads and keeps encoding. A draw that needs a build still in
//! flight is either left out of the frame or waited for, and the choice
//! turns on whether leaving it out can lose content for good. This module
//! holds the logic that choice and the queue rest on, without the threads:
//! the job tickets, the two-lane queue, the record of what the frame
//! cleared, and the skip predicate.

use std::collections::VecDeque;

use mtld3d_shared::{MetalHandle, mtl_handle::MTLTextureKind};
use rustc_hash::FxHashSet;

/// Identity of one queued build, unique within the encoder that queued it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct JobTicket(u64);

/// Hands out the tickets of one encoder's builds.
pub struct TicketSource {
    next: u64,
}

impl Default for TicketSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TicketSource {
    #[must_use]
    pub const fn new() -> Self {
        Self { next: 1 }
    }

    /// A ticket no earlier call returned.
    pub const fn issue(&mut self) -> JobTicket {
        let ticket = JobTicket(self.next);
        self.next = self.next.wrapping_add(1);
        ticket
    }
}

/// Which lane of a [`CompileLanes`] a job waits in.
#[derive(Clone, Copy)]
pub enum Lane {
    /// A draw the encoder is waiting for needs this job.
    Urgent,
    /// Nothing waits for this job; the draws that need it are left out until it lands.
    Normal,
}

/// The jobs no worker has started yet, urgent ones first.
///
/// A job leaves the lanes exactly once: a worker pops it, or the encoder
/// steals it to run on its own thread. So a ticket that is no longer here
/// names a job that is running or has finished.
pub struct CompileLanes<J> {
    urgent: VecDeque<(JobTicket, J)>,
    normal: VecDeque<(JobTicket, J)>,
}

impl<J> Default for CompileLanes<J> {
    fn default() -> Self {
        Self::new()
    }
}

impl<J> CompileLanes<J> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            urgent: VecDeque::new(),
            normal: VecDeque::new(),
        }
    }

    /// Queue `job` behind the others of its lane.
    pub fn push(&mut self, ticket: JobTicket, job: J, lane: Lane) {
        match lane {
            Lane::Urgent => self.urgent.push_back((ticket, job)),
            Lane::Normal => self.normal.push_back((ticket, job)),
        }
    }

    /// The next job to start: the oldest urgent one, else the oldest normal one.
    pub fn pop(&mut self) -> Option<(JobTicket, J)> {
        self.urgent.pop_front().or_else(|| self.normal.pop_front())
    }

    /// Move the unstarted job `ticket` to the back of the urgent lane.
    ///
    /// Answers whether the job is now waiting in the urgent lane, which it
    /// also is when it was there already. `false` means no worker is going
    /// to start it from here: it is running or finished.
    pub fn promote(&mut self, ticket: JobTicket) -> bool {
        if self.urgent.iter().any(|(queued, _)| *queued == ticket) {
            return true;
        }
        let Some(position) = self.normal.iter().position(|(queued, _)| *queued == ticket) else {
            return false;
        };
        if let Some(entry) = self.normal.remove(position) {
            self.urgent.push_back(entry);
        }
        true
    }

    /// Take the unstarted urgent job `ticket`, so its caller can run it instead of a worker.
    pub fn steal(&mut self, ticket: JobTicket) -> Option<J> {
        let position = self
            .urgent
            .iter()
            .position(|(queued, _)| *queued == ticket)?;
        self.urgent.remove(position).map(|(_, job)| job)
    }

    /// Take the oldest unstarted urgent job, whichever it is.
    pub fn steal_urgent(&mut self) -> Option<(JobTicket, J)> {
        self.urgent.pop_front()
    }

    /// How many jobs wait in both lanes together.
    #[must_use]
    pub fn len(&self) -> usize {
        self.urgent.len() + self.normal.len()
    }

    /// Whether no job waits in either lane.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.urgent.is_empty() && self.normal.is_empty()
    }
}

bitflags::bitflags! {
    /// What the frame so far says about one attachment a draw writes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct TargetFlags: u8 {
        /// The attachment is the back buffer, or the multisampled companion that resolves into it.
        const BACK_BUFFER = 1 << 0;
        /// A `Clear` covering the whole attachment ran earlier in this frame.
        const CLEARED = 1 << 1;
    }
}

/// Whether a draw whose library or pipeline is still building may be left out of this frame.
///
/// Leaving a draw out is safe when the content it would have written is
/// rewritten from scratch each frame, so the frame after the build lands
/// shows it: the back buffer, or a target the frame cleared before drawing
/// into it. Anything else may be a target the application draws into once
/// and reads for the rest of its life, and the draw has to wait for its
/// build instead. `color_written` says whether the draw writes a colour
/// target; one that writes none, a depth-only pass, is judged by its depth
/// attachment alone.
#[must_use]
pub fn may_skip_draw(color_written: bool, color: &[TargetFlags], depth: TargetFlags) -> bool {
    if color_written {
        color
            .iter()
            .all(|target| target.intersects(TargetFlags::BACK_BUFFER | TargetFlags::CLEARED))
    } else {
        depth.contains(TargetFlags::CLEARED)
    }
}

/// The attachments a whole-target `Clear` has reached in the current frame.
///
/// Keyed by the texture's identity handle and the subresource (slice in the
/// low half, level in the high half for colour, the level for depth), so a
/// clear of one face or level says nothing about another.
#[derive(Default)]
pub struct ClearedTargets {
    cleared: FxHashSet<(MetalHandle<MTLTextureKind>, u32)>,
}

impl ClearedTargets {
    /// Remember that `texture` at `subresource` was cleared; a null texture is ignored.
    pub fn record(&mut self, texture: MetalHandle<MTLTextureKind>, subresource: u32) {
        if !texture.is_null() {
            self.cleared.insert((texture, subresource));
        }
    }

    /// Whether `texture` at `subresource` was cleared in this frame.
    #[must_use]
    pub fn contains(&self, texture: MetalHandle<MTLTextureKind>, subresource: u32) -> bool {
        !texture.is_null() && self.cleared.contains(&(texture, subresource))
    }

    /// Forget every clear, at the start of a frame.
    pub fn reset(&mut self) {
        self.cleared.clear();
    }
}

/// What resolving a library or pipeline for a draw found.
pub enum Resolution<H> {
    /// Built, with these handles.
    Ready(H),
    /// Queued or building under this ticket.
    Pending(JobTicket),
    /// The build failed; the draw is dropped.
    Failed,
}

#[cfg(test)]
mod tests;
