//! Pure parts of building shader libraries and render pipelines off the encoder thread.
//!
//! The encoder hands a library or pipeline it has never built to a pool of
//! worker threads and keeps encoding. A draw that needs a build still in
//! flight is either left out of the frame or waited for, and the choice
//! turns on whether leaving it out can lose content for good. This module
//! holds the logic that choice and the queue rest on, without the threads:
//! the job tickets, the two-lane queue, the record of which attachments
//! recent frames cleared, and the skip predicate.

use std::collections::VecDeque;

use mtld3d_shared::{MetalHandle, mtl_handle::MTLTextureKind};
use rustc_hash::FxHashMap;

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

/// The jobs no worker has started yet, urgent ones first.
///
/// The normal lane holds the jobs nothing waits for; the urgent lane the
/// ones a draw the encoder is waiting on needs. A job leaves the lanes
/// exactly once: a worker pops it, or the encoder steals it to run on its
/// own thread. So a ticket that is no longer here names a job that is
/// running or has finished.
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

    /// Queue `job` behind the others nothing waits for.
    pub fn push_normal(&mut self, ticket: JobTicket, job: J) {
        self.normal.push_back((ticket, job));
    }

    /// Queue `job` behind the other urgent ones, ahead of every normal job.
    pub fn push_urgent(&mut self, ticket: JobTicket, job: J) {
        self.urgent.push_back((ticket, job));
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
    /// The planes of one attachment a clear reached.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ClearPlanes: u8 {
        const COLOR = 1 << 0;
        const DEPTH = 1 << 1;
        const STENCIL = 1 << 2;
    }
}

/// Whether a draw whose library or pipeline is still building may be left out of this frame.
///
/// Leaving a draw out is safe only when every attachment its result lands
/// in, or that later draws read it back through, is rebuilt from scratch
/// every frame, so the frame after the build lands shows the draw. Each
/// argument answers that for one attachment plane: `color` for every colour
/// target the pass attaches (a draw that writes only depth still shapes what
/// a later depth-tested draw writes into them), `depth` and `stencil` for
/// the planes the draw tests or writes, `None` for a plane it leaves alone.
/// A target cleared once and drawn once, at load, is exactly the one a skip
/// would lose for good, so a clear in this frame alone does not qualify; see
/// [`ClearHistory::regenerated`].
#[must_use]
pub fn may_skip_draw(color: &[bool], depth: Option<bool>, stencil: Option<bool>) -> bool {
    color.iter().all(|&regenerated| regenerated)
        && depth.is_none_or(|regenerated| regenerated)
        && stencil.is_none_or(|regenerated| regenerated)
}

/// Per attachment plane, the presented frames whose whole-target `Clear` reached it.
///
/// Keyed by the texture's identity handle, the subresource (slice in the
/// low half, level in the high half for colour, the level for depth and
/// stencil) and the plane, so a clear of one face, level or plane says
/// nothing about another. Each entry keeps the index of the last frame that
/// cleared it and of the frame that cleared it before that, which is all
/// "cleared this frame and the one before" needs. An entry no clear reached
/// in the current or the previous frame is dropped when the next frame
/// begins, so the map holds only the attachments cleared recently.
pub struct ClearHistory {
    /// Index of the current presented frame; starts at 1, so 0 means "never".
    frame: u64,
    entries: FxHashMap<(MetalHandle<MTLTextureKind>, u32, u8), ClearRecord>,
}

struct ClearRecord {
    last: u64,
    before: u64,
}

impl Default for ClearHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl ClearHistory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame: 1,
            entries: FxHashMap::default(),
        }
    }

    /// Start the next presented frame, forgetting attachments no recent frame cleared.
    ///
    /// A mid-frame flush is not a new frame: the application's frame goes
    /// on, and so do its clears.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.entries.retain(|_, record| record.last + 1 >= frame);
    }

    /// Remember that `planes` of `texture` at `subresource` were cleared whole this frame.
    ///
    /// A null texture is ignored.
    pub fn record(
        &mut self,
        texture: MetalHandle<MTLTextureKind>,
        subresource: u32,
        planes: ClearPlanes,
    ) {
        if texture.is_null() {
            return;
        }
        for plane in planes.iter() {
            let record = self
                .entries
                .entry((texture, subresource, plane.bits()))
                .or_insert(ClearRecord { last: 0, before: 0 });
            if record.last != self.frame {
                record.before = record.last;
                record.last = self.frame;
            }
        }
    }

    /// Whether `plane` of `texture` at `subresource` was cleared whole in this frame and the last.
    ///
    /// Two consecutive frames are the evidence that the application rebuilds
    /// the attachment every frame: a clear in this frame alone is just as
    /// likely to open a one-off render whose draw a skip would lose.
    #[must_use]
    pub fn regenerated(
        &self,
        texture: MetalHandle<MTLTextureKind>,
        subresource: u32,
        plane: ClearPlanes,
    ) -> bool {
        if texture.is_null() {
            return false;
        }
        self.entries
            .get(&(texture, subresource, plane.bits()))
            .is_some_and(|record| {
                record.last == self.frame && record.before != 0 && record.before + 1 == self.frame
            })
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
