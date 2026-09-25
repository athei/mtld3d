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

use crate::passes::PassState;

/// Presented frames a texture stays marked as feeding kept content after its last such read.
///
/// A read into kept content that happens every few frames or less (a
/// periodic impostor or environment-map refresh) must not lose its mark
/// between reads, so the mark outlasts many frames; a target that fed kept
/// content once waits for its builds rather than skip them for a long
/// while after, and forever while it keeps feeding. Ten seconds at 60 Hz.
pub const FEED_MEMORY_FRAMES: u64 = 600;

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
        /// A colour target.
        const COLOR = 1 << 0;
        /// The depth plane of a depth-stencil attachment.
        const DEPTH = 1 << 1;
        /// The stencil plane of a depth-stencil attachment, tracked apart from its depth.
        const STENCIL = 1 << 2;
    }
}

/// Whether a draw whose library or pipeline is still building may be left out of this frame.
///
/// Leaving a draw out is safe only when every attachment its result lands
/// in, or that later draws read it back through, is rebuilt from scratch
/// every frame and read only by work rebuilt every frame, so the frame after
/// the build lands shows the draw. Each argument answers that for one
/// attachment plane: `color` for every colour target the pass attaches (a
/// draw that writes only depth still shapes what a later depth-tested draw
/// writes into them), `depth` and `stencil` for the planes the draw tests or
/// writes, `None` for a plane it leaves alone. A target cleared once and
/// drawn once, at load, is exactly the one a skip would lose for good, so a
/// clear in this frame alone does not qualify; see
/// [`ClearHistory::regenerated`] and [`ClearHistory::feeds_persistent`].
#[must_use]
pub fn may_skip_draw(color: &[bool], depth: Option<bool>, stencil: Option<bool>) -> bool {
    color.iter().all(|&rebuilt| rebuilt)
        && depth.is_none_or(|rebuilt| rebuilt)
        && stencil.is_none_or(|rebuilt| rebuilt)
}

/// Per texture, the recent frames that cleared its planes, and whether it feeds kept content.
///
/// Keyed by the texture's identity handle. Each texture keeps, per subresource
/// (slice in the low half, level in the high half for colour, the level for
/// depth and stencil) and plane, the index of the last frame a whole clear
/// reached it and of the frame before that, which is all "cleared this frame
/// and the one before" needs; a clear of one face, level or plane says nothing
/// about another. It also keeps the last frame its content was read into
/// something kept (a copy out of it into a kept target, a pass into a kept
/// target sampling it), which holds for [`FEED_MEMORY_FRAMES`] after it. A
/// texture neither cleared in the current or the previous frame nor read that
/// way within that span is dropped when the next frame begins, so the map holds
/// only what was touched recently. Texture handles are addresses Metal hands
/// out again, so a texture that is destroyed is forgotten ([`Self::forget`])
/// before its address can name another.
pub struct ClearHistory {
    /// Index of the current presented frame; starts at 1, so 0 means "never".
    frame: u64,
    textures: FxHashMap<MetalHandle<MTLTextureKind>, TextureHistory>,
}

#[derive(Default)]
struct TextureHistory {
    /// `(subresource, plane bit, record)` per plane a recent clear reached.
    clears: Vec<(u32, u8, ClearRecord)>,
    /// Last frame the texture's content was read into kept content; 0 for never.
    fed_kept: u64,
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
            textures: FxHashMap::default(),
        }
    }

    /// Whether any texture has a recent clear or read on record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }

    /// Start the next presented frame, forgetting textures no recent frame cleared or read.
    ///
    /// A mid-frame flush is not a new frame: the application's frame goes
    /// on, and so do its clears.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.textures.retain(|_, history| {
            history
                .clears
                .retain(|(_, _, record)| record.last + 1 >= frame);
            !history.clears.is_empty() || history.fed_within(frame)
        });
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
        let frame = self.frame;
        let history = self.textures.entry(texture).or_default();
        for plane in planes.iter() {
            let bit = plane.bits();
            let position = history
                .clears
                .iter()
                .position(|(sub, plane, _)| *sub == subresource && *plane == bit);
            let index = position.unwrap_or_else(|| {
                history
                    .clears
                    .push((subresource, bit, ClearRecord { last: 0, before: 0 }));
                history.clears.len() - 1
            });
            let record = &mut history.clears[index].2;
            if record.last != frame {
                record.before = record.last;
                record.last = frame;
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
        let Some(history) = self.textures.get(&texture) else {
            return false;
        };
        history
            .clears
            .iter()
            .find(|(sub, bit, _)| *sub == subresource && *bit == plane.bits())
            .is_some_and(|(_, _, record)| {
                record.last == self.frame && record.before != 0 && record.before + 1 == self.frame
            })
    }

    /// Whether `texture` has a recent clear on record, so a draw into it could ever be skipped.
    #[must_use]
    pub fn tracks(&self, texture: MetalHandle<MTLTextureKind>) -> bool {
        self.textures
            .get(&texture)
            .is_some_and(|history| !history.clears.is_empty())
    }

    /// Remember that `texture`'s content was just read into something kept.
    ///
    /// A `StretchRect` out of it into a target that is not rebuilt every
    /// frame, or a pass into such a target sampling it. A draw left out of
    /// `texture` would then be baked into that kept content, so none is for
    /// the next [`FEED_MEMORY_FRAMES`]. Answers whether the texture was not
    /// marked before, so a caller propagating marks knows when to stop. A
    /// null texture is ignored.
    pub fn mark_feeds_persistent(&mut self, texture: MetalHandle<MTLTextureKind>) -> bool {
        if texture.is_null() {
            return false;
        }
        let frame = self.frame;
        let history = self.textures.entry(texture).or_default();
        let newly = !history.fed_within(frame);
        history.fed_kept = frame;
        newly
    }

    /// Whether `texture`'s content was read into kept content in the last [`FEED_MEMORY_FRAMES`].
    #[must_use]
    pub fn feeds_persistent(&self, texture: MetalHandle<MTLTextureKind>) -> bool {
        self.textures
            .get(&texture)
            .is_some_and(|history| history.fed_within(self.frame))
    }

    /// Forget everything about `texture`, which is being destroyed.
    ///
    /// Its address can name the next texture Metal creates, which must not
    /// inherit this one's clears.
    pub fn forget(&mut self, texture: MetalHandle<MTLTextureKind>) {
        self.textures.remove(&texture);
    }

    /// Forget every texture, at a device `Reset`.
    ///
    /// A `Reset` recreates the implicit surfaces and ends the application's
    /// frame without a `Present`, so no clear before it vouches for a frame
    /// after it.
    pub fn clear(&mut self) {
        self.textures.clear();
    }
}

impl TextureHistory {
    /// Whether a read into kept content happened within [`FEED_MEMORY_FRAMES`] of `frame`.
    const fn fed_within(&self, frame: u64) -> bool {
        self.fed_kept != 0 && self.fed_kept + FEED_MEMORY_FRAMES >= frame
    }
}

/// Mark every recently cleared texture a kept pass of this submission sampled.
///
/// `passes` holds the submission's passes and the texture binds recorded
/// with them ([`PassState::pass_reads`]), judged before any pass rule
/// removes or merges a pass. A pass is kept when a colour target it
/// attaches or its depth plane is not rebuilt every frame, which includes a
/// target already marked as feeding kept content; each texture it sampled
/// that has a recent clear (no other can ever have a draw left out of it)
/// is marked. Marking a texture can make a pass that writes into it kept,
/// so the walk repeats until nothing new is marked, and a chain of scratch
/// targets that sample each other into a kept one is marked whole in the
/// submission that reads it. A link of the chain made by a `StretchRect` is
/// judged when the copy runs, against what is marked by then, and a link
/// across a mid-frame flush is judged in the later submission, so each such
/// link can leave one more frame unprotected.
pub fn mark_kept_reads(passes: &PassState, history: &mut ClearHistory) {
    let reads = passes.pass_reads();
    if reads.is_empty() || history.is_empty() {
        return;
    }
    // Each round that marks something marks at least one more texture, so
    // the reads bound the rounds.
    for _ in 0..=reads.len() {
        let mut marked = false;
        let mut verdict: Option<(usize, bool)> = None;
        for &(pass, texture) in reads {
            if !history.tracks(texture) {
                continue;
            }
            let kept = match verdict {
                Some((judged, kept)) if judged == pass => kept,
                _ => {
                    let kept = pass_is_kept(passes, history, pass);
                    verdict = Some((pass, kept));
                    kept
                }
            };
            if kept {
                marked |= history.mark_feeds_persistent(texture);
            }
        }
        if !marked {
            break;
        }
    }
}

/// Whether recorded pass `index` writes into a target that is not rebuilt every frame.
///
/// A pass that cannot be found is kept: answering rebuilt for it would let
/// what it read lose its draws.
fn pass_is_kept(passes: &PassState, history: &ClearHistory, index: usize) -> bool {
    let Some(pass) = passes.pass_of_read(index) else {
        return true;
    };
    let color_rebuilt = |texture: MetalHandle<MTLTextureKind>, subresource: u32| {
        (passes.is_discarded_back_buffer(texture)
            || history.regenerated(texture, subresource, ClearPlanes::COLOR))
            && !history.feeds_persistent(texture)
    };
    let rt0 = pass.color_texture();
    if !rt0.is_null() && !color_rebuilt(rt0, pass.color_slice() | (pass.color_level() << 16)) {
        return true;
    }
    let extra_kept = pass
        .extra_color()
        .iter()
        .filter(|attachment| attachment.is_bound())
        .any(|attachment| {
            !color_rebuilt(
                attachment.texture(),
                attachment.slice() | (attachment.level() << 16),
            )
        });
    let depth = pass.depth_texture();
    let depth_rebuilt = depth.is_null()
        || (history.regenerated(depth, pass.depth_level(), ClearPlanes::DEPTH)
            && !history.feeds_persistent(depth));
    extra_kept || !depth_rebuilt
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
