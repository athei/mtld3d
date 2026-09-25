use mtld3d_shared::{MetalHandle, mtl_handle::MTLTextureKind};

use super::{ClearHistory, ClearPlanes, CompileLanes, TicketSource, may_skip_draw};

fn texture(raw: u64) -> MetalHandle<MTLTextureKind> {
    // SAFETY: the value is an opaque test identity; nothing dereferences it.
    unsafe { MetalHandle::new(raw) }
}

/// Urgent jobs start before normal ones, and each lane keeps its order.
#[test]
fn urgent_jobs_pop_first_in_order() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let first = tickets.issue();
    let second = tickets.issue();
    let urgent = tickets.issue();
    lanes.push_normal(first, "first");
    lanes.push_normal(second, "second");
    lanes.push_urgent(urgent, "urgent");
    assert_eq!(lanes.len(), 3);
    assert_eq!(lanes.pop(), Some((urgent, "urgent")));
    assert_eq!(lanes.pop(), Some((first, "first")));
    assert_eq!(lanes.pop(), Some((second, "second")));
    assert_eq!(lanes.pop(), None);
    assert!(lanes.is_empty());
}

/// Promoting a queued normal job moves it ahead of every other normal job.
#[test]
fn promote_moves_a_normal_job_to_the_urgent_lane() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let first = tickets.issue();
    let waited = tickets.issue();
    lanes.push_normal(first, 1);
    lanes.push_normal(waited, 2);
    assert!(lanes.promote(waited));
    assert!(lanes.promote(waited), "promoting twice keeps it urgent");
    assert_eq!(lanes.len(), 2, "promotion moves, it does not copy");
    assert_eq!(lanes.pop(), Some((waited, 2)));
    assert_eq!(lanes.pop(), Some((first, 1)));
}

/// A job a worker already took cannot be promoted or stolen.
#[test]
fn a_started_job_is_neither_promoted_nor_stolen() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let started = tickets.issue();
    lanes.push_normal(started, ());
    assert_eq!(lanes.pop(), Some((started, ())));
    assert!(!lanes.promote(started));
    assert_eq!(lanes.steal(started), None);
}

/// Stealing takes an unstarted urgent job by ticket and leaves the rest queued.
#[test]
fn steal_takes_only_the_named_urgent_job() {
    let mut tickets = TicketSource::new();
    let mut lanes = CompileLanes::new();
    let normal = tickets.issue();
    let urgent = tickets.issue();
    let other = tickets.issue();
    lanes.push_normal(normal, "normal");
    lanes.push_urgent(urgent, "urgent");
    lanes.push_urgent(other, "other");
    assert_eq!(lanes.steal(normal), None, "a normal job is promoted first");
    assert_eq!(lanes.steal(other), Some("other"));
    assert_eq!(lanes.steal(other), None, "a job leaves the lanes once");
    assert_eq!(lanes.steal_urgent(), Some((urgent, "urgent")));
    assert_eq!(lanes.steal_urgent(), None);
    assert_eq!(lanes.pop(), Some((normal, "normal")));
}

/// Every attached colour target and every plane the draw uses has to be regenerated.
#[test]
fn a_draw_skips_only_when_everything_it_depends_on_is_regenerated() {
    assert!(may_skip_draw(&[true], None, None));
    assert!(!may_skip_draw(&[false], None, None));
    assert!(
        !may_skip_draw(&[true, false], None, None),
        "every attached colour target counts"
    );
    assert!(may_skip_draw(&[true], Some(true), Some(true)));
    assert!(
        !may_skip_draw(&[true], Some(false), None),
        "a depth test against a kept attachment waits"
    );
    assert!(
        !may_skip_draw(&[true], Some(true), Some(false)),
        "a stencil test against a kept plane waits"
    );
    assert!(
        !may_skip_draw(&[false], Some(true), None),
        "a depth-only draw into a regenerated depth still feeds a kept colour target"
    );
}

/// One clear is not enough; a clear in each of two consecutive frames is.
#[test]
fn an_attachment_is_regenerated_after_two_consecutive_cleared_frames() {
    let mut history = ClearHistory::new();
    let rt = texture(0x100);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "a first clear may open a one-off render"
    );
    history.begin_frame();
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.record(rt, 0, ClearPlanes::COLOR);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "not before this frame's own clear"
    );
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
}

/// A frame without the clear breaks the streak and the entry is dropped.
#[test]
fn a_skipped_clear_restarts_the_streak() {
    let mut history = ClearHistory::new();
    let rt = texture(0x100);
    history.record(rt, 0, ClearPlanes::COLOR);
    history.begin_frame();
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    history.begin_frame();
    history.begin_frame();
    assert!(history.is_empty(), "stale attachments are forgotten");
}

/// Planes and subresources are tracked apart, and a null texture never counts.
#[test]
fn planes_and_subresources_are_separate() {
    let mut history = ClearHistory::new();
    let ds = texture(0x200);
    for _ in 0..2 {
        history.begin_frame();
        history.record(ds, 0, ClearPlanes::DEPTH);
        history.record(ds, 1, ClearPlanes::DEPTH | ClearPlanes::STENCIL);
        history.record(MetalHandle::NULL, 0, ClearPlanes::COLOR);
    }
    assert!(history.regenerated(ds, 0, ClearPlanes::DEPTH));
    assert!(
        !history.regenerated(ds, 0, ClearPlanes::STENCIL),
        "a depth-only clear leaves stencil kept"
    );
    assert!(history.regenerated(ds, 1, ClearPlanes::STENCIL));
    assert!(!history.regenerated(ds, 2, ClearPlanes::DEPTH));
    assert!(!history.regenerated(MetalHandle::NULL, 0, ClearPlanes::COLOR));
}

/// A destroyed texture's address names a new one, which inherits none of its clears.
#[test]
fn a_forgotten_texture_starts_over_at_its_address() {
    let mut history = ClearHistory::new();
    let rt = texture(0x300);
    history.record(rt, 0, ClearPlanes::COLOR);
    history.begin_frame();
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(history.regenerated(rt, 0, ClearPlanes::COLOR));
    let before = history.generation();
    history.forget(rt);
    assert_ne!(
        history.generation(),
        before,
        "cached answers are invalidated"
    );
    history.record(rt, 0, ClearPlanes::COLOR);
    assert!(
        !history.regenerated(rt, 0, ClearPlanes::COLOR),
        "a one-off target at a reused address is not rebuilt every frame"
    );
    assert!(history.tracks(rt));
}

/// A `Reset` forgets every texture at once.
#[test]
fn clearing_forgets_every_texture() {
    let mut history = ClearHistory::new();
    let rt = texture(0x300);
    let ds = texture(0x400);
    for _ in 0..2 {
        history.begin_frame();
        history.record(rt, 0, ClearPlanes::COLOR);
        history.record(ds, 0, ClearPlanes::DEPTH);
    }
    history.mark_feeds_persistent(rt);
    history.clear();
    assert!(history.is_empty());
    assert!(!history.regenerated(rt, 0, ClearPlanes::COLOR));
    assert!(!history.regenerated(ds, 0, ClearPlanes::DEPTH));
    assert!(!history.feeds_persistent(rt));
}

/// A read into kept content holds for the frame it happened in and the next, then lapses.
#[test]
fn feeding_kept_content_lasts_this_frame_and_the_next() {
    let mut history = ClearHistory::new();
    let scratch = texture(0x500);
    assert!(!history.feeds_persistent(scratch));
    assert!(!history.tracks(scratch));
    history.mark_feeds_persistent(scratch);
    assert!(history.feeds_persistent(scratch));
    assert!(!history.tracks(scratch), "a read is not a clear");
    history.begin_frame();
    assert!(history.feeds_persistent(scratch), "still the next frame");
    history.begin_frame();
    assert!(!history.feeds_persistent(scratch));
    assert!(history.is_empty(), "and the texture is forgotten");
    history.mark_feeds_persistent(MetalHandle::NULL);
    assert!(history.is_empty());
}

/// Every change an answer depends on moves the generation; a repeat does not.
#[test]
fn the_generation_moves_with_every_answer_change() {
    let mut history = ClearHistory::new();
    let rt = texture(0x600);
    let start = history.generation();
    history.record(rt, 0, ClearPlanes::COLOR);
    let recorded = history.generation();
    assert_ne!(recorded, start);
    history.record(rt, 0, ClearPlanes::COLOR);
    assert_eq!(
        history.generation(),
        recorded,
        "a second clear in one frame changes nothing"
    );
    history.mark_feeds_persistent(rt);
    let marked = history.generation();
    assert_ne!(marked, recorded);
    history.mark_feeds_persistent(rt);
    assert_eq!(history.generation(), marked);
    history.begin_frame();
    assert_ne!(history.generation(), marked);
}
