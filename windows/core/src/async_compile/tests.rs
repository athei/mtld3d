use mtld3d_shared::{MetalHandle, mtl_handle::MTLTextureKind};

use super::{ClearedTargets, CompileLanes, Lane, TargetFlags, TicketSource, may_skip_draw};

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
    lanes.push(first, "first", Lane::Normal);
    lanes.push(second, "second", Lane::Normal);
    lanes.push(urgent, "urgent", Lane::Urgent);
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
    lanes.push(first, 1, Lane::Normal);
    lanes.push(waited, 2, Lane::Normal);
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
    lanes.push(started, (), Lane::Normal);
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
    lanes.push(normal, "normal", Lane::Normal);
    lanes.push(urgent, "urgent", Lane::Urgent);
    lanes.push(other, "other", Lane::Urgent);
    assert_eq!(lanes.steal(normal), None, "a normal job is promoted first");
    assert_eq!(lanes.steal(other), Some("other"));
    assert_eq!(lanes.steal(other), None, "a job leaves the lanes once");
    assert_eq!(lanes.steal_urgent(), Some((urgent, "urgent")));
    assert_eq!(lanes.steal_urgent(), None);
    assert_eq!(lanes.pop(), Some((normal, "normal")));
}

/// The back buffer and a target cleared this frame may lose one frame's draw.
#[test]
fn a_color_draw_skips_only_into_rewritten_targets() {
    let back_buffer = TargetFlags::BACK_BUFFER;
    let cleared = TargetFlags::CLEARED;
    let kept = TargetFlags::empty();
    assert!(may_skip_draw(true, &[back_buffer], kept));
    assert!(may_skip_draw(true, &[cleared], kept));
    assert!(
        !may_skip_draw(true, &[kept], cleared),
        "depth does not vouch for colour"
    );
    assert!(
        !may_skip_draw(true, &[back_buffer, kept], cleared),
        "every target the draw writes has to be rewritten"
    );
    assert!(may_skip_draw(true, &[back_buffer, cleared], kept));
}

/// A draw that writes no colour is judged by its depth attachment.
#[test]
fn a_depth_only_draw_skips_only_into_a_cleared_depth_target() {
    assert!(may_skip_draw(false, &[], TargetFlags::CLEARED));
    assert!(!may_skip_draw(false, &[], TargetFlags::empty()));
    assert!(
        !may_skip_draw(false, &[TargetFlags::BACK_BUFFER], TargetFlags::empty()),
        "an unwritten back buffer does not vouch for depth"
    );
}

/// A clear is remembered per texture and subresource until the frame resets.
#[test]
fn cleared_targets_are_per_subresource_and_per_frame() {
    let mut cleared = ClearedTargets::default();
    let rt = texture(0x100);
    cleared.record(rt, 0);
    cleared.record(MetalHandle::NULL, 0);
    assert!(cleared.contains(rt, 0));
    assert!(
        !cleared.contains(rt, 1 << 16),
        "another level was not cleared"
    );
    assert!(!cleared.contains(texture(0x200), 0));
    assert!(!cleared.contains(MetalHandle::NULL, 0));
    cleared.reset();
    assert!(!cleared.contains(rt, 0));
}
