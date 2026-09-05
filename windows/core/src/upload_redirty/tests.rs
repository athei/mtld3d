use super::{EmittedUpload, MAX_REDIRTY_ATTEMPTS, RedirtyEntry, RedirtyQueue, RedirtySubresource};
use crate::{dirty_rect::DirtyRect, ids::TextureId};

fn subresource(index: u32) -> RedirtySubresource {
    RedirtySubresource {
        texture_id: TextureId::new_unique(),
        index,
    }
}

fn emitted(subresource: RedirtySubresource) -> EmittedUpload {
    EmittedUpload {
        subresource,
        level: subresource.index,
        generation: 0,
        releases_staging: false,
    }
}

fn releasing(subresource: RedirtySubresource, generation: u32) -> EmittedUpload {
    EmittedUpload {
        generation,
        releases_staging: true,
        ..emitted(subresource)
    }
}

fn entry(subresource: RedirtySubresource, rect: DirtyRect) -> RedirtyEntry {
    RedirtyEntry {
        subresource,
        face: 0,
        level: subresource.index,
        rect,
    }
}

#[test]
fn a_fresh_queue_has_nothing_to_drain() {
    let queue = RedirtyQueue::new();
    assert!(!queue.has_pending());
    assert!(queue.take_pending().is_empty());
    assert!(queue.take_released().is_empty());
}

#[test]
fn a_declined_upload_comes_back_with_its_rect() {
    let queue = RedirtyQueue::new();
    let sub = subresource(2);
    let rect = DirtyRect {
        x: 16,
        y: 8,
        w: 32,
        h: 4,
    };
    assert!(queue.decline(entry(sub, rect)));
    assert!(queue.has_pending());

    let drained = queue.take_pending();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].subresource, sub);
    assert_eq!(drained[0].level, 2);
    assert_eq!(drained[0].rect, rect);
    assert!(!queue.has_pending());
    assert!(queue.take_pending().is_empty());
}

#[test]
fn every_declined_subresource_is_reported_separately() {
    let queue = RedirtyQueue::new();
    let first = subresource(0);
    let second = subresource(1);
    assert!(queue.decline(entry(first, DirtyRect::full(8, 8))));
    assert!(queue.decline(entry(second, DirtyRect::full(4, 4))));

    let drained = queue.take_pending();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].subresource, first);
    assert_eq!(drained[1].subresource, second);
}

#[test]
fn a_subresource_that_keeps_declining_stops_being_retried() {
    let queue = RedirtyQueue::new();
    let sub = subresource(0);
    let rect = DirtyRect::full(16, 16);
    for _ in 0..MAX_REDIRTY_ATTEMPTS {
        assert!(queue.decline(entry(sub, rect)));
    }
    assert!(!queue.decline(entry(sub, rect)));

    let drained = queue.take_pending();
    assert_eq!(drained.len(), MAX_REDIRTY_ATTEMPTS as usize);
}

#[test]
fn the_budget_is_per_subresource() {
    let queue = RedirtyQueue::new();
    let spent = subresource(0);
    let fresh = subresource(1);
    let rect = DirtyRect::full(16, 16);
    for _ in 0..=MAX_REDIRTY_ATTEMPTS {
        queue.decline(entry(spent, rect));
    }
    assert!(!queue.decline(entry(spent, rect)));
    assert!(queue.decline(entry(fresh, rect)));
}

#[test]
fn an_emitted_upload_gives_the_subresource_its_budget_back() {
    let queue = RedirtyQueue::new();
    let sub = subresource(0);
    let rect = DirtyRect::full(16, 16);
    for _ in 0..MAX_REDIRTY_ATTEMPTS {
        assert!(queue.decline(entry(sub, rect)));
    }
    assert!(!queue.decline(entry(sub, rect)));

    queue.note_emitted(emitted(sub));
    assert!(queue.decline(entry(sub, rect)));
}

#[test]
fn acknowledging_an_untouched_subresource_changes_nothing() {
    let queue = RedirtyQueue::new();
    let sub = subresource(0);
    queue.note_emitted(emitted(sub));
    assert!(!queue.has_pending());
    assert!(queue.decline(entry(sub, DirtyRect::full(2, 2))));
}

#[test]
fn an_emitted_upload_that_asked_for_it_reports_its_staging_released() {
    let queue = RedirtyQueue::new();
    let sub = subresource(3);
    queue.note_emitted(releasing(sub, 7));
    assert!(queue.has_pending());

    let released = queue.take_released();
    assert_eq!(released.len(), 1);
    assert_eq!(released[0].subresource, sub);
    assert_eq!(released[0].level, 3);
    assert_eq!(released[0].generation, 7);
    assert!(queue.take_pending().is_empty());
    assert!(!queue.has_pending());
    assert!(queue.take_released().is_empty());
}

#[test]
fn an_emitted_upload_that_keeps_its_staging_releases_nothing() {
    let queue = RedirtyQueue::new();
    queue.note_emitted(emitted(subresource(0)));
    assert!(!queue.has_pending());
    assert!(queue.take_released().is_empty());
}

#[test]
fn a_decline_cancels_a_release_the_same_subresource_is_waiting_for() {
    let queue = RedirtyQueue::new();
    let sub = subresource(1);
    queue.note_emitted(releasing(sub, 1));
    assert!(queue.decline(entry(sub, DirtyRect::full(8, 8))));

    assert!(queue.take_released().is_empty());
    assert_eq!(queue.take_pending().len(), 1);
}

#[test]
fn a_decline_leaves_another_subresources_release_alone() {
    let queue = RedirtyQueue::new();
    let released = subresource(0);
    let declined = subresource(1);
    queue.note_emitted(releasing(released, 1));
    assert!(queue.decline(entry(declined, DirtyRect::full(8, 8))));

    let drained = queue.take_released();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].subresource, released);
    assert_eq!(queue.take_pending().len(), 1);
}

#[test]
fn an_emitted_upload_that_releases_its_staging_also_gives_the_budget_back() {
    let queue = RedirtyQueue::new();
    let sub = subresource(0);
    let rect = DirtyRect::full(16, 16);
    for _ in 0..MAX_REDIRTY_ATTEMPTS {
        assert!(queue.decline(entry(sub, rect)));
    }
    assert!(!queue.decline(entry(sub, rect)));

    queue.note_emitted(releasing(sub, 1));
    assert!(queue.decline(entry(sub, rect)));
}
