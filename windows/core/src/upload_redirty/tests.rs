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

fn guest_lease(
    queue: &std::sync::Arc<RedirtyQueue>,
    sub: RedirtySubresource,
    generation: u32,
) -> super::GuestRedirtyLease {
    super::GuestRedirtyLease::new(
        std::sync::Arc::clone(queue),
        entry(sub, DirtyRect::full(8, 4)),
        releasing(sub, generation),
    )
}

#[test]
fn guest_feedback_preserves_retry_cap_and_success_resets_it() {
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(0);
    for attempt in 0..=MAX_REDIRTY_ATTEMPTS {
        let mut lease = guest_lease(&queue, sub, attempt);
        // SAFETY: this one-job lease retains both cells through the proxy's final drop.
        let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
        assert!(native.decline(entry(sub, DirtyRect::full(8, 4))));
        assert!(!lease.maintain());
        drop(native);
        assert!(lease.maintain());
        assert_eq!(
            queue.take_pending().len(),
            usize::from(attempt < MAX_REDIRTY_ATTEMPTS)
        );
        assert!(lease.maintain());
        assert!(queue.take_pending().is_empty());
    }
    let mut lease = guest_lease(&queue, sub, 9);
    // SAFETY: the unique publication remains retained through native completion.
    let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
    native.note_emitted(releasing(sub, 9));
    drop(native);
    assert!(lease.maintain());
    let release = queue.take_released();
    assert_eq!(release.len(), 1);
    assert_eq!(release[0].generation, 9);
    assert!(queue.decline(entry(sub, DirtyRect::full(8, 4))));
}

#[test]
fn guest_decline_cancels_prior_staging_release() {
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(3);
    queue.note_emitted(releasing(sub, 1));
    let mut lease = guest_lease(&queue, sub, 2);
    // SAFETY: the lease remains retained and is adopted exactly once.
    let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
    native.decline(entry(sub, DirtyRect::full(8, 4)));
    drop(native);
    assert!(lease.maintain());
    assert!(queue.take_released().is_empty());
    assert_eq!(queue.take_pending().len(), 1);
}

#[test]
fn guest_cancellation_and_unanswered_native_drop_restore_dirty_state() {
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(0);
    let mut canceled = guest_lease(&queue, sub, 1);
    // SAFETY: this publication never reached native code.
    unsafe { canceled.cancel_unadopted() };
    assert_eq!(queue.take_pending().len(), 1);
    let mut unanswered = guest_lease(&queue, sub, 2);
    // SAFETY: this unique lease remains retained until the proxy is destroyed.
    let native = unsafe { unanswered.descriptor().adopt() }.expect("native feedback");
    drop(native);
    assert!(unanswered.maintain());
    assert_eq!(queue.take_pending().len(), 1);
}

#[test]
fn guest_feedback_orders_events_across_retained_frames() {
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(1);
    assert!(queue.decline(entry(sub, DirtyRect::full(8, 4))));
    let _ = queue.take_pending();
    let mut earlier = guest_lease(&queue, sub, 1);
    let mut later = guest_lease(&queue, sub, 2);
    // SAFETY: both distinct leases retain their mailboxes and original shared queue.
    let first = unsafe { earlier.descriptor().adopt() }.expect("first proxy");
    // SAFETY: later is a distinct retained one-job publication.
    let second = unsafe { later.descriptor().adopt() }.expect("second proxy");
    first.note_emitted(releasing(sub, 1));
    second.decline(entry(sub, DirtyRect::full(8, 4)));
    drop(second);
    assert!(later.maintain());
    assert!(queue.take_pending().is_empty());
    assert!(!earlier.maintain());
    assert_eq!(queue.take_pending().len(), 1);
    assert!(queue.take_released().is_empty());
    drop(first);
    assert!(earlier.maintain());
    for _ in 1..MAX_REDIRTY_ATTEMPTS {
        assert!(queue.decline(entry(sub, DirtyRect::full(8, 4))));
    }
    assert!(!queue.decline(entry(sub, DirtyRect::full(8, 4))));
}

#[test]
fn replay_failure_is_a_second_ordered_event_for_the_same_upload() {
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(1);
    let mut lease = guest_lease(&queue, sub, 4);
    // SAFETY: retained lease pins both feedback cells and the original queue counter.
    let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
    native.note_emitted(releasing(sub, 4));
    assert!(!lease.maintain());
    assert_eq!(queue.take_released().len(), 1);
    native.decline(entry(sub, DirtyRect::full(8, 4)));
    assert!(!lease.maintain());
    assert_eq!(queue.take_pending().len(), 1);
    drop(native);
    assert!(lease.maintain());
    assert!(queue.take_pending().is_empty());
}

#[test]
fn pooled_feedback_notifies_early_events_and_recycles_after_final_consumption() {
    use crate::guest_completions::{CompletionDrain, CompletionPool};
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(1);
    let mut lease = super::GuestRedirtyLease::new_pooled(
        std::sync::Arc::clone(&queue),
        entry(sub, DirtyRect::full(8, 4)),
        releasing(sub, 2),
        &pool,
    );
    let tokens = lease.tokens();
    // SAFETY: lease retains the queue counter and both pooled notification slots.
    let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
    native.note_emitted(releasing(sub, 2));
    let mut notifications = Vec::new();
    pool.drain(&mut cursor, 1, |event| notifications.push(event));
    assert_eq!(notifications, vec![tokens[0].expect("event token") * 2]);
    assert!(!lease.maintain());
    assert_eq!(queue.take_released().len(), 1);
    drop(native);
    assert!(!lease.maintain());
    pool.drain(&mut cursor, 8, |event| notifications.push(event));
    assert_eq!(notifications.len(), 3);
    assert!(lease.maintain());
    assert!(queue.take_pending().is_empty());
    for slot in lease.into_slots().into_iter().flatten() {
        pool.recycle(slot);
    }
}

#[test]
fn pooled_cancellation_restores_dirty_state_and_consumes_absent_event() {
    use crate::guest_completions::{CompletionDrain, CompletionPool};
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(1);
    let mut lease = super::GuestRedirtyLease::new_pooled(
        std::sync::Arc::clone(&queue),
        entry(sub, DirtyRect::full(8, 4)),
        releasing(sub, 2),
        &pool,
    );
    // SAFETY: the publication was never adopted.
    unsafe { lease.cancel_unadopted() };
    assert_eq!(queue.take_pending().len(), 1);
    assert!(!lease.maintain());
    let mut notifications = 0;
    pool.drain(&mut cursor, 8, |_| notifications += 1);
    assert_eq!(notifications, 3);
    assert!(lease.maintain());
    for slot in lease.into_slots().into_iter().flatten() {
        pool.recycle(slot);
    }
}

#[test]
fn pooled_feedback_record_reuses_backing_after_all_events_are_consumed() {
    use crate::guest_completions::{CompletionDrain, CompletionPool};
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let queue = std::sync::Arc::new(RedirtyQueue::new());
    let sub = subresource(1);
    let mut last_address = None;
    for generation in 0..4 {
        let mut lease = super::GuestRedirtyLease::new_pooled(
            std::sync::Arc::clone(&queue),
            entry(sub, DirtyRect::full(8, 4)),
            releasing(sub, generation),
            &pool,
        );
        let address = lease.descriptor().wire_fields()[0];
        if let Some(previous) = last_address {
            assert_eq!(address, previous);
        }
        last_address = Some(address);
        // SAFETY: each loop uses a fresh retained publication whose predecessor was consumed.
        let native = unsafe { lease.descriptor().adopt() }.expect("native feedback");
        native.note_emitted(releasing(sub, generation));
        drop(native);
        pool.drain(&mut cursor, 8, |_| {});
        assert!(lease.maintain());
        assert_eq!(queue.take_released()[0].generation, generation);
        for slot in lease.into_slots().into_iter().flatten() {
            pool.recycle(slot);
        }
    }
}
