use super::{CompletionDrain, CompletionPool};

#[test]
fn unused_acquisition_reuses_queued_slot_without_publishing_an_event() {
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let mut token = None;
    for needs_acquire in [false, true, false] {
        let slot = pool.allocate(needs_acquire);
        assert_eq!(*token.get_or_insert_with(|| slot.token()), slot.token());
        assert_eq!(slot.acquired().is_complete(), !needs_acquire);
        assert!(!slot.completion().is_complete());

        slot.acquired().publish();
        slot.acquired().publish();
        let mut events = Vec::new();
        pool.drain(&mut cursor, 8, |event| events.push(event));
        if needs_acquire {
            assert_eq!(events, [slot.token() * 2]);
        } else {
            assert!(events.is_empty(), "unused acquisition must remain unqueued");
        }
        assert!(slot.acquired().is_complete());
        assert!(
            !slot.completion().is_complete(),
            "final owner still retains the slot"
        );
        slot.completion().publish();
        assert!(
            !slot.completion().is_complete(),
            "publication alone cannot recycle"
        );
        events.clear();
        pool.drain(&mut cursor, 8, |event| events.push(event));
        assert_eq!(events, [slot.token() * 2 + 1]);
        assert!(slot.completion().is_complete());
        pool.recycle(slot);
    }
}

#[test]
fn queued_cells_require_consumption_and_ignore_duplicate_publish() {
    let pool = CompletionPool::new();
    let slot = pool.allocate(true);
    slot.acquired().publish();
    slot.completion().publish();
    slot.completion().publish();
    assert!(!slot.completion().is_complete());
    let mut cursor = CompletionDrain::default();
    let mut events = Vec::new();
    pool.drain(&mut cursor, 1, |event| events.push(event));
    assert_eq!(events, [slot.token() * 2 + 1]);
    assert!(!slot.acquired().is_complete());
    pool.drain(&mut cursor, 8, |event| events.push(event));
    assert_eq!(events.len(), 2);
    assert!(slot.acquired().is_complete());
    let token = slot.token();
    pool.recycle(slot);
    let next = pool.allocate(false);
    assert_eq!(next.token(), token);
    assert!(!next.completion().is_complete());
    next.completion().publish();
    pool.drain(&mut cursor, 8, |_| {});
    pool.recycle(next);
}

#[test]
fn concurrent_publish_and_bounded_draining_reuses_blocks() {
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    for _ in 0..8 {
        let slots: Vec<_> = (0..300).map(|_| pool.allocate(true)).collect();
        let mut seen = vec![0u8; 600];
        std::thread::scope(|scope| {
            for group in slots.chunks(50) {
                scope.spawn(move || {
                    for slot in group {
                        slot.acquired().publish();
                        slot.completion().publish();
                    }
                });
            }
            let mut count = 0;
            while count != 600 {
                pool.drain(&mut cursor, 17, |event| {
                    seen[usize::try_from(event).expect("test token fits")] += 1;
                    count += 1;
                });
                std::thread::yield_now();
            }
        });
        assert!(seen.iter().all(|count| *count == 1));
        for slot in slots {
            pool.recycle(slot);
        }
    }
}

#[test]
fn pooled_read_handoff_retains_owner_through_cached_native_reference() {
    use std::sync::Arc;

    use crate::{
        guest_pages::GuestPageLease,
        page_box::{PageBox, PageBoxRead},
    };
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let owner = Arc::new(PageBox::new_zeroed(8));
    let mut lease = GuestPageLease::for_read_pooled(PageBoxRead::new(Arc::clone(&owner)), &pool);
    // SAFETY: lease keeps its original reader, backing and cells alive through adoption.
    let native = unsafe { lease.descriptor().adopt_read() }.expect("native read");
    assert!(!lease.maintain());
    assert!(owner.has_readers());
    pool.drain(&mut cursor, 8, |_| {});
    assert!(!lease.maintain());
    let cached = Arc::clone(native.backing());
    drop(native);
    assert!(!owner.has_readers());
    assert!(!lease.maintain());
    drop(cached);
    assert!(
        !lease.maintain(),
        "notification must be consumed before owner release"
    );
    pool.drain(&mut cursor, 8, |_| {});
    assert!(lease.maintain());
    pool.recycle(lease.into_slot().expect("pooled"));
}

#[test]
fn rejected_pooled_read_consumes_both_events_before_recycling() {
    use std::sync::Arc;

    use crate::{
        guest_pages::GuestPageLease,
        page_box::{PageBox, PageBoxRead},
    };
    let pool = CompletionPool::new();
    let mut cursor = CompletionDrain::default();
    let owner = Arc::new(PageBox::new_zeroed(8));
    let mut lease = GuestPageLease::for_read_pooled(PageBoxRead::new(Arc::clone(&owner)), &pool);
    // SAFETY: descriptor has never been adopted and cannot reach native code.
    unsafe { lease.descriptor().cancel_unadopted() }.expect("cancel");
    pool.drain(&mut cursor, 1, |_| {});
    assert!(!lease.maintain(), "acquisition notification still pending");
    pool.drain(&mut cursor, 1, |_| {});
    assert!(lease.maintain());
    assert!(!owner.has_readers());
    pool.recycle(lease.into_slot().expect("pooled"));
}

#[test]
fn detached_cursor_retains_blocks_and_rejects_another_device() {
    let pool = CompletionPool::new();
    let slot = pool.allocate(true);
    slot.acquired().publish();
    slot.completion().publish();
    let mut cursor = CompletionDrain::default();
    pool.drain(&mut cursor, 1, |_| {});
    let weak = std::sync::Arc::downgrade(&slot.block);
    drop(slot);
    drop(pool);
    assert!(
        weak.upgrade().is_some(),
        "detached tail keeps backing alive"
    );
    let other = CompletionPool::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        other.drain(&mut cursor, 8, |_| {});
    }));
    assert!(result.is_err(), "cursor cannot switch device queues");
    drop(cursor);
    assert!(weak.upgrade().is_none());
}
