use super::{Arc, AtomicU32, AtomicU64, Ordering, ReplyBool, ReplyU64};

#[test]
fn native_reply_borrows_without_changing_guest_reference_count() {
    let cell = Arc::new(AtomicU64::new(0));
    let retained = ReplyU64::from(Arc::clone(&cell));
    // SAFETY: retained and cell keep the aligned atomic alive until the borrowed reply drops.
    let native = unsafe { ReplyU64::from_guest(retained.address()) };
    assert_eq!(Arc::strong_count(&cell), 2);
    native.store(0x1234_5678_9abc_def0, Ordering::Release);
    assert_eq!(cell.load(Ordering::Acquire), 0x1234_5678_9abc_def0);
    drop(native);
    assert_eq!(Arc::strong_count(&cell), 2);
    drop(retained);
    assert_eq!(Arc::strong_count(&cell), 1);
}

#[test]
fn success_reply_uses_fixed_width_publication() {
    let cell = Arc::new(AtomicU32::new(0));
    let retained = ReplyBool::from(Arc::clone(&cell));
    // SAFETY: retained and cell pin the aligned atomic through all borrowed accesses below.
    let native = unsafe { ReplyBool::from_guest(retained.address()) };
    native.store(true, Ordering::Release);
    assert_eq!(cell.load(Ordering::Acquire), 1);
    native.store(false, Ordering::Release);
    assert_eq!(cell.load(Ordering::Acquire), 0);
}
