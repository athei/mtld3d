//! Class split, release rule, and the state machine a re-created backing follows.
//!
//! The gauge counters are process-wide, and `cargo test` runs this file's
//! tests as threads of one process, so a backing another test creates or drops
//! moves the counters under the one test that asserts byte deltas on them.
//! Every test that creates a backing therefore runs its body inside
//! `with_gauge`, which holds one lock for the length of the body: `backing`
//! asks for the guard, so a test cannot make one without it.

use std::sync::{Mutex, MutexGuard, PoisonError};

use mtld3d_types::{
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC,
    D3DUSAGE_SOFTWAREPROCESSING, D3DUSAGE_WRITEONLY,
};

use super::*;

/// A buffer length that pads to exactly one page.
const SMALL: u32 = 256;

/// Run `body` while holding the lock that keeps the gauge counters to the calling test.
///
/// Everything the body creates drops before it returns, so a backing's bytes
/// are charged and discharged inside the lock. The lock guards no data, only
/// the order of the tests, so a test that panicked while holding it poisons
/// nothing the next one relies on.
fn with_gauge(body: impl FnOnce(&MutexGuard<'static, ()>)) {
    /// The gauge lock every backing-creating test in this file holds.
    ///
    /// A `static` because the resource is process-wide: the three counters
    /// behind `live_backing_bytes` are, and the tests share the process.
    static GAUGE: Mutex<()> = Mutex::new(());
    let gauge = GAUGE.lock().unwrap_or_else(PoisonError::into_inner);
    body(&gauge);
}

/// A one-page backing of the class `usage` and `pool` select.
///
/// The guard argument is the proof that the caller holds the gauge lock.
fn backing(_gauge: &MutexGuard<'static, ()>, usage: u32, pool: u32) -> BufferBacking {
    BufferBacking::new(
        PageBox::new_zeroed(SMALL as usize),
        SMALL,
        classify_backing(usage, pool),
    )
}

// ── the class split and the release rule ──

#[test]
fn only_default_writeonly_statics_may_release_their_backing() {
    assert!(may_release_backing(D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT));
    // Dynamic is the zero-copy shape: the GPU reads this very memory.
    assert!(!may_release_backing(
        D3DUSAGE_WRITEONLY | D3DUSAGE_DYNAMIC,
        D3DPOOL_DEFAULT
    ));
    // Without WRITEONLY the application may read the buffer back.
    assert!(!may_release_backing(0, D3DPOOL_DEFAULT));
    // ProcessVertices reads a source buffer's bytes on the CPU.
    assert!(!may_release_backing(
        D3DUSAGE_WRITEONLY | D3DUSAGE_SOFTWAREPROCESSING,
        D3DPOOL_DEFAULT
    ));
    // The lockable pools keep a system-memory copy by definition.
    assert!(!may_release_backing(D3DUSAGE_WRITEONLY, D3DPOOL_MANAGED));
    assert!(!may_release_backing(D3DUSAGE_WRITEONLY, D3DPOOL_SYSTEMMEM));
}

#[test]
fn the_gauge_class_follows_the_release_rule() {
    assert_eq!(
        classify_backing(D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT),
        BackingClass::WriteOnlyStatic
    );
    assert_eq!(
        classify_backing(D3DUSAGE_WRITEONLY | D3DUSAGE_DYNAMIC, D3DPOOL_DEFAULT),
        BackingClass::Dynamic
    );
    assert_eq!(
        classify_backing(D3DUSAGE_DYNAMIC, D3DPOOL_MANAGED),
        BackingClass::Dynamic
    );
    assert_eq!(classify_backing(0, D3DPOOL_DEFAULT), BackingClass::Other);
    assert_eq!(
        classify_backing(D3DUSAGE_WRITEONLY, D3DPOOL_SYSTEMMEM),
        BackingClass::Other
    );
    assert_eq!(
        classify_backing(
            D3DUSAGE_WRITEONLY | D3DUSAGE_SOFTWAREPROCESSING,
            D3DPOOL_DEFAULT
        ),
        BackingClass::Other
    );
}

// ── the gauge ──

#[test]
fn live_bytes_track_every_backing_transition() {
    with_gauge(|gauge| {
        let base = live_backing_bytes();
        let mut writeonly = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        let padded = writeonly.padded_len() as u64;
        assert_eq!(
            live_backing_bytes().write_only_static,
            base.write_only_static + padded
        );

        // A release returns the bytes; a restore charges them again.
        drop(
            writeonly
                .release()
                .expect("a fresh backing holds a page box"),
        );
        assert_eq!(
            live_backing_bytes().write_only_static,
            base.write_only_static
        );
        writeonly.restore(PageBox::new_zeroed(SMALL as usize));
        assert_eq!(
            live_backing_bytes().write_only_static,
            base.write_only_static + padded
        );

        // A rename charges the replacement and hands the old box back.
        let mut dynamic = backing(gauge, D3DUSAGE_DYNAMIC, D3DPOOL_DEFAULT);
        let old = dynamic
            .replace(PageBox::new_zeroed(SMALL as usize))
            .expect("a live backing hands its page box back on a rename");
        drop(old);
        assert_eq!(live_backing_bytes().dynamic, base.dynamic + padded);

        // The lockable pools land on the third counter, and dropping a whole
        // backing returns its bytes with no explicit release.
        let other = backing(gauge, 0, D3DPOOL_SYSTEMMEM);
        assert_eq!(live_backing_bytes().other, base.other + padded);
        assert_eq!(live_backing_bytes().total(), base.total() + 3 * padded);

        drop(other);
        drop(dynamic);
        drop(writeonly);
        assert_eq!(live_backing_bytes().total(), base.total());
    });
}

// ── the state machine ──

#[test]
fn a_fresh_backing_mirrors_the_device_and_may_widen() {
    with_gauge(|gauge| {
        let b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        assert_eq!(b.state(), BackingState::Mirrors);
        assert!(!b.is_released());
        assert!(b.may_widen_upload());
    });
}

#[test]
fn a_re_created_backing_is_partial_and_may_not_widen() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        assert_eq!(b.state(), BackingState::Released);
        assert!(b.is_released());
        assert!(!b.may_widen_upload());
        assert_eq!(b.ptr(), 0);
        assert!(b.as_slice().is_empty());
        assert!(b.read_ptr_at(0).is_none());

        b.restore(PageBox::new_zeroed(SMALL as usize));
        assert_eq!(b.state(), BackingState::Partial);
        assert!(!b.is_released());
        assert!(!b.may_widen_upload());
        assert_ne!(b.ptr(), 0);
    });
}

#[test]
fn a_release_after_a_restore_hands_the_second_box_back() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        assert!(b.release().is_none(), "nothing left to release");
        b.restore(PageBox::new_zeroed(SMALL as usize));
        assert!(b.release().is_some(), "the restored box comes back");
    });
}

#[test]
fn a_whole_buffer_upload_returns_a_partial_backing_to_mirroring() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        b.restore(PageBox::new_zeroed(SMALL as usize));

        // A sub-range upload leaves the rest of the buffer unaccounted for.
        b.note_upload(0, SMALL - 1);
        assert_eq!(b.state(), BackingState::Partial);
        b.note_upload(1, SMALL);
        assert_eq!(b.state(), BackingState::Partial);

        // Covering every byte makes the device buffer a copy of the backing.
        b.note_upload(0, SMALL);
        assert_eq!(b.state(), BackingState::Mirrors);
        assert!(b.may_widen_upload());
    });
}

#[test]
fn note_upload_never_moves_a_released_backing() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        b.note_upload(0, SMALL);
        assert_eq!(b.state(), BackingState::Released);
    });
}

#[test]
fn the_padded_length_survives_a_release() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        let padded = b.padded_len();
        assert!(padded >= SMALL as usize);
        drop(b.release());
        assert_eq!(
            b.padded_len(),
            padded,
            "draw snapshots still carry the buffer's length"
        );
    });
}

#[test]
fn write_pointers_land_inside_the_backing_and_stop_at_its_end() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        let base = b.ptr();
        let padded = b.padded_len();
        let at = b.write_ptr_at(64).expect("64 is inside the backing");
        assert_eq!(at as usize, base + 64);
        assert!(
            b.write_ptr_at(padded).is_some(),
            "one past the end is legal"
        );
        assert!(b.write_ptr_at(padded + 1).is_none());
        drop(b.release());
        assert!(b.write_ptr_at(0).is_none());
    });
}

// ── the read-back copy the indexed triangle-fan rewrite installs ──

#[test]
fn a_read_back_copy_mirrors_the_device_buffer_and_pins_the_backing() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        assert!(!b.is_pinned());
        drop(b.release());
        b.adopt_device_copy(PageBox::new_zeroed(SMALL as usize));
        assert_eq!(
            b.state(),
            BackingState::Mirrors,
            "the copy came out of the device buffer, so it holds every byte"
        );
        assert!(b.may_widen_upload());
        assert!(
            b.is_pinned(),
            "the read cost a GPU stall, so it is paid once"
        );
        assert_ne!(b.ptr(), 0);
    });
}

#[test]
fn a_read_back_copy_is_dropped_when_the_buffer_still_holds_a_backing() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        let live = b.ptr();
        b.adopt_device_copy(PageBox::new_zeroed(SMALL as usize));
        assert_eq!(b.ptr(), live, "the live backing stands");
        assert!(!b.is_pinned());
    });
}

#[test]
fn a_pinned_backing_keeps_the_pin_across_an_upload() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        b.adopt_device_copy(PageBox::new_zeroed(SMALL as usize));
        b.note_upload(0, SMALL);
        assert!(b.is_pinned());
    });
}

#[test]
fn a_read_back_copy_completes_partial_backing_without_moving_a_live_lock() {
    with_gauge(|gauge| {
        let mut b = backing(gauge, D3DUSAGE_WRITEONLY, D3DPOOL_DEFAULT);
        drop(b.release());
        b.restore(PageBox::new_zeroed(SMALL as usize));
        b.as_mut_slice()[16..32].fill(7);
        b.note_upload(16, 32);
        let address = b.ptr();
        let generation = b.generation();
        let mut device_copy = PageBox::new_zeroed(SMALL as usize);
        device_copy.as_mut_slice().fill(3);
        device_copy.as_mut_slice()[16..32].fill(7);
        b.adopt_device_copy(device_copy);
        assert_eq!(b.state(), BackingState::Mirrors);
        assert!(b.is_pinned());
        assert_eq!(b.ptr(), address, "an outstanding lock keeps its pointer");
        assert_eq!(b.generation(), generation);
        assert!(b.as_slice()[..16].iter().all(|&byte| byte == 3));
        assert!(b.as_slice()[16..32].iter().all(|&byte| byte == 7));
        assert!(b.as_slice()[32..].iter().all(|&byte| byte == 3));
    });
}
