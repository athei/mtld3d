//! Unit tests for the page-aligned heap backing behind dynamic VB/IB data.
//!
//! Two halves. The allocator cutoff maths: the over-1-MiB uncached band stays
//! derived from snmalloc's chunk rounding rather than hardcoded, and the bound
//! agrees with upstream's mask-bits form at every chunk size. Then `PageBox`
//! itself: page-aligned pointers, a padded length rounded up to a page multiple
//! (one page even at zero), and the full logical range readable and writable.

use std::sync::{Arc, mpsc};

use super::{
    PAGE_SIZE, PageBox, PageBoxRead, SNMALLOC_LOCAL_CACHE_BYTES, bypasses_local_cache,
    snmalloc_chunk_size,
};

#[test]
fn read_guards_exclude_cached_owners_and_survive_independent_drops() {
    let backing = Arc::new(PageBox::new_zeroed(64));
    let cached = Arc::clone(&backing);
    let other = Arc::new(PageBox::new_zeroed(64));
    assert!(!backing.has_readers());
    let job = PageBoxRead::new(Arc::clone(&backing));
    let emitted = PageBoxRead::new(Arc::clone(job.backing()));
    assert!(backing.has_readers());
    assert!(!other.has_readers());
    drop(job);
    assert!(backing.has_readers());
    drop(emitted);
    assert!(!backing.has_readers());
    assert_eq!(Arc::strong_count(&backing), 2);
    assert!(Arc::ptr_eq(&backing, &cached));
}

#[test]
fn moving_read_guard_across_threads_keeps_it_live_until_release() {
    let backing = Arc::new(PageBox::new_zeroed(64));
    let read = PageBoxRead::new(Arc::clone(&backing));
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let (released_tx, released_rx) = mpsc::sync_channel(0);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            release_rx.recv().expect("release signal");
            drop(read);
            released_tx.send(()).expect("released signal");
        });
        assert!(backing.has_readers());
        release_tx.send(()).expect("release signal");
        released_rx.recv().expect("released signal");
        assert!(!backing.has_readers());
    });
}

#[test]
fn recycled_backing_has_no_reader_state_to_reset() {
    let backing = Arc::new(PageBox::new_zeroed(64));
    let read = PageBoxRead::new(Arc::clone(&backing));
    let backing = Arc::try_unwrap(backing)
        .err()
        .expect("read retains backing");
    drop(read);
    let backing = Arc::try_unwrap(backing).ok().expect("last read released");
    let generation = backing.generation();
    let pool = crate::page_box_pool::PageBoxPool::new(PAGE_SIZE);
    assert!(pool.recycle(backing).is_none());
    let reused = pool.acquire(128).expect("same page class");
    assert_eq!(reused.generation(), generation);
    assert!(!reused.has_readers());
    let reused = Arc::new(reused);
    let read = PageBoxRead::new(Arc::clone(&reused));
    assert!(reused.has_readers());
    drop(read);
    assert!(!reused.has_readers());
}

/// The derived cutoff sits just above 1 MiB, and nothing hardcodes it.
#[test]
fn local_cache_cutoff_is_one_mib_exclusive() {
    const MIB: usize = 1024 * 1024;
    assert!(!bypasses_local_cache(MIB));
    assert!(bypasses_local_cache(MIB + 1));
    // The measured dominant renamed VB: 176 pages, rounding to 4 MiB.
    assert!(bypasses_local_cache(176 * PAGE_SIZE));
    assert_eq!(snmalloc_chunk_size(176 * PAGE_SIZE), 4 * MIB);
}

/// Our `>= 2^21` test and upstream's `>= 2^21 - 1` agree on every chunk.
///
/// Chunk sizes are always powers of two, and none lies between the two
/// bounds, so the apparent off-by-one is not one.
#[test]
fn chunk_bound_agrees_with_upstream_mask_bits() {
    let upstream_bound = SNMALLOC_LOCAL_CACHE_BYTES - 1;
    let mut chunk = PAGE_SIZE;
    while chunk <= 64 * 1024 * 1024 {
        assert_eq!(
            chunk >= SNMALLOC_LOCAL_CACHE_BYTES,
            chunk >= upstream_bound,
            "disagreement at chunk size {chunk}"
        );
        chunk *= 2;
    }
}

#[test]
fn uninit_alloc_is_page_aligned_and_page_sized() {
    let pb = PageBox::new_uninit(1);
    assert_eq!(pb.as_ptr() as usize % PAGE_SIZE, 0);
    assert_eq!(pb.len(), PAGE_SIZE);
    assert_eq!(pb.logical_len(), 1);
}

#[test]
fn length_rounds_up_to_page_multiple() {
    let pb = PageBox::new_uninit(PAGE_SIZE + 1);
    assert_eq!(pb.len(), 2 * PAGE_SIZE);
    assert_eq!(pb.logical_len(), PAGE_SIZE + 1);
}

#[test]
fn exact_page_multiple_is_not_rounded_further() {
    let pb = PageBox::new_uninit(3 * PAGE_SIZE);
    assert_eq!(pb.len(), 3 * PAGE_SIZE);
}

#[test]
fn zero_length_still_allocates_one_page() {
    let pb = PageBox::new_uninit(0);
    assert_eq!(pb.len(), PAGE_SIZE);
    assert_eq!(pb.logical_len(), 0);
}

#[test]
fn writable_across_full_logical_len() {
    let logical = 5000usize;
    let mut pb = PageBox::new_uninit(logical);
    let ptr = pb.as_mut_ptr();
    for i in 0..logical {
        let byte = u8::try_from(i & 0xff).expect("masked to 0xFF fits u8");
        // SAFETY: `ptr + i` stays within the just-allocated `logical`-byte slab.
        let dst = unsafe { ptr.add(i) };
        // SAFETY: same slab; `u8` writes are always aligned.
        unsafe { dst.write(byte) };
    }
    let rp = pb.as_ptr();
    for i in 0..logical {
        // SAFETY: `rp + i` stays within the just-written `logical`-byte slab.
        let src = unsafe { rp.add(i) };
        // SAFETY: same slab; `u8` reads are always aligned.
        let v = unsafe { src.read() };
        let expected = u8::try_from(i & 0xff).expect("masked to 0xFF fits u8");
        assert_eq!(v, expected);
    }
}

#[test]
fn multiple_allocs_do_not_alias() {
    let a = PageBox::new_uninit(PAGE_SIZE);
    let b = PageBox::new_uninit(PAGE_SIZE);
    assert_ne!(a.as_ptr(), b.as_ptr());
}

#[test]
fn zeroed_init_is_actually_zero() {
    let pb = PageBox::new_zeroed(100);
    let p = pb.as_ptr();
    for i in 0..100 {
        // SAFETY: `p + i` stays within the just-allocated 100-byte zeroed slab.
        let byte_ptr = unsafe { p.add(i) };
        // SAFETY: same slab; `byte_ptr` is well-aligned for `u8`.
        let byte = unsafe { byte_ptr.read() };
        assert_eq!(byte, 0);
    }
}

#[test]
fn drop_does_not_panic() {
    for _ in 0..16 {
        drop(PageBox::new_uninit(64 * 1024));
        drop(PageBox::new_zeroed(8 * 1024));
    }
}
