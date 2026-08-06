//! Bounded per-size-class recycle pool for retired [`PageBox`]es.
//!
//! The VB/IB Lock-rename path allocates a fresh box per contended Lock on
//! the API thread and frees the retired one from the encoder thread a
//! frame later. Cycling those boxes through the global allocator lets its
//! page-return policy decommit them, so the game's first touch of the
//! next fresh box pays a zero-fill fault plus mapping syscalls (expensive
//! under Wine + Rosetta). Parking retired boxes here and popping a
//! same-size one on the next rename keeps the pages committed and warm,
//! and skips the allocator's cross-thread free path entirely.
//!
//! Topology: the encoder thread pushes (retention drain, batched once per
//! frame), the API thread pops (Lock-rename alloc). Both sides are well
//! under a thousand operations per second with O(1) critical sections, so
//! one plain `Mutex` is enough; the parked-bytes gauge is mirrored in an
//! atomic for lock-free reads by the perf summary.

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::page_box::{PAGE_SIZE, PageBox};

/// Largest box the pool parks, in 16 KiB pages (256 pages = 4 MiB).
///
/// Sized from measurement, not guesswork: the dominant renamed buffer in
/// the target game is ~2.75 MB (176 pages), renamed once or twice per
/// frame, and it is precisely the class that hurts most in the allocator
/// (>= 2 MiB chunks bypass snmalloc's thread-local cache, so every free
/// decommits and every alloc re-commits). 4 MiB leaves headroom above it;
/// anything larger is a rare one-off that would crowd the byte cap out of
/// the hot classes and drops to the allocator instead.
pub const MAX_POOL_CLASSES: usize = 256;

/// Mutex-guarded pool state.
struct PoolInner {
    /// One LIFO stack per size class; index = padded pages - 1.
    ///
    /// LIFO on purpose: the most recently freed box is the most likely to
    /// still have warm, committed pages.
    classes: Vec<Vec<PageBox>>,
    /// Padded bytes parked across all classes.
    bytes: usize,
}

/// Bounded recycle pool for retired [`PageBox`]es.
///
/// Constructed once with a byte cap (`0` = disabled: `acquire` never
/// hits, `recycle` returns every box to the caller for a plain drop).
/// Boxes are matched by exact padded size; there is no splitting or
/// coalescing, because the rename workload re-requests the same handful
/// of buffer sizes within a frame or two.
pub struct PageBoxPool {
    inner: Mutex<PoolInner>,
    /// Mirror of `PoolInner::bytes` for lock-free gauge reads.
    pooled_bytes: AtomicUsize,
    /// Byte cap; `0` disables the pool.
    cap_bytes: usize,
}

impl PageBoxPool {
    /// Pool with `cap_bytes` of parking budget (`0` = disabled).
    #[must_use]
    pub fn new(cap_bytes: usize) -> Self {
        let classes = if cap_bytes == 0 {
            Vec::new()
        } else {
            (0..MAX_POOL_CLASSES).map(|_| Vec::new()).collect()
        };
        Self {
            inner: Mutex::new(PoolInner { classes, bytes: 0 }),
            pooled_bytes: AtomicUsize::new(0),
            cap_bytes,
        }
    }

    /// True when a non-zero cap was configured.
    ///
    /// Callers use this to keep hit/miss counters silent while the pool
    /// is off, so the A/B baseline arm reads 0/0 instead of all-miss.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.cap_bytes != 0
    }

    /// Pop a parked box whose padded size matches `logical_len`'s class.
    ///
    /// Returns `None` when the pool is disabled, the class is out of
    /// range, or no box of that class is parked; the caller then
    /// allocates fresh as before. A hit is retargeted to `logical_len`
    /// and carries stale contents (the `new_uninit` contract).
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned, i.e. a previous holder
    /// panicked mid-operation; nothing recoverable remains then.
    #[must_use]
    pub fn acquire(&self, logical_len: usize) -> Option<PageBox> {
        if self.cap_bytes == 0 {
            return None;
        }
        let class = PageBox::padded_len(logical_len) / PAGE_SIZE - 1;
        if class >= MAX_POOL_CLASSES {
            return None;
        }
        let mut pb = {
            let mut inner = self.inner.lock().expect("PageBoxPool mutex poisoned");
            let pb = inner.classes[class].pop()?;
            inner.bytes -= pb.len();
            self.pooled_bytes.store(inner.bytes, Ordering::Relaxed);
            pb
        };
        pb.set_logical_len(logical_len);
        Some(pb)
    }

    /// Park a retired box, or hand it back for a plain drop.
    ///
    /// Returns `Some(pb)` when the pool refuses it (disabled, oversize
    /// class, or the byte cap is reached) so drop responsibility stays
    /// explicit at the call site; `None` means the box was parked.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned, same as [`Self::acquire`].
    #[must_use]
    pub fn recycle(&self, pb: PageBox) -> Option<PageBox> {
        if self.cap_bytes == 0 {
            return Some(pb);
        }
        let class = pb.len() / PAGE_SIZE - 1;
        if class >= MAX_POOL_CLASSES {
            return Some(pb);
        }
        let mut inner = self.inner.lock().expect("PageBoxPool mutex poisoned");
        if inner.bytes + pb.len() > self.cap_bytes {
            return Some(pb);
        }
        inner.bytes += pb.len();
        self.pooled_bytes.store(inner.bytes, Ordering::Relaxed);
        inner.classes[class].push(pb);
        None
    }

    /// Padded bytes currently parked (lock-free Relaxed read).
    #[must_use]
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_POOL_CLASSES, PageBoxPool};
    use crate::page_box::{PAGE_SIZE, PageBox};

    #[test]
    fn disabled_pool_never_stores() {
        let pool = PageBoxPool::new(0);
        assert!(!pool.enabled());
        let pb = PageBox::new_uninit(PAGE_SIZE);
        assert!(pool.recycle(pb).is_some(), "disabled pool must reject");
        assert!(pool.acquire(PAGE_SIZE).is_none());
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn exact_class_round_trip() {
        let pool = PageBoxPool::new(1024 * 1024);
        let pb = PageBox::new_uninit(3 * PAGE_SIZE);
        let ptr = pb.as_ptr();
        assert!(pool.recycle(pb).is_none(), "under cap must park");
        assert_eq!(pool.pooled_bytes(), 3 * PAGE_SIZE);

        // A different class misses.
        assert!(pool.acquire(PAGE_SIZE).is_none());
        // Same padded class hits and is retargeted, same backing pages.
        let hit = pool.acquire(2 * PAGE_SIZE + 1).expect("class hit");
        assert_eq!(hit.as_ptr(), ptr);
        assert_eq!(hit.len(), 3 * PAGE_SIZE);
        assert_eq!(hit.logical_len(), 2 * PAGE_SIZE + 1);
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn cap_rejection_returns_the_box() {
        let pool = PageBoxPool::new(2 * PAGE_SIZE);
        assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
        assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
        // Third box would exceed the cap; it comes back for a plain drop.
        let reject = pool.recycle(PageBox::new_uninit(PAGE_SIZE));
        assert!(reject.is_some());
        assert_eq!(pool.pooled_bytes(), 2 * PAGE_SIZE);
    }

    #[test]
    fn oversize_class_is_rejected() {
        let pool = PageBoxPool::new(usize::MAX);
        let jumbo = PageBox::new_uninit((MAX_POOL_CLASSES + 1) * PAGE_SIZE);
        assert!(pool.recycle(jumbo).is_some());
        assert!(pool.acquire((MAX_POOL_CLASSES + 1) * PAGE_SIZE).is_none());
    }

    #[test]
    fn largest_class_is_accepted() {
        let pool = PageBoxPool::new(usize::MAX);
        let pb = PageBox::new_uninit(MAX_POOL_CLASSES * PAGE_SIZE);
        assert!(pool.recycle(pb).is_none());
        assert!(pool.acquire(MAX_POOL_CLASSES * PAGE_SIZE).is_some());
    }

    #[test]
    fn lifo_returns_most_recently_parked() {
        let pool = PageBoxPool::new(usize::MAX);
        let first = PageBox::new_uninit(PAGE_SIZE);
        let second = PageBox::new_uninit(PAGE_SIZE);
        let second_ptr = second.as_ptr();
        assert!(pool.recycle(first).is_none());
        assert!(pool.recycle(second).is_none());
        let hit = pool.acquire(PAGE_SIZE).expect("hit");
        assert_eq!(hit.as_ptr(), second_ptr, "LIFO: newest box pops first");
    }

    #[test]
    fn byte_accounting_across_mixed_classes() {
        let pool = PageBoxPool::new(usize::MAX);
        assert!(pool.recycle(PageBox::new_uninit(PAGE_SIZE)).is_none());
        assert!(pool.recycle(PageBox::new_uninit(4 * PAGE_SIZE)).is_none());
        assert_eq!(pool.pooled_bytes(), 5 * PAGE_SIZE);
        let _ = pool.acquire(4 * PAGE_SIZE).expect("hit");
        assert_eq!(pool.pooled_bytes(), PAGE_SIZE);
    }
}
