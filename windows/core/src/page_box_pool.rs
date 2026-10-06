//! Bounded per-size-class recycle pool for retired [`PageBox`]es.
//!
//! The VB/IB Lock-rename path allocates a fresh box per contended Lock on
//! the API thread and frees the retired one a frame later. Cycling those
//! boxes through the global allocator lets its page-return policy
//! decommit them, so the game's first touch of the next fresh box pays a
//! zero-fill fault plus mapping syscalls (expensive under Wine + Rosetta).
//! Parking retired boxes here and popping a same-size one on the next
//! rename keeps the pages committed and warm, and skips the allocator's
//! free path entirely.
//!
//! Texture staging uses a second lane of the same pool. `CreateTexture`
//! allocates one box per mip and the last owner of each drops it a few
//! frames after the texture is released, on the API thread; recycling
//! them turns the per-mip allocation into a vector pop. The two lanes keep
//! separate class stacks and share one byte cap, and the staging lane may
//! hold at most [`STAGING_SHARE_DIVISOR`]th of it, so staging parked in
//! classes nobody requests again cannot take the budget the rename path
//! depends on.
//!
//! Every push and pop runs inside a D3D9 call or the submission it makes
//! (lease retirement runs there under the device's retirement lock), at
//! a few thousand O(1) critical sections per second at most, so one plain
//! `Mutex` is enough. A texture create takes it once for all its levels
//! ([`StagingTake`]). The parked-bytes gauge is mirrored in an atomic for
//! lock-free reads by the perf summary.

use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicUsize, Ordering},
};

use crate::page_box::{PAGE_SIZE, PageBox};

#[cfg(perf_tracking)]
mod diagnostics;

/// Largest box the pool parks, in 16 KiB pages (256 pages = 4 MiB).
///
/// Sized from measurement, not guesswork: the dominant renamed buffer in
/// the target game is ~2.75 MB (176 pages), renamed once or twice per
/// frame, and it is precisely the class that hurts most in the allocator.
/// Any request over 1 MiB rounds up to a chunk of at least
/// [`crate::page_box::SNMALLOC_LOCAL_CACHE_BYTES`], which no longer fits
/// snmalloc's per-thread budget, so every free decommits and every alloc
/// re-commits; 2.75 MB rounds to a 4 MiB chunk and pays that round trip
/// on both ends. Parking the box here turns the syscall pair into a
/// vector pop. 4 MiB leaves headroom above the dominant class; anything
/// larger is a rare one-off that would crowd the byte cap out of the hot
/// classes and drops to the allocator instead.
pub const MAX_POOL_CLASSES: usize = 256;

/// The staging lane parks at most `cap / STAGING_SHARE_DIVISOR` bytes.
///
/// Texture staging classes follow whatever texture set the game streams, so
/// a zone unload can park many boxes of sizes nothing requests again. The
/// rename path's parked set peaks near 59 MB under the default 128 MiB cap;
/// a quarter for staging leaves it 96 MiB, while the few frames of retired
/// staging a streaming engine cycles through fit in a few MiB.
pub const STAGING_SHARE_DIVISOR: usize = 4;

/// One lane's class stacks and parked bytes.
struct Lane {
    /// One LIFO stack per size class; index = padded pages - 1.
    ///
    /// LIFO on purpose: the most recently freed box is the most likely to
    /// still have warm, committed pages.
    classes: Vec<Vec<PageBox>>,
    /// Padded bytes parked across this lane's classes.
    bytes: usize,
    #[cfg(perf_tracking)]
    diagnostics: diagnostics::Diagnostics,
}

impl Lane {
    fn new() -> Self {
        Self {
            classes: (0..MAX_POOL_CLASSES).map(|_| Vec::new()).collect(),
            bytes: 0,
            #[cfg(perf_tracking)]
            diagnostics: diagnostics::Diagnostics::empty(),
        }
    }
}

/// The two users of the pool, each served only from its own class stacks.
#[derive(Clone, Copy)]
enum LaneKind {
    /// VB/IB backing retired by Lock-rename.
    Buffer = 0,
    /// Texture staging whose last owner dropped it.
    Staging = 1,
}

/// Mutex-guarded pool state.
struct PoolInner {
    lanes: [Lane; 2],
}

impl PoolInner {
    const fn total_bytes(&self) -> usize {
        self.lanes[0].bytes + self.lanes[1].bytes
    }

    /// Pop a parked box of `logical_len`'s class from `kind`'s lane, keeping `gauge` in step.
    ///
    /// The caller has checked that the pool is enabled.
    fn pop(&mut self, kind: LaneKind, logical_len: usize, gauge: &AtomicUsize) -> Option<PageBox> {
        // The page count less one, which is what the padded length names
        // without forming it: a length too long to round up is oversize too.
        let class = logical_len.max(1).div_ceil(PAGE_SIZE) - 1;
        let lane = &mut self.lanes[kind as usize];
        if class >= MAX_POOL_CLASSES {
            #[cfg(perf_tracking)]
            lane.diagnostics
                .acquire(diagnostics::Acquire::Oversize, logical_len);
            return None;
        }
        let pb = lane.classes[class].pop();
        #[cfg(perf_tracking)]
        lane.diagnostics.acquire(
            if pb.is_some() {
                diagnostics::Acquire::Hit
            } else {
                diagnostics::Acquire::Empty
            },
            logical_len,
        );
        let mut pb = pb?;
        lane.bytes -= pb.len();
        gauge.store(self.total_bytes(), Ordering::Relaxed);
        pb.set_logical_len(logical_len);
        Some(pb)
    }
}

/// The texture staging one API call allocates, taken under one pool lock.
///
/// `CreateTexture` takes a box per mip, so the lock is held from
/// [`PageBoxPool::take_staging`] until [`Self::finish`] rather than taken
/// once per level. A miss allocates while the lock is held; nothing the
/// allocator does reaches the pool, and every other pool user is an API
/// call or its submission, so the hold costs no one a wait in practice.
/// Nothing may touch the pool again on this thread before `finish`, since
/// the mutex is not reentrant.
///
/// Counts hits and, while the pool is enabled, misses; a disabled pool
/// takes no lock and counts neither, so the baseline arm of an A/B reads 0/0.
pub struct StagingTake<'a> {
    pool: &'a PageBoxPool,
    /// The pool's state while it is enabled; `None` falls every take through to the allocator.
    guard: Option<MutexGuard<'a, PoolInner>>,
    hits: u32,
    misses: u32,
}

impl StagingTake<'_> {
    /// Staging for `logical_len` bytes: a parked box of the same padded size, else a fresh one.
    ///
    /// Either way the contents are uninitialized, the contract of
    /// [`PageBox::new_uninit`].
    ///
    /// # Panics
    ///
    /// When no box is parked and the allocation fails, as
    /// [`PageBox::new_uninit`] does.
    pub fn take(&mut self, logical_len: usize) -> PageBox {
        self.try_take(logical_len).expect("PageBox alloc failed")
    }

    /// [`Self::take`] that answers `None` when no box is parked and the allocation fails.
    ///
    /// For the staging a texture gets at creation, the one place a staging
    /// allocation may fail (see [`PageBox::try_new_uninit`]).
    pub fn try_take(&mut self, logical_len: usize) -> Option<PageBox> {
        let Some(inner) = self.guard.as_mut() else {
            #[cfg(perf_tracking)]
            self.pool.record_acquire(
                LaneKind::Staging,
                diagnostics::Acquire::Disabled,
                logical_len,
            );
            return PageBox::try_new_uninit(logical_len);
        };
        if let Some(pb) = inner.pop(LaneKind::Staging, logical_len, &self.pool.pooled_bytes) {
            self.hits = self.hits.saturating_add(1);
            return Some(pb);
        }
        self.misses = self.misses.saturating_add(1);
        PageBox::try_new_uninit(logical_len)
    }

    /// Release the pool and report `(hits, misses)` for the device's counters.
    #[must_use]
    pub fn finish(self) -> (u32, u32) {
        (self.hits, self.misses)
    }
}

/// Snapshot of one pool's VB/IB parks for the perf summary (see [`PageBoxPool::buffer_traffic`]).
#[cfg(perf_tracking)]
pub struct PoolTraffic {
    /// Boxes the VB/IB lane has parked since the pool was made.
    pub recycled: u64,
    /// Padded bytes behind `recycled`.
    pub recycled_bytes: u64,
    /// Padded bytes parked across both lanes now.
    pub parked_bytes: u64,
}

/// Bounded recycle pool for retired [`PageBox`]es.
///
/// Constructed with a byte cap (`0` = disabled: nothing is ever parked and
/// every acquire misses) that [`Self::set_cap`] can move later, since the
/// pool outlives the configuration that sizes it. Boxes are matched by
/// exact padded size; there is no splitting or coalescing, because both
/// workloads re-request the same handful of sizes within a frame or two.
/// The cap covers both lanes together.
pub struct PageBoxPool {
    inner: Mutex<PoolInner>,
    /// Mirror of both lanes' parked bytes for lock-free gauge reads.
    pooled_bytes: AtomicUsize,
    /// Byte cap; `0` disables the pool.
    ///
    /// Read without the lock on the fast path and written by
    /// [`Self::set_cap`]; a box parked under an earlier, larger cap stays
    /// parked, so parked bytes can exceed a lowered cap until they are
    /// acquired.
    cap_bytes: AtomicUsize,
}

impl PageBoxPool {
    /// Pool with `cap_bytes` of parking budget (`0` = disabled).
    #[must_use]
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                lanes: [Lane::new(), Lane::new()],
            }),
            pooled_bytes: AtomicUsize::new(0),
            cap_bytes: AtomicUsize::new(cap_bytes),
        }
    }

    /// Move the parking budget to `cap_bytes` (`0` = disabled).
    pub fn set_cap(&self, cap_bytes: usize) {
        self.cap_bytes.store(cap_bytes, Ordering::Relaxed);
    }

    /// The parking budget in bytes; `0` while the pool is disabled.
    #[must_use]
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes.load(Ordering::Relaxed)
    }

    /// True when a non-zero cap is configured.
    ///
    /// Callers use this to keep hit/miss counters silent while the pool
    /// is off, so the A/B baseline arm reads 0/0 instead of all-miss.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.cap_bytes() != 0
    }

    /// Pop a parked VB/IB box whose padded size matches `logical_len`'s class.
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
        self.acquire_in(LaneKind::Buffer, logical_len)
    }

    /// Pop a parked texture staging box whose padded size matches `logical_len`'s class.
    ///
    /// The staging-lane twin of [`Self::acquire`], with the same contract:
    /// a hit keeps its pages, alignment and generation and carries stale
    /// contents, which every staging consumer tolerates because staging is
    /// created uninitialized.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    #[must_use]
    pub fn acquire_staging(&self, logical_len: usize) -> Option<PageBox> {
        self.acquire_in(LaneKind::Staging, logical_len)
    }

    /// Start taking one API call's texture staging, under one lock while the pool is enabled.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    #[must_use]
    pub fn take_staging(&self) -> StagingTake<'_> {
        StagingTake {
            pool: self,
            guard: self
                .enabled()
                .then(|| self.inner.lock().expect("PageBoxPool mutex poisoned")),
            hits: 0,
            misses: 0,
        }
    }

    /// Park a retired VB/IB box, or hand it back for a plain drop.
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
        self.recycle_in(LaneKind::Buffer, pb)
    }

    /// Drop one owner of a texture staging box, parking the box if that was the last owner.
    ///
    /// Parks only when `Arc::into_inner` proves no other owner remains,
    /// which is what makes the pages unreachable: every upload job, reader
    /// guard and native lease of a staging box holds an `Arc` of it, and
    /// native code cannot adopt the pages except through a lease that PE
    /// retains until native code acknowledges its last use. A box that
    /// still counts a reader without any other owner breaks that
    /// invariant, and is dropped rather than parked. Returns true when the
    /// box was parked; false leaves it with its other owners or drops it
    /// to the allocator (disabled pool, oversize class, cap or share
    /// reached, or a borrowed guest page).
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    pub fn recycle_staging(&self, backing: Arc<PageBox>) -> bool {
        let Some(pb) = Arc::into_inner(backing) else {
            return false;
        };
        if pb.has_readers() {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "page-box pool: a staging box with no owner left still counts a reader → dropped, \
                 not parked");
            return false;
        }
        self.recycle_in(LaneKind::Staging, pb).is_none()
    }

    /// Free every parked texture staging box, leaving the VB/IB lane alone.
    ///
    /// Device teardown calls this: staging classes follow the texture set
    /// of the device that created them, and a device that is gone should
    /// not keep committed pages parked in the 32-bit address space on its
    /// behalf. The lane is process-wide, so this also frees boxes parked by
    /// other devices that are still alive; they lose warm pages, nothing
    /// else, since a parked box has no owner. The boxes are dropped after
    /// the lock is released. Returns the padded bytes freed.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    pub fn drain_staging(&self) -> usize {
        let mut inner = self.inner.lock().expect("PageBoxPool mutex poisoned");
        let lane = &mut inner.lanes[LaneKind::Staging as usize];
        let bytes = core::mem::take(&mut lane.bytes);
        let classes: Vec<Vec<PageBox>> = lane
            .classes
            .iter_mut()
            .filter(|class| !class.is_empty())
            .map(core::mem::take)
            .collect();
        self.pooled_bytes
            .store(inner.total_bytes(), Ordering::Relaxed);
        drop(inner);
        drop(classes);
        bytes
    }

    /// Padded bytes parked in the texture staging lane.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    #[must_use]
    pub fn staging_bytes(&self) -> usize {
        self.inner.lock().expect("PageBoxPool mutex poisoned").lanes[LaneKind::Staging as usize]
            .bytes
    }

    /// Report pool totals on the renderer's existing performance cadence.
    ///
    /// Counters include every device sharing this pool and never reset at frame boundaries.
    /// The snapshot uses the pool mutex, so each outcome partition is internally consistent.
    /// `runtime` names the binary that owns the pool, closing the line, since each runtime
    /// logs its own pool's.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    #[cfg(perf_tracking)]
    pub fn log_diagnostics(&self, runtime: &str) {
        let text = self.diagnostics_summary();
        log::info!(target: "mtld3d::perf", "{text} runtime={runtime}");
    }

    /// The VB/IB lane's parks so far and the bytes both lanes hold now, for the perf summary.
    ///
    /// The parks are cumulative; a reader deltas two of these. The texture
    /// staging lane's parks are left out, as the summary's pool row counts
    /// only the buffer lane's.
    ///
    /// # Panics
    ///
    /// Panics if the pool mutex was poisoned.
    #[cfg(perf_tracking)]
    #[must_use]
    pub fn buffer_traffic(&self) -> PoolTraffic {
        let (recycled, recycled_bytes) =
            self.inner.lock().expect("PageBoxPool mutex poisoned").lanes[LaneKind::Buffer as usize]
                .diagnostics
                .parked();
        PoolTraffic {
            recycled,
            recycled_bytes,
            parked_bytes: self.pooled_bytes() as u64,
        }
    }

    /// Padded bytes currently parked across both lanes (lock-free Relaxed read).
    #[must_use]
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes.load(Ordering::Relaxed)
    }

    fn acquire_in(&self, kind: LaneKind, logical_len: usize) -> Option<PageBox> {
        if !self.enabled() {
            #[cfg(perf_tracking)]
            self.record_acquire(kind, diagnostics::Acquire::Disabled, logical_len);
            return None;
        }
        if PageBox::padded_len(logical_len) / PAGE_SIZE > MAX_POOL_CLASSES {
            #[cfg(perf_tracking)]
            self.record_acquire(kind, diagnostics::Acquire::Oversize, logical_len);
            return None;
        }
        self.inner.lock().expect("PageBoxPool mutex poisoned").pop(
            kind,
            logical_len,
            &self.pooled_bytes,
        )
    }

    fn recycle_in(&self, kind: LaneKind, pb: PageBox) -> Option<PageBox> {
        if !pb.is_native_owned() {
            return Some(pb);
        }
        let cap_bytes = self.cap_bytes();
        if cap_bytes == 0 {
            #[cfg(perf_tracking)]
            self.record_recycle(kind, diagnostics::Recycle::Disabled, pb.len());
            return Some(pb);
        }
        let class = pb.len() / PAGE_SIZE - 1;
        if class >= MAX_POOL_CLASSES {
            #[cfg(perf_tracking)]
            self.record_recycle(kind, diagnostics::Recycle::Oversize, pb.len());
            return Some(pb);
        }
        let lane_cap = match kind {
            LaneKind::Buffer => cap_bytes,
            LaneKind::Staging => cap_bytes / STAGING_SHARE_DIVISOR,
        };
        let len = pb.len();
        let mut inner = self.inner.lock().expect("PageBoxPool mutex poisoned");
        let total = inner.total_bytes();
        let lane = &mut inner.lanes[kind as usize];
        if total + len > cap_bytes || lane.bytes + len > lane_cap {
            #[cfg(perf_tracking)]
            lane.diagnostics.recycle(diagnostics::Recycle::Full, len);
            drop(inner);
            return Some(pb);
        }
        #[cfg(perf_tracking)]
        lane.diagnostics.recycle(diagnostics::Recycle::Parked, len);
        lane.bytes += len;
        self.pooled_bytes.store(total + len, Ordering::Relaxed);
        lane.classes[class].push(pb);
        drop(inner);
        None
    }

    #[cfg(perf_tracking)]
    fn diagnostics_summary(&self) -> String {
        let inner = self.inner.lock().expect("PageBoxPool mutex poisoned");
        let buffer = inner.lanes[LaneKind::Buffer as usize].diagnostics.summary();
        let staging = inner.lanes[LaneKind::Staging as usize]
            .diagnostics
            .summary();
        drop(inner);
        format!("{buffer} staging: {staging}")
    }

    #[cfg(perf_tracking)]
    fn record_acquire(&self, kind: LaneKind, outcome: diagnostics::Acquire, logical_len: usize) {
        self.inner.lock().expect("PageBoxPool mutex poisoned").lanes[kind as usize]
            .diagnostics
            .acquire(outcome, logical_len);
    }

    #[cfg(perf_tracking)]
    fn record_recycle(&self, kind: LaneKind, outcome: diagnostics::Recycle, padded_len: usize) {
        self.inner.lock().expect("PageBoxPool mutex poisoned").lanes[kind as usize]
            .diagnostics
            .recycle(outcome, padded_len);
    }
}

#[cfg(test)]
mod tests;
