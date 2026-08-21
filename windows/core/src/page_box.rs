//! Page-aligned, page-sized heap backing for dynamic VB/IB data.
//!
//! Metal's `newBufferWithBytesNoCopy:length:options:deallocator:` requires
//! both the backing pointer and the length to be page-aligned. 16 KiB
//! covers both Apple Silicon (16 KiB pages) and x86 macOS (4 KiB pages,
//! so a 16 KiB multiple is also 4 KiB-aligned).
//!
//! `logical_len` is the unrounded length the game sees through Lock; the
//! raw `len` is the rounded-up page multiple that Metal sees. Everything
//! past `logical_len` is padding — game writes never reach it, GPU reads
//! stay within the vertex/index stride × count the draw specifies.

#[cfg(perf_tracking)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    alloc::{self, Layout},
    ptr::NonNull,
};

/// Cumulative count of `PageBox` allocations served by the global allocator.
#[cfg(perf_tracking)]
static PAGEBOX_ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Cumulative padded bytes behind `PAGEBOX_ALLOCS`.
#[cfg(perf_tracking)]
static PAGEBOX_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
/// Cumulative count of `PageBox` frees returned to the global allocator.
#[cfg(perf_tracking)]
static PAGEBOX_FREES: AtomicU64 = AtomicU64::new(0);
/// Cumulative padded bytes behind `PAGEBOX_FREES`.
#[cfg(perf_tracking)]
static PAGEBOX_FREE_BYTES: AtomicU64 = AtomicU64::new(0);
/// Subset of `PAGEBOX_ALLOCS` that snmalloc cannot serve from its cache.
///
/// See [`bypasses_local_cache`]: each one costs a commit on the way in
/// and a decommit on the way out. `PageBoxPool` hits never reach here,
/// so what this counts is traffic no pool is currently absorbing.
#[cfg(perf_tracking)]
static PAGEBOX_UNCACHED_ALLOCS: AtomicU64 = AtomicU64::new(0);

/// Count one `PageBox` allocation of `len` padded bytes.
///
/// Relaxed bumps on process-wide statics: the constructors run on the API
/// thread and the encoder thread (padded blit staging), so per-frame
/// counter homes would need two copies. Cheap enough to stay ungated.
#[cfg(perf_tracking)]
fn note_alloc(len: usize) {
    PAGEBOX_ALLOCS.fetch_add(1, Ordering::Relaxed);
    PAGEBOX_ALLOC_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    if bypasses_local_cache(len) {
        PAGEBOX_UNCACHED_ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Twin of `note_alloc`: compiles away without `perf_tracking`.
#[cfg(not(perf_tracking))]
const fn note_alloc(_len: usize) {}

/// Count one `PageBox` free of `len` padded bytes.
///
/// Same statics discipline as `note_alloc`; called from `Drop`, which can
/// run on either thread.
#[cfg(perf_tracking)]
fn note_free(len: usize) {
    PAGEBOX_FREES.fetch_add(1, Ordering::Relaxed);
    PAGEBOX_FREE_BYTES.fetch_add(len as u64, Ordering::Relaxed);
}

/// Twin of `note_free`: compiles away without `perf_tracking`.
#[cfg(not(perf_tracking))]
const fn note_free(_len: usize) {}

/// Snapshot of the cumulative `PageBox` allocator-traffic counters.
///
/// All values are process-wide and monotonically increasing since start;
/// consumers delta two snapshots to get a window. `frees` lags `allocs`
/// by the number of live boxes.
pub struct PageBoxVolume {
    pub allocs: u64,
    pub alloc_bytes: u64,
    pub frees: u64,
    pub free_bytes: u64,
    /// Subset of `allocs` that missed snmalloc's per-thread cache.
    ///
    /// Each one is a commit/decommit syscall round trip rather than a
    /// buddy operation, so this is the count worth driving to zero. See
    /// [`bypasses_local_cache`].
    pub uncached_allocs: u64,
}

impl Default for PageBoxVolume {
    fn default() -> Self {
        Self::new()
    }
}

impl PageBoxVolume {
    /// All-zero snapshot, doubling as the "no baseline yet" sentinel.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allocs: 0,
            alloc_bytes: 0,
            frees: 0,
            free_bytes: 0,
            uncached_allocs: 0,
        }
    }

    /// True when nothing has ever been counted.
    ///
    /// A real process is never at zero after the first buffer create, so
    /// this identifies a fresh baseline (or the `not(perf_tracking)` twin).
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.allocs == 0 && self.frees == 0
    }

    /// Element-wise `self - prev`, saturating.
    ///
    /// Turns two cumulative snapshots into a per-window volume.
    #[must_use]
    pub const fn delta(&self, prev: &Self) -> Self {
        Self {
            allocs: self.allocs.saturating_sub(prev.allocs),
            alloc_bytes: self.alloc_bytes.saturating_sub(prev.alloc_bytes),
            frees: self.frees.saturating_sub(prev.frees),
            free_bytes: self.free_bytes.saturating_sub(prev.free_bytes),
            uncached_allocs: self.uncached_allocs.saturating_sub(prev.uncached_allocs),
        }
    }
}

/// Snapshot the cumulative `PageBox` traffic counters (Relaxed loads).
///
/// The counters only ever grow, so a delta between two snapshots is the
/// traffic that reached the global allocator in between: a recycle path
/// that reuses boxes without dropping them is invisible here by design.
#[cfg(perf_tracking)]
#[must_use]
pub fn pagebox_volume() -> PageBoxVolume {
    PageBoxVolume {
        allocs: PAGEBOX_ALLOCS.load(Ordering::Relaxed),
        alloc_bytes: PAGEBOX_ALLOC_BYTES.load(Ordering::Relaxed),
        frees: PAGEBOX_FREES.load(Ordering::Relaxed),
        free_bytes: PAGEBOX_FREE_BYTES.load(Ordering::Relaxed),
        uncached_allocs: PAGEBOX_UNCACHED_ALLOCS.load(Ordering::Relaxed),
    }
}

/// Twin of `pagebox_volume`: no counters exist without `perf_tracking`.
#[cfg(not(perf_tracking))]
#[must_use]
pub const fn pagebox_volume() -> PageBoxVolume {
    PageBoxVolume::new()
}

/// Apple Silicon page size.
///
/// Metal's `newBufferWithBytesNoCopy` demands the backing be page-aligned +
/// page-sized; 16 KiB satisfies both `ASi` and x86 macOS.
pub const PAGE_SIZE: usize = 16 * 1024;

/// snmalloc's per-thread large-object cache budget, in bytes.
///
/// `LocalCacheSizeBits = 21` in the vendored `backend/base_constants.h`,
/// with no target `#ifdef` around it, so the budget is the same on every
/// target we build. This is the cache *budget*, not an allocation-size
/// cutoff: a chunk that large cannot fit inside the budget, so
/// `backend_helpers/largebuddyrange.h` forwards both its alloc and its
/// free past the thread-local buddy to `CommitRange`, turning every
/// alloc into a `VirtualAlloc(MEM_COMMIT)` and every free into a
/// `VirtualFree(MEM_DECOMMIT)`.
pub const SNMALLOC_LOCAL_CACHE_BYTES: usize = 2 * 1024 * 1024;

/// Chunk size snmalloc serves a large request from.
///
/// `large_size_to_chunk_size` is `bits::next_pow2` in the vendored
/// `mem/sizeclasstable.h`, so a large request rounds up to the next
/// power of two. That rounding is why the uncached band starts well
/// below [`SNMALLOC_LOCAL_CACHE_BYTES`].
///
/// Only meaningful above the small-sizeclass ceiling; below it snmalloc
/// serves exact classes from a slab and does not round. That ceiling is
/// 64 KiB on i686 and 512 KiB on the 64-bit targets, so callers that
/// depend on the exact value must stay above 512 KiB.
#[must_use]
pub const fn snmalloc_chunk_size(padded: usize) -> usize {
    padded.next_power_of_two()
}

/// Does an allocation of `padded` bytes miss snmalloc's per-thread cache?
///
/// True means every alloc/free pair at this size is a commit/decommit
/// syscall round trip instead of a thread-local buddy operation.
///
/// The effective cutoff is *over 1 MiB*, derived rather than written
/// down so that it tracks both upstream facts automatically:
///
/// | `padded` | [`snmalloc_chunk_size`] | result |
/// |---|---|---|
/// | 1 MiB | 1 MiB | `false`, cached |
/// | 1 MiB + 1 | 2 MiB | `true`, uncached |
/// | 2.75 MB | 4 MiB | `true`, uncached |
///
/// Hardcoding 1 MiB instead would go stale the moment either the budget
/// or the rounding rule changed, with nothing to catch it.
///
/// Upstream tests `>= mask_bits(21)`, i.e. `2^21 - 1`, where this tests
/// `>= 2^21`. The two agree on every input, because the argument is
/// always a power of two and none lies in between.
#[must_use]
pub const fn bypasses_local_cache(padded: usize) -> bool {
    snmalloc_chunk_size(padded) >= SNMALLOC_LOCAL_CACHE_BYTES
}

/// RAII wrapper around a `std::alloc::alloc`-ed page-aligned byte region.
///
/// Sized to the next page multiple of `logical_len`; both the raw pointer
/// and the reported length are safe to hand to Metal via
/// `newBufferWithBytesNoCopy:`.
pub struct PageBox {
    ptr: NonNull<u8>,
    /// Rounded-up page multiple.
    ///
    /// What `len()` reports and what the `MTLBuffer` wraps.
    len: usize,
    /// Original request from the caller. Game's visible buffer length.
    logical_len: usize,
    /// Layout used for `alloc` — stored so `Drop` can match `dealloc`.
    layout: Layout,
}

// SAFETY: PageBox owns a heap allocation with no interior sharing. Transferring
// ownership across threads is sound — callers must externally synchronize to
// prevent simultaneous reads/writes, the same contract as `Box<[u8]>`.
unsafe impl Send for PageBox {}
// SAFETY: same as Send above — `&PageBox` only exposes the raw pointer, so
// `Sync` requires the same caller-synchronized contract.
unsafe impl Sync for PageBox {}

impl PageBox {
    /// Allocate `logical_len` bytes rounded up to the next page multiple.
    ///
    /// Contents are uninitialized. For any caller that returns the
    /// pointer to the game, the game is expected to write every byte it
    /// later reads.
    ///
    /// # Panics
    ///
    /// Panics when the allocator returns null. Both profiles build with
    /// `panic = "abort"`, so this aborts either way — but panicking runs
    /// the panic hook first, which dumps the crumb ring
    /// (`std::alloc::handle_alloc_error` would abort straight away with no
    /// trace). There is no recovery to attempt: retained VB/IB bytes are
    /// bounded proactively by the retention cap long before the address
    /// space runs out.
    #[must_use]
    pub fn new_uninit(logical_len: usize) -> Self {
        let (len, layout) = Self::layout_for(logical_len);
        // SAFETY: `layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).expect("PageBox alloc failed");
        note_alloc(len);
        Self {
            ptr,
            len,
            logical_len,
            layout,
        }
    }

    /// Same as `new_uninit` but zero-initializes the full padded region.
    ///
    /// Used by VB/IB creation so a first-draw-before-Lock sees defined
    /// bytes. Costs one `bzero` per buffer create — negligible compared
    /// to the alternative of every rename paying for the same zero init.
    ///
    /// # Panics
    ///
    /// Same allocation-failure contract as `new_uninit`.
    #[must_use]
    pub fn new_zeroed(logical_len: usize) -> Self {
        let (len, layout) = Self::layout_for(logical_len);
        // SAFETY: `layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("PageBox alloc failed");
        note_alloc(len);
        Self {
            ptr,
            len,
            logical_len,
            layout,
        }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Borrow the full padded region as a slice.
    ///
    /// Matches `Box<[u8]>`'s deref semantic: callers that consume `&[u8]`
    /// see exactly the same byte range.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null, valid for `len` bytes (we allocated
        // it that way), and the lifetime is tied to `&self`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable counterpart of `as_slice`.
    ///
    /// Caller takes the unique-borrow guarantee from `&mut self`.
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is non-null, valid for `len` bytes, and the
        // unique borrow on `&mut self` keeps the slice exclusive.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Padded length (page multiple).
    ///
    /// This is what the Metal `MTLBuffer` wrapper sees.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Original `logical_len` the caller requested.
    ///
    /// What the game sees through Lock.
    #[must_use]
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.logical_len == 0
    }

    /// Padded length a `logical_len` request allocates (next page multiple, min one page).
    ///
    /// Public so the recycle pool can classify a request by the padded
    /// size the constructors would produce, without allocating.
    ///
    /// # Panics
    ///
    /// Panics when rounding up overflows `usize` — unreachable for any
    /// length a 32-bit D3D9 buffer can carry.
    #[must_use]
    pub fn padded_len(logical_len: usize) -> usize {
        logical_len
            .max(1)
            .div_ceil(PAGE_SIZE)
            .checked_mul(PAGE_SIZE)
            .expect("PageBox length overflow")
    }

    /// Retarget a recycled box to a new request of the same padded size.
    ///
    /// Only the logical length changes; pointer, padded length, and layout
    /// stay. Contents are stale bytes from the previous owner, the same
    /// caller contract as `new_uninit`.
    pub fn set_logical_len(&mut self, logical_len: usize) {
        debug_assert_eq!(
            Self::padded_len(logical_len),
            self.len,
            "recycled PageBox must keep its padded length"
        );
        self.logical_len = logical_len;
    }

    fn layout_for(logical_len: usize) -> (usize, Layout) {
        // A zero-length PageBox is still allocated at one page so the
        // returned pointer is non-null and the MTLBuffer wrap doesn't
        // choke on length=0.
        let padded = Self::padded_len(logical_len);
        let layout = Layout::from_size_align(padded, PAGE_SIZE).expect("valid page-aligned layout");
        (padded, layout)
    }
}

impl Drop for PageBox {
    fn drop(&mut self) {
        note_free(self.len);
        // SAFETY: same layout used to alloc; pointer came from that
        // allocator; nothing else owns this allocation.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests;
