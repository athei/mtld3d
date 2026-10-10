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
    sync::{Arc, atomic::AtomicU32},
};

#[cfg(not(windows))]
use mtld3d_shared::encoder_wire::{LeaseCompletion, LeaseCompletionPtr};

/// Bytes held by live `PageBox`es, always on: one add per alloc, one sub per free.
///
/// The address-space watch reports it next to the free-space figures, so a
/// 32-bit game's log says how much of the space is ours.
static PAGEBOX_LIVE_BYTES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Bytes currently held by live `PageBox`es.
#[must_use]
pub fn live_bytes() -> u64 {
    PAGEBOX_LIVE_BYTES.load(core::sync::atomic::Ordering::Relaxed)
}

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
/// `ds/sizeclasstable.h`, so a large request rounds up to the next
/// power of two. That rounding is why the uncached band starts well
/// below [`SNMALLOC_LOCAL_CACHE_BYTES`].
///
/// Only meaningful above the small-sizeclass ceiling; below it snmalloc
/// serves size classes from slabs. That ceiling is 64 KiB on every supported
/// target, so callers that depend on the exact value must stay above 64 KiB.
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
/// Monotonic identity for every fresh [`PageBox`] allocation.
///
/// The global allocator can hand a freed allocation's address back to a
/// later one, and a `bytesNoCopy` `MTLBuffer` cached against the address
/// alone would then pair GPU-pinned pages of the dead allocation with CPU
/// writes into the new one. The generation names the allocation, not the
/// address, so identity checks survive address reuse. A pool re-acquire
/// keeps the parked allocation's value: parked pages were never freed, so
/// they stay resident and coherent.
static PAGE_BOX_GENERATION: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub struct PageBox {
    ptr: NonNull<u8>,
    /// Rounded-up page multiple.
    ///
    /// What `len()` reports and what the `MTLBuffer` wraps.
    len: usize,
    /// Original request from the caller. Game's visible buffer length.
    logical_len: usize,
    /// Allocator ownership or a guest lease acknowledged when the last native reader leaves.
    ownership: PageOwnership,
    /// This allocation's [`PAGE_BOX_GENERATION`] stamp.
    generation: u64,
    /// Upload jobs and emitted reads, excluding cached buffer-wrapper owners.
    readers: AtomicU32,
}

/// The runtime responsible for releasing a page allocation.
enum PageOwnership {
    Native(Layout),
    #[cfg(not(windows))]
    Guest {
        completion: LeaseCompletionPtr,
        readers: u64,
    },
}

// SAFETY: The allocation is owned here or pinned by a guest lease until Drop.
// The constructors require externally synchronized access to its bytes, and
// the guest completion cell is atomic. Moving the owner preserves that contract.
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
    /// Panics when the allocation fails. Both profiles build with
    /// `panic = "abort"`, so this aborts either way, but panicking runs
    /// the panic hook first, which dumps the crumb ring
    /// (`std::alloc::handle_alloc_error` would abort straight away with no
    /// trace). Every allocation is infallible this way except two, which go
    /// through [`Self::try_new_uninit`] and [`Self::try_new_zeroed`]: the
    /// system-memory copy a resource gets at creation, so the create can answer
    /// `E_OUTOFMEMORY` as D3D9 does, and a chunk of the texture upload-snapshot
    /// arena, whose refusal leaves the upload reading its staging. Otherwise
    /// there is no recovery to attempt past creation: retained VB/IB bytes are
    /// bounded proactively by the retention cap long before the address space
    /// runs out.
    #[must_use]
    pub fn new_uninit(logical_len: usize) -> Self {
        Self::try_new_uninit(logical_len).expect("PageBox alloc failed")
    }

    /// Same as `new_uninit` but zero-initializes the full padded region.
    ///
    /// Used by VB/IB creation so a first-draw-before-Lock sees defined
    /// bytes. Costs one `bzero` per buffer create, negligible compared
    /// to the alternative of every rename paying for the same zero init.
    ///
    /// # Panics
    ///
    /// Same allocation-failure contract as `new_uninit`.
    #[must_use]
    pub fn new_zeroed(logical_len: usize) -> Self {
        Self::try_new_zeroed(logical_len).expect("PageBox alloc failed")
    }

    /// [`Self::new_uninit`] that answers `None` where that one panics.
    ///
    /// For the two allocations with a fallback: a length whose page-rounded
    /// layout the platform cannot express, or an allocation the allocator
    /// refuses, is `None`, which a resource create answers with
    /// `E_OUTOFMEMORY` and the upload-snapshot arena with an upload from the
    /// staging.
    #[must_use]
    pub fn try_new_uninit(logical_len: usize) -> Option<Self> {
        let (len, layout) = Self::try_layout_for(logical_len)?;
        // SAFETY: `try_layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = NonNull::new(unsafe { alloc::alloc(layout) })?;
        Some(Self::adopt(ptr, len, logical_len, layout))
    }

    /// [`Self::new_zeroed`] that answers `None` where that one panics.
    ///
    /// The same fallback-only contract as [`Self::try_new_uninit`].
    #[must_use]
    pub fn try_new_zeroed(logical_len: usize) -> Option<Self> {
        let (len, layout) = Self::try_layout_for(logical_len)?;
        // SAFETY: `try_layout_for` returns a non-zero-size, page-aligned Layout.
        let ptr = NonNull::new(unsafe { alloc::alloc_zeroed(layout) })?;
        Some(Self::adopt(ptr, len, logical_len, layout))
    }

    /// Wrap a fresh allocation of `layout` and count it.
    fn adopt(ptr: NonNull<u8>, len: usize, logical_len: usize, layout: Layout) -> Self {
        note_alloc(len);
        PAGEBOX_LIVE_BYTES.fetch_add(len as u64, core::sync::atomic::Ordering::Relaxed);
        Self {
            ptr,
            len,
            logical_len,
            ownership: PageOwnership::Native(layout),
            generation: PAGE_BOX_GENERATION.fetch_add(1, core::sync::atomic::Ordering::Relaxed) + 1,
            readers: AtomicU32::new(0),
        }
    }

    /// Whether a queued, replayable or in-flight upload still reads these pages.
    #[must_use]
    pub fn has_readers(&self) -> bool {
        self.reader_count()
            .load(core::sync::atomic::Ordering::Acquire)
            != 0
    }

    /// Borrow pages retained by the guest until this lease is acknowledged.
    ///
    /// Native wrappers may retain the resulting box but cannot recycle its allocation.
    ///
    /// # Safety
    ///
    /// `ptr` names a `len`-byte allocation with `PAGE_SIZE` alignment, `len` is a
    /// nonzero page multiple, and `logical_len <= len`. Every range the consumer
    /// reads must be initialized, as with an owned `new_uninit` allocation;
    /// transferring a lease does not initialize unused pages or padding. The caller keeps the
    /// allocation and the aligned completion cell alive until the completion is
    /// observed with Acquire ordering. The lease grants the same externally
    /// synchronized access as an owned box, and this is its sole native owner.
    /// `readers` names the original box's aligned `AtomicU32` reader count and
    /// stays alive for the same lease. A read guard must already exist before
    /// publication whenever the native consumer will acquire a new read.
    #[must_use]
    #[cfg(not(windows))]
    pub const unsafe fn from_guest_lease(
        ptr: NonNull<u8>,
        len: usize,
        logical_len: usize,
        generation: u64,
        completion: LeaseCompletionPtr,
        readers: u64,
    ) -> Self {
        Self {
            ptr,
            len,
            logical_len,
            ownership: PageOwnership::Guest {
                completion,
                readers,
            },
            generation,
            readers: AtomicU32::new(0),
        }
    }

    /// Whether the allocating runtime can park this box in its recycle pool.
    #[must_use]
    pub const fn is_native_owned(&self) -> bool {
        matches!(self.ownership, PageOwnership::Native(_))
    }

    /// Address of the fixed-width reader count kept alive by a page lease.
    #[must_use]
    pub fn reader_count_ptr(&self) -> u64 {
        core::ptr::from_ref(self.reader_count()) as u64
    }

    const fn reader_count(&self) -> &AtomicU32 {
        match &self.ownership {
            PageOwnership::Native(_) => &self.readers,
            #[cfg(not(windows))]
            PageOwnership::Guest { readers, .. } => {
                // SAFETY: from_guest_lease pins the original aligned counter until Drop.
                unsafe { &*(*readers as *const AtomicU32) }
            }
        }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The allocation's identity stamp (see `PAGE_BOX_GENERATION`).
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Borrow the full padded region as a slice.
    ///
    /// Matches `Box<[u8]>`'s deref semantic: callers that consume `&[u8]`
    /// see exactly the same byte range.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: The allocator or guest lease keeps these initialized bytes
        // alive through this borrow, under the constructor's access contract.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable counterpart of `as_slice`.
    ///
    /// Caller takes the unique-borrow guarantee from `&mut self`.
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The allocator or guest lease keeps these bytes alive and
        // the constructor's access contract makes this mutable borrow exclusive.
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

    /// The padded length and layout of a `logical_len` request, `None` past what a layout holds.
    fn try_layout_for(logical_len: usize) -> Option<(usize, Layout)> {
        // A zero-length PageBox is still allocated at one page so the
        // returned pointer is non-null and the MTLBuffer wrap doesn't
        // choke on length=0.
        let padded = logical_len
            .max(1)
            .div_ceil(PAGE_SIZE)
            .checked_mul(PAGE_SIZE)?;
        let layout = Layout::from_size_align(padded, PAGE_SIZE).ok()?;
        Some((padded, layout))
    }
}

impl Drop for PageBox {
    fn drop(&mut self) {
        let layout = match &self.ownership {
            PageOwnership::Native(layout) => *layout,
            #[cfg(not(windows))]
            PageOwnership::Guest { completion, .. } => {
                // SAFETY: from_guest_lease keeps this cell alive until this final acknowledgment.
                let completion = unsafe { &*(completion.raw() as *const LeaseCompletion) };
                completion.publish();
                return;
            }
        };
        note_free(self.len);
        PAGEBOX_LIVE_BYTES.fetch_sub(self.len as u64, core::sync::atomic::Ordering::Relaxed);
        // SAFETY: same layout used to alloc; pointer came from that
        // allocator; nothing else owns this allocation.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// A queued, replayable or in-flight read of a `PageBox`.
///
/// The owning `Arc` keeps the allocation alive. Its separate reader count lets
/// a writer distinguish reads from cached wrappers that only keep pages alive.
/// New reads must be acquired before publication to another thread, or from
/// an already-held read, so a writer observing zero cannot race a new reader.
pub struct PageBoxRead {
    backing: Arc<PageBox>,
}

impl PageBoxRead {
    /// Acquire one published read of an allocation.
    ///
    /// # Panics
    ///
    /// Panics if the fixed-width shared reader count is exhausted.
    #[must_use]
    pub fn new(backing: Arc<PageBox>) -> Self {
        // The fixed-width count is shared across PE and native runtimes. Reject
        // exhaustion rather than publishing a wrapped zero to a guest writer.
        backing
            .reader_count()
            .fetch_update(
                core::sync::atomic::Ordering::Relaxed,
                core::sync::atomic::Ordering::Relaxed,
                |count| count.checked_add(1),
            )
            .expect("PageBox reader count exhausted");
        Self { backing }
    }

    #[must_use]
    pub const fn backing(&self) -> &Arc<PageBox> {
        &self.backing
    }
}

impl Drop for PageBoxRead {
    fn drop(&mut self) {
        // Construction increments once and the guard is move-only,
        // so every drop has exactly one outstanding increment to release.
        self.backing
            .reader_count()
            .fetch_sub(1, core::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod tests;
