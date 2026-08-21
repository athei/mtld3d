//! Per-frame bump arena for payloads handed from the API thread to the encoder thread.
//!
//! Payloads cross via `Op` variants in `windows/d3d9/src/encoder.rs`.
//!
//! # Lifetime precondition
//!
//! Every allocation shares the frame's lifetime: the API thread writes
//! before `stamp_and_swap`, the encoder thread reads while running the
//! frame's op stream, and `clear()` frees in bulk at the next
//! `begin_frame`. Pointer stability across that window is load-bearing —
//! the unix submit thread dereferences scratch pointers during
//! `SubmitFrame`. Work that doesn't fit this uniform-lifetime invariant
//! (cmd-buf-spanning ownership, retained Metal handles, etc.) goes
//! through `Op::Closure(Box<dyn FnOnce>)` instead.
//!
//! # Why chunked instead of flat
//!
//! A flat `Vec<u8>` / `Box<[u8]>` reallocates on overflow and
//! invalidates every pointer handed out earlier in the frame — UB once
//! the encoder dereferences. The only flat alternatives preserve
//! pointer stability either by pre-allocating to a known upper bound
//! (impossible without one) or by reserving virtual address space and
//! committing pages on demand (real but platform-specific and overkill
//! at this scale). Chunked storage sidesteps both: each chunk is its
//! own immovable heap block, growth = `Vec::push(new_chunk)`, and
//! existing pointers stay valid because their chunk wasn't touched.
//!
//! # Why arena over `Box<T>`
//!
//! Fragmentation is not the concern (snmalloc handles it). `Op` enum
//! size is not the concern either — both `Box<T>` and an arena pointer
//! are 8 B inline, so neither inflates the variant. The real concern is
//! *allocator-call frequency*: ~1800 scratch allocations per frame on
//! the per-draw path. snmalloc's thread-local fast path is ~30-50 ns;
//! bump is ~5-10 ns. Per call the difference is small, but at this
//! call count the gap compounds into ~45 µs/frame (~45 ns/draw at
//! ~1000 draws/frame) — matching the measured win when the arena
//! shipped.
//!
//! # High-water retention
//!
//! `clear()` keeps the small-chunk vec intact and only resets the
//! cursor + current-chunk index, so steady-state frames after warm-up
//! touch the allocator zero times on the small path. RSS impact is
//! bounded by peak-frame demand (the small-chunk vec retains its
//! high-water length forever within a session). See
//! `reserve_walks_existing_chunks_after_clear` for the invariant.

use std::ptr;

pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

const ALIGN: usize = 16;

/// Per-frame scratch arena.
///
/// Small allocations bump-pack into default-sized chunks; when the current
/// small chunk is full, the cursor walks forward to the next retained
/// chunk (or appends a new one if past the high-water mark). Requests
/// larger than `chunk_size` go to `oversized`, dedicated chunks sized to
/// the exact request, so they never displace the hot cursor.
///
/// `clear()` resets the cursor + chunk index without dropping any small
/// chunks. After warm-up, steady-state frames touch the allocator zero
/// times on the small path; `oversized` (if ever used) is dropped each
/// frame because per-chunk-per-request can't be re-used as bump space.
pub struct ScratchArena {
    small_chunks: Vec<Box<[u8]>>,
    oversized: Vec<Box<[u8]>>,
    /// Index of the small chunk the cursor is currently inside.
    ///
    /// `clear()` resets this to 0 without dropping any chunks, so
    /// subsequent frames re-fill `small_chunks` from the start.
    /// `reserve()` advances it (and allocates a new chunk only when it
    /// walks past the end) — the high-water `Vec` length is retained
    /// across frames.
    current_chunk_idx: usize,
    cursor: usize,
    chunk_size: usize,
}

impl ScratchArena {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    #[must_use]
    pub const fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            small_chunks: Vec::new(),
            oversized: Vec::new(),
            current_chunk_idx: 0,
            cursor: 0,
            chunk_size,
        }
    }

    /// Reserve `size` bytes in the arena and return an uninitialised pointer.
    ///
    /// Underpins both `alloc` (which then memcpys data in) and
    /// `alloc_uninit` (which lets the caller write via raw ptr).
    ///
    /// The in-chunk bump is a compare and an add; keeping it inline (and
    /// the refill/oversized tail outlined `#[cold]`) is what lets the
    /// per-draw snapshot bumps avoid a call per allocation. Unsplit, the
    /// body is large enough that LLVM declines to inline it even under
    /// fat LTO, and every bump then pays a full call.
    #[inline]
    fn reserve(&mut self, size: usize) -> *mut u8 {
        let aligned = align_up(size, ALIGN);
        // Fast path: current chunk has room.
        if aligned <= self.chunk_size
            && !self.small_chunks.is_empty()
            && self.cursor + aligned <= self.hot_chunk_len()
        {
            return self.bump_in_current_chunk(aligned);
        }
        self.reserve_slow(aligned)
    }

    /// Refill tail of [`Self::reserve`]: oversized requests, cold arena, full chunk.
    ///
    /// Outlined and `#[cold]` so the chunk-allocation machinery does not
    /// count against the inline cost of the fast path above.
    #[cold]
    #[inline(never)]
    fn reserve_slow(&mut self, aligned: usize) -> *mut u8 {
        if aligned > self.chunk_size {
            return self.reserve_oversized(aligned);
        }
        // Cursor doesn't fit (or arena is cold). Walk forward to the next
        // retained chunk if there is one; otherwise grow the vec.
        if self.small_chunks.is_empty() {
            self.small_chunks
                .push(alloc_zeroed_chunk(self.chunk_size).into_boxed_slice());
            self.current_chunk_idx = 0;
        } else {
            self.current_chunk_idx += 1;
            if self.current_chunk_idx >= self.small_chunks.len() {
                self.small_chunks
                    .push(alloc_zeroed_chunk(self.chunk_size).into_boxed_slice());
            }
        }
        self.cursor = 0;
        self.bump_in_current_chunk(aligned)
    }

    #[inline]
    fn bump_in_current_chunk(&mut self, aligned: usize) -> *mut u8 {
        let chunk = &mut self.small_chunks[self.current_chunk_idx];
        // SAFETY: caller (`reserve`) verified `self.cursor + aligned`
        // fits within `chunk.len()`.
        let ptr = unsafe { chunk.as_mut_ptr().add(self.cursor) };
        self.cursor += aligned;
        ptr
    }

    fn reserve_oversized(&mut self, aligned: usize) -> *mut u8 {
        let mut chunk = alloc_zeroed_chunk(aligned).into_boxed_slice();
        let ptr = chunk.as_mut_ptr();
        self.oversized.push(chunk);
        ptr
    }

    /// Copy `data` into the arena and return a stable pointer cast to `u64`.
    ///
    /// Pointer validity ends at the next `clear()`.
    #[inline]
    pub fn alloc(&mut self, data: &[u8]) -> u64 {
        let ptr = self.reserve(data.len());
        // SAFETY: `reserve` returned `data.len()`-bytes-aligned-up space;
        // `data` and the chunk are disjoint allocations.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        ptr as u64
    }

    /// Bump-allocate uninitialised space for one `T` and return a raw pointer.
    ///
    /// Caller writes the value via `ptr::write` or per-field
    /// `addr_of_mut!(...).write(...)` — useful when avoiding a stack
    /// temp that would otherwise be memcpy'd in via `alloc_value`.
    ///
    /// The returned pointer is aligned to the arena's `ALIGN` (16 B),
    /// which exceeds any primitive's alignment requirement.
    pub fn alloc_uninit<T>(&mut self) -> *mut T {
        self.reserve(core::mem::size_of::<T>()).cast::<T>()
    }

    /// Bump-allocate uninitialised space for `count` `T`s and return a raw pointer.
    ///
    /// Caller must initialise every element before any read; arena
    /// chunks are zero-init on creation but reused regions carry stale
    /// bytes.
    ///
    /// # Panics
    ///
    /// Panics if `count * size_of::<T>()` overflows `usize`.
    pub fn alloc_uninit_slice<T>(&mut self, count: usize) -> *mut T {
        let bytes = count
            .checked_mul(core::mem::size_of::<T>())
            .expect("scratch alloc_uninit_slice: byte length overflow");
        self.reserve(bytes).cast::<T>()
    }

    /// Memcpy the bytes of `*value` into the arena and return a typed pointer.
    ///
    /// Like `alloc_value` but takes a reference, so works for non-Copy
    /// types.
    ///
    /// # Safety
    ///
    /// The scratch copy is never dropped, so this is sound only when
    /// `T` has no Drop with side effects (e.g. owns no heap memory
    /// the original `*value` will also drop). Bit-identical duplicate
    /// would-be owners of a `Vec` / `Box` / refcount would silently
    /// leak or alias.
    pub unsafe fn alloc_from<T>(&mut self, value: &T) -> *mut T {
        // SAFETY: bytewise view of any T is sound. Caller covers Drop
        // soundness per the contract above.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref::<T>(value).cast::<u8>(),
                core::mem::size_of::<T>(),
            )
        };
        self.alloc(bytes) as *mut T
    }

    /// Bump-copy a single `T` into the arena and return a typed pointer.
    ///
    /// The arena's `ALIGN` (16 bytes) is ≥ any primitive's alignment, so
    /// `T: Copy` with native primitive fields is safe. Caller asserts `T`
    /// has no padding-sensitive invariants.
    pub fn alloc_value<T: Copy>(&mut self, value: T) -> *mut T {
        // SAFETY: T is Copy, so a byte-level view is sound. The
        // returned pointer is aligned to ALIGN (16), which exceeds any
        // primitive alignment requirement.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                core::ptr::from_ref::<T>(&value).cast::<u8>(),
                core::mem::size_of::<T>(),
            )
        };
        self.alloc(bytes) as *mut T
    }

    /// Bump-copy a slice of `T` into the arena and return a typed pointer + length.
    ///
    /// Same alignment notes as `alloc_value`.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len()` exceeds `u32::MAX` — unreachable in any
    /// realistic per-frame workload.
    pub fn alloc_slice<T: Copy>(&mut self, slice: &[T]) -> (*mut T, u32) {
        // SAFETY: T is Copy and slice is `&[T]`; bytewise view is sound.
        let bytes = unsafe {
            core::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), core::mem::size_of_val(slice))
        };
        let ptr = self.alloc(bytes) as *mut T;
        let len = u32::try_from(slice.len()).expect("scratch alloc_slice: len fits u32");
        (ptr, len)
    }

    /// Reset the bump cursor to the start of `small_chunks[0]` without dropping any small chunks.
    ///
    /// The high-water-mark `Vec` length is retained across frames. Once
    /// the workload stabilises, subsequent frames touch the allocator
    /// zero times on the small path.
    ///
    /// `oversized` chunks are sized to a one-shot request and can't be
    /// reused as bump space (each is fully consumed by a single
    /// allocation), so they are dropped — keeping them would require a
    /// free-list, not a bump arena. In d3d9 the oversized path is
    /// effectively unused (max scratch payload ~4 KB ≪ 64 KB chunk).
    pub fn clear(&mut self) {
        self.current_chunk_idx = 0;
        self.cursor = 0;
        self.oversized.clear();
    }

    /// Total chunk count across small + oversized arenas. Diagnostic only.
    ///
    /// # Panics
    ///
    /// Panics if the total exceeds `u32::MAX` — unreachable, the arena would
    /// have run out of address space first.
    #[must_use]
    pub fn chunk_count(&self) -> u32 {
        u32::try_from(self.small_chunks.len() + self.oversized.len())
            .expect("chunk count ≤ u32::MAX in any realistic workload")
    }

    /// Count of bump-packed chunks.
    ///
    /// Subset of `chunk_count()` that excludes oversized one-shot chunks;
    /// surfaces separately in the perf diag so a reader can tell whether
    /// the arena's footprint is dominated by reusable bump space (small)
    /// or by request-sized one-offs (oversized).
    ///
    /// # Panics
    ///
    /// Panics if the count exceeds `u32::MAX` — see `chunk_count`.
    #[must_use]
    pub fn small_chunk_count(&self) -> u32 {
        u32::try_from(self.small_chunks.len())
            .expect("small chunk count ≤ u32::MAX in any realistic workload")
    }

    /// Count of dedicated chunks allocated for requests larger than `chunk_size`.
    ///
    /// In d3d9 this is normally 0 (max scratch payload is ~4 KB, well
    /// under the 64 KB chunk size) — a non-zero value here is the signal
    /// that some payload is overflowing and motivating its own chunk
    /// every frame.
    ///
    /// # Panics
    ///
    /// Panics if the count exceeds `u32::MAX` — see `chunk_count`.
    #[must_use]
    pub fn oversized_chunk_count(&self) -> u32 {
        u32::try_from(self.oversized.len())
            .expect("oversized chunk count ≤ u32::MAX in any realistic workload")
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        let small: u64 = self.small_chunks.iter().map(|c| c.len() as u64).sum();
        let over: u64 = self.oversized.iter().map(|c| c.len() as u64).sum();
        small + over
    }

    /// Bytes actually written this frame.
    ///
    /// Every chunk before `current_chunk_idx` is fully consumed, plus
    /// `cursor` worth of the current chunk, plus oversized chunks (which
    /// are always fully used — each is sized to its one request).
    /// Retained chunks past the cursor are excluded; they are reserved
    /// capacity, not live use.
    #[must_use]
    pub fn bytes_used(&self) -> u64 {
        let small_full: u64 = if self.small_chunks.is_empty() {
            0
        } else {
            self.small_chunks[..self.current_chunk_idx]
                .iter()
                .map(|c| c.len() as u64)
                .sum::<u64>()
                + self.cursor as u64
        };
        let over: u64 = self.oversized.iter().map(|c| c.len() as u64).sum();
        small_full + over
    }

    #[inline]
    fn hot_chunk_len(&self) -> usize {
        self.small_chunks
            .get(self.current_chunk_idx)
            .map_or(0, |c| c.len())
    }
}

impl Default for ScratchArena {
    fn default() -> Self {
        Self::new()
    }
}

const fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

fn alloc_zeroed_chunk(size: usize) -> Vec<u8> {
    vec![0u8; size]
}

#[cfg(test)]
mod tests;
