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
    fn reserve(&mut self, size: usize) -> *mut u8 {
        let aligned = align_up(size, ALIGN);
        if aligned > self.chunk_size {
            return self.reserve_oversized(aligned);
        }
        // Fast path: current chunk has room.
        if !self.small_chunks.is_empty() && self.cursor + aligned <= self.hot_chunk_len() {
            return self.bump_in_current_chunk(aligned);
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

    /// Bump-copy only the `Some` slots of `src` into the arena, in ascending bit-order of `mask`.
    ///
    /// Returns a pointer to the packed array of `popcount(mask)` `T`s
    /// (or `None` when `mask == 0`).
    ///
    /// `mask` must agree with `src`: bit `b` set iff `src[b].is_some()`.
    /// A `debug_assert!` enforces this; in release builds a stale-bit
    /// mismatch would unwind via the `expect` on the slot.
    ///
    /// Designed for per-draw snapshot bumps where only a few of N
    /// possible slots are populated — collapses a `~N × size_of::<T>()`
    /// memcpy to `popcount × size_of::<T>()`.
    ///
    /// # Safety
    ///
    /// The scratch copy is never dropped. Sound only when `T` has no
    /// `Drop` side effects (same contract as `alloc_from`).
    ///
    /// # Panics
    ///
    /// Panics if `mask` has a bit set whose corresponding `src` slot is
    /// `None` — caller violated the agreement invariant.
    pub unsafe fn alloc_packed_by_mask<T, const N: usize>(
        &mut self,
        mask: u32,
        src: &[Option<T>; N],
    ) -> Option<*mut T> {
        debug_assert!(
            N <= 32,
            "alloc_packed_by_mask: mask is u32, supports at most 32 slots"
        );
        let count = mask.count_ones() as usize;
        if count == 0 {
            return None;
        }
        let dst = self.reserve(count * core::mem::size_of::<T>()).cast::<T>();
        let mut remaining = mask;
        let mut out_idx: usize = 0;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            debug_assert!(bit < N, "alloc_packed_by_mask: mask bit beyond slot count");
            let slot = src[bit]
                .as_ref()
                .expect("alloc_packed_by_mask: mask bit set but slot is None");
            // SAFETY: `dst` has `count` consecutive `T` slots reserved
            // and `out_idx` is < count, so `dst.add(out_idx)` is
            // in-bounds.
            let dst_slot = unsafe { dst.add(out_idx) };
            // SAFETY: source (stack/heap) and destination (scratch
            // arena) are disjoint allocations and both cover one `T`.
            unsafe {
                core::ptr::copy_nonoverlapping(core::ptr::from_ref(slot), dst_slot, 1);
            }
            out_idx += 1;
        }
        Some(dst)
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
mod tests {
    use super::*;

    const TEST_CHUNK: usize = 256;

    #[test]
    fn alloc_returns_stable_pointer_across_subsequent_allocs() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        let payloads: Vec<Vec<u8>> = (0u8..5)
            .map(|i| (0u8..32).map(move |b| i * 32 + b).collect())
            .collect();
        let ptrs: Vec<u64> = payloads.iter().map(|p| arena.alloc(p)).collect();

        for _ in 0..64 {
            let filler = [0xABu8; 48];
            arena.alloc(&filler);
        }

        for (ptr, expected) in ptrs.iter().zip(payloads.iter()) {
            // SAFETY: `*ptr` was just returned by `arena.alloc(expected)` and
            // covers `expected.len()` bytes within the arena slab.
            let slice = unsafe { std::slice::from_raw_parts(*ptr as *const u8, expected.len()) };
            assert_eq!(slice, expected.as_slice());
        }
    }

    #[test]
    fn oversized_request_gets_own_chunk_and_preserves_hot_chunk() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        let small_a = arena.alloc(&[1u8; 32]);
        assert_eq!(arena.chunk_count(), 1);

        let huge = vec![0xCDu8; TEST_CHUNK * 3];
        let _big = arena.alloc(&huge);
        assert_eq!(arena.chunk_count(), 2);

        let small_b = arena.alloc(&[2u8; 32]);
        assert_eq!(arena.chunk_count(), 2, "oversized must not reset hot chunk");
        assert_eq!(small_b, small_a + 32, "next small alloc follows small_a");
    }

    #[test]
    fn clear_retains_high_water() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        for _ in 0..20 {
            arena.alloc(&[0u8; 64]);
        }
        let peak = arena.small_chunk_count();
        assert!(peak >= 2);

        arena.clear();
        // High-water mark of small chunks survives clear; bytes_used
        // resets to 0 because the cursor is back at the start of chunk 0.
        assert_eq!(arena.small_chunk_count(), peak);
        assert_eq!(arena.bytes_used(), 0);

        // First allocation after clear lands at the start of chunk 0,
        // not in a new chunk past the peak.
        let post_clear = arena.alloc(&[0u8; 16]);
        let chunk0_start = arena
            .small_chunks
            .first()
            .map(|c| c.as_ptr() as u64)
            .expect("chunk 0 retained");
        assert_eq!(post_clear, chunk0_start);
        assert_eq!(arena.small_chunk_count(), peak);
    }

    /// After clear, the cursor walks forward through retained chunks.
    ///
    /// It does so before any new allocation hits the heap. Validates the
    /// high-water promise: steady-state frames touch the allocator zero
    /// times.
    #[test]
    fn reserve_walks_existing_chunks_after_clear() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        // Fill enough to span 3 chunks: TEST_CHUNK / 64 = 4 slots per
        // chunk; 9 allocs fill 2 chunks and bleed into a third.
        for _ in 0..9 {
            arena.alloc(&[0u8; 64]);
        }
        let peak = arena.small_chunk_count();
        assert!(
            peak >= 3,
            "test setup expects at least 3 chunks, got {peak}"
        );
        let chunk_ptrs: Vec<u64> = arena
            .small_chunks
            .iter()
            .map(|c| c.as_ptr() as u64)
            .collect();

        arena.clear();
        assert_eq!(
            arena.small_chunk_count(),
            peak,
            "clear retains the chunk vec",
        );

        // Fill the same shape again; each chunk's start address should
        // match the pre-clear pointers — no new chunk pushed.
        for i in 0..9 {
            let p = arena.alloc(&[0u8; 64]);
            let chunk_idx = i / 4;
            let slot_in_chunk = (i % 4) as u64 * 64;
            assert_eq!(
                p,
                chunk_ptrs[chunk_idx] + slot_in_chunk,
                "alloc {i} should land in the same chunk slot as pre-clear",
            );
        }
        assert_eq!(
            arena.small_chunk_count(),
            peak,
            "no new chunk allocated when walking retained ones",
        );
    }

    #[test]
    fn alignment_padding_is_16_bytes() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        let a = arena.alloc(&[0u8; 1]);
        let b = arena.alloc(&[0u8; 1]);
        assert_eq!(b - a, ALIGN as u64);
    }

    #[test]
    fn empty_arena_reports_zero() {
        let arena = ScratchArena::new();
        assert_eq!(arena.chunk_count(), 0);
        assert_eq!(arena.capacity_bytes(), 0);
        assert_eq!(arena.bytes_used(), 0);
    }

    #[test]
    fn alloc_uninit_slice_round_trips_when_written() {
        const COUNT: usize = 5;
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        let ptr: *mut [f32; 4] = arena.alloc_uninit_slice::<[f32; 4]>(COUNT);
        // SAFETY: `alloc_uninit_slice` reserved `COUNT` consecutive
        // `[f32; 4]` slots starting at `ptr`; viewing them as a `&mut`
        // slice of `MaybeUninit` lets every write be a safe assignment.
        let slots: &mut [core::mem::MaybeUninit<[f32; 4]>] = unsafe {
            core::slice::from_raw_parts_mut(ptr.cast::<core::mem::MaybeUninit<[f32; 4]>>(), COUNT)
        };
        let payload: [[f32; 4]; COUNT] = [
            [0.0, 0.5, 1.0, -1.0],
            [1.0, 1.5, 1.0, -1.0],
            [2.0, 2.5, 1.0, -1.0],
            [3.0, 3.5, 1.0, -1.0],
            [4.0, 4.5, 1.0, -1.0],
        ];
        for (slot, row) in slots.iter_mut().zip(payload.iter()) {
            *slot = core::mem::MaybeUninit::new(*row);
        }
        // SAFETY: every slot was just initialised; reinterpret as the
        // concrete `[f32; 4]` slice for read-back.
        let read: &[[f32; 4]] = unsafe { core::slice::from_raw_parts(ptr.cast_const(), COUNT) };
        for (row, expected) in read.iter().zip(payload.iter()) {
            for lane in 0..4 {
                assert_eq!(row[lane].to_bits(), expected[lane].to_bits());
            }
        }
    }

    #[test]
    fn alloc_uninit_slice_is_16_byte_aligned() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        arena.alloc(&[0u8; 1]);
        let ptr: *mut [f32; 4] = arena.alloc_uninit_slice::<[f32; 4]>(3);
        assert_eq!(ptr.addr() % ALIGN, 0);
    }

    #[test]
    fn alloc_packed_by_mask_round_trips_sparse_slots() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct Slot {
            a: u32,
            b: u32,
            c: u32,
        }
        let mut src: [Option<Slot>; 16] = [None; 16];
        src[0] = Some(Slot {
            a: 10,
            b: 11,
            c: 12,
        });
        src[3] = Some(Slot {
            a: 30,
            b: 31,
            c: 32,
        });
        src[7] = Some(Slot {
            a: 70,
            b: 71,
            c: 72,
        });
        let mask: u32 = (1 << 0) | (1 << 3) | (1 << 7);

        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        // SAFETY: Slot is Copy with trivial Drop; bytewise scratch copy
        // is sound per `alloc_packed_by_mask`'s contract.
        let ptr =
            unsafe { arena.alloc_packed_by_mask::<Slot, 16>(mask, &src) }.expect("non-empty mask");
        // SAFETY: `alloc_packed_by_mask` reserved `popcount(mask) = 3`
        // consecutive `Slot`s starting at `ptr`.
        let packed: &[Slot] = unsafe { core::slice::from_raw_parts(ptr.cast_const(), 3) };
        assert_eq!(
            packed[0],
            Slot {
                a: 10,
                b: 11,
                c: 12
            }
        );
        assert_eq!(
            packed[1],
            Slot {
                a: 30,
                b: 31,
                c: 32
            }
        );
        assert_eq!(
            packed[2],
            Slot {
                a: 70,
                b: 71,
                c: 72
            }
        );
    }

    #[test]
    fn alloc_packed_by_mask_empty_mask_returns_none() {
        let src: [Option<u32>; 16] = [None; 16];
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        // SAFETY: u32 is trivially Copy.
        let result = unsafe { arena.alloc_packed_by_mask::<u32, 16>(0, &src) };
        assert!(result.is_none());
        assert_eq!(arena.bytes_used(), 0, "empty mask must not bump");
    }

    #[test]
    fn alloc_packed_by_mask_full_mask_copies_every_slot() {
        let src: [Option<u32>; 8] =
            core::array::from_fn(|i| Some(u32::try_from(i).expect("i ≤ 7 fits u32") * 100));
        let mask: u32 = 0xFF;
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        // SAFETY: u32 is trivially Copy.
        let ptr = unsafe { arena.alloc_packed_by_mask::<u32, 8>(mask, &src) }.expect("full mask");
        // SAFETY: reserved 8 consecutive u32s starting at ptr.
        let packed: &[u32] = unsafe { core::slice::from_raw_parts(ptr.cast_const(), 8) };
        for (i, &v) in packed.iter().enumerate() {
            assert_eq!(v, u32::try_from(i).expect("i ≤ 7 fits u32") * 100);
        }
    }

    #[test]
    fn bytes_used_tracks_cursor() {
        let mut arena = ScratchArena::with_chunk_size(TEST_CHUNK);
        arena.alloc(&[0u8; 20]);
        assert_eq!(arena.bytes_used(), ALIGN as u64 * 2);
        arena.alloc(&[0u8; 16]);
        assert_eq!(arena.bytes_used(), ALIGN as u64 * 3);
    }
}
