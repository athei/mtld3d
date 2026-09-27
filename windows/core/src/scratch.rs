//! Per-frame bump arena for payloads handed from the API thread to the encoder thread.
//!
//! Captured commands and payloads share one arena owned by the PE frame packet.
//!
//! # Lifetime precondition
//!
//! The API thread completes command headers and regions before admission seals
//! the packet. Native replay and submit retain the storage lease while any
//! borrowed payload may still be read. Only a successful replay acknowledgment
//! permits arena reuse; rejected or uncertain submissions remain quarantined
//! until the runtime has quiesced. `clear` resets both logical cursors before
//! the next frame records into this storage.
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
//! `clear()` keeps standard-size chunks in the shared pool and resets both
//! cursors plus the next unused chunk index, so steady-state frames after warm-up
//! touch the allocator zero times on the small path. RSS impact is
//! bounded by peak-frame demand (the small-chunk vec retains its
//! high-water length forever within a session). See
//! `reserve_walks_existing_chunks_after_clear` for the invariant.

use std::ptr;

pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

const ALIGN: usize = 16;

/// One frame owner for stable command and external payload allocations.
///
/// A shared reusable chunk pool supplies two independent bump cursors. Payload
/// capture never interrupts a flat command region. Clear is permitted only after
/// the packet's replay lease ends; it resets both cursors and drops oversized
/// chunks while preserving the standard-size high-water capacity.
pub struct ScratchArena {
    #[cfg(perf_tracking)]
    perf_address: u64,
    #[cfg(not(perf_tracking))]
    perf: crate::perf::FramePerfPayload,
    chunks: Vec<AlignedChunk>,
    next_chunk: usize,
    payload_chunk: Option<usize>,
    command_chunk: Option<usize>,
    chunk_size: usize,
}

/// A completed command and the contiguous region that now contains it.
pub struct CommandAllocation {
    pub address: u64,
    pub record_bytes: usize,
    pub region_address: u64,
    pub region_bytes: usize,
}

impl ScratchArena {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    #[must_use]
    pub const fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            #[cfg(perf_tracking)]
            perf_address: 0,
            #[cfg(not(perf_tracking))]
            perf: crate::perf::FramePerfPayload::new(),
            chunks: Vec::new(),
            next_chunk: 0,
            payload_chunk: None,
            command_chunk: None,
            chunk_size,
        }
    }

    /// Borrow this arena's source-clock telemetry, or immutable zero telemetry before first use.
    #[must_use]
    pub const fn perf(&self) -> &crate::perf::FramePerfPayload {
        #[cfg(perf_tracking)]
        {
            if self.perf_address == 0 {
                const EMPTY: crate::perf::FramePerfPayload = crate::perf::FramePerfPayload::new();
                return &EMPTY;
            }
            // SAFETY: perf_mut initializes this arena slot; clear resets the token before reuse.
            unsafe { &*(self.perf_address as *const crate::perf::FramePerfPayload) }
        }
        #[cfg(not(perf_tracking))]
        {
            &self.perf
        }
    }

    /// Initialize telemetry directly in its final arena slot on first use.
    #[cfg(perf_tracking)]
    pub fn perf_mut(&mut self) -> &mut crate::perf::FramePerfPayload {
        if self.perf_address == 0 {
            let payload = self.alloc_uninit::<crate::perf::FramePerfPayload>();
            // SAFETY: the fresh aligned slot is exclusive and contains a destructor-free value.
            unsafe {
                payload.write(crate::perf::FramePerfPayload::new());
            }
            self.perf_address = payload as u64;
        }
        // SAFETY: the exclusive arena borrow excludes clear, replacement and another payload borrow.
        unsafe { &mut *(self.perf_address as *mut crate::perf::FramePerfPayload) }
    }

    /// Borrow the zero-sized local payload when telemetry is disabled.
    #[cfg(not(perf_tracking))]
    pub const fn perf_mut(&mut self) -> &mut crate::perf::FramePerfPayload {
        &mut self.perf
    }

    /// Allocation ranges retained by the single arena owner through frame replay.
    pub fn allocation_ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.chunks
            .iter()
            .map(|chunk| (chunk.as_ptr() as u64, chunk.len() as u64))
    }

    #[inline]
    fn reserve(&mut self, size: usize) -> *mut u8 {
        let aligned = align_up(size, ALIGN);
        if let Some(index) = self.payload_chunk {
            let chunk = &mut self.chunks[index];
            if aligned <= chunk.len() - chunk.used {
                let pointer = chunk.as_mut_ptr().wrapping_add(chunk.used);
                chunk.used += aligned;
                return pointer;
            }
        }
        self.reserve_slow(aligned)
    }

    #[cold]
    #[inline(never)]
    fn reserve_slow(&mut self, aligned: usize) -> *mut u8 {
        let index = self.acquire_chunk(aligned);
        if aligned <= self.chunk_size {
            self.payload_chunk = Some(index);
        }
        let chunk = &mut self.chunks[index];
        chunk.used = aligned;
        chunk.as_mut_ptr()
    }

    #[cold]
    fn acquire_chunk(&mut self, required: usize) -> usize {
        let index = self.next_chunk;
        let capacity = required.max(self.chunk_size);
        if index == self.chunks.len() {
            self.chunks.push(alloc_zeroed_chunk(capacity));
        } else if self.chunks[index].len() < capacity {
            self.chunks[index] = alloc_zeroed_chunk(capacity);
        }
        self.chunks[index].used = 0;
        self.next_chunk += 1;
        index
    }

    /// Initialize a command in a contiguous command region of this arena.
    ///
    /// Payload allocations use a separate cursor into the same owned chunk pool.
    /// A failed callback leaves the committed region length unchanged. Successful
    /// records contain an exact logical length and zero alignment padding.
    ///
    /// # Errors
    /// Returns overflow, invalid callback length or the callback's error.
    pub fn write_command(
        &mut self,
        opcode: u16,
        operand: u16,
        payload_bound: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<usize, mtld3d_shared::encoder_wire::WireError>,
    ) -> Result<CommandAllocation, mtld3d_shared::encoder_wire::WireError> {
        use mtld3d_shared::{
            command_header::{COMMAND_HEADER_BYTES, CommandHeader},
            encoder_wire::WireError,
        };
        let bound = payload_bound
            .checked_add(COMMAND_HEADER_BYTES)
            .ok_or(WireError::TooLarge)?;
        if bound > u32::MAX as usize || bound > isize::MAX as usize {
            return Err(WireError::TooLarge);
        }
        let reserved = bound.checked_add(15).ok_or(WireError::TooLarge)? & !15;
        let index = if let Some(index) = self.command_chunk
            && reserved <= self.chunks[index].len() - self.chunks[index].used
        {
            index
        } else {
            let index = self.acquire_chunk(reserved);
            self.command_chunk = Some(index);
            index
        };
        let chunk = &mut self.chunks[index];
        let pointer = chunk.as_mut_ptr().wrapping_add(chunk.used);
        // SAFETY: the exclusive aligned reservation contains this payload window.
        let destination = unsafe {
            core::slice::from_raw_parts_mut(
                pointer.wrapping_add(COMMAND_HEADER_BYTES),
                payload_bound,
            )
        };
        let payload_used = fill(destination)?;
        if payload_used > payload_bound {
            return Err(WireError::TooLarge);
        }
        let used = payload_used + COMMAND_HEADER_BYTES;
        let aligned_used = (used + 15) & !15;
        let padding = pointer.wrapping_add(used);
        // Fixed command fields are eight-byte aligned. Keep their usual padding
        // stores inline instead of calling memset for zero or eight bytes.
        match aligned_used - used {
            0 => {}
            8 => {
                // SAFETY: exactly eight padding bytes lie within this exclusive reservation.
                unsafe { padding.write_bytes(0, 8) };
            }
            length => {
                // SAFETY: all remaining padding bytes lie within this exclusive reservation.
                unsafe { padding.write_bytes(0, length) };
            }
        }
        let record_bytes = u32::try_from(used).map_err(|_| WireError::TooLarge)?;
        let header_pointer = pointer as usize as *mut CommandHeader;
        // SAFETY: aligned chunk storage and cursor establish header alignment;
        // this exclusive reservation holds a complete initialized command.
        unsafe {
            header_pointer.write(CommandHeader {
                opcode,
                operand,
                record_bytes,
                reserved: 0,
            });
        };
        chunk.used += aligned_used;
        Ok(CommandAllocation {
            address: pointer as u64,
            record_bytes: used,
            region_address: chunk.as_ptr() as u64,
            region_bytes: chunk.used,
        })
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

    /// Reuse the single retained pool after all frame readers have released it.
    ///
    /// Oversized regions are discarded; both logical cursors start unassigned.
    pub fn clear(&mut self) {
        #[cfg(perf_tracking)]
        {
            self.perf_address = 0;
        }
        self.chunks.retain(|chunk| chunk.len() <= self.chunk_size);
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
        self.next_chunk = 0;
        self.payload_chunk = None;
        self.command_chunk = None;
    }

    /// Number of owned chunks, including oversized allocations.
    ///
    /// # Panics
    /// Panics if the chunk count cannot fit u32.
    #[must_use]
    pub fn chunk_count(&self) -> u32 {
        u32::try_from(self.chunks.len()).expect("chunk count fits u32")
    }

    /// Number of reusable chunks within the standard chunk-size budget.
    ///
    /// # Panics
    /// Panics if the chunk count cannot fit u32.
    #[must_use]
    pub fn small_chunk_count(&self) -> u32 {
        u32::try_from(
            self.chunks
                .iter()
                .filter(|chunk| chunk.len() <= self.chunk_size)
                .count(),
        )
        .expect("chunk count fits u32")
    }

    /// Number of oversized chunks that will be dropped on clear.
    ///
    /// # Panics
    /// Panics if the chunk count cannot fit u32.
    #[must_use]
    pub fn oversized_chunk_count(&self) -> u32 {
        u32::try_from(
            self.chunks
                .iter()
                .filter(|chunk| chunk.len() > self.chunk_size)
                .count(),
        )
        .expect("chunk count fits u32")
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.len() as u64).sum()
    }

    /// Sum of committed or allocated bytes in both logical cursors this frame.
    #[must_use]
    pub fn bytes_used(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.used as u64).sum()
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

#[repr(C, align(16))]
struct ArenaWord {
    _bytes: [u8; 16],
}

struct AlignedChunk {
    words: Box<[ArenaWord]>,
    length: usize,
    used: usize,
}

impl AlignedChunk {
    fn as_ptr(&self) -> *const u8 {
        self.words.as_ptr().cast::<u8>()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.words.as_mut_ptr().cast::<u8>()
    }
    const fn len(&self) -> usize {
        self.length
    }
}

fn alloc_zeroed_chunk(size: usize) -> AlignedChunk {
    let words = std::iter::repeat_with(|| ArenaWord { _bytes: [0; 16] })
        .take(size.div_ceil(16).max(1))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    AlignedChunk {
        words,
        length: size,
        used: 0,
    }
}

#[cfg(test)]
mod tests;
