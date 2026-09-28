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
    /// First byte of the open command region, as an exposed address; zero when none is open.
    ///
    /// While a region is open its chunk's `used` stays zero and the three addresses below
    /// describe it: `command_base <= command_cursor <= command_limit`, all within the chunk.
    /// Closing the region stores `command_cursor - command_base` into the chunk.
    command_base: usize,
    /// Address at which the next command in the open region begins.
    command_cursor: usize,
    /// Address one past the last byte of the open region.
    command_limit: usize,
    /// One descriptor per command region, in record order.
    ///
    /// `open_command_region` is the only way a region opens, and it pushes the descriptor
    /// before any command is written there, so every command lands in a region the table
    /// names. The open region's length lives in the cursor; `publish_command_region` copies it
    /// into the last descriptor, which the region's closing and the frame's sealing both do.
    /// A region stays empty only when the record that opened it failed, and that failure is
    /// latched, so its frame is rejected before replay.
    command_regions: Vec<mtld3d_shared::command_header::CommandRegion>,
    chunk_size: usize,
}

/// A command header immediately followed by its fixed payload.
#[repr(C)]
struct FramedCommand<T> {
    header: mtld3d_shared::command_header::CommandHeader,
    payload: T,
}

/// Room for one fixed command at the open region's cursor, committed by `write`.
pub struct CommandSlot<'a, T> {
    arena: &'a mut ScratchArena,
    cursor: usize,
    record_bytes: u32,
    payload: core::marker::PhantomData<T>,
}

impl<T: crate::encoder_records::CommandRecord> CommandSlot<'_, T> {
    /// Write the header and payload in place and advance the region past them.
    #[inline]
    pub const fn write(self, opcode: u16, operand: u16, payload: T) {
        let command = FramedCommand {
            header: mtld3d_shared::command_header::CommandHeader {
                opcode,
                operand,
                record_bytes: self.record_bytes,
            },
            payload,
        };
        // SAFETY: `command_slot` found the whole command inside the open region, an exclusive
        // range of one exposed chunk allocation, at a command-aligned cursor.
        unsafe { ptr::with_exposed_provenance_mut::<FramedCommand<T>>(self.cursor).write(command) };
        self.arena.command_cursor = self.cursor + size_of::<FramedCommand<T>>();
    }
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
            command_base: 0,
            command_cursor: 0,
            command_limit: 0,
            command_regions: Vec::new(),
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

    /// Bytes a command with a `payload_bound`-byte payload reserves, header and padding included.
    const fn command_reservation(
        payload_bound: usize,
    ) -> Result<usize, mtld3d_shared::encoder_wire::WireError> {
        use mtld3d_shared::{
            command_header::{COMMAND_ALIGNMENT, COMMAND_HEADER_BYTES},
            encoder_wire::WireError,
        };
        let Some(bound) = payload_bound.checked_add(COMMAND_HEADER_BYTES) else {
            return Err(WireError::TooLarge);
        };
        if bound > u32::MAX as usize || bound > isize::MAX as usize {
            return Err(WireError::TooLarge);
        }
        let Some(padded) = bound.checked_add(COMMAND_ALIGNMENT - 1) else {
            return Err(WireError::TooLarge);
        };
        Ok(padded & !(COMMAND_ALIGNMENT - 1))
    }

    /// Whether the open command region can take a `reserved`-byte command.
    const fn has_command_room(&self, reserved: usize) -> bool {
        self.command_limit - self.command_cursor >= reserved
    }

    /// Bytes committed to the open command region, zero when none is open.
    const fn open_command_bytes(&self) -> usize {
        self.command_cursor - self.command_base
    }

    /// Store the open region's committed length into its descriptor.
    ///
    /// # Errors
    /// Returns `TooLarge` if the region length does not fit its descriptor.
    pub fn publish_command_region(&mut self) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        let used = u32::try_from(self.open_command_bytes())
            .map_err(|_| mtld3d_shared::encoder_wire::WireError::TooLarge)?;
        if let Some(region) = self.command_regions.last_mut() {
            region.used_bytes = used;
        }
        Ok(())
    }

    /// Borrow the published region table as its fixed-width descriptor bytes.
    #[must_use]
    pub const fn command_descriptor_bytes(&self) -> &[u8] {
        // SAFETY: CommandRegion has no padding, every scalar field is initialized,
        // and this borrow prevents changes to the descriptor vector.
        unsafe {
            core::slice::from_raw_parts(
                self.command_regions.as_ptr().cast::<u8>(),
                core::mem::size_of_val(self.command_regions.as_slice()),
            )
        }
    }

    /// Publish the open region, then list every region's address and committed length.
    ///
    /// # Panics
    /// Panics if the open region's length does not fit its descriptor.
    #[cfg(test)]
    pub fn command_ranges(&mut self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.publish_command_region()
            .expect("test regions fit their descriptors");
        self.command_regions
            .iter()
            .map(|region| (region.address, u64::from(region.used_bytes)))
    }

    /// Close the open command region and open one with room for `reserved` bytes.
    ///
    /// The new region's descriptor is pushed before any command is written there. The closed
    /// region keeps its committed length in its chunk and its descriptor.
    ///
    /// # Errors
    /// Returns an allocation failure before the arena changes, or a closed region's length error.
    #[cold]
    #[inline(never)]
    fn open_command_region(
        &mut self,
        reserved: usize,
    ) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        self.command_regions
            .try_reserve(1)
            .map_err(|_| mtld3d_shared::encoder_wire::WireError::AllocationFailed)?;
        self.publish_command_region()?;
        if let Some(index) = self.command_chunk {
            self.chunks[index].used = self.open_command_bytes();
        }
        let index = self.acquire_chunk(reserved);
        self.command_chunk = Some(index);
        let chunk = &mut self.chunks[index];
        let base = chunk.as_mut_ptr().expose_provenance();
        self.command_base = base;
        self.command_cursor = base;
        self.command_limit = base + chunk.len();
        self.command_regions
            .push(mtld3d_shared::command_header::CommandRegion {
                address: base as u64,
                used_bytes: 0,
                reserved: 0,
            });
        Ok(())
    }

    /// Claim room for one complete fixed command in the open region.
    ///
    /// Returns `None` when no region is open or the command does not fit. The payload size is
    /// a multiple of the command alignment, so no padding follows it. Nothing is committed
    /// until the slot is written.
    #[inline]
    pub fn command_slot<T: crate::encoder_records::CommandRecord>(
        &mut self,
    ) -> Option<CommandSlot<'_, T>> {
        use mtld3d_shared::command_header::{COMMAND_ALIGNMENT, COMMAND_HEADER_BYTES};
        const {
            assert!(align_of::<T>() <= COMMAND_ALIGNMENT);
            assert!(size_of::<T>().is_multiple_of(COMMAND_ALIGNMENT));
            assert!(size_of::<FramedCommand<T>>() == COMMAND_HEADER_BYTES + size_of::<T>());
            assert!(size_of::<FramedCommand<T>>() <= u32::MAX as usize);
        }
        let record_bytes = u32::try_from(size_of::<FramedCommand<T>>()).ok()?;
        let cursor = self.command_cursor;
        if self.command_limit - cursor < size_of::<FramedCommand<T>>() {
            return None;
        }
        Some(CommandSlot {
            arena: self,
            cursor,
            record_bytes,
            payload: core::marker::PhantomData,
        })
    }

    /// Initialize a command in a contiguous command region of this arena.
    ///
    /// Payload allocations use a separate cursor into the same owned chunk pool.
    /// A command that does not fit opens and names a new region first. A failed callback
    /// leaves the committed region length unchanged. Successful records contain an exact
    /// logical length and zero alignment padding.
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
            command_header::{COMMAND_ALIGNMENT, COMMAND_HEADER_BYTES, CommandHeader},
            encoder_wire::WireError,
        };
        let reserved = Self::command_reservation(payload_bound)?;
        if !self.has_command_room(reserved) {
            self.open_command_region(reserved)?;
        }
        let alignment_mask = COMMAND_ALIGNMENT - 1;
        let pointer = ptr::with_exposed_provenance_mut::<u8>(self.command_cursor);
        // SAFETY: the open region holds this exclusive aligned reservation, which contains
        // the header and this payload window.
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
        let aligned_used = (used + alignment_mask) & !alignment_mask;
        let padding = pointer.wrapping_add(used);
        match aligned_used - used {
            0 => {}
            4 => {
                // SAFETY: these four padding bytes lie within the exclusive reservation.
                unsafe { padding.write_bytes(0, 4) };
            }
            count => {
                // SAFETY: all padding bytes lie within this exclusive reservation.
                unsafe { padding.write_bytes(0, count) };
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
            });
        };
        self.command_cursor += aligned_used;
        Ok(CommandAllocation {
            address: pointer as u64,
            record_bytes: used,
            region_address: self.command_base as u64,
            region_bytes: self.open_command_bytes(),
        })
    }

    /// Initialize a command whose payload fills exactly `payload_bytes`.
    ///
    /// # Errors
    /// Returns the callback's error or a reservation or region failure.
    pub fn push_fixed_record(
        &mut self,
        tag: u16,
        operand: u16,
        payload_bytes: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), mtld3d_shared::encoder_wire::WireError>,
    ) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        self.push_initialized_record(tag, operand, payload_bytes, |destination| {
            fill(destination)?;
            Ok(payload_bytes)
        })
    }

    /// Initialize a command whose callback reports how much of `bound` it used.
    ///
    /// # Errors
    /// Returns the callback's error or a reservation or region failure.
    pub fn push_initialized_record(
        &mut self,
        tag: u16,
        operand: u16,
        bound: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<usize, mtld3d_shared::encoder_wire::WireError>,
    ) -> Result<(), mtld3d_shared::encoder_wire::WireError> {
        self.write_command(tag, operand, bound, fill).map(drop)
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
        self.command_base = 0;
        self.command_cursor = 0;
        self.command_limit = 0;
        self.command_regions.clear();
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
        let committed: u64 = self.chunks.iter().map(|chunk| chunk.used as u64).sum();
        committed + self.open_command_bytes() as u64
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
