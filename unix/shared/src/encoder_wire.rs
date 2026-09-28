//! Fixed-width frame records and completion mailboxes for native encoding.
//!
//! Slabs remain owned by their allocating runtime. A reader borrows immutable
//! bytes until replay consumes them, and never reconstructs a Rust owner.
//! Record headers contain a little-endian `u16` tag and `u32` payload length.
//! Callers report errors with the operation and device context they own.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    Truncated,
    TooLarge,
    AllocationFailed,
    InvalidValue,
}

/// Reusable storage written directly as API operations are captured.
#[derive(Default)]
pub struct FrameSlab {
    bytes: Vec<u8>,
}

impl FrameSlab {
    #[must_use]
    pub const fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Reuse storage only after every native reader has released its lease.
    pub fn clear(&mut self) {
        self.bytes.clear();
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Update the trailing fixed-width length field of a descriptor record.
    ///
    /// # Panics
    /// Panics if the slab has fewer than four bytes.
    pub fn replace_last_u32(&mut self, value: u32) {
        let end = self.bytes.len();
        self.bytes[end - 4..].copy_from_slice(&value.to_le_bytes());
    }

    /// Append one complete operation directly into the slab.
    ///
    /// The callback writes only payload fields. A failure rolls back the entire
    /// record, so earlier operations remain readable and no partial header leaks.
    ///
    /// # Errors
    ///
    /// Returns the callback error, `TooLarge` beyond the `u32` byte limit, or
    /// `AllocationFailed` when the backing allocation cannot grow.
    pub fn push_record(
        &mut self,
        tag: u16,
        write: impl FnOnce(&mut WireWriter<'_>) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        let start = self.bytes.len();
        let result = self.append_record(tag, write);
        if result.is_err() {
            self.bytes.truncate(start);
        }
        result
    }

    fn append_record(
        &mut self,
        tag: u16,
        write: impl FnOnce(&mut WireWriter<'_>) -> Result<(), WireError>,
    ) -> Result<(), WireError> {
        let mut writer = WireWriter {
            storage: WireStorage::Vector(&mut self.bytes),
        };
        writer.u16(tag)?;
        let length_offset = writer.len();
        writer.u32(0)?;
        let payload_start = writer.len();
        write(&mut writer)?;
        let length =
            u32::try_from(writer.len() - payload_start).map_err(|_| WireError::TooLarge)?;
        writer.overwrite(length_offset, &length.to_le_bytes());
        Ok(())
    }
}

enum WireStorage<'a> {
    Vector(&'a mut Vec<u8>),
    Slice { bytes: &'a mut [u8], used: usize },
}

pub struct WireWriter<'a> {
    storage: WireStorage<'a>,
}

impl WireWriter<'_> {
    const fn len(&self) -> usize {
        match &self.storage {
            WireStorage::Vector(bytes) => bytes.len(),
            WireStorage::Slice { used, .. } => *used,
        }
    }

    fn overwrite(&mut self, offset: usize, value: &[u8]) {
        match &mut self.storage {
            WireStorage::Vector(bytes) => {
                bytes[offset..offset + value.len()].copy_from_slice(value);
            }
            WireStorage::Slice { bytes, .. } => {
                bytes[offset..offset + value.len()].copy_from_slice(value);
            }
        }
    }

    /// Write only payload fields into an already reserved command destination.
    ///
    /// # Errors
    /// Returns the callback error or a bounds failure.
    pub fn payload_into(
        destination: &mut [u8],
        write: impl FnOnce(&mut WireWriter<'_>) -> Result<(), WireError>,
    ) -> Result<usize, WireError> {
        let mut writer = WireWriter {
            storage: WireStorage::Slice {
                bytes: destination,
                used: 0,
            },
        };
        write(&mut writer)?;
        Ok(writer.len())
    }

    /// Reserve an initialized byte window for an explicit fixed-layout payload.
    ///
    /// # Errors
    /// Returns an allocation or bounds failure.
    pub fn reserve_bytes(&mut self, length: usize) -> Result<&mut [u8], WireError> {
        match &mut self.storage {
            WireStorage::Vector(bytes) => {
                let start = bytes.len();
                let end = start.checked_add(length).ok_or(WireError::TooLarge)?;
                if end > u32::MAX as usize {
                    return Err(WireError::TooLarge);
                }
                bytes
                    .try_reserve(length)
                    .map_err(|_| WireError::AllocationFailed)?;
                bytes.resize(end, 0);
                Ok(&mut bytes[start..end])
            }
            WireStorage::Slice { bytes, used } => {
                let start = *used;
                let end = start.checked_add(length).ok_or(WireError::TooLarge)?;
                let destination = bytes.get_mut(start..end).ok_or(WireError::TooLarge)?;
                *used = end;
                Ok(destination)
            }
        }
    }

    /// Write a record directly into a reserved arena window without allocating.
    ///
    /// # Errors
    /// Returns the payload error or `TooLarge` if the reserved window is exhausted.
    pub fn record_into(
        destination: &mut [u8],
        tag: u16,
        write: impl FnOnce(&mut WireWriter<'_>) -> Result<(), WireError>,
    ) -> Result<usize, WireError> {
        u32::try_from(destination.len()).map_err(|_| WireError::TooLarge)?;
        let mut writer = WireWriter {
            storage: WireStorage::Slice {
                bytes: destination,
                used: 0,
            },
        };
        writer.u16(tag)?;
        writer.u32(0)?;
        write(&mut writer)?;
        let total = writer.len();
        let payload = u32::try_from(total - 6).map_err(|_| WireError::TooLarge)?;
        writer.overwrite(2, &payload.to_le_bytes());
        Ok(total)
    }

    /// Append a little-endian `u8` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn u8(&mut self, value: u8) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Append a little-endian `u16` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn u16(&mut self, value: u16) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Append a little-endian `u32` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn u32(&mut self, value: u32) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Append a little-endian `u64` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn u64(&mut self, value: u64) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Append a little-endian `i32` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn i32(&mut self, value: i32) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Append a little-endian `f32` field.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    #[inline]
    pub fn f32(&mut self, value: f32) -> Result<(), WireError> {
        self.fixed(value.to_le_bytes())
    }

    /// Keep scalar stores visible to callers without inlining vector growth.
    ///
    /// The reserved window was checked against the wire size limit at construction.
    #[inline]
    fn fixed<const N: usize>(&mut self, value: [u8; N]) -> Result<(), WireError> {
        match &mut self.storage {
            WireStorage::Slice { bytes, used } => {
                let destination = bytes
                    .get_mut(*used..)
                    .and_then(<[u8]>::first_chunk_mut::<N>)
                    .ok_or(WireError::TooLarge)?;
                *destination = value;
                *used += N;
                Ok(())
            }
            WireStorage::Vector(bytes) => Self::append_vector(bytes, &value),
        }
    }

    /// Append bytes without adding a length prefix.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    pub fn bytes(&mut self, value: &[u8]) -> Result<(), WireError> {
        let length = self
            .len()
            .checked_add(value.len())
            .ok_or(WireError::TooLarge)?;
        u32::try_from(length).map_err(|_| WireError::TooLarge)?;
        match &mut self.storage {
            WireStorage::Vector(bytes) => {
                Self::append_vector(bytes, value)?;
            }
            WireStorage::Slice { bytes, used } => {
                bytes
                    .get_mut(*used..length)
                    .ok_or(WireError::TooLarge)?
                    .copy_from_slice(value);
                *used = length;
            }
        }
        Ok(())
    }

    /// Keep fallible vector growth out of each inlined reserved scalar store.
    ///
    /// Production codegen otherwise duplicates allocation paths and leaves some
    /// scalar writes out of line even when every operation uses a reserved slice.
    #[inline(never)]
    fn append_vector(bytes: &mut Vec<u8>, value: &[u8]) -> Result<(), WireError> {
        let length = bytes
            .len()
            .checked_add(value.len())
            .ok_or(WireError::TooLarge)?;
        u32::try_from(length).map_err(|_| WireError::TooLarge)?;
        bytes
            .try_reserve(value.len())
            .map_err(|_| WireError::AllocationFailed)?;
        bytes.extend_from_slice(value);
        Ok(())
    }
}

pub struct WireRecord<'a> {
    pub tag: u16,
    pub payload: WireReader<'a>,
}

pub struct WireReader<'a> {
    remaining: &'a [u8],
    trusted_addresses: bool,
    retained_ranges: Option<&'a [(u64, u64)]>,
}

impl<'a> WireReader<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            remaining: bytes,
            trusted_addresses: false,
            retained_ranges: None,
        }
    }

    /// Read a frame whose address fields are covered by a live producer lease.
    ///
    /// # Safety
    ///
    /// `bytes` must come from the paired in-process encoder using the current
    /// schema. Every nonzero address must identify the expected object kind or
    /// allocation with the alignment and extent the field requires. The caller
    /// must keep each object and allocation alive until every decoded consumer
    /// releases it, including deferred replay and GPU use after this reader drops.
    #[must_use]
    pub const unsafe fn new_trusted(bytes: &'a [u8]) -> Self {
        Self {
            remaining: bytes,
            trusted_addresses: true,
            retained_ranges: None,
        }
    }

    /// Read a frame with an explicit inventory of retained byte allocations.
    ///
    /// # Safety
    ///
    /// The object-kind and lifetime requirements of `new_trusted` apply. Each
    /// `(address, length)` range must additionally describe live retained storage.
    /// The inventory must come from the owner, independently of operation fields.
    #[must_use]
    pub const unsafe fn new_trusted_with_ranges(bytes: &'a [u8], ranges: &'a [(u64, u64)]) -> Self {
        Self {
            remaining: bytes,
            trusted_addresses: true,
            retained_ranges: Some(ranges),
        }
    }

    /// Whether a byte extent is fully contained in one retained allocation.
    #[must_use]
    pub fn permits_range(&self, address: u64, length: u64) -> bool {
        if length == 0 {
            return true;
        }
        let Some(end) = address.checked_add(length) else {
            return false;
        };
        if !self.trusted_addresses || address == 0 || length > isize::MAX as u64 {
            return false;
        }
        self.retained_ranges.is_none_or(|ranges| {
            ranges.iter().any(|&(base, size)| {
                base != 0
                    && base <= address
                    && base.checked_add(size).is_some_and(|limit| end <= limit)
            })
        })
    }

    /// Whether the caller established the address and lease contract.
    #[must_use]
    pub const fn has_trusted_addresses(&self) -> bool {
        self.trusted_addresses
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }

    #[must_use]
    pub const fn remaining_len(&self) -> usize {
        self.remaining.len()
    }

    /// Consume the next complete record, borrowing its payload.
    ///
    /// Empty input is the normal end of the stream. On malformed input the
    /// reader stays at the failed record, so failure cannot hide trailing bytes.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` for an incomplete header or payload.
    pub fn next_record(&mut self) -> Result<Option<WireRecord<'a>>, WireError> {
        if self.is_empty() {
            // Exhaustion is a normal stream terminator, not a dropped record.
            return Ok(None);
        }
        let mut cursor = Self {
            remaining: self.remaining,
            trusted_addresses: self.trusted_addresses,
            retained_ranges: self.retained_ranges,
        };
        let tag = cursor.u16()?;
        let length = cursor.u32()?;
        let payload = Self {
            remaining: cursor.bytes(length)?,
            trusted_addresses: self.trusted_addresses,
            retained_ranges: self.retained_ranges,
        };
        self.remaining = cursor.remaining;
        Ok(Some(WireRecord { tag, payload }))
    }

    /// Read a little-endian `u8` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn u8(&mut self) -> Result<u8, WireError> {
        Ok(u8::from_le_bytes(self.array()?))
    }

    /// Read a little-endian `u16` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    /// Read a little-endian `u32` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    /// Read a little-endian `u64` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// Read a little-endian `i32` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn i32(&mut self) -> Result<i32, WireError> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    /// Read a little-endian `f32` field.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the field is incomplete.
    #[inline]
    pub fn f32(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    /// Borrow a byte range without crossing the current payload's bounds.
    ///
    /// # Errors
    ///
    /// Returns `Truncated` without advancing when the range is incomplete.
    pub fn bytes(&mut self, length: u32) -> Result<&'a [u8], WireError> {
        let length = usize::try_from(length).map_err(|_| WireError::TooLarge)?;
        let (result, remaining) = self
            .remaining
            .split_at_checked(length)
            .ok_or(WireError::Truncated)?;
        self.remaining = remaining;
        Ok(result)
    }

    #[inline]
    fn array<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let (result, remaining) = self
            .remaining
            .split_first_chunk::<N>()
            .ok_or(WireError::Truncated)?;
        self.remaining = remaining;
        Ok(*result)
    }
}

/// One allocation lease's native acknowledgment.
///
/// Queued cells become complete only after the owner consumes their notification.
/// Each publication is unique. Reuse requires consumption and no remaining publisher.
#[repr(C, align(8))]
pub struct LeaseCompletion {
    complete: AtomicU32,
    reserved: u32,
    queue: AtomicU64,
    next: AtomicU64,
    token: AtomicU64,
}

impl Default for LeaseCompletion {
    fn default() -> Self {
        Self::new()
    }
}

impl LeaseCompletion {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            complete: AtomicU32::new(0),
            reserved: 0,
            queue: AtomicU64::new(0),
            next: AtomicU64::new(0),
            token: AtomicU64::new(0),
        }
    }

    /// Initialize an unpublished cell for a retained notification queue.
    ///
    /// # Safety
    ///
    /// The queue must stay initialized at this address until this publication has
    /// been consumed. No publisher may retain the previous use of this cell.
    #[must_use]
    pub const unsafe fn queued(queue: u64, token: u64) -> Self {
        Self {
            complete: AtomicU32::new(0),
            reserved: 0,
            queue: AtomicU64::new(queue),
            next: AtomicU64::new(0),
            token: AtomicU64::new(token),
        }
    }

    /// Reset a fully consumed cell without changing its stable address.
    ///
    /// # Safety
    ///
    /// Every previous publisher and consumer has finished. The new queue stays
    /// live until consumption; no descriptor exposes this cell during reset.
    pub unsafe fn reset_queued(&self, queue: u64, token: u64) {
        self.complete.store(0, Ordering::Relaxed);
        self.queue.store(queue, Ordering::Relaxed);
        self.next.store(0, Ordering::Relaxed);
        self.token.store(token, Ordering::Relaxed);
    }

    /// Initialize an unused, unqueued cell as complete without publishing an event.
    ///
    /// This does not synchronize with an observer. The old next/token fields cannot
    /// be consumed while unqueued and complete; `reset_queued` replaces them on reuse.
    ///
    /// # Safety
    ///
    /// Every previous publisher and consumer has finished, and no descriptor or
    /// observer exposes this cell during this exclusive initialization.
    pub unsafe fn reset_completed(&self) {
        self.queue.store(0, Ordering::Relaxed);
        self.complete.store(1, Ordering::Relaxed);
    }

    pub fn publish(&self) {
        self.publish_state(1);
    }

    /// Publish frame rejection after all partial native adoption has been dropped.
    pub fn publish_rejected(&self) {
        self.publish_state(2);
    }

    fn publish_state(&self, state: u32) {
        let queue_address = self.queue.load(Ordering::Relaxed);
        if self
            .complete
            .compare_exchange(0, state, Ordering::Release, Ordering::Relaxed)
            .is_err()
        {
            // Duplicate cancellation cannot enqueue the same intrusive node twice.
            return;
        }
        if queue_address == 0 {
            return;
        }
        // SAFETY: queued() requires this queue to outlive publication and consumption.
        let queue = unsafe { &*(queue_address as *const CompletionQueue) };
        let address = core::ptr::from_ref(self) as u64;
        let mut head = queue.head.load(Ordering::Relaxed);
        loop {
            // Never dereference the old head: the consumer may already have removed it.
            self.next.store(head, Ordering::Relaxed);
            match queue.head.compare_exchange_weak(
                head,
                address,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    /// Whether decoding rejected the frame before any operation executed.
    #[must_use]
    pub fn was_rejected(&self) -> bool {
        self.is_complete() && self.complete.load(Ordering::Acquire) & 3 == 2
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        let value = self.complete.load(Ordering::Acquire);
        if self.queue.load(Ordering::Relaxed) == 0 {
            value != 0
        } else {
            value & 4 != 0
        }
    }

    /// Consume a removed ready node before its owner can reuse the backing.
    ///
    /// # Safety
    ///
    /// This node was removed from its queue by the sole consumer, and is consumed
    /// exactly once. It remains retained until this method returns.
    #[must_use]
    pub unsafe fn consume(&self) -> (u64, u64) {
        let next = self.next.load(Ordering::Relaxed);
        let token = self.token.load(Ordering::Relaxed);
        self.complete.fetch_or(4, Ordering::Release);
        (next, token)
    }
}

/// Nonblocking producer notifications retained in their own lease cells.
///
/// There is no bounded ring to fill while the API thread waits for admission.
/// The owner drains a detached list with a bounded budget and retains its tail.
#[repr(C, align(8))]
#[derive(Default)]
pub struct CompletionQueue {
    head: AtomicU64,
}

impl CompletionQueue {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            head: AtomicU64::new(0),
        }
    }

    /// Detach all currently ready cells for the sole consumer.
    #[must_use]
    pub fn take_ready(&self) -> u64 {
        self.head.swap(0, Ordering::Acquire)
    }

    /// Whether a publication is waiting, read without detaching anything.
    ///
    /// A publication racing with this read may be missed; the next read sees it.
    /// `take_ready` provides the ordering for the cells it detaches.
    #[must_use]
    pub fn has_ready(&self) -> bool {
        self.head.load(Ordering::Relaxed) != 0
    }

    /// Return a detached, unconsumed list to the queue, ahead of newer publications.
    ///
    /// The consumer uses this when its budget ends inside a detached list, so the
    /// queue holds every unconsumed publication and `has_ready` stays exact.
    ///
    /// # Safety
    ///
    /// `first` heads a list the sole consumer detached with `take_ready` (or a
    /// suffix of one) and has not consumed. Every node stays retained until it is.
    pub unsafe fn requeue(&self, first: u64) {
        if first == 0 {
            return;
        }
        let mut last = first;
        loop {
            // SAFETY: every node of this detached list is retained and unconsumed.
            let cell = unsafe { &*(last as *const LeaseCompletion) };
            let next = cell.next.load(Ordering::Relaxed);
            if next == 0 {
                break;
            }
            last = next;
        }
        // SAFETY: `last` is the retained final node of the same detached list.
        let tail = unsafe { &*(last as *const LeaseCompletion) };
        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            tail.next.store(head, Ordering::Relaxed);
            match self
                .head
                .compare_exchange_weak(head, first, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }
}

/// Ordered replay-consumed sequences for a single device.
///
/// The submit owner publishes increasing sequences after it stops borrowing
/// each frame's bytes. Sequence zero denotes no completed frame. The PE owner
/// keeps this mailbox alive until the native device and its workers shut down.
#[repr(C, align(8))]
pub struct ReplayMailbox {
    completed: AtomicU64,
}

impl Default for ReplayMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayMailbox {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            completed: AtomicU64::new(0),
        }
    }

    pub fn publish(&self, sequence: u64) {
        self.completed.store(sequence, Ordering::Release);
    }

    #[must_use]
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }
}

/// Borrowed address of a PE-owned `LeaseCompletion` mailbox.
#[repr(transparent)]
pub struct LeaseCompletionPtr(u64);

impl LeaseCompletionPtr {
    /// Tag an explicitly retained mailbox address for the wire.
    ///
    /// # Safety
    ///
    /// `raw` must address an initialized, 8-byte-aligned `LeaseCompletion`.
    /// Its owner must keep it at that address while any native work can access it.
    /// Only the mailbox's atomic methods may mutate the shared storage.
    #[must_use]
    pub const unsafe fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

/// Borrowed address of a PE-owned `ReplayMailbox` mailbox.
#[repr(transparent)]
pub struct ReplayMailboxPtr(u64);

impl ReplayMailboxPtr {
    /// Tag an explicitly retained mailbox address for the wire.
    ///
    /// # Safety
    ///
    /// `raw` must address an initialized, 8-byte-aligned `ReplayMailbox`.
    /// Its owner must keep it at that address while any native work can access it.
    /// Only the mailbox's atomic methods may mutate the shared storage.
    #[must_use]
    pub const unsafe fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

const _: () = {
    assert!(cfg!(target_has_atomic = "64"));
    assert!(size_of::<AtomicU64>() == 8);
    assert!(std::mem::offset_of!(LeaseCompletion, complete) == 0);
    assert!(std::mem::offset_of!(LeaseCompletion, reserved) == 4);
    assert!(std::mem::offset_of!(ReplayMailbox, completed) == 0);
    assert!(size_of::<LeaseCompletion>() == 32);
    assert!(std::mem::offset_of!(LeaseCompletion, queue) == 8);
    assert!(std::mem::offset_of!(LeaseCompletion, next) == 16);
    assert!(std::mem::offset_of!(LeaseCompletion, token) == 24);
    assert!(size_of::<CompletionQueue>() == 8);
    assert!(align_of::<CompletionQueue>() == 8);
    assert!(std::mem::offset_of!(CompletionQueue, head) == 0);
    assert!(align_of::<LeaseCompletion>() == 8);
    assert!(size_of::<ReplayMailbox>() == 8);
    assert!(align_of::<ReplayMailbox>() == 8);
    assert!(size_of::<LeaseCompletionPtr>() == 8);
    assert!(size_of::<ReplayMailboxPtr>() == 8);
};
