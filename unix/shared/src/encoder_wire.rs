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
            bytes: &mut self.bytes,
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

pub struct WireWriter<'a> {
    bytes: &'a mut Vec<u8>,
}

impl WireWriter<'_> {
    const fn len(&self) -> usize {
        self.bytes.len()
    }

    fn overwrite(&mut self, offset: usize, value: &[u8]) {
        self.bytes[offset..offset + value.len()].copy_from_slice(value);
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

    #[inline]
    fn fixed<const N: usize>(&mut self, value: [u8; N]) -> Result<(), WireError> {
        Self::append_vector(self.bytes, &value)
    }

    /// Append bytes without adding a length prefix.
    ///
    /// # Errors
    ///
    /// Returns `TooLarge` beyond the slab byte limit or `AllocationFailed`.
    pub fn bytes(&mut self, value: &[u8]) -> Result<(), WireError> {
        Self::append_vector(self.bytes, value)
    }

    /// Append to the slab, refusing growth past the `u32` byte limit.
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
}

impl<'a> WireReader<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
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
        };
        let tag = cursor.u16()?;
        let length = cursor.u32()?;
        let payload = Self {
            remaining: cursor.bytes(length)?,
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

const _: () = {
    assert!(cfg!(target_has_atomic = "64"));
    assert!(size_of::<AtomicU64>() == 8);
    assert!(std::mem::offset_of!(LeaseCompletion, complete) == 0);
    assert!(std::mem::offset_of!(LeaseCompletion, reserved) == 4);
    assert!(size_of::<LeaseCompletion>() == 32);
    assert!(std::mem::offset_of!(LeaseCompletion, queue) == 8);
    assert!(std::mem::offset_of!(LeaseCompletion, next) == 16);
    assert!(std::mem::offset_of!(LeaseCompletion, token) == 24);
    assert!(size_of::<CompletionQueue>() == 8);
    assert!(align_of::<CompletionQueue>() == 8);
    assert!(std::mem::offset_of!(CompletionQueue, head) == 0);
    assert!(align_of::<LeaseCompletion>() == 8);
    assert!(size_of::<LeaseCompletionPtr>() == 8);
};
