//! Device-owned completion cells recycled after native notifications are consumed.
//!
//! Recording leases share a slot allocator. Native publishers touch only the fixed-layout
//! cells and queue, never this allocator or a PE-owned Rust object. Every publication the
//! device waits for, a lease's and a frame packet's replay completion alike, goes onto the
//! one queue, so an empty queue means nothing has arrived since the last drain.

use std::sync::{Arc, Mutex};

use mtld3d_shared::encoder_wire::{CompletionQueue, LeaseCompletion};

/// Token of a frame packet's replay completion, outside every slot's token range.
///
/// Slot tokens are `index * 2` and `index * 2 + 1` for indices that fit a `usize`.
pub const REPLAY_COMPLETION_TOKEN: u64 = u64::MAX;

const SLOTS_PER_BLOCK: usize = 128;

struct CompletionBlock {
    queue: Arc<CompletionQueue>,
    cells: [[LeaseCompletion; 2]; SLOTS_PER_BLOCK],
}

#[derive(Default)]
struct SlotAllocator {
    blocks: Vec<Arc<CompletionBlock>>,
    free: Vec<usize>,
    allocated: usize,
}

/// One device's shared capture allocator and nonblocking completion queue.
///
/// Clone into each recording frame. The mutex is taken only for lease creation
/// or recycling, never for ordinary draw capture or native publication.
#[derive(Clone)]
pub struct CompletionPool {
    queue: Arc<CompletionQueue>,
    allocator: Arc<Mutex<SlotAllocator>>,
}

impl Default for CompletionPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionPool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: Arc::new(CompletionQueue::new()),
            allocator: Arc::default(),
        }
    }

    /// Reserve adjacent acquisition and final-completion cells.
    ///
    /// # Panics
    ///
    /// Panics if a slot index cannot fit the fixed-width wire token.
    #[must_use]
    pub fn allocate(&self, needs_acquire: bool) -> CompletionSlot {
        let mut allocator = self
            .allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = allocator.free.pop().unwrap_or_else(|| {
            let index = allocator.allocated;
            allocator.allocated += 1;
            index
        });
        let block_index = index / SLOTS_PER_BLOCK;
        if block_index == allocator.blocks.len() {
            allocator.blocks.push(Arc::new(CompletionBlock {
                queue: Arc::clone(&self.queue),
                cells: std::array::from_fn(|_| std::array::from_fn(|_| LeaseCompletion::new())),
            }));
        }
        let block = Arc::clone(&allocator.blocks[block_index]);
        drop(allocator);
        let token = u64::try_from(index).expect("slot index fits u64");
        let address = Arc::as_ptr(&block.queue) as u64;
        let cells = &block.cells[index % SLOTS_PER_BLOCK];
        if needs_acquire {
            // SAFETY: a fresh or recycled slot has no old publisher or consumer.
            // Its block retains the queue until every slot handle is released.
            unsafe { cells[0].reset_queued(address, token * 2) };
        } else {
            // SAFETY: this exclusive, unexposed slot has no old publisher or consumer.
            unsafe { cells[0].reset_completed() };
        }
        // SAFETY: the same exclusive slot reservation retains the queue and final cell.
        unsafe { cells[1].reset_queued(address, token * 2 + 1) };
        CompletionSlot { block, index }
    }

    /// A mailbox for a frame packet's replay completion, published onto this pool's queue.
    ///
    /// Its notification carries [`REPLAY_COMPLETION_TOKEN`] and has to be consumed by a
    /// drain before the cell reports completion or its owner may free it. The owner
    /// keeps a clone of this pool, and so the queue, until then.
    #[must_use]
    pub fn replay_completion(&self) -> Box<LeaseCompletion> {
        let address = Arc::as_ptr(&self.queue) as u64;
        // SAFETY: the caller retains a clone of this pool, which keeps the queue at this
        // address until the notification has been consumed. The cell is new.
        Box::new(unsafe { LeaseCompletion::queued(address, REPLAY_COMPLETION_TOKEN) })
    }

    /// Whether any notification waits to be drained, read without a lock.
    ///
    /// False means no lease or replay completion has been published since the last
    /// drain consumed the queue, up to a publication racing with this read.
    #[must_use]
    pub fn has_ready(&self) -> bool {
        self.queue.has_ready()
    }

    /// Consume at most `budget` notifications without inspecting live leases.
    ///
    /// One cursor is used by the device's sole consumer. The callback receives
    /// `slot * 2` for acquisition, `slot * 2 + 1` for final completion and
    /// [`REPLAY_COMPLETION_TOKEN`] for a packet's replay completion. What the
    /// budget leaves unconsumed goes back onto the queue. Returns the count consumed.
    ///
    /// # Panics
    ///
    /// Panics if a cursor from another device is used.
    pub fn drain(
        &self,
        cursor: &mut CompletionDrain,
        budget: usize,
        mut consume: impl FnMut(u64),
    ) -> usize {
        if let Some(queue) = &cursor.queue {
            assert!(
                Arc::ptr_eq(queue, &self.queue),
                "completion cursor belongs to device"
            );
        } else {
            cursor.queue = Some(Arc::clone(&self.queue));
            cursor.allocator = Some(Arc::clone(&self.allocator));
        }
        let mut pending = self.queue.take_ready();
        let mut consumed = 0;
        while pending != 0 {
            if consumed == budget {
                // SAFETY: `pending` heads the unconsumed remainder of the list this sole
                // consumer detached above; every node stays retained until consumed.
                unsafe { self.queue.requeue(pending) };
                break;
            }
            // SAFETY: every queued node belongs to a retained slot or packet. Only this
            // cursor consumes the detached list, and recycling requires consumption first.
            let cell = unsafe { &*(pending as *const LeaseCompletion) };
            // SAFETY: this node was uniquely detached and stays retained by its owner.
            let (next, token) = unsafe { cell.consume() };
            pending = next;
            consumed += 1;
            consume(token);
        }
        consumed
    }

    /// Return a slot only after both cells have been consumed.
    ///
    /// # Panics
    ///
    /// Panics if the slot still has a publisher, notification or different device owner.
    pub fn recycle(&self, slot: CompletionSlot) {
        self.check_recyclable(&slot);
        let CompletionSlot { block, index } = slot;
        drop(block);
        self.allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .free
            .push(index);
    }

    /// Return every slot in `slots` under one allocator lock, leaving the vector empty.
    ///
    /// Each slot obeys the same rule as [`Self::recycle`]. The vector keeps its capacity.
    ///
    /// # Panics
    ///
    /// Panics if a slot still has a publisher, notification or different device owner.
    pub fn recycle_all(&self, slots: &mut Vec<CompletionSlot>) {
        if slots.is_empty() {
            return;
        }
        for slot in slots.iter() {
            self.check_recyclable(slot);
        }
        self.allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .free
            .extend(slots.iter().map(|slot| slot.index));
        slots.clear();
    }

    fn check_recyclable(&self, slot: &CompletionSlot) {
        assert!(
            Arc::ptr_eq(&self.queue, &slot.block.queue),
            "completion slot belongs to device"
        );
        assert!(
            slot.acquired().is_complete() && slot.completion().is_complete(),
            "completion slot consumed"
        );
    }
}

/// The device's sole-consumer cursor, bound to one pool's queue on first use.
///
/// It retains the queue and the slot blocks, so a notification left unconsumed on
/// the queue keeps its cell alive for as long as the cursor can still reach it.
#[derive(Default)]
pub struct CompletionDrain {
    queue: Option<Arc<CompletionQueue>>,
    allocator: Option<Arc<Mutex<SlotAllocator>>>,
}

/// Stable mailbox pair retained independently of frame replay storage.
pub struct CompletionSlot {
    block: Arc<CompletionBlock>,
    index: usize,
}

impl CompletionSlot {
    /// Stable slot identity in this device pool.
    ///
    /// # Panics
    ///
    /// Panics if the slot index cannot fit the fixed-width wire token.
    #[must_use]
    pub fn token(&self) -> u64 {
        u64::try_from(self.index).expect("slot index fits u64")
    }

    #[must_use]
    pub fn acquired(&self) -> &LeaseCompletion {
        &self.block.cells[self.index % SLOTS_PER_BLOCK][0]
    }

    #[must_use]
    pub fn completion(&self) -> &LeaseCompletion {
        &self.block.cells[self.index % SLOTS_PER_BLOCK][1]
    }
}

/// Mailbox backing for a pooled runtime lease or an isolated protocol test.
pub enum LeaseCells {
    Standalone(Box<[LeaseCompletion; 2]>),
    Pooled(CompletionSlot),
}

impl Default for LeaseCells {
    fn default() -> Self {
        Self::Standalone(Box::new(std::array::from_fn(|_| LeaseCompletion::new())))
    }
}

impl LeaseCells {
    #[must_use]
    pub fn acquired(&self) -> &LeaseCompletion {
        match self {
            Self::Standalone(cells) => &cells[0],
            Self::Pooled(slot) => slot.acquired(),
        }
    }
    #[must_use]
    pub fn completion(&self) -> &LeaseCompletion {
        match self {
            Self::Standalone(cells) => &cells[1],
            Self::Pooled(slot) => slot.completion(),
        }
    }
    #[must_use]
    pub fn token(&self) -> Option<u64> {
        match self {
            Self::Standalone(_) => None,
            Self::Pooled(slot) => Some(slot.token()),
        }
    }
    #[must_use]
    pub fn into_slot(self) -> Option<CompletionSlot> {
        match self {
            Self::Standalone(_) => None,
            Self::Pooled(slot) => Some(slot),
        }
    }
    #[must_use]
    pub fn reusable(&self) -> bool {
        self.completion().is_complete() && (self.token().is_none() || self.acquired().is_complete())
    }
}

#[cfg(test)]
mod tests;
