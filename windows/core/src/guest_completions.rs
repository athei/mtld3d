//! Device-owned completion cells recycled after native notifications are consumed.
//!
//! Recording leases share a slot allocator. Native publishers touch only the fixed-layout
//! cells and queue, never this allocator or a PE-owned Rust object.

use std::sync::{Arc, Mutex};

use mtld3d_shared::encoder_wire::{CompletionQueue, LeaseCompletion};

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
        // SAFETY: a fresh slot or a recycled slot has no old publisher or consumer.
        // Its block retains the queue until every slot handle is released.
        unsafe { cells[0].reset_queued(if needs_acquire { address } else { 0 }, token * 2) };
        // SAFETY: the same exclusive slot reservation retains the queue and final cell.
        unsafe { cells[1].reset_queued(address, token * 2 + 1) };
        if !needs_acquire {
            cells[0].publish();
        }
        CompletionSlot { block, index }
    }

    /// Consume at most `budget` notifications without inspecting live leases.
    ///
    /// One cursor is used by the device's sole consumer. The callback receives
    /// `slot * 2` for acquisition and `slot * 2 + 1` for final completion.
    ///
    /// # Panics
    ///
    /// Panics if a cursor from another device is used.
    pub fn drain(&self, cursor: &mut CompletionDrain, budget: usize, mut consume: impl FnMut(u64)) {
        if let Some(queue) = &cursor.queue {
            assert!(
                Arc::ptr_eq(queue, &self.queue),
                "completion cursor belongs to device"
            );
        } else {
            cursor.queue = Some(Arc::clone(&self.queue));
            cursor.allocator = Some(Arc::clone(&self.allocator));
        }
        for _ in 0..budget {
            if cursor.pending == 0 {
                cursor.pending = self.queue.take_ready();
            }
            if cursor.pending == 0 {
                break;
            }
            // SAFETY: every queued node belongs to a retained slot. Only this cursor
            // consumes the detached list, and recycling requires consumption first.
            let cell = unsafe { &*(cursor.pending as *const LeaseCompletion) };
            // SAFETY: this node was uniquely detached and stays retained by its lease.
            let (next, token) = unsafe { cell.consume() };
            cursor.pending = next;
            consume(token);
        }
    }

    /// Return a slot only after both cells have been consumed.
    ///
    /// # Panics
    ///
    /// Panics if the slot still has a publisher, notification or different device owner.
    pub fn recycle(&self, slot: CompletionSlot) {
        assert!(
            Arc::ptr_eq(&self.queue, &slot.block.queue),
            "completion slot belongs to device"
        );
        assert!(
            slot.acquired().is_complete() && slot.completion().is_complete(),
            "completion slot consumed"
        );
        let CompletionSlot { block, index } = slot;
        drop(block);
        self.allocator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .free
            .push(index);
    }
}

/// Sole-consumer cursor retaining a detached notification tail between bounded drains.
#[derive(Default)]
pub struct CompletionDrain {
    pending: u64,
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
