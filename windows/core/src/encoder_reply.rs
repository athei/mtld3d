//! Small reply cells retained by a frame's guest owner.
//!
//! The native view only borrows the cell and never changes the guest's reference count.

use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering},
};

/// A retained 64-bit reply cell or its native borrowed view.
pub struct ReplyU64 {
    backing: ReplyBacking<AtomicU64>,
}

/// A fixed-width success reply or its native borrowed view.
pub struct ReplyBool {
    backing: ReplyBacking<AtomicU32>,
}

enum ReplyBacking<T> {
    Owned(Arc<T>),
    Guest(u64),
}

impl ReplyU64 {
    /// Borrow a reply cell pinned until the frame's replay acknowledgment.
    ///
    /// # Safety
    ///
    /// `address` names a live aligned `AtomicU64` for every use of this reply.
    #[must_use]
    pub const unsafe fn from_guest(address: u64) -> Self {
        Self {
            backing: ReplyBacking::Guest(address),
        }
    }

    /// Address of the retained reply cell, for the explicit wire pointer.
    #[must_use]
    pub fn address(&self) -> u64 {
        match &self.backing {
            ReplyBacking::Owned(value) => Arc::as_ptr(value) as u64,
            ReplyBacking::Guest(address) => *address,
        }
    }

    pub fn store(&self, value: u64, ordering: Ordering) {
        match &self.backing {
            ReplyBacking::Owned(cell) => cell.store(value, ordering),
            ReplyBacking::Guest(address) => {
                // SAFETY: from_guest pins this aligned atomic through every reply access.
                let cell = unsafe { &*(*address as *const AtomicU64) };
                cell.store(value, ordering);
            }
        }
    }
}

impl From<Arc<AtomicU64>> for ReplyU64 {
    fn from(value: Arc<AtomicU64>) -> Self {
        Self {
            backing: ReplyBacking::Owned(value),
        }
    }
}

impl ReplyBool {
    /// Borrow a success cell pinned until the frame's replay acknowledgment.
    ///
    /// # Safety
    ///
    /// `address` names a live aligned `AtomicU32` for every use of this reply.
    #[must_use]
    pub const unsafe fn from_guest(address: u64) -> Self {
        Self {
            backing: ReplyBacking::Guest(address),
        }
    }

    /// Address of the retained reply cell, for the explicit wire pointer.
    #[must_use]
    pub fn address(&self) -> u64 {
        match &self.backing {
            ReplyBacking::Owned(value) => Arc::as_ptr(value) as u64,
            ReplyBacking::Guest(address) => *address,
        }
    }

    pub fn store(&self, value: bool, ordering: Ordering) {
        let value = u32::from(value);
        match &self.backing {
            ReplyBacking::Owned(cell) => cell.store(value, ordering),
            ReplyBacking::Guest(address) => {
                // SAFETY: from_guest pins this aligned atomic through every reply access.
                let cell = unsafe { &*(*address as *const AtomicU32) };
                cell.store(value, ordering);
            }
        }
    }
}

impl From<Arc<AtomicU32>> for ReplyBool {
    fn from(value: Arc<AtomicU32>) -> Self {
        Self {
            backing: ReplyBacking::Owned(value),
        }
    }
}

#[cfg(test)]
mod tests;
