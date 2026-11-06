//! Small reply cells retained by a frame's guest owner.
//!
//! The native side writes through the cell's address and never changes the guest's reference
//! count.

use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64},
};

/// A retained 64-bit reply cell.
pub struct ReplyU64 {
    cell: Arc<AtomicU64>,
}

/// A retained fixed-width success reply cell.
pub struct ReplyBool {
    cell: Arc<AtomicU32>,
}

impl ReplyU64 {
    /// Address of the retained reply cell, for the explicit wire pointer.
    #[must_use]
    pub fn address(&self) -> u64 {
        Arc::as_ptr(&self.cell) as u64
    }
}

impl From<Arc<AtomicU64>> for ReplyU64 {
    fn from(cell: Arc<AtomicU64>) -> Self {
        Self { cell }
    }
}

impl ReplyBool {
    /// Address of the retained reply cell, for the explicit wire pointer.
    #[must_use]
    pub fn address(&self) -> u64 {
        Arc::as_ptr(&self.cell) as u64
    }
}

impl From<Arc<AtomicU32>> for ReplyBool {
    fn from(cell: Arc<AtomicU32>) -> Self {
        Self { cell }
    }
}
