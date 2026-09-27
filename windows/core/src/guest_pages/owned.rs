//! Unique guest pages retained until native and GPU consumers retire.

#[cfg(not(windows))]
use mtld3d_shared::encoder_wire::WireError;
use mtld3d_shared::{
    InPtr,
    encoder_wire::{LeaseCompletion, LeaseCompletionPtr},
};

#[cfg(not(windows))]
use super::valid_range;
#[cfg(not(windows))]
use crate::page_box::PAGE_SIZE;
use crate::{
    guest_completions::{CompletionPool, CompletionSlot},
    page_box::PageBox,
    page_box_pool::PageBoxPool,
};

/// PE ownership of unique buffer backing or upload snapshots and their final acknowledgment.
///
/// Moving this owner never moves the allocation or the pooled completion cell. Unlike shared
/// staging leases, it publishes no pointer into its `PageBox` metadata and needs no metadata Arc.
pub struct GuestOwnedPageLease {
    owner: PageBox,
    slot: CompletionSlot,
    recycle_pool: Option<&'static PageBoxPool>,
}

impl GuestOwnedPageLease {
    #[must_use]
    pub fn new(
        owner: PageBox,
        pool: &CompletionPool,
        recycle_pool: Option<&'static PageBoxPool>,
    ) -> Self {
        Self {
            owner,
            slot: pool.allocate(false),
            recycle_pool,
        }
    }

    #[must_use]
    pub fn token(&self) -> u64 {
        self.slot.token()
    }

    /// Publish only independently allocated storage, never movable owner metadata.
    #[must_use]
    pub fn descriptor(&self) -> GuestOwnedPageDescriptor {
        GuestOwnedPageDescriptor {
            source: self.owner.as_ptr() as u64,
            padded_len: self.owner.len() as u64,
            generation: self.owner.generation(),
            completion: core::ptr::from_ref(self.slot.completion()) as u64,
        }
    }

    /// A queued notification must be consumed before its owner or cell is recycled.
    #[must_use]
    pub fn completed(&self) -> bool {
        self.slot.completion().is_complete()
    }

    /// Return original pages to PE's pool after the final acknowledgment is consumed.
    ///
    /// # Panics
    /// Panics if completion has not been consumed or the page pool mutex is poisoned.
    #[must_use]
    pub fn into_slot(self) -> CompletionSlot {
        assert!(self.completed(), "retirement completion consumed");
        if let Some(pool) = self.recycle_pool {
            drop(pool.recycle(self.owner));
        }
        self.slot
    }

    /// Cancel only after every existing native user of these bytes has ended.
    ///
    /// # Safety
    /// No native decoder, cache, worker, callback or GPU work can access these bytes or adopt
    /// this descriptor later. A retired backing can have consumers from preceding frames.
    pub unsafe fn cancel_unadopted(&self) {
        self.slot.completion().publish();
    }
}

/// Ownership-only wire descriptor. It cannot expose or acquire a shared reader counter.
#[repr(C, align(8))]
pub struct GuestOwnedPageDescriptor {
    source: u64,
    padded_len: u64,
    generation: u64,
    completion: u64,
}

const _: () = {
    assert!(size_of::<GuestOwnedPageDescriptor>() == 32);
    assert!(align_of::<GuestOwnedPageDescriptor>() == 8);
    assert!(core::mem::offset_of!(GuestOwnedPageDescriptor, source) == 0);
    assert!(core::mem::offset_of!(GuestOwnedPageDescriptor, padded_len) == 8);
    assert!(core::mem::offset_of!(GuestOwnedPageDescriptor, generation) == 16);
    assert!(core::mem::offset_of!(GuestOwnedPageDescriptor, completion) == 24);
};

impl GuestOwnedPageDescriptor {
    /// Adopt a single native ownership guard without borrowing PE owner metadata.
    ///
    /// # Safety
    /// The unique PE lease retains its bytes and cell until this guard drops.
    /// This is the descriptor's only adoption. No writer can modify the published allocation.
    ///
    /// # Errors
    /// Rejects an invalid page extent or completion address before accessing either.
    #[cfg(not(windows))]
    pub unsafe fn adopt(&self) -> Result<GuestOwnedPage, WireError> {
        let len = usize::try_from(self.padded_len).map_err(|_| WireError::TooLarge)?;
        if len == 0
            || !len.is_multiple_of(PAGE_SIZE)
            || !valid_range(self.source, self.padded_len, PAGE_SIZE)
            || !valid_range(self.completion, size_of::<LeaseCompletion>() as u64, 8)
        {
            return Err(WireError::InvalidValue);
        }
        // SAFETY: validation checks the cell address; the caller retains it until final drop.
        let completion = unsafe { LeaseCompletionPtr::new(self.completion) };
        Ok(GuestOwnedPage {
            source: self.source,
            len,
            generation: self.generation,
            completion,
        })
    }
}

/// Native ownership guard with no reader-count or allocator access.
pub struct GuestOwnedPage {
    source: u64,
    len: usize,
    generation: u64,
    completion: LeaseCompletionPtr,
}

impl GuestOwnedPage {
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.source as *const u8
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for GuestOwnedPage {
    fn drop(&mut self) {
        // SAFETY: adoption retains this initialized cell until the sole guard's final drop.
        let completion =
            unsafe { InPtr::<LeaseCompletion>::new(self.completion.raw() as *const _) };
        completion.publish();
    }
}

#[cfg(test)]
mod tests;
