//! Guest allocation leases retained until native ownership ends.
//!
//! Only fixed-width addresses cross the runtime boundary. PE owns every original allocation,
//! reader guard and acknowledgment cell; native wrappers borrow them without running PE drops.

#[cfg(not(windows))]
use std::ptr::NonNull;
use std::sync::Arc;

#[cfg(not(windows))]
use mtld3d_shared::encoder_wire::LeaseCompletionPtr;
use mtld3d_shared::{
    InPtr,
    encoder_wire::{LeaseCompletion, WireError, WireReader, WireWriter},
};
use rustc_hash::FxHashMap;

use crate::{
    encoder_value::WireValue,
    guest_completions::{CompletionPool, CompletionSlot, LeaseCells},
    page_box::{PAGE_SIZE, PageBox, PageBoxRead},
    page_box_pool::PageBoxPool,
};

mod owned;
pub use owned::{GuestOwnedPage, GuestOwnedPageDescriptor, GuestOwnedPageLease};

/// PE-owned allocation and acknowledgment storage for one published native lease.
///
/// Retain this value until `maintain` returns true, or cancel it before native adoption. Native
/// adoption is unsafe and requires this retention contract. Moving the lease keeps all shared
/// addresses stable because its allocation and mailboxes have independent heap owners.
pub struct GuestPageLease {
    owner: Arc<PageBox>,
    read: Option<PageBoxRead>,
    cells: LeaseCells,
    /// Staging lane the owner is offered to once native ownership has ended.
    recycle_pool: Option<&'static PageBoxPool>,
}

impl GuestPageLease {
    /// Retain an existing read until native acquisition closes the handoff.
    #[must_use]
    pub fn for_read(read: PageBoxRead) -> Self {
        Self {
            owner: Arc::clone(read.backing()),
            read: Some(read),
            cells: LeaseCells::default(),
            recycle_pool: None,
        }
    }

    /// A pooled read lease whose owner is offered to `recycle_pool` at retirement.
    ///
    /// The offer parks the pages only when this lease held their last owner.
    #[must_use]
    pub fn for_read_pooled(
        read: PageBoxRead,
        pool: &CompletionPool,
        recycle_pool: Option<&'static PageBoxPool>,
    ) -> Self {
        Self {
            owner: Arc::clone(read.backing()),
            read: Some(read),
            cells: LeaseCells::Pooled(pool.allocate(true)),
            recycle_pool,
        }
    }

    #[must_use]
    pub fn token(&self) -> Option<u64> {
        self.cells.token()
    }

    /// Return mailbox storage after retirement, offering the owner to the staging lane.
    ///
    /// The owner is offered only once both acknowledgments were observed, so native code has
    /// dropped every adopted page, wrapper keepalive and reader of these bytes, after its
    /// wrapper destroys and the GPU completion that gates them. Before that, or without a
    /// pool, the owner drops as before.
    #[must_use]
    pub fn into_slot(self) -> Option<CompletionSlot> {
        let Self {
            owner,
            read,
            cells,
            recycle_pool,
        } = self;
        drop(read);
        if let Some(pool) = recycle_pool
            && cells.reusable()
        {
            pool.recycle_staging(owner);
        }
        cells.into_slot()
    }

    /// Describe one native adoption while this PE lease remains retained.
    #[must_use]
    pub fn descriptor(&self) -> GuestPageDescriptor {
        GuestPageDescriptor {
            source: self.owner.as_ptr() as u64,
            padded_len: self.owner.len() as u64,
            logical_len: self.owner.logical_len() as u64,
            generation: self.owner.generation(),
            readers: self.owner.reader_count_ptr(),
            completion: core::ptr::from_ref(self.cells.completion()) as u64,
            read_acquired: if self.read.is_some() {
                core::ptr::from_ref(self.cells.acquired()) as u64
            } else {
                0
            },
        }
    }

    /// Release the original read after acquisition, and report final ownership completion.
    ///
    /// An unacquired read also ends when a rejected descriptor is canceled. Cached native
    /// ownership keeps the original allocation alive but does not keep its reader count raised.
    pub fn maintain(&mut self) -> bool {
        let complete = self.cells.completion().is_complete();
        if complete || self.cells.acquired().is_complete() {
            self.read = None;
        }
        self.cells.reusable()
    }

    /// Cancel a publication that native code has not adopted.
    ///
    /// # Safety
    ///
    /// No native consumer has adopted or can subsequently access this descriptor. Cancellation
    /// and native adoption are mutually exclusive, and the caller invokes cancellation once.
    pub unsafe fn cancel_unadopted(&mut self) {
        self.cells.acquired().publish();
        self.cells.completion().publish();
        self.read = None;
    }
}

/// Padded bytes of the allocations only page leases keep, tallied lease by lease.
///
/// A lease keeps its allocation through its owner and, until native acquisition, through its
/// read as well. An allocation whose strong count is no more than the references the tallied
/// leases hold on it has no other PE holder: the texture let go of that staging and only the
/// leases, waiting on native code, keep it. Leases of one allocation share its owner `Arc`, so
/// they are grouped by its address and each allocation counts once.
#[derive(Default)]
pub struct LeaseOnlyPages {
    allocations: FxHashMap<usize, LeasedAllocation>,
}

impl LeaseOnlyPages {
    /// Count the references `lease` holds on its allocation.
    pub fn add(&mut self, lease: &GuestPageLease) {
        let refs = 1 + usize::from(lease.read.is_some());
        self.allocations
            .entry(Arc::as_ptr(&lease.owner).addr())
            .or_insert_with(|| LeasedAllocation {
                padded: lease.owner.len() as u64,
                lease_refs: 0,
                strong: Arc::strong_count(&lease.owner),
            })
            .lease_refs += refs;
    }

    /// Padded bytes of every tallied allocation no holder but the leases keeps.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.allocations
            .values()
            .filter(|allocation| allocation.lease_refs >= allocation.strong)
            .map(|allocation| allocation.padded)
            .sum()
    }
}

/// Field-wise wire descriptor for one borrowed guest allocation.
///
/// All addresses are `u64` regardless of guest pointer width. Lengths count bytes. `read_acquired`
/// is zero for ownership-only publication; otherwise it addresses a separate acquisition cell.
/// A descriptor has exactly one terminal consumer: native adoption or cancellation.
#[repr(C, align(8))]
pub struct GuestPageDescriptor {
    source: u64,
    padded_len: u64,
    logical_len: u64,
    generation: u64,
    readers: u64,
    completion: u64,
    read_acquired: u64,
}

const _: () = {
    assert!(size_of::<GuestPageDescriptor>() == 56);
    assert!(align_of::<GuestPageDescriptor>() == 8);
    assert!(core::mem::offset_of!(GuestPageDescriptor, source) == 0);
    assert!(core::mem::offset_of!(GuestPageDescriptor, padded_len) == 8);
    assert!(core::mem::offset_of!(GuestPageDescriptor, logical_len) == 16);
    assert!(core::mem::offset_of!(GuestPageDescriptor, generation) == 24);
    assert!(core::mem::offset_of!(GuestPageDescriptor, readers) == 32);
    assert!(core::mem::offset_of!(GuestPageDescriptor, completion) == 40);
    assert!(core::mem::offset_of!(GuestPageDescriptor, read_acquired) == 48);
};

impl GuestPageDescriptor {
    /// Exact fixed-width descriptor fields used by packet publication validation.
    #[must_use]
    pub const fn wire_fields(&self) -> [u64; 7] {
        [
            self.source,
            self.padded_len,
            self.logical_len,
            self.generation,
            self.readers,
            self.completion,
            self.read_acquired,
        ]
    }

    /// Acquire the native read before acknowledging the original PE read.
    ///
    /// # Safety
    ///
    /// PE retains its original read guard, allocation and acknowledgment cells until acquisition
    /// is observed. This is the only adoption. PE then retains the allocation and cells until
    /// final completion; native cached owners may outlive this read.
    ///
    /// # Errors
    ///
    /// Rejects invalid ranges or a descriptor without an acquisition cell.
    ///
    /// # Panics
    ///
    /// Panics if the allocation's shared reader count is exhausted.
    #[cfg(not(windows))]
    pub unsafe fn adopt_read(&self) -> Result<PageBoxRead, WireError> {
        if self.read_acquired == 0 {
            return Err(WireError::InvalidValue);
        }
        self.validate()?;
        // SAFETY: validation checks alignment and the caller retains this initialized cell.
        let acquired = unsafe { InPtr::<LeaseCompletion>::new(self.read_acquired as *const _) };
        // SAFETY: the caller retains the sole lease and its original read until acquisition.
        let backing = Arc::new(unsafe { self.adopt_page()? });
        let read = PageBoxRead::new(backing);
        acquired.publish();
        Ok(read)
    }

    /// Acknowledge rejection before any native adoption.
    ///
    /// # Safety
    ///
    /// PE keeps the initialized completion cell alive until this acknowledgment is observed.
    /// No native consumer has adopted or can later access this descriptor; cancellation runs once.
    ///
    /// # Errors
    ///
    /// Rejects invalid descriptor ranges without accessing an acknowledgment cell.
    pub unsafe fn cancel_unadopted(self) -> Result<(), WireError> {
        self.validate()?;
        // SAFETY: validation checks alignment and the caller retains the initialized cell.
        let completion = unsafe { InPtr::<LeaseCompletion>::new(self.completion as *const _) };
        if self.read_acquired != 0 {
            // SAFETY: the retained descriptor owns the aligned acquisition cell.
            let acquired = unsafe { InPtr::<LeaseCompletion>::new(self.read_acquired as *const _) };
            acquired.publish();
        }
        completion.publish();
        Ok(())
    }

    #[cfg(not(windows))]
    unsafe fn adopt_page(&self) -> Result<PageBox, WireError> {
        self.validate()?;
        let source = NonNull::new(self.source as *mut u8).ok_or(WireError::InvalidValue)?;
        // SAFETY: validate checks the fixed-layout cell, and adoption retains it until final drop.
        let completion = unsafe { LeaseCompletionPtr::new(self.completion) };
        // SAFETY: validation checks page alignment and extents; the caller grants unique native
        // ownership of the retained PE allocation, original reader counter and completion cell.
        Ok(unsafe {
            PageBox::from_guest_lease(
                source,
                usize::try_from(self.padded_len).map_err(|_| WireError::TooLarge)?,
                usize::try_from(self.logical_len).map_err(|_| WireError::TooLarge)?,
                self.generation,
                completion,
                self.readers,
            )
        })
    }

    fn validate(&self) -> Result<(), WireError> {
        let padded = usize::try_from(self.padded_len).map_err(|_| WireError::TooLarge)?;
        if padded == 0
            || !padded.is_multiple_of(PAGE_SIZE)
            || self.logical_len > self.padded_len
            || !valid_range(self.source, self.padded_len, PAGE_SIZE)
            || !valid_range(self.readers, 4, 4)
            || !valid_range(self.completion, size_of::<LeaseCompletion>() as u64, 8)
            || (self.read_acquired != 0
                && !valid_range(self.read_acquired, size_of::<LeaseCompletion>() as u64, 8))
        {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }
}

impl WireValue for GuestPageDescriptor {
    const MIN_WIRE_BYTES: usize = 7 * size_of::<u64>();

    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u64(self.source)?;
        writer.u64(self.padded_len)?;
        writer.u64(self.logical_len)?;
        writer.u64(self.generation)?;
        writer.u64(self.readers)?;
        writer.u64(self.completion)?;
        writer.u64(self.read_acquired)
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let value = Self {
            source: reader.u64()?,
            padded_len: reader.u64()?,
            logical_len: reader.u64()?,
            generation: reader.u64()?,
            readers: reader.u64()?,
            completion: reader.u64()?,
            read_acquired: reader.u64()?,
        };
        value.validate()?;
        Ok(value)
    }
}

/// One allocation's share of [`LeaseOnlyPages`].
struct LeasedAllocation {
    padded: u64,
    /// The references the tallied leases hold: each owner, and each read not yet released.
    lease_refs: usize,
    /// The owner `Arc`'s strong count when the allocation was first tallied.
    strong: usize,
}

fn valid_range(address: u64, length: u64, alignment: usize) -> bool {
    let Ok(address) = usize::try_from(address) else {
        return false;
    };
    let Ok(length) = usize::try_from(length) else {
        return false;
    };
    address != 0
        && address.is_multiple_of(alignment)
        && isize::try_from(length).is_ok()
        && address.checked_add(length).is_some()
}

#[cfg(test)]
mod tests;
