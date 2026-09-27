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

use crate::{
    encoder_value::WireValue,
    guest_completions::{CompletionPool, CompletionSlot, LeaseCells},
    page_box::{PAGE_SIZE, PageBox, PageBoxRead},
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
}

impl GuestPageLease {
    /// Retain an existing read until native acquisition closes the handoff.
    #[must_use]
    pub fn for_read(read: PageBoxRead) -> Self {
        Self {
            owner: Arc::clone(read.backing()),
            read: Some(read),
            cells: LeaseCells::default(),
        }
    }

    #[must_use]
    pub fn for_read_pooled(read: PageBoxRead, pool: &CompletionPool) -> Self {
        Self {
            owner: Arc::clone(read.backing()),
            read: Some(read),
            cells: LeaseCells::Pooled(pool.allocate(true)),
        }
    }

    #[must_use]
    pub fn token(&self) -> Option<u64> {
        self.cells.token()
    }

    /// Return mailbox storage after retirement.
    #[must_use]
    pub fn into_slot(self) -> Option<CompletionSlot> {
        self.cells.into_slot()
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
