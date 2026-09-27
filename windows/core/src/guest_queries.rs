//! Device-local native query identity backed by PE-owned atomic mailboxes.
//!
//! PE retains its own query core until acknowledgment. Native cores borrow only the mailbox,
//! and a weak cache keeps repeated BEGIN and END operations on the same native Arc identity.

use std::sync::{Arc, Weak};

use mtld3d_shared::{
    InPtr,
    encoder_wire::{LeaseCompletion, LeaseCompletionPtr, WireError, WireReader, WireWriter},
    query_mailbox::QueryMailbox,
};
use rustc_hash::FxHashMap;

use crate::{
    encoder_value::WireValue,
    guest_completions::{CompletionPool, CompletionSlot, LeaseCells},
    visibility::VisibilityQueryCore,
};

/// PE ownership retained until a native query lease completes.
pub struct GuestQueryLease {
    owner: Arc<VisibilityQueryCore>,
    cells: LeaseCells,
}

impl GuestQueryLease {
    #[must_use]
    pub fn new(owner: Arc<VisibilityQueryCore>) -> Self {
        Self {
            owner,
            cells: LeaseCells::default(),
        }
    }

    #[must_use]
    pub fn new_pooled(owner: Arc<VisibilityQueryCore>, pool: &CompletionPool) -> Self {
        Self {
            owner,
            cells: LeaseCells::Pooled(pool.allocate(false)),
        }
    }

    #[must_use]
    pub fn token(&self) -> Option<u64> {
        self.cells.token()
    }

    #[must_use]
    pub fn into_slot(self) -> Option<CompletionSlot> {
        self.cells.into_slot()
    }

    /// Describe a single adoption while retaining this lease until completion.
    #[must_use]
    pub fn descriptor(&self) -> GuestQueryDescriptor {
        GuestQueryDescriptor {
            mailbox: self.owner.mailbox_address(),
            completion: core::ptr::from_ref(self.cells.completion()) as u64,
        }
    }

    #[must_use]
    pub fn completed(&self) -> bool {
        self.cells.reusable()
    }

    /// Cancel a rejected publication before native adoption.
    ///
    /// # Safety
    ///
    /// No native consumer adopted or can later access this descriptor. Cancellation runs once.
    pub unsafe fn cancel_unadopted(&self) {
        self.cells.completion().publish();
    }
}

/// Fixed-width addresses of a query mailbox and its independent lease acknowledgment.
///
/// Each descriptor is adopted once. Repeated operations use distinct leases even when their
/// mailbox addresses agree. No Arc representation or destructor crosses the boundary.
pub struct GuestQueryDescriptor {
    mailbox: u64,
    completion: u64,
}

impl GuestQueryDescriptor {
    /// Exact fixed-width descriptor fields used by packet publication validation.
    #[must_use]
    pub const fn wire_fields(&self) -> [u64; 2] {
        [self.mailbox, self.completion]
    }

    const fn into_parts(self) -> (u64, u64) {
        (self.mailbox, self.completion)
    }

    fn validate(&self) -> Result<(), WireError> {
        if !valid_cell::<QueryMailbox>(self.mailbox)
            || !valid_cell::<LeaseCompletion>(self.completion)
        {
            return Err(WireError::InvalidValue);
        }
        Ok(())
    }
}

impl WireValue for GuestQueryDescriptor {
    const MIN_WIRE_BYTES: usize = 2 * size_of::<u64>();

    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u64(self.mailbox)?;
        writer.u64(self.completion)
    }

    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let value = Self {
            mailbox: reader.u64()?,
            completion: reader.u64()?,
        };
        value.validate()?;
        Ok(value)
    }
}

/// Native weak identity cache owned by one encoder device.
///
/// Weak entries never retain a guest lease. Active and pending query lists own the native cores;
/// the first adopted lease pins each mailbox until those lists release their final reference.
#[derive(Default)]
pub struct QueryLeaseCache {
    cores: FxHashMap<u64, Weak<VisibilityQueryCore>>,
}

impl QueryLeaseCache {
    /// Adopt one publication, preserving native identity for an already-live mailbox.
    ///
    /// # Safety
    ///
    /// PE retains the original query core and completion cell until acknowledgment. Both addresses
    /// are initialized, live and suitably aligned. This descriptor's completion cell belongs to a
    /// distinct lease not previously adopted or canceled. No native runtime owns a PE Arc.
    ///
    /// # Errors
    ///
    /// Rejects invalid numeric address ranges before accessing either cell.
    pub unsafe fn adopt(
        &mut self,
        descriptor: GuestQueryDescriptor,
    ) -> Result<Arc<VisibilityQueryCore>, WireError> {
        descriptor.validate()?;
        let (mailbox, completion_address) = descriptor.into_parts();
        if let Some(core) = self.cores.get(&mailbox).and_then(Weak::upgrade) {
            // SAFETY: this distinct lease's cell is retained by PE. The upgraded native core
            // already pins the original mailbox through its first lease, so this lease can end.
            let completion =
                unsafe { InPtr::<LeaseCompletion>::new(completion_address as *const _) };
            completion.publish();
            return Ok(core);
        }
        // SAFETY: the caller retains this initialized completion cell through the native core.
        let completion = unsafe { LeaseCompletionPtr::new(completion_address) };
        // SAFETY: the retained PE lease owns the aligned mailbox until this native core drops.
        let core = Arc::new(unsafe { VisibilityQueryCore::from_guest(mailbox, completion) });
        self.cores.insert(mailbox, Arc::downgrade(&core));
        Ok(core)
    }

    /// Remove dead identities at a frame boundary without retaining live query cores.
    pub fn maintain(&mut self) {
        self.cores.retain(|_, core| core.strong_count() != 0);
    }
}

fn valid_cell<T>(address: u64) -> bool {
    let Ok(address) = usize::try_from(address) else {
        return false;
    };
    address != 0
        && address.is_multiple_of(align_of::<T>())
        && address.checked_add(size_of::<T>()).is_some()
}

#[cfg(test)]
mod tests;
