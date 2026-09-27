//! Per-upload feedback leases with retry bookkeeping owned entirely by PE.

use std::sync::Arc;

use mtld3d_shared::{
    InPtr,
    encoder_wire::{LeaseCompletion, WireError, WireReader, WireWriter},
    upload_feedback::{UploadFeedback, UploadOutcome},
};
use rustc_hash::FxHashMap;

use super::{EmittedUpload, RedirtyEntry, RedirtyQueue};
use crate::{
    encoder_value::WireValue,
    guest_completions::{CompletionPool, CompletionSlot, LeaseCells},
};

/// Original upload metadata and queue retained until native feedback is complete.
pub struct GuestRedirtyLease {
    owner: Arc<RedirtyQueue>,
    declined: Option<RedirtyEntry>,
    emitted: Option<EmittedUpload>,
    feedback: FeedbackStorage,
    events: LeaseCells,
    final_cells: LeaseCells,
}

impl GuestRedirtyLease {
    #[must_use]
    pub fn new(owner: Arc<RedirtyQueue>, declined: RedirtyEntry, emitted: EmittedUpload) -> Self {
        Self::with_cells(
            owner,
            declined,
            emitted,
            LeaseCells::default(),
            LeaseCells::default(),
        )
    }

    #[must_use]
    pub fn new_pooled(
        owner: Arc<RedirtyQueue>,
        declined: RedirtyEntry,
        emitted: EmittedUpload,
        pool: &CompletionPool,
    ) -> Self {
        Self::with_cells(
            owner,
            declined,
            emitted,
            LeaseCells::Pooled(pool.allocate(true)),
            LeaseCells::Pooled(pool.allocate(false)),
        )
    }

    fn with_cells(
        owner: Arc<RedirtyQueue>,
        declined: RedirtyEntry,
        emitted: EmittedUpload,
        events: LeaseCells,
        final_cells: LeaseCells,
    ) -> Self {
        let sequence = core::ptr::from_ref(&owner.feedback_sequence) as u64;
        let emitted_notification = core::ptr::from_ref(events.acquired()) as u64;
        let declined_notification = core::ptr::from_ref(events.completion()) as u64;
        let feedback = if events.token().is_some() {
            owner
                .feedback_records
                .lock()
                .expect("feedback record pool poisoned")
                .allocate(sequence, emitted_notification, declined_notification)
        } else {
            FeedbackStorage::Standalone(Box::new(UploadFeedback::new(
                sequence,
                emitted_notification,
                declined_notification,
            )))
        };
        Self {
            owner,
            declined: Some(declined),
            emitted: Some(emitted),
            feedback,
            events,
            final_cells,
        }
    }

    #[must_use]
    pub fn tokens(&self) -> [Option<u64>; 2] {
        [self.events.token(), self.final_cells.token()]
    }

    /// Return completion slots and recycle their associated feedback record.
    ///
    /// # Panics
    ///
    /// Panics if a pooled record still has unconsumed notifications or its pool is poisoned.
    #[must_use]
    pub fn into_slots(self) -> [Option<CompletionSlot>; 2] {
        if let FeedbackStorage::Pooled { index, .. } = self.feedback {
            assert!(
                self.events.reusable() && self.final_cells.reusable(),
                "feedback notifications consumed"
            );
            self.owner
                .feedback_records
                .lock()
                .expect("feedback record pool poisoned")
                .free
                .push(index);
        }
        [self.events.into_slot(), self.final_cells.into_slot()]
    }

    #[must_use]
    pub fn descriptor(&self) -> GuestRedirtyDescriptor {
        GuestRedirtyDescriptor {
            feedback: core::ptr::from_ref(self.feedback.cell()) as u64,
            completion: core::ptr::from_ref(self.final_cells.completion()) as u64,
        }
    }

    /// Apply published native events in queue order; report when the lease can be reclaimed.
    ///
    /// # Panics
    ///
    /// Panics if the original queue's mutex is poisoned.
    pub fn maintain(&mut self) -> bool {
        let complete = self.final_cells.reusable();
        let (emitted, declined) = self.feedback.cell().events();
        if emitted != 0
            && let Some(value) = self.emitted.take()
        {
            self.apply(emitted, FeedbackEvent::Emitted(value));
        }
        if declined != 0
            && let Some(value) = self.declined.take()
        {
            self.apply(declined, FeedbackEvent::Declined(value));
        }
        complete && self.events.reusable()
    }

    fn apply(&self, sequence: u64, event: FeedbackEvent) {
        let mut order = self
            .owner
            .feedback_order
            .lock()
            .expect("upload feedback order poisoned");
        order.apply(&self.owner, sequence, event);
    }

    /// Restore the dirty state after an upload publication was rejected before adoption.
    ///
    /// # Safety
    ///
    /// No native consumer adopted or can subsequently access this descriptor.
    ///
    /// # Panics
    ///
    /// Panics if the original queue's mutex is poisoned while restoring the dirty state.
    pub unsafe fn cancel_unadopted(&mut self) {
        // SAFETY: this PE lease retains its queue's counter and cannot race native publication.
        unsafe { self.feedback.cell().publish(&UploadOutcome::Declined) };
        // SAFETY: cancellation excludes native consumers and retains all notification cells.
        unsafe { self.feedback.cell().finish_notifications() };
        self.final_cells.completion().publish();
        self.maintain();
    }
}

/// Two fixed-width addresses, never a PE queue or Arc representation.
pub struct GuestRedirtyDescriptor {
    feedback: u64,
    completion: u64,
}

impl GuestRedirtyDescriptor {
    #[must_use]
    pub const fn wire_fields(&self) -> [u64; 2] {
        [self.feedback, self.completion]
    }

    /// Reconstruct one native-only proxy retaining no PE-owned Rust object.
    ///
    /// # Safety
    ///
    /// PE retains both initialized aligned cells and the feedback cell's queue counter until
    /// completion. The descriptor is adopted exactly once. Native writes cease before its final
    /// queue reference publishes completion.
    ///
    /// # Errors
    ///
    /// Rejects invalid numeric address ranges before dereferencing either cell.
    pub unsafe fn adopt(self) -> Result<Arc<RedirtyQueue>, WireError> {
        self.validate()?;
        let mut queue = RedirtyQueue::new();
        queue.native_feedback = Some(NativeFeedback { descriptor: self });
        Ok(Arc::new(queue))
    }

    fn validate(&self) -> Result<(), WireError> {
        for (address, length) in [
            (self.feedback, size_of::<UploadFeedback>()),
            (self.completion, size_of::<LeaseCompletion>()),
        ] {
            let address = usize::try_from(address).map_err(|_| WireError::TooLarge)?;
            if address == 0 || !address.is_multiple_of(8) || address.checked_add(length).is_none() {
                return Err(WireError::InvalidValue);
            }
        }
        Ok(())
    }
}

impl WireValue for GuestRedirtyDescriptor {
    const MIN_WIRE_BYTES: usize = 16;
    fn write_wire(&self, writer: &mut WireWriter<'_>) -> Result<(), WireError> {
        writer.u64(self.feedback)?;
        writer.u64(self.completion)
    }
    fn read_wire(reader: &mut WireReader<'_>) -> Result<Self, WireError> {
        let value = Self {
            feedback: reader.u64()?,
            completion: reader.u64()?,
        };
        value.validate()?;
        Ok(value)
    }
}

pub struct NativeFeedback {
    descriptor: GuestRedirtyDescriptor,
}

impl NativeFeedback {
    pub fn publish(&self, outcome: &UploadOutcome) {
        // SAFETY: adopt retains this aligned initialized PE cell until this proxy's final drop.
        let feedback =
            unsafe { InPtr::<UploadFeedback>::new(self.descriptor.feedback as *const _) };
        // SAFETY: adoption pins the original queue counter through final proxy destruction.
        unsafe { feedback.publish(outcome) };
    }
}

impl Drop for NativeFeedback {
    fn drop(&mut self) {
        // SAFETY: adoption pins this initialized cell until the final completion publication.
        let feedback =
            unsafe { InPtr::<UploadFeedback>::new(self.descriptor.feedback as *const _) };
        if feedback.events() == (0, 0) {
            // SAFETY: the queue counter remains pinned, and no proxy reader survives final drop.
            unsafe { feedback.publish(&UploadOutcome::Declined) };
        }
        // SAFETY: the PE lease retains both event notifications through consumption; no further
        // publication is possible after the final native proxy reference is destroyed.
        unsafe { feedback.finish_notifications() };
        // SAFETY: the PE lease retains this aligned initialized cell until the store below.
        let completion =
            unsafe { InPtr::<LeaseCompletion>::new(self.descriptor.completion as *const _) };
        completion.publish();
    }
}

pub struct FeedbackOrder {
    next: u64,
    pending: FxHashMap<u64, FeedbackEvent>,
}

impl FeedbackOrder {
    fn apply(&mut self, owner: &RedirtyQueue, sequence: u64, event: FeedbackEvent) {
        self.pending.insert(sequence, event);
        loop {
            let next = self.next;
            let Some(event) = self.pending.remove(&next) else {
                break;
            };
            self.next = next
                .checked_add(1)
                .expect("upload feedback sequence exhausted");
            match event {
                FeedbackEvent::Emitted(value) => owner.note_emitted(value),
                FeedbackEvent::Declined(value) => {
                    let subresource = value.subresource;
                    if !owner.decline(value) {
                        mtld3d_shared::log_once_warn_by!(target: crate::LOG_TARGET,
                            key: subresource.texture_id.raw(),
                            "texture {:#x} subresource {} exhausted its upload retry budget",
                            subresource.texture_id.raw(), subresource.index);
                    }
                }
            }
        }
    }

    pub fn new() -> Self {
        Self {
            next: 1,
            pending: FxHashMap::default(),
        }
    }
}

enum FeedbackEvent {
    Emitted(EmittedUpload),
    Declined(RedirtyEntry),
}

#[derive(Default)]
pub struct FeedbackRecords {
    blocks: Vec<Arc<FeedbackBlock>>,
    free: Vec<usize>,
    allocated: usize,
}

impl FeedbackRecords {
    fn allocate(&mut self, sequence: u64, emitted: u64, declined: u64) -> FeedbackStorage {
        let index = self.free.pop().unwrap_or_else(|| {
            let index = self.allocated;
            self.allocated += 1;
            index
        });
        let block_index = index / 128;
        if block_index == self.blocks.len() {
            self.blocks.push(Arc::new(FeedbackBlock {
                records: std::array::from_fn(|_| UploadFeedback::new(0, 0, 0)),
            }));
        }
        let block = Arc::clone(&self.blocks[block_index]);
        // SAFETY: this record is new or returned only after its event and final cells were consumed.
        unsafe { block.records[index % 128].reset(sequence, emitted, declined) };
        FeedbackStorage::Pooled { index, block }
    }
}

struct FeedbackBlock {
    records: [UploadFeedback; 128],
}

enum FeedbackStorage {
    Standalone(Box<UploadFeedback>),
    Pooled {
        index: usize,
        block: Arc<FeedbackBlock>,
    },
}

impl FeedbackStorage {
    fn cell(&self) -> &UploadFeedback {
        match self {
            Self::Standalone(value) => value,
            Self::Pooled { index, block } => &block.records[index % 128],
        }
    }
}
