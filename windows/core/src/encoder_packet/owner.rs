use mtld3d_shared::encoder_wire::{LeaseCompletion, WireError};

use super::{FrameRecorder, metadata::MetadataStorage};
use crate::{
    encoder_data::FrameData,
    guest_pages::{GuestOwnedPageLease, GuestPageLease},
    guest_queries::GuestQueryLease,
    scratch::ScratchArena,
    upload_redirty::GuestRedirtyLease,
};

bitflags::bitflags! {
    #[derive(Default)]
    struct PacketFlags: u8 {
        const ADMITTED = 1;
        const CANCELLED = 2;
    }
}

/// Resource lease moved into the device completion registry after admission.
pub enum PacketLease {
    Page(GuestPageLease),
    OwnedPage(GuestOwnedPageLease),
    Query(GuestQueryLease),
    Redirty(GuestRedirtyLease),
}

impl PacketLease {
    #[must_use]
    pub fn tokens(&self) -> [Option<u64>; 2] {
        match self {
            Self::Page(lease) => [lease.token(), None],
            Self::OwnedPage(lease) => [Some(lease.token()), None],
            Self::Query(lease) => [lease.token(), None],
            Self::Redirty(lease) => lease.tokens(),
        }
    }
    /// Apply the notified acquisition or retirement; true permits recycling.
    #[must_use]
    pub fn maintain(&mut self) -> bool {
        match self {
            Self::Page(lease) => lease.maintain(),
            Self::OwnedPage(lease) => lease.completed(),
            Self::Query(lease) => lease.completed(),
            Self::Redirty(lease) => lease.maintain(),
        }
    }
    #[must_use]
    pub fn into_slots(self) -> [Option<crate::guest_completions::CompletionSlot>; 2] {
        match self {
            Self::Page(lease) => [lease.into_slot(), None],
            Self::OwnedPage(lease) => [Some(lease.into_slot()), None],
            Self::Query(lease) => [lease.into_slot(), None],
            Self::Redirty(lease) => lease.into_slots(),
        }
    }
}

/// PE-owned frame bytes and leases retained until native acknowledgment.
pub struct FramePacket {
    metadata: MetadataStorage,
    completion_pool: crate::guest_completions::CompletionPool,
    pub(super) frame: Option<FrameData>,
    pub(super) recorder: Option<FrameRecorder>,
    pub(super) pages: Vec<GuestPageLease>,
    pub(super) owned_pages: Vec<GuestOwnedPageLease>,
    pub(super) queries: Vec<GuestQueryLease>,
    redirties: Vec<GuestRedirtyLease>,
    completion: Option<Box<LeaseCompletion>>,
    flags: PacketFlags,
    recording_error: Option<WireError>,
}

impl FramePacket {
    /// Seal metadata after operations have already been recorded.
    ///
    /// # Panics
    ///
    /// Panics if the newly constructed packet has no frame or recorder.
    ///
    /// # Errors
    ///
    /// Returns the error and retained packet. Failed recording may own backing still used
    /// by an earlier native frame, so the returned owner must survive until native quiescence.
    pub fn new(mut frame: FrameData) -> Result<Self, (WireError, Box<Self>)> {
        let mut recorder = frame.recorder.take().unwrap_or_default();
        #[cfg(not(windows))]
        for op in core::mem::take(&mut frame.ops) {
            recorder.record(&mut frame.scratch, op);
        }
        // The replay completion is queued on the device's pool with the lease notifications,
        // so the one queue tells maintenance whether anything has arrived.
        let completion = recorder.completion_pool.replay_completion();
        let mut packet = Self {
            completion_pool: recorder.completion_pool.clone(),
            metadata: core::mem::take(&mut recorder.metadata),
            frame: Some(frame),
            recorder: Some(recorder),
            pages: Vec::new(),
            owned_pages: Vec::new(),
            queries: Vec::new(),
            redirties: Vec::new(),
            completion: Some(completion),
            flags: PacketFlags::empty(),
            recording_error: None,
        };
        let result = {
            let Self {
                metadata,
                frame,
                recorder,
                ..
            } = &mut packet;
            let recorder = recorder.as_mut().expect("new packet owns recorder");
            recorder.error.take().map_or_else(
                || metadata.seal(frame.as_mut().expect("new packet owns frame"), recorder),
                Err,
            )
        };
        if let Some(recorder) = &mut packet.recorder {
            packet.pages = core::mem::take(&mut recorder.pages);
            packet.owned_pages = core::mem::take(&mut recorder.owned_pages);
            packet.queries = core::mem::take(&mut recorder.queries);
            packet.redirties = core::mem::take(&mut recorder.redirties);
        }
        match result {
            Ok(()) => Ok(packet),
            Err(error) => {
                packet.recording_error = Some(error);
                Err((error, Box::new(packet)))
            }
        }
    }

    #[must_use]
    pub const fn metadata_bytes(&self) -> &[u8] {
        // SAFETY: this packet retains the sealed frame arena until take_recording_storage
        // moves the metadata out and clears its token before returning the arena for reuse.
        unsafe { self.metadata.as_bytes() }
    }

    #[must_use]
    pub fn operation_bytes(&self) -> &[u8] {
        self.recorder
            .as_ref()
            .map_or(&[], |recorder| recorder.slab.descriptor_bytes())
    }

    #[must_use]
    pub fn completion_address(&self) -> u64 {
        self.completion
            .as_ref()
            .map_or(0, |cell| core::ptr::from_ref(cell.as_ref()) as u64)
    }

    /// Whether a drain has already consumed this packet's replay completion.
    #[must_use]
    pub fn replay_consumed(&self) -> bool {
        self.completion
            .as_ref()
            .is_some_and(|cell| cell.is_complete())
    }

    #[must_use]
    pub fn was_rejected(&self) -> bool {
        self.completion
            .as_ref()
            .is_some_and(|cell| cell.was_rejected())
    }

    #[must_use]
    pub fn registrations(&self) -> &[u64] {
        self.recorder
            .as_ref()
            .map_or(&[], |recorder| recorder.registrations.as_slice())
    }

    /// Take registrations that native ordered replay has not adopted.
    #[must_use]
    pub fn take_rejected_registrations(&mut self) -> Vec<u64> {
        if !self.flags.contains(PacketFlags::ADMITTED) || self.was_rejected() {
            self.recorder.as_mut().map_or_else(Vec::new, |recorder| {
                core::mem::take(&mut recorder.registrations)
            })
        } else {
            Vec::new()
        }
    }

    /// Move pooled resource owners into the device registry after successful replay.
    ///
    /// Standalone test leases remain owned by this packet.
    pub fn take_leases(&mut self) -> impl Iterator<Item = PacketLease> + '_ {
        let admitted = self.flags.contains(PacketFlags::ADMITTED) && self.can_retire_leases();
        self.pages
            .extract_if(.., |lease| lease.token().is_some())
            .map(PacketLease::Page)
            .chain(
                self.owned_pages
                    .extract_if(.., |_| true)
                    .map(PacketLease::OwnedPage),
            )
            .chain(
                self.queries
                    .extract_if(.., |lease| lease.token().is_some())
                    .map(PacketLease::Query),
            )
            .chain(
                self.redirties
                    .extract_if(.., |lease| lease.tokens()[0].is_some())
                    .map(PacketLease::Redirty),
            )
            .take(if admitted { usize::MAX } else { 0 })
    }

    /// Record successful admission while preserving all published owners.
    ///
    /// # Safety
    ///
    /// Exactly one native consumer may decode this successful packet. Retain its backing
    /// until completion, or fully quiesce the native runtime before cancellation.
    pub unsafe fn mark_admitted(&mut self) {
        self.flags.insert(PacketFlags::ADMITTED);
    }

    /// Cancel a packet after native ownership has ended or before any native use exists.
    ///
    /// # Safety
    ///
    /// No native decoder, replay, cache, worker or GPU work can access any published owner.
    /// A failed recording can contain retirements of previously used pages and requires
    /// runtime quiescence even though this particular packet was never admitted.
    pub unsafe fn cancel_unadopted(&mut self) {
        for page in &mut self.pages {
            // SAFETY: the caller guarantees that all native references have ended.
            unsafe {
                page.cancel_unadopted();
            }
        }
        for page in &self.owned_pages {
            // SAFETY: the caller guarantees all native and GPU references have ended.
            unsafe { page.cancel_unadopted() };
        }
        for query in &self.queries {
            // SAFETY: the caller guarantees that all native references have ended.
            unsafe {
                query.cancel_unadopted();
            }
        }
        for redirty in &mut self.redirties {
            // SAFETY: the caller guarantees native feedback references have ended.
            unsafe {
                redirty.cancel_unadopted();
            }
        }
        if let Some(completion) = &self.completion {
            completion.publish();
        }
        self.flags.remove(PacketFlags::ADMITTED);
        self.flags.insert(PacketFlags::CANCELLED);
        self.recording_error = None;
    }

    /// Recover recording capacity after replay, independently of retained resource leases.
    #[must_use]
    pub fn take_recording_storage(&mut self) -> Option<(ScratchArena, FrameRecorder)> {
        if self.recording_error.is_some()
            || self.was_rejected()
            || !self.completion.as_ref()?.is_complete()
        {
            return None;
        }
        let mut frame = self.frame.take()?;
        let mut recorder = self.recorder.take()?;
        recorder.reset();
        // Replay completion lets owners move to the registry before storage recovery.
        // Reuse only an empty allocation, never a lease still owned by this packet, so the
        // next frame records its leases without growing these vectors again.
        if self.pages.is_empty() {
            recorder.pages = core::mem::take(&mut self.pages);
        }
        if self.owned_pages.is_empty() {
            recorder.owned_pages = core::mem::take(&mut self.owned_pages);
        }
        if self.queries.is_empty() {
            recorder.queries = core::mem::take(&mut self.queries);
        }
        if self.redirties.is_empty() {
            recorder.redirties = core::mem::take(&mut self.redirties);
        }
        recorder.metadata = core::mem::take(&mut self.metadata);
        recorder.metadata.clear();
        Some((frame.take_recording_scratch(), recorder))
    }

    /// Consume test-fixture notifications through the production device queue.
    #[cfg(test)]
    pub(super) fn drain_test_completions(
        &self,
        cursor: &mut crate::guest_completions::CompletionDrain,
    ) -> usize {
        let mut total = 0;
        loop {
            let consumed = self.completion_pool.drain(cursor, 4096, |_| {});
            total += consumed;
            if consumed < 4096 {
                return total;
            }
        }
    }

    fn can_retire_leases(&self) -> bool {
        self.recording_error.is_none()
            && self
                .completion
                .as_ref()
                .is_some_and(|cell| cell.is_complete())
            && (self.flags.contains(PacketFlags::CANCELLED)
                || (self.flags.contains(PacketFlags::ADMITTED) && !self.was_rejected()))
    }

    /// Retire resource owners whose native acknowledgments have arrived.
    #[must_use]
    pub fn maintain(&mut self) -> bool {
        if !self.can_retire_leases() {
            return false;
        }
        let replay_complete = self
            .completion
            .as_ref()
            .is_some_and(|cell| cell.is_complete());
        for page in self.pages.extract_if(.., GuestPageLease::maintain) {
            if let Some(slot) = page.into_slot() {
                self.completion_pool.recycle(slot);
            }
        }
        for page in self.owned_pages.extract_if(.., |page| page.completed()) {
            self.completion_pool.recycle(page.into_slot());
        }
        for query in self.queries.extract_if(.., |query| query.completed()) {
            if let Some(slot) = query.into_slot() {
                self.completion_pool.recycle(slot);
            }
        }
        for redirty in self.redirties.extract_if(.., GuestRedirtyLease::maintain) {
            for slot in redirty.into_slots().into_iter().flatten() {
                self.completion_pool.recycle(slot);
            }
        }
        replay_complete
            && self.pages.is_empty()
            && self.owned_pages.is_empty()
            && self.queries.is_empty()
            && self.redirties.is_empty()
    }
}

impl Drop for FramePacket {
    fn drop(&mut self) {
        if !self.maintain() {
            mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET,
                "dropping an active frame packet; retaining guest backing still in use");
            self.metadata.clear();
            core::mem::forget(self.frame.take());
            core::mem::forget(self.recorder.take());
            core::mem::forget(core::mem::take(&mut self.pages));
            core::mem::forget(core::mem::take(&mut self.owned_pages));
            core::mem::forget(core::mem::take(&mut self.queries));
            core::mem::forget(core::mem::take(&mut self.redirties));
            core::mem::forget(self.completion.take());
        }
    }
}
