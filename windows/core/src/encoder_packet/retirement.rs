//! Device-side retirement of submitted frame packets and the resource leases they hand over.
//!
//! A pass consumes the device's completion queue before it walks the retained packets, so a
//! packet whose replay completion has just arrived gives up its recording storage and its
//! leases in that same pass. A handed-over lease is registered by completion token and
//! retires once its final notification has been consumed. Every slot a pass retires goes
//! back to the pool at the end of the pass, under one allocator lock.
//!
//! Native publishers only ever write the queue, so an empty queue means a pass would find
//! nothing new; `CompletionPool::has_ready` lets the owner skip the pass without a lock.

use crate::{
    encoder_packet::{FramePacket, FrameRecorder, PacketLease},
    guest_completions::{CompletionDrain, CompletionPool, CompletionSlot, REPLAY_COMPLETION_TOKEN},
    guest_pages::LeaseOnlyPages,
    scratch::ScratchArena,
};

/// Notifications one drain consumes at most; the rest stay queued for the next pass.
const DRAIN_BUDGET: usize = 4096;

/// Recovered recording storages kept for the frames that follow.
const RETAINED_STORAGE: usize = 2;

/// The device operations a maintenance pass calls out to.
pub trait RetirementHooks {
    /// Native decoding rejected an admitted packet.
    fn packet_rejected(&mut self);
    /// Cancel a shader registration that native replay did not adopt.
    fn cancel_registration(&mut self, registration: u64);
}

/// One device's retained packets, the leases they handed over and recovered recording storage.
#[derive(Default)]
pub struct PacketRetirement {
    pending: Vec<FramePacket>,
    leases: LeaseRegistry,
    storage: Vec<(ScratchArena, FrameRecorder)>,
}

impl PacketRetirement {
    /// Retain a packet until native replay and every lease it hands over have finished.
    pub fn push(&mut self, packet: FramePacket) {
        self.pending.push(packet);
    }

    /// Retain a packet just submitted, and maintain at once when nothing else will do it soon.
    ///
    /// A `synchronous` submission returns only after native code has released what it
    /// retires, and its caller counts on those owners being freed on return, so it is
    /// maintained here. So is a packet whose replay completion a drain on another thread
    /// already consumed: its notification is no longer on the queue to announce it.
    /// Otherwise the next frame's pass picks the packet up, as for every `Present`.
    pub fn push_submitted(
        &mut self,
        packet: FramePacket,
        synchronous: bool,
        pool: &CompletionPool,
        hooks: &mut impl RetirementHooks,
    ) {
        let consumed = packet.replay_consumed();
        self.pending.push(packet);
        if synchronous || consumed {
            self.maintain(pool, hooks);
        }
    }

    /// Recording storage a finished packet gave up, for the next frame to record into.
    pub fn take_storage(&mut self) -> Option<(ScratchArena, FrameRecorder)> {
        self.storage.pop()
    }

    /// Retire everything the native side has finished with, in one pass.
    ///
    /// The drain comes first. A packet whose replay completion it consumed then hands its
    /// leases to the registry, where one whose final notification was consumed as well
    /// retires at once, and gives up its recording storage. A rejected packet reports
    /// through `hooks`, and so does every registration a packet's replay never adopted.
    pub fn maintain(&mut self, pool: &CompletionPool, hooks: &mut impl RetirementHooks) {
        self.leases.drain(pool);
        let Self {
            pending,
            leases,
            storage,
        } = self;
        pending.retain_mut(|packet| {
            for lease in packet.take_leases() {
                leases.insert(lease);
            }
            if packet.was_rejected() {
                hooks.packet_rejected();
            }
            for registration in packet.take_rejected_registrations() {
                hooks.cancel_registration(registration);
            }
            if let Some(recovered) = packet.take_recording_storage()
                && storage.len() < RETAINED_STORAGE
            {
                storage.push(recovered);
            }
            !packet.maintain()
        });
        leases.recycle_retired(pool);
    }

    /// Cancel every retained packet and consume every notification left on the queue.
    ///
    /// # Safety
    ///
    /// Native destruction has joined every worker and retired every GPU reference: no
    /// decoder, replay, worker or command buffer can reach a retained owner, and no native
    /// publisher remains.
    pub unsafe fn cancel_after_quiescence(&mut self, pool: &CompletionPool) {
        for packet in &mut self.pending {
            // SAFETY: the caller guarantees that every native user of the packet has ended.
            unsafe { packet.cancel_unadopted() };
        }
        // Consume every cancellation notification before dropping the packet that owns it.
        while self.leases.drain(pool) == DRAIN_BUDGET {}
        self.pending.clear();
        self.leases.recycle_retired(pool);
        self.leases.entries.clear();
        self.leases.aliases.clear();
    }

    /// Tally every page lease this device still retains, handed over to the registry or not.
    pub fn tally_page_leases(&self, tally: &mut LeaseOnlyPages) {
        for lease in self.pending.iter().flat_map(|packet| &packet.pages) {
            tally.add(lease);
        }
        for lease in self.leases.entries.iter().flatten() {
            if let PacketLease::Page(lease) = lease {
                tally.add(lease);
            }
        }
    }

    /// Leak every owner native code might still reach, after native destruction failed.
    pub fn forget_native_owners(&mut self) {
        core::mem::forget(core::mem::take(&mut self.pending));
        core::mem::forget(core::mem::take(&mut self.leases));
    }
}

/// Handed-over leases by completion token, awaiting their final notification.
#[derive(Default)]
struct LeaseRegistry {
    entries: Vec<Option<PacketLease>>,
    aliases: Vec<Option<usize>>,
    drain: CompletionDrain,
    /// Slots retired since the last recycle, returned to the pool together.
    retired: Vec<CompletionSlot>,
}

impl LeaseRegistry {
    fn insert(&mut self, mut lease: PacketLease) {
        if lease.maintain() {
            self.retired
                .extend(lease.into_slots().into_iter().flatten());
            return;
        }
        let tokens = lease.tokens();
        let primary = usize::try_from(tokens[0].expect("pooled packet lease"))
            .expect("local slot token fits usize");
        if self.entries.len() <= primary {
            self.entries.resize_with(primary + 1, || None);
        }
        assert!(
            self.entries[primary].is_none(),
            "completion owner is unique"
        );
        for token in tokens.into_iter().flatten() {
            let token = usize::try_from(token).expect("local slot token fits usize");
            if self.aliases.len() <= token {
                self.aliases.resize(token + 1, None);
            }
            assert!(self.aliases[token].is_none(), "completion alias is unique");
            self.aliases[token] = Some(primary);
        }
        self.entries[primary] = Some(lease);
    }

    fn drain(&mut self, pool: &CompletionPool) -> usize {
        let Self {
            entries,
            aliases,
            drain,
            retired,
        } = self;
        pool.drain(drain, DRAIN_BUDGET, |event| {
            // A packet's replay completion is consumed here and read from its own cell by
            // the packet walk that follows.
            if event == REPLAY_COMPLETION_TOKEN {
                return;
            }
            let Ok(token) = usize::try_from(event / 2) else {
                return;
            };
            let Some(primary) = aliases.get(token).copied().flatten() else {
                return;
            };
            let Some(entry) = entries.get_mut(primary) else {
                return;
            };
            if entry.as_mut().is_some_and(PacketLease::maintain)
                && let Some(lease) = entry.take()
            {
                for token in lease.tokens().into_iter().flatten() {
                    let token = usize::try_from(token).expect("registered local token");
                    aliases[token] = None;
                }
                retired.extend(lease.into_slots().into_iter().flatten());
            }
            // Events preceding replay completion leave their consumed state in the
            // retained cells. Inserting the lease checks that state once.
        })
    }

    fn recycle_retired(&mut self, pool: &CompletionPool) {
        pool.recycle_all(&mut self.retired);
    }
}

#[cfg(test)]
mod tests;
