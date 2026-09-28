//! Native submission inputs borrowed only while the submit worker owns the frame payload.

use std::{
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use mtld3d_shared::{
    BlitCommand, MetalHandle, PassDescriptor,
    mtl::SnapshotFlags,
    mtl_handle::{CAMetalLayerKind, MTLTextureKind, NSViewKind},
    perf::SubmitTimings,
};

/// A device-owned retirement atomic, including copies held by GPU callbacks.
///
/// Copying this capability does not acquire ownership. The constructor requires
/// the device lifecycle to drain all uses before freeing the atomic.
#[derive(Clone, Copy)]
pub struct RetirementCounter(Option<NonZeroU64>);

impl RetirementCounter {
    pub const NONE: Self = Self(None);

    /// Wrap a device-owned atomic address, or zero for an absent counter.
    ///
    /// # Safety
    /// A nonzero address must name an aligned live `AtomicU64`. Its owner must
    /// retain it through every copied capability, queued submission, registered
    /// command buffer and completion callback, including failed-submit cleanup.
    pub const unsafe fn from_address(address: u64) -> Self {
        if address == 0 {
            Self::NONE
        } else {
            Self(NonZeroU64::new(address))
        }
    }

    /// Identity used by the existing pending-buffer map and control waits.
    pub const fn address(self) -> u64 {
        match self.0 {
            Some(address) => address.get(),
            None => 0,
        }
    }

    pub const fn is_present(self) -> bool {
        self.0.is_some()
    }

    /// Observe retirement, or zero when this submission has no counter.
    pub fn load(self) -> u64 {
        self.0.map_or(0, |address| {
            // SAFETY: construction keeps the atomic live through all uses.
            unsafe { &*(address.get() as *const AtomicU64) }.load(Ordering::Acquire)
        })
    }

    /// Publish completed or failed work before an acquiring API observation.
    pub fn publish(self, seq: u64) {
        if let Some(address) = self.0 {
            // SAFETY: construction keeps the atomic live through all uses.
            unsafe { &*(address.get() as *const AtomicU64) }.fetch_max(seq, Ordering::Release);
        }
    }
}

/// Owned scalar state queued beside the existing owned frame payload.
pub struct SubmitDescription {
    pub blit_commands_need_encoder: bool,
    pub upload_pass_count: usize,
    pub present_layer: MetalHandle<CAMetalLayerKind>,
    pub present_texture: MetalHandle<MTLTextureKind>,
    pub present_view: MetalHandle<NSViewKind>,
    pub submit_seq: u64,
    pub draw_retirement: RetirementCounter,
    pub upload_retirement: RetirementCounter,
    pub failed_submission: RetirementCounter,
}

/// Arrays borrowed from the payload only during native command replay.
pub struct FrameSubmission<'a> {
    pub description: &'a SubmitDescription,
    pub blits: &'a [BlitCommand],
    pub passes: &'a [PassDescriptor],
}

/// Native submission result, with elapsed timings in nanoseconds.
pub struct SubmissionOutcome {
    pub success: bool,
    pub drawable_wait_ns: u64,
    pub present_wait_ns: u64,
    pub snapshot_flags: SnapshotFlags,
    pub timings: SubmitTimings,
}

impl SubmissionOutcome {
    pub const fn new() -> Self {
        Self {
            success: false,
            drawable_wait_ns: 0,
            present_wait_ns: 0,
            snapshot_flags: SnapshotFlags::empty(),
            timings: SubmitTimings::new(),
        }
    }
}

impl Default for SubmissionOutcome {
    fn default() -> Self {
        Self::new()
    }
}
