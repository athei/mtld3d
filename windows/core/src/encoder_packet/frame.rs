//! Native frame ownership around an immutable canonical metadata view.

use std::ptr::NonNull;

use mtld3d_shared::frame_metadata::FrameMetadata;

use super::{ReplayCompletion, metadata::FrameView};
use crate::{guest_pages::GuestOwnedPage, ids::BufferId};

/// Native retirement of a uniquely owned PE buffer allocation.
pub struct NativeVbibRetention {
    pub buffer_id: BufferId,
    pub page_box: GuestOwnedPage,
    pub last_submit_seq: u64,
}

/// Native owners and the immutable metadata lease retained through final submit reads.
///
/// Snapshots decoded from the frame live in the encoder's own arena, which the
/// submission's payload carries and returns, so dropping this owner frees no
/// snapshot storage on the submit thread.
pub struct NativeFrame {
    header: NonNull<FrameMetadata>,
    pending_vbib_retentions: Vec<NativeVbibRetention>,
    // Last: publish only after every other native owner has been dropped.
    pub(super) replay_completion: Option<ReplayCompletion>,
}

// SAFETY: the metadata is immutable under the packet lease; the frame moves exclusively
// from the encoder to submit, and no encoder work accesses it after that transfer.
unsafe impl Send for NativeFrame {}

impl NativeFrame {
    /// The validated metadata and all referenced regions remain immutable until completion.
    pub(super) unsafe fn new(view: &FrameView<'_>) -> Self {
        Self {
            header: NonNull::from(view.header()),
            pending_vbib_retentions: Vec::new(),
            replay_completion: None,
        }
    }

    #[must_use]
    pub const fn view(&self) -> FrameView<'_> {
        // SAFETY: construction validated this header and its regions. The packet lease
        // retains them, and this borrow cannot outlive the native frame owner.
        let header = unsafe { self.header.as_ref() };
        // SAFETY: the same validated immutable lease covers every array in this header.
        unsafe { FrameView::from_validated_header(header) }
    }

    pub fn retain_vbib(&mut self, entry: NativeVbibRetention) {
        self.pending_vbib_retentions.push(entry);
    }

    pub fn take_vbib_retentions(&mut self) -> Vec<NativeVbibRetention> {
        core::mem::take(&mut self.pending_vbib_retentions)
    }
}
