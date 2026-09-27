//! Native frame ownership around an immutable canonical metadata view.

use std::ptr::NonNull;

use mtld3d_shared::frame_metadata::FrameMetadata;

use super::{ReplayCompletion, metadata::FrameView};
use crate::scratch::ScratchArena;

/// Native allocations and the immutable metadata lease retained through final submit reads.
pub struct NativeFrame {
    header: NonNull<FrameMetadata>,
    scratch: ScratchArena,
    pending_vbib_retentions: Vec<crate::encoder_data::PendingVbibRetention>,
    // Last: publish only after every native scratch reference has become unreachable.
    pub(super) replay_completion: Option<ReplayCompletion>,
}

// SAFETY: the metadata is immutable under the packet lease; native scratch moves exclusively
// from the encoder to submit, and no encoder work accesses it after that transfer.
unsafe impl Send for NativeFrame {}

impl NativeFrame {
    /// The validated metadata and all referenced regions remain immutable until completion.
    pub(super) unsafe fn new(view: &FrameView<'_>) -> Self {
        Self {
            header: NonNull::from(view.header()),
            scratch: ScratchArena::new(),
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

    pub fn retain_vbib(&mut self, entry: crate::encoder_data::PendingVbibRetention) {
        self.pending_vbib_retentions.push(entry);
    }

    pub fn take_vbib_retentions(&mut self) -> Vec<crate::encoder_data::PendingVbibRetention> {
        core::mem::take(&mut self.pending_vbib_retentions)
    }

    /// Access native snapshot storage without changing its ownership.
    ///
    /// # Safety
    /// Do not clear, replace or release the arena while any decoded snapshot refers to it.
    pub const unsafe fn scratch_mut(&mut self) -> &mut ScratchArena {
        &mut self.scratch
    }
}
