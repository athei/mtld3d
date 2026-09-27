//! Atomic visibility-query state shared with the guest API.
//!
//! Every field has a fixed width and alignment across PE and native architectures.

use core::sync::atomic::{AtomicU32, AtomicU64};

/// One query's publication state, pinned until its native lease is released.
#[repr(C, align(8))]
pub struct QueryMailbox {
    pub issue_generation: AtomicU64,
    pub seq_begin: AtomicU64,
    pub seq_end: AtomicU64,
    pub carried: AtomicU64,
    pub accumulated: AtomicU64,
    pub logical_area: AtomicU64,
    pub render_area: AtomicU64,
    pub draws_at_begin: AtomicU64,
    pub draws_in_span: AtomicU64,
    pub offset_begin: AtomicU32,
    pub status: AtomicU32,
    pub end_requested: AtomicU32,
    pub uncounted: AtomicU32,
    pub requested_generation: AtomicU64,
    pub result_generation: AtomicU64,
    pub end_generation: AtomicU64,
}

impl QueryMailbox {
    /// An unissued query with no pending segments.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            issue_generation: AtomicU64::new(0),
            seq_begin: AtomicU64::new(0),
            seq_end: AtomicU64::new(0),
            carried: AtomicU64::new(0),
            accumulated: AtomicU64::new(0),
            logical_area: AtomicU64::new(0),
            render_area: AtomicU64::new(0),
            draws_at_begin: AtomicU64::new(0),
            draws_in_span: AtomicU64::new(0),
            offset_begin: AtomicU32::new(0),
            status: AtomicU32::new(0),
            end_requested: AtomicU32::new(0),
            uncounted: AtomicU32::new(0),
            requested_generation: AtomicU64::new(0),
            result_generation: AtomicU64::new(0),
            end_generation: AtomicU64::new(0),
        }
    }
}

impl Default for QueryMailbox {
    fn default() -> Self {
        Self::new()
    }
}

const _: () = {
    assert!(size_of::<QueryMailbox>() == 112);
    assert!(align_of::<QueryMailbox>() == 8);
    assert!(core::mem::offset_of!(QueryMailbox, issue_generation) == 0);
    assert!(core::mem::offset_of!(QueryMailbox, seq_begin) == 8);
    assert!(core::mem::offset_of!(QueryMailbox, seq_end) == 16);
    assert!(core::mem::offset_of!(QueryMailbox, carried) == 24);
    assert!(core::mem::offset_of!(QueryMailbox, accumulated) == 32);
    assert!(core::mem::offset_of!(QueryMailbox, logical_area) == 40);
    assert!(core::mem::offset_of!(QueryMailbox, render_area) == 48);
    assert!(core::mem::offset_of!(QueryMailbox, draws_at_begin) == 56);
    assert!(core::mem::offset_of!(QueryMailbox, draws_in_span) == 64);
    assert!(core::mem::offset_of!(QueryMailbox, offset_begin) == 72);
    assert!(core::mem::offset_of!(QueryMailbox, status) == 76);
    assert!(core::mem::offset_of!(QueryMailbox, end_requested) == 80);
    assert!(core::mem::offset_of!(QueryMailbox, uncounted) == 84);
    assert!(core::mem::offset_of!(QueryMailbox, requested_generation) == 88);
    assert!(core::mem::offset_of!(QueryMailbox, result_generation) == 96);
    assert!(core::mem::offset_of!(QueryMailbox, end_generation) == 104);
};
