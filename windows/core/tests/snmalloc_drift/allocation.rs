//! Keeps PE probe allocations separate from the system-allocated test harness.

use std::{
    alloc::{GlobalAlloc, Layout},
    ptr::NonNull,
};

use snmalloc_rs::SnMalloc;

pub struct SnmallocAllocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl SnmallocAllocation {
    #[must_use]
    pub fn new(layout: Layout) -> Self {
        assert_ne!(layout.size(), 0, "probe allocation must be nonempty");
        let ptr = SnMalloc
            .alloc_aligned(layout)
            .expect("probe allocation failed");
        Self { ptr, layout }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Free once and return the address only for the subsequent `VirtualQuery`.
    #[must_use]
    pub fn release(self) -> *const u8 {
        let ptr = self.ptr.as_ptr().cast_const();
        drop(self);
        ptr
    }
}

impl Drop for SnmallocAllocation {
    fn drop(&mut self) {
        // SAFETY: this owner holds the live SnMalloc allocation and its exact
        // original layout. It cannot be cloned, and release consumes the owner.
        unsafe { SnMalloc.dealloc(self.ptr.as_ptr(), self.layout) };
    }
}
