//! One D3D device's unix-side record: the queue it renders on and its presentation state.
//!
//! `CreateCommandQueue` builds the record, hands the PE side a
//! [`DeviceRecordHandle`] for it, and every later thunk that acts on that
//! device names it with that handle. `DestroyCommandQueue` gives the handle
//! back, which retires the presenter thread and drops the record, releasing
//! the queue with it. Nothing indexes a device by address: what the caller
//! holds is the record.
//!
//! The record is shared: the API thread creates and destroys it, the submit
//! thread pushes frames onto it, its own presenter thread consumes them, and
//! Metal's completion and presented handlers touch what they captured. An
//! `Arc` counts those references, and the presenter's own `Arc` is what
//! keeps the record alive between the handle's release and the thread's
//! last instruction.

use std::{path::PathBuf, sync::Arc};

use mtld3d_shared::{
    mtl_handle::{MTLCommandQueueKind, MetalHandle},
    record_handle::DeviceRecordHandle,
};

use super::{
    handle::ReleaseRetain,
    presenter::{PresentState, Presented},
    upscale::UpscaleCache,
};

/// Everything one device owns on the unix side.
pub struct DeviceRecord {
    /// One retain on the device's `MTLCommandQueue`, held for the record's life.
    ///
    /// Taken by `create_command_queue` before the presenter thread exists,
    /// so the thread's own retain at start lands on a live object, and
    /// released by [`Drop`] once every reference to the record is gone,
    /// which is after the thread was joined.
    queue: MetalHandle<MTLCommandQueueKind>,
    present: PresentState,
    presented: Presented,
    upscale: UpscaleCache,
}

impl Drop for DeviceRecord {
    fn drop(&mut self) {
        // SAFETY: the handle holds the retain `create_command_queue` took and
        // no other copy of it is used; the presenter thread, which is the
        // only other user of the queue, has ended, since it held an `Arc` of
        // this record while it ran.
        unsafe { self.queue.release_retain() };
    }
}

impl DeviceRecord {
    /// Build the record for a queue whose retain it takes over.
    pub fn new(queue: MetalHandle<MTLCommandQueueKind>, gate: Option<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            queue,
            present: PresentState::new(gate),
            presented: Presented::new(),
            upscale: UpscaleCache::new(),
        })
    }

    /// The device's command queue. Every submit, readback and present uses it.
    pub const fn queue(&self) -> MetalHandle<MTLCommandQueueKind> {
        self.queue
    }

    /// The device's presentation state: its packets, slots and presenter thread.
    pub const fn present(&self) -> &PresentState {
        &self.present
    }

    /// The device's presented-cadence probe.
    pub const fn presented(&self) -> &Presented {
        &self.presented
    }

    /// The device's `MetalFX` scalers and scratch targets.
    pub const fn upscale(&self) -> &UpscaleCache {
        &self.upscale
    }

    /// Hand the record to the PE side as an opaque handle.
    ///
    /// Consumes one `Arc` reference into the handle, which the matching
    /// [`Self::consume`] takes back. The address is the record's, so a
    /// handle that comes back names the record it was made from.
    pub fn into_handle(self: Arc<Self>) -> DeviceRecordHandle {
        let raw = Arc::into_raw(self) as u64;
        // SAFETY: `raw` is the address `Arc::into_raw` just produced, and
        // the reference it counts is now owned by the handle.
        unsafe { DeviceRecordHandle::new(raw) }
    }

    /// Borrow the record a handle names, for the duration of one thunk.
    ///
    /// `None` for the null handle, which is a device whose creation failed
    /// or one already destroyed; every caller warns once and returns.
    ///
    /// # Safety
    ///
    /// Caller asserts `handle` is null or came from [`Self::into_handle`]
    /// and has not been through [`Self::consume`].
    pub unsafe fn borrow(handle: DeviceRecordHandle) -> Option<Arc<Self>> {
        if handle.is_null() {
            return None;
        }
        let ptr = handle.raw() as *const Self;
        // SAFETY: per the contract above, `ptr` is a live `Arc` allocation
        // this module produced; the increment pairs with the `Arc` the
        // reconstruction below drops at the end of this function, leaving
        // the handle's own reference untouched.
        unsafe { Arc::increment_strong_count(ptr) };
        // SAFETY: the count was just incremented for exactly this `Arc`.
        Some(unsafe { Arc::from_raw(ptr) })
    }

    /// Take the record back from the handle, ending the PE side's reference.
    ///
    /// `None` for the null handle. The record drops once the returned `Arc`
    /// and every other reference to it are gone.
    ///
    /// # Safety
    ///
    /// As [`Self::borrow`], and the caller asserts no later thunk will pass
    /// this handle: `DestroyCommandQueue` is the one caller.
    pub unsafe fn consume(handle: DeviceRecordHandle) -> Option<Arc<Self>> {
        if handle.is_null() {
            return None;
        }
        // SAFETY: per the contract above, the handle owns one reference
        // produced by `into_handle`, which this reconstruction takes over.
        Some(unsafe { Arc::from_raw(handle.raw() as *const Self) })
    }
}
