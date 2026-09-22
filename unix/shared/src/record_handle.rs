//! The handle that names one device's unix-side record on the wire.
//!
//! `CreateCommandQueue` builds a record on the unix side for the device it
//! creates: the command queue and its retain, the presentation state, and
//! the presenter thread. The PE side never looks inside it. It keeps the
//! handle on `DeviceInner`, hands it back on every later thunk that acts on
//! that device, and returns it with `DestroyCommandQueue`, which is what
//! frees the record.
//!
//! It replaces the `MTLCommandQueue` address that used to identify a device
//! on the wire, so the unix side resolves a device by dereferencing what the
//! caller holds rather than by looking an address up in a process-wide map,
//! and no Metal object pointer is exposed to the PE side to outlive its
//! owner.

use core::fmt;

/// Opaque identity of one device's unix-side record. Wire-compatible with `u64`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DeviceRecordHandle(u64);

impl DeviceRecordHandle {
    /// The null handle: a device with no record, before create or after destroy.
    pub const NULL: Self = Self(0);

    /// Tag a raw wire `u64` as a record handle.
    ///
    /// # Safety
    ///
    /// Caller asserts: `raw` is `0`, or a value a unix-side
    /// `DeviceRecord::into_handle` produced whose matching
    /// `DestroyCommandQueue` has not run. The unix side dereferences it.
    #[must_use]
    pub const unsafe fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw `u64` representation, for logging and the unix-side deref.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

// SAFETY: a `#[repr(transparent)]` newtype over `u64`. The value is an
// address the unix side owns; the PE side only stores and forwards it, and
// it travels between the API, encoder and submit threads by design.
unsafe impl Send for DeviceRecordHandle {}
// SAFETY: as the `Send` impl above: concurrent reads of a wire `u64` are
// race-free, and the record behind it is internally synchronised.
unsafe impl Sync for DeviceRecordHandle {}

impl fmt::LowerHex for DeviceRecordHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}
