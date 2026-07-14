//! Typed handles for Metal protocol objects crossing the PE/unix boundary.
//!
//! Metal protocol pointers (`id<MTLDevice>`, `id<MTLTexture>`, …) ride
//! across the FFI seam as `u64`. Each PE-side wire field is logically
//! tagged with which protocol it holds, but the wire type itself is
//! untyped. [`MetalHandle`] is a `#[repr(transparent)]` newtype over
//! `u64` carrying a marker tag in its `PhantomData`, so the unix side
//! can recover the protocol identity at compile time and convert to a
//! `Retained<ProtocolObject<dyn …>>` via a safe method (see
//! `unix/src/metal/handle.rs`).
//!
//! The constructor is `unsafe`: the caller asserts the supplied `u64`
//! is either zero or the address of a previously-retained
//! `id<K::Real>`. The conversion method on the unix side is **safe** —
//! the invariant rides on the type.
//!
//! No `objc2-*` dependency lives in this crate; marker types are plain
//! ZSTs. The protocol-to-marker table lives on the unix side where the
//! Metal bindings are visible.

use core::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

/// Typed Metal protocol-object handle. Wire-compatible with `u64`.
#[repr(transparent)]
pub struct MetalHandle<K>(u64, PhantomData<*const K>);

impl<K> MetalHandle<K> {
    /// The null handle.
    pub const NULL: Self = Self(0, PhantomData);

    /// Tag a raw wire `u64` with the protocol identity it holds.
    ///
    /// # Safety
    ///
    /// Caller asserts: `raw` is `0` OR the address of a previously-retained
    /// `id<K::Real>` (per the unix-side `ToMetalProtocol` mapping). The
    /// retain on that object is logically transferred into this handle —
    /// the unix side will bump it again via `Retained::retain` when
    /// converting to an objc2 `Retained`, but the original
    /// retain must remain live until the PE side drops this handle.
    #[must_use]
    pub const unsafe fn new(raw: u64) -> Self {
        Self(raw, PhantomData)
    }

    /// Raw `u64` representation.
    ///
    /// Used for logging and wire-format reads where the typed identity
    /// has already been consumed (e.g. ABI shims that hand this on to
    /// other handlers untyped).
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl<K> Clone for MetalHandle<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for MetalHandle<K> {}

// SAFETY: `MetalHandle<K>` is a `#[repr(transparent)]` newtype over `u64`
// — the `PhantomData<*const K>` exists purely for type-tagging and never
// holds a real pointer. The u64 itself is a Metal protocol-object address
// that the unix encoder thread, PE API thread, and any worker thread
// hand across in `Send`-bounded channels and closures by design.
unsafe impl<K> Send for MetalHandle<K> {}
// SAFETY: as the `Send` impl above — `MetalHandle<K>` is a transparent
// `u64` newtype; concurrent reads of the wire value are race-free.
unsafe impl<K> Sync for MetalHandle<K> {}

impl<K> Default for MetalHandle<K> {
    fn default() -> Self {
        Self::NULL
    }
}

impl<K> PartialEq for MetalHandle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K> Eq for MetalHandle<K> {}

impl<K> Hash for MetalHandle<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<K> fmt::Debug for MetalHandle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalHandle<{}>({:#x})",
            core::any::type_name::<K>(),
            self.0
        )
    }
}

impl<K> fmt::LowerHex for MetalHandle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

// Marker tags. Plain ZSTs; the protocol-to-marker mapping lives on the
// unix side (`ToMetalProtocol` trait in `unix/src/metal/handle.rs`).
pub struct MTLDeviceKind;
pub struct MTLTextureKind;
pub struct MTLBufferKind;
pub struct MTLCommandQueueKind;
pub struct MTLCommandBufferKind;
pub struct MTLRenderPipelineStateKind;
pub struct MTLDepthStencilStateKind;
pub struct MTLSamplerStateKind;
pub struct MTLLibraryKind;
pub struct MTLFunctionKind;
pub struct CAMetalLayerKind;
/// Marker tag for an `NSView` handle.
///
/// C-managed by macdrv; has no `ToMetalProtocol` impl. Exists for
/// compile-time slot-safety on the PE side.
pub struct NSViewKind;

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        assert!(core::mem::size_of::<MetalHandle<MTLDeviceKind>>() == core::mem::size_of::<u64>());
        assert!(core::mem::size_of::<MetalHandle<MTLTextureKind>>() == core::mem::size_of::<u64>());
        assert!(
            core::mem::align_of::<MetalHandle<MTLDeviceKind>>() == core::mem::align_of::<u64>()
        );
    };

    #[test]
    fn null_is_null() {
        assert!(MetalHandle::<MTLDeviceKind>::NULL.is_null());
        assert_eq!(MetalHandle::<MTLDeviceKind>::NULL.raw(), 0);
    }

    #[test]
    fn non_null_round_trip() {
        // SAFETY: in tests we never dereference; the value is opaque.
        let h = unsafe { MetalHandle::<MTLDeviceKind>::new(0x1234_5678_9abc_def0) };
        assert!(!h.is_null());
        assert_eq!(h.raw(), 0x1234_5678_9abc_def0);
    }

    #[test]
    fn default_is_null() {
        let h: MetalHandle<MTLDeviceKind> = MetalHandle::default();
        assert!(h.is_null());
    }

    #[test]
    fn lower_hex_matches_raw() {
        // SAFETY: opaque value, not dereferenced.
        let h = unsafe { MetalHandle::<MTLTextureKind>::new(0xdead_beef) };
        assert_eq!(format!("{h:#x}"), "0xdeadbeef");
    }
}
