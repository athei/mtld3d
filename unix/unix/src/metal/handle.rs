//! Unix-side bridge between [`MetalHandle<K>`] and the real `objc2_metal` protocol types.
//!
//! The handle is a wire-side `u64` tagged with a marker.
//!
//! [`MetalHandle::new`] is `unsafe` (caller asserts the wire `u64` is
//! either zero or a retained `id<K::Real>`); the [`into_retained`]
//! conversion below is **safe** because the invariant rides on the
//! [`MetalHandle`] type.

use core::hash::Hash;

use mtld3d_shared::{
    MetalHandle,
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLCommandBufferKind, MTLCommandQueueKind,
        MTLComputePipelineStateKind, MTLDepthStencilStateKind, MTLDeviceKind, MTLFunctionKind,
        MTLLibraryKind, MTLRenderPipelineStateKind, MTLSamplerStateKind, MTLTextureKind,
    },
};
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDepthStencilState,
    MTLDevice, MTLFunction, MTLLibrary, MTLRenderPipelineState, MTLSamplerState, MTLTexture,
};
use objc2_quartz_core::CAMetalLayer;
use rustc_hash::FxHashMap;

/// Maps a wire-side marker kind to the real Metal protocol type.
pub trait ToMetalProtocol {
    type Real: ?Sized + NSObjectProtocol;
}

impl ToMetalProtocol for MTLDeviceKind {
    type Real = dyn MTLDevice;
}
impl ToMetalProtocol for MTLTextureKind {
    type Real = dyn MTLTexture;
}
impl ToMetalProtocol for MTLBufferKind {
    type Real = dyn MTLBuffer;
}
impl ToMetalProtocol for MTLCommandQueueKind {
    type Real = dyn MTLCommandQueue;
}
impl ToMetalProtocol for MTLCommandBufferKind {
    type Real = dyn MTLCommandBuffer;
}
impl ToMetalProtocol for MTLComputePipelineStateKind {
    type Real = dyn MTLComputePipelineState;
}
impl ToMetalProtocol for MTLRenderPipelineStateKind {
    type Real = dyn MTLRenderPipelineState;
}
impl ToMetalProtocol for MTLDepthStencilStateKind {
    type Real = dyn MTLDepthStencilState;
}
impl ToMetalProtocol for MTLSamplerStateKind {
    type Real = dyn MTLSamplerState;
}
impl ToMetalProtocol for MTLLibraryKind {
    type Real = dyn MTLLibrary;
}
impl ToMetalProtocol for MTLFunctionKind {
    type Real = dyn MTLFunction;
}

/// Safe protocol-handle → `Retained<ProtocolObject<dyn …>>` conversion.
///
/// The unsafe lives at [`MetalHandle::new`] (where the caller asserted
/// the wire `u64` is a retained `id<K::Real>`); this method just bumps
/// the refcount via [`Retained::retain`].
pub trait IntoRetained {
    type Object: ?Sized;
    fn into_retained(self) -> Option<Retained<ProtocolObject<Self::Object>>>;
}

impl<K: ToMetalProtocol> IntoRetained for MetalHandle<K> {
    type Object = K::Real;
    fn into_retained(self) -> Option<Retained<ProtocolObject<Self::Object>>> {
        if self.is_null() {
            return None;
        }
        // SAFETY: type invariant — `MetalHandle::new` asserted at construction
        // that `raw` is either 0 (filtered above) or a valid retained
        // `id<K::Real>`. `Retained::retain` bumps the refcount; the caller's
        // retain stays live until they drop their handle.
        unsafe { Retained::retain(self.raw() as *mut ProtocolObject<K::Real>) }
    }
}

/// `CAMetalLayer` is a concrete `objc2_quartz_core` class (not a protocol).
///
/// Its retain dance is the same shape but `Retained::retain` takes
/// `*mut CAMetalLayer` rather than `*mut ProtocolObject<dyn …>`.
pub trait IntoRetainedLayer {
    fn into_retained(self) -> Option<Retained<CAMetalLayer>>;
}

impl IntoRetainedLayer for MetalHandle<CAMetalLayerKind> {
    fn into_retained(self) -> Option<Retained<CAMetalLayer>> {
        if self.is_null() {
            return None;
        }
        // SAFETY: as `IntoRetained` above; `CAMetalLayer` is a concrete
        // class so the cast targets the class type directly.
        unsafe { Retained::retain(self.raw() as *mut CAMetalLayer) }
    }
}

/// Borrow the object a handle's canonical retain keeps alive.
///
/// The narrow companion to [`IntoRetained::into_retained`]: a replay
/// command that needs the object only for the call it is making reads
/// through the canonical retain instead of taking and dropping one of
/// its own. Implemented for the kinds whose destroy ordering against
/// the replay has been established, and for no other: every remaining
/// kind converts through the retained path, which needs no lifetime
/// argument from its caller.
///
/// # Safety
///
/// The canonical retain this handle stands for must stay live for the
/// whole of the returned reference's lifetime. `MetalHandle<K>` is
/// `Copy` and that lifetime is unconstrained, so the compiler can
/// enforce neither; the call site names the ownership that holds the
/// object up, the way the [`ReleaseRetain`] call sites name the
/// ownership they consume.
pub unsafe trait BorrowRetained {
    /// The Metal protocol the borrowed object conforms to.
    type Object: ?Sized;

    /// The object this handle addresses, or `None` when it is null.
    ///
    /// # Safety
    ///
    /// As [`BorrowRetained`].
    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<Self::Object>>;
}

/// The object `handle` addresses, read through the canonical retain.
///
/// The one dereference every [`BorrowRetained`] implementation shares.
///
/// # Safety
///
/// As [`BorrowRetained`]: the canonical retain must outlive `'a`.
const unsafe fn borrow_canonical<'a, K: ToMetalProtocol>(
    handle: MetalHandle<K>,
) -> Option<&'a ProtocolObject<K::Real>> {
    if handle.is_null() {
        return None;
    }
    // SAFETY: type invariant. `MetalHandle::new` asserted at construction
    // that `raw` is either 0 (filtered above) or a valid retained
    // `id<K::Real>`, and the caller asserted that retain outlives `'a`.
    Some(unsafe { &*(handle.raw() as *const ProtocolObject<K::Real>) })
}

// SAFETY: trait contract delegates the invariant: each call site names the
// ownership that keeps the buffer alive for the borrow it takes. A buffer
// wrapper the PE side gives up is parked on the resource-retention queue
// stamped with the submit seq of the frame whose commands still name it, and
// freed only once that seq has retired on the GPU.
unsafe impl BorrowRetained for MetalHandle<MTLBufferKind> {
    type Object = dyn MTLBuffer;

    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<dyn MTLBuffer>> {
        // SAFETY: the caller's assertion carries through unchanged.
        unsafe { borrow_canonical(self) }
    }
}

// SAFETY: as the buffer impl; a texture leaves through the same seq-gated
// retention queue. The implicit surfaces that skip it are destroyed behind a
// drain of the submit thread and a GPU-idle wait, bar the back buffer's sRGB
// twin at device destroy, which only ever serves as a pass attachment and so
// is named by no command a replay reads.
unsafe impl BorrowRetained for MetalHandle<MTLTextureKind> {
    type Object = dyn MTLTexture;

    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<dyn MTLTexture>> {
        // SAFETY: the caller's assertion carries through unchanged.
        unsafe { borrow_canonical(self) }
    }
}

// SAFETY: as the buffer impl; a sampler state lives in a PE-side cache that
// never evicts, and every entry is destroyed in the encoder's shutdown, behind
// the same submit-thread drain and GPU-idle wait.
unsafe impl BorrowRetained for MetalHandle<MTLSamplerStateKind> {
    type Object = dyn MTLSamplerState;

    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<dyn MTLSamplerState>> {
        // SAFETY: the caller's assertion carries through unchanged.
        unsafe { borrow_canonical(self) }
    }
}

// SAFETY: as the sampler impl; the depth-stencil states share that cache's
// shape and that destroy path.
unsafe impl BorrowRetained for MetalHandle<MTLDepthStencilStateKind> {
    type Object = dyn MTLDepthStencilState;

    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<dyn MTLDepthStencilState>> {
        // SAFETY: the caller's assertion carries through unchanged.
        unsafe { borrow_canonical(self) }
    }
}

// SAFETY: as the sampler impl; the render pipeline states share that cache's
// shape and that destroy path.
unsafe impl BorrowRetained for MetalHandle<MTLRenderPipelineStateKind> {
    type Object = dyn MTLRenderPipelineState;

    unsafe fn borrow_retained<'a>(self) -> Option<&'a ProtocolObject<dyn MTLRenderPipelineState>> {
        // SAFETY: the caller's assertion carries through unchanged.
        unsafe { borrow_canonical(self) }
    }
}

/// Consume the canonical retain this handle stands for and release.
///
/// Use at destroy sites — the PE side has agreed to drop its only copy
/// of the handle, so we take ownership of the retain via
/// `Retained::from_raw` and drop it (decrement). The companion to
/// [`IntoRetained::into_retained`], which bumps the refcount; this
/// takes one without bumping.
///
/// # Safety
/// Caller guarantees no other live copy of this handle will be used
/// after this call returns. `MetalHandle<K>` is `Copy`, so the
/// compiler cannot enforce this — destroy paths typically queue this
/// thunk only after PE side has flushed the GPU and dropped its
/// canonical reference, satisfying the invariant by construction.
pub unsafe trait ReleaseRetain {
    /// Take the canonical retain this handle stands for and drop it.
    ///
    /// # Safety
    ///
    /// As [`ReleaseRetain`].
    unsafe fn release_retain(self);
}

// SAFETY: trait contract delegates the invariant — each call site asserts
// it holds the canonical retain and no surviving copy will be used.
unsafe impl<K: ToMetalProtocol> ReleaseRetain for MetalHandle<K> {
    unsafe fn release_retain(self) {
        if self.is_null() {
            return;
        }
        // SAFETY: invariant deferred to caller — the handle holds the
        // canonical retain on `id<K::Real>`, and no surviving copy is
        // used after this call.
        unsafe {
            drop(Retained::from_raw(
                self.raw() as *mut ProtocolObject<K::Real>
            ));
        }
    }
}

// SAFETY: as the impl above; `CAMetalLayer` is a concrete class.
unsafe impl ReleaseRetain for MetalHandle<CAMetalLayerKind> {
    unsafe fn release_retain(self) {
        if self.is_null() {
            return;
        }
        // SAFETY: invariant deferred to caller; `CAMetalLayer` is a
        // concrete class.
        unsafe {
            drop(Retained::from_raw(self.raw() as *mut CAMetalLayer));
        }
    }
}

/// Cache `built` under `key` unless an entry is already there, and return the cached handle.
///
/// The process-wide helper caches build a missing entry outside their lock,
/// so two threads that miss the same key both build one. The first stored
/// is kept; a later `built` loses and its retain is released here, where
/// dropping the `u64` would leak the object.
///
/// # Safety
///
/// `built` holds the retain `Retained::into_raw` gave it, and the caller
/// keeps no other copy of it.
pub unsafe fn keep_first<Key: Hash + Eq, K: ToMetalProtocol>(
    cache: &mut FxHashMap<Key, MetalHandle<K>>,
    key: Key,
    built: MetalHandle<K>,
) -> MetalHandle<K> {
    let kept = *cache.entry(key).or_insert(built);
    if kept.raw() != built.raw() {
        // SAFETY: the caller's assertion; `built` lost, so nothing holds it.
        unsafe { built.release_retain() };
    }
    kept
}

#[cfg(test)]
mod tests;
