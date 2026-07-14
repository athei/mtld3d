//! Unix-side bridge between [`MetalHandle<K>`] and the real `objc2_metal` protocol types.
//!
//! The handle is a wire-side `u64` tagged with a marker.
//!
//! [`MetalHandle::new`] is `unsafe` (caller asserts the wire `u64` is
//! either zero or a retained `id<K::Real>`); the [`into_retained`]
//! conversion below is **safe** because the invariant rides on the
//! [`MetalHandle`] type.

use mtld3d_shared::{
    MetalHandle,
    mtl_handle::{
        CAMetalLayerKind, MTLBufferKind, MTLCommandBufferKind, MTLCommandQueueKind,
        MTLDepthStencilStateKind, MTLDeviceKind, MTLFunctionKind, MTLLibraryKind,
        MTLRenderPipelineStateKind, MTLSamplerStateKind, MTLTextureKind,
    },
};
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDepthStencilState, MTLDevice, MTLFunction,
    MTLLibrary, MTLRenderPipelineState, MTLSamplerState, MTLTexture,
};
use objc2_quartz_core::CAMetalLayer;

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
