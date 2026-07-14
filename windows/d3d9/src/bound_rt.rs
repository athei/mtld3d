//! Bound render-target + depth-stencil surfaces owned by `DeviceInner`.
//!
//! Keeps COM `AddRef`/`Release` pairing correct for `SetRenderTarget` and
//! `SetDepthStencilSurface`. Actual pass-break decisions happen on the
//! encoder thread via closures pushed from those handlers; this struct
//! just owns the API-thread-side surface references.

use crate::{
    com_ref::{Bound, CachedComPtr},
    surface::Direct3DSurface9,
};

pub struct BoundRt {
    /// Bound render-target slot.
    ///
    /// Uses the `Bound` ownership marker — swaps bump the wrapper's
    /// `private_refcount` inline.
    render_target: CachedComPtr<Direct3DSurface9, Bound>,
    rt_width: u32,
    rt_height: u32,
    /// Bound depth-stencil slot.
    ///
    /// Same `Bound` semantics as `render_target` above.
    depth_stencil: CachedComPtr<Direct3DSurface9, Bound>,
}

impl BoundRt {
    pub const fn new(backbuffer_width: u32, backbuffer_height: u32) -> Self {
        Self {
            render_target: CachedComPtr::null(),
            rt_width: backbuffer_width,
            rt_height: backbuffer_height,
            depth_stencil: CachedComPtr::null(),
        }
    }

    /// Swap the bound render target, handling COM ref counts.
    ///
    /// `new` may be null (which disables the bound slot). `width`/`height`
    /// track the new surface dimensions for callers that query the bound
    /// RT's size.
    pub fn replace_render_target(&mut self, new: *mut Direct3DSurface9, width: u32, height: u32) {
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; the
        // caller (the SetRenderTarget thunk) guarantees it is either null
        // or a valid *mut Direct3DSurface9.
        self.render_target = unsafe { CachedComPtr::adopt(new) };
        self.rt_width = width;
        self.rt_height = height;
    }

    /// Swap the bound depth/stencil surface (COM-ref-correct).
    ///
    /// `new` may be null to clear the slot. Tracks the pointer but does not
    /// redirect the actual render pass's depth attachment.
    pub fn replace_depth_stencil(&mut self, new: *mut Direct3DSurface9) {
        // SAFETY: `new` came from the IDirect3DDevice9 vtable layer; same
        // contract as `replace_render_target`.
        self.depth_stencil = unsafe { CachedComPtr::adopt(new) };
    }

    /// Release and null both surface slots.
    ///
    /// Used from the device release path.
    pub fn teardown(&mut self) {
        self.render_target = CachedComPtr::null();
        self.depth_stencil = CachedComPtr::null();
        self.rt_width = 0;
        self.rt_height = 0;
    }

    /// Currently bound depth-stencil surface, or null when the app has not bound one.
    ///
    /// Null means the device default depth-stencil is in effect. Read on the
    /// API thread to derive the pipeline's depth/stencil formats from the
    /// surface's actual format rather than the device default — the two
    /// diverge when `WoW` renders to a custom-sized RT with its own matching
    /// depth-stencil.
    pub const fn depth_stencil(&self) -> *mut Direct3DSurface9 {
        self.depth_stencil.raw()
    }

    /// Currently bound render-target surface, or null when no custom RT has been set.
    ///
    /// Null means the device default backbuffer is in effect. Read by
    /// `GetRenderTarget` to hand back the *actual* surface instead of
    /// fabricating an empty standalone — fabricating one breaks every
    /// downstream consumer (`StretchRect`, mid-frame readback) that needs a
    /// Metal handle.
    pub const fn render_target(&self) -> *mut Direct3DSurface9 {
        self.render_target.raw()
    }
}
