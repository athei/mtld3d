//! `IDirect3DVertexDeclaration9` COM wrapper.
//!
//! Stores the raw element array the game passed to
//! `CreateVertexDeclaration`; the per-VS attribute resolution happens at
//! draw time in `snapshot_shared` via `convert::resolve_attrs_for_vs` /
//! `resolve_attrs_for_ff`.
//!
//! Multi-stream declarations are accepted (D3D9 creation validates structure
//! only), but only stream 0 is rendered — `resolve_attrs_for_vs` /
//! `resolve_attrs_for_ff` drop elements on other streams. `WoW` is
//! single-stream; full multi-stream VB wiring in the pipeline descriptor is a
//! bigger change deferred until a target workload needs it.

use core::ffi::c_void;

use log::trace;
use mtld3d_core::convert::pack_vertex_decl;
use mtld3d_shared::InPtr;
use mtld3d_types::{D3DVERTEXELEMENT9, Guid, IDirect3DVertexDeclaration9Vtbl};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET, com_ref::ComUnknown,
    device::DeviceInner, null_out,
};

static DIRECT3D_VERTEX_DECLARATION9_VTBL: IDirect3DVertexDeclaration9Vtbl =
    IDirect3DVertexDeclaration9Vtbl {
        query_interface: vd_query_interface,
        add_ref: vd_add_ref,
        release: vd_release,
        get_device: vd_get_device,
        get_declaration: vd_get_declaration,
    };

#[repr(C)]
pub struct Direct3DVertexDeclaration9 {
    vtbl: *const IDirect3DVertexDeclaration9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut VertexDeclInner,
}

pub struct VertexDeclInner {
    device_inner: *mut DeviceInner,
    /// Elements as received from the game.
    ///
    /// Terminator included, so `GetDeclaration` can memcpy the whole slice
    /// back.
    elements_with_end: Vec<D3DVERTEXELEMENT9>,
    hash: u64,
}

impl VertexDeclInner {
    /// Real elements, excluding the `D3DDECL_END` terminator.
    pub fn elements(&self) -> &[D3DVERTEXELEMENT9] {
        let n = self.elements_with_end.len().saturating_sub(1);
        &self.elements_with_end[..n]
    }

    pub fn elements_with_end(&self) -> &[D3DVERTEXELEMENT9] {
        &self.elements_with_end
    }

    pub const fn hash(&self) -> u64 {
        self.hash
    }
}

pub struct VertexDeclCreateInfo<'a> {
    pub device_inner: *mut DeviceInner,
    pub elements: &'a [D3DVERTEXELEMENT9],
}

impl Direct3DVertexDeclaration9 {
    /// Build a wrapper from a slice that already includes the `D3DDECL_END` terminator.
    ///
    /// Returns `None` only if the slice has no terminator. Multi-stream
    /// layouts are accepted (only stream 0 renders).
    pub fn new(info: &VertexDeclCreateInfo<'_>) -> Option<Self> {
        let (elements_with_end, hash) = pack_vertex_decl(info.elements)?;
        let inner = Box::into_raw(Box::new(VertexDeclInner {
            device_inner: info.device_inner,
            elements_with_end,
            hash,
        }));
        Some(Self {
            vtbl: &raw const DIRECT3D_VERTEX_DECLARATION9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        })
    }

    pub const fn vtbl(&self) -> &IDirect3DVertexDeclaration9Vtbl {
        // SAFETY: `self.vtbl` is the `'static`
        // `DIRECT3D_VERTEX_DECLARATION9_VTBL` installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn inner(&self) -> &VertexDeclInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `vd_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }
}

#[inline]
fn vdecl_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexDeclaration9 per ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DVertexDeclaration9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::VertexDecl)
}

extern "system" fn vd_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _timer = vdecl_timer(this);
    // SAFETY: vtable in-param; `riid` is *const Guid per IUnknown::QueryInterface ABI.
    let riid_lo = (unsafe { InPtr::<Guid>::opt(riid.cast()) }).map_or(0, |g| g.data1);
    trace!(target: LOG_TARGET, "IDirect3DVertexDeclaration9::QueryInterface(riid_lo={riid_lo:#010x})");
    null_out(ppv);
    E_NOINTERFACE
}

extern "system" fn vd_add_ref(this: *mut c_void) -> u32 {
    let _timer = vdecl_timer(this);
    // SAFETY: IDirect3DVertexDeclaration9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DVertexDeclaration9>(this) }
}

extern "system" fn vd_release(this: *mut c_void) -> u32 {
    let _timer = vdecl_timer(this);
    // SAFETY: IDirect3DVertexDeclaration9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DVertexDeclaration9>(this) }
}

/// Destroy a `Direct3DVertexDeclaration9` wrapper once both counters have reached zero.
///
/// Frees the inner + outer allocations; no encoder-thread or registry
/// interaction is required.
///
/// # Safety
///
/// `this` must point to a live `Direct3DVertexDeclaration9` wrapper with
/// both counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_vertex_decl(this: *mut Direct3DVertexDeclaration9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(VertexDeclInner)` from `Self::new` and no other
    // reference can survive.
    drop(unsafe { Box::from_raw(inner_ptr) });
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DVertexDeclaration9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DVertexDeclaration9 {
    fn vtbl_add_ref(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().add_ref
    }
    fn vtbl_release(&self) -> unsafe extern "system" fn(*mut c_void) -> u32 {
        self.vtbl().release
    }
    fn private_refcount_inc(&mut self) {
        self.private_refcount += 1;
    }
    unsafe fn private_refcount_dec_maybe_finalize(this: *mut Self) {
        // SAFETY: caller asserts `this` points to a live wrapper with
        // at least one private refcount outstanding.
        let obj = unsafe { &mut *this };
        obj.private_refcount -= 1;
        if obj.refcount == 0 && obj.private_refcount == 0 {
            // SAFETY: both counters reached zero — no other reference
            // can survive; finalize takes exclusive ownership.
            unsafe { finalize_vertex_decl(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DVertexDeclaration9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn private_refcount(&self) -> u32 {
        self.private_refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        crate::device::device_wrapper_from(self.inner().device_inner)
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — both counters are zero.
        unsafe { finalize_vertex_decl(this) };
    }
}

extern "system" fn vd_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _timer = vdecl_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexDeclaration9::GetDevice → INVALIDCALL");
    null_out(device);
    D3DERR_INVALIDCALL
}

extern "system" fn vd_get_declaration(
    this: *mut c_void,
    out_elements: *mut D3DVERTEXELEMENT9,
    num_elements: *mut u32,
) -> i32 {
    let _timer = vdecl_timer(this);
    if num_elements.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexDeclaration9 per ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DVertexDeclaration9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let src = obj.inner().elements_with_end();
    let n = u32::try_from(src.len()).expect("D3D9 vertex decl ≤ 16 elements");
    // SAFETY: `num_elements` is non-null (checked above) and per the D3D9
    // ABI points to a writable `u32` slot owned by the caller.
    unsafe { *num_elements = n };
    if !out_elements.is_null() {
        // SAFETY: per D3D9 spec, `out_elements` points to a buffer sized
        // for at least `num_elements` `D3DVERTEXELEMENT9` entries; `src`
        // is a live slice on the wrapper's inner. Source and destination
        // do not alias.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), out_elements, src.len());
        }
    }
    D3D_OK
}
