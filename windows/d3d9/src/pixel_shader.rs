use core::ffi::c_void;

use mtld3d_core::ids::ProgramId;
use mtld3d_shared::InPtr;
use mtld3d_types::{Guid, IDirect3DPixelShader9Vtbl};

use super::{
    D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, E_NOINTERFACE, com_ref::ComUnknown,
    device::DeviceInner,
};

static DIRECT3D_PIXEL_SHADER9_VTBL: IDirect3DPixelShader9Vtbl = IDirect3DPixelShader9Vtbl {
    query_interface: ps_query_interface,
    add_ref: ps_add_ref,
    release: ps_release,
    get_device: ps_get_device,
    get_function: ps_get_function,
};

#[repr(C)]
pub struct Direct3DPixelShader9 {
    vtbl: *const IDirect3DPixelShader9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut PixelShaderInner,
}

impl Direct3DPixelShader9 {
    pub fn new(
        device_inner: *mut DeviceInner,
        shader_id: ProgramId,
        max_const_used: u32,
        uses_bump_env: bool,
    ) -> Self {
        let inner = Box::into_raw(Box::new(PixelShaderInner {
            device_inner,
            shader_id,
            max_const_used,
            uses_bump_env,
        }));
        Self {
            vtbl: &raw const DIRECT3D_PIXEL_SHADER9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    pub const fn vtbl(&self) -> &IDirect3DPixelShader9Vtbl {
        // SAFETY: `self.vtbl` is the `'static` `DIRECT3D_PIXEL_SHADER9_VTBL`
        // installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn shader_id(&self) -> ProgramId {
        self.inner().shader_id
    }

    pub fn max_const_used(&self) -> u32 {
        self.inner().max_const_used
    }

    pub fn uses_bump_env(&self) -> bool {
        self.inner().uses_bump_env
    }

    fn inner(&self) -> &PixelShaderInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `ps_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }
}

struct PixelShaderInner {
    device_inner: *mut DeviceInner,
    shader_id: ProgramId,
    max_const_used: u32,
    uses_bump_env: bool,
}

#[inline]
fn ps_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DPixelShader9 per ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DPixelShader9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::PixelShader)
}

extern "system" fn ps_query_interface(
    this: *mut c_void,
    _riid: *const Guid,
    _ppv: *mut *mut c_void,
) -> i32 {
    let _timer = ps_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DPixelShader9::QueryInterface → E_NOINTERFACE");
    E_NOINTERFACE
}

extern "system" fn ps_add_ref(this: *mut c_void) -> u32 {
    let _timer = ps_timer(this);
    // SAFETY: IDirect3DPixelShader9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DPixelShader9>(this) }
}

extern "system" fn ps_release(this: *mut c_void) -> u32 {
    let _timer = ps_timer(this);
    // SAFETY: IDirect3DPixelShader9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DPixelShader9>(this) }
}

/// Destroy a `Direct3DPixelShader9` wrapper once `refcount` and `private_refcount` reach zero.
///
/// Frees the inner + outer allocations; the parsed DXSO program lives on
/// the encoder thread and is keyed by `shader_id`, so no encoder-thread
/// cleanup is needed here.
///
/// # Safety
///
/// `this` must point to a live `Direct3DPixelShader9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_pixel_shader(this: *mut Direct3DPixelShader9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(PixelShaderInner)` from `Self::new` and no other
    // reference can survive.
    drop(unsafe { Box::from_raw(inner_ptr) });
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DPixelShader9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DPixelShader9 {
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
            unsafe { finalize_pixel_shader(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DPixelShader9 {
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
        unsafe { finalize_pixel_shader(this) };
    }
}

extern "system" fn ps_get_device(this: *mut c_void, _device: *mut *mut c_void) -> i32 {
    let _timer = ps_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DPixelShader9::GetDevice → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn ps_get_function(
    this: *mut c_void,
    _data: *mut c_void,
    _size_of_data: *mut u32,
) -> i32 {
    let _timer = ps_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DPixelShader9::GetFunction → NOTAVAILABLE");
    D3DERR_NOTAVAILABLE
}
