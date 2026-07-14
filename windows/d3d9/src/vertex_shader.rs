use core::ffi::c_void;

use mtld3d_core::{convert::InputSemantic, ids::ProgramId};
use mtld3d_shared::InPtr;
use mtld3d_types::{Guid, IDirect3DVertexShader9Vtbl};

use super::{
    D3DERR_INVALIDCALL, D3DERR_NOTAVAILABLE, E_NOINTERFACE, com_ref::ComUnknown,
    device::DeviceInner,
};

static DIRECT3D_VERTEX_SHADER9_VTBL: IDirect3DVertexShader9Vtbl = IDirect3DVertexShader9Vtbl {
    query_interface: vs_query_interface,
    add_ref: vs_add_ref,
    release: vs_release,
    get_device: vs_get_device,
    get_function: vs_get_function,
};

/// PE-side wrapper for `IDirect3DVertexShader9`.
///
/// The parsed DXSO program lives on the encoder thread
/// (`FrameEncoder::program_cache`, keyed on `shader_id`). The wrapper itself
/// only records the identity bits the API path needs to read synchronously.
#[repr(C)]
pub struct Direct3DVertexShader9 {
    vtbl: *const IDirect3DVertexShader9Vtbl,
    refcount: u32,
    /// Device-internal "bound slot" refcount, kept in sync by `CachedComPtr<_, Bound>`.
    ///
    /// The wrapper is destroyed only when both `refcount` and
    /// `private_refcount` reach zero.
    private_refcount: u32,
    inner: *mut VertexShaderInner,
}

impl Direct3DVertexShader9 {
    pub fn new(
        device_inner: *mut DeviceInner,
        shader_id: ProgramId,
        max_const_used: u32,
        uses_rel_const: bool,
        uses_int_const: bool,
        input_semantics: Vec<InputSemantic>,
    ) -> Self {
        let mut flags = VsConstUsage::empty();
        flags.set(VsConstUsage::USES_REL_CONST, uses_rel_const);
        flags.set(VsConstUsage::USES_INT_CONST, uses_int_const);
        let inner = Box::into_raw(Box::new(VertexShaderInner {
            device_inner,
            shader_id,
            max_const_used,
            flags,
            input_semantics,
        }));
        Self {
            vtbl: &raw const DIRECT3D_VERTEX_SHADER9_VTBL,
            refcount: 1,
            private_refcount: 0,
            inner,
        }
    }

    pub const fn vtbl(&self) -> &IDirect3DVertexShader9Vtbl {
        // SAFETY: `self.vtbl` is the `'static` `DIRECT3D_VERTEX_SHADER9_VTBL`
        // installed at `Self::new`.
        unsafe { &*self.vtbl }
    }

    pub fn shader_id(&self) -> ProgramId {
        self.inner().shader_id
    }

    pub fn max_const_used(&self) -> u32 {
        self.inner().max_const_used
    }

    /// `true` when the shader reads constants via relative addressing (`c[a0.x + N]`).
    ///
    /// Those draws upload the full populated constant prefix instead of just
    /// `max_const_used + 1` rows — the static bound from `max_const_reg` is
    /// blind to a0-indexed reads into the bone palette that
    /// `SetVertexShaderConstantF` populates.
    pub fn uses_rel_const(&self) -> bool {
        self.inner().flags.contains(VsConstUsage::USES_REL_CONST)
    }

    /// `true` when the shader reads a *dynamic* integer constant.
    ///
    /// That constant is a non-`defi` `iN`, typically a `loop`/`rep` counter fed
    /// by `SetVertexShaderConstantI`. Draws binding such a shader upload + bind
    /// the integer-constant buffer (vertex slot 14); every other draw skips it.
    pub fn uses_int_const(&self) -> bool {
        self.inner().flags.contains(VsConstUsage::USES_INT_CONST)
    }

    pub fn input_semantics(&self) -> &[InputSemantic] {
        &self.inner().input_semantics
    }

    fn inner(&self) -> &VertexShaderInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `vs_release` at refcount
        // zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }
}

bitflags::bitflags! {
    /// Constant-addressing modes a compiled VS uses.
    ///
    /// Gates the per-draw constant-buffer upload shape.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct VsConstUsage: u8 {
        /// Reads constants via relative addressing (`c[a0.x + N]`).
        ///
        /// Such draws upload the full populated constant prefix instead of
        /// just `max_const_used + 1` rows.
        const USES_REL_CONST = 1 << 0;
        /// Reads a *dynamic* integer constant.
        ///
        /// That is, a non-`defi` `iN`, typically a `loop`/`rep` counter fed by
        /// `SetVertexShaderConstantI`; such draws upload + bind the
        /// integer-constant buffer (vertex slot 14).
        const USES_INT_CONST = 1 << 1;
    }
}

struct VertexShaderInner {
    device_inner: *mut DeviceInner,
    shader_id: ProgramId,
    max_const_used: u32,
    flags: VsConstUsage,
    input_semantics: Vec<InputSemantic>,
}

#[inline]
fn vs_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DVertexShader9 per ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DVertexShader9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::VertexShader)
}

extern "system" fn vs_query_interface(
    this: *mut c_void,
    _riid: *const Guid,
    _ppv: *mut *mut c_void,
) -> i32 {
    let _timer = vs_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexShader9::QueryInterface → E_NOINTERFACE");
    E_NOINTERFACE
}

extern "system" fn vs_add_ref(this: *mut c_void) -> u32 {
    let _timer = vs_timer(this);
    // SAFETY: IDirect3DVertexShader9 IUnknown AddRef thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DVertexShader9>(this) }
}

extern "system" fn vs_release(this: *mut c_void) -> u32 {
    let _timer = vs_timer(this);
    // SAFETY: IDirect3DVertexShader9 IUnknown Release thunk; the D3D9 ABI
    // guarantees `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DVertexShader9>(this) }
}

/// Destroy a `Direct3DVertexShader9` wrapper.
///
/// Runs once both `refcount` and `private_refcount` have reached zero. Frees
/// the inner + outer allocations; the parsed DXSO program lives on the encoder
/// thread and is keyed by `shader_id`, so no encoder-thread cleanup is needed
/// here.
///
/// # Safety
///
/// `this` must point to a live `Direct3DVertexShader9` wrapper with both
/// counters at zero; caller must not access the wrapper afterwards.
unsafe fn finalize_vertex_shader(this: *mut Direct3DVertexShader9) {
    // SAFETY: caller asserts wrapper still live; both counters at zero
    // means no other reference can be outstanding.
    let obj = unsafe { &*this };
    let inner_ptr = obj.inner;
    // SAFETY: both counters reached zero; `inner_ptr` is the original
    // `Box::into_raw(VertexShaderInner)` from `Self::new` and no
    // other reference can survive.
    drop(unsafe { Box::from_raw(inner_ptr) });
    // SAFETY: both counters reached zero; `this` is the original
    // `Box::into_raw(Direct3DVertexShader9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

impl ComUnknown for Direct3DVertexShader9 {
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
            unsafe { finalize_vertex_shader(this) };
        }
    }
}

// SAFETY: `refcount_mut`/`private_refcount` expose this wrapper's own counters;
// `finalize` frees it exactly once when both reach zero.
unsafe impl crate::com_ref::ComChild for Direct3DVertexShader9 {
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
        unsafe { finalize_vertex_shader(this) };
    }
}

extern "system" fn vs_get_device(this: *mut c_void, _device: *mut *mut c_void) -> i32 {
    let _timer = vs_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexShader9::GetDevice → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn vs_get_function(
    this: *mut c_void,
    _data: *mut c_void,
    _size_of_data: *mut u32,
) -> i32 {
    let _timer = vs_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DVertexShader9::GetFunction → NOTAVAILABLE");
    D3DERR_NOTAVAILABLE
}
