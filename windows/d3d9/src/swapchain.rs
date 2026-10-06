//! `IDirect3DSwapChain9` — the implicit swapchain and the additional ones.
//!
//! `GetSwapChain(0)` hands out the device's implicit swapchain; the additional
//! ones are the objects returned by `CreateAdditionalSwapChain`.
//!
//! mtld3d drives a single `CAMetalLayer` drawable, so every swapchain's
//! `GetBackBuffer` resolves to the device's one backbuffer texture. The object
//! exists mainly to carry the present parameters (`GetPresentParameters`) and to
//! satisfy the COM lifecycle the D3D9 swapchain battery exercises.

use core::ffi::c_void;

use mtld3d_shared::{InPtr, OutPtr};
use mtld3d_types::{
    D3DDISPLAYMODE, D3DERR_DEVICENOTRESET, D3DPRESENT_PARAMETERS, D3DPRESENTSTATS, Guid,
    IDirect3DSwapChain9ExVtbl, IDirect3DSwapChain9Vtbl,
};

use super::{D3D_OK, D3DERR_INVALIDCALL, LOG_TARGET, device::DeviceInner};
use crate::surface::Direct3DSurface9;

/// The vtable every swap chain carries.
///
/// A swap chain answers `IID_IDirect3DSwapChain9Ex` only on an extended device.
pub static DIRECT3D_SWAPCHAIN9_VTBL: IDirect3DSwapChain9ExVtbl = IDirect3DSwapChain9ExVtbl {
    base: IDirect3DSwapChain9Vtbl {
        query_interface: swapchain_query_interface,
        add_ref: swapchain_add_ref,
        release: swapchain_release,
        present: swapchain_present,
        get_front_buffer_data: swapchain_get_front_buffer_data,
        get_back_buffer: swapchain_get_back_buffer,
        get_raster_status: swapchain_get_raster_status,
        get_display_mode: swapchain_get_display_mode,
        get_device: swapchain_get_device,
        get_present_parameters: swapchain_get_present_parameters,
    },
    get_last_present_count: swapchain_get_last_present_count,
    get_present_stats: swapchain_get_present_stats,
    get_display_mode_ex: swapchain_get_display_mode_ex,
};

#[repr(C)]
pub struct Direct3DSwapChain9 {
    vtbl: *const IDirect3DSwapChain9Vtbl,
    refcount: u32,
    inner: *mut SwapChainInner,
}

impl Direct3DSwapChain9 {
    fn with_owner(
        device_inner: *mut DeviceInner,
        present_params: D3DPRESENT_PARAMETERS,
        owned_by_device: bool,
    ) -> Self {
        let inner = Box::into_raw(Box::new(SwapChainInner {
            device_inner,
            present_params,
            owned_by_device,
            backbuffer_surface: 0,
            backbuffer_pins: 0,
        }));
        Self {
            vtbl: &raw const DIRECT3D_SWAPCHAIN9_VTBL.base,
            // Implicit (device-owned) swapchains start at refcount 0 and forward
            // to the device on the 0↔1 boundary (D3D9 implicit-object model);
            // app-owned additional swapchains start at 1 and own their create
            // reference. `!owned_by_device`: implicit → 0, additional → 1.
            refcount: u32::from(!owned_by_device),
            inner,
        }
    }

    /// An app-owned additional swapchain (`CreateAdditionalSwapChain`).
    ///
    /// `present_params` must already be normalised (dimensions resolved,
    /// back-buffer count clamped to >= 1). Freed when the app releases its last
    /// reference.
    pub fn new(device_inner: *mut DeviceInner, present_params: D3DPRESENT_PARAMETERS) -> Self {
        Self::with_owner(device_inner, present_params, false)
    }

    /// The device's implicit swapchain (`GetSwapChain(0)`).
    ///
    /// It is owned by the device: `Release` never frees it, matching the D3D9
    /// contract where the implicit swapchain outlives an app `Release` and is
    /// destroyed with the device. The small shell is leaked at teardown, like
    /// the device wrapper.
    pub fn new_implicit(
        device_inner: *mut DeviceInner,
        present_params: D3DPRESENT_PARAMETERS,
    ) -> Self {
        Self::with_owner(device_inner, present_params, true)
    }

    /// Overwrite the present parameters this swapchain reports.
    ///
    /// The device keeps its cached implicit swapchain in lockstep after Reset
    /// or an automatic resize, so `GetSwapChain(0).GetPresentParameters`
    /// reflects the live geometry rather than the values captured at first
    /// hand-out.
    pub fn set_present_params(&mut self, present_params: D3DPRESENT_PARAMETERS) {
        // SAFETY: `self.inner` is a live `Box::into_raw` (see `inner()`), valid
        // for every live wrapper reference.
        unsafe { (*self.inner).present_params = present_params };
    }

    /// The owning device's `Direct3DDevice9`* wrapper, or null if the device pointer is unset.
    fn device_wrapper(&self) -> *mut c_void {
        let device_inner = self.inner().device_inner;
        if device_inner.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: `device_inner` is the live owning device (see `SwapChainInner`),
        // alive past its child swapchains per D3D9 lifetime rules.
        unsafe { (*device_inner).device_wrapper() }
    }

    /// Whether the owning device is extended; `false` once the device pointer is unset.
    fn device_is_extended(&self) -> bool {
        // SAFETY: `device_inner` is null or the live owning device (see
        // `SwapChainInner`), alive past its child swapchains.
        unsafe { self.inner().device_inner.as_ref() }.is_some_and(DeviceInner::is_extended)
    }

    fn inner(&self) -> &SwapChainInner {
        // SAFETY: `self.inner` was installed by a constructor as a
        // `Box::into_raw` and is dropped only in `swapchain_release` at
        // refcount zero, so it stays live for every live wrapper reference.
        unsafe { &*self.inner }
    }
}

struct SwapChainInner {
    /// The owning device.
    ///
    /// Borrowed (not `AddRef`'d) — D3D9 lifetime rules keep the device alive
    /// past its child swapchains.
    device_inner: *mut DeviceInner,
    present_params: D3DPRESENT_PARAMETERS,
    /// `true` for the implicit swapchain: `Release` never frees it (the device owns it).
    ///
    /// `false` for additional swapchains, freed at refcount zero.
    owned_by_device: bool,
    /// This (app-owned) swapchain's cached backbuffer surface.
    ///
    /// `0` until the first `GetBackBuffer`. Like the device's implicit render
    /// target it is a `Backbuffer`-kind surface (refcount 0, forwards the device
    /// refcount on its 0↔1 edge, never freed by `Release`); the difference is the
    /// swapchain owns it and finalizes it in `finalize_swapchain`, and its
    /// `GetContainer` is this swapchain. Returning one cached object keeps
    /// `GetBackBuffer` identity stable so a `Release`-to-0-then-`AddRef` no
    /// longer reuses a freed wrapper. Unused for the implicit swapchain (its
    /// backbuffer is the device's implicit RT).
    backbuffer_surface: u64,
    /// Whether the back buffer is referenced, publicly or by a device binding: 1 or 0.
    ///
    /// The back buffer pins its swap chain while anything references it, the
    /// way D3D9 keeps a swap chain alive for as long as its back buffer is, so
    /// the swap chain is finalized only once both this and its public count
    /// are zero. Whichever of the two reaches zero last finalizes it, and the
    /// back buffer with it.
    backbuffer_pins: u32,
}

extern "system" fn swapchain_query_interface(
    this: *mut c_void,
    riid: *const Guid,
    ppv: *mut *mut c_void,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // A swap chain answers the extended IID when its device is extended,
    // whichever create made the device.
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let extended = unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }
        .is_some_and(|obj| obj.device_is_extended());
    let accepted: &[Guid] = if extended {
        &[
            mtld3d_types::IID_IUNKNOWN,
            mtld3d_types::IID_IDIRECT3DSWAPCHAIN9,
            mtld3d_types::IID_IDIRECT3DSWAPCHAIN9EX,
        ]
    } else {
        &[
            mtld3d_types::IID_IUNKNOWN,
            mtld3d_types::IID_IDIRECT3DSWAPCHAIN9,
        ]
    };
    // SAFETY: vtable thunk; `this`, `riid` and `ppv` are the caller's per the
    // IUnknown::QueryInterface ABI.
    unsafe {
        crate::com_ref::com_query_interface(
            this,
            riid,
            ppv,
            accepted,
            swapchain_add_ref,
            "IDirect3DSwapChain9",
        )
    }
}

extern "system" fn swapchain_add_ref(this: *mut c_void) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: IDirect3DSwapChain9 IUnknown AddRef thunk; the D3D9 ABI guarantees
    // `this` is the live wrapper for the call. The engine forwards the device
    // reference for the device-owned implicit swapchain on its 0→1 transition.
    unsafe { crate::com_ref::com_add_ref::<Direct3DSwapChain9>(this) }
}

extern "system" fn swapchain_release(this: *mut c_void) -> u32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: IDirect3DSwapChain9 IUnknown Release thunk; the D3D9 ABI guarantees
    // `this` is the live wrapper for the call. The engine frees an app-owned
    // additional swapchain on its 1→0 transition and forwards the device release
    // for the device-owned implicit swapchain (which is never freed here).
    unsafe { crate::com_ref::com_release::<Direct3DSwapChain9>(this) }
}

/// Pin an additional swap chain while its back buffer is referenced.
///
/// Called by the back buffer when its first reference of either kind, public
/// or a device binding, is taken.
///
/// # Safety
/// `swap_chain` is the live app-owned `Direct3DSwapChain9` the back buffer
/// names as its container.
pub unsafe fn pin_for_backbuffer(swap_chain: u64) {
    // SAFETY: the caller passes the live swap chain wrapper.
    let inner_ptr = unsafe { (*(swap_chain as *mut Direct3DSwapChain9)).inner };
    // SAFETY: its `inner` is the live `Box::into_raw(SwapChainInner)` for as
    // long as the wrapper is.
    let inner = unsafe { &mut *inner_ptr };
    inner.backbuffer_pins += 1;
}

/// Drop the pin the back buffer holds, finalizing the swap chain when nothing else holds it.
///
/// Called by the back buffer when its last reference of either kind goes.
/// When the application has already released the swap chain, this is the
/// release that frees it, and the back buffer with it, so the caller must
/// not touch the back buffer afterwards.
///
/// # Safety
/// `swap_chain` is the live app-owned `Direct3DSwapChain9` the back buffer
/// names as its container, pinned by a matching [`pin_for_backbuffer`].
pub unsafe fn unpin_for_backbuffer(swap_chain: u64) {
    let this = swap_chain as *mut Direct3DSwapChain9;
    let finalize_now = {
        // SAFETY: the caller passes the live, pinned swap chain wrapper.
        let obj = unsafe { &mut *this };
        // SAFETY: its `inner` is live for as long as the wrapper is.
        let inner = unsafe { &mut *obj.inner };
        debug_assert!(
            inner.backbuffer_pins > 0,
            "unpinned a swap chain that was not pinned"
        );
        if inner.backbuffer_pins == 0 {
            // A back buffer's references and its pin went out of step. The
            // swap chain is left alone: whatever freed the pin's owner may
            // already be finalizing it.
            mtld3d_shared::log_once_warn!(target: LOG_TARGET,
                "swap chain {swap_chain:#x}: a back buffer released a pin it did not hold; the swap chain is not finalized here");
            return;
        }
        inner.backbuffer_pins -= 1;
        inner.backbuffer_pins == 0 && obj.refcount == 0 && !inner.owned_by_device
    };
    if finalize_now {
        // SAFETY: the public count and the back buffer's pin are both zero, so
        // no reference to the swap chain survives.
        unsafe { finalize_swapchain(this) };
    }
}

/// Destroy an app-owned `Direct3DSwapChain9` wrapper once its refcount has reached zero.
///
/// Its back buffer goes with it, so this runs only once the back buffer is
/// referenced by nothing either: the last public `Release` of an unpinned
/// swap chain, or [`unpin_for_backbuffer`] after it. The device-owned
/// implicit swapchain is never finalized (its shell is leaked at device
/// teardown).
///
/// # Safety
/// `this` must point to a live, app-owned `Direct3DSwapChain9` wrapper at
/// refcount zero; caller must not access the wrapper afterwards.
unsafe fn finalize_swapchain(this: *mut Direct3DSwapChain9) {
    // SAFETY: refcount reached zero on an app-owned swapchain; `(*this).inner`
    // is the original `Box::into_raw(SwapChainInner)` and no other reference can
    // survive a zero refcount.
    let inner = unsafe { (*this).inner };
    // Finalize the swapchain-owned cached backbuffer surface (a `Backbuffer`-kind
    // surface never freed by its own `Release` — destroyed with its owner here,
    // mirroring how `device_release` finalizes the implicit RT/DS surfaces). It
    // pins this swap chain while referenced, so nothing references it now.
    // SAFETY: `inner` is live (sole owner); read the cached pointer before the
    // box is freed.
    let backbuffer_surface = unsafe { (*inner).backbuffer_surface };
    if backbuffer_surface != 0 {
        // SAFETY: a non-zero `backbuffer_surface` is a live `Backbuffer`-kind
        // surface created by this swapchain; finalized exactly once here.
        unsafe { crate::surface::finalize_implicit_surface(backbuffer_surface) };
    }
    // SAFETY: as above — sole owner of the inner allocation.
    drop(unsafe { Box::from_raw(inner) });
    // SAFETY: refcount reached zero; `this` is the original
    // `Box::into_raw(Direct3DSwapChain9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

// SAFETY: `refcount_mut` exposes this wrapper's own counter and
// `private_refcount` the back buffer's pin; `finalize` frees an app-owned
// swapchain exactly once when both are zero. The device-owned implicit
// swapchain forwards its refcount to the device and is never finalized here.
unsafe impl crate::com_ref::ComChild for Direct3DSwapChain9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn private_refcount(&self) -> u32 {
        // The back buffer's pin keeps a swap chain whose public count reached
        // zero alive; `unpin_for_backbuffer` finalizes it when the pin goes.
        self.inner().backbuffer_pins
    }
    fn owning_device(&self) -> *mut c_void {
        // Both the device-owned implicit swapchain (forwards on its 0→1 edge)
        // and an app-owned additional swapchain (registered at creation, forwards
        // its release at teardown) hold one reference on the device.
        self.device_wrapper()
    }
    fn finalizes_on_zero(&self) -> bool {
        !self.inner().owned_by_device
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — refcount is zero and the swapchain
        // is app-owned (`finalizes_on_zero()` true).
        unsafe { finalize_swapchain(this) };
    }
}

extern "system" fn swapchain_present(
    this: *mut c_void,
    source_rect: *const c_void,
    dest_rect: *const c_void,
    dest_window_override: usize,
    dirty_region: *const c_void,
    flags: u32,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    crate::device::warn_ignored_present_arguments(
        source_rect,
        dest_rect,
        dest_window_override != 0,
        dirty_region,
    );
    crate::device::warn_ignored_present_flags(flags);
    if !obj.inner().owned_by_device {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "IDirect3DSwapChain9::Present on an additional swap chain presents the device's \
             back buffer into the device window; additional swap chains have no back buffer or \
             window of their own");
    }
    let device_inner = obj.inner().device_inner;
    // SAFETY: `device_inner` was stamped from a live `DeviceInner` that
    // outlives its swapchains per D3D9 lifetime rules. There is one drawable,
    // so presenting any swapchain presents the device frame.
    let dev = unsafe { &mut *device_inner };
    if dev.needs_reset() {
        return D3DERR_DEVICENOTRESET;
    }
    dev.present()
}

extern "system" fn swapchain_get_front_buffer_data(this: *mut c_void, surface: *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: vtable thunk; `this` is a live swapchain pointer or null.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject swapchain GetFrontBufferData: null swapchain → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    if !obj.inner().owned_by_device {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject swapchain GetFrontBufferData: additional swapchain has no front image → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: the destination is a caller-owned IDirect3DSurface9 or null.
    let Some(dst) = (unsafe { InPtr::<Direct3DSurface9>::opt(surface) }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject swapchain GetFrontBufferData: null destination → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: a referenced implicit swapchain pins its owning device. The API
    // lock above serializes access; an unset owner is rejected below.
    let Some(dev) = (unsafe { obj.inner().device_inner.as_mut() }) else {
        mtld3d_shared::log_once_warn!(target: LOG_TARGET,
            "reject swapchain GetFrontBufferData: no owning device → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    };
    dev.read_front_buffer(&dst)
}

extern "system" fn swapchain_get_back_buffer(
    this: *mut c_void,
    i_back_buffer: u32,
    _type: u32,
    back_buffer: *mut *mut c_void,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    if back_buffer.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // Out-of-range index fails and leaves the caller's out-param untouched, per
    // the D3D9 contract for a failed interface-returning call.
    if i_back_buffer >= obj.inner().present_params.back_buffer_count {
        return D3DERR_INVALIDCALL;
    }
    // The implicit swapchain's backbuffer IS the device's implicit render target:
    // return the same device-owned cached surface so `GetSwapChain(0)
    // .GetBackBuffer(0)`, `GetRenderTarget(0)` and `GetBackBuffer(0)` share one
    // identity (the `pRenderTarget == pBackBuffer` invariant the suite checks).
    if obj.inner().owned_by_device {
        // SAFETY: `device_inner` is the live owning device (see `SwapChainInner`).
        let dev = unsafe { &mut *obj.inner().device_inner };
        let surf = dev.get_or_create_implicit_render_target();
        // SAFETY: `surf` is the live cached implicit RT surface.
        let add_ref = unsafe { (*surf).vtbl().add_ref };
        // SAFETY: calling the surface AddRef thunk; D3D9 mandates AddRef on return.
        unsafe { add_ref(surf.cast::<c_void>()) };
        // SAFETY: vtable out-param; `back_buffer` is *mut *mut c_void per the ABI.
        unsafe { *back_buffer = surf.cast::<c_void>() };
        return D3D_OK;
    }
    // App-owned additional swapchain: return its cached, swapchain-owned
    // backbuffer surface (a `Backbuffer`-kind surface — refcount 0, forwards the
    // device refcount on its 0↔1 edge and pins this swapchain while referenced,
    // never freed by `Release`; finalized in `finalize_swapchain`). One cached object keeps `GetBackBuffer` identity
    // stable, so a `Release`-to-0-then-`AddRef` no longer reuses a freed wrapper.
    // There is one Metal drawable, so it aliases the device backbuffer (resolved
    // live, like the device's implicit RT) and `this` is its `GetContainer`.
    // SAFETY: `obj.inner` is the live `SwapChainInner`; access is exclusive
    // (D3D9 objects are single-threaded, or serialised by the device `ApiLock`
    // under `D3DCREATE_MULTITHREADED`), so the transient exclusive borrow to
    // lazily cache the backbuffer is sound.
    let inner_mut = unsafe { &mut *obj.inner };
    if inner_mut.backbuffer_surface == 0 {
        let surf = Direct3DSurface9::new_swap_chain_backbuffer(inner_mut.device_inner, this as u64);
        inner_mut.backbuffer_surface = Box::into_raw(Box::new(surf)) as u64;
    }
    let surf = inner_mut.backbuffer_surface as *mut Direct3DSurface9;
    // SAFETY: `surf` is the live cached backbuffer surface.
    let add_ref = unsafe { (*surf).vtbl().add_ref };
    // SAFETY: calling the surface AddRef thunk; D3D9 mandates AddRef on return —
    // the engine forwards the device refcount on the backbuffer's 0→1 edge.
    unsafe { add_ref(surf.cast::<c_void>()) };
    // SAFETY: vtable out-param; `back_buffer` is *mut *mut c_void per the ABI.
    unsafe { OutPtr::write_opt(back_buffer, surf.cast::<c_void>()) };
    D3D_OK
}

extern "system" fn swapchain_get_raster_status(this: *mut c_void, _status: *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    mtld3d_shared::log_once_warn!(target: LOG_TARGET, "stub IDirect3DSwapChain9::GetRasterStatus → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn swapchain_get_display_mode(this: *mut c_void, mode: *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    if mode.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // Mirror `device_get_display_mode`: fullscreen reports the honored mode,
    // windowed reports the desktop's. `present_params` is normalised at
    // creation and refreshed on Reset (`DeviceInner::set_present_params`), so the
    // implicit swapchain stays in sync with the device.
    let pp = obj.inner().present_params;
    // SAFETY: `mode` is non-null (checked) and per the D3D9 ABI points to a
    // writable `D3DDISPLAYMODE` slot owned by the caller.
    unsafe {
        *mode.cast::<D3DDISPLAYMODE>() = crate::direct3d9::reported_display_mode(&pp);
    }
    D3D_OK
}

extern "system" fn swapchain_get_device(this: *mut c_void, device: *mut *mut c_void) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    unsafe { crate::com_ref::com_get_device::<Direct3DSwapChain9>(this, device) }
}

extern "system" fn swapchain_get_present_parameters(
    this: *mut c_void,
    parameters: *mut c_void,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    if parameters.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // SAFETY: `parameters` is non-null (checked) and points to a writable
    // `D3DPRESENT_PARAMETERS` per the D3D9 ABI.
    unsafe { *parameters.cast::<D3DPRESENT_PARAMETERS>() = obj.inner().present_params };
    D3D_OK
}

extern "system" fn swapchain_get_last_present_count(this: *mut c_void, count: *mut u32) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    mtld3d_shared::log_once_warn!(target: LOG_TARGET,
        "stub IDirect3DSwapChain9Ex::GetLastPresentCount → 0 (presents are not counted)");
    if count.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable out-param; `count` is non-null (checked) and points to a
    // writable UINT per the IDirect3DSwapChain9Ex ABI.
    unsafe { OutPtr::write_opt(count, 0) };
    D3D_OK
}

extern "system" fn swapchain_get_present_stats(
    this: *mut c_void,
    stats: *mut D3DPRESENTSTATS,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    mtld3d_shared::log_once_warn!(target: LOG_TARGET,
        "stub IDirect3DSwapChain9Ex::GetPresentStats → zeroed statistics");
    if stats.is_null() {
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable out-param; `stats` is non-null (checked) and points to a
    // writable `D3DPRESENTSTATS` per the IDirect3DSwapChain9Ex ABI.
    unsafe {
        OutPtr::write_opt(
            stats,
            D3DPRESENTSTATS {
                present_count: 0,
                present_refresh_count: 0,
                sync_refresh_count: 0,
                pad0: 0,
                sync_qpc_time: [0; 2],
                sync_gpu_time: [0; 2],
            },
        );
    };
    D3D_OK
}

extern "system" fn swapchain_get_display_mode_ex(
    this: *mut c_void,
    mode: *mut c_void,
    rotation: *mut u32,
) -> i32 {
    let _api = crate::com_ref::com_api_lock::<Direct3DSwapChain9>(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DSwapChain9 per the ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DSwapChain9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    // The mode `GetDisplayMode` reports, from the same present parameters.
    let current = crate::direct3d9::reported_display_mode(&obj.inner().present_params);
    // SAFETY: vtable out-params; `mode` and `rotation` are null or writable per
    // the IDirect3DSwapChain9Ex ABI.
    unsafe { crate::direct3d9::write_display_mode_ex(mode, rotation, &current) }
}
