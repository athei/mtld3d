//! The `IDirect3DDevice9Ex` entry points past the base table.
//!
//! An extended device is the base device with `DeviceFlags::EXTENDED` set;
//! these fifteen methods are the part of its contract the base table has no
//! slot for. Each one either shares a base implementation (`PresentEx`,
//! `ResetEx`, the three surface creates, `GetDisplayModeEx`), stores and
//! reports a value (the frame latency), or is a stub that answers what a
//! device with nothing to do there answers, logged once.

use core::ffi::c_void;

use mtld3d_core::{
    extended::{ex_create_usage_valid, frame_latency},
    perf::DeviceSubCategory,
};
use mtld3d_shared::{InPtr, InPtrMut, OutPtr};
use mtld3d_types::{
    D3DDISPLAYMODEEX, D3DUSAGE_RESTRICT_SHARED_RESOURCE, D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER,
    D3DUSAGE_RESTRICTED_CONTENT,
};

use super::{
    D3D_OK, D3DERR_INVALIDCALL, Direct3DDevice9, LOG_TARGET, ResetCall, device_api_lock,
    device_create_depth_stencil_surface, device_create_offscreen_plain_surface,
    device_create_render_target, device_timer, present_impl, reset_impl,
};

/// `SetConvolutionMonoKernel`: the device offers no convolution filter, so the call is invalid.
pub extern "system" fn set_convolution_mono_kernel(
    this: *mut c_void,
    _width: u32,
    _height: u32,
    _rows: *mut f32,
    _columns: *mut f32,
) -> i32 {
    let _api = device_api_lock(this);
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "IDirect3DDevice9Ex::SetConvolutionMonoKernel: no convolution filter → INVALIDCALL"
    );
    D3DERR_INVALIDCALL
}

/// `ComposeRects`: a stub that succeeds and composes nothing.
pub extern "system" fn compose_rects(
    this: *mut c_void,
    _src: *mut c_void,
    _dst: *mut c_void,
    _src_rect_descs: *mut c_void,
    _rect_count: u32,
    _dst_rect_descs: *mut c_void,
    _operation: u32,
    _offset_x: i32,
    _offset_y: i32,
) -> i32 {
    let _api = device_api_lock(this);
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub IDirect3DDevice9Ex::ComposeRects → OK (nothing is composed)"
    );
    D3D_OK
}

/// `PresentEx`: `Present` with flags, which are logged and not honoured.
pub extern "system" fn present_ex(
    this: *mut c_void,
    src_rect: *const c_void,
    dst_rect: *const c_void,
    dst_window_override: *mut c_void,
    dirty_region: *const c_void,
    flags: u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Frame);
    present_impl(
        this,
        src_rect,
        dst_rect,
        dst_window_override,
        dirty_region,
        flags,
    )
}

/// `GetGPUThreadPriority`: a stub that reports the normal priority, 0.
pub extern "system" fn get_gpu_thread_priority(this: *mut c_void, priority: *mut i32) -> i32 {
    let _api = device_api_lock(this);
    if priority.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject IDirect3DDevice9Ex::GetGPUThreadPriority: null out slot → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub IDirect3DDevice9Ex::GetGPUThreadPriority → 0"
    );
    // SAFETY: vtable out-param; `priority` is non-null (checked) and points to
    // a writable INT per the IDirect3DDevice9Ex ABI.
    unsafe { OutPtr::write_opt(priority, 0) };
    D3D_OK
}

/// `SetGPUThreadPriority`: a stub that succeeds and stores nothing.
pub extern "system" fn set_gpu_thread_priority(this: *mut c_void, priority: i32) -> i32 {
    let _api = device_api_lock(this);
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub IDirect3DDevice9Ex::SetGPUThreadPriority({priority}) → OK (not applied)"
    );
    D3D_OK
}

/// `WaitForVBlank`: a stub that returns at once for the implicit swap chain.
pub extern "system" fn wait_for_vblank(this: *mut c_void, swap_chain: u32) -> i32 {
    let _api = device_api_lock(this);
    if swap_chain != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject IDirect3DDevice9Ex::WaitForVBlank(swap_chain={swap_chain}) → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    mtld3d_shared::log_once_warn!(
        target: LOG_TARGET,
        "stub IDirect3DDevice9Ex::WaitForVBlank → OK without waiting"
    );
    D3D_OK
}

/// `CheckResourceResidency`: every resource is resident in unified memory.
pub extern "system" fn check_resource_residency(
    this: *mut c_void,
    _resources: *mut *mut c_void,
    _count: u32,
) -> i32 {
    let _api = device_api_lock(this);
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "IDirect3DDevice9Ex::CheckResourceResidency → OK (no residency to report)"
    );
    D3D_OK
}

/// `SetMaximumFrameLatency`: stored for `GetMaximumFrameLatency`, not enforced.
///
/// Zero restores the default and a value past the limit is invalid. A value
/// below the default is logged once: the encoder's own queue depth bounds
/// how far the application runs ahead, so a lower latency is not reached.
pub extern "system" fn set_maximum_frame_latency(this: *mut c_void, latency: u32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per the IDirect3DDevice9Ex ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let Some(stored) = frame_latency(latency) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject IDirect3DDevice9Ex::SetMaximumFrameLatency({latency}) → INVALIDCALL (past the \
             limit)"
        );
        return D3DERR_INVALIDCALL;
    };
    if stored < mtld3d_core::extended::DEFAULT_FRAME_LATENCY {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "IDirect3DDevice9Ex::SetMaximumFrameLatency({stored}): stored and reported, not \
             enforced; the encoder's queue depth bounds the frames in flight"
        );
    }
    obj.inner().set_max_frame_latency(stored);
    D3D_OK
}

/// `GetMaximumFrameLatency`: the value the last `SetMaximumFrameLatency` stored.
pub extern "system" fn get_maximum_frame_latency(this: *mut c_void, latency: *mut u32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per the IDirect3DDevice9Ex ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    if latency.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject IDirect3DDevice9Ex::GetMaximumFrameLatency: null out slot → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable out-param; `latency` is non-null (checked) and points to a
    // writable UINT per the IDirect3DDevice9Ex ABI.
    unsafe { OutPtr::write_opt(latency, obj.inner().max_frame_latency()) };
    D3D_OK
}

/// `CheckDeviceState`: the device's failure latch, `D3D_OK` without one.
///
/// No exclusive mode is taken and no window can occlude a present, so the
/// occlusion and mode-change codes never arise.
pub extern "system" fn check_device_state(this: *mut c_void, _window: *mut c_void) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per the IDirect3DDevice9Ex ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    mtld3d_shared::log_once_info!(
        target: LOG_TARGET,
        "IDirect3DDevice9Ex::CheckDeviceState: reports the failure latch alone; no occlusion or \
         mode-change status"
    );
    match obj.inner().encoder_status() {
        Ok(()) => D3D_OK,
        Err(hr) => hr,
    }
}

/// `CreateRenderTargetEx`: `CreateRenderTarget` with an extended usage.
///
/// A usage it refuses leaves the out slot as the caller passed it.
pub extern "system" fn create_render_target_ex(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    multi_sample: u32,
    multi_sample_quality: u32,
    lockable: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
    usage: u32,
) -> i32 {
    let _api = device_api_lock(this);
    if !usage_accepted(usage, shared_handle, "CreateRenderTargetEx") {
        return D3DERR_INVALIDCALL;
    }
    let hr = device_create_render_target(
        this,
        width,
        height,
        format,
        multi_sample,
        multi_sample_quality,
        lockable,
        surface,
        shared_handle,
    );
    report_usage(hr, surface, usage)
}

/// `CreateOffscreenPlainSurfaceEx`: `CreateOffscreenPlainSurface` with an extended usage.
pub extern "system" fn create_offscreen_plain_surface_ex(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    pool: u32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
    usage: u32,
) -> i32 {
    let _api = device_api_lock(this);
    if !usage_accepted(usage, shared_handle, "CreateOffscreenPlainSurfaceEx") {
        return D3DERR_INVALIDCALL;
    }
    let hr = device_create_offscreen_plain_surface(
        this,
        width,
        height,
        format,
        pool,
        surface,
        shared_handle,
    );
    report_usage(hr, surface, usage)
}

/// `CreateDepthStencilSurfaceEx`: `CreateDepthStencilSurface` with an extended usage.
pub extern "system" fn create_depth_stencil_surface_ex(
    this: *mut c_void,
    width: u32,
    height: u32,
    format: u32,
    multi_sample: u32,
    multi_sample_quality: u32,
    discard: i32,
    surface: *mut *mut c_void,
    shared_handle: *mut c_void,
    usage: u32,
) -> i32 {
    let _api = device_api_lock(this);
    if !usage_accepted(usage, shared_handle, "CreateDepthStencilSurfaceEx") {
        return D3DERR_INVALIDCALL;
    }
    let hr = device_create_depth_stencil_surface(
        this,
        width,
        height,
        format,
        multi_sample,
        multi_sample_quality,
        discard,
        surface,
        shared_handle,
    );
    report_usage(hr, surface, usage)
}

/// `ResetEx`: `Reset` with a display mode, which must agree with the present parameters.
///
/// A fullscreen request names a mode of the back buffer's size and a
/// windowed one names none; a disagreement is an invalid call that leaves
/// the device as it was. The mode's refresh rate and format are not used: a
/// fullscreen device sets the mode its back buffer names.
pub extern "system" fn reset_ex(
    this: *mut c_void,
    present_params: *mut c_void,
    mode: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if present_params.is_null() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject ResetEx: null present parameters → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable in-param; `mode` is null or a readable `D3DDISPLAYMODEEX`
    // per the IDirect3DDevice9Ex ABI.
    let mode = unsafe { InPtr::<D3DDISPLAYMODEEX>::opt(mode) }.map(|m| (m.width, m.height));
    reset_impl(this, present_params, ResetCall::ResetEx { mode })
}

/// `GetDisplayModeEx`: the mode `GetDisplayMode` reports, progressive, with the identity rotation.
pub extern "system" fn get_display_mode_ex(
    this: *mut c_void,
    swap_chain: u32,
    mode: *mut c_void,
    rotation: *mut u32,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if swap_chain != 0 {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "reject IDirect3DDevice9Ex::GetDisplayModeEx(swap_chain={swap_chain}) → INVALIDCALL"
        );
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per the IDirect3DDevice9Ex ABI.
    let Some(obj) = (unsafe { InPtrMut::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let current = crate::direct3d9::reported_display_mode(obj.inner().present_params());
    // SAFETY: vtable out-params; `mode` and `rotation` are null or writable per
    // the IDirect3DDevice9Ex ABI.
    unsafe { crate::direct3d9::write_display_mode_ex(mode, rotation, &current) }
}

/// Add an extended create's usage to the surface it made, passing its `hr` through.
///
/// `GetDesc` reports the usage the extended create was given, on top of the
/// one the base create implies.
fn report_usage(hr: i32, surface: *mut *mut c_void, usage: u32) -> i32 {
    if hr != D3D_OK || usage == 0 {
        return hr;
    }
    // SAFETY: the create succeeded, so `surface` is the caller's writable out
    // slot and holds the surface it handed out.
    let Some(created) =
        (unsafe { mtld3d_shared::ValueIn::<*mut c_void>::read_opt(surface.cast()) })
    else {
        return hr;
    };
    if !created.is_null() {
        // SAFETY: `created` is the live surface the create just handed out.
        unsafe { crate::surface::add_reported_usage(created, usage) };
    }
    hr
}

/// Whether an extended surface create's `usage` is one it accepts, warned once otherwise.
fn usage_accepted(usage: u32, shared_handle: *mut c_void, entry_point: &str) -> bool {
    if !ex_create_usage_valid(usage, !shared_handle.is_null()) {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: u64::from(usage),
            "reject {entry_point}(usage={usage:#x}) → INVALIDCALL (only the restricted-content \
             and shared-resource restrictions are extended usages, the latter with a \
             pSharedHandle)"
        );
        return false;
    }
    if usage
        & (D3DUSAGE_RESTRICTED_CONTENT
            | D3DUSAGE_RESTRICT_SHARED_RESOURCE
            | D3DUSAGE_RESTRICT_SHARED_RESOURCE_DRIVER)
        != 0
    {
        mtld3d_shared::log_once_info!(
            target: LOG_TARGET,
            "{entry_point}: content and sharing restrictions (usage {usage:#x}) have nothing to \
             restrict and are not applied"
        );
    }
    true
}
