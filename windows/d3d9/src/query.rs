//! `IDirect3DQuery9` implementation.
//!
//! - `EVENT`: synchronous-at-the-API-boundary ("completed immediately").
//! - `TIMESTAMP`: stub — returns 0 ticks. Not implemented, logs once.
//! - `OCCLUSION`: real Metal visibility-result query. `Issue(BEGIN/END)`
//!   pushes closures onto the current frame that bump the encoder's
//!   visibility offset allocator and emit
//!   `setVisibilityResultMode:offset:` commands; `GetData` polls the
//!   shared `VisibilityQueryCore` for the finalized pixel count.
//!   See `mtld3d_core::visibility` for the sum + pool machinery.

use core::ffi::c_void;
use std::sync::Arc;

use mtld3d_core::visibility::{QueryStatus, VisibilityQueryCore};
use mtld3d_shared::{InPtr, OutPtr};
use mtld3d_types::{
    D3DGETDATA_FLUSH, D3DISSUE_BEGIN, D3DISSUE_END, D3DQUERYTYPE_EVENT, D3DQUERYTYPE_OCCLUSION,
    D3DQUERYTYPE_TIMESTAMP, Guid, IDirect3DQuery9Vtbl,
};

use super::{D3D_OK, D3DERR_INVALIDCALL, E_NOINTERFACE, LOG_TARGET, device::DeviceInner};

pub static DIRECT3D_QUERY9_VTBL: IDirect3DQuery9Vtbl = IDirect3DQuery9Vtbl {
    query_interface: query_query_interface,
    add_ref: query_add_ref,
    release: query_release,
    get_device: query_get_device,
    get_type: query_get_type,
    get_data_size: query_get_data_size,
    issue: query_issue,
    get_data: query_get_data,
};

#[repr(C)]
pub struct Direct3DQuery9 {
    vtbl: *const IDirect3DQuery9Vtbl,
    refcount: u32,
    inner: *mut QueryInner,
}

impl Direct3DQuery9 {
    pub fn new(device_inner: *mut DeviceInner, query_type: u32, data_size: u32) -> Self {
        let core = if query_type == D3DQUERYTYPE_OCCLUSION {
            Some(VisibilityQueryCore::new())
        } else {
            None
        };
        let inner = Box::into_raw(Box::new(QueryInner {
            device_inner,
            query_type,
            data_size,
            core,
        }));
        Self {
            vtbl: &raw const DIRECT3D_QUERY9_VTBL,
            refcount: 1,
            inner,
        }
    }

    fn inner(&self) -> &QueryInner {
        // SAFETY: `self.inner` was installed by `Self::new` as a
        // `Box::into_raw` and is dropped only in `query_release` at
        // refcount zero, so it stays live for every live wrapper
        // reference.
        unsafe { &*self.inner }
    }
}

/// Returns the byte count `GetDataSize` reports.
///
/// `None` doubles as the flag telling the caller whether the query type is
/// supported.
pub const fn data_size_for(query_type: u32) -> Option<u32> {
    match query_type {
        // BOOL (EVENT) / DWORD pixel count (OCCLUSION) — both u32-sized.
        D3DQUERYTYPE_EVENT | D3DQUERYTYPE_OCCLUSION => Some(4),
        D3DQUERYTYPE_TIMESTAMP => Some(8), // UINT64
        _ => None,
    }
}

struct QueryInner {
    device_inner: *mut DeviceInner,
    query_type: u32,
    /// Number of bytes `GetData` should write when `data != null`.
    data_size: u32,
    /// For OCCLUSION queries: the shared counter behind the COM wrapper.
    ///
    /// BEGIN/END closures on the encoder thread mutate this via atomics;
    /// `intake_visibility` finalizes it post-GPU. `None` for
    /// EVENT / TIMESTAMP.
    core: Option<Arc<VisibilityQueryCore>>,
}

#[inline]
fn query_timer(this: *mut c_void) -> mtld3d_core::perf::ApiTimer {
    use mtld3d_core::perf::{ApiCategory, ApiTimer};
    // SAFETY: vtable thunk; `this` is *mut Direct3DQuery9 per IDirect3DQuery9 ABI.
    let perf_ptr = (unsafe { InPtr::<Direct3DQuery9>::opt(this) })
        .map_or(core::ptr::null_mut(), |obj| {
            crate::device::DeviceInner::perf_ptr_of(obj.inner().device_inner)
        });
    ApiTimer::start(perf_ptr, ApiCategory::Query)
}

extern "system" fn query_query_interface(
    this: *mut c_void,
    _riid: *const Guid,
    _ppv: *mut *mut c_void,
) -> i32 {
    let _timer = query_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DQuery9::QueryInterface → E_NOINTERFACE");
    E_NOINTERFACE
}

extern "system" fn query_add_ref(this: *mut c_void) -> u32 {
    let _timer = query_timer(this);
    // SAFETY: IDirect3DQuery9 IUnknown AddRef thunk; the D3D9 ABI guarantees
    // `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_add_ref::<Direct3DQuery9>(this) }
}

extern "system" fn query_release(this: *mut c_void) -> u32 {
    let _timer = query_timer(this);
    // SAFETY: IDirect3DQuery9 IUnknown Release thunk; the D3D9 ABI guarantees
    // `this` is the live wrapper for the call.
    unsafe { crate::com_ref::com_release::<Direct3DQuery9>(this) }
}

/// Destroy a `Direct3DQuery9` wrapper once its refcount has reached zero.
///
/// # Safety
/// `this` must point to a live `Direct3DQuery9` wrapper at refcount zero;
/// caller must not access the wrapper afterwards.
unsafe fn finalize_query(this: *mut Direct3DQuery9) {
    // SAFETY: refcount reached zero; `(*this).inner` is the original
    // `Box::into_raw(QueryInner)` from `Self::new` and no other reference
    // can survive a zero refcount.
    let inner = unsafe { (*this).inner };
    // SAFETY: as above — sole owner of the inner allocation.
    drop(unsafe { Box::from_raw(inner) });
    // SAFETY: refcount reached zero; `this` is the original
    // `Box::into_raw(Direct3DQuery9)` allocation.
    drop(unsafe { Box::from_raw(this) });
}

// SAFETY: `refcount_mut` exposes this wrapper's own counter; `finalize` frees
// it exactly once at refcount zero. Queries have no bound-slot (private)
// refcount; they forward one device reference for their public lifetime.
unsafe impl crate::com_ref::ComChild for Direct3DQuery9 {
    fn refcount_mut(&mut self) -> &mut u32 {
        &mut self.refcount
    }
    fn device_forward_target(&self) -> *mut c_void {
        crate::device::device_wrapper_from(self.inner().device_inner)
    }
    unsafe fn finalize(this: *mut Self) {
        // SAFETY: forwarded from the engine — refcount is zero.
        unsafe { finalize_query(this) };
    }
}

extern "system" fn query_get_device(this: *mut c_void, _device: *mut *mut c_void) -> i32 {
    let _timer = query_timer(this);
    mtld3d_shared::log_once_warn!(target: crate::LOG_TARGET, "stub IDirect3DQuery9::GetDevice → INVALIDCALL");
    D3DERR_INVALIDCALL
}

extern "system" fn query_get_type(this: *mut c_void) -> u32 {
    let _timer = query_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DQuery9 per IDirect3DQuery9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DQuery9>::opt(this) }) else {
        return 0;
    };
    obj.inner().query_type
}

extern "system" fn query_get_data_size(this: *mut c_void) -> u32 {
    let _timer = query_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DQuery9 per IDirect3DQuery9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DQuery9>::opt(this) }) else {
        return 0;
    };
    obj.inner().data_size
}

extern "system" fn query_issue(this: *mut c_void, flags: u32) -> i32 {
    let _timer = query_timer(this);
    let unknown = flags & !(D3DISSUE_BEGIN | D3DISSUE_END);
    if unknown != 0 {
        mtld3d_shared::log_once_warn_by!(
            target: LOG_TARGET,
            key: u64::from(flags),
            "Query::Issue(flags={flags:#x}) has unknown bits {unknown:#x} — accepting as no-op",
        );
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DQuery9 per IDirect3DQuery9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DQuery9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    if inner.query_type != D3DQUERYTYPE_OCCLUSION {
        return D3D_OK;
    }
    let Some(core) = inner.core.clone() else {
        return D3D_OK;
    };
    let device_inner = inner.device_inner;
    // SAFETY: `inner.device_inner` was stamped at `Self::new` from a live
    // `DeviceInner` and is kept alive by the device — query outlives the
    // app's device only if the app violates D3D9 lifetime rules.
    let dev = unsafe { &mut *device_inner };
    // Clone once for BEGIN (defensive — both bits can be set in one
    // call), move the original into END so we don't waste a refcount
    // bump in the common END-only path.
    if flags & D3DISSUE_BEGIN != 0 {
        // Reflect "query armed" synchronously so a no-Present
        // `GetData(D3DGETDATA_FLUSH)` sees `Pending` (and, under the
        // blocking config, flushes the recording frame to run this
        // closure) rather than hitting the initial `NeverIssued`
        // short-circuit. The closure resets the accumulator + slot as
        // usual when the encoder drains it.
        core.mark_armed();
        let c = core.clone();
        dev.push_op(Box::new(move |enc| enc.begin_visibility_query(&c)));
    }
    if flags & D3DISSUE_END != 0 {
        // Mark "end issued" synchronously so a no-Present `GetData(FLUSH)`
        // knows the span is closed and safe to flush (an *open* query must
        // not be flushed — that splits it across submits and zeroes it).
        core.mark_end_requested();
        dev.push_op(Box::new(move |enc| enc.end_visibility_query(core)));
    }
    D3D_OK
}

/// Write the low `min(size, 8)` bytes of a 64-bit occlusion result.
///
/// D3D9 advertises `GetDataSize == 4` (DWORD) but the runtime backs
/// occlusion with a UINT64 and honors partial/oversized reads, so the
/// width is the caller's `size` capped at 8 — NOT the advertised DWORD.
unsafe fn write_occlusion(data: *mut c_void, size: u32, value: u64) {
    let n = (size as usize).min(8);
    let bytes = value.to_le_bytes();
    // SAFETY: caller guarantees `data` is non-null with >= `size` writable
    // bytes and `size >= 1` (size == 0 returns earlier); `n <= size`.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), n) };
}

extern "system" fn query_get_data(
    this: *mut c_void,
    data: *mut c_void,
    size: u32,
    flags: u32,
) -> i32 {
    let _timer = query_timer(this);
    // SAFETY: vtable thunk; `this` is *mut Direct3DQuery9 per IDirect3DQuery9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DQuery9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let inner = obj.inner();
    if data.is_null() || size == 0 {
        // Caller is polling for "is it done?"; always yes.
        return D3D_OK;
    }
    let wanted = inner.data_size.min(size) as usize;
    match inner.query_type {
        D3DQUERYTYPE_EVENT if wanted >= 4 => {
            // BOOL TRUE = event "completed".
            // SAFETY: vtable out-param; `data` is the typed out-buffer per ABI.
            unsafe { OutPtr::write_opt(data.cast::<u32>(), 1) };
            D3D_OK
        }
        D3DQUERYTYPE_OCCLUSION => {
            // `wanted` (capped at the advertised DWORD) is wrong here: the
            // runtime backs occlusion with a UINT64 and honors partial
            // (`size < 4`) and oversized (`size >= 8`) reads, so the write
            // width is the caller's `size` capped at 8. `size == 0` already
            // returned `D3D_OK` above, so `size >= 1` here.
            let Some(core) = inner.core.as_ref() else {
                // No backing visibility slot (e.g. pool exhaustion). Report the
                // permissive "fully visible" pixel count (`u32::MAX`), matching
                // the FLUSH stub below and the exhaustion path in `visibility`,
                // so a missing query never makes a title cull geometry it would
                // otherwise draw (lens flares, occlusion-gated effects).
                // SAFETY: `data` is non-null with >= `size` writable bytes per
                // the ABI and `size >= 1`; `write_occlusion` writes `min(size, 8)`.
                unsafe { write_occlusion(data, size, u64::from(u32::MAX)) };
                return D3D_OK;
            };
            match core.status() {
                QueryStatus::NeverIssued => {
                    // A query that has never been issued (`Issue(END)` never
                    // called) returns the runtime's uninitialised-result
                    // poison: every byte `0xdd`.
                    // SAFETY: as above — non-null `data`, `size >= 1`.
                    unsafe { write_occlusion(data, size, 0xdddd_dddd_dddd_dddd) };
                    D3D_OK
                }
                QueryStatus::Pending => {
                    if flags & D3DGETDATA_FLUSH != 0 {
                        if crate::config::CONFIG.query_flush_immediate {
                            // D3D9-era games use FLUSH-poll as a
                            // poor-man's GPU fence to compensate for
                            // 2004-era drivers that lacked resource
                            // hazard tracking. Metal tracks hazards
                            // explicitly, so the fence buys nothing
                            // for correctness — it just throttles
                            // our API thread (the bottleneck) while
                            // the GPU (which has headroom) is given
                            // time it doesn't need. Return a
                            // permissive stub per the polarity rule
                            // (`u32::MAX` = "fully visible") so any
                            // unusual reader doesn't cull geometry;
                            // the real count finalizes naturally on
                            // the next `begin_frame` intake if the
                            // game ever reads via `flags = 0`.
                            // SAFETY: as above — non-null `data`, `size >= 1`.
                            unsafe { write_occlusion(data, size, u64::from(u32::MAX)) };
                            return D3D_OK;
                        }
                        // Spec-correct fallback (config off). The
                        // Present-driven encoder may not have run this
                        // query's BEGIN/END closures yet (a D3D9 app can
                        // poll a query with no intervening Present), so
                        // first flush the current recording frame: that
                        // drains the closures (assigning the visibility
                        // slots + `seq_end`) and submits the counting
                        // pass to the GPU. Then block on the GPU retiring
                        // `seq_end` so intake folds the per-fragment
                        // counts in and the status read below sees
                        // `Issued`. Bracket with `CycleAddTimer` so the
                        // kernel sleep shows up as the `Wait for GPU`
                        // sub-row under `Query` in the perf summary.
                        //
                        // Only do this once END has been issued. A query
                        // still open (begun, not ended) has its counting
                        // draws recorded *after* this point; flushing now
                        // would close the recording frame between BEGIN
                        // and the draws, splitting the span across two
                        // submits (the count then reads 0). Reporting
                        // `S_FALSE` for an open query keeps BEGIN + draws
                        // + END in one frame for the flush that the
                        // END-side poll triggers.
                        if core.end_requested() {
                            // SAFETY: `inner.device_inner` was stamped at
                            // `Self::new` from a live `DeviceInner` and is kept
                            // alive by the device for the wrapper's lifetime.
                            let dev = unsafe { &mut *inner.device_inner };
                            {
                                let _wait = mtld3d_core::perf::CycleAddTimer::start(
                                    dev.perf_mut().query_wait_cycles_ptr(),
                                );
                                dev.flush_current_frame_blocking();
                                dev.encoder_intake_visibility_for(core.seq_end_loaded());
                            }
                            if core.status() == QueryStatus::Issued {
                                // SAFETY: as above — non-null `data`, `size >= 1`.
                                unsafe { write_occlusion(data, size, core.get_u64()) };
                                return D3D_OK;
                            }
                        }
                    }
                    // S_FALSE (0x1) — still not ready; caller will
                    // retry.
                    1
                }
                QueryStatus::Issued => {
                    // SAFETY: as above — non-null `data`, `size >= 1`.
                    unsafe { write_occlusion(data, size, core.get_u64()) };
                    D3D_OK
                }
            }
        }
        D3DQUERYTYPE_TIMESTAMP if wanted >= 8 => {
            // SAFETY: vtable out-param; `data` is the typed out-buffer per ABI.
            unsafe { OutPtr::write_opt(data.cast::<u64>(), 0) };
            mtld3d_shared::log_once_warn!(
                target: LOG_TARGET,
                "stub IDirect3DQuery9::GetData for D3DQUERYTYPE_TIMESTAMP → reporting 0 ticks"
            );
            D3D_OK
        }
        other => {
            // SAFETY: `data` is non-null (checked above) and per the D3D9
            // ABI points to a buffer of at least `size` bytes; `wanted =
            // min(data_size, size)` stays within that buffer.
            unsafe { core::ptr::write_bytes(data.cast::<u8>(), 0, wanted) };
            mtld3d_shared::log_once_warn_by!(
                target: LOG_TARGET,
                key: u64::from(other),
                "stub IDirect3DQuery9::GetData for unknown query_type={other} → reporting zeros",
            );
            D3D_OK
        }
    }
}
