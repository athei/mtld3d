//! The cursor methods of `IDirect3DDevice9` and the words the unix side publishes for a device.
//!
//! The entry points unpack the D3D9 arguments and hand the operation to the
//! device's [`CursorState`].
//!
//! [`CursorState`]: crate::platform::cursor::CursorState

use core::{
    ffi::c_void,
    ptr::null_mut,
    sync::atomic::{AtomicU32, Ordering},
};

use log::warn;
use mtld3d_core::{cursor::BitmapLayout, perf::DeviceSubCategory};
use mtld3d_shared::{InPtr, mtl::ClientSurfaceRequest};
use mtld3d_types::{D3DLOCK_READONLY, D3DLOCKED_RECT, D3DSURFACE_DESC, IDirect3DSurface9Vtbl};

use super::{Direct3DDevice9, device_api_lock, device_timer};
use crate::{D3D_OK, D3DERR_INVALIDCALL};

/// Cursor-specific log sub-target.
///
/// Inherits filtering from any broader `mtld3d::d3d9` or `mtld3d` selector by
/// `env_logger`'s `::`-prefix matching, so `RUST_LOG=mtld3d::d3d9=warn` still
/// catches the `warn!`s below. The dedicated target lets us crank trace
/// separately:
///   `RUST_LOG=mtld3d::d3d9::cursor=trace`
pub const LOG_TARGET: &str = "mtld3d::d3d9::cursor";

/// The three words the unix side writes into from its main thread, at a stable address.
///
/// `AttachMetalLayer` hands the unix side the addresses, and the record it
/// keeps for the device's view writes through them outside any thunk: the
/// backing scale at attach and whenever the display-follow reconciliation
/// derives another, the cursor kick when the pointer comes back from another
/// process, the client surface request at attach and when Wine hides the
/// device's client view. The window procedure sets the request too, when the
/// window's client area may have moved. The box is created before the attach
/// and owned by the device's `CursorState`, so the addresses hold for the
/// device's lifetime. Ordering against teardown is what makes that enough:
/// device release issues `DestroyCommandQueue` while the inner box is still
/// alive, that thunk unregisters the record under the unix registry lock,
/// every unix write into a sink runs under the same lock with a liveness
/// check, and the inner box drops only after the thunk has returned. So a
/// write lands before the record is gone or not at all.
pub struct DisplaySinks {
    /// Wine's retina factor for the layer (2 in retina mode, else 1); `0` until published.
    backing_scale: AtomicU32,
    /// Set by the unix side to ask for a cursor re-apply; taken back to zero here.
    ///
    /// A system tool that borrows the pointer (the screenshot crosshair)
    /// leaves its own cursor on screen, and Wine re-applies its cursor only
    /// on a handle change. The cursor module takes the flag at the next
    /// `WM_SETCURSOR` or `ShowCursor(TRUE)` and answers it with the
    /// null-then-set kick.
    cursor_kick: AtomicU32,
    /// The `ClientSurfaceRequest` bits asked since the last `Present` took them.
    ///
    /// Some Wines hide the client view a device presents through whenever
    /// they flush the window surface, and leave its frame where it was when
    /// the window resizes, counting on winemac's `nextDrawable` override to
    /// put both right; the layer presents without it. The attach sets
    /// `ATTACHED` here too, so the device's first submitted frame tells Wine
    /// it has drawn. The next `Present` that submits a frame takes the bits
    /// and sends `PresentClientSurface`, which has Wine show the surface
    /// again, sized to the client area.
    client_surface_request: AtomicU32,
}

impl DisplaySinks {
    pub const fn new() -> Self {
        Self {
            backing_scale: AtomicU32::new(0),
            cursor_kick: AtomicU32::new(0),
            client_surface_request: AtomicU32::new(0),
        }
    }

    /// The address `AttachMetalLayerParams::client_surface_request_ptr` carries.
    pub fn client_surface_request_ptr(&self) -> u64 {
        (&raw const self.client_surface_request) as u64
    }

    /// Ask the next `Present` to have Wine show the client surface again, for `reason`.
    pub fn request_client_surface(&self, reason: ClientSurfaceRequest) {
        self.client_surface_request
            .fetch_or(reason.bits(), Ordering::Release);
    }

    /// Take the client surface request, empty when nothing asked since the last take.
    ///
    /// One relaxed load on a frame nothing asked on, which is every frame but
    /// the first after an attach and the ones after a resize or a hidden
    /// client view; the swap that takes the bits pairs its `Acquire` with the
    /// writers' `Release`.
    pub fn take_client_surface_request(&self) -> ClientSurfaceRequest {
        if self.client_surface_request.load(Ordering::Relaxed) == 0 {
            return ClientSurfaceRequest::empty();
        }
        ClientSurfaceRequest::from_bits_truncate(
            self.client_surface_request.swap(0, Ordering::AcqRel),
        )
    }

    /// The address `AttachMetalLayerParams::backing_scale_ptr` carries.
    pub fn backing_scale_ptr(&self) -> u64 {
        (&raw const self.backing_scale) as u64
    }

    /// The address `AttachMetalLayerParams::cursor_kick_ptr` carries.
    pub fn cursor_kick_ptr(&self) -> u64 {
        (&raw const self.cursor_kick) as u64
    }

    /// Wine's retina factor as last published, or `None` before attach published one.
    pub fn display_backing_scale(&self) -> Option<u32> {
        match self.backing_scale.load(Ordering::Relaxed) {
            0 => None,
            scale => Some(scale),
        }
    }

    /// Take the pending cursor re-apply request, if the unix side left one.
    ///
    /// `Acquire` pairs with the unix side's `Release` store; the flag is the
    /// whole message, so nothing else is read behind it.
    pub fn take_cursor_kick(&self) -> bool {
        self.cursor_kick.swap(0, Ordering::AcqRel) != 0
    }
}

impl Default for DisplaySinks {
    fn default() -> Self {
        Self::new()
    }
}

// ── Vtable entry points ──

pub extern "system" fn device_set_cursor_properties(
    this: *mut c_void,
    x_hotspot: u32,
    y_hotspot: u32,
    cursor_bitmap: *mut c_void,
) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    if cursor_bitmap.is_null() {
        warn!(target: LOG_TARGET, "reject SetCursorProperties(null bitmap) → INVALIDCALL");
        return D3DERR_INVALIDCALL;
    }
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    let dev = obj.inner();

    // SAFETY: `cursor_bitmap` is a valid SurfaceHead via the D3D9 ABI;
    // its `vtbl` field is a non-null pointer to a static vtable.
    let surf_head = unsafe { &*(cursor_bitmap as *const SurfaceHead) };
    // SAFETY: same invariant — vtbl pointer is static.
    let surf_vtbl = unsafe { &*surf_head.vtbl };
    let mut desc = D3DSURFACE_DESC {
        format: 0,
        resource_type: 0,
        usage: 0,
        pool: 0,
        multi_sample_type: 0,
        multi_sample_quality: 0,
        width: 0,
        height: 0,
    };
    // SAFETY: calling the just-loaded `get_desc` thunk through
    // `surf_vtbl`; `cursor_bitmap` is the IDirect3DSurface9 `this` per
    // D3D9 ABI and `desc` is a writable local.
    if unsafe { (surf_vtbl.get_desc)(cursor_bitmap, &raw mut desc) } != 0 {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: surface GetDesc failed");
        return D3DERR_INVALIDCALL;
    }
    let width = desc.width;
    let height = desc.height;
    let layout = BitmapLayout {
        width,
        height,
        x_hotspot,
        y_hotspot,
    };
    let mode = crate::direct3d9::current_adapter_display_mode();
    if !layout.valid(
        desc.format,
        (mode.width, mode.height),
        dev.cursor_mut().scale(),
    ) {
        warn!(
            target: LOG_TARGET,
            "reject SetCursorProperties: invalid format, extent or scaled bitmap/hotspot \
             (format={} {width}x{height} hotspot={x_hotspot},{y_hotspot})",
            desc.format,
        );
        return D3DERR_INVALIDCALL;
    }

    let mut locked = D3DLOCKED_RECT {
        pitch: 0,
        bits: null_mut(),
    };
    // SAFETY: calling the just-loaded `lock_rect` thunk through
    // `surf_vtbl`; `cursor_bitmap` is the IDirect3DSurface9 `this` per
    // D3D9 ABI, `locked` is a writable local, and a null rect locks
    // the entire surface.
    if unsafe {
        (surf_vtbl.lock_rect)(
            cursor_bitmap,
            &raw mut locked,
            core::ptr::null(),
            D3DLOCK_READONLY,
        )
    } != 0
    {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: surface LockRect failed");
        return D3DERR_INVALIDCALL;
    }

    let Some(pitch) = layout.row_pitch(locked.bits as usize, locked.pitch) else {
        warn!(target: LOG_TARGET, "reject SetCursorProperties: invalid locked pointer or pitch");
        // SAFETY: this surface was successfully locked above; no pixels were read.
        unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };
        return D3DERR_INVALIDCALL;
    };
    let src = locked.bits as *const u8;
    let cur = dev.cursor_mut();
    // SAFETY: `src` is the surface `LockRect` locked above, whose extent
    // `layout` holds, and `row_pitch` checked `pitch` against it; the lock
    // holds until the unlock below.
    let selected = unsafe { cur.select_bitmap(&layout, src, pitch) };
    // SAFETY: calling the just-loaded `unlock_rect` thunk through
    // `surf_vtbl`; paired with the `lock_rect` call above.
    unsafe { (surf_vtbl.unlock_rect)(cursor_bitmap) };
    let Some(selected) = selected else {
        return D3DERR_INVALIDCALL;
    };
    cur.realize_bitmap(&selected, &layout, &desc);
    D3D_OK
}

pub extern "system" fn device_set_cursor_position(this: *mut c_void, x: i32, y: i32, _flags: u32) {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return;
    };
    obj.inner().cursor_mut().set_position(x, y);
}

pub extern "system" fn device_show_cursor(this: *mut c_void, show: i32) -> i32 {
    let _api = device_api_lock(this);
    let _timer = device_timer(this, DeviceSubCategory::Misc);
    // SAFETY: vtable thunk; `this` is *mut Direct3DDevice9 per IDirect3DDevice9 ABI.
    let Some(obj) = (unsafe { InPtr::<Direct3DDevice9>::opt(this) }) else {
        return D3DERR_INVALIDCALL;
    };
    obj.inner().cursor_mut().show(show)
}

/// Minimal layout for reading the surface's vtable pointer.
///
/// Avoids pulling the full `Direct3DSurface9` type into this module.
#[repr(C)]
struct SurfaceHead {
    vtbl: *const IDirect3DSurface9Vtbl,
}
