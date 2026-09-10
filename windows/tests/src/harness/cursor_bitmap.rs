//! Locked cursor-surface fault injection through the real device COM entry point.

use core::ffi::c_void;

use mtld3d_types::{D3DLOCKED_RECT, D3DSURFACE_DESC, IDirect3DSurface9Vtbl};

use super::Harness;
use crate::{resource::Surface, vtbl::deref_vtbl};

/// A live surface proxy whose lock layout is deliberately invalid.
#[repr(C)]
struct BitmapProbe {
    vtbl: *const IDirect3DSurface9Vtbl,
    surface: *mut c_void,
    pitch: i32,
    null_bits: bool,
    locks: u32,
    unlocks: u32,
}

impl Harness {
    /// Call `SetCursorProperties` through a proxy with an invalid locked layout.
    ///
    /// The real surface is locked and unlocked by the proxy. Returns the HRESULT
    /// and the number of successful locks and unlock calls made by the device.
    #[must_use]
    pub fn cursor_with_invalid_layout(
        &self,
        surface: &Surface<'_>,
        pitch: i32,
        null_bits: bool,
    ) -> (i32, u32, u32) {
        // SAFETY: surface is live and starts with its static IDirect3DSurface9 vtable.
        let original = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(surface.as_ptr()) };
        // SAFETY: the vtable contains only function pointers, with no owned fields.
        let mut vtbl = unsafe { core::ptr::read(original) };
        vtbl.get_desc = get_desc;
        vtbl.lock_rect = lock_rect;
        vtbl.unlock_rect = unlock_rect;
        let mut probe = BitmapProbe {
            vtbl: &raw const vtbl,
            surface: surface.as_ptr(),
            pitch,
            null_bits,
            locks: 0,
            unlocks: 0,
        };
        // SAFETY: the proxy and its local vtable outlive this synchronous call.
        // SetCursorProperties uses only the three proxy methods replaced above.
        let hr = unsafe {
            (self.dev_vtbl().set_cursor_properties)(self.device, 0, 0, (&raw mut probe).cast())
        };
        let result = (hr, probe.locks, probe.unlocks);
        if probe.locks > probe.unlocks {
            // Leave the real resource usable even if the regression assertion fails.
            // SAFETY: the proxy successfully locked this still-live surface.
            unsafe { (original.unlock_rect)(surface.as_ptr()) };
        }
        result
    }
}

unsafe extern "system" fn get_desc(this: *mut c_void, output: *mut D3DSURFACE_DESC) -> i32 {
    // SAFETY: installed only on the stack proxy, live for this COM call.
    let probe = unsafe { &*this.cast::<BitmapProbe>() };
    // SAFETY: the caller holds the real surface alive for the proxy's lifetime.
    let vtbl = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(probe.surface) };
    // SAFETY: forwards the device's writable descriptor to the real surface.
    unsafe { (vtbl.get_desc)(probe.surface, output) }
}

unsafe extern "system" fn lock_rect(
    this: *mut c_void,
    output: *mut D3DLOCKED_RECT,
    rect: *const c_void,
    flags: u32,
) -> i32 {
    // SAFETY: installed only on the stack proxy, exclusively called here.
    let probe = unsafe { &mut *this.cast::<BitmapProbe>() };
    // SAFETY: the caller holds the real surface alive for the proxy's lifetime.
    let vtbl = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(probe.surface) };
    // SAFETY: forwards the device's lock arguments to the real surface.
    let hr = unsafe { (vtbl.lock_rect)(probe.surface, output, rect, flags) };
    if hr == 0 {
        probe.locks += 1;
        // SAFETY: successful LockRect initialized the caller's writable output.
        let locked = unsafe { &mut *output };
        locked.pitch = probe.pitch;
        if probe.null_bits {
            locked.bits = core::ptr::null_mut();
        }
    }
    hr
}

unsafe extern "system" fn unlock_rect(this: *mut c_void) -> i32 {
    // SAFETY: installed only on the stack proxy, exclusively called here.
    let probe = unsafe { &mut *this.cast::<BitmapProbe>() };
    probe.unlocks += 1;
    // SAFETY: the caller holds the real surface alive for the proxy's lifetime.
    let vtbl = unsafe { deref_vtbl::<IDirect3DSurface9Vtbl>(probe.surface) };
    // SAFETY: paired with the forwarded successful LockRect.
    unsafe { (vtbl.unlock_rect)(probe.surface) }
}
