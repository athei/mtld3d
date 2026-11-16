//! The memory DC `IDirect3DSurface9::GetDC` hands out over a surface's own pixel store.

use core::ffi::c_void;

#[link(name = "gdi32")]
unsafe extern "system" {
    fn CreateCompatibleDC(dc: *mut c_void) -> *mut c_void;
    fn DeleteDC(dc: *mut c_void) -> i32;
    fn D3DKMTCreateDCFromMemory(desc: *mut D3DKMT_CREATEDCFROMMEMORY) -> u32;
    fn D3DKMTDestroyDCFromMemory(desc: *const D3DKMT_DESTROYDCFROMMEMORY) -> u32;
}

/// Win32 `D3DKMT_CREATEDCFROMMEMORY`: wraps an existing pixel store, no copy.
///
/// The pixel store is wrapped in a memory DC + DIB section. Unlike
/// `CreateDIBSection`, the kernel side (`NtGdiDdDDICreateDCFromMemory`) leaves
/// `biSizeImage` 0 and stores `biHeight` as `-Height` (top-down), which
/// `GetObject` flips back to `+Height` — matching the `DIBSECTION` values GDI
/// reports for such a DIB. `#[repr(C)]` matches the documented struct; `h_dc` /
/// `h_bitmap` are filled by the call.
// The field names mirror the canonical Win32 SDK names verbatim.
#[repr(C)]
struct D3DKMT_CREATEDCFROMMEMORY {
    p_memory: *mut c_void,
    format: u32,
    width: u32,
    height: u32,
    pitch: u32,
    h_device_dc: *mut c_void,
    p_color_table: *const c_void,
    h_dc: *mut c_void,
    h_bitmap: *mut c_void,
}

/// Win32 `D3DKMT_DESTROYDCFROMMEMORY`: the teardown counterpart.
///
/// Destroys the DC + DIB pair produced by [`D3DKMT_CREATEDCFROMMEMORY`].
#[repr(C)]
struct D3DKMT_DESTROYDCFROMMEMORY {
    h_dc: *mut c_void,
    h_bitmap: *mut c_void,
}

/// The pixel store and geometry a `GetDC` wraps.
pub struct SurfaceDcSource {
    pub memory: *mut u8,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
}

/// The GDI objects backing an outstanding `GetDC`.
///
/// A memory DC and its DIB section, both produced by
/// `D3DKMTCreateDCFromMemory` so the DIB aliases the surface's own pixel store
/// directly (no separate GDI-allocated bits, no seed or write-back copy). Torn
/// down as a pair by `D3DKMTDestroyDCFromMemory`. All-null when no DC is held.
#[derive(Clone, Copy)]
pub struct SurfaceDc {
    dc: *mut c_void,
    bitmap: *mut c_void,
}

impl SurfaceDc {
    /// No DC held.
    pub const NULL: Self = Self {
        dc: core::ptr::null_mut(),
        bitmap: core::ptr::null_mut(),
    };

    /// The `HDC` `GetDC` hands the application.
    pub const fn handle(&self) -> *mut c_void {
        self.dc
    }
}

/// Wrap `source` in a memory DC whose DIB aliases the pixels, `None` when GDI refuses.
///
/// # Safety
///
/// `source.memory` points at `source.height` rows of `source.pitch` bytes. GDI
/// reads and writes them until [`destroy_surface_dc`] tears the DC down, so
/// they stay valid until then.
pub unsafe fn create_surface_dc(source: &SurfaceDcSource) -> Option<SurfaceDc> {
    // Wrap the pixel store in a memory DC + DIB via
    // `D3DKMTCreateDCFromMemory`. Unlike `CreateDIBSection`, the
    // kernel leaves `biSizeImage` 0 and stores a negative top-down `biHeight`,
    // matching the `DIBSECTION` values GDI reports for such a DIB. `h_device_dc`
    // is a throwaway template DC deleted right after the call.
    // SAFETY: `CreateCompatibleDC(NULL)` is the documented form for a memory DC
    // compatible with the screen; null on failure is tolerated (template only —
    // `D3DKMTCreateDCFromMemory` reports the real failure via its status).
    let device_dc = unsafe { CreateCompatibleDC(core::ptr::null_mut()) };
    let mut desc = D3DKMT_CREATEDCFROMMEMORY {
        p_memory: source.memory.cast::<c_void>(),
        format: source.format,
        width: source.width,
        height: source.height,
        pitch: source.pitch,
        h_device_dc: device_dc,
        p_color_table: core::ptr::null(),
        h_dc: core::ptr::null_mut(),
        h_bitmap: core::ptr::null_mut(),
    };
    // SAFETY: `&mut desc` is a live, aligned `D3DKMT_CREATEDCFROMMEMORY`; the call
    // fills `h_dc` / `h_bitmap` on success and returns a non-zero NTSTATUS on
    // failure. The caller keeps `p_memory` valid for the DC's lifetime.
    let status = unsafe { D3DKMTCreateDCFromMemory(&raw mut desc) };
    if !device_dc.is_null() {
        // SAFETY: `device_dc` is the live template memory DC created above; it has
        // served its purpose (the created DC is `desc.h_dc`) and is deleted here.
        unsafe { DeleteDC(device_dc) };
    }
    if status != 0 || desc.h_dc.is_null() {
        return None;
    }
    Some(SurfaceDc {
        dc: desc.h_dc,
        bitmap: desc.h_bitmap,
    })
}

/// Tear down the GDI objects of an outstanding `GetDC`.
///
/// Destroy the memory DC and its DIB section as a pair via
/// `D3DKMTDestroyDCFromMemory` (the counterpart to the
/// `D3DKMTCreateDCFromMemory` that produced them). A no-op on a [`SurfaceDc::NULL`]
/// (no DC held).
pub fn destroy_surface_dc(held: SurfaceDc) {
    if held.dc.is_null() {
        return;
    }
    let desc = D3DKMT_DESTROYDCFROMMEMORY {
        h_dc: held.dc,
        h_bitmap: held.bitmap,
    };
    // SAFETY: `held.dc` + `held.bitmap` are the live DC / DIB pair produced by
    // `D3DKMTCreateDCFromMemory` at `GetDC`; `&desc` is a live, aligned struct
    // read only for the duration of the call.
    unsafe { D3DKMTDestroyDCFromMemory(&raw const desc) };
}
