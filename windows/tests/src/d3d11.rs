//! A minimal Direct3D 11 device, for the tests that share a texture with it.
//!
//! An extended D3D9 device shares textures with Direct3D 11 through handles, so a test of
//! that needs the other runtime in the same process. This is the least of it:
//! a hardware device, a shared BGRA render-target texture and its handle,
//! and a clear. `d3d11.dll` is loaded by name rather
//! than imported, so a Wine without it still runs the suite, and every method
//! is called through its vtable slot, which the Direct3D 11 ABI fixes.

use core::ffi::{c_char, c_void};

use mtld3d_types::{
    D3D_DRIVER_TYPE_HARDWARE, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
    D3D11_RESOURCE_MISC_SHARED, D3D11_SDK_VERSION, DXGI_FORMAT_B8G8R8A8_UNORM,
};

/// `IID_IDXGIResource`, `035f3ab4-482e-4e50-b41f-8a7f8bd8960b`.
const IID_DXGI_RESOURCE: [u8; 16] = [
    0xb4, 0x3a, 0x5f, 0x03, 0x2e, 0x48, 0x50, 0x4e, 0xb4, 0x1f, 0x8a, 0x7f, 0x8b, 0xd8, 0x96, 0x0b,
];

/// Vtable slots, counted from `IUnknown::QueryInterface` at 0.
mod slot {
    pub const QUERY_INTERFACE: usize = 0;
    pub const RELEASE: usize = 2;
    pub const DEVICE_CREATE_TEXTURE2D: usize = 5;
    pub const DEVICE_CREATE_RENDER_TARGET_VIEW: usize = 9;
    pub const CONTEXT_CLEAR_RENDER_TARGET_VIEW: usize = 50;
    pub const CONTEXT_FLUSH: usize = 111;
    pub const DXGI_RESOURCE_GET_SHARED_HANDLE: usize = 8;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(name: *const c_char) -> usize;
    fn GetModuleHandleA(name: *const c_char) -> usize;
    fn GetProcAddress(module: usize, name: *const c_char) -> *mut c_void;
}

/// A hardware Direct3D 11 device and its immediate context.
pub struct D3d11Device {
    device: *mut c_void,
    context: *mut c_void,
}

/// An `ID3D11Texture2D` the device created.
pub struct D3d11Texture {
    texture: *mut c_void,
}

impl D3d11Device {
    /// A hardware device, `None` when the Wine has no Direct3D 11 that makes one.
    #[must_use]
    pub fn create() -> Option<Self> {
        // SAFETY: kernel32 export with a NUL-terminated name.
        let module = unsafe { LoadLibraryA(c"d3d11.dll".as_ptr()) };
        if module == 0 {
            return None;
        }
        // SAFETY: kernel32 export; `module` is loaded and the name NUL-terminated.
        let entry = unsafe { GetProcAddress(module, c"D3D11CreateDevice".as_ptr()) };
        if entry.is_null() {
            return None;
        }
        // SAFETY: `entry` is d3d11's `D3D11CreateDevice`, whose signature is
        // the function-pointer type it is turned into.
        let create = unsafe {
            core::mem::transmute::<
                *mut c_void,
                unsafe extern "system" fn(
                    adapter: *mut c_void,
                    driver_type: u32,
                    software: usize,
                    flags: u32,
                    feature_levels: *const u32,
                    feature_level_count: u32,
                    sdk_version: u32,
                    device: *mut *mut c_void,
                    feature_level: *mut u32,
                    context: *mut *mut c_void,
                ) -> i32,
            >(entry)
        };
        let mut device = core::ptr::null_mut();
        let mut context = core::ptr::null_mut();
        // SAFETY: both out slots are writable and every optional argument is
        // null or zero, which asks for the default adapter and feature levels.
        let hr = unsafe {
            create(
                core::ptr::null_mut(),
                D3D_DRIVER_TYPE_HARDWARE,
                0,
                0,
                core::ptr::null(),
                0,
                D3D11_SDK_VERSION,
                &raw mut device,
                core::ptr::null_mut(),
                &raw mut context,
            )
        };
        (hr >= 0 && !device.is_null() && !context.is_null()).then_some(Self { device, context })
    }

    /// Whether the device is DXMT's, the runtime whose shared handles D3D9 opens.
    ///
    /// DXMT reaches Metal through `winemetal.dll`, which no other Direct3D 11
    /// loads, so the module being present once a device exists names it.
    #[must_use]
    pub fn is_dxmt(&self) -> bool {
        // SAFETY: kernel32 export with a NUL-terminated name.
        let module = unsafe { GetModuleHandleA(c"winemetal.dll".as_ptr()) };
        module != 0
    }

    /// A shared BGRA render-target texture and the legacy handle that names it.
    ///
    /// `None` when the create or `IDXGIResource::GetSharedHandle` fails.
    #[must_use]
    pub fn create_shared_texture(&self, width: u32, height: u32) -> Option<(D3d11Texture, usize)> {
        // D3D11_TEXTURE2D_DESC: Width, Height, MipLevels, ArraySize, Format,
        // SampleDesc.Count, SampleDesc.Quality, Usage, BindFlags,
        // CPUAccessFlags, MiscFlags.
        let desc: [u32; 11] = [
            width,
            height,
            1,
            1,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            1,
            0,
            0,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
            0,
            D3D11_RESOURCE_MISC_SHARED,
        ];
        let mut texture = core::ptr::null_mut();
        // SAFETY: slot 5 of a live `ID3D11Device` is `CreateTexture2D`.
        let entry = unsafe { vtbl_entry(self.device, slot::DEVICE_CREATE_TEXTURE2D) };
        // SAFETY: the entry has `CreateTexture2D`'s signature.
        let create = unsafe {
            core::mem::transmute::<
                *const c_void,
                unsafe extern "system" fn(
                    *mut c_void,
                    *const u32,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32,
            >(entry)
        };
        // SAFETY: `desc` is a live description, no initial data, a writable out slot.
        let hr = unsafe {
            create(
                self.device,
                desc.as_ptr(),
                core::ptr::null(),
                &raw mut texture,
            )
        };
        if hr < 0 || texture.is_null() {
            return None;
        }
        let texture = D3d11Texture { texture };
        let resource = query_interface(texture.texture, &IID_DXGI_RESOURCE)?;
        let mut handle = 0usize;
        // SAFETY: slot 8 of a live `IDXGIResource` is `GetSharedHandle`.
        let entry = unsafe { vtbl_entry(resource, slot::DXGI_RESOURCE_GET_SHARED_HANDLE) };
        // SAFETY: the entry has `GetSharedHandle`'s signature.
        let get = unsafe {
            core::mem::transmute::<
                *const c_void,
                unsafe extern "system" fn(*mut c_void, *mut usize) -> i32,
            >(entry)
        };
        // SAFETY: `resource` is live and `handle` a writable `HANDLE` slot.
        let hr = unsafe { get(resource, &raw mut handle) };
        release(resource);
        (hr >= 0 && handle != 0).then_some((texture, handle))
    }

    /// Clear `texture` to `rgba` through a render-target view and flush the context.
    ///
    /// Returns whether the view was created. The clear is submitted when this
    /// returns, not finished: nothing here waits for the GPU.
    #[must_use]
    pub fn clear(&self, texture: &D3d11Texture, rgba: [f32; 4]) -> bool {
        let mut view = core::ptr::null_mut();
        // SAFETY: slot 9 of a live `ID3D11Device` is `CreateRenderTargetView`.
        let entry = unsafe { vtbl_entry(self.device, slot::DEVICE_CREATE_RENDER_TARGET_VIEW) };
        // SAFETY: the entry has `CreateRenderTargetView`'s signature.
        let create = unsafe {
            core::mem::transmute::<
                *const c_void,
                unsafe extern "system" fn(
                    *mut c_void,
                    *mut c_void,
                    *const c_void,
                    *mut *mut c_void,
                ) -> i32,
            >(entry)
        };
        // SAFETY: a live device and resource, the default view, a writable out slot.
        let hr = unsafe {
            create(
                self.device,
                texture.texture,
                core::ptr::null(),
                &raw mut view,
            )
        };
        if hr < 0 || view.is_null() {
            return false;
        }
        // SAFETY: slot 50 of a live `ID3D11DeviceContext` is `ClearRenderTargetView`.
        let entry = unsafe { vtbl_entry(self.context, slot::CONTEXT_CLEAR_RENDER_TARGET_VIEW) };
        // SAFETY: the entry has `ClearRenderTargetView`'s signature.
        let clear = unsafe {
            core::mem::transmute::<
                *const c_void,
                unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32),
            >(entry)
        };
        // SAFETY: a live context and view, and four floats to read.
        unsafe { clear(self.context, view, rgba.as_ptr()) };
        release(view);
        // SAFETY: slot 111 of a live `ID3D11DeviceContext` is `Flush`.
        let entry = unsafe { vtbl_entry(self.context, slot::CONTEXT_FLUSH) };
        // SAFETY: the entry has `Flush`'s signature.
        let flush = unsafe {
            core::mem::transmute::<*const c_void, unsafe extern "system" fn(*mut c_void)>(entry)
        };
        // SAFETY: a live context.
        unsafe { flush(self.context) };
        true
    }
}

impl Drop for D3d11Device {
    fn drop(&mut self) {
        release(self.context);
        release(self.device);
    }
}

impl Drop for D3d11Texture {
    fn drop(&mut self) {
        release(self.texture);
    }
}

/// `QueryInterface` on a live COM object, `None` when it has no such interface.
fn query_interface(object: *mut c_void, iid: &[u8; 16]) -> Option<*mut c_void> {
    let mut out = core::ptr::null_mut();
    // SAFETY: slot 0 of every live COM object is `QueryInterface`.
    let entry = unsafe { vtbl_entry(object, slot::QUERY_INTERFACE) };
    // SAFETY: the entry has `QueryInterface`'s signature.
    let query = unsafe {
        core::mem::transmute::<
            *const c_void,
            unsafe extern "system" fn(*mut c_void, *const [u8; 16], *mut *mut c_void) -> i32,
        >(entry)
    };
    // SAFETY: a live object, a live interface id and a writable out slot.
    let hr = unsafe { query(object, core::ptr::from_ref(iid), &raw mut out) };
    (hr >= 0 && !out.is_null()).then_some(out)
}

/// `Release` on a live COM object this module holds a reference on.
fn release(object: *mut c_void) {
    // SAFETY: slot 2 of every live COM object is `Release`.
    let entry = unsafe { vtbl_entry(object, slot::RELEASE) };
    // SAFETY: the entry has `Release`'s signature.
    let release = unsafe {
        core::mem::transmute::<*const c_void, unsafe extern "system" fn(*mut c_void) -> u32>(entry)
    };
    // SAFETY: the caller holds the reference this gives back.
    unsafe { release(object) };
}

/// Entry `index` of the vtable of the COM object at `object`.
///
/// # Safety
///
/// `object` is a live COM interface pointer whose vtable has more than
/// `index` entries.
const unsafe fn vtbl_entry(object: *mut c_void, index: usize) -> *const c_void {
    // SAFETY: a COM object starts with the pointer to its vtable.
    let vtbl = unsafe { object.cast::<*const *const c_void>().read() };
    // SAFETY: per the contract the vtable holds more than `index` entries.
    let entry = unsafe { vtbl.add(index) };
    // SAFETY: `entry` is inside the vtable, as above.
    unsafe { entry.read() }
}
