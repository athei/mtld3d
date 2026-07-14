//! `IDirect3DIndexBuffer9` FFI vtable and descriptor.

use core::ffi::c_void;

use super::Guid;

#[repr(C)]
pub struct D3DINDEXBUFFER_DESC {
    pub format: u32,
    pub resource_type: u32,
    pub usage: u32,
    pub pool: u32,
    pub size: u32,
}

#[repr(C)]
pub struct IDirect3DIndexBuffer9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DResource9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub set_priority: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_priority: unsafe extern "system" fn(*mut c_void) -> u32,
    pub pre_load: unsafe extern "system" fn(*mut c_void),
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DIndexBuffer9
    pub lock: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void, u32) -> i32,
    pub unlock: unsafe extern "system" fn(*mut c_void) -> i32,
    pub get_desc: unsafe extern "system" fn(*mut c_void, *mut D3DINDEXBUFFER_DESC) -> i32,
}
