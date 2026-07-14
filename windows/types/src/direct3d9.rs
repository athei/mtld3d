use core::ffi::c_void;

use super::D3DCAPS9;

/// SDK version passed to `Direct3DCreate9` (`D3D_SDK_VERSION`).
pub const D3DSDK_VERSION: u32 = 32;

/// `D3DCREATE_HARDWARE_VERTEXPROCESSING` — `CreateDevice` behaviour flag.
pub const D3DCREATE_HARDWARE_VERTEXPROCESSING: u32 = 0x40;

/// `D3DSWAPEFFECT_DISCARD` — `D3DPRESENT_PARAMETERS::SwapEffect`.
pub const D3DSWAPEFFECT_DISCARD: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// `IID_IUnknown` — `{00000000-0000-0000-C000-000000000046}`.
pub const IID_IUNKNOWN: Guid = Guid {
    data1: 0x0000_0000,
    data2: 0x0000,
    data3: 0x0000,
    data4: [0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46],
};

/// `IID_IDirect3DResource9` — `{05EEC05D-8F7D-4362-B999-D1BAF357C704}`.
pub const IID_IDIRECT3DRESOURCE9: Guid = Guid {
    data1: 0x05EE_C05D,
    data2: 0x8F7D,
    data3: 0x4362,
    data4: [0xB9, 0x99, 0xD1, 0xBA, 0xF3, 0x57, 0xC7, 0x04],
};

/// `IID_IDirect3DBaseTexture9` — `{580CA87E-1D3C-4D54-991D-B7D3E3C298CE}`.
pub const IID_IDIRECT3DBASETEXTURE9: Guid = Guid {
    data1: 0x580C_A87E,
    data2: 0x1D3C,
    data3: 0x4D54,
    data4: [0x99, 0x1D, 0xB7, 0xD3, 0xE3, 0xC2, 0x98, 0xCE],
};

/// `IID_IDirect3DTexture9` — `{85C31227-3DE5-4F00-9B3A-F11AC38C18B5}`.
pub const IID_IDIRECT3DTEXTURE9: Guid = Guid {
    data1: 0x85C3_1227,
    data2: 0x3DE5,
    data3: 0x4F00,
    data4: [0x9B, 0x3A, 0xF1, 0x1A, 0xC3, 0x8C, 0x18, 0xB5],
};

/// `SetPrivateData` flag: the data pointer is an `IUnknown*`.
///
/// The runtime `AddRef`s on store and `Release`s on overwrite/free/destroy.
pub const D3DSPD_IUNKNOWN: u32 = 0x0000_0001;

// ── IDirect3D9 vtable ──

#[repr(C)]
pub struct IDirect3D9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3D9
    pub register_software_device: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_adapter_count: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_adapter_identifier:
        unsafe extern "system" fn(*mut c_void, u32, u32, *mut D3DADAPTER_IDENTIFIER9) -> i32,
    pub get_adapter_mode_count: unsafe extern "system" fn(*mut c_void, u32, u32) -> u32,
    pub enum_adapter_modes:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, *mut c_void) -> i32,
    pub get_adapter_display_mode: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub check_device_type: unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, i32) -> i32,
    pub check_device_format:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, u32, u32) -> i32,
    pub check_device_multi_sample_type:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, i32, u32, *mut u32) -> i32,
    pub check_depth_stencil_match:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, u32) -> i32,
    pub check_device_format_conversion:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32) -> i32,
    pub get_device_caps: unsafe extern "system" fn(*mut c_void, u32, u32, *mut D3DCAPS9) -> i32,
    pub get_adapter_monitor: unsafe extern "system" fn(*mut c_void, u32) -> *mut c_void,
    pub create_device: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        *mut c_void,
        u32,
        *mut c_void,
        *mut *mut c_void,
    ) -> i32,
}

#[repr(C)]
pub struct D3DADAPTER_IDENTIFIER9 {
    pub driver: [u8; 512],
    pub description: [u8; 512],
    pub device_name: [u8; 32],
    // Win32 `LARGE_INTEGER DriverVersion` — two 32-bit halves (LowPart/HighPart).
    // Deliberately NOT an `i64`: Rust aligns `i64` to 8, but the 32-bit (i686)
    // Windows ABI aligns 8-byte members to 4, so a caller's stack
    // `D3DADAPTER_IDENTIFIER9` is only 4-aligned there. An 8-aligned field would
    // make `*mut D3DADAPTER_IDENTIFIER9` ops (e.g. zeroing the out-param) trip
    // the misaligned-pointer precondition on that caller. `[u32; 2]` is 4-aligned
    // on every target and keeps the field offset (1056) and struct size identical.
    pub driver_version: [u32; 2],
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    pub device_identifier: [u8; 16],
    pub whql_level: u32,
}
