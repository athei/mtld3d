use core::ffi::c_void;

use super::D3DCAPS9;

/// SDK version passed to `Direct3DCreate9` (`D3D_SDK_VERSION`).
pub const D3DSDK_VERSION: u32 = 32;

/// `D3DCREATE_HARDWARE_VERTEXPROCESSING` — `CreateDevice` behaviour flag.
pub const D3DCREATE_HARDWARE_VERTEXPROCESSING: u32 = 0x40;

/// `CreateDevice` behaviour flag `D3DCREATE_MULTITHREADED`.
///
/// The app may call the device and its resources from any thread; every
/// entry point then serialises on the device's `ApiLock`.
pub const D3DCREATE_MULTITHREADED: u32 = 0x4;

/// `CreateDevice` behaviour flag `D3DCREATE_NOWINDOWCHANGES`.
///
/// The app manages the device window itself, so the device leaves its style,
/// rect and visibility alone.
pub const D3DCREATE_NOWINDOWCHANGES: u32 = 0x800;

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

/// `IID_IDirect3D9` — `{81BDCBCA-64D4-426D-AE8D-AD0147F4275C}`.
pub const IID_IDIRECT3D9: Guid = Guid {
    data1: 0x81BD_CBCA,
    data2: 0x64D4,
    data3: 0x426D,
    data4: [0xAE, 0x8D, 0xAD, 0x01, 0x47, 0xF4, 0x27, 0x5C],
};

/// `IID_IDirect3DDevice9` — `{D0223B96-BF7A-43FD-92BD-A43B0D82B9EB}`.
pub const IID_IDIRECT3DDEVICE9: Guid = Guid {
    data1: 0xD022_3B96,
    data2: 0xBF7A,
    data3: 0x43FD,
    data4: [0x92, 0xBD, 0xA4, 0x3B, 0x0D, 0x82, 0xB9, 0xEB],
};

/// `IID_IDirect3DSwapChain9` — `{794950F2-ADFC-458A-905E-10A10B0B503B}`.
pub const IID_IDIRECT3DSWAPCHAIN9: Guid = Guid {
    data1: 0x7949_50F2,
    data2: 0xADFC,
    data3: 0x458A,
    data4: [0x90, 0x5E, 0x10, 0xA1, 0x0B, 0x0B, 0x50, 0x3B],
};

/// `IID_IDirect3DSurface9` — `{0CFBAF3A-9FF6-429A-99B3-A2796AF8B89B}`.
pub const IID_IDIRECT3DSURFACE9: Guid = Guid {
    data1: 0x0CFB_AF3A,
    data2: 0x9FF6,
    data3: 0x429A,
    data4: [0x99, 0xB3, 0xA2, 0x79, 0x6A, 0xF8, 0xB8, 0x9B],
};

/// `IID_IDirect3DVertexBuffer9` — `{B64BB1B5-FD70-4DF6-BF91-19D0A12455E3}`.
pub const IID_IDIRECT3DVERTEXBUFFER9: Guid = Guid {
    data1: 0xB64B_B1B5,
    data2: 0xFD70,
    data3: 0x4DF6,
    data4: [0xBF, 0x91, 0x19, 0xD0, 0xA1, 0x24, 0x55, 0xE3],
};

/// `IID_IDirect3DIndexBuffer9` — `{7C9DD65E-D3F7-4529-ACEE-785830ACDE35}`.
pub const IID_IDIRECT3DINDEXBUFFER9: Guid = Guid {
    data1: 0x7C9D_D65E,
    data2: 0xD3F7,
    data3: 0x4529,
    data4: [0xAC, 0xEE, 0x78, 0x58, 0x30, 0xAC, 0xDE, 0x35],
};

/// `IID_IDirect3DVertexDeclaration9` — `{DD13C59C-36FA-4098-A8FB-C7ED39DC8546}`.
pub const IID_IDIRECT3DVERTEXDECLARATION9: Guid = Guid {
    data1: 0xDD13_C59C,
    data2: 0x36FA,
    data3: 0x4098,
    data4: [0xA8, 0xFB, 0xC7, 0xED, 0x39, 0xDC, 0x85, 0x46],
};

/// `IID_IDirect3DVertexShader9` — `{EFC5557E-6265-4613-8A94-43857889EB36}`.
pub const IID_IDIRECT3DVERTEXSHADER9: Guid = Guid {
    data1: 0xEFC5_557E,
    data2: 0x6265,
    data3: 0x4613,
    data4: [0x8A, 0x94, 0x43, 0x85, 0x78, 0x89, 0xEB, 0x36],
};

/// `IID_IDirect3DPixelShader9` — `{6D3BDBDC-5B02-4415-B852-CE5E8BCCB289}`.
pub const IID_IDIRECT3DPIXELSHADER9: Guid = Guid {
    data1: 0x6D3B_DBDC,
    data2: 0x5B02,
    data3: 0x4415,
    data4: [0xB8, 0x52, 0xCE, 0x5E, 0x8B, 0xCC, 0xB2, 0x89],
};

/// `IID_IDirect3DStateBlock9` — `{B07C4FE5-310D-4BA8-A23C-4F0F206F218B}`.
pub const IID_IDIRECT3DSTATEBLOCK9: Guid = Guid {
    data1: 0xB07C_4FE5,
    data2: 0x310D,
    data3: 0x4BA8,
    data4: [0xA2, 0x3C, 0x4F, 0x0F, 0x20, 0x6F, 0x21, 0x8B],
};

/// `IID_IDirect3DQuery9` — `{D9771460-A695-4F26-BBD3-27B840B541CC}`.
pub const IID_IDIRECT3DQUERY9: Guid = Guid {
    data1: 0xD977_1460,
    data2: 0xA695,
    data3: 0x4F26,
    data4: [0xBB, 0xD3, 0x27, 0xB8, 0x40, 0xB5, 0x41, 0xCC],
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

/// `IID_IDirect3DCubeTexture9`: `{FFF32F81-D953-473A-9223-93D652ABA93F}`.
pub const IID_IDIRECT3DCUBETEXTURE9: Guid = Guid {
    data1: 0xFFF3_2F81,
    data2: 0xD953,
    data3: 0x473A,
    data4: [0x92, 0x23, 0x93, 0xD6, 0x52, 0xAB, 0xA9, 0x3F],
};

/// `IID_IDirect3DVolumeTexture9`: `{2518526C-E789-4111-A7B9-47EF328D13E6}`.
pub const IID_IDIRECT3DVOLUMETEXTURE9: Guid = Guid {
    data1: 0x2518_526C,
    data2: 0xE789,
    data3: 0x4111,
    data4: [0xA7, 0xB9, 0x47, 0xEF, 0x32, 0x8D, 0x13, 0xE6],
};

/// `IID_IDirect3DVolume9`: `{24F416E6-1F67-4AA7-B88E-D33F6F3128A1}`.
pub const IID_IDIRECT3DVOLUME9: Guid = Guid {
    data1: 0x24F4_16E6,
    data2: 0x1F67,
    data3: 0x4AA7,
    data4: [0xB8, 0x8E, 0xD3, 0x3F, 0x6F, 0x31, 0x28, 0xA1],
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
