//! `IDirect3DVertexDeclaration9` FFI vtable and element definition.

use core::ffi::c_void;

use super::Guid;

/// Matches `D3DVERTEXELEMENT9` in d3d9types.h.
///
/// The declaration array passed to `CreateVertexDeclaration` is an ordered
/// list terminated by `D3DDECL_END` (`stream == 0xFF`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct D3DVERTEXELEMENT9 {
    pub stream: u16,
    pub offset: u16,
    pub type_: u8,
    pub method: u8,
    pub usage: u8,
    pub usage_index: u8,
}

pub const D3DDECL_END_STREAM: u16 = 0xFF;

/// The `D3DDECL_END()` terminator element (`stream == 0xFF`).
///
/// Ends every declaration's element array (`type == D3DDECLTYPE_UNUSED`).
/// Appended when synthesising the implicit declaration for an FVF so a
/// `GetDeclaration` round-trip matches the byte pattern apps expect.
pub const D3DDECL_END: D3DVERTEXELEMENT9 = D3DVERTEXELEMENT9 {
    stream: D3DDECL_END_STREAM,
    offset: 0,
    type_: D3DDECLTYPE_UNUSED,
    method: 0,
    usage: 0,
    usage_index: 0,
};

// D3DDECLTYPE values (d3d9types.h)
pub const D3DDECLTYPE_FLOAT1: u8 = 0;
pub const D3DDECLTYPE_FLOAT2: u8 = 1;
pub const D3DDECLTYPE_FLOAT3: u8 = 2;
pub const D3DDECLTYPE_FLOAT4: u8 = 3;
pub const D3DDECLTYPE_D3DCOLOR: u8 = 4;
pub const D3DDECLTYPE_UBYTE4: u8 = 5;
pub const D3DDECLTYPE_SHORT2: u8 = 6;
pub const D3DDECLTYPE_SHORT4: u8 = 7;
pub const D3DDECLTYPE_UBYTE4N: u8 = 8;
pub const D3DDECLTYPE_SHORT2N: u8 = 9;
pub const D3DDECLTYPE_SHORT4N: u8 = 10;
pub const D3DDECLTYPE_USHORT2N: u8 = 11;
pub const D3DDECLTYPE_USHORT4N: u8 = 12;
pub const D3DDECLTYPE_UDEC3: u8 = 13;
pub const D3DDECLTYPE_DEC3N: u8 = 14;
pub const D3DDECLTYPE_FLOAT16_2: u8 = 15;
pub const D3DDECLTYPE_FLOAT16_4: u8 = 16;
pub const D3DDECLTYPE_UNUSED: u8 = 17;

// D3DDECLUSAGE values (d3d9types.h). 1:1 with DXSO's DeclUsage enum.
pub const D3DDECLUSAGE_POSITION: u8 = 0;
pub const D3DDECLUSAGE_BLENDWEIGHT: u8 = 1;
pub const D3DDECLUSAGE_BLENDINDICES: u8 = 2;
pub const D3DDECLUSAGE_NORMAL: u8 = 3;
pub const D3DDECLUSAGE_PSIZE: u8 = 4;
pub const D3DDECLUSAGE_TEXCOORD: u8 = 5;
pub const D3DDECLUSAGE_TANGENT: u8 = 6;
pub const D3DDECLUSAGE_BINORMAL: u8 = 7;
pub const D3DDECLUSAGE_TESSFACTOR: u8 = 8;
pub const D3DDECLUSAGE_POSITIONT: u8 = 9;
pub const D3DDECLUSAGE_COLOR: u8 = 10;
pub const D3DDECLUSAGE_FOG: u8 = 11;
pub const D3DDECLUSAGE_DEPTH: u8 = 12;
pub const D3DDECLUSAGE_SAMPLE: u8 = 13;

#[repr(C)]
pub struct IDirect3DVertexDeclaration9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DVertexDeclaration9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_declaration:
        unsafe extern "system" fn(*mut c_void, *mut D3DVERTEXELEMENT9, *mut u32) -> i32,
}
