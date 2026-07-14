//! D3D9 primitive types and the full FVF (flexible vertex format) flag set.
//!
//! These are the D3D9 ABI values for `D3DPRIMITIVETYPE` and the FVF DWORD.
//! `mtld3d-core::convert` maps FVF → vertex attributes; the integration-test
//! harness reads the position/colour bits directly — both resolve the flags
//! from here so there is a single home for the encoding.

// ── D3DPRIMITIVETYPE (`D3DPT_*`) ──

pub const D3DPT_POINTLIST: u32 = 1;
pub const D3DPT_LINELIST: u32 = 2;
pub const D3DPT_LINESTRIP: u32 = 3;
pub const D3DPT_TRIANGLELIST: u32 = 4;
pub const D3DPT_TRIANGLESTRIP: u32 = 5;
pub const D3DPT_TRIANGLEFAN: u32 = 6;

// ── D3DFVF flags ──

/// Position-type field of the FVF DWORD (`XYZ` through `XYZB5`).
pub const D3DFVF_POSITION_MASK: u32 = 0x0000_000E;
pub const D3DFVF_XYZ: u32 = 0x0000_0002;
pub const D3DFVF_XYZRHW: u32 = 0x0000_0004;
pub const D3DFVF_XYZB1: u32 = 0x0000_0006;
pub const D3DFVF_XYZB2: u32 = 0x0000_0008;
pub const D3DFVF_XYZB3: u32 = 0x0000_000A;
pub const D3DFVF_XYZB4: u32 = 0x0000_000C;
pub const D3DFVF_XYZB5: u32 = 0x0000_000E;
pub const D3DFVF_XYZW: u32 = 0x0000_4002;
pub const D3DFVF_NORMAL: u32 = 0x0000_0010;
pub const D3DFVF_PSIZE: u32 = 0x0000_0020;
pub const D3DFVF_DIFFUSE: u32 = 0x0000_0040;
pub const D3DFVF_SPECULAR: u32 = 0x0000_0080;
pub const D3DFVF_TEX1: u32 = 0x0000_0100; // 1 texcoord set << D3DFVF_TEXCOUNT_SHIFT(8)
pub const D3DFVF_TEXCOUNT_MASK: u32 = 0x0000_0F00;
pub const D3DFVF_TEXCOUNT_SHIFT: u32 = 8;
pub const D3DFVF_LASTBETA_UBYTE4: u32 = 0x0000_1000;
pub const D3DFVF_LASTBETA_D3DCOLOR: u32 = 0x0000_8000;

// FVF texcoord-size selector, applied per-texcoord at bit offset 16 + i*2.
// `_FORMAT2 = 0` is the wildcard fallback in the texcoord-size match; no
// reified constant is needed for it.
pub const D3DFVF_TEXTUREFORMAT3: u32 = 1;
pub const D3DFVF_TEXTUREFORMAT4: u32 = 2;
pub const D3DFVF_TEXTUREFORMAT1: u32 = 3;
