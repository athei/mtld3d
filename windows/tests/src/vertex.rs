//! `#[repr(C)]` vertex layouts shared by the end-to-end draw tests.
//!
//! Each struct matches a D3D9 FVF the device understands; the field order is
//! the byte order the vertex shader / fixed-function input assembler expects.
//! `Harness::draw_primitive_up` is generic over these and derives the stride
//! from `size_of`, so a test never restates a vertex size.

/// Position + diffuse colour (`D3DFVF_XYZ | D3DFVF_DIFFUSE`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// `D3DCOLOR` in ARGB byte order.
    pub color: u32,
}

/// Position + diffuse colour + one texture coordinate.
///
/// The FVF is `D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TexturedVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: u32,
    pub u: f32,
    pub v: f32,
}

/// Position + normal (`D3DFVF_XYZ | D3DFVF_NORMAL`).
///
/// For lit FF draws that take their colors from the material.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LitVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

/// Position + diffuse + specular colour (`D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_SPECULAR`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpecularVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// `D3DCOLOR` in ARGB byte order.
    pub diffuse: u32,
    /// `D3DCOLOR` in ARGB byte order.
    pub specular: u32,
}

/// Position only (`D3DFVF_XYZ`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PosVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Position + diffuse colour, used where a distinct name aids the test's intent.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PosColorVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: u32,
}

/// Pre-transformed screen-space position + diffuse colour (`D3DFVF_XYZRHW | D3DFVF_DIFFUSE`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RhwVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rhw: f32,
    pub color: u32,
}
