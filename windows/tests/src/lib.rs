//! Shared support library for the mtld3d end-to-end test suite.
//!
//! Every integration test under `tests/` drives the real `d3d9.dll` through a
//! [`Harness`]: one factory + window + device, with safe wrappers around the
//! COM vtables and RAII `resource` handles. All FFI and `unsafe` live here so
//! the test files read as plain D3D9 call sequences with pixel/`HRESULT`
//! assertions. D3D9 constants come from `mtld3d_types`; nothing is restated.

mod check;
mod ffi;
mod harness;
mod pixel;
mod resource;
mod vertex;
mod vtbl;
mod win32;

pub use harness::{DrawIndexedUpParams, Harness, HarnessConfig};
pub use pixel::{Rgba8, assert_pixel_approx, assert_pixel_eq};
pub use resource::{
    BufferLock, IndexBuffer, LockedRect, PixelShader, Query, StateBlock, Surface, Texture,
    VertexBuffer, VertexDeclaration, VertexShader,
};
pub use vertex::{
    LitVertex, PosColorVertex, PosVertex, RhwVertex, SpecularVertex, TexturedVertex, Vertex,
};
