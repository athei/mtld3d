//! The D3DKMT, D3D11 and DXGI values a shared resource is described with.
//!
//! A `D3D9Ex` shared handle is a D3DKMT global handle: a kernel-mode-thunk
//! resource whose private runtime data the creating Direct3D runtime writes
//! and the opening one reads. For a texture that data names D3D11 and DXGI
//! values, and the resource itself is created with the D3DKMT flags here.

/// `D3D_DRIVER_TYPE_HARDWARE`, the driver type `D3D11CreateDevice` takes for a real adapter.
pub const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
/// `D3D11_SDK_VERSION`, the version `D3D11CreateDevice` is called with.
pub const D3D11_SDK_VERSION: u32 = 7;
/// `D3D11_RESOURCE_DIMENSION_TEXTURE2D`.
pub const D3D11_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;
/// `D3D11_BIND_SHADER_RESOURCE`.
pub const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
/// `D3D11_BIND_RENDER_TARGET`.
pub const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
/// `D3D11_RESOURCE_MISC_SHARED`, a resource shared through a legacy global handle.
pub const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
/// `DXGI_FORMAT_B8G8R8A8_UNORM`.
pub const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;
/// `DXGI_FORMAT_B8G8R8X8_UNORM`.
pub const DXGI_FORMAT_B8G8R8X8_UNORM: u32 = 88;

/// The bits of a `D3DKMT_HANDLE` of which a global handle has at least one set.
///
/// An NT handle, which names a resource shared with
/// `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, has neither.
pub const D3DKMT_GLOBAL_HANDLE_MASK: u32 = 0xc000_0000;

/// `D3DKMT_STANDARDALLOCATIONTYPE_EXISTINGHEAP`.
pub const D3DKMT_STANDARDALLOCATIONTYPE_EXISTINGHEAP: u32 = 1;
/// `D3DKMT_CREATEALLOCATIONFLAGS::CreateResource`.
pub const D3DKMT_CREATEALLOCATION_CREATE_RESOURCE: u32 = 1 << 0;
/// `D3DKMT_CREATEALLOCATIONFLAGS::CreateShared`.
pub const D3DKMT_CREATEALLOCATION_CREATE_SHARED: u32 = 1 << 1;
/// `D3DKMT_CREATEALLOCATIONFLAGS::ExistingSysMem`.
pub const D3DKMT_CREATEALLOCATION_EXISTING_SYS_MEM: u32 = 1 << 5;
/// `D3DKMT_CREATEALLOCATIONFLAGS::StandardAllocation`.
pub const D3DKMT_CREATEALLOCATION_STANDARD_ALLOCATION: u32 = 1 << 16;
