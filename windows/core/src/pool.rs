//! `D3DPOOL` residency classification and the usage a pool refuses.
//!
//! D3D9 splits the four pools along one axis that matters to a Metal
//! backend: whether the device may ever touch the resource. `D3DPOOL_DEFAULT`
//! and `D3DPOOL_MANAGED` are GPU-resident and get an `MTLTexture`;
//! `D3DPOOL_SYSTEMMEM` and `D3DPOOL_SCRATCH` live in system memory and get
//! none, so their bytes are reachable only through `Lock`, `UpdateTexture` /
//! `UpdateSurface`, and `GetRenderTargetData`.

use mtld3d_types::{D3DPOOL_MANAGED, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC};

/// Whether a resource created in `pool` is system memory with no GPU allocation.
///
/// The two CPU-only pools differ from each other only in what the runtime
/// accepts them for (`UpdateTexture` reads a `D3DPOOL_SYSTEMMEM` source and
/// rejects a scratch one), never in where the bytes live, so residency is one
/// predicate over both.
#[must_use]
pub const fn is_cpu_only(pool: u32) -> bool {
    matches!(pool, D3DPOOL_SYSTEMMEM | D3DPOOL_SCRATCH)
}

/// Whether `usage` asks for something `pool` cannot give a texture.
///
/// `D3DUSAGE_DYNAMIC` says the application rewrites the resource often enough
/// that it wants to drive the copy the device reads. `D3DPOOL_MANAGED` says
/// the opposite: the runtime owns the system-memory copy and decides when to
/// re-upload it. D3D9 rejects the pair at creation with `D3DERR_INVALIDCALL`
/// rather than picking one of the two owners, and it does so for every format.
#[must_use]
pub const fn usage_conflicts_with_pool(usage: u32, pool: u32) -> bool {
    usage & D3DUSAGE_DYNAMIC != 0 && pool == D3DPOOL_MANAGED
}

#[cfg(test)]
mod tests;
