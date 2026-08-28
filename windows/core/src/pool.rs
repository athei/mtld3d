//! `D3DPOOL` residency classification.
//!
//! D3D9 splits the four pools along one axis that matters to a Metal
//! backend: whether the device may ever touch the resource. `D3DPOOL_DEFAULT`
//! and `D3DPOOL_MANAGED` are GPU-resident and get an `MTLTexture`;
//! `D3DPOOL_SYSTEMMEM` and `D3DPOOL_SCRATCH` live in system memory and get
//! none, so their bytes are reachable only through `Lock`, `UpdateTexture` /
//! `UpdateSurface`, and `GetRenderTargetData`.

use mtld3d_types::{D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM};

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

#[cfg(test)]
mod tests;
