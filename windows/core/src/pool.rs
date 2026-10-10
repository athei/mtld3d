//! `D3DPOOL` residency classification and the usage a pool refuses.
//!
//! D3D9 splits the four pools along one axis that matters to a Metal
//! backend: whether the device may ever touch the resource. `D3DPOOL_DEFAULT`
//! and `D3DPOOL_MANAGED` are GPU-resident and get an `MTLTexture`;
//! `D3DPOOL_SYSTEMMEM` and `D3DPOOL_SCRATCH` live in system memory and get
//! none, so their bytes are reachable only through `Lock`, `UpdateTexture` /
//! `UpdateSurface`, and `GetRenderTargetData`.

use mtld3d_types::{
    D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_MANAGED_EX, D3DPOOL_SCRATCH, D3DPOOL_SYSTEMMEM,
    D3DUSAGE_DYNAMIC,
};

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

/// Whether the runtime, not the application, owns the system-memory copy of `pool`.
///
/// `D3DPOOL_MANAGED` is the one pool that keeps such a copy: the runtime
/// uploads it, may drop the device copy whenever it likes, and puts it back
/// from that system-memory copy on the next use. This is what
/// `EvictManagedResources` acts on, and why it acts on nothing else. A
/// `D3DPOOL_DEFAULT` resource has no runtime copy to replay, so its current
/// pixels can exist on the device alone, and the CPU-only pools have no device
/// copy to drop in the first place.
#[must_use]
pub const fn is_runtime_managed(pool: u32) -> bool {
    pool == D3DPOOL_MANAGED
}

/// Whether `usage` asks for something `pool` cannot give a texture.
///
/// `D3DUSAGE_DYNAMIC` says the application rewrites the resource often enough
/// that it wants to drive the copy the device reads. Two pools cannot hand
/// that copy over. `D3DPOOL_MANAGED` says the opposite outright: the runtime
/// owns the system-memory copy and decides when to re-upload it.
/// `D3DPOOL_SCRATCH` has no copy to drive, the device never reads one of its
/// resources at all. D3D9 rejects both pairs at creation with
/// `D3DERR_INVALIDCALL` rather than picking an owner or ignoring the flag, and
/// it does so for every format and every texture type.
#[must_use]
pub const fn usage_conflicts_with_pool(usage: u32, pool: u32) -> bool {
    usage & D3DUSAGE_DYNAMIC != 0 && matches!(pool, D3DPOOL_MANAGED | D3DPOOL_SCRATCH)
}

/// The pool a create that names `pool` makes its resource in.
///
/// `D3DPOOL_MANAGED_EX` is the managed pool under another value, on either
/// kind of device; every other value is the pool it names. A create resolves
/// its pool after [`refused_on_extended`] has seen the value the caller
/// passed, since an extended device refuses `D3DPOOL_MANAGED` and takes
/// `D3DPOOL_MANAGED_EX`.
#[must_use]
pub const fn resolve(pool: u32) -> u32 {
    if pool == D3DPOOL_MANAGED_EX {
        D3DPOOL_MANAGED
    } else {
        pool
    }
}

/// Whether an extended device refuses a create in `pool`.
///
/// An extended device keeps its default-pool resources across `Reset`, so
/// the copy the managed pool exists to restore is never needed, and a create
/// that names `D3DPOOL_MANAGED` is `D3DERR_INVALIDCALL`. It still takes the
/// managed pool under the value `D3DPOOL_MANAGED_EX`, which is not refused
/// here. A plain device refuses nothing here.
#[must_use]
pub const fn refused_on_extended(pool: u32, extended: bool) -> bool {
    extended && pool == D3DPOOL_MANAGED
}

/// Whether `SetPriority` stores a value for a resource in `pool`.
///
/// The priority orders what a memory manager evicts first. On a plain device
/// that is the runtime's managed pool. An extended device lets the driver
/// page its default-pool resources instead, so there the default pool takes
/// the priority. Callers ask with the device's kind for a default-pool
/// resource alone and with `extended` false otherwise, so a managed
/// resource, which an extended device makes through `D3DPOOL_MANAGED_EX`,
/// stores its priority on either kind of device. Every other pair keeps the
/// priority at zero.
#[must_use]
pub const fn priority_settable(pool: u32, extended: bool) -> bool {
    if extended {
        pool == D3DPOOL_DEFAULT
    } else {
        pool == D3DPOOL_MANAGED
    }
}

#[cfg(test)]
mod tests;
