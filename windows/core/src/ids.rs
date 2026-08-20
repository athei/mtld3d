//! Strongly-typed identifiers for the encoder's cache keys.
//!
//! Each newtype wraps a private `u64` and is constructed only through a
//! domain-specific factory that takes authentic source material — there is no
//! raw-u64 constructor, so miswiring (e.g. passing a Metal handle where a
//! `TextureId` is expected) is a compile error.

use std::{
    fmt,
    hash::Hasher,
    sync::atomic::{AtomicU64, Ordering},
};

use xxhash_rust::xxh3::Xxh3;

pub use crate::{depth_stencil_state::DepthStencilKey, sampler_state::SamplerKey};

/// Content-hash identity for a parsed DXSO program.
///
/// Stable across destroy/recreate of shader COM objects carrying
/// identical bytecode.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProgramId(u64);

/// Process-unique id for an `IDirect3DTexture9`.
///
/// Keys the encoder's `texture_cache` and survives across draws.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureId(u64);

/// Process-unique id for an `IDirect3DVertexBuffer9` / `IDirect3DIndexBuffer9`.
///
/// Keys the encoder's `buffer_cache` so a VB/IB's lazily-wrapped `MTLBuffer`
/// survives across draws. Minted once at Create and never reused.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferId(u64);

impl ProgramId {
    /// Mint from a DXSO token stream.
    ///
    /// The token bytes are hashed into a stable u64 that survives
    /// shader-object churn.
    ///
    /// `xxh3_64` (not `DefaultHasher`) so the value is stable across
    /// `rustc` versions — the same `ProgramId` is the on-disk shader-cache
    /// key (`shader_cache.rs`), so a hasher whose output is allowed to
    /// shift between toolchains would silently invalidate the cache.
    #[must_use]
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut h = Xxh3::new();
        for &t in tokens {
            h.write_u32(t);
        }
        Self(h.finish())
    }

    /// Raw u64 for use as the on-disk shader-cache key.
    ///
    /// The disk record stores this directly so the warm-cache lookup at
    /// next launch reproduces the same `ProgramId`.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl TextureId {
    /// Mint the next process-wide unique texture id.
    pub fn new_unique() -> Self {
        Self(NEXT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Inner u64.
    ///
    /// Used as a dedup key for `log_once_trace_by!` at the
    /// `texture_unlock_rect` deferred-upload trace site.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl BufferId {
    /// Mint the next process-wide unique buffer id.
    pub fn new_unique() -> Self {
        Self(NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Inner u64.
    ///
    /// Used as a dedup key for `log_once_trace_by!` at the VB/IB
    /// wrap-fail early-return sites in `emit_draw` so one trace line
    /// fires per distinct failing buffer.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::LowerHex for ProgramId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for TextureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

static NEXT_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
