//! Shader identities shared by the live draw path and the on-disk shader cache.
//!
//! The record kind and the content keys a compiled shader is known by. The draw
//! path and the perf records use them in every build; the cache that stores
//! records under them sits behind `mtld3d-core`'s `disk-cache` feature.

use std::hash::{Hash, Hasher};

use xxhash_rust::xxh3::Xxh3;

use crate::{
    dxso::{VariantKey, VsSamplerKinds},
    ids::ProgramId,
    shader_compile_stats::CompileBucket,
};

/// On-disk record kind.
///
/// Discriminants are wire bytes: never reorder without bumping
/// `SHADER_CACHE_SCHEMA_VERSION`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CachedKind {
    FfVs = 0,
    FfPs = 1,
    Sm1Vs = 2,
    Sm1Ps = 3,
    Sm2Vs = 4,
    Sm2Ps = 5,
    Sm3Vs = 6,
    Sm3Ps = 7,
}

impl CachedKind {
    /// Round-trip helper for the parser.
    ///
    /// `None` if the byte is outside the discriminant range, and the parser
    /// drops the record.
    #[must_use]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::FfVs),
            1 => Some(Self::FfPs),
            2 => Some(Self::Sm1Vs),
            3 => Some(Self::Sm1Ps),
            4 => Some(Self::Sm2Vs),
            5 => Some(Self::Sm2Ps),
            6 => Some(Self::Sm3Vs),
            7 => Some(Self::Sm3Ps),
            _ => None,
        }
    }

    /// Map to the live-compile bucket.
    ///
    /// Pre-warm uses the same `(FF, SM1, SM2, SM3)` breakdown as the
    /// existing burst log.
    #[must_use]
    pub const fn compile_bucket(self) -> CompileBucket {
        match self {
            Self::FfVs | Self::FfPs => CompileBucket::Ff,
            Self::Sm1Vs | Self::Sm1Ps => CompileBucket::Sm1,
            Self::Sm2Vs | Self::Sm2Ps => CompileBucket::Sm2,
            Self::Sm3Vs | Self::Sm3Ps => CompileBucket::Sm3,
        }
    }

    #[must_use]
    pub const fn is_vertex(self) -> bool {
        matches!(self, Self::FfVs | Self::Sm1Vs | Self::Sm2Vs | Self::Sm3Vs)
    }

    #[must_use]
    pub const fn is_pixel(self) -> bool {
        matches!(self, Self::FfPs | Self::Sm1Ps | Self::Sm2Ps | Self::Sm3Ps)
    }

    #[must_use]
    pub const fn is_programmable(self) -> bool {
        !matches!(self, Self::FfVs | Self::FfPs)
    }

    /// Programmable: derive the kind from `(sm_major, is_pixel_shader)`.
    ///
    /// `None` for SM majors d3d9 should never see (DX10+).
    #[must_use]
    pub const fn from_programmable(sm_major: u8, is_pixel: bool) -> Option<Self> {
        match (sm_major, is_pixel) {
            (1, false) => Some(Self::Sm1Vs),
            (1, true) => Some(Self::Sm1Ps),
            (2, false) => Some(Self::Sm2Vs),
            (2, true) => Some(Self::Sm2Ps),
            (3, false) => Some(Self::Sm3Vs),
            (3, true) => Some(Self::Sm3Ps),
            _ => None,
        }
    }

    /// Per-shader Metal entry-point name, e.g. `mtld3d_vs_ff_5f3a0001`, `mtld3d_ps_sm3_a2b1c4d8`.
    ///
    /// The same string is written into the MSL function definition by the
    /// emitter and looked up via `newFunctionWithName:` on the unix side,
    /// so each compiled `MTLFunction` reports a distinct name in Xcode's
    /// pipeline-state inspector. Live-path (`encoder.rs`) and cache-load
    /// (`shader_prewarm.rs`) must share this helper to stay consistent.
    #[must_use]
    pub fn entry_name(self, disk_key: u64) -> String {
        let stage = match self {
            Self::FfVs | Self::Sm1Vs | Self::Sm2Vs | Self::Sm3Vs => "vs",
            Self::FfPs | Self::Sm1Ps | Self::Sm2Ps | Self::Sm3Ps => "ps",
        };
        let kind_label = match self {
            Self::FfVs | Self::FfPs => "ff",
            Self::Sm1Vs | Self::Sm1Ps => "sm1",
            Self::Sm2Vs | Self::Sm2Ps => "sm2",
            Self::Sm3Vs | Self::Sm3Ps => "sm3",
        };
        format!("mtld3d_{stage}_{kind_label}_{disk_key:08x}")
    }
}

/// Hash any `Hash`-implementing FF state key to a u64 disk identifier.
///
/// `FfVsKey` / `FfPsKey` already implement `Hash` via `derive`, so this
/// is a one-liner at every call site.
pub fn ff_key_hash<T: Hash>(key: &T) -> u64 {
    let mut h = Xxh3::new();
    key.hash(&mut h);
    h.finish()
}

/// Content identity of a programmable vertex shader and its emission inputs.
#[must_use]
pub fn vs_source_disk_key_programmable(
    vs_id: ProgramId,
    provided_input_mask: u16,
    clip_plane_count: u8,
    sampler_kinds: VsSamplerKinds,
) -> u64 {
    ff_key_hash(&(
        vs_id.raw(),
        provided_input_mask,
        clip_plane_count,
        sampler_kinds,
    ))
}

/// Content identity of a programmable pixel shader and its emission inputs.
#[must_use]
pub fn ps_source_disk_key_programmable(ps_id: ProgramId, variant: VariantKey) -> u64 {
    ff_key_hash(&(ps_id.raw(), variant))
}

#[cfg(test)]
mod tests;
