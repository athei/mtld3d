//! Single source of truth for D3D9 → Metal sampler-state translation.
//!
//! Mirrors `pipeline_state` but for samplers: one `SamplerSnapshot` input
//! drives both the cache `SamplerKey` and the wire-format
//! `CreateSamplerStateParams`. Per-field unit tests assert the
//! static invariant that mutating any snapshot field produces a
//! different key, so the pipeline-style silent-drop bug (state classified
//! Consumed but value never reaches the sampler) is unrepresentable.
//!
//! Translation is 1:1 with no implicit promotes. Promoting
//! `MIPFILTER NONE → LINEAR` or `MINFILTER LINEAR → ANISOTROPIC` would layer
//! aniso onto box-filter-generated mip chains on textures the game intended
//! to be sampled bilinearly, producing distance shimmer that the 1:1 mapping
//! does not.

use std::fmt;

use mtld3d_shared::{CreateSamplerStateParams, MetalHandle, mtl_handle::MTLDeviceKind};
use mtld3d_types::{
    D3DSAMP_ADDRESSU, D3DSAMP_ADDRESSV, D3DSAMP_ADDRESSW, D3DSAMP_MAGFILTER, D3DSAMP_MAXANISOTROPY,
    D3DSAMP_MAXMIPLEVEL, D3DSAMP_MINFILTER, D3DSAMP_MIPFILTER, D3DSAMP_SRGBTEXTURE,
    SAMPLER_STATE_COUNT,
};

use crate::convert::{
    d3d_to_metal_address_mode, d3d_to_metal_min_mag_filter, d3d_to_metal_mip_filter,
};

/// Upper LOD clamp passed to every `MTLSamplerDescriptor`.
///
/// The fixed `1000.0f` is the D3D9 convention for "no upper clamp", with
/// Metal naturally capping selection at the texture's actual mip count.
const LOD_MAX_CLAMP: f32 = 1000.0;

bitflags::bitflags! {
    /// Sampler cache-key booleans that aren't sourced from a D3DSAMP slot.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct SamplerFlags: u8 {
        /// Set when the sampler is bound to a depth-format texture (sampleable shadow map).
        ///
        /// Adds `compareFunction = LessEqual` to the Metal sampler so MSL
        /// `sample_compare` returns the D3D9 hardware-shadow PCF result
        /// (1 = lit, 0 = shadowed). Folded into the cache key so a single
        /// D3D9 sampler bound to both colour and depth slots in different
        /// draws cleanly fans out into two `MTLSamplerState`s.
        const IS_COMPARE = 1 << 0;
        /// `D3DSAMP_SRGBTEXTURE`.
        ///
        /// The game requests `sRGB → linear` decode on texture read so
        /// subsequent shading math runs in linear space. Doesn't affect the
        /// Metal sampler descriptor — sRGB is expressed as the *texture
        /// view*'s pixel format, not sampler state — but folds into the cache
        /// key so the bind-side resolution of which `MTLTexture` handle to
        /// bind (linear vs. sRGB view) stays in sync with the sampler's
        /// intent.
        const SRGB_TEXTURE = 1 << 1;
    }
}

/// Input view of the D3DSAMP state that participates in pipeline/cache decisions.
///
/// Raw D3D values (`u32`) preserve 1:1 fidelity with the game input;
/// translation happens inside `key_from_snapshot` and `params_from_snapshot`
/// so both consumers see identical translated values.
pub struct SamplerSnapshot {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub max_anisotropy: u32,
    /// `D3DSAMP_MAXMIPLEVEL`.
    ///
    /// D3D9 spec: the *minimum* fine mip level the sampler may select
    /// (counterintuitive name). Maps to Metal's `setLodMinClamp`.
    /// Zero = default (no clamp).
    pub max_mip_level: u32,
    /// Cache-key booleans not sourced from a D3DSAMP slot value (`IS_COMPARE` / `SRGB_TEXTURE`).
    ///
    /// See [`SamplerFlags`].
    pub flags: SamplerFlags,
}

/// Packed-bits sampler cache key.
///
/// Layout (u64 low-to-high):
/// - 0..3   `min_filter`
/// - 4..7   `mag_filter`
/// - 8..11  `mip_filter`
/// - 12..15 `address_u`
/// - 16..19 `address_v`
/// - 20..23 `address_w`
/// - 24..31 `max_anisotropy`
/// - 32..36 `max_mip_level` (5 bits — D3D9 mip count fits in 5)
/// - 37     `is_compare` (depth-bound shadow sampler)
/// - 38     `srgb_texture` (`D3DSAMP_SRGBTEXTURE` — picks linear vs sRGB texture-view at bind)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerKey(u64);

impl fmt::LowerHex for SamplerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl SamplerKey {
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

/// Build a `SamplerSnapshot` from the device's per-stage D3DSAMP array.
///
/// `is_compare` is supplied separately by the caller (encoder) from the
/// per-stage depth-sampler mask — it isn't a D3DSAMP_* slot.
#[must_use]
pub const fn snapshot_from_state(
    ss: &[u32; SAMPLER_STATE_COUNT],
    is_compare: bool,
) -> SamplerSnapshot {
    // const fn: bitflags `.set()` isn't const, so build the bit pattern
    // directly from the flag constants' `.bits()`.
    let mut flag_bits = 0u8;
    if is_compare {
        flag_bits |= SamplerFlags::IS_COMPARE.bits();
    }
    if ss[D3DSAMP_SRGBTEXTURE as usize] != 0 {
        flag_bits |= SamplerFlags::SRGB_TEXTURE.bits();
    }
    SamplerSnapshot {
        min_filter: ss[D3DSAMP_MINFILTER as usize],
        mag_filter: ss[D3DSAMP_MAGFILTER as usize],
        mip_filter: ss[D3DSAMP_MIPFILTER as usize],
        address_u: ss[D3DSAMP_ADDRESSU as usize],
        address_v: ss[D3DSAMP_ADDRESSV as usize],
        address_w: ss[D3DSAMP_ADDRESSW as usize],
        max_anisotropy: ss[D3DSAMP_MAXANISOTROPY as usize],
        max_mip_level: ss[D3DSAMP_MAXMIPLEVEL as usize],
        flags: SamplerFlags::from_bits_truncate(flag_bits),
    }
}

#[must_use]
pub const fn key_from_snapshot(s: &SamplerSnapshot) -> SamplerKey {
    SamplerKey(
        (s.min_filter as u64 & 0xF)
            | ((s.mag_filter as u64 & 0xF) << 4)
            | ((s.mip_filter as u64 & 0xF) << 8)
            | ((s.address_u as u64 & 0xF) << 12)
            | ((s.address_v as u64 & 0xF) << 16)
            | ((s.address_w as u64 & 0xF) << 20)
            | ((s.max_anisotropy as u64 & 0xFF) << 24)
            | ((s.max_mip_level as u64 & 0x1F) << 32)
            | ((s.flags.contains(SamplerFlags::IS_COMPARE) as u64) << 37)
            | ((s.flags.contains(SamplerFlags::SRGB_TEXTURE) as u64) << 38),
    )
}

/// Translate a snapshot into the wire-format `CreateSamplerStateParams`.
///
/// # Panics
///
/// Panics if `s.max_mip_level` exceeds `u16::MAX`. Unreachable in practice:
/// D3D9's max mip count is 14 (a 16384-pixel texture has 15 mip levels).
#[must_use]
pub fn params_from_snapshot(
    s: &SamplerSnapshot,
    key: SamplerKey,
    device_handle: MetalHandle<MTLDeviceKind>,
) -> CreateSamplerStateParams {
    // D3DSAMP_MAXMIPLEVEL is a u32 but the spec caps it at the mip count
    // (≤14 in practice). u16::try_from + f32::from is exact.
    let max_mip_u16 = u16::try_from(s.max_mip_level).expect("D3D9 mip level ≤ 14 fits u16");
    CreateSamplerStateParams {
        device_handle,
        id: key.raw(),
        min_filter: d3d_to_metal_min_mag_filter(s.min_filter),
        mag_filter: d3d_to_metal_min_mag_filter(s.mag_filter),
        mip_filter: d3d_to_metal_mip_filter(s.mip_filter),
        address_u: d3d_to_metal_address_mode(s.address_u),
        address_v: d3d_to_metal_address_mode(s.address_v),
        address_w: d3d_to_metal_address_mode(s.address_w),
        max_anisotropy: s.max_anisotropy.max(1),
        lod_min_clamp: f32::from(max_mip_u16).to_bits(),
        lod_max_clamp: LOD_MAX_CLAMP.to_bits(),
        is_compare: u32::from(s.flags.contains(SamplerFlags::IS_COMPARE)),
        sampler_handle: MetalHandle::NULL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> SamplerSnapshot {
        SamplerSnapshot {
            min_filter: 2, // D3DTEXF_LINEAR
            mag_filter: 2, // D3DTEXF_LINEAR
            mip_filter: 2, // D3DTEXF_LINEAR
            address_u: 1,  // D3DTADDRESS_WRAP
            address_v: 1,  // D3DTADDRESS_WRAP
            address_w: 1,  // D3DTADDRESS_WRAP
            max_anisotropy: 1,
            max_mip_level: 0,
            flags: SamplerFlags::empty(),
        }
    }

    #[test]
    fn key_changes_on_every_field() {
        let k0 = key_from_snapshot(&base());
        let mutate = |f: fn(&mut SamplerSnapshot)| {
            let mut s = base();
            f(&mut s);
            key_from_snapshot(&s)
        };
        assert_ne!(k0, mutate(|s| s.min_filter = 1), "min_filter");
        assert_ne!(k0, mutate(|s| s.mag_filter = 1), "mag_filter");
        assert_ne!(k0, mutate(|s| s.mip_filter = 1), "mip_filter");
        assert_ne!(k0, mutate(|s| s.address_u = 2), "address_u");
        assert_ne!(k0, mutate(|s| s.address_v = 2), "address_v");
        assert_ne!(k0, mutate(|s| s.address_w = 2), "address_w");
        assert_ne!(k0, mutate(|s| s.max_anisotropy = 8), "max_anisotropy");
        assert_ne!(k0, mutate(|s| s.max_mip_level = 3), "max_mip_level");
        assert_ne!(
            k0,
            mutate(|s| s.flags.insert(SamplerFlags::IS_COMPARE)),
            "is_compare"
        );
        assert_ne!(
            k0,
            mutate(|s| s.flags.insert(SamplerFlags::SRGB_TEXTURE)),
            "srgb_texture"
        );
    }

    #[test]
    fn srgb_texture_lives_in_bit_38() {
        // is_compare lives in bit 37 — the next free bit is 38, where
        // srgb_texture must land so existing key consumers don't shift.
        let mut s = base();
        s.flags.insert(SamplerFlags::SRGB_TEXTURE);
        let k = key_from_snapshot(&s);
        assert_eq!(k.raw() & (1 << 38), 1 << 38);
        assert_eq!(k.raw() & (1 << 37), 0); // is_compare untouched
    }

    #[test]
    fn raw_filters_pass_through_1_to_1() {
        // Sampler translation is identity — no LINEAR→ANISO or NONE→LINEAR
        // promote. Verify each raw filter value lands unchanged in the key.
        let mut s = base();
        s.min_filter = 2; // D3DTEXF_LINEAR
        s.mip_filter = 0; // D3DTEXF_NONE
        s.max_anisotropy = 1;
        let k = key_from_snapshot(&s);
        assert_eq!(k.raw() & 0xF, 2, "min_filter raw=LINEAR preserved");
        assert_eq!((k.raw() >> 8) & 0xF, 0, "mip_filter raw=NONE preserved");
        assert_eq!((k.raw() >> 24) & 0xFF, 1, "max_anisotropy preserved");
    }

    #[test]
    fn params_match_snapshot_on_default() {
        // SAFETY: tests; opaque values never dereferenced.
        let dev = unsafe { MetalHandle::new(0xDEAD) };
        let s = base();
        let key = key_from_snapshot(&s);
        let p = params_from_snapshot(&s, key, dev);
        assert_eq!(p.device_handle, dev);
        assert_eq!(p.id, key.raw());
        assert_eq!(p.max_anisotropy, 1);
        assert_eq!(p.lod_min_clamp, 0.0_f32.to_bits());
        assert_eq!(p.lod_max_clamp, 1000.0_f32.to_bits());

        let mut s2 = base();
        s2.max_mip_level = 3;
        let key2 = key_from_snapshot(&s2);
        let p2 = params_from_snapshot(&s2, key2, dev);
        assert_eq!(p2.id, key2.raw());
        assert_eq!(p2.lod_min_clamp, 3.0_f32.to_bits());
        assert_eq!(p2.lod_max_clamp, 1000.0_f32.to_bits());
    }
}
