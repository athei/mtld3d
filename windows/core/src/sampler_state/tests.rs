//! Unit tests for D3D9 sampler-state translation.
//!
//! The per-field sweep asserts that mutating any `SamplerSnapshot` field changes
//! the cache key, which is what makes a silently dropped sampler state
//! impossible: a new field that never reaches the key fails here. The rest pins
//! the packed key layout by bit position, the 1:1 filter mapping (no implicit
//! promote), and that `params_from_snapshot` agrees with the key it was given.

use mtld3d_shared::mtl::MinMagFilter;
use mtld3d_types::{
    D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP, D3DTEXF_GAUSSIANQUAD, D3DTEXF_LINEAR, D3DTEXF_NONE,
};

use super::*;
use crate::passes::LastBoundCache;

/// The D3DSAMP array the tests start from, with the three filters on LINEAR.
fn linear_state() -> [u32; SAMPLER_STATE_COUNT] {
    let mut ss = sampler_state_defaults();
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_LINEAR;
    ss[D3DSAMP_MAGFILTER as usize] = D3DTEXF_LINEAR;
    ss[D3DSAMP_MIPFILTER as usize] = D3DTEXF_LINEAR;
    ss
}

fn base() -> SamplerSnapshot {
    snapshot_from_state(&linear_state(), false)
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
    assert_ne!(k0, mutate(|s| s.border_color = 0xFFFF_FFFF), "border_color");
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
fn border_preset_lives_in_bits_39_and_40() {
    // The key carries the Metal preset, not the raw D3DCOLOR: two colours
    // that reduce to the same preset share a sampler, and only the three
    // presets (plus the black fallback) can ever appear in the field.
    let mut s = base();
    s.border_color = 0xFFFF_FFFF;
    let white = key_from_snapshot(&s);
    assert_eq!((white.raw() >> 39) & 0x3, BorderColor::OpaqueWhite as u64);
    assert_eq!(white.raw() & (1 << 38), 0, "srgb_texture untouched");

    s.border_color = 0xFF00_0000;
    let black = key_from_snapshot(&s);
    assert_eq!((black.raw() >> 39) & 0x3, BorderColor::OpaqueBlack as u64);

    s.border_color = 0xFF10_2030;
    let fallback = key_from_snapshot(&s);
    assert_eq!(
        fallback, black,
        "non-preset colour shares the black sampler"
    );

    // SAFETY: tests; opaque values never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let p = params_from_snapshot(&s, fallback, dev);
    assert_eq!(p.border_color, BorderColor::OpaqueBlack);
}

#[test]
fn raw_filters_pass_through_1_to_1() {
    // Sampler translation is identity — no LINEAR→ANISO or NONE→LINEAR
    // promote. Verify each raw filter value lands unchanged in the key.
    let mut ss = linear_state();
    ss[D3DSAMP_MIPFILTER as usize] = D3DTEXF_NONE;
    ss[D3DSAMP_MAXANISOTROPY as usize] = 1;
    let k = key_from_snapshot(&snapshot_from_state(&ss, false));
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

#[test]
fn lod_bias_decodes_the_raw_float() {
    let mut ss = [0u32; SAMPLER_STATE_COUNT];
    assert_eq!(
        lod_bias(&ss).to_bits(),
        0.0_f32.to_bits(),
        "default is zero"
    );

    ss[D3DSAMP_MIPMAPLODBIAS as usize] = (-1.5_f32).to_bits();
    assert_eq!(lod_bias(&ss).to_bits(), (-1.5_f32).to_bits());

    ss[D3DSAMP_MIPMAPLODBIAS as usize] = 2.25_f32.to_bits();
    assert_eq!(lod_bias(&ss).to_bits(), 2.25_f32.to_bits());
}

#[test]
fn lod_bias_folds_nan_and_clamps_the_magnitude() {
    let mut ss = [0u32; SAMPLER_STATE_COUNT];
    ss[D3DSAMP_MIPMAPLODBIAS as usize] = f32::NAN.to_bits();
    assert_eq!(
        lod_bias(&ss).to_bits(),
        0.0_f32.to_bits(),
        "NaN reads as no bias"
    );

    ss[D3DSAMP_MIPMAPLODBIAS as usize] = f32::INFINITY.to_bits();
    assert_eq!(lod_bias(&ss).to_bits(), LOD_BIAS_LIMIT.to_bits());

    ss[D3DSAMP_MIPMAPLODBIAS as usize] = f32::NEG_INFINITY.to_bits();
    assert_eq!(lod_bias(&ss).to_bits(), (-LOD_BIAS_LIMIT).to_bits());
}

#[test]
fn lod_bias_active_ignores_both_zeroes() {
    assert!(!lod_bias_active(0.0));
    assert!(!lod_bias_active(-0.0));
    assert!(lod_bias_active(0.25));
    assert!(lod_bias_active(-0.25));
}

#[test]
fn lod_bias_bytes_carry_the_bias_and_its_exponent() {
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    biases[3] = 2.0;
    let bytes = build_lod_bias_bytes(&biases);
    assert_eq!(bytes.len(), LOD_BIAS_BYTES);

    let row = |slot: usize, lane: usize| {
        let base = slot * 16 + lane * 4;
        f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ])
    };
    assert_eq!(row(3, 0).to_bits(), 2.0_f32.to_bits(), "bias lane");
    assert_eq!(row(3, 1).to_bits(), 4.0_f32.to_bits(), "exp2 lane");
    // An unbiased slot must leave the sample unshifted: bias 0, scale 1.
    assert_eq!(row(0, 0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(row(0, 1).to_bits(), 1.0_f32.to_bits());
}

#[test]
fn lod_bias_table_cache_rebuilds_only_for_new_input_bits() {
    let mut cache = LodBiasTableCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    biases[3] = -0.75;

    assert!(cache.update(&biases), "first table is built");
    assert_eq!(cache.bytes(), &build_lod_bias_bytes(&biases));
    assert!(!cache.update(&biases), "identical inputs reuse the table");

    biases[3] = 0.0;
    biases[7] = -0.75;
    assert!(
        cache.update(&biases),
        "a sampled-slot transition rebuilds the effective table"
    );
    assert_eq!(cache.bytes(), &build_lod_bias_bytes(&biases));
}

#[test]
fn lod_bias_table_cache_keys_signed_zero_and_nan_by_bits() {
    let mut cache = LodBiasTableCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    assert!(cache.update(&biases));

    biases[2] = -0.0;
    assert!(cache.update(&biases), "signed zero changes the bias lane");
    assert!(!cache.update(&biases));

    biases[2] = f32::from_bits(0x7FC0_0001);
    assert!(
        cache.update(&biases),
        "NaN input payload participates in identity"
    );
    assert!(!cache.update(&biases));
    biases[2] = f32::from_bits(0x7FC0_0002);
    assert!(cache.update(&biases), "a different NaN payload rebuilds");
}

#[test]
fn lod_bias_table_cache_is_independent_of_pass_bindings() {
    let mut table = LodBiasTableCache::new();
    let mut bound = LastBoundCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    biases[0] = -0.75;

    assert!(table.update(&biases));
    assert!(bound.ps_lod_bias_changed(table.bytes()));
    assert!(!table.update(&biases));
    assert!(!bound.ps_lod_bias_changed(table.bytes()));

    bound.reset();
    assert!(
        !table.update(&biases),
        "a new pass reuses the derived table"
    );
    assert!(
        bound.ps_lod_bias_changed(table.bytes()),
        "a new pass still binds the table"
    );
}

#[test]
fn anisotropy_clamps_to_the_advertised_ceiling() {
    // SAFETY: tests; opaque values never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let mut ss = linear_state();
    ss[D3DSAMP_MAXANISOTROPY as usize] = 64;
    let s = snapshot_from_state(&ss, false);
    let p = params_from_snapshot(&s, key_from_snapshot(&s), dev);
    assert_eq!(p.max_anisotropy, MAX_ANISOTROPY);
    ss[D3DSAMP_MAXANISOTROPY as usize] = 0;
    let s = snapshot_from_state(&ss, false);
    let p = params_from_snapshot(&s, key_from_snapshot(&s), dev);
    assert_eq!(p.max_anisotropy, 1);
}

#[test]
fn in_space_states_keep_their_key_layout() {
    // The narrowing must not move a bit for a state inside its D3D9 value
    // space: these three keys were read off the module before it landed, and
    // a shift would re-key every sampler a running game has cached.
    let defaults = snapshot_from_state(&sampler_state_defaults(), false);
    assert_eq!(key_from_snapshot(&defaults).raw(), 0x0111_1011, "defaults");

    let compare = snapshot_from_state(&sampler_state_defaults(), true);
    assert_eq!(
        key_from_snapshot(&compare).raw(),
        0x20_0111_1011,
        "defaults, depth-bound"
    );

    let mut ss = linear_state();
    ss[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_CLAMP;
    ss[D3DSAMP_ADDRESSV as usize] = D3DTADDRESS_CLAMP;
    ss[D3DSAMP_MAXANISOTROPY as usize] = MAX_ANISOTROPY;
    ss[D3DSAMP_MAXMIPLEVEL as usize] = 2;
    ss[D3DSAMP_BORDERCOLOR as usize] = 0xFFFF_FFFF;
    ss[D3DSAMP_SRGBTEXTURE as usize] = 1;
    assert_eq!(
        key_from_snapshot(&snapshot_from_state(&ss, false)).raw(),
        0x142_1013_3222,
        "trilinear, clamped, anisotropic, sRGB, white border"
    );
}

#[test]
fn an_in_space_filter_the_translator_skips_still_reaches_its_fallback() {
    // `D3DTEXF_GAUSSIANQUAD` is a D3D9 filter the Metal translation has no
    // arm for. It is inside the space, so it keeps reaching that translator's
    // own logged fallback instead of being substituted here.
    // SAFETY: tests; opaque values never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let mut ss = linear_state();
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_GAUSSIANQUAD;
    let s = snapshot_from_state(&ss, false);
    assert_eq!(u32::from(s.min_filter), D3DTEXF_GAUSSIANQUAD);
    let p = params_from_snapshot(&s, key_from_snapshot(&s), dev);
    assert_eq!(p.min_filter, MinMagFilter::Nearest);
}

#[test]
fn out_of_space_states_read_the_d3d9_default() {
    // `SetSamplerState` takes a DWORD. 0x12 and 0x13 sit above the four bits
    // the key packs, so the low nibble used to name LINEAR and CLAMP while
    // the translation took its unmapped arm.
    let mut ss = linear_state();
    ss[D3DSAMP_MINFILTER as usize] = 0x12;
    ss[D3DSAMP_MAGFILTER as usize] = 0x1_0000;
    ss[D3DSAMP_MIPFILTER as usize] = 9;
    ss[D3DSAMP_ADDRESSU as usize] = 0x13;
    ss[D3DSAMP_ADDRESSV as usize] = 0;
    ss[D3DSAMP_ADDRESSW as usize] = 6;
    let s = snapshot_from_state(&ss, false);
    assert_eq!(u32::from(s.min_filter), D3DTEXF_POINT, "MINFILTER default");
    assert_eq!(u32::from(s.mag_filter), D3DTEXF_POINT, "MAGFILTER default");
    assert_eq!(u32::from(s.mip_filter), D3DTEXF_NONE, "MIPFILTER default");
    assert_eq!(u32::from(s.address_u), D3DTADDRESS_WRAP, "ADDRESSU default");
    assert_eq!(u32::from(s.address_v), D3DTADDRESS_WRAP, "ADDRESSV default");
    assert_eq!(u32::from(s.address_w), D3DTADDRESS_WRAP, "ADDRESSW default");

    // The defaults the substitution reads are the spec ones.
    let defaults = sampler_state_defaults();
    assert_eq!(defaults[D3DSAMP_MINFILTER as usize], D3DTEXF_POINT);
    assert_eq!(defaults[D3DSAMP_MIPFILTER as usize], D3DTEXF_NONE);
    assert_eq!(defaults[D3DSAMP_ADDRESSU as usize], D3DTADDRESS_WRAP);
}

#[test]
fn out_of_space_states_key_as_the_default_they_read() {
    // The bug: a state above bit 3 keyed as its low nibble while the params
    // translated the whole DWORD, so whichever of the two states reached the
    // cache first decided what the other one drew with.
    let mut ss = linear_state();
    ss[D3DSAMP_MINFILTER as usize] = 0x12;
    ss[D3DSAMP_ADDRESSU as usize] = 0x13;
    let wide = key_from_snapshot(&snapshot_from_state(&ss, false));

    let mut narrow_ss = linear_state();
    narrow_ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_POINT;
    narrow_ss[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_WRAP;
    let narrow = key_from_snapshot(&snapshot_from_state(&narrow_ss, false));
    assert_eq!(wide, narrow, "the substituted defaults key as themselves");

    let mut aliased = linear_state();
    aliased[D3DSAMP_MINFILTER as usize] = D3DTEXF_LINEAR;
    aliased[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_CLAMP;
    assert_ne!(
        wide,
        key_from_snapshot(&snapshot_from_state(&aliased, false)),
        "0x12 must not share the LINEAR sampler's key"
    );
}

#[test]
fn anisotropy_is_limited_before_the_key() {
    // The key used to carry the raw low byte while the params clamped, so
    // maxanisotropy 5 and 0x105 shared a key and 5 was served whichever
    // sampler arrived first.
    let key_for = |value: u32| {
        let mut ss = linear_state();
        ss[D3DSAMP_MAXANISOTROPY as usize] = value;
        key_from_snapshot(&snapshot_from_state(&ss, false))
    };
    assert_eq!(key_for(0x105), key_for(MAX_ANISOTROPY), "both limit alike");
    assert_ne!(key_for(0x105), key_for(5), "0x105 is not 5");
    assert_eq!(key_for(0), key_for(1), "zero reads as the D3D9 default");
}

#[test]
fn forcing_point_keeps_the_filters_in_space() {
    let mut s = base();
    s.force_point_filter();
    assert_eq!(u32::from(s.min_filter), D3DTEXF_POINT);
    assert_eq!(u32::from(s.mag_filter), D3DTEXF_POINT);
    assert_eq!(u32::from(s.mip_filter), D3DTEXF_NONE);
    assert_eq!(s.max_anisotropy, 1);
    assert_eq!(
        key_from_snapshot(&s).raw() & 0xFFF,
        u64::from(D3DTEXF_POINT) | (u64::from(D3DTEXF_POINT) << 4),
        "the forced filters pack into the low three nibbles"
    );
}
