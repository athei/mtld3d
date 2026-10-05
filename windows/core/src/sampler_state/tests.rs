//! Unit tests for D3D9 sampler-state translation.
//!
//! The per-field sweep asserts that mutating any `SamplerSnapshot` field changes
//! the cache key, which is what makes a silently dropped sampler state
//! impossible: a new field that never reaches the key fails here. The rest pins
//! the packed key layout by bit position, the filter mapping (no implicit
//! promote, the filters no sampler offers reading as LINEAR), and that
//! `description_from_snapshot` agrees with the key it was given.

use mtld3d_shared::mtl::{MinMagFilter, MipFilter};
use mtld3d_types::{
    D3DTADDRESS_CLAMP, D3DTADDRESS_WRAP, D3DTEXF_ANISOTROPIC, D3DTEXF_CONVOLUTIONMONO,
    D3DTEXF_GAUSSIANQUAD, D3DTEXF_LINEAR, D3DTEXF_NONE, D3DTEXF_PYRAMIDALQUAD,
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

    let p = description_from_snapshot(&s, fallback);
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
    let s = base();
    let key = key_from_snapshot(&s);
    let p = description_from_snapshot(&s, key);
    assert_eq!(p.id, key.raw());
    assert_eq!(p.max_anisotropy, 1);
    assert_eq!(p.lod_min_clamp.to_bits(), 0.0_f32.to_bits());
    assert_eq!(p.lod_max_clamp.to_bits(), 1000.0_f32.to_bits());

    let mut s2 = base();
    s2.max_mip_level = 3;
    let key2 = key_from_snapshot(&s2);
    let p2 = description_from_snapshot(&s2, key2);
    assert_eq!(p2.id, key2.raw());
    assert_eq!(p2.lod_min_clamp.to_bits(), 3.0_f32.to_bits());
    assert_eq!(p2.lod_max_clamp.to_bits(), 1000.0_f32.to_bits());
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
    let bytes = build_lod_bias_bytes(&biases, &EXPLICIT_LOD_OPEN_ROWS);
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
    // An open explicit row leaves an explicit level where the shader put it.
    assert_eq!(row(0, 2).to_bits(), 0.0_f32.to_bits());
    assert_eq!(row(0, 3).to_bits(), (-f32::MAX).to_bits());
}

#[test]
fn lod_table_bytes_carry_the_explicit_rows() {
    let mut explicit = EXPLICIT_LOD_OPEN_ROWS;
    explicit[5] = [1.5, 2.0];
    let bytes = build_lod_bias_bytes(&[0.0; LOD_BIAS_SLOTS], &explicit);
    let lane = |slot: usize, lane: usize| {
        let base = slot * 16 + lane * 4;
        f32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ])
    };
    assert_eq!(lane(5, 2).to_bits(), 1.5_f32.to_bits(), "offset lane");
    assert_eq!(lane(5, 3).to_bits(), 2.0_f32.to_bits(), "floor lane");
}

#[test]
fn lod_bias_table_cache_rebuilds_for_a_new_explicit_row() {
    let mut cache = LodBiasTableCache::new();
    let biases = [0.0_f32; LOD_BIAS_SLOTS];
    let mut explicit = EXPLICIT_LOD_OPEN_ROWS;
    assert!(cache.update(&biases, &explicit));
    explicit[1] = [0.0, 2.0];
    assert!(cache.update(&biases, &explicit), "a new clamp rebuilds");
    assert_eq!(cache.bytes(), &build_lod_bias_bytes(&biases, &explicit));
    assert!(!cache.update(&biases, &explicit));
}

/// [`linear_state`] with `D3DSAMP_MAXMIPLEVEL`, the texture LOD and the mip filter set.
fn lod_state(max_mip_level: u32, lod: u32, mip_filter: u32) -> [u32; SAMPLER_STATE_COUNT] {
    let mut ss = linear_state();
    ss[D3DSAMP_MAXMIPLEVEL as usize] = max_mip_level;
    ss[TEXTURE_LOD_SLOT] = lod;
    ss[D3DSAMP_MIPFILTER as usize] = mip_filter;
    ss
}

#[test]
fn the_finest_level_is_the_coarser_of_max_mip_level_and_the_lod() {
    let level = |ss| snapshot_from_state(&ss, false).max_mip_level;
    assert_eq!(level(lod_state(0, 0, D3DTEXF_LINEAR)), 0);
    assert_eq!(level(lod_state(2, 0, D3DTEXF_LINEAR)), 2);
    assert_eq!(level(lod_state(1, 3, D3DTEXF_POINT)), 3);
    assert_eq!(level(lod_state(3, 1, D3DTEXF_POINT)), 3);
    assert_eq!(level(lod_state(40, 0, D3DTEXF_POINT)), MAX_MIP_LEVEL);
    assert_eq!(level(lod_state(0, 40, D3DTEXF_POINT)), MAX_MIP_LEVEL);
}

#[test]
fn without_mipmapping_only_the_lod_selects_a_level() {
    let s = snapshot_from_state(&lod_state(3, 0, D3DTEXF_NONE), false);
    assert_eq!(
        s.max_mip_level, 0,
        "MAXMIPLEVEL is ignored without mipmapping"
    );
    let p = description_from_snapshot(&s, key_from_snapshot(&s));
    assert_eq!(p.mip_filter, MipFilter::NotMipmapped);
    assert_eq!(p.lod_min_clamp.to_bits(), 0.0_f32.to_bits());

    let s = snapshot_from_state(&lod_state(0, 2, D3DTEXF_NONE), false);
    assert_eq!(s.max_mip_level, 2);
    let p = description_from_snapshot(&s, key_from_snapshot(&s));
    assert_eq!(
        p.mip_filter,
        MipFilter::Nearest,
        "a sampler without mipmapping would read level 0"
    );
    assert_eq!(p.lod_min_clamp.to_bits(), 2.0_f32.to_bits());
    assert_eq!(
        p.lod_max_clamp.to_bits(),
        2.0_f32.to_bits(),
        "a minified sample must not reach a coarser level"
    );
}

#[test]
fn a_pinned_level_keys_apart_from_a_clamped_one() {
    let pinned = key_from_snapshot(&snapshot_from_state(&lod_state(0, 2, D3DTEXF_NONE), false));
    let clamped = key_from_snapshot(&snapshot_from_state(&lod_state(2, 0, D3DTEXF_POINT), false));
    assert_ne!(pinned, clamped);
}

#[test]
fn the_texture_lod_slot_keys_only_through_the_level_it_selects() {
    let key = |ss| key_from_snapshot(&snapshot_from_state(&ss, false));
    assert_eq!(
        key(lod_state(2, 1, D3DTEXF_LINEAR)),
        key(lod_state(2, 0, D3DTEXF_LINEAR)),
        "a LOD finer than MAXMIPLEVEL selects the same sampler"
    );
}

#[test]
fn explicit_rows_offset_by_the_lod_and_clamp_by_the_finest_level() {
    let row = |ss| explicit_lod_row(&ss);
    assert_eq!(
        row(lod_state(0, 0, D3DTEXF_LINEAR)),
        None,
        "nothing to apply"
    );
    assert_eq!(
        row(lod_state(2, 0, D3DTEXF_LINEAR)),
        Some([0.0, 2.0]),
        "MAXMIPLEVEL clamps"
    );
    assert_eq!(
        row(lod_state(0, 2, D3DTEXF_POINT)),
        Some([2.0, 2.0]),
        "the LOD is a base level the explicit LOD counts from"
    );
    assert_eq!(row(lod_state(3, 1, D3DTEXF_POINT)), Some([1.0, 3.0]));
}

#[test]
fn explicit_rows_carry_the_game_bias() {
    let mut ss = lod_state(0, 0, D3DTEXF_LINEAR);
    ss[D3DSAMP_MIPMAPLODBIAS as usize] = 1.0_f32.to_bits();
    assert_eq!(explicit_lod_row(&ss), Some([1.0, -f32::MAX]));
    ss[TEXTURE_LOD_SLOT] = 1;
    assert_eq!(explicit_lod_row(&ss), Some([2.0, 1.0]));
    ss[D3DSAMP_MIPMAPLODBIAS as usize] = mtld3d_types::FETCH4_ENABLE;
    assert_eq!(
        explicit_lod_row(&ss),
        Some([1.0, 1.0]),
        "a Fetch4 command is no bias"
    );
}

#[test]
fn explicit_rows_without_mipmapping_pin_the_lod_level() {
    assert_eq!(explicit_lod_row(&lod_state(3, 0, D3DTEXF_NONE)), None);
    let mut ss = lod_state(3, 2, D3DTEXF_NONE);
    ss[D3DSAMP_MIPMAPLODBIAS as usize] = 1.0_f32.to_bits();
    let [offset, floor] = explicit_lod_row(&ss).expect("a pinned level");
    assert_eq!(floor.to_bits(), 2.0_f32.to_bits());
    assert_eq!(
        (100.0 + offset).max(floor).to_bits(),
        floor.to_bits(),
        "no explicit LOD the shader asks for leaves the level"
    );
}

#[test]
fn lod_bias_table_cache_rebuilds_only_for_new_input_bits() {
    let mut cache = LodBiasTableCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    biases[3] = -0.75;

    assert!(
        cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "first table is built"
    );
    assert_eq!(
        cache.bytes(),
        &build_lod_bias_bytes(&biases, &EXPLICIT_LOD_OPEN_ROWS)
    );
    assert!(
        !cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "identical inputs reuse the table"
    );

    biases[3] = 0.0;
    biases[7] = -0.75;
    assert!(
        cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "a sampled-slot transition rebuilds the effective table"
    );
    assert_eq!(
        cache.bytes(),
        &build_lod_bias_bytes(&biases, &EXPLICIT_LOD_OPEN_ROWS)
    );
}

#[test]
fn lod_bias_table_cache_keys_signed_zero_and_nan_by_bits() {
    let mut cache = LodBiasTableCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    assert!(cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS));

    biases[2] = -0.0;
    assert!(
        cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "signed zero changes the bias lane"
    );
    assert!(!cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS));

    biases[2] = f32::from_bits(0x7FC0_0001);
    assert!(
        cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "NaN input payload participates in identity"
    );
    assert!(!cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS));
    biases[2] = f32::from_bits(0x7FC0_0002);
    assert!(
        cache.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "a different NaN payload rebuilds"
    );
}

#[test]
fn lod_bias_table_cache_is_independent_of_pass_bindings() {
    let mut table = LodBiasTableCache::new();
    let mut bound = LastBoundCache::new();
    let mut biases = [0.0_f32; LOD_BIAS_SLOTS];
    biases[0] = -0.75;

    assert!(table.update(&biases, &EXPLICIT_LOD_OPEN_ROWS));
    assert!(bound.ps_lod_bias_changed(table.bytes()));
    assert!(!table.update(&biases, &EXPLICIT_LOD_OPEN_ROWS));
    assert!(!bound.ps_lod_bias_changed(table.bytes()));

    bound.reset();
    assert!(
        !table.update(&biases, &EXPLICIT_LOD_OPEN_ROWS),
        "a new pass reuses the derived table"
    );
    assert!(
        bound.ps_lod_bias_changed(table.bytes()),
        "a new pass still binds the table"
    );
}

/// [`linear_state`] with an anisotropic min filter.
fn anisotropic_state() -> [u32; SAMPLER_STATE_COUNT] {
    let mut ss = linear_state();
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_ANISOTROPIC;
    ss
}

#[test]
fn anisotropy_clamps_to_the_advertised_ceiling() {
    let mut ss = anisotropic_state();
    ss[D3DSAMP_MAXANISOTROPY as usize] = 64;
    let s = snapshot_from_state(&ss, false);
    let p = description_from_snapshot(&s, key_from_snapshot(&s));
    assert_eq!(p.max_anisotropy, MAX_ANISOTROPY);
    ss[D3DSAMP_MAXANISOTROPY as usize] = 0;
    let s = snapshot_from_state(&ss, false);
    let p = description_from_snapshot(&s, key_from_snapshot(&s));
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

    let mut ss = anisotropic_state();
    ss[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_CLAMP;
    ss[D3DSAMP_ADDRESSV as usize] = D3DTADDRESS_CLAMP;
    ss[D3DSAMP_MAXANISOTROPY as usize] = MAX_ANISOTROPY;
    ss[D3DSAMP_MAXMIPLEVEL as usize] = 2;
    ss[D3DSAMP_BORDERCOLOR as usize] = 0xFFFF_FFFF;
    ss[D3DSAMP_SRGBTEXTURE as usize] = 1;
    assert_eq!(
        key_from_snapshot(&snapshot_from_state(&ss, false)).raw(),
        0x142_1013_3222,
        "anisotropic min, linear mag and mip, clamped, sRGB, white border"
    );
}

#[test]
fn max_anisotropy_needs_an_anisotropic_filter() {
    // MAXANISOTROPY alone leaves a LINEAR stage isotropic, so it reads the
    // default 1 and keys as the isotropic sampler it builds.
    let mut ss = linear_state();
    ss[D3DSAMP_MAXANISOTROPY as usize] = MAX_ANISOTROPY;
    let s = snapshot_from_state(&ss, false);
    assert_eq!(s.max_anisotropy, 1, "LINEAR filters sample isotropically");
    assert_eq!(
        key_from_snapshot(&s),
        key_from_snapshot(&base()),
        "MAXANISOTROPY without an anisotropic filter shares the isotropic key"
    );
    assert_eq!(
        description_from_snapshot(&s, key_from_snapshot(&s)).max_anisotropy,
        1
    );

    for state in [D3DSAMP_MINFILTER, D3DSAMP_MAGFILTER, D3DSAMP_MIPFILTER] {
        let mut aniso = ss;
        aniso[state as usize] = D3DTEXF_ANISOTROPIC;
        let s = snapshot_from_state(&aniso, false);
        assert_eq!(
            u32::from(s.max_anisotropy),
            MAX_ANISOTROPY,
            "an ANISOTROPIC D3DSAMP_{state} turns anisotropy on"
        );
    }
}

#[test]
fn filters_without_a_sampler_filter_read_as_linear() {
    // The quad filters and CONVOLUTIONMONO name filtering no sampler offers;
    // they filter linearly, as ANISOTROPIC does apart from its anisotropy.
    for filter in [
        D3DTEXF_ANISOTROPIC,
        D3DTEXF_PYRAMIDALQUAD,
        D3DTEXF_GAUSSIANQUAD,
        D3DTEXF_CONVOLUTIONMONO,
    ] {
        let mut ss = sampler_state_defaults();
        ss[D3DSAMP_MINFILTER as usize] = filter;
        ss[D3DSAMP_MAGFILTER as usize] = filter;
        ss[D3DSAMP_MIPFILTER as usize] = filter;
        let s = snapshot_from_state(&ss, false);
        assert_eq!(u32::from(s.min_filter), D3DTEXF_LINEAR, "min {filter}");
        assert_eq!(u32::from(s.mag_filter), D3DTEXF_LINEAR, "mag {filter}");
        assert_eq!(u32::from(s.mip_filter), D3DTEXF_LINEAR, "mip {filter}");
        let p = description_from_snapshot(&s, key_from_snapshot(&s));
        assert_eq!(p.min_filter, MinMagFilter::Linear, "min {filter}");
        assert_eq!(p.mag_filter, MinMagFilter::Linear, "mag {filter}");
        assert_eq!(p.mip_filter, MipFilter::Linear, "mip {filter}");
    }
}

#[test]
fn a_min_or_mag_filter_of_none_samples_as_point() {
    // NONE turns mipmapping off but names no min or mag filter; it samples
    // as POINT and keys as POINT, while the mip filter keeps it.
    let mut ss = sampler_state_defaults();
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_NONE;
    ss[D3DSAMP_MAGFILTER as usize] = D3DTEXF_NONE;
    let s = snapshot_from_state(&ss, false);
    assert_eq!(u32::from(s.min_filter), D3DTEXF_POINT);
    assert_eq!(u32::from(s.mag_filter), D3DTEXF_POINT);
    assert_eq!(u32::from(s.mip_filter), D3DTEXF_NONE);
    assert_eq!(
        key_from_snapshot(&s),
        key_from_snapshot(&snapshot_from_state(&sampler_state_defaults(), false))
    );
}

#[test]
fn out_of_space_states_read_their_d3d9_reading() {
    // `SetSamplerState` takes a DWORD. A filter no `D3DTEXF_*` names is above
    // LINEAR and filters linearly; an address mode no `D3DTADDRESS_*` names
    // reads the default WRAP. 0x11 and 0x13 sit above the four bits the key
    // packs, so their low nibbles name POINT and CLAMP.
    let mut ss = sampler_state_defaults();
    ss[D3DSAMP_MINFILTER as usize] = 0x11;
    ss[D3DSAMP_MAGFILTER as usize] = 0x1_0000;
    ss[D3DSAMP_MIPFILTER as usize] = 9;
    ss[D3DSAMP_ADDRESSU as usize] = 0x13;
    ss[D3DSAMP_ADDRESSV as usize] = 0;
    ss[D3DSAMP_ADDRESSW as usize] = 6;
    let s = snapshot_from_state(&ss, false);
    assert_eq!(u32::from(s.min_filter), D3DTEXF_LINEAR, "MINFILTER");
    assert_eq!(u32::from(s.mag_filter), D3DTEXF_LINEAR, "MAGFILTER");
    assert_eq!(u32::from(s.mip_filter), D3DTEXF_LINEAR, "MIPFILTER");
    assert_eq!(u32::from(s.address_u), D3DTADDRESS_WRAP, "ADDRESSU default");
    assert_eq!(u32::from(s.address_v), D3DTADDRESS_WRAP, "ADDRESSV default");
    assert_eq!(u32::from(s.address_w), D3DTADDRESS_WRAP, "ADDRESSW default");

    // The default the address substitution reads is the spec one.
    let defaults = sampler_state_defaults();
    assert_eq!(defaults[D3DSAMP_ADDRESSU as usize], D3DTADDRESS_WRAP);
}

#[test]
fn out_of_space_states_key_as_what_they_read() {
    // The bug: a state above bit 3 keyed as its low nibble while the params
    // translated the whole DWORD, so whichever of the two states reached the
    // cache first decided what the other one drew with.
    let mut ss = linear_state();
    ss[D3DSAMP_MINFILTER as usize] = 0x11;
    ss[D3DSAMP_ADDRESSU as usize] = 0x13;
    let wide = key_from_snapshot(&snapshot_from_state(&ss, false));

    let mut narrow_ss = linear_state();
    narrow_ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_LINEAR;
    narrow_ss[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_WRAP;
    let narrow = key_from_snapshot(&snapshot_from_state(&narrow_ss, false));
    assert_eq!(wide, narrow, "the substituted readings key as themselves");

    let mut aliased = linear_state();
    aliased[D3DSAMP_MINFILTER as usize] = D3DTEXF_POINT;
    aliased[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_CLAMP;
    assert_ne!(
        wide,
        key_from_snapshot(&snapshot_from_state(&aliased, false)),
        "0x11 must not share the POINT sampler's key"
    );
}

#[test]
fn anisotropy_is_limited_before_the_key() {
    // The key used to carry the raw low byte while the params clamped, so
    // maxanisotropy 5 and 0x105 shared a key and 5 was served whichever
    // sampler arrived first.
    let key_for = |value: u32| {
        let mut ss = anisotropic_state();
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

#[test]
fn sampling_unfiltered_keeps_mip_selection_and_blends_nothing() {
    let mut ss = anisotropic_state();
    ss[D3DSAMP_MAXANISOTROPY as usize] = MAX_ANISOTROPY;
    sample_unfiltered(&mut ss);
    let s = snapshot_from_state(&ss, false);
    assert_eq!(u32::from(s.min_filter), D3DTEXF_POINT, "min");
    assert_eq!(u32::from(s.mag_filter), D3DTEXF_POINT, "mag");
    assert_eq!(
        u32::from(s.mip_filter),
        D3DTEXF_POINT,
        "a blending mip filter points"
    );
    assert_eq!(s.max_anisotropy, 1, "no filter is anisotropic any more");

    let mut ss = linear_state();
    ss[D3DSAMP_MIPFILTER as usize] = D3DTEXF_NONE;
    sample_unfiltered(&mut ss);
    assert_eq!(
        ss[D3DSAMP_MIPFILTER as usize], D3DTEXF_NONE,
        "mipmapping stays off"
    );
}

#[test]
fn fetch4_commands_are_not_lod_biases() {
    let mut states = mtld3d_types::sampler_state_defaults();
    for command in [u32::from_le_bytes(*b"GET4"), u32::from_le_bytes(*b"GET1")] {
        states[mtld3d_types::D3DSAMP_MIPMAPLODBIAS as usize] = command;
        assert_eq!(super::lod_bias(&states).to_bits(), 0.0f32.to_bits());
    }
}

#[test]
fn displacement_and_multi_element_sampler_states_are_obsolete() {
    // DMAPOFFSET offsets the displacement map of N-patch tessellation and
    // ELEMENTINDEX picks an element of a multi-element texture; neither
    // feature exists here or on a modern driver, so a write logs at info.
    for type_ in [
        mtld3d_types::D3DSAMP_DMAPOFFSET,
        mtld3d_types::D3DSAMP_ELEMENTINDEX,
    ] {
        assert!(
            matches!(samp_classify(type_), SampClass::Obsolete(_)),
            "D3DSAMP_{type_} is not classified Obsolete"
        );
    }
}

#[test]
fn every_translated_sampler_state_is_consumed() {
    for type_ in [
        D3DSAMP_ADDRESSU,
        D3DSAMP_ADDRESSV,
        D3DSAMP_ADDRESSW,
        D3DSAMP_BORDERCOLOR,
        D3DSAMP_MAGFILTER,
        D3DSAMP_MINFILTER,
        D3DSAMP_MIPFILTER,
        D3DSAMP_MIPMAPLODBIAS,
        D3DSAMP_MAXMIPLEVEL,
        D3DSAMP_MAXANISOTROPY,
        D3DSAMP_SRGBTEXTURE,
    ] {
        assert!(
            matches!(samp_classify(type_), SampClass::Consumed),
            "D3DSAMP_{type_} is not classified Consumed"
        );
    }
}

/// Row `slot` of a vertex LOD table's bytes, as `(offset, floor)` bits.
fn vertex_row(table: &VertexLodTable, slot: usize) -> [u32; 2] {
    let lane = |at: usize| {
        let bytes = table.bytes();
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    [lane(slot * 8), lane(slot * 8 + 4)]
}

#[test]
fn a_new_vertex_lod_table_leaves_every_level_where_the_shader_names_it() {
    let table = VertexLodTable::new();
    assert_eq!(table.mask(), 0);
    assert_eq!(table.bytes().len(), VS_LOD_BYTES);
    for slot in 0..VS_LOD_SLOTS {
        assert_eq!(
            vertex_row(&table, slot),
            EXPLICIT_LOD_OPEN.map(f32::to_bits),
            "slot {slot} is open"
        );
    }
}

#[test]
fn a_vertex_slot_takes_the_row_of_its_lod_bias_and_finest_level() {
    let mut table = VertexLodTable::new();
    table.set_slot(2, &lod_state(3, 1, D3DTEXF_POINT));
    assert_eq!(table.mask(), 0b0100);
    assert_eq!(
        vertex_row(&table, 2),
        [1.0_f32.to_bits(), 3.0_f32.to_bits()]
    );

    let mut biased = lod_state(0, 0, D3DTEXF_LINEAR);
    biased[D3DSAMP_MIPMAPLODBIAS as usize] = (-1.0_f32).to_bits();
    table.set_slot(0, &biased);
    assert_eq!(table.mask(), 0b0101);
    assert_eq!(
        vertex_row(&table, 0),
        [(-1.0_f32).to_bits(), (-f32::MAX).to_bits()]
    );

    table.set_slot(2, &lod_state(0, 0, D3DTEXF_POINT));
    assert_eq!(
        table.mask(),
        0b0001,
        "a slot back at its defaults needs no row"
    );
    assert_eq!(vertex_row(&table, 2), EXPLICIT_LOD_OPEN.map(f32::to_bits));
}

#[test]
fn a_vertex_slot_without_mipmapping_pins_the_texture_lod() {
    let mut table = VertexLodTable::new();
    table.set_slot(1, &lod_state(3, 2, D3DTEXF_NONE));
    assert_eq!(table.mask(), 0b0010);
    assert_eq!(
        vertex_row(&table, 1),
        [(-f32::MAX).to_bits(), 2.0_f32.to_bits()]
    );
}

#[test]
fn a_vertex_lod_table_ignores_slots_past_the_four_vertex_samplers() {
    let mut table = VertexLodTable::new();
    table.set_slot(VS_LOD_SLOTS, &lod_state(3, 1, D3DTEXF_POINT));
    assert_eq!(table.mask(), 0);
    assert_eq!(table.bytes(), VertexLodTable::new().bytes());
}

#[test]
fn the_last_bound_cache_rebinds_the_vertex_lod_table_once_per_pass_and_change() {
    let mut bound = LastBoundCache::new();
    let mut table = VertexLodTable::new();
    table.set_slot(0, &lod_state(2, 0, D3DTEXF_POINT));
    assert!(bound.vs_lod_changed(table.bytes()));
    assert!(!bound.vs_lod_changed(table.bytes()));
    table.set_slot(0, &lod_state(3, 0, D3DTEXF_POINT));
    assert!(bound.vs_lod_changed(table.bytes()), "a new row rebinds");
    bound.reset();
    assert!(bound.vs_lod_changed(table.bytes()), "a new pass rebinds");
}
