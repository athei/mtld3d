//! Unit tests for D3D9 sampler-state translation.
//!
//! The per-field sweep asserts that mutating any `SamplerSnapshot` field changes
//! the cache key, which is what makes a silently dropped sampler state
//! impossible: a new field that never reaches the key fails here. The rest pins
//! the packed key layout by bit position, the 1:1 filter mapping (no implicit
//! promote), and that `params_from_snapshot` agrees with the key it was given.

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
        border_color: 0,
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
