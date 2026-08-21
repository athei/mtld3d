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
