//! Unit tests for the depth/stencil snapshot, its cache key, and the wire params.
//!
//! Every snapshot field must reach the key, and the key packs translated Metal enums so an
//! out-of-range `SetRenderState` value cannot alias a valid one and be handed the wrong
//! `MTLDepthStencilState`. The folding rules are pinned too: a disabled depth or stencil test
//! collapses its own fields, the CCW states stay inert until two-sided mode, the masks narrow to
//! the width `Stencil8` observes, and each clear-quad state keeps a key of its own.

use mtld3d_types::{
    D3DCMP_ALWAYS, D3DCMP_EQUAL, D3DCMP_LESSEQUAL, D3DSTENCILOP_KEEP, D3DSTENCILOP_REPLACE,
    render_state_defaults,
};

use super::*;

/// D3D enum constant at the snapshot's narrow width.
fn narrow(v: u32) -> u8 {
    u8::try_from(v).expect("D3D9 enum render-state value ≤ u8::MAX")
}

fn base() -> DepthStencilSnapshot {
    let mut s = snapshot_from_state(&render_state_defaults());
    s.stencil_enable = 1;
    s
}

#[test]
fn key_changes_on_every_field() {
    let k0 = key_from_snapshot(&base());
    let mutate = |f: fn(&mut DepthStencilSnapshot)| {
        let mut s = base();
        f(&mut s);
        key_from_snapshot(&s)
    };
    assert_ne!(k0, mutate(|s| s.depth_enable = 0), "depth_enable");
    assert_ne!(k0, mutate(|s| s.depth_write = 0), "depth_write");
    assert_ne!(
        k0,
        mutate(|s| s.depth_func = narrow(D3DCMP_EQUAL)),
        "depth_func"
    );
    assert_ne!(k0, mutate(|s| s.stencil_enable = 0), "stencil_enable");
    assert_ne!(
        k0,
        mutate(|s| s.front.func = narrow(D3DCMP_EQUAL)),
        "front.func"
    );
    assert_ne!(
        k0,
        mutate(|s| s.front.fail_op = narrow(D3DSTENCILOP_REPLACE)),
        "front.fail_op"
    );
    assert_ne!(
        k0,
        mutate(|s| s.front.depth_fail_op = narrow(D3DSTENCILOP_REPLACE)),
        "front.depth_fail_op"
    );
    assert_ne!(
        k0,
        mutate(|s| s.front.pass_op = narrow(D3DSTENCILOP_REPLACE)),
        "front.pass_op"
    );
    assert_ne!(
        k0,
        mutate(|s| s.back.func = narrow(D3DCMP_EQUAL)),
        "back.func"
    );
    assert_ne!(
        k0,
        mutate(|s| s.back.fail_op = narrow(D3DSTENCILOP_REPLACE)),
        "back.fail_op"
    );
    assert_ne!(
        k0,
        mutate(|s| s.back.depth_fail_op = narrow(D3DSTENCILOP_REPLACE)),
        "back.depth_fail_op"
    );
    assert_ne!(
        k0,
        mutate(|s| s.back.pass_op = narrow(D3DSTENCILOP_REPLACE)),
        "back.pass_op"
    );
    assert_ne!(k0, mutate(|s| s.read_mask = 0x0F), "read_mask");
    assert_ne!(k0, mutate(|s| s.write_mask = 0x0F), "write_mask");
}

#[test]
fn disabled_tests_fold_their_own_fields() {
    // One Metal object serves every disabled-test state, so the keys must
    // collapse too or teardown releases that object once per key.
    let mut a = base();
    a.stencil_enable = 0;
    a.front.func = narrow(D3DCMP_EQUAL);
    a.write_mask = 0x0F;
    let mut b = base();
    b.stencil_enable = 0;
    b.front.pass_op = narrow(D3DSTENCILOP_REPLACE);
    assert_eq!(key_from_snapshot(&a), key_from_snapshot(&b));

    let mut c = base();
    c.depth_enable = 0;
    c.depth_func = narrow(D3DCMP_EQUAL);
    let mut d = base();
    d.depth_enable = 0;
    d.depth_write = 0;
    assert_eq!(key_from_snapshot(&c), key_from_snapshot(&d));
}

#[test]
fn stencil_state_stays_out_of_the_key_while_the_test_is_off() {
    // The default render state has the stencil test off, so a game that
    // never touches D3DRS_STENCIL* must keep the pre-stencil key.
    let defaults = snapshot_from_state(&render_state_defaults());
    let key = key_from_snapshot(&defaults);
    assert_eq!(key.raw() >> 5, 0, "no stencil bits set");
}

#[test]
fn back_face_follows_the_front_until_two_sided_mode_is_on() {
    let mut rs = render_state_defaults();
    rs[D3DRS_STENCILFUNC as usize] = D3DCMP_LESSEQUAL;
    rs[D3DRS_CCW_STENCILFUNC as usize] = D3DCMP_EQUAL;

    let one_sided = snapshot_from_state(&rs);
    assert_eq!(one_sided.back, one_sided.front, "CCW states are inert");

    rs[D3DRS_TWOSIDEDSTENCILMODE as usize] = 1;
    let two_sided = snapshot_from_state(&rs);
    assert_eq!(two_sided.front.func, narrow(D3DCMP_LESSEQUAL));
    assert_eq!(two_sided.back.func, narrow(D3DCMP_EQUAL));
}

#[test]
fn out_of_range_enums_do_not_collide_with_valid_ones() {
    // SetRenderState accepts any DWORD. A value outside the D3DCMP_* range
    // translates to the Always fallback, so it must not share a key with
    // whichever valid value happens to match it in the low bits: the
    // second state to arrive would be handed the first one's Metal object.
    let mut rs = render_state_defaults();
    rs[D3DRS_STENCILENABLE as usize] = 1;

    rs[D3DRS_STENCILFUNC as usize] = D3DCMP_LESSEQUAL;
    let valid = key_from_snapshot(&snapshot_from_state(&rs));

    rs[D3DRS_STENCILFUNC as usize] = D3DCMP_LESSEQUAL + 16;
    let out_of_range = key_from_snapshot(&snapshot_from_state(&rs));

    assert_ne!(valid, out_of_range);

    // And the fallback keys the same as an explicit ALWAYS, since that is
    // the state the Metal object is actually built with.
    rs[D3DRS_STENCILFUNC as usize] = D3DCMP_ALWAYS;
    assert_eq!(out_of_range, key_from_snapshot(&snapshot_from_state(&rs)));
}

#[test]
fn masks_narrow_to_the_width_the_attachment_observes() {
    // Stencil8: bits above the low 8 cannot change the result, and the
    // key and the Metal object must be built from the same value.
    let mut rs = render_state_defaults();
    rs[D3DRS_STENCILENABLE as usize] = 1;
    rs[D3DRS_STENCILMASK as usize] = 0xFFFF_FF12;
    rs[D3DRS_STENCILWRITEMASK as usize] = 0xFFFF_FF34;
    let wide = snapshot_from_state(&rs);
    assert_eq!(
        wide.read_mask, 0xFFFF_FF12,
        "kept full width in the snapshot"
    );

    rs[D3DRS_STENCILMASK as usize] = 0x12;
    rs[D3DRS_STENCILWRITEMASK as usize] = 0x34;
    let narrow = snapshot_from_state(&rs);
    assert_eq!(key_from_snapshot(&wide), key_from_snapshot(&narrow));

    // SAFETY: tests; opaque value never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let params = params_from_snapshot(&wide, key_from_snapshot(&wide), dev);
    assert_eq!(params.stencil_read_mask, 0x12);
    assert_eq!(params.stencil_write_mask, 0x34);
}

#[test]
fn clear_quad_states_are_distinct_and_write_what_they_claim() {
    // SAFETY: tests; opaque value never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let states = [
        DepthStencilSnapshot::inert(),
        DepthStencilSnapshot::depth_overwrite(),
        DepthStencilSnapshot::stencil_overwrite(),
        DepthStencilSnapshot::depth_stencil_overwrite(),
    ];
    let keys: Vec<_> = states.iter().map(key_from_snapshot).collect();
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b, "each clear-quad state needs its own Metal object");
        }
    }

    let both = DepthStencilSnapshot::depth_stencil_overwrite();
    let params = params_from_snapshot(&both, key_from_snapshot(&both), dev);
    assert_eq!(params.depth_write_enable, 1);
    assert_eq!(params.depth_compare_func, d3d_to_metal_cmp(D3DCMP_ALWAYS));
    assert_eq!(params.stencil_test_enable, 1);
    assert_eq!(
        params.front.pass_op,
        d3d_to_metal_stencil_op(D3DSTENCILOP_REPLACE)
    );
    assert_eq!(params.front, params.back);
    assert_eq!(params.stencil_write_mask, STENCIL_MASK_BITS);
}

#[test]
fn params_translate_the_default_state() {
    // SAFETY: tests; opaque value never dereferenced.
    let dev = unsafe { MetalHandle::new(0xDEAD) };
    let s = base();
    let params = params_from_snapshot(&s, key_from_snapshot(&s), dev);
    assert_eq!(params.stencil_test_enable, 1);
    assert_eq!(params.front.compare_func, d3d_to_metal_cmp(D3DCMP_ALWAYS));
    assert_eq!(
        params.front.pass_op,
        d3d_to_metal_stencil_op(D3DSTENCILOP_KEEP)
    );
    assert_eq!(params.front, params.back, "one-sided default");
    assert_eq!(params.id, key_from_snapshot(&s).raw());
}
