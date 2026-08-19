//! Single source of truth for D3D9 → Metal depth/stencil-state translation.
//!
//! Mirrors `sampler_state` but for the depth/stencil test: one
//! `DepthStencilSnapshot` drives both the cache `DepthStencilKey` and the
//! wire-format `CreateDepthStencilStateParams`, so a render state the
//! classifier calls Consumed cannot reach one consumer and not the other.
//! Per-field unit tests assert that mutating any snapshot field produces a
//! different key.

use std::fmt;

use mtld3d_shared::{
    CreateDepthStencilStateParams, MetalHandle, StencilFaceParams, mtl_handle::MTLDeviceKind,
};
use mtld3d_types::{
    D3DCMP_ALWAYS, D3DRS_CCW_STENCILFAIL, D3DRS_CCW_STENCILFUNC, D3DRS_CCW_STENCILPASS,
    D3DRS_CCW_STENCILZFAIL, D3DRS_STENCILENABLE, D3DRS_STENCILFAIL, D3DRS_STENCILFUNC,
    D3DRS_STENCILMASK, D3DRS_STENCILPASS, D3DRS_STENCILWRITEMASK, D3DRS_STENCILZFAIL,
    D3DRS_TWOSIDEDSTENCILMODE, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE, D3DSTENCILOP_KEEP,
    RENDER_STATE_COUNT,
};

use crate::convert::{d3d_to_metal_cmp, d3d_to_metal_stencil_op};

/// Packed-bits key for the depth-stencil state cache.
///
/// Lossless compression of the depth test plus both stencil faces into a
/// single u64; see `key_from_snapshot` for the layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DepthStencilKey(u64);

impl fmt::LowerHex for DepthStencilKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl DepthStencilKey {
    /// Inner u64.
    ///
    /// Used as the descriptor-side label payload at `CreateDepthStencilState`
    /// thunk time.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// One face of the D3D9 stencil test, as raw `D3DCMP_*` / `D3DSTENCILOP_*` values.
///
/// D3D9 stores these as DWORDs but every value is a small enum, so the
/// snapshot narrows them to `u8` and stays cheap to carry per draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFaceState {
    pub func: u8,
    pub fail_op: u8,
    pub depth_fail_op: u8,
    pub pass_op: u8,
}

/// Stencil mask width the attachment can actually observe.
///
/// The only stencil format in play is `Depth32Float_Stencil8`, so mask bits
/// above the low 8 cannot change the test result.
pub const STENCIL_MASK_BITS: u32 = 0xFF;

/// `D3DCMP_ALWAYS` / `D3DSTENCILOP_KEEP` at the snapshot's narrow width.
///
/// Literals rather than casts so the surrounding `const` items stay free of
/// truncating `as`; the asserts pin them to the ABI constants.
const CMP_ALWAYS: u8 = 8;
const OP_KEEP: u8 = 1;
const _: () = assert!(CMP_ALWAYS as u32 == D3DCMP_ALWAYS);
const _: () = assert!(OP_KEEP as u32 == D3DSTENCILOP_KEEP);

/// The D3D9 default face: always compare, never modify.
///
/// What the fixed-state constructors below give both faces.
const KEEP_FACE: StencilFaceState = StencilFaceState {
    func: CMP_ALWAYS,
    fail_op: OP_KEEP,
    depth_fail_op: OP_KEEP,
    pass_op: OP_KEEP,
};

/// Input view of the render states that select an `MTLDepthStencilState`.
///
/// Raw D3D values preserve 1:1 fidelity with the game input. Both
/// `key_from_snapshot` and `params_from_snapshot` translate them through the
/// same `convert` helpers, so a state can never be keyed as one thing and
/// built as another. `D3DRS_STENCILREF` is absent by
/// design: Metal carries the reference value on the encoder
/// (`setStencilReferenceValue`), not on the state object, so folding it in
/// here would mint a distinct Metal object per reference value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilSnapshot {
    pub depth_enable: u8,
    pub depth_write: u8,
    pub depth_func: u8,
    pub stencil_enable: u8,
    pub front: StencilFaceState,
    /// Back face, already resolved against `D3DRS_TWOSIDEDSTENCILMODE`.
    ///
    /// D3D9 applies the `D3DRS_CCW_STENCIL*` states only while two-sided mode
    /// is on; with it off both faces take the front-face states.
    pub back: StencilFaceState,
    /// `D3DRS_STENCILMASK`, as the game set it.
    ///
    /// Kept full-width because the D3D9 masks are unbounded DWORDs, unlike
    /// the enum-valued states. `key_from_snapshot` and `params_from_snapshot`
    /// both apply `STENCIL_MASK_BITS`, so the key and the Metal object are
    /// still built from one value.
    pub read_mask: u32,
    /// `D3DRS_STENCILWRITEMASK`, as the game set it.
    pub write_mask: u32,
}

impl DepthStencilSnapshot {
    /// Depth and stencil both inert.
    ///
    /// The state for a helper draw that must leave the attachment untouched.
    #[must_use]
    pub const fn inert() -> Self {
        Self {
            depth_enable: 0,
            depth_write: 0,
            depth_func: CMP_ALWAYS,
            stencil_enable: 0,
            front: KEEP_FACE,
            back: KEEP_FACE,
            read_mask: 0,
            write_mask: 0,
        }
    }

    /// Depth written unconditionally, stencil inert: the depth clear quad.
    #[must_use]
    pub const fn depth_overwrite() -> Self {
        Self {
            depth_enable: 1,
            depth_write: 1,
            ..Self::inert()
        }
    }

    /// Drop the stencil test when the bound attachment carries no stencil plane.
    ///
    /// An app can leave `D3DRS_STENCILENABLE` set from an earlier pass and then
    /// bind a depth-only surface (D16 / D24X8 / D32). A stencil-enabled
    /// `MTLDepthStencilState` against a render pass with no stencil attachment
    /// is a Metal validation failure.
    #[must_use]
    pub const fn gated_on_stencil_attachment(mut self, attachment_has_stencil: bool) -> Self {
        if !attachment_has_stencil {
            self.stencil_enable = 0;
        }
        self
    }
}

/// Build a `DepthStencilSnapshot` from the device's render-state array.
///
/// # Panics
///
/// Panics if an enum-valued stencil or depth render state holds a value wider
/// than a byte. The boolean states are read as `!= 0` and never panic.
#[must_use]
pub fn snapshot_from_state(rs: &[u32; RENDER_STATE_COUNT]) -> DepthStencilSnapshot {
    let to_u8 = |v: u32| u8::try_from(v).expect("D3D9 enum render-state value fits u8");
    let front = StencilFaceState {
        func: to_u8(rs[D3DRS_STENCILFUNC as usize]),
        fail_op: to_u8(rs[D3DRS_STENCILFAIL as usize]),
        depth_fail_op: to_u8(rs[D3DRS_STENCILZFAIL as usize]),
        pass_op: to_u8(rs[D3DRS_STENCILPASS as usize]),
    };
    let back = if rs[D3DRS_TWOSIDEDSTENCILMODE as usize] == 0 {
        front
    } else {
        StencilFaceState {
            func: to_u8(rs[D3DRS_CCW_STENCILFUNC as usize]),
            fail_op: to_u8(rs[D3DRS_CCW_STENCILFAIL as usize]),
            depth_fail_op: to_u8(rs[D3DRS_CCW_STENCILZFAIL as usize]),
            pass_op: to_u8(rs[D3DRS_CCW_STENCILPASS as usize]),
        }
    };
    DepthStencilSnapshot {
        depth_enable: u8::from(rs[D3DRS_ZENABLE as usize] != 0),
        depth_write: u8::from(rs[D3DRS_ZWRITEENABLE as usize] != 0),
        depth_func: to_u8(rs[D3DRS_ZFUNC as usize]),
        stencil_enable: u8::from(rs[D3DRS_STENCILENABLE as usize] != 0),
        front,
        back,
        read_mask: rs[D3DRS_STENCILMASK as usize],
        write_mask: rs[D3DRS_STENCILWRITEMASK as usize],
    }
}

/// Pack a snapshot into the depth/stencil cache key.
///
/// Layout (u64 low-to-high):
/// - 0      `depth_enable`
/// - 1      `depth_write`
/// - 2..4   `depth_func`
/// - 5      `stencil_enable`
/// - 6..17  front face (`func`, `fail_op`, `depth_fail_op`, `pass_op`, 3 bits each)
/// - 18..29 back face, same order
/// - 30..37 `read_mask`
/// - 38..45 `write_mask`
///
/// The enum fields are packed after translation, not as the raw D3D values.
/// `SetRenderState` takes any DWORD, and two values that differ only above the
/// field width would otherwise share a key while translating to different
/// Metal enums, so the second state to arrive would be served the first one's
/// object.
///
/// Each disabled test folds its own fields to zero. Without that, every
/// (write, func) combination behind `depth_enable == 0` would be a distinct
/// key aliasing the one Metal object the unix side builds for disabled depth,
/// and teardown would release it once per key.
#[must_use]
pub fn key_from_snapshot(s: &DepthStencilSnapshot) -> DepthStencilKey {
    let mut bits = 0u64;
    if s.depth_enable != 0 {
        bits |= 1
            | u64::from(s.depth_write != 0) << 1
            | ((d3d_to_metal_cmp(u32::from(s.depth_func)) as u64) << 2);
    }
    if s.stencil_enable != 0 {
        bits |= 1 << 5
            | (pack_face(s.front) << 6)
            | (pack_face(s.back) << 18)
            | (u64::from(s.read_mask & STENCIL_MASK_BITS) << 30)
            | (u64::from(s.write_mask & STENCIL_MASK_BITS) << 38);
    }
    DepthStencilKey(bits)
}

fn pack_face(f: StencilFaceState) -> u64 {
    (d3d_to_metal_cmp(u32::from(f.func)) as u64)
        | ((d3d_to_metal_stencil_op(u32::from(f.fail_op)) as u64) << 3)
        | ((d3d_to_metal_stencil_op(u32::from(f.depth_fail_op)) as u64) << 6)
        | ((d3d_to_metal_stencil_op(u32::from(f.pass_op)) as u64) << 9)
}

/// Translate a snapshot into the wire-format `CreateDepthStencilStateParams`.
#[must_use]
pub fn params_from_snapshot(
    s: &DepthStencilSnapshot,
    key: DepthStencilKey,
    device_handle: MetalHandle<MTLDeviceKind>,
) -> CreateDepthStencilStateParams {
    CreateDepthStencilStateParams {
        device_handle,
        depth_test_enable: u32::from(s.depth_enable),
        depth_write_enable: u32::from(s.depth_write),
        depth_compare_func: d3d_to_metal_cmp(u32::from(s.depth_func)),
        stencil_test_enable: u32::from(s.stencil_enable),
        front: face_params(s.front),
        back: face_params(s.back),
        stencil_read_mask: s.read_mask & STENCIL_MASK_BITS,
        stencil_write_mask: s.write_mask & STENCIL_MASK_BITS,
        id: key.raw(),
        state_handle: MetalHandle::NULL,
    }
}

fn face_params(f: StencilFaceState) -> StencilFaceParams {
    StencilFaceParams {
        compare_func: d3d_to_metal_cmp(u32::from(f.func)),
        stencil_fail_op: d3d_to_metal_stencil_op(u32::from(f.fail_op)),
        depth_fail_op: d3d_to_metal_stencil_op(u32::from(f.depth_fail_op)),
        pass_op: d3d_to_metal_stencil_op(u32::from(f.pass_op)),
    }
}

#[cfg(test)]
mod tests {
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
}
