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
    D3DSTENCILOP_REPLACE, RENDER_STATE_COUNT,
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
const OP_REPLACE: u8 = 3;
const _: () = assert!(CMP_ALWAYS as u32 == D3DCMP_ALWAYS);
const _: () = assert!(OP_KEEP as u32 == D3DSTENCILOP_KEEP);
const _: () = assert!(OP_REPLACE as u32 == D3DSTENCILOP_REPLACE);

/// The D3D9 default face: always compare, never modify.
///
/// What the fixed-state constructors below give both faces.
const KEEP_FACE: StencilFaceState = StencilFaceState {
    func: CMP_ALWAYS,
    fail_op: OP_KEEP,
    depth_fail_op: OP_KEEP,
    pass_op: OP_KEEP,
};

/// Writes the reference to every fragment, whatever the depth test did.
const REPLACE_FACE: StencilFaceState = StencilFaceState {
    func: CMP_ALWAYS,
    fail_op: OP_REPLACE,
    depth_fail_op: OP_REPLACE,
    pass_op: OP_REPLACE,
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

    /// Stencil overwritten with the reference, depth untouched.
    ///
    /// The state for the stencil clear quad.
    ///
    /// MSL cannot export a stencil value, so the clear writes through the
    /// stencil operation instead: compare `Always` and `Replace` on every
    /// outcome, with the clear value supplied as the encoder's stencil
    /// reference. `read_mask` is irrelevant under an always-true compare.
    #[must_use]
    pub const fn stencil_overwrite() -> Self {
        Self {
            stencil_enable: 1,
            front: REPLACE_FACE,
            back: REPLACE_FACE,
            read_mask: 0,
            write_mask: STENCIL_MASK_BITS,
            ..Self::inert()
        }
    }

    /// Depth and stencil both overwritten: the combined `Clear(ZBUFFER | STENCIL)` quad.
    ///
    /// `depth_overwrite` and `stencil_overwrite` in one state, so a mid-frame
    /// clear of both planes is a single draw.
    #[must_use]
    pub const fn depth_stencil_overwrite() -> Self {
        Self {
            depth_enable: 1,
            depth_write: 1,
            ..Self::stencil_overwrite()
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
mod tests;
