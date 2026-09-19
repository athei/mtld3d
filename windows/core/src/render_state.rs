//! Value spaces of the enum-valued D3D9 render states.
//!
//! `SetRenderState` takes a DWORD and D3D9 stores it whatever it holds, so a
//! render state read at draw time is game input, not a value the API already
//! constrained. The snapshot and cache-key structs carry the enum states as
//! bytes (CONVENTIONS.md §Narrowest type for the range), and this module is
//! the one place that turns a DWORD into that byte: a value inside the state's
//! enum space passes through, anything else reads as the state's D3D9 default
//! and is warned once. The state array keeps the raw DWORD, so `GetRenderState`
//! and a state block still hand back exactly what the game wrote.

use mtld3d_types::{
    D3DBLEND_INVSRCCOLOR2, D3DBLEND_ONE, D3DBLEND_ZERO, D3DBLENDOP_ADD, D3DBLENDOP_MAX,
    D3DCMP_ALWAYS, D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCULL_CCW, D3DCULL_NONE, D3DFILL_POINT,
    D3DFILL_SOLID, D3DRS_ALPHAFUNC, D3DRS_BLENDOP, D3DRS_BLENDOPALPHA, D3DRS_CCW_STENCILFAIL,
    D3DRS_CCW_STENCILFUNC, D3DRS_CCW_STENCILPASS, D3DRS_CCW_STENCILZFAIL, D3DRS_COLORWRITEENABLE,
    D3DRS_COLORWRITEENABLE1, D3DRS_COLORWRITEENABLE2, D3DRS_COLORWRITEENABLE3, D3DRS_CULLMODE,
    D3DRS_DESTBLEND, D3DRS_DESTBLENDALPHA, D3DRS_FILLMODE, D3DRS_SRCBLEND, D3DRS_SRCBLENDALPHA,
    D3DRS_STENCILFAIL, D3DRS_STENCILFUNC, D3DRS_STENCILPASS, D3DRS_STENCILZFAIL, D3DRS_ZFUNC,
    D3DSTENCILOP_DECR, D3DSTENCILOP_KEEP, RENDER_STATE_COUNT,
};

/// The D3D9 enum bounds at the byte width the snapshots carry.
///
/// Narrow copies of the ABI constants, each pinned to its `mtld3d-types`
/// definition by the assert below, so the table can name a bound in `u8`
/// without a truncating cast.
const CMP_FIRST: u8 = 1;
const CMP_LAST: u8 = 8;
const CMP_LESSEQUAL: u8 = 4;
const CMP_ALWAYS: u8 = 8;
const BLEND_FIRST: u8 = 1;
const BLEND_LAST: u8 = 17;
const BLEND_ZERO: u8 = 1;
const BLEND_ONE: u8 = 2;
const BLENDOP_FIRST: u8 = 1;
const BLENDOP_LAST: u8 = 5;
const BLENDOP_ADD: u8 = 1;
const CULL_FIRST: u8 = 1;
const CULL_LAST: u8 = 3;
const CULL_CCW: u8 = 3;
const FILL_FIRST: u8 = 1;
const FILL_LAST: u8 = 3;
const STENCILOP_FIRST: u8 = 1;
const STENCILOP_LAST: u8 = 8;
const STENCILOP_KEEP: u8 = 1;

/// The four `D3DCOLORWRITEENABLE_*` channel bits.
///
/// `D3DRS_COLORWRITEENABLE*` is a mask, not an enum: bits above the four
/// channels select nothing, so they are dropped rather than rejected.
const COLOR_WRITE_BITS: u8 = 0x0F;

const _: () = assert!(CMP_FIRST as u32 == D3DCMP_NEVER);
const _: () = assert!(CMP_LAST as u32 == D3DCMP_ALWAYS);
const _: () = assert!(CMP_LESSEQUAL as u32 == D3DCMP_LESSEQUAL);
const _: () = assert!(CMP_ALWAYS as u32 == D3DCMP_ALWAYS);
const _: () = assert!(BLEND_FIRST as u32 == D3DBLEND_ZERO);
const _: () = assert!(BLEND_LAST as u32 == D3DBLEND_INVSRCCOLOR2);
const _: () = assert!(BLEND_ZERO as u32 == D3DBLEND_ZERO);
const _: () = assert!(BLEND_ONE as u32 == D3DBLEND_ONE);
const _: () = assert!(BLENDOP_FIRST as u32 == D3DBLENDOP_ADD);
const _: () = assert!(BLENDOP_LAST as u32 == D3DBLENDOP_MAX);
const _: () = assert!(BLENDOP_ADD as u32 == D3DBLENDOP_ADD);
const _: () = assert!(CULL_FIRST as u32 == D3DCULL_NONE);
const _: () = assert!(CULL_LAST as u32 == D3DCULL_CCW);
const _: () = assert!(CULL_CCW as u32 == D3DCULL_CCW);
const _: () = assert!(FILL_FIRST as u32 == D3DFILL_POINT);
const _: () = assert!(FILL_LAST as u32 == D3DFILL_SOLID);
const _: () = assert!(STENCILOP_FIRST as u32 == D3DSTENCILOP_KEEP);
const _: () = assert!(STENCILOP_LAST as u32 == D3DSTENCILOP_DECR);
const _: () = assert!(STENCILOP_KEEP as u32 == D3DSTENCILOP_KEEP);

/// What a render state accepts, at the width a snapshot carries.
enum Space {
    /// A contiguous enum space; a value outside it reads as `default`.
    Enum { first: u8, last: u8, default: u8 },
    /// A bit mask; bits outside `bits` name nothing and are dropped.
    Mask { bits: u8 },
}

/// The space `state` accepts, or `None` when it carries no enum.
///
/// Only the states whose consumers narrow them are listed. A state whose
/// value reaches a `match` with a logged fallback arm (the fog modes, the
/// material sources, the vertex-blend count) is left to that arm, which can
/// name the substitution it makes in terms of the feature it drives.
const fn space(state: u32) -> Option<Space> {
    match state {
        D3DRS_ZFUNC => Some(cmp(CMP_LESSEQUAL)),
        D3DRS_ALPHAFUNC | D3DRS_STENCILFUNC | D3DRS_CCW_STENCILFUNC => Some(cmp(CMP_ALWAYS)),
        D3DRS_SRCBLEND | D3DRS_SRCBLENDALPHA => Some(blend(BLEND_ONE)),
        D3DRS_DESTBLEND | D3DRS_DESTBLENDALPHA => Some(blend(BLEND_ZERO)),
        D3DRS_BLENDOP | D3DRS_BLENDOPALPHA => Some(Space::Enum {
            first: BLENDOP_FIRST,
            last: BLENDOP_LAST,
            default: BLENDOP_ADD,
        }),
        D3DRS_FILLMODE => Some(Space::Enum {
            first: FILL_FIRST,
            last: FILL_LAST,
            default: FILL_LAST,
        }),
        D3DRS_CULLMODE => Some(Space::Enum {
            first: CULL_FIRST,
            last: CULL_LAST,
            default: CULL_CCW,
        }),
        D3DRS_STENCILFAIL
        | D3DRS_STENCILZFAIL
        | D3DRS_STENCILPASS
        | D3DRS_CCW_STENCILFAIL
        | D3DRS_CCW_STENCILZFAIL
        | D3DRS_CCW_STENCILPASS => Some(Space::Enum {
            first: STENCILOP_FIRST,
            last: STENCILOP_LAST,
            default: STENCILOP_KEEP,
        }),
        D3DRS_COLORWRITEENABLE
        | D3DRS_COLORWRITEENABLE1
        | D3DRS_COLORWRITEENABLE2
        | D3DRS_COLORWRITEENABLE3 => Some(Space::Mask {
            bits: COLOR_WRITE_BITS,
        }),
        _ => None,
    }
}

const fn cmp(default: u8) -> Space {
    Space::Enum {
        first: CMP_FIRST,
        last: CMP_LAST,
        default,
    }
}

const fn blend(default: u8) -> Space {
    Space::Enum {
        first: BLEND_FIRST,
        last: BLEND_LAST,
        default,
    }
}

/// Render state `state`, narrowed to the byte a snapshot carries.
///
/// A value outside the state's enum space reads as that state's D3D9 default,
/// which is what a driver handed a value it does not recognise settles on.
/// Truncating the DWORD instead would run the draw under a different enum
/// than either side asked for, and rejecting the write in `SetRenderState`
/// would make `GetRenderState` disagree with the DWORD the game passed.
#[must_use]
pub fn enum_value(rs: &[u32; RENDER_STATE_COUNT], state: u32) -> u8 {
    let value = rs[state as usize];
    // Exact for every value a space accepts: each enum space and the
    // colour-write mask fit in a byte.
    let byte = value.to_le_bytes()[0];
    match space(state) {
        Some(Space::Enum {
            first,
            last,
            default,
        }) => {
            if u32::from(byte) == value && first <= byte && byte <= last {
                return byte;
            }
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(state),
                "D3DRS_{state} = {value:#x} outside its {first}..={last} value space → reading the D3D9 default {default:#x}"
            );
            default
        }
        Some(Space::Mask { bits }) => {
            if value & !u32::from(bits) != 0 {
                mtld3d_shared::log_once_warn_by!(
                    target: crate::LOG_TARGET,
                    key: u64::from(state),
                    "D3DRS_{state} = {value:#x} sets bits outside the {bits:#x} mask → dropping them"
                );
            }
            byte & bits
        }
        None => {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: u64::from(state),
                "D3DRS_{state} narrowed as an enum but carries no enum space → low byte {byte:#x}"
            );
            byte
        }
    }
}

#[cfg(test)]
mod tests;
