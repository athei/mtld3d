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
//!
//! It also holds [`rs_classify`], the table the device's silent-write audit
//! reads to decide whether a non-default render-state write reaches a
//! consumer, is a no-op by design, or is a feature gap worth a warning.

use mtld3d_types::{
    D3DBLEND_INVSRCCOLOR2, D3DBLEND_ONE, D3DBLEND_ZERO, D3DBLENDOP_ADD, D3DBLENDOP_MAX,
    D3DCMP_ALWAYS, D3DCMP_LESSEQUAL, D3DCMP_NEVER, D3DCULL_CCW, D3DCULL_NONE, D3DFILL_POINT,
    D3DFILL_SOLID, D3DFMT_ATOC, D3DFMT_NVDB, D3DRS_ADAPTIVETESS_W, D3DRS_ADAPTIVETESS_X,
    D3DRS_ADAPTIVETESS_Y, D3DRS_ADAPTIVETESS_Z, D3DRS_ALPHABLENDENABLE, D3DRS_ALPHAFUNC,
    D3DRS_ALPHAREF, D3DRS_ALPHATESTENABLE, D3DRS_AMBIENT, D3DRS_AMBIENTMATERIALSOURCE,
    D3DRS_ANTIALIASEDLINEENABLE, D3DRS_BLENDFACTOR, D3DRS_BLENDOP, D3DRS_BLENDOPALPHA,
    D3DRS_CCW_STENCILFAIL, D3DRS_CCW_STENCILFUNC, D3DRS_CCW_STENCILPASS, D3DRS_CCW_STENCILZFAIL,
    D3DRS_CLIPPING, D3DRS_CLIPPLANEENABLE, D3DRS_COLORVERTEX, D3DRS_COLORWRITEENABLE,
    D3DRS_COLORWRITEENABLE1, D3DRS_COLORWRITEENABLE2, D3DRS_COLORWRITEENABLE3, D3DRS_CULLMODE,
    D3DRS_DEBUGMONITORTOKEN, D3DRS_DEPTHBIAS, D3DRS_DESTBLEND, D3DRS_DESTBLENDALPHA,
    D3DRS_DIFFUSEMATERIALSOURCE, D3DRS_DITHERENABLE, D3DRS_EMISSIVEMATERIALSOURCE,
    D3DRS_ENABLEADAPTIVETESSELLATION, D3DRS_FILLMODE, D3DRS_FOGCOLOR, D3DRS_FOGDENSITY,
    D3DRS_FOGENABLE, D3DRS_FOGEND, D3DRS_FOGSTART, D3DRS_FOGTABLEMODE, D3DRS_FOGVERTEXMODE,
    D3DRS_INDEXEDVERTEXBLENDENABLE, D3DRS_LIGHTING, D3DRS_LOCALVIEWER, D3DRS_MAXTESSELLATIONLEVEL,
    D3DRS_MINTESSELLATIONLEVEL, D3DRS_MULTISAMPLEANTIALIAS, D3DRS_MULTISAMPLEMASK,
    D3DRS_NORMALDEGREE, D3DRS_NORMALIZENORMALS, D3DRS_PATCHEDGESTYLE, D3DRS_POINTSCALE_A,
    D3DRS_POINTSCALE_B, D3DRS_POINTSCALE_C, D3DRS_POINTSCALEENABLE, D3DRS_POINTSIZE,
    D3DRS_POINTSIZE_MAX, D3DRS_POINTSIZE_MIN, D3DRS_POINTSPRITEENABLE, D3DRS_POSITIONDEGREE,
    D3DRS_RANGEFOGENABLE, D3DRS_SCISSORTESTENABLE, D3DRS_SEPARATEALPHABLENDENABLE, D3DRS_SHADEMODE,
    D3DRS_SLOPESCALEDEPTHBIAS, D3DRS_SPECULARENABLE, D3DRS_SPECULARMATERIALSOURCE, D3DRS_SRCBLEND,
    D3DRS_SRCBLENDALPHA, D3DRS_SRGBWRITEENABLE, D3DRS_STENCILENABLE, D3DRS_STENCILFAIL,
    D3DRS_STENCILFUNC, D3DRS_STENCILMASK, D3DRS_STENCILPASS, D3DRS_STENCILREF,
    D3DRS_STENCILWRITEMASK, D3DRS_STENCILZFAIL, D3DRS_TEXTUREFACTOR, D3DRS_TWEENFACTOR,
    D3DRS_TWOSIDEDSTENCILMODE, D3DRS_VERTEXBLEND, D3DRS_ZENABLE, D3DRS_ZFUNC, D3DRS_ZWRITEENABLE,
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

/// How the silent-write audit treats a non-default `SetRenderState` write.
///
/// The device keeps one warn latch per slot and asks [`rs_classify`] the
/// first time a slot receives a value other than its D3D9 default.
pub enum RsClass {
    /// A consumer reads the slot, so the write is honoured and nothing is logged.
    Consumed,
    /// A no-op by design, logged once at info with the reason.
    ///
    /// Metal has no analog, or the feature is obsolete on every modern
    /// driver, so the no-op is the complete correct behaviour and not a port
    /// candidate: the `log_once_info!` side of the info-versus-warn line.
    Obsolete(&'static str),
    /// Nothing reads the slot, so the write is lost and warned once.
    NotImplemented,
}

/// Classify a non-default write of `value` to render state `index`.
///
/// A slot is `Consumed` only while a snapshot, key or uniform builder reads
/// it; the comment on each group names that reader. A new consumer moves its
/// slot here in the same change, so every warning the audit prints stays a
/// real gap.
#[must_use]
pub const fn rs_classify(index: u32, value: u32) -> RsClass {
    match index {
        // Alpha to coverage: `multisample::alpha_to_coverage_requested` reads
        // the ATOC token. Zero is the D3D9 default and never reaches here.
        D3DRS_ADAPTIVETESS_Y if matches!(value, 0 | D3DFMT_ATOC) => RsClass::Consumed,
        // Depth, blend and colour-write state: the RS snapshot built on the
        // API thread, keyed through `pipeline_state::key_from_snapshot` and
        // `depth_stencil_state::snapshot_from_state`, whose per-field tests
        // assert that mutating any of these produces a different key.
        D3DRS_ZENABLE
        | D3DRS_ZWRITEENABLE
        | D3DRS_ZFUNC
        | D3DRS_ALPHABLENDENABLE
        | D3DRS_SRCBLEND
        | D3DRS_DESTBLEND
        | D3DRS_BLENDOP
        | D3DRS_BLENDOPALPHA
        | D3DRS_SEPARATEALPHABLENDENABLE
        | D3DRS_SRCBLENDALPHA
        | D3DRS_DESTBLENDALPHA
        // SRGBWRITEENABLE binds the colour attachment's sRGB twin view for
        // the pass, so Metal encodes after the blender. A target with no
        // sRGB Metal view falls back to the pixel-shader OETF variant
        // (`VariantFlags::SRGB_WRITE`) with `Clear` converting its colour
        // through the same curve.
        | D3DRS_SRGBWRITEENABLE
        | D3DRS_COLORWRITEENABLE
        | D3DRS_COLORWRITEENABLE1
        | D3DRS_COLORWRITEENABLE2
        | D3DRS_COLORWRITEENABLE3
        | D3DRS_CULLMODE
        | D3DRS_FILLMODE
        | D3DRS_SCISSORTESTENABLE
        // SHADEMODE keys `VariantFlags::FLAT_SHADE` in `FfState::variant_key`:
        // under `D3DSHADE_FLAT` both pixel-shader sources declare the two
        // colour varyings `[[flat]]`, so they come from the provoking vertex.
        | D3DRS_SHADEMODE
        // Alpha test: `FfState::variant_key` (alpha function) and the PS
        // slot-14 alpha-ref bytes.
        | D3DRS_ALPHATESTENABLE
        | D3DRS_ALPHAFUNC
        | D3DRS_ALPHAREF
        // Fixed-function lighting, fog and texture factor: `FfState`'s VS
        // and PS keys and constant builders.
        | D3DRS_LIGHTING
        | D3DRS_AMBIENT
        | D3DRS_TEXTUREFACTOR
        | D3DRS_FOGENABLE
        | D3DRS_FOGVERTEXMODE
        | D3DRS_FOGTABLEMODE
        | D3DRS_RANGEFOGENABLE
        | D3DRS_FOGCOLOR
        | D3DRS_FOGSTART
        | D3DRS_FOGEND
        | D3DRS_FOGDENSITY
        // COLORVERTEX and the four material sources feed the fixed-function
        // VS emitter's `resolve_mat` (`dxso::ff`) through `FfVsFlags` and
        // the `FfVsKey` material-source fields.
        | D3DRS_COLORVERTEX
        | D3DRS_DIFFUSEMATERIALSOURCE
        | D3DRS_AMBIENTMATERIALSOURCE
        | D3DRS_SPECULARMATERIALSOURCE
        | D3DRS_EMISSIVEMATERIALSOURCE
        // NORMALIZENORMALS is the `FfVsFlags::NORMALIZE_NORMALS` key bit on a
        // lit draw with a normal: the VS renormalizes the eye-space normal.
        | D3DRS_NORMALIZENORMALS
        // SPECULARENABLE gates the specular colour output of the FF VS.
        | D3DRS_SPECULARENABLE
        // LOCALVIEWER selects the specular view-vector model (per-vertex
        // normalize(-posEye) or the constant infinite-viewer direction);
        // feeds `FfVsFlags::LOCAL_VIEWER`.
        | D3DRS_LOCALVIEWER
        // VERTEXBLEND and INDEXEDVERTEXBLENDENABLE feed
        // `FfState::build_vs_key` through `resolve_vertex_blend_count`: the
        // VS blends position and normal across the world-matrix palette.
        | D3DRS_VERTEXBLEND
        | D3DRS_INDEXEDVERTEXBLENDENABLE
        // POINTSIZE, its clamp and POINTSCALE_A..C ride the per-draw VsDraw
        // uniform (`vs_draw`) that every vertex shader clamps
        // `[[point_size]]` from; POINTSIZE also carries the A2M and RESZ
        // control tokens. POINTSCALEENABLE is the `FfVsFlags::POINT_SCALE`
        // key bit; POINTSPRITEENABLE is the `VariantFlags::POINT_SPRITE` PS
        // variant that samples `[[point_coord]]`.
        | D3DRS_POINTSIZE
        | D3DRS_POINTSIZE_MIN
        | D3DRS_POINTSIZE_MAX
        | D3DRS_POINTSCALE_A
        | D3DRS_POINTSCALE_B
        | D3DRS_POINTSCALE_C
        | D3DRS_POINTSCALEENABLE
        | D3DRS_POINTSPRITEENABLE
        // CLIPPING is the master clipping switch: it gates the user clip
        // planes (`vs_draw::clip_plane_count`); its frustum half is a no-op,
        // since Metal always clips to the viewport. CLIPPLANEENABLE selects
        // which of the `SetClipPlane` planes the VsDraw uniform packs and
        // keys the `[[clip_distance]]` lane count of both vertex-shader
        // sources.
        | D3DRS_CLIPPING
        | D3DRS_CLIPPLANEENABLE
        // BLENDFACTOR is the encoder's constant blend colour
        // (`Command::set_blend_color`), set whenever the draw's value
        // differs from the one bound.
        | D3DRS_BLENDFACTOR
        // DEPTHBIAS feeds the vertex shaders' `pos_fixup.depth_bias` and
        // SLOPESCALEDEPTHBIAS Metal's per-encoder rasterizer offset
        // (`Command::set_depth_bias`), both resolved per draw.
        | D3DRS_DEPTHBIAS
        | D3DRS_SLOPESCALEDEPTHBIAS
        // The stencil states reach Metal through
        // `depth_stencil_state::snapshot_from_state`. STENCILREF is the
        // exception by design: it rides the encoder as
        // `SetStencilReference`, not the state object.
        | D3DRS_STENCILENABLE
        | D3DRS_STENCILFAIL
        | D3DRS_STENCILZFAIL
        | D3DRS_STENCILPASS
        | D3DRS_STENCILFUNC
        | D3DRS_STENCILMASK
        | D3DRS_STENCILWRITEMASK
        | D3DRS_STENCILREF
        | D3DRS_TWOSIDEDSTENCILMODE
        | D3DRS_CCW_STENCILFAIL
        | D3DRS_CCW_STENCILZFAIL
        | D3DRS_CCW_STENCILPASS
        | D3DRS_CCW_STENCILFUNC
        // MULTISAMPLEMASK narrows the samples a draw covers; the pixel-shader
        // variant writes it to a `[[sample_mask]]` output, which is where
        // Metal takes a coverage mask.
        | D3DRS_MULTISAMPLEMASK => RsClass::Consumed,

        // MULTISAMPLEANTIALIAS asks the rasterizer to drop to one sample for
        // a draw on a multisampled target. Metal ties the pipeline's
        // `rasterSampleCount` to the attachment's, so there is no per-draw
        // switch to honour it with.
        D3DRS_MULTISAMPLEANTIALIAS => RsClass::Obsolete(
            "Metal has no per-draw multisample toggle (D3DPRASTERCAPS_MULTISAMPLE_TOGGLE is not advertised)",
        ),
        D3DRS_PATCHEDGESTYLE | D3DRS_POSITIONDEGREE | D3DRS_NORMALDEGREE => {
            RsClass::Obsolete("N-patch tessellation is obsolete; every modern driver ignores it")
        }
        // The adaptive tessellation states drive RT-patch and N-patch
        // tessellation, which is not implemented (`DrawRectPatch` and
        // `DrawTriPatch` fail). NVDB on ADAPTIVETESS_X is the depth-bounds
        // switch instead, a missing feature, so it keeps the warning; the
        // bounds it reads from Z and W log here.
        D3DRS_ADAPTIVETESS_X if value == D3DFMT_NVDB => RsClass::NotImplemented,
        D3DRS_MINTESSELLATIONLEVEL
        | D3DRS_MAXTESSELLATIONLEVEL
        | D3DRS_ADAPTIVETESS_X
        | D3DRS_ADAPTIVETESS_Y
        | D3DRS_ADAPTIVETESS_Z
        | D3DRS_ADAPTIVETESS_W
        | D3DRS_ENABLEADAPTIVETESSELLATION => {
            RsClass::Obsolete("adaptive patch tessellation is obsolete and not implemented")
        }
        D3DRS_TWEENFACTOR => RsClass::Obsolete("fixed-function vertex tweening is obsolete"),
        D3DRS_DEBUGMONITORTOKEN => RsClass::Obsolete("debug-only token with no rendering effect"),
        // Dithering changes nothing on the 8-bit and wider targets rendered
        // to, which is why `D3DPRASTERCAPS_DITHER` is advertised.
        D3DRS_DITHERENABLE => {
            RsClass::Obsolete("dithering has no effect on 8-bit and wider render targets")
        }
        // Metal rasterizes lines aliased only, so the cap stays clear and the
        // write has nothing to switch.
        D3DRS_ANTIALIASEDLINEENABLE => RsClass::Obsolete(
            "Metal has no antialiased line rasterization (D3DLINECAPS_ANTIALIAS is not advertised)",
        ),

        // Nothing reads the slot.
        _ => RsClass::NotImplemented,
    }
}

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
