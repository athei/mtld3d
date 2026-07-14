//! D3D9 Fixed-Function pipeline types and enum constants.
//!
//! The structs here mirror the D3D9 SDK layouts bit-for-bit so that application
//! code on the PE side can pass pointers through `SetMaterial(*const D3DMATERIAL9)`
//! etc. without any shuffling.

// ── Primitive struct types ──

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DCOLORVALUE {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DVECTOR {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DMATRIX {
    pub m: [f32; 16],
}

impl D3DMATRIX {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ],
    };
}

impl Default for D3DMATRIX {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DMATERIAL9 {
    pub diffuse: D3DCOLORVALUE,
    pub ambient: D3DCOLORVALUE,
    pub specular: D3DCOLORVALUE,
    pub emissive: D3DCOLORVALUE,
    pub power: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DLIGHT9 {
    pub type_: u32,
    pub diffuse: D3DCOLORVALUE,
    pub specular: D3DCOLORVALUE,
    pub ambient: D3DCOLORVALUE,
    pub position: D3DVECTOR,
    pub direction: D3DVECTOR,
    pub range: f32,
    pub falloff: f32,
    pub attenuation0: f32,
    pub attenuation1: f32,
    pub attenuation2: f32,
    pub theta: f32,
    pub phi: f32,
}

impl Default for D3DLIGHT9 {
    fn default() -> Self {
        Self {
            type_: D3DLIGHT_DIRECTIONAL,
            diffuse: D3DCOLORVALUE::default(),
            specular: D3DCOLORVALUE::default(),
            ambient: D3DCOLORVALUE::default(),
            position: D3DVECTOR::default(),
            direction: D3DVECTOR {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            range: 0.0,
            falloff: 0.0,
            attenuation0: 0.0,
            attenuation1: 0.0,
            attenuation2: 0.0,
            theta: 0.0,
            phi: 0.0,
        }
    }
}

// ── D3DTRANSFORMSTATETYPE ──
//
// D3D9's transform state space is sparse. We collapse it to the subset we
// actually honour: VIEW, PROJECTION, TEXTURE0..7, and WORLD (== WORLDMATRIX(0)).
// Higher WORLDMATRIX indices are ignored (vertex blending is not supported).

pub const D3DTS_VIEW: u32 = 2;
pub const D3DTS_PROJECTION: u32 = 3;
pub const D3DTS_TEXTURE0: u32 = 16;
pub const D3DTS_TEXTURE1: u32 = 17;
pub const D3DTS_TEXTURE2: u32 = 18;
pub const D3DTS_TEXTURE3: u32 = 19;
pub const D3DTS_TEXTURE4: u32 = 20;
pub const D3DTS_TEXTURE5: u32 = 21;
pub const D3DTS_TEXTURE6: u32 = 22;
pub const D3DTS_TEXTURE7: u32 = 23;
pub const D3DTS_WORLD: u32 = 256; // WORLDMATRIX(0)

// ── D3DTEXTURESTAGESTATETYPE ──

pub const D3DTSS_COLOROP: u32 = 1;
pub const D3DTSS_COLORARG1: u32 = 2;
pub const D3DTSS_COLORARG2: u32 = 3;
pub const D3DTSS_ALPHAOP: u32 = 4;
pub const D3DTSS_ALPHAARG1: u32 = 5;
pub const D3DTSS_ALPHAARG2: u32 = 6;
pub const D3DTSS_BUMPENVMAT00: u32 = 7;
pub const D3DTSS_BUMPENVMAT01: u32 = 8;
pub const D3DTSS_BUMPENVMAT10: u32 = 9;
pub const D3DTSS_BUMPENVMAT11: u32 = 10;
pub const D3DTSS_TEXCOORDINDEX: u32 = 11;
pub const D3DTSS_BUMPENVLSCALE: u32 = 22;
pub const D3DTSS_BUMPENVLOFFSET: u32 = 23;
pub const D3DTSS_TEXTURETRANSFORMFLAGS: u32 = 24;
pub const D3DTSS_COLORARG0: u32 = 26;
pub const D3DTSS_ALPHAARG0: u32 = 27;
pub const D3DTSS_RESULTARG: u32 = 28;
pub const D3DTSS_CONSTANT: u32 = 32;
pub const TEXTURE_STAGE_STATE_COUNT: usize = 33;

/// Returns D3D9-spec defaults for the texture stage state array of stage `i`.
///
/// Stage 0 defaults to MODULATE on COLOROP (enabled); stages 1..7 default to
/// DISABLE. Standard D3D9 app-compat behaviour.
#[must_use]
pub const fn texture_stage_state_defaults(stage: u8) -> [u32; TEXTURE_STAGE_STATE_COUNT] {
    let mut s = [0u32; TEXTURE_STAGE_STATE_COUNT];
    s[D3DTSS_COLOROP as usize] = if stage == 0 {
        D3DTOP_MODULATE
    } else {
        D3DTOP_DISABLE
    };
    s[D3DTSS_COLORARG1 as usize] = D3DTA_TEXTURE;
    s[D3DTSS_COLORARG2 as usize] = D3DTA_CURRENT;
    s[D3DTSS_ALPHAOP as usize] = if stage == 0 {
        D3DTOP_SELECTARG1
    } else {
        D3DTOP_DISABLE
    };
    s[D3DTSS_ALPHAARG1 as usize] = D3DTA_TEXTURE;
    s[D3DTSS_ALPHAARG2 as usize] = D3DTA_CURRENT;
    // const fn ⇒ can't use `u32::from(stage)` yet (From not const-stable);
    // bit-explicit widening avoids the `as` cast.
    s[D3DTSS_TEXCOORDINDEX as usize] = u32::from_le_bytes([stage, 0, 0, 0]);
    s[D3DTSS_TEXTURETRANSFORMFLAGS as usize] = D3DTTFF_DISABLE;
    s[D3DTSS_COLORARG0 as usize] = D3DTA_CURRENT;
    s[D3DTSS_ALPHAARG0 as usize] = D3DTA_CURRENT;
    s[D3DTSS_RESULTARG as usize] = D3DTA_CURRENT;
    s
}

impl crate::StateBlockType {
    /// Whether a state block of this type captures/restores texture-stage state `ty` (`D3DTSS_*`).
    ///
    /// Mirrors Wine's `vertex_states_texture` (texcoord index and transform
    /// flags) and `pixel_states_texture` (the colour/alpha op chain, bump-env,
    /// and result arg, which also cover texcoord index and transform flags).
    /// Defined alongside the `D3DTSS_*` constants so the membership list stays
    /// next to the values it classifies.
    #[must_use]
    pub const fn includes_tss(self, ty: u32) -> bool {
        match self {
            Self::All => true,
            Self::Vertex => matches!(ty, D3DTSS_TEXCOORDINDEX | D3DTSS_TEXTURETRANSFORMFLAGS),
            Self::Pixel => matches!(
                ty,
                D3DTSS_COLOROP
                    | D3DTSS_COLORARG0
                    | D3DTSS_COLORARG1
                    | D3DTSS_COLORARG2
                    | D3DTSS_ALPHAOP
                    | D3DTSS_ALPHAARG0
                    | D3DTSS_ALPHAARG1
                    | D3DTSS_ALPHAARG2
                    | D3DTSS_BUMPENVMAT00
                    | D3DTSS_BUMPENVMAT01
                    | D3DTSS_BUMPENVMAT10
                    | D3DTSS_BUMPENVMAT11
                    | D3DTSS_BUMPENVLSCALE
                    | D3DTSS_BUMPENVLOFFSET
                    | D3DTSS_RESULTARG
                    | D3DTSS_TEXCOORDINDEX
                    | D3DTSS_TEXTURETRANSFORMFLAGS
            ),
        }
    }
}

// ── D3DLIGHTTYPE ──

pub const D3DLIGHT_POINT: u32 = 1;
pub const D3DLIGHT_SPOT: u32 = 2;
pub const D3DLIGHT_DIRECTIONAL: u32 = 3;

// ── D3DTEXTUREOP ──

pub const D3DTOP_DISABLE: u32 = 1;
pub const D3DTOP_SELECTARG1: u32 = 2;
pub const D3DTOP_SELECTARG2: u32 = 3;
pub const D3DTOP_MODULATE: u32 = 4;
pub const D3DTOP_MODULATE2X: u32 = 5;
pub const D3DTOP_MODULATE4X: u32 = 6;
pub const D3DTOP_ADD: u32 = 7;
pub const D3DTOP_ADDSIGNED: u32 = 8;
pub const D3DTOP_ADDSIGNED2X: u32 = 9;
pub const D3DTOP_SUBTRACT: u32 = 10;
pub const D3DTOP_ADDSMOOTH: u32 = 11;
pub const D3DTOP_BLENDDIFFUSEALPHA: u32 = 12;
pub const D3DTOP_BLENDTEXTUREALPHA: u32 = 13;
pub const D3DTOP_BLENDFACTORALPHA: u32 = 14;
pub const D3DTOP_BLENDTEXTUREALPHAPM: u32 = 15;
pub const D3DTOP_BLENDCURRENTALPHA: u32 = 16;
pub const D3DTOP_PREMODULATE: u32 = 17;
pub const D3DTOP_MODULATEALPHA_ADDCOLOR: u32 = 18;
pub const D3DTOP_MODULATECOLOR_ADDALPHA: u32 = 19;
pub const D3DTOP_MODULATEINVALPHA_ADDCOLOR: u32 = 20;
pub const D3DTOP_MODULATEINVCOLOR_ADDALPHA: u32 = 21;
pub const D3DTOP_BUMPENVMAP: u32 = 22;
pub const D3DTOP_BUMPENVMAPLUMINANCE: u32 = 23;
pub const D3DTOP_DOTPRODUCT3: u32 = 24;
pub const D3DTOP_MULTIPLYADD: u32 = 25;
pub const D3DTOP_LERP: u32 = 26;

// ── D3DTEXTUREARG ──

pub const D3DTA_DIFFUSE: u32 = 0;
pub const D3DTA_CURRENT: u32 = 1;
pub const D3DTA_TEXTURE: u32 = 2;
pub const D3DTA_TFACTOR: u32 = 3;
pub const D3DTA_SPECULAR: u32 = 4;
pub const D3DTA_TEMP: u32 = 5;
pub const D3DTA_CONSTANT: u32 = 6;
pub const D3DTA_SELECTMASK: u32 = 0x0000_000f;
pub const D3DTA_COMPLEMENT: u32 = 0x0000_0010;
pub const D3DTA_ALPHAREPLICATE: u32 = 0x0000_0020;

// ── D3DMATERIALCOLORSOURCE ──

pub const D3DMCS_MATERIAL: u32 = 0;
pub const D3DMCS_COLOR1: u32 = 1;
pub const D3DMCS_COLOR2: u32 = 2;

// ── D3DTEXTURETRANSFORMFLAGS ──

pub const D3DTTFF_DISABLE: u32 = 0;
pub const D3DTTFF_COUNT1: u32 = 1;
pub const D3DTTFF_COUNT2: u32 = 2;
pub const D3DTTFF_COUNT3: u32 = 3;
pub const D3DTTFF_COUNT4: u32 = 4;
pub const D3DTTFF_PROJECTED: u32 = 256;
