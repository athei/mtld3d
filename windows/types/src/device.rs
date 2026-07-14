use core::ffi::c_void;

use super::Guid;

// ── D3D9 render state indices ──

pub const D3DRS_ZENABLE: u32 = 7;
pub const D3DRS_FILLMODE: u32 = 8;
pub const D3DRS_SHADEMODE: u32 = 9;
pub const D3DRS_ZWRITEENABLE: u32 = 14;
pub const D3DRS_ALPHATESTENABLE: u32 = 15;
pub const D3DRS_LASTPIXEL: u32 = 16;
pub const D3DRS_SRCBLEND: u32 = 19;
pub const D3DRS_DESTBLEND: u32 = 20;
pub const D3DRS_CULLMODE: u32 = 22;
pub const D3DRS_ZFUNC: u32 = 23;
pub const D3DRS_ALPHAREF: u32 = 24;
pub const D3DRS_ALPHAFUNC: u32 = 25;
pub const D3DRS_DITHERENABLE: u32 = 26;
pub const D3DRS_ALPHABLENDENABLE: u32 = 27;
pub const D3DRS_FOGENABLE: u32 = 28;
pub const D3DRS_SPECULARENABLE: u32 = 29;
pub const D3DRS_FOGCOLOR: u32 = 34;
pub const D3DRS_FOGTABLEMODE: u32 = 35;
pub const D3DRS_FOGSTART: u32 = 36;
pub const D3DRS_FOGEND: u32 = 37;
pub const D3DRS_FOGDENSITY: u32 = 38;
pub const D3DRS_RANGEFOGENABLE: u32 = 48;
pub const D3DRS_STENCILENABLE: u32 = 52;
pub const D3DRS_STENCILFAIL: u32 = 53;
pub const D3DRS_STENCILZFAIL: u32 = 54;
pub const D3DRS_STENCILPASS: u32 = 55;
pub const D3DRS_STENCILFUNC: u32 = 56;
pub const D3DRS_STENCILREF: u32 = 57;
pub const D3DRS_STENCILMASK: u32 = 58;
pub const D3DRS_STENCILWRITEMASK: u32 = 59;
pub const D3DRS_TEXTUREFACTOR: u32 = 60;
pub const D3DRS_WRAP0: u32 = 128;
pub const D3DRS_WRAP1: u32 = 129;
pub const D3DRS_WRAP2: u32 = 130;
pub const D3DRS_WRAP3: u32 = 131;
pub const D3DRS_WRAP4: u32 = 132;
pub const D3DRS_WRAP5: u32 = 133;
pub const D3DRS_WRAP6: u32 = 134;
pub const D3DRS_WRAP7: u32 = 135;
pub const D3DRS_CLIPPING: u32 = 136;
pub const D3DRS_LIGHTING: u32 = 137;
pub const D3DRS_AMBIENT: u32 = 139;
pub const D3DRS_FOGVERTEXMODE: u32 = 140;
pub const D3DRS_COLORVERTEX: u32 = 141;
pub const D3DRS_LOCALVIEWER: u32 = 142;
pub const D3DRS_NORMALIZENORMALS: u32 = 143;
pub const D3DRS_DIFFUSEMATERIALSOURCE: u32 = 145;
pub const D3DRS_SPECULARMATERIALSOURCE: u32 = 146;
pub const D3DRS_AMBIENTMATERIALSOURCE: u32 = 147;
pub const D3DRS_EMISSIVEMATERIALSOURCE: u32 = 148;
pub const D3DRS_VERTEXBLEND: u32 = 151;
pub const D3DRS_CLIPPLANEENABLE: u32 = 152;
pub const D3DRS_POINTSIZE: u32 = 154;
pub const D3DRS_POINTSIZE_MIN: u32 = 155;
pub const D3DRS_POINTSPRITEENABLE: u32 = 156;
pub const D3DRS_POINTSCALEENABLE: u32 = 157;
pub const D3DRS_POINTSCALE_A: u32 = 158;
pub const D3DRS_POINTSCALE_B: u32 = 159;
pub const D3DRS_POINTSCALE_C: u32 = 160;
pub const D3DRS_MULTISAMPLEANTIALIAS: u32 = 161;
pub const D3DRS_MULTISAMPLEMASK: u32 = 162;
pub const D3DRS_PATCHEDGESTYLE: u32 = 163;
pub const D3DRS_DEBUGMONITORTOKEN: u32 = 165;
pub const D3DRS_POINTSIZE_MAX: u32 = 166;
pub const D3DRS_INDEXEDVERTEXBLENDENABLE: u32 = 167;
pub const D3DRS_COLORWRITEENABLE: u32 = 168;
pub const D3DRS_TWEENFACTOR: u32 = 170;
pub const D3DRS_BLENDOP: u32 = 171;
pub const D3DRS_POSITIONDEGREE: u32 = 172;
pub const D3DRS_NORMALDEGREE: u32 = 173;
pub const D3DRS_SCISSORTESTENABLE: u32 = 174;
pub const D3DRS_SLOPESCALEDEPTHBIAS: u32 = 175;
pub const D3DRS_ANTIALIASEDLINEENABLE: u32 = 176;
pub const D3DRS_MINTESSELLATIONLEVEL: u32 = 178;
pub const D3DRS_MAXTESSELLATIONLEVEL: u32 = 179;
pub const D3DRS_ADAPTIVETESS_X: u32 = 180;
pub const D3DRS_ADAPTIVETESS_Y: u32 = 181;
pub const D3DRS_ADAPTIVETESS_Z: u32 = 182;
pub const D3DRS_ADAPTIVETESS_W: u32 = 183;
pub const D3DRS_ENABLEADAPTIVETESSELLATION: u32 = 184;
pub const D3DRS_TWOSIDEDSTENCILMODE: u32 = 185;
pub const D3DRS_CCW_STENCILFAIL: u32 = 186;
pub const D3DRS_CCW_STENCILZFAIL: u32 = 187;
pub const D3DRS_CCW_STENCILPASS: u32 = 188;
pub const D3DRS_CCW_STENCILFUNC: u32 = 189;
pub const D3DRS_COLORWRITEENABLE1: u32 = 190;
pub const D3DRS_COLORWRITEENABLE2: u32 = 191;
pub const D3DRS_COLORWRITEENABLE3: u32 = 192;
pub const D3DRS_BLENDFACTOR: u32 = 193;
pub const D3DRS_SRGBWRITEENABLE: u32 = 194;
pub const D3DRS_DEPTHBIAS: u32 = 195;
pub const D3DRS_WRAP8: u32 = 198;
pub const D3DRS_WRAP9: u32 = 199;
pub const D3DRS_WRAP10: u32 = 200;
pub const D3DRS_WRAP11: u32 = 201;
pub const D3DRS_WRAP12: u32 = 202;
pub const D3DRS_WRAP13: u32 = 203;
pub const D3DRS_WRAP14: u32 = 204;
pub const D3DRS_WRAP15: u32 = 205;
pub const D3DRS_SEPARATEALPHABLENDENABLE: u32 = 206;
pub const D3DRS_SRCBLENDALPHA: u32 = 207;
pub const D3DRS_DESTBLENDALPHA: u32 = 208;
pub const D3DRS_BLENDOPALPHA: u32 = 209;

// ── D3D9 compare functions ──

pub const D3DCMP_NEVER: u32 = 1;
pub const D3DCMP_LESS: u32 = 2;
pub const D3DCMP_EQUAL: u32 = 3;
pub const D3DCMP_LESSEQUAL: u32 = 4;
pub const D3DCMP_GREATER: u32 = 5;
pub const D3DCMP_NOTEQUAL: u32 = 6;
pub const D3DCMP_GREATEREQUAL: u32 = 7;
pub const D3DCMP_ALWAYS: u32 = 8;

// ── D3D9 blend factors ──

pub const D3DBLEND_ZERO: u32 = 1;
pub const D3DBLEND_ONE: u32 = 2;
pub const D3DBLEND_SRCCOLOR: u32 = 3;
pub const D3DBLEND_INVSRCCOLOR: u32 = 4;
pub const D3DBLEND_SRCALPHA: u32 = 5;
pub const D3DBLEND_INVSRCALPHA: u32 = 6;
pub const D3DBLEND_DESTALPHA: u32 = 7;
pub const D3DBLEND_INVDESTALPHA: u32 = 8;
pub const D3DBLEND_DESTCOLOR: u32 = 9;
pub const D3DBLEND_INVDESTCOLOR: u32 = 10;
pub const D3DBLEND_SRCALPHASAT: u32 = 11;
pub const D3DBLEND_BLENDFACTOR: u32 = 14;
pub const D3DBLEND_INVBLENDFACTOR: u32 = 15;

// ── D3D9 blend operations ──

pub const D3DBLENDOP_ADD: u32 = 1;
pub const D3DBLENDOP_SUBTRACT: u32 = 2;
pub const D3DBLENDOP_REVSUBTRACT: u32 = 3;
pub const D3DBLENDOP_MIN: u32 = 4;
pub const D3DBLENDOP_MAX: u32 = 5;

// ── D3D9 cull modes ──

pub const D3DCULL_NONE: u32 = 1;
pub const D3DCULL_CW: u32 = 2;
pub const D3DCULL_CCW: u32 = 3;

// ── D3D9 vertex-blend modes (`D3DRS_VERTEXBLEND` value space) ──

pub const D3DVBF_DISABLE: u32 = 0;
pub const D3DVBF_1WEIGHTS: u32 = 1;
pub const D3DVBF_2WEIGHTS: u32 = 2;
pub const D3DVBF_3WEIGHTS: u32 = 3;
pub const D3DVBF_TWEENING: u32 = 255;
pub const D3DVBF_0WEIGHTS: u32 = 256;

// ── D3D9 device caps2 bits (`D3DCAPS9::DevCaps2`) ──

/// The device can `StretchRect` from a texture-level surface into a render target.
///
/// Advertised by default because the blit path supports it.
pub const D3DDEVCAPS2_CAN_STRETCHRECT_FROM_TEXTURES: u32 = 0x0000_0010;

// ── D3D9 clear flags ──

pub const D3DCLEAR_TARGET: u32 = 0x0000_0001;
pub const D3DCLEAR_ZBUFFER: u32 = 0x0000_0002;
pub const D3DCLEAR_STENCIL: u32 = 0x0000_0004;

// ── D3D9 lock flags (`Lock` / `LockRect`) ──

pub const D3DLOCK_READONLY: u32 = 0x0000_0010;
pub const D3DLOCK_NOSYSLOCK: u32 = 0x0000_0800;
pub const D3DLOCK_NOOVERWRITE: u32 = 0x0000_1000;
pub const D3DLOCK_DISCARD: u32 = 0x0000_2000;
pub const D3DLOCK_DONOTWAIT: u32 = 0x0000_4000;
pub const D3DLOCK_NO_DIRTY_UPDATE: u32 = 0x0000_8000;

/// Bits we either explicitly handle or know are safe to ignore.
///
/// See the per-flag handling at each `Lock` call site.
///
/// Any flag bit *outside* this mask should be surfaced via `log_once_warn!`
/// so a future game's stricter expectations don't silently break.
pub const D3DLOCK_KNOWN_BITS: u32 = D3DLOCK_READONLY
    | D3DLOCK_NOSYSLOCK
    | D3DLOCK_NOOVERWRITE
    | D3DLOCK_DISCARD
    | D3DLOCK_DONOTWAIT
    | D3DLOCK_NO_DIRTY_UPDATE;

// ── D3D9 usage flags (`CreateVertexBuffer` / `CreateIndexBuffer` /
// `CreateTexture`) ──

pub const D3DUSAGE_RENDERTARGET: u32 = 0x0000_0001;
pub const D3DUSAGE_DEPTHSTENCIL: u32 = 0x0000_0002;
pub const D3DUSAGE_WRITEONLY: u32 = 0x0000_0008;
pub const D3DUSAGE_SOFTWAREPROCESSING: u32 = 0x0000_0010;
pub const D3DUSAGE_DONOTCLIP: u32 = 0x0000_0020;
pub const D3DUSAGE_POINTS: u32 = 0x0000_0040;
pub const D3DUSAGE_RTPATCHES: u32 = 0x0000_0080;
pub const D3DUSAGE_NPATCHES: u32 = 0x0000_0100;
pub const D3DUSAGE_DYNAMIC: u32 = 0x0000_0200;
pub const D3DUSAGE_AUTOGENMIPMAP: u32 = 0x0000_0400;
pub const D3DUSAGE_QUERY_SRGBREAD: u32 = 0x0001_0000;
pub const D3DUSAGE_QUERY_SRGBWRITE: u32 = 0x0004_0000;
pub const D3DUSAGE_NONSECURE: u32 = 0x0080_0000;

// ── D3D9 resource pools (`D3DPOOL`, the `Pool` arg of the Create*
// methods) ──

pub const D3DPOOL_DEFAULT: u32 = 0;
pub const D3DPOOL_MANAGED: u32 = 1;
pub const D3DPOOL_SYSTEMMEM: u32 = 2;
pub const D3DPOOL_SCRATCH: u32 = 3;

// ── D3D9 device types (`D3DDEVTYPE`) ──

pub const D3DDEVTYPE_HAL: u32 = 1;

// ── D3D9 resource types (`D3DRESOURCETYPE`, reported by `GetType` /
// `GetDesc`) ──

pub const D3DRTYPE_SURFACE: u32 = 1;
pub const D3DRTYPE_VOLUME: u32 = 2;
pub const D3DRTYPE_TEXTURE: u32 = 3;
pub const D3DRTYPE_VOLUMETEXTURE: u32 = 4;
pub const D3DRTYPE_CUBETEXTURE: u32 = 5;
pub const D3DRTYPE_VERTEXBUFFER: u32 = 6;
pub const D3DRTYPE_INDEXBUFFER: u32 = 7;

// ── D3D9 multisample types (`D3DMULTISAMPLE_TYPE`) ──

pub const D3DMULTISAMPLE_NONE: u32 = 0;

// ── D3D9 state-block types (`D3DSTATEBLOCKTYPE`) ──

pub const D3DSBT_ALL: u32 = 1;
pub const D3DSBT_PIXELSTATE: u32 = 2;
pub const D3DSBT_VERTEXSTATE: u32 = 3;

// ── D3D9 texture formats ──

pub const D3DFMT_R8G8B8: u32 = 20;
pub const D3DFMT_R5G6B5: u32 = 23;
pub const D3DFMT_X1R5G5B5: u32 = 24;
pub const D3DFMT_A1R5G5B5: u32 = 25;
pub const D3DFMT_A8R8G8B8: u32 = 21;
pub const D3DFMT_X8R8G8B8: u32 = 22;
pub const D3DFMT_A2R10G10B10: u32 = 35;
pub const D3DFMT_A4R4G4B4: u32 = 26;
pub const D3DFMT_A8: u32 = 28;
pub const D3DFMT_L8: u32 = 50;
pub const D3DFMT_L16: u32 = 81;
pub const D3DFMT_A8L8: u32 = 51;
pub const D3DFMT_V8U8: u32 = 60;
pub const D3DFMT_R16F: u32 = 111;
pub const D3DFMT_R32F: u32 = 114;
pub const D3DFMT_A32B32G32R32F: u32 = 116;
pub const D3DFMT_DXT1: u32 = 0x3154_5844;
pub const D3DFMT_DXT2: u32 = 0x3254_5844;
pub const D3DFMT_DXT3: u32 = 0x3354_5844;
pub const D3DFMT_DXT4: u32 = 0x3454_5844;
pub const D3DFMT_DXT5: u32 = 0x3554_5844;
pub const D3DFMT_YUY2: u32 = 0x3259_5559;
pub const D3DFMT_UYVY: u32 = 0x5956_5955;
/// `MAKEFOURCC('A','T','I','1')` — ATI1N / BC4 single-channel block format.
pub const D3DFMT_ATI1: u32 = 0x3149_5441;

// ── D3D9 sampler state types ──

pub const D3DSAMP_ADDRESSU: u32 = 1;
pub const D3DSAMP_ADDRESSV: u32 = 2;
pub const D3DSAMP_ADDRESSW: u32 = 3;
pub const D3DSAMP_BORDERCOLOR: u32 = 4;
pub const D3DSAMP_MAGFILTER: u32 = 5;
pub const D3DSAMP_MINFILTER: u32 = 6;
pub const D3DSAMP_MIPFILTER: u32 = 7;
pub const D3DSAMP_MIPMAPLODBIAS: u32 = 8;
pub const D3DSAMP_MAXMIPLEVEL: u32 = 9;
pub const D3DSAMP_MAXANISOTROPY: u32 = 10;
pub const D3DSAMP_SRGBTEXTURE: u32 = 11;
pub const D3DSAMP_ELEMENTINDEX: u32 = 12;
pub const D3DSAMP_DMAPOFFSET: u32 = 13;
pub const SAMPLER_STATE_COUNT: usize = 14;

// ── D3D9 texture filter types ──

pub const D3DTEXF_NONE: u32 = 0;
pub const D3DTEXF_POINT: u32 = 1;
pub const D3DTEXF_LINEAR: u32 = 2;
pub const D3DTEXF_ANISOTROPIC: u32 = 3;

// ── D3D9 texture address modes ──

pub const D3DTADDRESS_WRAP: u32 = 1;
pub const D3DTADDRESS_MIRROR: u32 = 2;
pub const D3DTADDRESS_CLAMP: u32 = 3;
pub const D3DTADDRESS_BORDER: u32 = 4;
pub const D3DTADDRESS_MIRRORONCE: u32 = 5;

// ── D3D9 depth/stencil formats ──

pub const D3DFMT_D16_LOCKABLE: u32 = 70;
pub const D3DFMT_D32: u32 = 71;
pub const D3DFMT_D15S1: u32 = 73;
pub const D3DFMT_D24S8: u32 = 75;
pub const D3DFMT_D24X8: u32 = 77;
pub const D3DFMT_D24X4S4: u32 = 79;
pub const D3DFMT_D16: u32 = 80;
pub const D3DFMT_D32F_LOCKABLE: u32 = 82;
pub const D3DFMT_D24FS8: u32 = 83;

// ── D3D9 FOURCC sampleable-depth formats ──
//
// Hardware-shadow-mapping textures: created with `D3DUSAGE_DEPTHSTENCIL`,
// bound as the depth target during the caster pass, and sampled as a depth
// texture during the lit pass. Every D3D9-era vendor exposes some subset;
// `INTZ`/`DF24` are 24-bit-depth, `DF16` is 16-bit. Engines gate their CSM
// path on at least one being available.
pub const D3DFMT_INTZ: u32 = 0x5A54_4E49; // 'INTZ' little-endian
pub const D3DFMT_DF24: u32 = 0x3432_4644; // 'DF24'
pub const D3DFMT_DF16: u32 = 0x3631_4644; // 'DF16'

// ── D3D9 buffer formats (vertex-/index-buffer `Format`) ──

pub const D3DFMT_VERTEXDATA: u32 = 100; // what `GetDesc` reports for VBs
pub const D3DFMT_INDEX16: u32 = 101;
pub const D3DFMT_INDEX32: u32 = 102;

// ── D3D9 fill modes ──

pub const D3DFILL_POINT: u32 = 1;
pub const D3DFILL_WIREFRAME: u32 = 2;
pub const D3DFILL_SOLID: u32 = 3;

// ── D3D9 shade modes ──

pub const D3DSHADE_FLAT: u32 = 1;
pub const D3DSHADE_GOURAUD: u32 = 2;

// ── D3D9 stencil operations ──

pub const D3DSTENCILOP_KEEP: u32 = 1;
pub const D3DSTENCILOP_ZERO: u32 = 2;
pub const D3DSTENCILOP_REPLACE: u32 = 3;
pub const D3DSTENCILOP_INCRSAT: u32 = 4;
pub const D3DSTENCILOP_DECRSAT: u32 = 5;
pub const D3DSTENCILOP_INVERT: u32 = 6;
pub const D3DSTENCILOP_INCR: u32 = 7;
pub const D3DSTENCILOP_DECR: u32 = 8;

// ── Render state count ──

pub const RENDER_STATE_COUNT: usize = 210;

/// The maximum point size we advertise as `D3DCAPS9::MaxPointSize`.
///
/// Point sprites / point scaling are not implemented and Metal renders
/// 1-pixel points, so this caps at 1.0 to steer titles away from large points.
/// D3D9 defines the `D3DRS_POINTSIZE_MAX` render-state default as *equal to*
/// this cap, so [`render_state_defaults`] and `caps.max_point_size` both read
/// it — the two can't drift out of the spec relationship.
pub const MAX_POINT_SIZE: f32 = 1.0;

// ── D3D9 state-block filtering (`D3DSTATEBLOCKTYPE`) ──

/// Which slice of device state a `CreateStateBlock` snapshot captures and, on `Apply`, writes back.
///
/// `Vertex`/`Pixel` implement the D3D9 `D3DSBT_VERTEXSTATE` /
/// `D3DSBT_PIXELSTATE` classification of which states each filtered block owns:
/// a filtered block restores only its own pipeline's states and leaves the rest
/// of the device untouched, so applying e.g. a `D3DSBT_VERTEXSTATE` block must
/// not clobber blend/stencil (pixel) state. `All` captures and restores
/// everything.
///
/// The per-state membership predicates ([`includes_render_state`](Self::includes_render_state),
/// [`includes_sampler_state`](Self::includes_sampler_state),
/// [`includes_tss`](Self::includes_tss)) are the single source of truth for the
/// filter; the state-block apply path consults them per state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateBlockType {
    All,
    Vertex,
    Pixel,
}

impl StateBlockType {
    /// Map a `D3DSTATEBLOCKTYPE` value to its filter, or `None` for an unrecognised type.
    ///
    /// The caller rejects with `D3DERR_INVALIDCALL`.
    #[must_use]
    pub const fn from_d3dsbt(type_: u32) -> Option<Self> {
        match type_ {
            D3DSBT_ALL => Some(Self::All),
            D3DSBT_VERTEXSTATE => Some(Self::Vertex),
            D3DSBT_PIXELSTATE => Some(Self::Pixel),
            _ => None,
        }
    }

    /// Whether this block restores vertex-pipeline objects.
    ///
    /// The vertex shader and its constants (float/int/bool), the vertex
    /// declaration, and FVF. (Lights are vertex-pipeline too but ride along in
    /// the FF snapshot.)
    #[must_use]
    pub const fn includes_vertex_pipeline(self) -> bool {
        matches!(self, Self::All | Self::Vertex)
    }

    /// Whether this block restores pixel-pipeline objects.
    ///
    /// The pixel shader and its constants (float/int/bool).
    #[must_use]
    pub const fn includes_pixel_pipeline(self) -> bool {
        matches!(self, Self::All | Self::Pixel)
    }

    /// Whether a block of this type captures/restores render state `idx`.
    ///
    /// The `Vertex`/`Pixel` sets follow the D3D9 `D3DSBT_VERTEXSTATE` /
    /// `D3DSBT_PIXELSTATE` render-state classification; `SHADEMODE` and the fog
    /// scalars are members of both, by design.
    #[must_use]
    pub const fn includes_render_state(self, idx: u32) -> bool {
        match self {
            Self::All => true,
            Self::Vertex => matches!(
                idx,
                D3DRS_ADAPTIVETESS_W
                    | D3DRS_ADAPTIVETESS_X
                    | D3DRS_ADAPTIVETESS_Y
                    | D3DRS_ADAPTIVETESS_Z
                    | D3DRS_AMBIENT
                    | D3DRS_AMBIENTMATERIALSOURCE
                    | D3DRS_CLIPPING
                    | D3DRS_CLIPPLANEENABLE
                    | D3DRS_COLORVERTEX
                    | D3DRS_CULLMODE
                    | D3DRS_DIFFUSEMATERIALSOURCE
                    | D3DRS_EMISSIVEMATERIALSOURCE
                    | D3DRS_ENABLEADAPTIVETESSELLATION
                    | D3DRS_FOGCOLOR
                    | D3DRS_FOGDENSITY
                    | D3DRS_FOGENABLE
                    | D3DRS_FOGEND
                    | D3DRS_FOGSTART
                    | D3DRS_FOGTABLEMODE
                    | D3DRS_FOGVERTEXMODE
                    | D3DRS_INDEXEDVERTEXBLENDENABLE
                    | D3DRS_LIGHTING
                    | D3DRS_LOCALVIEWER
                    | D3DRS_MAXTESSELLATIONLEVEL
                    | D3DRS_MINTESSELLATIONLEVEL
                    | D3DRS_MULTISAMPLEANTIALIAS
                    | D3DRS_MULTISAMPLEMASK
                    | D3DRS_NORMALDEGREE
                    | D3DRS_NORMALIZENORMALS
                    | D3DRS_PATCHEDGESTYLE
                    | D3DRS_POINTSCALE_A
                    | D3DRS_POINTSCALE_B
                    | D3DRS_POINTSCALE_C
                    | D3DRS_POINTSCALEENABLE
                    | D3DRS_POINTSIZE
                    | D3DRS_POINTSIZE_MAX
                    | D3DRS_POINTSIZE_MIN
                    | D3DRS_POINTSPRITEENABLE
                    | D3DRS_POSITIONDEGREE
                    | D3DRS_RANGEFOGENABLE
                    | D3DRS_SHADEMODE
                    | D3DRS_SPECULARENABLE
                    | D3DRS_SPECULARMATERIALSOURCE
                    | D3DRS_TWEENFACTOR
                    | D3DRS_VERTEXBLEND
            ),
            Self::Pixel => matches!(
                idx,
                D3DRS_ALPHABLENDENABLE
                    | D3DRS_ALPHAFUNC
                    | D3DRS_ALPHAREF
                    | D3DRS_ALPHATESTENABLE
                    | D3DRS_ANTIALIASEDLINEENABLE
                    | D3DRS_BLENDFACTOR
                    | D3DRS_BLENDOP
                    | D3DRS_BLENDOPALPHA
                    | D3DRS_CCW_STENCILFAIL
                    | D3DRS_CCW_STENCILPASS
                    | D3DRS_CCW_STENCILZFAIL
                    | D3DRS_COLORWRITEENABLE
                    | D3DRS_COLORWRITEENABLE1
                    | D3DRS_COLORWRITEENABLE2
                    | D3DRS_COLORWRITEENABLE3
                    | D3DRS_DEPTHBIAS
                    | D3DRS_DESTBLEND
                    | D3DRS_DESTBLENDALPHA
                    | D3DRS_DITHERENABLE
                    | D3DRS_FILLMODE
                    | D3DRS_FOGDENSITY
                    | D3DRS_FOGEND
                    | D3DRS_FOGSTART
                    | D3DRS_LASTPIXEL
                    | D3DRS_SCISSORTESTENABLE
                    | D3DRS_SEPARATEALPHABLENDENABLE
                    | D3DRS_SHADEMODE
                    | D3DRS_SLOPESCALEDEPTHBIAS
                    | D3DRS_SRCBLEND
                    | D3DRS_SRCBLENDALPHA
                    | D3DRS_SRGBWRITEENABLE
                    | D3DRS_STENCILENABLE
                    | D3DRS_STENCILFAIL
                    | D3DRS_STENCILFUNC
                    | D3DRS_STENCILMASK
                    | D3DRS_STENCILPASS
                    | D3DRS_STENCILREF
                    | D3DRS_STENCILWRITEMASK
                    | D3DRS_STENCILZFAIL
                    | D3DRS_TEXTUREFACTOR
                    | D3DRS_TWOSIDEDSTENCILMODE
                    | D3DRS_WRAP0..=D3DRS_WRAP7
                    | D3DRS_WRAP8..=D3DRS_WRAP15
                    | D3DRS_ZENABLE
                    | D3DRS_ZFUNC
                    | D3DRS_ZWRITEENABLE
            ),
        }
    }

    /// Whether a block of this type captures/restores sampler state `ty` (`D3DSAMP_*`).
    ///
    /// `Vertex` covers only `DMAPOFFSET` (vertex-sampler displacement, the sole
    /// sampler state in the D3D9 `D3DSBT_VERTEXSTATE` set; the vertex-sampler
    /// register range 256+ is not modelled, so this is a no-op for the colour
    /// samplers 0-15). `Pixel` covers the colour sampler states.
    #[must_use]
    pub const fn includes_sampler_state(self, ty: u32) -> bool {
        match self {
            Self::All => true,
            Self::Vertex => ty == D3DSAMP_DMAPOFFSET,
            Self::Pixel => matches!(
                ty,
                D3DSAMP_ADDRESSU
                    | D3DSAMP_ADDRESSV
                    | D3DSAMP_ADDRESSW
                    | D3DSAMP_BORDERCOLOR
                    | D3DSAMP_MAGFILTER
                    | D3DSAMP_MINFILTER
                    | D3DSAMP_MIPFILTER
                    | D3DSAMP_MIPMAPLODBIAS
                    | D3DSAMP_MAXMIPLEVEL
                    | D3DSAMP_MAXANISOTROPY
                    | D3DSAMP_SRGBTEXTURE
                    | D3DSAMP_ELEMENTINDEX
            ),
        }
    }
}

/// Returns an array of D3D9 render state defaults per the specification.
#[must_use]
pub const fn render_state_defaults() -> [u32; RENDER_STATE_COUNT] {
    let mut rs = [0u32; RENDER_STATE_COUNT];

    rs[D3DRS_ZENABLE as usize] = 1; // D3DZB_TRUE (when depth buffer present)
    rs[D3DRS_FILLMODE as usize] = D3DFILL_SOLID;
    rs[D3DRS_SHADEMODE as usize] = D3DSHADE_GOURAUD;
    rs[D3DRS_ZWRITEENABLE as usize] = 1; // TRUE
    rs[D3DRS_ALPHATESTENABLE as usize] = 0; // FALSE
    rs[D3DRS_LASTPIXEL as usize] = 1; // TRUE
    rs[D3DRS_SRCBLEND as usize] = D3DBLEND_ONE;
    rs[D3DRS_DESTBLEND as usize] = D3DBLEND_ZERO;
    rs[D3DRS_CULLMODE as usize] = D3DCULL_CCW;
    rs[D3DRS_ZFUNC as usize] = D3DCMP_LESSEQUAL;
    rs[D3DRS_ALPHAREF as usize] = 0;
    rs[D3DRS_ALPHAFUNC as usize] = D3DCMP_ALWAYS;
    rs[D3DRS_DITHERENABLE as usize] = 0; // FALSE
    rs[D3DRS_ALPHABLENDENABLE as usize] = 0; // FALSE
    rs[D3DRS_FOGENABLE as usize] = 0; // FALSE
    rs[D3DRS_SPECULARENABLE as usize] = 0; // FALSE
    rs[D3DRS_FOGCOLOR as usize] = 0;
    rs[D3DRS_FOGTABLEMODE as usize] = 0; // D3DFOG_NONE
    rs[D3DRS_FOGSTART as usize] = 0; // 0.0f
    rs[D3DRS_FOGEND as usize] = f32::to_bits(1.0);
    rs[D3DRS_FOGDENSITY as usize] = f32::to_bits(1.0);
    rs[D3DRS_RANGEFOGENABLE as usize] = 0; // FALSE
    rs[D3DRS_STENCILENABLE as usize] = 0; // FALSE
    rs[D3DRS_STENCILFAIL as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_STENCILZFAIL as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_STENCILPASS as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_STENCILFUNC as usize] = D3DCMP_ALWAYS;
    rs[D3DRS_STENCILREF as usize] = 0;
    rs[D3DRS_STENCILMASK as usize] = 0xFFFF_FFFF;
    rs[D3DRS_STENCILWRITEMASK as usize] = 0xFFFF_FFFF;
    rs[D3DRS_TEXTUREFACTOR as usize] = 0xFFFF_FFFF;
    rs[D3DRS_WRAP0 as usize] = 0;
    rs[D3DRS_CLIPPING as usize] = 1; // TRUE — spec default; Metal always
    // clips to the viewport so honoring this toggle is a no-op on our
    // side, but mirroring the spec's default here suppresses a warn
    // when the game affirmatively sets CLIPPING=TRUE at startup.
    rs[D3DRS_LIGHTING as usize] = 1; // TRUE
    rs[D3DRS_AMBIENT as usize] = 0;
    rs[D3DRS_FOGVERTEXMODE as usize] = 0; // D3DFOG_NONE
    rs[D3DRS_COLORVERTEX as usize] = 1; // TRUE
    rs[D3DRS_LOCALVIEWER as usize] = 1; // TRUE
    rs[D3DRS_NORMALIZENORMALS as usize] = 0; // FALSE
    rs[D3DRS_DIFFUSEMATERIALSOURCE as usize] = 1; // D3DMCS_COLOR1
    rs[D3DRS_SPECULARMATERIALSOURCE as usize] = 2; // D3DMCS_COLOR2
    rs[D3DRS_AMBIENTMATERIALSOURCE as usize] = 0; // D3DMCS_MATERIAL
    rs[D3DRS_EMISSIVEMATERIALSOURCE as usize] = 0; // D3DMCS_MATERIAL
    rs[D3DRS_VERTEXBLEND as usize] = D3DVBF_DISABLE;
    rs[D3DRS_CLIPPLANEENABLE as usize] = 0;
    rs[D3DRS_POINTSIZE as usize] = f32::to_bits(1.0);
    rs[D3DRS_POINTSIZE_MIN as usize] = f32::to_bits(1.0);
    rs[D3DRS_POINTSPRITEENABLE as usize] = 0; // FALSE
    rs[D3DRS_POINTSCALEENABLE as usize] = 0; // FALSE
    rs[D3DRS_POINTSCALE_A as usize] = f32::to_bits(1.0);
    rs[D3DRS_POINTSCALE_B as usize] = 0; // 0.0f
    rs[D3DRS_POINTSCALE_C as usize] = 0; // 0.0f
    rs[D3DRS_MULTISAMPLEANTIALIAS as usize] = 1; // TRUE
    rs[D3DRS_MULTISAMPLEMASK as usize] = 0xFFFF_FFFF;
    rs[D3DRS_PATCHEDGESTYLE as usize] = 0; // D3DPATCHEDGE_DISCRETE
    rs[D3DRS_DEBUGMONITORTOKEN as usize] = 0; // D3DDMT_ENABLE
    rs[D3DRS_POINTSIZE_MAX as usize] = f32::to_bits(MAX_POINT_SIZE);
    rs[D3DRS_INDEXEDVERTEXBLENDENABLE as usize] = 0; // FALSE
    rs[D3DRS_COLORWRITEENABLE as usize] = 0x0000_000F; // all channels
    rs[D3DRS_TWEENFACTOR as usize] = 0; // 0.0f
    rs[D3DRS_BLENDOP as usize] = D3DBLENDOP_ADD;
    rs[D3DRS_POSITIONDEGREE as usize] = 3; // D3DDEGREE_CUBIC
    rs[D3DRS_NORMALDEGREE as usize] = 1; // D3DDEGREE_LINEAR
    rs[D3DRS_SCISSORTESTENABLE as usize] = 0; // FALSE
    rs[D3DRS_SLOPESCALEDEPTHBIAS as usize] = 0; // 0.0f
    rs[D3DRS_ANTIALIASEDLINEENABLE as usize] = 0; // FALSE
    rs[D3DRS_MINTESSELLATIONLEVEL as usize] = f32::to_bits(1.0);
    rs[D3DRS_MAXTESSELLATIONLEVEL as usize] = f32::to_bits(1.0);
    rs[D3DRS_ADAPTIVETESS_X as usize] = 0; // 0.0f
    rs[D3DRS_ADAPTIVETESS_Y as usize] = 0; // 0.0f
    rs[D3DRS_ADAPTIVETESS_Z as usize] = f32::to_bits(1.0);
    rs[D3DRS_ADAPTIVETESS_W as usize] = 0; // 0.0f
    rs[D3DRS_ENABLEADAPTIVETESSELLATION as usize] = 0; // FALSE
    rs[D3DRS_TWOSIDEDSTENCILMODE as usize] = 0; // FALSE
    rs[D3DRS_CCW_STENCILFAIL as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_CCW_STENCILZFAIL as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_CCW_STENCILPASS as usize] = D3DSTENCILOP_KEEP;
    rs[D3DRS_CCW_STENCILFUNC as usize] = D3DCMP_ALWAYS;
    rs[D3DRS_COLORWRITEENABLE1 as usize] = 0x0000_000F;
    rs[D3DRS_COLORWRITEENABLE2 as usize] = 0x0000_000F;
    rs[D3DRS_COLORWRITEENABLE3 as usize] = 0x0000_000F;
    rs[D3DRS_BLENDFACTOR as usize] = 0xFFFF_FFFF;
    rs[D3DRS_SRGBWRITEENABLE as usize] = 0;
    rs[D3DRS_DEPTHBIAS as usize] = 0; // 0.0f
    rs[D3DRS_SEPARATEALPHABLENDENABLE as usize] = 0; // FALSE
    rs[D3DRS_SRCBLENDALPHA as usize] = D3DBLEND_ONE;
    rs[D3DRS_DESTBLENDALPHA as usize] = D3DBLEND_ZERO;
    rs[D3DRS_BLENDOPALPHA as usize] = D3DBLENDOP_ADD;

    rs
}

/// Returns an array of D3D9 sampler state defaults per the specification.
#[must_use]
pub const fn sampler_state_defaults() -> [u32; SAMPLER_STATE_COUNT] {
    let mut ss = [0u32; SAMPLER_STATE_COUNT];
    ss[D3DSAMP_ADDRESSU as usize] = D3DTADDRESS_WRAP;
    ss[D3DSAMP_ADDRESSV as usize] = D3DTADDRESS_WRAP;
    ss[D3DSAMP_ADDRESSW as usize] = D3DTADDRESS_WRAP;
    ss[D3DSAMP_MAGFILTER as usize] = D3DTEXF_POINT;
    ss[D3DSAMP_MINFILTER as usize] = D3DTEXF_POINT;
    ss[D3DSAMP_MIPFILTER as usize] = D3DTEXF_NONE;
    ss[D3DSAMP_MAXANISOTROPY as usize] = 1;
    ss[D3DSAMP_MAXMIPLEVEL as usize] = 0;
    ss
}

// ── D3D types ──

#[repr(C)]
pub struct VShaderCaps20 {
    pub caps: u32,
    pub dynamic_flow_control_depth: i32,
    pub num_temps: i32,
    pub static_flow_control_depth: i32,
}

#[repr(C)]
pub struct PShaderCaps20 {
    pub caps: u32,
    pub dynamic_flow_control_depth: i32,
    pub num_temps: i32,
    pub static_flow_control_depth: i32,
    pub num_instruction_slots: i32,
}

#[repr(C)]
pub struct D3DCAPS9 {
    pub device_type: u32,
    pub adapter_ordinal: u32,
    pub caps: u32,
    pub caps2: u32,
    pub caps3: u32,
    pub presentation_intervals: u32,
    pub cursor_caps: u32,
    pub dev_caps: u32,
    pub primitive_misc_caps: u32,
    pub raster_caps: u32,
    pub z_cmp_caps: u32,
    pub src_blend_caps: u32,
    pub dest_blend_caps: u32,
    pub alpha_cmp_caps: u32,
    pub shade_caps: u32,
    pub texture_caps: u32,
    pub texture_filter_caps: u32,
    pub cube_texture_filter_caps: u32,
    pub volume_texture_filter_caps: u32,
    pub texture_address_caps: u32,
    pub volume_texture_address_caps: u32,
    pub line_caps: u32,
    pub max_texture_width: u32,
    pub max_texture_height: u32,
    pub max_volume_extent: u32,
    pub max_texture_repeat: u32,
    pub max_texture_aspect_ratio: u32,
    pub max_anisotropy: u32,
    pub max_vertex_w: f32,
    pub guard_band_left: f32,
    pub guard_band_top: f32,
    pub guard_band_right: f32,
    pub guard_band_bottom: f32,
    pub extents_adjust: f32,
    pub stencil_caps: u32,
    pub fvf_caps: u32,
    pub texture_op_caps: u32,
    pub max_texture_blend_stages: u32,
    pub max_simultaneous_textures: u32,
    pub vertex_processing_caps: u32,
    pub max_active_lights: u32,
    pub max_user_clip_planes: u32,
    pub max_vertex_blend_matrices: u32,
    pub max_vertex_blend_matrix_index: u32,
    pub max_point_size: f32,
    pub max_primitive_count: u32,
    pub max_vertex_index: u32,
    pub max_streams: u32,
    pub max_stream_stride: u32,
    pub vertex_shader_version: u32,
    pub max_vertex_shader_const: u32,
    pub pixel_shader_version: u32,
    pub pixel_shader_1x_max_value: f32,
    pub dev_caps2: u32,
    pub max_npatch_tessellation_level: f32,
    pub reserved5: u32,
    pub master_adapter_ordinal: u32,
    pub adapter_ordinal_in_group: u32,
    pub number_of_adapters_in_group: u32,
    pub decl_types: u32,
    pub num_simultaneous_rts: u32,
    pub stretch_rect_filter_caps: u32,
    pub vs20_caps: VShaderCaps20,
    pub ps20_caps: PShaderCaps20,
    pub vertex_texture_filter_caps: u32,
    pub max_v_shader_instructions_executed: u32,
    pub max_p_shader_instructions_executed: u32,
    pub max_vertex_shader_30_instruction_slots: u32,
    pub max_pixel_shader_30_instruction_slots: u32,
}

// D3DPRESENTFLAG_* bits for D3DPRESENT_PARAMETERS::flags. Only the
// LOCKABLE_BACKBUFFER bit is honoured today (WoW portrait read-back path).
pub const D3DPRESENTFLAG_LOCKABLE_BACKBUFFER: u32 = 0x0000_0001;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DPRESENT_PARAMETERS {
    pub back_buffer_width: u32,
    pub back_buffer_height: u32,
    pub back_buffer_format: u32,
    pub back_buffer_count: u32,
    pub multi_sample_type: u32,
    pub multi_sample_quality: u32,
    pub swap_effect: u32,
    pub device_window: usize, // HWND
    pub windowed: u32,
    pub enable_auto_depth_stencil: u32,
    pub auto_depth_stencil_format: u32,
    pub flags: u32,
    pub full_screen_refresh_rate_in_hz: u32,
    pub presentation_interval: u32,
}

// ── IDirect3DDevice9 vtable ──

#[repr(C)]
pub struct IDirect3DDevice9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DDevice9
    pub test_cooperative_level: unsafe extern "system" fn(*mut c_void) -> i32,
    pub get_available_texture_mem: unsafe extern "system" fn(*mut c_void) -> u32,
    pub evict_managed_resources: unsafe extern "system" fn(*mut c_void) -> i32,
    pub get_direct3d: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_device_caps: unsafe extern "system" fn(*mut c_void, *mut D3DCAPS9) -> i32,
    pub get_display_mode: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub get_creation_parameters: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub set_cursor_properties: unsafe extern "system" fn(*mut c_void, u32, u32, *mut c_void) -> i32,
    pub set_cursor_position: unsafe extern "system" fn(*mut c_void, i32, i32, u32),
    pub show_cursor: unsafe extern "system" fn(*mut c_void, i32) -> i32,
    pub create_additional_swap_chain:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut *mut c_void) -> i32,
    pub get_swap_chain: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub get_number_of_swap_chains: unsafe extern "system" fn(*mut c_void) -> u32,
    pub reset: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub present: unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        *const c_void,
        *mut c_void,
        *const c_void,
    ) -> i32,
    pub get_back_buffer:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, *mut *mut c_void) -> i32,
    pub get_raster_status: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub set_dialog_box_mode: unsafe extern "system" fn(*mut c_void, i32) -> i32,
    pub set_gamma_ramp: unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void),
    pub get_gamma_ramp: unsafe extern "system" fn(*mut c_void, u32, *mut c_void),
    pub create_texture: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_volume_texture: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_cube_texture: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_vertex_buffer: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_index_buffer: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_render_target: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        i32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub create_depth_stencil_surface: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        i32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub update_surface: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        *mut c_void,
        *const c_void,
    ) -> i32,
    pub update_texture: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    pub get_render_target_data:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    pub get_front_buffer_data: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub stretch_rect: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        *mut c_void,
        *const c_void,
        u32,
    ) -> i32,
    pub color_fill: unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, u32) -> i32,
    pub create_offscreen_plain_surface: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *mut *mut c_void,
        *mut c_void,
    ) -> i32,
    pub set_render_target: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub get_render_target: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub set_depth_stencil_surface: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_depth_stencil_surface: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub begin_scene: unsafe extern "system" fn(*mut c_void) -> i32,
    pub end_scene: unsafe extern "system" fn(*mut c_void) -> i32,
    pub clear:
        unsafe extern "system" fn(*mut c_void, u32, *const c_void, u32, u32, f32, u32) -> i32,
    pub set_transform: unsafe extern "system" fn(*mut c_void, u32, *const c_void) -> i32,
    pub get_transform: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub multiply_transform: unsafe extern "system" fn(*mut c_void, u32, *const c_void) -> i32,
    pub set_viewport: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
    pub get_viewport: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub set_material: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
    pub get_material: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub set_light: unsafe extern "system" fn(*mut c_void, u32, *const c_void) -> i32,
    pub get_light: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub light_enable: unsafe extern "system" fn(*mut c_void, u32, i32) -> i32,
    pub get_light_enable: unsafe extern "system" fn(*mut c_void, u32, *mut i32) -> i32,
    pub set_clip_plane: unsafe extern "system" fn(*mut c_void, u32, *const f32) -> i32,
    pub get_clip_plane: unsafe extern "system" fn(*mut c_void, u32, *mut f32) -> i32,
    pub set_render_state: unsafe extern "system" fn(*mut c_void, u32, u32) -> i32,
    pub get_render_state: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    pub create_state_block: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub begin_state_block: unsafe extern "system" fn(*mut c_void) -> i32,
    pub end_state_block: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_clip_status: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
    pub get_clip_status: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_texture: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub set_texture: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub get_texture_stage_state: unsafe extern "system" fn(*mut c_void, u32, u32, *mut u32) -> i32,
    pub set_texture_stage_state: unsafe extern "system" fn(*mut c_void, u32, u32, u32) -> i32,
    pub get_sampler_state: unsafe extern "system" fn(*mut c_void, u32, u32, *mut u32) -> i32,
    pub set_sampler_state: unsafe extern "system" fn(*mut c_void, u32, u32, u32) -> i32,
    pub validate_device: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    pub set_palette_entries: unsafe extern "system" fn(*mut c_void, u32, *const c_void) -> i32,
    pub get_palette_entries: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub set_current_texture_palette: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_current_texture_palette: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    pub set_scissor_rect: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
    pub get_scissor_rect: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub set_software_vertex_processing: unsafe extern "system" fn(*mut c_void, i32) -> i32,
    pub get_software_vertex_processing: unsafe extern "system" fn(*mut c_void) -> i32,
    pub set_npatch_mode: unsafe extern "system" fn(*mut c_void, f32) -> i32,
    pub get_npatch_mode: unsafe extern "system" fn(*mut c_void) -> f32,
    pub draw_primitive: unsafe extern "system" fn(*mut c_void, u32, u32, u32) -> i32,
    pub draw_indexed_primitive:
        unsafe extern "system" fn(*mut c_void, u32, i32, u32, u32, u32, u32) -> i32,
    pub draw_primitive_up:
        unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void, u32) -> i32,
    pub draw_indexed_primitive_up: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        *const c_void,
        u32,
        *const c_void,
        u32,
    ) -> i32,
    pub process_vertices:
        unsafe extern "system" fn(*mut c_void, u32, u32, u32, *mut c_void, *mut c_void, u32) -> i32,
    pub create_vertex_declaration:
        unsafe extern "system" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32,
    pub set_vertex_declaration: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_vertex_declaration: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_fvf: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_fvf: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32,
    pub create_vertex_shader:
        unsafe extern "system" fn(*mut c_void, *const u32, *mut *mut c_void) -> i32,
    pub set_vertex_shader: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_vertex_shader: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_vertex_shader_constant_f:
        unsafe extern "system" fn(*mut c_void, u32, *const f32, u32) -> i32,
    pub get_vertex_shader_constant_f:
        unsafe extern "system" fn(*mut c_void, u32, *mut f32, u32) -> i32,
    pub set_vertex_shader_constant_i:
        unsafe extern "system" fn(*mut c_void, u32, *const i32, u32) -> i32,
    pub get_vertex_shader_constant_i:
        unsafe extern "system" fn(*mut c_void, u32, *mut i32, u32) -> i32,
    pub set_vertex_shader_constant_b:
        unsafe extern "system" fn(*mut c_void, u32, *const i32, u32) -> i32,
    pub get_vertex_shader_constant_b:
        unsafe extern "system" fn(*mut c_void, u32, *mut i32, u32) -> i32,
    pub set_stream_source:
        unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32, u32) -> i32,
    pub get_stream_source:
        unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void, *mut u32, *mut u32) -> i32,
    pub set_stream_source_freq: unsafe extern "system" fn(*mut c_void, u32, u32) -> i32,
    pub get_stream_source_freq: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    pub set_indices: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_indices: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub create_pixel_shader:
        unsafe extern "system" fn(*mut c_void, *const u32, *mut *mut c_void) -> i32,
    pub set_pixel_shader: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_pixel_shader: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_pixel_shader_constant_f:
        unsafe extern "system" fn(*mut c_void, u32, *const f32, u32) -> i32,
    pub get_pixel_shader_constant_f:
        unsafe extern "system" fn(*mut c_void, u32, *mut f32, u32) -> i32,
    pub set_pixel_shader_constant_i:
        unsafe extern "system" fn(*mut c_void, u32, *const i32, u32) -> i32,
    pub get_pixel_shader_constant_i:
        unsafe extern "system" fn(*mut c_void, u32, *mut i32, u32) -> i32,
    pub set_pixel_shader_constant_b:
        unsafe extern "system" fn(*mut c_void, u32, *const i32, u32) -> i32,
    pub get_pixel_shader_constant_b:
        unsafe extern "system" fn(*mut c_void, u32, *mut i32, u32) -> i32,
    pub draw_rect_patch:
        unsafe extern "system" fn(*mut c_void, u32, *const f32, *const c_void) -> i32,
    pub draw_tri_patch:
        unsafe extern "system" fn(*mut c_void, u32, *const f32, *const c_void) -> i32,
    pub delete_patch: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub create_query: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
}

// ── D3DDISPLAYMODE ──

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DDISPLAYMODE {
    pub width: u32,
    pub height: u32,
    pub refresh_rate: u32,
    pub format: u32,
}

// ── D3DDEVICE_CREATION_PARAMETERS ──

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DDEVICE_CREATION_PARAMETERS {
    pub adapter_ordinal: u32,
    pub device_type: u32,
    pub focus_window: usize, // HWND
    pub behavior_flags: u32,
}

// ── D3DLOCKED_RECT ──

#[repr(C)]
pub struct D3DLOCKED_RECT {
    pub pitch: i32,
    pub bits: *mut c_void,
}

// ── D3DLOCKED_BOX (volume / 3D texture lock) ──

#[repr(C)]
pub struct D3DLOCKED_BOX {
    pub row_pitch: i32,
    pub slice_pitch: i32,
    pub bits: *mut c_void,
}

// ── D3DSURFACE_DESC ──

#[repr(C)]
pub struct D3DSURFACE_DESC {
    pub format: u32,
    pub resource_type: u32,
    pub usage: u32,
    pub pool: u32,
    pub multi_sample_type: u32,
    pub multi_sample_quality: u32,
    pub width: u32,
    pub height: u32,
}

// ── IDirect3DTexture9 vtable ──

#[repr(C)]
pub struct IDirect3DTexture9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DResource9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub set_priority: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_priority: unsafe extern "system" fn(*mut c_void) -> u32,
    pub pre_load: unsafe extern "system" fn(*mut c_void),
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DBaseTexture9
    pub set_lod: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_lod: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_level_count: unsafe extern "system" fn(*mut c_void) -> u32,
    pub set_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void) -> u32,
    pub generate_mip_sub_levels: unsafe extern "system" fn(*mut c_void),
    // IDirect3DTexture9
    pub get_level_desc: unsafe extern "system" fn(*mut c_void, u32, *mut D3DSURFACE_DESC) -> i32,
    pub get_surface_level: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub lock_rect:
        unsafe extern "system" fn(*mut c_void, u32, *mut D3DLOCKED_RECT, *const c_void, u32) -> i32,
    pub unlock_rect: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub add_dirty_rect: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
}

/// `IDirect3DVolumeTexture9` vtable.
///
/// Identical to `IDirect3DTexture9Vtbl` through the `IDirect3DBaseTexture9`
/// block (the first 18 slots — same thunks reused), then the 3D-specific tail.
/// `D3DVOLUME_DESC` / `D3DBOX` parameters are typed as `*mut c_void` /
/// `*const c_void`: the ABI is a pointer either way and the methods that take
/// them are stubbed today.
#[repr(C)]
pub struct IDirect3DVolumeTexture9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DResource9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub set_priority: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_priority: unsafe extern "system" fn(*mut c_void) -> u32,
    pub pre_load: unsafe extern "system" fn(*mut c_void),
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DBaseTexture9
    pub set_lod: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_lod: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_level_count: unsafe extern "system" fn(*mut c_void) -> u32,
    pub set_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void) -> u32,
    pub generate_mip_sub_levels: unsafe extern "system" fn(*mut c_void),
    // IDirect3DVolumeTexture9
    pub get_level_desc: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub get_volume_level: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
    pub lock_box:
        unsafe extern "system" fn(*mut c_void, u32, *mut D3DLOCKED_BOX, *const c_void, u32) -> i32,
    pub unlock_box: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub add_dirty_box: unsafe extern "system" fn(*mut c_void, *const c_void) -> i32,
}

// ── D3DVOLUME_DESC ──

#[repr(C)]
pub struct D3DVOLUME_DESC {
    pub format: u32,
    pub resource_type: u32,
    pub usage: u32,
    pub pool: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

// ── IDirect3DVolume9 vtable ──
/// A single level of a volume texture (`GetVolumeLevel`).
///
/// Unlike the texture interfaces, `IDirect3DVolume9` is NOT an
/// `IDirect3DResource9` — it has no priority/type/preload block — so the
/// vtable is just `IUnknown` plus the eight volume methods.
#[repr(C)]
pub struct IDirect3DVolume9Vtbl {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub get_container: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub get_desc: unsafe extern "system" fn(*mut c_void, *mut D3DVOLUME_DESC) -> i32,
    pub lock_box:
        unsafe extern "system" fn(*mut c_void, *mut D3DLOCKED_BOX, *const c_void, u32) -> i32,
    pub unlock_box: unsafe extern "system" fn(*mut c_void) -> i32,
}

// ── IDirect3DShaderValidator9 vtable ──
/// Returned by the `Direct3DShaderValidatorCreate9` export.
///
/// An undocumented interface used by the conformance suite + shader tools
/// (fxc); games never touch it. `IUnknown` plus `Begin` / `Instruction` /
/// `End`. `Begin`'s callback and `Instruction`'s `file`/`tokens` are opaque
/// pointers to the stub.
#[repr(C)]
pub struct IDirect3DShaderValidator9Vtbl {
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    pub begin: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, usize) -> i32,
    pub instruction:
        unsafe extern "system" fn(*mut c_void, *const c_void, i32, *const u32, u32) -> i32,
    pub end: unsafe extern "system" fn(*mut c_void) -> i32,
}

// ── IDirect3DCubeTexture9 vtable ──

#[repr(C)]
pub struct IDirect3DCubeTexture9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DResource9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub set_priority: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_priority: unsafe extern "system" fn(*mut c_void) -> u32,
    pub pre_load: unsafe extern "system" fn(*mut c_void),
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DBaseTexture9
    pub set_lod: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_lod: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_level_count: unsafe extern "system" fn(*mut c_void) -> u32,
    pub set_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_auto_gen_filter_type: unsafe extern "system" fn(*mut c_void) -> u32,
    pub generate_mip_sub_levels: unsafe extern "system" fn(*mut c_void),
    // IDirect3DCubeTexture9
    pub get_level_desc: unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32,
    pub get_cube_map_surface:
        unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32,
    pub lock_rect: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        *mut D3DLOCKED_RECT,
        *const c_void,
        u32,
    ) -> i32,
    pub unlock_rect: unsafe extern "system" fn(*mut c_void, u32, u32) -> i32,
    pub add_dirty_rect: unsafe extern "system" fn(*mut c_void, u32, *const c_void) -> i32,
}

// ── IDirect3DSurface9 vtable ──

#[repr(C)]
pub struct IDirect3DSurface9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DResource9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub set_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *const c_void, u32, u32) -> i32,
    pub get_private_data:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut c_void, *mut u32) -> i32,
    pub free_private_data: unsafe extern "system" fn(*mut c_void, *const Guid) -> i32,
    pub set_priority: unsafe extern "system" fn(*mut c_void, u32) -> u32,
    pub get_priority: unsafe extern "system" fn(*mut c_void) -> u32,
    pub pre_load: unsafe extern "system" fn(*mut c_void),
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DSurface9
    pub get_container: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub get_desc: unsafe extern "system" fn(*mut c_void, *mut D3DSURFACE_DESC) -> i32,
    pub lock_rect:
        unsafe extern "system" fn(*mut c_void, *mut D3DLOCKED_RECT, *const c_void, u32) -> i32,
    pub unlock_rect: unsafe extern "system" fn(*mut c_void) -> i32,
    pub get_dc: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub release_dc: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
}

// ── IDirect3DVertexShader9 vtable ──

#[repr(C)]
pub struct IDirect3DVertexShader9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DVertexShader9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_function: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u32) -> i32,
}

// ── IDirect3DPixelShader9 vtable ──

#[repr(C)]
pub struct IDirect3DPixelShader9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DPixelShader9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_function: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u32) -> i32,
}

// ── IDirect3DStateBlock9 vtable ──

#[repr(C)]
pub struct IDirect3DStateBlock9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DStateBlock9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub capture: unsafe extern "system" fn(*mut c_void) -> i32,
    pub apply: unsafe extern "system" fn(*mut c_void) -> i32,
}

// ── IDirect3DQuery9 vtable ──

#[repr(C)]
pub struct IDirect3DQuery9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DQuery9
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_type: unsafe extern "system" fn(*mut c_void) -> u32,
    pub get_data_size: unsafe extern "system" fn(*mut c_void) -> u32,
    pub issue: unsafe extern "system" fn(*mut c_void, u32) -> i32,
    pub get_data: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32) -> i32,
}

// ── IDirect3DSwapChain9 vtable ──

#[repr(C)]
pub struct IDirect3DSwapChain9Vtbl {
    // IUnknown
    pub query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32,
    pub add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    pub release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDirect3DSwapChain9. `present` takes the source/dest RECTs, the
    // hDestWindowOverride HWND (pointer-sized), the dirty RGNDATA, and flags.
    pub present: unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        *const c_void,
        usize,
        *const c_void,
        u32,
    ) -> i32,
    pub get_front_buffer_data: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_back_buffer: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32,
    pub get_raster_status: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_display_mode: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
    pub get_device: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32,
    pub get_present_parameters: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32,
}

// ── D3D9 query constants ──

pub const D3DQUERYTYPE_EVENT: u32 = 8;
pub const D3DQUERYTYPE_OCCLUSION: u32 = 9;
pub const D3DQUERYTYPE_TIMESTAMP: u32 = 10;

pub const D3DISSUE_BEGIN: u32 = 1 << 1;
pub const D3DISSUE_END: u32 = 1 << 0;

// `GetData` flags. `FLUSH` asks for the freshest result.
pub const D3DGETDATA_FLUSH: u32 = 0x0000_0001;

/// `D3DRECT` for `SetScissorRect` / `GetScissorRect`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DRECT {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

/// `D3DBOX` for `IDirect3DVolume(Texture)9::LockBox`.
///
/// A half-open `[left,right)` × `[top,bottom)` × `[front,back)` region in texel
/// coordinates (all unsigned).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct D3DBOX {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub front: u32,
    pub back: u32,
}

/// `D3DVIEWPORT9` for `SetViewport` / `GetViewport`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3DVIEWPORT9 {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub min_z: f32,
    pub max_z: f32,
}

#[cfg(test)]
mod tests {
    use super::{
        D3DSBT_ALL, D3DSBT_PIXELSTATE, D3DSBT_VERTEXSTATE, RENDER_STATE_COUNT, SAMPLER_STATE_COUNT,
        StateBlockType,
    };
    use crate::TEXTURE_STAGE_STATE_COUNT;

    fn bound(n: usize) -> u32 {
        u32::try_from(n).expect("state count fits u32")
    }
    fn count_render(ty: StateBlockType) -> usize {
        (0..bound(RENDER_STATE_COUNT))
            .filter(|&i| ty.includes_render_state(i))
            .count()
    }
    fn count_sampler(ty: StateBlockType) -> usize {
        (0..bound(SAMPLER_STATE_COUNT))
            .filter(|&i| ty.includes_sampler_state(i))
            .count()
    }
    fn count_tss(ty: StateBlockType) -> usize {
        (0..bound(TEXTURE_STAGE_STATE_COUNT))
            .filter(|&i| ty.includes_tss(i))
            .count()
    }

    #[test]
    fn from_d3dsbt_maps_known_types() {
        assert_eq!(
            StateBlockType::from_d3dsbt(D3DSBT_ALL),
            Some(StateBlockType::All)
        );
        assert_eq!(
            StateBlockType::from_d3dsbt(D3DSBT_VERTEXSTATE),
            Some(StateBlockType::Vertex)
        );
        assert_eq!(
            StateBlockType::from_d3dsbt(D3DSBT_PIXELSTATE),
            Some(StateBlockType::Pixel)
        );
        assert_eq!(StateBlockType::from_d3dsbt(0), None);
        assert_eq!(StateBlockType::from_d3dsbt(99), None);
    }

    /// Membership counts must match the D3D9 state-block classification.
    ///
    /// `D3DSBT_VERTEXSTATE` / `D3DSBT_PIXELSTATE`: 45 vertex render states and
    /// 60 pixel render states.
    #[test]
    fn classification_set_sizes_match_d3d9() {
        assert_eq!(
            count_render(StateBlockType::Vertex),
            45,
            "vertex render states"
        );
        assert_eq!(
            count_render(StateBlockType::Pixel),
            60,
            "pixel render states"
        );
        assert_eq!(
            count_sampler(StateBlockType::Vertex),
            1,
            "vertex sampler states"
        );
        assert_eq!(
            count_sampler(StateBlockType::Pixel),
            12,
            "pixel sampler states"
        );
        assert_eq!(
            count_tss(StateBlockType::Vertex),
            2,
            "vertex texture-stage states"
        );
        assert_eq!(
            count_tss(StateBlockType::Pixel),
            17,
            "pixel texture-stage states"
        );
    }

    #[test]
    fn all_includes_everything() {
        for i in 0..bound(RENDER_STATE_COUNT) {
            assert!(StateBlockType::All.includes_render_state(i));
        }
        for i in 0..bound(SAMPLER_STATE_COUNT) {
            assert!(StateBlockType::All.includes_sampler_state(i));
        }
        for i in 0..bound(TEXTURE_STAGE_STATE_COUNT) {
            assert!(StateBlockType::All.includes_tss(i));
        }
    }

    #[test]
    fn shademode_and_fog_scalars_are_in_both_render_sets() {
        for rs in [
            super::D3DRS_SHADEMODE,
            super::D3DRS_FOGSTART,
            super::D3DRS_FOGEND,
            super::D3DRS_FOGDENSITY,
        ] {
            assert!(StateBlockType::Vertex.includes_render_state(rs));
            assert!(StateBlockType::Pixel.includes_render_state(rs));
        }
        // A pixel-only / vertex-only spot check.
        assert!(StateBlockType::Pixel.includes_render_state(super::D3DRS_ALPHABLENDENABLE));
        assert!(!StateBlockType::Vertex.includes_render_state(super::D3DRS_ALPHABLENDENABLE));
        assert!(StateBlockType::Vertex.includes_render_state(super::D3DRS_LIGHTING));
        assert!(!StateBlockType::Pixel.includes_render_state(super::D3DRS_LIGHTING));
        // All 16 texture-wrap states are pixel state.
        for rs in super::D3DRS_WRAP0..=super::D3DRS_WRAP7 {
            assert!(StateBlockType::Pixel.includes_render_state(rs));
        }
        for rs in super::D3DRS_WRAP8..=super::D3DRS_WRAP15 {
            assert!(StateBlockType::Pixel.includes_render_state(rs));
        }
    }
}
