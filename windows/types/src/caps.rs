//! `D3DCAPS9` bit sets and the numeric limits the caps set advertises.
//!
//! Every bitmask field of `D3DCAPS9` gets one `bitflags!` type here, holding
//! the complete set of bits `d3d9caps.h` defines for that field. Members keep
//! the spec suffix (`RasterCaps::SLOPESCALEDEPTHBIAS` ↔
//! `D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS`) so they stay greppable against the
//! header.
//!
//! No type carries an unnamed `_ = !0` member: `Flags::all()` must mean
//! "every bit the spec defines", which is what the caps-advertising diagnostic
//! in `mtld3d-core` ORs in. Which subset is advertised on the default path is
//! renderer policy and lives with that code, not here.

// ── Numeric limits named by the D3D9 ABI ──

/// Vertex streams a D3D9 device addresses (`SetStreamSource` stream index range).
///
/// Advertised as `D3DCAPS9::MaxStreams` and used by the binding layer to size
/// its stream-slot array, so the advertised cap and the slot count cannot
/// drift. Stream `n` is fetched from Metal vertex buffer slot `n`.
pub const MAX_STREAMS: u32 = 16;

/// Render targets a D3D9 device can bind simultaneously (`D3D_MAX_SIMULTANEOUS_RENDERTARGETS`).
pub const D3D_MAX_SIMULTANEOUS_RENDERTARGETS: u32 = 4;

/// `D3DVS30_INSTRUCTIONSLOTS_MAX`: the SM3 spec ceiling on vertex-shader instruction slots.
pub const D3DVS30_INSTRUCTIONSLOTS_MAX: u32 = 32768;

/// `D3DPS30_INSTRUCTIONSLOTS_MAX`: the SM3 spec ceiling on pixel-shader instruction slots.
pub const D3DPS30_INSTRUCTIONSLOTS_MAX: u32 = 32768;

/// `D3DVS20_MAX_DYNAMICFLOWCONTROLDEPTH`: ceiling of the VS20 dynamic flow-control depth.
pub const D3DVS20_MAX_DYNAMICFLOWCONTROLDEPTH: i32 = 24;

/// `D3DVS20_MAX_NUMTEMPS`: ceiling of the VS20 temporary-register count.
pub const D3DVS20_MAX_NUMTEMPS: i32 = 32;

/// `D3DVS20_MAX_STATICFLOWCONTROLDEPTH`: ceiling of the VS20 static flow-control depth.
pub const D3DVS20_MAX_STATICFLOWCONTROLDEPTH: i32 = 4;

/// `D3DPS20_MAX_DYNAMICFLOWCONTROLDEPTH`: ceiling of the PS20 dynamic flow-control depth.
pub const D3DPS20_MAX_DYNAMICFLOWCONTROLDEPTH: i32 = 24;

/// `D3DPS20_MAX_NUMTEMPS`: ceiling of the PS20 temporary-register count.
pub const D3DPS20_MAX_NUMTEMPS: i32 = 32;

/// `D3DPS20_MAX_STATICFLOWCONTROLDEPTH`: ceiling of the PS20 static flow-control depth.
pub const D3DPS20_MAX_STATICFLOWCONTROLDEPTH: i32 = 4;

/// `D3DPS20_MAX_NUMINSTRUCTIONSLOTS`: ceiling of the PS20 instruction-slot count.
pub const D3DPS20_MAX_NUMINSTRUCTIONSLOTS: i32 = 512;

/// Float constant registers an SM3 vertex shader can address (`c0`..`c255`).
pub const MAX_VERTEX_SHADER_CONST: u32 = 256;

/// `D3DVS_VERSION(major, minor)`: the value reported as `D3DCAPS9::VertexShaderVersion`.
///
/// D3D9 packs a shader version as `0xFFFE_<major><minor>` for vertex shaders;
/// the runtime inspects it to decide which bytecode versions the game may
/// compile against.
#[must_use]
pub const fn d3dvs_version(major: u32, minor: u32) -> u32 {
    0xFFFE_0000 | (major << 8) | minor
}

/// `D3DPS_VERSION(major, minor)`: the value reported as `D3DCAPS9::PixelShaderVersion`.
///
/// Same encoding as [`d3dvs_version`] with the `0xFFFF` marker that
/// distinguishes a pixel-shader version.
#[must_use]
pub const fn d3dps_version(major: u32, minor: u32) -> u32 {
    0xFFFF_0000 | (major << 8) | minor
}

// ── D3DCAPS9 bitmask fields ──

bitflags::bitflags! {
    /// `D3DVS20CAPS_*` bits (`D3DVSHADERCAPS2_0::Caps`, the `VS20Caps` member).
    pub struct Vs20Caps: u32 {
        const PREDICATION = 0x0000_0001;
    }
}

bitflags::bitflags! {
    /// `D3DPS20CAPS_*` bits (`D3DPSHADERCAPS2_0::Caps`, the `PS20Caps` member).
    pub struct Ps20Caps: u32 {
        const ARBITRARYSWIZZLE = 0x0000_0001;
        const GRADIENTINSTRUCTIONS = 0x0000_0002;
        const PREDICATION = 0x0000_0004;
        const NODEPENDENTREADLIMIT = 0x0000_0008;
        const NOTEXINSTRUCTIONLIMIT = 0x0000_0010;
    }
}

bitflags::bitflags! {
    /// `D3DCAPS2_*` bits (`D3DCAPS9::Caps2`).
    ///
    /// `D3DCAPS2_RESERVED` (0x0200_0000) is deliberately absent: it names a
    /// reserved bit, never a capability, and would otherwise land in
    /// `all()`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Caps2: u32 {
        const FULLSCREENGAMMA = 0x0002_0000;
        const CANCALIBRATEGAMMA = 0x0010_0000;
        const CANMANAGERESOURCE = 0x1000_0000;
        const DYNAMICTEXTURES = 0x2000_0000;
        const CANAUTOGENMIPMAP = 0x4000_0000;
        const CANSHARERESOURCE = 0x8000_0000;
    }
}

bitflags::bitflags! {
    /// `D3DCAPS3_*` bits (`D3DCAPS9::Caps3`).
    ///
    /// `D3DCAPS3_RESERVED` (0x8000_001F) is deliberately absent for the same
    /// reason as [`Caps2`]'s reserved bit.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Caps3: u32 {
        const ALPHA_FULLSCREEN_FLIP_OR_DISCARD = 0x0000_0020;
        const LINEAR_TO_SRGB_PRESENTATION = 0x0000_0080;
        const COPY_TO_VIDMEM = 0x0000_0100;
        const COPY_TO_SYSTEMMEM = 0x0000_0200;
        const DXVAHD = 0x0000_0400;
        const DXVAHD_LIMITED = 0x0000_0800;
    }
}

bitflags::bitflags! {
    /// `D3DCURSORCAPS_*` bits (`D3DCAPS9::CursorCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CursorCaps: u32 {
        /// Full-colour hardware cursor at any display resolution.
        const COLOR = 0x0000_0001;
        /// Hardware cursor only below 400 scan lines.
        const LOWRES = 0x0000_0002;
    }
}

bitflags::bitflags! {
    /// `D3DDEVCAPS_*` bits (`D3DCAPS9::DevCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DevCaps: u32 {
        const EXECUTESYSTEMMEMORY = 0x0000_0010;
        const EXECUTEVIDEOMEMORY = 0x0000_0020;
        const TLVERTEXSYSTEMMEMORY = 0x0000_0040;
        const TLVERTEXVIDEOMEMORY = 0x0000_0080;
        const TEXTURESYSTEMMEMORY = 0x0000_0100;
        const TEXTUREVIDEOMEMORY = 0x0000_0200;
        const DRAWPRIMTLVERTEX = 0x0000_0400;
        const CANRENDERAFTERFLIP = 0x0000_0800;
        const TEXTURENONLOCALVIDMEM = 0x0000_1000;
        const DRAWPRIMITIVES2 = 0x0000_2000;
        const SEPARATETEXTUREMEMORIES = 0x0000_4000;
        const DRAWPRIMITIVES2EX = 0x0000_8000;
        const HWTRANSFORMANDLIGHT = 0x0001_0000;
        const CANBLTSYSTONONLOCAL = 0x0002_0000;
        /// The device rasterizes through hardware acceleration.
        const HWRASTERIZATION = 0x0008_0000;
        const PUREDEVICE = 0x0010_0000;
        const QUINTICRTPATCHES = 0x0020_0000;
        const RTPATCHES = 0x0040_0000;
        const RTPATCHHANDLEZERO = 0x0080_0000;
        const NPATCHES = 0x0100_0000;
    }
}

bitflags::bitflags! {
    /// `D3DDEVCAPS2_*` bits (`D3DCAPS9::DevCaps2`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DevCaps2: u32 {
        const STREAMOFFSET = 0x0000_0001;
        const DMAPNPATCH = 0x0000_0002;
        const ADAPTIVETESSRTPATCH = 0x0000_0004;
        const ADAPTIVETESSNPATCH = 0x0000_0008;
        /// `StretchRect` from a texture-level surface into a render target.
        const CAN_STRETCHRECT_FROM_TEXTURES = 0x0000_0010;
        const PRESAMPLEDDMAPNPATCH = 0x0000_0020;
        const VERTEXELEMENTSCANSHARESTREAMOFFSET = 0x0000_0040;
    }
}

bitflags::bitflags! {
    /// `D3DPMISCCAPS_*` bits (`D3DCAPS9::PrimitiveMiscCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PrimitiveMiscCaps: u32 {
        const MASKZ = 0x0000_0002;
        const LINEPATTERNREP = 0x0000_0004;
        const CULLNONE = 0x0000_0010;
        const CULLCW = 0x0000_0020;
        const CULLCCW = 0x0000_0040;
        const COLORWRITEENABLE = 0x0000_0080;
        const CLIPPLANESCALEDPOINTS = 0x0000_0100;
        const CLIPTLVERTS = 0x0000_0200;
        const TSSARGTEMP = 0x0000_0400;
        const BLENDOP = 0x0000_0800;
        const NULLREFERENCE = 0x0000_1000;
        const INDEPENDENTWRITEMASKS = 0x0000_4000;
        const PERSTAGECONSTANT = 0x0000_8000;
        const FOGANDSPECULARALPHA = 0x0001_0000;
        const SEPARATEALPHABLEND = 0x0002_0000;
        const MRTINDEPENDENTBITDEPTHS = 0x0004_0000;
        const MRTPOSTPIXELSHADERBLENDING = 0x0008_0000;
        const FOGVERTEXCLAMPED = 0x0010_0000;
        const POSTBLENDSRGBCONVERT = 0x0020_0000;
    }
}

bitflags::bitflags! {
    /// `D3DPRASTERCAPS_*` bits (`D3DCAPS9::RasterCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct RasterCaps: u32 {
        const DITHER = 0x0000_0001;
        const ZTEST = 0x0000_0010;
        const FOGVERTEX = 0x0000_0080;
        const FOGTABLE = 0x0000_0100;
        const MIPMAPLODBIAS = 0x0000_2000;
        const ZBUFFERLESSHSR = 0x0000_8000;
        const FOGRANGE = 0x0001_0000;
        const ANISOTROPY = 0x0002_0000;
        const WBUFFER = 0x0004_0000;
        const WFOG = 0x0010_0000;
        const ZFOG = 0x0020_0000;
        const COLORPERSPECTIVE = 0x0040_0000;
        const SCISSORTEST = 0x0100_0000;
        const SLOPESCALEDEPTHBIAS = 0x0200_0000;
        const DEPTHBIAS = 0x0400_0000;
        const MULTISAMPLE_TOGGLE = 0x0800_0000;
    }
}

bitflags::bitflags! {
    /// `D3DPCMPCAPS_*` bits (`D3DCAPS9::ZCmpCaps` and `AlphaCmpCaps`).
    ///
    /// One bit per `D3DCMP_*` comparison function, in `D3DCMP` order, so
    /// `all()` is the full set of eight.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CmpCaps: u32 {
        const NEVER = 0x0000_0001;
        const LESS = 0x0000_0002;
        const EQUAL = 0x0000_0004;
        const LESSEQUAL = 0x0000_0008;
        const GREATER = 0x0000_0010;
        const NOTEQUAL = 0x0000_0020;
        const GREATEREQUAL = 0x0000_0040;
        const ALWAYS = 0x0000_0080;
    }
}

bitflags::bitflags! {
    /// `D3DPBLENDCAPS_*` bits (`D3DCAPS9::SrcBlendCaps` and `DestBlendCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BlendCaps: u32 {
        const ZERO = 0x0000_0001;
        const ONE = 0x0000_0002;
        const SRCCOLOR = 0x0000_0004;
        const INVSRCCOLOR = 0x0000_0008;
        const SRCALPHA = 0x0000_0010;
        const INVSRCALPHA = 0x0000_0020;
        const DESTALPHA = 0x0000_0040;
        const INVDESTALPHA = 0x0000_0080;
        const DESTCOLOR = 0x0000_0100;
        const INVDESTCOLOR = 0x0000_0200;
        const SRCALPHASAT = 0x0000_0400;
        const BOTHSRCALPHA = 0x0000_0800;
        const BOTHINVSRCALPHA = 0x0000_1000;
        const BLENDFACTOR = 0x0000_2000;
        /// Dual-source blend factor, SM3 only.
        const SRCCOLOR2 = 0x0000_4000;
        /// Dual-source blend factor, SM3 only.
        const INVSRCCOLOR2 = 0x0000_8000;
    }
}

bitflags::bitflags! {
    /// `D3DPSHADECAPS_*` bits (`D3DCAPS9::ShadeCaps`).
    ///
    /// The `*FLAT*` variants are DX7-era and not part of the D3D9 spec, so
    /// they have no members here.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ShadeCaps: u32 {
        const COLORGOURAUDRGB = 0x0000_0008;
        const SPECULARGOURAUDRGB = 0x0000_0200;
        const ALPHAGOURAUDBLEND = 0x0000_4000;
        const FOGGOURAUD = 0x0008_0000;
    }
}

bitflags::bitflags! {
    /// `D3DPTEXTURECAPS_*` bits (`D3DCAPS9::TextureCaps`).
    ///
    /// The field mixes capability bits with *restriction* bits (see
    /// [`TextureCaps::RESTRICTIONS`]), which is why it can never be advertised
    /// as a plain `all()`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct TextureCaps: u32 {
        const PERSPECTIVE = 0x0000_0001;
        /// Restriction: texture dimensions must be powers of two.
        const POW2 = 0x0000_0002;
        const ALPHA = 0x0000_0004;
        /// Restriction: textures must be square.
        const SQUAREONLY = 0x0000_0020;
        const TEXREPEATNOTSCALEDBYSIZE = 0x0000_0040;
        const ALPHAPALETTE = 0x0000_0080;
        /// Restriction: non-power-of-two support comes with conditions.
        ///
        /// Only meaningful alongside [`TextureCaps::POW2`]; on its own it
        /// tells a game nothing it can act on.
        const NONPOW2CONDITIONAL = 0x0000_0100;
        const PROJECTED = 0x0000_0400;
        const CUBEMAP = 0x0000_0800;
        const VOLUMEMAP = 0x0000_2000;
        const MIPMAP = 0x0000_4000;
        const MIPVOLUMEMAP = 0x0000_8000;
        const MIPCUBEMAP = 0x0001_0000;
        /// Restriction: cube-map edge length must be a power of two.
        const CUBEMAP_POW2 = 0x0002_0000;
        /// Restriction: volume-texture dimensions must be powers of two.
        const VOLUMEMAP_POW2 = 0x0004_0000;
        const NOPROJECTEDBUMPENV = 0x0020_0000;
    }
}

impl TextureCaps {
    /// Bits that state a *limitation* on texture creation, not a capability.
    ///
    /// A game reads each of these as "you must", so a set bit removes
    /// freedom the renderer actually has. They are the one group that must
    /// stay clear even when everything else is over-advertised.
    pub const RESTRICTIONS: Self = Self::POW2
        .union(Self::SQUAREONLY)
        .union(Self::NONPOW2CONDITIONAL)
        .union(Self::CUBEMAP_POW2)
        .union(Self::VOLUMEMAP_POW2);
}

bitflags::bitflags! {
    /// `D3DPTFILTERCAPS_*` bits (the five `D3DCAPS9::*FilterCaps` fields).
    ///
    /// Three independent filter stages (MIN / MIP / MAG), each with its own
    /// bit per filter kind.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FilterCaps: u32 {
        const MINFPOINT = 0x0000_0100;
        const MINFLINEAR = 0x0000_0200;
        const MINFANISOTROPIC = 0x0000_0400;
        const MINFPYRAMIDALQUAD = 0x0000_0800;
        const MINFGAUSSIANQUAD = 0x0000_1000;
        const MIPFPOINT = 0x0001_0000;
        const MIPFLINEAR = 0x0002_0000;
        /// Mono convolution filter, a filter kind of its own rather than a MIN/MIP/MAG mode.
        const CONVOLUTIONMONO = 0x0004_0000;
        const MAGFPOINT = 0x0100_0000;
        const MAGFLINEAR = 0x0200_0000;
        const MAGFANISOTROPIC = 0x0400_0000;
        const MAGFPYRAMIDALQUAD = 0x0800_0000;
        const MAGFGAUSSIANQUAD = 0x1000_0000;
    }
}

bitflags::bitflags! {
    /// `D3DPTADDRESSCAPS_*` bits (`D3DCAPS9::TextureAddressCaps` and `VolumeTextureAddressCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AddressCaps: u32 {
        const WRAP = 0x0000_0001;
        const MIRROR = 0x0000_0002;
        const CLAMP = 0x0000_0004;
        const BORDER = 0x0000_0008;
        /// Addressing mode can differ per coordinate.
        const INDEPENDENTUV = 0x0000_0010;
        const MIRRORONCE = 0x0000_0020;
    }
}

bitflags::bitflags! {
    /// `D3DSTENCILCAPS_*` bits (`D3DCAPS9::StencilCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct StencilCaps: u32 {
        const KEEP = 0x0000_0001;
        const ZERO = 0x0000_0002;
        const REPLACE = 0x0000_0004;
        const INCRSAT = 0x0000_0008;
        const DECRSAT = 0x0000_0010;
        const INVERT = 0x0000_0020;
        const INCR = 0x0000_0040;
        const DECR = 0x0000_0080;
        const TWOSIDED = 0x0000_0100;
    }
}

bitflags::bitflags! {
    /// `D3DTEXOPCAPS_*` bits (`D3DCAPS9::TextureOpCaps`).
    ///
    /// One bit per `D3DTOP_*` fixed-function blend operation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct TexOpCaps: u32 {
        const DISABLE = 0x0000_0001;
        const SELECTARG1 = 0x0000_0002;
        const SELECTARG2 = 0x0000_0004;
        const MODULATE = 0x0000_0008;
        const MODULATE2X = 0x0000_0010;
        const MODULATE4X = 0x0000_0020;
        const ADD = 0x0000_0040;
        const ADDSIGNED = 0x0000_0080;
        const ADDSIGNED2X = 0x0000_0100;
        const SUBTRACT = 0x0000_0200;
        const ADDSMOOTH = 0x0000_0400;
        const BLENDDIFFUSEALPHA = 0x0000_0800;
        const BLENDTEXTUREALPHA = 0x0000_1000;
        const BLENDFACTORALPHA = 0x0000_2000;
        const BLENDTEXTUREALPHAPM = 0x0000_4000;
        const BLENDCURRENTALPHA = 0x0000_8000;
        const PREMODULATE = 0x0001_0000;
        const MODULATEALPHA_ADDCOLOR = 0x0002_0000;
        const MODULATECOLOR_ADDALPHA = 0x0004_0000;
        const MODULATEINVALPHA_ADDCOLOR = 0x0008_0000;
        const MODULATEINVCOLOR_ADDALPHA = 0x0010_0000;
        const BUMPENVMAP = 0x0020_0000;
        const BUMPENVMAPLUMINANCE = 0x0040_0000;
        const DOTPRODUCT3 = 0x0080_0000;
        const MULTIPLYADD = 0x0100_0000;
        const LERP = 0x0200_0000;
    }
}

bitflags::bitflags! {
    /// `D3DLINECAPS_*` bits (`D3DCAPS9::LineCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LineCaps: u32 {
        const TEXTURE = 0x0000_0001;
        const ZTEST = 0x0000_0002;
        const BLEND = 0x0000_0004;
        const ALPHACMP = 0x0000_0008;
        const FOG = 0x0000_0010;
        const ANTIALIAS = 0x0000_0020;
    }
}

bitflags::bitflags! {
    /// `D3DVTXPCAPS_*` bits (`D3DCAPS9::VertexProcessingCaps`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VtxpCaps: u32 {
        const TEXGEN = 0x0000_0001;
        const MATERIALSOURCE7 = 0x0000_0002;
        const DIRECTIONALLIGHTS = 0x0000_0008;
        const POSITIONALLIGHTS = 0x0000_0010;
        const LOCALVIEWER = 0x0000_0020;
        const TWEENING = 0x0000_0040;
        const TEXGEN_SPHEREMAP = 0x0000_0100;
        const NO_TEXGEN_NONLOCALVIEWER = 0x0000_0200;
    }
}

bitflags::bitflags! {
    /// `D3DFVFCAPS_*` bits (`D3DCAPS9::FVFCaps`).
    ///
    /// The low 16 bits are not flags: they hold the number of texture
    /// coordinate sets the device supports, masked by
    /// [`FvfCaps::TEXCOORDCOUNTMASK`] and written via
    /// [`FvfCaps::texcoord_sets`]. `all()` is therefore meaningless for this
    /// field.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FvfCaps: u32 {
        /// Mask over the texture-coordinate-set *count*, not a flag.
        const TEXCOORDCOUNTMASK = 0x0000_FFFF;
        const DONOTSTRIPELEMENTS = 0x0008_0000;
        const PSIZE = 0x0010_0000;
    }
}

impl FvfCaps {
    /// The texture-coordinate-set count, as the low bits of the `FVFCaps` field.
    ///
    /// Composes with the flag members through the usual `union`; the count is
    /// truncated to [`FvfCaps::TEXCOORDCOUNTMASK`] so a caller cannot spill
    /// into the flag bits.
    #[must_use]
    pub const fn texcoord_sets(count: u32) -> Self {
        Self::from_bits_retain(count & Self::TEXCOORDCOUNTMASK.bits())
    }
}

bitflags::bitflags! {
    /// `D3DDTCAPS_*` bits (`D3DCAPS9::DeclTypes`).
    ///
    /// One bit per optional `D3DDECLTYPE_*` vertex-element type. `FLOAT1`
    /// through `FLOAT4` and `D3DCOLOR` are baseline and have no bit.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DeclTypeCaps: u32 {
        const UBYTE4 = 0x0000_0001;
        const UBYTE4N = 0x0000_0002;
        const SHORT2N = 0x0000_0004;
        const SHORT4N = 0x0000_0008;
        const USHORT2N = 0x0000_0010;
        const USHORT4N = 0x0000_0020;
        const UDEC3 = 0x0000_0040;
        const DEC3N = 0x0000_0080;
        const FLOAT16_2 = 0x0000_0100;
        const FLOAT16_4 = 0x0000_0200;
    }
}
