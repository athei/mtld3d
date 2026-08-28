//! Metal wire values shared between the Windows PE side and the macOS Unix side.
//!
//! The PE side is `d3d9.dll` and `mtld3d.dll`; the Unix side is `mtld3d.so`.
//!
//! Every integer that crosses the PE/Unix boundary with symbolic meaning
//! lives here as a typed `#[repr(u32)]` enum or `bitflags!` struct — never a
//! magic literal at a call site, never a per-side `const` restatement.
//!
//! Discriminant values match the corresponding `objc2_metal` enum
//! wherever a native Metal counterpart exists.
//!
//! ## Soundness
//!
//! The `#[repr(u32)]` enums below appear as fields of `#[repr(C, align(8))]`
//! thunk param structs. Reading an enum field whose bit pattern is not a
//! declared variant is undefined behavior. The mtld3d build model makes this
//! sound: `d3d9.dll`, `mtld3d.dll`, and `mtld3d.so` are rebuilt atomically by
//! `make` and installed together by `make install`, so every side sees the
//! same `mtl` definitions. Any wire-format change (new variant, new
//! discriminant) is a coupled edit across both sides in the same commit.
//!
//! When a thunk param carries a polymorphic `u32` whose interpretation
//! depends on another field (e.g. `Command::param_a`, whose meaning depends
//! on `Command::cmd`), typed decoding uses `Enum::from_repr(raw) ->
//! Option<Self>` from `strum::FromRepr`. Never cast, never transmute.

use bitflags::bitflags;
use strum::FromRepr;

/// `MTLStorageMode` wire encoding. Matches the native Metal enum values.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum StorageMode {
    Shared = 0,
    Managed = 1,
    Private = 2,
    Memoryless = 3,
}

/// `MTLPixelFormat` wire encoding.
///
/// Discriminants match `objc2_metal::MTLPixelFormat` raw values. Only formats
/// mtld3d actually plumbs are listed — adding a new format is a coupled edit:
/// add the variant here, update encoder on the PE side, update the exhaustive
/// decode on the Unix side.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum PixelFormat {
    A8Unorm = 1,
    R8Unorm = 10,
    /// 16-bit single-channel unorm.
    ///
    /// D3D9 `D3DFMT_L16` (luminance) promotes here with a `.rrr1` swizzle.
    R16Unorm = 20,
    /// 16-bit single-channel float. D3D9 `D3DFMT_R16F`.
    R16Float = 25,
    Rg8Unorm = 30,
    Rg8Snorm = 32,
    /// 16-bit packed 5/6/5. D3D9 `D3DFMT_R5G6B5`.
    B5G6R5Unorm = 40,
    /// 16-bit packed 4/4/4/4. D3D9 `D3DFMT_A4R4G4B4` (with a sampler swizzle).
    Abgr4Unorm = 42,
    /// 16-bit packed 5/5/5/1. D3D9 `D3DFMT_A1R5G5B5`.
    Bgr5A1Unorm = 43,
    /// 32-bit two-channel unorm. D3D9 `D3DFMT_G16R16`.
    Rg16Unorm = 60,
    /// 32-bit single-channel float. D3D9 `D3DFMT_R32F`.
    R32Float = 55,
    /// 16-bit two-channel float. D3D9 `D3DFMT_G16R16F`.
    Rg16Float = 65,
    Bgra8Unorm = 80,
    /// sRGB-encoded twin of `Bgra8Unorm`.
    ///
    /// Pixels are stored in gamma space and the GPU applies `sRGB → linear`
    /// on read / `linear → sRGB` on write. Used as the pixel format of a
    /// `newTextureViewWithPixelFormat:` over a `Bgra8Unorm` texture when a
    /// sampler requests `D3DSAMP_SRGBTEXTURE`.
    Bgra8UnormSrgb = 81,
    /// 64-bit two-channel float. D3D9 `D3DFMT_G32R32F`.
    Rg32Float = 105,
    /// 64-bit four-channel unorm. D3D9 `D3DFMT_A16B16G16R16`.
    Rgba16Unorm = 110,
    /// 64-bit four-channel float. D3D9 `D3DFMT_A16B16G16R16F`.
    Rgba16Float = 115,
    /// 128-bit four-channel float. D3D9 `D3DFMT_A32B32G32R32F`.
    Rgba32Float = 125,
    Bc1Rgba = 130,
    /// sRGB-encoded twin of `Bc1Rgba` (DXT1).
    Bc1RgbaSrgb = 131,
    Bc2Rgba = 132,
    /// sRGB-encoded twin of `Bc2Rgba` (DXT3).
    Bc2RgbaSrgb = 133,
    Bc3Rgba = 134,
    /// sRGB-encoded twin of `Bc3Rgba` (DXT5).
    Bc3RgbaSrgb = 135,
    /// Single-channel block-compressed unorm (BC4). D3D9 `D3DFMT_ATI1` (ATI1N).
    Bc4RUnorm = 140,
    /// Depth-only 32-bit float depth attachment.
    ///
    /// Apple Silicon has no 24-bit depth format, so D3D9 D24X8 / D24 / D32 /
    /// D16 all promote here. Sampleable when the texture is created with
    /// `MTLTextureUsage::ShaderRead` (sampleable shadow maps).
    Depth32Float = 252,
    /// Combined 32-bit float depth + 8-bit stencil.
    ///
    /// D3D9 D24S8 / D24FS8 / D24X4S4 / D15S1 promote here.
    Depth32FloatStencil8 = 260,
}

impl PixelFormat {
    /// The sRGB-encoded twin of a linear color format, if one exists.
    ///
    /// Only `Bgra8Unorm` and the BC1/2/3 compressed colour families have
    /// sRGB pairs in mtld3d's wire today. Depth formats, single-channel
    /// formats (A8/R8) and float formats have no sRGB encoding.
    ///
    /// Drives both sRGB render states. `create_texture` eagerly creates a
    /// view of the sRGB twin next to every colour texture whose format has
    /// one; `D3DSAMP_SRGBTEXTURE=1` selects it as the sampled view, and
    /// `D3DRS_SRGBWRITEENABLE=1` attaches it as the render pass's colour
    /// attachment so the hardware encodes after the blender. A colour target
    /// whose format returns `None` here keeps the pixel-shader OETF variant,
    /// which encodes before the blender.
    ///
    /// Returning `None` means the linear format is the only thing mtld3d
    /// supports — callers fall back to the linear view with a once-per-id
    /// info line.
    #[must_use]
    pub const fn srgb_twin(self) -> Option<Self> {
        match self {
            Self::Bgra8Unorm => Some(Self::Bgra8UnormSrgb),
            Self::Bc1Rgba => Some(Self::Bc1RgbaSrgb),
            Self::Bc2Rgba => Some(Self::Bc2RgbaSrgb),
            Self::Bc3Rgba => Some(Self::Bc3RgbaSrgb),
            _ => None,
        }
    }
}

/// `MTLLoadAction` wire encoding for render-pass color/depth attachments.
///
/// Discriminants match the native `MTLLoadAction` enum so the unix side can
/// pass them through to `setLoadAction:` without re-mapping.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum LoadAction {
    DontCare = 0,
    Load = 1,
    Clear = 2,
}

/// `MTLStoreAction` wire encoding for render-pass color/depth/stencil attachments.
///
/// Discriminants match the native `MTLStoreAction` enum. The two resolve
/// variants are only legal on an attachment whose texture is multisampled and
/// whose descriptor names a single-sample resolve texture; `MultisampleResolve`
/// additionally discards the multisample content, so it is emitted only where
/// the load/store rules already decided the attachment itself is dead.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum StoreAction {
    DontCare = 0,
    Store = 1,
    MultisampleResolve = 2,
    StoreAndMultisampleResolve = 3,
}

impl StoreAction {
    /// The variant that also resolves into the attachment's resolve texture.
    ///
    /// `Store` keeps the multisample content for a later pass in the same
    /// submission; `DontCare` drops it once the resolve has been taken.
    #[must_use]
    pub const fn with_resolve(self) -> Self {
        match self {
            Self::DontCare | Self::MultisampleResolve => Self::MultisampleResolve,
            Self::Store | Self::StoreAndMultisampleResolve => Self::StoreAndMultisampleResolve,
        }
    }
}

/// `MTLVisibilityResultMode` wire encoding.
///
/// Matches the native Metal enum: `Disabled = 0`, `Boolean = 1`,
/// `Counting = 2`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum VisibilityResultMode {
    Disabled = 0,
    Boolean = 1,
    Counting = 2,
}

/// `MTLCompareFunction` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum CompareFunc {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}

/// `MTLStencilOperation` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum StencilOp {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementClamp = 3,
    DecrementClamp = 4,
    Invert = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
}

/// `MTLBlendFactor` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SourceColor = 2,
    OneMinusSourceColor = 3,
    SourceAlpha = 4,
    OneMinusSourceAlpha = 5,
    DestinationAlpha = 6,
    OneMinusDestinationAlpha = 7,
    DestinationColor = 8,
    OneMinusDestinationColor = 9,
    SourceAlphaSaturated = 10,
    BlendColor = 11,
    OneMinusBlendColor = 12,
}

/// `MTLBlendOperation` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum BlendOperation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

/// `MTLPrimitiveType` wire encoding.
///
/// Appears in `Command::param_a` for `DrawPrimitives` /
/// `DrawIndexedPrimitives` — decode via `PrimitiveType::from_repr` on the
/// Unix side.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum PrimitiveType {
    Point = 0,
    Line = 1,
    LineStrip = 2,
    Triangle = 3,
    TriangleStrip = 4,
}

/// `MTLCullMode` wire encoding.
///
/// Appears in `Command::param_a` for `SetCullMode` — decode via
/// `CullMode::from_repr` on the Unix side.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum CullMode {
    None = 0,
    Front = 1,
    Back = 2,
}

/// `MTLIndexType` wire encoding.
///
/// Appears packed into the low 8 bits of `Command::param_d` for
/// `DrawIndexedPrimitives` — decode via `IndexType::from_repr` on the Unix
/// side.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum IndexType {
    UInt16 = 0,
    UInt32 = 1,
}

/// `MTLSamplerMinMagFilter` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum MinMagFilter {
    Nearest = 0,
    Linear = 1,
}

/// `MTLSamplerMipFilter` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum MipFilter {
    NotMipmapped = 0,
    Nearest = 1,
    Linear = 2,
}

/// `MTLSamplerAddressMode` wire encoding.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum AddressMode {
    ClampToEdge = 0,
    MirrorClampToEdge = 1,
    Repeat = 2,
    MirrorRepeat = 3,
    ClampToZero = 4,
    ClampToBorderColor = 5,
}

/// `MTLSamplerBorderColor` wire encoding: the three presets Metal offers.
///
/// A D3D9 border colour that is not one of them falls back to opaque black
/// on the PE side, which logs the substitution once per colour.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum BorderColor {
    TransparentBlack = 0,
    OpaqueBlack = 1,
    OpaqueWhite = 2,
}

/// Shader stage selector for `CompileShaderLibraryParams::stage_tag`.
///
/// Selects the MSL compile options for the stage: the Unix side compiles the
/// vertex path with `MTLMathMode::Safe` (FP reassociation would defeat
/// `[[position, invariant]]` across pipelines) and the fragment path with
/// `MTLMathMode::Fast`. The entry-point name is carried separately in
/// `CompileShaderLibraryParams::entry_ptr`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum StageTag {
    Vertex = 0,
    Fragment = 1,
}

/// Resource kind for `DestroyResourcesBulkParams`.
///
/// The Unix side dispatches on this to release the matching MTL protocol type
/// — every release reduces to `objc_release`, but the typed enum keeps the
/// wire format honest and the dispatch exhaustive.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum DestroyKind {
    Buffer = 0,
    Texture = 1,
    RenderPipeline = 2,
    ShaderLibrary = 3,
    ShaderFunction = 4,
    SamplerState = 5,
    DepthStencilState = 6,
}

/// `MTLTextureSwizzle` wire encoding for per-channel texture-view swizzles.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum Swizzle {
    Zero = 0,
    One = 1,
    Red = 2,
    Green = 3,
    Blue = 4,
    Alpha = 5,
}

/// `MTLVertexFormat` wire encoding.
///
/// Only the formats mtld3d emits from `decl_type_to_metal_format` on the PE
/// side are listed. `Invalid` is the sentinel for a D3DDECLTYPE the project
/// doesn't map (caller drops the element). Discriminants match
/// `objc2_metal::MTLVertexFormat` raw values.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum VertexFormat {
    Invalid = 0,
    UChar4 = 3,
    UChar4Normalized = 9,
    UChar4NormalizedBgra = 42,
    Short2 = 16,
    Short4 = 17,
    UShort2Normalized = 20,
    UShort4Normalized = 21,
    Short2Normalized = 22,
    Short4Normalized = 23,
    Half2 = 25,
    Half4 = 27,
    Float = 28,
    Float2 = 29,
    Float3 = 30,
    Float4 = 31,
}

/// `MTLVertexStepFunction` wire encoding for one vertex buffer layout.
///
/// Discriminants match the native Metal enum. `Constant` is the layout of a
/// stream the declaration references but nothing feeds: every vertex and
/// instance reads offset 0 of whatever is bound, with `step_rate` 0. Metal
/// has no zero-stride layout, so it is also how a `D3DSTREAMSOURCE_INSTANCEDATA`
/// stream with frequency 0 is expressed.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum VertexStepFunction {
    Constant = 0,
    PerVertex = 1,
    PerInstance = 2,
}

/// Vertex-stage buffer slots the D3D9 vertex streams occupy.
///
/// Stream `n` of `SetStreamSource` binds at Metal vertex buffer index `n`,
/// so the slot count equals `D3DCAPS9::MaxStreams`. mtld3d's own vertex
/// uniforms sit above this range (see [`VS_POS_FIXUP_SLOT`]).
pub const VERTEX_STREAM_SLOTS: u32 = 16;

/// Vertex-stage buffer slot of the half-pixel rasterization fixup uniform.
///
/// The three uniform slots sit at the top of Metal's 31-entry vertex buffer
/// table so no D3D9 stream index can collide with them. Shared by the MSL
/// emitters (`[[buffer(N)]]`) and the encoder's `setVertexBytes` binds.
pub const VS_POS_FIXUP_SLOT: u32 = 28;

/// Vertex-stage buffer slot of the runtime integer constant table (`vs_i`).
pub const VS_INT_CONST_SLOT: u32 = 29;

/// Vertex-stage buffer slot of the runtime boolean constant bitmask (`vs_b`).
///
/// One `uint`: bit N is `bN`. Bound only for a VS that reads a boolean
/// constant no `defb` defines.
pub const VS_BOOL_CONST_SLOT: u32 = 26;

/// Vertex-stage buffer slot of the float constant table (`vs_c`).
pub const VS_FLOAT_CONST_SLOT: u32 = 30;

/// Vertex-stage buffer slot of the per-draw `VsDraw` uniform.
///
/// Point size, its clamp range and the point scale factors, serialised by
/// `mtld3d_core::vs_draw` and read by both vertex-shader emitters.
pub const VS_DRAW_SLOT: u32 = 27;

/// Fragment-stage buffer slot of the runtime integer constant table (`ps_i`).
///
/// The fragment uniforms count down from 15 (`ps_c`, alpha ref, fog, bump
/// env at 15..12); the two constant files sit below them. Bound only for a
/// PS that reads an integer constant no `defi` defines.
pub const PS_INT_CONST_SLOT: u32 = 11;

/// Fragment-stage buffer slot of the runtime boolean constant bitmask (`ps_b`).
///
/// One `uint`: bit N is `bN`. Bound only for a PS that reads a boolean
/// constant no `defb` defines.
pub const PS_BOOL_CONST_SLOT: u32 = 10;

/// Fragment-stage buffer slot of the per-sampler LOD-bias table (`lod_bias`).
///
/// One `float4` row per fragment sampler slot: `.x` is `D3DSAMP_MIPMAPLODBIAS`
/// as a clamped float, `.y` is `exp2(.x)` so an explicit-gradient sample can
/// scale its derivatives instead of adding a bias Metal does not accept there.
/// Metal has no sampler-level LOD bias, so the value is applied at the sample
/// site; bound only for a draw whose bound stages carry a non-zero bias.
pub const PS_LOD_BIAS_SLOT: u32 = 9;

// The uniform slots must clear every stream slot and stay inside Metal's
// 31-entry vertex buffer table.
const _: () = {
    assert!(VS_DRAW_SLOT >= VERTEX_STREAM_SLOTS);
    assert!(VS_POS_FIXUP_SLOT >= VERTEX_STREAM_SLOTS);
    assert!(VS_INT_CONST_SLOT >= VERTEX_STREAM_SLOTS);
    assert!(VS_BOOL_CONST_SLOT >= VERTEX_STREAM_SLOTS);
    assert!(VS_FLOAT_CONST_SLOT >= VERTEX_STREAM_SLOTS);
    assert!(VS_FLOAT_CONST_SLOT <= 30);
};

/// `BufferCreateDesc::kind` — what role the buffer plays on the PE side.
///
/// Mostly used to compose a human-readable `setLabel` for Xcode captures so
/// `MTLBuffer` rows surface as `mtld3d-vbib-…` / `mtld3d-vis-…` / etc. instead of
/// "Buffer (8KB)". The one exception is [`BufferKind::VbIbDevice`], which also
/// selects the Metal-allocated `StorageModePrivate` create path (no caller
/// backing) instead of `newBufferWithBytesNoCopy`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum BufferKind {
    /// Vertex / index buffer wrap (one per `BufferId`).
    VbIb = 0,
    /// Per-mip texture upload staging.
    TexStaging = 1,
    /// Per-frame visibility-result-pool buffer.
    Visibility = 2,
    /// Transient padded blit-source repack.
    Repack = 3,
    /// GPU-read device buffer for a `Staged` VB/IB — Metal-allocated `StorageModePrivate`.
    ///
    /// Written only by the staging-upload blit and bound as the draw's
    /// vertex/index source. `backing_ptr` is unused.
    VbIbDevice = 4,
}

bitflags! {
    /// Shape and view attributes for `TextureCreateDesc`.
    ///
    /// Kept in the descriptor slot formerly occupied by `has_swizzle`, so the
    /// PE to Unix wire layout remains 56 bytes while texture dimensionality is
    /// explicit instead of inferred from depth.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct TextureCreateFlags: u32 {
        const HAS_SWIZZLE = 1 << 0;
        const TYPE_3D = 1 << 1;
        const TYPE_CUBE = 1 << 2;
    }
}

bitflags! {
    /// Boolean device capabilities answered by the `GetDeviceInfo` thunk.
    ///
    /// One flags word carries every per-device yes/no the PE side needs for
    /// caps advertisement, so a single `GetDeviceInfo` call (cached process-
    /// wide on the PE side) serves all consumers. New device-dependent caps
    /// bits belong here, not in new thunks or new param fields.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DeviceCapsFlags: u32 {
        /// Border-colour samplers are creatable (Mac2 family, not paravirtual).
        const SAMPLER_BORDER = 1 << 0;
        /// The packed 16-bit pixel formats exist natively on this device.
        ///
        /// `B5G6R5Unorm` / `Bgr5A1Unorm` / `Abgr4Unorm` are an
        /// Apple-GPU-family feature; the Intel/AMD Bronze driver (Mac2)
        /// aborts texture creation on them.
        const NATIVE_PACKED16 = 1 << 1;
        /// Single-precision float textures sample with linear filtering.
        ///
        /// `MTLDevice.supports32BitFloatFiltering`, which covers exactly
        /// `R32Float` / `RG32Float` / `RGBA32Float`. Apple-family GPUs
        /// report it; Mac2 devices answer for themselves, and the
        /// Intel/AMD Bronze driver typically says no. The half-float
        /// formats filter on every family and are not covered by this bit.
        const FLOAT32_FILTERING = 1 << 2;
        /// `supportsTextureSampleCount:2` answered yes.
        const SAMPLE_COUNT_2 = 1 << 3;
        /// `supportsTextureSampleCount:4` answered yes.
        const SAMPLE_COUNT_4 = 1 << 4;
        /// `supportsTextureSampleCount:8` answered yes.
        const SAMPLE_COUNT_8 = 1 << 5;
    }
}

impl DeviceCapsFlags {
    /// Whether `sample_count` is a creatable multisample texture count here.
    ///
    /// `1` (no multisampling) is always creatable. Every other count needs the
    /// matching bit the `GetDeviceInfo` thunk filled in from
    /// `supportsTextureSampleCount:`; counts mtld3d does not plumb answer
    /// `false` even when Metal would accept them.
    #[must_use]
    pub const fn supports_sample_count(self, sample_count: u32) -> bool {
        match sample_count {
            1 => true,
            2 => self.contains(Self::SAMPLE_COUNT_2),
            4 => self.contains(Self::SAMPLE_COUNT_4),
            8 => self.contains(Self::SAMPLE_COUNT_8),
            _ => false,
        }
    }
}

bitflags! {
    /// `TextureCreateDesc::usage_flags` bits.
    ///
    /// `RENDER_TARGET` requests `MTLTextureUsage::RenderTarget` so the
    /// texture can be bound as a color attachment. `DEPTH_STENCIL` requests
    /// RT usage for a depth/stencil pixel format — the Unix side still picks
    /// the Metal format from the adjacent `pixel_format` field; this bit only
    /// toggles the RT usage bit.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        const RENDER_TARGET = 1 << 0;
        const DEPTH_STENCIL = 1 << 1;
    }
}

/// mtld3d user-facing `color.space` policy crossing PE→Unix on `AttachMetalLayerParams`.
///
/// Picked from `mtld3d.conf` (`color.space = passthrough | accurate`);
/// the unix side branches on this when selecting the
/// `CAMetalLayer.colorspace` tag.
///
/// `Passthrough` (the default) tags the layer with the display's own
/// `CGColorSpace`, so D3D9's untagged values land at the panel's native
/// primaries (max vibrance per display). `Accurate` overrides that with the
/// sRGB family (`kCGColorSpaceSRGB` for SDR, `kCGColorSpaceExtendedLinearSRGB`
/// for HDR), so guest assets authored against sRGB render with their
/// designer-intended hues instead of being stretched onto the panel's wider
/// gamut.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum ColorSpacePolicy {
    Passthrough = 0,
    Accurate = 1,
}

bitflags! {
    /// `MTLColorWriteMask` wire encoding.
    ///
    /// Metal packs the channels high-to-low: bit 3 = Red, bit 2 = Green,
    /// bit 1 = Blue, bit 0 = Alpha. The PE side produces these bits from
    /// D3D9's inverse layout in `d3d_to_metal_write_mask`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ColorWriteMask: u32 {
        const ALPHA = 1 << 0;
        const BLUE = 1 << 1;
        const GREEN = 1 << 2;
        const RED = 1 << 3;
        const ALL = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits() | Self::ALPHA.bits();
    }
}

bitflags! {
    /// Attachment shape a clear-quad pipeline must declare to bind against the live render pass.
    ///
    /// `HAS_COLOR` adds the fragment function and the color-attachment pixel
    /// format (from the adjacent `color_format`); `HAS_DEPTH` declares the
    /// depth-attachment pixel format (from `depth_format`) — omit it on a
    /// no-depth pass, since Metal rejects a pipeline that declares a depth
    /// attachment when no depth texture is set; `HAS_STENCIL` promotes the
    /// depth format to the combined depth+stencil variant (and implies a
    /// depth attachment).
    ///
    /// `COLOR_FORMAT_NO_WRITE` declares the color attachment's pixel format
    /// (from `color_format`) with a zero write mask and *no* fragment function:
    /// a depth-only clear-quad that runs in a pass which still has a colour
    /// attachment bound (Metal requires the pipeline's colour format to match
    /// the attachment even when nothing is written). Distinct from `HAS_COLOR`,
    /// which actually writes colour via the fragment function. Mutually
    /// exclusive with `HAS_COLOR`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ClearQuadFlags: u32 {
        const HAS_COLOR = 1 << 0;
        const HAS_DEPTH = 1 << 1;
        const HAS_STENCIL = 1 << 2;
        const COLOR_FORMAT_NO_WRITE = 1 << 3;
    }
}

/// Which render-quad pipeline `EnsureBlitPipeline` resolves.
///
/// Both kinds draw a single fullscreen triangle into a colour attachment and
/// are cached per destination pixel format, so they share one thunk and
/// differ only in their fragment function.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum QuadPipelineKind {
    /// Scaling `StretchRect`: samples a source texture across the destination rect.
    StretchBlit = 0,
    /// Texture upload: decodes packed staging bytes read out of an `MTLBuffer`.
    TextureUpload = 1,
}

#[cfg(test)]
mod tests;
