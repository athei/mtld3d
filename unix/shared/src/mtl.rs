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
    /// 32-bit single-channel float. D3D9 `D3DFMT_R32F`.
    R32Float = 55,
    Bgra8Unorm = 80,
    /// sRGB-encoded twin of `Bgra8Unorm`.
    ///
    /// Pixels are stored in gamma space and the GPU applies `sRGB → linear`
    /// on read / `linear → sRGB` on write. Used as the pixel format of a
    /// `newTextureViewWithPixelFormat:` over a `Bgra8Unorm` texture when a
    /// sampler requests `D3DSAMP_SRGBTEXTURE`.
    Bgra8UnormSrgb = 81,
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
    /// Drives two paths:
    /// - `D3DSAMP_SRGBTEXTURE=1`: eagerly create a
    ///   `newTextureViewWithPixelFormat:` of the sRGB twin at
    ///   `CreateTexture` time and bind that view at draw time.
    /// - `D3DRS_SRGBWRITEENABLE=1`: upgrade the colour-attachment format
    ///   of the pipeline state to the sRGB twin.
    ///
    /// Returning `None` means the linear format is the only thing mtld3d
    /// supports — callers should fall back to the linear path with a
    /// once-per-format warn.
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
/// Only the two values mtld3d currently emits are present — MSAA resolve
/// variants land here when we wire MSAA.
///
/// Discriminants match the native `MTLStoreAction` enum.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, FromRepr)]
pub enum StoreAction {
    DontCare = 0,
    Store = 1,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: every enum variant round-trips through `as u32 → from_repr`.
    ///
    /// If a variant is added without a discriminant, the wire encoding
    /// could silently shift — this catches that.
    #[test]
    fn enum_discriminants_round_trip() {
        assert_eq!(StorageMode::from_repr(0), Some(StorageMode::Shared));
        assert_eq!(StorageMode::from_repr(1), Some(StorageMode::Managed));
        assert_eq!(StorageMode::from_repr(2), Some(StorageMode::Private));
        assert_eq!(StorageMode::from_repr(3), Some(StorageMode::Memoryless));
        assert_eq!(StorageMode::from_repr(4), None);

        assert_eq!(PixelFormat::B5G6R5Unorm as u32, 40);
        assert_eq!(PixelFormat::Abgr4Unorm as u32, 42);
        assert_eq!(PixelFormat::Bgr5A1Unorm as u32, 43);
        assert_eq!(PixelFormat::from_repr(40), Some(PixelFormat::B5G6R5Unorm));
        assert_eq!(PixelFormat::from_repr(42), Some(PixelFormat::Abgr4Unorm));
        assert_eq!(PixelFormat::from_repr(43), Some(PixelFormat::Bgr5A1Unorm));
        assert_eq!(PixelFormat::Bgra8Unorm as u32, 80);
        assert_eq!(PixelFormat::Bgra8UnormSrgb as u32, 81);
        assert_eq!(PixelFormat::from_repr(80), Some(PixelFormat::Bgra8Unorm));
        assert_eq!(
            PixelFormat::from_repr(81),
            Some(PixelFormat::Bgra8UnormSrgb)
        );
        assert_eq!(PixelFormat::from_repr(130), Some(PixelFormat::Bc1Rgba));
        assert_eq!(PixelFormat::from_repr(131), Some(PixelFormat::Bc1RgbaSrgb));
        assert_eq!(PixelFormat::from_repr(133), Some(PixelFormat::Bc2RgbaSrgb));
        assert_eq!(PixelFormat::from_repr(135), Some(PixelFormat::Bc3RgbaSrgb));
        assert_eq!(PixelFormat::Depth32Float as u32, 252);
        assert_eq!(PixelFormat::Depth32FloatStencil8 as u32, 260);
        assert_eq!(PixelFormat::from_repr(252), Some(PixelFormat::Depth32Float));
        assert_eq!(
            PixelFormat::from_repr(260),
            Some(PixelFormat::Depth32FloatStencil8)
        );
        assert_eq!(PixelFormat::from_repr(9999), None);
    }

    #[test]
    fn srgb_twin_table() {
        // Linear → sRGB-twin pairs that mtld3d's wire actually plumbs today.
        assert_eq!(
            PixelFormat::Bgra8Unorm.srgb_twin(),
            Some(PixelFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            PixelFormat::Bc1Rgba.srgb_twin(),
            Some(PixelFormat::Bc1RgbaSrgb)
        );
        assert_eq!(
            PixelFormat::Bc2Rgba.srgb_twin(),
            Some(PixelFormat::Bc2RgbaSrgb)
        );
        assert_eq!(
            PixelFormat::Bc3Rgba.srgb_twin(),
            Some(PixelFormat::Bc3RgbaSrgb)
        );

        // Already-sRGB formats are their own input, not their own twin —
        // callers should never request the twin of an sRGB format.
        assert_eq!(PixelFormat::Bgra8UnormSrgb.srgb_twin(), None);
        assert_eq!(PixelFormat::Bc1RgbaSrgb.srgb_twin(), None);

        // No sRGB encoding for single-channel, float, or depth formats.
        assert_eq!(PixelFormat::A8Unorm.srgb_twin(), None);
        assert_eq!(PixelFormat::R8Unorm.srgb_twin(), None);
        assert_eq!(PixelFormat::Rg8Unorm.srgb_twin(), None);
        assert_eq!(PixelFormat::Rgba16Float.srgb_twin(), None);
        assert_eq!(PixelFormat::Depth32Float.srgb_twin(), None);
        assert_eq!(PixelFormat::Depth32FloatStencil8.srgb_twin(), None);

        assert_eq!(LoadAction::Clear as u32, 2);
        assert_eq!(CompareFunc::Always as u32, 7);
        assert_eq!(BlendFactor::OneMinusBlendColor as u32, 12);
        assert_eq!(BlendOperation::Add as u32, 0);
        assert_eq!(BlendOperation::Max as u32, 4);
        assert_eq!(BlendOperation::from_repr(3), Some(BlendOperation::Min));
        assert_eq!(BlendOperation::from_repr(5), None);
        assert_eq!(PrimitiveType::TriangleStrip as u32, 4);
        assert_eq!(CullMode::Back as u32, 2);
        assert_eq!(IndexType::UInt32 as u32, 1);
        assert_eq!(AddressMode::ClampToZero as u32, 4);
        assert_eq!(Swizzle::Alpha as u32, 5);
        assert_eq!(VertexFormat::Float4 as u32, 31);
        assert_eq!(VertexFormat::from_repr(0), Some(VertexFormat::Invalid));

        assert_eq!(DestroyKind::Buffer as u32, 0);
        assert_eq!(DestroyKind::DepthStencilState as u32, 6);
        assert_eq!(DestroyKind::from_repr(3), Some(DestroyKind::ShaderLibrary));
        assert_eq!(DestroyKind::from_repr(7), None);

        assert_eq!(BufferKind::VbIb as u32, 0);
        assert_eq!(BufferKind::TexStaging as u32, 1);
        assert_eq!(BufferKind::Visibility as u32, 2);
        assert_eq!(BufferKind::Repack as u32, 3);
        assert_eq!(BufferKind::VbIbDevice as u32, 4);
        assert_eq!(BufferKind::from_repr(0), Some(BufferKind::VbIb));
        assert_eq!(BufferKind::from_repr(4), Some(BufferKind::VbIbDevice));
        assert_eq!(BufferKind::from_repr(5), None);

        assert_eq!(ColorSpacePolicy::Passthrough as u32, 0);
        assert_eq!(ColorSpacePolicy::Accurate as u32, 1);
        assert_eq!(
            ColorSpacePolicy::from_repr(0),
            Some(ColorSpacePolicy::Passthrough)
        );
        assert_eq!(
            ColorSpacePolicy::from_repr(1),
            Some(ColorSpacePolicy::Accurate)
        );
        assert_eq!(ColorSpacePolicy::from_repr(2), None);
    }

    #[test]
    fn texture_usage_bits_match_legacy_wire() {
        // Wire-encoding pin: RENDER_TARGET is bit 0, DEPTH_STENCIL is bit 1.
        // Changing either bit is a coupled PE/unix wire break.
        assert_eq!(TextureUsage::RENDER_TARGET.bits(), 0b01);
        assert_eq!(TextureUsage::DEPTH_STENCIL.bits(), 0b10);
    }

    #[test]
    fn enum_layout_is_u32() {
        // Thunk params are `#[repr(C, align(8))]` with u32/u64 fields.
        // Our enums must be exactly 4 bytes with 4-byte alignment so
        // they slot in where a `u32` used to live without shifting any
        // other field's offset.
        assert_eq!(core::mem::size_of::<StorageMode>(), 4);
        assert_eq!(core::mem::align_of::<StorageMode>(), 4);
        assert_eq!(core::mem::size_of::<PixelFormat>(), 4);
        assert_eq!(core::mem::size_of::<LoadAction>(), 4);
        assert_eq!(core::mem::size_of::<CompareFunc>(), 4);
        assert_eq!(core::mem::size_of::<BlendFactor>(), 4);
        assert_eq!(core::mem::size_of::<BlendOperation>(), 4);
        assert_eq!(core::mem::size_of::<MinMagFilter>(), 4);
        assert_eq!(core::mem::size_of::<MipFilter>(), 4);
        assert_eq!(core::mem::size_of::<AddressMode>(), 4);
        assert_eq!(core::mem::size_of::<StageTag>(), 4);
        assert_eq!(core::mem::size_of::<Swizzle>(), 4);
        assert_eq!(core::mem::size_of::<VertexFormat>(), 4);
        assert_eq!(core::mem::size_of::<TextureUsage>(), 4);
        assert_eq!(core::mem::align_of::<TextureUsage>(), 4);
        assert_eq!(core::mem::size_of::<ColorWriteMask>(), 4);
        assert_eq!(core::mem::size_of::<DestroyKind>(), 4);
        assert_eq!(core::mem::align_of::<DestroyKind>(), 4);
        assert_eq!(core::mem::size_of::<ColorSpacePolicy>(), 4);
        assert_eq!(core::mem::align_of::<ColorSpacePolicy>(), 4);
    }
}
