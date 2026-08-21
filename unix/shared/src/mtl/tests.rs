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
    assert_eq!(StencilOp::Keep as u32, 0);
    assert_eq!(StencilOp::DecrementWrap as u32, 7);
    assert_eq!(StencilOp::from_repr(8), None);
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
    assert_eq!(VertexStepFunction::Constant as u32, 0);
    assert_eq!(VertexStepFunction::PerVertex as u32, 1);
    assert_eq!(VertexStepFunction::PerInstance as u32, 2);
    assert_eq!(
        VertexStepFunction::from_repr(2),
        Some(VertexStepFunction::PerInstance)
    );
    assert_eq!(VertexStepFunction::from_repr(3), None);

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
    assert_eq!(core::mem::size_of::<VertexStepFunction>(), 4);
    assert_eq!(core::mem::size_of::<TextureUsage>(), 4);
    assert_eq!(core::mem::align_of::<TextureUsage>(), 4);
    assert_eq!(core::mem::size_of::<ColorWriteMask>(), 4);
    assert_eq!(core::mem::size_of::<DestroyKind>(), 4);
    assert_eq!(core::mem::align_of::<DestroyKind>(), 4);
    assert_eq!(core::mem::size_of::<ColorSpacePolicy>(), 4);
    assert_eq!(core::mem::align_of::<ColorSpacePolicy>(), 4);
}
