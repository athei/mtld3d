use mtld3d_shared::{
    MetalHandle,
    encoder_wire::{FrameSlab, WireError, WireReader},
    mtl::{PixelFormat, Swizzle, TextureCreateFlags, TextureUsage},
};

use crate::{
    encoder_data::{
        BindDepthOp, BindDepthOpFlags, ClearColorRectsOp, DepthBinding, RtBinding, StretchKind,
        TextureInfo,
    },
    encoder_value::WireValue,
    ids::TextureId,
    render_scale::RenderScale,
};

fn encode<T: WireValue>(value: &T) -> Vec<u8> {
    let mut slab = FrameSlab::new();
    slab.push_record(17, |writer| value.write_wire(writer))
        .unwrap();
    slab.as_bytes().to_vec()
}

fn round_trip<T: WireValue>(value: &T) -> T {
    let bytes = encode(value);
    let mut stream = WireReader::new(&bytes);
    let mut record = stream.next_record().unwrap().unwrap();
    let decoded = T::read_wire(&mut record.payload).unwrap();
    assert!(record.payload.is_empty());
    assert_eq!(encode(&decoded), bytes);
    decoded
}

fn texture() -> TextureInfo {
    TextureInfo {
        texture_id: TextureId::new_unique(),
        d3d_format: 21,
        width: 1001,
        height: 537,
        depth: 6,
        levels: 8,
        pixel_format: PixelFormat::Bgra8Unorm,
        create_flags: TextureCreateFlags::empty(),
        swizzle: [Swizzle::Blue, Swizzle::Green, Swizzle::Red, Swizzle::One],
        usage_flags: TextureUsage::empty(),
    }
}

#[test]
fn texture_metadata_preserves_identity_and_subresource_shape() {
    let original = texture();
    let decoded = round_trip(&original);
    assert_eq!(decoded.texture_id, original.texture_id);
    assert_eq!(
        (decoded.width, decoded.height, decoded.depth, decoded.levels),
        (1001, 537, 6, 8)
    );
    assert_eq!(decoded.pixel_format, PixelFormat::Bgra8Unorm);
    assert_eq!(decoded.swizzle, original.swizzle);
}

#[test]
fn binding_variants_keep_tags_and_payloads() {
    round_trip(&DepthBinding::None);
    round_trip(&DepthBinding::Eager(
        MetalHandle::NULL,
        (1920, 1080),
        RenderScale::from_percent(75),
    ));
    let decoded = round_trip(&BindDepthOp {
        binding: DepthBinding::Lazy(texture(), 3, RenderScale::from_percent(50)),
        sample_count: 4,
        flags: BindDepthOpFlags::SAMPLEABLE | BindDepthOpFlags::HAS_STENCIL,
    });
    assert_eq!(decoded.sample_count, 4);
    assert!(decoded.flags.contains(BindDepthOpFlags::HAS_STENCIL));
    assert!(matches!(decoded.binding, DepthBinding::Lazy(_, 3, scale) if scale.percent() == 50));
    round_trip(&RtBinding::Backbuffer {
        handle: MetalHandle::NULL,
        msaa: MetalHandle::NULL,
        msaa_srgb: MetalHandle::NULL,
        sample_count: 1,
        width: 1280,
        height: 720,
    });
    round_trip(&RtBinding::StandaloneColor {
        handle: MetalHandle::NULL,
        srgb: MetalHandle::NULL,
        msaa: MetalHandle::NULL,
        msaa_srgb: MetalHandle::NULL,
        sample_count: 4,
        format: PixelFormat::Bgra8Unorm,
        has_alpha: true,
        width: 800,
        height: 600,
    });
    round_trip(&RtBinding::Texture {
        info: texture(),
        has_alpha: false,
        width: 64,
        height: 32,
        slice: 5,
        level: 2,
    });
    round_trip(&StretchKind::Texture(texture()));
    round_trip(&StretchKind::Backbuffer(MetalHandle::NULL));
    round_trip(&StretchKind::DepthStencil(MetalHandle::NULL));
}

#[test]
fn clear_rectangles_preserve_signed_coordinates_and_color_bits() {
    let rects = vec![(-3, 2, 50, 80), (i32::MIN, 0, i32::MAX, 12)];
    let decoded = round_trip(&ClearColorRectsOp {
        r_bits: 0x3f80_0000,
        g_bits: 0x8000_0000,
        b_bits: 0,
        a_bits: 0x3f00_0000,
        srgb_write: true,
        rects: rects.clone(),
    });
    assert_eq!(decoded.rects, rects);
    assert_eq!(decoded.g_bits, 0x8000_0000);
    assert!(decoded.srgb_write);
}

#[test]
fn malformed_tags_flags_scale_and_truncated_payloads_are_rejected() {
    assert!(matches!(
        DepthBinding::read_wire(&mut WireReader::new(&[3])),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        RtBinding::read_wire(&mut WireReader::new(&[3])),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        StretchKind::read_wire(&mut WireReader::new(&[3])),
        Err(WireError::InvalidValue)
    ));
    assert!(matches!(
        BindDepthOpFlags::read_wire(&mut WireReader::new(&[4])),
        Err(WireError::InvalidValue)
    ));
    for percent in [0u32, 101, u32::MAX] {
        assert!(matches!(
            RenderScale::read_wire(&mut WireReader::new(&percent.to_le_bytes())),
            Err(WireError::InvalidValue)
        ));
    }
    assert!(matches!(
        Swizzle::read_wire(&mut WireReader::new(&6u32.to_le_bytes())),
        Err(WireError::InvalidValue)
    ));
    let bytes = encode(&texture());
    let mut stream = WireReader::new(&bytes);
    let record = stream.next_record().unwrap().unwrap();
    let payload = record.payload.remaining_len();
    for length in 0..payload {
        assert!(TextureInfo::read_wire(&mut WireReader::new(&bytes[6..6 + length])).is_err());
    }
}

#[test]
fn safe_reader_rejects_nonzero_handles_and_record_readers_preserve_trust() {
    use mtld3d_shared::mtl_handle::MTLTextureKind;

    let raw = 1u64.to_le_bytes();
    assert!(matches!(
        MetalHandle::<MTLTextureKind>::read_wire(&mut WireReader::new(&raw)),
        Err(WireError::InvalidValue)
    ));
    let bytes = encode(&MetalHandle::<MTLTextureKind>::NULL);
    assert!(!WireReader::new(&bytes).has_trusted_addresses());
    // SAFETY: this matching-schema record contains only a null Metal handle,
    // so there is no addressed object or allocation requiring a lease.
    let mut reader = unsafe { WireReader::new_trusted(&bytes) };
    let mut record = reader.next_record().unwrap().unwrap();
    assert!(record.payload.has_trusted_addresses());
    assert!(
        MetalHandle::<MTLTextureKind>::read_wire(&mut record.payload)
            .unwrap()
            .is_null()
    );
}
