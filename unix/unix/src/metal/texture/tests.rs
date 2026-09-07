use mtld3d_shared::{
    MetalHandle,
    mtl::PixelFormat,
    mtl_handle::{MTLCommandQueueKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLRenderPassDescriptor, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};

use super::{
    TRANSPARENT_BLACK, clear_new_color_textures, is_resolvable_color_format, mtl_pixel_format,
    wire_pixel_format,
};

/// Every wire pixel format, in declaration order.
const ALL: [PixelFormat; 29] = [
    PixelFormat::A8Unorm,
    PixelFormat::R8Unorm,
    PixelFormat::R16Unorm,
    PixelFormat::R16Float,
    PixelFormat::Rg8Unorm,
    PixelFormat::Rg8Snorm,
    PixelFormat::B5G6R5Unorm,
    PixelFormat::Abgr4Unorm,
    PixelFormat::Bgr5A1Unorm,
    PixelFormat::Rg16Unorm,
    PixelFormat::R32Float,
    PixelFormat::Rg16Float,
    PixelFormat::Rgba8Unorm,
    PixelFormat::Rgba8UnormSrgb,
    PixelFormat::Bgra8Unorm,
    PixelFormat::Bgra8UnormSrgb,
    PixelFormat::Rg32Float,
    PixelFormat::Rgba16Unorm,
    PixelFormat::Rgba16Float,
    PixelFormat::Rgba32Float,
    PixelFormat::Bc1Rgba,
    PixelFormat::Bc1RgbaSrgb,
    PixelFormat::Bc2Rgba,
    PixelFormat::Bc2RgbaSrgb,
    PixelFormat::Bc3Rgba,
    PixelFormat::Bc3RgbaSrgb,
    PixelFormat::Bc4RUnorm,
    PixelFormat::Depth32Float,
    PixelFormat::Depth32FloatStencil8,
];

#[test]
fn creation_clear_writes_every_cube_face_and_mip() {
    check_color_clears(MTLTextureType::TypeCube, 3, 6);
}

#[test]
fn creation_clear_writes_multisample_contents() {
    check_color_clears(MTLTextureType::Type2DMultisample, 1, 1);
}

#[test]
fn wire_pixel_format_inverts_mtl_pixel_format_for_every_format() {
    for format in ALL {
        assert_eq!(
            wire_pixel_format(mtl_pixel_format(format)),
            Some(format),
            "{format:?} round-trips through its Metal format"
        );
    }
}

#[test]
fn wire_pixel_format_declines_a_format_mtld3d_never_creates() {
    assert_eq!(wire_pixel_format(MTLPixelFormat::RGB10A2Unorm), None);
    assert_eq!(wire_pixel_format(MTLPixelFormat::Invalid), None);
}

#[test]
fn only_uncompressed_colour_formats_are_resolvable() {
    let resolvable: Vec<PixelFormat> = ALL
        .into_iter()
        .filter(|format| is_resolvable_color_format(*format))
        .collect();
    assert_eq!(
        resolvable.len(),
        20,
        "20 uncompressed colour formats: {resolvable:?}"
    );
    assert!(!is_resolvable_color_format(PixelFormat::Bc1Rgba));
    assert!(!is_resolvable_color_format(PixelFormat::Bc4RUnorm));
    assert!(!is_resolvable_color_format(PixelFormat::Depth32Float));
    assert!(is_resolvable_color_format(PixelFormat::Rgba32Float));
    assert!(is_resolvable_color_format(PixelFormat::A8Unorm));
}

/// Clear to a visible colour, then zero, without relying on allocation contents.
fn check_color_clears(texture_type: MTLTextureType, levels: usize, slices: usize) {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device for the clear test");
    let queue = device.newCommandQueue().expect("a clear-test queue");
    queue.setLabel(Some(&NSString::from_str("mtld3d-test-clear-queue")));
    let texture = test_texture(&device, texture_type, levels);
    // SAFETY: the retained queue stays alive until all clears and reads finish.
    let queue_handle =
        unsafe { MetalHandle::<MTLCommandQueueKind>::new(Retained::as_ptr(&queue) as u64) };
    // SAFETY: the retained texture stays alive until all clears and reads finish.
    let texture_handle =
        unsafe { MetalHandle::<MTLTextureKind>::new(Retained::as_ptr(&texture) as u64) };
    let magenta = MTLClearColor {
        red: 1.0,
        green: 0.0,
        blue: 1.0,
        alpha: 1.0,
    };
    for (color, expected) in [(magenta, [255, 0, 255, 255]), (TRANSPARENT_BLACK, [0; 4])] {
        clear_new_color_textures(queue_handle, &[texture_handle], color);
        for slice in 0..slices {
            for level in 0..levels {
                assert_eq!(
                    read_first_pixel(&queue, &texture, slice, level),
                    expected,
                    "slice {slice}, level {level} holds the requested clear colour",
                );
            }
        }
    }
}

/// A small private colour texture with the requested shape.
fn test_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    texture_type: MTLTextureType,
    levels: usize,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(texture_type);
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    // SAFETY: positive dimensions within every supported device's limits.
    unsafe { desc.setWidth(16) };
    // SAFETY: the cube is square, and the other shapes are 2D.
    unsafe { desc.setHeight(16) };
    // SAFETY: callers request one or three levels, both within the 16x16 chain.
    unsafe { desc.setMipmapLevelCount(levels) };
    if texture_type == MTLTextureType::Type2DMultisample {
        // SAFETY: every supported GPU has 4x MSAA; this shape has one mip.
        unsafe { desc.setSampleCount(4) };
    }
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setUsage(MTLTextureUsage::RenderTarget);
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("test texture");
    texture.setLabel(Some(&NSString::from_str("mtld3d-test-clear-texture")));
    texture
}

/// Read one BGRA pixel after the preceding clear, resolving MSAA first.
fn read_first_pixel(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
    slice: usize,
    level: usize,
) -> [u8; 4] {
    let device = queue.device();
    let cmd = queue.commandBuffer().expect("readback command buffer");
    cmd.setLabel(Some(&NSString::from_str("mtld3d-test-clear-readback")));
    let resolved;
    let source = if texture.sampleCount() > 1 {
        resolved = test_texture(&device, MTLTextureType::Type2D, 1);
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: colour attachment zero is valid on every render pass.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(texture));
        color.setResolveTexture(Some(&resolved));
        color.setLoadAction(MTLLoadAction::Load);
        color.setStoreAction(MTLStoreAction::StoreAndMultisampleResolve);
        let encoder = cmd
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("resolve encoder");
        encoder.setLabel(Some(&NSString::from_str("mtld3d-test-clear-resolve")));
        encoder.endEncoding();
        &*resolved
    } else {
        texture
    };
    // A 256-byte row accommodates both Apple and Intel texture-copy alignment.
    let buffer = device
        .newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
        .expect("readback buffer");
    buffer.setLabel(Some(&NSString::from_str("mtld3d-test-clear-pixel")));
    let blit = cmd.blitCommandEncoder().expect("readback blit");
    blit.setLabel(Some(&NSString::from_str("mtld3d-test-clear-copy")));
    // SAFETY: the source subresource exists and is at least 1x1. The destination
    // holds the whole aligned row and remains alive until GPU completion.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            source,
            slice,
            level,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: 1, height: 1, depth: 1 },
            &buffer,
            0,
            256,
            256,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    assert_eq!(
        cmd.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cmd.error()
    );
    // SAFETY: the shared buffer holds at least four initialized bytes after
    // the completed GPU copy and is retained until this read returns.
    unsafe { buffer.contents().cast::<[u8; 4]>().read() }
}
