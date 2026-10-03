use std::{cell::Cell, time::Duration};

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
    REFUSED_CREATE_BACKOFF, TRANSPARENT_BLACK, clear_new_color_textures,
    is_resolvable_color_format, mtl_pixel_format, retry_refused_create, retry_with_backoff,
    wire_pixel_format,
};

/// Every wire pixel format, in declaration order.
const ALL: [PixelFormat; 34] = [
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
    PixelFormat::Rg16Snorm,
    PixelFormat::R32Float,
    PixelFormat::Rg16Float,
    PixelFormat::Rgba8Unorm,
    PixelFormat::Rgba8UnormSrgb,
    PixelFormat::Rgba8Snorm,
    PixelFormat::Bgra8Unorm,
    PixelFormat::Bgra8UnormSrgb,
    PixelFormat::Rgb10A2Unorm,
    PixelFormat::Bgr10A2Unorm,
    PixelFormat::Rg32Float,
    PixelFormat::Rgba16Unorm,
    PixelFormat::Rgba16Snorm,
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
fn batch_creation_keeps_views_and_ordered_clears_after_pool_drain() {
    use mtld3d_shared::{
        TextureCreateDesc,
        mtl::{StorageMode, Swizzle, TextureCreateFlags, TextureUsage},
        texture_views::TextureViews,
    };
    use objc2::rc::autoreleasepool;

    autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device for texture batch");
        let queue = device.newCommandQueue().expect("texture batch queue");
        // SAFETY: queue stays retained through creation and all ordered readbacks.
        let queue_handle = unsafe { MetalHandle::new(Retained::as_ptr(&queue) as u64) };
        let descriptors = [false, true].map(|cube| TextureCreateDesc {
            tex_id: 1 + u64::from(cube),
            width: 16,
            height: 16,
            depth: 1,
            levels: 3,
            pixel_format: PixelFormat::Bgra8Unorm,
            storage_mode: StorageMode::Private,
            flags: TextureCreateFlags::CLEAR_ON_CREATE
                | if cube {
                    TextureCreateFlags::TYPE_CUBE
                } else {
                    TextureCreateFlags::empty()
                },
            swizzle_r: Swizzle::Red,
            swizzle_g: Swizzle::Green,
            swizzle_b: Swizzle::Blue,
            swizzle_a: Swizzle::Alpha,
            usage_flags: TextureUsage::RENDER_TARGET,
        });
        let mut views = [TextureViews::EMPTY; 2];
        let mut clears = super::TextureClearBatch::new();
        for (descriptor, view) in descriptors.iter().zip(&mut views) {
            assert!(autoreleasepool(|_| super::create_textures(
                &device,
                core::slice::from_ref(descriptor),
                core::slice::from_mut(view),
                &mut clears,
            )));
        }
        // The batch must own these resources even when the texture cache releases
        // its canonical handles before the frame reaches submission.
        for view in &views {
            for handle in view.owned_handles() {
                super::destroy_texture(handle.raw());
            }
        }
        let textures = clears.textures.clone();
        assert_eq!(textures.len(), 2, "batch retains both released textures");
        let committed = autoreleasepool(|_| clears.commit(queue_handle, TRANSPARENT_BLACK))
            .expect("two creations share one actual initialization command buffer");
        assert!(clears.commit(queue_handle, TRANSPARENT_BLACK).is_none());
        for (index, (view, texture)) in views.iter().zip(&textures).enumerate() {
            assert_eq!(texture.mipmapLevelCount(), 3);
            for slice in 0..if index == 0 { 1 } else { 6 } {
                for level in 0..3 {
                    assert_eq!(read_first_pixel(&queue, texture, slice, level), [0; 4]);
                }
            }
            let red = MTLClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            };
            clear_new_color_textures(queue_handle, &[view.linear], red);
            assert_eq!(read_first_pixel(&queue, texture, 0, 0), [0, 0, 255, 255]);
        }
        assert_eq!(committed.status(), MTLCommandBufferStatus::Completed);
        assert!(super::create_textures(&device, &[], &mut [], &mut clears));
        assert!(clears.commit(queue_handle, TRANSPARENT_BLACK).is_none());
    });
}

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
    assert_eq!(wire_pixel_format(MTLPixelFormat::RGB10A2Uint), None);
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
        25,
        "25 uncompressed colour formats: {resolvable:?}"
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

#[test]
fn texture_roles_transfer_only_one_retain_for_each_native_object() {
    use objc2::runtime::NSObjectProtocol;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let texture = device.newTextureWithDescriptor(&desc).expect("texture");
    let before = texture.retainCount();
    let views = super::mint_texture_views([
        Some(texture.clone()),
        Some(texture.clone()),
        Some(texture.clone()),
        Some(texture.clone()),
    ]);
    assert_eq!(views.owned_handles().count(), 1);
    assert_eq!(texture.retainCount(), before + 1);
    for handle in views.owned_handles() {
        super::destroy_texture(handle.raw());
    }
    assert_eq!(texture.retainCount(), before);
}

#[test]
fn render_target_views_preserve_usage_shape_and_role_swizzles() {
    use mtld3d_shared::{
        TextureCreateDesc,
        mtl::{StorageMode, Swizzle, TextureCreateFlags, TextureUsage},
    };

    use crate::metal::handle::IntoRetained;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    for cube in [false, true] {
        for format in [
            PixelFormat::R16Float,
            PixelFormat::Rg16Unorm,
            PixelFormat::Rg16Float,
            PixelFormat::R32Float,
            PixelFormat::Rg32Float,
            PixelFormat::Bgra8Unorm,
            PixelFormat::Rgba8Unorm,
        ] {
            let flags = TextureCreateFlags::HAS_SWIZZLE
                | if cube {
                    TextureCreateFlags::TYPE_CUBE
                } else {
                    TextureCreateFlags::empty()
                };
            let desc = TextureCreateDesc {
                tex_id: 1,
                width: 16,
                height: 16,
                depth: 1,
                levels: 3,
                pixel_format: format,
                storage_mode: StorageMode::Private,
                flags,
                swizzle_r: Swizzle::Red,
                swizzle_g: Swizzle::Green,
                swizzle_b: Swizzle::One,
                swizzle_a: Swizzle::One,
                usage_flags: TextureUsage::RENDER_TARGET,
            };
            let views = super::create_texture(&device, &desc).expect("required views");
            let base = views.linear.into_retained().expect("base");
            let sample = views.sample_linear.into_retained().expect("sample");
            assert_ne!(views.linear, views.sample_linear);
            assert_eq!(base.mipmapLevelCount(), 3);
            assert_eq!(sample.mipmapLevelCount(), 3);
            assert_eq!(base.textureType(), sample.textureType());
            assert_eq!(base.swizzle(), super::IDENTITY_SWIZZLE);
            assert_eq!(sample.swizzle().alpha, objc2_metal::MTLTextureSwizzle::One);
            assert_eq!(base.usage(), super::texture_usage(&device, true, true));
            if format.srgb_twin().is_some() {
                assert_eq!(views.owned_handles().count(), 4);
                let srgb = views.srgb.into_retained().expect("sRGB attachment");
                let sample_srgb = views.sample_srgb.into_retained().expect("sRGB sample");
                assert_eq!(srgb.swizzle(), super::IDENTITY_SWIZZLE);
                assert_eq!(
                    sample_srgb.swizzle().alpha,
                    objc2_metal::MTLTextureSwizzle::One
                );
            } else {
                assert_eq!(views.owned_handles().count(), 2);
                assert!(views.srgb.is_null());
                assert!(views.sample_srgb.is_null());
            }
            for handle in views.owned_handles() {
                super::destroy_texture(handle.raw());
            }
        }
    }
}

#[test]
fn required_view_failure_drops_every_temporary_owner_before_minting() {
    use objc2::runtime::NSObjectProtocol;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let texture = device.newTextureWithDescriptor(&desc).expect("texture");
    let before = texture.retainCount();
    for fail_srgb_sample in [false, true] {
        let result = super::assemble_texture_views(
            texture.clone(),
            true,
            true,
            |_, srgb| {
                if srgb == fail_srgb_sample {
                    None
                } else {
                    Some(texture.clone())
                }
            },
            |_| Some(texture.clone()),
        );
        assert!(result.is_none());
        assert_eq!(
            texture.retainCount(),
            before,
            "failed sRGB sample={fail_srgb_sample}"
        );
        let raw = Retained::as_ptr(&texture) as u64;
        assert!(!super::LIVE_TEXTURES.lock().expect("ledger").contains(&raw));
    }
}

#[test]
fn optional_srgb_refusal_keeps_the_required_linear_view() {
    use objc2::runtime::NSObjectProtocol;
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let texture = device.newTextureWithDescriptor(&desc).expect("texture");
    let before = texture.retainCount();
    let views = super::assemble_texture_views(
        texture.clone(),
        true,
        true,
        |_, srgb| {
            assert!(!srgb);
            Some(texture.clone())
        },
        |_| None,
    )
    .expect("linear fallback");
    assert!(views.srgb.is_null());
    assert!(views.sample_srgb.is_null());
    assert_eq!(views.sample_linear, views.linear);
    assert_eq!(texture.retainCount(), before + 1);
    for handle in views.owned_handles() {
        super::destroy_texture(handle.raw());
    }
    assert_eq!(texture.retainCount(), before);
}

#[test]
fn identity_roles_do_not_create_sampling_views() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    let texture = device.newTextureWithDescriptor(&desc).expect("texture");
    let views = super::assemble_texture_views(
        texture,
        true,
        false,
        |_, _| panic!("identity allocation must not create a sampling view"),
        |_| None,
    )
    .expect("identity allocation");
    assert_eq!(views.linear, views.sample_linear);
    assert_eq!(views.owned_handles().count(), 1);
    for handle in views.owned_handles() {
        super::destroy_texture(handle.raw());
    }
}

#[test]
fn a_final_refusal_is_not_retried() {
    let calls = Cell::new(0_u32);
    let mut waits = Vec::new();
    let outcome = retry_with_backoff(
        false,
        &REFUSED_CREATE_BACKOFF,
        || {
            calls.set(calls.get() + 1);
            Some(())
        },
        |delay| waits.push(delay),
    );
    assert!(outcome.value.is_none());
    assert_eq!(outcome.attempts, 0);
    assert_eq!(outcome.waited, Duration::ZERO);
    assert_eq!(
        calls.get(),
        0,
        "a device whose refusals are final is not asked again"
    );
    assert!(waits.is_empty(), "nor waited on");
}

#[test]
fn a_transient_refusal_is_retried_until_the_create_succeeds() {
    let calls = Cell::new(0_u32);
    let mut waits = Vec::new();
    let outcome = retry_with_backoff(
        true,
        &REFUSED_CREATE_BACKOFF,
        || {
            calls.set(calls.get() + 1);
            (calls.get() == 3).then_some(7_u32)
        },
        |delay| waits.push(delay),
    );
    assert_eq!(outcome.value, Some(7));
    assert_eq!(outcome.attempts, 3);
    assert_eq!(calls.get(), 3, "no attempt after the one that succeeded");
    assert_eq!(
        waits,
        REFUSED_CREATE_BACKOFF[..3],
        "each attempt follows its own wait"
    );
    assert_eq!(outcome.waited, Duration::from_millis(1 + 2 + 4));
}

#[test]
fn a_refusal_that_outlasts_the_schedule_fails_after_every_wait() {
    let calls = Cell::new(0_u32);
    let mut waits = Vec::new();
    let outcome = retry_with_backoff(
        true,
        &REFUSED_CREATE_BACKOFF,
        || {
            calls.set(calls.get() + 1);
            None::<()>
        },
        |delay| waits.push(delay),
    );
    assert!(outcome.value.is_none());
    assert_eq!(outcome.attempts, 8);
    assert_eq!(calls.get(), 8);
    assert_eq!(waits, REFUSED_CREATE_BACKOFF);
    assert_eq!(
        outcome.waited,
        Duration::from_millis(255),
        "a quarter of a second at most"
    );
}

#[test]
fn the_refused_create_backoff_doubles_from_a_millisecond() {
    assert_eq!(REFUSED_CREATE_BACKOFF[0], Duration::from_millis(1));
    for pair in REFUSED_CREATE_BACKOFF.windows(2) {
        assert_eq!(pair[1], pair[0] * 2);
    }
}

#[test]
fn a_real_gpu_fails_a_refused_create_without_asking_again() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device for the retry gate");
    if crate::metal::device::refuses_creates_transiently(&device) {
        // A VM's device takes the retries; the device tests pin the gate by
        // name, so this one has nothing to show there.
        return;
    }
    let calls = Cell::new(0_u32);
    let created = retry_refused_create(&device, "mtld3d-test", "texture", || {
        calls.set(calls.get() + 1);
        Some(())
    });
    assert!(created.is_none(), "the refusal stands on a real GPU");
    assert_eq!(calls.get(), 0, "and the create is not asked for again");
}
