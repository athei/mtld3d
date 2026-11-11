use mtld3d_core::shared_texture::SERVICE_NAME_PREFIX;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLLoadAction, MTLOrigin,
    MTLRenderPassDescriptor, MTLResourceOptions, MTLSize, MTLStoreAction,
};

use super::*;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;

fn request(tex_id: u64, width: u32, usage: TextureUsage) -> ShareRequest {
    ShareRequest {
        tex_id,
        width,
        height: HEIGHT,
        pixel_format: PixelFormat::Bgra8Unorm,
        usage,
    }
}

/// Clear `texture` to `value` in all four channels on `queue` and wait for the GPU.
fn clear(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
    value: u8,
) {
    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    // SAFETY: index 0 of a fresh descriptor's colour attachments.
    let color0 = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setTexture(Some(texture));
    color0.setLoadAction(MTLLoadAction::Clear);
    color0.setStoreAction(MTLStoreAction::Store);
    let level = f64::from(value) / 255.0;
    color0.setClearColor(MTLClearColor {
        red: level,
        green: level,
        blue: level,
        alpha: level,
    });
    let cmd = queue.commandBuffer().expect("a command buffer");
    let encoder = cmd
        .renderCommandEncoderWithDescriptor(&pass)
        .expect("a render encoder");
    encoder.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
}

/// Read the first texel of `texture` on `queue`.
fn first_texel(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
) -> [u8; 4] {
    let width = usize::try_from(WIDTH).expect("u32 fits usize");
    let height = usize::try_from(HEIGHT).expect("u32 fits usize");
    let row = width * 4;
    let buffer = device
        .newBufferWithLength_options(row * height, MTLResourceOptions::StorageModeShared)
        .expect("a readback buffer");
    let cmd = queue.commandBuffer().expect("a command buffer");
    let blit = cmd.blitCommandEncoder().expect("a blit encoder");
    // SAFETY: the copy covers exactly the texture's one level and the buffer
    // holds `row * height` bytes.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width, height, depth: 1 },
            &buffer,
            0,
            row,
            row * height,
        );
    }
    blit.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    let mut texel = [0u8; 4];
    // SAFETY: a shared-storage buffer of at least four bytes, read after the
    // command buffer that fills it completed.
    unsafe {
        core::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr().cast::<u8>(),
            texel.as_mut_ptr(),
            4,
        );
    }
    texel
}

/// A created texture opens by its registered name as the same storage, in both directions.
///
/// `create` runs the Metal and bootstrap calls the other runtime's allocation
/// runs, `open` is what a D3D9 open of that runtime's handle runs, and what
/// one command queue renders into either texture the other queue reads from
/// the other texture.
#[test]
fn a_created_texture_opens_by_name_as_the_same_storage() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let parked = ParkedTextures::new();
    let name = create(
        &device,
        &parked,
        &request(0x7001, WIDTH, TextureUsage::RENDER_TARGET),
    )
    .expect("the shared texture is created and registered");
    assert!(name_str(&name).starts_with(SERVICE_NAME_PREFIX));
    let created = parked.take(0x7001).expect("create parks the texture");
    assert!(
        parked.take(0x7001).is_none(),
        "a parked texture is taken once"
    );

    assert!(open(
        &device,
        &parked,
        &request(0x7002, WIDTH, TextureUsage::RENDER_TARGET),
        &name
    ));
    let opened = parked.take(0x7002).expect("open parks the texture");
    assert_eq!(
        (opened.width(), opened.height()),
        (
            usize::try_from(WIDTH).expect("u32 fits usize"),
            usize::try_from(HEIGHT).expect("u32 fits usize")
        )
    );
    assert_eq!(opened.pixelFormat(), MTLPixelFormat::BGRA8Unorm);

    // Two queues, as two runtimes have.
    let producer = device.newCommandQueue().expect("a queue");
    let consumer = device.newCommandQueue().expect("a queue");
    clear(&producer, &created, 0x40);
    assert_eq!(first_texel(&device, &consumer, &opened), [0x40; 4]);
    clear(&producer, &opened, 0xc0);
    assert_eq!(first_texel(&device, &consumer, &created), [0xc0; 4]);

    // Every colour texture of this layer gets an sRGB twin view, which a
    // shared texture has to allow although it was not created for views.
    assert!(
        opened
            .newTextureViewWithPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB)
            .is_some()
    );
}

#[test]
fn an_open_refuses_another_extent_and_an_unknown_name() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let parked = ParkedTextures::new();
    let name = create(
        &device,
        &parked,
        &request(0x7101, WIDTH, TextureUsage::empty()),
    )
    .expect("the shared texture is created and registered");
    let _created = parked.take(0x7101).expect("create parks the texture");

    assert!(!open(
        &device,
        &parked,
        &request(0x7102, WIDTH * 2, TextureUsage::empty()),
        &name
    ));
    assert!(
        parked.take(0x7102).is_none(),
        "a refused open parks nothing"
    );

    let mut unknown = [0u8; SERVICE_NAME_LEN];
    unknown[..22].copy_from_slice(b"MTLD3D_shared_missing_");
    assert!(!open(
        &device,
        &parked,
        &request(0x7103, WIDTH, TextureUsage::empty()),
        &unknown
    ));

    let unterminated = [b'x'; SERVICE_NAME_LEN];
    assert!(!open(
        &device,
        &parked,
        &request(0x7104, WIDTH, TextureUsage::empty()),
        &unterminated
    ));
}

#[test]
fn a_forgotten_texture_is_no_longer_parked() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let parked = ParkedTextures::new();
    create(
        &device,
        &parked,
        &request(0x7201, WIDTH, TextureUsage::empty()),
    )
    .expect("the shared texture is created and registered");
    parked.forget(0x7201);
    assert!(parked.take(0x7201).is_none());
    // Forgetting an id that was never parked, or twice, changes nothing.
    parked.forget(0x7201);
    parked.forget(0x7299);
}

#[test]
fn a_record_that_drops_releases_what_is_still_parked() {
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let parked = ParkedTextures::new();
    let name = create(
        &device,
        &parked,
        &request(0x7301, WIDTH, TextureUsage::empty()),
    )
    .expect("the shared texture is created and registered");
    assert!(open(
        &device,
        &parked,
        &request(0x7302, WIDTH, TextureUsage::empty()),
        &name
    ));
    let opened = parked.take(0x7302).expect("open parks the texture");
    // The texture `create` parked goes with the table; the storage lives on
    // through the opened wrapper, which still reads back.
    drop(parked);
    let queue = device.newCommandQueue().expect("a queue");
    clear(&queue, &opened, 0x80);
    assert_eq!(first_texel(&device, &queue, &opened), [0x80; 4]);
}
