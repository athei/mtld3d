//! Unit tests for the creation-time zero-fill band plan.

use mtld3d_shared::mtl::PixelFormat;

use super::{BAND_BYTES, Band, ROW_ALIGN, Shape, plan};

fn shape(width: u32, height: u32, levels: u32) -> Shape {
    Shape {
        width,
        height,
        depth: 1,
        levels,
        slices: 1,
    }
}

fn bands(shape: Shape, format: PixelFormat) -> (Vec<Band>, usize) {
    let mut out = Vec::new();
    let needed = plan(shape, format.block_layout(), &mut out);
    (out, needed)
}

/// Every pixel row of every subresource is covered exactly once.
fn assert_covers(shape: Shape, format: PixelFormat, out: &[Band]) {
    for level in 0..shape.levels {
        let height = (shape.height >> level).max(1);
        let width = (shape.width >> level).max(1);
        for slice in 0..shape.slices {
            for z in 0..(shape.depth >> level).max(1) {
                let mut rows: Vec<&Band> = out
                    .iter()
                    .filter(|b| b.level == level && b.slice == slice && b.z == z)
                    .collect();
                rows.sort_by_key(|b| b.y);
                let mut next = 0;
                for band in rows {
                    assert_eq!(band.y, next, "{format:?} level {level} slice {slice} z {z}");
                    assert_eq!(band.width, width);
                    assert_eq!(band.bytes_per_row % ROW_ALIGN, 0);
                    assert_eq!(band.y % format.block_layout().height(), 0);
                    next += band.height;
                }
                assert_eq!(next, height, "{format:?} level {level} covered to the edge");
            }
        }
    }
}

#[test]
fn bc1_2048_full_chain_covers_every_level_in_bounded_bands() {
    let s = shape(2048, 2048, 12);
    let (out, needed) = bands(s, PixelFormat::Bc1Rgba);
    assert_covers(s, PixelFormat::Bc1Rgba, &out);
    // Level 0: 512 blocks x 8 bytes = 4096 bytes a row, 512 block rows = 2 MiB, two bands.
    assert_eq!(out.iter().filter(|b| b.level == 0).count(), 2);
    assert!(needed <= BAND_BYTES);
    assert!(out.iter().all(|b| b.bytes <= needed));
}

#[test]
fn sub_block_levels_reach_the_edge_with_one_band() {
    let s = shape(8, 8, 4);
    let (out, _) = bands(s, PixelFormat::Bc3Rgba);
    assert_covers(s, PixelFormat::Bc3Rgba, &out);
    let last = out.iter().find(|b| b.level == 3).expect("1x1 level");
    assert_eq!((last.width, last.height, last.y), (1, 1, 0));
    assert_eq!(
        last.bytes, ROW_ALIGN,
        "one block row, padded to the alignment"
    );
}

#[test]
fn non_square_uncompressed_and_cube_slices() {
    let s = Shape {
        width: 300,
        height: 7,
        depth: 1,
        levels: 3,
        slices: 6,
    };
    let (out, _) = bands(s, PixelFormat::Bgra8Unorm);
    assert_covers(s, PixelFormat::Bgra8Unorm, &out);
    assert_eq!(out.iter().filter(|b| b.level == 0).count(), 6);
    assert_eq!(
        out[0].bytes_per_row, 1280,
        "1200 tight bytes round up to 256"
    );
}

#[test]
fn volume_levels_fill_each_plane() {
    let s = Shape {
        width: 16,
        height: 16,
        depth: 4,
        levels: 3,
        slices: 1,
    };
    let (out, _) = bands(s, PixelFormat::Rgba16Float);
    assert_covers(s, PixelFormat::Rgba16Float, &out);
    assert_eq!(out.iter().filter(|b| b.level == 0).count(), 4);
    assert_eq!(out.iter().filter(|b| b.level == 2).count(), 1);
}

#[test]
fn a_row_wider_than_a_band_gets_a_band_of_its_own() {
    let s = shape(16384, 4, 1);
    let (out, needed) = bands(s, PixelFormat::Rgba32Float);
    assert_covers(s, PixelFormat::Rgba32Float, &out);
    // 16384 x 16 bytes = 256 KiB a row, four rows per 1 MiB band.
    assert_eq!(out.len(), 1);
    assert_eq!(needed, 4 * 256 * 1024);
}

#[test]
fn empty_shapes_plan_nothing() {
    for s in [shape(0, 16, 1), shape(16, 0, 1), shape(16, 16, 0)] {
        let (out, needed) = bands(s, PixelFormat::Bc1Rgba);
        assert_eq!(out, []);
        assert_eq!(needed, 0);
    }
}

mod gpu {
    use mtld3d_shared::mtl::PixelFormat;
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
        MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLOrigin,
        MTLPixelFormat, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture,
        MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
    };

    use super::super::{Band, ZeroSource, encode, plan, shape_of};

    type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

    const SCRATCH: usize = 1 << 16;

    fn make(
        device: &ProtocolObject<dyn MTLDevice>,
        ty: MTLTextureType,
        format: MTLPixelFormat,
        size: usize,
        levels: usize,
    ) -> Texture {
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(ty);
        desc.setPixelFormat(format);
        // SAFETY: a small positive extent.
        unsafe { desc.setWidth(size) };
        // SAFETY: as above.
        unsafe { desc.setHeight(size) };
        // SAFETY: a level count within the chain of that extent.
        unsafe { desc.setMipmapLevelCount(levels) };
        desc.setStorageMode(MTLStorageMode::Private);
        desc.setUsage(MTLTextureUsage::ShaderRead);
        device
            .newTextureWithDescriptor(&desc)
            .expect("test texture")
    }

    fn run(queue: &Queue, encode_into: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>)) {
        let cmd = queue.commandBuffer().expect("command buffer");
        encode_into(&cmd);
        cmd.commit();
        cmd.waitUntilCompleted();
        assert_eq!(
            cmd.status(),
            MTLCommandBufferStatus::Completed,
            "{:?}",
            cmd.error()
        );
    }

    fn origin(band: &Band) -> (MTLOrigin, MTLSize) {
        (
            MTLOrigin {
                x: 0,
                y: band.y as usize,
                z: band.z as usize,
            },
            MTLSize {
                width: band.width as usize,
                height: band.height as usize,
                depth: 1,
            },
        )
    }

    /// Write `byte` over every band of `texture` from the scratch buffer.
    fn scribble(queue: &Queue, scratch: &Buffer, texture: &Texture, bands: &[Band], byte: u8) {
        // SAFETY: the shared scratch is idle between waited command buffers.
        unsafe { core::ptr::write_bytes(scratch.contents().as_ptr().cast::<u8>(), byte, SCRATCH) };
        run(queue, |cmd| {
            let blit = cmd.blitCommandEncoder().expect("blit");
            for band in bands {
                let (at, size) = origin(band);
                // SAFETY: the band lies inside its level and the scratch holds it.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        scratch, 0, band.bytes_per_row, band.bytes, size,
                        texture, band.slice as usize, band.level as usize, at,
                    );
                }
            }
            blit.endEncoding();
        });
    }

    /// Whether every texel byte of `band` reads back as zero.
    fn band_is_zero(
        queue: &Queue,
        scratch: &Buffer,
        texture: &Texture,
        band: &Band,
        format: PixelFormat,
    ) -> bool {
        // SAFETY: as in `scribble`; 0xFF makes an unwritten byte stand out.
        unsafe { core::ptr::write_bytes(scratch.contents().as_ptr().cast::<u8>(), 0xFF, SCRATCH) };
        run(queue, |cmd| {
            let blit = cmd.blitCommandEncoder().expect("blit");
            let (at, size) = origin(band);
            // SAFETY: the band lies inside its level and the scratch holds it.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    texture, band.slice as usize, band.level as usize, at, size,
                    scratch, 0, band.bytes_per_row, band.bytes,
                );
            }
            blit.endEncoding();
        });
        let layout = format.block_layout();
        let tight = (band.width.div_ceil(layout.width()) * layout.bytes()) as usize;
        let row = band.bytes_per_row;
        // SAFETY: the completed copy wrote `band.bytes` bytes of the scratch.
        let bytes = unsafe {
            core::slice::from_raw_parts(scratch.contents().as_ptr().cast::<u8>(), band.bytes)
        };
        (0..band.bytes / row).all(|r| bytes[r * row..r * row + tight].iter().all(|&b| b == 0))
    }

    /// A private texture scribbled with 0xAB reads back zero in every subresource after `encode`.
    #[test]
    fn encode_zeroes_every_subresource() {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return;
        };
        let queue = device.newCommandQueue().expect("zero-fill test queue");
        let scratch = device
            .newBufferWithLength_options(SCRATCH, MTLResourceOptions::StorageModeShared)
            .expect("scratch buffer");
        let cases = [
            (
                make(
                    &device,
                    MTLTextureType::Type2D,
                    MTLPixelFormat::BC1_RGBA,
                    64,
                    7,
                ),
                PixelFormat::Bc1Rgba,
            ),
            (
                make(
                    &device,
                    MTLTextureType::TypeCube,
                    MTLPixelFormat::BGRA8Unorm,
                    8,
                    4,
                ),
                PixelFormat::Bgra8Unorm,
            ),
        ];
        let mut source = ZeroSource::default();
        let mut scratch_bands = Vec::new();
        for (texture, format) in &cases {
            let mut bands = Vec::new();
            plan(shape_of(texture), format.block_layout(), &mut bands);
            scribble(&queue, &scratch, texture, &bands, 0xAB);
            assert!(
                !band_is_zero(&queue, &scratch, texture, &bands[0], *format),
                "scribbled"
            );
            run(&queue, |cmd| {
                encode(
                    cmd,
                    core::slice::from_ref(texture),
                    &mut source,
                    &mut scratch_bands,
                );
            });
            assert!(
                scratch_bands.is_empty(),
                "band scratch cleared for the next batch"
            );
            for band in &bands {
                assert!(
                    band_is_zero(&queue, &scratch, texture, band, *format),
                    "{format:?} {band:?}"
                );
            }
        }
    }
}
