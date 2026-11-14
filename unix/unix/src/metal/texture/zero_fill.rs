//! Creation-time zero fill of sampled textures (`texture.zeroOnCreate`).
//!
//! Metal leaves a new `Private` texture's contents undefined. A level the
//! application never uploads, or whose upload was dropped, then samples
//! whatever the allocation held before, which on a block-compressed format is
//! rainbow block garbage. D3D9 backs a managed texture with zeroed system
//! memory, so the same level reads black there. The fill is one blit per row
//! band of every subresource from a shared zero buffer, encoded into the
//! creation-time initialisation command buffer, which commits ahead of the
//! frame that uploads into the texture (see `TextureClearBatch`).
//!
//! The band plan is pure ([`plan`]) so its geometry stays under test; the
//! encoding is in [`encode`].

use mtld3d_shared::mtl::{BlockLayout, PixelFormat};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLDevice, MTLOrigin, MTLResource, MTLResourceOptions, MTLSize, MTLTexture, MTLTextureType,
};

#[cfg(test)]
mod tests;

/// Bytes one band of a fill reads from the zero buffer, at most.
///
/// A band is a whole number of block rows, so a row wider than this still
/// gets a band of one row and the buffer grows to hold it (see [`plan`]).
/// 1 MiB covers a 2048-wide BC1 level in 4 bands and lives on the native
/// side, outside the 32-bit guest's address space.
pub const BAND_BYTES: usize = 1 << 20;

/// The row stride every band uses, in bytes.
///
/// Above Metal's minimum linear row alignment on every Mac (16 on Apple
/// Silicon, 256 on Mac2) and a multiple of every block and pixel size on
/// the wire, so the copy is valid whatever the tight pitch.
pub const ROW_ALIGN: usize = 256;

/// One `copyFromBuffer:toTexture:` of a fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Band {
    /// Mip level.
    pub level: u32,
    /// Array slice or cube face.
    pub slice: u32,
    /// Depth plane of a volume level, 0 otherwise.
    pub z: u32,
    /// First pixel row.
    pub y: u32,
    /// Pixel width, the level's.
    pub width: u32,
    /// Pixel height, a whole number of blocks unless it reaches the level's bottom edge.
    pub height: u32,
    /// Source row stride, a multiple of [`ROW_ALIGN`].
    pub bytes_per_row: usize,
    /// Source bytes the band reads: `bytes_per_row` times its block rows.
    pub bytes: usize,
}

/// The shape of a texture to fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Level-0 pixel width.
    pub width: u32,
    /// Level-0 pixel height.
    pub height: u32,
    /// Level-0 depth of a volume, 1 otherwise.
    pub depth: u32,
    /// Mip levels.
    pub levels: u32,
    /// Array slices, 6 for a cube.
    pub slices: u32,
}

/// Append the bands that zero `shape` in `block`'s format to `out`; return the bytes they need.
///
/// Every subresource is covered whole: each level of each slice and plane is
/// split into bands of at most [`BAND_BYTES`] of whole block rows, the last
/// band reaching the level's bottom edge. The answer is the largest band's
/// byte count, which the zero buffer has to hold. A zero extent or level
/// count plans nothing and needs nothing.
pub fn plan(shape: Shape, block: BlockLayout, out: &mut Vec<Band>) -> usize {
    let (block_w, block_h, block_bytes) = (
        block.width().max(1),
        block.height().max(1),
        block.bytes().max(1) as usize,
    );
    let mut needed = 0;
    if shape.width == 0 || shape.height == 0 {
        return 0;
    }
    for level in 0..shape.levels.min(32) {
        let width = (shape.width >> level).max(1);
        let height = (shape.height >> level).max(1);
        let planes = (shape.depth >> level).max(1);
        let row_bytes = width.div_ceil(block_w) as usize * block_bytes;
        let bytes_per_row = row_bytes.next_multiple_of(ROW_ALIGN);
        let block_rows = height.div_ceil(block_h);
        let band_rows = u32::try_from((BAND_BYTES / bytes_per_row).max(1))
            .unwrap_or(u32::MAX)
            .min(block_rows);
        for slice in 0..shape.slices.max(1) {
            for z in 0..planes {
                let mut row = 0;
                while row < block_rows {
                    let rows = band_rows.min(block_rows - row);
                    let y = row * block_h;
                    let band = Band {
                        level,
                        slice,
                        z,
                        y,
                        width,
                        height: (rows * block_h).min(height - y),
                        bytes_per_row,
                        bytes: bytes_per_row * rows as usize,
                    };
                    needed = needed.max(band.bytes);
                    out.push(band);
                    row += rows;
                }
            }
        }
    }
    needed
}

/// The shared zero buffer the fills read, grown on demand.
///
/// Owned by the batch that encodes the fills, so it lives as long as the
/// encoder and is created at the first fill.
#[derive(Default)]
pub struct ZeroSource {
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl ZeroSource {
    /// No buffer yet; the first fill creates it.
    pub const fn new() -> Self {
        Self { buffer: None }
    }

    /// A zero buffer of at least `bytes`, or `None` when the device refuses one.
    fn get(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        let held = self.buffer.as_ref().map_or(0, |buffer| buffer.length());
        if held < bytes {
            let len = bytes.max(BAND_BYTES);
            let buffer =
                device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)?;
            // SAFETY: `contents` is the CPU address of a fresh shared buffer of
            // `len` bytes that no command buffer references yet.
            unsafe {
                core::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, len);
            }
            buffer.setLabel(Some(&NSString::from_str("mtld3d-zero-fill")));
            // The old buffer stays alive in the command buffers that read it.
            self.buffer = Some(buffer);
        }
        self.buffer.as_deref()
    }
}

/// Encode the zero fill of every texture in `textures` into `cmd_buf`, one blit encoder for all.
///
/// A texture whose pixel format is not on the wire is skipped with a log
/// line: its block size is unknown, and a fill sized wrong would be
/// rejected by Metal's validation rather than fill anything. A missing
/// zero buffer or encoder leaves the textures as creation left them, which
/// is the behaviour without the key.
pub fn encode(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    textures: &[Retained<ProtocolObject<dyn MTLTexture>>],
    source: &mut ZeroSource,
    bands: &mut Vec<Band>,
) {
    if textures.is_empty() {
        return;
    }
    let mut plans = Vec::with_capacity(textures.len());
    let mut needed = 0;
    for texture in textures {
        let Some(block) = u32::try_from(texture.pixelFormat().0)
            .ok()
            .and_then(PixelFormat::from_repr)
            .map(PixelFormat::block_layout)
        else {
            mtld3d_shared::log_once_warn_by!(
                target: crate::LOG_TARGET,
                key: texture.pixelFormat().0 as u64,
                "texture.zeroOnCreate: pixel format {:?} is not on the wire; a texture of it \
                 starts with undefined contents",
                texture.pixelFormat()
            );
            continue;
        };
        let start = bands.len();
        let shape = shape_of(texture);
        needed = needed.max(plan(shape, block, bands));
        plans.push((texture, start, bands.len()));
    }
    if plans.is_empty() || needed == 0 {
        bands.clear();
        return;
    }
    let device = cmd_buf.device();
    let Some(zero) = source.get(&device, needed) else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "texture.zeroOnCreate: no {needed}-byte zero buffer; new textures start with \
             undefined contents"
        );
        bands.clear();
        return;
    };
    let Some(blit) = cmd_buf.blitCommandEncoder() else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "texture.zeroOnCreate: no blit encoder; new textures start with undefined contents"
        );
        bands.clear();
        return;
    };
    blit.setLabel(Some(&NSString::from_str("mtld3d-init-zero")));
    for (texture, start, end) in plans {
        for band in &bands[start..end] {
            // SAFETY: objc2 typed binding. The band lies inside the level by
            // construction (`plan`), its row stride is a multiple of the
            // format's block size and above the linear alignment, and the
            // zero buffer holds `needed >= band.bytes` bytes. The command
            // buffer retains both resources.
            unsafe {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                    zero,
                    0,
                    band.bytes_per_row,
                    band.bytes,
                    MTLSize {
                        width: band.width as usize,
                        height: band.height as usize,
                        depth: 1,
                    },
                    texture,
                    band.slice as usize,
                    band.level as usize,
                    MTLOrigin {
                        x: 0,
                        y: band.y as usize,
                        z: band.z as usize,
                    },
                    MTLBlitOption::empty(),
                );
            }
        }
    }
    blit.endEncoding();
    bands.clear();
}

/// The fill shape of a created texture, read back from the texture itself.
///
/// A cube reports `arrayLength() == 1` while carrying six faces, so its
/// slice count is the constant.
fn shape_of(texture: &ProtocolObject<dyn MTLTexture>) -> Shape {
    let clamp = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    let ty = texture.textureType();
    Shape {
        width: clamp(texture.width()),
        height: clamp(texture.height()),
        depth: if ty == MTLTextureType::Type3D {
            clamp(texture.depth())
        } else {
            1
        },
        levels: clamp(texture.mipmapLevelCount()),
        slices: if ty == MTLTextureType::TypeCube {
            6
        } else {
            clamp(texture.arrayLength())
        },
    }
}
