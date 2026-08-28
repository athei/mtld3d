//! Which texture uploads the GPU upload pass serves, and how it decodes them.
//!
//! Two upload shapes cannot ride `copyFromBuffer:toTexture:` as they are:
//! a packed 16-bit staging feeding a `Bgra8Unorm` texture on a device
//! without the native packed formats (2 bpp source, 4 bpp destination), and
//! any mip whose row pitch is below Metal's
//! `minimumLinearTextureAlignmentForPixelFormat:` (16 bytes on Apple
//! Silicon, 256 on Mac2, which puts every small mip of a mipped texture
//! under it). Both are served by a render pass whose fragment function reads
//! the staging slab as a buffer argument, which is bound by neither the row
//! alignment nor the source layout.
//!
//! The destination has to carry `RenderTarget` usage for that, so the
//! selection is frozen at texture-create time: [`needs_render_target`] is the
//! create-side predicate and is a superset of the per-upload
//! [`upload_decode`], so an upload that resolves a decode is always looking
//! at a texture that can be an attachment.

use mtld3d_shared::mtl::PixelFormat;
use mtld3d_types::{
    D3DFMT_A1R5G5B5, D3DFMT_A4R4G4B4, D3DFMT_A8R8G8B8, D3DFMT_R5G6B5, D3DFMT_X1R5G5B5,
};

/// Source layout the upload quad's fragment function decodes.
///
/// The discriminants are the wire values the shader switches on; keep them in
/// step with the `decode` cases in the unix-side upload-quad MSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum UploadDecode {
    /// 2 bpp `R5G6B5` widened to `Bgra8Unorm`, alpha forced opaque.
    R5G6B5 = 0,
    /// 2 bpp `A1R5G5B5` widened to `Bgra8Unorm`.
    A1R5G5B5 = 1,
    /// 2 bpp `A4R4G4B4` widened to `Bgra8Unorm`.
    A4R4G4B4 = 2,
    /// Verbatim copy of a 4-byte BGRA texel into the matching `Bgra8Unorm`.
    CopyBgra8 = 3,
    /// 2 bpp `X1R5G5B5` widened to `Bgra8Unorm`, alpha forced opaque.
    X1R5G5B5 = 4,
}

impl UploadDecode {
    /// The `decode` value the fragment function reads.
    #[must_use]
    pub const fn wire(self) -> u32 {
        self as u32
    }

    /// Source bytes per texel this decode addresses the staging slab with.
    #[must_use]
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R5G6B5 | Self::A1R5G5B5 | Self::A4R4G4B4 | Self::X1R5G5B5 => 2,
            Self::CopyBgra8 => 4,
        }
    }
}

/// Decode for an upload of `src_d3d_format` into a `gpu_format` texture.
///
/// `Some` for the packed 16-bit formats a device without native support
/// backs with `Bgra8Unorm`, and for the one uncompressed colour
/// format whose Metal counterpart is renderable, carries no sampler swizzle,
/// and stores its channels one unorm byte each. Everything else is `None`
/// and keeps the blit upload (with its CPU repack when the pitch is under
/// the alignment): a compressed destination cannot be an attachment at all,
/// and a swizzled one cannot be handed out as one without losing the
/// swizzle.
#[must_use]
pub const fn upload_decode(src_d3d_format: u32, gpu_format: PixelFormat) -> Option<UploadDecode> {
    if !matches!(gpu_format, PixelFormat::Bgra8Unorm) {
        return None;
    }
    match src_d3d_format {
        D3DFMT_R5G6B5 => Some(UploadDecode::R5G6B5),
        D3DFMT_A1R5G5B5 => Some(UploadDecode::A1R5G5B5),
        D3DFMT_X1R5G5B5 => Some(UploadDecode::X1R5G5B5),
        D3DFMT_A4R4G4B4 => Some(UploadDecode::A4R4G4B4),
        D3DFMT_A8R8G8B8 => Some(UploadDecode::CopyBgra8),
        _ => None,
    }
}

/// True when a decode reads a layout the destination texture cannot hold verbatim.
///
/// Such an upload has no blit form at all, so it takes the pass whatever the
/// row pitch is; a verbatim-copy decode only takes it when the pitch is under
/// the linear-texture alignment.
#[must_use]
pub const fn is_expansion(decode: UploadDecode) -> bool {
    matches!(
        decode,
        UploadDecode::R5G6B5
            | UploadDecode::A1R5G5B5
            | UploadDecode::A4R4G4B4
            | UploadDecode::X1R5G5B5
    )
}

/// True when this (source, GPU) format pair uploads through an expansion.
///
/// The create paths use it to reject a packed 16-bit render target on a
/// device that has no such format: the texture is BGRA8 underneath, so a
/// lockable render target in that format would pair a 16-bit CPU staging
/// with a 32-bit surface across the readback blits.
#[must_use]
pub const fn is_expanded_upload(src_d3d_format: u32, gpu_format: PixelFormat) -> bool {
    match upload_decode(src_d3d_format, gpu_format) {
        Some(decode) => is_expansion(decode),
        None => false,
    }
}

/// Create-time predicate: does this texture need `RenderTarget` usage for its uploads?
///
/// `width` and `levels` are the texture's level-0 width and mip count;
/// `min_linear_texture_align` is the device's linear-texture row alignment.
/// The smallest mip's tight pitch bounds every other mip's, and a real
/// staging pitch is never below the tight one, so a texture this returns
/// `false` for can never present an upload the pass would have taken.
#[must_use]
pub const fn needs_render_target(
    src_d3d_format: u32,
    gpu_format: PixelFormat,
    width: u32,
    levels: u32,
    min_linear_texture_align: u32,
) -> bool {
    let Some(decode) = upload_decode(src_d3d_format, gpu_format) else {
        return false;
    };
    if is_expansion(decode) {
        return true;
    }
    let shift = if levels == 0 { 0 } else { levels - 1 };
    let smallest = if width >> shift == 0 {
        1
    } else {
        width >> shift
    };
    smallest * decode.bytes_per_texel() < min_linear_texture_align
}

#[cfg(test)]
mod tests;
