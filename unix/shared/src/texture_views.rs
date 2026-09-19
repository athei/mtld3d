//! Texture roles sharing one resource allocation and a unique set of owned retains.

use crate::mtl_handle::{MTLTextureKind, MetalHandle};

/// Attachment and sampling handles returned by one texture creation.
///
/// Equal non-null handles share one canonical retain. Every distinct non-null
/// handle owns exactly one retain, even when several roles name it.
#[repr(C, align(8))]
pub struct TextureViews {
    pub linear: MetalHandle<MTLTextureKind>,
    pub srgb: MetalHandle<MTLTextureKind>,
    pub sample_linear: MetalHandle<MTLTextureKind>,
    pub sample_srgb: MetalHandle<MTLTextureKind>,
}

impl TextureViews {
    pub const EMPTY: Self = Self {
        linear: MetalHandle::NULL,
        srgb: MetalHandle::NULL,
        sample_linear: MetalHandle::NULL,
        sample_srgb: MetalHandle::NULL,
    };

    /// Iterate the canonical retains once, independent of role aliasing.
    pub fn owned_handles(&self) -> impl Iterator<Item = MetalHandle<MTLTextureKind>> {
        let handles = [self.linear, self.srgb, self.sample_linear, self.sample_srgb];
        handles
            .into_iter()
            .enumerate()
            .filter_map(move |(index, handle)| {
                (!handle.is_null() && !handles[..index].contains(&handle)).then_some(handle)
            })
    }
}

const _: () = {
    assert!(core::mem::size_of::<TextureViews>() == 32);
    assert!(core::mem::align_of::<TextureViews>() == 8);
};

#[cfg(test)]
mod tests;
