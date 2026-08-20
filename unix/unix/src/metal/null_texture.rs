//! Shared opaque-black textures for unbound-but-declared pixel-shader samplers.
//!
//! A D3D9 pixel shader may declare a sampler (`dcl_2d`/`dcl_cube`/`dcl_volume`)
//! yet the game binds no texture to that stage. The spec requires such a sample
//! to read opaque black `(0, 0, 0, 1)`, and Metal requires every declared
//! `[[texture(n)]]` argument to be bound. The PE side detects the case and emits
//! [`CommandType::SetFragmentNullTexture`]; this module supplies the 1×1
//! opaque-black texture of the matching type plus a default sampler to bind.
//!
//! [`CommandType::SetFragmentNullTexture`]: mtld3d_shared::CommandType::SetFragmentNullTexture

use std::sync::OnceLock;

use mtld3d_shared::NullTextureKind;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResource, MTLSamplerDescriptor, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::LOG_TARGET;

/// Handles to the three opaque-black textures and their default sampler.
///
/// Raw pointers (`Retained::into_raw`) so the set is `Copy`/`Send`/`Sync` and
/// caches in a `OnceLock`; the objects leak for the process lifetime, which is
/// also the device's. Mirrors `present::PresentPipelines`.
#[derive(Clone, Copy)]
pub struct NullTextures {
    texture_2d: u64,
    texture_cube: u64,
    texture_3d: u64,
    sampler: u64,
}

impl NullTextures {
    /// The black-texture handle whose type matches `kind`.
    #[must_use]
    pub const fn texture(&self, kind: NullTextureKind) -> u64 {
        match kind {
            NullTextureKind::Texture2D => self.texture_2d,
            NullTextureKind::TextureCube => self.texture_cube,
            NullTextureKind::Texture3D => self.texture_3d,
        }
    }

    /// The shared default-sampler handle.
    #[must_use]
    pub const fn sampler(&self) -> u64 {
        self.sampler
    }
}

static NULL_TEXTURES: OnceLock<NullTextures> = OnceLock::new();

/// Lazily create and cache the opaque-black textures + default sampler.
///
/// Called from the command loop the first time a draw binds a null texture.
/// Returns `None` (with an error at the failure site) if any Metal object
/// cannot be created; the caller then leaves the argument unbound, the same
/// state as before this path existed.
pub fn ensure(device: &ProtocolObject<dyn MTLDevice>) -> Option<NullTextures> {
    if let Some(r) = NULL_TEXTURES.get() {
        return Some(*r);
    }
    let created = create(device)?;
    Some(*NULL_TEXTURES.get_or_init(|| created))
}

fn create(device: &ProtocolObject<dyn MTLDevice>) -> Option<NullTextures> {
    let texture_2d = make_black_texture(device, MTLTextureType::Type2D, 1)?;
    let texture_cube = make_black_texture(device, MTLTextureType::TypeCube, 6)?;
    let texture_3d = make_black_texture(device, MTLTextureType::Type3D, 1)?;

    let sampler_desc = MTLSamplerDescriptor::new();
    let Some(sampler) = device.newSamplerStateWithDescriptor(&sampler_desc) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "null texture: default sampler creation failed; unbound declared samplers stay unbound",
        );
        return None;
    };

    Some(NullTextures {
        texture_2d: Retained::into_raw(texture_2d) as u64,
        texture_cube: Retained::into_raw(texture_cube) as u64,
        texture_3d: Retained::into_raw(texture_3d) as u64,
        sampler: Retained::into_raw(sampler) as u64,
    })
}

/// A 1×1 (per slice) `RGBA8Unorm` texture filled with opaque black.
///
/// `slices` is 6 for a cube (one per face), 1 otherwise. Shared storage so the
/// pixel can be written from the CPU; the four bytes are `(R, G, B, A) =
/// (0, 0, 0, 255)`, which samples as `(0, 0, 0, 1)`.
fn make_black_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    texture_type: MTLTextureType,
    slices: usize,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(texture_type);
    desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setWidth(1) };
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setHeight(1) };
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setDepth(1) };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(MTLStorageMode::Shared);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str("mtld3d-null-black");
    texture.setLabel(Some(&label));

    let black: [u8; 4] = [0, 0, 0, 255];
    let black_ptr = core::ptr::NonNull::from(&black).cast::<core::ffi::c_void>();
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    for slice in 0..slices {
        // SAFETY: `black` is a 4-byte buffer, `bytesPerRow`/`bytesPerImage` are
        // 4 for the 1×1 region, and `slice` is within `slices` (the descriptor's
        // array length), so the write stays inside the level's storage.
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region, 0, slice, black_ptr, 4, 4,
            );
        }
    }
    Some(texture)
}
