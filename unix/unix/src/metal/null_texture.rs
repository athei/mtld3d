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

use mtld3d_shared::{MetalHandle, NullTextureKind, mtl_handle::MTLSamplerStateKind};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLRegion, MTLRenderPassDescriptor, MTLResource, MTLSamplerDescriptor,
    MTLSamplerState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

use crate::{
    LOG_TARGET,
    metal::{device::cpu_written_texture_storage, handle::BorrowRetained},
};

/// Handles to the four fallback textures and their default sampler.
///
/// Raw pointers (`Retained::into_raw`) so the set is `Copy`/`Send`/`Sync` and
/// caches in a `OnceLock`; the objects leak for the process lifetime, which is
/// also the device's. Mirrors `present::PresentPipelines`.
#[derive(Clone, Copy)]
pub struct NullTextures {
    texture_2d: u64,
    texture_cube: u64,
    texture_3d: u64,
    depth_2d: u64,
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
            NullTextureKind::Depth2D => self.depth_2d,
        }
    }

    /// The shared default-sampler handle.
    #[must_use]
    pub const fn sampler(&self) -> u64 {
        self.sampler
    }
}

/// The set `create` built, or `None` once it failed.
///
/// Process-wide: the textures and sampler are built on the one `MTLDevice`
/// the process uses, which every D3D device shares. A failed creation is
/// kept too, so a draw that binds a null texture after it asks Metal once
/// rather than on every draw.
static NULL_TEXTURES: OnceLock<Option<NullTextures>> = OnceLock::new();

/// The shared default sampler, borrowed for as long as the caller needs it.
///
/// What a slot gets when the draw names no usable sampler of its own: the
/// sampler a null-texture bind installs, and the substitute for a sampler
/// state the device declined to create. `None` only when the default sampler
/// itself cannot be created. The signature is safe and the reference
/// `'static` because this module, not the caller, owns the lifetime: the
/// retain `create` gave up is never taken back, so the sampler outlives the
/// process.
pub fn default_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<&'static ProtocolObject<dyn MTLSamplerState>> {
    let null = ensure(device)?;
    // SAFETY: the handle came from `create`'s `Retained::into_raw`, alive for
    // the process lifetime.
    let handle = unsafe { MetalHandle::<MTLSamplerStateKind>::new(null.sampler()) };
    // SAFETY: that retain is never released, so it outlives every borrow.
    unsafe { handle.borrow_retained() }
}

/// Lazily create and cache the opaque-black textures + default sampler.
///
/// Called from the command loop the first time a draw binds a null texture.
/// Returns `None` (with a warning at the failure site, logged once) if any
/// Metal object cannot be created; the caller then leaves the argument
/// unbound. The failure is cached with the set, so later draws are not
/// retried against Metal.
pub fn ensure(device: &ProtocolObject<dyn MTLDevice>) -> Option<NullTextures> {
    *NULL_TEXTURES.get_or_init(|| create(device))
}

fn create(device: &ProtocolObject<dyn MTLDevice>) -> Option<NullTextures> {
    let texture_2d = make_black_texture(device, MTLTextureType::Type2D, 1)?;
    let texture_cube = make_black_texture(device, MTLTextureType::TypeCube, 6)?;
    let texture_3d = make_black_texture(device, MTLTextureType::Type3D, 1)?;
    let depth_2d = make_zero_depth_texture(device)?;

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
        depth_2d: Retained::into_raw(depth_2d) as u64,
        sampler: Retained::into_raw(sampler) as u64,
    })
}

/// A 1×1 (per slice) `RGBA8Unorm` texture filled with opaque black.
///
/// `slices` is 6 for a cube (one per face), 1 otherwise. CPU-writable storage
/// for the device (see [`cpu_written_texture_storage`]) so the pixel can be
/// written with `replaceRegion:`; the four bytes are `(R, G, B, A) =
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
    desc.setStorageMode(cpu_written_texture_storage(device));

    let Some(texture) = device.newTextureWithDescriptor(&desc) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "null texture: the 1x1 black {texture_type:?} texture could not be created; \
             unbound declared samplers stay unbound"
        );
        return None;
    };
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

/// A 1x1 `Depth32Float` texture holding depth zero, for a `depth2d<float>` slot.
///
/// A depth texture's storage is private to the GPU, so it is cleared by a
/// one-off render pass on a queue of its own, waited for once: a raw read of
/// it returns zero and a comparison against it fails for every reference
/// above zero, so it samples as black like the colour fallbacks.
fn make_zero_depth_texture(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type2D);
    desc.setPixelFormat(MTLPixelFormat::Depth32Float);
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setWidth(1) };
    // SAFETY: plain property setter on a fresh descriptor.
    unsafe { desc.setHeight(1) };
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
    desc.setStorageMode(MTLStorageMode::Private);
    let Some(texture) = device.newTextureWithDescriptor(&desc) else {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "null texture: the 1x1 depth texture could not be created; unbound declared \
             samplers stay unbound"
        );
        return None;
    };
    texture.setLabel(Some(&objc2_foundation::NSString::from_str(
        "mtld3d-null-depth",
    )));
    let cleared = device.newCommandQueue().and_then(|queue| {
        queue.setLabel(Some(&objc2_foundation::NSString::from_str(
            "mtld3d-null-depth-clear",
        )));
        let cmd_buf = queue.commandBuffer()?;
        cmd_buf.setLabel(Some(&objc2_foundation::NSString::from_str(
            "mtld3d-null-depth-clear",
        )));
        let pass_desc = MTLRenderPassDescriptor::new();
        let depth = pass_desc.depthAttachment();
        depth.setTexture(Some(&texture));
        depth.setLoadAction(MTLLoadAction::Clear);
        depth.setClearDepth(0.0);
        depth.setStoreAction(MTLStoreAction::Store);
        let encoder = cmd_buf.renderCommandEncoderWithDescriptor(&pass_desc)?;
        encoder.setLabel(Some(&objc2_foundation::NSString::from_str(
            "mtld3d-null-depth-clear",
        )));
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        Some(())
    });
    if cleared.is_none() {
        mtld3d_shared::log_once_warn!(
            target: LOG_TARGET,
            "null texture: the 1x1 depth texture could not be cleared; unbound declared \
             samplers stay unbound"
        );
        return None;
    }
    Some(texture)
}
