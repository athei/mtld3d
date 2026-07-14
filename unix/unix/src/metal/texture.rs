use mtld3d_shared::{
    MetalHandle, TextureCreateDesc,
    mtl::{
        BlendFactor as WireBlendFactor, CompareFunc, PixelFormat, StorageMode, Swizzle,
        TextureUsage,
    },
    mtl_handle::{MTLDepthStencilStateKind, MTLDeviceKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlendFactor, MTLCompareFunction, MTLDepthStencilDescriptor, MTLDevice, MTLPixelFormat,
    MTLResource, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureUsage,
};

use crate::metal::handle::{IntoRetained, ReleaseRetain};

/// Creates a persistent `BGRA8Unorm` render target texture for use as a backbuffer.
pub fn create_backbuffer(
    device_handle: MetalHandle<MTLDeviceKind>,
    width: u32,
    height: u32,
) -> Option<MetalHandle<MTLTextureKind>> {
    // Metal raises an NSException (→ abort) for a zero or over-large texture
    // dimension. Reject such a request so a degenerate backbuffer size — e.g.
    // resolved from the off-screen monitor geometry the conformance suite
    // probes — fails CreateBackbuffer gracefully instead of aborting the
    // process. `MAX_TEXTURE_DIM` is the Metal 2D limit on the supported GPUs.
    const MAX_TEXTURE_DIM: u32 = 16384;
    if width == 0 || height == 0 || width > MAX_TEXTURE_DIM || height > MAX_TEXTURE_DIM {
        return None;
    }
    let device = device_handle.into_retained()?;

    // SAFETY: objc2 typed binding; class-method constructor on
    // `MTLTextureDescriptor` returns a freshly autoreleased descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::BGRA8Unorm,
            width as usize,
            height as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str("mtld3d-backbuffer");
    texture.setLabel(Some(&label));
    // SAFETY: `Retained::into_raw` transfers the retain into a raw
    // pointer; `MetalHandle::new` adopts it as the canonical retain.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(Retained::into_raw(texture) as u64) })
}

/// Creates a standalone depth/stencil texture for `CreateDepthStencilSurface`.
///
/// `pixel_format` is the Metal-side enum already resolved from the D3D9
/// depth format on the PE side via `mtld3d_core::format::map_d3d_depth_format`.
pub fn create_depth_texture(
    device_handle: MetalHandle<MTLDeviceKind>,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
) -> Option<MetalHandle<MTLTextureKind>> {
    let device = device_handle.into_retained()?;
    let mtl_format = mtl_pixel_format(pixel_format);

    // SAFETY: objc2 typed binding; class-method constructor on
    // `MTLTextureDescriptor` returns a freshly autoreleased descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            mtl_format,
            width as usize,
            height as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::RenderTarget);

    // Depth textures must be in private storage on Apple Silicon
    desc.setStorageMode(objc2_metal::MTLStorageMode::Private);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str("mtld3d-depth");
    texture.setLabel(Some(&label));
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(Retained::into_raw(texture) as u64) })
}

/// Creates a standalone color render-target texture.
///
/// Serves `CreateRenderTarget` and
/// `CreateOffscreenPlainSurface(D3DPOOL_DEFAULT)`. `pixel_format` is the
/// Metal-side enum already resolved from the D3D9 color format on the PE side
/// via `mtld3d_core::format::map_d3d_format`. Usage mirrors the backbuffer
/// (`RenderTarget | ShaderRead`) so the result can be both rendered to and
/// sampled / used as a `StretchRect` source.
pub fn create_color_target(
    device_handle: MetalHandle<MTLDeviceKind>,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
) -> Option<MetalHandle<MTLTextureKind>> {
    let device = device_handle.into_retained()?;
    let mtl_format = mtl_pixel_format(pixel_format);

    // SAFETY: objc2 typed binding; class-method constructor on
    // `MTLTextureDescriptor` returns a freshly autoreleased descriptor.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            mtl_format,
            width as usize,
            height as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str("mtld3d-color-target");
    texture.setLabel(Some(&label));
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(Retained::into_raw(texture) as u64) })
}

/// Creates an `MTLDepthStencilState` object.
///
/// `compare_func` is an `MTLCompareFunction` raw value.
pub fn create_depth_stencil_state(
    device_handle: MetalHandle<MTLDeviceKind>,
    depth_test_enable: u32,
    depth_write_enable: u32,
    compare_func: CompareFunc,
    id: u64,
) -> Option<MetalHandle<MTLDepthStencilStateKind>> {
    let device = device_handle.into_retained()?;

    let desc = MTLDepthStencilDescriptor::new();

    if depth_test_enable != 0 {
        desc.setDepthCompareFunction(mtl_compare_function(compare_func));
        desc.setDepthWriteEnabled(depth_write_enable != 0);
    } else {
        desc.setDepthCompareFunction(MTLCompareFunction::Always);
        desc.setDepthWriteEnabled(false);
    }

    let label = objc2_foundation::NSString::from_str(&format!("mtld3d-dss-{id:#x}"));
    desc.setLabel(Some(&label));

    let state = device.newDepthStencilStateWithDescriptor(&desc)?;
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    Some(unsafe { MetalHandle::<MTLDepthStencilStateKind>::new(Retained::into_raw(state) as u64) })
}

pub const fn mtl_blend_factor(wire: WireBlendFactor) -> MTLBlendFactor {
    match wire {
        WireBlendFactor::Zero => MTLBlendFactor::Zero,
        WireBlendFactor::One => MTLBlendFactor::One,
        WireBlendFactor::SourceColor => MTLBlendFactor::SourceColor,
        WireBlendFactor::OneMinusSourceColor => MTLBlendFactor::OneMinusSourceColor,
        WireBlendFactor::SourceAlpha => MTLBlendFactor::SourceAlpha,
        WireBlendFactor::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        WireBlendFactor::DestinationAlpha => MTLBlendFactor::DestinationAlpha,
        WireBlendFactor::OneMinusDestinationAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        WireBlendFactor::DestinationColor => MTLBlendFactor::DestinationColor,
        WireBlendFactor::OneMinusDestinationColor => MTLBlendFactor::OneMinusDestinationColor,
        WireBlendFactor::SourceAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        WireBlendFactor::BlendColor => MTLBlendFactor::BlendColor,
        WireBlendFactor::OneMinusBlendColor => MTLBlendFactor::OneMinusBlendColor,
    }
}

/// Creates a texture for sampling.
///
/// Pixel format and swizzle are Metal-level values (already translated from
/// D3D9 on the PE side).
///
/// Depth-format textures (sampleable shadow maps from
/// `CreateTexture(format=D24X8, usage=D3DUSAGE_DEPTHSTENCIL)`) take the
/// `RenderTarget | ShaderRead` usage path: the PE side flags
/// `TextureUsage::DEPTH_STENCIL` AND picks a depth `PixelFormat`, the
/// Metal texture is bindable as a depth attachment and sampleable in the
/// subsequent lit pass. Swizzle views aren't applicable to depth formats.
///
/// One descriptor → one `MTLTexture`. The batched handler iterates this
/// per element; same call shape used by both load-phase warmup batches
/// and one-off lazy creates.
pub fn create_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &TextureCreateDesc,
) -> Option<u64> {
    let mtl_format = mtl_pixel_format(desc.pixel_format);
    let is_depth = is_depth_pixel_format(desc.pixel_format);

    // `depth > 1` is a volume (3D) texture (`D3DFMT_*` `CreateVolumeTexture`);
    // there is no `texture3DDescriptor…` convenience constructor, so build the
    // descriptor by hand. Everything else (usage, storage, swizzle, label) is
    // shared with the 2D path below.
    let tex_desc = if desc.depth > 1 {
        let d = MTLTextureDescriptor::new();
        d.setTextureType(objc2_metal::MTLTextureType::Type3D);
        d.setPixelFormat(mtl_format);
        // SAFETY: objc2 typed property setters on a freshly-allocated descriptor.
        unsafe { d.setWidth(desc.width as usize) };
        // SAFETY: as above.
        unsafe { d.setHeight(desc.height as usize) };
        // SAFETY: as above.
        unsafe { d.setDepth(desc.depth as usize) };
        d
    } else {
        // SAFETY: objc2 typed binding; class-method constructor on
        // `MTLTextureDescriptor` returns a freshly autoreleased descriptor.
        unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                mtl_format,
                desc.width as usize,
                desc.height as usize,
                desc.levels > 1,
            )
        }
    };
    // SAFETY: objc2 typed binding; pure accessor passthrough.
    unsafe { tex_desc.setMipmapLevelCount(desc.levels as usize) };

    // Honor RT bits: RENDER_TARGET textures also keep ShaderRead so the
    // subsequent pass can sample them (water reflection, portrait models).
    // Depth textures with DEPTH_STENCIL flag are sampleable shadow maps:
    // they need both the depth-attachment binding AND ShaderRead. We
    // deliberately don't add `PixelFormatView`: it blocks Metal's
    // lossless framebuffer compression on every BGRA8 / BC1-3 colour
    // texture, and the only reason to want it would be an eager sRGB
    // twin view for `D3DSAMP_SRGBTEXTURE`, which no mtld3d target game
    // actually sets.
    let is_render_target = desc
        .usage_flags
        .intersects(TextureUsage::RENDER_TARGET | TextureUsage::DEPTH_STENCIL);
    let mut usage = MTLTextureUsage::ShaderRead;
    if is_render_target {
        usage |= MTLTextureUsage::RenderTarget;
    }
    tex_desc.setUsage(usage);

    // Depth textures must live in private storage on Apple Silicon, regardless
    // of what the PE side requested (CPU upload of a depth texture is meaningless).
    if is_depth {
        tex_desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    } else {
        tex_desc.setStorageMode(mtl_storage_mode(desc.storage_mode));
    }

    let texture = device.newTextureWithDescriptor(&tex_desc)?;

    // Label the handle the PE side will see — surfaces the mtld3d TextureId
    // alongside the MTLTexture in Xcode frame captures, which is the
    // mapping every cross-recreate / handle-recycle texture debugging
    // step needs to correlate an MTLTexture back to its mtld3d TextureId.
    let label_str = if is_depth {
        format!("mtld3d-depthtex-{:#x}", desc.tex_id)
    } else {
        format!("mtld3d-tex-{:#x}", desc.tex_id)
    };
    let label = objc2_foundation::NSString::from_str(&label_str);

    // Swizzle views don't apply to depth formats — depth shaders sample
    // via the `depth2d<float>` MSL type which returns a single channel.
    //
    // Render targets are excluded too: Metal forbids `RenderTarget` usage on a
    // texture view that carries a non-identity swizzle (you cannot render
    // *through* a channel swizzle), so the view silently drops to `ShaderRead`
    // only. Handing that view back as the texture's handle then fails render-
    // pass validation the moment it is bound as a colour attachment (e.g. an
    // `X8R8G8B8` `D3DUSAGE_RENDERTARGET` surface, whose swizzle just forces the
    // X channel to read as alpha=1 when *sampled*). For a render target the
    // base texture is bound directly; the sample-time alpha fixup is sacrificed
    // (X8 render targets sampling their own alpha is an undefined-value corner
    // of D3D9), which is the right trade against a hard validation/UB crash.
    if !is_depth && !is_render_target && desc.has_swizzle != 0 {
        let swizzle_channels = MTLTextureSwizzleChannels {
            red: mtl_texture_swizzle(desc.swizzle_r),
            green: mtl_texture_swizzle(desc.swizzle_g),
            blue: mtl_texture_swizzle(desc.swizzle_b),
            alpha: mtl_texture_swizzle(desc.swizzle_a),
        };
        // SAFETY: objc2 typed binding; `texture` is the freshly retained
        // texture above; levels/slices ranges match its descriptor.
        let view = unsafe {
            texture.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                mtl_format,
                texture.textureType(),
                objc2_foundation::NSRange::new(0, desc.levels as usize),
                objc2_foundation::NSRange::new(0, 1),
                swizzle_channels,
            )
        };
        if let Some(view) = view {
            view.setLabel(Some(&label));
            return Some(Retained::into_raw(view) as u64);
        }
    }

    texture.setLabel(Some(&label));

    Some(Retained::into_raw(texture) as u64)
}

/// Release a Metal texture handle.
pub fn destroy_texture(texture_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has dropped its only copy of `texture_handle`.
    let handle = unsafe { MetalHandle::<MTLTextureKind>::new(texture_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}

/// Release a Metal depth-stencil-state handle.
pub fn destroy_depth_stencil_state(state_handle: u64) {
    // SAFETY: bulk-destroy thunk; PE side has dropped its only copy of `state_handle`.
    let handle = unsafe { MetalHandle::<MTLDepthStencilStateKind>::new(state_handle) };
    // SAFETY: just wrapped the unique canonical retain.
    unsafe { handle.release_retain() };
}

const fn mtl_compare_function(wire: CompareFunc) -> MTLCompareFunction {
    match wire {
        CompareFunc::Never => MTLCompareFunction::Never,
        CompareFunc::Less => MTLCompareFunction::Less,
        CompareFunc::Equal => MTLCompareFunction::Equal,
        CompareFunc::LessEqual => MTLCompareFunction::LessEqual,
        CompareFunc::Greater => MTLCompareFunction::Greater,
        CompareFunc::NotEqual => MTLCompareFunction::NotEqual,
        CompareFunc::GreaterEqual => MTLCompareFunction::GreaterEqual,
        CompareFunc::Always => MTLCompareFunction::Always,
    }
}

pub const fn mtl_pixel_format(wire: PixelFormat) -> MTLPixelFormat {
    match wire {
        PixelFormat::A8Unorm => MTLPixelFormat::A8Unorm,
        PixelFormat::R8Unorm => MTLPixelFormat::R8Unorm,
        PixelFormat::R16Unorm => MTLPixelFormat::R16Unorm,
        PixelFormat::R16Float => MTLPixelFormat::R16Float,
        PixelFormat::R32Float => MTLPixelFormat::R32Float,
        PixelFormat::Bc4RUnorm => MTLPixelFormat::BC4_RUnorm,
        PixelFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        PixelFormat::Rg8Snorm => MTLPixelFormat::RG8Snorm,
        PixelFormat::B5G6R5Unorm => MTLPixelFormat::B5G6R5Unorm,
        PixelFormat::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
        PixelFormat::Bgr5A1Unorm => MTLPixelFormat::BGR5A1Unorm,
        PixelFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        PixelFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        PixelFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        PixelFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        PixelFormat::Bc1Rgba => MTLPixelFormat::BC1_RGBA,
        PixelFormat::Bc1RgbaSrgb => MTLPixelFormat::BC1_RGBA_sRGB,
        PixelFormat::Bc2Rgba => MTLPixelFormat::BC2_RGBA,
        PixelFormat::Bc2RgbaSrgb => MTLPixelFormat::BC2_RGBA_sRGB,
        PixelFormat::Bc3Rgba => MTLPixelFormat::BC3_RGBA,
        PixelFormat::Bc3RgbaSrgb => MTLPixelFormat::BC3_RGBA_sRGB,
        PixelFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        PixelFormat::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

/// True for depth/stencil pixel formats.
///
/// Used by `create_texture` to route shadow-map textures into the depth
/// attachment + sampleable usage path.
pub const fn is_depth_pixel_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8
    )
}

const fn mtl_texture_swizzle(wire: Swizzle) -> MTLTextureSwizzle {
    match wire {
        Swizzle::Zero => MTLTextureSwizzle::Zero,
        Swizzle::One => MTLTextureSwizzle::One,
        Swizzle::Red => MTLTextureSwizzle::Red,
        Swizzle::Green => MTLTextureSwizzle::Green,
        Swizzle::Blue => MTLTextureSwizzle::Blue,
        Swizzle::Alpha => MTLTextureSwizzle::Alpha,
    }
}

const fn mtl_storage_mode(wire: StorageMode) -> MTLStorageMode {
    match wire {
        StorageMode::Shared => MTLStorageMode::Shared,
        StorageMode::Managed => MTLStorageMode::Managed,
        StorageMode::Private => MTLStorageMode::Private,
        StorageMode::Memoryless => MTLStorageMode::Memoryless,
    }
}
