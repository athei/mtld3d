use mtld3d_shared::{
    CreateDepthStencilStateParams, CreateTextureSliceViewParams, MetalHandle, StencilFaceParams,
    TextureCreateDesc,
    mtl::{
        BlendFactor as WireBlendFactor, CompareFunc, PixelFormat, StencilOp, StorageMode, Swizzle,
        TextureCreateFlags, TextureUsage,
    },
    mtl_handle::{MTLCommandQueueKind, MTLDepthStencilStateKind, MTLDeviceKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLBlendFactor, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompareFunction, MTLDepthStencilDescriptor, MTLDevice, MTLLoadAction, MTLPixelFormat,
    MTLRenderPassDescriptor, MTLResource, MTLStencilDescriptor, MTLStencilOperation,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle,
    MTLTextureSwizzleChannels, MTLTextureType, MTLTextureUsage,
};

use crate::metal::handle::{IntoRetained, ReleaseRetain};

/// The texture handles this side has minted and not yet destroyed.
///
/// A handle enters at [`mint`] and leaves at [`destroy_texture`], so a
/// destroy of a handle that is not here is a second destroy of one that was,
/// or a value that never was one: the release it asks for would hand the
/// Objective-C runtime an object that is gone, which ends the process where
/// the memory has been reused and corrupts a live object where it has not.
/// The check costs one hash per create and per destroy, both rare next to
/// what they do.
static LIVE_TEXTURES: std::sync::LazyLock<std::sync::Mutex<rustc_hash::FxHashSet<u64>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(rustc_hash::FxHashSet::default()));

/// Hand a texture's retain to a raw handle, recording it as live.
///
/// Every texture handle that reaches the PE side is minted here, so the
/// ledger and [`destroy_texture`]'s check agree on what is live.
fn mint<T: objc2::Message>(texture: Retained<T>) -> u64 {
    let raw = Retained::into_raw(texture).cast::<core::ffi::c_void>() as usize as u64;
    LIVE_TEXTURES
        .lock()
        .expect("live-texture ledger poisoned")
        .insert(raw);
    raw
}

/// Creates a persistent `BGRA8Unorm` render target texture for use as a backbuffer.
///
/// The new texture is cleared to opaque black before it is returned. A
/// fresh `MTLTexture` has undefined contents, and the back buffer is
/// presentable before the application's first draw or clear reaches it
/// (a `Present` with no rendering is legal D3D9, and routine right after
/// a `Reset`). D3D9 leaves post-`Reset` contents formally undefined, but
/// real drivers hand out zeroed surfaces, and applications visibly rely
/// on that during scene transitions. The clear is encoded on the frame
/// queue, so commit order is the fence: every later frame command buffer
/// observes a black back buffer, with no CPU wait.
pub fn create_backbuffer(
    device_handle: MetalHandle<MTLDeviceKind>,
    queue_handle: MetalHandle<MTLCommandQueueKind>,
    width: u32,
    height: u32,
) -> Option<(MetalHandle<MTLTextureKind>, u64)> {
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
    let texture = create_color_texture(
        &device,
        width,
        height,
        MTLPixelFormat::BGRA8Unorm,
        "mtld3d-backbuffer",
    )?;
    clear_texture_black(queue_handle, &texture);
    let srgb_handle = srgb_twin_view(
        &texture,
        PixelFormat::Bgra8Unorm,
        1,
        1,
        IDENTITY_SWIZZLE,
        "mtld3d-backbuffer",
    );
    // SAFETY: `Retained::into_raw` transfers the retain into a raw
    // pointer; `MetalHandle::new` adopts it as the canonical retain.
    Some((
        unsafe { MetalHandle::<MTLTextureKind>::new(mint(texture)) },
        srgb_handle,
    ))
}

/// The channel order a view leaves untouched.
const IDENTITY_SWIZZLE: MTLTextureSwizzleChannels = MTLTextureSwizzleChannels {
    red: objc2_metal::MTLTextureSwizzle::Red,
    green: objc2_metal::MTLTextureSwizzle::Green,
    blue: objc2_metal::MTLTextureSwizzle::Blue,
    alpha: objc2_metal::MTLTextureSwizzle::Alpha,
};

/// Usage bits for a texture, by what it serves.
///
/// `ShaderRead` always, `RenderTarget` for an attachment. `PixelFormatView` is
/// what a view of another pixel format needs on its base texture, and a colour
/// texture takes such views: the sRGB twin, and the channel swizzle a format
/// without an alpha lane samples through. An Apple-family GPU exempts a view
/// that changes only transfer function or swizzle from the usage, and the bit
/// would cost it lossless compression on every colour texture, so it stays
/// off there; every other family refuses the view without it. Depth textures
/// take no view and never carry the bit.
fn texture_usage(
    device: &ProtocolObject<dyn MTLDevice>,
    render_target: bool,
    takes_views: bool,
) -> MTLTextureUsage {
    let mut usage = MTLTextureUsage::ShaderRead;
    if render_target {
        usage |= MTLTextureUsage::RenderTarget;
    }
    if takes_views && !super::device::is_apple_family(device) {
        usage |= MTLTextureUsage::PixelFormatView;
    }
    usage
}

/// The sRGB twin view of a colour texture, or 0 when the format has no twin.
///
/// The base texture's usage allows the view (see [`texture_usage`]), and the
/// view inherits that usage, which keeps a render target's twin
/// render-targetable. `swizzle` mirrors whichever channel order the base
/// handle is handed out with, so the two views differ only in transfer
/// function; a render target is always handed out unswizzled, since Metal
/// forbids rendering through a channel swizzle.
fn srgb_twin_view(
    texture: &ProtocolObject<dyn MTLTexture>,
    format: PixelFormat,
    levels: usize,
    slices: usize,
    swizzle: MTLTextureSwizzleChannels,
    label: &str,
) -> u64 {
    let Some(srgb_format) = format.srgb_twin() else {
        return 0;
    };
    // SAFETY: objc2 typed binding; `texture` is live and the ranges match
    // the descriptor it was created with.
    let view = unsafe {
        texture.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
            mtl_pixel_format(srgb_format),
            texture.textureType(),
            objc2_foundation::NSRange::new(0, levels),
            objc2_foundation::NSRange::new(0, slices),
            swizzle,
        )
    };
    view.map_or(0, |view| {
        let srgb_label = objc2_foundation::NSString::from_str(&format!("{label}-srgb"));
        view.setLabel(Some(&srgb_label));
        mint(view)
    })
}

/// Clear a freshly created render target to opaque black.
///
/// One empty render pass with `LoadAction::Clear` in its own command
/// buffer. Runs on the creation paths only (device create, `Reset`,
/// auto-resize), so cost is irrelevant. Failure to encode leaves the
/// texture with undefined contents, which is what creation produced
/// anyway, so it is logged and tolerated rather than failing creation.
fn clear_texture_black(
    queue_handle: MetalHandle<MTLCommandQueueKind>,
    texture: &ProtocolObject<dyn MTLTexture>,
) {
    let Some(queue) = queue_handle.into_retained() else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "create_backbuffer: no queue for the creation-time clear; \
             the new backbuffer starts with undefined contents",
        );
        return;
    };
    let Some(cmd_buf) = queue.commandBuffer() else {
        mtld3d_shared::log_once_warn!(
            target: crate::LOG_TARGET,
            "create_backbuffer: commandBuffer() returned nil for the creation-time \
             clear; the new backbuffer starts with undefined contents",
        );
        return;
    };
    let pass_desc = MTLRenderPassDescriptor::new();
    // SAFETY: `colorAttachments()` returns a non-null descriptor array;
    // subscript 0 is always valid.
    let color0 = unsafe { pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
    color0.setTexture(Some(texture));
    color0.setLoadAction(MTLLoadAction::Clear);
    color0.setClearColor(MTLClearColor {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    });
    color0.setStoreAction(MTLStoreAction::Store);
    if let Some(enc) = cmd_buf.renderCommandEncoderWithDescriptor(&pass_desc) {
        let label = objc2_foundation::NSString::from_str("mtld3d-backbuffer-init-clear");
        enc.setLabel(Some(&label));
        enc.endEncoding();
    }
    cmd_buf.commit();
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
    sample_count: u32,
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
    if sample_count > 1 {
        desc.setTextureType(MTLTextureType::Type2DMultisample);
        // SAFETY: objc2 typed binding; the count was validated against
        // `supportsTextureSampleCount:` before it crossed the boundary.
        unsafe { desc.setSampleCount(sample_count as usize) };
    }

    // Depth textures must be in private storage on Apple Silicon
    desc.setStorageMode(objc2_metal::MTLStorageMode::Private);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str("mtld3d-depth");
    texture.setLabel(Some(&label));
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(mint(texture)) })
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
) -> Option<(MetalHandle<MTLTextureKind>, u64)> {
    let device = device_handle.into_retained()?;
    let texture = create_color_texture(
        &device,
        width,
        height,
        mtl_pixel_format(pixel_format),
        "mtld3d-color-target",
    )?;
    let srgb_handle = srgb_twin_view(
        &texture,
        pixel_format,
        1,
        1,
        IDENTITY_SWIZZLE,
        "mtld3d-color-target",
    );
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some((
        unsafe { MetalHandle::<MTLTextureKind>::new(mint(texture)) },
        srgb_handle,
    ))
}

/// Creates the colour texture a `MetalFX` upscale reads or writes.
///
/// Same shape as [`create_color_target`] without the sRGB twin: nothing
/// samples the scratch through a transfer function. `MTLFXSpatialScaler`
/// rejects an output texture that is not `Private`, which every colour
/// texture here is.
///
/// Takes the device by reference rather than by handle because `submit_frame`
/// reaches this through `cmd_buf.device()` and carries no device handle of its
/// own.
pub fn create_upscale_target(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
) -> Option<MetalHandle<MTLTextureKind>> {
    let texture = create_color_texture(
        device,
        width,
        height,
        mtl_pixel_format(pixel_format),
        "mtld3d-upscale-scratch",
    )?;
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(mint(texture)) })
}

/// Shared body of the colour-texture creators.
///
/// `Private` storage: nothing on the CPU touches a colour texture (every
/// upload is a blit from a staging buffer and every readback a blit into one),
/// and the descriptor's default, `Managed`, would keep a CPU mirror the GPU
/// has to synchronise. Hands back the retained texture rather than a handle,
/// so the caller can take an sRGB view of it before transferring the retain
/// into the handle it returns.
fn create_color_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    mtl_format: objc2_metal::MTLPixelFormat,
    label: &str,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
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
    desc.setUsage(texture_usage(device, true, true));
    desc.setStorageMode(MTLStorageMode::Private);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let label = objc2_foundation::NSString::from_str(label);
    texture.setLabel(Some(&label));
    Some(texture)
}

/// Creates the multisampled companion of a single-sample render target.
///
/// The result is the colour attachment every pass renders into; the
/// single-sample texture it was made for is its resolve target and stays the
/// only one anything samples, blits, reads back or presents. `Private`
/// storage rather than `Memoryless` because a D3D9 frame can render into the
/// same target across several passes before the content is consumed, and only
/// the last of those passes takes the resolve; the passes in between store
/// the multisample content, which a memoryless texture cannot do.
///
/// The second element is the companion's sRGB twin view, or 0 when the format
/// has no sRGB counterpart. A multisampled pass writing sRGB attaches that
/// view and resolves into the single-sample texture's own twin, which Metal
/// requires to carry the attachment's pixel format.
///
/// Returns `None` when `sample_count` is 1 (nothing to create) or when Metal
/// declines the descriptor; the caller knows which by its own `sample_count`
/// and reports the second case as a failed create.
pub fn create_msaa_companion(
    device_handle: MetalHandle<MTLDeviceKind>,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    sample_count: u32,
    label: &str,
) -> Option<(MetalHandle<MTLTextureKind>, u64)> {
    if sample_count <= 1 {
        return None;
    }
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
    desc.setTextureType(MTLTextureType::Type2DMultisample);
    // SAFETY: objc2 typed binding; the count was validated against
    // `supportsTextureSampleCount:` before it crossed the boundary.
    unsafe { desc.setSampleCount(sample_count as usize) };
    desc.setUsage(texture_usage(&device, true, true));
    desc.setStorageMode(MTLStorageMode::Private);

    let texture = device.newTextureWithDescriptor(&desc)?;
    let ns_label = objc2_foundation::NSString::from_str(label);
    texture.setLabel(Some(&ns_label));
    let srgb_handle = srgb_twin_view(&texture, pixel_format, 1, 1, IDENTITY_SWIZZLE, label);
    // SAFETY: `Retained::into_raw` transfers the retain; `MetalHandle::new`
    // adopts it as canonical.
    Some((
        unsafe { MetalHandle::<MTLTextureKind>::new(mint(texture)) },
        srgb_handle,
    ))
}

/// Creates an `MTLDepthStencilState` object.
pub fn create_depth_stencil_state(
    params: &CreateDepthStencilStateParams,
) -> Option<MetalHandle<MTLDepthStencilStateKind>> {
    let device = params.device_handle.into_retained()?;

    let desc = MTLDepthStencilDescriptor::new();

    if params.depth_test_enable != 0 {
        desc.setDepthCompareFunction(mtl_compare_function(params.depth_compare_func));
        desc.setDepthWriteEnabled(params.depth_write_enable != 0);
    } else {
        desc.setDepthCompareFunction(MTLCompareFunction::Always);
        desc.setDepthWriteEnabled(false);
    }

    if params.stencil_test_enable != 0 {
        desc.setFrontFaceStencil(Some(&stencil_face_descriptor(
            params.front,
            params.stencil_read_mask,
            params.stencil_write_mask,
        )));
        desc.setBackFaceStencil(Some(&stencil_face_descriptor(
            params.back,
            params.stencil_read_mask,
            params.stencil_write_mask,
        )));
    }

    let id = params.id;
    let label = objc2_foundation::NSString::from_str(&format!("mtld3d-dss-{id:#x}"));
    desc.setLabel(Some(&label));

    let state = device.newDepthStencilStateWithDescriptor(&desc)?;
    // SAFETY: Retained::into_raw transfers the retain into the typed handle.
    Some(unsafe { MetalHandle::<MTLDepthStencilStateKind>::new(Retained::into_raw(state) as u64) })
}

/// Builds one `MTLStencilDescriptor` face.
///
/// Metal carries the read/write masks per face; D3D9 has one pair of masks
/// (`D3DRS_STENCILMASK` / `D3DRS_STENCILWRITEMASK`) covering both, so the
/// same pair is written to each face.
fn stencil_face_descriptor(
    face: StencilFaceParams,
    read_mask: u32,
    write_mask: u32,
) -> Retained<MTLStencilDescriptor> {
    let desc = MTLStencilDescriptor::new();
    desc.setStencilCompareFunction(mtl_compare_function(face.compare_func));
    desc.setStencilFailureOperation(mtl_stencil_operation(face.stencil_fail_op));
    desc.setDepthFailureOperation(mtl_stencil_operation(face.depth_fail_op));
    desc.setDepthStencilPassOperation(mtl_stencil_operation(face.pass_op));
    desc.setReadMask(read_mask);
    desc.setWriteMask(write_mask);
    desc
}

pub const fn mtl_stencil_operation(wire: StencilOp) -> MTLStencilOperation {
    match wire {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
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
///
/// Returns `(handle, srgb_handle)`: the texture (or its swizzle view) plus
/// the eagerly-created sRGB twin view when the format has one
/// (`PixelFormat::srgb_twin`), else 0. The draw-time bind picks the twin
/// when the stage's sampler has `D3DSAMP_SRGBTEXTURE=1`, giving the
/// hardware sRGB→linear decode D3D9 promises for that state, and the render
/// pass attaches it in place of the base texture under
/// `D3DRS_SRGBWRITEENABLE`, giving the post-blend encode.
pub fn create_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    desc: &TextureCreateDesc,
) -> Option<(u64, u64)> {
    let mtl_format = mtl_pixel_format(desc.pixel_format);
    let is_depth = is_depth_pixel_format(desc.pixel_format);

    // Texture shape is carried explicitly. This keeps cube and single-slice
    // volume descriptors distinct without changing the wire layout.
    // there is no `texture3DDescriptor…` convenience constructor, so build the
    // descriptor by hand. Everything else (usage, storage, swizzle, label) is
    // shared with the 2D path below.
    let tex_desc = if desc.flags.contains(TextureCreateFlags::TYPE_CUBE) {
        let d = MTLTextureDescriptor::new();
        d.setTextureType(objc2_metal::MTLTextureType::TypeCube);
        d.setPixelFormat(mtl_format);
        // SAFETY: objc2 typed property setters on a fresh descriptor.
        unsafe { d.setWidth(desc.width as usize) };
        // SAFETY: cube faces are square; height is still set for an explicit,
        // self-contained descriptor.
        unsafe { d.setHeight(desc.height as usize) };
        d
    } else if desc.flags.contains(TextureCreateFlags::TYPE_3D) {
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
    // they need both the depth-attachment binding AND ShaderRead.
    let is_render_target = desc
        .usage_flags
        .intersects(TextureUsage::RENDER_TARGET | TextureUsage::DEPTH_STENCIL);
    tex_desc.setUsage(texture_usage(device, is_render_target, !is_depth));

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

    // sRGB twin view, created eagerly for every colour format that has one so
    // the draw-time bind can honour `D3DSAMP_SRGBTEXTURE=1` without a
    // mid-frame crossing.
    // The twin mirrors the base handle's swizzle (when the base is handed out
    // as a swizzle view below) so the two views only ever differ in transfer
    // function. A render target never takes the swizzle branch, so the twin
    // of one carries the identity swizzle and stays legal as a colour
    // attachment; a view inherits its base texture's usage, so the twin of a
    // render target is render-targetable too.
    let srgb_handle = if is_depth {
        0
    } else {
        let base_is_swizzle_view =
            !is_render_target && desc.flags.contains(TextureCreateFlags::HAS_SWIZZLE);
        let swizzle = if base_is_swizzle_view {
            MTLTextureSwizzleChannels {
                red: mtl_texture_swizzle(desc.swizzle_r),
                green: mtl_texture_swizzle(desc.swizzle_g),
                blue: mtl_texture_swizzle(desc.swizzle_b),
                alpha: mtl_texture_swizzle(desc.swizzle_a),
            }
        } else {
            IDENTITY_SWIZZLE
        };
        srgb_twin_view(
            &texture,
            desc.pixel_format,
            desc.levels as usize,
            if desc.flags.contains(TextureCreateFlags::TYPE_CUBE) {
                6
            } else {
                1
            },
            swizzle,
            &label_str,
        )
    };

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
    if !is_depth && !is_render_target && desc.flags.contains(TextureCreateFlags::HAS_SWIZZLE) {
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
                objc2_foundation::NSRange::new(
                    0,
                    if desc.flags.contains(TextureCreateFlags::TYPE_CUBE) {
                        6
                    } else {
                        1
                    },
                ),
                swizzle_channels,
            )
        };
        if let Some(view) = view {
            view.setLabel(Some(&label));
            return Some((mint(view), srgb_handle));
        }
    }

    texture.setLabel(Some(&label));

    Some((mint(texture), srgb_handle))
}

/// Create a single-slice, 2D view of one array slice of a texture.
///
/// The scaling `StretchRect` fragment function declares its source as
/// `texture2d<float>`. A cube-map source therefore has to reach it as a 2D view
/// of the face the D3D9 call named: the cube texture binds as a `texturecube`
/// and the sample lands on face 0 whichever face the blit addressed. The view
/// keeps the base format and spans every mip level, so no GPU family asks for
/// `PixelFormatView` usage on the base texture, and the explicit `level()` the
/// fragment function passes still selects the source mip.
///
/// The returned view is a fresh object whose retain the caller owns; the PE
/// side destroys it once the frame that binds it has retired.
pub fn create_texture_slice_view(
    params: &CreateTextureSliceViewParams,
) -> Option<MetalHandle<MTLTextureKind>> {
    let texture = params.texture_handle.into_retained()?;
    let levels = texture.mipmapLevelCount();
    // SAFETY: objc2 typed binding; `texture` is retained for the call, the
    // format is the base texture's own, and the level range covers exactly the
    // levels it declares.
    let view = unsafe {
        texture.newTextureViewWithPixelFormat_textureType_levels_slices(
            texture.pixelFormat(),
            MTLTextureType::Type2D,
            objc2_foundation::NSRange::new(0, levels),
            objc2_foundation::NSRange::new(params.slice as usize, 1),
        )
    }?;
    let label = objc2_foundation::NSString::from_str(&format!(
        "mtld3d-sliceview-{:#x}-{}",
        params.texture_handle, params.slice
    ));
    view.setLabel(Some(&label));
    // SAFETY: `Retained::into_raw` hands over the view's only retain, which
    // the typed handle carries to the PE side.
    Some(unsafe { MetalHandle::<MTLTextureKind>::new(mint(view)) })
}

/// Release a Metal texture handle.
pub fn destroy_texture(texture_handle: u64) {
    if texture_handle == 0 {
        return;
    }
    let was_live = LIVE_TEXTURES
        .lock()
        .expect("live-texture ledger poisoned")
        .remove(&texture_handle);
    if !was_live {
        // Not once: each occurrence names its handle, and the debug lines
        // around it say which destroy asked twice.
        log::warn!(
            target: crate::LOG_TARGET,
            "destroy_texture: {texture_handle:#x} is not a live texture handle; a second \
             destroy of a texture already released, or a value that never was one; skipped",
        );
        return;
    }
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
        PixelFormat::Rg16Unorm => MTLPixelFormat::RG16Unorm,
        PixelFormat::Rg16Float => MTLPixelFormat::RG16Float,
        PixelFormat::Rg32Float => MTLPixelFormat::RG32Float,
        PixelFormat::B5G6R5Unorm => MTLPixelFormat::B5G6R5Unorm,
        PixelFormat::Abgr4Unorm => MTLPixelFormat::ABGR4Unorm,
        PixelFormat::Bgr5A1Unorm => MTLPixelFormat::BGR5A1Unorm,
        PixelFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        PixelFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        PixelFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        PixelFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        PixelFormat::Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
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

/// The wire format a Metal pixel format came from, `None` for one mtld3d never sends.
///
/// The inverse of [`mtl_pixel_format`], for a path that starts from a live
/// texture rather than from a wire message: the readback resolve creates its
/// scratch in the source texture's own format.
pub const fn wire_pixel_format(mtl: MTLPixelFormat) -> Option<PixelFormat> {
    Some(match mtl {
        MTLPixelFormat::A8Unorm => PixelFormat::A8Unorm,
        MTLPixelFormat::R8Unorm => PixelFormat::R8Unorm,
        MTLPixelFormat::R16Unorm => PixelFormat::R16Unorm,
        MTLPixelFormat::R16Float => PixelFormat::R16Float,
        MTLPixelFormat::R32Float => PixelFormat::R32Float,
        MTLPixelFormat::BC4_RUnorm => PixelFormat::Bc4RUnorm,
        MTLPixelFormat::RG8Unorm => PixelFormat::Rg8Unorm,
        MTLPixelFormat::RG8Snorm => PixelFormat::Rg8Snorm,
        MTLPixelFormat::RG16Unorm => PixelFormat::Rg16Unorm,
        MTLPixelFormat::RG16Float => PixelFormat::Rg16Float,
        MTLPixelFormat::RG32Float => PixelFormat::Rg32Float,
        MTLPixelFormat::B5G6R5Unorm => PixelFormat::B5G6R5Unorm,
        MTLPixelFormat::ABGR4Unorm => PixelFormat::Abgr4Unorm,
        MTLPixelFormat::BGR5A1Unorm => PixelFormat::Bgr5A1Unorm,
        MTLPixelFormat::RGBA8Unorm => PixelFormat::Rgba8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB => PixelFormat::Rgba8UnormSrgb,
        MTLPixelFormat::BGRA8Unorm => PixelFormat::Bgra8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB => PixelFormat::Bgra8UnormSrgb,
        MTLPixelFormat::RGBA16Unorm => PixelFormat::Rgba16Unorm,
        MTLPixelFormat::RGBA16Float => PixelFormat::Rgba16Float,
        MTLPixelFormat::RGBA32Float => PixelFormat::Rgba32Float,
        MTLPixelFormat::BC1_RGBA => PixelFormat::Bc1Rgba,
        MTLPixelFormat::BC1_RGBA_sRGB => PixelFormat::Bc1RgbaSrgb,
        MTLPixelFormat::BC2_RGBA => PixelFormat::Bc2Rgba,
        MTLPixelFormat::BC2_RGBA_sRGB => PixelFormat::Bc2RgbaSrgb,
        MTLPixelFormat::BC3_RGBA => PixelFormat::Bc3Rgba,
        MTLPixelFormat::BC3_RGBA_sRGB => PixelFormat::Bc3RgbaSrgb,
        MTLPixelFormat::Depth32Float => PixelFormat::Depth32Float,
        MTLPixelFormat::Depth32Float_Stencil8 => PixelFormat::Depth32FloatStencil8,
        _ => return None,
    })
}

/// True for a colour format a render pass can write and a sampler can read.
///
/// The readback resolve renders the scaled source into a scratch of the same
/// format, which rules out the block-compressed formats (never render
/// targets) and the depth formats (read back through their own path).
pub const fn is_resolvable_color_format(fmt: PixelFormat) -> bool {
    !matches!(
        fmt,
        PixelFormat::Bc1Rgba
            | PixelFormat::Bc1RgbaSrgb
            | PixelFormat::Bc2Rgba
            | PixelFormat::Bc2RgbaSrgb
            | PixelFormat::Bc3Rgba
            | PixelFormat::Bc3RgbaSrgb
            | PixelFormat::Bc4RUnorm
            | PixelFormat::Depth32Float
            | PixelFormat::Depth32FloatStencil8
    )
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

#[cfg(test)]
mod tests;
