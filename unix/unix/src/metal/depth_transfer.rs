//! Sample-zero and nearest depth/stencil transfer for CPU-backed depth textures.

use mtld3d_shared::{
    BlitCommand, MetalHandle,
    mtl::DepthTransferKind,
    mtl_handle::{MTLComputePipelineStateKind, MTLDeviceKind, MTLTextureKind},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBlitOption, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLDevice, MTLLanguageVersion, MTLLibrary,
    MTLMathMode, MTLOrigin, MTLPixelFormat, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use super::{
    command::diagnostics,
    handle::{IntoRetained, ReleaseRetain},
};
use crate::LOG_TARGET;

/// Compile only the concrete source layout requested by the device owner.
pub fn create_pipeline(
    device: MetalHandle<MTLDeviceKind>,
    kind: DepthTransferKind,
) -> Option<MetalHandle<MTLComputePipelineStateKind>> {
    let device = device.into_retained()?;
    let multiple = matches!(
        kind,
        DepthTransferKind::MultisampleDepth | DepthTransferKind::MultisampleDepthStencil
    );
    let stencil = matches!(
        kind,
        DepthTransferKind::DepthStencil | DepthTransferKind::MultisampleDepthStencil
    );
    let depth_argument = if multiple {
        "depth2d_ms<float, access::read> d [[texture(0)]],"
    } else {
        "device const float* id [[buffer(0)]],"
    };
    let depth_read = if multiple {
        "d.read(source, 0)"
    } else {
        "id[source.y * sizes[4] + source.x]"
    };
    let stencil_argument = match (multiple, stencil) {
        (true, true) => "texture2d_ms<uint, access::read> s [[texture(1)]],",
        (false, true) => "device const uchar* is [[buffer(1)]],",
        (_, false) => "",
    };
    let stencil_read = match (multiple, stencil) {
        (true, true) => "os[p.y * sizes[7] + p.x] = uchar(s.read(source, 0).r);",
        (false, true) => "os[p.y * sizes[7] + p.x] = is[source.y * sizes[5] + source.x];",
        (_, false) => "",
    };
    let stencil_output = if stencil {
        "device uchar* os [[buffer(3)]],"
    } else {
        ""
    };
    let source = format!(
        r"
#include <metal_stdlib>
using namespace metal;
kernel void transfer({depth_argument} {stencil_argument}
    device float* od [[buffer(2)]], {stencil_output}
    constant uint* sizes [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {{
    if (p.x >= sizes[2] || p.y >= sizes[3]) return;
    uint2 source = min(uint2(((2*p.x+1)*sizes[0])/(2*sizes[2]),
                            ((2*p.y+1)*sizes[1])/(2*sizes[3])), uint2(sizes[0]-1, sizes[1]-1));
    od[p.y * sizes[6] + p.x] = {depth_read};
    {stencil_read}
}}
"
    );
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version2_4);
    options.setMathMode(MTLMathMode::Safe);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
        .map_err(|e| log::error!(target: LOG_TARGET, "depth transfer library failed: {e}"))
        .ok()?;
    library.setLabel(Some(&NSString::from_str("mtld3d-depth-transfer")));
    let function = library.newFunctionWithName(&NSString::from_str("transfer"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| log::error!(target: LOG_TARGET, "depth transfer pipeline failed: {e}"))
        .ok()?;
    // SAFETY: the canonical retain is transferred to the device's encoder owner.
    Some(unsafe { MetalHandle::new(Retained::into_raw(pipeline) as u64) })
}

/// Release the device owner's canonical pipeline retain after GPU retirement.
pub fn destroy_pipeline(handle: u64) {
    // SAFETY: DestroyKind::ComputePipeline carries a live canonical compute retain.
    let handle = unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(handle) };
    // SAFETY: encoder teardown drains its queue before releasing pipeline ownership.
    unsafe { handle.release_retain() };
}

/// Encode one transfer on the retained-resource command buffer supplied by submission.
///
/// Native temporary objects can leave this function after encoding: the command
/// buffer retains every bound texture, view, buffer and pipeline until completion.
/// Its creation is centralized in `command::diagnostics::command_buffer`.
pub fn encode(cb: &ProtocolObject<dyn MTLCommandBuffer>, command: &BlitCommand) -> bool {
    // SAFETY: TransferDepth encodes source and destination canonical texture handles.
    let source_handle = unsafe { MetalHandle::<MTLTextureKind>::new(command.src_handle) };
    // SAFETY: as above; the destination handle is retained by the frame's resource owner.
    let destination_handle = unsafe { MetalHandle::<MTLTextureKind>::new(command.dst_handle) };
    let (Some(source), Some(destination)) = (
        source_handle.into_retained(),
        destination_handle.into_retained(),
    ) else {
        log::error!(target: LOG_TARGET, "depth transfer: missing source or destination");
        return false;
    };
    if !cb.retainedReferences() {
        log::error!(target: LOG_TARGET, "depth transfer: command buffer does not retain resources");
        return false;
    }
    let source_level = command.mip_level as usize;
    let destination_level = command.dst_mip_level as usize;
    if source_level >= source.mipmapLevelCount()
        || destination_level >= destination.mipmapLevelCount()
        || destination.sampleCount() != 1
        || !depth_format(source.pixelFormat())
        || !depth_format(destination.pixelFormat())
    {
        log::error!(target: LOG_TARGET, "depth transfer: invalid level, format or sample count");
        return false;
    }
    let width = (source.width() >> source_level).max(1);
    let height = (source.height() >> source_level).max(1);
    let out_width = (destination.width() >> destination_level).max(1);
    let out_height = (destination.height() >> destination_level).max(1);
    diagnostics::depth_transfer(
        cb,
        &diagnostics::DepthTransfer {
            source: &source,
            source_level,
            destination: &destination,
            destination_level,
            width: out_width,
            height: out_height,
        },
    );
    let stencil = source.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8
        && destination.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8;
    let device = source.device();
    let Some(output) = PlaneBuffers::new(&device, out_width, out_height, stencil) else {
        return false;
    };
    if source.sampleCount() == 1 && width == out_width && height == out_height {
        if !extract_planes(cb, &source, source_level, width, height, &output) {
            return false;
        }
    } else {
        // SAFETY: the typed transfer constructor carries the encoder-owned pipeline retain.
        let pipeline_handle =
            unsafe { MetalHandle::<MTLComputePipelineStateKind>::new(command.src_offset) };
        let Some(pipeline) = pipeline_handle.into_retained() else {
            log::error!(target: LOG_TARGET, "depth transfer: missing resample pipeline");
            return false;
        };
        let input = if source.sampleCount() == 1 {
            let Some(input) = PlaneBuffers::new(&device, width, height, stencil) else {
                return false;
            };
            if !extract_planes(cb, &source, source_level, width, height, &input) {
                return false;
            }
            Some(input)
        } else {
            None
        };
        let sampleable = if input.is_none() {
            let Some(texture) = copy_multisample_source(cb, &source, source_level, width, height)
            else {
                return false;
            };
            Some(texture)
        } else {
            None
        };
        let view = if stencil && let Some(sampleable) = sampleable.as_ref() {
            let Some(view) = sampleable.newTextureViewWithPixelFormat(MTLPixelFormat::X32_Stencil8)
            else {
                log::error!(target: LOG_TARGET, "depth transfer: stencil view allocation failed");
                return false;
            };
            view.setLabel(Some(&NSString::from_str("mtld3d-depth-transfer-stencil")));
            Some(view)
        } else {
            None
        };
        let Some(compute) = cb.computeCommandEncoder() else {
            log::error!(target: LOG_TARGET, "depth transfer: compute encoder failed");
            return false;
        };
        compute.setLabel(Some(&NSString::from_str("mtld3d-depth-transfer-resample")));
        compute.setComputePipelineState(&pipeline);
        if let Some(input) = input.as_ref() {
            // SAFETY: the single-sample kernel reads the full extracted depth plane at slot zero.
            unsafe {
                compute.setBuffer_offset_atIndex(Some(&input.depth), 0, 0);
            }
            if let Some(stencil) = input.stencil.as_ref() {
                // SAFETY: the stencil kernel reads the full extracted stencil plane at slot one.
                unsafe {
                    compute.setBuffer_offset_atIndex(Some(stencil), 0, 1);
                }
            }
        }
        if let Some(sampleable) = sampleable.as_ref() {
            // SAFETY: the multisample kernel reads this retained matching depth texture at slot zero.
            unsafe {
                compute.setTexture_atIndex(Some(sampleable), 0);
            }
        }
        if let Some(view) = view.as_ref() {
            // SAFETY: the stencil kernel's slot one is the matching uint stencil view.
            unsafe {
                compute.setTexture_atIndex(Some(view), 1);
            }
        }
        // SAFETY: the kernel bounds every write against the validated output dimensions.
        unsafe {
            compute.setBuffer_offset_atIndex(Some(&output.depth), 0, 2);
        }
        if let Some(stencil) = output.stencil.as_ref() {
            // SAFETY: the selected stencil kernel writes this full-sized byte plane.
            unsafe {
                compute.setBuffer_offset_atIndex(Some(stencil), 0, 3);
            }
        }
        let sizes = [
            width,
            height,
            out_width,
            out_height,
            input.as_ref().map_or(0, |p| p.depth_pitch / 4),
            input.as_ref().map_or(0, |p| p.stencil_pitch),
            output.depth_pitch / 4,
            output.stencil_pitch,
        ]
        .map(|n| u32::try_from(n).expect("Metal depth dimensions fit u32"));
        // SAFETY: the encoder copies this eight-word kernel argument before returning.
        unsafe {
            compute.setBytes_length_atIndex(
                core::ptr::NonNull::from(&sizes).cast(),
                core::mem::size_of_val(&sizes),
                4,
            );
        }
        let grid = MTLSize {
            width: out_width.div_ceil(8),
            height: out_height.div_ceil(8),
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        };
        diagnostics::depth_transfer_resample(
            cb,
            &diagnostics::DepthTransferResample {
                source: sampleable.as_deref(),
                source_stencil: view.as_deref(),
                input_depth: input.as_ref().map(|planes| &*planes.depth),
                input_stencil: input.as_ref().and_then(|planes| planes.stencil.as_deref()),
                output_depth: &output.depth,
                output_stencil: output.stencil.as_deref(),
                sizes,
                grid,
                threadgroup,
            },
        );
        compute.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
        compute.endEncoding();
    }
    let Some(blit) = cb.blitCommandEncoder() else {
        log::error!(target: LOG_TARGET, "depth transfer: destination blit encoder failed");
        return false;
    };
    blit.setLabel(Some(&NSString::from_str(
        "mtld3d-depth-transfer-destination-copy",
    )));
    let depth_option = if destination.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
        MTLBlitOption::DepthFromDepthStencil
    } else {
        MTLBlitOption::empty()
    };
    let region = PlaneDestination {
        texture: &destination,
        level: destination_level,
        width: out_width,
        height: out_height,
    };
    insert_plane(
        &blit,
        &output.depth,
        output.depth_pitch,
        depth_option,
        &region,
    );
    if let Some(stencil) = output.stencil.as_ref() {
        insert_plane(
            &blit,
            stencil,
            output.stencil_pitch,
            MTLBlitOption::StencilFromDepthStencil,
            &region,
        );
    }
    blit.endEncoding();
    true
}

struct PlaneBuffers {
    depth: Retained<ProtocolObject<dyn MTLBuffer>>,
    stencil: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    depth_pitch: usize,
    stencil_pitch: usize,
}

impl PlaneBuffers {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        width: usize,
        height: usize,
        stencil: bool,
    ) -> Option<Self> {
        let depth_pitch = (width * 4).next_multiple_of(256);
        let stencil_pitch = width.next_multiple_of(256);
        let depth = device.newBufferWithLength_options(
            depth_pitch * height,
            MTLResourceOptions::StorageModePrivate,
        )?;
        depth.setLabel(Some(&NSString::from_str(
            "mtld3d-depth-transfer-depth-plane",
        )));
        let stencil = if stencil {
            let buffer = device.newBufferWithLength_options(
                stencil_pitch * height,
                MTLResourceOptions::StorageModePrivate,
            )?;
            buffer.setLabel(Some(&NSString::from_str(
                "mtld3d-depth-transfer-stencil-plane",
            )));
            Some(buffer)
        } else {
            None
        };
        Some(Self {
            depth,
            stencil,
            depth_pitch,
            stencil_pitch,
        })
    }
}

fn extract_planes(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    width: usize,
    height: usize,
    output: &PlaneBuffers,
) -> bool {
    let Some(blit) = cb.blitCommandEncoder() else {
        return false;
    };
    blit.setLabel(Some(&NSString::from_str("mtld3d-depth-transfer-extract")));
    let depth_option = if source.pixelFormat() == MTLPixelFormat::Depth32Float_Stencil8 {
        MTLBlitOption::DepthFromDepthStencil
    } else {
        MTLBlitOption::empty()
    };
    // SAFETY: full validated single-sample mip, float plane stride and private allocation.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
            source, 0, level, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width, height, depth: 1 },
            &output.depth, 0, output.depth_pitch, 0, depth_option);
    }
    if let Some(stencil) = output.stencil.as_ref() {
        // SAFETY: both endpoints have stencil; the buffer holds the full byte plane.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                source, 0, level, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width, height, depth: 1 },
                stencil, 0, output.stencil_pitch, 0, MTLBlitOption::StencilFromDepthStencil);
        }
    }
    blit.endEncoding();
    true
}

fn copy_multisample_source(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    source: &ProtocolObject<dyn MTLTexture>,
    level: usize,
    width: usize,
    height: usize,
) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    // SAFETY: source mip dimensions and format are already validated.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            source.pixelFormat(),
            width,
            height,
            false,
        )
    };
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setTextureType(MTLTextureType::Type2DMultisample);
    // SAFETY: the live source's sample count is supported on the same device.
    unsafe {
        desc.setSampleCount(source.sampleCount());
    }
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
    let sampleable = source.device().newTextureWithDescriptor(&desc)?;
    sampleable.setLabel(Some(&NSString::from_str("mtld3d-depth-transfer-source")));
    let blit = cb.blitCommandEncoder()?;
    blit.setLabel(Some(&NSString::from_str(
        "mtld3d-depth-transfer-source-copy",
    )));
    // SAFETY: matching format, sample count and validated same-sized source mip.
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            source, 0, level, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width, height, depth: 1 },
            &sampleable, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
    }
    blit.endEncoding();
    Some(sampleable)
}

struct PlaneDestination<'a> {
    texture: &'a ProtocolObject<dyn MTLTexture>,
    level: usize,
    width: usize,
    height: usize,
}

fn insert_plane(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    pitch: usize,
    options: MTLBlitOption,
    destination: &PlaneDestination<'_>,
) {
    // SAFETY: caller allocated the plane for these aligned rows and validated the destination mip.
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
        buffer, 0, pitch, 0, MTLSize { width: destination.width, height: destination.height, depth: 1 },
        destination.texture, 0, destination.level, MTLOrigin { x: 0, y: 0, z: 0 }, options,
    );
    }
}

const fn depth_format(format: MTLPixelFormat) -> bool {
    matches!(
        format,
        MTLPixelFormat::Depth32Float | MTLPixelFormat::Depth32Float_Stencil8
    )
}
