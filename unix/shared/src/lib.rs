use strum::{EnumCount, VariantArray};

mod commands;
pub mod crumb;
pub mod ffi_boundary;
pub mod ftol;
mod log_filter;
mod log_helpers;
pub mod mtl;
pub mod mtl_handle;
mod params;
pub mod perf;
pub mod trig;
pub mod tsc;

pub use commands::{
    BlitCommand, BlitCommandType, Command, CommandType, CopyBufferToBufferInfo,
    CopyBufferToTextureInfo, CopyTextureSubRectInfo,
};
pub use ffi_boundary::{InPtr, InPtrMut, OutPtr, ValueIn, VtableThis};
pub use log_filter::init_logger;
pub use mtl_handle::MetalHandle;
pub use params::{
    AttachMetalLayerParams, BlitTextureToBufferParams, BufferCreateDesc,
    CompileShaderLibraryParams, CreateBackbufferParams, CreateBuffersBatchParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthStencilStateParams,
    CreateDepthTextureParams, CreateRenderPipelineParams, CreateSamplerStateParams,
    CreateTexturesBatchParams, DestroyCommandQueueParams, DestroyResourcesBulkParams,
    EnsureBlitPipelineParams, EnsureClearQuadPipelineParams, GetDeviceInfoParams,
    GetPrimaryDisplayModeParams, InitLoggerParams, PassDescriptor, SetDisplaySyncEnabledParams,
    SetLayerDrawableSizeParams, StartGpuCaptureParams, StopGpuCaptureParams, SubmitFrameParams,
    TextureCreateDesc, VertexAttrDesc, WaitForGpuRetireParams,
};

#[repr(u32)]
#[derive(Clone, Copy, EnumCount, VariantArray)]
pub enum Thunks {
    InitLogger,
    GetDeviceInfo,
    CreateCommandQueue,
    AttachMetalLayer,
    DestroyCommandQueue,
    CreateBackbuffer,
    CreateRenderPipeline,
    SubmitFrame,
    CreateDepthTexture,
    CreateColorTarget,
    CreateDepthStencilState,
    CreateTexturesBatch,
    CreateSamplerState,
    CompileShaderLibrary,
    CreateBuffersBatch,
    BlitTextureToBuffer,
    SetDisplaySyncEnabled,
    DestroyResourcesBulk,
    SetLayerDrawableSize,
    WaitForGpuRetire,
    StartGpuCapture,
    StopGpuCapture,
    GetPrimaryDisplayMode,
    EnsureClearQuadPipeline,
    EnsureBlitPipeline,
}

pub trait Thunk {
    const CODE: u32;
}
