use strum::{EnumCount, VariantArray};

pub mod blit_geometry;
mod commands;
pub mod crumb;
pub mod ffi_boundary;
pub mod ftol;
pub mod identity;
mod log_filter;
mod log_helpers;
pub mod log_paths;
pub mod mtl;
pub mod mtl_handle;
mod params;
pub mod perf;
pub mod trig;
pub mod tsc;

pub use commands::{
    BlitCommand, BlitCommandType, Command, CommandType, CopyBufferToBufferInfo,
    CopyBufferToTextureInfo, CopyTextureSubRectInfo, NullTextureKind,
};
pub use ffi_boundary::{InPtr, InPtrMut, OutPtr, ValueIn, VtableThis};
pub use log_filter::{init_logger, init_logger_to};
pub use mtl_handle::MetalHandle;
pub use params::{
    AttachMetalLayerParams, BlitTextureToBufferParams, BufferCreateDesc,
    CompileShaderLibraryParams, CreateBackbufferParams, CreateBuffersBatchParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthStencilStateParams,
    CreateDepthTextureParams, CreateRenderPipelineParams, CreateSamplerStateParams,
    CreateTextureSliceViewParams, CreateTexturesBatchParams, DestroyCommandQueueParams,
    DestroyResourcesBulkParams, EnsureBlitPipelineParams, EnsureClearQuadPipelineParams,
    ExtraColorAttachmentParams, ExtraColorDesc, GetDeviceInfoParams, GetTaskFaultsParams,
    InitLoggerParams, OpenLogParams, PassDescriptor, SetDisplaySyncEnabledParams,
    StartGpuCaptureParams, StencilFaceParams, StopGpuCaptureParams, SubmitFrameParams,
    TextureCreateDesc, VertexAttrDesc, VertexBufferLayoutDesc, WaitForGpuRetireParams,
    WriteLogParams,
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
    WaitForGpuRetire,
    StartGpuCapture,
    StopGpuCapture,
    EnsureClearQuadPipeline,
    EnsureBlitPipeline,
    CreateTextureSliceView,
    GetTaskFaults,
    WriteLog,
    OpenLog,
}

pub trait Thunk {
    const CODE: u32;
}
