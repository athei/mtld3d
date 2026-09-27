use strum::{EnumCount, VariantArray};

pub mod blit_geometry;
pub mod bounded_cache;
pub mod clock_calibration;
mod commands;
pub mod crumb;
pub mod encoder_protocol;
pub mod encoder_runtime;
pub mod encoder_wire;
pub mod fatal;
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
pub mod query_mailbox;
pub mod record_handle;
pub mod shader_create;
pub mod texture_views;
pub mod trig;
pub mod tsc;
pub mod upload_feedback;

pub use commands::{
    BlitCommand, BlitCommandType, Command, CommandType, CopyBufferToBufferInfo,
    CopyBufferToTextureInfo, CopyTextureSubRectInfo, NullTextureKind,
};
pub use ffi_boundary::{InPtr, InPtrMut, OutPtr, ValueIn, VtableThis, slice_from_caller};
pub use log_filter::{init_logger, init_logger_to, init_logger_to_filter};
pub use mtl_handle::MetalHandle;
pub use params::{
    AttachMetalLayerParams, BlitTextureToBufferParams, BufferCreateDesc,
    CompileShaderLibraryParams, CreateBackbufferParams, CreateBuffersBatchParams,
    CreateColorTargetParams, CreateCommandQueueParams, CreateDepthStencilStateParams,
    CreateDepthTextureParams, CreateDepthTransferPipelineParams, CreateRenderPipelineParams,
    CreateSamplerStateParams, CreateTextureSliceViewParams, CreateTexturesBatchParams,
    DestroyCommandQueueParams, DestroyResourcesBulkParams, DetachMetalLayerParams,
    EnsureBlitPipelineParams, EnsureClearQuadPipelineParams, ExtraColorAttachmentParams,
    ExtraColorDesc, GetDeviceInfoParams, GetTaskFaultsParams, InitLoggerParams, OpenLogParams,
    PassDescriptor, SetCursorOverlayParams, SetDisplaySyncEnabledParams, SetGammaRampParams,
    SetPresentWaitPolicyParams, StartGpuCaptureParams, StencilFaceParams, StopGpuCaptureParams,
    SubmitFrameParams, TextureCreateDesc, VertexAttrDesc, VertexBufferLayoutDesc,
    WaitForGpuRetireParams, WaitForPresentIdleParams, WriteLogParams,
};
pub use record_handle::DeviceRecordHandle;

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
    SetCursorOverlay,
    DetachMetalLayer,
    SetPresentWaitPolicy,
    WaitForPresentIdle,
    CreateDepthTransferPipeline,
    SetGammaRamp,
    CreateEncoder,
    DestroyEncoder,
    SubmitEncoderFrame,
    EncoderControl,
    CreateShaderProgram,
    CancelShaderProgram,
}

pub trait Thunk {
    const CODE: u32;

    /// Submission mode for the opt-in dispatch trace, absent for other requests.
    ///
    /// An invalid raw submission mode also returns `None` instead of constructing an enum.
    #[cfg(perf_tracking)]
    fn perf_submit_mode(&self) -> Option<encoder_protocol::EncoderSubmitMode> {
        None
    }
}
